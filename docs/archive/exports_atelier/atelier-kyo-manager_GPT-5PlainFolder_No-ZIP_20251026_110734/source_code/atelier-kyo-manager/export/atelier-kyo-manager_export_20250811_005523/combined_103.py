
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\dense.py ===
"""

Module for the ddm_* routines for operating on a matrix in list of lists
matrix representation.

These routines are used internally by the DDM class which also provides a
friendlier interface for them. The idea here is to implement core matrix
routines in a way that can be applied to any simple list representation
without the need to use any particular matrix class. For example we can
compute the RREF of a matrix like:

    >>> from sympy.polys.matrices.dense import ddm_irref
    >>> M = [[1, 2, 3], [4, 5, 6]]
    >>> pivots = ddm_irref(M)
    >>> M
    [[1.0, 0.0, -1.0], [0, 1.0, 2.0]]

These are lower-level routines that work mostly in place.The routines at this
level should not need to know what the domain of the elements is but should
ideally document what operations they will use and what functions they need to
be provided with.

The next-level up is the DDM class which uses these routines but wraps them up
with an interface that handles copying etc and keeps track of the Domain of
the elements of the matrix:

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.matrices.ddm import DDM
    >>> M = DDM([[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]], (2, 3), QQ)
    >>> M
    [[1, 2, 3], [4, 5, 6]]
    >>> Mrref, pivots = M.rref()
    >>> Mrref
    [[1, 0, -1], [0, 1, 2]]

"""
from __future__ import annotations
from operator import mul
from .exceptions import (
    DMShapeError,
    DMDomainError,
    DMNonInvertibleMatrixError,
    DMNonSquareMatrixError,
)
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement


#: Type variable for the elements of the matrix
T = TypeVar('T')

#: Type variable for the elements of the matrix that are in a ring
R = TypeVar('R', bound=RingElement)


def ddm_transpose(matrix: Sequence[Sequence[T]]) -> list[list[T]]:
    """matrix transpose"""
    return list(map(list, zip(*matrix)))


def ddm_iadd(a: list[list[R]], b: Sequence[Sequence[R]]) -> None:
    """a += b"""
    for ai, bi in zip(a, b):
        for j, bij in enumerate(bi):
            ai[j] += bij


def ddm_isub(a: list[list[R]], b: Sequence[Sequence[R]]) -> None:
    """a -= b"""
    for ai, bi in zip(a, b):
        for j, bij in enumerate(bi):
            ai[j] -= bij


def ddm_ineg(a: list[list[R]]) -> None:
    """a <-- -a"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = -aij


def ddm_imul(a: list[list[R]], b: R) -> None:
    """a <-- a*b"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = aij * b


def ddm_irmul(a: list[list[R]], b: R) -> None:
    """a <-- b*a"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = b * aij


def ddm_imatmul(
    a: list[list[R]], b: Sequence[Sequence[R]], c: Sequence[Sequence[R]]
) -> None:
    """a += b @ c"""
    cT = list(zip(*c))

    for bi, ai in zip(b, a):
        for j, cTj in enumerate(cT):
            ai[j] = sum(map(mul, bi, cTj), ai[j])


def ddm_irref(a, _partial_pivot=False):
    """In-place reduced row echelon form of a matrix.

    Compute the reduced row echelon form of $a$. Modifies $a$ in place and
    returns a list of the pivot columns.

    Uses naive Gauss-Jordan elimination in the ground domain which must be a
    field.

    This routine is only really suitable for use with simple field domains like
    :ref:`GF(p)`, :ref:`QQ` and :ref:`QQ(a)` although even for :ref:`QQ` with
    larger matrices it is possibly more efficient to use fraction free
    approaches.

    This method is not suitable for use with rational function fields
    (:ref:`K(x)`) because the elements will blowup leading to costly gcd
    operations. In this case clearing denominators and using fraction free
    approaches is likely to be more efficient.

    For inexact numeric domains like :ref:`RR` and :ref:`CC` pass
    ``_partial_pivot=True`` to use partial pivoting to control rounding errors.

    Examples
    ========

    >>> from sympy.polys.matrices.dense import ddm_irref
    >>> from sympy import QQ
    >>> M = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]]
    >>> pivots = ddm_irref(M)
    >>> M
    [[1, 0, -1], [0, 1, 2]]
    >>> pivots
    [0, 1]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        Higher level interface to this routine.
    ddm_irref_den
        The fraction free version of this routine.
    sdm_irref
        A sparse version of this routine.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form
    """
    # We compute aij**-1 below and then use multiplication instead of division
    # in the innermost loop. The domain here is a field so either operation is
    # defined. There are significant performance differences for some domains
    # though. In the case of e.g. QQ or QQ(x) inversion is free but
    # multiplication and division have the same cost so it makes no difference.
    # In cases like GF(p), QQ<sqrt(2)>, RR or CC though multiplication is
    # faster than division so reusing a precomputed inverse for many
    # multiplications can be a lot faster. The biggest win is QQ<a> when
    # deg(minpoly(a)) is large.
    #
    # With domains like QQ(x) this can perform badly for other reasons.
    # Typically the initial matrix has simple denominators and the
    # fraction-free approach with exquo (ddm_irref_den) will preserve that
    # property throughout. The method here causes denominator blowup leading to
    # expensive gcd reductions in the intermediate expressions. With many
    # generators like QQ(x,y,z,...) this is extremely bad.
    #
    # TODO: Use a nontrivial pivoting strategy to control intermediate
    # expression growth. Rearranging rows and/or columns could defer the most
    # complicated elements until the end. If the first pivot is a
    # complicated/large element then the first round of reduction will
    # immediately introduce expression blowup across the whole matrix.

    # a is (m x n)
    m = len(a)
    if not m:
        return []
    n = len(a[0])

    i = 0
    pivots = []

    for j in range(n):
        # Proper pivoting should be used for all domains for performance
        # reasons but it is only strictly needed for RR and CC (and possibly
        # other domains like RR(x)). This path is used by DDM.rref() if the
        # domain is RR or CC. It uses partial (row) pivoting based on the
        # absolute value of the pivot candidates.
        if _partial_pivot:
            ip = max(range(i, m), key=lambda ip: abs(a[ip][j]))
            a[i], a[ip] = a[ip], a[i]

        # pivot
        aij = a[i][j]

        # zero-pivot
        if not aij:
            for ip in range(i+1, m):
                aij = a[ip][j]
                # row-swap
                if aij:
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # next column
                continue

        # normalise row
        ai = a[i]
        aijinv = aij**-1
        for l in range(j, n):
            ai[l] *= aijinv # ai[j] = one

        # eliminate above and below to the right
        for k, ak in enumerate(a):
            if k == i or not ak[j]:
                continue
            akj = ak[j]
            ak[j] -= akj # ak[j] = zero
            for l in range(j+1, n):
                ak[l] -= akj * ai[l]

        # next row
        pivots.append(j)
        i += 1

        # no more rows?
        if i >= m:
            break

    return pivots


def ddm_irref_den(a, K):
    """a  <--  rref(a); return (den, pivots)

    Compute the fraction-free reduced row echelon form (RREF) of $a$. Modifies
    $a$ in place and returns a tuple containing the denominator of the RREF and
    a list of the pivot columns.

    Explanation
    ===========

    The algorithm used is the fraction-free version of Gauss-Jordan elimination
    described as FFGJ in [1]_. Here it is modified to handle zero or missing
    pivots and to avoid redundant arithmetic.

    The domain $K$ must support exact division (``K.exquo``) but does not need
    to be a field. This method is suitable for most exact rings and fields like
    :ref:`ZZ`, :ref:`QQ` and :ref:`QQ(a)`. In the case of :ref:`QQ` or
    :ref:`K(x)` it might be more efficient to clear denominators and use
    :ref:`ZZ` or :ref:`K[x]` instead.

    For inexact domains like :ref:`RR` and :ref:`CC` use ``ddm_irref`` instead.

    Examples
    ========

    >>> from sympy.polys.matrices.dense import ddm_irref_den
    >>> from sympy import ZZ, Matrix
    >>> M = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)]]
    >>> den, pivots = ddm_irref_den(M, ZZ)
    >>> M
    [[-3, 0, 3], [0, -3, -6]]
    >>> den
    -3
    >>> pivots
    [0, 1]
    >>> Matrix(M).rref()[0]
    Matrix([
    [1, 0, -1],
    [0, 1,  2]])

    See Also
    ========

    ddm_irref
        A version of this routine that uses field division.
    sdm_irref
        A sparse version of :func:`ddm_irref`.
    sdm_rref_den
        A sparse version of :func:`ddm_irref_den`.
    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        Higher level interface.

    References
    ==========

    .. [1] Fraction-free algorithms for linear and polynomial equations.
        George C. Nakos , Peter R. Turner , Robert M. Williams.
        https://dl.acm.org/doi/10.1145/271130.271133
    """
    #
    # A simpler presentation of this algorithm is given in [1]:
    #
    # Given an n x n matrix A and n x 1 matrix b:
    #
    #   for i in range(n):
    #       if i != 0:
    #           d = a[i-1][i-1]
    #       for j in range(n):
    #           if j == i:
    #               continue
    #           b[j] = a[i][i]*b[j] - a[j][i]*b[i]
    #           for k in range(n):
    #               a[j][k] = a[i][i]*a[j][k] - a[j][i]*a[i][k]
    #               if i != 0:
    #                   a[j][k] /= d
    #
    # Our version here is a bit more complicated because:
    #
    #  1. We use row-swaps to avoid zero pivots.
    #  2. We allow for some columns to be missing pivots.
    #  3. We avoid a lot of redundant arithmetic.
    #
    # TODO: Use a non-trivial pivoting strategy. Even just row swapping makes a
    # big difference to performance if e.g. the upper-left entry of the matrix
    # is a huge polynomial.

    # a is (m x n)
    m = len(a)
    if not m:
        return K.one, []
    n = len(a[0])

    d = None
    pivots = []
    no_pivots = []

    # i, j will be the row and column indices of the current pivot
    i = 0
    for j in range(n):
        # next pivot?
        aij = a[i][j]

        # swap rows if zero
        if not aij:
            for ip in range(i+1, m):
                aij = a[ip][j]
                # row-swap
                if aij:
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # go to next column
                no_pivots.append(j)
                continue

        # Now aij is the pivot and i,j are the row and column. We need to clear
        # the column above and below but we also need to keep track of the
        # denominator of the RREF which means also multiplying everything above
        # and to the left by the current pivot aij and dividing by d (which we
        # multiplied everything by in the previous iteration so this is an
        # exact division).
        #
        # First handle the upper left corner which is usually already diagonal
        # with all diagonal entries equal to the current denominator but there
        # can be other non-zero entries in any column that has no pivot.

        # Update previous pivots in the matrix
        if pivots:
            pivot_val = aij * a[0][pivots[0]]
            # Divide out the common factor
            if d is not None:
                pivot_val = K.exquo(pivot_val, d)

            # Could defer this until the end but it is pretty cheap and
            # helps when debugging.
            for ip, jp in enumerate(pivots):
                a[ip][jp] = pivot_val

        # Update columns without pivots
        for jnp in no_pivots:
            for ip in range(i):
                aijp = a[ip][jnp]
                if aijp:
                    aijp *= aij
                    if d is not None:
                        aijp = K.exquo(aijp, d)
                    a[ip][jnp] = aijp

        # Eliminate above, below and to the right as in ordinary division free
        # Gauss-Jordan elmination except also dividing out d from every entry.

        for jp, aj in enumerate(a):

            # Skip the current row
            if jp == i:
                continue

            # Eliminate to the right in all rows
            for kp in range(j+1, n):
                ajk = aij * aj[kp] - aj[j] * a[i][kp]
                if d is not None:
                    ajk = K.exquo(ajk, d)
                aj[kp] = ajk

            # Set to zero above and below the pivot
            aj[j] = K.zero

        # next row
        pivots.append(j)
        i += 1

        # no more rows left?
        if i >= m:
            break

        if not K.is_one(aij):
            d = aij
        else:
            d = None

    if not pivots:
        denom = K.one
    else:
        denom = a[0][pivots[0]]

    return denom, pivots


def ddm_idet(a, K):
    """a  <--  echelon(a); return det

    Explanation
    ===========

    Compute the determinant of $a$ using the Bareiss fraction-free algorithm.
    The matrix $a$ is modified in place. Its diagonal elements are the
    determinants of the leading principal minors. The determinant of $a$ is
    returned.

    The domain $K$ must support exact division (``K.exquo``). This method is
    suitable for most exact rings and fields like :ref:`ZZ`, :ref:`QQ` and
    :ref:`QQ(a)` but not for inexact domains like :ref:`RR` and :ref:`CC`.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices.ddm import ddm_idet
    >>> a = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]]
    >>> a
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> ddm_idet(a, ZZ)
    0
    >>> a
    [[1, 2, 3], [4, -3, -6], [7, -6, 0]]
    >>> [a[i][i] for i in range(len(a))]
    [1, -3, 0]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.det

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bareiss_algorithm
    .. [2] https://www.math.usm.edu/perry/Research/Thesis_DRL.pdf
    """
    # Bareiss algorithm
    # https://www.math.usm.edu/perry/Research/Thesis_DRL.pdf

    # a is (m x n)
    m = len(a)
    if not m:
        return K.one
    n = len(a[0])

    exquo = K.exquo
    # uf keeps track of the sign change from row swaps
    uf = K.one

    for k in range(n-1):
        if not a[k][k]:
            for i in range(k+1, n):
                if a[i][k]:
                    a[k], a[i] = a[i], a[k]
                    uf = -uf
                    break
            else:
                return K.zero

        akkm1 = a[k-1][k-1] if k else K.one

        for i in range(k+1, n):
            for j in range(k+1, n):
                a[i][j] = exquo(a[i][j]*a[k][k] - a[i][k]*a[k][j], akkm1)

    return uf * a[-1][-1]


def ddm_iinv(ainv, a, K):
    """ainv  <--  inv(a)

    Compute the inverse of a matrix $a$ over a field $K$ using Gauss-Jordan
    elimination. The result is stored in $ainv$.

    Uses division in the ground domain which should be an exact field.

    Examples
    ========

    >>> from sympy.polys.matrices.ddm import ddm_iinv, ddm_imatmul
    >>> from sympy import QQ
    >>> a = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    >>> ainv = [[None, None], [None, None]]
    >>> ddm_iinv(ainv, a, QQ)
    >>> ainv
    [[-2, 1], [3/2, -1/2]]
    >>> result = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    >>> ddm_imatmul(result, a, ainv)
    >>> result
    [[1, 0], [0, 1]]

    See Also
    ========

    ddm_irref: the underlying routine.
    """
    if not K.is_Field:
        raise DMDomainError('Not a field')

    # a is (m x n)
    m = len(a)
    if not m:
        return
    n = len(a[0])
    if m != n:
        raise DMNonSquareMatrixError

    eye = [[K.one if i==j else K.zero for j in range(n)] for i in range(n)]
    Aaug = [row + eyerow for row, eyerow in zip(a, eye)]
    pivots = ddm_irref(Aaug)
    if pivots != list(range(n)):
        raise DMNonInvertibleMatrixError('Matrix det == 0; not invertible.')
    ainv[:] = [row[n:] for row in Aaug]


def ddm_ilu_split(L, U, K):
    """L, U  <--  LU(U)

    Compute the LU decomposition of a matrix $L$ in place and store the lower
    and upper triangular matrices in $L$ and $U$, respectively. Returns a list
    of row swaps that were performed.

    Uses division in the ground domain which should be an exact field.

    Examples
    ========

    >>> from sympy.polys.matrices.ddm import ddm_ilu_split
    >>> from sympy import QQ
    >>> L = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    >>> U = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    >>> swaps = ddm_ilu_split(L, U, QQ)
    >>> swaps
    []
    >>> L
    [[0, 0], [3, 0]]
    >>> U
    [[1, 2], [0, -2]]

    See Also
    ========

    ddm_ilu
    ddm_ilu_solve
    """
    m = len(U)
    if not m:
        return []
    n = len(U[0])

    swaps = ddm_ilu(U)

    zeros = [K.zero] * min(m, n)
    for i in range(1, m):
        j = min(i, n)
        L[i][:j] = U[i][:j]
        U[i][:j] = zeros[:j]

    return swaps


def ddm_ilu(a):
    """a  <--  LU(a)

    Computes the LU decomposition of a matrix in place. Returns a list of
    row swaps that were performed.

    Uses division in the ground domain which should be an exact field.

    This is only suitable for domains like :ref:`GF(p)`, :ref:`QQ`, :ref:`QQ_I`
    and :ref:`QQ(a)`. With a rational function field like :ref:`K(x)` it is
    better to clear denominators and use division-free algorithms. Pivoting is
    used to avoid exact zeros but not for floating point accuracy so :ref:`RR`
    and :ref:`CC` are not suitable (use :func:`ddm_irref` instead).

    Examples
    ========

    >>> from sympy.polys.matrices.dense import ddm_ilu
    >>> from sympy import QQ
    >>> a = [[QQ(1, 2), QQ(1, 3)], [QQ(1, 4), QQ(1, 5)]]
    >>> swaps = ddm_ilu(a)
    >>> swaps
    []
    >>> a
    [[1/2, 1/3], [1/2, 1/30]]

    The same example using ``Matrix``:

    >>> from sympy import Matrix, S
    >>> M = Matrix([[S(1)/2, S(1)/3], [S(1)/4, S(1)/5]])
    >>> L, U, swaps = M.LUdecomposition()
    >>> L
    Matrix([
    [  1, 0],
    [1/2, 1]])
    >>> U
    Matrix([
    [1/2,  1/3],
    [  0, 1/30]])
    >>> swaps
    []

    See Also
    ========

    ddm_irref
    ddm_ilu_solve
    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    """
    m = len(a)
    if not m:
        return []
    n = len(a[0])

    swaps = []

    for i in range(min(m, n)):
        if not a[i][i]:
            for ip in range(i+1, m):
                if a[ip][i]:
                    swaps.append((i, ip))
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # M = Matrix([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 2]])
                continue
        for j in range(i+1, m):
            l_ji = a[j][i] / a[i][i]
            a[j][i] = l_ji
            for k in range(i+1, n):
                a[j][k] -= l_ji * a[i][k]

    return swaps


def ddm_ilu_solve(x, L, U, swaps, b):
    """x  <--  solve(L*U*x = swaps(b))

    Solve a linear system, $A*x = b$, given an LU factorization of $A$.

    Uses division in the ground domain which must be a field.

    Modifies $x$ in place.

    Examples
    ========

    Compute the LU decomposition of $A$ (in place):

    >>> from sympy import QQ
    >>> from sympy.polys.matrices.dense import ddm_ilu, ddm_ilu_solve
    >>> A = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    >>> swaps = ddm_ilu(A)
    >>> A
    [[1, 2], [3, -2]]
    >>> L = U = A

    Solve the linear system:

    >>> b = [[QQ(5)], [QQ(6)]]
    >>> x = [[None], [None]]
    >>> ddm_ilu_solve(x, L, U, swaps, b)
    >>> x
    [[-4], [9/2]]

    See Also
    ========

    ddm_ilu
        Compute the LU decomposition of a matrix in place.
    ddm_ilu_split
        Compute the LU decomposition of a matrix and separate $L$ and $U$.
    sympy.polys.matrices.domainmatrix.DomainMatrix.lu_solve
        Higher level interface to this function.
    """
    m = len(U)
    if not m:
        return
    n = len(U[0])

    m2 = len(b)
    if not m2:
        raise DMShapeError("Shape mismtch")
    o = len(b[0])

    if m != m2:
        raise DMShapeError("Shape mismtch")
    if m < n:
        raise NotImplementedError("Underdetermined")

    if swaps:
        b = [row[:] for row in b]
        for i1, i2 in swaps:
            b[i1], b[i2] = b[i2], b[i1]

    # solve Ly = b
    y = [[None] * o for _ in range(m)]
    for k in range(o):
        for i in range(m):
            rhs = b[i][k]
            for j in range(i):
                rhs -= L[i][j] * y[j][k]
            y[i][k] = rhs

    if m > n:
        for i in range(n, m):
            for j in range(o):
                if y[i][j]:
                    raise DMNonInvertibleMatrixError

    # Solve Ux = y
    for k in range(o):
        for i in reversed(range(n)):
            if not U[i][i]:
                raise DMNonInvertibleMatrixError
            rhs = y[i][k]
            for j in range(i+1, n):
                rhs -= U[i][j] * x[j][k]
            x[i][k] = rhs / U[i][i]


def ddm_berk(M, K):
    """
    Berkowitz algorithm for computing the characteristic polynomial.

    Explanation
    ===========

    The Berkowitz algorithm is a division-free algorithm for computing the
    characteristic polynomial of a matrix over any commutative ring using only
    arithmetic in the coefficient ring.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices.dense import ddm_berk
    >>> from sympy.polys.domains import ZZ
    >>> M = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    >>> ddm_berk(M, ZZ)
    [[1], [-5], [-2]]
    >>> Matrix(M).charpoly()
    PurePoly(lambda**2 - 5*lambda - 2, lambda, domain='ZZ')

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
        The high-level interface to this function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Samuelson%E2%80%93Berkowitz_algorithm
    """
    m = len(M)
    if not m:
        return [[K.one]]
    n = len(M[0])

    if m != n:
        raise DMShapeError("Not square")

    if n == 1:
        return [[K.one], [-M[0][0]]]

    a = M[0][0]
    R = [M[0][1:]]
    C = [[row[0]] for row in M[1:]]
    A = [row[1:] for row in M[1:]]

    q = ddm_berk(A, K)

    T = [[K.zero] * n for _ in range(n+1)]
    for i in range(n):
        T[i][i] = K.one
        T[i+1][i] = -a
    for i in range(2, n+1):
        if i == 2:
            AnC = C
        else:
            C = AnC
            AnC = [[K.zero] for row in C]
            ddm_imatmul(AnC, A, C)
        RAnC = [[K.zero]]
        ddm_imatmul(RAnC, R, AnC)
        for j in range(0, n+1-i):
            T[i+j][j] = -RAnC[0][0]

    qout = [[K.zero] for _ in range(n+1)]
    ddm_imatmul(qout, T, q)
    return qout

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\resolution\resolvelib\factory.py ===
import contextlib
import functools
import logging
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    cast,
)

from pip._vendor.packaging.requirements import InvalidRequirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import InvalidVersion, Version
from pip._vendor.resolvelib import ResolutionImpossible

from pip._internal.cache import CacheEntry, WheelCache
from pip._internal.exceptions import (
    DistributionNotFound,
    InstallationError,
    InvalidInstalledPackage,
    MetadataInconsistent,
    MetadataInvalid,
    UnsupportedPythonVersion,
    UnsupportedWheel,
)
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_default_environment
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import (
    install_req_drop_extras,
    install_req_from_link_and_ireq,
)
from pip._internal.req.req_install import (
    InstallRequirement,
    check_invalid_constraint_type,
)
from pip._internal.resolution.base import InstallRequirementProvider
from pip._internal.utils.compatibility_tags import get_supported
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.virtualenv import running_under_virtualenv

from .base import Candidate, Constraint, Requirement
from .candidates import (
    AlreadyInstalledCandidate,
    BaseCandidate,
    EditableCandidate,
    ExtrasCandidate,
    LinkCandidate,
    RequiresPythonCandidate,
    as_base_candidate,
)
from .found_candidates import FoundCandidates, IndexCandidateInfo
from .requirements import (
    ExplicitRequirement,
    RequiresPythonRequirement,
    SpecifierRequirement,
    SpecifierWithoutExtrasRequirement,
    UnsatisfiableRequirement,
)

if TYPE_CHECKING:

    class ConflictCause(Protocol):
        requirement: RequiresPythonRequirement
        parent: Candidate


logger = logging.getLogger(__name__)

C = TypeVar("C")
Cache = Dict[Link, C]


class CollectedRootRequirements(NamedTuple):
    requirements: List[Requirement]
    constraints: Dict[str, Constraint]
    user_requested: Dict[str, int]


class Factory:
    def __init__(
        self,
        finder: PackageFinder,
        preparer: RequirementPreparer,
        make_install_req: InstallRequirementProvider,
        wheel_cache: Optional[WheelCache],
        use_user_site: bool,
        force_reinstall: bool,
        ignore_installed: bool,
        ignore_requires_python: bool,
        py_version_info: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self._finder = finder
        self.preparer = preparer
        self._wheel_cache = wheel_cache
        self._python_candidate = RequiresPythonCandidate(py_version_info)
        self._make_install_req_from_spec = make_install_req
        self._use_user_site = use_user_site
        self._force_reinstall = force_reinstall
        self._ignore_requires_python = ignore_requires_python

        self._build_failures: Cache[InstallationError] = {}
        self._link_candidate_cache: Cache[LinkCandidate] = {}
        self._editable_candidate_cache: Cache[EditableCandidate] = {}
        self._installed_candidate_cache: Dict[str, AlreadyInstalledCandidate] = {}
        self._extras_candidate_cache: Dict[
            Tuple[int, FrozenSet[NormalizedName]], ExtrasCandidate
        ] = {}
        self._supported_tags_cache = get_supported()

        if not ignore_installed:
            env = get_default_environment()
            self._installed_dists = {
                dist.canonical_name: dist
                for dist in env.iter_installed_distributions(local_only=False)
            }
        else:
            self._installed_dists = {}

    @property
    def force_reinstall(self) -> bool:
        return self._force_reinstall

    def _fail_if_link_is_unsupported_wheel(self, link: Link) -> None:
        if not link.is_wheel:
            return
        wheel = Wheel(link.filename)
        if wheel.supported(self._finder.target_python.get_unsorted_tags()):
            return
        msg = f"{link.filename} is not a supported wheel on this platform."
        raise UnsupportedWheel(msg)

    def _make_extras_candidate(
        self,
        base: BaseCandidate,
        extras: FrozenSet[str],
        *,
        comes_from: Optional[InstallRequirement] = None,
    ) -> ExtrasCandidate:
        cache_key = (id(base), frozenset(canonicalize_name(e) for e in extras))
        try:
            candidate = self._extras_candidate_cache[cache_key]
        except KeyError:
            candidate = ExtrasCandidate(base, extras, comes_from=comes_from)
            self._extras_candidate_cache[cache_key] = candidate
        return candidate

    def _make_candidate_from_dist(
        self,
        dist: BaseDistribution,
        extras: FrozenSet[str],
        template: InstallRequirement,
    ) -> Candidate:
        try:
            base = self._installed_candidate_cache[dist.canonical_name]
        except KeyError:
            base = AlreadyInstalledCandidate(dist, template, factory=self)
            self._installed_candidate_cache[dist.canonical_name] = base
        if not extras:
            return base
        return self._make_extras_candidate(base, extras, comes_from=template)

    def _make_candidate_from_link(
        self,
        link: Link,
        extras: FrozenSet[str],
        template: InstallRequirement,
        name: Optional[NormalizedName],
        version: Optional[Version],
    ) -> Optional[Candidate]:
        base: Optional[BaseCandidate] = self._make_base_candidate_from_link(
            link, template, name, version
        )
        if not extras or base is None:
            return base
        return self._make_extras_candidate(base, extras, comes_from=template)

    def _make_base_candidate_from_link(
        self,
        link: Link,
        template: InstallRequirement,
        name: Optional[NormalizedName],
        version: Optional[Version],
    ) -> Optional[BaseCandidate]:
        # TODO: Check already installed candidate, and use it if the link and
        # editable flag match.

        if link in self._build_failures:
            # We already tried this candidate before, and it does not build.
            # Don't bother trying again.
            return None

        if template.editable:
            if link not in self._editable_candidate_cache:
                try:
                    self._editable_candidate_cache[link] = EditableCandidate(
                        link,
                        template,
                        factory=self,
                        name=name,
                        version=version,
                    )
                except (MetadataInconsistent, MetadataInvalid) as e:
                    logger.info(
                        "Discarding [blue underline]%s[/]: [yellow]%s[reset]",
                        link,
                        e,
                        extra={"markup": True},
                    )
                    self._build_failures[link] = e
                    return None

            return self._editable_candidate_cache[link]
        else:
            if link not in self._link_candidate_cache:
                try:
                    self._link_candidate_cache[link] = LinkCandidate(
                        link,
                        template,
                        factory=self,
                        name=name,
                        version=version,
                    )
                except MetadataInconsistent as e:
                    logger.info(
                        "Discarding [blue underline]%s[/]: [yellow]%s[reset]",
                        link,
                        e,
                        extra={"markup": True},
                    )
                    self._build_failures[link] = e
                    return None
            return self._link_candidate_cache[link]

    def _iter_found_candidates(
        self,
        ireqs: Sequence[InstallRequirement],
        specifier: SpecifierSet,
        hashes: Hashes,
        prefers_installed: bool,
        incompatible_ids: Set[int],
    ) -> Iterable[Candidate]:
        if not ireqs:
            return ()

        # The InstallRequirement implementation requires us to give it a
        # "template". Here we just choose the first requirement to represent
        # all of them.
        # Hopefully the Project model can correct this mismatch in the future.
        template = ireqs[0]
        assert template.req, "Candidates found on index must be PEP 508"
        name = canonicalize_name(template.req.name)

        extras: FrozenSet[str] = frozenset()
        for ireq in ireqs:
            assert ireq.req, "Candidates found on index must be PEP 508"
            specifier &= ireq.req.specifier
            hashes &= ireq.hashes(trust_internet=False)
            extras |= frozenset(ireq.extras)

        def _get_installed_candidate() -> Optional[Candidate]:
            """Get the candidate for the currently-installed version."""
            # If --force-reinstall is set, we want the version from the index
            # instead, so we "pretend" there is nothing installed.
            if self._force_reinstall:
                return None
            try:
                installed_dist = self._installed_dists[name]
            except KeyError:
                return None

            try:
                # Don't use the installed distribution if its version
                # does not fit the current dependency graph.
                if not specifier.contains(installed_dist.version, prereleases=True):
                    return None
            except InvalidVersion as e:
                raise InvalidInstalledPackage(dist=installed_dist, invalid_exc=e)

            candidate = self._make_candidate_from_dist(
                dist=installed_dist,
                extras=extras,
                template=template,
            )
            # The candidate is a known incompatibility. Don't use it.
            if id(candidate) in incompatible_ids:
                return None
            return candidate

        def iter_index_candidate_infos() -> Iterator[IndexCandidateInfo]:
            result = self._finder.find_best_candidate(
                project_name=name,
                specifier=specifier,
                hashes=hashes,
            )
            icans = result.applicable_candidates

            # PEP 592: Yanked releases are ignored unless the specifier
            # explicitly pins a version (via '==' or '===') that can be
            # solely satisfied by a yanked release.
            all_yanked = all(ican.link.is_yanked for ican in icans)

            def is_pinned(specifier: SpecifierSet) -> bool:
                for sp in specifier:
                    if sp.operator == "===":
                        return True
                    if sp.operator != "==":
                        continue
                    if sp.version.endswith(".*"):
                        continue
                    return True
                return False

            pinned = is_pinned(specifier)

            # PackageFinder returns earlier versions first, so we reverse.
            for ican in reversed(icans):
                if not (all_yanked and pinned) and ican.link.is_yanked:
                    continue
                func = functools.partial(
                    self._make_candidate_from_link,
                    link=ican.link,
                    extras=extras,
                    template=template,
                    name=name,
                    version=ican.version,
                )
                yield ican.version, func

        return FoundCandidates(
            iter_index_candidate_infos,
            _get_installed_candidate(),
            prefers_installed,
            incompatible_ids,
        )

    def _iter_explicit_candidates_from_base(
        self,
        base_requirements: Iterable[Requirement],
        extras: FrozenSet[str],
    ) -> Iterator[Candidate]:
        """Produce explicit candidates from the base given an extra-ed package.

        :param base_requirements: Requirements known to the resolver. The
            requirements are guaranteed to not have extras.
        :param extras: The extras to inject into the explicit requirements'
            candidates.
        """
        for req in base_requirements:
            lookup_cand, _ = req.get_candidate_lookup()
            if lookup_cand is None:  # Not explicit.
                continue
            # We've stripped extras from the identifier, and should always
            # get a BaseCandidate here, unless there's a bug elsewhere.
            base_cand = as_base_candidate(lookup_cand)
            assert base_cand is not None, "no extras here"
            yield self._make_extras_candidate(base_cand, extras)

    def _iter_candidates_from_constraints(
        self,
        identifier: str,
        constraint: Constraint,
        template: InstallRequirement,
    ) -> Iterator[Candidate]:
        """Produce explicit candidates from constraints.

        This creates "fake" InstallRequirement objects that are basically clones
        of what "should" be the template, but with original_link set to link.
        """
        for link in constraint.links:
            self._fail_if_link_is_unsupported_wheel(link)
            candidate = self._make_base_candidate_from_link(
                link,
                template=install_req_from_link_and_ireq(link, template),
                name=canonicalize_name(identifier),
                version=None,
            )
            if candidate:
                yield candidate

    def find_candidates(
        self,
        identifier: str,
        requirements: Mapping[str, Iterable[Requirement]],
        incompatibilities: Mapping[str, Iterator[Candidate]],
        constraint: Constraint,
        prefers_installed: bool,
        is_satisfied_by: Callable[[Requirement, Candidate], bool],
    ) -> Iterable[Candidate]:
        # Collect basic lookup information from the requirements.
        explicit_candidates: Set[Candidate] = set()
        ireqs: List[InstallRequirement] = []
        for req in requirements[identifier]:
            cand, ireq = req.get_candidate_lookup()
            if cand is not None:
                explicit_candidates.add(cand)
            if ireq is not None:
                ireqs.append(ireq)

        # If the current identifier contains extras, add requires and explicit
        # candidates from entries from extra-less identifier.
        with contextlib.suppress(InvalidRequirement):
            parsed_requirement = get_requirement(identifier)
            if parsed_requirement.name != identifier:
                explicit_candidates.update(
                    self._iter_explicit_candidates_from_base(
                        requirements.get(parsed_requirement.name, ()),
                        frozenset(parsed_requirement.extras),
                    ),
                )
                for req in requirements.get(parsed_requirement.name, []):
                    _, ireq = req.get_candidate_lookup()
                    if ireq is not None:
                        ireqs.append(ireq)

        # Add explicit candidates from constraints. We only do this if there are
        # known ireqs, which represent requirements not already explicit. If
        # there are no ireqs, we're constraining already-explicit requirements,
        # which is handled later when we return the explicit candidates.
        if ireqs:
            try:
                explicit_candidates.update(
                    self._iter_candidates_from_constraints(
                        identifier,
                        constraint,
                        template=ireqs[0],
                    ),
                )
            except UnsupportedWheel:
                # If we're constrained to install a wheel incompatible with the
                # target architecture, no candidates will ever be valid.
                return ()

        # Since we cache all the candidates, incompatibility identification
        # can be made quicker by comparing only the id() values.
        incompat_ids = {id(c) for c in incompatibilities.get(identifier, ())}

        # If none of the requirements want an explicit candidate, we can ask
        # the finder for candidates.
        if not explicit_candidates:
            return self._iter_found_candidates(
                ireqs,
                constraint.specifier,
                constraint.hashes,
                prefers_installed,
                incompat_ids,
            )

        return (
            c
            for c in explicit_candidates
            if id(c) not in incompat_ids
            and constraint.is_satisfied_by(c)
            and all(is_satisfied_by(req, c) for req in requirements[identifier])
        )

    def _make_requirements_from_install_req(
        self, ireq: InstallRequirement, requested_extras: Iterable[str]
    ) -> Iterator[Requirement]:
        """
        Returns requirement objects associated with the given InstallRequirement. In
        most cases this will be a single object but the following special cases exist:
            - the InstallRequirement has markers that do not apply -> result is empty
            - the InstallRequirement has both a constraint (or link) and extras
                -> result is split in two requirement objects: one with the constraint
                (or link) and one with the extra. This allows centralized constraint
                handling for the base, resulting in fewer candidate rejections.
        """
        if not ireq.match_markers(requested_extras):
            logger.info(
                "Ignoring %s: markers '%s' don't match your environment",
                ireq.name,
                ireq.markers,
            )
        elif not ireq.link:
            if ireq.extras and ireq.req is not None and ireq.req.specifier:
                yield SpecifierWithoutExtrasRequirement(ireq)
            yield SpecifierRequirement(ireq)
        else:
            self._fail_if_link_is_unsupported_wheel(ireq.link)
            # Always make the link candidate for the base requirement to make it
            # available to `find_candidates` for explicit candidate lookup for any
            # set of extras.
            # The extras are required separately via a second requirement.
            cand = self._make_base_candidate_from_link(
                ireq.link,
                template=install_req_drop_extras(ireq) if ireq.extras else ireq,
                name=canonicalize_name(ireq.name) if ireq.name else None,
                version=None,
            )
            if cand is None:
                # There's no way we can satisfy a URL requirement if the underlying
                # candidate fails to build. An unnamed URL must be user-supplied, so
                # we fail eagerly. If the URL is named, an unsatisfiable requirement
                # can make the resolver do the right thing, either backtrack (and
                # maybe find some other requirement that's buildable) or raise a
                # ResolutionImpossible eventually.
                if not ireq.name:
                    raise self._build_failures[ireq.link]
                yield UnsatisfiableRequirement(canonicalize_name(ireq.name))
            else:
                # require the base from the link
                yield self.make_requirement_from_candidate(cand)
                if ireq.extras:
                    # require the extras on top of the base candidate
                    yield self.make_requirement_from_candidate(
                        self._make_extras_candidate(cand, frozenset(ireq.extras))
                    )

    def collect_root_requirements(
        self, root_ireqs: List[InstallRequirement]
    ) -> CollectedRootRequirements:
        collected = CollectedRootRequirements([], {}, {})
        for i, ireq in enumerate(root_ireqs):
            if ireq.constraint:
                # Ensure we only accept valid constraints
                problem = check_invalid_constraint_type(ireq)
                if problem:
                    raise InstallationError(problem)
                if not ireq.match_markers():
                    continue
                assert ireq.name, "Constraint must be named"
                name = canonicalize_name(ireq.name)
                if name in collected.constraints:
                    collected.constraints[name] &= ireq
                else:
                    collected.constraints[name] = Constraint.from_ireq(ireq)
            else:
                reqs = list(
                    self._make_requirements_from_install_req(
                        ireq,
                        requested_extras=(),
                    )
                )
                if not reqs:
                    continue
                template = reqs[0]
                if ireq.user_supplied and template.name not in collected.user_requested:
                    collected.user_requested[template.name] = i
                collected.requirements.extend(reqs)
        # Put requirements with extras at the end of the root requires. This does not
        # affect resolvelib's picking preference but it does affect its initial criteria
        # population: by putting extras at the end we enable the candidate finder to
        # present resolvelib with a smaller set of candidates to resolvelib, already
        # taking into account any non-transient constraints on the associated base. This
        # means resolvelib will have fewer candidates to visit and reject.
        # Python's list sort is stable, meaning relative order is kept for objects with
        # the same key.
        collected.requirements.sort(key=lambda r: r.name != r.project_name)
        return collected

    def make_requirement_from_candidate(
        self, candidate: Candidate
    ) -> ExplicitRequirement:
        return ExplicitRequirement(candidate)

    def make_requirements_from_spec(
        self,
        specifier: str,
        comes_from: Optional[InstallRequirement],
        requested_extras: Iterable[str] = (),
    ) -> Iterator[Requirement]:
        """
        Returns requirement objects associated with the given specifier. In most cases
        this will be a single object but the following special cases exist:
            - the specifier has markers that do not apply -> result is empty
            - the specifier has both a constraint and extras -> result is split
                in two requirement objects: one with the constraint and one with the
                extra. This allows centralized constraint handling for the base,
                resulting in fewer candidate rejections.
        """
        ireq = self._make_install_req_from_spec(specifier, comes_from)
        return self._make_requirements_from_install_req(ireq, requested_extras)

    def make_requires_python_requirement(
        self,
        specifier: SpecifierSet,
    ) -> Optional[Requirement]:
        if self._ignore_requires_python:
            return None
        # Don't bother creating a dependency for an empty Requires-Python.
        if not str(specifier):
            return None
        return RequiresPythonRequirement(specifier, self._python_candidate)

    def get_wheel_cache_entry(
        self, link: Link, name: Optional[str]
    ) -> Optional[CacheEntry]:
        """Look up the link in the wheel cache.

        If ``preparer.require_hashes`` is True, don't use the wheel cache,
        because cached wheels, always built locally, have different hashes
        than the files downloaded from the index server and thus throw false
        hash mismatches. Furthermore, cached wheels at present have
        nondeterministic contents due to file modification times.
        """
        if self._wheel_cache is None:
            return None
        return self._wheel_cache.get_cache_entry(
            link=link,
            package_name=name,
            supported_tags=self._supported_tags_cache,
        )

    def get_dist_to_uninstall(self, candidate: Candidate) -> Optional[BaseDistribution]:
        # TODO: Are there more cases this needs to return True? Editable?
        dist = self._installed_dists.get(candidate.project_name)
        if dist is None:  # Not installed, no uninstallation required.
            return None

        # We're installing into global site. The current installation must
        # be uninstalled, no matter it's in global or user site, because the
        # user site installation has precedence over global.
        if not self._use_user_site:
            return dist

        # We're installing into user site. Remove the user site installation.
        if dist.in_usersite:
            return dist

        # We're installing into user site, but the installed incompatible
        # package is in global site. We can't uninstall that, and would let
        # the new user installation to "shadow" it. But shadowing won't work
        # in virtual environments, so we error out.
        if running_under_virtualenv() and dist.in_site_packages:
            message = (
                f"Will not install to the user site because it will lack "
                f"sys.path precedence to {dist.raw_name} in {dist.location}"
            )
            raise InstallationError(message)
        return None

    def _report_requires_python_error(
        self, causes: Sequence["ConflictCause"]
    ) -> UnsupportedPythonVersion:
        assert causes, "Requires-Python error reported with no cause"

        version = self._python_candidate.version

        if len(causes) == 1:
            specifier = str(causes[0].requirement.specifier)
            message = (
                f"Package {causes[0].parent.name!r} requires a different "
                f"Python: {version} not in {specifier!r}"
            )
            return UnsupportedPythonVersion(message)

        message = f"Packages require a different Python. {version} not in:"
        for cause in causes:
            package = cause.parent.format_for_error()
            specifier = str(cause.requirement.specifier)
            message += f"\n{specifier!r} (required by {package})"
        return UnsupportedPythonVersion(message)

    def _report_single_requirement_conflict(
        self, req: Requirement, parent: Optional[Candidate]
    ) -> DistributionNotFound:
        if parent is None:
            req_disp = str(req)
        else:
            req_disp = f"{req} (from {parent.name})"

        cands = self._finder.find_all_candidates(req.project_name)
        skipped_by_requires_python = self._finder.requires_python_skipped_reasons()

        versions_set: Set[Version] = set()
        yanked_versions_set: Set[Version] = set()
        for c in cands:
            is_yanked = c.link.is_yanked if c.link else False
            if is_yanked:
                yanked_versions_set.add(c.version)
            else:
                versions_set.add(c.version)

        versions = [str(v) for v in sorted(versions_set)]
        yanked_versions = [str(v) for v in sorted(yanked_versions_set)]

        if yanked_versions:
            # Saying "version X is yanked" isn't entirely accurate.
            # https://github.com/pypa/pip/issues/11745#issuecomment-1402805842
            logger.critical(
                "Ignored the following yanked versions: %s",
                ", ".join(yanked_versions) or "none",
            )
        if skipped_by_requires_python:
            logger.critical(
                "Ignored the following versions that require a different python "
                "version: %s",
                "; ".join(skipped_by_requires_python) or "none",
            )
        logger.critical(
            "Could not find a version that satisfies the requirement %s "
            "(from versions: %s)",
            req_disp,
            ", ".join(versions) or "none",
        )
        if str(req) == "requirements.txt":
            logger.info(
                "HINT: You are attempting to install a package literally "
                'named "requirements.txt" (which cannot exist). Consider '
                "using the '-r' flag to install the packages listed in "
                "requirements.txt"
            )

        return DistributionNotFound(f"No matching distribution found for {req}")

    def get_installation_error(
        self,
        e: "ResolutionImpossible[Requirement, Candidate]",
        constraints: Dict[str, Constraint],
    ) -> InstallationError:
        assert e.causes, "Installation error reported with no cause"

        # If one of the things we can't solve is "we need Python X.Y",
        # that is what we report.
        requires_python_causes = [
            cause
            for cause in e.causes
            if isinstance(cause.requirement, RequiresPythonRequirement)
            and not cause.requirement.is_satisfied_by(self._python_candidate)
        ]
        if requires_python_causes:
            # The comprehension above makes sure all Requirement instances are
            # RequiresPythonRequirement, so let's cast for convenience.
            return self._report_requires_python_error(
                cast("Sequence[ConflictCause]", requires_python_causes),
            )

        # Otherwise, we have a set of causes which can't all be satisfied
        # at once.

        # The simplest case is when we have *one* cause that can't be
        # satisfied. We just report that case.
        if len(e.causes) == 1:
            req, parent = next(iter(e.causes))
            if req.name not in constraints:
                return self._report_single_requirement_conflict(req, parent)

        # OK, we now have a list of requirements that can't all be
        # satisfied at once.

        # A couple of formatting helpers
        def text_join(parts: List[str]) -> str:
            if len(parts) == 1:
                return parts[0]

            return ", ".join(parts[:-1]) + " and " + parts[-1]

        def describe_trigger(parent: Candidate) -> str:
            ireq = parent.get_install_requirement()
            if not ireq or not ireq.comes_from:
                return f"{parent.name}=={parent.version}"
            if isinstance(ireq.comes_from, InstallRequirement):
                return str(ireq.comes_from.name)
            return str(ireq.comes_from)

        triggers = set()
        for req, parent in e.causes:
            if parent is None:
                # This is a root requirement, so we can report it directly
                trigger = req.format_for_error()
            else:
                trigger = describe_trigger(parent)
            triggers.add(trigger)

        if triggers:
            info = text_join(sorted(triggers))
        else:
            info = "the requested packages"

        msg = (
            f"Cannot install {info} because these package versions "
            "have conflicting dependencies."
        )
        logger.critical(msg)
        msg = "\nThe conflict is caused by:"

        relevant_constraints = set()
        for req, parent in e.causes:
            if req.name in constraints:
                relevant_constraints.add(req.name)
            msg = msg + "\n    "
            if parent:
                msg = msg + f"{parent.name} {parent.version} depends on "
            else:
                msg = msg + "The user requested "
            msg = msg + req.format_for_error()
        for key in relevant_constraints:
            spec = constraints[key].specifier
            msg += f"\n    The user requested (constraint) {key}{spec}"

        msg = (
            msg
            + "\n\n"
            + "To fix this you could try to:\n"
            + "1. loosen the range of package versions you've specified\n"
            + "2. remove package versions to allow pip to attempt to solve "
            + "the dependency conflict\n"
        )

        logger.info(msg)

        return DistributionNotFound(
            "ResolutionImpossible: for help visit "
            "https://pip.pypa.io/en/latest/topics/dependency-resolution/"
            "#dealing-with-dependency-conflicts"
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\longformer\benchmark_longformer.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
#
# This script run benchmark of latency or peak memory usage of Longformer model inference.
# Please run convert_to_onnx.py to get onnx model before running benchmark.
#
# It is tested with python 3.8, onnxruntime-gpu 1.11.0, PyTorch 1.11.0, transformers 4.18.0, CUDA 11.3 like:
#   conda create -n gpu_env python=3.8
#   conda activate gpu_env
#   pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#   pip3 install onnx transformers onnxruntime-gpu numpy sympy coloredlogs psutil py3nvml
#   python benchmark_longformer.py
#
# When there is no parameter, pre-defined tests will run on the longformer-base-4096 model.

# Benchmark the latency:
#   python benchmark_longformer.py --model longformer-base-4096 --batch_sizes 1 --sequence_lengths 512 1024 2048 4096 \
#          --global_lengths 8 --onnx ./longformer-base-4096_fp16.onnx -t 100
#
# Benchmark GPU peak memory:
#   export ORT_LONGFORMER_COMPACT_MEMORY=0
#   python benchmark_longformer.py --model longformer-base-4096 --batch_sizes 1 --sequence_lengths 4096 \
#          --global_lengths 8 --onnx ./longformer-base-4096_fp32.onnx --memory -t 10 --engine onnxruntime
#   export ORT_LONGFORMER_COMPACT_MEMORY=1
#   python benchmark_longformer.py --model longformer-base-4096 --batch_sizes 1 --sequence_lengths 4096 \
#          --global_lengths 8 --onnx ./longformer-base-4096_fp32.onnx --memory -t 10 --engine onnxruntime
#
# By default, compact memory kernel is enabled. To disable it, set environment variable ORT_LONGFORMER_COMPACT_MEMORY=0.

import argparse
import csv
import logging
import math
import os
import re
import sys
import timeit
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

import benchmark_helper
import numpy as np
import torch
from longformer_helper import PRETRAINED_LONGFORMER_MODELS, LongformerHelper, LongformerInputs
from transformers import LongformerModel

import onnxruntime

logger = logging.getLogger("")


def test_torch_latency(
    device,
    model,
    model_name,
    batch_sizes,
    sequence_lengths,
    global_lengths,
    test_times,
    num_threads,
) -> list[dict[str, Any]]:
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    results = []
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            for global_length in global_lengths:
                logger.info(f"batch_size={batch_size} sequence_length={sequence_length} global_length={global_length}")
                inputs: LongformerInputs = LongformerHelper.get_dummy_inputs(
                    batch_size, sequence_length, global_length, device
                )
                input_list = inputs.to_list()

                _ = model(*input_list)
                runtimes = timeit.repeat(lambda: model(*input_list), repeat=test_times, number=1)  # noqa: B023
                result = {
                    "engine": "torch",  # TODO: test torchscript
                    "version": torch.__version__,
                    "device": "cuda",
                    "optimizer": "",
                    "precision": "fp32",
                    "io_binding": "",
                    "model_name": model_name,
                    "description": model_name + " [torch]",
                    "inputs": 3,
                    "threads": num_threads,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "global_length": global_length,
                    "datetime": str(datetime.now()),
                    "memory": "NA",
                    "diff_max": 0,
                    "diff_90_percentile": 0,
                    "diff_95_percentile": 0,
                    "diff_99_percentile": 0,
                    "use_compact_memory": "NA",
                }
                result.update(benchmark_helper.get_latency_result(runtimes, batch_size))
                logger.info("%s", result)
                results.append(result)
    return results


def test_parity(device, model, ort_session, batch_size, sequence_length, global_length, verbose=True):
    parameters = f"batch_size={batch_size} sequence_length={sequence_length} global_length={global_length}"
    logger.info(f"Comparing Torch and ORT outputs for {parameters}...")
    dummy_inputs: LongformerInputs = LongformerHelper.get_dummy_inputs(
        batch_size, sequence_length, global_length, device
    )
    ort_inputs = dummy_inputs.get_ort_inputs()
    ort_outputs = ort_session.run(None, ort_inputs)
    input_list = dummy_inputs.to_list()
    torch_outputs = model(*input_list)
    max_diff = np.amax(torch_outputs[0].cpu().numpy() - ort_outputs[0])
    logger.info(f"last_state max diff = {max_diff}")
    if verbose and (math.isnan(max_diff) or max_diff > 0.001):
        print("torch last_state:", torch_outputs[0])
        print("ort last_state:", ort_outputs[0])
    return float(max_diff)


def test_ort_latency(
    device,
    model,
    model_name,
    description,
    ort_session,
    batch_sizes,
    sequence_lengths,
    global_lengths,
    test_times,
    num_threads,
    optimizer=False,
    precision="fp32",
    disable_io_binding=False,
    verbose=True,
    use_compact_memory=False,
    use_half4=False,
    disable_parity=False,
) -> list[dict[str, Any]]:
    results = []
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            for global_length in global_lengths:
                assert global_length <= model.config.attention_window[0], (
                    "Limitation of current implementation: number of global token <= attention_window"
                )

                logger.info(
                    f"Testing batch_size={batch_size} sequence_length={sequence_length} global_length={global_length} "
                    f"optimizer={optimizer}, precision={precision} io_binding={not disable_io_binding}..."
                )
                dummy_inputs: LongformerInputs = LongformerHelper.get_dummy_inputs(
                    batch_size, sequence_length, global_length, device
                )

                # Run OnnxRuntime
                ort_inputs = dummy_inputs.get_ort_inputs()

                if verbose:
                    print(ort_inputs)

                # run one query for warm up
                ort_outputs = ort_session.run(None, ort_inputs)

                result_template = {
                    "model_name": model_name,
                    "description": description,
                    "inputs": 3,
                    "engine": "OnnxRuntime",
                    "version": str(onnxruntime.__version__),
                    "device": "cuda",
                    "precision": str(precision),
                    "optimizer": int(optimizer),
                    "threads": int(num_threads),
                    "batch_size": int(batch_size),
                    "sequence_length": int(sequence_length),
                    "global_length": int(global_length),
                    "test_times": int(test_times),
                    "datetime": str(datetime.now()),
                    "memory": "",
                    "diff_max": None,
                    "diff_90_percentile": None,
                    "diff_95_percentile": None,
                    "diff_99_percentile": None,
                    "use_compact_memory": use_compact_memory,
                    "use_half4": use_half4,
                }

                if not disable_io_binding:
                    max_last_state_size = max(batch_sizes) * max(sequence_lengths) * model.config.hidden_size
                    max_pooler_size = max(batch_sizes) * max(sequence_lengths)
                    result = benchmark_helper.inference_ort_with_io_binding(
                        ort_session,
                        ort_inputs,
                        result_template=result_template,
                        repeat_times=test_times,
                        ort_output_names=["last_state", "pooler"],
                        ort_outputs=ort_outputs,
                        output_buffers=[],
                        output_buffer_max_sizes=[max_last_state_size, max_pooler_size],
                        batch_size=batch_size,
                        device=device,
                        data_type=np.longlong,  # input data type
                    )
                else:
                    result = benchmark_helper.inference_ort(
                        ort_session,
                        ort_inputs,
                        result_template=result_template,
                        repeat_times=test_times,
                        batch_size=batch_size,
                    )

                # measure result difference between PyTorch and OnnxRuntime
                if not disable_parity:
                    diff_results = [
                        test_parity(
                            device,
                            model,
                            ort_session,
                            batch_size,
                            sequence_length,
                            global_length,
                            verbose,
                        )
                        for _ in range(test_times)
                    ]

                    result["diff_max"] = max(diff_results)
                    result["diff_90_percentile"] = np.percentile(diff_results, 90)
                    result["diff_95_percentile"] = np.percentile(diff_results, 95)
                    result["diff_99_percentile"] = np.percentile(diff_results, 99)

                results.append(result)
    return results


def test_ort_memory(
    device,
    onnx_model_path,
    batch_size,
    sequence_length,
    global_length,
    test_times,
    num_threads,
) -> dict[str, Any]:
    logger.info(
        f"Testing memory for model={onnx_model_path}, batch_size={batch_size}, sequence_length={sequence_length}, "
        f"global_length={global_length}, test_times={test_times}, num_threads={num_threads}"
    )

    def inference():
        # Update Arena strategy so that we can measure the minimum memory required
        cuda_provider_options = {"arena_extend_strategy": "kSameAsRequested"}
        provider_options = {"CUDAExecutionProvider": cuda_provider_options}
        session = benchmark_helper.create_onnxruntime_session(
            onnx_model_path,
            use_gpu=True,
            enable_all_optimization=True,
            num_threads=num_threads,
            provider_options=provider_options,
        )

        dummy_inputs: LongformerInputs = LongformerHelper.get_dummy_inputs(
            batch_size, sequence_length, global_length, device
        )
        ort_inputs = dummy_inputs.get_ort_inputs()
        for _ in range(test_times):
            _ = session.run(None, ort_inputs)

    memory_used = benchmark_helper.measure_memory(is_gpu=True, func=inference)

    return {
        "onnx_model": onnx_model_path,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "global_length": global_length,
        "test_times": test_times,
        "num_threads": num_threads,
        "memory": memory_used,
    }


def load_torch_model(model_name, device):
    torch_model_name_or_dir = PRETRAINED_LONGFORMER_MODELS.get(model_name, model_name)
    model = LongformerModel.from_pretrained(torch_model_name_or_dir)
    model.to(device)
    return model


def find_onnx_model(model_name, onnx_dir="."):
    # Search onnx model in the following order: optimized fp16 model, optimized fp32 model, raw model
    onnx_model_path = os.path.join(onnx_dir, model_name + ".onnx")
    optimized_fp32_model = os.path.join(onnx_dir, model_name + "_fp32.onnx")
    optimized_fp16_model = os.path.join(onnx_dir, model_name + "_fp16.onnx")
    if os.path.isfile(optimized_fp16_model):
        onnx_model_path = optimized_fp16_model
    elif os.path.isfile(optimized_fp32_model):
        onnx_model_path = optimized_fp32_model
    return onnx_model_path


def test_memory(args, device) -> dict[str, Any]:
    if len(args.batch_sizes) > 1:
        raise RuntimeError("For memory test, only one batch_size (-b) is allowed.")
    if len(args.sequence_lengths) > 1:
        raise RuntimeError("For memory test, only one sequence_length (-s) is allowed.")
    if len(args.global_lengths) > 1:
        raise RuntimeError("For memory test, only one global_length (-g) is allowed.")

    model_name = args.model
    onnx_model_path = find_onnx_model(model_name) if not args.onnx else args.onnx

    torch.cuda.empty_cache()
    return test_ort_memory(
        device,
        onnx_model_path,
        args.batch_sizes[0],
        args.sequence_lengths[0],
        args.global_lengths[0],
        args.test_times,
        args.num_threads,
    )


def test_ort(args, device) -> list[dict[str, Any]]:
    model_name = args.model

    onnx_model_path = find_onnx_model(model_name) if not args.onnx else args.onnx

    optimized = onnx_model_path.endswith("_fp16.onnx") or onnx_model_path.endswith("_fp32.onnx")  # noqa: PIE810
    precision = "fp32" if not onnx_model_path.endswith("_fp16.onnx") else "fp16"

    model = load_torch_model(model_name, device)

    num_threads = args.num_threads

    cuda_provider_options = {"arena_extend_strategy": "kSameAsRequested"}
    provider_options = {"CUDAExecutionProvider": cuda_provider_options}
    session = benchmark_helper.create_onnxruntime_session(
        onnx_model_path,
        use_gpu=True,
        enable_all_optimization=True,
        num_threads=num_threads,
        provider_options=provider_options,
    )
    if session is None:
        raise RuntimeError(f"Failed to create ORT session from ONNX file {onnx_model_path}")

    use_compact_memory = os.environ.get("ORT_LONGFORMER_COMPACT_MEMORY", "1") == "1"
    description = onnx_model_path
    if not use_compact_memory:
        description += "[non_compact_memory]"

    if args.use_half4:
        description += "[half4]" if precision == "fp16" else "[float4]"
    else:
        description += "[half2]" if precision == "fp16" else "[float4]"

    return test_ort_latency(
        device,
        model,
        model_name,
        description,
        session,
        args.batch_sizes,
        args.sequence_lengths,
        args.global_lengths,
        args.test_times,
        num_threads,
        optimized,
        precision,
        args.disable_io_binding,
        args.verbose,
        use_compact_memory,
        args.use_half4,
        args.disable_parity,
    )


def test_torch(args, device) -> list[dict[str, Any]]:
    model = load_torch_model(args.model, device)
    return test_torch_latency(
        device,
        model,
        args.model,
        args.batch_sizes,
        args.sequence_lengths,
        args.global_lengths,
        args.test_times,
        args.num_threads,
    )


def test_latency(args, device) -> list[dict[str, Any]]:
    if args.engine == "onnxruntime":
        return test_ort(args, device)

    return test_torch(args, device)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=False,
        type=str,
        default="longformer-base-4096",
        help="Checkpoint directory or pre-trained model names in the list: "
        + ", ".join(PRETRAINED_LONGFORMER_MODELS.keys()),
    )

    parser.add_argument(
        "-e",
        "--engine",
        required=False,
        type=str,
        default="onnxruntime",
        choices=["onnxruntime", "torch"],
        help="Engine to benchmark.",
    )

    parser.add_argument(
        "-t",
        "--test_times",
        required=False,
        default=1000,
        type=int,
        help="Number of repeat times to get average inference latency.",
    )

    parser.add_argument("-b", "--batch_sizes", nargs="+", type=int, default=[1])

    # If --export_padding is not used in exporting onnx model, there is no padding in ONNX model,
    # and you will need padding inputs by yourself before running onnx model.
    # Here, we only test sequence length that is multiple of attention window size.
    parser.add_argument(
        "-s",
        "--sequence_lengths",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="Sequence lengths. It could have multiple values in latency test."
        "If --export_padding is not used, sequence length shall be multiple of window size.",
    )

    parser.add_argument("--onnx", required=False, type=str, default=None, help="Onnx model path")

    parser.add_argument(
        "-g",
        "--global_lengths",
        nargs="+",
        type=int,
        default=[0],
        help="Number of global tokens. It could have multiple values in latency test.",
    )

    parser.add_argument(
        "-n",
        "--num_threads",
        required=False,
        type=int,
        default=0,
        help="Threads to use.",
    )

    parser.add_argument(
        "--disable_io_binding",
        required=False,
        action="store_true",
        help="Do not use IO Binding.",
    )

    parser.add_argument(
        "--memory",
        required=False,
        action="store_true",
        help="Test memory usage instead of latency.",
    )

    parser.add_argument("--verbose", required=False, action="store_true", help="Print more information.")
    parser.set_defaults(verbose=False)

    parser.add_argument("--use_half4", required=False, action="store_true", help="Use half4 kernel.")
    parser.set_defaults(use_half4=False)

    parser.add_argument("--disable_parity", required=False, action="store_true", help="Do not run parity test.")
    parser.set_defaults(disable_parity=False)

    args = parser.parse_args(argv)

    return args


def output_details(results, csv_filename):
    latency_results = [result for result in results if "average_latency_ms" in result]
    if len(latency_results) == 0:
        print("No latency results for output.")
        return

    with open(csv_filename, mode="a", newline="", encoding="ascii") as csv_file:
        column_names = [
            "engine",
            "version",
            "device",
            "precision",
            "optimizer",
            "io_binding",
            "model_name",
            "inputs",
            "threads",
            "datetime",
            "test_times",
            "description",
            "batch_size",
            "sequence_length",
            "global_length",
            "use_compact_memory",
            "use_half4",
            "diff_max",
            "diff_90_percentile",
            "diff_95_percentile",
            "diff_99_percentile",
            "memory",
            "QPS",
            "average_latency_ms",
            "latency_variance",
            "latency_90_percentile",
            "latency_95_percentile",
            "latency_99_percentile",
        ]

        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        for result in latency_results:
            print(result)
            csv_writer.writerow(result)

        csv_file.flush()

    print(f"Detail results are saved to csv file: {csv_filename}")


def run(args) -> list[dict[str, Any]]:
    torch.set_grad_enabled(False)

    # set random seed manually to get deterministic results
    benchmark_helper.set_random_seed(123)

    # Currently, the longformer attention operator could only run in GPU (no CPU implementation yet).
    device = torch.device("cuda:0")

    if args.memory:
        return [test_memory(args, device)]  # Convert to List so that return type is same as test_latency

    return test_latency(args, device)


def launch_test(arguments) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("Please install PyTorch with Cuda, and use a machine with GPU for testing gpu performance.")

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run, [arguments]))
        assert len(results) == 1
        return results[0]


def run_tests(
    use_compact_memory=True,
    run_torch=False,
    run_memory=True,
    use_io_binding=True,
    use_fp16=True,
    use_merged_qkv_weights=True,
    use_half4=True,
    batch_size=1,
):
    compact_memory = "1" if use_compact_memory else "0"
    os.environ["ORT_LONGFORMER_COMPACT_MEMORY"] = compact_memory
    logger.info(f"ORT_LONGFORMER_COMPACT_MEMORY={compact_memory}")

    os.environ["ORT_LONGFORMER_USE_HALF4"] = "1" if use_half4 else "0"
    logger.info("ORT_LONGFORMER_USE_HALF4={}".format("1" if use_half4 else "0"))  # noqa: G001

    results = []
    test_times = 1000
    sequence_lengths = [4096, 2048, 1024, 512]
    batch_sizes = [batch_size]
    for model_name in ["longformer-base-4096"]:
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                for global_length in [16]:
                    if run_torch:
                        engine_name = "torch"
                        args = parse_arguments(
                            f"-e {engine_name} -t {test_times} -b {batch_size} -s {sequence_length} -g {global_length} "
                            f"-t {test_times} -m {model_name}".split(" ")
                        )
                        results += run(args)

                    engine_name = "onnxruntime"
                    file_format = 1 if use_merged_qkv_weights else 0
                    onnx_path = (
                        f"{model_name}_f{file_format}_fp16.onnx"
                        if use_fp16
                        else f"{model_name}_f{file_format}_fp32.onnx"
                    )
                    if not os.path.exists(onnx_path):
                        raise RuntimeError(f"onnx file not exists:{onnx_path}")

                    arguments = (
                        f"-e {engine_name} --onnx {onnx_path} "
                        f"-b {batch_size} -s {sequence_length} -g {global_length} -m {model_name}"
                    )

                    if not use_io_binding:
                        arguments += " --disable_io_binding"

                    if use_half4:
                        arguments += " --use_half4"

                    # Disable parity test to avoid out of memory for large batch size
                    if batch_size >= 4:
                        arguments += " --disable_parity"

                    memory_results = None
                    try:
                        if run_memory:
                            args = parse_arguments(f"{arguments} -t 10 --memory".split(" "))
                            memory_results = launch_test(args)

                        args = parse_arguments(f"{arguments} -t {test_times}".split(" "))
                        latency_results = launch_test(args)
                    except KeyboardInterrupt as exc:
                        raise RuntimeError("Keyboard Interrupted") from exc
                    except Exception:
                        traceback.print_exc()
                        continue

                    if len(latency_results) == 1:
                        latency_results[0]["memory"] = memory_results[0]["memory"] if memory_results else "N/A"
                    else:
                        raise RuntimeError("length of latency_results should be 1")

                    logger.info("%s", latency_results)

                    results += latency_results
    return results


def output_summary(results, csv_filename, data_field="average_latency_ms"):
    with open(csv_filename, mode="a", newline="", encoding="ascii") as csv_file:
        header_names = [
            "model_name",
            "precision",
            "engine",
            "version",
            "global_length",
            "use_compact_memory",
            "use_half4",
            "description",
        ]

        description_list = list({result["description"] for result in results})
        description_list.sort()

        batch_sizes = list({result["batch_size"] for result in results})
        batch_sizes.sort()

        sequence_lengths = list({result["sequence_length"] for result in results})
        sequence_lengths.sort()

        data_names = []
        for sequence_length in sequence_lengths:
            for batch_size in batch_sizes:
                data_names.append(f"b{batch_size}_s{sequence_length}")

        csv_writer = csv.DictWriter(csv_file, fieldnames=header_names + data_names)
        csv_writer.writeheader()

        for description in description_list:
            row = {}

            sum_latency = {}
            sum_latency.update({k: 0 for k in data_names})

            count_latency = {}
            count_latency.update({k: 0 for k in data_names})

            for result in results:
                if result["description"] == description and result[data_field]:
                    headers = {k: v for k, v in result.items() if k in header_names}
                    if not row:
                        row.update(headers)
                    else:
                        for k in header_names:
                            if row[k] != headers[k]:
                                raise RuntimeError("Description shall be unique")

                    batch_size = result["batch_size"]
                    sequence_length = result["sequence_length"]
                    key = f"b{batch_size}_s{sequence_length}"

                    try:
                        latency = float(result[data_field])
                    except ValueError:
                        continue

                    sum_latency[key] += latency
                    count_latency[key] += 1

            if row:
                for key in data_names:
                    if key in count_latency and count_latency[key] > 0:
                        row[key] = sum_latency[key] / count_latency[key]
                    else:
                        row[key] = ""

                csv_writer.writerow(row)

        csv_file.flush()


def run_experiments(use_fp16, batch_size, is_baseline=False):
    """Run experiments to compare different algorithms on one batch size"""
    test_results = run_tests(
        use_fp16=use_fp16,
        use_merged_qkv_weights=True,
        use_half4=False,
        batch_size=batch_size,
    )

    if is_baseline:
        return test_results

    if use_fp16:
        test_results += run_tests(
            use_fp16=use_fp16,
            use_merged_qkv_weights=True,
            use_half4=True,
            batch_size=batch_size,
        )

        test_results += run_tests(
            use_fp16=use_fp16,
            use_merged_qkv_weights=False,
            use_half4=True,
            batch_size=batch_size,
        )

    test_results += run_tests(
        use_fp16=use_fp16,
        use_merged_qkv_weights=False,
        use_half4=False,
        batch_size=batch_size,
    )

    return test_results


def main():
    torch.multiprocessing.set_start_method("spawn")

    args = parse_arguments()

    benchmark_helper.setup_logger(args.verbose)

    if len(sys.argv) > 1:
        test_results = launch_test(args)
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_filename = f"benchmark_detail_{time_stamp}.csv"
        output_details(test_results, csv_filename)
        return

    gpu_list = benchmark_helper.get_gpu_info()
    logger.info("GPU info: %s", gpu_list)
    fp16_batch_sizes = [16, 8, 4, 2, 1]
    fp32_batch_sizes = [4, 2, 1]
    if gpu_list and gpu_list[0]["total"] >= 32 * 1024 * 1024 * 1024:  # 32 GB
        fp16_batch_sizes = [64, 32, 16, 8, 4, 2, 1]
        fp32_batch_sizes = [16, 8, 4, 2, 1]

    gpu_name = re.sub(r"(?u)[^-\w.]", "_", gpu_list[0]["name"]) if gpu_list else "gpu"
    is_baseline = os.environ.get("ORT_LONGFORMER_BASELINE", "0") == "1"
    experiment_name = f"longformer_base_{gpu_name}" + ("_baseline" if is_baseline else "")
    logger.info(
        f"experiment_name={experiment_name}, fp16_batch_sizes={fp16_batch_sizes}, fp32_batch_sizes={fp32_batch_sizes}"
    )

    total_runs = 1
    all_results = []
    for _ in range(total_runs):
        for batch_size in fp16_batch_sizes:
            fp16_results = run_experiments(use_fp16=True, batch_size=batch_size, is_baseline=is_baseline)
            output_details(fp16_results, "longformer_base_fp16.csv")
            all_results += fp16_results
    for metric_name in ["average_latency_ms", "QPS", "memory", "diff_90_percentile"]:
        output_summary(all_results, f"{experiment_name}_{metric_name}.csv", metric_name)

    all_results = []
    for _ in range(total_runs):
        for batch_size in fp32_batch_sizes:
            fp32_results = run_experiments(use_fp16=False, batch_size=batch_size, is_baseline=is_baseline)
            output_details(fp32_results, "longformer_base_fp32.csv")
            all_results += fp32_results
    for metric_name in ["average_latency_ms", "QPS", "memory", "diff_90_percentile"]:
        output_summary(all_results, f"{experiment_name}_{metric_name}.csv", metric_name)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\construction.py ===
"""
Constructor functions intended to be shared by pd.array, Series.__init__,
and Index.__new__.

These should not depend on core.internals.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
    cast,
    overload,
)
import warnings

import numpy as np
from numpy import ma

from pandas._config import using_string_dtype

from pandas._libs import lib
from pandas._libs.tslibs import (
    Period,
    get_supported_dtype,
    is_supported_dtype,
)
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    Dtype,
    DtypeObj,
    T,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    construct_1d_object_array_from_listlike,
    maybe_cast_to_datetime,
    maybe_cast_to_integer_array,
    maybe_convert_platform,
    maybe_infer_to_datetimelike,
    maybe_promote,
)
from pandas.core.dtypes.common import (
    is_list_like,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCExtensionArray,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

import pandas.core.common as com

if TYPE_CHECKING:
    from pandas import (
        Index,
        Series,
    )
    from pandas.core.arrays.base import ExtensionArray


def array(
    data: Sequence[object] | AnyArrayLike,
    dtype: Dtype | None = None,
    copy: bool = True,
) -> ExtensionArray:
    """
    Create an array.

    Parameters
    ----------
    data : Sequence of objects
        The scalars inside `data` should be instances of the
        scalar type for `dtype`. It's expected that `data`
        represents a 1-dimensional array of data.

        When `data` is an Index or Series, the underlying array
        will be extracted from `data`.

    dtype : str, np.dtype, or ExtensionDtype, optional
        The dtype to use for the array. This may be a NumPy
        dtype or an extension type registered with pandas using
        :meth:`pandas.api.extensions.register_extension_dtype`.

        If not specified, there are two possibilities:

        1. When `data` is a :class:`Series`, :class:`Index`, or
           :class:`ExtensionArray`, the `dtype` will be taken
           from the data.
        2. Otherwise, pandas will attempt to infer the `dtype`
           from the data.

        Note that when `data` is a NumPy array, ``data.dtype`` is
        *not* used for inferring the array type. This is because
        NumPy cannot represent all the types of data that can be
        held in extension arrays.

        Currently, pandas will infer an extension dtype for sequences of

        ============================== =======================================
        Scalar Type                    Array Type
        ============================== =======================================
        :class:`pandas.Interval`       :class:`pandas.arrays.IntervalArray`
        :class:`pandas.Period`         :class:`pandas.arrays.PeriodArray`
        :class:`datetime.datetime`     :class:`pandas.arrays.DatetimeArray`
        :class:`datetime.timedelta`    :class:`pandas.arrays.TimedeltaArray`
        :class:`int`                   :class:`pandas.arrays.IntegerArray`
        :class:`float`                 :class:`pandas.arrays.FloatingArray`
        :class:`str`                   :class:`pandas.arrays.StringArray` or
                                       :class:`pandas.arrays.ArrowStringArray`
        :class:`bool`                  :class:`pandas.arrays.BooleanArray`
        ============================== =======================================

        The ExtensionArray created when the scalar type is :class:`str` is determined by
        ``pd.options.mode.string_storage`` if the dtype is not explicitly given.

        For all other cases, NumPy's usual inference rules will be used.
    copy : bool, default True
        Whether to copy the data, even if not necessary. Depending
        on the type of `data`, creating the new array may require
        copying data, even if ``copy=False``.

    Returns
    -------
    ExtensionArray
        The newly created array.

    Raises
    ------
    ValueError
        When `data` is not 1-dimensional.

    See Also
    --------
    numpy.array : Construct a NumPy array.
    Series : Construct a pandas Series.
    Index : Construct a pandas Index.
    arrays.NumpyExtensionArray : ExtensionArray wrapping a NumPy array.
    Series.array : Extract the array stored within a Series.

    Notes
    -----
    Omitting the `dtype` argument means pandas will attempt to infer the
    best array type from the values in the data. As new array types are
    added by pandas and 3rd party libraries, the "best" array type may
    change. We recommend specifying `dtype` to ensure that

    1. the correct array type for the data is returned
    2. the returned array type doesn't change as new extension types
       are added by pandas and third-party libraries

    Additionally, if the underlying memory representation of the returned
    array matters, we recommend specifying the `dtype` as a concrete object
    rather than a string alias or allowing it to be inferred. For example,
    a future version of pandas or a 3rd-party library may include a
    dedicated ExtensionArray for string data. In this event, the following
    would no longer return a :class:`arrays.NumpyExtensionArray` backed by a
    NumPy array.

    >>> pd.array(['a', 'b'], dtype=str)
    <NumpyExtensionArray>
    ['a', 'b']
    Length: 2, dtype: str32

    This would instead return the new ExtensionArray dedicated for string
    data. If you really need the new array to be backed by a  NumPy array,
    specify that in the dtype.

    >>> pd.array(['a', 'b'], dtype=np.dtype("<U1"))
    <NumpyExtensionArray>
    ['a', 'b']
    Length: 2, dtype: str32

    Finally, Pandas has arrays that mostly overlap with NumPy

      * :class:`arrays.DatetimeArray`
      * :class:`arrays.TimedeltaArray`

    When data with a ``datetime64[ns]`` or ``timedelta64[ns]`` dtype is
    passed, pandas will always return a ``DatetimeArray`` or ``TimedeltaArray``
    rather than a ``NumpyExtensionArray``. This is for symmetry with the case of
    timezone-aware data, which NumPy does not natively support.

    >>> pd.array(['2015', '2016'], dtype='datetime64[ns]')
    <DatetimeArray>
    ['2015-01-01 00:00:00', '2016-01-01 00:00:00']
    Length: 2, dtype: datetime64[ns]

    >>> pd.array(["1h", "2h"], dtype='timedelta64[ns]')
    <TimedeltaArray>
    ['0 days 01:00:00', '0 days 02:00:00']
    Length: 2, dtype: timedelta64[ns]

    Examples
    --------
    If a dtype is not specified, pandas will infer the best dtype from the values.
    See the description of `dtype` for the types pandas infers for.

    >>> pd.array([1, 2])
    <IntegerArray>
    [1, 2]
    Length: 2, dtype: Int64

    >>> pd.array([1, 2, np.nan])
    <IntegerArray>
    [1, 2, <NA>]
    Length: 3, dtype: Int64

    >>> pd.array([1.1, 2.2])
    <FloatingArray>
    [1.1, 2.2]
    Length: 2, dtype: Float64

    >>> pd.array(["a", None, "c"])
    <StringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

    >>> with pd.option_context("string_storage", "pyarrow"):
    ...     arr = pd.array(["a", None, "c"])
    ...
    >>> arr
    <ArrowStringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

    >>> pd.array([pd.Period('2000', freq="D"), pd.Period("2000", freq="D")])
    <PeriodArray>
    ['2000-01-01', '2000-01-01']
    Length: 2, dtype: period[D]

    You can use the string alias for `dtype`

    >>> pd.array(['a', 'b', 'a'], dtype='category')
    ['a', 'b', 'a']
    Categories (2, object): ['a', 'b']

    Or specify the actual dtype

    >>> pd.array(['a', 'b', 'a'],
    ...          dtype=pd.CategoricalDtype(['a', 'b', 'c'], ordered=True))
    ['a', 'b', 'a']
    Categories (3, object): ['a' < 'b' < 'c']

    If pandas does not infer a dedicated extension type a
    :class:`arrays.NumpyExtensionArray` is returned.

    >>> pd.array([1 + 1j, 3 + 2j])
    <NumpyExtensionArray>
    [(1+1j), (3+2j)]
    Length: 2, dtype: complex128

    As mentioned in the "Notes" section, new extension types may be added
    in the future (by pandas or 3rd party libraries), causing the return
    value to no longer be a :class:`arrays.NumpyExtensionArray`. Specify the
    `dtype` as a NumPy dtype if you need to ensure there's no future change in
    behavior.

    >>> pd.array([1, 2], dtype=np.dtype("int32"))
    <NumpyExtensionArray>
    [1, 2]
    Length: 2, dtype: int32

    `data` must be 1-dimensional. A ValueError is raised when the input
    has the wrong dimensionality.

    >>> pd.array(1)
    Traceback (most recent call last):
      ...
    ValueError: Cannot pass scalar '1' to 'pandas.array'.
    """
    from pandas.core.arrays import (
        BooleanArray,
        DatetimeArray,
        ExtensionArray,
        FloatingArray,
        IntegerArray,
        IntervalArray,
        NumpyExtensionArray,
        PeriodArray,
        TimedeltaArray,
    )
    from pandas.core.arrays.string_ import StringDtype

    if lib.is_scalar(data):
        msg = f"Cannot pass scalar '{data}' to 'pandas.array'."
        raise ValueError(msg)
    elif isinstance(data, ABCDataFrame):
        raise TypeError("Cannot pass DataFrame to 'pandas.array'")

    if dtype is None and isinstance(data, (ABCSeries, ABCIndex, ExtensionArray)):
        # Note: we exclude np.ndarray here, will do type inference on it
        dtype = data.dtype

    data = extract_array(data, extract_numpy=True)

    # this returns None for not-found dtypes.
    if dtype is not None:
        dtype = pandas_dtype(dtype)

    if isinstance(data, ExtensionArray) and (dtype is None or data.dtype == dtype):
        # e.g. TimedeltaArray[s], avoid casting to NumpyExtensionArray
        if copy:
            return data.copy()
        return data

    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        return cls._from_sequence(data, dtype=dtype, copy=copy)

    if dtype is None:
        inferred_dtype = lib.infer_dtype(data, skipna=True)
        if inferred_dtype == "period":
            period_data = cast(Union[Sequence[Optional[Period]], AnyArrayLike], data)
            return PeriodArray._from_sequence(period_data, copy=copy)

        elif inferred_dtype == "interval":
            return IntervalArray(data, copy=copy)

        elif inferred_dtype.startswith("datetime"):
            # datetime, datetime64
            try:
                return DatetimeArray._from_sequence(data, copy=copy)
            except ValueError:
                # Mixture of timezones, fall back to NumpyExtensionArray
                pass

        elif inferred_dtype.startswith("timedelta"):
            # timedelta, timedelta64
            return TimedeltaArray._from_sequence(data, copy=copy)

        elif inferred_dtype == "string":
            # StringArray/ArrowStringArray depending on pd.options.mode.string_storage
            dtype = StringDtype()
            cls = dtype.construct_array_type()
            return cls._from_sequence(data, dtype=dtype, copy=copy)

        elif inferred_dtype == "integer":
            return IntegerArray._from_sequence(data, copy=copy)
        elif inferred_dtype == "empty" and not hasattr(data, "dtype") and not len(data):
            return FloatingArray._from_sequence(data, copy=copy)
        elif (
            inferred_dtype in ("floating", "mixed-integer-float")
            and getattr(data, "dtype", None) != np.float16
        ):
            # GH#44715 Exclude np.float16 bc FloatingArray does not support it;
            #  we will fall back to NumpyExtensionArray.
            return FloatingArray._from_sequence(data, copy=copy)

        elif inferred_dtype == "boolean":
            return BooleanArray._from_sequence(data, dtype="boolean", copy=copy)

    # Pandas overrides NumPy for
    #   1. datetime64[ns,us,ms,s]
    #   2. timedelta64[ns,us,ms,s]
    # so that a DatetimeArray is returned.
    if lib.is_np_dtype(dtype, "M") and is_supported_dtype(dtype):
        return DatetimeArray._from_sequence(data, dtype=dtype, copy=copy)
    if lib.is_np_dtype(dtype, "m") and is_supported_dtype(dtype):
        return TimedeltaArray._from_sequence(data, dtype=dtype, copy=copy)

    elif lib.is_np_dtype(dtype, "mM"):
        warnings.warn(
            r"datetime64 and timedelta64 dtype resolutions other than "
            r"'s', 'ms', 'us', and 'ns' are deprecated. "
            r"In future releases passing unsupported resolutions will "
            r"raise an exception.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    return NumpyExtensionArray._from_sequence(data, dtype=dtype, copy=copy)


_typs = frozenset(
    {
        "index",
        "rangeindex",
        "multiindex",
        "datetimeindex",
        "timedeltaindex",
        "periodindex",
        "categoricalindex",
        "intervalindex",
        "series",
    }
)


@overload
def extract_array(
    obj: Series | Index, extract_numpy: bool = ..., extract_range: bool = ...
) -> ArrayLike:
    ...


@overload
def extract_array(
    obj: T, extract_numpy: bool = ..., extract_range: bool = ...
) -> T | ArrayLike:
    ...


def extract_array(
    obj: T, extract_numpy: bool = False, extract_range: bool = False
) -> T | ArrayLike:
    """
    Extract the ndarray or ExtensionArray from a Series or Index.

    For all other types, `obj` is just returned as is.

    Parameters
    ----------
    obj : object
        For Series / Index, the underlying ExtensionArray is unboxed.

    extract_numpy : bool, default False
        Whether to extract the ndarray from a NumpyExtensionArray.

    extract_range : bool, default False
        If we have a RangeIndex, return range._values if True
        (which is a materialized integer ndarray), otherwise return unchanged.

    Returns
    -------
    arr : object

    Examples
    --------
    >>> extract_array(pd.Series(['a', 'b', 'c'], dtype='category'))
    ['a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Other objects like lists, arrays, and DataFrames are just passed through.

    >>> extract_array([1, 2, 3])
    [1, 2, 3]

    For an ndarray-backed Series / Index the ndarray is returned.

    >>> extract_array(pd.Series([1, 2, 3]))
    array([1, 2, 3])

    To extract all the way down to the ndarray, pass ``extract_numpy=True``.

    >>> extract_array(pd.Series([1, 2, 3]), extract_numpy=True)
    array([1, 2, 3])
    """
    typ = getattr(obj, "_typ", None)
    if typ in _typs:
        # i.e. isinstance(obj, (ABCIndex, ABCSeries))
        if typ == "rangeindex":
            if extract_range:
                # error: "T" has no attribute "_values"
                return obj._values  # type: ignore[attr-defined]
            return obj

        # error: "T" has no attribute "_values"
        return obj._values  # type: ignore[attr-defined]

    elif extract_numpy and typ == "npy_extension":
        # i.e. isinstance(obj, ABCNumpyExtensionArray)
        # error: "T" has no attribute "to_numpy"
        return obj.to_numpy()  # type: ignore[attr-defined]

    return obj


def ensure_wrapped_if_datetimelike(arr):
    """
    Wrap datetime64 and timedelta64 ndarrays in DatetimeArray/TimedeltaArray.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind == "M":
            from pandas.core.arrays import DatetimeArray

            dtype = get_supported_dtype(arr.dtype)
            return DatetimeArray._from_sequence(arr, dtype=dtype)

        elif arr.dtype.kind == "m":
            from pandas.core.arrays import TimedeltaArray

            dtype = get_supported_dtype(arr.dtype)
            return TimedeltaArray._from_sequence(arr, dtype=dtype)

    return arr


def sanitize_masked_array(data: ma.MaskedArray) -> np.ndarray:
    """
    Convert numpy MaskedArray to ensure mask is softened.
    """
    mask = ma.getmaskarray(data)
    if mask.any():
        dtype, fill_value = maybe_promote(data.dtype, np.nan)
        dtype = cast(np.dtype, dtype)
        data = ma.asarray(data.astype(dtype, copy=True))
        data.soften_mask()  # set hardmask False if it was True
        data[mask] = fill_value
    else:
        data = data.copy()
    return data


def sanitize_array(
    data,
    index: Index | None,
    dtype: DtypeObj | None = None,
    copy: bool = False,
    *,
    allow_2d: bool = False,
) -> ArrayLike:
    """
    Sanitize input data to an ndarray or ExtensionArray, copy if specified,
    coerce to the dtype if specified.

    Parameters
    ----------
    data : Any
    index : Index or None, default None
    dtype : np.dtype, ExtensionDtype, or None, default None
    copy : bool, default False
    allow_2d : bool, default False
        If False, raise if we have a 2D Arraylike.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    original_dtype = dtype
    if isinstance(data, ma.MaskedArray):
        data = sanitize_masked_array(data)

    if isinstance(dtype, NumpyEADtype):
        # Avoid ending up with a NumpyExtensionArray
        dtype = dtype.numpy_dtype

    object_index = False
    if isinstance(data, ABCIndex) and data.dtype == object and dtype is None:
        object_index = True

    # extract ndarray or ExtensionArray, ensure we have no NumpyExtensionArray
    data = extract_array(data, extract_numpy=True, extract_range=True)

    if isinstance(data, np.ndarray) and data.ndim == 0:
        if dtype is None:
            dtype = data.dtype
        data = lib.item_from_zerodim(data)
    elif isinstance(data, range):
        # GH#16804
        data = range_to_ndarray(data)
        copy = False

    if not is_list_like(data):
        if index is None:
            raise ValueError("index must be specified when data is not list-like")
        if isinstance(data, str) and using_string_dtype() and original_dtype is None:
            from pandas.core.arrays.string_ import StringDtype

            dtype = StringDtype(na_value=np.nan)
        data = construct_1d_arraylike_from_scalar(data, len(index), dtype)

        return data

    elif isinstance(data, ABCExtensionArray):
        # it is already ensured above this is not a NumpyExtensionArray
        # Until GH#49309 is fixed this check needs to come before the
        #  ExtensionDtype check
        if dtype is not None:
            subarr = data.astype(dtype, copy=copy)
        elif copy:
            subarr = data.copy()
        else:
            subarr = data

    elif isinstance(dtype, ExtensionDtype):
        # create an extension array from its dtype
        _sanitize_non_ordered(data)
        cls = dtype.construct_array_type()
        if not hasattr(data, "__array__"):
            data = list(data)
        subarr = cls._from_sequence(data, dtype=dtype, copy=copy)

    # GH#846
    elif isinstance(data, np.ndarray):
        if isinstance(data, np.matrix):
            data = data.A

        if dtype is None:
            subarr = data
            if data.dtype == object:
                subarr = maybe_infer_to_datetimelike(data)
                if object_index and using_string_dtype() and is_string_dtype(subarr):
                    # Avoid inference when string option is set
                    subarr = data
            elif data.dtype.kind == "U" and using_string_dtype():
                from pandas.core.arrays.string_ import StringDtype

                dtype = StringDtype(na_value=np.nan)
                subarr = dtype.construct_array_type()._from_sequence(data, dtype=dtype)

            if (
                subarr is data
                or (subarr.dtype == "str" and subarr.dtype.storage == "python")  # type: ignore[union-attr]
            ) and copy:
                subarr = subarr.copy()

        else:
            # we will try to copy by-definition here
            subarr = _try_cast(data, dtype, copy)

    elif hasattr(data, "__array__"):
        # e.g. dask array GH#38645
        if not copy:
            data = np.asarray(data)
        else:
            data = np.array(data, copy=copy)
        return sanitize_array(
            data,
            index=index,
            dtype=dtype,
            copy=False,
            allow_2d=allow_2d,
        )

    else:
        _sanitize_non_ordered(data)
        # materialize e.g. generators, convert e.g. tuples, abc.ValueView
        data = list(data)

        if len(data) == 0 and dtype is None:
            # We default to float64, matching numpy
            subarr = np.array([], dtype=np.float64)

        elif dtype is not None:
            subarr = _try_cast(data, dtype, copy)

        else:
            subarr = maybe_convert_platform(data)
            if subarr.dtype == object:
                subarr = cast(np.ndarray, subarr)
                subarr = maybe_infer_to_datetimelike(subarr)

    subarr = _sanitize_ndim(subarr, data, dtype, index, allow_2d=allow_2d)

    if isinstance(subarr, np.ndarray):
        # at this point we should have dtype be None or subarr.dtype == dtype
        dtype = cast(np.dtype, dtype)
        subarr = _sanitize_str_dtypes(subarr, data, dtype, copy)

    return subarr


def range_to_ndarray(rng: range) -> np.ndarray:
    """
    Cast a range object to ndarray.
    """
    # GH#30171 perf avoid realizing range as a list in np.array
    try:
        arr = np.arange(rng.start, rng.stop, rng.step, dtype="int64")
    except OverflowError:
        # GH#30173 handling for ranges that overflow int64
        if (rng.start >= 0 and rng.step > 0) or (rng.step < 0 <= rng.stop):
            try:
                arr = np.arange(rng.start, rng.stop, rng.step, dtype="uint64")
            except OverflowError:
                arr = construct_1d_object_array_from_listlike(list(rng))
        else:
            arr = construct_1d_object_array_from_listlike(list(rng))
    return arr


def _sanitize_non_ordered(data) -> None:
    """
    Raise only for unordered sets, e.g., not for dict_keys
    """
    if isinstance(data, (set, frozenset)):
        raise TypeError(f"'{type(data).__name__}' type is unordered")


def _sanitize_ndim(
    result: ArrayLike,
    data,
    dtype: DtypeObj | None,
    index: Index | None,
    *,
    allow_2d: bool = False,
) -> ArrayLike:
    """
    Ensure we have a 1-dimensional result array.
    """
    if getattr(result, "ndim", 0) == 0:
        raise ValueError("result should be arraylike with ndim > 0")

    if result.ndim == 1:
        # the result that we want
        result = _maybe_repeat(result, index)

    elif result.ndim > 1:
        if isinstance(data, np.ndarray):
            if allow_2d:
                return result
            raise ValueError(
                f"Data must be 1-dimensional, got ndarray of shape {data.shape} instead"
            )
        if is_object_dtype(dtype) and isinstance(dtype, ExtensionDtype):
            # i.e. NumpyEADtype("O")

            result = com.asarray_tuplesafe(data, dtype=np.dtype("object"))
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)
        else:
            # error: Argument "dtype" to "asarray_tuplesafe" has incompatible type
            # "Union[dtype[Any], ExtensionDtype, None]"; expected "Union[str,
            # dtype[Any], None]"
            result = com.asarray_tuplesafe(data, dtype=dtype)  # type: ignore[arg-type]
    return result


def _sanitize_str_dtypes(
    result: np.ndarray, data, dtype: np.dtype | None, copy: bool
) -> np.ndarray:
    """
    Ensure we have a dtype that is supported by pandas.
    """

    # This is to prevent mixed-type Series getting all casted to
    # NumPy string type, e.g. NaN --> '-1#IND'.
    if issubclass(result.dtype.type, str):
        # GH#16605
        # If not empty convert the data to dtype
        # GH#19853: If data is a scalar, result has already the result
        if not lib.is_scalar(data):
            if not np.all(isna(data)):
                data = np.asarray(data, dtype=dtype)
            if not copy:
                result = np.asarray(data, dtype=object)
            else:
                result = np.array(data, dtype=object, copy=copy)
    return result


def _maybe_repeat(arr: ArrayLike, index: Index | None) -> ArrayLike:
    """
    If we have a length-1 array and an index describing how long we expect
    the result to be, repeat the array.
    """
    if index is not None:
        if 1 == len(arr) != len(index):
            arr = arr.repeat(len(index))
    return arr


def _try_cast(
    arr: list | np.ndarray,
    dtype: np.dtype,
    copy: bool,
) -> ArrayLike:
    """
    Convert input to numpy ndarray and optionally cast to a given dtype.

    Parameters
    ----------
    arr : ndarray or list
        Excludes: ExtensionArray, Series, Index.
    dtype : np.dtype
    copy : bool
        If False, don't copy the data if not needed.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    is_ndarray = isinstance(arr, np.ndarray)

    if dtype == object:
        if not is_ndarray:
            subarr = construct_1d_object_array_from_listlike(arr)
            return subarr
        return ensure_wrapped_if_datetimelike(arr).astype(dtype, copy=copy)

    elif dtype.kind == "U":
        # TODO: test cases with arr.dtype.kind in "mM"
        if is_ndarray:
            arr = cast(np.ndarray, arr)
            shape = arr.shape
            if arr.ndim > 1:
                arr = arr.ravel()
        else:
            shape = (len(arr),)
        return lib.ensure_string_array(arr, convert_na_value=False, copy=copy).reshape(
            shape
        )

    elif dtype.kind in "mM":
        return maybe_cast_to_datetime(arr, dtype)

    # GH#15832: Check if we are requesting a numeric dtype and
    # that we can convert the data to the requested dtype.
    elif dtype.kind in "iu":
        # this will raise if we have e.g. floats

        subarr = maybe_cast_to_integer_array(arr, dtype)
    elif not copy:
        subarr = np.asarray(arr, dtype=dtype)
    else:
        subarr = np.array(arr, dtype=dtype, copy=copy)

    return subarr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\handlers\sets.py ===
"""
Handlers for predicates related to set membership: integer, rational, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Mul, Pow, S
from sympy.core.numbers import (AlgebraicNumber, ComplexInfinity, Exp1, Float,
    GoldenRatio, ImaginaryUnit, Infinity, Integer, NaN, NegativeInfinity,
    Number, NumberSymbol, Pi, pi, Rational, TribonacciConstant, E)
from sympy.core.logic import fuzzy_bool
from sympy.functions import (Abs, acos, acot, asin, atan, cos, cot, exp, im,
    log, re, sin, tan)
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.functions.elementary.complexes import conjugate
from sympy.matrices import Determinant, MatrixBase, Trace
from sympy.matrices.expressions.matexpr import MatrixElement

from sympy.multipledispatch import MDNotImplementedError

from .common import test_closed_group, ask_all, ask_any
from ..predicates.sets import (IntegerPredicate, RationalPredicate,
    IrrationalPredicate, RealPredicate, ExtendedRealPredicate,
    HermitianPredicate, ComplexPredicate, ImaginaryPredicate,
    AntihermitianPredicate, AlgebraicPredicate)


# IntegerPredicate

def _IntegerPredicate_number(expr, assumptions):
    # helper function
        try:
            i = int(expr.round())
            if not (expr - i).equals(0):
                raise TypeError
            return True
        except TypeError:
            return False

@IntegerPredicate.register_many(int, Integer) # type:ignore
def _(expr, assumptions):
    return True

@IntegerPredicate.register_many(Exp1, GoldenRatio, ImaginaryUnit, Infinity,
        NegativeInfinity, Pi, Rational, TribonacciConstant)
def _(expr, assumptions):
    return False

@IntegerPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_integer
    if ret is None:
        raise MDNotImplementedError
    return ret

@IntegerPredicate.register(Add)
def _(expr, assumptions):
    """
    * Integer + Integer       -> Integer
    * Integer + !Integer      -> !Integer
    * !Integer + !Integer -> ?
    """
    if expr.is_number:
        return _IntegerPredicate_number(expr, assumptions)
    return test_closed_group(expr, assumptions, Q.integer)

@IntegerPredicate.register(Pow)
def _(expr,assumptions):
    if expr.is_number:
        return _IntegerPredicate_number(expr, assumptions)
    if ask_all(~Q.zero(expr.base), Q.finite(expr.base), Q.zero(expr.exp), assumptions=assumptions):
        return True
    if ask_all(Q.integer(expr.base), Q.integer(expr.exp), assumptions=assumptions):
        if ask_any(Q.positive(expr.exp), Q.nonnegative(expr.exp) & ~Q.zero(expr.base), Q.zero(expr.base-1), Q.zero(expr.base+1), assumptions=assumptions):
            return True

@IntegerPredicate.register(Mul)
def _(expr, assumptions):
    """
    * Integer*Integer      -> Integer
    * Integer*Irrational   -> !Integer
    * Odd/Even             -> !Integer
    * Integer*Rational     -> ?
    """
    if expr.is_number:
        return _IntegerPredicate_number(expr, assumptions)
    _output = True
    for arg in expr.args:
        if not ask(Q.integer(arg), assumptions):
            if arg.is_Rational:
                if arg.q == 2:
                    return ask(Q.even(2*expr), assumptions)
                if ~(arg.q & 1):
                    return None
            elif ask(Q.irrational(arg), assumptions):
                if _output:
                    _output = False
                else:
                    return
            else:
                return

    return _output

@IntegerPredicate.register(Abs)
def _(expr, assumptions):
    if ask(Q.integer(expr.args[0]), assumptions):
        return True

@IntegerPredicate.register_many(Determinant, MatrixElement, Trace)
def _(expr, assumptions):
    return ask(Q.integer_elements(expr.args[0]), assumptions)


# RationalPredicate

@RationalPredicate.register(Rational)
def _(expr, assumptions):
    return True

@RationalPredicate.register(Float)
def _(expr, assumptions):
    return None

@RationalPredicate.register_many(Exp1, GoldenRatio, ImaginaryUnit, Infinity,
    NegativeInfinity, Pi, TribonacciConstant)
def _(expr, assumptions):
    return False

@RationalPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_rational
    if ret is None:
        raise MDNotImplementedError
    return ret

@RationalPredicate.register_many(Add, Mul)
def _(expr, assumptions):
    """
    * Rational + Rational     -> Rational
    * Rational + !Rational    -> !Rational
    * !Rational + !Rational   -> ?
    """
    if expr.is_number:
        if expr.as_real_imag()[1]:
            return False
    return test_closed_group(expr, assumptions, Q.rational)

@RationalPredicate.register(Pow)
def _(expr, assumptions):
    """
    * Rational ** Integer      -> Rational
    * Irrational ** Rational   -> Irrational
    * Rational ** Irrational   -> ?
    """
    if expr.base == E:
        x = expr.exp
        if ask(Q.rational(x), assumptions):
            return ask(Q.zero(x), assumptions)
        return

    is_exp_integer = ask(Q.integer(expr.exp), assumptions)
    if is_exp_integer:
        is_base_rational = ask(Q.rational(expr.base),assumptions)
        if is_base_rational:
            is_base_zero = ask(Q.zero(expr.base),assumptions)
            if is_base_zero is False:
                return True
            if is_base_zero and ask(Q.positive(expr.exp)):
                return True
        if ask(Q.algebraic(expr.base),assumptions) is False:
            return ask(Q.zero(expr.exp), assumptions)
        if ask(Q.irrational(expr.base),assumptions) and ask(Q.eq(expr.exp,-1)):
            return False
        return
    elif ask(Q.rational(expr.exp), assumptions):
        if ask(Q.prime(expr.base), assumptions) and is_exp_integer is False:
            return False
        if ask(Q.zero(expr.base)) and ask(Q.positive(expr.exp)):
            return True
        if ask(Q.eq(expr.base,1)):
            return True

@RationalPredicate.register_many(asin, atan, cos, sin, tan)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@RationalPredicate.register(exp)
def _(expr, assumptions):
    x = expr.exp
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@RationalPredicate.register_many(acot, cot)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return False

@RationalPredicate.register_many(acos, log)
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.rational(x), assumptions):
        return ask(~Q.nonzero(x - 1), assumptions)


# IrrationalPredicate

@IrrationalPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_irrational
    if ret is None:
        raise MDNotImplementedError
    return ret

@IrrationalPredicate.register(Basic)
def _(expr, assumptions):
    _real = ask(Q.real(expr), assumptions)
    if _real:
        _rational = ask(Q.rational(expr), assumptions)
        if _rational is None:
            return None
        return not _rational
    else:
        return _real


# RealPredicate

def _RealPredicate_number(expr, assumptions):
    # let as_real_imag() work first since the expression may
    # be simpler to evaluate
    i = expr.as_real_imag()[1].evalf(2)
    if i._prec != 1:
        return not i
    # allow None to be returned if we couldn't show for sure
    # that i was 0

@RealPredicate.register_many(Abs, Exp1, Float, GoldenRatio, im, Pi, Rational,
    re, TribonacciConstant)
def _(expr, assumptions):
    return True

@RealPredicate.register_many(ImaginaryUnit, Infinity, NegativeInfinity)
def _(expr, assumptions):
    return False

@RealPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_real
    if ret is None:
        raise MDNotImplementedError
    return ret

@RealPredicate.register(Add)
def _(expr, assumptions):
    """
    * Real + Real              -> Real
    * Real + (Complex & !Real) -> !Real
    """
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)
    return test_closed_group(expr, assumptions, Q.real)

@RealPredicate.register(Mul)
def _(expr, assumptions):
    """
    * Real*Real               -> Real
    * Real*Imaginary          -> !Real
    * Imaginary*Imaginary     -> Real
    """
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)
    result = True
    for arg in expr.args:
        if ask(Q.real(arg), assumptions):
            pass
        elif ask(Q.imaginary(arg), assumptions):
            result = result ^ True
        else:
            break
    else:
        return result

@RealPredicate.register(Pow)
def _(expr, assumptions):
    """
    * Real**Integer              -> Real
    * Positive**Real             -> Real
    * Negative**Real             -> ?
    * Real**(Integer/Even)       -> Real if base is nonnegative
    * Real**(Integer/Odd)        -> Real
    * Imaginary**(Integer/Even)  -> Real
    * Imaginary**(Integer/Odd)   -> not Real
    * Imaginary**Real            -> ? since Real could be 0 (giving real)
                                    or 1 (giving imaginary)
    * b**Imaginary               -> Real if log(b) is imaginary and b != 0
                                    and exponent != integer multiple of
                                    I*pi/log(b)
    * Real**Real                 -> ? e.g. sqrt(-1) is imaginary and
                                    sqrt(2) is not
    """
    if expr.is_number:
        return _RealPredicate_number(expr, assumptions)

    if expr.base == E:
        return ask(
            Q.integer(expr.exp/I/pi) | Q.real(expr.exp), assumptions
        )

    if expr.base.func == exp or (expr.base.is_Pow and expr.base.base == E):
        if ask(Q.imaginary(expr.base.exp), assumptions):
            if ask(Q.imaginary(expr.exp), assumptions):
                return True
        # If the i = (exp's arg)/(I*pi) is an integer or half-integer
        # multiple of I*pi then 2*i will be an integer. In addition,
        # exp(i*I*pi) = (-1)**i so the overall realness of the expr
        # can be determined by replacing exp(i*I*pi) with (-1)**i.
        i = expr.base.exp/I/pi
        if ask(Q.integer(2*i), assumptions):
            return ask(Q.real((S.NegativeOne**i)**expr.exp), assumptions)
        return

    if ask(Q.imaginary(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            odd = ask(Q.odd(expr.exp), assumptions)
            if odd is not None:
                return not odd
            return

    if ask(Q.imaginary(expr.exp), assumptions):
        imlog = ask(Q.imaginary(log(expr.base)), assumptions)
        if imlog is not None:
            # I**i -> real, log(I) is imag;
            # (2*I)**i -> complex, log(2*I) is not imag
            return imlog

    if ask(Q.real(expr.base), assumptions):
        if ask(Q.real(expr.exp), assumptions):
            if ask(Q.zero(expr.base), assumptions) is not False:
                if ask(Q.positive(expr.exp), assumptions):
                    return True
                return
            if expr.exp.is_Rational and \
                    ask(Q.even(expr.exp.q), assumptions):
                return ask(Q.positive(expr.base), assumptions)
            elif ask(Q.integer(expr.exp), assumptions):
                return True
            elif ask(Q.positive(expr.base), assumptions):
                return True

@RealPredicate.register_many(cos, sin)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
            return True

@RealPredicate.register(exp)
def _(expr, assumptions):
    return ask(
        Q.integer(expr.exp/I/pi) | Q.real(expr.exp), assumptions
    )

@RealPredicate.register(log)
def _(expr, assumptions):
    return ask(Q.positive(expr.args[0]), assumptions)

@RealPredicate.register_many(Determinant, MatrixElement, Trace)
def _(expr, assumptions):
    return ask(Q.real_elements(expr.args[0]), assumptions)


# ExtendedRealPredicate

@ExtendedRealPredicate.register(object)
def _(expr, assumptions):
    return ask(Q.negative_infinite(expr)
               | Q.negative(expr)
               | Q.zero(expr)
               | Q.positive(expr)
               | Q.positive_infinite(expr),
            assumptions)

@ExtendedRealPredicate.register_many(Infinity, NegativeInfinity)
def _(expr, assumptions):
    return True

@ExtendedRealPredicate.register_many(Add, Mul, Pow) # type:ignore
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.extended_real)


# HermitianPredicate

@HermitianPredicate.register(object) # type:ignore
def _(expr, assumptions):
    if isinstance(expr, MatrixBase):
        return None
    return ask(Q.real(expr), assumptions)

@HermitianPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian + Hermitian  -> Hermitian
    * Hermitian + !Hermitian -> !Hermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    return test_closed_group(expr, assumptions, Q.hermitian)

@HermitianPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    As long as there is at most only one noncommutative term:

    * Hermitian*Hermitian         -> Hermitian
    * Hermitian*Antihermitian     -> !Hermitian
    * Antihermitian*Antihermitian -> Hermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    nccount = 0
    result = True
    for arg in expr.args:
        if ask(Q.antihermitian(arg), assumptions):
            result = result ^ True
        elif not ask(Q.hermitian(arg), assumptions):
            break
        if ask(~Q.commutative(arg), assumptions):
            nccount += 1
            if nccount > 1:
                break
    else:
        return result

@HermitianPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian**Integer -> Hermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    if expr.base == E:
        if ask(Q.hermitian(expr.exp), assumptions):
            return True
        raise MDNotImplementedError
    if ask(Q.hermitian(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            return True
    raise MDNotImplementedError

@HermitianPredicate.register_many(cos, sin) # type:ignore
def _(expr, assumptions):
    if ask(Q.hermitian(expr.args[0]), assumptions):
        return True
    raise MDNotImplementedError

@HermitianPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    if ask(Q.hermitian(expr.exp), assumptions):
        return True
    raise MDNotImplementedError

@HermitianPredicate.register(MatrixBase) # type:ignore
def _(mat, assumptions):
    rows, cols = mat.shape
    ret_val = True
    for i in range(rows):
        for j in range(i, cols):
            cond = fuzzy_bool(Eq(mat[i, j], conjugate(mat[j, i])))
            if cond is None:
                ret_val = None
            if cond == False:
                return False
    if ret_val is None:
        raise MDNotImplementedError
    return ret_val


# ComplexPredicate

@ComplexPredicate.register_many(Abs, cos, exp, im, ImaginaryUnit, log, Number, # type:ignore
    NumberSymbol, re, sin)
def _(expr, assumptions):
    return True

@ComplexPredicate.register_many(Infinity, NegativeInfinity) # type:ignore
def _(expr, assumptions):
    return False

@ComplexPredicate.register(Expr) # type:ignore
def _(expr, assumptions):
    ret = expr.is_complex
    if ret is None:
        raise MDNotImplementedError
    return ret

@ComplexPredicate.register_many(Add, Mul) # type:ignore
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.complex)

@ComplexPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    if expr.base == E:
        return True
    return test_closed_group(expr, assumptions, Q.complex)

@ComplexPredicate.register_many(Determinant, MatrixElement, Trace) # type:ignore
def _(expr, assumptions):
    return ask(Q.complex_elements(expr.args[0]), assumptions)

@ComplexPredicate.register(NaN) # type:ignore
def _(expr, assumptions):
    return None


# ImaginaryPredicate

def _Imaginary_number(expr, assumptions):
    # let as_real_imag() work first since the expression may
    # be simpler to evaluate
    r = expr.as_real_imag()[0].evalf(2)
    if r._prec != 1:
        return not r
    # allow None to be returned if we couldn't show for sure
    # that r was 0

@ImaginaryPredicate.register(ImaginaryUnit) # type:ignore
def _(expr, assumptions):
    return True

@ImaginaryPredicate.register(Expr) # type:ignore
def _(expr, assumptions):
    ret = expr.is_imaginary
    if ret is None:
        raise MDNotImplementedError
    return ret

@ImaginaryPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Imaginary + Imaginary -> Imaginary
    * Imaginary + Complex   -> ?
    * Imaginary + Real      -> !Imaginary
    """
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)

    reals = 0
    for arg in expr.args:
        if ask(Q.imaginary(arg), assumptions):
            pass
        elif ask(Q.real(arg), assumptions):
            reals += 1
        else:
            break
    else:
        if reals == 0:
            return True
        if reals in (1, len(expr.args)):
            # two reals could sum 0 thus giving an imaginary
            return False

@ImaginaryPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    * Real*Imaginary      -> Imaginary
    * Imaginary*Imaginary -> Real
    """
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)
    result = False
    reals = 0
    for arg in expr.args:
        if ask(Q.imaginary(arg), assumptions):
            result = result ^ True
        elif not ask(Q.real(arg), assumptions):
            break
    else:
        if reals == len(expr.args):
            return False
        return result

@ImaginaryPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Imaginary**Odd        -> Imaginary
    * Imaginary**Even       -> Real
    * b**Imaginary          -> !Imaginary if exponent is an integer
                               multiple of I*pi/log(b)
    * Imaginary**Real       -> ?
    * Positive**Real        -> Real
    * Negative**Integer     -> Real
    * Negative**(Integer/2) -> Imaginary
    * Negative**Real        -> not Imaginary if exponent is not Rational
    """
    if expr.is_number:
        return _Imaginary_number(expr, assumptions)

    if expr.base == E:
        a = expr.exp/I/pi
        return ask(Q.integer(2*a) & ~Q.integer(a), assumptions)

    if expr.base.func == exp or (expr.base.is_Pow and expr.base.base == E):
        if ask(Q.imaginary(expr.base.exp), assumptions):
            if ask(Q.imaginary(expr.exp), assumptions):
                return False
            i = expr.base.exp/I/pi
            if ask(Q.integer(2*i), assumptions):
                return ask(Q.imaginary((S.NegativeOne**i)**expr.exp), assumptions)

    if ask(Q.imaginary(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            odd = ask(Q.odd(expr.exp), assumptions)
            if odd is not None:
                return odd
            return

    if ask(Q.imaginary(expr.exp), assumptions):
        imlog = ask(Q.imaginary(log(expr.base)), assumptions)
        if imlog is not None:
            # I**i -> real; (2*I)**i -> complex ==> not imaginary
            return False

    if ask(Q.real(expr.base) & Q.real(expr.exp), assumptions):
        if ask(Q.positive(expr.base), assumptions):
            return False
        else:
            rat = ask(Q.rational(expr.exp), assumptions)
            if not rat:
                return rat
            if ask(Q.integer(expr.exp), assumptions):
                return False
            else:
                half = ask(Q.integer(2*expr.exp), assumptions)
                if half:
                    return ask(Q.negative(expr.base), assumptions)
                return half

@ImaginaryPredicate.register(log) # type:ignore
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        if ask(Q.positive(expr.args[0]), assumptions):
            return False
        return
    # XXX it should be enough to do
    # return ask(Q.nonpositive(expr.args[0]), assumptions)
    # but ask(Q.nonpositive(exp(x)), Q.imaginary(x)) -> None;
    # it should return True since exp(x) will be either 0 or complex
    if expr.args[0].func == exp or (expr.args[0].is_Pow and expr.args[0].base == E):
        if expr.args[0].exp in [I, -I]:
            return True
    im = ask(Q.imaginary(expr.args[0]), assumptions)
    if im is False:
        return False

@ImaginaryPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    a = expr.exp/I/pi
    return ask(Q.integer(2*a) & ~Q.integer(a), assumptions)

@ImaginaryPredicate.register_many(Number, NumberSymbol) # type:ignore
def _(expr, assumptions):
    return not (expr.as_real_imag()[1] == 0)

@ImaginaryPredicate.register(NaN) # type:ignore
def _(expr, assumptions):
    return None


# AntihermitianPredicate

@AntihermitianPredicate.register(object) # type:ignore
def _(expr, assumptions):
    if isinstance(expr, MatrixBase):
        return None
    if ask(Q.zero(expr), assumptions):
        return True
    return ask(Q.imaginary(expr), assumptions)

@AntihermitianPredicate.register(Add) # type:ignore
def _(expr, assumptions):
    """
    * Antihermitian + Antihermitian  -> Antihermitian
    * Antihermitian + !Antihermitian -> !Antihermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    return test_closed_group(expr, assumptions, Q.antihermitian)

@AntihermitianPredicate.register(Mul) # type:ignore
def _(expr, assumptions):
    """
    As long as there is at most only one noncommutative term:

    * Hermitian*Hermitian         -> !Antihermitian
    * Hermitian*Antihermitian     -> Antihermitian
    * Antihermitian*Antihermitian -> !Antihermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    nccount = 0
    result = False
    for arg in expr.args:
        if ask(Q.antihermitian(arg), assumptions):
            result = result ^ True
        elif not ask(Q.hermitian(arg), assumptions):
            break
        if ask(~Q.commutative(arg), assumptions):
            nccount += 1
            if nccount > 1:
                break
    else:
        return result

@AntihermitianPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    """
    * Hermitian**Integer  -> !Antihermitian
    * Antihermitian**Even -> !Antihermitian
    * Antihermitian**Odd  -> Antihermitian
    """
    if expr.is_number:
        raise MDNotImplementedError
    if ask(Q.hermitian(expr.base), assumptions):
        if ask(Q.integer(expr.exp), assumptions):
            return False
    elif ask(Q.antihermitian(expr.base), assumptions):
        if ask(Q.even(expr.exp), assumptions):
            return False
        elif ask(Q.odd(expr.exp), assumptions):
            return True
    raise MDNotImplementedError

@AntihermitianPredicate.register(MatrixBase) # type:ignore
def _(mat, assumptions):
    rows, cols = mat.shape
    ret_val = True
    for i in range(rows):
        for j in range(i, cols):
            cond = fuzzy_bool(Eq(mat[i, j], -conjugate(mat[j, i])))
            if cond is None:
                ret_val = None
            if cond == False:
                return False
    if ret_val is None:
        raise MDNotImplementedError
    return ret_val


# AlgebraicPredicate

@AlgebraicPredicate.register_many(AlgebraicNumber, Float, GoldenRatio, # type:ignore
    ImaginaryUnit, TribonacciConstant)
def _(expr, assumptions):
    return True

@AlgebraicPredicate.register_many(ComplexInfinity, Exp1, Infinity, # type:ignore
    NegativeInfinity, Pi)
def _(expr, assumptions):
    return False

@AlgebraicPredicate.register_many(Add, Mul) # type:ignore
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.algebraic)

@AlgebraicPredicate.register(Pow) # type:ignore
def _(expr, assumptions):
    if expr.base == E:
        if ask(Q.algebraic(expr.exp), assumptions):
            return ask(~Q.nonzero(expr.exp), assumptions)
        return
    if expr.base == pi:
        if ask(Q.integer(expr.exp), assumptions) and ask(Q.positive(expr.exp), assumptions):
            return False
        return
    exp_rational = ask(Q.rational(expr.exp), assumptions)
    base_algebraic = ask(Q.algebraic(expr.base), assumptions)
    exp_algebraic = ask(Q.algebraic(expr.exp),assumptions)
    if base_algebraic and exp_algebraic:
        if exp_rational:
            return True
        # Check based on the Gelfond-Schneider theorem:
        # If the base is algebraic and not equal to 0 or 1, and the exponent
        # is irrational,then the result is transcendental.
        if ask(Q.ne(expr.base,0) & Q.ne(expr.base,1)) and exp_rational is False:
            return False

@AlgebraicPredicate.register(Rational) # type:ignore
def _(expr, assumptions):
    return expr.q != 0

@AlgebraicPredicate.register_many(asin, atan, cos, sin, tan) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@AlgebraicPredicate.register(exp) # type:ignore
def _(expr, assumptions):
    x = expr.exp
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x), assumptions)

@AlgebraicPredicate.register_many(acot, cot) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return False

@AlgebraicPredicate.register_many(acos, log) # type:ignore
def _(expr, assumptions):
    x = expr.args[0]
    if ask(Q.algebraic(x), assumptions):
        return ask(~Q.nonzero(x - 1), assumptions)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\continuum_mechanics\cable.py ===
"""
This module can be used to solve problems related
to 2D Cables.
"""

from sympy.core.sympify import sympify
from sympy.core.symbol import Symbol,symbols
from sympy import sin, cos, pi, atan, diff, Piecewise, solve, rad
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.solvers.solveset import linsolve
from sympy.matrices import Matrix
from sympy.plotting import plot

class Cable:
    """
    Cables are structures in engineering that support
    the applied transverse loads through the tensile
    resistance developed in its members.

    Cables are widely used in suspension bridges, tension
    leg offshore platforms, transmission lines, and find
    use in several other engineering applications.

    Examples
    ========
    A cable is supported at (0, 10) and (10, 10). Two point loads
    acting vertically downwards act on the cable, one with magnitude 3 kN
    and acting 2 meters from the left support and 3 meters below it, while
    the other with magnitude 2 kN is 6 meters from the left support and
    6 meters below it.

    >>> from sympy.physics.continuum_mechanics.cable import Cable
    >>> c = Cable(('A', 0, 10), ('B', 10, 10))
    >>> c.apply_load(-1, ('P', 2, 7, 3, 270))
    >>> c.apply_load(-1, ('Q', 6, 4, 2, 270))
    >>> c.loads
    {'distributed': {}, 'point_load': {'P': [3, 270], 'Q': [2, 270]}}
    >>> c.loads_position
    {'P': [2, 7], 'Q': [6, 4]}
    """
    def __init__(self, support_1, support_2):
        """
        Initializes the class.

        Parameters
        ==========

        support_1 and support_2 are tuples of the form
        (label, x, y), where

        label : String or symbol
            The label of the support

        x : Sympifyable
            The x coordinate of the position of the support

        y : Sympifyable
            The y coordinate of the position of the support
        """
        self._left_support = []
        self._right_support = []
        self._supports = {}
        self._support_labels = []
        self._loads = {"distributed": {}, "point_load": {}}
        self._loads_position = {}
        self._length = 0
        self._reaction_loads = {}
        self._tension = {}
        self._lowest_x_global = sympify(0)
        self._lowest_y_global = sympify(0)
        self._cable_eqn = None
        self._tension_func = None
        if support_1[0] == support_2[0]:
            raise ValueError("Supports can not have the same label")

        elif support_1[1] == support_2[1]:
            raise ValueError("Supports can not be at the same location")

        x1 = sympify(support_1[1])
        y1 = sympify(support_1[2])
        self._supports[support_1[0]] = [x1, y1]

        x2 = sympify(support_2[1])
        y2 = sympify(support_2[2])
        self._supports[support_2[0]] = [x2, y2]

        if support_1[1] < support_2[1]:
            self._left_support.append(x1)
            self._left_support.append(y1)
            self._right_support.append(x2)
            self._right_support.append(y2)
            self._support_labels.append(support_1[0])
            self._support_labels.append(support_2[0])

        else:
            self._left_support.append(x2)
            self._left_support.append(y2)
            self._right_support.append(x1)
            self._right_support.append(y1)
            self._support_labels.append(support_2[0])
            self._support_labels.append(support_1[0])

        for i in self._support_labels:
            self._reaction_loads[Symbol("R_"+ i +"_x")] = 0
            self._reaction_loads[Symbol("R_"+ i +"_y")] = 0

    @property
    def supports(self):
        """
        Returns the supports of the cable along with their
        positions.
        """
        return self._supports

    @property
    def left_support(self):
        """
        Returns the position of the left support.
        """
        return self._left_support

    @property
    def right_support(self):
        """
        Returns the position of the right support.
        """
        return self._right_support

    @property
    def loads(self):
        """
        Returns the magnitude and direction of the loads
        acting on the cable.
        """
        return self._loads

    @property
    def loads_position(self):
        """
        Returns the position of the point loads acting on the
        cable.
        """
        return self._loads_position

    @property
    def length(self):
        """
        Returns the length of the cable.
        """
        return self._length

    @property
    def reaction_loads(self):
        """
        Returns the reaction forces at the supports, which are
        initialized to 0.
        """
        return self._reaction_loads

    @property
    def tension(self):
        """
        Returns the tension developed in the cable due to the loads
        applied.
        """
        return self._tension

    def tension_at(self, x):
        """
        Returns the tension at a given value of x developed due to
        distributed load.
        """
        if 'distributed' not in self._tension.keys():
            raise ValueError("No distributed load added or solve method not called")

        if x > self._right_support[0] or x < self._left_support[0]:
            raise ValueError("The value of x should be between the two supports")

        A = self._tension['distributed']
        X = Symbol('X')

        return A.subs({X:(x-self._lowest_x_global)})

    def apply_length(self, length):
        """
        This method specifies the length of the cable

        Parameters
        ==========

        length : Sympifyable
            The length of the cable

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_length(20)
        >>> c.length
        20
        """
        dist = ((self._left_support[0] - self._right_support[0])**2
                - (self._left_support[1] - self._right_support[1])**2)**(1/2)

        if length < dist:
            raise ValueError("length should not be less than the distance between the supports")

        self._length = length

    def change_support(self, label, new_support):
        """
        This method changes the mentioned support with a new support.

        Parameters
        ==========
        label: String or symbol
            The label of the support to be changed

        new_support: Tuple of the form (new_label, x, y)
            new_label: String or symbol
                The label of the new support

            x: Sympifyable
                The x-coordinate of the position of the new support.

            y: Sympifyable
                The y-coordinate of the position of the new support.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.supports
        {'A': [0, 10], 'B': [10, 10]}
        >>> c.change_support('B', ('C', 5, 6))
        >>> c.supports
        {'A': [0, 10], 'C': [5, 6]}
        """
        if label not in self._supports:
            raise ValueError("No support exists with the given label")

        i = self._support_labels.index(label)
        rem_label = self._support_labels[(i+1)%2]
        x1 = self._supports[rem_label][0]
        y1 = self._supports[rem_label][1]

        x = sympify(new_support[1])
        y = sympify(new_support[2])

        for l in self._loads_position:
            if l[0] >= max(x, x1) or l[0] <= min(x, x1):
                raise ValueError("The change in support will throw an existing load out of range")

        self._supports.pop(label)
        self._left_support.clear()
        self._right_support.clear()
        self._reaction_loads.clear()
        self._support_labels.remove(label)

        self._supports[new_support[0]] = [x, y]

        if x1 < x:
            self._left_support.append(x1)
            self._left_support.append(y1)
            self._right_support.append(x)
            self._right_support.append(y)
            self._support_labels.append(new_support[0])

        else:
            self._left_support.append(x)
            self._left_support.append(y)
            self._right_support.append(x1)
            self._right_support.append(y1)
            self._support_labels.insert(0, new_support[0])

        for i in self._support_labels:
            self._reaction_loads[Symbol("R_"+ i +"_x")] = 0
            self._reaction_loads[Symbol("R_"+ i +"_y")] = 0

    def apply_load(self, order, load):
        """
        This method adds load to the cable.

        Parameters
        ==========

        order : Integer
            The order of the applied load.

                - For point loads, order = -1
                - For distributed load, order = 0

        load : tuple

            * For point loads, load is of the form (label, x, y, magnitude, direction), where:

            label : String or symbol
                The label of the load

            x : Sympifyable
                The x coordinate of the position of the load

            y : Sympifyable
                The y coordinate of the position of the load

            magnitude : Sympifyable
                The magnitude of the load. It must always be positive

            direction : Sympifyable
                The angle, in degrees, that the load vector makes with the horizontal
                in the counter-clockwise direction. It takes the values 0 to 360,
                inclusive.


            * For uniformly distributed load, load is of the form (label, magnitude)

            label : String or symbol
                The label of the load

            magnitude : Sympifyable
                The magnitude of the load. It must always be positive

        Examples
        ========

        For a point load of magnitude 12 units inclined at 30 degrees with the horizontal:

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))
        >>> c.loads
        {'distributed': {}, 'point_load': {'Z': [12, 30]}}
        >>> c.loads_position
        {'Z': [5, 5]}


        For a uniformly distributed load of magnitude 9 units:

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(0, ('X', 9))
        >>> c.loads
        {'distributed': {'X': 9}, 'point_load': {}}
        """
        if order == -1:
            if len(self._loads["distributed"]) != 0:
                raise ValueError("Distributed load already exists")

            label = load[0]
            if label in self._loads["point_load"]:
                raise ValueError("Label already exists")

            x = sympify(load[1])
            y = sympify(load[2])

            if x > self._right_support[0] or x < self._left_support[0]:
                raise ValueError("The load should be positioned between the supports")

            magnitude = sympify(load[3])
            direction = sympify(load[4])

            self._loads["point_load"][label] = [magnitude, direction]
            self._loads_position[label] = [x, y]

        elif order == 0:
            if len(self._loads_position) != 0:
                raise ValueError("Point load(s) already exist")

            label = load[0]
            if label in self._loads["distributed"]:
                raise ValueError("Label already exists")

            magnitude = sympify(load[1])

            self._loads["distributed"][label] = magnitude

        else:
            raise ValueError("Order should be either -1 or 0")

    def remove_loads(self, *args):
        """
        This methods removes the specified loads.

        Parameters
        ==========
        This input takes multiple label(s) as input
        label(s): String or symbol
            The label(s) of the loads to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))
        >>> c.loads
        {'distributed': {}, 'point_load': {'Z': [12, 30]}}
        >>> c.remove_loads('Z')
        >>> c.loads
        {'distributed': {}, 'point_load': {}}
        """
        for i in args:
            if len(self._loads_position) == 0:
                if i not in self._loads['distributed']:
                    raise ValueError("Error removing load " + i + ": no such load exists")

                else:
                    self._loads['disrtibuted'].pop(i)

            else:
                if i not in self._loads['point_load']:
                    raise ValueError("Error removing load " + i + ": no such load exists")

                else:
                    self._loads['point_load'].pop(i)
                    self._loads_position.pop(i)

    def solve(self, *args):
        """
        This method solves for the reaction forces at the supports, the tension developed in
        the cable, and updates the length of the cable.

        Parameters
        ==========
        This method requires no input when solving for point loads
        For distributed load, the x and y coordinates of the lowest point of the cable are
        required as

        x: Sympifyable
            The x coordinate of the lowest point

        y: Sympifyable
            The y coordinate of the lowest point

        Examples
        ========
        For point loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(("A", 0, 10), ("B", 10, 10))
        >>> c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
        >>> c.apply_load(-1, ('X', 4, 6, 8, 270))
        >>> c.solve()
        >>> c.tension
        {A_Z: 8.91403453669861, X_B: 19*sqrt(13)/10, Z_X: 4.79150773600774}
        >>> c.reaction_loads
        {R_A_x: -5.25547445255474, R_A_y: 7.2, R_B_x: 5.25547445255474, R_B_y: 3.8}
        >>> c.length
        5.7560958484519 + 2*sqrt(13)

        For distributed load,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c=Cable(("A", 0, 40),("B", 100, 20))
        >>> c.apply_load(0, ("X", 850))
        >>> c.solve(58.58)
        >>> c.tension
        {'distributed': 36465.0*sqrt(0.00054335718671383*X**2 + 1)}
        >>> c.tension_at(0)
        61717.4130533677
        >>> c.reaction_loads
        {R_A_x: 36465.0, R_A_y: -49793.0, R_B_x: 44399.9537590861, R_B_y: 42868.2071025955}
        """

        if len(self._loads_position) != 0:
            sorted_position = sorted(self._loads_position.items(), key = lambda item : item[1][0])

            sorted_position.append(self._support_labels[1])
            sorted_position.insert(0, self._support_labels[0])

            self._tension.clear()
            moment_sum_from_left_support = 0
            moment_sum_from_right_support = 0
            F_x = 0
            F_y = 0
            self._length = 0
            tension_func = []
            x = symbols('x')
            for i in range(1, len(sorted_position)-1):
                if i == 1:
                    self._length+=sqrt((self._left_support[0] - self._loads_position[sorted_position[i][0]][0])**2 + (self._left_support[1] - self._loads_position[sorted_position[i][0]][1])**2)

                else:
                    self._length+=sqrt((self._loads_position[sorted_position[i-1][0]][0] - self._loads_position[sorted_position[i][0]][0])**2 + (self._loads_position[sorted_position[i-1][0]][1] - self._loads_position[sorted_position[i][0]][1])**2)

                if i == len(sorted_position)-2:
                    self._length+=sqrt((self._right_support[0] - self._loads_position[sorted_position[i][0]][0])**2 + (self._right_support[1] - self._loads_position[sorted_position[i][0]][1])**2)

                moment_sum_from_left_support += self._loads['point_load'][sorted_position[i][0]][0] * cos(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._left_support[1] - self._loads_position[sorted_position[i][0]][1])
                moment_sum_from_left_support += self._loads['point_load'][sorted_position[i][0]][0] * sin(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._left_support[0] - self._loads_position[sorted_position[i][0]][0])

                F_x += self._loads['point_load'][sorted_position[i][0]][0] * cos(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180)
                F_y += self._loads['point_load'][sorted_position[i][0]][0] * sin(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180)

                label = Symbol(sorted_position[i][0]+"_"+sorted_position[i+1][0])
                y2 = self._loads_position[sorted_position[i][0]][1]
                x2 = self._loads_position[sorted_position[i][0]][0]
                y1 = 0
                x1 = 0

                if i == len(sorted_position)-2:
                    x1 = self._right_support[0]
                    y1 = self._right_support[1]

                else:
                    x1 = self._loads_position[sorted_position[i+1][0]][0]
                    y1 = self._loads_position[sorted_position[i+1][0]][1]

                angle_with_horizontal = atan((y1 - y2)/(x1 - x2))

                tension = -(moment_sum_from_left_support)/(abs(self._left_support[1] - self._loads_position[sorted_position[i][0]][1])*cos(angle_with_horizontal) + abs(self._left_support[0] - self._loads_position[sorted_position[i][0]][0])*sin(angle_with_horizontal))
                self._tension[label] = tension
                tension_func.append((tension, x<=x1))
                moment_sum_from_right_support += self._loads['point_load'][sorted_position[i][0]][0] * cos(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._right_support[1] - self._loads_position[sorted_position[i][0]][1])
                moment_sum_from_right_support += self._loads['point_load'][sorted_position[i][0]][0] * sin(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._right_support[0] - self._loads_position[sorted_position[i][0]][0])

            label = Symbol(sorted_position[0][0]+"_"+sorted_position[1][0])
            y2 = self._loads_position[sorted_position[1][0]][1]
            x2 = self._loads_position[sorted_position[1][0]][0]
            x1 = self._left_support[0]
            y1 = self._left_support[1]

            angle_with_horizontal = -atan((y2 - y1)/(x2 - x1))
            tension = -(moment_sum_from_right_support)/(abs(self._right_support[1] - self._loads_position[sorted_position[1][0]][1])*cos(angle_with_horizontal) + abs(self._right_support[0] - self._loads_position[sorted_position[1][0]][0])*sin(angle_with_horizontal))
            self._tension[label] = tension

            tension_func.insert(0,(tension, x<=x2))
            self._tension_func = Piecewise(*tension_func)
            angle_with_horizontal = pi/2 - angle_with_horizontal
            label = self._support_labels[0]
            self._reaction_loads[Symbol("R_"+label+"_x")] = -sin(angle_with_horizontal) * tension
            F_x += -sin(angle_with_horizontal) * tension
            self._reaction_loads[Symbol("R_"+label+"_y")] = cos(angle_with_horizontal) * tension
            F_y += cos(angle_with_horizontal) * tension

            label = self._support_labels[1]
            self._reaction_loads[Symbol("R_"+label+"_x")] = -F_x
            self._reaction_loads[Symbol("R_"+label+"_y")] = -F_y

        elif len(self._loads['distributed']) != 0 :

            if len(args) == 0:
                raise ValueError("Provide the lowest point of the cable")

            lowest_x = sympify(args[0])
            self._lowest_x_global = lowest_x

            a = Symbol('a', positive=True)
            c = Symbol('c')
            # augmented matrix form of linsolve

            M = Matrix(
                [[(self._left_support[0]-lowest_x)**2, 1, self._left_support[1]],
                [(self._right_support[0]-lowest_x)**2, 1, self._right_support[1]],
                ])

            coefficient_solution = list(linsolve(M, (a, c)))
            if len(coefficient_solution) ==0 or coefficient_solution[0][0]== 0:
                raise ValueError("The lowest point is inconsistent with the supports")

            A = coefficient_solution[0][0]
            C = coefficient_solution[0][1] + coefficient_solution[0][0]*lowest_x**2
            B = -2*coefficient_solution[0][0]*lowest_x
            self._lowest_y_global = coefficient_solution[0][1]
            lowest_y = self._lowest_y_global

            # y = A*x**2 + B*x + C
            # shifting origin to lowest point
            X = Symbol('X')
            Y = Symbol('Y')
            Y = A*(X + lowest_x)**2 + B*(X + lowest_x) + C - lowest_y

            temp_list = list(self._loads['distributed'].values())
            applied_force = temp_list[0]

            horizontal_force_constant = (applied_force * (self._right_support[0] - lowest_x)**2) / (2 * (self._right_support[1] - lowest_y))

            self._tension.clear()
            tangent_slope_to_curve = diff(Y, X)
            self._tension['distributed'] = horizontal_force_constant / (cos(atan(tangent_slope_to_curve)))

            label = self._support_labels[0]
            self._reaction_loads[Symbol("R_"+label+"_x")] = self.tension_at(self._left_support[0]) * cos(atan(tangent_slope_to_curve.subs(X, self._left_support[0] - lowest_x)))
            self._reaction_loads[Symbol("R_"+label+"_y")] = self.tension_at(self._left_support[0]) * sin(atan(tangent_slope_to_curve.subs(X, self._left_support[0] - lowest_x)))

            label = self._support_labels[1]
            self._reaction_loads[Symbol("R_"+label+"_x")] = self.tension_at(self._left_support[0]) * cos(atan(tangent_slope_to_curve.subs(X, self._right_support[0] - lowest_x)))
            self._reaction_loads[Symbol("R_"+label+"_y")] = self.tension_at(self._left_support[0]) * sin(atan(tangent_slope_to_curve.subs(X, self._right_support[0] - lowest_x)))

    def draw(self):
        """
        This method is used to obtain a plot for the specified cable with its supports,
        shape and loads.

        Examples
        ========

        For point loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(("A", 0, 10), ("B", 10, 10))
        >>> c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
        >>> c.apply_load(-1, ('X', 4, 6, 8, 270))
        >>> c.solve()
        >>> p = c.draw()
        >>> p  # doctest: +ELLIPSIS
        Plot object containing:
        [0]: cartesian line: Piecewise((10 - 1.37*x, x <= 2), (8.52 - 0.63*x, x <= 4), (2*x/3 + 10/3, x <= 10)) for x over (0.0, 10.0)
        ...
        >>> p.show()

        For uniformly distributed loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c=Cable(("A", 0, 40),("B", 100, 20))
        >>> c.apply_load(0, ("X", 850))
        >>> c.solve(58.58)
        >>> p = c.draw()
        >>> p # doctest: +ELLIPSIS
        Plot object containing:
        [0]: cartesian line: 0.0116550116550117*(x - 58.58)**2 + 0.00447086247086247 for x over (0.0, 100.0)
        [1]: cartesian line: -7.49552913752915 for x over (0.0, 100.0)
        ...
        >>> p.show()
        """
        x = Symbol("x")
        annotations = []
        support_rectangles = self._draw_supports()

        xy_min = min(self._left_support[0],self._lowest_y_global)
        xy_max = max(self._right_support[0], max(self._right_support[1],self._left_support[1]))
        max_diff = xy_max - xy_min
        if len(self._loads_position) != 0:
            self._cable_eqn = self._draw_cable(-1)
            annotations += self._draw_loads(-1)

        elif len(self._loads['distributed']) != 0 :
            self._cable_eqn = self._draw_cable(0)
            annotations += self._draw_loads(0)

        if not self._cable_eqn:
            raise ValueError("solve method not called and/or values provided for loads and supports not adequate")

        cab_plot = plot(*self._cable_eqn,(x,self._left_support[0],self._right_support[0]),
                        xlim=(xy_min-0.5*max_diff,xy_max+0.5*max_diff),
                        ylim=(xy_min-0.5*max_diff,xy_max+0.5*max_diff),
                        rectangles=support_rectangles,show= False,annotations=annotations, axis=False)

        return cab_plot

    def _draw_supports(self):
        member_rectangles = []
        xy_min = min(self._left_support[0],self._lowest_y_global)
        xy_max = max(self._right_support[0], max(self._right_support[1],self._left_support[1]))
        max_diff = xy_max - xy_min

        supp_width = 0.075*max_diff

        member_rectangles.append(
            {
                'xy': (self._left_support[0]-supp_width,self._left_support[1]),
                'width': supp_width,
                'height':supp_width,
                'color':'brown',
                'fill': False
            }
        )

        member_rectangles.append(
            {
                'xy': (self._right_support[0],self._right_support[1]),
                'width': supp_width,
                'height':supp_width,
                'color':'brown',
                'fill': False
            }
        )

        return member_rectangles

    def _draw_cable(self,order):
        xy_min = min(self._left_support[0],self._lowest_y_global)
        xy_max = max(self._right_support[0], max(self._right_support[1],self._left_support[1]))
        max_diff = xy_max - xy_min
        if order == -1 :
            x,y = symbols('x y')
            line_func = []
            sorted_position = sorted(self._loads_position.items(), key = lambda item : item[1][0])

            for i in range(len(sorted_position)):
                if(i==0):
                    y = ((sorted_position[i][1][1] - self._left_support[1])*(x-self._left_support[0]))/(sorted_position[i][1][0]- self._left_support[0]) + self._left_support[1]
                else:
                    y = ((sorted_position[i][1][1] - sorted_position[i-1][1][1] )*(x-sorted_position[i-1][1][0]))/(sorted_position[i][1][0]- sorted_position[i-1][1][0]) + sorted_position[i-1][1][1]
                line_func.append((y,x<=sorted_position[i][1][0]))

            y = ((sorted_position[len(sorted_position)-1][1][1] - self._right_support[1])*(x-self._right_support[0]))/(sorted_position[i][1][0]- self._right_support[0]) + self._right_support[1]
            line_func.append((y,x<=self._right_support[0]))
            return [Piecewise(*line_func)]

        elif order == 0:
            x0 = self._lowest_x_global
            diff_force_height = max_diff*0.075

            a,c,x,y = symbols('a c x y')
            parabola_eqn = a*(x-x0)**2 + c - y

            points = [(self._left_support[0],self._left_support[1]),(self._right_support[0],self._right_support[1])]
            equations = []
            for px, py in points:
                equations.append(parabola_eqn.subs({x: px, y: py}))
            solution = solve(equations, (a, c))
            parabola_eqn = solution[a]*(x-x0)**2 + solution[c]
            return [parabola_eqn, self._lowest_y_global - diff_force_height]

    def _draw_loads(self,order):
        xy_min = min(self._left_support[0],self._lowest_y_global)
        xy_max = max(self._right_support[0], max(self._right_support[1],self._left_support[1]))
        max_diff = xy_max - xy_min
        if(order==-1):
            arrow_length = max_diff*0.1
            force_arrows = []
            for key in self._loads['point_load']:
                force_arrows.append(
                    {
                        'text': '',
                        'xy':(self._loads_position[key][0]+arrow_length*cos(rad(self._loads['point_load'][key][1])),\
                              self._loads_position[key][1] + arrow_length*sin(rad(self._loads['point_load'][key][1]))),
                        'xytext': (self._loads_position[key][0],self._loads_position[key][1]),
                        'arrowprops': {'width': 1, 'headlength':3, 'headwidth':3 , 'facecolor': 'black', }
                    }
                )
                mag = self._loads['point_load'][key][0]
                force_arrows.append(
                    {
                        'text':f'{mag}N',
                        'xy': (self._loads_position[key][0]+arrow_length*1.6*cos(rad(self._loads['point_load'][key][1])),\
                               self._loads_position[key][1] + arrow_length*1.6*sin(rad(self._loads['point_load'][key][1]))),
                    }
                )
            return force_arrows

        elif (order == 0):
            x = symbols('x')
            force_arrows = []
            x_val = [self._left_support[0] + ((self._right_support[0]-self._left_support[0])/10)*i for i in range(1,10)]
            for i in x_val:
                force_arrows.append(
                    {
                        'text':'',
                        'xytext':(
                            i,
                            self._cable_eqn[0].subs(x,i)
                        ),
                        'xy':(
                            i,
                            self._cable_eqn[1].subs(x,i)
                        ),
                        'arrowprops':{'width':1, 'headlength':3.5, 'headwidth':3.5, 'facecolor':'black'}
                    }
                )
            mag = 0
            for key in self._loads['distributed']:
                mag += self._loads['distributed'][key]

            force_arrows.append(
                {
                    'text':f'{mag} N/m',
                    'xy':((self._left_support[0]+self._right_support[0])/2,self._lowest_y_global - max_diff*0.15)
                }
            )
            return force_arrows

    def plot_tension(self):
        """
        Returns the diagram/plot of the tension generated in the cable at various points.

        Examples
        ========

        For point loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(("A", 0, 10), ("B", 10, 10))
        >>> c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
        >>> c.apply_load(-1, ('X', 4, 6, 8, 270))
        >>> c.solve()
        >>> p = c.plot_tension()
        >>> p
        Plot object containing:
        [0]: cartesian line: Piecewise((8.91403453669861, x <= 2), (4.79150773600774, x <= 4), (19*sqrt(13)/10, x <= 10)) for x over (0.0, 10.0)
        >>> p.show()

        For uniformly distributed loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c=Cable(("A", 0, 40),("B", 100, 20))
        >>> c.apply_load(0, ("X", 850))
        >>> c.solve(58.58)
        >>> p = c.plot_tension()
        >>> p
        Plot object containing:
        [0]: cartesian line: 36465.0*sqrt(0.00054335718671383*X**2 + 1) for X over (0.0, 100.0)
        >>> p.show()

        """
        if len(self._loads_position) != 0:
            x = symbols('x')
            tension_plot  = plot(self._tension_func, (x,self._left_support[0],self._right_support[0]), show=False)
        else:
            X = symbols('X')
            tension_plot  = plot(self._tension['distributed'], (X,self._left_support[0],self._right_support[0]), show=False)
        return tension_plot

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sortedcontainers\sorteddict.py ===
"""Sorted Dict
==============

:doc:`Sorted Containers<index>` is an Apache2 licensed Python sorted
collections library, written in pure-Python, and fast as C-extensions. The
:doc:`introduction<introduction>` is the best way to get started.

Sorted dict implementations:

.. currentmodule:: sortedcontainers

* :class:`SortedDict`
* :class:`SortedKeysView`
* :class:`SortedItemsView`
* :class:`SortedValuesView`

"""

import sys
import warnings

from itertools import chain

from .sortedlist import SortedList, recursive_repr
from .sortedset import SortedSet

###############################################################################
# BEGIN Python 2/3 Shims
###############################################################################

try:
    from collections.abc import (
        ItemsView, KeysView, Mapping, ValuesView, Sequence
    )
except ImportError:
    from collections import ItemsView, KeysView, Mapping, ValuesView, Sequence

###############################################################################
# END Python 2/3 Shims
###############################################################################


class SortedDict(dict):
    """Sorted dict is a sorted mutable mapping.

    Sorted dict keys are maintained in sorted order. The design of sorted dict
    is simple: sorted dict inherits from dict to store items and maintains a
    sorted list of keys.

    Sorted dict keys must be hashable and comparable. The hash and total
    ordering of keys must not change while they are stored in the sorted dict.

    Mutable mapping methods:

    * :func:`SortedDict.__getitem__` (inherited from dict)
    * :func:`SortedDict.__setitem__`
    * :func:`SortedDict.__delitem__`
    * :func:`SortedDict.__iter__`
    * :func:`SortedDict.__len__` (inherited from dict)

    Methods for adding items:

    * :func:`SortedDict.setdefault`
    * :func:`SortedDict.update`

    Methods for removing items:

    * :func:`SortedDict.clear`
    * :func:`SortedDict.pop`
    * :func:`SortedDict.popitem`

    Methods for looking up items:

    * :func:`SortedDict.__contains__` (inherited from dict)
    * :func:`SortedDict.get` (inherited from dict)
    * :func:`SortedDict.peekitem`

    Methods for views:

    * :func:`SortedDict.keys`
    * :func:`SortedDict.items`
    * :func:`SortedDict.values`

    Methods for miscellany:

    * :func:`SortedDict.copy`
    * :func:`SortedDict.fromkeys`
    * :func:`SortedDict.__reversed__`
    * :func:`SortedDict.__eq__` (inherited from dict)
    * :func:`SortedDict.__ne__` (inherited from dict)
    * :func:`SortedDict.__repr__`
    * :func:`SortedDict._check`

    Sorted list methods available (applies to keys):

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.count`
    * :func:`SortedList.index`
    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList._reset`

    Additional sorted list methods available, if key-function used:

    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Sorted dicts may only be compared for equality and inequality.

    """
    def __init__(self, *args, **kwargs):
        """Initialize sorted dict instance.

        Optional key-function argument defines a callable that, like the `key`
        argument to the built-in `sorted` function, extracts a comparison key
        from each dictionary key. If no function is specified, the default
        compares the dictionary keys directly. The key-function argument must
        be provided as a positional argument and must come before all other
        arguments.

        Optional iterable argument provides an initial sequence of pairs to
        initialize the sorted dict. Each pair in the sequence defines the key
        and corresponding value. If a key is seen more than once, the last
        value associated with it is stored in the new sorted dict.

        Optional mapping argument provides an initial mapping of items to
        initialize the sorted dict.

        If keyword arguments are given, the keywords themselves, with their
        associated values, are added as items to the dictionary. If a key is
        specified both in the positional argument and as a keyword argument,
        the value associated with the keyword is stored in the
        sorted dict.

        Sorted dict keys must be hashable, per the requirement for Python's
        dictionaries. Keys (or the result of the key-function) must also be
        comparable, per the requirement for sorted lists.

        >>> d = {'alpha': 1, 'beta': 2}
        >>> SortedDict([('alpha', 1), ('beta', 2)]) == d
        True
        >>> SortedDict({'alpha': 1, 'beta': 2}) == d
        True
        >>> SortedDict(alpha=1, beta=2) == d
        True

        """
        if args and (args[0] is None or callable(args[0])):
            _key = self._key = args[0]
            args = args[1:]
        else:
            _key = self._key = None

        self._list = SortedList(key=_key)

        # Reaching through ``self._list`` repeatedly adds unnecessary overhead
        # so cache references to sorted list methods.

        _list = self._list
        self._list_add = _list.add
        self._list_clear = _list.clear
        self._list_iter = _list.__iter__
        self._list_reversed = _list.__reversed__
        self._list_pop = _list.pop
        self._list_remove = _list.remove
        self._list_update = _list.update

        # Expose some sorted list methods publicly.

        self.bisect_left = _list.bisect_left
        self.bisect = _list.bisect_right
        self.bisect_right = _list.bisect_right
        self.index = _list.index
        self.irange = _list.irange
        self.islice = _list.islice
        self._reset = _list._reset

        if _key is not None:
            self.bisect_key_left = _list.bisect_key_left
            self.bisect_key_right = _list.bisect_key_right
            self.bisect_key = _list.bisect_key
            self.irange_key = _list.irange_key

        self._update(*args, **kwargs)


    @property
    def key(self):
        """Function used to extract comparison key from keys.

        Sorted dict compares keys directly when the key function is none.

        """
        return self._key


    @property
    def iloc(self):
        """Cached reference of sorted keys view.

        Deprecated in version 2 of Sorted Containers. Use
        :func:`SortedDict.keys` instead.

        """
        # pylint: disable=attribute-defined-outside-init
        try:
            return self._iloc
        except AttributeError:
            warnings.warn(
                'sorted_dict.iloc is deprecated.'
                ' Use SortedDict.keys() instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            _iloc = self._iloc = SortedKeysView(self)
            return _iloc


    def clear(self):

        """Remove all items from sorted dict.

        Runtime complexity: `O(n)`

        """
        dict.clear(self)
        self._list_clear()


    def __delitem__(self, key):
        """Remove item from sorted dict identified by `key`.

        ``sd.__delitem__(key)`` <==> ``del sd[key]``

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
        >>> del sd['b']
        >>> sd
        SortedDict({'a': 1, 'c': 3})
        >>> del sd['z']
        Traceback (most recent call last):
          ...
        KeyError: 'z'

        :param key: `key` for item lookup
        :raises KeyError: if key not found

        """
        dict.__delitem__(self, key)
        self._list_remove(key)


    def __iter__(self):
        """Return an iterator over the keys of the sorted dict.

        ``sd.__iter__()`` <==> ``iter(sd)``

        Iterating the sorted dict while adding or deleting items may raise a
        :exc:`RuntimeError` or fail to iterate over all keys.

        """
        return self._list_iter()


    def __reversed__(self):
        """Return a reverse iterator over the keys of the sorted dict.

        ``sd.__reversed__()`` <==> ``reversed(sd)``

        Iterating the sorted dict while adding or deleting items may raise a
        :exc:`RuntimeError` or fail to iterate over all keys.

        """
        return self._list_reversed()


    def __setitem__(self, key, value):
        """Store item in sorted dict with `key` and corresponding `value`.

        ``sd.__setitem__(key, value)`` <==> ``sd[key] = value``

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict()
        >>> sd['c'] = 3
        >>> sd['a'] = 1
        >>> sd['b'] = 2
        >>> sd
        SortedDict({'a': 1, 'b': 2, 'c': 3})

        :param key: key for item
        :param value: value for item

        """
        if key not in self:
            self._list_add(key)
        dict.__setitem__(self, key, value)

    _setitem = __setitem__


    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        items = chain(self.items(), other.items())
        return self.__class__(self._key, items)


    def __ror__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        items = chain(other.items(), self.items())
        return self.__class__(self._key, items)


    def __ior__(self, other):
        self._update(other)
        return self


    def copy(self):
        """Return a shallow copy of the sorted dict.

        Runtime complexity: `O(n)`

        :return: new sorted dict

        """
        return self.__class__(self._key, self.items())

    __copy__ = copy


    @classmethod
    def fromkeys(cls, iterable, value=None):
        """Return a new sorted dict initailized from `iterable` and `value`.

        Items in the sorted dict have keys from `iterable` and values equal to
        `value`.

        Runtime complexity: `O(n*log(n))`

        :return: new sorted dict

        """
        return cls((key, value) for key in iterable)


    def keys(self):
        """Return new sorted keys view of the sorted dict's keys.

        See :class:`SortedKeysView` for details.

        :return: new sorted keys view

        """
        return SortedKeysView(self)


    def items(self):
        """Return new sorted items view of the sorted dict's items.

        See :class:`SortedItemsView` for details.

        :return: new sorted items view

        """
        return SortedItemsView(self)


    def values(self):
        """Return new sorted values view of the sorted dict's values.

        See :class:`SortedValuesView` for details.

        :return: new sorted values view

        """
        return SortedValuesView(self)


    if sys.hexversion < 0x03000000:
        def __make_raise_attributeerror(original, alternate):
            # pylint: disable=no-self-argument
            message = (
                'SortedDict.{original}() is not implemented.'
                ' Use SortedDict.{alternate}() instead.'
            ).format(original=original, alternate=alternate)
            def method(self):
                # pylint: disable=missing-docstring,unused-argument
                raise AttributeError(message)
            method.__name__ = original  # pylint: disable=non-str-assignment-to-dunder-name
            method.__doc__ = message
            return property(method)

        iteritems = __make_raise_attributeerror('iteritems', 'items')
        iterkeys = __make_raise_attributeerror('iterkeys', 'keys')
        itervalues = __make_raise_attributeerror('itervalues', 'values')
        viewitems = __make_raise_attributeerror('viewitems', 'items')
        viewkeys = __make_raise_attributeerror('viewkeys', 'keys')
        viewvalues = __make_raise_attributeerror('viewvalues', 'values')


    class _NotGiven(object):
        # pylint: disable=too-few-public-methods
        def __repr__(self):
            return '<not-given>'

    __not_given = _NotGiven()

    def pop(self, key, default=__not_given):
        """Remove and return value for item identified by `key`.

        If the `key` is not found then return `default` if given. If `default`
        is not given then raise :exc:`KeyError`.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
        >>> sd.pop('c')
        3
        >>> sd.pop('z', 26)
        26
        >>> sd.pop('y')
        Traceback (most recent call last):
          ...
        KeyError: 'y'

        :param key: `key` for item
        :param default: `default` value if key not found (optional)
        :return: value for item
        :raises KeyError: if `key` not found and `default` not given

        """
        if key in self:
            self._list_remove(key)
            return dict.pop(self, key)
        else:
            if default is self.__not_given:
                raise KeyError(key)
            return default


    def popitem(self, index=-1):
        """Remove and return ``(key, value)`` pair at `index` from sorted dict.

        Optional argument `index` defaults to -1, the last item in the sorted
        dict. Specify ``index=0`` for the first item in the sorted dict.

        If the sorted dict is empty, raises :exc:`KeyError`.

        If the `index` is out of range, raises :exc:`IndexError`.

        Runtime complexity: `O(log(n))`

        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
        >>> sd.popitem()
        ('c', 3)
        >>> sd.popitem(0)
        ('a', 1)
        >>> sd.popitem(100)
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        :param int index: `index` of item (default -1)
        :return: key and value pair
        :raises KeyError: if sorted dict is empty
        :raises IndexError: if `index` out of range

        """
        if not self:
            raise KeyError('popitem(): dictionary is empty')

        key = self._list_pop(index)
        value = dict.pop(self, key)
        return (key, value)


    def peekitem(self, index=-1):
        """Return ``(key, value)`` pair at `index` in sorted dict.

        Optional argument `index` defaults to -1, the last item in the sorted
        dict. Specify ``index=0`` for the first item in the sorted dict.

        Unlike :func:`SortedDict.popitem`, the sorted dict is not modified.

        If the `index` is out of range, raises :exc:`IndexError`.

        Runtime complexity: `O(log(n))`

        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
        >>> sd.peekitem()
        ('c', 3)
        >>> sd.peekitem(0)
        ('a', 1)
        >>> sd.peekitem(100)
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        :param int index: index of item (default -1)
        :return: key and value pair
        :raises IndexError: if `index` out of range

        """
        key = self._list[index]
        return key, self[key]


    def setdefault(self, key, default=None):
        """Return value for item identified by `key` in sorted dict.

        If `key` is in the sorted dict then return its value. If `key` is not
        in the sorted dict then insert `key` with value `default` and return
        `default`.

        Optional argument `default` defaults to none.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict()
        >>> sd.setdefault('a', 1)
        1
        >>> sd.setdefault('a', 10)
        1
        >>> sd
        SortedDict({'a': 1})

        :param key: key for item
        :param default: value for item (default None)
        :return: value for item identified by `key`

        """
        if key in self:
            return self[key]
        dict.__setitem__(self, key, default)
        self._list_add(key)
        return default


    def update(self, *args, **kwargs):
        """Update sorted dict with items from `args` and `kwargs`.

        Overwrites existing items.

        Optional arguments `args` and `kwargs` may be a mapping, an iterable of
        pairs or keyword arguments. See :func:`SortedDict.__init__` for
        details.

        :param args: mapping or iterable of pairs
        :param kwargs: keyword arguments mapping

        """
        if not self:
            dict.update(self, *args, **kwargs)
            self._list_update(dict.__iter__(self))
            return

        if not kwargs and len(args) == 1 and isinstance(args[0], dict):
            pairs = args[0]
        else:
            pairs = dict(*args, **kwargs)

        if (10 * len(pairs)) > len(self):
            dict.update(self, pairs)
            self._list_clear()
            self._list_update(dict.__iter__(self))
        else:
            for key in pairs:
                self._setitem(key, pairs[key])

    _update = update


    def __reduce__(self):
        """Support for pickle.

        The tricks played with caching references in
        :func:`SortedDict.__init__` confuse pickle so customize the reducer.

        """
        items = dict.copy(self)
        return (type(self), (self._key, items))


    @recursive_repr()
    def __repr__(self):
        """Return string representation of sorted dict.

        ``sd.__repr__()`` <==> ``repr(sd)``

        :return: string representation

        """
        _key = self._key
        type_name = type(self).__name__
        key_arg = '' if _key is None else '{0!r}, '.format(_key)
        item_format = '{0!r}: {1!r}'.format
        items = ', '.join(item_format(key, self[key]) for key in self._list)
        return '{0}({1}{{{2}}})'.format(type_name, key_arg, items)


    def _check(self):
        """Check invariants of sorted dict.

        Runtime complexity: `O(n)`

        """
        _list = self._list
        _list._check()
        assert len(self) == len(_list)
        assert all(key in self for key in _list)


def _view_delitem(self, index):
    """Remove item at `index` from sorted dict.

    ``view.__delitem__(index)`` <==> ``del view[index]``

    Supports slicing.

    Runtime complexity: `O(log(n))` -- approximate.

    >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
    >>> view = sd.keys()
    >>> del view[0]
    >>> sd
    SortedDict({'b': 2, 'c': 3})
    >>> del view[-1]
    >>> sd
    SortedDict({'b': 2})
    >>> del view[:]
    >>> sd
    SortedDict({})

    :param index: integer or slice for indexing
    :raises IndexError: if index out of range

    """
    _mapping = self._mapping
    _list = _mapping._list
    dict_delitem = dict.__delitem__
    if isinstance(index, slice):
        keys = _list[index]
        del _list[index]
        for key in keys:
            dict_delitem(_mapping, key)
    else:
        key = _list.pop(index)
        dict_delitem(_mapping, key)


class SortedKeysView(KeysView, Sequence):
    """Sorted keys view is a dynamic view of the sorted dict's keys.

    When the sorted dict's keys change, the view reflects those changes.

    The keys view implements the set and sequence abstract base classes.

    """
    __slots__ = ()


    @classmethod
    def _from_iterable(cls, it):
        return SortedSet(it)


    def __getitem__(self, index):
        """Lookup key at `index` in sorted keys views.

        ``skv.__getitem__(index)`` <==> ``skv[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
        >>> skv = sd.keys()
        >>> skv[0]
        'a'
        >>> skv[-1]
        'c'
        >>> skv[:]
        ['a', 'b', 'c']
        >>> skv[100]
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        :param index: integer or slice for indexing
        :return: key or list of keys
        :raises IndexError: if index out of range

        """
        return self._mapping._list[index]


    __delitem__ = _view_delitem


class SortedItemsView(ItemsView, Sequence):
    """Sorted items view is a dynamic view of the sorted dict's items.

    When the sorted dict's items change, the view reflects those changes.

    The items view implements the set and sequence abstract base classes.

    """
    __slots__ = ()


    @classmethod
    def _from_iterable(cls, it):
        return SortedSet(it)


    def __getitem__(self, index):
        """Lookup item at `index` in sorted items view.

        ``siv.__getitem__(index)`` <==> ``siv[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
        >>> siv = sd.items()
        >>> siv[0]
        ('a', 1)
        >>> siv[-1]
        ('c', 3)
        >>> siv[:]
        [('a', 1), ('b', 2), ('c', 3)]
        >>> siv[100]
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        :param index: integer or slice for indexing
        :return: item or list of items
        :raises IndexError: if index out of range

        """
        _mapping = self._mapping
        _mapping_list = _mapping._list

        if isinstance(index, slice):
            keys = _mapping_list[index]
            return [(key, _mapping[key]) for key in keys]

        key = _mapping_list[index]
        return key, _mapping[key]


    __delitem__ = _view_delitem


class SortedValuesView(ValuesView, Sequence):
    """Sorted values view is a dynamic view of the sorted dict's values.

    When the sorted dict's values change, the view reflects those changes.

    The values view implements the sequence abstract base class.

    """
    __slots__ = ()


    def __getitem__(self, index):
        """Lookup value at `index` in sorted values view.

        ``siv.__getitem__(index)`` <==> ``siv[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
        >>> svv = sd.values()
        >>> svv[0]
        1
        >>> svv[-1]
        3
        >>> svv[:]
        [1, 2, 3]
        >>> svv[100]
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        :param index: integer or slice for indexing
        :return: value or list of values
        :raises IndexError: if index out of range

        """
        _mapping = self._mapping
        _mapping_list = _mapping._list

        if isinstance(index, slice):
            keys = _mapping_list[index]
            return [_mapping[key] for key in keys]

        key = _mapping_list[index]
        return _mapping[key]


    __delitem__ = _view_delitem

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\capi_maps.py ===
"""
Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
from . import __version__

f2py_version = __version__.version

import copy
import os
import re

from . import cb_rules
from ._isocbind import iso_c2py_map, iso_c_binding_map, isoc_c2pycode_map

# The environment provided by auxfuncs.py is needed for some calls to eval.
# As the needed functions cannot be determined by static inspection of the
# code, it is safest to use import * pending a major refactoring of f2py.
from .auxfuncs import *
from .crackfortran import markoutercomma

__all__ = [
    'getctype', 'getstrlength', 'getarrdims', 'getpydocsign',
    'getarrdocsign', 'getinit', 'sign2map', 'routsign2map', 'modsign2map',
    'cb_sign2map', 'cb_routsign2map', 'common_sign2map', 'process_f2cmap_dict'
]


depargs = []
lcb_map = {}
lcb2_map = {}
# forced casting: mainly caused by the fact that Python or Numeric
#                 C/APIs do not support the corresponding C types.
c2py_map = {'double': 'float',
            'float': 'float',                          # forced casting
            'long_double': 'float',                    # forced casting
            'char': 'int',                             # forced casting
            'signed_char': 'int',                      # forced casting
            'unsigned_char': 'int',                    # forced casting
            'short': 'int',                            # forced casting
            'unsigned_short': 'int',                   # forced casting
            'int': 'int',                              # forced casting
            'long': 'int',
            'long_long': 'long',
            'unsigned': 'int',                         # forced casting
            'complex_float': 'complex',                # forced casting
            'complex_double': 'complex',
            'complex_long_double': 'complex',          # forced casting
            'string': 'string',
            'character': 'bytes',
            }

c2capi_map = {'double': 'NPY_DOUBLE',
                'float': 'NPY_FLOAT',
                'long_double': 'NPY_LONGDOUBLE',
                'char': 'NPY_BYTE',
                'unsigned_char': 'NPY_UBYTE',
                'signed_char': 'NPY_BYTE',
                'short': 'NPY_SHORT',
                'unsigned_short': 'NPY_USHORT',
                'int': 'NPY_INT',
                'unsigned': 'NPY_UINT',
                'long': 'NPY_LONG',
                'unsigned_long': 'NPY_ULONG',
                'long_long': 'NPY_LONGLONG',
                'unsigned_long_long': 'NPY_ULONGLONG',
                'complex_float': 'NPY_CFLOAT',
                'complex_double': 'NPY_CDOUBLE',
                'complex_long_double': 'NPY_CDOUBLE',
                'string': 'NPY_STRING',
                'character': 'NPY_STRING'}

c2pycode_map = {'double': 'd',
                'float': 'f',
                'long_double': 'g',
                'char': 'b',
                'unsigned_char': 'B',
                'signed_char': 'b',
                'short': 'h',
                'unsigned_short': 'H',
                'int': 'i',
                'unsigned': 'I',
                'long': 'l',
                'unsigned_long': 'L',
                'long_long': 'q',
                'unsigned_long_long': 'Q',
                'complex_float': 'F',
                'complex_double': 'D',
                'complex_long_double': 'G',
                'string': 'S',
                'character': 'c'}

# https://docs.python.org/3/c-api/arg.html#building-values
c2buildvalue_map = {'double': 'd',
                    'float': 'f',
                    'char': 'b',
                    'signed_char': 'b',
                    'short': 'h',
                    'int': 'i',
                    'long': 'l',
                    'long_long': 'L',
                    'complex_float': 'N',
                    'complex_double': 'N',
                    'complex_long_double': 'N',
                    'string': 'y',
                    'character': 'c'}

f2cmap_all = {'real': {'': 'float', '4': 'float', '8': 'double',
                       '12': 'long_double', '16': 'long_double'},
              'integer': {'': 'int', '1': 'signed_char', '2': 'short',
                          '4': 'int', '8': 'long_long',
                          '-1': 'unsigned_char', '-2': 'unsigned_short',
                          '-4': 'unsigned', '-8': 'unsigned_long_long'},
              'complex': {'': 'complex_float', '8': 'complex_float',
                          '16': 'complex_double', '24': 'complex_long_double',
                          '32': 'complex_long_double'},
              'complexkind': {'': 'complex_float', '4': 'complex_float',
                              '8': 'complex_double', '12': 'complex_long_double',
                              '16': 'complex_long_double'},
              'logical': {'': 'int', '1': 'char', '2': 'short', '4': 'int',
                          '8': 'long_long'},
              'double complex': {'': 'complex_double'},
              'double precision': {'': 'double'},
              'byte': {'': 'char'},
              }

# Add ISO_C handling
c2pycode_map.update(isoc_c2pycode_map)
c2py_map.update(iso_c2py_map)
f2cmap_all, _ = process_f2cmap_dict(f2cmap_all, iso_c_binding_map, c2py_map)
# End ISO_C handling
f2cmap_default = copy.deepcopy(f2cmap_all)

f2cmap_mapped = []

def load_f2cmap_file(f2cmap_file):
    global f2cmap_all, f2cmap_mapped

    f2cmap_all = copy.deepcopy(f2cmap_default)

    if f2cmap_file is None:
        # Default value
        f2cmap_file = '.f2py_f2cmap'
        if not os.path.isfile(f2cmap_file):
            return

    # User defined additions to f2cmap_all.
    # f2cmap_file must contain a dictionary of dictionaries, only. For
    # example, {'real':{'low':'float'}} means that Fortran 'real(low)' is
    # interpreted as C 'float'. This feature is useful for F90/95 users if
    # they use PARAMETERS in type specifications.
    try:
        outmess(f'Reading f2cmap from {f2cmap_file!r} ...\n')
        with open(f2cmap_file) as f:
            d = eval(f.read().lower(), {}, {})
        f2cmap_all, f2cmap_mapped = process_f2cmap_dict(f2cmap_all, d, c2py_map, True)
        outmess('Successfully applied user defined f2cmap changes\n')
    except Exception as msg:
        errmess(f'Failed to apply user defined f2cmap changes: {msg}. Skipping.\n')


cformat_map = {'double': '%g',
               'float': '%g',
               'long_double': '%Lg',
               'char': '%d',
               'signed_char': '%d',
               'unsigned_char': '%hhu',
               'short': '%hd',
               'unsigned_short': '%hu',
               'int': '%d',
               'unsigned': '%u',
               'long': '%ld',
               'unsigned_long': '%lu',
               'long_long': '%ld',
               'complex_float': '(%g,%g)',
               'complex_double': '(%g,%g)',
               'complex_long_double': '(%Lg,%Lg)',
               'string': '\\"%s\\"',
               'character': "'%c'",
               }

# Auxiliary functions


def getctype(var):
    """
    Determines C type
    """
    ctype = 'void'
    if isfunction(var):
        if 'result' in var:
            a = var['result']
        else:
            a = var['name']
        if a in var['vars']:
            return getctype(var['vars'][a])
        else:
            errmess(f'getctype: function {a} has no return value?!\n')
    elif issubroutine(var):
        return ctype
    elif ischaracter_or_characterarray(var):
        return 'character'
    elif isstring_or_stringarray(var):
        return 'string'
    elif 'typespec' in var and var['typespec'].lower() in f2cmap_all:
        typespec = var['typespec'].lower()
        f2cmap = f2cmap_all[typespec]
        ctype = f2cmap['']  # default type
        if 'kindselector' in var:
            if '*' in var['kindselector']:
                try:
                    ctype = f2cmap[var['kindselector']['*']]
                except KeyError:
                    errmess('getctype: "%s %s %s" not supported.\n' %
                            (var['typespec'], '*', var['kindselector']['*']))
            elif 'kind' in var['kindselector']:
                if typespec + 'kind' in f2cmap_all:
                    f2cmap = f2cmap_all[typespec + 'kind']
                try:
                    ctype = f2cmap[var['kindselector']['kind']]
                except KeyError:
                    if typespec in f2cmap_all:
                        f2cmap = f2cmap_all[typespec]
                    try:
                        ctype = f2cmap[str(var['kindselector']['kind'])]
                    except KeyError:
                        errmess('getctype: "%s(kind=%s)" is mapped to C "%s" (to override define dict(%s = dict(%s="<C typespec>")) in %s/.f2py_f2cmap file).\n'
                                % (typespec, var['kindselector']['kind'], ctype,
                                   typespec, var['kindselector']['kind'], os.getcwd()))
    elif not isexternal(var):
        errmess(f'getctype: No C-type found in "{var}", assuming void.\n')
    return ctype


def f2cexpr(expr):
    """Rewrite Fortran expression as f2py supported C expression.

    Due to the lack of a proper expression parser in f2py, this
    function uses a heuristic approach that assumes that Fortran
    arithmetic expressions are valid C arithmetic expressions when
    mapping Fortran function calls to the corresponding C function/CPP
    macros calls.

    """
    # TODO: support Fortran `len` function with optional kind parameter
    expr = re.sub(r'\blen\b', 'f2py_slen', expr)
    return expr


def getstrlength(var):
    if isstringfunction(var):
        if 'result' in var:
            a = var['result']
        else:
            a = var['name']
        if a in var['vars']:
            return getstrlength(var['vars'][a])
        else:
            errmess(f'getstrlength: function {a} has no return value?!\n')
    if not isstring(var):
        errmess(
            f'getstrlength: expected a signature of a string but got: {repr(var)}\n')
    len = '1'
    if 'charselector' in var:
        a = var['charselector']
        if '*' in a:
            len = a['*']
        elif 'len' in a:
            len = f2cexpr(a['len'])
    if re.match(r'\(\s*(\*|:)\s*\)', len) or re.match(r'(\*|:)', len):
        if isintent_hide(var):
            errmess('getstrlength:intent(hide): expected a string with defined length but got: %s\n' % (
                repr(var)))
        len = '-1'
    return len


def getarrdims(a, var, verbose=0):
    ret = {}
    if isstring(var) and not isarray(var):
        ret['size'] = getstrlength(var)
        ret['rank'] = '0'
        ret['dims'] = ''
    elif isscalar(var):
        ret['size'] = '1'
        ret['rank'] = '0'
        ret['dims'] = ''
    elif isarray(var):
        dim = copy.copy(var['dimension'])
        ret['size'] = '*'.join(dim)
        try:
            ret['size'] = repr(eval(ret['size']))
        except Exception:
            pass
        ret['dims'] = ','.join(dim)
        ret['rank'] = repr(len(dim))
        ret['rank*[-1]'] = repr(len(dim) * [-1])[1:-1]
        for i in range(len(dim)):  # solve dim for dependencies
            v = []
            if dim[i] in depargs:
                v = [dim[i]]
            else:
                for va in depargs:
                    if re.match(r'.*?\b%s\b.*' % va, dim[i]):
                        v.append(va)
            for va in v:
                if depargs.index(va) > depargs.index(a):
                    dim[i] = '*'
                    break
        ret['setdims'], i = '', -1
        for d in dim:
            i = i + 1
            if d not in ['*', ':', '(*)', '(:)']:
                ret['setdims'] = '%s#varname#_Dims[%d]=%s,' % (
                    ret['setdims'], i, d)
        if ret['setdims']:
            ret['setdims'] = ret['setdims'][:-1]
        ret['cbsetdims'], i = '', -1
        for d in var['dimension']:
            i = i + 1
            if d not in ['*', ':', '(*)', '(:)']:
                ret['cbsetdims'] = '%s#varname#_Dims[%d]=%s,' % (
                    ret['cbsetdims'], i, d)
            elif isintent_in(var):
                outmess('getarrdims:warning: assumed shape array, using 0 instead of %r\n'
                        % (d))
                ret['cbsetdims'] = '%s#varname#_Dims[%d]=%s,' % (
                    ret['cbsetdims'], i, 0)
            elif verbose:
                errmess(
                    f'getarrdims: If in call-back function: array argument {repr(a)} must have bounded dimensions: got {repr(d)}\n')
        if ret['cbsetdims']:
            ret['cbsetdims'] = ret['cbsetdims'][:-1]
#         if not isintent_c(var):
#             var['dimension'].reverse()
    return ret


def getpydocsign(a, var):
    global lcb_map
    if isfunction(var):
        if 'result' in var:
            af = var['result']
        else:
            af = var['name']
        if af in var['vars']:
            return getpydocsign(af, var['vars'][af])
        else:
            errmess(f'getctype: function {af} has no return value?!\n')
        return '', ''
    sig, sigout = a, a
    opt = ''
    if isintent_in(var):
        opt = 'input'
    elif isintent_inout(var):
        opt = 'in/output'
    out_a = a
    if isintent_out(var):
        for k in var['intent']:
            if k[:4] == 'out=':
                out_a = k[4:]
                break
    init = ''
    ctype = getctype(var)

    if hasinitvalue(var):
        init, showinit = getinit(a, var)
        init = f', optional\\n    Default: {showinit}'
    if isscalar(var):
        if isintent_inout(var):
            sig = '%s : %s rank-0 array(%s,\'%s\')%s' % (a, opt, c2py_map[ctype],
                                                         c2pycode_map[ctype], init)
        else:
            sig = f'{a} : {opt} {c2py_map[ctype]}{init}'
        sigout = f'{out_a} : {c2py_map[ctype]}'
    elif isstring(var):
        if isintent_inout(var):
            sig = '%s : %s rank-0 array(string(len=%s),\'c\')%s' % (
                a, opt, getstrlength(var), init)
        else:
            sig = f'{a} : {opt} string(len={getstrlength(var)}){init}'
        sigout = f'{out_a} : string(len={getstrlength(var)})'
    elif isarray(var):
        dim = var['dimension']
        rank = repr(len(dim))
        sig = '%s : %s rank-%s array(\'%s\') with bounds (%s)%s' % (a, opt, rank,
                                                                    c2pycode_map[
                                                                        ctype],
                                                                    ','.join(dim), init)
        if a == out_a:
            sigout = '%s : rank-%s array(\'%s\') with bounds (%s)'\
                % (a, rank, c2pycode_map[ctype], ','.join(dim))
        else:
            sigout = '%s : rank-%s array(\'%s\') with bounds (%s) and %s storage'\
                % (out_a, rank, c2pycode_map[ctype], ','.join(dim), a)
    elif isexternal(var):
        ua = ''
        if a in lcb_map and lcb_map[a] in lcb2_map and 'argname' in lcb2_map[lcb_map[a]]:
            ua = lcb2_map[lcb_map[a]]['argname']
            if not ua == a:
                ua = f' => {ua}'
            else:
                ua = ''
        sig = f'{a} : call-back function{ua}'
        sigout = sig
    else:
        errmess(
            f'getpydocsign: Could not resolve docsignature for "{a}".\n')
    return sig, sigout


def getarrdocsign(a, var):
    ctype = getctype(var)
    if isstring(var) and (not isarray(var)):
        sig = f'{a} : rank-0 array(string(len={getstrlength(var)}),\'c\')'
    elif isscalar(var):
        sig = f'{a} : rank-0 array({c2py_map[ctype]},\'{c2pycode_map[ctype]}\')'
    elif isarray(var):
        dim = var['dimension']
        rank = repr(len(dim))
        sig = '%s : rank-%s array(\'%s\') with bounds (%s)' % (a, rank,
                                                               c2pycode_map[
                                                                   ctype],
                                                               ','.join(dim))
    return sig


def getinit(a, var):
    if isstring(var):
        init, showinit = '""', "''"
    else:
        init, showinit = '', ''
    if hasinitvalue(var):
        init = var['=']
        showinit = init
        if iscomplex(var) or iscomplexarray(var):
            ret = {}

            try:
                v = var["="]
                if ',' in v:
                    ret['init.r'], ret['init.i'] = markoutercomma(
                        v[1:-1]).split('@,@')
                else:
                    v = eval(v, {}, {})
                    ret['init.r'], ret['init.i'] = str(v.real), str(v.imag)
            except Exception:
                raise ValueError(
                    f'getinit: expected complex number `(r,i)\' but got `{init}\' as initial value of {a!r}.')
            if isarray(var):
                init = f"(capi_c.r={ret['init.r']},capi_c.i={ret['init.i']},capi_c)"
        elif isstring(var):
            if not init:
                init, showinit = '""', "''"
            if init[0] == "'":
                init = '"%s"' % (init[1:-1].replace('"', '\\"'))
            if init[0] == '"':
                showinit = f"'{init[1:-1]}'"
    return init, showinit


def get_elsize(var):
    if isstring(var) or isstringarray(var):
        elsize = getstrlength(var)
        # override with user-specified length when available:
        elsize = var['charselector'].get('f2py_len', elsize)
        return elsize
    if ischaracter(var) or ischaracterarray(var):
        return '1'
    # for numerical types, PyArray_New* functions ignore specified
    # elsize, so we just return 1 and let elsize be determined at
    # runtime, see fortranobject.c
    return '1'


def sign2map(a, var):
    """
    varname,ctype,atype
    init,init.r,init.i,pytype
    vardebuginfo,vardebugshowvalue,varshowvalue
    varrformat

    intent
    """
    out_a = a
    if isintent_out(var):
        for k in var['intent']:
            if k[:4] == 'out=':
                out_a = k[4:]
                break
    ret = {'varname': a, 'outvarname': out_a, 'ctype': getctype(var)}
    intent_flags = []
    for f, s in isintent_dict.items():
        if f(var):
            intent_flags.append(f'F2PY_{s}')
    if intent_flags:
        # TODO: Evaluate intent_flags here.
        ret['intent'] = '|'.join(intent_flags)
    else:
        ret['intent'] = 'F2PY_INTENT_IN'
    if isarray(var):
        ret['varrformat'] = 'N'
    elif ret['ctype'] in c2buildvalue_map:
        ret['varrformat'] = c2buildvalue_map[ret['ctype']]
    else:
        ret['varrformat'] = 'O'
    ret['init'], ret['showinit'] = getinit(a, var)
    if hasinitvalue(var) and iscomplex(var) and not isarray(var):
        ret['init.r'], ret['init.i'] = markoutercomma(
            ret['init'][1:-1]).split('@,@')
    if isexternal(var):
        ret['cbnamekey'] = a
        if a in lcb_map:
            ret['cbname'] = lcb_map[a]
            ret['maxnofargs'] = lcb2_map[lcb_map[a]]['maxnofargs']
            ret['nofoptargs'] = lcb2_map[lcb_map[a]]['nofoptargs']
            ret['cbdocstr'] = lcb2_map[lcb_map[a]]['docstr']
            ret['cblatexdocstr'] = lcb2_map[lcb_map[a]]['latexdocstr']
        else:
            ret['cbname'] = a
            errmess('sign2map: Confused: external %s is not in lcb_map%s.\n' % (
                a, list(lcb_map.keys())))
    if isstring(var):
        ret['length'] = getstrlength(var)
    if isarray(var):
        ret = dictappend(ret, getarrdims(a, var))
        dim = copy.copy(var['dimension'])
    if ret['ctype'] in c2capi_map:
        ret['atype'] = c2capi_map[ret['ctype']]
        ret['elsize'] = get_elsize(var)
    # Debug info
    if debugcapi(var):
        il = [isintent_in, 'input', isintent_out, 'output',
              isintent_inout, 'inoutput', isrequired, 'required',
              isoptional, 'optional', isintent_hide, 'hidden',
              iscomplex, 'complex scalar',
              l_and(isscalar, l_not(iscomplex)), 'scalar',
              isstring, 'string', isarray, 'array',
              iscomplexarray, 'complex array', isstringarray, 'string array',
              iscomplexfunction, 'complex function',
              l_and(isfunction, l_not(iscomplexfunction)), 'function',
              isexternal, 'callback',
              isintent_callback, 'callback',
              isintent_aux, 'auxiliary',
              ]
        rl = []
        for i in range(0, len(il), 2):
            if il[i](var):
                rl.append(il[i + 1])
        if isstring(var):
            rl.append(f"slen({a})={ret['length']}")
        if isarray(var):
            ddim = ','.join(
                map(lambda x, y: f'{x}|{y}', var['dimension'], dim))
            rl.append(f'dims({ddim})')
        if isexternal(var):
            ret['vardebuginfo'] = f"debug-capi:{a}=>{ret['cbname']}:{','.join(rl)}"
        else:
            ret['vardebuginfo'] = 'debug-capi:%s %s=%s:%s' % (
                ret['ctype'], a, ret['showinit'], ','.join(rl))
        if isscalar(var):
            if ret['ctype'] in cformat_map:
                ret['vardebugshowvalue'] = f"debug-capi:{a}={cformat_map[ret['ctype']]}"
        if isstring(var):
            ret['vardebugshowvalue'] = 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"' % (
                a, a)
        if isexternal(var):
            ret['vardebugshowvalue'] = f'debug-capi:{a}=%p'
    if ret['ctype'] in cformat_map:
        ret['varshowvalue'] = f"#name#:{a}={cformat_map[ret['ctype']]}"
        ret['showvalueformat'] = f"{cformat_map[ret['ctype']]}"
    if isstring(var):
        ret['varshowvalue'] = '#name#:slen(%s)=%%d %s=\\"%%s\\"' % (a, a)
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    if hasnote(var):
        ret['note'] = var['note']
    return ret


def routsign2map(rout):
    """
    name,NAME,begintitle,endtitle
    rname,ctype,rformat
    routdebugshowvalue
    """
    global lcb_map
    name = rout['name']
    fname = getfortranname(rout)
    ret = {'name': name,
           'texname': name.replace('_', '\\_'),
           'name_lower': name.lower(),
           'NAME': name.upper(),
           'begintitle': gentitle(name),
           'endtitle': gentitle(f'end of {name}'),
           'fortranname': fname,
           'FORTRANNAME': fname.upper(),
           'callstatement': getcallstatement(rout) or '',
           'usercode': getusercode(rout) or '',
           'usercode1': getusercode1(rout) or '',
           }
    if '_' in fname:
        ret['F_FUNC'] = 'F_FUNC_US'
    else:
        ret['F_FUNC'] = 'F_FUNC'
    if '_' in name:
        ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC_US'
    else:
        ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC'
    lcb_map = {}
    if 'use' in rout:
        for u in rout['use'].keys():
            if u in cb_rules.cb_map:
                for un in cb_rules.cb_map[u]:
                    ln = un[0]
                    if 'map' in rout['use'][u]:
                        for k in rout['use'][u]['map'].keys():
                            if rout['use'][u]['map'][k] == un[0]:
                                ln = k
                                break
                    lcb_map[ln] = un[1]
    elif rout.get('externals'):
        errmess('routsign2map: Confused: function %s has externals %s but no "use" statement.\n' % (
            ret['name'], repr(rout['externals'])))
    ret['callprotoargument'] = getcallprotoargument(rout, lcb_map) or ''
    if isfunction(rout):
        if 'result' in rout:
            a = rout['result']
        else:
            a = rout['name']
        ret['rname'] = a
        ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, rout)
        ret['ctype'] = getctype(rout['vars'][a])
        if hasresultnote(rout):
            ret['resultnote'] = rout['vars'][a]['note']
            rout['vars'][a]['note'] = ['See elsewhere.']
        if ret['ctype'] in c2buildvalue_map:
            ret['rformat'] = c2buildvalue_map[ret['ctype']]
        else:
            ret['rformat'] = 'O'
            errmess('routsign2map: no c2buildvalue key for type %s\n' %
                    (repr(ret['ctype'])))
        if debugcapi(rout):
            if ret['ctype'] in cformat_map:
                ret['routdebugshowvalue'] = 'debug-capi:%s=%s' % (
                    a, cformat_map[ret['ctype']])
            if isstringfunction(rout):
                ret['routdebugshowvalue'] = 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"' % (
                    a, a)
        if isstringfunction(rout):
            ret['rlength'] = getstrlength(rout['vars'][a])
            if ret['rlength'] == '-1':
                errmess('routsign2map: expected explicit specification of the length of the string returned by the fortran function %s; taking 10.\n' % (
                    repr(rout['name'])))
                ret['rlength'] = '10'
    if hasnote(rout):
        ret['note'] = rout['note']
        rout['note'] = ['See elsewhere.']
    return ret


def modsign2map(m):
    """
    modulename
    """
    if ismodule(m):
        ret = {'f90modulename': m['name'],
               'F90MODULENAME': m['name'].upper(),
               'texf90modulename': m['name'].replace('_', '\\_')}
    else:
        ret = {'modulename': m['name'],
               'MODULENAME': m['name'].upper(),
               'texmodulename': m['name'].replace('_', '\\_')}
    ret['restdoc'] = getrestdoc(m) or []
    if hasnote(m):
        ret['note'] = m['note']
    ret['usercode'] = getusercode(m) or ''
    ret['usercode1'] = getusercode1(m) or ''
    if m['body']:
        ret['interface_usercode'] = getusercode(m['body'][0]) or ''
    else:
        ret['interface_usercode'] = ''
    ret['pymethoddef'] = getpymethoddef(m) or ''
    if 'gil_used' in m:
        ret['gil_used'] = m['gil_used']
    if 'coutput' in m:
        ret['coutput'] = m['coutput']
    if 'f2py_wrapper_output' in m:
        ret['f2py_wrapper_output'] = m['f2py_wrapper_output']
    return ret


def cb_sign2map(a, var, index=None):
    ret = {'varname': a}
    ret['varname_i'] = ret['varname']
    ret['ctype'] = getctype(var)
    if ret['ctype'] in c2capi_map:
        ret['atype'] = c2capi_map[ret['ctype']]
        ret['elsize'] = get_elsize(var)
    if ret['ctype'] in cformat_map:
        ret['showvalueformat'] = f"{cformat_map[ret['ctype']]}"
    if isarray(var):
        ret = dictappend(ret, getarrdims(a, var))
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    if hasnote(var):
        ret['note'] = var['note']
        var['note'] = ['See elsewhere.']
    return ret


def cb_routsign2map(rout, um):
    """
    name,begintitle,endtitle,argname
    ctype,rctype,maxnofargs,nofoptargs,returncptr
    """
    ret = {'name': f"cb_{rout['name']}_in_{um}",
           'returncptr': ''}
    if isintent_callback(rout):
        if '_' in rout['name']:
            F_FUNC = 'F_FUNC_US'
        else:
            F_FUNC = 'F_FUNC'
        ret['callbackname'] = f"{F_FUNC}({rout['name'].lower()},{rout['name'].upper()})"
        ret['static'] = 'extern'
    else:
        ret['callbackname'] = ret['name']
        ret['static'] = 'static'
    ret['argname'] = rout['name']
    ret['begintitle'] = gentitle(ret['name'])
    ret['endtitle'] = gentitle(f"end of {ret['name']}")
    ret['ctype'] = getctype(rout)
    ret['rctype'] = 'void'
    if ret['ctype'] == 'string':
        ret['rctype'] = 'void'
    else:
        ret['rctype'] = ret['ctype']
    if ret['rctype'] != 'void':
        if iscomplexfunction(rout):
            ret['returncptr'] = """
#ifdef F2PY_CB_RETURNCOMPLEX
return_value=
#endif
"""
        else:
            ret['returncptr'] = 'return_value='
    if ret['ctype'] in cformat_map:
        ret['showvalueformat'] = f"{cformat_map[ret['ctype']]}"
    if isstringfunction(rout):
        ret['strlength'] = getstrlength(rout)
    if isfunction(rout):
        if 'result' in rout:
            a = rout['result']
        else:
            a = rout['name']
        if hasnote(rout['vars'][a]):
            ret['note'] = rout['vars'][a]['note']
            rout['vars'][a]['note'] = ['See elsewhere.']
        ret['rname'] = a
        ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, rout)
        if iscomplexfunction(rout):
            ret['rctype'] = """
#ifdef F2PY_CB_RETURNCOMPLEX
#ctype#
#else
void
#endif
"""
    elif hasnote(rout):
        ret['note'] = rout['note']
        rout['note'] = ['See elsewhere.']
    nofargs = 0
    nofoptargs = 0
    if 'args' in rout and 'vars' in rout:
        for a in rout['args']:
            var = rout['vars'][a]
            if l_or(isintent_in, isintent_inout)(var):
                nofargs = nofargs + 1
                if isoptional(var):
                    nofoptargs = nofoptargs + 1
    ret['maxnofargs'] = repr(nofargs)
    ret['nofoptargs'] = repr(nofoptargs)
    if hasnote(rout) and isfunction(rout) and 'result' in rout:
        ret['routnote'] = rout['note']
        rout['note'] = ['See elsewhere.']
    return ret


def common_sign2map(a, var):  # obsolete
    ret = {'varname': a, 'ctype': getctype(var)}
    if isstringarray(var):
        ret['ctype'] = 'char'
    if ret['ctype'] in c2capi_map:
        ret['atype'] = c2capi_map[ret['ctype']]
        ret['elsize'] = get_elsize(var)
    if ret['ctype'] in cformat_map:
        ret['showvalueformat'] = f"{cformat_map[ret['ctype']]}"
    if isarray(var):
        ret = dictappend(ret, getarrdims(a, var))
    elif isstring(var):
        ret['size'] = getstrlength(var)
        ret['rank'] = '1'
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    if hasnote(var):
        ret['note'] = var['note']
        var['note'] = ['See elsewhere.']
    # for strings this returns 0-rank but actually is 1-rank
    ret['arrdocstr'] = getarrdocsign(a, var)
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\path_registry.py ===
# orm/path_registry.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
"""Path tracking utilities, representing mapper graph traversals.

"""

from __future__ import annotations

from functools import reduce
from itertools import chain
import logging
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from . import base as orm_base
from ._typing import insp_is_mapper_property
from .. import exc
from .. import util
from ..sql import visitors
from ..sql.cache_key import HasCacheKey

if TYPE_CHECKING:
    from ._typing import _InternalEntityType
    from .interfaces import StrategizedProperty
    from .mapper import Mapper
    from .relationships import RelationshipProperty
    from .util import AliasedInsp
    from ..sql.cache_key import _CacheKeyTraversalType
    from ..sql.elements import BindParameter
    from ..sql.visitors import anon_map
    from ..util.typing import _LiteralStar
    from ..util.typing import TypeGuard

    def is_root(path: PathRegistry) -> TypeGuard[RootRegistry]: ...

    def is_entity(path: PathRegistry) -> TypeGuard[AbstractEntityRegistry]: ...

else:
    is_root = operator.attrgetter("is_root")
    is_entity = operator.attrgetter("is_entity")


_SerializedPath = List[Any]
_StrPathToken = str
_PathElementType = Union[
    _StrPathToken, "_InternalEntityType[Any]", "StrategizedProperty[Any]"
]

# the representation is in fact
# a tuple with alternating:
# [_InternalEntityType[Any], Union[str, StrategizedProperty[Any]],
# _InternalEntityType[Any], Union[str, StrategizedProperty[Any]], ...]
# this might someday be a tuple of 2-tuples instead, but paths can be
# chopped at odd intervals as well so this is less flexible
_PathRepresentation = Tuple[_PathElementType, ...]

# NOTE: these names are weird since the array is 0-indexed,
# the "_Odd" entries are at 0, 2, 4, etc
_OddPathRepresentation = Sequence["_InternalEntityType[Any]"]
_EvenPathRepresentation = Sequence[Union["StrategizedProperty[Any]", str]]


log = logging.getLogger(__name__)


def _unreduce_path(path: _SerializedPath) -> PathRegistry:
    return PathRegistry.deserialize(path)


_WILDCARD_TOKEN: _LiteralStar = "*"
_DEFAULT_TOKEN = "_sa_default"


class PathRegistry(HasCacheKey):
    """Represent query load paths and registry functions.

    Basically represents structures like:

    (<User mapper>, "orders", <Order mapper>, "items", <Item mapper>)

    These structures are generated by things like
    query options (joinedload(), subqueryload(), etc.) and are
    used to compose keys stored in the query._attributes dictionary
    for various options.

    They are then re-composed at query compile/result row time as
    the query is formed and as rows are fetched, where they again
    serve to compose keys to look up options in the context.attributes
    dictionary, which is copied from query._attributes.

    The path structure has a limited amount of caching, where each
    "root" ultimately pulls from a fixed registry associated with
    the first mapper, that also contains elements for each of its
    property keys.  However paths longer than two elements, which
    are the exception rather than the rule, are generated on an
    as-needed basis.

    """

    __slots__ = ()

    is_token = False
    is_root = False
    has_entity = False
    is_property = False
    is_entity = False

    is_unnatural: bool

    path: _PathRepresentation
    natural_path: _PathRepresentation
    parent: Optional[PathRegistry]
    root: RootRegistry

    _cache_key_traversal: _CacheKeyTraversalType = [
        ("path", visitors.ExtendedInternalTraversal.dp_has_cache_key_list)
    ]

    def __eq__(self, other: Any) -> bool:
        try:
            return other is not None and self.path == other._path_for_compare
        except AttributeError:
            util.warn(
                "Comparison of PathRegistry to %r is not supported"
                % (type(other))
            )
            return False

    def __ne__(self, other: Any) -> bool:
        try:
            return other is None or self.path != other._path_for_compare
        except AttributeError:
            util.warn(
                "Comparison of PathRegistry to %r is not supported"
                % (type(other))
            )
            return True

    @property
    def _path_for_compare(self) -> Optional[_PathRepresentation]:
        return self.path

    def odd_element(self, index: int) -> _InternalEntityType[Any]:
        return self.path[index]  # type: ignore

    def set(self, attributes: Dict[Any, Any], key: Any, value: Any) -> None:
        log.debug("set '%s' on path '%s' to '%s'", key, self, value)
        attributes[(key, self.natural_path)] = value

    def setdefault(
        self, attributes: Dict[Any, Any], key: Any, value: Any
    ) -> None:
        log.debug("setdefault '%s' on path '%s' to '%s'", key, self, value)
        attributes.setdefault((key, self.natural_path), value)

    def get(
        self, attributes: Dict[Any, Any], key: Any, value: Optional[Any] = None
    ) -> Any:
        key = (key, self.natural_path)
        if key in attributes:
            return attributes[key]
        else:
            return value

    def __len__(self) -> int:
        return len(self.path)

    def __hash__(self) -> int:
        return id(self)

    @overload
    def __getitem__(self, entity: _StrPathToken) -> TokenRegistry: ...

    @overload
    def __getitem__(self, entity: int) -> _PathElementType: ...

    @overload
    def __getitem__(self, entity: slice) -> _PathRepresentation: ...

    @overload
    def __getitem__(
        self, entity: _InternalEntityType[Any]
    ) -> AbstractEntityRegistry: ...

    @overload
    def __getitem__(
        self, entity: StrategizedProperty[Any]
    ) -> PropRegistry: ...

    def __getitem__(
        self,
        entity: Union[
            _StrPathToken,
            int,
            slice,
            _InternalEntityType[Any],
            StrategizedProperty[Any],
        ],
    ) -> Union[
        TokenRegistry,
        _PathElementType,
        _PathRepresentation,
        PropRegistry,
        AbstractEntityRegistry,
    ]:
        raise NotImplementedError()

    # TODO: what are we using this for?
    @property
    def length(self) -> int:
        return len(self.path)

    def pairs(
        self,
    ) -> Iterator[
        Tuple[_InternalEntityType[Any], Union[str, StrategizedProperty[Any]]]
    ]:
        odd_path = cast(_OddPathRepresentation, self.path)
        even_path = cast(_EvenPathRepresentation, odd_path)
        for i in range(0, len(odd_path), 2):
            yield odd_path[i], even_path[i + 1]

    def contains_mapper(self, mapper: Mapper[Any]) -> bool:
        _m_path = cast(_OddPathRepresentation, self.path)
        for path_mapper in [_m_path[i] for i in range(0, len(_m_path), 2)]:
            if path_mapper.mapper.isa(mapper):
                return True
        else:
            return False

    def contains(self, attributes: Dict[Any, Any], key: Any) -> bool:
        return (key, self.path) in attributes

    def __reduce__(self) -> Any:
        return _unreduce_path, (self.serialize(),)

    @classmethod
    def _serialize_path(cls, path: _PathRepresentation) -> _SerializedPath:
        _m_path = cast(_OddPathRepresentation, path)
        _p_path = cast(_EvenPathRepresentation, path)

        return list(
            zip(
                tuple(
                    m.class_ if (m.is_mapper or m.is_aliased_class) else str(m)
                    for m in [_m_path[i] for i in range(0, len(_m_path), 2)]
                ),
                tuple(
                    p.key if insp_is_mapper_property(p) else str(p)
                    for p in [_p_path[i] for i in range(1, len(_p_path), 2)]
                )
                + (None,),
            )
        )

    @classmethod
    def _deserialize_path(cls, path: _SerializedPath) -> _PathRepresentation:
        def _deserialize_mapper_token(mcls: Any) -> Any:
            return (
                # note: we likely dont want configure=True here however
                # this is maintained at the moment for backwards compatibility
                orm_base._inspect_mapped_class(mcls, configure=True)
                if mcls not in PathToken._intern
                else PathToken._intern[mcls]
            )

        def _deserialize_key_token(mcls: Any, key: Any) -> Any:
            if key is None:
                return None
            elif key in PathToken._intern:
                return PathToken._intern[key]
            else:
                mp = orm_base._inspect_mapped_class(mcls, configure=True)
                assert mp is not None
                return mp.attrs[key]

        p = tuple(
            chain(
                *[
                    (
                        _deserialize_mapper_token(mcls),
                        _deserialize_key_token(mcls, key),
                    )
                    for mcls, key in path
                ]
            )
        )
        if p and p[-1] is None:
            p = p[0:-1]
        return p

    def serialize(self) -> _SerializedPath:
        path = self.path
        return self._serialize_path(path)

    @classmethod
    def deserialize(cls, path: _SerializedPath) -> PathRegistry:
        assert path is not None
        p = cls._deserialize_path(path)
        return cls.coerce(p)

    @overload
    @classmethod
    def per_mapper(cls, mapper: Mapper[Any]) -> CachingEntityRegistry: ...

    @overload
    @classmethod
    def per_mapper(cls, mapper: AliasedInsp[Any]) -> SlotsEntityRegistry: ...

    @classmethod
    def per_mapper(
        cls, mapper: _InternalEntityType[Any]
    ) -> AbstractEntityRegistry:
        if mapper.is_mapper:
            return CachingEntityRegistry(cls.root, mapper)
        else:
            return SlotsEntityRegistry(cls.root, mapper)

    @classmethod
    def coerce(cls, raw: _PathRepresentation) -> PathRegistry:
        def _red(prev: PathRegistry, next_: _PathElementType) -> PathRegistry:
            return prev[next_]

        # can't quite get mypy to appreciate this one :)
        return reduce(_red, raw, cls.root)  # type: ignore

    def __add__(self, other: PathRegistry) -> PathRegistry:
        def _red(prev: PathRegistry, next_: _PathElementType) -> PathRegistry:
            return prev[next_]

        return reduce(_red, other.path, self)

    def __str__(self) -> str:
        return f"ORM Path[{' -> '.join(str(elem) for elem in self.path)}]"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path!r})"


class CreatesToken(PathRegistry):
    __slots__ = ()

    is_aliased_class: bool
    is_root: bool

    def token(self, token: _StrPathToken) -> TokenRegistry:
        if token.endswith(f":{_WILDCARD_TOKEN}"):
            return TokenRegistry(self, token)
        elif token.endswith(f":{_DEFAULT_TOKEN}"):
            return TokenRegistry(self.root, token)
        else:
            raise exc.ArgumentError(f"invalid token: {token}")


class RootRegistry(CreatesToken):
    """Root registry, defers to mappers so that
    paths are maintained per-root-mapper.

    """

    __slots__ = ()

    inherit_cache = True

    path = natural_path = ()
    has_entity = False
    is_aliased_class = False
    is_root = True
    is_unnatural = False

    def _getitem(
        self, entity: Any
    ) -> Union[TokenRegistry, AbstractEntityRegistry]:
        if entity in PathToken._intern:
            if TYPE_CHECKING:
                assert isinstance(entity, _StrPathToken)
            return TokenRegistry(self, PathToken._intern[entity])
        else:
            try:
                return entity._path_registry  # type: ignore
            except AttributeError:
                raise IndexError(
                    f"invalid argument for RootRegistry.__getitem__: {entity}"
                )

    def _truncate_recursive(self) -> RootRegistry:
        return self

    if not TYPE_CHECKING:
        __getitem__ = _getitem


PathRegistry.root = RootRegistry()


class PathToken(orm_base.InspectionAttr, HasCacheKey, str):
    """cacheable string token"""

    _intern: Dict[str, PathToken] = {}

    def _gen_cache_key(
        self, anon_map: anon_map, bindparams: List[BindParameter[Any]]
    ) -> Tuple[Any, ...]:
        return (str(self),)

    @property
    def _path_for_compare(self) -> Optional[_PathRepresentation]:
        return None

    @classmethod
    def intern(cls, strvalue: str) -> PathToken:
        if strvalue in cls._intern:
            return cls._intern[strvalue]
        else:
            cls._intern[strvalue] = result = PathToken(strvalue)
            return result


class TokenRegistry(PathRegistry):
    __slots__ = ("token", "parent", "path", "natural_path")

    inherit_cache = True

    token: _StrPathToken
    parent: CreatesToken

    def __init__(self, parent: CreatesToken, token: _StrPathToken):
        token = PathToken.intern(token)

        self.token = token
        self.parent = parent
        self.path = parent.path + (token,)
        self.natural_path = parent.natural_path + (token,)

    has_entity = False

    is_token = True

    def generate_for_superclasses(self) -> Iterator[PathRegistry]:
        # NOTE: this method is no longer used.  consider removal
        parent = self.parent
        if is_root(parent):
            yield self
            return

        if TYPE_CHECKING:
            assert isinstance(parent, AbstractEntityRegistry)
        if not parent.is_aliased_class:
            for mp_ent in parent.mapper.iterate_to_root():
                yield TokenRegistry(parent.parent[mp_ent], self.token)
        elif (
            parent.is_aliased_class
            and cast(
                "AliasedInsp[Any]",
                parent.entity,
            )._is_with_polymorphic
        ):
            yield self
            for ent in cast(
                "AliasedInsp[Any]", parent.entity
            )._with_polymorphic_entities:
                yield TokenRegistry(parent.parent[ent], self.token)
        else:
            yield self

    def _generate_natural_for_superclasses(
        self,
    ) -> Iterator[_PathRepresentation]:
        parent = self.parent
        if is_root(parent):
            yield self.natural_path
            return

        if TYPE_CHECKING:
            assert isinstance(parent, AbstractEntityRegistry)
        for mp_ent in parent.mapper.iterate_to_root():
            yield TokenRegistry(parent.parent[mp_ent], self.token).natural_path
        if (
            parent.is_aliased_class
            and cast(
                "AliasedInsp[Any]",
                parent.entity,
            )._is_with_polymorphic
        ):
            yield self.natural_path
            for ent in cast(
                "AliasedInsp[Any]", parent.entity
            )._with_polymorphic_entities:
                yield (
                    TokenRegistry(parent.parent[ent], self.token).natural_path
                )
        else:
            yield self.natural_path

    def _getitem(self, entity: Any) -> Any:
        try:
            return self.path[entity]
        except TypeError as err:
            raise IndexError(f"{entity}") from err

    if not TYPE_CHECKING:
        __getitem__ = _getitem


class PropRegistry(PathRegistry):
    __slots__ = (
        "prop",
        "parent",
        "path",
        "natural_path",
        "has_entity",
        "entity",
        "mapper",
        "_wildcard_path_loader_key",
        "_default_path_loader_key",
        "_loader_key",
        "is_unnatural",
    )
    inherit_cache = True
    is_property = True

    prop: StrategizedProperty[Any]
    mapper: Optional[Mapper[Any]]
    entity: Optional[_InternalEntityType[Any]]

    def __init__(
        self, parent: AbstractEntityRegistry, prop: StrategizedProperty[Any]
    ):

        # restate this path in terms of the
        # given StrategizedProperty's parent.
        insp = cast("_InternalEntityType[Any]", parent[-1])
        natural_parent: AbstractEntityRegistry = parent

        # inherit "is_unnatural" from the parent
        self.is_unnatural = parent.parent.is_unnatural or bool(
            parent.mapper.inherits
        )

        if not insp.is_aliased_class or insp._use_mapper_path:  # type: ignore
            parent = natural_parent = parent.parent[prop.parent]
        elif (
            insp.is_aliased_class
            and insp.with_polymorphic_mappers
            and prop.parent in insp.with_polymorphic_mappers
        ):
            subclass_entity: _InternalEntityType[Any] = parent[-1]._entity_for_mapper(prop.parent)  # type: ignore  # noqa: E501
            parent = parent.parent[subclass_entity]

            # when building a path where with_polymorphic() is in use,
            # special logic to determine the "natural path" when subclass
            # entities are used.
            #
            # here we are trying to distinguish between a path that starts
            # on a with_polymorphic entity vs. one that starts on a
            # normal entity that introduces a with_polymorphic() in the
            # middle using of_type():
            #
            #  # as in test_polymorphic_rel->
            #  #    test_subqueryload_on_subclass_uses_path_correctly
            #  wp = with_polymorphic(RegularEntity, "*")
            #  sess.query(wp).options(someload(wp.SomeSubEntity.foos))
            #
            # vs
            #
            #  # as in test_relationship->JoinedloadWPolyOfTypeContinued
            #  wp = with_polymorphic(SomeFoo, "*")
            #  sess.query(RegularEntity).options(
            #       someload(RegularEntity.foos.of_type(wp))
            #       .someload(wp.SubFoo.bar)
            #   )
            #
            # in the former case, the Query as it generates a path that we
            # want to match will be in terms of the with_polymorphic at the
            # beginning.  in the latter case, Query will generate simple
            # paths that don't know about this with_polymorphic, so we must
            # use a separate natural path.
            #
            #
            if parent.parent:
                natural_parent = parent.parent[subclass_entity.mapper]
                self.is_unnatural = True
            else:
                natural_parent = parent
        elif (
            natural_parent.parent
            and insp.is_aliased_class
            and prop.parent  # this should always be the case here
            is not insp.mapper
            and insp.mapper.isa(prop.parent)
        ):
            natural_parent = parent.parent[prop.parent]

        self.prop = prop
        self.parent = parent
        self.path = parent.path + (prop,)
        self.natural_path = natural_parent.natural_path + (prop,)

        self.has_entity = prop._links_to_entity
        if prop._is_relationship:
            if TYPE_CHECKING:
                assert isinstance(prop, RelationshipProperty)
            self.entity = prop.entity
            self.mapper = prop.mapper
        else:
            self.entity = None
            self.mapper = None

        self._wildcard_path_loader_key = (
            "loader",
            parent.natural_path + self.prop._wildcard_token,
        )
        self._default_path_loader_key = self.prop._default_path_loader_key
        self._loader_key = ("loader", self.natural_path)

    def _truncate_recursive(self) -> PropRegistry:
        earliest = None
        for i, token in enumerate(reversed(self.path[:-1])):
            if token is self.prop:
                earliest = i

        if earliest is None:
            return self
        else:
            return self.coerce(self.path[0 : -(earliest + 1)])  # type: ignore

    @property
    def entity_path(self) -> AbstractEntityRegistry:
        assert self.entity is not None
        return self[self.entity]

    def _getitem(
        self, entity: Union[int, slice, _InternalEntityType[Any]]
    ) -> Union[AbstractEntityRegistry, _PathElementType, _PathRepresentation]:
        if isinstance(entity, (int, slice)):
            return self.path[entity]
        else:
            return SlotsEntityRegistry(self, entity)

    if not TYPE_CHECKING:
        __getitem__ = _getitem


class AbstractEntityRegistry(CreatesToken):
    __slots__ = (
        "key",
        "parent",
        "is_aliased_class",
        "path",
        "entity",
        "natural_path",
    )

    has_entity = True
    is_entity = True

    parent: Union[RootRegistry, PropRegistry]
    key: _InternalEntityType[Any]
    entity: _InternalEntityType[Any]
    is_aliased_class: bool

    def __init__(
        self,
        parent: Union[RootRegistry, PropRegistry],
        entity: _InternalEntityType[Any],
    ):
        self.key = entity
        self.parent = parent
        self.is_aliased_class = entity.is_aliased_class
        self.entity = entity
        self.path = parent.path + (entity,)

        # the "natural path" is the path that we get when Query is traversing
        # from the lead entities into the various relationships; it corresponds
        # to the structure of mappers and relationships. when we are given a
        # path that comes from loader options, as of 1.3 it can have ac-hoc
        # with_polymorphic() and other AliasedInsp objects inside of it, which
        # are usually not present in mappings.  So here we track both the
        # "enhanced" path in self.path and the "natural" path that doesn't
        # include those objects so these two traversals can be matched up.

        # the test here for "(self.is_aliased_class or parent.is_unnatural)"
        # are to avoid the more expensive conditional logic that follows if we
        # know we don't have to do it.   This conditional can just as well be
        # "if parent.path:", it just is more function calls.
        #
        # This is basically the only place that the "is_unnatural" flag
        # actually changes behavior.
        if parent.path and (self.is_aliased_class or parent.is_unnatural):
            # this is an infrequent code path used only for loader strategies
            # that also make use of of_type().
            if entity.mapper.isa(parent.natural_path[-1].mapper):  # type: ignore # noqa: E501
                self.natural_path = parent.natural_path + (entity.mapper,)
            else:
                self.natural_path = parent.natural_path + (
                    parent.natural_path[-1].entity,  # type: ignore
                )
        # it seems to make sense that since these paths get mixed up
        # with statements that are cached or not, we should make
        # sure the natural path is cacheable across different occurrences
        # of equivalent AliasedClass objects.  however, so far this
        # does not seem to be needed for whatever reason.
        # elif not parent.path and self.is_aliased_class:
        #     self.natural_path = (self.entity._generate_cache_key()[0], )
        else:
            self.natural_path = self.path

    def _truncate_recursive(self) -> AbstractEntityRegistry:
        return self.parent._truncate_recursive()[self.entity]

    @property
    def root_entity(self) -> _InternalEntityType[Any]:
        return self.odd_element(0)

    @property
    def entity_path(self) -> PathRegistry:
        return self

    @property
    def mapper(self) -> Mapper[Any]:
        return self.entity.mapper

    def __bool__(self) -> bool:
        return True

    def _getitem(
        self, entity: Any
    ) -> Union[_PathElementType, _PathRepresentation, PathRegistry]:
        if isinstance(entity, (int, slice)):
            return self.path[entity]
        elif entity in PathToken._intern:
            return TokenRegistry(self, PathToken._intern[entity])
        else:
            return PropRegistry(self, entity)

    if not TYPE_CHECKING:
        __getitem__ = _getitem


class SlotsEntityRegistry(AbstractEntityRegistry):
    # for aliased class, return lightweight, no-cycles created
    # version
    inherit_cache = True


class _ERDict(Dict[Any, Any]):
    def __init__(self, registry: CachingEntityRegistry):
        self.registry = registry

    def __missing__(self, key: Any) -> PropRegistry:
        self[key] = item = PropRegistry(self.registry, key)

        return item


class CachingEntityRegistry(AbstractEntityRegistry):
    # for long lived mapper, return dict based caching
    # version that creates reference cycles

    __slots__ = ("_cache",)

    inherit_cache = True

    def __init__(
        self,
        parent: Union[RootRegistry, PropRegistry],
        entity: _InternalEntityType[Any],
    ):
        super().__init__(parent, entity)
        self._cache = _ERDict(self)

    def pop(self, key: Any, default: Any) -> Any:
        return self._cache.pop(key, default)

    def _getitem(self, entity: Any) -> Any:
        if isinstance(entity, (int, slice)):
            return self.path[entity]
        elif isinstance(entity, PathToken):
            return TokenRegistry(self, entity)
        else:
            return self._cache[entity]

    if not TYPE_CHECKING:
        __getitem__ = _getitem


if TYPE_CHECKING:

    def path_is_entity(
        path: PathRegistry,
    ) -> TypeGuard[AbstractEntityRegistry]: ...

    def path_is_property(path: PathRegistry) -> TypeGuard[PropRegistry]: ...

else:
    path_is_entity = operator.attrgetter("is_entity")
    path_is_property = operator.attrgetter("is_property")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\qubit.py ===
"""Qubits for quantum computing.

Todo:
* Finish implementing measurement logic. This should include POVM.
* Update docstrings.
* Update tests.
"""


import math

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import log
from sympy.core.basic import _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import Matrix, zeros
from sympy.printing.pretty.stringpict import prettyForm

from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.state import Ket, Bra, State

from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import (
    numpy_ndarray, scipy_sparse_matrix
)
from mpmath.libmp.libintmath import bitcount

__all__ = [
    'Qubit',
    'QubitBra',
    'IntQubit',
    'IntQubitBra',
    'qubit_to_matrix',
    'matrix_to_qubit',
    'matrix_to_density',
    'measure_all',
    'measure_partial',
    'measure_partial_oneshot',
    'measure_all_oneshot'
]

#-----------------------------------------------------------------------------
# Qubit Classes
#-----------------------------------------------------------------------------


class QubitState(State):
    """Base class for Qubit and QubitBra."""

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        # If we are passed a QubitState or subclass, we just take its qubit
        # values directly.
        if len(args) == 1 and isinstance(args[0], QubitState):
            return args[0].qubit_values

        # Turn strings into tuple of strings
        if len(args) == 1 and isinstance(args[0], str):
            args = tuple( S.Zero if qb == "0" else S.One for qb in args[0])
        else:
            args = tuple( S.Zero if qb == "0" else S.One if qb == "1" else qb for qb in args)
        args = tuple(_sympify(arg) for arg in args)

        # Validate input (must have 0 or 1 input)
        for element in args:
            if element not in (S.Zero, S.One):
                raise ValueError(
                    "Qubit values must be 0 or 1, got: %r" % element)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        return ComplexSpace(2)**len(args)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def dimension(self):
        """The number of Qubits in the state."""
        return len(self.qubit_values)

    @property
    def nqubits(self):
        return self.dimension

    @property
    def qubit_values(self):
        """Returns the values of the qubits as a tuple."""
        return self.label

    #-------------------------------------------------------------------------
    # Special methods
    #-------------------------------------------------------------------------

    def __len__(self):
        return self.dimension

    def __getitem__(self, bit):
        return self.qubit_values[int(self.dimension - bit - 1)]

    #-------------------------------------------------------------------------
    # Utility methods
    #-------------------------------------------------------------------------

    def flip(self, *bits):
        """Flip the bit(s) given."""
        newargs = list(self.qubit_values)
        for i in bits:
            bit = int(self.dimension - i - 1)
            if newargs[bit] == 1:
                newargs[bit] = 0
            else:
                newargs[bit] = 1
        return self.__class__(*tuple(newargs))


class Qubit(QubitState, Ket):
    """A multi-qubit ket in the computational (z) basis.

    We use the normal convention that the least significant qubit is on the
    right, so ``|00001>`` has a 1 in the least significant qubit.

    Parameters
    ==========

    values : list, str
        The qubit values as a list of ints ([0,0,0,1,1,]) or a string ('011').

    Examples
    ========

    Create a qubit in a couple of different ways and look at their attributes:

        >>> from sympy.physics.quantum.qubit import Qubit
        >>> Qubit(0,0,0)
        |000>
        >>> q = Qubit('0101')
        >>> q
        |0101>

        >>> q.nqubits
        4
        >>> len(q)
        4
        >>> q.dimension
        4
        >>> q.qubit_values
        (0, 1, 0, 1)

    We can flip the value of an individual qubit:

        >>> q.flip(1)
        |0111>

    We can take the dagger of a Qubit to get a bra:

        >>> from sympy.physics.quantum.dagger import Dagger
        >>> Dagger(q)
        <0101|
        >>> type(Dagger(q))
        <class 'sympy.physics.quantum.qubit.QubitBra'>

    Inner products work as expected:

        >>> ip = Dagger(q)*q
        >>> ip
        <0101|0101>
        >>> ip.doit()
        1
    """

    @classmethod
    def dual_class(self):
        return QubitBra

    def _eval_innerproduct_QubitBra(self, bra, **hints):
        if self.label == bra.label:
            return S.One
        else:
            return S.Zero

    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        """Represent this qubits in the computational basis (ZGate).
        """
        _format = options.get('format', 'sympy')
        n = 1
        definite_state = 0
        for it in reversed(self.qubit_values):
            definite_state += n*it
            n = n*2
        result = [0]*(2**self.dimension)
        result[int(definite_state)] = 1
        if _format == 'sympy':
            return Matrix(result)
        elif _format == 'numpy':
            import numpy as np
            return np.array(result, dtype='complex').transpose()
        elif _format == 'scipy.sparse':
            from scipy import sparse
            return sparse.csr_matrix(result, dtype='complex').transpose()

    def _eval_trace(self, bra, **kwargs):
        indices = kwargs.get('indices', [])

        #sort index list to begin trace from most-significant
        #qubit
        sorted_idx = list(indices)
        if len(sorted_idx) == 0:
            sorted_idx = list(range(0, self.nqubits))
        sorted_idx.sort()

        #trace out for each of index
        new_mat = self*bra
        for i in range(len(sorted_idx) - 1, -1, -1):
            # start from tracing out from leftmost qubit
            new_mat = self._reduced_density(new_mat, int(sorted_idx[i]))

        if (len(sorted_idx) == self.nqubits):
            #in case full trace was requested
            return new_mat[0]
        else:
            return matrix_to_density(new_mat)

    def _reduced_density(self, matrix, qubit, **options):
        """Compute the reduced density matrix by tracing out one qubit.
           The qubit argument should be of type Python int, since it is used
           in bit operations
        """
        def find_index_that_is_projected(j, k, qubit):
            bit_mask = 2**qubit - 1
            return ((j >> qubit) << (1 + qubit)) + (j & bit_mask) + (k << qubit)

        old_matrix = represent(matrix, **options)
        old_size = old_matrix.cols
        #we expect the old_size to be even
        new_size = old_size//2
        new_matrix = Matrix().zeros(new_size)

        for i in range(new_size):
            for j in range(new_size):
                for k in range(2):
                    col = find_index_that_is_projected(j, k, qubit)
                    row = find_index_that_is_projected(i, k, qubit)
                    new_matrix[i, j] += old_matrix[row, col]

        return new_matrix


class QubitBra(QubitState, Bra):
    """A multi-qubit bra in the computational (z) basis.

    We use the normal convention that the least significant qubit is on the
    right, so ``|00001>`` has a 1 in the least significant qubit.

    Parameters
    ==========

    values : list, str
        The qubit values as a list of ints ([0,0,0,1,1,]) or a string ('011').

    See also
    ========

    Qubit: Examples using qubits

    """
    @classmethod
    def dual_class(self):
        return Qubit


class IntQubitState(QubitState):
    """A base class for qubits that work with binary representations."""

    @classmethod
    def _eval_args(cls, args, nqubits=None):
        # The case of a QubitState instance
        if len(args) == 1 and isinstance(args[0], QubitState):
            return QubitState._eval_args(args)
        # otherwise, args should be integer
        elif not all(isinstance(a, (int, Integer)) for a in args):
            raise ValueError('values must be integers, got (%s)' % (tuple(type(a) for a in args),))
        # use nqubits if specified
        if nqubits is not None:
            if not isinstance(nqubits, (int, Integer)):
                raise ValueError('nqubits must be an integer, got (%s)' % type(nqubits))
            if len(args) != 1:
                raise ValueError(
                    'too many positional arguments (%s). should be (number, nqubits=n)' % (args,))
            return cls._eval_args_with_nqubits(args[0], nqubits)
        # For a single argument, we construct the binary representation of
        # that integer with the minimal number of bits.
        if len(args) == 1 and args[0] > 1:
            #rvalues is the minimum number of bits needed to express the number
            rvalues = reversed(range(bitcount(abs(args[0]))))
            qubit_values = [(args[0] >> i) & 1 for i in rvalues]
            return QubitState._eval_args(qubit_values)
        # For two numbers, the second number is the number of bits
        # on which it is expressed, so IntQubit(0,5) == |00000>.
        elif len(args) == 2 and args[1] > 1:
            return cls._eval_args_with_nqubits(args[0], args[1])
        else:
            return QubitState._eval_args(args)

    @classmethod
    def _eval_args_with_nqubits(cls, number, nqubits):
        need = bitcount(abs(number))
        if nqubits < need:
            raise ValueError(
                'cannot represent %s with %s bits' % (number, nqubits))
        qubit_values = [(number >> i) & 1 for i in reversed(range(nqubits))]
        return QubitState._eval_args(qubit_values)

    def as_int(self):
        """Return the numerical value of the qubit."""
        number = 0
        n = 1
        for i in reversed(self.qubit_values):
            number += n*i
            n = n << 1
        return number

    def _print_label(self, printer, *args):
        return str(self.as_int())

    def _print_label_pretty(self, printer, *args):
        label = self._print_label(printer, *args)
        return prettyForm(label)

    _print_label_repr = _print_label
    _print_label_latex = _print_label


class IntQubit(IntQubitState, Qubit):
    """A qubit ket that store integers as binary numbers in qubit values.

    The differences between this class and ``Qubit`` are:

    * The form of the constructor.
    * The qubit values are printed as their corresponding integer, rather
      than the raw qubit values. The internal storage format of the qubit
      values in the same as ``Qubit``.

    Parameters
    ==========

    values : int, tuple
        If a single argument, the integer we want to represent in the qubit
        values. This integer will be represented using the fewest possible
        number of qubits.
        If a pair of integers and the second value is more than one, the first
        integer gives the integer to represent in binary form and the second
        integer gives the number of qubits to use.
        List of zeros and ones is also accepted to generate qubit by bit pattern.

    nqubits : int
        The integer that represents the number of qubits.
        This number should be passed with keyword ``nqubits=N``.
        You can use this in order to avoid ambiguity of Qubit-style tuple of bits.
        Please see the example below for more details.

    Examples
    ========

    Create a qubit for the integer 5:

        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.qubit import Qubit
        >>> q = IntQubit(5)
        >>> q
        |5>

    We can also create an ``IntQubit`` by passing a ``Qubit`` instance.

        >>> q = IntQubit(Qubit('101'))
        >>> q
        |5>
        >>> q.as_int()
        5
        >>> q.nqubits
        3
        >>> q.qubit_values
        (1, 0, 1)

    We can go back to the regular qubit form.

        >>> Qubit(q)
        |101>

    Please note that ``IntQubit`` also accepts a ``Qubit``-style list of bits.
    So, the code below yields qubits 3, not a single bit ``1``.

        >>> IntQubit(1, 1)
        |3>

    To avoid ambiguity, use ``nqubits`` parameter.
    Use of this keyword is recommended especially when you provide the values by variables.

        >>> IntQubit(1, nqubits=1)
        |1>
        >>> a = 1
        >>> IntQubit(a, nqubits=1)
        |1>
    """
    @classmethod
    def dual_class(self):
        return IntQubitBra

    def _eval_innerproduct_IntQubitBra(self, bra, **hints):
        return Qubit._eval_innerproduct_QubitBra(self, bra)

class IntQubitBra(IntQubitState, QubitBra):
    """A qubit bra that store integers as binary numbers in qubit values."""

    @classmethod
    def dual_class(self):
        return IntQubit


#-----------------------------------------------------------------------------
# Qubit <---> Matrix conversion functions
#-----------------------------------------------------------------------------


def matrix_to_qubit(matrix):
    """Convert from the matrix repr. to a sum of Qubit objects.

    Parameters
    ----------
    matrix : Matrix, numpy.matrix, scipy.sparse
        The matrix to build the Qubit representation of. This works with
        SymPy matrices, numpy matrices and scipy.sparse sparse matrices.

    Examples
    ========

    Represent a state and then go back to its qubit form:

        >>> from sympy.physics.quantum.qubit import matrix_to_qubit, Qubit
        >>> from sympy.physics.quantum.represent import represent
        >>> q = Qubit('01')
        >>> matrix_to_qubit(represent(q))
        |01>
    """
    # Determine the format based on the type of the input matrix
    format = 'sympy'
    if isinstance(matrix, numpy_ndarray):
        format = 'numpy'
    if isinstance(matrix, scipy_sparse_matrix):
        format = 'scipy.sparse'

    # Make sure it is of correct dimensions for a Qubit-matrix representation.
    # This logic should work with sympy, numpy or scipy.sparse matrices.
    if matrix.shape[0] == 1:
        mlistlen = matrix.shape[1]
        nqubits = log(mlistlen, 2)
        ket = False
        cls = QubitBra
    elif matrix.shape[1] == 1:
        mlistlen = matrix.shape[0]
        nqubits = log(mlistlen, 2)
        ket = True
        cls = Qubit
    else:
        raise QuantumError(
            'Matrix must be a row/column vector, got %r' % matrix
        )
    if not isinstance(nqubits, Integer):
        raise QuantumError('Matrix must be a row/column vector of size '
                           '2**nqubits, got: %r' % matrix)
    # Go through each item in matrix, if element is non-zero, make it into a
    # Qubit item times the element.
    result = 0
    for i in range(mlistlen):
        if ket:
            element = matrix[i, 0]
        else:
            element = matrix[0, i]
        if format in ('numpy', 'scipy.sparse'):
            element = complex(element)
        if element:
            # Form Qubit array; 0 in bit-locations where i is 0, 1 in
            # bit-locations where i is 1
            qubit_array = [int(i & (1 << x) != 0) for x in range(nqubits)]
            qubit_array.reverse()
            result = result + element*cls(*qubit_array)

    # If SymPy simplified by pulling out a constant coefficient, undo that.
    if isinstance(result, (Mul, Add, Pow)):
        result = result.expand()

    return result


def matrix_to_density(mat):
    """
    Works by finding the eigenvectors and eigenvalues of the matrix.
    We know we can decompose rho by doing:
    sum(EigenVal*|Eigenvect><Eigenvect|)
    """
    from sympy.physics.quantum.density import Density
    eigen = mat.eigenvects()
    args = [[matrix_to_qubit(Matrix(
        [vector, ])), x[0]] for x in eigen for vector in x[2] if x[0] != 0]
    if (len(args) == 0):
        return S.Zero
    else:
        return Density(*args)


def qubit_to_matrix(qubit, format='sympy'):
    """Converts an Add/Mul of Qubit objects into it's matrix representation

    This function is the inverse of ``matrix_to_qubit`` and is a shorthand
    for ``represent(qubit)``.
    """
    return represent(qubit, format=format)


#-----------------------------------------------------------------------------
# Measurement
#-----------------------------------------------------------------------------


def measure_all(qubit, format='sympy', normalize=True):
    """Perform an ensemble measurement of all qubits.

    Parameters
    ==========

    qubit : Qubit, Add
        The qubit to measure. This can be any Qubit or a linear combination
        of them.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    =======

    result : list
        A list that consists of primitive states and their probabilities.

    Examples
    ========

        >>> from sympy.physics.quantum.qubit import Qubit, measure_all
        >>> from sympy.physics.quantum.gate import H
        >>> from sympy.physics.quantum.qapply import qapply

        >>> c = H(0)*H(1)*Qubit('00')
        >>> c
        H(0)*H(1)*|00>
        >>> q = qapply(c)
        >>> measure_all(q)
        [(|00>, 1/4), (|01>, 1/4), (|10>, 1/4), (|11>, 1/4)]
    """
    m = qubit_to_matrix(qubit, format)

    if format == 'sympy':
        results = []

        if normalize:
            m = m.normalized()

        size = max(m.shape)  # Max of shape to account for bra or ket
        nqubits = int(math.log(size)/math.log(2))
        for i in range(size):
            if m[i]:
                results.append(
                    (Qubit(IntQubit(i, nqubits=nqubits)), m[i]*conjugate(m[i]))
                )
        return results
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


def measure_partial(qubit, bits, format='sympy', normalize=True):
    """Perform a partial ensemble measure on the specified qubits.

    Parameters
    ==========

    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    bits : tuple
        The qubits to measure.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    =======

    result : list
        A list that consists of primitive states and their probabilities.

    Examples
    ========

        >>> from sympy.physics.quantum.qubit import Qubit, measure_partial
        >>> from sympy.physics.quantum.gate import H
        >>> from sympy.physics.quantum.qapply import qapply

        >>> c = H(0)*H(1)*Qubit('00')
        >>> c
        H(0)*H(1)*|00>
        >>> q = qapply(c)
        >>> measure_partial(q, (0,))
        [(sqrt(2)*|00>/2 + sqrt(2)*|10>/2, 1/2), (sqrt(2)*|01>/2 + sqrt(2)*|11>/2, 1/2)]
    """
    m = qubit_to_matrix(qubit, format)

    if isinstance(bits, (SYMPY_INTS, Integer)):
        bits = (int(bits),)

    if format == 'sympy':
        if normalize:
            m = m.normalized()

        possible_outcomes = _get_possible_outcomes(m, bits)

        # Form output from function.
        output = []
        for outcome in possible_outcomes:
            # Calculate probability of finding the specified bits with
            # given values.
            prob_of_outcome = 0
            prob_of_outcome += (outcome.H*outcome)[0]

            # If the output has a chance, append it to output with found
            # probability.
            if prob_of_outcome != 0:
                if normalize:
                    next_matrix = matrix_to_qubit(outcome.normalized())
                else:
                    next_matrix = matrix_to_qubit(outcome)

                output.append((
                    next_matrix,
                    prob_of_outcome
                ))

        return output
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


def measure_partial_oneshot(qubit, bits, format='sympy'):
    """Perform a partial oneshot measurement on the specified qubits.

    A oneshot measurement is equivalent to performing a measurement on a
    quantum system. This type of measurement does not return the probabilities
    like an ensemble measurement does, but rather returns *one* of the
    possible resulting states. The exact state that is returned is determined
    by picking a state randomly according to the ensemble probabilities.

    Parameters
    ----------
    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    bits : tuple
        The qubits to measure.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    -------
    result : Qubit
        The qubit that the system collapsed to upon measurement.
    """
    import random
    m = qubit_to_matrix(qubit, format)

    if format == 'sympy':
        m = m.normalized()
        possible_outcomes = _get_possible_outcomes(m, bits)

        # Form output from function
        random_number = random.random()
        total_prob = 0
        for outcome in possible_outcomes:
            # Calculate probability of finding the specified bits
            # with given values
            total_prob += (outcome.H*outcome)[0]
            if total_prob >= random_number:
                return matrix_to_qubit(outcome.normalized())
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )


def _get_possible_outcomes(m, bits):
    """Get the possible states that can be produced in a measurement.

    Parameters
    ----------
    m : Matrix
        The matrix representing the state of the system.
    bits : tuple, list
        Which bits will be measured.

    Returns
    -------
    result : list
        The list of possible states which can occur given this measurement.
        These are un-normalized so we can derive the probability of finding
        this state by taking the inner product with itself
    """

    # This is filled with loads of dirty binary tricks...You have been warned

    size = max(m.shape)  # Max of shape to account for bra or ket
    nqubits = int(math.log2(size) + .1)  # Number of qubits possible

    # Make the output states and put in output_matrices, nothing in them now.
    # Each state will represent a possible outcome of the measurement
    # Thus, output_matrices[0] is the matrix which we get when all measured
    # bits return 0. and output_matrices[1] is the matrix for only the 0th
    # bit being true
    output_matrices = []
    for i in range(1 << len(bits)):
        output_matrices.append(zeros(2**nqubits, 1))

    # Bitmasks will help sort how to determine possible outcomes.
    # When the bit mask is and-ed with a matrix-index,
    # it will determine which state that index belongs to
    bit_masks = []
    for bit in bits:
        bit_masks.append(1 << bit)

    # Make possible outcome states
    for i in range(2**nqubits):
        trueness = 0  # This tells us to which output_matrix this value belongs
        # Find trueness
        for j in range(len(bit_masks)):
            if i & bit_masks[j]:
                trueness += j + 1
        # Put the value in the correct output matrix
        output_matrices[trueness][i] = m[i]
    return output_matrices


def measure_all_oneshot(qubit, format='sympy'):
    """Perform a oneshot ensemble measurement on all qubits.

    A oneshot measurement is equivalent to performing a measurement on a
    quantum system. This type of measurement does not return the probabilities
    like an ensemble measurement does, but rather returns *one* of the
    possible resulting states. The exact state that is returned is determined
    by picking a state randomly according to the ensemble probabilities.

    Parameters
    ----------
    qubits : Qubit
        The qubit to measure.  This can be any Qubit or a linear combination
        of them.
    format : str
        The format of the intermediate matrices to use. Possible values are
        ('sympy','numpy','scipy.sparse'). Currently only 'sympy' is
        implemented.

    Returns
    -------
    result : Qubit
        The qubit that the system collapsed to upon measurement.
    """
    import random
    m = qubit_to_matrix(qubit)

    if format == 'sympy':
        m = m.normalized()
        random_number = random.random()
        total = 0
        result = 0
        for i in m:
            total += i*i.conjugate()
            if total > random_number:
                break
            result += 1
        return Qubit(IntQubit(result, nqubits=int(math.log2(max(m.shape)) + .1)))
    else:
        raise NotImplementedError(
            "This function cannot handle non-SymPy matrix formats yet"
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\fourier.py ===
"""Fourier Series"""

from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence


__doctest_requires__ = {('fourier_series',): ['matplotlib']}


def fourier_cos_seq(func, limits, n):
    """Returns the cos sequence in a Fourier series"""
    from sympy.integrals import integrate
    x, L = limits[0], limits[2] - limits[1]
    cos_term = cos(2*n*pi*x / L)
    formula = 2 * cos_term * integrate(func * cos_term, limits) / L
    a0 = formula.subs(n, S.Zero) / 2
    return a0, SeqFormula(2 * cos_term * integrate(func * cos_term, limits)
                          / L, (n, 1, oo))


def fourier_sin_seq(func, limits, n):
    """Returns the sin sequence in a Fourier series"""
    from sympy.integrals import integrate
    x, L = limits[0], limits[2] - limits[1]
    sin_term = sin(2*n*pi*x / L)
    return SeqFormula(2 * sin_term * integrate(func * sin_term, limits)
                      / L, (n, 1, oo))


def _process_limits(func, limits):
    """
    Limits should be of the form (x, start, stop).
    x should be a symbol. Both start and stop should be bounded.

    Explanation
    ===========

    * If x is not given, x is determined from func.
    * If limits is None. Limit of the form (x, -pi, pi) is returned.

    Examples
    ========

    >>> from sympy.series.fourier import _process_limits as pari
    >>> from sympy.abc import x
    >>> pari(x**2, (x, -2, 2))
    (x, -2, 2)
    >>> pari(x**2, (-2, 2))
    (x, -2, 2)
    >>> pari(x**2, None)
    (x, -pi, pi)
    """
    def _find_x(func):
        free = func.free_symbols
        if len(free) == 1:
            return free.pop()
        elif not free:
            return Dummy('k')
        else:
            raise ValueError(
                " specify dummy variables for %s. If the function contains"
                " more than one free symbol, a dummy variable should be"
                " supplied explicitly e.g. FourierSeries(m*n**2, (n, -pi, pi))"
                % func)

    x, start, stop = None, None, None
    if limits is None:
        x, start, stop = _find_x(func), -pi, pi
    if is_sequence(limits, Tuple):
        if len(limits) == 3:
            x, start, stop = limits
        elif len(limits) == 2:
            x = _find_x(func)
            start, stop = limits

    if not isinstance(x, Symbol) or start is None or stop is None:
        raise ValueError('Invalid limits given: %s' % str(limits))

    unbounded = [S.NegativeInfinity, S.Infinity]
    if start in unbounded or stop in unbounded:
        raise ValueError("Both the start and end value should be bounded")

    return sympify((x, start, stop))


def finite_check(f, x, L):

    def check_fx(exprs, x):
        return x not in exprs.free_symbols

    def check_sincos(_expr, x, L):
        if isinstance(_expr, (sin, cos)):
            sincos_args = _expr.args[0]

            if sincos_args.match(a*(pi/L)*x + b) is not None:
                return True
            else:
                return False

    from sympy.simplify.fu import TR2, TR1, sincos_to_sum
    _expr = sincos_to_sum(TR2(TR1(f)))
    add_coeff = _expr.as_coeff_add()

    a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k != S.Zero, ])
    b = Wild('b', properties=[lambda k: x not in k.free_symbols, ])

    for s in add_coeff[1]:
        mul_coeffs = s.as_coeff_mul()[1]
        for t in mul_coeffs:
            if not (check_fx(t, x) or check_sincos(t, x, L)):
                return False, f

    return True, _expr


class FourierSeries(SeriesBase):
    r"""Represents Fourier sine/cosine series.

    Explanation
    ===========

    This class only represents a fourier series.
    No computation is performed.

    For how to compute Fourier series, see the :func:`fourier_series`
    docstring.

    See Also
    ========

    sympy.series.fourier.fourier_series
    """
    def __new__(cls, *args):
        args = map(sympify, args)
        return Expr.__new__(cls, *args)

    @property
    def function(self):
        return self.args[0]

    @property
    def x(self):
        return self.args[1][0]

    @property
    def period(self):
        return (self.args[1][1], self.args[1][2])

    @property
    def a0(self):
        return self.args[2][0]

    @property
    def an(self):
        return self.args[2][1]

    @property
    def bn(self):
        return self.args[2][2]

    @property
    def interval(self):
        return Interval(0, oo)

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return oo

    @property
    def L(self):
        return abs(self.period[1] - self.period[0]) / 2

    def _eval_subs(self, old, new):
        x = self.x
        if old.has(x):
            return self

    def truncate(self, n=3):
        """
        Return the first n nonzero terms of the series.

        If ``n`` is None return an iterator.

        Parameters
        ==========

        n : int or None
            Amount of non-zero terms in approximation or None.

        Returns
        =======

        Expr or iterator :
            Approximation of function expanded into Fourier series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x, (x, -pi, pi))
        >>> s.truncate(4)
        2*sin(x) - sin(2*x) + 2*sin(3*x)/3 - sin(4*x)/2

        See Also
        ========

        sympy.series.fourier.FourierSeries.sigma_approximation
        """
        if n is None:
            return iter(self)

        terms = []
        for t in self:
            if len(terms) == n:
                break
            if t is not S.Zero:
                terms.append(t)

        return Add(*terms)

    def sigma_approximation(self, n=3):
        r"""
        Return :math:`\sigma`-approximation of Fourier series with respect
        to order n.

        Explanation
        ===========

        Sigma approximation adjusts a Fourier summation to eliminate the Gibbs
        phenomenon which would otherwise occur at discontinuities.
        A sigma-approximated summation for a Fourier series of a T-periodical
        function can be written as

        .. math::
            s(\theta) = \frac{1}{2} a_0 + \sum _{k=1}^{m-1}
            \operatorname{sinc} \Bigl( \frac{k}{m} \Bigr) \cdot
            \left[ a_k \cos \Bigl( \frac{2\pi k}{T} \theta \Bigr)
            + b_k \sin \Bigl( \frac{2\pi k}{T} \theta \Bigr) \right],

        where :math:`a_0, a_k, b_k, k=1,\ldots,{m-1}` are standard Fourier
        series coefficients and
        :math:`\operatorname{sinc} \Bigl( \frac{k}{m} \Bigr)` is a Lanczos
        :math:`\sigma` factor (expressed in terms of normalized
        :math:`\operatorname{sinc}` function).

        Parameters
        ==========

        n : int
            Highest order of the terms taken into account in approximation.

        Returns
        =======

        Expr :
            Sigma approximation of function expanded into Fourier series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x, (x, -pi, pi))
        >>> s.sigma_approximation(4)
        2*sin(x)*sinc(pi/4) - 2*sin(2*x)/pi + 2*sin(3*x)*sinc(3*pi/4)/3

        See Also
        ========

        sympy.series.fourier.FourierSeries.truncate

        Notes
        =====

        The behaviour of
        :meth:`~sympy.series.fourier.FourierSeries.sigma_approximation`
        is different from :meth:`~sympy.series.fourier.FourierSeries.truncate`
        - it takes all nonzero terms of degree smaller than n, rather than
        first n nonzero ones.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Gibbs_phenomenon
        .. [2] https://en.wikipedia.org/wiki/Sigma_approximation
        """
        terms = [sinc(pi * i / n) * t for i, t in enumerate(self[:n])
                 if t is not S.Zero]
        return Add(*terms)

    def shift(self, s):
        """
        Shift the function by a term independent of x.

        Explanation
        ===========

        f(x) -> f(x) + s

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.shift(1).truncate()
        -4*cos(x) + cos(2*x) + 1 + pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        a0 = self.a0 + s
        sfunc = self.function + s

        return self.func(sfunc, self.args[1], (a0, self.an, self.bn))

    def shiftx(self, s):
        """
        Shift x by a term independent of x.

        Explanation
        ===========

        f(x) -> f(x + s)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.shiftx(1).truncate()
        -4*cos(x + 1) + cos(2*x + 2) + pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        an = self.an.subs(x, x + s)
        bn = self.bn.subs(x, x + s)
        sfunc = self.function.subs(x, x + s)

        return self.func(sfunc, self.args[1], (self.a0, an, bn))

    def scale(self, s):
        """
        Scale the function by a term independent of x.

        Explanation
        ===========

        f(x) -> s * f(x)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.scale(2).truncate()
        -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        an = self.an.coeff_mul(s)
        bn = self.bn.coeff_mul(s)
        a0 = self.a0 * s
        sfunc = self.args[0] * s

        return self.func(sfunc, self.args[1], (a0, an, bn))

    def scalex(self, s):
        """
        Scale x by a term independent of x.

        Explanation
        ===========

        f(x) -> f(s*x)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.scalex(2).truncate()
        -4*cos(2*x) + cos(4*x) + pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        an = self.an.subs(x, x * s)
        bn = self.bn.subs(x, x * s)
        sfunc = self.function.subs(x, x * s)

        return self.func(sfunc, self.args[1], (self.a0, an, bn))

    def _eval_as_leading_term(self, x, logx, cdir):
        for t in self:
            if t is not S.Zero:
                return t

    def _eval_term(self, pt):
        if pt == 0:
            return self.a0
        return self.an.coeff(pt) + self.bn.coeff(pt)

    def __neg__(self):
        return self.scale(-1)

    def __add__(self, other):
        if isinstance(other, FourierSeries):
            if self.period != other.period:
                raise ValueError("Both the series should have same periods")

            x, y = self.x, other.x
            function = self.function + other.function.subs(y, x)

            if self.x not in function.free_symbols:
                return function

            an = self.an + other.an
            bn = self.bn + other.bn
            a0 = self.a0 + other.a0

            return self.func(function, self.args[1], (a0, an, bn))

        return Add(self, other)

    def __sub__(self, other):
        return self.__add__(-other)


class FiniteFourierSeries(FourierSeries):
    r"""Represents Finite Fourier sine/cosine series.

    For how to compute Fourier series, see the :func:`fourier_series`
    docstring.

    Parameters
    ==========

    f : Expr
        Expression for finding fourier_series

    limits : ( x, start, stop)
        x is the independent variable for the expression f
        (start, stop) is the period of the fourier series

    exprs: (a0, an, bn) or Expr
        a0 is the constant term a0 of the fourier series
        an is a dictionary of coefficients of cos terms
         an[k] = coefficient of cos(pi*(k/L)*x)
        bn is a dictionary of coefficients of sin terms
         bn[k] = coefficient of sin(pi*(k/L)*x)

        or exprs can be an expression to be converted to fourier form

    Methods
    =======

    This class is an extension of FourierSeries class.
    Please refer to sympy.series.fourier.FourierSeries for
    further information.

    See Also
    ========

    sympy.series.fourier.FourierSeries
    sympy.series.fourier.fourier_series
    """

    def __new__(cls, f, limits, exprs):
        f = sympify(f)
        limits = sympify(limits)
        exprs = sympify(exprs)

        if not (isinstance(exprs, Tuple) and len(exprs) == 3):  # exprs is not of form (a0, an, bn)
            # Converts the expression to fourier form
            c, e = exprs.as_coeff_add()
            from sympy.simplify.fu import TR10
            rexpr = c + Add(*[TR10(i) for i in e])
            a0, exp_ls = rexpr.expand(trig=False, power_base=False, power_exp=False, log=False).as_coeff_add()

            x = limits[0]
            L = abs(limits[2] - limits[1]) / 2

            a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k is not S.Zero, ])
            b = Wild('b', properties=[lambda k: x not in k.free_symbols, ])

            an = {}
            bn = {}

            # separates the coefficients of sin and cos terms in dictionaries an, and bn
            for p in exp_ls:
                t = p.match(b * cos(a * (pi / L) * x))
                q = p.match(b * sin(a * (pi / L) * x))
                if t:
                    an[t[a]] = t[b] + an.get(t[a], S.Zero)
                elif q:
                    bn[q[a]] = q[b] + bn.get(q[a], S.Zero)
                else:
                    a0 += p

            exprs = Tuple(a0, an, bn)

        return Expr.__new__(cls, f, limits, exprs)

    @property
    def interval(self):
        _length = 1 if self.a0 else 0
        _length += max(set(self.an.keys()).union(set(self.bn.keys()))) + 1
        return Interval(0, _length)

    @property
    def length(self):
        return self.stop - self.start

    def shiftx(self, s):
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        _expr = self.truncate().subs(x, x + s)
        sfunc = self.function.subs(x, x + s)

        return self.func(sfunc, self.args[1], _expr)

    def scale(self, s):
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        _expr = self.truncate() * s
        sfunc = self.function * s

        return self.func(sfunc, self.args[1], _expr)

    def scalex(self, s):
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        _expr = self.truncate().subs(x, x * s)
        sfunc = self.function.subs(x, x * s)

        return self.func(sfunc, self.args[1], _expr)

    def _eval_term(self, pt):
        if pt == 0:
            return self.a0

        _term = self.an.get(pt, S.Zero) * cos(pt * (pi / self.L) * self.x) \
                + self.bn.get(pt, S.Zero) * sin(pt * (pi / self.L) * self.x)
        return _term

    def __add__(self, other):
        if isinstance(other, FourierSeries):
            return other.__add__(fourier_series(self.function, self.args[1],\
                                                finite=False))
        elif isinstance(other, FiniteFourierSeries):
            if self.period != other.period:
                raise ValueError("Both the series should have same periods")

            x, y = self.x, other.x
            function = self.function + other.function.subs(y, x)

            if self.x not in function.free_symbols:
                return function

            return fourier_series(function, limits=self.args[1])


def fourier_series(f, limits=None, finite=True):
    r"""Computes the Fourier trigonometric series expansion.

    Explanation
    ===========

    Fourier trigonometric series of $f(x)$ over the interval $(a, b)$
    is defined as:

    .. math::
        \frac{a_0}{2} + \sum_{n=1}^{\infty}
        (a_n \cos(\frac{2n \pi x}{L}) + b_n \sin(\frac{2n \pi x}{L}))

    where the coefficients are:

    .. math::
        L = b - a

    .. math::
        a_0 = \frac{2}{L} \int_{a}^{b}{f(x) dx}

    .. math::
        a_n = \frac{2}{L} \int_{a}^{b}{f(x) \cos(\frac{2n \pi x}{L}) dx}

    .. math::
        b_n = \frac{2}{L} \int_{a}^{b}{f(x) \sin(\frac{2n \pi x}{L}) dx}

    The condition whether the function $f(x)$ given should be periodic
    or not is more than necessary, because it is sufficient to consider
    the series to be converging to $f(x)$ only in the given interval,
    not throughout the whole real line.

    This also brings a lot of ease for the computation because
    you do not have to make $f(x)$ artificially periodic by
    wrapping it with piecewise, modulo operations,
    but you can shape the function to look like the desired periodic
    function only in the interval $(a, b)$, and the computed series will
    automatically become the series of the periodic version of $f(x)$.

    This property is illustrated in the examples section below.

    Parameters
    ==========

    limits : (sym, start, end), optional
        *sym* denotes the symbol the series is computed with respect to.

        *start* and *end* denotes the start and the end of the interval
        where the fourier series converges to the given function.

        Default range is specified as $-\pi$ and $\pi$.

    Returns
    =======

    FourierSeries
        A symbolic object representing the Fourier trigonometric series.

    Examples
    ========

    Computing the Fourier series of $f(x) = x^2$:

    >>> from sympy import fourier_series, pi
    >>> from sympy.abc import x
    >>> f = x**2
    >>> s = fourier_series(f, (x, -pi, pi))
    >>> s1 = s.truncate(n=3)
    >>> s1
    -4*cos(x) + cos(2*x) + pi**2/3

    Shifting of the Fourier series:

    >>> s.shift(1).truncate()
    -4*cos(x) + cos(2*x) + 1 + pi**2/3
    >>> s.shiftx(1).truncate()
    -4*cos(x + 1) + cos(2*x + 2) + pi**2/3

    Scaling of the Fourier series:

    >>> s.scale(2).truncate()
    -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
    >>> s.scalex(2).truncate()
    -4*cos(2*x) + cos(4*x) + pi**2/3

    Computing the Fourier series of $f(x) = x$:

    This illustrates how truncating to the higher order gives better
    convergence.

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import fourier_series, pi, plot
        >>> from sympy.abc import x
        >>> f = x
        >>> s = fourier_series(f, (x, -pi, pi))
        >>> s1 = s.truncate(n = 3)
        >>> s2 = s.truncate(n = 5)
        >>> s3 = s.truncate(n = 7)
        >>> p = plot(f, s1, s2, s3, (x, -pi, pi), show=False, legend=True)

        >>> p[0].line_color = (0, 0, 0)
        >>> p[0].label = 'x'
        >>> p[1].line_color = (0.7, 0.7, 0.7)
        >>> p[1].label = 'n=3'
        >>> p[2].line_color = (0.5, 0.5, 0.5)
        >>> p[2].label = 'n=5'
        >>> p[3].line_color = (0.3, 0.3, 0.3)
        >>> p[3].label = 'n=7'

        >>> p.show()

    This illustrates how the series converges to different sawtooth
    waves if the different ranges are specified.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> s1 = fourier_series(x, (x, -1, 1)).truncate(10)
        >>> s2 = fourier_series(x, (x, -pi, pi)).truncate(10)
        >>> s3 = fourier_series(x, (x, 0, 1)).truncate(10)
        >>> p = plot(x, s1, s2, s3, (x, -5, 5), show=False, legend=True)

        >>> p[0].line_color = (0, 0, 0)
        >>> p[0].label = 'x'
        >>> p[1].line_color = (0.7, 0.7, 0.7)
        >>> p[1].label = '[-1, 1]'
        >>> p[2].line_color = (0.5, 0.5, 0.5)
        >>> p[2].label = '[-pi, pi]'
        >>> p[3].line_color = (0.3, 0.3, 0.3)
        >>> p[3].label = '[0, 1]'

        >>> p.show()

    Notes
    =====

    Computing Fourier series can be slow
    due to the integration required in computing
    an, bn.

    It is faster to compute Fourier series of a function
    by using shifting and scaling on an already
    computed Fourier series rather than computing
    again.

    e.g. If the Fourier series of ``x**2`` is known
    the Fourier series of ``x**2 - 1`` can be found by shifting by ``-1``.

    See Also
    ========

    sympy.series.fourier.FourierSeries

    References
    ==========

    .. [1] https://mathworld.wolfram.com/FourierSeries.html
    """
    f = sympify(f)

    limits = _process_limits(f, limits)
    x = limits[0]

    if x not in f.free_symbols:
        return f

    if finite:
        L = abs(limits[2] - limits[1]) / 2
        is_finite, res_f = finite_check(f, x, L)
        if is_finite:
            return FiniteFourierSeries(f, limits, res_f)

    n = Dummy('n')
    center = (limits[1] + limits[2]) / 2
    if center.is_zero:
        neg_f = f.subs(x, -x)
        if f == neg_f:
            a0, an = fourier_cos_seq(f, limits, n)
            bn = SeqFormula(0, (1, oo))
            return FourierSeries(f, limits, (a0, an, bn))
        elif f == -neg_f:
            a0 = S.Zero
            an = SeqFormula(0, (1, oo))
            bn = fourier_sin_seq(f, limits, n)
            return FourierSeries(f, limits, (a0, an, bn))
    a0, an = fourier_cos_seq(f, limits, n)
    bn = fourier_sin_seq(f, limits, n)
    return FourierSeries(f, limits, (a0, an, bn))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_embedlayer.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_base import Fusion
from fusion_utils import FusionUtils
from onnx import NodeProto, TensorProto, helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionEmbedLayerNoMask(Fusion):
    """
    Fuse embedding layer into one node (EmbedLayerNormalization).
    It supports the following model types: BERT, DistilBert, ALBert.
    """

    def __init__(self, model: OnnxModel, description: str = "no mask"):
        super().__init__(
            model,
            "EmbedLayerNormalization",
            ["LayerNormalization", "SkipLayerNormalization"],
            description,
        )
        self.utils = FusionUtils(model)
        self.shape_infer = None
        self.shape_infer_done = False

        # The following will be reset in each fuse call of FusionEmbedLayerNormalization
        self.attention = None
        self.embed_node = None

    def match_two_gather(self, add: NodeProto) -> None | tuple[NodeProto, NodeProto]:
        gather_0_path = self.model.match_parent_path(add, ["Gather"], [0])
        if gather_0_path is None:
            return None

        gather_1_path = self.model.match_parent_path(add, ["Gather"], [1])
        if gather_1_path is None:
            return None

        return gather_0_path[0], gather_1_path[0]

    def check_attention_subgraph(
        self,
        layernorm: NodeProto,
        input_name_to_nodes: dict[str, list[NodeProto]],
        is_distil_bert: bool,
    ) -> bool:
        """Check that LayerNormalization has a child of Attention node or subgraph like Attention.

        Args:
            layernorm (NodeProto): LayerNormalization node
            input_name_to_nodes (Dict[str, List[NodeProto]]): map from input name to nodes
            is_distil_bert (bool): whether it is DistilBert or not

        Returns:
            bool: whether there is Attention node or subgraph like Attention
        """
        self.attention = self.model.find_first_child_by_type(
            layernorm, "Attention", input_name_to_nodes, recursive=False
        )

        if self.attention is not None:
            return True

        if layernorm.output[0] not in input_name_to_nodes:
            return False
        children = input_name_to_nodes[layernorm.output[0]]
        children_types = sorted([child.op_type for child in children])

        # Try find MultiHeadAttention
        if children_types == ["MatMul", "MatMul", "MatMul", "SkipLayerNormalization"]:
            for node in children:
                if node.op_type == "SkipLayerNormalization":
                    path1 = self.model.match_parent_path(
                        node,
                        ["Add", "MatMul", "MultiHeadAttention", "MatMul"],
                        [None, None, 0, 0],
                    )
                    if path1 is not None and path1[-1].input[0] == layernorm.output[0]:
                        self.cross_attention = path1[2]
                        return True

        # In case user disables attention fusion, check whether subgraph looks like Attention.
        # For Albert, there is MatMul+Add after embedding layer before attention.
        if len(children) == 1 and children[0].op_type == "MatMul" and children[0].output[0] in input_name_to_nodes:
            grandchildren = input_name_to_nodes[children[0].output[0]]
            if (
                len(grandchildren) == 1
                and grandchildren[0].op_type == "Add"
                and grandchildren[0].output[0] in input_name_to_nodes
            ):
                nodes = input_name_to_nodes[grandchildren[0].output[0]]
                for node in nodes:
                    if node.op_type == "Attention":
                        self.attention = node
                        return True
                children_types = sorted([child.op_type for child in nodes])

        # Two Shape nodes might be merged by ORT
        if is_distil_bert:
            # SkipLayerNormailization might exist when model has been optimized by ORT first.
            if (
                children_types != ["MatMul", "MatMul", "MatMul", "Shape", "SkipLayerNormalization"]
                and children_types != ["Add", "MatMul", "MatMul", "MatMul", "Shape", "Shape"]
                and children_types != ["Add", "MatMul", "MatMul", "MatMul", "Shape"]
            ):
                logger.debug("No Attention like subgraph in children of LayerNormalization")
                return False
        else:
            if children_types != [
                "Add",
                "MatMul",
                "MatMul",
                "MatMul",
            ] and children_types != [
                "MatMul",
                "MatMul",
                "MatMul",
                "SkipLayerNormalization",
            ]:
                logger.debug("No Attention like subgraph in children of LayerNormalization")
                return False

        return True

    def match_position_embedding_distilbert(self, position_embedding_gather, input_ids, output_name_to_node):
        """  Match position embedding path from input_ids to Gather for DistilBert.

        Pattern is like the following:
                 (input_ids)
                      |
                     Shape
                       |   \
                       |    Gather (indices=1)
                       |       |
                       |      Cast (optional)
                       |       |
                       |      Range (start=0, end=*, delta=1)
                       |       |
                       |    Unsqueeze
                       |    /
                      Expand
                        |
                      Gather
        """
        # remove after tests pass
        path1 = self.model.match_parent_path(position_embedding_gather, ["Expand", "Shape"], [1, 1])
        if path1 is None:
            path1 = self.model.match_parent_path(
                position_embedding_gather,
                ["Expand", "Where", "Reshape", "Shape"],
                [1, 1, 2, 0],
            )
            if path1 is None:
                return False

        expand, shape = path1[0], path1[-1]
        if shape.input[0] != input_ids:
            return False

        _, path2, _ = self.model.match_parent_paths(
            expand,
            [
                (["Unsqueeze", "Range", "Cast", "Gather", "Shape"], [0, 0, 1, 0, 0]),
                (["Unsqueeze", "Range", "Gather", "Shape"], [0, 0, 1, 0]),
            ],
            output_name_to_node,
        )
        if path2 is None:
            return False

        range_node = path2[1]
        if not (
            self.utils.check_node_input_value(range_node, 0, 0) and self.utils.check_node_input_value(range_node, 2, 1)
        ):
            return False

        gather_node = path2[-2]
        if not (self.utils.check_node_input_value(gather_node, 1, 1)):
            return False

        shape_node = path2[-1]
        if shape_node.input[0] != input_ids:
            return False

        return True

    def match_position_embedding_roberta(self, position_embedding_gather, input_ids, output_name_to_node):
        """Match position embedding path from input_ids to Gather for Roberta.

        Roberta Embedding Layer Pattern (* is optional since it might be removed by ORT, ? is the padding word id):
          (input_ids) --> Equal(B=?) -- Not -- Cast(to=6) -- CumSum(axis=1) -- Mul -- Cast(to=7) -- Add(B=1) -- Cast(to=7)* --> Gather
                                                |                              ^
                                                V                              |
                                                +------------------------------+

        Roberta new pattern from transformers v4.9:
           (input_ids) --> Equal(B=?) -- Not -- Cast(to=6) -- CumSum(axis=1) -- Add(B=0) -- Mul -- Cast(to=7) -- Add(B=1) --> Gather
                                                |                                           ^
                                                V                                           |
                                                +-------------------------------------------+

        start_node = position_embedding_gather
        start_index = 1

        # match optional Cast node.
        parent = self.model.get_parent(start_node, start_index, output_name_to_node)
        if parent is None:
            return
        if parent.op_type == "Cast":
            if OnnxModel.get_node_attribute(parent, "to") != 7:
                return
            start_node = parent
            start_index = 0

        i, path, return_indices = self.model.match_parent_paths(
            start_node,
            [ (['Add', 'Cast', 'Mul', 'CumSum', 'Cast', 'Not', 'Equal'], [start_index, 0, 0, 0, 0, 0, 0]),
              (['Add', 'Cast', 'Mul', 'Add', 'CumSum', 'Cast', 'Not', 'Equal'], [start_index, 0, 0, 0, 0, 0, 0, 0])],
            output_name_to_node)

        if path is not None:
            # constant input of Add shall be 1.
            i, value = self.model.get_constant_input(path[0])
            if value != 1:
                return False

            _, self.padding_word_id = self.model.get_constant_input(path[-1])

            return input_ids == path[-1].input[0]
        """

        return False

    def match_position_embedding_bert(self, position_embedding_gather, input_ids, output_name_to_node):
        """  Match position embedding path from input_ids to Gather for BERT.

        BERT Embedding Layer Pattern:
                                    (input_ids)
                                   /         \
                                 /          Shape
                                /              |
                              /              Gather (indices=1)
                             /                  |
                            /                  Add (optional, B=0)
                           /                    |
                        Gather (segment_ids) Unsqueeze (axes=0)
                           \\        |           |
                            \\     Gather      Slice (data[1,512], starts=0, ends=*, axes=1, steps=1)
                              \\    /            |
                                Add          Gather
                                   \\       /
                                      Add
                                       |
                                LayerNormalization
        """
        path = self.model.match_parent_path(
            position_embedding_gather,
            ["Slice", "Unsqueeze"],
            [1, 2],
            output_name_to_node,
        )
        if path is None:
            return False

        slice, unsqueeze = path
        slice_weight = self.model.get_constant_value(slice.input[0])
        if not (
            slice_weight is not None
            and len(slice_weight.shape) == 2
            and slice_weight.shape[0] == 1
            and self.utils.check_node_input_value(slice, 1, [0])
            and self.utils.check_node_input_value(slice, 3, [1])
            and (len(slice.input) == 4 or self.utils.check_node_input_value(slice, 4, [1]))
        ):
            return False

        opset_version = self.model.get_opset_version()
        if opset_version < 13:
            if not FusionUtils.check_node_attribute(unsqueeze, "axes", [0]):
                return False
        else:
            if not self.utils.check_node_input_value(unsqueeze, 1, [0]):
                return False

        node = self.model.get_parent(unsqueeze, 0, output_name_to_node)
        if node is None:
            return False
        if node.op_type == "Add":
            if not self.utils.check_node_input_value(node, 1, 0):
                return False
            gather = self.model.get_parent(node, 0, output_name_to_node)
        else:
            gather = node

        if gather is None or gather.op_type != "Gather":
            return False
        if not (self.utils.check_node_input_value(gather, 1, 1)):
            return False

        shape = self.model.get_parent(gather, 0, output_name_to_node)
        if shape is None or shape.op_type != "Shape":
            return False

        return input_ids == shape.input[0]

    def match_position_embedding(self, position_embedding_gather, input_ids, output_name_to_node):
        if self.match_position_embedding_bert(position_embedding_gather, input_ids, output_name_to_node):
            return True

        # TODO: Support roberta (position starts from 2 instead of 0) in EmbedLayerNormalization kernel
        #       related: https://github.com/huggingface/transformers/issues/10736
        # if self.match_position_embedding_roberta(position_embedding_gather, input_ids, output_name_to_node):
        #    return True

        if self.match_position_embedding_distilbert(position_embedding_gather, input_ids, output_name_to_node):
            return True

        return False

    def check_embedding(self, word_embedding_gather, segment_embedding_gather, position_embedding_gather):
        """Sanity check of embedding weights, and match hidden_size of weights and shape of inputs."""
        input_ids = word_embedding_gather.input[1]
        segment_ids = segment_embedding_gather.input[1] if segment_embedding_gather else None
        position_ids = position_embedding_gather.input[1]

        if not self.shape_infer_done:
            self.shape_infer = self.model.infer_runtime_shape(update=True)
            self.shape_infer_done = True

        if self.shape_infer is not None:
            input_ids_shape = self.shape_infer.get_edge_shape(input_ids)
            position_ids_shape = self.shape_infer.get_edge_shape(position_ids)
            assert input_ids_shape and position_ids_shape
            if not (
                len(input_ids_shape) == 2
                and len(position_ids_shape) == 2
                and input_ids_shape[1] == position_ids_shape[1]
            ):
                logger.info(
                    f"Cannot fuse EmbedLayerNormalization: input_ids and position_ids not matched in 2nd dimension: {input_ids_shape} vs {position_ids_shape}"
                )
                return False

            if segment_ids and not self.shape_infer.compare_shape(input_ids, segment_ids):
                logger.info(
                    f"Cannot fuse EmbedLayerNormalization: input_ids and segment_ids does not have same shape: {input_ids_shape} != {self.shape_infer.get_edge_shape(segment_ids)}"
                )
                return False

        word_embedding_table = self.model.get_constant_value(word_embedding_gather.input[0])
        if word_embedding_table is None or len(word_embedding_table.shape) != 2:
            logger.info("Cannot fuse EmbedLayerNormalization: word embedding table is not expected")
            return False

        position_embedding_table = self.model.get_constant_value(position_embedding_gather.input[0])
        if (
            position_embedding_table is None
            or len(position_embedding_table.shape) != 2
            or (word_embedding_table.shape[1] != position_embedding_table.shape[1])
        ):
            logger.info("Cannot fuse EmbedLayerNormalization: position embedding table is not expected")
            return False

        if segment_ids:
            segment_embedding_table = self.model.get_constant_value(segment_embedding_gather.input[0])
            if (
                segment_embedding_table is None
                or len(segment_embedding_table.shape) != 2
                or (word_embedding_table.shape[1] != segment_embedding_table.shape[1])
            ):
                logger.info("Cannot fuse EmbedLayerNormalization: segment embedding table is not expected")
                return False

        # In normal case, word embedding table is the largest, and segment embedding table is the smallest, while position embedding table is in between.
        # TODO: use other information (like initializer names) to identify different embedding weights automatically.
        if word_embedding_table.shape[0] <= position_embedding_table.shape[0]:
            logger.warning(
                f"word_embedding_table ({word_embedding_gather.input[0]}) size {word_embedding_table.shape[0]} <= position_embedding_table ({position_embedding_gather.input[0]}) size {position_embedding_table.shape[0]}"
            )

        if segment_ids:
            if word_embedding_table.shape[0] <= segment_embedding_table.shape[0]:
                logger.warning(
                    f"word_embedding_table ({word_embedding_gather.input[0]}) size {word_embedding_table.shape[0]} <= segment_embedding_table ({segment_embedding_gather.input[0]}) size {segment_embedding_table.shape[0]}"
                )

            if position_embedding_table.shape[0] <= segment_embedding_table.shape[0]:
                logger.warning(
                    f"position_embedding_table ({position_embedding_gather.input[0]}) size {position_embedding_table.shape[0]} <= segment_embedding_table ({segment_embedding_gather.input[0]}) size {segment_embedding_table.shape[0]}"
                )

        return True

    def cast_to_int32(self, input_name: str) -> tuple[str, None | NodeProto]:
        """Cast a graph input or node input to int32.

        Args:
            input_name (str): name of graph input or node input

        Returns:
            A tuple of casted input name and the cast node.
            int32_output (str): If input is int32, it is the input name, Otherwise it is output name of Cast node.
            input_cast_node (Union[None, NodeProto]): Cast node. It could be None if input is int32.
        """
        input_cast_node = None
        graph_input = self.model.find_graph_input(input_name)
        if graph_input is not None:
            if graph_input.type.tensor_type.elem_type != TensorProto.INT32:
                int32_output, input_cast_node = self.utils.cast_input_to_int32(input_name)
            else:
                int32_output = input_name
        else:
            int32_output, input_cast_node = self.utils.cast_input_to_int32(input_name)

        return int32_output, input_cast_node

    def create_fused_node(
        self,
        input_ids: str,
        layernorm: NodeProto,
        word_embedding_gather: NodeProto,
        position_embedding_gather: NodeProto,
        segment_embedding_gather: None | NodeProto,
        position_ids: str | None = None,
        embedding_sum_output=False,
        embedding_sum_name=None,
    ):
        """Create an EmbedLayerNormalization node. Note that segment embedding is optional.

        Args:
            input_ids (str): input_ids for word embeddings
            layernorm (NodeProto): LayerNormalization or SkipLayerNormalization node.
            word_embedding_gather (NodeProto): the Gather node for word embedding
            position_embedding_gather (NodeProto): the Gather node for position embedding
            segment_embedding_gather (Union[None, NodeProto]): the Gather node for segment embedding, or None.

        Returns:
            NodeProto: the EmbedLayerNormalization node created.
        """
        nodes_to_add = []
        input_ids, _ = self.cast_to_int32(input_ids)

        node_name = self.model.create_node_name("EmbedLayerNormalization")

        if layernorm.op_type == "LayerNormalization":
            gamma = layernorm.input[1]
            beta = layernorm.input[2]
        else:  # SkipLayerNormalization
            gamma = layernorm.input[2]
            beta = layernorm.input[3]

        embed_node_inputs = None
        if segment_embedding_gather is not None:
            segment_ids, _ = self.cast_to_int32(segment_embedding_gather.input[1])

            embed_node_inputs = [
                input_ids,
                segment_ids,
                word_embedding_gather.input[0],
                position_embedding_gather.input[0],
                segment_embedding_gather.input[0],
                gamma,
                beta,
            ]
        else:  # no segment embedding
            embed_node_inputs = [
                input_ids,
                "",
                word_embedding_gather.input[0],
                position_embedding_gather.input[0],
                "",
                gamma,
                beta,
            ]

        if position_ids is not None:
            # Adding an empty input for mask before position_ids
            embed_node_inputs.append("")
            position_ids, _ = self.cast_to_int32(position_ids)
            embed_node_inputs.append(position_ids)

        embed_node_outputs = [node_name + "_output", node_name + "_dummy_mask_index"]
        if embedding_sum_output:
            name = embedding_sum_name if embedding_sum_name is not None else node_name + "_embedding_sum"
            embed_node_outputs.append(name)

        embed_node = helper.make_node(
            "EmbedLayerNormalization",
            embed_node_inputs,
            outputs=embed_node_outputs,
            name=node_name,
        )

        embed_node.domain = "com.microsoft"

        # Pass attribute "epsilon" from normalize node to EmbedLayerNormalization.
        for att in layernorm.attribute:
            if att.name == "epsilon":
                embed_node.attribute.extend([att])

        # Set default value to 1e-12 if no attribute is found.
        # OnnxRuntime 1.2.0 or older has no epsilon attribute. The optimized model can only work for 1.3.0 or later.
        if len(embed_node.attribute) == 0:
            embed_node.attribute.extend([helper.make_attribute("epsilon", 1.0e-12)])

        # Make sure new EmbedLayerNormalization node is the last one in self.nodes_to_add.
        nodes_to_add.append(embed_node)
        for node in nodes_to_add:
            self.node_name_to_graph_name[node.name] = self.this_graph_name
        self.nodes_to_add.extend(nodes_to_add)

        self.embed_node = embed_node
        return embed_node

    def finish_fusion(self, layernorm, embed_node):
        self.model.replace_input_of_all_nodes(layernorm.output[0], embed_node.output[0])
        # use prune graph to remove nodes that is not needed
        self.prune_graph = True

    def is_skip_layer_norm_with_sum_output(self, node):
        return (node.op_type == "SkipLayerNormalization") and len(node.output) > 3 and len(node.output[3]) > 0

    def fuse_gpt2(
        self, layernorm, add_before_layernorm, input_name_to_nodes, output_name_to_node, optional_segment_gather=None
    ):
        # graph checks
        # gpt2 has optional segment embedding, subgraph pattern is like
        #                      input_ids  position_ids
        #                         |        |
        #  token_ids           Gather    Gather
        #       |                   \   /
        #   Gather (optional)        Add _ _ _ _ _
        #                   \         |           |
        #                     LayerNormalization  |
        #                             |           |
        #                          Attention      |
        #                             |           |
        #                           Matmul        |
        #                             |          /
        #                            Add        /
        #                              \       /
        #                                 Add
        two_gather = self.match_two_gather(add_before_layernorm)
        if two_gather is None:
            return False

        word_embedding_gather, position_embedding_gather = two_gather
        input_ids = word_embedding_gather.input[1]
        position_ids = position_embedding_gather.input[1]

        if not self.check_attention_subgraph(layernorm, input_name_to_nodes, is_distil_bert=False):
            return False

        if not self.check_embedding(word_embedding_gather, None, position_embedding_gather):
            return False

        # If layernorm node is SkipLayerNormalization, we need look at its optional fourth output.
        # If the add_before_layernorm node is an Add node, then the add_output output is the first output of this node.
        # If the add_before_layernorm node is a SkipLayerNormalization node, then the add_output output
        # is the (optional) fourth index output of this node.
        # When add_before_layernorm is SkipLayerNormalization, add_before_layernorm and layernorm are same node.
        if layernorm.op_type == "SkipLayerNormalization":
            need_embedding_sum_output = self.is_skip_layer_norm_with_sum_output(layernorm)
            sum_output_index = 3
            node_with_sum_output = layernorm
            sum_output = layernorm.output[3] if need_embedding_sum_output else None
            is_sum_graph_output = (sum_output is not None) and (self.model.find_graph_output(sum_output) is not None)
        else:  # layernorm.op_type == "LayerNormalization"
            node_with_sum_output = add_before_layernorm
            sum_output_index = 0 if add_before_layernorm.op_type == "Add" else 3
            sum_output = (
                add_before_layernorm.output[sum_output_index]
                if len(add_before_layernorm.output) > sum_output_index
                else None
            )
            is_sum_graph_output = (sum_output is not None) and (self.model.find_graph_output(sum_output) is not None)
            is_sum_used_by_multiple_nodes = (
                sum_output and (sum_output in input_name_to_nodes) and len(input_name_to_nodes[sum_output]) > 1
            )
            need_embedding_sum_output = (sum_output is not None) and (
                add_before_layernorm.op_type != "Add" or is_sum_graph_output or is_sum_used_by_multiple_nodes
            )

        # make the fused node
        embed_node = self.create_fused_node(
            input_ids,
            layernorm,
            word_embedding_gather,
            position_embedding_gather,
            optional_segment_gather,
            position_ids,
            embedding_sum_output=need_embedding_sum_output,
            embedding_sum_name=sum_output if is_sum_graph_output else None,
        )

        if need_embedding_sum_output:
            node_with_sum_output.output[sum_output_index] = "_no_use__to_be_removed_"
            if not is_sum_graph_output:
                self.model.replace_input_of_all_nodes(sum_output, embed_node.output[2])

        self.finish_fusion(layernorm, embed_node)
        return True

    def fuse_distilbert(self, layernorm, add_before_layernorm, input_name_to_nodes, output_name_to_node):
        """Fuse embedding layer for DistilBert
        Args:
            layernorm (NodeProto): node of LayerNormalization or SkipLayerNormalization
            add_before_layernorm (NodeProto): the Add node before LayerNormalization, or the SkipLayerNormalization itself
            input_name_to_nodes (Dict[str, List[NodeProto]]): map from input name to nodes
            output_name_to_node (Dict[str, List[NodeProto]]): map from output name to nodes
        """

        # DistilBert has no segment embedding, subgraph pattern is like
        #       input_ids
        #        |      \
        #        |     (position_embedding_subgraph)
        #        |        |
        #     Gather    Gather
        #          \   /
        #           Add
        #            |
        #    LayerNormalization
        two_gather = self.match_two_gather(add_before_layernorm)
        if two_gather is None:
            return False

        word_embedding_gather, position_embedding_gather = two_gather
        input_ids = word_embedding_gather.input[1]

        if not self.check_attention_subgraph(layernorm, input_name_to_nodes, is_distil_bert=True):
            return False

        if not self.match_position_embedding(position_embedding_gather, input_ids, output_name_to_node):
            return False

        if not self.check_embedding(word_embedding_gather, None, position_embedding_gather):
            return False

        embed_node = self.create_fused_node(
            input_ids, layernorm, word_embedding_gather, position_embedding_gather, None
        )
        self.finish_fusion(layernorm, embed_node)
        return True

    def fuse_bert(self, layernorm, add_before_layernorm, input_name_to_nodes, output_name_to_node):
        """Fuse embedding layer for Bert
        Args:
            layernorm (NodeProto): node of LayerNormalization or SkipLayerNormalization
            add_before_layernorm (NodeProto): the Add node before LayerNormalization, or the SkipLayerNormalization itself
            input_name_to_nodes (Dict[str, List[NodeProto]]): map from input name to nodes
            output_name_to_node (Dict[str, List[NodeProto]]): map from output name to nodes
        """

        add_2_gather = self.model.match_parent_path(add_before_layernorm, ["Add"], [0])
        if add_2_gather is None:
            return False

        two_gather = self.match_two_gather(add_2_gather[0])
        if two_gather is None:
            return False

        word_embedding_gather, segment_embedding_gather = two_gather

        input_ids = word_embedding_gather.input[1]

        if not self.check_attention_subgraph(layernorm, input_name_to_nodes, is_distil_bert=False):
            return False

        position_embedding_path = self.model.match_parent_path(add_before_layernorm, ["Gather"], [1])
        if position_embedding_path is None:
            return False

        position_embedding_gather = position_embedding_path[0]
        if not self.match_position_embedding(position_embedding_gather, input_ids, output_name_to_node):
            if not self.match_position_embedding(segment_embedding_gather, input_ids, output_name_to_node):
                return False
            # position and segment are switched
            temp = segment_embedding_gather
            segment_embedding_gather = position_embedding_gather
            position_embedding_gather = temp

        if not self.check_embedding(word_embedding_gather, segment_embedding_gather, position_embedding_gather):
            return False

        embed_node = self.create_fused_node(
            input_ids,
            layernorm,
            word_embedding_gather,
            position_embedding_gather,
            segment_embedding_gather,
        )
        self.finish_fusion(layernorm, embed_node)
        return True

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        first_add_path = self.model.match_parent_path(node, ["Add"], [0])
        if node.op_type == "LayerNormalization":
            if first_add_path is None:
                return
            add_before_layernorm = first_add_path[0]
            optional_segment_gather = None
        else:  # SkipLayerNormalization
            gather_0_path = self.model.match_parent_path(node, ["Gather"], [0])
            gather_1_path = self.model.match_parent_path(node, ["Gather"], [1])
            if gather_0_path is None and gather_1_path is not None:
                if first_add_path is None:
                    return
                add_before_layernorm = first_add_path[0]
                optional_segment_gather = gather_1_path[0]
            elif gather_0_path is not None and gather_1_path is None:
                first_add_path = self.model.match_parent_path(node, ["Add"], [1])
                if first_add_path is None:
                    return
                add_before_layernorm = first_add_path[0]
                optional_segment_gather = gather_0_path[0]
            else:
                add_before_layernorm = node  # Add is fused into SkipLayerNormalization
                optional_segment_gather = None

        if self.fuse_gpt2(
            node, add_before_layernorm, input_name_to_nodes, output_name_to_node, optional_segment_gather
        ):
            return

        if self.fuse_distilbert(node, add_before_layernorm, input_name_to_nodes, output_name_to_node):
            return

        if self.fuse_bert(node, add_before_layernorm, input_name_to_nodes, output_name_to_node):
            return


class FusionEmbedLayerNormalization(FusionEmbedLayerNoMask):
    def __init__(self, model: OnnxModel, use_mask_index=False):
        super().__init__(model, "with mask")
        self.use_mask_index = use_mask_index

    def replace_mask(self, mask_int32, attention_nodes):
        # Inputs of EmbedLayerNorm: input_ids, segment_ids (optional), word_embedding, position_embedding,
        #           segment_embedding (optional), gamma, beta, mask (optional), position_ids (optional)
        embed_node = self.embed_node
        if len(embed_node.input) == 7:
            embed_node.input.append(mask_int32)
            logger.debug("append mask to %s", embed_node.name)
        elif len(embed_node.input) > 7 and not embed_node.input[7]:
            embed_node.input[7] = mask_int32
            logger.debug("replace mask in %s", embed_node.name)
        else:
            logger.debug("skip mask in %s", embed_node.name)
            return

        for attention_node in attention_nodes:
            logger.debug("update mask_index in %s", attention_node.name)
            if attention_node.op_type == "Attention":
                attention_node.input[3] = embed_node.output[1]
            elif attention_node.op_type == "MultiHeadAttention":
                attention_node.input[4] = embed_node.output[1]

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # Reset attention and embed_node so that we know fusion is successful when they are not None.
        self.attention = None
        self.cross_attention = None
        self.embed_node = None
        super().fuse(node, input_name_to_nodes, output_name_to_node)

        if self.embed_node is None:
            return

        if not self.use_mask_index:
            logger.debug("--use_mask_index is not set: EmbedLayerNormalization will not have mask")
            self.increase_counter("EmbedLayerNormalization(no mask)")
            return

        if self.attention is None and self.cross_attention is None:
            logger.debug("EmbedLayerNormalization will not have mask since attention node is not found")
            self.increase_counter("EmbedLayerNormalization(no mask)")
            return

        if self.attention:
            mask_int32 = self.attention.input[3]
        else:
            mask_int32 = self.cross_attention.input[4]

        children_nodes = input_name_to_nodes[mask_int32]
        if self.model.find_graph_input(mask_int32):
            attention_nodes = [node for node in children_nodes if node.op_type in ["Attention", "MultiHeadAttention"]]
            self.replace_mask(mask_int32, attention_nodes)
            self.increase_counter("EmbedLayerNormalization(with mask)")
            return

        if mask_int32 not in output_name_to_node:
            logger.debug("EmbedLayerNormalization will not have mask since %s is not a node output", mask_int32)
            self.increase_counter("EmbedLayerNormalization(no mask)")
            return

        node = output_name_to_node[mask_int32]
        if node.op_type in ["ReduceSum", "Cast"]:
            attention_nodes = [node for node in children_nodes if node.op_type in ["Attention", "MultiHeadAttention"]]
            if node.op_type == "ReduceSum":
                mask_int32 = node.input[0]
                if len(children_nodes) == len(attention_nodes):
                    self.nodes_to_remove.append(node)
            self.replace_mask(mask_int32, attention_nodes)
            self.increase_counter("EmbedLayerNormalization(with mask)")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\dtypes\missing.py ===
"""
missing types & inference
"""
from __future__ import annotations

from decimal import Decimal
from functools import partial
from typing import (
    TYPE_CHECKING,
    overload,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
    NaT,
    iNaT,
)

from pandas.core.dtypes.common import (
    DT64NS_DTYPE,
    TD64NS_DTYPE,
    ensure_object,
    is_scalar,
    is_string_or_object_np_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_list_like

if TYPE_CHECKING:
    from re import Pattern

    from pandas._typing import (
        ArrayLike,
        DtypeObj,
        NDFrame,
        NDFrameT,
        Scalar,
        npt,
    )

    from pandas import Series
    from pandas.core.indexes.base import Index


isposinf_scalar = libmissing.isposinf_scalar
isneginf_scalar = libmissing.isneginf_scalar

nan_checker = np.isnan
INF_AS_NA = False
_dtype_object = np.dtype("object")
_dtype_str = np.dtype(str)


@overload
def isna(obj: Scalar | Pattern) -> bool:
    ...


@overload
def isna(
    obj: ArrayLike | Index | list,
) -> npt.NDArray[np.bool_]:
    ...


@overload
def isna(obj: NDFrameT) -> NDFrameT:
    ...


# handle unions
@overload
def isna(obj: NDFrameT | ArrayLike | Index | list) -> NDFrameT | npt.NDArray[np.bool_]:
    ...


@overload
def isna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame:
    ...


def isna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame:
    """
    Detect missing values for an array-like object.

    This function takes a scalar or array-like object and indicates
    whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``
    in object arrays, ``NaT`` in datetimelike).

    Parameters
    ----------
    obj : scalar or array-like
        Object to check for null or missing values.

    Returns
    -------
    bool or array-like of bool
        For scalar input, returns a scalar boolean.
        For array input, returns an array of boolean indicating whether each
        corresponding element is missing.

    See Also
    --------
    notna : Boolean inverse of pandas.isna.
    Series.isna : Detect missing values in a Series.
    DataFrame.isna : Detect missing values in a DataFrame.
    Index.isna : Detect missing values in an Index.

    Examples
    --------
    Scalar arguments (including strings) result in a scalar boolean.

    >>> pd.isna('dog')
    False

    >>> pd.isna(pd.NA)
    True

    >>> pd.isna(np.nan)
    True

    ndarrays result in an ndarray of booleans.

    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])
    >>> pd.isna(array)
    array([[False,  True, False],
           [False, False,  True]])

    For indexes, an ndarray of booleans is returned.

    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,
    ...                           "2017-07-08"])
    >>> index
    DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],
                  dtype='datetime64[ns]', freq=None)
    >>> pd.isna(index)
    array([False, False,  True, False])

    For Series and DataFrame, the same type is returned, containing booleans.

    >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
    >>> df
         0     1    2
    0  ant   bee  cat
    1  dog  None  fly
    >>> pd.isna(df)
           0      1      2
    0  False  False  False
    1  False   True  False

    >>> pd.isna(df[1])
    0    False
    1     True
    Name: 1, dtype: bool
    """
    return _isna(obj)


isnull = isna


def _isna(obj, inf_as_na: bool = False):
    """
    Detect missing values, treating None, NaN or NA as null. Infinite
    values will also be treated as null if inf_as_na is True.

    Parameters
    ----------
    obj: ndarray or object value
        Input array or scalar value.
    inf_as_na: bool
        Whether to treat infinity as null.

    Returns
    -------
    boolean ndarray or boolean
    """
    if is_scalar(obj):
        return libmissing.checknull(obj, inf_as_na=inf_as_na)
    elif isinstance(obj, ABCMultiIndex):
        raise NotImplementedError("isna is not defined for MultiIndex")
    elif isinstance(obj, type):
        return False
    elif isinstance(obj, (np.ndarray, ABCExtensionArray)):
        return _isna_array(obj, inf_as_na=inf_as_na)
    elif isinstance(obj, ABCIndex):
        # Try to use cached isna, which also short-circuits for integer dtypes
        #  and avoids materializing RangeIndex._values
        if not obj._can_hold_na:
            return obj.isna()
        return _isna_array(obj._values, inf_as_na=inf_as_na)

    elif isinstance(obj, ABCSeries):
        result = _isna_array(obj._values, inf_as_na=inf_as_na)
        # box
        result = obj._constructor(result, index=obj.index, name=obj.name, copy=False)
        return result
    elif isinstance(obj, ABCDataFrame):
        return obj.isna()
    elif isinstance(obj, list):
        return _isna_array(np.asarray(obj, dtype=object), inf_as_na=inf_as_na)
    elif hasattr(obj, "__array__"):
        return _isna_array(np.asarray(obj), inf_as_na=inf_as_na)
    else:
        return False


def _use_inf_as_na(key) -> None:
    """
    Option change callback for na/inf behaviour.

    Choose which replacement for numpy.isnan / -numpy.isfinite is used.

    Parameters
    ----------
    flag: bool
        True means treat None, NaN, INF, -INF as null (old way),
        False means None and NaN are null, but INF, -INF are not null
        (new way).

    Notes
    -----
    This approach to setting global module values is discussed and
    approved here:

    * https://stackoverflow.com/questions/4859217/
      programmatically-creating-variables-in-python/4859312#4859312
    """
    inf_as_na = get_option(key)
    globals()["_isna"] = partial(_isna, inf_as_na=inf_as_na)
    if inf_as_na:
        globals()["nan_checker"] = lambda x: ~np.isfinite(x)
        globals()["INF_AS_NA"] = True
    else:
        globals()["nan_checker"] = np.isnan
        globals()["INF_AS_NA"] = False


def _isna_array(values: ArrayLike, inf_as_na: bool = False):
    """
    Return an array indicating which values of the input array are NaN / NA.

    Parameters
    ----------
    obj: ndarray or ExtensionArray
        The input array whose elements are to be checked.
    inf_as_na: bool
        Whether or not to treat infinite values as NA.

    Returns
    -------
    array-like
        Array of boolean values denoting the NA status of each element.
    """
    dtype = values.dtype

    if not isinstance(values, np.ndarray):
        # i.e. ExtensionArray
        if inf_as_na and isinstance(dtype, CategoricalDtype):
            result = libmissing.isnaobj(values.to_numpy(), inf_as_na=inf_as_na)
        else:
            # error: Incompatible types in assignment (expression has type
            # "Union[ndarray[Any, Any], ExtensionArraySupportsAnyAll]", variable has
            # type "ndarray[Any, dtype[bool_]]")
            result = values.isna()  # type: ignore[assignment]
    elif isinstance(values, np.rec.recarray):
        # GH 48526
        result = _isna_recarray_dtype(values, inf_as_na=inf_as_na)
    elif is_string_or_object_np_dtype(values.dtype):
        result = _isna_string_dtype(values, inf_as_na=inf_as_na)
    elif dtype.kind in "mM":
        # this is the NaT pattern
        result = values.view("i8") == iNaT
    else:
        if inf_as_na:
            result = ~np.isfinite(values)
        else:
            result = np.isnan(values)

    return result


def _isna_string_dtype(values: np.ndarray, inf_as_na: bool) -> npt.NDArray[np.bool_]:
    # Working around NumPy ticket 1542
    dtype = values.dtype

    if dtype.kind in ("S", "U"):
        result = np.zeros(values.shape, dtype=bool)
    else:
        if values.ndim in {1, 2}:
            result = libmissing.isnaobj(values, inf_as_na=inf_as_na)
        else:
            # 0-D, reached via e.g. mask_missing
            result = libmissing.isnaobj(values.ravel(), inf_as_na=inf_as_na)
            result = result.reshape(values.shape)

    return result


def _has_record_inf_value(record_as_array: np.ndarray) -> np.bool_:
    is_inf_in_record = np.zeros(len(record_as_array), dtype=bool)
    for i, value in enumerate(record_as_array):
        is_element_inf = False
        try:
            is_element_inf = np.isinf(value)
        except TypeError:
            is_element_inf = False
        is_inf_in_record[i] = is_element_inf

    return np.any(is_inf_in_record)


def _isna_recarray_dtype(
    values: np.rec.recarray, inf_as_na: bool
) -> npt.NDArray[np.bool_]:
    result = np.zeros(values.shape, dtype=bool)
    for i, record in enumerate(values):
        record_as_array = np.array(record.tolist())
        does_record_contain_nan = isna_all(record_as_array)
        does_record_contain_inf = False
        if inf_as_na:
            does_record_contain_inf = bool(_has_record_inf_value(record_as_array))
        result[i] = np.any(
            np.logical_or(does_record_contain_nan, does_record_contain_inf)
        )

    return result


@overload
def notna(obj: Scalar) -> bool:
    ...


@overload
def notna(
    obj: ArrayLike | Index | list,
) -> npt.NDArray[np.bool_]:
    ...


@overload
def notna(obj: NDFrameT) -> NDFrameT:
    ...


# handle unions
@overload
def notna(obj: NDFrameT | ArrayLike | Index | list) -> NDFrameT | npt.NDArray[np.bool_]:
    ...


@overload
def notna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame:
    ...


def notna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame:
    """
    Detect non-missing values for an array-like object.

    This function takes a scalar or array-like object and indicates
    whether values are valid (not missing, which is ``NaN`` in numeric
    arrays, ``None`` or ``NaN`` in object arrays, ``NaT`` in datetimelike).

    Parameters
    ----------
    obj : array-like or object value
        Object to check for *not* null or *non*-missing values.

    Returns
    -------
    bool or array-like of bool
        For scalar input, returns a scalar boolean.
        For array input, returns an array of boolean indicating whether each
        corresponding element is valid.

    See Also
    --------
    isna : Boolean inverse of pandas.notna.
    Series.notna : Detect valid values in a Series.
    DataFrame.notna : Detect valid values in a DataFrame.
    Index.notna : Detect valid values in an Index.

    Examples
    --------
    Scalar arguments (including strings) result in a scalar boolean.

    >>> pd.notna('dog')
    True

    >>> pd.notna(pd.NA)
    False

    >>> pd.notna(np.nan)
    False

    ndarrays result in an ndarray of booleans.

    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])
    >>> pd.notna(array)
    array([[ True, False,  True],
           [ True,  True, False]])

    For indexes, an ndarray of booleans is returned.

    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,
    ...                          "2017-07-08"])
    >>> index
    DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],
                  dtype='datetime64[ns]', freq=None)
    >>> pd.notna(index)
    array([ True,  True, False,  True])

    For Series and DataFrame, the same type is returned, containing booleans.

    >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
    >>> df
         0     1    2
    0  ant   bee  cat
    1  dog  None  fly
    >>> pd.notna(df)
          0      1     2
    0  True   True  True
    1  True  False  True

    >>> pd.notna(df[1])
    0     True
    1    False
    Name: 1, dtype: bool
    """
    res = isna(obj)
    if isinstance(res, bool):
        return not res
    return ~res


notnull = notna


def array_equivalent(
    left,
    right,
    strict_nan: bool = False,
    dtype_equal: bool = False,
) -> bool:
    """
    True if two arrays, left and right, have equal non-NaN elements, and NaNs
    in corresponding locations.  False otherwise. It is assumed that left and
    right are NumPy arrays of the same dtype. The behavior of this function
    (particularly with respect to NaNs) is not defined if the dtypes are
    different.

    Parameters
    ----------
    left, right : ndarrays
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    dtype_equal : bool, default False
        Whether `left` and `right` are known to have the same dtype
        according to `is_dtype_equal`. Some methods like `BlockManager.equals`.
        require that the dtypes match. Setting this to ``True`` can improve
        performance, but will give different results for arrays that are
        equal but different dtypes.

    Returns
    -------
    b : bool
        Returns True if the arrays are equivalent.

    Examples
    --------
    >>> array_equivalent(
    ...     np.array([1, 2, np.nan]),
    ...     np.array([1, 2, np.nan]))
    True
    >>> array_equivalent(
    ...     np.array([1, np.nan, 2]),
    ...     np.array([1, 2, np.nan]))
    False
    """
    left, right = np.asarray(left), np.asarray(right)

    # shape compat
    if left.shape != right.shape:
        return False

    if dtype_equal:
        # fastpath when we require that the dtypes match (Block.equals)
        if left.dtype.kind in "fc":
            return _array_equivalent_float(left, right)
        elif left.dtype.kind in "mM":
            return _array_equivalent_datetimelike(left, right)
        elif is_string_or_object_np_dtype(left.dtype):
            # TODO: fastpath for pandas' StringDtype
            return _array_equivalent_object(left, right, strict_nan)
        else:
            return np.array_equal(left, right)

    # Slow path when we allow comparing different dtypes.
    # Object arrays can contain None, NaN and NaT.
    # string dtypes must be come to this path for NumPy 1.7.1 compat
    if left.dtype.kind in "OSU" or right.dtype.kind in "OSU":
        # Note: `in "OSU"` is non-trivially faster than `in ["O", "S", "U"]`
        #  or `in ("O", "S", "U")`
        return _array_equivalent_object(left, right, strict_nan)

    # NaNs can occur in float and complex arrays.
    if left.dtype.kind in "fc":
        if not (left.size and right.size):
            return True
        return ((left == right) | (isna(left) & isna(right))).all()

    elif left.dtype.kind in "mM" or right.dtype.kind in "mM":
        # datetime64, timedelta64, Period
        if left.dtype != right.dtype:
            return False

        left = left.view("i8")
        right = right.view("i8")

    # if we have structured dtypes, compare first
    if (
        left.dtype.type is np.void or right.dtype.type is np.void
    ) and left.dtype != right.dtype:
        return False

    return np.array_equal(left, right)


def _array_equivalent_float(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(((left == right) | (np.isnan(left) & np.isnan(right))).all())


def _array_equivalent_datetimelike(left: np.ndarray, right: np.ndarray):
    return np.array_equal(left.view("i8"), right.view("i8"))


def _array_equivalent_object(left: np.ndarray, right: np.ndarray, strict_nan: bool):
    left = ensure_object(left)
    right = ensure_object(right)

    mask: npt.NDArray[np.bool_] | None = None
    if strict_nan:
        mask = isna(left) & isna(right)
        if not mask.any():
            mask = None

    try:
        if mask is None:
            return lib.array_equivalent_object(left, right)
        if not lib.array_equivalent_object(left[~mask], right[~mask]):
            return False
        left_remaining = left[mask]
        right_remaining = right[mask]
    except ValueError:
        # can raise a ValueError if left and right cannot be
        # compared (e.g. nested arrays)
        left_remaining = left
        right_remaining = right

    for left_value, right_value in zip(left_remaining, right_remaining):
        if left_value is NaT and right_value is not NaT:
            return False

        elif left_value is libmissing.NA and right_value is not libmissing.NA:
            return False

        elif isinstance(left_value, float) and np.isnan(left_value):
            if not isinstance(right_value, float) or not np.isnan(right_value):
                return False
        else:
            with warnings.catch_warnings():
                # suppress numpy's "elementwise comparison failed"
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    if np.any(np.asarray(left_value != right_value)):
                        return False
                except TypeError as err:
                    if "boolean value of NA is ambiguous" in str(err):
                        return False
                    raise
                except ValueError:
                    # numpy can raise a ValueError if left and right cannot be
                    # compared (e.g. nested arrays)
                    return False
    return True


def array_equals(left: ArrayLike, right: ArrayLike) -> bool:
    """
    ExtensionArray-compatible implementation of array_equivalent.
    """
    if left.dtype != right.dtype:
        return False
    elif isinstance(left, ABCExtensionArray):
        return left.equals(right)
    else:
        return array_equivalent(left, right, dtype_equal=True)


def infer_fill_value(val):
    """
    infer the fill value for the nan/NaT from the provided
    scalar/ndarray/list-like if we are a NaT, return the correct dtyped
    element to provide proper block construction
    """
    if not is_list_like(val):
        val = [val]
    val = np.asarray(val)
    if val.dtype.kind in "mM":
        return np.array("NaT", dtype=val.dtype)
    elif val.dtype == object:
        dtype = lib.infer_dtype(ensure_object(val), skipna=False)
        if dtype in ["datetime", "datetime64"]:
            return np.array("NaT", dtype=DT64NS_DTYPE)
        elif dtype in ["timedelta", "timedelta64"]:
            return np.array("NaT", dtype=TD64NS_DTYPE)
        return np.array(np.nan, dtype=object)
    elif val.dtype.kind == "U":
        return np.array(np.nan, dtype=val.dtype)
    return np.nan


def construct_1d_array_from_inferred_fill_value(
    value: object, length: int
) -> ArrayLike:
    # Find our empty_value dtype by constructing an array
    #  from our value and doing a .take on it
    from pandas.core.algorithms import take_nd
    from pandas.core.construction import sanitize_array
    from pandas.core.indexes.base import Index

    arr = sanitize_array(value, Index(range(1)), copy=False)
    taker = -1 * np.ones(length, dtype=np.intp)
    return take_nd(arr, taker)


def maybe_fill(arr: np.ndarray) -> np.ndarray:
    """
    Fill numpy.ndarray with NaN, unless we have a integer or boolean dtype.
    """
    if arr.dtype.kind not in "iub":
        arr.fill(np.nan)
    return arr


def na_value_for_dtype(dtype: DtypeObj, compat: bool = True):
    """
    Return a dtype compat na value

    Parameters
    ----------
    dtype : string / dtype
    compat : bool, default True

    Returns
    -------
    np.dtype or a pandas dtype

    Examples
    --------
    >>> na_value_for_dtype(np.dtype('int64'))
    0
    >>> na_value_for_dtype(np.dtype('int64'), compat=False)
    nan
    >>> na_value_for_dtype(np.dtype('float64'))
    nan
    >>> na_value_for_dtype(np.dtype('bool'))
    False
    >>> na_value_for_dtype(np.dtype('datetime64[ns]'))
    numpy.datetime64('NaT')
    """

    if isinstance(dtype, ExtensionDtype):
        return dtype.na_value
    elif dtype.kind in "mM":
        unit = np.datetime_data(dtype)[0]
        return dtype.type("NaT", unit)
    elif dtype.kind == "f":
        return np.nan
    elif dtype.kind in "iu":
        if compat:
            return 0
        return np.nan
    elif dtype.kind == "b":
        if compat:
            return False
        return np.nan
    return np.nan


def remove_na_arraylike(arr: Series | Index | np.ndarray):
    """
    Return array-like containing only true/non-NaN values, possibly empty.
    """
    if isinstance(arr.dtype, ExtensionDtype):
        return arr[notna(arr)]
    else:
        return arr[notna(np.asarray(arr))]


def is_valid_na_for_dtype(obj, dtype: DtypeObj) -> bool:
    """
    isna check that excludes incompatible dtypes

    Parameters
    ----------
    obj : object
    dtype : np.datetime64, np.timedelta64, DatetimeTZDtype, or PeriodDtype

    Returns
    -------
    bool
    """
    if not lib.is_scalar(obj) or not isna(obj):
        return False
    elif dtype.kind == "M":
        if isinstance(dtype, np.dtype):
            # i.e. not tzaware
            return not isinstance(obj, (np.timedelta64, Decimal))
        # we have to rule out tznaive dt64("NaT")
        return not isinstance(obj, (np.timedelta64, np.datetime64, Decimal))
    elif dtype.kind == "m":
        return not isinstance(obj, (np.datetime64, Decimal))
    elif dtype.kind in "iufc":
        # Numeric
        return obj is not NaT and not isinstance(obj, (np.datetime64, np.timedelta64))
    elif dtype.kind == "b":
        # We allow pd.NA, None, np.nan in BooleanArray (same as IntervalDtype)
        return lib.is_float(obj) or obj is None or obj is libmissing.NA

    elif dtype == _dtype_str:
        # numpy string dtypes to avoid float np.nan
        return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal, float))

    elif dtype == _dtype_object:
        # This is needed for Categorical, but is kind of weird
        return True

    elif isinstance(dtype, PeriodDtype):
        return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal))

    elif isinstance(dtype, IntervalDtype):
        return lib.is_float(obj) or obj is None or obj is libmissing.NA

    elif isinstance(dtype, CategoricalDtype):
        return is_valid_na_for_dtype(obj, dtype.categories.dtype)

    # fallback, default to allowing NaN, None, NA, NaT
    return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal))


def isna_all(arr: ArrayLike) -> bool:
    """
    Optimized equivalent to isna(arr).all()
    """
    total_len = len(arr)

    # Usually it's enough to check but a small fraction of values to see if
    #  a block is NOT null, chunks should help in such cases.
    #  parameters 1000 and 40 were chosen arbitrarily
    chunk_len = max(total_len // 40, 1000)

    dtype = arr.dtype
    if lib.is_np_dtype(dtype, "f"):
        checker = nan_checker

    elif (lib.is_np_dtype(dtype, "mM")) or isinstance(
        dtype, (DatetimeTZDtype, PeriodDtype)
    ):
        # error: Incompatible types in assignment (expression has type
        # "Callable[[Any], Any]", variable has type "ufunc")
        checker = lambda x: np.asarray(x.view("i8")) == iNaT  # type: ignore[assignment]

    else:
        # error: Incompatible types in assignment (expression has type "Callable[[Any],
        # Any]", variable has type "ufunc")
        checker = lambda x: _isna_array(  # type: ignore[assignment]
            x, inf_as_na=INF_AS_NA
        )

    return all(
        checker(arr[i : i + chunk_len]).all() for i in range(0, total_len, chunk_len)
    )