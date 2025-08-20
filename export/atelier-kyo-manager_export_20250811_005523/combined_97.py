
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\solvers.py ===
from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy, uniquely_named_symbol, symbols
from sympy.utilities.iterables import numbered_symbols

from .exceptions import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .eigen import _fuzzy_positive_definite
from .utilities import _get_intermediate_simp, _iszero


def _diagonal_solve(M, rhs):
    """Solves ``Ax = B`` efficiently, where A is a diagonal Matrix,
    with non-zero diagonal entries.

    Examples
    ========

    >>> from sympy import Matrix, eye
    >>> A = eye(2)*2
    >>> B = Matrix([[1, 2], [3, 4]])
    >>> A.diagonal_solve(B) == B/2
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_diagonal():
        raise TypeError("Matrix should be diagonal")
    if rhs.rows != M.rows:
        raise TypeError("Size mismatch")

    return M._new(
        rhs.rows, rhs.cols, lambda i, j: rhs[i, j] / M[i, i])


def _lower_triangular_solve(M, rhs):
    """Solves ``Ax = B``, where A is a lower triangular matrix.

    See Also
    ========

    upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrices size mismatch.")
    if not M.is_lower:
        raise ValueError("Matrix must be lower triangular.")

    dps = _get_intermediate_simp()
    X   = MutableDenseMatrix.zeros(M.rows, rhs.cols)

    for j in range(rhs.cols):
        for i in range(M.rows):
            if M[i, i] == 0:
                raise TypeError("Matrix must be non-singular.")

            X[i, j] = dps((rhs[i, j] - sum(M[i, k]*X[k, j]
                                        for k in range(i))) / M[i, i])

    return M._new(X)

def _lower_triangular_solve_sparse(M, rhs):
    """Solves ``Ax = B``, where A is a lower triangular matrix.

    See Also
    ========

    upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrices size mismatch.")
    if not M.is_lower:
        raise ValueError("Matrix must be lower triangular.")

    dps  = _get_intermediate_simp()
    rows = [[] for i in range(M.rows)]

    for i, j, v in M.row_list():
        if i > j:
            rows[i].append((j, v))

    X = rhs.as_mutable()

    for j in range(rhs.cols):
        for i in range(rhs.rows):
            for u, v in rows[i]:
                X[i, j] -= v*X[u, j]

            X[i, j] = dps(X[i, j] / M[i, i])

    return M._new(X)


def _upper_triangular_solve(M, rhs):
    """Solves ``Ax = B``, where A is an upper triangular matrix.

    See Also
    ========

    lower_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrix size mismatch.")
    if not M.is_upper:
        raise TypeError("Matrix is not upper triangular.")

    dps = _get_intermediate_simp()
    X   = MutableDenseMatrix.zeros(M.rows, rhs.cols)

    for j in range(rhs.cols):
        for i in reversed(range(M.rows)):
            if M[i, i] == 0:
                raise ValueError("Matrix must be non-singular.")

            X[i, j] = dps((rhs[i, j] - sum(M[i, k]*X[k, j]
                                        for k in range(i + 1, M.rows))) / M[i, i])

    return M._new(X)

def _upper_triangular_solve_sparse(M, rhs):
    """Solves ``Ax = B``, where A is an upper triangular matrix.

    See Also
    ========

    lower_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrix size mismatch.")
    if not M.is_upper:
        raise TypeError("Matrix is not upper triangular.")

    dps  = _get_intermediate_simp()
    rows = [[] for i in range(M.rows)]

    for i, j, v in M.row_list():
        if i < j:
            rows[i].append((j, v))

    X = rhs.as_mutable()

    for j in range(rhs.cols):
        for i in reversed(range(rhs.rows)):
            for u, v in reversed(rows[i]):
                X[i, j] -= v*X[u, j]

            X[i, j] = dps(X[i, j] / M[i, i])

    return M._new(X)


def _cholesky_solve(M, rhs):
    """Solves ``Ax = B`` using Cholesky decomposition,
    for a general square non-singular matrix.
    For a non-square matrix with rows > cols,
    the least squares solution is returned.

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if M.rows < M.cols:
        raise NotImplementedError(
            'Under-determined System. Try M.gauss_jordan_solve(rhs)')

    hermitian = True
    reform    = False

    if M.is_symmetric():
        hermitian = False
    elif not M.is_hermitian:
        reform = True

    if reform or _fuzzy_positive_definite(M) is False:
        H         = M.H
        M         = H.multiply(M)
        rhs       = H.multiply(rhs)
        hermitian = not M.is_symmetric()

    L = M.cholesky(hermitian=hermitian)
    Y = L.lower_triangular_solve(rhs)

    if hermitian:
        return (L.H).upper_triangular_solve(Y)
    else:
        return (L.T).upper_triangular_solve(Y)


def _LDLsolve(M, rhs):
    """Solves ``Ax = B`` using LDL decomposition,
    for a general square and non-singular matrix.

    For a non-square matrix with rows > cols,
    the least squares solution is returned.

    Examples
    ========

    >>> from sympy import Matrix, eye
    >>> A = eye(2)*2
    >>> B = Matrix([[1, 2], [3, 4]])
    >>> A.LDLsolve(B) == B/2
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if M.rows < M.cols:
        raise NotImplementedError(
            'Under-determined System. Try M.gauss_jordan_solve(rhs)')

    hermitian = True
    reform    = False

    if M.is_symmetric():
        hermitian = False
    elif not M.is_hermitian:
        reform = True

    if reform or _fuzzy_positive_definite(M) is False:
        H         = M.H
        M         = H.multiply(M)
        rhs       = H.multiply(rhs)
        hermitian = not M.is_symmetric()

    L, D = M.LDLdecomposition(hermitian=hermitian)
    Y    = L.lower_triangular_solve(rhs)
    Z    = D.diagonal_solve(Y)

    if hermitian:
        return (L.H).upper_triangular_solve(Z)
    else:
        return (L.T).upper_triangular_solve(Z)


def _LUsolve(M, rhs, iszerofunc=_iszero):
    """Solve the linear system ``Ax = rhs`` for ``x`` where ``A = M``.

    This is for symbolic matrices, for real or complex ones use
    mpmath.lu_solve or mpmath.qr_solve.

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    QRsolve
    pinv_solve
    LUdecomposition
    cramer_solve
    """

    if rhs.rows != M.rows:
        raise ShapeError(
            "``M`` and ``rhs`` must have the same number of rows.")

    m = M.rows
    n = M.cols

    if m < n:
        raise NotImplementedError("Underdetermined systems not supported.")

    try:
        A, perm = M.LUdecomposition_Simple(
            iszerofunc=iszerofunc, rankcheck=True)
    except ValueError:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    dps = _get_intermediate_simp()
    b   = rhs.permute_rows(perm).as_mutable()

    # forward substitution, all diag entries are scaled to 1
    for i in range(m):
        for j in range(min(i, n)):
            scale = A[i, j]
            b.zip_row_op(i, j, lambda x, y: dps(x - scale * y))

    # consistency check for overdetermined systems
    if m > n:
        for i in range(n, m):
            for j in range(b.cols):
                if not iszerofunc(b[i, j]):
                    raise ValueError("The system is inconsistent.")

        b = b[0:n, :]   # truncate zero rows if consistent

    # backward substitution
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            scale = A[i, j]
            b.zip_row_op(i, j, lambda x, y: dps(x - scale * y))

        scale = A[i, i]
        b.row_op(i, lambda x, _: dps(scale**-1 * x))

    return rhs.__class__(b)


def _QRsolve(M, b):
    """Solve the linear system ``Ax = b``.

    ``M`` is the matrix ``A``, the method argument is the vector
    ``b``.  The method returns the solution vector ``x``.  If ``b`` is a
    matrix, the system is solved for each column of ``b`` and the
    return value is a matrix of the same shape as ``b``.

    This method is slower (approximately by a factor of 2) but
    more stable for floating-point arithmetic than the LUsolve method.
    However, LUsolve usually uses an exact arithmetic, so you do not need
    to use QRsolve.

    This is mainly for educational purposes and symbolic matrices, for real
    (or complex) matrices use mpmath.qr_solve.

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    pinv_solve
    QRdecomposition
    cramer_solve
    """

    dps  = _get_intermediate_simp(expand_mul, expand_mul)
    Q, R = M.QRdecomposition()
    y    = Q.T * b

    # back substitution to solve R*x = y:
    # We build up the result "backwards" in the vector 'x' and reverse it
    # only in the end.
    x = []
    n = R.rows

    for j in range(n - 1, -1, -1):
        tmp = y[j, :]

        for k in range(j + 1, n):
            tmp -= R[j, k] * x[n - 1 - k]

        tmp = dps(tmp)

        x.append(tmp / R[j, j])

    return M.vstack(*x[::-1])


def _gauss_jordan_solve(M, B, freevar=False):
    """
    Solves ``Ax = B`` using Gauss Jordan elimination.

    There may be zero, one, or infinite solutions.  If one solution
    exists, it will be returned. If infinite solutions exist, it will
    be returned parametrically. If no solutions exist, It will throw
    ValueError.

    Parameters
    ==========

    B : Matrix
        The right hand side of the equation to be solved for.  Must have
        the same number of rows as matrix A.

    freevar : boolean, optional
        Flag, when set to `True` will return the indices of the free
        variables in the solutions (column Matrix), for a system that is
        undetermined (e.g. A has more columns than rows), for which
        infinite solutions are possible, in terms of arbitrary
        values of free variables. Default `False`.

    Returns
    =======

    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    params : Matrix
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of arbitrary
        parameters. These arbitrary parameters are returned as params
        Matrix.

    free_var_index : List, optional
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of arbitrary
        values of free variables. Then the indices of the free variables
        in the solutions (column Matrix) are returned by free_var_index,
        if the flag `freevar` is set to `True`.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 1, 1], [1, 2, 2, -1], [2, 4, 0, 6]])
    >>> B = Matrix([7, 12, 4])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [-2*tau0 - 3*tau1 + 2],
    [                 tau0],
    [           2*tau1 + 5],
    [                 tau1]])
    >>> params
    Matrix([
    [tau0],
    [tau1]])
    >>> taus_zeroes = { tau:0 for tau in params }
    >>> sol_unique = sol.xreplace(taus_zeroes)
    >>> sol_unique
        Matrix([
    [2],
    [0],
    [5],
    [0]])


    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    >>> B = Matrix([3, 6, 9])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [-1],
    [ 2],
    [ 0]])
    >>> params
    Matrix(0, 1, [])

    >>> A = Matrix([[2, -7], [-1, 4]])
    >>> B = Matrix([[-21, 3], [12, -2]])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [0, -2],
    [3, -1]])
    >>> params
    Matrix(0, 2, [])


    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 1, 1], [1, 2, 2, -1], [2, 4, 0, 6]])
    >>> B = Matrix([7, 12, 4])
    >>> sol, params, freevars = A.gauss_jordan_solve(B, freevar=True)
    >>> sol
    Matrix([
    [-2*tau0 - 3*tau1 + 2],
    [                 tau0],
    [           2*tau1 + 5],
    [                 tau1]])
    >>> params
    Matrix([
    [tau0],
    [tau1]])
    >>> freevars
    [1, 3]


    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_elimination

    """

    from sympy.matrices import Matrix, zeros

    cls      = M.__class__
    aug      = M.hstack(M.copy(), B.copy())
    B_cols   = B.cols
    row, col = aug[:, :-B_cols].shape

    # solve by reduced row echelon form
    A, pivots = aug.rref(simplify=True)
    A, v      = A[:, :-B_cols], A[:, -B_cols:]
    pivots    = list(filter(lambda p: p < col, pivots))
    rank      = len(pivots)

    # Get index of free symbols (free parameters)
    # non-pivots columns are free variables
    free_var_index = [c for c in range(A.cols) if c not in pivots]

    # Bring to block form
    permutation = Matrix(pivots + free_var_index).T

    # check for existence of solutions
    # rank of aug Matrix should be equal to rank of coefficient matrix
    if not v[rank:, :].is_zero_matrix:
        raise ValueError("Linear system has no solution")

    # Free parameters
    # what are current unnumbered free symbol names?
    name = uniquely_named_symbol('tau', [aug],
            compare=lambda i: str(i).rstrip('1234567890'),
            modify=lambda s: '_' + s).name
    gen  = numbered_symbols(name)
    tau  = Matrix([next(gen) for k in range((col - rank)*B_cols)]).reshape(
            col - rank, B_cols)

    # Full parametric solution
    V        = A[:rank, free_var_index]
    vt       = v[:rank, :]
    free_sol = tau.vstack(vt - V * tau, tau)

    # Undo permutation
    sol = zeros(col, B_cols)

    for k in range(col):
        sol[permutation[k], :] = free_sol[k,:]

    sol, tau = cls(sol), cls(tau)

    if freevar:
        return sol, tau, free_var_index
    else:
        return sol, tau


def _pinv_solve(M, B, arbitrary_matrix=None):
    """Solve ``Ax = B`` using the Moore-Penrose pseudoinverse.

    There may be zero, one, or infinite solutions.  If one solution
    exists, it will be returned.  If infinite solutions exist, one will
    be returned based on the value of arbitrary_matrix.  If no solutions
    exist, the least-squares solution is returned.

    Parameters
    ==========

    B : Matrix
        The right hand side of the equation to be solved for.  Must have
        the same number of rows as matrix A.
    arbitrary_matrix : Matrix
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of an arbitrary
        matrix.  This parameter may be set to a specific matrix to use
        for that purpose; if so, it must be the same shape as x, with as
        many rows as matrix A has columns, and as many columns as matrix
        B.  If left as None, an appropriate matrix containing dummy
        symbols in the form of ``wn_m`` will be used, with n and m being
        row and column position of each symbol.

    Returns
    =======

    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> B = Matrix([7, 8])
    >>> A.pinv_solve(B)
    Matrix([
    [ _w0_0/6 - _w1_0/3 + _w2_0/6 - 55/18],
    [-_w0_0/3 + 2*_w1_0/3 - _w2_0/3 + 1/9],
    [ _w0_0/6 - _w1_0/3 + _w2_0/6 + 59/18]])
    >>> A.pinv_solve(B, arbitrary_matrix=Matrix([0, 0, 0]))
    Matrix([
    [-55/18],
    [   1/9],
    [ 59/18]])

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv

    Notes
    =====

    This may return either exact solutions or least squares solutions.
    To determine which, check ``A * A.pinv() * B == B``.  It will be
    True if exact solutions exist, and False if only a least-squares
    solution exists.  Be aware that the left hand side of that equation
    may need to be simplified to correctly compare to the right hand
    side.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse#Obtaining_all_solutions_of_a_linear_system

    """

    from sympy.matrices import eye

    A      = M
    A_pinv = M.pinv()

    if arbitrary_matrix is None:
        rows, cols       = A.cols, B.cols
        w                = symbols('w:{}_:{}'.format(rows, cols), cls=Dummy)
        arbitrary_matrix = M.__class__(cols, rows, w).T

    return A_pinv.multiply(B) + (eye(A.cols) -
            A_pinv.multiply(A)).multiply(arbitrary_matrix)


def _cramer_solve(M, rhs, det_method="laplace"):
    """Solves system of linear equations using Cramer's rule.

    This method is relatively inefficient compared to other methods.
    However it only uses a single division, assuming a division-free determinant
    method is provided. This is helpful to minimize the chance of divide-by-zero
    cases in symbolic solutions to linear systems.

    Parameters
    ==========
    M : Matrix
        The matrix representing the left hand side of the equation.
    rhs : Matrix
        The matrix representing the right hand side of the equation.
    det_method : str or callable
        The method to use to calculate the determinant of the matrix.
        The default is ``'laplace'``.  If a callable is passed, it should take a
        single argument, the matrix, and return the determinant of the matrix.

    Returns
    =======
    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[0, -6, 1], [0, -6, -1], [-5, -2, 3]])
    >>> B = Matrix([[-30, -9], [-18, -27], [-26, 46]])
    >>> x = A.cramer_solve(B)
    >>> x
    Matrix([
    [ 0, -5],
    [ 4,  3],
    [-6,  9]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cramer%27s_rule#Explicit_formulas_for_small_systems

    """
    from .dense import zeros

    def entry(i, j):
        return rhs[i, sol] if j == col else M[i, j]

    if det_method == "bird":
        from .determinant import _det_bird
        det = _det_bird
    elif det_method == "laplace":
        from .determinant import _det_laplace
        det = _det_laplace
    elif isinstance(det_method, str):
        det = lambda matrix: matrix.det(method=det_method)
    else:
        det = det_method
    det_M = det(M)
    x = zeros(*rhs.shape)
    for sol in range(rhs.shape[1]):
        for col in range(rhs.shape[0]):
            x[col, sol] = det(M.__class__(*M.shape, entry)) / det_M
    return M.__class__(x)


def _solve(M, rhs, method='GJ'):
    """Solves linear equation where the unique solution exists.

    Parameters
    ==========

    rhs : Matrix
        Vector representing the right hand side of the linear equation.

    method : string, optional
        If set to ``'GJ'`` or ``'GE'``, the Gauss-Jordan elimination will be
        used, which is implemented in the routine ``gauss_jordan_solve``.

        If set to ``'LU'``, ``LUsolve`` routine will be used.

        If set to ``'QR'``, ``QRsolve`` routine will be used.

        If set to ``'PINV'``, ``pinv_solve`` routine will be used.

        If set to ``'CRAMER'``, ``cramer_solve`` routine will be used.

        It also supports the methods available for special linear systems

        For positive definite systems:

        If set to ``'CH'``, ``cholesky_solve`` routine will be used.

        If set to ``'LDL'``, ``LDLsolve`` routine will be used.

        To use a different method and to compute the solution via the
        inverse, use a method defined in the .inv() docstring.

    Returns
    =======

    solutions : Matrix
        Vector representing the solution.

    Raises
    ======

    ValueError
        If there is not a unique solution then a ``ValueError`` will be
        raised.

        If ``M`` is not square, a ``ValueError`` and a different routine
        for solving the system will be suggested.
    """

    if method in ('GJ', 'GE'):
        try:
            soln, param = M.gauss_jordan_solve(rhs)

            if param:
                raise NonInvertibleMatrixError("Matrix det == 0; not invertible. "
                "Try ``M.gauss_jordan_solve(rhs)`` to obtain a parametric solution.")

        except ValueError:
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

        return soln

    elif method == 'LU':
        return M.LUsolve(rhs)
    elif method == 'CH':
        return M.cholesky_solve(rhs)
    elif method == 'QR':
        return M.QRsolve(rhs)
    elif method == 'LDL':
        return M.LDLsolve(rhs)
    elif method == 'PINV':
        return M.pinv_solve(rhs)
    elif method == 'CRAMER':
        return M.cramer_solve(rhs)
    else:
        return M.inv(method=method).multiply(rhs)


def _solve_least_squares(M, rhs, method='CH'):
    """Return the least-square fit to the data.

    Parameters
    ==========

    rhs : Matrix
        Vector representing the right hand side of the linear equation.

    method : string or boolean, optional
        If set to ``'CH'``, ``cholesky_solve`` routine will be used.

        If set to ``'LDL'``, ``LDLsolve`` routine will be used.

        If set to ``'QR'``, ``QRsolve`` routine will be used.

        If set to ``'PINV'``, ``pinv_solve`` routine will be used.

        Otherwise, the conjugate of ``M`` will be used to create a system
        of equations that is passed to ``solve`` along with the hint
        defined by ``method``.

    Returns
    =======

    solutions : Matrix
        Vector representing the solution.

    Examples
    ========

    >>> from sympy import Matrix, ones
    >>> A = Matrix([1, 2, 3])
    >>> B = Matrix([2, 3, 4])
    >>> S = Matrix(A.row_join(B))
    >>> S
    Matrix([
    [1, 2],
    [2, 3],
    [3, 4]])

    If each line of S represent coefficients of Ax + By
    and x and y are [2, 3] then S*xy is:

    >>> r = S*Matrix([2, 3]); r
    Matrix([
    [ 8],
    [13],
    [18]])

    But let's add 1 to the middle value and then solve for the
    least-squares value of xy:

    >>> xy = S.solve_least_squares(Matrix([8, 14, 18])); xy
    Matrix([
    [ 5/3],
    [10/3]])

    The error is given by S*xy - r:

    >>> S*xy - r
    Matrix([
    [1/3],
    [1/3],
    [1/3]])
    >>> _.norm().n(2)
    0.58

    If a different xy is used, the norm will be higher:

    >>> xy += ones(2, 1)/10
    >>> (S*xy - r).norm().n(2)
    1.5

    """

    if method == 'CH':
        return M.cholesky_solve(rhs)
    elif method == 'QR':
        return M.QRsolve(rhs)
    elif method == 'LDL':
        return M.LDLsolve(rhs)
    elif method == 'PINV':
        return M.pinv_solve(rhs)
    else:
        t = M.H
        return (t * M).solve(t * rhs, method=method)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\__init__.py ===
"""
NumPy
=====

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <https://numpy.org>`_.

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as ``np``::

  >>> import numpy as np

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.

Available subpackages
---------------------
lib
    Basic functions used by several sub-packages.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
polynomial
    Polynomial tools
testing
    NumPy testing tools
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more (for Python <= 3.11)

Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
__version__
    NumPy version string

Viewing documentation using IPython
-----------------------------------

Start IPython and import `numpy` usually under the alias ``np``: `import
numpy as np`.  Then, directly past or use the ``%cpaste`` magic to paste
examples into the shell.  To see which functions are available in `numpy`,
type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

Copies vs. in-place operation
-----------------------------
Most of the functions in `numpy` return a copy of the array argument
(e.g., `np.sort`).  In-place versions of these functions are often
available as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.
Exceptions to this rule are documented.

"""


# start delvewheel patch
def _delvewheel_patch_1_10_1():
    import os
    if os.path.isdir(libs_dir := os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'numpy.libs'))):
        os.add_dll_directory(libs_dir)


_delvewheel_patch_1_10_1()
del _delvewheel_patch_1_10_1
# end delvewheel patch

import os
import sys
import warnings

# If a version with git hash was stored, use that instead
from . import version
from ._expired_attrs_2_0 import __expired_attributes__
from ._globals import _CopyMode, _NoValue
from .version import __version__

# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __NUMPY_SETUP__  # noqa: B018
except NameError:
    __NUMPY_SETUP__ = False

if __NUMPY_SETUP__:
    sys.stderr.write('Running from numpy source directory.\n')
else:
    # Allow distributors to run custom init code before importing numpy._core
    from . import _distributor_init

    try:
        from numpy.__config__ import show_config
    except ImportError as e:
        msg = """Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there."""
        raise ImportError(msg) from e

    from . import _core
    from ._core import (
        False_,
        ScalarType,
        True_,
        abs,
        absolute,
        acos,
        acosh,
        add,
        all,
        allclose,
        amax,
        amin,
        any,
        arange,
        arccos,
        arccosh,
        arcsin,
        arcsinh,
        arctan,
        arctan2,
        arctanh,
        argmax,
        argmin,
        argpartition,
        argsort,
        argwhere,
        around,
        array,
        array2string,
        array_equal,
        array_equiv,
        array_repr,
        array_str,
        asanyarray,
        asarray,
        ascontiguousarray,
        asfortranarray,
        asin,
        asinh,
        astype,
        atan,
        atan2,
        atanh,
        atleast_1d,
        atleast_2d,
        atleast_3d,
        base_repr,
        binary_repr,
        bitwise_and,
        bitwise_count,
        bitwise_invert,
        bitwise_left_shift,
        bitwise_not,
        bitwise_or,
        bitwise_right_shift,
        bitwise_xor,
        block,
        bool,
        bool_,
        broadcast,
        busday_count,
        busday_offset,
        busdaycalendar,
        byte,
        bytes_,
        can_cast,
        cbrt,
        cdouble,
        ceil,
        character,
        choose,
        clip,
        clongdouble,
        complex64,
        complex128,
        complexfloating,
        compress,
        concat,
        concatenate,
        conj,
        conjugate,
        convolve,
        copysign,
        copyto,
        correlate,
        cos,
        cosh,
        count_nonzero,
        cross,
        csingle,
        cumprod,
        cumsum,
        cumulative_prod,
        cumulative_sum,
        datetime64,
        datetime_as_string,
        datetime_data,
        deg2rad,
        degrees,
        diagonal,
        divide,
        divmod,
        dot,
        double,
        dtype,
        e,
        einsum,
        einsum_path,
        empty,
        empty_like,
        equal,
        errstate,
        euler_gamma,
        exp,
        exp2,
        expm1,
        fabs,
        finfo,
        flatiter,
        flatnonzero,
        flexible,
        float16,
        float32,
        float64,
        float_power,
        floating,
        floor,
        floor_divide,
        fmax,
        fmin,
        fmod,
        format_float_positional,
        format_float_scientific,
        frexp,
        from_dlpack,
        frombuffer,
        fromfile,
        fromfunction,
        fromiter,
        frompyfunc,
        fromstring,
        full,
        full_like,
        gcd,
        generic,
        geomspace,
        get_printoptions,
        getbufsize,
        geterr,
        geterrcall,
        greater,
        greater_equal,
        half,
        heaviside,
        hstack,
        hypot,
        identity,
        iinfo,
        indices,
        inexact,
        inf,
        inner,
        int8,
        int16,
        int32,
        int64,
        int_,
        intc,
        integer,
        intp,
        invert,
        is_busday,
        isclose,
        isdtype,
        isfinite,
        isfortran,
        isinf,
        isnan,
        isnat,
        isscalar,
        issubdtype,
        lcm,
        ldexp,
        left_shift,
        less,
        less_equal,
        lexsort,
        linspace,
        little_endian,
        log,
        log1p,
        log2,
        log10,
        logaddexp,
        logaddexp2,
        logical_and,
        logical_not,
        logical_or,
        logical_xor,
        logspace,
        long,
        longdouble,
        longlong,
        matmul,
        matrix_transpose,
        matvec,
        max,
        maximum,
        may_share_memory,
        mean,
        memmap,
        min,
        min_scalar_type,
        minimum,
        mod,
        modf,
        moveaxis,
        multiply,
        nan,
        ndarray,
        ndim,
        nditer,
        negative,
        nested_iters,
        newaxis,
        nextafter,
        nonzero,
        not_equal,
        number,
        object_,
        ones,
        ones_like,
        outer,
        partition,
        permute_dims,
        pi,
        positive,
        pow,
        power,
        printoptions,
        prod,
        promote_types,
        ptp,
        put,
        putmask,
        rad2deg,
        radians,
        ravel,
        recarray,
        reciprocal,
        record,
        remainder,
        repeat,
        require,
        reshape,
        resize,
        result_type,
        right_shift,
        rint,
        roll,
        rollaxis,
        round,
        sctypeDict,
        searchsorted,
        set_printoptions,
        setbufsize,
        seterr,
        seterrcall,
        shape,
        shares_memory,
        short,
        sign,
        signbit,
        signedinteger,
        sin,
        single,
        sinh,
        size,
        sort,
        spacing,
        sqrt,
        square,
        squeeze,
        stack,
        std,
        str_,
        subtract,
        sum,
        swapaxes,
        take,
        tan,
        tanh,
        tensordot,
        timedelta64,
        trace,
        transpose,
        true_divide,
        trunc,
        typecodes,
        ubyte,
        ufunc,
        uint,
        uint8,
        uint16,
        uint32,
        uint64,
        uintc,
        uintp,
        ulong,
        ulonglong,
        unsignedinteger,
        unstack,
        ushort,
        var,
        vdot,
        vecdot,
        vecmat,
        void,
        vstack,
        where,
        zeros,
        zeros_like,
    )

    # NOTE: It's still under discussion whether these aliases
    # should be removed.
    for ta in ["float96", "float128", "complex192", "complex256"]:
        try:
            globals()[ta] = getattr(_core, ta)
        except AttributeError:
            pass
    del ta

    from . import lib
    from . import matrixlib as _mat
    from .lib import scimath as emath
    from .lib._arraypad_impl import pad
    from .lib._arraysetops_impl import (
        ediff1d,
        in1d,
        intersect1d,
        isin,
        setdiff1d,
        setxor1d,
        union1d,
        unique,
        unique_all,
        unique_counts,
        unique_inverse,
        unique_values,
    )
    from .lib._function_base_impl import (
        angle,
        append,
        asarray_chkfinite,
        average,
        bartlett,
        bincount,
        blackman,
        copy,
        corrcoef,
        cov,
        delete,
        diff,
        digitize,
        extract,
        flip,
        gradient,
        hamming,
        hanning,
        i0,
        insert,
        interp,
        iterable,
        kaiser,
        median,
        meshgrid,
        percentile,
        piecewise,
        place,
        quantile,
        rot90,
        select,
        sinc,
        sort_complex,
        trapezoid,
        trapz,
        trim_zeros,
        unwrap,
        vectorize,
    )
    from .lib._histograms_impl import histogram, histogram_bin_edges, histogramdd
    from .lib._index_tricks_impl import (
        c_,
        diag_indices,
        diag_indices_from,
        fill_diagonal,
        index_exp,
        ix_,
        mgrid,
        ndenumerate,
        ndindex,
        ogrid,
        r_,
        ravel_multi_index,
        s_,
        unravel_index,
    )
    from .lib._nanfunctions_impl import (
        nanargmax,
        nanargmin,
        nancumprod,
        nancumsum,
        nanmax,
        nanmean,
        nanmedian,
        nanmin,
        nanpercentile,
        nanprod,
        nanquantile,
        nanstd,
        nansum,
        nanvar,
    )
    from .lib._npyio_impl import (
        fromregex,
        genfromtxt,
        load,
        loadtxt,
        packbits,
        save,
        savetxt,
        savez,
        savez_compressed,
        unpackbits,
    )
    from .lib._polynomial_impl import (
        poly,
        poly1d,
        polyadd,
        polyder,
        polydiv,
        polyfit,
        polyint,
        polymul,
        polysub,
        polyval,
        roots,
    )
    from .lib._shape_base_impl import (
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
        row_stack,
        split,
        take_along_axis,
        tile,
        vsplit,
    )
    from .lib._stride_tricks_impl import (
        broadcast_arrays,
        broadcast_shapes,
        broadcast_to,
    )
    from .lib._twodim_base_impl import (
        diag,
        diagflat,
        eye,
        fliplr,
        flipud,
        histogram2d,
        mask_indices,
        tri,
        tril,
        tril_indices,
        tril_indices_from,
        triu,
        triu_indices,
        triu_indices_from,
        vander,
    )
    from .lib._type_check_impl import (
        common_type,
        imag,
        iscomplex,
        iscomplexobj,
        isreal,
        isrealobj,
        mintypecode,
        nan_to_num,
        real,
        real_if_close,
        typename,
    )
    from .lib._ufunclike_impl import fix, isneginf, isposinf
    from .lib._utils_impl import get_include, info, show_runtime
    from .matrixlib import asmatrix, bmat, matrix

    # public submodules are imported lazily, therefore are accessible from
    # __getattr__. Note that `distutils` (deprecated) and `array_api`
    # (experimental label) are not added here, because `from numpy import *`
    # must not raise any warnings - that's too disruptive.
    __numpy_submodules__ = {
        "linalg", "fft", "dtypes", "random", "polynomial", "ma",
        "exceptions", "lib", "ctypeslib", "testing", "typing",
        "f2py", "test", "rec", "char", "core", "strings",
    }

    # We build warning messages for former attributes
    _msg = (
        "module 'numpy' has no attribute '{n}'.\n"
        "`np.{n}` was a deprecated alias for the builtin `{n}`. "
        "To avoid this error in existing code, use `{n}` by itself. "
        "Doing this will not modify any behavior and is safe. {extended_msg}\n"
        "The aliases was originally deprecated in NumPy 1.20; for more "
        "details and guidance see the original release note at:\n"
        "    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations")

    _specific_msg = (
        "If you specifically wanted the numpy scalar type, use `np.{}` here.")

    _int_extended_msg = (
        "When replacing `np.{}`, you may wish to use e.g. `np.int64` "
        "or `np.int32` to specify the precision. If you wish to review "
        "your current use, check the release note link for "
        "additional information.")

    _type_info = [
        ("object", ""),  # The NumPy scalar only exists by name.
        ("float", _specific_msg.format("float64")),
        ("complex", _specific_msg.format("complex128")),
        ("str", _specific_msg.format("str_")),
        ("int", _int_extended_msg.format("int"))]

    __former_attrs__ = {
         n: _msg.format(n=n, extended_msg=extended_msg)
         for n, extended_msg in _type_info
     }

    # Some of these could be defined right away, but most were aliases to
    # the Python objects and only removed in NumPy 1.24.  Defining them should
    # probably wait for NumPy 1.26 or 2.0.
    # When defined, these should possibly not be added to `__all__` to avoid
    # import with `from numpy import *`.
    __future_scalars__ = {"str", "bytes", "object"}

    __array_api_version__ = "2024.12"

    from ._array_api_info import __array_namespace_info__

    # now that numpy core module is imported, can initialize limits
    _core.getlimits._register_known_types()

    __all__ = list(
        __numpy_submodules__ |
        set(_core.__all__) |
        set(_mat.__all__) |
        set(lib._histograms_impl.__all__) |
        set(lib._nanfunctions_impl.__all__) |
        set(lib._function_base_impl.__all__) |
        set(lib._twodim_base_impl.__all__) |
        set(lib._shape_base_impl.__all__) |
        set(lib._type_check_impl.__all__) |
        set(lib._arraysetops_impl.__all__) |
        set(lib._ufunclike_impl.__all__) |
        set(lib._arraypad_impl.__all__) |
        set(lib._utils_impl.__all__) |
        set(lib._stride_tricks_impl.__all__) |
        set(lib._polynomial_impl.__all__) |
        set(lib._npyio_impl.__all__) |
        set(lib._index_tricks_impl.__all__) |
        {"emath", "show_config", "__version__", "__array_namespace_info__"}
    )

    # Filter out Cython harmless warnings
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    def __getattr__(attr):
        # Warn for expired attributes
        import warnings

        if attr == "linalg":
            import numpy.linalg as linalg
            return linalg
        elif attr == "fft":
            import numpy.fft as fft
            return fft
        elif attr == "dtypes":
            import numpy.dtypes as dtypes
            return dtypes
        elif attr == "random":
            import numpy.random as random
            return random
        elif attr == "polynomial":
            import numpy.polynomial as polynomial
            return polynomial
        elif attr == "ma":
            import numpy.ma as ma
            return ma
        elif attr == "ctypeslib":
            import numpy.ctypeslib as ctypeslib
            return ctypeslib
        elif attr == "exceptions":
            import numpy.exceptions as exceptions
            return exceptions
        elif attr == "testing":
            import numpy.testing as testing
            return testing
        elif attr == "matlib":
            import numpy.matlib as matlib
            return matlib
        elif attr == "f2py":
            import numpy.f2py as f2py
            return f2py
        elif attr == "typing":
            import numpy.typing as typing
            return typing
        elif attr == "rec":
            import numpy.rec as rec
            return rec
        elif attr == "char":
            import numpy.char as char
            return char
        elif attr == "array_api":
            raise AttributeError("`numpy.array_api` is not available from "
                                 "numpy 2.0 onwards", name=None)
        elif attr == "core":
            import numpy.core as core
            return core
        elif attr == "strings":
            import numpy.strings as strings
            return strings
        elif attr == "distutils":
            if 'distutils' in __numpy_submodules__:
                import numpy.distutils as distutils
                return distutils
            else:
                raise AttributeError("`numpy.distutils` is not available from "
                                     "Python 3.12 onwards", name=None)

        if attr in __future_scalars__:
            # And future warnings for those that will change, but also give
            # the AttributeError
            warnings.warn(
                f"In the future `np.{attr}` will be defined as the "
                "corresponding NumPy scalar.", FutureWarning, stacklevel=2)

        if attr in __former_attrs__:
            raise AttributeError(__former_attrs__[attr], name=None)

        if attr in __expired_attributes__:
            raise AttributeError(
                f"`np.{attr}` was removed in the NumPy 2.0 release. "
                f"{__expired_attributes__[attr]}",
                name=None
            )

        if attr == "chararray":
            warnings.warn(
                "`np.chararray` is deprecated and will be removed from "
                "the main namespace in the future. Use an array with a string "
                "or bytes dtype instead.", DeprecationWarning, stacklevel=2)
            import numpy.char as char
            return char.chararray

        raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")

    def __dir__():
        public_symbols = (
            globals().keys() | __numpy_submodules__
        )
        public_symbols -= {
            "matrixlib", "matlib", "tests", "conftest", "version",
            "distutils", "array_api"
        }
        return list(public_symbols)

    # Pytest testing
    from numpy._pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester

    def _sanity_check():
        """
        Quick sanity checks for common bugs caused by environment.
        There are some cases e.g. with wrong BLAS ABI that cause wrong
        results under specific runtime conditions that are not necessarily
        achieved during test suite runs, and it is useful to catch those early.

        See https://github.com/numpy/numpy/issues/8577 and other
        similar bug reports.

        """
        try:
            x = ones(2, dtype=float32)
            if not abs(x.dot(x) - float32(2.0)) < 1e-5:
                raise AssertionError
        except AssertionError:
            msg = ("The current Numpy installation ({!r}) fails to "
                   "pass simple sanity checks. This can be caused for example "
                   "by incorrect BLAS library being linked in, or by mixing "
                   "package managers (pip, conda, apt, ...). Search closed "
                   "numpy issues for similar problems.")
            raise RuntimeError(msg.format(__file__)) from None

    _sanity_check()
    del _sanity_check

    def _mac_os_check():
        """
        Quick Sanity check for Mac OS look for accelerate build bugs.
        Testing numpy polyfit calls init_dgelsd(LAPACK)
        """
        try:
            c = array([3., 2., 1.])
            x = linspace(0, 2, 5)
            y = polyval(c, x)
            _ = polyfit(x, y, 2, cov=True)
        except ValueError:
            pass

    if sys.platform == "darwin":
        from . import exceptions
        with warnings.catch_warnings(record=True) as w:
            _mac_os_check()
            # Throw runtime error, if the test failed
            # Check for warning and report the error_message
            if len(w) > 0:
                for _wn in w:
                    if _wn.category is exceptions.RankWarning:
                        # Ignore other warnings, they may not be relevant (see gh-25433)
                        error_message = (
                            f"{_wn.category.__name__}: {_wn.message}"
                        )
                        msg = (
                            "Polyfit sanity test emitted a warning, most likely due "
                            "to using a buggy Accelerate backend."
                            "\nIf you compiled yourself, more information is available at:"  # noqa: E501
                            "\nhttps://numpy.org/devdocs/building/index.html"
                            "\nOtherwise report this to the vendor "
                            f"that provided NumPy.\n\n{error_message}\n")
                        raise RuntimeError(msg)
                del _wn
            del w
    del _mac_os_check

    def hugepage_setup():
        """
        We usually use madvise hugepages support, but on some old kernels it
        is slow and thus better avoided. Specifically kernel version 4.6
        had a bug fix which probably fixed this:
        https://github.com/torvalds/linux/commit/7cf91a98e607c2f935dbcc177d70011e95b8faff
        """
        use_hugepage = os.environ.get("NUMPY_MADVISE_HUGEPAGE", None)
        if sys.platform == "linux" and use_hugepage is None:
            # If there is an issue with parsing the kernel version,
            # set use_hugepage to 0. Usage of LooseVersion will handle
            # the kernel version parsing better, but avoided since it
            # will increase the import time.
            # See: #16679 for related discussion.
            try:
                use_hugepage = 1
                kernel_version = os.uname().release.split(".")[:2]
                kernel_version = tuple(int(v) for v in kernel_version)
                if kernel_version < (4, 6):
                    use_hugepage = 0
            except ValueError:
                use_hugepage = 0
        elif use_hugepage is None:
            # This is not Linux, so it should not matter, just enable anyway
            use_hugepage = 1
        else:
            use_hugepage = int(use_hugepage)
        return use_hugepage

    # Note that this will currently only make a difference on Linux
    _core.multiarray._set_madvise_hugepage(hugepage_setup())
    del hugepage_setup

    # Give a warning if NumPy is reloaded or imported on a sub-interpreter
    # We do this from python, since the C-module may not be reloaded and
    # it is tidier organized.
    _core.multiarray._multiarray_umath._reload_guard()

    # TODO: Remove the environment variable entirely now that it is "weak"
    if (os.environ.get("NPY_PROMOTION_STATE", "weak") != "weak"):
        warnings.warn(
            "NPY_PROMOTION_STATE was a temporary feature for NumPy 2.0 "
            "transition and is ignored after NumPy 2.2.",
            UserWarning, stacklevel=2)

    # Tell PyInstaller where to find hook-numpy.py
    def _pyinstaller_hooks_dir():
        from pathlib import Path
        return [str(Path(__file__).with_name("_pyinstaller").resolve())]


# Remove symbols imported for internal use
del os, sys, warnings

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\config_init.py ===
"""
This module is imported from the pandas package __init__.py file
in order to ensure that the core.config options registered here will
be available as soon as the user loads the package. if register_option
is invoked inside specific modules, they will not be registered until that
module is imported, which may or may not be a problem.

If you need to make sure options are available even before a certain
module is imported, register them here rather than in the module.

"""
from __future__ import annotations

import os
from typing import (
    Any,
    Callable,
)

import pandas._config.config as cf
from pandas._config.config import (
    is_bool,
    is_callable,
    is_instance_factory,
    is_int,
    is_nonnegative_int,
    is_one_of_factory,
    is_str,
    is_text,
)

# compute

use_bottleneck_doc = """
: bool
    Use the bottleneck library to accelerate if it is installed,
    the default is True
    Valid values: False,True
"""


def use_bottleneck_cb(key) -> None:
    from pandas.core import nanops

    nanops.set_use_bottleneck(cf.get_option(key))


use_numexpr_doc = """
: bool
    Use the numexpr library to accelerate computation if it is installed,
    the default is True
    Valid values: False,True
"""


def use_numexpr_cb(key) -> None:
    from pandas.core.computation import expressions

    expressions.set_use_numexpr(cf.get_option(key))


use_numba_doc = """
: bool
    Use the numba engine option for select operations if it is installed,
    the default is False
    Valid values: False,True
"""


def use_numba_cb(key) -> None:
    from pandas.core.util import numba_

    numba_.set_use_numba(cf.get_option(key))


with cf.config_prefix("compute"):
    cf.register_option(
        "use_bottleneck",
        True,
        use_bottleneck_doc,
        validator=is_bool,
        cb=use_bottleneck_cb,
    )
    cf.register_option(
        "use_numexpr", True, use_numexpr_doc, validator=is_bool, cb=use_numexpr_cb
    )
    cf.register_option(
        "use_numba", False, use_numba_doc, validator=is_bool, cb=use_numba_cb
    )
#
# options from the "display" namespace

pc_precision_doc = """
: int
    Floating point output precision in terms of number of places after the
    decimal, for regular formatting as well as scientific notation. Similar
    to ``precision`` in :meth:`numpy.set_printoptions`.
"""

pc_colspace_doc = """
: int
    Default space for DataFrame columns.
"""

pc_max_rows_doc = """
: int
    If max_rows is exceeded, switch to truncate view. Depending on
    `large_repr`, objects are either centrally truncated or printed as
    a summary view. 'None' value means unlimited.

    In case python/IPython is running in a terminal and `large_repr`
    equals 'truncate' this can be set to 0 and pandas will auto-detect
    the height of the terminal and print a truncated object which fits
    the screen height. The IPython notebook, IPython qtconsole, or
    IDLE do not run in a terminal and hence it is not possible to do
    correct auto-detection.
"""

pc_min_rows_doc = """
: int
    The numbers of rows to show in a truncated view (when `max_rows` is
    exceeded). Ignored when `max_rows` is set to None or 0. When set to
    None, follows the value of `max_rows`.
"""

pc_max_cols_doc = """
: int
    If max_cols is exceeded, switch to truncate view. Depending on
    `large_repr`, objects are either centrally truncated or printed as
    a summary view. 'None' value means unlimited.

    In case python/IPython is running in a terminal and `large_repr`
    equals 'truncate' this can be set to 0 or None and pandas will auto-detect
    the width of the terminal and print a truncated object which fits
    the screen width. The IPython notebook, IPython qtconsole, or IDLE
    do not run in a terminal and hence it is not possible to do
    correct auto-detection and defaults to 20.
"""

pc_max_categories_doc = """
: int
    This sets the maximum number of categories pandas should output when
    printing out a `Categorical` or a Series of dtype "category".
"""

pc_max_info_cols_doc = """
: int
    max_info_columns is used in DataFrame.info method to decide if
    per column information will be printed.
"""

pc_nb_repr_h_doc = """
: boolean
    When True, IPython notebook will use html representation for
    pandas objects (if it is available).
"""

pc_pprint_nest_depth = """
: int
    Controls the number of nested levels to process when pretty-printing
"""

pc_multi_sparse_doc = """
: boolean
    "sparsify" MultiIndex display (don't display repeated
    elements in outer levels within groups)
"""

float_format_doc = """
: callable
    The callable should accept a floating point number and return
    a string with the desired format of the number. This is used
    in some places like SeriesFormatter.
    See formats.format.EngFormatter for an example.
"""

max_colwidth_doc = """
: int or None
    The maximum width in characters of a column in the repr of
    a pandas data structure. When the column overflows, a "..."
    placeholder is embedded in the output. A 'None' value means unlimited.
"""

colheader_justify_doc = """
: 'left'/'right'
    Controls the justification of column headers. used by DataFrameFormatter.
"""

pc_expand_repr_doc = """
: boolean
    Whether to print out the full DataFrame repr for wide DataFrames across
    multiple lines, `max_columns` is still respected, but the output will
    wrap-around across multiple "pages" if its width exceeds `display.width`.
"""

pc_show_dimensions_doc = """
: boolean or 'truncate'
    Whether to print out dimensions at the end of DataFrame repr.
    If 'truncate' is specified, only print out the dimensions if the
    frame is truncated (e.g. not display all rows and/or columns)
"""

pc_east_asian_width_doc = """
: boolean
    Whether to use the Unicode East Asian Width to calculate the display text
    width.
    Enabling this may affect to the performance (default: False)
"""

pc_ambiguous_as_wide_doc = """
: boolean
    Whether to handle Unicode characters belong to Ambiguous as Wide (width=2)
    (default: False)
"""

pc_table_schema_doc = """
: boolean
    Whether to publish a Table Schema representation for frontends
    that support it.
    (default: False)
"""

pc_html_border_doc = """
: int
    A ``border=value`` attribute is inserted in the ``<table>`` tag
    for the DataFrame HTML repr.
"""

pc_html_use_mathjax_doc = """\
: boolean
    When True, Jupyter notebook will process table contents using MathJax,
    rendering mathematical expressions enclosed by the dollar symbol.
    (default: True)
"""

pc_max_dir_items = """\
: int
    The number of items that will be added to `dir(...)`. 'None' value means
    unlimited. Because dir is cached, changing this option will not immediately
    affect already existing dataframes until a column is deleted or added.

    This is for instance used to suggest columns from a dataframe to tab
    completion.
"""

pc_width_doc = """
: int
    Width of the display in characters. In case python/IPython is running in
    a terminal this can be set to None and pandas will correctly auto-detect
    the width.
    Note that the IPython notebook, IPython qtconsole, or IDLE do not run in a
    terminal and hence it is not possible to correctly detect the width.
"""

pc_chop_threshold_doc = """
: float or None
    if set to a float value, all float values smaller than the given threshold
    will be displayed as exactly 0 by repr and friends.
"""

pc_max_seq_items = """
: int or None
    When pretty-printing a long sequence, no more then `max_seq_items`
    will be printed. If items are omitted, they will be denoted by the
    addition of "..." to the resulting string.

    If set to None, the number of items to be printed is unlimited.
"""

pc_max_info_rows_doc = """
: int
    df.info() will usually show null-counts for each column.
    For large frames this can be quite slow. max_info_rows and max_info_cols
    limit this null check only to frames with smaller dimensions than
    specified.
"""

pc_large_repr_doc = """
: 'truncate'/'info'
    For DataFrames exceeding max_rows/max_cols, the repr (and HTML repr) can
    show a truncated table, or switch to the view from
    df.info() (the behaviour in earlier versions of pandas).
"""

pc_memory_usage_doc = """
: bool, string or None
    This specifies if the memory usage of a DataFrame should be displayed when
    df.info() is called. Valid values True,False,'deep'
"""


def table_schema_cb(key) -> None:
    from pandas.io.formats.printing import enable_data_resource_formatter

    enable_data_resource_formatter(cf.get_option(key))


def is_terminal() -> bool:
    """
    Detect if Python is running in a terminal.

    Returns True if Python is running in a terminal or False if not.
    """
    try:
        # error: Name 'get_ipython' is not defined
        ip = get_ipython()  # type: ignore[name-defined]
    except NameError:  # assume standard Python interpreter in a terminal
        return True
    else:
        if hasattr(ip, "kernel"):  # IPython as a Jupyter kernel
            return False
        else:  # IPython in a terminal
            return True


with cf.config_prefix("display"):
    cf.register_option("precision", 6, pc_precision_doc, validator=is_nonnegative_int)
    cf.register_option(
        "float_format",
        None,
        float_format_doc,
        validator=is_one_of_factory([None, is_callable]),
    )
    cf.register_option(
        "max_info_rows",
        1690785,
        pc_max_info_rows_doc,
        validator=is_int,
    )
    cf.register_option("max_rows", 60, pc_max_rows_doc, validator=is_nonnegative_int)
    cf.register_option(
        "min_rows",
        10,
        pc_min_rows_doc,
        validator=is_instance_factory([type(None), int]),
    )
    cf.register_option("max_categories", 8, pc_max_categories_doc, validator=is_int)

    cf.register_option(
        "max_colwidth",
        50,
        max_colwidth_doc,
        validator=is_nonnegative_int,
    )
    if is_terminal():
        max_cols = 0  # automatically determine optimal number of columns
    else:
        max_cols = 20  # cannot determine optimal number of columns
    cf.register_option(
        "max_columns", max_cols, pc_max_cols_doc, validator=is_nonnegative_int
    )
    cf.register_option(
        "large_repr",
        "truncate",
        pc_large_repr_doc,
        validator=is_one_of_factory(["truncate", "info"]),
    )
    cf.register_option("max_info_columns", 100, pc_max_info_cols_doc, validator=is_int)
    cf.register_option(
        "colheader_justify", "right", colheader_justify_doc, validator=is_text
    )
    cf.register_option("notebook_repr_html", True, pc_nb_repr_h_doc, validator=is_bool)
    cf.register_option("pprint_nest_depth", 3, pc_pprint_nest_depth, validator=is_int)
    cf.register_option("multi_sparse", True, pc_multi_sparse_doc, validator=is_bool)
    cf.register_option("expand_frame_repr", True, pc_expand_repr_doc)
    cf.register_option(
        "show_dimensions",
        "truncate",
        pc_show_dimensions_doc,
        validator=is_one_of_factory([True, False, "truncate"]),
    )
    cf.register_option("chop_threshold", None, pc_chop_threshold_doc)
    cf.register_option("max_seq_items", 100, pc_max_seq_items)
    cf.register_option(
        "width", 80, pc_width_doc, validator=is_instance_factory([type(None), int])
    )
    cf.register_option(
        "memory_usage",
        True,
        pc_memory_usage_doc,
        validator=is_one_of_factory([None, True, False, "deep"]),
    )
    cf.register_option(
        "unicode.east_asian_width", False, pc_east_asian_width_doc, validator=is_bool
    )
    cf.register_option(
        "unicode.ambiguous_as_wide", False, pc_east_asian_width_doc, validator=is_bool
    )
    cf.register_option(
        "html.table_schema",
        False,
        pc_table_schema_doc,
        validator=is_bool,
        cb=table_schema_cb,
    )
    cf.register_option("html.border", 1, pc_html_border_doc, validator=is_int)
    cf.register_option(
        "html.use_mathjax", True, pc_html_use_mathjax_doc, validator=is_bool
    )
    cf.register_option(
        "max_dir_items", 100, pc_max_dir_items, validator=is_nonnegative_int
    )

tc_sim_interactive_doc = """
: boolean
    Whether to simulate interactive mode for purposes of testing
"""

with cf.config_prefix("mode"):
    cf.register_option("sim_interactive", False, tc_sim_interactive_doc)

use_inf_as_na_doc = """
: boolean
    True means treat None, NaN, INF, -INF as NA (old way),
    False means None and NaN are null, but INF, -INF are not NA
    (new way).

    This option is deprecated in pandas 2.1.0 and will be removed in 3.0.
"""

# We don't want to start importing everything at the global context level
# or we'll hit circular deps.


def use_inf_as_na_cb(key) -> None:
    # TODO(3.0): enforcing this deprecation will close GH#52501
    from pandas.core.dtypes.missing import _use_inf_as_na

    _use_inf_as_na(key)


with cf.config_prefix("mode"):
    cf.register_option("use_inf_as_na", False, use_inf_as_na_doc, cb=use_inf_as_na_cb)

cf.deprecate_option(
    # GH#51684
    "mode.use_inf_as_na",
    "use_inf_as_na option is deprecated and will be removed in a future "
    "version. Convert inf values to NaN before operating instead.",
)

data_manager_doc = """
: string
    Internal data manager type; can be "block" or "array". Defaults to "block",
    unless overridden by the 'PANDAS_DATA_MANAGER' environment variable (needs
    to be set before pandas is imported).
"""


with cf.config_prefix("mode"):
    cf.register_option(
        "data_manager",
        # Get the default from an environment variable, if set, otherwise defaults
        # to "block". This environment variable can be set for testing.
        os.environ.get("PANDAS_DATA_MANAGER", "block"),
        data_manager_doc,
        validator=is_one_of_factory(["block", "array"]),
    )

cf.deprecate_option(
    # GH#55043
    "mode.data_manager",
    "data_manager option is deprecated and will be removed in a future "
    "version. Only the BlockManager will be available.",
)


# TODO better name?
copy_on_write_doc = """
: bool
    Use new copy-view behaviour using Copy-on-Write. Defaults to False,
    unless overridden by the 'PANDAS_COPY_ON_WRITE' environment variable
    (if set to "1" for True, needs to be set before pandas is imported).
"""


with cf.config_prefix("mode"):
    cf.register_option(
        "copy_on_write",
        # Get the default from an environment variable, if set, otherwise defaults
        # to False. This environment variable can be set for testing.
        "warn"
        if os.environ.get("PANDAS_COPY_ON_WRITE", "0") == "warn"
        else os.environ.get("PANDAS_COPY_ON_WRITE", "0") == "1",
        copy_on_write_doc,
        validator=is_one_of_factory([True, False, "warn"]),
    )


# user warnings
chained_assignment = """
: string
    Raise an exception, warn, or no action if trying to use chained assignment,
    The default is warn
"""

with cf.config_prefix("mode"):
    cf.register_option(
        "chained_assignment",
        "warn",
        chained_assignment,
        validator=is_one_of_factory([None, "warn", "raise"]),
    )


string_storage_doc = """
: string
    The default storage for StringDtype.
"""


def is_valid_string_storage(value: Any) -> None:
    legal_values = ["auto", "python", "pyarrow"]
    if value not in legal_values:
        msg = "Value must be one of python|pyarrow"
        if value == "pyarrow_numpy":
            # TODO: we can remove extra message after 3.0
            msg += (
                ". 'pyarrow_numpy' was specified, but this option should be "
                "enabled using pandas.options.future.infer_string instead"
            )
        raise ValueError(msg)


with cf.config_prefix("mode"):
    cf.register_option(
        "string_storage",
        "auto",
        string_storage_doc,
        # validator=is_one_of_factory(["python", "pyarrow"]),
        validator=is_valid_string_storage,
    )


# Set up the io.excel specific reader configuration.
reader_engine_doc = """
: string
    The default Excel reader engine for '{ext}' files. Available options:
    auto, {others}.
"""

_xls_options = ["xlrd", "calamine"]
_xlsm_options = ["xlrd", "openpyxl", "calamine"]
_xlsx_options = ["xlrd", "openpyxl", "calamine"]
_ods_options = ["odf", "calamine"]
_xlsb_options = ["pyxlsb", "calamine"]


with cf.config_prefix("io.excel.xls"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xls", others=", ".join(_xls_options)),
        validator=is_one_of_factory(_xls_options + ["auto"]),
    )

with cf.config_prefix("io.excel.xlsm"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xlsm", others=", ".join(_xlsm_options)),
        validator=is_one_of_factory(_xlsm_options + ["auto"]),
    )


with cf.config_prefix("io.excel.xlsx"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xlsx", others=", ".join(_xlsx_options)),
        validator=is_one_of_factory(_xlsx_options + ["auto"]),
    )


with cf.config_prefix("io.excel.ods"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="ods", others=", ".join(_ods_options)),
        validator=is_one_of_factory(_ods_options + ["auto"]),
    )

with cf.config_prefix("io.excel.xlsb"):
    cf.register_option(
        "reader",
        "auto",
        reader_engine_doc.format(ext="xlsb", others=", ".join(_xlsb_options)),
        validator=is_one_of_factory(_xlsb_options + ["auto"]),
    )

# Set up the io.excel specific writer configuration.
writer_engine_doc = """
: string
    The default Excel writer engine for '{ext}' files. Available options:
    auto, {others}.
"""

_xlsm_options = ["openpyxl"]
_xlsx_options = ["openpyxl", "xlsxwriter"]
_ods_options = ["odf"]


with cf.config_prefix("io.excel.xlsm"):
    cf.register_option(
        "writer",
        "auto",
        writer_engine_doc.format(ext="xlsm", others=", ".join(_xlsm_options)),
        validator=str,
    )


with cf.config_prefix("io.excel.xlsx"):
    cf.register_option(
        "writer",
        "auto",
        writer_engine_doc.format(ext="xlsx", others=", ".join(_xlsx_options)),
        validator=str,
    )


with cf.config_prefix("io.excel.ods"):
    cf.register_option(
        "writer",
        "auto",
        writer_engine_doc.format(ext="ods", others=", ".join(_ods_options)),
        validator=str,
    )


# Set up the io.parquet specific configuration.
parquet_engine_doc = """
: string
    The default parquet reader/writer engine. Available options:
    'auto', 'pyarrow', 'fastparquet', the default is 'auto'
"""

with cf.config_prefix("io.parquet"):
    cf.register_option(
        "engine",
        "auto",
        parquet_engine_doc,
        validator=is_one_of_factory(["auto", "pyarrow", "fastparquet"]),
    )


# Set up the io.sql specific configuration.
sql_engine_doc = """
: string
    The default sql reader/writer engine. Available options:
    'auto', 'sqlalchemy', the default is 'auto'
"""

with cf.config_prefix("io.sql"):
    cf.register_option(
        "engine",
        "auto",
        sql_engine_doc,
        validator=is_one_of_factory(["auto", "sqlalchemy"]),
    )

# --------
# Plotting
# ---------

plotting_backend_doc = """
: str
    The plotting backend to use. The default value is "matplotlib", the
    backend provided with pandas. Other backends can be specified by
    providing the name of the module that implements the backend.
"""


def register_plotting_backend_cb(key) -> None:
    if key == "matplotlib":
        # We defer matplotlib validation, since it's the default
        return
    from pandas.plotting._core import _get_plot_backend

    _get_plot_backend(key)


with cf.config_prefix("plotting"):
    cf.register_option(
        "backend",
        defval="matplotlib",
        doc=plotting_backend_doc,
        validator=register_plotting_backend_cb,
    )


register_converter_doc = """
: bool or 'auto'.
    Whether to register converters with matplotlib's units registry for
    dates, times, datetimes, and Periods. Toggling to False will remove
    the converters, restoring any converters that pandas overwrote.
"""


def register_converter_cb(key) -> None:
    from pandas.plotting import (
        deregister_matplotlib_converters,
        register_matplotlib_converters,
    )

    if cf.get_option(key):
        register_matplotlib_converters()
    else:
        deregister_matplotlib_converters()


with cf.config_prefix("plotting.matplotlib"):
    cf.register_option(
        "register_converters",
        "auto",
        register_converter_doc,
        validator=is_one_of_factory(["auto", True, False]),
        cb=register_converter_cb,
    )

# ------
# Styler
# ------

styler_sparse_index_doc = """
: bool
    Whether to sparsify the display of a hierarchical index. Setting to False will
    display each explicit level element in a hierarchical key for each row.
"""

styler_sparse_columns_doc = """
: bool
    Whether to sparsify the display of hierarchical columns. Setting to False will
    display each explicit level element in a hierarchical key for each column.
"""

styler_render_repr = """
: str
    Determine which output to use in Jupyter Notebook in {"html", "latex"}.
"""

styler_max_elements = """
: int
    The maximum number of data-cell (<td>) elements that will be rendered before
    trimming will occur over columns, rows or both if needed.
"""

styler_max_rows = """
: int, optional
    The maximum number of rows that will be rendered. May still be reduced to
    satisfy ``max_elements``, which takes precedence.
"""

styler_max_columns = """
: int, optional
    The maximum number of columns that will be rendered. May still be reduced to
    satisfy ``max_elements``, which takes precedence.
"""

styler_precision = """
: int
    The precision for floats and complex numbers.
"""

styler_decimal = """
: str
    The character representation for the decimal separator for floats and complex.
"""

styler_thousands = """
: str, optional
    The character representation for thousands separator for floats, int and complex.
"""

styler_na_rep = """
: str, optional
    The string representation for values identified as missing.
"""

styler_escape = """
: str, optional
    Whether to escape certain characters according to the given context; html or latex.
"""

styler_formatter = """
: str, callable, dict, optional
    A formatter object to be used as default within ``Styler.format``.
"""

styler_multirow_align = """
: {"c", "t", "b"}
    The specifier for vertical alignment of sparsified LaTeX multirows.
"""

styler_multicol_align = r"""
: {"r", "c", "l", "naive-l", "naive-r"}
    The specifier for horizontal alignment of sparsified LaTeX multicolumns. Pipe
    decorators can also be added to non-naive values to draw vertical
    rules, e.g. "\|r" will draw a rule on the left side of right aligned merged cells.
"""

styler_hrules = """
: bool
    Whether to add horizontal rules on top and bottom and below the headers.
"""

styler_environment = """
: str
    The environment to replace ``\\begin{table}``. If "longtable" is used results
    in a specific longtable environment format.
"""

styler_encoding = """
: str
    The encoding used for output HTML and LaTeX files.
"""

styler_mathjax = """
: bool
    If False will render special CSS classes to table attributes that indicate Mathjax
    will not be used in Jupyter Notebook.
"""

with cf.config_prefix("styler"):
    cf.register_option("sparse.index", True, styler_sparse_index_doc, validator=is_bool)

    cf.register_option(
        "sparse.columns", True, styler_sparse_columns_doc, validator=is_bool
    )

    cf.register_option(
        "render.repr",
        "html",
        styler_render_repr,
        validator=is_one_of_factory(["html", "latex"]),
    )

    cf.register_option(
        "render.max_elements",
        2**18,
        styler_max_elements,
        validator=is_nonnegative_int,
    )

    cf.register_option(
        "render.max_rows",
        None,
        styler_max_rows,
        validator=is_nonnegative_int,
    )

    cf.register_option(
        "render.max_columns",
        None,
        styler_max_columns,
        validator=is_nonnegative_int,
    )

    cf.register_option("render.encoding", "utf-8", styler_encoding, validator=is_str)

    cf.register_option("format.decimal", ".", styler_decimal, validator=is_str)

    cf.register_option(
        "format.precision", 6, styler_precision, validator=is_nonnegative_int
    )

    cf.register_option(
        "format.thousands",
        None,
        styler_thousands,
        validator=is_instance_factory([type(None), str]),
    )

    cf.register_option(
        "format.na_rep",
        None,
        styler_na_rep,
        validator=is_instance_factory([type(None), str]),
    )

    cf.register_option(
        "format.escape",
        None,
        styler_escape,
        validator=is_one_of_factory([None, "html", "latex", "latex-math"]),
    )

    cf.register_option(
        "format.formatter",
        None,
        styler_formatter,
        validator=is_instance_factory([type(None), dict, Callable, str]),
    )

    cf.register_option("html.mathjax", True, styler_mathjax, validator=is_bool)

    cf.register_option(
        "latex.multirow_align",
        "c",
        styler_multirow_align,
        validator=is_one_of_factory(["c", "t", "b", "naive"]),
    )

    val_mca = ["r", "|r|", "|r", "r|", "c", "|c|", "|c", "c|", "l", "|l|", "|l", "l|"]
    val_mca += ["naive-l", "naive-r"]
    cf.register_option(
        "latex.multicol_align",
        "r",
        styler_multicol_align,
        validator=is_one_of_factory(val_mca),
    )

    cf.register_option("latex.hrules", False, styler_hrules, validator=is_bool)

    cf.register_option(
        "latex.environment",
        None,
        styler_environment,
        validator=is_instance_factory([type(None), str]),
    )


with cf.config_prefix("future"):
    cf.register_option(
        "infer_string",
        True if os.environ.get("PANDAS_FUTURE_INFER_STRING", "0") == "1" else False,
        "Whether to infer sequence of str objects as pyarrow string "
        "dtype, which will be the default in pandas 3.0 "
        "(at which point this option will be deprecated).",
        validator=is_one_of_factory([True, False]),
    )

    cf.register_option(
        "no_silent_downcasting",
        False,
        "Whether to opt-in to the future behavior which will *not* silently "
        "downcast results from Series and DataFrame `where`, `mask`, and `clip` "
        "methods. "
        "Silent downcasting will be removed in pandas 3.0 "
        "(at which point this option will be deprecated).",
        validator=is_one_of_factory([True, False]),
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\filters\__init__.py ===
"""
    pygments.filters
    ~~~~~~~~~~~~~~~~

    Module containing filter lookup functions and default
    filters.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re

from pip._vendor.pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
    string_to_tokentype
from pip._vendor.pygments.filter import Filter
from pip._vendor.pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
    get_choice_opt, ClassNotFound, OptionError
from pip._vendor.pygments.plugin import find_plugin_filters


def find_filter_class(filtername):
    """Lookup a filter by name. Return None if not found."""
    if filtername in FILTERS:
        return FILTERS[filtername]
    for name, cls in find_plugin_filters():
        if name == filtername:
            return cls
    return None


def get_filter_by_name(filtername, **options):
    """Return an instantiated filter.

    Options are passed to the filter initializer if wanted.
    Raise a ClassNotFound if not found.
    """
    cls = find_filter_class(filtername)
    if cls:
        return cls(**options)
    else:
        raise ClassNotFound(f'filter {filtername!r} not found')


def get_all_filters():
    """Return a generator of all filter names."""
    yield from FILTERS
    for name, _ in find_plugin_filters():
        yield name


def _replace_special(ttype, value, regex, specialttype,
                     replacefunc=lambda x: x):
    last = 0
    for match in regex.finditer(value):
        start, end = match.start(), match.end()
        if start != last:
            yield ttype, value[last:start]
        yield specialttype, replacefunc(value[start:end])
        last = end
    if last != len(value):
        yield ttype, value[last:]


class CodeTagFilter(Filter):
    """Highlight special code tags in comments and docstrings.

    Options accepted:

    `codetags` : list of strings
       A list of strings that are flagged as code tags.  The default is to
       highlight ``XXX``, ``TODO``, ``FIXME``, ``BUG`` and ``NOTE``.

    .. versionchanged:: 2.13
       Now recognizes ``FIXME`` by default.
    """

    def __init__(self, **options):
        Filter.__init__(self, **options)
        tags = get_list_opt(options, 'codetags',
                            ['XXX', 'TODO', 'FIXME', 'BUG', 'NOTE'])
        self.tag_re = re.compile(r'\b({})\b'.format('|'.join([
            re.escape(tag) for tag in tags if tag
        ])))

    def filter(self, lexer, stream):
        regex = self.tag_re
        for ttype, value in stream:
            if ttype in String.Doc or \
               ttype in Comment and \
               ttype not in Comment.Preproc:
                yield from _replace_special(ttype, value, regex, Comment.Special)
            else:
                yield ttype, value


class SymbolFilter(Filter):
    """Convert mathematical symbols such as \\<longrightarrow> in Isabelle
    or \\longrightarrow in LaTeX into Unicode characters.

    This is mostly useful for HTML or console output when you want to
    approximate the source rendering you'd see in an IDE.

    Options accepted:

    `lang` : string
       The symbol language. Must be one of ``'isabelle'`` or
       ``'latex'``.  The default is ``'isabelle'``.
    """

    latex_symbols = {
        '\\alpha'                : '\U000003b1',
        '\\beta'                 : '\U000003b2',
        '\\gamma'                : '\U000003b3',
        '\\delta'                : '\U000003b4',
        '\\varepsilon'           : '\U000003b5',
        '\\zeta'                 : '\U000003b6',
        '\\eta'                  : '\U000003b7',
        '\\vartheta'             : '\U000003b8',
        '\\iota'                 : '\U000003b9',
        '\\kappa'                : '\U000003ba',
        '\\lambda'               : '\U000003bb',
        '\\mu'                   : '\U000003bc',
        '\\nu'                   : '\U000003bd',
        '\\xi'                   : '\U000003be',
        '\\pi'                   : '\U000003c0',
        '\\varrho'               : '\U000003c1',
        '\\sigma'                : '\U000003c3',
        '\\tau'                  : '\U000003c4',
        '\\upsilon'              : '\U000003c5',
        '\\varphi'               : '\U000003c6',
        '\\chi'                  : '\U000003c7',
        '\\psi'                  : '\U000003c8',
        '\\omega'                : '\U000003c9',
        '\\Gamma'                : '\U00000393',
        '\\Delta'                : '\U00000394',
        '\\Theta'                : '\U00000398',
        '\\Lambda'               : '\U0000039b',
        '\\Xi'                   : '\U0000039e',
        '\\Pi'                   : '\U000003a0',
        '\\Sigma'                : '\U000003a3',
        '\\Upsilon'              : '\U000003a5',
        '\\Phi'                  : '\U000003a6',
        '\\Psi'                  : '\U000003a8',
        '\\Omega'                : '\U000003a9',
        '\\leftarrow'            : '\U00002190',
        '\\longleftarrow'        : '\U000027f5',
        '\\rightarrow'           : '\U00002192',
        '\\longrightarrow'       : '\U000027f6',
        '\\Leftarrow'            : '\U000021d0',
        '\\Longleftarrow'        : '\U000027f8',
        '\\Rightarrow'           : '\U000021d2',
        '\\Longrightarrow'       : '\U000027f9',
        '\\leftrightarrow'       : '\U00002194',
        '\\longleftrightarrow'   : '\U000027f7',
        '\\Leftrightarrow'       : '\U000021d4',
        '\\Longleftrightarrow'   : '\U000027fa',
        '\\mapsto'               : '\U000021a6',
        '\\longmapsto'           : '\U000027fc',
        '\\relbar'               : '\U00002500',
        '\\Relbar'               : '\U00002550',
        '\\hookleftarrow'        : '\U000021a9',
        '\\hookrightarrow'       : '\U000021aa',
        '\\leftharpoondown'      : '\U000021bd',
        '\\rightharpoondown'     : '\U000021c1',
        '\\leftharpoonup'        : '\U000021bc',
        '\\rightharpoonup'       : '\U000021c0',
        '\\rightleftharpoons'    : '\U000021cc',
        '\\leadsto'              : '\U0000219d',
        '\\downharpoonleft'      : '\U000021c3',
        '\\downharpoonright'     : '\U000021c2',
        '\\upharpoonleft'        : '\U000021bf',
        '\\upharpoonright'       : '\U000021be',
        '\\restriction'          : '\U000021be',
        '\\uparrow'              : '\U00002191',
        '\\Uparrow'              : '\U000021d1',
        '\\downarrow'            : '\U00002193',
        '\\Downarrow'            : '\U000021d3',
        '\\updownarrow'          : '\U00002195',
        '\\Updownarrow'          : '\U000021d5',
        '\\langle'               : '\U000027e8',
        '\\rangle'               : '\U000027e9',
        '\\lceil'                : '\U00002308',
        '\\rceil'                : '\U00002309',
        '\\lfloor'               : '\U0000230a',
        '\\rfloor'               : '\U0000230b',
        '\\flqq'                 : '\U000000ab',
        '\\frqq'                 : '\U000000bb',
        '\\bot'                  : '\U000022a5',
        '\\top'                  : '\U000022a4',
        '\\wedge'                : '\U00002227',
        '\\bigwedge'             : '\U000022c0',
        '\\vee'                  : '\U00002228',
        '\\bigvee'               : '\U000022c1',
        '\\forall'               : '\U00002200',
        '\\exists'               : '\U00002203',
        '\\nexists'              : '\U00002204',
        '\\neg'                  : '\U000000ac',
        '\\Box'                  : '\U000025a1',
        '\\Diamond'              : '\U000025c7',
        '\\vdash'                : '\U000022a2',
        '\\models'               : '\U000022a8',
        '\\dashv'                : '\U000022a3',
        '\\surd'                 : '\U0000221a',
        '\\le'                   : '\U00002264',
        '\\ge'                   : '\U00002265',
        '\\ll'                   : '\U0000226a',
        '\\gg'                   : '\U0000226b',
        '\\lesssim'              : '\U00002272',
        '\\gtrsim'               : '\U00002273',
        '\\lessapprox'           : '\U00002a85',
        '\\gtrapprox'            : '\U00002a86',
        '\\in'                   : '\U00002208',
        '\\notin'                : '\U00002209',
        '\\subset'               : '\U00002282',
        '\\supset'               : '\U00002283',
        '\\subseteq'             : '\U00002286',
        '\\supseteq'             : '\U00002287',
        '\\sqsubset'             : '\U0000228f',
        '\\sqsupset'             : '\U00002290',
        '\\sqsubseteq'           : '\U00002291',
        '\\sqsupseteq'           : '\U00002292',
        '\\cap'                  : '\U00002229',
        '\\bigcap'               : '\U000022c2',
        '\\cup'                  : '\U0000222a',
        '\\bigcup'               : '\U000022c3',
        '\\sqcup'                : '\U00002294',
        '\\bigsqcup'             : '\U00002a06',
        '\\sqcap'                : '\U00002293',
        '\\Bigsqcap'             : '\U00002a05',
        '\\setminus'             : '\U00002216',
        '\\propto'               : '\U0000221d',
        '\\uplus'                : '\U0000228e',
        '\\bigplus'              : '\U00002a04',
        '\\sim'                  : '\U0000223c',
        '\\doteq'                : '\U00002250',
        '\\simeq'                : '\U00002243',
        '\\approx'               : '\U00002248',
        '\\asymp'                : '\U0000224d',
        '\\cong'                 : '\U00002245',
        '\\equiv'                : '\U00002261',
        '\\Join'                 : '\U000022c8',
        '\\bowtie'               : '\U00002a1d',
        '\\prec'                 : '\U0000227a',
        '\\succ'                 : '\U0000227b',
        '\\preceq'               : '\U0000227c',
        '\\succeq'               : '\U0000227d',
        '\\parallel'             : '\U00002225',
        '\\mid'                  : '\U000000a6',
        '\\pm'                   : '\U000000b1',
        '\\mp'                   : '\U00002213',
        '\\times'                : '\U000000d7',
        '\\div'                  : '\U000000f7',
        '\\cdot'                 : '\U000022c5',
        '\\star'                 : '\U000022c6',
        '\\circ'                 : '\U00002218',
        '\\dagger'               : '\U00002020',
        '\\ddagger'              : '\U00002021',
        '\\lhd'                  : '\U000022b2',
        '\\rhd'                  : '\U000022b3',
        '\\unlhd'                : '\U000022b4',
        '\\unrhd'                : '\U000022b5',
        '\\triangleleft'         : '\U000025c3',
        '\\triangleright'        : '\U000025b9',
        '\\triangle'             : '\U000025b3',
        '\\triangleq'            : '\U0000225c',
        '\\oplus'                : '\U00002295',
        '\\bigoplus'             : '\U00002a01',
        '\\otimes'               : '\U00002297',
        '\\bigotimes'            : '\U00002a02',
        '\\odot'                 : '\U00002299',
        '\\bigodot'              : '\U00002a00',
        '\\ominus'               : '\U00002296',
        '\\oslash'               : '\U00002298',
        '\\dots'                 : '\U00002026',
        '\\cdots'                : '\U000022ef',
        '\\sum'                  : '\U00002211',
        '\\prod'                 : '\U0000220f',
        '\\coprod'               : '\U00002210',
        '\\infty'                : '\U0000221e',
        '\\int'                  : '\U0000222b',
        '\\oint'                 : '\U0000222e',
        '\\clubsuit'             : '\U00002663',
        '\\diamondsuit'          : '\U00002662',
        '\\heartsuit'            : '\U00002661',
        '\\spadesuit'            : '\U00002660',
        '\\aleph'                : '\U00002135',
        '\\emptyset'             : '\U00002205',
        '\\nabla'                : '\U00002207',
        '\\partial'              : '\U00002202',
        '\\flat'                 : '\U0000266d',
        '\\natural'              : '\U0000266e',
        '\\sharp'                : '\U0000266f',
        '\\angle'                : '\U00002220',
        '\\copyright'            : '\U000000a9',
        '\\textregistered'       : '\U000000ae',
        '\\textonequarter'       : '\U000000bc',
        '\\textonehalf'          : '\U000000bd',
        '\\textthreequarters'    : '\U000000be',
        '\\textordfeminine'      : '\U000000aa',
        '\\textordmasculine'     : '\U000000ba',
        '\\euro'                 : '\U000020ac',
        '\\pounds'               : '\U000000a3',
        '\\yen'                  : '\U000000a5',
        '\\textcent'             : '\U000000a2',
        '\\textcurrency'         : '\U000000a4',
        '\\textdegree'           : '\U000000b0',
    }

    isabelle_symbols = {
        '\\<zero>'                 : '\U0001d7ec',
        '\\<one>'                  : '\U0001d7ed',
        '\\<two>'                  : '\U0001d7ee',
        '\\<three>'                : '\U0001d7ef',
        '\\<four>'                 : '\U0001d7f0',
        '\\<five>'                 : '\U0001d7f1',
        '\\<six>'                  : '\U0001d7f2',
        '\\<seven>'                : '\U0001d7f3',
        '\\<eight>'                : '\U0001d7f4',
        '\\<nine>'                 : '\U0001d7f5',
        '\\<A>'                    : '\U0001d49c',
        '\\<B>'                    : '\U0000212c',
        '\\<C>'                    : '\U0001d49e',
        '\\<D>'                    : '\U0001d49f',
        '\\<E>'                    : '\U00002130',
        '\\<F>'                    : '\U00002131',
        '\\<G>'                    : '\U0001d4a2',
        '\\<H>'                    : '\U0000210b',
        '\\<I>'                    : '\U00002110',
        '\\<J>'                    : '\U0001d4a5',
        '\\<K>'                    : '\U0001d4a6',
        '\\<L>'                    : '\U00002112',
        '\\<M>'                    : '\U00002133',
        '\\<N>'                    : '\U0001d4a9',
        '\\<O>'                    : '\U0001d4aa',
        '\\<P>'                    : '\U0001d4ab',
        '\\<Q>'                    : '\U0001d4ac',
        '\\<R>'                    : '\U0000211b',
        '\\<S>'                    : '\U0001d4ae',
        '\\<T>'                    : '\U0001d4af',
        '\\<U>'                    : '\U0001d4b0',
        '\\<V>'                    : '\U0001d4b1',
        '\\<W>'                    : '\U0001d4b2',
        '\\<X>'                    : '\U0001d4b3',
        '\\<Y>'                    : '\U0001d4b4',
        '\\<Z>'                    : '\U0001d4b5',
        '\\<a>'                    : '\U0001d5ba',
        '\\<b>'                    : '\U0001d5bb',
        '\\<c>'                    : '\U0001d5bc',
        '\\<d>'                    : '\U0001d5bd',
        '\\<e>'                    : '\U0001d5be',
        '\\<f>'                    : '\U0001d5bf',
        '\\<g>'                    : '\U0001d5c0',
        '\\<h>'                    : '\U0001d5c1',
        '\\<i>'                    : '\U0001d5c2',
        '\\<j>'                    : '\U0001d5c3',
        '\\<k>'                    : '\U0001d5c4',
        '\\<l>'                    : '\U0001d5c5',
        '\\<m>'                    : '\U0001d5c6',
        '\\<n>'                    : '\U0001d5c7',
        '\\<o>'                    : '\U0001d5c8',
        '\\<p>'                    : '\U0001d5c9',
        '\\<q>'                    : '\U0001d5ca',
        '\\<r>'                    : '\U0001d5cb',
        '\\<s>'                    : '\U0001d5cc',
        '\\<t>'                    : '\U0001d5cd',
        '\\<u>'                    : '\U0001d5ce',
        '\\<v>'                    : '\U0001d5cf',
        '\\<w>'                    : '\U0001d5d0',
        '\\<x>'                    : '\U0001d5d1',
        '\\<y>'                    : '\U0001d5d2',
        '\\<z>'                    : '\U0001d5d3',
        '\\<AA>'                   : '\U0001d504',
        '\\<BB>'                   : '\U0001d505',
        '\\<CC>'                   : '\U0000212d',
        '\\<DD>'                   : '\U0001d507',
        '\\<EE>'                   : '\U0001d508',
        '\\<FF>'                   : '\U0001d509',
        '\\<GG>'                   : '\U0001d50a',
        '\\<HH>'                   : '\U0000210c',
        '\\<II>'                   : '\U00002111',
        '\\<JJ>'                   : '\U0001d50d',
        '\\<KK>'                   : '\U0001d50e',
        '\\<LL>'                   : '\U0001d50f',
        '\\<MM>'                   : '\U0001d510',
        '\\<NN>'                   : '\U0001d511',
        '\\<OO>'                   : '\U0001d512',
        '\\<PP>'                   : '\U0001d513',
        '\\<QQ>'                   : '\U0001d514',
        '\\<RR>'                   : '\U0000211c',
        '\\<SS>'                   : '\U0001d516',
        '\\<TT>'                   : '\U0001d517',
        '\\<UU>'                   : '\U0001d518',
        '\\<VV>'                   : '\U0001d519',
        '\\<WW>'                   : '\U0001d51a',
        '\\<XX>'                   : '\U0001d51b',
        '\\<YY>'                   : '\U0001d51c',
        '\\<ZZ>'                   : '\U00002128',
        '\\<aa>'                   : '\U0001d51e',
        '\\<bb>'                   : '\U0001d51f',
        '\\<cc>'                   : '\U0001d520',
        '\\<dd>'                   : '\U0001d521',
        '\\<ee>'                   : '\U0001d522',
        '\\<ff>'                   : '\U0001d523',
        '\\<gg>'                   : '\U0001d524',
        '\\<hh>'                   : '\U0001d525',
        '\\<ii>'                   : '\U0001d526',
        '\\<jj>'                   : '\U0001d527',
        '\\<kk>'                   : '\U0001d528',
        '\\<ll>'                   : '\U0001d529',
        '\\<mm>'                   : '\U0001d52a',
        '\\<nn>'                   : '\U0001d52b',
        '\\<oo>'                   : '\U0001d52c',
        '\\<pp>'                   : '\U0001d52d',
        '\\<qq>'                   : '\U0001d52e',
        '\\<rr>'                   : '\U0001d52f',
        '\\<ss>'                   : '\U0001d530',
        '\\<tt>'                   : '\U0001d531',
        '\\<uu>'                   : '\U0001d532',
        '\\<vv>'                   : '\U0001d533',
        '\\<ww>'                   : '\U0001d534',
        '\\<xx>'                   : '\U0001d535',
        '\\<yy>'                   : '\U0001d536',
        '\\<zz>'                   : '\U0001d537',
        '\\<alpha>'                : '\U000003b1',
        '\\<beta>'                 : '\U000003b2',
        '\\<gamma>'                : '\U000003b3',
        '\\<delta>'                : '\U000003b4',
        '\\<epsilon>'              : '\U000003b5',
        '\\<zeta>'                 : '\U000003b6',
        '\\<eta>'                  : '\U000003b7',
        '\\<theta>'                : '\U000003b8',
        '\\<iota>'                 : '\U000003b9',
        '\\<kappa>'                : '\U000003ba',
        '\\<lambda>'               : '\U000003bb',
        '\\<mu>'                   : '\U000003bc',
        '\\<nu>'                   : '\U000003bd',
        '\\<xi>'                   : '\U000003be',
        '\\<pi>'                   : '\U000003c0',
        '\\<rho>'                  : '\U000003c1',
        '\\<sigma>'                : '\U000003c3',
        '\\<tau>'                  : '\U000003c4',
        '\\<upsilon>'              : '\U000003c5',
        '\\<phi>'                  : '\U000003c6',
        '\\<chi>'                  : '\U000003c7',
        '\\<psi>'                  : '\U000003c8',
        '\\<omega>'                : '\U000003c9',
        '\\<Gamma>'                : '\U00000393',
        '\\<Delta>'                : '\U00000394',
        '\\<Theta>'                : '\U00000398',
        '\\<Lambda>'               : '\U0000039b',
        '\\<Xi>'                   : '\U0000039e',
        '\\<Pi>'                   : '\U000003a0',
        '\\<Sigma>'                : '\U000003a3',
        '\\<Upsilon>'              : '\U000003a5',
        '\\<Phi>'                  : '\U000003a6',
        '\\<Psi>'                  : '\U000003a8',
        '\\<Omega>'                : '\U000003a9',
        '\\<bool>'                 : '\U0001d539',
        '\\<complex>'              : '\U00002102',
        '\\<nat>'                  : '\U00002115',
        '\\<rat>'                  : '\U0000211a',
        '\\<real>'                 : '\U0000211d',
        '\\<int>'                  : '\U00002124',
        '\\<leftarrow>'            : '\U00002190',
        '\\<longleftarrow>'        : '\U000027f5',
        '\\<rightarrow>'           : '\U00002192',
        '\\<longrightarrow>'       : '\U000027f6',
        '\\<Leftarrow>'            : '\U000021d0',
        '\\<Longleftarrow>'        : '\U000027f8',
        '\\<Rightarrow>'           : '\U000021d2',
        '\\<Longrightarrow>'       : '\U000027f9',
        '\\<leftrightarrow>'       : '\U00002194',
        '\\<longleftrightarrow>'   : '\U000027f7',
        '\\<Leftrightarrow>'       : '\U000021d4',
        '\\<Longleftrightarrow>'   : '\U000027fa',
        '\\<mapsto>'               : '\U000021a6',
        '\\<longmapsto>'           : '\U000027fc',
        '\\<midarrow>'             : '\U00002500',
        '\\<Midarrow>'             : '\U00002550',
        '\\<hookleftarrow>'        : '\U000021a9',
        '\\<hookrightarrow>'       : '\U000021aa',
        '\\<leftharpoondown>'      : '\U000021bd',
        '\\<rightharpoondown>'     : '\U000021c1',
        '\\<leftharpoonup>'        : '\U000021bc',
        '\\<rightharpoonup>'       : '\U000021c0',
        '\\<rightleftharpoons>'    : '\U000021cc',
        '\\<leadsto>'              : '\U0000219d',
        '\\<downharpoonleft>'      : '\U000021c3',
        '\\<downharpoonright>'     : '\U000021c2',
        '\\<upharpoonleft>'        : '\U000021bf',
        '\\<upharpoonright>'       : '\U000021be',
        '\\<restriction>'          : '\U000021be',
        '\\<Colon>'                : '\U00002237',
        '\\<up>'                   : '\U00002191',
        '\\<Up>'                   : '\U000021d1',
        '\\<down>'                 : '\U00002193',
        '\\<Down>'                 : '\U000021d3',
        '\\<updown>'               : '\U00002195',
        '\\<Updown>'               : '\U000021d5',
        '\\<langle>'               : '\U000027e8',
        '\\<rangle>'               : '\U000027e9',
        '\\<lceil>'                : '\U00002308',
        '\\<rceil>'                : '\U00002309',
        '\\<lfloor>'               : '\U0000230a',
        '\\<rfloor>'               : '\U0000230b',
        '\\<lparr>'                : '\U00002987',
        '\\<rparr>'                : '\U00002988',
        '\\<lbrakk>'               : '\U000027e6',
        '\\<rbrakk>'               : '\U000027e7',
        '\\<lbrace>'               : '\U00002983',
        '\\<rbrace>'               : '\U00002984',
        '\\<guillemotleft>'        : '\U000000ab',
        '\\<guillemotright>'       : '\U000000bb',
        '\\<bottom>'               : '\U000022a5',
        '\\<top>'                  : '\U000022a4',
        '\\<and>'                  : '\U00002227',
        '\\<And>'                  : '\U000022c0',
        '\\<or>'                   : '\U00002228',
        '\\<Or>'                   : '\U000022c1',
        '\\<forall>'               : '\U00002200',
        '\\<exists>'               : '\U00002203',
        '\\<nexists>'              : '\U00002204',
        '\\<not>'                  : '\U000000ac',
        '\\<box>'                  : '\U000025a1',
        '\\<diamond>'              : '\U000025c7',
        '\\<turnstile>'            : '\U000022a2',
        '\\<Turnstile>'            : '\U000022a8',
        '\\<tturnstile>'           : '\U000022a9',
        '\\<TTurnstile>'           : '\U000022ab',
        '\\<stileturn>'            : '\U000022a3',
        '\\<surd>'                 : '\U0000221a',
        '\\<le>'                   : '\U00002264',
        '\\<ge>'                   : '\U00002265',
        '\\<lless>'                : '\U0000226a',
        '\\<ggreater>'             : '\U0000226b',
        '\\<lesssim>'              : '\U00002272',
        '\\<greatersim>'           : '\U00002273',
        '\\<lessapprox>'           : '\U00002a85',
        '\\<greaterapprox>'        : '\U00002a86',
        '\\<in>'                   : '\U00002208',
        '\\<notin>'                : '\U00002209',
        '\\<subset>'               : '\U00002282',
        '\\<supset>'               : '\U00002283',
        '\\<subseteq>'             : '\U00002286',
        '\\<supseteq>'             : '\U00002287',
        '\\<sqsubset>'             : '\U0000228f',
        '\\<sqsupset>'             : '\U00002290',
        '\\<sqsubseteq>'           : '\U00002291',
        '\\<sqsupseteq>'           : '\U00002292',
        '\\<inter>'                : '\U00002229',
        '\\<Inter>'                : '\U000022c2',
        '\\<union>'                : '\U0000222a',
        '\\<Union>'                : '\U000022c3',
        '\\<squnion>'              : '\U00002294',
        '\\<Squnion>'              : '\U00002a06',
        '\\<sqinter>'              : '\U00002293',
        '\\<Sqinter>'              : '\U00002a05',
        '\\<setminus>'             : '\U00002216',
        '\\<propto>'               : '\U0000221d',
        '\\<uplus>'                : '\U0000228e',
        '\\<Uplus>'                : '\U00002a04',
        '\\<noteq>'                : '\U00002260',
        '\\<sim>'                  : '\U0000223c',
        '\\<doteq>'                : '\U00002250',
        '\\<simeq>'                : '\U00002243',
        '\\<approx>'               : '\U00002248',
        '\\<asymp>'                : '\U0000224d',
        '\\<cong>'                 : '\U00002245',
        '\\<smile>'                : '\U00002323',
        '\\<equiv>'                : '\U00002261',
        '\\<frown>'                : '\U00002322',
        '\\<Join>'                 : '\U000022c8',
        '\\<bowtie>'               : '\U00002a1d',
        '\\<prec>'                 : '\U0000227a',
        '\\<succ>'                 : '\U0000227b',
        '\\<preceq>'               : '\U0000227c',
        '\\<succeq>'               : '\U0000227d',
        '\\<parallel>'             : '\U00002225',
        '\\<bar>'                  : '\U000000a6',
        '\\<plusminus>'            : '\U000000b1',
        '\\<minusplus>'            : '\U00002213',
        '\\<times>'                : '\U000000d7',
        '\\<div>'                  : '\U000000f7',
        '\\<cdot>'                 : '\U000022c5',
        '\\<star>'                 : '\U000022c6',
        '\\<bullet>'               : '\U00002219',
        '\\<circ>'                 : '\U00002218',
        '\\<dagger>'               : '\U00002020',
        '\\<ddagger>'              : '\U00002021',
        '\\<lhd>'                  : '\U000022b2',
        '\\<rhd>'                  : '\U000022b3',
        '\\<unlhd>'                : '\U000022b4',
        '\\<unrhd>'                : '\U000022b5',
        '\\<triangleleft>'         : '\U000025c3',
        '\\<triangleright>'        : '\U000025b9',
        '\\<triangle>'             : '\U000025b3',
        '\\<triangleq>'            : '\U0000225c',
        '\\<oplus>'                : '\U00002295',
        '\\<Oplus>'                : '\U00002a01',
        '\\<otimes>'               : '\U00002297',
        '\\<Otimes>'               : '\U00002a02',
        '\\<odot>'                 : '\U00002299',
        '\\<Odot>'                 : '\U00002a00',
        '\\<ominus>'               : '\U00002296',
        '\\<oslash>'               : '\U00002298',
        '\\<dots>'                 : '\U00002026',
        '\\<cdots>'                : '\U000022ef',
        '\\<Sum>'                  : '\U00002211',
        '\\<Prod>'                 : '\U0000220f',
        '\\<Coprod>'               : '\U00002210',
        '\\<infinity>'             : '\U0000221e',
        '\\<integral>'             : '\U0000222b',
        '\\<ointegral>'            : '\U0000222e',
        '\\<clubsuit>'             : '\U00002663',
        '\\<diamondsuit>'          : '\U00002662',
        '\\<heartsuit>'            : '\U00002661',
        '\\<spadesuit>'            : '\U00002660',
        '\\<aleph>'                : '\U00002135',
        '\\<emptyset>'             : '\U00002205',
        '\\<nabla>'                : '\U00002207',
        '\\<partial>'              : '\U00002202',
        '\\<flat>'                 : '\U0000266d',
        '\\<natural>'              : '\U0000266e',
        '\\<sharp>'                : '\U0000266f',
        '\\<angle>'                : '\U00002220',
        '\\<copyright>'            : '\U000000a9',
        '\\<registered>'           : '\U000000ae',
        '\\<hyphen>'               : '\U000000ad',
        '\\<inverse>'              : '\U000000af',
        '\\<onequarter>'           : '\U000000bc',
        '\\<onehalf>'              : '\U000000bd',
        '\\<threequarters>'        : '\U000000be',
        '\\<ordfeminine>'          : '\U000000aa',
        '\\<ordmasculine>'         : '\U000000ba',
        '\\<section>'              : '\U000000a7',
        '\\<paragraph>'            : '\U000000b6',
        '\\<exclamdown>'           : '\U000000a1',
        '\\<questiondown>'         : '\U000000bf',
        '\\<euro>'                 : '\U000020ac',
        '\\<pounds>'               : '\U000000a3',
        '\\<yen>'                  : '\U000000a5',
        '\\<cent>'                 : '\U000000a2',
        '\\<currency>'             : '\U000000a4',
        '\\<degree>'               : '\U000000b0',
        '\\<amalg>'                : '\U00002a3f',
        '\\<mho>'                  : '\U00002127',
        '\\<lozenge>'              : '\U000025ca',
        '\\<wp>'                   : '\U00002118',
        '\\<wrong>'                : '\U00002240',
        '\\<struct>'               : '\U000022c4',
        '\\<acute>'                : '\U000000b4',
        '\\<index>'                : '\U00000131',
        '\\<dieresis>'             : '\U000000a8',
        '\\<cedilla>'              : '\U000000b8',
        '\\<hungarumlaut>'         : '\U000002dd',
        '\\<some>'                 : '\U000003f5',
        '\\<newline>'              : '\U000023ce',
        '\\<open>'                 : '\U00002039',
        '\\<close>'                : '\U0000203a',
        '\\<here>'                 : '\U00002302',
        '\\<^sub>'                 : '\U000021e9',
        '\\<^sup>'                 : '\U000021e7',
        '\\<^bold>'                : '\U00002759',
        '\\<^bsub>'                : '\U000021d8',
        '\\<^esub>'                : '\U000021d9',
        '\\<^bsup>'                : '\U000021d7',
        '\\<^esup>'                : '\U000021d6',
    }

    lang_map = {'isabelle' : isabelle_symbols, 'latex' : latex_symbols}

    def __init__(self, **options):
        Filter.__init__(self, **options)
        lang = get_choice_opt(options, 'lang',
                              ['isabelle', 'latex'], 'isabelle')
        self.symbols = self.lang_map[lang]

    def filter(self, lexer, stream):
        for ttype, value in stream:
            if value in self.symbols:
                yield ttype, self.symbols[value]
            else:
                yield ttype, value


class KeywordCaseFilter(Filter):
    """Convert keywords to lowercase or uppercase or capitalize them, which
    means first letter uppercase, rest lowercase.

    This can be useful e.g. if you highlight Pascal code and want to adapt the
    code to your styleguide.

    Options accepted:

    `case` : string
       The casing to convert keywords to. Must be one of ``'lower'``,
       ``'upper'`` or ``'capitalize'``.  The default is ``'lower'``.
    """

    def __init__(self, **options):
        Filter.__init__(self, **options)
        case = get_choice_opt(options, 'case',
                              ['lower', 'upper', 'capitalize'], 'lower')
        self.convert = getattr(str, case)

    def filter(self, lexer, stream):
        for ttype, value in stream:
            if ttype in Keyword:
                yield ttype, self.convert(value)
            else:
                yield ttype, value


class NameHighlightFilter(Filter):
    """Highlight a normal Name (and Name.*) token with a different token type.

    Example::

        filter = NameHighlightFilter(
            names=['foo', 'bar', 'baz'],
            tokentype=Name.Function,
        )

    This would highlight the names "foo", "bar" and "baz"
    as functions. `Name.Function` is the default token type.

    Options accepted:

    `names` : list of strings
      A list of names that should be given the different token type.
      There is no default.
    `tokentype` : TokenType or string
      A token type or a string containing a token type name that is
      used for highlighting the strings in `names`.  The default is
      `Name.Function`.
    """

    def __init__(self, **options):
        Filter.__init__(self, **options)
        self.names = set(get_list_opt(options, 'names', []))
        tokentype = options.get('tokentype')
        if tokentype:
            self.tokentype = string_to_tokentype(tokentype)
        else:
            self.tokentype = Name.Function

    def filter(self, lexer, stream):
        for ttype, value in stream:
            if ttype in Name and value in self.names:
                yield self.tokentype, value
            else:
                yield ttype, value


class ErrorToken(Exception):
    pass


class RaiseOnErrorTokenFilter(Filter):
    """Raise an exception when the lexer generates an error token.

    Options accepted:

    `excclass` : Exception class
      The exception class to raise.
      The default is `pygments.filters.ErrorToken`.

    .. versionadded:: 0.8
    """

    def __init__(self, **options):
        Filter.__init__(self, **options)
        self.exception = options.get('excclass', ErrorToken)
        try:
            # issubclass() will raise TypeError if first argument is not a class
            if not issubclass(self.exception, Exception):
                raise TypeError
        except TypeError:
            raise OptionError('excclass option is not an exception class')

    def filter(self, lexer, stream):
        for ttype, value in stream:
            if ttype is Error:
                raise self.exception(value)
            yield ttype, value


class VisibleWhitespaceFilter(Filter):
    """Convert tabs, newlines and/or spaces to visible characters.

    Options accepted:

    `spaces` : string or bool
      If this is a one-character string, spaces will be replaces by this string.
      If it is another true value, spaces will be replaced by ``·`` (unicode
      MIDDLE DOT).  If it is a false value, spaces will not be replaced.  The
      default is ``False``.
    `tabs` : string or bool
      The same as for `spaces`, but the default replacement character is ``»``
      (unicode RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK).  The default value
      is ``False``.  Note: this will not work if the `tabsize` option for the
      lexer is nonzero, as tabs will already have been expanded then.
    `tabsize` : int
      If tabs are to be replaced by this filter (see the `tabs` option), this
      is the total number of characters that a tab should be expanded to.
      The default is ``8``.
    `newlines` : string or bool
      The same as for `spaces`, but the default replacement character is ``¶``
      (unicode PILCROW SIGN).  The default value is ``False``.
    `wstokentype` : bool
      If true, give whitespace the special `Whitespace` token type.  This allows
      styling the visible whitespace differently (e.g. greyed out), but it can
      disrupt background colors.  The default is ``True``.

    .. versionadded:: 0.8
    """

    def __init__(self, **options):
        Filter.__init__(self, **options)
        for name, default in [('spaces',   '·'),
                              ('tabs',     '»'),
                              ('newlines', '¶')]:
            opt = options.get(name, False)
            if isinstance(opt, str) and len(opt) == 1:
                setattr(self, name, opt)
            else:
                setattr(self, name, (opt and default or ''))
        tabsize = get_int_opt(options, 'tabsize', 8)
        if self.tabs:
            self.tabs += ' ' * (tabsize - 1)
        if self.newlines:
            self.newlines += '\n'
        self.wstt = get_bool_opt(options, 'wstokentype', True)

    def filter(self, lexer, stream):
        if self.wstt:
            spaces = self.spaces or ' '
            tabs = self.tabs or '\t'
            newlines = self.newlines or '\n'
            regex = re.compile(r'\s')

            def replacefunc(wschar):
                if wschar == ' ':
                    return spaces
                elif wschar == '\t':
                    return tabs
                elif wschar == '\n':
                    return newlines
                return wschar

            for ttype, value in stream:
                yield from _replace_special(ttype, value, regex, Whitespace,
                                            replacefunc)
        else:
            spaces, tabs, newlines = self.spaces, self.tabs, self.newlines
            # simpler processing
            for ttype, value in stream:
                if spaces:
                    value = value.replace(' ', spaces)
                if tabs:
                    value = value.replace('\t', tabs)
                if newlines:
                    value = value.replace('\n', newlines)
                yield ttype, value


class GobbleFilter(Filter):
    """Gobbles source code lines (eats initial characters).

    This filter drops the first ``n`` characters off every line of code.  This
    may be useful when the source code fed to the lexer is indented by a fixed
    amount of space that isn't desired in the output.

    Options accepted:

    `n` : int
       The number of characters to gobble.

    .. versionadded:: 1.2
    """
    def __init__(self, **options):
        Filter.__init__(self, **options)
        self.n = get_int_opt(options, 'n', 0)

    def gobble(self, value, left):
        if left < len(value):
            return value[left:], 0
        else:
            return '', left - len(value)

    def filter(self, lexer, stream):
        n = self.n
        left = n  # How many characters left to gobble.
        for ttype, value in stream:
            # Remove ``left`` tokens from first line, ``n`` from all others.
            parts = value.split('\n')
            (parts[0], left) = self.gobble(parts[0], left)
            for i in range(1, len(parts)):
                (parts[i], left) = self.gobble(parts[i], n)
            value = '\n'.join(parts)

            if value != '':
                yield ttype, value


class TokenMergeFilter(Filter):
    """Merges consecutive tokens with the same token type in the output
    stream of a lexer.

    .. versionadded:: 1.2
    """
    def __init__(self, **options):
        Filter.__init__(self, **options)

    def filter(self, lexer, stream):
        current_type = None
        current_value = None
        for ttype, value in stream:
            if ttype is current_type:
                current_value += value
            else:
                if current_type is not None:
                    yield current_type, current_value
                current_type = ttype
                current_value = value
        if current_type is not None:
            yield current_type, current_value


FILTERS = {
    'codetagify':     CodeTagFilter,
    'keywordcase':    KeywordCaseFilter,
    'highlight':      NameHighlightFilter,
    'raiseonerror':   RaiseOnErrorTokenFilter,
    'whitespace':     VisibleWhitespaceFilter,
    'gobble':         GobbleFilter,
    'tokenmerge':     TokenMergeFilter,
    'symbols':        SymbolFilter,
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\libmp\libmpi.py ===
"""
Computational functions for interval arithmetic.

"""

from .backend import xrange

from .libmpf import (
    ComplexResult,
    round_down, round_up, round_floor, round_ceiling, round_nearest,
    prec_to_dps, repr_dps, dps_to_prec,
    bitcount,
    from_float,
    fnan, finf, fninf, fzero, fhalf, fone, fnone,
    mpf_sign, mpf_lt, mpf_le, mpf_gt, mpf_ge, mpf_eq, mpf_cmp,
    mpf_min_max,
    mpf_floor, from_int, to_int, to_str, from_str,
    mpf_abs, mpf_neg, mpf_pos, mpf_add, mpf_sub, mpf_mul, mpf_mul_int,
    mpf_div, mpf_shift, mpf_pow_int,
    from_man_exp, MPZ_ONE)

from .libelefun import (
    mpf_log, mpf_exp, mpf_sqrt, mpf_atan, mpf_atan2,
    mpf_pi, mod_pi2, mpf_cos_sin
)

from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma

def mpi_str(s, prec):
    sa, sb = s
    dps = prec_to_dps(prec) + 5
    return "[%s, %s]" % (to_str(sa, dps), to_str(sb, dps))
    #dps = prec_to_dps(prec)
    #m = mpi_mid(s, prec)
    #d = mpf_shift(mpi_delta(s, 20), -1)
    #return "%s +/- %s" % (to_str(m, dps), to_str(d, 3))

mpi_zero = (fzero, fzero)
mpi_one = (fone, fone)

def mpi_eq(s, t):
    return s == t

def mpi_ne(s, t):
    return s != t

def mpi_lt(s, t):
    sa, sb = s
    ta, tb = t
    if mpf_lt(sb, ta): return True
    if mpf_ge(sa, tb): return False
    return None

def mpi_le(s, t):
    sa, sb = s
    ta, tb = t
    if mpf_le(sb, ta): return True
    if mpf_gt(sa, tb): return False
    return None

def mpi_gt(s, t): return mpi_lt(t, s)
def mpi_ge(s, t): return mpi_le(t, s)

def mpi_add(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    a = mpf_add(sa, ta, prec, round_floor)
    b = mpf_add(sb, tb, prec, round_ceiling)
    if a == fnan: a = fninf
    if b == fnan: b = finf
    return a, b

def mpi_sub(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    a = mpf_sub(sa, tb, prec, round_floor)
    b = mpf_sub(sb, ta, prec, round_ceiling)
    if a == fnan: a = fninf
    if b == fnan: b = finf
    return a, b

def mpi_delta(s, prec):
    sa, sb = s
    return mpf_sub(sb, sa, prec, round_up)

def mpi_mid(s, prec):
    sa, sb = s
    return mpf_shift(mpf_add(sa, sb, prec, round_nearest), -1)

def mpi_pos(s, prec):
    sa, sb = s
    a = mpf_pos(sa, prec, round_floor)
    b = mpf_pos(sb, prec, round_ceiling)
    return a, b

def mpi_neg(s, prec=0):
    sa, sb = s
    a = mpf_neg(sb, prec, round_floor)
    b = mpf_neg(sa, prec, round_ceiling)
    return a, b

def mpi_abs(s, prec=0):
    sa, sb = s
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    # Both points nonnegative?
    if sas >= 0:
        a = mpf_pos(sa, prec, round_floor)
        b = mpf_pos(sb, prec, round_ceiling)
    # Upper point nonnegative?
    elif sbs >= 0:
        a = fzero
        negsa = mpf_neg(sa)
        if mpf_lt(negsa, sb):
            b = mpf_pos(sb, prec, round_ceiling)
        else:
            b = mpf_pos(negsa, prec, round_ceiling)
    # Both negative?
    else:
        a = mpf_neg(sb, prec, round_floor)
        b = mpf_neg(sa, prec, round_ceiling)
    return a, b

# TODO: optimize
def mpi_mul_mpf(s, t, prec):
    return mpi_mul(s, (t, t), prec)

def mpi_div_mpf(s, t, prec):
    return mpi_div(s, (t, t), prec)

def mpi_mul(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    tas = mpf_sign(ta)
    tbs = mpf_sign(tb)
    if sas == sbs == 0:
        # Should maybe be undefined
        if ta == fninf or tb == finf:
            return fninf, finf
        return fzero, fzero
    if tas == tbs == 0:
        # Should maybe be undefined
        if sa == fninf or sb == finf:
            return fninf, finf
        return fzero, fzero
    if sas >= 0:
        # positive * positive
        if tas >= 0:
            a = mpf_mul(sa, ta, prec, round_floor)
            b = mpf_mul(sb, tb, prec, round_ceiling)
            if a == fnan: a = fzero
            if b == fnan: b = finf
        # positive * negative
        elif tbs <= 0:
            a = mpf_mul(sb, ta, prec, round_floor)
            b = mpf_mul(sa, tb, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = fzero
        # positive * both signs
        else:
            a = mpf_mul(sb, ta, prec, round_floor)
            b = mpf_mul(sb, tb, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = finf
    elif sbs <= 0:
        # negative * positive
        if tas >= 0:
            a = mpf_mul(sa, tb, prec, round_floor)
            b = mpf_mul(sb, ta, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = fzero
        # negative * negative
        elif tbs <= 0:
            a = mpf_mul(sb, tb, prec, round_floor)
            b = mpf_mul(sa, ta, prec, round_ceiling)
            if a == fnan: a = fzero
            if b == fnan: b = finf
        # negative * both signs
        else:
            a = mpf_mul(sa, tb, prec, round_floor)
            b = mpf_mul(sa, ta, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = finf
    else:
        # General case: perform all cross-multiplications and compare
        # Since the multiplications can be done exactly, we need only
        # do 4 (instead of 8: two for each rounding mode)
        cases = [mpf_mul(sa, ta), mpf_mul(sa, tb), mpf_mul(sb, ta), mpf_mul(sb, tb)]
        if fnan in cases:
            a, b = (fninf, finf)
        else:
            a, b = mpf_min_max(cases)
            a = mpf_pos(a, prec, round_floor)
            b = mpf_pos(b, prec, round_ceiling)
    return a, b

def mpi_square(s, prec=0):
    sa, sb = s
    if mpf_ge(sa, fzero):
        a = mpf_mul(sa, sa, prec, round_floor)
        b = mpf_mul(sb, sb, prec, round_ceiling)
    elif mpf_le(sb, fzero):
        a = mpf_mul(sb, sb, prec, round_floor)
        b = mpf_mul(sa, sa, prec, round_ceiling)
    else:
        sa = mpf_neg(sa)
        sa, sb = mpf_min_max([sa, sb])
        a = fzero
        b = mpf_mul(sb, sb, prec, round_ceiling)
    return a, b

def mpi_div(s, t, prec):
    sa, sb = s
    ta, tb = t
    sas = mpf_sign(sa)
    sbs = mpf_sign(sb)
    tas = mpf_sign(ta)
    tbs = mpf_sign(tb)
    # 0 / X
    if sas == sbs == 0:
        # 0 / <interval containing 0>
        if (tas < 0 and tbs > 0) or (tas == 0 or tbs == 0):
            return fninf, finf
        return fzero, fzero
    # Denominator contains both negative and positive numbers;
    # this should properly be a multi-interval, but the closest
    # match is the entire (extended) real line
    if tas < 0 and tbs > 0:
        return fninf, finf
    # Assume denominator to be nonnegative
    if tas < 0:
        return mpi_div(mpi_neg(s), mpi_neg(t), prec)
    # Division by zero
    # XXX: make sure all results make sense
    if tas == 0:
        # Numerator contains both signs?
        if sas < 0 and sbs > 0:
            return fninf, finf
        if tas == tbs:
            return fninf, finf
        # Numerator positive?
        if sas >= 0:
            a = mpf_div(sa, tb, prec, round_floor)
            b = finf
        if sbs <= 0:
            a = fninf
            b = mpf_div(sb, tb, prec, round_ceiling)
    # Division with positive denominator
    # We still have to handle nans resulting from inf/0 or inf/inf
    else:
        # Nonnegative numerator
        if sas >= 0:
            a = mpf_div(sa, tb, prec, round_floor)
            b = mpf_div(sb, ta, prec, round_ceiling)
            if a == fnan: a = fzero
            if b == fnan: b = finf
        # Nonpositive numerator
        elif sbs <= 0:
            a = mpf_div(sa, ta, prec, round_floor)
            b = mpf_div(sb, tb, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = fzero
        # Numerator contains both signs?
        else:
            a = mpf_div(sa, ta, prec, round_floor)
            b = mpf_div(sb, ta, prec, round_ceiling)
            if a == fnan: a = fninf
            if b == fnan: b = finf
    return a, b

def mpi_pi(prec):
    a = mpf_pi(prec, round_floor)
    b = mpf_pi(prec, round_ceiling)
    return a, b

def mpi_exp(s, prec):
    sa, sb = s
    # exp is monotonic
    a = mpf_exp(sa, prec, round_floor)
    b = mpf_exp(sb, prec, round_ceiling)
    return a, b

def mpi_log(s, prec):
    sa, sb = s
    # log is monotonic
    a = mpf_log(sa, prec, round_floor)
    b = mpf_log(sb, prec, round_ceiling)
    return a, b

def mpi_sqrt(s, prec):
    sa, sb = s
    # sqrt is monotonic
    a = mpf_sqrt(sa, prec, round_floor)
    b = mpf_sqrt(sb, prec, round_ceiling)
    return a, b

def mpi_atan(s, prec):
    sa, sb = s
    a = mpf_atan(sa, prec, round_floor)
    b = mpf_atan(sb, prec, round_ceiling)
    return a, b

def mpi_pow_int(s, n, prec):
    sa, sb = s
    if n < 0:
        return mpi_div((fone, fone), mpi_pow_int(s, -n, prec+20), prec)
    if n == 0:
        return (fone, fone)
    if n == 1:
        return s
    if n == 2:
        return mpi_square(s, prec)
    # Odd -- signs are preserved
    if n & 1:
        a = mpf_pow_int(sa, n, prec, round_floor)
        b = mpf_pow_int(sb, n, prec, round_ceiling)
    # Even -- important to ensure positivity
    else:
        sas = mpf_sign(sa)
        sbs = mpf_sign(sb)
        # Nonnegative?
        if sas >= 0:
            a = mpf_pow_int(sa, n, prec, round_floor)
            b = mpf_pow_int(sb, n, prec, round_ceiling)
        # Nonpositive?
        elif sbs <= 0:
            a = mpf_pow_int(sb, n, prec, round_floor)
            b = mpf_pow_int(sa, n, prec, round_ceiling)
        # Mixed signs?
        else:
            a = fzero
            # max(-a,b)**n
            sa = mpf_neg(sa)
            if mpf_ge(sa, sb):
                b = mpf_pow_int(sa, n, prec, round_ceiling)
            else:
                b = mpf_pow_int(sb, n, prec, round_ceiling)
    return a, b

def mpi_pow(s, t, prec):
    ta, tb = t
    if ta == tb and ta not in (finf, fninf):
        if ta == from_int(to_int(ta)):
            return mpi_pow_int(s, to_int(ta), prec)
        if ta == fhalf:
            return mpi_sqrt(s, prec)
    u = mpi_log(s, prec + 20)
    v = mpi_mul(u, t, prec + 20)
    return mpi_exp(v, prec)

def MIN(x, y):
    if mpf_le(x, y):
        return x
    return y

def MAX(x, y):
    if mpf_ge(x, y):
        return x
    return y

def cos_sin_quadrant(x, wp):
    sign, man, exp, bc = x
    if x == fzero:
        return fone, fzero, 0
    # TODO: combine evaluation code to avoid duplicate modulo
    c, s = mpf_cos_sin(x, wp)
    t, n, wp_ = mod_pi2(man, exp, exp+bc, 15)
    if sign:
        n = -1-n
    return c, s, n

def mpi_cos_sin(x, prec):
    a, b = x
    if a == b == fzero:
        return (fone, fone), (fzero, fzero)
    # Guaranteed to contain both -1 and 1
    if (finf in x) or (fninf in x):
        return (fnone, fone), (fnone, fone)
    wp = prec + 20
    ca, sa, na = cos_sin_quadrant(a, wp)
    cb, sb, nb = cos_sin_quadrant(b, wp)
    ca, cb = mpf_min_max([ca, cb])
    sa, sb = mpf_min_max([sa, sb])
    # Both functions are monotonic within one quadrant
    if na == nb:
        pass
    # Guaranteed to contain both -1 and 1
    elif nb - na >= 4:
        return (fnone, fone), (fnone, fone)
    else:
        # cos has maximum between a and b
        if na//4 != nb//4:
            cb = fone
        # cos has minimum
        if (na-2)//4 != (nb-2)//4:
            ca = fnone
        # sin has maximum
        if (na-1)//4 != (nb-1)//4:
            sb = fone
        # sin has minimum
        if (na-3)//4 != (nb-3)//4:
            sa = fnone
    # Perturb to force interval rounding
    more = from_man_exp((MPZ_ONE<<wp) + (MPZ_ONE<<10), -wp)
    less = from_man_exp((MPZ_ONE<<wp) - (MPZ_ONE<<10), -wp)
    def finalize(v, rounding):
        if bool(v[0]) == (rounding == round_floor):
            p = more
        else:
            p = less
        v = mpf_mul(v, p, prec, rounding)
        sign, man, exp, bc = v
        if exp+bc >= 1:
            if sign:
                return fnone
            return fone
        return v
    ca = finalize(ca, round_floor)
    cb = finalize(cb, round_ceiling)
    sa = finalize(sa, round_floor)
    sb = finalize(sb, round_ceiling)
    return (ca,cb), (sa,sb)

def mpi_cos(x, prec):
    return mpi_cos_sin(x, prec)[0]

def mpi_sin(x, prec):
    return mpi_cos_sin(x, prec)[1]

def mpi_tan(x, prec):
    cos, sin = mpi_cos_sin(x, prec+20)
    return mpi_div(sin, cos, prec)

def mpi_cot(x, prec):
    cos, sin = mpi_cos_sin(x, prec+20)
    return mpi_div(cos, sin, prec)

def mpi_from_str_a_b(x, y, percent, prec):
    wp = prec + 20
    xa = from_str(x, wp, round_floor)
    xb = from_str(x, wp, round_ceiling)
    #ya = from_str(y, wp, round_floor)
    y = from_str(y, wp, round_ceiling)
    assert mpf_ge(y, fzero)
    if percent:
        y = mpf_mul(MAX(mpf_abs(xa), mpf_abs(xb)), y, wp, round_ceiling)
        y = mpf_div(y, from_int(100), wp, round_ceiling)
    a = mpf_sub(xa, y, prec, round_floor)
    b = mpf_add(xb, y, prec, round_ceiling)
    return a, b

def mpi_from_str(s, prec):
    """
    Parse an interval number given as a string.

    Allowed forms are

    "-1.23e-27"
        Any single decimal floating-point literal.
    "a +- b"  or  "a (b)"
        a is the midpoint of the interval and b is the half-width
    "a +- b%"  or  "a (b%)"
        a is the midpoint of the interval and the half-width
        is b percent of a (`a \times b / 100`).
    "[a, b]"
        The interval indicated directly.
    "x[y,z]e"
        x are shared digits, y and z are unequal digits, e is the exponent.

    """
    e = ValueError("Improperly formed interval number '%s'" % s)
    s = s.replace(" ", "")
    wp = prec + 20
    if "+-" in s:
        x, y = s.split("+-")
        return mpi_from_str_a_b(x, y, False, prec)
    # case 2
    elif "(" in s:
        # Don't confuse with a complex number (x,y)
        if s[0] == "(" or ")" not in s:
            raise e
        s = s.replace(")", "")
        percent = False
        if "%" in s:
            if s[-1] != "%":
                raise e
            percent = True
            s = s.replace("%", "")
        x, y = s.split("(")
        return mpi_from_str_a_b(x, y, percent, prec)
    elif "," in s:
        if ('[' not in s) or (']' not in s):
            raise e
        if s[0] == '[':
            # case 3
            s = s.replace("[", "")
            s = s.replace("]", "")
            a, b = s.split(",")
            a = from_str(a, prec, round_floor)
            b = from_str(b, prec, round_ceiling)
            return a, b
        else:
            # case 4
            x, y = s.split('[')
            y, z = y.split(',')
            if 'e' in s:
                z, e = z.split(']')
            else:
                z, e = z.rstrip(']'), ''
            a = from_str(x+y+e, prec, round_floor)
            b = from_str(x+z+e, prec, round_ceiling)
            return a, b
    else:
        a = from_str(s, prec, round_floor)
        b = from_str(s, prec, round_ceiling)
        return a, b

def mpi_to_str(x, dps, use_spaces=True, brackets='[]', mode='brackets', error_dps=4, **kwargs):
    """
    Convert a mpi interval to a string.

    **Arguments**

    *dps*
        decimal places to use for printing
    *use_spaces*
        use spaces for more readable output, defaults to true
    *brackets*
        pair of strings (or two-character string) giving left and right brackets
    *mode*
        mode of display: 'plusminus', 'percent', 'brackets' (default) or 'diff'
    *error_dps*
        limit the error to *error_dps* digits (mode 'plusminus and 'percent')

    Additional keyword arguments are forwarded to the mpf-to-string conversion
    for the components of the output.

    **Examples**

        >>> from mpmath import mpi, mp
        >>> mp.dps = 30
        >>> x = mpi(1, 2)._mpi_
        >>> mpi_to_str(x, 2, mode='plusminus')
        '1.5 +- 0.5'
        >>> mpi_to_str(x, 2, mode='percent')
        '1.5 (33.33%)'
        >>> mpi_to_str(x, 2, mode='brackets')
        '[1.0, 2.0]'
        >>> mpi_to_str(x, 2, mode='brackets' , brackets=('<', '>'))
        '<1.0, 2.0>'
        >>> x = mpi('5.2582327113062393041', '5.2582327113062749951')._mpi_
        >>> mpi_to_str(x, 15, mode='diff')
        '5.2582327113062[4, 7]'
        >>> mpi_to_str(mpi(0)._mpi_, 2, mode='percent')
        '0.0 (0.0%)'

    """
    prec = dps_to_prec(dps)
    wp = prec + 20
    a, b = x
    mid = mpi_mid(x, prec)
    delta = mpi_delta(x, prec)
    a_str = to_str(a, dps, **kwargs)
    b_str = to_str(b, dps, **kwargs)
    mid_str = to_str(mid, dps, **kwargs)
    sp = ""
    if use_spaces:
        sp = " "
    br1, br2 = brackets
    if mode == 'plusminus':
        delta_str = to_str(mpf_shift(delta,-1), dps, **kwargs)
        s = mid_str + sp + "+-" + sp + delta_str
    elif mode == 'percent':
        if mid == fzero:
            p = fzero
        else:
            # p = 100 * delta(x) / (2*mid(x))
            p = mpf_mul(delta, from_int(100))
            p = mpf_div(p, mpf_mul(mid, from_int(2)), wp)
        s = mid_str + sp + "(" + to_str(p, error_dps) + "%)"
    elif mode == 'brackets':
        s = br1 + a_str + "," + sp + b_str + br2
    elif mode == 'diff':
        # use more digits if str(x.a) and str(x.b) are equal
        if a_str == b_str:
            a_str = to_str(a, dps+3, **kwargs)
            b_str = to_str(b, dps+3, **kwargs)
        # separate mantissa and exponent
        a = a_str.split('e')
        if len(a) == 1:
            a.append('')
        b = b_str.split('e')
        if len(b) == 1:
            b.append('')
        if a[1] == b[1]:
            if a[0] != b[0]:
                for i in xrange(len(a[0]) + 1):
                    if a[0][i] != b[0][i]:
                        break
                s = (a[0][:i] + br1 + a[0][i:] + ',' + sp + b[0][i:] + br2
                     + 'e'*min(len(a[1]), 1) + a[1])
            else: # no difference
                s = a[0] + br1 + br2 + 'e'*min(len(a[1]), 1) + a[1]
        else:
            s = br1 + 'e'.join(a) + ',' + sp + 'e'.join(b) + br2
    else:
        raise ValueError("'%s' is unknown mode for printing mpi" % mode)
    return s

def mpci_add(x, y, prec):
    a, b = x
    c, d = y
    return mpi_add(a, c, prec), mpi_add(b, d, prec)

def mpci_sub(x, y, prec):
    a, b = x
    c, d = y
    return mpi_sub(a, c, prec), mpi_sub(b, d, prec)

def mpci_neg(x, prec=0):
    a, b = x
    return mpi_neg(a, prec), mpi_neg(b, prec)

def mpci_pos(x, prec):
    a, b = x
    return mpi_pos(a, prec), mpi_pos(b, prec)

def mpci_mul(x, y, prec):
    # TODO: optimize for real/imag cases
    a, b = x
    c, d = y
    r1 = mpi_mul(a,c)
    r2 = mpi_mul(b,d)
    re = mpi_sub(r1,r2,prec)
    i1 = mpi_mul(a,d)
    i2 = mpi_mul(b,c)
    im = mpi_add(i1,i2,prec)
    return re, im

def mpci_div(x, y, prec):
    # TODO: optimize for real/imag cases
    a, b = x
    c, d = y
    wp = prec+20
    m1 = mpi_square(c)
    m2 = mpi_square(d)
    m = mpi_add(m1,m2,wp)
    re = mpi_add(mpi_mul(a,c), mpi_mul(b,d), wp)
    im = mpi_sub(mpi_mul(b,c), mpi_mul(a,d), wp)
    re = mpi_div(re, m, prec)
    im = mpi_div(im, m, prec)
    return re, im

def mpci_exp(x, prec):
    a, b = x
    wp = prec+20
    r = mpi_exp(a, wp)
    c, s = mpi_cos_sin(b, wp)
    a = mpi_mul(r, c, prec)
    b = mpi_mul(r, s, prec)
    return a, b

def mpi_shift(x, n):
    a, b = x
    return mpf_shift(a,n), mpf_shift(b,n)

def mpi_cosh_sinh(x, prec):
    # TODO: accuracy for small x
    wp = prec+20
    e1 = mpi_exp(x, wp)
    e2 = mpi_div(mpi_one, e1, wp)
    c = mpi_add(e1, e2, prec)
    s = mpi_sub(e1, e2, prec)
    c = mpi_shift(c, -1)
    s = mpi_shift(s, -1)
    return c, s

def mpci_cos(x, prec):
    a, b = x
    wp = prec+10
    c, s = mpi_cos_sin(a, wp)
    ch, sh = mpi_cosh_sinh(b, wp)
    re = mpi_mul(c, ch, prec)
    im = mpi_mul(s, sh, prec)
    return re, mpi_neg(im)

def mpci_sin(x, prec):
    a, b = x
    wp = prec+10
    c, s = mpi_cos_sin(a, wp)
    ch, sh = mpi_cosh_sinh(b, wp)
    re = mpi_mul(s, ch, prec)
    im = mpi_mul(c, sh, prec)
    return re, im

def mpci_abs(x, prec):
    a, b = x
    if a == mpi_zero:
        return mpi_abs(b)
    if b == mpi_zero:
        return mpi_abs(a)
    # Important: nonnegative
    a = mpi_square(a)
    b = mpi_square(b)
    t = mpi_add(a, b, prec+20)
    return mpi_sqrt(t, prec)

def mpi_atan2(y, x, prec):
    ya, yb = y
    xa, xb = x
    # Constrained to the real line
    if ya == yb == fzero:
        if mpf_ge(xa, fzero):
            return mpi_zero
        return mpi_pi(prec)
    # Right half-plane
    if mpf_ge(xa, fzero):
        if mpf_ge(ya, fzero):
            a = mpf_atan2(ya, xb, prec, round_floor)
        else:
            a = mpf_atan2(ya, xa, prec, round_floor)
        if mpf_ge(yb, fzero):
            b = mpf_atan2(yb, xa, prec, round_ceiling)
        else:
            b = mpf_atan2(yb, xb, prec, round_ceiling)
    # Upper half-plane
    elif mpf_ge(ya, fzero):
        b = mpf_atan2(ya, xa, prec, round_ceiling)
        if mpf_le(xb, fzero):
            a = mpf_atan2(yb, xb, prec, round_floor)
        else:
            a = mpf_atan2(ya, xb, prec, round_floor)
    # Lower half-plane
    elif mpf_le(yb, fzero):
        a = mpf_atan2(yb, xa, prec, round_floor)
        if mpf_le(xb, fzero):
            b = mpf_atan2(ya, xb, prec, round_ceiling)
        else:
            b = mpf_atan2(yb, xb, prec, round_ceiling)
    # Covering the origin
    else:
        b = mpf_pi(prec, round_ceiling)
        a = mpf_neg(b)
    return a, b

def mpci_arg(z, prec):
    x, y = z
    return mpi_atan2(y, x, prec)

def mpci_log(z, prec):
    x, y = z
    re = mpi_log(mpci_abs(z, prec+20), prec)
    im = mpci_arg(z, prec)
    return re, im

def mpci_pow(x, y, prec):
    # TODO: recognize/speed up real cases, integer y
    yre, yim = y
    if yim == mpi_zero:
        ya, yb = yre
        if ya == yb:
            sign, man, exp, bc = yb
            if man and exp >= 0:
                return mpci_pow_int(x, (-1)**sign * int(man<<exp), prec)
            # x^0
            if yb == fzero:
                return mpci_pow_int(x, 0, prec)
    wp = prec+20
    return mpci_exp(mpci_mul(y, mpci_log(x, wp), wp), prec)

def mpci_square(x, prec):
    a, b = x
    # (a+bi)^2 = (a^2-b^2) + 2abi
    re = mpi_sub(mpi_square(a), mpi_square(b), prec)
    im = mpi_mul(a, b, prec)
    im = mpi_shift(im, 1)
    return re, im

def mpci_pow_int(x, n, prec):
    if n < 0:
        return mpci_div((mpi_one,mpi_zero), mpci_pow_int(x, -n, prec+20), prec)
    if n == 0:
        return mpi_one, mpi_zero
    if n == 1:
        return mpci_pos(x, prec)
    if n == 2:
        return mpci_square(x, prec)
    wp = prec + 20
    result = (mpi_one, mpi_zero)
    while n:
        if n & 1:
            result = mpci_mul(result, x, wp)
            n -= 1
        x = mpci_square(x, wp)
        n >>= 1
    return mpci_pos(result, prec)

gamma_min_a = from_float(1.46163214496)
gamma_min_b = from_float(1.46163214497)
gamma_min = (gamma_min_a, gamma_min_b)
gamma_mono_imag_a = from_float(-1.1)
gamma_mono_imag_b = from_float(1.1)

def mpi_overlap(x, y):
    a, b = x
    c, d = y
    if mpf_lt(d, a): return False
    if mpf_gt(c, b): return False
    return True

# type = 0 -- gamma
# type = 1 -- factorial
# type = 2 -- 1/gamma
# type = 3 -- log-gamma

def mpi_gamma(z, prec, type=0):
    a, b = z
    wp = prec+20

    if type == 1:
        return mpi_gamma(mpi_add(z, mpi_one, wp), prec, 0)

    # increasing
    if mpf_gt(a, gamma_min_b):
        if type == 0:
            c = mpf_gamma(a, prec, round_floor)
            d = mpf_gamma(b, prec, round_ceiling)
        elif type == 2:
            c = mpf_rgamma(b, prec, round_floor)
            d = mpf_rgamma(a, prec, round_ceiling)
        elif type == 3:
            c = mpf_loggamma(a, prec, round_floor)
            d = mpf_loggamma(b, prec, round_ceiling)
    # decreasing
    elif mpf_gt(a, fzero) and mpf_lt(b, gamma_min_a):
        if type == 0:
            c = mpf_gamma(b, prec, round_floor)
            d = mpf_gamma(a, prec, round_ceiling)
        elif type == 2:
            c = mpf_rgamma(a, prec, round_floor)
            d = mpf_rgamma(b, prec, round_ceiling)
        elif type == 3:
            c = mpf_loggamma(b, prec, round_floor)
            d = mpf_loggamma(a, prec, round_ceiling)
    else:
        # TODO: reflection formula
        znew = mpi_add(z, mpi_one, wp)
        if type == 0: return mpi_div(mpi_gamma(znew, prec+2, 0), z, prec)
        if type == 2: return mpi_mul(mpi_gamma(znew, prec+2, 2), z, prec)
        if type == 3: return mpi_sub(mpi_gamma(znew, prec+2, 3), mpi_log(z, prec+2), prec)
    return c, d

def mpci_gamma(z, prec, type=0):
    (a1,a2), (b1,b2) = z

    # Real case
    if b1 == b2 == fzero and (type != 3 or mpf_gt(a1,fzero)):
        return mpi_gamma(z, prec, type), mpi_zero

    # Estimate precision
    wp = prec+20
    if type != 3:
        amag = a2[2]+a2[3]
        bmag = b2[2]+b2[3]
        if a2 != fzero:
            mag = max(amag, bmag)
        else:
            mag = bmag
        an = abs(to_int(a2))
        bn = abs(to_int(b2))
        absn = max(an, bn)
        gamma_size = max(0,absn*mag)
        wp += bitcount(gamma_size)

    # Assume type != 1
    if type == 1:
        (a1,a2) = mpi_add((a1,a2), mpi_one, wp); z = (a1,a2), (b1,b2)
        type = 0

    # Avoid non-monotonic region near the negative real axis
    if mpf_lt(a1, gamma_min_b):
        if mpi_overlap((b1,b2), (gamma_mono_imag_a, gamma_mono_imag_b)):
            # TODO: reflection formula
            #if mpf_lt(a2, mpf_shift(fone,-1)):
            #    znew = mpci_sub((mpi_one,mpi_zero),z,wp)
            #    ...
            # Recurrence:
            # gamma(z) = gamma(z+1)/z
            znew = mpi_add((a1,a2), mpi_one, wp), (b1,b2)
            if type == 0: return mpci_div(mpci_gamma(znew, prec+2, 0), z, prec)
            if type == 2: return mpci_mul(mpci_gamma(znew, prec+2, 2), z, prec)
            if type == 3: return mpci_sub(mpci_gamma(znew, prec+2, 3), mpci_log(z,prec+2), prec)

    # Use monotonicity (except for a small region close to the
    # origin and near poles)
    # upper half-plane
    if mpf_ge(b1, fzero):
        minre = mpc_loggamma((a1,b2), wp, round_floor)
        maxre = mpc_loggamma((a2,b1), wp, round_ceiling)
        minim = mpc_loggamma((a1,b1), wp, round_floor)
        maxim = mpc_loggamma((a2,b2), wp, round_ceiling)
    # lower half-plane
    elif mpf_le(b2, fzero):
        minre = mpc_loggamma((a1,b1), wp, round_floor)
        maxre = mpc_loggamma((a2,b2), wp, round_ceiling)
        minim = mpc_loggamma((a2,b1), wp, round_floor)
        maxim = mpc_loggamma((a1,b2), wp, round_ceiling)
    # crosses real axis
    else:
        maxre = mpc_loggamma((a2,fzero), wp, round_ceiling)
        # stretches more into the lower half-plane
        if mpf_gt(mpf_neg(b1), b2):
            minre = mpc_loggamma((a1,b1), wp, round_ceiling)
        else:
            minre = mpc_loggamma((a1,b2), wp, round_ceiling)
        minim = mpc_loggamma((a2,b1), wp, round_floor)
        maxim = mpc_loggamma((a2,b2), wp, round_floor)

    w = (minre[0], maxre[0]), (minim[1], maxim[1])
    if type == 3:
        return mpi_pos(w[0], prec), mpi_pos(w[1], prec)
    if type == 2:
        w = mpci_neg(w)
    return mpci_exp(w, prec)

def mpi_loggamma(z, prec): return mpi_gamma(z, prec, type=3)
def mpci_loggamma(z, prec): return mpci_gamma(z, prec, type=3)

def mpi_rgamma(z, prec): return mpi_gamma(z, prec, type=2)
def mpci_rgamma(z, prec): return mpci_gamma(z, prec, type=2)

def mpi_factorial(z, prec): return mpi_gamma(z, prec, type=1)
def mpci_factorial(z, prec): return mpci_gamma(z, prec, type=1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\req\req_install.py ===
import functools
import logging
import os
import shutil
import sys
import uuid
import zipfile
from optparse import Values
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Union

from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.pyproject_hooks import BuildBackendHookCaller

from pip._internal.build_env import BuildEnvironment, NoOpBuildEnvironment
from pip._internal.exceptions import InstallationError, PreviousBuildDirError
from pip._internal.locations import get_scheme
from pip._internal.metadata import (
    BaseDistribution,
    get_default_environment,
    get_directory_distribution,
    get_wheel_distribution,
)
from pip._internal.metadata.base import FilesystemWheel
from pip._internal.models.direct_url import DirectUrl
from pip._internal.models.link import Link
from pip._internal.operations.build.metadata import generate_metadata
from pip._internal.operations.build.metadata_editable import generate_editable_metadata
from pip._internal.operations.build.metadata_legacy import (
    generate_metadata as generate_metadata_legacy,
)
from pip._internal.operations.install.editable_legacy import (
    install_editable as install_editable_legacy,
)
from pip._internal.operations.install.wheel import install_wheel
from pip._internal.pyproject import load_pyproject_toml, make_pyproject_path
from pip._internal.req.req_uninstall import UninstallPathSet
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
    ConfiguredBuildBackendHookCaller,
    ask_path_exists,
    backup_dir,
    display_path,
    hide_url,
    is_installable_dir,
    redact_auth_from_requirement,
    redact_auth_from_url,
)
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
from pip._internal.utils.unpacking import unpack_file
from pip._internal.utils.virtualenv import running_under_virtualenv
from pip._internal.vcs import vcs

logger = logging.getLogger(__name__)


class InstallRequirement:
    """
    Represents something that may be installed later on, may have information
    about where to fetch the relevant requirement and also contains logic for
    installing the said requirement.
    """

    def __init__(
        self,
        req: Optional[Requirement],
        comes_from: Optional[Union[str, "InstallRequirement"]],
        editable: bool = False,
        link: Optional[Link] = None,
        markers: Optional[Marker] = None,
        use_pep517: Optional[bool] = None,
        isolated: bool = False,
        *,
        global_options: Optional[List[str]] = None,
        hash_options: Optional[Dict[str, List[str]]] = None,
        config_settings: Optional[Dict[str, Union[str, List[str]]]] = None,
        constraint: bool = False,
        extras: Collection[str] = (),
        user_supplied: bool = False,
        permit_editable_wheels: bool = False,
    ) -> None:
        assert req is None or isinstance(req, Requirement), req
        self.req = req
        self.comes_from = comes_from
        self.constraint = constraint
        self.editable = editable
        self.permit_editable_wheels = permit_editable_wheels

        # source_dir is the local directory where the linked requirement is
        # located, or unpacked. In case unpacking is needed, creating and
        # populating source_dir is done by the RequirementPreparer. Note this
        # is not necessarily the directory where pyproject.toml or setup.py is
        # located - that one is obtained via unpacked_source_directory.
        self.source_dir: Optional[str] = None
        if self.editable:
            assert link
            if link.is_file:
                self.source_dir = os.path.normpath(os.path.abspath(link.file_path))

        # original_link is the direct URL that was provided by the user for the
        # requirement, either directly or via a constraints file.
        if link is None and req and req.url:
            # PEP 508 URL requirement
            link = Link(req.url)
        self.link = self.original_link = link

        # When this InstallRequirement is a wheel obtained from the cache of locally
        # built wheels, this is the source link corresponding to the cache entry, which
        # was used to download and build the cached wheel.
        self.cached_wheel_source_link: Optional[Link] = None

        # Information about the location of the artifact that was downloaded . This
        # property is guaranteed to be set in resolver results.
        self.download_info: Optional[DirectUrl] = None

        # Path to any downloaded or already-existing package.
        self.local_file_path: Optional[str] = None
        if self.link and self.link.is_file:
            self.local_file_path = self.link.file_path

        if extras:
            self.extras = extras
        elif req:
            self.extras = req.extras
        else:
            self.extras = set()
        if markers is None and req:
            markers = req.marker
        self.markers = markers

        # This holds the Distribution object if this requirement is already installed.
        self.satisfied_by: Optional[BaseDistribution] = None
        # Whether the installation process should try to uninstall an existing
        # distribution before installing this requirement.
        self.should_reinstall = False
        # Temporary build location
        self._temp_build_dir: Optional[TempDirectory] = None
        # Set to True after successful installation
        self.install_succeeded: Optional[bool] = None
        # Supplied options
        self.global_options = global_options if global_options else []
        self.hash_options = hash_options if hash_options else {}
        self.config_settings = config_settings
        # Set to True after successful preparation of this requirement
        self.prepared = False
        # User supplied requirement are explicitly requested for installation
        # by the user via CLI arguments or requirements files, as opposed to,
        # e.g. dependencies, extras or constraints.
        self.user_supplied = user_supplied

        self.isolated = isolated
        self.build_env: BuildEnvironment = NoOpBuildEnvironment()

        # For PEP 517, the directory where we request the project metadata
        # gets stored. We need this to pass to build_wheel, so the backend
        # can ensure that the wheel matches the metadata (see the PEP for
        # details).
        self.metadata_directory: Optional[str] = None

        # The static build requirements (from pyproject.toml)
        self.pyproject_requires: Optional[List[str]] = None

        # Build requirements that we will check are available
        self.requirements_to_check: List[str] = []

        # The PEP 517 backend we should use to build the project
        self.pep517_backend: Optional[BuildBackendHookCaller] = None

        # Are we using PEP 517 for this requirement?
        # After pyproject.toml has been loaded, the only valid values are True
        # and False. Before loading, None is valid (meaning "use the default").
        # Setting an explicit value before loading pyproject.toml is supported,
        # but after loading this flag should be treated as read only.
        self.use_pep517 = use_pep517

        # If config settings are provided, enforce PEP 517.
        if self.config_settings:
            if self.use_pep517 is False:
                logger.warning(
                    "--no-use-pep517 ignored for %s "
                    "because --config-settings are specified.",
                    self,
                )
            self.use_pep517 = True

        # This requirement needs more preparation before it can be built
        self.needs_more_preparation = False

        # This requirement needs to be unpacked before it can be installed.
        self._archive_source: Optional[Path] = None

    def __str__(self) -> str:
        if self.req:
            s = redact_auth_from_requirement(self.req)
            if self.link:
                s += f" from {redact_auth_from_url(self.link.url)}"
        elif self.link:
            s = redact_auth_from_url(self.link.url)
        else:
            s = "<InstallRequirement>"
        if self.satisfied_by is not None:
            if self.satisfied_by.location is not None:
                location = display_path(self.satisfied_by.location)
            else:
                location = "<memory>"
            s += f" in {location}"
        if self.comes_from:
            if isinstance(self.comes_from, str):
                comes_from: Optional[str] = self.comes_from
            else:
                comes_from = self.comes_from.from_path()
            if comes_from:
                s += f" (from {comes_from})"
        return s

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} object: "
            f"{str(self)} editable={self.editable!r}>"
        )

    def format_debug(self) -> str:
        """An un-tested helper for getting state, for debugging."""
        attributes = vars(self)
        names = sorted(attributes)

        state = (f"{attr}={attributes[attr]!r}" for attr in sorted(names))
        return "<{name} object: {{{state}}}>".format(
            name=self.__class__.__name__,
            state=", ".join(state),
        )

    # Things that are valid for all kinds of requirements?
    @property
    def name(self) -> Optional[str]:
        if self.req is None:
            return None
        return self.req.name

    @functools.cached_property
    def supports_pyproject_editable(self) -> bool:
        if not self.use_pep517:
            return False
        assert self.pep517_backend
        with self.build_env:
            runner = runner_with_spinner_message(
                "Checking if build backend supports build_editable"
            )
            with self.pep517_backend.subprocess_runner(runner):
                return "build_editable" in self.pep517_backend._supported_features()

    @property
    def specifier(self) -> SpecifierSet:
        assert self.req is not None
        return self.req.specifier

    @property
    def is_direct(self) -> bool:
        """Whether this requirement was specified as a direct URL."""
        return self.original_link is not None

    @property
    def is_pinned(self) -> bool:
        """Return whether I am pinned to an exact version.

        For example, some-package==1.2 is pinned; some-package>1.2 is not.
        """
        assert self.req is not None
        specifiers = self.req.specifier
        return len(specifiers) == 1 and next(iter(specifiers)).operator in {"==", "==="}

    def match_markers(self, extras_requested: Optional[Iterable[str]] = None) -> bool:
        if not extras_requested:
            # Provide an extra to safely evaluate the markers
            # without matching any extra
            extras_requested = ("",)
        if self.markers is not None:
            return any(
                self.markers.evaluate({"extra": extra}) for extra in extras_requested
            )
        else:
            return True

    @property
    def has_hash_options(self) -> bool:
        """Return whether any known-good hashes are specified as options.

        These activate --require-hashes mode; hashes specified as part of a
        URL do not.

        """
        return bool(self.hash_options)

    def hashes(self, trust_internet: bool = True) -> Hashes:
        """Return a hash-comparer that considers my option- and URL-based
        hashes to be known-good.

        Hashes in URLs--ones embedded in the requirements file, not ones
        downloaded from an index server--are almost peers with ones from
        flags. They satisfy --require-hashes (whether it was implicitly or
        explicitly activated) but do not activate it. md5 and sha224 are not
        allowed in flags, which should nudge people toward good algos. We
        always OR all hashes together, even ones from URLs.

        :param trust_internet: Whether to trust URL-based (#md5=...) hashes
            downloaded from the internet, as by populate_link()

        """
        good_hashes = self.hash_options.copy()
        if trust_internet:
            link = self.link
        elif self.is_direct and self.user_supplied:
            link = self.original_link
        else:
            link = None
        if link and link.hash:
            assert link.hash_name is not None
            good_hashes.setdefault(link.hash_name, []).append(link.hash)
        return Hashes(good_hashes)

    def from_path(self) -> Optional[str]:
        """Format a nice indicator to show where this "comes from" """
        if self.req is None:
            return None
        s = str(self.req)
        if self.comes_from:
            comes_from: Optional[str]
            if isinstance(self.comes_from, str):
                comes_from = self.comes_from
            else:
                comes_from = self.comes_from.from_path()
            if comes_from:
                s += "->" + comes_from
        return s

    def ensure_build_location(
        self, build_dir: str, autodelete: bool, parallel_builds: bool
    ) -> str:
        assert build_dir is not None
        if self._temp_build_dir is not None:
            assert self._temp_build_dir.path
            return self._temp_build_dir.path
        if self.req is None:
            # Some systems have /tmp as a symlink which confuses custom
            # builds (such as numpy). Thus, we ensure that the real path
            # is returned.
            self._temp_build_dir = TempDirectory(
                kind=tempdir_kinds.REQ_BUILD, globally_managed=True
            )

            return self._temp_build_dir.path

        # This is the only remaining place where we manually determine the path
        # for the temporary directory. It is only needed for editables where
        # it is the value of the --src option.

        # When parallel builds are enabled, add a UUID to the build directory
        # name so multiple builds do not interfere with each other.
        dir_name: str = canonicalize_name(self.req.name)
        if parallel_builds:
            dir_name = f"{dir_name}_{uuid.uuid4().hex}"

        # FIXME: Is there a better place to create the build_dir? (hg and bzr
        # need this)
        if not os.path.exists(build_dir):
            logger.debug("Creating directory %s", build_dir)
            os.makedirs(build_dir)
        actual_build_dir = os.path.join(build_dir, dir_name)
        # `None` indicates that we respect the globally-configured deletion
        # settings, which is what we actually want when auto-deleting.
        delete_arg = None if autodelete else False
        return TempDirectory(
            path=actual_build_dir,
            delete=delete_arg,
            kind=tempdir_kinds.REQ_BUILD,
            globally_managed=True,
        ).path

    def _set_requirement(self) -> None:
        """Set requirement after generating metadata."""
        assert self.req is None
        assert self.metadata is not None
        assert self.source_dir is not None

        # Construct a Requirement object from the generated metadata
        if isinstance(parse_version(self.metadata["Version"]), Version):
            op = "=="
        else:
            op = "==="

        self.req = get_requirement(
            "".join(
                [
                    self.metadata["Name"],
                    op,
                    self.metadata["Version"],
                ]
            )
        )

    def warn_on_mismatching_name(self) -> None:
        assert self.req is not None
        metadata_name = canonicalize_name(self.metadata["Name"])
        if canonicalize_name(self.req.name) == metadata_name:
            # Everything is fine.
            return

        # If we're here, there's a mismatch. Log a warning about it.
        logger.warning(
            "Generating metadata for package %s "
            "produced metadata for project name %s. Fix your "
            "#egg=%s fragments.",
            self.name,
            metadata_name,
            self.name,
        )
        self.req = get_requirement(metadata_name)

    def check_if_exists(self, use_user_site: bool) -> None:
        """Find an installed distribution that satisfies or conflicts
        with this requirement, and set self.satisfied_by or
        self.should_reinstall appropriately.
        """
        if self.req is None:
            return
        existing_dist = get_default_environment().get_distribution(self.req.name)
        if not existing_dist:
            return

        version_compatible = self.req.specifier.contains(
            existing_dist.version,
            prereleases=True,
        )
        if not version_compatible:
            self.satisfied_by = None
            if use_user_site:
                if existing_dist.in_usersite:
                    self.should_reinstall = True
                elif running_under_virtualenv() and existing_dist.in_site_packages:
                    raise InstallationError(
                        f"Will not install to the user site because it will "
                        f"lack sys.path precedence to {existing_dist.raw_name} "
                        f"in {existing_dist.location}"
                    )
            else:
                self.should_reinstall = True
        else:
            if self.editable:
                self.should_reinstall = True
                # when installing editables, nothing pre-existing should ever
                # satisfy
                self.satisfied_by = None
            else:
                self.satisfied_by = existing_dist

    # Things valid for wheels
    @property
    def is_wheel(self) -> bool:
        if not self.link:
            return False
        return self.link.is_wheel

    @property
    def is_wheel_from_cache(self) -> bool:
        # When True, it means that this InstallRequirement is a local wheel file in the
        # cache of locally built wheels.
        return self.cached_wheel_source_link is not None

    # Things valid for sdists
    @property
    def unpacked_source_directory(self) -> str:
        assert self.source_dir, f"No source dir for {self}"
        return os.path.join(
            self.source_dir, self.link and self.link.subdirectory_fragment or ""
        )

    @property
    def setup_py_path(self) -> str:
        assert self.source_dir, f"No source dir for {self}"
        setup_py = os.path.join(self.unpacked_source_directory, "setup.py")

        return setup_py

    @property
    def setup_cfg_path(self) -> str:
        assert self.source_dir, f"No source dir for {self}"
        setup_cfg = os.path.join(self.unpacked_source_directory, "setup.cfg")

        return setup_cfg

    @property
    def pyproject_toml_path(self) -> str:
        assert self.source_dir, f"No source dir for {self}"
        return make_pyproject_path(self.unpacked_source_directory)

    def load_pyproject_toml(self) -> None:
        """Load the pyproject.toml file.

        After calling this routine, all of the attributes related to PEP 517
        processing for this requirement have been set. In particular, the
        use_pep517 attribute can be used to determine whether we should
        follow the PEP 517 or legacy (setup.py) code path.
        """
        pyproject_toml_data = load_pyproject_toml(
            self.use_pep517, self.pyproject_toml_path, self.setup_py_path, str(self)
        )

        if pyproject_toml_data is None:
            assert not self.config_settings
            self.use_pep517 = False
            return

        self.use_pep517 = True
        requires, backend, check, backend_path = pyproject_toml_data
        self.requirements_to_check = check
        self.pyproject_requires = requires
        self.pep517_backend = ConfiguredBuildBackendHookCaller(
            self,
            self.unpacked_source_directory,
            backend,
            backend_path=backend_path,
        )

    def isolated_editable_sanity_check(self) -> None:
        """Check that an editable requirement if valid for use with PEP 517/518.

        This verifies that an editable that has a pyproject.toml either supports PEP 660
        or as a setup.py or a setup.cfg
        """
        if (
            self.editable
            and self.use_pep517
            and not self.supports_pyproject_editable
            and not os.path.isfile(self.setup_py_path)
            and not os.path.isfile(self.setup_cfg_path)
        ):
            raise InstallationError(
                f"Project {self} has a 'pyproject.toml' and its build "
                f"backend is missing the 'build_editable' hook. Since it does not "
                f"have a 'setup.py' nor a 'setup.cfg', "
                f"it cannot be installed in editable mode. "
                f"Consider using a build backend that supports PEP 660."
            )

    def prepare_metadata(self) -> None:
        """Ensure that project metadata is available.

        Under PEP 517 and PEP 660, call the backend hook to prepare the metadata.
        Under legacy processing, call setup.py egg-info.
        """
        assert self.source_dir, f"No source dir for {self}"
        details = self.name or f"from {self.link}"

        if self.use_pep517:
            assert self.pep517_backend is not None
            if (
                self.editable
                and self.permit_editable_wheels
                and self.supports_pyproject_editable
            ):
                self.metadata_directory = generate_editable_metadata(
                    build_env=self.build_env,
                    backend=self.pep517_backend,
                    details=details,
                )
            else:
                self.metadata_directory = generate_metadata(
                    build_env=self.build_env,
                    backend=self.pep517_backend,
                    details=details,
                )
        else:
            self.metadata_directory = generate_metadata_legacy(
                build_env=self.build_env,
                setup_py_path=self.setup_py_path,
                source_dir=self.unpacked_source_directory,
                isolated=self.isolated,
                details=details,
            )

        # Act on the newly generated metadata, based on the name and version.
        if not self.name:
            self._set_requirement()
        else:
            self.warn_on_mismatching_name()

        self.assert_source_matches_version()

    @property
    def metadata(self) -> Any:
        if not hasattr(self, "_metadata"):
            self._metadata = self.get_dist().metadata

        return self._metadata

    def get_dist(self) -> BaseDistribution:
        if self.metadata_directory:
            return get_directory_distribution(self.metadata_directory)
        elif self.local_file_path and self.is_wheel:
            assert self.req is not None
            return get_wheel_distribution(
                FilesystemWheel(self.local_file_path),
                canonicalize_name(self.req.name),
            )
        raise AssertionError(
            f"InstallRequirement {self} has no metadata directory and no wheel: "
            f"can't make a distribution."
        )

    def assert_source_matches_version(self) -> None:
        assert self.source_dir, f"No source dir for {self}"
        version = self.metadata["version"]
        if self.req and self.req.specifier and version not in self.req.specifier:
            logger.warning(
                "Requested %s, but installing version %s",
                self,
                version,
            )
        else:
            logger.debug(
                "Source in %s has version %s, which satisfies requirement %s",
                display_path(self.source_dir),
                version,
                self,
            )

    # For both source distributions and editables
    def ensure_has_source_dir(
        self,
        parent_dir: str,
        autodelete: bool = False,
        parallel_builds: bool = False,
    ) -> None:
        """Ensure that a source_dir is set.

        This will create a temporary build dir if the name of the requirement
        isn't known yet.

        :param parent_dir: The ideal pip parent_dir for the source_dir.
            Generally src_dir for editables and build_dir for sdists.
        :return: self.source_dir
        """
        if self.source_dir is None:
            self.source_dir = self.ensure_build_location(
                parent_dir,
                autodelete=autodelete,
                parallel_builds=parallel_builds,
            )

    def needs_unpacked_archive(self, archive_source: Path) -> None:
        assert self._archive_source is None
        self._archive_source = archive_source

    def ensure_pristine_source_checkout(self) -> None:
        """Ensure the source directory has not yet been built in."""
        assert self.source_dir is not None
        if self._archive_source is not None:
            unpack_file(str(self._archive_source), self.source_dir)
        elif is_installable_dir(self.source_dir):
            # If a checkout exists, it's unwise to keep going.
            # version inconsistencies are logged later, but do not fail
            # the installation.
            raise PreviousBuildDirError(
                f"pip can't proceed with requirements '{self}' due to a "
                f"pre-existing build directory ({self.source_dir}). This is likely "
                "due to a previous installation that failed . pip is "
                "being responsible and not assuming it can delete this. "
                "Please delete it and try again."
            )

    # For editable installations
    def update_editable(self) -> None:
        if not self.link:
            logger.debug(
                "Cannot update repository at %s; repository location is unknown",
                self.source_dir,
            )
            return
        assert self.editable
        assert self.source_dir
        if self.link.scheme == "file":
            # Static paths don't get updated
            return
        vcs_backend = vcs.get_backend_for_scheme(self.link.scheme)
        # Editable requirements are validated in Requirement constructors.
        # So here, if it's neither a path nor a valid VCS URL, it's a bug.
        assert vcs_backend, f"Unsupported VCS URL {self.link.url}"
        hidden_url = hide_url(self.link.url)
        vcs_backend.obtain(self.source_dir, url=hidden_url, verbosity=0)

    # Top-level Actions
    def uninstall(
        self, auto_confirm: bool = False, verbose: bool = False
    ) -> Optional[UninstallPathSet]:
        """
        Uninstall the distribution currently satisfying this requirement.

        Prompts before removing or modifying files unless
        ``auto_confirm`` is True.

        Refuses to delete or modify files outside of ``sys.prefix`` -
        thus uninstallation within a virtual environment can only
        modify that virtual environment, even if the virtualenv is
        linked to global site-packages.

        """
        assert self.req
        dist = get_default_environment().get_distribution(self.req.name)
        if not dist:
            logger.warning("Skipping %s as it is not installed.", self.name)
            return None
        logger.info("Found existing installation: %s", dist)

        uninstalled_pathset = UninstallPathSet.from_dist(dist)
        uninstalled_pathset.remove(auto_confirm, verbose)
        return uninstalled_pathset

    def _get_archive_name(self, path: str, parentdir: str, rootdir: str) -> str:
        def _clean_zip_name(name: str, prefix: str) -> str:
            assert name.startswith(
                prefix + os.path.sep
            ), f"name {name!r} doesn't start with prefix {prefix!r}"
            name = name[len(prefix) + 1 :]
            name = name.replace(os.path.sep, "/")
            return name

        assert self.req is not None
        path = os.path.join(parentdir, path)
        name = _clean_zip_name(path, rootdir)
        return self.req.name + "/" + name

    def archive(self, build_dir: Optional[str]) -> None:
        """Saves archive to provided build_dir.

        Used for saving downloaded VCS requirements as part of `pip download`.
        """
        assert self.source_dir
        if build_dir is None:
            return

        create_archive = True
        archive_name = "{}-{}.zip".format(self.name, self.metadata["version"])
        archive_path = os.path.join(build_dir, archive_name)

        if os.path.exists(archive_path):
            response = ask_path_exists(
                f"The file {display_path(archive_path)} exists. (i)gnore, (w)ipe, "
                "(b)ackup, (a)bort ",
                ("i", "w", "b", "a"),
            )
            if response == "i":
                create_archive = False
            elif response == "w":
                logger.warning("Deleting %s", display_path(archive_path))
                os.remove(archive_path)
            elif response == "b":
                dest_file = backup_dir(archive_path)
                logger.warning(
                    "Backing up %s to %s",
                    display_path(archive_path),
                    display_path(dest_file),
                )
                shutil.move(archive_path, dest_file)
            elif response == "a":
                sys.exit(-1)

        if not create_archive:
            return

        zip_output = zipfile.ZipFile(
            archive_path,
            "w",
            zipfile.ZIP_DEFLATED,
            allowZip64=True,
        )
        with zip_output:
            dir = os.path.normcase(os.path.abspath(self.unpacked_source_directory))
            for dirpath, dirnames, filenames in os.walk(dir):
                for dirname in dirnames:
                    dir_arcname = self._get_archive_name(
                        dirname,
                        parentdir=dirpath,
                        rootdir=dir,
                    )
                    zipdir = zipfile.ZipInfo(dir_arcname + "/")
                    zipdir.external_attr = 0x1ED << 16  # 0o755
                    zip_output.writestr(zipdir, "")
                for filename in filenames:
                    file_arcname = self._get_archive_name(
                        filename,
                        parentdir=dirpath,
                        rootdir=dir,
                    )
                    filename = os.path.join(dirpath, filename)
                    zip_output.write(filename, file_arcname)

        logger.info("Saved %s", display_path(archive_path))

    def install(
        self,
        global_options: Optional[Sequence[str]] = None,
        root: Optional[str] = None,
        home: Optional[str] = None,
        prefix: Optional[str] = None,
        warn_script_location: bool = True,
        use_user_site: bool = False,
        pycompile: bool = True,
    ) -> None:
        assert self.req is not None
        scheme = get_scheme(
            self.req.name,
            user=use_user_site,
            home=home,
            root=root,
            isolated=self.isolated,
            prefix=prefix,
        )

        if self.editable and not self.is_wheel:
            deprecated(
                reason=(
                    f"Legacy editable install of {self} (setup.py develop) "
                    "is deprecated."
                ),
                replacement=(
                    "to add a pyproject.toml or enable --use-pep517, "
                    "and use setuptools >= 64. "
                    "If the resulting installation is not behaving as expected, "
                    "try using --config-settings editable_mode=compat. "
                    "Please consult the setuptools documentation for more information"
                ),
                gone_in="25.3",
                issue=11457,
            )
            if self.config_settings:
                logger.warning(
                    "--config-settings ignored for legacy editable install of %s. "
                    "Consider upgrading to a version of setuptools "
                    "that supports PEP 660 (>= 64).",
                    self,
                )
            install_editable_legacy(
                global_options=global_options if global_options is not None else [],
                prefix=prefix,
                home=home,
                use_user_site=use_user_site,
                name=self.req.name,
                setup_py_path=self.setup_py_path,
                isolated=self.isolated,
                build_env=self.build_env,
                unpacked_source_directory=self.unpacked_source_directory,
            )
            self.install_succeeded = True
            return

        assert self.is_wheel
        assert self.local_file_path

        install_wheel(
            self.req.name,
            self.local_file_path,
            scheme=scheme,
            req_description=str(self.req),
            pycompile=pycompile,
            warn_script_location=warn_script_location,
            direct_url=self.download_info if self.is_direct else None,
            requested=self.user_supplied,
        )
        self.install_succeeded = True


def check_invalid_constraint_type(req: InstallRequirement) -> str:
    # Check for unsupported forms
    problem = ""
    if not req.name:
        problem = "Unnamed requirements are not allowed as constraints"
    elif req.editable:
        problem = "Editable requirements are not allowed as constraints"
    elif req.extras:
        problem = "Constraints cannot have extras"

    if problem:
        deprecated(
            reason=(
                "Constraints are only allowed to take the form of a package "
                "name and a version specifier. Other forms were originally "
                "permitted as an accident of the implementation, but were "
                "undocumented. The new implementation of the resolver no "
                "longer supports these forms."
            ),
            replacement="replacing the constraint with a requirement",
            # No plan yet for when the new resolver becomes default
            gone_in=None,
            issue=8210,
        )

    return problem


def _has_option(options: Values, reqs: List[InstallRequirement], option: str) -> bool:
    if getattr(options, option, None):
        return True
    for req in reqs:
        if getattr(req, option, None):
            return True
    return False


def check_legacy_setup_py_options(
    options: Values,
    reqs: List[InstallRequirement],
) -> None:
    has_build_options = _has_option(options, reqs, "build_options")
    has_global_options = _has_option(options, reqs, "global_options")
    if has_build_options or has_global_options:
        deprecated(
            reason="--build-option and --global-option are deprecated.",
            issue=11859,
            replacement="to use --config-settings",
            gone_in="25.3",
        )
        logger.warning(
            "Implying --no-binary=:all: due to the presence of "
            "--build-option / --global-option. "
        )
        options.format_control.disallow_binaries()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model_phi.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

import numpy as np
from dynamo_onnx_helper import DynamoOnnxHelper
from fusion_base import Fusion
from fusion_options import AttentionOpType, FusionOptions
from fusion_skiplayernorm import FusionBiasSkipLayerNormalization, FusionSkipLayerNormalization
from fusion_utils import NumpyHelper
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class ProcessGemmWFunc:
    def __call__(self, x):
        return np.transpose(x, (1, 0))


class ProcessMatMulQFunc:
    def __call__(self, x):
        return np.transpose(np.split(x, 3, 0)[0], (1, 0))


class ProcessMatMulKFunc:
    def __call__(self, x):
        return np.transpose(np.split(x, 3, 0)[1], (1, 0))


class ProcessMatMulVFunc:
    def __call__(self, x):
        return np.transpose(np.split(x, 3, 0)[2], (1, 0))


class ProcessBiasQFunc:
    def __call__(self, x):
        x = np.split(x, 3, -1)[0]
        return x


class ProcessBiasKFunc:
    def __call__(self, x):
        x = np.split(x, 3, -1)[1]
        return x


class ProcessBiasVFunc:
    def __call__(self, x):
        x = np.split(x, 3, -1)[2]
        return x


class ProcessRotCacheFunc:
    def __call__(self, x):
        # half rotary embedding
        assert len(x.shape) == 2
        if x.shape[1] == 32:
            return x[:, 0:16]
        return x


# TODO: move to a separate file
class Fission(Fusion):
    def __init__(
        self,
        model: OnnxModel,
        nodes_to_find: list[str],
    ):
        super().__init__(model, "DONOTUSE", nodes_to_find)

    def set_attention_op_type(self, attn_op_type: AttentionOpType):
        self.attn_op_type = attn_op_type

    def get_uname(self, layer_id, name):
        return name + "_" + str(layer_id)

    def get_edge_by_name(self, edges, name):
        for edge in edges:
            if edge == name or edge.endswith(name) or edge.startswith(name):
                return edge
        raise ValueError(f"Edge {name} not found")

    def get_input_by_name(self, node, name):
        return self.get_edge_by_name(node.input, name)

    def get_output_by_name(self, node, name):
        return self.get_edge_by_name(node.output, name)

    def process_initializer(self, initializer_name, functor, custom_name=None):
        i = self.model.get_initializer(initializer_name)
        i_np_array = NumpyHelper.to_array(i)
        processed_i_np_array = functor(i_np_array)
        new_tensor = helper.make_tensor(
            initializer_name + "_processed" if custom_name is None else custom_name,
            data_type=TensorProto.FLOAT,
            dims=processed_i_np_array.shape,
            vals=processed_i_np_array.flatten().tobytes(),
            raw=True,
        )
        self.model.add_initializer(new_tensor, self.this_graph_name)
        return new_tensor.name

    def add_fp32_value_info(self, name):
        new_value_info = self.model.graph().value_info.add()
        new_value_info.name = name
        new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT

    def add_int64_value_info(self, name):
        new_value_info = self.model.graph().value_info.add()
        new_value_info.name = name
        new_value_info.type.tensor_type.elem_type = TensorProto.INT64

    def replace_fp32_value_info(self, name, shape):
        for value_info in self.model.graph().value_info:
            if value_info.name == name:
                self.model.graph().value_info.remove(value_info)
                break
        new_value_info = helper.make_tensor_value_info(
            name,
            elem_type=TensorProto.FLOAT,
            shape=shape,
        )
        self.model.graph().value_info.extend([new_value_info])

    def set_unique_name_and_add_nodes(
        self, subgraph_nodes: list[NodeProto], layer_id: int, layer_known_edges_names: list[str]
    ):
        for new_node in subgraph_nodes:
            for i, name in enumerate(new_node.input):
                if name == "":
                    continue
                elif name not in layer_known_edges_names:
                    new_node.input[i] = self.get_uname(layer_id, name)
                    self.add_fp32_value_info(new_node.input[i])
            for i, name in enumerate(new_node.output):
                if name == "":
                    continue
                elif name not in layer_known_edges_names:
                    new_node.output[i] = self.get_uname(layer_id, name)
                    self.add_fp32_value_info(new_node.output[i])
            new_node.name = self.get_uname(layer_id, new_node.name)
            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

    def layernorm(self, inputs: list[str], outputs: list[str], prefix: str = ""):
        assert len(inputs) == 3
        assert len(outputs) == 1
        node = helper.make_node(
            "LayerNormalization",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "_LayerNormalization",
            epsilon=9.999999747378752e-06,
        )
        return [node]

    def gemm(self, inputs: list[str], outputs: list[str], prefix: str = ""):
        assert len(inputs) == 3
        assert len(outputs) == 1
        matmul = helper.make_node(
            "MatMul",
            inputs=[inputs[0], inputs[1]],
            outputs=[prefix + "matmul_out"],
            name=prefix + "MatMul",
        )
        add = helper.make_node(
            "Add",
            inputs=[prefix + "matmul_out", inputs[2]],
            outputs=outputs,
            name=prefix + "Bias",
        )
        return [matmul, add]

    def rotary(self, inputs: list[str], outputs: list[str], prefix: str = "", rot_dim=32, num_heads=32):
        assert len(inputs) == 4
        assert len(outputs) == 1
        node = helper.make_node(
            "RotaryEmbedding",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "RotaryEmbedding",
            domain="com.microsoft",
            rotary_embedding_dim=rot_dim,
            num_heads=num_heads,
        )
        return [node]

    def fastgelu(self, inputs: list[str], outputs: list[str], prefix: str = ""):
        assert len(inputs) == 1
        assert len(outputs) == 1
        node = helper.make_node(
            "FastGelu",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "FastGelu",
            domain="com.microsoft",
        )
        return [node]

    def add(self, inputs: list[str], outputs: list[str], prefix: str = ""):
        assert len(inputs) == 2
        assert len(outputs) == 1
        node = helper.make_node(
            "Add",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "Add",
        )
        return [node]

    def mha(self, inputs: list[str], outputs: list[str], prefix: str = "", num_heads=32):
        assert len(inputs) == 8
        assert len(outputs) == 3
        node = helper.make_node(
            "MultiHeadAttention",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "MultiHeadAttention",
            domain="com.microsoft",
            num_heads=num_heads,
            unidirectional=1,
        )
        return [node]

    def gqa(self, inputs: list[str], outputs: list[str], prefix: str = "", num_heads=32):
        assert len(inputs) == 7
        assert len(outputs) == 3
        node = helper.make_node(
            "GroupQueryAttention",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "GroupQueryAttention",
            domain="com.microsoft",
            num_heads=num_heads,
            kv_num_heads=num_heads,
        )
        return [node]

    def attention(self, inputs: list[str], outputs: list[str], prefix: str = "", num_heads=32):
        assert len(inputs) == 5
        assert len(outputs) == 2
        node = helper.make_node(
            "Attention",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "Attention",
            domain="com.microsoft",
            num_heads=num_heads,
            unidirectional=1,
            do_rotary=1,
            rotary_embedding_dim=32,
        )
        return [node]

    def paged_attn(
        self,
        inputs: list[str],
        outputs: list[str],
        prefix: str = "",
        num_heads=32,
        head_size=80,
        scale=0.11180339753627777,
    ):
        assert len(inputs) == 6
        assert len(outputs) == 1
        node = helper.make_node(
            "PagedAttention",
            inputs=inputs,
            outputs=outputs,
            name=prefix + "PagedAttention",
            domain="vllm.ort.ext",
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_size=head_size,
            scale=scale,
        )
        return [node]


class Phi2PreProcessor(DynamoOnnxHelper):
    def __init__(self, model: ModelProto, num_heads: int, hidden_size: int):
        super().__init__(model)
        self.num_hidden_layers = 32
        self.num_attention_heads = num_heads
        self.hidden_size = hidden_size

        self.func_name = "modeling_phi_PhiModel_model_1"

    def get_phi2_edge_dict(self) -> dict:
        edge_dict = {}
        edge_dict["lm_head_1"] = "logits"
        edge_dict["l_input_ids_"] = "input_ids"
        edge_dict["key_states"] = "past_key_0"
        edge_dict["value_states"] = "past_value_0"
        for i in range(1, self.num_hidden_layers, 1):
            edge_dict[f"key_states_{i}"] = f"past_key_{i}"
            edge_dict[f"value_states_{i}"] = f"past_value_{i}"
            edge_dict[f"model_layers_{i}_1"] = f"present_key_{i}"
            edge_dict[f"model_layers_{i}_1_1"] = f"present_value_{i}"

        outputs = [o.name for o in self.model.graph.output]
        if "model_layers_0_1_1" in outputs and "model_layers_0_1_2" in outputs:
            edge_dict["model_layers_0_1_1"] = "present_key_0"
            edge_dict["model_layers_0_1_2"] = "present_value_0"
        else:
            assert "model_layers_0_1" in outputs and "model_layers_0_1_1" in outputs
            edge_dict["model_layers_0_1"] = "present_key_0"
            edge_dict["model_layers_0_1_1"] = "present_value_0"
        return edge_dict

    def simplify_phi2_op_type(self):
        phi2_transformer_layer_name = "modeling_phi_PhiDecoderLayer_model_layers"
        for node in self.model.graph.node:
            index = node.op_type.find(phi2_transformer_layer_name)
            if index != -1:
                node.op_type = node.op_type[index:]

    def process_graph_io(self, attn_op_type: AttentionOpType):
        self.use_attn = attn_op_type == AttentionOpType.Attention
        self.use_vllm = attn_op_type == AttentionOpType.PagedAttention
        graph = self.model.graph
        new_inputs = []
        for vi in graph.input:
            if "input_ids" in vi.name:
                vi_iid = helper.make_tensor_value_info(
                    vi.name,
                    elem_type=TensorProto.INT32 if not self.use_vllm else TensorProto.INT64,
                    shape=["batch_size", "seq_len"],
                )
                vi_step = helper.make_tensor_value_info(
                    "step",
                    elem_type=TensorProto.INT64,
                    shape=[1],
                )
                vi_pid = helper.make_tensor_value_info(
                    "position_ids",
                    elem_type=TensorProto.INT64,
                    shape=["batch_size", "seq_len"],
                )
                vi_mask = helper.make_tensor_value_info(
                    "attention_mask",
                    elem_type=TensorProto.INT32,
                    shape=["batch_size", "seq_len"],
                )
                vi_meta = helper.make_tensor_value_info(
                    "input_metadata",
                    elem_type=TensorProto.INT64,
                    shape=[1],
                )
                (
                    new_inputs.extend([vi_iid, vi_step, vi_mask])
                    if not self.use_vllm
                    else new_inputs.extend([vi_iid, vi_pid, vi_meta])
                )
            if self.use_attn:
                if "past_key" in vi.name:
                    vi_cache = helper.make_tensor_value_info(
                        vi.name.replace("past_key", "past"),
                        elem_type=vi.type.tensor_type.elem_type,
                        shape=[
                            2,
                            "batch_size",
                            self.num_attention_heads,
                            "past_seq_len",
                            self.hidden_size // self.num_attention_heads,
                        ],
                    )
                    new_inputs.extend([vi_cache])
            elif self.use_vllm:
                if "past_key" in vi.name:
                    vi_cache = helper.make_tensor_value_info(
                        vi.name,
                        elem_type=vi.type.tensor_type.elem_type,
                        shape=["num_blocks", "num_heads", "head_size_x", "block_size", "block_x"],
                    )
                    new_inputs.extend([vi_cache])
                if "past_value" in vi.name:
                    vi_cache = helper.make_tensor_value_info(
                        vi.name,
                        elem_type=vi.type.tensor_type.elem_type,
                        shape=[
                            "num_blocks",
                            "num_heads",
                            "head_size",
                            "block_size",
                        ],
                    )
                    new_inputs.extend([vi_cache])
            else:
                if "past_key" in vi.name or "past_value" in vi.name:
                    vi_cache = helper.make_tensor_value_info(
                        vi.name,
                        elem_type=vi.type.tensor_type.elem_type,
                        shape=[
                            "batch_size",
                            self.num_attention_heads,
                            "past_seq_len",
                            self.hidden_size // self.num_attention_heads,
                        ],
                    )
                    new_inputs.extend([vi_cache])

        graph.ClearField("input")
        graph.input.extend(new_inputs)

        new_outputs = []
        for i, vi in enumerate(graph.output):
            if i == 0:
                new_outputs.extend([vi])
            else:
                if self.use_attn:
                    if "present_key" in vi.name:
                        vi_cache = helper.make_tensor_value_info(
                            vi.name.replace("present_key", "present"),
                            elem_type=vi.type.tensor_type.elem_type,
                            shape=[
                                2,
                                "batch_size",
                                self.num_attention_heads,
                                "total_seq_len",
                                self.hidden_size // self.num_attention_heads,
                            ],
                        )
                        new_outputs.extend([vi_cache])
                elif self.use_vllm:
                    pass
                else:
                    vi_cache = helper.make_tensor_value_info(
                        vi.name,
                        elem_type=vi.type.tensor_type.elem_type,
                        shape=[
                            "batch_size",
                            self.num_attention_heads,
                            "total_seq_len",
                            self.hidden_size // self.num_attention_heads,
                        ],
                    )
                    new_outputs.extend([vi_cache])

        graph.ClearField("output")
        graph.output.extend(new_outputs)

    def preprocess_onnx(self, attn_op_type: AttentionOpType):
        function_name = None
        for func in self.model.functions:
            if func.name.endswith(self.func_name):
                function_name = func.name
                break
        assert function_name is not None
        self.unroll_function(function_name)
        self.update_edges(self.get_phi2_edge_dict())
        self.simplify_phi2_op_type()
        self.remove_dropout_layer()
        if attn_op_type == AttentionOpType.PagedAttention:
            self.remove_lm_head_layer()
        self.process_graph_io(attn_op_type)


class FissionTransformerEmbeddingPhi(Fission):
    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, ["torch_nn_modules_sparse_Embedding_model_embed_tokens_1"])

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        logger.info("Optimizing %s...", node.name)

        assert len(node.input) == 2
        assert len(node.output) == 1

        input = node.input[0]
        output = node.output[0]

        embedding = self.get_input_by_name(node, "embed_tokens.weight")

        layer_known_edges_names = [input, output, embedding]

        subgraph_nodes = [
            helper.make_node(
                "Gather",
                inputs=[embedding, input],
                outputs=[output],
                name="Embedding_Gather",
            ),
        ]

        self.set_unique_name_and_add_nodes(subgraph_nodes, 0, layer_known_edges_names)
        self.nodes_to_remove.append(node)
        self.prune_graph = True


class FissionTransformerLayerNormPhi(Fission):
    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, ["torch_nn_modules_normalization_LayerNorm_model_final_layernorm_1"])

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        logger.info("Optimizing %s...", node.name)

        assert len(node.input) == 3
        assert len(node.output) == 1

        input = node.input[0]
        output = node.output[0]

        ln_weight = self.get_input_by_name(node, "final_layernorm.weight")
        ln_bias = self.get_input_by_name(node, "final_layernorm.bias")

        layer_known_edges_names = [input, output, ln_weight, ln_bias]

        subgraph_nodes = []
        subgraph_nodes.extend(self.layernorm([input, ln_weight, ln_bias], [output], "Final"))

        self.set_unique_name_and_add_nodes(subgraph_nodes, 99, layer_known_edges_names)

        self.replace_fp32_value_info(input, ["batch_size", "seq_len", "hidden_size"])
        self.replace_fp32_value_info(output, ["batch_size", "seq_len", "hidden_size"])

        self.nodes_to_remove.append(node)
        self.prune_graph = True


class FissionTransformerCausalLMHeadPhi(Fission):
    def __init__(
        self,
        model: OnnxModel,
    ):
        super().__init__(model, ["torch_nn_modules_linear_Linear_lm_head_1"])

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        logger.info("Optimizing %s...", node.name)

        assert len(node.input) == 5
        assert len(node.output) == 1

        input = node.input[2]
        output = node.output[0]

        fc_weight = self.process_initializer(self.get_input_by_name(node, "lm_head.weight"), ProcessGemmWFunc())
        fc_bias = self.get_input_by_name(node, "lm_head.bias")

        layer_known_edges_names = [input, output, fc_weight, fc_bias]

        subgraph_nodes = []
        subgraph_nodes.extend(self.gemm([input, fc_weight, fc_bias], [output], "LMHead_"))

        self.set_unique_name_and_add_nodes(subgraph_nodes, 99, layer_known_edges_names)

        self.replace_fp32_value_info(input, ["batch_size", "seq_len", "hidden_size"])
        self.replace_fp32_value_info(output, ["batch_size", "seq_len", 51200])

        self.nodes_to_remove.append(node)
        self.prune_graph = True


class FissionTransformerBlockPhi(Fission):
    def __init__(
        self,
        model: OnnxModel,
        num_heads: int,
    ):
        self.num_heads = num_heads
        max_num_layers = 32
        self.func_to_layer_id = {}
        nodes_to_find = []
        for layer in range(max_num_layers):
            func_name = f"modeling_phi_PhiDecoderLayer_model_layers_{layer}_1"
            nodes_to_find.append(func_name)
            self.func_to_layer_id[func_name] = layer

        super().__init__(model, nodes_to_find)

    def get_layer_id(self, node):
        return self.func_to_layer_id[node.op_type]

    def get_gqa_aux_nodes(self):
        gqa_aux_nodes = [
            helper.make_node(
                "Cast",
                inputs=["attention_mask"],
                outputs=["mask_int64"],
                name="Cast_gqa_aux_0",
                to=TensorProto.INT64,
            ),
            helper.make_node(
                "ReduceSum",
                inputs=["mask_int64", "one"],
                outputs=["mask_row_sums"],
                name="ReduceSum_gqa_aux",
            ),
            helper.make_node(
                "Sub",
                inputs=["mask_row_sums", "one"],
                outputs=["seqlens_k_int64"],
                name="Sub_gqa_aux",
            ),
            helper.make_node(
                "Cast",
                inputs=["seqlens_k_int64"],
                outputs=["seqlens_k"],
                name="Cast_gqa_aux_1",
                to=TensorProto.INT32,
            ),
            helper.make_node("Shape", inputs=["mask_int64"], outputs=["mask_shape"], name="Shape_gqa_aux_0"),
            helper.make_node(
                "Gather",
                inputs=["mask_shape", "one"],
                outputs=["total_seq_len_int64"],
                name="Gather_gqa_aux_0",
                axis=0,
            ),
            helper.make_node(
                "Cast",
                inputs=["total_seq_len_int64"],
                outputs=["total_sequence_length"],
                name="Cast_gqa_aux_2",
                to=TensorProto.INT32,
            ),
        ]
        return gqa_aux_nodes

    def pack_qkv_gemm(self, q_w, k_w, v_w, q_b, k_b, v_b, weight_name, bias_name):
        q_weight = self.model.get_initializer(q_w)
        k_weight = self.model.get_initializer(k_w)
        v_weight = self.model.get_initializer(v_w)
        qw = np.transpose(NumpyHelper.to_array(q_weight), (1, 0))
        kw = np.transpose(NumpyHelper.to_array(k_weight), (1, 0))
        vw = np.transpose(NumpyHelper.to_array(v_weight), (1, 0))
        qkv_weight = np.stack((qw, kw, vw), axis=1)

        q_bias = self.model.get_initializer(q_b)
        k_bias = self.model.get_initializer(k_b)
        v_bias = self.model.get_initializer(v_b)
        qb = NumpyHelper.to_array(q_bias)
        kb = NumpyHelper.to_array(k_bias)
        vb = NumpyHelper.to_array(v_bias)
        qkv_bias = np.stack((qb, kb, vb), axis=0)

        hidden_size = qkv_weight.shape[0]

        weight = helper.make_tensor(
            weight_name,
            data_type=TensorProto.FLOAT,
            dims=[hidden_size, hidden_size * 3],
            vals=qkv_weight.flatten().tobytes(),
            raw=True,
        )
        self.model.add_initializer(weight, self.this_graph_name)

        bias = helper.make_tensor(
            bias_name,
            data_type=TensorProto.FLOAT,
            dims=[hidden_size * 3],
            vals=qkv_bias.flatten().tobytes(),
            raw=True,
        )
        self.model.add_initializer(bias, self.this_graph_name)

        self.add_fp32_value_info(weight.name)
        self.add_fp32_value_info(bias.name)

        return weight_name, bias_name

    def fuse(
        self,
        node,
        input_name_to_nodes,
        output_name_to_node,
    ):
        logger.info("Optimizing %s...", node.name)

        logger.info(f"AttentionOpType: {self.attn_op_type}")

        layer_id = self.get_layer_id(node)

        i_hidden_states = node.input[0]
        i_key_cache = self.get_input_by_name(node, "past_key")
        i_value_cache = self.get_input_by_name(node, "past_value")

        o_hidden_states = node.output[-1]
        o_key_cache = self.get_output_by_name(node, "present_key")
        o_value_cache = self.get_output_by_name(node, "present_value")

        ln_weight = self.get_input_by_name(node, "input_layernorm.weight")
        ln_bias = self.get_input_by_name(node, "input_layernorm.bias")

        attn_q_weight, attn_q_bias, attn_k_weight, attn_k_bias, attn_v_weight, attn_v_bias = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        attn_qkv_weight, attn_qkv_bias = None, None
        cos_cache, sin_cache = None, None

        if self.attn_op_type != AttentionOpType.Attention:
            attn_q_weight = self.process_initializer(
                self.get_input_by_name(node, "self_attn.q_proj.weight"), ProcessGemmWFunc()
            )
            attn_k_weight = self.process_initializer(
                self.get_input_by_name(node, "self_attn.k_proj.weight"), ProcessGemmWFunc()
            )
            attn_v_weight = self.process_initializer(
                self.get_input_by_name(node, "self_attn.v_proj.weight"), ProcessGemmWFunc()
            )
            attn_q_bias = self.get_input_by_name(node, "self_attn.q_proj.bias")
            attn_k_bias = self.get_input_by_name(node, "self_attn.k_proj.bias")
            attn_v_bias = self.get_input_by_name(node, "self_attn.v_proj.bias")

            cos_cache = self.process_initializer(
                self.get_input_by_name(node, "rotary_emb.cos_cached"), ProcessRotCacheFunc()
            )
            sin_cache = self.process_initializer(
                self.get_input_by_name(node, "rotary_emb.sin_cached"), ProcessRotCacheFunc()
            )
        else:
            attn_qkv_weight, attn_qkv_bias = self.pack_qkv_gemm(
                self.get_input_by_name(node, "self_attn.q_proj.weight"),
                self.get_input_by_name(node, "self_attn.k_proj.weight"),
                self.get_input_by_name(node, "self_attn.v_proj.weight"),
                self.get_input_by_name(node, "self_attn.q_proj.bias"),
                self.get_input_by_name(node, "self_attn.k_proj.bias"),
                self.get_input_by_name(node, "self_attn.v_proj.bias"),
                self.get_uname(layer_id, "attn_qkv_weight"),
                self.get_uname(layer_id, "attn_qkv_bias"),
            )

        attn_out_weight = self.process_initializer(
            self.get_input_by_name(node, "self_attn.dense.weight"), ProcessGemmWFunc()
        )
        attn_out_bias = self.get_input_by_name(node, "self_attn.dense.bias")

        mlp_fc1_weight = self.process_initializer(self.get_input_by_name(node, "mlp.fc1.weight"), ProcessGemmWFunc())
        mlp_fc2_weight = self.process_initializer(self.get_input_by_name(node, "mlp.fc2.weight"), ProcessGemmWFunc())
        mlp_fc1_bias = self.get_input_by_name(node, "mlp.fc1.bias")
        mlp_fc2_bias = self.get_input_by_name(node, "mlp.fc2.bias")

        layer_known_edges_names = []
        layer_known_edges_names.extend([i_hidden_states, i_key_cache, i_value_cache])
        layer_known_edges_names.extend([o_hidden_states, o_key_cache, o_value_cache])
        layer_known_edges_names.extend([ln_weight, ln_bias])
        if self.attn_op_type != AttentionOpType.Attention:
            layer_known_edges_names.extend(
                [
                    attn_q_weight,
                    attn_q_bias,
                    attn_k_weight,
                    attn_k_bias,
                    attn_v_weight,
                    attn_v_bias,
                    cos_cache,
                    sin_cache,
                ]
            )
        else:
            layer_known_edges_names.extend([attn_qkv_weight, attn_qkv_bias])
        layer_known_edges_names.extend(
            [attn_out_weight, attn_out_bias, mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias]
        )
        layer_known_edges_names.extend(
            ["attention_mask", "step", "seqlens_k", "total_sequence_length", "input_metadata", "position_ids"]
        )

        subgraph_nodes = []
        subgraph_nodes.extend(self.layernorm([i_hidden_states, ln_weight, ln_bias], ["ln_out"]))
        subgraph_nodes.extend(self.gemm(["attn_out", attn_out_weight, attn_out_bias], ["attn_add_out"], "OutProj_"))
        subgraph_nodes.extend(self.gemm(["ln_out", mlp_fc1_weight, mlp_fc1_bias], ["fc1_out"], "FC1_"))
        subgraph_nodes.extend(self.fastgelu(["fc1_out"], ["gelu_out"]))
        subgraph_nodes.extend(self.gemm(["gelu_out", mlp_fc2_weight, mlp_fc2_bias], ["fc2_out"], "FC2_"))
        subgraph_nodes.extend(self.add(["attn_add_out", "fc2_out"], ["residual_1_out"], "Residual_1"))
        subgraph_nodes.extend(self.add([i_hidden_states, "residual_1_out"], [o_hidden_states], "Residual_2"))
        if self.attn_op_type != AttentionOpType.Attention:
            subgraph_nodes.extend(self.gemm(["ln_out", attn_q_weight, attn_q_bias], ["query"], "Q_"))
            subgraph_nodes.extend(self.gemm(["ln_out", attn_k_weight, attn_k_bias], ["key"], "K_"))
            subgraph_nodes.extend(self.gemm(["ln_out", attn_v_weight, attn_v_bias], ["value"], "V_"))
            # vllm engine requires full position ids as the input
            pos_ids_name = "position_ids" if self.attn_op_type == AttentionOpType.PagedAttention else "step"
            subgraph_nodes.extend(self.rotary(["query", pos_ids_name, cos_cache, sin_cache], ["query_rot"], "Q_"))
            subgraph_nodes.extend(self.rotary(["key", pos_ids_name, cos_cache, sin_cache], ["key_rot"], "K_"))
            if self.attn_op_type == AttentionOpType.MultiHeadAttention:
                subgraph_nodes.extend(
                    self.mha(
                        ["query_rot", "key_rot", "value", "", "attention_mask", "", i_key_cache, i_value_cache],
                        ["attn_out", o_key_cache, o_value_cache],
                    )
                )
            elif self.attn_op_type == AttentionOpType.GroupQueryAttention:
                subgraph_nodes.extend(
                    self.gqa(
                        [
                            "query_rot",
                            "key_rot",
                            "value",
                            i_key_cache,
                            i_value_cache,
                            "seqlens_k",
                            "total_sequence_length",
                        ],
                        ["attn_out", o_key_cache, o_value_cache],
                    )
                )
                if layer_id == 0:
                    gqa_aux_nodes = self.get_gqa_aux_nodes()
                    for new_node in gqa_aux_nodes:
                        self.nodes_to_add.append(new_node)
                        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
                    self.model.add_initializer(
                        numpy_helper.from_array(np.array([1], dtype="int64"), name="one"), self.this_graph_name
                    )
            elif self.attn_op_type == AttentionOpType.PagedAttention:
                subgraph_nodes.extend(
                    self.paged_attn(
                        ["query_rot", "key_rot", "value", i_key_cache, i_value_cache, "input_metadata"],
                        ["attn_out"],
                    )
                )
        else:
            past_name = f"past_{layer_id}"
            present_name = f"present_{layer_id}"
            layer_known_edges_names.extend([past_name, present_name])
            subgraph_nodes.extend(
                self.attention(
                    ["ln_out", attn_qkv_weight, attn_qkv_bias, "attention_mask", past_name], ["attn_out", present_name]
                )
            )

        self.set_unique_name_and_add_nodes(subgraph_nodes, layer_id, layer_known_edges_names)

        self.replace_fp32_value_info(i_hidden_states, ["batch_size", "seq_len", "hidden_size"])
        self.replace_fp32_value_info(o_hidden_states, ["batch_size", "seq_len", "hidden_size"])

        self.nodes_to_remove.append(node)
        self.prune_graph = True


class PhiOnnxModel(OnnxModel):
    def __init__(self, model: ModelProto, num_heads: int, hidden_size: int):
        super().__init__(model)
        self.phi2_preprocessor = Phi2PreProcessor(self.model, num_heads, hidden_size)
        self.fission_transformer_block = FissionTransformerBlockPhi(self, num_heads)
        self.fission_causal_lm_head = FissionTransformerCausalLMHeadPhi(self)
        self.fission_transformer_layernorm = FissionTransformerLayerNormPhi(self)
        self.fission_transformer_embedding = FissionTransformerEmbeddingPhi(self)

    def optimize(self, options: FusionOptions | None = None, add_dynamic_axes: bool = False):
        assert options is not None
        attn_op_type = options.attention_op_type

        self.fission_transformer_block.set_attention_op_type(attn_op_type)

        self.phi2_preprocessor.preprocess_onnx(attn_op_type)

        self.fission_transformer_block.apply()
        self.fission_transformer_layernorm.apply()
        self.fission_causal_lm_head.apply()
        self.fission_transformer_embedding.apply()

        super().prune_graph()

        # SLN ctor is placed here intentionally to delay the symbolic shape inference
        self.fuse_sln = FusionSkipLayerNormalization(self)
        self.fuse_bias_sln = FusionBiasSkipLayerNormalization(self)
        self.fuse_sln.apply()
        self.fuse_bias_sln.apply()

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            "Attention",
            "MultiHeadAttention",
            "GroupQueryAttention",
            "PagedAttention",
            "Gelu",
            "BiasGelu",
            "FastGelu",
            "LayerNormalization",
            "SkipLayerNormalization",
        ]
        for op in ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)

        logger.info(f"Optimized operators: {op_count}")
        return op_count

    def is_fully_optimized(self, fused_op_count=None):
        """
        Returns True when the model is fully optimized.
        """
        if fused_op_count is None:
            fused_op_count = self.get_fused_operator_statistics()

        def op_count(op_name: str):
            return fused_op_count.get(op_name) or 0

        attention = (
            op_count("Attention")
            + op_count("MultiHeadAttention")
            + op_count("GroupQueryAttention")
            + op_count("PagedAttention")
        )
        gelu = op_count("Gelu") + op_count("BiasGelu") + op_count("FastGelu")
        layer_norm = op_count("LayerNormalization") + op_count("SkipLayerNormalization")

        is_perfect = (attention > 0) and (attention == gelu) and (layer_norm >= attention)

        if layer_norm == 0:
            logger.debug("Layer Normalization not fused")

        if gelu == 0:
            logger.debug("Gelu (or FastGelu) not fused")

        if attention == 0:
            logger.warning("Attention (or MultiHeadAttention) not fused")

        return is_perfect

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\msgpack\fallback.py ===
"""Fallback pure Python implementation of msgpack"""

import struct
import sys
from datetime import datetime as _DateTime

if hasattr(sys, "pypy_version_info"):
    from __pypy__ import newlist_hint
    from __pypy__.builders import BytesBuilder

    _USING_STRINGBUILDER = True

    class BytesIO:
        def __init__(self, s=b""):
            if s:
                self.builder = BytesBuilder(len(s))
                self.builder.append(s)
            else:
                self.builder = BytesBuilder()

        def write(self, s):
            if isinstance(s, memoryview):
                s = s.tobytes()
            elif isinstance(s, bytearray):
                s = bytes(s)
            self.builder.append(s)

        def getvalue(self):
            return self.builder.build()

else:
    from io import BytesIO

    _USING_STRINGBUILDER = False

    def newlist_hint(size):
        return []


from .exceptions import BufferFull, ExtraData, FormatError, OutOfData, StackError
from .ext import ExtType, Timestamp

EX_SKIP = 0
EX_CONSTRUCT = 1
EX_READ_ARRAY_HEADER = 2
EX_READ_MAP_HEADER = 3

TYPE_IMMEDIATE = 0
TYPE_ARRAY = 1
TYPE_MAP = 2
TYPE_RAW = 3
TYPE_BIN = 4
TYPE_EXT = 5

DEFAULT_RECURSE_LIMIT = 511


def _check_type_strict(obj, t, type=type, tuple=tuple):
    if type(t) is tuple:
        return type(obj) in t
    else:
        return type(obj) is t


def _get_data_from_buffer(obj):
    view = memoryview(obj)
    if view.itemsize != 1:
        raise ValueError("cannot unpack from multi-byte object")
    return view


def unpackb(packed, **kwargs):
    """
    Unpack an object from `packed`.

    Raises ``ExtraData`` when *packed* contains extra bytes.
    Raises ``ValueError`` when *packed* is incomplete.
    Raises ``FormatError`` when *packed* is not valid msgpack.
    Raises ``StackError`` when *packed* contains too nested.
    Other exceptions can be raised during unpacking.

    See :class:`Unpacker` for options.
    """
    unpacker = Unpacker(None, max_buffer_size=len(packed), **kwargs)
    unpacker.feed(packed)
    try:
        ret = unpacker._unpack()
    except OutOfData:
        raise ValueError("Unpack failed: incomplete input")
    except RecursionError:
        raise StackError
    if unpacker._got_extradata():
        raise ExtraData(ret, unpacker._get_extradata())
    return ret


_NO_FORMAT_USED = ""
_MSGPACK_HEADERS = {
    0xC4: (1, _NO_FORMAT_USED, TYPE_BIN),
    0xC5: (2, ">H", TYPE_BIN),
    0xC6: (4, ">I", TYPE_BIN),
    0xC7: (2, "Bb", TYPE_EXT),
    0xC8: (3, ">Hb", TYPE_EXT),
    0xC9: (5, ">Ib", TYPE_EXT),
    0xCA: (4, ">f"),
    0xCB: (8, ">d"),
    0xCC: (1, _NO_FORMAT_USED),
    0xCD: (2, ">H"),
    0xCE: (4, ">I"),
    0xCF: (8, ">Q"),
    0xD0: (1, "b"),
    0xD1: (2, ">h"),
    0xD2: (4, ">i"),
    0xD3: (8, ">q"),
    0xD4: (1, "b1s", TYPE_EXT),
    0xD5: (2, "b2s", TYPE_EXT),
    0xD6: (4, "b4s", TYPE_EXT),
    0xD7: (8, "b8s", TYPE_EXT),
    0xD8: (16, "b16s", TYPE_EXT),
    0xD9: (1, _NO_FORMAT_USED, TYPE_RAW),
    0xDA: (2, ">H", TYPE_RAW),
    0xDB: (4, ">I", TYPE_RAW),
    0xDC: (2, ">H", TYPE_ARRAY),
    0xDD: (4, ">I", TYPE_ARRAY),
    0xDE: (2, ">H", TYPE_MAP),
    0xDF: (4, ">I", TYPE_MAP),
}


class Unpacker:
    """Streaming unpacker.

    Arguments:

    :param file_like:
        File-like object having `.read(n)` method.
        If specified, unpacker reads serialized data from it and `.feed()` is not usable.

    :param int read_size:
        Used as `file_like.read(read_size)`. (default: `min(16*1024, max_buffer_size)`)

    :param bool use_list:
        If true, unpack msgpack array to Python list.
        Otherwise, unpack to Python tuple. (default: True)

    :param bool raw:
        If true, unpack msgpack raw to Python bytes.
        Otherwise, unpack to Python str by decoding with UTF-8 encoding (default).

    :param int timestamp:
        Control how timestamp type is unpacked:

            0 - Timestamp
            1 - float  (Seconds from the EPOCH)
            2 - int  (Nanoseconds from the EPOCH)
            3 - datetime.datetime  (UTC).

    :param bool strict_map_key:
        If true (default), only str or bytes are accepted for map (dict) keys.

    :param object_hook:
        When specified, it should be callable.
        Unpacker calls it with a dict argument after unpacking msgpack map.
        (See also simplejson)

    :param object_pairs_hook:
        When specified, it should be callable.
        Unpacker calls it with a list of key-value pairs after unpacking msgpack map.
        (See also simplejson)

    :param str unicode_errors:
        The error handler for decoding unicode. (default: 'strict')
        This option should be used only when you have msgpack data which
        contains invalid UTF-8 string.

    :param int max_buffer_size:
        Limits size of data waiting unpacked.  0 means 2**32-1.
        The default value is 100*1024*1024 (100MiB).
        Raises `BufferFull` exception when it is insufficient.
        You should set this parameter when unpacking data from untrusted source.

    :param int max_str_len:
        Deprecated, use *max_buffer_size* instead.
        Limits max length of str. (default: max_buffer_size)

    :param int max_bin_len:
        Deprecated, use *max_buffer_size* instead.
        Limits max length of bin. (default: max_buffer_size)

    :param int max_array_len:
        Limits max length of array.
        (default: max_buffer_size)

    :param int max_map_len:
        Limits max length of map.
        (default: max_buffer_size//2)

    :param int max_ext_len:
        Deprecated, use *max_buffer_size* instead.
        Limits max size of ext type.  (default: max_buffer_size)

    Example of streaming deserialize from file-like object::

        unpacker = Unpacker(file_like)
        for o in unpacker:
            process(o)

    Example of streaming deserialize from socket::

        unpacker = Unpacker()
        while True:
            buf = sock.recv(1024**2)
            if not buf:
                break
            unpacker.feed(buf)
            for o in unpacker:
                process(o)

    Raises ``ExtraData`` when *packed* contains extra bytes.
    Raises ``OutOfData`` when *packed* is incomplete.
    Raises ``FormatError`` when *packed* is not valid msgpack.
    Raises ``StackError`` when *packed* contains too nested.
    Other exceptions can be raised during unpacking.
    """

    def __init__(
        self,
        file_like=None,
        *,
        read_size=0,
        use_list=True,
        raw=False,
        timestamp=0,
        strict_map_key=True,
        object_hook=None,
        object_pairs_hook=None,
        list_hook=None,
        unicode_errors=None,
        max_buffer_size=100 * 1024 * 1024,
        ext_hook=ExtType,
        max_str_len=-1,
        max_bin_len=-1,
        max_array_len=-1,
        max_map_len=-1,
        max_ext_len=-1,
    ):
        if unicode_errors is None:
            unicode_errors = "strict"

        if file_like is None:
            self._feeding = True
        else:
            if not callable(file_like.read):
                raise TypeError("`file_like.read` must be callable")
            self.file_like = file_like
            self._feeding = False

        #: array of bytes fed.
        self._buffer = bytearray()
        #: Which position we currently reads
        self._buff_i = 0

        # When Unpacker is used as an iterable, between the calls to next(),
        # the buffer is not "consumed" completely, for efficiency sake.
        # Instead, it is done sloppily.  To make sure we raise BufferFull at
        # the correct moments, we have to keep track of how sloppy we were.
        # Furthermore, when the buffer is incomplete (that is: in the case
        # we raise an OutOfData) we need to rollback the buffer to the correct
        # state, which _buf_checkpoint records.
        self._buf_checkpoint = 0

        if not max_buffer_size:
            max_buffer_size = 2**31 - 1
        if max_str_len == -1:
            max_str_len = max_buffer_size
        if max_bin_len == -1:
            max_bin_len = max_buffer_size
        if max_array_len == -1:
            max_array_len = max_buffer_size
        if max_map_len == -1:
            max_map_len = max_buffer_size // 2
        if max_ext_len == -1:
            max_ext_len = max_buffer_size

        self._max_buffer_size = max_buffer_size
        if read_size > self._max_buffer_size:
            raise ValueError("read_size must be smaller than max_buffer_size")
        self._read_size = read_size or min(self._max_buffer_size, 16 * 1024)
        self._raw = bool(raw)
        self._strict_map_key = bool(strict_map_key)
        self._unicode_errors = unicode_errors
        self._use_list = use_list
        if not (0 <= timestamp <= 3):
            raise ValueError("timestamp must be 0..3")
        self._timestamp = timestamp
        self._list_hook = list_hook
        self._object_hook = object_hook
        self._object_pairs_hook = object_pairs_hook
        self._ext_hook = ext_hook
        self._max_str_len = max_str_len
        self._max_bin_len = max_bin_len
        self._max_array_len = max_array_len
        self._max_map_len = max_map_len
        self._max_ext_len = max_ext_len
        self._stream_offset = 0

        if list_hook is not None and not callable(list_hook):
            raise TypeError("`list_hook` is not callable")
        if object_hook is not None and not callable(object_hook):
            raise TypeError("`object_hook` is not callable")
        if object_pairs_hook is not None and not callable(object_pairs_hook):
            raise TypeError("`object_pairs_hook` is not callable")
        if object_hook is not None and object_pairs_hook is not None:
            raise TypeError("object_pairs_hook and object_hook are mutually exclusive")
        if not callable(ext_hook):
            raise TypeError("`ext_hook` is not callable")

    def feed(self, next_bytes):
        assert self._feeding
        view = _get_data_from_buffer(next_bytes)
        if len(self._buffer) - self._buff_i + len(view) > self._max_buffer_size:
            raise BufferFull

        # Strip buffer before checkpoint before reading file.
        if self._buf_checkpoint > 0:
            del self._buffer[: self._buf_checkpoint]
            self._buff_i -= self._buf_checkpoint
            self._buf_checkpoint = 0

        # Use extend here: INPLACE_ADD += doesn't reliably typecast memoryview in jython
        self._buffer.extend(view)
        view.release()

    def _consume(self):
        """Gets rid of the used parts of the buffer."""
        self._stream_offset += self._buff_i - self._buf_checkpoint
        self._buf_checkpoint = self._buff_i

    def _got_extradata(self):
        return self._buff_i < len(self._buffer)

    def _get_extradata(self):
        return self._buffer[self._buff_i :]

    def read_bytes(self, n):
        ret = self._read(n, raise_outofdata=False)
        self._consume()
        return ret

    def _read(self, n, raise_outofdata=True):
        # (int) -> bytearray
        self._reserve(n, raise_outofdata=raise_outofdata)
        i = self._buff_i
        ret = self._buffer[i : i + n]
        self._buff_i = i + len(ret)
        return ret

    def _reserve(self, n, raise_outofdata=True):
        remain_bytes = len(self._buffer) - self._buff_i - n

        # Fast path: buffer has n bytes already
        if remain_bytes >= 0:
            return

        if self._feeding:
            self._buff_i = self._buf_checkpoint
            raise OutOfData

        # Strip buffer before checkpoint before reading file.
        if self._buf_checkpoint > 0:
            del self._buffer[: self._buf_checkpoint]
            self._buff_i -= self._buf_checkpoint
            self._buf_checkpoint = 0

        # Read from file
        remain_bytes = -remain_bytes
        if remain_bytes + len(self._buffer) > self._max_buffer_size:
            raise BufferFull
        while remain_bytes > 0:
            to_read_bytes = max(self._read_size, remain_bytes)
            read_data = self.file_like.read(to_read_bytes)
            if not read_data:
                break
            assert isinstance(read_data, bytes)
            self._buffer += read_data
            remain_bytes -= len(read_data)

        if len(self._buffer) < n + self._buff_i and raise_outofdata:
            self._buff_i = 0  # rollback
            raise OutOfData

    def _read_header(self):
        typ = TYPE_IMMEDIATE
        n = 0
        obj = None
        self._reserve(1)
        b = self._buffer[self._buff_i]
        self._buff_i += 1
        if b & 0b10000000 == 0:
            obj = b
        elif b & 0b11100000 == 0b11100000:
            obj = -1 - (b ^ 0xFF)
        elif b & 0b11100000 == 0b10100000:
            n = b & 0b00011111
            typ = TYPE_RAW
            if n > self._max_str_len:
                raise ValueError(f"{n} exceeds max_str_len({self._max_str_len})")
            obj = self._read(n)
        elif b & 0b11110000 == 0b10010000:
            n = b & 0b00001111
            typ = TYPE_ARRAY
            if n > self._max_array_len:
                raise ValueError(f"{n} exceeds max_array_len({self._max_array_len})")
        elif b & 0b11110000 == 0b10000000:
            n = b & 0b00001111
            typ = TYPE_MAP
            if n > self._max_map_len:
                raise ValueError(f"{n} exceeds max_map_len({self._max_map_len})")
        elif b == 0xC0:
            obj = None
        elif b == 0xC2:
            obj = False
        elif b == 0xC3:
            obj = True
        elif 0xC4 <= b <= 0xC6:
            size, fmt, typ = _MSGPACK_HEADERS[b]
            self._reserve(size)
            if len(fmt) > 0:
                n = struct.unpack_from(fmt, self._buffer, self._buff_i)[0]
            else:
                n = self._buffer[self._buff_i]
            self._buff_i += size
            if n > self._max_bin_len:
                raise ValueError(f"{n} exceeds max_bin_len({self._max_bin_len})")
            obj = self._read(n)
        elif 0xC7 <= b <= 0xC9:
            size, fmt, typ = _MSGPACK_HEADERS[b]
            self._reserve(size)
            L, n = struct.unpack_from(fmt, self._buffer, self._buff_i)
            self._buff_i += size
            if L > self._max_ext_len:
                raise ValueError(f"{L} exceeds max_ext_len({self._max_ext_len})")
            obj = self._read(L)
        elif 0xCA <= b <= 0xD3:
            size, fmt = _MSGPACK_HEADERS[b]
            self._reserve(size)
            if len(fmt) > 0:
                obj = struct.unpack_from(fmt, self._buffer, self._buff_i)[0]
            else:
                obj = self._buffer[self._buff_i]
            self._buff_i += size
        elif 0xD4 <= b <= 0xD8:
            size, fmt, typ = _MSGPACK_HEADERS[b]
            if self._max_ext_len < size:
                raise ValueError(f"{size} exceeds max_ext_len({self._max_ext_len})")
            self._reserve(size + 1)
            n, obj = struct.unpack_from(fmt, self._buffer, self._buff_i)
            self._buff_i += size + 1
        elif 0xD9 <= b <= 0xDB:
            size, fmt, typ = _MSGPACK_HEADERS[b]
            self._reserve(size)
            if len(fmt) > 0:
                (n,) = struct.unpack_from(fmt, self._buffer, self._buff_i)
            else:
                n = self._buffer[self._buff_i]
            self._buff_i += size
            if n > self._max_str_len:
                raise ValueError(f"{n} exceeds max_str_len({self._max_str_len})")
            obj = self._read(n)
        elif 0xDC <= b <= 0xDD:
            size, fmt, typ = _MSGPACK_HEADERS[b]
            self._reserve(size)
            (n,) = struct.unpack_from(fmt, self._buffer, self._buff_i)
            self._buff_i += size
            if n > self._max_array_len:
                raise ValueError(f"{n} exceeds max_array_len({self._max_array_len})")
        elif 0xDE <= b <= 0xDF:
            size, fmt, typ = _MSGPACK_HEADERS[b]
            self._reserve(size)
            (n,) = struct.unpack_from(fmt, self._buffer, self._buff_i)
            self._buff_i += size
            if n > self._max_map_len:
                raise ValueError(f"{n} exceeds max_map_len({self._max_map_len})")
        else:
            raise FormatError("Unknown header: 0x%x" % b)
        return typ, n, obj

    def _unpack(self, execute=EX_CONSTRUCT):
        typ, n, obj = self._read_header()

        if execute == EX_READ_ARRAY_HEADER:
            if typ != TYPE_ARRAY:
                raise ValueError("Expected array")
            return n
        if execute == EX_READ_MAP_HEADER:
            if typ != TYPE_MAP:
                raise ValueError("Expected map")
            return n
        # TODO should we eliminate the recursion?
        if typ == TYPE_ARRAY:
            if execute == EX_SKIP:
                for i in range(n):
                    # TODO check whether we need to call `list_hook`
                    self._unpack(EX_SKIP)
                return
            ret = newlist_hint(n)
            for i in range(n):
                ret.append(self._unpack(EX_CONSTRUCT))
            if self._list_hook is not None:
                ret = self._list_hook(ret)
            # TODO is the interaction between `list_hook` and `use_list` ok?
            return ret if self._use_list else tuple(ret)
        if typ == TYPE_MAP:
            if execute == EX_SKIP:
                for i in range(n):
                    # TODO check whether we need to call hooks
                    self._unpack(EX_SKIP)
                    self._unpack(EX_SKIP)
                return
            if self._object_pairs_hook is not None:
                ret = self._object_pairs_hook(
                    (self._unpack(EX_CONSTRUCT), self._unpack(EX_CONSTRUCT)) for _ in range(n)
                )
            else:
                ret = {}
                for _ in range(n):
                    key = self._unpack(EX_CONSTRUCT)
                    if self._strict_map_key and type(key) not in (str, bytes):
                        raise ValueError("%s is not allowed for map key" % str(type(key)))
                    if isinstance(key, str):
                        key = sys.intern(key)
                    ret[key] = self._unpack(EX_CONSTRUCT)
                if self._object_hook is not None:
                    ret = self._object_hook(ret)
            return ret
        if execute == EX_SKIP:
            return
        if typ == TYPE_RAW:
            if self._raw:
                obj = bytes(obj)
            else:
                obj = obj.decode("utf_8", self._unicode_errors)
            return obj
        if typ == TYPE_BIN:
            return bytes(obj)
        if typ == TYPE_EXT:
            if n == -1:  # timestamp
                ts = Timestamp.from_bytes(bytes(obj))
                if self._timestamp == 1:
                    return ts.to_unix()
                elif self._timestamp == 2:
                    return ts.to_unix_nano()
                elif self._timestamp == 3:
                    return ts.to_datetime()
                else:
                    return ts
            else:
                return self._ext_hook(n, bytes(obj))
        assert typ == TYPE_IMMEDIATE
        return obj

    def __iter__(self):
        return self

    def __next__(self):
        try:
            ret = self._unpack(EX_CONSTRUCT)
            self._consume()
            return ret
        except OutOfData:
            self._consume()
            raise StopIteration
        except RecursionError:
            raise StackError

    next = __next__

    def skip(self):
        self._unpack(EX_SKIP)
        self._consume()

    def unpack(self):
        try:
            ret = self._unpack(EX_CONSTRUCT)
        except RecursionError:
            raise StackError
        self._consume()
        return ret

    def read_array_header(self):
        ret = self._unpack(EX_READ_ARRAY_HEADER)
        self._consume()
        return ret

    def read_map_header(self):
        ret = self._unpack(EX_READ_MAP_HEADER)
        self._consume()
        return ret

    def tell(self):
        return self._stream_offset


class Packer:
    """
    MessagePack Packer

    Usage::

        packer = Packer()
        astream.write(packer.pack(a))
        astream.write(packer.pack(b))

    Packer's constructor has some keyword arguments:

    :param default:
        When specified, it should be callable.
        Convert user type to builtin type that Packer supports.
        See also simplejson's document.

    :param bool use_single_float:
        Use single precision float type for float. (default: False)

    :param bool autoreset:
        Reset buffer after each pack and return its content as `bytes`. (default: True).
        If set this to false, use `bytes()` to get content and `.reset()` to clear buffer.

    :param bool use_bin_type:
        Use bin type introduced in msgpack spec 2.0 for bytes.
        It also enables str8 type for unicode. (default: True)

    :param bool strict_types:
        If set to true, types will be checked to be exact. Derived classes
        from serializable types will not be serialized and will be
        treated as unsupported type and forwarded to default.
        Additionally tuples will not be serialized as lists.
        This is useful when trying to implement accurate serialization
        for python types.

    :param bool datetime:
        If set to true, datetime with tzinfo is packed into Timestamp type.
        Note that the tzinfo is stripped in the timestamp.
        You can get UTC datetime with `timestamp=3` option of the Unpacker.

    :param str unicode_errors:
        The error handler for encoding unicode. (default: 'strict')
        DO NOT USE THIS!!  This option is kept for very specific usage.

    :param int buf_size:
        Internal buffer size. This option is used only for C implementation.
    """

    def __init__(
        self,
        *,
        default=None,
        use_single_float=False,
        autoreset=True,
        use_bin_type=True,
        strict_types=False,
        datetime=False,
        unicode_errors=None,
        buf_size=None,
    ):
        self._strict_types = strict_types
        self._use_float = use_single_float
        self._autoreset = autoreset
        self._use_bin_type = use_bin_type
        self._buffer = BytesIO()
        self._datetime = bool(datetime)
        self._unicode_errors = unicode_errors or "strict"
        if default is not None and not callable(default):
            raise TypeError("default must be callable")
        self._default = default

    def _pack(
        self,
        obj,
        nest_limit=DEFAULT_RECURSE_LIMIT,
        check=isinstance,
        check_type_strict=_check_type_strict,
    ):
        default_used = False
        if self._strict_types:
            check = check_type_strict
            list_types = list
        else:
            list_types = (list, tuple)
        while True:
            if nest_limit < 0:
                raise ValueError("recursion limit exceeded")
            if obj is None:
                return self._buffer.write(b"\xc0")
            if check(obj, bool):
                if obj:
                    return self._buffer.write(b"\xc3")
                return self._buffer.write(b"\xc2")
            if check(obj, int):
                if 0 <= obj < 0x80:
                    return self._buffer.write(struct.pack("B", obj))
                if -0x20 <= obj < 0:
                    return self._buffer.write(struct.pack("b", obj))
                if 0x80 <= obj <= 0xFF:
                    return self._buffer.write(struct.pack("BB", 0xCC, obj))
                if -0x80 <= obj < 0:
                    return self._buffer.write(struct.pack(">Bb", 0xD0, obj))
                if 0xFF < obj <= 0xFFFF:
                    return self._buffer.write(struct.pack(">BH", 0xCD, obj))
                if -0x8000 <= obj < -0x80:
                    return self._buffer.write(struct.pack(">Bh", 0xD1, obj))
                if 0xFFFF < obj <= 0xFFFFFFFF:
                    return self._buffer.write(struct.pack(">BI", 0xCE, obj))
                if -0x80000000 <= obj < -0x8000:
                    return self._buffer.write(struct.pack(">Bi", 0xD2, obj))
                if 0xFFFFFFFF < obj <= 0xFFFFFFFFFFFFFFFF:
                    return self._buffer.write(struct.pack(">BQ", 0xCF, obj))
                if -0x8000000000000000 <= obj < -0x80000000:
                    return self._buffer.write(struct.pack(">Bq", 0xD3, obj))
                if not default_used and self._default is not None:
                    obj = self._default(obj)
                    default_used = True
                    continue
                raise OverflowError("Integer value out of range")
            if check(obj, (bytes, bytearray)):
                n = len(obj)
                if n >= 2**32:
                    raise ValueError("%s is too large" % type(obj).__name__)
                self._pack_bin_header(n)
                return self._buffer.write(obj)
            if check(obj, str):
                obj = obj.encode("utf-8", self._unicode_errors)
                n = len(obj)
                if n >= 2**32:
                    raise ValueError("String is too large")
                self._pack_raw_header(n)
                return self._buffer.write(obj)
            if check(obj, memoryview):
                n = obj.nbytes
                if n >= 2**32:
                    raise ValueError("Memoryview is too large")
                self._pack_bin_header(n)
                return self._buffer.write(obj)
            if check(obj, float):
                if self._use_float:
                    return self._buffer.write(struct.pack(">Bf", 0xCA, obj))
                return self._buffer.write(struct.pack(">Bd", 0xCB, obj))
            if check(obj, (ExtType, Timestamp)):
                if check(obj, Timestamp):
                    code = -1
                    data = obj.to_bytes()
                else:
                    code = obj.code
                    data = obj.data
                assert isinstance(code, int)
                assert isinstance(data, bytes)
                L = len(data)
                if L == 1:
                    self._buffer.write(b"\xd4")
                elif L == 2:
                    self._buffer.write(b"\xd5")
                elif L == 4:
                    self._buffer.write(b"\xd6")
                elif L == 8:
                    self._buffer.write(b"\xd7")
                elif L == 16:
                    self._buffer.write(b"\xd8")
                elif L <= 0xFF:
                    self._buffer.write(struct.pack(">BB", 0xC7, L))
                elif L <= 0xFFFF:
                    self._buffer.write(struct.pack(">BH", 0xC8, L))
                else:
                    self._buffer.write(struct.pack(">BI", 0xC9, L))
                self._buffer.write(struct.pack("b", code))
                self._buffer.write(data)
                return
            if check(obj, list_types):
                n = len(obj)
                self._pack_array_header(n)
                for i in range(n):
                    self._pack(obj[i], nest_limit - 1)
                return
            if check(obj, dict):
                return self._pack_map_pairs(len(obj), obj.items(), nest_limit - 1)

            if self._datetime and check(obj, _DateTime) and obj.tzinfo is not None:
                obj = Timestamp.from_datetime(obj)
                default_used = 1
                continue

            if not default_used and self._default is not None:
                obj = self._default(obj)
                default_used = 1
                continue

            if self._datetime and check(obj, _DateTime):
                raise ValueError(f"Cannot serialize {obj!r} where tzinfo=None")

            raise TypeError(f"Cannot serialize {obj!r}")

    def pack(self, obj):
        try:
            self._pack(obj)
        except:
            self._buffer = BytesIO()  # force reset
            raise
        if self._autoreset:
            ret = self._buffer.getvalue()
            self._buffer = BytesIO()
            return ret

    def pack_map_pairs(self, pairs):
        self._pack_map_pairs(len(pairs), pairs)
        if self._autoreset:
            ret = self._buffer.getvalue()
            self._buffer = BytesIO()
            return ret

    def pack_array_header(self, n):
        if n >= 2**32:
            raise ValueError
        self._pack_array_header(n)
        if self._autoreset:
            ret = self._buffer.getvalue()
            self._buffer = BytesIO()
            return ret

    def pack_map_header(self, n):
        if n >= 2**32:
            raise ValueError
        self._pack_map_header(n)
        if self._autoreset:
            ret = self._buffer.getvalue()
            self._buffer = BytesIO()
            return ret

    def pack_ext_type(self, typecode, data):
        if not isinstance(typecode, int):
            raise TypeError("typecode must have int type.")
        if not 0 <= typecode <= 127:
            raise ValueError("typecode should be 0-127")
        if not isinstance(data, bytes):
            raise TypeError("data must have bytes type")
        L = len(data)
        if L > 0xFFFFFFFF:
            raise ValueError("Too large data")
        if L == 1:
            self._buffer.write(b"\xd4")
        elif L == 2:
            self._buffer.write(b"\xd5")
        elif L == 4:
            self._buffer.write(b"\xd6")
        elif L == 8:
            self._buffer.write(b"\xd7")
        elif L == 16:
            self._buffer.write(b"\xd8")
        elif L <= 0xFF:
            self._buffer.write(b"\xc7" + struct.pack("B", L))
        elif L <= 0xFFFF:
            self._buffer.write(b"\xc8" + struct.pack(">H", L))
        else:
            self._buffer.write(b"\xc9" + struct.pack(">I", L))
        self._buffer.write(struct.pack("B", typecode))
        self._buffer.write(data)

    def _pack_array_header(self, n):
        if n <= 0x0F:
            return self._buffer.write(struct.pack("B", 0x90 + n))
        if n <= 0xFFFF:
            return self._buffer.write(struct.pack(">BH", 0xDC, n))
        if n <= 0xFFFFFFFF:
            return self._buffer.write(struct.pack(">BI", 0xDD, n))
        raise ValueError("Array is too large")

    def _pack_map_header(self, n):
        if n <= 0x0F:
            return self._buffer.write(struct.pack("B", 0x80 + n))
        if n <= 0xFFFF:
            return self._buffer.write(struct.pack(">BH", 0xDE, n))
        if n <= 0xFFFFFFFF:
            return self._buffer.write(struct.pack(">BI", 0xDF, n))
        raise ValueError("Dict is too large")

    def _pack_map_pairs(self, n, pairs, nest_limit=DEFAULT_RECURSE_LIMIT):
        self._pack_map_header(n)
        for k, v in pairs:
            self._pack(k, nest_limit - 1)
            self._pack(v, nest_limit - 1)

    def _pack_raw_header(self, n):
        if n <= 0x1F:
            self._buffer.write(struct.pack("B", 0xA0 + n))
        elif self._use_bin_type and n <= 0xFF:
            self._buffer.write(struct.pack(">BB", 0xD9, n))
        elif n <= 0xFFFF:
            self._buffer.write(struct.pack(">BH", 0xDA, n))
        elif n <= 0xFFFFFFFF:
            self._buffer.write(struct.pack(">BI", 0xDB, n))
        else:
            raise ValueError("Raw is too large")

    def _pack_bin_header(self, n):
        if not self._use_bin_type:
            return self._pack_raw_header(n)
        elif n <= 0xFF:
            return self._buffer.write(struct.pack(">BB", 0xC4, n))
        elif n <= 0xFFFF:
            return self._buffer.write(struct.pack(">BH", 0xC5, n))
        elif n <= 0xFFFFFFFF:
            return self._buffer.write(struct.pack(">BI", 0xC6, n))
        else:
            raise ValueError("Bin is too large")

    def bytes(self):
        """Return internal buffer contents as bytes object"""
        return self._buffer.getvalue()

    def reset(self):
        """Reset internal buffer.

        This method is useful only when autoreset=False.
        """
        self._buffer = BytesIO()

    def getbuffer(self):
        """Return view of internal buffer."""
        if _USING_STRINGBUILDER:
            return memoryview(self.bytes())
        else:
            return self._buffer.getbuffer()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\routing\rules.py ===
from __future__ import annotations

import ast
import re
import typing as t
from dataclasses import dataclass
from string import Template
from types import CodeType
from urllib.parse import quote

from ..datastructures import iter_multi_items
from ..urls import _urlencode
from .converters import ValidationError

if t.TYPE_CHECKING:
    from .converters import BaseConverter
    from .map import Map


class Weighting(t.NamedTuple):
    number_static_weights: int
    static_weights: list[tuple[int, int]]
    number_argument_weights: int
    argument_weights: list[int]


@dataclass
class RulePart:
    """A part of a rule.

    Rules can be represented by parts as delimited by `/` with
    instances of this class representing those parts. The *content* is
    either the raw content if *static* or a regex string to match
    against. The *weight* can be used to order parts when matching.

    """

    content: str
    final: bool
    static: bool
    suffixed: bool
    weight: Weighting


_part_re = re.compile(
    r"""
    (?:
        (?P<slash>/)                                 # a slash
      |
        (?P<static>[^</]+)                           # static rule data
      |
        (?:
          <
            (?:
              (?P<converter>[a-zA-Z_][a-zA-Z0-9_]*)   # converter name
              (?:\((?P<arguments>.*?)\))?             # converter arguments
              :                                       # variable delimiter
            )?
            (?P<variable>[a-zA-Z_][a-zA-Z0-9_]*)      # variable name
           >
        )
    )
    """,
    re.VERBOSE,
)

_simple_rule_re = re.compile(r"<([^>]+)>")
_converter_args_re = re.compile(
    r"""
    \s*
    ((?P<name>\w+)\s*=\s*)?
    (?P<value>
        True|False|
        \d+.\d+|
        \d+.|
        \d+|
        [\w\d_.]+|
        [urUR]?(?P<stringval>"[^"]*?"|'[^']*')
    )\s*,
    """,
    re.VERBOSE,
)


_PYTHON_CONSTANTS = {"None": None, "True": True, "False": False}


def _find(value: str, target: str, pos: int) -> int:
    """Find the *target* in *value* after *pos*.

    Returns the *value* length if *target* isn't found.
    """
    try:
        return value.index(target, pos)
    except ValueError:
        return len(value)


def _pythonize(value: str) -> None | bool | int | float | str:
    if value in _PYTHON_CONSTANTS:
        return _PYTHON_CONSTANTS[value]
    for convert in int, float:
        try:
            return convert(value)
        except ValueError:
            pass
    if value[:1] == value[-1:] and value[0] in "\"'":
        value = value[1:-1]
    return str(value)


def parse_converter_args(argstr: str) -> tuple[tuple[t.Any, ...], dict[str, t.Any]]:
    argstr += ","
    args = []
    kwargs = {}
    position = 0

    for item in _converter_args_re.finditer(argstr):
        if item.start() != position:
            raise ValueError(
                f"Cannot parse converter argument '{argstr[position:item.start()]}'"
            )

        value = item.group("stringval")
        if value is None:
            value = item.group("value")
        value = _pythonize(value)
        if not item.group("name"):
            args.append(value)
        else:
            name = item.group("name")
            kwargs[name] = value
        position = item.end()

    return tuple(args), kwargs


class RuleFactory:
    """As soon as you have more complex URL setups it's a good idea to use rule
    factories to avoid repetitive tasks.  Some of them are builtin, others can
    be added by subclassing `RuleFactory` and overriding `get_rules`.
    """

    def get_rules(self, map: Map) -> t.Iterable[Rule]:
        """Subclasses of `RuleFactory` have to override this method and return
        an iterable of rules."""
        raise NotImplementedError()


class Subdomain(RuleFactory):
    """All URLs provided by this factory have the subdomain set to a
    specific domain. For example if you want to use the subdomain for
    the current language this can be a good setup::

        url_map = Map([
            Rule('/', endpoint='#select_language'),
            Subdomain('<string(length=2):lang_code>', [
                Rule('/', endpoint='index'),
                Rule('/about', endpoint='about'),
                Rule('/help', endpoint='help')
            ])
        ])

    All the rules except for the ``'#select_language'`` endpoint will now
    listen on a two letter long subdomain that holds the language code
    for the current request.
    """

    def __init__(self, subdomain: str, rules: t.Iterable[RuleFactory]) -> None:
        self.subdomain = subdomain
        self.rules = rules

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.subdomain = self.subdomain
                yield rule


class Submount(RuleFactory):
    """Like `Subdomain` but prefixes the URL rule with a given string::

        url_map = Map([
            Rule('/', endpoint='index'),
            Submount('/blog', [
                Rule('/', endpoint='blog/index'),
                Rule('/entry/<entry_slug>', endpoint='blog/show')
            ])
        ])

    Now the rule ``'blog/show'`` matches ``/blog/entry/<entry_slug>``.
    """

    def __init__(self, path: str, rules: t.Iterable[RuleFactory]) -> None:
        self.path = path.rstrip("/")
        self.rules = rules

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.rule = self.path + rule.rule
                yield rule


class EndpointPrefix(RuleFactory):
    """Prefixes all endpoints (which must be strings for this factory) with
    another string. This can be useful for sub applications::

        url_map = Map([
            Rule('/', endpoint='index'),
            EndpointPrefix('blog/', [Submount('/blog', [
                Rule('/', endpoint='index'),
                Rule('/entry/<entry_slug>', endpoint='show')
            ])])
        ])
    """

    def __init__(self, prefix: str, rules: t.Iterable[RuleFactory]) -> None:
        self.prefix = prefix
        self.rules = rules

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.endpoint = self.prefix + rule.endpoint
                yield rule


class RuleTemplate:
    """Returns copies of the rules wrapped and expands string templates in
    the endpoint, rule, defaults or subdomain sections.

    Here a small example for such a rule template::

        from werkzeug.routing import Map, Rule, RuleTemplate

        resource = RuleTemplate([
            Rule('/$name/', endpoint='$name.list'),
            Rule('/$name/<int:id>', endpoint='$name.show')
        ])

        url_map = Map([resource(name='user'), resource(name='page')])

    When a rule template is called the keyword arguments are used to
    replace the placeholders in all the string parameters.
    """

    def __init__(self, rules: t.Iterable[Rule]) -> None:
        self.rules = list(rules)

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> RuleTemplateFactory:
        return RuleTemplateFactory(self.rules, dict(*args, **kwargs))


class RuleTemplateFactory(RuleFactory):
    """A factory that fills in template variables into rules.  Used by
    `RuleTemplate` internally.

    :internal:
    """

    def __init__(
        self, rules: t.Iterable[RuleFactory], context: dict[str, t.Any]
    ) -> None:
        self.rules = rules
        self.context = context

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                new_defaults = subdomain = None
                if rule.defaults:
                    new_defaults = {}
                    for key, value in rule.defaults.items():
                        if isinstance(value, str):
                            value = Template(value).substitute(self.context)
                        new_defaults[key] = value
                if rule.subdomain is not None:
                    subdomain = Template(rule.subdomain).substitute(self.context)
                new_endpoint = rule.endpoint
                if isinstance(new_endpoint, str):
                    new_endpoint = Template(new_endpoint).substitute(self.context)
                yield Rule(
                    Template(rule.rule).substitute(self.context),
                    new_defaults,
                    subdomain,
                    rule.methods,
                    rule.build_only,
                    new_endpoint,
                    rule.strict_slashes,
                )


_ASTT = t.TypeVar("_ASTT", bound=ast.AST)


def _prefix_names(src: str, expected_type: type[_ASTT]) -> _ASTT:
    """ast parse and prefix names with `.` to avoid collision with user vars"""
    tree: ast.AST = ast.parse(src).body[0]
    if isinstance(tree, ast.Expr):
        tree = tree.value
    if not isinstance(tree, expected_type):
        raise TypeError(
            f"AST node is of type {type(tree).__name__}, not {expected_type.__name__}"
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = f".{node.id}"
    return tree


_CALL_CONVERTER_CODE_FMT = "self._converters[{elem!r}].to_url()"
_IF_KWARGS_URL_ENCODE_CODE = """\
if kwargs:
    params = self._encode_query_vars(kwargs)
    q = "?" if params else ""
else:
    q = params = ""
"""
_IF_KWARGS_URL_ENCODE_AST = _prefix_names(_IF_KWARGS_URL_ENCODE_CODE, ast.If)
_URL_ENCODE_AST_NAMES = (
    _prefix_names("q", ast.Name),
    _prefix_names("params", ast.Name),
)


class Rule(RuleFactory):
    """A Rule represents one URL pattern.  There are some options for `Rule`
    that change the way it behaves and are passed to the `Rule` constructor.
    Note that besides the rule-string all arguments *must* be keyword arguments
    in order to not break the application on Werkzeug upgrades.

    `string`
        Rule strings basically are just normal URL paths with placeholders in
        the format ``<converter(arguments):name>`` where the converter and the
        arguments are optional.  If no converter is defined the `default`
        converter is used which means `string` in the normal configuration.

        URL rules that end with a slash are branch URLs, others are leaves.
        If you have `strict_slashes` enabled (which is the default), all
        branch URLs that are matched without a trailing slash will trigger a
        redirect to the same URL with the missing slash appended.

        The converters are defined on the `Map`.

    `endpoint`
        The endpoint for this rule. This can be anything. A reference to a
        function, a string, a number etc.  The preferred way is using a string
        because the endpoint is used for URL generation.

    `defaults`
        An optional dict with defaults for other rules with the same endpoint.
        This is a bit tricky but useful if you want to have unique URLs::

            url_map = Map([
                Rule('/all/', defaults={'page': 1}, endpoint='all_entries'),
                Rule('/all/page/<int:page>', endpoint='all_entries')
            ])

        If a user now visits ``http://example.com/all/page/1`` they will be
        redirected to ``http://example.com/all/``.  If `redirect_defaults` is
        disabled on the `Map` instance this will only affect the URL
        generation.

    `subdomain`
        The subdomain rule string for this rule. If not specified the rule
        only matches for the `default_subdomain` of the map.  If the map is
        not bound to a subdomain this feature is disabled.

        Can be useful if you want to have user profiles on different subdomains
        and all subdomains are forwarded to your application::

            url_map = Map([
                Rule('/', subdomain='<username>', endpoint='user/homepage'),
                Rule('/stats', subdomain='<username>', endpoint='user/stats')
            ])

    `methods`
        A sequence of http methods this rule applies to.  If not specified, all
        methods are allowed. For example this can be useful if you want different
        endpoints for `POST` and `GET`.  If methods are defined and the path
        matches but the method matched against is not in this list or in the
        list of another rule for that path the error raised is of the type
        `MethodNotAllowed` rather than `NotFound`.  If `GET` is present in the
        list of methods and `HEAD` is not, `HEAD` is added automatically.

    `strict_slashes`
        Override the `Map` setting for `strict_slashes` only for this rule. If
        not specified the `Map` setting is used.

    `merge_slashes`
        Override :attr:`Map.merge_slashes` for this rule.

    `build_only`
        Set this to True and the rule will never match but will create a URL
        that can be build. This is useful if you have resources on a subdomain
        or folder that are not handled by the WSGI application (like static data)

    `redirect_to`
        If given this must be either a string or callable.  In case of a
        callable it's called with the url adapter that triggered the match and
        the values of the URL as keyword arguments and has to return the target
        for the redirect, otherwise it has to be a string with placeholders in
        rule syntax::

            def foo_with_slug(adapter, id):
                # ask the database for the slug for the old id.  this of
                # course has nothing to do with werkzeug.
                return f'foo/{Foo.get_slug_for_id(id)}'

            url_map = Map([
                Rule('/foo/<slug>', endpoint='foo'),
                Rule('/some/old/url/<slug>', redirect_to='foo/<slug>'),
                Rule('/other/old/url/<int:id>', redirect_to=foo_with_slug)
            ])

        When the rule is matched the routing system will raise a
        `RequestRedirect` exception with the target for the redirect.

        Keep in mind that the URL will be joined against the URL root of the
        script so don't use a leading slash on the target URL unless you
        really mean root of that domain.

    `alias`
        If enabled this rule serves as an alias for another rule with the same
        endpoint and arguments.

    `host`
        If provided and the URL map has host matching enabled this can be
        used to provide a match rule for the whole host.  This also means
        that the subdomain feature is disabled.

    `websocket`
        If ``True``, this rule is only matches for WebSocket (``ws://``,
        ``wss://``) requests. By default, rules will only match for HTTP
        requests.

    .. versionchanged:: 2.1
        Percent-encoded newlines (``%0a``), which are decoded by WSGI
        servers, are considered when routing instead of terminating the
        match early.

    .. versionadded:: 1.0
        Added ``websocket``.

    .. versionadded:: 1.0
        Added ``merge_slashes``.

    .. versionadded:: 0.7
        Added ``alias`` and ``host``.

    .. versionchanged:: 0.6.1
       ``HEAD`` is added to ``methods`` if ``GET`` is present.
    """

    def __init__(
        self,
        string: str,
        defaults: t.Mapping[str, t.Any] | None = None,
        subdomain: str | None = None,
        methods: t.Iterable[str] | None = None,
        build_only: bool = False,
        endpoint: t.Any | None = None,
        strict_slashes: bool | None = None,
        merge_slashes: bool | None = None,
        redirect_to: str | t.Callable[..., str] | None = None,
        alias: bool = False,
        host: str | None = None,
        websocket: bool = False,
    ) -> None:
        if not string.startswith("/"):
            raise ValueError(f"URL rule '{string}' must start with a slash.")

        self.rule = string
        self.is_leaf = not string.endswith("/")
        self.is_branch = string.endswith("/")

        self.map: Map = None  # type: ignore
        self.strict_slashes = strict_slashes
        self.merge_slashes = merge_slashes
        self.subdomain = subdomain
        self.host = host
        self.defaults = defaults
        self.build_only = build_only
        self.alias = alias
        self.websocket = websocket

        if methods is not None:
            if isinstance(methods, str):
                raise TypeError("'methods' should be a list of strings.")

            methods = {x.upper() for x in methods}

            if "HEAD" not in methods and "GET" in methods:
                methods.add("HEAD")

            if websocket and methods - {"GET", "HEAD", "OPTIONS"}:
                raise ValueError(
                    "WebSocket rules can only use 'GET', 'HEAD', and 'OPTIONS' methods."
                )

        self.methods = methods
        self.endpoint: t.Any = endpoint
        self.redirect_to = redirect_to

        if defaults:
            self.arguments = set(map(str, defaults))
        else:
            self.arguments = set()

        self._converters: dict[str, BaseConverter] = {}
        self._trace: list[tuple[bool, str]] = []
        self._parts: list[RulePart] = []

    def empty(self) -> Rule:
        """
        Return an unbound copy of this rule.

        This can be useful if want to reuse an already bound URL for another
        map.  See ``get_empty_kwargs`` to override what keyword arguments are
        provided to the new copy.
        """
        return type(self)(self.rule, **self.get_empty_kwargs())

    def get_empty_kwargs(self) -> t.Mapping[str, t.Any]:
        """
        Provides kwargs for instantiating empty copy with empty()

        Use this method to provide custom keyword arguments to the subclass of
        ``Rule`` when calling ``some_rule.empty()``.  Helpful when the subclass
        has custom keyword arguments that are needed at instantiation.

        Must return a ``dict`` that will be provided as kwargs to the new
        instance of ``Rule``, following the initial ``self.rule`` value which
        is always provided as the first, required positional argument.
        """
        defaults = None
        if self.defaults:
            defaults = dict(self.defaults)
        return dict(
            defaults=defaults,
            subdomain=self.subdomain,
            methods=self.methods,
            build_only=self.build_only,
            endpoint=self.endpoint,
            strict_slashes=self.strict_slashes,
            redirect_to=self.redirect_to,
            alias=self.alias,
            host=self.host,
        )

    def get_rules(self, map: Map) -> t.Iterator[Rule]:
        yield self

    def refresh(self) -> None:
        """Rebinds and refreshes the URL.  Call this if you modified the
        rule in place.

        :internal:
        """
        self.bind(self.map, rebind=True)

    def bind(self, map: Map, rebind: bool = False) -> None:
        """Bind the url to a map and create a regular expression based on
        the information from the rule itself and the defaults from the map.

        :internal:
        """
        if self.map is not None and not rebind:
            raise RuntimeError(f"url rule {self!r} already bound to map {self.map!r}")
        self.map = map
        if self.strict_slashes is None:
            self.strict_slashes = map.strict_slashes
        if self.merge_slashes is None:
            self.merge_slashes = map.merge_slashes
        if self.subdomain is None:
            self.subdomain = map.default_subdomain
        self.compile()

    def get_converter(
        self,
        variable_name: str,
        converter_name: str,
        args: tuple[t.Any, ...],
        kwargs: t.Mapping[str, t.Any],
    ) -> BaseConverter:
        """Looks up the converter for the given parameter.

        .. versionadded:: 0.9
        """
        if converter_name not in self.map.converters:
            raise LookupError(f"the converter {converter_name!r} does not exist")
        return self.map.converters[converter_name](self.map, *args, **kwargs)

    def _encode_query_vars(self, query_vars: t.Mapping[str, t.Any]) -> str:
        items: t.Iterable[tuple[str, str]] = iter_multi_items(query_vars)

        if self.map.sort_parameters:
            items = sorted(items, key=self.map.sort_key)

        return _urlencode(items)

    def _parse_rule(self, rule: str) -> t.Iterable[RulePart]:
        content = ""
        static = True
        argument_weights = []
        static_weights: list[tuple[int, int]] = []
        final = False
        convertor_number = 0

        pos = 0
        while pos < len(rule):
            match = _part_re.match(rule, pos)
            if match is None:
                raise ValueError(f"malformed url rule: {rule!r}")

            data = match.groupdict()
            if data["static"] is not None:
                static_weights.append((len(static_weights), -len(data["static"])))
                self._trace.append((False, data["static"]))
                content += data["static"] if static else re.escape(data["static"])

            if data["variable"] is not None:
                if static:
                    # Switching content to represent regex, hence the need to escape
                    content = re.escape(content)
                static = False
                c_args, c_kwargs = parse_converter_args(data["arguments"] or "")
                convobj = self.get_converter(
                    data["variable"], data["converter"] or "default", c_args, c_kwargs
                )
                self._converters[data["variable"]] = convobj
                self.arguments.add(data["variable"])
                if not convobj.part_isolating:
                    final = True
                content += f"(?P<__werkzeug_{convertor_number}>{convobj.regex})"
                convertor_number += 1
                argument_weights.append(convobj.weight)
                self._trace.append((True, data["variable"]))

            if data["slash"] is not None:
                self._trace.append((False, "/"))
                if final:
                    content += "/"
                else:
                    if not static:
                        content += r"\Z"
                    weight = Weighting(
                        -len(static_weights),
                        static_weights,
                        -len(argument_weights),
                        argument_weights,
                    )
                    yield RulePart(
                        content=content,
                        final=final,
                        static=static,
                        suffixed=False,
                        weight=weight,
                    )
                    content = ""
                    static = True
                    argument_weights = []
                    static_weights = []
                    final = False
                    convertor_number = 0

            pos = match.end()

        suffixed = False
        if final and content[-1] == "/":
            # If a converter is part_isolating=False (matches slashes) and ends with a
            # slash, augment the regex to support slash redirects.
            suffixed = True
            content = content[:-1] + "(?<!/)(/?)"
        if not static:
            content += r"\Z"
        weight = Weighting(
            -len(static_weights),
            static_weights,
            -len(argument_weights),
            argument_weights,
        )
        yield RulePart(
            content=content,
            final=final,
            static=static,
            suffixed=suffixed,
            weight=weight,
        )
        if suffixed:
            yield RulePart(
                content="", final=False, static=True, suffixed=False, weight=weight
            )

    def compile(self) -> None:
        """Compiles the regular expression and stores it."""
        assert self.map is not None, "rule not bound"

        if self.map.host_matching:
            domain_rule = self.host or ""
        else:
            domain_rule = self.subdomain or ""
        self._parts = []
        self._trace = []
        self._converters = {}
        if domain_rule == "":
            self._parts = [
                RulePart(
                    content="",
                    final=False,
                    static=True,
                    suffixed=False,
                    weight=Weighting(0, [], 0, []),
                )
            ]
        else:
            self._parts.extend(self._parse_rule(domain_rule))
        self._trace.append((False, "|"))
        rule = self.rule
        if self.merge_slashes:
            rule = re.sub("/{2,}?", "/", self.rule)
        self._parts.extend(self._parse_rule(rule))

        self._build: t.Callable[..., tuple[str, str]]
        self._build = self._compile_builder(False).__get__(self, None)
        self._build_unknown: t.Callable[..., tuple[str, str]]
        self._build_unknown = self._compile_builder(True).__get__(self, None)

    @staticmethod
    def _get_func_code(code: CodeType, name: str) -> t.Callable[..., tuple[str, str]]:
        globs: dict[str, t.Any] = {}
        locs: dict[str, t.Any] = {}
        exec(code, globs, locs)
        return locs[name]  # type: ignore

    def _compile_builder(
        self, append_unknown: bool = True
    ) -> t.Callable[..., tuple[str, str]]:
        defaults = self.defaults or {}
        dom_ops: list[tuple[bool, str]] = []
        url_ops: list[tuple[bool, str]] = []

        opl = dom_ops
        for is_dynamic, data in self._trace:
            if data == "|" and opl is dom_ops:
                opl = url_ops
                continue
            # this seems like a silly case to ever come up but:
            # if a default is given for a value that appears in the rule,
            # resolve it to a constant ahead of time
            if is_dynamic and data in defaults:
                data = self._converters[data].to_url(defaults[data])
                opl.append((False, data))
            elif not is_dynamic:
                # safe = https://url.spec.whatwg.org/#url-path-segment-string
                opl.append((False, quote(data, safe="!$&'()*+,/:;=@")))
            else:
                opl.append((True, data))

        def _convert(elem: str) -> ast.Call:
            ret = _prefix_names(_CALL_CONVERTER_CODE_FMT.format(elem=elem), ast.Call)
            ret.args = [ast.Name(elem, ast.Load())]
            return ret

        def _parts(ops: list[tuple[bool, str]]) -> list[ast.expr]:
            parts: list[ast.expr] = [
                _convert(elem) if is_dynamic else ast.Constant(elem)
                for is_dynamic, elem in ops
            ]
            parts = parts or [ast.Constant("")]
            # constant fold
            ret = [parts[0]]
            for p in parts[1:]:
                if isinstance(p, ast.Constant) and isinstance(ret[-1], ast.Constant):
                    ret[-1] = ast.Constant(ret[-1].value + p.value)
                else:
                    ret.append(p)
            return ret

        dom_parts = _parts(dom_ops)
        url_parts = _parts(url_ops)
        body: list[ast.stmt]
        if not append_unknown:
            body = []
        else:
            body = [_IF_KWARGS_URL_ENCODE_AST]
            url_parts.extend(_URL_ENCODE_AST_NAMES)

        def _join(parts: list[ast.expr]) -> ast.expr:
            if len(parts) == 1:  # shortcut
                return parts[0]
            return ast.JoinedStr(parts)

        body.append(
            ast.Return(ast.Tuple([_join(dom_parts), _join(url_parts)], ast.Load()))
        )

        pargs = [
            elem
            for is_dynamic, elem in dom_ops + url_ops
            if is_dynamic and elem not in defaults
        ]
        kargs = [str(k) for k in defaults]

        func_ast = _prefix_names("def _(): pass", ast.FunctionDef)
        func_ast.name = f"<builder:{self.rule!r}>"
        func_ast.args.args.append(ast.arg(".self", None))
        for arg in pargs + kargs:
            func_ast.args.args.append(ast.arg(arg, None))
        func_ast.args.kwarg = ast.arg(".kwargs", None)
        for _ in kargs:
            func_ast.args.defaults.append(ast.Constant(""))
        func_ast.body = body

        # Use `ast.parse` instead of `ast.Module` for better portability, since the
        # signature of `ast.Module` can change.
        module = ast.parse("")
        module.body = [func_ast]

        # mark everything as on line 1, offset 0
        # less error-prone than `ast.fix_missing_locations`
        # bad line numbers cause an assert to fail in debug builds
        for node in ast.walk(module):
            if "lineno" in node._attributes:
                node.lineno = 1  # type: ignore[attr-defined]
            if "end_lineno" in node._attributes:
                node.end_lineno = node.lineno  # type: ignore[attr-defined]
            if "col_offset" in node._attributes:
                node.col_offset = 0  # type: ignore[attr-defined]
            if "end_col_offset" in node._attributes:
                node.end_col_offset = node.col_offset  # type: ignore[attr-defined]

        code = compile(module, "<werkzeug routing>", "exec")
        return self._get_func_code(code, func_ast.name)

    def build(
        self, values: t.Mapping[str, t.Any], append_unknown: bool = True
    ) -> tuple[str, str] | None:
        """Assembles the relative url for that rule and the subdomain.
        If building doesn't work for some reasons `None` is returned.

        :internal:
        """
        try:
            if append_unknown:
                return self._build_unknown(**values)
            else:
                return self._build(**values)
        except ValidationError:
            return None

    def provides_defaults_for(self, rule: Rule) -> bool:
        """Check if this rule has defaults for a given rule.

        :internal:
        """
        return bool(
            not self.build_only
            and self.defaults
            and self.endpoint == rule.endpoint
            and self != rule
            and self.arguments == rule.arguments
        )

    def suitable_for(
        self, values: t.Mapping[str, t.Any], method: str | None = None
    ) -> bool:
        """Check if the dict of values has enough data for url generation.

        :internal:
        """
        # if a method was given explicitly and that method is not supported
        # by this rule, this rule is not suitable.
        if (
            method is not None
            and self.methods is not None
            and method not in self.methods
        ):
            return False

        defaults = self.defaults or ()

        # all arguments required must be either in the defaults dict or
        # the value dictionary otherwise it's not suitable
        for key in self.arguments:
            if key not in defaults and key not in values:
                return False

        # in case defaults are given we ensure that either the value was
        # skipped or the value is the same as the default value.
        if defaults:
            for key, value in defaults.items():
                if key in values and value != values[key]:
                    return False

        return True

    def build_compare_key(self) -> tuple[int, int, int]:
        """The build compare key for sorting.

        :internal:
        """
        return (1 if self.alias else 0, -len(self.arguments), -len(self.defaults or ()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._trace == other._trace

    __hash__ = None  # type: ignore

    def __str__(self) -> str:
        return self.rule

    def __repr__(self) -> str:
        if self.map is None:
            return f"<{type(self).__name__} (unbound)>"
        parts = []
        for is_dynamic, data in self._trace:
            if is_dynamic:
                parts.append(f"<{data}>")
            else:
                parts.append(data)
        parts_str = "".join(parts).lstrip("|")
        methods = f" ({', '.join(self.methods)})" if self.methods is not None else ""
        return f"<{type(self).__name__} {parts_str!r}{methods} -> {self.endpoint}>"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\url.py ===
# engine/url.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Provides the :class:`~sqlalchemy.engine.url.URL` class which encapsulates
information about a database connection specification.

The URL object is created automatically when
:func:`~sqlalchemy.engine.create_engine` is called with a string
argument; alternatively, the URL is a public-facing construct which can
be used directly and is also accepted directly by ``create_engine()``.
"""

from __future__ import annotations

import collections.abc as collections_abc
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import quote_plus
from urllib.parse import unquote

from .interfaces import Dialect
from .. import exc
from .. import util
from ..dialects import plugins
from ..dialects import registry


class URL(NamedTuple):
    """
    Represent the components of a URL used to connect to a database.

    URLs are typically constructed from a fully formatted URL string, where the
    :func:`.make_url` function is used internally by the
    :func:`_sa.create_engine` function in order to parse the URL string into
    its individual components, which are then used to construct a new
    :class:`.URL` object. When parsing from a formatted URL string, the parsing
    format generally follows
    `RFC-1738 <https://www.ietf.org/rfc/rfc1738.txt>`_, with some exceptions.

    A :class:`_engine.URL` object may also be produced directly, either by
    using the :func:`.make_url` function with a fully formed URL string, or
    by using the :meth:`_engine.URL.create` constructor in order
    to construct a :class:`_engine.URL` programmatically given individual
    fields. The resulting :class:`.URL` object may be passed directly to
    :func:`_sa.create_engine` in place of a string argument, which will bypass
    the usage of :func:`.make_url` within the engine's creation process.

    .. versionchanged:: 1.4

        The :class:`_engine.URL` object is now an immutable object.  To
        create a URL, use the :func:`_engine.make_url` or
        :meth:`_engine.URL.create` function / method.  To modify
        a :class:`_engine.URL`, use methods like
        :meth:`_engine.URL.set` and
        :meth:`_engine.URL.update_query_dict` to return a new
        :class:`_engine.URL` object with modifications.   See notes for this
        change at :ref:`change_5526`.

    .. seealso::

        :ref:`database_urls`

    :class:`_engine.URL` contains the following attributes:

    * :attr:`_engine.URL.drivername`: database backend and driver name, such as
      ``postgresql+psycopg2``
    * :attr:`_engine.URL.username`: username string
    * :attr:`_engine.URL.password`: password string
    * :attr:`_engine.URL.host`: string hostname
    * :attr:`_engine.URL.port`: integer port number
    * :attr:`_engine.URL.database`: string database name
    * :attr:`_engine.URL.query`: an immutable mapping representing the query
      string.  contains strings for keys and either strings or tuples of
      strings for values.


    """

    drivername: str
    """database backend and driver name, such as
    ``postgresql+psycopg2``

    """

    username: Optional[str]
    "username string"

    password: Optional[str]
    """password, which is normally a string but may also be any
    object that has a ``__str__()`` method."""

    host: Optional[str]
    """hostname or IP number.  May also be a data source name for some
    drivers."""

    port: Optional[int]
    """integer port number"""

    database: Optional[str]
    """database name"""

    query: util.immutabledict[str, Union[Tuple[str, ...], str]]
    """an immutable mapping representing the query string.  contains strings
       for keys and either strings or tuples of strings for values, e.g.::

            >>> from sqlalchemy.engine import make_url
            >>> url = make_url(
            ...     "postgresql+psycopg2://user:pass@host/dbname?alt_host=host1&alt_host=host2&ssl_cipher=%2Fpath%2Fto%2Fcrt"
            ... )
            >>> url.query
            immutabledict({'alt_host': ('host1', 'host2'), 'ssl_cipher': '/path/to/crt'})

         To create a mutable copy of this mapping, use the ``dict`` constructor::

            mutable_query_opts = dict(url.query)

       .. seealso::

          :attr:`_engine.URL.normalized_query` - normalizes all values into sequences
          for consistent processing

          Methods for altering the contents of :attr:`_engine.URL.query`:

          :meth:`_engine.URL.update_query_dict`

          :meth:`_engine.URL.update_query_string`

          :meth:`_engine.URL.update_query_pairs`

          :meth:`_engine.URL.difference_update_query`

    """  # noqa: E501

    @classmethod
    def create(
        cls,
        drivername: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        query: Mapping[str, Union[Sequence[str], str]] = util.EMPTY_DICT,
    ) -> URL:
        """Create a new :class:`_engine.URL` object.

        .. seealso::

            :ref:`database_urls`

        :param drivername: the name of the database backend. This name will
          correspond to a module in sqlalchemy/databases or a third party
          plug-in.
        :param username: The user name.
        :param password: database password.  Is typically a string, but may
          also be an object that can be stringified with ``str()``.

          .. note:: The password string should **not** be URL encoded when
             passed as an argument to :meth:`_engine.URL.create`; the string
             should contain the password characters exactly as they would be
             typed.

          .. note::  A password-producing object will be stringified only
             **once** per :class:`_engine.Engine` object.  For dynamic password
             generation per connect, see :ref:`engines_dynamic_tokens`.

        :param host: The name of the host.
        :param port: The port number.
        :param database: The database name.
        :param query: A dictionary of string keys to string values to be passed
          to the dialect and/or the DBAPI upon connect.   To specify non-string
          parameters to a Python DBAPI directly, use the
          :paramref:`_sa.create_engine.connect_args` parameter to
          :func:`_sa.create_engine`.   See also
          :attr:`_engine.URL.normalized_query` for a dictionary that is
          consistently string->list of string.
        :return: new :class:`_engine.URL` object.

        .. versionadded:: 1.4

            The :class:`_engine.URL` object is now an **immutable named
            tuple**.  In addition, the ``query`` dictionary is also immutable.
            To create a URL, use the :func:`_engine.url.make_url` or
            :meth:`_engine.URL.create` function/ method.  To modify a
            :class:`_engine.URL`, use the :meth:`_engine.URL.set` and
            :meth:`_engine.URL.update_query` methods.

        """

        return cls(
            cls._assert_str(drivername, "drivername"),
            cls._assert_none_str(username, "username"),
            password,
            cls._assert_none_str(host, "host"),
            cls._assert_port(port),
            cls._assert_none_str(database, "database"),
            cls._str_dict(query),
        )

    @classmethod
    def _assert_port(cls, port: Optional[int]) -> Optional[int]:
        if port is None:
            return None
        try:
            return int(port)
        except TypeError:
            raise TypeError("Port argument must be an integer or None")

    @classmethod
    def _assert_str(cls, v: str, paramname: str) -> str:
        if not isinstance(v, str):
            raise TypeError("%s must be a string" % paramname)
        return v

    @classmethod
    def _assert_none_str(
        cls, v: Optional[str], paramname: str
    ) -> Optional[str]:
        if v is None:
            return v

        return cls._assert_str(v, paramname)

    @classmethod
    def _str_dict(
        cls,
        dict_: Optional[
            Union[
                Sequence[Tuple[str, Union[Sequence[str], str]]],
                Mapping[str, Union[Sequence[str], str]],
            ]
        ],
    ) -> util.immutabledict[str, Union[Tuple[str, ...], str]]:
        if dict_ is None:
            return util.EMPTY_DICT

        @overload
        def _assert_value(
            val: str,
        ) -> str: ...

        @overload
        def _assert_value(
            val: Sequence[str],
        ) -> Union[str, Tuple[str, ...]]: ...

        def _assert_value(
            val: Union[str, Sequence[str]],
        ) -> Union[str, Tuple[str, ...]]:
            if isinstance(val, str):
                return val
            elif isinstance(val, collections_abc.Sequence):
                return tuple(_assert_value(elem) for elem in val)
            else:
                raise TypeError(
                    "Query dictionary values must be strings or "
                    "sequences of strings"
                )

        def _assert_str(v: str) -> str:
            if not isinstance(v, str):
                raise TypeError("Query dictionary keys must be strings")
            return v

        dict_items: Iterable[Tuple[str, Union[Sequence[str], str]]]
        if isinstance(dict_, collections_abc.Sequence):
            dict_items = dict_
        else:
            dict_items = dict_.items()

        return util.immutabledict(
            {
                _assert_str(key): _assert_value(
                    value,
                )
                for key, value in dict_items
            }
        )

    def set(
        self,
        drivername: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        query: Optional[Mapping[str, Union[Sequence[str], str]]] = None,
    ) -> URL:
        """return a new :class:`_engine.URL` object with modifications.

        Values are used if they are non-None.  To set a value to ``None``
        explicitly, use the :meth:`_engine.URL._replace` method adapted
        from ``namedtuple``.

        :param drivername: new drivername
        :param username: new username
        :param password: new password
        :param host: new hostname
        :param port: new port
        :param query: new query parameters, passed a dict of string keys
         referring to string or sequence of string values.  Fully
         replaces the previous list of arguments.

        :return: new :class:`_engine.URL` object.

        .. versionadded:: 1.4

        .. seealso::

            :meth:`_engine.URL.update_query_dict`

        """

        kw: Dict[str, Any] = {}
        if drivername is not None:
            kw["drivername"] = drivername
        if username is not None:
            kw["username"] = username
        if password is not None:
            kw["password"] = password
        if host is not None:
            kw["host"] = host
        if port is not None:
            kw["port"] = port
        if database is not None:
            kw["database"] = database
        if query is not None:
            kw["query"] = query

        return self._assert_replace(**kw)

    def _assert_replace(self, **kw: Any) -> URL:
        """argument checks before calling _replace()"""

        if "drivername" in kw:
            self._assert_str(kw["drivername"], "drivername")
        for name in "username", "host", "database":
            if name in kw:
                self._assert_none_str(kw[name], name)
        if "port" in kw:
            self._assert_port(kw["port"])
        if "query" in kw:
            kw["query"] = self._str_dict(kw["query"])

        return self._replace(**kw)

    def update_query_string(
        self, query_string: str, append: bool = False
    ) -> URL:
        """Return a new :class:`_engine.URL` object with the :attr:`_engine.URL.query`
        parameter dictionary updated by the given query string.

        E.g.::

            >>> from sqlalchemy.engine import make_url
            >>> url = make_url("postgresql+psycopg2://user:pass@host/dbname")
            >>> url = url.update_query_string(
            ...     "alt_host=host1&alt_host=host2&ssl_cipher=%2Fpath%2Fto%2Fcrt"
            ... )
            >>> str(url)
            'postgresql+psycopg2://user:pass@host/dbname?alt_host=host1&alt_host=host2&ssl_cipher=%2Fpath%2Fto%2Fcrt'

        :param query_string: a URL escaped query string, not including the
         question mark.

        :param append: if True, parameters in the existing query string will
         not be removed; new parameters will be in addition to those present.
         If left at its default of False, keys present in the given query
         parameters will replace those of the existing query string.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_engine.URL.query`

            :meth:`_engine.URL.update_query_dict`

        """  # noqa: E501
        return self.update_query_pairs(parse_qsl(query_string), append=append)

    def update_query_pairs(
        self,
        key_value_pairs: Iterable[Tuple[str, Union[str, List[str]]]],
        append: bool = False,
    ) -> URL:
        """Return a new :class:`_engine.URL` object with the
        :attr:`_engine.URL.query`
        parameter dictionary updated by the given sequence of key/value pairs

        E.g.::

            >>> from sqlalchemy.engine import make_url
            >>> url = make_url("postgresql+psycopg2://user:pass@host/dbname")
            >>> url = url.update_query_pairs(
            ...     [
            ...         ("alt_host", "host1"),
            ...         ("alt_host", "host2"),
            ...         ("ssl_cipher", "/path/to/crt"),
            ...     ]
            ... )
            >>> str(url)
            'postgresql+psycopg2://user:pass@host/dbname?alt_host=host1&alt_host=host2&ssl_cipher=%2Fpath%2Fto%2Fcrt'

        :param key_value_pairs: A sequence of tuples containing two strings
         each.

        :param append: if True, parameters in the existing query string will
         not be removed; new parameters will be in addition to those present.
         If left at its default of False, keys present in the given query
         parameters will replace those of the existing query string.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_engine.URL.query`

            :meth:`_engine.URL.difference_update_query`

            :meth:`_engine.URL.set`

        """  # noqa: E501

        existing_query = self.query
        new_keys: Dict[str, Union[str, List[str]]] = {}

        for key, value in key_value_pairs:
            if key in new_keys:
                new_keys[key] = util.to_list(new_keys[key])
                cast("List[str]", new_keys[key]).append(cast(str, value))
            else:
                new_keys[key] = (
                    list(value) if isinstance(value, (list, tuple)) else value
                )

        new_query: Mapping[str, Union[str, Sequence[str]]]
        if append:
            new_query = {}

            for k in new_keys:
                if k in existing_query:
                    new_query[k] = tuple(
                        util.to_list(existing_query[k])
                        + util.to_list(new_keys[k])
                    )
                else:
                    new_query[k] = new_keys[k]

            new_query.update(
                {
                    k: existing_query[k]
                    for k in set(existing_query).difference(new_keys)
                }
            )
        else:
            new_query = self.query.union(
                {
                    k: tuple(v) if isinstance(v, list) else v
                    for k, v in new_keys.items()
                }
            )
        return self.set(query=new_query)

    def update_query_dict(
        self,
        query_parameters: Mapping[str, Union[str, List[str]]],
        append: bool = False,
    ) -> URL:
        """Return a new :class:`_engine.URL` object with the
        :attr:`_engine.URL.query` parameter dictionary updated by the given
        dictionary.

        The dictionary typically contains string keys and string values.
        In order to represent a query parameter that is expressed multiple
        times, pass a sequence of string values.

        E.g.::


            >>> from sqlalchemy.engine import make_url
            >>> url = make_url("postgresql+psycopg2://user:pass@host/dbname")
            >>> url = url.update_query_dict(
            ...     {"alt_host": ["host1", "host2"], "ssl_cipher": "/path/to/crt"}
            ... )
            >>> str(url)
            'postgresql+psycopg2://user:pass@host/dbname?alt_host=host1&alt_host=host2&ssl_cipher=%2Fpath%2Fto%2Fcrt'


        :param query_parameters: A dictionary with string keys and values
         that are either strings, or sequences of strings.

        :param append: if True, parameters in the existing query string will
         not be removed; new parameters will be in addition to those present.
         If left at its default of False, keys present in the given query
         parameters will replace those of the existing query string.


        .. versionadded:: 1.4

        .. seealso::

            :attr:`_engine.URL.query`

            :meth:`_engine.URL.update_query_string`

            :meth:`_engine.URL.update_query_pairs`

            :meth:`_engine.URL.difference_update_query`

            :meth:`_engine.URL.set`

        """  # noqa: E501
        return self.update_query_pairs(query_parameters.items(), append=append)

    def difference_update_query(self, names: Iterable[str]) -> URL:
        """
        Remove the given names from the :attr:`_engine.URL.query` dictionary,
        returning the new :class:`_engine.URL`.

        E.g.::

            url = url.difference_update_query(["foo", "bar"])

        Equivalent to using :meth:`_engine.URL.set` as follows::

            url = url.set(
                query={
                    key: url.query[key]
                    for key in set(url.query).difference(["foo", "bar"])
                }
            )

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_engine.URL.query`

            :meth:`_engine.URL.update_query_dict`

            :meth:`_engine.URL.set`

        """

        if not set(names).intersection(self.query):
            return self

        return URL(
            self.drivername,
            self.username,
            self.password,
            self.host,
            self.port,
            self.database,
            util.immutabledict(
                {
                    key: self.query[key]
                    for key in set(self.query).difference(names)
                }
            ),
        )

    @property
    def normalized_query(self) -> Mapping[str, Sequence[str]]:
        """Return the :attr:`_engine.URL.query` dictionary with values normalized
        into sequences.

        As the :attr:`_engine.URL.query` dictionary may contain either
        string values or sequences of string values to differentiate between
        parameters that are specified multiple times in the query string,
        code that needs to handle multiple parameters generically will wish
        to use this attribute so that all parameters present are presented
        as sequences.   Inspiration is from Python's ``urllib.parse.parse_qs``
        function.  E.g.::


            >>> from sqlalchemy.engine import make_url
            >>> url = make_url(
            ...     "postgresql+psycopg2://user:pass@host/dbname?alt_host=host1&alt_host=host2&ssl_cipher=%2Fpath%2Fto%2Fcrt"
            ... )
            >>> url.query
            immutabledict({'alt_host': ('host1', 'host2'), 'ssl_cipher': '/path/to/crt'})
            >>> url.normalized_query
            immutabledict({'alt_host': ('host1', 'host2'), 'ssl_cipher': ('/path/to/crt',)})

        """  # noqa: E501

        return util.immutabledict(
            {
                k: (v,) if not isinstance(v, tuple) else v
                for k, v in self.query.items()
            }
        )

    @util.deprecated(
        "1.4",
        "The :meth:`_engine.URL.__to_string__ method is deprecated and will "
        "be removed in a future release.  Please use the "
        ":meth:`_engine.URL.render_as_string` method.",
    )
    def __to_string__(self, hide_password: bool = True) -> str:
        """Render this :class:`_engine.URL` object as a string.

        :param hide_password: Defaults to True.   The password is not shown
         in the string unless this is set to False.

        """
        return self.render_as_string(hide_password=hide_password)

    def render_as_string(self, hide_password: bool = True) -> str:
        """Render this :class:`_engine.URL` object as a string.

        This method is used when the ``__str__()`` or ``__repr__()``
        methods are used.   The method directly includes additional options.

        :param hide_password: Defaults to True.   The password is not shown
         in the string unless this is set to False.

        """
        s = self.drivername + "://"
        if self.username is not None:
            s += quote(self.username, safe=" +")
            if self.password is not None:
                s += ":" + (
                    "***"
                    if hide_password
                    else quote(str(self.password), safe=" +")
                )
            s += "@"
        if self.host is not None:
            if ":" in self.host:
                s += f"[{self.host}]"
            else:
                s += self.host
        if self.port is not None:
            s += ":" + str(self.port)
        if self.database is not None:
            s += "/" + self.database
        if self.query:
            keys = list(self.query)
            keys.sort()
            s += "?" + "&".join(
                f"{quote_plus(k)}={quote_plus(element)}"
                for k in keys
                for element in util.to_list(self.query[k])
            )
        return s

    def __repr__(self) -> str:
        return self.render_as_string()

    def __copy__(self) -> URL:
        return self.__class__.create(
            self.drivername,
            self.username,
            self.password,
            self.host,
            self.port,
            self.database,
            # note this is an immutabledict of str-> str / tuple of str,
            # also fully immutable.  does not require deepcopy
            self.query,
        )

    def __deepcopy__(self, memo: Any) -> URL:
        return self.__copy__()

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, URL)
            and self.drivername == other.drivername
            and self.username == other.username
            and self.password == other.password
            and self.host == other.host
            and self.database == other.database
            and self.query == other.query
            and self.port == other.port
        )

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def get_backend_name(self) -> str:
        """Return the backend name.

        This is the name that corresponds to the database backend in
        use, and is the portion of the :attr:`_engine.URL.drivername`
        that is to the left of the plus sign.

        """
        if "+" not in self.drivername:
            return self.drivername
        else:
            return self.drivername.split("+")[0]

    def get_driver_name(self) -> str:
        """Return the backend name.

        This is the name that corresponds to the DBAPI driver in
        use, and is the portion of the :attr:`_engine.URL.drivername`
        that is to the right of the plus sign.

        If the :attr:`_engine.URL.drivername` does not include a plus sign,
        then the default :class:`_engine.Dialect` for this :class:`_engine.URL`
        is imported in order to get the driver name.

        """

        if "+" not in self.drivername:
            return self.get_dialect().driver
        else:
            return self.drivername.split("+")[1]

    def _instantiate_plugins(
        self, kwargs: Mapping[str, Any]
    ) -> Tuple[URL, List[Any], Dict[str, Any]]:
        plugin_names = util.to_list(self.query.get("plugin", ()))
        plugin_names += kwargs.get("plugins", [])

        kwargs = dict(kwargs)

        loaded_plugins = [
            plugins.load(plugin_name)(self, kwargs)
            for plugin_name in plugin_names
        ]

        u = self.difference_update_query(["plugin", "plugins"])

        for plugin in loaded_plugins:
            new_u = plugin.update_url(u)
            if new_u is not None:
                u = new_u

        kwargs.pop("plugins", None)

        return u, loaded_plugins, kwargs

    def _get_entrypoint(self) -> Type[Dialect]:
        """Return the "entry point" dialect class.

        This is normally the dialect itself except in the case when the
        returned class implements the get_dialect_cls() method.

        """
        if "+" not in self.drivername:
            name = self.drivername
        else:
            name = self.drivername.replace("+", ".")
        cls = registry.load(name)
        # check for legacy dialects that
        # would return a module with 'dialect' as the
        # actual class
        if (
            hasattr(cls, "dialect")
            and isinstance(cls.dialect, type)
            and issubclass(cls.dialect, Dialect)
        ):
            return cls.dialect
        else:
            return cast("Type[Dialect]", cls)

    def get_dialect(self, _is_async: bool = False) -> Type[Dialect]:
        """Return the SQLAlchemy :class:`_engine.Dialect` class corresponding
        to this URL's driver name.

        """
        entrypoint = self._get_entrypoint()
        if _is_async:
            dialect_cls = entrypoint.get_async_dialect_cls(self)
        else:
            dialect_cls = entrypoint.get_dialect_cls(self)
        return dialect_cls

    def translate_connect_args(
        self, names: Optional[List[str]] = None, **kw: Any
    ) -> Dict[str, Any]:
        r"""Translate url attributes into a dictionary of connection arguments.

        Returns attributes of this url (`host`, `database`, `username`,
        `password`, `port`) as a plain dictionary.  The attribute names are
        used as the keys by default.  Unset or false attributes are omitted
        from the final dictionary.

        :param \**kw: Optional, alternate key names for url attributes.

        :param names: Deprecated.  Same purpose as the keyword-based alternate
            names, but correlates the name to the original positionally.
        """

        if names is not None:
            util.warn_deprecated(
                "The `URL.translate_connect_args.name`s parameter is "
                "deprecated. Please pass the "
                "alternate names as kw arguments.",
                "1.4",
            )

        translated = {}
        attribute_names = ["host", "database", "username", "password", "port"]
        for sname in attribute_names:
            if names:
                name = names.pop(0)
            elif sname in kw:
                name = kw[sname]
            else:
                name = sname
            if name is not None and getattr(self, sname, False):
                if sname == "password":
                    translated[name] = str(getattr(self, sname))
                else:
                    translated[name] = getattr(self, sname)

        return translated


def make_url(name_or_url: Union[str, URL]) -> URL:
    """Given a string, produce a new URL instance.

    The format of the URL generally follows `RFC-1738
    <https://www.ietf.org/rfc/rfc1738.txt>`_, with some exceptions, including
    that underscores, and not dashes or periods, are accepted within the
    "scheme" portion.

    If a :class:`.URL` object is passed, it is returned as is.

    .. seealso::

        :ref:`database_urls`

    """

    if isinstance(name_or_url, str):
        return _parse_url(name_or_url)
    elif not isinstance(name_or_url, URL) and not hasattr(
        name_or_url, "_sqla_is_testing_if_this_is_a_mock_object"
    ):
        raise exc.ArgumentError(
            f"Expected string or URL object, got {name_or_url!r}"
        )
    else:
        return name_or_url


def _parse_url(name: str) -> URL:
    pattern = re.compile(
        r"""
            (?P<name>[\w\+]+)://
            (?:
                (?P<username>[^:/]*)
                (?::(?P<password>[^@]*))?
            @)?
            (?:
                (?:
                    \[(?P<ipv6host>[^/\?]+)\] |
                    (?P<ipv4host>[^/:\?]+)
                )?
                (?::(?P<port>[^/\?]*))?
            )?
            (?:/(?P<database>[^\?]*))?
            (?:\?(?P<query>.*))?
            """,
        re.X,
    )

    m = pattern.match(name)
    if m is not None:
        components = m.groupdict()
        query: Optional[Dict[str, Union[str, List[str]]]]
        if components["query"] is not None:
            query = {}

            for key, value in parse_qsl(components["query"]):
                if key in query:
                    query[key] = util.to_list(query[key])
                    cast("List[str]", query[key]).append(value)
                else:
                    query[key] = value
        else:
            query = None
        components["query"] = query

        if components["username"] is not None:
            components["username"] = unquote(components["username"])

        if components["password"] is not None:
            components["password"] = unquote(components["password"])

        ipv4host = components.pop("ipv4host")
        ipv6host = components.pop("ipv6host")
        components["host"] = ipv4host or ipv6host
        name = components.pop("name")

        if components["port"]:
            components["port"] = int(components["port"])

        return URL.create(name, **components)  # type: ignore

    else:
        raise exc.ArgumentError(
            "Could not parse SQLAlchemy URL from given URL string"
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\gaussopt.py ===
"""
Gaussian optics.

The module implements:

- Ray transfer matrices for geometrical and gaussian optics.

  See RayTransferMatrix, GeometricRay and BeamParameter

- Conjugation relations for geometrical and gaussian optics.

  See geometric_conj*, gauss_conj and conjugate_gauss_beams

The conventions for the distances are as follows:

focal distance
    positive for convergent lenses
object distance
    positive for real objects
image distance
    positive for real images
"""

__all__ = [
    'RayTransferMatrix',
    'FreeSpace',
    'FlatRefraction',
    'CurvedRefraction',
    'FlatMirror',
    'CurvedMirror',
    'ThinLens',
    'GeometricRay',
    'BeamParameter',
    'waist2rayleigh',
    'rayleigh2waist',
    'geometric_conj_ab',
    'geometric_conj_af',
    'geometric_conj_bf',
    'gaussian_conj',
    'conjugate_gauss_beams',
]


from sympy.core.expr import Expr
from sympy.core.numbers import (I, pi)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2
from sympy.matrices.dense import Matrix, MutableDenseMatrix
from sympy.polys.rationaltools import together
from sympy.utilities.misc import filldedent

###
# A, B, C, D matrices
###


class RayTransferMatrix(MutableDenseMatrix):
    """
    Base class for a Ray Transfer Matrix.

    It should be used if there is not already a more specific subclass mentioned
    in See Also.

    Parameters
    ==========

    parameters :
        A, B, C and D or 2x2 matrix (Matrix(2, 2, [A, B, C, D]))

    Examples
    ========

    >>> from sympy.physics.optics import RayTransferMatrix, ThinLens
    >>> from sympy import Symbol, Matrix

    >>> mat = RayTransferMatrix(1, 2, 3, 4)
    >>> mat
    Matrix([
    [1, 2],
    [3, 4]])

    >>> RayTransferMatrix(Matrix([[1, 2], [3, 4]]))
    Matrix([
    [1, 2],
    [3, 4]])

    >>> mat.A
    1

    >>> f = Symbol('f')
    >>> lens = ThinLens(f)
    >>> lens
    Matrix([
    [   1, 0],
    [-1/f, 1]])

    >>> lens.C
    -1/f

    See Also
    ========

    GeometricRay, BeamParameter,
    FreeSpace, FlatRefraction, CurvedRefraction,
    FlatMirror, CurvedMirror, ThinLens

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    """

    def __new__(cls, *args):

        if len(args) == 4:
            temp = ((args[0], args[1]), (args[2], args[3]))
        elif len(args) == 1 \
            and isinstance(args[0], Matrix) \
                and args[0].shape == (2, 2):
            temp = args[0]
        else:
            raise ValueError(filldedent('''
                Expecting 2x2 Matrix or the 4 elements of
                the Matrix but got %s''' % str(args)))
        return Matrix.__new__(cls, temp)

    def __mul__(self, other):
        if isinstance(other, RayTransferMatrix):
            return RayTransferMatrix(Matrix(self)*Matrix(other))
        elif isinstance(other, GeometricRay):
            return GeometricRay(Matrix(self)*Matrix(other))
        elif isinstance(other, BeamParameter):
            temp = Matrix(self)*Matrix(((other.q,), (1,)))
            q = (temp[0]/temp[1]).expand(complex=True)
            return BeamParameter(other.wavelen,
                                 together(re(q)),
                                 z_r=together(im(q)))
        else:
            return Matrix.__mul__(self, other)

    @property
    def A(self):
        """
        The A parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.A
        1
        """
        return self[0, 0]

    @property
    def B(self):
        """
        The B parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.B
        2
        """
        return self[0, 1]

    @property
    def C(self):
        """
        The C parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.C
        3
        """
        return self[1, 0]

    @property
    def D(self):
        """
        The D parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.D
        4
        """
        return self[1, 1]


class FreeSpace(RayTransferMatrix):
    """
    Ray Transfer Matrix for free space.

    Parameters
    ==========

    distance

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FreeSpace
    >>> from sympy import symbols
    >>> d = symbols('d')
    >>> FreeSpace(d)
    Matrix([
    [1, d],
    [0, 1]])
    """
    def __new__(cls, d):
        return RayTransferMatrix.__new__(cls, 1, d, 0, 1)


class FlatRefraction(RayTransferMatrix):
    """
    Ray Transfer Matrix for refraction.

    Parameters
    ==========

    n1 :
        Refractive index of one medium.
    n2 :
        Refractive index of other medium.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FlatRefraction
    >>> from sympy import symbols
    >>> n1, n2 = symbols('n1 n2')
    >>> FlatRefraction(n1, n2)
    Matrix([
    [1,     0],
    [0, n1/n2]])
    """
    def __new__(cls, n1, n2):
        n1, n2 = map(sympify, (n1, n2))
        return RayTransferMatrix.__new__(cls, 1, 0, 0, n1/n2)


class CurvedRefraction(RayTransferMatrix):
    """
    Ray Transfer Matrix for refraction on curved interface.

    Parameters
    ==========

    R :
        Radius of curvature (positive for concave).
    n1 :
        Refractive index of one medium.
    n2 :
        Refractive index of other medium.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import CurvedRefraction
    >>> from sympy import symbols
    >>> R, n1, n2 = symbols('R n1 n2')
    >>> CurvedRefraction(R, n1, n2)
    Matrix([
    [               1,     0],
    [(n1 - n2)/(R*n2), n1/n2]])
    """
    def __new__(cls, R, n1, n2):
        R, n1, n2 = map(sympify, (R, n1, n2))
        return RayTransferMatrix.__new__(cls, 1, 0, (n1 - n2)/R/n2, n1/n2)


class FlatMirror(RayTransferMatrix):
    """
    Ray Transfer Matrix for reflection.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FlatMirror
    >>> FlatMirror()
    Matrix([
    [1, 0],
    [0, 1]])
    """
    def __new__(cls):
        return RayTransferMatrix.__new__(cls, 1, 0, 0, 1)


class CurvedMirror(RayTransferMatrix):
    """
    Ray Transfer Matrix for reflection from curved surface.

    Parameters
    ==========

    R : radius of curvature (positive for concave)

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import CurvedMirror
    >>> from sympy import symbols
    >>> R = symbols('R')
    >>> CurvedMirror(R)
    Matrix([
    [   1, 0],
    [-2/R, 1]])
    """
    def __new__(cls, R):
        R = sympify(R)
        return RayTransferMatrix.__new__(cls, 1, 0, -2/R, 1)


class ThinLens(RayTransferMatrix):
    """
    Ray Transfer Matrix for a thin lens.

    Parameters
    ==========

    f :
        The focal distance.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import ThinLens
    >>> from sympy import symbols
    >>> f = symbols('f')
    >>> ThinLens(f)
    Matrix([
    [   1, 0],
    [-1/f, 1]])
    """
    def __new__(cls, f):
        f = sympify(f)
        return RayTransferMatrix.__new__(cls, 1, 0, -1/f, 1)


###
# Representation for geometric ray
###

class GeometricRay(MutableDenseMatrix):
    """
    Representation for a geometric ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    h : height, and
    angle : angle, or
    matrix : a 2x1 matrix (Matrix(2, 1, [height, angle]))

    Examples
    ========

    >>> from sympy.physics.optics import GeometricRay, FreeSpace
    >>> from sympy import symbols, Matrix
    >>> d, h, angle = symbols('d, h, angle')

    >>> GeometricRay(h, angle)
    Matrix([
    [    h],
    [angle]])

    >>> FreeSpace(d)*GeometricRay(h, angle)
    Matrix([
    [angle*d + h],
    [      angle]])

    >>> GeometricRay( Matrix( ((h,), (angle,)) ) )
    Matrix([
    [    h],
    [angle]])

    See Also
    ========

    RayTransferMatrix

    """

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], Matrix) \
                and args[0].shape == (2, 1):
            temp = args[0]
        elif len(args) == 2:
            temp = ((args[0],), (args[1],))
        else:
            raise ValueError(filldedent('''
                Expecting 2x1 Matrix or the 2 elements of
                the Matrix but got %s''' % str(args)))
        return Matrix.__new__(cls, temp)

    @property
    def height(self):
        """
        The distance from the optical axis.

        Examples
        ========

        >>> from sympy.physics.optics import GeometricRay
        >>> from sympy import symbols
        >>> h, angle = symbols('h, angle')
        >>> gRay = GeometricRay(h, angle)
        >>> gRay.height
        h
        """
        return self[0]

    @property
    def angle(self):
        """
        The angle with the optical axis.

        Examples
        ========

        >>> from sympy.physics.optics import GeometricRay
        >>> from sympy import symbols
        >>> h, angle = symbols('h, angle')
        >>> gRay = GeometricRay(h, angle)
        >>> gRay.angle
        angle
        """
        return self[1]


###
# Representation for gauss beam
###

class BeamParameter(Expr):
    """
    Representation for a gaussian ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    wavelen : the wavelength,
    z : the distance to waist, and
    w : the waist, or
    z_r : the rayleigh range.
    n : the refractive index of medium.

    Examples
    ========

    >>> from sympy.physics.optics import BeamParameter
    >>> p = BeamParameter(530e-9, 1, w=1e-3)
    >>> p.q
    1 + 1.88679245283019*I*pi

    >>> p.q.n()
    1.0 + 5.92753330865999*I
    >>> p.w_0.n()
    0.00100000000000000
    >>> p.z_r.n()
    5.92753330865999

    >>> from sympy.physics.optics import FreeSpace
    >>> fs = FreeSpace(10)
    >>> p1 = fs*p
    >>> p.w.n()
    0.00101413072159615
    >>> p1.w.n()
    0.00210803120913829

    See Also
    ========

    RayTransferMatrix

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_beam_parameter
    .. [2] https://en.wikipedia.org/wiki/Gaussian_beam
    """
    #TODO A class Complex may be implemented. The BeamParameter may
    # subclass it. See:
    # https://groups.google.com/d/topic/sympy/7XkU07NRBEs/discussion

    def __new__(cls, wavelen, z, z_r=None, w=None, n=1):
        wavelen = sympify(wavelen)
        z = sympify(z)
        n = sympify(n)

        if z_r is not None and w is None:
            z_r = sympify(z_r)
        elif w is not None and z_r is None:
            z_r = waist2rayleigh(sympify(w), wavelen, n)
        elif z_r is None and w is None:
            raise ValueError('Must specify one of w and z_r.')

        return Expr.__new__(cls, wavelen, z, z_r, n)

    @property
    def wavelen(self):
        return self.args[0]

    @property
    def z(self):
        return self.args[1]

    @property
    def z_r(self):
        return self.args[2]

    @property
    def n(self):
        return self.args[3]

    @property
    def q(self):
        """
        The complex parameter representing the beam.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.q
        1 + 1.88679245283019*I*pi
        """
        return self.z + I*self.z_r

    @property
    def radius(self):
        """
        The radius of curvature of the phase front.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.radius
        1 + 3.55998576005696*pi**2
        """
        return self.z*(1 + (self.z_r/self.z)**2)

    @property
    def w(self):
        """
        The radius of the beam w(z), at any position z along the beam.
        The beam radius at `1/e^2` intensity (axial value).

        See Also
        ========

        w_0 :
            The minimal radius of beam.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.w
        0.001*sqrt(0.2809/pi**2 + 1)
        """
        return self.w_0*sqrt(1 + (self.z/self.z_r)**2)

    @property
    def w_0(self):
        """
         The minimal radius of beam at `1/e^2` intensity (peak value).

        See Also
        ========

        w : the beam radius at `1/e^2` intensity (axial value).

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.w_0
        0.00100000000000000
        """
        return sqrt(self.z_r/(pi*self.n)*self.wavelen)

    @property
    def divergence(self):
        """
        Half of the total angular spread.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.divergence
        0.00053/pi
        """
        return self.wavelen/pi/self.w_0

    @property
    def gouy(self):
        """
        The Gouy phase.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.gouy
        atan(0.53/pi)
        """
        return atan2(self.z, self.z_r)

    @property
    def waist_approximation_limit(self):
        """
        The minimal waist for which the gauss beam approximation is valid.

        Explanation
        ===========

        The gauss beam is a solution to the paraxial equation. For curvatures
        that are too great it is not a valid approximation.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.waist_approximation_limit
        1.06e-6/pi
        """
        return 2*self.wavelen/pi


###
# Utilities
###

def waist2rayleigh(w, wavelen, n=1):
    """
    Calculate the rayleigh range from the waist of a gaussian beam.

    See Also
    ========

    rayleigh2waist, BeamParameter

    Examples
    ========

    >>> from sympy.physics.optics import waist2rayleigh
    >>> from sympy import symbols
    >>> w, wavelen = symbols('w wavelen')
    >>> waist2rayleigh(w, wavelen)
    pi*w**2/wavelen
    """
    w, wavelen = map(sympify, (w, wavelen))
    return w**2*n*pi/wavelen


def rayleigh2waist(z_r, wavelen):
    """Calculate the waist from the rayleigh range of a gaussian beam.

    See Also
    ========

    waist2rayleigh, BeamParameter

    Examples
    ========

    >>> from sympy.physics.optics import rayleigh2waist
    >>> from sympy import symbols
    >>> z_r, wavelen = symbols('z_r wavelen')
    >>> rayleigh2waist(z_r, wavelen)
    sqrt(wavelen*z_r)/sqrt(pi)
    """
    z_r, wavelen = map(sympify, (z_r, wavelen))
    return sqrt(z_r/pi*wavelen)


def geometric_conj_ab(a, b):
    """
    Conjugation relation for geometrical beams under paraxial conditions.

    Explanation
    ===========

    Takes the distances to the optical element and returns the needed
    focal distance.

    See Also
    ========

    geometric_conj_af, geometric_conj_bf

    Examples
    ========

    >>> from sympy.physics.optics import geometric_conj_ab
    >>> from sympy import symbols
    >>> a, b = symbols('a b')
    >>> geometric_conj_ab(a, b)
    a*b/(a + b)
    """
    a, b = map(sympify, (a, b))
    if a.is_infinite or b.is_infinite:
        return a if b.is_infinite else b
    else:
        return a*b/(a + b)


def geometric_conj_af(a, f):
    """
    Conjugation relation for geometrical beams under paraxial conditions.

    Explanation
    ===========

    Takes the object distance (for geometric_conj_af) or the image distance
    (for geometric_conj_bf) to the optical element and the focal distance.
    Then it returns the other distance needed for conjugation.

    See Also
    ========

    geometric_conj_ab

    Examples
    ========

    >>> from sympy.physics.optics.gaussopt import geometric_conj_af, geometric_conj_bf
    >>> from sympy import symbols
    >>> a, b, f = symbols('a b f')
    >>> geometric_conj_af(a, f)
    a*f/(a - f)
    >>> geometric_conj_bf(b, f)
    b*f/(b - f)
    """
    a, f = map(sympify, (a, f))
    return -geometric_conj_ab(a, -f)

geometric_conj_bf = geometric_conj_af


def gaussian_conj(s_in, z_r_in, f):
    """
    Conjugation relation for gaussian beams.

    Parameters
    ==========

    s_in :
        The distance to optical element from the waist.
    z_r_in :
        The rayleigh range of the incident beam.
    f :
        The focal length of the optical element.

    Returns
    =======

    a tuple containing (s_out, z_r_out, m)
    s_out :
        The distance between the new waist and the optical element.
    z_r_out :
        The rayleigh range of the emergent beam.
    m :
        The ration between the new and the old waists.

    Examples
    ========

    >>> from sympy.physics.optics import gaussian_conj
    >>> from sympy import symbols
    >>> s_in, z_r_in, f = symbols('s_in z_r_in f')

    >>> gaussian_conj(s_in, z_r_in, f)[0]
    1/(-1/(s_in + z_r_in**2/(-f + s_in)) + 1/f)

    >>> gaussian_conj(s_in, z_r_in, f)[1]
    z_r_in/(1 - s_in**2/f**2 + z_r_in**2/f**2)

    >>> gaussian_conj(s_in, z_r_in, f)[2]
    1/sqrt(1 - s_in**2/f**2 + z_r_in**2/f**2)
    """
    s_in, z_r_in, f = map(sympify, (s_in, z_r_in, f))
    s_out = 1 / ( -1/(s_in + z_r_in**2/(s_in - f)) + 1/f )
    m = 1/sqrt((1 - (s_in/f)**2) + (z_r_in/f)**2)
    z_r_out = z_r_in / ((1 - (s_in/f)**2) + (z_r_in/f)**2)
    return (s_out, z_r_out, m)


def conjugate_gauss_beams(wavelen, waist_in, waist_out, **kwargs):
    """
    Find the optical setup conjugating the object/image waists.

    Parameters
    ==========

    wavelen :
        The wavelength of the beam.
    waist_in and waist_out :
        The waists to be conjugated.
    f :
        The focal distance of the element used in the conjugation.

    Returns
    =======

    a tuple containing (s_in, s_out, f)
    s_in :
        The distance before the optical element.
    s_out :
        The distance after the optical element.
    f :
        The focal distance of the optical element.

    Examples
    ========

    >>> from sympy.physics.optics import conjugate_gauss_beams
    >>> from sympy import symbols, factor
    >>> l, w_i, w_o, f = symbols('l w_i w_o f')

    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[0]
    f*(1 - sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)))

    >>> factor(conjugate_gauss_beams(l, w_i, w_o, f=f)[1])
    f*w_o**2*(w_i**2/w_o**2 - sqrt(w_i**2/w_o**2 -
              pi**2*w_i**4/(f**2*l**2)))/w_i**2

    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[2]
    f
    """
    #TODO add the other possible arguments
    wavelen, waist_in, waist_out = map(sympify, (wavelen, waist_in, waist_out))
    m = waist_out / waist_in
    z = waist2rayleigh(waist_in, wavelen)
    if len(kwargs) != 1:
        raise ValueError("The function expects only one named argument")
    elif 'dist' in kwargs:
        raise NotImplementedError(filldedent('''
            Currently only focal length is supported as a parameter'''))
    elif 'f' in kwargs:
        f = sympify(kwargs['f'])
        s_in = f * (1 - sqrt(1/m**2 - z**2/f**2))
        s_out = gaussian_conj(s_in, z, f)[0]
    elif 's_in' in kwargs:
        raise NotImplementedError(filldedent('''
            Currently only focal length is supported as a parameter'''))
    else:
        raise ValueError(filldedent('''
            The functions expects the focal length as a named argument'''))
    return (s_in, s_out, f)

#TODO
#def plot_beam():
#    """Plot the beam radius as it propagates in space."""
#    pass

#TODO
#def plot_beam_conjugation():
#    """
#    Plot the intersection of two beams.
#
#    Represents the conjugation relation.
#
#    See Also
#    ========
#
#    conjugate_gauss_beams
#    """
#    pass