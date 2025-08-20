
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\tests\test_matexpr.py ===
from sympy.concrete.summations import Sum
from sympy.core.exprtools import gcd_terms
from sympy.core.function import (diff, expand)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol, Str)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.polys.polytools import factor

from sympy.core import (S, symbols, Add, Mul, SympifyError, Rational,
                    Function)
from sympy.functions import sin, cos, tan, sqrt, cbrt, exp
from sympy.simplify import simplify
from sympy.matrices import (ImmutableMatrix, Inverse, MatAdd, MatMul,
        MatPow, Matrix, MatrixExpr, MatrixSymbol,
        SparseMatrix, Transpose, Adjoint, MatrixSet)
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.expressions.determinant import Determinant, det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.testing.pytest import raises, XFAIL, skip
from importlib.metadata import version

n, m, l, k, p = symbols('n m l k p', integer=True)
x = symbols('x')
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, l)
C = MatrixSymbol('C', n, n)
D = MatrixSymbol('D', n, n)
E = MatrixSymbol('E', m, n)
w = MatrixSymbol('w', n, 1)


def test_matrix_symbol_creation():
    assert MatrixSymbol('A', 2, 2)
    assert MatrixSymbol('A', 0, 0)
    raises(ValueError, lambda: MatrixSymbol('A', -1, 2))
    raises(ValueError, lambda: MatrixSymbol('A', 2.0, 2))
    raises(ValueError, lambda: MatrixSymbol('A', 2j, 2))
    raises(ValueError, lambda: MatrixSymbol('A', 2, -1))
    raises(ValueError, lambda: MatrixSymbol('A', 2, 2.0))
    raises(ValueError, lambda: MatrixSymbol('A', 2, 2j))

    n = symbols('n')
    assert MatrixSymbol('A', n, n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: MatrixSymbol('A', n, n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: MatrixSymbol('A', n, n))


def test_matexpr_properties():
    assert A.shape == (n, m)
    assert (A * B).shape == (n, l)
    assert A[0, 1].indices == (0, 1)
    assert A[0, 0].symbol == A
    assert A[0, 0].symbol.name == 'A'


def test_matexpr():
    assert (x*A).shape == A.shape
    assert (x*A).__class__ == MatMul
    assert 2*A - A - A == ZeroMatrix(*A.shape)
    assert (A*B).shape == (n, l)


def test_matexpr_subs():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, l)
    C = MatrixSymbol('C', m, l)

    assert A.subs(n, m).shape == (m, m)
    assert (A*B).subs(B, C) == A*C
    assert (A*B).subs(l, n).is_square

    W = MatrixSymbol("W", 3, 3)
    X = MatrixSymbol("X", 2, 2)
    Y = MatrixSymbol("Y", 1, 2)
    Z = MatrixSymbol("Z", n, 2)
    # no restrictions on Symbol replacement
    assert X.subs(X, Y) == Y
    # it might be better to just change the name
    y = Str('y')
    assert X.subs(Str("X"), y).args == (y, 2, 2)
    # it's ok to introduce a wider matrix
    assert X[1, 1].subs(X, W) == W[1, 1]
    # but for a given MatrixExpression, only change
    # name if indexing on the new shape is valid.
    # Here, X is 2,2; Y is 1,2 and Y[1, 1] is out
    # of range so an error is raised
    raises(IndexError, lambda: X[1, 1].subs(X, Y))
    # here, [0, 1] is in range so the subs succeeds
    assert X[0, 1].subs(X, Y) == Y[0, 1]
    # and here the size of n will accept any index
    # in the first position
    assert W[2, 1].subs(W, Z) == Z[2, 1]
    # but not in the second position
    raises(IndexError, lambda: W[2, 2].subs(W, Z))
    # any matrix should raise if invalid
    raises(IndexError, lambda: W[2, 2].subs(W, zeros(2)))

    A = SparseMatrix([[1, 2], [3, 4]])
    B = Matrix([[1, 2], [3, 4]])
    C, D = MatrixSymbol('C', 2, 2), MatrixSymbol('D', 2, 2)

    assert (C*D).subs({C: A, D: B}) == MatMul(A, B)


def test_addition():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', n, m)

    assert isinstance(A + B, MatAdd)
    assert (A + B).shape == A.shape
    assert isinstance(A - A + 2*B, MatMul)

    raises(TypeError, lambda: A + 1)
    raises(TypeError, lambda: 5 + A)
    raises(TypeError, lambda: 5 - A)

    assert A + ZeroMatrix(n, m) - A == ZeroMatrix(n, m)
    raises(TypeError, lambda: ZeroMatrix(n, m) + S.Zero)


def test_multiplication():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, l)
    C = MatrixSymbol('C', n, n)

    assert (2*A*B).shape == (n, l)
    assert (A*0*B) == ZeroMatrix(n, l)
    assert (2*A).shape == A.shape

    assert A * ZeroMatrix(m, m) * B == ZeroMatrix(n, l)

    assert C * Identity(n) * C.I == Identity(n)

    assert B/2 == S.Half*B
    raises(NotImplementedError, lambda: 2/B)

    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    assert Identity(n) * (A + B) == A + B

    assert A**2*A == A**3
    assert A**2*(A.I)**3 == A.I
    assert A**3*(A.I)**2 == A


def test_MatPow():
    A = MatrixSymbol('A', n, n)

    AA = MatPow(A, 2)
    assert AA.exp == 2
    assert AA.base == A
    assert (A**n).exp == n

    assert A**0 == Identity(n)
    assert A**1 == A
    assert A**2 == AA
    assert A**-1 == Inverse(A)
    assert (A**-1)**-1 == A
    assert (A**2)**3 == A**6
    assert A**S.Half == sqrt(A)
    assert A**Rational(1, 3) == cbrt(A)
    raises(NonSquareMatrixError, lambda: MatrixSymbol('B', 3, 2)**2)


def test_MatrixSymbol():
    n, m, t = symbols('n,m,t')
    X = MatrixSymbol('X', n, m)
    assert X.shape == (n, m)
    raises(TypeError, lambda: MatrixSymbol('X', n, m)(t))  # issue 5855
    assert X.doit() == X


def test_dense_conversion():
    X = MatrixSymbol('X', 2, 2)
    assert ImmutableMatrix(X) == ImmutableMatrix(2, 2, lambda i, j: X[i, j])
    assert Matrix(X) == Matrix(2, 2, lambda i, j: X[i, j])


def test_free_symbols():
    assert (C*D).free_symbols == {C, D}


def test_zero_matmul():
    assert isinstance(S.Zero * MatrixSymbol('X', 2, 2), MatrixExpr)


def test_matadd_simplify():
    A = MatrixSymbol('A', 1, 1)
    assert simplify(MatAdd(A, ImmutableMatrix([[sin(x)**2 + cos(x)**2]]))) == \
        MatAdd(A, Matrix([[1]]))


def test_matmul_simplify():
    A = MatrixSymbol('A', 1, 1)
    assert simplify(MatMul(A, ImmutableMatrix([[sin(x)**2 + cos(x)**2]]))) == \
        MatMul(A, Matrix([[1]]))


def test_invariants():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', m, l)
    X = MatrixSymbol('X', n, n)
    objs = [Identity(n), ZeroMatrix(m, n), A, MatMul(A, B), MatAdd(A, A),
            Transpose(A), Adjoint(A), Inverse(X), MatPow(X, 2), MatPow(X, -1),
            MatPow(X, 0)]
    for obj in objs:
        assert obj == obj.__class__(*obj.args)


def test_matexpr_indexing():
    A = MatrixSymbol('A', n, m)
    A[1, 2]
    A[l, k]
    A[l + 1, k + 1]
    A = MatrixSymbol('A', 2, 1)
    for i in range(-2, 2):
        for j in range(-1, 1):
            A[i, j]


def test_single_indexing():
    A = MatrixSymbol('A', 2, 3)
    assert A[1] == A[0, 1]
    assert A[int(1)] == A[0, 1]
    assert A[3] == A[1, 0]
    assert list(A[:2, :2]) == [A[0, 0], A[0, 1], A[1, 0], A[1, 1]]
    raises(IndexError, lambda: A[6])
    raises(IndexError, lambda: A[n])
    B = MatrixSymbol('B', n, m)
    raises(IndexError, lambda: B[1])
    B = MatrixSymbol('B', n, 3)
    assert B[3] == B[1, 0]


def test_MatrixElement_commutative():
    assert A[0, 1]*A[1, 0] == A[1, 0]*A[0, 1]


def test_MatrixSymbol_determinant():
    A = MatrixSymbol('A', 4, 4)
    assert A.as_explicit().det() == A[0, 0]*A[1, 1]*A[2, 2]*A[3, 3] - \
        A[0, 0]*A[1, 1]*A[2, 3]*A[3, 2] - A[0, 0]*A[1, 2]*A[2, 1]*A[3, 3] + \
        A[0, 0]*A[1, 2]*A[2, 3]*A[3, 1] + A[0, 0]*A[1, 3]*A[2, 1]*A[3, 2] - \
        A[0, 0]*A[1, 3]*A[2, 2]*A[3, 1] - A[0, 1]*A[1, 0]*A[2, 2]*A[3, 3] + \
        A[0, 1]*A[1, 0]*A[2, 3]*A[3, 2] + A[0, 1]*A[1, 2]*A[2, 0]*A[3, 3] - \
        A[0, 1]*A[1, 2]*A[2, 3]*A[3, 0] - A[0, 1]*A[1, 3]*A[2, 0]*A[3, 2] + \
        A[0, 1]*A[1, 3]*A[2, 2]*A[3, 0] + A[0, 2]*A[1, 0]*A[2, 1]*A[3, 3] - \
        A[0, 2]*A[1, 0]*A[2, 3]*A[3, 1] - A[0, 2]*A[1, 1]*A[2, 0]*A[3, 3] + \
        A[0, 2]*A[1, 1]*A[2, 3]*A[3, 0] + A[0, 2]*A[1, 3]*A[2, 0]*A[3, 1] - \
        A[0, 2]*A[1, 3]*A[2, 1]*A[3, 0] - A[0, 3]*A[1, 0]*A[2, 1]*A[3, 2] + \
        A[0, 3]*A[1, 0]*A[2, 2]*A[3, 1] + A[0, 3]*A[1, 1]*A[2, 0]*A[3, 2] - \
        A[0, 3]*A[1, 1]*A[2, 2]*A[3, 0] - A[0, 3]*A[1, 2]*A[2, 0]*A[3, 1] + \
        A[0, 3]*A[1, 2]*A[2, 1]*A[3, 0]

    B = MatrixSymbol('B', 4, 4)
    assert Determinant(A + B).doit() == det(A + B) == (A + B).det()


def test_MatrixElement_diff():
    assert (A[3, 0]*A[0, 0]).diff(A[0, 0]) == A[3, 0]


def test_MatrixElement_doit():
    u = MatrixSymbol('u', 2, 1)
    v = ImmutableMatrix([3, 5])
    assert u[0, 0].subs(u, v).doit() == v[0, 0]


def test_identity_powers():
    M = Identity(n)
    assert MatPow(M, 3).doit() == M**3
    assert M**n == M
    assert MatPow(M, 0).doit() == M**2
    assert M**-2 == M
    assert MatPow(M, -2).doit() == M**0
    N = Identity(3)
    assert MatPow(N, 2).doit() == N**n
    assert MatPow(N, 3).doit() == N
    assert MatPow(N, -2).doit() == N**4
    assert MatPow(N, 2).doit() == N**0


def test_Zero_power():
    z1 = ZeroMatrix(n, n)
    assert z1**4 == z1
    raises(ValueError, lambda:z1**-2)
    assert z1**0 == Identity(n)
    assert MatPow(z1, 2).doit() == z1**2
    raises(ValueError, lambda:MatPow(z1, -2).doit())
    z2 = ZeroMatrix(3, 3)
    assert MatPow(z2, 4).doit() == z2**4
    raises(ValueError, lambda:z2**-3)
    assert z2**3 == MatPow(z2, 3).doit()
    assert z2**0 == Identity(3)
    raises(ValueError, lambda:MatPow(z2, -1).doit())


def test_matrixelement_diff():
    dexpr = diff((D*w)[k,0], w[p,0])

    assert w[k, p].diff(w[k, p]) == 1
    assert w[k, p].diff(w[0, 0]) == KroneckerDelta(0, k, (0, n-1))*KroneckerDelta(0, p, (0, 0))
    _i_1 = Dummy("_i_1")
    assert dexpr.dummy_eq(Sum(KroneckerDelta(_i_1, p, (0, n-1))*D[k, _i_1], (_i_1, 0, n - 1)))
    assert dexpr.doit() == D[k, p]


def test_MatrixElement_with_values():
    x, y, z, w = symbols("x y z w")
    M = Matrix([[x, y], [z, w]])
    i, j = symbols("i, j")
    Mij = M[i, j]
    assert isinstance(Mij, MatrixElement)
    Ms = SparseMatrix([[2, 3], [4, 5]])
    msij = Ms[i, j]
    assert isinstance(msij, MatrixElement)
    for oi, oj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert Mij.subs({i: oi, j: oj}) == M[oi, oj]
        assert msij.subs({i: oi, j: oj}) == Ms[oi, oj]
    A = MatrixSymbol("A", 2, 2)
    assert A[0, 0].subs(A, M) == x
    assert A[i, j].subs(A, M) == M[i, j]
    assert M[i, j].subs(M, A) == A[i, j]

    assert isinstance(M[3*i - 2, j], MatrixElement)
    assert M[3*i - 2, j].subs({i: 1, j: 0}) == M[1, 0]
    assert isinstance(M[i, 0], MatrixElement)
    assert M[i, 0].subs(i, 0) == M[0, 0]
    assert M[0, i].subs(i, 1) == M[0, 1]

    assert M[i, j].diff(x) == Matrix([[1, 0], [0, 0]])[i, j]

    raises(ValueError, lambda: M[i, 2])
    raises(ValueError, lambda: M[i, -1])
    raises(ValueError, lambda: M[2, i])
    raises(ValueError, lambda: M[-1, i])


def test_inv():
    B = MatrixSymbol('B', 3, 3)
    assert B.inv() == B**-1

    # https://github.com/sympy/sympy/issues/19162
    X = MatrixSymbol('X', 1, 1).as_explicit()
    assert X.inv() == Matrix([[1/X[0, 0]]])

    X = MatrixSymbol('X', 2, 2).as_explicit()
    detX = X[0, 0]*X[1, 1] - X[0, 1]*X[1, 0]
    invX = Matrix([[ X[1, 1], -X[0, 1]],
                   [-X[1, 0],  X[0, 0]]]) / detX
    assert X.inv() == invX


@XFAIL
def test_factor_expand():
    A = MatrixSymbol("A", n, n)
    B = MatrixSymbol("B", n, n)
    expr1 = (A + B)*(C + D)
    expr2 = A*C + B*C + A*D + B*D
    assert expr1 != expr2
    assert expand(expr1) == expr2
    assert factor(expr2) == expr1

    expr = B**(-1)*(A**(-1)*B**(-1) - A**(-1)*C*B**(-1))**(-1)*A**(-1)
    I = Identity(n)
    # Ideally we get the first, but we at least don't want a wrong answer
    assert factor(expr) in [I - C, B**-1*(A**-1*(I - C)*B**-1)**-1*A**-1]

def test_numpy_conversion():
    try:
        from numpy import array, array_equal
    except ImportError:
        skip('NumPy must be available to test creating matrices from ndarrays')
    A = MatrixSymbol('A', 2, 2)
    np_array = array([[MatrixElement(A, 0, 0), MatrixElement(A, 0, 1)],
    [MatrixElement(A, 1, 0), MatrixElement(A, 1, 1)]])
    assert array_equal(array(A), np_array)
    assert array_equal(array(A, copy=True), np_array)
    if(int(version('numpy').split('.')[0]) >= 2): #run this test only if numpy is new enough that copy variable is passed properly.
        raises(TypeError, lambda: array(A, copy=False))

def test_issue_2749():
    A = MatrixSymbol("A", 5, 2)
    assert (A.T * A).I.as_explicit() == Matrix([[(A.T * A).I[0, 0], (A.T * A).I[0, 1]], \
    [(A.T * A).I[1, 0], (A.T * A).I[1, 1]]])


def test_issue_2750():
    x = MatrixSymbol('x', 1, 1)
    assert (x.T*x).as_explicit()**-1 == Matrix([[x[0, 0]**(-2)]])


def test_issue_7842():
    A = MatrixSymbol('A', 3, 1)
    B = MatrixSymbol('B', 2, 1)
    assert Eq(A, B) == False
    assert Eq(A[1,0], B[1, 0]).func is Eq
    A = ZeroMatrix(2, 3)
    B = ZeroMatrix(2, 3)
    assert Eq(A, B) == True


def test_issue_21195():
    t = symbols('t')
    x = Function('x')(t)
    dx = x.diff(t)
    exp1 = cos(x) + cos(x)*dx
    exp2 = sin(x) + tan(x)*(dx.diff(t))
    exp3 = sin(x)*sin(t)*(dx.diff(t)).diff(t)
    A = Matrix([[exp1], [exp2], [exp3]])
    B = Matrix([[exp1.diff(x)], [exp2.diff(x)], [exp3.diff(x)]])
    assert A.diff(x) == B


def test_issue_24859():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 3, 2)
    J = A*B
    Jinv = Matrix(J).adjugate()
    u = MatrixSymbol('u', 2, 3)
    Jk = Jinv.subs(A, A + x*u)

    expected = B[0, 1]*u[1, 0] + B[1, 1]*u[1, 1] + B[2, 1]*u[1, 2]
    assert Jk[0, 0].diff(x) == expected
    assert diff(Jk[0, 0], x).doit() == expected


def test_MatMul_postprocessor():
    z = zeros(2)
    z1 = ZeroMatrix(2, 2)
    assert Mul(0, z) == Mul(z, 0) in [z, z1]

    M = Matrix([[1, 2], [3, 4]])
    Mx = Matrix([[x, 2*x], [3*x, 4*x]])
    assert Mul(x, M) == Mul(M, x) == Mx

    A = MatrixSymbol("A", 2, 2)
    assert Mul(A, M) == MatMul(A, M)
    assert Mul(M, A) == MatMul(M, A)
    # Scalars should be absorbed into constant matrices
    a = Mul(x, M, A)
    b = Mul(M, x, A)
    c = Mul(M, A, x)
    assert a == b == c == MatMul(Mx, A)
    a = Mul(x, A, M)
    b = Mul(A, x, M)
    c = Mul(A, M, x)
    assert a == b == c == MatMul(A, Mx)
    assert Mul(M, M) == M**2
    assert Mul(A, M, M) == MatMul(A, M**2)
    assert Mul(M, M, A) == MatMul(M**2, A)
    assert Mul(M, A, M) == MatMul(M, A, M)

    assert Mul(A, x, M, M, x) == MatMul(A, Mx**2)


@XFAIL
def test_MatAdd_postprocessor_xfail():
    # This is difficult to get working because of the way that Add processes
    # its args.
    z = zeros(2)
    assert Add(z, S.NaN) == Add(S.NaN, z)


def test_MatAdd_postprocessor():
    # Some of these are nonsensical, but we do not raise errors for Add
    # because that breaks algorithms that want to replace matrices with dummy
    # symbols.

    z = zeros(2)

    assert Add(0, z) == Add(z, 0) == z

    a = Add(S.Infinity, z)
    assert a == Add(z, S.Infinity)
    assert isinstance(a, Add)
    assert a.args == (S.Infinity, z)

    a = Add(S.ComplexInfinity, z)
    assert a == Add(z, S.ComplexInfinity)
    assert isinstance(a, Add)
    assert a.args == (S.ComplexInfinity, z)

    a = Add(z, S.NaN)
    # assert a == Add(S.NaN, z) # See the XFAIL above
    assert isinstance(a, Add)
    assert a.args == (S.NaN, z)

    M = Matrix([[1, 2], [3, 4]])
    a = Add(x, M)
    assert a == Add(M, x)
    assert isinstance(a, Add)
    assert a.args == (x, M)

    A = MatrixSymbol("A", 2, 2)
    assert Add(A, M) == Add(M, A) == A + M

    # Scalars should be absorbed into constant matrices (producing an error)
    a = Add(x, M, A)
    assert a == Add(M, x, A) == Add(M, A, x) == Add(x, A, M) == Add(A, x, M) == Add(A, M, x)
    assert isinstance(a, Add)
    assert a.args == (x, A + M)

    assert Add(M, M) == 2*M
    assert Add(M, A, M) == Add(M, M, A) == Add(A, M, M) == A + 2*M

    a = Add(A, x, M, M, x)
    assert isinstance(a, Add)
    assert a.args == (2*x, A + 2*M)


def test_simplify_matrix_expressions():
    # Various simplification functions
    assert type(gcd_terms(C*D + D*C)) == MatAdd
    a = gcd_terms(2*C*D + 4*D*C)
    assert type(a) == MatAdd
    assert a.args == (2*C*D, 4*D*C)


def test_exp():
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 2, 2)
    expr1 = exp(A)*exp(B)
    expr2 = exp(B)*exp(A)
    assert expr1 != expr2
    assert expr1 - expr2 != 0
    assert not isinstance(expr1, exp)
    assert not isinstance(expr2, exp)


def test_invalid_args():
    raises(SympifyError, lambda: MatrixSymbol(1, 2, 'A'))


def test_matrixsymbol_from_symbol():
    # The label should be preserved during doit and subs
    A_label = Symbol('A', complex=True)
    A = MatrixSymbol(A_label, 2, 2)

    A_1 = A.doit()
    A_2 = A.subs(2, 3)
    assert A_1.args == A.args
    assert A_2.args[0] == A.args[0]


def test_as_explicit():
    Z = MatrixSymbol('Z', 2, 3)
    assert Z.as_explicit() == ImmutableMatrix([
        [Z[0, 0], Z[0, 1], Z[0, 2]],
        [Z[1, 0], Z[1, 1], Z[1, 2]],
    ])
    raises(ValueError, lambda: A.as_explicit())


def test_MatrixSet():
    M = MatrixSet(2, 2, set=S.Reals)
    assert M.shape == (2, 2)
    assert M.set == S.Reals
    X = Matrix([[1, 2], [3, 4]])
    assert X in M
    X = ZeroMatrix(2, 2)
    assert X in M
    raises(TypeError, lambda: A in M)
    raises(TypeError, lambda: 1 in M)
    M = MatrixSet(n, m, set=S.Reals)
    assert A in M
    raises(TypeError, lambda: C in M)
    raises(TypeError, lambda: X in M)
    M = MatrixSet(2, 2, set={1, 2, 3})
    X = Matrix([[1, 2], [3, 4]])
    Y = Matrix([[1, 2]])
    assert (X in M) == S.false
    assert (Y in M) == S.false
    raises(ValueError, lambda: MatrixSet(2, -2, S.Reals))
    raises(ValueError, lambda: MatrixSet(2.4, -1, S.Reals))
    raises(TypeError, lambda: MatrixSet(2, 2, (1, 2, 3)))


def test_matrixsymbol_solving():
    A = MatrixSymbol('A', 2, 2)
    B = MatrixSymbol('B', 2, 2)
    Z = ZeroMatrix(2, 2)
    assert -(-A + B) - A + B == Z
    assert (-(-A + B) - A + B).simplify() == Z
    assert (-(-A + B) - A + B).expand() == Z
    assert (-(-A + B) - A + B - Z).simplify() == Z
    assert (-(-A + B) - A + B - Z).expand() == Z
    assert (A*(A + B) + B*(A.T + B.T)).expand() == A**2 + A*B + B*A.T + B*B.T

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\fft\tests\test_pocketfft.py ===
import queue
import threading

import pytest

import numpy as np
from numpy.random import random
from numpy.testing import IS_WASM, assert_allclose, assert_array_equal, assert_raises


def fft1(x):
    L = len(x)
    phase = -2j * np.pi * (np.arange(L) / L)
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x * np.exp(phase), axis=1)


class TestFFTShift:

    def test_fft_n(self):
        assert_raises(ValueError, np.fft.fft, [1, 2, 3], 0)


class TestFFT1D:

    def test_identity(self):
        maxlen = 512
        x = random(maxlen) + 1j * random(maxlen)
        xr = random(maxlen)
        for i in range(1, maxlen):
            assert_allclose(np.fft.ifft(np.fft.fft(x[0:i])), x[0:i],
                            atol=1e-12)
            assert_allclose(np.fft.irfft(np.fft.rfft(xr[0:i]), i),
                            xr[0:i], atol=1e-12)

    @pytest.mark.parametrize("dtype", [np.single, np.double, np.longdouble])
    def test_identity_long_short(self, dtype):
        # Test with explicitly given number of points, both for n
        # smaller and for n larger than the input size.
        maxlen = 16
        atol = 5 * np.spacing(np.array(1., dtype=dtype))
        x = random(maxlen).astype(dtype) + 1j * random(maxlen).astype(dtype)
        xx = np.concatenate([x, np.zeros_like(x)])
        xr = random(maxlen).astype(dtype)
        xxr = np.concatenate([xr, np.zeros_like(xr)])
        for i in range(1, maxlen * 2):
            check_c = np.fft.ifft(np.fft.fft(x, n=i), n=i)
            assert check_c.real.dtype == dtype
            assert_allclose(check_c, xx[0:i], atol=atol, rtol=0)
            check_r = np.fft.irfft(np.fft.rfft(xr, n=i), n=i)
            assert check_r.dtype == dtype
            assert_allclose(check_r, xxr[0:i], atol=atol, rtol=0)

    @pytest.mark.parametrize("dtype", [np.single, np.double, np.longdouble])
    def test_identity_long_short_reversed(self, dtype):
        # Also test explicitly given number of points in reversed order.
        maxlen = 16
        atol = 5 * np.spacing(np.array(1., dtype=dtype))
        x = random(maxlen).astype(dtype) + 1j * random(maxlen).astype(dtype)
        xx = np.concatenate([x, np.zeros_like(x)])
        for i in range(1, maxlen * 2):
            check_via_c = np.fft.fft(np.fft.ifft(x, n=i), n=i)
            assert check_via_c.dtype == x.dtype
            assert_allclose(check_via_c, xx[0:i], atol=atol, rtol=0)
            # For irfft, we can neither recover the imaginary part of
            # the first element, nor the imaginary part of the last
            # element if npts is even.  So, set to 0 for the comparison.
            y = x.copy()
            n = i // 2 + 1
            y.imag[0] = 0
            if i % 2 == 0:
                y.imag[n - 1:] = 0
            yy = np.concatenate([y, np.zeros_like(y)])
            check_via_r = np.fft.rfft(np.fft.irfft(x, n=i), n=i)
            assert check_via_r.dtype == x.dtype
            assert_allclose(check_via_r, yy[0:n], atol=atol, rtol=0)

    def test_fft(self):
        x = random(30) + 1j * random(30)
        assert_allclose(fft1(x), np.fft.fft(x), atol=1e-6)
        assert_allclose(fft1(x), np.fft.fft(x, norm="backward"), atol=1e-6)
        assert_allclose(fft1(x) / np.sqrt(30),
                        np.fft.fft(x, norm="ortho"), atol=1e-6)
        assert_allclose(fft1(x) / 30.,
                        np.fft.fft(x, norm="forward"), atol=1e-6)

    @pytest.mark.parametrize("axis", (0, 1))
    @pytest.mark.parametrize("dtype", (complex, float))
    @pytest.mark.parametrize("transpose", (True, False))
    def test_fft_out_argument(self, dtype, transpose, axis):
        def zeros_like(x):
            if transpose:
                return np.zeros_like(x.T).T
            else:
                return np.zeros_like(x)

        # tests below only test the out parameter
        if dtype is complex:
            y = random((10, 20)) + 1j * random((10, 20))
            fft, ifft = np.fft.fft, np.fft.ifft
        else:
            y = random((10, 20))
            fft, ifft = np.fft.rfft, np.fft.irfft

        expected = fft(y, axis=axis)
        out = zeros_like(expected)
        result = fft(y, out=out, axis=axis)
        assert result is out
        assert_array_equal(result, expected)

        expected2 = ifft(expected, axis=axis)
        out2 = out if dtype is complex else zeros_like(expected2)
        result2 = ifft(out, out=out2, axis=axis)
        assert result2 is out2
        assert_array_equal(result2, expected2)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_fft_inplace_out(self, axis):
        # Test some weirder in-place combinations
        y = random((20, 20)) + 1j * random((20, 20))
        # Fully in-place.
        y1 = y.copy()
        expected1 = np.fft.fft(y1, axis=axis)
        result1 = np.fft.fft(y1, axis=axis, out=y1)
        assert result1 is y1
        assert_array_equal(result1, expected1)
        # In-place of part of the array; rest should be unchanged.
        y2 = y.copy()
        out2 = y2[:10] if axis == 0 else y2[:, :10]
        expected2 = np.fft.fft(y2, n=10, axis=axis)
        result2 = np.fft.fft(y2, n=10, axis=axis, out=out2)
        assert result2 is out2
        assert_array_equal(result2, expected2)
        if axis == 0:
            assert_array_equal(y2[10:], y[10:])
        else:
            assert_array_equal(y2[:, 10:], y[:, 10:])
        # In-place of another part of the array.
        y3 = y.copy()
        y3_sel = y3[5:] if axis == 0 else y3[:, 5:]
        out3 = y3[5:15] if axis == 0 else y3[:, 5:15]
        expected3 = np.fft.fft(y3_sel, n=10, axis=axis)
        result3 = np.fft.fft(y3_sel, n=10, axis=axis, out=out3)
        assert result3 is out3
        assert_array_equal(result3, expected3)
        if axis == 0:
            assert_array_equal(y3[:5], y[:5])
            assert_array_equal(y3[15:], y[15:])
        else:
            assert_array_equal(y3[:, :5], y[:, :5])
            assert_array_equal(y3[:, 15:], y[:, 15:])
        # In-place with n > nin; rest should be unchanged.
        y4 = y.copy()
        y4_sel = y4[:10] if axis == 0 else y4[:, :10]
        out4 = y4[:15] if axis == 0 else y4[:, :15]
        expected4 = np.fft.fft(y4_sel, n=15, axis=axis)
        result4 = np.fft.fft(y4_sel, n=15, axis=axis, out=out4)
        assert result4 is out4
        assert_array_equal(result4, expected4)
        if axis == 0:
            assert_array_equal(y4[15:], y[15:])
        else:
            assert_array_equal(y4[:, 15:], y[:, 15:])
        # Overwrite in a transpose.
        y5 = y.copy()
        out5 = y5.T
        result5 = np.fft.fft(y5, axis=axis, out=out5)
        assert result5 is out5
        assert_array_equal(result5, expected1)
        # Reverse strides.
        y6 = y.copy()
        out6 = y6[::-1] if axis == 0 else y6[:, ::-1]
        result6 = np.fft.fft(y6, axis=axis, out=out6)
        assert result6 is out6
        assert_array_equal(result6, expected1)

    def test_fft_bad_out(self):
        x = np.arange(30.)
        with pytest.raises(TypeError, match="must be of ArrayType"):
            np.fft.fft(x, out="")
        with pytest.raises(ValueError, match="has wrong shape"):
            np.fft.fft(x, out=np.zeros_like(x).reshape(5, -1))
        with pytest.raises(TypeError, match="Cannot cast"):
            np.fft.fft(x, out=np.zeros_like(x, dtype=float))

    @pytest.mark.parametrize('norm', (None, 'backward', 'ortho', 'forward'))
    def test_ifft(self, norm):
        x = random(30) + 1j * random(30)
        assert_allclose(
            x, np.fft.ifft(np.fft.fft(x, norm=norm), norm=norm),
            atol=1e-6)
        # Ensure we get the correct error message
        with pytest.raises(ValueError,
                           match='Invalid number of FFT data points'):
            np.fft.ifft([], norm=norm)

    def test_fft2(self):
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(np.fft.fft(np.fft.fft(x, axis=1), axis=0),
                        np.fft.fft2(x), atol=1e-6)
        assert_allclose(np.fft.fft2(x),
                        np.fft.fft2(x, norm="backward"), atol=1e-6)
        assert_allclose(np.fft.fft2(x) / np.sqrt(30 * 20),
                        np.fft.fft2(x, norm="ortho"), atol=1e-6)
        assert_allclose(np.fft.fft2(x) / (30. * 20.),
                        np.fft.fft2(x, norm="forward"), atol=1e-6)

    def test_ifft2(self):
        x = random((30, 20)) + 1j * random((30, 20))
        assert_allclose(np.fft.ifft(np.fft.ifft(x, axis=1), axis=0),
                        np.fft.ifft2(x), atol=1e-6)
        assert_allclose(np.fft.ifft2(x),
                        np.fft.ifft2(x, norm="backward"), atol=1e-6)
        assert_allclose(np.fft.ifft2(x) * np.sqrt(30 * 20),
                        np.fft.ifft2(x, norm="ortho"), atol=1e-6)
        assert_allclose(np.fft.ifft2(x) * (30. * 20.),
                        np.fft.ifft2(x, norm="forward"), atol=1e-6)

    def test_fftn(self):
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        assert_allclose(
            np.fft.fft(np.fft.fft(np.fft.fft(x, axis=2), axis=1), axis=0),
            np.fft.fftn(x), atol=1e-6)
        assert_allclose(np.fft.fftn(x),
                        np.fft.fftn(x, norm="backward"), atol=1e-6)
        assert_allclose(np.fft.fftn(x) / np.sqrt(30 * 20 * 10),
                        np.fft.fftn(x, norm="ortho"), atol=1e-6)
        assert_allclose(np.fft.fftn(x) / (30. * 20. * 10.),
                        np.fft.fftn(x, norm="forward"), atol=1e-6)

    def test_ifftn(self):
        x = random((30, 20, 10)) + 1j * random((30, 20, 10))
        assert_allclose(
            np.fft.ifft(np.fft.ifft(np.fft.ifft(x, axis=2), axis=1), axis=0),
            np.fft.ifftn(x), atol=1e-6)
        assert_allclose(np.fft.ifftn(x),
                        np.fft.ifftn(x, norm="backward"), atol=1e-6)
        assert_allclose(np.fft.ifftn(x) * np.sqrt(30 * 20 * 10),
                        np.fft.ifftn(x, norm="ortho"), atol=1e-6)
        assert_allclose(np.fft.ifftn(x) * (30. * 20. * 10.),
                        np.fft.ifftn(x, norm="forward"), atol=1e-6)

    def test_rfft(self):
        x = random(30)
        for n in [x.size, 2 * x.size]:
            for norm in [None, 'backward', 'ortho', 'forward']:
                assert_allclose(
                    np.fft.fft(x, n=n, norm=norm)[:(n // 2 + 1)],
                    np.fft.rfft(x, n=n, norm=norm), atol=1e-6)
            assert_allclose(
                np.fft.rfft(x, n=n),
                np.fft.rfft(x, n=n, norm="backward"), atol=1e-6)
            assert_allclose(
                np.fft.rfft(x, n=n) / np.sqrt(n),
                np.fft.rfft(x, n=n, norm="ortho"), atol=1e-6)
            assert_allclose(
                np.fft.rfft(x, n=n) / n,
                np.fft.rfft(x, n=n, norm="forward"), atol=1e-6)

    def test_rfft_even(self):
        x = np.arange(8)
        n = 4
        y = np.fft.rfft(x, n)
        assert_allclose(y, np.fft.fft(x[:n])[:n // 2 + 1], rtol=1e-14)

    def test_rfft_odd(self):
        x = np.array([1, 0, 2, 3, -3])
        y = np.fft.rfft(x)
        assert_allclose(y, np.fft.fft(x)[:3], rtol=1e-14)

    def test_irfft(self):
        x = random(30)
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x)), atol=1e-6)
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x, norm="backward"),
                        norm="backward"), atol=1e-6)
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x, norm="ortho"),
                        norm="ortho"), atol=1e-6)
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x, norm="forward"),
                        norm="forward"), atol=1e-6)

    def test_rfft2(self):
        x = random((30, 20))
        assert_allclose(np.fft.fft2(x)[:, :11], np.fft.rfft2(x), atol=1e-6)
        assert_allclose(np.fft.rfft2(x),
                        np.fft.rfft2(x, norm="backward"), atol=1e-6)
        assert_allclose(np.fft.rfft2(x) / np.sqrt(30 * 20),
                        np.fft.rfft2(x, norm="ortho"), atol=1e-6)
        assert_allclose(np.fft.rfft2(x) / (30. * 20.),
                        np.fft.rfft2(x, norm="forward"), atol=1e-6)

    def test_irfft2(self):
        x = random((30, 20))
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x)), atol=1e-6)
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x, norm="backward"),
                        norm="backward"), atol=1e-6)
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x, norm="ortho"),
                        norm="ortho"), atol=1e-6)
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x, norm="forward"),
                        norm="forward"), atol=1e-6)

    def test_rfftn(self):
        x = random((30, 20, 10))
        assert_allclose(np.fft.fftn(x)[:, :, :6], np.fft.rfftn(x), atol=1e-6)
        assert_allclose(np.fft.rfftn(x),
                        np.fft.rfftn(x, norm="backward"), atol=1e-6)
        assert_allclose(np.fft.rfftn(x) / np.sqrt(30 * 20 * 10),
                        np.fft.rfftn(x, norm="ortho"), atol=1e-6)
        assert_allclose(np.fft.rfftn(x) / (30. * 20. * 10.),
                        np.fft.rfftn(x, norm="forward"), atol=1e-6)
        # Regression test for gh-27159
        x = np.ones((2, 3))
        result = np.fft.rfftn(x, axes=(0, 0, 1), s=(10, 20, 40))
        assert result.shape == (10, 21)
        expected = np.fft.fft(np.fft.fft(np.fft.rfft(x, axis=1, n=40),
                            axis=0, n=20), axis=0, n=10)
        assert expected.shape == (10, 21)
        assert_allclose(result, expected, atol=1e-6)

    def test_irfftn(self):
        x = random((30, 20, 10))
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x)), atol=1e-6)
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x, norm="backward"),
                        norm="backward"), atol=1e-6)
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x, norm="ortho"),
                        norm="ortho"), atol=1e-6)
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x, norm="forward"),
                        norm="forward"), atol=1e-6)

    def test_hfft(self):
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        assert_allclose(np.fft.fft(x), np.fft.hfft(x_herm), atol=1e-6)
        assert_allclose(np.fft.hfft(x_herm),
                        np.fft.hfft(x_herm, norm="backward"), atol=1e-6)
        assert_allclose(np.fft.hfft(x_herm) / np.sqrt(30),
                        np.fft.hfft(x_herm, norm="ortho"), atol=1e-6)
        assert_allclose(np.fft.hfft(x_herm) / 30.,
                        np.fft.hfft(x_herm, norm="forward"), atol=1e-6)

    def test_ihfft(self):
        x = random(14) + 1j * random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm)), atol=1e-6)
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm,
                        norm="backward"), norm="backward"), atol=1e-6)
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm,
                        norm="ortho"), norm="ortho"), atol=1e-6)
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm,
                        norm="forward"), norm="forward"), atol=1e-6)

    @pytest.mark.parametrize("op", [np.fft.fftn, np.fft.ifftn,
                                    np.fft.rfftn, np.fft.irfftn])
    def test_axes(self, op):
        x = random((30, 20, 10))
        axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        for a in axes:
            op_tr = op(np.transpose(x, a))
            tr_op = np.transpose(op(x, axes=a), a)
            assert_allclose(op_tr, tr_op, atol=1e-6)

    @pytest.mark.parametrize("op", [np.fft.fftn, np.fft.ifftn,
                                    np.fft.fft2, np.fft.ifft2])
    def test_s_negative_1(self, op):
        x = np.arange(100).reshape(10, 10)
        # should use the whole input array along the first axis
        assert op(x, s=(-1, 5), axes=(0, 1)).shape == (10, 5)

    @pytest.mark.parametrize("op", [np.fft.fftn, np.fft.ifftn,
                                    np.fft.rfftn, np.fft.irfftn])
    def test_s_axes_none(self, op):
        x = np.arange(100).reshape(10, 10)
        with pytest.warns(match='`axes` should not be `None` if `s`'):
            op(x, s=(-1, 5))

    @pytest.mark.parametrize("op", [np.fft.fft2, np.fft.ifft2])
    def test_s_axes_none_2D(self, op):
        x = np.arange(100).reshape(10, 10)
        with pytest.warns(match='`axes` should not be `None` if `s`'):
            op(x, s=(-1, 5), axes=None)

    @pytest.mark.parametrize("op", [np.fft.fftn, np.fft.ifftn,
                                    np.fft.rfftn, np.fft.irfftn,
                                    np.fft.fft2, np.fft.ifft2])
    def test_s_contains_none(self, op):
        x = random((30, 20, 10))
        with pytest.warns(match='array containing `None` values to `s`'):
            op(x, s=(10, None, 10), axes=(0, 1, 2))

    def test_all_1d_norm_preserving(self):
        # verify that round-trip transforms are norm-preserving
        x = random(30)
        x_norm = np.linalg.norm(x)
        n = x.size * 2
        func_pairs = [(np.fft.fft, np.fft.ifft),
                      (np.fft.rfft, np.fft.irfft),
                      # hfft: order so the first function takes x.size samples
                      #       (necessary for comparison to x_norm above)
                      (np.fft.ihfft, np.fft.hfft),
                      ]
        for forw, back in func_pairs:
            for n in [x.size, 2 * x.size]:
                for norm in [None, 'backward', 'ortho', 'forward']:
                    tmp = forw(x, n=n, norm=norm)
                    tmp = back(tmp, n=n, norm=norm)
                    assert_allclose(x_norm,
                                    np.linalg.norm(tmp), atol=1e-6)

    @pytest.mark.parametrize("axes", [(0, 1), (0, 2), None])
    @pytest.mark.parametrize("dtype", (complex, float))
    @pytest.mark.parametrize("transpose", (True, False))
    def test_fftn_out_argument(self, dtype, transpose, axes):
        def zeros_like(x):
            if transpose:
                return np.zeros_like(x.T).T
            else:
                return np.zeros_like(x)

        # tests below only test the out parameter
        if dtype is complex:
            x = random((10, 5, 6)) + 1j * random((10, 5, 6))
            fft, ifft = np.fft.fftn, np.fft.ifftn
        else:
            x = random((10, 5, 6))
            fft, ifft = np.fft.rfftn, np.fft.irfftn

        expected = fft(x, axes=axes)
        out = zeros_like(expected)
        result = fft(x, out=out, axes=axes)
        assert result is out
        assert_array_equal(result, expected)

        expected2 = ifft(expected, axes=axes)
        out2 = out if dtype is complex else zeros_like(expected2)
        result2 = ifft(out, out=out2, axes=axes)
        assert result2 is out2
        assert_array_equal(result2, expected2)

    @pytest.mark.parametrize("fft", [np.fft.fftn, np.fft.ifftn, np.fft.rfftn])
    def test_fftn_out_and_s_interaction(self, fft):
        # With s, shape varies, so generally one cannot pass in out.
        if fft is np.fft.rfftn:
            x = random((10, 5, 6))
        else:
            x = random((10, 5, 6)) + 1j * random((10, 5, 6))
        with pytest.raises(ValueError, match="has wrong shape"):
            fft(x, out=np.zeros_like(x), s=(3, 3, 3), axes=(0, 1, 2))
        # Except on the first axis done (which is the last of axes).
        s = (10, 5, 5)
        expected = fft(x, s=s, axes=(0, 1, 2))
        out = np.zeros_like(expected)
        result = fft(x, s=s, axes=(0, 1, 2), out=out)
        assert result is out
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("s", [(9, 5, 5), (3, 3, 3)])
    def test_irfftn_out_and_s_interaction(self, s):
        # Since for irfftn, the output is real and thus cannot be used for
        # intermediate steps, it should always work.
        x = random((9, 5, 6, 2)) + 1j * random((9, 5, 6, 2))
        expected = np.fft.irfftn(x, s=s, axes=(0, 1, 2))
        out = np.zeros_like(expected)
        result = np.fft.irfftn(x, s=s, axes=(0, 1, 2), out=out)
        assert result is out
        assert_array_equal(result, expected)


@pytest.mark.parametrize(
        "dtype",
        [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("order", ["F", 'non-contiguous'])
@pytest.mark.parametrize(
        "fft",
        [np.fft.fft, np.fft.fft2, np.fft.fftn,
         np.fft.ifft, np.fft.ifft2, np.fft.ifftn])
def test_fft_with_order(dtype, order, fft):
    # Check that FFT/IFFT produces identical results for C, Fortran and
    # non contiguous arrays
    rng = np.random.RandomState(42)
    X = rng.rand(8, 7, 13).astype(dtype, copy=False)
    # See discussion in pull/14178
    _tol = 8.0 * np.sqrt(np.log2(X.size)) * np.finfo(X.dtype).eps
    if order == 'F':
        Y = np.asfortranarray(X)
    else:
        # Make a non contiguous array
        Y = X[::-1]
        X = np.ascontiguousarray(X[::-1])

    if fft.__name__.endswith('fft'):
        for axis in range(3):
            X_res = fft(X, axis=axis)
            Y_res = fft(Y, axis=axis)
            assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
    elif fft.__name__.endswith(('fft2', 'fftn')):
        axes = [(0, 1), (1, 2), (0, 2)]
        if fft.__name__.endswith('fftn'):
            axes.extend([(0,), (1,), (2,), None])
        for ax in axes:
            X_res = fft(X, axes=ax)
            Y_res = fft(Y, axes=ax)
            assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
    else:
        raise ValueError


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("n", [None, 7, 12])
def test_fft_output_order(order, n):
    rng = np.random.RandomState(42)
    x = rng.rand(10)
    x = np.asarray(x, dtype=np.complex64, order=order)
    res = np.fft.fft(x, n=n)
    assert res.flags.c_contiguous == x.flags.c_contiguous
    assert res.flags.f_contiguous == x.flags.f_contiguous

@pytest.mark.skipif(IS_WASM, reason="Cannot start thread")
class TestFFTThreadSafe:
    threads = 16
    input_shape = (800, 200)

    def _test_mtsame(self, func, *args):
        def worker(args, q):
            q.put(func(*args))

        q = queue.Queue()
        expected = func(*args)

        # Spin off a bunch of threads to call the same function simultaneously
        t = [threading.Thread(target=worker, args=(args, q))
             for i in range(self.threads)]
        [x.start() for x in t]

        [x.join() for x in t]
        # Make sure all threads returned the correct value
        for i in range(self.threads):
            assert_array_equal(q.get(timeout=5), expected,
                'Function returned wrong value in multithreaded context')

    def test_fft(self):
        a = np.ones(self.input_shape) * 1 + 0j
        self._test_mtsame(np.fft.fft, a)

    def test_ifft(self):
        a = np.ones(self.input_shape) * 1 + 0j
        self._test_mtsame(np.fft.ifft, a)

    def test_rfft(self):
        a = np.ones(self.input_shape)
        self._test_mtsame(np.fft.rfft, a)

    def test_irfft(self):
        a = np.ones(self.input_shape) * 1 + 0j
        self._test_mtsame(np.fft.irfft, a)


def test_irfft_with_n_1_regression():
    # Regression test for gh-25661
    x = np.arange(10)
    np.fft.irfft(x, n=1)
    np.fft.hfft(x, n=1)
    np.fft.irfft(np.array([0], complex), n=10)


def test_irfft_with_n_large_regression():
    # Regression test for gh-25679
    x = np.arange(5) * (1 + 1j)
    result = np.fft.hfft(x, n=10)
    expected = np.array([20., 9.91628173, -11.8819096, 7.1048486,
                         -6.62459848, 4., -3.37540152, -0.16057669,
                         1.8819096, -20.86055364])
    assert_allclose(result, expected)


@pytest.mark.parametrize("fft", [
    np.fft.fft, np.fft.ifft, np.fft.rfft, np.fft.irfft
])
@pytest.mark.parametrize("data", [
    np.array([False, True, False]),
    np.arange(10, dtype=np.uint8),
    np.arange(5, dtype=np.int16),
])
def test_fft_with_integer_or_bool_input(data, fft):
    # Regression test for gh-25819
    result = fft(data)
    float_data = data.astype(np.result_type(data, 1.))
    expected = fft(float_data)
    assert_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_info.py ===
from io import StringIO
import re
from string import ascii_uppercase
import sys
import textwrap

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import (
    HAS_PYARROW,
    IS64,
    PYPY,
    is_platform_arm,
)

from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
    option_context,
)
import pandas._testing as tm
from pandas.util.version import Version


@pytest.fixture
def duplicate_columns_frame():
    """Dataframe with duplicate column names."""
    return DataFrame(
        np.random.default_rng(2).standard_normal((1500, 4)),
        columns=["a", "a", "b", "b"],
    )


def test_info_empty():
    # GH #45494
    df = DataFrame()
    buf = StringIO()
    df.info(buf=buf)
    result = buf.getvalue()
    expected = textwrap.dedent(
        """\
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 0 entries
        Empty DataFrame\n"""
    )
    assert result == expected


def test_info_categorical_column_smoke_test():
    n = 2500
    df = DataFrame({"int64": np.random.default_rng(2).integers(100, size=n, dtype=int)})
    df["category"] = Series(
        np.array(list("abcdefghij")).take(
            np.random.default_rng(2).integers(0, 10, size=n, dtype=int)
        )
    ).astype("category")
    df.isna()
    buf = StringIO()
    df.info(buf=buf)

    df2 = df[df["category"] == "d"]
    buf = StringIO()
    df2.info(buf=buf)


@pytest.mark.parametrize(
    "fixture_func_name",
    [
        "int_frame",
        "float_frame",
        "datetime_frame",
        "duplicate_columns_frame",
        "float_string_frame",
    ],
)
def test_info_smoke_test(fixture_func_name, request):
    frame = request.getfixturevalue(fixture_func_name)
    buf = StringIO()
    frame.info(buf=buf)
    result = buf.getvalue().splitlines()
    assert len(result) > 10

    buf = StringIO()
    frame.info(buf=buf, verbose=False)


def test_info_smoke_test2(float_frame):
    # pretty useless test, used to be mixed into the repr tests
    buf = StringIO()
    float_frame.reindex(columns=["A"]).info(verbose=False, buf=buf)
    float_frame.reindex(columns=["A", "B"]).info(verbose=False, buf=buf)

    # no columns or index
    DataFrame().info(buf=buf)


@pytest.mark.parametrize(
    "num_columns, max_info_columns, verbose",
    [
        (10, 100, True),
        (10, 11, True),
        (10, 10, True),
        (10, 9, False),
        (10, 1, False),
    ],
)
def test_info_default_verbose_selection(num_columns, max_info_columns, verbose):
    frame = DataFrame(np.random.default_rng(2).standard_normal((5, num_columns)))
    with option_context("display.max_info_columns", max_info_columns):
        io_default = StringIO()
        frame.info(buf=io_default)
        result = io_default.getvalue()

        io_explicit = StringIO()
        frame.info(buf=io_explicit, verbose=verbose)
        expected = io_explicit.getvalue()

        assert result == expected


def test_info_verbose_check_header_separator_body():
    buf = StringIO()
    size = 1001
    start = 5
    frame = DataFrame(np.random.default_rng(2).standard_normal((3, size)))
    frame.info(verbose=True, buf=buf)

    res = buf.getvalue()
    header = " #     Column  Dtype  \n---    ------  -----  "
    assert header in res

    frame.info(verbose=True, buf=buf)
    buf.seek(0)
    lines = buf.readlines()
    assert len(lines) > 0

    for i, line in enumerate(lines):
        if start <= i < start + size:
            line_nr = f" {i - start} "
            assert line.startswith(line_nr)


@pytest.mark.parametrize(
    "size, header_exp, separator_exp, first_line_exp, last_line_exp",
    [
        (
            4,
            " #   Column  Non-Null Count  Dtype  ",
            "---  ------  --------------  -----  ",
            " 0   0       3 non-null      float64",
            " 3   3       3 non-null      float64",
        ),
        (
            11,
            " #   Column  Non-Null Count  Dtype  ",
            "---  ------  --------------  -----  ",
            " 0   0       3 non-null      float64",
            " 10  10      3 non-null      float64",
        ),
        (
            101,
            " #    Column  Non-Null Count  Dtype  ",
            "---   ------  --------------  -----  ",
            " 0    0       3 non-null      float64",
            " 100  100     3 non-null      float64",
        ),
        (
            1001,
            " #     Column  Non-Null Count  Dtype  ",
            "---    ------  --------------  -----  ",
            " 0     0       3 non-null      float64",
            " 1000  1000    3 non-null      float64",
        ),
        (
            10001,
            " #      Column  Non-Null Count  Dtype  ",
            "---     ------  --------------  -----  ",
            " 0      0       3 non-null      float64",
            " 10000  10000   3 non-null      float64",
        ),
    ],
)
def test_info_verbose_with_counts_spacing(
    size, header_exp, separator_exp, first_line_exp, last_line_exp
):
    """Test header column, spacer, first line and last line in verbose mode."""
    frame = DataFrame(np.random.default_rng(2).standard_normal((3, size)))
    with StringIO() as buf:
        frame.info(verbose=True, show_counts=True, buf=buf)
        all_lines = buf.getvalue().splitlines()
    # Here table would contain only header, separator and table lines
    # dframe repr, index summary, memory usage and dtypes are excluded
    table = all_lines[3:-2]
    header, separator, first_line, *rest, last_line = table
    assert header == header_exp
    assert separator == separator_exp
    assert first_line == first_line_exp
    assert last_line == last_line_exp


def test_info_memory():
    # https://github.com/pandas-dev/pandas/issues/21056
    df = DataFrame({"a": Series([1, 2], dtype="i8")})
    buf = StringIO()
    df.info(buf=buf)
    result = buf.getvalue()
    bytes = float(df.memory_usage().sum())
    expected = textwrap.dedent(
        f"""\
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2 entries, 0 to 1
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       2 non-null      int64
    dtypes: int64(1)
    memory usage: {bytes} bytes
    """
    )
    assert result == expected


def test_info_wide():
    io = StringIO()
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 101)))
    df.info(buf=io)

    io = StringIO()
    df.info(buf=io, max_cols=101)
    result = io.getvalue()
    assert len(result.splitlines()) > 100

    expected = result
    with option_context("display.max_info_columns", 101):
        io = StringIO()
        df.info(buf=io)
        result = io.getvalue()
        assert result == expected


def test_info_duplicate_columns_shows_correct_dtypes():
    # GH11761
    io = StringIO()
    frame = DataFrame([[1, 2.0]], columns=["a", "a"])
    frame.info(buf=io)
    lines = io.getvalue().splitlines(True)
    assert " 0   a       1 non-null      int64  \n" == lines[5]
    assert " 1   a       1 non-null      float64\n" == lines[6]


def test_info_shows_column_dtypes():
    dtypes = [
        "int64",
        "float64",
        "datetime64[ns]",
        "timedelta64[ns]",
        "complex128",
        "object",
        "bool",
    ]
    data = {}
    n = 10
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.default_rng(2).integers(2, size=n).astype(dtype)
    df = DataFrame(data)
    buf = StringIO()
    df.info(buf=buf)
    res = buf.getvalue()
    header = (
        " #   Column  Non-Null Count  Dtype          \n"
        "---  ------  --------------  -----          "
    )
    assert header in res
    for i, dtype in enumerate(dtypes):
        name = f" {i:d}   {i:d}       {n:d} non-null     {dtype}"
        assert name in res


def test_info_max_cols():
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    for len_, verbose in [(5, None), (5, False), (12, True)]:
        # For verbose always      ^ setting  ^ summarize ^ full output
        with option_context("max_info_columns", 4):
            buf = StringIO()
            df.info(buf=buf, verbose=verbose)
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_

    for len_, verbose in [(12, None), (5, False), (12, True)]:
        # max_cols not exceeded
        with option_context("max_info_columns", 5):
            buf = StringIO()
            df.info(buf=buf, verbose=verbose)
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_

    for len_, max_cols in [(12, 5), (5, 4)]:
        # setting truncates
        with option_context("max_info_columns", 4):
            buf = StringIO()
            df.info(buf=buf, max_cols=max_cols)
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_

        # setting wouldn't truncate
        with option_context("max_info_columns", 5):
            buf = StringIO()
            df.info(buf=buf, max_cols=max_cols)
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_


def test_info_memory_usage():
    # Ensure memory usage is displayed, when asserted, on the last line
    dtypes = [
        "int64",
        "float64",
        "datetime64[ns]",
        "timedelta64[ns]",
        "complex128",
        "object",
        "bool",
    ]
    data = {}
    n = 10
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.default_rng(2).integers(2, size=n).astype(dtype)
    df = DataFrame(data)
    buf = StringIO()

    # display memory usage case
    df.info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()
    assert "memory usage: " in res[-1]

    # do not display memory usage case
    df.info(buf=buf, memory_usage=False)
    res = buf.getvalue().splitlines()
    assert "memory usage: " not in res[-1]

    df.info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()

    # memory usage is a lower bound, so print it as XYZ+ MB
    assert re.match(r"memory usage: [^+]+\+", res[-1])

    df.iloc[:, :5].info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()

    # excluded column with object dtype, so estimate is accurate
    assert not re.match(r"memory usage: [^+]+\+", res[-1])

    # Test a DataFrame with duplicate columns
    dtypes = ["int64", "int64", "int64", "float64"]
    data = {}
    n = 100
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.default_rng(2).integers(2, size=n).astype(dtype)
    df = DataFrame(data)
    df.columns = dtypes

    df_with_object_index = DataFrame({"a": [1]}, index=Index(["foo"], dtype=object))
    df_with_object_index.info(buf=buf, memory_usage=True)
    res = buf.getvalue().splitlines()
    assert re.match(r"memory usage: [^+]+\+", res[-1])

    df_with_object_index.info(buf=buf, memory_usage="deep")
    res = buf.getvalue().splitlines()
    assert re.match(r"memory usage: [^+]+$", res[-1])

    # Ensure df size is as expected
    # (cols * rows * bytes) + index size
    df_size = df.memory_usage().sum()
    exp_size = len(dtypes) * n * 8 + df.index.nbytes
    assert df_size == exp_size

    # Ensure number of cols in memory_usage is the same as df
    size_df = np.size(df.columns.values) + 1  # index=True; default
    assert size_df == np.size(df.memory_usage())

    # assert deep works only on object
    assert df.memory_usage().sum() == df.memory_usage(deep=True).sum()

    # test for validity
    DataFrame(1, index=["a"], columns=["A"]).memory_usage(index=True)
    DataFrame(1, index=["a"], columns=["A"]).index.nbytes
    df = DataFrame(
        data=1, index=MultiIndex.from_product([["a"], range(1000)]), columns=["A"]
    )
    df.index.nbytes
    df.memory_usage(index=True)
    df.index.values.nbytes

    mem = df.memory_usage(deep=True).sum()
    assert mem > 0


@pytest.mark.skipif(PYPY, reason="on PyPy deep=True doesn't change result")
def test_info_memory_usage_deep_not_pypy():
    df_with_object_index = DataFrame({"a": [1]}, index=Index(["foo"], dtype=object))
    assert (
        df_with_object_index.memory_usage(index=True, deep=True).sum()
        > df_with_object_index.memory_usage(index=True).sum()
    )

    df_object = DataFrame({"a": Series(["a"], dtype=object)})
    assert df_object.memory_usage(deep=True).sum() > df_object.memory_usage().sum()


@pytest.mark.xfail(not PYPY, reason="on PyPy deep=True does not change result")
def test_info_memory_usage_deep_pypy():
    df_with_object_index = DataFrame({"a": [1]}, index=Index(["foo"], dtype=object))
    assert (
        df_with_object_index.memory_usage(index=True, deep=True).sum()
        == df_with_object_index.memory_usage(index=True).sum()
    )

    df_object = DataFrame({"a": Series(["a"], dtype=object)})
    assert df_object.memory_usage(deep=True).sum() == df_object.memory_usage().sum()


@pytest.mark.skipif(PYPY, reason="PyPy getsizeof() fails by design")
def test_usage_via_getsizeof():
    df = DataFrame(
        data=1, index=MultiIndex.from_product([["a"], range(1000)]), columns=["A"]
    )
    mem = df.memory_usage(deep=True).sum()
    # sys.getsizeof will call the .memory_usage with
    # deep=True, and add on some GC overhead
    diff = mem - sys.getsizeof(df)
    assert abs(diff) < 100


def test_info_memory_usage_qualified(using_infer_string):
    buf = StringIO()
    df = DataFrame(1, columns=list("ab"), index=[1, 2, 3])
    df.info(buf=buf)
    assert "+" not in buf.getvalue()

    buf = StringIO()
    df = DataFrame(1, columns=list("ab"), index=Index(list("ABC"), dtype=object))
    df.info(buf=buf)
    assert "+" in buf.getvalue()

    buf = StringIO()
    df = DataFrame(1, columns=list("ab"), index=Index(list("ABC"), dtype="str"))
    df.info(buf=buf)
    if using_infer_string and HAS_PYARROW:
        assert "+" not in buf.getvalue()
    else:
        assert "+" in buf.getvalue()

    buf = StringIO()
    df = DataFrame(
        1, columns=list("ab"), index=MultiIndex.from_product([range(3), range(3)])
    )
    df.info(buf=buf)
    assert "+" not in buf.getvalue()

    buf = StringIO()
    df = DataFrame(
        1, columns=list("ab"), index=MultiIndex.from_product([range(3), ["foo", "bar"]])
    )
    df.info(buf=buf)
    if using_infer_string and HAS_PYARROW:
        assert "+" not in buf.getvalue()
    else:
        assert "+" in buf.getvalue()


def test_info_memory_usage_bug_on_multiindex():
    # GH 14308
    # memory usage introspection should not materialize .values

    def memory_usage(f):
        return f.memory_usage(deep=True).sum()

    N = 100
    M = len(ascii_uppercase)
    index = MultiIndex.from_product(
        [list(ascii_uppercase), date_range("20160101", periods=N)],
        names=["id", "date"],
    )
    df = DataFrame(
        {"value": np.random.default_rng(2).standard_normal(N * M)}, index=index
    )

    unstacked = df.unstack("id")
    assert df.values.nbytes == unstacked.values.nbytes
    assert memory_usage(df) > memory_usage(unstacked)

    # high upper bound
    assert memory_usage(unstacked) - memory_usage(df) < 2000


def test_info_categorical():
    # GH14298
    idx = CategoricalIndex(["a", "b"])
    df = DataFrame(np.zeros((2, 2)), index=idx, columns=idx)

    buf = StringIO()
    df.info(buf=buf)


@pytest.mark.xfail(not IS64, reason="GH 36579: fail on 32-bit system")
def test_info_int_columns(using_infer_string):
    # GH#37245
    df = DataFrame({1: [1, 2], 2: [2, 3]}, index=["A", "B"])
    buf = StringIO()
    df.info(show_counts=True, buf=buf)
    result = buf.getvalue()
    expected = textwrap.dedent(
        f"""\
        <class 'pandas.core.frame.DataFrame'>
        Index: 2 entries, A to B
        Data columns (total 2 columns):
         #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
         0   1       2 non-null      int64
         1   2       2 non-null      int64
        dtypes: int64(2)
        memory usage: {'50.0' if using_infer_string and HAS_PYARROW else '48.0+'} bytes
        """
    )
    assert result == expected


@pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
def test_memory_usage_empty_no_warning(using_infer_string):
    # GH#50066
    df = DataFrame(index=["a", "b"])
    with tm.assert_produces_warning(None):
        result = df.memory_usage()
    if using_infer_string and HAS_PYARROW:
        value = 18
    else:
        value = 16 if IS64 else 8
    expected = Series(value, index=["Index"])
    tm.assert_series_equal(result, expected)


@pytest.mark.single_cpu
def test_info_compute_numba():
    # GH#51922
    numba = pytest.importorskip("numba")
    if Version(numba.__version__) == Version("0.61") and is_platform_arm():
        pytest.skip(f"Segfaults on ARM platforms with numba {numba.__version__}")
    df = DataFrame([[1, 2], [3, 4]])

    with option_context("compute.use_numba", True):
        buf = StringIO()
        df.info(buf=buf)
        result = buf.getvalue()

    buf = StringIO()
    df.info(buf=buf)
    expected = buf.getvalue()
    assert result == expected


@pytest.mark.parametrize(
    "row, columns, show_counts, result",
    [
        [20, 20, None, True],
        [20, 20, True, True],
        [20, 20, False, False],
        [5, 5, None, False],
        [5, 5, True, False],
        [5, 5, False, False],
    ],
)
def test_info_show_counts(row, columns, show_counts, result):
    # Explicit cast to float to avoid implicit cast when setting nan
    df = DataFrame(1, columns=range(10), index=range(10)).astype({1: "float"})
    df.iloc[1, 1] = np.nan

    with option_context(
        "display.max_info_rows", row, "display.max_info_columns", columns
    ):
        with StringIO() as buf:
            df.info(buf=buf, show_counts=show_counts)
            assert ("non-null" in buf.getvalue()) is result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_setitem.py ===
import numpy as np
import pytest

from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
    isna,
    notna,
)
import pandas._testing as tm


def assert_equal(a, b):
    assert a == b


class TestMultiIndexSetItem:
    def check(self, target, indexers, value, compare_fn=assert_equal, expected=None):
        target.loc[indexers] = value
        result = target.loc[indexers]
        if expected is None:
            expected = value
        compare_fn(result, expected)

    def test_setitem_multiindex(self):
        # GH#7190
        cols = ["A", "w", "l", "a", "x", "X", "d", "profit"]
        index = MultiIndex.from_product(
            [np.arange(0, 100), np.arange(0, 80)], names=["time", "firm"]
        )
        t, n = 0, 2

        df = DataFrame(
            np.nan,
            columns=cols,
            index=index,
        )
        self.check(target=df, indexers=((t, n), "X"), value=0)

        df = DataFrame(-999, columns=cols, index=index)
        self.check(target=df, indexers=((t, n), "X"), value=1)

        df = DataFrame(columns=cols, index=index)
        self.check(target=df, indexers=((t, n), "X"), value=2)

        # gh-7218: assigning with 0-dim arrays
        df = DataFrame(-999, columns=cols, index=index)
        self.check(
            target=df,
            indexers=((t, n), "X"),
            value=np.array(3),
            expected=3,
        )

    def test_setitem_multiindex2(self):
        # GH#5206
        df = DataFrame(
            np.arange(25).reshape(5, 5), columns="A,B,C,D,E".split(","), dtype=float
        )
        df["F"] = 99
        row_selection = df["A"] % 2 == 0
        col_selection = ["B", "C"]
        df.loc[row_selection, col_selection] = df["F"]
        output = DataFrame(99.0, index=[0, 2, 4], columns=["B", "C"])
        tm.assert_frame_equal(df.loc[row_selection, col_selection], output)
        self.check(
            target=df,
            indexers=(row_selection, col_selection),
            value=df["F"],
            compare_fn=tm.assert_frame_equal,
            expected=output,
        )

    def test_setitem_multiindex3(self):
        # GH#11372
        idx = MultiIndex.from_product(
            [["A", "B", "C"], date_range("2015-01-01", "2015-04-01", freq="MS")]
        )
        cols = MultiIndex.from_product(
            [["foo", "bar"], date_range("2016-01-01", "2016-02-01", freq="MS")]
        )

        df = DataFrame(
            np.random.default_rng(2).random((12, 4)), index=idx, columns=cols
        )

        subidx = MultiIndex.from_arrays(
            [["A", "A"], date_range("2015-01-01", "2015-02-01", freq="MS")]
        )
        subcols = MultiIndex.from_arrays(
            [["foo", "foo"], date_range("2016-01-01", "2016-02-01", freq="MS")]
        )

        vals = DataFrame(
            np.random.default_rng(2).random((2, 2)), index=subidx, columns=subcols
        )
        self.check(
            target=df,
            indexers=(subidx, subcols),
            value=vals,
            compare_fn=tm.assert_frame_equal,
        )
        # set all columns
        vals = DataFrame(
            np.random.default_rng(2).random((2, 4)), index=subidx, columns=cols
        )
        self.check(
            target=df,
            indexers=(subidx, slice(None, None, None)),
            value=vals,
            compare_fn=tm.assert_frame_equal,
        )
        # identity
        copy = df.copy()
        self.check(
            target=df,
            indexers=(df.index, df.columns),
            value=df,
            compare_fn=tm.assert_frame_equal,
            expected=copy,
        )

    # TODO(ArrayManager) df.loc["bar"] *= 2 doesn't raise an error but results in
    # all NaNs -> doesn't work in the "split" path (also for BlockManager actually)
    @td.skip_array_manager_not_yet_implemented
    def test_multiindex_setitem(self):
        # GH 3738
        # setting with a multi-index right hand side
        arrays = [
            np.array(["bar", "bar", "baz", "qux", "qux", "bar"]),
            np.array(["one", "two", "one", "one", "two", "one"]),
            np.arange(0, 6, 1),
        ]

        df_orig = DataFrame(
            np.random.default_rng(2).standard_normal((6, 3)),
            index=arrays,
            columns=["A", "B", "C"],
        ).sort_index()

        expected = df_orig.loc[["bar"]] * 2
        df = df_orig.copy()
        df.loc[["bar"]] *= 2
        tm.assert_frame_equal(df.loc[["bar"]], expected)

        # raise because these have differing levels
        msg = "cannot align on a multi-index with out specifying the join levels"
        with pytest.raises(TypeError, match=msg):
            df.loc["bar"] *= 2

    def test_multiindex_setitem2(self):
        # from SO
        # https://stackoverflow.com/questions/24572040/pandas-access-the-level-of-multiindex-for-inplace-operation
        df_orig = DataFrame.from_dict(
            {
                "price": {
                    ("DE", "Coal", "Stock"): 2,
                    ("DE", "Gas", "Stock"): 4,
                    ("DE", "Elec", "Demand"): 1,
                    ("FR", "Gas", "Stock"): 5,
                    ("FR", "Solar", "SupIm"): 0,
                    ("FR", "Wind", "SupIm"): 0,
                }
            }
        )
        df_orig.index = MultiIndex.from_tuples(
            df_orig.index, names=["Sit", "Com", "Type"]
        )

        expected = df_orig.copy()
        expected.iloc[[0, 1, 3]] *= 2

        idx = pd.IndexSlice
        df = df_orig.copy()
        df.loc[idx[:, :, "Stock"], :] *= 2
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc[idx[:, :, "Stock"], "price"] *= 2
        tm.assert_frame_equal(df, expected)

    def test_multiindex_assignment(self):
        # GH3777 part 2

        # mixed dtype
        df = DataFrame(
            np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3),
            columns=list("abc"),
            index=[[4, 4, 8], [8, 10, 12]],
        )
        df["d"] = np.nan
        arr = np.array([0.0, 1.0])

        df.loc[4, "d"] = arr
        tm.assert_series_equal(df.loc[4, "d"], Series(arr, index=[8, 10], name="d"))

    def test_multiindex_assignment_single_dtype(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # GH3777 part 2b
        # single dtype
        arr = np.array([0.0, 1.0])

        df = DataFrame(
            np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3),
            columns=list("abc"),
            index=[[4, 4, 8], [8, 10, 12]],
            dtype=np.int64,
        )
        view = df["c"].iloc[:2].values

        # arr can be losslessly cast to int, so this setitem is inplace
        # INFO(CoW-warn) this does not warn because we directly took .values
        # above, so no reference to a pandas object is alive for `view`
        df.loc[4, "c"] = arr
        exp = Series(arr, index=[8, 10], name="c", dtype="int64")
        result = df.loc[4, "c"]
        tm.assert_series_equal(result, exp)

        # extra check for inplace-ness
        if not using_copy_on_write:
            tm.assert_numpy_array_equal(view, exp.values)

        # arr + 0.5 cannot be cast losslessly to int, so we upcast
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            df.loc[4, "c"] = arr + 0.5
        result = df.loc[4, "c"]
        exp = exp + 0.5
        tm.assert_series_equal(result, exp)

        # scalar ok
        with tm.assert_cow_warning(warn_copy_on_write):
            df.loc[4, "c"] = 10
        exp = Series(10, index=[8, 10], name="c", dtype="float64")
        tm.assert_series_equal(df.loc[4, "c"], exp)

        # invalid assignments
        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df.loc[4, "c"] = [0, 1, 2, 3]

        with pytest.raises(ValueError, match=msg):
            df.loc[4, "c"] = [0]

        # But with a length-1 listlike column indexer this behaves like
        #  `df.loc[4, "c"] = 0
        with tm.assert_cow_warning(warn_copy_on_write):
            df.loc[4, ["c"]] = [0]
        assert (df.loc[4, "c"] == 0).all()

    def test_groupby_example(self):
        # groupby example
        NUM_ROWS = 100
        NUM_COLS = 10
        col_names = ["A" + num for num in map(str, np.arange(NUM_COLS).tolist())]
        index_cols = col_names[:5]

        df = DataFrame(
            np.random.default_rng(2).integers(5, size=(NUM_ROWS, NUM_COLS)),
            dtype=np.int64,
            columns=col_names,
        )
        df = df.set_index(index_cols).sort_index()
        grp = df.groupby(level=index_cols[:4])
        df["new_col"] = np.nan

        # we are actually operating on a copy here
        # but in this case, that's ok
        for name, df2 in grp:
            new_vals = np.arange(df2.shape[0])
            df.loc[name, "new_col"] = new_vals

    def test_series_setitem(
        self, multiindex_year_month_day_dataframe_random_data, warn_copy_on_write
    ):
        ymd = multiindex_year_month_day_dataframe_random_data
        s = ymd["A"]

        with tm.assert_cow_warning(warn_copy_on_write):
            s[2000, 3] = np.nan
        assert isna(s.values[42:65]).all()
        assert notna(s.values[:42]).all()
        assert notna(s.values[65:]).all()

        with tm.assert_cow_warning(warn_copy_on_write):
            s[2000, 3, 10] = np.nan
        assert isna(s.iloc[49])

        with pytest.raises(KeyError, match="49"):
            # GH#33355 dont fall-back to positional when leading level is int
            s[49]

    def test_frame_getitem_setitem_boolean(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        df = frame.T.copy()
        values = df.values.copy()

        result = df[df > 0]
        expected = df.where(df > 0)
        tm.assert_frame_equal(result, expected)

        df[df > 0] = 5
        values[values > 0] = 5
        tm.assert_almost_equal(df.values, values)

        df[df == 5] = 0
        values[values == 5] = 0
        tm.assert_almost_equal(df.values, values)

        # a df that needs alignment first
        df[df[:-1] < 0] = 2
        np.putmask(values[:-1], values[:-1] < 0, 2)
        tm.assert_almost_equal(df.values, values)

        with pytest.raises(TypeError, match="boolean values only"):
            df[df * 0] = 2

    def test_frame_getitem_setitem_multislice(self):
        levels = [["t1", "t2"], ["a", "b", "c"]]
        codes = [[0, 0, 0, 1, 1], [0, 1, 2, 0, 1]]
        midx = MultiIndex(codes=codes, levels=levels, names=[None, "id"])
        df = DataFrame({"value": [1, 2, 3, 7, 8]}, index=midx)

        result = df.loc[:, "value"]
        tm.assert_series_equal(df["value"], result)

        result = df.loc[df.index[1:3], "value"]
        tm.assert_series_equal(df["value"][1:3], result)

        result = df.loc[:, :]
        tm.assert_frame_equal(df, result)

        result = df
        df.loc[:, "value"] = 10
        result["value"] = 10
        tm.assert_frame_equal(df, result)

        df.loc[:, :] = 10
        tm.assert_frame_equal(df, result)

    def test_frame_setitem_multi_column(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=[["a", "a", "b", "b"], [0, 1, 0, 1]],
        )

        cp = df.copy()
        cp["a"] = cp["b"]
        tm.assert_frame_equal(cp["a"], cp["b"])

        # set with ndarray
        cp = df.copy()
        cp["a"] = cp["b"].values
        tm.assert_frame_equal(cp["a"], cp["b"])

    def test_frame_setitem_multi_column2(self):
        # ---------------------------------------
        # GH#1803
        columns = MultiIndex.from_tuples([("A", "1"), ("A", "2"), ("B", "1")])
        df = DataFrame(index=[1, 3, 5], columns=columns)

        # Works, but adds a column instead of updating the two existing ones
        df["A"] = 0.0  # Doesn't work
        assert (df["A"].values == 0).all()

        # it broadcasts
        df["B", "1"] = [1, 2, 3]
        df["A"] = df["B", "1"]

        sliced_a1 = df["A", "1"]
        sliced_a2 = df["A", "2"]
        sliced_b1 = df["B", "1"]
        tm.assert_series_equal(sliced_a1, sliced_b1, check_names=False)
        tm.assert_series_equal(sliced_a2, sliced_b1, check_names=False)
        assert sliced_a1.name == ("A", "1")
        assert sliced_a2.name == ("A", "2")
        assert sliced_b1.name == ("B", "1")

    def test_loc_getitem_tuple_plus_columns(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        # GH #1013
        ymd = multiindex_year_month_day_dataframe_random_data
        df = ymd[:5]

        result = df.loc[(2000, 1, 6), ["A", "B", "C"]]
        expected = df.loc[2000, 1, 6][["A", "B", "C"]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    def test_loc_getitem_setitem_slice_integers(self, frame_or_series):
        index = MultiIndex(
            levels=[[0, 1, 2], [0, 2]], codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]]
        )

        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 4)),
            index=index,
            columns=["a", "b", "c", "d"],
        )
        obj = tm.get_obj(obj, frame_or_series)

        res = obj.loc[1:2]
        exp = obj.reindex(obj.index[2:])
        tm.assert_equal(res, exp)

        obj.loc[1:2] = 7
        assert (obj.loc[1:2] == 7).values.all()

    def test_setitem_change_dtype(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        dft = frame.T
        s = dft["foo", "two"]
        dft["foo", "two"] = s > s.median()
        tm.assert_series_equal(dft["foo", "two"], s > s.median())
        # assert isinstance(dft._data.blocks[1].items, MultiIndex)

        reindexed = dft.reindex(columns=[("foo", "two")])
        tm.assert_series_equal(reindexed["foo", "two"], s > s.median())

    def test_set_column_scalar_with_loc(
        self, multiindex_dataframe_random_data, using_copy_on_write, warn_copy_on_write
    ):
        frame = multiindex_dataframe_random_data
        subset = frame.index[[1, 4, 5]]

        frame.loc[subset] = 99
        assert (frame.loc[subset].values == 99).all()

        frame_original = frame.copy()
        col = frame["B"]
        with tm.assert_cow_warning(warn_copy_on_write):
            col[subset] = 97
        if using_copy_on_write:
            # chained setitem doesn't work with CoW
            tm.assert_frame_equal(frame, frame_original)
        else:
            assert (frame.loc[subset, "B"] == 97).all()

    def test_nonunique_assignment_1750(self):
        df = DataFrame(
            [[1, 1, "x", "X"], [1, 1, "y", "Y"], [1, 2, "z", "Z"]], columns=list("ABCD")
        )

        df = df.set_index(["A", "B"])
        mi = MultiIndex.from_tuples([(1, 1)])

        df.loc[mi, "C"] = "_"

        assert (df.xs((1, 1))["C"] == "_").all()

    def test_astype_assignment_with_dups(self):
        # GH 4686
        # assignment with dups that has a dtype change
        cols = MultiIndex.from_tuples([("A", "1"), ("B", "1"), ("A", "2")])
        df = DataFrame(np.arange(3).reshape((1, 3)), columns=cols, dtype=object)
        index = df.index.copy()

        df["A"] = df["A"].astype(np.float64)
        tm.assert_index_equal(df.index, index)

    def test_setitem_nonmonotonic(self):
        # https://github.com/pandas-dev/pandas/issues/31449
        index = MultiIndex.from_tuples(
            [("a", "c"), ("b", "x"), ("a", "d")], names=["l1", "l2"]
        )
        df = DataFrame(data=[0, 1, 2], index=index, columns=["e"])
        df.loc["a", "e"] = np.arange(99, 101, dtype="int64")
        expected = DataFrame({"e": [99, 1, 100]}, index=index)
        tm.assert_frame_equal(df, expected)


class TestSetitemWithExpansionMultiIndex:
    def test_setitem_new_column_mixed_depth(self):
        arrays = [
            ["a", "top", "top", "routine1", "routine1", "routine2"],
            ["", "OD", "OD", "result1", "result2", "result1"],
            ["", "wx", "wy", "", "", ""],
        ]

        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

        result = df.copy()
        expected = df.copy()
        result["b"] = [1, 2, 3, 4]
        expected["b", "", ""] = [1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    def test_setitem_new_column_all_na(self):
        # GH#1534
        mix = MultiIndex.from_tuples([("1a", "2a"), ("1a", "2b"), ("1a", "2c")])
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mix)
        s = Series({(1, 1): 1, (1, 2): 2})
        df["new"] = s
        assert df["new"].isna().all()

    def test_setitem_enlargement_keep_index_names(self):
        # GH#53053
        mi = MultiIndex.from_tuples([(1, 2, 3)], names=["i1", "i2", "i3"])
        df = DataFrame(data=[[10, 20, 30]], index=mi, columns=["A", "B", "C"])
        df.loc[(0, 0, 0)] = df.loc[(1, 2, 3)]
        mi_expected = MultiIndex.from_tuples(
            [(1, 2, 3), (0, 0, 0)], names=["i1", "i2", "i3"]
        )
        expected = DataFrame(
            data=[[10, 20, 30], [10, 20, 30]],
            index=mi_expected,
            columns=["A", "B", "C"],
        )
        tm.assert_frame_equal(df, expected)


@td.skip_array_manager_invalid_test  # df["foo"] select multiple columns -> .values
# is not a view
def test_frame_setitem_view_direct(
    multiindex_dataframe_random_data, using_copy_on_write
):
    # this works because we are modifying the underlying array
    # really a no-no
    df = multiindex_dataframe_random_data.T
    if using_copy_on_write:
        with pytest.raises(ValueError, match="read-only"):
            df["foo"].values[:] = 0
        assert (df["foo"].values != 0).all()
    else:
        df["foo"].values[:] = 0
        assert (df["foo"].values == 0).all()


def test_frame_setitem_copy_raises(
    multiindex_dataframe_random_data, using_copy_on_write, warn_copy_on_write
):
    # will raise/warn as its chained assignment
    df = multiindex_dataframe_random_data.T
    if using_copy_on_write or warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df["foo"]["one"] = 2
    else:
        msg = "A value is trying to be set on a copy of a slice from a DataFrame"
        with pytest.raises(SettingWithCopyError, match=msg):
            with tm.raises_chained_assignment_error():
                df["foo"]["one"] = 2


def test_frame_setitem_copy_no_write(
    multiindex_dataframe_random_data, using_copy_on_write, warn_copy_on_write
):
    frame = multiindex_dataframe_random_data.T
    expected = frame
    df = frame.copy()
    if using_copy_on_write or warn_copy_on_write:
        with tm.raises_chained_assignment_error():
            df["foo"]["one"] = 2
    else:
        msg = "A value is trying to be set on a copy of a slice from a DataFrame"
        with pytest.raises(SettingWithCopyError, match=msg):
            with tm.raises_chained_assignment_error():
                df["foo"]["one"] = 2

    result = df
    tm.assert_frame_equal(result, expected)


def test_frame_setitem_partial_multiindex():
    # GH 54875
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 4, 5],
            "c": 6,
            "d": 7,
        }
    ).set_index(["a", "b", "c"])
    ser = Series(8, index=df.index.droplevel("c"))
    result = df.copy()
    result["d"] = ser
    expected = df.copy()
    expected["d"] = 8
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\tests\test_codegen_octave.py ===
from io import StringIO

from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy


x, y, z = symbols('x,y,z')


def test_empty_m_code():
    code_gen = OctaveCodeGen()
    output = StringIO()
    code_gen.dump_m([], output, "file", header=False, empty=False)
    source = output.getvalue()
    assert source == ""


def test_m_simple_code():
    name_expr = ("test", (x + y)*z)
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    assert result[0] == "test.m"
    source = result[1]
    expected = (
        "function out1 = test(x, y, z)\n"
        "  out1 = z.*(x + y);\n"
        "end\n"
    )
    assert source == expected


def test_m_simple_code_with_header():
    name_expr = ("test", (x + y)*z)
    result, = codegen(name_expr, "Octave", header=True, empty=False)
    assert result[0] == "test.m"
    source = result[1]
    expected = (
        "function out1 = test(x, y, z)\n"
        "  %TEST  Autogenerated by SymPy\n"
        "  %   Code generated with SymPy " + sympy.__version__ + "\n"
        "  %\n"
        "  %   See http://www.sympy.org/ for more information.\n"
        "  %\n"
        "  %   This file is part of 'project'\n"
        "  out1 = z.*(x + y);\n"
        "end\n"
    )
    assert source == expected


def test_m_simple_code_nameout():
    expr = Equality(z, (x + y))
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function z = test(x, y)\n"
        "  z = x + y;\n"
        "end\n"
    )
    assert source == expected


def test_m_numbersymbol():
    name_expr = ("test", pi**Catalan)
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function out1 = test()\n"
        "  out1 = pi^%s;\n"
        "end\n"
    ) % Catalan.evalf(17)
    assert source == expected


@XFAIL
def test_m_numbersymbol_no_inline():
    # FIXME: how to pass inline=False to the OctaveCodePrinter?
    name_expr = ("test", [pi**Catalan, EulerGamma])
    result, = codegen(name_expr, "Octave", header=False,
                      empty=False, inline=False)
    source = result[1]
    expected = (
        "function [out1, out2] = test()\n"
        "  Catalan = 0.915965594177219;  % constant\n"
        "  EulerGamma = 0.5772156649015329;  % constant\n"
        "  out1 = pi^Catalan;\n"
        "  out2 = EulerGamma;\n"
        "end\n"
    )
    assert source == expected


def test_m_code_argument_order():
    expr = x + y
    routine = make_routine("test", expr, argument_sequence=[z, x, y], language="octave")
    code_gen = OctaveCodeGen()
    output = StringIO()
    code_gen.dump_m([routine], output, "test", header=False, empty=False)
    source = output.getvalue()
    expected = (
        "function out1 = test(z, x, y)\n"
        "  out1 = x + y;\n"
        "end\n"
    )
    assert source == expected


def test_multiple_results_m():
    # Here the output order is the input order
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    name_expr = ("test", [expr1, expr2])
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function [out1, out2] = test(x, y, z)\n"
        "  out1 = z.*(x + y);\n"
        "  out2 = z.*(x - y);\n"
        "end\n"
    )
    assert source == expected


def test_results_named_unordered():
    # Here output order is based on name_expr
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function [C, A, B] = test(x, y, z)\n"
        "  C = z.*(x + y);\n"
        "  A = z.*(x - y);\n"
        "  B = 2*x;\n"
        "end\n"
    )
    assert source == expected


def test_results_named_ordered():
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result = codegen(name_expr, "Octave", header=False, empty=False,
                     argument_sequence=(x, z, y))
    assert result[0][0] == "test.m"
    source = result[0][1]
    expected = (
        "function [C, A, B] = test(x, z, y)\n"
        "  C = z.*(x + y);\n"
        "  A = z.*(x - y);\n"
        "  B = 2*x;\n"
        "end\n"
    )
    assert source == expected


def test_complicated_m_codegen():
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    name_expr = ("testlong",
            [ ((sin(x) + cos(y) + tan(z))**3).expand(),
            cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))
    ])
    result = codegen(name_expr, "Octave", header=False, empty=False)
    assert result[0][0] == "testlong.m"
    source = result[0][1]
    expected = (
        "function [out1, out2] = testlong(x, y, z)\n"
        "  out1 = sin(x).^3 + 3*sin(x).^2.*cos(y) + 3*sin(x).^2.*tan(z)"
        " + 3*sin(x).*cos(y).^2 + 6*sin(x).*cos(y).*tan(z) + 3*sin(x).*tan(z).^2"
        " + cos(y).^3 + 3*cos(y).^2.*tan(z) + 3*cos(y).*tan(z).^2 + tan(z).^3;\n"
        "  out2 = cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))));\n"
        "end\n"
    )
    assert source == expected


def test_m_output_arg_mixed_unordered():
    # named outputs are alphabetical, unnamed output appear in the given order
    from sympy.functions.elementary.trigonometric import (cos, sin)
    a = symbols("a")
    name_expr = ("foo", [cos(2*x), Equality(y, sin(x)), cos(x), Equality(a, sin(2*x))])
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    assert result[0] == "foo.m"
    source = result[1]
    expected = (
        'function [out1, y, out3, a] = foo(x)\n'
        '  out1 = cos(2*x);\n'
        '  y = sin(x);\n'
        '  out3 = cos(x);\n'
        '  a = sin(2*x);\n'
        'end\n'
    )
    assert source == expected


def test_m_piecewise_():
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True), evaluate=False)
    name_expr = ("pwtest", pw)
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function out1 = pwtest(x)\n"
        "  out1 = ((x < -1).*(0) + (~(x < -1)).*( ...\n"
        "  (x <= 1).*(x.^2) + (~(x <= 1)).*( ...\n"
        "  (x > 1).*(2 - x) + (~(x > 1)).*(1))));\n"
        "end\n"
    )
    assert source == expected


@XFAIL
def test_m_piecewise_no_inline():
    # FIXME: how to pass inline=False to the OctaveCodePrinter?
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True))
    name_expr = ("pwtest", pw)
    result, = codegen(name_expr, "Octave", header=False, empty=False,
                      inline=False)
    source = result[1]
    expected = (
        "function out1 = pwtest(x)\n"
        "  if (x < -1)\n"
        "    out1 = 0;\n"
        "  elseif (x <= 1)\n"
        "    out1 = x.^2;\n"
        "  elseif (x > 1)\n"
        "    out1 = -x + 2;\n"
        "  else\n"
        "    out1 = 1;\n"
        "  end\n"
        "end\n"
    )
    assert source == expected


def test_m_multifcns_per_file():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result = codegen(name_expr, "Octave", header=False, empty=False)
    assert result[0][0] == "foo.m"
    source = result[0][1]
    expected = (
        "function [out1, out2] = foo(x, y)\n"
        "  out1 = 2*x;\n"
        "  out2 = 3*y;\n"
        "end\n"
        "function [out1, out2] = bar(y)\n"
        "  out1 = y.^2;\n"
        "  out2 = 4*y;\n"
        "end\n"
    )
    assert source == expected


def test_m_multifcns_per_file_w_header():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result = codegen(name_expr, "Octave", header=True, empty=False)
    assert result[0][0] == "foo.m"
    source = result[0][1]
    expected = (
        "function [out1, out2] = foo(x, y)\n"
        "  %FOO  Autogenerated by SymPy\n"
        "  %   Code generated with SymPy " + sympy.__version__ + "\n"
        "  %\n"
        "  %   See http://www.sympy.org/ for more information.\n"
        "  %\n"
        "  %   This file is part of 'project'\n"
        "  out1 = 2*x;\n"
        "  out2 = 3*y;\n"
        "end\n"
        "function [out1, out2] = bar(y)\n"
        "  out1 = y.^2;\n"
        "  out2 = 4*y;\n"
        "end\n"
    )
    assert source == expected


def test_m_filename_match_first_fcn():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    raises(ValueError, lambda: codegen(name_expr,
                        "Octave", prefix="bar", header=False, empty=False))


def test_m_matrix_named():
    e2 = Matrix([[x, 2*y, pi*z]])
    name_expr = ("test", Equality(MatrixSymbol('myout1', 1, 3), e2))
    result = codegen(name_expr, "Octave", header=False, empty=False)
    assert result[0][0] == "test.m"
    source = result[0][1]
    expected = (
        "function myout1 = test(x, y, z)\n"
        "  myout1 = [x 2*y pi*z];\n"
        "end\n"
    )
    assert source == expected


def test_m_matrix_named_matsym():
    myout1 = MatrixSymbol('myout1', 1, 3)
    e2 = Matrix([[x, 2*y, pi*z]])
    name_expr = ("test", Equality(myout1, e2, evaluate=False))
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function myout1 = test(x, y, z)\n"
        "  myout1 = [x 2*y pi*z];\n"
        "end\n"
    )
    assert source == expected


def test_m_matrix_output_autoname():
    expr = Matrix([[x, x+y, 3]])
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function out1 = test(x, y)\n"
        "  out1 = [x x + y 3];\n"
        "end\n"
    )
    assert source == expected


def test_m_matrix_output_autoname_2():
    e1 = (x + y)
    e2 = Matrix([[2*x, 2*y, 2*z]])
    e3 = Matrix([[x], [y], [z]])
    e4 = Matrix([[x, y], [z, 16]])
    name_expr = ("test", (e1, e2, e3, e4))
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function [out1, out2, out3, out4] = test(x, y, z)\n"
        "  out1 = x + y;\n"
        "  out2 = [2*x 2*y 2*z];\n"
        "  out3 = [x; y; z];\n"
        "  out4 = [x y; z 16];\n"
        "end\n"
    )
    assert source == expected


def test_m_results_matrix_named_ordered():
    B, C = symbols('B,C')
    A = MatrixSymbol('A', 1, 3)
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, Matrix([[1, 2, x]]))
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result, = codegen(name_expr, "Octave", header=False, empty=False,
                     argument_sequence=(x, z, y))
    source = result[1]
    expected = (
        "function [C, A, B] = test(x, z, y)\n"
        "  C = z.*(x + y);\n"
        "  A = [1 2 x];\n"
        "  B = 2*x;\n"
        "end\n"
    )
    assert source == expected


def test_m_matrixsymbol_slice():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    D = MatrixSymbol('D', 2, 1)
    name_expr = ("test", [Equality(B, A[0, :]),
                          Equality(C, A[1, :]),
                          Equality(D, A[:, 2])])
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function [B, C, D] = test(A)\n"
        "  B = A(1, :);\n"
        "  C = A(2, :);\n"
        "  D = A(:, 3);\n"
        "end\n"
    )
    assert source == expected


def test_m_matrixsymbol_slice2():
    A = MatrixSymbol('A', 3, 4)
    B = MatrixSymbol('B', 2, 2)
    C = MatrixSymbol('C', 2, 2)
    name_expr = ("test", [Equality(B, A[0:2, 0:2]),
                          Equality(C, A[0:2, 1:3])])
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function [B, C] = test(A)\n"
        "  B = A(1:2, 1:2);\n"
        "  C = A(1:2, 2:3);\n"
        "end\n"
    )
    assert source == expected


def test_m_matrixsymbol_slice3():
    A = MatrixSymbol('A', 8, 7)
    B = MatrixSymbol('B', 2, 2)
    C = MatrixSymbol('C', 4, 2)
    name_expr = ("test", [Equality(B, A[6:, 1::3]),
                          Equality(C, A[::2, ::3])])
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function [B, C] = test(A)\n"
        "  B = A(7:end, 2:3:end);\n"
        "  C = A(1:2:end, 1:3:end);\n"
        "end\n"
    )
    assert source == expected


def test_m_matrixsymbol_slice_autoname():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 1, 3)
    name_expr = ("test", [Equality(B, A[0,:]), A[1,:], A[:,0], A[:,1]])
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function [B, out2, out3, out4] = test(A)\n"
        "  B = A(1, :);\n"
        "  out2 = A(2, :);\n"
        "  out3 = A(:, 1);\n"
        "  out4 = A(:, 2);\n"
        "end\n"
    )
    assert source == expected


def test_m_loops():
    # Note: an Octave programmer would probably vectorize this across one or
    # more dimensions.  Also, size(A) would be used rather than passing in m
    # and n.  Perhaps users would expect us to vectorize automatically here?
    # Or is it possible to represent such things using IndexedBase?
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    result, = codegen(('mat_vec_mult', Eq(y[i], A[i, j]*x[j])), "Octave",
                      header=False, empty=False)
    source = result[1]
    expected = (
        'function y = mat_vec_mult(A, m, n, x)\n'
        '  for i = 1:m\n'
        '    y(i) = 0;\n'
        '  end\n'
        '  for i = 1:m\n'
        '    for j = 1:n\n'
        '      y(i) = %(rhs)s + y(i);\n'
        '    end\n'
        '  end\n'
        'end\n'
    )
    assert (source == expected % {'rhs': 'A(%s, %s).*x(j)' % (i, j)} or
            source == expected % {'rhs': 'x(j).*A(%s, %s)' % (i, j)})


def test_m_tensor_loops_multiple_contractions():
    # see comments in previous test about vectorizing
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m, o, p = symbols('n m o p', integer=True)
    A = IndexedBase('A')
    B = IndexedBase('B')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)
    result, = codegen(('tensorthing', Eq(y[i], B[j, k, l]*A[i, j, k, l])),
                      "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        'function y = tensorthing(A, B, m, n, o, p)\n'
        '  for i = 1:m\n'
        '    y(i) = 0;\n'
        '  end\n'
        '  for i = 1:m\n'
        '    for j = 1:n\n'
        '      for k = 1:o\n'
        '        for l = 1:p\n'
        '          y(i) = A(i, j, k, l).*B(j, k, l) + y(i);\n'
        '        end\n'
        '      end\n'
        '    end\n'
        '  end\n'
        'end\n'
    )
    assert source == expected


def test_m_InOutArgument():
    expr = Equality(x, x**2)
    name_expr = ("mysqr", expr)
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function x = mysqr(x)\n"
        "  x = x.^2;\n"
        "end\n"
    )
    assert source == expected


def test_m_InOutArgument_order():
    # can specify the order as (x, y)
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Octave", header=False,
                      empty=False, argument_sequence=(x,y))
    source = result[1]
    expected = (
        "function x = test(x, y)\n"
        "  x = x.^2 + y;\n"
        "end\n"
    )
    assert source == expected
    # make sure it gives (x, y) not (y, x)
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function x = test(x, y)\n"
        "  x = x.^2 + y;\n"
        "end\n"
    )
    assert source == expected


def test_m_not_supported():
    f = Function('f')
    name_expr = ("test", [f(x).diff(x), S.ComplexInfinity])
    result, = codegen(name_expr, "Octave", header=False, empty=False)
    source = result[1]
    expected = (
        "function [out1, out2] = test(x)\n"
        "  % unsupported: Derivative(f(x), x)\n"
        "  % unsupported: zoo\n"
        "  out1 = Derivative(f(x), x);\n"
        "  out2 = zoo;\n"
        "end\n"
    )
    assert source == expected


def test_global_vars_octave():
    x, y, z, t = symbols("x y z t")
    result = codegen(('f', x*y), "Octave", header=False, empty=False,
                     global_vars=(y,))
    source = result[0][1]
    expected = (
        "function out1 = f(x)\n"
        "  global y\n"
        "  out1 = x.*y;\n"
        "end\n"
        )
    assert source == expected

    result = codegen(('f', x*y+z), "Octave", header=False, empty=False,
                     argument_sequence=(x, y), global_vars=(z, t))
    source = result[0][1]
    expected = (
        "function out1 = f(x, y)\n"
        "  global t z\n"
        "  out1 = x.*y + z;\n"
        "end\n"
    )
    assert source == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\decimal\test_decimal.py ===
from __future__ import annotations

import decimal
import operator

import numpy as np
import pytest

from pandas.compat.numpy import np_version_gt2

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
    make_data,
    to_decimal,
)


@pytest.fixture
def dtype():
    return DecimalDtype()


@pytest.fixture
def data():
    return DecimalArray(make_data())


@pytest.fixture
def data_for_twos():
    return DecimalArray([decimal.Decimal(2) for _ in range(100)])


@pytest.fixture
def data_missing():
    return DecimalArray([decimal.Decimal("NaN"), decimal.Decimal(1)])


@pytest.fixture
def data_for_sorting():
    return DecimalArray(
        [decimal.Decimal("1"), decimal.Decimal("2"), decimal.Decimal("0")]
    )


@pytest.fixture
def data_missing_for_sorting():
    return DecimalArray(
        [decimal.Decimal("1"), decimal.Decimal("NaN"), decimal.Decimal("0")]
    )


@pytest.fixture
def na_cmp():
    return lambda x, y: x.is_nan() and y.is_nan()


@pytest.fixture
def data_for_grouping():
    b = decimal.Decimal("1.0")
    a = decimal.Decimal("0.0")
    c = decimal.Decimal("2.0")
    na = decimal.Decimal("NaN")
    return DecimalArray([b, b, na, na, a, a, b, c])


class TestDecimalArray(base.ExtensionTests):
    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | tuple[type[Exception], ...] | None:
        return None

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        if op_name == "count":
            return super().check_reduce(ser, op_name, skipna)
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(np.asarray(ser), op_name)()
            tm.assert_almost_equal(result, expected)

    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna, request):
        if all_numeric_reductions in ["kurt", "skew", "sem", "median"]:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        op_name = all_numeric_reductions
        if op_name in ["skew", "median"]:
            mark = pytest.mark.xfail(raises=NotImplementedError)
            request.applymarker(mark)

        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0.5)

    def test_compare_array(self, data, comparison_op):
        ser = pd.Series(data)

        alter = np.random.default_rng(2).choice([-1, 0, 1], len(data))
        # Randomly double, halve or keep same value
        other = pd.Series(data) * [decimal.Decimal(pow(2.0, i)) for i in alter]
        self._compare_other(ser, data, comparison_op, other)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        op_name = all_arithmetic_operators
        ser = pd.Series(data)

        context = decimal.getcontext()
        divbyzerotrap = context.traps[decimal.DivisionByZero]
        invalidoptrap = context.traps[decimal.InvalidOperation]
        context.traps[decimal.DivisionByZero] = 0
        context.traps[decimal.InvalidOperation] = 0

        # Decimal supports ops with int, but not float
        other = pd.Series([int(d * 100) for d in data])
        self.check_opname(ser, op_name, other)

        if "mod" not in op_name:
            self.check_opname(ser, op_name, ser * 2)

        self.check_opname(ser, op_name, 0)
        self.check_opname(ser, op_name, 5)
        context.traps[decimal.DivisionByZero] = divbyzerotrap
        context.traps[decimal.InvalidOperation] = invalidoptrap

    def test_fillna_frame(self, data_missing):
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            super().test_fillna_frame(data_missing)

    def test_fillna_limit_pad(self, data_missing):
        msg = "ExtensionArray.fillna 'method' keyword is deprecated"
        with tm.assert_produces_warning(
            DeprecationWarning,
            match=msg,
            check_stacklevel=False,
            raise_on_extra_warnings=False,
        ):
            super().test_fillna_limit_pad(data_missing)

        msg = "The 'method' keyword in DecimalArray.fillna is deprecated"
        with tm.assert_produces_warning(
            FutureWarning,
            match=msg,
            check_stacklevel=False,
            raise_on_extra_warnings=False,
        ):
            super().test_fillna_limit_pad(data_missing)

    @pytest.mark.parametrize(
        "limit_area, input_ilocs, expected_ilocs",
        [
            ("outside", [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),
            ("outside", [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),
            ("outside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]),
            ("outside", [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]),
            ("inside", [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]),
            ("inside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),
            ("inside", [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]),
        ],
    )
    def test_ffill_limit_area(
        self, data_missing, limit_area, input_ilocs, expected_ilocs
    ):
        # GH#56616
        msg = "ExtensionArray.fillna 'method' keyword is deprecated"
        with tm.assert_produces_warning(
            DeprecationWarning,
            match=msg,
            check_stacklevel=False,
            raise_on_extra_warnings=False,
        ):
            msg = "DecimalArray does not implement limit_area"
            with pytest.raises(NotImplementedError, match=msg):
                super().test_ffill_limit_area(
                    data_missing, limit_area, input_ilocs, expected_ilocs
                )

    def test_fillna_limit_backfill(self, data_missing):
        msg = "Series.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(
            FutureWarning,
            match=msg,
            check_stacklevel=False,
            raise_on_extra_warnings=False,
        ):
            super().test_fillna_limit_backfill(data_missing)

        msg = "ExtensionArray.fillna 'method' keyword is deprecated"
        with tm.assert_produces_warning(
            DeprecationWarning,
            match=msg,
            check_stacklevel=False,
            raise_on_extra_warnings=False,
        ):
            super().test_fillna_limit_backfill(data_missing)

        msg = "The 'method' keyword in DecimalArray.fillna is deprecated"
        with tm.assert_produces_warning(
            FutureWarning,
            match=msg,
            check_stacklevel=False,
            raise_on_extra_warnings=False,
        ):
            super().test_fillna_limit_backfill(data_missing)

    def test_fillna_no_op_returns_copy(self, data):
        msg = "|".join(
            [
                "ExtensionArray.fillna 'method' keyword is deprecated",
                "The 'method' keyword in DecimalArray.fillna is deprecated",
            ]
        )
        with tm.assert_produces_warning(
            (FutureWarning, DeprecationWarning), match=msg, check_stacklevel=False
        ):
            super().test_fillna_no_op_returns_copy(data)

    def test_fillna_series(self, data_missing):
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            super().test_fillna_series(data_missing)

    def test_fillna_series_method(self, data_missing, fillna_method):
        msg = "|".join(
            [
                "ExtensionArray.fillna 'method' keyword is deprecated",
                "The 'method' keyword in DecimalArray.fillna is deprecated",
            ]
        )
        with tm.assert_produces_warning(
            (FutureWarning, DeprecationWarning), match=msg, check_stacklevel=False
        ):
            super().test_fillna_series_method(data_missing, fillna_method)

    def test_fillna_copy_frame(self, data_missing, using_copy_on_write):
        warn = DeprecationWarning if not using_copy_on_write else None
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
            super().test_fillna_copy_frame(data_missing)

    def test_fillna_copy_series(self, data_missing, using_copy_on_write):
        warn = DeprecationWarning if not using_copy_on_write else None
        msg = "ExtensionArray.fillna added a 'copy' keyword"
        with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
            super().test_fillna_copy_series(data_missing)

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna, request):
        all_data = all_data[:10]
        if dropna:
            other = np.array(all_data[~all_data.isna()])
        else:
            other = all_data

        vcs = pd.Series(all_data).value_counts(dropna=dropna)
        vcs_ex = pd.Series(other).value_counts(dropna=dropna)

        with decimal.localcontext() as ctx:
            # avoid raising when comparing Decimal("NAN") < Decimal(2)
            ctx.traps[decimal.InvalidOperation] = False

            result = vcs.sort_index()
            expected = vcs_ex.sort_index()

        tm.assert_series_equal(result, expected)

    def test_series_repr(self, data):
        # Overriding this base test to explicitly test that
        # the custom _formatter is used
        ser = pd.Series(data)
        assert data.dtype.name in repr(ser)
        assert "Decimal: " in repr(ser)

    @pytest.mark.xfail(reason="Inconsistent array-vs-scalar behavior")
    @pytest.mark.parametrize("ufunc", [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data, ufunc):
        super().test_unary_ufunc_dunder_equivalence(data, ufunc)

    def test_array_interface_copy(self, data):
        result_copy1 = np.array(data, copy=True)
        result_copy2 = np.array(data, copy=True)
        assert not np.may_share_memory(result_copy1, result_copy2)
        if not np_version_gt2:
            # copy=False semantics are only supported in NumPy>=2.
            return

        try:
            result_nocopy1 = np.array(data, copy=False)
        except ValueError:
            # An error is always acceptable for `copy=False`
            return

        result_nocopy2 = np.array(data, copy=False)
        # If copy=False was given and did not raise, these must share the same data
        assert np.may_share_memory(result_nocopy1, result_nocopy2)


def test_take_na_value_other_decimal():
    arr = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    result = arr.take([0, -1], allow_fill=True, fill_value=decimal.Decimal("-1.0"))
    expected = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("-1.0")])
    tm.assert_extension_array_equal(result, expected)


def test_series_constructor_coerce_data_to_extension_dtype():
    dtype = DecimalDtype()
    ser = pd.Series([0, 1, 2], dtype=dtype)

    arr = DecimalArray(
        [decimal.Decimal(0), decimal.Decimal(1), decimal.Decimal(2)],
        dtype=dtype,
    )
    exp = pd.Series(arr)
    tm.assert_series_equal(ser, exp)


def test_series_constructor_with_dtype():
    arr = DecimalArray([decimal.Decimal("10.0")])
    result = pd.Series(arr, dtype=DecimalDtype())
    expected = pd.Series(arr)
    tm.assert_series_equal(result, expected)

    result = pd.Series(arr, dtype="int64")
    expected = pd.Series([10])
    tm.assert_series_equal(result, expected)


def test_dataframe_constructor_with_dtype():
    arr = DecimalArray([decimal.Decimal("10.0")])

    result = pd.DataFrame({"A": arr}, dtype=DecimalDtype())
    expected = pd.DataFrame({"A": arr})
    tm.assert_frame_equal(result, expected)

    arr = DecimalArray([decimal.Decimal("10.0")])
    result = pd.DataFrame({"A": arr}, dtype="int64")
    expected = pd.DataFrame({"A": [10]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("frame", [True, False])
def test_astype_dispatches(frame):
    # This is a dtype-specific test that ensures Series[decimal].astype
    # gets all the way through to ExtensionArray.astype
    # Designing a reliable smoke test that works for arbitrary data types
    # is difficult.
    data = pd.Series(DecimalArray([decimal.Decimal(2)]), name="a")
    ctx = decimal.Context()
    ctx.prec = 5

    if frame:
        data = data.to_frame()

    result = data.astype(DecimalDtype(ctx))

    if frame:
        result = result["a"]

    assert result.dtype.context.prec == ctx.prec


class DecimalArrayWithoutFromSequence(DecimalArray):
    """Helper class for testing error handling in _from_sequence."""

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        raise KeyError("For the test")


class DecimalArrayWithoutCoercion(DecimalArrayWithoutFromSequence):
    @classmethod
    def _create_arithmetic_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False)


DecimalArrayWithoutCoercion._add_arithmetic_ops()


def test_combine_from_sequence_raises(monkeypatch):
    # https://github.com/pandas-dev/pandas/issues/22850
    cls = DecimalArrayWithoutFromSequence

    @classmethod
    def construct_array_type(cls):
        return DecimalArrayWithoutFromSequence

    monkeypatch.setattr(DecimalDtype, "construct_array_type", construct_array_type)

    arr = cls([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    ser = pd.Series(arr)
    result = ser.combine(ser, operator.add)

    # note: object dtype
    expected = pd.Series(
        [decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object"
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "class_", [DecimalArrayWithoutFromSequence, DecimalArrayWithoutCoercion]
)
def test_scalar_ops_from_sequence_raises(class_):
    # op(EA, EA) should return an EA, or an ndarray if it's not possible
    # to return an EA with the return values.
    arr = class_([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    result = arr + arr
    expected = np.array(
        [decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object"
    )
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "reverse, expected_div, expected_mod",
    [(False, [0, 1, 1, 2], [1, 0, 1, 0]), (True, [2, 1, 0, 0], [0, 0, 2, 2])],
)
def test_divmod_array(reverse, expected_div, expected_mod):
    # https://github.com/pandas-dev/pandas/issues/22930
    arr = to_decimal([1, 2, 3, 4])
    if reverse:
        div, mod = divmod(2, arr)
    else:
        div, mod = divmod(arr, 2)
    expected_div = to_decimal(expected_div)
    expected_mod = to_decimal(expected_mod)

    tm.assert_extension_array_equal(div, expected_div)
    tm.assert_extension_array_equal(mod, expected_mod)


def test_ufunc_fallback(data):
    a = data[:5]
    s = pd.Series(a, index=range(3, 8))
    result = np.abs(s)
    expected = pd.Series(np.abs(a), index=range(3, 8))
    tm.assert_series_equal(result, expected)


def test_array_ufunc():
    a = to_decimal([1, 2, 3])
    result = np.exp(a)
    expected = to_decimal(np.exp(a._data))
    tm.assert_extension_array_equal(result, expected)


def test_array_ufunc_series():
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)
    result = np.exp(s)
    expected = pd.Series(to_decimal(np.exp(a._data)))
    tm.assert_series_equal(result, expected)


def test_array_ufunc_series_scalar_other():
    # check _HANDLED_TYPES
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)
    result = np.add(s, decimal.Decimal(1))
    expected = pd.Series(np.add(a, decimal.Decimal(1)))
    tm.assert_series_equal(result, expected)


def test_array_ufunc_series_defer():
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)

    expected = pd.Series(to_decimal([2, 4, 6]))
    r1 = np.add(s, a)
    r2 = np.add(a, s)

    tm.assert_series_equal(r1, expected)
    tm.assert_series_equal(r2, expected)


def test_groupby_agg():
    # Ensure that the result of agg is inferred to be decimal dtype
    # https://github.com/pandas-dev/pandas/issues/29141

    data = make_data()[:5]
    df = pd.DataFrame(
        {"id1": [0, 0, 0, 1, 1], "id2": [0, 1, 0, 1, 1], "decimals": DecimalArray(data)}
    )

    # single key, selected column
    expected = pd.Series(to_decimal([data[0], data[3]]))
    result = df.groupby("id1")["decimals"].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby(df["id1"]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)

    # multiple keys, selected column
    expected = pd.Series(
        to_decimal([data[0], data[1], data[3]]),
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 1)]),
    )
    result = df.groupby(["id1", "id2"])["decimals"].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby([df["id1"], df["id2"]]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)

    # multiple columns
    expected = pd.DataFrame({"id2": [0, 1], "decimals": to_decimal([data[0], data[3]])})
    result = df.groupby("id1").agg(lambda x: x.iloc[0])
    tm.assert_frame_equal(result, expected, check_names=False)


def test_groupby_agg_ea_method(monkeypatch):
    # Ensure that the result of agg is inferred to be decimal dtype
    # https://github.com/pandas-dev/pandas/issues/29141

    def DecimalArray__my_sum(self):
        return np.sum(np.array(self))

    monkeypatch.setattr(DecimalArray, "my_sum", DecimalArray__my_sum, raising=False)

    data = make_data()[:5]
    df = pd.DataFrame({"id": [0, 0, 0, 1, 1], "decimals": DecimalArray(data)})
    expected = pd.Series(to_decimal([data[0] + data[1] + data[2], data[3] + data[4]]))

    result = df.groupby("id")["decimals"].agg(lambda x: x.values.my_sum())
    tm.assert_series_equal(result, expected, check_names=False)
    s = pd.Series(DecimalArray(data))
    grouper = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    result = s.groupby(grouper).agg(lambda x: x.values.my_sum())
    tm.assert_series_equal(result, expected, check_names=False)


def test_indexing_no_materialize(monkeypatch):
    # See https://github.com/pandas-dev/pandas/issues/29708
    # Ensure that indexing operations do not materialize (convert to a numpy
    # array) the ExtensionArray unnecessary

    def DecimalArray__array__(self, dtype=None):
        raise Exception("tried to convert a DecimalArray to a numpy array")

    monkeypatch.setattr(DecimalArray, "__array__", DecimalArray__array__, raising=False)

    data = make_data()
    s = pd.Series(DecimalArray(data))
    df = pd.DataFrame({"a": s, "b": range(len(s))})

    # ensure the following operations do not raise an error
    s[s > 0.5]
    df[s > 0.5]
    s.at[0]
    df.at[0, "a"]


def test_to_numpy_keyword():
    # test the extra keyword
    values = [decimal.Decimal("1.1111"), decimal.Decimal("2.2222")]
    expected = np.array(
        [decimal.Decimal("1.11"), decimal.Decimal("2.22")], dtype="object"
    )
    a = pd.array(values, dtype="decimal")
    result = a.to_numpy(decimals=2)
    tm.assert_numpy_array_equal(result, expected)

    result = pd.Series(a).to_numpy(decimals=2)
    tm.assert_numpy_array_equal(result, expected)


def test_array_copy_on_write(using_copy_on_write):
    df = pd.DataFrame({"a": [decimal.Decimal(2), decimal.Decimal(3)]}, dtype="object")
    df2 = df.astype(DecimalDtype())
    df.iloc[0, 0] = 0
    if using_copy_on_write:
        expected = pd.DataFrame(
            {"a": [decimal.Decimal(2), decimal.Decimal(3)]}, dtype=DecimalDtype()
        )
        tm.assert_equal(df2.values, expected.values)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_round_trip.py ===
import datetime
import re

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    _testing as tm,
    bdate_range,
    date_range,
    read_hdf,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)
from pandas.util import _test_decorators as td

pytestmark = [pytest.mark.single_cpu]


def test_conv_read_write():
    with tm.ensure_clean() as path:

        def roundtrip(key, obj, **kwargs):
            obj.to_hdf(path, key=key, **kwargs)
            return read_hdf(path, key)

        o = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        tm.assert_series_equal(o, roundtrip("series", o))

        o = Series(range(10), dtype="float64", index=[f"i_{i}" for i in range(10)])
        tm.assert_series_equal(o, roundtrip("string_series", o))

        o = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )
        tm.assert_frame_equal(o, roundtrip("frame", o))

        # table
        df = DataFrame({"A": range(5), "B": range(5)})
        df.to_hdf(path, key="table", append=True)
        result = read_hdf(path, "table", where=["index>2"])
        tm.assert_frame_equal(df[df.index > 2], result)


def test_long_strings(setup_path):
    # GH6166
    data = ["a" * 50] * 10
    df = DataFrame({"a": data}, index=data)

    with ensure_clean_store(setup_path) as store:
        store.append("df", df, data_columns=["a"])

        result = store.select("df")
        tm.assert_frame_equal(df, result)


def test_api(tmp_path, setup_path):
    # GH4584
    # API issue when to_hdf doesn't accept append AND format args
    path = tmp_path / setup_path

    df = DataFrame(range(20))
    df.iloc[:10].to_hdf(path, key="df", append=True, format="table")
    df.iloc[10:].to_hdf(path, key="df", append=True, format="table")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    # append to False
    df.iloc[:10].to_hdf(path, key="df", append=False, format="table")
    df.iloc[10:].to_hdf(path, key="df", append=True, format="table")
    tm.assert_frame_equal(read_hdf(path, "df"), df)


def test_api_append(tmp_path, setup_path):
    path = tmp_path / setup_path

    df = DataFrame(range(20))
    df.iloc[:10].to_hdf(path, key="df", append=True)
    df.iloc[10:].to_hdf(path, key="df", append=True, format="table")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    # append to False
    df.iloc[:10].to_hdf(path, key="df", append=False, format="table")
    df.iloc[10:].to_hdf(path, key="df", append=True)
    tm.assert_frame_equal(read_hdf(path, "df"), df)


def test_api_2(tmp_path, setup_path):
    path = tmp_path / setup_path

    df = DataFrame(range(20))
    df.to_hdf(path, key="df", append=False, format="fixed")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    df.to_hdf(path, key="df", append=False, format="f")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    df.to_hdf(path, key="df", append=False)
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    df.to_hdf(path, key="df")
    tm.assert_frame_equal(read_hdf(path, "df"), df)

    with ensure_clean_store(setup_path) as store:
        df = DataFrame(range(20))

        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=True, format="table")
        store.append("df", df.iloc[10:], append=True, format="table")
        tm.assert_frame_equal(store.select("df"), df)

        # append to False
        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=False, format="table")
        store.append("df", df.iloc[10:], append=True, format="table")
        tm.assert_frame_equal(store.select("df"), df)

        # formats
        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=False, format="table")
        store.append("df", df.iloc[10:], append=True, format="table")
        tm.assert_frame_equal(store.select("df"), df)

        _maybe_remove(store, "df")
        store.append("df", df.iloc[:10], append=False, format="table")
        store.append("df", df.iloc[10:], append=True, format=None)
        tm.assert_frame_equal(store.select("df"), df)


def test_api_invalid(tmp_path, setup_path):
    path = tmp_path / setup_path
    # Invalid.
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    msg = "Can only append to Tables"

    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key="df", append=True, format="f")

    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key="df", append=True, format="fixed")

    msg = r"invalid HDFStore format specified \[foo\]"

    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, key="df", append=True, format="foo")

    with pytest.raises(TypeError, match=msg):
        df.to_hdf(path, key="df", append=False, format="foo")

    # File path doesn't exist
    path = ""
    msg = f"File {path} does not exist"

    with pytest.raises(FileNotFoundError, match=msg):
        read_hdf(path, "df")


def test_get(setup_path):
    with ensure_clean_store(setup_path) as store:
        store["a"] = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        left = store.get("a")
        right = store["a"]
        tm.assert_series_equal(left, right)

        left = store.get("/a")
        right = store["/a"]
        tm.assert_series_equal(left, right)

        with pytest.raises(KeyError, match="'No object named b in the file'"):
            store.get("b")


def test_put_integer(setup_path):
    # non-date, non-string index
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 100)))
    _check_roundtrip(df, tm.assert_frame_equal, setup_path)


def test_table_values_dtypes_roundtrip(setup_path, using_infer_string):
    with ensure_clean_store(setup_path) as store:
        df1 = DataFrame({"a": [1, 2, 3]}, dtype="f8")
        store.append("df_f8", df1)
        tm.assert_series_equal(df1.dtypes, store["df_f8"].dtypes)

        df2 = DataFrame({"a": [1, 2, 3]}, dtype="i8")
        store.append("df_i8", df2)
        tm.assert_series_equal(df2.dtypes, store["df_i8"].dtypes)

        # incompatible dtype
        msg = re.escape(
            "Cannot serialize the column [a] "
            "because its data contents are not [float] "
            "but [integer] object dtype"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df_i8", df1)

        # check creation/storage/retrieval of float32 (a bit hacky to
        # actually create them thought)
        df1 = DataFrame(np.array([[1], [2], [3]], dtype="f4"), columns=["A"])
        store.append("df_f4", df1)
        tm.assert_series_equal(df1.dtypes, store["df_f4"].dtypes)
        assert df1.dtypes.iloc[0] == "float32"

        # check with mixed dtypes
        df1 = DataFrame(
            {
                c: Series(np.random.default_rng(2).integers(5), dtype=c)
                for c in ["float32", "float64", "int32", "int64", "int16", "int8"]
            }
        )
        df1["string"] = "foo"
        df1["float322"] = 1.0
        df1["float322"] = df1["float322"].astype("float32")
        df1["bool"] = df1["float32"] > 0
        df1["time1"] = Timestamp("20130101")
        df1["time2"] = Timestamp("20130102")

        store.append("df_mixed_dtypes1", df1)
        result = store.select("df_mixed_dtypes1").dtypes.value_counts()
        result.index = [str(i) for i in result.index]
        str_dtype = "str" if using_infer_string else "object"
        expected = Series(
            {
                "float32": 2,
                "float64": 1,
                "int32": 1,
                "bool": 1,
                "int16": 1,
                "int8": 1,
                "int64": 1,
                str_dtype: 1,
                "datetime64[ns]": 2,
            },
            name="count",
        )
        result = result.sort_index()
        expected = expected.sort_index()
        tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_series(setup_path):
    s = Series(range(10), dtype="float64", index=[f"i_{i}" for i in range(10)])
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)

    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    _check_roundtrip(ts, tm.assert_series_equal, path=setup_path)

    ts2 = Series(ts.index, Index(ts.index))
    _check_roundtrip(ts2, tm.assert_series_equal, path=setup_path)

    ts3 = Series(ts.values, Index(np.asarray(ts.index)))
    _check_roundtrip(
        ts3, tm.assert_series_equal, path=setup_path, check_index_type=False
    )


def test_float_index(setup_path):
    # GH #454
    index = np.random.default_rng(2).standard_normal(10)
    s = Series(np.random.default_rng(2).standard_normal(10), index=index)
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)


def test_tuple_index(setup_path):
    # GH #492
    col = np.arange(10)
    idx = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    data = np.random.default_rng(2).standard_normal(30).reshape((3, 10))
    DF = DataFrame(data, index=idx, columns=col)

    with tm.assert_produces_warning(pd.errors.PerformanceWarning):
        _check_roundtrip(DF, tm.assert_frame_equal, path=setup_path)


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_index_types(setup_path):
    values = np.random.default_rng(2).standard_normal(2)

    func = lambda lhs, rhs: tm.assert_series_equal(lhs, rhs, check_index_type=True)

    ser = Series(values, [0, "y"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.datetime.today(), 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, ["y", 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.date.today(), "a"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [0, "y"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.datetime.today(), 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, ["y", 0])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [datetime.date.today(), "a"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [1.23, "b"])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [1, 1.53])
    _check_roundtrip(ser, func, path=setup_path)

    ser = Series(values, [1, 5])
    _check_roundtrip(ser, func, path=setup_path)

    dti = DatetimeIndex(["2012-01-01", "2012-01-02"], dtype="M8[ns]")
    ser = Series(values, index=dti)
    _check_roundtrip(ser, func, path=setup_path)

    ser.index = ser.index.as_unit("s")
    _check_roundtrip(ser, func, path=setup_path)


def test_timeseries_preepoch(setup_path, request):
    dr = bdate_range("1/1/1940", "1/1/1960")
    ts = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
    try:
        _check_roundtrip(ts, tm.assert_series_equal, path=setup_path)
    except OverflowError:
        if is_platform_windows():
            request.applymarker(
                pytest.mark.xfail("known failure on some windows platforms")
            )
        raise


@pytest.mark.parametrize(
    "compression", [False, pytest.param(True, marks=td.skip_if_windows)]
)
def test_frame(compression, setup_path):
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    # put in some random NAs
    df.iloc[0, 0] = np.nan
    df.iloc[5, 3] = np.nan

    _check_roundtrip_table(
        df, tm.assert_frame_equal, path=setup_path, compression=compression
    )
    _check_roundtrip(
        df, tm.assert_frame_equal, path=setup_path, compression=compression
    )

    tdf = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    _check_roundtrip(
        tdf, tm.assert_frame_equal, path=setup_path, compression=compression
    )

    with ensure_clean_store(setup_path) as store:
        # not consolidated
        df["foo"] = np.random.default_rng(2).standard_normal(len(df))
        store["df"] = df
        recons = store["df"]
        assert recons._mgr.is_consolidated()

    # empty
    df2 = df[:0]
    # Prevent df2 from having index with inferred_type as string
    df2.index = Index([])
    _check_roundtrip(df2[:0], tm.assert_frame_equal, path=setup_path)


def test_empty_series_frame(setup_path):
    s0 = Series(dtype=object)
    s1 = Series(name="myseries", dtype=object)
    df0 = DataFrame()
    df1 = DataFrame(index=["a", "b", "c"])
    df2 = DataFrame(columns=["d", "e", "f"])

    _check_roundtrip(s0, tm.assert_series_equal, path=setup_path)
    _check_roundtrip(s1, tm.assert_series_equal, path=setup_path)
    _check_roundtrip(df0, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df1, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=setup_path)


@pytest.mark.parametrize("dtype", [np.int64, np.float64, object, "m8[ns]", "M8[ns]"])
def test_empty_series(dtype, setup_path):
    s = Series(dtype=dtype)
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)


def test_can_serialize_dates(setup_path):
    rng = [x.date() for x in bdate_range("1/1/2000", "1/30/2000")]
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )

    _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)


def test_store_hierarchical(
    setup_path, using_infer_string, multiindex_dataframe_random_data
):
    frame = multiindex_dataframe_random_data

    if using_infer_string:
        # TODO(infer_string) make this work for string dtype
        msg = "Saving a MultiIndex with an extension dtype is not supported."
        with pytest.raises(NotImplementedError, match=msg):
            _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)
        return
    _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(frame.T, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(frame["A"], tm.assert_series_equal, path=setup_path)

    # check that the names are stored
    with ensure_clean_store(setup_path) as store:
        store["frame"] = frame
        recons = store["frame"]
        tm.assert_frame_equal(recons, frame)


@pytest.mark.parametrize(
    "compression", [False, pytest.param(True, marks=td.skip_if_windows)]
)
def test_store_mixed(compression, setup_path):
    def _make_one():
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )
        df["obj1"] = "foo"
        df["obj2"] = "bar"
        df["bool1"] = df["A"] > 0
        df["bool2"] = df["B"] > 0
        df["int1"] = 1
        df["int2"] = 2
        return df._consolidate()

    df1 = _make_one()
    df2 = _make_one()

    _check_roundtrip(df1, tm.assert_frame_equal, path=setup_path)
    _check_roundtrip(df2, tm.assert_frame_equal, path=setup_path)

    with ensure_clean_store(setup_path) as store:
        store["obj"] = df1
        tm.assert_frame_equal(store["obj"], df1)
        store["obj"] = df2
        tm.assert_frame_equal(store["obj"], df2)

    # check that can store Series of all of these types
    _check_roundtrip(
        df1["obj1"],
        tm.assert_series_equal,
        path=setup_path,
        compression=compression,
    )
    _check_roundtrip(
        df1["bool1"],
        tm.assert_series_equal,
        path=setup_path,
        compression=compression,
    )
    _check_roundtrip(
        df1["int1"],
        tm.assert_series_equal,
        path=setup_path,
        compression=compression,
    )


def _check_roundtrip(obj, comparator, path, compression=False, **kwargs):
    options = {}
    if compression:
        options["complib"] = "blosc"

    with ensure_clean_store(path, "w", **options) as store:
        store["obj"] = obj
        retrieved = store["obj"]
        comparator(retrieved, obj, **kwargs)


def _check_roundtrip_table(obj, comparator, path, compression=False):
    options = {}
    if compression:
        options["complib"] = "blosc"

    with ensure_clean_store(path, "w", **options) as store:
        store.put("obj", obj, format="table")
        retrieved = store["obj"]

        comparator(retrieved, obj)


def test_unicode_index(setup_path):
    unicode_values = ["\u03c3", "\u03c3\u03c3"]

    s = Series(
        np.random.default_rng(2).standard_normal(len(unicode_values)),
        unicode_values,
    )
    _check_roundtrip(s, tm.assert_series_equal, path=setup_path)


def test_unicode_longer_encoded(setup_path):
    # GH 11234
    char = "\u0394"
    df = DataFrame({"A": [char]})
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", encoding="utf-8")
        result = store.get("df")
        tm.assert_frame_equal(result, df)

    df = DataFrame({"A": ["a", char], "B": ["b", "b"]})
    with ensure_clean_store(setup_path) as store:
        store.put("df", df, format="table", encoding="utf-8")
        result = store.get("df")
        tm.assert_frame_equal(result, df)


def test_store_datetime_mixed(setup_path):
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    df["d"] = ts.index[:3]
    _check_roundtrip(df, tm.assert_frame_equal, path=setup_path)


def test_round_trip_equals(tmp_path, setup_path):
    # GH 9330
    df = DataFrame({"B": [1, 2], "A": ["x", "y"]})

    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format="table")
    other = read_hdf(path, "df")
    tm.assert_frame_equal(df, other)
    assert df.equals(other)
    assert other.equals(df)


def test_infer_string_columns(tmp_path, setup_path):
    # GH#
    pytest.importorskip("pyarrow")
    path = tmp_path / setup_path
    with pd.option_context("future.infer_string", True):
        df = DataFrame(1, columns=list("ABCD"), index=list(range(10))).set_index(
            ["A", "B"]
        )
        expected = df.copy()
        df.to_hdf(path, key="df", format="table")

        result = read_hdf(path, "df")
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\util\test_assert_almost_equal.py ===
import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    Index,
    NaT,
    Series,
    Timestamp,
)
import pandas._testing as tm


def _assert_almost_equal_both(a, b, **kwargs):
    """
    Check that two objects are approximately equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    tm.assert_almost_equal(a, b, **kwargs)
    tm.assert_almost_equal(b, a, **kwargs)


def _assert_not_almost_equal(a, b, **kwargs):
    """
    Check that two objects are not approximately equal.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    try:
        tm.assert_almost_equal(a, b, **kwargs)
        msg = f"{a} and {b} were approximately equal when they shouldn't have been"
        pytest.fail(reason=msg)
    except AssertionError:
        pass


def _assert_not_almost_equal_both(a, b, **kwargs):
    """
    Check that two objects are not approximately equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    _assert_not_almost_equal(a, b, **kwargs)
    _assert_not_almost_equal(b, a, **kwargs)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),
        (1.1, 1.100001),
        (np.int16(1), 1.000001),
        (np.float64(1.1), 1.1),
        (np.uint32(5), 5),
    ],
)
def test_assert_almost_equal_numbers(a, b):
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1),
        (1.1, True),
        (1, 2),
        (1.0001, np.int16(1)),
        # The following two examples are not "almost equal" due to tol.
        (0.1, 0.1001),
        (0.0011, 0.0012),
    ],
)
def test_assert_not_almost_equal_numbers(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),
        (1.1, 1.100001),
        (1.1, 1.1001),
        (0.000001, 0.000005),
        (1000.0, 1000.0005),
        # Testing this example, as per #13357
        (0.000011, 0.000012),
    ],
)
def test_assert_almost_equal_numbers_atol(a, b):
    # Equivalent to the deprecated check_less_precise=True, enforced in 2.0
    _assert_almost_equal_both(a, b, rtol=0.5e-3, atol=0.5e-3)


@pytest.mark.parametrize("a,b", [(1.1, 1.11), (0.1, 0.101), (0.000011, 0.001012)])
def test_assert_not_almost_equal_numbers_atol(a, b):
    _assert_not_almost_equal_both(a, b, atol=1e-3)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.1, 1.1),
        (1.1, 1.100001),
        (1.1, 1.1001),
        (1000.0, 1000.0005),
        (1.1, 1.11),
        (0.1, 0.101),
    ],
)
def test_assert_almost_equal_numbers_rtol(a, b):
    _assert_almost_equal_both(a, b, rtol=0.05)


@pytest.mark.parametrize("a,b", [(0.000011, 0.000012), (0.000001, 0.000005)])
def test_assert_not_almost_equal_numbers_rtol(a, b):
    _assert_not_almost_equal_both(a, b, rtol=0.05)


@pytest.mark.parametrize(
    "a,b,rtol",
    [
        (1.00001, 1.00005, 0.001),
        (-0.908356 + 0.2j, -0.908358 + 0.2j, 1e-3),
        (0.1 + 1.009j, 0.1 + 1.006j, 0.1),
        (0.1001 + 2.0j, 0.1 + 2.001j, 0.01),
    ],
)
def test_assert_almost_equal_complex_numbers(a, b, rtol):
    _assert_almost_equal_both(a, b, rtol=rtol)
    _assert_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)


@pytest.mark.parametrize(
    "a,b,rtol",
    [
        (0.58310768, 0.58330768, 1e-7),
        (-0.908 + 0.2j, -0.978 + 0.2j, 0.001),
        (0.1 + 1j, 0.1 + 2j, 0.01),
        (-0.132 + 1.001j, -0.132 + 1.005j, 1e-5),
        (0.58310768j, 0.58330768j, 1e-9),
    ],
)
def test_assert_not_almost_equal_complex_numbers(a, b, rtol):
    _assert_not_almost_equal_both(a, b, rtol=rtol)
    _assert_not_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_not_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)


@pytest.mark.parametrize("a,b", [(0, 0), (0, 0.0), (0, np.float64(0)), (0.00000001, 0)])
def test_assert_almost_equal_numbers_with_zeros(a, b):
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [(0.001, 0), (1, 0)])
def test_assert_not_almost_equal_numbers_with_zeros(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize("a,b", [(1, "abc"), (1, [1]), (1, object())])
def test_assert_not_almost_equal_numbers_with_mixed(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "left_dtype", ["M8[ns]", "m8[ns]", "float64", "int64", "object"]
)
@pytest.mark.parametrize(
    "right_dtype", ["M8[ns]", "m8[ns]", "float64", "int64", "object"]
)
def test_assert_almost_equal_edge_case_ndarrays(left_dtype, right_dtype):
    # Empty compare.
    _assert_almost_equal_both(
        np.array([], dtype=left_dtype),
        np.array([], dtype=right_dtype),
        check_dtype=False,
    )


def test_assert_almost_equal_sets():
    # GH#51727
    _assert_almost_equal_both({1, 2, 3}, {1, 2, 3})


def test_assert_almost_not_equal_sets():
    # GH#51727
    msg = r"{1, 2, 3} != {1, 2, 4}"
    with pytest.raises(AssertionError, match=msg):
        _assert_almost_equal_both({1, 2, 3}, {1, 2, 4})


def test_assert_almost_equal_dicts():
    _assert_almost_equal_both({"a": 1, "b": 2}, {"a": 1, "b": 2})


@pytest.mark.parametrize(
    "a,b",
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}),
        ({"a": 1}, 1),
        ({"a": 1}, "abc"),
        ({"a": 1}, [1]),
    ],
)
def test_assert_not_almost_equal_dicts(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize("val", [1, 2])
def test_assert_almost_equal_dict_like_object(val):
    dict_val = 1
    real_dict = {"a": val}

    class DictLikeObj:
        def keys(self):
            return ("a",)

        def __getitem__(self, item):
            if item == "a":
                return dict_val

    func = (
        _assert_almost_equal_both if val == dict_val else _assert_not_almost_equal_both
    )
    func(real_dict, DictLikeObj(), check_dtype=False)


def test_assert_almost_equal_strings():
    _assert_almost_equal_both("abc", "abc")


@pytest.mark.parametrize(
    "a,b", [("abc", "abcd"), ("abc", "abd"), ("abc", 1), ("abc", [1])]
)
def test_assert_not_almost_equal_strings(a, b):
    _assert_not_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b", [([1, 2, 3], [1, 2, 3]), (np.array([1, 2, 3]), np.array([1, 2, 3]))]
)
def test_assert_almost_equal_iterables(a, b):
    _assert_almost_equal_both(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        # Class is different.
        (np.array([1, 2, 3]), [1, 2, 3]),
        # Dtype is different.
        (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])),
        # Can't compare generators.
        (iter([1, 2, 3]), [1, 2, 3]),
        ([1, 2, 3], [1, 2, 4]),
        ([1, 2, 3], [1, 2, 3, 4]),
        ([1, 2, 3], 1),
    ],
)
def test_assert_not_almost_equal_iterables(a, b):
    _assert_not_almost_equal(a, b)


def test_assert_almost_equal_null():
    _assert_almost_equal_both(None, None)


@pytest.mark.parametrize("a,b", [(None, np.nan), (None, 0), (np.nan, 0)])
def test_assert_not_almost_equal_null(a, b):
    _assert_not_almost_equal(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        (np.inf, np.inf),
        (np.inf, float("inf")),
        (np.array([np.inf, np.nan, -np.inf]), np.array([np.inf, np.nan, -np.inf])),
    ],
)
def test_assert_almost_equal_inf(a, b):
    _assert_almost_equal_both(a, b)


objs = [NA, np.nan, NaT, None, np.datetime64("NaT"), np.timedelta64("NaT")]


@pytest.mark.parametrize("left", objs)
@pytest.mark.parametrize("right", objs)
def test_mismatched_na_assert_almost_equal_deprecation(left, right):
    left_arr = np.array([left], dtype=object)
    right_arr = np.array([right], dtype=object)

    msg = "Mismatched null-like values"

    if left is right:
        _assert_almost_equal_both(left, right, check_dtype=False)
        tm.assert_numpy_array_equal(left_arr, right_arr)
        tm.assert_index_equal(
            Index(left_arr, dtype=object), Index(right_arr, dtype=object)
        )
        tm.assert_series_equal(
            Series(left_arr, dtype=object), Series(right_arr, dtype=object)
        )
        tm.assert_frame_equal(
            DataFrame(left_arr, dtype=object), DataFrame(right_arr, dtype=object)
        )

    else:
        with tm.assert_produces_warning(FutureWarning, match=msg):
            _assert_almost_equal_both(left, right, check_dtype=False)

        # TODO: to get the same deprecation in assert_numpy_array_equal we need
        #  to change/deprecate the default for strict_nan to become True
        # TODO: to get the same deprecation in assert_index_equal we need to
        #  change/deprecate array_equivalent_object to be stricter, as
        #  assert_index_equal uses Index.equal which uses array_equivalent.
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_series_equal(
                Series(left_arr, dtype=object), Series(right_arr, dtype=object)
            )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tm.assert_frame_equal(
                DataFrame(left_arr, dtype=object), DataFrame(right_arr, dtype=object)
            )


def test_assert_not_almost_equal_inf():
    _assert_not_almost_equal_both(np.inf, 0)


@pytest.mark.parametrize(
    "a,b",
    [
        (Index([1.0, 1.1]), Index([1.0, 1.100001])),
        (Series([1.0, 1.1]), Series([1.0, 1.100001])),
        (np.array([1.1, 2.000001]), np.array([1.1, 2.0])),
        (DataFrame({"a": [1.0, 1.1]}), DataFrame({"a": [1.0, 1.100001]})),
    ],
)
def test_assert_almost_equal_pandas(a, b):
    _assert_almost_equal_both(a, b)


def test_assert_almost_equal_object():
    a = [Timestamp("2011-01-01"), Timestamp("2011-01-01")]
    b = [Timestamp("2011-01-01"), Timestamp("2011-01-01")]
    _assert_almost_equal_both(a, b)


def test_assert_almost_equal_value_mismatch():
    msg = "expected 2\\.00000 but got 1\\.00000, with rtol=1e-05, atol=1e-08"

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(1, 2)


@pytest.mark.parametrize(
    "a,b,klass1,klass2",
    [(np.array([1]), 1, "ndarray", "int"), (1, np.array([1]), "int", "ndarray")],
)
def test_assert_almost_equal_class_mismatch(a, b, klass1, klass2):
    msg = f"""numpy array are different

numpy array classes are different
\\[left\\]:  {klass1}
\\[right\\]: {klass2}"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(a, b)


def test_assert_almost_equal_value_mismatch1():
    msg = """numpy array are different

numpy array values are different \\(66\\.66667 %\\)
\\[left\\]:  \\[nan, 2\\.0, 3\\.0\\]
\\[right\\]: \\[1\\.0, nan, 3\\.0\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([np.nan, 2, 3]), np.array([1, np.nan, 3]))


def test_assert_almost_equal_value_mismatch2():
    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([1, 2]), np.array([1, 3]))


def test_assert_almost_equal_value_mismatch3():
    msg = """numpy array are different

numpy array values are different \\(16\\.66667 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\], \\[5, 6\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\], \\[5, 6\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(
            np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3], [3, 4], [5, 6]])
        )


def test_assert_almost_equal_value_mismatch4():
    msg = """numpy array are different

numpy array values are different \\(25\\.0 %\\)
\\[left\\]:  \\[\\[1, 2\\], \\[3, 4\\]\\]
\\[right\\]: \\[\\[1, 3\\], \\[3, 4\\]\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([[1, 2], [3, 4]]), np.array([[1, 3], [3, 4]]))


def test_assert_almost_equal_shape_mismatch_override():
    msg = """Index are different

Index shapes are different
\\[left\\]:  \\(2L*,\\)
\\[right\\]: \\(3L*,\\)"""
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array([1, 2]), np.array([3, 4, 5]), obj="Index")


def test_assert_almost_equal_unicode():
    # see gh-20503
    msg = """numpy array are different

numpy array values are different \\(33\\.33333 %\\)
\\[left\\]:  \\[á, à, ä\\]
\\[right\\]: \\[á, à, å\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array(["á", "à", "ä"]), np.array(["á", "à", "å"]))


def test_assert_almost_equal_timestamp():
    a = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-01")])
    b = np.array([Timestamp("2011-01-01"), Timestamp("2011-01-02")])

    msg = """numpy array are different

numpy array values are different \\(50\\.0 %\\)
\\[left\\]:  \\[2011-01-01 00:00:00, 2011-01-01 00:00:00\\]
\\[right\\]: \\[2011-01-01 00:00:00, 2011-01-02 00:00:00\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(a, b)


def test_assert_almost_equal_iterable_length_mismatch():
    msg = """Iterable are different

Iterable length are different
\\[left\\]:  2
\\[right\\]: 3"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal([1, 2], [3, 4, 5])


def test_assert_almost_equal_iterable_values_mismatch():
    msg = """Iterable are different

Iterable values are different \\(50\\.0 %\\)
\\[left\\]:  \\[1, 2\\]
\\[right\\]: \\[1, 3\\]"""

    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal([1, 2], [1, 3])


subarr = np.empty(2, dtype=object)
subarr[:] = [np.array([None, "b"], dtype=object), np.array(["c", "d"], dtype=object)]

NESTED_CASES = [
    # nested array
    (
        np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object),
        np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object),
    ),
    # >1 level of nesting
    (
        np.array(
            [
                np.array([np.array([50, 70]), np.array([90])], dtype=object),
                np.array([np.array([20, 30])], dtype=object),
            ],
            dtype=object,
        ),
        np.array(
            [
                np.array([np.array([50, 70]), np.array([90])], dtype=object),
                np.array([np.array([20, 30])], dtype=object),
            ],
            dtype=object,
        ),
    ),
    # lists
    (
        np.array([[50, 70, 90], [20, 30]], dtype=object),
        np.array([[50, 70, 90], [20, 30]], dtype=object),
    ),
    # mixed array/list
    (
        np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object),
        np.array([[1, 2, 3], [4, 5]], dtype=object),
    ),
    (
        np.array(
            [
                np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object),
                np.array(
                    [np.array([6]), np.array([7, 8]), np.array([9])], dtype=object
                ),
            ],
            dtype=object,
        ),
        np.array([[[1, 2, 3], [4, 5]], [[6], [7, 8], [9]]], dtype=object),
    ),
    # same-length lists
    (
        np.array([subarr, None], dtype=object),
        np.array([[[None, "b"], ["c", "d"]], None], dtype=object),
    ),
    # dicts
    (
        np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object),
        np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object),
    ),
    (
        np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object),
        np.array([{"f1": 1, "f2": ["a", "b"]}], dtype=object),
    ),
    # array/list of dicts
    (
        np.array(
            [
                np.array(
                    [{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object
                ),
                np.array([], dtype=object),
            ],
            dtype=object,
        ),
        np.array([[{"f1": 1, "f2": ["a", "b"]}], []], dtype=object),
    ),
]


@pytest.mark.filterwarnings("ignore:elementwise comparison failed:DeprecationWarning")
@pytest.mark.parametrize("a,b", NESTED_CASES)
def test_assert_almost_equal_array_nested(a, b):
    _assert_almost_equal_both(a, b)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_formats.py ===
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
    date_range,
    option_context,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class TestSeriesRepr:
    def test_multilevel_name_print_0(self):
        # GH#55415 None does not get printed, but 0 does
        # (matching DataFrame and flat index behavior)
        mi = pd.MultiIndex.from_product([range(2, 3), range(3, 4)], names=[0, None])
        ser = Series(1.5, index=mi)

        res = repr(ser)
        expected = "0   \n2  3    1.5\ndtype: float64"
        assert res == expected

    def test_multilevel_name_print(self, lexsorted_two_level_string_multiindex):
        index = lexsorted_two_level_string_multiindex
        ser = Series(range(len(index)), index=index, name="sth")
        expected = [
            "first  second",
            "foo    one       0",
            "       two       1",
            "       three     2",
            "bar    one       3",
            "       two       4",
            "baz    two       5",
            "       three     6",
            "qux    one       7",
            "       two       8",
            "       three     9",
            "Name: sth, dtype: int64",
        ]
        expected = "\n".join(expected)
        assert repr(ser) == expected

    def test_small_name_printing(self):
        # Test small Series.
        s = Series([0, 1, 2])

        s.name = "test"
        assert "Name: test" in repr(s)

        s.name = None
        assert "Name:" not in repr(s)

    def test_big_name_printing(self):
        # Test big Series (diff code path).
        s = Series(range(1000))

        s.name = "test"
        assert "Name: test" in repr(s)

        s.name = None
        assert "Name:" not in repr(s)

    def test_empty_name_printing(self):
        s = Series(index=date_range("20010101", "20020101"), name="test", dtype=object)
        assert "Name: test" in repr(s)

    @pytest.mark.parametrize("args", [(), (0, -1)])
    def test_float_range(self, args):
        str(
            Series(
                np.random.default_rng(2).standard_normal(1000),
                index=np.arange(1000, *args),
            )
        )

    def test_empty_object(self):
        # empty
        str(Series(dtype=object))

    def test_string(self, string_series):
        str(string_series)
        str(string_series.astype(int))

        # with NaNs
        string_series[5:7] = np.nan
        str(string_series)

    def test_object(self, object_series):
        str(object_series)

    def test_datetime(self, datetime_series):
        str(datetime_series)
        # with Nones
        ots = datetime_series.astype("O")
        ots[::2] = None
        repr(ots)

    @pytest.mark.parametrize(
        "name",
        [
            "",
            1,
            1.2,
            "foo",
            "\u03B1\u03B2\u03B3",
            "loooooooooooooooooooooooooooooooooooooooooooooooooooong",
            ("foo", "bar", "baz"),
            (1, 2),
            ("foo", 1, 2.3),
            ("\u03B1", "\u03B2", "\u03B3"),
            ("\u03B1", "bar"),
        ],
    )
    def test_various_names(self, name, string_series):
        # various names
        string_series.name = name
        repr(string_series)

    def test_tuple_name(self):
        biggie = Series(
            np.random.default_rng(2).standard_normal(1000),
            index=np.arange(1000),
            name=("foo", "bar", "baz"),
        )
        repr(biggie)

    @pytest.mark.parametrize("arg", [100, 1001])
    def test_tidy_repr_name_0(self, arg):
        # tidy repr
        ser = Series(np.random.default_rng(2).standard_normal(arg), name=0)
        rep_str = repr(ser)
        assert "Name: 0" in rep_str

    def test_newline(self, any_string_dtype):
        ser = Series(
            ["a\n\r\tb"],
            name="a\n\r\td",
            index=Index(["a\n\r\tf"], dtype=any_string_dtype),
            dtype=any_string_dtype,
        )
        assert "\t" not in repr(ser)
        assert "\r" not in repr(ser)
        assert "a\n" not in repr(ser)

    @pytest.mark.parametrize(
        "name, expected",
        [
            ["foo", "Series([], Name: foo, dtype: int64)"],
            [None, "Series([], dtype: int64)"],
        ],
    )
    def test_empty_int64(self, name, expected):
        # with empty series (#4651)
        s = Series([], dtype=np.int64, name=name)
        assert repr(s) == expected

    def test_repr_bool_fails(self, capsys):
        s = Series(
            [
                DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
                for i in range(5)
            ]
        )

        # It works (with no Cython exception barf)!
        repr(s)

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_repr_name_iterable_indexable(self):
        s = Series([1, 2, 3], name=np.int64(3))

        # it works!
        repr(s)

        s.name = ("\u05d0",) * 2
        repr(s)

    def test_repr_max_rows(self):
        # GH 6863
        with option_context("display.max_rows", None):
            str(Series(range(1001)))  # should not raise exception

    def test_unicode_string_with_unicode(self):
        df = Series(["\u05d0"], name="\u05d1")
        str(df)

        ser = Series(["\u03c3"] * 10)
        repr(ser)

        ser2 = Series(["\u05d0"] * 1000)
        ser2.name = "title1"
        repr(ser2)

    def test_str_to_bytes_raises(self):
        # GH 26447
        df = Series(["abc"], name="abc")
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(df)

    def test_timeseries_repr_object_dtype(self):
        index = Index(
            [datetime(2000, 1, 1) + timedelta(i) for i in range(1000)], dtype=object
        )
        ts = Series(np.random.default_rng(2).standard_normal(len(index)), index)
        repr(ts)

        ts = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        assert repr(ts).splitlines()[-1].startswith("Freq:")

        ts2 = ts.iloc[np.random.default_rng(2).integers(0, len(ts) - 1, 400)]
        repr(ts2).splitlines()[-1]

    def test_latex_repr(self):
        pytest.importorskip("jinja2")  # uses Styler implementation
        result = r"""\begin{tabular}{ll}
\toprule
 & 0 \\
\midrule
0 & $\alpha$ \\
1 & b \\
2 & c \\
\bottomrule
\end{tabular}
"""
        with option_context(
            "styler.format.escape", None, "styler.render.repr", "latex"
        ):
            s = Series([r"$\alpha$", "b", "c"])
            assert result == s._repr_latex_()

        assert s._repr_latex_() is None

    def test_index_repr_in_frame_with_nan(self):
        # see gh-25061
        i = Index([1, np.nan])
        s = Series([1, 2], index=i)
        exp = """1.0    1\nNaN    2\ndtype: int64"""

        assert repr(s) == exp

    def test_format_pre_1900_dates(self):
        rng = date_range("1/1/1850", "1/1/1950", freq="YE-DEC")
        msg = "DatetimeIndex.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            rng.format()
        ts = Series(1, index=rng)
        repr(ts)

    def test_series_repr_nat(self):
        series = Series([0, 1000, 2000, pd.NaT._value], dtype="M8[ns]")

        result = repr(series)
        expected = (
            "0   1970-01-01 00:00:00.000000\n"
            "1   1970-01-01 00:00:00.000001\n"
            "2   1970-01-01 00:00:00.000002\n"
            "3                          NaT\n"
            "dtype: datetime64[ns]"
        )
        assert result == expected

    def test_float_repr(self):
        # GH#35603
        # check float format when cast to object
        ser = Series([1.0]).astype(object)
        expected = "0    1.0\ndtype: object"
        assert repr(ser) == expected

    def test_different_null_objects(self):
        # GH#45263
        ser = Series([1, 2, 3, 4], [True, None, np.nan, pd.NaT])
        result = repr(ser)
        expected = "True    1\nNone    2\nNaN     3\nNaT     4\ndtype: int64"
        assert result == expected


class TestCategoricalRepr:
    def test_categorical_repr_unicode(self):
        # see gh-21002

        class County:
            name = "San Sebastián"
            state = "PR"

            def __repr__(self) -> str:
                return self.name + ", " + self.state

        cat = Categorical([County() for _ in range(61)])
        idx = Index(cat)
        ser = idx.to_series()

        repr(ser)
        str(ser)

    def test_categorical_repr(self, using_infer_string):
        a = Series(Categorical([1, 2, 3, 4]))
        exp = (
            "0    1\n1    2\n2    3\n3    4\n"
            "dtype: category\nCategories (4, int64): [1, 2, 3, 4]"
        )

        assert exp == a.__str__()

        a = Series(Categorical(["a", "b"] * 25))
        if using_infer_string:
            exp = (
                "0     a\n1     b\n"
                "     ..\n"
                "48    a\n49    b\n"
                "Length: 50, dtype: category\nCategories (2, str): [a, b]"
            )
        else:
            exp = (
                "0     a\n1     b\n"
                "     ..\n"
                "48    a\n49    b\n"
                "Length: 50, dtype: category\nCategories (2, object): ['a', 'b']"
            )
        with option_context("display.max_rows", 5):
            assert exp == repr(a)

        levs = list("abcdefghijklmnopqrstuvwxyz")
        a = Series(Categorical(["a", "b"], categories=levs, ordered=True))
        if using_infer_string:
            exp = (
                "0    a\n1    b\n"
                "dtype: category\n"
                "Categories (26, str): [a < b < c < d ... w < x < y < z]"
            )
        else:
            exp = (
                "0    a\n1    b\n"
                "dtype: category\n"
                "Categories (26, object): ['a' < 'b' < 'c' < 'd' ... "
                "'w' < 'x' < 'y' < 'z']"
            )
        assert exp == a.__str__()

    def test_categorical_series_repr(self):
        s = Series(Categorical([1, 2, 3]))
        exp = """0    1
1    2
2    3
dtype: category
Categories (3, int64): [1, 2, 3]"""

        assert repr(s) == exp

        s = Series(Categorical(np.arange(10)))
        exp = f"""0    0
1    1
2    2
3    3
4    4
5    5
6    6
7    7
8    8
9    9
dtype: category
Categories (10, {np.dtype(int)}): [0, 1, 2, 3, ..., 6, 7, 8, 9]"""

        assert repr(s) == exp

    def test_categorical_series_repr_ordered(self):
        s = Series(Categorical([1, 2, 3], ordered=True))
        exp = """0    1
1    2
2    3
dtype: category
Categories (3, int64): [1 < 2 < 3]"""

        assert repr(s) == exp

        s = Series(Categorical(np.arange(10), ordered=True))
        exp = f"""0    0
1    1
2    2
3    3
4    4
5    5
6    6
7    7
8    8
9    9
dtype: category
Categories (10, {np.dtype(int)}): [0 < 1 < 2 < 3 ... 6 < 7 < 8 < 9]"""

        assert repr(s) == exp

    def test_categorical_series_repr_datetime(self):
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        s = Series(Categorical(idx))
        exp = """0   2011-01-01 09:00:00
1   2011-01-01 10:00:00
2   2011-01-01 11:00:00
3   2011-01-01 12:00:00
4   2011-01-01 13:00:00
dtype: category
Categories (5, datetime64[ns]): [2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00,
                                 2011-01-01 12:00:00, 2011-01-01 13:00:00]"""  # noqa: E501

        assert repr(s) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        s = Series(Categorical(idx))
        exp = """0   2011-01-01 09:00:00-05:00
1   2011-01-01 10:00:00-05:00
2   2011-01-01 11:00:00-05:00
3   2011-01-01 12:00:00-05:00
4   2011-01-01 13:00:00-05:00
dtype: category
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,
                                             2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        assert repr(s) == exp

    def test_categorical_series_repr_datetime_ordered(self):
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = """0   2011-01-01 09:00:00
1   2011-01-01 10:00:00
2   2011-01-01 11:00:00
3   2011-01-01 12:00:00
4   2011-01-01 13:00:00
dtype: category
Categories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <
                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]"""  # noqa: E501

        assert repr(s) == exp

        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        s = Series(Categorical(idx, ordered=True))
        exp = """0   2011-01-01 09:00:00-05:00
1   2011-01-01 10:00:00-05:00
2   2011-01-01 11:00:00-05:00
3   2011-01-01 12:00:00-05:00
4   2011-01-01 13:00:00-05:00
dtype: category
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <
                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        assert repr(s) == exp

    def test_categorical_series_repr_period(self):
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        s = Series(Categorical(idx))
        exp = """0    2011-01-01 09:00
1    2011-01-01 10:00
2    2011-01-01 11:00
3    2011-01-01 12:00
4    2011-01-01 13:00
dtype: category
Categories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(s) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        s = Series(Categorical(idx))
        exp = """0    2011-01
1    2011-02
2    2011-03
3    2011-04
4    2011-05
dtype: category
Categories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]"""

        assert repr(s) == exp

    def test_categorical_series_repr_period_ordered(self):
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = """0    2011-01-01 09:00
1    2011-01-01 10:00
2    2011-01-01 11:00
3    2011-01-01 12:00
4    2011-01-01 13:00
dtype: category
Categories (5, period[h]): [2011-01-01 09:00 < 2011-01-01 10:00 < 2011-01-01 11:00 < 2011-01-01 12:00 <
                            2011-01-01 13:00]"""  # noqa: E501

        assert repr(s) == exp

        idx = period_range("2011-01", freq="M", periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = """0    2011-01
1    2011-02
2    2011-03
3    2011-04
4    2011-05
dtype: category
Categories (5, period[M]): [2011-01 < 2011-02 < 2011-03 < 2011-04 < 2011-05]"""

        assert repr(s) == exp

    def test_categorical_series_repr_timedelta(self):
        idx = timedelta_range("1 days", periods=5)
        s = Series(Categorical(idx))
        exp = """0   1 days
1   2 days
2   3 days
3   4 days
4   5 days
dtype: category
Categories (5, timedelta64[ns]): [1 days, 2 days, 3 days, 4 days, 5 days]"""

        assert repr(s) == exp

        idx = timedelta_range("1 hours", periods=10)
        s = Series(Categorical(idx))
        exp = """0   0 days 01:00:00
1   1 days 01:00:00
2   2 days 01:00:00
3   3 days 01:00:00
4   4 days 01:00:00
5   5 days 01:00:00
6   6 days 01:00:00
7   7 days 01:00:00
8   8 days 01:00:00
9   9 days 01:00:00
dtype: category
Categories (10, timedelta64[ns]): [0 days 01:00:00, 1 days 01:00:00, 2 days 01:00:00,
                                   3 days 01:00:00, ..., 6 days 01:00:00, 7 days 01:00:00,
                                   8 days 01:00:00, 9 days 01:00:00]"""  # noqa: E501

        assert repr(s) == exp

    def test_categorical_series_repr_timedelta_ordered(self):
        idx = timedelta_range("1 days", periods=5)
        s = Series(Categorical(idx, ordered=True))
        exp = """0   1 days
1   2 days
2   3 days
3   4 days
4   5 days
dtype: category
Categories (5, timedelta64[ns]): [1 days < 2 days < 3 days < 4 days < 5 days]"""

        assert repr(s) == exp

        idx = timedelta_range("1 hours", periods=10)
        s = Series(Categorical(idx, ordered=True))
        exp = """0   0 days 01:00:00
1   1 days 01:00:00
2   2 days 01:00:00
3   3 days 01:00:00
4   4 days 01:00:00
5   5 days 01:00:00
6   6 days 01:00:00
7   7 days 01:00:00
8   8 days 01:00:00
9   9 days 01:00:00
dtype: category
Categories (10, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 6 days 01:00:00 < 7 days 01:00:00 <
                                   8 days 01:00:00 < 9 days 01:00:00]"""  # noqa: E501

        assert repr(s) == exp

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_join.py ===
from datetime import datetime

import numpy as np
import pytest

from pandas.errors import MergeError

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    date_range,
    period_range,
)
import pandas._testing as tm
from pandas.core.reshape.concat import concat


@pytest.fixture
def frame_with_period_index():
    return DataFrame(
        data=np.arange(20).reshape(4, 5),
        columns=list("abcde"),
        index=period_range(start="2000", freq="Y", periods=4),
    )


@pytest.fixture
def left():
    return DataFrame({"a": [20, 10, 0]}, index=[2, 1, 0])


@pytest.fixture
def right():
    return DataFrame({"b": [300, 100, 200]}, index=[3, 1, 2])


@pytest.fixture
def left_no_dup():
    return DataFrame(
        {"a": ["a", "b", "c", "d"], "b": ["cat", "dog", "weasel", "horse"]},
        index=range(4),
    )


@pytest.fixture
def right_no_dup():
    return DataFrame(
        {
            "a": ["a", "b", "c", "d", "e"],
            "c": ["meow", "bark", "um... weasel noise?", "nay", "chirp"],
        },
        index=range(5),
    ).set_index("a")


@pytest.fixture
def left_w_dups(left_no_dup):
    return concat(
        [left_no_dup, DataFrame({"a": ["a"], "b": ["cow"]}, index=[3])], sort=True
    )


@pytest.fixture
def right_w_dups(right_no_dup):
    return concat(
        [right_no_dup, DataFrame({"a": ["e"], "c": ["moo"]}, index=[3])]
    ).set_index("a")


@pytest.mark.parametrize(
    "how, sort, expected",
    [
        ("inner", False, DataFrame({"a": [20, 10], "b": [200, 100]}, index=[2, 1])),
        ("inner", True, DataFrame({"a": [10, 20], "b": [100, 200]}, index=[1, 2])),
        (
            "left",
            False,
            DataFrame({"a": [20, 10, 0], "b": [200, 100, np.nan]}, index=[2, 1, 0]),
        ),
        (
            "left",
            True,
            DataFrame({"a": [0, 10, 20], "b": [np.nan, 100, 200]}, index=[0, 1, 2]),
        ),
        (
            "right",
            False,
            DataFrame({"a": [np.nan, 10, 20], "b": [300, 100, 200]}, index=[3, 1, 2]),
        ),
        (
            "right",
            True,
            DataFrame({"a": [10, 20, np.nan], "b": [100, 200, 300]}, index=[1, 2, 3]),
        ),
        (
            "outer",
            False,
            DataFrame(
                {"a": [0, 10, 20, np.nan], "b": [np.nan, 100, 200, 300]},
                index=[0, 1, 2, 3],
            ),
        ),
        (
            "outer",
            True,
            DataFrame(
                {"a": [0, 10, 20, np.nan], "b": [np.nan, 100, 200, 300]},
                index=[0, 1, 2, 3],
            ),
        ),
    ],
)
def test_join(left, right, how, sort, expected):
    result = left.join(right, how=how, sort=sort, validate="1:1")
    tm.assert_frame_equal(result, expected)


def test_suffix_on_list_join():
    first = DataFrame({"key": [1, 2, 3, 4, 5]})
    second = DataFrame({"key": [1, 8, 3, 2, 5], "v1": [1, 2, 3, 4, 5]})
    third = DataFrame({"keys": [5, 2, 3, 4, 1], "v2": [1, 2, 3, 4, 5]})

    # check proper errors are raised
    msg = "Suffixes not supported when joining multiple DataFrames"
    with pytest.raises(ValueError, match=msg):
        first.join([second], lsuffix="y")
    with pytest.raises(ValueError, match=msg):
        first.join([second, third], rsuffix="x")
    with pytest.raises(ValueError, match=msg):
        first.join([second, third], lsuffix="y", rsuffix="x")
    with pytest.raises(ValueError, match="Indexes have overlapping values"):
        first.join([second, third])

    # no errors should be raised
    arr_joined = first.join([third])
    norm_joined = first.join(third)
    tm.assert_frame_equal(arr_joined, norm_joined)


def test_join_invalid_validate(left_no_dup, right_no_dup):
    # GH 46622
    # Check invalid arguments
    msg = (
        '"invalid" is not a valid argument. '
        "Valid arguments are:\n"
        '- "1:1"\n'
        '- "1:m"\n'
        '- "m:1"\n'
        '- "m:m"\n'
        '- "one_to_one"\n'
        '- "one_to_many"\n'
        '- "many_to_one"\n'
        '- "many_to_many"'
    )
    with pytest.raises(ValueError, match=msg):
        left_no_dup.merge(right_no_dup, on="a", validate="invalid")


@pytest.mark.parametrize("dtype", ["object", "string[pyarrow]"])
def test_join_on_single_col_dup_on_right(left_no_dup, right_w_dups, dtype):
    # GH 46622
    # Dups on right allowed by one_to_many constraint
    if dtype == "string[pyarrow]":
        pytest.importorskip("pyarrow")
    left_no_dup = left_no_dup.astype(dtype)
    right_w_dups.index = right_w_dups.index.astype(dtype)
    left_no_dup.join(
        right_w_dups,
        on="a",
        validate="one_to_many",
    )

    # Dups on right not allowed by one_to_one constraint
    msg = "Merge keys are not unique in right dataset; not a one-to-one merge"
    with pytest.raises(MergeError, match=msg):
        left_no_dup.join(
            right_w_dups,
            on="a",
            validate="one_to_one",
        )


def test_join_on_single_col_dup_on_left(left_w_dups, right_no_dup):
    # GH 46622
    # Dups on left allowed by many_to_one constraint
    left_w_dups.join(
        right_no_dup,
        on="a",
        validate="many_to_one",
    )

    # Dups on left not allowed by one_to_one constraint
    msg = "Merge keys are not unique in left dataset; not a one-to-one merge"
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(
            right_no_dup,
            on="a",
            validate="one_to_one",
        )


def test_join_on_single_col_dup_on_both(left_w_dups, right_w_dups):
    # GH 46622
    # Dups on both allowed by many_to_many constraint
    left_w_dups.join(right_w_dups, on="a", validate="many_to_many")

    # Dups on both not allowed by many_to_one constraint
    msg = "Merge keys are not unique in right dataset; not a many-to-one merge"
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(
            right_w_dups,
            on="a",
            validate="many_to_one",
        )

    # Dups on both not allowed by one_to_many constraint
    msg = "Merge keys are not unique in left dataset; not a one-to-many merge"
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(
            right_w_dups,
            on="a",
            validate="one_to_many",
        )


def test_join_on_multi_col_check_dup():
    # GH 46622
    # Two column join, dups in both, but jointly no dups
    left = DataFrame(
        {
            "a": ["a", "a", "b", "b"],
            "b": [0, 1, 0, 1],
            "c": ["cat", "dog", "weasel", "horse"],
        },
        index=range(4),
    ).set_index(["a", "b"])

    right = DataFrame(
        {
            "a": ["a", "a", "b"],
            "b": [0, 1, 0],
            "d": ["meow", "bark", "um... weasel noise?"],
        },
        index=range(3),
    ).set_index(["a", "b"])

    expected_multi = DataFrame(
        {
            "a": ["a", "a", "b"],
            "b": [0, 1, 0],
            "c": ["cat", "dog", "weasel"],
            "d": ["meow", "bark", "um... weasel noise?"],
        },
        index=range(3),
    ).set_index(["a", "b"])

    # Jointly no dups allowed by one_to_one constraint
    result = left.join(right, how="inner", validate="1:1")
    tm.assert_frame_equal(result, expected_multi)


def test_join_index(float_frame):
    # left / right

    f = float_frame.loc[float_frame.index[:10], ["A", "B"]]
    f2 = float_frame.loc[float_frame.index[5:], ["C", "D"]].iloc[::-1]

    joined = f.join(f2)
    tm.assert_index_equal(f.index, joined.index)
    expected_columns = Index(["A", "B", "C", "D"])
    tm.assert_index_equal(joined.columns, expected_columns)

    joined = f.join(f2, how="left")
    tm.assert_index_equal(joined.index, f.index)
    tm.assert_index_equal(joined.columns, expected_columns)

    joined = f.join(f2, how="right")
    tm.assert_index_equal(joined.index, f2.index)
    tm.assert_index_equal(joined.columns, expected_columns)

    # inner

    joined = f.join(f2, how="inner")
    tm.assert_index_equal(joined.index, f.index[5:10])
    tm.assert_index_equal(joined.columns, expected_columns)

    # outer

    joined = f.join(f2, how="outer")
    tm.assert_index_equal(joined.index, float_frame.index.sort_values())
    tm.assert_index_equal(joined.columns, expected_columns)

    with pytest.raises(ValueError, match="join method"):
        f.join(f2, how="foo")

    # corner case - overlapping columns
    msg = "columns overlap but no suffix"
    for how in ("outer", "left", "inner"):
        with pytest.raises(ValueError, match=msg):
            float_frame.join(float_frame, how=how)


def test_join_index_more(float_frame):
    af = float_frame.loc[:, ["A", "B"]]
    bf = float_frame.loc[::2, ["C", "D"]]

    expected = af.copy()
    expected["C"] = float_frame["C"][::2]
    expected["D"] = float_frame["D"][::2]

    result = af.join(bf)
    tm.assert_frame_equal(result, expected)

    result = af.join(bf, how="right")
    tm.assert_frame_equal(result, expected[::2])

    result = bf.join(af, how="right")
    tm.assert_frame_equal(result, expected.loc[:, result.columns])


def test_join_index_series(float_frame):
    df = float_frame.copy()
    ser = df.pop(float_frame.columns[-1])
    joined = df.join(ser)

    tm.assert_frame_equal(joined, float_frame)

    ser.name = None
    with pytest.raises(ValueError, match="must have a name"):
        df.join(ser)


def test_join_overlap(float_frame):
    df1 = float_frame.loc[:, ["A", "B", "C"]]
    df2 = float_frame.loc[:, ["B", "C", "D"]]

    joined = df1.join(df2, lsuffix="_df1", rsuffix="_df2")
    df1_suf = df1.loc[:, ["B", "C"]].add_suffix("_df1")
    df2_suf = df2.loc[:, ["B", "C"]].add_suffix("_df2")

    no_overlap = float_frame.loc[:, ["A", "D"]]
    expected = df1_suf.join(df2_suf).join(no_overlap)

    # column order not necessarily sorted
    tm.assert_frame_equal(joined, expected.loc[:, joined.columns])


def test_join_period_index(frame_with_period_index):
    other = frame_with_period_index.rename(columns=lambda key: f"{key}{key}")

    joined_values = np.concatenate([frame_with_period_index.values] * 2, axis=1)

    joined_cols = frame_with_period_index.columns.append(other.columns)

    joined = frame_with_period_index.join(other)
    expected = DataFrame(
        data=joined_values, columns=joined_cols, index=frame_with_period_index.index
    )

    tm.assert_frame_equal(joined, expected)


def test_join_left_sequence_non_unique_index():
    # https://github.com/pandas-dev/pandas/issues/19607
    df1 = DataFrame({"a": [0, 10, 20]}, index=[1, 2, 3])
    df2 = DataFrame({"b": [100, 200, 300]}, index=[4, 3, 2])
    df3 = DataFrame({"c": [400, 500, 600]}, index=[2, 2, 4])

    joined = df1.join([df2, df3], how="left")

    expected = DataFrame(
        {
            "a": [0, 10, 10, 20],
            "b": [np.nan, 300, 300, 200],
            "c": [np.nan, 400, 500, np.nan],
        },
        index=[1, 2, 2, 3],
    )

    tm.assert_frame_equal(joined, expected)


def test_join_list_series(float_frame):
    # GH#46850
    # Join a DataFrame with a list containing both a Series and a DataFrame
    left = float_frame.A.to_frame()
    right = [float_frame.B, float_frame[["C", "D"]]]
    result = left.join(right)
    tm.assert_frame_equal(result, float_frame)


@pytest.mark.parametrize("sort_kw", [True, False])
def test_suppress_future_warning_with_sort_kw(sort_kw):
    a = DataFrame({"col1": [1, 2]}, index=["c", "a"])

    b = DataFrame({"col2": [4, 5]}, index=["b", "a"])

    c = DataFrame({"col3": [7, 8]}, index=["a", "b"])

    expected = DataFrame(
        {
            "col1": {"a": 2.0, "b": float("nan"), "c": 1.0},
            "col2": {"a": 5.0, "b": 4.0, "c": float("nan")},
            "col3": {"a": 7.0, "b": 8.0, "c": float("nan")},
        }
    )
    if sort_kw is False:
        expected = expected.reindex(index=["c", "a", "b"])

    with tm.assert_produces_warning(None):
        result = a.join([b, c], how="outer", sort=sort_kw)
    tm.assert_frame_equal(result, expected)


class TestDataFrameJoin:
    def test_join(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        a = frame.loc[frame.index[:5], ["A"]]
        b = frame.loc[frame.index[2:], ["B", "C"]]

        joined = a.join(b, how="outer").reindex(frame.index)
        expected = frame.copy().values.copy()
        expected[np.isnan(joined.values)] = np.nan
        expected = DataFrame(expected, index=frame.index, columns=frame.columns)

        assert not np.isnan(joined.values).all()

        tm.assert_frame_equal(joined, expected)

    def test_join_segfault(self):
        # GH#1532
        df1 = DataFrame({"a": [1, 1], "b": [1, 2], "x": [1, 2]})
        df2 = DataFrame({"a": [2, 2], "b": [1, 2], "y": [1, 2]})
        df1 = df1.set_index(["a", "b"])
        df2 = df2.set_index(["a", "b"])
        # it works!
        for how in ["left", "right", "outer"]:
            df1.join(df2, how=how)

    def test_join_str_datetime(self):
        str_dates = ["20120209", "20120222"]
        dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]

        A = DataFrame(str_dates, index=range(2), columns=["aa"])
        C = DataFrame([[1, 2], [3, 4]], index=str_dates, columns=dt_dates)

        tst = A.join(C, on="aa")

        assert len(tst.columns) == 3

    def test_join_multiindex_leftright(self):
        # GH 10741
        df1 = DataFrame(
            [
                ["a", "x", 0.471780],
                ["a", "y", 0.774908],
                ["a", "z", 0.563634],
                ["b", "x", -0.353756],
                ["b", "y", 0.368062],
                ["b", "z", -1.721840],
                ["c", "x", 1],
                ["c", "y", 2],
                ["c", "z", 3],
            ],
            columns=["first", "second", "value1"],
        ).set_index(["first", "second"])

        df2 = DataFrame([["a", 10], ["b", 20]], columns=["first", "value2"]).set_index(
            ["first"]
        )

        exp = DataFrame(
            [
                [0.471780, 10],
                [0.774908, 10],
                [0.563634, 10],
                [-0.353756, 20],
                [0.368062, 20],
                [-1.721840, 20],
                [1.000000, np.nan],
                [2.000000, np.nan],
                [3.000000, np.nan],
            ],
            index=df1.index,
            columns=["value1", "value2"],
        )

        # these must be the same results (but columns are flipped)
        tm.assert_frame_equal(df1.join(df2, how="left"), exp)
        tm.assert_frame_equal(df2.join(df1, how="right"), exp[["value2", "value1"]])

        exp_idx = MultiIndex.from_product(
            [["a", "b"], ["x", "y", "z"]], names=["first", "second"]
        )
        exp = DataFrame(
            [
                [0.471780, 10],
                [0.774908, 10],
                [0.563634, 10],
                [-0.353756, 20],
                [0.368062, 20],
                [-1.721840, 20],
            ],
            index=exp_idx,
            columns=["value1", "value2"],
        )

        tm.assert_frame_equal(df1.join(df2, how="right"), exp)
        tm.assert_frame_equal(df2.join(df1, how="left"), exp[["value2", "value1"]])

    def test_join_multiindex_dates(self):
        # GH 33692
        date = pd.Timestamp(2000, 1, 1).date()

        df1_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        df1 = DataFrame({"col1": [0]}, index=df1_index)
        df2_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        df2 = DataFrame({"col2": [0]}, index=df2_index)
        df3_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        df3 = DataFrame({"col3": [0]}, index=df3_index)

        result = df1.join([df2, df3])

        expected_index = MultiIndex.from_tuples([(0, date)], names=["index_0", "date"])
        expected = DataFrame(
            {"col1": [0], "col2": [0], "col3": [0]}, index=expected_index
        )

        tm.assert_equal(result, expected)

    def test_merge_join_different_levels_raises(self):
        # GH#9455
        # GH 40993: For raising, enforced in 2.0

        # first dataframe
        df1 = DataFrame(columns=["a", "b"], data=[[1, 11], [0, 22]])

        # second dataframe
        columns = MultiIndex.from_tuples([("a", ""), ("c", "c1")])
        df2 = DataFrame(columns=columns, data=[[1, 33], [0, 44]])

        # merge
        with pytest.raises(
            MergeError, match="Not allowed to merge between different levels"
        ):
            pd.merge(df1, df2, on="a")

        # join, see discussion in GH#12219
        with pytest.raises(
            MergeError, match="Not allowed to merge between different levels"
        ):
            df1.join(df2, on="a")

    def test_frame_join_tzaware(self):
        test1 = DataFrame(
            np.zeros((6, 3)),
            index=date_range(
                "2012-11-15 00:00:00", periods=6, freq="100ms", tz="US/Central"
            ),
        )
        test2 = DataFrame(
            np.zeros((3, 3)),
            index=date_range(
                "2012-11-15 00:00:00", periods=3, freq="250ms", tz="US/Central"
            ),
            columns=range(3, 6),
        )

        result = test1.join(test2, how="outer")
        expected = test1.index.union(test2.index)

        tm.assert_index_equal(result.index, expected)
        assert result.index.tz.zone == "US/Central"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\blinker\__init__.py ===
from __future__ import annotations

from .base import ANY
from .base import default_namespace
from .base import NamedSignal
from .base import Namespace
from .base import Signal
from .base import signal

__all__ = [
    "ANY",
    "default_namespace",
    "NamedSignal",
    "Namespace",
    "Signal",
    "signal",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\signals.py ===
from __future__ import annotations

from blinker import Namespace

# This namespace is only for signals provided by Flask itself.
_signals = Namespace()

template_rendered = _signals.signal("template-rendered")
before_render_template = _signals.signal("before-render-template")
request_started = _signals.signal("request-started")
request_finished = _signals.signal("request-finished")
request_tearing_down = _signals.signal("request-tearing-down")
got_request_exception = _signals.signal("got-request-exception")
appcontext_tearing_down = _signals.signal("appcontext-tearing-down")
appcontext_pushed = _signals.signal("appcontext-pushed")
appcontext_popped = _signals.signal("appcontext-popped")
message_flashed = _signals.signal("message-flashed")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_wtf\recaptcha\fields.py ===
from wtforms.fields import Field

from . import widgets
from .validators import Recaptcha

__all__ = ["RecaptchaField"]


class RecaptchaField(Field):
    widget = widgets.RecaptchaWidget()

    # error message if recaptcha validation fails
    recaptcha_error = None

    def __init__(self, label="", validators=None, **kwargs):
        validators = validators or [Recaptcha()]
        super().__init__(label, validators, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flatbuffers\_version.py ===
# Copyright 2019 Google Inc. All rights reserved.
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

# Placeholder, to be updated during the release process
# by the setup.py
__version__ = u"25.2.10"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\fft\helper.py ===
def __getattr__(attr_name):
    import warnings

    from numpy.fft import _helper
    ret = getattr(_helper, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.fft.helper' has no attribute {attr_name}")
    warnings.warn(
        "The numpy.fft.helper has been made private and renamed to "
        "numpy.fft._helper. All four functions exported by it (i.e. fftshift, "
        "ifftshift, fftfreq, rfftfreq) are available from numpy.fft. "
        f"Please use numpy.fft.{attr_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\linalg\linalg.py ===
def __getattr__(attr_name):
    import warnings

    from numpy.linalg import _linalg
    ret = getattr(_linalg, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.linalg.linalg' has no attribute {attr_name}")
    warnings.warn(
        "The numpy.linalg.linalg has been made private and renamed to "
        "numpy.linalg._linalg. All public functions exported by it are "
        f"available from numpy.linalg. Please use numpy.linalg.{attr_name} "
        "instead.",
        DeprecationWarning,
        stacklevel=3
    )
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\compat\product.py ===
# Copyright (c) 2010-2024 openpyxl

"""
math.prod equivalent for < Python 3.8
"""

import functools
import operator

def product(sequence):
    return functools.reduce(operator.mul, sequence)


try:
    from math import prod
except ImportError:
    prod = product

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\relation.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.xml.constants import CHART_NS

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors.excel import Relation


class ChartRelation(Serialisable):

    tagname = "chart"
    namespace = CHART_NS

    id = Relation()

    def __init__(self, id):
        self.id = id

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\protection.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors import Bool
from openpyxl.descriptors.serialisable import Serialisable


class Protection(Serialisable):
    """Protection options for use in styles."""

    tagname = "protection"

    locked = Bool()
    hidden = Bool()

    def __init__(self, locked=True, hidden=False):
        self.locked = locked
        self.hidden = hidden

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\utils\__init__.py ===
# Copyright (c) 2010-2024 openpyxl


from .cell import (
    absolute_coordinate,
    cols_from_range,
    column_index_from_string,
    coordinate_to_tuple,
    get_column_letter,
    get_column_interval,
    quote_sheetname,
    range_boundaries,
    range_to_tuple,
    rows_from_range,
)

from .formulas import FORMULAE

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\related.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors.excel import Relation


class Related(Serialisable):

    id = Relation()


    def __init__(self, id=None):
        self.id = id


    def to_tree(self, tagname, idx=None):
        return super().to_tree(tagname)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\api\indexers\__init__.py ===
"""
Public API for Rolling Window Indexers.
"""

from pandas.core.indexers import check_array_indexer
from pandas.core.indexers.objects import (
    BaseIndexer,
    FixedForwardWindowIndexer,
    VariableOffsetWindowIndexer,
)

__all__ = [
    "check_array_indexer",
    "BaseIndexer",
    "FixedForwardWindowIndexer",
    "VariableOffsetWindowIndexer",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\__main__.py ===
"""
    pygments.__main__
    ~~~~~~~~~~~~~~~~~

    Main entry point for ``python -m pygments``.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import sys
from pip._vendor.pygments.cmdline import main

try:
    sys.exit(main(sys.argv))
except KeyboardInterrupt:
    sys.exit(1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\certs.py ===
#!/usr/bin/env python

"""
requests.certs
~~~~~~~~~~~~~~

This module returns the preferred default CA certificate bundle. There is
only one — the one from the certifi package.

If you are packaging Requests, e.g., for a Linux distribution or a managed
environment, you can change the definition of where() to return a separately
packaged CA bundle.
"""
from pip._vendor.certifi import where

if __name__ == "__main__":
    print(where())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_pick.py ===
from typing import Optional


def pick_bool(*values: Optional[bool]) -> bool:
    """Pick the first non-none bool or return the last value.

    Args:
        *values (bool): Any number of boolean or None values.

    Returns:
        bool: First non-none boolean.
    """
    assert values, "1 or more values required"
    for value in values:
        if value is not None:
            return value
    return bool(value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\certs.py ===
#!/usr/bin/env python

"""
requests.certs
~~~~~~~~~~~~~~

This module returns the preferred default CA certificate bundle. There is
only one — the one from the certifi package.

If you are packaging Requests, e.g., for a Linux distribution or a managed
environment, you can change the definition of where() to return a separately
packaged CA bundle.
"""
from certifi import where

if __name__ == "__main__":
    print(where())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sniffio\__init__.py ===
"""Top-level package for sniffio."""

__all__ = [
    "current_async_library",
    "AsyncLibraryNotFoundError",
    "current_async_library_cvar",
    "thread_local",
]

from ._version import __version__

from ._impl import (
    current_async_library,
    AsyncLibraryNotFoundError,
    current_async_library_cvar,
    thread_local,
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\events.py ===
# events.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Core event interfaces."""

from __future__ import annotations

from .engine.events import ConnectionEvents
from .engine.events import DialectEvents
from .pool import PoolResetState
from .pool.events import PoolEvents
from .sql.base import SchemaEventTarget
from .sql.events import DDLEvents

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\control\__init__.py ===
from .lti import (TransferFunction, PIDController, Series, MIMOSeries, Parallel, MIMOParallel,
    Feedback, MIMOFeedback, TransferFunctionMatrix, StateSpace, gbt, bilinear, forward_diff,
    backward_diff, phase_margin, gain_margin)
from .control_plots import (pole_zero_numerical_data, pole_zero_plot, step_response_numerical_data,
    step_response_plot, impulse_response_numerical_data, impulse_response_plot, ramp_response_numerical_data,
    ramp_response_plot, bode_magnitude_numerical_data, bode_phase_numerical_data, bode_magnitude_plot,
    bode_phase_plot, bode_plot, nyquist_plot_expr, nyquist_plot, nichols_plot_expr, nichols_plot)

__all__ = ['TransferFunction', 'PIDController', 'Series', 'MIMOSeries', 'Parallel',
    'MIMOParallel', 'Feedback', 'MIMOFeedback', 'TransferFunctionMatrix', 'StateSpace',
    'gbt', 'bilinear', 'forward_diff', 'backward_diff', 'phase_margin', 'gain_margin',
    'pole_zero_numerical_data', 'pole_zero_plot', 'step_response_numerical_data',
    'step_response_plot', 'impulse_response_numerical_data', 'impulse_response_plot',
    'ramp_response_numerical_data', 'ramp_response_plot',
    'bode_magnitude_numerical_data', 'bode_phase_numerical_data',
    'bode_magnitude_plot', 'bode_phase_plot', 'bode_plot', 'nyquist_plot_expr', 'nyquist_plot',
    'nichols_plot_expr', 'nichols_plot']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\tests\test_quantities.py ===
import warnings

from sympy.core.add import Add
from sympy.core.function import (Function, diff)
from sympy.core.numbers import (Number, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate
from sympy.physics.units import (amount_of_substance, area, convert_to, find_unit,
                                 volume, kilometer, joule, molar_gas_constant,
                                 vacuum_permittivity, elementary_charge, volt,
                                 ohm)
from sympy.physics.units.definitions import (amu, au, centimeter, coulomb,
    day, foot, grams, hour, inch, kg, km, m, meter, millimeter,
    minute, quart, s, second, speed_of_light, bit,
    byte, kibibyte, mebibyte, gibibyte, tebibyte, pebibyte, exbibyte,
    kilogram, gravitational_constant, electron_rest_mass)

from sympy.physics.units.definitions.dimension_definitions import (
    Dimension, charge, length, time, temperature, pressure,
    energy, mass
)
from sympy.physics.units.prefixes import PREFIXES, kilo
from sympy.physics.units.quantities import PhysicalConstant, Quantity
from sympy.physics.units.systems import SI
from sympy.testing.pytest import raises

k = PREFIXES["k"]


def test_str_repr():
    assert str(kg) == "kilogram"


def test_eq():
    # simple test
    assert 10*m == 10*m
    assert 10*m != 10*s


def test_convert_to():
    q = Quantity("q1")
    q.set_global_relative_scale_factor(S(5000), meter)

    assert q.convert_to(m) == 5000*m

    assert speed_of_light.convert_to(m / s) == 299792458 * m / s
    assert day.convert_to(s) == 86400*s

    # Wrong dimension to convert:
    assert q.convert_to(s) == q
    assert speed_of_light.convert_to(m) == speed_of_light

    expr = joule*second
    conv = convert_to(expr, joule)
    assert conv == joule*second


def test_Quantity_definition():
    q = Quantity("s10", abbrev="sabbr")
    q.set_global_relative_scale_factor(10, second)
    u = Quantity("u", abbrev="dam")
    u.set_global_relative_scale_factor(10, meter)
    km = Quantity("km")
    km.set_global_relative_scale_factor(kilo, meter)
    v = Quantity("u")
    v.set_global_relative_scale_factor(5*kilo, meter)

    assert q.scale_factor == 10
    assert q.dimension == time
    assert q.abbrev == Symbol("sabbr")

    assert u.dimension == length
    assert u.scale_factor == 10
    assert u.abbrev == Symbol("dam")

    assert km.scale_factor == 1000
    assert km.func(*km.args) == km
    assert km.func(*km.args).args == km.args

    assert v.dimension == length
    assert v.scale_factor == 5000


def test_abbrev():
    u = Quantity("u")
    u.set_global_relative_scale_factor(S.One, meter)

    assert u.name == Symbol("u")
    assert u.abbrev == Symbol("u")

    u = Quantity("u", abbrev="om")
    u.set_global_relative_scale_factor(S(2), meter)

    assert u.name == Symbol("u")
    assert u.abbrev == Symbol("om")
    assert u.scale_factor == 2
    assert isinstance(u.scale_factor, Number)

    u = Quantity("u", abbrev="ikm")
    u.set_global_relative_scale_factor(3*kilo, meter)

    assert u.abbrev == Symbol("ikm")
    assert u.scale_factor == 3000


def test_print():
    u = Quantity("unitname", abbrev="dam")
    assert repr(u) == "unitname"
    assert str(u) == "unitname"


def test_Quantity_eq():
    u = Quantity("u", abbrev="dam")
    v = Quantity("v1")
    assert u != v
    v = Quantity("v2", abbrev="ds")
    assert u != v
    v = Quantity("v3", abbrev="dm")
    assert u != v


def test_add_sub():
    u = Quantity("u")
    v = Quantity("v")
    w = Quantity("w")

    u.set_global_relative_scale_factor(S(10), meter)
    v.set_global_relative_scale_factor(S(5), meter)
    w.set_global_relative_scale_factor(S(2), second)

    assert isinstance(u + v, Add)
    assert (u + v.convert_to(u)) == (1 + S.Half)*u
    assert isinstance(u - v, Add)
    assert (u - v.convert_to(u)) == S.Half*u


def test_quantity_abs():
    v_w1 = Quantity('v_w1')
    v_w2 = Quantity('v_w2')
    v_w3 = Quantity('v_w3')

    v_w1.set_global_relative_scale_factor(1, meter/second)
    v_w2.set_global_relative_scale_factor(1, meter/second)
    v_w3.set_global_relative_scale_factor(1, meter/second)

    expr = v_w3 - Abs(v_w1 - v_w2)

    assert SI.get_dimensional_expr(v_w1) == (length/time).name

    Dq = Dimension(SI.get_dimensional_expr(expr))

    assert SI.get_dimension_system().get_dimensional_dependencies(Dq) == {
        length: 1,
        time: -1,
    }
    assert meter == sqrt(meter**2)


def test_check_unit_consistency():
    u = Quantity("u")
    v = Quantity("v")
    w = Quantity("w")

    u.set_global_relative_scale_factor(S(10), meter)
    v.set_global_relative_scale_factor(S(5), meter)
    w.set_global_relative_scale_factor(S(2), second)

    def check_unit_consistency(expr):
        SI._collect_factor_and_dimension(expr)

    raises(ValueError, lambda: check_unit_consistency(u + w))
    raises(ValueError, lambda: check_unit_consistency(u - w))
    raises(ValueError, lambda: check_unit_consistency(u + 1))
    raises(ValueError, lambda: check_unit_consistency(u - 1))
    raises(ValueError, lambda: check_unit_consistency(1 - exp(u / w)))


def test_mul_div():
    u = Quantity("u")
    v = Quantity("v")
    t = Quantity("t")
    ut = Quantity("ut")
    v2 = Quantity("v")

    u.set_global_relative_scale_factor(S(10), meter)
    v.set_global_relative_scale_factor(S(5), meter)
    t.set_global_relative_scale_factor(S(2), second)
    ut.set_global_relative_scale_factor(S(20), meter*second)
    v2.set_global_relative_scale_factor(S(5), meter/second)

    assert 1 / u == u**(-1)
    assert u / 1 == u

    v1 = u / t
    v2 = v

    # Pow only supports structural equality:
    assert v1 != v2
    assert v1 == v2.convert_to(v1)

    # TODO: decide whether to allow such expression in the future
    # (requires somehow manipulating the core).
    # assert u / Quantity('l2', dimension=length, scale_factor=2) == 5

    assert u * 1 == u

    ut1 = u * t
    ut2 = ut

    # Mul only supports structural equality:
    assert ut1 != ut2
    assert ut1 == ut2.convert_to(ut1)

    # Mul only supports structural equality:
    lp1 = Quantity("lp1")
    lp1.set_global_relative_scale_factor(S(2), 1/meter)
    assert u * lp1 != 20

    assert u**0 == 1
    assert u**1 == u

    # TODO: Pow only support structural equality:
    u2 = Quantity("u2")
    u3 = Quantity("u3")
    u2.set_global_relative_scale_factor(S(100), meter**2)
    u3.set_global_relative_scale_factor(Rational(1, 10), 1/meter)

    assert u ** 2 != u2
    assert u ** -1 != u3

    assert u ** 2 == u2.convert_to(u)
    assert u ** -1 == u3.convert_to(u)


def test_units():
    assert convert_to((5*m/s * day) / km, 1) == 432
    assert convert_to(foot / meter, meter) == Rational(3048, 10000)
    # amu is a pure mass so mass/mass gives a number, not an amount (mol)
    # TODO: need better simplification routine:
    assert str(convert_to(grams/amu, grams).n(2)) == '6.0e+23'

    # Light from the sun needs about 8.3 minutes to reach earth
    t = (1*au / speed_of_light) / minute
    # TODO: need a better way to simplify expressions containing units:
    t = convert_to(convert_to(t, meter / minute), meter)
    assert t.simplify() == Rational(49865956897, 5995849160)

    # TODO: fix this, it should give `m` without `Abs`
    assert sqrt(m**2) == m
    assert (sqrt(m))**2 == m

    t = Symbol('t')
    assert integrate(t*m/s, (t, 1*s, 5*s)) == 12*m*s
    assert (t * m/s).integrate((t, 1*s, 5*s)) == 12*m*s


def test_issue_quart():
    assert convert_to(4 * quart / inch ** 3, meter) == 231
    assert convert_to(4 * quart / inch ** 3, millimeter) == 231

def test_electron_rest_mass():
    assert convert_to(electron_rest_mass, kilogram) == 9.1093837015e-31*kilogram
    assert convert_to(electron_rest_mass, grams) == 9.1093837015e-28*grams

def test_issue_5565():
    assert (m < s).is_Relational


def test_find_unit():
    assert find_unit('coulomb') == ['coulomb', 'coulombs', 'coulomb_constant']
    assert find_unit(coulomb) == ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    assert find_unit(charge) == ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    assert find_unit(inch) == [
        'm', 'au', 'cm', 'dm', 'ft', 'km', 'ly', 'mi', 'mm', 'nm', 'pm', 'um', 'yd',
        'nmi', 'feet', 'foot', 'inch', 'mile', 'yard', 'meter', 'miles', 'yards',
        'inches', 'meters', 'micron', 'microns', 'angstrom', 'angstroms', 'decimeter',
        'kilometer', 'lightyear', 'nanometer', 'picometer', 'centimeter', 'decimeters',
        'kilometers', 'lightyears', 'micrometer', 'millimeter', 'nanometers', 'picometers',
        'centimeters', 'micrometers', 'millimeters', 'nautical_mile', 'planck_length',
        'nautical_miles', 'astronomical_unit', 'astronomical_units']
    assert find_unit(inch**-1) == ['D', 'dioptre', 'optical_power']
    assert find_unit(length**-1) == ['D', 'dioptre', 'optical_power']
    assert find_unit(inch ** 2) == ['ha', 'hectare', 'planck_area']
    assert find_unit(inch ** 3) == [
        'L', 'l', 'cL', 'cl', 'dL', 'dl', 'mL', 'ml', 'liter', 'quart', 'liters', 'quarts',
        'deciliter', 'centiliter', 'deciliters', 'milliliter',
        'centiliters', 'milliliters', 'planck_volume']
    assert find_unit('voltage') == ['V', 'v', 'volt', 'volts', 'planck_voltage']
    assert find_unit(grams) == ['g', 't', 'Da', 'kg', 'me', 'mg', 'ug', 'amu', 'mmu', 'amus',
                                'gram', 'mmus', 'grams', 'pound', 'tonne', 'dalton', 'pounds',
                                'kilogram', 'kilograms', 'microgram', 'milligram', 'metric_ton',
                                'micrograms', 'milligrams', 'planck_mass', 'milli_mass_unit', 'atomic_mass_unit',
                                'electron_rest_mass', 'atomic_mass_constant']


def test_Quantity_derivative():
    x = symbols("x")
    assert diff(x*meter, x) == meter
    assert diff(x**3*meter**2, x) == 3*x**2*meter**2
    assert diff(meter, meter) == 1
    assert diff(meter**2, meter) == 2*meter


def test_quantity_postprocessing():
    q1 = Quantity('q1')
    q2 = Quantity('q2')

    SI.set_quantity_dimension(q1, length*pressure**2*temperature/time)
    SI.set_quantity_dimension(q2, energy*pressure*temperature/(length**2*time))

    assert q1 + q2
    q = q1 + q2
    Dq = Dimension(SI.get_dimensional_expr(q))
    assert SI.get_dimension_system().get_dimensional_dependencies(Dq) == {
        length: -1,
        mass: 2,
        temperature: 1,
        time: -5,
    }


def test_factor_and_dimension():
    assert (3000, Dimension(1)) == SI._collect_factor_and_dimension(3000)
    assert (1001, length) == SI._collect_factor_and_dimension(meter + km)
    assert (2, length/time) == SI._collect_factor_and_dimension(
        meter/second + 36*km/(10*hour))

    x, y = symbols('x y')
    assert (x + y/100, length) == SI._collect_factor_and_dimension(
        x*m + y*centimeter)

    cH = Quantity('cH')
    SI.set_quantity_dimension(cH, amount_of_substance/volume)

    pH = -log(cH)

    assert (1, volume/amount_of_substance) == SI._collect_factor_and_dimension(
        exp(pH))

    v_w1 = Quantity('v_w1')
    v_w2 = Quantity('v_w2')

    v_w1.set_global_relative_scale_factor(Rational(3, 2), meter/second)
    v_w2.set_global_relative_scale_factor(2, meter/second)

    expr = Abs(v_w1/2 - v_w2)
    assert (Rational(5, 4), length/time) == \
        SI._collect_factor_and_dimension(expr)

    expr = Rational(5, 2)*second/meter*v_w1 - 3000
    assert (-(2996 + Rational(1, 4)), Dimension(1)) == \
        SI._collect_factor_and_dimension(expr)

    expr = v_w1**(v_w2/v_w1)
    assert ((Rational(3, 2))**Rational(4, 3), (length/time)**Rational(4, 3)) == \
        SI._collect_factor_and_dimension(expr)


def test_dimensional_expr_of_derivative():
    l = Quantity('l')
    t = Quantity('t')
    t1 = Quantity('t1')
    l.set_global_relative_scale_factor(36, km)
    t.set_global_relative_scale_factor(1, hour)
    t1.set_global_relative_scale_factor(1, second)
    x = Symbol('x')
    y = Symbol('y')
    f = Function('f')
    dfdx = f(x, y).diff(x, y)
    dl_dt = dfdx.subs({f(x, y): l, x: t, y: t1})
    assert SI.get_dimensional_expr(dl_dt) ==\
        SI.get_dimensional_expr(l / t / t1) ==\
        Symbol("length")/Symbol("time")**2
    assert SI._collect_factor_and_dimension(dl_dt) ==\
        SI._collect_factor_and_dimension(l / t / t1) ==\
        (10, length/time**2)


def test_get_dimensional_expr_with_function():
    v_w1 = Quantity('v_w1')
    v_w2 = Quantity('v_w2')
    v_w1.set_global_relative_scale_factor(1, meter/second)
    v_w2.set_global_relative_scale_factor(1, meter/second)

    assert SI.get_dimensional_expr(sin(v_w1)) == \
        sin(SI.get_dimensional_expr(v_w1))
    assert SI.get_dimensional_expr(sin(v_w1/v_w2)) == 1


def test_binary_information():
    assert convert_to(kibibyte, byte) == 1024*byte
    assert convert_to(mebibyte, byte) == 1024**2*byte
    assert convert_to(gibibyte, byte) == 1024**3*byte
    assert convert_to(tebibyte, byte) == 1024**4*byte
    assert convert_to(pebibyte, byte) == 1024**5*byte
    assert convert_to(exbibyte, byte) == 1024**6*byte

    assert kibibyte.convert_to(bit) == 8*1024*bit
    assert byte.convert_to(bit) == 8*bit

    a = 10*kibibyte*hour

    assert convert_to(a, byte) == 10240*byte*hour
    assert convert_to(a, minute) == 600*kibibyte*minute
    assert convert_to(a, [byte, minute]) == 614400*byte*minute


def test_conversion_with_2_nonstandard_dimensions():
    good_grade = Quantity("good_grade")
    kilo_good_grade = Quantity("kilo_good_grade")
    centi_good_grade = Quantity("centi_good_grade")

    kilo_good_grade.set_global_relative_scale_factor(1000, good_grade)
    centi_good_grade.set_global_relative_scale_factor(S.One/10**5, kilo_good_grade)

    charity_points = Quantity("charity_points")
    milli_charity_points = Quantity("milli_charity_points")
    missions = Quantity("missions")

    milli_charity_points.set_global_relative_scale_factor(S.One/1000, charity_points)
    missions.set_global_relative_scale_factor(251, charity_points)

    assert convert_to(
        kilo_good_grade*milli_charity_points*millimeter,
        [centi_good_grade, missions, centimeter]
    ) == S.One * 10**5 / (251*1000) / 10 * centi_good_grade*missions*centimeter


def test_eval_subs():
    energy, mass, force = symbols('energy mass force')
    expr1 = energy/mass
    units = {energy: kilogram*meter**2/second**2, mass: kilogram}
    assert expr1.subs(units) == meter**2/second**2
    expr2 = force/mass
    units = {force:gravitational_constant*kilogram**2/meter**2, mass:kilogram}
    assert expr2.subs(units) == gravitational_constant*kilogram/meter**2


def test_issue_14932():
    assert (log(inch) - log(2)).simplify() == log(inch/2)
    assert (log(inch) - log(foot)).simplify() == -log(12)
    p = symbols('p', positive=True)
    assert (log(inch) - log(p)).simplify() == log(inch/p)


def test_issue_14547():
    # the root issue is that an argument with dimensions should
    # not raise an error when the `arg - 1` calculation is
    # performed in the assumptions system
    from sympy.physics.units import foot, inch
    from sympy.core.relational import Eq
    assert log(foot).is_zero is None
    assert log(foot).is_positive is None
    assert log(foot).is_nonnegative is None
    assert log(foot).is_negative is None
    assert log(foot).is_algebraic is None
    assert log(foot).is_rational is None
    # doesn't raise error
    assert Eq(log(foot), log(inch)) is not None  # might be False or unevaluated

    x = Symbol('x')
    e = foot + x
    assert e.is_Add and set(e.args) == {foot, x}
    e = foot + 1
    assert e.is_Add and set(e.args) == {foot, 1}


def test_issue_22164():
    warnings.simplefilter("error")
    dm = Quantity("dm")
    SI.set_quantity_dimension(dm, length)
    SI.set_quantity_scale_factor(dm, 1)

    bad_exp = Quantity("bad_exp")
    SI.set_quantity_dimension(bad_exp, length)
    SI.set_quantity_scale_factor(bad_exp, 1)

    expr = dm ** bad_exp

    # deprecation warning is not expected here
    SI._collect_factor_and_dimension(expr)


def test_issue_22819():
    from sympy.physics.units import tonne, gram, Da
    from sympy.physics.units.systems.si import dimsys_SI
    assert tonne.convert_to(gram) == 1000000*gram
    assert dimsys_SI.get_dimensional_dependencies(area) == {length: 2}
    assert Da.scale_factor == 1.66053906660000e-24


def test_issue_20288():
    from sympy.core.numbers import E
    from sympy.physics.units import energy
    u = Quantity('u')
    v = Quantity('v')
    SI.set_quantity_dimension(u, energy)
    SI.set_quantity_dimension(v, energy)
    u.set_global_relative_scale_factor(1, joule)
    v.set_global_relative_scale_factor(1, joule)
    expr = 1 + exp(u**2/v**2)
    assert SI._collect_factor_and_dimension(expr) == (1 + E, Dimension(1))


def test_issue_24062():
    from sympy.core.numbers import E
    from sympy.physics.units import impedance, capacitance, time, ohm, farad, second

    R = Quantity('R')
    C = Quantity('C')
    T = Quantity('T')
    SI.set_quantity_dimension(R, impedance)
    SI.set_quantity_dimension(C, capacitance)
    SI.set_quantity_dimension(T, time)
    R.set_global_relative_scale_factor(1, ohm)
    C.set_global_relative_scale_factor(1, farad)
    T.set_global_relative_scale_factor(1, second)
    expr = T / (R * C)
    dim = SI._collect_factor_and_dimension(expr)[1]
    assert SI.get_dimension_system().is_dimensionless(dim)

    exp_expr = 1 + exp(expr)
    assert SI._collect_factor_and_dimension(exp_expr) == (1 + E, Dimension(1))

def test_issue_24211():
    from sympy.physics.units import time, velocity, acceleration, second, meter
    V1 = Quantity('V1')
    SI.set_quantity_dimension(V1, velocity)
    SI.set_quantity_scale_factor(V1, 1 * meter / second)
    A1 = Quantity('A1')
    SI.set_quantity_dimension(A1, acceleration)
    SI.set_quantity_scale_factor(A1, 1 * meter / second**2)
    T1 = Quantity('T1')
    SI.set_quantity_dimension(T1, time)
    SI.set_quantity_scale_factor(T1, 1 * second)

    expr = A1*T1 + V1
    # should not throw ValueError here
    SI._collect_factor_and_dimension(expr)


def test_prefixed_property():
    assert not meter.is_prefixed
    assert not joule.is_prefixed
    assert not day.is_prefixed
    assert not second.is_prefixed
    assert not volt.is_prefixed
    assert not ohm.is_prefixed
    assert centimeter.is_prefixed
    assert kilometer.is_prefixed
    assert kilogram.is_prefixed
    assert pebibyte.is_prefixed

def test_physics_constant():
    from sympy.physics.units import definitions

    for name in dir(definitions):
        quantity = getattr(definitions, name)
        if not isinstance(quantity, Quantity):
            continue
        if name.endswith('_constant'):
            assert isinstance(quantity, PhysicalConstant), f"{quantity} must be PhysicalConstant, but is {type(quantity)}"
            assert quantity.is_physical_constant, f"{name} is not marked as physics constant when it should be"

    for const in [gravitational_constant, molar_gas_constant, vacuum_permittivity, speed_of_light, elementary_charge]:
        assert isinstance(const, PhysicalConstant), f"{const} must be PhysicalConstant, but is {type(const)}"
        assert const.is_physical_constant, f"{const} is not marked as physics constant when it should be"

    assert not meter.is_physical_constant
    assert not joule.is_physical_constant

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\plot_object.py ===
class PlotObject:
    """
    Base class for objects which can be displayed in
    a Plot.
    """
    visible = True

    def _draw(self):
        if self.visible:
            self.draw()

    def draw(self):
        """
        OpenGL rendering code for the plot object.
        Override in base class.
        """
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\util.py ===
from sympy.core.basic import Basic

new = Basic.__new__


def assoc(d, k, v):
    d = d.copy()
    d[k] = v
    return d


basic_fns = {'op': type,
             'new': Basic.__new__,
             'leaf': lambda x: not isinstance(x, Basic) or x.is_Atom,
             'children': lambda x: x.args}

expr_fns = assoc(basic_fns, 'new', lambda op, *args: op(*args))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_categorical.py ===
import re

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    Index,
    Interval,
    Series,
    Timedelta,
    Timestamp,
    option_context,
)
import pandas._testing as tm


@pytest.fixture
def df():
    return DataFrame(
        {
            "A": np.arange(6, dtype="int64"),
        },
        index=CategoricalIndex(
            list("aabbca"), dtype=CategoricalDtype(list("cab")), name="B"
        ),
    )


@pytest.fixture
def df2():
    return DataFrame(
        {
            "A": np.arange(6, dtype="int64"),
        },
        index=CategoricalIndex(
            list("aabbca"), dtype=CategoricalDtype(list("cabe")), name="B"
        ),
    )


class TestCategoricalIndex:
    def test_loc_scalar(self, df):
        dtype = CategoricalDtype(list("cab"))
        result = df.loc["a"]
        bidx = Series(list("aaa"), name="B").astype(dtype)
        assert bidx.dtype == dtype

        expected = DataFrame({"A": [0, 1, 5]}, index=Index(bidx))
        tm.assert_frame_equal(result, expected)

        df = df.copy()
        df.loc["a"] = 20
        bidx2 = Series(list("aabbca"), name="B").astype(dtype)
        assert bidx2.dtype == dtype
        expected = DataFrame(
            {
                "A": [20, 20, 2, 3, 4, 20],
            },
            index=Index(bidx2),
        )
        tm.assert_frame_equal(df, expected)

        # value not in the categories
        with pytest.raises(KeyError, match=r"^'d'$"):
            df.loc["d"]

        df2 = df.copy()
        expected = df2.copy()
        expected.index = expected.index.astype(object)
        expected.loc["d"] = 10
        df2.loc["d"] = 10
        tm.assert_frame_equal(df2, expected)

    def test_loc_setitem_with_expansion_non_category(self, df):
        # Setting-with-expansion with a new key "d" that is not among caegories
        df.loc["a"] = 20

        # Setting a new row on an existing column
        df3 = df.copy()
        df3.loc["d", "A"] = 10
        bidx3 = Index(list("aabbcad"), name="B")
        expected3 = DataFrame(
            {
                "A": [20, 20, 2, 3, 4, 20, 10.0],
            },
            index=Index(bidx3),
        )
        tm.assert_frame_equal(df3, expected3)

        # Setting a new row _and_ new column
        df4 = df.copy()
        df4.loc["d", "C"] = 10
        expected3 = DataFrame(
            {
                "A": [20, 20, 2, 3, 4, 20, np.nan],
                "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10],
            },
            index=Index(bidx3),
        )
        tm.assert_frame_equal(df4, expected3)

    def test_loc_getitem_scalar_non_category(self, df):
        with pytest.raises(KeyError, match="^1$"):
            df.loc[1]

    def test_slicing(self):
        cat = Series(Categorical([1, 2, 3, 4]))
        reverse = cat[::-1]
        exp = np.array([4, 3, 2, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(reverse.__array__(), exp)

        df = DataFrame({"value": (np.arange(100) + 1).astype("int64")})
        df["D"] = pd.cut(df.value, bins=[0, 25, 50, 75, 100])

        expected = Series([11, Interval(0, 25)], index=["value", "D"], name=10)
        result = df.iloc[10]
        tm.assert_series_equal(result, expected)

        expected = DataFrame(
            {"value": np.arange(11, 21).astype("int64")},
            index=np.arange(10, 20).astype("int64"),
        )
        expected["D"] = pd.cut(expected.value, bins=[0, 25, 50, 75, 100])
        result = df.iloc[10:20]
        tm.assert_frame_equal(result, expected)

        expected = Series([9, Interval(0, 25)], index=["value", "D"], name=8)
        result = df.loc[8]
        tm.assert_series_equal(result, expected)

    def test_slicing_and_getting_ops(self):
        # systematically test the slicing operations:
        #  for all slicing ops:
        #   - returning a dataframe
        #   - returning a column
        #   - returning a row
        #   - returning a single value

        cats = Categorical(
            ["a", "c", "b", "c", "c", "c", "c"], categories=["a", "b", "c"]
        )
        idx = Index(["h", "i", "j", "k", "l", "m", "n"])
        values = [1, 2, 3, 4, 5, 6, 7]
        df = DataFrame({"cats": cats, "values": values}, index=idx)

        # the expected values
        cats2 = Categorical(["b", "c"], categories=["a", "b", "c"])
        idx2 = Index(["j", "k"])
        values2 = [3, 4]

        # 2:4,: | "j":"k",:
        exp_df = DataFrame({"cats": cats2, "values": values2}, index=idx2)

        # :,"cats" | :,0
        exp_col = Series(cats, index=idx, name="cats")

        # "j",: | 2,:
        exp_row = Series(["b", 3], index=["cats", "values"], dtype="object", name="j")

        # "j","cats | 2,0
        exp_val = "b"

        # iloc
        # frame
        res_df = df.iloc[2:4, :]
        tm.assert_frame_equal(res_df, exp_df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

        # row
        res_row = df.iloc[2, :]
        tm.assert_series_equal(res_row, exp_row)
        assert isinstance(res_row["cats"], str)

        # col
        res_col = df.iloc[:, 0]
        tm.assert_series_equal(res_col, exp_col)
        assert isinstance(res_col.dtype, CategoricalDtype)

        # single value
        res_val = df.iloc[2, 0]
        assert res_val == exp_val

        # loc
        # frame
        res_df = df.loc["j":"k", :]
        tm.assert_frame_equal(res_df, exp_df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

        # row
        res_row = df.loc["j", :]
        tm.assert_series_equal(res_row, exp_row)
        assert isinstance(res_row["cats"], str)

        # col
        res_col = df.loc[:, "cats"]
        tm.assert_series_equal(res_col, exp_col)
        assert isinstance(res_col.dtype, CategoricalDtype)

        # single value
        res_val = df.loc["j", "cats"]
        assert res_val == exp_val

        # single value
        res_val = df.loc["j", df.columns[0]]
        assert res_val == exp_val

        # iat
        res_val = df.iat[2, 0]
        assert res_val == exp_val

        # at
        res_val = df.at["j", "cats"]
        assert res_val == exp_val

        # fancy indexing
        exp_fancy = df.iloc[[2]]

        res_fancy = df[df["cats"] == "b"]
        tm.assert_frame_equal(res_fancy, exp_fancy)
        res_fancy = df[df["values"] == 3]
        tm.assert_frame_equal(res_fancy, exp_fancy)

        # get_value
        res_val = df.at["j", "cats"]
        assert res_val == exp_val

        # i : int, slice, or sequence of integers
        res_row = df.iloc[2]
        tm.assert_series_equal(res_row, exp_row)
        assert isinstance(res_row["cats"], str)

        res_df = df.iloc[slice(2, 4)]
        tm.assert_frame_equal(res_df, exp_df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

        res_df = df.iloc[[2, 3]]
        tm.assert_frame_equal(res_df, exp_df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

        res_col = df.iloc[:, 0]
        tm.assert_series_equal(res_col, exp_col)
        assert isinstance(res_col.dtype, CategoricalDtype)

        res_df = df.iloc[:, slice(0, 2)]
        tm.assert_frame_equal(res_df, df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

        res_df = df.iloc[:, [0, 1]]
        tm.assert_frame_equal(res_df, df)
        assert isinstance(res_df["cats"].dtype, CategoricalDtype)

    def test_slicing_doc_examples(self):
        # GH 7918
        cats = Categorical(
            ["a", "b", "b", "b", "c", "c", "c"], categories=["a", "b", "c"]
        )
        idx = Index(["h", "i", "j", "k", "l", "m", "n"])
        values = [1, 2, 2, 2, 3, 4, 5]
        df = DataFrame({"cats": cats, "values": values}, index=idx)

        result = df.iloc[2:4, :]
        expected = DataFrame(
            {
                "cats": Categorical(["b", "b"], categories=["a", "b", "c"]),
                "values": [2, 2],
            },
            index=["j", "k"],
        )
        tm.assert_frame_equal(result, expected)

        result = df.iloc[2:4, :].dtypes
        expected = Series(["category", "int64"], ["cats", "values"], dtype=object)
        tm.assert_series_equal(result, expected)

        result = df.loc["h":"j", "cats"]
        expected = Series(
            Categorical(["a", "b", "b"], categories=["a", "b", "c"]),
            index=["h", "i", "j"],
            name="cats",
        )
        tm.assert_series_equal(result, expected)

        result = df.loc["h":"j", df.columns[0:1]]
        expected = DataFrame(
            {"cats": Categorical(["a", "b", "b"], categories=["a", "b", "c"])},
            index=["h", "i", "j"],
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_listlike_labels(self, df):
        # list of labels
        result = df.loc[["c", "a"]]
        expected = df.iloc[[4, 0, 1, 5]]
        tm.assert_frame_equal(result, expected, check_index_type=True)

    def test_loc_getitem_listlike_unused_category(self, df2):
        # GH#37901 a label that is in index.categories but not in index
        # listlike containing an element in the categories but not in the values
        with pytest.raises(KeyError, match=re.escape("['e'] not in index")):
            df2.loc[["a", "b", "e"]]

    def test_loc_getitem_label_unused_category(self, df2):
        # element in the categories but not in the values
        with pytest.raises(KeyError, match=r"^'e'$"):
            df2.loc["e"]

    def test_loc_getitem_non_category(self, df2):
        # not all labels in the categories
        with pytest.raises(KeyError, match=re.escape("['d'] not in index")):
            df2.loc[["a", "d"]]

    def test_loc_setitem_expansion_label_unused_category(self, df2):
        # assigning with a label that is in the categories but not in the index
        df = df2.copy()
        df.loc["e"] = 20
        result = df.loc[["a", "b", "e"]]
        exp_index = CategoricalIndex(list("aaabbe"), categories=list("cabe"), name="B")
        expected = DataFrame({"A": [0, 1, 5, 2, 3, 20]}, index=exp_index)
        tm.assert_frame_equal(result, expected)

    def test_loc_listlike_dtypes(self):
        # GH 11586

        # unique categories and codes
        index = CategoricalIndex(["a", "b", "c"])
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=index)

        # unique slice
        res = df.loc[["a", "b"]]
        exp_index = CategoricalIndex(["a", "b"], categories=index.categories)
        exp = DataFrame({"A": [1, 2], "B": [4, 5]}, index=exp_index)
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # duplicated slice
        res = df.loc[["a", "a", "b"]]

        exp_index = CategoricalIndex(["a", "a", "b"], categories=index.categories)
        exp = DataFrame({"A": [1, 1, 2], "B": [4, 4, 5]}, index=exp_index)
        tm.assert_frame_equal(res, exp, check_index_type=True)

        with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
            df.loc[["a", "x"]]

    def test_loc_listlike_dtypes_duplicated_categories_and_codes(self):
        # duplicated categories and codes
        index = CategoricalIndex(["a", "b", "a"])
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=index)

        # unique slice
        res = df.loc[["a", "b"]]
        exp = DataFrame(
            {"A": [1, 3, 2], "B": [4, 6, 5]}, index=CategoricalIndex(["a", "a", "b"])
        )
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # duplicated slice
        res = df.loc[["a", "a", "b"]]
        exp = DataFrame(
            {"A": [1, 3, 1, 3, 2], "B": [4, 6, 4, 6, 5]},
            index=CategoricalIndex(["a", "a", "a", "a", "b"]),
        )
        tm.assert_frame_equal(res, exp, check_index_type=True)

        with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
            df.loc[["a", "x"]]

    def test_loc_listlike_dtypes_unused_category(self):
        # contains unused category
        index = CategoricalIndex(["a", "b", "a", "c"], categories=list("abcde"))
        df = DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=index)

        res = df.loc[["a", "b"]]
        exp = DataFrame(
            {"A": [1, 3, 2], "B": [5, 7, 6]},
            index=CategoricalIndex(["a", "a", "b"], categories=list("abcde")),
        )
        tm.assert_frame_equal(res, exp, check_index_type=True)

        # duplicated slice
        res = df.loc[["a", "a", "b"]]
        exp = DataFrame(
            {"A": [1, 3, 1, 3, 2], "B": [5, 7, 5, 7, 6]},
            index=CategoricalIndex(["a", "a", "a", "a", "b"], categories=list("abcde")),
        )
        tm.assert_frame_equal(res, exp, check_index_type=True)

        with pytest.raises(KeyError, match=re.escape("['x'] not in index")):
            df.loc[["a", "x"]]

    def test_loc_getitem_listlike_unused_category_raises_keyerror(self):
        # key that is an *unused* category raises
        index = CategoricalIndex(["a", "b", "a", "c"], categories=list("abcde"))
        df = DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=index)

        with pytest.raises(KeyError, match="e"):
            # For comparison, check the scalar behavior
            df.loc["e"]

        with pytest.raises(KeyError, match=re.escape("['e'] not in index")):
            df.loc[["a", "e"]]

    def test_ix_categorical_index(self):
        # GH 12531
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=list("ABC"),
            columns=list("XYZ"),
        )
        cdf = df.copy()
        cdf.index = CategoricalIndex(df.index)
        cdf.columns = CategoricalIndex(df.columns)

        expect = Series(df.loc["A", :], index=cdf.columns, name="A")
        tm.assert_series_equal(cdf.loc["A", :], expect)

        expect = Series(df.loc[:, "X"], index=cdf.index, name="X")
        tm.assert_series_equal(cdf.loc[:, "X"], expect)

        exp_index = CategoricalIndex(list("AB"), categories=["A", "B", "C"])
        expect = DataFrame(df.loc[["A", "B"], :], columns=cdf.columns, index=exp_index)
        tm.assert_frame_equal(cdf.loc[["A", "B"], :], expect)

        exp_columns = CategoricalIndex(list("XY"), categories=["X", "Y", "Z"])
        expect = DataFrame(df.loc[:, ["X", "Y"]], index=cdf.index, columns=exp_columns)
        tm.assert_frame_equal(cdf.loc[:, ["X", "Y"]], expect)

    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    def test_ix_categorical_index_non_unique(self, infer_string):
        # non-unique
        with option_context("future.infer_string", infer_string):
            df = DataFrame(
                np.random.default_rng(2).standard_normal((3, 3)),
                index=list("ABA"),
                columns=list("XYX"),
            )
            cdf = df.copy()
            cdf.index = CategoricalIndex(df.index)
            cdf.columns = CategoricalIndex(df.columns)

            exp_index = CategoricalIndex(list("AA"), categories=["A", "B"])
            expect = DataFrame(df.loc["A", :], columns=cdf.columns, index=exp_index)
            tm.assert_frame_equal(cdf.loc["A", :], expect)

            exp_columns = CategoricalIndex(list("XX"), categories=["X", "Y"])
            expect = DataFrame(df.loc[:, "X"], index=cdf.index, columns=exp_columns)
            tm.assert_frame_equal(cdf.loc[:, "X"], expect)

            expect = DataFrame(
                df.loc[["A", "B"], :],
                columns=cdf.columns,
                index=CategoricalIndex(list("AAB")),
            )
            tm.assert_frame_equal(cdf.loc[["A", "B"], :], expect)

            expect = DataFrame(
                df.loc[:, ["X", "Y"]],
                index=cdf.index,
                columns=CategoricalIndex(list("XXY")),
            )
            tm.assert_frame_equal(cdf.loc[:, ["X", "Y"]], expect)

    def test_loc_slice(self, df):
        # GH9748
        msg = (
            "cannot do slice indexing on CategoricalIndex with these "
            r"indexers \[1\] of type int"
        )
        with pytest.raises(TypeError, match=msg):
            df.loc[1:5]

        result = df.loc["b":"c"]
        expected = df.iloc[[2, 3, 4]]
        tm.assert_frame_equal(result, expected)

    def test_loc_and_at_with_categorical_index(self):
        # GH 20629
        df = DataFrame(
            [[1, 2], [3, 4], [5, 6]], index=CategoricalIndex(["A", "B", "C"])
        )

        s = df[0]
        assert s.loc["A"] == 1
        assert s.at["A"] == 1

        assert df.loc["B", 1] == 4
        assert df.at["B", 1] == 4

    @pytest.mark.parametrize(
        "idx_values",
        [
            # python types
            [1, 2, 3],
            [-1, -2, -3],
            [1.5, 2.5, 3.5],
            [-1.5, -2.5, -3.5],
            # numpy int/uint
            *(np.array([1, 2, 3], dtype=dtype) for dtype in tm.ALL_INT_NUMPY_DTYPES),
            # numpy floats
            *(np.array([1.5, 2.5, 3.5], dtype=dtyp) for dtyp in tm.FLOAT_NUMPY_DTYPES),
            # numpy object
            np.array([1, "b", 3.5], dtype=object),
            # pandas scalars
            [Interval(1, 4), Interval(4, 6), Interval(6, 9)],
            [Timestamp(2019, 1, 1), Timestamp(2019, 2, 1), Timestamp(2019, 3, 1)],
            [Timedelta(1, "d"), Timedelta(2, "d"), Timedelta(3, "D")],
            # pandas Integer arrays
            *(pd.array([1, 2, 3], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES),
            # other pandas arrays
            pd.IntervalIndex.from_breaks([1, 4, 6, 9]).array,
            pd.date_range("2019-01-01", periods=3).array,
            pd.timedelta_range(start="1d", periods=3).array,
        ],
    )
    def test_loc_getitem_with_non_string_categories(self, idx_values, ordered):
        # GH-17569
        cat_idx = CategoricalIndex(idx_values, ordered=ordered)
        df = DataFrame({"A": ["foo", "bar", "baz"]}, index=cat_idx)
        sl = slice(idx_values[0], idx_values[1])

        # scalar selection
        result = df.loc[idx_values[0]]
        expected = Series(["foo"], index=["A"], name=idx_values[0])
        tm.assert_series_equal(result, expected)

        # list selection
        result = df.loc[idx_values[:2]]
        expected = DataFrame(["foo", "bar"], index=cat_idx[:2], columns=["A"])
        tm.assert_frame_equal(result, expected)

        # slice selection
        result = df.loc[sl]
        expected = DataFrame(["foo", "bar"], index=cat_idx[:2], columns=["A"])
        tm.assert_frame_equal(result, expected)

        # scalar assignment
        result = df.copy()
        result.loc[idx_values[0]] = "qux"
        expected = DataFrame({"A": ["qux", "bar", "baz"]}, index=cat_idx)
        tm.assert_frame_equal(result, expected)

        # list assignment
        result = df.copy()
        result.loc[idx_values[:2], "A"] = ["qux", "qux2"]
        expected = DataFrame({"A": ["qux", "qux2", "baz"]}, index=cat_idx)
        tm.assert_frame_equal(result, expected)

        # slice assignment
        result = df.copy()
        result.loc[sl, "A"] = ["qux", "qux2"]
        expected = DataFrame({"A": ["qux", "qux2", "baz"]}, index=cat_idx)
        tm.assert_frame_equal(result, expected)

    def test_getitem_categorical_with_nan(self):
        # GH#41933
        ci = CategoricalIndex(["A", "B", np.nan])

        ser = Series(range(3), index=ci)

        assert ser[np.nan] == 2
        assert ser.loc[np.nan] == 2

        df = DataFrame(ser)
        assert df.loc[np.nan, 0] == 2
        assert df.loc[np.nan][0] == 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\test_legendre.py ===
"""Tests for legendre module.

"""
from functools import reduce

import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

L0 = np.array([1])
L1 = np.array([0, 1])
L2 = np.array([-1, 0, 3]) / 2
L3 = np.array([0, -3, 0, 5]) / 2
L4 = np.array([3, 0, -30, 0, 35]) / 8
L5 = np.array([0, 15, 0, -70, 0, 63]) / 8
L6 = np.array([-5, 0, 105, 0, -315, 0, 231]) / 16
L7 = np.array([0, -35, 0, 315, 0, -693, 0, 429]) / 16
L8 = np.array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128
L9 = np.array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128

Llist = [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9]


def trim(x):
    return leg.legtrim(x, tol=1e-6)


class TestConstants:

    def test_legdomain(self):
        assert_equal(leg.legdomain, [-1, 1])

    def test_legzero(self):
        assert_equal(leg.legzero, [0])

    def test_legone(self):
        assert_equal(leg.legone, [1])

    def test_legx(self):
        assert_equal(leg.legx, [0, 1])


class TestArithmetic:
    x = np.linspace(-1, 1, 100)

    def test_legadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = leg.legadd([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = leg.legsub([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legmulx(self):
        assert_equal(leg.legmulx([0]), [0])
        assert_equal(leg.legmulx([1]), [0, 1])
        for i in range(1, 5):
            tmp = 2 * i + 1
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp]
            assert_equal(leg.legmulx(ser), tgt)

    def test_legmul(self):
        # check values of result
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = leg.legval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0] * j + [1]
                val2 = leg.legval(self.x, pol2)
                pol3 = leg.legmul(pol1, pol2)
                val3 = leg.legval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_legdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = leg.legadd(ci, cj)
                quo, rem = leg.legdiv(tgt, ci)
                res = leg.legadd(leg.legmul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(leg.legmul, [c] * j, np.array([1]))
                res = leg.legpow(c, j)
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([2., 2., 2.])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1
    y = polyval(x, [1., 2., 3.])

    def test_legval(self):
        # check empty input
        assert_equal(leg.legval([], [1]).size, 0)

        # check normal input)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Llist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = leg.legval(x, [0] * i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = np.zeros(dims)
            assert_equal(leg.legval(x, [1]).shape, dims)
            assert_equal(leg.legval(x, [1, 0]).shape, dims)
            assert_equal(leg.legval(x, [1, 0, 0]).shape, dims)

    def test_legval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises(ValueError, leg.legval2d, x1, x2[:2], self.c2d)

        # test values
        tgt = y1 * y2
        res = leg.legval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = leg.legval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_legval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises(ValueError, leg.legval3d, x1, x2, x3[:2], self.c3d)

        # test values
        tgt = y1 * y2 * y3
        res = leg.legval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = leg.legval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_leggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = leg.leggrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = leg.leggrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3) * 2)

    def test_leggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = leg.leggrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = leg.leggrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3) * 3)


class TestIntegral:

    def test_legint(self):
        # check exceptions
        assert_raises(TypeError, leg.legint, [0], .5)
        assert_raises(ValueError, leg.legint, [0], -1)
        assert_raises(ValueError, leg.legint, [0], 1, [0, 0])
        assert_raises(ValueError, leg.legint, [0], lbnd=[0])
        assert_raises(ValueError, leg.legint, [0], scl=[0])
        assert_raises(TypeError, leg.legint, [0], axis=.5)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = leg.legint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            legpol = leg.poly2leg(pol)
            legint = leg.legint(legpol, m=1, k=[i])
            res = leg.leg2poly(legint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            legpol = leg.poly2leg(pol)
            legint = leg.legint(legpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(leg.legval(-1, legint), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            legpol = leg.poly2leg(pol)
            legint = leg.legint(legpol, m=1, k=[i], scl=2)
            res = leg.leg2poly(legint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = leg.legint(tgt, m=1)
                res = leg.legint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = leg.legint(tgt, m=1, k=[k])
                res = leg.legint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = leg.legint(tgt, m=1, k=[k], lbnd=-1)
                res = leg.legint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = leg.legint(tgt, m=1, k=[k], scl=2)
                res = leg.legint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_legint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([leg.legint(c) for c in c2d.T]).T
        res = leg.legint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([leg.legint(c) for c in c2d])
        res = leg.legint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([leg.legint(c, k=3) for c in c2d])
        res = leg.legint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)

    def test_legint_zerointord(self):
        assert_equal(leg.legint((1, 2, 3), 0), (1, 2, 3))


class TestDerivative:

    def test_legder(self):
        # check exceptions
        assert_raises(TypeError, leg.legder, [0], .5)
        assert_raises(ValueError, leg.legder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = leg.legder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = leg.legder(leg.legint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = leg.legder(leg.legint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_legder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([leg.legder(c) for c in c2d.T]).T
        res = leg.legder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([leg.legder(c) for c in c2d])
        res = leg.legder(c2d, axis=1)
        assert_almost_equal(res, tgt)

    def test_legder_orderhigherthancoeff(self):
        c = (1, 2, 3, 4)
        assert_equal(leg.legder(c, 4), [0])

class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1

    def test_legvander(self):
        # check for 1d x
        x = np.arange(3)
        v = leg.legvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], leg.legval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = leg.legvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], leg.legval(x, coef))

    def test_legvander2d(self):
        # also tests polyval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = leg.legvander2d(x1, x2, [1, 2])
        tgt = leg.legval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = leg.legvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_legvander3d(self):
        # also tests polyval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = leg.legvander3d(x1, x2, x3, [1, 2, 3])
        tgt = leg.legval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = leg.legvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))

    def test_legvander_negdeg(self):
        assert_raises(ValueError, leg.legvander, (1, 2, 3), -1)


class TestFitting:

    def test_legfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        assert_raises(ValueError, leg.legfit, [1], [1], -1)
        assert_raises(TypeError, leg.legfit, [[1]], [1], 0)
        assert_raises(TypeError, leg.legfit, [], [1], 0)
        assert_raises(TypeError, leg.legfit, [1], [[[1]]], 0)
        assert_raises(TypeError, leg.legfit, [1, 2], [1], 0)
        assert_raises(TypeError, leg.legfit, [1], [1, 2], 0)
        assert_raises(TypeError, leg.legfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, leg.legfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, leg.legfit, [1], [1], [-1,])
        assert_raises(ValueError, leg.legfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, leg.legfit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = leg.legfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(leg.legval(x, coef3), y)
        coef3 = leg.legfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(leg.legval(x, coef3), y)
        #
        coef4 = leg.legfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(leg.legval(x, coef4), y)
        coef4 = leg.legfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(leg.legval(x, coef4), y)
        # check things still work if deg is not in strict increasing
        coef4 = leg.legfit(x, y, [2, 3, 4, 1, 0])
        assert_equal(len(coef4), 5)
        assert_almost_equal(leg.legval(x, coef4), y)
        #
        coef2d = leg.legfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = leg.legfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = leg.legfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = leg.legfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = leg.legfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = leg.legfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(leg.legfit(x, x, 1), [0, 1])
        assert_almost_equal(leg.legfit(x, x, [0, 1]), [0, 1])
        # test fitting only even Legendre polynomials
        x = np.linspace(-1, 1)
        y = f2(x)
        coef1 = leg.legfit(x, y, 4)
        assert_almost_equal(leg.legval(x, coef1), y)
        coef2 = leg.legfit(x, y, [0, 2, 4])
        assert_almost_equal(leg.legval(x, coef2), y)
        assert_almost_equal(coef1, coef2)


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, leg.legcompanion, [])
        assert_raises(ValueError, leg.legcompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            assert_(leg.legcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(leg.legcompanion([1, 2])[0, 0] == -.5)


class TestGauss:

    def test_100(self):
        x, w = leg.leggauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = leg.legvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1 / np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # check that the integral of 1 is correct
        tgt = 2.0
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_legfromroots(self):
        res = leg.legfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
            pol = leg.legfromroots(roots)
            res = leg.legval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(leg.leg2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_legroots(self):
        assert_almost_equal(leg.legroots([1]), [])
        assert_almost_equal(leg.legroots([1, 2]), [-.5])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = leg.legroots(leg.legfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_legtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, leg.legtrim, coef, -1)

        # Test results
        assert_equal(leg.legtrim(coef), coef[:-1])
        assert_equal(leg.legtrim(coef, 1), coef[:-3])
        assert_equal(leg.legtrim(coef, 2), [0])

    def test_legline(self):
        assert_equal(leg.legline(3, 4), [3, 4])

    def test_legline_zeroscl(self):
        assert_equal(leg.legline(3, 0), [3])

    def test_leg2poly(self):
        for i in range(10):
            assert_almost_equal(leg.leg2poly([0] * i + [1]), Llist[i])

    def test_poly2leg(self):
        for i in range(10):
            assert_almost_equal(leg.poly2leg(Llist[i]), [0] * i + [1])

    def test_weight(self):
        x = np.linspace(-1, 1, 11)
        tgt = 1.
        res = leg.legweight(x)
        assert_almost_equal(res, tgt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_half.py ===
import platform

import pytest

import numpy as np
from numpy import float16, float32, float64, uint16
from numpy.testing import IS_WASM, assert_, assert_equal


def assert_raises_fpe(strmatch, callable, *args, **kwargs):
    try:
        callable(*args, **kwargs)
    except FloatingPointError as exc:
        assert_(str(exc).find(strmatch) >= 0,
                f"Did not raise floating point {strmatch} error")
    else:
        assert_(False,
                f"Did not raise floating point {strmatch} error")

class TestHalf:
    def setup_method(self):
        # An array of all possible float16 values
        self.all_f16 = np.arange(0x10000, dtype=uint16)
        self.all_f16.dtype = float16

        # NaN value can cause an invalid FP exception if HW is being used
        with np.errstate(invalid='ignore'):
            self.all_f32 = np.array(self.all_f16, dtype=float32)
            self.all_f64 = np.array(self.all_f16, dtype=float64)

        # An array of all non-NaN float16 values, in sorted order
        self.nonan_f16 = np.concatenate(
                                (np.arange(0xfc00, 0x7fff, -1, dtype=uint16),
                                 np.arange(0x0000, 0x7c01, 1, dtype=uint16)))
        self.nonan_f16.dtype = float16
        self.nonan_f32 = np.array(self.nonan_f16, dtype=float32)
        self.nonan_f64 = np.array(self.nonan_f16, dtype=float64)

        # An array of all finite float16 values, in sorted order
        self.finite_f16 = self.nonan_f16[1:-1]
        self.finite_f32 = self.nonan_f32[1:-1]
        self.finite_f64 = self.nonan_f64[1:-1]

    def test_half_conversions(self):
        """Checks that all 16-bit values survive conversion
           to/from 32-bit and 64-bit float"""
        # Because the underlying routines preserve the NaN bits, every
        # value is preserved when converting to/from other floats.

        # Convert from float32 back to float16
        with np.errstate(invalid='ignore'):
            b = np.array(self.all_f32, dtype=float16)
        # avoid testing NaNs due to differing bit patterns in Q/S NaNs
        b_nn = b == b
        assert_equal(self.all_f16[b_nn].view(dtype=uint16),
                     b[b_nn].view(dtype=uint16))

        # Convert from float64 back to float16
        with np.errstate(invalid='ignore'):
            b = np.array(self.all_f64, dtype=float16)
        b_nn = b == b
        assert_equal(self.all_f16[b_nn].view(dtype=uint16),
                     b[b_nn].view(dtype=uint16))

        # Convert float16 to longdouble and back
        # This doesn't necessarily preserve the extra NaN bits,
        # so exclude NaNs.
        a_ld = np.array(self.nonan_f16, dtype=np.longdouble)
        b = np.array(a_ld, dtype=float16)
        assert_equal(self.nonan_f16.view(dtype=uint16),
                     b.view(dtype=uint16))

        # Check the range for which all integers can be represented
        i_int = np.arange(-2048, 2049)
        i_f16 = np.array(i_int, dtype=float16)
        j = np.array(i_f16, dtype=int)
        assert_equal(i_int, j)

    @pytest.mark.parametrize("string_dt", ["S", "U"])
    def test_half_conversion_to_string(self, string_dt):
        # Currently uses S/U32 (which is sufficient for float32)
        expected_dt = np.dtype(f"{string_dt}32")
        assert np.promote_types(np.float16, string_dt) == expected_dt
        assert np.promote_types(string_dt, np.float16) == expected_dt

        arr = np.ones(3, dtype=np.float16).astype(string_dt)
        assert arr.dtype == expected_dt

    @pytest.mark.parametrize("string_dt", ["S", "U"])
    def test_half_conversion_from_string(self, string_dt):
        string = np.array("3.1416", dtype=string_dt)
        assert string.astype(np.float16) == np.array(3.1416, dtype=np.float16)

    @pytest.mark.parametrize("offset", [None, "up", "down"])
    @pytest.mark.parametrize("shift", [None, "up", "down"])
    @pytest.mark.parametrize("float_t", [np.float32, np.float64])
    def test_half_conversion_rounding(self, float_t, shift, offset):
        # Assumes that round to even is used during casting.
        max_pattern = np.float16(np.finfo(np.float16).max).view(np.uint16)

        # Test all (positive) finite numbers, denormals are most interesting
        # however:
        f16s_patterns = np.arange(0, max_pattern + 1, dtype=np.uint16)
        f16s_float = f16s_patterns.view(np.float16).astype(float_t)

        # Shift the values by half a bit up or a down (or do not shift),
        if shift == "up":
            f16s_float = 0.5 * (f16s_float[:-1] + f16s_float[1:])[1:]
        elif shift == "down":
            f16s_float = 0.5 * (f16s_float[:-1] + f16s_float[1:])[:-1]
        else:
            f16s_float = f16s_float[1:-1]

        # Increase the float by a minimal value:
        if offset == "up":
            f16s_float = np.nextafter(f16s_float, float_t(np.inf))
        elif offset == "down":
            f16s_float = np.nextafter(f16s_float, float_t(-np.inf))

        # Convert back to float16 and its bit pattern:
        res_patterns = f16s_float.astype(np.float16).view(np.uint16)

        # The above calculation tries the original values, or the exact
        # midpoints between the float16 values. It then further offsets them
        # by as little as possible. If no offset occurs, "round to even"
        # logic will be necessary, an arbitrarily small offset should cause
        # normal up/down rounding always.

        # Calculate the expected pattern:
        cmp_patterns = f16s_patterns[1:-1].copy()

        if shift == "down" and offset != "up":
            shift_pattern = -1
        elif shift == "up" and offset != "down":
            shift_pattern = 1
        else:
            # There cannot be a shift, either shift is None, so all rounding
            # will go back to original, or shift is reduced by offset too much.
            shift_pattern = 0

        # If rounding occurs, is it normal rounding or round to even?
        if offset is None:
            # Round to even occurs, modify only non-even, cast to allow + (-1)
            cmp_patterns[0::2].view(np.int16)[...] += shift_pattern
        else:
            cmp_patterns.view(np.int16)[...] += shift_pattern

        assert_equal(res_patterns, cmp_patterns)

    @pytest.mark.parametrize(["float_t", "uint_t", "bits"],
                             [(np.float32, np.uint32, 23),
                              (np.float64, np.uint64, 52)])
    def test_half_conversion_denormal_round_even(self, float_t, uint_t, bits):
        # Test specifically that all bits are considered when deciding
        # whether round to even should occur (i.e. no bits are lost at the
        # end. Compare also gh-12721. The most bits can get lost for the
        # smallest denormal:
        smallest_value = np.uint16(1).view(np.float16).astype(float_t)
        assert smallest_value == 2**-24

        # Will be rounded to zero based on round to even rule:
        rounded_to_zero = smallest_value / float_t(2)
        assert rounded_to_zero.astype(np.float16) == 0

        # The significand will be all 0 for the float_t, test that we do not
        # lose the lower ones of these:
        for i in range(bits):
            # slightly increasing the value should make it round up:
            larger_pattern = rounded_to_zero.view(uint_t) | uint_t(1 << i)
            larger_value = larger_pattern.view(float_t)
            assert larger_value.astype(np.float16) == smallest_value

    def test_nans_infs(self):
        with np.errstate(all='ignore'):
            # Check some of the ufuncs
            assert_equal(np.isnan(self.all_f16), np.isnan(self.all_f32))
            assert_equal(np.isinf(self.all_f16), np.isinf(self.all_f32))
            assert_equal(np.isfinite(self.all_f16), np.isfinite(self.all_f32))
            assert_equal(np.signbit(self.all_f16), np.signbit(self.all_f32))
            assert_equal(np.spacing(float16(65504)), np.inf)

            # Check comparisons of all values with NaN
            nan = float16(np.nan)

            assert_(not (self.all_f16 == nan).any())
            assert_(not (nan == self.all_f16).any())

            assert_((self.all_f16 != nan).all())
            assert_((nan != self.all_f16).all())

            assert_(not (self.all_f16 < nan).any())
            assert_(not (nan < self.all_f16).any())

            assert_(not (self.all_f16 <= nan).any())
            assert_(not (nan <= self.all_f16).any())

            assert_(not (self.all_f16 > nan).any())
            assert_(not (nan > self.all_f16).any())

            assert_(not (self.all_f16 >= nan).any())
            assert_(not (nan >= self.all_f16).any())

    def test_half_values(self):
        """Confirms a small number of known half values"""
        a = np.array([1.0, -1.0,
                      2.0, -2.0,
                      0.0999755859375, 0.333251953125,  # 1/10, 1/3
                      65504, -65504,           # Maximum magnitude
                      2.0**(-14), -2.0**(-14),  # Minimum normal
                      2.0**(-24), -2.0**(-24),  # Minimum subnormal
                      0, -1 / 1e1000,            # Signed zeros
                      np.inf, -np.inf])
        b = np.array([0x3c00, 0xbc00,
                      0x4000, 0xc000,
                      0x2e66, 0x3555,
                      0x7bff, 0xfbff,
                      0x0400, 0x8400,
                      0x0001, 0x8001,
                      0x0000, 0x8000,
                      0x7c00, 0xfc00], dtype=uint16)
        b.dtype = float16
        assert_equal(a, b)

    def test_half_rounding(self):
        """Checks that rounding when converting to half is correct"""
        a = np.array([2.0**-25 + 2.0**-35,  # Rounds to minimum subnormal
                      2.0**-25,       # Underflows to zero (nearest even mode)
                      2.0**-26,       # Underflows to zero
                      1.0 + 2.0**-11 + 2.0**-16,  # rounds to 1.0+2**(-10)
                      1.0 + 2.0**-11,   # rounds to 1.0 (nearest even mode)
                      1.0 + 2.0**-12,   # rounds to 1.0
                      65519,          # rounds to 65504
                      65520],         # rounds to inf
                      dtype=float64)
        rounded = [2.0**-24,
                   0.0,
                   0.0,
                   1.0 + 2.0**(-10),
                   1.0,
                   1.0,
                   65504,
                   np.inf]

        # Check float64->float16 rounding
        with np.errstate(over="ignore"):
            b = np.array(a, dtype=float16)
        assert_equal(b, rounded)

        # Check float32->float16 rounding
        a = np.array(a, dtype=float32)
        with np.errstate(over="ignore"):
            b = np.array(a, dtype=float16)
        assert_equal(b, rounded)

    def test_half_correctness(self):
        """Take every finite float16, and check the casting functions with
           a manual conversion."""

        # Create an array of all finite float16s
        a_bits = self.finite_f16.view(dtype=uint16)

        # Convert to 64-bit float manually
        a_sgn = (-1.0)**((a_bits & 0x8000) >> 15)
        a_exp = np.array((a_bits & 0x7c00) >> 10, dtype=np.int32) - 15
        a_man = (a_bits & 0x03ff) * 2.0**(-10)
        # Implicit bit of normalized floats
        a_man[a_exp != -15] += 1
        # Denormalized exponent is -14
        a_exp[a_exp == -15] = -14

        a_manual = a_sgn * a_man * 2.0**a_exp

        a32_fail = np.nonzero(self.finite_f32 != a_manual)[0]
        if len(a32_fail) != 0:
            bad_index = a32_fail[0]
            assert_equal(self.finite_f32, a_manual,
                 "First non-equal is half value 0x%x -> %g != %g" %
                            (a_bits[bad_index],
                             self.finite_f32[bad_index],
                             a_manual[bad_index]))

        a64_fail = np.nonzero(self.finite_f64 != a_manual)[0]
        if len(a64_fail) != 0:
            bad_index = a64_fail[0]
            assert_equal(self.finite_f64, a_manual,
                 "First non-equal is half value 0x%x -> %g != %g" %
                            (a_bits[bad_index],
                             self.finite_f64[bad_index],
                             a_manual[bad_index]))

    def test_half_ordering(self):
        """Make sure comparisons are working right"""

        # All non-NaN float16 values in reverse order
        a = self.nonan_f16[::-1].copy()

        # 32-bit float copy
        b = np.array(a, dtype=float32)

        # Should sort the same
        a.sort()
        b.sort()
        assert_equal(a, b)

        # Comparisons should work
        assert_((a[:-1] <= a[1:]).all())
        assert_(not (a[:-1] > a[1:]).any())
        assert_((a[1:] >= a[:-1]).all())
        assert_(not (a[1:] < a[:-1]).any())
        # All != except for +/-0
        assert_equal(np.nonzero(a[:-1] < a[1:])[0].size, a.size - 2)
        assert_equal(np.nonzero(a[1:] > a[:-1])[0].size, a.size - 2)

    def test_half_funcs(self):
        """Test the various ArrFuncs"""

        # fill
        assert_equal(np.arange(10, dtype=float16),
                     np.arange(10, dtype=float32))

        # fillwithscalar
        a = np.zeros((5,), dtype=float16)
        a.fill(1)
        assert_equal(a, np.ones((5,), dtype=float16))

        # nonzero and copyswap
        a = np.array([0, 0, -1, -1 / 1e20, 0, 2.0**-24, 7.629e-6], dtype=float16)
        assert_equal(a.nonzero()[0],
                     [2, 5, 6])
        a = a.byteswap()
        a = a.view(a.dtype.newbyteorder())
        assert_equal(a.nonzero()[0],
                     [2, 5, 6])

        # dot
        a = np.arange(0, 10, 0.5, dtype=float16)
        b = np.ones((20,), dtype=float16)
        assert_equal(np.dot(a, b),
                     95)

        # argmax
        a = np.array([0, -np.inf, -2, 0.5, 12.55, 7.3, 2.1, 12.4], dtype=float16)
        assert_equal(a.argmax(),
                     4)
        a = np.array([0, -np.inf, -2, np.inf, 12.55, np.nan, 2.1, 12.4], dtype=float16)
        assert_equal(a.argmax(),
                     5)

        # getitem
        a = np.arange(10, dtype=float16)
        for i in range(10):
            assert_equal(a.item(i), i)

    def test_spacing_nextafter(self):
        """Test np.spacing and np.nextafter"""
        # All non-negative finite #'s
        a = np.arange(0x7c00, dtype=uint16)
        hinf = np.array((np.inf,), dtype=float16)
        hnan = np.array((np.nan,), dtype=float16)
        a_f16 = a.view(dtype=float16)

        assert_equal(np.spacing(a_f16[:-1]), a_f16[1:] - a_f16[:-1])

        assert_equal(np.nextafter(a_f16[:-1], hinf), a_f16[1:])
        assert_equal(np.nextafter(a_f16[0], -hinf), -a_f16[1])
        assert_equal(np.nextafter(a_f16[1:], -hinf), a_f16[:-1])

        assert_equal(np.nextafter(hinf, a_f16), a_f16[-1])
        assert_equal(np.nextafter(-hinf, a_f16), -a_f16[-1])

        assert_equal(np.nextafter(hinf, hinf), hinf)
        assert_equal(np.nextafter(hinf, -hinf), a_f16[-1])
        assert_equal(np.nextafter(-hinf, hinf), -a_f16[-1])
        assert_equal(np.nextafter(-hinf, -hinf), -hinf)

        assert_equal(np.nextafter(a_f16, hnan), hnan[0])
        assert_equal(np.nextafter(hnan, a_f16), hnan[0])

        assert_equal(np.nextafter(hnan, hnan), hnan)
        assert_equal(np.nextafter(hinf, hnan), hnan)
        assert_equal(np.nextafter(hnan, hinf), hnan)

        # switch to negatives
        a |= 0x8000

        assert_equal(np.spacing(a_f16[0]), np.spacing(a_f16[1]))
        assert_equal(np.spacing(a_f16[1:]), a_f16[:-1] - a_f16[1:])

        assert_equal(np.nextafter(a_f16[0], hinf), -a_f16[1])
        assert_equal(np.nextafter(a_f16[1:], hinf), a_f16[:-1])
        assert_equal(np.nextafter(a_f16[:-1], -hinf), a_f16[1:])

        assert_equal(np.nextafter(hinf, a_f16), -a_f16[-1])
        assert_equal(np.nextafter(-hinf, a_f16), a_f16[-1])

        assert_equal(np.nextafter(a_f16, hnan), hnan[0])
        assert_equal(np.nextafter(hnan, a_f16), hnan[0])

    def test_half_ufuncs(self):
        """Test the various ufuncs"""

        a = np.array([0, 1, 2, 4, 2], dtype=float16)
        b = np.array([-2, 5, 1, 4, 3], dtype=float16)
        c = np.array([0, -1, -np.inf, np.nan, 6], dtype=float16)

        assert_equal(np.add(a, b), [-2, 6, 3, 8, 5])
        assert_equal(np.subtract(a, b), [2, -4, 1, 0, -1])
        assert_equal(np.multiply(a, b), [0, 5, 2, 16, 6])
        assert_equal(np.divide(a, b), [0, 0.199951171875, 2, 1, 0.66650390625])

        assert_equal(np.equal(a, b), [False, False, False, True, False])
        assert_equal(np.not_equal(a, b), [True, True, True, False, True])
        assert_equal(np.less(a, b), [False, True, False, False, True])
        assert_equal(np.less_equal(a, b), [False, True, False, True, True])
        assert_equal(np.greater(a, b), [True, False, True, False, False])
        assert_equal(np.greater_equal(a, b), [True, False, True, True, False])
        assert_equal(np.logical_and(a, b), [False, True, True, True, True])
        assert_equal(np.logical_or(a, b), [True, True, True, True, True])
        assert_equal(np.logical_xor(a, b), [True, False, False, False, False])
        assert_equal(np.logical_not(a), [True, False, False, False, False])

        assert_equal(np.isnan(c), [False, False, False, True, False])
        assert_equal(np.isinf(c), [False, False, True, False, False])
        assert_equal(np.isfinite(c), [True, True, False, False, True])
        assert_equal(np.signbit(b), [True, False, False, False, False])

        assert_equal(np.copysign(b, a), [2, 5, 1, 4, 3])

        assert_equal(np.maximum(a, b), [0, 5, 2, 4, 3])

        x = np.maximum(b, c)
        assert_(np.isnan(x[3]))
        x[3] = 0
        assert_equal(x, [0, 5, 1, 0, 6])

        assert_equal(np.minimum(a, b), [-2, 1, 1, 4, 2])

        x = np.minimum(b, c)
        assert_(np.isnan(x[3]))
        x[3] = 0
        assert_equal(x, [-2, -1, -np.inf, 0, 3])

        assert_equal(np.fmax(a, b), [0, 5, 2, 4, 3])
        assert_equal(np.fmax(b, c), [0, 5, 1, 4, 6])
        assert_equal(np.fmin(a, b), [-2, 1, 1, 4, 2])
        assert_equal(np.fmin(b, c), [-2, -1, -np.inf, 4, 3])

        assert_equal(np.floor_divide(a, b), [0, 0, 2, 1, 0])
        assert_equal(np.remainder(a, b), [0, 1, 0, 0, 2])
        assert_equal(np.divmod(a, b), ([0, 0, 2, 1, 0], [0, 1, 0, 0, 2]))
        assert_equal(np.square(b), [4, 25, 1, 16, 9])
        assert_equal(np.reciprocal(b), [-0.5, 0.199951171875, 1, 0.25, 0.333251953125])
        assert_equal(np.ones_like(b), [1, 1, 1, 1, 1])
        assert_equal(np.conjugate(b), b)
        assert_equal(np.absolute(b), [2, 5, 1, 4, 3])
        assert_equal(np.negative(b), [2, -5, -1, -4, -3])
        assert_equal(np.positive(b), b)
        assert_equal(np.sign(b), [-1, 1, 1, 1, 1])
        assert_equal(np.modf(b), ([0, 0, 0, 0, 0], b))
        assert_equal(np.frexp(b), ([-0.5, 0.625, 0.5, 0.5, 0.75], [2, 3, 1, 3, 2]))
        assert_equal(np.ldexp(b, [0, 1, 2, 4, 2]), [-2, 10, 4, 64, 12])

    def test_half_coercion(self):
        """Test that half gets coerced properly with the other types"""
        a16 = np.array((1,), dtype=float16)
        a32 = np.array((1,), dtype=float32)
        b16 = float16(1)
        b32 = float32(1)

        assert np.power(a16, 2).dtype == float16
        assert np.power(a16, 2.0).dtype == float16
        assert np.power(a16, b16).dtype == float16
        assert np.power(a16, b32).dtype == float32
        assert np.power(a16, a16).dtype == float16
        assert np.power(a16, a32).dtype == float32

        assert np.power(b16, 2).dtype == float16
        assert np.power(b16, 2.0).dtype == float16
        assert np.power(b16, b16).dtype, float16
        assert np.power(b16, b32).dtype, float32
        assert np.power(b16, a16).dtype, float16
        assert np.power(b16, a32).dtype, float32

        assert np.power(a32, a16).dtype == float32
        assert np.power(a32, b16).dtype == float32
        assert np.power(b32, a16).dtype == float32
        assert np.power(b32, b16).dtype == float32

    @pytest.mark.skipif(platform.machine() == "armv5tel",
                        reason="See gh-413.")
    @pytest.mark.skipif(IS_WASM,
                        reason="fp exceptions don't work in wasm.")
    def test_half_fpe(self):
        with np.errstate(all='raise'):
            sx16 = np.array((1e-4,), dtype=float16)
            bx16 = np.array((1e4,), dtype=float16)
            sy16 = float16(1e-4)
            by16 = float16(1e4)

            # Underflow errors
            assert_raises_fpe('underflow', lambda a, b: a * b, sx16, sx16)
            assert_raises_fpe('underflow', lambda a, b: a * b, sx16, sy16)
            assert_raises_fpe('underflow', lambda a, b: a * b, sy16, sx16)
            assert_raises_fpe('underflow', lambda a, b: a * b, sy16, sy16)
            assert_raises_fpe('underflow', lambda a, b: a / b, sx16, bx16)
            assert_raises_fpe('underflow', lambda a, b: a / b, sx16, by16)
            assert_raises_fpe('underflow', lambda a, b: a / b, sy16, bx16)
            assert_raises_fpe('underflow', lambda a, b: a / b, sy16, by16)
            assert_raises_fpe('underflow', lambda a, b: a / b,
                                             float16(2.**-14), float16(2**11))
            assert_raises_fpe('underflow', lambda a, b: a / b,
                                             float16(-2.**-14), float16(2**11))
            assert_raises_fpe('underflow', lambda a, b: a / b,
                                             float16(2.**-14 + 2**-24), float16(2))
            assert_raises_fpe('underflow', lambda a, b: a / b,
                                             float16(-2.**-14 - 2**-24), float16(2))
            assert_raises_fpe('underflow', lambda a, b: a / b,
                                             float16(2.**-14 + 2**-23), float16(4))

            # Overflow errors
            assert_raises_fpe('overflow', lambda a, b: a * b, bx16, bx16)
            assert_raises_fpe('overflow', lambda a, b: a * b, bx16, by16)
            assert_raises_fpe('overflow', lambda a, b: a * b, by16, bx16)
            assert_raises_fpe('overflow', lambda a, b: a * b, by16, by16)
            assert_raises_fpe('overflow', lambda a, b: a / b, bx16, sx16)
            assert_raises_fpe('overflow', lambda a, b: a / b, bx16, sy16)
            assert_raises_fpe('overflow', lambda a, b: a / b, by16, sx16)
            assert_raises_fpe('overflow', lambda a, b: a / b, by16, sy16)
            assert_raises_fpe('overflow', lambda a, b: a + b,
                                             float16(65504), float16(17))
            assert_raises_fpe('overflow', lambda a, b: a - b,
                                             float16(-65504), float16(17))
            assert_raises_fpe('overflow', np.nextafter, float16(65504), float16(np.inf))
            assert_raises_fpe('overflow', np.nextafter, float16(-65504), float16(-np.inf))
            assert_raises_fpe('overflow', np.spacing, float16(65504))

            # Invalid value errors
            assert_raises_fpe('invalid', np.divide, float16(np.inf), float16(np.inf))
            assert_raises_fpe('invalid', np.spacing, float16(np.inf))
            assert_raises_fpe('invalid', np.spacing, float16(np.nan))

            # These should not raise
            float16(65472) + float16(32)
            float16(2**-13) / float16(2)
            float16(2**-14) / float16(2**10)
            np.spacing(float16(-65504))
            np.nextafter(float16(65504), float16(-np.inf))
            np.nextafter(float16(-65504), float16(np.inf))
            np.nextafter(float16(np.inf), float16(0))
            np.nextafter(float16(-np.inf), float16(0))
            np.nextafter(float16(0), float16(np.nan))
            np.nextafter(float16(np.nan), float16(0))
            float16(2**-14) / float16(2**10)
            float16(-2**-14) / float16(2**10)
            float16(2**-14 + 2**-23) / float16(2)
            float16(-2**-14 - 2**-23) / float16(2)

    def test_half_array_interface(self):
        """Test that half is compatible with __array_interface__"""
        class Dummy:
            pass

        a = np.ones((1,), dtype=float16)
        b = Dummy()
        b.__array_interface__ = a.__array_interface__
        c = np.array(b)
        assert_(c.dtype == float16)
        assert_equal(a, c)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\series\test_logical_ops.py ===
from datetime import datetime
import operator

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas import (
    ArrowDtype,
    DataFrame,
    Index,
    Series,
    StringDtype,
    bdate_range,
)
import pandas._testing as tm
from pandas.core import ops


class TestSeriesLogicalOps:
    @pytest.mark.filterwarnings("ignore:Downcasting object dtype arrays:FutureWarning")
    @pytest.mark.parametrize("bool_op", [operator.and_, operator.or_, operator.xor])
    def test_bool_operators_with_nas(self, bool_op):
        # boolean &, |, ^ should work with object arrays and propagate NAs
        ser = Series(bdate_range("1/1/2000", periods=10), dtype=object)
        ser[::2] = np.nan

        mask = ser.isna()
        filled = ser.fillna(ser[0])

        result = bool_op(ser < ser[9], ser > ser[3])

        expected = bool_op(filled < filled[9], filled > filled[3])
        expected[mask] = False
        tm.assert_series_equal(result, expected)

    def test_logical_operators_bool_dtype_with_empty(self):
        # GH#9016: support bitwise op for integer types
        index = list("bca")

        s_tft = Series([True, False, True], index=index)
        s_fff = Series([False, False, False], index=index)
        s_empty = Series([], dtype=object)

        res = s_tft & s_empty
        expected = s_fff.sort_index()
        tm.assert_series_equal(res, expected)

        res = s_tft | s_empty
        expected = s_tft.sort_index()
        tm.assert_series_equal(res, expected)

    def test_logical_operators_int_dtype_with_int_dtype(self):
        # GH#9016: support bitwise op for integer types

        s_0123 = Series(range(4), dtype="int64")
        s_3333 = Series([3] * 4)
        s_4444 = Series([4] * 4)

        res = s_0123 & s_3333
        expected = Series(range(4), dtype="int64")
        tm.assert_series_equal(res, expected)

        res = s_0123 | s_4444
        expected = Series(range(4, 8), dtype="int64")
        tm.assert_series_equal(res, expected)

        s_1111 = Series([1] * 4, dtype="int8")
        res = s_0123 & s_1111
        expected = Series([0, 1, 0, 1], dtype="int64")
        tm.assert_series_equal(res, expected)

        res = s_0123.astype(np.int16) | s_1111.astype(np.int32)
        expected = Series([1, 1, 3, 3], dtype="int32")
        tm.assert_series_equal(res, expected)

    def test_logical_operators_int_dtype_with_int_scalar(self):
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        res = s_0123 & 0
        expected = Series([0] * 4)
        tm.assert_series_equal(res, expected)

        res = s_0123 & 1
        expected = Series([0, 1, 0, 1])
        tm.assert_series_equal(res, expected)

    def test_logical_operators_int_dtype_with_float(self):
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        warn_msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        with pytest.raises(TypeError, match=msg):
            s_0123 & np.nan
        with pytest.raises(TypeError, match=msg):
            s_0123 & 3.14
        msg = "unsupported operand type.+for &:"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                s_0123 & [0.1, 4, 3.14, 2]
        with pytest.raises(TypeError, match=msg):
            s_0123 & np.array([0.1, 4, 3.14, 2])
        with pytest.raises(TypeError, match=msg):
            s_0123 & Series([0.1, 4, -3.14, 2])

    def test_logical_operators_int_dtype_with_str(self):
        s_1111 = Series([1] * 4, dtype="int8")

        warn_msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        msg = "Cannot perform 'and_' with a dtyped.+array and scalar of type"
        with pytest.raises(TypeError, match=msg):
            s_1111 & "a"
        with pytest.raises(TypeError, match="unsupported operand.+for &"):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                s_1111 & ["a", "b", "c", "d"]

    def test_logical_operators_int_dtype_with_bool(self):
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        expected = Series([False] * 4)

        result = s_0123 & False
        tm.assert_series_equal(result, expected)

        warn_msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            result = s_0123 & [False]
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            result = s_0123 & (False,)
        tm.assert_series_equal(result, expected)

        result = s_0123 ^ False
        expected = Series([False, True, True, True])
        tm.assert_series_equal(result, expected)

    def test_logical_operators_int_dtype_with_object(self):
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        result = s_0123 & Series([False, np.nan, False, False])
        expected = Series([False] * 4)
        tm.assert_series_equal(result, expected)

        s_abNd = Series(["a", "b", np.nan, "d"])
        with pytest.raises(
            TypeError, match="unsupported.* 'int' and 'str'|'rand_' not supported"
        ):
            s_0123 & s_abNd

    def test_logical_operators_bool_dtype_with_int(self):
        index = list("bca")

        s_tft = Series([True, False, True], index=index)
        s_fff = Series([False, False, False], index=index)

        res = s_tft & 0
        expected = s_fff
        tm.assert_series_equal(res, expected)

        res = s_tft & 1
        expected = s_tft
        tm.assert_series_equal(res, expected)

    def test_logical_ops_bool_dtype_with_ndarray(self):
        # make sure we operate on ndarray the same as Series
        left = Series([True, True, True, False, True])
        right = [True, False, None, True, np.nan]

        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        expected = Series([True, False, False, False, False])
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = left & right
        tm.assert_series_equal(result, expected)
        result = left & np.array(right)
        tm.assert_series_equal(result, expected)
        result = left & Index(right)
        tm.assert_series_equal(result, expected)
        result = left & Series(right)
        tm.assert_series_equal(result, expected)

        expected = Series([True, True, True, True, True])
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = left | right
        tm.assert_series_equal(result, expected)
        result = left | np.array(right)
        tm.assert_series_equal(result, expected)
        result = left | Index(right)
        tm.assert_series_equal(result, expected)
        result = left | Series(right)
        tm.assert_series_equal(result, expected)

        expected = Series([False, True, True, True, True])
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = left ^ right
        tm.assert_series_equal(result, expected)
        result = left ^ np.array(right)
        tm.assert_series_equal(result, expected)
        result = left ^ Index(right)
        tm.assert_series_equal(result, expected)
        result = left ^ Series(right)
        tm.assert_series_equal(result, expected)

    def test_logical_operators_int_dtype_with_bool_dtype_and_reindex(self):
        # GH#9016: support bitwise op for integer types

        index = list("bca")

        s_tft = Series([True, False, True], index=index)
        s_tft = Series([True, False, True], index=index)
        s_tff = Series([True, False, False], index=index)

        s_0123 = Series(range(4), dtype="int64")

        # s_0123 will be all false now because of reindexing like s_tft
        expected = Series([False] * 7, index=[0, 1, 2, 3, "a", "b", "c"])
        with tm.assert_produces_warning(FutureWarning):
            result = s_tft & s_0123
        tm.assert_series_equal(result, expected)

        # GH 52538: Deprecate casting to object type when reindex is needed;
        # matches DataFrame behavior
        expected = Series([False] * 7, index=[0, 1, 2, 3, "a", "b", "c"])
        with tm.assert_produces_warning(FutureWarning):
            result = s_0123 & s_tft
        tm.assert_series_equal(result, expected)

        s_a0b1c0 = Series([1], list("b"))

        with tm.assert_produces_warning(FutureWarning):
            res = s_tft & s_a0b1c0
        expected = s_tff.reindex(list("abc"))
        tm.assert_series_equal(res, expected)

        with tm.assert_produces_warning(FutureWarning):
            res = s_tft | s_a0b1c0
        expected = s_tft.reindex(list("abc"))
        tm.assert_series_equal(res, expected)

    def test_scalar_na_logical_ops_corners(self):
        s = Series([2, 3, 4, 5, 6, 7, 8, 9, 10])

        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        with pytest.raises(TypeError, match=msg):
            s & datetime(2005, 1, 1)

        s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
        s[::2] = np.nan

        expected = Series(True, index=s.index)
        expected[::2] = False

        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s & list(s)
        tm.assert_series_equal(result, expected)

    def test_scalar_na_logical_ops_corners_aligns(self):
        s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
        s[::2] = np.nan
        d = DataFrame({"A": s})

        expected = DataFrame(False, index=range(9), columns=["A"] + list(range(9)))

        result = s & d
        tm.assert_frame_equal(result, expected)

        result = d & s
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
    def test_logical_ops_with_index(self, op):
        # GH#22092, GH#19792
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False])
        idx2 = Index([1, 0, 1, 0])

        expected = Series([op(ser[n], idx1[n]) for n in range(len(ser))])

        result = op(ser, idx1)
        tm.assert_series_equal(result, expected)

        expected = Series([op(ser[n], idx2[n]) for n in range(len(ser))], dtype=bool)

        result = op(ser, idx2)
        tm.assert_series_equal(result, expected)

    def test_reversed_xor_with_index_returns_series(self):
        # GH#22092, GH#19792 pre-2.0 these were aliased to setops
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False], dtype=bool)
        idx2 = Index([1, 0, 1, 0])

        expected = Series([False, True, True, False])
        result = idx1 ^ ser
        tm.assert_series_equal(result, expected)

        result = idx2 ^ ser
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "op",
        [
            ops.rand_,
            ops.ror_,
        ],
    )
    def test_reversed_logical_op_with_index_returns_series(self, op):
        # GH#22092, GH#19792
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False])
        idx2 = Index([1, 0, 1, 0])

        expected = Series(op(idx1.values, ser.values))
        result = op(ser, idx1)
        tm.assert_series_equal(result, expected)

        expected = op(ser, Series(idx2))
        result = op(ser, idx2)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "op, expected",
        [
            (ops.rand_, Series([False, False])),
            (ops.ror_, Series([True, True])),
            (ops.rxor, Series([True, True])),
        ],
    )
    def test_reverse_ops_with_index(self, op, expected):
        # https://github.com/pandas-dev/pandas/pull/23628
        # multi-set Index ops are buggy, so let's avoid duplicates...
        # GH#49503
        ser = Series([True, False])
        idx = Index([False, True])

        result = op(ser, idx)
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
    def test_logical_ops_label_based(self, using_infer_string):
        # GH#4947
        # logical ops should be label based

        a = Series([True, False, True], list("bca"))
        b = Series([False, True, False], list("abc"))

        expected = Series([False, True, False], list("abc"))
        result = a & b
        tm.assert_series_equal(result, expected)

        expected = Series([True, True, False], list("abc"))
        result = a | b
        tm.assert_series_equal(result, expected)

        expected = Series([True, False, False], list("abc"))
        result = a ^ b
        tm.assert_series_equal(result, expected)

        # rhs is bigger
        a = Series([True, False, True], list("bca"))
        b = Series([False, True, False, True], list("abcd"))

        expected = Series([False, True, False, False], list("abcd"))
        result = a & b
        tm.assert_series_equal(result, expected)

        expected = Series([True, True, False, False], list("abcd"))
        result = a | b
        tm.assert_series_equal(result, expected)

        # filling

        # vs empty
        empty = Series([], dtype=object)

        result = a & empty.copy()
        expected = Series([False, False, False], list("abc"))
        tm.assert_series_equal(result, expected)

        result = a | empty.copy()
        expected = Series([True, True, False], list("abc"))
        tm.assert_series_equal(result, expected)

        # vs non-matching
        with tm.assert_produces_warning(FutureWarning):
            result = a & Series([1], ["z"])
        expected = Series([False, False, False, False], list("abcz"))
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning):
            result = a | Series([1], ["z"])
        expected = Series([True, True, False, False], list("abcz"))
        tm.assert_series_equal(result, expected)

        # identity
        # we would like s[s|e] == s to hold for any e, whether empty or not
        with tm.assert_produces_warning(FutureWarning):
            for e in [
                empty.copy(),
                Series([1], ["z"]),
                Series(np.nan, b.index),
                Series(np.nan, a.index),
            ]:
                result = a[a | e]
                tm.assert_series_equal(result, a[a])

        for e in [Series(["z"])]:
            if using_infer_string:
                # TODO(infer_string) should this behave differently?
                # -> https://github.com/pandas-dev/pandas/issues/60234
                with pytest.raises(
                    TypeError, match="not supported for dtype|unsupported operand type"
                ):
                    result = a[a | e]
            else:
                result = a[a | e]
            tm.assert_series_equal(result, a[a])

        # vs scalars
        index = list("bca")
        t = Series([True, False, True])

        for v in [True, 1, 2]:
            result = Series([True, False, True], index=index) | v
            expected = Series([True, True, True], index=index)
            tm.assert_series_equal(result, expected)

        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        for v in [np.nan, "foo"]:
            with pytest.raises(TypeError, match=msg):
                t | v

        for v in [False, 0]:
            result = Series([True, False, True], index=index) | v
            expected = Series([True, False, True], index=index)
            tm.assert_series_equal(result, expected)

        for v in [True, 1]:
            result = Series([True, False, True], index=index) & v
            expected = Series([True, False, True], index=index)
            tm.assert_series_equal(result, expected)

        for v in [False, 0]:
            result = Series([True, False, True], index=index) & v
            expected = Series([False, False, False], index=index)
            tm.assert_series_equal(result, expected)
        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        for v in [np.nan]:
            with pytest.raises(TypeError, match=msg):
                t & v

    def test_logical_ops_df_compat(self):
        # GH#1134
        s1 = Series([True, False, True], index=list("ABC"), name="x")
        s2 = Series([True, True, False], index=list("ABD"), name="x")

        exp = Series([True, False, False, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s1 & s2, exp)
        tm.assert_series_equal(s2 & s1, exp)

        # True | np.nan => True
        exp_or1 = Series([True, True, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s1 | s2, exp_or1)
        # np.nan | True => np.nan, filled with False
        exp_or = Series([True, True, False, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s2 | s1, exp_or)

        # DataFrame doesn't fill nan with False
        tm.assert_frame_equal(s1.to_frame() & s2.to_frame(), exp.to_frame())
        tm.assert_frame_equal(s2.to_frame() & s1.to_frame(), exp.to_frame())

        exp = DataFrame({"x": [True, True, np.nan, np.nan]}, index=list("ABCD"))
        tm.assert_frame_equal(s1.to_frame() | s2.to_frame(), exp_or1.to_frame())
        tm.assert_frame_equal(s2.to_frame() | s1.to_frame(), exp_or.to_frame())

        # different length
        s3 = Series([True, False, True], index=list("ABC"), name="x")
        s4 = Series([True, True, True, True], index=list("ABCD"), name="x")

        exp = Series([True, False, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s3 & s4, exp)
        tm.assert_series_equal(s4 & s3, exp)

        # np.nan | True => np.nan, filled with False
        exp_or1 = Series([True, True, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s3 | s4, exp_or1)
        # True | np.nan => True
        exp_or = Series([True, True, True, True], index=list("ABCD"), name="x")
        tm.assert_series_equal(s4 | s3, exp_or)

        tm.assert_frame_equal(s3.to_frame() & s4.to_frame(), exp.to_frame())
        tm.assert_frame_equal(s4.to_frame() & s3.to_frame(), exp.to_frame())

        tm.assert_frame_equal(s3.to_frame() | s4.to_frame(), exp_or1.to_frame())
        tm.assert_frame_equal(s4.to_frame() | s3.to_frame(), exp_or.to_frame())

    @pytest.mark.xfail(reason="Will pass once #52839 deprecation is enforced")
    def test_int_dtype_different_index_not_bool(self):
        # GH 52500
        ser1 = Series([1, 2, 3], index=[10, 11, 23], name="a")
        ser2 = Series([10, 20, 30], index=[11, 10, 23], name="a")
        result = np.bitwise_xor(ser1, ser2)
        expected = Series([21, 8, 29], index=[10, 11, 23], name="a")
        tm.assert_series_equal(result, expected)

        result = ser1 ^ ser2
        tm.assert_series_equal(result, expected)

    # TODO: this belongs in comparison tests
    def test_pyarrow_numpy_string_invalid(self):
        # GH#56008
        pa = pytest.importorskip("pyarrow")
        ser = Series([False, True])
        ser2 = Series(["a", "b"], dtype=StringDtype(na_value=np.nan))
        result = ser == ser2
        expected_eq = Series(False, index=ser.index)
        tm.assert_series_equal(result, expected_eq)

        result = ser != ser2
        expected_ne = Series(True, index=ser.index)
        tm.assert_series_equal(result, expected_ne)

        with pytest.raises(TypeError, match="Invalid comparison"):
            ser > ser2

        # GH#59505
        ser3 = ser2.astype("string[pyarrow]")
        result3_eq = ser3 == ser
        tm.assert_series_equal(result3_eq, expected_eq.astype("bool[pyarrow]"))
        result3_ne = ser3 != ser
        tm.assert_series_equal(result3_ne, expected_ne.astype("bool[pyarrow]"))

        with pytest.raises(TypeError, match="Invalid comparison"):
            ser > ser3

        ser4 = ser2.astype(ArrowDtype(pa.string()))
        result4_eq = ser4 == ser
        tm.assert_series_equal(result4_eq, expected_eq.astype("bool[pyarrow]"))
        result4_ne = ser4 != ser
        tm.assert_series_equal(result4_ne, expected_ne.astype("bool[pyarrow]"))

        with pytest.raises(TypeError, match="Invalid comparison"):
            ser > ser4

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_python_parser_only.py ===
"""
Tests that apply specifically to the Python parser. Unless specifically
stated as a Python-specific issue, the goal is to eventually move as many of
these tests out of this module as soon as the C parser can accept further
arguments when parsing.
"""
from __future__ import annotations

import csv
from io import (
    BytesIO,
    StringIO,
    TextIOWrapper,
)
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandas.errors import (
    ParserError,
    ParserWarning,
)

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_default_separator(python_parser_only):
    # see gh-17333
    #
    # csv.Sniffer in Python treats "o" as separator.
    data = "aob\n1o2\n3o4"
    parser = python_parser_only
    expected = DataFrame({"a": [1, 3], "b": [2, 4]})

    result = parser.read_csv(StringIO(data), sep=None)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("skipfooter", ["foo", 1.5, True])
def test_invalid_skipfooter_non_int(python_parser_only, skipfooter):
    # see gh-15925 (comment)
    data = "a\n1\n2"
    parser = python_parser_only
    msg = "skipfooter must be an integer"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=skipfooter)


def test_invalid_skipfooter_negative(python_parser_only):
    # see gh-15925 (comment)
    data = "a\n1\n2"
    parser = python_parser_only
    msg = "skipfooter cannot be negative"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=-1)


@pytest.mark.parametrize("kwargs", [{"sep": None}, {"delimiter": "|"}])
def test_sniff_delimiter(python_parser_only, kwargs):
    data = """index|A|B|C
foo|1|2|3
bar|4|5|6
baz|7|8|9
"""
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "C"],
        index=Index(["foo", "bar", "baz"], name="index"),
    )
    tm.assert_frame_equal(result, expected)


def test_sniff_delimiter_comment(python_parser_only):
    data = """# comment line
index|A|B|C
# comment line
foo|1|2|3 # ignore | this
bar|4|5|6
baz|7|8|9
"""
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), index_col=0, sep=None, comment="#")
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "C"],
        index=Index(["foo", "bar", "baz"], name="index"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("encoding", [None, "utf-8"])
def test_sniff_delimiter_encoding(python_parser_only, encoding):
    parser = python_parser_only
    data = """ignore this
ignore this too
index|A|B|C
foo|1|2|3
bar|4|5|6
baz|7|8|9
"""

    if encoding is not None:
        data = data.encode(encoding)
        data = BytesIO(data)
        data = TextIOWrapper(data, encoding=encoding)
    else:
        data = StringIO(data)

    result = parser.read_csv(data, index_col=0, sep=None, skiprows=2, encoding=encoding)
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["A", "B", "C"],
        index=Index(["foo", "bar", "baz"], name="index"),
    )
    tm.assert_frame_equal(result, expected)


def test_single_line(python_parser_only):
    # see gh-6607: sniff separator
    parser = python_parser_only
    result = parser.read_csv(StringIO("1,2"), names=["a", "b"], header=None, sep=None)

    expected = DataFrame({"a": [1], "b": [2]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{"skipfooter": 2}, {"nrows": 3}])
def test_skipfooter(python_parser_only, kwargs):
    # see gh-6607
    data = """A,B,C
1,2,3
4,5,6
7,8,9
want to skip this
also also skip this
"""
    parser = python_parser_only
    result = parser.read_csv(StringIO(data), **kwargs)

    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "compression,klass", [("gzip", "GzipFile"), ("bz2", "BZ2File")]
)
def test_decompression_regex_sep(python_parser_only, csv1, compression, klass):
    # see gh-6607
    parser = python_parser_only

    with open(csv1, "rb") as f:
        data = f.read()

    data = data.replace(b",", b"::")
    expected = parser.read_csv(csv1)

    module = pytest.importorskip(compression)
    klass = getattr(module, klass)

    with tm.ensure_clean() as path:
        with klass(path, mode="wb") as tmp:
            tmp.write(data)

        result = parser.read_csv(path, sep="::", compression=compression)
        tm.assert_frame_equal(result, expected)


def test_read_csv_buglet_4x_multi_index(python_parser_only):
    # see gh-6607
    data = """                      A       B       C       D        E
one two three   four
a   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640
a   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744
x   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838"""
    parser = python_parser_only

    expected = DataFrame(
        [
            [-0.5109, -2.3358, -0.4645, 0.05076, 0.3640],
            [0.4473, 1.4152, 0.2834, 1.00661, 0.1744],
            [-0.6662, -0.5243, -0.3580, 0.89145, 2.5838],
        ],
        columns=["A", "B", "C", "D", "E"],
        index=MultiIndex.from_tuples(
            [("a", "b", 10.0032, 5), ("a", "q", 20, 4), ("x", "q", 30, 3)],
            names=["one", "two", "three", "four"],
        ),
    )
    result = parser.read_csv(StringIO(data), sep=r"\s+")
    tm.assert_frame_equal(result, expected)


def test_read_csv_buglet_4x_multi_index2(python_parser_only):
    # see gh-6893
    data = "      A B C\na b c\n1 3 7 0 3 6\n3 1 4 1 5 9"
    parser = python_parser_only

    expected = DataFrame.from_records(
        [(1, 3, 7, 0, 3, 6), (3, 1, 4, 1, 5, 9)],
        columns=list("abcABC"),
        index=list("abc"),
    )
    result = parser.read_csv(StringIO(data), sep=r"\s+")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("add_footer", [True, False])
def test_skipfooter_with_decimal(python_parser_only, add_footer):
    # see gh-6971
    data = "1#2\n3#4"
    parser = python_parser_only
    expected = DataFrame({"a": [1.2, 3.4]})

    if add_footer:
        # The stray footer line should not mess with the
        # casting of the first two lines if we skip it.
        kwargs = {"skipfooter": 1}
        data += "\nFooter"
    else:
        kwargs = {}

    result = parser.read_csv(StringIO(data), names=["a"], decimal="#", **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sep", ["::", "#####", "!!!", "123", "#1!c5", "%!c!d", "@@#4:2", "_!pd#_"]
)
@pytest.mark.parametrize(
    "encoding", ["utf-16", "utf-16-be", "utf-16-le", "utf-32", "cp037"]
)
def test_encoding_non_utf8_multichar_sep(python_parser_only, sep, encoding):
    # see gh-3404
    expected = DataFrame({"a": [1], "b": [2]})
    parser = python_parser_only

    data = "1" + sep + "2"
    encoded_data = data.encode(encoding)

    result = parser.read_csv(
        BytesIO(encoded_data), sep=sep, names=["a", "b"], encoding=encoding
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("quoting", [csv.QUOTE_MINIMAL, csv.QUOTE_NONE])
def test_multi_char_sep_quotes(python_parser_only, quoting):
    # see gh-13374
    kwargs = {"sep": ",,"}
    parser = python_parser_only

    data = 'a,,b\n1,,a\n2,,"2,,b"'

    if quoting == csv.QUOTE_NONE:
        msg = "Expected 2 fields in line 3, saw 3"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)
    else:
        msg = "ignored when a multi-char delimiter is used"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), quoting=quoting, **kwargs)


def test_none_delimiter(python_parser_only):
    # see gh-13374 and gh-17465
    parser = python_parser_only
    data = "a,b,c\n0,1,2\n3,4,5,6\n7,8,9"
    expected = DataFrame({"a": [0, 7], "b": [1, 8], "c": [2, 9]})

    # We expect the third line in the data to be
    # skipped because it is malformed, but we do
    # not expect any errors to occur.
    with tm.assert_produces_warning(
        ParserWarning, match="Skipping line 3", check_stacklevel=False
    ):
        result = parser.read_csv(
            StringIO(data), header=0, sep=None, on_bad_lines="warn"
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("data", ['a\n1\n"b"a', 'a,b,c\ncat,foo,bar\ndog,foo,"baz'])
@pytest.mark.parametrize("skipfooter", [0, 1])
def test_skipfooter_bad_row(python_parser_only, data, skipfooter):
    # see gh-13879 and gh-15910
    parser = python_parser_only
    if skipfooter:
        msg = "parsing errors in the skipped footer rows"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)
    else:
        msg = "unexpected end of data|expected after"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), skipfooter=skipfooter)


def test_malformed_skipfooter(python_parser_only):
    parser = python_parser_only
    data = """ignore
A,B,C
1,2,3 # comment
1,2,3,4,5
2,3,4
footer
"""
    msg = "Expected 3 fields in line 4, saw 5"
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), header=1, comment="#", skipfooter=1)


def test_python_engine_file_no_next(python_parser_only):
    parser = python_parser_only

    class NoNextBuffer:
        def __init__(self, csv_data) -> None:
            self.data = csv_data

        def __iter__(self) -> Iterator:
            return self.data.__iter__()

        def read(self):
            return self.data

        def readline(self):
            return self.data

    parser.read_csv(NoNextBuffer("a\n1"))


@pytest.mark.parametrize("bad_line_func", [lambda x: ["2", "3"], lambda x: x[:2]])
def test_on_bad_lines_callable(python_parser_only, bad_line_func):
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    bad_sio = StringIO(data)
    result = parser.read_csv(bad_sio, on_bad_lines=bad_line_func)
    expected = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    tm.assert_frame_equal(result, expected)


def test_on_bad_lines_callable_write_to_external_list(python_parser_only):
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    bad_sio = StringIO(data)
    lst = []

    def bad_line_func(bad_line: list[str]) -> list[str]:
        lst.append(bad_line)
        return ["2", "3"]

    result = parser.read_csv(bad_sio, on_bad_lines=bad_line_func)
    expected = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    tm.assert_frame_equal(result, expected)
    assert lst == [["2", "3", "4", "5", "6"]]


@pytest.mark.parametrize("bad_line_func", [lambda x: ["foo", "bar"], lambda x: x[:2]])
@pytest.mark.parametrize("sep", [",", "111"])
def test_on_bad_lines_callable_iterator_true(python_parser_only, bad_line_func, sep):
    # GH 5686
    # iterator=True has a separate code path than iterator=False
    parser = python_parser_only
    data = f"""
0{sep}1
hi{sep}there
foo{sep}bar{sep}baz
good{sep}bye
"""
    bad_sio = StringIO(data)
    result_iter = parser.read_csv(
        bad_sio, on_bad_lines=bad_line_func, chunksize=1, iterator=True, sep=sep
    )
    expecteds = [
        {"0": "hi", "1": "there"},
        {"0": "foo", "1": "bar"},
        {"0": "good", "1": "bye"},
    ]
    for i, (result, expected) in enumerate(zip(result_iter, expecteds)):
        expected = DataFrame(expected, index=range(i, i + 1))
        tm.assert_frame_equal(result, expected)


def test_on_bad_lines_callable_dont_swallow_errors(python_parser_only):
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    bad_sio = StringIO(data)
    msg = "This function is buggy."

    def bad_line_func(bad_line):
        raise ValueError(msg)

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(bad_sio, on_bad_lines=bad_line_func)


def test_on_bad_lines_callable_not_expected_length(python_parser_only):
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    bad_sio = StringIO(data)

    result = parser.read_csv_check_warnings(
        ParserWarning, "Length of header or names", bad_sio, on_bad_lines=lambda x: x
    )
    expected = DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    tm.assert_frame_equal(result, expected)


def test_on_bad_lines_callable_returns_none(python_parser_only):
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2
2,3,4,5,6
3,4
"""
    bad_sio = StringIO(data)

    result = parser.read_csv(bad_sio, on_bad_lines=lambda x: None)
    expected = DataFrame({"a": [1, 3], "b": [2, 4]})
    tm.assert_frame_equal(result, expected)


def test_on_bad_lines_index_col_inferred(python_parser_only):
    # GH 5686
    parser = python_parser_only
    data = """a,b
1,2,3
4,5,6
"""
    bad_sio = StringIO(data)

    result = parser.read_csv(bad_sio, on_bad_lines=lambda x: ["99", "99"])
    expected = DataFrame({"a": [2, 5], "b": [3, 6]}, index=[1, 4])
    tm.assert_frame_equal(result, expected)


def test_index_col_false_and_header_none(python_parser_only):
    # GH#46955
    parser = python_parser_only
    data = """
0.5,0.03
0.1,0.2,0.3,2
"""
    result = parser.read_csv_check_warnings(
        ParserWarning,
        "Length of header",
        StringIO(data),
        sep=",",
        header=None,
        index_col=False,
    )
    expected = DataFrame({0: [0.5, 0.1], 1: [0.03, 0.2]})
    tm.assert_frame_equal(result, expected)


def test_header_int_do_not_infer_multiindex_names_on_different_line(python_parser_only):
    # GH#46569
    parser = python_parser_only
    data = StringIO("a\na,b\nc,d,e\nf,g,h")
    result = parser.read_csv_check_warnings(
        ParserWarning, "Length of header", data, engine="python", index_col=False
    )
    expected = DataFrame({"a": ["a", "c", "f"]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype", [{"a": object}, {"a": str, "b": np.int64, "c": np.int64}]
)
def test_no_thousand_convert_with_dot_for_non_numeric_cols(python_parser_only, dtype):
    # GH#50270
    parser = python_parser_only
    data = """\
a;b;c
0000.7995;16.000;0
3.03.001.00514;0;4.000
4923.600.041;23.000;131"""
    result = parser.read_csv(
        StringIO(data),
        sep=";",
        dtype=dtype,
        thousands=".",
    )
    expected = DataFrame(
        {
            "a": ["0000.7995", "3.03.001.00514", "4923.600.041"],
            "b": [16000, 0, 23000],
            "c": [0, 4000, 131],
        }
    )
    if dtype["a"] == object:
        expected["a"] = expected["a"].astype(object)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (
            {"a": str, "b": np.float64, "c": np.int64},
            DataFrame(
                {
                    "b": [16000.1, 0, 23000],
                    "c": [0, 4001, 131],
                }
            ),
        ),
        (
            str,
            DataFrame(
                {
                    "b": ["16,000.1", "0", "23,000"],
                    "c": ["0", "4,001", "131"],
                }
            ),
        ),
    ],
)
def test_no_thousand_convert_for_non_numeric_cols(python_parser_only, dtype, expected):
    # GH#50270
    parser = python_parser_only
    data = """a;b;c
0000,7995;16,000.1;0
3,03,001,00514;0;4,001
4923,600,041;23,000;131
"""
    result = parser.read_csv(
        StringIO(data),
        sep=";",
        dtype=dtype,
        thousands=",",
    )
    expected.insert(0, "a", ["0000,7995", "3,03,001,00514", "4923,600,041"])
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tests\test_permutations.py ===
from itertools import permutations
from copy import copy

from sympy.core.expr import unchanged
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.combinatorics.permutations import \
    Permutation, _af_parity, _af_rmul, _af_rmuln, AppliedPermutation, Cycle
from sympy.printing import sstr, srepr, pretty, latex
from sympy.testing.pytest import raises, warns_deprecated_sympy


rmul = Permutation.rmul
a = Symbol('a', integer=True)


def test_Permutation():
    # don't auto fill 0
    raises(ValueError, lambda: Permutation([1]))
    p = Permutation([0, 1, 2, 3])
    # call as bijective
    assert [p(i) for i in range(p.size)] == list(p)
    # call as operator
    assert p(list(range(p.size))) == list(p)
    # call as function
    assert list(p(1, 2)) == [0, 2, 1, 3]
    raises(TypeError, lambda: p(-1))
    raises(TypeError, lambda: p(5))
    # conversion to list
    assert list(p) == list(range(4))
    assert p.copy() == p
    assert copy(p) == p
    assert Permutation(size=4) == Permutation(3)
    assert Permutation(Permutation(3), size=5) == Permutation(4)
    # cycle form with size
    assert Permutation([[1, 2]], size=4) == Permutation([[1, 2], [0], [3]])
    # random generation
    assert Permutation.random(2) in (Permutation([1, 0]), Permutation([0, 1]))

    p = Permutation([2, 5, 1, 6, 3, 0, 4])
    q = Permutation([[1], [0, 3, 5, 6, 2, 4]])
    assert len({p, p}) == 1
    r = Permutation([1, 3, 2, 0, 4, 6, 5])
    ans = Permutation(_af_rmuln(*[w.array_form for w in (p, q, r)])).array_form
    assert rmul(p, q, r).array_form == ans
    # make sure no other permutation of p, q, r could have given
    # that answer
    for a, b, c in permutations((p, q, r)):
        if (a, b, c) == (p, q, r):
            continue
        assert rmul(a, b, c).array_form != ans

    assert p.support() == list(range(7))
    assert q.support() == [0, 2, 3, 4, 5, 6]
    assert Permutation(p.cyclic_form).array_form == p.array_form
    assert p.cardinality == 5040
    assert q.cardinality == 5040
    assert q.cycles == 2
    assert rmul(q, p) == Permutation([4, 6, 1, 2, 5, 3, 0])
    assert rmul(p, q) == Permutation([6, 5, 3, 0, 2, 4, 1])
    assert _af_rmul(p.array_form, q.array_form) == \
        [6, 5, 3, 0, 2, 4, 1]

    assert rmul(Permutation([[1, 2, 3], [0, 4]]),
                Permutation([[1, 2, 4], [0], [3]])).cyclic_form == \
        [[0, 4, 2], [1, 3]]
    assert q.array_form == [3, 1, 4, 5, 0, 6, 2]
    assert q.cyclic_form == [[0, 3, 5, 6, 2, 4]]
    assert q.full_cyclic_form == [[0, 3, 5, 6, 2, 4], [1]]
    assert p.cyclic_form == [[0, 2, 1, 5], [3, 6, 4]]
    t = p.transpositions()
    assert t == [(0, 5), (0, 1), (0, 2), (3, 4), (3, 6)]
    assert Permutation.rmul(*[Permutation(Cycle(*ti)) for ti in (t)])
    assert Permutation([1, 0]).transpositions() == [(0, 1)]

    assert p**13 == p
    assert q**0 == Permutation(list(range(q.size)))
    assert q**-2 == ~q**2
    assert q**2 == Permutation([5, 1, 0, 6, 3, 2, 4])
    assert q**3 == q**2*q
    assert q**4 == q**2*q**2

    a = Permutation(1, 3)
    b = Permutation(2, 0, 3)
    I = Permutation(3)
    assert ~a == a**-1
    assert a*~a == I
    assert a*b**-1 == a*~b

    ans = Permutation(0, 5, 3, 1, 6)(2, 4)
    assert (p + q.rank()).rank() == ans.rank()
    assert (p + q.rank())._rank == ans.rank()
    assert (q + p.rank()).rank() == ans.rank()
    raises(TypeError, lambda: p + Permutation(list(range(10))))

    assert (p - q.rank()).rank() == Permutation(0, 6, 3, 1, 2, 5, 4).rank()
    assert p.rank() - q.rank() < 0  # for coverage: make sure mod is used
    assert (q - p.rank()).rank() == Permutation(1, 4, 6, 2)(3, 5).rank()

    assert p*q == Permutation(_af_rmuln(*[list(w) for w in (q, p)]))
    assert p*Permutation([]) == p
    assert Permutation([])*p == p
    assert p*Permutation([[0, 1]]) == Permutation([2, 5, 0, 6, 3, 1, 4])
    assert Permutation([[0, 1]])*p == Permutation([5, 2, 1, 6, 3, 0, 4])

    pq = p ^ q
    assert pq == Permutation([5, 6, 0, 4, 1, 2, 3])
    assert pq == rmul(q, p, ~q)
    qp = q ^ p
    assert qp == Permutation([4, 3, 6, 2, 1, 5, 0])
    assert qp == rmul(p, q, ~p)
    raises(ValueError, lambda: p ^ Permutation([]))

    assert p.commutator(q) == Permutation(0, 1, 3, 4, 6, 5, 2)
    assert q.commutator(p) == Permutation(0, 2, 5, 6, 4, 3, 1)
    assert p.commutator(q) == ~q.commutator(p)
    raises(ValueError, lambda: p.commutator(Permutation([])))

    assert len(p.atoms()) == 7
    assert q.atoms() == {0, 1, 2, 3, 4, 5, 6}

    assert p.inversion_vector() == [2, 4, 1, 3, 1, 0]
    assert q.inversion_vector() == [3, 1, 2, 2, 0, 1]

    assert Permutation.from_inversion_vector(p.inversion_vector()) == p
    assert Permutation.from_inversion_vector(q.inversion_vector()).array_form\
        == q.array_form
    raises(ValueError, lambda: Permutation.from_inversion_vector([0, 2]))
    assert Permutation(list(range(500, -1, -1))).inversions() == 125250

    s = Permutation([0, 4, 1, 3, 2])
    assert s.parity() == 0
    _ = s.cyclic_form  # needed to create a value for _cyclic_form
    assert len(s._cyclic_form) != s.size and s.parity() == 0
    assert not s.is_odd
    assert s.is_even
    assert Permutation([0, 1, 4, 3, 2]).parity() == 1
    assert _af_parity([0, 4, 1, 3, 2]) == 0
    assert _af_parity([0, 1, 4, 3, 2]) == 1

    s = Permutation([0])

    assert s.is_Singleton
    assert Permutation([]).is_Empty

    r = Permutation([3, 2, 1, 0])
    assert (r**2).is_Identity

    assert rmul(~p, p).is_Identity
    assert (~p)**13 == Permutation([5, 2, 0, 4, 6, 1, 3])
    assert p.max() == 6
    assert p.min() == 0

    q = Permutation([[6], [5], [0, 1, 2, 3, 4]])

    assert q.max() == 4
    assert q.min() == 0

    p = Permutation([1, 5, 2, 0, 3, 6, 4])
    q = Permutation([[1, 2, 3, 5, 6], [0, 4]])

    assert p.ascents() == [0, 3, 4]
    assert q.ascents() == [1, 2, 4]
    assert r.ascents() == []

    assert p.descents() == [1, 2, 5]
    assert q.descents() == [0, 3, 5]
    assert Permutation(r.descents()).is_Identity

    assert p.inversions() == 7
    # test the merge-sort with a longer permutation
    big = list(p) + list(range(p.max() + 1, p.max() + 130))
    assert Permutation(big).inversions() == 7
    assert p.signature() == -1
    assert q.inversions() == 11
    assert q.signature() == -1
    assert rmul(p, ~p).inversions() == 0
    assert rmul(p, ~p).signature() == 1

    assert p.order() == 6
    assert q.order() == 10
    assert (p**(p.order())).is_Identity

    assert p.length() == 6
    assert q.length() == 7
    assert r.length() == 4

    assert p.runs() == [[1, 5], [2], [0, 3, 6], [4]]
    assert q.runs() == [[4], [2, 3, 5], [0, 6], [1]]
    assert r.runs() == [[3], [2], [1], [0]]

    assert p.index() == 8
    assert q.index() == 8
    assert r.index() == 3

    assert p.get_precedence_distance(q) == q.get_precedence_distance(p)
    assert p.get_adjacency_distance(q) == p.get_adjacency_distance(q)
    assert p.get_positional_distance(q) == p.get_positional_distance(q)
    p = Permutation([0, 1, 2, 3])
    q = Permutation([3, 2, 1, 0])
    assert p.get_precedence_distance(q) == 6
    assert p.get_adjacency_distance(q) == 3
    assert p.get_positional_distance(q) == 8
    p = Permutation([0, 3, 1, 2, 4])
    q = Permutation.josephus(4, 5, 2)
    assert p.get_adjacency_distance(q) == 3
    raises(ValueError, lambda: p.get_adjacency_distance(Permutation([])))
    raises(ValueError, lambda: p.get_positional_distance(Permutation([])))
    raises(ValueError, lambda: p.get_precedence_distance(Permutation([])))

    a = [Permutation.unrank_nonlex(4, i) for i in range(5)]
    iden = Permutation([0, 1, 2, 3])
    for i in range(5):
        for j in range(i + 1, 5):
            assert a[i].commutes_with(a[j]) == \
                (rmul(a[i], a[j]) == rmul(a[j], a[i]))
            if a[i].commutes_with(a[j]):
                assert a[i].commutator(a[j]) == iden
                assert a[j].commutator(a[i]) == iden

    a = Permutation(3)
    b = Permutation(0, 6, 3)(1, 2)
    assert a.cycle_structure == {1: 4}
    assert b.cycle_structure == {2: 1, 3: 1, 1: 2}
    # issue 11130
    raises(ValueError, lambda: Permutation(3, size=3))
    raises(ValueError, lambda: Permutation([1, 2, 0, 3], size=3))


def test_Permutation_subclassing():
    # Subclass that adds permutation application on iterables
    class CustomPermutation(Permutation):
        def __call__(self, *i):
            try:
                return super().__call__(*i)
            except TypeError:
                pass

            try:
                perm_obj = i[0]
                return [self._array_form[j] for j in perm_obj]
            except TypeError:
                raise TypeError('unrecognized argument')

        def __eq__(self, other):
            if isinstance(other, Permutation):
                return self._hashable_content() == other._hashable_content()
            else:
                return super().__eq__(other)

        def __hash__(self):
            return super().__hash__()

    p = CustomPermutation([1, 2, 3, 0])
    q = Permutation([1, 2, 3, 0])

    assert p == q
    raises(TypeError, lambda: q([1, 2]))
    assert [2, 3] == p([1, 2])

    assert type(p * q) == CustomPermutation
    assert type(q * p) == Permutation  # True because q.__mul__(p) is called!

    # Run all tests for the Permutation class also on the subclass
    def wrapped_test_Permutation():
        # Monkeypatch the class definition in the globals
        globals()['__Perm'] = globals()['Permutation']
        globals()['Permutation'] = CustomPermutation
        test_Permutation()
        globals()['Permutation'] = globals()['__Perm']  # Restore
        del globals()['__Perm']

    wrapped_test_Permutation()


def test_josephus():
    assert Permutation.josephus(4, 6, 1) == Permutation([3, 1, 0, 2, 5, 4])
    assert Permutation.josephus(1, 5, 1).is_Identity


def test_ranking():
    assert Permutation.unrank_lex(5, 10).rank() == 10
    p = Permutation.unrank_lex(15, 225)
    assert p.rank() == 225
    p1 = p.next_lex()
    assert p1.rank() == 226
    assert Permutation.unrank_lex(15, 225).rank() == 225
    assert Permutation.unrank_lex(10, 0).is_Identity
    p = Permutation.unrank_lex(4, 23)
    assert p.rank() == 23
    assert p.array_form == [3, 2, 1, 0]
    assert p.next_lex() is None

    p = Permutation([1, 5, 2, 0, 3, 6, 4])
    q = Permutation([[1, 2, 3, 5, 6], [0, 4]])
    a = [Permutation.unrank_trotterjohnson(4, i).array_form for i in range(5)]
    assert a == [[0, 1, 2, 3], [0, 1, 3, 2], [0, 3, 1, 2], [3, 0, 1,
        2], [3, 0, 2, 1] ]
    assert [Permutation(pa).rank_trotterjohnson() for pa in a] == list(range(5))
    assert Permutation([0, 1, 2, 3]).next_trotterjohnson() == \
        Permutation([0, 1, 3, 2])

    assert q.rank_trotterjohnson() == 2283
    assert p.rank_trotterjohnson() == 3389
    assert Permutation([1, 0]).rank_trotterjohnson() == 1
    a = Permutation(list(range(3)))
    b = a
    l = []
    tj = []
    for i in range(6):
        l.append(a)
        tj.append(b)
        a = a.next_lex()
        b = b.next_trotterjohnson()
    assert a == b is None
    assert {tuple(a) for a in l} == {tuple(a) for a in tj}

    p = Permutation([2, 5, 1, 6, 3, 0, 4])
    q = Permutation([[6], [5], [0, 1, 2, 3, 4]])
    assert p.rank() == 1964
    assert q.rank() == 870
    assert Permutation([]).rank_nonlex() == 0
    prank = p.rank_nonlex()
    assert prank == 1600
    assert Permutation.unrank_nonlex(7, 1600) == p
    qrank = q.rank_nonlex()
    assert qrank == 41
    assert Permutation.unrank_nonlex(7, 41) == Permutation(q.array_form)

    a = [Permutation.unrank_nonlex(4, i).array_form for i in range(24)]
    assert a == [
        [1, 2, 3, 0], [3, 2, 0, 1], [1, 3, 0, 2], [1, 2, 0, 3], [2, 3, 1, 0],
        [2, 0, 3, 1], [3, 0, 1, 2], [2, 0, 1, 3], [1, 3, 2, 0], [3, 0, 2, 1],
        [1, 0, 3, 2], [1, 0, 2, 3], [2, 1, 3, 0], [2, 3, 0, 1], [3, 1, 0, 2],
        [2, 1, 0, 3], [3, 2, 1, 0], [0, 2, 3, 1], [0, 3, 1, 2], [0, 2, 1, 3],
        [3, 1, 2, 0], [0, 3, 2, 1], [0, 1, 3, 2], [0, 1, 2, 3]]

    N = 10
    p1 = Permutation(a[0])
    for i in range(1, N+1):
        p1 = p1*Permutation(a[i])
    p2 = Permutation.rmul_with_af(*[Permutation(h) for h in a[N::-1]])
    assert p1 == p2

    ok = []
    p = Permutation([1, 0])
    for i in range(3):
        ok.append(p.array_form)
        p = p.next_nonlex()
        if p is None:
            ok.append(None)
            break
    assert ok == [[1, 0], [0, 1], None]
    assert Permutation([3, 2, 0, 1]).next_nonlex() == Permutation([1, 3, 0, 2])
    assert [Permutation(pa).rank_nonlex() for pa in a] == list(range(24))


def test_mul():
    a, b = [0, 2, 1, 3], [0, 1, 3, 2]
    assert _af_rmul(a, b) == [0, 2, 3, 1]
    assert _af_rmuln(a, b, list(range(4))) == [0, 2, 3, 1]
    assert rmul(Permutation(a), Permutation(b)).array_form == [0, 2, 3, 1]

    a = Permutation([0, 2, 1, 3])
    b = (0, 1, 3, 2)
    c = (3, 1, 2, 0)
    assert Permutation.rmul(a, b, c) == Permutation([1, 2, 3, 0])
    assert Permutation.rmul(a, c) == Permutation([3, 2, 1, 0])
    raises(TypeError, lambda: Permutation.rmul(b, c))

    n = 6
    m = 8
    a = [Permutation.unrank_nonlex(n, i).array_form for i in range(m)]
    h = list(range(n))
    for i in range(m):
        h = _af_rmul(h, a[i])
        h2 = _af_rmuln(*a[:i + 1])
        assert h == h2


def test_args():
    p = Permutation([(0, 3, 1, 2), (4, 5)])
    assert p._cyclic_form is None
    assert Permutation(p) == p
    assert p.cyclic_form == [[0, 3, 1, 2], [4, 5]]
    assert p._array_form == [3, 2, 0, 1, 5, 4]
    p = Permutation((0, 3, 1, 2))
    assert p._cyclic_form is None
    assert p._array_form == [0, 3, 1, 2]
    assert Permutation([0]) == Permutation((0, ))
    assert Permutation([[0], [1]]) == Permutation(((0, ), (1, ))) == \
        Permutation(((0, ), [1]))
    assert Permutation([[1, 2]]) == Permutation([0, 2, 1])
    assert Permutation([[1], [4, 2]]) == Permutation([0, 1, 4, 3, 2])
    assert Permutation([[1], [4, 2]], size=1) == Permutation([0, 1, 4, 3, 2])
    assert Permutation(
        [[1], [4, 2]], size=6) == Permutation([0, 1, 4, 3, 2, 5])
    assert Permutation([[0, 1], [0, 2]]) == Permutation(0, 1, 2)
    assert Permutation([], size=3) == Permutation([0, 1, 2])
    assert Permutation(3).list(5) == [0, 1, 2, 3, 4]
    assert Permutation(3).list(-1) == []
    assert Permutation(5)(1, 2).list(-1) == [0, 2, 1]
    assert Permutation(5)(1, 2).list() == [0, 2, 1, 3, 4, 5]
    raises(ValueError, lambda: Permutation([1, 2], [0]))
           # enclosing brackets needed
    raises(ValueError, lambda: Permutation([[1, 2], 0]))
           # enclosing brackets needed on 0
    raises(ValueError, lambda: Permutation([1, 1, 0]))
    raises(ValueError, lambda: Permutation([4, 5], size=10))  # where are 0-3?
    # but this is ok because cycles imply that only those listed moved
    assert Permutation(4, 5) == Permutation([0, 1, 2, 3, 5, 4])


def test_Cycle():
    assert str(Cycle()) == '()'
    assert Cycle(Cycle(1,2)) == Cycle(1, 2)
    assert Cycle(1,2).copy() == Cycle(1,2)
    assert list(Cycle(1, 3, 2)) == [0, 3, 1, 2]
    assert Cycle(1, 2)(2, 3) == Cycle(1, 3, 2)
    assert Cycle(1, 2)(2, 3)(4, 5) == Cycle(1, 3, 2)(4, 5)
    assert Permutation(Cycle(1, 2)(2, 1, 0, 3)).cyclic_form, Cycle(0, 2, 1)
    raises(ValueError, lambda: Cycle().list())
    assert Cycle(1, 2).list() == [0, 2, 1]
    assert Cycle(1, 2).list(4) == [0, 2, 1, 3]
    assert Cycle(3).list(2) == [0, 1]
    assert Cycle(3).list(6) == [0, 1, 2, 3, 4, 5]
    assert Permutation(Cycle(1, 2), size=4) == \
        Permutation([0, 2, 1, 3])
    assert str(Cycle(1, 2)(4, 5)) == '(1 2)(4 5)'
    assert str(Cycle(1, 2)) == '(1 2)'
    assert Cycle(Permutation(list(range(3)))) == Cycle()
    assert Cycle(1, 2).list() == [0, 2, 1]
    assert Cycle(1, 2).list(4) == [0, 2, 1, 3]
    assert Cycle().size == 0
    raises(ValueError, lambda: Cycle((1, 2)))
    raises(ValueError, lambda: Cycle(1, 2, 1))
    raises(TypeError, lambda: Cycle(1, 2)*{})
    raises(ValueError, lambda: Cycle(4)[a])
    raises(ValueError, lambda: Cycle(2, -4, 3))

    # check round-trip
    p = Permutation([[1, 2], [4, 3]], size=5)
    assert Permutation(Cycle(p)) == p


def test_from_sequence():
    assert Permutation.from_sequence('SymPy') == Permutation(4)(0, 1, 3)
    assert Permutation.from_sequence('SymPy', key=lambda x: x.lower()) == \
        Permutation(4)(0, 2)(1, 3)


def test_resize():
    p = Permutation(0, 1, 2)
    assert p.resize(5) == Permutation(0, 1, 2, size=5)
    assert p.resize(4) == Permutation(0, 1, 2, size=4)
    assert p.resize(3) == p
    raises(ValueError, lambda: p.resize(2))

    p = Permutation(0, 1, 2)(3, 4)(5, 6)
    assert p.resize(3) == Permutation(0, 1, 2)
    raises(ValueError, lambda: p.resize(4))


def test_printing_cyclic():
    p1 = Permutation([0, 2, 1])
    assert repr(p1) == 'Permutation(1, 2)'
    assert str(p1) == '(1 2)'
    p2 = Permutation()
    assert repr(p2) == 'Permutation()'
    assert str(p2) == '()'
    p3 = Permutation([1, 2, 0, 3])
    assert repr(p3) == 'Permutation(3)(0, 1, 2)'


def test_printing_non_cyclic():
    p1 = Permutation([0, 1, 2, 3, 4, 5])
    assert srepr(p1, perm_cyclic=False) == 'Permutation([], size=6)'
    assert sstr(p1, perm_cyclic=False) == 'Permutation([], size=6)'
    p2 = Permutation([0, 1, 2])
    assert srepr(p2, perm_cyclic=False) == 'Permutation([0, 1, 2])'
    assert sstr(p2, perm_cyclic=False) == 'Permutation([0, 1, 2])'

    p3 = Permutation([0, 2, 1])
    assert srepr(p3, perm_cyclic=False) == 'Permutation([0, 2, 1])'
    assert sstr(p3, perm_cyclic=False) == 'Permutation([0, 2, 1])'
    p4 = Permutation([0, 1, 3, 2, 4, 5, 6, 7])
    assert srepr(p4, perm_cyclic=False) == 'Permutation([0, 1, 3, 2], size=8)'


def test_deprecated_print_cyclic():
    p = Permutation(0, 1, 2)
    try:
        Permutation.print_cyclic = True
        with warns_deprecated_sympy():
            assert sstr(p) == '(0 1 2)'
        with warns_deprecated_sympy():
            assert srepr(p) == 'Permutation(0, 1, 2)'
        with warns_deprecated_sympy():
            assert pretty(p) == '(0 1 2)'
        with warns_deprecated_sympy():
            assert latex(p) == r'\left( 0\; 1\; 2\right)'

        Permutation.print_cyclic = False
        with warns_deprecated_sympy():
            assert sstr(p) == 'Permutation([1, 2, 0])'
        with warns_deprecated_sympy():
            assert srepr(p) == 'Permutation([1, 2, 0])'
        with warns_deprecated_sympy():
            assert pretty(p, use_unicode=False) == '/0 1 2\\\n\\1 2 0/'
        with warns_deprecated_sympy():
            assert latex(p) == \
                r'\begin{pmatrix} 0 & 1 & 2 \\ 1 & 2 & 0 \end{pmatrix}'
    finally:
        Permutation.print_cyclic = None


def test_permutation_equality():
    a = Permutation(0, 1, 2)
    b = Permutation(0, 1, 2)
    assert Eq(a, b) is S.true
    c = Permutation(0, 2, 1)
    assert Eq(a, c) is S.false

    d = Permutation(0, 1, 2, size=4)
    assert unchanged(Eq, a, d)
    e = Permutation(0, 2, 1, size=4)
    assert unchanged(Eq, a, e)

    i = Permutation()
    assert unchanged(Eq, i, 0)
    assert unchanged(Eq, 0, i)


def test_issue_17661():
    c1 = Cycle(1,2)
    c2 = Cycle(1,2)
    assert c1 == c2
    assert repr(c1) == 'Cycle(1, 2)'
    assert c1 == c2


def test_permutation_apply():
    x = Symbol('x')
    p = Permutation(0, 1, 2)
    assert p.apply(0) == 1
    assert isinstance(p.apply(0), Integer)
    assert p.apply(x) == AppliedPermutation(p, x)
    assert AppliedPermutation(p, x).subs(x, 0) == 1

    x = Symbol('x', integer=False)
    raises(NotImplementedError, lambda: p.apply(x))
    x = Symbol('x', negative=True)
    raises(NotImplementedError, lambda: p.apply(x))


def test_AppliedPermutation():
    x = Symbol('x')
    p = Permutation(0, 1, 2)
    raises(ValueError, lambda: AppliedPermutation((0, 1, 2), x))
    assert AppliedPermutation(p, 1, evaluate=True) == 2
    assert AppliedPermutation(p, 1, evaluate=False).__class__ == \
        AppliedPermutation