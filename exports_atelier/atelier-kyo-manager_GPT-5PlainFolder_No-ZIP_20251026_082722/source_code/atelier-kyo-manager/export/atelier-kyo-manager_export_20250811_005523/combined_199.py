
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\tests\test_array_expressions.py ===
import random

from sympy import tensordiagonal, eye, KroneckerDelta, Array
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.combinatorics import Permutation
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, ArraySymbol, ArrayElement, \
    PermuteDims, ArrayContraction, ArrayTensorProduct, ArrayDiagonal, \
    ArrayAdd, nest_permutation, ArrayElementwiseApplyFunc, _EditArrayContraction, _ArgE, _array_tensor_product, \
    _array_contraction, _array_diagonal, _array_add, _permute_dims, Reshape
from sympy.testing.pytest import raises

i, j, k, l, m, n = symbols("i j k l m n")


M = ArraySymbol("M", (k, k))
N = ArraySymbol("N", (k, k))
P = ArraySymbol("P", (k, k))
Q = ArraySymbol("Q", (k, k))

A = ArraySymbol("A", (k, k))
B = ArraySymbol("B", (k, k))
C = ArraySymbol("C", (k, k))
D = ArraySymbol("D", (k, k))

X = ArraySymbol("X", (k, k))
Y = ArraySymbol("Y", (k, k))

a = ArraySymbol("a", (k, 1))
b = ArraySymbol("b", (k, 1))
c = ArraySymbol("c", (k, 1))
d = ArraySymbol("d", (k, 1))


def test_array_symbol_and_element():
    A = ArraySymbol("A", (2,))
    A0 = ArrayElement(A, (0,))
    A1 = ArrayElement(A, (1,))
    assert A[0] == A0
    assert A[1] != A0
    assert A.as_explicit() == ImmutableDenseNDimArray([A0, A1])

    A2 = tensorproduct(A, A)
    assert A2.shape == (2, 2)
    # TODO: not yet supported:
    # assert A2.as_explicit() == Array([[A[0]*A[0], A[1]*A[0]], [A[0]*A[1], A[1]*A[1]]])
    A3 = tensorcontraction(A2, (0, 1))
    assert A3.shape == ()
    # TODO: not yet supported:
    # assert A3.as_explicit() == Array([])

    A = ArraySymbol("A", (2, 3, 4))
    Ae = A.as_explicit()
    assert Ae == ImmutableDenseNDimArray(
        [[[ArrayElement(A, (i, j, k)) for k in range(4)] for j in range(3)] for i in range(2)])

    p = _permute_dims(A, Permutation(0, 2, 1))
    assert isinstance(p, PermuteDims)

    A = ArraySymbol("A", (2,))
    raises(IndexError, lambda: A[()])
    raises(IndexError, lambda: A[0, 1])
    raises(ValueError, lambda: A[-1])
    raises(ValueError, lambda: A[2])

    O = OneArray(3, 4)
    Z = ZeroArray(m, n)

    raises(IndexError, lambda: O[()])
    raises(IndexError, lambda: O[1, 2, 3])
    raises(ValueError, lambda: O[3, 0])
    raises(ValueError, lambda: O[0, 4])

    assert O[1, 2] == 1
    assert Z[1, 2] == 0


def test_zero_array():
    assert ZeroArray() == 0
    assert ZeroArray().is_Integer

    za = ZeroArray(3, 2, 4)
    assert za.shape == (3, 2, 4)
    za_e = za.as_explicit()
    assert za_e.shape == (3, 2, 4)

    m, n, k = symbols("m n k")
    za = ZeroArray(m, n, k, 2)
    assert za.shape == (m, n, k, 2)
    raises(ValueError, lambda: za.as_explicit())


def test_one_array():
    assert OneArray() == 1
    assert OneArray().is_Integer

    oa = OneArray(3, 2, 4)
    assert oa.shape == (3, 2, 4)
    oa_e = oa.as_explicit()
    assert oa_e.shape == (3, 2, 4)

    m, n, k = symbols("m n k")
    oa = OneArray(m, n, k, 2)
    assert oa.shape == (m, n, k, 2)
    raises(ValueError, lambda: oa.as_explicit())


def test_arrayexpr_contraction_construction():

    cg = _array_contraction(A)
    assert cg == A

    cg = _array_contraction(_array_tensor_product(A, B), (1, 0))
    assert cg == _array_contraction(_array_tensor_product(A, B), (0, 1))

    cg = _array_contraction(_array_tensor_product(M, N), (0, 1))
    indtup = cg._get_contraction_tuples()
    assert indtup == [[(0, 0), (0, 1)]]
    assert cg._contraction_tuples_to_contraction_indices(cg.expr, indtup) == [(0, 1)]

    cg = _array_contraction(_array_tensor_product(M, N), (1, 2))
    indtup = cg._get_contraction_tuples()
    assert indtup == [[(0, 1), (1, 0)]]
    assert cg._contraction_tuples_to_contraction_indices(cg.expr, indtup) == [(1, 2)]

    cg = _array_contraction(_array_tensor_product(M, M, N), (1, 4), (2, 5))
    indtup = cg._get_contraction_tuples()
    assert indtup == [[(0, 0), (1, 1)], [(0, 1), (2, 0)]]
    assert cg._contraction_tuples_to_contraction_indices(cg.expr, indtup) == [(0, 3), (1, 4)]

    # Test removal of trivial contraction:
    assert _array_contraction(a, (1,)) == a
    assert _array_contraction(
        _array_tensor_product(a, b), (0, 2), (1,), (3,)) == _array_contraction(
        _array_tensor_product(a, b), (0, 2))


def test_arrayexpr_array_flatten():

    # Flatten nested ArrayTensorProduct objects:
    expr1 = _array_tensor_product(M, N)
    expr2 = _array_tensor_product(P, Q)
    expr = _array_tensor_product(expr1, expr2)
    assert expr == _array_tensor_product(M, N, P, Q)
    assert expr.args == (M, N, P, Q)

    # Flatten mixed ArrayTensorProduct and ArrayContraction objects:
    cg1 = _array_contraction(expr1, (1, 2))
    cg2 = _array_contraction(expr2, (0, 3))

    expr = _array_tensor_product(cg1, cg2)
    assert expr == _array_contraction(_array_tensor_product(M, N, P, Q), (1, 2), (4, 7))

    expr = _array_tensor_product(M, cg1)
    assert expr == _array_contraction(_array_tensor_product(M, M, N), (3, 4))

    # Flatten nested ArrayContraction objects:
    cgnested = _array_contraction(cg1, (0, 1))
    assert cgnested == _array_contraction(_array_tensor_product(M, N), (0, 3), (1, 2))

    cgnested = _array_contraction(_array_tensor_product(cg1, cg2), (0, 3))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 6), (1, 2), (4, 7))

    cg3 = _array_contraction(_array_tensor_product(M, N, P, Q), (1, 3), (2, 4))
    cgnested = _array_contraction(cg3, (0, 1))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 5), (1, 3), (2, 4))

    cgnested = _array_contraction(cg3, (0, 3), (1, 2))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 7), (1, 3), (2, 4), (5, 6))

    cg4 = _array_contraction(_array_tensor_product(M, N, P, Q), (1, 5), (3, 7))
    cgnested = _array_contraction(cg4, (0, 1))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 2), (1, 5), (3, 7))

    cgnested = _array_contraction(cg4, (0, 1), (2, 3))
    assert cgnested == _array_contraction(_array_tensor_product(M, N, P, Q), (0, 2), (1, 5), (3, 7), (4, 6))

    cg = _array_diagonal(cg4)
    assert cg == cg4
    assert isinstance(cg, type(cg4))

    # Flatten nested ArrayDiagonal objects:
    cg1 = _array_diagonal(expr1, (1, 2))
    cg2 = _array_diagonal(expr2, (0, 3))
    cg3 = _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 3), (2, 4))
    cg4 = _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 5), (3, 7))

    cgnested = _array_diagonal(cg1, (0, 1))
    assert cgnested == _array_diagonal(_array_tensor_product(M, N), (1, 2), (0, 3))

    cgnested = _array_diagonal(cg3, (1, 2))
    assert cgnested == _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 3), (2, 4), (5, 6))

    cgnested = _array_diagonal(cg4, (1, 2))
    assert cgnested == _array_diagonal(_array_tensor_product(M, N, P, Q), (1, 5), (3, 7), (2, 4))

    cg = _array_add(M, N)
    cg2 = _array_add(cg, P)
    assert isinstance(cg2, ArrayAdd)
    assert cg2.args == (M, N, P)
    assert cg2.shape == (k, k)

    expr = _array_tensor_product(_array_diagonal(X, (0, 1)), _array_diagonal(A, (0, 1)))
    assert expr == _array_diagonal(_array_tensor_product(X, A), (0, 1), (2, 3))

    expr1 = _array_diagonal(_array_tensor_product(X, A), (1, 2))
    expr2 = _array_tensor_product(expr1, a)
    assert expr2 == _permute_dims(_array_diagonal(_array_tensor_product(X, A, a), (1, 2)), [0, 1, 4, 2, 3])

    expr1 = _array_contraction(_array_tensor_product(X, A), (1, 2))
    expr2 = _array_tensor_product(expr1, a)
    assert isinstance(expr2, ArrayContraction)
    assert isinstance(expr2.expr, ArrayTensorProduct)

    cg = _array_tensor_product(_array_diagonal(_array_tensor_product(A, X, Y), (0, 3), (1, 5)), a, b)
    assert cg == _permute_dims(_array_diagonal(_array_tensor_product(A, X, Y, a, b), (0, 3), (1, 5)), [0, 1, 6, 7, 2, 3, 4, 5])


def test_arrayexpr_array_diagonal():
    cg = _array_diagonal(M, (1, 0))
    assert cg == _array_diagonal(M, (0, 1))

    cg = _array_diagonal(_array_tensor_product(M, N, P), (4, 1), (2, 0))
    assert cg == _array_diagonal(_array_tensor_product(M, N, P), (1, 4), (0, 2))

    cg = _array_diagonal(_array_tensor_product(M, N), (1, 2), (3,), allow_trivial_diags=True)
    assert cg == _permute_dims(_array_diagonal(_array_tensor_product(M, N), (1, 2)), [0, 2, 1])

    Ax = ArraySymbol("Ax", shape=(1, 2, 3, 4, 3, 5, 6, 2, 7))
    cg = _array_diagonal(Ax, (1, 7), (3,), (2, 4), (6,), allow_trivial_diags=True)
    assert cg == _permute_dims(_array_diagonal(Ax, (1, 7), (2, 4)), [0, 2, 4, 5, 1, 6, 3])

    cg = _array_diagonal(M, (0,), allow_trivial_diags=True)
    assert cg == _permute_dims(M, [1, 0])

    raises(ValueError, lambda: _array_diagonal(M, (0, 0)))


def test_arrayexpr_array_shape():
    expr = _array_tensor_product(M, N, P, Q)
    assert expr.shape == (k, k, k, k, k, k, k, k)
    Z = MatrixSymbol("Z", m, n)
    expr = _array_tensor_product(M, Z)
    assert expr.shape == (k, k, m, n)
    expr2 = _array_contraction(expr, (0, 1))
    assert expr2.shape == (m, n)
    expr2 = _array_diagonal(expr, (0, 1))
    assert expr2.shape == (m, n, k)
    exprp = _permute_dims(expr, [2, 1, 3, 0])
    assert exprp.shape == (m, k, n, k)
    expr3 = _array_tensor_product(N, Z)
    expr2 = _array_add(expr, expr3)
    assert expr2.shape == (k, k, m, n)

    # Contraction along axes with discordant dimensions:
    raises(ValueError, lambda: _array_contraction(expr, (1, 2)))
    # Also diagonal needs the same dimensions:
    raises(ValueError, lambda: _array_diagonal(expr, (1, 2)))
    # Diagonal requires at least to axes to compute the diagonal:
    raises(ValueError, lambda: _array_diagonal(expr, (1,)))


def test_arrayexpr_permutedims_sink():

    cg = _permute_dims(_array_tensor_product(M, N), [0, 1, 3, 2], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_tensor_product(M, _permute_dims(N, [1, 0]))

    cg = _permute_dims(_array_tensor_product(M, N), [1, 0, 3, 2], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_tensor_product(_permute_dims(M, [1, 0]), _permute_dims(N, [1, 0]))

    cg = _permute_dims(_array_tensor_product(M, N), [3, 2, 1, 0], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_tensor_product(_permute_dims(N, [1, 0]), _permute_dims(M, [1, 0]))

    cg = _permute_dims(_array_contraction(_array_tensor_product(M, N), (1, 2)), [1, 0], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_contraction(_permute_dims(_array_tensor_product(M, N), [[0, 3]]), (1, 2))

    cg = _permute_dims(_array_tensor_product(M, N), [1, 0, 3, 2], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_tensor_product(_permute_dims(M, [1, 0]), _permute_dims(N, [1, 0]))

    cg = _permute_dims(_array_contraction(_array_tensor_product(M, N, P), (1, 2), (3, 4)), [1, 0], nest_permutation=False)
    sunk = nest_permutation(cg)
    assert sunk == _array_contraction(_permute_dims(_array_tensor_product(M, N, P), [[0, 5]]), (1, 2), (3, 4))


def test_arrayexpr_push_indices_up_and_down():

    indices = list(range(12))

    contr_diag_indices = [(0, 6), (2, 8)]
    assert ArrayContraction._push_indices_down(contr_diag_indices, indices) == (1, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15)
    assert ArrayContraction._push_indices_up(contr_diag_indices, indices) == (None, 0, None, 1, 2, 3, None, 4, None, 5, 6, 7)

    assert ArrayDiagonal._push_indices_down(contr_diag_indices, indices, 10) == (1, 3, 4, 5, 7, 9, (0, 6), (2, 8), None, None, None, None)
    assert ArrayDiagonal._push_indices_up(contr_diag_indices, indices, 10) == (6, 0, 7, 1, 2, 3, 6, 4, 7, 5, None, None)

    contr_diag_indices = [(1, 2), (7, 8)]
    assert ArrayContraction._push_indices_down(contr_diag_indices, indices) == (0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)
    assert ArrayContraction._push_indices_up(contr_diag_indices, indices) == (0, None, None, 1, 2, 3, 4, None, None, 5, 6, 7)

    assert ArrayDiagonal._push_indices_down(contr_diag_indices, indices, 10) == (0, 3, 4, 5, 6, 9, (1, 2), (7, 8), None, None, None, None)
    assert ArrayDiagonal._push_indices_up(contr_diag_indices, indices, 10) == (0, 6, 6, 1, 2, 3, 4, 7, 7, 5, None, None)


def test_arrayexpr_split_multiple_contractions():
    a = MatrixSymbol("a", k, 1)
    b = MatrixSymbol("b", k, 1)
    A = MatrixSymbol("A", k, k)
    B = MatrixSymbol("B", k, k)
    C = MatrixSymbol("C", k, k)
    X = MatrixSymbol("X", k, k)

    cg = _array_contraction(_array_tensor_product(A.T, a, b, b.T, (A*X*b).applyfunc(cos)), (1, 2, 8), (5, 6, 9))
    expected = _array_contraction(_array_tensor_product(A.T, DiagMatrix(a), OneArray(1), b, b.T, (A*X*b).applyfunc(cos)), (1, 3), (2, 9), (6, 7, 10))
    assert cg.split_multiple_contractions().dummy_eq(expected)

    # Check no overlap of lines:

    cg = _array_contraction(_array_tensor_product(A, a, C, a, B), (1, 2, 4), (5, 6, 8), (3, 7))
    assert cg.split_multiple_contractions() == cg

    cg = _array_contraction(_array_tensor_product(a, b, A), (0, 2, 4), (1, 3))
    assert cg.split_multiple_contractions() == cg


def test_arrayexpr_nested_permutations():

    cg = _permute_dims(_permute_dims(M, (1, 0)), (1, 0))
    assert cg == M

    times = 3
    plist1 = [list(range(6)) for i in range(times)]
    plist2 = [list(range(6)) for i in range(times)]

    for i in range(times):
        random.shuffle(plist1[i])
        random.shuffle(plist2[i])

    plist1.append([2, 5, 4, 1, 0, 3])
    plist2.append([3, 5, 0, 4, 1, 2])

    plist1.append([2, 5, 4, 0, 3, 1])
    plist2.append([3, 0, 5, 1, 2, 4])

    plist1.append([5, 4, 2, 0, 3, 1])
    plist2.append([4, 5, 0, 2, 3, 1])

    Me = M.subs(k, 3).as_explicit()
    Ne = N.subs(k, 3).as_explicit()
    Pe = P.subs(k, 3).as_explicit()
    cge = tensorproduct(Me, Ne, Pe)

    for permutation_array1, permutation_array2 in zip(plist1, plist2):
        p1 = Permutation(permutation_array1)
        p2 = Permutation(permutation_array2)

        cg = _permute_dims(
            _permute_dims(
                _array_tensor_product(M, N, P),
                p1),
            p2
        )
        result = _permute_dims(
            _array_tensor_product(M, N, P),
            p2*p1
        )
        assert cg == result

        # Check that `permutedims` behaves the same way with explicit-component arrays:
        result1 = _permute_dims(_permute_dims(cge, p1), p2)
        result2 = _permute_dims(cge, p2*p1)
        assert result1 == result2


def test_arrayexpr_contraction_permutation_mix():

    Me = M.subs(k, 3).as_explicit()
    Ne = N.subs(k, 3).as_explicit()

    cg1 = _array_contraction(PermuteDims(_array_tensor_product(M, N), Permutation([0, 2, 1, 3])), (2, 3))
    cg2 = _array_contraction(_array_tensor_product(M, N), (1, 3))
    assert cg1 == cg2
    cge1 = tensorcontraction(permutedims(tensorproduct(Me, Ne), Permutation([0, 2, 1, 3])), (2, 3))
    cge2 = tensorcontraction(tensorproduct(Me, Ne), (1, 3))
    assert cge1 == cge2

    cg1 = _permute_dims(_array_tensor_product(M, N), Permutation([0, 1, 3, 2]))
    cg2 = _array_tensor_product(M, _permute_dims(N, Permutation([1, 0])))
    assert cg1 == cg2

    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(M, N, P, Q), Permutation([0, 2, 3, 1, 4, 5, 7, 6])),
        (1, 2), (3, 5)
    )
    cg2 = _array_contraction(
        _array_tensor_product(M, N, P, _permute_dims(Q, Permutation([1, 0]))),
        (1, 5), (2, 3)
    )
    assert cg1 == cg2

    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(M, N, P, Q), Permutation([1, 0, 4, 6, 2, 7, 5, 3])),
        (0, 1), (2, 6), (3, 7)
    )
    cg2 = _permute_dims(
        _array_contraction(
            _array_tensor_product(M, P, Q, N),
            (0, 1), (2, 3), (4, 7)),
        [1, 0]
    )
    assert cg1 == cg2

    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(M, N, P, Q), Permutation([1, 0, 4, 6, 7, 2, 5, 3])),
        (0, 1), (2, 6), (3, 7)
    )
    cg2 = _permute_dims(
        _array_contraction(
            _array_tensor_product(_permute_dims(M, [1, 0]), N, P, Q),
            (0, 1), (3, 6), (4, 5)
        ),
        Permutation([1, 0])
    )
    assert cg1 == cg2


def test_arrayexpr_permute_tensor_product():
    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 1, 0, 5, 4, 6, 7]))
    cg2 = _array_tensor_product(N, _permute_dims(M, [1, 0]),
                                    _permute_dims(P, [1, 0]), Q)
    assert cg1 == cg2

    # TODO: reverse operation starting with `PermuteDims` and getting down to `bb`...
    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 4, 5, 0, 1, 6, 7]))
    cg2 = _array_tensor_product(N, P, M, Q)
    assert cg1 == cg2

    cg1 = _permute_dims(_array_tensor_product(M, N, P, Q), Permutation([2, 3, 4, 6, 5, 7, 0, 1]))
    assert cg1.expr == _array_tensor_product(N, P, Q, M)
    assert cg1.permutation == Permutation([0, 1, 2, 4, 3, 5, 6, 7])

    cg1 = _array_contraction(
        _permute_dims(
            _array_tensor_product(N, Q, Q, M),
            [2, 1, 5, 4, 0, 3, 6, 7]),
        [1, 2, 6])
    cg2 = _permute_dims(_array_contraction(_array_tensor_product(Q, Q, N, M), (3, 5, 6)), [0, 2, 3, 1, 4])
    assert cg1 == cg2

    cg1 = _array_contraction(
        _array_contraction(
            _array_contraction(
                _array_contraction(
                    _permute_dims(
                        _array_tensor_product(N, Q, Q, M),
                        [2, 1, 5, 4, 0, 3, 6, 7]),
                    [1, 2, 6]),
                [1, 3, 4]),
            [1]),
        [0])
    cg2 = _array_contraction(_array_tensor_product(M, N, Q, Q), (0, 3, 5), (1, 4, 7), (2,), (6,))
    assert cg1 == cg2


def test_arrayexpr_canonicalize_diagonal__permute_dims():
    tp = _array_tensor_product(M, Q, N, P)
    expr = _array_diagonal(
        _permute_dims(tp, [0, 1, 2, 4, 7, 6, 3, 5]), (2, 4, 5), (6, 7),
        (0, 3))
    result = _array_diagonal(tp, (2, 6, 7), (3, 5), (0, 4))
    assert expr == result

    tp = _array_tensor_product(M, N, P, Q)
    expr = _array_diagonal(_permute_dims(tp, [0, 5, 2, 4, 1, 6, 3, 7]), (1, 2, 6), (3, 4))
    result = _array_diagonal(_array_tensor_product(M, P, N, Q), (3, 4, 5), (1, 2))
    assert expr == result


def test_arrayexpr_canonicalize_diagonal_contraction():
    tp = _array_tensor_product(M, N, P, Q)
    expr = _array_contraction(_array_diagonal(tp, (1, 3, 4)), (0, 3))
    result = _array_diagonal(_array_contraction(_array_tensor_product(M, N, P, Q), (0, 6)), (0, 2, 3))
    assert expr == result

    expr = _array_contraction(_array_diagonal(tp, (0, 1, 2, 3, 7)), (1, 2, 3))
    result = _array_contraction(_array_tensor_product(M, N, P, Q), (0, 1, 2, 3, 5, 6, 7))
    assert expr == result

    expr = _array_contraction(_array_diagonal(tp, (0, 2, 6, 7)), (1, 2, 3))
    result = _array_diagonal(_array_contraction(tp, (3, 4, 5)), (0, 2, 3, 4))
    assert expr == result

    td = _array_diagonal(_array_tensor_product(M, N, P, Q), (0, 3))
    expr = _array_contraction(td, (2, 1), (0, 4, 6, 5, 3))
    result = _array_contraction(_array_tensor_product(M, N, P, Q), (0, 1, 3, 5, 6, 7), (2, 4))
    assert expr == result


def test_arrayexpr_array_wrong_permutation_size():
    cg = _array_tensor_product(M, N)
    raises(ValueError, lambda: _permute_dims(cg, [1, 0]))
    raises(ValueError, lambda: _permute_dims(cg, [1, 0, 2, 3, 5, 4]))


def test_arrayexpr_nested_array_elementwise_add():
    cg = _array_contraction(_array_add(
        _array_tensor_product(M, N),
        _array_tensor_product(N, M)
    ), (1, 2))
    result = _array_add(
        _array_contraction(_array_tensor_product(M, N), (1, 2)),
        _array_contraction(_array_tensor_product(N, M), (1, 2))
    )
    assert cg == result

    cg = _array_diagonal(_array_add(
        _array_tensor_product(M, N),
        _array_tensor_product(N, M)
    ), (1, 2))
    result = _array_add(
        _array_diagonal(_array_tensor_product(M, N), (1, 2)),
        _array_diagonal(_array_tensor_product(N, M), (1, 2))
    )
    assert cg == result


def test_arrayexpr_array_expr_zero_array():
    za1 = ZeroArray(k, l, m, n)
    zm1 = ZeroMatrix(m, n)

    za2 = ZeroArray(k, m, m, n)
    zm2 = ZeroMatrix(m, m)
    zm3 = ZeroMatrix(k, k)

    assert _array_tensor_product(M, N, za1) == ZeroArray(k, k, k, k, k, l, m, n)
    assert _array_tensor_product(M, N, zm1) == ZeroArray(k, k, k, k, m, n)

    assert _array_contraction(za1, (3,)) == ZeroArray(k, l, m)
    assert _array_contraction(zm1, (1,)) == ZeroArray(m)
    assert _array_contraction(za2, (1, 2)) == ZeroArray(k, n)
    assert _array_contraction(zm2, (0, 1)) == 0

    assert _array_diagonal(za2, (1, 2)) == ZeroArray(k, n, m)
    assert _array_diagonal(zm2, (0, 1)) == ZeroArray(m)

    assert _permute_dims(za1, [2, 1, 3, 0]) == ZeroArray(m, l, n, k)
    assert _permute_dims(zm1, [1, 0]) == ZeroArray(n, m)

    assert _array_add(za1) == za1
    assert _array_add(zm1) == ZeroArray(m, n)
    tp1 = _array_tensor_product(MatrixSymbol("A", k, l), MatrixSymbol("B", m, n))
    assert _array_add(tp1, za1) == tp1
    tp2 = _array_tensor_product(MatrixSymbol("C", k, l), MatrixSymbol("D", m, n))
    assert _array_add(tp1, za1, tp2) == _array_add(tp1, tp2)
    assert _array_add(M, zm3) == M
    assert _array_add(M, N, zm3) == _array_add(M, N)


def test_arrayexpr_array_expr_applyfunc():

    A = ArraySymbol("A", (3, k, 2))
    aaf = ArrayElementwiseApplyFunc(sin, A)
    assert aaf.shape == (3, k, 2)


def test_edit_array_contraction():
    cg = _array_contraction(_array_tensor_product(A, B, C, D), (1, 2, 5))
    ecg = _EditArrayContraction(cg)
    assert ecg.to_array_contraction() == cg

    ecg.args_with_ind[1], ecg.args_with_ind[2] = ecg.args_with_ind[2], ecg.args_with_ind[1]
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, C, B, D), (1, 3, 4))

    ci = ecg.get_new_contraction_index()
    new_arg = _ArgE(X)
    new_arg.indices = [ci, ci]
    ecg.args_with_ind.insert(2, new_arg)
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, C, X, B, D), (1, 3, 6), (4, 5))

    assert ecg.get_contraction_indices() == [[1, 3, 6], [4, 5]]
    assert [[tuple(j) for j in i] for i in ecg.get_contraction_indices_to_ind_rel_pos()] == [[(0, 1), (1, 1), (3, 0)], [(2, 0), (2, 1)]]
    assert [list(i) for i in ecg.get_mapping_for_index(0)] == [[0, 1], [1, 1], [3, 0]]
    assert [list(i) for i in ecg.get_mapping_for_index(1)] == [[2, 0], [2, 1]]
    raises(ValueError, lambda: ecg.get_mapping_for_index(2))

    ecg.args_with_ind.pop(1)
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, B, D), (1, 4), (2, 3))

    ecg.args_with_ind[0].indices[1] = ecg.args_with_ind[1].indices[0]
    ecg.args_with_ind[1].indices[1] = ecg.args_with_ind[2].indices[0]
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, B, D), (1, 2), (3, 4))

    ecg.insert_after(ecg.args_with_ind[1], _ArgE(C))
    assert ecg.to_array_contraction() == _array_contraction(_array_tensor_product(A, X, C, B, D), (1, 2), (3, 6))


def test_array_expressions_no_canonicalization():

    tp = _array_tensor_product(M, N, P)

    # ArrayTensorProduct:

    expr = ArrayTensorProduct(tp, N)
    assert str(expr) == "ArrayTensorProduct(ArrayTensorProduct(M, N, P), N)"
    assert expr.doit() == ArrayTensorProduct(M, N, P, N)

    expr = ArrayTensorProduct(ArrayContraction(M, (0, 1)), N)
    assert str(expr) == "ArrayTensorProduct(ArrayContraction(M, (0, 1)), N)"
    assert expr.doit() == ArrayContraction(ArrayTensorProduct(M, N), (0, 1))

    expr = ArrayTensorProduct(ArrayDiagonal(M, (0, 1)), N)
    assert str(expr) == "ArrayTensorProduct(ArrayDiagonal(M, (0, 1)), N)"
    assert expr.doit() == PermuteDims(ArrayDiagonal(ArrayTensorProduct(M, N), (0, 1)), [2, 0, 1])

    expr = ArrayTensorProduct(PermuteDims(M, [1, 0]), N)
    assert str(expr) == "ArrayTensorProduct(PermuteDims(M, (0 1)), N)"
    assert expr.doit() == PermuteDims(ArrayTensorProduct(M, N), [1, 0, 2, 3])

    # ArrayContraction:

    expr = ArrayContraction(_array_contraction(tp, (0, 2)), (0, 1))
    assert isinstance(expr, ArrayContraction)
    assert isinstance(expr.expr, ArrayContraction)
    assert str(expr) == "ArrayContraction(ArrayContraction(ArrayTensorProduct(M, N, P), (0, 2)), (0, 1))"
    assert expr.doit() == ArrayContraction(tp, (0, 2), (1, 3))

    expr = ArrayContraction(ArrayContraction(ArrayContraction(tp, (0, 1)), (0, 1)), (0, 1))
    assert expr.doit() == ArrayContraction(tp, (0, 1), (2, 3), (4, 5))
    # assert expr._canonicalize() == ArrayContraction(ArrayContraction(tp, (0, 1)), (0, 1), (2, 3))

    expr = ArrayContraction(ArrayDiagonal(tp, (0, 1)), (0, 1))
    assert str(expr) == "ArrayContraction(ArrayDiagonal(ArrayTensorProduct(M, N, P), (0, 1)), (0, 1))"
    assert expr.doit() == ArrayDiagonal(ArrayContraction(ArrayTensorProduct(N, M, P), (0, 1)), (0, 1))

    expr = ArrayContraction(PermuteDims(M, [1, 0]), (0, 1))
    assert str(expr) == "ArrayContraction(PermuteDims(M, (0 1)), (0, 1))"
    assert expr.doit() == ArrayContraction(M, (0, 1))

    # ArrayDiagonal:

    expr = ArrayDiagonal(ArrayDiagonal(tp, (0, 2)), (0, 1))
    assert str(expr) == "ArrayDiagonal(ArrayDiagonal(ArrayTensorProduct(M, N, P), (0, 2)), (0, 1))"
    assert expr.doit() == ArrayDiagonal(tp, (0, 2), (1, 3))

    expr = ArrayDiagonal(ArrayDiagonal(ArrayDiagonal(tp, (0, 1)), (0, 1)), (0, 1))
    assert expr.doit() == ArrayDiagonal(tp, (0, 1), (2, 3), (4, 5))
    assert expr._canonicalize() == expr.doit()

    expr = ArrayDiagonal(ArrayContraction(tp, (0, 1)), (0, 1))
    assert str(expr) == "ArrayDiagonal(ArrayContraction(ArrayTensorProduct(M, N, P), (0, 1)), (0, 1))"
    assert expr.doit() == expr

    expr = ArrayDiagonal(PermuteDims(M, [1, 0]), (0, 1))
    assert str(expr) == "ArrayDiagonal(PermuteDims(M, (0 1)), (0, 1))"
    assert expr.doit() == ArrayDiagonal(M, (0, 1))

    # ArrayAdd:

    expr = ArrayAdd(M)
    assert isinstance(expr, ArrayAdd)
    assert expr.doit() == M

    expr = ArrayAdd(ArrayAdd(M, N), P)
    assert str(expr) == "ArrayAdd(ArrayAdd(M, N), P)"
    assert expr.doit() == ArrayAdd(M, N, P)

    expr = ArrayAdd(M, ArrayAdd(N, ArrayAdd(P, M)))
    assert expr.doit() == ArrayAdd(M, N, P, M)
    assert expr._canonicalize() == ArrayAdd(M, N, ArrayAdd(P, M))

    expr = ArrayAdd(M, ZeroArray(k, k), N)
    assert str(expr) == "ArrayAdd(M, ZeroArray(k, k), N)"
    assert expr.doit() == ArrayAdd(M, N)

    # PermuteDims:

    expr = PermuteDims(PermuteDims(M, [1, 0]), [1, 0])
    assert str(expr) == "PermuteDims(PermuteDims(M, (0 1)), (0 1))"
    assert expr.doit() == M

    expr = PermuteDims(PermuteDims(PermuteDims(M, [1, 0]), [1, 0]), [1, 0])
    assert expr.doit() == PermuteDims(M, [1, 0])
    assert expr._canonicalize() == expr.doit()

    # Reshape

    expr = Reshape(A, (k**2,))
    assert expr.shape == (k**2,)
    assert isinstance(expr, Reshape)


def test_array_expr_construction_with_functions():

    tp = tensorproduct(M, N)
    assert tp == ArrayTensorProduct(M, N)

    expr = tensorproduct(A, eye(2))
    assert expr == ArrayTensorProduct(A, eye(2))

    # Contraction:

    expr = tensorcontraction(M, (0, 1))
    assert expr == ArrayContraction(M, (0, 1))

    expr = tensorcontraction(tp, (1, 2))
    assert expr == ArrayContraction(tp, (1, 2))

    expr = tensorcontraction(tensorcontraction(tp, (1, 2)), (0, 1))
    assert expr == ArrayContraction(tp, (0, 3), (1, 2))

    # Diagonalization:

    expr = tensordiagonal(M, (0, 1))
    assert expr == ArrayDiagonal(M, (0, 1))

    expr = tensordiagonal(tensordiagonal(tp, (0, 1)), (0, 1))
    assert expr == ArrayDiagonal(tp, (0, 1), (2, 3))

    # Permutation of dimensions:

    expr = permutedims(M, [1, 0])
    assert expr == PermuteDims(M, [1, 0])

    expr = permutedims(PermuteDims(tp, [1, 0, 2, 3]), [0, 1, 3, 2])
    assert expr == PermuteDims(tp, [1, 0, 3, 2])

    expr = PermuteDims(tp, index_order_new=["a", "b", "c", "d"], index_order_old=["d", "c", "b", "a"])
    assert expr == PermuteDims(tp, [3, 2, 1, 0])

    arr = Array(range(32)).reshape(2, 2, 2, 2, 2)
    expr = PermuteDims(arr, index_order_new=["a", "b", "c", "d", "e"], index_order_old=['b', 'e', 'a', 'd', 'c'])
    assert expr == PermuteDims(arr, [2, 0, 4, 3, 1])
    assert expr.as_explicit() == permutedims(arr, index_order_new=["a", "b", "c", "d", "e"], index_order_old=['b', 'e', 'a', 'd', 'c'])


def test_array_element_expressions():
    # Check commutative property:
    assert M[0, 0]*N[0, 0] == N[0, 0]*M[0, 0]

    # Check derivatives:
    assert M[0, 0].diff(M[0, 0]) == 1
    assert M[0, 0].diff(M[1, 0]) == 0
    assert M[0, 0].diff(N[0, 0]) == 0
    assert M[0, 1].diff(M[i, j]) == KroneckerDelta(i, 0)*KroneckerDelta(j, 1)
    assert M[0, 1].diff(N[i, j]) == 0

    K4 = ArraySymbol("K4", shape=(k, k, k, k))

    assert K4[i, j, k, l].diff(K4[1, 2, 3, 4]) == (
        KroneckerDelta(i, 1)*KroneckerDelta(j, 2)*KroneckerDelta(k, 3)*KroneckerDelta(l, 4)
    )


def test_array_expr_reshape():

    A = MatrixSymbol("A", 2, 2)
    B = ArraySymbol("B", (2, 2, 2))
    C = Array([1, 2, 3, 4])

    expr = Reshape(A, (4,))
    assert expr.expr == A
    assert expr.shape == (4,)
    assert expr.as_explicit() == Array([A[0, 0], A[0, 1], A[1, 0], A[1, 1]])

    expr = Reshape(B, (2, 4))
    assert expr.expr == B
    assert expr.shape == (2, 4)
    ee = expr.as_explicit()
    assert isinstance(ee, ImmutableDenseNDimArray)
    assert ee.shape == (2, 4)
    assert ee == Array([[B[0, 0, 0], B[0, 0, 1], B[0, 1, 0], B[0, 1, 1]], [B[1, 0, 0], B[1, 0, 1], B[1, 1, 0], B[1, 1, 1]]])

    expr = Reshape(A, (k, 2))
    assert expr.shape == (k, 2)

    raises(ValueError, lambda: Reshape(A, (2, 3)))
    raises(ValueError, lambda: Reshape(A, (3,)))

    expr = Reshape(C, (2, 2))
    assert expr.expr == C
    assert expr.shape == (2, 2)
    assert expr.doit() == Array([[1, 2], [3, 4]])


def test_array_expr_as_explicit_with_explicit_component_arrays():
    # Test if .as_explicit() works with explicit-component arrays
    # nested in array expressions:
    from sympy.abc import x, y, z, t
    A = Array([[x, y], [z, t]])
    assert ArrayTensorProduct(A, A).as_explicit() == tensorproduct(A, A)
    assert ArrayDiagonal(A, (0, 1)).as_explicit() == tensordiagonal(A, (0, 1))
    assert ArrayContraction(A, (0, 1)).as_explicit() == tensorcontraction(A, (0, 1))
    assert ArrayAdd(A, A).as_explicit() == A + A
    assert ArrayElementwiseApplyFunc(sin, A).as_explicit() == A.applyfunc(sin)
    assert PermuteDims(A, [1, 0]).as_explicit() == permutedims(A, [1, 0])
    assert Reshape(A, [4]).as_explicit() == A.reshape(4)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tests\test_bessel.py ===
from itertools import product

from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (conjugate, polar_lift)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely, hankel1, hankel2, hn1, hn2, jn, jn_zeros, yn)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.series.series import series
from sympy.functions.special.bessel import (airyai, airybi,
                                            airyaiprime, airybiprime, marcumq)
from sympy.core.random import (random_complex_number as randcplx,
                                      verify_numerically as tn,
                                      test_derivative_numerically as td,
                                      _randint)
from sympy.simplify import besselsimp
from sympy.testing.pytest import raises, slow

from sympy.abc import z, n, k, x

randint = _randint()


def test_bessel_rand():
    for f in [besselj, bessely, besseli, besselk, hankel1, hankel2]:
        assert td(f(randcplx(), z), z)

    for f in [jn, yn, hn1, hn2]:
        assert td(f(randint(-10, 10), z), z)


def test_bessel_twoinputs():
    for f in [besselj, bessely, besseli, besselk, hankel1, hankel2, jn, yn]:
        raises(TypeError, lambda: f(1))
        raises(TypeError, lambda: f(1, 2, 3))


def test_besselj_leading_term():
    assert besselj(0, x).as_leading_term(x) == 1
    assert besselj(1, sin(x)).as_leading_term(x) == x/2
    assert besselj(1, 2*sqrt(x)).as_leading_term(x) == sqrt(x)

    # https://github.com/sympy/sympy/issues/21701
    assert (besselj(z, x)/x**z).as_leading_term(x) == 1/(2**z*gamma(z + 1))


def test_bessely_leading_term():
    assert bessely(0, x).as_leading_term(x) == (2*log(x) - 2*log(2) + 2*S.EulerGamma)/pi
    assert bessely(1, sin(x)).as_leading_term(x) == -2/(pi*x)
    assert bessely(1, 2*sqrt(x)).as_leading_term(x) == -1/(pi*sqrt(x))


def test_besseli_leading_term():
    assert besseli(0, x).as_leading_term(x) == 1
    assert besseli(1, sin(x)).as_leading_term(x) == x/2
    assert besseli(1, 2*sqrt(x)).as_leading_term(x) == sqrt(x)


def test_besselk_leading_term():
    assert besselk(0, x).as_leading_term(x) == -log(x) - S.EulerGamma + log(2)
    assert besselk(1, sin(x)).as_leading_term(x) == 1/x
    assert besselk(1, 2*sqrt(x)).as_leading_term(x) == 1/(2*sqrt(x))
    assert besselk(S(5)/3, x).as_leading_term(x) == 2**(S(2)/3)*gamma(S(5)/3)/x**(S(5)/3)
    assert besselk(S(2)/3, x).as_leading_term(x) == besselk(-S(2)/3, x).as_leading_term(x)
    assert besselk(1,cos(x)).as_leading_term(x) == besselk(1,1)
    assert besselk(3,1/x).as_leading_term(x) == sqrt(pi)*exp(-(1/x))/sqrt(2/x)
    assert besselk(3,1/sin(x)).as_leading_term(x) == sqrt(pi)*exp(-(1/x))/sqrt(2/x)

    nz = Symbol("nz", nonzero=True)
    assert besselk(nz, x).as_leading_term(x).subs({nz:S(5)/7}) == besselk(S(5)/7, x).series(x).as_leading_term(x)
    assert besselk(nz, x).as_leading_term(x).subs({nz:S(-15)/7}) == besselk(S(-15)/7, x).series(x).as_leading_term(x)
    assert besselk(nz, x).as_leading_term(x).subs({nz:3}) == besselk(3, x).series(x).as_leading_term(x)
    assert besselk(nz, x).as_leading_term(x).subs({nz:-2}) == besselk(-2, x).series(x).as_leading_term(x)


def test_besselj_series():
    assert besselj(0, x).series(x) == 1 - x**2/4 + x**4/64 + O(x**6)
    assert besselj(0, x**(1.1)).series(x) == 1 + x**4.4/64 - x**2.2/4 + O(x**6)
    assert besselj(0, x**2 + x).series(x) == 1 - x**2/4 - x**3/2\
        - 15*x**4/64 + x**5/16 + O(x**6)
    assert besselj(0, sqrt(x) + x).series(x, n=4) == 1 - x/4 - 15*x**2/64\
        + 215*x**3/2304 - x**Rational(3, 2)/2 + x**Rational(5, 2)/16\
        + 23*x**Rational(7, 2)/384 + O(x**4)
    assert besselj(0, x/(1 - x)).series(x) == 1 - x**2/4 - x**3/2 - 47*x**4/64\
        - 15*x**5/16 + O(x**6)
    assert besselj(0, log(1 + x)).series(x) == 1 - x**2/4 + x**3/4\
        - 41*x**4/192 + 17*x**5/96 + O(x**6)
    assert besselj(1, sin(x)).series(x) == x/2 - 7*x**3/48 + 73*x**5/1920 + O(x**6)
    assert besselj(1, 2*sqrt(x)).series(x) == sqrt(x) - x**Rational(3, 2)/2\
        + x**Rational(5, 2)/12 - x**Rational(7, 2)/144 + x**Rational(9, 2)/2880\
        - x**Rational(11, 2)/86400 + O(x**6)
    assert besselj(-2, sin(x)).series(x, n=4) == besselj(2, sin(x)).series(x, n=4)


def test_bessely_series():
    const = 2*S.EulerGamma/pi - 2*log(2)/pi + 2*log(x)/pi
    assert bessely(0, x).series(x, n=4) == const + x**2*(-log(x)/(2*pi)\
        + (2 - 2*S.EulerGamma)/(4*pi) + log(2)/(2*pi)) + O(x**4*log(x))
    assert bessely(1, x).series(x, n=4) == -2/(pi*x) + x*(log(x)/pi - log(2)/pi - \
        (1 - 2*S.EulerGamma)/(2*pi)) + x**3*(-log(x)/(8*pi) + \
        (S(5)/2 - 2*S.EulerGamma)/(16*pi) + log(2)/(8*pi)) + O(x**4*log(x))
    assert bessely(2, x).series(x, n=4) == -4/(pi*x**2) - 1/pi + x**2*(log(x)/(4*pi) - \
        log(2)/(4*pi) - (S(3)/2 - 2*S.EulerGamma)/(8*pi)) + O(x**4*log(x))
    assert bessely(3, x).series(x, n=4) == -16/(pi*x**3) - 2/(pi*x) - \
        x/(4*pi) + x**3*(log(x)/(24*pi) - log(2)/(24*pi) - \
        (S(11)/6 - 2*S.EulerGamma)/(48*pi)) + O(x**4*log(x))
    assert bessely(0, x**(1.1)).series(x, n=4) == 2*S.EulerGamma/pi\
        - 2*log(2)/pi + 2.2*log(x)/pi + x**2.2*(-0.55*log(x)/pi\
        + (2 - 2*S.EulerGamma)/(4*pi) + log(2)/(2*pi)) + O(x**4*log(x))
    assert bessely(0, x**2 + x).series(x, n=4) == \
        const - (2 - 2*S.EulerGamma)*(-x**3/(2*pi) - x**2/(4*pi)) + 2*x/pi\
        + x**2*(-log(x)/(2*pi) - 1/pi + log(2)/(2*pi))\
        + x**3*(-log(x)/pi + 1/(6*pi) + log(2)/pi) + O(x**4*log(x))
    assert bessely(0, x/(1 - x)).series(x, n=3) == const\
        + 2*x/pi + x**2*(-log(x)/(2*pi) + (2 - 2*S.EulerGamma)/(4*pi)\
        + log(2)/(2*pi) + 1/pi) + O(x**3*log(x))
    assert bessely(0, log(1 + x)).series(x, n=3) == const\
        - x/pi + x**2*(-log(x)/(2*pi) + (2 - 2*S.EulerGamma)/(4*pi)\
        + log(2)/(2*pi) + 5/(12*pi)) + O(x**3*log(x))
    assert bessely(1, sin(x)).series(x, n=4) == -1/(pi*(-x**3/12 + x/2)) - \
        (1 - 2*S.EulerGamma)*(-x**3/12 + x/2)/pi + x*(log(x)/pi - log(2)/pi) + \
        x**3*(-7*log(x)/(24*pi) - 1/(6*pi) + (S(5)/2 - 2*S.EulerGamma)/(16*pi) +
        7*log(2)/(24*pi)) + O(x**4*log(x))
    assert bessely(1, 2*sqrt(x)).series(x, n=3) == -1/(pi*sqrt(x)) + \
        sqrt(x)*(log(x)/pi - (1 - 2*S.EulerGamma)/pi) + x**(S(3)/2)*(-log(x)/(2*pi) + \
        (S(5)/2 - 2*S.EulerGamma)/(2*pi)) + x**(S(5)/2)*(log(x)/(12*pi) - \
        (S(10)/3 - 2*S.EulerGamma)/(12*pi)) + O(x**3*log(x))
    assert bessely(-2, sin(x)).series(x, n=4) == bessely(2, sin(x)).series(x, n=4)


def test_besseli_series():
    assert besseli(0, x).series(x) == 1 + x**2/4 + x**4/64 + O(x**6)
    assert besseli(0, x**(1.1)).series(x) == 1 + x**4.4/64 + x**2.2/4 + O(x**6)
    assert besseli(0, x**2 + x).series(x) == 1 + x**2/4 + x**3/2 + 17*x**4/64 + \
        x**5/16 + O(x**6)
    assert besseli(0, sqrt(x) + x).series(x, n=4) == 1 + x/4 + 17*x**2/64 + \
        217*x**3/2304 + x**(S(3)/2)/2 + x**(S(5)/2)/16 + 25*x**(S(7)/2)/384 + O(x**4)
    assert besseli(0, x/(1 - x)).series(x) == 1 + x**2/4 + x**3/2 + 49*x**4/64 + \
        17*x**5/16 + O(x**6)
    assert besseli(0, log(1 + x)).series(x) == 1 + x**2/4 - x**3/4 + 47*x**4/192 - \
        23*x**5/96 + O(x**6)
    assert besseli(1, sin(x)).series(x) == x/2 - x**3/48 - 47*x**5/1920 + O(x**6)
    assert besseli(1, 2*sqrt(x)).series(x) == sqrt(x) + x**(S(3)/2)/2 + x**(S(5)/2)/12 + \
        x**(S(7)/2)/144 + x**(S(9)/2)/2880 + x**(S(11)/2)/86400 + O(x**6)
    assert besseli(-2, sin(x)).series(x, n=4) == besseli(2, sin(x)).series(x, n=4)

    #test for aseries
    assert besseli(0,x).series(x, oo, n=4) == sqrt(2)*(sqrt(1/x) - (1/x)**(S(3)/2)/8 - \
        3*(1/x)**(S(5)/2)/128 - 15*(1/x)**(S(7)/2)/1024 + O((1/x)**(S(9)/2), (x, oo)))*exp(x)/(2*sqrt(pi))
    assert besseli(0,x).series(x,-oo, n=4) == sqrt(2)*(sqrt(-1/x) - (-1/x)**(S(3)/2)/8 - 3*(-1/x)**(S(5)/2)/128 - \
        15*(-1/x)**(S(7)/2)/1024 + O((-1/x)**(S(9)/2), (x, -oo)))*exp(-x)/(2*sqrt(pi))


def test_besselk_series():
    const = log(2) - S.EulerGamma - log(x)
    assert besselk(0, x).series(x, n=4) == const + \
        x**2*(-log(x)/4 - S.EulerGamma/4 + log(2)/4 + S(1)/4) + O(x**4*log(x))
    assert besselk(1, x).series(x, n=4) == 1/x + x*(log(x)/2 - log(2)/2 - \
        S(1)/4 + S.EulerGamma/2) + x**3*(log(x)/16 - S(5)/64 - log(2)/16 + \
        S.EulerGamma/16) + O(x**4*log(x))
    assert besselk(2, x).series(x, n=4) == 2/x**2 - S(1)/2 + x**2*(-log(x)/8 - \
        S.EulerGamma/8 + log(2)/8 + S(3)/32) + O(x**4*log(x))
    assert besselk(2, x).series(x, n=1) == 2/x**2 - S(1)/2 + O(x) #edge case for series truncation
    assert besselk(0, x**(1.1)).series(x, n=4) == log(2) - S.EulerGamma - \
        1.1*log(x) + x**2.2*(-0.275*log(x) - S.EulerGamma/4 + \
        log(2)/4 + S(1)/4) + O(x**4*log(x))
    assert besselk(0, x**2 + x).series(x, n=4) == const + \
        (2 - 2*S.EulerGamma)*(x**3/4 + x**2/8) - x + x**2*(-log(x)/4 + \
        log(2)/4 + S(1)/2) + x**3*(-log(x)/2 - S(7)/12 + log(2)/2) + O(x**4*log(x))
    assert besselk(0, x/(1 - x)).series(x, n=3) == const - x + x**2*(-log(x)/4 - \
        S(1)/4 - S.EulerGamma/4 + log(2)/4) + O(x**3*log(x))
    assert besselk(0, log(1 + x)).series(x, n=3) == const + x/2 + \
        x**2*(-log(x)/4 - S.EulerGamma/4 + S(1)/24 + log(2)/4) + O(x**3*log(x))
    assert besselk(1, 2*sqrt(x)).series(x, n=3) == 1/(2*sqrt(x)) + \
        sqrt(x)*(log(x)/2 - S(1)/2 + S.EulerGamma) + x**(S(3)/2)*(log(x)/4 - S(5)/8 + \
        S.EulerGamma/2) + x**(S(5)/2)*(log(x)/24 - S(5)/36 + S.EulerGamma/12) + O(x**3*log(x))
    assert besselk(-2, sin(x)).series(x, n=4) == besselk(2, sin(x)).series(x, n=4)
    assert besselk(2, x**2).series(x, n=2) == 2/x**4 - S(1)/2 + O(x**2) #edge case for series truncation
    assert besselk(2, x**2).series(x, n=6) == 2/x**4 - S(1)/2 + x**4*(-log(x)/4 - S.EulerGamma/8 + log(2)/8 + S(3)/32) + O(x**6*log(x))
    assert (x**2*besselk(2, x)).series(x, n=2) == 2 + O(x**2)

    #test for aseries
    assert besselk(0,x).series(x, oo, n=4) == sqrt(2)*sqrt(pi)*(sqrt(1/x) + (1/x)**(S(3)/2)/8 - \
            3*(1/x)**(S(5)/2)/128 + 15*(1/x)**(S(7)/2)/1024 + O((1/x)**(S(9)/2), (x, oo)))*exp(-x)/2
    assert besselk(0,x).series(x, -oo, n=4) == sqrt(2)*sqrt(pi)*(-I*sqrt(-1/x) + I*(-1/x)**(S(3)/2)/8 + \
            3*I*(-1/x)**(S(5)/2)/128 + 15*I*(-1/x)**(S(7)/2)/1024 + O((-1/x)**(S(9)/2), (x, -oo)))*exp(-x)/2


def test_besselk_frac_order_series():
    assert besselk(S(5)/3, x).series(x, n=2) == 2**(S(2)/3)*gamma(S(5)/3)/x**(S(5)/3) - \
        3*gamma(S(5)/3)*x**(S(1)/3)/(4*2**(S(1)/3)) + \
            gamma(-S(5)/3)*x**(S(5)/3)/(4*2**(S(2)/3)) + O(x**2)
    assert besselk(S(1)/2, x).series(x, n=2) == sqrt(pi/2)/sqrt(x) - \
        sqrt(pi*x/2) + x**(S(3)/2)*sqrt(pi/2)/2 + O(x**2)
    assert besselk(S(1)/2, sqrt(x)).series(x, n=2) == sqrt(pi/2)/x**(S(1)/4) - \
        sqrt(pi/2)*x**(S(1)/4) + sqrt(pi/2)*x**(S(3)/4)/2 - \
            sqrt(pi/2)*x**(S(5)/4)/6 + sqrt(pi/2)*x**(S(7)/4)/24 + O(x**2)
    assert besselk(S(1)/2, x**2).series(x, n=2) == sqrt(pi/2)/x \
        - sqrt(pi/2)*x + O(x**2)
    assert besselk(-S(1)/2, x).series(x) == besselk(S(1)/2, x).series(x)
    assert besselk(-S(7)/6, x).series(x) == besselk(S(7)/6, x).series(x)


def test_diff():
    assert besselj(n, z).diff(z) == besselj(n - 1, z)/2 - besselj(n + 1, z)/2
    assert bessely(n, z).diff(z) == bessely(n - 1, z)/2 - bessely(n + 1, z)/2
    assert besseli(n, z).diff(z) == besseli(n - 1, z)/2 + besseli(n + 1, z)/2
    assert besselk(n, z).diff(z) == -besselk(n - 1, z)/2 - besselk(n + 1, z)/2
    assert hankel1(n, z).diff(z) == hankel1(n - 1, z)/2 - hankel1(n + 1, z)/2
    assert hankel2(n, z).diff(z) == hankel2(n - 1, z)/2 - hankel2(n + 1, z)/2


def test_rewrite():
    assert besselj(n, z).rewrite(jn) == sqrt(2*z/pi)*jn(n - S.Half, z)
    assert bessely(n, z).rewrite(yn) == sqrt(2*z/pi)*yn(n - S.Half, z)
    assert besseli(n, z).rewrite(besselj) == \
        exp(-I*n*pi/2)*besselj(n, polar_lift(I)*z)
    assert besselj(n, z).rewrite(besseli) == \
        exp(I*n*pi/2)*besseli(n, polar_lift(-I)*z)

    nu = randcplx()

    assert tn(besselj(nu, z), besselj(nu, z).rewrite(besseli), z)
    assert tn(besselj(nu, z), besselj(nu, z).rewrite(bessely), z)

    assert tn(besseli(nu, z), besseli(nu, z).rewrite(besselj), z)
    assert tn(besseli(nu, z), besseli(nu, z).rewrite(bessely), z)

    assert tn(bessely(nu, z), bessely(nu, z).rewrite(besselj), z)
    assert tn(bessely(nu, z), bessely(nu, z).rewrite(besseli), z)

    assert tn(besselk(nu, z), besselk(nu, z).rewrite(besselj), z)
    assert tn(besselk(nu, z), besselk(nu, z).rewrite(besseli), z)
    assert tn(besselk(nu, z), besselk(nu, z).rewrite(bessely), z)

    # check that a rewrite was triggered, when the order is set to a generic
    # symbol 'nu'
    assert yn(nu, z) != yn(nu, z).rewrite(jn)
    assert hn1(nu, z) != hn1(nu, z).rewrite(jn)
    assert hn2(nu, z) != hn2(nu, z).rewrite(jn)
    assert jn(nu, z) != jn(nu, z).rewrite(yn)
    assert hn1(nu, z) != hn1(nu, z).rewrite(yn)
    assert hn2(nu, z) != hn2(nu, z).rewrite(yn)

    # rewriting spherical bessel functions (SBFs) w.r.t. besselj, bessely is
    # not allowed if a generic symbol 'nu' is used as the order of the SBFs
    # to avoid inconsistencies (the order of bessel[jy] is allowed to be
    # complex-valued, whereas SBFs are defined only for integer orders)
    order = nu
    for f in (besselj, bessely):
        assert hn1(order, z) == hn1(order, z).rewrite(f)
        assert hn2(order, z) == hn2(order, z).rewrite(f)

    assert jn(order, z).rewrite(besselj) == sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(order + S.Half, z)/2
    assert jn(order, z).rewrite(bessely) == (-1)**nu*sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(-order - S.Half, z)/2

    # for integral orders rewriting SBFs w.r.t bessel[jy] is allowed
    N = Symbol('n', integer=True)
    ri = randint(-11, 10)
    for order in (ri, N):
        for f in (besselj, bessely):
            assert yn(order, z) != yn(order, z).rewrite(f)
            assert jn(order, z) != jn(order, z).rewrite(f)
            assert hn1(order, z) != hn1(order, z).rewrite(f)
            assert hn2(order, z) != hn2(order, z).rewrite(f)

    for func, refunc in product((yn, jn, hn1, hn2),
                                (jn, yn, besselj, bessely)):
        assert tn(func(ri, z), func(ri, z).rewrite(refunc), z)


def test_expand():
    assert expand_func(besselj(S.Half, z).rewrite(jn)) == \
        sqrt(2)*sin(z)/(sqrt(pi)*sqrt(z))
    assert expand_func(bessely(S.Half, z).rewrite(yn)) == \
        -sqrt(2)*cos(z)/(sqrt(pi)*sqrt(z))

    # XXX: teach sin/cos to work around arguments like
    # x*exp_polar(I*pi*n/2).  Then change besselsimp -> expand_func
    assert besselsimp(besselj(S.Half, z)) == sqrt(2)*sin(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(besselj(Rational(-1, 2), z)) == sqrt(2)*cos(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(besselj(Rational(5, 2), z)) == \
        -sqrt(2)*(z**2*sin(z) + 3*z*cos(z) - 3*sin(z))/(sqrt(pi)*z**Rational(5, 2))
    assert besselsimp(besselj(Rational(-5, 2), z)) == \
        -sqrt(2)*(z**2*cos(z) - 3*z*sin(z) - 3*cos(z))/(sqrt(pi)*z**Rational(5, 2))

    assert besselsimp(bessely(S.Half, z)) == \
        -(sqrt(2)*cos(z))/(sqrt(pi)*sqrt(z))
    assert besselsimp(bessely(Rational(-1, 2), z)) == sqrt(2)*sin(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(bessely(Rational(5, 2), z)) == \
        sqrt(2)*(z**2*cos(z) - 3*z*sin(z) - 3*cos(z))/(sqrt(pi)*z**Rational(5, 2))
    assert besselsimp(bessely(Rational(-5, 2), z)) == \
        -sqrt(2)*(z**2*sin(z) + 3*z*cos(z) - 3*sin(z))/(sqrt(pi)*z**Rational(5, 2))

    assert besselsimp(besseli(S.Half, z)) == sqrt(2)*sinh(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(besseli(Rational(-1, 2), z)) == \
        sqrt(2)*cosh(z)/(sqrt(pi)*sqrt(z))
    assert besselsimp(besseli(Rational(5, 2), z)) == \
        sqrt(2)*(z**2*sinh(z) - 3*z*cosh(z) + 3*sinh(z))/(sqrt(pi)*z**Rational(5, 2))
    assert besselsimp(besseli(Rational(-5, 2), z)) == \
        sqrt(2)*(z**2*cosh(z) - 3*z*sinh(z) + 3*cosh(z))/(sqrt(pi)*z**Rational(5, 2))

    assert besselsimp(besselk(S.Half, z)) == \
        besselsimp(besselk(Rational(-1, 2), z)) == sqrt(pi)*exp(-z)/(sqrt(2)*sqrt(z))
    assert besselsimp(besselk(Rational(5, 2), z)) == \
        besselsimp(besselk(Rational(-5, 2), z)) == \
        sqrt(2)*sqrt(pi)*(z**2 + 3*z + 3)*exp(-z)/(2*z**Rational(5, 2))

    n = Symbol('n', integer=True, positive=True)

    assert expand_func(besseli(n + 2, z)) == \
        besseli(n, z) + (-2*n - 2)*(-2*n*besseli(n, z)/z + besseli(n - 1, z))/z
    assert expand_func(besselj(n + 2, z)) == \
        -besselj(n, z) + (2*n + 2)*(2*n*besselj(n, z)/z - besselj(n - 1, z))/z
    assert expand_func(besselk(n + 2, z)) == \
        besselk(n, z) + (2*n + 2)*(2*n*besselk(n, z)/z + besselk(n - 1, z))/z
    assert expand_func(bessely(n + 2, z)) == \
        -bessely(n, z) + (2*n + 2)*(2*n*bessely(n, z)/z - bessely(n - 1, z))/z

    assert expand_func(besseli(n + S.Half, z).rewrite(jn)) == \
        (sqrt(2)*sqrt(z)*exp(-I*pi*(n + S.Half)/2) *
         exp_polar(I*pi/4)*jn(n, z*exp_polar(I*pi/2))/sqrt(pi))
    assert expand_func(besselj(n + S.Half, z).rewrite(jn)) == \
        sqrt(2)*sqrt(z)*jn(n, z)/sqrt(pi)

    r = Symbol('r', real=True)
    p = Symbol('p', positive=True)
    i = Symbol('i', integer=True)

    for besselx in [besselj, bessely, besseli, besselk]:
        assert besselx(i, p).is_extended_real is True
        assert besselx(i, x).is_extended_real is None
        assert besselx(x, z).is_extended_real is None

    for besselx in [besselj, besseli]:
        assert besselx(i, r).is_extended_real is True
    for besselx in [bessely, besselk]:
        assert besselx(i, r).is_extended_real is None

    for besselx in [besselj, bessely, besseli, besselk]:
        assert expand_func(besselx(oo, x)) == besselx(oo, x, evaluate=False)
        assert expand_func(besselx(-oo, x)) == besselx(-oo, x, evaluate=False)


# Quite varying time, but often really slow
@slow
def test_slow_expand():
    def check(eq, ans):
        return tn(eq, ans) and eq == ans

    rn = randcplx(a=1, b=0, d=0, c=2)

    for besselx in [besselj, bessely, besseli, besselk]:
        ri = S(2*randint(-11, 10) + 1) / 2  # half integer in [-21/2, 21/2]
        assert tn(besselsimp(besselx(ri, z)), besselx(ri, z))

    assert check(expand_func(besseli(rn, x)),
                 besseli(rn - 2, x) - 2*(rn - 1)*besseli(rn - 1, x)/x)
    assert check(expand_func(besseli(-rn, x)),
                 besseli(-rn + 2, x) + 2*(-rn + 1)*besseli(-rn + 1, x)/x)

    assert check(expand_func(besselj(rn, x)),
                 -besselj(rn - 2, x) + 2*(rn - 1)*besselj(rn - 1, x)/x)
    assert check(expand_func(besselj(-rn, x)),
                 -besselj(-rn + 2, x) + 2*(-rn + 1)*besselj(-rn + 1, x)/x)

    assert check(expand_func(besselk(rn, x)),
                 besselk(rn - 2, x) + 2*(rn - 1)*besselk(rn - 1, x)/x)
    assert check(expand_func(besselk(-rn, x)),
                 besselk(-rn + 2, x) - 2*(-rn + 1)*besselk(-rn + 1, x)/x)

    assert check(expand_func(bessely(rn, x)),
                 -bessely(rn - 2, x) + 2*(rn - 1)*bessely(rn - 1, x)/x)
    assert check(expand_func(bessely(-rn, x)),
                 -bessely(-rn + 2, x) + 2*(-rn + 1)*bessely(-rn + 1, x)/x)


def mjn(n, z):
    return expand_func(jn(n, z))


def myn(n, z):
    return expand_func(yn(n, z))


def test_jn():
    z = symbols("z")
    assert jn(0, 0) == 1
    assert jn(1, 0) == 0
    assert jn(-1, 0) == S.ComplexInfinity
    assert jn(z, 0) == jn(z, 0, evaluate=False)
    assert jn(0, oo) == 0
    assert jn(0, -oo) == 0

    assert mjn(0, z) == sin(z)/z
    assert mjn(1, z) == sin(z)/z**2 - cos(z)/z
    assert mjn(2, z) == (3/z**3 - 1/z)*sin(z) - (3/z**2) * cos(z)
    assert mjn(3, z) == (15/z**4 - 6/z**2)*sin(z) + (1/z - 15/z**3)*cos(z)
    assert mjn(4, z) == (1/z + 105/z**5 - 45/z**3)*sin(z) + \
        (-105/z**4 + 10/z**2)*cos(z)
    assert mjn(5, z) == (945/z**6 - 420/z**4 + 15/z**2)*sin(z) + \
        (-1/z - 945/z**5 + 105/z**3)*cos(z)
    assert mjn(6, z) == (-1/z + 10395/z**7 - 4725/z**5 + 210/z**3)*sin(z) + \
        (-10395/z**6 + 1260/z**4 - 21/z**2)*cos(z)

    assert expand_func(jn(n, z)) == jn(n, z)

    # SBFs not defined for complex-valued orders
    assert jn(2+3j, 5.2+0.3j).evalf() == jn(2+3j, 5.2+0.3j)

    assert eq([jn(2, 5.2+0.3j).evalf(10)],
              [0.09941975672 - 0.05452508024*I])


def test_yn():
    z = symbols("z")
    assert myn(0, z) == -cos(z)/z
    assert myn(1, z) == -cos(z)/z**2 - sin(z)/z
    assert myn(2, z) == -((3/z**3 - 1/z)*cos(z) + (3/z**2)*sin(z))
    assert expand_func(yn(n, z)) == yn(n, z)

    # SBFs not defined for complex-valued orders
    assert yn(2+3j, 5.2+0.3j).evalf() == yn(2+3j, 5.2+0.3j)

    assert eq([yn(2, 5.2+0.3j).evalf(10)],
              [0.185250342 + 0.01489557397*I])


def test_sympify_yn():
    assert S(15) in myn(3, pi).atoms()
    assert myn(3, pi) == 15/pi**4 - 6/pi**2


def eq(a, b, tol=1e-6):
    for u, v in zip(a, b):
        if not (abs(u - v) < tol):
            return False
    return True


def test_jn_zeros():
    assert eq(jn_zeros(0, 4), [3.141592, 6.283185, 9.424777, 12.566370])
    assert eq(jn_zeros(1, 4), [4.493409, 7.725251, 10.904121, 14.066193])
    assert eq(jn_zeros(2, 4), [5.763459, 9.095011, 12.322940, 15.514603])
    assert eq(jn_zeros(3, 4), [6.987932, 10.417118, 13.698023, 16.923621])
    assert eq(jn_zeros(4, 4), [8.182561, 11.704907, 15.039664, 18.301255])


def test_bessel_eval():
    n, m, k = Symbol('n', integer=True), Symbol('m'), Symbol('k', integer=True, zero=False)

    for f in [besselj, besseli]:
        assert f(0, 0) is S.One
        assert f(2.1, 0) is S.Zero
        assert f(-3, 0) is S.Zero
        assert f(-10.2, 0) is S.ComplexInfinity
        assert f(1 + 3*I, 0) is S.Zero
        assert f(-3 + I, 0) is S.ComplexInfinity
        assert f(-2*I, 0) is S.NaN
        assert f(n, 0) != S.One and f(n, 0) != S.Zero
        assert f(m, 0) != S.One and f(m, 0) != S.Zero
        assert f(k, 0) is S.Zero

    assert bessely(0, 0) is S.NegativeInfinity
    assert besselk(0, 0) is S.Infinity
    for f in [bessely, besselk]:
        assert f(1 + I, 0) is S.ComplexInfinity
        assert f(I, 0) is S.NaN

    for f in [besselj, bessely]:
        assert f(m, S.Infinity) is S.Zero
        assert f(m, S.NegativeInfinity) is S.Zero

    for f in [besseli, besselk]:
        assert f(m, I*S.Infinity) is S.Zero
        assert f(m, I*S.NegativeInfinity) is S.Zero

    for f in [besseli, besselk]:
        assert f(-4, z) == f(4, z)
        assert f(-3, z) == f(3, z)
        assert f(-n, z) == f(n, z)
        assert f(-m, z) != f(m, z)

    for f in [besselj, bessely]:
        assert f(-4, z) == f(4, z)
        assert f(-3, z) == -f(3, z)
        assert f(-n, z) == (-1)**n*f(n, z)
        assert f(-m, z) != (-1)**m*f(m, z)

    for f in [besselj, besseli]:
        assert f(m, -z) == (-z)**m*z**(-m)*f(m, z)

    assert besseli(2, -z) == besseli(2, z)
    assert besseli(3, -z) == -besseli(3, z)

    assert besselj(0, -z) == besselj(0, z)
    assert besselj(1, -z) == -besselj(1, z)

    assert besseli(0, I*z) == besselj(0, z)
    assert besseli(1, I*z) == I*besselj(1, z)
    assert besselj(3, I*z) == -I*besseli(3, z)


def test_bessel_nan():
    # FIXME: could have these return NaN; for now just fix infinite recursion
    for f in [besselj, bessely, besseli, besselk, hankel1, hankel2, yn, jn]:
        assert f(1, S.NaN) == f(1, S.NaN, evaluate=False)


def test_meromorphic():
    assert besselj(2, x).is_meromorphic(x, 1) == True
    assert besselj(2, x).is_meromorphic(x, 0) == True
    assert besselj(2, x).is_meromorphic(x, oo) == False
    assert besselj(S(2)/3, x).is_meromorphic(x, 1) == True
    assert besselj(S(2)/3, x).is_meromorphic(x, 0) == False
    assert besselj(S(2)/3, x).is_meromorphic(x, oo) == False
    assert besselj(x, 2*x).is_meromorphic(x, 2) == False
    assert besselk(0, x).is_meromorphic(x, 1) == True
    assert besselk(2, x).is_meromorphic(x, 0) == True
    assert besseli(0, x).is_meromorphic(x, 1) == True
    assert besseli(2, x).is_meromorphic(x, 0) == True
    assert bessely(0, x).is_meromorphic(x, 1) == True
    assert bessely(0, x).is_meromorphic(x, 0) == False
    assert bessely(2, x).is_meromorphic(x, 0) == True
    assert hankel1(3, x**2 + 2*x).is_meromorphic(x, 1) == True
    assert hankel1(0, x).is_meromorphic(x, 0) == False
    assert hankel2(11, 4).is_meromorphic(x, 5) == True
    assert hn1(6, 7*x**3 + 4).is_meromorphic(x, 7) == True
    assert hn2(3, 2*x).is_meromorphic(x, 9) == True
    assert jn(5, 2*x + 7).is_meromorphic(x, 4) == True
    assert yn(8, x**2 + 11).is_meromorphic(x, 6) == True


def test_conjugate():
    n = Symbol('n')
    z = Symbol('z', extended_real=False)
    x = Symbol('x', extended_real=True)
    y = Symbol('y', positive=True)
    t = Symbol('t', negative=True)

    for f in [besseli, besselj, besselk, bessely, hankel1, hankel2]:
        assert f(n, -1).conjugate() != f(conjugate(n), -1)
        assert f(n, x).conjugate() != f(conjugate(n), x)
        assert f(n, t).conjugate() != f(conjugate(n), t)

    rz = randcplx(b=0.5)

    for f in [besseli, besselj, besselk, bessely]:
        assert f(n, 1 + I).conjugate() == f(conjugate(n), 1 - I)
        assert f(n, 0).conjugate() == f(conjugate(n), 0)
        assert f(n, 1).conjugate() == f(conjugate(n), 1)
        assert f(n, z).conjugate() == f(conjugate(n), conjugate(z))
        assert f(n, y).conjugate() == f(conjugate(n), y)
        assert tn(f(n, rz).conjugate(), f(conjugate(n), conjugate(rz)))

    assert hankel1(n, 1 + I).conjugate() == hankel2(conjugate(n), 1 - I)
    assert hankel1(n, 0).conjugate() == hankel2(conjugate(n), 0)
    assert hankel1(n, 1).conjugate() == hankel2(conjugate(n), 1)
    assert hankel1(n, y).conjugate() == hankel2(conjugate(n), y)
    assert hankel1(n, z).conjugate() == hankel2(conjugate(n), conjugate(z))
    assert tn(hankel1(n, rz).conjugate(), hankel2(conjugate(n), conjugate(rz)))

    assert hankel2(n, 1 + I).conjugate() == hankel1(conjugate(n), 1 - I)
    assert hankel2(n, 0).conjugate() == hankel1(conjugate(n), 0)
    assert hankel2(n, 1).conjugate() == hankel1(conjugate(n), 1)
    assert hankel2(n, y).conjugate() == hankel1(conjugate(n), y)
    assert hankel2(n, z).conjugate() == hankel1(conjugate(n), conjugate(z))
    assert tn(hankel2(n, rz).conjugate(), hankel1(conjugate(n), conjugate(rz)))


def test_branching():
    assert besselj(polar_lift(k), x) == besselj(k, x)
    assert besseli(polar_lift(k), x) == besseli(k, x)

    n = Symbol('n', integer=True)
    assert besselj(n, exp_polar(2*pi*I)*x) == besselj(n, x)
    assert besselj(n, polar_lift(x)) == besselj(n, x)
    assert besseli(n, exp_polar(2*pi*I)*x) == besseli(n, x)
    assert besseli(n, polar_lift(x)) == besseli(n, x)

    def tn(func, s):
        from sympy.core.random import uniform
        c = uniform(1, 5)
        expr = func(s, c*exp_polar(I*pi)) - func(s, c*exp_polar(-I*pi))
        eps = 1e-15
        expr2 = func(s + eps, -c + eps*I) - func(s + eps, -c - eps*I)
        return abs(expr.n() - expr2.n()).n() < 1e-10

    nu = Symbol('nu')
    assert besselj(nu, exp_polar(2*pi*I)*x) == exp(2*pi*I*nu)*besselj(nu, x)
    assert besseli(nu, exp_polar(2*pi*I)*x) == exp(2*pi*I*nu)*besseli(nu, x)
    assert tn(besselj, 2)
    assert tn(besselj, pi)
    assert tn(besselj, I)
    assert tn(besseli, 2)
    assert tn(besseli, pi)
    assert tn(besseli, I)


def test_airy_base():
    z = Symbol('z')
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)

    assert conjugate(airyai(z)) == airyai(conjugate(z))
    assert airyai(x).is_extended_real

    assert airyai(x+I*y).as_real_imag() == (
            airyai(x - I*y)/2 + airyai(x + I*y)/2,
            I*(airyai(x - I*y) - airyai(x + I*y))/2)


def test_airyai():
    z = Symbol('z', real=False)
    t = Symbol('t', negative=True)
    p = Symbol('p', positive=True)

    assert isinstance(airyai(z), airyai)

    assert airyai(0) == 3**Rational(1, 3)/(3*gamma(Rational(2, 3)))
    assert airyai(oo) == 0
    assert airyai(-oo) == 0

    assert diff(airyai(z), z) == airyaiprime(z)

    assert series(airyai(z), z, 0, 3) == (
        3**Rational(5, 6)*gamma(Rational(1, 3))/(6*pi) - 3**Rational(1, 6)*z*gamma(Rational(2, 3))/(2*pi) + O(z**3))

    assert airyai(z).rewrite(hyper) == (
        -3**Rational(2, 3)*z*hyper((), (Rational(4, 3),), z**3/9)/(3*gamma(Rational(1, 3))) +
         3**Rational(1, 3)*hyper((), (Rational(2, 3),), z**3/9)/(3*gamma(Rational(2, 3))))

    assert isinstance(airyai(z).rewrite(besselj), airyai)
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)
    assert airyai(z).rewrite(besseli) == (
        -z*besseli(Rational(1, 3), 2*z**Rational(3, 2)/3)/(3*(z**Rational(3, 2))**Rational(1, 3)) +
         (z**Rational(3, 2))**Rational(1, 3)*besseli(Rational(-1, 3), 2*z**Rational(3, 2)/3)/3)
    assert airyai(p).rewrite(besseli) == (
        sqrt(p)*(besseli(Rational(-1, 3), 2*p**Rational(3, 2)/3) -
                 besseli(Rational(1, 3), 2*p**Rational(3, 2)/3))/3)

    assert expand_func(airyai(2*(3*z**5)**Rational(1, 3))) == (
        -sqrt(3)*(-1 + (z**5)**Rational(1, 3)/z**Rational(5, 3))*airybi(2*3**Rational(1, 3)*z**Rational(5, 3))/6 +
         (1 + (z**5)**Rational(1, 3)/z**Rational(5, 3))*airyai(2*3**Rational(1, 3)*z**Rational(5, 3))/2)


def test_airybi():
    z = Symbol('z', real=False)
    t = Symbol('t', negative=True)
    p = Symbol('p', positive=True)

    assert isinstance(airybi(z), airybi)

    assert airybi(0) == 3**Rational(5, 6)/(3*gamma(Rational(2, 3)))
    assert airybi(oo) is oo
    assert airybi(-oo) == 0

    assert diff(airybi(z), z) == airybiprime(z)

    assert series(airybi(z), z, 0, 3) == (
        3**Rational(1, 3)*gamma(Rational(1, 3))/(2*pi) + 3**Rational(2, 3)*z*gamma(Rational(2, 3))/(2*pi) + O(z**3))

    assert airybi(z).rewrite(hyper) == (
        3**Rational(1, 6)*z*hyper((), (Rational(4, 3),), z**3/9)/gamma(Rational(1, 3)) +
        3**Rational(5, 6)*hyper((), (Rational(2, 3),), z**3/9)/(3*gamma(Rational(2, 3))))

    assert isinstance(airybi(z).rewrite(besselj), airybi)
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)
    assert airybi(z).rewrite(besseli) == (
        sqrt(3)*(z*besseli(Rational(1, 3), 2*z**Rational(3, 2)/3)/(z**Rational(3, 2))**Rational(1, 3) +
                 (z**Rational(3, 2))**Rational(1, 3)*besseli(Rational(-1, 3), 2*z**Rational(3, 2)/3))/3)
    assert airybi(p).rewrite(besseli) == (
        sqrt(3)*sqrt(p)*(besseli(Rational(-1, 3), 2*p**Rational(3, 2)/3) +
                         besseli(Rational(1, 3), 2*p**Rational(3, 2)/3))/3)

    assert expand_func(airybi(2*(3*z**5)**Rational(1, 3))) == (
        sqrt(3)*(1 - (z**5)**Rational(1, 3)/z**Rational(5, 3))*airyai(2*3**Rational(1, 3)*z**Rational(5, 3))/2 +
        (1 + (z**5)**Rational(1, 3)/z**Rational(5, 3))*airybi(2*3**Rational(1, 3)*z**Rational(5, 3))/2)


def test_airyaiprime():
    z = Symbol('z', real=False)
    t = Symbol('t', negative=True)
    p = Symbol('p', positive=True)

    assert isinstance(airyaiprime(z), airyaiprime)

    assert airyaiprime(0) == -3**Rational(2, 3)/(3*gamma(Rational(1, 3)))
    assert airyaiprime(oo) == 0

    assert diff(airyaiprime(z), z) == z*airyai(z)

    assert series(airyaiprime(z), z, 0, 3) == (
        -3**Rational(2, 3)/(3*gamma(Rational(1, 3))) + 3**Rational(1, 3)*z**2/(6*gamma(Rational(2, 3))) + O(z**3))

    assert airyaiprime(z).rewrite(hyper) == (
        3**Rational(1, 3)*z**2*hyper((), (Rational(5, 3),), z**3/9)/(6*gamma(Rational(2, 3))) -
        3**Rational(2, 3)*hyper((), (Rational(1, 3),), z**3/9)/(3*gamma(Rational(1, 3))))

    assert isinstance(airyaiprime(z).rewrite(besselj), airyaiprime)
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)
    assert airyaiprime(z).rewrite(besseli) == (
        z**2*besseli(Rational(2, 3), 2*z**Rational(3, 2)/3)/(3*(z**Rational(3, 2))**Rational(2, 3)) -
        (z**Rational(3, 2))**Rational(2, 3)*besseli(Rational(-1, 3), 2*z**Rational(3, 2)/3)/3)
    assert airyaiprime(p).rewrite(besseli) == (
        p*(-besseli(Rational(-2, 3), 2*p**Rational(3, 2)/3) + besseli(Rational(2, 3), 2*p**Rational(3, 2)/3))/3)

    assert expand_func(airyaiprime(2*(3*z**5)**Rational(1, 3))) == (
        sqrt(3)*(z**Rational(5, 3)/(z**5)**Rational(1, 3) - 1)*airybiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/6 +
        (z**Rational(5, 3)/(z**5)**Rational(1, 3) + 1)*airyaiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2)


def test_airybiprime():
    z = Symbol('z', real=False)
    t = Symbol('t', negative=True)
    p = Symbol('p', positive=True)

    assert isinstance(airybiprime(z), airybiprime)

    assert airybiprime(0) == 3**Rational(1, 6)/gamma(Rational(1, 3))
    assert airybiprime(oo) is oo
    assert airybiprime(-oo) == 0

    assert diff(airybiprime(z), z) == z*airybi(z)

    assert series(airybiprime(z), z, 0, 3) == (
        3**Rational(1, 6)/gamma(Rational(1, 3)) + 3**Rational(5, 6)*z**2/(6*gamma(Rational(2, 3))) + O(z**3))

    assert airybiprime(z).rewrite(hyper) == (
        3**Rational(5, 6)*z**2*hyper((), (Rational(5, 3),), z**3/9)/(6*gamma(Rational(2, 3))) +
        3**Rational(1, 6)*hyper((), (Rational(1, 3),), z**3/9)/gamma(Rational(1, 3)))

    assert isinstance(airybiprime(z).rewrite(besselj), airybiprime)
    assert airyai(t).rewrite(besselj) == (
        sqrt(-t)*(besselj(Rational(-1, 3), 2*(-t)**Rational(3, 2)/3) +
                  besselj(Rational(1, 3), 2*(-t)**Rational(3, 2)/3))/3)
    assert airybiprime(z).rewrite(besseli) == (
        sqrt(3)*(z**2*besseli(Rational(2, 3), 2*z**Rational(3, 2)/3)/(z**Rational(3, 2))**Rational(2, 3) +
                 (z**Rational(3, 2))**Rational(2, 3)*besseli(Rational(-2, 3), 2*z**Rational(3, 2)/3))/3)
    assert airybiprime(p).rewrite(besseli) == (
        sqrt(3)*p*(besseli(Rational(-2, 3), 2*p**Rational(3, 2)/3) + besseli(Rational(2, 3), 2*p**Rational(3, 2)/3))/3)

    assert expand_func(airybiprime(2*(3*z**5)**Rational(1, 3))) == (
        sqrt(3)*(z**Rational(5, 3)/(z**5)**Rational(1, 3) - 1)*airyaiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2 +
        (z**Rational(5, 3)/(z**5)**Rational(1, 3) + 1)*airybiprime(2*3**Rational(1, 3)*z**Rational(5, 3))/2)


def test_marcumq():
    m = Symbol('m')
    a = Symbol('a')
    b = Symbol('b')

    assert marcumq(0, 0, 0) == 0
    assert marcumq(m, 0, b) == uppergamma(m, b**2/2)/gamma(m)
    assert marcumq(2, 0, 5) == 27*exp(Rational(-25, 2))/2
    assert marcumq(0, a, 0) == 1 - exp(-a**2/2)
    assert marcumq(0, pi, 0) == 1 - exp(-pi**2/2)
    assert marcumq(1, a, a) == S.Half + exp(-a**2)*besseli(0, a**2)/2
    assert marcumq(2, a, a) == S.Half + exp(-a**2)*besseli(0, a**2)/2 + exp(-a**2)*besseli(1, a**2)

    assert diff(marcumq(1, a, 3), a) == a*(-marcumq(1, a, 3) + marcumq(2, a, 3))
    assert diff(marcumq(2, 3, b), b) == -b**2*exp(-b**2/2 - Rational(9, 2))*besseli(1, 3*b)/3

    x = Symbol('x')
    assert marcumq(2, 3, 4).rewrite(Integral, x=x) == \
           Integral(x**2*exp(-x**2/2 - Rational(9, 2))*besseli(1, 3*x), (x, 4, oo))/3
    assert eq([marcumq(5, -2, 3).rewrite(Integral).evalf(10)],
              [0.7905769565])

    k = Symbol('k')
    assert marcumq(-3, -5, -7).rewrite(Sum, k=k) == \
           exp(-37)*Sum((Rational(5, 7))**k*besseli(k, 35), (k, 4, oo))
    assert eq([marcumq(1, 3, 1).rewrite(Sum).evalf(10)],
              [0.9891705502])

    assert marcumq(1, a, a, evaluate=False).rewrite(besseli) == S.Half + exp(-a**2)*besseli(0, a**2)/2
    assert marcumq(2, a, a, evaluate=False).rewrite(besseli) == S.Half + exp(-a**2)*besseli(0, a**2)/2 + \
           exp(-a**2)*besseli(1, a**2)
    assert marcumq(3, a, a).rewrite(besseli) == (besseli(1, a**2) + besseli(2, a**2))*exp(-a**2) + \
           S.Half + exp(-a**2)*besseli(0, a**2)/2
    assert marcumq(5, 8, 8).rewrite(besseli) == exp(-64)*besseli(0, 64)/2 + \
           (besseli(4, 64) + besseli(3, 64) + besseli(2, 64) + besseli(1, 64))*exp(-64) + S.Half
    assert marcumq(m, a, a).rewrite(besseli) == marcumq(m, a, a)

    x = Symbol('x', integer=True)
    assert marcumq(x, a, a).rewrite(besseli) == marcumq(x, a, a)


def test_issue_26134():
    x = Symbol('x')
    assert marcumq(2, 3, 4).rewrite(Integral, x=x).dummy_eq(
        Integral(x**2*exp(-x**2/2 - Rational(9, 2))*besseli(1, 3*x), (x, 4, oo))/3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\tests\test_public_api.py ===
import functools
import importlib
import inspect
import pkgutil
import subprocess
import sys
import sysconfig
import types
import warnings

import pytest

import numpy
import numpy as np
from numpy.testing import IS_WASM

try:
    import ctypes
except ImportError:
    ctypes = None


def check_dir(module, module_name=None):
    """Returns a mapping of all objects with the wrong __module__ attribute."""
    if module_name is None:
        module_name = module.__name__
    results = {}
    for name in dir(module):
        if name == "core":
            continue
        item = getattr(module, name)
        if (hasattr(item, '__module__') and hasattr(item, '__name__')
                and item.__module__ != module_name):
            results[name] = item.__module__ + '.' + item.__name__
    return results


def test_numpy_namespace():
    # We override dir to not show these members
    allowlist = {
        'recarray': 'numpy.rec.recarray',
    }
    bad_results = check_dir(np)
    # pytest gives better error messages with the builtin assert than with
    # assert_equal
    assert bad_results == allowlist


@pytest.mark.skipif(IS_WASM, reason="can't start subprocess")
@pytest.mark.parametrize('name', ['testing'])
def test_import_lazy_import(name):
    """Make sure we can actually use the modules we lazy load.

    While not exported as part of the public API, it was accessible.  With the
    use of __getattr__ and __dir__, this isn't always true It can happen that
    an infinite recursion may happen.

    This is the only way I found that would force the failure to appear on the
    badly implemented code.

    We also test for the presence of the lazily imported modules in dir

    """
    exe = (sys.executable, '-c', "import numpy; numpy." + name)
    result = subprocess.check_output(exe)
    assert not result

    # Make sure they are still in the __dir__
    assert name in dir(np)


def test_dir_testing():
    """Assert that output of dir has only one "testing/tester"
    attribute without duplicate"""
    assert len(dir(np)) == len(set(dir(np)))


def test_numpy_linalg():
    bad_results = check_dir(np.linalg)
    assert bad_results == {}


def test_numpy_fft():
    bad_results = check_dir(np.fft)
    assert bad_results == {}


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available in this python")
def test_NPY_NO_EXPORT():
    cdll = ctypes.CDLL(np._core._multiarray_tests.__file__)
    # Make sure an arbitrary NPY_NO_EXPORT function is actually hidden
    f = getattr(cdll, 'test_not_exported', None)
    assert f is None, ("'test_not_exported' is mistakenly exported, "
                      "NPY_NO_EXPORT does not work")


# Historically NumPy has not used leading underscores for private submodules
# much.  This has resulted in lots of things that look like public modules
# (i.e. things that can be imported as `import numpy.somesubmodule.somefile`),
# but were never intended to be public.  The PUBLIC_MODULES list contains
# modules that are either public because they were meant to be, or because they
# contain public functions/objects that aren't present in any other namespace
# for whatever reason and therefore should be treated as public.
#
# The PRIVATE_BUT_PRESENT_MODULES list contains modules that look public (lack
# of underscores) but should not be used.  For many of those modules the
# current status is fine.  For others it may make sense to work on making them
# private, to clean up our public API and avoid confusion.
PUBLIC_MODULES = ['numpy.' + s for s in [
    "ctypeslib",
    "dtypes",
    "exceptions",
    "f2py",
    "fft",
    "lib",
    "lib.array_utils",
    "lib.format",
    "lib.introspect",
    "lib.mixins",
    "lib.npyio",
    "lib.recfunctions",  # note: still needs cleaning, was forgotten for 2.0
    "lib.scimath",
    "lib.stride_tricks",
    "linalg",
    "ma",
    "ma.extras",
    "ma.mrecords",
    "polynomial",
    "polynomial.chebyshev",
    "polynomial.hermite",
    "polynomial.hermite_e",
    "polynomial.laguerre",
    "polynomial.legendre",
    "polynomial.polynomial",
    "random",
    "strings",
    "testing",
    "testing.overrides",
    "typing",
    "typing.mypy_plugin",
    "version",
]]
if sys.version_info < (3, 12):
    PUBLIC_MODULES += [
        'numpy.' + s for s in [
            "distutils",
            "distutils.cpuinfo",
            "distutils.exec_command",
            "distutils.misc_util",
            "distutils.log",
            "distutils.system_info",
        ]
    ]


PUBLIC_ALIASED_MODULES = [
    "numpy.char",
    "numpy.emath",
    "numpy.rec",
]


PRIVATE_BUT_PRESENT_MODULES = ['numpy.' + s for s in [
    "conftest",
    "core",
    "core.multiarray",
    "core.numeric",
    "core.umath",
    "core.arrayprint",
    "core.defchararray",
    "core.einsumfunc",
    "core.fromnumeric",
    "core.function_base",
    "core.getlimits",
    "core.numerictypes",
    "core.overrides",
    "core.records",
    "core.shape_base",
    "f2py.auxfuncs",
    "f2py.capi_maps",
    "f2py.cb_rules",
    "f2py.cfuncs",
    "f2py.common_rules",
    "f2py.crackfortran",
    "f2py.diagnose",
    "f2py.f2py2e",
    "f2py.f90mod_rules",
    "f2py.func2subr",
    "f2py.rules",
    "f2py.symbolic",
    "f2py.use_rules",
    "fft.helper",
    "lib.user_array",  # note: not in np.lib, but probably should just be deleted
    "linalg.lapack_lite",
    "linalg.linalg",
    "ma.core",
    "ma.testutils",
    "matlib",
    "matrixlib",
    "matrixlib.defmatrix",
    "polynomial.polyutils",
    "random.mtrand",
    "random.bit_generator",
    "testing.print_coercion_tables",
]]
if sys.version_info < (3, 12):
    PRIVATE_BUT_PRESENT_MODULES += [
        'numpy.' + s for s in [
            "distutils.armccompiler",
            "distutils.fujitsuccompiler",
            "distutils.ccompiler",
            'distutils.ccompiler_opt',
            "distutils.command",
            "distutils.command.autodist",
            "distutils.command.bdist_rpm",
            "distutils.command.build",
            "distutils.command.build_clib",
            "distutils.command.build_ext",
            "distutils.command.build_py",
            "distutils.command.build_scripts",
            "distutils.command.build_src",
            "distutils.command.config",
            "distutils.command.config_compiler",
            "distutils.command.develop",
            "distutils.command.egg_info",
            "distutils.command.install",
            "distutils.command.install_clib",
            "distutils.command.install_data",
            "distutils.command.install_headers",
            "distutils.command.sdist",
            "distutils.conv_template",
            "distutils.core",
            "distutils.extension",
            "distutils.fcompiler",
            "distutils.fcompiler.absoft",
            "distutils.fcompiler.arm",
            "distutils.fcompiler.compaq",
            "distutils.fcompiler.environment",
            "distutils.fcompiler.g95",
            "distutils.fcompiler.gnu",
            "distutils.fcompiler.hpux",
            "distutils.fcompiler.ibm",
            "distutils.fcompiler.intel",
            "distutils.fcompiler.lahey",
            "distutils.fcompiler.mips",
            "distutils.fcompiler.nag",
            "distutils.fcompiler.none",
            "distutils.fcompiler.pathf95",
            "distutils.fcompiler.pg",
            "distutils.fcompiler.nv",
            "distutils.fcompiler.sun",
            "distutils.fcompiler.vast",
            "distutils.fcompiler.fujitsu",
            "distutils.from_template",
            "distutils.intelccompiler",
            "distutils.lib2def",
            "distutils.line_endings",
            "distutils.mingw32ccompiler",
            "distutils.msvccompiler",
            "distutils.npy_pkg_config",
            "distutils.numpy_distribution",
            "distutils.pathccompiler",
            "distutils.unixccompiler",
        ]
    ]


def is_unexpected(name):
    """Check if this needs to be considered."""
    return (
        '._' not in name and '.tests' not in name and '.setup' not in name
        and name not in PUBLIC_MODULES
        and name not in PUBLIC_ALIASED_MODULES
        and name not in PRIVATE_BUT_PRESENT_MODULES
    )


if sys.version_info >= (3, 12):
    SKIP_LIST = []
else:
    SKIP_LIST = ["numpy.distutils.msvc9compiler"]


def test_all_modules_are_expected():
    """
    Test that we don't add anything that looks like a new public module by
    accident.  Check is based on filenames.
    """

    modnames = []
    for _, modname, ispkg in pkgutil.walk_packages(path=np.__path__,
                                                   prefix=np.__name__ + '.',
                                                   onerror=None):
        if is_unexpected(modname) and modname not in SKIP_LIST:
            # We have a name that is new.  If that's on purpose, add it to
            # PUBLIC_MODULES.  We don't expect to have to add anything to
            # PRIVATE_BUT_PRESENT_MODULES.  Use an underscore in the name!
            modnames.append(modname)

    if modnames:
        raise AssertionError(f'Found unexpected modules: {modnames}')


# Stuff that clearly shouldn't be in the API and is detected by the next test
# below
SKIP_LIST_2 = [
    'numpy.lib.math',
    'numpy.matlib.char',
    'numpy.matlib.rec',
    'numpy.matlib.emath',
    'numpy.matlib.exceptions',
    'numpy.matlib.math',
    'numpy.matlib.linalg',
    'numpy.matlib.fft',
    'numpy.matlib.random',
    'numpy.matlib.ctypeslib',
    'numpy.matlib.ma',
]
if sys.version_info < (3, 12):
    SKIP_LIST_2 += [
        'numpy.distutils.log.sys',
        'numpy.distutils.log.logging',
        'numpy.distutils.log.warnings',
    ]


def test_all_modules_are_expected_2():
    """
    Method checking all objects. The pkgutil-based method in
    `test_all_modules_are_expected` does not catch imports into a namespace,
    only filenames.  So this test is more thorough, and checks this like:

        import .lib.scimath as emath

    To check if something in a module is (effectively) public, one can check if
    there's anything in that namespace that's a public function/object but is
    not exposed in a higher-level namespace.  For example for a `numpy.lib`
    submodule::

        mod = np.lib.mixins
        for obj in mod.__all__:
            if obj in np.__all__:
                continue
            elif obj in np.lib.__all__:
                continue

            else:
                print(obj)

    """

    def find_unexpected_members(mod_name):
        members = []
        module = importlib.import_module(mod_name)
        if hasattr(module, '__all__'):
            objnames = module.__all__
        else:
            objnames = dir(module)

        for objname in objnames:
            if not objname.startswith('_'):
                fullobjname = mod_name + '.' + objname
                if isinstance(getattr(module, objname), types.ModuleType):
                    if is_unexpected(fullobjname):
                        if fullobjname not in SKIP_LIST_2:
                            members.append(fullobjname)

        return members

    unexpected_members = find_unexpected_members("numpy")
    for modname in PUBLIC_MODULES:
        unexpected_members.extend(find_unexpected_members(modname))

    if unexpected_members:
        raise AssertionError("Found unexpected object(s) that look like "
                             f"modules: {unexpected_members}")


def test_api_importable():
    """
    Check that all submodules listed higher up in this file can be imported

    Note that if a PRIVATE_BUT_PRESENT_MODULES entry goes missing, it may
    simply need to be removed from the list (deprecation may or may not be
    needed - apply common sense).
    """
    def check_importable(module_name):
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError):
            return False

        return True

    module_names = []
    for module_name in PUBLIC_MODULES:
        if not check_importable(module_name):
            module_names.append(module_name)

    if module_names:
        raise AssertionError("Modules in the public API that cannot be "
                             f"imported: {module_names}")

    for module_name in PUBLIC_ALIASED_MODULES:
        try:
            eval(module_name)
        except AttributeError:
            module_names.append(module_name)

    if module_names:
        raise AssertionError("Modules in the public API that were not "
                             f"found: {module_names}")

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', category=DeprecationWarning)
        warnings.filterwarnings('always', category=ImportWarning)
        for module_name in PRIVATE_BUT_PRESENT_MODULES:
            if not check_importable(module_name):
                module_names.append(module_name)

    if module_names:
        raise AssertionError("Modules that are not really public but looked "
                             "public and can not be imported: "
                             f"{module_names}")


@pytest.mark.xfail(
    sysconfig.get_config_var("Py_DEBUG") not in (None, 0, "0"),
    reason=(
        "NumPy possibly built with `USE_DEBUG=True ./tools/travis-test.sh`, "
        "which does not expose the `array_api` entry point. "
        "See https://github.com/numpy/numpy/pull/19800"
    ),
)
def test_array_api_entry_point():
    """
    Entry point for Array API implementation can be found with importlib and
    returns the main numpy namespace.
    """
    # For a development install that did not go through meson-python,
    # the entrypoint will not have been installed. So ensure this test fails
    # only if numpy is inside site-packages.
    numpy_in_sitepackages = sysconfig.get_path('platlib') in np.__file__

    eps = importlib.metadata.entry_points()
    xp_eps = eps.select(group="array_api")
    if len(xp_eps) == 0:
        if numpy_in_sitepackages:
            msg = "No entry points for 'array_api' found"
            raise AssertionError(msg) from None
        return

    try:
        ep = next(ep for ep in xp_eps if ep.name == "numpy")
    except StopIteration:
        if numpy_in_sitepackages:
            msg = "'numpy' not in array_api entry points"
            raise AssertionError(msg) from None
        return

    if ep.value == 'numpy.array_api':
        # Looks like the entrypoint for the current numpy build isn't
        # installed, but an older numpy is also installed and hence the
        # entrypoint is pointing to the old (no longer existing) location.
        # This isn't a problem except for when running tests with `spin` or an
        # in-place build.
        return

    xp = ep.load()
    msg = (
        f"numpy entry point value '{ep.value}' "
        "does not point to our Array API implementation"
    )
    assert xp is numpy, msg


def test_main_namespace_all_dir_coherence():
    """
    Checks if `dir(np)` and `np.__all__` are consistent and return
    the same content, excluding exceptions and private members.
    """
    def _remove_private_members(member_set):
        return {m for m in member_set if not m.startswith('_')}

    def _remove_exceptions(member_set):
        return member_set.difference({
            "bool"  # included only in __dir__
        })

    all_members = _remove_private_members(np.__all__)
    all_members = _remove_exceptions(all_members)

    dir_members = _remove_private_members(np.__dir__())
    dir_members = _remove_exceptions(dir_members)

    assert all_members == dir_members, (
        "Members that break symmetry: "
        f"{all_members.symmetric_difference(dir_members)}"
    )


@pytest.mark.filterwarnings(
    r"ignore:numpy.core(\.\w+)? is deprecated:DeprecationWarning"
)
def test_core_shims_coherence():
    """
    Check that all "semi-public" members of `numpy._core` are also accessible
    from `numpy.core` shims.
    """
    import numpy.core as core

    for member_name in dir(np._core):
        # Skip private and test members. Also if a module is aliased,
        # no need to add it to np.core
        if (
            member_name.startswith("_")
            or member_name in ["tests", "strings"]
            or f"numpy.{member_name}" in PUBLIC_ALIASED_MODULES
        ):
            continue

        member = getattr(np._core, member_name)

        # np.core is a shim and all submodules of np.core are shims
        # but we should be able to import everything in those shims
        # that are available in the "real" modules in np._core, with
        # the exception of the namespace packages (__spec__.origin is None),
        # like numpy._core.include, or numpy._core.lib.pkgconfig.
        if (
            inspect.ismodule(member)
            and member.__spec__ and member.__spec__.origin is not None
        ):
            submodule = member
            submodule_name = member_name
            for submodule_member_name in dir(submodule):
                # ignore dunder names
                if submodule_member_name.startswith("__"):
                    continue
                submodule_member = getattr(submodule, submodule_member_name)

                core_submodule = __import__(
                    f"numpy.core.{submodule_name}",
                    fromlist=[submodule_member_name]
                )

                assert submodule_member is getattr(
                    core_submodule, submodule_member_name
                )

        else:
            assert member is getattr(core, member_name)


def test_functions_single_location():
    """
    Check that each public function is available from one location only.

    Test performs BFS search traversing NumPy's public API. It flags
    any function-like object that is accessible from more that one place.
    """
    from collections.abc import Callable
    from typing import Any

    from numpy._core._multiarray_umath import (
        _ArrayFunctionDispatcher as dispatched_function,
    )

    visited_modules: set[types.ModuleType] = {np}
    visited_functions: set[Callable[..., Any]] = set()
    # Functions often have `__name__` overridden, therefore we need
    # to keep track of locations where functions have been found.
    functions_original_paths: dict[Callable[..., Any], str] = {}

    # Here we aggregate functions with more than one location.
    # It must be empty for the test to pass.
    duplicated_functions: list[tuple] = []

    modules_queue = [np]

    while len(modules_queue) > 0:

        module = modules_queue.pop()

        for member_name in dir(module):
            member = getattr(module, member_name)

            # first check if we got a module
            if (
                inspect.ismodule(member) and  # it's a module
                "numpy" in member.__name__ and  # inside NumPy
                not member_name.startswith("_") and  # not private
                "numpy._core" not in member.__name__ and  # outside _core
                # not a legacy or testing module
                member_name not in ["f2py", "ma", "testing", "tests"] and
                member not in visited_modules  # not visited yet
            ):
                modules_queue.append(member)
                visited_modules.add(member)

            # else check if we got a function-like object
            elif (
                inspect.isfunction(member) or
                isinstance(member, (dispatched_function, np.ufunc))
            ):
                if member in visited_functions:

                    # skip main namespace functions with aliases
                    if (
                        member.__name__ in [
                            "absolute",  # np.abs
                            "arccos",  # np.acos
                            "arccosh",  # np.acosh
                            "arcsin",  # np.asin
                            "arcsinh",  # np.asinh
                            "arctan",  # np.atan
                            "arctan2",  # np.atan2
                            "arctanh",  # np.atanh
                            "left_shift",  # np.bitwise_left_shift
                            "right_shift",  # np.bitwise_right_shift
                            "conjugate",  # np.conj
                            "invert",  # np.bitwise_not & np.bitwise_invert
                            "remainder",  # np.mod
                            "divide",  # np.true_divide
                            "concatenate",  # np.concat
                            "power",  # np.pow
                            "transpose",  # np.permute_dims
                        ] and
                        module.__name__ == "numpy"
                    ):
                        continue
                    # skip trimcoef from numpy.polynomial as it is
                    # duplicated by design.
                    if (
                        member.__name__ == "trimcoef" and
                        module.__name__.startswith("numpy.polynomial")
                    ):
                        continue

                    # skip ufuncs that are exported in np.strings as well
                    if member.__name__ in (
                        "add",
                        "equal",
                        "not_equal",
                        "greater",
                        "greater_equal",
                        "less",
                        "less_equal",
                    ) and module.__name__ == "numpy.strings":
                        continue

                    # numpy.char reexports all numpy.strings functions for
                    # backwards-compatibility
                    if module.__name__ == "numpy.char":
                        continue

                    # function is present in more than one location!
                    duplicated_functions.append(
                        (member.__name__,
                         module.__name__,
                         functions_original_paths[member])
                    )
                else:
                    visited_functions.add(member)
                    functions_original_paths[member] = module.__name__

    del visited_functions, visited_modules, functions_original_paths

    assert len(duplicated_functions) == 0, duplicated_functions


def test___module___attribute():
    modules_queue = [np]
    visited_modules = {np}
    visited_functions = set()
    incorrect_entries = []

    while len(modules_queue) > 0:
        module = modules_queue.pop()
        for member_name in dir(module):
            member = getattr(module, member_name)
            # first check if we got a module
            if (
                inspect.ismodule(member) and  # it's a module
                "numpy" in member.__name__ and  # inside NumPy
                not member_name.startswith("_") and  # not private
                "numpy._core" not in member.__name__ and  # outside _core
                # not in a skip module list
                member_name not in [
                    "char", "core", "f2py", "ma", "lapack_lite", "mrecords",
                    "testing", "tests", "polynomial", "typing", "mtrand",
                    "bit_generator",
                ] and
                member not in visited_modules  # not visited yet
            ):
                modules_queue.append(member)
                visited_modules.add(member)
            elif (
                not inspect.ismodule(member) and
                hasattr(member, "__name__") and
                not member.__name__.startswith("_") and
                member.__module__ != module.__name__ and
                member not in visited_functions
            ):
                # skip ufuncs that are exported in np.strings as well
                if member.__name__ in (
                    "add", "equal", "not_equal", "greater", "greater_equal",
                    "less", "less_equal",
                ) and module.__name__ == "numpy.strings":
                    continue

                # recarray and record are exported in np and np.rec
                if (
                    (member.__name__ == "recarray" and module.__name__ == "numpy") or
                    (member.__name__ == "record" and module.__name__ == "numpy.rec")
                ):
                    continue

                # ctypeslib exports ctypes c_long/c_longlong
                if (
                    member.__name__ in ("c_long", "c_longlong") and
                    module.__name__ == "numpy.ctypeslib"
                ):
                    continue

                # skip cdef classes
                if member.__name__ in (
                    "BitGenerator", "Generator", "MT19937", "PCG64", "PCG64DXSM",
                    "Philox", "RandomState", "SFC64", "SeedSequence",
                ):
                    continue

                incorrect_entries.append(
                    {
                        "Func": member.__name__,
                        "actual": member.__module__,
                        "expected": module.__name__,
                    }
                )
                visited_functions.add(member)

    if incorrect_entries:
        assert len(incorrect_entries) == 0, incorrect_entries


def _check_correct_qualname_and_module(obj) -> bool:
    qualname = obj.__qualname__
    name = obj.__name__
    module_name = obj.__module__
    assert name == qualname.split(".")[-1]

    module = sys.modules[module_name]
    actual_obj = functools.reduce(getattr, qualname.split("."), module)
    return (
        actual_obj is obj or
        # `obj` may be a bound method/property of `actual_obj`:
        (
            hasattr(actual_obj, "__get__") and hasattr(obj, "__self__") and
            actual_obj.__module__ == obj.__module__ and
            actual_obj.__qualname__ == qualname
        )
    )


def test___qualname___and___module___attribute():
    # NumPy messes with module and name/qualname attributes, but any object
    # should be discoverable based on its module and qualname, so test that.
    # We do this for anything with a name (ensuring qualname is also set).
    modules_queue = [np]
    visited_modules = {np}
    visited_functions = set()
    incorrect_entries = []

    while len(modules_queue) > 0:
        module = modules_queue.pop()
        for member_name in dir(module):
            member = getattr(module, member_name)
            # first check if we got a module
            if (
                inspect.ismodule(member) and  # it's a module
                "numpy" in member.__name__ and  # inside NumPy
                not member_name.startswith("_") and  # not private
                member_name not in {"tests", "typing"} and  # 2024-12: type names don't match
                "numpy._core" not in member.__name__ and  # outside _core
                member not in visited_modules  # not visited yet
            ):
                modules_queue.append(member)
                visited_modules.add(member)
            elif (
                not inspect.ismodule(member) and
                hasattr(member, "__name__") and
                not member.__name__.startswith("_") and
                not member_name.startswith("_") and
                not _check_correct_qualname_and_module(member) and
                member not in visited_functions
            ):
                incorrect_entries.append(
                    {
                        "found_at": f"{module.__name__}:{member_name}",
                        "advertises": f"{member.__module__}:{member.__qualname__}",
                    }
                )
                visited_functions.add(member)

    if incorrect_entries:
        assert len(incorrect_entries) == 0, incorrect_entries

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\__init__.py ===
# -*- coding: utf-8 -*-
import sys

try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'

__all__ = ['easter', 'parser', 'relativedelta', 'rrule', 'tz',
           'utils', 'zoneinfo']

def __getattr__(name):
    import importlib

    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(
        "module {!r} has not attribute {!r}".format(__name__, name)
    )


def __dir__():
    # __dir__ should include all the lazy-importable modules as well.
    return [x for x in globals() if x not in sys.modules] + __all__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\utils\formulas.py ===
# Copyright (c) 2010-2024 openpyxl

"""
List of builtin formulae
"""

FORMULAE = ("CUBEKPIMEMBER", "CUBEMEMBER", "CUBEMEMBERPROPERTY", "CUBERANKEDMEMBER", "CUBESET", "CUBESETCOUNT", "CUBEVALUE", "DAVERAGE", "DCOUNT", "DCOUNTA", "DGET", "DMAX", "DMIN", "DPRODUCT", "DSTDEV", "DSTDEVP", "DSUM", "DVAR", "DVARP", "DATE", "DATEDIF", "DATEVALUE", "DAY", "DAYS360", "EDATE", "EOMONTH", "HOUR", "MINUTE", "MONTH", "NETWORKDAYS", "NETWORKDAYS.INTL", "NOW", "SECOND", "TIME", "TIMEVALUE", "TODAY", "WEEKDAY", "WEEKNUM", "WORKDAY", "WORKDAY.INTL", "YEAR", "YEARFRAC", "BESSELI", "BESSELJ", "BESSELK", "BESSELY", "BIN2DEC", "BIN2HEX", "BIN2OCT", "COMPLEX", "CONVERT", "DEC2BIN", "DEC2HEX", "DEC2OCT", "DELTA", "ERF", "ERFC", "GESTEP", "HEX2BIN", "HEX2DEC", "HEX2OCT", "IMABS", "IMAGINARY", "IMARGUMENT", "IMCONJUGATE", "IMCOS", "IMDIV", "IMEXP", "IMLN", "IMLOG10", "IMLOG2", "IMPOWER", "IMPRODUCT", "IMREAL", "IMSIN", "IMSQRT", "IMSUB", "IMSUM", "OCT2BIN", "OCT2DEC", "OCT2HEX", "ACCRINT", "ACCRINTM", "AMORDEGRC", "AMORLINC", "COUPDAYBS", "COUPDAYS", "COUPDAYSNC", "COUPNCD", "COUPNUM", "COUPPCD", "CUMIPMT", "CUMPRINC", "DB", "DDB", "DISC", "DOLLARDE", "DOLLARFR", "DURATION", "EFFECT", "FV", "FVSCHEDULE", "INTRATE", "IPMT", "IRR", "ISPMT", "MDURATION", "MIRR", "NOMINAL", "NPER", "NPV", "ODDFPRICE", "ODDFYIELD", "ODDLPRICE", "ODDLYIELD", "PMT", "PPMT", "PRICE", "PRICEDISC", "PRICEMAT", "PV", "RATE", "RECEIVED", "SLN", "SYD", "TBILLEQ", "TBILLPRICE", "TBILLYIELD", "VDB", "XIRR", "XNPV", "YIELD", "YIELDDISC", "YIELDMAT", "CELL", "ERROR.TYPE", "INFO", "ISBLANK", "ISERR", "ISERROR", "ISEVEN", "ISLOGICAL", "ISNA", "ISNONTEXT", "ISNUMBER", "ISODD", "ISREF", "ISTEXT", "N", "NA", "TYPE", "AND", "FALSE", "IF", "IFERROR", "NOT", "OR", "TRUE", "ADDRESS", "AREAS", "CHOOSE", "COLUMN", "COLUMNS", "GETPIVOTDATA", "HLOOKUP", "HYPERLINK", "INDEX", "INDIRECT", "LOOKUP", "MATCH", "OFFSET", "ROW", "ROWS", "RTD", "TRANSPOSE", "VLOOKUP", "ABS", "ACOS", "ACOSH", "ASIN", "ASINH", "ATAN", "ATAN2", "ATANH", "CEILING", "COMBIN", "COS", "COSH", "DEGREES", "ECMA.CEILING", "EVEN", "EXP", "FACT", "FACTDOUBLE", "FLOOR", "GCD", "INT", "ISO.CEILING", "LCM", "LN", "LOG", "LOG10", "MDETERM", "MINVERSE", "MMULT", "MOD", "MROUND", "MULTINOMIAL", "ODD", "PI", "POWER", "PRODUCT", "QUOTIENT", "RADIANS", "RAND", "RANDBETWEEN", "ROMAN", "ROUND", "ROUNDDOWN", "ROUNDUP", "SERIESSUM", "SIGN", "SIN", "SINH", "SQRT", "SQRTPI", "SUBTOTAL", "SUM", "SUMIF", "SUMIFS", "SUMPRODUCT", "SUMSQ", "SUMX2MY2", "SUMX2PY2", "SUMXMY2", "TAN", "TANH", "TRUNC", "AVEDEV", "AVERAGE", "AVERAGEA", "AVERAGEIF", "AVERAGEIFS", "BETADIST", "BETAINV", "BINOMDIST", "CHIDIST", "CHIINV", "CHITEST", "CONFIDENCE", "CORREL", "COUNT", "COUNTA", "COUNTBLANK", "COUNTIF", "COUNTIFS", "COVAR", "CRITBINOM", "DEVSQ", "EXPONDIST", "FDIST", "FINV", "FISHER", "FISHERINV", "FORECAST", "FREQUENCY", "FTEST", "GAMMADIST", "GAMMAINV", "GAMMALN", "GEOMEAN", "GROWTH", "HARMEAN", "HYPGEOMDIST", "INTERCEPT", "KURT", "LARGE", "LINEST", "LOGEST", "LOGINV", "LOGNORMDIST", "MAX", "MAXA", "MEDIAN", "MIN", "MINA", "MODE", "NEGBINOMDIST", "NORMDIST", "NORMINV", "NORMSDIST", "NORMSINV", "PEARSON", "PERCENTILE", "PERCENTRANK", "PERMUT", "POISSON", "PROB", "QUARTILE", "RANK", "RSQ", "SKEW", "SLOPE", "SMALL", "STANDARDIZE", "STDEV", "STDEVA", "STDEVP", "STDEVPA", "STEYX", "TDIST", "TINV", "TREND", "TRIMMEAN", "TTEST", "VAR", "VARA", "VARP", "VARPA", "WEIBULL", "ZTEST", "ASC", "BAHTTEXT", "CHAR", "CLEAN", "CODE", "CONCATENATE", "DOLLAR", "EXACT", "FIND", "FINDB", "FIXED", "JIS", "LEFT", "LEFTB", "LEN", "LENB", "LOWER", "MID", "MIDB", "PHONETIC", "PROPER", "REPLACE", "REPLACEB", "REPT", "RIGHT", "RIGHTB", "SEARCH", "SEARCHB", "SUBSTITUTE", "T", "TEXT", "TRIM", "UPPER", "VALUE")

FORMULAE = frozenset(FORMULAE)


from openpyxl.formula import Tokenizer


def validate(formula):
    """
    Utility function for checking whether a formula is syntactically correct
    """
    assert formula.startswith("=")
    formula = Tokenizer(formula)
    for t in formula.items:
        if t.type == "FUNC" and t.subtype == "OPEN":
            if not t.value.startswith("_xlfn.") and t.value[:-1] not in FORMULAE:
                raise ValueError(f"Unknown function {t.value} in {formula.formula}. The function may need a prefix")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\__main__.py ===
import os
import sys

# Remove '' and current working directory from the first entry
# of sys.path, if present to avoid using current directory
# in pip commands check, freeze, install, list and show,
# when invoked as python -m pip <command>
if sys.path[0] in ("", os.getcwd()):
    sys.path.pop(0)

# If we are running from a wheel, add the wheel to sys.path
# This allows the usage python pip-*.whl/pip install pip-*.whl
if __package__ == "":
    # __file__ is pip-*.whl/pip/__main__.py
    # first dirname call strips of '/__main__.py', second strips off '/pip'
    # Resulting path is the name of the wheel itself
    # Add that to sys.path so we can import pip
    path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, path)

if __name__ == "__main__":
    from pip._internal.cli.main import main as _main

    sys.exit(_main())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_fileno.py ===
from __future__ import annotations

from typing import IO, Callable


def get_fileno(file_like: IO[str]) -> int | None:
    """Get fileno() from a file, accounting for poorly implemented file-like objects.

    Args:
        file_like (IO): A file-like object.

    Returns:
        int | None: The result of fileno if available, or None if operation failed.
    """
    fileno: Callable[[], int] | None = getattr(file_like, "fileno", None)
    if fileno is not None:
        try:
            return fileno()
        except Exception:
            # `fileno` is documented as potentially raising a OSError
            # Alas, from the issues, there are so many poorly implemented file-like objects,
            # that `fileno()` can raise just about anything.
            return None
    return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\window.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The WindowTypes implementation."""


class WindowTypes:
    """Set of supported window types."""

    TAB = "tab"
    WINDOW = "window"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\actions\mouse_button.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


class MouseButton:
    LEFT = 0
    MIDDLE = 1
    RIGHT = 2
    BACK = 3
    FORWARD = 4

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\bidi\console.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from enum import Enum


class Console(Enum):
    ALL = "all"
    LOG = "log"
    ERROR = "error"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\pyutils.py ===
from sympy.printing.pycode import PythonCodePrinter

""" This module collects utilities for rendering Python code. """


def render_as_module(content, standard='python3'):
    """Renders Python code as a module (with the required imports).

    Parameters
    ==========

    standard :
        See the parameter ``standard`` in
        :meth:`sympy.printing.pycode.pycode`
    """

    printer = PythonCodePrinter({'standard':standard})
    pystr = printer.doprint(content)
    if printer._settings['fully_qualified_modules']:
        module_imports_str = '\n'.join('import %s' % k for k in printer.module_imports)
    else:
        module_imports_str = '\n'.join(['from %s import %s' % (k, ', '.join(v)) for
                                        k, v in printer.module_imports.items()])
    return module_imports_str + '\n\n' + pystr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\__init__.py ===
""" The ``sympy.codegen`` module contains classes and functions for building
abstract syntax trees of algorithms. These trees may then be printed by the
code-printers in ``sympy.printing``.

There are several submodules available:
- ``sympy.codegen.ast``: AST nodes useful across multiple languages.
- ``sympy.codegen.cnodes``: AST nodes useful for the C family of languages.
- ``sympy.codegen.fnodes``: AST nodes useful for Fortran.
- ``sympy.codegen.cfunctions``: functions specific to C (C99 math functions)
- ``sympy.codegen.ffunctions``: functions specific to Fortran (e.g. ``kind``).



"""
from .ast import (
    Assignment, aug_assign, CodeBlock, For, Attribute, Variable, Declaration,
    While, Scope, Print, FunctionPrototype, FunctionDefinition, FunctionCall
)

__all__ = [
    'Assignment', 'aug_assign', 'CodeBlock', 'For', 'Attribute', 'Variable',
    'Declaration', 'While', 'Scope', 'Print', 'FunctionPrototype',
    'FunctionDefinition', 'FunctionCall',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\dynkin_diagram.py ===
from .cartan_type import CartanType


def DynkinDiagram(t):
    """Display the Dynkin diagram of a given Lie algebra

    Works by generating the CartanType for the input, t, and then returning the
    Dynkin diagram method from the individual classes.

    Examples
    ========

    >>> from sympy.liealgebras.dynkin_diagram import DynkinDiagram
    >>> print(DynkinDiagram("A3"))
    0---0---0
    1   2   3

    >>> print(DynkinDiagram("B4"))
    0---0---0=>=0
    1   2   3   4

    """

    return CartanType(t).dynkin_diagram()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\backends\textbackend\text.py ===
import sympy.plotting.backends.base_backend as base_backend
from sympy.plotting.series import LineOver1DRangeSeries
from sympy.plotting.textplot import textplot


class TextBackend(base_backend.Plot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def show(self):
        if not base_backend._show:
            return
        if len(self._series) != 1:
            raise ValueError(
                'The TextBackend supports only one graph per Plot.')
        elif not isinstance(self._series[0], LineOver1DRangeSeries):
            raise ValueError(
                'The TextBackend supports only expressions over a 1D range')
        else:
            ser = self._series[0]
            textplot(ser.expr, ser.start, ser.end)

    def close(self):
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_slice.py ===
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.errors import UnsortedIndexError

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl


class TestMultiIndexSlicers:
    def test_per_axis_per_level_getitem(self):
        # GH6134
        # example test case
        ix = MultiIndex.from_product(
            [_mklbl("A", 5), _mklbl("B", 7), _mklbl("C", 4), _mklbl("D", 2)]
        )
        df = DataFrame(np.arange(len(ix.to_numpy())), index=ix)

        result = df.loc[(slice("A1", "A3"), slice(None), ["C1", "C3"]), :]
        expected = df.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in df.index.values
                if a in ("A1", "A2", "A3") and c in ("C1", "C3")
            ]
        ]
        tm.assert_frame_equal(result, expected)

        expected = df.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in df.index.values
                if a in ("A1", "A2", "A3") and c in ("C1", "C2", "C3")
            ]
        ]
        result = df.loc[(slice("A1", "A3"), slice(None), slice("C1", "C3")), :]
        tm.assert_frame_equal(result, expected)

        # test multi-index slicing with per axis and per index controls
        index = MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("A", 3), ("B", 1)], names=["one", "two"]
        )
        columns = MultiIndex.from_tuples(
            [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")],
            names=["lvl0", "lvl1"],
        )

        df = DataFrame(
            np.arange(16, dtype="int64").reshape(4, 4), index=index, columns=columns
        )
        df = df.sort_index(axis=0).sort_index(axis=1)

        # identity
        result = df.loc[(slice(None), slice(None)), :]
        tm.assert_frame_equal(result, df)
        result = df.loc[(slice(None), slice(None)), (slice(None), slice(None))]
        tm.assert_frame_equal(result, df)
        result = df.loc[:, (slice(None), slice(None))]
        tm.assert_frame_equal(result, df)

        # index
        result = df.loc[(slice(None), [1]), :]
        expected = df.iloc[[0, 3]]
        tm.assert_frame_equal(result, expected)

        result = df.loc[(slice(None), 1), :]
        expected = df.iloc[[0, 3]]
        tm.assert_frame_equal(result, expected)

        # columns
        result = df.loc[:, (slice(None), ["foo"])]
        expected = df.iloc[:, [1, 3]]
        tm.assert_frame_equal(result, expected)

        # both
        result = df.loc[(slice(None), 1), (slice(None), ["foo"])]
        expected = df.iloc[[0, 3], [1, 3]]
        tm.assert_frame_equal(result, expected)

        result = df.loc["A", "a"]
        expected = DataFrame(
            {"bar": [1, 5, 9], "foo": [0, 4, 8]},
            index=Index([1, 2, 3], name="two"),
            columns=Index(["bar", "foo"], name="lvl1"),
        )
        tm.assert_frame_equal(result, expected)

        result = df.loc[(slice(None), [1, 2]), :]
        expected = df.iloc[[0, 1, 3]]
        tm.assert_frame_equal(result, expected)

        # multi-level series
        s = Series(np.arange(len(ix.to_numpy())), index=ix)
        result = s.loc["A1":"A3", :, ["C1", "C3"]]
        expected = s.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in s.index.values
                if a in ("A1", "A2", "A3") and c in ("C1", "C3")
            ]
        ]
        tm.assert_series_equal(result, expected)

        # boolean indexers
        result = df.loc[(slice(None), df.loc[:, ("a", "bar")] > 5), :]
        expected = df.iloc[[2, 3]]
        tm.assert_frame_equal(result, expected)

        msg = (
            "cannot index with a boolean indexer "
            "that is not the same length as the index"
        )
        with pytest.raises(ValueError, match=msg):
            df.loc[(slice(None), np.array([True, False])), :]

        with pytest.raises(KeyError, match=r"\[1\] not in index"):
            # slice(None) is on the index, [1] is on the columns, but 1 is
            #  not in the columns, so we raise
            #  This used to treat [1] as positional GH#16396
            df.loc[slice(None), [1]]

        # not lexsorted
        assert df.index._lexsort_depth == 2
        df = df.sort_index(level=1, axis=0)
        assert df.index._lexsort_depth == 0

        msg = (
            "MultiIndex slicing requires the index to be "
            r"lexsorted: slicing on levels \[1\], lexsort depth 0"
        )
        with pytest.raises(UnsortedIndexError, match=msg):
            df.loc[(slice(None), slice("bar")), :]

        # GH 16734: not sorted, but no real slicing
        result = df.loc[(slice(None), df.loc[:, ("a", "bar")] > 5), :]
        tm.assert_frame_equal(result, df.iloc[[1, 3], :])

    def test_multiindex_slicers_non_unique(self):
        # GH 7106
        # non-unique mi index support
        df = (
            DataFrame(
                {
                    "A": ["foo", "foo", "foo", "foo"],
                    "B": ["a", "a", "a", "a"],
                    "C": [1, 2, 1, 3],
                    "D": [1, 2, 3, 4],
                }
            )
            .set_index(["A", "B", "C"])
            .sort_index()
        )
        assert not df.index.is_unique
        expected = (
            DataFrame({"A": ["foo", "foo"], "B": ["a", "a"], "C": [1, 1], "D": [1, 3]})
            .set_index(["A", "B", "C"])
            .sort_index()
        )
        result = df.loc[(slice(None), slice(None), 1), :]
        tm.assert_frame_equal(result, expected)

        # this is equivalent of an xs expression
        result = df.xs(1, level=2, drop_level=False)
        tm.assert_frame_equal(result, expected)

        df = (
            DataFrame(
                {
                    "A": ["foo", "foo", "foo", "foo"],
                    "B": ["a", "a", "a", "a"],
                    "C": [1, 2, 1, 2],
                    "D": [1, 2, 3, 4],
                }
            )
            .set_index(["A", "B", "C"])
            .sort_index()
        )
        assert not df.index.is_unique
        expected = (
            DataFrame({"A": ["foo", "foo"], "B": ["a", "a"], "C": [1, 1], "D": [1, 3]})
            .set_index(["A", "B", "C"])
            .sort_index()
        )
        result = df.loc[(slice(None), slice(None), 1), :]
        assert not result.index.is_unique
        tm.assert_frame_equal(result, expected)

        # GH12896
        # numpy-implementation dependent bug
        ints = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            12,
            13,
            14,
            14,
            16,
            17,
            18,
            19,
            200000,
            200000,
        ]
        n = len(ints)
        idx = MultiIndex.from_arrays([["a"] * n, ints])
        result = Series([1] * n, index=idx)
        result = result.sort_index()
        result = result.loc[(slice(None), slice(100000))]
        expected = Series([1] * (n - 2), index=idx[:-2]).sort_index()
        tm.assert_series_equal(result, expected)

    def test_multiindex_slicers_datetimelike(self):
        # GH 7429
        # buggy/inconsistent behavior when slicing with datetime-like
        dates = [datetime(2012, 1, 1, 12, 12, 12) + timedelta(days=i) for i in range(6)]
        freq = [1, 2]
        index = MultiIndex.from_product([dates, freq], names=["date", "frequency"])

        df = DataFrame(
            np.arange(6 * 2 * 4, dtype="int64").reshape(-1, 4),
            index=index,
            columns=list("ABCD"),
        )

        # multi-axis slicing
        idx = pd.IndexSlice
        expected = df.iloc[[0, 2, 4], [0, 1]]
        result = df.loc[
            (
                slice(
                    Timestamp("2012-01-01 12:12:12"), Timestamp("2012-01-03 12:12:12")
                ),
                slice(1, 1),
            ),
            slice("A", "B"),
        ]
        tm.assert_frame_equal(result, expected)

        result = df.loc[
            (
                idx[
                    Timestamp("2012-01-01 12:12:12") : Timestamp("2012-01-03 12:12:12")
                ],
                idx[1:1],
            ),
            slice("A", "B"),
        ]
        tm.assert_frame_equal(result, expected)

        result = df.loc[
            (
                slice(
                    Timestamp("2012-01-01 12:12:12"), Timestamp("2012-01-03 12:12:12")
                ),
                1,
            ),
            slice("A", "B"),
        ]
        tm.assert_frame_equal(result, expected)

        # with strings
        result = df.loc[
            (slice("2012-01-01 12:12:12", "2012-01-03 12:12:12"), slice(1, 1)),
            slice("A", "B"),
        ]
        tm.assert_frame_equal(result, expected)

        result = df.loc[
            (idx["2012-01-01 12:12:12":"2012-01-03 12:12:12"], 1), idx["A", "B"]
        ]
        tm.assert_frame_equal(result, expected)

    def test_multiindex_slicers_edges(self):
        # GH 8132
        # various edge cases
        df = DataFrame(
            {
                "A": ["A0"] * 5 + ["A1"] * 5 + ["A2"] * 5,
                "B": ["B0", "B0", "B1", "B1", "B2"] * 3,
                "DATE": [
                    "2013-06-11",
                    "2013-07-02",
                    "2013-07-09",
                    "2013-07-30",
                    "2013-08-06",
                    "2013-06-11",
                    "2013-07-02",
                    "2013-07-09",
                    "2013-07-30",
                    "2013-08-06",
                    "2013-09-03",
                    "2013-10-01",
                    "2013-07-09",
                    "2013-08-06",
                    "2013-09-03",
                ],
                "VALUES": [22, 35, 14, 9, 4, 40, 18, 4, 2, 5, 1, 2, 3, 4, 2],
            }
        )

        df["DATE"] = pd.to_datetime(df["DATE"])
        df1 = df.set_index(["A", "B", "DATE"])
        df1 = df1.sort_index()

        # A1 - Get all values under "A0" and "A1"
        result = df1.loc[(slice("A1")), :]
        expected = df1.iloc[0:10]
        tm.assert_frame_equal(result, expected)

        # A2 - Get all values from the start to "A2"
        result = df1.loc[(slice("A2")), :]
        expected = df1
        tm.assert_frame_equal(result, expected)

        # A3 - Get all values under "B1" or "B2"
        result = df1.loc[(slice(None), slice("B1", "B2")), :]
        expected = df1.iloc[[2, 3, 4, 7, 8, 9, 12, 13, 14]]
        tm.assert_frame_equal(result, expected)

        # A4 - Get all values between 2013-07-02 and 2013-07-09
        result = df1.loc[(slice(None), slice(None), slice("20130702", "20130709")), :]
        expected = df1.iloc[[1, 2, 6, 7, 12]]
        tm.assert_frame_equal(result, expected)

        # B1 - Get all values in B0 that are also under A0, A1 and A2
        result = df1.loc[(slice("A2"), slice("B0")), :]
        expected = df1.iloc[[0, 1, 5, 6, 10, 11]]
        tm.assert_frame_equal(result, expected)

        # B2 - Get all values in B0, B1 and B2 (similar to what #2 is doing for
        # the As)
        result = df1.loc[(slice(None), slice("B2")), :]
        expected = df1
        tm.assert_frame_equal(result, expected)

        # B3 - Get all values from B1 to B2 and up to 2013-08-06
        result = df1.loc[(slice(None), slice("B1", "B2"), slice("2013-08-06")), :]
        expected = df1.iloc[[2, 3, 4, 7, 8, 9, 12, 13]]
        tm.assert_frame_equal(result, expected)

        # B4 - Same as A4 but the start of the date slice is not a key.
        #      shows indexing on a partial selection slice
        result = df1.loc[(slice(None), slice(None), slice("20130701", "20130709")), :]
        expected = df1.iloc[[1, 2, 6, 7, 12]]
        tm.assert_frame_equal(result, expected)

    def test_per_axis_per_level_doc_examples(self):
        # test index maker
        idx = pd.IndexSlice

        # from indexing.rst / advanced
        index = MultiIndex.from_product(
            [_mklbl("A", 4), _mklbl("B", 2), _mklbl("C", 4), _mklbl("D", 2)]
        )
        columns = MultiIndex.from_tuples(
            [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")],
            names=["lvl0", "lvl1"],
        )
        df = DataFrame(
            np.arange(len(index) * len(columns), dtype="int64").reshape(
                (len(index), len(columns))
            ),
            index=index,
            columns=columns,
        )
        result = df.loc[(slice("A1", "A3"), slice(None), ["C1", "C3"]), :]
        expected = df.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in df.index.values
                if a in ("A1", "A2", "A3") and c in ("C1", "C3")
            ]
        ]
        tm.assert_frame_equal(result, expected)
        result = df.loc[idx["A1":"A3", :, ["C1", "C3"]], :]
        tm.assert_frame_equal(result, expected)

        result = df.loc[(slice(None), slice(None), ["C1", "C3"]), :]
        expected = df.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in df.index.values
                if c in ("C1", "C3")
            ]
        ]
        tm.assert_frame_equal(result, expected)
        result = df.loc[idx[:, :, ["C1", "C3"]], :]
        tm.assert_frame_equal(result, expected)

        # not sorted
        msg = (
            "MultiIndex slicing requires the index to be lexsorted: "
            r"slicing on levels \[1\], lexsort depth 1"
        )
        with pytest.raises(UnsortedIndexError, match=msg):
            df.loc["A1", ("a", slice("foo"))]

        # GH 16734: not sorted, but no real slicing
        tm.assert_frame_equal(
            df.loc["A1", (slice(None), "foo")], df.loc["A1"].iloc[:, [0, 2]]
        )

        df = df.sort_index(axis=1)

        # slicing
        df.loc["A1", (slice(None), "foo")]
        df.loc[(slice(None), slice(None), ["C1", "C3"]), (slice(None), "foo")]

        # setitem
        df.loc(axis=0)[:, :, ["C1", "C3"]] = -10

    def test_loc_axis_arguments(self):
        index = MultiIndex.from_product(
            [_mklbl("A", 4), _mklbl("B", 2), _mklbl("C", 4), _mklbl("D", 2)]
        )
        columns = MultiIndex.from_tuples(
            [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")],
            names=["lvl0", "lvl1"],
        )
        df = (
            DataFrame(
                np.arange(len(index) * len(columns), dtype="int64").reshape(
                    (len(index), len(columns))
                ),
                index=index,
                columns=columns,
            )
            .sort_index()
            .sort_index(axis=1)
        )

        # axis 0
        result = df.loc(axis=0)["A1":"A3", :, ["C1", "C3"]]
        expected = df.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in df.index.values
                if a in ("A1", "A2", "A3") and c in ("C1", "C3")
            ]
        ]
        tm.assert_frame_equal(result, expected)

        result = df.loc(axis="index")[:, :, ["C1", "C3"]]
        expected = df.loc[
            [
                (
                    a,
                    b,
                    c,
                    d,
                )
                for a, b, c, d in df.index.values
                if c in ("C1", "C3")
            ]
        ]
        tm.assert_frame_equal(result, expected)

        # axis 1
        result = df.loc(axis=1)[:, "foo"]
        expected = df.loc[:, (slice(None), "foo")]
        tm.assert_frame_equal(result, expected)

        result = df.loc(axis="columns")[:, "foo"]
        expected = df.loc[:, (slice(None), "foo")]
        tm.assert_frame_equal(result, expected)

        # invalid axis
        for i in [-1, 2, "foo"]:
            msg = f"No axis named {i} for object type DataFrame"
            with pytest.raises(ValueError, match=msg):
                df.loc(axis=i)[:, :, ["C1", "C3"]]

    def test_loc_axis_single_level_multi_col_indexing_multiindex_col_df(self):
        # GH29519
        df = DataFrame(
            np.arange(27).reshape(3, 9),
            columns=MultiIndex.from_product([["a1", "a2", "a3"], ["b1", "b2", "b3"]]),
        )
        result = df.loc(axis=1)["a1":"a2"]
        expected = df.iloc[:, :-3]

        tm.assert_frame_equal(result, expected)

    def test_loc_axis_single_level_single_col_indexing_multiindex_col_df(self):
        # GH29519
        df = DataFrame(
            np.arange(27).reshape(3, 9),
            columns=MultiIndex.from_product([["a1", "a2", "a3"], ["b1", "b2", "b3"]]),
        )
        result = df.loc(axis=1)["a1"]
        expected = df.iloc[:, :3]
        expected.columns = ["b1", "b2", "b3"]

        tm.assert_frame_equal(result, expected)

    def test_loc_ax_single_level_indexer_simple_df(self):
        # GH29519
        # test single level indexing on single index column data frame
        df = DataFrame(np.arange(9).reshape(3, 3), columns=["a", "b", "c"])
        result = df.loc(axis=1)["a"]
        expected = Series(np.array([0, 3, 6]), name="a")
        tm.assert_series_equal(result, expected)

    def test_per_axis_per_level_setitem(self):
        # test index maker
        idx = pd.IndexSlice

        # test multi-index slicing with per axis and per index controls
        index = MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("A", 3), ("B", 1)], names=["one", "two"]
        )
        columns = MultiIndex.from_tuples(
            [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")],
            names=["lvl0", "lvl1"],
        )

        df_orig = DataFrame(
            np.arange(16, dtype="int64").reshape(4, 4), index=index, columns=columns
        )
        df_orig = df_orig.sort_index(axis=0).sort_index(axis=1)

        # identity
        df = df_orig.copy()
        df.loc[(slice(None), slice(None)), :] = 100
        expected = df_orig.copy()
        expected.iloc[:, :] = 100
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc(axis=0)[:, :] = 100
        expected = df_orig.copy()
        expected.iloc[:, :] = 100
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc[(slice(None), slice(None)), (slice(None), slice(None))] = 100
        expected = df_orig.copy()
        expected.iloc[:, :] = 100
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc[:, (slice(None), slice(None))] = 100
        expected = df_orig.copy()
        expected.iloc[:, :] = 100
        tm.assert_frame_equal(df, expected)

        # index
        df = df_orig.copy()
        df.loc[(slice(None), [1]), :] = 100
        expected = df_orig.copy()
        expected.iloc[[0, 3]] = 100
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc[(slice(None), 1), :] = 100
        expected = df_orig.copy()
        expected.iloc[[0, 3]] = 100
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc(axis=0)[:, 1] = 100
        expected = df_orig.copy()
        expected.iloc[[0, 3]] = 100
        tm.assert_frame_equal(df, expected)

        # columns
        df = df_orig.copy()
        df.loc[:, (slice(None), ["foo"])] = 100
        expected = df_orig.copy()
        expected.iloc[:, [1, 3]] = 100
        tm.assert_frame_equal(df, expected)

        # both
        df = df_orig.copy()
        df.loc[(slice(None), 1), (slice(None), ["foo"])] = 100
        expected = df_orig.copy()
        expected.iloc[[0, 3], [1, 3]] = 100
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc[idx[:, 1], idx[:, ["foo"]]] = 100
        expected = df_orig.copy()
        expected.iloc[[0, 3], [1, 3]] = 100
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc["A", "a"] = 100
        expected = df_orig.copy()
        expected.iloc[0:3, 0:2] = 100
        tm.assert_frame_equal(df, expected)

        # setting with a list-like
        df = df_orig.copy()
        df.loc[(slice(None), 1), (slice(None), ["foo"])] = np.array(
            [[100, 100], [100, 100]], dtype="int64"
        )
        expected = df_orig.copy()
        expected.iloc[[0, 3], [1, 3]] = 100
        tm.assert_frame_equal(df, expected)

        # not enough values
        df = df_orig.copy()

        msg = "setting an array element with a sequence."
        with pytest.raises(ValueError, match=msg):
            df.loc[(slice(None), 1), (slice(None), ["foo"])] = np.array(
                [[100], [100, 100]], dtype="int64"
            )

        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df.loc[(slice(None), 1), (slice(None), ["foo"])] = np.array(
                [100, 100, 100, 100], dtype="int64"
            )

        # with an alignable rhs
        df = df_orig.copy()
        df.loc[(slice(None), 1), (slice(None), ["foo"])] = (
            df.loc[(slice(None), 1), (slice(None), ["foo"])] * 5
        )
        expected = df_orig.copy()
        expected.iloc[[0, 3], [1, 3]] = expected.iloc[[0, 3], [1, 3]] * 5
        tm.assert_frame_equal(df, expected)

        df = df_orig.copy()
        df.loc[(slice(None), 1), (slice(None), ["foo"])] *= df.loc[
            (slice(None), 1), (slice(None), ["foo"])
        ]
        expected = df_orig.copy()
        expected.iloc[[0, 3], [1, 3]] *= expected.iloc[[0, 3], [1, 3]]
        tm.assert_frame_equal(df, expected)

        rhs = df_orig.loc[(slice(None), 1), (slice(None), ["foo"])].copy()
        rhs.loc[:, ("c", "bah")] = 10
        df = df_orig.copy()
        df.loc[(slice(None), 1), (slice(None), ["foo"])] *= rhs
        expected = df_orig.copy()
        expected.iloc[[0, 3], [1, 3]] *= expected.iloc[[0, 3], [1, 3]]
        tm.assert_frame_equal(df, expected)

    def test_multiindex_label_slicing_with_negative_step(self):
        ser = Series(
            np.arange(20), MultiIndex.from_product([list("abcde"), np.arange(4)])
        )
        SLC = pd.IndexSlice

        tm.assert_indexing_slices_equivalent(ser, SLC[::-1], SLC[::-1])

        tm.assert_indexing_slices_equivalent(ser, SLC["d"::-1], SLC[15::-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[("d",)::-1], SLC[15::-1])

        tm.assert_indexing_slices_equivalent(ser, SLC[:"d":-1], SLC[:11:-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[:("d",):-1], SLC[:11:-1])

        tm.assert_indexing_slices_equivalent(ser, SLC["d":"b":-1], SLC[15:3:-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[("d",):"b":-1], SLC[15:3:-1])
        tm.assert_indexing_slices_equivalent(ser, SLC["d":("b",):-1], SLC[15:3:-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[("d",):("b",):-1], SLC[15:3:-1])
        tm.assert_indexing_slices_equivalent(ser, SLC["b":"d":-1], SLC[:0])

        tm.assert_indexing_slices_equivalent(ser, SLC[("c", 2)::-1], SLC[10::-1])
        tm.assert_indexing_slices_equivalent(ser, SLC[:("c", 2):-1], SLC[:9:-1])
        tm.assert_indexing_slices_equivalent(
            ser, SLC[("e", 0):("c", 2):-1], SLC[16:9:-1]
        )

    def test_multiindex_slice_first_level(self):
        # GH 12697
        freq = ["a", "b", "c", "d"]
        idx = MultiIndex.from_product([freq, range(500)])
        df = DataFrame(list(range(2000)), index=idx, columns=["Test"])
        df_slice = df.loc[pd.IndexSlice[:, 30:70], :]
        result = df_slice.loc["a"]
        expected = DataFrame(list(range(30, 71)), columns=["Test"], index=range(30, 71))
        tm.assert_frame_equal(result, expected)
        result = df_slice.loc["d"]
        expected = DataFrame(
            list(range(1530, 1571)), columns=["Test"], index=range(30, 71)
        )
        tm.assert_frame_equal(result, expected)

    def test_int_series_slicing(self, multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data
        s = ymd["A"]
        result = s[5:]
        expected = s.reindex(s.index[5:])
        tm.assert_series_equal(result, expected)

        s = ymd["A"].copy()
        exp = ymd["A"].copy()
        s[5:] = 0
        exp.iloc[5:] = 0
        tm.assert_numpy_array_equal(s.values, exp.values)

        result = ymd[5:]
        expected = ymd.reindex(s.index[5:])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype, loc, iloc",
        [
            # dtype = int, step = -1
            ("int", slice(None, None, -1), slice(None, None, -1)),
            ("int", slice(3, None, -1), slice(3, None, -1)),
            ("int", slice(None, 1, -1), slice(None, 0, -1)),
            ("int", slice(3, 1, -1), slice(3, 0, -1)),
            # dtype = int, step = -2
            ("int", slice(None, None, -2), slice(None, None, -2)),
            ("int", slice(3, None, -2), slice(3, None, -2)),
            ("int", slice(None, 1, -2), slice(None, 0, -2)),
            ("int", slice(3, 1, -2), slice(3, 0, -2)),
            # dtype = str, step = -1
            ("str", slice(None, None, -1), slice(None, None, -1)),
            ("str", slice("d", None, -1), slice(3, None, -1)),
            ("str", slice(None, "b", -1), slice(None, 0, -1)),
            ("str", slice("d", "b", -1), slice(3, 0, -1)),
            # dtype = str, step = -2
            ("str", slice(None, None, -2), slice(None, None, -2)),
            ("str", slice("d", None, -2), slice(3, None, -2)),
            ("str", slice(None, "b", -2), slice(None, 0, -2)),
            ("str", slice("d", "b", -2), slice(3, 0, -2)),
        ],
    )
    def test_loc_slice_negative_stepsize(self, dtype, loc, iloc):
        # GH#38071
        labels = {
            "str": list("abcde"),
            "int": range(5),
        }[dtype]

        mi = MultiIndex.from_arrays([labels] * 2)
        df = DataFrame(1.0, index=mi, columns=["A"])

        SLC = pd.IndexSlice

        expected = df.iloc[iloc, :]
        result_get_loc = df.loc[SLC[loc], :]
        result_get_locs_level_0 = df.loc[SLC[loc, :], :]
        result_get_locs_level_1 = df.loc[SLC[:, loc], :]

        tm.assert_frame_equal(result_get_loc, expected)
        tm.assert_frame_equal(result_get_locs_level_0, expected)
        tm.assert_frame_equal(result_get_locs_level_1, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\test_cut.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    cut,
    date_range,
    interval_range,
    isna,
    qcut,
    timedelta_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod


def test_simple():
    data = np.ones(5, dtype="int64")
    result = cut(data, 4, labels=False)

    expected = np.array([1, 1, 1, 1, 1])
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("func", [list, np.array])
def test_bins(func):
    data = func([0.2, 1.4, 2.5, 6.2, 9.7, 2.1])
    result, bins = cut(data, 3, retbins=True)

    intervals = IntervalIndex.from_breaks(bins.round(3))
    intervals = intervals.take([0, 0, 0, 1, 2, 0])
    expected = Categorical(intervals, ordered=True)

    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.1905, 3.36666667, 6.53333333, 9.7]))


def test_right():
    data = np.array([0.2, 1.4, 2.5, 6.2, 9.7, 2.1, 2.575])
    result, bins = cut(data, 4, right=True, retbins=True)

    intervals = IntervalIndex.from_breaks(bins.round(3))
    expected = Categorical(intervals, ordered=True)
    expected = expected.take([0, 0, 0, 2, 3, 0, 0])

    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.1905, 2.575, 4.95, 7.325, 9.7]))


def test_no_right():
    data = np.array([0.2, 1.4, 2.5, 6.2, 9.7, 2.1, 2.575])
    result, bins = cut(data, 4, right=False, retbins=True)

    intervals = IntervalIndex.from_breaks(bins.round(3), closed="left")
    intervals = intervals.take([0, 0, 0, 2, 3, 0, 1])
    expected = Categorical(intervals, ordered=True)

    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.2, 2.575, 4.95, 7.325, 9.7095]))


def test_bins_from_interval_index():
    c = cut(range(5), 3)
    expected = c
    result = cut(range(5), bins=expected.categories)
    tm.assert_categorical_equal(result, expected)

    expected = Categorical.from_codes(
        np.append(c.codes, -1), categories=c.categories, ordered=True
    )
    result = cut(range(6), bins=expected.categories)
    tm.assert_categorical_equal(result, expected)


def test_bins_from_interval_index_doc_example():
    # Make sure we preserve the bins.
    ages = np.array([10, 15, 13, 12, 23, 25, 28, 59, 60])
    c = cut(ages, bins=[0, 18, 35, 70])
    expected = IntervalIndex.from_tuples([(0, 18), (18, 35), (35, 70)])
    tm.assert_index_equal(c.categories, expected)

    result = cut([25, 20, 50], bins=c.categories)
    tm.assert_index_equal(result.categories, expected)
    tm.assert_numpy_array_equal(result.codes, np.array([1, 1, 2], dtype="int8"))


def test_bins_not_overlapping_from_interval_index():
    # see gh-23980
    msg = "Overlapping IntervalIndex is not accepted"
    ii = IntervalIndex.from_tuples([(0, 10), (2, 12), (4, 14)])

    with pytest.raises(ValueError, match=msg):
        cut([5, 6], bins=ii)


def test_bins_not_monotonic():
    msg = "bins must increase monotonically"
    data = [0.2, 1.4, 2.5, 6.2, 9.7, 2.1]

    with pytest.raises(ValueError, match=msg):
        cut(data, [0.1, 1.5, 1, 10])


@pytest.mark.parametrize(
    "x, bins, expected",
    [
        (
            date_range("2017-12-31", periods=3),
            [Timestamp.min, Timestamp("2018-01-01"), Timestamp.max],
            IntervalIndex.from_tuples(
                [
                    (Timestamp.min, Timestamp("2018-01-01")),
                    (Timestamp("2018-01-01"), Timestamp.max),
                ]
            ),
        ),
        (
            [-1, 0, 1],
            np.array(
                [np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], dtype="int64"
            ),
            IntervalIndex.from_tuples(
                [(np.iinfo(np.int64).min, 0), (0, np.iinfo(np.int64).max)]
            ),
        ),
        (
            [
                np.timedelta64(-1, "ns"),
                np.timedelta64(0, "ns"),
                np.timedelta64(1, "ns"),
            ],
            np.array(
                [
                    np.timedelta64(-np.iinfo(np.int64).max, "ns"),
                    np.timedelta64(0, "ns"),
                    np.timedelta64(np.iinfo(np.int64).max, "ns"),
                ]
            ),
            IntervalIndex.from_tuples(
                [
                    (
                        np.timedelta64(-np.iinfo(np.int64).max, "ns"),
                        np.timedelta64(0, "ns"),
                    ),
                    (
                        np.timedelta64(0, "ns"),
                        np.timedelta64(np.iinfo(np.int64).max, "ns"),
                    ),
                ]
            ),
        ),
    ],
)
def test_bins_monotonic_not_overflowing(x, bins, expected):
    # GH 26045
    result = cut(x, bins)
    tm.assert_index_equal(result.categories, expected)


def test_wrong_num_labels():
    msg = "Bin labels must be one fewer than the number of bin edges"
    data = [0.2, 1.4, 2.5, 6.2, 9.7, 2.1]

    with pytest.raises(ValueError, match=msg):
        cut(data, [0, 1, 10], labels=["foo", "bar", "baz"])


@pytest.mark.parametrize(
    "x,bins,msg",
    [
        ([], 2, "Cannot cut empty array"),
        ([1, 2, 3], 0.5, "`bins` should be a positive integer"),
    ],
)
def test_cut_corner(x, bins, msg):
    with pytest.raises(ValueError, match=msg):
        cut(x, bins)


@pytest.mark.parametrize("arg", [2, np.eye(2), DataFrame(np.eye(2))])
@pytest.mark.parametrize("cut_func", [cut, qcut])
def test_cut_not_1d_arg(arg, cut_func):
    msg = "Input array must be 1 dimensional"
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)


@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3, 4, np.inf],
        [-np.inf, 0, 1, 2, 3, 4],
        [-np.inf, 0, 1, 2, 3, 4, np.inf],
    ],
)
def test_int_bins_with_inf(data):
    # GH 24314
    msg = "cannot specify integer `bins` when input data contains infinity"
    with pytest.raises(ValueError, match=msg):
        cut(data, bins=3)


def test_cut_out_of_range_more():
    # see gh-1511
    name = "x"

    ser = Series([0, -1, 0, 1, -3], name=name)
    ind = cut(ser, [0, 1], labels=False)

    exp = Series([np.nan, np.nan, np.nan, 0, np.nan], name=name)
    tm.assert_series_equal(ind, exp)


@pytest.mark.parametrize(
    "right,breaks,closed",
    [
        (True, [-1e-3, 0.25, 0.5, 0.75, 1], "right"),
        (False, [0, 0.25, 0.5, 0.75, 1 + 1e-3], "left"),
    ],
)
def test_labels(right, breaks, closed):
    arr = np.tile(np.arange(0, 1.01, 0.1), 4)

    result, bins = cut(arr, 4, retbins=True, right=right)
    ex_levels = IntervalIndex.from_breaks(breaks, closed=closed)
    tm.assert_index_equal(result.categories, ex_levels)


def test_cut_pass_series_name_to_factor():
    name = "foo"
    ser = Series(np.random.default_rng(2).standard_normal(100), name=name)

    factor = cut(ser, 4)
    assert factor.name == name


def test_label_precision():
    arr = np.arange(0, 0.73, 0.01)
    result = cut(arr, 4, precision=2)

    ex_levels = IntervalIndex.from_breaks([-0.00072, 0.18, 0.36, 0.54, 0.72])
    tm.assert_index_equal(result.categories, ex_levels)


@pytest.mark.parametrize("labels", [None, False])
def test_na_handling(labels):
    arr = np.arange(0, 0.75, 0.01)
    arr[::3] = np.nan

    result = cut(arr, 4, labels=labels)
    result = np.asarray(result)

    expected = np.where(isna(arr), np.nan, result)
    tm.assert_almost_equal(result, expected)


def test_inf_handling():
    data = np.arange(6)
    data_ser = Series(data, dtype="int64")

    bins = [-np.inf, 2, 4, np.inf]
    result = cut(data, bins)
    result_ser = cut(data_ser, bins)

    ex_uniques = IntervalIndex.from_breaks(bins)
    tm.assert_index_equal(result.categories, ex_uniques)

    assert result[5] == Interval(4, np.inf)
    assert result[0] == Interval(-np.inf, 2)
    assert result_ser[5] == Interval(4, np.inf)
    assert result_ser[0] == Interval(-np.inf, 2)


def test_cut_out_of_bounds():
    arr = np.random.default_rng(2).standard_normal(100)
    result = cut(arr, [-1, 0, 1])

    mask = isna(result)
    ex_mask = (arr < -1) | (arr > 1)
    tm.assert_numpy_array_equal(mask, ex_mask)


@pytest.mark.parametrize(
    "get_labels,get_expected",
    [
        (
            lambda labels: labels,
            lambda labels: Categorical(
                ["Medium"] + 4 * ["Small"] + ["Medium", "Large"],
                categories=labels,
                ordered=True,
            ),
        ),
        (
            lambda labels: Categorical.from_codes([0, 1, 2], labels),
            lambda labels: Categorical.from_codes([1] + 4 * [0] + [1, 2], labels),
        ),
    ],
)
def test_cut_pass_labels(get_labels, get_expected):
    bins = [0, 25, 50, 100]
    arr = [50, 5, 10, 15, 20, 30, 70]
    labels = ["Small", "Medium", "Large"]

    result = cut(arr, bins, labels=get_labels(labels))
    tm.assert_categorical_equal(result, get_expected(labels))


def test_cut_pass_labels_compat():
    # see gh-16459
    arr = [50, 5, 10, 15, 20, 30, 70]
    labels = ["Good", "Medium", "Bad"]

    result = cut(arr, 3, labels=labels)
    exp = cut(arr, 3, labels=Categorical(labels, categories=labels, ordered=True))
    tm.assert_categorical_equal(result, exp)


@pytest.mark.parametrize("x", [np.arange(11.0), np.arange(11.0) / 1e10])
def test_round_frac_just_works(x):
    # It works.
    cut(x, 2)


@pytest.mark.parametrize(
    "val,precision,expected",
    [
        (-117.9998, 3, -118),
        (117.9998, 3, 118),
        (117.9998, 2, 118),
        (0.000123456, 2, 0.00012),
    ],
)
def test_round_frac(val, precision, expected):
    # see gh-1979
    result = tmod._round_frac(val, precision=precision)
    assert result == expected


def test_cut_return_intervals():
    ser = Series([0, 1, 2, 3, 4, 5, 6, 7, 8])
    result = cut(ser, 3)

    exp_bins = np.linspace(0, 8, num=4).round(3)
    exp_bins[0] -= 0.008

    expected = Series(
        IntervalIndex.from_breaks(exp_bins, closed="right").take(
            [0, 0, 0, 1, 1, 1, 2, 2, 2]
        )
    ).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)


def test_series_ret_bins():
    # see gh-8589
    ser = Series(np.arange(4))
    result, bins = cut(ser, 2, retbins=True)

    expected = Series(
        IntervalIndex.from_breaks([-0.003, 1.5, 3], closed="right").repeat(2)
    ).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"duplicates": "drop"}, None),
        ({}, "Bin edges must be unique"),
        ({"duplicates": "raise"}, "Bin edges must be unique"),
        ({"duplicates": "foo"}, "invalid value for 'duplicates' parameter"),
    ],
)
def test_cut_duplicates_bin(kwargs, msg):
    # see gh-20947
    bins = [0, 2, 4, 6, 10, 10]
    values = Series(np.array([1, 3, 5, 7, 9]), index=["a", "b", "c", "d", "e"])

    if msg is not None:
        with pytest.raises(ValueError, match=msg):
            cut(values, bins, **kwargs)
    else:
        result = cut(values, bins, **kwargs)
        expected = cut(values, pd.unique(np.asarray(bins)))
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("data", [9.0, -9.0, 0.0])
@pytest.mark.parametrize("length", [1, 2])
def test_single_bin(data, length):
    # see gh-14652, gh-15428
    ser = Series([data] * length)
    result = cut(ser, 1, labels=False)

    expected = Series([0] * length, dtype=np.intp)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "array_1_writeable,array_2_writeable", [(True, True), (True, False), (False, False)]
)
def test_cut_read_only(array_1_writeable, array_2_writeable):
    # issue 18773
    array_1 = np.arange(0, 100, 10)
    array_1.flags.writeable = array_1_writeable

    array_2 = np.arange(0, 100, 10)
    array_2.flags.writeable = array_2_writeable

    hundred_elements = np.arange(100)
    tm.assert_categorical_equal(
        cut(hundred_elements, array_1), cut(hundred_elements, array_2)
    )


@pytest.mark.parametrize(
    "conv",
    [
        lambda v: Timestamp(v),
        lambda v: to_datetime(v),
        lambda v: np.datetime64(v),
        lambda v: Timestamp(v).to_pydatetime(),
    ],
)
def test_datetime_bin(conv):
    data = [np.datetime64("2012-12-13"), np.datetime64("2012-12-15")]
    bin_data = ["2012-12-12", "2012-12-14", "2012-12-16"]

    expected = Series(
        IntervalIndex(
            [
                Interval(Timestamp(bin_data[0]), Timestamp(bin_data[1])),
                Interval(Timestamp(bin_data[1]), Timestamp(bin_data[2])),
            ]
        )
    ).astype(CategoricalDtype(ordered=True))

    bins = [conv(v) for v in bin_data]
    result = Series(cut(data, bins=bins))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("box", [Series, Index, np.array, list])
def test_datetime_cut(unit, box):
    # see gh-14714
    #
    # Testing time data when it comes in various collection types.
    data = to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"]).astype(f"M8[{unit}]")
    data = box(data)
    result, _ = cut(data, 3, retbins=True)

    if box is list:
        # We don't (yet) do inference on these, so get nanos
        unit = "ns"

    if unit == "s":
        # See https://github.com/pandas-dev/pandas/pull/56101#discussion_r1405325425
        # for why we round to 8 seconds instead of 7
        left = DatetimeIndex(
            ["2012-12-31 23:57:08", "2013-01-01 16:00:00", "2013-01-02 08:00:00"],
            dtype=f"M8[{unit}]",
        )
    else:
        left = DatetimeIndex(
            [
                "2012-12-31 23:57:07.200000",
                "2013-01-01 16:00:00",
                "2013-01-02 08:00:00",
            ],
            dtype=f"M8[{unit}]",
        )
    right = DatetimeIndex(
        ["2013-01-01 16:00:00", "2013-01-02 08:00:00", "2013-01-03 00:00:00"],
        dtype=f"M8[{unit}]",
    )

    exp_intervals = IntervalIndex.from_arrays(left, right)
    expected = Series(exp_intervals).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(Series(result), expected)


@pytest.mark.parametrize("box", [list, np.array, Index, Series])
def test_datetime_tz_cut_mismatched_tzawareness(box):
    # GH#54964
    bins = box(
        [
            Timestamp("2013-01-01 04:57:07.200000"),
            Timestamp("2013-01-01 21:00:00"),
            Timestamp("2013-01-02 13:00:00"),
            Timestamp("2013-01-03 05:00:00"),
        ]
    )
    ser = Series(date_range("20130101", periods=3, tz="US/Eastern"))

    msg = "Cannot use timezone-naive bins with timezone-aware values"
    with pytest.raises(ValueError, match=msg):
        cut(ser, bins)


@pytest.mark.parametrize(
    "bins",
    [
        3,
        [
            Timestamp("2013-01-01 04:57:07.200000", tz="UTC").tz_convert("US/Eastern"),
            Timestamp("2013-01-01 21:00:00", tz="UTC").tz_convert("US/Eastern"),
            Timestamp("2013-01-02 13:00:00", tz="UTC").tz_convert("US/Eastern"),
            Timestamp("2013-01-03 05:00:00", tz="UTC").tz_convert("US/Eastern"),
        ],
    ],
)
@pytest.mark.parametrize("box", [list, np.array, Index, Series])
def test_datetime_tz_cut(bins, box):
    # see gh-19872
    tz = "US/Eastern"
    ser = Series(date_range("20130101", periods=3, tz=tz))

    if not isinstance(bins, int):
        bins = box(bins)

    result = cut(ser, bins)
    expected = Series(
        IntervalIndex(
            [
                Interval(
                    Timestamp("2012-12-31 23:57:07.200000", tz=tz),
                    Timestamp("2013-01-01 16:00:00", tz=tz),
                ),
                Interval(
                    Timestamp("2013-01-01 16:00:00", tz=tz),
                    Timestamp("2013-01-02 08:00:00", tz=tz),
                ),
                Interval(
                    Timestamp("2013-01-02 08:00:00", tz=tz),
                    Timestamp("2013-01-03 00:00:00", tz=tz),
                ),
            ]
        )
    ).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)


def test_datetime_nan_error():
    msg = "bins must be of datetime64 dtype"

    with pytest.raises(ValueError, match=msg):
        cut(date_range("20130101", periods=3), bins=[0, 2, 4])


def test_datetime_nan_mask():
    result = cut(
        date_range("20130102", periods=5), bins=date_range("20130101", periods=2)
    )

    mask = result.categories.isna()
    tm.assert_numpy_array_equal(mask, np.array([False]))

    mask = result.isna()
    tm.assert_numpy_array_equal(mask, np.array([False, True, True, True, True]))


@pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
def test_datetime_cut_roundtrip(tz, unit):
    # see gh-19891
    ser = Series(date_range("20180101", periods=3, tz=tz, unit=unit))
    result, result_bins = cut(ser, 2, retbins=True)

    expected = cut(ser, result_bins)
    tm.assert_series_equal(result, expected)

    if unit == "s":
        # TODO: constructing DatetimeIndex with dtype="M8[s]" without truncating
        #  the first entry here raises in array_to_datetime. Should truncate
        #  instead of raising?
        # See https://github.com/pandas-dev/pandas/pull/56101#discussion_r1405325425
        # for why we round to 8 seconds instead of 7
        expected_bins = DatetimeIndex(
            ["2017-12-31 23:57:08", "2018-01-02 00:00:00", "2018-01-03 00:00:00"],
            dtype=f"M8[{unit}]",
        )
    else:
        expected_bins = DatetimeIndex(
            [
                "2017-12-31 23:57:07.200000",
                "2018-01-02 00:00:00",
                "2018-01-03 00:00:00",
            ],
            dtype=f"M8[{unit}]",
        )
    expected_bins = expected_bins.tz_localize(tz)
    tm.assert_index_equal(result_bins, expected_bins)


def test_timedelta_cut_roundtrip():
    # see gh-19891
    ser = Series(timedelta_range("1day", periods=3))
    result, result_bins = cut(ser, 2, retbins=True)

    expected = cut(ser, result_bins)
    tm.assert_series_equal(result, expected)

    expected_bins = TimedeltaIndex(
        ["0 days 23:57:07.200000", "2 days 00:00:00", "3 days 00:00:00"]
    )
    tm.assert_index_equal(result_bins, expected_bins)


@pytest.mark.parametrize("bins", [6, 7])
@pytest.mark.parametrize(
    "box, compare",
    [
        (Series, tm.assert_series_equal),
        (np.array, tm.assert_categorical_equal),
        (list, tm.assert_equal),
    ],
)
def test_cut_bool_coercion_to_int(bins, box, compare):
    # issue 20303
    data_expected = box([0, 1, 1, 0, 1] * 10)
    data_result = box([False, True, True, False, True] * 10)
    expected = cut(data_expected, bins, duplicates="drop")
    result = cut(data_result, bins, duplicates="drop")
    compare(result, expected)


@pytest.mark.parametrize("labels", ["foo", 1, True])
def test_cut_incorrect_labels(labels):
    # GH 13318
    values = range(5)
    msg = "Bin labels must either be False, None or passed in as a list-like argument"
    with pytest.raises(ValueError, match=msg):
        cut(values, 4, labels=labels)


@pytest.mark.parametrize("bins", [3, [0, 5, 15]])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
def test_cut_nullable_integer(bins, right, include_lowest):
    a = np.random.default_rng(2).integers(0, 10, size=50).astype(float)
    a[::2] = np.nan
    result = cut(
        pd.array(a, dtype="Int64"), bins, right=right, include_lowest=include_lowest
    )
    expected = cut(a, bins, right=right, include_lowest=include_lowest)
    tm.assert_categorical_equal(result, expected)


@pytest.mark.parametrize(
    "data, bins, labels, expected_codes, expected_labels",
    [
        ([15, 17, 19], [14, 16, 18, 20], ["A", "B", "A"], [0, 1, 0], ["A", "B"]),
        ([1, 3, 5], [0, 2, 4, 6, 8], [2, 0, 1, 2], [2, 0, 1], [0, 1, 2]),
    ],
)
def test_cut_non_unique_labels(data, bins, labels, expected_codes, expected_labels):
    # GH 33141
    result = cut(data, bins=bins, labels=labels, ordered=False)
    expected = Categorical.from_codes(
        expected_codes, categories=expected_labels, ordered=False
    )
    tm.assert_categorical_equal(result, expected)


@pytest.mark.parametrize(
    "data, bins, labels, expected_codes, expected_labels",
    [
        ([15, 17, 19], [14, 16, 18, 20], ["C", "B", "A"], [0, 1, 2], ["C", "B", "A"]),
        ([1, 3, 5], [0, 2, 4, 6, 8], [3, 0, 1, 2], [0, 1, 2], [3, 0, 1, 2]),
    ],
)
def test_cut_unordered_labels(data, bins, labels, expected_codes, expected_labels):
    # GH 33141
    result = cut(data, bins=bins, labels=labels, ordered=False)
    expected = Categorical.from_codes(
        expected_codes, categories=expected_labels, ordered=False
    )
    tm.assert_categorical_equal(result, expected)


def test_cut_unordered_with_missing_labels_raises_error():
    # GH 33141
    msg = "'labels' must be provided if 'ordered = False'"
    with pytest.raises(ValueError, match=msg):
        cut([0.5, 3], bins=[0, 1, 2], ordered=False)


def test_cut_unordered_with_series_labels():
    # https://github.com/pandas-dev/pandas/issues/36603
    ser = Series([1, 2, 3, 4, 5])
    bins = Series([0, 2, 4, 6])
    labels = Series(["a", "b", "c"])
    result = cut(ser, bins=bins, labels=labels, ordered=False)
    expected = Series(["a", "a", "b", "b", "c"], dtype="category")
    tm.assert_series_equal(result, expected)


def test_cut_no_warnings():
    df = DataFrame({"value": np.random.default_rng(2).integers(0, 100, 20)})
    labels = [f"{i} - {i + 9}" for i in range(0, 100, 10)]
    with tm.assert_produces_warning(False):
        df["group"] = cut(df.value, range(0, 105, 10), right=False, labels=labels)


def test_cut_with_duplicated_index_lowest_included():
    # GH 42185
    expected = Series(
        [Interval(-0.001, 2, closed="right")] * 3
        + [Interval(2, 4, closed="right"), Interval(-0.001, 2, closed="right")],
        index=[0, 1, 2, 3, 0],
        dtype="category",
    ).cat.as_ordered()

    ser = Series([0, 1, 2, 3, 0], index=[0, 1, 2, 3, 0])
    result = cut(ser, bins=[0, 2, 4], include_lowest=True)
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_cut_with_nonexact_categorical_indices():
    # GH 42424

    ser = Series(range(100))
    ser1 = cut(ser, 10).value_counts().head(5)
    ser2 = cut(ser, 10).value_counts().tail(5)
    result = DataFrame({"1": ser1, "2": ser2})

    index = pd.CategoricalIndex(
        [
            Interval(-0.099, 9.9, closed="right"),
            Interval(9.9, 19.8, closed="right"),
            Interval(19.8, 29.7, closed="right"),
            Interval(29.7, 39.6, closed="right"),
            Interval(39.6, 49.5, closed="right"),
            Interval(49.5, 59.4, closed="right"),
            Interval(59.4, 69.3, closed="right"),
            Interval(69.3, 79.2, closed="right"),
            Interval(79.2, 89.1, closed="right"),
            Interval(89.1, 99, closed="right"),
        ],
        ordered=True,
    )

    expected = DataFrame(
        {"1": [10] * 5 + [np.nan] * 5, "2": [np.nan] * 5 + [10] * 5}, index=index
    )

    tm.assert_frame_equal(expected, result)


def test_cut_with_timestamp_tuple_labels():
    # GH 40661
    labels = [(Timestamp(10),), (Timestamp(20),), (Timestamp(30),)]
    result = cut([2, 4, 6], bins=[1, 3, 5, 7], labels=labels)

    expected = Categorical.from_codes([0, 1, 2], labels, ordered=True)
    tm.assert_categorical_equal(result, expected)


def test_cut_bins_datetime_intervalindex():
    # https://github.com/pandas-dev/pandas/issues/46218
    bins = interval_range(Timestamp("2022-02-25"), Timestamp("2022-02-27"), freq="1D")
    # passing Series instead of list is important to trigger bug
    result = cut(Series([Timestamp("2022-02-26")]).astype("M8[ns]"), bins=bins)
    expected = Categorical.from_codes([0], bins, ordered=True)
    tm.assert_categorical_equal(result.array, expected)


def test_cut_with_nullable_int64():
    # GH 30787
    series = Series([0, 1, 2, 3, 4, pd.NA, 6, 7], dtype="Int64")
    bins = [0, 2, 4, 6, 8]
    intervals = IntervalIndex.from_breaks(bins)

    expected = Series(
        Categorical.from_codes([-1, 0, 0, 1, 1, -1, 2, 3], intervals, ordered=True)
    )

    result = cut(series, bins=bins)

    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_overrides.py ===
import inspect
import os
import pickle
import sys
import tempfile
from io import StringIO
from unittest import mock

import pytest

import numpy as np
from numpy._core.overrides import (
    _get_implementing_args,
    array_function_dispatch,
    verify_matching_signatures,
)
from numpy.testing import assert_, assert_equal, assert_raises, assert_raises_regex
from numpy.testing.overrides import get_overridable_numpy_array_functions


def _return_not_implemented(self, *args, **kwargs):
    return NotImplemented


# need to define this at the top level to test pickling
@array_function_dispatch(lambda array: (array,))
def dispatched_one_arg(array):
    """Docstring."""
    return 'original'


@array_function_dispatch(lambda array1, array2: (array1, array2))
def dispatched_two_arg(array1, array2):
    """Docstring."""
    return 'original'


class TestGetImplementingArgs:

    def test_ndarray(self):
        array = np.array(1)

        args = _get_implementing_args([array])
        assert_equal(list(args), [array])

        args = _get_implementing_args([array, array])
        assert_equal(list(args), [array])

        args = _get_implementing_args([array, 1])
        assert_equal(list(args), [array])

        args = _get_implementing_args([1, array])
        assert_equal(list(args), [array])

    def test_ndarray_subclasses(self):

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        class NoOverrideSub(np.ndarray):
            pass

        array = np.array(1).view(np.ndarray)
        override_sub = np.array(1).view(OverrideSub)
        no_override_sub = np.array(1).view(NoOverrideSub)

        args = _get_implementing_args([array, override_sub])
        assert_equal(list(args), [override_sub, array])

        args = _get_implementing_args([array, no_override_sub])
        assert_equal(list(args), [no_override_sub, array])

        args = _get_implementing_args(
            [override_sub, no_override_sub])
        assert_equal(list(args), [override_sub, no_override_sub])

    def test_ndarray_and_duck_array(self):

        class Other:
            __array_function__ = _return_not_implemented

        array = np.array(1)
        other = Other()

        args = _get_implementing_args([other, array])
        assert_equal(list(args), [other, array])

        args = _get_implementing_args([array, other])
        assert_equal(list(args), [array, other])

    def test_ndarray_subclass_and_duck_array(self):

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        class Other:
            __array_function__ = _return_not_implemented

        array = np.array(1)
        subarray = np.array(1).view(OverrideSub)
        other = Other()

        assert_equal(_get_implementing_args([array, subarray, other]),
                     [subarray, array, other])
        assert_equal(_get_implementing_args([array, other, subarray]),
                     [subarray, array, other])

    def test_many_duck_arrays(self):

        class A:
            __array_function__ = _return_not_implemented

        class B(A):
            __array_function__ = _return_not_implemented

        class C(A):
            __array_function__ = _return_not_implemented

        class D:
            __array_function__ = _return_not_implemented

        a = A()
        b = B()
        c = C()
        d = D()

        assert_equal(_get_implementing_args([1]), [])
        assert_equal(_get_implementing_args([a]), [a])
        assert_equal(_get_implementing_args([a, 1]), [a])
        assert_equal(_get_implementing_args([a, a, a]), [a])
        assert_equal(_get_implementing_args([a, d, a]), [a, d])
        assert_equal(_get_implementing_args([a, b]), [b, a])
        assert_equal(_get_implementing_args([b, a]), [b, a])
        assert_equal(_get_implementing_args([a, b, c]), [b, c, a])
        assert_equal(_get_implementing_args([a, c, b]), [c, b, a])

    def test_too_many_duck_arrays(self):
        namespace = {'__array_function__': _return_not_implemented}
        types = [type('A' + str(i), (object,), namespace) for i in range(65)]
        relevant_args = [t() for t in types]

        actual = _get_implementing_args(relevant_args[:64])
        assert_equal(actual, relevant_args[:64])

        with assert_raises_regex(TypeError, 'distinct argument types'):
            _get_implementing_args(relevant_args)


class TestNDArrayArrayFunction:

    def test_method(self):

        class Other:
            __array_function__ = _return_not_implemented

        class NoOverrideSub(np.ndarray):
            pass

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        array = np.array([1])
        other = Other()
        no_override_sub = array.view(NoOverrideSub)
        override_sub = array.view(OverrideSub)

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray,),
                                          args=(array, 1.), kwargs={})
        assert_equal(result, 'original')

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, Other),
                                          args=(array, other), kwargs={})
        assert_(result is NotImplemented)

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, NoOverrideSub),
                                          args=(array, no_override_sub),
                                          kwargs={})
        assert_equal(result, 'original')

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, OverrideSub),
                                          args=(array, override_sub),
                                          kwargs={})
        assert_equal(result, 'original')

        with assert_raises_regex(TypeError, 'no implementation found'):
            np.concatenate((array, other))

        expected = np.concatenate((array, array))
        result = np.concatenate((array, no_override_sub))
        assert_equal(result, expected.view(NoOverrideSub))
        result = np.concatenate((array, override_sub))
        assert_equal(result, expected.view(OverrideSub))

    def test_no_wrapper(self):
        # Regular numpy functions have wrappers, but do not presume
        # all functions do (array creation ones do not): check that
        # we just call the function in that case.
        array = np.array(1)
        func = lambda x: x * 2
        result = array.__array_function__(func=func, types=(np.ndarray,),
                                          args=(array,), kwargs={})
        assert_equal(result, array * 2)

    def test_wrong_arguments(self):
        # Check our implementation guards against wrong arguments.
        a = np.array([1, 2])
        with pytest.raises(TypeError, match="args must be a tuple"):
            a.__array_function__(np.reshape, (np.ndarray,), a, (2, 1))
        with pytest.raises(TypeError, match="kwargs must be a dict"):
            a.__array_function__(np.reshape, (np.ndarray,), (a,), (2, 1))


class TestArrayFunctionDispatch:

    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            roundtripped = pickle.loads(
                    pickle.dumps(dispatched_one_arg, protocol=proto))
            assert_(roundtripped is dispatched_one_arg)

    def test_name_and_docstring(self):
        assert_equal(dispatched_one_arg.__name__, 'dispatched_one_arg')
        if sys.flags.optimize < 2:
            assert_equal(dispatched_one_arg.__doc__, 'Docstring.')

    def test_interface(self):

        class MyArray:
            def __array_function__(self, func, types, args, kwargs):
                return (self, func, types, args, kwargs)

        original = MyArray()
        (obj, func, types, args, kwargs) = dispatched_one_arg(original)
        assert_(obj is original)
        assert_(func is dispatched_one_arg)
        assert_equal(set(types), {MyArray})
        # assert_equal uses the overloaded np.iscomplexobj() internally
        assert_(args == (original,))
        assert_equal(kwargs, {})

    def test_not_implemented(self):

        class MyArray:
            def __array_function__(self, func, types, args, kwargs):
                return NotImplemented

        array = MyArray()
        with assert_raises_regex(TypeError, 'no implementation found'):
            dispatched_one_arg(array)

    def test_where_dispatch(self):

        class DuckArray:
            def __array_function__(self, ufunc, method, *inputs, **kwargs):
                return "overridden"

        array = np.array(1)
        duck_array = DuckArray()

        result = np.std(array, where=duck_array)

        assert_equal(result, "overridden")


class TestVerifyMatchingSignatures:

    def test_verify_matching_signatures(self):

        verify_matching_signatures(lambda x: 0, lambda x: 0)
        verify_matching_signatures(lambda x=None: 0, lambda x=None: 0)
        verify_matching_signatures(lambda x=1: 0, lambda x=None: 0)

        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda a: 0, lambda b: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x: 0, lambda x=None: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x=None: 0, lambda y=None: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x=1: 0, lambda y=1: 0)

    def test_array_function_dispatch(self):

        with assert_raises(RuntimeError):
            @array_function_dispatch(lambda x: (x,))
            def f(y):
                pass

        # should not raise
        @array_function_dispatch(lambda x: (x,), verify=False)
        def f(y):
            pass


def _new_duck_type_and_implements():
    """Create a duck array type and implements functions."""
    HANDLED_FUNCTIONS = {}

    class MyArray:
        def __array_function__(self, func, types, args, kwargs):
            if func not in HANDLED_FUNCTIONS:
                return NotImplemented
            if not all(issubclass(t, MyArray) for t in types):
                return NotImplemented
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def implements(numpy_function):
        """Register an __array_function__ implementations."""
        def decorator(func):
            HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator

    return (MyArray, implements)


class TestArrayFunctionImplementation:

    def test_one_arg(self):
        MyArray, implements = _new_duck_type_and_implements()

        @implements(dispatched_one_arg)
        def _(array):
            return 'myarray'

        assert_equal(dispatched_one_arg(1), 'original')
        assert_equal(dispatched_one_arg(MyArray()), 'myarray')

    def test_optional_args(self):
        MyArray, implements = _new_duck_type_and_implements()

        @array_function_dispatch(lambda array, option=None: (array,))
        def func_with_option(array, option='default'):
            return option

        @implements(func_with_option)
        def my_array_func_with_option(array, new_option='myarray'):
            return new_option

        # we don't need to implement every option on __array_function__
        # implementations
        assert_equal(func_with_option(1), 'default')
        assert_equal(func_with_option(1, option='extra'), 'extra')
        assert_equal(func_with_option(MyArray()), 'myarray')
        with assert_raises(TypeError):
            func_with_option(MyArray(), option='extra')

        # but new options on implementations can't be used
        result = my_array_func_with_option(MyArray(), new_option='yes')
        assert_equal(result, 'yes')
        with assert_raises(TypeError):
            func_with_option(MyArray(), new_option='no')

    def test_not_implemented(self):
        MyArray, implements = _new_duck_type_and_implements()

        @array_function_dispatch(lambda array: (array,), module='my')
        def func(array):
            return array

        array = np.array(1)
        assert_(func(array) is array)
        assert_equal(func.__module__, 'my')

        with assert_raises_regex(
                TypeError, "no implementation found for 'my.func'"):
            func(MyArray())

    @pytest.mark.parametrize("name", ["concatenate", "mean", "asarray"])
    def test_signature_error_message_simple(self, name):
        func = getattr(np, name)
        try:
            # all of these functions need an argument:
            func()
        except TypeError as e:
            exc = e

        assert exc.args[0].startswith(f"{name}()")

    def test_signature_error_message(self):
        # The lambda function will be named "<lambda>", but the TypeError
        # should show the name as "func"
        def _dispatcher():
            return ()

        @array_function_dispatch(_dispatcher)
        def func():
            pass

        try:
            func._implementation(bad_arg=3)
        except TypeError as e:
            expected_exception = e

        try:
            func(bad_arg=3)
            raise AssertionError("must fail")
        except TypeError as exc:
            if exc.args[0].startswith("_dispatcher"):
                # We replace the qualname currently, but it used `__name__`
                # (relevant functions have the same name and qualname anyway)
                pytest.skip("Python version is not using __qualname__ for "
                            "TypeError formatting.")

            assert exc.args == expected_exception.args

    @pytest.mark.parametrize("value", [234, "this func is not replaced"])
    def test_dispatcher_error(self, value):
        # If the dispatcher raises an error, we must not attempt to mutate it
        error = TypeError(value)

        def dispatcher():
            raise error

        @array_function_dispatch(dispatcher)
        def func():
            return 3

        try:
            func()
            raise AssertionError("must fail")
        except TypeError as exc:
            assert exc is error  # unmodified exception

    def test_properties(self):
        # Check that str and repr are sensible
        func = dispatched_two_arg
        assert str(func) == str(func._implementation)
        repr_no_id = repr(func).split("at ")[0]
        repr_no_id_impl = repr(func._implementation).split("at ")[0]
        assert repr_no_id == repr_no_id_impl

    @pytest.mark.parametrize("func", [
            lambda x, y: 0,  # no like argument
            lambda like=None: 0,  # not keyword only
            lambda *, like=None, a=3: 0,  # not last (not that it matters)
        ])
    def test_bad_like_sig(self, func):
        # We sanity check the signature, and these should fail.
        with pytest.raises(RuntimeError):
            array_function_dispatch()(func)

    def test_bad_like_passing(self):
        # Cover internal sanity check for passing like as first positional arg
        def func(*, like=None):
            pass

        func_with_like = array_function_dispatch()(func)
        with pytest.raises(TypeError):
            func_with_like()
        with pytest.raises(TypeError):
            func_with_like(like=234)

    def test_too_many_args(self):
        # Mainly a unit-test to increase coverage
        objs = []
        for i in range(80):
            class MyArr:
                def __array_function__(self, *args, **kwargs):
                    return NotImplemented

            objs.append(MyArr())

        def _dispatch(*args):
            return args

        @array_function_dispatch(_dispatch)
        def func(*args):
            pass

        with pytest.raises(TypeError, match="maximum number"):
            func(*objs)


class TestNDArrayMethods:

    def test_repr(self):
        # gh-12162: should still be defined even if __array_function__ doesn't
        # implement np.array_repr()

        class MyArray(np.ndarray):
            def __array_function__(*args, **kwargs):
                return NotImplemented

        array = np.array(1).view(MyArray)
        assert_equal(repr(array), 'MyArray(1)')
        assert_equal(str(array), '1')


class TestNumPyFunctions:

    def test_set_module(self):
        assert_equal(np.sum.__module__, 'numpy')
        assert_equal(np.char.equal.__module__, 'numpy.char')
        assert_equal(np.fft.fft.__module__, 'numpy.fft')
        assert_equal(np.linalg.solve.__module__, 'numpy.linalg')

    def test_inspect_sum(self):
        signature = inspect.signature(np.sum)
        assert_('axis' in signature.parameters)

    def test_override_sum(self):
        MyArray, implements = _new_duck_type_and_implements()

        @implements(np.sum)
        def _(array):
            return 'yes'

        assert_equal(np.sum(MyArray()), 'yes')

    def test_sum_on_mock_array(self):

        # We need a proxy for mocks because __array_function__ is only looked
        # up in the class dict
        class ArrayProxy:
            def __init__(self, value):
                self.value = value

            def __array_function__(self, *args, **kwargs):
                return self.value.__array_function__(*args, **kwargs)

            def __array__(self, *args, **kwargs):
                return self.value.__array__(*args, **kwargs)

        proxy = ArrayProxy(mock.Mock(spec=ArrayProxy))
        proxy.value.__array_function__.return_value = 1
        result = np.sum(proxy)
        assert_equal(result, 1)
        proxy.value.__array_function__.assert_called_once_with(
            np.sum, (ArrayProxy,), (proxy,), {})
        proxy.value.__array__.assert_not_called()

    def test_sum_forwarding_implementation(self):

        class MyArray(np.ndarray):

            def sum(self, axis, out):
                return 'summed'

            def __array_function__(self, func, types, args, kwargs):
                return super().__array_function__(func, types, args, kwargs)

        # note: the internal implementation of np.sum() calls the .sum() method
        array = np.array(1).view(MyArray)
        assert_equal(np.sum(array), 'summed')


class TestArrayLike:
    def setup_method(self):
        class MyArray:
            def __init__(self, function=None):
                self.function = function

            def __array_function__(self, func, types, args, kwargs):
                assert func is getattr(np, func.__name__)
                try:
                    my_func = getattr(self, func.__name__)
                except AttributeError:
                    return NotImplemented
                return my_func(*args, **kwargs)

        self.MyArray = MyArray

        class MyNoArrayFunctionArray:
            def __init__(self, function=None):
                self.function = function

        self.MyNoArrayFunctionArray = MyNoArrayFunctionArray

        class MySubclass(np.ndarray):
            def __array_function__(self, func, types, args, kwargs):
                result = super().__array_function__(func, types, args, kwargs)
                return result.view(self.__class__)

        self.MySubclass = MySubclass

    def add_method(self, name, arr_class, enable_value_error=False):
        def _definition(*args, **kwargs):
            # Check that `like=` isn't propagated downstream
            assert 'like' not in kwargs

            if enable_value_error and 'value_error' in kwargs:
                raise ValueError

            return arr_class(getattr(arr_class, name))
        setattr(arr_class, name, _definition)

    def func_args(*args, **kwargs):
        return args, kwargs

    def test_array_like_not_implemented(self):
        self.add_method('array', self.MyArray)

        ref = self.MyArray.array()

        with assert_raises_regex(TypeError, 'no implementation found'):
            array_like = np.asarray(1, like=ref)

    _array_tests = [
        ('array', *func_args((1,))),
        ('asarray', *func_args((1,))),
        ('asanyarray', *func_args((1,))),
        ('ascontiguousarray', *func_args((2, 3))),
        ('asfortranarray', *func_args((2, 3))),
        ('require', *func_args((np.arange(6).reshape(2, 3),),
                               requirements=['A', 'F'])),
        ('empty', *func_args((1,))),
        ('full', *func_args((1,), 2)),
        ('ones', *func_args((1,))),
        ('zeros', *func_args((1,))),
        ('arange', *func_args(3)),
        ('frombuffer', *func_args(b'\x00' * 8, dtype=int)),
        ('fromiter', *func_args(range(3), dtype=int)),
        ('fromstring', *func_args('1,2', dtype=int, sep=',')),
        ('loadtxt', *func_args(lambda: StringIO('0 1\n2 3'))),
        ('genfromtxt', *func_args(lambda: StringIO('1,2.1'),
                                  dtype=[('int', 'i8'), ('float', 'f8')],
                                  delimiter=',')),
    ]

    def test_nep35_functions_as_array_functions(self,):
        all_array_functions = get_overridable_numpy_array_functions()
        like_array_functions_subset = {
            getattr(np, func_name) for func_name, *_ in self.__class__._array_tests
        }
        assert like_array_functions_subset.issubset(all_array_functions)

        nep35_python_functions = {
            np.eye, np.fromfunction, np.full, np.genfromtxt,
            np.identity, np.loadtxt, np.ones, np.require, np.tri,
        }
        assert nep35_python_functions.issubset(all_array_functions)

        nep35_C_functions = {
            np.arange, np.array, np.asanyarray, np.asarray,
            np.ascontiguousarray, np.asfortranarray, np.empty,
            np.frombuffer, np.fromfile, np.fromiter, np.fromstring,
            np.zeros,
        }
        assert nep35_C_functions.issubset(all_array_functions)

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    @pytest.mark.parametrize('numpy_ref', [True, False])
    def test_array_like(self, function, args, kwargs, numpy_ref):
        self.add_method('array', self.MyArray)
        self.add_method(function, self.MyArray)
        np_func = getattr(np, function)
        my_func = getattr(self.MyArray, function)

        if numpy_ref is True:
            ref = np.array(1)
        else:
            ref = self.MyArray.array()

        like_args = tuple(a() if callable(a) else a for a in args)
        array_like = np_func(*like_args, **kwargs, like=ref)

        if numpy_ref is True:
            assert type(array_like) is np.ndarray

            np_args = tuple(a() if callable(a) else a for a in args)
            np_arr = np_func(*np_args, **kwargs)

            # Special-case np.empty to ensure values match
            if function == "empty":
                np_arr.fill(1)
                array_like.fill(1)

            assert_equal(array_like, np_arr)
        else:
            assert type(array_like) is self.MyArray
            assert array_like.function is my_func

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    @pytest.mark.parametrize('ref', [1, [1], "MyNoArrayFunctionArray"])
    def test_no_array_function_like(self, function, args, kwargs, ref):
        self.add_method('array', self.MyNoArrayFunctionArray)
        self.add_method(function, self.MyNoArrayFunctionArray)
        np_func = getattr(np, function)

        # Instantiate ref if it's the MyNoArrayFunctionArray class
        if ref == "MyNoArrayFunctionArray":
            ref = self.MyNoArrayFunctionArray.array()

        like_args = tuple(a() if callable(a) else a for a in args)

        with assert_raises_regex(TypeError,
                'The `like` argument must be an array-like that implements'):
            np_func(*like_args, **kwargs, like=ref)

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    def test_subclass(self, function, args, kwargs):
        ref = np.array(1).view(self.MySubclass)
        np_func = getattr(np, function)
        like_args = tuple(a() if callable(a) else a for a in args)
        array_like = np_func(*like_args, **kwargs, like=ref)
        assert type(array_like) is self.MySubclass
        if np_func is np.empty:
            return
        np_args = tuple(a() if callable(a) else a for a in args)
        np_arr = np_func(*np_args, **kwargs)
        assert_equal(array_like.view(np.ndarray), np_arr)

    @pytest.mark.parametrize('numpy_ref', [True, False])
    def test_array_like_fromfile(self, numpy_ref):
        self.add_method('array', self.MyArray)
        self.add_method("fromfile", self.MyArray)

        if numpy_ref is True:
            ref = np.array(1)
        else:
            ref = self.MyArray.array()

        data = np.random.random(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "testfile")
            data.tofile(fname)

            array_like = np.fromfile(fname, like=ref)
            if numpy_ref is True:
                assert type(array_like) is np.ndarray
                np_res = np.fromfile(fname, like=ref)
                assert_equal(np_res, data)
                assert_equal(array_like, np_res)
            else:
                assert type(array_like) is self.MyArray
                assert array_like.function is self.MyArray.fromfile

    def test_exception_handling(self):
        self.add_method('array', self.MyArray, enable_value_error=True)

        ref = self.MyArray.array()

        with assert_raises(TypeError):
            # Raises the error about `value_error` being invalid first
            np.array(1, value_error=True, like=ref)

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    def test_like_as_none(self, function, args, kwargs):
        self.add_method('array', self.MyArray)
        self.add_method(function, self.MyArray)
        np_func = getattr(np, function)

        like_args = tuple(a() if callable(a) else a for a in args)
        # required for loadtxt and genfromtxt to init w/o error.
        like_args_exp = tuple(a() if callable(a) else a for a in args)

        array_like = np_func(*like_args, **kwargs, like=None)
        expected = np_func(*like_args_exp, **kwargs)
        # Special-case np.empty to ensure values match
        if function == "empty":
            array_like.fill(1)
            expected.fill(1)
        assert_equal(array_like, expected)


def test_function_like():
    # We provide a `__get__` implementation, make sure it works
    assert type(np.mean) is np._core._multiarray_umath._ArrayFunctionDispatcher

    class MyClass:
        def __array__(self, dtype=None, copy=None):
            # valid argument to mean:
            return np.arange(3)

        func1 = staticmethod(np.mean)
        func2 = np.mean
        func3 = classmethod(np.mean)

    m = MyClass()
    assert m.func1([10]) == 10
    assert m.func2() == 1  # mean of the arange
    with pytest.raises(TypeError, match="unsupported operand type"):
        # Tries to operate on the class
        m.func3()

    # Manual binding also works (the above may shortcut):
    bound = np.mean.__get__(m, MyClass)
    assert bound() == 1

    bound = np.mean.__get__(None, MyClass)  # unbound actually
    assert bound([10]) == 10

    bound = np.mean.__get__(MyClass)  # classmethod
    with pytest.raises(TypeError, match="unsupported operand type"):
        bound()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_constructors.py ===
from datetime import (
    date,
    datetime,
)

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import HAS_PYARROW

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class TestCategoricalConstructors:
    def test_fastpath_deprecated(self):
        codes = np.array([1, 2, 3])
        dtype = CategoricalDtype(categories=["a", "b", "c", "d"], ordered=False)
        msg = "The 'fastpath' keyword in Categorical is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            Categorical(codes, dtype=dtype, fastpath=True)

    def test_categorical_from_cat_and_dtype_str_preserve_ordered(self):
        # GH#49309 we should preserve orderedness in `res`
        cat = Categorical([3, 1], categories=[3, 2, 1], ordered=True)

        res = Categorical(cat, dtype="category")
        assert res.dtype.ordered

    def test_categorical_disallows_scalar(self):
        # GH#38433
        with pytest.raises(TypeError, match="Categorical input must be list-like"):
            Categorical("A", categories=["A", "B"])

    def test_categorical_1d_only(self):
        # ndim > 1
        msg = "> 1 ndim Categorical are not supported at this time"
        with pytest.raises(NotImplementedError, match=msg):
            Categorical(np.array([list("abcd")]))

    def test_validate_ordered(self):
        # see gh-14058
        exp_msg = "'ordered' must either be 'True' or 'False'"
        exp_err = TypeError

        # This should be a boolean.
        ordered = np.array([0, 1, 2])

        with pytest.raises(exp_err, match=exp_msg):
            Categorical([1, 2, 3], ordered=ordered)

        with pytest.raises(exp_err, match=exp_msg):
            Categorical.from_codes(
                [0, 0, 1], categories=["a", "b", "c"], ordered=ordered
            )

    def test_constructor_empty(self):
        # GH 17248
        c = Categorical([])
        expected = Index([])
        tm.assert_index_equal(c.categories, expected)

        c = Categorical([], categories=[1, 2, 3])
        expected = Index([1, 2, 3], dtype=np.int64)
        tm.assert_index_equal(c.categories, expected)

    def test_constructor_empty_boolean(self):
        # see gh-22702
        cat = Categorical([], categories=[True, False])
        categories = sorted(cat.categories.tolist())
        assert categories == [False, True]

    def test_constructor_tuples(self):
        values = np.array([(1,), (1, 2), (1,), (1, 2)], dtype=object)
        result = Categorical(values)
        expected = Index([(1,), (1, 2)], tupleize_cols=False)
        tm.assert_index_equal(result.categories, expected)
        assert result.ordered is False

    def test_constructor_tuples_datetimes(self):
        # numpy will auto reshape when all of the tuples are the
        # same len, so add an extra one with 2 items and slice it off
        values = np.array(
            [
                (Timestamp("2010-01-01"),),
                (Timestamp("2010-01-02"),),
                (Timestamp("2010-01-01"),),
                (Timestamp("2010-01-02"),),
                ("a", "b"),
            ],
            dtype=object,
        )[:-1]
        result = Categorical(values)
        expected = Index(
            [(Timestamp("2010-01-01"),), (Timestamp("2010-01-02"),)],
            tupleize_cols=False,
        )
        tm.assert_index_equal(result.categories, expected)

    def test_constructor_unsortable(self):
        # it works!
        arr = np.array([1, 2, 3, datetime.now()], dtype="O")
        factor = Categorical(arr, ordered=False)
        assert not factor.ordered

        # this however will raise as cannot be sorted
        msg = (
            "'values' is not ordered, please explicitly specify the "
            "categories order by passing in a categories argument."
        )
        with pytest.raises(TypeError, match=msg):
            Categorical(arr, ordered=True)

    def test_constructor_interval(self):
        result = Categorical(
            [Interval(1, 2), Interval(2, 3), Interval(3, 6)], ordered=True
        )
        ii = IntervalIndex([Interval(1, 2), Interval(2, 3), Interval(3, 6)])
        exp = Categorical(ii, ordered=True)
        tm.assert_categorical_equal(result, exp)
        tm.assert_index_equal(result.categories, ii)

    def test_constructor(self):
        exp_arr = np.array(["a", "b", "c", "a", "b", "c"], dtype=np.object_)
        c1 = Categorical(exp_arr)
        tm.assert_numpy_array_equal(c1.__array__(), exp_arr)
        c2 = Categorical(exp_arr, categories=["a", "b", "c"])
        tm.assert_numpy_array_equal(c2.__array__(), exp_arr)
        c2 = Categorical(exp_arr, categories=["c", "b", "a"])
        tm.assert_numpy_array_equal(c2.__array__(), exp_arr)

        # categories must be unique
        msg = "Categorical categories must be unique"
        with pytest.raises(ValueError, match=msg):
            Categorical([1, 2], [1, 2, 2])

        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], ["a", "b", "b"])

        # The default should be unordered
        c1 = Categorical(["a", "b", "c", "a"])
        assert not c1.ordered

        # Categorical as input
        c1 = Categorical(["a", "b", "c", "a"])
        c2 = Categorical(c1)
        tm.assert_categorical_equal(c1, c2)

        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        c2 = Categorical(c1)
        tm.assert_categorical_equal(c1, c2)

        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "c", "b"])
        c2 = Categorical(c1)
        tm.assert_categorical_equal(c1, c2)

        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "c", "b"])
        c2 = Categorical(c1, categories=["a", "b", "c"])
        tm.assert_numpy_array_equal(c1.__array__(), c2.__array__())
        tm.assert_index_equal(c2.categories, Index(["a", "b", "c"]))

        # Series of dtype category
        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        c2 = Categorical(Series(c1))
        tm.assert_categorical_equal(c1, c2)

        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "c", "b"])
        c2 = Categorical(Series(c1))
        tm.assert_categorical_equal(c1, c2)

        # Series
        c1 = Categorical(["a", "b", "c", "a"])
        c2 = Categorical(Series(["a", "b", "c", "a"]))
        tm.assert_categorical_equal(c1, c2)

        c1 = Categorical(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        c2 = Categorical(Series(["a", "b", "c", "a"]), categories=["a", "b", "c", "d"])
        tm.assert_categorical_equal(c1, c2)

        # This should result in integer categories, not float!
        cat = Categorical([1, 2, 3, np.nan], categories=[1, 2, 3])
        assert is_integer_dtype(cat.categories)

        # https://github.com/pandas-dev/pandas/issues/3678
        cat = Categorical([np.nan, 1, 2, 3])
        assert is_integer_dtype(cat.categories)

        # this should result in floats
        cat = Categorical([np.nan, 1, 2.0, 3])
        assert is_float_dtype(cat.categories)

        cat = Categorical([np.nan, 1.0, 2.0, 3.0])
        assert is_float_dtype(cat.categories)

        # This doesn't work -> this would probably need some kind of "remember
        # the original type" feature to try to cast the array interface result
        # to...

        # vals = np.asarray(cat[cat.notna()])
        # assert is_integer_dtype(vals)

        # corner cases
        cat = Categorical([1])
        assert len(cat.categories) == 1
        assert cat.categories[0] == 1
        assert len(cat.codes) == 1
        assert cat.codes[0] == 0

        cat = Categorical(["a"])
        assert len(cat.categories) == 1
        assert cat.categories[0] == "a"
        assert len(cat.codes) == 1
        assert cat.codes[0] == 0

        # two arrays
        #  - when the first is an integer dtype and the second is not
        #  - when the resulting codes are all -1/NaN
        with tm.assert_produces_warning(None):
            Categorical([0, 1, 2, 0, 1, 2], categories=["a", "b", "c"])

        with tm.assert_produces_warning(None):
            Categorical([0, 1, 2, 0, 1, 2], categories=[3, 4, 5])

        # the next one are from the old docs
        with tm.assert_produces_warning(None):
            Categorical([0, 1, 2, 0, 1, 2], [1, 2, 3])
            cat = Categorical([1, 2], categories=[1, 2, 3])

        # this is a legitimate constructor
        with tm.assert_produces_warning(None):
            Categorical(np.array([], dtype="int64"), categories=[3, 2, 1], ordered=True)

    def test_constructor_with_existing_categories(self):
        # GH25318: constructing with pd.Series used to bogusly skip recoding
        # categories
        c0 = Categorical(["a", "b", "c", "a"])
        c1 = Categorical(["a", "b", "c", "a"], categories=["b", "c"])

        c2 = Categorical(c0, categories=c1.categories)
        tm.assert_categorical_equal(c1, c2)

        c3 = Categorical(Series(c0), categories=c1.categories)
        tm.assert_categorical_equal(c1, c3)

    def test_constructor_not_sequence(self):
        # https://github.com/pandas-dev/pandas/issues/16022
        msg = r"^Parameter 'categories' must be list-like, was"
        with pytest.raises(TypeError, match=msg):
            Categorical(["a", "b"], categories="a")

    def test_constructor_with_null(self):
        # Cannot have NaN in categories
        msg = "Categorical categories cannot be null"
        with pytest.raises(ValueError, match=msg):
            Categorical([np.nan, "a", "b", "c"], categories=[np.nan, "a", "b", "c"])

        with pytest.raises(ValueError, match=msg):
            Categorical([None, "a", "b", "c"], categories=[None, "a", "b", "c"])

        with pytest.raises(ValueError, match=msg):
            Categorical(
                DatetimeIndex(["nat", "20160101"]),
                categories=[NaT, Timestamp("20160101")],
            )

    def test_constructor_with_index(self):
        ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
        tm.assert_categorical_equal(ci.values, Categorical(ci))

        ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
        tm.assert_categorical_equal(
            ci.values, Categorical(ci.astype(object), categories=ci.categories)
        )

    def test_constructor_with_generator(self):
        # This was raising an Error in isna(single_val).any() because isna
        # returned a scalar for a generator

        exp = Categorical([0, 1, 2])
        cat = Categorical(x for x in [0, 1, 2])
        tm.assert_categorical_equal(cat, exp)
        cat = Categorical(range(3))
        tm.assert_categorical_equal(cat, exp)

        MultiIndex.from_product([range(5), ["a", "b", "c"]])

        # check that categories accept generators and sequences
        cat = Categorical([0, 1, 2], categories=(x for x in [0, 1, 2]))
        tm.assert_categorical_equal(cat, exp)
        cat = Categorical([0, 1, 2], categories=range(3))
        tm.assert_categorical_equal(cat, exp)

    def test_constructor_with_rangeindex(self):
        # RangeIndex is preserved in Categories
        rng = Index(range(3))

        cat = Categorical(rng)
        tm.assert_index_equal(cat.categories, rng, exact=True)

        cat = Categorical([1, 2, 0], categories=rng)
        tm.assert_index_equal(cat.categories, rng, exact=True)

    @pytest.mark.parametrize(
        "dtl",
        [
            date_range("1995-01-01 00:00:00", periods=5, freq="s"),
            date_range("1995-01-01 00:00:00", periods=5, freq="s", tz="US/Eastern"),
            timedelta_range("1 day", periods=5, freq="s"),
        ],
    )
    def test_constructor_with_datetimelike(self, dtl):
        # see gh-12077
        # constructor with a datetimelike and NaT

        s = Series(dtl)
        c = Categorical(s)

        expected = type(dtl)(s)
        expected._data.freq = None

        tm.assert_index_equal(c.categories, expected)
        tm.assert_numpy_array_equal(c.codes, np.arange(5, dtype="int8"))

        # with NaT
        s2 = s.copy()
        s2.iloc[-1] = NaT
        c = Categorical(s2)

        expected = type(dtl)(s2.dropna())
        expected._data.freq = None

        tm.assert_index_equal(c.categories, expected)

        exp = np.array([0, 1, 2, 3, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(c.codes, exp)

        result = repr(c)
        assert "NaT" in result

    def test_constructor_from_index_series_datetimetz(self):
        idx = date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern")
        idx = idx._with_freq(None)  # freq not preserved in result.categories
        result = Categorical(idx)
        tm.assert_index_equal(result.categories, idx)

        result = Categorical(Series(idx))
        tm.assert_index_equal(result.categories, idx)

    def test_constructor_date_objects(self):
        # we dont cast date objects to timestamps, matching Index constructor
        v = date.today()

        cat = Categorical([v, v])
        assert cat.categories.dtype == object
        assert type(cat.categories[0]) is date

    def test_constructor_from_index_series_timedelta(self):
        idx = timedelta_range("1 days", freq="D", periods=3)
        idx = idx._with_freq(None)  # freq not preserved in result.categories
        result = Categorical(idx)
        tm.assert_index_equal(result.categories, idx)

        result = Categorical(Series(idx))
        tm.assert_index_equal(result.categories, idx)

    def test_constructor_from_index_series_period(self):
        idx = period_range("2015-01-01", freq="D", periods=3)
        result = Categorical(idx)
        tm.assert_index_equal(result.categories, idx)

        result = Categorical(Series(idx))
        tm.assert_index_equal(result.categories, idx)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1.0, 1.2, 1.8, np.nan]),
            np.array([1, 2, 3], dtype="int64"),
            ["a", "b", "c", np.nan],
            [pd.Period("2014-01"), pd.Period("2014-02"), NaT],
            [Timestamp("2014-01-01"), Timestamp("2014-01-02"), NaT],
            [
                Timestamp("2014-01-01", tz="US/Eastern"),
                Timestamp("2014-01-02", tz="US/Eastern"),
                NaT,
            ],
        ],
    )
    def test_constructor_invariant(self, values):
        # GH 14190
        c = Categorical(values)
        c2 = Categorical(c)
        tm.assert_categorical_equal(c, c2)

    @pytest.mark.parametrize("ordered", [True, False])
    def test_constructor_with_dtype(self, ordered):
        categories = ["b", "a", "c"]
        dtype = CategoricalDtype(categories, ordered=ordered)
        result = Categorical(["a", "b", "a", "c"], dtype=dtype)
        expected = Categorical(
            ["a", "b", "a", "c"], categories=categories, ordered=ordered
        )
        tm.assert_categorical_equal(result, expected)
        assert result.ordered is ordered

    def test_constructor_dtype_and_others_raises(self):
        dtype = CategoricalDtype(["a", "b"], ordered=True)
        msg = "Cannot specify `categories` or `ordered` together with `dtype`."
        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], categories=["a", "b"], dtype=dtype)

        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], ordered=True, dtype=dtype)

        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], ordered=False, dtype=dtype)

    @pytest.mark.parametrize("categories", [None, ["a", "b"], ["a", "c"]])
    @pytest.mark.parametrize("ordered", [True, False])
    def test_constructor_str_category(self, categories, ordered):
        result = Categorical(
            ["a", "b"], categories=categories, ordered=ordered, dtype="category"
        )
        expected = Categorical(["a", "b"], categories=categories, ordered=ordered)
        tm.assert_categorical_equal(result, expected)

    def test_constructor_str_unknown(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            Categorical([1, 2], dtype="foo")

    @pytest.mark.xfail(
        using_string_dtype() and HAS_PYARROW, reason="Can't be NumPy strings"
    )
    def test_constructor_np_strs(self):
        # GH#31499 Hashtable.map_locations needs to work on np.str_ objects
        cat = Categorical(["1", "0", "1"], [np.str_("0"), np.str_("1")])
        assert all(isinstance(x, np.str_) for x in cat.categories)

    def test_constructor_from_categorical_with_dtype(self):
        dtype = CategoricalDtype(["a", "b", "c"], ordered=True)
        values = Categorical(["a", "b", "d"])
        result = Categorical(values, dtype=dtype)
        # We use dtype.categories, not values.categories
        expected = Categorical(
            ["a", "b", "d"], categories=["a", "b", "c"], ordered=True
        )
        tm.assert_categorical_equal(result, expected)

    def test_constructor_from_categorical_with_unknown_dtype(self):
        dtype = CategoricalDtype(None, ordered=True)
        values = Categorical(["a", "b", "d"])
        result = Categorical(values, dtype=dtype)
        # We use values.categories, not dtype.categories
        expected = Categorical(
            ["a", "b", "d"], categories=["a", "b", "d"], ordered=True
        )
        tm.assert_categorical_equal(result, expected)

    def test_constructor_from_categorical_string(self):
        values = Categorical(["a", "b", "d"])
        # use categories, ordered
        result = Categorical(
            values, categories=["a", "b", "c"], ordered=True, dtype="category"
        )
        expected = Categorical(
            ["a", "b", "d"], categories=["a", "b", "c"], ordered=True
        )
        tm.assert_categorical_equal(result, expected)

        # No string
        result = Categorical(values, categories=["a", "b", "c"], ordered=True)
        tm.assert_categorical_equal(result, expected)

    def test_constructor_with_categorical_categories(self):
        # GH17884
        expected = Categorical(["a", "b"], categories=["a", "b", "c"])

        result = Categorical(["a", "b"], categories=Categorical(["a", "b", "c"]))
        tm.assert_categorical_equal(result, expected)

        result = Categorical(["a", "b"], categories=CategoricalIndex(["a", "b", "c"]))
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("klass", [lambda x: np.array(x, dtype=object), list])
    def test_construction_with_null(self, klass, nulls_fixture):
        # https://github.com/pandas-dev/pandas/issues/31927
        values = klass(["a", nulls_fixture, "b"])
        result = Categorical(values)

        dtype = CategoricalDtype(["a", "b"])
        codes = [0, -1, 1]
        expected = Categorical.from_codes(codes=codes, dtype=dtype)

        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("validate", [True, False])
    def test_from_codes_nullable_int_categories(self, any_numeric_ea_dtype, validate):
        # GH#39649
        cats = pd.array(range(5), dtype=any_numeric_ea_dtype)
        codes = np.random.default_rng(2).integers(5, size=3)
        dtype = CategoricalDtype(cats)
        arr = Categorical.from_codes(codes, dtype=dtype, validate=validate)
        assert arr.categories.dtype == cats.dtype
        tm.assert_index_equal(arr.categories, Index(cats))

    def test_from_codes_empty(self):
        cat = ["a", "b", "c"]
        result = Categorical.from_codes([], categories=cat)
        expected = Categorical([], categories=cat)

        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("validate", [True, False])
    def test_from_codes_validate(self, validate):
        # GH53122
        dtype = CategoricalDtype(["a", "b"])
        if validate:
            with pytest.raises(ValueError, match="codes need to be between "):
                Categorical.from_codes([4, 5], dtype=dtype, validate=validate)
        else:
            # passes, though has incorrect codes, but that's the user responsibility
            Categorical.from_codes([4, 5], dtype=dtype, validate=validate)

    def test_from_codes_too_few_categories(self):
        dtype = CategoricalDtype(categories=[1, 2])
        msg = "codes need to be between "
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([1, 2], categories=dtype.categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([1, 2], dtype=dtype)

    def test_from_codes_non_int_codes(self):
        dtype = CategoricalDtype(categories=[1, 2])
        msg = "codes need to be array-like integers"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(["a"], categories=dtype.categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(["a"], dtype=dtype)

    def test_from_codes_non_unique_categories(self):
        with pytest.raises(ValueError, match="Categorical categories must be unique"):
            Categorical.from_codes([0, 1, 2], categories=["a", "a", "b"])

    def test_from_codes_nan_cat_included(self):
        with pytest.raises(ValueError, match="Categorical categories cannot be null"):
            Categorical.from_codes([0, 1, 2], categories=["a", "b", np.nan])

    def test_from_codes_too_negative(self):
        dtype = CategoricalDtype(categories=["a", "b", "c"])
        msg = r"codes need to be between -1 and len\(categories\)-1"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([-2, 1, 2], categories=dtype.categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([-2, 1, 2], dtype=dtype)

    def test_from_codes(self):
        dtype = CategoricalDtype(categories=["a", "b", "c"])
        exp = Categorical(["a", "b", "c"], ordered=False)
        res = Categorical.from_codes([0, 1, 2], categories=dtype.categories)
        tm.assert_categorical_equal(exp, res)

        res = Categorical.from_codes([0, 1, 2], dtype=dtype)
        tm.assert_categorical_equal(exp, res)

    @pytest.mark.parametrize("klass", [Categorical, CategoricalIndex])
    def test_from_codes_with_categorical_categories(self, klass):
        # GH17884
        expected = Categorical(["a", "b"], categories=["a", "b", "c"])

        result = Categorical.from_codes([0, 1], categories=klass(["a", "b", "c"]))
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("klass", [Categorical, CategoricalIndex])
    def test_from_codes_with_non_unique_categorical_categories(self, klass):
        with pytest.raises(ValueError, match="Categorical categories must be unique"):
            Categorical.from_codes([0, 1], klass(["a", "b", "a"]))

    def test_from_codes_with_nan_code(self):
        # GH21767
        codes = [1, 2, np.nan]
        dtype = CategoricalDtype(categories=["a", "b", "c"])
        with pytest.raises(ValueError, match="codes need to be array-like integers"):
            Categorical.from_codes(codes, categories=dtype.categories)
        with pytest.raises(ValueError, match="codes need to be array-like integers"):
            Categorical.from_codes(codes, dtype=dtype)

    @pytest.mark.parametrize("codes", [[1.0, 2.0, 0], [1.1, 2.0, 0]])
    def test_from_codes_with_float(self, codes):
        # GH21767
        # float codes should raise even if values are equal to integers
        dtype = CategoricalDtype(categories=["a", "b", "c"])

        msg = "codes need to be array-like integers"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes, dtype.categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes, dtype=dtype)

    def test_from_codes_with_dtype_raises(self):
        msg = "Cannot specify"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(
                [0, 1], categories=["a", "b"], dtype=CategoricalDtype(["a", "b"])
            )

        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(
                [0, 1], ordered=True, dtype=CategoricalDtype(["a", "b"])
            )

    def test_from_codes_neither(self):
        msg = "Both were None"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([0, 1])

    def test_from_codes_with_nullable_int(self):
        codes = pd.array([0, 1], dtype="Int64")
        categories = ["a", "b"]

        result = Categorical.from_codes(codes, categories=categories)
        expected = Categorical.from_codes(codes.to_numpy(int), categories=categories)

        tm.assert_categorical_equal(result, expected)

    def test_from_codes_with_nullable_int_na_raises(self):
        codes = pd.array([0, None], dtype="Int64")
        categories = ["a", "b"]

        msg = "codes cannot contain NA values"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes, categories=categories)

    @pytest.mark.parametrize("dtype", [None, "category"])
    def test_from_inferred_categories(self, dtype):
        cats = ["a", "b"]
        codes = np.array([0, 0, 1, 1], dtype="i8")
        result = Categorical._from_inferred_categories(cats, codes, dtype)
        expected = Categorical.from_codes(codes, cats)
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("dtype", [None, "category"])
    def test_from_inferred_categories_sorts(self, dtype):
        cats = ["b", "a"]
        codes = np.array([0, 1, 1, 1], dtype="i8")
        result = Categorical._from_inferred_categories(cats, codes, dtype)
        expected = Categorical.from_codes([1, 0, 0, 0], ["a", "b"])
        tm.assert_categorical_equal(result, expected)

    def test_from_inferred_categories_dtype(self):
        cats = ["a", "b", "d"]
        codes = np.array([0, 1, 0, 2], dtype="i8")
        dtype = CategoricalDtype(["c", "b", "a"], ordered=True)
        result = Categorical._from_inferred_categories(cats, codes, dtype)
        expected = Categorical(
            ["a", "b", "a", "d"], categories=["c", "b", "a"], ordered=True
        )
        tm.assert_categorical_equal(result, expected)

    def test_from_inferred_categories_coerces(self):
        cats = ["1", "2", "bad"]
        codes = np.array([0, 0, 1, 2], dtype="i8")
        dtype = CategoricalDtype([1, 2])
        result = Categorical._from_inferred_categories(cats, codes, dtype)
        expected = Categorical([1, 1, 2, np.nan])
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("ordered", [None, True, False])
    def test_construction_with_ordered(self, ordered):
        # GH 9347, 9190
        cat = Categorical([0, 1, 2], ordered=ordered)
        assert cat.ordered == bool(ordered)

    def test_constructor_imaginary(self):
        values = [1, 2, 3 + 1j]
        c1 = Categorical(values)
        tm.assert_index_equal(c1.categories, Index(values))
        tm.assert_numpy_array_equal(np.array(c1), np.array(values))

    def test_constructor_string_and_tuples(self):
        # GH 21416
        c = Categorical(np.array(["c", ("a", "b"), ("b", "a"), "c"], dtype=object))
        expected_index = Index([("a", "b"), ("b", "a"), "c"])
        assert c.categories.equals(expected_index)

    def test_interval(self):
        idx = pd.interval_range(0, 10, periods=10)
        cat = Categorical(idx, categories=idx)
        expected_codes = np.arange(10, dtype="int8")
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

        # infer categories
        cat = Categorical(idx)
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

        # list values
        cat = Categorical(list(idx))
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

        # list values, categories
        cat = Categorical(list(idx), categories=list(idx))
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

        # shuffled
        values = idx.take([1, 2, 0])
        cat = Categorical(values, categories=idx)
        tm.assert_numpy_array_equal(cat.codes, np.array([1, 2, 0], dtype="int8"))
        tm.assert_index_equal(cat.categories, idx)

        # extra
        values = pd.interval_range(8, 11, periods=3)
        cat = Categorical(values, categories=idx)
        expected_codes = np.array([8, 9, -1], dtype="int8")
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

        # overlapping
        idx = IntervalIndex([Interval(0, 2), Interval(0, 1)])
        cat = Categorical(idx, categories=idx)
        expected_codes = np.array([0, 1], dtype="int8")
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        tm.assert_index_equal(cat.categories, idx)

    def test_categorical_extension_array_nullable(self, nulls_fixture):
        # GH:
        arr = pd.arrays.StringArray._from_sequence(
            [nulls_fixture] * 2, dtype=pd.StringDtype()
        )
        result = Categorical(arr)
        assert arr.dtype == result.categories.dtype
        expected = Categorical(Series([pd.NA, pd.NA], dtype=arr.dtype))
        tm.assert_categorical_equal(result, expected)

    def test_from_sequence_copy(self):
        cat = Categorical(np.arange(5).repeat(2))
        result = Categorical._from_sequence(cat, dtype=cat.dtype, copy=False)

        # more generally, we'd be OK with a view
        assert result._codes is cat._codes

        result = Categorical._from_sequence(cat, dtype=cat.dtype, copy=True)

        assert not tm.shares_memory(result, cat)

    def test_constructor_datetime64_non_nano(self):
        categories = np.arange(10).view("M8[D]")
        values = categories[::2].copy()

        cat = Categorical(values, categories=categories)
        assert (cat == values).all()

    def test_constructor_preserves_freq(self):
        # GH33830 freq retention in categorical
        dti = date_range("2016-01-01", periods=5)

        expected = dti.freq

        cat = Categorical(dti)
        result = cat.categories.freq

        assert expected == result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\string_\test_string.py ===
"""
This module tests the functionality of StringArray and ArrowStringArray.
Tests for the str accessors are in pandas/tests/strings/test_string_array.py
"""
import operator

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat.pyarrow import (
    pa_version_under12p0,
    pa_version_under19p0,
)

from pandas.core.dtypes.common import is_dtype_equal

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import StringArrayNumpySemantics
from pandas.core.arrays.string_arrow import (
    ArrowStringArray,
    ArrowStringArrayNumpySemantics,
)


@pytest.fixture
def dtype(string_dtype_arguments):
    """Fixture giving StringDtype from parametrized storage and na_value arguments"""
    storage, na_value = string_dtype_arguments
    return pd.StringDtype(storage=storage, na_value=na_value)


@pytest.fixture
def dtype2(string_dtype_arguments2):
    storage, na_value = string_dtype_arguments2
    return pd.StringDtype(storage=storage, na_value=na_value)


@pytest.fixture
def cls(dtype):
    """Fixture giving array type from parametrized 'dtype'"""
    return dtype.construct_array_type()


def test_dtype_constructor():
    pytest.importorskip("pyarrow")

    with tm.assert_produces_warning(FutureWarning):
        dtype = pd.StringDtype("pyarrow_numpy")
    assert dtype == pd.StringDtype("pyarrow", na_value=np.nan)


def test_dtype_equality():
    pytest.importorskip("pyarrow")

    dtype1 = pd.StringDtype("python")
    dtype2 = pd.StringDtype("pyarrow")
    dtype3 = pd.StringDtype("pyarrow", na_value=np.nan)

    assert dtype1 == pd.StringDtype("python", na_value=pd.NA)
    assert dtype1 != dtype2
    assert dtype1 != dtype3

    assert dtype2 == pd.StringDtype("pyarrow", na_value=pd.NA)
    assert dtype2 != dtype1
    assert dtype2 != dtype3

    assert dtype3 == pd.StringDtype("pyarrow", na_value=np.nan)
    assert dtype3 == pd.StringDtype("pyarrow", na_value=float("nan"))
    assert dtype3 != dtype1
    assert dtype3 != dtype2


def test_repr(dtype):
    df = pd.DataFrame({"A": pd.array(["a", pd.NA, "b"], dtype=dtype)})
    if dtype.na_value is np.nan:
        expected = "     A\n0    a\n1  NaN\n2    b"
    else:
        expected = "      A\n0     a\n1  <NA>\n2     b"
    assert repr(df) == expected

    if dtype.na_value is np.nan:
        expected = "0      a\n1    NaN\n2      b\nName: A, dtype: str"
    else:
        expected = "0       a\n1    <NA>\n2       b\nName: A, dtype: string"
    assert repr(df.A) == expected

    if dtype.storage == "pyarrow" and dtype.na_value is pd.NA:
        arr_name = "ArrowStringArray"
        expected = f"<{arr_name}>\n['a', <NA>, 'b']\nLength: 3, dtype: string"
    elif dtype.storage == "pyarrow" and dtype.na_value is np.nan:
        arr_name = "ArrowStringArrayNumpySemantics"
        expected = f"<{arr_name}>\n['a', nan, 'b']\nLength: 3, dtype: str"
    elif dtype.storage == "python" and dtype.na_value is np.nan:
        arr_name = "StringArrayNumpySemantics"
        expected = f"<{arr_name}>\n['a', nan, 'b']\nLength: 3, dtype: str"
    else:
        arr_name = "StringArray"
        expected = f"<{arr_name}>\n['a', <NA>, 'b']\nLength: 3, dtype: string"
    assert repr(df.A.array) == expected


def test_none_to_nan(cls, dtype):
    a = cls._from_sequence(["a", None, "b"], dtype=dtype)
    assert a[1] is not None
    assert a[1] is a.dtype.na_value


def test_setitem_validates(cls, dtype):
    arr = cls._from_sequence(["a", "b"], dtype=dtype)

    msg = "Invalid value '10' for dtype 'str"
    with pytest.raises(TypeError, match=msg):
        arr[0] = 10

    msg = "Invalid value for dtype 'str"
    with pytest.raises(TypeError, match=msg):
        arr[:] = np.array([1, 2])


def test_setitem_with_scalar_string(dtype):
    # is_float_dtype considers some strings, like 'd', to be floats
    # which can cause issues.
    arr = pd.array(["a", "c"], dtype=dtype)
    arr[0] = "d"
    expected = pd.array(["d", "c"], dtype=dtype)
    tm.assert_extension_array_equal(arr, expected)


def test_setitem_with_array_with_missing(dtype):
    # ensure that when setting with an array of values, we don't mutate the
    # array `value` in __setitem__(self, key, value)
    arr = pd.array(["a", "b", "c"], dtype=dtype)
    value = np.array(["A", None])
    value_orig = value.copy()
    arr[[0, 1]] = value

    expected = pd.array(["A", pd.NA, "c"], dtype=dtype)
    tm.assert_extension_array_equal(arr, expected)
    tm.assert_numpy_array_equal(value, value_orig)


def test_astype_roundtrip(dtype):
    ser = pd.Series(pd.date_range("2000", periods=12))
    ser[0] = None

    casted = ser.astype(dtype)
    assert is_dtype_equal(casted.dtype, dtype)

    result = casted.astype("datetime64[ns]")
    tm.assert_series_equal(result, ser)

    # GH#38509 same thing for timedelta64
    ser2 = ser - ser.iloc[-1]
    casted2 = ser2.astype(dtype)
    assert is_dtype_equal(casted2.dtype, dtype)

    result2 = casted2.astype(ser2.dtype)
    tm.assert_series_equal(result2, ser2)


def test_add(dtype):
    a = pd.Series(["a", "b", "c", None, None], dtype=dtype)
    b = pd.Series(["x", "y", None, "z", None], dtype=dtype)

    result = a + b
    expected = pd.Series(["ax", "by", None, None, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = a.add(b)
    tm.assert_series_equal(result, expected)

    result = a.radd(b)
    expected = pd.Series(["xa", "yb", None, None, None], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = a.add(b, fill_value="-")
    expected = pd.Series(["ax", "by", "c-", "-z", None], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_add_2d(dtype, request):
    if dtype.storage == "pyarrow":
        reason = "Failed: DID NOT RAISE <class 'ValueError'>"
        mark = pytest.mark.xfail(raises=None, reason=reason)
        request.applymarker(mark)

    a = pd.array(["a", "b", "c"], dtype=dtype)
    b = np.array([["a", "b", "c"]], dtype=object)
    with pytest.raises(ValueError, match="3 != 1"):
        a + b

    s = pd.Series(a)
    with pytest.raises(ValueError, match="3 != 1"):
        s + b


def test_add_sequence(dtype):
    a = pd.array(["a", "b", None, None], dtype=dtype)
    other = ["x", None, "y", None]

    result = a + other
    expected = pd.array(["ax", None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = other + a
    expected = pd.array(["xa", None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_mul(dtype):
    a = pd.array(["a", "b", None], dtype=dtype)
    result = a * 2
    expected = pd.array(["aa", "bb", None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = 2 * a
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.xfail(reason="GH-28527")
def test_add_strings(dtype):
    arr = pd.array(["a", "b", "c", "d"], dtype=dtype)
    df = pd.DataFrame([["t", "y", "v", "w"]], dtype=object)
    assert arr.__add__(df) is NotImplemented

    result = arr + df
    expected = pd.DataFrame([["at", "by", "cv", "dw"]]).astype(dtype)
    tm.assert_frame_equal(result, expected)

    result = df + arr
    expected = pd.DataFrame([["ta", "yb", "vc", "wd"]]).astype(dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(reason="GH-28527")
def test_add_frame(dtype):
    arr = pd.array(["a", "b", np.nan, np.nan], dtype=dtype)
    df = pd.DataFrame([["x", np.nan, "y", np.nan]])

    assert arr.__add__(df) is NotImplemented

    result = arr + df
    expected = pd.DataFrame([["ax", np.nan, np.nan, np.nan]]).astype(dtype)
    tm.assert_frame_equal(result, expected)

    result = df + arr
    expected = pd.DataFrame([["xa", np.nan, np.nan, np.nan]]).astype(dtype)
    tm.assert_frame_equal(result, expected)


def test_comparison_methods_scalar(comparison_op, dtype):
    op_name = f"__{comparison_op.__name__}__"
    a = pd.array(["a", None, "c"], dtype=dtype)
    other = "a"
    result = getattr(a, op_name)(other)
    if dtype.na_value is np.nan:
        expected = np.array([getattr(item, op_name)(other) for item in a])
        if comparison_op == operator.ne:
            expected[1] = True
        else:
            expected[1] = False
        tm.assert_numpy_array_equal(result, expected.astype(np.bool_))
    else:
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = np.array([getattr(item, op_name)(other) for item in a], dtype=object)
        expected = pd.array(expected, dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_scalar_pd_na(comparison_op, dtype):
    op_name = f"__{comparison_op.__name__}__"
    a = pd.array(["a", None, "c"], dtype=dtype)
    result = getattr(a, op_name)(pd.NA)

    if dtype.na_value is np.nan:
        if operator.ne == comparison_op:
            expected = np.array([True, True, True])
        else:
            expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)
    else:
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = pd.array([None, None, None], dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)
        tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_scalar_not_string(comparison_op, dtype):
    op_name = f"__{comparison_op.__name__}__"

    a = pd.array(["a", None, "c"], dtype=dtype)
    other = 42

    if op_name not in ["__eq__", "__ne__"]:
        with pytest.raises(TypeError, match="Invalid comparison|not supported between"):
            getattr(a, op_name)(other)

        return

    result = getattr(a, op_name)(other)

    if dtype.na_value is np.nan:
        expected_data = {
            "__eq__": [False, False, False],
            "__ne__": [True, True, True],
        }[op_name]
        expected = np.array(expected_data)
        tm.assert_numpy_array_equal(result, expected)
    else:
        expected_data = {"__eq__": [False, None, False], "__ne__": [True, None, True]}[
            op_name
        ]
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = pd.array(expected_data, dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)


def test_comparison_methods_array(comparison_op, dtype):
    op_name = f"__{comparison_op.__name__}__"

    a = pd.array(["a", None, "c"], dtype=dtype)
    other = [None, None, "c"]
    result = getattr(a, op_name)(other)
    if dtype.na_value is np.nan:
        if operator.ne == comparison_op:
            expected = np.array([True, True, False])
        else:
            expected = np.array([False, False, False])
            expected[-1] = getattr(other[-1], op_name)(a[-1])
        tm.assert_numpy_array_equal(result, expected)

        result = getattr(a, op_name)(pd.NA)
        if operator.ne == comparison_op:
            expected = np.array([True, True, True])
        else:
            expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(result, expected)

    else:
        expected_dtype = "boolean[pyarrow]" if dtype.storage == "pyarrow" else "boolean"
        expected = np.full(len(a), fill_value=None, dtype="object")
        expected[-1] = getattr(other[-1], op_name)(a[-1])
        expected = pd.array(expected, dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)

        result = getattr(a, op_name)(pd.NA)
        expected = pd.array([None, None, None], dtype=expected_dtype)
        tm.assert_extension_array_equal(result, expected)


def test_constructor_raises(cls):
    if cls is pd.arrays.StringArray:
        msg = "StringArray requires a sequence of strings or pandas.NA"
    elif cls is StringArrayNumpySemantics:
        msg = "StringArrayNumpySemantics requires a sequence of strings or NaN"
    else:
        msg = "Unsupported type '<class 'numpy.ndarray'>' for ArrowExtensionArray"

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", "b"], dtype="S1"))

    with pytest.raises(ValueError, match=msg):
        cls(np.array([]))

    if cls is pd.arrays.StringArray or cls is StringArrayNumpySemantics:
        # GH#45057 np.nan and None do NOT raise, as they are considered valid NAs
        #  for string dtype
        cls(np.array(["a", np.nan], dtype=object))
        cls(np.array(["a", None], dtype=object))
    else:
        with pytest.raises(ValueError, match=msg):
            cls(np.array(["a", np.nan], dtype=object))
        with pytest.raises(ValueError, match=msg):
            cls(np.array(["a", None], dtype=object))

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", pd.NaT], dtype=object))

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", np.datetime64("NaT", "ns")], dtype=object))

    with pytest.raises(ValueError, match=msg):
        cls(np.array(["a", np.timedelta64("NaT", "ns")], dtype=object))


@pytest.mark.parametrize("na", [np.nan, np.float64("nan"), float("nan"), None, pd.NA])
def test_constructor_nan_like(na):
    expected = pd.arrays.StringArray(np.array(["a", pd.NA]))
    tm.assert_extension_array_equal(
        pd.arrays.StringArray(np.array(["a", na], dtype="object")), expected
    )


@pytest.mark.parametrize("copy", [True, False])
def test_from_sequence_no_mutate(copy, cls, dtype):
    nan_arr = np.array(["a", np.nan], dtype=object)
    expected_input = nan_arr.copy()
    na_arr = np.array(["a", pd.NA], dtype=object)

    result = cls._from_sequence(nan_arr, dtype=dtype, copy=copy)

    if cls in (ArrowStringArray, ArrowStringArrayNumpySemantics):
        import pyarrow as pa

        expected = cls(pa.array(na_arr, type=pa.string(), from_pandas=True))
    elif cls is StringArrayNumpySemantics:
        expected = cls(nan_arr)
    else:
        expected = cls(na_arr)

    tm.assert_extension_array_equal(result, expected)
    tm.assert_numpy_array_equal(nan_arr, expected_input)


def test_astype_int(dtype):
    arr = pd.array(["1", "2", "3"], dtype=dtype)
    result = arr.astype("int64")
    expected = np.array([1, 2, 3], dtype="int64")
    tm.assert_numpy_array_equal(result, expected)

    arr = pd.array(["1", pd.NA, "3"], dtype=dtype)
    if dtype.na_value is np.nan:
        err = ValueError
        msg = "cannot convert float NaN to integer"
    else:
        err = TypeError
        msg = (
            r"int\(\) argument must be a string, a bytes-like "
            r"object or a( real)? number"
        )
    with pytest.raises(err, match=msg):
        arr.astype("int64")


def test_astype_nullable_int(dtype):
    arr = pd.array(["1", pd.NA, "3"], dtype=dtype)

    result = arr.astype("Int64")
    expected = pd.array([1, pd.NA, 3], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)


def test_astype_float(dtype, any_float_dtype):
    # Don't compare arrays (37974)
    ser = pd.Series(["1.1", pd.NA, "3.3"], dtype=dtype)
    result = ser.astype(any_float_dtype)
    expected = pd.Series([1.1, np.nan, 3.3], dtype=any_float_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("skipna", [True, False])
def test_reduce(skipna, dtype):
    arr = pd.Series(["a", "b", "c"], dtype=dtype)
    result = arr.sum(skipna=skipna)
    assert result == "abc"


@pytest.mark.parametrize("skipna", [True, False])
def test_reduce_missing(skipna, dtype):
    arr = pd.Series([None, "a", None, "b", "c", None], dtype=dtype)
    result = arr.sum(skipna=skipna)
    if skipna:
        assert result == "abc"
    else:
        assert pd.isna(result)


@pytest.mark.parametrize("method", ["min", "max"])
@pytest.mark.parametrize("skipna", [True, False])
def test_min_max(method, skipna, dtype):
    arr = pd.Series(["a", "b", "c", None], dtype=dtype)
    result = getattr(arr, method)(skipna=skipna)
    if skipna:
        expected = "a" if method == "min" else "c"
        assert result == expected
    else:
        assert result is arr.dtype.na_value


@pytest.mark.parametrize("method", ["min", "max"])
@pytest.mark.parametrize("box", [pd.Series, pd.array])
def test_min_max_numpy(method, box, dtype, request):
    if dtype.storage == "pyarrow" and box is pd.array:
        if box is pd.array:
            reason = "'<=' not supported between instances of 'str' and 'NoneType'"
        else:
            reason = "'ArrowStringArray' object has no attribute 'max'"
        mark = pytest.mark.xfail(raises=TypeError, reason=reason)
        request.applymarker(mark)

    arr = box(["a", "b", "c", None], dtype=dtype)
    result = getattr(np, method)(arr)
    expected = "a" if method == "min" else "c"
    assert result == expected


def test_fillna_args(dtype):
    # GH 37987

    arr = pd.array(["a", pd.NA], dtype=dtype)

    res = arr.fillna(value="b")
    expected = pd.array(["a", "b"], dtype=dtype)
    tm.assert_extension_array_equal(res, expected)

    res = arr.fillna(value=np.str_("b"))
    expected = pd.array(["a", "b"], dtype=dtype)
    tm.assert_extension_array_equal(res, expected)

    msg = "Invalid value '1' for dtype 'str"
    with pytest.raises(TypeError, match=msg):
        arr.fillna(value=1)


def test_arrow_array(dtype):
    # protocol added in 0.15.0
    pa = pytest.importorskip("pyarrow")
    import pyarrow.compute as pc

    data = pd.array(["a", "b", "c"], dtype=dtype)
    arr = pa.array(data)
    expected = pa.array(list(data), type=pa.large_string(), from_pandas=True)
    if dtype.storage == "pyarrow" and pa_version_under12p0:
        expected = pa.chunked_array(expected)
    if dtype.storage == "python":
        expected = pc.cast(expected, pa.string())
    assert arr.equals(expected)


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_arrow_roundtrip(dtype, string_storage, using_infer_string):
    # roundtrip possible from arrow 1.0.0
    pa = pytest.importorskip("pyarrow")

    data = pd.array(["a", "b", None], dtype=dtype)
    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    if dtype.storage == "python":
        assert table.field("a").type == "string"
    else:
        assert table.field("a").type == "large_string"
    with pd.option_context("string_storage", string_storage):
        result = table.to_pandas()
    if dtype.na_value is np.nan and not using_infer_string:
        assert result["a"].dtype == "object"
    else:
        assert isinstance(result["a"].dtype, pd.StringDtype)
        expected = df.astype(pd.StringDtype(string_storage, na_value=dtype.na_value))
        if using_infer_string:
            expected.columns = expected.columns.astype(
                pd.StringDtype(string_storage, na_value=np.nan)
            )
        tm.assert_frame_equal(result, expected)
        # ensure the missing value is represented by NA and not np.nan or None
        assert result.loc[2, "a"] is result["a"].dtype.na_value


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_arrow_from_string(using_infer_string):
    # not roundtrip,  but starting with pyarrow table without pandas metadata
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"a": pa.array(["a", "b", None], type=pa.string())})

    result = table.to_pandas()

    if using_infer_string and not pa_version_under19p0:
        expected = pd.DataFrame({"a": ["a", "b", None]}, dtype="str")
    else:
        expected = pd.DataFrame({"a": ["a", "b", None]}, dtype="object")
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings("ignore:Passing a BlockManager:DeprecationWarning")
def test_arrow_load_from_zero_chunks(dtype, string_storage, using_infer_string):
    # GH-41040
    pa = pytest.importorskip("pyarrow")

    data = pd.array([], dtype=dtype)
    df = pd.DataFrame({"a": data})
    table = pa.table(df)
    if dtype.storage == "python":
        assert table.field("a").type == "string"
    else:
        assert table.field("a").type == "large_string"
    # Instantiate the same table with no chunks at all
    table = pa.table([pa.chunked_array([], type=pa.string())], schema=table.schema)
    with pd.option_context("string_storage", string_storage):
        result = table.to_pandas()

    if dtype.na_value is np.nan and not using_string_dtype():
        assert result["a"].dtype == "object"
    else:
        assert isinstance(result["a"].dtype, pd.StringDtype)
        expected = df.astype(pd.StringDtype(string_storage, na_value=dtype.na_value))
        if using_infer_string:
            expected.columns = expected.columns.astype(
                pd.StringDtype(string_storage, na_value=np.nan)
            )
        tm.assert_frame_equal(result, expected)


def test_value_counts_na(dtype):
    if dtype.na_value is np.nan:
        exp_dtype = "int64"
    elif dtype.storage == "pyarrow":
        exp_dtype = "int64[pyarrow]"
    else:
        exp_dtype = "Int64"
    arr = pd.array(["a", "b", "a", pd.NA], dtype=dtype)
    result = arr.value_counts(dropna=False)
    expected = pd.Series([2, 1, 1], index=arr[[0, 1, 3]], dtype=exp_dtype, name="count")
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([2, 1], index=arr[:2], dtype=exp_dtype, name="count")
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize(dtype):
    if dtype.na_value is np.nan:
        exp_dtype = np.float64
    elif dtype.storage == "pyarrow":
        exp_dtype = "double[pyarrow]"
    else:
        exp_dtype = "Float64"
    ser = pd.Series(["a", "b", "a", pd.NA], dtype=dtype)
    result = ser.value_counts(normalize=True)
    expected = pd.Series([2, 1], index=ser[:2], dtype=exp_dtype, name="proportion") / 3
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "values, expected",
    [
        (["a", "b", "c"], np.array([False, False, False])),
        (["a", "b", None], np.array([False, False, True])),
    ],
)
def test_use_inf_as_na(values, expected, dtype):
    # https://github.com/pandas-dev/pandas/issues/33655
    values = pd.array(values, dtype=dtype)
    msg = "use_inf_as_na option is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context("mode.use_inf_as_na", True):
            result = values.isna()
            tm.assert_numpy_array_equal(result, expected)

            result = pd.Series(values).isna()
            expected = pd.Series(expected)
            tm.assert_series_equal(result, expected)

            result = pd.DataFrame(values).isna()
            expected = pd.DataFrame(expected)
            tm.assert_frame_equal(result, expected)


def test_value_counts_sort_false(dtype):
    if dtype.na_value is np.nan:
        exp_dtype = "int64"
    elif dtype.storage == "pyarrow":
        exp_dtype = "int64[pyarrow]"
    else:
        exp_dtype = "Int64"
    ser = pd.Series(["a", "b", "c", "b"], dtype=dtype)
    result = ser.value_counts(sort=False)
    expected = pd.Series([1, 2, 1], index=ser[:3], dtype=exp_dtype, name="count")
    tm.assert_series_equal(result, expected)


def test_memory_usage(dtype):
    # GH 33963

    if dtype.storage == "pyarrow":
        pytest.skip(f"not applicable for {dtype.storage}")

    series = pd.Series(["a", "b", "c"], dtype=dtype)

    assert 0 < series.nbytes <= series.memory_usage() < series.memory_usage(deep=True)


@pytest.mark.parametrize("float_dtype", [np.float16, np.float32, np.float64])
def test_astype_from_float_dtype(float_dtype, dtype):
    # https://github.com/pandas-dev/pandas/issues/36451
    ser = pd.Series([0.1], dtype=float_dtype)
    result = ser.astype(dtype)
    expected = pd.Series(["0.1"], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_to_numpy_returns_pdna_default(dtype):
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    result = np.array(arr)
    expected = np.array(["a", dtype.na_value, "b"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_na_value(dtype, nulls_fixture):
    na_value = nulls_fixture
    arr = pd.array(["a", pd.NA, "b"], dtype=dtype)
    result = arr.to_numpy(na_value=na_value)
    expected = np.array(["a", na_value, "b"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_isin(dtype, fixed_now_ts):
    s = pd.Series(["a", "b", None], dtype=dtype)

    result = s.isin(["a", "c"])
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(["a", pd.NA])
    expected = pd.Series([True, False, True])
    tm.assert_series_equal(result, expected)

    result = s.isin([])
    expected = pd.Series([False, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(["a", fixed_now_ts])
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin([fixed_now_ts])
    expected = pd.Series([False, False, False])
    tm.assert_series_equal(result, expected)


def test_isin_string_array(dtype, dtype2):
    s = pd.Series(["a", "b", None], dtype=dtype)

    result = s.isin(pd.array(["a", "c"], dtype=dtype2))
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(pd.array(["a", None], dtype=dtype2))
    expected = pd.Series([True, False, True])
    tm.assert_series_equal(result, expected)


def test_isin_arrow_string_array(dtype):
    pa = pytest.importorskip("pyarrow")
    s = pd.Series(["a", "b", None], dtype=dtype)

    result = s.isin(pd.array(["a", "c"], dtype=pd.ArrowDtype(pa.string())))
    expected = pd.Series([True, False, False])
    tm.assert_series_equal(result, expected)

    result = s.isin(pd.array(["a", None], dtype=pd.ArrowDtype(pa.string())))
    expected = pd.Series([True, False, True])
    tm.assert_series_equal(result, expected)


def test_setitem_scalar_with_mask_validation(dtype):
    # https://github.com/pandas-dev/pandas/issues/47628
    # setting None with a boolean mask (through _putmaks) should still result
    # in pd.NA values in the underlying array
    ser = pd.Series(["a", "b", "c"], dtype=dtype)
    mask = np.array([False, True, False])

    ser[mask] = None
    assert ser.array[1] is ser.dtype.na_value

    # for other non-string we should also raise an error
    ser = pd.Series(["a", "b", "c"], dtype=dtype)
    msg = "Invalid value '1' for dtype 'str"
    with pytest.raises(TypeError, match=msg):
        ser[mask] = 1


def test_from_numpy_str(dtype):
    vals = ["a", "b", "c"]
    arr = np.array(vals, dtype=np.str_)
    result = pd.array(arr, dtype=dtype)
    expected = pd.array(vals, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_tolist(dtype):
    vals = ["a", "b", "c"]
    arr = pd.array(vals, dtype=dtype)
    result = arr.tolist()
    expected = vals
    tm.assert_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\test_boxplot_method.py ===
""" Test cases for .boxplot method """

from __future__ import annotations

import itertools
import string

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
    plotting,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_axes_shape,
    _check_box_return_type,
    _check_plot_works,
    _check_ticks_props,
    _check_visible,
)
from pandas.util.version import Version

from pandas.io.formats.printing import pprint_thing

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")


def _check_ax_limits(col, ax):
    y_min, y_max = ax.get_ylim()
    assert y_min <= col.min()
    assert y_max >= col.max()


if Version(mpl.__version__) < Version("3.10"):
    verts: list[dict[str, bool | str]] = [{"vert": False}, {"vert": True}]
else:
    verts = [{"orientation": "horizontal"}, {"orientation": "vertical"}]


@pytest.fixture(params=verts)
def vert(request):
    return request.param


class TestDataFramePlots:
    def test_stacked_boxplot_set_axis(self):
        # GH2980
        import matplotlib.pyplot as plt

        n = 80
        df = DataFrame(
            {
                "Clinical": np.random.default_rng(2).choice([0, 1, 2, 3], n),
                "Confirmed": np.random.default_rng(2).choice([0, 1, 2, 3], n),
                "Discarded": np.random.default_rng(2).choice([0, 1, 2, 3], n),
            },
            index=np.arange(0, n),
        )
        ax = df.plot(kind="bar", stacked=True)
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == df.index.to_list()
        ax.set_xticks(np.arange(0, 80, 10))
        plt.draw()  # Update changes
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == list(
            np.arange(0, 80, 10)
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs, warn",
        [
            [{"return_type": "dict"}, None],
            [{"column": ["one", "two"]}, None],
            [{"column": ["one", "two"], "by": "indic"}, UserWarning],
            [{"column": ["one"], "by": ["indic", "indic2"]}, None],
            [{"by": "indic"}, UserWarning],
            [{"by": ["indic", "indic2"]}, UserWarning],
            [{"notch": 1}, None],
            [{"by": "indic", "notch": 1}, UserWarning],
        ],
    )
    def test_boxplot_legacy1(self, kwargs, warn):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        df["indic"] = ["foo", "bar"] * 3
        df["indic2"] = ["foo", "bar", "foo"] * 2

        # _check_plot_works can add an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(warn, check_stacklevel=False):
            _check_plot_works(df.boxplot, **kwargs)

    def test_boxplot_legacy1_series(self):
        ser = Series(np.random.default_rng(2).standard_normal(6))
        _check_plot_works(plotting._core.boxplot, data=ser, return_type="dict")

    def test_boxplot_legacy2(self):
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.boxplot, by="X")

    def test_boxplot_legacy2_with_ax(self):
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        # When ax is supplied and required number of axes is 1,
        # passed ax should be used:
        _, ax = mpl.pyplot.subplots()
        axes = df.boxplot("Col1", by="X", ax=ax)
        ax_axes = ax.axes
        assert ax_axes is axes

    def test_boxplot_legacy2_with_ax_return_type(self):
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        fig, ax = mpl.pyplot.subplots()
        axes = df.groupby("Y").boxplot(ax=ax, return_type="axes")
        ax_axes = ax.axes
        assert ax_axes is axes["A"]

    def test_boxplot_legacy2_with_multi_col(self):
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        # Multiple columns with an ax argument should use same figure
        fig, ax = mpl.pyplot.subplots()
        with tm.assert_produces_warning(UserWarning):
            axes = df.boxplot(
                column=["Col1", "Col2"], by="X", ax=ax, return_type="axes"
            )
        assert axes["Col1"].get_figure() is fig

    def test_boxplot_legacy2_by_none(self):
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        df["Y"] = Series(["A"] * 10)
        # When by is None, check that all relevant lines are present in the
        # dict
        _, ax = mpl.pyplot.subplots()
        d = df.boxplot(ax=ax, return_type="dict")
        lines = list(itertools.chain.from_iterable(d.values()))
        assert len(ax.get_lines()) == len(lines)

    def test_boxplot_return_type_none(self, hist_df):
        # GH 12216; return_type=None & by=None -> axes
        result = hist_df.boxplot()
        assert isinstance(result, mpl.pyplot.Axes)

    def test_boxplot_return_type_legacy(self):
        # API change in https://github.com/pandas-dev/pandas/pull/7096

        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        msg = "return_type must be {'axes', 'dict', 'both'}"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(return_type="NOT_A_TYPE")

        result = df.boxplot()
        _check_box_return_type(result, "axes")

    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    def test_boxplot_return_type_legacy_return_type(self, return_type):
        # API change in https://github.com/pandas-dev/pandas/pull/7096

        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        with tm.assert_produces_warning(False):
            result = df.boxplot(return_type=return_type)
        _check_box_return_type(result, return_type)

    def test_boxplot_axis_limits(self, hist_df):
        df = hist_df.copy()
        df["age"] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        # One full row
        height_ax, weight_ax = df.boxplot(["height", "weight"], by="category")
        _check_ax_limits(df["height"], height_ax)
        _check_ax_limits(df["weight"], weight_ax)
        assert weight_ax._sharey == height_ax

    def test_boxplot_axis_limits_two_rows(self, hist_df):
        df = hist_df.copy()
        df["age"] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        # Two rows, one partial
        p = df.boxplot(["height", "weight", "age"], by="category")
        height_ax, weight_ax, age_ax = p[0, 0], p[0, 1], p[1, 0]
        dummy_ax = p[1, 1]

        _check_ax_limits(df["height"], height_ax)
        _check_ax_limits(df["weight"], weight_ax)
        _check_ax_limits(df["age"], age_ax)
        assert weight_ax._sharey == height_ax
        assert age_ax._sharey == height_ax
        assert dummy_ax._sharey is None

    def test_boxplot_empty_column(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((20, 4)))
        df.loc[:, 0] = np.nan
        _check_plot_works(df.boxplot, return_type="axes")

    def test_figsize(self):
        df = DataFrame(
            np.random.default_rng(2).random((10, 5)), columns=["A", "B", "C", "D", "E"]
        )
        result = df.boxplot(return_type="axes", figsize=(12, 8))
        assert result.figure.bbox_inches.width == 12
        assert result.figure.bbox_inches.height == 8

    def test_fontsize(self):
        df = DataFrame({"a": [1, 2, 3, 4, 5, 6]})
        _check_ticks_props(df.boxplot("a", fontsize=16), xlabelsize=16, ylabelsize=16)

    def test_boxplot_numeric_data(self):
        # GH 22799
        df = DataFrame(
            {
                "a": date_range("2012-01-01", periods=100),
                "b": np.random.default_rng(2).standard_normal(100),
                "c": np.random.default_rng(2).standard_normal(100) + 2,
                "d": date_range("2012-01-01", periods=100).astype(str),
                "e": date_range("2012-01-01", periods=100, tz="UTC"),
                "f": timedelta_range("1 days", periods=100),
            }
        )
        ax = df.plot(kind="box")
        assert [x.get_text() for x in ax.get_xticklabels()] == ["b", "c"]

    @pytest.mark.parametrize(
        "colors_kwd, expected",
        [
            (
                {"boxes": "r", "whiskers": "b", "medians": "g", "caps": "c"},
                {"boxes": "r", "whiskers": "b", "medians": "g", "caps": "c"},
            ),
            ({"boxes": "r"}, {"boxes": "r"}),
            ("r", {"boxes": "r", "whiskers": "r", "medians": "r", "caps": "r"}),
        ],
    )
    def test_color_kwd(self, colors_kwd, expected):
        # GH: 26214
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        result = df.boxplot(color=colors_kwd, return_type="dict")
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize(
        "scheme,expected",
        [
            (
                "dark_background",
                {
                    "boxes": "#8dd3c7",
                    "whiskers": "#8dd3c7",
                    "medians": "#bfbbd9",
                    "caps": "#8dd3c7",
                },
            ),
            (
                "default",
                {
                    "boxes": "#1f77b4",
                    "whiskers": "#1f77b4",
                    "medians": "#2ca02c",
                    "caps": "#1f77b4",
                },
            ),
        ],
    )
    def test_colors_in_theme(self, scheme, expected):
        # GH: 40769
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        import matplotlib.pyplot as plt

        plt.style.use(scheme)
        result = df.plot.box(return_type="dict")
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize(
        "dict_colors, msg",
        [({"boxes": "r", "invalid_key": "r"}, "invalid key 'invalid_key'")],
    )
    def test_color_kwd_errors(self, dict_colors, msg):
        # GH: 26214
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        with pytest.raises(ValueError, match=msg):
            df.boxplot(color=dict_colors, return_type="dict")

    @pytest.mark.parametrize(
        "props, expected",
        [
            ("boxprops", "boxes"),
            ("whiskerprops", "whiskers"),
            ("capprops", "caps"),
            ("medianprops", "medians"),
        ],
    )
    def test_specified_props_kwd(self, props, expected):
        # GH 30346
        df = DataFrame({k: np.random.default_rng(2).random(10) for k in "ABC"})
        kwd = {props: {"color": "C1"}}
        result = df.boxplot(return_type="dict", **kwd)

        assert result[expected][0].get_color() == "C1"

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_plot_xlabel_ylabel(self, vert):
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        xlabel, ylabel = "x", "y"
        ax = df.plot(kind="box", xlabel=xlabel, ylabel=ylabel, **vert)
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_plot_box(self, vert):
        # GH 54941
        rng = np.random.default_rng(2)
        df1 = DataFrame(rng.integers(0, 100, size=(100, 4)), columns=list("ABCD"))
        df2 = DataFrame(rng.integers(0, 100, size=(100, 4)), columns=list("ABCD"))

        xlabel, ylabel = "x", "y"
        _, axs = plt.subplots(ncols=2, figsize=(10, 7), sharey=True)
        df1.plot.box(ax=axs[0], xlabel=xlabel, ylabel=ylabel, **vert)
        df2.plot.box(ax=axs[1], xlabel=xlabel, ylabel=ylabel, **vert)
        for ax in axs:
            assert ax.get_xlabel() == xlabel
            assert ax.get_ylabel() == ylabel
        mpl.pyplot.close()

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_boxplot_xlabel_ylabel(self, vert):
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        xlabel, ylabel = "x", "y"
        ax = df.boxplot(xlabel=xlabel, ylabel=ylabel, **vert)
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_boxplot_group_xlabel_ylabel(self, vert):
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        xlabel, ylabel = "x", "y"
        ax = df.boxplot(by="group", xlabel=xlabel, ylabel=ylabel, **vert)
        for subplot in ax:
            assert subplot.get_xlabel() == xlabel
            assert subplot.get_ylabel() == ylabel
        mpl.pyplot.close()

    @pytest.mark.filterwarnings("ignore:set_ticklabels:UserWarning")
    def test_boxplot_group_no_xlabel_ylabel(self, vert, request):
        if Version(mpl.__version__) >= Version("3.10") and vert == {
            "orientation": "horizontal"
        }:
            request.applymarker(
                pytest.mark.xfail(reason=f"{vert} fails starting with matplotlib 3.10")
            )
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        ax = df.boxplot(by="group", **vert)
        for subplot in ax:
            target_label = (
                subplot.get_xlabel()
                if vert == {"vert": True}  # noqa: PLR1714
                or vert == {"orientation": "vertical"}
                else subplot.get_ylabel()
            )
            assert target_label == pprint_thing(["group"])
        mpl.pyplot.close()


class TestDataFrameGroupByPlots:
    def test_boxplot_legacy1(self, hist_df):
        grouped = hist_df.groupby(by="gender")
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(grouped.boxplot, return_type="axes")
        _check_axes_shape(list(axes.values), axes_num=2, layout=(1, 2))

    def test_boxplot_legacy1_return_type(self, hist_df):
        grouped = hist_df.groupby(by="gender")
        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type="axes")
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.slow
    def test_boxplot_legacy2(self):
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=MultiIndex.from_tuples(tuples),
        )
        grouped = df.groupby(level=1)
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(grouped.boxplot, return_type="axes")
        _check_axes_shape(list(axes.values), axes_num=10, layout=(4, 3))

    @pytest.mark.slow
    def test_boxplot_legacy2_return_type(self):
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=MultiIndex.from_tuples(tuples),
        )
        grouped = df.groupby(level=1)
        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type="axes")
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.parametrize(
        "subplots, warn, axes_num, layout",
        [[True, UserWarning, 3, (2, 2)], [False, None, 1, (1, 1)]],
    )
    def test_boxplot_legacy3(self, subplots, warn, axes_num, layout):
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=MultiIndex.from_tuples(tuples),
        )
        msg = "DataFrame.groupby with axis=1 is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            grouped = df.unstack(level=1).groupby(level=0, axis=1)
        with tm.assert_produces_warning(warn, check_stacklevel=False):
            axes = _check_plot_works(
                grouped.boxplot, subplots=subplots, return_type="axes"
            )
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    def test_grouped_plot_fignums(self):
        n = 10
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)
        df = DataFrame({"height": height, "weight": weight, "gender": gender})
        gb = df.groupby("gender")

        res = gb.plot()
        assert len(mpl.pyplot.get_fignums()) == 2
        assert len(res) == 2
        plt.close("all")

        res = gb.boxplot(return_type="axes")
        assert len(mpl.pyplot.get_fignums()) == 1
        assert len(res) == 2

    def test_grouped_plot_fignums_excluded_col(self):
        n = 10
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)
        df = DataFrame({"height": height, "weight": weight, "gender": gender})
        # now works with GH 5610 as gender is excluded
        df.groupby("gender").hist()

    @pytest.mark.slow
    def test_grouped_box_return_type(self, hist_df):
        df = hist_df

        # old style: return_type=None
        result = df.boxplot(by="gender")
        assert isinstance(result, np.ndarray)
        _check_box_return_type(
            result, None, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.slow
    def test_grouped_box_return_type_groupby(self, hist_df):
        df = hist_df
        # now for groupby
        result = df.groupby("gender").boxplot(return_type="dict")
        _check_box_return_type(result, "dict", expected_keys=["Male", "Female"])

    @pytest.mark.slow
    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    def test_grouped_box_return_type_arg(self, hist_df, return_type):
        df = hist_df

        returned = df.groupby("classroom").boxplot(return_type=return_type)
        _check_box_return_type(returned, return_type, expected_keys=["A", "B", "C"])

        returned = df.boxplot(by="classroom", return_type=return_type)
        _check_box_return_type(
            returned, return_type, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    def test_grouped_box_return_type_arg_duplcate_cats(self, return_type):
        columns2 = "X B C D A".split()
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((6, 5)), columns=columns2
        )
        categories2 = "A B".split()
        df2["category"] = categories2 * 3

        returned = df2.groupby("category").boxplot(return_type=return_type)
        _check_box_return_type(returned, return_type, expected_keys=categories2)

        returned = df2.boxplot(by="category", return_type=return_type)
        _check_box_return_type(returned, return_type, expected_keys=columns2)

    @pytest.mark.slow
    def test_grouped_box_layout_too_small(self, hist_df):
        df = hist_df

        msg = "Layout of 1x1 must be larger than required size 2"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=["weight", "height"], by=df.gender, layout=(1, 1))

    @pytest.mark.slow
    def test_grouped_box_layout_needs_by(self, hist_df):
        df = hist_df
        msg = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(
                column=["height", "weight", "category"],
                layout=(2, 1),
                return_type="dict",
            )

    @pytest.mark.slow
    def test_grouped_box_layout_positive_layout(self, hist_df):
        df = hist_df
        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=["weight", "height"], by=df.gender, layout=(-1, -1))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "gb_key, axes_num, rows",
        [["gender", 2, 1], ["category", 4, 2], ["classroom", 3, 2]],
    )
    def test_grouped_box_layout_positive_layout_axes(
        self, hist_df, gb_key, axes_num, rows
    ):
        df = hist_df
        # _check_plot_works adds an ax so catch warning. see GH #13188 GH 6769
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(
                df.groupby(gb_key).boxplot, column="height", return_type="dict"
            )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=axes_num, layout=(rows, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "col, visible", [["height", False], ["weight", True], ["category", True]]
    )
    def test_grouped_box_layout_visible(self, hist_df, col, visible):
        df = hist_df
        # GH 5897
        axes = df.boxplot(
            column=["height", "weight", "category"], by="gender", return_type="axes"
        )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2))
        ax = axes[col]
        _check_visible(ax.get_xticklabels(), visible=visible)
        _check_visible([ax.xaxis.get_label()], visible=visible)

    @pytest.mark.slow
    def test_grouped_box_layout_shape(self, hist_df):
        df = hist_df
        df.groupby("classroom").boxplot(
            column=["height", "weight", "category"], return_type="dict"
        )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize("cols", [2, -1])
    def test_grouped_box_layout_works(self, hist_df, cols):
        df = hist_df
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(
                df.groupby("category").boxplot,
                column="height",
                layout=(3, cols),
                return_type="dict",
            )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=4, layout=(3, 2))

    @pytest.mark.slow
    @pytest.mark.parametrize("rows, res", [[4, 4], [-1, 3]])
    def test_grouped_box_layout_axes_shape_rows(self, hist_df, rows, res):
        df = hist_df
        df.boxplot(
            column=["height", "weight", "category"], by="gender", layout=(rows, 1)
        )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(res, 1))

    @pytest.mark.slow
    @pytest.mark.parametrize("cols, res", [[4, 4], [-1, 3]])
    def test_grouped_box_layout_axes_shape_cols_groupby(self, hist_df, cols, res):
        df = hist_df
        df.groupby("classroom").boxplot(
            column=["height", "weight", "category"],
            layout=(1, cols),
            return_type="dict",
        )
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(1, res))

    @pytest.mark.slow
    def test_grouped_box_multiple_axes(self, hist_df):
        # GH 6970, GH 7069
        df = hist_df

        # check warning to ignore sharex / sharey
        # this check should be done in the first function which
        # passes multiple axes to plot, hist or boxplot
        # location should be changed if other test is added
        # which has earlier alphabetical order
        with tm.assert_produces_warning(UserWarning):
            _, axes = mpl.pyplot.subplots(2, 2)
            df.groupby("category").boxplot(column="height", return_type="axes", ax=axes)
            _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=4, layout=(2, 2))

    @pytest.mark.slow
    def test_grouped_box_multiple_axes_on_fig(self, hist_df):
        # GH 6970, GH 7069
        df = hist_df
        fig, axes = mpl.pyplot.subplots(2, 3)
        with tm.assert_produces_warning(UserWarning):
            returned = df.boxplot(
                column=["height", "weight", "category"],
                by="gender",
                return_type="axes",
                ax=axes[0],
            )
        returned = np.array(list(returned.values))
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[0])
        assert returned[0].figure is fig

        # draw on second row
        with tm.assert_produces_warning(UserWarning):
            returned = df.groupby("classroom").boxplot(
                column=["height", "weight", "category"], return_type="axes", ax=axes[1]
            )
        returned = np.array(list(returned.values))
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        tm.assert_numpy_array_equal(returned, axes[1])
        assert returned[0].figure is fig

    @pytest.mark.slow
    def test_grouped_box_multiple_axes_ax_error(self, hist_df):
        # GH 6970, GH 7069
        df = hist_df
        msg = "The number of passed axes must be 3, the same as the output plot"
        with pytest.raises(ValueError, match=msg):
            fig, axes = mpl.pyplot.subplots(2, 3)
            # pass different number of axes from required
            with tm.assert_produces_warning(UserWarning):
                axes = df.groupby("classroom").boxplot(ax=axes)

    def test_fontsize(self):
        df = DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]})
        _check_ticks_props(
            df.boxplot("a", by="b", fontsize=16), xlabelsize=16, ylabelsize=16
        )

    @pytest.mark.parametrize(
        "col, expected_xticklabel",
        [
            ("v", ["(a, v)", "(b, v)", "(c, v)", "(d, v)", "(e, v)"]),
            (["v"], ["(a, v)", "(b, v)", "(c, v)", "(d, v)", "(e, v)"]),
            ("v1", ["(a, v1)", "(b, v1)", "(c, v1)", "(d, v1)", "(e, v1)"]),
            (
                ["v", "v1"],
                [
                    "(a, v)",
                    "(a, v1)",
                    "(b, v)",
                    "(b, v1)",
                    "(c, v)",
                    "(c, v1)",
                    "(d, v)",
                    "(d, v1)",
                    "(e, v)",
                    "(e, v1)",
                ],
            ),
            (
                None,
                [
                    "(a, v)",
                    "(a, v1)",
                    "(b, v)",
                    "(b, v1)",
                    "(c, v)",
                    "(c, v1)",
                    "(d, v)",
                    "(d, v1)",
                    "(e, v)",
                    "(e, v1)",
                ],
            ),
        ],
    )
    def test_groupby_boxplot_subplots_false(self, col, expected_xticklabel):
        # GH 16748
        df = DataFrame(
            {
                "cat": np.random.default_rng(2).choice(list("abcde"), 100),
                "v": np.random.default_rng(2).random(100),
                "v1": np.random.default_rng(2).random(100),
            }
        )
        grouped = df.groupby("cat")

        axes = _check_plot_works(
            grouped.boxplot, subplots=False, column=col, return_type="axes"
        )

        result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
        assert expected_xticklabel == result_xticklabel

    def test_groupby_boxplot_object(self, hist_df):
        # GH 43480
        df = hist_df.astype("object")
        grouped = df.groupby("gender")
        msg = "boxplot method requires numerical columns, nothing to plot"
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(grouped.boxplot, subplots=False)

    def test_boxplot_multiindex_column(self):
        # GH 16748
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = list(zip(*arrays))
        index = MultiIndex.from_tuples(tuples, names=["first", "second"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 8)),
            index=["A", "B", "C"],
            columns=index,
        )

        col = [("bar", "one"), ("bar", "two")]
        axes = _check_plot_works(df.boxplot, column=col, return_type="axes")

        expected_xticklabel = ["(bar, one)", "(bar, two)"]
        result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
        assert expected_xticklabel == result_xticklabel

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_reset_index.py ===
from datetime import datetime
from itertools import product

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
)

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    RangeIndex,
    Series,
    Timestamp,
    cut,
    date_range,
)
import pandas._testing as tm


@pytest.fixture()
def multiindex_df():
    levels = [["A", ""], ["B", "b"]]
    return DataFrame([[0, 2], [1, 3]], columns=MultiIndex.from_tuples(levels))


class TestResetIndex:
    def test_reset_index_empty_rangeindex(self):
        # GH#45230
        df = DataFrame(
            columns=["brand"], dtype=np.int64, index=RangeIndex(0, 0, 1, name="foo")
        )

        df2 = df.set_index([df.index, "brand"])

        result = df2.reset_index([1], drop=True)
        tm.assert_frame_equal(result, df[[]], check_index_type=True)

    def test_set_reset(self):
        idx = Index([2**63, 2**63 + 5, 2**63 + 10], name="foo")

        # set/reset
        df = DataFrame({"A": [0, 1, 2]}, index=idx)
        result = df.reset_index()
        assert result["foo"].dtype == np.dtype("uint64")

        df = result.set_index("foo")
        tm.assert_index_equal(df.index, idx)

    def test_set_index_reset_index_dt64tz(self):
        idx = Index(date_range("20130101", periods=3, tz="US/Eastern"), name="foo")

        # set/reset
        df = DataFrame({"A": [0, 1, 2]}, index=idx)
        result = df.reset_index()
        assert result["foo"].dtype == "datetime64[ns, US/Eastern]"

        df = result.set_index("foo")
        tm.assert_index_equal(df.index, idx)

    def test_reset_index_tz(self, tz_aware_fixture):
        # GH 3950
        # reset_index with single level
        tz = tz_aware_fixture
        idx = date_range("1/1/2011", periods=5, freq="D", tz=tz, name="idx")
        df = DataFrame({"a": range(5), "b": ["A", "B", "C", "D", "E"]}, index=idx)

        expected = DataFrame(
            {
                "idx": idx,
                "a": range(5),
                "b": ["A", "B", "C", "D", "E"],
            },
            columns=["idx", "a", "b"],
        )
        result = df.reset_index()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_frame_reset_index_tzaware_index(self, tz):
        dr = date_range("2012-06-02", periods=10, tz=tz)
        df = DataFrame(np.random.default_rng(2).standard_normal(len(dr)), dr)
        roundtripped = df.reset_index().set_index("index")
        xp = df.index.tz
        rs = roundtripped.index.tz
        assert xp == rs

    def test_reset_index_with_intervals(self):
        idx = IntervalIndex.from_breaks(np.arange(11), name="x")
        original = DataFrame({"x": idx, "y": np.arange(10)})[["x", "y"]]

        result = original.set_index("x")
        expected = DataFrame({"y": np.arange(10)}, index=idx)
        tm.assert_frame_equal(result, expected)

        result2 = result.reset_index()
        tm.assert_frame_equal(result2, original)

    def test_reset_index(self, float_frame):
        stacked = float_frame.stack(future_stack=True)[::2]
        stacked = DataFrame({"foo": stacked, "bar": stacked})

        names = ["first", "second"]
        stacked.index.names = names
        deleveled = stacked.reset_index()
        for i, (lev, level_codes) in enumerate(
            zip(stacked.index.levels, stacked.index.codes)
        ):
            values = lev.take(level_codes)
            name = names[i]
            tm.assert_index_equal(values, Index(deleveled[name]))

        stacked.index.names = [None, None]
        deleveled2 = stacked.reset_index()
        tm.assert_series_equal(
            deleveled["first"], deleveled2["level_0"], check_names=False
        )
        tm.assert_series_equal(
            deleveled["second"], deleveled2["level_1"], check_names=False
        )

        # default name assigned
        rdf = float_frame.reset_index()
        exp = Series(float_frame.index.values, name="index")
        tm.assert_series_equal(rdf["index"], exp)

        # default name assigned, corner case
        df = float_frame.copy()
        df["index"] = "foo"
        rdf = df.reset_index()
        exp = Series(float_frame.index.values, name="level_0")
        tm.assert_series_equal(rdf["level_0"], exp)

        # but this is ok
        float_frame.index.name = "index"
        deleveled = float_frame.reset_index()
        tm.assert_series_equal(deleveled["index"], Series(float_frame.index))
        tm.assert_index_equal(deleveled.index, Index(range(len(deleveled))), exact=True)

        # preserve column names
        float_frame.columns.name = "columns"
        reset = float_frame.reset_index()
        assert reset.columns.name == "columns"

        # only remove certain columns
        df = float_frame.reset_index().set_index(["index", "A", "B"])
        rs = df.reset_index(["A", "B"])

        tm.assert_frame_equal(rs, float_frame)

        rs = df.reset_index(["index", "A", "B"])
        tm.assert_frame_equal(rs, float_frame.reset_index())

        rs = df.reset_index(["index", "A", "B"])
        tm.assert_frame_equal(rs, float_frame.reset_index())

        rs = df.reset_index("A")
        xp = float_frame.reset_index().set_index(["index", "B"])
        tm.assert_frame_equal(rs, xp)

        # test resetting in place
        df = float_frame.copy()
        reset = float_frame.reset_index()
        return_value = df.reset_index(inplace=True)
        assert return_value is None
        tm.assert_frame_equal(df, reset)

        df = float_frame.reset_index().set_index(["index", "A", "B"])
        rs = df.reset_index("A", drop=True)
        xp = float_frame.copy()
        del xp["A"]
        xp = xp.set_index(["B"], append=True)
        tm.assert_frame_equal(rs, xp)

    def test_reset_index_name(self):
        df = DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            columns=["A", "B", "C", "D"],
            index=Index(range(2), name="x"),
        )
        assert df.reset_index().index.name is None
        assert df.reset_index(drop=True).index.name is None
        return_value = df.reset_index(inplace=True)
        assert return_value is None
        assert df.index.name is None

    @pytest.mark.parametrize("levels", [["A", "B"], [0, 1]])
    def test_reset_index_level(self, levels):
        df = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["A", "B", "C", "D"])

        # With MultiIndex
        result = df.set_index(["A", "B"]).reset_index(level=levels[0])
        tm.assert_frame_equal(result, df.set_index("B"))

        result = df.set_index(["A", "B"]).reset_index(level=levels[:1])
        tm.assert_frame_equal(result, df.set_index("B"))

        result = df.set_index(["A", "B"]).reset_index(level=levels)
        tm.assert_frame_equal(result, df)

        result = df.set_index(["A", "B"]).reset_index(level=levels, drop=True)
        tm.assert_frame_equal(result, df[["C", "D"]])

        # With single-level Index (GH 16263)
        result = df.set_index("A").reset_index(level=levels[0])
        tm.assert_frame_equal(result, df)

        result = df.set_index("A").reset_index(level=levels[:1])
        tm.assert_frame_equal(result, df)

        result = df.set_index(["A"]).reset_index(level=levels[0], drop=True)
        tm.assert_frame_equal(result, df[["B", "C", "D"]])

    @pytest.mark.parametrize("idx_lev", [["A", "B"], ["A"]])
    def test_reset_index_level_missing(self, idx_lev):
        # Missing levels - for both MultiIndex and single-level Index:
        df = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["A", "B", "C", "D"])

        with pytest.raises(KeyError, match=r"(L|l)evel \(?E\)?"):
            df.set_index(idx_lev).reset_index(level=["A", "E"])
        with pytest.raises(IndexError, match="Too many levels"):
            df.set_index(idx_lev).reset_index(level=[0, 1, 2])

    def test_reset_index_right_dtype(self):
        time = np.arange(0.0, 10, np.sqrt(2) / 2)
        s1 = Series(
            (9.81 * time**2) / 2, index=Index(time, name="time"), name="speed"
        )
        df = DataFrame(s1)

        reset = s1.reset_index()
        assert reset["time"].dtype == np.float64

        reset = df.reset_index()
        assert reset["time"].dtype == np.float64

    def test_reset_index_multiindex_col(self):
        vals = np.random.default_rng(2).standard_normal((3, 3)).astype(object)
        idx = ["x", "y", "z"]
        full = np.hstack(([[x] for x in idx], vals))
        df = DataFrame(
            vals,
            Index(idx, name="a"),
            columns=[["b", "b", "c"], ["mean", "median", "mean"]],
        )
        rs = df.reset_index()
        xp = DataFrame(
            full, columns=[["a", "b", "b", "c"], ["", "mean", "median", "mean"]]
        )
        tm.assert_frame_equal(rs, xp)

        rs = df.reset_index(col_fill=None)
        xp = DataFrame(
            full, columns=[["a", "b", "b", "c"], ["a", "mean", "median", "mean"]]
        )
        tm.assert_frame_equal(rs, xp)

        rs = df.reset_index(col_level=1, col_fill="blah")
        xp = DataFrame(
            full, columns=[["blah", "b", "b", "c"], ["a", "mean", "median", "mean"]]
        )
        tm.assert_frame_equal(rs, xp)

        df = DataFrame(
            vals,
            MultiIndex.from_arrays([[0, 1, 2], ["x", "y", "z"]], names=["d", "a"]),
            columns=[["b", "b", "c"], ["mean", "median", "mean"]],
        )
        rs = df.reset_index("a")
        xp = DataFrame(
            full,
            Index([0, 1, 2], name="d"),
            columns=[["a", "b", "b", "c"], ["", "mean", "median", "mean"]],
        )
        tm.assert_frame_equal(rs, xp)

        rs = df.reset_index("a", col_fill=None)
        xp = DataFrame(
            full,
            Index(range(3), name="d"),
            columns=[["a", "b", "b", "c"], ["a", "mean", "median", "mean"]],
        )
        tm.assert_frame_equal(rs, xp)

        rs = df.reset_index("a", col_fill="blah", col_level=1)
        xp = DataFrame(
            full,
            Index(range(3), name="d"),
            columns=[["blah", "b", "b", "c"], ["a", "mean", "median", "mean"]],
        )
        tm.assert_frame_equal(rs, xp)

    def test_reset_index_multiindex_nan(self):
        # GH#6322, testing reset_index on MultiIndexes
        # when we have a nan or all nan
        df = DataFrame(
            {
                "A": ["a", "b", "c"],
                "B": [0, 1, np.nan],
                "C": np.random.default_rng(2).random(3),
            }
        )
        rs = df.set_index(["A", "B"]).reset_index()
        tm.assert_frame_equal(rs, df)

        df = DataFrame(
            {
                "A": [np.nan, "b", "c"],
                "B": [0, 1, 2],
                "C": np.random.default_rng(2).random(3),
            }
        )
        rs = df.set_index(["A", "B"]).reset_index()
        tm.assert_frame_equal(rs, df)

        df = DataFrame({"A": ["a", "b", "c"], "B": [0, 1, 2], "C": [np.nan, 1.1, 2.2]})
        rs = df.set_index(["A", "B"]).reset_index()
        tm.assert_frame_equal(rs, df)

        df = DataFrame(
            {
                "A": ["a", "b", "c"],
                "B": [np.nan, np.nan, np.nan],
                "C": np.random.default_rng(2).random(3),
            }
        )
        rs = df.set_index(["A", "B"]).reset_index()
        tm.assert_frame_equal(rs, df)

    @pytest.mark.parametrize(
        "name",
        [
            None,
            "foo",
            2,
            3.0,
            pd.Timedelta(6),
            Timestamp("2012-12-30", tz="UTC"),
            "2012-12-31",
        ],
    )
    def test_reset_index_with_datetimeindex_cols(self, name):
        # GH#5818
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=date_range("1/1/2013", "1/2/2013"),
            index=["A", "B"],
        )
        df.index.name = name

        result = df.reset_index()

        item = name if name is not None else "index"
        columns = Index([item, datetime(2013, 1, 1), datetime(2013, 1, 2)])
        if isinstance(item, str) and item == "2012-12-31":
            columns = columns.astype("datetime64[ns]")
        else:
            assert columns.dtype == object

        expected = DataFrame(
            [["A", 1, 2], ["B", 3, 4]],
            columns=columns,
        )
        tm.assert_frame_equal(result, expected)

    def test_reset_index_range(self):
        # GH#12071
        df = DataFrame([[0, 0], [1, 1]], columns=["A", "B"], index=RangeIndex(stop=2))
        result = df.reset_index()
        assert isinstance(result.index, RangeIndex)
        expected = DataFrame(
            [[0, 0, 0], [1, 1, 1]],
            columns=["index", "A", "B"],
            index=RangeIndex(stop=2),
        )
        tm.assert_frame_equal(result, expected)

    def test_reset_index_multiindex_columns(self, multiindex_df):
        result = multiindex_df[["B"]].rename_axis("A").reset_index()
        tm.assert_frame_equal(result, multiindex_df)

        # GH#16120: already existing column
        msg = r"cannot insert \('A', ''\), already exists"
        with pytest.raises(ValueError, match=msg):
            multiindex_df.rename_axis("A").reset_index()

        # GH#16164: multiindex (tuple) full key
        result = multiindex_df.set_index([("A", "")]).reset_index()
        tm.assert_frame_equal(result, multiindex_df)

        # with additional (unnamed) index level
        idx_col = DataFrame(
            [[0], [1]], columns=MultiIndex.from_tuples([("level_0", "")])
        )
        expected = pd.concat([idx_col, multiindex_df[[("B", "b"), ("A", "")]]], axis=1)
        result = multiindex_df.set_index([("B", "b")], append=True).reset_index()
        tm.assert_frame_equal(result, expected)

        # with index name which is a too long tuple...
        msg = "Item must have length equal to number of levels."
        with pytest.raises(ValueError, match=msg):
            multiindex_df.rename_axis([("C", "c", "i")]).reset_index()

        # or too short...
        levels = [["A", "a", ""], ["B", "b", "i"]]
        df2 = DataFrame([[0, 2], [1, 3]], columns=MultiIndex.from_tuples(levels))
        idx_col = DataFrame(
            [[0], [1]], columns=MultiIndex.from_tuples([("C", "c", "ii")])
        )
        expected = pd.concat([idx_col, df2], axis=1)
        result = df2.rename_axis([("C", "c")]).reset_index(col_fill="ii")
        tm.assert_frame_equal(result, expected)

        # ... which is incompatible with col_fill=None
        with pytest.raises(
            ValueError,
            match=(
                "col_fill=None is incompatible with "
                r"incomplete column name \('C', 'c'\)"
            ),
        ):
            df2.rename_axis([("C", "c")]).reset_index(col_fill=None)

        # with col_level != 0
        result = df2.rename_axis([("c", "ii")]).reset_index(col_level=1, col_fill="C")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("flag", [False, True])
    @pytest.mark.parametrize("allow_duplicates", [False, True])
    def test_reset_index_duplicate_columns_allow(
        self, multiindex_df, flag, allow_duplicates
    ):
        # GH#44755 reset_index with duplicate column labels
        df = multiindex_df.rename_axis("A")
        df = df.set_flags(allows_duplicate_labels=flag)

        if flag and allow_duplicates:
            result = df.reset_index(allow_duplicates=allow_duplicates)
            levels = [["A", ""], ["A", ""], ["B", "b"]]
            expected = DataFrame(
                [[0, 0, 2], [1, 1, 3]], columns=MultiIndex.from_tuples(levels)
            )
            tm.assert_frame_equal(result, expected)
        else:
            if not flag and allow_duplicates:
                msg = (
                    "Cannot specify 'allow_duplicates=True' when "
                    "'self.flags.allows_duplicate_labels' is False"
                )
            else:
                msg = r"cannot insert \('A', ''\), already exists"
            with pytest.raises(ValueError, match=msg):
                df.reset_index(allow_duplicates=allow_duplicates)

    @pytest.mark.parametrize("flag", [False, True])
    def test_reset_index_duplicate_columns_default(self, multiindex_df, flag):
        df = multiindex_df.rename_axis("A")
        df = df.set_flags(allows_duplicate_labels=flag)

        msg = r"cannot insert \('A', ''\), already exists"
        with pytest.raises(ValueError, match=msg):
            df.reset_index()

    @pytest.mark.parametrize("allow_duplicates", ["bad value"])
    def test_reset_index_allow_duplicates_check(self, multiindex_df, allow_duplicates):
        with pytest.raises(ValueError, match="expected type bool"):
            multiindex_df.reset_index(allow_duplicates=allow_duplicates)

    def test_reset_index_datetime(self, tz_naive_fixture):
        # GH#3950
        tz = tz_naive_fixture
        idx1 = date_range("1/1/2011", periods=5, freq="D", tz=tz, name="idx1")
        idx2 = Index(range(5), name="idx2", dtype="int64")
        idx = MultiIndex.from_arrays([idx1, idx2])
        df = DataFrame(
            {"a": np.arange(5, dtype="int64"), "b": ["A", "B", "C", "D", "E"]},
            index=idx,
        )

        expected = DataFrame(
            {
                "idx1": idx1,
                "idx2": np.arange(5, dtype="int64"),
                "a": np.arange(5, dtype="int64"),
                "b": ["A", "B", "C", "D", "E"],
            },
            columns=["idx1", "idx2", "a", "b"],
        )

        tm.assert_frame_equal(df.reset_index(), expected)

    def test_reset_index_datetime2(self, tz_naive_fixture):
        tz = tz_naive_fixture
        idx1 = date_range("1/1/2011", periods=5, freq="D", tz=tz, name="idx1")
        idx2 = Index(range(5), name="idx2", dtype="int64")
        idx3 = date_range(
            "1/1/2012", periods=5, freq="MS", tz="Europe/Paris", name="idx3"
        )
        idx = MultiIndex.from_arrays([idx1, idx2, idx3])
        df = DataFrame(
            {"a": np.arange(5, dtype="int64"), "b": ["A", "B", "C", "D", "E"]},
            index=idx,
        )

        expected = DataFrame(
            {
                "idx1": idx1,
                "idx2": np.arange(5, dtype="int64"),
                "idx3": idx3,
                "a": np.arange(5, dtype="int64"),
                "b": ["A", "B", "C", "D", "E"],
            },
            columns=["idx1", "idx2", "idx3", "a", "b"],
        )
        result = df.reset_index()
        tm.assert_frame_equal(result, expected)

    def test_reset_index_datetime3(self, tz_naive_fixture):
        # GH#7793
        tz = tz_naive_fixture
        dti = date_range("20130101", periods=3, tz=tz)
        idx = MultiIndex.from_product([["a", "b"], dti])
        df = DataFrame(
            np.arange(6, dtype="int64").reshape(6, 1), columns=["a"], index=idx
        )

        expected = DataFrame(
            {
                "level_0": "a a a b b b".split(),
                "level_1": dti.append(dti),
                "a": np.arange(6, dtype="int64"),
            },
            columns=["level_0", "level_1", "a"],
        )
        result = df.reset_index()
        tm.assert_frame_equal(result, expected)

    def test_reset_index_period(self):
        # GH#7746
        idx = MultiIndex.from_product(
            [pd.period_range("20130101", periods=3, freq="M"), list("abc")],
            names=["month", "feature"],
        )

        df = DataFrame(
            np.arange(9, dtype="int64").reshape(-1, 1), index=idx, columns=["a"]
        )
        expected = DataFrame(
            {
                "month": (
                    [pd.Period("2013-01", freq="M")] * 3
                    + [pd.Period("2013-02", freq="M")] * 3
                    + [pd.Period("2013-03", freq="M")] * 3
                ),
                "feature": ["a", "b", "c"] * 3,
                "a": np.arange(9, dtype="int64"),
            },
            columns=["month", "feature", "a"],
        )
        result = df.reset_index()
        tm.assert_frame_equal(result, expected)

    def test_reset_index_delevel_infer_dtype(self):
        tuples = list(product(["foo", "bar"], [10, 20], [1.0, 1.1]))
        index = MultiIndex.from_tuples(tuples, names=["prm0", "prm1", "prm2"])
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 3)),
            columns=["A", "B", "C"],
            index=index,
        )
        deleveled = df.reset_index()
        assert is_integer_dtype(deleveled["prm1"])
        assert is_float_dtype(deleveled["prm2"])

    def test_reset_index_with_drop(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        ymd = multiindex_year_month_day_dataframe_random_data

        deleveled = ymd.reset_index(drop=True)
        assert len(deleveled.columns) == len(ymd.columns)
        assert deleveled.index.name == ymd.index.name

    @pytest.mark.parametrize(
        "ix_data, exp_data",
        [
            (
                [(pd.NaT, 1), (pd.NaT, 2)],
                {"a": [pd.NaT, pd.NaT], "b": [1, 2], "x": [11, 12]},
            ),
            (
                [(pd.NaT, 1), (Timestamp("2020-01-01"), 2)],
                {"a": [pd.NaT, Timestamp("2020-01-01")], "b": [1, 2], "x": [11, 12]},
            ),
            (
                [(pd.NaT, 1), (pd.Timedelta(123, "d"), 2)],
                {"a": [pd.NaT, pd.Timedelta(123, "d")], "b": [1, 2], "x": [11, 12]},
            ),
        ],
    )
    def test_reset_index_nat_multiindex(self, ix_data, exp_data):
        # GH#36541: that reset_index() does not raise ValueError
        ix = MultiIndex.from_tuples(ix_data, names=["a", "b"])
        result = DataFrame({"x": [11, 12]}, index=ix)
        result = result.reset_index()

        expected = DataFrame(exp_data)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "codes", ([[0, 0, 1, 1], [0, 1, 0, 1]], [[0, 0, -1, 1], [0, 1, 0, 1]])
    )
    def test_rest_index_multiindex_categorical_with_missing_values(self, codes):
        # GH#24206

        index = MultiIndex(
            [CategoricalIndex(["A", "B"]), CategoricalIndex(["a", "b"])], codes
        )
        data = {"col": range(len(index))}
        df = DataFrame(data=data, index=index)

        expected = DataFrame(
            {
                "level_0": Categorical.from_codes(codes[0], categories=["A", "B"]),
                "level_1": Categorical.from_codes(codes[1], categories=["a", "b"]),
                "col": range(4),
            }
        )

        res = df.reset_index()
        tm.assert_frame_equal(res, expected)

        # roundtrip
        res = expected.set_index(["level_0", "level_1"]).reset_index()
        tm.assert_frame_equal(res, expected)


@pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string) - GH#60338")
@pytest.mark.parametrize(
    "array, dtype",
    [
        (["a", "b"], object),
        (
            pd.period_range("12-1-2000", periods=2, freq="Q-DEC"),
            pd.PeriodDtype(freq="Q-DEC"),
        ),
    ],
)
def test_reset_index_dtypes_on_empty_frame_with_multiindex(
    array, dtype, using_infer_string
):
    # GH 19602 - Preserve dtype on empty DataFrame with MultiIndex
    idx = MultiIndex.from_product([[0, 1], [0.5, 1.0], array])
    result = DataFrame(index=idx)[:0].reset_index().dtypes
    if using_infer_string and dtype == object:
        dtype = pd.StringDtype(na_value=np.nan)
    expected = Series({"level_0": np.int64, "level_1": np.float64, "level_2": dtype})
    tm.assert_series_equal(result, expected)


def test_reset_index_empty_frame_with_datetime64_multiindex():
    # https://github.com/pandas-dev/pandas/issues/35606
    dti = pd.DatetimeIndex(["2020-07-20 00:00:00"], dtype="M8[ns]")
    idx = MultiIndex.from_product([dti, [3, 4]], names=["a", "b"])[:0]
    df = DataFrame(index=idx, columns=["c", "d"])
    result = df.reset_index()
    expected = DataFrame(
        columns=list("abcd"), index=RangeIndex(start=0, stop=0, step=1)
    )
    expected["a"] = expected["a"].astype("datetime64[ns]")
    expected["b"] = expected["b"].astype("int64")
    tm.assert_frame_equal(result, expected)


def test_reset_index_empty_frame_with_datetime64_multiindex_from_groupby(
    using_infer_string,
):
    # https://github.com/pandas-dev/pandas/issues/35657
    dti = pd.DatetimeIndex(["2020-01-01"], dtype="M8[ns]")
    df = DataFrame({"c1": [10.0], "c2": ["a"], "c3": dti})
    df = df.head(0).groupby(["c2", "c3"])[["c1"]].sum()
    result = df.reset_index()
    expected = DataFrame(
        columns=["c2", "c3", "c1"], index=RangeIndex(start=0, stop=0, step=1)
    )
    expected["c3"] = expected["c3"].astype("datetime64[ns]")
    expected["c1"] = expected["c1"].astype("float64")
    if using_infer_string:
        expected["c2"] = expected["c2"].astype("str")
    tm.assert_frame_equal(result, expected)


def test_reset_index_multiindex_nat():
    # GH 11479
    idx = range(3)
    tstamp = date_range("2015-07-01", freq="D", periods=3)
    df = DataFrame({"id": idx, "tstamp": tstamp, "a": list("abc")})
    df.loc[2, "tstamp"] = pd.NaT
    result = df.set_index(["id", "tstamp"]).reset_index("id")
    exp_dti = pd.DatetimeIndex(
        ["2015-07-01", "2015-07-02", "NaT"], dtype="M8[ns]", name="tstamp"
    )
    expected = DataFrame(
        {"id": range(3), "a": list("abc")},
        index=exp_dti,
    )
    tm.assert_frame_equal(result, expected)


def test_reset_index_interval_columns_object_cast():
    # GH 19136
    df = DataFrame(
        np.eye(2), index=Index([1, 2], name="Year"), columns=cut([1, 2], [0, 1, 2])
    )
    result = df.reset_index()
    expected = DataFrame(
        [[1, 1.0, 0.0], [2, 0.0, 1.0]],
        columns=Index(["Year", Interval(0, 1), Interval(1, 2)]),
    )
    tm.assert_frame_equal(result, expected)


def test_reset_index_rename(float_frame):
    # GH 6878
    result = float_frame.reset_index(names="new_name")
    expected = Series(float_frame.index.values, name="new_name")
    tm.assert_series_equal(result["new_name"], expected)

    result = float_frame.reset_index(names=123)
    expected = Series(float_frame.index.values, name=123)
    tm.assert_series_equal(result[123], expected)


def test_reset_index_rename_multiindex(float_frame):
    # GH 6878
    stacked_df = float_frame.stack(future_stack=True)[::2]
    stacked_df = DataFrame({"foo": stacked_df, "bar": stacked_df})

    names = ["first", "second"]
    stacked_df.index.names = names

    result = stacked_df.reset_index()
    expected = stacked_df.reset_index(names=["new_first", "new_second"])
    tm.assert_series_equal(result["first"], expected["new_first"], check_names=False)
    tm.assert_series_equal(result["second"], expected["new_second"], check_names=False)


def test_errorreset_index_rename(float_frame):
    # GH 6878
    stacked_df = float_frame.stack(future_stack=True)[::2]
    stacked_df = DataFrame({"first": stacked_df, "second": stacked_df})

    with pytest.raises(
        ValueError, match="Index names must be str or 1-dimensional list"
    ):
        stacked_df.reset_index(names={"first": "new_first", "second": "new_second"})

    with pytest.raises(IndexError, match="list index out of range"):
        stacked_df.reset_index(names=["new_first"])


def test_reset_index_false_index_name():
    result_series = Series(data=range(5, 10), index=range(5))
    result_series.index.name = False
    result_series.reset_index()
    expected_series = Series(range(5, 10), RangeIndex(range(5), name=False))
    tm.assert_series_equal(result_series, expected_series)

    # GH 38147
    result_frame = DataFrame(data=range(5, 10), index=range(5))
    result_frame.index.name = False
    result_frame.reset_index()
    expected_frame = DataFrame(range(5, 10), RangeIndex(range(5), name=False))
    tm.assert_frame_equal(result_frame, expected_frame)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_factortools.py ===
"""Tools for polynomial factorization routines in characteristic zero. """

from sympy.polys.rings import ring, xring
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX

from sympy.polys import polyconfig as config
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyclasses import ANP
from sympy.polys.specialpolys import f_polys, w_polys

from sympy.core.numbers import I
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises, XFAIL


f_0, f_1, f_2, f_3, f_4, f_5, f_6 = f_polys()
w_1, w_2 = w_polys()

def test_dup_trial_division():
    R, x = ring("x", ZZ)
    assert R.dup_trial_division(x**5 + 8*x**4 + 25*x**3 + 38*x**2 + 28*x + 8, (x + 1, x + 2)) == [(x + 1, 2), (x + 2, 3)]


def test_dmp_trial_division():
    R, x, y = ring("x,y", ZZ)
    assert R.dmp_trial_division(x**5 + 8*x**4 + 25*x**3 + 38*x**2 + 28*x + 8, (x + 1, x + 2)) == [(x + 1, 2), (x + 2, 3)]


def test_dup_zz_mignotte_bound():
    R, x = ring("x", ZZ)
    assert R.dup_zz_mignotte_bound(2*x**2 + 3*x + 4) == 6
    assert R.dup_zz_mignotte_bound(x**3 + 14*x**2 + 56*x + 64) == 152


def test_dmp_zz_mignotte_bound():
    R, x, y = ring("x,y", ZZ)
    assert R.dmp_zz_mignotte_bound(2*x**2 + 3*x + 4) == 32


def test_dup_zz_hensel_step():
    R, x = ring("x", ZZ)

    f = x**4 - 1
    g = x**3 + 2*x**2 - x - 2
    h = x - 2
    s = -2
    t = 2*x**2 - 2*x - 1

    G, H, S, T = R.dup_zz_hensel_step(5, f, g, h, s, t)

    assert G == x**3 + 7*x**2 - x - 7
    assert H == x - 7
    assert S == 8
    assert T == -8*x**2 - 12*x - 1


def test_dup_zz_hensel_lift():
    R, x = ring("x", ZZ)

    f = x**4 - 1
    F = [x - 1, x - 2, x + 2, x + 1]

    assert R.dup_zz_hensel_lift(ZZ(5), f, F, 4) == \
        [x - 1, x - 182, x + 182, x + 1]


def test_dup_zz_irreducible_p():
    R, x = ring("x", ZZ)

    assert R.dup_zz_irreducible_p(3*x**4 + 2*x**3 + 6*x**2 + 8*x + 7) is None
    assert R.dup_zz_irreducible_p(3*x**4 + 2*x**3 + 6*x**2 + 8*x + 4) is None

    assert R.dup_zz_irreducible_p(3*x**4 + 2*x**3 + 6*x**2 + 8*x + 10) is True
    assert R.dup_zz_irreducible_p(3*x**4 + 2*x**3 + 6*x**2 + 8*x + 14) is True


def test_dup_cyclotomic_p():
    R, x = ring("x", ZZ)

    assert R.dup_cyclotomic_p(x - 1) is True
    assert R.dup_cyclotomic_p(x + 1) is True
    assert R.dup_cyclotomic_p(x**2 + x + 1) is True
    assert R.dup_cyclotomic_p(x**2 + 1) is True
    assert R.dup_cyclotomic_p(x**4 + x**3 + x**2 + x + 1) is True
    assert R.dup_cyclotomic_p(x**2 - x + 1) is True
    assert R.dup_cyclotomic_p(x**6 + x**5 + x**4 + x**3 + x**2 + x + 1) is True
    assert R.dup_cyclotomic_p(x**4 + 1) is True
    assert R.dup_cyclotomic_p(x**6 + x**3 + 1) is True

    assert R.dup_cyclotomic_p(0) is False
    assert R.dup_cyclotomic_p(1) is False
    assert R.dup_cyclotomic_p(x) is False
    assert R.dup_cyclotomic_p(x + 2) is False
    assert R.dup_cyclotomic_p(3*x + 1) is False
    assert R.dup_cyclotomic_p(x**2 - 1) is False

    f = x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1
    assert R.dup_cyclotomic_p(f) is False

    g = x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1
    assert R.dup_cyclotomic_p(g) is True

    R, x = ring("x", QQ)
    assert R.dup_cyclotomic_p(x**2 + x + 1) is True
    assert R.dup_cyclotomic_p(QQ(1,2)*x**2 + x + 1) is False

    R, x = ring("x", ZZ["y"])
    assert R.dup_cyclotomic_p(x**2 + x + 1) is False


def test_dup_zz_cyclotomic_poly():
    R, x = ring("x", ZZ)

    assert R.dup_zz_cyclotomic_poly(1) == x - 1
    assert R.dup_zz_cyclotomic_poly(2) == x + 1
    assert R.dup_zz_cyclotomic_poly(3) == x**2 + x + 1
    assert R.dup_zz_cyclotomic_poly(4) == x**2 + 1
    assert R.dup_zz_cyclotomic_poly(5) == x**4 + x**3 + x**2 + x + 1
    assert R.dup_zz_cyclotomic_poly(6) == x**2 - x + 1
    assert R.dup_zz_cyclotomic_poly(7) == x**6 + x**5 + x**4 + x**3 + x**2 + x + 1
    assert R.dup_zz_cyclotomic_poly(8) == x**4 + 1
    assert R.dup_zz_cyclotomic_poly(9) == x**6 + x**3 + 1


def test_dup_zz_cyclotomic_factor():
    R, x = ring("x", ZZ)

    assert R.dup_zz_cyclotomic_factor(0) is None
    assert R.dup_zz_cyclotomic_factor(1) is None

    assert R.dup_zz_cyclotomic_factor(2*x**10 - 1) is None
    assert R.dup_zz_cyclotomic_factor(x**10 - 3) is None
    assert R.dup_zz_cyclotomic_factor(x**10 + x**5 - 1) is None

    assert R.dup_zz_cyclotomic_factor(x + 1) == [x + 1]
    assert R.dup_zz_cyclotomic_factor(x - 1) == [x - 1]

    assert R.dup_zz_cyclotomic_factor(x**2 + 1) == [x**2 + 1]
    assert R.dup_zz_cyclotomic_factor(x**2 - 1) == [x - 1, x + 1]

    assert R.dup_zz_cyclotomic_factor(x**27 + 1) == \
        [x + 1, x**2 - x + 1, x**6 - x**3 + 1, x**18 - x**9 + 1]
    assert R.dup_zz_cyclotomic_factor(x**27 - 1) == \
        [x - 1, x**2 + x + 1, x**6 + x**3 + 1, x**18 + x**9 + 1]


def test_dup_zz_factor():
    R, x = ring("x", ZZ)

    assert R.dup_zz_factor(0) == (0, [])
    assert R.dup_zz_factor(7) == (7, [])
    assert R.dup_zz_factor(-7) == (-7, [])

    assert R.dup_zz_factor_sqf(0) == (0, [])
    assert R.dup_zz_factor_sqf(7) == (7, [])
    assert R.dup_zz_factor_sqf(-7) == (-7, [])

    assert R.dup_zz_factor(2*x + 4) == (2, [(x + 2, 1)])
    assert R.dup_zz_factor_sqf(2*x + 4) == (2, [x + 2])

    f = x**4 + x + 1

    for i in range(0, 20):
        assert R.dup_zz_factor(f) == (1, [(f, 1)])

    assert R.dup_zz_factor(x**2 + 2*x + 2) == \
        (1, [(x**2 + 2*x + 2, 1)])

    assert R.dup_zz_factor(18*x**2 + 12*x + 2) == \
        (2, [(3*x + 1, 2)])

    assert R.dup_zz_factor(-9*x**2 + 1) == \
        (-1, [(3*x - 1, 1),
              (3*x + 1, 1)])

    assert R.dup_zz_factor_sqf(-9*x**2 + 1) == \
        (-1, [3*x - 1,
              3*x + 1])

    # The order of the factors will be different when the ground types are
    # flint. At the higher level dup_factor_list will sort the factors.
    c, factors = R.dup_zz_factor(x**3 - 6*x**2 + 11*x - 6)
    assert c == 1
    assert set(factors) == {(x - 3, 1), (x - 2, 1), (x - 1, 1)}

    assert R.dup_zz_factor_sqf(x**3 - 6*x**2 + 11*x - 6) == \
        (1, [x - 3,
             x - 2,
             x - 1])

    assert R.dup_zz_factor(3*x**3 + 10*x**2 + 13*x + 10) == \
        (1, [(x + 2, 1),
             (3*x**2 + 4*x + 5, 1)])

    assert R.dup_zz_factor_sqf(3*x**3 + 10*x**2 + 13*x + 10) == \
        (1, [x + 2,
             3*x**2 + 4*x + 5])

    c, factors = R.dup_zz_factor(-x**6 + x**2)
    assert c == -1
    assert set(factors) == {(x, 2), (x - 1, 1), (x + 1, 1), (x**2 + 1, 1)}

    f = 1080*x**8 + 5184*x**7 + 2099*x**6 + 744*x**5 + 2736*x**4 - 648*x**3 + 129*x**2 - 324

    assert R.dup_zz_factor(f) == \
        (1, [(5*x**4 + 24*x**3 + 9*x**2 + 12, 1),
             (216*x**4 + 31*x**2 - 27, 1)])

    f = -29802322387695312500000000000000000000*x**25 \
      + 2980232238769531250000000000000000*x**20 \
      + 1743435859680175781250000000000*x**15 \
      + 114142894744873046875000000*x**10 \
      - 210106372833251953125*x**5 \
      + 95367431640625

    c, factors = R.dup_zz_factor(f)
    assert c == -95367431640625
    assert set(factors) == {
        (5*x - 1, 1),
        (100*x**2 + 10*x - 1, 2),
        (625*x**4 + 125*x**3 + 25*x**2 + 5*x + 1, 1),
        (10000*x**4 - 3000*x**3 + 400*x**2 - 20*x + 1, 2),
        (10000*x**4 + 2000*x**3 + 400*x**2 + 30*x + 1, 2),
    }

    f = x**10 - 1

    config.setup('USE_CYCLOTOMIC_FACTOR', True)
    c0, F_0 = R.dup_zz_factor(f)

    config.setup('USE_CYCLOTOMIC_FACTOR', False)
    c1, F_1 = R.dup_zz_factor(f)

    assert c0 == c1 == 1
    assert set(F_0) == set(F_1) == {
        (x - 1, 1),
        (x + 1, 1),
        (x**4 - x**3 + x**2 - x + 1, 1),
        (x**4 + x**3 + x**2 + x + 1, 1),
    }

    config.setup('USE_CYCLOTOMIC_FACTOR')

    f = x**10 + 1

    config.setup('USE_CYCLOTOMIC_FACTOR', True)
    F_0 = R.dup_zz_factor(f)

    config.setup('USE_CYCLOTOMIC_FACTOR', False)
    F_1 = R.dup_zz_factor(f)

    assert F_0 == F_1 == \
        (1, [(x**2 + 1, 1),
             (x**8 - x**6 + x**4 - x**2 + 1, 1)])

    config.setup('USE_CYCLOTOMIC_FACTOR')

def test_dmp_zz_wang():
    R, x,y,z = ring("x,y,z", ZZ)
    UV, _x = ring("x", ZZ)

    p = ZZ(nextprime(R.dmp_zz_mignotte_bound(w_1)))
    assert p == 6291469

    t_1, k_1, e_1 = y, 1, ZZ(-14)
    t_2, k_2, e_2 = z, 2, ZZ(3)
    t_3, k_3, e_3 = y + z, 2, ZZ(-11)
    t_4, k_4, e_4 = y - z, 1, ZZ(-17)

    T = [t_1, t_2, t_3, t_4]
    K = [k_1, k_2, k_3, k_4]
    E = [e_1, e_2, e_3, e_4]

    T = zip([ t.drop(x) for t in T ], K)

    A = [ZZ(-14), ZZ(3)]

    S = R.dmp_eval_tail(w_1, A)
    cs, s = UV.dup_primitive(S)

    assert cs == 1 and s == S == \
        1036728*_x**6 + 915552*_x**5 + 55748*_x**4 + 105621*_x**3 - 17304*_x**2 - 26841*_x - 644

    assert R.dmp_zz_wang_non_divisors(E, cs, ZZ(4)) == [7, 3, 11, 17]
    assert UV.dup_sqf_p(s) and UV.dup_degree(s) == R.dmp_degree(w_1)

    _, H = UV.dup_zz_factor_sqf(s)

    h_1 = 44*_x**2 + 42*_x + 1
    h_2 = 126*_x**2 - 9*_x + 28
    h_3 = 187*_x**2 - 23

    assert H == [h_1, h_2, h_3]

    LC = [ lc.drop(x) for lc in [-4*y - 4*z, -y*z**2, y**2 - z**2] ]

    assert R.dmp_zz_wang_lead_coeffs(w_1, T, cs, E, H, A) == (w_1, H, LC)

    factors = R.dmp_zz_wang_hensel_lifting(w_1, H, LC, A, p)
    assert R.dmp_expand(factors) == w_1


@XFAIL
def test_dmp_zz_wang_fail():
    R, x,y,z = ring("x,y,z", ZZ)
    UV, _x = ring("x", ZZ)

    p = ZZ(nextprime(R.dmp_zz_mignotte_bound(w_1)))
    assert p == 6291469

    H_1 = [44*x**2 + 42*x + 1, 126*x**2 - 9*x + 28, 187*x**2 - 23]
    H_2 = [-4*x**2*y - 12*x**2 - 3*x*y + 1, -9*x**2*y - 9*x - 2*y, x**2*y**2 - 9*x**2 + y - 9]
    H_3 = [-4*x**2*y - 12*x**2 - 3*x*y + 1, -9*x**2*y - 9*x - 2*y, x**2*y**2 - 9*x**2 + y - 9]

    c_1 = -70686*x**5 - 5863*x**4 - 17826*x**3 + 2009*x**2 + 5031*x + 74
    c_2 = 9*x**5*y**4 + 12*x**5*y**3 - 45*x**5*y**2 - 108*x**5*y - 324*x**5 + 18*x**4*y**3 - 216*x**4*y**2 - 810*x**4*y + 2*x**3*y**4 + 9*x**3*y**3 - 252*x**3*y**2 - 288*x**3*y - 945*x**3 - 30*x**2*y**2 - 414*x**2*y + 2*x*y**3 - 54*x*y**2 - 3*x*y + 81*x + 12*y
    c_3 = -36*x**4*y**2 - 108*x**4*y - 27*x**3*y**2 - 36*x**3*y - 108*x**3 - 8*x**2*y**2 - 42*x**2*y - 6*x*y**2 + 9*x + 2*y

    assert R.dmp_zz_diophantine(H_1, c_1, [], 5, p) == [-3*x, -2, 1]
    assert R.dmp_zz_diophantine(H_2, c_2, [ZZ(-14)], 5, p) == [-x*y, -3*x, -6]
    assert R.dmp_zz_diophantine(H_3, c_3, [ZZ(-14)], 5, p) == [0, 0, -1]


def test_issue_6355():
    # This tests a bug in the Wang algorithm that occurred only with a very
    # specific set of random numbers.
    random_sequence = [-1, -1, 0, 0, 0, 0, -1, -1, 0, -1, 3, -1, 3, 3, 3, 3, -1, 3]

    R, x, y, z = ring("x,y,z", ZZ)
    f = 2*x**2 + y*z - y - z**2 + z

    assert R.dmp_zz_wang(f, seed=random_sequence) == [f]


def test_dmp_zz_factor():
    R, x = ring("x", ZZ)
    assert R.dmp_zz_factor(0) == (0, [])
    assert R.dmp_zz_factor(7) == (7, [])
    assert R.dmp_zz_factor(-7) == (-7, [])

    assert R.dmp_zz_factor(x**2 - 9) == (1, [(x - 3, 1), (x + 3, 1)])

    R, x, y = ring("x,y", ZZ)
    assert R.dmp_zz_factor(0) == (0, [])
    assert R.dmp_zz_factor(7) == (7, [])
    assert R.dmp_zz_factor(-7) == (-7, [])

    assert R.dmp_zz_factor(x) == (1, [(x, 1)])
    assert R.dmp_zz_factor(4*x) == (4, [(x, 1)])
    assert R.dmp_zz_factor(4*x + 2) == (2, [(2*x + 1, 1)])
    assert R.dmp_zz_factor(x*y + 1) == (1, [(x*y + 1, 1)])
    assert R.dmp_zz_factor(y**2 + 1) == (1, [(y**2 + 1, 1)])
    assert R.dmp_zz_factor(y**2 - 1) == (1, [(y - 1, 1), (y + 1, 1)])

    assert R.dmp_zz_factor(x**2*y**2 + 6*x**2*y + 9*x**2 - 1) == (1, [(x*y + 3*x - 1, 1), (x*y + 3*x + 1, 1)])
    assert R.dmp_zz_factor(x**2*y**2 - 9) == (1, [(x*y - 3, 1), (x*y + 3, 1)])

    R, x, y, z = ring("x,y,z", ZZ)
    assert R.dmp_zz_factor(x**2*y**2*z**2 - 9) == \
        (1, [(x*y*z - 3, 1),
             (x*y*z + 3, 1)])

    R, x, y, z, u = ring("x,y,z,u", ZZ)
    assert R.dmp_zz_factor(x**2*y**2*z**2*u**2 - 9) == \
        (1, [(x*y*z*u - 3, 1),
             (x*y*z*u + 3, 1)])

    R, x, y, z = ring("x,y,z", ZZ)
    assert R.dmp_zz_factor(f_1) == \
        (1, [(x + y*z + 20, 1),
             (x*y + z + 10, 1),
             (x*z + y + 30, 1)])

    assert R.dmp_zz_factor(f_2) == \
        (1, [(x**2*y**2 + x**2*z**2 + y + 90, 1),
             (x**3*y + x**3*z + z - 11, 1)])

    assert R.dmp_zz_factor(f_3) == \
        (1, [(x**2*y**2 + x*z**4 + x + z, 1),
             (x**3 + x*y*z + y**2 + y*z**3, 1)])

    assert R.dmp_zz_factor(f_4) == \
        (-1, [(x*y**3 + z**2, 1),
              (x**2*z + y**4*z**2 + 5, 1),
              (x**3*y - z**2 - 3, 1),
              (x**3*y**4 + z**2, 1)])

    assert R.dmp_zz_factor(f_5) == \
        (-1, [(x + y - z, 3)])

    R, x, y, z, t = ring("x,y,z,t", ZZ)
    assert R.dmp_zz_factor(f_6) == \
        (1, [(47*x*y + z**3*t**2 - t**2, 1),
             (45*x**3 - 9*y**3 - y**2 + 3*z**3 + 2*z*t, 1)])

    R, x, y, z = ring("x,y,z", ZZ)
    assert R.dmp_zz_factor(w_1) == \
        (1, [(x**2*y**2 - x**2*z**2 + y - z**2, 1),
             (x**2*y*z**2 + 3*x*z + 2*y, 1),
             (4*x**2*y + 4*x**2*z + x*y*z - 1, 1)])

    R, x, y = ring("x,y", ZZ)
    f = -12*x**16*y + 240*x**12*y**3 - 768*x**10*y**4 + 1080*x**8*y**5 - 768*x**6*y**6 + 240*x**4*y**7 - 12*y**9

    assert R.dmp_zz_factor(f) == \
        (-12, [(y, 1),
               (x**2 - y, 6),
               (x**4 + 6*x**2*y + y**2, 1)])


def test_dup_qq_i_factor():
    R, x = ring("x", QQ_I)
    i = QQ_I(0, 1)

    assert R.dup_qq_i_factor(x**2 - 2) == (QQ_I(1, 0), [(x**2 - 2, 1)])

    assert R.dup_qq_i_factor(x**2 - 1) == (QQ_I(1, 0), [(x - 1, 1), (x + 1, 1)])

    assert R.dup_qq_i_factor(x**2 + 1) == (QQ_I(1, 0), [(x - i, 1), (x + i, 1)])

    assert R.dup_qq_i_factor(x**2/4 + 1) == \
            (QQ_I(QQ(1, 4), 0), [(x - 2*i, 1), (x + 2*i, 1)])

    assert R.dup_qq_i_factor(x**2 + 4) == \
            (QQ_I(1, 0), [(x - 2*i, 1), (x + 2*i, 1)])

    assert R.dup_qq_i_factor(x**2 + 2*x + 1) == \
            (QQ_I(1, 0), [(x + 1, 2)])

    assert R.dup_qq_i_factor(x**2 + 2*i*x - 1) == \
            (QQ_I(1, 0), [(x + i, 2)])

    f = 8192*x**2 + x*(22656 + 175232*i) - 921416 + 242313*i

    assert R.dup_qq_i_factor(f) == \
            (QQ_I(8192, 0), [(x + QQ_I(QQ(177, 128), QQ(1369, 128)), 2)])


def test_dmp_qq_i_factor():
    R, x, y = ring("x, y", QQ_I)
    i = QQ_I(0, 1)

    assert R.dmp_qq_i_factor(x**2 + 2*y**2) == \
            (QQ_I(1, 0), [(x**2 + 2*y**2, 1)])

    assert R.dmp_qq_i_factor(x**2 + y**2) == \
            (QQ_I(1, 0), [(x - i*y, 1), (x + i*y, 1)])

    assert R.dmp_qq_i_factor(x**2 + y**2/4) == \
            (QQ_I(1, 0), [(x - i*y/2, 1), (x + i*y/2, 1)])

    assert R.dmp_qq_i_factor(4*x**2 + y**2) == \
            (QQ_I(4, 0), [(x - i*y/2, 1), (x + i*y/2, 1)])


def test_dup_zz_i_factor():
    R, x = ring("x", ZZ_I)
    i = ZZ_I(0, 1)

    assert R.dup_zz_i_factor(x**2 - 2) == (ZZ_I(1, 0), [(x**2 - 2, 1)])

    assert R.dup_zz_i_factor(x**2 - 1) == (ZZ_I(1, 0), [(x - 1, 1), (x + 1, 1)])

    assert R.dup_zz_i_factor(x**2 + 1) == (ZZ_I(1, 0), [(x - i, 1), (x + i, 1)])

    assert R.dup_zz_i_factor(x**2 + 4) == \
            (ZZ_I(1, 0), [(x - 2*i, 1), (x + 2*i, 1)])

    assert R.dup_zz_i_factor(x**2 + 2*x + 1) == \
            (ZZ_I(1, 0), [(x + 1, 2)])

    assert R.dup_zz_i_factor(x**2 + 2*i*x - 1) == \
            (ZZ_I(1, 0), [(x + i, 2)])

    f = 8192*x**2 + x*(22656 + 175232*i) - 921416 + 242313*i

    assert R.dup_zz_i_factor(f) == \
            (ZZ_I(0, 1), [((64 - 64*i)*x + (773 + 596*i), 2)])


def test_dmp_zz_i_factor():
    R, x, y = ring("x, y", ZZ_I)
    i = ZZ_I(0, 1)

    assert R.dmp_zz_i_factor(x**2 + 2*y**2) == \
            (ZZ_I(1, 0), [(x**2 + 2*y**2, 1)])

    assert R.dmp_zz_i_factor(x**2 + y**2) == \
            (ZZ_I(1, 0), [(x - i*y, 1), (x + i*y, 1)])

    assert R.dmp_zz_i_factor(4*x**2 + y**2) == \
            (ZZ_I(1, 0), [(2*x - i*y, 1), (2*x + i*y, 1)])


def test_dup_ext_factor():
    R, x = ring("x", QQ.algebraic_field(I))
    def anp(element):
        return ANP(element, [QQ(1), QQ(0), QQ(1)], QQ)

    assert R.dup_ext_factor(0) == (anp([]), [])

    f = anp([QQ(1)])*x + anp([QQ(1)])

    assert R.dup_ext_factor(f) == (anp([QQ(1)]), [(f, 1)])

    g = anp([QQ(2)])*x + anp([QQ(2)])

    assert R.dup_ext_factor(g) == (anp([QQ(2)]), [(f, 1)])

    f = anp([QQ(7)])*x**4 + anp([QQ(1, 1)])
    g = anp([QQ(1)])*x**4 + anp([QQ(1, 7)])

    assert R.dup_ext_factor(f) == (anp([QQ(7)]), [(g, 1)])

    f = anp([QQ(1)])*x**4 + anp([QQ(1)])

    assert R.dup_ext_factor(f) == \
        (anp([QQ(1, 1)]), [(anp([QQ(1)])*x**2 + anp([QQ(-1), QQ(0)]), 1),
                           (anp([QQ(1)])*x**2 + anp([QQ( 1), QQ(0)]), 1)])

    f = anp([QQ(4, 1)])*x**2 + anp([QQ(9, 1)])

    assert R.dup_ext_factor(f) == \
        (anp([QQ(4, 1)]), [(anp([QQ(1, 1)])*x + anp([-QQ(3, 2), QQ(0, 1)]), 1),
                           (anp([QQ(1, 1)])*x + anp([ QQ(3, 2), QQ(0, 1)]), 1)])

    f = anp([QQ(4, 1)])*x**4 + anp([QQ(8, 1)])*x**3 + anp([QQ(77, 1)])*x**2 + anp([QQ(18, 1)])*x + anp([QQ(153, 1)])

    assert R.dup_ext_factor(f) == \
        (anp([QQ(4, 1)]), [(anp([QQ(1, 1)])*x + anp([-QQ(4, 1), QQ(1, 1)]), 1),
                           (anp([QQ(1, 1)])*x + anp([-QQ(3, 2), QQ(0, 1)]), 1),
                           (anp([QQ(1, 1)])*x + anp([ QQ(3, 2), QQ(0, 1)]), 1),
                           (anp([QQ(1, 1)])*x + anp([ QQ(4, 1), QQ(1, 1)]), 1)])

    R, x = ring("x", QQ.algebraic_field(sqrt(2)))
    def anp(element):
        return ANP(element, [QQ(1), QQ(0), QQ(-2)], QQ)

    f = anp([QQ(1)])*x**4 + anp([QQ(1, 1)])

    assert R.dup_ext_factor(f) == \
        (anp([QQ(1)]), [(anp([QQ(1)])*x**2 + anp([QQ(-1), QQ(0)])*x + anp([QQ(1)]), 1),
                        (anp([QQ(1)])*x**2 + anp([QQ( 1), QQ(0)])*x + anp([QQ(1)]), 1)])

    f = anp([QQ(1, 1)])*x**2 + anp([QQ(2), QQ(0)])*x + anp([QQ(2, 1)])

    assert R.dup_ext_factor(f) == \
        (anp([QQ(1, 1)]), [(anp([1])*x + anp([1, 0]), 2)])

    assert R.dup_ext_factor(f**3) == \
        (anp([QQ(1, 1)]), [(anp([1])*x + anp([1, 0]), 6)])

    f *= anp([QQ(2, 1)])

    assert R.dup_ext_factor(f) == \
        (anp([QQ(2, 1)]), [(anp([1])*x + anp([1, 0]), 2)])

    assert R.dup_ext_factor(f**3) == \
        (anp([QQ(8, 1)]), [(anp([1])*x + anp([1, 0]), 6)])


def test_dmp_ext_factor():
    K = QQ.algebraic_field(sqrt(2))
    R, x,y = ring("x,y", K)
    sqrt2 = K.unit

    def anp(x):
        return ANP(x, [QQ(1), QQ(0), QQ(-2)], QQ)

    assert R.dmp_ext_factor(0) == (anp([]), [])

    f = anp([QQ(1)])*x + anp([QQ(1)])

    assert R.dmp_ext_factor(f) == (anp([QQ(1)]), [(f, 1)])

    g = anp([QQ(2)])*x + anp([QQ(2)])

    assert R.dmp_ext_factor(g) == (anp([QQ(2)]), [(f, 1)])

    f = anp([QQ(1)])*x**2 + anp([QQ(-2)])*y**2

    assert R.dmp_ext_factor(f) == \
        (anp([QQ(1)]), [(anp([QQ(1)])*x + anp([QQ(-1), QQ(0)])*y, 1),
                        (anp([QQ(1)])*x + anp([QQ( 1), QQ(0)])*y, 1)])

    f = anp([QQ(2)])*x**2 + anp([QQ(-4)])*y**2

    assert R.dmp_ext_factor(f) == \
        (anp([QQ(2)]), [(anp([QQ(1)])*x + anp([QQ(-1), QQ(0)])*y, 1),
                        (anp([QQ(1)])*x + anp([QQ( 1), QQ(0)])*y, 1)])

    f1 = y + 1
    f2 = y + sqrt2
    f3 = x**2 + x + 2 + 3*sqrt2
    f = f1**2 * f2**2 * f3**2
    assert R.dmp_ext_factor(f) == (K.one, [(f1, 2), (f2, 2), (f3, 2)])


def test_dup_factor_list():
    R, x = ring("x", ZZ)
    assert R.dup_factor_list(0) == (0, [])
    assert R.dup_factor_list(7) == (7, [])

    R, x = ring("x", QQ)
    assert R.dup_factor_list(0) == (0, [])
    assert R.dup_factor_list(QQ(1, 7)) == (QQ(1, 7), [])

    R, x = ring("x", ZZ['t'])
    assert R.dup_factor_list(0) == (0, [])
    assert R.dup_factor_list(7) == (7, [])

    R, x = ring("x", QQ['t'])
    assert R.dup_factor_list(0) == (0, [])
    assert R.dup_factor_list(QQ(1, 7)) == (QQ(1, 7), [])

    R, x = ring("x", ZZ)
    assert R.dup_factor_list_include(0) == [(0, 1)]
    assert R.dup_factor_list_include(7) == [(7, 1)]

    assert R.dup_factor_list(x**2 + 2*x + 1) == (1, [(x + 1, 2)])
    assert R.dup_factor_list_include(x**2 + 2*x + 1) == [(x + 1, 2)]
    # issue 8037
    assert R.dup_factor_list(6*x**2 - 5*x - 6) == (1, [(2*x - 3, 1), (3*x + 2, 1)])

    R, x = ring("x", QQ)
    assert R.dup_factor_list(QQ(1,2)*x**2 + x + QQ(1,2)) == (QQ(1, 2), [(x + 1, 2)])

    R, x = ring("x", FF(2))
    assert R.dup_factor_list(x**2 + 1) == (1, [(x + 1, 2)])

    R, x = ring("x", RR)
    assert R.dup_factor_list(1.0*x**2 + 2.0*x + 1.0) == (1.0, [(1.0*x + 1.0, 2)])
    assert R.dup_factor_list(2.0*x**2 + 4.0*x + 2.0) == (2.0, [(1.0*x + 1.0, 2)])

    f = 6.7225336055071*x**2 - 10.6463972754741*x - 0.33469524022264
    coeff, factors = R.dup_factor_list(f)
    assert coeff == RR(10.6463972754741)
    assert len(factors) == 1
    assert factors[0][0].max_norm() == RR(1.0)
    assert factors[0][1] == 1

    Rt, t = ring("t", ZZ)
    R, x = ring("x", Rt)

    f = 4*t*x**2 + 4*t**2*x

    assert R.dup_factor_list(f) == \
        (4*t, [(x, 1),
             (x + t, 1)])

    Rt, t = ring("t", QQ)
    R, x = ring("x", Rt)

    f = QQ(1, 2)*t*x**2 + QQ(1, 2)*t**2*x

    assert R.dup_factor_list(f) == \
        (QQ(1, 2)*t, [(x, 1),
                    (x + t, 1)])

    R, x = ring("x", QQ.algebraic_field(I))
    def anp(element):
        return ANP(element, [QQ(1), QQ(0), QQ(1)], QQ)

    f = anp([QQ(1, 1)])*x**4 + anp([QQ(2, 1)])*x**2

    assert R.dup_factor_list(f) == \
        (anp([QQ(1, 1)]), [(anp([QQ(1, 1)])*x, 2),
                           (anp([QQ(1, 1)])*x**2 + anp([])*x + anp([QQ(2, 1)]), 1)])

    R, x = ring("x", EX)
    raises(DomainError, lambda: R.dup_factor_list(EX(sin(1))))


def test_dmp_factor_list():
    R, x, y = ring("x,y", ZZ)
    assert R.dmp_factor_list(0) == (ZZ(0), [])
    assert R.dmp_factor_list(7) == (7, [])

    R, x, y = ring("x,y", QQ)
    assert R.dmp_factor_list(0) == (QQ(0), [])
    assert R.dmp_factor_list(QQ(1, 7)) == (QQ(1, 7), [])

    Rt, t = ring("t", ZZ)
    R, x, y = ring("x,y", Rt)
    assert R.dmp_factor_list(0) == (0, [])
    assert R.dmp_factor_list(7) == (ZZ(7), [])

    Rt, t = ring("t", QQ)
    R, x, y = ring("x,y", Rt)
    assert R.dmp_factor_list(0) == (0, [])
    assert R.dmp_factor_list(QQ(1, 7)) == (QQ(1, 7), [])

    R, x, y = ring("x,y", ZZ)
    assert R.dmp_factor_list_include(0) == [(0, 1)]
    assert R.dmp_factor_list_include(7) == [(7, 1)]

    R, X = xring("x:200", ZZ)

    f, g = X[0]**2 + 2*X[0] + 1, X[0] + 1
    assert R.dmp_factor_list(f) == (1, [(g, 2)])

    f, g = X[-1]**2 + 2*X[-1] + 1, X[-1] + 1
    assert R.dmp_factor_list(f) == (1, [(g, 2)])

    R, x = ring("x", ZZ)
    assert R.dmp_factor_list(x**2 + 2*x + 1) == (1, [(x + 1, 2)])
    R, x = ring("x", QQ)
    assert R.dmp_factor_list(QQ(1,2)*x**2 + x + QQ(1,2)) == (QQ(1,2), [(x + 1, 2)])

    R, x, y = ring("x,y", ZZ)
    assert R.dmp_factor_list(x**2 + 2*x + 1) == (1, [(x + 1, 2)])
    R, x, y = ring("x,y", QQ)
    assert R.dmp_factor_list(QQ(1,2)*x**2 + x + QQ(1,2)) == (QQ(1,2), [(x + 1, 2)])

    R, x, y = ring("x,y", ZZ)
    f = 4*x**2*y + 4*x*y**2

    assert R.dmp_factor_list(f) == \
        (4, [(y, 1),
             (x, 1),
             (x + y, 1)])

    assert R.dmp_factor_list_include(f) == \
        [(4*y, 1),
         (x, 1),
         (x + y, 1)]

    R, x, y = ring("x,y", QQ)
    f = QQ(1,2)*x**2*y + QQ(1,2)*x*y**2

    assert R.dmp_factor_list(f) == \
        (QQ(1,2), [(y, 1),
                   (x, 1),
                   (x + y, 1)])

    R, x, y = ring("x,y", RR)
    f = 2.0*x**2 - 8.0*y**2

    assert R.dmp_factor_list(f) == \
        (RR(8.0), [(0.5*x - y, 1),
                   (0.5*x + y, 1)])

    f = 6.7225336055071*x**2*y**2 - 10.6463972754741*x*y - 0.33469524022264
    coeff, factors = R.dmp_factor_list(f)
    assert coeff == RR(10.6463972754741)
    assert len(factors) == 1
    assert factors[0][0].max_norm() == RR(1.0)
    assert factors[0][1] == 1

    Rt, t = ring("t", ZZ)
    R, x, y = ring("x,y", Rt)
    f = 4*t*x**2 + 4*t**2*x

    assert R.dmp_factor_list(f) == \
        (4*t, [(x, 1),
             (x + t, 1)])

    Rt, t = ring("t", QQ)
    R, x, y = ring("x,y", Rt)
    f = QQ(1, 2)*t*x**2 + QQ(1, 2)*t**2*x

    assert R.dmp_factor_list(f) == \
        (QQ(1, 2)*t, [(x, 1),
                    (x + t, 1)])

    R, x, y = ring("x,y", FF(2))
    raises(NotImplementedError, lambda: R.dmp_factor_list(x**2 + y**2))

    R, x, y = ring("x,y", EX)
    raises(DomainError, lambda: R.dmp_factor_list(EX(sin(1))))


def test_dup_irreducible_p():
    R, x = ring("x", ZZ)
    assert R.dup_irreducible_p(x**2 + x + 1) is True
    assert R.dup_irreducible_p(x**2 + 2*x + 1) is False


def test_dmp_irreducible_p():
    R, x, y = ring("x,y", ZZ)
    assert R.dmp_irreducible_p(x**2 + x + 1) is True
    assert R.dmp_irreducible_p(x**2 + 2*x + 1) is False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\parser\test_na_values.py ===
"""
Tests that NA values are properly handled during
parsing for all of the parsers defined in parsers.py
"""
from io import StringIO

import numpy as np
import pytest

from pandas._libs.parsers import STR_NA_VALUES

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


def test_string_nas(all_parsers):
    parser = all_parsers
    data = """A,B,C
a,b,c
d,,f
,g,h
"""
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [["a", "b", "c"], ["d", np.nan, "f"], [np.nan, "g", "h"]],
        columns=["A", "B", "C"],
    )
    if parser.engine == "pyarrow":
        expected.loc[2, "A"] = None
        expected.loc[1, "B"] = None
    tm.assert_frame_equal(result, expected)


def test_detect_string_na(all_parsers):
    parser = all_parsers
    data = """A,B
foo,bar
NA,baz
NaN,nan
"""
    expected = DataFrame(
        [["foo", "bar"], [np.nan, "baz"], [np.nan, np.nan]], columns=["A", "B"]
    )
    if parser.engine == "pyarrow":
        expected.loc[[1, 2], "A"] = None
        expected.loc[2, "B"] = None
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "na_values",
    [
        ["-999.0", "-999"],
        [-999, -999.0],
        [-999.0, -999],
        ["-999.0"],
        ["-999"],
        [-999.0],
        [-999],
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        """A,B
-999,1.2
2,-999
3,4.5
""",
        """A,B
-999,1.200
2,-999.000
3,4.500
""",
    ],
)
def test_non_string_na_values(all_parsers, data, na_values, request):
    # see gh-3611: with an odd float format, we can't match
    # the string "999.0" exactly but still need float matching
    parser = all_parsers
    expected = DataFrame([[np.nan, 1.2], [2.0, np.nan], [3.0, 4.5]], columns=["A", "B"])

    if parser.engine == "pyarrow" and not all(isinstance(x, str) for x in na_values):
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return
    elif parser.engine == "pyarrow" and "-999.000" in data:
        # bc the pyarrow engine does not include the float-ified version
        #  of "-999" -> -999, it does not match the entry with the trailing
        #  zeros, so "-999.000" is not treated as null.
        mark = pytest.mark.xfail(
            reason="pyarrow engined does not recognize equivalent floats"
        )
        request.applymarker(mark)

    result = parser.read_csv(StringIO(data), na_values=na_values)
    tm.assert_frame_equal(result, expected)


def test_default_na_values(all_parsers):
    _NA_VALUES = {
        "-1.#IND",
        "1.#QNAN",
        "1.#IND",
        "-1.#QNAN",
        "#N/A",
        "N/A",
        "n/a",
        "NA",
        "<NA>",
        "#NA",
        "NULL",
        "null",
        "NaN",
        "nan",
        "-NaN",
        "-nan",
        "#N/A N/A",
        "",
        "None",
    }
    assert _NA_VALUES == STR_NA_VALUES

    parser = all_parsers
    nv = len(_NA_VALUES)

    def f(i, v):
        if i == 0:
            buf = ""
        elif i > 0:
            buf = "".join([","] * i)

        buf = f"{buf}{v}"

        if i < nv - 1:
            joined = "".join([","] * (nv - i - 1))
            buf = f"{buf}{joined}"

        return buf

    data = StringIO("\n".join([f(i, v) for i, v in enumerate(_NA_VALUES)]))
    expected = DataFrame(np.nan, columns=range(nv), index=range(nv))

    result = parser.read_csv(data, header=None)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("na_values", ["baz", ["baz"]])
def test_custom_na_values(all_parsers, na_values):
    parser = all_parsers
    data = """A,B,C
ignore,this,row
1,NA,3
-1.#IND,5,baz
7,8,NaN
"""
    expected = DataFrame(
        [[1.0, np.nan, 3], [np.nan, 5, np.nan], [7, 8, np.nan]], columns=["A", "B", "C"]
    )
    if parser.engine == "pyarrow":
        msg = "skiprows argument must be an integer when using engine='pyarrow'"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
        return

    result = parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
    tm.assert_frame_equal(result, expected)


def test_bool_na_values(all_parsers):
    data = """A,B,C
True,False,True
NA,True,False
False,NA,True"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        {
            "A": np.array([True, np.nan, False], dtype=object),
            "B": np.array([False, True, np.nan], dtype=object),
            "C": [True, False, True],
        }
    )
    if parser.engine == "pyarrow":
        expected.loc[1, "A"] = None
        expected.loc[2, "B"] = None
    tm.assert_frame_equal(result, expected)


def test_na_value_dict(all_parsers):
    data = """A,B,C
foo,bar,NA
bar,foo,foo
foo,bar,NA
bar,foo,foo"""
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={"A": ["foo"], "B": ["bar"]})
        return

    df = parser.read_csv(StringIO(data), na_values={"A": ["foo"], "B": ["bar"]})
    expected = DataFrame(
        {
            "A": [np.nan, "bar", np.nan, "bar"],
            "B": [np.nan, "foo", np.nan, "foo"],
            "C": [np.nan, "foo", np.nan, "foo"],
        }
    )
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "index_col,expected",
    [
        (
            [0],
            DataFrame({"b": [np.nan], "c": [1], "d": [5]}, index=Index([0], name="a")),
        ),
        (
            [0, 2],
            DataFrame(
                {"b": [np.nan], "d": [5]},
                index=MultiIndex.from_tuples([(0, 1)], names=["a", "c"]),
            ),
        ),
        (
            ["a", "c"],
            DataFrame(
                {"b": [np.nan], "d": [5]},
                index=MultiIndex.from_tuples([(0, 1)], names=["a", "c"]),
            ),
        ),
    ],
)
def test_na_value_dict_multi_index(all_parsers, index_col, expected):
    data = """\
a,b,c,d
0,NA,1,5
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), na_values=set(), index_col=index_col)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {},
            DataFrame(
                {
                    "A": ["a", "b", np.nan, "d", "e", np.nan, "g"],
                    "B": [1, 2, 3, 4, 5, 6, 7],
                    "C": ["one", "two", "three", np.nan, "five", np.nan, "seven"],
                }
            ),
        ),
        (
            {"na_values": {"A": [], "C": []}, "keep_default_na": False},
            DataFrame(
                {
                    "A": ["a", "b", "", "d", "e", "nan", "g"],
                    "B": [1, 2, 3, 4, 5, 6, 7],
                    "C": ["one", "two", "three", "nan", "five", "", "seven"],
                }
            ),
        ),
        (
            {"na_values": ["a"], "keep_default_na": False},
            DataFrame(
                {
                    "A": [np.nan, "b", "", "d", "e", "nan", "g"],
                    "B": [1, 2, 3, 4, 5, 6, 7],
                    "C": ["one", "two", "three", "nan", "five", "", "seven"],
                }
            ),
        ),
        (
            {"na_values": {"A": [], "C": []}},
            DataFrame(
                {
                    "A": ["a", "b", np.nan, "d", "e", np.nan, "g"],
                    "B": [1, 2, 3, 4, 5, 6, 7],
                    "C": ["one", "two", "three", np.nan, "five", np.nan, "seven"],
                }
            ),
        ),
    ],
)
def test_na_values_keep_default(
    all_parsers, kwargs, expected, request, using_infer_string
):
    data = """\
A,B,C
a,1,one
b,2,two
,3,three
d,4,nan
e,5,five
nan,6,
g,7,seven
"""
    parser = all_parsers
    if parser.engine == "pyarrow":
        if "na_values" in kwargs and isinstance(kwargs["na_values"], dict):
            msg = "The pyarrow engine doesn't support passing a dict for na_values"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(StringIO(data), **kwargs)
            return
        if not using_infer_string or "na_values" in kwargs:
            mark = pytest.mark.xfail()
            request.applymarker(mark)

    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)


def test_no_na_values_no_keep_default(all_parsers):
    # see gh-4318: passing na_values=None and
    # keep_default_na=False yields 'None" as a na_value
    data = """\
A,B,C
a,1,None
b,2,two
,3,None
d,4,nan
e,5,five
nan,6,
g,7,seven
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), keep_default_na=False)

    expected = DataFrame(
        {
            "A": ["a", "b", "", "d", "e", "nan", "g"],
            "B": [1, 2, 3, 4, 5, 6, 7],
            "C": ["None", "two", "None", "nan", "five", "", "seven"],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_no_keep_default_na_dict_na_values(all_parsers):
    # see gh-19227
    data = "a,b\n,2"
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), na_values={"b": ["2"]}, keep_default_na=False
            )
        return

    result = parser.read_csv(
        StringIO(data), na_values={"b": ["2"]}, keep_default_na=False
    )
    expected = DataFrame({"a": [""], "b": [np.nan]})
    tm.assert_frame_equal(result, expected)


def test_no_keep_default_na_dict_na_scalar_values(all_parsers):
    # see gh-19227
    #
    # Scalar values shouldn't cause the parsing to crash or fail.
    data = "a,b\n1,2"
    parser = all_parsers

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values={"b": 2}, keep_default_na=False)
        return

    df = parser.read_csv(StringIO(data), na_values={"b": 2}, keep_default_na=False)
    expected = DataFrame({"a": [1], "b": [np.nan]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("col_zero_na_values", [113125, "113125"])
def test_no_keep_default_na_dict_na_values_diff_reprs(all_parsers, col_zero_na_values):
    # see gh-19227
    data = """\
113125,"blah","/blaha",kjsdkj,412.166,225.874,214.008
729639,"qwer","",asdfkj,466.681,,252.373
"""
    parser = all_parsers
    expected = DataFrame(
        {
            0: [np.nan, 729639.0],
            1: [np.nan, "qwer"],
            2: ["/blaha", np.nan],
            3: ["kjsdkj", "asdfkj"],
            4: [412.166, 466.681],
            5: ["225.874", ""],
            6: [np.nan, 252.373],
        }
    )

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                header=None,
                keep_default_na=False,
                na_values={2: "", 6: "214.008", 1: "blah", 0: col_zero_na_values},
            )
        return

    result = parser.read_csv(
        StringIO(data),
        header=None,
        keep_default_na=False,
        na_values={2: "", 6: "214.008", 1: "blah", 0: col_zero_na_values},
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "na_filter,row_data",
    [
        (True, [[1, "A"], [np.nan, np.nan], [3, "C"]]),
        (False, [["1", "A"], ["nan", "B"], ["3", "C"]]),
    ],
)
def test_na_values_na_filter_override(
    request, all_parsers, na_filter, row_data, using_infer_string
):
    parser = all_parsers
    if parser.engine == "pyarrow":
        # mismatched dtypes in both cases, FutureWarning in the True case
        if not (using_infer_string and na_filter):
            mark = pytest.mark.xfail(reason="pyarrow doesn't support this.")
            request.applymarker(mark)
    data = """\
A,B
1,A
nan,B
3,C
"""
    result = parser.read_csv(StringIO(data), na_values=["B"], na_filter=na_filter)

    expected = DataFrame(row_data, columns=["A", "B"])
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Expected 8 columns, got 5:
def test_na_trailing_columns(all_parsers):
    parser = all_parsers
    data = """Date,Currency,Symbol,Type,Units,UnitPrice,Cost,Tax
2012-03-14,USD,AAPL,BUY,1000
2012-05-12,USD,SBUX,SELL,500"""

    # Trailing columns should be all NaN.
    result = parser.read_csv(StringIO(data))
    expected = DataFrame(
        [
            ["2012-03-14", "USD", "AAPL", "BUY", 1000, np.nan, np.nan, np.nan],
            ["2012-05-12", "USD", "SBUX", "SELL", 500, np.nan, np.nan, np.nan],
        ],
        columns=[
            "Date",
            "Currency",
            "Symbol",
            "Type",
            "Units",
            "UnitPrice",
            "Cost",
            "Tax",
        ],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "na_values,row_data",
    [
        (1, [[np.nan, 2.0], [2.0, np.nan]]),
        ({"a": 2, "b": 1}, [[1.0, 2.0], [np.nan, np.nan]]),
    ],
)
def test_na_values_scalar(all_parsers, na_values, row_data):
    # see gh-12224
    parser = all_parsers
    names = ["a", "b"]
    data = "1,2\n2,1"

    if parser.engine == "pyarrow" and isinstance(na_values, dict):
        if isinstance(na_values, dict):
            err = ValueError
            msg = "The pyarrow engine doesn't support passing a dict for na_values"
        else:
            err = TypeError
            msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(err, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return
    elif parser.engine == "pyarrow":
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return

    result = parser.read_csv(StringIO(data), names=names, na_values=na_values)
    expected = DataFrame(row_data, columns=names)
    tm.assert_frame_equal(result, expected)


def test_na_values_dict_aliasing(all_parsers):
    parser = all_parsers
    na_values = {"a": 2, "b": 1}
    na_values_copy = na_values.copy()

    names = ["a", "b"]
    data = "1,2\n2,1"

    expected = DataFrame([[1.0, 2.0], [np.nan, np.nan]], columns=names)

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return

    result = parser.read_csv(StringIO(data), names=names, na_values=na_values)

    tm.assert_frame_equal(result, expected)
    tm.assert_dict_equal(na_values, na_values_copy)


def test_na_values_dict_col_index(all_parsers):
    # see gh-14203
    data = "a\nfoo\n1"
    parser = all_parsers
    na_values = {0: "foo"}

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return

    result = parser.read_csv(StringIO(data), na_values=na_values)
    expected = DataFrame({"a": [np.nan, 1]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            str(2**63) + "\n" + str(2**63 + 1),
            {"na_values": [2**63]},
            DataFrame([str(2**63), str(2**63 + 1)]),
        ),
        (str(2**63) + ",1" + "\n,2", {}, DataFrame([[str(2**63), 1], ["", 2]])),
        (str(2**63) + "\n1", {"na_values": [2**63]}, DataFrame([np.nan, 1])),
    ],
)
def test_na_values_uint64(all_parsers, data, kwargs, expected, request):
    # see gh-14983
    parser = all_parsers

    if parser.engine == "pyarrow" and "na_values" in kwargs:
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), header=None, **kwargs)
        return
    elif parser.engine == "pyarrow":
        mark = pytest.mark.xfail(reason="Returns float64 instead of object")
        request.applymarker(mark)

    result = parser.read_csv(StringIO(data), header=None, **kwargs)
    tm.assert_frame_equal(result, expected)


def test_empty_na_values_no_default_with_index(all_parsers):
    # see gh-15835
    data = "a,1\nb,2"
    parser = all_parsers
    expected = DataFrame({"1": [2]}, index=Index(["b"], name="a"))

    result = parser.read_csv(StringIO(data), index_col=0, keep_default_na=False)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "na_filter,index_data", [(False, ["", "5"]), (True, [np.nan, 5.0])]
)
def test_no_na_filter_on_index(all_parsers, na_filter, index_data, request):
    # see gh-5239
    #
    # Don't parse NA-values in index unless na_filter=True
    parser = all_parsers
    data = "a,b,c\n1,,3\n4,5,6"

    if parser.engine == "pyarrow" and na_filter is False:
        mark = pytest.mark.xfail(reason="mismatched index result")
        request.applymarker(mark)

    expected = DataFrame({"a": [1, 4], "c": [3, 6]}, index=Index(index_data, name="b"))
    result = parser.read_csv(StringIO(data), index_col=[1], na_filter=na_filter)
    tm.assert_frame_equal(result, expected)


def test_inf_na_values_with_int_index(all_parsers):
    # see gh-17128
    parser = all_parsers
    data = "idx,col1,col2\n1,3,4\n2,inf,-inf"

    # Don't fail with OverflowError with inf's and integer index column.
    out = parser.read_csv(StringIO(data), index_col=[0], na_values=["inf", "-inf"])
    expected = DataFrame(
        {"col1": [3, np.nan], "col2": [4, np.nan]}, index=Index([1, 2], name="idx")
    )
    tm.assert_frame_equal(out, expected)


@xfail_pyarrow  # mismatched shape
@pytest.mark.parametrize("na_filter", [True, False])
def test_na_values_with_dtype_str_and_na_filter(all_parsers, na_filter):
    # see gh-20377
    parser = all_parsers
    data = "a,b,c\n1,,3\n4,5,6"

    # na_filter=True --> missing value becomes NaN.
    # na_filter=False --> missing value remains empty string.
    empty = np.nan if na_filter else ""
    expected = DataFrame({"a": ["1", "4"], "b": [empty, "5"], "c": ["3", "6"]})

    result = parser.read_csv(StringIO(data), na_filter=na_filter, dtype=str)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # mismatched exception message
@pytest.mark.parametrize(
    "data, na_values",
    [
        ("false,1\n,1\ntrue", None),
        ("false,1\nnull,1\ntrue", None),
        ("false,1\nnan,1\ntrue", None),
        ("false,1\nfoo,1\ntrue", "foo"),
        ("false,1\nfoo,1\ntrue", ["foo"]),
        ("false,1\nfoo,1\ntrue", {"a": "foo"}),
    ],
)
def test_cast_NA_to_bool_raises_error(all_parsers, data, na_values):
    parser = all_parsers
    msg = "|".join(
        [
            "Bool column has NA values in column [0a]",
            "cannot safely convert passed user dtype of "
            "bool for object dtyped data in column 0",
        ]
    )

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(
            StringIO(data),
            header=None,
            names=["a", "b"],
            dtype={"a": "bool"},
            na_values=na_values,
        )


# TODO: this test isn't about the na_values keyword, it is about the empty entries
#  being returned with NaN entries, whereas the pyarrow engine returns "nan"
@xfail_pyarrow  # mismatched shapes
def test_str_nan_dropped(all_parsers):
    # see gh-21131
    parser = all_parsers

    data = """File: small.csv,,
10010010233,0123,654
foo,,bar
01001000155,4530,898"""

    result = parser.read_csv(
        StringIO(data),
        header=None,
        names=["col1", "col2", "col3"],
        dtype={"col1": str, "col2": str, "col3": str},
    ).dropna()

    expected = DataFrame(
        {
            "col1": ["10010010233", "01001000155"],
            "col2": ["0123", "4530"],
            "col3": ["654", "898"],
        },
        index=[1, 3],
    )

    tm.assert_frame_equal(result, expected)


def test_nan_multi_index(all_parsers):
    # GH 42446
    parser = all_parsers
    data = "A,B,B\nX,Y,Z\n1,2,inf"

    if parser.engine == "pyarrow":
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data), header=list(range(2)), na_values={("B", "Z"): "inf"}
            )
        return

    result = parser.read_csv(
        StringIO(data), header=list(range(2)), na_values={("B", "Z"): "inf"}
    )

    expected = DataFrame(
        {
            ("A", "X"): [1],
            ("B", "Y"): [2],
            ("B", "Z"): [np.nan],
        }
    )

    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # Failed: DID NOT RAISE <class 'ValueError'>; it casts the NaN to False
def test_bool_and_nan_to_bool(all_parsers):
    # GH#42808
    parser = all_parsers
    data = """0
NaN
True
False
"""
    with pytest.raises(ValueError, match="NA values"):
        parser.read_csv(StringIO(data), dtype="bool")


def test_bool_and_nan_to_int(all_parsers):
    # GH#42808
    parser = all_parsers
    data = """0
NaN
True
False
"""
    with pytest.raises(ValueError, match="convert|NoneType"):
        parser.read_csv(StringIO(data), dtype="int")


def test_bool_and_nan_to_float(all_parsers):
    # GH#42808
    parser = all_parsers
    data = """0
NaN
True
False
"""
    result = parser.read_csv(StringIO(data), dtype="float")
    expected = DataFrame.from_dict({"0": [np.nan, 1.0, 0.0]})
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\strings\test_strings.py ===
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import is_object_or_nan_string_dtype


@pytest.mark.parametrize("pattern", [0, True, Series(["foo", "bar"])])
def test_startswith_endswith_non_str_patterns(pattern):
    # GH3485
    ser = Series(["foo", "bar"])
    msg = f"expected a string or tuple, not {type(pattern).__name__}"
    with pytest.raises(TypeError, match=msg):
        ser.str.startswith(pattern)
    with pytest.raises(TypeError, match=msg):
        ser.str.endswith(pattern)


def test_iter_raises():
    # GH 54173
    ser = Series(["foo", "bar"])
    with pytest.raises(TypeError, match="'StringMethods' object is not iterable"):
        iter(ser.str)


# test integer/float dtypes (inferred by constructor) and mixed


def test_count(any_string_dtype):
    ser = Series(["foo", "foofoo", np.nan, "foooofooofommmfoo"], dtype=any_string_dtype)
    result = ser.str.count("f[o]+")
    expected_dtype = (
        np.float64 if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )
    expected = Series([1, 2, np.nan, 4], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_count_mixed_object():
    ser = Series(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=object,
    )
    result = ser.str.count("a")
    expected = Series([1, np.nan, 0, np.nan, np.nan, 0, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_repeat(any_string_dtype):
    ser = Series(["a", "b", np.nan, "c", np.nan, "d"], dtype=any_string_dtype)

    result = ser.str.repeat(3)
    expected = Series(
        ["aaa", "bbb", np.nan, "ccc", np.nan, "ddd"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.repeat([1, 2, 3, 4, 5, 6])
    expected = Series(
        ["a", "bb", np.nan, "cccc", np.nan, "dddddd"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)


def test_repeat_mixed_object():
    ser = Series(["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0])
    result = ser.str.repeat(3)
    expected = Series(
        ["aaa", np.nan, "bbb", np.nan, np.nan, "foofoofoo", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("arg, repeat", [[None, 4], ["b", None]])
def test_repeat_with_null(any_string_dtype, arg, repeat):
    # GH: 31632
    ser = Series(["a", arg], dtype=any_string_dtype)
    result = ser.str.repeat([3, repeat])
    expected = Series(["aaa", None], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_empty_str_methods(any_string_dtype):
    empty_str = empty = Series(dtype=any_string_dtype)
    empty_inferred_str = Series(dtype="str")
    if is_object_or_nan_string_dtype(any_string_dtype):
        empty_int = Series(dtype="int64")
        empty_bool = Series(dtype=bool)
    else:
        empty_int = Series(dtype="Int64")
        empty_bool = Series(dtype="boolean")
    empty_object = Series(dtype=object)
    empty_bytes = Series(dtype=object)
    empty_df = DataFrame()

    # GH7241
    # (extract) on empty series

    tm.assert_series_equal(empty_str, empty.str.cat(empty))
    assert "" == empty.str.cat()
    tm.assert_series_equal(empty_str, empty.str.title())
    tm.assert_series_equal(empty_int, empty.str.count("a"))
    tm.assert_series_equal(empty_bool, empty.str.contains("a"))
    tm.assert_series_equal(empty_bool, empty.str.startswith("a"))
    tm.assert_series_equal(empty_bool, empty.str.endswith("a"))
    tm.assert_series_equal(empty_str, empty.str.lower())
    tm.assert_series_equal(empty_str, empty.str.upper())
    tm.assert_series_equal(empty_str, empty.str.replace("a", "b"))
    tm.assert_series_equal(empty_str, empty.str.repeat(3))
    tm.assert_series_equal(empty_bool, empty.str.match("^a"))
    tm.assert_frame_equal(
        DataFrame(columns=[0], dtype=any_string_dtype),
        empty.str.extract("()", expand=True),
    )
    tm.assert_frame_equal(
        DataFrame(columns=[0, 1], dtype=any_string_dtype),
        empty.str.extract("()()", expand=True),
    )
    tm.assert_series_equal(empty_str, empty.str.extract("()", expand=False))
    tm.assert_frame_equal(
        DataFrame(columns=[0, 1], dtype=any_string_dtype),
        empty.str.extract("()()", expand=False),
    )
    tm.assert_frame_equal(empty_df.set_axis([], axis=1), empty.str.get_dummies())
    tm.assert_series_equal(empty_str, empty_str.str.join(""))
    tm.assert_series_equal(empty_int, empty.str.len())
    tm.assert_series_equal(empty_object, empty_str.str.findall("a"))
    tm.assert_series_equal(empty_int, empty.str.find("a"))
    tm.assert_series_equal(empty_int, empty.str.rfind("a"))
    tm.assert_series_equal(empty_str, empty.str.pad(42))
    tm.assert_series_equal(empty_str, empty.str.center(42))
    tm.assert_series_equal(empty_object, empty.str.split("a"))
    tm.assert_series_equal(empty_object, empty.str.rsplit("a"))
    tm.assert_series_equal(empty_object, empty.str.partition("a", expand=False))
    tm.assert_frame_equal(empty_df, empty.str.partition("a"))
    tm.assert_series_equal(empty_object, empty.str.rpartition("a", expand=False))
    tm.assert_frame_equal(empty_df, empty.str.rpartition("a"))
    tm.assert_series_equal(empty_str, empty.str.slice(stop=1))
    tm.assert_series_equal(empty_str, empty.str.slice(step=1))
    tm.assert_series_equal(empty_str, empty.str.strip())
    tm.assert_series_equal(empty_str, empty.str.lstrip())
    tm.assert_series_equal(empty_str, empty.str.rstrip())
    tm.assert_series_equal(empty_str, empty.str.wrap(42))
    tm.assert_series_equal(empty_str, empty.str.get(0))
    tm.assert_series_equal(empty_inferred_str, empty_bytes.str.decode("ascii"))
    tm.assert_series_equal(empty_bytes, empty.str.encode("ascii"))
    # ismethods should always return boolean (GH 29624)
    tm.assert_series_equal(empty_bool, empty.str.isalnum())
    tm.assert_series_equal(empty_bool, empty.str.isalpha())
    tm.assert_series_equal(empty_bool, empty.str.isdigit())
    tm.assert_series_equal(empty_bool, empty.str.isspace())
    tm.assert_series_equal(empty_bool, empty.str.islower())
    tm.assert_series_equal(empty_bool, empty.str.isupper())
    tm.assert_series_equal(empty_bool, empty.str.istitle())
    tm.assert_series_equal(empty_bool, empty.str.isnumeric())
    tm.assert_series_equal(empty_bool, empty.str.isdecimal())
    tm.assert_series_equal(empty_str, empty.str.capitalize())
    tm.assert_series_equal(empty_str, empty.str.swapcase())
    tm.assert_series_equal(empty_str, empty.str.normalize("NFC"))

    table = str.maketrans("a", "b")
    tm.assert_series_equal(empty_str, empty.str.translate(table))


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isalnum", [True, True, True, True, True, False, True, True, False, False]),
        ("isalpha", [True, True, True, False, False, False, True, False, False, False]),
        (
            "isdigit",
            [False, False, False, True, False, False, False, True, False, False],
        ),
        (
            "isnumeric",
            [False, False, False, True, False, False, False, True, False, False],
        ),
        (
            "isspace",
            [False, False, False, False, False, False, False, False, False, True],
        ),
        (
            "islower",
            [False, True, False, False, False, False, False, False, False, False],
        ),
        (
            "isupper",
            [True, False, False, False, True, False, True, False, False, False],
        ),
        (
            "istitle",
            [True, False, True, False, True, False, False, False, False, False],
        ),
    ],
)
def test_ismethods(method, expected, any_string_dtype):
    ser = Series(
        ["A", "b", "Xy", "4", "3A", "", "TT", "55", "-", "  "], dtype=any_string_dtype
    )
    expected_dtype = (
        "bool" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)

    # compare with standard library
    expected_stdlib = [getattr(item, method)() for item in ser]
    assert list(result) == expected_stdlib

    # with missing value
    ser.iloc[[1, 2, 3, 4]] = np.nan
    result = getattr(ser.str, method)()
    if ser.dtype == "object":
        expected = expected.astype(object)
        expected.iloc[[1, 2, 3, 4]] = np.nan
    elif ser.dtype == "str":
        # NaN propagates as False
        expected.iloc[[1, 2, 3, 4]] = False
    else:
        # nullable dtypes propagate NaN
        expected.iloc[[1, 2, 3, 4]] = np.nan


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, True, True, False, True, True, False]),
        ("isdecimal", [False, True, False, False, False, True, False]),
    ],
)
def test_isnumeric_unicode(method, expected, any_string_dtype):
    # 0x00bc: ¼ VULGAR FRACTION ONE QUARTER
    # 0x2605: ★ not number
    # 0x1378: ፸ ETHIOPIC NUMBER SEVENTY
    # 0xFF13: ３ Em 3  # noqa: RUF003
    ser = Series(
        ["A", "3", "¼", "★", "፸", "３", "four"], dtype=any_string_dtype  # noqa: RUF001
    )
    expected_dtype = (
        "bool" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)

    # compare with standard library
    expected = [getattr(item, method)() for item in ser]
    assert list(result) == expected


@pytest.mark.filterwarnings("ignore:Downcasting object dtype arrays:FutureWarning")
@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, np.nan, True, False, np.nan, True, False]),
        ("isdecimal", [False, np.nan, False, False, np.nan, True, False]),
    ],
)
def test_isnumeric_unicode_missing(method, expected, any_string_dtype):
    values = ["A", np.nan, "¼", "★", np.nan, "３", "four"]  # noqa: RUF001
    ser = Series(values, dtype=any_string_dtype)
    if any_string_dtype == "str":
        # NaN propagates as False
        expected = Series(expected, dtype=object).fillna(False).astype(bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)


def test_spilt_join_roundtrip(any_string_dtype):
    ser = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    result = ser.str.split("_").str.join("_")
    expected = ser.astype(object)
    tm.assert_series_equal(result, expected)


def test_spilt_join_roundtrip_mixed_object():
    ser = Series(
        ["a_b", np.nan, "asdf_cas_asdf", True, datetime.today(), "foo", None, 1, 2.0]
    )
    result = ser.str.split("_").str.join("_")
    expected = Series(
        ["a_b", np.nan, "asdf_cas_asdf", np.nan, np.nan, "foo", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


def test_len(any_string_dtype):
    ser = Series(
        ["foo", "fooo", "fooooo", np.nan, "fooooooo", "foo\n", "あ"],
        dtype=any_string_dtype,
    )
    result = ser.str.len()
    expected_dtype = (
        "float64" if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )
    expected = Series([3, 4, 6, np.nan, 8, 4, 1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_len_mixed():
    ser = Series(
        ["a_b", np.nan, "asdf_cas_asdf", True, datetime.today(), "foo", None, 1, 2.0]
    )
    result = ser.str.len()
    expected = Series([3, np.nan, 13, np.nan, np.nan, 3, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method,sub,start,end,expected",
    [
        ("index", "EF", None, None, [4, 3, 1, 0]),
        ("rindex", "EF", None, None, [4, 5, 7, 4]),
        ("index", "EF", 3, None, [4, 3, 7, 4]),
        ("rindex", "EF", 3, None, [4, 5, 7, 4]),
        ("index", "E", 4, 8, [4, 5, 7, 4]),
        ("rindex", "E", 0, 5, [4, 3, 1, 4]),
    ],
)
def test_index(method, sub, start, end, index_or_series, any_string_dtype, expected):
    obj = index_or_series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"], dtype=any_string_dtype
    )
    expected_dtype = (
        np.int64 if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )
    expected = index_or_series(expected, dtype=expected_dtype)

    result = getattr(obj.str, method)(sub, start, end)

    if index_or_series is Series:
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)

    # compare with standard library
    expected = [getattr(item, method)(sub, start, end) for item in obj]
    assert list(result) == expected


def test_index_not_found_raises(index_or_series, any_string_dtype):
    obj = index_or_series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"], dtype=any_string_dtype
    )
    with pytest.raises(ValueError, match="substring not found"):
        obj.str.index("DE")


@pytest.mark.parametrize("method", ["index", "rindex"])
def test_index_wrong_type_raises(index_or_series, any_string_dtype, method):
    obj = index_or_series([], dtype=any_string_dtype)
    msg = "expected a string object, not int"

    with pytest.raises(TypeError, match=msg):
        getattr(obj.str, method)(0)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["index", [1, 1, 0]],
        ["rindex", [3, 1, 2]],
    ],
)
def test_index_missing(any_string_dtype, method, exp):
    ser = Series(["abcb", "ab", "bcbe", np.nan], dtype=any_string_dtype)
    expected_dtype = (
        np.float64 if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )

    result = getattr(ser.str, method)("b")
    expected = Series(exp + [np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_pipe_failures(any_string_dtype):
    # #2119
    ser = Series(["A|B|C"], dtype=any_string_dtype)

    result = ser.str.split("|")
    expected = Series([["A", "B", "C"]], dtype=object)
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("|", " ", regex=False)
    expected = Series(["A B C"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (2, 5, None, ["foo", "bar", np.nan, "baz"]),
        (0, 3, -1, ["", "", np.nan, ""]),
        (None, None, -1, ["owtoofaa", "owtrabaa", np.nan, "xuqzabaa"]),
        (None, 2, -1, ["owtoo", "owtra", np.nan, "xuqza"]),
        (3, 10, 2, ["oto", "ato", np.nan, "aqx"]),
        (3, 0, -1, ["ofa", "aba", np.nan, "aba"]),
    ],
)
def test_slice(start, stop, step, expected, any_string_dtype):
    ser = Series(["aafootwo", "aabartwo", np.nan, "aabazqux"], dtype=any_string_dtype)
    result = ser.str.slice(start, stop, step)
    expected = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (2, 5, None, ["foo", np.nan, "bar", np.nan, np.nan, None, np.nan, np.nan]),
        (4, 1, -1, ["oof", np.nan, "rab", np.nan, np.nan, None, np.nan, np.nan]),
    ],
)
def test_slice_mixed_object(start, stop, step, expected):
    ser = Series(["aafootwo", np.nan, "aabartwo", True, datetime.today(), None, 1, 2.0])
    result = ser.str.slice(start, stop, step)
    expected = Series(expected, dtype=object)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start,stop,repl,expected",
    [
        (2, 3, None, ["shrt", "a it longer", "evnlongerthanthat", "", np.nan]),
        (2, 3, "z", ["shzrt", "a zit longer", "evznlongerthanthat", "z", np.nan]),
        (2, 2, "z", ["shzort", "a zbit longer", "evzenlongerthanthat", "z", np.nan]),
        (2, 1, "z", ["shzort", "a zbit longer", "evzenlongerthanthat", "z", np.nan]),
        (-1, None, "z", ["shorz", "a bit longez", "evenlongerthanthaz", "z", np.nan]),
        (None, -2, "z", ["zrt", "zer", "zat", "z", np.nan]),
        (6, 8, "z", ["shortz", "a bit znger", "evenlozerthanthat", "z", np.nan]),
        (-10, 3, "z", ["zrt", "a zit longer", "evenlongzerthanthat", "z", np.nan]),
    ],
)
def test_slice_replace(start, stop, repl, expected, any_string_dtype):
    ser = Series(
        ["short", "a bit longer", "evenlongerthanthat", "", np.nan],
        dtype=any_string_dtype,
    )
    expected = Series(expected, dtype=any_string_dtype)
    result = ser.str.slice_replace(start, stop, repl)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["aa", "bb", np.nan, "cc"]],
        ["lstrip", ["aa   ", "bb \n", np.nan, "cc  "]],
        ["rstrip", ["  aa", " bb", np.nan, "cc"]],
    ],
)
def test_strip_lstrip_rstrip(any_string_dtype, method, exp):
    ser = Series(["  aa   ", " bb \n", np.nan, "cc  "], dtype=any_string_dtype)

    result = getattr(ser.str, method)()
    expected = Series(exp, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["aa", np.nan, "bb"]],
        ["lstrip", ["aa  ", np.nan, "bb \t\n"]],
        ["rstrip", ["  aa", np.nan, " bb"]],
    ],
)
def test_strip_lstrip_rstrip_mixed_object(method, exp):
    ser = Series(["  aa  ", np.nan, " bb \t\n", True, datetime.today(), None, 1, 2.0])

    result = getattr(ser.str, method)()
    expected = Series(exp + [np.nan, np.nan, None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["ABC", " BNSD", "LDFJH "]],
        ["lstrip", ["ABCxx", " BNSD", "LDFJH xx"]],
        ["rstrip", ["xxABC", "xx BNSD", "LDFJH "]],
    ],
)
def test_strip_lstrip_rstrip_args(any_string_dtype, method, exp):
    ser = Series(["xxABCxx", "xx BNSD", "LDFJH xx"], dtype=any_string_dtype)

    result = getattr(ser.str, method)("x")
    expected = Series(exp, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "prefix, expected", [("a", ["b", " b c", "bc"]), ("ab", ["", "a b c", "bc"])]
)
def test_removeprefix(any_string_dtype, prefix, expected):
    ser = Series(["ab", "a b c", "bc"], dtype=any_string_dtype)
    result = ser.str.removeprefix(prefix)
    ser_expected = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, ser_expected)


@pytest.mark.parametrize(
    "suffix, expected", [("c", ["ab", "a b ", "b"]), ("bc", ["ab", "a b c", ""])]
)
def test_removesuffix(any_string_dtype, suffix, expected):
    ser = Series(["ab", "a b c", "bc"], dtype=any_string_dtype)
    result = ser.str.removesuffix(suffix)
    ser_expected = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, ser_expected)


def test_string_slice_get_syntax(any_string_dtype):
    ser = Series(
        ["YYY", "B", "C", "YYYYYYbYYY", "BYYYcYYY", np.nan, "CYYYBYYY", "dog", "cYYYt"],
        dtype=any_string_dtype,
    )

    result = ser.str[0]
    expected = ser.str.get(0)
    tm.assert_series_equal(result, expected)

    result = ser.str[:3]
    expected = ser.str.slice(stop=3)
    tm.assert_series_equal(result, expected)

    result = ser.str[2::-1]
    expected = ser.str.slice(start=2, step=-1)
    tm.assert_series_equal(result, expected)


def test_string_slice_out_of_bounds_nested():
    ser = Series([(1, 2), (1,), (3, 4, 5)])
    result = ser.str[1]
    expected = Series([2, np.nan, 4])
    tm.assert_series_equal(result, expected)


def test_string_slice_out_of_bounds(any_string_dtype):
    ser = Series(["foo", "b", "ba"], dtype=any_string_dtype)
    result = ser.str[1]
    expected = Series(["o", np.nan, "a"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_encode_decode(any_string_dtype):
    ser = Series(["a", "b", "a\xe4"], dtype=any_string_dtype).str.encode("utf-8")
    result = ser.str.decode("utf-8")
    expected = Series(["a", "b", "a\xe4"], dtype="str")
    tm.assert_series_equal(result, expected)


def test_encode_errors_kwarg(any_string_dtype):
    ser = Series(["a", "b", "a\x9d"], dtype=any_string_dtype)

    msg = (
        r"'charmap' codec can't encode character '\\x9d' in position 1: "
        "character maps to <undefined>"
    )
    with pytest.raises(UnicodeEncodeError, match=msg):
        ser.str.encode("cp1252")

    result = ser.str.encode("cp1252", "ignore")
    expected = ser.map(lambda x: x.encode("cp1252", "ignore"))
    tm.assert_series_equal(result, expected)


def test_decode_errors_kwarg():
    ser = Series([b"a", b"b", b"a\x9d"])

    msg = (
        "'charmap' codec can't decode byte 0x9d in position 1: "
        "character maps to <undefined>"
    )
    with pytest.raises(UnicodeDecodeError, match=msg):
        ser.str.decode("cp1252")

    result = ser.str.decode("cp1252", "ignore")
    expected = ser.map(lambda x: x.decode("cp1252", "ignore")).astype("str")
    tm.assert_series_equal(result, expected)


def test_decode_string_dtype(string_dtype):
    # https://github.com/pandas-dev/pandas/pull/60940
    ser = Series([b"a", b"b"])
    result = ser.str.decode("utf-8", dtype=string_dtype)
    expected = Series(["a", "b"], dtype=string_dtype)
    tm.assert_series_equal(result, expected)


def test_decode_object_dtype(object_dtype):
    # https://github.com/pandas-dev/pandas/pull/60940
    ser = Series([b"a", rb"\ud800"])
    result = ser.str.decode("utf-8", dtype=object_dtype)
    expected = Series(["a", r"\ud800"], dtype=object_dtype)
    tm.assert_series_equal(result, expected)


def test_decode_bad_dtype():
    # https://github.com/pandas-dev/pandas/pull/60940
    ser = Series([b"a", b"b"])
    msg = "dtype must be string or object, got dtype='int64'"
    with pytest.raises(ValueError, match=msg):
        ser.str.decode("utf-8", dtype="int64")


@pytest.mark.parametrize(
    "form, expected",
    [
        ("NFKC", ["ABC", "ABC", "123", np.nan, "アイエ"]),
        ("NFC", ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"]),  # noqa: RUF001
    ],
)
def test_normalize(form, expected, any_string_dtype):
    ser = Series(
        ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"],  # noqa: RUF001
        index=["a", "b", "c", "d", "e"],
        dtype=any_string_dtype,
    )
    expected = Series(expected, index=["a", "b", "c", "d", "e"], dtype=any_string_dtype)
    result = ser.str.normalize(form)
    tm.assert_series_equal(result, expected)


def test_normalize_bad_arg_raises(any_string_dtype):
    ser = Series(
        ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"],  # noqa: RUF001
        index=["a", "b", "c", "d", "e"],
        dtype=any_string_dtype,
    )
    with pytest.raises(ValueError, match="invalid normalization form"):
        ser.str.normalize("xxx")


def test_normalize_index():
    idx = Index(["ＡＢＣ", "１２３", "ｱｲｴ"])  # noqa: RUF001
    expected = Index(["ABC", "123", "アイエ"])
    result = idx.str.normalize("NFKC")
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "values,inferred_type",
    [
        (["a", "b"], "string"),
        (["a", "b", 1], "mixed-integer"),
        (["a", "b", 1.3], "mixed"),
        (["a", "b", 1.3, 1], "mixed-integer"),
        (["aa", datetime(2011, 1, 1)], "mixed"),
    ],
)
def test_index_str_accessor_visibility(values, inferred_type, index_or_series):
    obj = index_or_series(values)
    if index_or_series is Index:
        assert obj.inferred_type == inferred_type

    assert isinstance(obj.str, StringMethods)


@pytest.mark.parametrize(
    "values,inferred_type",
    [
        ([1, np.nan], "floating"),
        ([datetime(2011, 1, 1)], "datetime64"),
        ([timedelta(1)], "timedelta64"),
    ],
)
def test_index_str_accessor_non_string_values_raises(
    values, inferred_type, index_or_series
):
    obj = index_or_series(values)
    if index_or_series is Index:
        assert obj.inferred_type == inferred_type

    msg = "Can only use .str accessor with string values"
    with pytest.raises(AttributeError, match=msg):
        obj.str


def test_index_str_accessor_multiindex_raises():
    # MultiIndex has mixed dtype, but not allow to use accessor
    idx = MultiIndex.from_tuples([("a", "b"), ("a", "b")])
    assert idx.inferred_type == "mixed"

    msg = "Can only use .str accessor with Index, not MultiIndex"
    with pytest.raises(AttributeError, match=msg):
        idx.str


def test_str_accessor_no_new_attributes(any_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/10673
    ser = Series(list("aabbcde"), dtype=any_string_dtype)
    with pytest.raises(AttributeError, match="You cannot add any new attribute"):
        ser.str.xlabel = "a"


def test_cat_on_bytes_raises():
    lhs = Series(np.array(list("abc"), "S1").astype(object))
    rhs = Series(np.array(list("def"), "S1").astype(object))
    msg = "Cannot use .str.cat with values of inferred dtype 'bytes'"
    with pytest.raises(TypeError, match=msg):
        lhs.str.cat(rhs)


def test_str_accessor_in_apply_func():
    # https://github.com/pandas-dev/pandas/issues/38979
    df = DataFrame(zip("abc", "def"))
    expected = Series(["A/D", "B/E", "C/F"])
    result = df.apply(lambda f: "/".join(f.str.upper()), axis=1)
    tm.assert_series_equal(result, expected)


def test_zfill():
    # https://github.com/pandas-dev/pandas/issues/20868
    value = Series(["-1", "1", "1000", 10, np.nan])
    expected = Series(["-01", "001", "1000", np.nan, np.nan], dtype=object)
    tm.assert_series_equal(value.str.zfill(3), expected)

    value = Series(["-2", "+5"])
    expected = Series(["-0002", "+0005"])
    tm.assert_series_equal(value.str.zfill(5), expected)


def test_zfill_with_non_integer_argument():
    value = Series(["-2", "+5"])
    wid = "a"
    msg = f"width must be of integer type, not {type(wid).__name__}"
    with pytest.raises(TypeError, match=msg):
        value.str.zfill(wid)


def test_zfill_with_leading_sign():
    value = Series(["-cat", "-1", "+dog"])
    expected = Series(["-0cat", "-0001", "+0dog"])
    tm.assert_series_equal(value.str.zfill(5), expected)


def test_get_with_dict_label():
    # GH47911
    s = Series(
        [
            {"name": "Hello", "value": "World"},
            {"name": "Goodbye", "value": "Planet"},
            {"value": "Sea"},
        ]
    )
    result = s.str.get("name")
    expected = Series(["Hello", "Goodbye", None], dtype=object)
    tm.assert_series_equal(result, expected)
    result = s.str.get("value")
    expected = Series(["World", "Planet", "Sea"], dtype=object)
    tm.assert_series_equal(result, expected)


def test_series_str_decode():
    # GH 22613
    result = Series([b"x", b"y"]).str.decode(encoding="UTF-8", errors="strict")
    expected = Series(["x", "y"], dtype="str")
    tm.assert_series_equal(result, expected)