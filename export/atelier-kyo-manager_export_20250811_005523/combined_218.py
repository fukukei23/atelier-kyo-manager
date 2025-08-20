
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_gate.py ===
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.symbol import (Wild, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices import Matrix, ImmutableMatrix

from sympy.physics.quantum.gate import (XGate, YGate, ZGate, random_circuit,
        CNOT, IdentityGate, H, X, Y, S, T, Z, SwapGate, gate_simp, gate_sort,
        CNotGate, TGate, HadamardGate, PhaseGate, UGate, CGate)
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import Qubit, IntQubit, qubit_to_matrix, \
    matrix_to_qubit
from sympy.physics.quantum.matrixutils import matrix_to_zero
from sympy.physics.quantum.matrixcache import sqrt2_inv
from sympy.physics.quantum import Dagger


def test_gate():
    """Test a basic gate."""
    h = HadamardGate(1)
    assert h.min_qubits == 2
    assert h.nqubits == 1

    i0 = Wild('i0')
    i1 = Wild('i1')
    h0_w1 = HadamardGate(i0)
    h0_w2 = HadamardGate(i0)
    h1_w1 = HadamardGate(i1)

    assert h0_w1 == h0_w2
    assert h0_w1 != h1_w1
    assert h1_w1 != h0_w2

    cnot_10_w1 = CNOT(i1, i0)
    cnot_10_w2 = CNOT(i1, i0)
    cnot_01_w1 = CNOT(i0, i1)

    assert cnot_10_w1 == cnot_10_w2
    assert cnot_10_w1 != cnot_01_w1
    assert cnot_10_w2 != cnot_01_w1


def test_UGate():
    a, b, c, d = symbols('a,b,c,d')
    uMat = Matrix([[a, b], [c, d]])

    # Test basic case where gate exists in 1-qubit space
    u1 = UGate((0,), uMat)
    assert represent(u1, nqubits=1) == uMat
    assert qapply(u1*Qubit('0')) == a*Qubit('0') + c*Qubit('1')
    assert qapply(u1*Qubit('1')) == b*Qubit('0') + d*Qubit('1')

    # Test case where gate exists in a larger space
    u2 = UGate((1,), uMat)
    u2Rep = represent(u2, nqubits=2)
    for i in range(4):
        assert u2Rep*qubit_to_matrix(IntQubit(i, 2)) == \
            qubit_to_matrix(qapply(u2*IntQubit(i, 2)))


def test_cgate():
    """Test the general CGate."""
    # Test single control functionality
    CNOTMatrix = Matrix(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert represent(CGate(1, XGate(0)), nqubits=2) == CNOTMatrix

    # Test multiple control bit functionality
    ToffoliGate = CGate((1, 2), XGate(0))
    assert represent(ToffoliGate, nqubits=3) == \
        Matrix(
            [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0,
        1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]])

    ToffoliGate = CGate((3, 0), XGate(1))
    assert qapply(ToffoliGate*Qubit('1001')) == \
        matrix_to_qubit(represent(ToffoliGate*Qubit('1001'), nqubits=4))
    assert qapply(ToffoliGate*Qubit('0000')) == \
        matrix_to_qubit(represent(ToffoliGate*Qubit('0000'), nqubits=4))

    CYGate = CGate(1, YGate(0))
    CYGate_matrix = Matrix(
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, -I), (0, 0, I, 0)))
    # Test 2 qubit controlled-Y gate decompose method.
    assert represent(CYGate.decompose(), nqubits=2) == CYGate_matrix

    CZGate = CGate(0, ZGate(1))
    CZGate_matrix = Matrix(
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, -1)))
    assert qapply(CZGate*Qubit('11')) == -Qubit('11')
    assert matrix_to_qubit(represent(CZGate*Qubit('11'), nqubits=2)) == \
        -Qubit('11')
    # Test 2 qubit controlled-Z gate decompose method.
    assert represent(CZGate.decompose(), nqubits=2) == CZGate_matrix

    CPhaseGate = CGate(0, PhaseGate(1))
    assert qapply(CPhaseGate*Qubit('11')) == \
        I*Qubit('11')
    assert matrix_to_qubit(represent(CPhaseGate*Qubit('11'), nqubits=2)) == \
        I*Qubit('11')

    # Test that the dagger, inverse, and power of CGate is evaluated properly
    assert Dagger(CZGate) == CZGate
    assert pow(CZGate, 1) == Dagger(CZGate)
    assert Dagger(CZGate) == CZGate.inverse()
    assert Dagger(CPhaseGate) != CPhaseGate
    assert Dagger(CPhaseGate) == CPhaseGate.inverse()
    assert Dagger(CPhaseGate) == pow(CPhaseGate, -1)
    assert pow(CPhaseGate, -1) == CPhaseGate.inverse()


def test_UGate_CGate_combo():
    a, b, c, d = symbols('a,b,c,d')
    uMat = Matrix([[a, b], [c, d]])
    cMat = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, a, b], [0, 0, c, d]])

    # Test basic case where gate exists in 1-qubit space.
    u1 = UGate((0,), uMat)
    cu1 = CGate(1, u1)
    assert represent(cu1, nqubits=2) == cMat
    assert qapply(cu1*Qubit('10')) == a*Qubit('10') + c*Qubit('11')
    assert qapply(cu1*Qubit('11')) == b*Qubit('10') + d*Qubit('11')
    assert qapply(cu1*Qubit('01')) == Qubit('01')
    assert qapply(cu1*Qubit('00')) == Qubit('00')

    # Test case where gate exists in a larger space.
    u2 = UGate((1,), uMat)
    u2Rep = represent(u2, nqubits=2)
    for i in range(4):
        assert u2Rep*qubit_to_matrix(IntQubit(i, 2)) == \
            qubit_to_matrix(qapply(u2*IntQubit(i, 2)))

def test_UGate_OneQubitGate_combo():
    v, w, f, g = symbols('v w f g')
    uMat1 = ImmutableMatrix([[v, w], [f, g]])
    cMat1 = Matrix([[v, w + 1, 0, 0], [f + 1, g, 0, 0], [0, 0, v, w + 1], [0, 0, f + 1, g]])
    u1 = X(0) + UGate(0, uMat1)
    assert represent(u1, nqubits=2) == cMat1

    uMat2 = ImmutableMatrix([[1/sqrt(2), 1/sqrt(2)], [I/sqrt(2), -I/sqrt(2)]])
    cMat2_1 = Matrix([[Rational(1, 2) + I/2, Rational(1, 2) - I/2],
                      [Rational(1, 2) - I/2, Rational(1, 2) + I/2]])
    cMat2_2 = Matrix([[1, 0], [0, I]])
    u2 = UGate(0, uMat2)
    assert represent(H(0)*u2, nqubits=1) == cMat2_1
    assert represent(u2*H(0), nqubits=1) == cMat2_2

def test_represent_hadamard():
    """Test the representation of the hadamard gate."""
    circuit = HadamardGate(0)*Qubit('00')
    answer = represent(circuit, nqubits=2)
    # Check that the answers are same to within an epsilon.
    assert answer == Matrix([sqrt2_inv, sqrt2_inv, 0, 0])


def test_represent_xgate():
    """Test the representation of the X gate."""
    circuit = XGate(0)*Qubit('00')
    answer = represent(circuit, nqubits=2)
    assert Matrix([0, 1, 0, 0]) == answer


def test_represent_ygate():
    """Test the representation of the Y gate."""
    circuit = YGate(0)*Qubit('00')
    answer = represent(circuit, nqubits=2)
    assert answer[0] == 0 and answer[1] == I and \
        answer[2] == 0 and answer[3] == 0


def test_represent_zgate():
    """Test the representation of the Z gate."""
    circuit = ZGate(0)*Qubit('00')
    answer = represent(circuit, nqubits=2)
    assert Matrix([1, 0, 0, 0]) == answer


def test_represent_phasegate():
    """Test the representation of the S gate."""
    circuit = PhaseGate(0)*Qubit('01')
    answer = represent(circuit, nqubits=2)
    assert Matrix([0, I, 0, 0]) == answer


def test_represent_tgate():
    """Test the representation of the T gate."""
    circuit = TGate(0)*Qubit('01')
    assert Matrix([0, exp(I*pi/4), 0, 0]) == represent(circuit, nqubits=2)


def test_compound_gates():
    """Test a compound gate representation."""
    circuit = YGate(0)*ZGate(0)*XGate(0)*HadamardGate(0)*Qubit('00')
    answer = represent(circuit, nqubits=2)
    assert Matrix([I/sqrt(2), I/sqrt(2), 0, 0]) == answer


def test_cnot_gate():
    """Test the CNOT gate."""
    circuit = CNotGate(1, 0)
    assert represent(circuit, nqubits=2) == \
        Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    circuit = circuit*Qubit('111')
    assert matrix_to_qubit(represent(circuit, nqubits=3)) == \
        qapply(circuit)

    circuit = CNotGate(1, 0)
    assert Dagger(circuit) == circuit
    assert Dagger(Dagger(circuit)) == circuit
    assert circuit*circuit == 1


def test_gate_sort():
    """Test gate_sort."""
    for g in (X, Y, Z, H, S, T):
        assert gate_sort(g(2)*g(1)*g(0)) == g(0)*g(1)*g(2)
    e = gate_sort(X(1)*H(0)**2*CNOT(0, 1)*X(1)*X(0))
    assert e == H(0)**2*CNOT(0, 1)*X(0)*X(1)**2
    assert gate_sort(Z(0)*X(0)) == -X(0)*Z(0)
    assert gate_sort(Z(0)*X(0)**2) == X(0)**2*Z(0)
    assert gate_sort(Y(0)*H(0)) == -H(0)*Y(0)
    assert gate_sort(Y(0)*X(0)) == -X(0)*Y(0)
    assert gate_sort(Z(0)*Y(0)) == -Y(0)*Z(0)
    assert gate_sort(T(0)*S(0)) == S(0)*T(0)
    assert gate_sort(Z(0)*S(0)) == S(0)*Z(0)
    assert gate_sort(Z(0)*T(0)) == T(0)*Z(0)
    assert gate_sort(Z(0)*CNOT(0, 1)) == CNOT(0, 1)*Z(0)
    assert gate_sort(S(0)*CNOT(0, 1)) == CNOT(0, 1)*S(0)
    assert gate_sort(T(0)*CNOT(0, 1)) == CNOT(0, 1)*T(0)
    assert gate_sort(X(1)*CNOT(0, 1)) == CNOT(0, 1)*X(1)
    # This takes a long time and should only be uncommented once in a while.
    # nqubits = 5
    # ngates = 10
    # trials = 10
    # for i in range(trials):
    #     c = random_circuit(ngates, nqubits)
    #     assert represent(c, nqubits=nqubits) == \
    #            represent(gate_sort(c), nqubits=nqubits)


def test_gate_simp():
    """Test gate_simp."""
    e = H(0)*X(1)*H(0)**2*CNOT(0, 1)*X(1)**3*X(0)*Z(3)**2*S(4)**3
    assert gate_simp(e) == H(0)*CNOT(0, 1)*S(4)*X(0)*Z(4)
    assert gate_simp(X(0)*X(0)) == 1
    assert gate_simp(Y(0)*Y(0)) == 1
    assert gate_simp(Z(0)*Z(0)) == 1
    assert gate_simp(H(0)*H(0)) == 1
    assert gate_simp(T(0)*T(0)) == S(0)
    assert gate_simp(S(0)*S(0)) == Z(0)
    assert gate_simp(Integer(1)) == Integer(1)
    assert gate_simp(X(0)**2 + Y(0)**2) == Integer(2)


def test_swap_gate():
    """Test the SWAP gate."""
    swap_gate_matrix = Matrix(
        ((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    assert represent(SwapGate(1, 0).decompose(), nqubits=2) == swap_gate_matrix
    assert qapply(SwapGate(1, 3)*Qubit('0010')) == Qubit('1000')
    nqubits = 4
    for i in range(nqubits):
        for j in range(i):
            assert represent(SwapGate(i, j), nqubits=nqubits) == \
                represent(SwapGate(i, j).decompose(), nqubits=nqubits)


def test_one_qubit_commutators():
    """Test single qubit gate commutation relations."""
    for g1 in (IdentityGate, X, Y, Z, H, T, S):
        for g2 in (IdentityGate, X, Y, Z, H, T, S):
            e = Commutator(g1(0), g2(0))
            a = matrix_to_zero(represent(e, nqubits=1, format='sympy'))
            b = matrix_to_zero(represent(e.doit(), nqubits=1, format='sympy'))
            assert a == b

            e = Commutator(g1(0), g2(1))
            assert e.doit() == 0


def test_one_qubit_anticommutators():
    """Test single qubit gate anticommutation relations."""
    for g1 in (IdentityGate, X, Y, Z, H):
        for g2 in (IdentityGate, X, Y, Z, H):
            e = AntiCommutator(g1(0), g2(0))
            a = matrix_to_zero(represent(e, nqubits=1, format='sympy'))
            b = matrix_to_zero(represent(e.doit(), nqubits=1, format='sympy'))
            assert a == b
            e = AntiCommutator(g1(0), g2(1))
            a = matrix_to_zero(represent(e, nqubits=2, format='sympy'))
            b = matrix_to_zero(represent(e.doit(), nqubits=2, format='sympy'))
            assert a == b


def test_cnot_commutators():
    """Test commutators of involving CNOT gates."""
    assert Commutator(CNOT(0, 1), Z(0)).doit() == 0
    assert Commutator(CNOT(0, 1), T(0)).doit() == 0
    assert Commutator(CNOT(0, 1), S(0)).doit() == 0
    assert Commutator(CNOT(0, 1), X(1)).doit() == 0
    assert Commutator(CNOT(0, 1), CNOT(0, 1)).doit() == 0
    assert Commutator(CNOT(0, 1), CNOT(0, 2)).doit() == 0
    assert Commutator(CNOT(0, 2), CNOT(0, 1)).doit() == 0
    assert Commutator(CNOT(1, 2), CNOT(1, 0)).doit() == 0


def test_random_circuit():
    c = random_circuit(10, 3)
    assert isinstance(c, Mul)
    m = represent(c, nqubits=3)
    assert m.shape == (8, 8)
    assert isinstance(m, Matrix)


def test_hermitian_XGate():
    x = XGate(1, 2)
    x_dagger = Dagger(x)

    assert (x == x_dagger)


def test_hermitian_YGate():
    y = YGate(1, 2)
    y_dagger = Dagger(y)

    assert (y == y_dagger)


def test_hermitian_ZGate():
    z = ZGate(1, 2)
    z_dagger = Dagger(z)

    assert (z == z_dagger)


def test_unitary_XGate():
    x = XGate(1, 2)
    x_dagger = Dagger(x)

    assert (x*x_dagger == 1)


def test_unitary_YGate():
    y = YGate(1, 2)
    y_dagger = Dagger(y)

    assert (y*y_dagger == 1)


def test_unitary_ZGate():
    z = ZGate(1, 2)
    z_dagger = Dagger(z)

    assert (z*z_dagger == 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_bar.py ===
import io

import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    read_csv,
)

pytest.importorskip("jinja2")


def bar_grad(a=None, b=None, c=None, d=None):
    """Used in multiple tests to simplify formatting of expected result"""
    ret = [("width", "10em")]
    if all(x is None for x in [a, b, c, d]):
        return ret
    return ret + [
        (
            "background",
            f"linear-gradient(90deg,{','.join([x for x in [a, b, c, d] if x])})",
        )
    ]


def no_bar():
    return bar_grad()


def bar_to(x, color="#d65f5f"):
    return bar_grad(f" {color} {x:.1f}%", f" transparent {x:.1f}%")


def bar_from_to(x, y, color="#d65f5f"):
    return bar_grad(
        f" transparent {x:.1f}%",
        f" {color} {x:.1f}%",
        f" {color} {y:.1f}%",
        f" transparent {y:.1f}%",
    )


@pytest.fixture
def df_pos():
    return DataFrame([[1], [2], [3]])


@pytest.fixture
def df_neg():
    return DataFrame([[-1], [-2], [-3]])


@pytest.fixture
def df_mix():
    return DataFrame([[-3], [1], [2]])


@pytest.mark.parametrize(
    "align, exp",
    [
        ("left", [no_bar(), bar_to(50), bar_to(100)]),
        ("right", [bar_to(100), bar_from_to(50, 100), no_bar()]),
        ("mid", [bar_to(33.33), bar_to(66.66), bar_to(100)]),
        ("zero", [bar_from_to(50, 66.7), bar_from_to(50, 83.3), bar_from_to(50, 100)]),
        ("mean", [bar_to(50), no_bar(), bar_from_to(50, 100)]),
        (2.0, [bar_to(50), no_bar(), bar_from_to(50, 100)]),
        (np.median, [bar_to(50), no_bar(), bar_from_to(50, 100)]),
    ],
)
def test_align_positive_cases(df_pos, align, exp):
    # test different align cases for all positive values
    result = df_pos.style.bar(align=align)._compute().ctx
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    assert result == expected


@pytest.mark.parametrize(
    "align, exp",
    [
        ("left", [bar_to(100), bar_to(50), no_bar()]),
        ("right", [no_bar(), bar_from_to(50, 100), bar_to(100)]),
        ("mid", [bar_from_to(66.66, 100), bar_from_to(33.33, 100), bar_to(100)]),
        ("zero", [bar_from_to(33.33, 50), bar_from_to(16.66, 50), bar_to(50)]),
        ("mean", [bar_from_to(50, 100), no_bar(), bar_to(50)]),
        (-2.0, [bar_from_to(50, 100), no_bar(), bar_to(50)]),
        (np.median, [bar_from_to(50, 100), no_bar(), bar_to(50)]),
    ],
)
def test_align_negative_cases(df_neg, align, exp):
    # test different align cases for all negative values
    result = df_neg.style.bar(align=align)._compute().ctx
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    assert result == expected


@pytest.mark.parametrize(
    "align, exp",
    [
        ("left", [no_bar(), bar_to(80), bar_to(100)]),
        ("right", [bar_to(100), bar_from_to(80, 100), no_bar()]),
        ("mid", [bar_to(60), bar_from_to(60, 80), bar_from_to(60, 100)]),
        ("zero", [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]),
        ("mean", [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]),
        (-0.0, [bar_to(50), bar_from_to(50, 66.66), bar_from_to(50, 83.33)]),
        (np.nanmedian, [bar_to(50), no_bar(), bar_from_to(50, 62.5)]),
    ],
)
@pytest.mark.parametrize("nans", [True, False])
def test_align_mixed_cases(df_mix, align, exp, nans):
    # test different align cases for mixed positive and negative values
    # also test no impact of NaNs and no_bar
    expected = {(0, 0): exp[0], (1, 0): exp[1], (2, 0): exp[2]}
    if nans:
        df_mix.loc[3, :] = np.nan
        expected.update({(3, 0): no_bar()})
    result = df_mix.style.bar(align=align)._compute().ctx
    assert result == expected


@pytest.mark.parametrize(
    "align, exp",
    [
        (
            "left",
            {
                "index": [[no_bar(), no_bar()], [bar_to(100), bar_to(100)]],
                "columns": [[no_bar(), bar_to(100)], [no_bar(), bar_to(100)]],
                "none": [[no_bar(), bar_to(33.33)], [bar_to(66.66), bar_to(100)]],
            },
        ),
        (
            "mid",
            {
                "index": [[bar_to(33.33), bar_to(50)], [bar_to(100), bar_to(100)]],
                "columns": [[bar_to(50), bar_to(100)], [bar_to(75), bar_to(100)]],
                "none": [[bar_to(25), bar_to(50)], [bar_to(75), bar_to(100)]],
            },
        ),
        (
            "zero",
            {
                "index": [
                    [bar_from_to(50, 66.66), bar_from_to(50, 75)],
                    [bar_from_to(50, 100), bar_from_to(50, 100)],
                ],
                "columns": [
                    [bar_from_to(50, 75), bar_from_to(50, 100)],
                    [bar_from_to(50, 87.5), bar_from_to(50, 100)],
                ],
                "none": [
                    [bar_from_to(50, 62.5), bar_from_to(50, 75)],
                    [bar_from_to(50, 87.5), bar_from_to(50, 100)],
                ],
            },
        ),
        (
            2,
            {
                "index": [
                    [bar_to(50), no_bar()],
                    [bar_from_to(50, 100), bar_from_to(50, 100)],
                ],
                "columns": [
                    [bar_to(50), no_bar()],
                    [bar_from_to(50, 75), bar_from_to(50, 100)],
                ],
                "none": [
                    [bar_from_to(25, 50), no_bar()],
                    [bar_from_to(50, 75), bar_from_to(50, 100)],
                ],
            },
        ),
    ],
)
@pytest.mark.parametrize("axis", ["index", "columns", "none"])
def test_align_axis(align, exp, axis):
    # test all axis combinations with positive values and different aligns
    data = DataFrame([[1, 2], [3, 4]])
    result = (
        data.style.bar(align=align, axis=None if axis == "none" else axis)
        ._compute()
        .ctx
    )
    expected = {
        (0, 0): exp[axis][0][0],
        (0, 1): exp[axis][0][1],
        (1, 0): exp[axis][1][0],
        (1, 1): exp[axis][1][1],
    }
    assert result == expected


@pytest.mark.parametrize(
    "values, vmin, vmax",
    [
        ("positive", 1.5, 2.5),
        ("negative", -2.5, -1.5),
        ("mixed", -2.5, 1.5),
    ],
)
@pytest.mark.parametrize("nullify", [None, "vmin", "vmax"])  # test min/max separately
@pytest.mark.parametrize("align", ["left", "right", "zero", "mid"])
def test_vmin_vmax_clipping(df_pos, df_neg, df_mix, values, vmin, vmax, nullify, align):
    # test that clipping occurs if any vmin > data_values or vmax < data_values
    if align == "mid":  # mid acts as left or right in each case
        if values == "positive":
            align = "left"
        elif values == "negative":
            align = "right"
    df = {"positive": df_pos, "negative": df_neg, "mixed": df_mix}[values]
    vmin = None if nullify == "vmin" else vmin
    vmax = None if nullify == "vmax" else vmax

    clip_df = df.where(df <= (vmax if vmax else 999), other=vmax)
    clip_df = clip_df.where(clip_df >= (vmin if vmin else -999), other=vmin)

    result = (
        df.style.bar(align=align, vmin=vmin, vmax=vmax, color=["red", "green"])
        ._compute()
        .ctx
    )
    expected = clip_df.style.bar(align=align, color=["red", "green"])._compute().ctx
    assert result == expected


@pytest.mark.parametrize(
    "values, vmin, vmax",
    [
        ("positive", 0.5, 4.5),
        ("negative", -4.5, -0.5),
        ("mixed", -4.5, 4.5),
    ],
)
@pytest.mark.parametrize("nullify", [None, "vmin", "vmax"])  # test min/max separately
@pytest.mark.parametrize("align", ["left", "right", "zero", "mid"])
def test_vmin_vmax_widening(df_pos, df_neg, df_mix, values, vmin, vmax, nullify, align):
    # test that widening occurs if any vmax > data_values or vmin < data_values
    if align == "mid":  # mid acts as left or right in each case
        if values == "positive":
            align = "left"
        elif values == "negative":
            align = "right"
    df = {"positive": df_pos, "negative": df_neg, "mixed": df_mix}[values]
    vmin = None if nullify == "vmin" else vmin
    vmax = None if nullify == "vmax" else vmax

    expand_df = df.copy()
    expand_df.loc[3, :], expand_df.loc[4, :] = vmin, vmax

    result = (
        df.style.bar(align=align, vmin=vmin, vmax=vmax, color=["red", "green"])
        ._compute()
        .ctx
    )
    expected = expand_df.style.bar(align=align, color=["red", "green"])._compute().ctx
    assert result.items() <= expected.items()


def test_numerics():
    # test data is pre-selected for numeric values
    data = DataFrame([[1, "a"], [2, "b"]])
    result = data.style.bar()._compute().ctx
    assert (0, 1) not in result
    assert (1, 1) not in result


@pytest.mark.parametrize(
    "align, exp",
    [
        ("left", [no_bar(), bar_to(100, "green")]),
        ("right", [bar_to(100, "red"), no_bar()]),
        ("mid", [bar_to(25, "red"), bar_from_to(25, 100, "green")]),
        ("zero", [bar_from_to(33.33, 50, "red"), bar_from_to(50, 100, "green")]),
    ],
)
def test_colors_mixed(align, exp):
    data = DataFrame([[-1], [3]])
    result = data.style.bar(align=align, color=["red", "green"])._compute().ctx
    assert result == {(0, 0): exp[0], (1, 0): exp[1]}


def test_bar_align_height():
    # test when keyword height is used 'no-repeat center' and 'background-size' present
    data = DataFrame([[1], [2]])
    result = data.style.bar(align="left", height=50)._compute().ctx
    bg_s = "linear-gradient(90deg, #d65f5f 100.0%, transparent 100.0%) no-repeat center"
    expected = {
        (0, 0): [("width", "10em")],
        (1, 0): [
            ("width", "10em"),
            ("background", bg_s),
            ("background-size", "100% 50.0%"),
        ],
    }
    assert result == expected


def test_bar_value_error_raises():
    df = DataFrame({"A": [-100, -60, -30, -20]})

    msg = "`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(align="poorly", color=["#d65f5f", "#5fba7d"]).to_html()

    msg = r"`width` must be a value in \[0, 100\]"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(width=200).to_html()

    msg = r"`height` must be a value in \[0, 100\]"
    with pytest.raises(ValueError, match=msg):
        df.style.bar(height=200).to_html()


def test_bar_color_and_cmap_error_raises():
    df = DataFrame({"A": [1, 2, 3, 4]})
    msg = "`color` and `cmap` cannot both be given"
    # Test that providing both color and cmap raises a ValueError
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color="#d65f5f", cmap="viridis").to_html()


def test_bar_invalid_color_type_error_raises():
    df = DataFrame({"A": [1, 2, 3, 4]})
    msg = (
        r"`color` must be string or list or tuple of 2 strings,"
        r"\(eg: color=\['#d65f5f', '#5fba7d'\]\)"
    )
    # Test that providing an invalid color type raises a ValueError
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=123).to_html()

    # Test that providing a color list with more than two elements raises a ValueError
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=["#d65f5f", "#5fba7d", "#abcdef"]).to_html()


def test_styler_bar_with_NA_values():
    df1 = DataFrame({"A": [1, 2, NA, 4]})
    df2 = DataFrame([[NA, NA], [NA, NA]])
    expected_substring = "style type="
    html_output1 = df1.style.bar(subset="A").to_html()
    html_output2 = df2.style.bar(align="left", axis=None).to_html()
    assert expected_substring in html_output1
    assert expected_substring in html_output2


def test_style_bar_with_pyarrow_NA_values():
    pytest.importorskip("pyarrow")
    data = """name,age,test1,test2,teacher
        Adam,15,95.0,80,Ashby
        Bob,16,81.0,82,Ashby
        Dave,16,89.0,84,Jones
        Fred,15,,88,Jones"""
    df = read_csv(io.StringIO(data), dtype_backend="pyarrow")
    expected_substring = "style type="
    html_output = df.style.bar(subset="test1").to_html()
    assert expected_substring in html_output

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\tests\test_latex.py ===
from sympy.testing.pytest import raises, XFAIL
from sympy.external import import_module

from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (E, oo)
from sympy.core.power import Pow
from sympy.core.relational import (GreaterThan, LessThan, StrictGreaterThan, StrictLessThan, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, conjugate)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import (ceiling, floor)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, csc, sec, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.series.limits import Limit

from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.physics.quantum.state import Bra, Ket
from sympy.abc import x, y, z, a, b, c, t, k, n
antlr4 = import_module("antlr4")

# disable tests if antlr4-python3-runtime is not present
disabled = antlr4 is None

theta = Symbol('theta')
f = Function('f')


# shorthand definitions
def _Add(a, b):
    return Add(a, b, evaluate=False)


def _Mul(a, b):
    return Mul(a, b, evaluate=False)


def _Pow(a, b):
    return Pow(a, b, evaluate=False)


def _Sqrt(a):
    return sqrt(a, evaluate=False)


def _Conjugate(a):
    return conjugate(a, evaluate=False)


def _Abs(a):
    return Abs(a, evaluate=False)


def _factorial(a):
    return factorial(a, evaluate=False)


def _exp(a):
    return exp(a, evaluate=False)


def _log(a, b):
    return log(a, b, evaluate=False)


def _binomial(n, k):
    return binomial(n, k, evaluate=False)


def test_import():
    from sympy.parsing.latex._build_latex_antlr import (
        build_parser,
        check_antlr_version,
        dir_latex_antlr
    )
    # XXX: It would be better to come up with a test for these...
    del build_parser, check_antlr_version, dir_latex_antlr


# These LaTeX strings should parse to the corresponding SymPy expression
GOOD_PAIRS = [
    (r"0", 0),
    (r"1", 1),
    (r"-3.14", -3.14),
    (r"(-7.13)(1.5)", _Mul(-7.13, 1.5)),
    (r"x", x),
    (r"2x", 2*x),
    (r"x^2", x**2),
    (r"x^\frac{1}{2}", _Pow(x, _Pow(2, -1))),
    (r"x^{3 + 1}", x**_Add(3, 1)),
    (r"-c", -c),
    (r"a \cdot b", a * b),
    (r"a / b", a / b),
    (r"a \div b", a / b),
    (r"a + b", a + b),
    (r"a + b - a", _Add(a+b, -a)),
    (r"a^2 + b^2 = c^2", Eq(a**2 + b**2, c**2)),
    (r"(x + y) z", _Mul(_Add(x, y), z)),
    (r"a'b+ab'", _Add(_Mul(Symbol("a'"), b), _Mul(a, Symbol("b'")))),
    (r"y''_1", Symbol("y_{1}''")),
    (r"y_1''", Symbol("y_{1}''")),
    (r"\left(x + y\right) z", _Mul(_Add(x, y), z)),
    (r"\left( x + y\right ) z", _Mul(_Add(x, y), z)),
    (r"\left(  x + y\right ) z", _Mul(_Add(x, y), z)),
    (r"\left[x + y\right] z", _Mul(_Add(x, y), z)),
    (r"\left\{x + y\right\} z", _Mul(_Add(x, y), z)),
    (r"1+1", _Add(1, 1)),
    (r"0+1", _Add(0, 1)),
    (r"1*2", _Mul(1, 2)),
    (r"0*1", _Mul(0, 1)),
    (r"1 \times 2 ", _Mul(1, 2)),
    (r"x = y", Eq(x, y)),
    (r"x \neq y", Ne(x, y)),
    (r"x < y", Lt(x, y)),
    (r"x > y", Gt(x, y)),
    (r"x \leq y", Le(x, y)),
    (r"x \geq y", Ge(x, y)),
    (r"x \le y", Le(x, y)),
    (r"x \ge y", Ge(x, y)),
    (r"\lfloor x \rfloor", floor(x)),
    (r"\lceil x \rceil", ceiling(x)),
    (r"\langle x |", Bra('x')),
    (r"| x \rangle", Ket('x')),
    (r"\sin \theta", sin(theta)),
    (r"\sin(\theta)", sin(theta)),
    (r"\sin^{-1} a", asin(a)),
    (r"\sin a \cos b", _Mul(sin(a), cos(b))),
    (r"\sin \cos \theta", sin(cos(theta))),
    (r"\sin(\cos \theta)", sin(cos(theta))),
    (r"\frac{a}{b}", a / b),
    (r"\dfrac{a}{b}", a / b),
    (r"\tfrac{a}{b}", a / b),
    (r"\frac12", _Pow(2, -1)),
    (r"\frac12y", _Mul(_Pow(2, -1), y)),
    (r"\frac1234", _Mul(_Pow(2, -1), 34)),
    (r"\frac2{3}", _Mul(2, _Pow(3, -1))),
    (r"\frac{\sin{x}}2", _Mul(sin(x), _Pow(2, -1))),
    (r"\frac{a + b}{c}", _Mul(a + b, _Pow(c, -1))),
    (r"\frac{7}{3}", _Mul(7, _Pow(3, -1))),
    (r"(\csc x)(\sec y)", csc(x)*sec(y)),
    (r"\lim_{x \to 3} a", Limit(a, x, 3, dir='+-')),
    (r"\lim_{x \rightarrow 3} a", Limit(a, x, 3, dir='+-')),
    (r"\lim_{x \Rightarrow 3} a", Limit(a, x, 3, dir='+-')),
    (r"\lim_{x \longrightarrow 3} a", Limit(a, x, 3, dir='+-')),
    (r"\lim_{x \Longrightarrow 3} a", Limit(a, x, 3, dir='+-')),
    (r"\lim_{x \to 3^{+}} a", Limit(a, x, 3, dir='+')),
    (r"\lim_{x \to 3^{-}} a", Limit(a, x, 3, dir='-')),
    (r"\lim_{x \to 3^+} a", Limit(a, x, 3, dir='+')),
    (r"\lim_{x \to 3^-} a", Limit(a, x, 3, dir='-')),
    (r"\infty", oo),
    (r"\lim_{x \to \infty} \frac{1}{x}", Limit(_Pow(x, -1), x, oo)),
    (r"\frac{d}{dx} x", Derivative(x, x)),
    (r"\frac{d}{dt} x", Derivative(x, t)),
    (r"f(x)", f(x)),
    (r"f(x, y)", f(x, y)),
    (r"f(x, y, z)", f(x, y, z)),
    (r"f'_1(x)", Function("f_{1}'")(x)),
    (r"f_{1}''(x+y)", Function("f_{1}''")(x+y)),
    (r"\frac{d f(x)}{dx}", Derivative(f(x), x)),
    (r"\frac{d\theta(x)}{dx}", Derivative(Function('theta')(x), x)),
    (r"x \neq y", Unequality(x, y)),
    (r"|x|", _Abs(x)),
    (r"||x||", _Abs(Abs(x))),
    (r"|x||y|", _Abs(x)*_Abs(y)),
    (r"||x||y||", _Abs(_Abs(x)*_Abs(y))),
    (r"\pi^{|xy|}", Symbol('pi')**_Abs(x*y)),
    (r"\int x dx", Integral(x, x)),
    (r"\int x d\theta", Integral(x, theta)),
    (r"\int (x^2 - y)dx", Integral(x**2 - y, x)),
    (r"\int x + a dx", Integral(_Add(x, a), x)),
    (r"\int da", Integral(1, a)),
    (r"\int_0^7 dx", Integral(1, (x, 0, 7))),
    (r"\int\limits_{0}^{1} x dx", Integral(x, (x, 0, 1))),
    (r"\int_a^b x dx", Integral(x, (x, a, b))),
    (r"\int^b_a x dx", Integral(x, (x, a, b))),
    (r"\int_{a}^b x dx", Integral(x, (x, a, b))),
    (r"\int^{b}_a x dx", Integral(x, (x, a, b))),
    (r"\int_{a}^{b} x dx", Integral(x, (x, a, b))),
    (r"\int^{b}_{a} x dx", Integral(x, (x, a, b))),
    (r"\int_{f(a)}^{f(b)} f(z) dz", Integral(f(z), (z, f(a), f(b)))),
    (r"\int (x+a)", Integral(_Add(x, a), x)),
    (r"\int a + b + c dx", Integral(_Add(_Add(a, b), c), x)),
    (r"\int \frac{dz}{z}", Integral(Pow(z, -1), z)),
    (r"\int \frac{3 dz}{z}", Integral(3*Pow(z, -1), z)),
    (r"\int \frac{1}{x} dx", Integral(Pow(x, -1), x)),
    (r"\int \frac{1}{a} + \frac{1}{b} dx",
     Integral(_Add(_Pow(a, -1), Pow(b, -1)), x)),
    (r"\int \frac{3 \cdot d\theta}{\theta}",
     Integral(3*_Pow(theta, -1), theta)),
    (r"\int \frac{1}{x} + 1 dx", Integral(_Add(_Pow(x, -1), 1), x)),
    (r"x_0", Symbol('x_{0}')),
    (r"x_{1}", Symbol('x_{1}')),
    (r"x_a", Symbol('x_{a}')),
    (r"x_{b}", Symbol('x_{b}')),
    (r"h_\theta", Symbol('h_{theta}')),
    (r"h_{\theta}", Symbol('h_{theta}')),
    (r"h_{\theta}(x_0, x_1)",
     Function('h_{theta}')(Symbol('x_{0}'), Symbol('x_{1}'))),
    (r"x!", _factorial(x)),
    (r"100!", _factorial(100)),
    (r"\theta!", _factorial(theta)),
    (r"(x + 1)!", _factorial(_Add(x, 1))),
    (r"(x!)!", _factorial(_factorial(x))),
    (r"x!!!", _factorial(_factorial(_factorial(x)))),
    (r"5!7!", _Mul(_factorial(5), _factorial(7))),
    (r"\sqrt{x}", sqrt(x)),
    (r"\sqrt{x + b}", sqrt(_Add(x, b))),
    (r"\sqrt[3]{\sin x}", root(sin(x), 3)),
    (r"\sqrt[y]{\sin x}", root(sin(x), y)),
    (r"\sqrt[\theta]{\sin x}", root(sin(x), theta)),
    (r"\sqrt{\frac{12}{6}}", _Sqrt(_Mul(12, _Pow(6, -1)))),
    (r"\overline{z}", _Conjugate(z)),
    (r"\overline{\overline{z}}", _Conjugate(_Conjugate(z))),
    (r"\overline{x + y}", _Conjugate(_Add(x, y))),
    (r"\overline{x} + \overline{y}", _Conjugate(x) + _Conjugate(y)),
    (r"x < y", StrictLessThan(x, y)),
    (r"x \leq y", LessThan(x, y)),
    (r"x > y", StrictGreaterThan(x, y)),
    (r"x \geq y", GreaterThan(x, y)),
    (r"\mathit{x}", Symbol('x')),
    (r"\mathit{test}", Symbol('test')),
    (r"\mathit{TEST}", Symbol('TEST')),
    (r"\mathit{HELLO world}", Symbol('HELLO world')),
    (r"\sum_{k = 1}^{3} c", Sum(c, (k, 1, 3))),
    (r"\sum_{k = 1}^3 c", Sum(c, (k, 1, 3))),
    (r"\sum^{3}_{k = 1} c", Sum(c, (k, 1, 3))),
    (r"\sum^3_{k = 1} c", Sum(c, (k, 1, 3))),
    (r"\sum_{k = 1}^{10} k^2", Sum(k**2, (k, 1, 10))),
    (r"\sum_{n = 0}^{\infty} \frac{1}{n!}",
     Sum(_Pow(_factorial(n), -1), (n, 0, oo))),
    (r"\prod_{a = b}^{c} x", Product(x, (a, b, c))),
    (r"\prod_{a = b}^c x", Product(x, (a, b, c))),
    (r"\prod^{c}_{a = b} x", Product(x, (a, b, c))),
    (r"\prod^c_{a = b} x", Product(x, (a, b, c))),
    (r"\exp x", _exp(x)),
    (r"\exp(x)", _exp(x)),
    (r"\lg x", _log(x, 10)),
    (r"\ln x", _log(x, E)),
    (r"\ln xy", _log(x*y, E)),
    (r"\log x", _log(x, E)),
    (r"\log xy", _log(x*y, E)),
    (r"\log_{2} x", _log(x, 2)),
    (r"\log_{a} x", _log(x, a)),
    (r"\log_{11} x", _log(x, 11)),
    (r"\log_{a^2} x", _log(x, _Pow(a, 2))),
    (r"[x]", x),
    (r"[a + b]", _Add(a, b)),
    (r"\frac{d}{dx} [ \tan x ]", Derivative(tan(x), x)),
    (r"\binom{n}{k}", _binomial(n, k)),
    (r"\tbinom{n}{k}", _binomial(n, k)),
    (r"\dbinom{n}{k}", _binomial(n, k)),
    (r"\binom{n}{0}", _binomial(n, 0)),
    (r"x^\binom{n}{k}", _Pow(x, _binomial(n, k))),
    (r"a \, b", _Mul(a, b)),
    (r"a \thinspace b", _Mul(a, b)),
    (r"a \: b", _Mul(a, b)),
    (r"a \medspace b", _Mul(a, b)),
    (r"a \; b", _Mul(a, b)),
    (r"a \thickspace b", _Mul(a, b)),
    (r"a \quad b", _Mul(a, b)),
    (r"a \qquad b", _Mul(a, b)),
    (r"a \! b", _Mul(a, b)),
    (r"a \negthinspace b", _Mul(a, b)),
    (r"a \negmedspace b", _Mul(a, b)),
    (r"a \negthickspace b", _Mul(a, b)),
    (r"\int x \, dx", Integral(x, x)),
    (r"\log_2 x", _log(x, 2)),
    (r"\log_a x", _log(x, a)),
    (r"5^0 - 4^0", _Add(_Pow(5, 0), _Mul(-1, _Pow(4, 0)))),
    (r"3x - 1", _Add(_Mul(3, x), -1))
]


def test_parseable():
    from sympy.parsing.latex import parse_latex
    for latex_str, sympy_expr in GOOD_PAIRS:
        assert parse_latex(latex_str) == sympy_expr, latex_str

# These bad LaTeX strings should raise a LaTeXParsingError when parsed
BAD_STRINGS = [
    r"(",
    r")",
    r"\frac{d}{dx}",
    r"(\frac{d}{dx})",
    r"\sqrt{}",
    r"\sqrt",
    r"\overline{}",
    r"\overline",
    r"{",
    r"}",
    r"\mathit{x + y}",
    r"\mathit{21}",
    r"\frac{2}{}",
    r"\frac{}{2}",
    r"\int",
    r"!",
    r"!0",
    r"_",
    r"^",
    r"|",
    r"||x|",
    r"()",
    r"((((((((((((((((()))))))))))))))))",
    r"-",
    r"\frac{d}{dx} + \frac{d}{dt}",
    r"f(x,,y)",
    r"f(x,y,",
    r"\sin^x",
    r"\cos^2",
    r"@",
    r"#",
    r"$",
    r"%",
    r"&",
    r"*",
    r"" "\\",
    r"~",
    r"\frac{(2 + x}{1 - x)}",
]

def test_not_parseable():
    from sympy.parsing.latex import parse_latex, LaTeXParsingError
    for latex_str in BAD_STRINGS:
        with raises(LaTeXParsingError):
            parse_latex(latex_str)

# At time of migration from latex2sympy, should fail but doesn't
FAILING_BAD_STRINGS = [
    r"\cos 1 \cos",
    r"f(,",
    r"f()",
    r"a \div \div b",
    r"a \cdot \cdot b",
    r"a // b",
    r"a +",
    r"1.1.1",
    r"1 +",
    r"a / b /",
]

@XFAIL
def test_failing_not_parseable():
    from sympy.parsing.latex import parse_latex, LaTeXParsingError
    for latex_str in FAILING_BAD_STRINGS:
        with raises(LaTeXParsingError):
            parse_latex(latex_str)

# In strict mode, FAILING_BAD_STRINGS would fail
def test_strict_mode():
    from sympy.parsing.latex import parse_latex, LaTeXParsingError
    for latex_str in FAILING_BAD_STRINGS:
        with raises(LaTeXParsingError):
            parse_latex(latex_str, strict=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_eigen_symmetric.py ===
#!/usr/bin/python
# -*- coding: utf-8 -*-

from mpmath import mp
from mpmath import libmp

xrange = libmp.backend.xrange

def run_eigsy(A, verbose = False):
    if verbose:
        print("original matrix:\n", str(A))

    D, Q = mp.eigsy(A)
    B = Q * mp.diag(D) * Q.transpose()
    C = A - B
    E = Q * Q.transpose() - mp.eye(A.rows)

    if verbose:
        print("eigenvalues:\n", D)
        print("eigenvectors:\n", Q)

    NC = mp.mnorm(C)
    NE = mp.mnorm(E)

    if verbose:
        print("difference:", NC, "\n", C, "\n")
        print("difference:", NE, "\n", E, "\n")

    eps = mp.exp( 0.8 * mp.log(mp.eps))

    assert NC < eps
    assert NE < eps

    return NC

def run_eighe(A, verbose = False):
    if verbose:
        print("original matrix:\n", str(A))

    D, Q = mp.eighe(A)
    B = Q * mp.diag(D) * Q.transpose_conj()
    C = A - B
    E = Q * Q.transpose_conj() - mp.eye(A.rows)

    if verbose:
        print("eigenvalues:\n", D)
        print("eigenvectors:\n", Q)

    NC = mp.mnorm(C)
    NE = mp.mnorm(E)

    if verbose:
        print("difference:", NC, "\n", C, "\n")
        print("difference:", NE, "\n", E, "\n")

    eps = mp.exp( 0.8 * mp.log(mp.eps))

    assert NC < eps
    assert NE < eps

    return NC

def run_svd_r(A, full_matrices = False, verbose = True):

    m, n = A.rows, A.cols

    eps = mp.exp(0.8 * mp.log(mp.eps))

    if verbose:
        print("original matrix:\n", str(A))
        print("full", full_matrices)

    U, S0, V = mp.svd_r(A, full_matrices = full_matrices)

    S = mp.zeros(U.cols, V.rows)
    for j in xrange(min(m, n)):
        S[j,j] = S0[j]

    if verbose:
        print("U:\n", str(U))
        print("S:\n", str(S0))
        print("V:\n", str(V))

    C = U * S * V - A
    err = mp.mnorm(C)
    if verbose:
        print("C\n", str(C), "\n", err)
    assert err < eps

    D = V * V.transpose() - mp.eye(V.rows)
    err = mp.mnorm(D)
    if verbose:
        print("D:\n", str(D), "\n", err)
    assert err < eps

    E = U.transpose() * U - mp.eye(U.cols)
    err = mp.mnorm(E)
    if verbose:
        print("E:\n", str(E), "\n", err)
    assert err < eps

def run_svd_c(A, full_matrices = False, verbose = True):

    m, n = A.rows, A.cols

    eps = mp.exp(0.8 * mp.log(mp.eps))

    if verbose:
        print("original matrix:\n", str(A))
        print("full", full_matrices)

    U, S0, V = mp.svd_c(A, full_matrices = full_matrices)

    S = mp.zeros(U.cols, V.rows)
    for j in xrange(min(m, n)):
        S[j,j] = S0[j]

    if verbose:
        print("U:\n", str(U))
        print("S:\n", str(S0))
        print("V:\n", str(V))

    C = U * S * V - A
    err = mp.mnorm(C)
    if verbose:
        print("C\n", str(C), "\n", err)
    assert err  < eps

    D = V * V.transpose_conj() - mp.eye(V.rows)
    err = mp.mnorm(D)
    if verbose:
        print("D:\n", str(D), "\n", err)
    assert err < eps

    E = U.transpose_conj() * U - mp.eye(U.cols)
    err = mp.mnorm(E)
    if verbose:
        print("E:\n", str(E), "\n", err)
    assert err < eps

def run_gauss(qtype, a, b):
    eps = 1e-5

    d, e = mp.gauss_quadrature(len(a), qtype)
    d -= mp.matrix(a)
    e -= mp.matrix(b)

    assert mp.mnorm(d) < eps
    assert mp.mnorm(e) < eps

def irandmatrix(n, range = 10):
    """
    random matrix with integer entries
    """
    A = mp.matrix(n, n)
    for i in xrange(n):
        for j in xrange(n):
            A[i,j]=int( (2 * mp.rand() - 1) * range)
    return A

#######################

def test_eighe_fixed_matrix():
    A = mp.matrix([[2, 3], [3, 5]])
    run_eigsy(A)
    run_eighe(A)

    A = mp.matrix([[7, -11], [-11, 13]])
    run_eigsy(A)
    run_eighe(A)

    A = mp.matrix([[2, 11, 7], [11, 3, 13], [7, 13, 5]])
    run_eigsy(A)
    run_eighe(A)

    A = mp.matrix([[2, 0, 7], [0, 3, 1], [7, 1, 5]])
    run_eigsy(A)
    run_eighe(A)

    #

    A = mp.matrix([[2, 3+7j], [3-7j, 5]])
    run_eighe(A)

    A = mp.matrix([[2, -11j, 0], [+11j, 3, 29j], [0, -29j, 5]])
    run_eighe(A)

    A = mp.matrix([[2, 11 + 17j, 7 + 19j], [11 - 17j, 3, -13 + 23j], [7 - 19j, -13 - 23j, 5]])
    run_eighe(A)

def test_eigsy_randmatrix():
    N = 5

    for a in xrange(10):
        A = 2 * mp.randmatrix(N, N) - 1

        for i in xrange(0, N):
            for j in xrange(i + 1, N):
                A[j,i] = A[i,j]

        run_eigsy(A)

def test_eighe_randmatrix():
    N = 5

    for a in xrange(10):
        A = (2 * mp.randmatrix(N, N) - 1) + 1j * (2 * mp.randmatrix(N, N) - 1)

        for i in xrange(0, N):
            A[i,i] = mp.re(A[i,i])
            for j in xrange(i + 1, N):
                A[j,i] = mp.conj(A[i,j])

        run_eighe(A)

def test_eigsy_irandmatrix():
    N = 4
    R = 4

    for a in xrange(10):
        A=irandmatrix(N, R)

        for i in xrange(0, N):
            for j in xrange(i + 1, N):
                A[j,i] = A[i,j]

        run_eigsy(A)

def test_eighe_irandmatrix():
    N = 4
    R = 4

    for a in xrange(10):
        A=irandmatrix(N, R) + 1j * irandmatrix(N, R)

        for i in xrange(0, N):
            A[i,i] = mp.re(A[i,i])
            for j in xrange(i + 1, N):
                A[j,i] = mp.conj(A[i,j])

        run_eighe(A)

def test_svd_r_rand():
    for i in xrange(5):
        full = mp.rand() > 0.5
        m = 1 + int(mp.rand() * 10)
        n = 1 + int(mp.rand() * 10)
        A = 2 * mp.randmatrix(m, n) - 1
        if mp.rand() > 0.5:
            A *= 10
            for x in xrange(m):
                for y in xrange(n):
                    A[x,y]=int(A[x,y])

        run_svd_r(A, full_matrices = full, verbose = False)

def test_svd_c_rand():
    for i in xrange(5):
        full = mp.rand() > 0.5
        m = 1 + int(mp.rand() * 10)
        n = 1 + int(mp.rand() * 10)
        A = (2 * mp.randmatrix(m, n) - 1) + 1j * (2 * mp.randmatrix(m, n) - 1)
        if mp.rand() > 0.5:
            A *= 10
            for x in xrange(m):
                for y in xrange(n):
                    A[x,y]=int(mp.re(A[x,y])) + 1j * int(mp.im(A[x,y]))

        run_svd_c(A, full_matrices=full, verbose=False)

def test_svd_test_case():
    # a test case from Golub and Reinsch
    #  (see wilkinson/reinsch: handbook for auto. comp., vol ii-linear algebra, 134-151(1971).)

    eps = mp.exp(0.8 * mp.log(mp.eps))

    a = [[22, 10,  2,   3,  7],
         [14,  7, 10,   0,  8],
         [-1, 13, -1, -11,  3],
         [-3, -2, 13,  -2,  4],
         [ 9,  8,  1,  -2,  4],
         [ 9,  1, -7,   5, -1],
         [ 2, -6,  6,   5,  1],
         [ 4,  5,  0,  -2,  2]]

    a = mp.matrix(a)
    b = mp.matrix([mp.sqrt(1248), 20, mp.sqrt(384), 0, 0])

    S = mp.svd_r(a, compute_uv = False)
    S -= b
    assert mp.mnorm(S) < eps

    S = mp.svd_c(a, compute_uv = False)
    S -= b
    assert mp.mnorm(S) < eps


def test_gauss_quadrature_static():
    a = [-0.57735027,  0.57735027]
    b = [ 1,  1]
    run_gauss("legendre", a , b)

    a = [ -0.906179846,  -0.538469310,   0,           0.538469310,   0.906179846]
    b = [  0.23692689,    0.47862867,    0.56888889,  0.47862867,    0.23692689]
    run_gauss("legendre", a , b)

    a = [ 0.06943184,  0.33000948,  0.66999052,  0.93056816]
    b = [ 0.17392742,  0.32607258,  0.32607258,  0.17392742]
    run_gauss("legendre01", a , b)

    a = [-0.70710678,  0.70710678]
    b = [ 0.88622693,  0.88622693]
    run_gauss("hermite", a , b)

    a = [ -2.02018287,  -0.958572465,   0,           0.958572465,   2.02018287]
    b = [  0.01995324,   0.39361932,    0.94530872,  0.39361932,    0.01995324]
    run_gauss("hermite", a , b)

    a = [ 0.41577456,  2.29428036,  6.28994508]
    b = [ 0.71109301,  0.27851773,  0.01038926]
    run_gauss("laguerre", a , b)

def test_gauss_quadrature_dynamic(verbose = False):
    n = 5

    A = mp.randmatrix(2 * n, 1)

    def F(x):
        r = 0
        for i in xrange(len(A) - 1, -1, -1):
            r = r * x + A[i]
        return r

    def run(qtype, FW, R, alpha = 0, beta = 0):
        X, W = mp.gauss_quadrature(n, qtype, alpha = alpha, beta = beta)

        a = 0
        for i in xrange(len(X)):
            a += W[i] * F(X[i])

        b = mp.quad(lambda x: FW(x) * F(x), R)

        c = mp.fabs(a - b)

        if verbose:
            print(qtype, c, a, b)

        assert c < 1e-5

    run("legendre", lambda x: 1, [-1, 1])
    run("legendre01", lambda x: 1, [0, 1])
    run("hermite", lambda x: mp.exp(-x*x), [-mp.inf, mp.inf])
    run("laguerre", lambda x: mp.exp(-x), [0, mp.inf])
    run("glaguerre", lambda x: mp.sqrt(x)*mp.exp(-x), [0, mp.inf], alpha = 1 / mp.mpf(2))
    run("chebyshev1", lambda x: 1/mp.sqrt(1-x*x), [-1, 1])
    run("chebyshev2", lambda x: mp.sqrt(1-x*x), [-1, 1])
    run("jacobi", lambda x: (1-x)**(1/mp.mpf(3)) * (1+x)**(1/mp.mpf(5)), [-1, 1], alpha = 1 / mp.mpf(3), beta = 1 / mp.mpf(5) )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\test_indexing.py ===
"""
test_indexing tests the following Index methods:
    __getitem__
    get_loc
    get_value
    __contains__
    take
    where
    get_indexer
    get_indexer_for
    slice_locs
    asof_locs

The corresponding tests.indexes.[index_type].test_indexing files
contain tests for the corresponding methods specific to those Index subclasses.
"""
import numpy as np
import pytest

from pandas.errors import InvalidIndexError

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_scalar,
)

from pandas import (
    NA,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    NaT,
    PeriodIndex,
    TimedeltaIndex,
)
import pandas._testing as tm


class TestTake:
    def test_take_invalid_kwargs(self, index):
        indices = [1, 2]

        msg = r"take\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            index.take(indices, foo=2)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            index.take(indices, out=indices)

        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            index.take(indices, mode="clip")

    def test_take(self, index):
        indexer = [4, 3, 0, 2]
        if len(index) < 5:
            pytest.skip("Test doesn't make sense since not enough elements")

        result = index.take(indexer)
        expected = index[indexer]
        assert result.equals(expected)

        if not isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)):
            # GH 10791
            msg = r"'(.*Index)' object has no attribute 'freq'"
            with pytest.raises(AttributeError, match=msg):
                index.freq

    def test_take_indexer_type(self):
        # GH#42875
        integer_index = Index([0, 1, 2, 3])
        scalar_index = 1
        msg = "Expected indices to be array-like"
        with pytest.raises(TypeError, match=msg):
            integer_index.take(scalar_index)

    def test_take_minus1_without_fill(self, index):
        # -1 does not get treated as NA unless allow_fill=True is passed
        if len(index) == 0:
            # Test is not applicable
            pytest.skip("Test doesn't make sense for empty index")

        result = index.take([0, 0, -1])

        expected = index.take([0, 0, len(index) - 1])
        tm.assert_index_equal(result, expected)


class TestContains:
    @pytest.mark.parametrize(
        "index,val",
        [
            (Index([0, 1, 2]), 2),
            (Index([0, 1, "2"]), "2"),
            (Index([0, 1, 2, np.inf, 4]), 4),
            (Index([0, 1, 2, np.nan, 4]), 4),
            (Index([0, 1, 2, np.inf]), np.inf),
            (Index([0, 1, 2, np.nan]), np.nan),
        ],
    )
    def test_index_contains(self, index, val):
        assert val in index

    @pytest.mark.parametrize(
        "index,val",
        [
            (Index([0, 1, 2]), "2"),
            (Index([0, 1, "2"]), 2),
            (Index([0, 1, 2, np.inf]), 4),
            (Index([0, 1, 2, np.nan]), 4),
            (Index([0, 1, 2, np.inf]), np.nan),
            (Index([0, 1, 2, np.nan]), np.inf),
            # Checking if np.inf in int64 Index should not cause an OverflowError
            # Related to GH 16957
            (Index([0, 1, 2], dtype=np.int64), np.inf),
            (Index([0, 1, 2], dtype=np.int64), np.nan),
            (Index([0, 1, 2], dtype=np.uint64), np.inf),
            (Index([0, 1, 2], dtype=np.uint64), np.nan),
        ],
    )
    def test_index_not_contains(self, index, val):
        assert val not in index

    @pytest.mark.parametrize(
        "index,val", [(Index([0, 1, "2"]), 0), (Index([0, 1, "2"]), "2")]
    )
    def test_mixed_index_contains(self, index, val):
        # GH#19860
        assert val in index

    @pytest.mark.parametrize(
        "index,val", [(Index([0, 1, "2"]), "1"), (Index([0, 1, "2"]), 2)]
    )
    def test_mixed_index_not_contains(self, index, val):
        # GH#19860
        assert val not in index

    def test_contains_with_float_index(self, any_real_numpy_dtype):
        # GH#22085
        dtype = any_real_numpy_dtype
        data = [0, 1, 2, 3] if not is_float_dtype(dtype) else [0.1, 1.1, 2.2, 3.3]
        index = Index(data, dtype=dtype)

        if not is_float_dtype(index.dtype):
            assert 1.1 not in index
            assert 1.0 in index
            assert 1 in index
        else:
            assert 1.1 in index
            assert 1.0 not in index
            assert 1 not in index

    def test_contains_requires_hashable_raises(self, index):
        if isinstance(index, MultiIndex):
            return  # TODO: do we want this to raise?

        msg = "unhashable type: 'list'"
        with pytest.raises(TypeError, match=msg):
            [] in index

        msg = "|".join(
            [
                r"unhashable type: 'dict'",
                r"must be real number, not dict",
                r"an integer is required",
                r"\{\}",
                r"pandas\._libs\.interval\.IntervalTree' is not iterable",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            {} in index._engine


class TestGetLoc:
    def test_get_loc_non_hashable(self, index):
        with pytest.raises(InvalidIndexError, match="[0, 1]"):
            index.get_loc([0, 1])

    def test_get_loc_non_scalar_hashable(self, index):
        # GH52877
        from enum import Enum

        class E(Enum):
            X1 = "x1"

        assert not is_scalar(E.X1)

        exc = KeyError
        msg = "<E.X1: 'x1'>"
        if isinstance(
            index,
            (
                DatetimeIndex,
                TimedeltaIndex,
                PeriodIndex,
                IntervalIndex,
            ),
        ):
            # TODO: make these more consistent?
            exc = InvalidIndexError
            msg = "E.X1"
        with pytest.raises(exc, match=msg):
            index.get_loc(E.X1)

    def test_get_loc_generator(self, index):
        exc = KeyError
        if isinstance(
            index,
            (
                DatetimeIndex,
                TimedeltaIndex,
                PeriodIndex,
                IntervalIndex,
                MultiIndex,
            ),
        ):
            # TODO: make these more consistent?
            exc = InvalidIndexError
        with pytest.raises(exc, match="generator object"):
            # MultiIndex specifically checks for generator; others for scalar
            index.get_loc(x for x in range(5))

    def test_get_loc_masked_duplicated_na(self):
        # GH#48411
        idx = Index([1, 2, NA, NA], dtype="Int64")
        result = idx.get_loc(NA)
        expected = np.array([False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)


class TestGetIndexer:
    def test_get_indexer_base(self, index):
        if index._index_as_unique:
            expected = np.arange(index.size, dtype=np.intp)
            actual = index.get_indexer(index)
            tm.assert_numpy_array_equal(expected, actual)
        else:
            msg = "Reindexing only valid with uniquely valued Index objects"
            with pytest.raises(InvalidIndexError, match=msg):
                index.get_indexer(index)

        with pytest.raises(ValueError, match="Invalid fill method"):
            index.get_indexer(index, method="invalid")

    def test_get_indexer_consistency(self, index):
        # See GH#16819

        if index._index_as_unique:
            indexer = index.get_indexer(index[0:2])
            assert isinstance(indexer, np.ndarray)
            assert indexer.dtype == np.intp
        else:
            msg = "Reindexing only valid with uniquely valued Index objects"
            with pytest.raises(InvalidIndexError, match=msg):
                index.get_indexer(index[0:2])

        indexer, _ = index.get_indexer_non_unique(index[0:2])
        assert isinstance(indexer, np.ndarray)
        assert indexer.dtype == np.intp

    def test_get_indexer_masked_duplicated_na(self):
        # GH#48411
        idx = Index([1, 2, NA, NA], dtype="Int64")
        result = idx.get_indexer_for(Index([1, NA], dtype="Int64"))
        expected = np.array([0, 2, 3], dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)


class TestConvertSliceIndexer:
    def test_convert_almost_null_slice(self, index):
        # slice with None at both ends, but not step

        key = slice(None, None, "foo")

        if isinstance(index, IntervalIndex):
            msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
            with pytest.raises(ValueError, match=msg):
                index._convert_slice_indexer(key, "loc")
        else:
            msg = "'>=' not supported between instances of 'str' and 'int'"
            with pytest.raises(TypeError, match=msg):
                index._convert_slice_indexer(key, "loc")


class TestPutmask:
    def test_putmask_with_wrong_mask(self, index):
        # GH#18368
        if not len(index):
            pytest.skip("Test doesn't make sense for empty index")

        fill = index[0]

        msg = "putmask: mask and data must be the same size"
        with pytest.raises(ValueError, match=msg):
            index.putmask(np.ones(len(index) + 1, np.bool_), fill)

        with pytest.raises(ValueError, match=msg):
            index.putmask(np.ones(len(index) - 1, np.bool_), fill)

        with pytest.raises(ValueError, match=msg):
            index.putmask("foo", fill)


@pytest.mark.parametrize(
    "idx", [Index([1, 2, 3]), Index([0.1, 0.2, 0.3]), Index(["a", "b", "c"])]
)
def test_getitem_deprecated_float(idx):
    # https://github.com/pandas-dev/pandas/issues/34191

    msg = "Indexing with a float is no longer supported"
    with pytest.raises(IndexError, match=msg):
        idx[1.0]


@pytest.mark.parametrize(
    "idx,target,expected",
    [
        ([np.nan, "var1", np.nan], [np.nan], np.array([0, 2], dtype=np.intp)),
        (
            [np.nan, "var1", np.nan],
            [np.nan, "var1"],
            np.array([0, 2, 1], dtype=np.intp),
        ),
        (
            np.array([np.nan, "var1", np.nan], dtype=object),
            [np.nan],
            np.array([0, 2], dtype=np.intp),
        ),
        (
            DatetimeIndex(["2020-08-05", NaT, NaT]),
            [NaT],
            np.array([1, 2], dtype=np.intp),
        ),
        (["a", "b", "a", np.nan], [np.nan], np.array([3], dtype=np.intp)),
        (
            np.array(["b", np.nan, float("NaN"), "b"], dtype=object),
            Index([np.nan], dtype=object),
            np.array([1, 2], dtype=np.intp),
        ),
    ],
)
def test_get_indexer_non_unique_multiple_nans(idx, target, expected):
    # GH 35392
    axis = Index(idx)
    actual = axis.get_indexer_for(target)
    tm.assert_numpy_array_equal(actual, expected)


def test_get_indexer_non_unique_nans_in_object_dtype_target(nulls_fixture):
    idx = Index([1.0, 2.0])
    target = Index([1, nulls_fixture], dtype="object")

    result_idx, result_missing = idx.get_indexer_non_unique(target)
    tm.assert_numpy_array_equal(result_idx, np.array([0, -1], dtype=np.intp))
    tm.assert_numpy_array_equal(result_missing, np.array([1], dtype=np.intp))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\base\test_value_counts.py ===
import collections
from datetime import timedelta

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    array,
)
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_value_counts(index_or_series_obj):
    obj = index_or_series_obj
    obj = np.repeat(obj, range(1, len(obj) + 1))
    result = obj.value_counts()

    counter = collections.Counter(obj)
    expected = Series(dict(counter.most_common()), dtype=np.int64, name="count")

    if obj.dtype != np.float16:
        expected.index = expected.index.astype(obj.dtype)
    else:
        with pytest.raises(NotImplementedError, match="float16 indexes are not "):
            expected.index.astype(obj.dtype)
        return
    if isinstance(expected.index, MultiIndex):
        expected.index.names = obj.names
    else:
        expected.index.name = obj.name

    if not isinstance(result.dtype, np.dtype):
        if getattr(obj.dtype, "storage", "") == "pyarrow":
            expected = expected.astype("int64[pyarrow]")
        else:
            # i.e IntegerDtype
            expected = expected.astype("Int64")

    # TODO(GH#32514): Order of entries with the same count is inconsistent
    #  on CI (gh-32449)
    if obj.duplicated().any():
        result = result.sort_index()
        expected = expected.sort_index()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("null_obj", [np.nan, None])
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_value_counts_null(null_obj, index_or_series_obj):
    orig = index_or_series_obj
    obj = orig.copy()

    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
    elif isinstance(orig, MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")

    values = obj._values
    values[0:2] = null_obj

    klass = type(obj)
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    obj = klass(repeated_values, dtype=obj.dtype)

    # because np.nan == np.nan is False, but None == None is True
    # np.nan would be duplicated, whereas None wouldn't
    counter = collections.Counter(obj.dropna())
    expected = Series(dict(counter.most_common()), dtype=np.int64, name="count")

    if obj.dtype != np.float16:
        expected.index = expected.index.astype(obj.dtype)
    else:
        with pytest.raises(NotImplementedError, match="float16 indexes are not "):
            expected.index.astype(obj.dtype)
        return
    expected.index.name = obj.name

    result = obj.value_counts()
    if obj.duplicated().any():
        # TODO(GH#32514):
        #  Order of entries with the same count is inconsistent on CI (gh-32449)
        expected = expected.sort_index()
        result = result.sort_index()

    if not isinstance(result.dtype, np.dtype):
        if getattr(obj.dtype, "storage", "") == "pyarrow":
            expected = expected.astype("int64[pyarrow]")
        else:
            # i.e IntegerDtype
            expected = expected.astype("Int64")
    tm.assert_series_equal(result, expected)

    expected[null_obj] = 3

    result = obj.value_counts(dropna=False)
    if obj.duplicated().any():
        # TODO(GH#32514):
        #  Order of entries with the same count is inconsistent on CI (gh-32449)
        expected = expected.sort_index()
        result = result.sort_index()
    tm.assert_series_equal(result, expected)


def test_value_counts_inferred(index_or_series, using_infer_string):
    klass = index_or_series
    s_values = ["a", "b", "b", "b", "b", "c", "d", "d", "a", "a"]
    s = klass(s_values)
    expected = Series([4, 3, 2, 1], index=["b", "a", "d", "c"], name="count")
    tm.assert_series_equal(s.value_counts(), expected)

    if isinstance(s, Index):
        exp = Index(np.unique(np.array(s_values, dtype=np.object_)))
        tm.assert_index_equal(s.unique(), exp)
    else:
        exp = np.unique(np.array(s_values, dtype=np.object_))
        if using_infer_string:
            exp = array(exp, dtype="str")
        tm.assert_equal(s.unique(), exp)

    assert s.nunique() == 4
    # don't sort, have to sort after the fact as not sorting is
    # platform-dep
    hist = s.value_counts(sort=False).sort_values()
    expected = Series([3, 1, 4, 2], index=list("acbd"), name="count").sort_values()
    tm.assert_series_equal(hist, expected)

    # sort ascending
    hist = s.value_counts(ascending=True)
    expected = Series([1, 2, 3, 4], index=list("cdab"), name="count")
    tm.assert_series_equal(hist, expected)

    # relative histogram.
    hist = s.value_counts(normalize=True)
    expected = Series(
        [0.4, 0.3, 0.2, 0.1], index=["b", "a", "d", "c"], name="proportion"
    )
    tm.assert_series_equal(hist, expected)


def test_value_counts_bins(index_or_series, using_infer_string):
    klass = index_or_series
    s_values = ["a", "b", "b", "b", "b", "c", "d", "d", "a", "a"]
    s = klass(s_values)

    # bins
    msg = "bins argument only works with numeric data"
    with pytest.raises(TypeError, match=msg):
        s.value_counts(bins=1)

    s1 = Series([1, 1, 2, 3])
    res1 = s1.value_counts(bins=1)
    exp1 = Series({Interval(0.997, 3.0): 4}, name="count")
    tm.assert_series_equal(res1, exp1)
    res1n = s1.value_counts(bins=1, normalize=True)
    exp1n = Series({Interval(0.997, 3.0): 1.0}, name="proportion")
    tm.assert_series_equal(res1n, exp1n)

    if isinstance(s1, Index):
        tm.assert_index_equal(s1.unique(), Index([1, 2, 3]))
    else:
        exp = np.array([1, 2, 3], dtype=np.int64)
        tm.assert_numpy_array_equal(s1.unique(), exp)

    assert s1.nunique() == 3

    # these return the same
    res4 = s1.value_counts(bins=4, dropna=True)
    intervals = IntervalIndex.from_breaks([0.997, 1.5, 2.0, 2.5, 3.0])
    exp4 = Series([2, 1, 1, 0], index=intervals.take([0, 1, 3, 2]), name="count")
    tm.assert_series_equal(res4, exp4)

    res4 = s1.value_counts(bins=4, dropna=False)
    intervals = IntervalIndex.from_breaks([0.997, 1.5, 2.0, 2.5, 3.0])
    exp4 = Series([2, 1, 1, 0], index=intervals.take([0, 1, 3, 2]), name="count")
    tm.assert_series_equal(res4, exp4)

    res4n = s1.value_counts(bins=4, normalize=True)
    exp4n = Series(
        [0.5, 0.25, 0.25, 0], index=intervals.take([0, 1, 3, 2]), name="proportion"
    )
    tm.assert_series_equal(res4n, exp4n)

    # handle NA's properly
    s_values = ["a", "b", "b", "b", np.nan, np.nan, "d", "d", "a", "a", "b"]
    s = klass(s_values)
    expected = Series([4, 3, 2], index=["b", "a", "d"], name="count")
    tm.assert_series_equal(s.value_counts(), expected)

    if isinstance(s, Index):
        exp = Index(["a", "b", np.nan, "d"])
        tm.assert_index_equal(s.unique(), exp)
    else:
        exp = np.array(["a", "b", np.nan, "d"], dtype=object)
        if using_infer_string:
            exp = array(exp, dtype="str")
        tm.assert_equal(s.unique(), exp)
    assert s.nunique() == 3

    s = klass({}) if klass is dict else klass({}, dtype=object)
    expected = Series([], dtype=np.int64, name="count")
    tm.assert_series_equal(s.value_counts(), expected, check_index_type=False)
    # returned dtype differs depending on original
    if isinstance(s, Index):
        tm.assert_index_equal(s.unique(), Index([]), exact=False)
    else:
        tm.assert_numpy_array_equal(s.unique(), np.array([]), check_dtype=False)

    assert s.nunique() == 0


def test_value_counts_datetime64(index_or_series, unit):
    klass = index_or_series

    # GH 3002, datetime64[ns]
    # don't test names though
    df = pd.DataFrame(
        {
            "person_id": ["xxyyzz", "xxyyzz", "xxyyzz", "xxyyww", "foofoo", "foofoo"],
            "dt": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-01-01",
                    "2010-01-01",
                    "2009-01-01",
                    "2008-09-09",
                    "2008-09-09",
                ]
            ).as_unit(unit),
            "food": ["PIE", "GUM", "EGG", "EGG", "PIE", "GUM"],
        }
    )

    s = klass(df["dt"].copy())
    s.name = None
    idx = pd.to_datetime(
        ["2010-01-01 00:00:00", "2008-09-09 00:00:00", "2009-01-01 00:00:00"]
    ).as_unit(unit)
    expected_s = Series([3, 2, 1], index=idx, name="count")
    tm.assert_series_equal(s.value_counts(), expected_s)

    expected = array(
        np.array(
            ["2010-01-01 00:00:00", "2009-01-01 00:00:00", "2008-09-09 00:00:00"],
            dtype=f"datetime64[{unit}]",
        )
    )
    result = s.unique()
    if isinstance(s, Index):
        tm.assert_index_equal(result, DatetimeIndex(expected))
    else:
        tm.assert_extension_array_equal(result, expected)

    assert s.nunique() == 3

    # with NaT
    s = df["dt"].copy()
    s = klass(list(s.values) + [pd.NaT] * 4)
    if klass is Series:
        s = s.dt.as_unit(unit)
    else:
        s = s.as_unit(unit)

    result = s.value_counts()
    assert result.index.dtype == f"datetime64[{unit}]"
    tm.assert_series_equal(result, expected_s)

    result = s.value_counts(dropna=False)
    expected_s = pd.concat(
        [
            Series([4], index=DatetimeIndex([pd.NaT]).as_unit(unit), name="count"),
            expected_s,
        ]
    )
    tm.assert_series_equal(result, expected_s)

    assert s.dtype == f"datetime64[{unit}]"
    unique = s.unique()
    assert unique.dtype == f"datetime64[{unit}]"

    # numpy_array_equal cannot compare pd.NaT
    if isinstance(s, Index):
        exp_idx = DatetimeIndex(expected.tolist() + [pd.NaT]).as_unit(unit)
        tm.assert_index_equal(unique, exp_idx)
    else:
        tm.assert_extension_array_equal(unique[:3], expected)
        assert pd.isna(unique[3])

    assert s.nunique() == 3
    assert s.nunique(dropna=False) == 4


def test_value_counts_timedelta64(index_or_series, unit):
    # timedelta64[ns]
    klass = index_or_series

    day = Timedelta(timedelta(1)).as_unit(unit)
    tdi = TimedeltaIndex([day], name="dt").as_unit(unit)

    tdvals = np.zeros(6, dtype=f"m8[{unit}]") + day
    td = klass(tdvals, name="dt")

    result = td.value_counts()
    expected_s = Series([6], index=tdi, name="count")
    tm.assert_series_equal(result, expected_s)

    expected = tdi
    result = td.unique()
    if isinstance(td, Index):
        tm.assert_index_equal(result, expected)
    else:
        tm.assert_extension_array_equal(result, expected._values)

    td2 = day + np.zeros(6, dtype=f"m8[{unit}]")
    td2 = klass(td2, name="dt")
    result2 = td2.value_counts()
    tm.assert_series_equal(result2, expected_s)


@pytest.mark.parametrize("dropna", [True, False])
def test_value_counts_with_nan(dropna, index_or_series):
    # GH31944
    klass = index_or_series
    values = [True, pd.NA, np.nan]
    obj = klass(values)
    res = obj.value_counts(dropna=dropna)
    if dropna is True:
        expected = Series([1], index=Index([True], dtype=obj.dtype), name="count")
    else:
        expected = Series([1, 1, 1], index=[True, pd.NA, np.nan], name="count")
    tm.assert_series_equal(res, expected)


def test_value_counts_object_inference_deprecated():
    # GH#56161
    dti = pd.date_range("2016-01-01", periods=3, tz="UTC")

    idx = dti.astype(object)
    msg = "The behavior of value_counts with object-dtype is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = idx.value_counts()

    exp = dti.value_counts()
    tm.assert_series_equal(res, exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_formats.py ===
from datetime import datetime

import dateutil.tz
import numpy as np
import pytest
import pytz

import pandas as pd
from pandas import (
    DatetimeIndex,
    NaT,
    Series,
)
import pandas._testing as tm


@pytest.fixture(params=["s", "ms", "us", "ns"])
def unit(request):
    return request.param


def test_get_values_for_csv():
    index = pd.date_range(freq="1D", periods=3, start="2017-01-01")

    # First, with no arguments.
    expected = np.array(["2017-01-01", "2017-01-02", "2017-01-03"], dtype=object)

    result = index._get_values_for_csv()
    tm.assert_numpy_array_equal(result, expected)

    # No NaN values, so na_rep has no effect
    result = index._get_values_for_csv(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)

    # Make sure date formatting works
    expected = np.array(["01-2017-01", "01-2017-02", "01-2017-03"], dtype=object)

    result = index._get_values_for_csv(date_format="%m-%Y-%d")
    tm.assert_numpy_array_equal(result, expected)

    # NULL object handling should work
    index = DatetimeIndex(["2017-01-01", NaT, "2017-01-03"])
    expected = np.array(["2017-01-01", "NaT", "2017-01-03"], dtype=object)

    result = index._get_values_for_csv(na_rep="NaT")
    tm.assert_numpy_array_equal(result, expected)

    expected = np.array(["2017-01-01", "pandas", "2017-01-03"], dtype=object)

    result = index._get_values_for_csv(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)

    result = index._get_values_for_csv(na_rep="NaT", date_format="%Y-%m-%d %H:%M:%S.%f")
    expected = np.array(
        ["2017-01-01 00:00:00.000000", "NaT", "2017-01-03 00:00:00.000000"],
        dtype=object,
    )
    tm.assert_numpy_array_equal(result, expected)

    # invalid format
    result = index._get_values_for_csv(na_rep="NaT", date_format="foo")
    expected = np.array(["foo", "NaT", "foo"], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


class TestDatetimeIndexRendering:
    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_dti_with_timezone_repr(self, tzstr):
        rng = pd.date_range("4/13/2010", "5/6/2010")

        rng_eastern = rng.tz_localize(tzstr)

        rng_repr = repr(rng_eastern)
        assert "2010-04-13 00:00:00" in rng_repr

    def test_dti_repr_dates(self):
        text = str(pd.to_datetime([datetime(2013, 1, 1), datetime(2014, 1, 1)]))
        assert "['2013-01-01'," in text
        assert ", '2014-01-01']" in text

    def test_dti_repr_mixed(self):
        text = str(
            pd.to_datetime(
                [datetime(2013, 1, 1), datetime(2014, 1, 1, 12), datetime(2014, 1, 1)]
            )
        )
        assert "'2013-01-01 00:00:00'," in text
        assert "'2014-01-01 00:00:00']" in text

    def test_dti_repr_short(self):
        dr = pd.date_range(start="1/1/2012", periods=1)
        repr(dr)

        dr = pd.date_range(start="1/1/2012", periods=2)
        repr(dr)

        dr = pd.date_range(start="1/1/2012", periods=3)
        repr(dr)

    @pytest.mark.parametrize(
        "dates, freq, expected_repr",
        [
            (
                ["2012-01-01 00:00:00"],
                "60min",
                (
                    "DatetimeIndex(['2012-01-01 00:00:00'], "
                    "dtype='datetime64[ns]', freq='60min')"
                ),
            ),
            (
                ["2012-01-01 00:00:00", "2012-01-01 01:00:00"],
                "60min",
                "DatetimeIndex(['2012-01-01 00:00:00', '2012-01-01 01:00:00'], "
                "dtype='datetime64[ns]', freq='60min')",
            ),
            (
                ["2012-01-01"],
                "24h",
                "DatetimeIndex(['2012-01-01'], dtype='datetime64[ns]', freq='24h')",
            ),
        ],
    )
    def test_dti_repr_time_midnight(self, dates, freq, expected_repr, unit):
        # GH53634
        dti = DatetimeIndex(dates, freq).as_unit(unit)
        actual_repr = repr(dti)
        assert actual_repr == expected_repr.replace("[ns]", f"[{unit}]")

    def test_dti_representation(self, unit):
        idxs = []
        idxs.append(DatetimeIndex([], freq="D"))
        idxs.append(DatetimeIndex(["2011-01-01"], freq="D"))
        idxs.append(DatetimeIndex(["2011-01-01", "2011-01-02"], freq="D"))
        idxs.append(DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D"))
        idxs.append(
            DatetimeIndex(
                ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
                freq="h",
                tz="Asia/Tokyo",
            )
        )
        idxs.append(
            DatetimeIndex(
                ["2011-01-01 09:00", "2011-01-01 10:00", NaT], tz="US/Eastern"
            )
        )
        idxs.append(
            DatetimeIndex(["2011-01-01 09:00", "2011-01-01 10:00", NaT], tz="UTC")
        )

        exp = []
        exp.append("DatetimeIndex([], dtype='datetime64[ns]', freq='D')")
        exp.append("DatetimeIndex(['2011-01-01'], dtype='datetime64[ns]', freq='D')")
        exp.append(
            "DatetimeIndex(['2011-01-01', '2011-01-02'], "
            "dtype='datetime64[ns]', freq='D')"
        )
        exp.append(
            "DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], "
            "dtype='datetime64[ns]', freq='D')"
        )
        exp.append(
            "DatetimeIndex(['2011-01-01 09:00:00+09:00', "
            "'2011-01-01 10:00:00+09:00', '2011-01-01 11:00:00+09:00']"
            ", dtype='datetime64[ns, Asia/Tokyo]', freq='h')"
        )
        exp.append(
            "DatetimeIndex(['2011-01-01 09:00:00-05:00', "
            "'2011-01-01 10:00:00-05:00', 'NaT'], "
            "dtype='datetime64[ns, US/Eastern]', freq=None)"
        )
        exp.append(
            "DatetimeIndex(['2011-01-01 09:00:00+00:00', "
            "'2011-01-01 10:00:00+00:00', 'NaT'], "
            "dtype='datetime64[ns, UTC]', freq=None)"
            ""
        )

        with pd.option_context("display.width", 300):
            for index, expected in zip(idxs, exp):
                index = index.as_unit(unit)
                expected = expected.replace("[ns", f"[{unit}")
                result = repr(index)
                assert result == expected
                result = str(index)
                assert result == expected

    # TODO: this is a Series.__repr__ test
    def test_dti_representation_to_series(self, unit):
        idx1 = DatetimeIndex([], freq="D")
        idx2 = DatetimeIndex(["2011-01-01"], freq="D")
        idx3 = DatetimeIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
            freq="h",
            tz="Asia/Tokyo",
        )
        idx6 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", NaT], tz="US/Eastern"
        )
        idx7 = DatetimeIndex(["2011-01-01 09:00", "2011-01-02 10:15"])

        exp1 = """Series([], dtype: datetime64[ns])"""

        exp2 = "0   2011-01-01\ndtype: datetime64[ns]"

        exp3 = "0   2011-01-01\n1   2011-01-02\ndtype: datetime64[ns]"

        exp4 = (
            "0   2011-01-01\n"
            "1   2011-01-02\n"
            "2   2011-01-03\n"
            "dtype: datetime64[ns]"
        )

        exp5 = (
            "0   2011-01-01 09:00:00+09:00\n"
            "1   2011-01-01 10:00:00+09:00\n"
            "2   2011-01-01 11:00:00+09:00\n"
            "dtype: datetime64[ns, Asia/Tokyo]"
        )

        exp6 = (
            "0   2011-01-01 09:00:00-05:00\n"
            "1   2011-01-01 10:00:00-05:00\n"
            "2                         NaT\n"
            "dtype: datetime64[ns, US/Eastern]"
        )

        exp7 = (
            "0   2011-01-01 09:00:00\n"
            "1   2011-01-02 10:15:00\n"
            "dtype: datetime64[ns]"
        )

        with pd.option_context("display.width", 300):
            for idx, expected in zip(
                [idx1, idx2, idx3, idx4, idx5, idx6, idx7],
                [exp1, exp2, exp3, exp4, exp5, exp6, exp7],
            ):
                ser = Series(idx.as_unit(unit))
                result = repr(ser)
                assert result == expected.replace("[ns", f"[{unit}")

    def test_dti_summary(self):
        # GH#9116
        idx1 = DatetimeIndex([], freq="D")
        idx2 = DatetimeIndex(["2011-01-01"], freq="D")
        idx3 = DatetimeIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
            freq="h",
            tz="Asia/Tokyo",
        )
        idx6 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", NaT], tz="US/Eastern"
        )

        exp1 = "DatetimeIndex: 0 entries\nFreq: D"

        exp2 = "DatetimeIndex: 1 entries, 2011-01-01 to 2011-01-01\nFreq: D"

        exp3 = "DatetimeIndex: 2 entries, 2011-01-01 to 2011-01-02\nFreq: D"

        exp4 = "DatetimeIndex: 3 entries, 2011-01-01 to 2011-01-03\nFreq: D"

        exp5 = (
            "DatetimeIndex: 3 entries, 2011-01-01 09:00:00+09:00 "
            "to 2011-01-01 11:00:00+09:00\n"
            "Freq: h"
        )

        exp6 = """DatetimeIndex: 3 entries, 2011-01-01 09:00:00-05:00 to NaT"""

        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6], [exp1, exp2, exp3, exp4, exp5, exp6]
        ):
            result = idx._summary()
            assert result == expected

    @pytest.mark.parametrize("tz", [None, pytz.utc, dateutil.tz.tzutc()])
    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_dti_business_repr_etc_smoke(self, tz, freq):
        # only really care that it works
        dti = pd.bdate_range(
            datetime(2009, 1, 1), datetime(2010, 1, 1), tz=tz, freq=freq
        )
        repr(dti)
        dti._summary()
        dti[2:2]._summary()


class TestFormat:
    def test_format(self):
        # GH#35439
        idx = pd.date_range("20130101", periods=5)
        expected = [f"{x:%Y-%m-%d}" for x in idx]
        msg = r"DatetimeIndex\.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert idx.format() == expected

    def test_format_with_name_time_info(self):
        # bug I fixed 12/20/2011
        dates = pd.date_range("2011-01-01 04:00:00", periods=10, name="something")

        msg = "DatetimeIndex.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = dates.format(name=True)
        assert formatted[0] == "something"

    def test_format_datetime_with_time(self):
        dti = DatetimeIndex([datetime(2012, 2, 7), datetime(2012, 2, 7, 23)])

        msg = "DatetimeIndex.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = dti.format()
        expected = ["2012-02-07 00:00:00", "2012-02-07 23:00:00"]
        assert len(result) == 2
        assert result == expected

    def test_format_datetime(self):
        msg = "DatetimeIndex.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = pd.to_datetime([datetime(2003, 1, 1, 12), NaT]).format()
        assert formatted[0] == "2003-01-01 12:00:00"
        assert formatted[1] == "NaT"

    def test_format_date(self):
        msg = "DatetimeIndex.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = pd.to_datetime([datetime(2003, 1, 1), NaT]).format()
        assert formatted[0] == "2003-01-01"
        assert formatted[1] == "NaT"

    def test_format_date_tz(self):
        dti = pd.to_datetime([datetime(2013, 1, 1)], utc=True)
        msg = "DatetimeIndex.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = dti.format()
        assert formatted[0] == "2013-01-01 00:00:00+00:00"

        dti = pd.to_datetime([datetime(2013, 1, 1), NaT], utc=True)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = dti.format()
        assert formatted[0] == "2013-01-01 00:00:00+00:00"

    def test_format_date_explicit_date_format(self):
        dti = pd.to_datetime([datetime(2003, 2, 1), NaT])
        msg = "DatetimeIndex.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = dti.format(date_format="%m-%d-%Y", na_rep="UT")
        assert formatted[0] == "02-01-2003"
        assert formatted[1] == "UT"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_multilevel.py ===
import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm


class TestMultiLevel:
    def test_reindex_level(self, multiindex_year_month_day_dataframe_random_data):
        # axis=0
        ymd = multiindex_year_month_day_dataframe_random_data

        month_sums = ymd.groupby("month").sum()
        result = month_sums.reindex(ymd.index, level=1)
        expected = ymd.groupby(level="month").transform("sum")

        tm.assert_frame_equal(result, expected)

        # Series
        result = month_sums["A"].reindex(ymd.index, level=1)
        expected = ymd["A"].groupby(level="month").transform("sum")
        tm.assert_series_equal(result, expected, check_names=False)

        # axis=1
        msg = "DataFrame.groupby with axis=1 is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            gb = ymd.T.groupby("month", axis=1)

        month_sums = gb.sum()
        result = month_sums.reindex(columns=ymd.index, level=1)
        expected = ymd.groupby(level="month").transform("sum").T
        tm.assert_frame_equal(result, expected)

    def test_reindex(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        expected = frame.iloc[[0, 3]]
        reindexed = frame.loc[[("foo", "one"), ("bar", "one")]]
        tm.assert_frame_equal(reindexed, expected)

    def test_reindex_preserve_levels(
        self, multiindex_year_month_day_dataframe_random_data, using_copy_on_write
    ):
        ymd = multiindex_year_month_day_dataframe_random_data

        new_index = ymd.index[::10]
        chunk = ymd.reindex(new_index)
        if using_copy_on_write:
            assert chunk.index.is_(new_index)
        else:
            assert chunk.index is new_index

        chunk = ymd.loc[new_index]
        assert chunk.index.equals(new_index)

        ymdT = ymd.T
        chunk = ymdT.reindex(columns=new_index)
        if using_copy_on_write:
            assert chunk.columns.is_(new_index)
        else:
            assert chunk.columns is new_index

        chunk = ymdT.loc[:, new_index]
        assert chunk.columns.equals(new_index)

    def test_groupby_transform(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        s = frame["A"]
        grouper = s.index.get_level_values(0)

        grouped = s.groupby(grouper, group_keys=False)

        applied = grouped.apply(lambda x: x * 2)
        expected = grouped.transform(lambda x: x * 2)
        result = applied.reindex(expected.index)
        tm.assert_series_equal(result, expected, check_names=False)

    def test_groupby_corner(self):
        midx = MultiIndex(
            levels=[["foo"], ["bar"], ["baz"]],
            codes=[[0], [0], [0]],
            names=["one", "two", "three"],
        )
        df = DataFrame(
            [np.random.default_rng(2).random(4)],
            columns=["a", "b", "c", "d"],
            index=midx,
        )
        # should work
        df.groupby(level="three")

    def test_groupby_level_no_obs(self):
        # #1697
        midx = MultiIndex.from_tuples(
            [
                ("f1", "s1"),
                ("f1", "s2"),
                ("f2", "s1"),
                ("f2", "s2"),
                ("f3", "s1"),
                ("f3", "s2"),
            ]
        )
        df = DataFrame([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], columns=midx)
        df1 = df.loc(axis=1)[df.columns.map(lambda u: u[0] in ["f2", "f3"])]

        msg = "DataFrame.groupby with axis=1 is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            grouped = df1.groupby(axis=1, level=0)
        result = grouped.sum()
        assert (result.columns == ["f2", "f3"]).all()

    def test_setitem_with_expansion_multiindex_columns(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        ymd = multiindex_year_month_day_dataframe_random_data

        df = ymd[:5].T
        df[2000, 1, 10] = df[2000, 1, 7]
        assert isinstance(df.columns, MultiIndex)
        assert (df[2000, 1, 10] == df[2000, 1, 7]).all()

    def test_alignment(self):
        x = Series(
            data=[1, 2, 3], index=MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3)])
        )

        y = Series(
            data=[4, 5, 6], index=MultiIndex.from_tuples([("Z", 1), ("Z", 2), ("B", 3)])
        )

        res = x - y
        exp_index = x.index.union(y.index)
        exp = x.reindex(exp_index) - y.reindex(exp_index)
        tm.assert_series_equal(res, exp)

        # hit non-monotonic code path
        res = x[::-1] - y[::-1]
        exp_index = x.index.union(y.index)
        exp = x.reindex(exp_index) - y.reindex(exp_index)
        tm.assert_series_equal(res, exp)

    def test_groupby_multilevel(self, multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data

        result = ymd.groupby(level=[0, 1]).mean()

        k1 = ymd.index.get_level_values(0)
        k2 = ymd.index.get_level_values(1)

        expected = ymd.groupby([k1, k2]).mean()

        # TODO groupby with level_values drops names
        tm.assert_frame_equal(result, expected, check_names=False)
        assert result.index.names == ymd.index.names[:2]

        result2 = ymd.groupby(level=ymd.index.names[:2]).mean()
        tm.assert_frame_equal(result, result2)

    def test_multilevel_consolidate(self):
        index = MultiIndex.from_tuples(
            [("foo", "one"), ("foo", "two"), ("bar", "one"), ("bar", "two")]
        )
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), index=index, columns=index
        )
        df["Totals", ""] = df.sum(1)
        df = df._consolidate()

    def test_level_with_tuples(self):
        index = MultiIndex(
            levels=[[("foo", "bar", 0), ("foo", "baz", 0), ("foo", "qux", 0)], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        series = Series(np.random.default_rng(2).standard_normal(6), index=index)
        frame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)

        result = series[("foo", "bar", 0)]
        result2 = series.loc[("foo", "bar", 0)]
        expected = series[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        with pytest.raises(KeyError, match=r"^\(\('foo', 'bar', 0\), 2\)$"):
            series[("foo", "bar", 0), 2]

        result = frame.loc[("foo", "bar", 0)]
        result2 = frame.xs(("foo", "bar", 0))
        expected = frame[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        index = MultiIndex(
            levels=[[("foo", "bar"), ("foo", "baz"), ("foo", "qux")], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        series = Series(np.random.default_rng(2).standard_normal(6), index=index)
        frame = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=index)

        result = series[("foo", "bar")]
        result2 = series.loc[("foo", "bar")]
        expected = series[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        result = frame.loc[("foo", "bar")]
        result2 = frame.xs(("foo", "bar"))
        expected = frame[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_reindex_level_partial_selection(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        result = frame.reindex(["foo", "qux"], level=0)
        expected = frame.iloc[[0, 1, 2, 7, 8, 9]]
        tm.assert_frame_equal(result, expected)

        result = frame.T.reindex(["foo", "qux"], axis=1, level=0)
        tm.assert_frame_equal(result, expected.T)

        result = frame.loc[["foo", "qux"]]
        tm.assert_frame_equal(result, expected)

        result = frame["A"].loc[["foo", "qux"]]
        tm.assert_series_equal(result, expected["A"])

        result = frame.T.loc[:, ["foo", "qux"]]
        tm.assert_frame_equal(result, expected.T)

    @pytest.mark.parametrize("d", [4, "d"])
    def test_empty_frame_groupby_dtypes_consistency(self, d):
        # GH 20888
        group_keys = ["a", "b", "c"]
        df = DataFrame({"a": [1], "b": [2], "c": [3], "d": [d]})

        g = df[df.a == 2].groupby(group_keys)
        result = g.first().index
        expected = MultiIndex(
            levels=[[1], [2], [3]], codes=[[], [], []], names=["a", "b", "c"]
        )

        tm.assert_index_equal(result, expected)

    def test_duplicate_groupby_issues(self):
        idx_tp = [
            ("600809", "20061231"),
            ("600809", "20070331"),
            ("600809", "20070630"),
            ("600809", "20070331"),
        ]
        dt = ["demo", "demo", "demo", "demo"]

        idx = MultiIndex.from_tuples(idx_tp, names=["STK_ID", "RPT_Date"])
        s = Series(dt, index=idx)

        result = s.groupby(s.index).first()
        assert len(result) == 3

    def test_subsets_multiindex_dtype(self):
        # GH 20757
        data = [["x", 1]]
        columns = [("a", "b", np.nan), ("a", "c", 0.0)]
        df = DataFrame(data, columns=MultiIndex.from_tuples(columns))
        expected = df.dtypes.a.b
        result = df.a.b.dtypes
        tm.assert_series_equal(result, expected)

    def test_datetime_object_multiindex(self):
        data_dic = {
            (0, datetime.date(2018, 3, 3)): {"A": 1, "B": 10},
            (0, datetime.date(2018, 3, 4)): {"A": 2, "B": 11},
            (1, datetime.date(2018, 3, 3)): {"A": 3, "B": 12},
            (1, datetime.date(2018, 3, 4)): {"A": 4, "B": 13},
        }
        result = DataFrame.from_dict(data_dic, orient="index")
        data = {"A": [1, 2, 3, 4], "B": [10, 11, 12, 13]}
        index = [
            [0, 0, 1, 1],
            [
                datetime.date(2018, 3, 3),
                datetime.date(2018, 3, 4),
                datetime.date(2018, 3, 3),
                datetime.date(2018, 3, 4),
            ],
        ]
        expected = DataFrame(data=data, index=index)

        tm.assert_frame_equal(result, expected)

    def test_multiindex_with_na(self):
        df = DataFrame(
            [
                ["A", np.nan, 1.23, 4.56],
                ["A", "G", 1.23, 4.56],
                ["A", "D", 9.87, 10.54],
            ],
            columns=["pivot_0", "pivot_1", "col_1", "col_2"],
        ).set_index(["pivot_0", "pivot_1"])

        df.at[("A", "F"), "col_2"] = 0.0

        expected = DataFrame(
            [
                ["A", np.nan, 1.23, 4.56],
                ["A", "G", 1.23, 4.56],
                ["A", "D", 9.87, 10.54],
                ["A", "F", np.nan, 0.0],
            ],
            columns=["pivot_0", "pivot_1", "col_1", "col_2"],
        ).set_index(["pivot_0", "pivot_1"])

        tm.assert_frame_equal(df, expected)


class TestSorted:
    """everything you wanted to test about sorting"""

    def test_sort_non_lexsorted(self):
        # degenerate case where we sort but don't
        # have a satisfying result :<
        # GH 15797
        idx = MultiIndex(
            [["A", "B", "C"], ["c", "b", "a"]], [[0, 1, 2, 0, 1, 2], [0, 2, 1, 1, 0, 2]]
        )

        df = DataFrame({"col": range(len(idx))}, index=idx, dtype="int64")
        assert df.index.is_monotonic_increasing is False

        sorted = df.sort_index()
        assert sorted.index.is_monotonic_increasing is True

        expected = DataFrame(
            {"col": [1, 4, 5, 2]},
            index=MultiIndex.from_tuples(
                [("B", "a"), ("B", "c"), ("C", "a"), ("C", "b")]
            ),
            dtype="int64",
        )
        result = sorted.loc[pd.IndexSlice["B":"C", "a":"c"], :]
        tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\categorical\test_analytics.py ===
import re
import sys

import numpy as np
import pytest

from pandas.compat import PYPY

from pandas import (
    Categorical,
    CategoricalDtype,
    DataFrame,
    Index,
    NaT,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.api.types import is_scalar


class TestCategoricalAnalytics:
    @pytest.mark.parametrize("aggregation", ["min", "max"])
    def test_min_max_not_ordered_raises(self, aggregation):
        # unordered cats have no min/max
        cat = Categorical(["a", "b", "c", "d"], ordered=False)
        msg = f"Categorical is not ordered for operation {aggregation}"
        agg_func = getattr(cat, aggregation)

        with pytest.raises(TypeError, match=msg):
            agg_func()

        ufunc = np.minimum if aggregation == "min" else np.maximum
        with pytest.raises(TypeError, match=msg):
            ufunc.reduce(cat)

    def test_min_max_ordered(self, index_or_series_or_array):
        cat = Categorical(["a", "b", "c", "d"], ordered=True)
        obj = index_or_series_or_array(cat)
        _min = obj.min()
        _max = obj.max()
        assert _min == "a"
        assert _max == "d"

        assert np.minimum.reduce(obj) == "a"
        assert np.maximum.reduce(obj) == "d"
        # TODO: raises if we pass axis=0  (on Index and Categorical, not Series)

        cat = Categorical(
            ["a", "b", "c", "d"], categories=["d", "c", "b", "a"], ordered=True
        )
        obj = index_or_series_or_array(cat)
        _min = obj.min()
        _max = obj.max()
        assert _min == "d"
        assert _max == "a"
        assert np.minimum.reduce(obj) == "d"
        assert np.maximum.reduce(obj) == "a"

    def test_min_max_reduce(self):
        # GH52788
        cat = Categorical(["a", "b", "c", "d"], ordered=True)
        df = DataFrame(cat)

        result_max = df.agg("max")
        expected_max = Series(Categorical(["d"], dtype=cat.dtype))
        tm.assert_series_equal(result_max, expected_max)

        result_min = df.agg("min")
        expected_min = Series(Categorical(["a"], dtype=cat.dtype))
        tm.assert_series_equal(result_min, expected_min)

    @pytest.mark.parametrize(
        "categories,expected",
        [
            (list("ABC"), np.nan),
            ([1, 2, 3], np.nan),
            pytest.param(
                Series(date_range("2020-01-01", periods=3), dtype="category"),
                NaT,
                marks=pytest.mark.xfail(
                    reason="https://github.com/pandas-dev/pandas/issues/29962"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("aggregation", ["min", "max"])
    def test_min_max_ordered_empty(self, categories, expected, aggregation):
        # GH 30227
        cat = Categorical([], categories=categories, ordered=True)

        agg_func = getattr(cat, aggregation)
        result = agg_func()
        assert result is expected

    @pytest.mark.parametrize(
        "values, categories",
        [(["a", "b", "c", np.nan], list("cba")), ([1, 2, 3, np.nan], [3, 2, 1])],
    )
    @pytest.mark.parametrize("skipna", [True, False])
    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_with_nan(self, values, categories, function, skipna):
        # GH 25303
        cat = Categorical(values, categories=categories, ordered=True)
        result = getattr(cat, function)(skipna=skipna)

        if skipna is False:
            assert result is np.nan
        else:
            expected = categories[0] if function == "min" else categories[2]
            assert result == expected

    @pytest.mark.parametrize("function", ["min", "max"])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_min_max_only_nan(self, function, skipna):
        # https://github.com/pandas-dev/pandas/issues/33450
        cat = Categorical([np.nan], categories=[1, 2], ordered=True)
        result = getattr(cat, function)(skipna=skipna)
        assert result is np.nan

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_numeric_only_min_max_raises(self, method):
        # GH 25303
        cat = Categorical(
            [np.nan, 1, 2, np.nan], categories=[5, 4, 3, 2, 1], ordered=True
        )
        with pytest.raises(TypeError, match=".* got an unexpected keyword"):
            getattr(cat, method)(numeric_only=True)

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_numpy_min_max_raises(self, method):
        cat = Categorical(["a", "b", "c", "b"], ordered=False)
        msg = (
            f"Categorical is not ordered for operation {method}\n"
            "you can use .as_ordered() to change the Categorical to an ordered one"
        )
        method = getattr(np, method)
        with pytest.raises(TypeError, match=re.escape(msg)):
            method(cat)

    @pytest.mark.parametrize("kwarg", ["axis", "out", "keepdims"])
    @pytest.mark.parametrize("method", ["min", "max"])
    def test_numpy_min_max_unsupported_kwargs_raises(self, method, kwarg):
        cat = Categorical(["a", "b", "c", "b"], ordered=True)
        msg = (
            f"the '{kwarg}' parameter is not supported in the pandas implementation "
            f"of {method}"
        )
        if kwarg == "axis":
            msg = r"`axis` must be fewer than the number of dimensions \(1\)"
        kwargs = {kwarg: 42}
        method = getattr(np, method)
        with pytest.raises(ValueError, match=msg):
            method(cat, **kwargs)

    @pytest.mark.parametrize("method, expected", [("min", "a"), ("max", "c")])
    def test_numpy_min_max_axis_equals_none(self, method, expected):
        cat = Categorical(["a", "b", "c", "b"], ordered=True)
        method = getattr(np, method)
        result = method(cat, axis=None)
        assert result == expected

    @pytest.mark.parametrize(
        "values,categories,exp_mode",
        [
            ([1, 1, 2, 4, 5, 5, 5], [5, 4, 3, 2, 1], [5]),
            ([1, 1, 1, 4, 5, 5, 5], [5, 4, 3, 2, 1], [5, 1]),
            ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1]),
            ([np.nan, np.nan, np.nan, 4, 5], [5, 4, 3, 2, 1], [5, 4]),
            ([np.nan, np.nan, np.nan, 4, 5, 4], [5, 4, 3, 2, 1], [4]),
            ([np.nan, np.nan, 4, 5, 4], [5, 4, 3, 2, 1], [4]),
        ],
    )
    def test_mode(self, values, categories, exp_mode):
        cat = Categorical(values, categories=categories, ordered=True)
        res = Series(cat).mode()._values
        exp = Categorical(exp_mode, categories=categories, ordered=True)
        tm.assert_categorical_equal(res, exp)

    def test_searchsorted(self, ordered):
        # https://github.com/pandas-dev/pandas/issues/8420
        # https://github.com/pandas-dev/pandas/issues/14522

        cat = Categorical(
            ["cheese", "milk", "apple", "bread", "bread"],
            categories=["cheese", "milk", "apple", "bread"],
            ordered=ordered,
        )
        ser = Series(cat)

        # Searching for single item argument, side='left' (default)
        res_cat = cat.searchsorted("apple")
        assert res_cat == 2
        assert is_scalar(res_cat)

        res_ser = ser.searchsorted("apple")
        assert res_ser == 2
        assert is_scalar(res_ser)

        # Searching for single item array, side='left' (default)
        res_cat = cat.searchsorted(["bread"])
        res_ser = ser.searchsorted(["bread"])
        exp = np.array([3], dtype=np.intp)
        tm.assert_numpy_array_equal(res_cat, exp)
        tm.assert_numpy_array_equal(res_ser, exp)

        # Searching for several items array, side='right'
        res_cat = cat.searchsorted(["apple", "bread"], side="right")
        res_ser = ser.searchsorted(["apple", "bread"], side="right")
        exp = np.array([3, 5], dtype=np.intp)
        tm.assert_numpy_array_equal(res_cat, exp)
        tm.assert_numpy_array_equal(res_ser, exp)

        # Searching for a single value that is not from the Categorical
        with pytest.raises(TypeError, match="cucumber"):
            cat.searchsorted("cucumber")
        with pytest.raises(TypeError, match="cucumber"):
            ser.searchsorted("cucumber")

        # Searching for multiple values one of each is not from the Categorical
        msg = (
            "Cannot setitem on a Categorical with a new category, "
            "set the categories first"
        )
        with pytest.raises(TypeError, match=msg):
            cat.searchsorted(["bread", "cucumber"])
        with pytest.raises(TypeError, match=msg):
            ser.searchsorted(["bread", "cucumber"])

    def test_unique(self, ordered):
        # GH38140
        dtype = CategoricalDtype(["a", "b", "c"], ordered=ordered)

        # categories are reordered based on value when ordered=False
        cat = Categorical(["a", "b", "c"], dtype=dtype)
        res = cat.unique()
        tm.assert_categorical_equal(res, cat)

        cat = Categorical(["a", "b", "a", "a"], dtype=dtype)
        res = cat.unique()
        tm.assert_categorical_equal(res, Categorical(["a", "b"], dtype=dtype))

        cat = Categorical(["c", "a", "b", "a", "a"], dtype=dtype)
        res = cat.unique()
        exp_cat = Categorical(["c", "a", "b"], dtype=dtype)
        tm.assert_categorical_equal(res, exp_cat)

        # nan must be removed
        cat = Categorical(["b", np.nan, "b", np.nan, "a"], dtype=dtype)
        res = cat.unique()
        exp_cat = Categorical(["b", np.nan, "a"], dtype=dtype)
        tm.assert_categorical_equal(res, exp_cat)

    def test_unique_index_series(self, ordered):
        # GH38140
        dtype = CategoricalDtype([3, 2, 1], ordered=ordered)

        c = Categorical([3, 1, 2, 2, 1], dtype=dtype)
        # Categorical.unique sorts categories by appearance order
        # if ordered=False
        exp = Categorical([3, 1, 2], dtype=dtype)
        tm.assert_categorical_equal(c.unique(), exp)

        tm.assert_index_equal(Index(c).unique(), Index(exp))
        tm.assert_categorical_equal(Series(c).unique(), exp)

        c = Categorical([1, 1, 2, 2], dtype=dtype)
        exp = Categorical([1, 2], dtype=dtype)
        tm.assert_categorical_equal(c.unique(), exp)
        tm.assert_index_equal(Index(c).unique(), Index(exp))
        tm.assert_categorical_equal(Series(c).unique(), exp)

    def test_shift(self):
        # GH 9416
        cat = Categorical(["a", "b", "c", "d", "a"])

        # shift forward
        sp1 = cat.shift(1)
        xp1 = Categorical([np.nan, "a", "b", "c", "d"])
        tm.assert_categorical_equal(sp1, xp1)
        tm.assert_categorical_equal(cat[:-1], sp1[1:])

        # shift back
        sn2 = cat.shift(-2)
        xp2 = Categorical(
            ["c", "d", "a", np.nan, np.nan], categories=["a", "b", "c", "d"]
        )
        tm.assert_categorical_equal(sn2, xp2)
        tm.assert_categorical_equal(cat[2:], sn2[:-2])

        # shift by zero
        tm.assert_categorical_equal(cat, cat.shift(0))

    def test_nbytes(self):
        cat = Categorical([1, 2, 3])
        exp = 3 + 3 * 8  # 3 int8s for values + 3 int64s for categories
        assert cat.nbytes == exp

    def test_memory_usage(self, using_infer_string):
        cat = Categorical([1, 2, 3])

        # .categories is an index, so we include the hashtable
        assert 0 < cat.nbytes <= cat.memory_usage()
        assert 0 < cat.nbytes <= cat.memory_usage(deep=True)

        cat = Categorical(["foo", "foo", "bar"])
        if using_infer_string:
            if cat.categories.dtype.storage == "python":
                assert cat.memory_usage(deep=True) > cat.nbytes
            else:
                assert cat.memory_usage(deep=True) >= cat.nbytes
        else:
            assert cat.memory_usage(deep=True) > cat.nbytes

        if not PYPY:
            # sys.getsizeof will call the .memory_usage with
            # deep=True, and add on some GC overhead
            diff = cat.memory_usage(deep=True) - sys.getsizeof(cat)
            assert abs(diff) < 100

    def test_map(self):
        c = Categorical(list("ABABC"), categories=list("CBA"), ordered=True)
        result = c.map(lambda x: x.lower(), na_action=None)
        exp = Categorical(list("ababc"), categories=list("cba"), ordered=True)
        tm.assert_categorical_equal(result, exp)

        c = Categorical(list("ABABC"), categories=list("ABC"), ordered=False)
        result = c.map(lambda x: x.lower(), na_action=None)
        exp = Categorical(list("ababc"), categories=list("abc"), ordered=False)
        tm.assert_categorical_equal(result, exp)

        result = c.map(lambda x: 1, na_action=None)
        # GH 12766: Return an index not an array
        tm.assert_index_equal(result, Index(np.array([1] * 5, dtype=np.int64)))

    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_inplace_raises(self, value):
        cat = Categorical(["A", "B", "B", "C", "A"])
        msg = (
            'For argument "inplace" expected type bool, '
            f"received type {type(value).__name__}"
        )

        with pytest.raises(ValueError, match=msg):
            cat.sort_values(inplace=value)

    def test_quantile_empty(self):
        # make sure we have correct itemsize on resulting codes
        cat = Categorical(["A", "B"])
        idx = Index([0.0, 0.5])
        result = cat[:0]._quantile(idx, interpolation="linear")
        assert result._codes.dtype == np.int8

        expected = cat.take([-1, -1], allow_fill=True)
        tm.assert_extension_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\tests\test_printing.py ===
# -*- coding: utf-8 -*-

from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
                                           vsprint, vsstrrepr, vlatex)


a, b, c = symbols('a, b, c')
alpha, omega, beta = dynamicsymbols('alpha, omega, beta')

A = ReferenceFrame('A')
N = ReferenceFrame('N')

v = a ** 2 * N.x + b * N.y + c * sin(alpha) * N.z
w = alpha * N.x + sin(omega) * N.y + alpha * beta * N.z
ww = alpha * N.x + asin(omega) * N.y - alpha.diff() * beta * N.z
o = a/b * N.x + (c+b)/a * N.y + c**2/b * N.z

y = a ** 2 * (N.x | N.y) + b * (N.y | N.y) + c * sin(alpha) * (N.z | N.y)
x = alpha * (N.x | N.x) + sin(omega) * (N.y | N.z) + alpha * beta * (N.z | N.x)
xx = N.x | (-N.y - N.z)
xx2 = N.x | (N.y + N.z)

def ascii_vpretty(expr):
    return vpprint(expr, use_unicode=False, wrap_line=False)


def unicode_vpretty(expr):
    return vpprint(expr, use_unicode=True, wrap_line=False)


def test_latex_printer():
    r = Function('r')('t')
    assert VectorLatexPrinter().doprint(r ** 2) == "r^{2}"
    r2 = Function('r^2')('t')
    assert VectorLatexPrinter().doprint(r2.diff()) == r'\dot{r^{2}}'
    ra = Function('r__a')('t')
    assert VectorLatexPrinter().doprint(ra.diff().diff()) == r'\ddot{r^{a}}'


def test_vector_pretty_print():

    # TODO : The unit vectors should print with subscripts but they just
    # print as `n_x` instead of making `x` a subscript with unicode.

    # TODO : The pretty print division does not print correctly here:
    # w = alpha * N.x + sin(omega) * N.y + alpha / beta * N.z

    expected = """\
 2                               \n\
a  n_x + b n_y + c*sin(alpha) n_z\
"""
    uexpected = """\
 2                           \n\
a  n_x + b n_y + c⋅sin(α) n_z\
"""

    assert ascii_vpretty(v) == expected
    assert unicode_vpretty(v) == uexpected

    expected = 'alpha n_x + sin(omega) n_y + alpha*beta n_z'
    uexpected = 'α n_x + sin(ω) n_y + α⋅β n_z'

    assert ascii_vpretty(w) == expected
    assert unicode_vpretty(w) == uexpected

    expected = """\
                     2    \n\
a       b + c       c     \n\
- n_x + ----- n_y + -- n_z\n\
b         a         b     \
"""
    uexpected = """\
                     2    \n\
a       b + c       c     \n\
─ n_x + ───── n_y + ── n_z\n\
b         a         b     \
"""

    assert ascii_vpretty(o) == expected
    assert unicode_vpretty(o) == uexpected

    # https://github.com/sympy/sympy/issues/26731
    assert ascii_vpretty(-A.x) == '-a_x'
    assert unicode_vpretty(-A.x) == '-a_x'

    # https://github.com/sympy/sympy/issues/26799
    assert ascii_vpretty(0*A.x) == '0'
    assert unicode_vpretty(0*A.x) == '0'


def test_vector_latex():

    a, b, c, d, omega = symbols('a, b, c, d, omega')

    v = (a ** 2 + b / c) * A.x + sqrt(d) * A.y + cos(omega) * A.z

    assert vlatex(v) == (r'(a^{2} + \frac{b}{c})\mathbf{\hat{a}_x} + '
                         r'\sqrt{d}\mathbf{\hat{a}_y} + '
                         r'\cos{\left(\omega \right)}'
                         r'\mathbf{\hat{a}_z}')

    theta, omega, alpha, q = dynamicsymbols('theta, omega, alpha, q')

    v = theta * A.x + omega * omega * A.y + (q * alpha) * A.z

    assert vlatex(v) == (r'\theta\mathbf{\hat{a}_x} + '
                         r'\omega^{2}\mathbf{\hat{a}_y} + '
                         r'\alpha q\mathbf{\hat{a}_z}')

    phi1, phi2, phi3 = dynamicsymbols('phi1, phi2, phi3')
    theta1, theta2, theta3 = symbols('theta1, theta2, theta3')

    v = (sin(theta1) * A.x +
         cos(phi1) * cos(phi2) * A.y +
         cos(theta1 + phi3) * A.z)

    assert vlatex(v) == (r'\sin{\left(\theta_{1} \right)}'
                         r'\mathbf{\hat{a}_x} + \cos{'
                         r'\left(\phi_{1} \right)} \cos{'
                         r'\left(\phi_{2} \right)}\mathbf{\hat{a}_y} + '
                         r'\cos{\left(\theta_{1} + '
                         r'\phi_{3} \right)}\mathbf{\hat{a}_z}')

    N = ReferenceFrame('N')

    a, b, c, d, omega = symbols('a, b, c, d, omega')

    v = (a ** 2 + b / c) * N.x + sqrt(d) * N.y + cos(omega) * N.z

    expected = (r'(a^{2} + \frac{b}{c})\mathbf{\hat{n}_x} + '
                r'\sqrt{d}\mathbf{\hat{n}_y} + '
                r'\cos{\left(\omega \right)}'
                r'\mathbf{\hat{n}_z}')

    assert vlatex(v) == expected

    # Try custom unit vectors.

    N = ReferenceFrame('N', latexs=(r'\hat{i}', r'\hat{j}', r'\hat{k}'))

    v = (a ** 2 + b / c) * N.x + sqrt(d) * N.y + cos(omega) * N.z

    expected = (r'(a^{2} + \frac{b}{c})\hat{i} + '
                r'\sqrt{d}\hat{j} + '
                r'\cos{\left(\omega \right)}\hat{k}')
    assert vlatex(v) == expected

    expected = r'\alpha\mathbf{\hat{n}_x} + \operatorname{asin}{\left(\omega ' \
        r'\right)}\mathbf{\hat{n}_y} -  \beta \dot{\alpha}\mathbf{\hat{n}_z}'
    assert vlatex(ww) == expected

    expected = r'- \mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} - ' \
        r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_z}'
    assert vlatex(xx) == expected

    expected = r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} + ' \
        r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_z}'
    assert vlatex(xx2) == expected


def test_vector_latex_arguments():
    assert vlatex(N.x * 3.0, full_prec=False) == r'3.0\mathbf{\hat{n}_x}'
    assert vlatex(N.x * 3.0, full_prec=True) == r'3.00000000000000\mathbf{\hat{n}_x}'


def test_vector_latex_with_functions():

    N = ReferenceFrame('N')

    omega, alpha = dynamicsymbols('omega, alpha')

    v = omega.diff() * N.x

    assert vlatex(v) == r'\dot{\omega}\mathbf{\hat{n}_x}'

    v = omega.diff() ** alpha * N.x

    assert vlatex(v) == (r'\dot{\omega}^{\alpha}'
                          r'\mathbf{\hat{n}_x}')


def test_dyadic_pretty_print():

    expected = """\
 2
a  n_x|n_y + b n_y|n_y + c*sin(alpha) n_z|n_y\
"""

    uexpected = """\
 2
a  n_x⊗n_y + b n_y⊗n_y + c⋅sin(α) n_z⊗n_y\
"""
    assert ascii_vpretty(y) == expected
    assert unicode_vpretty(y) == uexpected

    expected = 'alpha n_x|n_x + sin(omega) n_y|n_z + alpha*beta n_z|n_x'
    uexpected = 'α n_x⊗n_x + sin(ω) n_y⊗n_z + α⋅β n_z⊗n_x'
    assert ascii_vpretty(x) == expected
    assert unicode_vpretty(x) == uexpected

    assert ascii_vpretty(Dyadic([])) == '0'
    assert unicode_vpretty(Dyadic([])) == '0'

    assert ascii_vpretty(xx) == '- n_x|n_y - n_x|n_z'
    assert unicode_vpretty(xx) == '- n_x⊗n_y - n_x⊗n_z'

    assert ascii_vpretty(xx2) == 'n_x|n_y + n_x|n_z'
    assert unicode_vpretty(xx2) == 'n_x⊗n_y + n_x⊗n_z'


def test_dyadic_latex():

    expected = (r'a^{2}\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} + '
                r'b\mathbf{\hat{n}_y}\otimes \mathbf{\hat{n}_y} + '
                r'c \sin{\left(\alpha \right)}'
                r'\mathbf{\hat{n}_z}\otimes \mathbf{\hat{n}_y}')

    assert vlatex(y) == expected

    expected = (r'\alpha\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_x} + '
                r'\sin{\left(\omega \right)}\mathbf{\hat{n}_y}'
                r'\otimes \mathbf{\hat{n}_z} + '
                r'\alpha \beta\mathbf{\hat{n}_z}\otimes \mathbf{\hat{n}_x}')

    assert vlatex(x) == expected

    assert vlatex(Dyadic([])) == '0'


def test_dyadic_str():
    assert vsprint(Dyadic([])) == '0'
    assert vsprint(y) == 'a**2*(N.x|N.y) + b*(N.y|N.y) + c*sin(alpha)*(N.z|N.y)'
    assert vsprint(x) == 'alpha*(N.x|N.x) + sin(omega)*(N.y|N.z) + alpha*beta*(N.z|N.x)'
    assert vsprint(ww) == "alpha*N.x + asin(omega)*N.y - beta*alpha'*N.z"
    assert vsprint(xx) == '- (N.x|N.y) - (N.x|N.z)'
    assert vsprint(xx2) == '(N.x|N.y) + (N.x|N.z)'


def test_vlatex(): # vlatex is broken #12078
    from sympy.physics.vector import vlatex

    x = symbols('x')
    J = symbols('J')

    f = Function('f')
    g = Function('g')
    h = Function('h')

    expected = r'J \left(\frac{d}{d x} g{\left(x \right)} - \frac{d}{d x} h{\left(x \right)}\right)'

    expr = J*f(x).diff(x).subs(f(x), g(x)-h(x))

    assert vlatex(expr) == expected


def test_issue_13354():
    """
    Test for proper pretty printing of physics vectors with ADD
    instances in arguments.

    Test is exactly the one suggested in the original bug report by
    @moorepants.
    """

    a, b, c = symbols('a, b, c')
    A = ReferenceFrame('A')
    v = a * A.x + b * A.y + c * A.z
    w = b * A.x + c * A.y + a * A.z
    z = w + v

    expected = """(a + b) a_x + (b + c) a_y + (a + c) a_z"""

    assert ascii_vpretty(z) == expected


def test_vector_derivative_printing():
    # First order
    v = omega.diff() * N.x
    assert unicode_vpretty(v) == 'ω̇ n_x'
    assert ascii_vpretty(v) == "omega'(t) n_x"

    # Second order
    v = omega.diff().diff() * N.x

    assert vlatex(v) == r'\ddot{\omega}\mathbf{\hat{n}_x}'
    assert unicode_vpretty(v) == 'ω̈ n_x'
    assert ascii_vpretty(v) == "omega''(t) n_x"

    # Third order
    v = omega.diff().diff().diff() * N.x

    assert vlatex(v) == r'\dddot{\omega}\mathbf{\hat{n}_x}'
    assert unicode_vpretty(v) == 'ω⃛ n_x'
    assert ascii_vpretty(v) == "omega'''(t) n_x"

    # Fourth order
    v = omega.diff().diff().diff().diff() * N.x

    assert vlatex(v) == r'\ddddot{\omega}\mathbf{\hat{n}_x}'
    assert unicode_vpretty(v) == 'ω⃜ n_x'
    assert ascii_vpretty(v) == "omega''''(t) n_x"

    # Fifth order
    v = omega.diff().diff().diff().diff().diff() * N.x

    assert vlatex(v) == r'\frac{d^{5}}{d t^{5}} \omega\mathbf{\hat{n}_x}'
    expected = '''\
 5            \n\
d             \n\
---(omega) n_x\n\
  5           \n\
dt            \
'''
    uexpected = '''\
 5        \n\
d         \n\
───(ω) n_x\n\
  5       \n\
dt        \
'''
    assert unicode_vpretty(v) == uexpected
    assert ascii_vpretty(v) == expected


def test_vector_str_printing():
    assert vsprint(w) == 'alpha*N.x + sin(omega)*N.y + alpha*beta*N.z'
    assert vsprint(omega.diff() * N.x) == "omega'*N.x"
    assert vsstrrepr(w) == 'alpha*N.x + sin(omega)*N.y + alpha*beta*N.z'


def test_vector_str_arguments():
    assert vsprint(N.x * 3.0, full_prec=False) == '3.0*N.x'
    assert vsprint(N.x * 3.0, full_prec=True) == '3.00000000000000*N.x'


def test_issue_14041():
    import sympy.physics.mechanics as me

    A_frame = me.ReferenceFrame('A')
    thetad, phid = me.dynamicsymbols('theta, phi', 1)
    L = symbols('L')

    assert vlatex(L*(phid + thetad)**2*A_frame.x) == \
        r"L \left(\dot{\phi} + \dot{\theta}\right)^{2}\mathbf{\hat{a}_x}"
    assert vlatex((phid + thetad)**2*A_frame.x) == \
        r"\left(\dot{\phi} + \dot{\theta}\right)^{2}\mathbf{\hat{a}_x}"
    assert vlatex((phid*thetad)**a*A_frame.x) == \
        r"\left(\dot{\phi} \dot{\theta}\right)^{a}\mathbf{\hat{a}_x}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_fields.py ===
"""Test sparse rational functions. """

from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex

from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt

def test_FracField___init__():
    F1 = FracField("x,y", ZZ, lex)
    F2 = FracField("x,y", ZZ, lex)
    F3 = FracField("x,y,z", ZZ, lex)

    assert F1.x == F1.gens[0]
    assert F1.y == F1.gens[1]
    assert F1.x == F2.x
    assert F1.y == F2.y
    assert F1.x != F3.x
    assert F1.y != F3.y

def test_FracField___hash__():
    F, x, y, z = field("x,y,z", QQ)
    assert hash(F)

def test_FracField___eq__():
    assert field("x,y,z", QQ)[0] == field("x,y,z", QQ)[0]
    assert field("x,y,z", QQ)[0] != field("x,y,z", ZZ)[0]
    assert field("x,y,z", ZZ)[0] != field("x,y,z", QQ)[0]
    assert field("x,y,z", QQ)[0] != field("x,y", QQ)[0]
    assert field("x,y", QQ)[0] != field("x,y,z", QQ)[0]

def test_sfield():
    x = symbols("x")

    F = FracField((E, exp(exp(x)), exp(x)), ZZ, lex)
    e, exex, ex = F.gens
    assert sfield(exp(x)*exp(exp(x) + 1 + log(exp(x) + 3)/2)**2/(exp(x) + 3)) \
        == (F, e**2*exex**2*ex)

    F = FracField((x, exp(1/x), log(x), x**QQ(1, 3)), ZZ, lex)
    _, ex, lg, x3 = F.gens
    assert sfield(((x-3)*log(x)+4*x**2)*exp(1/x+log(x)/3)/x**2) == \
        (F, (4*F.x**2*ex + F.x*ex*lg - 3*ex*lg)/x3**5)

    F = FracField((x, log(x), sqrt(x + log(x))), ZZ, lex)
    _, lg, srt = F.gens
    assert sfield((x + 1) / (x * (x + log(x))**QQ(3, 2)) - 1/(x * log(x)**2)) \
        == (F, (F.x*lg**2 - F.x*srt + lg**2 - lg*srt)/
            (F.x**2*lg**2*srt + F.x*lg**3*srt))

def test_FracElement___hash__():
    F, x, y, z = field("x,y,z", QQ)
    assert hash(x*y/z)

def test_FracElement_copy():
    F, x, y, z = field("x,y,z", ZZ)

    f = x*y/3*z
    g = f.copy()

    assert f == g
    g.numer[(1, 1, 1)] = 7
    assert f != g

def test_FracElement_as_expr():
    F, x, y, z = field("x,y,z", ZZ)
    f = (3*x**2*y - x*y*z)/(7*z**3 + 1)

    X, Y, Z = F.symbols
    g = (3*X**2*Y - X*Y*Z)/(7*Z**3 + 1)

    assert f != g
    assert f.as_expr() == g

    X, Y, Z = symbols("x,y,z")
    g = (3*X**2*Y - X*Y*Z)/(7*Z**3 + 1)

    assert f != g
    assert f.as_expr(X, Y, Z) == g

    raises(ValueError, lambda: f.as_expr(X))

def test_FracElement_from_expr():
    x, y, z = symbols("x,y,z")
    F, X, Y, Z = field((x, y, z), ZZ)

    f = F.from_expr(1)
    assert f == 1 and F.is_element(f)

    f = F.from_expr(Rational(3, 7))
    assert f == F(3)/7 and F.is_element(f)

    f = F.from_expr(x)
    assert f == X and F.is_element(f)

    f = F.from_expr(Rational(3,7)*x)
    assert f == X*Rational(3, 7) and F.is_element(f)

    f = F.from_expr(1/x)
    assert f == 1/X and F.is_element(f)

    f = F.from_expr(x*y*z)
    assert f == X*Y*Z and F.is_element(f)

    f = F.from_expr(x*y/z)
    assert f == X*Y/Z and F.is_element(f)

    f = F.from_expr(x*y*z + x*y + x)
    assert f == X*Y*Z + X*Y + X and F.is_element(f)

    f = F.from_expr((x*y*z + x*y + x)/(x*y + 7))
    assert f == (X*Y*Z + X*Y + X)/(X*Y + 7) and F.is_element(f)

    f = F.from_expr(x**3*y*z + x**2*y**7 + 1)
    assert f == X**3*Y*Z + X**2*Y**7 + 1 and F.is_element(f)

    raises(ValueError, lambda: F.from_expr(2**x))
    raises(ValueError, lambda: F.from_expr(7*x + sqrt(2)))

    assert isinstance(ZZ[2**x].get_field().convert(2**(-x)),
        FracElement)
    assert isinstance(ZZ[x**2].get_field().convert(x**(-6)),
        FracElement)
    assert isinstance(ZZ[exp(Rational(1, 3))].get_field().convert(E),
        FracElement)


def test_FracField_nested():
    a, b, x = symbols('a b x')
    F1 = ZZ.frac_field(a, b)
    F2 = F1.frac_field(x)
    frac = F2(a + b)
    assert frac.numer == F1.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F1(a + b)]
    assert frac.denom == F1.poly_ring(x)(1)

    F3 = ZZ.poly_ring(a, b)
    F4 = F3.frac_field(x)
    frac = F4(a + b)
    assert frac.numer == F3.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F3(a + b)]
    assert frac.denom == F3.poly_ring(x)(1)

    frac = F2(F3(a + b))
    assert frac.numer == F1.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F1(a + b)]
    assert frac.denom == F1.poly_ring(x)(1)

    frac = F4(F1(a + b))
    assert frac.numer == F3.poly_ring(x)(a + b)
    assert frac.numer.coeffs() == [F3(a + b)]
    assert frac.denom == F3.poly_ring(x)(1)


def test_FracElement__lt_le_gt_ge__():
    F, x, y = field("x,y", ZZ)

    assert F(1) < 1/x < 1/x**2 < 1/x**3
    assert F(1) <= 1/x <= 1/x**2 <= 1/x**3

    assert -7/x < 1/x < 3/x < y/x < 1/x**2
    assert -7/x <= 1/x <= 3/x <= y/x <= 1/x**2

    assert 1/x**3 > 1/x**2 > 1/x > F(1)
    assert 1/x**3 >= 1/x**2 >= 1/x >= F(1)

    assert 1/x**2 > y/x > 3/x > 1/x > -7/x
    assert 1/x**2 >= y/x >= 3/x >= 1/x >= -7/x

def test_FracElement___neg__():
    F, x,y = field("x,y", QQ)

    f = (7*x - 9)/y
    g = (-7*x + 9)/y

    assert -f == g
    assert -g == f

def test_FracElement___add__():
    F, x,y = field("x,y", QQ)

    f, g = 1/x, 1/y
    assert f + g == g + f == (x + y)/(x*y)

    assert x + F.ring.gens[0] == F.ring.gens[0] + x == 2*x

    F, x,y = field("x,y", ZZ)
    assert x + 3 == 3 + x
    assert x + QQ(3,7) == QQ(3,7) + x == (7*x + 3)/7

    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)

    f = (u*v + x)/(y + u*v)
    assert dict(f.numer) == {(1, 0, 0, 0): 1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0): u*v}

    Ruv, u,v = ring("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Ruv)

    f = (u*v + x)/(y + u*v)
    assert dict(f.numer) == {(1, 0, 0, 0): 1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0): u*v}

def test_FracElement___sub__():
    F, x,y = field("x,y", QQ)

    f, g = 1/x, 1/y
    assert f - g == (-x + y)/(x*y)

    assert x - F.ring.gens[0] == F.ring.gens[0] - x == 0

    F, x,y = field("x,y", ZZ)
    assert x - 3 == -(3 - x)
    assert x - QQ(3,7) == -(QQ(3,7) - x) == (7*x - 3)/7

    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)

    f = (u*v - x)/(y - u*v)
    assert dict(f.numer) == {(1, 0, 0, 0):-1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0):-u*v}

    Ruv, u,v = ring("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Ruv)

    f = (u*v - x)/(y - u*v)
    assert dict(f.numer) == {(1, 0, 0, 0):-1, (0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(0, 1, 0, 0): 1, (0, 0, 0, 0):-u*v}

def test_FracElement___mul__():
    F, x,y = field("x,y", QQ)

    f, g = 1/x, 1/y
    assert f*g == g*f == 1/(x*y)

    assert x*F.ring.gens[0] == F.ring.gens[0]*x == x**2

    F, x,y = field("x,y", ZZ)
    assert x*3 == 3*x
    assert x*QQ(3,7) == QQ(3,7)*x == x*Rational(3, 7)

    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)

    f = ((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)
    assert dict(f.numer) == {(1, 1, 0, 0): u + 1, (0, 0, 0, 0): 1}
    assert dict(f.denom) == {(0, 0, 1, 0): v - 1, (0, 0, 0, 1): -u*v, (0, 0, 0, 0): -1}

    Ruv, u,v = ring("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Ruv)

    f = ((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)
    assert dict(f.numer) == {(1, 1, 0, 0): u + 1, (0, 0, 0, 0): 1}
    assert dict(f.denom) == {(0, 0, 1, 0): v - 1, (0, 0, 0, 1): -u*v, (0, 0, 0, 0): -1}

def test_FracElement___truediv__():
    F, x,y = field("x,y", QQ)

    f, g = 1/x, 1/y
    assert f/g == y/x

    assert x/F.ring.gens[0] == F.ring.gens[0]/x == 1

    F, x,y = field("x,y", ZZ)
    assert x*3 == 3*x
    assert x/QQ(3,7) == (QQ(3,7)/x)**-1 == x*Rational(7, 3)

    raises(ZeroDivisionError, lambda: x/0)
    raises(ZeroDivisionError, lambda: 1/(x - x))
    raises(ZeroDivisionError, lambda: x/(x - x))

    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)

    f = (u*v)/(x*y)
    assert dict(f.numer) == {(0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(1, 1, 0, 0): 1}

    g = (x*y)/(u*v)
    assert dict(g.numer) == {(1, 1, 0, 0): 1}
    assert dict(g.denom) == {(0, 0, 0, 0): u*v}

    Ruv, u,v = ring("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Ruv)

    f = (u*v)/(x*y)
    assert dict(f.numer) == {(0, 0, 0, 0): u*v}
    assert dict(f.denom) == {(1, 1, 0, 0): 1}

    g = (x*y)/(u*v)
    assert dict(g.numer) == {(1, 1, 0, 0): 1}
    assert dict(g.denom) == {(0, 0, 0, 0): u*v}

def test_FracElement___pow__():
    F, x,y = field("x,y", QQ)

    f, g = 1/x, 1/y

    assert f**3 == 1/x**3
    assert g**3 == 1/y**3

    assert (f*g)**3 == 1/(x**3*y**3)
    assert (f*g)**-3 == (x*y)**3

    raises(ZeroDivisionError, lambda: (x - x)**-3)

def test_FracElement_diff():
    F, x,y,z = field("x,y,z", ZZ)

    assert ((x**2 + y)/(z + 1)).diff(x) == 2*x/(z + 1)

@XFAIL
def test_FracElement___call__():
    F, x,y,z = field("x,y,z", ZZ)
    f = (x**2 + 3*y)/z

    r = f(1, 1, 1)
    assert r == 4 and not isinstance(r, FracElement)
    raises(ZeroDivisionError, lambda: f(1, 1, 0))

def test_FracElement_evaluate():
    F, x,y,z = field("x,y,z", ZZ)
    Fyz = field("y,z", ZZ)[0]
    f = (x**2 + 3*y)/z

    assert f.evaluate(x, 0) == 3*Fyz.y/Fyz.z
    raises(ZeroDivisionError, lambda: f.evaluate(z, 0))

def test_FracElement_subs():
    F, x,y,z = field("x,y,z", ZZ)
    f = (x**2 + 3*y)/z

    assert f.subs(x, 0) == 3*y/z
    raises(ZeroDivisionError, lambda: f.subs(z, 0))

def test_FracElement_compose():
    pass

def test_FracField_index():
    a = symbols("a")
    F, x, y, z = field('x y z', QQ)
    assert F.index(x) == 0
    assert F.index(y) == 1

    raises(ValueError, lambda: F.index(1))
    raises(ValueError, lambda: F.index(a))
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\tests\test_app.py ===
# -*- coding: utf-8 -*-
#
import os
import os.path
import ssl
import threading
import unittest

import websocket as ws

"""
test_app.py
websocket - WebSocket client library for Python

Copyright 2024 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Skip test to access the internet unless TEST_WITH_INTERNET == 1
TEST_WITH_INTERNET = os.environ.get("TEST_WITH_INTERNET", "0") == "1"
# Skip tests relying on local websockets server unless LOCAL_WS_SERVER_PORT != -1
LOCAL_WS_SERVER_PORT = os.environ.get("LOCAL_WS_SERVER_PORT", "-1")
TEST_WITH_LOCAL_SERVER = LOCAL_WS_SERVER_PORT != "-1"
TRACEABLE = True


class WebSocketAppTest(unittest.TestCase):
    class NotSetYet:
        """A marker class for signalling that a value hasn't been set yet."""

    def setUp(self):
        ws.enableTrace(TRACEABLE)

        WebSocketAppTest.keep_running_open = WebSocketAppTest.NotSetYet()
        WebSocketAppTest.keep_running_close = WebSocketAppTest.NotSetYet()
        WebSocketAppTest.get_mask_key_id = WebSocketAppTest.NotSetYet()
        WebSocketAppTest.on_error_data = WebSocketAppTest.NotSetYet()

    def tearDown(self):
        WebSocketAppTest.keep_running_open = WebSocketAppTest.NotSetYet()
        WebSocketAppTest.keep_running_close = WebSocketAppTest.NotSetYet()
        WebSocketAppTest.get_mask_key_id = WebSocketAppTest.NotSetYet()
        WebSocketAppTest.on_error_data = WebSocketAppTest.NotSetYet()

    def close(self):
        pass

    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_keep_running(self):
        """A WebSocketApp should keep running as long as its self.keep_running
        is not False (in the boolean context).
        """

        def on_open(self, *args, **kwargs):
            """Set the keep_running flag for later inspection and immediately
            close the connection.
            """
            self.send("hello!")
            WebSocketAppTest.keep_running_open = self.keep_running
            self.keep_running = False

        def on_message(_, message):
            print(message)
            self.close()

        def on_close(self, *args, **kwargs):
            """Set the keep_running flag for the test to use."""
            WebSocketAppTest.keep_running_close = self.keep_running

        app = ws.WebSocketApp(
            f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}",
            on_open=on_open,
            on_close=on_close,
            on_message=on_message,
        )
        app.run_forever()

    #    @unittest.skipUnless(TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled")
    @unittest.skipUnless(False, "Test disabled for now (requires rel)")
    def test_run_forever_dispatcher(self):
        """A WebSocketApp should keep running as long as its self.keep_running
        is not False (in the boolean context).
        """

        def on_open(self, *args, **kwargs):
            """Send a message, receive, and send one more"""
            self.send("hello!")
            self.recv()
            self.send("goodbye!")

        def on_message(_, message):
            print(message)
            self.close()

        app = ws.WebSocketApp(
            f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}",
            on_open=on_open,
            on_message=on_message,
        )
        app.run_forever(dispatcher="Dispatcher")  # doesn't work

    #        app.run_forever(dispatcher=rel)          # would work
    #        rel.dispatch()

    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_run_forever_teardown_clean_exit(self):
        """The WebSocketApp.run_forever() method should return `False` when the application ends gracefully."""
        app = ws.WebSocketApp(f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}")
        threading.Timer(interval=0.2, function=app.close).start()
        teardown = app.run_forever()
        self.assertEqual(teardown, False)

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_sock_mask_key(self):
        """A WebSocketApp should forward the received mask_key function down
        to the actual socket.
        """

        def my_mask_key_func():
            return "\x00\x00\x00\x00"

        app = ws.WebSocketApp(
            "wss://api-pub.bitfinex.com/ws/1", get_mask_key=my_mask_key_func
        )

        # if numpy is installed, this assertion fail
        # Note: We can't use 'is' for comparing the functions directly, need to use 'id'.
        self.assertEqual(id(app.get_mask_key), id(my_mask_key_func))

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_invalid_ping_interval_ping_timeout(self):
        """Test exception handling if ping_interval < ping_timeout"""

        def on_ping(app, _):
            print("Got a ping!")
            app.close()

        def on_pong(app, _):
            print("Got a pong! No need to respond")
            app.close()

        app = ws.WebSocketApp(
            "wss://api-pub.bitfinex.com/ws/1", on_ping=on_ping, on_pong=on_pong
        )
        self.assertRaises(
            ws.WebSocketException,
            app.run_forever,
            ping_interval=1,
            ping_timeout=2,
            sslopt={"cert_reqs": ssl.CERT_NONE},
        )

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_ping_interval(self):
        """Test WebSocketApp proper ping functionality"""

        def on_ping(app, _):
            print("Got a ping!")
            app.close()

        def on_pong(app, _):
            print("Got a pong! No need to respond")
            app.close()

        app = ws.WebSocketApp(
            "wss://api-pub.bitfinex.com/ws/1", on_ping=on_ping, on_pong=on_pong
        )
        app.run_forever(
            ping_interval=2, ping_timeout=1, sslopt={"cert_reqs": ssl.CERT_NONE}
        )

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_opcode_close(self):
        """Test WebSocketApp close opcode"""

        app = ws.WebSocketApp("wss://tsock.us1.twilio.com/v3/wsconnect")
        app.run_forever(ping_interval=2, ping_timeout=1, ping_payload="Ping payload")

    # This is commented out because the URL no longer responds in the expected way
    # @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    # def testOpcodeBinary(self):
    #     """ Test WebSocketApp binary opcode
    #     """
    #     app = ws.WebSocketApp('wss://streaming.vn.teslamotors.com/streaming/')
    #     app.run_forever(ping_interval=2, ping_timeout=1, ping_payload="Ping payload")

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_bad_ping_interval(self):
        """A WebSocketApp handling of negative ping_interval"""
        app = ws.WebSocketApp("wss://api-pub.bitfinex.com/ws/1")
        self.assertRaises(
            ws.WebSocketException,
            app.run_forever,
            ping_interval=-5,
            sslopt={"cert_reqs": ssl.CERT_NONE},
        )

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_bad_ping_timeout(self):
        """A WebSocketApp handling of negative ping_timeout"""
        app = ws.WebSocketApp("wss://api-pub.bitfinex.com/ws/1")
        self.assertRaises(
            ws.WebSocketException,
            app.run_forever,
            ping_timeout=-3,
            sslopt={"cert_reqs": ssl.CERT_NONE},
        )

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_close_status_code(self):
        """Test extraction of close frame status code and close reason in WebSocketApp"""

        def on_close(wsapp, close_status_code, close_msg):
            print("on_close reached")

        app = ws.WebSocketApp(
            "wss://tsock.us1.twilio.com/v3/wsconnect", on_close=on_close
        )
        closeframe = ws.ABNF(
            opcode=ws.ABNF.OPCODE_CLOSE, data=b"\x03\xe8no-init-from-client"
        )
        self.assertEqual([1000, "no-init-from-client"], app._get_close_args(closeframe))

        closeframe = ws.ABNF(opcode=ws.ABNF.OPCODE_CLOSE, data=b"")
        self.assertEqual([None, None], app._get_close_args(closeframe))

        app2 = ws.WebSocketApp("wss://tsock.us1.twilio.com/v3/wsconnect")
        closeframe = ws.ABNF(opcode=ws.ABNF.OPCODE_CLOSE, data=b"")
        self.assertEqual([None, None], app2._get_close_args(closeframe))

        self.assertRaises(
            ws.WebSocketConnectionClosedException,
            app.send,
            data="test if connection is closed",
        )

    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_callback_function_exception(self):
        """Test callback function exception handling"""

        exc = None
        passed_app = None

        def on_open(app):
            raise RuntimeError("Callback failed")

        def on_error(app, err):
            nonlocal passed_app
            passed_app = app
            nonlocal exc
            exc = err

        def on_pong(app, _):
            app.close()

        app = ws.WebSocketApp(
            f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}",
            on_open=on_open,
            on_error=on_error,
            on_pong=on_pong,
        )
        app.run_forever(ping_interval=2, ping_timeout=1)

        self.assertEqual(passed_app, app)
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(str(exc), "Callback failed")

    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_callback_method_exception(self):
        """Test callback method exception handling"""

        class Callbacks:
            def __init__(self):
                self.exc = None
                self.passed_app = None
                self.app = ws.WebSocketApp(
                    f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}",
                    on_open=self.on_open,
                    on_error=self.on_error,
                    on_pong=self.on_pong,
                )
                self.app.run_forever(ping_interval=2, ping_timeout=1)

            def on_open(self, _):
                raise RuntimeError("Callback failed")

            def on_error(self, app, err):
                self.passed_app = app
                self.exc = err

            def on_pong(self, app, _):
                app.close()

        callbacks = Callbacks()

        self.assertEqual(callbacks.passed_app, callbacks.app)
        self.assertIsInstance(callbacks.exc, RuntimeError)
        self.assertEqual(str(callbacks.exc), "Callback failed")

    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_reconnect(self):
        """Test reconnect"""
        pong_count = 0
        exc = None

        def on_error(_, err):
            nonlocal exc
            exc = err

        def on_pong(app, _):
            nonlocal pong_count
            pong_count += 1
            if pong_count == 1:
                # First pong, shutdown socket, enforce read error
                app.sock.shutdown()
            if pong_count >= 2:
                # Got second pong after reconnect
                app.close()

        app = ws.WebSocketApp(
            f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}", on_pong=on_pong, on_error=on_error
        )
        app.run_forever(ping_interval=2, ping_timeout=1, reconnect=3)

        self.assertEqual(pong_count, 2)
        self.assertIsInstance(exc, ws.WebSocketTimeoutException)
        self.assertEqual(str(exc), "ping/pong timed out")


if __name__ == "__main__":
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\__init__.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

# Copyright 2007 Google Inc. All Rights Reserved.

__version__ = '6.31.1'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\arrayprint.py ===
def __getattr__(attr_name):
    from numpy._core import arrayprint

    from ._utils import _raise_warning
    ret = getattr(arrayprint, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.arrayprint' has no attribute {attr_name}")
    _raise_warning(attr_name, "arrayprint")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\defchararray.py ===
def __getattr__(attr_name):
    from numpy._core import defchararray

    from ._utils import _raise_warning
    ret = getattr(defchararray, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.defchararray' has no attribute {attr_name}")
    _raise_warning(attr_name, "defchararray")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\einsumfunc.py ===
def __getattr__(attr_name):
    from numpy._core import einsumfunc

    from ._utils import _raise_warning
    ret = getattr(einsumfunc, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.einsumfunc' has no attribute {attr_name}")
    _raise_warning(attr_name, "einsumfunc")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\fromnumeric.py ===
def __getattr__(attr_name):
    from numpy._core import fromnumeric

    from ._utils import _raise_warning
    ret = getattr(fromnumeric, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.fromnumeric' has no attribute {attr_name}")
    _raise_warning(attr_name, "fromnumeric")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\function_base.py ===
def __getattr__(attr_name):
    from numpy._core import function_base

    from ._utils import _raise_warning
    ret = getattr(function_base, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.function_base' has no attribute {attr_name}")
    _raise_warning(attr_name, "function_base")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\getlimits.py ===
def __getattr__(attr_name):
    from numpy._core import getlimits

    from ._utils import _raise_warning
    ret = getattr(getlimits, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.getlimits' has no attribute {attr_name}")
    _raise_warning(attr_name, "getlimits")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\numerictypes.py ===
def __getattr__(attr_name):
    from numpy._core import numerictypes

    from ._utils import _raise_warning
    ret = getattr(numerictypes, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.numerictypes' has no attribute {attr_name}")
    _raise_warning(attr_name, "numerictypes")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\overrides.py ===
def __getattr__(attr_name):
    from numpy._core import overrides

    from ._utils import _raise_warning
    ret = getattr(overrides, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.overrides' has no attribute {attr_name}")
    _raise_warning(attr_name, "overrides")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\records.py ===
def __getattr__(attr_name):
    from numpy._core import records

    from ._utils import _raise_warning
    ret = getattr(records, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.records' has no attribute {attr_name}")
    _raise_warning(attr_name, "records")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\shape_base.py ===
def __getattr__(attr_name):
    from numpy._core import shape_base

    from ._utils import _raise_warning
    ret = getattr(shape_base, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.shape_base' has no attribute {attr_name}")
    _raise_warning(attr_name, "shape_base")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\umath.py ===
def __getattr__(attr_name):
    from numpy._core import umath

    from ._utils import _raise_warning
    ret = getattr(umath, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.umath' has no attribute {attr_name}")
    _raise_warning(attr_name, "umath")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\_dtype.py ===
def __getattr__(attr_name):
    from numpy._core import _dtype

    from ._utils import _raise_warning
    ret = getattr(_dtype, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core._dtype' has no attribute {attr_name}")
    _raise_warning(attr_name, "_dtype")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\core\_dtype_ctypes.py ===
def __getattr__(attr_name):
    from numpy._core import _dtype_ctypes

    from ._utils import _raise_warning
    ret = getattr(_dtype_ctypes, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core._dtype_ctypes' has no attribute {attr_name}")
    _raise_warning(attr_name, "_dtype_ctypes")
    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_cython.py ===
import os
import subprocess
import sys
import sysconfig
from datetime import datetime

import pytest

import numpy as np
from numpy.testing import IS_EDITABLE, IS_WASM, assert_array_equal

# This import is copied from random.tests.test_extending
try:
    import cython
    from Cython.Compiler.Version import version as cython_version
except ImportError:
    cython = None
else:
    from numpy._utils import _pep440

    # Note: keep in sync with the one in pyproject.toml
    required_version = "3.0.6"
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # too old or wrong cython, skip the test
        cython = None

pytestmark = pytest.mark.skipif(cython is None, reason="requires cython")


if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )


@pytest.fixture(scope='module')
def install_temp(tmpdir_factory):
    # Based in part on test_cython from random.tests.test_extending
    if IS_WASM:
        pytest.skip("No subprocess")

    srcdir = os.path.join(os.path.dirname(__file__), 'examples', 'cython')
    build_dir = tmpdir_factory.mktemp("cython_test") / "build"
    os.makedirs(build_dir, exist_ok=True)
    # Ensure we use the correct Python interpreter even when `meson` is
    # installed in a different Python environment (see gh-24956)
    native_file = str(build_dir / 'interpreter-native-file.ini')
    with open(native_file, 'w') as f:
        f.write("[binaries]\n")
        f.write(f"python = '{sys.executable}'\n")
        f.write(f"python3 = '{sys.executable}'")

    try:
        subprocess.check_call(["meson", "--version"])
    except FileNotFoundError:
        pytest.skip("No usable 'meson' found")
    if sysconfig.get_platform() == "win-arm64":
        pytest.skip("Meson unable to find MSVC linker on win-arm64")
    if sys.platform == "win32":
        subprocess.check_call(["meson", "setup",
                               "--buildtype=release",
                               "--vsenv", "--native-file", native_file,
                               str(srcdir)],
                              cwd=build_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup",
                               "--native-file", native_file, str(srcdir)],
                              cwd=build_dir
                              )
    try:
        subprocess.check_call(["meson", "compile", "-vv"], cwd=build_dir)
    except subprocess.CalledProcessError:
        print("----------------")
        print("meson build failed when doing")
        print(f"'meson setup --native-file {native_file} {srcdir}'")
        print("'meson compile -vv'")
        print(f"in {build_dir}")
        print("----------------")
        raise

    sys.path.append(str(build_dir))


def test_is_timedelta64_object(install_temp):
    import checks

    assert checks.is_td64(np.timedelta64(1234))
    assert checks.is_td64(np.timedelta64(1234, "ns"))
    assert checks.is_td64(np.timedelta64("NaT", "ns"))

    assert not checks.is_td64(1)
    assert not checks.is_td64(None)
    assert not checks.is_td64("foo")
    assert not checks.is_td64(np.datetime64("now", "s"))


def test_is_datetime64_object(install_temp):
    import checks

    assert checks.is_dt64(np.datetime64(1234, "ns"))
    assert checks.is_dt64(np.datetime64("NaT", "ns"))

    assert not checks.is_dt64(1)
    assert not checks.is_dt64(None)
    assert not checks.is_dt64("foo")
    assert not checks.is_dt64(np.timedelta64(1234))


def test_get_datetime64_value(install_temp):
    import checks

    dt64 = np.datetime64("2016-01-01", "ns")

    result = checks.get_dt64_value(dt64)
    expected = dt64.view("i8")

    assert result == expected


def test_get_timedelta64_value(install_temp):
    import checks

    td64 = np.timedelta64(12345, "h")

    result = checks.get_td64_value(td64)
    expected = td64.view("i8")

    assert result == expected


def test_get_datetime64_unit(install_temp):
    import checks

    dt64 = np.datetime64("2016-01-01", "ns")
    result = checks.get_dt64_unit(dt64)
    expected = 10
    assert result == expected

    td64 = np.timedelta64(12345, "h")
    result = checks.get_dt64_unit(td64)
    expected = 5
    assert result == expected


def test_abstract_scalars(install_temp):
    import checks

    assert checks.is_integer(1)
    assert checks.is_integer(np.int8(1))
    assert checks.is_integer(np.uint64(1))

def test_default_int(install_temp):
    import checks

    assert checks.get_default_integer() is np.dtype(int)


def test_ravel_axis(install_temp):
    import checks

    assert checks.get_ravel_axis() == np.iinfo("intc").min


def test_convert_datetime64_to_datetimestruct(install_temp):
    # GH#21199
    import checks

    res = checks.convert_datetime64_to_datetimestruct()

    exp = {
        "year": 2022,
        "month": 3,
        "day": 15,
        "hour": 20,
        "min": 1,
        "sec": 55,
        "us": 260292,
        "ps": 0,
        "as": 0,
    }

    assert res == exp


class TestDatetimeStrings:
    def test_make_iso_8601_datetime(self, install_temp):
        # GH#21199
        import checks
        dt = datetime(2016, 6, 2, 10, 45, 19)
        # uses NPY_FR_s
        result = checks.make_iso_8601_datetime(dt)
        assert result == b"2016-06-02T10:45:19"

    def test_get_datetime_iso_8601_strlen(self, install_temp):
        # GH#21199
        import checks
        # uses NPY_FR_ns
        res = checks.get_datetime_iso_8601_strlen()
        assert res == 48


@pytest.mark.parametrize(
    "arrays",
    [
        [np.random.rand(2)],
        [np.random.rand(2), np.random.rand(3, 1)],
        [np.random.rand(2), np.random.rand(2, 3, 2), np.random.rand(1, 3, 2)],
        [np.random.rand(2, 1)] * 4 + [np.random.rand(1, 1, 1)],
    ]
)
def test_multiiter_fields(install_temp, arrays):
    import checks
    bcast = np.broadcast(*arrays)

    assert bcast.ndim == checks.get_multiiter_number_of_dims(bcast)
    assert bcast.size == checks.get_multiiter_size(bcast)
    assert bcast.numiter == checks.get_multiiter_num_of_iterators(bcast)
    assert bcast.shape == checks.get_multiiter_shape(bcast)
    assert bcast.index == checks.get_multiiter_current_index(bcast)
    assert all(
        x.base is y.base
        for x, y in zip(bcast.iters, checks.get_multiiter_iters(bcast))
    )


def test_dtype_flags(install_temp):
    import checks
    dtype = np.dtype("i,O")  # dtype with somewhat interesting flags
    assert dtype.flags == checks.get_dtype_flags(dtype)


def test_conv_intp(install_temp):
    import checks

    class myint:
        def __int__(self):
            return 3

    # These conversion passes via `__int__`, not `__index__`:
    assert checks.conv_intp(3.) == 3
    assert checks.conv_intp(myint()) == 3


def test_npyiter_api(install_temp):
    import checks
    arr = np.random.rand(3, 2)

    it = np.nditer(arr)
    assert checks.get_npyiter_size(it) == it.itersize == np.prod(arr.shape)
    assert checks.get_npyiter_ndim(it) == it.ndim == 1
    assert checks.npyiter_has_index(it) == it.has_index == False

    it = np.nditer(arr, flags=["c_index"])
    assert checks.npyiter_has_index(it) == it.has_index == True
    assert (
        checks.npyiter_has_delayed_bufalloc(it)
        == it.has_delayed_bufalloc
        == False
    )

    it = np.nditer(arr, flags=["buffered", "delay_bufalloc"])
    assert (
        checks.npyiter_has_delayed_bufalloc(it)
        == it.has_delayed_bufalloc
        == True
    )

    it = np.nditer(arr, flags=["multi_index"])
    assert checks.get_npyiter_size(it) == it.itersize == np.prod(arr.shape)
    assert checks.npyiter_has_multi_index(it) == it.has_multi_index == True
    assert checks.get_npyiter_ndim(it) == it.ndim == 2
    assert checks.test_get_multi_index_iter_next(it, arr)

    arr2 = np.random.rand(2, 1, 2)
    it = np.nditer([arr, arr2])
    assert checks.get_npyiter_nop(it) == it.nop == 2
    assert checks.get_npyiter_size(it) == it.itersize == 12
    assert checks.get_npyiter_ndim(it) == it.ndim == 3
    assert all(
        x is y for x, y in zip(checks.get_npyiter_operands(it), it.operands)
    )
    assert all(
        np.allclose(x, y)
        for x, y in zip(checks.get_npyiter_itviews(it), it.itviews)
    )


def test_fillwithbytes(install_temp):
    import checks

    arr = checks.compile_fillwithbyte()
    assert_array_equal(arr, np.ones((1, 2)))


def test_complex(install_temp):
    from checks import inc2_cfloat_struct

    arr = np.array([0, 10 + 10j], dtype="F")
    inc2_cfloat_struct(arr)
    assert arr[1] == (12 + 12j)


def test_npystring_pack(install_temp):
    """Check that the cython API can write to a vstring array."""
    import checks

    arr = np.array(['a', 'b', 'c'], dtype='T')
    assert checks.npystring_pack(arr) == 0

    # checks.npystring_pack writes to the beginning of the array
    assert arr[0] == "Hello world"

def test_npystring_load(install_temp):
    """Check that the cython API can load strings from a vstring array."""
    import checks

    arr = np.array(['abcd', 'b', 'c'], dtype='T')
    result = checks.npystring_load(arr)
    assert result == 'abcd'


def test_npystring_multiple_allocators(install_temp):
    """Check that the cython API can acquire/release multiple vstring allocators."""
    import checks

    dt = np.dtypes.StringDType(na_object=None)
    arr1 = np.array(['abcd', 'b', 'c'], dtype=dt)
    arr2 = np.array(['a', 'b', 'c'], dtype=dt)

    assert checks.npystring_pack_multiple(arr1, arr2) == 0
    assert arr1[0] == "Hello world"
    assert arr1[-1] is None
    assert arr2[0] == "test this"


def test_npystring_allocators_other_dtype(install_temp):
    """Check that allocators for non-StringDType arrays is NULL."""
    import checks

    arr1 = np.array([1, 2, 3], dtype='i')
    arr2 = np.array([4, 5, 6], dtype='i')

    assert checks.npystring_allocators_other_types(arr1, arr2) == 0


@pytest.mark.skipif(sysconfig.get_platform() == 'win-arm64', reason='no checks module on win-arm64')
def test_npy_uintp_type_enum():
    import checks
    assert checks.check_npy_uintp_type_enum()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# appended to the __init__.py in the onnxruntime module's 'tools' folder from /tools/python/util/__init__append.py
import importlib.util

have_torch = importlib.util.find_spec("torch")
if have_torch:
    from .pytorch_export_helpers import infer_input_info  # noqa: F401

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\numpy_\test_numpy.py ===
"""
Additional tests for NumpyExtensionArray that aren't covered by
the interface tests.
"""
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import NumpyEADtype

import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray


@pytest.fixture(
    params=[
        np.array(["a", "b"], dtype=object),
        np.array([0, 1], dtype=float),
        np.array([0, 1], dtype=int),
        np.array([0, 1 + 2j], dtype=complex),
        np.array([True, False], dtype=bool),
        np.array([0, 1], dtype="datetime64[ns]"),
        np.array([0, 1], dtype="timedelta64[ns]"),
    ],
)
def any_numpy_array(request):
    """
    Parametrized fixture for NumPy arrays with different dtypes.

    This excludes string and bytes.
    """
    return request.param.copy()


# ----------------------------------------------------------------------------
# NumpyEADtype


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("bool", True),
        ("int", True),
        ("uint", True),
        ("float", True),
        ("complex", True),
        ("str", False),
        ("bytes", False),
        ("datetime64[ns]", False),
        ("object", False),
        ("void", False),
    ],
)
def test_is_numeric(dtype, expected):
    dtype = NumpyEADtype(dtype)
    assert dtype._is_numeric is expected


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("bool", True),
        ("int", False),
        ("uint", False),
        ("float", False),
        ("complex", False),
        ("str", False),
        ("bytes", False),
        ("datetime64[ns]", False),
        ("object", False),
        ("void", False),
    ],
)
def test_is_boolean(dtype, expected):
    dtype = NumpyEADtype(dtype)
    assert dtype._is_boolean is expected


def test_repr():
    dtype = NumpyEADtype(np.dtype("int64"))
    assert repr(dtype) == "NumpyEADtype('int64')"


def test_constructor_from_string():
    result = NumpyEADtype.construct_from_string("int64")
    expected = NumpyEADtype(np.dtype("int64"))
    assert result == expected


def test_dtype_idempotent(any_numpy_dtype):
    dtype = NumpyEADtype(any_numpy_dtype)

    result = NumpyEADtype(dtype)
    assert result == dtype


# ----------------------------------------------------------------------------
# Construction


def test_constructor_no_coercion():
    with pytest.raises(ValueError, match="NumPy array"):
        NumpyExtensionArray([1, 2, 3])


def test_series_constructor_with_copy():
    ndarray = np.array([1, 2, 3])
    ser = pd.Series(NumpyExtensionArray(ndarray), copy=True)

    assert ser.values is not ndarray


def test_series_constructor_with_astype():
    ndarray = np.array([1, 2, 3])
    result = pd.Series(NumpyExtensionArray(ndarray), dtype="float64")
    expected = pd.Series([1.0, 2.0, 3.0], dtype="float64")
    tm.assert_series_equal(result, expected)


def test_from_sequence_dtype():
    arr = np.array([1, 2, 3], dtype="int64")
    result = NumpyExtensionArray._from_sequence(arr, dtype="uint64")
    expected = NumpyExtensionArray(np.array([1, 2, 3], dtype="uint64"))
    tm.assert_extension_array_equal(result, expected)


def test_constructor_copy():
    arr = np.array([0, 1])
    result = NumpyExtensionArray(arr, copy=True)

    assert not tm.shares_memory(result, arr)


def test_constructor_with_data(any_numpy_array):
    nparr = any_numpy_array
    arr = NumpyExtensionArray(nparr)
    assert arr.dtype.numpy_dtype == nparr.dtype


# ----------------------------------------------------------------------------
# Conversion


def test_to_numpy():
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    result = arr.to_numpy()
    assert result is arr._ndarray

    result = arr.to_numpy(copy=True)
    assert result is not arr._ndarray

    result = arr.to_numpy(dtype="f8")
    expected = np.array([1, 2, 3], dtype="f8")
    tm.assert_numpy_array_equal(result, expected)


# ----------------------------------------------------------------------------
# Setitem


def test_setitem_series():
    ser = pd.Series([1, 2, 3])
    ser.array[0] = 10
    expected = pd.Series([10, 2, 3])
    tm.assert_series_equal(ser, expected)


def test_setitem(any_numpy_array):
    nparr = any_numpy_array
    arr = NumpyExtensionArray(nparr, copy=True)

    arr[0] = arr[1]
    nparr[0] = nparr[1]

    tm.assert_numpy_array_equal(arr.to_numpy(), nparr)


# ----------------------------------------------------------------------------
# Reductions


def test_bad_reduce_raises():
    arr = np.array([1, 2, 3], dtype="int64")
    arr = NumpyExtensionArray(arr)
    msg = "cannot perform not_a_method with type int"
    with pytest.raises(TypeError, match=msg):
        arr._reduce(msg)


def test_validate_reduction_keyword_args():
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    msg = "the 'keepdims' parameter is not supported .*all"
    with pytest.raises(ValueError, match=msg):
        arr.all(keepdims=True)


def test_np_max_nested_tuples():
    # case where checking in ufunc.nout works while checking for tuples
    #  does not
    vals = [
        (("j", "k"), ("l", "m")),
        (("l", "m"), ("o", "p")),
        (("o", "p"), ("j", "k")),
    ]
    ser = pd.Series(vals)
    arr = ser.array

    assert arr.max() is arr[2]
    assert ser.max() is arr[2]

    result = np.maximum.reduce(arr)
    assert result == arr[2]

    result = np.maximum.reduce(ser)
    assert result == arr[2]


def test_np_reduce_2d():
    raw = np.arange(12).reshape(4, 3)
    arr = NumpyExtensionArray(raw)

    res = np.maximum.reduce(arr, axis=0)
    tm.assert_extension_array_equal(res, arr[-1])

    alt = arr.max(axis=0)
    tm.assert_extension_array_equal(alt, arr[-1])


# ----------------------------------------------------------------------------
# Ops


@pytest.mark.parametrize("ufunc", [np.abs, np.negative, np.positive])
def test_ufunc_unary(ufunc):
    arr = NumpyExtensionArray(np.array([-1.0, 0.0, 1.0]))
    result = ufunc(arr)
    expected = NumpyExtensionArray(ufunc(arr._ndarray))
    tm.assert_extension_array_equal(result, expected)

    # same thing but with the 'out' keyword
    out = NumpyExtensionArray(np.array([-9.0, -9.0, -9.0]))
    ufunc(arr, out=out)
    tm.assert_extension_array_equal(out, expected)


def test_ufunc():
    arr = NumpyExtensionArray(np.array([-1.0, 0.0, 1.0]))

    r1, r2 = np.divmod(arr, np.add(arr, 2))
    e1, e2 = np.divmod(arr._ndarray, np.add(arr._ndarray, 2))
    e1 = NumpyExtensionArray(e1)
    e2 = NumpyExtensionArray(e2)
    tm.assert_extension_array_equal(r1, e1)
    tm.assert_extension_array_equal(r2, e2)


def test_basic_binop():
    # Just a basic smoke test. The EA interface tests exercise this
    # more thoroughly.
    x = NumpyExtensionArray(np.array([1, 2, 3]))
    result = x + x
    expected = NumpyExtensionArray(np.array([2, 4, 6]))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [None, object])
def test_setitem_object_typecode(dtype):
    arr = NumpyExtensionArray(np.array(["a", "b", "c"], dtype=dtype))
    arr[0] = "t"
    expected = NumpyExtensionArray(np.array(["t", "b", "c"], dtype=dtype))
    tm.assert_extension_array_equal(arr, expected)


def test_setitem_no_coercion():
    # https://github.com/pandas-dev/pandas/issues/28150
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="int"):
        arr[0] = "a"

    # With a value that we do coerce, check that we coerce the value
    #  and not the underlying array.
    arr[0] = 2.5
    assert isinstance(arr[0], (int, np.integer)), type(arr[0])


def test_setitem_preserves_views():
    # GH#28150, see also extension test of the same name
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    view1 = arr.view()
    view2 = arr[:]
    view3 = np.asarray(arr)

    arr[0] = 9
    assert view1[0] == 9
    assert view2[0] == 9
    assert view3[0] == 9

    arr[-1] = 2.5
    view1[-1] = 5
    assert arr[-1] == 5


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_quantile_empty(dtype):
    # we should get back np.nans, not -1s
    arr = NumpyExtensionArray(np.array([], dtype=dtype))
    idx = pd.Index([0.0, 0.5])

    result = arr._quantile(idx, interpolation="linear")
    expected = NumpyExtensionArray(np.array([np.nan, np.nan]))
    tm.assert_extension_array_equal(result, expected)


def test_factorize_unsigned():
    # don't raise when calling factorize on unsigned int NumpyExtensionArray
    arr = np.array([1, 2, 3], dtype=np.uint64)
    obj = NumpyExtensionArray(arr)

    res_codes, res_unique = obj.factorize()
    exp_codes, exp_unique = pd.factorize(arr)

    tm.assert_numpy_array_equal(res_codes, exp_codes)

    tm.assert_extension_array_equal(res_unique, NumpyExtensionArray(exp_unique))


# ----------------------------------------------------------------------------
# Output formatting


def test_array_repr(any_numpy_array):
    # GH#61085
    nparray = any_numpy_array
    arr = NumpyExtensionArray(nparray)
    if nparray.dtype == "object":
        values = "['a', 'b']"
    elif nparray.dtype == "float64":
        values = "[0.0, 1.0]"
    elif str(nparray.dtype).startswith("int"):
        values = "[0, 1]"
    elif nparray.dtype == "complex128":
        values = "[0j, (1+2j)]"
    elif nparray.dtype == "bool":
        values = "[True, False]"
    elif nparray.dtype == "datetime64[ns]":
        values = "[1970-01-01T00:00:00.000000000, 1970-01-01T00:00:00.000000001]"
    elif nparray.dtype == "timedelta64[ns]":
        values = "[0 nanoseconds, 1 nanoseconds]"
    expected = f"<NumpyExtensionArray>\n{values}\nLength: 2, dtype: {nparray.dtype}"
    result = repr(arr)
    assert result == expected, f"{result} vs {expected}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timestamp\methods\test_tz_localize.py ===
from datetime import timedelta
import re

from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
    AmbiguousTimeError,
    NonExistentTimeError,
)

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime

from pandas import (
    NaT,
    Timestamp,
)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Cannot assign to a type
    ZoneInfo = None  # type: ignore[misc, assignment]


class TestTimestampTZLocalize:
    @pytest.mark.skip_ubsan
    def test_tz_localize_pushes_out_of_bounds(self):
        # GH#12677
        # tz_localize that pushes away from the boundary is OK
        msg = (
            f"Converting {Timestamp.min.strftime('%Y-%m-%d %H:%M:%S')} "
            f"underflows past {Timestamp.min}"
        )
        pac = Timestamp.min.tz_localize("US/Pacific")
        assert pac._value > Timestamp.min._value
        pac.tz_convert("Asia/Tokyo")  # tz_convert doesn't change value
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.tz_localize("Asia/Tokyo")

        # tz_localize that pushes away from the boundary is OK
        msg = (
            f"Converting {Timestamp.max.strftime('%Y-%m-%d %H:%M:%S')} "
            f"overflows past {Timestamp.max}"
        )
        tokyo = Timestamp.max.tz_localize("Asia/Tokyo")
        assert tokyo._value < Timestamp.max._value
        tokyo.tz_convert("US/Pacific")  # tz_convert doesn't change value
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.tz_localize("US/Pacific")

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_tz_localize_ambiguous_bool(self, unit):
        # make sure that we are correctly accepting bool values as ambiguous
        # GH#14402
        ts = Timestamp("2015-11-01 01:00:03").as_unit(unit)
        expected0 = Timestamp("2015-11-01 01:00:03-0500", tz="US/Central")
        expected1 = Timestamp("2015-11-01 01:00:03-0600", tz="US/Central")

        msg = "Cannot infer dst time from 2015-11-01 01:00:03"
        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            ts.tz_localize("US/Central")

        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            ts.tz_localize("dateutil/US/Central")

        if ZoneInfo is not None:
            try:
                tz = ZoneInfo("US/Central")
            except KeyError:
                # no tzdata
                pass
            else:
                with pytest.raises(pytz.AmbiguousTimeError, match=msg):
                    ts.tz_localize(tz)

        result = ts.tz_localize("US/Central", ambiguous=True)
        assert result == expected0
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

        result = ts.tz_localize("US/Central", ambiguous=False)
        assert result == expected1
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    def test_tz_localize_ambiguous(self):
        ts = Timestamp("2014-11-02 01:00")
        ts_dst = ts.tz_localize("US/Eastern", ambiguous=True)
        ts_no_dst = ts.tz_localize("US/Eastern", ambiguous=False)

        assert ts_no_dst._value - ts_dst._value == 3600
        msg = re.escape(
            "'ambiguous' parameter must be one of: "
            "True, False, 'NaT', 'raise' (default)"
        )
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize("US/Eastern", ambiguous="infer")

        # GH#8025
        msg = "Cannot localize tz-aware Timestamp, use tz_convert for conversions"
        with pytest.raises(TypeError, match=msg):
            Timestamp("2011-01-01", tz="US/Eastern").tz_localize("Asia/Tokyo")

        msg = "Cannot convert tz-naive Timestamp, use tz_localize to localize"
        with pytest.raises(TypeError, match=msg):
            Timestamp("2011-01-01").tz_convert("Asia/Tokyo")

    @pytest.mark.parametrize(
        "stamp, tz",
        [
            ("2015-03-08 02:00", "US/Eastern"),
            ("2015-03-08 02:30", "US/Pacific"),
            ("2015-03-29 02:00", "Europe/Paris"),
            ("2015-03-29 02:30", "Europe/Belgrade"),
        ],
    )
    def test_tz_localize_nonexistent(self, stamp, tz):
        # GH#13057
        ts = Timestamp(stamp)
        with pytest.raises(NonExistentTimeError, match=stamp):
            ts.tz_localize(tz)
        # GH 22644
        with pytest.raises(NonExistentTimeError, match=stamp):
            ts.tz_localize(tz, nonexistent="raise")
        assert ts.tz_localize(tz, nonexistent="NaT") is NaT

    @pytest.mark.parametrize(
        "stamp, tz, forward_expected, backward_expected",
        [
            (
                "2015-03-29 02:00:00",
                "Europe/Warsaw",
                "2015-03-29 03:00:00",
                "2015-03-29 01:59:59",
            ),  # utc+1 -> utc+2
            (
                "2023-03-12 02:00:00",
                "America/Los_Angeles",
                "2023-03-12 03:00:00",
                "2023-03-12 01:59:59",
            ),  # utc-8 -> utc-7
            (
                "2023-03-26 01:00:00",
                "Europe/London",
                "2023-03-26 02:00:00",
                "2023-03-26 00:59:59",
            ),  # utc+0 -> utc+1
            (
                "2023-03-26 00:00:00",
                "Atlantic/Azores",
                "2023-03-26 01:00:00",
                "2023-03-25 23:59:59",
            ),  # utc-1 -> utc+0
        ],
    )
    def test_tz_localize_nonexistent_shift(
        self, stamp, tz, forward_expected, backward_expected
    ):
        ts = Timestamp(stamp)
        forward_ts = ts.tz_localize(tz, nonexistent="shift_forward")
        assert forward_ts == Timestamp(forward_expected, tz=tz)
        backward_ts = ts.tz_localize(tz, nonexistent="shift_backward")
        assert backward_ts == Timestamp(backward_expected, tz=tz)

    def test_tz_localize_ambiguous_raise(self):
        # GH#13057
        ts = Timestamp("2015-11-1 01:00")
        msg = "Cannot infer dst time from 2015-11-01 01:00:00,"
        with pytest.raises(AmbiguousTimeError, match=msg):
            ts.tz_localize("US/Pacific", ambiguous="raise")

    def test_tz_localize_nonexistent_invalid_arg(self, warsaw):
        # GH 22644
        tz = warsaw
        ts = Timestamp("2015-03-29 02:00:00")
        msg = (
            "The nonexistent argument must be one of 'raise', 'NaT', "
            "'shift_forward', 'shift_backward' or a timedelta object"
        )
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent="foo")

    @pytest.mark.parametrize(
        "stamp",
        [
            "2014-02-01 09:00",
            "2014-07-08 09:00",
            "2014-11-01 17:00",
            "2014-11-05 00:00",
        ],
    )
    def test_tz_localize_roundtrip(self, stamp, tz_aware_fixture):
        tz = tz_aware_fixture
        ts = Timestamp(stamp)
        localized = ts.tz_localize(tz)
        assert localized == Timestamp(stamp, tz=tz)

        msg = "Cannot localize tz-aware Timestamp"
        with pytest.raises(TypeError, match=msg):
            localized.tz_localize(tz)

        reset = localized.tz_localize(None)
        assert reset == ts
        assert reset.tzinfo is None

    def test_tz_localize_ambiguous_compat(self):
        # validate that pytz and dateutil are compat for dst
        # when the transition happens
        naive = Timestamp("2013-10-27 01:00:00")

        pytz_zone = "Europe/London"
        dateutil_zone = "dateutil/Europe/London"
        result_pytz = naive.tz_localize(pytz_zone, ambiguous=False)
        result_dateutil = naive.tz_localize(dateutil_zone, ambiguous=False)
        assert result_pytz._value == result_dateutil._value
        assert result_pytz._value == 1382835600

        # fixed ambiguous behavior
        # see gh-14621, GH#45087
        assert result_pytz.to_pydatetime().tzname() == "GMT"
        assert result_dateutil.to_pydatetime().tzname() == "GMT"
        assert str(result_pytz) == str(result_dateutil)

        # 1 hour difference
        result_pytz = naive.tz_localize(pytz_zone, ambiguous=True)
        result_dateutil = naive.tz_localize(dateutil_zone, ambiguous=True)
        assert result_pytz._value == result_dateutil._value
        assert result_pytz._value == 1382832000

        # see gh-14621
        assert str(result_pytz) == str(result_dateutil)
        assert (
            result_pytz.to_pydatetime().tzname()
            == result_dateutil.to_pydatetime().tzname()
        )

    @pytest.mark.parametrize(
        "tz",
        [
            pytz.timezone("US/Eastern"),
            gettz("US/Eastern"),
            "US/Eastern",
            "dateutil/US/Eastern",
        ],
    )
    def test_timestamp_tz_localize(self, tz):
        stamp = Timestamp("3/11/2012 04:00")

        result = stamp.tz_localize(tz)
        expected = Timestamp("3/11/2012 04:00", tz=tz)
        assert result.hour == expected.hour
        assert result == expected

    @pytest.mark.parametrize(
        "start_ts, tz, end_ts, shift",
        [
            ["2015-03-29 02:20:00", "Europe/Warsaw", "2015-03-29 03:00:00", "forward"],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:59:59.999999999",
                "backward",
            ],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 03:20:00",
                timedelta(hours=1),
            ],
            [
                "2015-03-29 02:20:00",
                "Europe/Warsaw",
                "2015-03-29 01:20:00",
                timedelta(hours=-1),
            ],
            ["2018-03-11 02:33:00", "US/Pacific", "2018-03-11 03:00:00", "forward"],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:59:59.999999999",
                "backward",
            ],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 03:33:00",
                timedelta(hours=1),
            ],
            [
                "2018-03-11 02:33:00",
                "US/Pacific",
                "2018-03-11 01:33:00",
                timedelta(hours=-1),
            ],
        ],
    )
    @pytest.mark.parametrize("tz_type", ["", "dateutil/"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_timestamp_tz_localize_nonexistent_shift(
        self, start_ts, tz, end_ts, shift, tz_type, unit
    ):
        # GH 8917, 24466
        tz = tz_type + tz
        if isinstance(shift, str):
            shift = "shift_" + shift
        ts = Timestamp(start_ts).as_unit(unit)
        result = ts.tz_localize(tz, nonexistent=shift)
        expected = Timestamp(end_ts).tz_localize(tz)

        if unit == "us":
            assert result == expected.replace(nanosecond=0)
        elif unit == "ms":
            micros = expected.microsecond - expected.microsecond % 1000
            assert result == expected.replace(microsecond=micros, nanosecond=0)
        elif unit == "s":
            assert result == expected.replace(microsecond=0, nanosecond=0)
        else:
            assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value

    @pytest.mark.parametrize("offset", [-1, 1])
    def test_timestamp_tz_localize_nonexistent_shift_invalid(self, offset, warsaw):
        # GH 8917, 24466
        tz = warsaw
        ts = Timestamp("2015-03-29 02:20:00")
        msg = "The provided timedelta will relocalize on a nonexistent time"
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent=timedelta(seconds=offset))

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_timestamp_tz_localize_nonexistent_NaT(self, warsaw, unit):
        # GH 8917
        tz = warsaw
        ts = Timestamp("2015-03-29 02:20:00").as_unit(unit)
        result = ts.tz_localize(tz, nonexistent="NaT")
        assert result is NaT

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
    def test_timestamp_tz_localize_nonexistent_raise(self, warsaw, unit):
        # GH 8917
        tz = warsaw
        ts = Timestamp("2015-03-29 02:20:00").as_unit(unit)
        msg = "2015-03-29 02:20:00"
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            ts.tz_localize(tz, nonexistent="raise")
        msg = (
            "The nonexistent argument must be one of 'raise', 'NaT', "
            "'shift_forward', 'shift_backward' or a timedelta object"
        )
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent="foo")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_week.py ===
"""
Tests for the following offsets:
- Week
- WeekOfMonth
- LastWeekOfMonth
"""
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)

import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    Day,
    LastWeekOfMonth,
    Week,
    WeekOfMonth,
)

import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
    WeekDay,
    assert_is_on_offset,
    assert_offset_equal,
)


class TestWeek:
    def test_repr(self):
        assert repr(Week(weekday=0)) == "<Week: weekday=0>"
        assert repr(Week(n=-1, weekday=0)) == "<-1 * Week: weekday=0>"
        assert repr(Week(n=-2, weekday=0)) == "<-2 * Weeks: weekday=0>"

    def test_corner(self):
        with pytest.raises(ValueError, match="Day must be"):
            Week(weekday=7)

        with pytest.raises(ValueError, match="Day must be"):
            Week(weekday=-1)

    def test_is_anchored(self):
        msg = "Week.is_anchored is deprecated "

        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert Week(weekday=0).is_anchored()
            assert not Week().is_anchored()
            assert not Week(2, weekday=2).is_anchored()
            assert not Week(2).is_anchored()

    offset_cases = []
    # not business week
    offset_cases.append(
        (
            Week(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 8),
                datetime(2008, 1, 4): datetime(2008, 1, 11),
                datetime(2008, 1, 5): datetime(2008, 1, 12),
                datetime(2008, 1, 6): datetime(2008, 1, 13),
                datetime(2008, 1, 7): datetime(2008, 1, 14),
            },
        )
    )

    # Mon
    offset_cases.append(
        (
            Week(weekday=0),
            {
                datetime(2007, 12, 31): datetime(2008, 1, 7),
                datetime(2008, 1, 4): datetime(2008, 1, 7),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 14),
            },
        )
    )

    # n=0 -> roll forward. Mon
    offset_cases.append(
        (
            Week(0, weekday=0),
            {
                datetime(2007, 12, 31): datetime(2007, 12, 31),
                datetime(2008, 1, 4): datetime(2008, 1, 7),
                datetime(2008, 1, 5): datetime(2008, 1, 7),
                datetime(2008, 1, 6): datetime(2008, 1, 7),
                datetime(2008, 1, 7): datetime(2008, 1, 7),
            },
        )
    )

    # n=0 -> roll forward. Mon
    offset_cases.append(
        (
            Week(-2, weekday=1),
            {
                datetime(2010, 4, 6): datetime(2010, 3, 23),
                datetime(2010, 4, 8): datetime(2010, 3, 30),
                datetime(2010, 4, 5): datetime(2010, 3, 23),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize("weekday", range(7))
    def test_is_on_offset(self, weekday):
        offset = Week(weekday=weekday)

        for day in range(1, 8):
            date = datetime(2008, 1, day)
            expected = day % 7 == weekday
        assert_is_on_offset(offset, date, expected)

    @pytest.mark.parametrize(
        "n,date",
        [
            (2, "1862-01-13 09:03:34.873477378+0210"),
            (-2, "1856-10-24 16:18:36.556360110-0717"),
        ],
    )
    def test_is_on_offset_weekday_none(self, n, date):
        # GH 18510 Week with weekday = None, normalize = False
        # should always be is_on_offset
        offset = Week(n=n, weekday=None)
        ts = Timestamp(date, tz="Africa/Lusaka")
        fast = offset.is_on_offset(ts)
        slow = (ts + offset) - offset == ts
        assert fast == slow

    def test_week_add_invalid(self):
        # Week with weekday should raise TypeError and _not_ AttributeError
        #  when adding invalid offset
        offset = Week(weekday=1)
        other = Day()
        with pytest.raises(TypeError, match="Cannot add"):
            offset + other


class TestWeekOfMonth:
    def test_constructor(self):
        with pytest.raises(ValueError, match="^Week"):
            WeekOfMonth(n=1, week=4, weekday=0)

        with pytest.raises(ValueError, match="^Week"):
            WeekOfMonth(n=1, week=-1, weekday=0)

        with pytest.raises(ValueError, match="^Day"):
            WeekOfMonth(n=1, week=0, weekday=-1)

        with pytest.raises(ValueError, match="^Day"):
            WeekOfMonth(n=1, week=0, weekday=-7)

    def test_repr(self):
        assert (
            repr(WeekOfMonth(weekday=1, week=2)) == "<WeekOfMonth: week=2, weekday=1>"
        )

    def test_offset(self):
        date1 = datetime(2011, 1, 4)  # 1st Tuesday of Month
        date2 = datetime(2011, 1, 11)  # 2nd Tuesday of Month
        date3 = datetime(2011, 1, 18)  # 3rd Tuesday of Month
        date4 = datetime(2011, 1, 25)  # 4th Tuesday of Month

        # see for loop for structure
        test_cases = [
            (-2, 2, 1, date1, datetime(2010, 11, 16)),
            (-2, 2, 1, date2, datetime(2010, 11, 16)),
            (-2, 2, 1, date3, datetime(2010, 11, 16)),
            (-2, 2, 1, date4, datetime(2010, 12, 21)),
            (-1, 2, 1, date1, datetime(2010, 12, 21)),
            (-1, 2, 1, date2, datetime(2010, 12, 21)),
            (-1, 2, 1, date3, datetime(2010, 12, 21)),
            (-1, 2, 1, date4, datetime(2011, 1, 18)),
            (0, 0, 1, date1, datetime(2011, 1, 4)),
            (0, 0, 1, date2, datetime(2011, 2, 1)),
            (0, 0, 1, date3, datetime(2011, 2, 1)),
            (0, 0, 1, date4, datetime(2011, 2, 1)),
            (0, 1, 1, date1, datetime(2011, 1, 11)),
            (0, 1, 1, date2, datetime(2011, 1, 11)),
            (0, 1, 1, date3, datetime(2011, 2, 8)),
            (0, 1, 1, date4, datetime(2011, 2, 8)),
            (0, 0, 1, date1, datetime(2011, 1, 4)),
            (0, 1, 1, date2, datetime(2011, 1, 11)),
            (0, 2, 1, date3, datetime(2011, 1, 18)),
            (0, 3, 1, date4, datetime(2011, 1, 25)),
            (1, 0, 0, date1, datetime(2011, 2, 7)),
            (1, 0, 0, date2, datetime(2011, 2, 7)),
            (1, 0, 0, date3, datetime(2011, 2, 7)),
            (1, 0, 0, date4, datetime(2011, 2, 7)),
            (1, 0, 1, date1, datetime(2011, 2, 1)),
            (1, 0, 1, date2, datetime(2011, 2, 1)),
            (1, 0, 1, date3, datetime(2011, 2, 1)),
            (1, 0, 1, date4, datetime(2011, 2, 1)),
            (1, 0, 2, date1, datetime(2011, 1, 5)),
            (1, 0, 2, date2, datetime(2011, 2, 2)),
            (1, 0, 2, date3, datetime(2011, 2, 2)),
            (1, 0, 2, date4, datetime(2011, 2, 2)),
            (1, 2, 1, date1, datetime(2011, 1, 18)),
            (1, 2, 1, date2, datetime(2011, 1, 18)),
            (1, 2, 1, date3, datetime(2011, 2, 15)),
            (1, 2, 1, date4, datetime(2011, 2, 15)),
            (2, 2, 1, date1, datetime(2011, 2, 15)),
            (2, 2, 1, date2, datetime(2011, 2, 15)),
            (2, 2, 1, date3, datetime(2011, 3, 15)),
            (2, 2, 1, date4, datetime(2011, 3, 15)),
        ]

        for n, week, weekday, dt, expected in test_cases:
            offset = WeekOfMonth(n, week=week, weekday=weekday)
            assert_offset_equal(offset, dt, expected)

        # try subtracting
        result = datetime(2011, 2, 1) - WeekOfMonth(week=1, weekday=2)
        assert result == datetime(2011, 1, 12)

        result = datetime(2011, 2, 3) - WeekOfMonth(week=0, weekday=2)
        assert result == datetime(2011, 2, 2)

    on_offset_cases = [
        (0, 0, datetime(2011, 2, 7), True),
        (0, 0, datetime(2011, 2, 6), False),
        (0, 0, datetime(2011, 2, 14), False),
        (1, 0, datetime(2011, 2, 14), True),
        (0, 1, datetime(2011, 2, 1), True),
        (0, 1, datetime(2011, 2, 8), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        week, weekday, dt, expected = case
        offset = WeekOfMonth(week=week, weekday=weekday)
        assert offset.is_on_offset(dt) == expected

    @pytest.mark.parametrize(
        "n,week,date,tz",
        [
            (2, 2, "1916-05-15 01:14:49.583410462+0422", "Asia/Qyzylorda"),
            (-3, 1, "1980-12-08 03:38:52.878321185+0500", "Asia/Oral"),
        ],
    )
    def test_is_on_offset_nanoseconds(self, n, week, date, tz):
        # GH 18864
        # Make sure that nanoseconds don't trip up is_on_offset (and with it apply)
        offset = WeekOfMonth(n=n, week=week, weekday=0)
        ts = Timestamp(date, tz=tz)
        fast = offset.is_on_offset(ts)
        slow = (ts + offset) - offset == ts
        assert fast == slow


class TestLastWeekOfMonth:
    def test_constructor(self):
        with pytest.raises(ValueError, match="^N cannot be 0"):
            LastWeekOfMonth(n=0, weekday=1)

        with pytest.raises(ValueError, match="^Day"):
            LastWeekOfMonth(n=1, weekday=-1)

        with pytest.raises(ValueError, match="^Day"):
            LastWeekOfMonth(n=1, weekday=7)

    def test_offset(self):
        # Saturday
        last_sat = datetime(2013, 8, 31)
        next_sat = datetime(2013, 9, 28)
        offset_sat = LastWeekOfMonth(n=1, weekday=5)

        one_day_before = last_sat + timedelta(days=-1)
        assert one_day_before + offset_sat == last_sat

        one_day_after = last_sat + timedelta(days=+1)
        assert one_day_after + offset_sat == next_sat

        # Test On that day
        assert last_sat + offset_sat == next_sat

        # Thursday

        offset_thur = LastWeekOfMonth(n=1, weekday=3)
        last_thurs = datetime(2013, 1, 31)
        next_thurs = datetime(2013, 2, 28)

        one_day_before = last_thurs + timedelta(days=-1)
        assert one_day_before + offset_thur == last_thurs

        one_day_after = last_thurs + timedelta(days=+1)
        assert one_day_after + offset_thur == next_thurs

        # Test on that day
        assert last_thurs + offset_thur == next_thurs

        three_before = last_thurs + timedelta(days=-3)
        assert three_before + offset_thur == last_thurs

        two_after = last_thurs + timedelta(days=+2)
        assert two_after + offset_thur == next_thurs

        offset_sunday = LastWeekOfMonth(n=1, weekday=WeekDay.SUN)
        assert datetime(2013, 7, 31) + offset_sunday == datetime(2013, 8, 25)

    on_offset_cases = [
        (WeekDay.SUN, datetime(2013, 1, 27), True),
        (WeekDay.SAT, datetime(2013, 3, 30), True),
        (WeekDay.MON, datetime(2013, 2, 18), False),  # Not the last Mon
        (WeekDay.SUN, datetime(2013, 2, 25), False),  # Not a SUN
        (WeekDay.MON, datetime(2013, 2, 25), True),
        (WeekDay.SAT, datetime(2013, 11, 30), True),
        (WeekDay.SAT, datetime(2006, 8, 26), True),
        (WeekDay.SAT, datetime(2007, 8, 25), True),
        (WeekDay.SAT, datetime(2008, 8, 30), True),
        (WeekDay.SAT, datetime(2009, 8, 29), True),
        (WeekDay.SAT, datetime(2010, 8, 28), True),
        (WeekDay.SAT, datetime(2011, 8, 27), True),
        (WeekDay.SAT, datetime(2019, 8, 31), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        weekday, dt, expected = case
        offset = LastWeekOfMonth(weekday=weekday)
        assert offset.is_on_offset(dt) == expected

    @pytest.mark.parametrize(
        "n,weekday,date,tz",
        [
            (4, 6, "1917-05-27 20:55:27.084284178+0200", "Europe/Warsaw"),
            (-4, 5, "2005-08-27 05:01:42.799392561-0500", "America/Rainy_River"),
        ],
    )
    def test_last_week_of_month_on_offset(self, n, weekday, date, tz):
        # GH 19036, GH 18977 _adjust_dst was incorrect for LastWeekOfMonth
        offset = LastWeekOfMonth(n=n, weekday=weekday)
        ts = Timestamp(date, tz=tz)
        slow = (ts + offset) - offset == ts
        fast = offset.is_on_offset(ts)
        assert fast == slow

    def test_repr(self):
        assert (
            repr(LastWeekOfMonth(n=2, weekday=1)) == "<2 * LastWeekOfMonths: weekday=1>"
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tseries\api.py ===
"""
Timeseries API
"""

from pandas._libs.tslibs.parsing import guess_datetime_format

from pandas.tseries import offsets
from pandas.tseries.frequencies import infer_freq

__all__ = ["infer_freq", "offsets", "guess_datetime_format"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\datetime.py ===
"""For when pip wants to check the date or time."""

import datetime


def today_is_later_than(year: int, month: int, day: int) -> bool:
    today = datetime.date.today()
    given = datetime.date(year, month, day)

    return today > given

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\region.py ===
from typing import NamedTuple


class Region(NamedTuple):
    """Defines a rectangular region of the screen."""

    x: int
    y: int
    width: int
    height: int

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_extension.py ===
from typing import Any


def load_ipython_extension(ip: Any) -> None:  # pragma: no cover
    # prevent circular import
    from pip._vendor.rich.pretty import install
    from pip._vendor.rich.traceback import install as tr_install

    install()
    tr_install()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\tomli\_types.py ===
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Taneli Hukkinen
# Licensed to PSF under a Contributor Agreement.

from typing import Any, Callable, Tuple

# Type annotations
ParseFloat = Callable[[str], Any]
Key = Tuple[str, ...]
Pos = int

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\lextab.py ===
# lextab.py. This file automatically created by PLY (version 3.10). Don't edit!
_tabversion   = '3.10'
_lextokens    = set(('AND', 'ANDEQUAL', 'ARROW', 'AUTO', 'BREAK', 'CASE', 'CHAR', 'CHAR_CONST', 'COLON', 'COMMA', 'CONDOP', 'CONST', 'CONTINUE', 'DEFAULT', 'DIVEQUAL', 'DIVIDE', 'DO', 'DOUBLE', 'ELLIPSIS', 'ELSE', 'ENUM', 'EQ', 'EQUALS', 'EXTERN', 'FLOAT', 'FLOAT_CONST', 'FOR', 'GE', 'GOTO', 'GT', 'HEX_FLOAT_CONST', 'ID', 'IF', 'INLINE', 'INT', 'INT_CONST_BIN', 'INT_CONST_CHAR', 'INT_CONST_DEC', 'INT_CONST_HEX', 'INT_CONST_OCT', 'LAND', 'LBRACE', 'LBRACKET', 'LE', 'LNOT', 'LONG', 'LOR', 'LPAREN', 'LSHIFT', 'LSHIFTEQUAL', 'LT', 'MINUS', 'MINUSEQUAL', 'MINUSMINUS', 'MOD', 'MODEQUAL', 'NE', 'NOT', 'OFFSETOF', 'OR', 'OREQUAL', 'PERIOD', 'PLUS', 'PLUSEQUAL', 'PLUSPLUS', 'PPHASH', 'PPPRAGMA', 'PPPRAGMASTR', 'RBRACE', 'RBRACKET', 'REGISTER', 'RESTRICT', 'RETURN', 'RPAREN', 'RSHIFT', 'RSHIFTEQUAL', 'SEMI', 'SHORT', 'SIGNED', 'SIZEOF', 'STATIC', 'STRING_LITERAL', 'STRUCT', 'SWITCH', 'TIMES', 'TIMESEQUAL', 'TYPEDEF', 'TYPEID', 'U16CHAR_CONST', 'U16STRING_LITERAL', 'U32CHAR_CONST', 'U32STRING_LITERAL', 'U8CHAR_CONST', 'U8STRING_LITERAL', 'UNION', 'UNSIGNED', 'VOID', 'VOLATILE', 'WCHAR_CONST', 'WHILE', 'WSTRING_LITERAL', 'XOR', 'XOREQUAL', '_ALIGNAS', '_ALIGNOF', '_ATOMIC', '_BOOL', '_COMPLEX', '_NORETURN', '_PRAGMA', '_STATIC_ASSERT', '_THREAD_LOCAL', '__INT128'))
_lexreflags   = 64
_lexliterals  = ''
_lexstateinfo = {'INITIAL': 'inclusive', 'ppline': 'exclusive', 'pppragma': 'exclusive'}
_lexstatere   = {'INITIAL': [('(?P<t_PPHASH>[ \\t]*\\#)|(?P<t_NEWLINE>\\n+)|(?P<t_LBRACE>\\{)|(?P<t_RBRACE>\\})|(?P<t_FLOAT_CONST>((((([0-9]*\\.[0-9]+)|([0-9]+\\.))([eE][-+]?[0-9]+)?)|([0-9]+([eE][-+]?[0-9]+)))[FfLl]?))|(?P<t_HEX_FLOAT_CONST>(0[xX]([0-9a-fA-F]+|((([0-9a-fA-F]+)?\\.[0-9a-fA-F]+)|([0-9a-fA-F]+\\.)))([pP][+-]?[0-9]+)[FfLl]?))|(?P<t_INT_CONST_HEX>0[xX][0-9a-fA-F]+(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?)|(?P<t_INT_CONST_BIN>0[bB][01]+(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?)|(?P<t_BAD_CONST_OCT>0[0-7]*[89])|(?P<t_INT_CONST_OCT>0[0-7]*(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?)|(?P<t_INT_CONST_DEC>(0(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?)|([1-9][0-9]*(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?))|(?P<t_INT_CONST_CHAR>\'([^\'\\\\\\n]|(\\\\(([a-wyzA-Z._~!=&\\^\\-\\\\?\'"]|x(?![0-9a-fA-F]))|(\\d+)(?!\\d)|(x[0-9a-fA-F]+)(?![0-9a-fA-F])))){2,4}\')|(?P<t_CHAR_CONST>\'([^\'\\\\\\n]|(\\\\(([a-wyzA-Z._~!=&\\^\\-\\\\?\'"]|x(?![0-9a-fA-F]))|(\\d+)(?!\\d)|(x[0-9a-fA-F]+)(?![0-9a-fA-F]))))\')|(?P<t_WCHAR_CONST>L\'([^\'\\\\\\n]|(\\\\(([a-wyzA-Z._~!=&\\^\\-\\\\?\'"]|x(?![0-9a-fA-F]))|(\\d+)(?!\\d)|(x[0-9a-fA-F]+)(?![0-9a-fA-F]))))\')|(?P<t_U8CHAR_CONST>u8\'([^\'\\\\\\n]|(\\\\(([a-wyzA-Z._~!=&\\^\\-\\\\?\'"]|x(?![0-9a-fA-F]))|(\\d+)(?!\\d)|(x[0-9a-fA-F]+)(?![0-9a-fA-F]))))\')|(?P<t_U16CHAR_CONST>u\'([^\'\\\\\\n]|(\\\\(([a-wyzA-Z._~!=&\\^\\-\\\\?\'"]|x(?![0-9a-fA-F]))|(\\d+)(?!\\d)|(x[0-9a-fA-F]+)(?![0-9a-fA-F]))))\')|(?P<t_U32CHAR_CONST>U\'([^\'\\\\\\n]|(\\\\(([a-wyzA-Z._~!=&\\^\\-\\\\?\'"]|x(?![0-9a-fA-F]))|(\\d+)(?!\\d)|(x[0-9a-fA-F]+)(?![0-9a-fA-F]))))\')|(?P<t_UNMATCHED_QUOTE>(\'([^\'\\\\\\n]|(\\\\(([a-wyzA-Z._~!=&\\^\\-\\\\?\'"]|x(?![0-9a-fA-F]))|(\\d+)(?!\\d)|(x[0-9a-fA-F]+)(?![0-9a-fA-F]))))*\\n)|(\'([^\'\\\\\\n]|(\\\\(([a-wyzA-Z._~!=&\\^\\-\\\\?\'"]|x(?![0-9a-fA-F]))|(\\d+)(?!\\d)|(x[0-9a-fA-F]+)(?![0-9a-fA-F]))))*$))|(?P<t_BAD_CHAR_CONST>(\'([^\'\\\\\\n]|(\\\\(([a-wyzA-Z._~!=&\\^\\-\\\\?\'"]|x(?![0-9a-fA-F]))|(\\d+)(?!\\d)|(x[0-9a-fA-F]+)(?![0-9a-fA-F]))))[^\'\n]+\')|(\'\')|(\'([\\\\][^a-zA-Z._~^!=&\\^\\-\\\\?\'"x0-9])[^\'\\n]*\'))|(?P<t_WSTRING_LITERAL>L"([^"\\\\\\n]|(\\\\[0-9a-zA-Z._~!=&\\^\\-\\\\?\'"]))*")|(?P<t_U8STRING_LITERAL>u8"([^"\\\\\\n]|(\\\\[0-9a-zA-Z._~!=&\\^\\-\\\\?\'"]))*")|(?P<t_U16STRING_LITERAL>u"([^"\\\\\\n]|(\\\\[0-9a-zA-Z._~!=&\\^\\-\\\\?\'"]))*")|(?P<t_U32STRING_LITERAL>U"([^"\\\\\\n]|(\\\\[0-9a-zA-Z._~!=&\\^\\-\\\\?\'"]))*")|(?P<t_BAD_STRING_LITERAL>"([^"\\\\\\n]|(\\\\[0-9a-zA-Z._~!=&\\^\\-\\\\?\'"]))*([\\\\][^a-zA-Z._~^!=&\\^\\-\\\\?\'"x0-9])([^"\\\\\\n]|(\\\\[0-9a-zA-Z._~!=&\\^\\-\\\\?\'"]))*")|(?P<t_ID>[a-zA-Z_$][0-9a-zA-Z_$]*)|(?P<t_STRING_LITERAL>"([^"\\\\\\n]|(\\\\[0-9a-zA-Z._~!=&\\^\\-\\\\?\'"]))*")|(?P<t_ELLIPSIS>\\.\\.\\.)|(?P<t_LOR>\\|\\|)|(?P<t_PLUSPLUS>\\+\\+)|(?P<t_LSHIFTEQUAL><<=)|(?P<t_OREQUAL>\\|=)|(?P<t_PLUSEQUAL>\\+=)|(?P<t_RSHIFTEQUAL>>>=)|(?P<t_TIMESEQUAL>\\*=)|(?P<t_XOREQUAL>\\^=)|(?P<t_ANDEQUAL>&=)|(?P<t_ARROW>->)|(?P<t_CONDOP>\\?)|(?P<t_DIVEQUAL>/=)|(?P<t_EQ>==)|(?P<t_GE>>=)|(?P<t_LAND>&&)|(?P<t_LBRACKET>\\[)|(?P<t_LE><=)|(?P<t_LPAREN>\\()|(?P<t_LSHIFT><<)|(?P<t_MINUSEQUAL>-=)|(?P<t_MINUSMINUS>--)|(?P<t_MODEQUAL>%=)|(?P<t_NE>!=)|(?P<t_OR>\\|)|(?P<t_PERIOD>\\.)|(?P<t_PLUS>\\+)|(?P<t_RBRACKET>\\])|(?P<t_RPAREN>\\))|(?P<t_RSHIFT>>>)|(?P<t_TIMES>\\*)|(?P<t_XOR>\\^)|(?P<t_AND>&)|(?P<t_COLON>:)|(?P<t_COMMA>,)|(?P<t_DIVIDE>/)|(?P<t_EQUALS>=)|(?P<t_GT>>)|(?P<t_LNOT>!)|(?P<t_LT><)|(?P<t_MINUS>-)|(?P<t_MOD>%)|(?P<t_NOT>~)|(?P<t_SEMI>;)', [None, ('t_PPHASH', 'PPHASH'), ('t_NEWLINE', 'NEWLINE'), ('t_LBRACE', 'LBRACE'), ('t_RBRACE', 'RBRACE'), ('t_FLOAT_CONST', 'FLOAT_CONST'), None, None, None, None, None, None, None, None, None, ('t_HEX_FLOAT_CONST', 'HEX_FLOAT_CONST'), None, None, None, None, None, None, None, ('t_INT_CONST_HEX', 'INT_CONST_HEX'), None, None, None, None, None, None, None, ('t_INT_CONST_BIN', 'INT_CONST_BIN'), None, None, None, None, None, None, None, ('t_BAD_CONST_OCT', 'BAD_CONST_OCT'), ('t_INT_CONST_OCT', 'INT_CONST_OCT'), None, None, None, None, None, None, None, ('t_INT_CONST_DEC', 'INT_CONST_DEC'), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, ('t_INT_CONST_CHAR', 'INT_CONST_CHAR'), None, None, None, None, None, None, ('t_CHAR_CONST', 'CHAR_CONST'), None, None, None, None, None, None, ('t_WCHAR_CONST', 'WCHAR_CONST'), None, None, None, None, None, None, ('t_U8CHAR_CONST', 'U8CHAR_CONST'), None, None, None, None, None, None, ('t_U16CHAR_CONST', 'U16CHAR_CONST'), None, None, None, None, None, None, ('t_U32CHAR_CONST', 'U32CHAR_CONST'), None, None, None, None, None, None, ('t_UNMATCHED_QUOTE', 'UNMATCHED_QUOTE'), None, None, None, None, None, None, None, None, None, None, None, None, None, None, ('t_BAD_CHAR_CONST', 'BAD_CHAR_CONST'), None, None, None, None, None, None, None, None, None, None, ('t_WSTRING_LITERAL', 'WSTRING_LITERAL'), None, None, ('t_U8STRING_LITERAL', 'U8STRING_LITERAL'), None, None, ('t_U16STRING_LITERAL', 'U16STRING_LITERAL'), None, None, ('t_U32STRING_LITERAL', 'U32STRING_LITERAL'), None, None, ('t_BAD_STRING_LITERAL', 'BAD_STRING_LITERAL'), None, None, None, None, None, ('t_ID', 'ID'), (None, 'STRING_LITERAL'), None, None, (None, 'ELLIPSIS'), (None, 'LOR'), (None, 'PLUSPLUS'), (None, 'LSHIFTEQUAL'), (None, 'OREQUAL'), (None, 'PLUSEQUAL'), (None, 'RSHIFTEQUAL'), (None, 'TIMESEQUAL'), (None, 'XOREQUAL'), (None, 'ANDEQUAL'), (None, 'ARROW'), (None, 'CONDOP'), (None, 'DIVEQUAL'), (None, 'EQ'), (None, 'GE'), (None, 'LAND'), (None, 'LBRACKET'), (None, 'LE'), (None, 'LPAREN'), (None, 'LSHIFT'), (None, 'MINUSEQUAL'), (None, 'MINUSMINUS'), (None, 'MODEQUAL'), (None, 'NE'), (None, 'OR'), (None, 'PERIOD'), (None, 'PLUS'), (None, 'RBRACKET'), (None, 'RPAREN'), (None, 'RSHIFT'), (None, 'TIMES'), (None, 'XOR'), (None, 'AND'), (None, 'COLON'), (None, 'COMMA'), (None, 'DIVIDE'), (None, 'EQUALS'), (None, 'GT'), (None, 'LNOT'), (None, 'LT'), (None, 'MINUS'), (None, 'MOD'), (None, 'NOT'), (None, 'SEMI')])], 'ppline': [('(?P<t_ppline_FILENAME>"([^"\\\\\\n]|(\\\\[0-9a-zA-Z._~!=&\\^\\-\\\\?\'"]))*")|(?P<t_ppline_LINE_NUMBER>(0(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?)|([1-9][0-9]*(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?))|(?P<t_ppline_NEWLINE>\\n)|(?P<t_ppline_PPLINE>line)', [None, ('t_ppline_FILENAME', 'FILENAME'), None, None, ('t_ppline_LINE_NUMBER', 'LINE_NUMBER'), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, ('t_ppline_NEWLINE', 'NEWLINE'), ('t_ppline_PPLINE', 'PPLINE')])], 'pppragma': [('(?P<t_pppragma_NEWLINE>\\n)|(?P<t_pppragma_PPPRAGMA>pragma)|(?P<t_pppragma_STR>.+)', [None, ('t_pppragma_NEWLINE', 'NEWLINE'), ('t_pppragma_PPPRAGMA', 'PPPRAGMA'), ('t_pppragma_STR', 'STR')])]}
_lexstateignore = {'INITIAL': ' \t', 'ppline': ' \t', 'pppragma': ' \t'}
_lexstateerrorf = {'INITIAL': 't_error', 'ppline': 't_ppline_error', 'pppragma': 't_pppragma_error'}
_lexstateeoff = {}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\tests\test_reductions.py ===
from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.testing.pytest import raises
from sympy.matrices import Matrix, zeros, eye
from sympy.core.symbol import Symbol
from sympy.core.numbers import Rational
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.simplify import simplify
from sympy.abc import x


# Matrix tests
def test_row_op():
    e = eye(3)

    raises(ValueError, lambda: e.elementary_row_op("abc"))
    raises(ValueError, lambda: e.elementary_row_op())
    raises(ValueError, lambda: e.elementary_row_op('n->kn', row=5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->kn', row=-5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=1, row2=5))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=5, row2=1))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=-5, row2=1))
    raises(ValueError, lambda: e.elementary_row_op('n<->m', row1=1, row2=-5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=1, row2=5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=5, row2=1, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=-5, row2=1, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=1, row2=-5, k=5))
    raises(ValueError, lambda: e.elementary_row_op('n->n+km', row1=1, row2=1, k=5))

    # test various ways to set arguments
    assert e.elementary_row_op("n->kn", 0, 5) == Matrix([[5, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->kn", 1, 5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->kn", row=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->kn", row1=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_row_op("n<->m", 0, 1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_row_op("n<->m", row1=0, row2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_row_op("n<->m", row=0, row2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->n+km", 0, 5, 1) == Matrix([[1, 5, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->n+km", row=0, k=5, row2=1) == Matrix([[1, 5, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_row_op("n->n+km", row1=0, k=5, row2=1) == Matrix([[1, 5, 0], [0, 1, 0], [0, 0, 1]])

    # make sure the matrix doesn't change size
    a = Matrix(2, 3, [0]*6)
    assert a.elementary_row_op("n->kn", 1, 5) == Matrix(2, 3, [0]*6)
    assert a.elementary_row_op("n<->m", 0, 1) == Matrix(2, 3, [0]*6)
    assert a.elementary_row_op("n->n+km", 0, 5, 1) == Matrix(2, 3, [0]*6)


def test_col_op():
    e = eye(3)

    raises(ValueError, lambda: e.elementary_col_op("abc"))
    raises(ValueError, lambda: e.elementary_col_op())
    raises(ValueError, lambda: e.elementary_col_op('n->kn', col=5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->kn', col=-5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=1, col2=5))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=5, col2=1))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=-5, col2=1))
    raises(ValueError, lambda: e.elementary_col_op('n<->m', col1=1, col2=-5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=5, col2=1, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=-5, col2=1, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=-5, k=5))
    raises(ValueError, lambda: e.elementary_col_op('n->n+km', col1=1, col2=1, k=5))

    # test various ways to set arguments
    assert e.elementary_col_op("n->kn", 0, 5) == Matrix([[5, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->kn", 1, 5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->kn", col=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->kn", col1=1, k=5) == Matrix([[1, 0, 0], [0, 5, 0], [0, 0, 1]])
    assert e.elementary_col_op("n<->m", 0, 1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op("n<->m", col1=0, col2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op("n<->m", col=0, col2=1) == Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->n+km", 0, 5, 1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->n+km", col=0, k=5, col2=1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])
    assert e.elementary_col_op("n->n+km", col1=0, k=5, col2=1) == Matrix([[1, 0, 0], [5, 1, 0], [0, 0, 1]])

    # make sure the matrix doesn't change size
    a = Matrix(2, 3, [0]*6)
    assert a.elementary_col_op("n->kn", 1, 5) == Matrix(2, 3, [0]*6)
    assert a.elementary_col_op("n<->m", 0, 1) == Matrix(2, 3, [0]*6)
    assert a.elementary_col_op("n->n+km", 0, 5, 1) == Matrix(2, 3, [0]*6)


def test_is_echelon():
    zro = zeros(3)
    ident = eye(3)

    assert zro.is_echelon
    assert ident.is_echelon

    a = Matrix(0, 0, [])
    assert a.is_echelon

    a = Matrix(2, 3, [3, 2, 1, 0, 0, 6])
    assert a.is_echelon

    a = Matrix(2, 3, [0, 0, 6, 3, 2, 1])
    assert not a.is_echelon

    x = Symbol('x')
    a = Matrix(3, 1, [x, 0, 0])
    assert a.is_echelon

    a = Matrix(3, 1, [x, x, 0])
    assert not a.is_echelon

    a = Matrix(3, 3, [0, 0, 0, 1, 2, 3, 0, 0, 0])
    assert not a.is_echelon


def test_echelon_form():
    # echelon form is not unique, but the result
    # must be row-equivalent to the original matrix
    # and it must be in echelon form.

    a = zeros(3)
    e = eye(3)

    # we can assume the zero matrix and the identity matrix shouldn't change
    assert a.echelon_form() == a
    assert e.echelon_form() == e

    a = Matrix(0, 0, [])
    assert a.echelon_form() == a

    a = Matrix(1, 1, [5])
    assert a.echelon_form() == a

    # now we get to the real tests

    def verify_row_null_space(mat, rows, nulls):
        for v in nulls:
            assert all(t.is_zero for t in a_echelon*v)
        for v in rows:
            if not all(t.is_zero for t in v):
                assert not all(t.is_zero for t in a_echelon*v.transpose())

    a = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    nulls = [Matrix([
                     [ 1],
                     [-2],
                     [ 1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)


    a = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 8])
    nulls = []
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)

    a = Matrix(3, 3, [2, 1, 3, 0, 0, 0, 2, 1, 3])
    nulls = [Matrix([
             [Rational(-1, 2)],
             [   1],
             [   0]]),
             Matrix([
             [Rational(-3, 2)],
             [   0],
             [   1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)

    # this one requires a row swap
    a = Matrix(3, 3, [2, 1, 3, 0, 0, 0, 1, 1, 3])
    nulls = [Matrix([
             [   0],
             [  -3],
             [   1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)

    a = Matrix(3, 3, [0, 3, 3, 0, 2, 2, 0, 1, 1])
    nulls = [Matrix([
             [1],
             [0],
             [0]]),
             Matrix([
             [ 0],
             [-1],
             [ 1]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)

    a = Matrix(2, 3, [2, 2, 3, 3, 3, 0])
    nulls = [Matrix([
             [-1],
             [1],
             [0]])]
    rows = [a[i, :] for i in range(a.rows)]
    a_echelon = a.echelon_form()
    assert a_echelon.is_echelon
    verify_row_null_space(a, rows, nulls)


def test_rref():
    e = Matrix(0, 0, [])
    assert e.rref(pivots=False) == e

    e = Matrix(1, 1, [1])
    a = Matrix(1, 1, [5])
    assert e.rref(pivots=False) == a.rref(pivots=False) == e

    a = Matrix(3, 1, [1, 2, 3])
    assert a.rref(pivots=False) == Matrix([[1], [0], [0]])

    a = Matrix(1, 3, [1, 2, 3])
    assert a.rref(pivots=False) == Matrix([[1, 2, 3]])

    a = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert a.rref(pivots=False) == Matrix([
                                     [1, 0, -1],
                                     [0, 1,  2],
                                     [0, 0,  0]])

    a = Matrix(3, 3, [1, 2, 3, 1, 2, 3, 1, 2, 3])
    b = Matrix(3, 3, [1, 2, 3, 0, 0, 0, 0, 0, 0])
    c = Matrix(3, 3, [0, 0, 0, 1, 2, 3, 0, 0, 0])
    d = Matrix(3, 3, [0, 0, 0, 0, 0, 0, 1, 2, 3])
    assert a.rref(pivots=False) == \
            b.rref(pivots=False) == \
            c.rref(pivots=False) == \
            d.rref(pivots=False) == b

    e = eye(3)
    z = zeros(3)
    assert e.rref(pivots=False) == e
    assert z.rref(pivots=False) == z

    a = Matrix([
            [ 0, 0,  1,  2,  2, -5,  3],
            [-1, 5,  2,  2,  1, -7,  5],
            [ 0, 0, -2, -3, -3,  8, -5],
            [-1, 5,  0, -1, -2,  1,  0]])
    mat, pivot_offsets = a.rref()
    assert mat == Matrix([
                     [1, -5, 0, 0, 1,  1, -1],
                     [0,  0, 1, 0, 0, -1,  1],
                     [0,  0, 0, 1, 1, -2,  1],
                     [0,  0, 0, 0, 0,  0,  0]])
    assert pivot_offsets == (0, 2, 3)

    a = Matrix([[Rational(1, 19),  Rational(1, 5),    2,    3],
                        [   4,    5,    6,    7],
                        [   8,    9,   10,   11],
                        [  12,   13,   14,   15]])
    assert a.rref(pivots=False) == Matrix([
                                         [1, 0, 0, Rational(-76, 157)],
                                         [0, 1, 0,  Rational(-5, 157)],
                                         [0, 0, 1, Rational(238, 157)],
                                         [0, 0, 0,       0]])

    x = Symbol('x')
    a = Matrix(2, 3, [x, 1, 1, sqrt(x), x, 1])
    for i, j in zip(a.rref(pivots=False),
            [1, 0, sqrt(x)*(-x + 1)/(-x**Rational(5, 2) + x),
                0, 1, 1/(sqrt(x) + x + 1)]):
        assert simplify(i - j).is_zero


def test_rref_rhs():
    a, b, c, d = symbols('a b c d')
    A = Matrix([[0, 0], [0, 0], [1, 2], [3, 4]])
    B = Matrix([a, b, c, d])
    assert A.rref_rhs(B) == (Matrix([
    [1, 0],
    [0, 1],
    [0, 0],
    [0, 0]]), Matrix([
    [   -2*c + d],
    [3*c/2 - d/2],
    [          a],
    [          b]]))


def test_issue_17827():
    C = Matrix([
        [3, 4, -1, 1],
        [9, 12, -3, 3],
        [0, 2, 1, 3],
        [2, 3, 0, -2],
        [0, 3, 3, -5],
        [8, 15, 0, 6]
    ])
    # Tests for row/col within valid range
    D = C.elementary_row_op('n<->m', row1=2, row2=5)
    E = C.elementary_row_op('n->n+km', row1=5, row2=3, k=-4)
    F = C.elementary_row_op('n->kn', row=5, k=2)
    assert(D[5, :] == Matrix([[0, 2, 1, 3]]))
    assert(E[5, :] == Matrix([[0, 3, 0, 14]]))
    assert(F[5, :] == Matrix([[16, 30, 0, 12]]))
    # Tests for row/col out of range
    raises(ValueError, lambda: C.elementary_row_op('n<->m', row1=2, row2=6))
    raises(ValueError, lambda: C.elementary_row_op('n->kn', row=7, k=2))
    raises(ValueError, lambda: C.elementary_row_op('n->n+km', row1=-1, row2=5, k=2))

def test_rank():
    m = Matrix([[1, 2], [x, 1 - 1/x]])
    assert m.rank() == 2
    n = Matrix(3, 3, range(1, 10))
    assert n.rank() == 2
    p = zeros(3)
    assert p.rank() == 0

def test_issue_11434():
    ax, ay, bx, by, cx, cy, dx, dy, ex, ey, t0, t1 = \
        symbols('a_x a_y b_x b_y c_x c_y d_x d_y e_x e_y t_0 t_1')
    M = Matrix([[ax, ay, ax*t0, ay*t0, 0],
                [bx, by, bx*t0, by*t0, 0],
                [cx, cy, cx*t0, cy*t0, 1],
                [dx, dy, dx*t0, dy*t0, 1],
                [ex, ey, 2*ex*t1 - ex*t0, 2*ey*t1 - ey*t0, 0]])
    assert M.rank() == 4

def test_rank_regression_from_so():
    # see:
    # https://stackoverflow.com/questions/19072700/why-does-sympy-give-me-the-wrong-answer-when-i-row-reduce-a-symbolic-matrix

    nu, lamb = symbols('nu, lambda')
    A = Matrix([[-3*nu,         1,                  0,  0],
                [ 3*nu, -2*nu - 1,                  2,  0],
                [    0,      2*nu, (-1*nu) - lamb - 2,  3],
                [    0,         0,          nu + lamb, -3]])
    expected_reduced = Matrix([[1, 0, 0, 1/(nu**2*(-lamb - nu))],
                               [0, 1, 0,    3/(nu*(-lamb - nu))],
                               [0, 0, 1,         3/(-lamb - nu)],
                               [0, 0, 0,                      0]])
    expected_pivots = (0, 1, 2)

    reduced, pivots = A.rref()

    assert simplify(expected_reduced - reduced) == zeros(*A.shape)
    assert pivots == expected_pivots

def test_issue_15872():
    A = Matrix([[1, 1, 1, 0], [-2, -1, 0, -1], [0, 0, -1, -1], [0, 0, 2, 1]])
    B = A - Matrix.eye(4) * I
    assert B.rank() == 3
    assert (B**2).rank() == 2
    assert (B**3).rank() == 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\continuum_mechanics\__init__.py ===
__all__ = ['Beam',
            'Truss',
            'Cable',
            'Arch'
            ]

from .beam import Beam
from .truss import Truss
from .cable import Cable
from .arch import Arch

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\aseries.py ===
from sympy.core.sympify import sympify


def aseries(expr, x=None, n=6, bound=0, hir=False):
    """
    See the docstring of Expr.aseries() for complete details of this wrapper.

    """
    expr = sympify(expr)
    return expr.aseries(x, n, bound, hir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\benchmarks\bench_order.py ===
from sympy.core.add import Add
from sympy.core.symbol import Symbol
from sympy.series.order import O

x = Symbol('x')
l = [x**i for i in range(1000)]
l.append(O(x**1001))

def timeit_order_1x():
    Add(*l)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\__init__.py ===
"""This module contains code for running the tests in SymPy."""


from .runtests import doctest
from .runtests_pytest import test


__all__ = [
    'test', 'doctest',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\period\test_formats.py ===
from contextlib import nullcontext
from datetime import (
    datetime,
    time,
)
import locale

import numpy as np
import pytest

import pandas as pd
from pandas import (
    PeriodIndex,
    Series,
)
import pandas._testing as tm


def get_local_am_pm():
    """Return the AM and PM strings returned by strftime in current locale."""
    am_local = time(1).strftime("%p")
    pm_local = time(13).strftime("%p")
    return am_local, pm_local


def test_get_values_for_csv():
    index = PeriodIndex(["2017-01-01", "2017-01-02", "2017-01-03"], freq="D")

    # First, with no arguments.
    expected = np.array(["2017-01-01", "2017-01-02", "2017-01-03"], dtype=object)

    result = index._get_values_for_csv()
    tm.assert_numpy_array_equal(result, expected)

    # No NaN values, so na_rep has no effect
    result = index._get_values_for_csv(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)

    # Make sure date formatting works
    expected = np.array(["01-2017-01", "01-2017-02", "01-2017-03"], dtype=object)

    result = index._get_values_for_csv(date_format="%m-%Y-%d")
    tm.assert_numpy_array_equal(result, expected)

    # NULL object handling should work
    index = PeriodIndex(["2017-01-01", pd.NaT, "2017-01-03"], freq="D")
    expected = np.array(["2017-01-01", "NaT", "2017-01-03"], dtype=object)

    result = index._get_values_for_csv(na_rep="NaT")
    tm.assert_numpy_array_equal(result, expected)

    expected = np.array(["2017-01-01", "pandas", "2017-01-03"], dtype=object)

    result = index._get_values_for_csv(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)


class TestPeriodIndexRendering:
    def test_format_empty(self):
        # GH#35712
        empty_idx = PeriodIndex([], freq="Y")
        msg = r"PeriodIndex\.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert empty_idx.format() == []
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert empty_idx.format(name=True) == [""]

    @pytest.mark.parametrize("method", ["__repr__", "__str__"])
    def test_representation(self, method):
        # GH#7601
        idx1 = PeriodIndex([], freq="D")
        idx2 = PeriodIndex(["2011-01-01"], freq="D")
        idx3 = PeriodIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = PeriodIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = PeriodIndex(["2011", "2012", "2013"], freq="Y")
        idx6 = PeriodIndex(["2011-01-01 09:00", "2012-02-01 10:00", "NaT"], freq="h")
        idx7 = pd.period_range("2013Q1", periods=1, freq="Q")
        idx8 = pd.period_range("2013Q1", periods=2, freq="Q")
        idx9 = pd.period_range("2013Q1", periods=3, freq="Q")
        idx10 = PeriodIndex(["2011-01-01", "2011-02-01"], freq="3D")

        exp1 = "PeriodIndex([], dtype='period[D]')"

        exp2 = "PeriodIndex(['2011-01-01'], dtype='period[D]')"

        exp3 = "PeriodIndex(['2011-01-01', '2011-01-02'], dtype='period[D]')"

        exp4 = (
            "PeriodIndex(['2011-01-01', '2011-01-02', '2011-01-03'], "
            "dtype='period[D]')"
        )

        exp5 = "PeriodIndex(['2011', '2012', '2013'], dtype='period[Y-DEC]')"

        exp6 = (
            "PeriodIndex(['2011-01-01 09:00', '2012-02-01 10:00', 'NaT'], "
            "dtype='period[h]')"
        )

        exp7 = "PeriodIndex(['2013Q1'], dtype='period[Q-DEC]')"

        exp8 = "PeriodIndex(['2013Q1', '2013Q2'], dtype='period[Q-DEC]')"

        exp9 = "PeriodIndex(['2013Q1', '2013Q2', '2013Q3'], dtype='period[Q-DEC]')"

        exp10 = "PeriodIndex(['2011-01-01', '2011-02-01'], dtype='period[3D]')"

        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idx10],
            [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10],
        ):
            result = getattr(idx, method)()
            assert result == expected

    # TODO: These are Series.__repr__ tests
    def test_representation_to_series(self):
        # GH#10971
        idx1 = PeriodIndex([], freq="D")
        idx2 = PeriodIndex(["2011-01-01"], freq="D")
        idx3 = PeriodIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = PeriodIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = PeriodIndex(["2011", "2012", "2013"], freq="Y")
        idx6 = PeriodIndex(["2011-01-01 09:00", "2012-02-01 10:00", "NaT"], freq="h")

        idx7 = pd.period_range("2013Q1", periods=1, freq="Q")
        idx8 = pd.period_range("2013Q1", periods=2, freq="Q")
        idx9 = pd.period_range("2013Q1", periods=3, freq="Q")

        exp1 = """Series([], dtype: period[D])"""

        exp2 = """0    2011-01-01
dtype: period[D]"""

        exp3 = """0    2011-01-01
1    2011-01-02
dtype: period[D]"""

        exp4 = """0    2011-01-01
1    2011-01-02
2    2011-01-03
dtype: period[D]"""

        exp5 = """0    2011
1    2012
2    2013
dtype: period[Y-DEC]"""

        exp6 = """0    2011-01-01 09:00
1    2012-02-01 10:00
2                 NaT
dtype: period[h]"""

        exp7 = """0    2013Q1
dtype: period[Q-DEC]"""

        exp8 = """0    2013Q1
1    2013Q2
dtype: period[Q-DEC]"""

        exp9 = """0    2013Q1
1    2013Q2
2    2013Q3
dtype: period[Q-DEC]"""

        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9],
            [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9],
        ):
            result = repr(Series(idx))
            assert result == expected

    def test_summary(self):
        # GH#9116
        idx1 = PeriodIndex([], freq="D")
        idx2 = PeriodIndex(["2011-01-01"], freq="D")
        idx3 = PeriodIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = PeriodIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = PeriodIndex(["2011", "2012", "2013"], freq="Y")
        idx6 = PeriodIndex(["2011-01-01 09:00", "2012-02-01 10:00", "NaT"], freq="h")

        idx7 = pd.period_range("2013Q1", periods=1, freq="Q")
        idx8 = pd.period_range("2013Q1", periods=2, freq="Q")
        idx9 = pd.period_range("2013Q1", periods=3, freq="Q")

        exp1 = """PeriodIndex: 0 entries
Freq: D"""

        exp2 = """PeriodIndex: 1 entries, 2011-01-01 to 2011-01-01
Freq: D"""

        exp3 = """PeriodIndex: 2 entries, 2011-01-01 to 2011-01-02
Freq: D"""

        exp4 = """PeriodIndex: 3 entries, 2011-01-01 to 2011-01-03
Freq: D"""

        exp5 = """PeriodIndex: 3 entries, 2011 to 2013
Freq: Y-DEC"""

        exp6 = """PeriodIndex: 3 entries, 2011-01-01 09:00 to NaT
Freq: h"""

        exp7 = """PeriodIndex: 1 entries, 2013Q1 to 2013Q1
Freq: Q-DEC"""

        exp8 = """PeriodIndex: 2 entries, 2013Q1 to 2013Q2
Freq: Q-DEC"""

        exp9 = """PeriodIndex: 3 entries, 2013Q1 to 2013Q3
Freq: Q-DEC"""

        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9],
            [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9],
        ):
            result = idx._summary()
            assert result == expected


class TestPeriodIndexFormat:
    def test_period_format_and_strftime_default(self):
        per = PeriodIndex([datetime(2003, 1, 1, 12), None], freq="h")

        # Default formatting
        msg = "PeriodIndex.format is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format()
        assert formatted[0] == "2003-01-01 12:00"  # default: minutes not shown
        assert formatted[1] == "NaT"
        # format is equivalent to strftime(None)...
        assert formatted[0] == per.strftime(None)[0]
        assert per.strftime(None)[1] is np.nan  # ...except for NaTs

        # Same test with nanoseconds freq
        per = pd.period_range("2003-01-01 12:01:01.123456789", periods=2, freq="ns")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format()
        assert (formatted == per.strftime(None)).all()
        assert formatted[0] == "2003-01-01 12:01:01.123456789"
        assert formatted[1] == "2003-01-01 12:01:01.123456790"

    def test_period_custom(self):
        # GH#46252 custom formatting directives %l (ms) and %u (us)
        msg = "PeriodIndex.format is deprecated"

        # 3 digits
        per = pd.period_range("2003-01-01 12:01:01.123", periods=2, freq="ms")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format(date_format="%y %I:%M:%S (ms=%l us=%u ns=%n)")
        assert formatted[0] == "03 12:01:01 (ms=123 us=123000 ns=123000000)"
        assert formatted[1] == "03 12:01:01 (ms=124 us=124000 ns=124000000)"

        # 6 digits
        per = pd.period_range("2003-01-01 12:01:01.123456", periods=2, freq="us")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format(date_format="%y %I:%M:%S (ms=%l us=%u ns=%n)")
        assert formatted[0] == "03 12:01:01 (ms=123 us=123456 ns=123456000)"
        assert formatted[1] == "03 12:01:01 (ms=123 us=123457 ns=123457000)"

        # 9 digits
        per = pd.period_range("2003-01-01 12:01:01.123456789", periods=2, freq="ns")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format(date_format="%y %I:%M:%S (ms=%l us=%u ns=%n)")
        assert formatted[0] == "03 12:01:01 (ms=123 us=123456 ns=123456789)"
        assert formatted[1] == "03 12:01:01 (ms=123 us=123456 ns=123456790)"

    def test_period_tz(self):
        # Formatting periods created from a datetime with timezone.
        msg = r"PeriodIndex\.format is deprecated"
        # This timestamp is in 2013 in Europe/Paris but is 2012 in UTC
        dt = pd.to_datetime(["2013-01-01 00:00:00+01:00"], utc=True)

        # Converting to a period looses the timezone information
        # Since tz is currently set as utc, we'll see 2012
        with tm.assert_produces_warning(UserWarning, match="will drop timezone"):
            per = dt.to_period(freq="h")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert per.format()[0] == "2012-12-31 23:00"

        # If tz is currently set as paris before conversion, we'll see 2013
        dt = dt.tz_convert("Europe/Paris")
        with tm.assert_produces_warning(UserWarning, match="will drop timezone"):
            per = dt.to_period(freq="h")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert per.format()[0] == "2013-01-01 00:00"

    @pytest.mark.parametrize(
        "locale_str",
        [
            pytest.param(None, id=str(locale.getlocale())),
            "it_IT.utf8",
            "it_IT",  # Note: encoding will be 'ISO8859-1'
            "zh_CN.utf8",
            "zh_CN",  # Note: encoding will be 'gb2312'
        ],
    )
    def test_period_non_ascii_fmt(self, locale_str):
        # GH#46468 non-ascii char in input format string leads to wrong output

        # Skip if locale cannot be set
        if locale_str is not None and not tm.can_set_locale(locale_str, locale.LC_ALL):
            pytest.skip(f"Skipping as locale '{locale_str}' cannot be set on host.")

        # Change locale temporarily for this test.
        with tm.set_locale(locale_str, locale.LC_ALL) if locale_str else nullcontext():
            # Scalar
            per = pd.Period("2018-03-11 13:00", freq="h")
            assert per.strftime("%y é") == "18 é"

            # Index
            per = pd.period_range("2003-01-01 01:00:00", periods=2, freq="12h")
            msg = "PeriodIndex.format is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=msg):
                formatted = per.format(date_format="%y é")
            assert formatted[0] == "03 é"
            assert formatted[1] == "03 é"

    @pytest.mark.parametrize(
        "locale_str",
        [
            pytest.param(None, id=str(locale.getlocale())),
            "it_IT.utf8",
            "it_IT",  # Note: encoding will be 'ISO8859-1'
            "zh_CN.utf8",
            "zh_CN",  # Note: encoding will be 'gb2312'
        ],
    )
    def test_period_custom_locale_directive(self, locale_str):
        # GH#46319 locale-specific directive leads to non-utf8 c strftime char* result

        # Skip if locale cannot be set
        if locale_str is not None and not tm.can_set_locale(locale_str, locale.LC_ALL):
            pytest.skip(f"Skipping as locale '{locale_str}' cannot be set on host.")

        # Change locale temporarily for this test.
        with tm.set_locale(locale_str, locale.LC_ALL) if locale_str else nullcontext():
            # Get locale-specific reference
            am_local, pm_local = get_local_am_pm()

            # Scalar
            per = pd.Period("2018-03-11 13:00", freq="h")
            assert per.strftime("%p") == pm_local

            # Index
            per = pd.period_range("2003-01-01 01:00:00", periods=2, freq="12h")
            msg = "PeriodIndex.format is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=msg):
                formatted = per.format(date_format="%y %I:%M:%S%p")
            assert formatted[0] == f"03 01:00:00{am_local}"
            assert formatted[1] == f"03 01:00:00{pm_local}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_dense.py ===
from sympy.testing.pytest import raises

from sympy.polys import ZZ, QQ

from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.dense import (
        ddm_transpose,
        ddm_iadd, ddm_isub, ddm_ineg, ddm_imatmul, ddm_imul, ddm_irref,
        ddm_idet, ddm_iinv, ddm_ilu, ddm_ilu_split, ddm_ilu_solve, ddm_berk)

from sympy.polys.matrices.exceptions import (
    DMDomainError,
    DMNonInvertibleMatrixError,
    DMNonSquareMatrixError,
    DMShapeError,
)


def test_ddm_transpose():
    a = [[1, 2], [3, 4]]
    assert ddm_transpose(a) == [[1, 3], [2, 4]]


def test_ddm_iadd():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    ddm_iadd(a, b)
    assert a == [[6, 8], [10, 12]]


def test_ddm_isub():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    ddm_isub(a, b)
    assert a == [[-4, -4], [-4, -4]]


def test_ddm_ineg():
    a = [[1, 2], [3, 4]]
    ddm_ineg(a)
    assert a == [[-1, -2], [-3, -4]]


def test_ddm_matmul():
    a = [[1, 2], [3, 4]]
    ddm_imul(a, 2)
    assert a == [[2, 4], [6, 8]]

    a = [[1, 2], [3, 4]]
    ddm_imul(a, 0)
    assert a == [[0, 0], [0, 0]]


def test_ddm_imatmul():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[1, 2], [3, 4], [5, 6]]

    c1 = [[0, 0], [0, 0]]
    ddm_imatmul(c1, a, b)
    assert c1 == [[22, 28], [49, 64]]

    c2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ddm_imatmul(c2, b, a)
    assert c2 == [[9, 12, 15], [19, 26, 33], [29, 40, 51]]

    b3 = [[1], [2], [3]]
    c3 = [[0], [0]]
    ddm_imatmul(c3, a, b3)
    assert c3 == [[14], [32]]


def test_ddm_irref():
    # Empty matrix
    A = []
    Ar = []
    pivots = []
    assert ddm_irref(A) == pivots
    assert A == Ar

    # Standard square case
    A = [[QQ(0), QQ(1)], [QQ(1), QQ(1)]]
    Ar = [[QQ(1), QQ(0)], [QQ(0), QQ(1)]]
    pivots = [0, 1]
    assert ddm_irref(A) == pivots
    assert A == Ar

    # m < n  case
    A = [[QQ(1), QQ(2), QQ(1)], [QQ(3), QQ(4), QQ(1)]]
    Ar = [[QQ(1), QQ(0), QQ(-1)], [QQ(0), QQ(1), QQ(1)]]
    pivots = [0, 1]
    assert ddm_irref(A) == pivots
    assert A == Ar

    # same m < n  but reversed
    A = [[QQ(3), QQ(4), QQ(1)], [QQ(1), QQ(2), QQ(1)]]
    Ar = [[QQ(1), QQ(0), QQ(-1)], [QQ(0), QQ(1), QQ(1)]]
    pivots = [0, 1]
    assert ddm_irref(A) == pivots
    assert A == Ar

    # m > n case
    A = [[QQ(1), QQ(0)], [QQ(1), QQ(3)], [QQ(0), QQ(1)]]
    Ar = [[QQ(1), QQ(0)], [QQ(0), QQ(1)], [QQ(0), QQ(0)]]
    pivots = [0, 1]
    assert ddm_irref(A) == pivots
    assert A == Ar

    # Example with missing pivot
    A = [[QQ(1), QQ(0), QQ(1)], [QQ(3), QQ(0), QQ(1)]]
    Ar = [[QQ(1), QQ(0), QQ(0)], [QQ(0), QQ(0), QQ(1)]]
    pivots = [0, 2]
    assert ddm_irref(A) == pivots
    assert A == Ar

    # Example with missing pivot and no replacement
    A = [[QQ(0), QQ(1)], [QQ(0), QQ(2)], [QQ(1), QQ(0)]]
    Ar = [[QQ(1), QQ(0)], [QQ(0), QQ(1)], [QQ(0), QQ(0)]]
    pivots = [0, 1]
    assert ddm_irref(A) == pivots
    assert A == Ar


def test_ddm_idet():
    A = []
    assert ddm_idet(A, ZZ) == ZZ(1)

    A = [[ZZ(2)]]
    assert ddm_idet(A, ZZ) == ZZ(2)

    A = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    assert ddm_idet(A, ZZ) == ZZ(-2)

    A = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(3), ZZ(5)]]
    assert ddm_idet(A, ZZ) == ZZ(-1)

    A = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(2), ZZ(5)]]
    assert ddm_idet(A, ZZ) == ZZ(0)

    A = [[QQ(1, 2), QQ(1, 2)], [QQ(1, 3), QQ(1, 4)]]
    assert ddm_idet(A, QQ) == QQ(-1, 24)


def test_ddm_inv():
    A = []
    Ainv = []
    ddm_iinv(Ainv, A, QQ)
    assert Ainv == A

    A = []
    Ainv = []
    raises(DMDomainError, lambda: ddm_iinv(Ainv, A, ZZ))

    A = [[QQ(1), QQ(2)]]
    Ainv = [[QQ(0), QQ(0)]]
    raises(DMNonSquareMatrixError, lambda: ddm_iinv(Ainv, A, QQ))

    A = [[QQ(1, 1), QQ(2, 1)], [QQ(3, 1), QQ(4, 1)]]
    Ainv = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    Ainv_expected = [[QQ(-2, 1), QQ(1, 1)], [QQ(3, 2), QQ(-1, 2)]]
    ddm_iinv(Ainv, A, QQ)
    assert Ainv == Ainv_expected

    A = [[QQ(1, 1), QQ(2, 1)], [QQ(2, 1), QQ(4, 1)]]
    Ainv = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    raises(DMNonInvertibleMatrixError, lambda: ddm_iinv(Ainv, A, QQ))


def test_ddm_ilu():
    A = []
    Alu = []
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []

    A = [[]]
    Alu = [[]]
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []

    A = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    Alu = [[QQ(1), QQ(2)], [QQ(3), QQ(-2)]]
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []

    A = [[QQ(0), QQ(2)], [QQ(3), QQ(4)]]
    Alu = [[QQ(3), QQ(4)], [QQ(0), QQ(2)]]
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == [(0, 1)]

    A = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)], [QQ(7), QQ(8), QQ(9)]]
    Alu = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(-3), QQ(-6)], [QQ(7), QQ(2), QQ(0)]]
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []

    A = [[QQ(0), QQ(1), QQ(2)], [QQ(0), QQ(1), QQ(3)], [QQ(1), QQ(1), QQ(2)]]
    Alu = [[QQ(1), QQ(1), QQ(2)], [QQ(0), QQ(1), QQ(3)], [QQ(0), QQ(1), QQ(-1)]]
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == [(0, 2)]

    A = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]]
    Alu = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(-3), QQ(-6)]]
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []

    A = [[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]]
    Alu = [[QQ(1), QQ(2)], [QQ(3), QQ(-2)], [QQ(5), QQ(2)]]
    swaps = ddm_ilu(A)
    assert A == Alu
    assert swaps == []


def test_ddm_ilu_split():
    U = []
    L = []
    Uexp = []
    Lexp = []
    swaps = ddm_ilu_split(L, U, QQ)
    assert U == Uexp
    assert L == Lexp
    assert swaps == []

    U = [[]]
    L = [[QQ(1)]]
    Uexp = [[]]
    Lexp = [[QQ(1)]]
    swaps = ddm_ilu_split(L, U, QQ)
    assert U == Uexp
    assert L == Lexp
    assert swaps == []

    U = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    L = [[QQ(1), QQ(0)], [QQ(0), QQ(1)]]
    Uexp = [[QQ(1), QQ(2)], [QQ(0), QQ(-2)]]
    Lexp = [[QQ(1), QQ(0)], [QQ(3), QQ(1)]]
    swaps = ddm_ilu_split(L, U, QQ)
    assert U == Uexp
    assert L == Lexp
    assert swaps == []

    U = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]]
    L = [[QQ(1), QQ(0)], [QQ(0), QQ(1)]]
    Uexp = [[QQ(1), QQ(2), QQ(3)], [QQ(0), QQ(-3), QQ(-6)]]
    Lexp = [[QQ(1), QQ(0)], [QQ(4), QQ(1)]]
    swaps = ddm_ilu_split(L, U, QQ)
    assert U == Uexp
    assert L == Lexp
    assert swaps == []

    U = [[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]]
    L = [[QQ(1), QQ(0), QQ(0)], [QQ(0), QQ(1), QQ(0)], [QQ(0), QQ(0), QQ(1)]]
    Uexp = [[QQ(1), QQ(2)], [QQ(0), QQ(-2)], [QQ(0), QQ(0)]]
    Lexp = [[QQ(1), QQ(0), QQ(0)], [QQ(3), QQ(1), QQ(0)], [QQ(5), QQ(2), QQ(1)]]
    swaps = ddm_ilu_split(L, U, QQ)
    assert U == Uexp
    assert L == Lexp
    assert swaps == []


def test_ddm_ilu_solve():
    # Basic example
    # A = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    U = [[QQ(1), QQ(2)], [QQ(0), QQ(-2)]]
    L = [[QQ(1), QQ(0)], [QQ(3), QQ(1)]]
    swaps = []
    b = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DDM([[QQ(0)], [QQ(0)]], (2, 1), QQ)
    xexp = DDM([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    ddm_ilu_solve(x, L, U, swaps, b)
    assert x == xexp

    # Example with swaps
    # A = [[QQ(0), QQ(2)], [QQ(3), QQ(4)]]
    U = [[QQ(3), QQ(4)], [QQ(0), QQ(2)]]
    L = [[QQ(1), QQ(0)], [QQ(0), QQ(1)]]
    swaps = [(0, 1)]
    b = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    x = DDM([[QQ(0)], [QQ(0)]], (2, 1), QQ)
    xexp = DDM([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    ddm_ilu_solve(x, L, U, swaps, b)
    assert x == xexp

    # Overdetermined, consistent
    # A = DDM([[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]], (3, 2), QQ)
    U = [[QQ(1), QQ(2)], [QQ(0), QQ(-2)], [QQ(0), QQ(0)]]
    L = [[QQ(1), QQ(0), QQ(0)], [QQ(3), QQ(1), QQ(0)], [QQ(5), QQ(2), QQ(1)]]
    swaps = []
    b = DDM([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    x = DDM([[QQ(0)], [QQ(0)]], (2, 1), QQ)
    xexp = DDM([[QQ(0)], [QQ(1, 2)]], (2, 1), QQ)
    ddm_ilu_solve(x, L, U, swaps, b)
    assert x == xexp

    # Overdetermined, inconsistent
    b = DDM([[QQ(1)], [QQ(2)], [QQ(4)]], (3, 1), QQ)
    raises(DMNonInvertibleMatrixError, lambda: ddm_ilu_solve(x, L, U, swaps, b))

    # Square, noninvertible
    # A = DDM([[QQ(1), QQ(2)], [QQ(1), QQ(2)]], (2, 2), QQ)
    U = [[QQ(1), QQ(2)], [QQ(0), QQ(0)]]
    L = [[QQ(1), QQ(0)], [QQ(1), QQ(1)]]
    swaps = []
    b = DDM([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    raises(DMNonInvertibleMatrixError, lambda: ddm_ilu_solve(x, L, U, swaps, b))

    # Underdetermined
    # A = DDM([[QQ(1), QQ(2)]], (1, 2), QQ)
    U = [[QQ(1), QQ(2)]]
    L = [[QQ(1)]]
    swaps = []
    b = DDM([[QQ(3)]], (1, 1), QQ)
    raises(NotImplementedError, lambda: ddm_ilu_solve(x, L, U, swaps, b))

    # Shape mismatch
    b3 = DDM([[QQ(1)], [QQ(2)], [QQ(3)]], (3, 1), QQ)
    raises(DMShapeError, lambda: ddm_ilu_solve(x, L, U, swaps, b3))

    # Empty shape mismatch
    U = [[QQ(1)]]
    L = [[QQ(1)]]
    swaps = []
    x = [[QQ(1)]]
    b = []
    raises(DMShapeError, lambda: ddm_ilu_solve(x, L, U, swaps, b))

    # Empty system
    U = []
    L = []
    swaps = []
    b = []
    x = []
    ddm_ilu_solve(x, L, U, swaps, b)
    assert x == []


def test_ddm_charpoly():
    A = []
    assert ddm_berk(A, ZZ) == [[ZZ(1)]]

    A = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]]
    Avec = [[ZZ(1)], [ZZ(-15)], [ZZ(-18)], [ZZ(0)]]
    assert ddm_berk(A, ZZ) == Avec

    A = DDM([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: ddm_berk(A, ZZ))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_sorting.py ===
import numpy as np
import pytest

from pandas.errors import (
    PerformanceWarning,
    UnsortedIndexError,
)

from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.indexes.frozen import FrozenList


def test_sortlevel(idx):
    tuples = list(idx)
    np.random.default_rng(2).shuffle(tuples)

    index = MultiIndex.from_tuples(tuples)

    sorted_idx, _ = index.sortlevel(0)
    expected = MultiIndex.from_tuples(sorted(tuples))
    assert sorted_idx.equals(expected)

    sorted_idx, _ = index.sortlevel(0, ascending=False)
    assert sorted_idx.equals(expected[::-1])

    sorted_idx, _ = index.sortlevel(1)
    by1 = sorted(tuples, key=lambda x: (x[1], x[0]))
    expected = MultiIndex.from_tuples(by1)
    assert sorted_idx.equals(expected)

    sorted_idx, _ = index.sortlevel(1, ascending=False)
    assert sorted_idx.equals(expected[::-1])


def test_sortlevel_not_sort_remaining():
    mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
    sorted_idx, _ = mi.sortlevel("A", sort_remaining=False)
    assert sorted_idx.equals(mi)


def test_sortlevel_deterministic():
    tuples = [
        ("bar", "one"),
        ("foo", "two"),
        ("qux", "two"),
        ("foo", "one"),
        ("baz", "two"),
        ("qux", "one"),
    ]

    index = MultiIndex.from_tuples(tuples)

    sorted_idx, _ = index.sortlevel(0)
    expected = MultiIndex.from_tuples(sorted(tuples))
    assert sorted_idx.equals(expected)

    sorted_idx, _ = index.sortlevel(0, ascending=False)
    assert sorted_idx.equals(expected[::-1])

    sorted_idx, _ = index.sortlevel(1)
    by1 = sorted(tuples, key=lambda x: (x[1], x[0]))
    expected = MultiIndex.from_tuples(by1)
    assert sorted_idx.equals(expected)

    sorted_idx, _ = index.sortlevel(1, ascending=False)
    assert sorted_idx.equals(expected[::-1])


def test_sortlevel_na_position():
    # GH#51612
    midx = MultiIndex.from_tuples([(1, np.nan), (1, 1)])
    result = midx.sortlevel(level=[0, 1], na_position="last")[0]
    expected = MultiIndex.from_tuples([(1, 1), (1, np.nan)])
    tm.assert_index_equal(result, expected)


def test_numpy_argsort(idx):
    result = np.argsort(idx)
    expected = idx.argsort()
    tm.assert_numpy_array_equal(result, expected)

    # these are the only two types that perform
    # pandas compatibility input validation - the
    # rest already perform separate (or no) such
    # validation via their 'values' attribute as
    # defined in pandas.core.indexes/base.py - they
    # cannot be changed at the moment due to
    # backwards compatibility concerns
    if isinstance(type(idx), (CategoricalIndex, RangeIndex)):
        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argsort(idx, axis=1)

        msg = "the 'kind' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argsort(idx, kind="mergesort")

        msg = "the 'order' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argsort(idx, order=("a", "b"))


def test_unsortedindex():
    # GH 11897
    mi = MultiIndex.from_tuples(
        [("z", "a"), ("x", "a"), ("y", "b"), ("x", "b"), ("y", "a"), ("z", "b")],
        names=["one", "two"],
    )
    df = DataFrame([[i, 10 * i] for i in range(6)], index=mi, columns=["one", "two"])

    # GH 16734: not sorted, but no real slicing
    result = df.loc(axis=0)["z", "a"]
    expected = df.iloc[0]
    tm.assert_series_equal(result, expected)

    msg = (
        "MultiIndex slicing requires the index to be lexsorted: "
        r"slicing on levels \[1\], lexsort depth 0"
    )
    with pytest.raises(UnsortedIndexError, match=msg):
        df.loc(axis=0)["z", slice("a")]
    df.sort_index(inplace=True)
    assert len(df.loc(axis=0)["z", :]) == 2

    with pytest.raises(KeyError, match="'q'"):
        df.loc(axis=0)["q", :]


def test_unsortedindex_doc_examples():
    # https://pandas.pydata.org/pandas-docs/stable/advanced.html#sorting-a-multiindex
    dfm = DataFrame(
        {
            "jim": [0, 0, 1, 1],
            "joe": ["x", "x", "z", "y"],
            "jolie": np.random.default_rng(2).random(4),
        }
    )

    dfm = dfm.set_index(["jim", "joe"])
    with tm.assert_produces_warning(PerformanceWarning):
        dfm.loc[(1, "z")]

    msg = r"Key length \(2\) was greater than MultiIndex lexsort depth \(1\)"
    with pytest.raises(UnsortedIndexError, match=msg):
        dfm.loc[(0, "y"):(1, "z")]

    assert not dfm.index._is_lexsorted()
    assert dfm.index._lexsort_depth == 1

    # sort it
    dfm = dfm.sort_index()
    dfm.loc[(1, "z")]
    dfm.loc[(0, "y"):(1, "z")]

    assert dfm.index._is_lexsorted()
    assert dfm.index._lexsort_depth == 2


def test_reconstruct_sort():
    # starts off lexsorted & monotonic
    mi = MultiIndex.from_arrays([["A", "A", "B", "B", "B"], [1, 2, 1, 2, 3]])
    assert mi.is_monotonic_increasing
    recons = mi._sort_levels_monotonic()
    assert recons.is_monotonic_increasing
    assert mi is recons

    assert mi.equals(recons)
    assert Index(mi.values).equals(Index(recons.values))

    # cannot convert to lexsorted
    mi = MultiIndex.from_tuples(
        [("z", "a"), ("x", "a"), ("y", "b"), ("x", "b"), ("y", "a"), ("z", "b")],
        names=["one", "two"],
    )
    assert not mi.is_monotonic_increasing
    recons = mi._sort_levels_monotonic()
    assert not recons.is_monotonic_increasing
    assert mi.equals(recons)
    assert Index(mi.values).equals(Index(recons.values))

    # cannot convert to lexsorted
    mi = MultiIndex(
        levels=[["b", "d", "a"], [1, 2, 3]],
        codes=[[0, 1, 0, 2], [2, 0, 0, 1]],
        names=["col1", "col2"],
    )
    assert not mi.is_monotonic_increasing
    recons = mi._sort_levels_monotonic()
    assert not recons.is_monotonic_increasing
    assert mi.equals(recons)
    assert Index(mi.values).equals(Index(recons.values))


def test_reconstruct_remove_unused():
    # xref to GH 2770
    df = DataFrame(
        [["deleteMe", 1, 9], ["keepMe", 2, 9], ["keepMeToo", 3, 9]],
        columns=["first", "second", "third"],
    )
    df2 = df.set_index(["first", "second"], drop=False)
    df2 = df2[df2["first"] != "deleteMe"]

    # removed levels are there
    expected = MultiIndex(
        levels=[["deleteMe", "keepMe", "keepMeToo"], [1, 2, 3]],
        codes=[[1, 2], [1, 2]],
        names=["first", "second"],
    )
    result = df2.index
    tm.assert_index_equal(result, expected)

    expected = MultiIndex(
        levels=[["keepMe", "keepMeToo"], [2, 3]],
        codes=[[0, 1], [0, 1]],
        names=["first", "second"],
    )
    result = df2.index.remove_unused_levels()
    tm.assert_index_equal(result, expected)

    # idempotent
    result2 = result.remove_unused_levels()
    tm.assert_index_equal(result2, expected)
    assert result2.is_(result)


@pytest.mark.parametrize(
    "first_type,second_type", [("int64", "int64"), ("datetime64[D]", "str")]
)
def test_remove_unused_levels_large(first_type, second_type):
    # GH16556

    # because tests should be deterministic (and this test in particular
    # checks that levels are removed, which is not the case for every
    # random input):
    rng = np.random.default_rng(10)  # seed is arbitrary value that works

    size = 1 << 16
    df = DataFrame(
        {
            "first": rng.integers(0, 1 << 13, size).astype(first_type),
            "second": rng.integers(0, 1 << 10, size).astype(second_type),
            "third": rng.random(size),
        }
    )
    df = df.groupby(["first", "second"]).sum()
    df = df[df.third < 0.1]

    result = df.index.remove_unused_levels()
    assert len(result.levels[0]) < len(df.index.levels[0])
    assert len(result.levels[1]) < len(df.index.levels[1])
    assert result.equals(df.index)

    expected = df.reset_index().set_index(["first", "second"]).index
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("level0", [["a", "d", "b"], ["a", "d", "b", "unused"]])
@pytest.mark.parametrize(
    "level1", [["w", "x", "y", "z"], ["w", "x", "y", "z", "unused"]]
)
def test_remove_unused_nan(level0, level1):
    # GH 18417
    mi = MultiIndex(levels=[level0, level1], codes=[[0, 2, -1, 1, -1], [0, 1, 2, 3, 2]])

    result = mi.remove_unused_levels()
    tm.assert_index_equal(result, mi)
    for level in 0, 1:
        assert "unused" not in result.levels[level]


def test_argsort(idx):
    result = idx.argsort()
    expected = idx.values.argsort()
    tm.assert_numpy_array_equal(result, expected)


def test_remove_unused_levels_with_nan():
    # GH 37510
    idx = Index([(1, np.nan), (3, 4)]).rename(["id1", "id2"])
    idx = idx.set_levels(["a", np.nan], level="id1")
    idx = idx.remove_unused_levels()
    result = idx.levels
    expected = FrozenList([["a", np.nan], [4]])
    assert str(result) == str(expected)


def test_sort_values_nan():
    # GH48495, GH48626
    midx = MultiIndex(levels=[["A", "B", "C"], ["D"]], codes=[[1, 0, 2], [-1, -1, 0]])
    result = midx.sort_values()
    expected = MultiIndex(
        levels=[["A", "B", "C"], ["D"]], codes=[[0, 1, 2], [-1, -1, 0]]
    )
    tm.assert_index_equal(result, expected)


def test_sort_values_incomparable():
    # GH48495
    mi = MultiIndex.from_arrays(
        [
            [1, Timestamp("2000-01-01")],
            [3, 4],
        ]
    )
    match = "'<' not supported between instances of 'Timestamp' and 'int'"
    with pytest.raises(TypeError, match=match):
        mi.sort_values()


@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.parametrize("dtype", ["float64", "Int64", "Float64"])
def test_sort_values_with_na_na_position(dtype, na_position):
    # 51612
    arrays = [
        Series([1, 1, 2], dtype=dtype),
        Series([1, None, 3], dtype=dtype),
    ]
    index = MultiIndex.from_arrays(arrays)
    result = index.sort_values(na_position=na_position)
    if na_position == "first":
        arrays = [
            Series([1, 1, 2], dtype=dtype),
            Series([None, 1, 3], dtype=dtype),
        ]
    else:
        arrays = [
            Series([1, 1, 2], dtype=dtype),
            Series([1, None, 3], dtype=dtype),
        ]
    expected = MultiIndex.from_arrays(arrays)
    tm.assert_index_equal(result, expected)


def test_sort_unnecessary_warning():
    # GH#55386
    midx = MultiIndex.from_tuples([(1.5, 2), (3.5, 3), (0, 1)])
    midx = midx.set_levels([2.5, np.nan, 1], level=0)
    result = midx.sort_values()
    expected = MultiIndex.from_tuples([(1, 3), (2.5, 1), (np.nan, 2)])
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\tests\test_residue.py ===
from collections import defaultdict
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.numbers import totient
from sympy.ntheory import n_order, is_primitive_root, is_quad_residue, \
    legendre_symbol, jacobi_symbol, primerange, sqrt_mod, \
    primitive_root, quadratic_residues, is_nthpow_residue, nthroot_mod, \
    sqrt_mod_iter, mobius, discrete_log, quadratic_congruence, \
    polynomial_congruence, sieve
from sympy.ntheory.residue_ntheory import _primitive_root_prime_iter, \
    _primitive_root_prime_power_iter, _primitive_root_prime_power2_iter, \
    _nthroot_mod_prime_power, _discrete_log_trial_mul, _discrete_log_shanks_steps, \
    _discrete_log_pollard_rho, _discrete_log_index_calculus, _discrete_log_pohlig_hellman, \
    _binomial_mod_prime_power, binomial_mod
from sympy.polys.domains import ZZ
from sympy.testing.pytest import raises
from sympy.core.random import randint, choice


def test_residue():
    assert n_order(2, 13) == 12
    assert [n_order(a, 7) for a in range(1, 7)] == \
           [1, 3, 6, 3, 6, 2]
    assert n_order(5, 17) == 16
    assert n_order(17, 11) == n_order(6, 11)
    assert n_order(101, 119) == 6
    assert n_order(11, (10**50 + 151)**2) == 10000000000000000000000000000000000000000000000030100000000000000000000000000000000000000000000022650
    raises(ValueError, lambda: n_order(6, 9))

    assert is_primitive_root(2, 7) is False
    assert is_primitive_root(3, 8) is False
    assert is_primitive_root(11, 14) is False
    assert is_primitive_root(12, 17) == is_primitive_root(29, 17)
    raises(ValueError, lambda: is_primitive_root(3, 6))

    for p in primerange(3, 100):
        li = list(_primitive_root_prime_iter(p))
        assert li[0] == min(li)
        for g in li:
            assert n_order(g, p) == p - 1
        assert len(li) == totient(totient(p))
        for e in range(1, 4):
            li_power = list(_primitive_root_prime_power_iter(p, e))
            li_power2 = list(_primitive_root_prime_power2_iter(p, e))
            assert len(li_power) == len(li_power2) == totient(totient(p**e))
    assert primitive_root(97) == 5
    assert n_order(primitive_root(97, False), 97) == totient(97)
    assert primitive_root(97**2) == 5
    assert n_order(primitive_root(97**2, False), 97**2) == totient(97**2)
    assert primitive_root(40487) == 5
    assert n_order(primitive_root(40487, False), 40487) == totient(40487)
    # note that primitive_root(40487) + 40487 = 40492 is a primitive root
    # of 40487**2, but it is not the smallest
    assert primitive_root(40487**2) == 10
    assert n_order(primitive_root(40487**2, False), 40487**2) == totient(40487**2)
    assert primitive_root(82) == 7
    assert n_order(primitive_root(82, False), 82) == totient(82)
    p = 10**50 + 151
    assert primitive_root(p) == 11
    assert n_order(primitive_root(p, False), p) == totient(p)
    assert primitive_root(2*p) == 11
    assert n_order(primitive_root(2*p, False), 2*p) == totient(2*p)
    assert primitive_root(p**2) == 11
    assert n_order(primitive_root(p**2, False), p**2) == totient(p**2)
    assert primitive_root(4 * 11) is None and primitive_root(4 * 11, False) is None
    assert primitive_root(15) is None and primitive_root(15, False) is None
    raises(ValueError, lambda: primitive_root(-3))

    assert is_quad_residue(3, 7) is False
    assert is_quad_residue(10, 13) is True
    assert is_quad_residue(12364, 139) == is_quad_residue(12364 % 139, 139)
    assert is_quad_residue(207, 251) is True
    assert is_quad_residue(0, 1) is True
    assert is_quad_residue(1, 1) is True
    assert is_quad_residue(0, 2) == is_quad_residue(1, 2) is True
    assert is_quad_residue(1, 4) is True
    assert is_quad_residue(2, 27) is False
    assert is_quad_residue(13122380800, 13604889600) is True
    assert [j for j in range(14) if is_quad_residue(j, 14)] == \
           [0, 1, 2, 4, 7, 8, 9, 11]
    raises(ValueError, lambda: is_quad_residue(1.1, 2))
    raises(ValueError, lambda: is_quad_residue(2, 0))

    assert quadratic_residues(S.One) == [0]
    assert quadratic_residues(1) == [0]
    assert quadratic_residues(12) == [0, 1, 4, 9]
    assert quadratic_residues(13) == [0, 1, 3, 4, 9, 10, 12]
    assert [len(quadratic_residues(i)) for i in range(1, 20)] == \
      [1, 2, 2, 2, 3, 4, 4, 3, 4, 6, 6, 4, 7, 8, 6, 4, 9, 8, 10]

    assert list(sqrt_mod_iter(6, 2)) == [0]
    assert sqrt_mod(3, 13) == 4
    assert sqrt_mod(3, -13) == 4
    assert sqrt_mod(6, 23) == 11
    assert sqrt_mod(345, 690) == 345
    assert sqrt_mod(67, 101) == None
    assert sqrt_mod(1020, 104729) == None

    for p in range(3, 100):
        d = defaultdict(list)
        for i in range(p):
            d[pow(i, 2, p)].append(i)
        for i in range(1, p):
            it = sqrt_mod_iter(i, p)
            v = sqrt_mod(i, p, True)
            if v:
                v = sorted(v)
                assert d[i] == v
            else:
                assert not d[i]

    assert sqrt_mod(9, 27, True) == [3, 6, 12, 15, 21, 24]
    assert sqrt_mod(9, 81, True) == [3, 24, 30, 51, 57, 78]
    assert sqrt_mod(9, 3**5, True) == [3, 78, 84, 159, 165, 240]
    assert sqrt_mod(81, 3**4, True) == [0, 9, 18, 27, 36, 45, 54, 63, 72]
    assert sqrt_mod(81, 3**5, True) == [9, 18, 36, 45, 63, 72, 90, 99, 117,\
            126, 144, 153, 171, 180, 198, 207, 225, 234]
    assert sqrt_mod(81, 3**6, True) == [9, 72, 90, 153, 171, 234, 252, 315,\
            333, 396, 414, 477, 495, 558, 576, 639, 657, 720]
    assert sqrt_mod(81, 3**7, True) == [9, 234, 252, 477, 495, 720, 738, 963,\
            981, 1206, 1224, 1449, 1467, 1692, 1710, 1935, 1953, 2178]

    for a, p in [(26214400, 32768000000), (26214400, 16384000000),
        (262144, 1048576), (87169610025, 163443018796875),
        (22315420166400, 167365651248000000)]:
        assert pow(sqrt_mod(a, p), 2, p) == a

    n = 70
    a, p = 5**2*3**n*2**n, 5**6*3**(n+1)*2**(n+2)
    it = sqrt_mod_iter(a, p)
    for i in range(10):
        assert pow(next(it), 2, p) == a
    a, p = 5**2*3**n*2**n, 5**6*3**(n+1)*2**(n+3)
    it = sqrt_mod_iter(a, p)
    for i in range(2):
        assert pow(next(it), 2, p) == a
    n = 100
    a, p = 5**2*3**n*2**n, 5**6*3**(n+1)*2**(n+1)
    it = sqrt_mod_iter(a, p)
    for i in range(2):
        assert pow(next(it), 2, p) == a

    assert type(next(sqrt_mod_iter(9, 27))) is int
    assert type(next(sqrt_mod_iter(9, 27, ZZ))) is type(ZZ(1))
    assert type(next(sqrt_mod_iter(1, 7, ZZ))) is type(ZZ(1))

    assert is_nthpow_residue(2, 1, 5)

    #issue 10816
    assert is_nthpow_residue(1, 0, 1) is False
    assert is_nthpow_residue(1, 0, 2) is True
    assert is_nthpow_residue(3, 0, 2) is True
    assert is_nthpow_residue(0, 1, 8) is True
    assert is_nthpow_residue(2, 3, 2) is True
    assert is_nthpow_residue(2, 3, 9) is False
    assert is_nthpow_residue(3, 5, 30) is True
    assert is_nthpow_residue(21, 11, 20) is True
    assert is_nthpow_residue(7, 10, 20) is False
    assert is_nthpow_residue(5, 10, 20) is True
    assert is_nthpow_residue(3, 10, 48) is False
    assert is_nthpow_residue(1, 10, 40) is True
    assert is_nthpow_residue(3, 10, 24) is False
    assert is_nthpow_residue(1, 10, 24) is True
    assert is_nthpow_residue(3, 10, 24) is False
    assert is_nthpow_residue(2, 10, 48) is False
    assert is_nthpow_residue(81, 3, 972) is False
    assert is_nthpow_residue(243, 5, 5103) is True
    assert is_nthpow_residue(243, 3, 1240029) is False
    assert is_nthpow_residue(36010, 8, 87382) is True
    assert is_nthpow_residue(28552, 6, 2218) is True
    assert is_nthpow_residue(92712, 9, 50026) is True
    x = {pow(i, 56, 1024) for i in range(1024)}
    assert {a for a in range(1024) if is_nthpow_residue(a, 56, 1024)} == x
    x = { pow(i, 256, 2048) for i in range(2048)}
    assert {a for a in range(2048) if is_nthpow_residue(a, 256, 2048)} == x
    x = { pow(i, 11, 324000) for i in range(1000)}
    assert [ is_nthpow_residue(a, 11, 324000) for a in x]
    x = { pow(i, 17, 22217575536) for i in range(1000)}
    assert [ is_nthpow_residue(a, 17, 22217575536) for a in x]
    assert is_nthpow_residue(676, 3, 5364)
    assert is_nthpow_residue(9, 12, 36)
    assert is_nthpow_residue(32, 10, 41)
    assert is_nthpow_residue(4, 2, 64)
    assert is_nthpow_residue(31, 4, 41)
    assert not is_nthpow_residue(2, 2, 5)
    assert is_nthpow_residue(8547, 12, 10007)
    assert is_nthpow_residue(Dummy(even=True) + 3, 3, 2) == True
    # _nthroot_mod_prime_power
    for p in primerange(2, 10):
        for a in range(3):
            for n in range(3, 5):
                ans = _nthroot_mod_prime_power(a, n, p, 1)
                assert isinstance(ans, list)
                if len(ans) == 0:
                    for b in range(p):
                        assert pow(b, n, p) != a % p
                    for k in range(2, 10):
                        assert _nthroot_mod_prime_power(a, n, p, k) == []
                else:
                    for b in range(p):
                        pred = pow(b, n, p) == a % p
                        assert not(pred ^ (b in ans))
                    for k in range(2, 10):
                        ans = _nthroot_mod_prime_power(a, n, p, k)
                        if not ans:
                            break
                        for b in ans:
                            assert pow(b, n , p**k) == a

    assert nthroot_mod(Dummy(odd=True), 3, 2) == 1
    assert nthroot_mod(29, 31, 74) == 45
    assert nthroot_mod(1801, 11, 2663) == 44
    for a, q, p in [(51922, 2, 203017), (43, 3, 109), (1801, 11, 2663),
          (26118163, 1303, 33333347), (1499, 7, 2663), (595, 6, 2663),
          (1714, 12, 2663), (28477, 9, 33343)]:
        r = nthroot_mod(a, q, p)
        assert pow(r, q, p) == a
    assert nthroot_mod(11, 3, 109) is None
    assert nthroot_mod(16, 5, 36, True) == [4, 22]
    assert nthroot_mod(9, 16, 36, True) == [3, 9, 15, 21, 27, 33]
    assert nthroot_mod(4, 3, 3249000) is None
    assert nthroot_mod(36010, 8, 87382, True) == [40208, 47174]
    assert nthroot_mod(0, 12, 37, True) == [0]
    assert nthroot_mod(0, 7, 100, True) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert nthroot_mod(4, 4, 27, True) == [5, 22]
    assert nthroot_mod(4, 4, 121, True) == [19, 102]
    assert nthroot_mod(2, 3, 7, True) == []
    for p in range(1, 20):
        for a in range(p):
            for n in range(1, p):
                ans = nthroot_mod(a, n, p, True)
                assert isinstance(ans, list)
                for b in range(p):
                    pred = pow(b, n, p) == a
                    assert not(pred ^ (b in ans))
                ans2 = nthroot_mod(a, n, p, False)
                if ans2 is None:
                    assert ans == []
                else:
                    assert ans2 in ans

    x = Symbol('x', positive=True)
    i = Symbol('i', integer=True)
    assert _discrete_log_trial_mul(587, 2**7, 2) == 7
    assert _discrete_log_trial_mul(941, 7**18, 7) == 18
    assert _discrete_log_trial_mul(389, 3**81, 3) == 81
    assert _discrete_log_trial_mul(191, 19**123, 19) == 123
    assert _discrete_log_shanks_steps(442879, 7**2, 7) == 2
    assert _discrete_log_shanks_steps(874323, 5**19, 5) == 19
    assert _discrete_log_shanks_steps(6876342, 7**71, 7) == 71
    assert _discrete_log_shanks_steps(2456747, 3**321, 3) == 321
    assert _discrete_log_pollard_rho(6013199, 2**6, 2, rseed=0) == 6
    assert _discrete_log_pollard_rho(6138719, 2**19, 2, rseed=0) == 19
    assert _discrete_log_pollard_rho(36721943, 2**40, 2, rseed=0) == 40
    assert _discrete_log_pollard_rho(24567899, 3**333, 3, rseed=0) == 333
    raises(ValueError, lambda: _discrete_log_pollard_rho(11, 7, 31, rseed=0))
    raises(ValueError, lambda: _discrete_log_pollard_rho(227, 3**7, 5, rseed=0))
    assert _discrete_log_index_calculus(983, 948, 2, 491) == 183
    assert _discrete_log_index_calculus(633383, 21794, 2, 316691) == 68048
    assert _discrete_log_index_calculus(941762639, 68822582, 2, 470881319) == 338029275
    assert _discrete_log_index_calculus(999231337607, 888188918786, 2, 499615668803) == 142811376514
    assert _discrete_log_index_calculus(47747730623, 19410045286, 43425105668, 645239603) == 590504662
    assert _discrete_log_pohlig_hellman(98376431, 11**9, 11) == 9
    assert _discrete_log_pohlig_hellman(78723213, 11**31, 11) == 31
    assert _discrete_log_pohlig_hellman(32942478, 11**98, 11) == 98
    assert _discrete_log_pohlig_hellman(14789363, 11**444, 11) == 444
    assert discrete_log(1, 0, 2) == 0
    raises(ValueError, lambda: discrete_log(-4, 1, 3))
    raises(ValueError, lambda: discrete_log(10, 3, 2))
    assert discrete_log(587, 2**9, 2) == 9
    assert discrete_log(2456747, 3**51, 3) == 51
    assert discrete_log(32942478, 11**127, 11) == 127
    assert discrete_log(432751500361, 7**324, 7) == 324
    assert discrete_log(265390227570863,184500076053622, 2) == 17835221372061
    assert discrete_log(22708823198678103974314518195029102158525052496759285596453269189798311427475159776411276642277139650833937,
                        17463946429475485293747680247507700244427944625055089103624311227422110546803452417458985046168310373075327,
                        123456) == 2068031853682195777930683306640554533145512201725884603914601918777510185469769997054750835368413389728895
    args = 5779, 3528, 6215
    assert discrete_log(*args) == 687
    assert discrete_log(*Tuple(*args)) == 687
    assert quadratic_congruence(400, 85, 125, 1600) == [295, 615, 935, 1255, 1575]
    assert quadratic_congruence(3, 6, 5, 25) == [3, 20]
    assert quadratic_congruence(120, 80, 175, 500) == []
    assert quadratic_congruence(15, 14, 7, 2) == [1]
    assert quadratic_congruence(8, 15, 7, 29) == [10, 28]
    assert quadratic_congruence(160, 200, 300, 461) == [144, 431]
    assert quadratic_congruence(100000, 123456, 7415263, 48112959837082048697) == [30417843635344493501, 36001135160550533083]
    assert quadratic_congruence(65, 121, 72, 277) == [249, 252]
    assert quadratic_congruence(5, 10, 14, 2) == [0]
    assert quadratic_congruence(10, 17, 19, 2) == [1]
    assert quadratic_congruence(10, 14, 20, 2) == [0, 1]
    assert quadratic_congruence(2**48-7, 2**48-1, 4, 2**48) == [8249717183797, 31960993774868]
    assert polynomial_congruence(6*x**5 + 10*x**4 + 5*x**3 + x**2 + x + 1,
        972000) == [220999, 242999, 463999, 485999, 706999, 728999, 949999, 971999]

    assert polynomial_congruence(x**3 - 10*x**2 + 12*x - 82, 33075) == [30287]
    assert polynomial_congruence(x**2 + x + 47, 2401) == [785, 1615]
    assert polynomial_congruence(10*x**2 + 14*x + 20, 2) == [0, 1]
    assert polynomial_congruence(x**3 + 3, 16) == [5]
    assert polynomial_congruence(65*x**2 + 121*x + 72, 277) == [249, 252]
    assert polynomial_congruence(x**4 - 4, 27) == [5, 22]
    assert polynomial_congruence(35*x**3 - 6*x**2 - 567*x + 2308, 148225) == [86957,
        111157, 122531, 146731]
    assert polynomial_congruence(x**16 - 9, 36) == [3, 9, 15, 21, 27, 33]
    assert polynomial_congruence(x**6 - 2*x**5 - 35, 6125) == [3257]
    raises(ValueError, lambda: polynomial_congruence(x**x, 6125))
    raises(ValueError, lambda: polynomial_congruence(x**i, 6125))
    raises(ValueError, lambda: polynomial_congruence(0.1*x**2 + 6, 100))

    assert binomial_mod(-1, 1, 10) == 0
    assert binomial_mod(1, -1, 10) == 0
    raises(ValueError, lambda: binomial_mod(2, 1, -1))
    assert binomial_mod(51, 10, 10) == 0
    assert binomial_mod(10**3, 500, 3**6) == 567
    assert binomial_mod(10**18 - 1, 123456789, 4) == 0
    assert binomial_mod(10**18, 10**12, (10**5 + 3)**2) == 3744312326


def test_binomial_p_pow():
    n, binomials, binomial = 1000, [1], 1
    for i in range(1, n + 1):
        binomial *= n - i + 1
        binomial //= i
        binomials.append(binomial)

    # Test powers of two, which the algorithm treats slightly differently
    trials_2 = 100
    for _ in range(trials_2):
        m, power = randint(0, n), randint(1, 20)
        assert _binomial_mod_prime_power(n, m, 2, power) == binomials[m] % 2**power

    # Test against other prime powers
    primes = list(sieve.primerange(2*n))
    trials = 1000
    for _ in range(trials):
        m, prime, power = randint(0, n), choice(primes), randint(1, 10)
        assert _binomial_mod_prime_power(n, m, prime, power) == binomials[m] % prime**power


def test_deprecated_ntheory_symbolic_functions():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert mobius(3) == -1
    with warns_deprecated_sympy():
        assert legendre_symbol(2, 3) == -1
    with warns_deprecated_sympy():
        assert jacobi_symbol(2, 3) == -1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_fsspec.py ===
import io

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas import (
    DataFrame,
    date_range,
    read_csv,
    read_excel,
    read_feather,
    read_json,
    read_parquet,
    read_pickle,
    read_stata,
    read_table,
)
import pandas._testing as tm
from pandas.util import _test_decorators as td

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def fsspectest():
    pytest.importorskip("fsspec")
    from fsspec import register_implementation
    from fsspec.implementations.memory import MemoryFileSystem
    from fsspec.registry import _registry as registry

    class TestMemoryFS(MemoryFileSystem):
        protocol = "testmem"
        test = [None]

        def __init__(self, **kwargs) -> None:
            self.test[0] = kwargs.pop("test", None)
            super().__init__(**kwargs)

    register_implementation("testmem", TestMemoryFS, clobber=True)
    yield TestMemoryFS()
    registry.pop("testmem", None)
    TestMemoryFS.test[0] = None
    TestMemoryFS.store.clear()


@pytest.fixture
def df1():
    return DataFrame(
        {
            "int": [1, 3],
            "float": [2.0, np.nan],
            "str": ["t", "s"],
            "dt": date_range("2018-06-18", periods=2),
        }
    )


@pytest.fixture
def cleared_fs():
    fsspec = pytest.importorskip("fsspec")

    memfs = fsspec.filesystem("memory")
    yield memfs
    memfs.store.clear()


def test_read_csv(cleared_fs, df1):
    text = str(df1.to_csv(index=False)).encode()
    with cleared_fs.open("test/test.csv", "wb") as w:
        w.write(text)
    df2 = read_csv("memory://test/test.csv", parse_dates=["dt"])

    tm.assert_frame_equal(df1, df2)


def test_reasonable_error(monkeypatch, cleared_fs):
    from fsspec.registry import known_implementations

    with pytest.raises(ValueError, match="nosuchprotocol"):
        read_csv("nosuchprotocol://test/test.csv")
    err_msg = "test error message"
    monkeypatch.setitem(
        known_implementations,
        "couldexist",
        {"class": "unimportable.CouldExist", "err": err_msg},
    )
    with pytest.raises(ImportError, match=err_msg):
        read_csv("couldexist://test/test.csv")


def test_to_csv(cleared_fs, df1):
    df1.to_csv("memory://test/test.csv", index=True)

    df2 = read_csv("memory://test/test.csv", parse_dates=["dt"], index_col=0)

    tm.assert_frame_equal(df1, df2)


def test_to_excel(cleared_fs, df1):
    pytest.importorskip("openpyxl")
    ext = "xlsx"
    path = f"memory://test/test.{ext}"
    df1.to_excel(path, index=True)

    df2 = read_excel(path, parse_dates=["dt"], index_col=0)

    tm.assert_frame_equal(df1, df2)


@pytest.mark.parametrize("binary_mode", [False, True])
def test_to_csv_fsspec_object(cleared_fs, binary_mode, df1):
    fsspec = pytest.importorskip("fsspec")

    path = "memory://test/test.csv"
    mode = "wb" if binary_mode else "w"
    with fsspec.open(path, mode=mode).open() as fsspec_object:
        df1.to_csv(fsspec_object, index=True)
        assert not fsspec_object.closed

    mode = mode.replace("w", "r")
    with fsspec.open(path, mode=mode) as fsspec_object:
        df2 = read_csv(
            fsspec_object,
            parse_dates=["dt"],
            index_col=0,
        )
        assert not fsspec_object.closed

    tm.assert_frame_equal(df1, df2)


def test_csv_options(fsspectest):
    df = DataFrame({"a": [0]})
    df.to_csv(
        "testmem://test/test.csv", storage_options={"test": "csv_write"}, index=False
    )
    assert fsspectest.test[0] == "csv_write"
    read_csv("testmem://test/test.csv", storage_options={"test": "csv_read"})
    assert fsspectest.test[0] == "csv_read"


def test_read_table_options(fsspectest):
    # GH #39167
    df = DataFrame({"a": [0]})
    df.to_csv(
        "testmem://test/test.csv", storage_options={"test": "csv_write"}, index=False
    )
    assert fsspectest.test[0] == "csv_write"
    read_table("testmem://test/test.csv", storage_options={"test": "csv_read"})
    assert fsspectest.test[0] == "csv_read"


def test_excel_options(fsspectest):
    pytest.importorskip("openpyxl")
    extension = "xlsx"

    df = DataFrame({"a": [0]})

    path = f"testmem://test/test.{extension}"

    df.to_excel(path, storage_options={"test": "write"}, index=False)
    assert fsspectest.test[0] == "write"
    read_excel(path, storage_options={"test": "read"})
    assert fsspectest.test[0] == "read"


def test_to_parquet_new_file(cleared_fs, df1):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip("fastparquet")

    df1.to_parquet(
        "memory://test/test.csv", index=True, engine="fastparquet", compression=None
    )


def test_arrowparquet_options(fsspectest):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip("pyarrow")
    df = DataFrame({"a": [0]})
    df.to_parquet(
        "testmem://test/test.csv",
        engine="pyarrow",
        compression=None,
        storage_options={"test": "parquet_write"},
    )
    assert fsspectest.test[0] == "parquet_write"
    read_parquet(
        "testmem://test/test.csv",
        engine="pyarrow",
        storage_options={"test": "parquet_read"},
    )
    assert fsspectest.test[0] == "parquet_read"


@td.skip_array_manager_not_yet_implemented  # TODO(ArrayManager) fastparquet
def test_fastparquet_options(fsspectest):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip("fastparquet")

    df = DataFrame({"a": [0]})
    df.to_parquet(
        "testmem://test/test.csv",
        engine="fastparquet",
        compression=None,
        storage_options={"test": "parquet_write"},
    )
    assert fsspectest.test[0] == "parquet_write"
    read_parquet(
        "testmem://test/test.csv",
        engine="fastparquet",
        storage_options={"test": "parquet_read"},
    )
    assert fsspectest.test[0] == "parquet_read"


@pytest.mark.single_cpu
def test_from_s3_csv(s3_public_bucket_with_data, tips_file, s3so):
    pytest.importorskip("s3fs")
    tm.assert_equal(
        read_csv(
            f"s3://{s3_public_bucket_with_data.name}/tips.csv", storage_options=s3so
        ),
        read_csv(tips_file),
    )
    # the following are decompressed by pandas, not fsspec
    tm.assert_equal(
        read_csv(
            f"s3://{s3_public_bucket_with_data.name}/tips.csv.gz", storage_options=s3so
        ),
        read_csv(tips_file),
    )
    tm.assert_equal(
        read_csv(
            f"s3://{s3_public_bucket_with_data.name}/tips.csv.bz2", storage_options=s3so
        ),
        read_csv(tips_file),
    )


@pytest.mark.single_cpu
@pytest.mark.parametrize("protocol", ["s3", "s3a", "s3n"])
def test_s3_protocols(s3_public_bucket_with_data, tips_file, protocol, s3so):
    pytest.importorskip("s3fs")
    tm.assert_equal(
        read_csv(
            f"{protocol}://{s3_public_bucket_with_data.name}/tips.csv",
            storage_options=s3so,
        ),
        read_csv(tips_file),
    )


@pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string) fastparquet")
@pytest.mark.single_cpu
@td.skip_array_manager_not_yet_implemented  # TODO(ArrayManager) fastparquet
def test_s3_parquet(s3_public_bucket, s3so, df1):
    pytest.importorskip("fastparquet")
    pytest.importorskip("s3fs")

    fn = f"s3://{s3_public_bucket.name}/test.parquet"
    df1.to_parquet(
        fn, index=False, engine="fastparquet", compression=None, storage_options=s3so
    )
    df2 = read_parquet(fn, engine="fastparquet", storage_options=s3so)
    tm.assert_equal(df1, df2)


@td.skip_if_installed("fsspec")
def test_not_present_exception():
    msg = "Missing optional dependency 'fsspec'|fsspec library is required"
    with pytest.raises(ImportError, match=msg):
        read_csv("memory://test/test.csv")


def test_feather_options(fsspectest):
    pytest.importorskip("pyarrow")
    df = DataFrame({"a": [0]})
    df.to_feather("testmem://mockfile", storage_options={"test": "feather_write"})
    assert fsspectest.test[0] == "feather_write"
    out = read_feather("testmem://mockfile", storage_options={"test": "feather_read"})
    assert fsspectest.test[0] == "feather_read"
    tm.assert_frame_equal(df, out)


def test_pickle_options(fsspectest):
    df = DataFrame({"a": [0]})
    df.to_pickle("testmem://mockfile", storage_options={"test": "pickle_write"})
    assert fsspectest.test[0] == "pickle_write"
    out = read_pickle("testmem://mockfile", storage_options={"test": "pickle_read"})
    assert fsspectest.test[0] == "pickle_read"
    tm.assert_frame_equal(df, out)


def test_json_options(fsspectest, compression):
    df = DataFrame({"a": [0]})
    df.to_json(
        "testmem://mockfile",
        compression=compression,
        storage_options={"test": "json_write"},
    )
    assert fsspectest.test[0] == "json_write"
    out = read_json(
        "testmem://mockfile",
        compression=compression,
        storage_options={"test": "json_read"},
    )
    assert fsspectest.test[0] == "json_read"
    tm.assert_frame_equal(df, out)


def test_stata_options(fsspectest):
    df = DataFrame({"a": [0]})
    df.to_stata(
        "testmem://mockfile", storage_options={"test": "stata_write"}, write_index=False
    )
    assert fsspectest.test[0] == "stata_write"
    out = read_stata("testmem://mockfile", storage_options={"test": "stata_read"})
    assert fsspectest.test[0] == "stata_read"
    tm.assert_frame_equal(df, out.astype("int64"))


def test_markdown_options(fsspectest):
    pytest.importorskip("tabulate")
    df = DataFrame({"a": [0]})
    df.to_markdown("testmem://mockfile", storage_options={"test": "md_write"})
    assert fsspectest.test[0] == "md_write"
    assert fsspectest.cat("testmem://mockfile")


def test_non_fsspec_options():
    pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="storage_options"):
        read_csv("localfile", storage_options={"a": True})
    with pytest.raises(ValueError, match="storage_options"):
        # separate test for parquet, which has a different code path
        read_parquet("localfile", storage_options={"a": True})
    by = io.BytesIO()

    with pytest.raises(ValueError, match="storage_options"):
        read_csv(by, storage_options={"a": True})

    df = DataFrame({"a": [0]})
    with pytest.raises(ValueError, match="storage_options"):
        df.to_parquet("nonfsspecpath", storage_options={"a": True})

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\biomechanics\tests\test_activation.py ===
"""Tests for the ``sympy.physics.biomechanics.activation.py`` module."""

import pytest

from sympy import Symbol
from sympy.core.numbers import Float, Integer, Rational
from sympy.functions.elementary.hyperbolic import tanh
from sympy.matrices import Matrix
from sympy.matrices.dense import zeros
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.biomechanics import (
    ActivationBase,
    FirstOrderActivationDeGroote2016,
    ZerothOrderActivation,
)
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.simplify.simplify import simplify


class TestZerothOrderActivation:

    @staticmethod
    def test_class():
        assert issubclass(ZerothOrderActivation, ActivationBase)
        assert issubclass(ZerothOrderActivation, _NamedMixin)
        assert ZerothOrderActivation.__name__ == 'ZerothOrderActivation'

    @pytest.fixture(autouse=True)
    def _zeroth_order_activation_fixture(self):
        self.name = 'name'
        self.e = dynamicsymbols('e_name')
        self.instance = ZerothOrderActivation(self.name)

    def test_instance(self):
        instance = ZerothOrderActivation(self.name)
        assert isinstance(instance, ZerothOrderActivation)

    def test_with_defaults(self):
        instance = ZerothOrderActivation.with_defaults(self.name)
        assert isinstance(instance, ZerothOrderActivation)
        assert instance == ZerothOrderActivation(self.name)

    def test_name(self):
        assert hasattr(self.instance, 'name')
        assert self.instance.name == self.name

    def test_order(self):
        assert hasattr(self.instance, 'order')
        assert self.instance.order == 0

    def test_excitation_attribute(self):
        assert hasattr(self.instance, 'e')
        assert hasattr(self.instance, 'excitation')
        e_expected = dynamicsymbols('e_name')
        assert self.instance.e == e_expected
        assert self.instance.excitation == e_expected
        assert self.instance.e is self.instance.excitation

    def test_activation_attribute(self):
        assert hasattr(self.instance, 'a')
        assert hasattr(self.instance, 'activation')
        a_expected = dynamicsymbols('e_name')
        assert self.instance.a == a_expected
        assert self.instance.activation == a_expected
        assert self.instance.a is self.instance.activation is self.instance.e

    def test_state_vars_attribute(self):
        assert hasattr(self.instance, 'x')
        assert hasattr(self.instance, 'state_vars')
        assert self.instance.x == self.instance.state_vars
        x_expected = zeros(0, 1)
        assert self.instance.x == x_expected
        assert self.instance.state_vars == x_expected
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        assert self.instance.x.shape == (0, 1)
        assert self.instance.state_vars.shape == (0, 1)

    def test_input_vars_attribute(self):
        assert hasattr(self.instance, 'r')
        assert hasattr(self.instance, 'input_vars')
        assert self.instance.r == self.instance.input_vars
        r_expected = Matrix([self.e])
        assert self.instance.r == r_expected
        assert self.instance.input_vars == r_expected
        assert isinstance(self.instance.r, Matrix)
        assert isinstance(self.instance.input_vars, Matrix)
        assert self.instance.r.shape == (1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    def test_constants_attribute(self):
        assert hasattr(self.instance, 'p')
        assert hasattr(self.instance, 'constants')
        assert self.instance.p == self.instance.constants
        p_expected = zeros(0, 1)
        assert self.instance.p == p_expected
        assert self.instance.constants == p_expected
        assert isinstance(self.instance.p, Matrix)
        assert isinstance(self.instance.constants, Matrix)
        assert self.instance.p.shape == (0, 1)
        assert self.instance.constants.shape == (0, 1)

    def test_M_attribute(self):
        assert hasattr(self.instance, 'M')
        M_expected = Matrix([])
        assert self.instance.M == M_expected
        assert isinstance(self.instance.M, Matrix)
        assert self.instance.M.shape == (0, 0)

    def test_F(self):
        assert hasattr(self.instance, 'F')
        F_expected = zeros(0, 1)
        assert self.instance.F == F_expected
        assert isinstance(self.instance.F, Matrix)
        assert self.instance.F.shape == (0, 1)

    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        rhs_expected = zeros(0, 1)
        rhs = self.instance.rhs()
        assert rhs == rhs_expected
        assert isinstance(rhs, Matrix)
        assert rhs.shape == (0, 1)

    def test_repr(self):
        expected = 'ZerothOrderActivation(\'name\')'
        assert repr(self.instance) == expected


class TestFirstOrderActivationDeGroote2016:

    @staticmethod
    def test_class():
        assert issubclass(FirstOrderActivationDeGroote2016, ActivationBase)
        assert issubclass(FirstOrderActivationDeGroote2016, _NamedMixin)
        assert FirstOrderActivationDeGroote2016.__name__ == 'FirstOrderActivationDeGroote2016'

    @pytest.fixture(autouse=True)
    def _first_order_activation_de_groote_2016_fixture(self):
        self.name = 'name'
        self.e = dynamicsymbols('e_name')
        self.a = dynamicsymbols('a_name')
        self.tau_a = Symbol('tau_a')
        self.tau_d = Symbol('tau_d')
        self.b = Symbol('b')
        self.instance = FirstOrderActivationDeGroote2016(
            self.name,
            self.tau_a,
            self.tau_d,
            self.b,
        )

    def test_instance(self):
        instance = FirstOrderActivationDeGroote2016(self.name)
        assert isinstance(instance, FirstOrderActivationDeGroote2016)

    def test_with_defaults(self):
        instance = FirstOrderActivationDeGroote2016.with_defaults(self.name)
        assert isinstance(instance, FirstOrderActivationDeGroote2016)
        assert instance.tau_a == Float('0.015')
        assert instance.activation_time_constant == Float('0.015')
        assert instance.tau_d == Float('0.060')
        assert instance.deactivation_time_constant == Float('0.060')
        assert instance.b == Float('10.0')
        assert instance.smoothing_rate == Float('10.0')

    def test_name(self):
        assert hasattr(self.instance, 'name')
        assert self.instance.name == self.name

    def test_order(self):
        assert hasattr(self.instance, 'order')
        assert self.instance.order == 1

    def test_excitation(self):
        assert hasattr(self.instance, 'e')
        assert hasattr(self.instance, 'excitation')
        e_expected = dynamicsymbols('e_name')
        assert self.instance.e == e_expected
        assert self.instance.excitation == e_expected
        assert self.instance.e is self.instance.excitation

    def test_excitation_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.e = None
        with pytest.raises(AttributeError):
            self.instance.excitation = None

    def test_activation(self):
        assert hasattr(self.instance, 'a')
        assert hasattr(self.instance, 'activation')
        a_expected = dynamicsymbols('a_name')
        assert self.instance.a == a_expected
        assert self.instance.activation == a_expected

    def test_activation_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.a = None
        with pytest.raises(AttributeError):
            self.instance.activation = None

    @pytest.mark.parametrize(
        'tau_a, expected',
        [
            (None, Symbol('tau_a_name')),
            (Symbol('tau_a'), Symbol('tau_a')),
            (Float('0.015'), Float('0.015')),
        ]
    )
    def test_activation_time_constant(self, tau_a, expected):
        instance = FirstOrderActivationDeGroote2016(
            'name', activation_time_constant=tau_a,
        )
        assert instance.tau_a == expected
        assert instance.activation_time_constant == expected
        assert instance.tau_a is instance.activation_time_constant

    def test_activation_time_constant_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.tau_a = None
        with pytest.raises(AttributeError):
            self.instance.activation_time_constant = None

    @pytest.mark.parametrize(
        'tau_d, expected',
        [
            (None, Symbol('tau_d_name')),
            (Symbol('tau_d'), Symbol('tau_d')),
            (Float('0.060'), Float('0.060')),
        ]
    )
    def test_deactivation_time_constant(self, tau_d, expected):
        instance = FirstOrderActivationDeGroote2016(
            'name', deactivation_time_constant=tau_d,
        )
        assert instance.tau_d == expected
        assert instance.deactivation_time_constant == expected
        assert instance.tau_d is instance.deactivation_time_constant

    def test_deactivation_time_constant_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.tau_d = None
        with pytest.raises(AttributeError):
            self.instance.deactivation_time_constant = None

    @pytest.mark.parametrize(
        'b, expected',
        [
            (None, Symbol('b_name')),
            (Symbol('b'), Symbol('b')),
            (Integer('10'), Integer('10')),
        ]
    )
    def test_smoothing_rate(self, b, expected):
        instance = FirstOrderActivationDeGroote2016(
            'name', smoothing_rate=b,
        )
        assert instance.b == expected
        assert instance.smoothing_rate == expected
        assert instance.b is instance.smoothing_rate

    def test_smoothing_rate_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.b = None
        with pytest.raises(AttributeError):
            self.instance.smoothing_rate = None

    def test_state_vars(self):
        assert hasattr(self.instance, 'x')
        assert hasattr(self.instance, 'state_vars')
        assert self.instance.x == self.instance.state_vars
        x_expected = Matrix([self.a])
        assert self.instance.x == x_expected
        assert self.instance.state_vars == x_expected
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        assert self.instance.x.shape == (1, 1)
        assert self.instance.state_vars.shape == (1, 1)

    def test_input_vars(self):
        assert hasattr(self.instance, 'r')
        assert hasattr(self.instance, 'input_vars')
        assert self.instance.r == self.instance.input_vars
        r_expected = Matrix([self.e])
        assert self.instance.r == r_expected
        assert self.instance.input_vars == r_expected
        assert isinstance(self.instance.r, Matrix)
        assert isinstance(self.instance.input_vars, Matrix)
        assert self.instance.r.shape == (1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    def test_constants(self):
        assert hasattr(self.instance, 'p')
        assert hasattr(self.instance, 'constants')
        assert self.instance.p == self.instance.constants
        p_expected = Matrix([self.tau_a, self.tau_d, self.b])
        assert self.instance.p == p_expected
        assert self.instance.constants == p_expected
        assert isinstance(self.instance.p, Matrix)
        assert isinstance(self.instance.constants, Matrix)
        assert self.instance.p.shape == (3, 1)
        assert self.instance.constants.shape == (3, 1)

    def test_M(self):
        assert hasattr(self.instance, 'M')
        M_expected = Matrix([1])
        assert self.instance.M == M_expected
        assert isinstance(self.instance.M, Matrix)
        assert self.instance.M.shape == (1, 1)

    def test_F(self):
        assert hasattr(self.instance, 'F')
        da_expr = (
            ((1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a))))
            *(self.e - self.a)
        )
        F_expected = Matrix([da_expr])
        assert self.instance.F == F_expected
        assert isinstance(self.instance.F, Matrix)
        assert self.instance.F.shape == (1, 1)

    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        da_expr = (
            ((1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a))))
            *(self.e - self.a)
        )
        rhs_expected = Matrix([da_expr])
        rhs = self.instance.rhs()
        assert rhs == rhs_expected
        assert isinstance(rhs, Matrix)
        assert rhs.shape == (1, 1)
        assert simplify(self.instance.M.solve(self.instance.F) - rhs) == zeros(1)

    def test_repr(self):
        expected = (
            'FirstOrderActivationDeGroote2016(\'name\', '
            'activation_time_constant=tau_a, '
            'deactivation_time_constant=tau_d, '
            'smoothing_rate=b)'
        )
        assert repr(self.instance) == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\timedeltas\test_indexing.py ===
from datetime import datetime
import re

import numpy as np
import pytest

from pandas import (
    Index,
    NaT,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    notna,
    offsets,
    timedelta_range,
    to_timedelta,
)
import pandas._testing as tm


class TestGetItem:
    def test_getitem_slice_keeps_name(self):
        # GH#4226
        tdi = timedelta_range("1d", "5d", freq="h", name="timebucket")
        assert tdi[1:].name == tdi.name

    def test_getitem(self):
        idx1 = timedelta_range("1 day", "31 day", freq="D", name="idx")

        for idx in [idx1]:
            result = idx[0]
            assert result == Timedelta("1 day")

            result = idx[0:5]
            expected = timedelta_range("1 day", "5 day", freq="D", name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            result = idx[0:10:2]
            expected = timedelta_range("1 day", "9 day", freq="2D", name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            result = idx[-20:-5:3]
            expected = timedelta_range("12 day", "24 day", freq="3D", name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            result = idx[4::-1]
            expected = TimedeltaIndex(
                ["5 day", "4 day", "3 day", "2 day", "1 day"], freq="-1D", name="idx"
            )
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

    @pytest.mark.parametrize(
        "key",
        [
            Timestamp("1970-01-01"),
            Timestamp("1970-01-02"),
            datetime(1970, 1, 1),
            Timestamp("1970-01-03").to_datetime64(),
            # non-matching NA values
            np.datetime64("NaT"),
        ],
    )
    def test_timestamp_invalid_key(self, key):
        # GH#20464
        tdi = timedelta_range(0, periods=10)
        with pytest.raises(KeyError, match=re.escape(repr(key))):
            tdi.get_loc(key)


class TestGetLoc:
    def test_get_loc_key_unit_mismatch(self):
        idx = to_timedelta(["0 days", "1 days", "2 days"])
        key = idx[1].as_unit("ms")
        loc = idx.get_loc(key)
        assert loc == 1

    def test_get_loc_key_unit_mismatch_not_castable(self):
        tdi = to_timedelta(["0 days", "1 days", "2 days"]).astype("m8[s]")
        assert tdi.dtype == "m8[s]"
        key = tdi[0].as_unit("ns") + Timedelta(1)

        with pytest.raises(KeyError, match=r"Timedelta\('0 days 00:00:00.000000001'\)"):
            tdi.get_loc(key)

        assert key not in tdi

    def test_get_loc(self):
        idx = to_timedelta(["0 days", "1 days", "2 days"])

        # GH 16909
        assert idx.get_loc(idx[1].to_timedelta64()) == 1

        # GH 16896
        assert idx.get_loc("0 days") == 0

    def test_get_loc_nat(self):
        tidx = TimedeltaIndex(["1 days 01:00:00", "NaT", "2 days 01:00:00"])

        assert tidx.get_loc(NaT) == 1
        assert tidx.get_loc(None) == 1
        assert tidx.get_loc(float("nan")) == 1
        assert tidx.get_loc(np.nan) == 1


class TestGetIndexer:
    def test_get_indexer(self):
        idx = to_timedelta(["0 days", "1 days", "2 days"])
        tm.assert_numpy_array_equal(
            idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp)
        )

        target = to_timedelta(["-1 hour", "12 hours", "1 day 1 hour"])
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2], dtype=np.intp)
        )
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 1, 1], dtype=np.intp)
        )

        res = idx.get_indexer(target, "nearest", tolerance=Timedelta("1 hour"))
        tm.assert_numpy_array_equal(res, np.array([0, -1, 1], dtype=np.intp))


class TestWhere:
    def test_where_doesnt_retain_freq(self):
        tdi = timedelta_range("1 day", periods=3, freq="D", name="idx")
        cond = [True, True, False]
        expected = TimedeltaIndex([tdi[0], tdi[1], tdi[0]], freq=None, name="idx")

        result = tdi.where(cond, tdi[::-1])
        tm.assert_index_equal(result, expected)

    def test_where_invalid_dtypes(self, fixed_now_ts):
        tdi = timedelta_range("1 day", periods=3, freq="D", name="idx")

        tail = tdi[2:].tolist()
        i2 = Index([NaT, NaT] + tail)
        mask = notna(i2)

        expected = Index([NaT._value, NaT._value] + tail, dtype=object, name="idx")
        assert isinstance(expected[0], int)
        result = tdi.where(mask, i2.asi8)
        tm.assert_index_equal(result, expected)

        ts = i2 + fixed_now_ts
        expected = Index([ts[0], ts[1]] + tail, dtype=object, name="idx")
        result = tdi.where(mask, ts)
        tm.assert_index_equal(result, expected)

        per = (i2 + fixed_now_ts).to_period("D")
        expected = Index([per[0], per[1]] + tail, dtype=object, name="idx")
        result = tdi.where(mask, per)
        tm.assert_index_equal(result, expected)

        ts = fixed_now_ts
        expected = Index([ts, ts] + tail, dtype=object, name="idx")
        result = tdi.where(mask, ts)
        tm.assert_index_equal(result, expected)

    def test_where_mismatched_nat(self):
        tdi = timedelta_range("1 day", periods=3, freq="D", name="idx")
        cond = np.array([True, False, False])

        dtnat = np.datetime64("NaT", "ns")
        expected = Index([tdi[0], dtnat, dtnat], dtype=object, name="idx")
        assert expected[2] is dtnat
        result = tdi.where(cond, dtnat)
        tm.assert_index_equal(result, expected)


class TestTake:
    def test_take(self):
        # GH 10295
        idx1 = timedelta_range("1 day", "31 day", freq="D", name="idx")

        for idx in [idx1]:
            result = idx.take([0])
            assert result == Timedelta("1 day")

            result = idx.take([-1])
            assert result == Timedelta("31 day")

            result = idx.take([0, 1, 2])
            expected = timedelta_range("1 day", "3 day", freq="D", name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            result = idx.take([0, 2, 4])
            expected = timedelta_range("1 day", "5 day", freq="2D", name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            result = idx.take([7, 4, 1])
            expected = timedelta_range("8 day", "2 day", freq="-3D", name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            result = idx.take([3, 2, 5])
            expected = TimedeltaIndex(["4 day", "3 day", "6 day"], name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq is None

            result = idx.take([-3, 2, 5])
            expected = TimedeltaIndex(["29 day", "3 day", "6 day"], name="idx")
            tm.assert_index_equal(result, expected)
            assert result.freq is None

    def test_take_invalid_kwargs(self):
        idx = timedelta_range("1 day", "31 day", freq="D", name="idx")
        indices = [1, 6, 5, 9, 10, 13, 15, 3]

        msg = r"take\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            idx.take(indices, foo=2)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, out=indices)

        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, mode="clip")

    def test_take_equiv_getitem(self):
        tds = ["1day 02:00:00", "1 day 04:00:00", "1 day 10:00:00"]
        idx = timedelta_range(start="1d", end="2d", freq="h", name="idx")
        expected = TimedeltaIndex(tds, freq=None, name="idx")

        taken1 = idx.take([2, 4, 10])
        taken2 = idx[[2, 4, 10]]

        for taken in [taken1, taken2]:
            tm.assert_index_equal(taken, expected)
            assert isinstance(taken, TimedeltaIndex)
            assert taken.freq is None
            assert taken.name == expected.name

    def test_take_fill_value(self):
        # GH 12631
        idx = TimedeltaIndex(["1 days", "2 days", "3 days"], name="xxx")
        result = idx.take(np.array([1, 0, -1]))
        expected = TimedeltaIndex(["2 days", "1 days", "3 days"], name="xxx")
        tm.assert_index_equal(result, expected)

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = TimedeltaIndex(["2 days", "1 days", "NaT"], name="xxx")
        tm.assert_index_equal(result, expected)

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = TimedeltaIndex(["2 days", "1 days", "3 days"], name="xxx")
        tm.assert_index_equal(result, expected)

        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))


class TestMaybeCastSliceBound:
    @pytest.fixture(params=["increasing", "decreasing", None])
    def monotonic(self, request):
        return request.param

    @pytest.fixture
    def tdi(self, monotonic):
        tdi = timedelta_range("1 Day", periods=10)
        if monotonic == "decreasing":
            tdi = tdi[::-1]
        elif monotonic is None:
            taker = np.arange(10, dtype=np.intp)
            np.random.default_rng(2).shuffle(taker)
            tdi = tdi.take(taker)
        return tdi

    def test_maybe_cast_slice_bound_invalid_str(self, tdi):
        # test the low-level _maybe_cast_slice_bound and that we get the
        #  expected exception+message all the way up the stack
        msg = (
            "cannot do slice indexing on TimedeltaIndex with these "
            r"indexers \[foo\] of type str"
        )
        with pytest.raises(TypeError, match=msg):
            tdi._maybe_cast_slice_bound("foo", side="left")
        with pytest.raises(TypeError, match=msg):
            tdi.get_slice_bound("foo", side="left")
        with pytest.raises(TypeError, match=msg):
            tdi.slice_locs("foo", None, None)

    def test_slice_invalid_str_with_timedeltaindex(
        self, tdi, frame_or_series, indexer_sl
    ):
        obj = frame_or_series(range(10), index=tdi)

        msg = (
            "cannot do slice indexing on TimedeltaIndex with these "
            r"indexers \[foo\] of type str"
        )
        with pytest.raises(TypeError, match=msg):
            indexer_sl(obj)["foo":]
        with pytest.raises(TypeError, match=msg):
            indexer_sl(obj)["foo":-1]
        with pytest.raises(TypeError, match=msg):
            indexer_sl(obj)[:"foo"]
        with pytest.raises(TypeError, match=msg):
            indexer_sl(obj)[tdi[0] : "foo"]


class TestContains:
    def test_contains_nonunique(self):
        # GH#9512
        for vals in (
            [0, 1, 0],
            [0, 0, -1],
            [0, -1, -1],
            ["00:01:00", "00:01:00", "00:02:00"],
            ["00:01:00", "00:01:00", "00:00:01"],
        ):
            idx = TimedeltaIndex(vals)
            assert idx[0] in idx

    def test_contains(self):
        # Checking for any NaT-like objects
        # GH#13603
        td = to_timedelta(range(5), unit="d") + offsets.Hour(1)
        for v in [NaT, None, float("nan"), np.nan]:
            assert v not in td

        td = to_timedelta([NaT])
        for v in [NaT, None, float("nan"), np.nan]:
            assert v in td

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_subresultants_qq_zz.py ===
from sympy.core.symbol import Symbol
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
from sympy.matrices import Matrix
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
    subresultants_sylv,   modified_subresultants_sylv,
    subresultants_bezout, modified_subresultants_bezout,
    backward_eye,
    sturm_pg, sturm_q, sturm_amv, euclid_pg, euclid_q,
    euclid_amv, modified_subresultants_pg, subresultants_pg,
    subresultants_amv_q, quo_z, rem_z, subresultants_amv,
    modified_subresultants_amv, subresultants_rem,
    subresultants_vv, subresultants_vv_2)


def test_sylvester():
    x = Symbol('x')

    assert sylvester(x**3 -7, 0, x) == sylvester(x**3 -7, 0, x, 1) == Matrix([[0]])
    assert sylvester(0, x**3 -7, x) == sylvester(0, x**3 -7, x, 1) == Matrix([[0]])
    assert sylvester(x**3 -7, 0, x, 2) == Matrix([[0]])
    assert sylvester(0, x**3 -7, x, 2) == Matrix([[0]])

    assert sylvester(x**3 -7, 7, x).det() == sylvester(x**3 -7, 7, x, 1).det() == 343
    assert sylvester(7, x**3 -7, x).det() == sylvester(7, x**3 -7, x, 1).det() == 343
    assert sylvester(x**3 -7, 7, x, 2).det() == -343
    assert sylvester(7, x**3 -7, x, 2).det() == 343

    assert sylvester(3, 7, x).det() == sylvester(3, 7, x, 1).det() == sylvester(3, 7, x, 2).det() == 1

    assert sylvester(3, 0, x).det() == sylvester(3, 0, x, 1).det() == sylvester(3, 0, x, 2).det() == 1

    assert sylvester(x - 3, x - 8, x) == sylvester(x - 3, x - 8, x, 1) == sylvester(x - 3, x - 8, x, 2) == Matrix([[1, -3], [1, -8]])

    assert sylvester(x**3 - 7*x + 7, 3*x**2 - 7, x) == sylvester(x**3 - 7*x + 7, 3*x**2 - 7, x, 1) == Matrix([[1, 0, -7,  7,  0], [0, 1,  0, -7,  7], [3, 0, -7,  0,  0], [0, 3,  0, -7,  0], [0, 0,  3,  0, -7]])

    assert sylvester(x**3 - 7*x + 7, 3*x**2 - 7, x, 2) == Matrix([
[1, 0, -7,  7,  0,  0], [0, 3,  0, -7,  0,  0], [0, 1,  0, -7,  7,  0], [0, 0,  3,  0, -7,  0], [0, 0,  1,  0, -7,  7], [0, 0,  0,  3,  0, -7]])

def test_subresultants_sylv():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert subresultants_sylv(p, q, x) == subresultants(p, q, x)
    assert subresultants_sylv(p, q, x)[-1] == res(p, q, x)
    assert subresultants_sylv(p, q, x) != euclid_amv(p, q, x)
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert subresultants_sylv(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert subresultants_sylv(p, q, x) == euclid_amv(p, q, x)

def test_modified_subresultants_sylv():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert modified_subresultants_sylv(p, q, x) == [i*j for i, j in zip(amv_factors, subresultants_amv(p, q, x))]
    assert modified_subresultants_sylv(p, q, x)[-1] != res_q(p + x**8, q, x)
    assert modified_subresultants_sylv(p, q, x) != sturm_amv(p, q, x)

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert modified_subresultants_sylv(p, q, x) == sturm_amv(p, q, x)
    assert modified_subresultants_sylv(-p, q, x) != sturm_amv(-p, q, x)

def test_res():
    x = Symbol('x')

    assert res(3, 5, x) == 1

def test_res_q():
    x = Symbol('x')

    assert res_q(3, 5, x) == 1

def test_res_z():
    x = Symbol('x')

    assert res_z(3, 5, x) == 1
    assert res(3, 5, x) == res_q(3, 5, x) == res_z(3, 5, x)

def test_bezout():
    x = Symbol('x')

    p = -2*x**5+7*x**3+9*x**2-3*x+1
    q = -10*x**4+21*x**2+18*x-3
    assert bezout(p, q, x, 'bz').det() == sylvester(p, q, x, 2).det()
    assert bezout(p, q, x, 'bz').det() != sylvester(p, q, x, 1).det()
    assert bezout(p, q, x, 'prs') == backward_eye(5) * bezout(p, q, x, 'bz') * backward_eye(5)

def test_subresultants_bezout():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert subresultants_bezout(p, q, x) == subresultants(p, q, x)
    assert subresultants_bezout(p, q, x)[-1] == sylvester(p, q, x).det()
    assert subresultants_bezout(p, q, x) != euclid_amv(p, q, x)
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert subresultants_bezout(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert subresultants_bezout(p, q, x) == euclid_amv(p, q, x)

def test_modified_subresultants_bezout():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert modified_subresultants_bezout(p, q, x) == [i*j for i, j in zip(amv_factors, subresultants_amv(p, q, x))]
    assert modified_subresultants_bezout(p, q, x)[-1] != sylvester(p + x**8, q, x).det()
    assert modified_subresultants_bezout(p, q, x) != sturm_amv(p, q, x)

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert modified_subresultants_bezout(p, q, x) == sturm_amv(p, q, x)
    assert modified_subresultants_bezout(-p, q, x) != sturm_amv(-p, q, x)

def test_sturm_pg():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert sturm_pg(p, q, x)[-1] != sylvester(p, q, x, 2).det()
    sam_factors = [1, 1, -1, -1, 1, 1]
    assert sturm_pg(p, q, x) == [i*j for i,j in zip(sam_factors, euclid_pg(p, q, x))]

    p = -9*x**5 - 5*x**3 - 9
    q = -45*x**4 - 15*x**2
    assert sturm_pg(p, q, x, 1)[-1] == sylvester(p, q, x, 1).det()
    assert sturm_pg(p, q, x)[-1] != sylvester(p, q, x, 2).det()
    assert sturm_pg(-p, q, x)[-1] == sylvester(-p, q, x, 2).det()
    assert sturm_pg(-p, q, x) == modified_subresultants_pg(-p, q, x)

def test_sturm_q():
    x = Symbol('x')

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert sturm_q(p, q, x) == sturm(p)
    assert sturm_q(-p, -q, x) != sturm(-p)


def test_sturm_amv():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert sturm_amv(p, q, x)[-1] != sylvester(p, q, x, 2).det()
    sam_factors = [1, 1, -1, -1, 1, 1]
    assert sturm_amv(p, q, x) == [i*j for i,j in zip(sam_factors, euclid_amv(p, q, x))]

    p = -9*x**5 - 5*x**3 - 9
    q = -45*x**4 - 15*x**2
    assert sturm_amv(p, q, x, 1)[-1] == sylvester(p, q, x, 1).det()
    assert sturm_amv(p, q, x)[-1] != sylvester(p, q, x, 2).det()
    assert sturm_amv(-p, q, x)[-1] == sylvester(-p, q, x, 2).det()
    assert sturm_pg(-p, q, x) == modified_subresultants_pg(-p, q, x)


def test_euclid_pg():
    x = Symbol('x')

    p = x**6+x**5-x**4-x**3+x**2-x+1
    q = 6*x**5+5*x**4-4*x**3-3*x**2+2*x-1
    assert euclid_pg(p, q, x)[-1] == sylvester(p, q, x).det()
    assert euclid_pg(p, q, x) == subresultants_pg(p, q, x)

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert euclid_pg(p, q, x)[-1] != sylvester(p, q, x, 2).det()
    sam_factors = [1, 1, -1, -1, 1, 1]
    assert euclid_pg(p, q, x) == [i*j for i,j in zip(sam_factors, sturm_pg(p, q, x))]


def test_euclid_q():
    x = Symbol('x')

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert euclid_q(p, q, x)[-1] == -sturm(p)[-1]


def test_euclid_amv():
    x = Symbol('x')

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert euclid_amv(p, q, x)[-1] == sylvester(p, q, x).det()
    assert euclid_amv(p, q, x) == subresultants_amv(p, q, x)

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert euclid_amv(p, q, x)[-1] != sylvester(p, q, x, 2).det()
    sam_factors = [1, 1, -1, -1, 1, 1]
    assert euclid_amv(p, q, x) == [i*j for i,j in zip(sam_factors, sturm_amv(p, q, x))]


def test_modified_subresultants_pg():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert modified_subresultants_pg(p, q, x) == [i*j for i, j in zip(amv_factors, subresultants_pg(p, q, x))]
    assert modified_subresultants_pg(p, q, x)[-1] != sylvester(p + x**8, q, x).det()
    assert modified_subresultants_pg(p, q, x) != sturm_pg(p, q, x)

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert modified_subresultants_pg(p, q, x) == sturm_pg(p, q, x)
    assert modified_subresultants_pg(-p, q, x) != sturm_pg(-p, q, x)


def test_subresultants_pg():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert subresultants_pg(p, q, x) == subresultants(p, q, x)
    assert subresultants_pg(p, q, x)[-1] == sylvester(p, q, x).det()
    assert subresultants_pg(p, q, x) != euclid_pg(p, q, x)
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert subresultants_pg(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert subresultants_pg(p, q, x) == euclid_pg(p, q, x)


def test_subresultants_amv_q():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert subresultants_amv_q(p, q, x) == subresultants(p, q, x)
    assert subresultants_amv_q(p, q, x)[-1] == sylvester(p, q, x).det()
    assert subresultants_amv_q(p, q, x) != euclid_amv(p, q, x)
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert subresultants_amv_q(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert subresultants_amv(p, q, x) == euclid_amv(p, q, x)


def test_rem_z():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert rem_z(p, -q, x) != prem(p, -q, x)

def test_quo_z():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert quo_z(p, -q, x) != pquo(p, -q, x)

    y = Symbol('y')
    q = 3*x**6 + 5*y**4 - 4*x**2 - 9*x + 21
    assert quo_z(p, -q, x) == pquo(p, -q, x)

def test_subresultants_amv():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert subresultants_amv(p, q, x) == subresultants(p, q, x)
    assert subresultants_amv(p, q, x)[-1] == sylvester(p, q, x).det()
    assert subresultants_amv(p, q, x) != euclid_amv(p, q, x)
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert subresultants_amv(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert subresultants_amv(p, q, x) == euclid_amv(p, q, x)


def test_modified_subresultants_amv():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert modified_subresultants_amv(p, q, x) == [i*j for i, j in zip(amv_factors, subresultants_amv(p, q, x))]
    assert modified_subresultants_amv(p, q, x)[-1] != sylvester(p + x**8, q, x).det()
    assert modified_subresultants_amv(p, q, x) != sturm_amv(p, q, x)

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert modified_subresultants_amv(p, q, x) == sturm_amv(p, q, x)
    assert modified_subresultants_amv(-p, q, x) != sturm_amv(-p, q, x)


def test_subresultants_rem():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert subresultants_rem(p, q, x) == subresultants(p, q, x)
    assert subresultants_rem(p, q, x)[-1] == sylvester(p, q, x).det()
    assert subresultants_rem(p, q, x) != euclid_amv(p, q, x)
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert subresultants_rem(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert subresultants_rem(p, q, x) == euclid_amv(p, q, x)


def test_subresultants_vv():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert subresultants_vv(p, q, x) == subresultants(p, q, x)
    assert subresultants_vv(p, q, x)[-1] == sylvester(p, q, x).det()
    assert subresultants_vv(p, q, x) != euclid_amv(p, q, x)
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert subresultants_vv(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert subresultants_vv(p, q, x) == euclid_amv(p, q, x)


def test_subresultants_vv_2():
    x = Symbol('x')

    p = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    q = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    assert subresultants_vv_2(p, q, x) == subresultants(p, q, x)
    assert subresultants_vv_2(p, q, x)[-1] == sylvester(p, q, x).det()
    assert subresultants_vv_2(p, q, x) != euclid_amv(p, q, x)
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert subresultants_vv_2(p, q, x) == [i*j for i, j in zip(amv_factors, modified_subresultants_amv(p, q, x))]

    p = x**3 - 7*x + 7
    q = 3*x**2 - 7
    assert subresultants_vv_2(p, q, x) == euclid_amv(p, q, x)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arrays\integer\test_arithmetic.py ===
import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray

# Basic test for the arithmetic array ops
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opname, exp",
    [("add", [1, 3, None, None, 9]), ("mul", [0, 2, None, None, 20])],
    ids=["add", "mul"],
)
def test_add_mul(dtype, opname, exp):
    a = pd.array([0, 1, None, 3, 4], dtype=dtype)
    b = pd.array([1, 2, 3, None, 5], dtype=dtype)

    # array / array
    expected = pd.array(exp, dtype=dtype)

    op = getattr(operator, opname)
    result = op(a, b)
    tm.assert_extension_array_equal(result, expected)

    op = getattr(ops, "r" + opname)
    result = op(a, b)
    tm.assert_extension_array_equal(result, expected)


def test_sub(dtype):
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    result = a - b
    expected = pd.array([1, 1, None, None, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_div(dtype):
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    result = a / b
    expected = pd.array([np.inf, 2, None, None, 1.25], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("zero, negative", [(0, False), (0.0, False), (-0.0, True)])
def test_divide_by_zero(zero, negative):
    # https://github.com/pandas-dev/pandas/issues/27398, GH#22793
    a = pd.array([0, 1, -1, None], dtype="Int64")
    result = a / zero
    expected = FloatingArray(
        np.array([np.nan, np.inf, -np.inf, 1], dtype="float64"),
        np.array([False, False, False, True]),
    )
    if negative:
        expected *= -1
    tm.assert_extension_array_equal(result, expected)


def test_floordiv(dtype):
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    result = a // b
    # Series op sets 1//0 to np.inf, which IntegerArray does not do (yet)
    expected = pd.array([0, 2, None, None, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_floordiv_by_int_zero_no_mask(any_int_ea_dtype):
    # GH 48223: Aligns with non-masked floordiv
    # but differs from numpy
    # https://github.com/pandas-dev/pandas/issues/30188#issuecomment-564452740
    ser = pd.Series([0, 1], dtype=any_int_ea_dtype)
    result = 1 // ser
    expected = pd.Series([np.inf, 1.0], dtype="Float64")
    tm.assert_series_equal(result, expected)

    ser_non_nullable = ser.astype(ser.dtype.numpy_dtype)
    result = 1 // ser_non_nullable
    expected = expected.astype(np.float64)
    tm.assert_series_equal(result, expected)


def test_mod(dtype):
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    result = a % b
    expected = pd.array([0, 0, None, None, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_pow_scalar():
    a = pd.array([-1, 0, 1, None, 2], dtype="Int64")
    result = a**0
    expected = pd.array([1, 1, 1, 1, 1], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = a**1
    expected = pd.array([-1, 0, 1, None, 2], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = a**pd.NA
    expected = pd.array([None, None, 1, None, None], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = a**np.nan
    expected = FloatingArray(
        np.array([np.nan, np.nan, 1, np.nan, np.nan], dtype="float64"),
        np.array([False, False, False, True, False]),
    )
    tm.assert_extension_array_equal(result, expected)

    # reversed
    a = a[1:]  # Can't raise integers to negative powers.

    result = 0**a
    expected = pd.array([1, 0, None, 0], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = 1**a
    expected = pd.array([1, 1, 1, 1], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = pd.NA**a
    expected = pd.array([1, None, None, None], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)

    result = np.nan**a
    expected = FloatingArray(
        np.array([1, np.nan, np.nan, np.nan], dtype="float64"),
        np.array([False, False, True, False]),
    )
    tm.assert_extension_array_equal(result, expected)


def test_pow_array():
    a = pd.array([0, 0, 0, 1, 1, 1, None, None, None])
    b = pd.array([0, 1, None, 0, 1, None, 0, 1, None])
    result = a**b
    expected = pd.array([1, 0, None, 1, 1, 1, 1, None, None])
    tm.assert_extension_array_equal(result, expected)


def test_rpow_one_to_na():
    # https://github.com/pandas-dev/pandas/issues/22022
    # https://github.com/pandas-dev/pandas/issues/29997
    arr = pd.array([np.nan, np.nan], dtype="Int64")
    result = np.array([1.0, 2.0]) ** arr
    expected = pd.array([1.0, np.nan], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("other", [0, 0.5])
def test_numpy_zero_dim_ndarray(other):
    arr = pd.array([1, None, 2])
    result = arr + np.array(other)
    expected = arr + other
    tm.assert_equal(result, expected)


# Test generic characteristics / errors
# -----------------------------------------------------------------------------


def test_error_invalid_values(data, all_arithmetic_operators):
    op = all_arithmetic_operators
    s = pd.Series(data)
    ops = getattr(s, op)

    # invalid scalars
    with tm.external_error_raised(TypeError):
        ops("foo")
    with tm.external_error_raised(TypeError):
        ops(pd.Timestamp("20180101"))

    # invalid array-likes
    str_ser = pd.Series("foo", index=s.index)
    # with pytest.raises(TypeError, match=msg):
    if all_arithmetic_operators in [
        "__mul__",
        "__rmul__",
    ]:  # (data[~data.isna()] >= 0).all():
        res = ops(str_ser)
        expected = pd.Series(["foo" * x for x in data], index=s.index)
        expected = expected.fillna(np.nan)
        # TODO: doing this fillna to keep tests passing as we make
        #  assert_almost_equal stricter, but the expected with pd.NA seems
        #  more-correct than np.nan here.
        tm.assert_series_equal(res, expected)
    else:
        with tm.external_error_raised(TypeError):
            ops(str_ser)

    with tm.external_error_raised(TypeError):
        ops(pd.Series(pd.date_range("20180101", periods=len(s))))


# Various
# -----------------------------------------------------------------------------


# TODO test unsigned overflow


def test_arith_coerce_scalar(data, all_arithmetic_operators):
    op = tm.get_op_from_name(all_arithmetic_operators)
    s = pd.Series(data)
    other = 0.01

    result = op(s, other)
    expected = op(s.astype(float), other)
    expected = expected.astype("Float64")

    # rmod results in NaN that wasn't NA in original nullable Series -> unmask it
    if all_arithmetic_operators == "__rmod__":
        mask = (s == 0).fillna(False).to_numpy(bool)
        expected.array._mask[mask] = False

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("other", [1.0, np.array(1.0)])
def test_arithmetic_conversion(all_arithmetic_operators, other):
    # if we have a float operand we should have a float result
    # if that is equal to an integer
    op = tm.get_op_from_name(all_arithmetic_operators)

    s = pd.Series([1, 2, 3], dtype="Int64")
    result = op(s, other)
    assert result.dtype == "Float64"


def test_cross_type_arithmetic():
    df = pd.DataFrame(
        {
            "A": pd.Series([1, 2, np.nan], dtype="Int64"),
            "B": pd.Series([1, np.nan, 3], dtype="UInt8"),
            "C": [1, 2, 3],
        }
    )

    result = df.A + df.C
    expected = pd.Series([2, 4, np.nan], dtype="Int64")
    tm.assert_series_equal(result, expected)

    result = (df.A + df.C) * 3 == 12
    expected = pd.Series([False, True, None], dtype="boolean")
    tm.assert_series_equal(result, expected)

    result = df.A + df.B
    expected = pd.Series([2, np.nan, np.nan], dtype="Int64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("op", ["mean"])
def test_reduce_to_float(op):
    # some reduce ops always return float, even if the result
    # is a rounded number
    df = pd.DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": pd.array([1, None, 3], dtype="Int64"),
        }
    )

    # op
    result = getattr(df.C, op)()
    assert isinstance(result, float)

    # groupby
    result = getattr(df.groupby("A"), op)()

    expected = pd.DataFrame(
        {"B": np.array([1.0, 3.0]), "C": pd.array([1, 3], dtype="Float64")},
        index=pd.Index(["a", "b"], name="A"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "source, neg_target, abs_target",
    [
        ([1, 2, 3], [-1, -2, -3], [1, 2, 3]),
        ([1, 2, None], [-1, -2, None], [1, 2, None]),
        ([-1, 0, 1], [1, 0, -1], [1, 0, 1]),
    ],
)
def test_unary_int_operators(any_signed_int_ea_dtype, source, neg_target, abs_target):
    dtype = any_signed_int_ea_dtype
    arr = pd.array(source, dtype=dtype)
    neg_result, pos_result, abs_result = -arr, +arr, abs(arr)
    neg_target = pd.array(neg_target, dtype=dtype)
    abs_target = pd.array(abs_target, dtype=dtype)

    tm.assert_extension_array_equal(neg_result, neg_target)
    tm.assert_extension_array_equal(pos_result, arr)
    assert not tm.shares_memory(pos_result, arr)
    tm.assert_extension_array_equal(abs_result, abs_target)


def test_values_multiplying_large_series_by_NA():
    # GH#33701

    result = pd.NA * pd.Series(np.zeros(10001))
    expected = pd.Series([pd.NA] * 10001)

    tm.assert_series_equal(result, expected)


def test_bitwise(dtype):
    left = pd.array([1, None, 3, 4], dtype=dtype)
    right = pd.array([None, 3, 5, 4], dtype=dtype)

    result = left | right
    expected = pd.array([None, None, 3 | 5, 4 | 4], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = left & right
    expected = pd.array([None, None, 3 & 5, 4 & 4], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    result = left ^ right
    expected = pd.array([None, None, 3 ^ 5, 4 ^ 4], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    # TODO: desired behavior when operating with boolean?  defer?

    floats = right.astype("Float64")
    with pytest.raises(TypeError, match="unsupported operand type"):
        left | floats
    with pytest.raises(TypeError, match="unsupported operand type"):
        left & floats
    with pytest.raises(TypeError, match="unsupported operand type"):
        left ^ floats

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\base\dim2.py ===
"""
Tests for 2D compatibility.
"""
import numpy as np
import pytest

from pandas._libs.missing import is_matching_na

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer_dtype,
)

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE


class Dim2CompatTests:
    # Note: these are ONLY for ExtensionArray subclasses that support 2D arrays.
    #  i.e. not for pyarrow-backed EAs.

    @pytest.fixture(autouse=True)
    def skip_if_doesnt_support_2d(self, dtype, request):
        if not dtype._supports_2d:
            node = request.node
            # In cases where we are mixed in to ExtensionTests, we only want to
            #  skip tests that are defined in Dim2CompatTests
            test_func = node._obj
            if test_func.__qualname__.startswith("Dim2CompatTests"):
                # TODO: is there a less hacky way of checking this?
                pytest.skip(f"{dtype} does not support 2D.")

    def test_transpose(self, data):
        arr2d = data.repeat(2).reshape(-1, 2)
        shape = arr2d.shape
        assert shape[0] != shape[-1]  # otherwise the rest of the test is useless

        assert arr2d.T.shape == shape[::-1]

    def test_frame_from_2d_array(self, data):
        arr2d = data.repeat(2).reshape(-1, 2)

        df = pd.DataFrame(arr2d)
        expected = pd.DataFrame({0: arr2d[:, 0], 1: arr2d[:, 1]})
        tm.assert_frame_equal(df, expected)

    def test_swapaxes(self, data):
        arr2d = data.repeat(2).reshape(-1, 2)

        result = arr2d.swapaxes(0, 1)
        expected = arr2d.T
        tm.assert_extension_array_equal(result, expected)

    def test_delete_2d(self, data):
        arr2d = data.repeat(3).reshape(-1, 3)

        # axis = 0
        result = arr2d.delete(1, axis=0)
        expected = data.delete(1).repeat(3).reshape(-1, 3)
        tm.assert_extension_array_equal(result, expected)

        # axis = 1
        result = arr2d.delete(1, axis=1)
        expected = data.repeat(2).reshape(-1, 2)
        tm.assert_extension_array_equal(result, expected)

    def test_take_2d(self, data):
        arr2d = data.reshape(-1, 1)

        result = arr2d.take([0, 0, -1], axis=0)

        expected = data.take([0, 0, -1]).reshape(-1, 1)
        tm.assert_extension_array_equal(result, expected)

    def test_repr_2d(self, data):
        # this could fail in a corner case where an element contained the name
        res = repr(data.reshape(1, -1))
        assert res.count(f"<{type(data).__name__}") == 1

        res = repr(data.reshape(-1, 1))
        assert res.count(f"<{type(data).__name__}") == 1

    def test_reshape(self, data):
        arr2d = data.reshape(-1, 1)
        assert arr2d.shape == (data.size, 1)
        assert len(arr2d) == len(data)

        arr2d = data.reshape((-1, 1))
        assert arr2d.shape == (data.size, 1)
        assert len(arr2d) == len(data)

        with pytest.raises(ValueError):
            data.reshape((data.size, 2))
        with pytest.raises(ValueError):
            data.reshape(data.size, 2)

    def test_getitem_2d(self, data):
        arr2d = data.reshape(1, -1)

        result = arr2d[0]
        tm.assert_extension_array_equal(result, data)

        with pytest.raises(IndexError):
            arr2d[1]

        with pytest.raises(IndexError):
            arr2d[-2]

        result = arr2d[:]
        tm.assert_extension_array_equal(result, arr2d)

        result = arr2d[:, :]
        tm.assert_extension_array_equal(result, arr2d)

        result = arr2d[:, 0]
        expected = data[[0]]
        tm.assert_extension_array_equal(result, expected)

        # dimension-expanding getitem on 1D
        result = data[:, np.newaxis]
        tm.assert_extension_array_equal(result, arr2d.T)

    def test_iter_2d(self, data):
        arr2d = data.reshape(1, -1)

        objs = list(iter(arr2d))
        assert len(objs) == arr2d.shape[0]

        for obj in objs:
            assert isinstance(obj, type(data))
            assert obj.dtype == data.dtype
            assert obj.ndim == 1
            assert len(obj) == arr2d.shape[1]

    def test_tolist_2d(self, data):
        arr2d = data.reshape(1, -1)

        result = arr2d.tolist()
        expected = [data.tolist()]

        assert isinstance(result, list)
        assert all(isinstance(x, list) for x in result)

        assert result == expected

    def test_concat_2d(self, data):
        left = type(data)._concat_same_type([data, data]).reshape(-1, 2)
        right = left.copy()

        # axis=0
        result = left._concat_same_type([left, right], axis=0)
        expected = data._concat_same_type([data] * 4).reshape(-1, 2)
        tm.assert_extension_array_equal(result, expected)

        # axis=1
        result = left._concat_same_type([left, right], axis=1)
        assert result.shape == (len(data), 4)
        tm.assert_extension_array_equal(result[:, :2], left)
        tm.assert_extension_array_equal(result[:, 2:], right)

        # axis > 1 -> invalid
        msg = "axis 2 is out of bounds for array of dimension 2"
        with pytest.raises(ValueError, match=msg):
            left._concat_same_type([left, right], axis=2)

    @pytest.mark.parametrize("method", ["backfill", "pad"])
    def test_fillna_2d_method(self, data_missing, method):
        # pad_or_backfill is always along axis=0
        arr = data_missing.repeat(2).reshape(2, 2)
        assert arr[0].isna().all()
        assert not arr[1].isna().any()

        result = arr._pad_or_backfill(method=method, limit=None)

        expected = data_missing._pad_or_backfill(method=method).repeat(2).reshape(2, 2)
        tm.assert_extension_array_equal(result, expected)

        # Reverse so that backfill is not a no-op.
        arr2 = arr[::-1]
        assert not arr2[0].isna().any()
        assert arr2[1].isna().all()

        result2 = arr2._pad_or_backfill(method=method, limit=None)

        expected2 = (
            data_missing[::-1]._pad_or_backfill(method=method).repeat(2).reshape(2, 2)
        )
        tm.assert_extension_array_equal(result2, expected2)

    @pytest.mark.parametrize("method", ["mean", "median", "var", "std", "sum", "prod"])
    def test_reductions_2d_axis_none(self, data, method):
        arr2d = data.reshape(1, -1)

        err_expected = None
        err_result = None
        try:
            expected = getattr(data, method)()
        except Exception as err:
            # if the 1D reduction is invalid, the 2D reduction should be as well
            err_expected = err
            try:
                result = getattr(arr2d, method)(axis=None)
            except Exception as err2:
                err_result = err2

        else:
            result = getattr(arr2d, method)(axis=None)

        if err_result is not None or err_expected is not None:
            assert type(err_result) == type(err_expected)
            return

        assert is_matching_na(result, expected) or result == expected

    @pytest.mark.parametrize("method", ["mean", "median", "var", "std", "sum", "prod"])
    @pytest.mark.parametrize("min_count", [0, 1])
    def test_reductions_2d_axis0(self, data, method, min_count):
        if min_count == 1 and method not in ["sum", "prod"]:
            pytest.skip(f"min_count not relevant for {method}")

        arr2d = data.reshape(1, -1)

        kwargs = {}
        if method in ["std", "var"]:
            # pass ddof=0 so we get all-zero std instead of all-NA std
            kwargs["ddof"] = 0
        elif method in ["prod", "sum"]:
            kwargs["min_count"] = min_count

        try:
            result = getattr(arr2d, method)(axis=0, **kwargs)
        except Exception as err:
            try:
                getattr(data, method)()
            except Exception as err2:
                assert type(err) == type(err2)
                return
            else:
                raise AssertionError("Both reductions should raise or neither")

        def get_reduction_result_dtype(dtype):
            # windows and 32bit builds will in some cases have int32/uint32
            #  where other builds will have int64/uint64.
            if dtype.itemsize == 8:
                return dtype
            elif dtype.kind in "ib":
                return NUMPY_INT_TO_DTYPE[np.dtype(int)]
            else:
                # i.e. dtype.kind == "u"
                return NUMPY_INT_TO_DTYPE[np.dtype("uint")]

        if method in ["sum", "prod"]:
            # std and var are not dtype-preserving
            expected = data
            if data.dtype.kind in "iub":
                dtype = get_reduction_result_dtype(data.dtype)
                expected = data.astype(dtype)
                assert dtype == expected.dtype

            if min_count == 0:
                fill_value = 1 if method == "prod" else 0
                expected = expected.fillna(fill_value)

            tm.assert_extension_array_equal(result, expected)
        elif method == "median":
            # std and var are not dtype-preserving
            expected = data
            tm.assert_extension_array_equal(result, expected)
        elif method in ["mean", "std", "var"]:
            if is_integer_dtype(data) or is_bool_dtype(data):
                data = data.astype("Float64")
            if method == "mean":
                tm.assert_extension_array_equal(result, data)
            else:
                tm.assert_extension_array_equal(result, data - data)

    @pytest.mark.parametrize("method", ["mean", "median", "var", "std", "sum", "prod"])
    def test_reductions_2d_axis1(self, data, method):
        arr2d = data.reshape(1, -1)

        try:
            result = getattr(arr2d, method)(axis=1)
        except Exception as err:
            try:
                getattr(data, method)()
            except Exception as err2:
                assert type(err) == type(err2)
                return
            else:
                raise AssertionError("Both reductions should raise or neither")

        # not necessarily type/dtype-preserving, so weaker assertions
        assert result.shape == (1,)
        expected_scalar = getattr(data, method)()
        res = result[0]
        assert is_matching_na(res, expected_scalar) or res == expected_scalar


class NDArrayBacked2DTests(Dim2CompatTests):
    # More specific tests for NDArrayBackedExtensionArray subclasses

    def test_copy_order(self, data):
        # We should be matching numpy semantics for the "order" keyword in 'copy'
        arr2d = data.repeat(2).reshape(-1, 2)
        assert arr2d._ndarray.flags["C_CONTIGUOUS"]

        res = arr2d.copy()
        assert res._ndarray.flags["C_CONTIGUOUS"]

        res = arr2d[::2, ::2].copy()
        assert res._ndarray.flags["C_CONTIGUOUS"]

        res = arr2d.copy("F")
        assert not res._ndarray.flags["C_CONTIGUOUS"]
        assert res._ndarray.flags["F_CONTIGUOUS"]

        res = arr2d.copy("K")
        assert res._ndarray.flags["C_CONTIGUOUS"]

        res = arr2d.T.copy("K")
        assert not res._ndarray.flags["C_CONTIGUOUS"]
        assert res._ndarray.flags["F_CONTIGUOUS"]

        # order not accepted by numpy
        msg = r"order must be one of 'C', 'F', 'A', or 'K' \(got 'Q'\)"
        with pytest.raises(ValueError, match=msg):
            arr2d.copy("Q")

        # neither contiguity
        arr_nc = arr2d[::2]
        assert not arr_nc._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc._ndarray.flags["F_CONTIGUOUS"]

        assert arr_nc.copy()._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc.copy()._ndarray.flags["F_CONTIGUOUS"]

        assert arr_nc.copy("C")._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc.copy("C")._ndarray.flags["F_CONTIGUOUS"]

        assert not arr_nc.copy("F")._ndarray.flags["C_CONTIGUOUS"]
        assert arr_nc.copy("F")._ndarray.flags["F_CONTIGUOUS"]

        assert arr_nc.copy("K")._ndarray.flags["C_CONTIGUOUS"]
        assert not arr_nc.copy("K")._ndarray.flags["F_CONTIGUOUS"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_basic.py ===
"""This tests sympy/core/basic.py with (ideally) no reference to subclasses
of Basic or Atom."""
import collections
from typing import TypeVar, Generic

from sympy.assumptions.ask import Q
from sympy.core.basic import (Basic, Atom, as_Basic,
    _atomic, _aresame)
from sympy.core.containers import Tuple
from sympy.core.function import Function, Lambda
from sympy.core.numbers import I, pi, Float
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.concrete.summations import Sum
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.functions.elementary.exponential import exp
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.relational import Eq

b1 = Basic()
b2 = Basic(b1)
b3 = Basic(b2)
b21 = Basic(b2, b1)
T = TypeVar('T')


def test__aresame():
    assert not _aresame(Basic(Tuple()), Basic())
    for i, j in [(S(2), S(2.)), (1., Float(1))]:
        for do in range(2):
            assert not _aresame(Basic(i), Basic(j))
            assert not _aresame(i, j)
            i, j = j, i


def test_structure():
    assert b21.args == (b2, b1)
    assert b21.func(*b21.args) == b21
    assert bool(b1)


def test_immutable():
    assert not hasattr(b1, '__dict__')
    with raises(AttributeError):
        b1.x = 1


def test_equality():
    instances = [b1, b2, b3, b21, Basic(b1, b1, b1), Basic]
    for i, b_i in enumerate(instances):
        for j, b_j in enumerate(instances):
            assert (b_i == b_j) == (i == j)
            assert (b_i != b_j) == (i != j)

    assert Basic() != []
    assert not(Basic() == [])
    assert Basic() != 0
    assert not(Basic() == 0)

    class Foo:
        """
        Class that is unaware of Basic, and relies on both classes returning
        the NotImplemented singleton for equivalence to evaluate to False.

        """

    b = Basic()
    foo = Foo()

    assert b != foo
    assert foo != b
    assert not b == foo
    assert not foo == b

    class Bar:
        """
        Class that considers itself equal to any instance of Basic, and relies
        on Basic returning the NotImplemented singleton in order to achieve
        a symmetric equivalence relation.

        """
        def __eq__(self, other):
            if isinstance(other, Basic):
                return True
            return NotImplemented

        def __ne__(self, other):
            return not self == other

    bar = Bar()

    assert b == bar
    assert bar == b
    assert not b != bar
    assert not bar != b


def test_matches_basic():
    instances = [Basic(b1, b1, b2), Basic(b1, b2, b1), Basic(b2, b1, b1),
                 Basic(b1, b2), Basic(b2, b1), b2, b1]
    for i, b_i in enumerate(instances):
        for j, b_j in enumerate(instances):
            if i == j:
                assert b_i.matches(b_j) == {}
            else:
                assert b_i.matches(b_j) is None
    assert b1.match(b1) == {}


def test_has():
    assert b21.has(b1)
    assert b21.has(b3, b1)
    assert b21.has(Basic)
    assert not b1.has(b21, b3)
    assert not b21.has()
    assert not b21.has(str)
    assert not Symbol("x").has("x")


def test_subs():
    assert b21.subs(b2, b1) == Basic(b1, b1)
    assert b21.subs(b2, b21) == Basic(b21, b1)
    assert b3.subs(b2, b1) == b2

    assert b21.subs([(b2, b1), (b1, b2)]) == Basic(b2, b2)

    assert b21.subs({b1: b2, b2: b1}) == Basic(b2, b2)
    assert b21.subs(collections.ChainMap({b1: b2}, {b2: b1})) == Basic(b2, b2)
    assert b21.subs(collections.OrderedDict([(b2, b1), (b1, b2)])) == Basic(b2, b2)

    raises(ValueError, lambda: b21.subs('bad arg'))
    raises(TypeError, lambda: b21.subs(b1, b2, b3))
    # dict(b1=foo) creates a string 'b1' but leaves foo unchanged; subs
    # will convert the first to a symbol but will raise an error if foo
    # cannot be sympified; sympification is strict if foo is not string
    raises(TypeError, lambda: b21.subs(b1='bad arg'))

    assert Symbol("text").subs({"text": b1}) == b1
    assert Symbol("s").subs({"s": 1}) == 1


def test_subs_with_unicode_symbols():
    expr = Symbol('var1')
    replaced = expr.subs('var1', 'x')
    assert replaced.name == 'x'

    replaced = expr.subs('var1', 'x')
    assert replaced.name == 'x'


def test_atoms():
    assert b21.atoms() == {Basic()}


def test_free_symbols_empty():
    assert b21.free_symbols == set()


def test_doit():
    assert b21.doit() == b21
    assert b21.doit(deep=False) == b21


def test_S():
    assert repr(S) == 'S'


def test_xreplace():
    assert b21.xreplace({b2: b1}) == Basic(b1, b1)
    assert b21.xreplace({b2: b21}) == Basic(b21, b1)
    assert b3.xreplace({b2: b1}) == b2
    assert Basic(b1, b2).xreplace({b1: b2, b2: b1}) == Basic(b2, b1)
    assert Atom(b1).xreplace({b1: b2}) == Atom(b1)
    assert Atom(b1).xreplace({Atom(b1): b2}) == b2
    raises(TypeError, lambda: b1.xreplace())
    raises(TypeError, lambda: b1.xreplace([b1, b2]))
    for f in (exp, Function('f')):
        assert f.xreplace({}) == f
        assert f.xreplace({}, hack2=True) == f
        assert f.xreplace({f: b1}) == b1
        assert f.xreplace({f: b1}, hack2=True) == b1


def test_sorted_args():
    x = symbols('x')
    assert b21._sorted_args == b21.args
    raises(AttributeError, lambda: x._sorted_args)

def test_call():
    x, y = symbols('x y')
    # See the long history of this in issues 5026 and 5105.

    raises(TypeError, lambda: sin(x)({ x : 1, sin(x) : 2}))
    raises(TypeError, lambda: sin(x)(1))

    # No effect as there are no callables
    assert sin(x).rcall(1) == sin(x)
    assert (1 + sin(x)).rcall(1) == 1 + sin(x)

    # Effect in the presence of callables
    l = Lambda(x, 2*x)
    assert (l + x).rcall(y) == 2*y + x
    assert (x**l).rcall(2) == x**4
    # TODO UndefinedFunction does not subclass Expr
    #f = Function('f')
    #assert (2*f)(x) == 2*f(x)

    assert (Q.real & Q.positive).rcall(x) == Q.real(x) & Q.positive(x)


def test_rewrite():
    x, y, z = symbols('x y z')
    a, b = symbols('a b')
    f1 = sin(x) + cos(x)
    assert f1.rewrite(cos,exp) == exp(I*x)/2 + sin(x) + exp(-I*x)/2
    assert f1.rewrite([cos],sin) == sin(x) + sin(x + pi/2, evaluate=False)
    f2 = sin(x) + cos(y)/gamma(z)
    assert f2.rewrite(sin,exp) == -I*(exp(I*x) - exp(-I*x))/2 + cos(y)/gamma(z)

    assert f1.rewrite() == f1

def test_literal_evalf_is_number_is_zero_is_comparable():
    x = symbols('x')
    f = Function('f')

    # issue 5033
    assert f.is_number is False
    # issue 6646
    assert f(1).is_number is False
    i = Integral(0, (x, x, x))
    # expressions that are symbolically 0 can be difficult to prove
    # so in case there is some easy way to know if something is 0
    # it should appear in the is_zero property for that object;
    # if is_zero is true evalf should always be able to compute that
    # zero
    assert i.n() == 0
    assert i.is_zero
    assert i.is_number is False
    assert i.evalf(2, strict=False) == 0

    # issue 10268
    n = sin(1)**2 + cos(1)**2 - 1
    assert n.is_comparable is False
    assert n.n(2).is_comparable is False
    assert n.n(2).n(2).is_comparable


def test_as_Basic():
    assert as_Basic(1) is S.One
    assert as_Basic(()) == Tuple()
    raises(TypeError, lambda: as_Basic([]))


def test_atomic():
    g, h = map(Function, 'gh')
    x = symbols('x')
    assert _atomic(g(x + h(x))) == {g(x + h(x))}
    assert _atomic(g(x + h(x)), recursive=True) == {h(x), x, g(x + h(x))}
    assert _atomic(1) == set()
    assert _atomic(Basic(S(1), S(2))) == set()


def test_as_dummy():
    u, v, x, y, z, _0, _1 = symbols('u v x y z _0 _1')
    assert Lambda(x, x + 1).as_dummy() == Lambda(_0, _0 + 1)
    assert Lambda(x, x + _0).as_dummy() == Lambda(_1, _0 + _1)
    eq = (1 + Sum(x, (x, 1, x)))
    ans = 1 + Sum(_0, (_0, 1, x))
    once = eq.as_dummy()
    assert once == ans
    twice = once.as_dummy()
    assert twice == ans
    assert Integral(x + _0, (x, x + 1), (_0, 1, 2)
        ).as_dummy() == Integral(_0 + _1, (_0, x + 1), (_1, 1, 2))
    for T in (Symbol, Dummy):
        d = T('x', real=True)
        D = d.as_dummy()
        assert D != d and D.func == Dummy and D.is_real is None
    assert Dummy().as_dummy().is_commutative
    assert Dummy(commutative=False).as_dummy().is_commutative is False


def test_canonical_variables():
    x, i0, i1 = symbols('x _:2')
    assert Integral(x, (x, x + 1)).canonical_variables == {x: i0}
    assert Integral(x, (x, x + 1), (i0, 1, 2)).canonical_variables == {
        x: i0, i0: i1}
    assert Integral(x, (x, x + i0)).canonical_variables == {x: i1}


def test_replace_exceptions():
    from sympy.core.symbol import Wild
    x, y = symbols('x y')
    e = (x**2 + x*y)
    raises(TypeError, lambda: e.replace(sin, 2))
    b = Wild('b')
    c = Wild('c')
    raises(TypeError, lambda: e.replace(b*c, c.is_real))
    raises(TypeError, lambda: e.replace(b.is_real, 1))
    raises(TypeError, lambda: e.replace(lambda d: d.is_Number, 1))


def test_ManagedProperties():
    # ManagedProperties is now deprecated. Here we do our best to check that if
    # someone is using it then it does work in the way that it previously did
    # but gives a deprecation warning.
    from sympy.core.assumptions import ManagedProperties

    myclasses = []

    class MyMeta(ManagedProperties):
        def __init__(cls, *args, **kwargs):
            myclasses.append('executed')
            super().__init__(*args, **kwargs)

    code = """
class MySubclass(Basic, metaclass=MyMeta):
    pass
"""
    with warns_deprecated_sympy():
        exec(code)

    assert myclasses == ['executed']


def test_generic():
    # https://github.com/sympy/sympy/issues/25399
    class A(Symbol, Generic[T]):
        pass

    class B(A[T]):
        pass


def test_rewrite_abs():
    # https://github.com/sympy/sympy/issues/27323
    x = Symbol('x')
    assert sign(x).rewrite(abs) == sign(x).rewrite(Abs)
    assert sign(x).rewrite(abs) == Piecewise((0, Eq(x, 0)), (x / Abs(x), True))