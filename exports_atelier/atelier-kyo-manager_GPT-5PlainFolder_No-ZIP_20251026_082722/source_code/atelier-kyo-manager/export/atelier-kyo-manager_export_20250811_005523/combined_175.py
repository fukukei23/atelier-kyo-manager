
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\tests\test_tensor.py ===
from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add, Wild, Function
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals import integrate
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
    get_symmetric_group_sgs, TensorIndex, tensor_mul, TensAdd, \
    riemann_cyclic_replace, riemann_cyclic, TensMul, tensor_heads, \
    TensorManager, TensExpr, TensorHead, canon_bp, \
    tensorhead, tensorsymmetry, TensorType, substitute_indices, \
    WildTensorIndex, WildTensorHead, _WildTensExpr
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag

def _is_equal(arg1, arg2):
    if isinstance(arg1, TensExpr):
        return arg1.equals(arg2)
    elif isinstance(arg2, TensExpr):
        return arg2.equals(arg1)
    return arg1 == arg2


#################### Tests from tensor_can.py #######################
def test_canonicalize_no_slot_sym():
    # A_d0 * B^d0; T_c = A^d0*B_d0
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, d0, d1 = tensor_indices('a,b,d0,d1', Lorentz)
    A, B = tensor_heads('A,B', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(-d0)*B(d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0)*B(-L_0)'

    # A^a * B^b;  T_c = T
    t = A(a)*B(b)
    tc = t.canon_bp()
    assert tc == t
    # B^b * A^a
    t1 = B(b)*A(a)
    tc = t1.canon_bp()
    assert str(tc) == 'A(a)*B(b)'

    # A symmetric
    # A^{b}_{d0}*A^{d0, a}; T_c = A^{a d0}*A{b}_{d0}
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(b, -d0)*A(d0, a)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, L_0)*A(b, -L_0)'

    # A^{d1}_{d0}*B^d0*C_d1
    # T_c = A^{d0 d1}*B_d0*C_d1
    B, C = tensor_heads('B,C', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(d1, -d0)*B(d0)*C(-d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_0)*C(-L_1)'

    # A without symmetry
    # A^{d1}_{d0}*B^d0*C_d1 ord=[d0,-d0,d1,-d1]; g = [2,1,0,3,4,5]
    # T_c = A^{d0 d1}*B_d1*C_d0; can = [0,2,3,1,4,5]
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    t = A(d1, -d0)*B(d0)*C(-d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_1)*C(-L_0)'

    # A, B without symmetry
    # A^{d1}_{d0}*B_{d1}^{d0}
    # T_c = A^{d0 d1}*B_{d0 d1}
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    t = A(d1, -d0)*B(-d1, d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_0, -L_1)'
    # A_{d0}^{d1}*B_{d1}^{d0}
    # T_c = A^{d0 d1}*B_{d1 d0}
    t = A(-d0, d1)*B(-d1, d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_1, -L_0)'

    # A, B, C without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b}
    # T_c=A^{d0 d1}*B_{a d1}*C_{d0 b}
    C = TensorHead('C', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    t = A(d1, d0)*B(-a, -d0)*C(-d1, -b)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_1)*C(-L_0, -b)'

    # A symmetric, B and C without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b}
    # T_c = A^{d0 d1}*B_{a d0}*C_{d1 b}
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(d1, d0)*B(-a, -d0)*C(-d1, -b)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_0)*C(-L_1, -b)'

    # A and C symmetric, B without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # T_c = A^{d0 d1}*B_{a d0}*C_{b d1}
    C = TensorHead('C', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(d1, d0)*B(-a, -d0)*C(-d1, -b)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_0)*C(-b, -L_1)'

def test_canonicalize_no_dummies():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a, b, c, d', Lorentz)

    # A commuting
    # A^c A^b A^a
    # T_c = A^a A^b A^c
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(c)*A(b)*A(a)
    tc = t.canon_bp()
    assert str(tc) == 'A(a)*A(b)*A(c)'

    # A anticommuting
    # A^c A^b A^a
    # T_c = -A^a A^b A^c
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1), 1)
    t = A(c)*A(b)*A(a)
    tc = t.canon_bp()
    assert str(tc) == '-A(a)*A(b)*A(c)'

    # A commuting and symmetric
    # A^{b,d}*A^{c,a}
    # T_c = A^{a c}*A^{b d}
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(b, d)*A(c, a)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, c)*A(b, d)'

    # A anticommuting and symmetric
    # A^{b,d}*A^{c,a}
    # T_c = -A^{a c}*A^{b d}
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2), 1)
    t = A(b, d)*A(c, a)
    tc = t.canon_bp()
    assert str(tc) == '-A(a, c)*A(b, d)'

    # A^{c,a}*A^{b,d}
    # T_c = A^{a c}*A^{b d}
    t = A(c, a)*A(b, d)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, c)*A(b, d)'

def test_tensorhead_construction_without_symmetry():
    L = TensorIndexType('Lorentz')
    A1 = TensorHead('A', [L, L])
    A2 = TensorHead('A', [L, L], TensorSymmetry.no_symmetry(2))
    assert A1 == A2
    A3 = TensorHead('A', [L, L], TensorSymmetry.fully_symmetric(2))  # Symmetric
    assert A1 != A3

def test_no_metric_symmetry():
    # no metric symmetry; A no symmetry
    # A^d1_d0 * A^d0_d1
    # T_c = A^d0_d1 * A^d1_d0
    Lorentz = TensorIndexType('Lorentz', dummy_name='L', metric_symmetry=0)
    d0, d1, d2, d3 = tensor_indices('d:4', Lorentz)
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    t = A(d1, -d0)*A(d0, -d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_0)'

    # A^d1_d2 * A^d0_d3 * A^d2_d1 * A^d3_d0
    # T_c = A^d0_d1 * A^d1_d0 * A^d2_d3 * A^d3_d2
    t = A(d1, -d2)*A(d0, -d3)*A(d2, -d1)*A(d3, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_0)*A(L_2, -L_3)*A(L_3, -L_2)'

    # A^d0_d2 * A^d1_d3 * A^d3_d0 * A^d2_d1
    # T_c = A^d0_d1 * A^d1_d2 * A^d2_d3 * A^d3_d0
    t = A(d0, -d1)*A(d1, -d2)*A(d2, -d3)*A(d3, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_2)*A(L_2, -L_3)*A(L_3, -L_0)'

def test_canonicalize1():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = \
        tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Lorentz)

    # A_d0*A^d0; ord = [d0,-d0]
    # T_c = A^d0*A_d0
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(-d0)*A(d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0)*A(-L_0)'

    # A commuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0
    # T_c = A^d0*A_d0*A^d1*A_d1*A^d2*A_d2
    t = A(-d0)*A(-d1)*A(-d2)*A(d2)*A(d1)*A(d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0)*A(-L_0)*A(L_1)*A(-L_1)*A(L_2)*A(-L_2)'

    # A anticommuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0
    # T_c 0
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1), 1)
    t = A(-d0)*A(-d1)*A(-d2)*A(d2)*A(d1)*A(d0)
    tc = t.canon_bp()
    assert tc == 0

    # A commuting symmetric
    # A^{d0 b}*A^a_d1*A^d1_d0
    # T_c = A^{a d0}*A^{b d1}*A_{d0 d1}
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(d0, b)*A(a, -d1)*A(d1, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, L_0)*A(b, L_1)*A(-L_0, -L_1)'

    # A, B commuting symmetric
    # A^{d0 b}*A^d1_d0*B^a_d1
    # T_c = A^{b d0}*A_d0^d1*B^a_d1
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(d0, b)*A(d1, -d0)*B(a, -d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(b, L_0)*A(-L_0, L_1)*B(a, -L_1)'

    # A commuting symmetric
    # A^{d1 d0 b}*A^{a}_{d1 d0}; ord=[a,b, d0,-d0,d1,-d1]
    # T_c = A^{a d0 d1}*A^{b}_{d0 d1}
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(3))
    t = A(d1, d0, b)*A(a, -d1, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, L_0, L_1)*A(b, -L_0, -L_1)'

    # A^{d3 d0 d2}*A^a0_{d1 d2}*A^d1_d3^a1*A^{a2 a3}_d0
    # T_c = A^{a0 d0 d1}*A^a1_d0^d2*A^{a2 a3 d3}*A_{d1 d2 d3}
    t = A(d3, d0, d2)*A(a0, -d1, -d2)*A(d1, -d3, a1)*A(a2, a3, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(a0, L_0, L_1)*A(a1, -L_0, L_2)*A(a2, a3, L_3)*A(-L_1, -L_2, -L_3)'

    # A commuting symmetric, B antisymmetric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # in this esxample and in the next three,
    # renaming dummy indices and using symmetry of A,
    # T = A^{d0 d1 d2} * A_{d0 d1 d3} * B_d2^d3
    # can = 0
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(3))
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    tc = t.canon_bp()
    assert tc == 0

    # A anticommuting symmetric, B antisymmetric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = A^{d0 d1 d2} * A_{d0 d1}^d3 * B_{d2 d3}
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1, L_2)*A(-L_0, -L_1, L_3)*B(-L_2, -L_3)'

    # A anticommuting symmetric, B antisymmetric commuting, antisymmetric metric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = -A^{d0 d1 d2} * A_{d0 d1}^d3 * B_{d2 d3}
    Spinor = TensorIndexType('Spinor', dummy_name='S', metric_symmetry=-1)
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = \
        tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Spinor)
    A = TensorHead('A', [Spinor]*3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Spinor]*2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    tc = t.canon_bp()
    assert str(tc) == '-A(S_0, S_1, S_2)*A(-S_0, -S_1, S_3)*B(-S_2, -S_3)'

    # A anticommuting symmetric, B antisymmetric anticommuting,
    # no metric symmetry
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = A^{d0 d1 d2} * A_{d0 d1 d3} * B_d2^d3
    Mat = TensorIndexType('Mat', metric_symmetry=0, dummy_name='M')
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = \
        tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Mat)
    A = TensorHead('A', [Mat]*3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Mat]*2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    tc = t.canon_bp()
    assert str(tc) == 'A(M_0, M_1, M_2)*A(-M_0, -M_1, -M_3)*B(-M_2, M_3)'

    # Gamma anticommuting
    # Gamma_{mu nu} * gamma^rho * Gamma^{nu mu alpha}
    # T_c = -Gamma^{mu nu} * gamma^rho * Gamma_{alpha mu nu}
    alpha, beta, gamma, mu, nu, rho = \
        tensor_indices('alpha,beta,gamma,mu,nu,rho', Lorentz)
    Gamma = TensorHead('Gamma', [Lorentz],
                       TensorSymmetry.fully_symmetric(1), 2)
    Gamma2 = TensorHead('Gamma', [Lorentz]*2,
                        TensorSymmetry.fully_symmetric(-2), 2)
    Gamma3 = TensorHead('Gamma', [Lorentz]*3,
                        TensorSymmetry.fully_symmetric(-3), 2)
    t = Gamma2(-mu, -nu)*Gamma(rho)*Gamma3(nu, mu, alpha)
    tc = t.canon_bp()
    assert str(tc) == '-Gamma(L_0, L_1)*Gamma(rho)*Gamma(alpha, -L_0, -L_1)'

    # Gamma_{mu nu} * Gamma^{gamma beta} * gamma_rho * Gamma^{nu mu alpha}
    # T_c = Gamma^{mu nu} * Gamma^{beta gamma} * gamma_rho * Gamma^alpha_{mu nu}
    t = Gamma2(mu, nu)*Gamma2(beta, gamma)*Gamma(-rho)*Gamma3(alpha, -mu, -nu)
    tc = t.canon_bp()
    assert str(tc) == 'Gamma(L_0, L_1)*Gamma(beta, gamma)*Gamma(-rho)*Gamma(alpha, -L_0, -L_1)'

    # f^a_{b,c} antisymmetric in b,c; A_mu^a no symmetry
    # f^c_{d a} * f_{c e b} * A_mu^d * A_nu^a * A^{nu e} * A^{mu b}
    # g = [8,11,5, 9,13,7, 1,10, 3,4, 2,12, 0,6, 14,15]
    # T_c = -f^{a b c} * f_a^{d e} * A^mu_b * A_{mu d} * A^nu_c * A_{nu e}
    Flavor = TensorIndexType('Flavor', dummy_name='F')
    a, b, c, d, e, ff = tensor_indices('a,b,c,d,e,f', Flavor)
    mu, nu = tensor_indices('mu,nu', Lorentz)
    f = TensorHead('f', [Flavor]*3, TensorSymmetry.direct_product(1, -2))
    A = TensorHead('A', [Lorentz, Flavor], TensorSymmetry.no_symmetry(2))
    t = f(c, -d, -a)*f(-c, -e, -b)*A(-mu, d)*A(-nu, a)*A(nu, e)*A(mu, b)
    tc = t.canon_bp()
    assert str(tc) == '-f(F_0, F_1, F_2)*f(-F_0, F_3, F_4)*A(L_0, -F_1)*A(-L_0, -F_3)*A(L_1, -F_2)*A(-L_1, -F_4)'


def test_bug_correction_tensor_indices():
    # to make sure that tensor_indices does not return a list if creating
    # only one index:
    A = TensorIndexType("A")
    i = tensor_indices('i', A)
    assert not isinstance(i, (tuple, list))
    assert isinstance(i, TensorIndex)


def test_riemann_invariants():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11 = \
        tensor_indices('d0:12', Lorentz)
    # R^{d0 d1}_{d1 d0}; ord = [d0,-d0,d1,-d1]
    # T_c = -R^{d0 d1}_{d0 d1}
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    t = R(d0, d1, -d1, -d0)
    tc = t.canon_bp()
    assert str(tc) == '-R(L_0, L_1, -L_0, -L_1)'

    # R_d11^d1_d0^d5 * R^{d6 d4 d0}_d5 * R_{d7 d2 d8 d9} *
    # R_{d10 d3 d6 d4} * R^{d2 d7 d11}_d1 * R^{d8 d9 d3 d10}
    # can = [0,2,4,6, 1,3,8,10, 5,7,12,14, 9,11,16,18, 13,15,20,22,
    #        17,19,21<F10,23, 24,25]
    # T_c = R^{d0 d1 d2 d3} * R_{d0 d1}^{d4 d5} * R_{d2 d3}^{d6 d7} *
    # R_{d4 d5}^{d8 d9} * R_{d6 d7}^{d10 d11} * R_{d8 d9 d10 d11}

    t = R(-d11,d1,-d0,d5)*R(d6,d4,d0,-d5)*R(-d7,-d2,-d8,-d9)* \
        R(-d10,-d3,-d6,-d4)*R(d2,d7,d11,-d1)*R(d8,d9,d3,d10)
    tc = t.canon_bp()
    assert str(tc) == 'R(L_0, L_1, L_2, L_3)*R(-L_0, -L_1, L_4, L_5)*R(-L_2, -L_3, L_6, L_7)*R(-L_4, -L_5, L_8, L_9)*R(-L_6, -L_7, L_10, L_11)*R(-L_8, -L_9, -L_10, -L_11)'

def test_riemann_products():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    d0, d1, d2, d3, d4, d5, d6 = tensor_indices('d0:7', Lorentz)
    a0, a1, a2, a3, a4, a5 = tensor_indices('a0:6', Lorentz)
    a, b = tensor_indices('a,b', Lorentz)
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    # R^{a b d0}_d0 = 0
    t = R(a, b, d0, -d0)
    tc = t.canon_bp()
    assert tc == 0

    # R^{d0 b a}_d0
    # T_c = -R^{a d0 b}_d0
    t = R(d0, b, a, -d0)
    tc = t.canon_bp()
    assert str(tc) == '-R(a, L_0, b, -L_0)'

    # R^d1_d2^b_d0 * R^{d0 a}_d1^d2; ord=[a,b,d0,-d0,d1,-d1,d2,-d2]
    # T_c = -R^{a d0 d1 d2}* R^b_{d0 d1 d2}
    t = R(d1, -d2, b, -d0)*R(d0, a, -d1, d2)
    tc = t.canon_bp()
    assert str(tc) == '-R(a, L_0, L_1, L_2)*R(b, -L_0, -L_1, -L_2)'

    # A symmetric commuting
    # R^{d6 d5}_d2^d1 * R^{d4 d0 d2 d3} * A_{d6 d0} A_{d3 d1} * A_{d4 d5}
    # g = [12,10,5,2, 8,0,4,6, 13,1, 7,3, 9,11,14,15]
    # T_c = -R^{d0 d1 d2 d3} * R_d0^{d4 d5 d6} * A_{d1 d4}*A_{d2 d5}*A_{d3 d6}
    V = TensorHead('V', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = R(d6, d5, -d2, d1)*R(d4, d0, d2, d3)*V(-d6, -d0)*V(-d3, -d1)*V(-d4, -d5)
    tc = t.canon_bp()
    assert str(tc) == '-R(L_0, L_1, L_2, L_3)*R(-L_0, L_4, L_5, L_6)*V(-L_1, -L_4)*V(-L_2, -L_5)*V(-L_3, -L_6)'

    # R^{d2 a0 a2 d0} * R^d1_d2^{a1 a3} * R^{a4 a5}_{d0 d1}
    # T_c = R^{a0 d0 a2 d1}*R^{a1 a3}_d0^d2*R^{a4 a5}_{d1 d2}
    t = R(d2, a0, a2, d0)*R(d1, -d2, a1, a3)*R(a4, a5, -d0, -d1)
    tc = t.canon_bp()
    assert str(tc) == 'R(a0, L_0, a2, L_1)*R(a1, a3, -L_0, L_2)*R(a4, a5, -L_1, -L_2)'
######################################################################


def test_canonicalize2():
    D = Symbol('D')
    Eucl = TensorIndexType('Eucl', metric_symmetry=1, dim=D, dummy_name='E')
    i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14 = \
        tensor_indices('i0:15', Eucl)
    A = TensorHead('A', [Eucl]*3, TensorSymmetry.fully_symmetric(-3))

    # two examples from Cvitanovic, Group Theory page 59
    # of identities for antisymmetric tensors of rank 3
    # contracted according to the Kuratowski graph  eq.(6.59)
    t = A(i0,i1,i2)*A(-i1,i3,i4)*A(-i3,i7,i5)*A(-i2,-i5,i6)*A(-i4,-i6,i8)
    t1 = t.canon_bp()
    assert t1 == 0

    # eq.(6.60)
    #t = A(i0,i1,i2)*A(-i1,i3,i4)*A(-i2,i5,i6)*A(-i3,i7,i8)*A(-i6,-i7,i9)*
    #    A(-i8,i10,i13)*A(-i5,-i10,i11)*A(-i4,-i11,i12)*A(-i3,-i12,i14)
    t = A(i0,i1,i2)*A(-i1,i3,i4)*A(-i2,i5,i6)*A(-i3,i7,i8)*A(-i6,-i7,i9)*\
        A(-i8,i10,i13)*A(-i5,-i10,i11)*A(-i4,-i11,i12)*A(-i9,-i12,i14)
    t1 = t.canon_bp()
    assert t1 == 0

def test_canonicalize3():
    D = Symbol('D')
    Spinor = TensorIndexType('Spinor', dim=D, metric_symmetry=-1, dummy_name='S')
    a0,a1,a2,a3,a4 = tensor_indices('a0:5', Spinor)
    chi, psi = tensor_heads('chi,psi', [Spinor], TensorSymmetry.no_symmetry(1), 1)

    t = chi(a1)*psi(a0)
    t1 = t.canon_bp()
    assert t1 == t

    t = psi(a1)*chi(a0)
    t1 = t.canon_bp()
    assert t1 == -chi(a0)*psi(a1)

def test_canonicalize4():
    #Check whether TensAdd.canon_bp chokes on a case where the type of the expression changes on calling expand
    Cartesian = TensorIndexType('Cartesian', dim=3)
    p = tensor_indices("p", Cartesian)
    K = TensorHead("K", [Cartesian])
    expr = TensAdd( K(p) , - 2*K(p) )
    assert expr.canon_bp() == -K(p)

def test_canonicalize5():
    R3 = TensorIndexType('R3', dim=3)
    p = tensor_indices("p", R3)
    K = TensorHead("K", [R3])
    f = symbols("f", cls=Function)
    x = symbols("x")

    expr = integrate(f(x), (x,0,1)) * K(p)
    assert expr.as_dummy().canon_bp() == integrate(f(x), (x,0,1)).as_dummy() * K(p)

def test_TensorIndexType():
    D = Symbol('D')
    Lorentz = TensorIndexType('Lorentz', metric_name='g', metric_symmetry=1,
                              dim=D, dummy_name='L')
    m0, m1, m2, m3, m4 = tensor_indices('m0:5', Lorentz)
    sym2 = TensorSymmetry.fully_symmetric(2)
    sym2n = TensorSymmetry(*get_symmetric_group_sgs(2))
    assert sym2 == sym2n
    g = Lorentz.metric
    assert str(g) == 'g(Lorentz,Lorentz)'
    assert Lorentz.eps_dim == Lorentz.dim

    TSpace = TensorIndexType('TSpace', dummy_name = 'TSpace')
    i0, i1 = tensor_indices('i0 i1', TSpace)
    g = TSpace.metric
    A = TensorHead('A', [TSpace]*2, sym2)
    assert str(A(i0,-i0).canon_bp()) == 'A(TSpace_0, -TSpace_0)'

def test_indices():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    assert a.tensor_index_type == Lorentz
    assert a != -a
    A, B = tensor_heads('A B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(a,b)*B(-b,c)
    indices = t.get_indices()
    L_0 = TensorIndex('L_0', Lorentz)
    assert indices == [a, L_0, -L_0, c]
    raises(ValueError, lambda: tensor_indices(3, Lorentz))
    raises(ValueError, lambda: A(a,b,c))

    A = TensorHead('A', [Lorentz, Lorentz])
    assert A('a', 'b') == A(TensorIndex('a', Lorentz),
                            TensorIndex('b', Lorentz))
    assert A('a', '-b') == A(TensorIndex('a', Lorentz),
                             TensorIndex('b', Lorentz, is_up=False))
    assert A('a', TensorIndex('b', Lorentz)) == A(TensorIndex('a', Lorentz),
                                                  TensorIndex('b', Lorentz))

def test_TensorSymmetry():
    assert TensorSymmetry.fully_symmetric(2) == \
        TensorSymmetry(get_symmetric_group_sgs(2))
    assert TensorSymmetry.fully_symmetric(-3) == \
        TensorSymmetry(get_symmetric_group_sgs(3, True))
    assert TensorSymmetry.direct_product(-4) == \
        TensorSymmetry.fully_symmetric(-4)
    assert TensorSymmetry.fully_symmetric(-1) == \
        TensorSymmetry.fully_symmetric(1)
    assert TensorSymmetry.direct_product(1, -1, 1) == \
        TensorSymmetry.no_symmetry(3)
    assert TensorSymmetry(get_symmetric_group_sgs(2)) == \
        TensorSymmetry(*get_symmetric_group_sgs(2))
    # TODO: add check for *get_symmetric_group_sgs(0)
    sym = TensorSymmetry.fully_symmetric(-3)
    assert sym.rank == 3
    assert sym.base == Tuple(0, 1)
    assert sym.generators == Tuple(Permutation(0, 1)(3, 4), Permutation(1, 2)(3, 4))

def test_TensExpr():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    g = Lorentz.metric
    A, B = tensor_heads('A B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    raises(ValueError, lambda: g(c, d)/g(a, b))
    raises(ValueError, lambda: S.One/g(a, b))
    raises(ValueError, lambda: (A(c, d) + g(c, d))/g(a, b))
    raises(ValueError, lambda: S.One/(A(c, d) + g(c, d)))
    raises(ValueError, lambda: A(a, b) + A(a, c))

    #t = A(a, b) + B(a, b) # assigned to t for below
    #raises(NotImplementedError, lambda: TensExpr.__mul__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__add__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__radd__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__sub__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__rsub__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__truediv__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__rtruediv__(t, 'a'))
    with warns_deprecated_sympy():
        # DO NOT REMOVE THIS AFTER DEPRECATION REMOVED:
        raises(ValueError, lambda: A(a, b)**2)
    raises(NotImplementedError, lambda: 2**A(a, b))
    raises(NotImplementedError, lambda: abs(A(a, b)))

def test_TensorHead():
    # simple example of algebraic expression
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    A = TensorHead('A', [Lorentz]*2)
    assert A.name == 'A'
    assert A.index_types == [Lorentz, Lorentz]
    assert A.rank == 2
    assert A.symmetry == TensorSymmetry.no_symmetry(2)
    assert A.comm == 0


def test_add1():
    assert TensAdd().args == ()
    assert TensAdd().doit() == 0
    # simple example of algebraic expression
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a,b,d0,d1,i,j,k = tensor_indices('a,b,d0,d1,i,j,k', Lorentz)
    # A, B symmetric
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t1 = A(b, -d0)*B(d0, a)
    assert TensAdd(t1).equals(t1)
    t2a = B(d0, a) + A(d0, a)
    t2 = A(b, -d0)*t2a
    assert str(t2) == 'A(b, -L_0)*(A(L_0, a) + B(L_0, a))'
    t2 = t2.expand()
    assert str(t2) == 'A(b, -L_0)*A(L_0, a) + A(b, -L_0)*B(L_0, a)'
    t2 = t2.canon_bp()
    assert str(t2) == 'A(a, L_0)*A(b, -L_0) + A(b, L_0)*B(a, -L_0)'
    t2b = t2 + t1
    assert str(t2b) == 'A(a, L_0)*A(b, -L_0) + A(b, -L_0)*B(L_0, a) + A(b, L_0)*B(a, -L_0)'
    t2b = t2b.canon_bp()
    assert str(t2b) == 'A(a, L_0)*A(b, -L_0) + 2*A(b, L_0)*B(a, -L_0)'
    p, q, r = tensor_heads('p,q,r', [Lorentz])
    t = q(d0)*2
    assert str(t) == '2*q(d0)'
    t = 2*q(d0)
    assert str(t) == '2*q(d0)'
    t1 = p(d0) + 2*q(d0)
    assert str(t1) == '2*q(d0) + p(d0)'
    t2 = p(-d0) + 2*q(-d0)
    assert str(t2) == '2*q(-d0) + p(-d0)'
    t1 = p(d0)
    t3 = t1*t2
    assert str(t3) == 'p(L_0)*(2*q(-L_0) + p(-L_0))'
    t3 = t3.expand()
    assert str(t3) == 'p(L_0)*p(-L_0) + 2*p(L_0)*q(-L_0)'
    t3 = t2*t1
    t3 = t3.expand()
    assert str(t3) == 'p(-L_0)*p(L_0) + 2*q(-L_0)*p(L_0)'
    t3 = t3.canon_bp()
    assert str(t3) == 'p(L_0)*p(-L_0) + 2*p(L_0)*q(-L_0)'
    t1 = p(d0) + 2*q(d0)
    t3 = t1*t2
    t3 = t3.canon_bp()
    assert str(t3) == 'p(L_0)*p(-L_0) + 4*p(L_0)*q(-L_0) + 4*q(L_0)*q(-L_0)'
    t1 = p(d0) - 2*q(d0)
    assert str(t1) == '-2*q(d0) + p(d0)'
    t2 = p(-d0) + 2*q(-d0)
    t3 = t1*t2
    t3 = t3.canon_bp()
    assert t3 == p(d0)*p(-d0) - 4*q(d0)*q(-d0)
    t = p(i)*p(j)*(p(k) + q(k)) + p(i)*(p(j) + q(j))*(p(k) - 3*q(k))
    t = t.canon_bp()
    assert t == 2*p(i)*p(j)*p(k) - 2*p(i)*p(j)*q(k) + p(i)*p(k)*q(j) - 3*p(i)*q(j)*q(k)
    t1 = (p(i) + q(i) + 2*r(i))*(p(j) - q(j))
    t2 = (p(j) + q(j) + 2*r(j))*(p(i) - q(i))
    t = t1 + t2
    t = t.canon_bp()
    assert t == 2*p(i)*p(j) + 2*p(i)*r(j) + 2*p(j)*r(i) - 2*q(i)*q(j) - 2*q(i)*r(j) - 2*q(j)*r(i)
    t = p(i)*q(j)/2
    assert 2*t == p(i)*q(j)
    t = (p(i) + q(i))/2
    assert 2*t == p(i) + q(i)

    t = S.One - p(i)*p(-i)
    t = t.canon_bp()
    tz1 = t + p(-j)*p(j)
    assert tz1 != 1
    tz1 = tz1.canon_bp()
    assert tz1.equals(1)
    t = S.One + p(i)*p(-i)
    assert (t - p(-j)*p(j)).canon_bp().equals(1)

    t = A(a, b) + B(a, b)
    assert t.rank == 2
    t1 = t - A(a, b) - B(a, b)
    assert t1 == 0
    t = 1 - (A(a, -a) + B(a, -a))
    t1 = 1 + (A(a, -a) + B(a, -a))
    assert (t + t1).expand().equals(2)
    t2 = 1 + A(a, -a)
    assert t1 != t2
    assert t2 != TensMul.from_data(0, [], [], [])

    #Test whether TensAdd.doit chokes on subterms that are zero.
    assert TensAdd(p(a), TensMul(0, p(a)) ).doit() == p(a)

def test_special_eq_ne():
    # test special equality cases:
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, d0, d1, i, j, k = tensor_indices('a,b,d0,d1,i,j,k', Lorentz)
    # A, B symmetric
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    p, q, r = tensor_heads('p,q,r', [Lorentz])

    t = 0*A(a, b)
    assert _is_equal(t, 0)
    assert _is_equal(t, S.Zero)

    assert p(i) != A(a, b)
    assert A(a, -a) != A(a, b)
    assert 0*(A(a, b) + B(a, b)) == 0
    assert 0*(A(a, b) + B(a, b)) is S.Zero

    assert 3*(A(a, b) - A(a, b)) is S.Zero

    assert p(i) + q(i) != A(a, b)
    assert p(i) + q(i) != A(a, b) + B(a, b)

    assert p(i) - p(i) == 0
    assert p(i) - p(i) is S.Zero

    assert _is_equal(A(a, b), A(b, a))

def test_add2():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    m, n, p, q = tensor_indices('m,n,p,q', Lorentz)
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(-3))
    t1 = 2*R(m, n, p, q) - R(m, q, n, p) + R(m, p, n, q)
    t2 = t1*A(-n, -p, -q)
    t2 = t2.canon_bp()
    assert t2 == 0
    t1 = Rational(2, 3)*R(m,n,p,q) - Rational(1, 3)*R(m,q,n,p) + Rational(1, 3)*R(m,p,n,q)
    t2 = t1*A(-n, -p, -q)
    t2 = t2.canon_bp()
    assert t2 == 0
    t = A(m, -m, n) + A(n, p, -p)
    t = t.canon_bp()
    assert t == 0

def test_add3():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    i0, i1 = tensor_indices('i0:2', Lorentz)
    E, px, py, pz = symbols('E px py pz')
    A = TensorHead('A', [Lorentz])
    B = TensorHead('B', [Lorentz])

    expr1 = A(i0)*A(-i0) - (E**2 - px**2 - py**2 - pz**2)
    assert expr1.args == (-E**2, px**2, py**2, pz**2, A(i0)*A(-i0))

    expr2 = E**2 - px**2 - py**2 - pz**2 - A(i0)*A(-i0)
    assert expr2.args == (E**2, -px**2, -py**2, -pz**2, -A(i0)*A(-i0))

    expr3 = A(i0)*A(-i0) - E**2 + px**2 + py**2 + pz**2
    assert expr3.args == (-E**2, px**2, py**2, pz**2, A(i0)*A(-i0))

    expr4 = B(i1)*B(-i1) + 2*E**2 - 2*px**2 - 2*py**2 - 2*pz**2 - A(i0)*A(-i0)
    assert expr4.args == (2*E**2, -2*px**2, -2*py**2, -2*pz**2, B(i1)*B(-i1), -A(i0)*A(-i0))


def test_mul():
    from sympy.abc import x
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    t = TensMul.from_data(S.One, [], [], [])
    assert str(t) == '1'
    A, B = tensor_heads('A B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = (1 + x)*A(a, b)
    assert str(t) == '(x + 1)*A(a, b)'
    assert t.index_types == [Lorentz, Lorentz]
    assert t.rank == 2
    assert t.dum == []
    assert t.coeff == 1 + x
    assert sorted(t.free) == [(a, 0), (b, 1)]
    assert t.components == [A]

    ts = A(a, b)
    assert str(ts) == 'A(a, b)'
    assert ts.index_types == [Lorentz, Lorentz]
    assert ts.rank == 2
    assert ts.dum == []
    assert ts.coeff == 1
    assert sorted(ts.free) == [(a, 0), (b, 1)]
    assert ts.components == [A]

    t = A(-b, a)*B(-a, c)*A(-c, d)
    t1 = tensor_mul(*t.split())
    assert t == t1
    assert tensor_mul(*[]) == TensMul.from_data(S.One, [], [], [])

    t = TensMul.from_data(1, [], [], [])
    C = TensorHead('C', [])
    assert str(C()) == 'C'
    assert str(t) == '1'
    assert t == 1
    raises(ValueError, lambda: A(a, b)*A(a, c))

def test_substitute_indices():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    i, j, k, l, m, n, p, q = tensor_indices('i,j,k,l,m,n,p,q', Lorentz)
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))

    p = TensorHead('p', [Lorentz])
    t = p(i)
    t1 = t.substitute_indices((j, k))
    assert t1 == t
    t1 = t.substitute_indices((i, j))
    assert t1 == p(j)
    t1 = t.substitute_indices((i, -j))
    assert t1 == p(-j)
    t1 = t.substitute_indices((-i, j))
    assert t1 == p(-j)
    t1 = t.substitute_indices((-i, -j))
    assert t1 == p(j)
    t = A(m, n)
    t1 = t.substitute_indices((m, i), (n, -i))
    assert t1 == A(n, -n)
    t1 = substitute_indices(t, (m, i), (n, -i))
    assert t1 == A(n, -n)

    t = A(i, k)*B(-k, -j)
    t1 = t.substitute_indices((i, j), (j, k))
    t1a = A(j, l)*B(-l, -k)
    assert t1 == t1a
    t1 = substitute_indices(t, (i, j), (j, k))
    assert t1 == t1a

    t = A(i, j) + B(i, j)
    t1 = t.substitute_indices((j, -i))
    t1a = A(i, -i) + B(i, -i)
    assert t1 == t1a
    t1 = substitute_indices(t, (j, -i))
    assert t1 == t1a

def test_riemann_cyclic_replace():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    m0, m1, m2, m3 = tensor_indices('m:4', Lorentz)
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    t = R(m0, m2, m1, m3)
    t1 = riemann_cyclic_replace(t)
    t1a = Rational(-1, 3)*R(m0, m3, m2, m1) + Rational(1, 3)*R(m0, m1, m2, m3) + Rational(2, 3)*R(m0, m2, m1, m3)
    assert t1 == t1a

def test_riemann_cyclic():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    i, j, k, l, m, n, p, q = tensor_indices('i,j,k,l,m,n,p,q', Lorentz)
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    t = R(i,j,k,l) + R(i,l,j,k) + R(i,k,l,j) - \
        R(i,j,l,k) - R(i,l,k,j) - R(i,k,j,l)
    t2 = t*R(-i,-j,-k,-l)
    t3 = riemann_cyclic(t2)
    assert t3 == 0
    t = R(i,j,k,l)*(R(-i,-j,-k,-l) - 2*R(-i,-k,-j,-l))
    t1 = riemann_cyclic(t)
    assert t1 == 0
    t = R(i,j,k,l)
    t1 = riemann_cyclic(t)
    assert t1 == Rational(-1, 3)*R(i, l, j, k) + Rational(1, 3)*R(i, k, j, l) + Rational(2, 3)*R(i, j, k, l)

    t = R(i,j,k,l)*R(-k,-l,m,n)*(R(-m,-n,-i,-j) + 2*R(-m,-j,-n,-i))
    t1 = riemann_cyclic(t)
    assert t1 == 0

@XFAIL
def test_div():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    m0, m1, m2, m3 = tensor_indices('m0:4', Lorentz)
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    t = R(m0,m1,-m1,m3)
    t1 = t/S(4)
    assert str(t1) == '(1/4)*R(m0, L_0, -L_0, m3)'
    t = t.canon_bp()
    assert not t1._is_canon_bp
    t1 = t*4
    assert t1._is_canon_bp
    t1 = t1/4
    assert t1._is_canon_bp

def test_contract_metric1():
    D = Symbol('D')
    Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    g = Lorentz.metric
    p = TensorHead('p', [Lorentz])
    t = g(a, b)*p(-b)
    t1 = t.contract_metric(g)
    assert t1 == p(a)
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))

    # case with g with all free indices
    t1 = A(a,b)*B(-b,c)*g(d, e)
    t2 = t1.contract_metric(g)
    assert t1 == t2

    # case of g(d, -d)
    t1 = A(a,b)*B(-b,c)*g(-d, d)
    t2 = t1.contract_metric(g)
    assert t2 == D*A(a, d)*B(-d, c)

    # g with one free index
    t1 = A(a,b)*B(-b,-c)*g(c, d)
    t2 = t1.contract_metric(g)
    assert t2 == A(a, c)*B(-c, d)

    # g with both indices contracted with another tensor
    t1 = A(a,b)*B(-b,-c)*g(c, -a)
    t2 = t1.contract_metric(g)
    assert _is_equal(t2, A(a, b)*B(-b, -a))

    t1 = A(a,b)*B(-b,-c)*g(c, d)*g(-a, -d)
    t2 = t1.contract_metric(g)
    assert _is_equal(t2, A(a,b)*B(-b,-a))

    t1 = A(a,b)*g(-a,-b)
    t2 = t1.contract_metric(g)
    assert _is_equal(t2, A(a, -a))
    assert not t2.free
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b = tensor_indices('a,b', Lorentz)
    g = Lorentz.metric
    assert _is_equal(g(a, -a).contract_metric(g), Lorentz.dim) # no dim

def test_contract_metric2():
    D = Symbol('D')
    Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
    a, b, c, d, e, L_0 = tensor_indices('a,b,c,d,e,L_0', Lorentz)
    g = Lorentz.metric
    p, q = tensor_heads('p,q', [Lorentz])

    t1 = g(a,b)*p(c)*p(-c)
    t2 = 3*g(-a,-b)*q(c)*q(-c)
    t = t1*t2
    t = t.contract_metric(g)
    assert t == 3*D*p(a)*p(-a)*q(b)*q(-b)
    t1 = g(a,b)*p(c)*p(-c)
    t2 = 3*q(-a)*q(-b)
    t = t1*t2
    t = t.contract_metric(g)
    t = t.canon_bp()
    assert t == 3*p(a)*p(-a)*q(b)*q(-b)

    t1 = 2*g(a,b)*p(c)*p(-c)
    t2 = - 3*g(-a,-b)*q(c)*q(-c)
    t = t1*t2
    t = t.contract_metric(g)
    t = 6*g(a,b)*g(-a,-b)*p(c)*p(-c)*q(d)*q(-d)
    t = t.contract_metric(g)

    t1 = 2*g(a,b)*p(c)*p(-c)
    t2 = q(-a)*q(-b) + 3*g(-a,-b)*q(c)*q(-c)
    t = t1*t2
    t = t.contract_metric(g)
    assert t == (2 + 6*D)*p(a)*p(-a)*q(b)*q(-b)

    t1 = p(a)*p(b) + p(a)*q(b) + 2*g(a,b)*p(c)*p(-c)
    t2 = q(-a)*q(-b) - g(-a,-b)*q(c)*q(-c)
    t = t1*t2
    t = t.contract_metric(g)
    t1 = (1 - 2*D)*p(a)*p(-a)*q(b)*q(-b) + p(a)*q(-a)*p(b)*q(-b)
    assert canon_bp(t - t1) == 0

    t = g(a,b)*g(c,d)*g(-b,-c)
    t1 = t.contract_metric(g)
    assert t1 == g(a, d)

    t1 = g(a,b)*g(c,d) + g(a,c)*g(b,d) + g(a,d)*g(b,c)
    t2 = t1.substitute_indices((a,-a),(b,-b),(c,-c),(d,-d))
    t = t1*t2
    t = t.contract_metric(g)
    assert t.equals(3*D**2 + 6*D)

    t = 2*p(a)*g(b,-b)
    t1 = t.contract_metric(g)
    assert t1.equals(2*D*p(a))

    t = 2*p(a)*g(b,-a)
    t1 = t.contract_metric(g)
    assert t1 == 2*p(b)

    M = Symbol('M')
    t = (p(a)*p(b) + g(a, b)*M**2)*g(-a, -b) - D*M**2
    t1 = t.contract_metric(g)
    assert t1 == p(a)*p(-a)

    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(a, b)*p(L_0)*g(-a, -b)
    t1 = t.contract_metric(g)
    assert str(t1) == 'A(L_1, -L_1)*p(L_0)' or str(t1) == 'A(-L_1, L_1)*p(L_0)'

def test_metric_contract3():
    D = Symbol('D')
    Spinor = TensorIndexType('Spinor', dim=D, metric_symmetry=-1, dummy_name='S')
    a0, a1, a2, a3, a4 = tensor_indices('a0:5', Spinor)
    C = Spinor.metric
    chi, psi = tensor_heads('chi,psi', [Spinor], TensorSymmetry.no_symmetry(1), 1)
    B = TensorHead('B', [Spinor]*2, TensorSymmetry.no_symmetry(2))

    t = C(a0,-a0)
    t1 = t.contract_metric(C)
    assert t1.equals(-D)

    t = C(-a0,a0)
    t1 = t.contract_metric(C)
    assert t1.equals(D)

    t = C(a0,a1)*C(-a0,-a1)
    t1 = t.contract_metric(C)
    assert t1.equals(D)

    t = C(a1,a0)*C(-a0,-a1)
    t1 = t.contract_metric(C)
    assert t1.equals(-D)

    t = C(-a0,a1)*C(a0,-a1)
    t1 = t.contract_metric(C)
    assert t1.equals(-D)

    t = C(a1,-a0)*C(a0,-a1)
    t1 = t.contract_metric(C)
    assert t1.equals(D)

    t = C(a0,a1)*B(-a1,-a0)
    t1 = t.contract_metric(C)
    t1 = t1.canon_bp()
    assert _is_equal(t1, B(a0,-a0))

    t = C(a1,a0)*B(-a1,-a0)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -B(a0,-a0))

    t = C(a0,-a1)*B(a1,-a0)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -B(a0,-a0))

    t = C(-a0,a1)*B(-a1,a0)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -B(a0,-a0))

    t = C(-a0,-a1)*B(a1,a0)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, B(a0,-a0))

    t = C(-a1, a0)*B(a1,-a0)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, B(a0,-a0))

    t = C(a0,a1)*psi(-a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, psi(a0))

    t = C(a1,a0)*psi(-a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -psi(a0))

    t = C(a0,a1)*chi(-a0)*psi(-a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -chi(a1)*psi(-a1))

    t = C(a1,a0)*chi(-a0)*psi(-a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, chi(a1)*psi(-a1))

    t = C(-a1,a0)*chi(-a0)*psi(a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, chi(-a1)*psi(a1))

    t = C(a0,-a1)*chi(-a0)*psi(a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -chi(-a1)*psi(a1))

    t = C(-a0,-a1)*chi(a0)*psi(a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, chi(-a1)*psi(a1))

    t = C(-a1,-a0)*chi(a0)*psi(a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -chi(-a1)*psi(a1))

    t = C(-a1,-a0)*B(a0,a2)*psi(a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -B(-a1,a2)*psi(a1))

    t = C(a1,a0)*B(-a2,-a0)*psi(-a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, B(-a2,a1)*psi(-a1))


def test_contract_metric4():
    R3 = TensorIndexType('R3', dim=3)
    p, q, r = tensor_indices("p q r", R3)
    delta = R3.delta
    eps = R3.epsilon
    K = TensorHead("K", [R3])

    #Check whether contract_metric chokes on an expandable expression which becomes zero on canonicalization (issue #24354)
    expr = eps(p,q,r)*( K(-p)*K(-q) + delta(-p,-q) )
    assert expr.contract_metric(delta) == 0


def test_contract_metric5():
    R3 = TensorIndexType('R3', dim=3)
    p, q, r = tensor_indices("p q r", R3)
    delta = R3.delta
    K = TensorHead("K", [R3])

    F = Function("F")
    x = Symbol("x")

    #Check if contract_metric gets into an infinite loop when given a TensMul whose coeff is an Add
    expr = (2+F(x))*K(-p)*K(-q)
    assert expr.contract_metric(delta) == expr


def test_epsilon():
    Lorentz = TensorIndexType('Lorentz', dim=4, dummy_name='L')
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    epsilon = Lorentz.epsilon
    p, q, r, s = tensor_heads('p,q,r,s', [Lorentz])

    t = epsilon(b,a,c,d)
    t1 = t.canon_bp()
    assert t1 == -epsilon(a,b,c,d)

    t = epsilon(c,b,d,a)
    t1 = t.canon_bp()
    assert t1 == epsilon(a,b,c,d)

    t = epsilon(c,a,d,b)
    t1 = t.canon_bp()
    assert t1 == -epsilon(a,b,c,d)

    t = epsilon(a,b,c,d)*p(-a)*q(-b)
    t1 = t.canon_bp()
    assert t1 == epsilon(c,d,a,b)*p(-a)*q(-b)

    t = epsilon(c,b,d,a)*p(-a)*q(-b)
    t1 = t.canon_bp()
    assert t1 == epsilon(c,d,a,b)*p(-a)*q(-b)

    t = epsilon(c,a,d,b)*p(-a)*q(-b)
    t1 = t.canon_bp()
    assert t1 == -epsilon(c,d,a,b)*p(-a)*q(-b)

    t = epsilon(c,a,d,b)*p(-a)*p(-b)
    t1 = t.canon_bp()
    assert t1 == 0

    t = epsilon(c,a,d,b)*p(-a)*q(-b) + epsilon(a,b,c,d)*p(-b)*q(-a)
    t1 = t.canon_bp()
    assert t1 == -2*epsilon(c,d,a,b)*p(-a)*q(-b)

    # Test that epsilon can be create with a SymPy integer:
    Lorentz = TensorIndexType('Lorentz', dim=Integer(4), dummy_name='L')
    epsilon = Lorentz.epsilon
    assert isinstance(epsilon, TensorHead)

def test_contract_delta1():
    # see Group Theory by Cvitanovic page 9
    n = Symbol('n')
    Color = TensorIndexType('Color', dim=n, dummy_name='C')
    a, b, c, d, e, f = tensor_indices('a,b,c,d,e,f', Color)
    delta = Color.delta

    def idn(a, b, d, c):
        assert a.is_up and d.is_up
        assert not (b.is_up or c.is_up)
        return delta(a,c)*delta(d,b)

    def T(a, b, d, c):
        assert a.is_up and d.is_up
        assert not (b.is_up or c.is_up)
        return delta(a,b)*delta(d,c)

    def P1(a, b, c, d):
        return idn(a,b,c,d) - 1/n*T(a,b,c,d)

    def P2(a, b, c, d):
        return 1/n*T(a,b,c,d)

    t = P1(a, -b, e, -f)*P1(f, -e, d, -c)
    t1 = t.contract_delta(delta)
    assert canon_bp(t1 - P1(a, -b, d, -c)) == 0

    t = P2(a, -b, e, -f)*P2(f, -e, d, -c)
    t1 = t.contract_delta(delta)
    assert t1 == P2(a, -b, d, -c)

    t = P1(a, -b, e, -f)*P2(f, -e, d, -c)
    t1 = t.contract_delta(delta)
    assert t1 == 0

    t = P1(a, -b, b, -a)
    t1 = t.contract_delta(delta)
    assert t1.equals(n**2 - 1)

def test_contract_delta2():
    R3 = TensorIndexType('R3', dim=3)
    p, q = tensor_indices("p q", R3)
    delta = R3.delta
    K = TensorHead("K", [R3])

    #Check if TensAdd.contract_delta can handle the case when the TensAdd has non-TensExpr args.
    expr = 1 + K(p)*K(q)*delta(-p,-q)
    assert expr.contract_delta(delta) == 1 + K(p)*K(-p)

def test_fun():
    with warns_deprecated_sympy():
        D = Symbol('D')
        Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
        a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
        g = Lorentz.metric

        p, q = tensor_heads('p q', [Lorentz])
        t = q(c)*p(a)*q(b) + g(a,b)*g(c,d)*q(-d)
        assert t(a,b,c) == t
        assert canon_bp(t - t(b,a,c) - q(c)*p(a)*q(b) + q(c)*p(b)*q(a)) == 0
        assert t(b,c,d) == q(d)*p(b)*q(c) + g(b,c)*g(d,e)*q(-e)
        t1 = t.substitute_indices((a,b),(b,a))
        assert canon_bp(t1 - q(c)*p(b)*q(a) - g(a,b)*g(c,d)*q(-d)) == 0

        # check that g_{a b; c} = 0
        # example taken from  L. Brewin
        # "A brief introduction to Cadabra" arxiv:0903.2085
        # dg_{a b c} = \partial_{a} g_{b c} is symmetric in b, c
        dg = TensorHead('dg', [Lorentz]*3, TensorSymmetry.direct_product(1, 2))
        # gamma^a_{b c} is the Christoffel symbol
        gamma = S.Half*g(a,d)*(dg(-b,-d,-c) + dg(-c,-b,-d) - dg(-d,-b,-c))
        # t = g_{a b; c}
        t = dg(-c,-a,-b) - g(-a,-d)*gamma(d,-b,-c) - g(-b,-d)*gamma(d,-a,-c)
        t = t.contract_metric(g)
        assert t == 0
        t = q(c)*p(a)*q(b)
        assert t(b,c,d) == q(d)*p(b)*q(c)

def test_TensorManager():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    LorentzH = TensorIndexType('LorentzH', dummy_name='LH')
    i, j = tensor_indices('i,j', Lorentz)
    ih, jh = tensor_indices('ih,jh', LorentzH)
    p, q = tensor_heads('p q', [Lorentz])
    ph, qh = tensor_heads('ph qh', [LorentzH])

    Gsymbol = Symbol('Gsymbol')
    GHsymbol = Symbol('GHsymbol')
    TensorManager.set_comm(Gsymbol, GHsymbol, 0)
    G = TensorHead('G', [Lorentz], TensorSymmetry.no_symmetry(1), Gsymbol)
    assert TensorManager._comm_i2symbol[G.comm] == Gsymbol
    GH = TensorHead('GH', [LorentzH], TensorSymmetry.no_symmetry(1), GHsymbol)
    ps = G(i)*p(-i)
    psh = GH(ih)*ph(-ih)
    t = ps + psh
    t1 = t*t
    assert canon_bp(t1 - ps*ps - 2*ps*psh - psh*psh) == 0
    qs = G(i)*q(-i)
    qsh = GH(ih)*qh(-ih)
    assert _is_equal(ps*qsh, qsh*ps)
    assert not _is_equal(ps*qs, qs*ps)
    n = TensorManager.comm_symbols2i(Gsymbol)
    assert TensorManager.comm_i2symbol(n) == Gsymbol

    assert GHsymbol in TensorManager._comm_symbols2i
    raises(ValueError, lambda: TensorManager.set_comm(GHsymbol, 1, 2))
    TensorManager.set_comms((Gsymbol,GHsymbol,0),(Gsymbol,1,1))
    assert TensorManager.get_comm(n, 1) == TensorManager.get_comm(1, n) == 1
    TensorManager.clear()
    assert TensorManager.comm == [{0:0, 1:0, 2:0}, {0:0, 1:1, 2:None}, {0:0, 1:None}]
    assert GHsymbol not in TensorManager._comm_symbols2i
    nh = TensorManager.comm_symbols2i(GHsymbol)
    assert TensorManager.comm_i2symbol(nh) == GHsymbol
    assert GHsymbol in TensorManager._comm_symbols2i


def test_hash():
    D = Symbol('D')
    Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    g = Lorentz.metric

    p, q = tensor_heads('p q', [Lorentz])
    p_type = p.args[1]
    t1 = p(a)*q(b)
    t2 = p(a)*p(b)
    assert hash(t1) != hash(t2)
    t3 = p(a)*p(b) + g(a,b)
    t4 = p(a)*p(b) - g(a,b)
    assert hash(t3) != hash(t4)

    assert a.func(*a.args) == a
    assert Lorentz.func(*Lorentz.args) == Lorentz
    assert g.func(*g.args) == g
    assert p.func(*p.args) == p
    assert p_type.func(*p_type.args) == p_type
    assert p(a).func(*(p(a)).args) == p(a)
    assert t1.func(*t1.args) == t1
    assert t2.func(*t2.args) == t2
    assert t3.func(*t3.args) == t3
    assert t4.func(*t4.args) == t4

    assert hash(a.func(*a.args)) == hash(a)
    assert hash(Lorentz.func(*Lorentz.args)) == hash(Lorentz)
    assert hash(g.func(*g.args)) == hash(g)
    assert hash(p.func(*p.args)) == hash(p)
    assert hash(p_type.func(*p_type.args)) == hash(p_type)
    assert hash(p(a).func(*(p(a)).args)) == hash(p(a))
    assert hash(t1.func(*t1.args)) == hash(t1)
    assert hash(t2.func(*t2.args)) == hash(t2)
    assert hash(t3.func(*t3.args)) == hash(t3)
    assert hash(t4.func(*t4.args)) == hash(t4)

    def check_all(obj):
        return all(isinstance(_, Basic) for _ in obj.args)

    assert check_all(a)
    assert check_all(Lorentz)
    assert check_all(g)
    assert check_all(p)
    assert check_all(p_type)
    assert check_all(p(a))
    assert check_all(t1)
    assert check_all(t2)
    assert check_all(t3)
    assert check_all(t4)

    tsymmetry = TensorSymmetry.direct_product(-2, 1, 3)

    assert tsymmetry.func(*tsymmetry.args) == tsymmetry
    assert hash(tsymmetry.func(*tsymmetry.args)) == hash(tsymmetry)
    assert check_all(tsymmetry)


### TEST VALUED TENSORS ###


def _get_valued_base_test_variables():
    minkowski = Matrix((
        (1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 0, -1),
    ))
    Lorentz = TensorIndexType('Lorentz', dim=4)
    Lorentz.data = minkowski

    i0, i1, i2, i3, i4 = tensor_indices('i0:5', Lorentz)

    E, px, py, pz = symbols('E px py pz')
    A = TensorHead('A', [Lorentz])
    A.data = [E, px, py, pz]
    B = TensorHead('B', [Lorentz], TensorSymmetry.no_symmetry(1), 'Gcomm')
    B.data = range(4)
    AB = TensorHead("AB", [Lorentz]*2)
    AB.data = minkowski

    ba_matrix = Matrix((
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 0, -1, -2),
        (-3, -4, -5, -6),
    ))

    BA = TensorHead("BA", [Lorentz]*2)
    BA.data = ba_matrix

    # Let's test the diagonal metric, with inverted Minkowski metric:
    LorentzD = TensorIndexType('LorentzD')
    LorentzD.data = [-1, 1, 1, 1]
    mu0, mu1, mu2 = tensor_indices('mu0:3', LorentzD)
    C = TensorHead('C', [LorentzD])
    C.data = [E, px, py, pz]

    ### non-diagonal metric ###
    ndm_matrix = (
        (1, 1, 0,),
        (1, 0, 1),
        (0, 1, 0,),
    )
    ndm = TensorIndexType("ndm")
    ndm.data = ndm_matrix
    n0, n1, n2 = tensor_indices('n0:3', ndm)
    NA = TensorHead('NA', [ndm])
    NA.data = range(10, 13)
    NB = TensorHead('NB', [ndm]*2)
    NB.data = [[i+j for j in range(10, 13)] for i in range(10, 13)]
    NC = TensorHead('NC', [ndm]*3)
    NC.data = [[[i+j+k for k in range(4, 7)] for j in range(1, 4)] for i in range(2, 5)]

    return (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
            n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4)


def test_valued_tensor_iter():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        list_BA = [Array([1, 2, 3, 4]), Array([5, 6, 7, 8]), Array([9, 0, -1, -2]), Array([-3, -4, -5, -6])]
        # iteration on VTensorHead
        assert list(A) == [E, px, py, pz]
        assert list(ba_matrix) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2, -3, -4, -5, -6]
        assert list(BA) == list_BA

        # iteration on VTensMul
        assert list(A(i1)) == [E, px, py, pz]
        assert list(BA(i1, i2)) == list_BA
        assert list(3 * BA(i1, i2)) == [3 * i for i in list_BA]
        assert list(-5 * BA(i1, i2)) == [-5 * i for i in list_BA]

        # iteration on VTensAdd
        # A(i1) + A(i1)
        assert list(A(i1) + A(i1)) == [2*E, 2*px, 2*py, 2*pz]
        assert BA(i1, i2) - BA(i1, i2) == 0
        assert list(BA(i1, i2) - 2 * BA(i1, i2)) == [-i for i in list_BA]


def test_valued_tensor_covariant_contravariant_elements():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        assert A(-i0)[0] == A(i0)[0]
        assert A(-i0)[1] == -A(i0)[1]

        assert AB(i0, i1)[1, 1] == -1
        assert AB(i0, -i1)[1, 1] == 1
        assert AB(-i0, -i1)[1, 1] == -1
        assert AB(-i0, i1)[1, 1] == 1


def test_valued_tensor_get_matrix():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        matab = AB(i0, i1).get_matrix()
        assert matab == Matrix([
                                [1,  0,  0,  0],
                                [0, -1,  0,  0],
                                [0,  0, -1,  0],
                                [0,  0,  0, -1],
                                ])
        # when alternating contravariant/covariant with [1, -1, -1, -1] metric
        # it becomes the identity matrix:
        assert AB(i0, -i1).get_matrix() == eye(4)

        # covariant and contravariant forms:
        assert A(i0).get_matrix() == Matrix([E, px, py, pz])
        assert A(-i0).get_matrix() == Matrix([E, -px, -py, -pz])

def test_valued_tensor_contraction():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        assert (A(i0) * A(-i0)).data == E ** 2 - px ** 2 - py ** 2 - pz ** 2
        assert (A(i0) * A(-i0)).data == A ** 2
        assert (A(i0) * A(-i0)).data == A(i0) ** 2
        assert (A(i0) * B(-i0)).data == -px - 2 * py - 3 * pz

        for i in range(4):
            for j in range(4):
                assert (A(i0) * B(-i1))[i, j] == [E, px, py, pz][i] * [0, -1, -2, -3][j]

        # test contraction on the alternative Minkowski metric: [-1, 1, 1, 1]
        assert (C(mu0) * C(-mu0)).data == -E ** 2 + px ** 2 + py ** 2 + pz ** 2

        contrexp = A(i0) * AB(i1, -i0)
        assert A(i0).rank == 1
        assert AB(i1, -i0).rank == 2
        assert contrexp.rank == 1
        for i in range(4):
            assert contrexp[i] == [E, px, py, pz][i]

def test_valued_tensor_self_contraction():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        assert AB(i0, -i0).data == 4
        assert BA(i0, -i0).data == 2


def test_valued_tensor_pow():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        assert C**2 == -E**2 + px**2 + py**2 + pz**2
        assert C**1 == sqrt(-E**2 + px**2 + py**2 + pz**2)
        assert C(mu0)**2 == C**2
        assert C(mu0)**1 == C**1


def test_valued_tensor_expressions():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        x1, x2, x3 = symbols('x1:4')

        # test coefficient in contraction:
        rank2coeff = x1 * A(i3) * B(i2)
        assert rank2coeff[1, 1] == x1 * px
        assert rank2coeff[3, 3] == 3 * pz * x1
        coeff_expr = ((x1 * A(i4)) * (B(-i4) / x2)).data

        assert coeff_expr.expand() == -px*x1/x2 - 2*py*x1/x2 - 3*pz*x1/x2

        add_expr = A(i0) + B(i0)

        assert add_expr[0] == E
        assert add_expr[1] == px + 1
        assert add_expr[2] == py + 2
        assert add_expr[3] == pz + 3

        sub_expr = A(i0) - B(i0)

        assert sub_expr[0] == E
        assert sub_expr[1] == px - 1
        assert sub_expr[2] == py - 2
        assert sub_expr[3] == pz - 3

        assert (add_expr * B(-i0)).data == -px - 2*py - 3*pz - 14

        expr1 = x1*A(i0) + x2*B(i0)
        expr2 = expr1 * B(i1) * (-4)
        expr3 = expr2 + 3*x3*AB(i0, i1)
        expr4 = expr3 / 2
        assert expr4 * 2 == expr3
        expr5 = (expr4 * BA(-i1, -i0))

        assert expr5.data.expand() == 28*E*x1 + 12*px*x1 + 20*py*x1 + 28*pz*x1 + 136*x2 + 3*x3


def test_valued_tensor_add_scalar():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # one scalar summand after the contracted tensor
        expr1 = A(i0)*A(-i0) - (E**2 - px**2 - py**2 - pz**2)
        assert expr1.data == 0

        # multiple scalar summands in front of the contracted tensor
        expr2 = E**2 - px**2 - py**2 - pz**2 - A(i0)*A(-i0)
        assert expr2.data == 0

        # multiple scalar summands after the contracted tensor
        expr3 =  A(i0)*A(-i0) - E**2 + px**2 + py**2 + pz**2
        assert expr3.data == 0

        # multiple scalar summands and multiple tensors
        expr4 = C(mu0)*C(-mu0) + 2*E**2 - 2*px**2 - 2*py**2 - 2*pz**2 - A(i0)*A(-i0)
        assert expr4.data == 0

def test_noncommuting_components():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        euclid = TensorIndexType('Euclidean')
        euclid.data = [1, 1]
        i1, i2, i3 = tensor_indices('i1:4', euclid)

        a, b, c, d = symbols('a b c d', commutative=False)
        V1 = TensorHead('V1', [euclid]*2)
        V1.data = [[a, b], (c, d)]
        V2 = TensorHead('V2', [euclid]*2)
        V2.data = [[a, c], [b, d]]

        vtp = V1(i1, i2) * V2(-i2, -i1)

        assert vtp.data == a**2 + b**2 + c**2 + d**2
        assert vtp.data != a**2 + 2*b*c + d**2

        vtp2 = V1(i1, i2)*V1(-i2, -i1)

        assert vtp2.data == a**2 + b*c + c*b + d**2
        assert vtp2.data != a**2 + 2*b*c + d**2

        Vc = (b * V1(i1, -i1)).data
        assert Vc.expand() == b * a + b * d


def test_valued_non_diagonal_metric():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        mmatrix = Matrix(ndm_matrix)
        assert (NA(n0)*NA(-n0)).data == (NA(n0).get_matrix().T * mmatrix * NA(n0).get_matrix())[0, 0]


def test_valued_assign_numpy_ndarray():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # this is needed to make sure that a numpy.ndarray can be assigned to a
        # tensor.
        arr = [E+1, px-1, py, pz]
        A.data = Array(arr)
        for i in range(4):
                assert A(i0).data[i] == arr[i]

        qx, qy, qz = symbols('qx qy qz')
        A(-i0).data = Array([E, qx, qy, qz])
        for i in range(4):
            assert A(i0).data[i] == [E, -qx, -qy, -qz][i]
            assert A.data[i] == [E, -qx, -qy, -qz][i]

        # test on multi-indexed tensors.
        random_4x4_data = [[(i**3-3*i**2)%(j+7) for i in range(4)] for j in range(4)]
        AB(-i0, -i1).data = random_4x4_data

        for i in range(4):
            for j in range(4):
                assert AB(i0, i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)*(-1 if j else 1)
                assert AB(-i0, i1).data[i, j] == random_4x4_data[i][j]*(-1 if j else 1)
                assert AB(i0, -i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)
                assert AB(-i0, -i1).data[i, j] == random_4x4_data[i][j]

        AB(-i0, i1).data = random_4x4_data
        for i in range(4):
            for j in range(4):
                assert AB(i0, i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)
                assert AB(-i0, i1).data[i, j] == random_4x4_data[i][j]
                assert AB(i0, -i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)*(-1 if j else 1)
                assert AB(-i0, -i1).data[i, j] == random_4x4_data[i][j]*(-1 if j else 1)


def test_valued_metric_inverse():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # let's assign some fancy matrix, just to verify it:
        # (this has no physical sense, it's just testing sympy);
        # it is symmetrical:
        md = [[2, 2, 2, 1], [2, 3, 1, 0], [2, 1, 2, 3], [1, 0, 3, 2]]
        Lorentz.data = md
        m = Matrix(md)
        metric = Lorentz.metric
        minv = m.inv()

        meye = eye(4)

        # the Kronecker Delta:
        KD = Lorentz.get_kronecker_delta()

        for i in range(4):
            for j in range(4):
                assert metric(i0, i1).data[i, j] == m[i, j]
                assert metric(-i0, -i1).data[i, j] == minv[i, j]
                assert metric(i0, -i1).data[i, j] == meye[i, j]
                assert metric(-i0, i1).data[i, j] == meye[i, j]
                assert metric(i0, i1)[i, j] == m[i, j]
                assert metric(-i0, -i1)[i, j] == minv[i, j]
                assert metric(i0, -i1)[i, j] == meye[i, j]
                assert metric(-i0, i1)[i, j] == meye[i, j]

                assert KD(i0, -i1)[i, j] == meye[i, j]


def test_valued_canon_bp_swapaxes():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        e1 = A(i1)*A(i0)
        e2 = e1.canon_bp()
        assert e2 == A(i0)*A(i1)
        for i in range(4):
            for j in range(4):
                assert e1[i, j] == e2[j, i]
        o1 = B(i2)*A(i1)*B(i0)
        o2 = o1.canon_bp()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    assert o1[i, j, k] == o2[j, i, k]


def test_valued_components_with_wrong_symmetry():
    with warns_deprecated_sympy():
        IT = TensorIndexType('IT', dim=3)
        i0, i1, i2, i3 = tensor_indices('i0:4', IT)
        IT.data = [1, 1, 1]
        A_nosym = TensorHead('A', [IT]*2)
        A_sym = TensorHead('A', [IT]*2, TensorSymmetry.fully_symmetric(2))
        A_antisym = TensorHead('A', [IT]*2, TensorSymmetry.fully_symmetric(-2))

        mat_nosym = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        mat_sym = mat_nosym + mat_nosym.T
        mat_antisym = mat_nosym - mat_nosym.T

        A_nosym.data = mat_nosym
        A_nosym.data = mat_sym
        A_nosym.data = mat_antisym

        def assign(A, dat):
            A.data = dat

        A_sym.data = mat_sym
        raises(ValueError, lambda: assign(A_sym, mat_nosym))
        raises(ValueError, lambda: assign(A_sym, mat_antisym))

        A_antisym.data = mat_antisym
        raises(ValueError, lambda: assign(A_antisym, mat_sym))
        raises(ValueError, lambda: assign(A_antisym, mat_nosym))

        A_sym.data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        A_antisym.data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

def test_issue_10972_TensMul_data():
    with warns_deprecated_sympy():
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='i', dim=2)
        Lorentz.data = [-1, 1]

        mu, nu, alpha, beta = tensor_indices('\\mu, \\nu, \\alpha, \\beta',
                                             Lorentz)

        u = TensorHead('u', [Lorentz])
        u.data = [1, 0]

        F = TensorHead('F', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        F.data = [[0, 1],
                  [-1, 0]]

        mul_1 = F(mu, alpha) * u(-alpha) * F(nu, beta) * u(-beta)
        assert (mul_1.data == Array([[0, 0], [0, 1]]))

        mul_2 = F(mu, alpha) * F(nu, beta) * u(-alpha) * u(-beta)
        assert (mul_2.data == mul_1.data)

        assert ((mul_1 + mul_1).data == 2 * mul_1.data)


def test_TensMul_data():
    with warns_deprecated_sympy():
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='L', dim=4)
        Lorentz.data = [-1, 1, 1, 1]

        mu, nu, alpha, beta = tensor_indices('\\mu, \\nu, \\alpha, \\beta',
                                             Lorentz)

        u = TensorHead('u', [Lorentz])
        u.data = [1, 0, 0, 0]

        F = TensorHead('F', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        Ex, Ey, Ez, Bx, By, Bz = symbols('E_x E_y E_z B_x B_y B_z')
        F.data = [
            [0, Ex, Ey, Ez],
            [-Ex, 0, Bz, -By],
            [-Ey, -Bz, 0, Bx],
            [-Ez, By, -Bx, 0]]

        E = F(mu, nu) * u(-nu)

        assert ((E(mu) * E(nu)).data ==
                Array([[0, 0, 0, 0],
                             [0, Ex ** 2, Ex * Ey, Ex * Ez],
                             [0, Ex * Ey, Ey ** 2, Ey * Ez],
                             [0, Ex * Ez, Ey * Ez, Ez ** 2]])
                )

        assert ((E(mu) * E(nu)).canon_bp().data == (E(mu) * E(nu)).data)

        assert ((F(mu, alpha) * F(beta, nu) * u(-alpha) * u(-beta)).data ==
                - (E(mu) * E(nu)).data
                )
        assert ((F(alpha, mu) * F(beta, nu) * u(-alpha) * u(-beta)).data ==
                (E(mu) * E(nu)).data
                )

        g = TensorHead('g', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        g.data = Lorentz.data

        # tensor 'perp' is orthogonal to vector 'u'
        perp = u(mu) * u(nu) + g(mu, nu)

        mul_1 = u(-mu) * perp(mu, nu)
        assert (mul_1.data == Array([0, 0, 0, 0]))

        mul_2 = u(-mu) * perp(mu, alpha) * perp(nu, beta)
        assert (mul_2.data == Array.zeros(4, 4, 4))

        Fperp = perp(mu, alpha) * perp(nu, beta) * F(-alpha, -beta)
        assert (Fperp.data[0, :] == Array([0, 0, 0, 0]))
        assert (Fperp.data[:, 0] == Array([0, 0, 0, 0]))

        mul_3 = u(-mu) * Fperp(mu, nu)
        assert (mul_3.data == Array([0, 0, 0, 0]))

        # Test the deleter
        del g.data

def test_issue_11020_TensAdd_data():
    with warns_deprecated_sympy():
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='i', dim=2)
        Lorentz.data = [-1, 1]

        a, b, c, d = tensor_indices('a, b, c, d', Lorentz)
        i0, i1 = tensor_indices('i_0:2', Lorentz)

        # metric tensor
        g = TensorHead('g', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        g.data = Lorentz.data

        u = TensorHead('u', [Lorentz])
        u.data = [1, 0]

        add_1 = g(b, c) * g(d, i0) * u(-i0) - g(b, c) * u(d)
        assert (add_1.data == Array.zeros(2, 2, 2))
        # Now let us replace index `d` with `a`:
        add_2 = g(b, c) * g(a, i0) * u(-i0) - g(b, c) * u(a)
        assert (add_2.data == Array.zeros(2, 2, 2))

        # some more tests
        # perp is tensor orthogonal to u^\mu
        perp = u(a) * u(b) + g(a, b)
        mul_1 = u(-a) * perp(a, b)
        assert (mul_1.data == Array([0, 0]))

        mul_2 = u(-c) * perp(c, a) * perp(d, b)
        assert (mul_2.data == Array.zeros(2, 2, 2))


def test_index_iteration():
    L = TensorIndexType("Lorentz", dummy_name="L")
    i0, i1, i2, i3, i4 = tensor_indices('i0:5', L)
    L0 = tensor_indices('L_0', L)
    L1 = tensor_indices('L_1', L)
    A = TensorHead("A", [L, L])
    B = TensorHead("B", [L, L], TensorSymmetry.fully_symmetric(2))

    e1 = A(i0,i2)
    e2 = A(i0,-i0)
    e3 = A(i0,i1)*B(i2,i3)
    e4 = A(i0,i1)*B(i2,-i1)
    e5 = A(i0,i1)*B(-i0,-i1)
    e6 = e1 + e4

    assert list(e1._iterate_free_indices) == [(i0, (1, 0)), (i2, (1, 1))]
    assert list(e1._iterate_dummy_indices) == []
    assert list(e1._iterate_indices) == [(i0, (1, 0)), (i2, (1, 1))]

    assert list(e2._iterate_free_indices) == []
    assert list(e2._iterate_dummy_indices) == [(L0, (1, 0)), (-L0, (1, 1))]
    assert list(e2._iterate_indices) == [(L0, (1, 0)), (-L0, (1, 1))]

    assert list(e3._iterate_free_indices) == [(i0, (0, 1, 0)), (i1, (0, 1, 1)), (i2, (1, 1, 0)), (i3, (1, 1, 1))]
    assert list(e3._iterate_dummy_indices) == []
    assert list(e3._iterate_indices) == [(i0, (0, 1, 0)), (i1, (0, 1, 1)), (i2, (1, 1, 0)), (i3, (1, 1, 1))]

    assert list(e4._iterate_free_indices) == [(i0, (0, 1, 0)), (i2, (1, 1, 0))]
    assert list(e4._iterate_dummy_indices) == [(L0, (0, 1, 1)), (-L0, (1, 1, 1))]
    assert list(e4._iterate_indices) == [(i0, (0, 1, 0)), (L0, (0, 1, 1)), (i2, (1, 1, 0)), (-L0, (1, 1, 1))]

    assert list(e5._iterate_free_indices) == []
    assert list(e5._iterate_dummy_indices) == [(L0, (0, 1, 0)), (L1, (0, 1, 1)), (-L0, (1, 1, 0)), (-L1, (1, 1, 1))]
    assert list(e5._iterate_indices) == [(L0, (0, 1, 0)), (L1, (0, 1, 1)), (-L0, (1, 1, 0)), (-L1, (1, 1, 1))]

    assert list(e6._iterate_free_indices) == [(i0, (0, 0, 1, 0)), (i2, (0, 1, 1, 0)), (i0, (1, 1, 0)), (i2, (1, 1, 1))]
    assert list(e6._iterate_dummy_indices) == [(L0, (0, 0, 1, 1)), (-L0, (0, 1, 1, 1))]
    assert list(e6._iterate_indices) == [(i0, (0, 0, 1, 0)), (L0, (0, 0, 1, 1)), (i2, (0, 1, 1, 0)), (-L0, (0, 1, 1, 1)), (i0, (1, 1, 0)), (i2, (1, 1, 1))]

    assert e1.get_indices() == [i0, i2]
    assert e1.get_free_indices() == [i0, i2]
    assert e2.get_indices() == [L0, -L0]
    assert e2.get_free_indices() == []
    assert e3.get_indices() == [i0, i1, i2, i3]
    assert e3.get_free_indices() == [i0, i1, i2, i3]
    assert e4.get_indices() == [i0, L0, i2, -L0]
    assert e4.get_free_indices() == [i0, i2]
    assert e5.get_indices() == [L0, L1, -L0, -L1]
    assert e5.get_free_indices() == []


def test_tensor_expand():
    L = TensorIndexType("L")

    i, j, k = tensor_indices("i j k", L)
    L_0 = TensorIndex("L_0", L)

    A, B, C, D = tensor_heads("A B C D", [L])

    F = Function("F")
    x = Symbol("x")

    assert isinstance(Add(A(i), B(i)), TensAdd)
    assert isinstance(expand(A(i)+B(i)), TensAdd)

    expr = A(i)*(A(-i)+B(-i))
    assert expr.args == (A(L_0), A(-L_0) + B(-L_0))
    assert expr != A(i)*A(-i) + A(i)*B(-i)
    assert expr.expand() == A(i)*A(-i) + A(i)*B(-i)
    assert str(expr) == "A(L_0)*(A(-L_0) + B(-L_0))"

    expr = A(i)*A(j) + A(i)*B(j)
    assert str(expr) == "A(i)*A(j) + A(i)*B(j)"

    expr = A(-i)*(A(i)*A(j) + A(i)*B(j)*C(k)*C(-k))
    assert expr != A(-i)*A(i)*A(j) + A(-i)*A(i)*B(j)*C(k)*C(-k)
    assert expr.expand() == A(-i)*A(i)*A(j) + A(-i)*A(i)*B(j)*C(k)*C(-k)
    assert str(expr) == "A(-L_0)*(A(L_0)*A(j) + A(L_0)*B(j)*C(L_1)*C(-L_1))"
    assert str(expr.canon_bp()) == 'A(j)*A(L_0)*A(-L_0) + A(L_0)*A(-L_0)*B(j)*C(L_1)*C(-L_1)'

    expr = A(-i)*(2*A(i)*A(j) + A(i)*B(j))
    assert expr.expand() == 2*A(-i)*A(i)*A(j) + A(-i)*A(i)*B(j)

    expr = 2*A(i)*A(-i)
    assert expr.coeff == 2

    expr = A(i)*(B(j)*C(k) + C(j)*(A(k) + D(k)))
    assert str(expr) == "A(i)*(B(j)*C(k) + C(j)*(A(k) + D(k)))"
    assert str(expr.expand()) == "A(i)*B(j)*C(k) + A(i)*C(j)*A(k) + A(i)*C(j)*D(k)"

    assert isinstance(TensMul(3), TensMul)
    tm = TensMul(3).doit()
    assert tm == 3
    assert isinstance(tm, Integer)

    p1 = B(j)*B(-j) + B(j)*C(-j)
    p2 = C(-i)*p1
    p3 = A(i)*p2
    assert p3.expand() == A(i)*C(-i)*B(j)*B(-j) + A(i)*C(-i)*B(j)*C(-j)

    expr = A(i)*(B(-i) + C(-i)*(B(j)*B(-j) + B(j)*C(-j)))
    assert expr.expand() == A(i)*B(-i) + A(i)*C(-i)*B(j)*B(-j) + A(i)*C(-i)*B(j)*C(-j)

    expr = C(-i)*(B(j)*B(-j) + B(j)*C(-j))
    assert expr.expand() == C(-i)*B(j)*B(-j) + C(-i)*B(j)*C(-j)

    """
    Test whether expand correctly handles the case where the coeff of a TensMul
    is an add. We do not directly check expr_expand == 2*A(i) + F(x)*A(i) since
    __add__ currently consolidates the coefficients automatically
    """
    expr = (2 + F(x))*A(i)
    expr_expand = expr.expand()
    assert isinstance(expr_expand, TensAdd)
    assert expr_expand.args == (2*A(i), F(x)*A(i))

    expr = (2 + F(x))*A(i) + B(i)
    expr_expand = expr.expand()
    assert isinstance(expr_expand, TensAdd)
    assert expr_expand.args == (2*A(i), F(x)*A(i), B(i))


def test_tensor_alternative_construction():
    L = TensorIndexType("L")
    i0, i1, i2, i3 = tensor_indices('i0:4', L)
    A = TensorHead("A", [L])
    x, y = symbols("x y")

    assert A(i0) == A(Symbol("i0"))
    assert A(-i0) == A(-Symbol("i0"))
    raises(TypeError, lambda: A(x+y))
    raises(ValueError, lambda: A(2*x))


def test_tensor_replacement():
    L = TensorIndexType("L")
    L2 = TensorIndexType("L2", dim=2)
    i, j, k, l = tensor_indices("i j k l", L)
    A, B, C, D = tensor_heads("A B C D", [L])
    H = TensorHead("H", [L, L])
    K = TensorHead("K", [L]*4)

    expr = H(i, j)
    repl = {H(i,-j): [[1,2],[3,4]], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([i, j], Array([[1, -2], [3, -4]]))

    assert expr.replace_with_arrays(repl) == Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [i, -j]) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [Symbol("i"), -Symbol("j")]) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [-i, j]) == Array([[1, -2], [-3, 4]])
    assert expr.replace_with_arrays(repl, [-i, -j]) == Array([[1, 2], [-3, -4]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[1, 3], [-2, -4]])
    assert expr.replace_with_arrays(repl, [j, -i]) == Array([[1, -3], [-2, 4]])
    assert expr.replace_with_arrays(repl, [-j, i]) == Array([[1, 3], [2, 4]])
    assert expr.replace_with_arrays(repl, [-j, -i]) == Array([[1, -3], [2, -4]])
    # Test stability of optional parameter 'indices'
    assert expr.replace_with_arrays(repl) == Array([[1, -2], [3, -4]])

    expr = H(i,j)
    repl = {H(i,j): [[1,2],[3,4]], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([i, j], Array([[1, 2], [3, 4]]))

    assert expr.replace_with_arrays(repl) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [i, -j]) == Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [-i, j]) == Array([[1, 2], [-3, -4]])
    assert expr.replace_with_arrays(repl, [-i, -j]) == Array([[1, -2], [-3, 4]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[1, 3], [2, 4]])
    assert expr.replace_with_arrays(repl, [j, -i]) == Array([[1, -3], [2, -4]])
    assert expr.replace_with_arrays(repl, [-j, i]) == Array([[1, 3], [-2, -4]])
    assert expr.replace_with_arrays(repl, [-j, -i]) == Array([[1, -3], [-2, 4]])

    # Not the same indices:
    expr = H(i,k)
    repl = {H(i,j): [[1,2],[3,4]], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([i, k], Array([[1, 2], [3, 4]]))

    expr = A(i)*A(-i)
    repl = {A(i): [1,2], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([], -3)
    assert expr.replace_with_arrays(repl, []) == -3

    expr = K(i, j, -j, k)*A(-i)*A(-k)
    repl = {A(i): [1, 2], K(i,j,k,l): Array([1]*2**4).reshape(2,2,2,2), L: diag(1, -1)}
    assert expr._extract_data(repl)

    expr = H(j, k)
    repl = {H(i,j): [[1,2],[3,4]], L: diag(1, -1)}
    raises(ValueError, lambda: expr._extract_data(repl))

    expr = A(i)
    repl = {B(i): [1, 2]}
    raises(ValueError, lambda: expr._extract_data(repl))

    expr = A(i)
    repl = {A(i): [[1, 2], [3, 4]]}
    raises(ValueError, lambda: expr._extract_data(repl))

    # TensAdd:
    expr = A(k)*H(i, j) + B(k)*H(i, j)
    repl = {A(k): [1], B(k): [1], H(i, j): [[1, 2],[3,4]], L:diag(1,1)}
    assert expr._extract_data(repl) == ([k, i, j], Array([[[2, 4], [6, 8]]]))
    assert expr.replace_with_arrays(repl, [k, i, j]) == Array([[[2, 4], [6, 8]]])
    assert expr.replace_with_arrays(repl, [k, j, i]) == Array([[[2, 6], [4, 8]]])

    expr = A(k)*A(-k) + 100
    repl = {A(k): [2, 3], L: diag(1, 1)}
    assert expr.replace_with_arrays(repl, []) == 113

    ## Symmetrization:
    expr = H(i, j) + H(j, i)
    repl = {H(i, j): [[1, 2], [3, 4]]}
    assert expr._extract_data(repl) == ([i, j], Array([[2, 5], [5, 8]]))
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[2, 5], [5, 8]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[2, 5], [5, 8]])

    ## Anti-symmetrization:
    expr = H(i, j) - H(j, i)
    repl = {H(i, j): [[1, 2], [3, 4]]}
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[0, -1], [1, 0]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[0, 1], [-1, 0]])

    # Tensors with contractions in replacements:
    expr = K(i, j, k, -k)
    repl = {K(i, j, k, -k): [[1, 2], [3, 4]]}
    assert expr._extract_data(repl) == ([i, j], Array([[1, 2], [3, 4]]))

    expr = H(i, -i)
    repl = {H(i, -i): 42}
    assert expr._extract_data(repl) == ([], 42)

    expr = H(i, -i)
    repl = {
        H(-i, -j): Array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
        L: Array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
    }
    assert expr._extract_data(repl) == ([], 4)

    # Replace with array, raise exception if indices are not compatible:
    expr = A(i)*A(j)
    repl = {A(i): [1, 2]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [j]))

    # Raise exception if array dimension is not compatible:
    expr = A(i)
    repl = {A(i): [[1, 2]]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [i]))

    # TensorIndexType with dimension, wrong dimension in replacement array:
    u1, u2, u3 = tensor_indices("u1:4", L2)
    U = TensorHead("U", [L2])
    expr = U(u1)*U(-u2)
    repl = {U(u1): [[1]]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [u1, -u2]))


def test_rewrite_tensor_to_Indexed():
    L = TensorIndexType("L", dim=4)
    A = TensorHead("A", [L]*4)
    B = TensorHead("B", [L])

    i0, i1, i2, i3 = symbols("i0:4")
    L_0, L_1 = symbols("L_0:2")

    a1 = A(i0, i1, i2, i3)
    assert a1.rewrite(Indexed) == Indexed(Symbol("A"), i0, i1, i2, i3)

    a2 = A(i0, -i0, i2, i3)
    assert a2.rewrite(Indexed) == Sum(Indexed(Symbol("A"), L_0, L_0, i2, i3), (L_0, 0, 3))

    a3 = a2 + A(i2, i3, i0, -i0)
    assert a3.rewrite(Indexed) == \
        Sum(Indexed(Symbol("A"), L_0, L_0, i2, i3), (L_0, 0, 3)) +\
        Sum(Indexed(Symbol("A"), i2, i3, L_0, L_0), (L_0, 0, 3))

    b1 = B(-i0)*a1
    assert b1.rewrite(Indexed) == Sum(Indexed(Symbol("B"), L_0)*Indexed(Symbol("A"), L_0, i1, i2, i3), (L_0, 0, 3))

    b2 = B(-i3)*a2
    assert b2.rewrite(Indexed) == Sum(Indexed(Symbol("B"), L_1)*Indexed(Symbol("A"), L_0, L_0, i2, L_1), (L_0, 0, 3), (L_1, 0, 3))

def test_tensor_matching():
    """
    Test match and replace with the pattern being a WildTensor or a WildTensorIndex
    """
    R3 = TensorIndexType('R3', dim=3)
    p, q, r = tensor_indices("p q r", R3)
    a,b,c = symbols("a b c", cls = WildTensorIndex, tensor_index_type=R3, ignore_updown=True)
    g = WildTensorIndex("g", R3)
    delta = R3.delta
    eps = R3.epsilon
    K = TensorHead("K", [R3])
    V = TensorHead("V", [R3])
    A = TensorHead("A", [R3, R3])
    W = WildTensorHead('W', unordered_indices=True)
    U = WildTensorHead('U')

    assert a.matches(q) == {a:q}
    assert a.matches(-q) == {a:-q}
    assert g.matches(-q) == None
    assert g.matches(q) == {g:q}
    assert eps(p,-a,a).matches( eps(p,q,r) ) == None
    assert eps(p,-b,a).matches( eps(p,q,r) ) == {a: r, -b: q}
    assert eps(p,-q,r).replace(eps(a,b,c), 1) == 1
    assert W().matches( K(p)*V(q) ) == {W(): K(p)*V(q)}
    assert W(a).matches( K(p) ) == {a:p, W(a).head: _WildTensExpr(K(p))}
    assert W(a,p).matches( K(p)*V(q) ) == {a:q, W(a,p).head: _WildTensExpr(K(p)*V(q))}
    assert W(p,q).matches( K(p)*V(q) ) == {W(p,q).head: _WildTensExpr(K(p)*V(q))}
    assert W(p,q).matches( A(q,p) ) == {W(p,q).head: _WildTensExpr(A(q, p))}
    assert U(p,q).matches( A(q,p) ) == None
    assert ( K(q)*K(p) ).replace( W(q,p), 1) == 1

    #Some tests for matching without Wild
    assert delta(p,q).matches(delta(q,p)) == {}
    assert eps(p,q,r).matches(eps(q,p,r)) is None
    assert eps(p,q,r).matches(eps(q,r,p)) == {}

def test_TensAdd_matching():
    """
    Test match and replace with the pattern being a TensAdd
    """
    R3 = TensorIndexType('R3', dim=3)
    p, q = tensor_indices("p q", R3)
    K = TensorHead("K", [R3])
    V = TensorHead("V", [R3])
    W = WildTensorHead('W', unordered_indices=True)

    assert ( K(p)*K(q) + V(p)*V(q) ).matches( K(p)*K(q) + V(p)*V(q) ) == {}
    assert ( K(p)*K(q) + V(p)*V(q) ).matches( K(p)*K(q) + V(p)*V(q) + K(p)*V(q) + V(p)*K(q) ) is None
    assert ( K(p)*K(q) + V(p)*V(q) + K(p)*V(q) + K(q)*V(p) ).replace(
        W(p,q) + K(p)*K(q) + V(p)*V(q),
        W(p,q) + 3*K(p)*V(q)
        ).doit() == K(q)*V(p) + 4*K(p)*V(q)

def test_TensMul_matching():
    """
    Test match and replace with the pattern being a TensMul
    """
    R3 = TensorIndexType('R3', dim=3)
    p, q, r, s, t = tensor_indices("p q r s t", R3)
    wi = Wild("wi")
    a,b,c,d,e,f = symbols("a b c d e f", cls = WildTensorIndex, tensor_index_type=R3, ignore_updown=True)
    delta = R3.delta
    eps = R3.epsilon
    K = TensorHead("K", [R3])
    V = TensorHead("V", [R3])
    W = WildTensorHead('W', unordered_indices=True)
    U = WildTensorHead('U')
    k = Symbol("K")

    assert ( wi*K(p) ).matches( K(p) ) == {wi: 1}
    assert ( wi * eps(p,q,r) ).matches(eps(p,r,q)) == {wi:-1}
    assert ( K(p)*V(-p) ).replace( W(a)*V(-a), 1) == 1
    assert ( K(q)*K(p)*V(-p) ).replace( W(q,a)*V(-a), 1) == 1
    assert ( K(p)*V(-p) ).replace( K(-a)*V(a), 1 ) == 1
    assert ( K(q)*K(p)*V(-p) ).replace( W(q)*U(p)*V(-p), 1) == 1
    assert (
        (K(p)*V(q)).replace(W()*K(p)*V(q), W()*V(p)*V(q)).doit()
        == V(p)*V(q)
        )
    assert (
        ( eps(r,p,q)*eps(-r,-s,-t) ).replace(
            eps(e,a,b)*eps(-e,c,d),
            delta(a,c)*delta(b,d) - delta(a,d)*delta(b,c),
            ).doit().canon_bp()
        == delta(p,-s)*delta(q,-t) - delta(p,-t)*delta(q,-s)
        )
    assert (
        ( eps(r,p,q)*eps(-r,-p,-q) ).replace(
            eps(c,a,b)*eps(-c,d,f),
            delta(a,d)*delta(b,f) - delta(a,f)*delta(b,d),
            ).contract_delta(delta).doit()
        == 6
        )
    assert ( V(-p)*V(q)*V(-q) ).replace( wi*W()*V(a)*V(-a), wi*W() ).doit() == V(-p)
    assert ( k**4*K(r)*K(-r) ).replace( wi*W()*K(a)*K(-a), wi*W()*k**2 ).doit() == k**6

    #Multiple occurrence of WildTensor in value
    assert (
        ( K(p)*V(q) ).replace(W(q)*K(p), W(p)*W(q))
        == V(p)*V(q)
        )
    assert (
        ( K(p)*V(q)*V(r) ).replace(W(q,r)*K(p), W(p,r)*W(q,s)*V(-s) )
        == V(p)*V(r)*V(q)*V(s)*V(-s)
        )

    #Edge case involving automatic index relabelling
    D0, D1, D2, D3 = tensor_indices("R_0 R_1 R_2 R_3", R3)
    expr = delta(-D0, -D1)*K(D2)*K(D3)*K(-D3)
    m = ( W()*K(a)*K(-a) ).matches(expr)
    assert D2 not in m.values()

def test_TensMul_subs():
    """
    Test subs and xreplace in TensMul. See bug #24337
    """
    R3 = TensorIndexType('R3', dim=3)
    p, q, r = tensor_indices("p q r", R3)
    K = TensorHead("K", [R3])
    V = TensorHead("V", [R3])
    A = TensorHead("A", [R3, R3])
    C0 = TensorIndex(R3.dummy_name + "_0", R3, True)
    a = WildTensorIndex("a", R3, ignore_updown=True)

    assert ( K(p)*V(r)*K(-p) ).subs({V(r): K(q)*K(-q)}) == K(p)*K(q)*K(-q)*K(-p)
    assert ( K(p)*V(r)*K(-p) ).xreplace({V(r): K(q)*K(-q)}) == K(p)*K(q)*K(-q)*K(-p)
    assert ( K(p)*V(r) ).xreplace({p: C0, V(r): K(q)*K(-q)}) == K(C0)*K(q)*K(-q)
    assert ( K(p)*A(q,-q)*K(-p) ).doit() == K(p)*A(q,-q)*K(-p)
    assert ( K(p)*V(-p) ).replace( K(a), V(a)*V(q)*V(-q) ) == V(p)*V(q)*V(-q)*V(-p)


def test_tensorsymmetry():
    with warns_deprecated_sympy():
        tensorsymmetry([1]*2)

def test_tensorhead():
    with warns_deprecated_sympy():
        tensorhead('A', [])

def test_TensorType():
    with warns_deprecated_sympy():
        sym2 = TensorSymmetry.fully_symmetric(2)
        Lorentz = TensorIndexType('Lorentz')
        S2 = TensorType([Lorentz]*2, sym2)
        assert isinstance(S2, TensorType)

def test_dummy_fmt():
    with warns_deprecated_sympy():
        TensorIndexType('Lorentz', dummy_fmt='L')

def test_postprocessor():
    """
    Test if substituting a Tensor into a Mul or Add automatically converts it
    to TensMul or TensAdd respectively. See github issue #25051
    """
    R3 = TensorIndexType('R3', dim=3)
    i = tensor_indices("i", R3)
    K = TensorHead("K", [R3])
    x,y,z = symbols("x y z")

    assert isinstance((x*2).xreplace({x: K(i)}), TensMul)
    assert isinstance((x+2).xreplace({x: K(i)*K(-i)}), TensAdd)

    assert isinstance((x*2).subs({x: K(i)}), TensMul)
    assert isinstance((x+2).subs({x: K(i)*K(-i)}), TensAdd)

    assert isinstance((x*2).replace(x, K(i)), TensMul)
    assert isinstance((x+2).replace(x, K(i)*K(-i)), TensAdd)

def test_TensMul_nocoeff():
    """
    Ensure that for any TensMul instance, self.coeff * self.nocoeff == self
    """

    R3 = TensorIndexType('R3', dim=3)
    i, j = tensor_indices("i j", R3)
    K = TensorHead("K", [R3])
    P = TensorHead("P", [R3])

    expr = TensMul(2, K(i), P(j))
    assert expr.coeff * expr.nocoeff == expr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\DeprecatedNodeIndexAndKernelDefHash.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

# deprecated: no longer using kernel def hashes
class DeprecatedNodeIndexAndKernelDefHash(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DeprecatedNodeIndexAndKernelDefHash()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDeprecatedNodeIndexAndKernelDefHash(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def DeprecatedNodeIndexAndKernelDefHashBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # DeprecatedNodeIndexAndKernelDefHash
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DeprecatedNodeIndexAndKernelDefHash
    def NodeIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # DeprecatedNodeIndexAndKernelDefHash
    def KernelDefHash(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

def DeprecatedNodeIndexAndKernelDefHashStart(builder):
    builder.StartObject(2)

def Start(builder):
    DeprecatedNodeIndexAndKernelDefHashStart(builder)

def DeprecatedNodeIndexAndKernelDefHashAddNodeIndex(builder, nodeIndex):
    builder.PrependUint32Slot(0, nodeIndex, 0)

def AddNodeIndex(builder, nodeIndex):
    DeprecatedNodeIndexAndKernelDefHashAddNodeIndex(builder, nodeIndex)

def DeprecatedNodeIndexAndKernelDefHashAddKernelDefHash(builder, kernelDefHash):
    builder.PrependUint64Slot(1, kernelDefHash, 0)

def AddKernelDefHash(builder, kernelDefHash):
    DeprecatedNodeIndexAndKernelDefHashAddKernelDefHash(builder, kernelDefHash)

def DeprecatedNodeIndexAndKernelDefHashEnd(builder):
    return builder.EndObject()

def End(builder):
    return DeprecatedNodeIndexAndKernelDefHashEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\sets.py ===
from sympy.core.assumptions import check_assumptions
from sympy.core.logic import fuzzy_and
from sympy.core.sympify import _sympify
from sympy.matrices.kind import MatrixKind
from sympy.sets.sets import Set, SetKind
from sympy.core.kind import NumberKind
from .matexpr import MatrixExpr


class MatrixSet(Set):
    """
    MatrixSet represents the set of matrices with ``shape = (n, m)`` over the
    given set.

    Examples
    ========

    >>> from sympy.matrices import MatrixSet
    >>> from sympy import S, I, Matrix
    >>> M = MatrixSet(2, 2, set=S.Reals)
    >>> X = Matrix([[1, 2], [3, 4]])
    >>> X in M
    True
    >>> X = Matrix([[1, 2], [I, 4]])
    >>> X in M
    False

    """
    is_empty = False

    def __new__(cls, n, m, set):
        n, m, set = _sympify(n), _sympify(m), _sympify(set)
        cls._check_dim(n)
        cls._check_dim(m)
        if not isinstance(set, Set):
            raise TypeError("{} should be an instance of Set.".format(set))
        return Set.__new__(cls, n, m, set)

    @property
    def shape(self):
        return self.args[:2]

    @property
    def set(self):
        return self.args[2]

    def _contains(self, other):
        if not isinstance(other, MatrixExpr):
            raise TypeError("{} should be an instance of MatrixExpr.".format(other))
        if other.shape != self.shape:
            are_symbolic = any(_sympify(x).is_Symbol for x in other.shape + self.shape)
            if are_symbolic:
                return None
            return False
        return fuzzy_and(self.set.contains(x) for x in other)

    @classmethod
    def _check_dim(cls, dim):
        """Helper function to check invalid matrix dimensions"""
        ok = not dim.is_Float and check_assumptions(
            dim, integer=True, nonnegative=True)
        if ok is False:
            raise ValueError(
                "The dimension specification {} should be "
                "a nonnegative integer.".format(dim))

    def _kind(self):
        return SetKind(MatrixKind(NumberKind))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\multipledispatch\conflict.py ===
from .utils import _toposort, groupby

class AmbiguityWarning(Warning):
    pass


def supercedes(a, b):
    """ A is consistent and strictly more specific than B """
    return len(a) == len(b) and all(map(issubclass, a, b))


def consistent(a, b):
    """ It is possible for an argument list to satisfy both A and B """
    return (len(a) == len(b) and
            all(issubclass(aa, bb) or issubclass(bb, aa)
                           for aa, bb in zip(a, b)))


def ambiguous(a, b):
    """ A is consistent with B but neither is strictly more specific """
    return consistent(a, b) and not (supercedes(a, b) or supercedes(b, a))


def ambiguities(signatures):
    """ All signature pairs such that A is ambiguous with B """
    signatures = list(map(tuple, signatures))
    return {(a, b) for a in signatures for b in signatures
                       if hash(a) < hash(b)
                       and ambiguous(a, b)
                       and not any(supercedes(c, a) and supercedes(c, b)
                                    for c in signatures)}


def super_signature(signatures):
    """ A signature that would break ambiguities """
    n = len(signatures[0])
    assert all(len(s) == n for s in signatures)

    return [max([type.mro(sig[i]) for sig in signatures], key=len)[0]
               for i in range(n)]


def edge(a, b, tie_breaker=hash):
    """ A should be checked before B

    Tie broken by tie_breaker, defaults to ``hash``
    """
    if supercedes(a, b):
        if supercedes(b, a):
            return tie_breaker(a) > tie_breaker(b)
        else:
            return True
    return False


def ordering(signatures):
    """ A sane ordering of signatures to check, first to last

    Topoological sort of edges as given by ``edge`` and ``supercedes``
    """
    signatures = list(map(tuple, signatures))
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    edges = groupby(lambda x: x[0], edges)
    for s in signatures:
        if s not in edges:
            edges[s] = []
    edges = {k: [b for a, b in v] for k, v in edges.items()}
    return _toposort(edges)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\plot_rotation.py ===
try:
    from ctypes import c_float
except ImportError:
    pass

import pyglet.gl as pgl
from math import sqrt as _sqrt, acos as _acos, pi


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def mag(a):
    return _sqrt(a[0]**2 + a[1]**2 + a[2]**2)


def norm(a):
    m = mag(a)
    return (a[0] / m, a[1] / m, a[2] / m)


def get_sphere_mapping(x, y, width, height):
    x = min([max([x, 0]), width])
    y = min([max([y, 0]), height])

    sr = _sqrt((width/2)**2 + (height/2)**2)
    sx = ((x - width / 2) / sr)
    sy = ((y - height / 2) / sr)

    sz = 1.0 - sx**2 - sy**2

    if sz > 0.0:
        sz = _sqrt(sz)
        return (sx, sy, sz)
    else:
        sz = 0
        return norm((sx, sy, sz))

rad2deg = 180.0 / pi


def get_spherical_rotatation(p1, p2, width, height, theta_multiplier):
    v1 = get_sphere_mapping(p1[0], p1[1], width, height)
    v2 = get_sphere_mapping(p2[0], p2[1], width, height)

    d = min(max([dot(v1, v2), -1]), 1)

    if abs(d - 1.0) < 0.000001:
        return None

    raxis = norm( cross(v1, v2) )
    rtheta = theta_multiplier * rad2deg * _acos(d)

    pgl.glPushMatrix()
    pgl.glLoadIdentity()
    pgl.glRotatef(rtheta, *raxis)
    mat = (c_float*16)()
    pgl.glGetFloatv(pgl.GL_MODELVIEW_MATRIX, mat)
    pgl.glPopMatrix()

    return mat

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\json\test_pandas.py ===
import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
    BytesIO,
    StringIO,
)
import json
import os
import sys
import time

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import IS64
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    NA,
    DataFrame,
    DatetimeIndex,
    Index,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    read_json,
)
import pandas._testing as tm

from pandas.io.json import ujson_dumps


def test_literal_json_deprecation():
    # PR 53409
    expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])

    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""

    msg = (
        "Passing literal json to 'read_json' is deprecated and "
        "will be removed in a future version. To read from a "
        "literal string, wrap it in a 'StringIO' object."
    )

    with tm.assert_produces_warning(FutureWarning, match=msg):
        try:
            read_json(jsonl, lines=False)
        except ValueError:
            pass

    with tm.assert_produces_warning(FutureWarning, match=msg):
        read_json(expected.to_json(), lines=False)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=True)
        tm.assert_frame_equal(result, expected)

    with tm.assert_produces_warning(FutureWarning, match=msg):
        try:
            result = read_json(
                '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n',
                lines=False,
            )
        except ValueError:
            pass

    with tm.assert_produces_warning(FutureWarning, match=msg):
        try:
            result = read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=False)
        except ValueError:
            pass
        tm.assert_frame_equal(result, expected)


def assert_json_roundtrip_equal(result, expected, orient):
    if orient in ("records", "values"):
        expected = expected.reset_index(drop=True)
    if orient == "values":
        expected.columns = range(len(expected.columns))
    tm.assert_frame_equal(result, expected)


class TestPandasContainer:
    @pytest.fixture
    def categorical_frame(self):
        data = {
            c: np.random.default_rng(i).standard_normal(30)
            for i, c in enumerate(list("ABCD"))
        }
        cat = ["bah"] * 5 + ["bar"] * 5 + ["baz"] * 5 + ["foo"] * 15
        data["E"] = list(reversed(cat))
        data["sort"] = np.arange(30, dtype="int64")
        return DataFrame(data, index=pd.CategoricalIndex(cat, name="E"))

    @pytest.fixture
    def datetime_series(self):
        # Same as usual datetime_series, but with index freq set to None,
        #  since that doesn't round-trip, see GH#33711
        ser = Series(
            1.1 * np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        ser.index = ser.index._with_freq(None)
        return ser

    @pytest.fixture
    def datetime_frame(self):
        # Same as usual datetime_frame, but with index freq set to None,
        #  since that doesn't round-trip, see GH#33711
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=30, freq="B"),
        )
        df.index = df.index._with_freq(None)
        return df

    def test_frame_double_encoded_labels(self, orient):
        df = DataFrame(
            [["a", "b"], ["c", "d"]],
            index=['index " 1', "index / 2"],
            columns=["a \\ b", "y / z"],
        )

        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient)
        expected = df.copy()
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("orient", ["split", "records", "values"])
    def test_frame_non_unique_index(self, orient):
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 1], columns=["x", "y"])
        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient)
        expected = df.copy()

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("orient", ["index", "columns"])
    def test_frame_non_unique_index_raises(self, orient):
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 1], columns=["x", "y"])
        msg = f"DataFrame index must be unique for orient='{orient}'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient)

    @pytest.mark.parametrize("orient", ["split", "values"])
    @pytest.mark.parametrize(
        "data",
        [
            [["a", "b"], ["c", "d"]],
            [[1.5, 2.5], [3.5, 4.5]],
            [[1, 2.5], [3, 4.5]],
            [[Timestamp("20130101"), 3.5], [Timestamp("20130102"), 4.5]],
        ],
    )
    def test_frame_non_unique_columns(self, orient, data):
        df = DataFrame(data, index=[1, 2], columns=["x", "x"])

        result = read_json(
            StringIO(df.to_json(orient=orient)), orient=orient, convert_dates=["x"]
        )
        if orient == "values":
            expected = DataFrame(data)
            if expected.iloc[:, 0].dtype == "datetime64[ns]":
                # orient == "values" by default will write Timestamp objects out
                # in milliseconds; these are internally stored in nanosecond,
                # so divide to get where we need
                # TODO: a to_epoch method would also solve; see GH 14772
                expected.isetitem(0, expected.iloc[:, 0].astype(np.int64) // 1000000)
        elif orient == "split":
            expected = df
            expected.columns = ["x", "x.1"]

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("orient", ["index", "columns", "records"])
    def test_frame_non_unique_columns_raises(self, orient):
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 2], columns=["x", "x"])

        msg = f"DataFrame columns must be unique for orient='{orient}'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient)

    def test_frame_default_orient(self, float_frame):
        assert float_frame.to_json() == float_frame.to_json(orient="columns")

    @pytest.mark.parametrize("dtype", [False, float])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_simple(self, orient, convert_axes, dtype, float_frame):
        data = StringIO(float_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)

        expected = float_frame

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("dtype", [False, np.int64])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_intframe(self, orient, convert_axes, dtype, int_frame):
        data = StringIO(int_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)
        expected = int_frame
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("dtype", [None, np.float64, int, "U3"])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_str_axes(self, orient, convert_axes, dtype):
        df = DataFrame(
            np.zeros((200, 4)),
            columns=[str(i) for i in range(4)],
            index=[str(i) for i in range(200)],
            dtype=dtype,
        )

        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)

        expected = df.copy()
        if not dtype:
            expected = expected.astype(np.int64)

        # index columns, and records orients cannot fully preserve the string
        # dtype for axes as the index and column labels are used as keys in
        # JSON objects. JSON keys are by definition strings, so there's no way
        # to disambiguate whether those keys actually were strings or numeric
        # beforehand and numeric wins out.
        if convert_axes and (orient in ("index", "columns")):
            expected.columns = expected.columns.astype(np.int64)
            expected.index = expected.index.astype(np.int64)
        elif orient == "records" and convert_axes:
            expected.columns = expected.columns.astype(np.int64)
        elif convert_axes and orient == "split":
            expected.columns = expected.columns.astype(np.int64)

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_categorical(
        self, request, orient, categorical_frame, convert_axes, using_infer_string
    ):
        # TODO: create a better frame to test with and improve coverage
        if orient in ("index", "columns"):
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"Can't have duplicate index values for orient '{orient}')"
                )
            )

        data = StringIO(categorical_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)

        expected = categorical_frame.copy()
        expected.index = expected.index.astype(
            str if not using_infer_string else "str"
        )  # Categorical not preserved
        expected.index.name = None  # index names aren't preserved in JSON
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_empty(self, orient, convert_axes):
        empty_frame = DataFrame()
        data = StringIO(empty_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)
        if orient == "split":
            idx = Index([], dtype=(float if convert_axes else object))
            expected = DataFrame(index=idx, columns=idx)
        elif orient in ["index", "columns"]:
            expected = DataFrame()
        else:
            expected = empty_frame.copy()

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_timestamp(self, orient, convert_axes, datetime_frame):
        # TODO: improve coverage with date_format parameter
        data = StringIO(datetime_frame.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)
        expected = datetime_frame.copy()

        if not convert_axes:  # one off for ts handling
            # DTI gets converted to epoch values
            idx = expected.index.view(np.int64) // 1000000
            if orient != "split":  # TODO: handle consistently across orients
                idx = idx.astype(str)

            expected.index = idx

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_mixed(self, orient, convert_axes):
        index = Index(["a", "b", "c", "d", "e"])
        values = {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0],
            "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
            "D": [True, False, True, False, True],
        }

        df = DataFrame(data=values, index=index)

        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient, convert_axes=convert_axes)

        expected = df.copy()
        expected = expected.assign(**expected.select_dtypes("number").astype(np.int64))

        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.xfail(
        reason="#50456 Column multiindex is stored and loaded differently",
        raises=AssertionError,
    )
    @pytest.mark.parametrize(
        "columns",
        [
            [["2022", "2022"], ["JAN", "FEB"]],
            [["2022", "2023"], ["JAN", "JAN"]],
            [["2022", "2022"], ["JAN", "JAN"]],
        ],
    )
    def test_roundtrip_multiindex(self, columns):
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.MultiIndex.from_arrays(columns),
        )
        data = StringIO(df.to_json(orient="split"))
        result = read_json(data, orient="split")
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize(
        "data,msg,orient",
        [
            ('{"key":b:a:d}', "Expected object or value", "columns"),
            # too few indices
            (
                '{"columns":["A","B"],'
                '"index":["2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                "|".join(
                    [
                        r"Length of values \(3\) does not match length of index \(2\)",
                    ]
                ),
                "split",
            ),
            # too many columns
            (
                '{"columns":["A","B","C"],'
                '"index":["1","2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                "3 columns passed, passed data had 2 columns",
                "split",
            ),
            # bad key
            (
                '{"badkey":["A","B"],'
                '"index":["2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                r"unexpected key\(s\): badkey",
                "split",
            ),
        ],
    )
    def test_frame_from_json_bad_data_raises(self, data, msg, orient):
        with pytest.raises(ValueError, match=msg):
            read_json(StringIO(data), orient=orient)

    @pytest.mark.parametrize("dtype", [True, False])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_frame_from_json_missing_data(self, orient, convert_axes, dtype):
        num_df = DataFrame([[1, 2], [4, 5, 6]])

        result = read_json(
            StringIO(num_df.to_json(orient=orient)),
            orient=orient,
            convert_axes=convert_axes,
            dtype=dtype,
        )
        assert np.isnan(result.iloc[0, 2])

        obj_df = DataFrame([["1", "2"], ["4", "5", "6"]])
        result = read_json(
            StringIO(obj_df.to_json(orient=orient)),
            orient=orient,
            convert_axes=convert_axes,
            dtype=dtype,
        )
        assert np.isnan(result.iloc[0, 2])

    @pytest.mark.parametrize("dtype", [True, False])
    def test_frame_read_json_dtype_missing_value(self, dtype):
        # GH28501 Parse missing values using read_json with dtype=False
        # to NaN instead of None
        result = read_json(StringIO("[null]"), dtype=dtype)
        expected = DataFrame([np.nan])

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("inf", [np.inf, -np.inf])
    @pytest.mark.parametrize("dtype", [True, False])
    def test_frame_infinity(self, inf, dtype):
        # infinities get mapped to nulls which get mapped to NaNs during
        # deserialisation
        df = DataFrame([[1, 2], [4, 5, 6]])
        df.loc[0, 2] = inf

        data = StringIO(df.to_json())
        result = read_json(data, dtype=dtype)
        assert np.isnan(result.iloc[0, 2])

    @pytest.mark.skipif(not IS64, reason="not compliant on 32-bit, xref #15865")
    @pytest.mark.parametrize(
        "value,precision,expected_val",
        [
            (0.95, 1, 1.0),
            (1.95, 1, 2.0),
            (-1.95, 1, -2.0),
            (0.995, 2, 1.0),
            (0.9995, 3, 1.0),
            (0.99999999999999944, 15, 1.0),
        ],
    )
    def test_frame_to_json_float_precision(self, value, precision, expected_val):
        df = DataFrame([{"a_float": value}])
        encoded = df.to_json(double_precision=precision)
        assert encoded == f'{{"a_float":{{"0":{expected_val}}}}}'

    def test_frame_to_json_except(self):
        df = DataFrame([1, 2, 3])
        msg = "Invalid value 'garbage' for option 'orient'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient="garbage")

    def test_frame_empty(self):
        df = DataFrame(columns=["jim", "joe"])
        assert not df._is_mixed_type

        data = StringIO(df.to_json())
        result = read_json(data, dtype=dict(df.dtypes))
        tm.assert_frame_equal(result, df, check_index_type=False)

    def test_frame_empty_to_json(self):
        # GH 7445
        df = DataFrame({"test": []}, index=[])
        result = df.to_json(orient="columns")
        expected = '{"test":{}}'
        assert result == expected

    def test_frame_empty_mixedtype(self):
        # mixed type
        df = DataFrame(columns=["jim", "joe"])
        df["joe"] = df["joe"].astype("i8")
        assert df._is_mixed_type
        data = df.to_json()
        tm.assert_frame_equal(
            read_json(StringIO(data), dtype=dict(df.dtypes)),
            df,
            check_index_type=False,
        )

    def test_frame_mixedtype_orient(self):  # GH10289
        vals = [
            [10, 1, "foo", 0.1, 0.01],
            [20, 2, "bar", 0.2, 0.02],
            [30, 3, "baz", 0.3, 0.03],
            [40, 4, "qux", 0.4, 0.04],
        ]

        df = DataFrame(
            vals, index=list("abcd"), columns=["1st", "2nd", "3rd", "4th", "5th"]
        )

        assert df._is_mixed_type
        right = df.copy()

        for orient in ["split", "index", "columns"]:
            inp = StringIO(df.to_json(orient=orient))
            left = read_json(inp, orient=orient, convert_axes=False)
            tm.assert_frame_equal(left, right)

        right.index = RangeIndex(len(df))
        inp = StringIO(df.to_json(orient="records"))
        left = read_json(inp, orient="records", convert_axes=False)
        tm.assert_frame_equal(left, right)

        right.columns = RangeIndex(df.shape[1])
        inp = StringIO(df.to_json(orient="values"))
        left = read_json(inp, orient="values", convert_axes=False)
        tm.assert_frame_equal(left, right)

    def test_v12_compat(self, datapath):
        dti = date_range("2000-01-03", "2000-01-07")
        # freq doesn't roundtrip
        dti = DatetimeIndex(np.asarray(dti), freq=None)
        df = DataFrame(
            [
                [1.56808523, 0.65727391, 1.81021139, -0.17251653],
                [-0.2550111, -0.08072427, -0.03202878, -0.17581665],
                [1.51493992, 0.11805825, 1.629455, -1.31506612],
                [-0.02765498, 0.44679743, 0.33192641, -0.27885413],
                [0.05951614, -2.69652057, 1.28163262, 0.34703478],
            ],
            columns=["A", "B", "C", "D"],
            index=dti,
        )
        df["date"] = Timestamp("19920106 18:21:32.12").as_unit("ns")
        df.iloc[3, df.columns.get_loc("date")] = Timestamp("20130101")
        df["modified"] = df["date"]
        df.iloc[1, df.columns.get_loc("modified")] = pd.NaT

        dirpath = datapath("io", "json", "data")
        v12_json = os.path.join(dirpath, "tsframe_v012.json")
        df_unser = read_json(v12_json)
        tm.assert_frame_equal(df, df_unser)

        df_iso = df.drop(["modified"], axis=1)
        v12_iso_json = os.path.join(dirpath, "tsframe_iso_v012.json")
        df_unser_iso = read_json(v12_iso_json)
        tm.assert_frame_equal(df_iso, df_unser_iso, check_column_type=False)

    def test_blocks_compat_GH9037(self, using_infer_string):
        index = date_range("20000101", periods=10, freq="h")
        # freq doesn't round-trip
        index = DatetimeIndex(list(index), freq=None)

        df_mixed = DataFrame(
            {
                "float_1": [
                    -0.92077639,
                    0.77434435,
                    1.25234727,
                    0.61485564,
                    -0.60316077,
                    0.24653374,
                    0.28668979,
                    -2.51969012,
                    0.95748401,
                    -1.02970536,
                ],
                "int_1": [
                    19680418,
                    75337055,
                    99973684,
                    65103179,
                    79373900,
                    40314334,
                    21290235,
                    4991321,
                    41903419,
                    16008365,
                ],
                "str_1": [
                    "78c608f1",
                    "64a99743",
                    "13d2ff52",
                    "ca7f4af2",
                    "97236474",
                    "bde7e214",
                    "1a6bde47",
                    "b1190be5",
                    "7a669144",
                    "8d64d068",
                ],
                "float_2": [
                    -0.0428278,
                    -1.80872357,
                    3.36042349,
                    -0.7573685,
                    -0.48217572,
                    0.86229683,
                    1.08935819,
                    0.93898739,
                    -0.03030452,
                    1.43366348,
                ],
                "str_2": [
                    "14f04af9",
                    "d085da90",
                    "4bcfac83",
                    "81504caf",
                    "2ffef4a9",
                    "08e2f5c4",
                    "07e1af03",
                    "addbd4a7",
                    "1f6a09ba",
                    "4bfc4d87",
                ],
                "int_2": [
                    86967717,
                    98098830,
                    51927505,
                    20372254,
                    12601730,
                    20884027,
                    34193846,
                    10561746,
                    24867120,
                    76131025,
                ],
            },
            index=index,
        )

        # JSON deserialisation always creates unicode strings
        df_mixed.columns = df_mixed.columns.astype(
            np.str_ if not using_infer_string else "str"
        )
        data = StringIO(df_mixed.to_json(orient="split"))
        df_roundtrip = read_json(data, orient="split")
        tm.assert_frame_equal(
            df_mixed,
            df_roundtrip,
            check_index_type=True,
            check_column_type=True,
            by_blocks=True,
            check_exact=True,
        )

    def test_frame_nonprintable_bytes(self):
        # GH14256: failing column caused segfaults, if it is not the last one

        class BinaryThing:
            def __init__(self, hexed) -> None:
                self.hexed = hexed
                self.binary = bytes.fromhex(hexed)

            def __str__(self) -> str:
                return self.hexed

        hexed = "574b4454ba8c5eb4f98a8f45"
        binthing = BinaryThing(hexed)

        # verify the proper conversion of printable content
        df_printable = DataFrame({"A": [binthing.hexed]})
        assert df_printable.to_json() == f'{{"A":{{"0":"{hexed}"}}}}'

        # check if non-printable content throws appropriate Exception
        df_nonprintable = DataFrame({"A": [binthing]})
        msg = "Unsupported UTF-8 sequence length when encoding string"
        with pytest.raises(OverflowError, match=msg):
            df_nonprintable.to_json()

        # the same with multiple columns threw segfaults
        df_mixed = DataFrame({"A": [binthing], "B": [1]}, columns=["A", "B"])
        with pytest.raises(OverflowError, match=msg):
            df_mixed.to_json()

        # default_handler should resolve exceptions for non-string types
        result = df_nonprintable.to_json(default_handler=str)
        expected = f'{{"A":{{"0":"{hexed}"}}}}'
        assert result == expected
        assert (
            df_mixed.to_json(default_handler=str)
            == f'{{"A":{{"0":"{hexed}"}},"B":{{"0":1}}}}'
        )

    def test_label_overflow(self):
        # GH14256: buffer length not checked when writing label
        result = DataFrame({"bar" * 100000: [1], "foo": [1337]}).to_json()
        expected = f'{{"{"bar" * 100000}":{{"0":1}},"foo":{{"0":1337}}}}'
        assert result == expected

    def test_series_non_unique_index(self):
        s = Series(["a", "b"], index=[1, 1])

        msg = "Series index must be unique for orient='index'"
        with pytest.raises(ValueError, match=msg):
            s.to_json(orient="index")

        tm.assert_series_equal(
            s,
            read_json(
                StringIO(s.to_json(orient="split")), orient="split", typ="series"
            ),
        )
        unserialized = read_json(
            StringIO(s.to_json(orient="records")), orient="records", typ="series"
        )
        tm.assert_equal(s.values, unserialized.values)

    def test_series_default_orient(self, string_series):
        assert string_series.to_json() == string_series.to_json(orient="index")

    def test_series_roundtrip_simple(self, orient, string_series, using_infer_string):
        data = StringIO(string_series.to_json(orient=orient))
        result = read_json(data, typ="series", orient=orient)

        expected = string_series
        if using_infer_string and orient in ("split", "index", "columns"):
            # These schemas don't contain dtypes, so we infer string
            expected.index = expected.index.astype("str")
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)
        if orient != "split":
            expected.name = None

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [False, None])
    def test_series_roundtrip_object(self, orient, dtype, object_series):
        data = StringIO(object_series.to_json(orient=orient))
        result = read_json(data, typ="series", orient=orient, dtype=dtype)

        expected = object_series
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)
        if orient != "split":
            expected.name = None

        if using_string_dtype():
            expected = expected.astype("str")

        tm.assert_series_equal(result, expected)

    def test_series_roundtrip_empty(self, orient):
        empty_series = Series([], index=[], dtype=np.float64)
        data = StringIO(empty_series.to_json(orient=orient))
        result = read_json(data, typ="series", orient=orient)

        expected = empty_series.reset_index(drop=True)
        if orient in ("split"):
            expected.index = expected.index.astype(np.float64)

        tm.assert_series_equal(result, expected)

    def test_series_roundtrip_timeseries(self, orient, datetime_series):
        data = StringIO(datetime_series.to_json(orient=orient))
        result = read_json(data, typ="series", orient=orient)

        expected = datetime_series
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)
        if orient != "split":
            expected.name = None

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float64, int])
    def test_series_roundtrip_numeric(self, orient, dtype):
        s = Series(range(6), index=["a", "b", "c", "d", "e", "f"])
        data = StringIO(s.to_json(orient=orient))
        result = read_json(data, typ="series", orient=orient)

        expected = s.copy()
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)

        tm.assert_series_equal(result, expected)

    def test_series_to_json_except(self):
        s = Series([1, 2, 3])
        msg = "Invalid value 'garbage' for option 'orient'"
        with pytest.raises(ValueError, match=msg):
            s.to_json(orient="garbage")

    def test_series_from_json_precise_float(self):
        s = Series([4.56, 4.56, 4.56])
        result = read_json(StringIO(s.to_json()), typ="series", precise_float=True)
        tm.assert_series_equal(result, s, check_index_type=False)

    def test_series_with_dtype(self):
        # GH 21986
        s = Series([4.56, 4.56, 4.56])
        result = read_json(StringIO(s.to_json()), typ="series", dtype=np.int64)
        expected = Series([4] * 3)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (True, Series(["2000-01-01"], dtype="datetime64[ns]")),
            (False, Series([946684800000])),
        ],
    )
    def test_series_with_dtype_datetime(self, dtype, expected):
        s = Series(["2000-01-01"], dtype="datetime64[ns]")
        data = StringIO(s.to_json())
        result = read_json(data, typ="series", dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_frame_from_json_precise_float(self):
        df = DataFrame([[4.56, 4.56, 4.56], [4.56, 4.56, 4.56]])
        result = read_json(StringIO(df.to_json()), precise_float=True)
        tm.assert_frame_equal(result, df)

    def test_typ(self):
        s = Series(range(6), index=["a", "b", "c", "d", "e", "f"], dtype="int64")
        result = read_json(StringIO(s.to_json()), typ=None)
        tm.assert_series_equal(result, s)

    def test_reconstruction_index(self):
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        result = read_json(StringIO(df.to_json()))
        tm.assert_frame_equal(result, df)

        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["A", "B", "C"])
        result = read_json(StringIO(df.to_json()))
        tm.assert_frame_equal(result, df)

    def test_path(self, float_frame, int_frame, datetime_frame):
        with tm.ensure_clean("test.json") as path:
            for df in [float_frame, int_frame, datetime_frame]:
                df.to_json(path)
                read_json(path)

    def test_axis_dates(self, datetime_series, datetime_frame):
        # frame
        json = StringIO(datetime_frame.to_json())
        result = read_json(json)
        tm.assert_frame_equal(result, datetime_frame)

        # series
        json = StringIO(datetime_series.to_json())
        result = read_json(json, typ="series")
        tm.assert_series_equal(result, datetime_series, check_names=False)
        assert result.name is None

    def test_convert_dates(self, datetime_series, datetime_frame):
        # frame
        df = datetime_frame
        df["date"] = Timestamp("20130101").as_unit("ns")

        json = StringIO(df.to_json())
        result = read_json(json)
        tm.assert_frame_equal(result, df)

        df["foo"] = 1.0
        json = StringIO(df.to_json(date_unit="ns"))

        result = read_json(json, convert_dates=False)
        expected = df.copy()
        expected["date"] = expected["date"].values.view("i8")
        expected["foo"] = expected["foo"].astype("int64")
        tm.assert_frame_equal(result, expected)

        # series
        ts = Series(Timestamp("20130101").as_unit("ns"), index=datetime_series.index)
        json = StringIO(ts.to_json())
        result = read_json(json, typ="series")
        tm.assert_series_equal(result, ts)

    @pytest.mark.parametrize("date_format", ["epoch", "iso"])
    @pytest.mark.parametrize("as_object", [True, False])
    @pytest.mark.parametrize("date_typ", [datetime.date, datetime.datetime, Timestamp])
    def test_date_index_and_values(self, date_format, as_object, date_typ):
        data = [date_typ(year=2020, month=1, day=1), pd.NaT]
        if as_object:
            data.append("a")

        ser = Series(data, index=data)
        result = ser.to_json(date_format=date_format)

        if date_format == "epoch":
            expected = '{"1577836800000":1577836800000,"null":null}'
        else:
            expected = (
                '{"2020-01-01T00:00:00.000":"2020-01-01T00:00:00.000","null":null}'
            )

        if as_object:
            expected = expected.replace("}", ',"a":"a"}')

        assert result == expected

    @pytest.mark.parametrize(
        "infer_word",
        [
            "trade_time",
            "date",
            "datetime",
            "sold_at",
            "modified",
            "timestamp",
            "timestamps",
        ],
    )
    def test_convert_dates_infer(self, infer_word):
        # GH10747

        data = [{"id": 1, infer_word: 1036713600000}, {"id": 2}]
        expected = DataFrame(
            [[1, Timestamp("2002-11-08")], [2, pd.NaT]], columns=["id", infer_word]
        )

        result = read_json(StringIO(ujson_dumps(data)))[["id", infer_word]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "date,date_unit",
        [
            ("20130101 20:43:42.123", None),
            ("20130101 20:43:42", "s"),
            ("20130101 20:43:42.123", "ms"),
            ("20130101 20:43:42.123456", "us"),
            ("20130101 20:43:42.123456789", "ns"),
        ],
    )
    def test_date_format_frame(self, date, date_unit, datetime_frame):
        df = datetime_frame

        df["date"] = Timestamp(date).as_unit("ns")
        df.iloc[1, df.columns.get_loc("date")] = pd.NaT
        df.iloc[5, df.columns.get_loc("date")] = pd.NaT
        if date_unit:
            json = df.to_json(date_format="iso", date_unit=date_unit)
        else:
            json = df.to_json(date_format="iso")

        result = read_json(StringIO(json))
        expected = df.copy()
        tm.assert_frame_equal(result, expected)

    def test_date_format_frame_raises(self, datetime_frame):
        df = datetime_frame
        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            df.to_json(date_format="iso", date_unit="foo")

    @pytest.mark.parametrize(
        "date,date_unit",
        [
            ("20130101 20:43:42.123", None),
            ("20130101 20:43:42", "s"),
            ("20130101 20:43:42.123", "ms"),
            ("20130101 20:43:42.123456", "us"),
            ("20130101 20:43:42.123456789", "ns"),
        ],
    )
    def test_date_format_series(self, date, date_unit, datetime_series):
        ts = Series(Timestamp(date).as_unit("ns"), index=datetime_series.index)
        ts.iloc[1] = pd.NaT
        ts.iloc[5] = pd.NaT
        if date_unit:
            json = ts.to_json(date_format="iso", date_unit=date_unit)
        else:
            json = ts.to_json(date_format="iso")

        result = read_json(StringIO(json), typ="series")
        expected = ts.copy()
        tm.assert_series_equal(result, expected)

    def test_date_format_series_raises(self, datetime_series):
        ts = Series(Timestamp("20130101 20:43:42.123"), index=datetime_series.index)
        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            ts.to_json(date_format="iso", date_unit="foo")

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_date_unit(self, unit, datetime_frame):
        df = datetime_frame
        df["date"] = Timestamp("20130101 20:43:42").as_unit("ns")
        dl = df.columns.get_loc("date")
        df.iloc[1, dl] = Timestamp("19710101 20:43:42")
        df.iloc[2, dl] = Timestamp("21460101 20:43:42")
        df.iloc[4, dl] = pd.NaT

        json = df.to_json(date_format="epoch", date_unit=unit)

        # force date unit
        result = read_json(StringIO(json), date_unit=unit)
        tm.assert_frame_equal(result, df)

        # detect date unit
        result = read_json(StringIO(json), date_unit=None)
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("unit", ["s", "ms", "us"])
    def test_iso_non_nano_datetimes(self, unit):
        # Test that numpy datetimes
        # in an Index or a column with non-nano resolution can be serialized
        # correctly
        # GH53686
        index = DatetimeIndex(
            [np.datetime64("2023-01-01T11:22:33.123456", unit)],
            dtype=f"datetime64[{unit}]",
        )
        df = DataFrame(
            {
                "date": Series(
                    [np.datetime64("2022-01-01T11:22:33.123456", unit)],
                    dtype=f"datetime64[{unit}]",
                    index=index,
                ),
                "date_obj": Series(
                    [np.datetime64("2023-01-01T11:22:33.123456", unit)],
                    dtype=object,
                    index=index,
                ),
            },
        )

        buf = StringIO()
        df.to_json(buf, date_format="iso", date_unit=unit)
        buf.seek(0)

        # read_json always reads datetimes in nanosecond resolution
        # TODO: check_dtype/check_index_type should be removable
        # once read_json gets non-nano support
        tm.assert_frame_equal(
            read_json(buf, convert_dates=["date", "date_obj"]),
            df,
            check_index_type=False,
            check_dtype=False,
        )

    def test_weird_nested_json(self):
        # this used to core dump the parser
        s = r"""{
        "status": "success",
        "data": {
        "posts": [
            {
            "id": 1,
            "title": "A blog post",
            "body": "Some useful content"
            },
            {
            "id": 2,
            "title": "Another blog post",
            "body": "More content"
            }
           ]
          }
        }"""
        read_json(StringIO(s))

    def test_doc_example(self):
        dfj2 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("AB")
        )
        dfj2["date"] = Timestamp("20130101")
        dfj2["ints"] = range(5)
        dfj2["bools"] = True
        dfj2.index = date_range("20130101", periods=5)

        json = StringIO(dfj2.to_json())
        result = read_json(json, dtype={"ints": np.int64, "bools": np.bool_})
        tm.assert_frame_equal(result, result)

    def test_round_trip_exception(self, datapath):
        # GH 3867
        path = datapath("io", "json", "data", "teams.csv")
        df = pd.read_csv(path)
        s = df.to_json()

        result = read_json(StringIO(s))
        res = result.reindex(index=df.index, columns=df.columns)
        msg = "The 'downcast' keyword in fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = res.fillna(np.nan, downcast=False)
        tm.assert_frame_equal(res, df)

    @pytest.mark.network
    @pytest.mark.single_cpu
    @pytest.mark.parametrize(
        "field,dtype",
        [
            ["created_at", pd.DatetimeTZDtype(tz="UTC")],
            ["closed_at", "datetime64[ns]"],
            ["updated_at", pd.DatetimeTZDtype(tz="UTC")],
        ],
    )
    def test_url(self, field, dtype, httpserver):
        data = '{"created_at": ["2023-06-23T18:21:36Z"], "closed_at": ["2023-06-23T18:21:36"], "updated_at": ["2023-06-23T18:21:36Z"]}\n'  # noqa: E501
        httpserver.serve_content(content=data)
        result = read_json(httpserver.url, convert_dates=True)
        assert result[field].dtype == dtype

    def test_timedelta(self):
        converter = lambda x: pd.to_timedelta(x, unit="ms")

        ser = Series([timedelta(23), timedelta(seconds=5)])
        assert ser.dtype == "timedelta64[ns]"

        result = read_json(StringIO(ser.to_json()), typ="series").apply(converter)
        tm.assert_series_equal(result, ser)

        ser = Series([timedelta(23), timedelta(seconds=5)], index=Index([0, 1]))
        assert ser.dtype == "timedelta64[ns]"
        result = read_json(StringIO(ser.to_json()), typ="series").apply(converter)
        tm.assert_series_equal(result, ser)

        frame = DataFrame([timedelta(23), timedelta(seconds=5)])
        assert frame[0].dtype == "timedelta64[ns]"
        tm.assert_frame_equal(
            frame, read_json(StringIO(frame.to_json())).apply(converter)
        )

    def test_timedelta2(self):
        frame = DataFrame(
            {
                "a": [timedelta(days=23), timedelta(seconds=5)],
                "b": [1, 2],
                "c": date_range(start="20130101", periods=2),
            }
        )
        data = StringIO(frame.to_json(date_unit="ns"))
        result = read_json(data)
        result["a"] = pd.to_timedelta(result.a, unit="ns")
        result["c"] = pd.to_datetime(result.c)
        tm.assert_frame_equal(frame, result)

    def test_mixed_timedelta_datetime(self):
        td = timedelta(23)
        ts = Timestamp("20130101")
        frame = DataFrame({"a": [td, ts]}, dtype=object)

        expected = DataFrame(
            {"a": [pd.Timedelta(td).as_unit("ns")._value, ts.as_unit("ns")._value]}
        )
        data = StringIO(frame.to_json(date_unit="ns"))
        result = read_json(data, dtype={"a": "int64"})
        tm.assert_frame_equal(result, expected, check_index_type=False)

    @pytest.mark.parametrize("as_object", [True, False])
    @pytest.mark.parametrize("date_format", ["iso", "epoch"])
    @pytest.mark.parametrize("timedelta_typ", [pd.Timedelta, timedelta])
    def test_timedelta_to_json(self, as_object, date_format, timedelta_typ):
        # GH28156: to_json not correctly formatting Timedelta
        data = [timedelta_typ(days=1), timedelta_typ(days=2), pd.NaT]
        if as_object:
            data.append("a")

        ser = Series(data, index=data)
        if date_format == "iso":
            expected = (
                '{"P1DT0H0M0S":"P1DT0H0M0S","P2DT0H0M0S":"P2DT0H0M0S","null":null}'
            )
        else:
            expected = '{"86400000":86400000,"172800000":172800000,"null":null}'

        if as_object:
            expected = expected.replace("}", ',"a":"a"}')

        result = ser.to_json(date_format=date_format)
        assert result == expected

    @pytest.mark.parametrize("as_object", [True, False])
    @pytest.mark.parametrize("timedelta_typ", [pd.Timedelta, timedelta])
    def test_timedelta_to_json_fractional_precision(self, as_object, timedelta_typ):
        data = [timedelta_typ(milliseconds=42)]
        ser = Series(data, index=data)
        if as_object:
            ser = ser.astype(object)

        result = ser.to_json()
        expected = '{"42":42}'
        assert result == expected

    def test_default_handler(self):
        value = object()
        frame = DataFrame({"a": [7, value]})
        expected = DataFrame({"a": [7, str(value)]})
        result = read_json(StringIO(frame.to_json(default_handler=str)))
        tm.assert_frame_equal(expected, result, check_index_type=False)

    def test_default_handler_indirect(self):
        def default(obj):
            if isinstance(obj, complex):
                return [("mathjs", "Complex"), ("re", obj.real), ("im", obj.imag)]
            return str(obj)

        df_list = [
            9,
            DataFrame(
                {"a": [1, "STR", complex(4, -5)], "b": [float("nan"), None, "N/A"]},
                columns=["a", "b"],
            ),
        ]
        expected = (
            '[9,[[1,null],["STR",null],[[["mathjs","Complex"],'
            '["re",4.0],["im",-5.0]],"N\\/A"]]]'
        )
        assert (
            ujson_dumps(df_list, default_handler=default, orient="values") == expected
        )

    def test_default_handler_numpy_unsupported_dtype(self):
        # GH12554 to_json raises 'Unhandled numpy dtype 15'
        df = DataFrame(
            {"a": [1, 2.3, complex(4, -5)], "b": [float("nan"), None, complex(1.2, 0)]},
            columns=["a", "b"],
        )
        expected = (
            '[["(1+0j)","(nan+0j)"],'
            '["(2.3+0j)","(nan+0j)"],'
            '["(4-5j)","(1.2+0j)"]]'
        )
        assert df.to_json(default_handler=str, orient="values") == expected

    def test_default_handler_raises(self):
        msg = "raisin"

        def my_handler_raises(obj):
            raise TypeError(msg)

        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": [1, 2, object()]}).to_json(
                default_handler=my_handler_raises
            )
        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": [1, 2, complex(4, -5)]}).to_json(
                default_handler=my_handler_raises
            )

    def test_categorical(self):
        # GH4377 df.to_json segfaults with non-ndarray blocks
        df = DataFrame({"A": ["a", "b", "c", "a", "b", "b", "a"]})
        df["B"] = df["A"]
        expected = df.to_json()

        df["B"] = df["A"].astype("category")
        assert expected == df.to_json()

        s = df["A"]
        sc = df["B"]
        assert s.to_json() == sc.to_json()

    def test_datetime_tz(self):
        # GH4377 df.to_json segfaults with non-ndarray blocks
        tz_range = date_range("20130101", periods=3, tz="US/Eastern")
        tz_naive = tz_range.tz_convert("utc").tz_localize(None)

        df = DataFrame({"A": tz_range, "B": date_range("20130101", periods=3)})

        df_naive = df.copy()
        df_naive["A"] = tz_naive
        expected = df_naive.to_json()
        assert expected == df.to_json()

        stz = Series(tz_range)
        s_naive = Series(tz_naive)
        assert stz.to_json() == s_naive.to_json()

    def test_sparse(self):
        # GH4377 df.to_json segfaults with non-ndarray blocks
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        df.loc[:8] = np.nan

        sdf = df.astype("Sparse")
        expected = df.to_json()
        assert expected == sdf.to_json()

        s = Series(np.random.default_rng(2).standard_normal(10))
        s.loc[:8] = np.nan
        ss = s.astype("Sparse")

        expected = s.to_json()
        assert expected == ss.to_json()

    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("2013-01-10 05:00:00Z"),
            Timestamp("2013-01-10 00:00:00", tz="US/Eastern"),
            Timestamp("2013-01-10 00:00:00-0500"),
        ],
    )
    def test_tz_is_utc(self, ts):
        exp = '"2013-01-10T05:00:00.000Z"'

        assert ujson_dumps(ts, iso_dates=True) == exp
        dt = ts.to_pydatetime()
        assert ujson_dumps(dt, iso_dates=True) == exp

    def test_tz_is_naive(self):
        ts = Timestamp("2013-01-10 05:00:00")
        exp = '"2013-01-10T05:00:00.000"'

        assert ujson_dumps(ts, iso_dates=True) == exp
        dt = ts.to_pydatetime()
        assert ujson_dumps(dt, iso_dates=True) == exp

    @pytest.mark.parametrize(
        "tz_range",
        [
            date_range("2013-01-01 05:00:00Z", periods=2),
            date_range("2013-01-01 00:00:00", periods=2, tz="US/Eastern"),
            date_range("2013-01-01 00:00:00-0500", periods=2),
        ],
    )
    def test_tz_range_is_utc(self, tz_range):
        exp = '["2013-01-01T05:00:00.000Z","2013-01-02T05:00:00.000Z"]'
        dfexp = (
            '{"DT":{'
            '"0":"2013-01-01T05:00:00.000Z",'
            '"1":"2013-01-02T05:00:00.000Z"}}'
        )

        assert ujson_dumps(tz_range, iso_dates=True) == exp
        dti = DatetimeIndex(tz_range)
        # Ensure datetimes in object array are serialized correctly
        # in addition to the normal DTI case
        assert ujson_dumps(dti, iso_dates=True) == exp
        assert ujson_dumps(dti.astype(object), iso_dates=True) == exp
        df = DataFrame({"DT": dti})
        result = ujson_dumps(df, iso_dates=True)
        assert result == dfexp
        assert ujson_dumps(df.astype({"DT": object}), iso_dates=True)

    def test_tz_range_is_naive(self):
        dti = date_range("2013-01-01 05:00:00", periods=2)

        exp = '["2013-01-01T05:00:00.000","2013-01-02T05:00:00.000"]'
        dfexp = '{"DT":{"0":"2013-01-01T05:00:00.000","1":"2013-01-02T05:00:00.000"}}'

        # Ensure datetimes in object array are serialized correctly
        # in addition to the normal DTI case
        assert ujson_dumps(dti, iso_dates=True) == exp
        assert ujson_dumps(dti.astype(object), iso_dates=True) == exp
        df = DataFrame({"DT": dti})
        result = ujson_dumps(df, iso_dates=True)
        assert result == dfexp
        assert ujson_dumps(df.astype({"DT": object}), iso_dates=True)

    def test_read_inline_jsonl(self):
        # GH9180

        result = read_json(StringIO('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n'), lines=True)
        expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.single_cpu
    @td.skip_if_not_us_locale
    def test_read_s3_jsonl(self, s3_public_bucket_with_data, s3so):
        # GH17200

        result = read_json(
            f"s3n://{s3_public_bucket_with_data.name}/items.jsonl",
            lines=True,
            storage_options=s3so,
        )
        expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_read_local_jsonl(self):
        # GH17200
        with tm.ensure_clean("tmp_items.json") as path:
            with open(path, "w", encoding="utf-8") as infile:
                infile.write('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n')
            result = read_json(path, lines=True)
            expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
            tm.assert_frame_equal(result, expected)

    def test_read_jsonl_unicode_chars(self):
        # GH15132: non-ascii unicode characters
        # \u201d == RIGHT DOUBLE QUOTATION MARK

        # simulate file handle
        json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
        json = StringIO(json)
        result = read_json(json, lines=True)
        expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

        # simulate string
        json = StringIO('{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n')
        result = read_json(json, lines=True)
        expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("bigNum", [sys.maxsize + 1, -(sys.maxsize + 2)])
    def test_to_json_large_numbers(self, bigNum):
        # GH34473
        series = Series(bigNum, dtype=object, index=["articleId"])
        json = series.to_json()
        expected = '{"articleId":' + str(bigNum) + "}"
        assert json == expected

        df = DataFrame(bigNum, dtype=object, index=["articleId"], columns=[0])
        json = df.to_json()
        expected = '{"0":{"articleId":' + str(bigNum) + "}}"
        assert json == expected

    @pytest.mark.parametrize("bigNum", [-(2**63) - 1, 2**64])
    def test_read_json_large_numbers(self, bigNum):
        # GH20599, 26068
        json = StringIO('{"articleId":' + str(bigNum) + "}")
        msg = r"Value is too small|Value is too big"
        with pytest.raises(ValueError, match=msg):
            read_json(json)

        json = StringIO('{"0":{"articleId":' + str(bigNum) + "}}")
        with pytest.raises(ValueError, match=msg):
            read_json(json)

    def test_read_json_large_numbers2(self):
        # GH18842
        json = '{"articleId": "1404366058080022500245"}'
        json = StringIO(json)
        result = read_json(json, typ="series")
        expected = Series(1.404366e21, index=["articleId"])
        tm.assert_series_equal(result, expected)

        json = '{"0": {"articleId": "1404366058080022500245"}}'
        json = StringIO(json)
        result = read_json(json)
        expected = DataFrame(1.404366e21, index=["articleId"], columns=[0])
        tm.assert_frame_equal(result, expected)

    def test_to_jsonl(self):
        # GH9180
        df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
        result = df.to_json(orient="records", lines=True)
        expected = '{"a":1,"b":2}\n{"a":1,"b":2}\n'
        assert result == expected

        df = DataFrame([["foo}", "bar"], ['foo"', "bar"]], columns=["a", "b"])
        result = df.to_json(orient="records", lines=True)
        expected = '{"a":"foo}","b":"bar"}\n{"a":"foo\\"","b":"bar"}\n'
        assert result == expected
        tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)

        # GH15096: escaped characters in columns and data
        df = DataFrame([["foo\\", "bar"], ['foo"', "bar"]], columns=["a\\", "b"])
        result = df.to_json(orient="records", lines=True)
        expected = '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n'
        assert result == expected

        tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)

    # TODO: there is a near-identical test for pytables; can we share?
    @pytest.mark.xfail(reason="GH#13774 encoding kwarg not supported", raises=TypeError)
    @pytest.mark.parametrize(
        "val",
        [
            [b"E\xc9, 17", b"", b"a", b"b", b"c"],
            [b"E\xc9, 17", b"a", b"b", b"c"],
            [b"EE, 17", b"", b"a", b"b", b"c"],
            [b"E\xc9, 17", b"\xf8\xfc", b"a", b"b", b"c"],
            [b"", b"a", b"b", b"c"],
            [b"\xf8\xfc", b"a", b"b", b"c"],
            [b"A\xf8\xfc", b"", b"a", b"b", b"c"],
            [np.nan, b"", b"b", b"c"],
            [b"A\xf8\xfc", np.nan, b"", b"b", b"c"],
        ],
    )
    @pytest.mark.parametrize("dtype", ["category", object])
    def test_latin_encoding(self, dtype, val):
        # GH 13774
        ser = Series(
            [x.decode("latin-1") if isinstance(x, bytes) else x for x in val],
            dtype=dtype,
        )
        encoding = "latin-1"
        with tm.ensure_clean("test.json") as path:
            ser.to_json(path, encoding=encoding)
            retr = read_json(StringIO(path), encoding=encoding)
            tm.assert_series_equal(ser, retr, check_categorical=False)

    def test_data_frame_size_after_to_json(self):
        # GH15344
        df = DataFrame({"a": [str(1)]})

        size_before = df.memory_usage(index=True, deep=True).sum()
        df.to_json()
        size_after = df.memory_usage(index=True, deep=True).sum()

        assert size_before == size_after

    @pytest.mark.parametrize(
        "index", [None, [1, 2], [1.0, 2.0], ["a", "b"], ["1", "2"], ["1.", "2."]]
    )
    @pytest.mark.parametrize("columns", [["a", "b"], ["1", "2"], ["1.", "2."]])
    def test_from_json_to_json_table_index_and_columns(self, index, columns):
        # GH25433 GH25435
        expected = DataFrame([[1, 2], [3, 4]], index=index, columns=columns)
        dfjson = expected.to_json(orient="table")

        result = read_json(StringIO(dfjson), orient="table")
        tm.assert_frame_equal(result, expected)

    def test_from_json_to_json_table_dtypes(self):
        # GH21345
        expected = DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["5", "6"]})
        dfjson = expected.to_json(orient="table")
        result = read_json(StringIO(dfjson), orient="table")
        tm.assert_frame_equal(result, expected)

    # TODO: We are casting to string which coerces None to NaN before casting back
    # to object, ending up with incorrect na values
    @pytest.mark.xfail(using_string_dtype(), reason="incorrect na conversion")
    @pytest.mark.parametrize("orient", ["split", "records", "index", "columns"])
    def test_to_json_from_json_columns_dtypes(self, orient):
        # GH21892 GH33205
        expected = DataFrame.from_dict(
            {
                "Integer": Series([1, 2, 3], dtype="int64"),
                "Float": Series([None, 2.0, 3.0], dtype="float64"),
                "Object": Series([None, "", "c"], dtype="object"),
                "Bool": Series([True, False, True], dtype="bool"),
                "Category": Series(["a", "b", None], dtype="category"),
                "Datetime": Series(
                    ["2020-01-01", None, "2020-01-03"], dtype="datetime64[ns]"
                ),
            }
        )
        dfjson = expected.to_json(orient=orient)

        result = read_json(
            StringIO(dfjson),
            orient=orient,
            dtype={
                "Integer": "int64",
                "Float": "float64",
                "Object": "object",
                "Bool": "bool",
                "Category": "category",
                "Datetime": "datetime64[ns]",
            },
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", [True, {"b": int, "c": int}])
    def test_read_json_table_dtype_raises(self, dtype):
        # GH21345
        df = DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["5", "6"]})
        dfjson = df.to_json(orient="table")
        msg = "cannot pass both dtype and orient='table'"
        with pytest.raises(ValueError, match=msg):
            read_json(dfjson, orient="table", dtype=dtype)

    @pytest.mark.parametrize("orient", ["index", "columns", "records", "values"])
    def test_read_json_table_empty_axes_dtype(self, orient):
        # GH28558

        expected = DataFrame()
        result = read_json(StringIO("{}"), orient=orient, convert_axes=True)
        tm.assert_index_equal(result.index, expected.index)
        tm.assert_index_equal(result.columns, expected.columns)

    def test_read_json_table_convert_axes_raises(self):
        # GH25433 GH25435
        df = DataFrame([[1, 2], [3, 4]], index=[1.0, 2.0], columns=["1.", "2."])
        dfjson = df.to_json(orient="table")
        msg = "cannot pass both convert_axes and orient='table'"
        with pytest.raises(ValueError, match=msg):
            read_json(dfjson, orient="table", convert_axes=True)

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                DataFrame([[1, 2], [4, 5]], columns=["a", "b"]),
                {"columns": ["a", "b"], "data": [[1, 2], [4, 5]]},
            ),
            (
                DataFrame([[1, 2], [4, 5]], columns=["a", "b"]).rename_axis("foo"),
                {"columns": ["a", "b"], "data": [[1, 2], [4, 5]]},
            ),
            (
                DataFrame(
                    [[1, 2], [4, 5]], columns=["a", "b"], index=[["a", "b"], ["c", "d"]]
                ),
                {"columns": ["a", "b"], "data": [[1, 2], [4, 5]]},
            ),
            (Series([1, 2, 3], name="A"), {"name": "A", "data": [1, 2, 3]}),
            (
                Series([1, 2, 3], name="A").rename_axis("foo"),
                {"name": "A", "data": [1, 2, 3]},
            ),
            (
                Series([1, 2], name="A", index=[["a", "b"], ["c", "d"]]),
                {"name": "A", "data": [1, 2]},
            ),
        ],
    )
    def test_index_false_to_json_split(self, data, expected):
        # GH 17394
        # Testing index=False in to_json with orient='split'

        result = data.to_json(orient="split", index=False)
        result = json.loads(result)

        assert result == expected

    @pytest.mark.parametrize(
        "data",
        [
            (DataFrame([[1, 2], [4, 5]], columns=["a", "b"])),
            (DataFrame([[1, 2], [4, 5]], columns=["a", "b"]).rename_axis("foo")),
            (
                DataFrame(
                    [[1, 2], [4, 5]], columns=["a", "b"], index=[["a", "b"], ["c", "d"]]
                )
            ),
            (Series([1, 2, 3], name="A")),
            (Series([1, 2, 3], name="A").rename_axis("foo")),
            (Series([1, 2], name="A", index=[["a", "b"], ["c", "d"]])),
        ],
    )
    def test_index_false_to_json_table(self, data):
        # GH 17394
        # Testing index=False in to_json with orient='table'

        result = data.to_json(orient="table", index=False)
        result = json.loads(result)

        expected = {
            "schema": pd.io.json.build_table_schema(data, index=False),
            "data": DataFrame(data).to_dict(orient="records"),
        }

        assert result == expected

    @pytest.mark.parametrize("orient", ["index", "columns"])
    def test_index_false_error_to_json(self, orient):
        # GH 17394, 25513
        # Testing error message from to_json with index=False

        df = DataFrame([[1, 2], [4, 5]], columns=["a", "b"])

        msg = (
            "'index=False' is only valid when 'orient' is 'split', "
            "'table', 'records', or 'values'"
        )
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient, index=False)

    @pytest.mark.parametrize("orient", ["records", "values"])
    def test_index_true_error_to_json(self, orient):
        # GH 25513
        # Testing error message from to_json with index=True

        df = DataFrame([[1, 2], [4, 5]], columns=["a", "b"])

        msg = (
            "'index=True' is only valid when 'orient' is 'split', "
            "'table', 'index', or 'columns'"
        )
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient, index=True)

    @pytest.mark.parametrize("orient", ["split", "table"])
    @pytest.mark.parametrize("index", [True, False])
    def test_index_false_from_json_to_json(self, orient, index):
        # GH25170
        # Test index=False in from_json to_json
        expected = DataFrame({"a": [1, 2], "b": [3, 4]})
        dfjson = expected.to_json(orient=orient, index=index)
        result = read_json(StringIO(dfjson), orient=orient)
        tm.assert_frame_equal(result, expected)

    def test_read_timezone_information(self):
        # GH 25546
        result = read_json(
            StringIO('{"2019-01-01T11:00:00.000Z":88}'), typ="series", orient="index"
        )
        exp_dti = DatetimeIndex(["2019-01-01 11:00:00"], dtype="M8[ns, UTC]")
        expected = Series([88], index=exp_dti)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "url",
        [
            "s3://example-fsspec/",
            "gcs://another-fsspec/file.json",
            "https://example-site.com/data",
            "some-protocol://data.txt",
        ],
    )
    def test_read_json_with_url_value(self, url):
        # GH 36271
        result = read_json(StringIO(f'{{"url":{{"0":"{url}"}}}}'))
        expected = DataFrame({"url": [url]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "compression",
        ["", ".gz", ".bz2", ".tar"],
    )
    def test_read_json_with_very_long_file_path(self, compression):
        # GH 46718
        long_json_path = f'{"a" * 1000}.json{compression}'
        with pytest.raises(
            FileNotFoundError, match=f"File {long_json_path} does not exist"
        ):
            # path too long for Windows is handled in file_exists() but raises in
            # _get_data_from_filepath()
            read_json(long_json_path)

    @pytest.mark.parametrize(
        "date_format,key", [("epoch", 86400000), ("iso", "P1DT0H0M0S")]
    )
    def test_timedelta_as_label(self, date_format, key):
        df = DataFrame([[1]], columns=[pd.Timedelta("1D")])
        expected = f'{{"{key}":{{"0":1}}}}'
        result = df.to_json(date_format=date_format)

        assert result == expected

    @pytest.mark.parametrize(
        "orient,expected",
        [
            ("index", "{\"('a', 'b')\":{\"('c', 'd')\":1}}"),
            ("columns", "{\"('c', 'd')\":{\"('a', 'b')\":1}}"),
            # TODO: the below have separate encoding procedures
            pytest.param(
                "split",
                "",
                marks=pytest.mark.xfail(
                    reason="Produces JSON but not in a consistent manner"
                ),
            ),
            pytest.param(
                "table",
                "",
                marks=pytest.mark.xfail(
                    reason="Produces JSON but not in a consistent manner"
                ),
            ),
        ],
    )
    def test_tuple_labels(self, orient, expected):
        # GH 20500
        df = DataFrame([[1]], index=[("a", "b")], columns=[("c", "d")])
        result = df.to_json(orient=orient)
        assert result == expected

    @pytest.mark.parametrize("indent", [1, 2, 4])
    def test_to_json_indent(self, indent):
        # GH 12004
        df = DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["a", "b"])

        result = df.to_json(indent=indent)
        spaces = " " * indent
        expected = f"""{{
{spaces}"a":{{
{spaces}{spaces}"0":"foo",
{spaces}{spaces}"1":"baz"
{spaces}}},
{spaces}"b":{{
{spaces}{spaces}"0":"bar",
{spaces}{spaces}"1":"qux"
{spaces}}}
}}"""

        assert result == expected

    @pytest.mark.skipif(
        using_string_dtype(),
        reason="Adjust expected when infer_string is default, no bug here, "
        "just a complicated parametrization",
    )
    @pytest.mark.parametrize(
        "orient,expected",
        [
            (
                "split",
                """{
    "columns":[
        "a",
        "b"
    ],
    "index":[
        0,
        1
    ],
    "data":[
        [
            "foo",
            "bar"
        ],
        [
            "baz",
            "qux"
        ]
    ]
}""",
            ),
            (
                "records",
                """[
    {
        "a":"foo",
        "b":"bar"
    },
    {
        "a":"baz",
        "b":"qux"
    }
]""",
            ),
            (
                "index",
                """{
    "0":{
        "a":"foo",
        "b":"bar"
    },
    "1":{
        "a":"baz",
        "b":"qux"
    }
}""",
            ),
            (
                "columns",
                """{
    "a":{
        "0":"foo",
        "1":"baz"
    },
    "b":{
        "0":"bar",
        "1":"qux"
    }
}""",
            ),
            (
                "values",
                """[
    [
        "foo",
        "bar"
    ],
    [
        "baz",
        "qux"
    ]
]""",
            ),
            (
                "table",
                """{
    "schema":{
        "fields":[
            {
                "name":"index",
                "type":"integer"
            },
            {
                "name":"a",
                "type":"string"
            },
            {
                "name":"b",
                "type":"string"
            }
        ],
        "primaryKey":[
            "index"
        ],
        "pandas_version":"1.4.0"
    },
    "data":[
        {
            "index":0,
            "a":"foo",
            "b":"bar"
        },
        {
            "index":1,
            "a":"baz",
            "b":"qux"
        }
    ]
}""",
            ),
        ],
    )
    def test_json_indent_all_orients(self, orient, expected):
        # GH 12004
        df = DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["a", "b"])
        result = df.to_json(orient=orient, indent=4)
        assert result == expected

    def test_json_negative_indent_raises(self):
        with pytest.raises(ValueError, match="must be a nonnegative integer"):
            DataFrame().to_json(indent=-1)

    def test_emca_262_nan_inf_support(self):
        # GH 12213
        data = StringIO(
            '["a", NaN, "NaN", Infinity, "Infinity", -Infinity, "-Infinity"]'
        )
        result = read_json(data)
        expected = DataFrame(
            ["a", None, "NaN", np.inf, "Infinity", -np.inf, "-Infinity"]
        )
        tm.assert_frame_equal(result, expected)

    def test_frame_int_overflow(self):
        # GH 30320
        encoded_json = json.dumps([{"col": "31900441201190696999"}, {"col": "Text"}])
        expected = DataFrame({"col": ["31900441201190696999", "Text"]})
        result = read_json(StringIO(encoded_json))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "dataframe,expected",
        [
            (
                DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}),
                '{"(0, \'x\')":1,"(0, \'y\')":"a","(1, \'x\')":2,'
                '"(1, \'y\')":"b","(2, \'x\')":3,"(2, \'y\')":"c"}',
            )
        ],
    )
    def test_json_multiindex(self, dataframe, expected):
        series = dataframe.stack(future_stack=True)
        result = series.to_json(orient="index")
        assert result == expected

    @pytest.mark.single_cpu
    def test_to_s3(self, s3_public_bucket, s3so):
        # GH 28375
        mock_bucket_name, target_file = s3_public_bucket.name, "test.json"
        df = DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        df.to_json(f"s3://{mock_bucket_name}/{target_file}", storage_options=s3so)
        timeout = 5
        while True:
            if target_file in (obj.key for obj in s3_public_bucket.objects.all()):
                break
            time.sleep(0.1)
            timeout -= 0.1
            assert timeout > 0, "Timed out waiting for file to appear on moto"

    def test_json_pandas_nulls(self, nulls_fixture, request):
        # GH 31615
        if isinstance(nulls_fixture, Decimal):
            mark = pytest.mark.xfail(reason="not implemented")
            request.applymarker(mark)

        result = DataFrame([[nulls_fixture]]).to_json()
        assert result == '{"0":{"0":null}}'

    def test_readjson_bool_series(self):
        # GH31464
        result = read_json(StringIO("[true, true, false]"), typ="series")
        expected = Series([True, True, False])
        tm.assert_series_equal(result, expected)

    def test_to_json_multiindex_escape(self):
        # GH 15273
        df = DataFrame(
            True,
            index=date_range("2017-01-20", "2017-01-23"),
            columns=["foo", "bar"],
        ).stack(future_stack=True)
        result = df.to_json()
        expected = (
            "{\"(Timestamp('2017-01-20 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-20 00:00:00'), 'bar')\":true,"
            "\"(Timestamp('2017-01-21 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-21 00:00:00'), 'bar')\":true,"
            "\"(Timestamp('2017-01-22 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-22 00:00:00'), 'bar')\":true,"
            "\"(Timestamp('2017-01-23 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-23 00:00:00'), 'bar')\":true}"
        )
        assert result == expected

    def test_to_json_series_of_objects(self):
        class _TestObject:
            def __init__(self, a, b, _c, d) -> None:
                self.a = a
                self.b = b
                self._c = _c
                self.d = d

            def e(self):
                return 5

        # JSON keys should be all non-callable non-underscore attributes, see GH-42768
        series = Series([_TestObject(a=1, b=2, _c=3, d=4)])
        assert json.loads(series.to_json()) == {"0": {"a": 1, "b": 2, "d": 4}}

    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                Series({0: -6 + 8j, 1: 0 + 1j, 2: 9 - 5j}),
                '{"0":{"imag":8.0,"real":-6.0},'
                '"1":{"imag":1.0,"real":0.0},'
                '"2":{"imag":-5.0,"real":9.0}}',
            ),
            (
                Series({0: -9.39 + 0.66j, 1: 3.95 + 9.32j, 2: 4.03 - 0.17j}),
                '{"0":{"imag":0.66,"real":-9.39},'
                '"1":{"imag":9.32,"real":3.95},'
                '"2":{"imag":-0.17,"real":4.03}}',
            ),
            (
                DataFrame([[-2 + 3j, -1 - 0j], [4 - 3j, -0 - 10j]]),
                '{"0":{"0":{"imag":3.0,"real":-2.0},'
                '"1":{"imag":-3.0,"real":4.0}},'
                '"1":{"0":{"imag":0.0,"real":-1.0},'
                '"1":{"imag":-10.0,"real":0.0}}}',
            ),
            (
                DataFrame(
                    [[-0.28 + 0.34j, -1.08 - 0.39j], [0.41 - 0.34j, -0.78 - 1.35j]]
                ),
                '{"0":{"0":{"imag":0.34,"real":-0.28},'
                '"1":{"imag":-0.34,"real":0.41}},'
                '"1":{"0":{"imag":-0.39,"real":-1.08},'
                '"1":{"imag":-1.35,"real":-0.78}}}',
            ),
        ],
    )
    def test_complex_data_tojson(self, data, expected):
        # GH41174
        result = data.to_json()
        assert result == expected

    def test_json_uint64(self):
        # GH21073
        expected = (
            '{"columns":["col1"],"index":[0,1],'
            '"data":[[13342205958987758245],[12388075603347835679]]}'
        )
        df = DataFrame(data={"col1": [13342205958987758245, 12388075603347835679]})
        result = df.to_json(orient="split")
        assert result == expected

    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)", strict=False)
    def test_read_json_dtype_backend(
        self, string_storage, dtype_backend, orient, using_infer_string
    ):
        # GH#50750
        df = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": [True, False, None],
                "f": [True, False, True],
                "g": ["a", "b", "c"],
                "h": ["a", "b", None],
            }
        )

        out = df.to_json(orient=orient)
        with pd.option_context("mode.string_storage", string_storage):
            result = read_json(
                StringIO(out), dtype_backend=dtype_backend, orient=orient
            )

        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")
            string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)

        expected = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": Series([True, False, NA], dtype="boolean"),
                "f": Series([True, False, True], dtype="boolean"),
                "g": Series(["a", "b", "c"], dtype=string_dtype),
                "h": Series(["a", "b", None], dtype=string_dtype),
            }
        )

        if dtype_backend == "pyarrow":
            pa = pytest.importorskip("pyarrow")
            from pandas.arrays import ArrowExtensionArray

            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )

        if orient == "values":
            expected.columns = list(range(8))

        # the storage of the str columns' Index is also affected by the
        # string_storage setting -> ignore that for checking the result
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.parametrize("orient", ["split", "records", "index"])
    def test_read_json_nullable_series(self, string_storage, dtype_backend, orient):
        # GH#50750
        pa = pytest.importorskip("pyarrow")
        ser = Series([1, np.nan, 3], dtype="Int64")

        out = ser.to_json(orient=orient)
        with pd.option_context("mode.string_storage", string_storage):
            result = read_json(
                StringIO(out), dtype_backend=dtype_backend, orient=orient, typ="series"
            )

        expected = Series([1, np.nan, 3], dtype="Int64")

        if dtype_backend == "pyarrow":
            from pandas.arrays import ArrowExtensionArray

            expected = Series(ArrowExtensionArray(pa.array(expected, from_pandas=True)))

        tm.assert_series_equal(result, expected)

    def test_invalid_dtype_backend(self):
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        with pytest.raises(ValueError, match=msg):
            read_json("test", dtype_backend="numpy")


def test_invalid_engine():
    # GH 48893
    ser = Series(range(1))
    out = ser.to_json()
    with pytest.raises(ValueError, match="The engine type foo"):
        read_json(out, engine="foo")


def test_pyarrow_engine_lines_false():
    # GH 48893
    ser = Series(range(1))
    out = ser.to_json()
    with pytest.raises(ValueError, match="currently pyarrow engine only supports"):
        read_json(out, engine="pyarrow", lines=False)


def test_json_roundtrip_string_inference(orient):
    df = DataFrame(
        [["a", "b"], ["c", "d"]], index=["row 1", "row 2"], columns=["col 1", "col 2"]
    )
    out = df.to_json()
    with pd.option_context("future.infer_string", True):
        result = read_json(StringIO(out))
    dtype = pd.StringDtype(na_value=np.nan)
    expected = DataFrame(
        [["a", "b"], ["c", "d"]],
        dtype=dtype,
        index=Index(["row 1", "row 2"], dtype=dtype),
        columns=Index(["col 1", "col 2"], dtype=dtype),
    )
    tm.assert_frame_equal(result, expected)


def test_json_pos_args_deprecation():
    # GH-54229
    df = DataFrame({"a": [1, 2, 3]})
    msg = (
        r"Starting with pandas version 3.0 all arguments of to_json except for the "
        r"argument 'path_or_buf' will be keyword-only."
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        buf = BytesIO()
        df.to_json(buf, "split")


@td.skip_if_no("pyarrow")
def test_to_json_ea_null():
    # GH#57224
    df = DataFrame(
        {
            "a": Series([1, NA], dtype="int64[pyarrow]"),
            "b": Series([2, NA], dtype="Int64"),
        }
    )
    result = df.to_json(orient="records", lines=True)
    expected = """{"a":1,"b":2}
{"a":null,"b":null}
"""
    assert result == expected


def test_read_json_lines_rangeindex():
    # GH 57429
    data = """
{"a": 1, "b": 2}
{"a": 3, "b": 4}
"""
    result = read_json(StringIO(data), lines=True).index
    expected = RangeIndex(2)
    tm.assert_index_equal(result, expected, exact=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_integrals.py ===
import math
from sympy.concrete.summations import (Sum, summation)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, Lambda, diff)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, I, Rational, nan, oo, pi, zoo, all_close)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, im, polar_lift, re, sign)
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (acosh, asinh, cosh, coth, csch, sinh, tanh, sech)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, sinc, tan, sec)
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import (Ci, Ei, Si, erf, erfc, erfi, fresnelc, li)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import lerchphi
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import (Idx, IndexedBase)
from sympy.core.expr import unchanged
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import Integral
from sympy.integrals.risch import NonElementaryIntegral
from sympy.physics import units
from sympy.testing.pytest import raises, slow, warns_deprecated_sympy, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically


x, y, z, a, b, c, d, e, s, t, x_1, x_2 = symbols('x y z a b c d e s t x_1 x_2')
n = Symbol('n', integer=True)
f = Function('f')


def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)


def test_poly_deprecated():
    p = Poly(2*x, x)
    assert p.integrate(x) == Poly(x**2, x, domain='QQ')
    # The stacklevel is based on Integral(Poly)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        integrate(p, x)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        Integral(p, (x,))


@slow
def test_principal_value():
    g = 1 / x
    assert Integral(g, (x, -oo, oo)).principal_value() == 0
    assert Integral(g, (y, -oo, oo)).principal_value() == oo * sign(1 / x)
    raises(ValueError, lambda: Integral(g, (x)).principal_value())
    raises(ValueError, lambda: Integral(g).principal_value())

    l = 1 / ((x ** 3) - 1)
    assert Integral(l, (x, -oo, oo)).principal_value().together() == -sqrt(3)*pi/3
    raises(ValueError, lambda: Integral(l, (x, -oo, 1)).principal_value())

    d = 1 / (x ** 2 - 1)
    assert Integral(d, (x, -oo, oo)).principal_value() == 0
    assert Integral(d, (x, -2, 2)).principal_value() == -log(3)

    v = x / (x ** 2 - 1)
    assert Integral(v, (x, -oo, oo)).principal_value() == 0
    assert Integral(v, (x, -2, 2)).principal_value() == 0

    s = x ** 2 / (x ** 2 - 1)
    assert Integral(s, (x, -oo, oo)).principal_value() is oo
    assert Integral(s, (x, -2, 2)).principal_value() == -log(3) + 4

    f = 1 / ((x ** 2 - 1) * (1 + x ** 2))
    assert Integral(f, (x, -oo, oo)).principal_value() == -pi / 2
    assert Integral(f, (x, -2, 2)).principal_value() == -atan(2) - log(3) / 2


def diff_test(i):
    """Return the set of symbols, s, which were used in testing that
    i.diff(s) agrees with i.doit().diff(s). If there is an error then
    the assertion will fail, causing the test to fail."""
    syms = i.free_symbols
    for s in syms:
        assert (i.diff(s).doit() - i.doit().diff(s)).expand() == 0
    return syms


def test_improper_integral():
    assert integrate(log(x), (x, 0, 1)) == -1
    assert integrate(x**(-2), (x, 1, oo)) == 1
    assert integrate(1/(1 + exp(x)), (x, 0, oo)) == log(2)


def test_constructor():
    # this is shared by Sum, so testing Integral's constructor
    # is equivalent to testing Sum's
    s1 = Integral(n, n)
    assert s1.limits == (Tuple(n),)
    s2 = Integral(n, (n,))
    assert s2.limits == (Tuple(n),)
    s3 = Integral(Sum(x, (x, 1, y)))
    assert s3.limits == (Tuple(y),)
    s4 = Integral(n, Tuple(n,))
    assert s4.limits == (Tuple(n),)

    s5 = Integral(n, (n, Interval(1, 2)))
    assert s5.limits == (Tuple(n, 1, 2),)

    # Testing constructor with inequalities:
    s6 = Integral(n, n > 10)
    assert s6.limits == (Tuple(n, 10, oo),)
    s7 = Integral(n, (n > 2) & (n < 5))
    assert s7.limits == (Tuple(n, 2, 5),)


def test_basics():

    assert Integral(0, x) != 0
    assert Integral(x, (x, 1, 1)) != 0
    assert Integral(oo, x) != oo
    assert Integral(S.NaN, x) is S.NaN

    assert diff(Integral(y, y), x) == 0
    assert diff(Integral(x, (x, 0, 1)), x) == 0
    assert diff(Integral(x, x), x) == x
    assert diff(Integral(t, (t, 0, x)), x) == x

    e = (t + 1)**2
    assert diff(integrate(e, (t, 0, x)), x) == \
        diff(Integral(e, (t, 0, x)), x).doit().expand() == \
        ((1 + x)**2).expand()
    assert diff(integrate(e, (t, 0, x)), t) == \
        diff(Integral(e, (t, 0, x)), t) == 0
    assert diff(integrate(e, (t, 0, x)), a) == \
        diff(Integral(e, (t, 0, x)), a) == 0
    assert diff(integrate(e, t), a) == diff(Integral(e, t), a) == 0

    assert integrate(e, (t, a, x)).diff(x) == \
        Integral(e, (t, a, x)).diff(x).doit().expand()
    assert Integral(e, (t, a, x)).diff(x).doit() == ((1 + x)**2)
    assert integrate(e, (t, x, a)).diff(x).doit() == (-(1 + x)**2).expand()

    assert integrate(t**2, (t, x, 2*x)).diff(x) == 7*x**2

    assert Integral(x, x).atoms() == {x}
    assert Integral(f(x), (x, 0, 1)).atoms() == {S.Zero, S.One, x}

    assert diff_test(Integral(x, (x, 3*y))) == {y}
    assert diff_test(Integral(x, (a, 3*y))) == {x, y}

    assert integrate(x, (x, oo, oo)) == 0 #issue 8171
    assert integrate(x, (x, -oo, -oo)) == 0

    # sum integral of terms
    assert integrate(y + x + exp(x), x) == x*y + x**2/2 + exp(x)

    assert Integral(x).is_commutative
    n = Symbol('n', commutative=False)
    assert Integral(n + x, x).is_commutative is False


def test_diff_wrt():
    class Test(Expr):
        _diff_wrt = True
        is_commutative = True

    t = Test()
    assert integrate(t + 1, t) == t**2/2 + t
    assert integrate(t + 1, (t, 0, 1)) == Rational(3, 2)

    raises(ValueError, lambda: integrate(x + 1, x + 1))
    raises(ValueError, lambda: integrate(x + 1, (x + 1, 0, 1)))


def test_basics_multiple():
    assert diff_test(Integral(x, (x, 3*x, 5*y), (y, x, 2*x))) == {x}
    assert diff_test(Integral(x, (x, 5*y), (y, x, 2*x))) == {x}
    assert diff_test(Integral(x, (x, 5*y), (y, y, 2*x))) == {x, y}
    assert diff_test(Integral(y, y, x)) == {x, y}
    assert diff_test(Integral(y*x, x, y)) == {x, y}
    assert diff_test(Integral(x + y, y, (y, 1, x))) == {x}
    assert diff_test(Integral(x + y, (x, x, y), (y, y, x))) == {x, y}


def test_conjugate_transpose():
    A, B = symbols("A B", commutative=False)

    x = Symbol("x", complex=True)
    p = Integral(A*B, (x,))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()

    x = Symbol("x", real=True)
    p = Integral(A*B, (x,))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()


def test_integration():
    assert integrate(0, (t, 0, x)) == 0
    assert integrate(3, (t, 0, x)) == 3*x
    assert integrate(t, (t, 0, x)) == x**2/2
    assert integrate(3*t, (t, 0, x)) == 3*x**2/2
    assert integrate(3*t**2, (t, 0, x)) == x**3
    assert integrate(1/t, (t, 1, x)) == log(x)
    assert integrate(-1/t**2, (t, 1, x)) == 1/x - 1
    assert integrate(t**2 + 5*t - 8, (t, 0, x)) == x**3/3 + 5*x**2/2 - 8*x
    assert integrate(x**2, x) == x**3/3
    assert integrate((3*t*x)**5, x) == (3*t)**5 * x**6 / 6

    b = Symbol("b")
    c = Symbol("c")
    assert integrate(a*t, (t, 0, x)) == a*x**2/2
    assert integrate(a*t**4, (t, 0, x)) == a*x**5/5
    assert integrate(a*t**2 + b*t + c, (t, 0, x)) == a*x**3/3 + b*x**2/2 + c*x


def test_multiple_integration():
    assert integrate((x**2)*(y**2), (x, 0, 1), (y, -1, 2)) == Rational(1)
    assert integrate((y**2)*(x**2), x, y) == Rational(1, 9)*(x**3)*(y**3)
    assert integrate(1/(x + 3)/(1 + x)**3, x) == \
        log(3 + x)*Rational(-1, 8) + log(1 + x)*Rational(1, 8) + x/(4 + 8*x + 4*x**2)
    assert integrate(sin(x*y)*y, (x, 0, 1), (y, 0, 1)) == -sin(1) + 1


def test_issue_3532():
    assert integrate(exp(-x), (x, 0, oo)) == 1


def test_issue_3560():
    assert integrate(sqrt(x)**3, x) == 2*sqrt(x)**5/5
    assert integrate(sqrt(x), x) == 2*sqrt(x)**3/3
    assert integrate(1/sqrt(x)**3, x) == -2/sqrt(x)


def test_issue_18038():
    raises(AttributeError, lambda: integrate((x, x)))


def test_integrate_poly():
    p = Poly(x + x**2*y + y**3, x, y)

    # The stacklevel is based on Integral(Poly)
    with warns_deprecated_sympy():
        qx = Integral(p, x)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        qx = integrate(p, x)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        qy = integrate(p, y)

    assert isinstance(qx, Poly) is True
    assert isinstance(qy, Poly) is True

    assert qx.gens == (x, y)
    assert qy.gens == (x, y)

    assert qx.as_expr() == x**2/2 + x**3*y/3 + x*y**3
    assert qy.as_expr() == x*y + x**2*y**2/2 + y**4/4


def test_integrate_poly_definite():
    p = Poly(x + x**2*y + y**3, x, y)

    with warns_deprecated_sympy():
        Qx = Integral(p, (x, 0, 1))
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        Qx = integrate(p, (x, 0, 1))
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        Qy = integrate(p, (y, 0, pi))

    assert isinstance(Qx, Poly) is True
    assert isinstance(Qy, Poly) is True

    assert Qx.gens == (y,)
    assert Qy.gens == (x,)

    assert Qx.as_expr() == S.Half + y/3 + y**3
    assert Qy.as_expr() == pi**4/4 + pi*x + pi**2*x**2/2


def test_integrate_omit_var():
    y = Symbol('y')

    assert integrate(x) == x**2/2

    raises(ValueError, lambda: integrate(2))
    raises(ValueError, lambda: integrate(x*y))


def test_integrate_poly_accurately():
    y = Symbol('y')
    assert integrate(x*sin(y), x) == x**2*sin(y)/2

    # when passed to risch_norman, this will be a CPU hog, so this really
    # checks, that integrated function is recognized as polynomial
    assert integrate(x**1000*sin(y), x) == x**1001*sin(y)/1001


def test_issue_3635():
    y = Symbol('y')
    assert integrate(x**2, y) == x**2*y
    assert integrate(x**2, (y, -1, 1)) == 2*x**2

# works in SymPy and py.test but hangs in `setup.py test`


def test_integrate_linearterm_pow():
    # check integrate((a*x+b)^c, x)  --  issue 3499
    y = Symbol('y', positive=True)
    # TODO: Remove conds='none' below, let the assumption take care of it.
    assert integrate(x**y, x, conds='none') == x**(y + 1)/(y + 1)
    assert integrate((exp(y)*x + 1/y)**(1 + sin(y)), x, conds='none') == \
        exp(-y)*(exp(y)*x + 1/y)**(2 + sin(y)) / (2 + sin(y))


def test_issue_3618():
    assert integrate(pi*sqrt(x), x) == 2*pi*sqrt(x)**3/3
    assert integrate(pi*sqrt(x) + E*sqrt(x)**3, x) == \
        2*pi*sqrt(x)**3/3 + 2*E *sqrt(x)**5/5


def test_issue_3623():
    assert integrate(cos((n + 1)*x), x) == Piecewise(
        (sin(x*(n + 1))/(n + 1), Ne(n + 1, 0)), (x, True))
    assert integrate(cos((n - 1)*x), x) == Piecewise(
        (sin(x*(n - 1))/(n - 1), Ne(n - 1, 0)), (x, True))
    assert integrate(cos((n + 1)*x) + cos((n - 1)*x), x) == \
        Piecewise((sin(x*(n - 1))/(n - 1), Ne(n - 1, 0)), (x, True)) + \
        Piecewise((sin(x*(n + 1))/(n + 1), Ne(n + 1, 0)), (x, True))


def test_issue_3664():
    n = Symbol('n', integer=True, nonzero=True)
    assert integrate(-1./2 * x * sin(n * pi * x/2), [x, -2, 0]) == \
        2.0*cos(pi*n)/(pi*n)
    assert integrate(x * sin(n * pi * x/2) * Rational(-1, 2), [x, -2, 0]) == \
        2*cos(pi*n)/(pi*n)


def test_issue_3679():
    # definite integration of rational functions gives wrong answers
    assert NS(Integral(1/(x**2 - 8*x + 17), (x, 2, 4))) == '1.10714871779409'


def test_issue_3686():  # remove this when fresnel integrals are implemented
    from sympy.core.function import expand_func
    from sympy.functions.special.error_functions import fresnels
    assert expand_func(integrate(sin(x**2), x)) == \
        sqrt(2)*sqrt(pi)*fresnels(sqrt(2)*x/sqrt(pi))/2


def test_integrate_units():
    m = units.m
    s = units.s
    assert integrate(x * m/s, (x, 1*s, 5*s)) == 12*m*s


def test_transcendental_functions():
    assert integrate(LambertW(2*x), x) == \
        -x + x*LambertW(2*x) + x/LambertW(2*x)


def test_log_polylog():
    assert integrate(log(1 - x)/x, (x, 0, 1)) == -pi**2/6
    assert integrate(log(x)*(1 - x)**(-1), (x, 0, 1)) == -pi**2/6


def test_issue_3740():
    f = 4*log(x) - 2*log(x)**2
    fid = diff(integrate(f, x), x)
    assert abs(f.subs(x, 42).evalf() - fid.subs(x, 42).evalf()) < 1e-10


def test_issue_3788():
    assert integrate(1/(1 + x**2), x) == atan(x)


def test_issue_3952():
    f = sin(x)
    assert integrate(f, x) == -cos(x)
    raises(ValueError, lambda: integrate(f, 2*x))


def test_issue_4516():
    assert integrate(2**x - 2*x, x) == 2**x/log(2) - x**2


def test_issue_7450():
    ans = integrate(exp(-(1 + I)*x), (x, 0, oo))
    assert re(ans) == S.Half and im(ans) == Rational(-1, 2)


def test_issue_8623():
    assert integrate((1 + cos(2*x)) / (3 - 2*cos(2*x)), (x, 0, pi)) == -pi/2 + sqrt(5)*pi/2
    assert integrate((1 + cos(2*x))/(3 - 2*cos(2*x))) == -x/2 + sqrt(5)*(atan(sqrt(5)*tan(x)) + \
        pi*floor((x - pi/2)/pi))/2


def test_issue_9569():
    assert integrate(1 / (2 - cos(x)), (x, 0, pi)) == pi/sqrt(3)
    assert integrate(1/(2 - cos(x))) == 2*sqrt(3)*(atan(sqrt(3)*tan(x/2)) + pi*floor((x/2 - pi/2)/pi))/3


def test_issue_13733():
    s = Symbol('s', positive=True)
    pz = exp(-(z - y)**2/(2*s*s))/sqrt(2*pi*s*s)
    pzgx = integrate(pz, (z, x, oo))
    assert integrate(pzgx, (x, 0, oo)) == sqrt(2)*s*exp(-y**2/(2*s**2))/(2*sqrt(pi)) + \
        y*erf(sqrt(2)*y/(2*s))/2 + y/2


def test_issue_13749():
    assert integrate(1 / (2 + cos(x)), (x, 0, pi)) == pi/sqrt(3)
    assert integrate(1/(2 + cos(x))) == 2*sqrt(3)*(atan(sqrt(3)*tan(x/2)/3) + pi*floor((x/2 - pi/2)/pi))/3


def test_issue_18133():
    assert integrate(exp(x)/(1 + x)**2, x) == NonElementaryIntegral(exp(x)/(x + 1)**2, x)


def test_issue_21741():
    a = 4e6
    b = 2.5e-7
    r = Piecewise((b*I*exp(-a*I*pi*t*y)*exp(-a*I*pi*x*z)/(pi*x), Ne(x, 0)),
                  (z*exp(-a*I*pi*t*y), True))
    fun = E**((-2*I*pi*(z*x+t*y))/(500*10**(-9)))
    assert all_close(integrate(fun, z), r)


def test_matrices():
    M = Matrix(2, 2, lambda i, j: (i + j + 1)*sin((i + j + 1)*x))

    assert integrate(M, x) == Matrix([
        [-cos(x), -cos(2*x)],
        [-cos(2*x), -cos(3*x)],
    ])


def test_integrate_functions():
    # issue 4111
    assert integrate(f(x), x) == Integral(f(x), x)
    assert integrate(f(x), (x, 0, 1)) == Integral(f(x), (x, 0, 1))
    assert integrate(f(x)*diff(f(x), x), x) == f(x)**2/2
    assert integrate(diff(f(x), x) / f(x), x) == log(f(x))


def test_integrate_derivatives():
    assert integrate(Derivative(f(x), x), x) == f(x)
    assert integrate(Derivative(f(y), y), x) == x*Derivative(f(y), y)
    assert integrate(Derivative(f(x), x)**2, x) == \
        Integral(Derivative(f(x), x)**2, x)


def test_transform():
    a = Integral(x**2 + 1, (x, -1, 2))
    fx = x
    fy = 3*y + 1
    assert a.doit() == a.transform(fx, fy).doit()
    assert a.transform(fx, fy).transform(fy, fx) == a
    fx = 3*x + 1
    fy = y
    assert a.transform(fx, fy).transform(fy, fx) == a
    a = Integral(sin(1/x), (x, 0, 1))
    assert a.transform(x, 1/y) == Integral(sin(y)/y**2, (y, 1, oo))
    assert a.transform(x, 1/y).transform(y, 1/x) == a
    a = Integral(exp(-x**2), (x, -oo, oo))
    assert a.transform(x, 2*y) == Integral(2*exp(-4*y**2), (y, -oo, oo))
    # < 3 arg limit handled properly
    assert Integral(x, x).transform(x, a*y).doit() == \
        Integral(y*a**2, y).doit()
    _3 = S(3)
    assert Integral(x, (x, 0, -_3)).transform(x, 1/y).doit() == \
        Integral(-1/x**3, (x, -oo, -1/_3)).doit()
    assert Integral(x, (x, 0, _3)).transform(x, 1/y) == \
        Integral(y**(-3), (y, 1/_3, oo))
    # issue 8400
    i = Integral(x + y, (x, 1, 2), (y, 1, 2))
    assert i.transform(x, (x + 2*y, x)).doit() == \
        i.transform(x, (x + 2*z, x)).doit() == 3

    i = Integral(x, (x, a, b))
    assert i.transform(x, 2*s) == Integral(4*s, (s, a/2, b/2))
    raises(ValueError, lambda: i.transform(x, 1))
    raises(ValueError, lambda: i.transform(x, s*t))
    raises(ValueError, lambda: i.transform(x, -s))
    raises(ValueError, lambda: i.transform(x, (s, t)))
    raises(ValueError, lambda: i.transform(2*x, 2*s))

    i = Integral(x**2, (x, 1, 2))
    raises(ValueError, lambda: i.transform(x**2, s))

    am = Symbol('a', negative=True)
    bp = Symbol('b', positive=True)
    i = Integral(x, (x, bp, am))
    i.transform(x, 2*s)
    assert i.transform(x, 2*s) == Integral(-4*s, (s, am/2, bp/2))

    i = Integral(x, (x, a))
    assert i.transform(x, 2*s) == Integral(4*s, (s, a/2))


def test_issue_4052():
    f = S.Half*asin(x) + x*sqrt(1 - x**2)/2

    assert integrate(cos(asin(x)), x) == f
    assert integrate(sin(acos(x)), x) == f


@slow
def test_evalf_integrals():
    assert NS(Integral(x, (x, 2, 5)), 15) == '10.5000000000000'
    gauss = Integral(exp(-x**2), (x, -oo, oo))
    assert NS(gauss, 15) == '1.77245385090552'
    assert NS(gauss**2 - pi + E*Rational(
        1, 10**20), 15) in ('2.71828182845904e-20', '2.71828182845905e-20')
    # A monster of an integral from http://mathworld.wolfram.com/DefiniteIntegral.html
    t = Symbol('t')
    a = 8*sqrt(3)/(1 + 3*t**2)
    b = 16*sqrt(2)*(3*t + 1)*sqrt(4*t**2 + t + 1)**3
    c = (3*t**2 + 1)*(11*t**2 + 2*t + 3)**2
    d = sqrt(2)*(249*t**2 + 54*t + 65)/(11*t**2 + 2*t + 3)**2
    f = a - b/c - d
    assert NS(Integral(f, (t, 0, 1)), 50) == \
        NS((3*sqrt(2) - 49*pi + 162*atan(sqrt(2)))/12, 50)
    # http://mathworld.wolfram.com/VardisIntegral.html
    assert NS(Integral(log(log(1/x))/(1 + x + x**2), (x, 0, 1)), 15) == \
        NS('pi/sqrt(3) * log(2*pi**(5/6) / gamma(1/6))', 15)
    # http://mathworld.wolfram.com/AhmedsIntegral.html
    assert NS(Integral(atan(sqrt(x**2 + 2))/(sqrt(x**2 + 2)*(x**2 + 1)), (x,
              0, 1)), 15) == NS(5*pi**2/96, 15)
    # http://mathworld.wolfram.com/AbelsIntegral.html
    assert NS(Integral(x/((exp(pi*x) - exp(
        -pi*x))*(x**2 + 1)), (x, 0, oo)), 15) == NS('log(2)/2-1/4', 15)
    # Complex part trimming
    # http://mathworld.wolfram.com/VardisIntegral.html
    assert NS(Integral(log(log(sin(x)/cos(x))), (x, pi/4, pi/2)), 15, chop=True) == \
        NS('pi/4*log(4*pi**3/gamma(1/4)**4)', 15)
    #
    # Endpoints causing trouble (rounding error in integration points -> complex log)
    assert NS(
        2 + Integral(log(2*cos(x/2)), (x, -pi, pi)), 17, chop=True) == NS(2, 17)
    assert NS(
        2 + Integral(log(2*cos(x/2)), (x, -pi, pi)), 20, chop=True) == NS(2, 20)
    assert NS(
        2 + Integral(log(2*cos(x/2)), (x, -pi, pi)), 22, chop=True) == NS(2, 22)
    # Needs zero handling
    assert NS(pi - 4*Integral(
        'sqrt(1-x**2)', (x, 0, 1)), 15, maxn=30, chop=True) in ('0.0', '0')
    # Oscillatory quadrature
    a = Integral(sin(x)/x**2, (x, 1, oo)).evalf(maxn=15)
    assert 0.49 < a < 0.51
    assert NS(
        Integral(sin(x)/x**2, (x, 1, oo)), quad='osc') == '0.504067061906928'
    assert NS(Integral(
        cos(pi*x + 1)/x, (x, -oo, -1)), quad='osc') == '0.276374705640365'
    # indefinite integrals aren't evaluated
    assert NS(Integral(x, x)) == 'Integral(x, x)'
    assert NS(Integral(x, (x, y))) == 'Integral(x, (x, y))'


def test_evalf_issue_939():
    # https://github.com/sympy/sympy/issues/4038

    # The output form of an integral may differ by a step function between
    # revisions, making this test a bit useless. This can't be said about
    # other two tests. For now, all values of this evaluation are used here,
    # but in future this should be reconsidered.
    assert NS(integrate(1/(x**5 + 1), x).subs(x, 4), chop=True) in \
        ['-0.000976138910649103', '0.965906660135753', '1.93278945918216']

    assert NS(Integral(1/(x**5 + 1), (x, 2, 4))) == '0.0144361088886740'
    assert NS(
        integrate(1/(x**5 + 1), (x, 2, 4)), chop=True) == '0.0144361088886740'


def test_double_previously_failing_integrals():
    # Double integrals not implemented <- Sure it is!
    res = integrate(sqrt(x) + x*y, (x, 1, 2), (y, -1, 1))
    # Old numerical test
    assert NS(res, 15) == '2.43790283299492'
    # Symbolic test
    assert res == Rational(-4, 3) + 8*sqrt(2)/3
    # double integral + zero detection
    assert integrate(sin(x + x*y), (x, -1, 1), (y, -1, 1)) is S.Zero


def test_integrate_SingularityFunction():
    in_1 = SingularityFunction(x, a, 3) + SingularityFunction(x, 5, -1)
    out_1 = SingularityFunction(x, a, 4)/4 + SingularityFunction(x, 5, 0)
    assert integrate(in_1, x) == out_1

    in_2 = 10*SingularityFunction(x, 4, 0) - 5*SingularityFunction(x, -6, -2)
    out_2 = 10*SingularityFunction(x, 4, 1) - 5*SingularityFunction(x, -6, -1)
    assert integrate(in_2, x) == out_2

    in_3 = 2*x**2*y -10*SingularityFunction(x, -4, 7) - 2*SingularityFunction(y, 10, -2)
    out_3_1 = 2*x**3*y/3 - 2*x*SingularityFunction(y, 10, -2) - 5*SingularityFunction(x, -4, 8)/4
    out_3_2 = x**2*y**2 - 10*y*SingularityFunction(x, -4, 7) - 2*SingularityFunction(y, 10, -1)
    assert integrate(in_3, x) == out_3_1
    assert integrate(in_3, y) == out_3_2

    assert unchanged(Integral, in_3, (x,))
    assert Integral(in_3, x) == Integral(in_3, (x,))
    assert Integral(in_3, x).doit() == out_3_1

    in_4 = 10*SingularityFunction(x, -4, 7) - 2*SingularityFunction(x, 10, -2)
    out_4 = 5*SingularityFunction(x, -4, 8)/4 - 2*SingularityFunction(x, 10, -1)
    assert integrate(in_4, (x, -oo, x)) == out_4

    assert integrate(SingularityFunction(x, 5, -1), x) == SingularityFunction(x, 5, 0)
    assert integrate(SingularityFunction(x, 0, -1), (x, -oo, oo)) == 1
    assert integrate(5*SingularityFunction(x, 5, -1), (x, -oo, oo)) == 5
    assert integrate(SingularityFunction(x, 5, -1) * f(x), (x, -oo, oo)) == f(5)


def test_integrate_DiracDelta():
    # This is here to check that deltaintegrate is being called, but also
    # to test definite integrals. More tests are in test_deltafunctions.py
    assert integrate(DiracDelta(x) * f(x), (x, -oo, oo)) == f(0)
    assert integrate(DiracDelta(x)**2, (x, -oo, oo)) == DiracDelta(0)
    # issue 4522
    assert integrate(integrate((4 - 4*x + x*y - 4*y) * \
        DiracDelta(x)*DiracDelta(y - 1), (x, 0, 1)), (y, 0, 1)) == 0
    # issue 5729
    p = exp(-(x**2 + y**2))/pi
    assert integrate(p*DiracDelta(x - 10*y), (x, -oo, oo), (y, -oo, oo)) == \
        integrate(p*DiracDelta(x - 10*y), (y, -oo, oo), (x, -oo, oo)) == \
        integrate(p*DiracDelta(10*x - y), (x, -oo, oo), (y, -oo, oo)) == \
        integrate(p*DiracDelta(10*x - y), (y, -oo, oo), (x, -oo, oo)) == \
        1/sqrt(101*pi)


def test_integrate_returns_piecewise():
    assert integrate(x**y, x) == Piecewise(
        (x**(y + 1)/(y + 1), Ne(y, -1)), (log(x), True))
    assert integrate(x**y, y) == Piecewise(
        (x**y/log(x), Ne(log(x), 0)), (y, True))
    assert integrate(exp(n*x), x) == Piecewise(
        (exp(n*x)/n, Ne(n, 0)), (x, True))
    assert integrate(x*exp(n*x), x) == Piecewise(
        ((n*x - 1)*exp(n*x)/n**2, Ne(n**2, 0)), (x**2/2, True))
    assert integrate(x**(n*y), x) == Piecewise(
        (x**(n*y + 1)/(n*y + 1), Ne(n*y, -1)), (log(x), True))
    assert integrate(x**(n*y), y) == Piecewise(
        (x**(n*y)/(n*log(x)), Ne(n*log(x), 0)), (y, True))
    assert integrate(cos(n*x), x) == Piecewise(
        (sin(n*x)/n, Ne(n, 0)), (x, True))
    assert integrate(cos(n*x)**2, x) == Piecewise(
        ((n*x/2 + sin(n*x)*cos(n*x)/2)/n, Ne(n, 0)), (x, True))
    assert integrate(x*cos(n*x), x) == Piecewise(
        (x*sin(n*x)/n + cos(n*x)/n**2, Ne(n, 0)), (x**2/2, True))
    assert integrate(sin(n*x), x) == Piecewise(
        (-cos(n*x)/n, Ne(n, 0)), (0, True))
    assert integrate(sin(n*x)**2, x) == Piecewise(
        ((n*x/2 - sin(n*x)*cos(n*x)/2)/n, Ne(n, 0)), (0, True))
    assert integrate(x*sin(n*x), x) == Piecewise(
        (-x*cos(n*x)/n + sin(n*x)/n**2, Ne(n, 0)), (0, True))
    assert integrate(exp(x*y), (x, 0, z)) == Piecewise(
        (exp(y*z)/y - 1/y, (y > -oo) & (y < oo) & Ne(y, 0)), (z, True))
    # https://github.com/sympy/sympy/issues/23707
    assert integrate(exp(t)*exp(-t*sqrt(x - y)), t) == Piecewise(
        (-exp(t)/(sqrt(x - y)*exp(t*sqrt(x - y)) - exp(t*sqrt(x - y))),
        Ne(x, y + 1)), (t, True))


def test_integrate_max_min():
    x = symbols('x', real=True)
    assert integrate(Min(x, 2), (x, 0, 3)) == 4
    assert integrate(Max(x**2, x**3), (x, 0, 2)) == Rational(49, 12)
    assert integrate(Min(exp(x), exp(-x))**2, x) == Piecewise( \
        (exp(2*x)/2, x <= 0), (1 - exp(-2*x)/2, True))
    # issue 7907
    c = symbols('c', extended_real=True)
    int1 = integrate(Max(c, x)*exp(-x**2), (x, -oo, oo))
    int2 = integrate(c*exp(-x**2), (x, -oo, c))
    int3 = integrate(x*exp(-x**2), (x, c, oo))
    assert int1 == int2 + int3 == sqrt(pi)*c*erf(c)/2 + \
        sqrt(pi)*c/2 + exp(-c**2)/2


def test_integrate_Abs_sign():
    assert integrate(Abs(x), (x, -2, 1)) == Rational(5, 2)
    assert integrate(Abs(x), (x, 0, 1)) == S.Half
    assert integrate(Abs(x + 1), (x, 0, 1)) == Rational(3, 2)
    assert integrate(Abs(x**2 - 1), (x, -2, 2)) == 4
    assert integrate(Abs(x**2 - 3*x), (x, -15, 15)) == 2259
    assert integrate(sign(x), (x, -1, 2)) == 1
    assert integrate(sign(x)*sin(x), (x, -pi, pi)) == 4
    assert integrate(sign(x - 2) * x**2, (x, 0, 3)) == Rational(11, 3)

    t, s = symbols('t s', real=True)
    assert integrate(Abs(t), t) == Piecewise(
        (-t**2/2, t <= 0), (t**2/2, True))
    assert integrate(Abs(2*t - 6), t) == Piecewise(
        (-t**2 + 6*t, t <= 3), (t**2 - 6*t + 18, True))
    assert (integrate(abs(t - s**2), (t, 0, 2)) ==
        2*s**2*Min(2, s**2) - 2*s**2 - Min(2, s**2)**2 + 2)
    assert integrate(exp(-Abs(t)), t) == Piecewise(
        (exp(t), t <= 0), (2 - exp(-t), True))
    assert integrate(sign(2*t - 6), t) == Piecewise(
        (-t, t < 3), (t - 6, True))
    assert integrate(2*t*sign(t**2 - 1), t) == Piecewise(
        (t**2, t < -1), (-t**2 + 2, t < 1), (t**2, True))
    assert integrate(sign(t), (t, s + 1)) == Piecewise(
        (s + 1, s + 1 > 0), (-s - 1, s + 1 < 0), (0, True))


def test_subs1():
    e = Integral(exp(x - y), x)
    assert e.subs(y, 3) == Integral(exp(x - 3), x)
    e = Integral(exp(x - y), (x, 0, 1))
    assert e.subs(y, 3) == Integral(exp(x - 3), (x, 0, 1))
    f = Lambda(x, exp(-x**2))
    conv = Integral(f(x - y)*f(y), (y, -oo, oo))
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo))


def test_subs2():
    e = Integral(exp(x - y), x, t)
    assert e.subs(y, 3) == Integral(exp(x - 3), x, t)
    e = Integral(exp(x - y), (x, 0, 1), (t, 0, 1))
    assert e.subs(y, 3) == Integral(exp(x - 3), (x, 0, 1), (t, 0, 1))
    f = Lambda(x, exp(-x**2))
    conv = Integral(f(x - y)*f(y), (y, -oo, oo), (t, 0, 1))
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo), (t, 0, 1))


def test_subs3():
    e = Integral(exp(x - y), (x, 0, y), (t, y, 1))
    assert e.subs(y, 3) == Integral(exp(x - 3), (x, 0, 3), (t, 3, 1))
    f = Lambda(x, exp(-x**2))
    conv = Integral(f(x - y)*f(y), (y, -oo, oo), (t, x, 1))
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo), (t, 0, 1))


def test_subs4():
    e = Integral(exp(x), (x, 0, y), (t, y, 1))
    assert e.subs(y, 3) == Integral(exp(x), (x, 0, 3), (t, 3, 1))
    f = Lambda(x, exp(-x**2))
    conv = Integral(f(y)*f(y), (y, -oo, oo), (t, x, 1))
    assert conv.subs({x: 0}) == Integral(exp(-2*y**2), (y, -oo, oo), (t, 0, 1))


def test_subs5():
    e = Integral(exp(-x**2), (x, -oo, oo))
    assert e.subs(x, 5) == e
    e = Integral(exp(-x**2 + y), x)
    assert e.subs(y, 5) == Integral(exp(-x**2 + 5), x)
    e = Integral(exp(-x**2 + y), (x, x))
    assert e.subs(x, 5) == Integral(exp(y - x**2), (x, 5))
    assert e.subs(y, 5) == Integral(exp(-x**2 + 5), x)
    e = Integral(exp(-x**2 + y), (y, -oo, oo), (x, -oo, oo))
    assert e.subs(x, 5) == e
    assert e.subs(y, 5) == e
    # Test evaluation of antiderivatives
    e = Integral(exp(-x**2), (x, x))
    assert e.subs(x, 5) == Integral(exp(-x**2), (x, 5))
    e = Integral(exp(x), x)
    assert (e.subs(x,1) - e.subs(x,0) - Integral(exp(x), (x, 0, 1))
        ).doit().is_zero


def test_subs6():
    a, b = symbols('a b')
    e = Integral(x*y, (x, f(x), f(y)))
    assert e.subs(x, 1) == Integral(x*y, (x, f(1), f(y)))
    assert e.subs(y, 1) == Integral(x, (x, f(x), f(1)))
    e = Integral(x*y, (x, f(x), f(y)), (y, f(x), f(y)))
    assert e.subs(x, 1) == Integral(x*y, (x, f(1), f(y)), (y, f(1), f(y)))
    assert e.subs(y, 1) == Integral(x*y, (x, f(x), f(y)), (y, f(x), f(1)))
    e = Integral(x*y, (x, f(x), f(a)), (y, f(x), f(a)))
    assert e.subs(a, 1) == Integral(x*y, (x, f(x), f(1)), (y, f(x), f(1)))


def test_subs7():
    e = Integral(x, (x, 1, y), (y, 1, 2))
    assert e.subs({x: 1, y: 2}) == e
    e = Integral(sin(x) + sin(y), (x, sin(x), sin(y)),
                                  (y, 1, 2))
    assert e.subs(sin(y), 1) == e
    assert e.subs(sin(x), 1) == Integral(sin(x) + sin(y), (x, 1, sin(y)),
                                         (y, 1, 2))

def test_expand():
    e = Integral(f(x)+f(x**2), (x, 1, y))
    assert e.expand() == Integral(f(x), (x, 1, y)) + Integral(f(x**2), (x, 1, y))
    e = Integral(f(x)+f(x**2), (x, 1, oo))
    assert e.expand() == e
    assert e.expand(force=True) == Integral(f(x), (x, 1, oo)) + \
           Integral(f(x**2), (x, 1, oo))


def test_integration_variable():
    raises(ValueError, lambda: Integral(exp(-x**2), 3))
    raises(ValueError, lambda: Integral(exp(-x**2), (3, -oo, oo)))


def test_expand_integral():
    assert Integral(cos(x**2)*(sin(x**2) + 1), (x, 0, 1)).expand() == \
        Integral(cos(x**2)*sin(x**2), (x, 0, 1)) + \
        Integral(cos(x**2), (x, 0, 1))
    assert Integral(cos(x**2)*(sin(x**2) + 1), x).expand() == \
        Integral(cos(x**2)*sin(x**2), x) + \
        Integral(cos(x**2), x)


def test_as_sum_midpoint1():
    e = Integral(sqrt(x**3 + 1), (x, 2, 10))
    assert e.as_sum(1, method="midpoint") == 8*sqrt(217)
    assert e.as_sum(2, method="midpoint") == 4*sqrt(65) + 12*sqrt(57)
    assert e.as_sum(3, method="midpoint") == 8*sqrt(217)/3 + \
        8*sqrt(3081)/27 + 8*sqrt(52809)/27
    assert e.as_sum(4, method="midpoint") == 2*sqrt(730) + \
        4*sqrt(7) + 4*sqrt(86) + 6*sqrt(14)
    assert abs(e.as_sum(4, method="midpoint").n() - e.n()) < 0.5

    e = Integral(sqrt(x**3 + y**3), (x, 2, 10), (y, 0, 10))
    raises(NotImplementedError, lambda: e.as_sum(4))


def test_as_sum_midpoint2():
    e = Integral((x + y)**2, (x, 0, 1))
    n = Symbol('n', positive=True, integer=True)
    assert e.as_sum(1, method="midpoint").expand() == Rational(1, 4) + y + y**2
    assert e.as_sum(2, method="midpoint").expand() == Rational(5, 16) + y + y**2
    assert e.as_sum(3, method="midpoint").expand() == Rational(35, 108) + y + y**2
    assert e.as_sum(4, method="midpoint").expand() == Rational(21, 64) + y + y**2
    assert e.as_sum(n, method="midpoint").expand() == \
        y**2 + y + Rational(1, 3) - 1/(12*n**2)


def test_as_sum_left():
    e = Integral((x + y)**2, (x, 0, 1))
    assert e.as_sum(1, method="left").expand() == y**2
    assert e.as_sum(2, method="left").expand() == Rational(1, 8) + y/2 + y**2
    assert e.as_sum(3, method="left").expand() == Rational(5, 27) + y*Rational(2, 3) + y**2
    assert e.as_sum(4, method="left").expand() == Rational(7, 32) + y*Rational(3, 4) + y**2
    assert e.as_sum(n, method="left").expand() == \
        y**2 + y + Rational(1, 3) - y/n - 1/(2*n) + 1/(6*n**2)
    assert e.as_sum(10, method="left", evaluate=False).has(Sum)


def test_as_sum_right():
    e = Integral((x + y)**2, (x, 0, 1))
    assert e.as_sum(1, method="right").expand() == 1 + 2*y + y**2
    assert e.as_sum(2, method="right").expand() == Rational(5, 8) + y*Rational(3, 2) + y**2
    assert e.as_sum(3, method="right").expand() == Rational(14, 27) + y*Rational(4, 3) + y**2
    assert e.as_sum(4, method="right").expand() == Rational(15, 32) + y*Rational(5, 4) + y**2
    assert e.as_sum(n, method="right").expand() == \
        y**2 + y + Rational(1, 3) + y/n + 1/(2*n) + 1/(6*n**2)


def test_as_sum_trapezoid():
    e = Integral((x + y)**2, (x, 0, 1))
    assert e.as_sum(1, method="trapezoid").expand() == y**2 + y + S.Half
    assert e.as_sum(2, method="trapezoid").expand() == y**2 + y + Rational(3, 8)
    assert e.as_sum(3, method="trapezoid").expand() == y**2 + y + Rational(19, 54)
    assert e.as_sum(4, method="trapezoid").expand() == y**2 + y + Rational(11, 32)
    assert e.as_sum(n, method="trapezoid").expand() == \
        y**2 + y + Rational(1, 3) + 1/(6*n**2)
    assert Integral(sign(x), (x, 0, 1)).as_sum(1, 'trapezoid') == S.Half


def test_as_sum_raises():
    e = Integral((x + y)**2, (x, 0, 1))
    raises(ValueError, lambda: e.as_sum(-1))
    raises(ValueError, lambda: e.as_sum(0))
    raises(ValueError, lambda: Integral(x).as_sum(3))
    raises(ValueError, lambda: e.as_sum(oo))
    raises(ValueError, lambda: e.as_sum(3, method='xxxx2'))


def test_nested_doit():
    e = Integral(Integral(x, x), x)
    f = Integral(x, x, x)
    assert e.doit() == f.doit()


def test_issue_4665():
    # Allow only upper or lower limit evaluation
    e = Integral(x**2, (x, None, 1))
    f = Integral(x**2, (x, 1, None))
    assert e.doit() == Rational(1, 3)
    assert f.doit() == Rational(-1, 3)
    assert Integral(x*y, (x, None, y)).subs(y, t) == Integral(x*t, (x, None, t))
    assert Integral(x*y, (x, y, None)).subs(y, t) == Integral(x*t, (x, t, None))
    assert integrate(x**2, (x, None, 1)) == Rational(1, 3)
    assert integrate(x**2, (x, 1, None)) == Rational(-1, 3)
    assert integrate("x**2", ("x", "1", None)) == Rational(-1, 3)


def test_integral_reconstruct():
    e = Integral(x**2, (x, -1, 1))
    assert e == Integral(*e.args)


def test_doit_integrals():
    e = Integral(Integral(2*x), (x, 0, 1))
    assert e.doit() == Rational(1, 3)
    assert e.doit(deep=False) == Rational(1, 3)
    f = Function('f')
    # doesn't matter if the integral can't be performed
    assert Integral(f(x), (x, 1, 1)).doit() == 0
    # doesn't matter if the limits can't be evaluated
    assert Integral(0, (x, 1, Integral(f(x), x))).doit() == 0
    assert Integral(x, (a, 0)).doit() == 0
    limits = ((a, 1, exp(x)), (x, 0))
    assert Integral(a, *limits).doit() == Rational(1, 4)
    assert Integral(a, *list(reversed(limits))).doit() == 0


def test_issue_4884():
    assert integrate(sqrt(x)*(1 + x)) == \
        Piecewise(
            (2*sqrt(x)*(x + 1)**2/5 - 2*sqrt(x)*(x + 1)/15 - 4*sqrt(x)/15,
            Abs(x + 1) > 1),
            (2*I*sqrt(-x)*(x + 1)**2/5 - 2*I*sqrt(-x)*(x + 1)/15 -
            4*I*sqrt(-x)/15, True))
    assert integrate(x**x*(1 + log(x))) == x**x

def test_issue_18153():
    assert integrate(x**n*log(x),x) == \
    Piecewise(
        (n*x*x**n*log(x)/(n**2 + 2*n + 1) +
    x*x**n*log(x)/(n**2 + 2*n + 1) - x*x**n/(n**2 + 2*n + 1)
    , Ne(n, -1)), (log(x)**2/2, True)
    )


def test_is_number():
    from sympy.abc import x, y, z
    assert Integral(x).is_number is False
    assert Integral(1, x).is_number is False
    assert Integral(1, (x, 1)).is_number is True
    assert Integral(1, (x, 1, 2)).is_number is True
    assert Integral(1, (x, 1, y)).is_number is False
    assert Integral(1, (x, y)).is_number is False
    assert Integral(x, y).is_number is False
    assert Integral(x, (y, 1, x)).is_number is False
    assert Integral(x, (y, 1, 2)).is_number is False
    assert Integral(x, (x, 1, 2)).is_number is True
    # `foo.is_number` should always be equivalent to `not foo.free_symbols`
    # in each of these cases, there are pseudo-free symbols
    i = Integral(x, (y, 1, 1))
    assert i.is_number is False and i.n() == 0
    i = Integral(x, (y, z, z))
    assert i.is_number is False and i.n() == 0
    i = Integral(1, (y, z, z + 2))
    assert i.is_number is False and i.n() == 2.0

    assert Integral(x*y, (x, 1, 2), (y, 1, 3)).is_number is True
    assert Integral(x*y, (x, 1, 2), (y, 1, z)).is_number is False
    assert Integral(x, (x, 1)).is_number is True
    assert Integral(x, (x, 1, Integral(y, (y, 1, 2)))).is_number is True
    assert Integral(Sum(z, (z, 1, 2)), (x, 1, 2)).is_number is True
    # it is possible to get a false negative if the integrand is
    # actually an unsimplified zero, but this is true of is_number in general.
    assert Integral(sin(x)**2 + cos(x)**2 - 1, x).is_number is False
    assert Integral(f(x), (x, 0, 1)).is_number is True


def test_free_symbols():
    from sympy.abc import x, y, z
    assert Integral(0, x).free_symbols == {x}
    assert Integral(x).free_symbols == {x}
    assert Integral(x, (x, None, y)).free_symbols == {y}
    assert Integral(x, (x, y, None)).free_symbols == {y}
    assert Integral(x, (x, 1, y)).free_symbols == {y}
    assert Integral(x, (x, y, 1)).free_symbols == {y}
    assert Integral(x, (x, x, y)).free_symbols == {x, y}
    assert Integral(x, x, y).free_symbols == {x, y}
    assert Integral(x, (x, 1, 2)).free_symbols == set()
    assert Integral(x, (y, 1, 2)).free_symbols == {x}
    # pseudo-free in this case
    assert Integral(x, (y, z, z)).free_symbols == {x, z}
    assert Integral(x, (y, 1, 2), (y, None, None)
        ).free_symbols == {x, y}
    assert Integral(x, (y, 1, 2), (x, 1, y)
        ).free_symbols == {y}
    assert Integral(2, (y, 1, 2), (y, 1, x), (x, 1, 2)
        ).free_symbols == set()
    assert Integral(2, (y, x, 2), (y, 1, x), (x, 1, 2)
        ).free_symbols == set()
    assert Integral(2, (x, 1, 2), (y, x, 2), (y, 1, 2)
        ).free_symbols == {x}
    assert Integral(f(x), (f(x), 1, y)).free_symbols == {y}
    assert Integral(f(x), (f(x), 1, x)).free_symbols == {x}


def test_is_zero():
    from sympy.abc import x, m
    assert Integral(0, (x, 1, x)).is_zero
    assert Integral(1, (x, 1, 1)).is_zero
    assert Integral(1, (x, 1, 2), (y, 2)).is_zero is False
    assert Integral(x, (m, 0)).is_zero
    assert Integral(x + m, (m, 0)).is_zero is None
    i = Integral(m, (m, 1, exp(x)), (x, 0))
    assert i.is_zero is None
    assert Integral(m, (x, 0), (m, 1, exp(x))).is_zero is True

    assert Integral(x, (x, oo, oo)).is_zero # issue 8171
    assert Integral(x, (x, -oo, -oo)).is_zero

    # this is zero but is beyond the scope of what is_zero
    # should be doing
    assert Integral(sin(x), (x, 0, 2*pi)).is_zero is None


def test_series():
    from sympy.abc import x
    i = Integral(cos(x), (x, x))
    e = i.lseries(x)
    assert i.nseries(x, n=8).removeO() == Add(*[next(e) for j in range(4)])


def test_trig_nonelementary_integrals():
    x = Symbol('x')
    assert integrate((1 + sin(x))/x, x) == log(x) + Si(x)
    # next one comes out as log(x) + log(x**2)/2 + Ci(x)
    # so not hardcoding this log ugliness
    assert integrate((cos(x) + 2)/x, x).has(Ci)


def test_issue_4403():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z', positive=True)
    assert integrate(sqrt(x**2 + z**2), x) == \
        z**2*asinh(x/z)/2 + x*sqrt(x**2 + z**2)/2
    assert integrate(sqrt(x**2 - z**2), x) == \
        x*sqrt(x**2 - z**2)/2 - z**2*log(x + sqrt(x**2 - z**2))/2

    x = Symbol('x', real=True)
    y = Symbol('y', positive=True)
    assert integrate(1/(x**2 + y**2)**S('3/2'), x) == \
        x/(y**2*sqrt(x**2 + y**2))
    # If y is real and nonzero, we get x*Abs(y)/(y**3*sqrt(x**2 + y**2)),
    # which results from sqrt(1 + x**2/y**2) = sqrt(x**2 + y**2)/|y|.


def test_issue_4403_2():
    assert integrate(sqrt(-x**2 - 4), x) == \
        -2*atan(x/sqrt(-4 - x**2)) + x*sqrt(-4 - x**2)/2


def test_issue_4100():
    R = Symbol('R', positive=True)
    assert integrate(sqrt(R**2 - x**2), (x, 0, R)) == pi*R**2/4


def test_issue_5167():
    from sympy.abc import w, x, y, z
    f = Function('f')
    assert Integral(Integral(f(x), x), x) == Integral(f(x), x, x)
    assert Integral(f(x)).args == (f(x), Tuple(x))
    assert Integral(Integral(f(x))).args == (f(x), Tuple(x), Tuple(x))
    assert Integral(Integral(f(x)), y).args == (f(x), Tuple(x), Tuple(y))
    assert Integral(Integral(f(x), z), y).args == (f(x), Tuple(z), Tuple(y))
    assert Integral(Integral(Integral(f(x), x), y), z).args == \
        (f(x), Tuple(x), Tuple(y), Tuple(z))
    assert integrate(Integral(f(x), x), x) == Integral(f(x), x, x)
    assert integrate(Integral(f(x), y), x) == y*Integral(f(x), x)
    assert integrate(Integral(f(x), x), y) in [Integral(y*f(x), x), y*Integral(f(x), x)]
    assert integrate(Integral(2, x), x) == x**2
    assert integrate(Integral(2, x), y) == 2*x*y
    # don't re-order given limits
    assert Integral(1, x, y).args != Integral(1, y, x).args
    # do as many as possible
    assert Integral(f(x), y, x, y, x).doit() == y**2*Integral(f(x), x, x)/2
    assert Integral(f(x), (x, 1, 2), (w, 1, x), (z, 1, y)).doit() == \
        y*(x - 1)*Integral(f(x), (x, 1, 2)) - (x - 1)*Integral(f(x), (x, 1, 2))


def test_issue_4890():
    z = Symbol('z', positive=True)
    assert integrate(exp(-log(x)**2), x) == \
        sqrt(pi)*exp(Rational(1, 4))*erf(log(x) - S.Half)/2
    assert integrate(exp(log(x)**2), x) == \
        sqrt(pi)*exp(Rational(-1, 4))*erfi(log(x)+S.Half)/2
    assert integrate(exp(-z*log(x)**2), x) == \
        sqrt(pi)*exp(1/(4*z))*erf(sqrt(z)*log(x) - 1/(2*sqrt(z)))/(2*sqrt(z))


def test_issue_4551():
    assert not integrate(1/(x*sqrt(1 - x**2)), x).has(Integral)


def test_issue_4376():
    n = Symbol('n', integer=True, positive=True)
    assert simplify(integrate(n*(x**(1/n) - 1), (x, 0, S.Half)) -
                (n**2 - 2**(1/n)*n**2 - n*2**(1/n))/(2**(1 + 1/n) + n*2**(1 + 1/n))) == 0


def test_issue_4517():
    assert integrate((sqrt(x) - x**3)/x**Rational(1, 3), x) == \
        6*x**Rational(7, 6)/7 - 3*x**Rational(11, 3)/11


def test_issue_4527():
    k, m = symbols('k m', integer=True)
    assert integrate(sin(k*x)*sin(m*x), (x, 0, pi)).simplify() == \
        Piecewise((0, Eq(k, 0) | Eq(m, 0)),
                  (-pi/2, Eq(k, -m) | (Eq(k, 0) & Eq(m, 0))),
                  (pi/2, Eq(k, m) | (Eq(k, 0) & Eq(m, 0))),
                  (0, True))
    # Should be possible to further simplify to:
    # Piecewise(
    #    (0, Eq(k, 0) | Eq(m, 0)),
    #    (-pi/2, Eq(k, -m)),
    #    (pi/2, Eq(k, m)),
    #    (0, True))
    assert integrate(sin(k*x)*sin(m*x), (x,)) == Piecewise(
        (0, And(Eq(k, 0), Eq(m, 0))),
        (-x*sin(m*x)**2/2 - x*cos(m*x)**2/2 + sin(m*x)*cos(m*x)/(2*m), Eq(k, -m)),
        (x*sin(m*x)**2/2 + x*cos(m*x)**2/2 - sin(m*x)*cos(m*x)/(2*m), Eq(k, m)),
        (m*sin(k*x)*cos(m*x)/(k**2 - m**2) -
         k*sin(m*x)*cos(k*x)/(k**2 - m**2), True))


def test_issue_4199():
    ypos = Symbol('y', positive=True)
    # TODO: Remove conds='none' below, let the assumption take care of it.
    assert integrate(exp(-I*2*pi*ypos*x)*x, (x, -oo, oo), conds='none') == \
        Integral(exp(-I*2*pi*ypos*x)*x, (x, -oo, oo))


def test_issue_3940():
    a, b, c, d = symbols('a:d', positive=True)
    assert integrate(exp(-x**2 + I*c*x), x) == \
        -sqrt(pi)*exp(-c**2/4)*erf(I*c/2 - x)/2
    assert integrate(exp(a*x**2 + b*x + c), x).equals(
        sqrt(pi)*exp(c - b**2/(4*a))*erfi((2*a*x + b)/(2*sqrt(a)))/(2*sqrt(a)))

    from sympy.core.function import expand_mul
    from sympy.abc import k
    assert expand_mul(integrate(exp(-x**2)*exp(I*k*x), (x, -oo, oo))) == \
        sqrt(pi)*exp(-k**2/4)
    a, d = symbols('a d', positive=True)
    assert expand_mul(integrate(exp(-a*x**2 + 2*d*x), (x, -oo, oo))) == \
        sqrt(pi)*exp(d**2/a)/sqrt(a)


def test_issue_5413():
    # Note that this is not the same as testing ratint() because integrate()
    # pulls out the coefficient.
    assert integrate(-a/(a**2 + x**2), x) == I*log(-I*a + x)/2 - I*log(I*a + x)/2


def test_issue_4892a():
    A, z = symbols('A z')
    c = Symbol('c', nonzero=True)
    P1 = -A*exp(-z)
    P2 = -A/(c*t)*(sin(x)**2 + cos(y)**2)

    h1 = -sin(x)**2 - cos(y)**2
    h2 = -sin(x)**2 + sin(y)**2 - 1

    # there is still some non-deterministic behavior in integrate
    # or trigsimp which permits one of the following
    assert integrate(c*(P2 - P1), t) in [
        c*(-A*(-h1)*log(c*t)/c + A*t*exp(-z)),
        c*(-A*(-h2)*log(c*t)/c + A*t*exp(-z)),
        c*( A* h1 *log(c*t)/c + A*t*exp(-z)),
        c*( A* h2 *log(c*t)/c + A*t*exp(-z)),
        (A*c*t - A*(-h1)*log(t)*exp(z))*exp(-z),
        (A*c*t - A*(-h2)*log(t)*exp(z))*exp(-z),
    ]


def test_issue_4892b():
    # Issues relating to issue 4596 are making the actual result of this hard
    # to test.  The answer should be something like
    #
    # (-sin(y) + sqrt(-72 + 48*cos(y) - 8*cos(y)**2)/2)*log(x + sqrt(-72 +
    # 48*cos(y) - 8*cos(y)**2)/(2*(3 - cos(y)))) + (-sin(y) - sqrt(-72 +
    # 48*cos(y) - 8*cos(y)**2)/2)*log(x - sqrt(-72 + 48*cos(y) -
    # 8*cos(y)**2)/(2*(3 - cos(y)))) + x**2*sin(y)/2 + 2*x*cos(y)

    expr = (sin(y)*x**3 + 2*cos(y)*x**2 + 12)/(x**2 + 2)
    assert trigsimp(factor(integrate(expr, x).diff(x) - expr)) == 0


def test_issue_5178():
    assert integrate(sin(x)*f(y, z), (x, 0, pi), (y, 0, pi), (z, 0, pi)) == \
        2*Integral(f(y, z), (y, 0, pi), (z, 0, pi))


def test_integrate_series():
    f = sin(x).series(x, 0, 10)
    g = x**2/2 - x**4/24 + x**6/720 - x**8/40320 + x**10/3628800 + O(x**11)

    assert integrate(f, x) == g
    assert diff(integrate(f, x), x) == f

    assert integrate(O(x**5), x) == O(x**6)


def test_atom_bug():
    from sympy.integrals.heurisch import heurisch
    assert heurisch(meijerg([], [], [1], [], x), x) is None


def test_limit_bug():
    z = Symbol('z', zero=False)
    assert integrate(sin(x*y*z), (x, 0, pi), (y, 0, pi)).together() == \
        (log(z) - Ci(pi**2*z) + EulerGamma + 2*log(pi))/z


def test_issue_4703():
    g = Function('g')
    assert integrate(exp(x)*g(x), x).has(Integral)


def test_issue_1888():
    f = Function('f')
    assert integrate(f(x).diff(x)**2, x).has(Integral)

# The following tests work using meijerint.


def test_issue_3558():
    assert integrate(cos(x*y), (x, -pi/2, pi/2), (y, 0, pi)) == 2*Si(pi**2/2)


def test_issue_4422():
    assert integrate(1/sqrt(16 + 4*x**2), x) == asinh(x/2) / 2


def test_issue_4493():
    assert simplify(integrate(x*sqrt(1 + 2*x), x)) == \
        sqrt(2*x + 1)*(6*x**2 + x - 1)/15


def test_issue_4737():
    assert integrate(sin(x)/x, (x, -oo, oo)) == pi
    assert integrate(sin(x)/x, (x, 0, oo)) == pi/2
    assert integrate(sin(x)/x, x) == Si(x)


def test_issue_4992():
    # Note: psi in _check_antecedents becomes NaN.
    from sympy.core.function import expand_func
    a = Symbol('a', positive=True)
    assert simplify(expand_func(integrate(exp(-x)*log(x)*x**a, (x, 0, oo)))) == \
        (a*polygamma(0, a) + 1)*gamma(a)


def test_issue_4487():
    from sympy.functions.special.gamma_functions import lowergamma
    assert simplify(integrate(exp(-x)*x**y, x)) == lowergamma(y + 1, x)


def test_issue_4215():
    x = Symbol("x")
    assert integrate(1/(x**2), (x, -1, 1)) is oo


def test_issue_4400():
    n = Symbol('n', integer=True, positive=True)
    assert integrate((x**n)*log(x), x) == \
        n*x*x**n*log(x)/(n**2 + 2*n + 1) + x*x**n*log(x)/(n**2 + 2*n + 1) - \
        x*x**n/(n**2 + 2*n + 1)


def test_issue_6253():
    # Note: this used to raise NotImplementedError
    # Note: psi in _check_antecedents becomes NaN.
    assert integrate((sqrt(1 - x) + sqrt(1 + x))**2/x, x, meijerg=True) == \
        Integral((sqrt(-x + 1) + sqrt(x + 1))**2/x, x)


def test_issue_4153():
    assert integrate(1/(1 + x + y + z), (x, 0, 1), (y, 0, 1), (z, 0, 1)) in [
        -12*log(3) - 3*log(6)/2 + 3*log(8)/2 + 5*log(2) + 7*log(4),
        6*log(2) + 8*log(4) - 27*log(3)/2, 22*log(2) - 27*log(3)/2,
        -12*log(3) - 3*log(6)/2 + 47*log(2)/2]


def test_issue_4326():
    R, b, h = symbols('R b h')
    # It doesn't matter if we can do the integral.  Just make sure the result
    # doesn't contain nan.  This is really a test against _eval_interval.
    e = integrate(((h*(x - R + b))/b)*sqrt(R**2 - x**2), (x, R - b, R))
    assert not e.has(nan)
    # See that it evaluates
    assert not e.has(Integral)


def test_powers():
    assert integrate(2**x + 3**x, x) == 2**x/log(2) + 3**x/log(3)


def test_manual_option():
    raises(ValueError, lambda: integrate(1/x, x, manual=True, meijerg=True))
    # an example of a function that manual integration cannot handle
    assert integrate(log(1+x)/x, (x, 0, 1), manual=True).has(Integral)


def test_meijerg_option():
    raises(ValueError, lambda: integrate(1/x, x, meijerg=True, risch=True))
    # an example of a function that meijerg integration cannot handle
    assert integrate(tan(x), x, meijerg=True) == Integral(tan(x), x)


def test_risch_option():
    # risch=True only allowed on indefinite integrals
    raises(ValueError, lambda: integrate(1/log(x), (x, 0, oo), risch=True))
    assert integrate(exp(-x**2), x, risch=True) == NonElementaryIntegral(exp(-x**2), x)
    assert integrate(log(1/x)*y, x, y, risch=True) == y**2*(x*log(1/x)/2 + x/2)
    assert integrate(erf(x), x, risch=True) == Integral(erf(x), x)
    # TODO: How to test risch=False?


@slow
def test_heurisch_option():
    raises(ValueError, lambda: integrate(1/x, x, risch=True, heurisch=True))
    # an integral that heurisch can handle
    assert integrate(exp(x**2), x, heurisch=True) == sqrt(pi)*erfi(x)/2
    # an integral that heurisch currently cannot handle
    assert integrate(exp(x)/x, x, heurisch=True) == Integral(exp(x)/x, x)
    # an integral where heurisch currently hangs, issue 15471
    assert integrate(log(x)*cos(log(x))/x**Rational(3, 4), x, heurisch=False) == (
        -128*x**Rational(1, 4)*sin(log(x))/289 + 240*x**Rational(1, 4)*cos(log(x))/289 +
        (16*x**Rational(1, 4)*sin(log(x))/17 + 4*x**Rational(1, 4)*cos(log(x))/17)*log(x))


def test_issue_6828():
    f = 1/(1.08*x**2 - 4.3)
    g = integrate(f, x).diff(x)
    assert verify_numerically(f, g, tol=1e-12)


def test_issue_4803():
    x_max = Symbol("x_max")
    assert integrate(y/pi*exp(-(x_max - x)/cos(a)), x) == \
        y*exp((x - x_max)/cos(a))*cos(a)/pi


def test_issue_4234():
    assert integrate(1/sqrt(1 + tan(x)**2)) == tan(x)/sqrt(1 + tan(x)**2)


def test_issue_4492():
    assert simplify(integrate(x**2 * sqrt(5 - x**2), x)).factor(
        deep=True) == Piecewise(
        (I*(2*x**5 - 15*x**3 + 25*x - 25*sqrt(x**2 - 5)*acosh(sqrt(5)*x/5)) /
            (8*sqrt(x**2 - 5)), (x > sqrt(5)) | (x < -sqrt(5))),
        ((2*x**5 - 15*x**3 + 25*x - 25*sqrt(5 - x**2)*asin(sqrt(5)*x/5)) /
            (-8*sqrt(-x**2 + 5)), True))


def test_issue_2708():
    # This test needs to use an integration function that can
    # not be evaluated in closed form.  Update as needed.
    f = 1/(a + z + log(z))
    integral_f = NonElementaryIntegral(f, (z, 2, 3))
    assert Integral(f, (z, 2, 3)).doit() == integral_f
    assert integrate(f + exp(z), (z, 2, 3)) == integral_f - exp(2) + exp(3)
    assert integrate(2*f + exp(z), (z, 2, 3)) == \
        2*integral_f - exp(2) + exp(3)
    assert integrate(exp(1.2*n*s*z*(-t + z)/t), (z, 0, x)) == \
        NonElementaryIntegral(exp(-1.2*n*s*z)*exp(1.2*n*s*z**2/t),
                                  (z, 0, x))


def test_issue_2884():
    f = (4.000002016020*x + 4.000002016020*y + 4.000006024032)*exp(10.0*x)
    e = integrate(f, (x, 0.1, 0.2))
    assert str(e) == '1.86831064982608*y + 2.16387491480008'


def test_issue_8368i():
    from sympy.functions.elementary.complexes import arg, Abs
    assert integrate(exp(-s*x)*cosh(x), (x, 0, oo)) == \
        Piecewise(
            (   pi*Piecewise(
                    (   -s/(pi*(-s**2 + 1)),
                        Abs(s**2) < 1),
                    (   1/(pi*s*(1 - 1/s**2)),
                        Abs(s**(-2)) < 1),
                    (   meijerg(
                            ((S.Half,), (0, 0)),
                            ((0, S.Half), (0,)),
                            polar_lift(s)**2),
                        True)
                ),
                s**2 > 1
            ),
            (
                Integral(exp(-s*x)*cosh(x), (x, 0, oo)),
                True))
    assert integrate(exp(-s*x)*sinh(x), (x, 0, oo)) == \
        Piecewise(
            (   -1/(s + 1)/2 - 1/(-s + 1)/2,
                And(
                    Abs(s) > 1,
                    Abs(arg(s)) < pi/2,
                    Abs(arg(s)) <= pi/2
                    )),
            (   Integral(exp(-s*x)*sinh(x), (x, 0, oo)),
                True))


def test_issue_8901():
    assert integrate(sinh(1.0*x)) == 1.0*cosh(1.0*x)
    assert integrate(tanh(1.0*x)) == 1.0*x - 1.0*log(tanh(1.0*x) + 1)
    assert integrate(tanh(x)) == x - log(tanh(x) + 1)


@slow
def test_issue_8945():
    assert integrate(sin(x)**3/x, (x, 0, 1)) == -Si(3)/4 + 3*Si(1)/4
    assert integrate(sin(x)**3/x, (x, 0, oo)) == pi/4
    assert integrate(cos(x)**2/x**2, x) == -Si(2*x) - cos(2*x)/(2*x) - 1/(2*x)


@slow
def test_issue_7130():
    i, L, a, b = symbols('i L a b')
    integrand = (cos(pi*i*x/L)**2 / (a + b*x)).rewrite(exp)
    assert x not in integrate(integrand, (x, 0, L)).free_symbols


def test_issue_10567():
    a, b, c, t = symbols('a b c t')
    vt = Matrix([a*t, b, c])
    assert integrate(vt, t) == Integral(vt, t).doit()
    assert integrate(vt, t) == Matrix([[a*t**2/2], [b*t], [c*t]])


def test_issue_11742():
    assert integrate(sqrt(-x**2 + 8*x + 48), (x, 4, 12)) == 16*pi


def test_issue_11856():
    t = symbols('t')
    assert integrate(sinc(pi*t), t) == Si(pi*t)/pi


@slow
def test_issue_11876():
    assert integrate(sqrt(log(1/x)), (x, 0, 1)) == sqrt(pi)/2


def test_issue_4950():
    assert integrate((-60*exp(x) - 19.2*exp(4*x))*exp(4*x), x) ==\
        -2.4*exp(8*x) - 12.0*exp(5*x)


def test_issue_4968():
    assert integrate(sin(log(x**2))) == x*sin(log(x**2))/5 - 2*x*cos(log(x**2))/5


def test_singularities():
    assert integrate(1/x**2, (x, -oo, oo)) is oo
    assert integrate(1/x**2, (x, -1, 1)) is oo
    assert integrate(1/(x - 1)**2, (x, -2, 2)) is oo

    assert integrate(1/x**2, (x, 1, -1)) is -oo
    assert integrate(1/(x - 1)**2, (x, 2, -2)) is -oo


def test_issue_12645():
    x, y = symbols('x y', real=True)
    assert (integrate(sin(x*x*x + y*y),
                      (x, -sqrt(pi - y*y), sqrt(pi - y*y)),
                      (y, -sqrt(pi), sqrt(pi)))
                == Integral(sin(x**3 + y**2),
                            (x, -sqrt(-y**2 + pi), sqrt(-y**2 + pi)),
                            (y, -sqrt(pi), sqrt(pi))))


def test_issue_12677():
    assert integrate(sin(x) / (cos(x)**3), (x, 0, pi/6)) == Rational(1, 6)


def test_issue_14078():
    assert integrate((cos(3*x)-cos(x))/x, (x, 0, oo)) == -log(3)


def test_issue_14064():
    assert integrate(1/cosh(x), (x, 0, oo)) == pi/2


def test_issue_14027():
    assert integrate(1/(1 + exp(x - S.Half)/(1 + exp(x))), x) == \
        x - exp(S.Half)*log(exp(x) + exp(S.Half)/(1 + exp(S.Half)))/(exp(S.Half) + E)


def test_issue_8170():
    assert integrate(tan(x), (x, 0, pi/2)) is S.Infinity


def test_issue_8440_14040():
    assert integrate(1/x, (x, -1, 1)) is S.NaN
    assert integrate(1/(x + 1), (x, -2, 3)) is S.NaN


def test_issue_14096():
    assert integrate(1/(x + y)**2, (x, 0, 1)) == -1/(y + 1) + 1/y
    assert integrate(1/(1 + x + y + z)**2, (x, 0, 1), (y, 0, 1), (z, 0, 1)) == \
        -4*log(4) - 6*log(2) + 9*log(3)


def test_issue_14144():
    assert Abs(integrate(1/sqrt(1 - x**3), (x, 0, 1)).n() - 1.402182) < 1e-6
    assert Abs(integrate(sqrt(1 - x**3), (x, 0, 1)).n() - 0.841309) < 1e-6


def test_issue_14375():
    # This raised a TypeError. The antiderivative has exp_polar, which
    # may be possible to unpolarify, so the exact output is not asserted here.
    assert integrate(exp(I*x)*log(x), x).has(Ei)


def test_issue_14437():
    f = Function('f')(x, y, z)
    assert integrate(f, (x, 0, 1), (y, 0, 2), (z, 0, 3)) == \
                Integral(f, (x, 0, 1), (y, 0, 2), (z, 0, 3))


def test_issue_14470():
    assert integrate(1/sqrt(exp(x) + 1), x) == log(sqrt(exp(x) + 1) - 1) - log(sqrt(exp(x) + 1) + 1)


def test_issue_14877():
    f = exp(1 - exp(x**2)*x + 2*x**2)*(2*x**3 + x)/(1 - exp(x**2)*x)**2
    assert integrate(f, x) == \
        -exp(2*x**2 - x*exp(x**2) + 1)/(x*exp(3*x**2) - exp(2*x**2))


def test_issue_14782():
    f = sqrt(-x**2 + 1)*(-x**2 + x)
    assert integrate(f, [x, -1, 1]) == - pi / 8


@slow
def test_issue_14782_slow():
    f = sqrt(-x**2 + 1)*(-x**2 + x)
    assert integrate(f, [x, 0, 1]) == S.One / 3 - pi / 16


def test_issue_12081():
    f = x**(Rational(-3, 2))*exp(-x)
    assert integrate(f, [x, 0, oo]) is oo


def test_issue_15285():
    y = 1/x - 1
    f = 4*y*exp(-2*y)/x**2
    assert integrate(f, [x, 0, 1]) == 1


def test_issue_15432():
    assert integrate(x**n * exp(-x) * log(x), (x, 0, oo)).gammasimp() == Piecewise(
        (gamma(n + 1)*polygamma(0, n) + gamma(n + 1)/n, re(n) + 1 > 0),
        (Integral(x**n*exp(-x)*log(x), (x, 0, oo)), True))


def test_issue_15124():
    omega = IndexedBase('omega')
    m, p = symbols('m p', cls=Idx)
    assert integrate(exp(x*I*(omega[m] + omega[p])), x, conds='none') == \
        -I*exp(I*x*omega[m])*exp(I*x*omega[p])/(omega[m] + omega[p])


def test_issue_15218():
    with warns_deprecated_sympy():
        Integral(Eq(x, y))
    with warns_deprecated_sympy():
        assert Integral(Eq(x, y), x) == Eq(Integral(x, x), Integral(y, x))
    with warns_deprecated_sympy():
        assert Integral(Eq(x, y), x).doit() == Eq(x**2/2, x*y)
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        # The warning is made in the ExprWithLimits superclass. The stacklevel
        # is correct for integrate(Eq) but not Eq.integrate
        assert Eq(x, y).integrate(x) == Eq(x**2/2, x*y)

    # These are not deprecated because they are definite integrals
    assert integrate(Eq(x, y), (x, 0, 1)) == Eq(S.Half, y)
    assert Eq(x, y).integrate((x, 0, 1)) == Eq(S.Half, y)


def test_issue_15292():
    res = integrate(exp(-x**2*cos(2*t)) * cos(x**2*sin(2*t)), (x, 0, oo))
    assert isinstance(res, Piecewise)
    assert gammasimp((res - sqrt(pi)/2 * cos(t)).subs(t, pi/6)) == 0


def test_issue_4514():
    assert integrate(sin(2*x)/sin(x), x) == 2*sin(x)


def test_issue_15457():
    x, a, b = symbols('x a b', real=True)
    definite = integrate(exp(Abs(x-2)), (x, a, b))
    indefinite = integrate(exp(Abs(x-2)), x)
    assert definite.subs({a: 1, b: 3}) == -2 + 2*E
    assert indefinite.subs(x, 3) - indefinite.subs(x, 1) == -2 + 2*E
    assert definite.subs({a: -3, b: -1}) == -exp(3) + exp(5)
    assert indefinite.subs(x, -1) - indefinite.subs(x, -3) == -exp(3) + exp(5)


def test_issue_15431():
    assert integrate(x*exp(x)*log(x), x) == \
        (x*exp(x) - exp(x))*log(x) - exp(x) + Ei(x)


def test_issue_15640_log_substitutions():
    f = x/log(x)
    F = Ei(2*log(x))
    assert integrate(f, x) == F and F.diff(x) == f
    f = x**3/log(x)**2
    F = -x**4/log(x) + 4*Ei(4*log(x))
    assert integrate(f, x) == F and F.diff(x) == f
    f = sqrt(log(x))/x**2
    F = -sqrt(pi)*erfc(sqrt(log(x)))/2 - sqrt(log(x))/x
    assert integrate(f, x) == F and F.diff(x) == f


def test_issue_15509():
    from sympy.vector import CoordSys3D
    N = CoordSys3D('N')
    x = N.x
    assert integrate(cos(a*x + b), (x, x_1, x_2), heurisch=True) == Piecewise(
        (-sin(a*x_1 + b)/a + sin(a*x_2 + b)/a, (a > -oo) & (a < oo) & Ne(a, 0)), \
            (-x_1*cos(b) + x_2*cos(b), True))


def test_issue_4311_fast():
    x = symbols('x', real=True)
    assert integrate(x*abs(9-x**2), x) == Piecewise(
        (x**4/4 - 9*x**2/2, x <= -3),
        (-x**4/4 + 9*x**2/2 - Rational(81, 2), x <= 3),
        (x**4/4 - 9*x**2/2, True))


def test_integrate_with_complex_constants():
    K = Symbol('K', positive=True)
    x = Symbol('x', real=True)
    m = Symbol('m', real=True)
    t = Symbol('t', real=True)
    assert integrate(exp(-I*K*x**2+m*x), x) == sqrt(pi)*exp(-I*m**2
                    /(4*K))*erfi((-2*I*K*x + m)/(2*sqrt(K)*sqrt(-I)))/(2*sqrt(K)*sqrt(-I))
    assert integrate(1/(1 + I*x**2), x) == (-I*(sqrt(-I)*log(x - I*sqrt(-I))/2
            - sqrt(-I)*log(x + I*sqrt(-I))/2))
    assert integrate(exp(-I*x**2), x) == sqrt(pi)*erf(sqrt(I)*x)/(2*sqrt(I))

    assert integrate((1/(exp(I*t)-2)), t) == -t/2 - I*log(exp(I*t) - 2)/2
    assert integrate((1/(exp(I*t)-2)), (t, 0, 2*pi)) == -pi


def test_issue_14241():
    x = Symbol('x')
    n = Symbol('n', positive=True, integer=True)
    assert integrate(n * x ** (n - 1) / (x + 1), x) == \
           n**2*x**n*lerchphi(x*exp_polar(I*pi), 1, n)*gamma(n)/gamma(n + 1)


def test_issue_13112():
    assert integrate(sin(t)**2 / (5 - 4*cos(t)), [t, 0, 2*pi]) == pi / 4


def test_issue_14709b():
    h = Symbol('h', positive=True)
    i = integrate(x*acos(1 - 2*x/h), (x, 0, h))
    assert i == 5*h**2*pi/16


def test_issue_8614():
    x = Symbol('x')
    t = Symbol('t')
    assert integrate(exp(t)/t, (t, -oo, x)) == Ei(x)
    assert integrate((exp(-x) - exp(-2*x))/x, (x, 0, oo)) == log(2)


@slow
def test_issue_15494():
    s = symbols('s', positive=True)

    integrand = (exp(s/2) - 2*exp(1.6*s) + exp(s))*exp(s)
    solution = integrate(integrand, s)
    assert solution != S.NaN
    # Not sure how to test this properly as it is a symbolic expression with floats
    # assert str(solution) == '0.666666666666667*exp(1.5*s) + 0.5*exp(2.0*s) - 0.769230769230769*exp(2.6*s)'
    # Maybe
    assert abs(solution.subs(s, 1) - (-3.67440080236188)) <= 1e-8

    integrand = (exp(s/2) - 2*exp(S(8)/5*s) + exp(s))*exp(s)
    assert integrate(integrand, s) == -10*exp(13*s/5)/13 + 2*exp(3*s/2)/3 + exp(2*s)/2


def test_li_integral():
    y = Symbol('y')
    assert Integral(li(y*x**2), x).doit() == Piecewise((x*li(x**2*y) - \
        x*Ei(3*log(x**2*y)/2)/sqrt(x**2*y),
        Ne(y, 0)), (0, True))


def test_issue_17473():
    x = Symbol('x')
    n = Symbol('n')
    h = S.Half
    ans = x**(n + 1)*gamma(h + h/n)*hyper((h + h/n,),
        (3*h, 3*h + h/n), -x**(2*n)/4)/(2*n*gamma(3*h + h/n))
    got = integrate(sin(x**n), x)
    assert got == ans
    _x = Symbol('x', zero=False)
    reps = {x: _x}
    assert integrate(sin(_x**n), _x) == ans.xreplace(reps).expand()


def test_issue_17671():
    assert integrate(log(log(x)) / x**2, [x, 1, oo]) == -EulerGamma
    assert integrate(log(log(x)) / x**3, [x, 1, oo]) == -log(2)/2 - EulerGamma/2
    assert integrate(log(log(x)) / x**10, [x, 1, oo]) == -log(9)/9 - EulerGamma/9


def test_issue_2975():
    w = Symbol('w')
    C = Symbol('C')
    y = Symbol('y')
    assert integrate(1/(y**2+C)**(S(3)/2), (y, -w/2, w/2)) == w/(C**(S(3)/2)*sqrt(1 + w**2/(4*C)))


def test_issue_7827():
    x, n, M = symbols('x n M')
    N = Symbol('N', integer=True)
    assert integrate(summation(x*n, (n, 1, N)), x) == x**2*(N**2/4 + N/4)
    assert integrate(summation(x*sin(n), (n,1,N)), x) == \
        Sum(x**2*sin(n)/2, (n, 1, N))
    assert integrate(summation(sin(n*x), (n,1,N)), x) == \
        Sum(Piecewise((-cos(n*x)/n, Ne(n, 0)), (0, True)), (n, 1, N))
    assert integrate(integrate(summation(sin(n*x), (n,1,N)), x), x) == \
        Piecewise((Sum(Piecewise((-sin(n*x)/n**2, Ne(n, 0)), (-x/n, True)),
        (n, 1, N)), (n > -oo) & (n < oo) & Ne(n, 0)), (0, True))
    assert integrate(Sum(x, (n, 1, M)), x) == M*x**2/2
    raises(ValueError, lambda: integrate(Sum(x, (x, y, n)), y))
    raises(ValueError, lambda: integrate(Sum(x, (x, 1, n)), n))
    raises(ValueError, lambda: integrate(Sum(x, (x, 1, y)), x))


def test_issue_4231():
    f = (1 + 2*x + sqrt(x + log(x))*(1 + 3*x) + x**2)/(x*(x + sqrt(x + log(x)))*sqrt(x + log(x)))
    assert integrate(f, x) == 2*sqrt(x + log(x)) + 2*log(x + sqrt(x + log(x)))


def test_issue_17841():
    f = diff(1/(x**2+x+I), x)
    assert integrate(f, x) == 1/(x**2 + x + I)


def test_issue_21034():
    x = Symbol('x', real=True, nonzero=True)
    f1 = x*(-x**4/asin(5)**4 - x*sinh(x + log(asin(5))) + 5)
    f2 = (x + cosh(cos(4)))/(x*(x + 1/(12*x)))

    assert integrate(f1, x) == \
        -x**6/(6*asin(5)**4) - x**2*cosh(x + log(asin(5))) + 5*x**2/2 + 2*x*sinh(x + log(asin(5))) - 2*cosh(x + log(asin(5)))

    assert integrate(f2, x) == \
        log(x**2 + S(1)/12)/2 + 2*sqrt(3)*cosh(cos(4))*atan(2*sqrt(3)*x)


def test_issue_4187():
    assert integrate(log(x)*exp(-x), x) == Ei(-x) - exp(-x)*log(x)
    assert integrate(log(x)*exp(-x), (x, 0, oo)) == -EulerGamma


def test_issue_5547():
    L = Symbol('L')
    z = Symbol('z')
    r0 = Symbol('r0')
    R0 = Symbol('R0')

    assert integrate(r0**2*cos(z)**2, (z, -L/2, L/2)) == -r0**2*(-L/4 -
                    sin(L/2)*cos(L/2)/2) + r0**2*(L/4 + sin(L/2)*cos(L/2)/2)

    assert integrate(r0**2*cos(R0*z)**2, (z, -L/2, L/2)) == Piecewise(
        (-r0**2*(-L*R0/4 - sin(L*R0/2)*cos(L*R0/2)/2)/R0 +
         r0**2*(L*R0/4 + sin(L*R0/2)*cos(L*R0/2)/2)/R0, (R0 > -oo) & (R0 < oo) & Ne(R0, 0)),
        (L*r0**2, True))

    w = 2*pi*z/L

    sol = sqrt(2)*sqrt(L)*r0**2*fresnelc(sqrt(2)*sqrt(L))*gamma(S.One/4)/(16*gamma(S(5)/4)) + L*r0**2/2

    assert integrate(r0**2*cos(w*z)**2, (z, -L/2, L/2)) == sol


def test_issue_15810():
    assert integrate(1/(2**(2*x/3) + 1), (x, 0, oo)) == Rational(3, 2)


def test_issue_21024():
    x = Symbol('x', real=True, nonzero=True)
    f = log(x)*log(4*x) + log(3*x + exp(2))
    F = x*log(x)**2 + x*log(3*x + exp(2)) + x*(1 - 2*log(2)) + \
        (-2*x + 2*x*log(2))*log(x) + exp(2)*log(3*x + exp(2))/3
    assert F == integrate(f, x)

    f = (x + exp(3))/x**2
    F = log(x) - exp(3)/x
    assert F == integrate(f, x)

    f = (x**2 + exp(5))/x
    F = x**2/2 + exp(5)*log(x)
    assert F == integrate(f, x)

    f = x/(2*x + tanh(1))
    F = x/2 - log(2*x + tanh(1))*tanh(1)/4
    assert F == integrate(f, x)

    f = x - sinh(4)/x
    F = x**2/2 - log(x)*sinh(4)
    assert F == integrate(f, x)

    f = log(x + exp(5)/x)
    F = x*log(x + exp(5)/x) - x + 2*exp(Rational(5, 2))*atan(x*exp(Rational(-5, 2)))
    assert F == integrate(f, x)

    f = x**5/(x + E)
    F = x**5/5 - E*x**4/4 + x**3*exp(2)/3 - x**2*exp(3)/2 + x*exp(4) - exp(5)*log(x + E)
    assert F == integrate(f, x)

    f = 4*x/(x + sinh(5))
    F = 4*x - 4*log(x + sinh(5))*sinh(5)
    assert F == integrate(f, x)

    f = x**2/(2*x + sinh(2))
    F = x**2/4 - x*sinh(2)/4 + log(2*x + sinh(2))*sinh(2)**2/8
    assert F == integrate(f, x)

    f = -x**2/(x + E)
    F = -x**2/2 + E*x - exp(2)*log(x + E)
    assert F == integrate(f, x)

    f = (2*x + 3)*exp(5)/x
    F = 2*x*exp(5) + 3*exp(5)*log(x)
    assert F == integrate(f, x)

    f = x + 2 + cosh(3)/x
    F = x**2/2 + 2*x + log(x)*cosh(3)
    assert F == integrate(f, x)

    f = x - tanh(1)/x**3
    F = x**2/2 + tanh(1)/(2*x**2)
    assert F == integrate(f, x)

    f = (3*x - exp(6))/x
    F = 3*x - exp(6)*log(x)
    assert F == integrate(f, x)

    f = x**4/(x + exp(5))**2 + x
    F = x**3/3 + x**2*(Rational(1, 2) - exp(5)) + 3*x*exp(10) - 4*exp(15)*log(x + exp(5)) - exp(20)/(x + exp(5))
    assert F == integrate(f, x)

    f = x*(x + exp(10)/x**2) + x
    F = x**3/3 + x**2/2 + exp(10)*log(x)
    assert F == integrate(f, x)

    f = x + x/(5*x + sinh(3))
    F = x**2/2 + x/5 - log(5*x + sinh(3))*sinh(3)/25
    assert F == integrate(f, x)

    f = (x + exp(3))/(2*x**2 + 2*x)
    F = exp(3)*log(x)/2 - exp(3)*log(x + 1)/2 + log(x + 1)/2
    assert F == integrate(f, x).expand()

    f = log(x + 4*sinh(4))
    F = x*log(x + 4*sinh(4)) - x + 4*log(x + 4*sinh(4))*sinh(4)
    assert F == integrate(f, x)

    f = -x + 20*(exp(-5) - atan(4)/x)**3*sin(4)/x
    F = (-x**2*exp(15)/2 + 20*log(x)*sin(4) - (-180*x**2*exp(5)*sin(4)*atan(4) + 90*x*exp(10)*sin(4)*atan(4)**2 - \
        20*exp(15)*sin(4)*atan(4)**3)/(3*x**3))*exp(-15)
    assert F == integrate(f, x)

    f = 2*x**2*exp(-4) + 6/x
    F_true = (2*x**3/3 + 6*exp(4)*log(x))*exp(-4)
    assert F_true == integrate(f, x)


def test_issue_21721():
    a = Symbol('a')
    assert integrate(1/(pi*(1+(x-a)**2)),(x,-oo,oo)).expand() == \
    -Heaviside(im(a) - 1, 0) + Heaviside(im(a) + 1, 0)


def test_issue_21831():
    theta = symbols('theta')
    assert integrate(cos(3*theta)/(5-4*cos(theta)), (theta, 0, 2*pi)) == pi/12
    integrand = cos(2*theta)/(5 - 4*cos(theta))
    assert integrate(integrand, (theta, 0, 2*pi)) == pi/6


@slow
def test_issue_22033_integral():
    assert integrate((x**2 - Rational(1, 4))**2 * sqrt(1 - x**2), (x, -1, 1)) == pi/32


@slow
def test_issue_21671():
    assert integrate(1,(z,x**2+y**2,2-x**2-y**2),(y,-sqrt(1-x**2),sqrt(1-x**2)),(x,-1,1)) == pi
    assert integrate(-4*(1 - x**2)**(S(3)/2)/3 + 2*sqrt(1 - x**2)*(2 - 2*x**2), (x, -1, 1)) == pi


def test_issue_18527():
    # The manual integrator can not currently solve this. Assert that it does
    # not give an incorrect result involving Abs when x has real assumptions.
    xr = symbols('xr', real=True)
    expr = (cos(x)/(4+(sin(x))**2))
    res_real = integrate(expr.subs(x, xr), xr, manual=True).subs(xr, x)
    assert integrate(expr, x, manual=True) == res_real == Integral(expr, x)


def test_issue_23718():
    f = 1/(b*cos(x) + a*sin(x))
    Fpos = (-log(-a/b + tan(x/2) - sqrt(a**2 + b**2)/b)/sqrt(a**2 + b**2)
            +log(-a/b + tan(x/2) + sqrt(a**2 + b**2)/b)/sqrt(a**2 + b**2))
    F = Piecewise(
        # XXX: The zoo case here is for a=b=0 so it should just be zoo or maybe
        # it doesn't really need to be included at all given that the original
        # integrand is really undefined in that case anyway.
        (zoo*(-log(tan(x/2) - 1) + log(tan(x/2) + 1)),  Eq(a, 0) & Eq(b, 0)),
        (log(tan(x/2))/a,                               Eq(b, 0)),
        (-I/(-I*b*sin(x) + b*cos(x)),                   Eq(a, -I*b)),
        (I/(I*b*sin(x) + b*cos(x)),                     Eq(a,  I*b)),
        (Fpos,                                          True),
    )
    assert integrate(f, x) == F

    ap, bp = symbols('a, b', positive=True)
    rep = {a: ap, b: bp}
    assert integrate(f.subs(rep), x) == Fpos.subs(rep)


def test_issue_23566():
    i = integrate(1/sqrt(x**2-1), (x, -2, -1))
    assert i == -log(2 - sqrt(3))
    assert math.isclose(i.n(), 1.31695789692482)


def test_pr_23583():
    # This result from meijerg is wrong. Check whether new result is correct when this test fail.
    assert integrate(1/sqrt((x - I)**2-1)) == Piecewise((acosh(x - I), Abs((x - I)**2) > 1), (-I*asin(x - I), True))


def test_issue_7264():
    assert integrate(exp(x)*sqrt(1 + exp(2*x))) == sqrt(exp(2*x) + 1)*exp(x)/2 + asinh(exp(x))/2


def test_issue_11254a():
    assert integrate(sech(x), (x, 0, 1)) == 2*atan(tanh(S.Half))


def test_issue_11254b():
    assert integrate(csch(x), x) == log(tanh(x/2))
    assert integrate(csch(x), (x, 0, 1)) == oo


def test_issue_11254d():
    # (sech(x)**2).rewrite(sinh)
    assert integrate(-1/sinh(x + I*pi/2, evaluate=False)**2, x) == -2/(exp(2*x) + 1)
    assert integrate(cosh(x)**(-2), x) == 2*tanh(x/2)/(tanh(x/2)**2 + 1)


def test_issue_22863():
    i = integrate((3*x**3-x**2+2*x-4)/sqrt(x**2-3*x+2), (x, 0, 1))
    assert i == -101*sqrt(2)/8 - 135*log(3 - 2*sqrt(2))/16
    assert math.isclose(i.n(), -2.98126694400554)


def test_issue_9723():
    assert integrate(sqrt(x + sqrt(x))) == \
        2*sqrt(sqrt(x) + x)*(sqrt(x)/12 + x/3 - S(1)/8) + log(2*sqrt(x) + 2*sqrt(sqrt(x) + x) + 1)/8
    assert integrate(sqrt(2*x+3+sqrt(4*x+5))**3) == \
        sqrt(2*x + sqrt(4*x + 5) + 3) * \
           (9*x/10 + 11*(4*x + 5)**(S(3)/2)/40 + sqrt(4*x + 5)/40 + (4*x + 5)**2/10 + S(11)/10)/2


def test_issue_23704():
    # XXX: This is testing that an exception is not raised in risch Ideally
    # manualintegrate (manual=True) would be able to compute this but
    # manualintegrate is very slow for this example so we don't test that here.
    assert (integrate(log(x)/x**2/(c*x**2+b*x+a),x, risch=True)
        == NonElementaryIntegral(log(x)/(a*x**2 + b*x**3 + c*x**4), x))


def test_exp_substitution():
    assert integrate(1/sqrt(1-exp(2*x))) == log(sqrt(1 - exp(2*x)) - 1)/2 - log(sqrt(1 - exp(2*x)) + 1)/2


def test_hyperbolic():
    assert integrate(coth(x)) == x - log(tanh(x) + 1) + log(tanh(x))
    assert integrate(sech(x)) == 2*atan(tanh(x/2))
    assert integrate(csch(x)) == log(tanh(x/2))


def test_nested_pow():
    assert integrate(sqrt(x**2)) == x*sqrt(x**2)/2
    assert integrate(sqrt(x**(S(5)/3))) == 6*x*sqrt(x**(S(5)/3))/11
    assert integrate(1/sqrt(x**2)) == x*log(x)/sqrt(x**2)
    assert integrate(x*sqrt(x**(-4))) == x**2*sqrt(x**-4)*log(x)


def test_sqrt_quadratic():
    assert integrate(1/sqrt(3*x**2+4*x+5)) == sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/3
    assert integrate(1/sqrt(-3*x**2+4*x+5)) == sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/3
    assert integrate(1/sqrt(3*x**2+4*x-5)) == sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/3
    assert integrate(1/sqrt(4*x**2-4*x+1)) == (x - S.Half)*log(x - S.Half)/(2*sqrt((x - S.Half)**2))
    assert integrate(1/sqrt(a+b*x+c*x**2), x) == \
        Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(c, 0) & Ne(a - b**2/(4*c), 0)),
                  ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), Ne(c, 0)),
                  (2*sqrt(a + b*x)/b, Ne(b, 0)), (x/sqrt(a), True))

    assert integrate((7*x+6)/sqrt(3*x**2+4*x+5)) == \
           7*sqrt(3*x**2 + 4*x + 5)/3 + 4*sqrt(3)*asinh(3*sqrt(11)*(x + S(2)/3)/11)/9
    assert integrate((7*x+6)/sqrt(-3*x**2+4*x+5)) == \
           -7*sqrt(-3*x**2 + 4*x + 5)/3 + 32*sqrt(3)*asin(3*sqrt(19)*(x - S(2)/3)/19)/9
    assert integrate((7*x+6)/sqrt(3*x**2+4*x-5)) == \
           7*sqrt(3*x**2 + 4*x - 5)/3 + 4*sqrt(3)*log(6*x + 2*sqrt(3)*sqrt(3*x**2 + 4*x - 5) + 4)/9
    assert integrate((d+e*x)/sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((-b*e/(2*c) + d) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   e*sqrt(a + b*x + c*x**2)/c, Ne(c, 0)),
                  ((2*d*sqrt(a + b*x) + 2*e*(-a*sqrt(a + b*x) + (a + b*x)**(S(3)/2)/3)/b)/b, Ne(b, 0)),
                  ((d*x + e*x**2/2)/sqrt(a), True))

    assert integrate((3*x**3-x**2+2*x-4)/sqrt(x**2-3*x+2)) == \
           sqrt(x**2 - 3*x + 2)*(x**2 + 13*x/4 + S(101)/8) + 135*log(2*x + 2*sqrt(x**2 - 3*x + 2) - 3)/16

    assert integrate(sqrt(53225*x**2-66732*x+23013)) == \
           (x/2 - S(16683)/53225)*sqrt(53225*x**2 - 66732*x + 23013) + \
           111576969*sqrt(2129)*asinh(53225*x/10563 - S(11122)/3521)/1133160250
    assert integrate(sqrt(a+b*x+c*x**2), x) == \
        Piecewise(((a/2 - b**2/(8*c)) *
                   Piecewise((log(b + 2*sqrt(c)*sqrt(a + b*x + c*x**2) + 2*c*x)/sqrt(c), Ne(a - b**2/(4*c), 0)),
                             ((b/(2*c) + x)*log(b/(2*c) + x)/sqrt(c*(b/(2*c) + x)**2), True)) +
                   (b/(4*c) + x/2)*sqrt(a + b*x + c*x**2), Ne(c, 0)),
                  (2*(a + b*x)**(S(3)/2)/(3*b), Ne(b, 0)),
                  (sqrt(a)*x, True))

    assert integrate(x*sqrt(x**2+2*x+4)) == \
        (x**2/3 + x/6 + S(5)/6)*sqrt(x**2 + 2*x + 4) - 3*asinh(sqrt(3)*(x + 1)/3)/2


def test_mul_pow_derivative():
    assert integrate(x*sec(x)*tan(x)) == x*sec(x) - log(tan(x) + sec(x))
    assert integrate(x*sec(x)**2, x) == x*tan(x) + log(cos(x))
    assert integrate(x**3*Derivative(f(x), (x, 4))) == \
           x**3*Derivative(f(x), (x, 3)) - 3*x**2*Derivative(f(x), (x, 2)) + 6*x*Derivative(f(x), x) - 6*f(x)


def test_issue_20782():
    fun1 = Piecewise((0, x < 0.0), (1, True))
    fun2 = -Piecewise((0, x < 1.0), (1, True))
    fun_sum = fun1 + fun2
    L = (x, -float('Inf'), 1)

    assert integrate(fun1, L) == 1
    assert integrate(fun2, L) == 0
    assert integrate(-fun1, L) == -1
    assert integrate(-fun2, L) == 0
    assert integrate(fun_sum, L) == 1.
    assert integrate(-fun_sum, L) == -1.


def test_issue_20781():
    P = lambda a: Piecewise((0, x < a), (1, x >= a))
    f = lambda a: P(int(a)) + P(float(a))
    L = (x, -float('Inf'), x)
    f1 = integrate(f(1), L)
    assert f1 == 2*x - Min(1.0, x) - Min(x, Max(1.0, 1, evaluate=False))
    # XXX is_zero is True for S(0) and Float(0) and this is baked into
    # the code more deeply than the issue of Float(0) != S(0)
    assert integrate(f(0), (x, -float('Inf'), x)
        ) == 2*x - 2*Min(0, x)


@slow
def test_issue_19427():
    # <https://github.com/sympy/sympy/issues/19427>
    x = Symbol("x")

    # Have always been okay:
    assert integrate((x ** 4) * sqrt(1 - x ** 2), (x, -1, 1)) == pi / 16
    assert integrate((-2 * x ** 2) * sqrt(1 - x ** 2), (x, -1, 1)) == -pi / 4
    assert integrate((1) * sqrt(1 - x ** 2), (x, -1, 1)) == pi / 2

    # Sum of the above, used to incorrectly return 0 for a while:
    assert integrate((x ** 4 - 2 * x ** 2 + 1) * sqrt(1 - x ** 2), (x, -1, 1)) == 5 * pi / 16


def test_issue_23942():
    I1 = Integral(1/sqrt(a*(1 + x)**3 + (1 + x)**2), (x, 0, z))
    assert I1.series(a, 1, n=1) == Integral(1/sqrt(x**3 + 4*x**2 + 5*x + 2), (x, 0, z)) + O(a - 1, (a, 1))
    I2 = Integral(1/sqrt(a*(4 - x)**4 + (5 + x)**2), (x, 0, z))
    assert I2.series(a, 2, n=1) == Integral(1/sqrt(2*x**4 - 32*x**3 + 193*x**2 - 502*x + 537), (x, 0, z)) + O(a - 2, (a, 2))


def test_issue_25886():
    # https://github.com/sympy/sympy/issues/25886
    f = (1-x)*exp(0.937098661j*x)
    F_exp = (1.0*(-1.0671234968289*I*y
             + 1.13875255748434
             + 1.0671234968289*I)*exp(0.937098661*I*y)
            - 1.13875255748434*exp(0.937098661*I))
    F = integrate(f, (x, y, 1.0))
    assert F.is_same(F_exp, math.isclose)


def test_old_issues():
    # https://github.com/sympy/sympy/issues/5212
    I1 = integrate(cos(log(x**2))/x)
    assert I1 == sin(log(x**2))/2
    # https://github.com/sympy/sympy/issues/5462
    I2 = integrate(1/(x**2+y**2)**(Rational(3,2)),x)
    assert I2 == x/(y**3*sqrt(x**2/y**2 + 1))
    # https://github.com/sympy/sympy/issues/6278
    I3 = integrate(1/(cos(x)+2),(x,0,2*pi))
    assert I3 == 2*sqrt(3)*pi/3


def test_integral_issue_26566():
    # Define the symbols
    x = symbols('x', real=True)
    a = symbols('a', real=True, positive=True)

    # Define the integral expression
    integral_expr = sin(a * (x + pi))**2
    symbolic_result = integrate(integral_expr, (x, -pi, -pi/2))

    # Known correct result
    correct_result = pi / 4

    # Substitute a specific value for 'a' to evaluate both results
    a_value = 1
    numeric_symbolic_result = symbolic_result.subs(a, a_value).evalf()
    numeric_correct_result = correct_result.evalf()

    # Assert that the symbolic result matches the correct value
    assert simplify(numeric_symbolic_result - numeric_correct_result) == 0


def test_definite_integral_with_floats_issue_27231():
    # Define the symbol and the integral expression
    x = symbols('x', real=True)
    integral_expr = sqrt(1 - 0.5625 * (x + 0.333333333333333) ** 2)

    # Perform the definite integral with the known limits
    result_symbolic = integrate(integral_expr, (x, -1, 1))
    result_numeric = result_symbolic.evalf()

    # Expected result with higher precision
    expected_result = sqrt(3) / 6 + 4 * pi / 9

    # Verify that the result is approximately equal within a larger tolerance
    assert abs(result_numeric - expected_result.evalf()) < 1e-8


def test_issue_27374():
    #https://github.com/sympy/sympy/issues/27374
    r = sqrt(x**2 + z**2)
    u = erf(a*r/sqrt(2))/r
    Ec = diff(u, z, z).subs([(x, sqrt(b*b-z*z))])
    expected_result = -2*sqrt(2)*b*a**3*exp(-b**2*a**2/2)/(3*sqrt(pi))
    assert simplify(integrate(Ec, (z, -b, b))) == expected_result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\arithmetic\test_timedelta64.py ===
# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.errors import (
    OutOfBoundsDatetime,
    PerformanceWarning,
)

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    offsets,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)


def assert_dtype(obj, expected_dtype):
    """
    Helper to check the dtype for a Series, Index, or single-column DataFrame.
    """
    dtype = tm.get_dtype(obj)

    assert dtype == expected_dtype


def get_expected_name(box, names):
    if box is DataFrame:
        # Since we are operating with a DataFrame and a non-DataFrame,
        # the non-DataFrame is cast to Series and its name ignored.
        exname = names[0]
    elif box in [tm.to_array, pd.array]:
        exname = names[1]
    else:
        exname = names[2]
    return exname


# ------------------------------------------------------------------
# Timedelta64[ns] dtype Comparisons


class TestTimedelta64ArrayLikeComparisons:
    # Comparison tests for timedelta64[ns] vectors fully parametrized over
    #  DataFrame/Series/TimedeltaIndex/TimedeltaArray.  Ideally all comparison
    #  tests will eventually end up here.

    def test_compare_timedelta64_zerodim(self, box_with_array):
        # GH#26689 should unbox when comparing with zerodim array
        box = box_with_array
        xbox = box_with_array if box_with_array not in [Index, pd.array] else np.ndarray

        tdi = timedelta_range("2h", periods=4)
        other = np.array(tdi.to_numpy()[0])

        tdi = tm.box_expected(tdi, box)
        res = tdi <= other
        expected = np.array([True, False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize(
        "td_scalar",
        [
            timedelta(days=1),
            Timedelta(days=1),
            Timedelta(days=1).to_timedelta64(),
            offsets.Hour(24),
        ],
    )
    def test_compare_timedeltalike_scalar(self, box_with_array, td_scalar):
        # regression test for GH#5963
        box = box_with_array
        xbox = box if box not in [Index, pd.array] else np.ndarray

        ser = Series([timedelta(days=1), timedelta(days=2)])
        ser = tm.box_expected(ser, box)
        actual = ser > td_scalar
        expected = Series([False, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "invalid",
        [
            345600000000000,
            "a",
            Timestamp("2021-01-01"),
            Timestamp("2021-01-01").now("UTC"),
            Timestamp("2021-01-01").now().to_datetime64(),
            Timestamp("2021-01-01").now().to_pydatetime(),
            Timestamp("2021-01-01").date(),
            np.array(4),  # zero-dim mismatched dtype
        ],
    )
    def test_td64_comparisons_invalid(self, box_with_array, invalid):
        # GH#13624 for str
        box = box_with_array

        rng = timedelta_range("1 days", periods=10)
        obj = tm.box_expected(rng, box)

        assert_invalid_comparison(obj, invalid, box)

    @pytest.mark.parametrize(
        "other",
        [
            list(range(10)),
            np.arange(10),
            np.arange(10).astype(np.float32),
            np.arange(10).astype(object),
            pd.date_range("1970-01-01", periods=10, tz="UTC").array,
            np.array(pd.date_range("1970-01-01", periods=10)),
            list(pd.date_range("1970-01-01", periods=10)),
            pd.date_range("1970-01-01", periods=10).astype(object),
            pd.period_range("1971-01-01", freq="D", periods=10).array,
            pd.period_range("1971-01-01", freq="D", periods=10).astype(object),
        ],
    )
    def test_td64arr_cmp_arraylike_invalid(self, other, box_with_array):
        # We don't parametrize this over box_with_array because listlike
        #  other plays poorly with assert_invalid_comparison reversed checks

        rng = timedelta_range("1 days", periods=10)._data
        rng = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(rng, other, box_with_array)

    def test_td64arr_cmp_mixed_invalid(self):
        rng = timedelta_range("1 days", periods=5)._data
        other = np.array([0, 1, 2, rng[3], Timestamp("2021-01-01")])

        result = rng == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = rng != other
        tm.assert_numpy_array_equal(result, ~expected)

        msg = "Invalid comparison between|Cannot compare type|not supported between"
        with pytest.raises(TypeError, match=msg):
            rng < other
        with pytest.raises(TypeError, match=msg):
            rng > other
        with pytest.raises(TypeError, match=msg):
            rng <= other
        with pytest.raises(TypeError, match=msg):
            rng >= other


class TestTimedelta64ArrayComparisons:
    # TODO: All of these need to be parametrized over box

    @pytest.mark.parametrize("dtype", [None, object])
    def test_comp_nat(self, dtype):
        left = TimedeltaIndex([Timedelta("1 days"), NaT, Timedelta("3 days")])
        right = TimedeltaIndex([NaT, NaT, Timedelta("3 days")])

        lhs, rhs = left, right
        if dtype is object:
            lhs, rhs = left.astype(object), right.astype(object)

        result = rhs == lhs
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = rhs != lhs
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs == NaT, expected)
        tm.assert_numpy_array_equal(NaT == rhs, expected)

        expected = np.array([True, True, True])
        tm.assert_numpy_array_equal(lhs != NaT, expected)
        tm.assert_numpy_array_equal(NaT != lhs, expected)

        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs < NaT, expected)
        tm.assert_numpy_array_equal(NaT > lhs, expected)

    @pytest.mark.parametrize(
        "idx2",
        [
            TimedeltaIndex(
                ["2 day", "2 day", NaT, NaT, "1 day 00:00:02", "5 days 00:00:03"]
            ),
            np.array(
                [
                    np.timedelta64(2, "D"),
                    np.timedelta64(2, "D"),
                    np.timedelta64("nat"),
                    np.timedelta64("nat"),
                    np.timedelta64(1, "D") + np.timedelta64(2, "s"),
                    np.timedelta64(5, "D") + np.timedelta64(3, "s"),
                ]
            ),
        ],
    )
    def test_comparisons_nat(self, idx2):
        idx1 = TimedeltaIndex(
            [
                "1 day",
                NaT,
                "1 day 00:00:01",
                NaT,
                "1 day 00:00:01",
                "5 day 00:00:03",
            ]
        )
        # Check pd.NaT is handles as the same as np.nan
        result = idx1 < idx2
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = idx2 > idx1
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = idx1 <= idx2
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)

        result = idx2 >= idx1
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)

        result = idx1 == idx2
        expected = np.array([False, False, False, False, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = idx1 != idx2
        expected = np.array([True, True, True, True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    # TODO: better name
    def test_comparisons_coverage(self):
        rng = timedelta_range("1 days", periods=10)

        result = rng < rng[3]
        expected = np.array([True, True, True] + [False] * 7)
        tm.assert_numpy_array_equal(result, expected)

        result = rng == list(rng)
        exp = rng == rng
        tm.assert_numpy_array_equal(result, exp)


# ------------------------------------------------------------------
# Timedelta64[ns] dtype Arithmetic Operations


class TestTimedelta64ArithmeticUnsorted:
    # Tests moved from type-specific test files but not
    #  yet sorted/parametrized/de-duplicated

    def test_ufunc_coercions(self):
        # normal ops are also tested in tseries/test_timedeltas.py
        idx = TimedeltaIndex(["2h", "4h", "6h", "8h", "10h"], freq="2h", name="x")

        for result in [idx * 2, np.multiply(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["4h", "8h", "12h", "16h", "20h"], freq="4h", name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq == "4h"

        for result in [idx / 2, np.divide(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["1h", "2h", "3h", "4h", "5h"], freq="h", name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq == "h"

        for result in [-idx, np.negative(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(
                ["-2h", "-4h", "-6h", "-8h", "-10h"], freq="-2h", name="x"
            )
            tm.assert_index_equal(result, exp)
            assert result.freq == "-2h"

        idx = TimedeltaIndex(["-2h", "-1h", "0h", "1h", "2h"], freq="h", name="x")
        for result in [abs(idx), np.absolute(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["2h", "1h", "0h", "1h", "2h"], freq=None, name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq is None

    def test_subtraction_ops(self):
        # with datetimes/timedelta and tdi/dti
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")
        td = Timedelta("1 days")
        dt = Timestamp("20130101")

        msg = "cannot subtract a datelike from a TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi - dt
        with pytest.raises(TypeError, match=msg):
            tdi - dti

        msg = r"unsupported operand type\(s\) for -"
        with pytest.raises(TypeError, match=msg):
            td - dt

        msg = "(bad|unsupported) operand type for unary"
        with pytest.raises(TypeError, match=msg):
            td - dti

        result = dt - dti
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"], name="bar")
        tm.assert_index_equal(result, expected)

        result = dti - dt
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"], name="bar")
        tm.assert_index_equal(result, expected)

        result = tdi - td
        expected = TimedeltaIndex(["0 days", NaT, "1 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = td - tdi
        expected = TimedeltaIndex(["0 days", NaT, "-1 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = dti - td
        expected = DatetimeIndex(
            ["20121231", "20130101", "20130102"], dtype="M8[ns]", freq="D", name="bar"
        )
        tm.assert_index_equal(result, expected)

        result = dt - tdi
        expected = DatetimeIndex(
            ["20121231", NaT, "20121230"], dtype="M8[ns]", name="foo"
        )
        tm.assert_index_equal(result, expected)

    def test_subtraction_ops_with_tz(self, box_with_array):
        # check that dt/dti subtraction ops with tz are validated
        dti = pd.date_range("20130101", periods=3)
        dti = tm.box_expected(dti, box_with_array)
        ts = Timestamp("20130101")
        dt = ts.to_pydatetime()
        dti_tz = pd.date_range("20130101", periods=3).tz_localize("US/Eastern")
        dti_tz = tm.box_expected(dti_tz, box_with_array)
        ts_tz = Timestamp("20130101").tz_localize("US/Eastern")
        ts_tz2 = Timestamp("20130101").tz_localize("CET")
        dt_tz = ts_tz.to_pydatetime()
        td = Timedelta("1 days")

        def _check(result, expected):
            assert result == expected
            assert isinstance(result, Timedelta)

        # scalars
        result = ts - ts
        expected = Timedelta("0 days")
        _check(result, expected)

        result = dt_tz - ts_tz
        expected = Timedelta("0 days")
        _check(result, expected)

        result = ts_tz - dt_tz
        expected = Timedelta("0 days")
        _check(result, expected)

        # tz mismatches
        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects."
        with pytest.raises(TypeError, match=msg):
            dt_tz - ts
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt_tz - dt
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt - dt_tz
        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects."
        with pytest.raises(TypeError, match=msg):
            ts - dt_tz
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - ts
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - dt

        msg = "Cannot subtract tz-naive and tz-aware"
        # with dti
        with pytest.raises(TypeError, match=msg):
            dti - ts_tz
        with pytest.raises(TypeError, match=msg):
            dti_tz - ts

        result = dti_tz - dt_tz
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        result = dt_tz - dti_tz
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        result = dti_tz - ts_tz
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        result = ts_tz - dti_tz
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        result = td - td
        expected = Timedelta("0 days")
        _check(result, expected)

        result = dti_tz - td
        expected = DatetimeIndex(
            ["20121231", "20130101", "20130102"], tz="US/Eastern"
        ).as_unit("ns")
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    def test_dti_tdi_numeric_ops(self):
        # These are normally union/diff set-like ops
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")

        result = tdi - tdi
        expected = TimedeltaIndex(["0 days", NaT, "0 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = tdi + tdi
        expected = TimedeltaIndex(["2 days", NaT, "4 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = dti - tdi  # name will be reset
        expected = DatetimeIndex(["20121231", NaT, "20130101"], dtype="M8[ns]")
        tm.assert_index_equal(result, expected)

    def test_addition_ops(self):
        # with datetimes/timedelta and tdi/dti
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")
        td = Timedelta("1 days")
        dt = Timestamp("20130101")

        result = tdi + dt
        expected = DatetimeIndex(
            ["20130102", NaT, "20130103"], dtype="M8[ns]", name="foo"
        )
        tm.assert_index_equal(result, expected)

        result = dt + tdi
        expected = DatetimeIndex(
            ["20130102", NaT, "20130103"], dtype="M8[ns]", name="foo"
        )
        tm.assert_index_equal(result, expected)

        result = td + tdi
        expected = TimedeltaIndex(["2 days", NaT, "3 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = tdi + td
        expected = TimedeltaIndex(["2 days", NaT, "3 days"], name="foo")
        tm.assert_index_equal(result, expected)

        # unequal length
        msg = "cannot add indices of unequal length"
        with pytest.raises(ValueError, match=msg):
            tdi + dti[0:1]
        with pytest.raises(ValueError, match=msg):
            tdi[0:1] + dti

        # random indexes
        msg = "Addition/subtraction of integers and integer-arrays"
        with pytest.raises(TypeError, match=msg):
            tdi + Index([1, 2, 3], dtype=np.int64)

        # this is a union!
        # FIXME: don't leave commented-out
        # pytest.raises(TypeError, lambda : Index([1,2,3]) + tdi)

        result = tdi + dti  # name will be reset
        expected = DatetimeIndex(["20130102", NaT, "20130105"], dtype="M8[ns]")
        tm.assert_index_equal(result, expected)

        result = dti + tdi  # name will be reset
        expected = DatetimeIndex(["20130102", NaT, "20130105"], dtype="M8[ns]")
        tm.assert_index_equal(result, expected)

        result = dt + td
        expected = Timestamp("20130102")
        assert result == expected

        result = td + dt
        expected = Timestamp("20130102")
        assert result == expected

    # TODO: Needs more informative name, probably split up into
    # more targeted tests
    @pytest.mark.parametrize("freq", ["D", "B"])
    def test_timedelta(self, freq):
        index = pd.date_range("1/1/2000", periods=50, freq=freq)

        shifted = index + timedelta(1)
        back = shifted + timedelta(-1)
        back = back._with_freq("infer")
        tm.assert_index_equal(index, back)

        if freq == "D":
            expected = pd.tseries.offsets.Day(1)
            assert index.freq == expected
            assert shifted.freq == expected
            assert back.freq == expected
        else:  # freq == 'B'
            assert index.freq == pd.tseries.offsets.BusinessDay(1)
            assert shifted.freq is None
            assert back.freq == pd.tseries.offsets.BusinessDay(1)

        result = index - timedelta(1)
        expected = index + timedelta(-1)
        tm.assert_index_equal(result, expected)

    def test_timedelta_tick_arithmetic(self):
        # GH#4134, buggy with timedeltas
        rng = pd.date_range("2013", "2014")
        s = Series(rng)
        result1 = rng - offsets.Hour(1)
        result2 = DatetimeIndex(s - np.timedelta64(100000000))
        result3 = rng - np.timedelta64(100000000)
        result4 = DatetimeIndex(s - offsets.Hour(1))

        assert result1.freq == rng.freq
        result1 = result1._with_freq(None)
        tm.assert_index_equal(result1, result4)

        assert result3.freq == rng.freq
        result3 = result3._with_freq(None)
        tm.assert_index_equal(result2, result3)

    def test_tda_add_sub_index(self):
        # Check that TimedeltaArray defers to Index on arithmetic ops
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"])
        tda = tdi.array

        dti = pd.date_range("1999-12-31", periods=3, freq="D")

        result = tda + dti
        expected = tdi + dti
        tm.assert_index_equal(result, expected)

        result = tda + tdi
        expected = tdi + tdi
        tm.assert_index_equal(result, expected)

        result = tda - tdi
        expected = tdi - tdi
        tm.assert_index_equal(result, expected)

    def test_tda_add_dt64_object_array(self, box_with_array, tz_naive_fixture):
        # Result should be cast back to DatetimeArray
        box = box_with_array

        dti = pd.date_range("2016-01-01", periods=3, tz=tz_naive_fixture)
        dti = dti._with_freq(None)
        tdi = dti - dti

        obj = tm.box_expected(tdi, box)
        other = tm.box_expected(dti, box)

        with tm.assert_produces_warning(PerformanceWarning):
            result = obj + other.astype(object)
        tm.assert_equal(result, other.astype(object))

    # -------------------------------------------------------------
    # Binary operations TimedeltaIndex and timedelta-like

    def test_tdi_iadd_timedeltalike(self, two_hours, box_with_array):
        # only test adding/sub offsets as + is now numeric
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("1 days 02:00:00", "10 days 02:00:00", freq="D")

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        orig_rng = rng
        rng += two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not Index:
            # Check that operation is actually inplace
            tm.assert_equal(orig_rng, expected)

    def test_tdi_isub_timedeltalike(self, two_hours, box_with_array):
        # only test adding/sub offsets as - is now numeric
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("0 days 22:00:00", "9 days 22:00:00")

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        orig_rng = rng
        rng -= two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not Index:
            # Check that operation is actually inplace
            tm.assert_equal(orig_rng, expected)

    # -------------------------------------------------------------

    def test_tdi_ops_attributes(self):
        rng = timedelta_range("2 days", periods=5, freq="2D", name="x")

        result = rng + 1 * rng.freq
        exp = timedelta_range("4 days", periods=5, freq="2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "2D"

        result = rng - 2 * rng.freq
        exp = timedelta_range("-2 days", periods=5, freq="2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "2D"

        result = rng * 2
        exp = timedelta_range("4 days", periods=5, freq="4D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "4D"

        result = rng / 2
        exp = timedelta_range("1 days", periods=5, freq="D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "D"

        result = -rng
        exp = timedelta_range("-2 days", periods=5, freq="-2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "-2D"

        rng = timedelta_range("-2 days", periods=5, freq="D", name="x")

        result = abs(rng)
        exp = TimedeltaIndex(
            ["2 days", "1 days", "0 days", "1 days", "2 days"], name="x"
        )
        tm.assert_index_equal(result, exp)
        assert result.freq is None


class TestAddSubNaTMasking:
    # TODO: parametrize over boxes

    @pytest.mark.parametrize("str_ts", ["1950-01-01", "1980-01-01"])
    def test_tdarr_add_timestamp_nat_masking(self, box_with_array, str_ts):
        # GH#17991 checking for overflow-masking with NaT
        tdinat = pd.to_timedelta(["24658 days 11:15:00", "NaT"])
        tdobj = tm.box_expected(tdinat, box_with_array)

        ts = Timestamp(str_ts)
        ts_variants = [
            ts,
            ts.to_pydatetime(),
            ts.to_datetime64().astype("datetime64[ns]"),
            ts.to_datetime64().astype("datetime64[D]"),
        ]

        for variant in ts_variants:
            res = tdobj + variant
            if box_with_array is DataFrame:
                assert res.iloc[1, 1] is NaT
            else:
                assert res[1] is NaT

    def test_tdi_add_overflow(self):
        # See GH#14068
        # preliminary test scalar analogue of vectorized tests below
        # TODO: Make raised error message more informative and test
        with pytest.raises(OutOfBoundsDatetime, match="10155196800000000000"):
            pd.to_timedelta(106580, "D") + Timestamp("2000")
        with pytest.raises(OutOfBoundsDatetime, match="10155196800000000000"):
            Timestamp("2000") + pd.to_timedelta(106580, "D")

        _NaT = NaT._value + 1
        msg = "Overflow in int64 addition"
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([106580], "D") + Timestamp("2000")
        with pytest.raises(OverflowError, match=msg):
            Timestamp("2000") + pd.to_timedelta([106580], "D")
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([_NaT]) - Timedelta("1 days")
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta(["5 days", _NaT]) - Timedelta("1 days")
        with pytest.raises(OverflowError, match=msg):
            (
                pd.to_timedelta([_NaT, "5 days", "1 hours"])
                - pd.to_timedelta(["7 seconds", _NaT, "4 hours"])
            )

        # These should not overflow!
        exp = TimedeltaIndex([NaT])
        result = pd.to_timedelta([NaT]) - Timedelta("1 days")
        tm.assert_index_equal(result, exp)

        exp = TimedeltaIndex(["4 days", NaT])
        result = pd.to_timedelta(["5 days", NaT]) - Timedelta("1 days")
        tm.assert_index_equal(result, exp)

        exp = TimedeltaIndex([NaT, NaT, "5 hours"])
        result = pd.to_timedelta([NaT, "5 days", "1 hours"]) + pd.to_timedelta(
            ["7 seconds", NaT, "4 hours"]
        )
        tm.assert_index_equal(result, exp)


class TestTimedeltaArraylikeAddSubOps:
    # Tests for timedelta64[ns] __add__, __sub__, __radd__, __rsub__

    def test_sub_nat_retain_unit(self):
        ser = pd.to_timedelta(Series(["00:00:01"])).astype("m8[s]")

        result = ser - NaT
        expected = Series([NaT], dtype="m8[s]")
        tm.assert_series_equal(result, expected)

    # TODO: moved from tests.indexes.timedeltas.test_arithmetic; needs
    #  parametrization+de-duplication
    def test_timedelta_ops_with_missing_values(self):
        # setup
        s1 = pd.to_timedelta(Series(["00:00:01"]))
        s2 = pd.to_timedelta(Series(["00:00:02"]))

        sn = pd.to_timedelta(Series([NaT], dtype="m8[ns]"))

        df1 = DataFrame(["00:00:01"]).apply(pd.to_timedelta)
        df2 = DataFrame(["00:00:02"]).apply(pd.to_timedelta)

        dfn = DataFrame([NaT._value]).apply(pd.to_timedelta)

        scalar1 = pd.to_timedelta("00:00:01")
        scalar2 = pd.to_timedelta("00:00:02")
        timedelta_NaT = pd.to_timedelta("NaT")

        actual = scalar1 + scalar1
        assert actual == scalar2
        actual = scalar2 - scalar1
        assert actual == scalar1

        actual = s1 + s1
        tm.assert_series_equal(actual, s2)
        actual = s2 - s1
        tm.assert_series_equal(actual, s1)

        actual = s1 + scalar1
        tm.assert_series_equal(actual, s2)
        actual = scalar1 + s1
        tm.assert_series_equal(actual, s2)
        actual = s2 - scalar1
        tm.assert_series_equal(actual, s1)
        actual = -scalar1 + s2
        tm.assert_series_equal(actual, s1)

        actual = s1 + timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual = timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)
        actual = s1 - timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual = -timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            s1 + np.nan
        with pytest.raises(TypeError, match=msg):
            np.nan + s1
        with pytest.raises(TypeError, match=msg):
            s1 - np.nan
        with pytest.raises(TypeError, match=msg):
            -np.nan + s1

        actual = s1 + NaT
        tm.assert_series_equal(actual, sn)
        actual = s2 - NaT
        tm.assert_series_equal(actual, sn)

        actual = s1 + df1
        tm.assert_frame_equal(actual, df2)
        actual = s2 - df1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + s1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - s1
        tm.assert_frame_equal(actual, df1)

        actual = df1 + df1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - df1
        tm.assert_frame_equal(actual, df1)

        actual = df1 + scalar1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - scalar1
        tm.assert_frame_equal(actual, df1)

        actual = df1 + timedelta_NaT
        tm.assert_frame_equal(actual, dfn)
        actual = df1 - timedelta_NaT
        tm.assert_frame_equal(actual, dfn)

        msg = "cannot subtract a datelike from|unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            df1 + np.nan
        with pytest.raises(TypeError, match=msg):
            df1 - np.nan

        actual = df1 + NaT  # NaT is datetime, not timedelta
        tm.assert_frame_equal(actual, dfn)
        actual = df1 - NaT
        tm.assert_frame_equal(actual, dfn)

    # TODO: moved from tests.series.test_operators, needs splitting, cleanup,
    # de-duplication, box-parametrization...
    def test_operators_timedelta64(self):
        # series ops
        v1 = pd.date_range("2012-1-1", periods=3, freq="D")
        v2 = pd.date_range("2012-1-2", periods=3, freq="D")
        rs = Series(v2) - Series(v1)
        xp = Series(1e9 * 3600 * 24, rs.index).astype("int64").astype("timedelta64[ns]")
        tm.assert_series_equal(rs, xp)
        assert rs.dtype == "timedelta64[ns]"

        df = DataFrame({"A": v1})
        td = Series([timedelta(days=i) for i in range(3)])
        assert td.dtype == "timedelta64[ns]"

        # series on the rhs
        result = df["A"] - df["A"].shift()
        assert result.dtype == "timedelta64[ns]"

        result = df["A"] + td
        assert result.dtype == "M8[ns]"

        # scalar Timestamp on rhs
        maxa = df["A"].max()
        assert isinstance(maxa, Timestamp)

        resultb = df["A"] - df["A"].max()
        assert resultb.dtype == "timedelta64[ns]"

        # timestamp on lhs
        result = resultb + df["A"]
        values = [Timestamp("20111230"), Timestamp("20120101"), Timestamp("20120103")]
        expected = Series(values, dtype="M8[ns]", name="A")
        tm.assert_series_equal(result, expected)

        # datetimes on rhs
        result = df["A"] - datetime(2001, 1, 1)
        expected = Series([timedelta(days=4017 + i) for i in range(3)], name="A")
        tm.assert_series_equal(result, expected)
        assert result.dtype == "m8[ns]"

        d = datetime(2001, 1, 1, 3, 4)
        resulta = df["A"] - d
        assert resulta.dtype == "m8[ns]"

        # roundtrip
        resultb = resulta + d
        tm.assert_series_equal(df["A"], resultb)

        # timedeltas on rhs
        td = timedelta(days=1)
        resulta = df["A"] + td
        resultb = resulta - td
        tm.assert_series_equal(resultb, df["A"])
        assert resultb.dtype == "M8[ns]"

        # roundtrip
        td = timedelta(minutes=5, seconds=3)
        resulta = df["A"] + td
        resultb = resulta - td
        tm.assert_series_equal(df["A"], resultb)
        assert resultb.dtype == "M8[ns]"

        # inplace
        value = rs[2] + np.timedelta64(timedelta(minutes=5, seconds=1))
        rs[2] += np.timedelta64(timedelta(minutes=5, seconds=1))
        assert rs[2] == value

    def test_timedelta64_ops_nat(self):
        # GH 11349
        timedelta_series = Series([NaT, Timedelta("1s")])
        nat_series_dtype_timedelta = Series([NaT, NaT], dtype="timedelta64[ns]")
        single_nat_dtype_timedelta = Series([NaT], dtype="timedelta64[ns]")

        # subtraction
        tm.assert_series_equal(timedelta_series - NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(-NaT + timedelta_series, nat_series_dtype_timedelta)

        tm.assert_series_equal(
            timedelta_series - single_nat_dtype_timedelta, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            -single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta
        )

        # addition
        tm.assert_series_equal(
            nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + single_nat_dtype_timedelta,
            nat_series_dtype_timedelta,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timedelta,
            nat_series_dtype_timedelta,
        )

        tm.assert_series_equal(timedelta_series + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + timedelta_series, nat_series_dtype_timedelta)

        tm.assert_series_equal(
            timedelta_series + single_nat_dtype_timedelta, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + single_nat_dtype_timedelta,
            nat_series_dtype_timedelta,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timedelta,
            nat_series_dtype_timedelta,
        )

        # multiplication
        tm.assert_series_equal(
            nat_series_dtype_timedelta * 1.0, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            1.0 * nat_series_dtype_timedelta, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(timedelta_series * 1, timedelta_series)
        tm.assert_series_equal(1 * timedelta_series, timedelta_series)

        tm.assert_series_equal(timedelta_series * 1.5, Series([NaT, Timedelta("1.5s")]))
        tm.assert_series_equal(1.5 * timedelta_series, Series([NaT, Timedelta("1.5s")]))

        tm.assert_series_equal(timedelta_series * np.nan, nat_series_dtype_timedelta)
        tm.assert_series_equal(np.nan * timedelta_series, nat_series_dtype_timedelta)

        # division
        tm.assert_series_equal(timedelta_series / 2, Series([NaT, Timedelta("0.5s")]))
        tm.assert_series_equal(timedelta_series / 2.0, Series([NaT, Timedelta("0.5s")]))
        tm.assert_series_equal(timedelta_series / np.nan, nat_series_dtype_timedelta)

    # -------------------------------------------------------------
    # Binary operations td64 arraylike and datetime-like

    @pytest.mark.parametrize("cls", [Timestamp, datetime, np.datetime64])
    def test_td64arr_add_sub_datetimelike_scalar(
        self, cls, box_with_array, tz_naive_fixture
    ):
        # GH#11925, GH#29558, GH#23215
        tz = tz_naive_fixture

        dt_scalar = Timestamp("2012-01-01", tz=tz)
        if cls is datetime:
            ts = dt_scalar.to_pydatetime()
        elif cls is np.datetime64:
            if tz_naive_fixture is not None:
                pytest.skip(f"{cls} doesn support {tz_naive_fixture}")
            ts = dt_scalar.to_datetime64()
        else:
            ts = dt_scalar

        tdi = timedelta_range("1 day", periods=3)
        expected = pd.date_range("2012-01-02", periods=3, tz=tz)

        tdarr = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        tm.assert_equal(ts + tdarr, expected)
        tm.assert_equal(tdarr + ts, expected)

        expected2 = pd.date_range("2011-12-31", periods=3, freq="-1D", tz=tz)
        expected2 = tm.box_expected(expected2, box_with_array)

        tm.assert_equal(ts - tdarr, expected2)
        tm.assert_equal(ts + (-tdarr), expected2)

        msg = "cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            tdarr - ts

    def test_td64arr_add_datetime64_nat(self, box_with_array):
        # GH#23215
        other = np.datetime64("NaT")

        tdi = timedelta_range("1 day", periods=3)
        expected = DatetimeIndex(["NaT", "NaT", "NaT"], dtype="M8[ns]")

        tdser = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        tm.assert_equal(tdser + other, expected)
        tm.assert_equal(other + tdser, expected)

    def test_td64arr_sub_dt64_array(self, box_with_array):
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = TimedeltaIndex(["-1 Day"] * 3)
        dtarr = dti.values
        expected = DatetimeIndex(dtarr) - tdi

        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        msg = "cannot subtract a datelike from"
        with pytest.raises(TypeError, match=msg):
            tdi - dtarr

        # TimedeltaIndex.__rsub__
        result = dtarr - tdi
        tm.assert_equal(result, expected)

    def test_td64arr_add_dt64_array(self, box_with_array):
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = TimedeltaIndex(["-1 Day"] * 3)
        dtarr = dti.values
        expected = DatetimeIndex(dtarr) + tdi

        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdi + dtarr
        tm.assert_equal(result, expected)
        result = dtarr + tdi
        tm.assert_equal(result, expected)

    # ------------------------------------------------------------------
    # Invalid __add__/__sub__ operations

    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "h"])
    @pytest.mark.parametrize("tdi_freq", [None, "h"])
    def test_td64arr_sub_periodlike(
        self, box_with_array, box_with_array2, tdi_freq, pi_freq
    ):
        # GH#20049 subtracting PeriodIndex should raise TypeError
        tdi = TimedeltaIndex(["1 hours", "2 hours"], freq=tdi_freq)
        dti = Timestamp("2018-03-07 17:16:40") + tdi
        pi = dti.to_period(pi_freq)
        per = pi[0]

        tdi = tm.box_expected(tdi, box_with_array)
        pi = tm.box_expected(pi, box_with_array2)
        msg = "cannot subtract|unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            tdi - pi

        # GH#13078 subtraction of Period scalar not supported
        with pytest.raises(TypeError, match=msg):
            tdi - per

    @pytest.mark.parametrize(
        "other",
        [
            # GH#12624 for str case
            "a",
            # GH#19123
            1,
            1.5,
            np.array(2),
        ],
    )
    def test_td64arr_addsub_numeric_scalar_invalid(self, box_with_array, other):
        # vector-like others are tested in test_td64arr_add_sub_numeric_arr_invalid
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        tdarr = tm.box_expected(tdser, box_with_array)

        assert_invalid_addsub_type(tdarr, other)

    @pytest.mark.parametrize(
        "vec",
        [
            np.array([1, 2, 3]),
            Index([1, 2, 3]),
            Series([1, 2, 3]),
            DataFrame([[1, 2, 3]]),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_addsub_numeric_arr_invalid(
        self, box_with_array, vec, any_real_numpy_dtype
    ):
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        tdarr = tm.box_expected(tdser, box_with_array)

        vector = vec.astype(any_real_numpy_dtype)
        assert_invalid_addsub_type(tdarr, vector)

    def test_td64arr_add_sub_int(self, box_with_array, one):
        # Variants of `one` for #19012, deprecated GH#22535
        rng = timedelta_range("1 days 09:00:00", freq="h", periods=10)
        tdarr = tm.box_expected(rng, box_with_array)

        msg = "Addition/subtraction of integers"
        assert_invalid_addsub_type(tdarr, one, msg)

        # TODO: get inplace ops into assert_invalid_addsub_type
        with pytest.raises(TypeError, match=msg):
            tdarr += one
        with pytest.raises(TypeError, match=msg):
            tdarr -= one

    def test_td64arr_add_sub_integer_array(self, box_with_array):
        # GH#19959, deprecated GH#22535
        # GH#22696 for DataFrame case, check that we don't dispatch to numpy
        #  implementation, which treats int64 as m8[ns]
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = timedelta_range("1 days 09:00:00", freq="h", periods=3)
        tdarr = tm.box_expected(rng, box)
        other = tm.box_expected([4, 3, 2], xbox)

        msg = "Addition/subtraction of integers and integer-arrays"
        assert_invalid_addsub_type(tdarr, other, msg)

    def test_td64arr_addsub_integer_array_no_freq(self, box_with_array):
        # GH#19959
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        tdi = TimedeltaIndex(["1 Day", "NaT", "3 Hours"])
        tdarr = tm.box_expected(tdi, box)
        other = tm.box_expected([14, -1, 16], xbox)

        msg = "Addition/subtraction of integers"
        assert_invalid_addsub_type(tdarr, other, msg)

    # ------------------------------------------------------------------
    # Operations with timedelta-like others

    def test_td64arr_add_sub_td64_array(self, box_with_array):
        box = box_with_array
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = dti - dti.shift(1)
        tdarr = tdi.values

        expected = 2 * tdi
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)

        result = tdi + tdarr
        tm.assert_equal(result, expected)
        result = tdarr + tdi
        tm.assert_equal(result, expected)

        expected_sub = 0 * tdi
        result = tdi - tdarr
        tm.assert_equal(result, expected_sub)
        result = tdarr - tdi
        tm.assert_equal(result, expected_sub)

    def test_td64arr_add_sub_tdi(self, box_with_array, names):
        # GH#17250 make sure result dtype is correct
        # GH#19043 make sure names are propagated correctly
        box = box_with_array
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(["0 days", "1 day"], name=names[1])
        tdi = np.array(tdi) if box in [tm.to_array, pd.array] else tdi
        ser = Series([Timedelta(hours=3), Timedelta(hours=4)], name=names[0])
        expected = Series([Timedelta(hours=3), Timedelta(days=1, hours=4)], name=exname)

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = tdi + ser
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")

        result = ser + tdi
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")

        expected = Series(
            [Timedelta(hours=-3), Timedelta(days=1, hours=-4)], name=exname
        )
        expected = tm.box_expected(expected, box)

        result = tdi - ser
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")

        result = ser - tdi
        tm.assert_equal(result, -expected)
        assert_dtype(result, "timedelta64[ns]")

    @pytest.mark.parametrize("tdnat", [np.timedelta64("NaT"), NaT])
    def test_td64arr_add_sub_td64_nat(self, box_with_array, tdnat):
        # GH#18808, GH#23320 special handling for timedelta64("NaT")
        box = box_with_array
        tdi = TimedeltaIndex([NaT, Timedelta("1s")])
        expected = TimedeltaIndex(["NaT"] * 2)

        obj = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)

        result = obj + tdnat
        tm.assert_equal(result, expected)
        result = tdnat + obj
        tm.assert_equal(result, expected)
        result = obj - tdnat
        tm.assert_equal(result, expected)
        result = tdnat - obj
        tm.assert_equal(result, expected)

    def test_td64arr_add_timedeltalike(self, two_hours, box_with_array):
        # only test adding/sub offsets as + is now numeric
        # GH#10699 for Tick cases
        box = box_with_array
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("1 days 02:00:00", "10 days 02:00:00", freq="D")
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, box)

        result = rng + two_hours
        tm.assert_equal(result, expected)

        result = two_hours + rng
        tm.assert_equal(result, expected)

    def test_td64arr_sub_timedeltalike(self, two_hours, box_with_array):
        # only test adding/sub offsets as - is now numeric
        # GH#10699 for Tick cases
        box = box_with_array
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("0 days 22:00:00", "9 days 22:00:00")

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, box)

        result = rng - two_hours
        tm.assert_equal(result, expected)

        result = two_hours - rng
        tm.assert_equal(result, -expected)

    # ------------------------------------------------------------------
    # __add__/__sub__ with DateOffsets and arrays of DateOffsets

    def test_td64arr_add_sub_offset_index(self, names, box_with_array):
        # GH#18849, GH#19744
        box = box_with_array
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        other = Index([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
        other = np.array(other) if box in [tm.to_array, pd.array] else other

        expected = TimedeltaIndex(
            [tdi[n] + other[n] for n in range(len(tdi))], freq="infer", name=exname
        )
        expected_sub = TimedeltaIndex(
            [tdi[n] - other[n] for n in range(len(tdi))], freq="infer", name=exname
        )

        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box).astype(object, copy=False)
        expected_sub = tm.box_expected(expected_sub, box).astype(object, copy=False)

        with tm.assert_produces_warning(PerformanceWarning):
            res = tdi + other
        tm.assert_equal(res, expected)

        with tm.assert_produces_warning(PerformanceWarning):
            res2 = other + tdi
        tm.assert_equal(res2, expected)

        with tm.assert_produces_warning(PerformanceWarning):
            res_sub = tdi - other
        tm.assert_equal(res_sub, expected_sub)

    def test_td64arr_add_sub_offset_array(self, box_with_array):
        # GH#18849, GH#18824
        box = box_with_array
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        other = np.array([offsets.Hour(n=1), offsets.Minute(n=-2)])

        expected = TimedeltaIndex(
            [tdi[n] + other[n] for n in range(len(tdi))], freq="infer"
        )
        expected_sub = TimedeltaIndex(
            [tdi[n] - other[n] for n in range(len(tdi))], freq="infer"
        )

        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box).astype(object)

        with tm.assert_produces_warning(PerformanceWarning):
            res = tdi + other
        tm.assert_equal(res, expected)

        with tm.assert_produces_warning(PerformanceWarning):
            res2 = other + tdi
        tm.assert_equal(res2, expected)

        expected_sub = tm.box_expected(expected_sub, box_with_array).astype(object)
        with tm.assert_produces_warning(PerformanceWarning):
            res_sub = tdi - other
        tm.assert_equal(res_sub, expected_sub)

    def test_td64arr_with_offset_series(self, names, box_with_array):
        # GH#18849
        box = box_with_array
        box2 = Series if box in [Index, tm.to_array, pd.array] else box
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        other = Series([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])

        expected_add = Series(
            [tdi[n] + other[n] for n in range(len(tdi))], name=exname, dtype=object
        )
        obj = tm.box_expected(tdi, box)
        expected_add = tm.box_expected(expected_add, box2).astype(object)

        with tm.assert_produces_warning(PerformanceWarning):
            res = obj + other
        tm.assert_equal(res, expected_add)

        with tm.assert_produces_warning(PerformanceWarning):
            res2 = other + obj
        tm.assert_equal(res2, expected_add)

        expected_sub = Series(
            [tdi[n] - other[n] for n in range(len(tdi))], name=exname, dtype=object
        )
        expected_sub = tm.box_expected(expected_sub, box2).astype(object)

        with tm.assert_produces_warning(PerformanceWarning):
            res3 = obj - other
        tm.assert_equal(res3, expected_sub)

    @pytest.mark.parametrize("obox", [np.array, Index, Series])
    def test_td64arr_addsub_anchored_offset_arraylike(self, obox, box_with_array):
        # GH#18824
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        tdi = tm.box_expected(tdi, box_with_array)

        anchored = obox([offsets.MonthEnd(), offsets.Day(n=2)])

        # addition/subtraction ops with anchored offsets should issue
        # a PerformanceWarning and _then_ raise a TypeError.
        msg = "has incorrect type|cannot add the type MonthEnd"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                tdi + anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                anchored + tdi
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                tdi - anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                anchored - tdi

    # ------------------------------------------------------------------
    # Unsorted

    def test_td64arr_add_sub_object_array(self, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        tdi = timedelta_range("1 day", periods=3, freq="D")
        tdarr = tm.box_expected(tdi, box)

        other = np.array([Timedelta(days=1), offsets.Day(2), Timestamp("2000-01-04")])

        with tm.assert_produces_warning(PerformanceWarning):
            result = tdarr + other

        expected = Index(
            [Timedelta(days=2), Timedelta(days=4), Timestamp("2000-01-07")]
        )
        expected = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(result, expected)

        msg = "unsupported operand type|cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                tdarr - other

        with tm.assert_produces_warning(PerformanceWarning):
            result = other - tdarr

        expected = Index([Timedelta(0), Timedelta(0), Timestamp("2000-01-01")])
        expected = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(result, expected)


class TestTimedeltaArraylikeMulDivOps:
    # Tests for timedelta64[ns]
    # __mul__, __rmul__, __div__, __rdiv__, __floordiv__, __rfloordiv__

    # ------------------------------------------------------------------
    # Multiplication
    # organized with scalar others first, then array-like

    def test_td64arr_mul_int(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)

        result = idx * 1
        tm.assert_equal(result, idx)

        result = 1 * idx
        tm.assert_equal(result, idx)

    def test_td64arr_mul_tdlike_scalar_raises(self, two_hours, box_with_array):
        rng = timedelta_range("1 days", "10 days", name="foo")
        rng = tm.box_expected(rng, box_with_array)
        msg = "|".join(
            [
                "argument must be an integer",
                "cannot use operands with types dtype",
                "Cannot multiply with",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            rng * two_hours

    def test_tdi_mul_int_array_zerodim(self, box_with_array):
        rng5 = np.arange(5, dtype="int64")
        idx = TimedeltaIndex(rng5)
        expected = TimedeltaIndex(rng5 * 5)

        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = idx * np.array(5, dtype="int64")
        tm.assert_equal(result, expected)

    def test_tdi_mul_int_array(self, box_with_array):
        rng5 = np.arange(5, dtype="int64")
        idx = TimedeltaIndex(rng5)
        expected = TimedeltaIndex(rng5**2)

        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = idx * rng5
        tm.assert_equal(result, expected)

    def test_tdi_mul_int_series(self, box_with_array):
        box = box_with_array
        xbox = Series if box in [Index, tm.to_array, pd.array] else box

        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        expected = TimedeltaIndex(np.arange(5, dtype="int64") ** 2)

        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, xbox)

        result = idx * Series(np.arange(5, dtype="int64"))
        tm.assert_equal(result, expected)

    def test_tdi_mul_float_series(self, box_with_array):
        box = box_with_array
        xbox = Series if box in [Index, tm.to_array, pd.array] else box

        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box)

        rng5f = np.arange(5, dtype="float64")
        expected = TimedeltaIndex(rng5f * (rng5f + 1.0))
        expected = tm.box_expected(expected, xbox)

        result = idx * Series(rng5f + 1.0)
        tm.assert_equal(result, expected)

    # TODO: Put Series/DataFrame in others?
    @pytest.mark.parametrize(
        "other",
        [
            np.arange(1, 11),
            Index(np.arange(1, 11), np.int64),
            Index(range(1, 11), np.uint64),
            Index(range(1, 11), np.float64),
            pd.RangeIndex(1, 11),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_tdi_rmul_arraylike(self, other, box_with_array):
        box = box_with_array

        tdi = TimedeltaIndex(["1 Day"] * 10)
        expected = timedelta_range("1 days", "10 days")._with_freq(None)

        tdi = tm.box_expected(tdi, box)
        xbox = get_upcast_box(tdi, other)

        expected = tm.box_expected(expected, xbox)

        result = other * tdi
        tm.assert_equal(result, expected)
        commute = tdi * other
        tm.assert_equal(commute, expected)

    # ------------------------------------------------------------------
    # __div__, __rdiv__

    def test_td64arr_div_nat_invalid(self, box_with_array):
        # don't allow division by NaT (maybe could in the future)
        rng = timedelta_range("1 days", "10 days", name="foo")
        rng = tm.box_expected(rng, box_with_array)

        with pytest.raises(TypeError, match="unsupported operand type"):
            rng / NaT
        with pytest.raises(TypeError, match="Cannot divide NaTType by"):
            NaT / rng

        dt64nat = np.datetime64("NaT", "ns")
        msg = "|".join(
            [
                # 'divide' on npdev as of 2021-12-18
                "ufunc '(true_divide|divide)' cannot use operands",
                "cannot perform __r?truediv__",
                "Cannot divide datetime64 by TimedeltaArray",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            rng / dt64nat
        with pytest.raises(TypeError, match=msg):
            dt64nat / rng

    def test_td64arr_div_td64nat(self, box_with_array):
        # GH#23829
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = timedelta_range("1 days", "10 days")
        rng = tm.box_expected(rng, box)

        other = np.timedelta64("NaT")

        expected = np.array([np.nan] * 10)
        expected = tm.box_expected(expected, xbox)

        result = rng / other
        tm.assert_equal(result, expected)

        result = other / rng
        tm.assert_equal(result, expected)

    def test_td64arr_div_int(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)

        result = idx / 1
        tm.assert_equal(result, idx)

        with pytest.raises(TypeError, match="Cannot divide"):
            # GH#23829
            1 / idx

    def test_td64arr_div_tdlike_scalar(self, two_hours, box_with_array):
        # GH#20088, GH#22163 ensure DataFrame returns correct dtype
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = timedelta_range("1 days", "10 days", name="foo")
        expected = Index((np.arange(10) + 1) * 12, dtype=np.float64, name="foo")

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        result = rng / two_hours
        tm.assert_equal(result, expected)

        result = two_hours / rng
        expected = 1 / expected
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("m", [1, 3, 10])
    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_td64arr_div_td64_scalar(self, m, unit, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        ser = Series([Timedelta(days=59)] * 3)
        ser[2] = np.nan
        flat = ser
        ser = tm.box_expected(ser, box)

        # op
        expected = Series([x / np.timedelta64(m, unit) for x in flat])
        expected = tm.box_expected(expected, xbox)
        result = ser / np.timedelta64(m, unit)
        tm.assert_equal(result, expected)

        # reverse op
        expected = Series([Timedelta(np.timedelta64(m, unit)) / x for x in flat])
        expected = tm.box_expected(expected, xbox)
        result = np.timedelta64(m, unit) / ser
        tm.assert_equal(result, expected)

    def test_td64arr_div_tdlike_scalar_with_nat(self, two_hours, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        expected = Index([12, np.nan, 24], dtype=np.float64, name="foo")

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        result = rng / two_hours
        tm.assert_equal(result, expected)

        result = two_hours / rng
        expected = 1 / expected
        tm.assert_equal(result, expected)

    def test_td64arr_div_td64_ndarray(self, box_with_array):
        # GH#22631
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = TimedeltaIndex(["1 days", NaT, "2 days"])
        expected = Index([12, np.nan, 24], dtype=np.float64)

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        other = np.array([2, 4, 2], dtype="m8[h]")
        result = rng / other
        tm.assert_equal(result, expected)

        result = rng / tm.box_expected(other, box)
        tm.assert_equal(result, expected)

        result = rng / other.astype(object)
        tm.assert_equal(result, expected.astype(object))

        result = rng / list(other)
        tm.assert_equal(result, expected)

        # reversed op
        expected = 1 / expected
        result = other / rng
        tm.assert_equal(result, expected)

        result = tm.box_expected(other, box) / rng
        tm.assert_equal(result, expected)

        result = other.astype(object) / rng
        tm.assert_equal(result, expected)

        result = list(other) / rng
        tm.assert_equal(result, expected)

    def test_tdarr_div_length_mismatch(self, box_with_array):
        rng = TimedeltaIndex(["1 days", NaT, "2 days"])
        mismatched = [1, 2, 3, 4]

        rng = tm.box_expected(rng, box_with_array)
        msg = "Cannot divide vectors|Unable to coerce to Series"
        for obj in [mismatched, mismatched[:2]]:
            # one shorter, one longer
            for other in [obj, np.array(obj), Index(obj)]:
                with pytest.raises(ValueError, match=msg):
                    rng / other
                with pytest.raises(ValueError, match=msg):
                    other / rng

    def test_td64_div_object_mixed_result(self, box_with_array):
        # Case where we having a NaT in the result inseat of timedelta64("NaT")
        #  is misleading
        orig = timedelta_range("1 Day", periods=3).insert(1, NaT)
        tdi = tm.box_expected(orig, box_with_array, transpose=False)

        other = np.array([orig[0], 1.5, 2.0, orig[2]], dtype=object)
        other = tm.box_expected(other, box_with_array, transpose=False)

        res = tdi / other

        expected = Index([1.0, np.timedelta64("NaT", "ns"), orig[0], 1.5], dtype=object)
        expected = tm.box_expected(expected, box_with_array, transpose=False)
        if isinstance(expected, NumpyExtensionArray):
            expected = expected.to_numpy()
        tm.assert_equal(res, expected)
        if box_with_array is DataFrame:
            # We have a np.timedelta64(NaT), not pd.NaT
            assert isinstance(res.iloc[1, 0], np.timedelta64)

        res = tdi // other

        expected = Index([1, np.timedelta64("NaT", "ns"), orig[0], 1], dtype=object)
        expected = tm.box_expected(expected, box_with_array, transpose=False)
        if isinstance(expected, NumpyExtensionArray):
            expected = expected.to_numpy()
        tm.assert_equal(res, expected)
        if box_with_array is DataFrame:
            # We have a np.timedelta64(NaT), not pd.NaT
            assert isinstance(res.iloc[1, 0], np.timedelta64)

    # ------------------------------------------------------------------
    # __floordiv__, __rfloordiv__

    def test_td64arr_floordiv_td64arr_with_nat(
        self, box_with_array, using_array_manager
    ):
        # GH#35529
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        left = Series([1000, 222330, 30], dtype="timedelta64[ns]")
        right = Series([1000, 222330, None], dtype="timedelta64[ns]")

        left = tm.box_expected(left, box)
        right = tm.box_expected(right, box)

        expected = np.array([1.0, 1.0, np.nan], dtype=np.float64)
        expected = tm.box_expected(expected, xbox)
        if box is DataFrame and using_array_manager:
            # INFO(ArrayManager) floordiv returns integer, and ArrayManager
            # performs ops column-wise and thus preserves int64 dtype for
            # columns without missing values
            expected[[0, 1]] = expected[[0, 1]].astype("int64")

        with tm.maybe_produces_warning(
            RuntimeWarning, box is pd.array, check_stacklevel=False
        ):
            result = left // right

        tm.assert_equal(result, expected)

        # case that goes through __rfloordiv__ with arraylike
        with tm.maybe_produces_warning(
            RuntimeWarning, box is pd.array, check_stacklevel=False
        ):
            result = np.asarray(left) // right
        tm.assert_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_td64arr_floordiv_tdscalar(self, box_with_array, scalar_td):
        # GH#18831, GH#19125
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        td = Timedelta("5m3s")  # i.e. (scalar_td - 1sec) / 2

        td1 = Series([td, td, NaT], dtype="m8[ns]")
        td1 = tm.box_expected(td1, box, transpose=False)

        expected = Series([0, 0, np.nan])
        expected = tm.box_expected(expected, xbox, transpose=False)

        result = td1 // scalar_td
        tm.assert_equal(result, expected)

        # Reversed op
        expected = Series([2, 2, np.nan])
        expected = tm.box_expected(expected, xbox, transpose=False)

        result = scalar_td // td1
        tm.assert_equal(result, expected)

        # same thing buts let's be explicit about calling __rfloordiv__
        result = td1.__rfloordiv__(scalar_td)
        tm.assert_equal(result, expected)

    def test_td64arr_floordiv_int(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)
        result = idx // 1
        tm.assert_equal(result, idx)

        pattern = "floor_divide cannot use operands|Cannot divide int by Timedelta*"
        with pytest.raises(TypeError, match=pattern):
            1 // idx

    # ------------------------------------------------------------------
    # mod, divmod
    # TODO: operations with timedelta-like arrays, numeric arrays,
    #  reversed ops

    def test_td64arr_mod_tdscalar(self, box_with_array, three_days):
        tdi = timedelta_range("1 Day", "9 days")
        tdarr = tm.box_expected(tdi, box_with_array)

        expected = TimedeltaIndex(["1 Day", "2 Days", "0 Days"] * 3)
        expected = tm.box_expected(expected, box_with_array)

        result = tdarr % three_days
        tm.assert_equal(result, expected)

        warn = None
        if box_with_array is DataFrame and isinstance(three_days, pd.DateOffset):
            warn = PerformanceWarning
            # TODO: making expected be object here a result of DataFrame.__divmod__
            #  being defined in a naive way that does not dispatch to the underlying
            #  array's __divmod__
            expected = expected.astype(object)

        with tm.assert_produces_warning(warn):
            result = divmod(tdarr, three_days)

        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], tdarr // three_days)

    def test_td64arr_mod_int(self, box_with_array):
        tdi = timedelta_range("1 ns", "10 ns", periods=10)
        tdarr = tm.box_expected(tdi, box_with_array)

        expected = TimedeltaIndex(["1 ns", "0 ns"] * 5)
        expected = tm.box_expected(expected, box_with_array)

        result = tdarr % 2
        tm.assert_equal(result, expected)

        msg = "Cannot divide int by"
        with pytest.raises(TypeError, match=msg):
            2 % tdarr

        result = divmod(tdarr, 2)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], tdarr // 2)

    def test_td64arr_rmod_tdscalar(self, box_with_array, three_days):
        tdi = timedelta_range("1 Day", "9 days")
        tdarr = tm.box_expected(tdi, box_with_array)

        expected = ["0 Days", "1 Day", "0 Days"] + ["3 Days"] * 6
        expected = TimedeltaIndex(expected)
        expected = tm.box_expected(expected, box_with_array)

        result = three_days % tdarr
        tm.assert_equal(result, expected)

        result = divmod(three_days, tdarr)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], three_days // tdarr)

    # ------------------------------------------------------------------
    # Operations with invalid others

    def test_td64arr_mul_tdscalar_invalid(self, box_with_array, scalar_td):
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        td1 = tm.box_expected(td1, box_with_array)

        # check that we are getting a TypeError
        # with 'operate' (from core/ops.py) for the ops that are not
        # defined
        pattern = "operate|unsupported|cannot|not supported"
        with pytest.raises(TypeError, match=pattern):
            td1 * scalar_td
        with pytest.raises(TypeError, match=pattern):
            scalar_td * td1

    def test_td64arr_mul_too_short_raises(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)
        msg = "|".join(
            [
                "cannot use operands with types dtype",
                "Cannot multiply with unequal lengths",
                "Unable to coerce to Series",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            # length check before dtype check
            idx * idx[:3]
        with pytest.raises(ValueError, match=msg):
            idx * np.array([1, 2])

    def test_td64arr_mul_td64arr_raises(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)
        msg = "cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            idx * idx

    # ------------------------------------------------------------------
    # Operations with numeric others

    def test_td64arr_mul_numeric_scalar(self, box_with_array, one):
        # GH#4521
        # divide/multiply by integers
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        expected = Series(["-59 Days", "-59 Days", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdser * (-one)
        tm.assert_equal(result, expected)
        result = (-one) * tdser
        tm.assert_equal(result, expected)

        expected = Series(["118 Days", "118 Days", "NaT"], dtype="timedelta64[ns]")
        expected = tm.box_expected(expected, box_with_array)

        result = tdser * (2 * one)
        tm.assert_equal(result, expected)
        result = (2 * one) * tdser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("two", [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_div_numeric_scalar(self, box_with_array, two):
        # GH#4521
        # divide/multiply by integers
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        expected = Series(["29.5D", "29.5D", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdser / two
        tm.assert_equal(result, expected)

        with pytest.raises(TypeError, match="Cannot divide"):
            two / tdser

    @pytest.mark.parametrize("two", [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_floordiv_numeric_scalar(self, box_with_array, two):
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        expected = Series(["29.5D", "29.5D", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdser // two
        tm.assert_equal(result, expected)

        with pytest.raises(TypeError, match="Cannot divide"):
            two // tdser

    @pytest.mark.parametrize(
        "vector",
        [np.array([20, 30, 40]), Index([20, 30, 40]), Series([20, 30, 40])],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_rmul_numeric_array(
        self,
        box_with_array,
        vector,
        any_real_numpy_dtype,
    ):
        # GH#4521
        # divide/multiply by integers

        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        vector = vector.astype(any_real_numpy_dtype)

        expected = Series(["1180 Days", "1770 Days", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        xbox = get_upcast_box(tdser, vector)

        expected = tm.box_expected(expected, xbox)

        result = tdser * vector
        tm.assert_equal(result, expected)

        result = vector * tdser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "vector",
        [np.array([20, 30, 40]), Index([20, 30, 40]), Series([20, 30, 40])],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_div_numeric_array(
        self, box_with_array, vector, any_real_numpy_dtype
    ):
        # GH#4521
        # divide/multiply by integers

        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        vector = vector.astype(any_real_numpy_dtype)

        expected = Series(["2.95D", "1D 23h 12m", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        xbox = get_upcast_box(tdser, vector)
        expected = tm.box_expected(expected, xbox)

        result = tdser / vector
        tm.assert_equal(result, expected)

        pattern = "|".join(
            [
                "true_divide'? cannot use operands",
                "cannot perform __div__",
                "cannot perform __truediv__",
                "unsupported operand",
                "Cannot divide",
                "ufunc 'divide' cannot use operands with types",
            ]
        )
        with pytest.raises(TypeError, match=pattern):
            vector / tdser

        result = tdser / vector.astype(object)
        if box_with_array is DataFrame:
            expected = [tdser.iloc[0, n] / vector[n] for n in range(len(vector))]
            expected = tm.box_expected(expected, xbox).astype(object)
            # We specifically expect timedelta64("NaT") here, not pd.NA
            msg = "The 'downcast' keyword in fillna"
            with tm.assert_produces_warning(FutureWarning, match=msg):
                expected[2] = expected[2].fillna(
                    np.timedelta64("NaT", "ns"), downcast=False
                )
        else:
            expected = [tdser[n] / vector[n] for n in range(len(tdser))]
            expected = [
                x if x is not NaT else np.timedelta64("NaT", "ns") for x in expected
            ]
            if xbox is tm.to_array:
                expected = tm.to_array(expected).astype(object)
            else:
                expected = xbox(expected, dtype=object)

        tm.assert_equal(result, expected)

        with pytest.raises(TypeError, match=pattern):
            vector.astype(object) / tdser

    def test_td64arr_mul_int_series(self, box_with_array, names):
        # GH#19042 test for correct name attachment
        box = box_with_array
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(
            ["0days", "1day", "2days", "3days", "4days"], name=names[0]
        )
        # TODO: Should we be parametrizing over types for `ser` too?
        ser = Series([0, 1, 2, 3, 4], dtype=np.int64, name=names[1])

        expected = Series(
            ["0days", "1day", "4days", "9days", "16days"],
            dtype="timedelta64[ns]",
            name=exname,
        )

        tdi = tm.box_expected(tdi, box)
        xbox = get_upcast_box(tdi, ser)

        expected = tm.box_expected(expected, xbox)

        result = ser * tdi
        tm.assert_equal(result, expected)

        result = tdi * ser
        tm.assert_equal(result, expected)

    # TODO: Should we be parametrizing over types for `ser` too?
    def test_float_series_rdiv_td64arr(self, box_with_array, names):
        # GH#19042 test for correct name attachment
        box = box_with_array
        tdi = TimedeltaIndex(
            ["0days", "1day", "2days", "3days", "4days"], name=names[0]
        )
        ser = Series([1.5, 3, 4.5, 6, 7.5], dtype=np.float64, name=names[1])

        xname = names[2] if box not in [tm.to_array, pd.array] else names[1]
        expected = Series(
            [tdi[n] / ser[n] for n in range(len(ser))],
            dtype="timedelta64[ns]",
            name=xname,
        )

        tdi = tm.box_expected(tdi, box)
        xbox = get_upcast_box(tdi, ser)
        expected = tm.box_expected(expected, xbox)

        result = ser.__rtruediv__(tdi)
        if box is DataFrame:
            assert result is NotImplemented
        else:
            tm.assert_equal(result, expected)

    def test_td64arr_all_nat_div_object_dtype_numeric(self, box_with_array):
        # GH#39750 make sure we infer the result as td64
        tdi = TimedeltaIndex([NaT, NaT])

        left = tm.box_expected(tdi, box_with_array)
        right = np.array([2, 2.0], dtype=object)

        tdnat = np.timedelta64("NaT", "ns")
        expected = Index([tdnat] * 2, dtype=object)
        if box_with_array is not Index:
            expected = tm.box_expected(expected, box_with_array).astype(object)
            if box_with_array in [Series, DataFrame]:
                msg = "The 'downcast' keyword in fillna is deprecated"
                with tm.assert_produces_warning(FutureWarning, match=msg):
                    expected = expected.fillna(tdnat, downcast=False)  # GH#18463

        result = left / right
        tm.assert_equal(result, expected)

        result = left // right
        tm.assert_equal(result, expected)


class TestTimedelta64ArrayLikeArithmetic:
    # Arithmetic tests for timedelta64[ns] vectors fully parametrized over
    #  DataFrame/Series/TimedeltaIndex/TimedeltaArray.  Ideally all arithmetic
    #  tests will eventually end up here.

    def test_td64arr_pow_invalid(self, scalar_td, box_with_array):
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        td1 = tm.box_expected(td1, box_with_array)

        # check that we are getting a TypeError
        # with 'operate' (from core/ops.py) for the ops that are not
        # defined
        pattern = "operate|unsupported|cannot|not supported"
        with pytest.raises(TypeError, match=pattern):
            scalar_td**td1

        with pytest.raises(TypeError, match=pattern):
            td1**scalar_td


def test_add_timestamp_to_timedelta():
    # GH: 35897
    timestamp = Timestamp("2021-01-01")
    result = timestamp + timedelta_range("0s", "1s", periods=31)
    expected = DatetimeIndex(
        [
            timestamp
            + (
                pd.to_timedelta("0.033333333s") * i
                + pd.to_timedelta("0.000000001s") * divmod(i, 3)[0]
            )
            for i in range(31)
        ]
    )
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_categorical.py ===
from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    qcut,
)
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args


def cartesian_product_for_groupers(result, args, names, fill_value=np.nan):
    """Reindex to a cartesian production for the groupers,
    preserving the nature (Categorical) of each grouper
    """

    def f(a):
        if isinstance(a, (CategoricalIndex, Categorical)):
            categories = a.categories
            a = Categorical.from_codes(
                np.arange(len(categories)), categories=categories, ordered=a.ordered
            )
        return a

    index = MultiIndex.from_product(map(f, args), names=names)
    return result.reindex(index, fill_value=fill_value).sort_index()


_results_for_groupbys_with_missing_categories = {
    # This maps the builtin groupby functions to their expected outputs for
    # missing categories when they are called on a categorical grouper with
    # observed=False. Some functions are expected to return NaN, some zero.
    # These expected values can be used across several tests (i.e. they are
    # the same for SeriesGroupBy and DataFrameGroupBy) but they should only be
    # hardcoded in one place.
    "all": np.nan,
    "any": np.nan,
    "count": 0,
    "corrwith": np.nan,
    "first": np.nan,
    "idxmax": np.nan,
    "idxmin": np.nan,
    "last": np.nan,
    "max": np.nan,
    "mean": np.nan,
    "median": np.nan,
    "min": np.nan,
    "nth": np.nan,
    "nunique": 0,
    "prod": np.nan,
    "quantile": np.nan,
    "sem": np.nan,
    "size": 0,
    "skew": np.nan,
    "std": np.nan,
    "sum": 0,
    "var": np.nan,
}


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_apply_use_categorical_name(df):
    cats = qcut(df.C, 4)

    def get_stats(group):
        return {
            "min": group.min(),
            "max": group.max(),
            "count": group.count(),
            "mean": group.mean(),
        }

    result = df.groupby(cats, observed=False).D.apply(get_stats)
    assert result.index.names[0] == "C"


def test_basic(using_infer_string):  # TODO: split this test
    cats = Categorical(
        ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    data = DataFrame({"a": [1, 1, 1, 2, 2, 2, 3, 4, 5], "b": cats})

    exp_index = CategoricalIndex(list("abcd"), name="b", ordered=True)
    expected = DataFrame({"a": [1, 2, 4, np.nan]}, index=exp_index)
    result = data.groupby("b", observed=False).mean()
    tm.assert_frame_equal(result, expected)

    cat1 = Categorical(["a", "a", "b", "b"], categories=["a", "b", "z"], ordered=True)
    cat2 = Categorical(["c", "d", "c", "d"], categories=["c", "d", "y"], ordered=True)
    df = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})

    # single grouper
    gb = df.groupby("A", observed=False)
    exp_idx = CategoricalIndex(["a", "b", "z"], name="A", ordered=True)
    expected = DataFrame({"values": Series([3, 7, 0], index=exp_idx)})
    result = gb.sum(numeric_only=True)
    tm.assert_frame_equal(result, expected)

    # GH 8623
    x = DataFrame(
        [[1, "John P. Doe"], [2, "Jane Dove"], [1, "John P. Doe"]],
        columns=["person_id", "person_name"],
    )
    x["person_name"] = Categorical(x.person_name)

    g = x.groupby(["person_id"], observed=False)
    result = g.transform(lambda x: x)
    tm.assert_frame_equal(result, x[["person_name"]])

    result = x.drop_duplicates("person_name")
    expected = x.iloc[[0, 1]]
    tm.assert_frame_equal(result, expected)

    def f(x):
        return x.drop_duplicates("person_name").iloc[0]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = g.apply(f)
    expected = x.iloc[[0, 1]].copy()
    expected.index = Index([1, 2], name="person_id")
    dtype = "str" if using_infer_string else object
    expected["person_name"] = expected["person_name"].astype(dtype)
    tm.assert_frame_equal(result, expected)

    # GH 9921
    # Monotonic
    df = DataFrame({"a": [5, 15, 25]})
    c = pd.cut(df.a, bins=[0, 10, 20, 30, 40])

    msg = "using SeriesGroupBy.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df["a"])

    tm.assert_series_equal(
        df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df["a"]
    )
    msg = "using DataFrameGroupBy.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result = df.groupby(c, observed=False).transform(sum)
    expected = df[["a"]]
    tm.assert_frame_equal(result, expected)

    gbc = df.groupby(c, observed=False)
    result = gbc.transform(lambda xs: np.max(xs, axis=0))
    tm.assert_frame_equal(result, df[["a"]])

    result2 = gbc.transform(lambda xs: np.max(xs, axis=0))
    msg = "using DataFrameGroupBy.max"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result3 = gbc.transform(max)
    result4 = gbc.transform(np.maximum.reduce)
    result5 = gbc.transform(lambda xs: np.maximum.reduce(xs))
    tm.assert_frame_equal(result2, df[["a"]], check_dtype=False)
    tm.assert_frame_equal(result3, df[["a"]], check_dtype=False)
    tm.assert_frame_equal(result4, df[["a"]])
    tm.assert_frame_equal(result5, df[["a"]])

    # Filter
    tm.assert_series_equal(df.a.groupby(c, observed=False).filter(np.all), df["a"])
    tm.assert_frame_equal(df.groupby(c, observed=False).filter(np.all), df)

    # Non-monotonic
    df = DataFrame({"a": [5, 15, 25, -5]})
    c = pd.cut(df.a, bins=[-10, 0, 10, 20, 30, 40])

    msg = "using SeriesGroupBy.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df["a"])

    tm.assert_series_equal(
        df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df["a"]
    )
    msg = "using DataFrameGroupBy.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result = df.groupby(c, observed=False).transform(sum)
    expected = df[["a"]]
    tm.assert_frame_equal(result, expected)

    tm.assert_frame_equal(
        df.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df[["a"]]
    )

    # GH 9603
    df = DataFrame({"a": [1, 0, 0, 0]})
    c = pd.cut(df.a, [0, 1, 2, 3, 4], labels=Categorical(list("abcd")))
    result = df.groupby(c, observed=False).apply(len)

    exp_index = CategoricalIndex(c.values.categories, ordered=c.values.ordered)
    expected = Series([1, 0, 0, 0], index=exp_index)
    expected.index.name = "a"
    tm.assert_series_equal(result, expected)

    # more basic
    levels = ["foo", "bar", "baz", "qux"]
    codes = np.random.default_rng(2).integers(0, 4, size=100)

    cats = Categorical.from_codes(codes, levels, ordered=True)

    data = DataFrame(np.random.default_rng(2).standard_normal((100, 4)))

    result = data.groupby(cats, observed=False).mean()

    expected = data.groupby(np.asarray(cats), observed=False).mean()
    exp_idx = CategoricalIndex(levels, categories=cats.categories, ordered=True)
    expected = expected.reindex(exp_idx)

    tm.assert_frame_equal(result, expected)

    grouped = data.groupby(cats, observed=False)
    desc_result = grouped.describe()

    idx = cats.codes.argsort()
    ord_labels = np.asarray(cats).take(idx)
    ord_data = data.take(idx)

    exp_cats = Categorical(
        ord_labels, ordered=True, categories=["foo", "bar", "baz", "qux"]
    )
    expected = ord_data.groupby(exp_cats, sort=False, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)

    # GH 10460
    expc = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    exp = CategoricalIndex(expc)
    tm.assert_index_equal(
        (desc_result.stack(future_stack=True).index.get_level_values(0)), exp
    )
    exp = Index(["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4)
    tm.assert_index_equal(
        (desc_result.stack(future_stack=True).index.get_level_values(1)), exp
    )


def test_level_get_group(observed):
    # GH15155
    df = DataFrame(
        data=np.arange(2, 22, 2),
        index=MultiIndex(
            levels=[CategoricalIndex(["a", "b"]), range(10)],
            codes=[[0] * 5 + [1] * 5, range(10)],
            names=["Index1", "Index2"],
        ),
    )
    g = df.groupby(level=["Index1"], observed=observed)

    # expected should equal test.loc[["a"]]
    # GH15166
    expected = DataFrame(
        data=np.arange(2, 12, 2),
        index=MultiIndex(
            levels=[CategoricalIndex(["a", "b"]), range(5)],
            codes=[[0] * 5, range(5)],
            names=["Index1", "Index2"],
        ),
    )
    msg = "you will need to pass a length-1 tuple"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#25971 - warn when not passing a length-1 tuple
        result = g.get_group("a")

    tm.assert_frame_equal(result, expected)


def test_sorting_with_different_categoricals():
    # GH 24271
    df = DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "dose": ["high", "med", "low"] * 4,
            "outcomes": np.arange(12.0),
        }
    )

    df.dose = Categorical(df.dose, categories=["low", "med", "high"], ordered=True)

    result = df.groupby("group")["dose"].value_counts()
    result = result.sort_index(level=0, sort_remaining=True)
    index = ["low", "med", "high", "low", "med", "high"]
    index = Categorical(index, categories=["low", "med", "high"], ordered=True)
    index = [["A", "A", "A", "B", "B", "B"], CategoricalIndex(index)]
    index = MultiIndex.from_arrays(index, names=["group", "dose"])
    expected = Series([2] * 6, index=index, name="count")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_apply(ordered):
    # GH 10138

    dense = Categorical(list("abc"), ordered=ordered)

    # 'b' is in the categories but not in the list
    missing = Categorical(list("aaa"), categories=["a", "b"], ordered=ordered)
    values = np.arange(len(dense))
    df = DataFrame({"missing": missing, "dense": dense, "values": values})
    grouped = df.groupby(["missing", "dense"], observed=True)

    # missing category 'b' should still exist in the output index
    idx = MultiIndex.from_arrays([missing, dense], names=["missing", "dense"])
    expected = DataFrame([0, 1, 2.0], index=idx, columns=["values"])

    result = grouped.apply(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)

    result = grouped.mean()
    tm.assert_frame_equal(result, expected)

    msg = "using DataFrameGroupBy.mean"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result = grouped.agg(np.mean)
    tm.assert_frame_equal(result, expected)

    # but for transform we should still get back the original index
    idx = MultiIndex.from_arrays([missing, dense], names=["missing", "dense"])
    expected = Series(1, index=idx)
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = grouped.apply(lambda x: 1)
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_observed(request, using_infer_string, observed):
    # multiple groupers, don't re-expand the output space
    # of the grouper
    # gh-14942 (implement)
    # gh-10132 (back-compat)
    # gh-8138 (back-compat)
    # gh-8869

    if using_infer_string and not observed:
        # TODO(infer_string) this fails with filling the string column with 0
        request.applymarker(pytest.mark.xfail(reason="TODO(infer_string)"))

    cat1 = Categorical(["a", "a", "b", "b"], categories=["a", "b", "z"], ordered=True)
    cat2 = Categorical(["c", "d", "c", "d"], categories=["c", "d", "y"], ordered=True)
    df = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})
    df["C"] = ["foo", "bar"] * 2

    # multiple groupers with a non-cat
    gb = df.groupby(["A", "B", "C"], observed=observed)
    exp_index = MultiIndex.from_arrays(
        [cat1, cat2, ["foo", "bar"] * 2], names=["A", "B", "C"]
    )
    expected = DataFrame({"values": Series([1, 2, 3, 4], index=exp_index)}).sort_index()
    result = gb.sum()
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [cat1, cat2, ["foo", "bar"]], list("ABC"), fill_value=0
        )

    tm.assert_frame_equal(result, expected)

    gb = df.groupby(["A", "B"], observed=observed)
    exp_index = MultiIndex.from_arrays([cat1, cat2], names=["A", "B"])
    expected = DataFrame(
        {"values": [1, 2, 3, 4], "C": ["foo", "bar", "foo", "bar"]}, index=exp_index
    )
    result = gb.sum()
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [cat1, cat2], list("AB"), fill_value=0
        )

    tm.assert_frame_equal(result, expected)

    # https://github.com/pandas-dev/pandas/issues/8138
    d = {
        "cat": Categorical(
            ["a", "b", "a", "b"], categories=["a", "b", "c"], ordered=True
        ),
        "ints": [1, 1, 2, 2],
        "val": [10, 20, 30, 40],
    }
    df = DataFrame(d)

    # Grouping on a single column
    groups_single_key = df.groupby("cat", observed=observed)
    result = groups_single_key.mean()

    exp_index = CategoricalIndex(
        list("ab"), name="cat", categories=list("abc"), ordered=True
    )
    expected = DataFrame({"ints": [1.5, 1.5], "val": [20.0, 30]}, index=exp_index)
    if not observed:
        index = CategoricalIndex(
            list("abc"), name="cat", categories=list("abc"), ordered=True
        )
        expected = expected.reindex(index)

    tm.assert_frame_equal(result, expected)

    # Grouping on two columns
    groups_double_key = df.groupby(["cat", "ints"], observed=observed)
    result = groups_double_key.agg("mean")
    expected = DataFrame(
        {
            "val": [10.0, 30.0, 20.0, 40.0],
            "cat": Categorical(
                ["a", "a", "b", "b"], categories=["a", "b", "c"], ordered=True
            ),
            "ints": [1, 2, 1, 2],
        }
    ).set_index(["cat", "ints"])
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [df.cat.values, [1, 2]], ["cat", "ints"]
        )

    tm.assert_frame_equal(result, expected)

    # GH 10132
    for key in [("a", 1), ("b", 2), ("b", 1), ("a", 2)]:
        c, i = key
        result = groups_double_key.get_group(key)
        expected = df[(df.cat == c) & (df.ints == i)]
        tm.assert_frame_equal(result, expected)

    # gh-8869
    # with as_index
    d = {
        "foo": [10, 8, 4, 8, 4, 1, 1],
        "bar": [10, 20, 30, 40, 50, 60, 70],
        "baz": ["d", "c", "e", "a", "a", "d", "c"],
    }
    df = DataFrame(d)
    cat = pd.cut(df["foo"], np.linspace(0, 10, 3))
    df["range"] = cat
    groups = df.groupby(["range", "baz"], as_index=False, observed=observed)
    result = groups.agg("mean")

    groups2 = df.groupby(["range", "baz"], as_index=True, observed=observed)
    expected = groups2.agg("mean").reset_index()
    tm.assert_frame_equal(result, expected)


def test_observed_codes_remap(observed):
    d = {"C1": [3, 3, 4, 5], "C2": [1, 2, 3, 4], "C3": [10, 100, 200, 34]}
    df = DataFrame(d)
    values = pd.cut(df["C1"], [1, 2, 3, 6])
    values.name = "cat"
    groups_double_key = df.groupby([values, "C2"], observed=observed)

    idx = MultiIndex.from_arrays([values, [1, 2, 3, 4]], names=["cat", "C2"])
    expected = DataFrame(
        {"C1": [3.0, 3.0, 4.0, 5.0], "C3": [10.0, 100.0, 200.0, 34.0]}, index=idx
    )
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [values.values, [1, 2, 3, 4]], ["cat", "C2"]
        )

    result = groups_double_key.agg("mean")
    tm.assert_frame_equal(result, expected)


def test_observed_perf():
    # we create a cartesian product, so this is
    # non-performant if we don't use observed values
    # gh-14942
    df = DataFrame(
        {
            "cat": np.random.default_rng(2).integers(0, 255, size=30000),
            "int_id": np.random.default_rng(2).integers(0, 255, size=30000),
            "other_id": np.random.default_rng(2).integers(0, 10000, size=30000),
            "foo": 0,
        }
    )
    df["cat"] = df.cat.astype(str).astype("category")

    grouped = df.groupby(["cat", "int_id", "other_id"], observed=True)
    result = grouped.count()
    assert result.index.levels[0].nunique() == df.cat.nunique()
    assert result.index.levels[1].nunique() == df.int_id.nunique()
    assert result.index.levels[2].nunique() == df.other_id.nunique()


def test_observed_groups(observed):
    # gh-20583
    # test that we have the appropriate groups

    cat = Categorical(["a", "c", "a"], categories=["a", "b", "c"])
    df = DataFrame({"cat": cat, "vals": [1, 2, 3]})
    g = df.groupby("cat", observed=observed)

    result = g.groups
    if observed:
        expected = {"a": Index([0, 2], dtype="int64"), "c": Index([1], dtype="int64")}
    else:
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "c": Index([1], dtype="int64"),
        }

    tm.assert_dict_equal(result, expected)


@pytest.mark.parametrize(
    "keys, expected_values, expected_index_levels",
    [
        ("a", [15, 9, 0], CategoricalIndex([1, 2, 3], name="a")),
        (
            ["a", "b"],
            [7, 8, 0, 0, 0, 9, 0, 0, 0],
            [CategoricalIndex([1, 2, 3], name="a"), Index([4, 5, 6])],
        ),
        (
            ["a", "a2"],
            [15, 0, 0, 0, 9, 0, 0, 0, 0],
            [
                CategoricalIndex([1, 2, 3], name="a"),
                CategoricalIndex([1, 2, 3], name="a"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("test_series", [True, False])
def test_unobserved_in_index(keys, expected_values, expected_index_levels, test_series):
    # GH#49354 - ensure unobserved cats occur when grouping by index levels
    df = DataFrame(
        {
            "a": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "a2": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    ).set_index(["a", "a2"])
    if "b" not in keys:
        # Only keep b when it is used for grouping for consistent columns in the result
        df = df.drop(columns="b")

    gb = df.groupby(keys, observed=False)
    if test_series:
        gb = gb["c"]
    result = gb.sum()

    if len(keys) == 1:
        index = expected_index_levels
    else:
        codes = [[0, 0, 0, 1, 1, 1, 2, 2, 2], 3 * [0, 1, 2]]
        index = MultiIndex(
            expected_index_levels,
            codes=codes,
            names=keys,
        )
    expected = DataFrame({"c": expected_values}, index=index)
    if test_series:
        expected = expected["c"]
    tm.assert_equal(result, expected)


def test_observed_groups_with_nan(observed):
    # GH 24740
    df = DataFrame(
        {
            "cat": Categorical(["a", np.nan, "a"], categories=["a", "b", "d"]),
            "vals": [1, 2, 3],
        }
    )
    g = df.groupby("cat", observed=observed)
    result = g.groups
    if observed:
        expected = {"a": Index([0, 2], dtype="int64")}
    else:
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "d": Index([], dtype="int64"),
        }
    tm.assert_dict_equal(result, expected)


def test_observed_nth():
    # GH 26385
    cat = Categorical(["a", np.nan, np.nan], categories=["a", "b", "c"])
    ser = Series([1, 2, 3])
    df = DataFrame({"cat": cat, "ser": ser})

    result = df.groupby("cat", observed=False)["ser"].nth(0)
    expected = df["ser"].iloc[[0]]
    tm.assert_series_equal(result, expected)


def test_dataframe_categorical_with_nan(observed):
    # GH 21151
    s1 = Categorical([np.nan, "a", np.nan, "a"], categories=["a", "b", "c"])
    s2 = Series([1, 2, 3, 4])
    df = DataFrame({"s1": s1, "s2": s2})
    result = df.groupby("s1", observed=observed).first().reset_index()
    if observed:
        expected = DataFrame(
            {"s1": Categorical(["a"], categories=["a", "b", "c"]), "s2": [2]}
        )
    else:
        expected = DataFrame(
            {
                "s1": Categorical(["a", "b", "c"], categories=["a", "b", "c"]),
                "s2": [2, np.nan, np.nan],
            }
        )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("observed", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def test_dataframe_categorical_ordered_observed_sort(ordered, observed, sort):
    # GH 25871: Fix groupby sorting on ordered Categoricals
    # GH 25167: Groupby with observed=True doesn't sort

    # Build a dataframe with cat having one unobserved category ('missing'),
    # and a Series with identical values
    label = Categorical(
        ["d", "a", "b", "a", "d", "b"],
        categories=["a", "b", "missing", "d"],
        ordered=ordered,
    )
    val = Series(["d", "a", "b", "a", "d", "b"])
    df = DataFrame({"label": label, "val": val})

    # aggregate on the Categorical
    result = df.groupby("label", observed=observed, sort=sort)["val"].aggregate("first")

    # If ordering works, we expect index labels equal to aggregation results,
    # except for 'observed=False': label 'missing' has aggregation None
    label = Series(result.index.array, dtype="object")
    aggr = Series(result.array)
    if not observed:
        aggr[aggr.isna()] = "missing"
    if not all(label == aggr):
        msg = (
            "Labels and aggregation results not consistently sorted\n"
            f"for (ordered={ordered}, observed={observed}, sort={sort})\n"
            f"Result:\n{result}"
        )
        assert False, msg


def test_datetime():
    # GH9049: ensure backward compatibility
    levels = pd.date_range("2014-01-01", periods=4)
    codes = np.random.default_rng(2).integers(0, 4, size=100)

    cats = Categorical.from_codes(codes, levels, ordered=True)

    data = DataFrame(np.random.default_rng(2).standard_normal((100, 4)))
    result = data.groupby(cats, observed=False).mean()

    expected = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )

    tm.assert_frame_equal(result, expected)

    grouped = data.groupby(cats, observed=False)
    desc_result = grouped.describe()

    idx = cats.codes.argsort()
    ord_labels = cats.take(idx)
    ord_data = data.take(idx)
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    tm.assert_frame_equal(desc_result, expected)
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )

    # GH 10460
    expc = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    exp = CategoricalIndex(expc)
    tm.assert_index_equal(
        (desc_result.stack(future_stack=True).index.get_level_values(0)), exp
    )
    exp = Index(["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4)
    tm.assert_index_equal(
        (desc_result.stack(future_stack=True).index.get_level_values(1)), exp
    )


def test_categorical_index():
    s = np.random.default_rng(2)
    levels = ["foo", "bar", "baz", "qux"]
    codes = s.integers(0, 4, size=20)
    cats = Categorical.from_codes(codes, levels, ordered=True)
    df = DataFrame(np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd"))
    df["cats"] = cats

    # with a cat index
    result = df.set_index("cats").groupby(level=0, observed=False).sum()
    expected = df[list("abcd")].groupby(cats.codes, observed=False).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True), name="cats"
    )
    tm.assert_frame_equal(result, expected)

    # with a cat column, should produce a cat index
    result = df.groupby("cats", observed=False).sum()
    expected = df[list("abcd")].groupby(cats.codes, observed=False).sum()
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True), name="cats"
    )
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns():
    # GH 11558
    cats = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 4)), columns=cats)
    result = df.groupby([1, 2, 3, 4] * 5).describe()

    tm.assert_index_equal(result.stack(future_stack=True).columns, cats)
    tm.assert_categorical_equal(
        result.stack(future_stack=True).columns.values, cats.values
    )


def test_unstack_categorical():
    # GH11558 (example is taken from the original issue)
    df = DataFrame(
        {"a": range(10), "medium": ["A", "B"] * 5, "artist": list("XYXXY") * 2}
    )
    df["medium"] = df["medium"].astype("category")

    gcat = df.groupby(["artist", "medium"], observed=False)["a"].count().unstack()
    result = gcat.describe()

    exp_columns = CategoricalIndex(["A", "B"], ordered=False, name="medium")
    tm.assert_index_equal(result.columns, exp_columns)
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)

    result = gcat["A"] + gcat["B"]
    expected = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    tm.assert_series_equal(result, expected)


def test_bins_unequal_len():
    # GH3011
    series = Series([np.nan, np.nan, 1, 1, 2, 2, 3, 3, 4, 4])
    bins = pd.cut(series.dropna().values, 4)

    # len(bins) != len(series) here
    with pytest.raises(ValueError, match="Grouper and axis must be same length"):
        series.groupby(bins).mean()


@pytest.mark.parametrize(
    ["series", "data"],
    [
        # Group a series with length and index equal to those of the grouper.
        (Series(range(4)), {"A": [0, 3], "B": [1, 2]}),
        # Group a series with length equal to that of the grouper and index unequal to
        # that of the grouper.
        (Series(range(4)).rename(lambda idx: idx + 1), {"A": [2], "B": [0, 1]}),
        # GH44179: Group a series with length unequal to that of the grouper.
        (Series(range(7)), {"A": [0, 3], "B": [1, 2]}),
    ],
)
def test_categorical_series(series, data):
    # Group the given series by a series with categorical data type such that group A
    # takes indices 0 and 3 and group B indices 1 and 2, obtaining the values mapped in
    # the given data.
    groupby = series.groupby(Series(list("ABBA"), dtype="category"), observed=False)
    result = groupby.aggregate(list)
    expected = Series(data, index=CategoricalIndex(data.keys()))
    tm.assert_series_equal(result, expected)


def test_as_index():
    # GH13204
    df = DataFrame(
        {
            "cat": Categorical([1, 2, 2], [1, 2, 3]),
            "A": [10, 11, 11],
            "B": [101, 102, 103],
        }
    )
    result = df.groupby(["cat", "A"], as_index=False, observed=True).sum()
    expected = DataFrame(
        {
            "cat": Categorical([1, 2], categories=df.cat.cat.categories),
            "A": [10, 11],
            "B": [101, 205],
        },
        columns=["cat", "A", "B"],
    )
    tm.assert_frame_equal(result, expected)

    # function grouper
    f = lambda r: df.loc[r, "A"]
    msg = "A grouping .* was excluded from the result"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(["cat", f], as_index=False, observed=True).sum()
    expected = DataFrame(
        {
            "cat": Categorical([1, 2], categories=df.cat.cat.categories),
            "A": [10, 22],
            "B": [101, 205],
        },
        columns=["cat", "A", "B"],
    )
    tm.assert_frame_equal(result, expected)

    # another not in-axis grouper (conflicting names in index)
    s = Series(["a", "b", "b"], name="cat")
    msg = "A grouping .* was excluded from the result"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby(["cat", s], as_index=False, observed=True).sum()
    tm.assert_frame_equal(result, expected)

    # is original index dropped?
    group_columns = ["cat", "A"]
    expected = DataFrame(
        {
            "cat": Categorical([1, 2], categories=df.cat.cat.categories),
            "A": [10, 11],
            "B": [101, 205],
        },
        columns=["cat", "A", "B"],
    )

    for name in [None, "X", "B"]:
        df.index = Index(list("abc"), name=name)
        result = df.groupby(group_columns, as_index=False, observed=True).sum()

        tm.assert_frame_equal(result, expected)


def test_preserve_categories():
    # GH-13179
    categories = list("abc")

    # ordered=True
    df = DataFrame({"A": Categorical(list("ba"), categories=categories, ordered=True)})
    sort_index = CategoricalIndex(categories, categories, ordered=True, name="A")
    nosort_index = CategoricalIndex(list("bac"), categories, ordered=True, name="A")
    tm.assert_index_equal(
        df.groupby("A", sort=True, observed=False).first().index, sort_index
    )
    # GH#42482 - don't sort result when sort=False, even when ordered=True
    tm.assert_index_equal(
        df.groupby("A", sort=False, observed=False).first().index, nosort_index
    )

    # ordered=False
    df = DataFrame({"A": Categorical(list("ba"), categories=categories, ordered=False)})
    sort_index = CategoricalIndex(categories, categories, ordered=False, name="A")
    # GH#48749 - don't change order of categories
    # GH#42482 - don't sort result when sort=False, even when ordered=True
    nosort_index = CategoricalIndex(list("bac"), list("abc"), ordered=False, name="A")
    tm.assert_index_equal(
        df.groupby("A", sort=True, observed=False).first().index, sort_index
    )
    tm.assert_index_equal(
        df.groupby("A", sort=False, observed=False).first().index, nosort_index
    )


def test_preserve_categorical_dtype():
    # GH13743, GH13854
    df = DataFrame(
        {
            "A": [1, 2, 1, 1, 2],
            "B": [10, 16, 22, 28, 34],
            "C1": Categorical(list("abaab"), categories=list("bac"), ordered=False),
            "C2": Categorical(list("abaab"), categories=list("bac"), ordered=True),
        }
    )
    # single grouper
    exp_full = DataFrame(
        {
            "A": [2.0, 1.0, np.nan],
            "B": [25.0, 20.0, np.nan],
            "C1": Categorical(list("bac"), categories=list("bac"), ordered=False),
            "C2": Categorical(list("bac"), categories=list("bac"), ordered=True),
        }
    )
    for col in ["C1", "C2"]:
        result1 = df.groupby(by=col, as_index=False, observed=False).mean(
            numeric_only=True
        )
        result2 = (
            df.groupby(by=col, as_index=True, observed=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        expected = exp_full.reindex(columns=result1.columns)
        tm.assert_frame_equal(result1, expected)
        tm.assert_frame_equal(result2, expected)


@pytest.mark.parametrize(
    "func, values",
    [
        ("first", ["second", "first"]),
        ("last", ["fourth", "third"]),
        ("min", ["fourth", "first"]),
        ("max", ["second", "third"]),
    ],
)
def test_preserve_on_ordered_ops(func, values):
    # gh-18502
    # preserve the categoricals on ops
    c = Categorical(["first", "second", "third", "fourth"], ordered=True)
    df = DataFrame({"payload": [-1, -2, -1, -2], "col": c})
    g = df.groupby("payload")
    result = getattr(g, func)()
    expected = DataFrame(
        {"payload": [-2, -1], "col": Series(values, dtype=c.dtype)}
    ).set_index("payload")
    tm.assert_frame_equal(result, expected)

    # we should also preserve categorical for SeriesGroupBy
    sgb = df.groupby("payload")["col"]
    result = getattr(sgb, func)()
    expected = expected["col"]
    tm.assert_series_equal(result, expected)


def test_categorical_no_compress():
    data = Series(np.random.default_rng(2).standard_normal(9))

    codes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    cats = Categorical.from_codes(codes, [0, 1, 2], ordered=True)

    result = data.groupby(cats, observed=False).mean()
    exp = data.groupby(codes, observed=False).mean()

    exp.index = CategoricalIndex(
        exp.index, categories=cats.categories, ordered=cats.ordered
    )
    tm.assert_series_equal(result, exp)

    codes = np.array([0, 0, 0, 1, 1, 1, 3, 3, 3])
    cats = Categorical.from_codes(codes, [0, 1, 2, 3], ordered=True)

    result = data.groupby(cats, observed=False).mean()
    exp = data.groupby(codes, observed=False).mean().reindex(cats.categories)
    exp.index = CategoricalIndex(
        exp.index, categories=cats.categories, ordered=cats.ordered
    )
    tm.assert_series_equal(result, exp)

    cats = Categorical(
        ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    data = DataFrame({"a": [1, 1, 1, 2, 2, 2, 3, 4, 5], "b": cats})

    result = data.groupby("b", observed=False).mean()
    result = result["a"].values
    exp = np.array([1, 2, 4, np.nan])
    tm.assert_numpy_array_equal(result, exp)


def test_groupby_empty_with_category():
    # GH-9614
    # test fix for when group by on None resulted in
    # coercion of dtype categorical -> float
    df = DataFrame({"A": [None] * 3, "B": Categorical(["train", "train", "test"])})
    result = df.groupby("A").first()["B"]
    expected = Series(
        Categorical([], categories=["test", "train"]),
        index=Series([], dtype="object", name="A"),
        name="B",
    )
    tm.assert_series_equal(result, expected)


def test_sort():
    # https://stackoverflow.com/questions/23814368/sorting-pandas-
    #        categorical-labels-after-groupby
    # This should result in a properly sorted Series so that the plot
    # has a sorted x axis
    # self.cat.groupby(['value_group'])['value_group'].count().plot(kind='bar')

    df = DataFrame({"value": np.random.default_rng(2).integers(0, 10000, 100)})
    labels = [f"{i} - {i+499}" for i in range(0, 10000, 500)]
    cat_labels = Categorical(labels, labels)

    df = df.sort_values(by=["value"], ascending=True)
    df["value_group"] = pd.cut(
        df.value, range(0, 10500, 500), right=False, labels=cat_labels
    )

    res = df.groupby(["value_group"], observed=False)["value_group"].count()
    exp = res[sorted(res.index, key=lambda x: float(x.split()[0]))]
    exp.index = CategoricalIndex(exp.index, name=exp.index.name)
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize("ordered", [True, False])
def test_sort2(sort, ordered):
    # dataframe groupby sort was being ignored # GH 8868
    # GH#48749 - don't change order of categories
    # GH#42482 - don't sort result when sort=False, even when ordered=True
    df = DataFrame(
        [
            ["(7.5, 10]", 10, 10],
            ["(7.5, 10]", 8, 20],
            ["(2.5, 5]", 5, 30],
            ["(5, 7.5]", 6, 40],
            ["(2.5, 5]", 4, 50],
            ["(0, 2.5]", 1, 60],
            ["(5, 7.5]", 7, 70],
        ],
        columns=["range", "foo", "bar"],
    )
    df["range"] = Categorical(df["range"], ordered=ordered)
    result = df.groupby("range", sort=sort, observed=False).first()

    if sort:
        data_values = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values = ["(0, 2.5]", "(2.5, 5]", "(5, 7.5]", "(7.5, 10]"]
    else:
        data_values = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values = ["(7.5, 10]", "(2.5, 5]", "(5, 7.5]", "(0, 2.5]"]
    expected = DataFrame(
        data_values,
        columns=["foo", "bar"],
        index=CategoricalIndex(index_values, name="range", ordered=ordered),
    )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_sort_datetimelike(sort, ordered):
    # GH10505
    # GH#42482 - don't sort result when sort=False, even when ordered=True

    # use same data as test_groupby_sort_categorical, which category is
    # corresponding to datetime.month
    df = DataFrame(
        {
            "dt": [
                datetime(2011, 7, 1),
                datetime(2011, 7, 1),
                datetime(2011, 2, 1),
                datetime(2011, 5, 1),
                datetime(2011, 2, 1),
                datetime(2011, 1, 1),
                datetime(2011, 5, 1),
            ],
            "foo": [10, 8, 5, 6, 4, 1, 7],
            "bar": [10, 20, 30, 40, 50, 60, 70],
        },
        columns=["dt", "foo", "bar"],
    )

    # ordered=True
    df["dt"] = Categorical(df["dt"], ordered=ordered)
    if sort:
        data_values = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values = [
            datetime(2011, 1, 1),
            datetime(2011, 2, 1),
            datetime(2011, 5, 1),
            datetime(2011, 7, 1),
        ]
    else:
        data_values = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values = [
            datetime(2011, 7, 1),
            datetime(2011, 2, 1),
            datetime(2011, 5, 1),
            datetime(2011, 1, 1),
        ]
    expected = DataFrame(
        data_values,
        columns=["foo", "bar"],
        index=CategoricalIndex(index_values, name="dt", ordered=ordered),
    )
    result = df.groupby("dt", sort=sort, observed=False).first()
    tm.assert_frame_equal(result, expected)


def test_empty_sum():
    # https://github.com/pandas-dev/pandas/issues/18678
    df = DataFrame(
        {"A": Categorical(["a", "a", "b"], categories=["a", "b", "c"]), "B": [1, 2, 1]}
    )
    expected_idx = CategoricalIndex(["a", "b", "c"], name="A")

    # 0 by default
    result = df.groupby("A", observed=False).B.sum()
    expected = Series([3, 1, 0], expected_idx, name="B")
    tm.assert_series_equal(result, expected)

    # min_count=0
    result = df.groupby("A", observed=False).B.sum(min_count=0)
    expected = Series([3, 1, 0], expected_idx, name="B")
    tm.assert_series_equal(result, expected)

    # min_count=1
    result = df.groupby("A", observed=False).B.sum(min_count=1)
    expected = Series([3, 1, np.nan], expected_idx, name="B")
    tm.assert_series_equal(result, expected)

    # min_count>1
    result = df.groupby("A", observed=False).B.sum(min_count=2)
    expected = Series([3, np.nan, np.nan], expected_idx, name="B")
    tm.assert_series_equal(result, expected)


def test_empty_prod():
    # https://github.com/pandas-dev/pandas/issues/18678
    df = DataFrame(
        {"A": Categorical(["a", "a", "b"], categories=["a", "b", "c"]), "B": [1, 2, 1]}
    )

    expected_idx = CategoricalIndex(["a", "b", "c"], name="A")

    # 1 by default
    result = df.groupby("A", observed=False).B.prod()
    expected = Series([2, 1, 1], expected_idx, name="B")
    tm.assert_series_equal(result, expected)

    # min_count=0
    result = df.groupby("A", observed=False).B.prod(min_count=0)
    expected = Series([2, 1, 1], expected_idx, name="B")
    tm.assert_series_equal(result, expected)

    # min_count=1
    result = df.groupby("A", observed=False).B.prod(min_count=1)
    expected = Series([2, 1, np.nan], expected_idx, name="B")
    tm.assert_series_equal(result, expected)


def test_groupby_multiindex_categorical_datetime():
    # https://github.com/pandas-dev/pandas/issues/21390

    df = DataFrame(
        {
            "key1": Categorical(list("abcbabcba")),
            "key2": Categorical(
                list(pd.date_range("2018-06-01 00", freq="1min", periods=3)) * 3
            ),
            "values": np.arange(9),
        }
    )
    result = df.groupby(["key1", "key2"], observed=False).mean()

    idx = MultiIndex.from_product(
        [
            Categorical(["a", "b", "c"]),
            Categorical(pd.date_range("2018-06-01 00", freq="1min", periods=3)),
        ],
        names=["key1", "key2"],
    )
    expected = DataFrame({"values": [0, 4, 8, 3, 4, 5, 6, np.nan, 2]}, index=idx)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "as_index, expected",
    [
        (
            True,
            Series(
                index=MultiIndex.from_arrays(
                    [Series([1, 1, 2], dtype="category"), [1, 2, 2]], names=["a", "b"]
                ),
                data=[1, 2, 3],
                name="x",
            ),
        ),
        (
            False,
            DataFrame(
                {
                    "a": Series([1, 1, 2], dtype="category"),
                    "b": [1, 2, 2],
                    "x": [1, 2, 3],
                }
            ),
        ),
    ],
)
def test_groupby_agg_observed_true_single_column(as_index, expected):
    # GH-23970
    df = DataFrame(
        {"a": Series([1, 1, 2], dtype="category"), "b": [1, 2, 2], "x": [1, 2, 3]}
    )

    result = df.groupby(["a", "b"], as_index=as_index, observed=True)["x"].sum()

    tm.assert_equal(result, expected)


@pytest.mark.parametrize("fill_value", [None, np.nan, pd.NaT])
def test_shift(fill_value):
    ct = Categorical(
        ["a", "b", "c", "d"], categories=["a", "b", "c", "d"], ordered=False
    )
    expected = Categorical(
        [None, "a", "b", "c"], categories=["a", "b", "c", "d"], ordered=False
    )
    res = ct.shift(1, fill_value=fill_value)
    tm.assert_equal(res, expected)


@pytest.fixture
def df_cat(df):
    """
    DataFrame with multiple categorical columns and a column of integers.
    Shortened so as not to contain all possible combinations of categories.
    Useful for testing `observed` kwarg functionality on GroupBy objects.

    Parameters
    ----------
    df: DataFrame
        Non-categorical, longer DataFrame from another fixture, used to derive
        this one

    Returns
    -------
    df_cat: DataFrame
    """
    df_cat = df.copy()[:4]  # leave out some groups
    df_cat["A"] = df_cat["A"].astype("category")
    df_cat["B"] = df_cat["B"].astype("category")
    df_cat["C"] = Series([1, 2, 3, 4])
    df_cat = df_cat.drop(["D"], axis=1)
    return df_cat


@pytest.mark.parametrize("operation", ["agg", "apply"])
def test_seriesgroupby_observed_true(df_cat, operation):
    # GH#24880
    # GH#49223 - order of results was wrong when grouping by index levels
    lev_a = Index(["bar", "bar", "foo", "foo"], dtype=df_cat["A"].dtype, name="A")
    lev_b = Index(["one", "three", "one", "two"], dtype=df_cat["B"].dtype, name="B")
    index = MultiIndex.from_arrays([lev_a, lev_b])
    expected = Series(data=[2, 4, 1, 3], index=index, name="C").sort_index()

    grouped = df_cat.groupby(["A", "B"], observed=True)["C"]
    msg = "using np.sum" if operation == "apply" else "using SeriesGroupBy.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("operation", ["agg", "apply"])
@pytest.mark.parametrize("observed", [False, None])
def test_seriesgroupby_observed_false_or_none(df_cat, observed, operation):
    # GH 24880
    # GH#49223 - order of results was wrong when grouping by index levels
    index, _ = MultiIndex.from_product(
        [
            CategoricalIndex(["bar", "foo"], ordered=False),
            CategoricalIndex(["one", "three", "two"], ordered=False),
        ],
        names=["A", "B"],
    ).sortlevel()

    expected = Series(data=[2, 4, np.nan, 1, np.nan, 3], index=index, name="C")
    if operation == "agg":
        msg = "The 'downcast' keyword in fillna is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = expected.fillna(0, downcast="infer")
    grouped = df_cat.groupby(["A", "B"], observed=observed)["C"]
    msg = "using SeriesGroupBy.sum" if operation == "agg" else "using np.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        result = getattr(grouped, operation)(sum)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "observed, index, data",
    [
        (
            True,
            MultiIndex.from_arrays(
                [
                    Index(["bar"] * 4 + ["foo"] * 4, dtype="category", name="A"),
                    Index(
                        ["one", "one", "three", "three", "one", "one", "two", "two"],
                        dtype="category",
                        name="B",
                    ),
                    Index(["min", "max"] * 4),
                ]
            ),
            [2, 2, 4, 4, 1, 1, 3, 3],
        ),
        (
            False,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
        (
            None,
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
    ],
)
def test_seriesgroupby_observed_apply_dict(df_cat, observed, index, data):
    # GH 24880
    expected = Series(data=data, index=index, name="C")
    result = df_cat.groupby(["A", "B"], observed=observed)["C"].apply(
        lambda x: {"min": x.min(), "max": x.max()}
    )
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_series_dataframe_consistent(df_cat):
    # GH 20416
    expected = df_cat.groupby(["A", "B"], observed=False)["C"].mean()
    result = df_cat.groupby(["A", "B"], observed=False).mean()["C"]
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("code", [([1, 0, 0]), ([0, 0, 0])])
def test_groupby_categorical_axis_1(code):
    # GH 13420
    df = DataFrame({"a": [1, 2, 3, 4], "b": [-1, -2, -3, -4], "c": [5, 6, 7, 8]})
    cat = Categorical.from_codes(code, categories=list("abc"))
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(cat, axis=1, observed=False)
    result = gb.mean()
    msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb2 = df.T.groupby(cat, axis=0, observed=False)
    expected = gb2.mean().T
    tm.assert_frame_equal(result, expected)


def test_groupby_cat_preserves_structure(observed, ordered):
    # GH 28787
    df = DataFrame(
        {"Name": Categorical(["Bob", "Greg"], ordered=ordered), "Item": [1, 2]},
        columns=["Name", "Item"],
    )
    expected = df.copy()

    result = (
        df.groupby("Name", observed=observed)
        .agg(DataFrame.sum, skipna=True)
        .reset_index()
    )

    tm.assert_frame_equal(result, expected)


def test_get_nonexistent_category():
    # Accessing a Category that is not in the dataframe
    df = DataFrame({"var": ["a", "a", "b", "b"], "val": range(4)})
    with pytest.raises(KeyError, match="'vau'"):
        df.groupby("var").apply(
            lambda rows: DataFrame(
                {"var": [rows.iloc[-1]["var"]], "val": [rows.iloc[-1]["vau"]]}
            )
        )


def test_series_groupby_on_2_categoricals_unobserved(reduction_func, observed):
    # GH 17605
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")

    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABCD")),
            "cat_2": Categorical(list("AB") * 2, categories=list("ABCD")),
            "value": [0.1] * 4,
        }
    )
    args = get_groupby_method_args(reduction_func, df)

    expected_length = 4 if observed else 16

    series_groupby = df.groupby(["cat_1", "cat_2"], observed=observed)["value"]

    if reduction_func == "corrwith":
        # TODO: implemented SeriesGroupBy.corrwith. See GH 32293
        assert not hasattr(series_groupby, reduction_func)
        return

    agg = getattr(series_groupby, reduction_func)

    if not observed and reduction_func in ["idxmin", "idxmax"]:
        # idxmin and idxmax are designed to fail on empty inputs
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            agg(*args)
        return

    result = agg(*args)

    assert len(result) == expected_length


def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func, request
):
    # GH 17605
    # Tests whether the unobserved categories in the result contain 0 or NaN

    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")

    if reduction_func == "corrwith":  # GH 32293
        mark = pytest.mark.xfail(
            reason="TODO: implemented SeriesGroupBy.corrwith. See GH 32293"
        )
        request.applymarker(mark)

    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABC")),
            "cat_2": Categorical(list("AB") * 2, categories=list("ABC")),
            "value": [0.1] * 4,
        }
    )
    unobserved = [tuple("AC"), tuple("BC"), tuple("CA"), tuple("CB"), tuple("CC")]
    args = get_groupby_method_args(reduction_func, df)

    series_groupby = df.groupby(["cat_1", "cat_2"], observed=False)["value"]
    agg = getattr(series_groupby, reduction_func)

    if reduction_func in ["idxmin", "idxmax"]:
        # idxmin and idxmax are designed to fail on empty inputs
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            agg(*args)
        return

    result = agg(*args)

    zero_or_nan = _results_for_groupbys_with_missing_categories[reduction_func]

    for idx in unobserved:
        val = result.loc[idx]
        assert (pd.isna(zero_or_nan) and pd.isna(val)) or (val == zero_or_nan)

    # If we expect unobserved values to be zero, we also expect the dtype to be int.
    # Except for .sum(). If the observed categories sum to dtype=float (i.e. their
    # sums have decimals), then the zeros for the missing categories should also be
    # floats.
    if zero_or_nan == 0 and reduction_func != "sum":
        assert np.issubdtype(result.dtype, np.integer)


def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(reduction_func):
    # GH 23865
    # GH 27075
    # Ensure that df.groupby, when 'by' is two Categorical variables,
    # does not return the categories that are not in df when observed=True
    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")

    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABC")),
            "cat_2": Categorical(list("1111"), categories=list("12")),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats = [("A", "2"), ("B", "2"), ("C", "1"), ("C", "2")]

    df_grp = df.groupby(["cat_1", "cat_2"], observed=True)

    args = get_groupby_method_args(reduction_func, df)
    res = getattr(df_grp, reduction_func)(*args)

    for cat in unobserved_cats:
        assert cat not in res.index


@pytest.mark.parametrize("observed", [False, None])
def test_dataframe_groupby_on_2_categoricals_when_observed_is_false(
    reduction_func, observed
):
    # GH 23865
    # GH 27075
    # Ensure that df.groupby, when 'by' is two Categorical variables,
    # returns the categories that are not in df when observed=False/None

    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")

    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABC")),
            "cat_2": Categorical(list("1111"), categories=list("12")),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    unobserved_cats = [("A", "2"), ("B", "2"), ("C", "1"), ("C", "2")]

    df_grp = df.groupby(["cat_1", "cat_2"], observed=observed)

    args = get_groupby_method_args(reduction_func, df)

    if not observed and reduction_func in ["idxmin", "idxmax"]:
        # idxmin and idxmax are designed to fail on empty inputs
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            getattr(df_grp, reduction_func)(*args)
        return

    res = getattr(df_grp, reduction_func)(*args)

    expected = _results_for_groupbys_with_missing_categories[reduction_func]

    if expected is np.nan:
        assert res.loc[unobserved_cats].isnull().all().all()
    else:
        assert (res.loc[unobserved_cats] == expected).all().all()


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_series_groupby_categorical_aggregation_getitem():
    # GH 8870
    d = {"foo": [10, 8, 4, 1], "bar": [10, 20, 30, 40], "baz": ["d", "c", "d", "c"]}
    df = DataFrame(d)
    cat = pd.cut(df["foo"], np.linspace(0, 20, 5))
    df["range"] = cat
    groups = df.groupby(["range", "baz"], as_index=True, sort=True, observed=False)
    result = groups["foo"].agg("mean")
    expected = groups.agg("mean")["foo"]
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, expected_values",
    [(Series.nunique, [1, 1, 2]), (Series.count, [1, 2, 2])],
)
def test_groupby_agg_categorical_columns(func, expected_values):
    # 31256
    df = DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "groups": [0, 1, 1, 2, 2],
            "value": Categorical([0, 0, 0, 0, 1]),
        }
    ).set_index("id")
    result = df.groupby("groups").agg(func)

    expected = DataFrame(
        {"value": expected_values}, index=Index([0, 1, 2], name="groups")
    )
    tm.assert_frame_equal(result, expected)


def test_groupby_agg_non_numeric():
    df = DataFrame({"A": Categorical(["a", "a", "b"], categories=["a", "b", "c"])})
    expected = DataFrame({"A": [2, 1]}, index=np.array([1, 2]))

    result = df.groupby([1, 2, 1]).agg(Series.nunique)
    tm.assert_frame_equal(result, expected)

    result = df.groupby([1, 2, 1]).nunique()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["first", "last"])
def test_groupby_first_returned_categorical_instead_of_dataframe(func):
    # GH 28641: groupby drops index, when grouping over categorical column with
    # first/last. Renamed Categorical instead of DataFrame previously.
    df = DataFrame({"A": [1997], "B": Series(["b"], dtype="category").cat.as_ordered()})
    df_grouped = df.groupby("A")["B"]
    result = getattr(df_grouped, func)()

    # ordered categorical dtype should be preserved
    expected = Series(
        ["b"], index=Index([1997], name="A"), name="B", dtype=df["B"].dtype
    )
    tm.assert_series_equal(result, expected)


def test_read_only_category_no_sort():
    # GH33410
    cats = np.array([1, 2])
    cats.flags.writeable = False
    df = DataFrame(
        {"a": [1, 3, 5, 7], "b": Categorical([1, 1, 2, 2], categories=Index(cats))}
    )
    expected = DataFrame(data={"a": [2.0, 6.0]}, index=CategoricalIndex(cats, name="b"))
    result = df.groupby("b", sort=False, observed=False).mean()
    tm.assert_frame_equal(result, expected)


def test_sorted_missing_category_values():
    # GH 28597
    df = DataFrame(
        {
            "foo": [
                "small",
                "large",
                "large",
                "large",
                "medium",
                "large",
                "large",
                "medium",
            ],
            "bar": ["C", "A", "A", "C", "A", "C", "A", "C"],
        }
    )
    df["foo"] = (
        df["foo"]
        .astype("category")
        .cat.set_categories(["tiny", "small", "medium", "large"], ordered=True)
    )

    expected = DataFrame(
        {
            "tiny": {"A": 0, "C": 0},
            "small": {"A": 0, "C": 1},
            "medium": {"A": 1, "C": 1},
            "large": {"A": 3, "C": 2},
        }
    )
    expected = expected.rename_axis("bar", axis="index")
    expected.columns = CategoricalIndex(
        ["tiny", "small", "medium", "large"],
        categories=["tiny", "small", "medium", "large"],
        ordered=True,
        name="foo",
        dtype="category",
    )

    result = df.groupby(["bar", "foo"], observed=False).size().unstack()

    tm.assert_frame_equal(result, expected)


def test_agg_cython_category_not_implemented_fallback():
    # https://github.com/pandas-dev/pandas/issues/31450
    df = DataFrame({"col_num": [1, 1, 2, 3]})
    df["col_cat"] = df["col_num"].astype("category")

    result = df.groupby("col_num").col_cat.first()

    # ordered categorical dtype should definitely be preserved;
    #  this is unordered, so is less-clear case (if anything, it should raise)
    expected = Series(
        [1, 2, 3],
        index=Index([1, 2, 3], name="col_num"),
        name="col_cat",
        dtype=df["col_cat"].dtype,
    )
    tm.assert_series_equal(result, expected)

    result = df.groupby("col_num").agg({"col_cat": "first"})
    expected = expected.to_frame()
    tm.assert_frame_equal(result, expected)


def test_aggregate_categorical_with_isnan():
    # GH 29837
    df = DataFrame(
        {
            "A": [1, 1, 1, 1],
            "B": [1, 2, 1, 2],
            "numerical_col": [0.1, 0.2, np.nan, 0.3],
            "object_col": ["foo", "bar", "foo", "fee"],
            "categorical_col": ["foo", "bar", "foo", "fee"],
        }
    )

    df = df.astype({"categorical_col": "category"})

    result = df.groupby(["A", "B"]).agg(lambda df: df.isna().sum())
    index = MultiIndex.from_arrays([[1, 1], [1, 2]], names=("A", "B"))
    expected = DataFrame(
        data={
            "numerical_col": [1, 0],
            "object_col": [0, 0],
            "categorical_col": [0, 0],
        },
        index=index,
    )
    tm.assert_frame_equal(result, expected)


def test_categorical_transform():
    # GH 29037
    df = DataFrame(
        {
            "package_id": [1, 1, 1, 2, 2, 3],
            "status": [
                "Waiting",
                "OnTheWay",
                "Delivered",
                "Waiting",
                "OnTheWay",
                "Waiting",
            ],
        }
    )

    delivery_status_type = pd.CategoricalDtype(
        categories=["Waiting", "OnTheWay", "Delivered"], ordered=True
    )
    df["status"] = df["status"].astype(delivery_status_type)
    msg = "using SeriesGroupBy.max"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # GH#53425
        df["last_status"] = df.groupby("package_id")["status"].transform(max)
    result = df.copy()

    expected = DataFrame(
        {
            "package_id": [1, 1, 1, 2, 2, 3],
            "status": [
                "Waiting",
                "OnTheWay",
                "Delivered",
                "Waiting",
                "OnTheWay",
                "Waiting",
            ],
            "last_status": [
                "Delivered",
                "Delivered",
                "Delivered",
                "OnTheWay",
                "OnTheWay",
                "Waiting",
            ],
        }
    )

    expected["status"] = expected["status"].astype(delivery_status_type)

    # .transform(max) should preserve ordered categoricals
    expected["last_status"] = expected["last_status"].astype(delivery_status_type)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["first", "last"])
def test_series_groupby_first_on_categorical_col_grouped_on_2_categoricals(
    func: str, observed: bool
):
    # GH 34951
    cat = Categorical([0, 0, 1, 1])
    val = [0, 1, 1, 0]
    df = DataFrame({"a": cat, "b": cat, "c": val})

    cat2 = Categorical([0, 1])
    idx = MultiIndex.from_product([cat2, cat2], names=["a", "b"])
    expected_dict = {
        "first": Series([0, np.nan, np.nan, 1], idx, name="c"),
        "last": Series([1, np.nan, np.nan, 0], idx, name="c"),
    }

    expected = expected_dict[func]
    if observed:
        expected = expected.dropna().astype(np.int64)

    srs_grp = df.groupby(["a", "b"], observed=observed)["c"]
    result = getattr(srs_grp, func)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["first", "last"])
def test_df_groupby_first_on_categorical_col_grouped_on_2_categoricals(
    func: str, observed: bool
):
    # GH 34951
    cat = Categorical([0, 0, 1, 1])
    val = [0, 1, 1, 0]
    df = DataFrame({"a": cat, "b": cat, "c": val})

    cat2 = Categorical([0, 1])
    idx = MultiIndex.from_product([cat2, cat2], names=["a", "b"])
    expected_dict = {
        "first": Series([0, np.nan, np.nan, 1], idx, name="c"),
        "last": Series([1, np.nan, np.nan, 0], idx, name="c"),
    }

    expected = expected_dict[func].to_frame()
    if observed:
        expected = expected.dropna().astype(np.int64)

    df_grp = df.groupby(["a", "b"], observed=observed)
    result = getattr(df_grp, func)()
    tm.assert_frame_equal(result, expected)


def test_groupby_categorical_indices_unused_categories():
    # GH#38642
    df = DataFrame(
        {
            "key": Categorical(["b", "b", "a"], categories=["a", "b", "c"]),
            "col": range(3),
        }
    )
    grouped = df.groupby("key", sort=False, observed=False)
    result = grouped.indices
    expected = {
        "b": np.array([0, 1], dtype="intp"),
        "a": np.array([2], dtype="intp"),
        "c": np.array([], dtype="intp"),
    }
    assert result.keys() == expected.keys()
    for key in result.keys():
        tm.assert_numpy_array_equal(result[key], expected[key])


@pytest.mark.parametrize("func", ["first", "last"])
def test_groupby_last_first_preserve_categoricaldtype(func):
    # GH#33090
    df = DataFrame({"a": [1, 2, 3]})
    df["b"] = df["a"].astype("category")
    result = getattr(df.groupby("a")["b"], func)()
    expected = Series(
        Categorical([1, 2, 3]), name="b", index=Index([1, 2, 3], name="a")
    )
    tm.assert_series_equal(expected, result)


def test_groupby_categorical_observed_nunique():
    # GH#45128
    df = DataFrame({"a": [1, 2], "b": [1, 2], "c": [10, 11]})
    df = df.astype(dtype={"a": "category", "b": "category"})
    result = df.groupby(["a", "b"], observed=True).nunique()["c"]
    expected = Series(
        [1, 1],
        index=MultiIndex.from_arrays(
            [CategoricalIndex([1, 2], name="a"), CategoricalIndex([1, 2], name="b")]
        ),
        name="c",
    )
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_aggregate_functions():
    # GH#37275
    dtype = pd.CategoricalDtype(categories=["small", "big"], ordered=True)
    df = DataFrame(
        [[1, "small"], [1, "big"], [2, "small"]], columns=["grp", "description"]
    ).astype({"description": dtype})

    result = df.groupby("grp")["description"].max()
    expected = Series(
        ["big", "small"],
        index=Index([1, 2], name="grp"),
        name="description",
        dtype=pd.CategoricalDtype(categories=["small", "big"], ordered=True),
    )

    tm.assert_series_equal(result, expected)


def test_groupby_categorical_dropna(observed, dropna):
    # GH#48645 - dropna should have no impact on the result when there are no NA values
    cat = Categorical([1, 2], categories=[1, 2, 3])
    df = DataFrame({"x": Categorical([1, 2], categories=[1, 2, 3]), "y": [3, 4]})
    gb = df.groupby("x", observed=observed, dropna=dropna)
    result = gb.sum()

    if observed:
        expected = DataFrame({"y": [3, 4]}, index=cat)
    else:
        index = CategoricalIndex([1, 2, 3], [1, 2, 3])
        expected = DataFrame({"y": [3, 4, 0]}, index=index)
    expected.index.name = "x"

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_reducer(
    request, as_index, sort, observed, reduction_func, index_kind, ordered
):
    # GH#48749
    if reduction_func == "corrwith" and not as_index:
        msg = "GH#49950 - corrwith with as_index=False may not have grouping column"
        request.applymarker(pytest.mark.xfail(reason=msg))
    elif index_kind != "range" and not as_index:
        pytest.skip(reason="Result doesn't have categories, nothing to test")
    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    if index_kind == "range":
        keys = ["a"]
    elif index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
    args = get_groupby_method_args(reduction_func, df)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)

    if not observed and reduction_func in ["idxmin", "idxmax"]:
        # idxmin and idxmax are designed to fail on empty inputs
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            getattr(gb, reduction_func)(*args)
        return

    op_result = getattr(gb, reduction_func)(*args)
    if as_index:
        result = op_result.index.get_level_values("a").categories
    else:
        result = op_result["a"].cat.categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)

    if index_kind == "multi":
        result = op_result.index.get_level_values("a2").categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["single", "multi"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_transformer(
    as_index, sort, observed, transformation_func, index_kind, ordered
):
    # GH#48749
    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    if index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
    args = get_groupby_method_args(transformation_func, df)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    warn = FutureWarning if transformation_func == "fillna" else None
    msg = "DataFrameGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=msg):
        op_result = getattr(gb, transformation_func)(*args)
    result = op_result.index.get_level_values("a").categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)

    if index_kind == "multi":
        result = op_result.index.get_level_values("a2").categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
@pytest.mark.parametrize("method", ["head", "tail"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_head_tail(
    as_index, sort, observed, method, index_kind, ordered
):
    # GH#48749
    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    if index_kind == "range":
        keys = ["a"]
    elif index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    op_result = getattr(gb, method)()
    if index_kind == "range":
        result = op_result["a"].cat.categories
    else:
        result = op_result.index.get_level_values("a").categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)

    if index_kind == "multi":
        result = op_result.index.get_level_values("a2").categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
@pytest.mark.parametrize("method", ["apply", "agg", "transform"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_apply(as_index, sort, observed, method, index_kind, ordered):
    # GH#48749
    if (method == "transform" and index_kind == "range") or (
        not as_index and index_kind != "range"
    ):
        pytest.skip("No categories in result, nothing to test")
    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    if index_kind == "range":
        keys = ["a"]
    elif index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    warn = FutureWarning if method == "apply" and index_kind == "range" else None
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(warn, match=msg):
        op_result = getattr(gb, method)(lambda x: x.sum(numeric_only=True))
    if (method == "transform" or not as_index) and index_kind == "range":
        result = op_result["a"].cat.categories
    else:
        result = op_result.index.get_level_values("a").categories
    expected = Index([1, 4, 3, 2])
    tm.assert_index_equal(result, expected)

    if index_kind == "multi":
        result = op_result.index.get_level_values("a2").categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
def test_many_categories(as_index, sort, index_kind, ordered):
    # GH#48749 - Test when the grouper has many categories
    if index_kind != "range" and not as_index:
        pytest.skip(reason="Result doesn't have categories, nothing to test")
    categories = np.arange(9999, -1, -1)
    grouper = Categorical([2, 1, 2, 3], categories=categories, ordered=ordered)
    df = DataFrame({"a": grouper, "b": range(4)})
    if index_kind == "range":
        keys = ["a"]
    elif index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=True)
    result = gb.sum()

    # Test is setup so that data and index are the same values
    data = [3, 2, 1] if sort else [2, 1, 3]

    index = CategoricalIndex(
        data, categories=grouper.categories, ordered=ordered, name="a"
    )
    if as_index:
        expected = DataFrame({"b": data})
        if index_kind == "multi":
            expected.index = MultiIndex.from_frame(DataFrame({"a": index, "a2": index}))
        else:
            expected.index = index
    elif index_kind == "multi":
        expected = DataFrame({"a": Series(index), "a2": Series(index), "b": data})
    else:
        expected = DataFrame({"a": Series(index), "b": data})

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("cat_columns", ["a", "b", ["a", "b"]])
@pytest.mark.parametrize("keys", ["a", "b", ["a", "b"]])
def test_groupby_default_depr(cat_columns, keys):
    # GH#43999
    df = DataFrame({"a": [1, 1, 2, 3], "b": [4, 5, 6, 7]})
    df[cat_columns] = df[cat_columns].astype("category")
    msg = "The default of observed=False is deprecated"
    klass = FutureWarning if set(cat_columns) & set(keys) else None
    with tm.assert_produces_warning(klass, match=msg):
        df.groupby(keys)


@pytest.mark.parametrize("test_series", [True, False])
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
def test_agg_list(request, as_index, observed, reduction_func, test_series, keys):
    # GH#52760
    if test_series and reduction_func == "corrwith":
        assert not hasattr(SeriesGroupBy, "corrwith")
        pytest.skip("corrwith not implemented for SeriesGroupBy")
    elif reduction_func == "corrwith":
        msg = "GH#32293: attempts to call SeriesGroupBy.corrwith"
        request.applymarker(pytest.mark.xfail(reason=msg))
    elif (
        reduction_func == "nunique"
        and not test_series
        and len(keys) != 1
        and not observed
        and not as_index
    ):
        msg = "GH#52848 - raises a ValueError"
        request.applymarker(pytest.mark.xfail(reason=msg))

    df = DataFrame({"a1": [0, 0, 1], "a2": [2, 3, 3], "b": [4, 5, 6]})
    df = df.astype({"a1": "category", "a2": "category"})
    if "a2" not in keys:
        df = df.drop(columns="a2")
    gb = df.groupby(by=keys, as_index=as_index, observed=observed)
    if test_series:
        gb = gb["b"]
    args = get_groupby_method_args(reduction_func, df)

    if not observed and reduction_func in ["idxmin", "idxmax"] and keys == ["a1", "a2"]:
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            gb.agg([reduction_func], *args)
        return

    result = gb.agg([reduction_func], *args)
    expected = getattr(gb, reduction_func)(*args)

    if as_index and (test_series or reduction_func == "size"):
        expected = expected.to_frame(reduction_func)
    if not test_series:
        expected.columns = MultiIndex.from_tuples(
            [(ind, "") for ind in expected.columns[:-1]] + [("b", reduction_func)]
        )
    elif not as_index:
        expected.columns = keys + [reduction_func]

    tm.assert_equal(result, expected)