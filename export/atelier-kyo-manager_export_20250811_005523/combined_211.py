
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\concrete\tests\test_delta.py ===
from sympy.concrete import Sum
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
from sympy.core import Eq, S, symbols, oo
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
from sympy.logic import And
from sympy.testing.pytest import raises

i, j, k, l, m = symbols("i j k l m", integer=True, finite=True)
x, y = symbols("x y", commutative=False)


def test_deltaproduct_trivial():
    assert dp(x, (j, 1, 0)) == 1
    assert dp(x, (j, 1, 3)) == x**3
    assert dp(x + y, (j, 1, 3)) == (x + y)**3
    assert dp(x*y, (j, 1, 3)) == (x*y)**3
    assert dp(KD(i, j), (k, 1, 3)) == KD(i, j)
    assert dp(x*KD(i, j), (k, 1, 3)) == x**3*KD(i, j)
    assert dp(x*y*KD(i, j), (k, 1, 3)) == (x*y)**3*KD(i, j)


def test_deltaproduct_basic():
    assert dp(KD(i, j), (j, 1, 3)) == 0
    assert dp(KD(i, j), (j, 1, 1)) == KD(i, 1)
    assert dp(KD(i, j), (j, 2, 2)) == KD(i, 2)
    assert dp(KD(i, j), (j, 3, 3)) == KD(i, 3)
    assert dp(KD(i, j), (j, 1, k)) == KD(i, 1)*KD(k, 1) + KD(k, 0)
    assert dp(KD(i, j), (j, k, 3)) == KD(i, 3)*KD(k, 3) + KD(k, 4)
    assert dp(KD(i, j), (j, k, l)) == KD(i, l)*KD(k, l) + KD(k, l + 1)


def test_deltaproduct_mul_x_kd():
    assert dp(x*KD(i, j), (j, 1, 3)) == 0
    assert dp(x*KD(i, j), (j, 1, 1)) == x*KD(i, 1)
    assert dp(x*KD(i, j), (j, 2, 2)) == x*KD(i, 2)
    assert dp(x*KD(i, j), (j, 3, 3)) == x*KD(i, 3)
    assert dp(x*KD(i, j), (j, 1, k)) == x*KD(i, 1)*KD(k, 1) + KD(k, 0)
    assert dp(x*KD(i, j), (j, k, 3)) == x*KD(i, 3)*KD(k, 3) + KD(k, 4)
    assert dp(x*KD(i, j), (j, k, l)) == x*KD(i, l)*KD(k, l) + KD(k, l + 1)


def test_deltaproduct_mul_add_x_y_kd():
    assert dp((x + y)*KD(i, j), (j, 1, 3)) == 0
    assert dp((x + y)*KD(i, j), (j, 1, 1)) == (x + y)*KD(i, 1)
    assert dp((x + y)*KD(i, j), (j, 2, 2)) == (x + y)*KD(i, 2)
    assert dp((x + y)*KD(i, j), (j, 3, 3)) == (x + y)*KD(i, 3)
    assert dp((x + y)*KD(i, j), (j, 1, k)) == \
        (x + y)*KD(i, 1)*KD(k, 1) + KD(k, 0)
    assert dp((x + y)*KD(i, j), (j, k, 3)) == \
        (x + y)*KD(i, 3)*KD(k, 3) + KD(k, 4)
    assert dp((x + y)*KD(i, j), (j, k, l)) == \
        (x + y)*KD(i, l)*KD(k, l) + KD(k, l + 1)


def test_deltaproduct_add_kd_kd():
    assert dp(KD(i, k) + KD(j, k), (k, 1, 3)) == 0
    assert dp(KD(i, k) + KD(j, k), (k, 1, 1)) == KD(i, 1) + KD(j, 1)
    assert dp(KD(i, k) + KD(j, k), (k, 2, 2)) == KD(i, 2) + KD(j, 2)
    assert dp(KD(i, k) + KD(j, k), (k, 3, 3)) == KD(i, 3) + KD(j, 3)
    assert dp(KD(i, k) + KD(j, k), (k, 1, l)) == KD(l, 0) + \
        KD(i, 1)*KD(l, 1) + KD(j, 1)*KD(l, 1) + \
        KD(i, 1)*KD(j, 2)*KD(l, 2) + KD(j, 1)*KD(i, 2)*KD(l, 2)
    assert dp(KD(i, k) + KD(j, k), (k, l, 3)) == KD(l, 4) + \
        KD(i, 3)*KD(l, 3) + KD(j, 3)*KD(l, 3) + \
        KD(i, 2)*KD(j, 3)*KD(l, 2) + KD(i, 3)*KD(j, 2)*KD(l, 2)
    assert dp(KD(i, k) + KD(j, k), (k, l, m)) == KD(l, m + 1) + \
        KD(i, m)*KD(l, m) + KD(j, m)*KD(l, m) + \
        KD(i, m)*KD(j, m - 1)*KD(l, m - 1) + KD(i, m - 1)*KD(j, m)*KD(l, m - 1)


def test_deltaproduct_mul_x_add_kd_kd():
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 1, 3)) == 0
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 1, 1)) == x*(KD(i, 1) + KD(j, 1))
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 2, 2)) == x*(KD(i, 2) + KD(j, 2))
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 3, 3)) == x*(KD(i, 3) + KD(j, 3))
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 1, l)) == KD(l, 0) + \
        x*KD(i, 1)*KD(l, 1) + x*KD(j, 1)*KD(l, 1) + \
        x**2*KD(i, 1)*KD(j, 2)*KD(l, 2) + x**2*KD(j, 1)*KD(i, 2)*KD(l, 2)
    assert dp(x*(KD(i, k) + KD(j, k)), (k, l, 3)) == KD(l, 4) + \
        x*KD(i, 3)*KD(l, 3) + x*KD(j, 3)*KD(l, 3) + \
        x**2*KD(i, 2)*KD(j, 3)*KD(l, 2) + x**2*KD(i, 3)*KD(j, 2)*KD(l, 2)
    assert dp(x*(KD(i, k) + KD(j, k)), (k, l, m)) == KD(l, m + 1) + \
        x*KD(i, m)*KD(l, m) + x*KD(j, m)*KD(l, m) + \
        x**2*KD(i, m - 1)*KD(j, m)*KD(l, m - 1) + \
        x**2*KD(i, m)*KD(j, m - 1)*KD(l, m - 1)


def test_deltaproduct_mul_add_x_y_add_kd_kd():
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 3)) == 0
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 1)) == \
        (x + y)*(KD(i, 1) + KD(j, 1))
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 2, 2)) == \
        (x + y)*(KD(i, 2) + KD(j, 2))
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 3, 3)) == \
        (x + y)*(KD(i, 3) + KD(j, 3))
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 1, l)) == KD(l, 0) + \
        (x + y)*KD(i, 1)*KD(l, 1) + (x + y)*KD(j, 1)*KD(l, 1) + \
        (x + y)**2*KD(i, 1)*KD(j, 2)*KD(l, 2) + \
        (x + y)**2*KD(j, 1)*KD(i, 2)*KD(l, 2)
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, l, 3)) == KD(l, 4) + \
        (x + y)*KD(i, 3)*KD(l, 3) + (x + y)*KD(j, 3)*KD(l, 3) + \
        (x + y)**2*KD(i, 2)*KD(j, 3)*KD(l, 2) + \
        (x + y)**2*KD(i, 3)*KD(j, 2)*KD(l, 2)
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, l, m)) == KD(l, m + 1) + \
        (x + y)*KD(i, m)*KD(l, m) + (x + y)*KD(j, m)*KD(l, m) + \
        (x + y)**2*KD(i, m - 1)*KD(j, m)*KD(l, m - 1) + \
        (x + y)**2*KD(i, m)*KD(j, m - 1)*KD(l, m - 1)


def test_deltaproduct_add_mul_x_y_mul_x_kd():
    assert dp(x*y + x*KD(i, j), (j, 1, 3)) == (x*y)**3 + \
        x*(x*y)**2*KD(i, 1) + (x*y)*x*(x*y)*KD(i, 2) + (x*y)**2*x*KD(i, 3)
    assert dp(x*y + x*KD(i, j), (j, 1, 1)) == x*y + x*KD(i, 1)
    assert dp(x*y + x*KD(i, j), (j, 2, 2)) == x*y + x*KD(i, 2)
    assert dp(x*y + x*KD(i, j), (j, 3, 3)) == x*y + x*KD(i, 3)
    assert dp(x*y + x*KD(i, j), (j, 1, k)) == \
        (x*y)**k + Piecewise(
            ((x*y)**(i - 1)*x*(x*y)**(k - i), And(1 <= i, i <= k)),
            (0, True)
        )
    assert dp(x*y + x*KD(i, j), (j, k, 3)) == \
        (x*y)**(-k + 4) + Piecewise(
            ((x*y)**(i - k)*x*(x*y)**(3 - i), And(k <= i, i <= 3)),
            (0, True)
        )
    assert dp(x*y + x*KD(i, j), (j, k, l)) == \
        (x*y)**(-k + l + 1) + Piecewise(
            ((x*y)**(i - k)*x*(x*y)**(l - i), And(k <= i, i <= l)),
            (0, True)
        )


def test_deltaproduct_mul_x_add_y_kd():
    assert dp(x*(y + KD(i, j)), (j, 1, 3)) == (x*y)**3 + \
        x*(x*y)**2*KD(i, 1) + (x*y)*x*(x*y)*KD(i, 2) + (x*y)**2*x*KD(i, 3)
    assert dp(x*(y + KD(i, j)), (j, 1, 1)) == x*(y + KD(i, 1))
    assert dp(x*(y + KD(i, j)), (j, 2, 2)) == x*(y + KD(i, 2))
    assert dp(x*(y + KD(i, j)), (j, 3, 3)) == x*(y + KD(i, 3))
    assert dp(x*(y + KD(i, j)), (j, 1, k)) == \
        (x*y)**k + Piecewise(
            ((x*y)**(i - 1)*x*(x*y)**(k - i), And(1 <= i, i <= k)),
            (0, True)
        ).expand()
    assert dp(x*(y + KD(i, j)), (j, k, 3)) == \
        ((x*y)**(-k + 4) + Piecewise(
            ((x*y)**(i - k)*x*(x*y)**(3 - i), And(k <= i, i <= 3)),
            (0, True)
        )).expand()
    assert dp(x*(y + KD(i, j)), (j, k, l)) == \
        ((x*y)**(-k + l + 1) + Piecewise(
            ((x*y)**(i - k)*x*(x*y)**(l - i), And(k <= i, i <= l)),
            (0, True)
        )).expand()


def test_deltaproduct_mul_x_add_y_twokd():
    assert dp(x*(y + 2*KD(i, j)), (j, 1, 3)) == (x*y)**3 + \
        2*x*(x*y)**2*KD(i, 1) + 2*x*y*x*x*y*KD(i, 2) + 2*(x*y)**2*x*KD(i, 3)
    assert dp(x*(y + 2*KD(i, j)), (j, 1, 1)) == x*(y + 2*KD(i, 1))
    assert dp(x*(y + 2*KD(i, j)), (j, 2, 2)) == x*(y + 2*KD(i, 2))
    assert dp(x*(y + 2*KD(i, j)), (j, 3, 3)) == x*(y + 2*KD(i, 3))
    assert dp(x*(y + 2*KD(i, j)), (j, 1, k)) == \
        (x*y)**k + Piecewise(
            (2*(x*y)**(i - 1)*x*(x*y)**(k - i), And(1 <= i, i <= k)),
            (0, True)
        ).expand()
    assert dp(x*(y + 2*KD(i, j)), (j, k, 3)) == \
        ((x*y)**(-k + 4) + Piecewise(
            (2*(x*y)**(i - k)*x*(x*y)**(3 - i), And(k <= i, i <= 3)),
            (0, True)
        )).expand()
    assert dp(x*(y + 2*KD(i, j)), (j, k, l)) == \
        ((x*y)**(-k + l + 1) + Piecewise(
            (2*(x*y)**(i - k)*x*(x*y)**(l - i), And(k <= i, i <= l)),
            (0, True)
        )).expand()


def test_deltaproduct_mul_add_x_y_add_y_kd():
    assert dp((x + y)*(y + KD(i, j)), (j, 1, 3)) == ((x + y)*y)**3 + \
        (x + y)*((x + y)*y)**2*KD(i, 1) + \
        (x + y)*y*(x + y)**2*y*KD(i, 2) + \
        ((x + y)*y)**2*(x + y)*KD(i, 3)
    assert dp((x + y)*(y + KD(i, j)), (j, 1, 1)) == (x + y)*(y + KD(i, 1))
    assert dp((x + y)*(y + KD(i, j)), (j, 2, 2)) == (x + y)*(y + KD(i, 2))
    assert dp((x + y)*(y + KD(i, j)), (j, 3, 3)) == (x + y)*(y + KD(i, 3))
    assert dp((x + y)*(y + KD(i, j)), (j, 1, k)) == \
        ((x + y)*y)**k + Piecewise(
            (((x + y)*y)**(-1)*((x + y)*y)**i*(x + y)*((x + y)*y
            )**k*((x + y)*y)**(-i), (i >= 1) & (i <= k)), (0, True))
    assert dp((x + y)*(y + KD(i, j)), (j, k, 3)) == (
        (x + y)*y)**4*((x + y)*y)**(-k) + Piecewise((((x + y)*y)**i*(
        (x + y)*y)**(-k)*(x + y)*((x + y)*y)**3*((x + y)*y)**(-i),
        (i >= k) & (i <= 3)), (0, True))
    assert dp((x + y)*(y + KD(i, j)), (j, k, l)) == \
        (x + y)*y*((x + y)*y)**l*((x + y)*y)**(-k) + Piecewise(
        (((x + y)*y)**i*((x + y)*y)**(-k)*(x + y)*((x + y)*y
        )**l*((x + y)*y)**(-i), (i >= k) & (i <= l)), (0, True))


def test_deltaproduct_mul_add_x_kd_add_y_kd():
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 1, 3)) == \
        KD(i, 1)*(KD(i, k) + x)*((KD(i, k) + x)*y)**2 + \
        KD(i, 2)*(KD(i, k) + x)*y*(KD(i, k) + x)**2*y + \
        KD(i, 3)*((KD(i, k) + x)*y)**2*(KD(i, k) + x) + \
        ((KD(i, k) + x)*y)**3
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 1, 1)) == \
        (x + KD(i, k))*(y + KD(i, 1))
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 2, 2)) == \
        (x + KD(i, k))*(y + KD(i, 2))
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 3, 3)) == \
        (x + KD(i, k))*(y + KD(i, 3))
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 1, k)) == \
        ((KD(i, k) + x)*y)**k + Piecewise(
        (((KD(i, k) + x)*y)**(-1)*((KD(i, k) + x)*y)**i*(KD(i, k) + x
        )*((KD(i, k) + x)*y)**k*((KD(i, k) + x)*y)**(-i), (i >= 1
        ) & (i <= k)), (0, True))
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, k, 3)) == (
        (KD(i, k) + x)*y)**4*((KD(i, k) + x)*y)**(-k) + Piecewise(
        (((KD(i, k) + x)*y)**i*((KD(i, k) + x)*y)**(-k)*(KD(i, k)
        + x)*((KD(i, k) + x)*y)**3*((KD(i, k) + x)*y)**(-i),
        (i >= k) & (i <= 3)), (0, True))
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, k, l)) == (
        KD(i, k) + x)*y*((KD(i, k) + x)*y)**l*((KD(i, k) + x)*y
        )**(-k) + Piecewise((((KD(i, k) + x)*y)**i*((KD(i, k) + x
        )*y)**(-k)*(KD(i, k) + x)*((KD(i, k) + x)*y)**l*((KD(i, k) + x
        )*y)**(-i), (i >= k) & (i <= l)), (0, True))


def test_deltasummation_trivial():
    assert ds(x, (j, 1, 0)) == 0
    assert ds(x, (j, 1, 3)) == 3*x
    assert ds(x + y, (j, 1, 3)) == 3*(x + y)
    assert ds(x*y, (j, 1, 3)) == 3*x*y
    assert ds(KD(i, j), (k, 1, 3)) == 3*KD(i, j)
    assert ds(x*KD(i, j), (k, 1, 3)) == 3*x*KD(i, j)
    assert ds(x*y*KD(i, j), (k, 1, 3)) == 3*x*y*KD(i, j)


def test_deltasummation_basic_numerical():
    n = symbols('n', integer=True, nonzero=True)
    assert ds(KD(n, 0), (n, 1, 3)) == 0

    # return unevaluated, until it gets implemented
    assert ds(KD(i**2, j**2), (j, -oo, oo)) == \
        Sum(KD(i**2, j**2), (j, -oo, oo))

    assert Piecewise((KD(i, k), And(1 <= i, i <= 3)), (0, True)) == \
        ds(KD(i, j)*KD(j, k), (j, 1, 3)) == \
        ds(KD(j, k)*KD(i, j), (j, 1, 3))

    assert ds(KD(i, k), (k, -oo, oo)) == 1
    assert ds(KD(i, k), (k, 0, oo)) == Piecewise((1, S.Zero <= i), (0, True))
    assert ds(KD(i, k), (k, 1, 3)) == \
        Piecewise((1, And(1 <= i, i <= 3)), (0, True))
    assert ds(k*KD(i, j)*KD(j, k), (k, -oo, oo)) == j*KD(i, j)
    assert ds(j*KD(i, j), (j, -oo, oo)) == i
    assert ds(i*KD(i, j), (i, -oo, oo)) == j
    assert ds(x, (i, 1, 3)) == 3*x
    assert ds((i + j)*KD(i, j), (j, -oo, oo)) == 2*i


def test_deltasummation_basic_symbolic():
    assert ds(KD(i, j), (j, 1, 3)) == \
        Piecewise((1, And(1 <= i, i <= 3)), (0, True))
    assert ds(KD(i, j), (j, 1, 1)) == Piecewise((1, Eq(i, 1)), (0, True))
    assert ds(KD(i, j), (j, 2, 2)) == Piecewise((1, Eq(i, 2)), (0, True))
    assert ds(KD(i, j), (j, 3, 3)) == Piecewise((1, Eq(i, 3)), (0, True))
    assert ds(KD(i, j), (j, 1, k)) == \
        Piecewise((1, And(1 <= i, i <= k)), (0, True))
    assert ds(KD(i, j), (j, k, 3)) == \
        Piecewise((1, And(k <= i, i <= 3)), (0, True))
    assert ds(KD(i, j), (j, k, l)) == \
        Piecewise((1, And(k <= i, i <= l)), (0, True))


def test_deltasummation_mul_x_kd():
    assert ds(x*KD(i, j), (j, 1, 3)) == \
        Piecewise((x, And(1 <= i, i <= 3)), (0, True))
    assert ds(x*KD(i, j), (j, 1, 1)) == Piecewise((x, Eq(i, 1)), (0, True))
    assert ds(x*KD(i, j), (j, 2, 2)) == Piecewise((x, Eq(i, 2)), (0, True))
    assert ds(x*KD(i, j), (j, 3, 3)) == Piecewise((x, Eq(i, 3)), (0, True))
    assert ds(x*KD(i, j), (j, 1, k)) == \
        Piecewise((x, And(1 <= i, i <= k)), (0, True))
    assert ds(x*KD(i, j), (j, k, 3)) == \
        Piecewise((x, And(k <= i, i <= 3)), (0, True))
    assert ds(x*KD(i, j), (j, k, l)) == \
        Piecewise((x, And(k <= i, i <= l)), (0, True))


def test_deltasummation_mul_add_x_y_kd():
    assert ds((x + y)*KD(i, j), (j, 1, 3)) == \
        Piecewise((x + y, And(1 <= i, i <= 3)), (0, True))
    assert ds((x + y)*KD(i, j), (j, 1, 1)) == \
        Piecewise((x + y, Eq(i, 1)), (0, True))
    assert ds((x + y)*KD(i, j), (j, 2, 2)) == \
        Piecewise((x + y, Eq(i, 2)), (0, True))
    assert ds((x + y)*KD(i, j), (j, 3, 3)) == \
        Piecewise((x + y, Eq(i, 3)), (0, True))
    assert ds((x + y)*KD(i, j), (j, 1, k)) == \
        Piecewise((x + y, And(1 <= i, i <= k)), (0, True))
    assert ds((x + y)*KD(i, j), (j, k, 3)) == \
        Piecewise((x + y, And(k <= i, i <= 3)), (0, True))
    assert ds((x + y)*KD(i, j), (j, k, l)) == \
        Piecewise((x + y, And(k <= i, i <= l)), (0, True))


def test_deltasummation_add_kd_kd():
    assert ds(KD(i, k) + KD(j, k), (k, 1, 3)) == piecewise_fold(
        Piecewise((1, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((1, And(1 <= j, j <= 3)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, 1, 1)) == piecewise_fold(
        Piecewise((1, Eq(i, 1)), (0, True)) +
        Piecewise((1, Eq(j, 1)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, 2, 2)) == piecewise_fold(
        Piecewise((1, Eq(i, 2)), (0, True)) +
        Piecewise((1, Eq(j, 2)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, 3, 3)) == piecewise_fold(
        Piecewise((1, Eq(i, 3)), (0, True)) +
        Piecewise((1, Eq(j, 3)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, 1, l)) == piecewise_fold(
        Piecewise((1, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((1, And(1 <= j, j <= l)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, l, 3)) == piecewise_fold(
        Piecewise((1, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((1, And(l <= j, j <= 3)), (0, True)))
    assert ds(KD(i, k) + KD(j, k), (k, l, m)) == piecewise_fold(
        Piecewise((1, And(l <= i, i <= m)), (0, True)) +
        Piecewise((1, And(l <= j, j <= m)), (0, True)))


def test_deltasummation_add_mul_x_kd_kd():
    assert ds(x*KD(i, k) + KD(j, k), (k, 1, 3)) == piecewise_fold(
        Piecewise((x, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((1, And(1 <= j, j <= 3)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, 1, 1)) == piecewise_fold(
        Piecewise((x, Eq(i, 1)), (0, True)) +
        Piecewise((1, Eq(j, 1)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, 2, 2)) == piecewise_fold(
        Piecewise((x, Eq(i, 2)), (0, True)) +
        Piecewise((1, Eq(j, 2)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, 3, 3)) == piecewise_fold(
        Piecewise((x, Eq(i, 3)), (0, True)) +
        Piecewise((1, Eq(j, 3)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, 1, l)) == piecewise_fold(
        Piecewise((x, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((1, And(1 <= j, j <= l)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, l, 3)) == piecewise_fold(
        Piecewise((x, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((1, And(l <= j, j <= 3)), (0, True)))
    assert ds(x*KD(i, k) + KD(j, k), (k, l, m)) == piecewise_fold(
        Piecewise((x, And(l <= i, i <= m)), (0, True)) +
        Piecewise((1, And(l <= j, j <= m)), (0, True)))


def test_deltasummation_mul_x_add_kd_kd():
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 1, 3)) == piecewise_fold(
        Piecewise((x, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((x, And(1 <= j, j <= 3)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 1, 1)) == piecewise_fold(
        Piecewise((x, Eq(i, 1)), (0, True)) +
        Piecewise((x, Eq(j, 1)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 2, 2)) == piecewise_fold(
        Piecewise((x, Eq(i, 2)), (0, True)) +
        Piecewise((x, Eq(j, 2)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 3, 3)) == piecewise_fold(
        Piecewise((x, Eq(i, 3)), (0, True)) +
        Piecewise((x, Eq(j, 3)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 1, l)) == piecewise_fold(
        Piecewise((x, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((x, And(1 <= j, j <= l)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, l, 3)) == piecewise_fold(
        Piecewise((x, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((x, And(l <= j, j <= 3)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, l, m)) == piecewise_fold(
        Piecewise((x, And(l <= i, i <= m)), (0, True)) +
        Piecewise((x, And(l <= j, j <= m)), (0, True)))


def test_deltasummation_mul_add_x_y_add_kd_kd():
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 3)) == piecewise_fold(
        Piecewise((x + y, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((x + y, And(1 <= j, j <= 3)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 1)) == piecewise_fold(
        Piecewise((x + y, Eq(i, 1)), (0, True)) +
        Piecewise((x + y, Eq(j, 1)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 2, 2)) == piecewise_fold(
        Piecewise((x + y, Eq(i, 2)), (0, True)) +
        Piecewise((x + y, Eq(j, 2)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 3, 3)) == piecewise_fold(
        Piecewise((x + y, Eq(i, 3)), (0, True)) +
        Piecewise((x + y, Eq(j, 3)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 1, l)) == piecewise_fold(
        Piecewise((x + y, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((x + y, And(1 <= j, j <= l)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, l, 3)) == piecewise_fold(
        Piecewise((x + y, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((x + y, And(l <= j, j <= 3)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, l, m)) == piecewise_fold(
        Piecewise((x + y, And(l <= i, i <= m)), (0, True)) +
        Piecewise((x + y, And(l <= j, j <= m)), (0, True)))


def test_deltasummation_add_mul_x_y_mul_x_kd():
    assert ds(x*y + x*KD(i, j), (j, 1, 3)) == \
        Piecewise((3*x*y + x, And(1 <= i, i <= 3)), (3*x*y, True))
    assert ds(x*y + x*KD(i, j), (j, 1, 1)) == \
        Piecewise((x*y + x, Eq(i, 1)), (x*y, True))
    assert ds(x*y + x*KD(i, j), (j, 2, 2)) == \
        Piecewise((x*y + x, Eq(i, 2)), (x*y, True))
    assert ds(x*y + x*KD(i, j), (j, 3, 3)) == \
        Piecewise((x*y + x, Eq(i, 3)), (x*y, True))
    assert ds(x*y + x*KD(i, j), (j, 1, k)) == \
        Piecewise((k*x*y + x, And(1 <= i, i <= k)), (k*x*y, True))
    assert ds(x*y + x*KD(i, j), (j, k, 3)) == \
        Piecewise(((4 - k)*x*y + x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    assert ds(x*y + x*KD(i, j), (j, k, l)) == Piecewise(
        ((l - k + 1)*x*y + x, And(k <= i, i <= l)), ((l - k + 1)*x*y, True))


def test_deltasummation_mul_x_add_y_kd():
    assert ds(x*(y + KD(i, j)), (j, 1, 3)) == \
        Piecewise((3*x*y + x, And(1 <= i, i <= 3)), (3*x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 1, 1)) == \
        Piecewise((x*y + x, Eq(i, 1)), (x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 2, 2)) == \
        Piecewise((x*y + x, Eq(i, 2)), (x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 3, 3)) == \
        Piecewise((x*y + x, Eq(i, 3)), (x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 1, k)) == \
        Piecewise((k*x*y + x, And(1 <= i, i <= k)), (k*x*y, True))
    assert ds(x*(y + KD(i, j)), (j, k, 3)) == \
        Piecewise(((4 - k)*x*y + x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    assert ds(x*(y + KD(i, j)), (j, k, l)) == Piecewise(
        ((l - k + 1)*x*y + x, And(k <= i, i <= l)), ((l - k + 1)*x*y, True))


def test_deltasummation_mul_x_add_y_twokd():
    assert ds(x*(y + 2*KD(i, j)), (j, 1, 3)) == \
        Piecewise((3*x*y + 2*x, And(1 <= i, i <= 3)), (3*x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 1, 1)) == \
        Piecewise((x*y + 2*x, Eq(i, 1)), (x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 2, 2)) == \
        Piecewise((x*y + 2*x, Eq(i, 2)), (x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 3, 3)) == \
        Piecewise((x*y + 2*x, Eq(i, 3)), (x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 1, k)) == \
        Piecewise((k*x*y + 2*x, And(1 <= i, i <= k)), (k*x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, k, 3)) == Piecewise(
        ((4 - k)*x*y + 2*x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, k, l)) == Piecewise(
        ((l - k + 1)*x*y + 2*x, And(k <= i, i <= l)), ((l - k + 1)*x*y, True))


def test_deltasummation_mul_add_x_y_add_y_kd():
    assert ds((x + y)*(y + KD(i, j)), (j, 1, 3)) == Piecewise(
        (3*(x + y)*y + x + y, And(1 <= i, i <= 3)), (3*(x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, 1, 1)) == \
        Piecewise(((x + y)*y + x + y, Eq(i, 1)), ((x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, 2, 2)) == \
        Piecewise(((x + y)*y + x + y, Eq(i, 2)), ((x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, 3, 3)) == \
        Piecewise(((x + y)*y + x + y, Eq(i, 3)), ((x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, 1, k)) == Piecewise(
        (k*(x + y)*y + x + y, And(1 <= i, i <= k)), (k*(x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, k, 3)) == Piecewise(
        ((4 - k)*(x + y)*y + x + y, And(k <= i, i <= 3)),
        ((4 - k)*(x + y)*y, True))
    assert ds((x + y)*(y + KD(i, j)), (j, k, l)) == Piecewise(
        ((l - k + 1)*(x + y)*y + x + y, And(k <= i, i <= l)),
        ((l - k + 1)*(x + y)*y, True))


def test_deltasummation_mul_add_x_kd_add_y_kd():
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 1, 3)) == piecewise_fold(
        Piecewise((KD(i, k) + x, And(1 <= i, i <= 3)), (0, True)) +
        3*(KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 1, 1)) == piecewise_fold(
        Piecewise((KD(i, k) + x, Eq(i, 1)), (0, True)) +
        (KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 2, 2)) == piecewise_fold(
        Piecewise((KD(i, k) + x, Eq(i, 2)), (0, True)) +
        (KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 3, 3)) == piecewise_fold(
        Piecewise((KD(i, k) + x, Eq(i, 3)), (0, True)) +
        (KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 1, k)) == piecewise_fold(
        Piecewise((KD(i, k) + x, And(1 <= i, i <= k)), (0, True)) +
        k*(KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, k, 3)) == piecewise_fold(
        Piecewise((KD(i, k) + x, And(k <= i, i <= 3)), (0, True)) +
        (4 - k)*(KD(i, k) + x)*y)
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, k, l)) == piecewise_fold(
        Piecewise((KD(i, k) + x, And(k <= i, i <= l)), (0, True)) +
        (l - k + 1)*(KD(i, k) + x)*y)


def test_extract_delta():
    raises(ValueError, lambda: _extract_delta(KD(i, j) + KD(k, l), i))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_radsimp.py ===
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import factor
from sympy.series.order import O
from sympy.simplify.radsimp import (collect, collect_const, fraction, radsimp, rcollect)

from sympy.core.expr import unchanged
from sympy.core.mul import _unevaluated_Mul as umul
from sympy.simplify.radsimp import (_unevaluated_Add,
    collect_sqrt, fraction_expand, collect_abs)
from sympy.testing.pytest import raises

from sympy.abc import x, y, z, a, b, c, d


def test_radsimp():
    r2 = sqrt(2)
    r3 = sqrt(3)
    r5 = sqrt(5)
    r7 = sqrt(7)
    assert fraction(radsimp(1/r2)) == (sqrt(2), 2)
    assert radsimp(1/(1 + r2)) == \
        -1 + sqrt(2)
    assert radsimp(1/(r2 + r3)) == \
        -sqrt(2) + sqrt(3)
    assert fraction(radsimp(1/(1 + r2 + r3))) == \
        (-sqrt(6) + sqrt(2) + 2, 4)
    assert fraction(radsimp(1/(r2 + r3 + r5))) == \
        (-sqrt(30) + 2*sqrt(3) + 3*sqrt(2), 12)
    assert fraction(radsimp(1/(1 + r2 + r3 + r5))) == (
        (-34*sqrt(10) - 26*sqrt(15) - 55*sqrt(3) - 61*sqrt(2) + 14*sqrt(30) +
        93 + 46*sqrt(6) + 53*sqrt(5), 71))
    assert fraction(radsimp(1/(r2 + r3 + r5 + r7))) == (
        (-50*sqrt(42) - 133*sqrt(5) - 34*sqrt(70) - 145*sqrt(3) + 22*sqrt(105)
        + 185*sqrt(2) + 62*sqrt(30) + 135*sqrt(7), 215))
    z = radsimp(1/(1 + r2/3 + r3/5 + r5 + r7))
    assert len((3616791619821680643598*z).args) == 16
    assert radsimp(1/z) == 1/z
    assert radsimp(1/z, max_terms=20).expand() == 1 + r2/3 + r3/5 + r5 + r7
    assert radsimp(1/(r2*3)) == \
        sqrt(2)/6
    assert radsimp(1/(r2*a + r3 + r5 + r7)) == (
        (8*sqrt(2)*a**7 - 8*sqrt(7)*a**6 - 8*sqrt(5)*a**6 - 8*sqrt(3)*a**6 -
        180*sqrt(2)*a**5 + 8*sqrt(30)*a**5 + 8*sqrt(42)*a**5 + 8*sqrt(70)*a**5
        - 24*sqrt(105)*a**4 + 84*sqrt(3)*a**4 + 100*sqrt(5)*a**4 +
        116*sqrt(7)*a**4 - 72*sqrt(70)*a**3 - 40*sqrt(42)*a**3 -
        8*sqrt(30)*a**3 + 782*sqrt(2)*a**3 - 462*sqrt(3)*a**2 -
        302*sqrt(7)*a**2 - 254*sqrt(5)*a**2 + 120*sqrt(105)*a**2 -
        795*sqrt(2)*a - 62*sqrt(30)*a + 82*sqrt(42)*a + 98*sqrt(70)*a -
        118*sqrt(105) + 59*sqrt(7) + 295*sqrt(5) + 531*sqrt(3))/(16*a**8 -
        480*a**6 + 3128*a**4 - 6360*a**2 + 3481))
    assert radsimp(1/(r2*a + r2*b + r3 + r7)) == (
        (sqrt(2)*a*(a + b)**2 - 5*sqrt(2)*a + sqrt(42)*a + sqrt(2)*b*(a +
        b)**2 - 5*sqrt(2)*b + sqrt(42)*b - sqrt(7)*(a + b)**2 - sqrt(3)*(a +
        b)**2 - 2*sqrt(3) + 2*sqrt(7))/(2*a**4 + 8*a**3*b + 12*a**2*b**2 -
        20*a**2 + 8*a*b**3 - 40*a*b + 2*b**4 - 20*b**2 + 8))
    assert radsimp(1/(r2*a + r2*b + r2*c + r2*d)) == \
        sqrt(2)/(2*a + 2*b + 2*c + 2*d)
    assert radsimp(1/(1 + r2*a + r2*b + r2*c + r2*d)) == (
        (sqrt(2)*a + sqrt(2)*b + sqrt(2)*c + sqrt(2)*d - 1)/(2*a**2 + 4*a*b +
        4*a*c + 4*a*d + 2*b**2 + 4*b*c + 4*b*d + 2*c**2 + 4*c*d + 2*d**2 - 1))
    assert radsimp((y**2 - x)/(y - sqrt(x))) == \
        sqrt(x) + y
    assert radsimp(-(y**2 - x)/(y - sqrt(x))) == \
        -(sqrt(x) + y)
    assert radsimp(1/(1 - I + a*I)) == \
        (-I*a + 1 + I)/(a**2 - 2*a + 2)
    assert radsimp(1/((-x + y)*(x - sqrt(y)))) == \
        (-x - sqrt(y))/((x - y)*(x**2 - y))
    e = (3 + 3*sqrt(2))*x*(3*x - 3*sqrt(y))
    assert radsimp(e) == x*(3 + 3*sqrt(2))*(3*x - 3*sqrt(y))
    assert radsimp(1/e) == (
        (-9*x + 9*sqrt(2)*x - 9*sqrt(y) + 9*sqrt(2)*sqrt(y))/(9*x*(9*x**2 -
        9*y)))
    assert radsimp(1 + 1/(1 + sqrt(3))) == \
        Mul(S.Half, -1 + sqrt(3), evaluate=False) + 1
    A = symbols("A", commutative=False)
    assert radsimp(x**2 + sqrt(2)*x**2 - sqrt(2)*x*A) == \
        x**2 + sqrt(2)*x**2 - sqrt(2)*x*A
    assert radsimp(1/sqrt(5 + 2 * sqrt(6))) == -sqrt(2) + sqrt(3)
    assert radsimp(1/sqrt(5 + 2 * sqrt(6))**3) == -(-sqrt(3) + sqrt(2))**3

    # issue 6532
    assert fraction(radsimp(1/sqrt(x))) == (sqrt(x), x)
    assert fraction(radsimp(1/sqrt(2*x + 3))) == (sqrt(2*x + 3), 2*x + 3)
    assert fraction(radsimp(1/sqrt(2*(x + 3)))) == (sqrt(2*x + 6), 2*x + 6)

    # issue 5994
    e = S('-(2 + 2*sqrt(2) + 4*2**(1/4))/'
        '(1 + 2**(3/4) + 3*2**(1/4) + 3*sqrt(2))')
    assert radsimp(e).expand() == -2*2**Rational(3, 4) - 2*2**Rational(1, 4) + 2 + 2*sqrt(2)

    # issue 5986 (modifications to radimp didn't initially recognize this so
    # the test is included here)
    assert radsimp(1/(-sqrt(5)/2 - S.Half + (-sqrt(5)/2 - S.Half)**2)) == 1

    # from issue 5934
    eq = (
        (-240*sqrt(2)*sqrt(sqrt(5) + 5)*sqrt(8*sqrt(5) + 40) -
        360*sqrt(2)*sqrt(-8*sqrt(5) + 40)*sqrt(-sqrt(5) + 5) -
        120*sqrt(10)*sqrt(-8*sqrt(5) + 40)*sqrt(-sqrt(5) + 5) +
        120*sqrt(2)*sqrt(-sqrt(5) + 5)*sqrt(8*sqrt(5) + 40) +
        120*sqrt(2)*sqrt(-8*sqrt(5) + 40)*sqrt(sqrt(5) + 5) +
        120*sqrt(10)*sqrt(-sqrt(5) + 5)*sqrt(8*sqrt(5) + 40) +
        120*sqrt(10)*sqrt(-8*sqrt(5) + 40)*sqrt(sqrt(5) + 5))/(-36000 -
        7200*sqrt(5) + (12*sqrt(10)*sqrt(sqrt(5) + 5) +
        24*sqrt(10)*sqrt(-sqrt(5) + 5))**2))
    assert radsimp(eq) is S.NaN  # it's 0/0

    # work with normal form
    e = 1/sqrt(sqrt(7)/7 + 2*sqrt(2) + 3*sqrt(3) + 5*sqrt(5)) + 3
    assert radsimp(e) == (
        -sqrt(sqrt(7) + 14*sqrt(2) + 21*sqrt(3) +
        35*sqrt(5))*(-11654899*sqrt(35) - 1577436*sqrt(210) - 1278438*sqrt(15)
        - 1346996*sqrt(10) + 1635060*sqrt(6) + 5709765 + 7539830*sqrt(14) +
        8291415*sqrt(21))/1300423175 + 3)

    # obey power rules
    base = sqrt(3) - sqrt(2)
    assert radsimp(1/base**3) == (sqrt(3) + sqrt(2))**3
    assert radsimp(1/(-base)**3) == -(sqrt(2) + sqrt(3))**3
    assert radsimp(1/(-base)**x) == (-base)**(-x)
    assert radsimp(1/base**x) == (sqrt(2) + sqrt(3))**x
    assert radsimp(root(1/(-1 - sqrt(2)), -x)) == (-1)**(-1/x)*(1 + sqrt(2))**(1/x)

    # recurse
    e = cos(1/(1 + sqrt(2)))
    assert radsimp(e) == cos(-sqrt(2) + 1)
    assert radsimp(e/2) == cos(-sqrt(2) + 1)/2
    assert radsimp(1/e) == 1/cos(-sqrt(2) + 1)
    assert radsimp(2/e) == 2/cos(-sqrt(2) + 1)
    assert fraction(radsimp(e/sqrt(x))) == (sqrt(x)*cos(-sqrt(2)+1), x)

    # test that symbolic denominators are not processed
    r = 1 + sqrt(2)
    assert radsimp(x/r, symbolic=False) == -x*(-sqrt(2) + 1)
    assert radsimp(x/(y + r), symbolic=False) == x/(y + 1 + sqrt(2))
    assert radsimp(x/(y + r)/r, symbolic=False) == \
        -x*(-sqrt(2) + 1)/(y + 1 + sqrt(2))

    # issue 7408
    eq = sqrt(x)/sqrt(y)
    assert radsimp(eq) == umul(sqrt(x), sqrt(y), 1/y)
    assert radsimp(eq, symbolic=False) == eq

    # issue 7498
    assert radsimp(sqrt(x)/sqrt(y)**3) == umul(sqrt(x), sqrt(y**3), 1/y**3)

    # for coverage
    eq = sqrt(x)/y**2
    assert radsimp(eq) == eq

    # handle non-Expr args
    from sympy.integrals.integrals import Integral
    eq = Integral(x/(sqrt(2) - 1), (x, 0, 1/(sqrt(2) + 1)))
    assert radsimp(eq) == Integral((sqrt(2) + 1)*x , (x, 0, sqrt(2) - 1))

    from sympy.sets import FiniteSet
    eq = FiniteSet(x/(sqrt(2) - 1))
    assert radsimp(eq) == FiniteSet((sqrt(2) + 1)*x)

def test_radsimp_issue_3214():
    c, p = symbols('c p', positive=True)
    s = sqrt(c**2 - p**2)
    b = (c + I*p - s)/(c + I*p + s)
    assert radsimp(b) == -I*(c + I*p - sqrt(c**2 - p**2))**2/(2*c*p)


def test_collect_1():
    """Collect with respect to Symbol"""
    x, y, z, n = symbols('x,y,z,n')
    assert collect(1, x) == 1
    assert collect( x + y*x, x ) == x * (1 + y)
    assert collect( x + x**2, x ) == x + x**2
    assert collect( x**2 + y*x**2, x ) == (x**2)*(1 + y)
    assert collect( x**2 + y*x, x ) == x*y + x**2
    assert collect( 2*x**2 + y*x**2 + 3*x*y, [x] ) == x**2*(2 + y) + 3*x*y
    assert collect( 2*x**2 + y*x**2 + 3*x*y, [y] ) == 2*x**2 + y*(x**2 + 3*x)

    assert collect( ((1 + y + x)**4).expand(), x) == ((1 + y)**4).expand() + \
        x*(4*(1 + y)**3).expand() + x**2*(6*(1 + y)**2).expand() + \
        x**3*(4*(1 + y)).expand() + x**4
    # symbols can be given as any iterable
    expr = x + y
    assert collect(expr, expr.free_symbols) == expr
    assert collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x, x, exact=None
        ) == x*exp(x) + 3*x + (y + 2)*sin(x)
    assert collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x + y*x +
        y*x*exp(x), x, exact=None
        ) == x*exp(x)*(y + 1) + (3 + y)*x + (y + 2)*sin(x)


def test_collect_2():
    """Collect with respect to a sum"""
    a, b, x = symbols('a,b,x')
    assert collect(a*(cos(x) + sin(x)) + b*(cos(x) + sin(x)),
        sin(x) + cos(x)) == (a + b)*(cos(x) + sin(x))


def test_collect_3():
    """Collect with respect to a product"""
    a, b, c = symbols('a,b,c')
    f = Function('f')
    x, y, z, n = symbols('x,y,z,n')

    assert collect(-x/8 + x*y, -x) == x*(y - Rational(1, 8))

    assert collect( 1 + x*(y**2), x*y ) == 1 + x*(y**2)
    assert collect( x*y + a*x*y, x*y) == x*y*(1 + a)
    assert collect( 1 + x*y + a*x*y, x*y) == 1 + x*y*(1 + a)
    assert collect(a*x*f(x) + b*(x*f(x)), x*f(x)) == x*(a + b)*f(x)

    assert collect(a*x*log(x) + b*(x*log(x)), x*log(x)) == x*(a + b)*log(x)
    assert collect(a*x**2*log(x)**2 + b*(x*log(x))**2, x*log(x)) == \
        x**2*log(x)**2*(a + b)

    # with respect to a product of three symbols
    assert collect(y*x*z + a*x*y*z, x*y*z) == (1 + a)*x*y*z


def test_collect_4():
    """Collect with respect to a power"""
    a, b, c, x = symbols('a,b,c,x')

    assert collect(a*x**c + b*x**c, x**c) == x**c*(a + b)
    # issue 6096: 2 stays with c (unless c is integer or x is positive0
    assert collect(a*x**(2*c) + b*x**(2*c), x**c) == x**(2*c)*(a + b)


def test_collect_5():
    """Collect with respect to a tuple"""
    a, x, y, z, n = symbols('a,x,y,z,n')
    assert collect(x**2*y**4 + z*(x*y**2)**2 + z + a*z, [x*y**2, z]) in [
        z*(1 + a + x**2*y**4) + x**2*y**4,
        z*(1 + a) + x**2*y**4*(1 + z) ]
    assert collect((1 + (x + y) + (x + y)**2).expand(),
                   [x, y]) == 1 + y + x*(1 + 2*y) + x**2 + y**2


def test_collect_pr19431():
    """Unevaluated collect with respect to a product"""
    a = symbols('a')
    assert collect(a**2*(a**2 + 1), a**2, evaluate=False)[a**2] == (a**2 + 1)


def test_collect_D():
    D = Derivative
    f = Function('f')
    x, a, b = symbols('x,a,b')
    fx = D(f(x), x)
    fxx = D(f(x), x, x)

    assert collect(a*fx + b*fx, fx) == (a + b)*fx
    assert collect(a*D(fx, x) + b*D(fx, x), fx) == (a + b)*D(fx, x)
    assert collect(a*fxx + b*fxx, fx) == (a + b)*D(fx, x)
    # issue 4784
    assert collect(5*f(x) + 3*fx, fx) == 5*f(x) + 3*fx
    assert collect(f(x) + f(x)*diff(f(x), x) + x*diff(f(x), x)*f(x), f(x).diff(x)) == \
        (x*f(x) + f(x))*D(f(x), x) + f(x)
    assert collect(f(x) + f(x)*diff(f(x), x) + x*diff(f(x), x)*f(x), f(x).diff(x), exact=True) == \
        (x*f(x) + f(x))*D(f(x), x) + f(x)
    assert collect(1/f(x) + 1/f(x)*diff(f(x), x) + x*diff(f(x), x)/f(x), f(x).diff(x), exact=True) == \
        (1/f(x) + x/f(x))*D(f(x), x) + 1/f(x)
    e = (1 + x*fx + fx)/f(x)
    assert collect(e.expand(), fx) == fx*(x/f(x) + 1/f(x)) + 1/f(x)


def test_collect_func():
    f = ((x + a + 1)**3).expand()

    assert collect(f, x) == a**3 + 3*a**2 + 3*a + x**3 + x**2*(3*a + 3) + \
        x*(3*a**2 + 6*a + 3) + 1
    assert collect(f, x, factor) == x**3 + 3*x**2*(a + 1) + 3*x*(a + 1)**2 + \
        (a + 1)**3

    assert collect(f, x, evaluate=False) == {
        S.One: a**3 + 3*a**2 + 3*a + 1,
        x: 3*a**2 + 6*a + 3, x**2: 3*a + 3,
        x**3: 1
    }

    assert collect(f, x, factor, evaluate=False) == {
        S.One: (a + 1)**3, x: 3*(a + 1)**2,
        x**2: umul(S(3), a + 1), x**3: 1}


def test_collect_order():
    a, b, x, t = symbols('a,b,x,t')

    assert collect(t + t*x + t*x**2 + O(x**3), t) == t*(1 + x + x**2 + O(x**3))
    assert collect(t + t*x + x**2 + O(x**3), t) == \
        t*(1 + x + O(x**3)) + x**2 + O(x**3)

    f = a*x + b*x + c*x**2 + d*x**2 + O(x**3)
    g = x*(a + b) + x**2*(c + d) + O(x**3)

    assert collect(f, x) == g
    assert collect(f, x, distribute_order_term=False) == g

    f = sin(a + b).series(b, 0, 10)

    assert collect(f, [sin(a), cos(a)]) == \
        sin(a)*cos(b).series(b, 0, 10) + cos(a)*sin(b).series(b, 0, 10)
    assert collect(f, [sin(a), cos(a)], distribute_order_term=False) == \
        sin(a)*cos(b).series(b, 0, 10).removeO() + \
        cos(a)*sin(b).series(b, 0, 10).removeO() + O(b**10)


def test_rcollect():
    assert rcollect((x**2*y + x*y + x + y)/(x + y), y) == \
        (x + y*(1 + x + x**2))/(x + y)
    assert rcollect(sqrt(-((x + 1)*(y + 1))), z) == sqrt(-((x + 1)*(y + 1)))


def test_collect_D_0():
    D = Derivative
    f = Function('f')
    x, a, b = symbols('x,a,b')
    fxx = D(f(x), x, x)

    assert collect(a*fxx + b*fxx, fxx) == (a + b)*fxx


def test_collect_Wild():
    """Collect with respect to functions with Wild argument"""
    a, b, x, y = symbols('a b x y')
    f = Function('f')
    w1 = Wild('.1')
    w2 = Wild('.2')
    assert collect(f(x) + a*f(x), f(w1)) == (1 + a)*f(x)
    assert collect(f(x, y) + a*f(x, y), f(w1)) == f(x, y) + a*f(x, y)
    assert collect(f(x, y) + a*f(x, y), f(w1, w2)) == (1 + a)*f(x, y)
    assert collect(f(x, y) + a*f(x, y), f(w1, w1)) == f(x, y) + a*f(x, y)
    assert collect(f(x, x) + a*f(x, x), f(w1, w1)) == (1 + a)*f(x, x)
    assert collect(a*(x + 1)**y + (x + 1)**y, w1**y) == (1 + a)*(x + 1)**y
    assert collect(a*(x + 1)**y + (x + 1)**y, w1**b) == \
        a*(x + 1)**y + (x + 1)**y
    assert collect(a*(x + 1)**y + (x + 1)**y, (x + 1)**w2) == \
        (1 + a)*(x + 1)**y
    assert collect(a*(x + 1)**y + (x + 1)**y, w1**w2) == (1 + a)*(x + 1)**y


def test_collect_const():
    # coverage not provided by above tests
    assert collect_const(2*sqrt(3) + 4*a*sqrt(5)) == \
        2*(2*sqrt(5)*a + sqrt(3))  # let the primitive reabsorb
    assert collect_const(2*sqrt(3) + 4*a*sqrt(5), sqrt(3)) == \
        2*sqrt(3) + 4*a*sqrt(5)
    assert collect_const(sqrt(2)*(1 + sqrt(2)) + sqrt(3) + x*sqrt(2)) == \
        sqrt(2)*(x + 1 + sqrt(2)) + sqrt(3)

    # issue 5290
    assert collect_const(2*x + 2*y + 1, 2) == \
        collect_const(2*x + 2*y + 1) == \
        Add(S.One, Mul(2, x + y, evaluate=False), evaluate=False)
    assert collect_const(-y - z) == Mul(-1, y + z, evaluate=False)
    assert collect_const(2*x - 2*y - 2*z, 2) == \
        Mul(2, x - y - z, evaluate=False)
    assert collect_const(2*x - 2*y - 2*z, -2) == \
        _unevaluated_Add(2*x, Mul(-2, y + z, evaluate=False))

    # this is why the content_primitive is used
    eq = (sqrt(15 + 5*sqrt(2))*x + sqrt(3 + sqrt(2))*y)*2
    assert collect_sqrt(eq + 2) == \
        2*sqrt(sqrt(2) + 3)*(sqrt(5)*x + y) + 2

    # issue 16296
    assert collect_const(a + b + x/2 + y/2) == a + b + Mul(S.Half, x + y, evaluate=False)


def test_issue_13143():
    f = Function('f')
    fx = f(x).diff(x)
    e = f(x) + fx + f(x)*fx
    # collect function before derivative
    assert collect(e, Wild('w')) == f(x)*(fx + 1) + fx
    e = f(x) + f(x)*fx + x*fx*f(x)
    assert collect(e, fx) == (x*f(x) + f(x))*fx + f(x)
    assert collect(e, f(x)) == (x*fx + fx + 1)*f(x)
    e = f(x) + fx + f(x)*fx
    assert collect(e, [f(x), fx]) == f(x)*(1 + fx) + fx
    assert collect(e, [fx, f(x)]) == fx*(1 + f(x)) + f(x)


def test_issue_6097():
    assert collect(a*y**(2.0*x) + b*y**(2.0*x), y**x) == (a + b)*(y**x)**2.0
    assert collect(a*2**(2.0*x) + b*2**(2.0*x), 2**x) == (a + b)*(2**x)**2.0


def test_fraction_expand():
    eq = (x + y)*y/x
    assert eq.expand(frac=True) == fraction_expand(eq) == (x*y + y**2)/x
    assert eq.expand() == y + y**2/x


def test_fraction():
    x, y, z = map(Symbol, 'xyz')
    A = Symbol('A', commutative=False)

    assert fraction(S.Half) == (1, 2)

    assert fraction(x) == (x, 1)
    assert fraction(1/x) == (1, x)
    assert fraction(x/y) == (x, y)
    assert fraction(x/2) == (x, 2)

    assert fraction(x*y/z) == (x*y, z)
    assert fraction(x/(y*z)) == (x, y*z)

    assert fraction(1/y**2) == (1, y**2)
    assert fraction(x/y**2) == (x, y**2)

    assert fraction((x**2 + 1)/y) == (x**2 + 1, y)
    assert fraction(x*(y + 1)/y**7) == (x*(y + 1), y**7)

    assert fraction(exp(-x), exact=True) == (exp(-x), 1)
    assert fraction((1/(x + y))/2, exact=True) == (1, Mul(2,(x + y), evaluate=False))

    assert fraction(x*A/y) == (x*A, y)
    assert fraction(x*A**-1/y) == (x*A**-1, y)

    n = symbols('n', negative=True)
    assert fraction(exp(n)) == (1, exp(-n))
    assert fraction(exp(-n)) == (exp(-n), 1)

    p = symbols('p', positive=True)
    assert fraction(exp(-p)*log(p), exact=True) == (exp(-p)*log(p), 1)

    m = Mul(1, 1, S.Half, evaluate=False)
    assert fraction(m) == (1, 2)
    assert fraction(m, exact=True) == (Mul(1, 1, evaluate=False), 2)

    m = Mul(1, 1, S.Half, S.Half, Pow(1, -1, evaluate=False), evaluate=False)
    assert fraction(m) == (1, 4)
    assert fraction(m, exact=True) == \
            (Mul(1, 1, evaluate=False), Mul(2, 2, 1, evaluate=False))


def test_issue_5615():
    aA, Re, a, b, D = symbols('aA Re a b D')
    e = ((D**3*a + b*aA**3)/Re).expand()
    assert collect(e, [aA**3/Re, a]) == e


def test_issue_5933():
    from sympy.geometry.polygon import (Polygon, RegularPolygon)
    from sympy.simplify.radsimp import denom
    x = Polygon(*RegularPolygon((0, 0), 1, 5).vertices).centroid.x
    assert abs(denom(x).n()) > 1e-12
    assert abs(denom(radsimp(x))) > 1e-12  # in case simplify didn't handle it


def test_issue_14608():
    a, b = symbols('a b', commutative=False)
    x, y = symbols('x y')
    raises(AttributeError, lambda: collect(a*b + b*a, a))
    assert collect(x*y + y*(x+1), a) == x*y + y*(x+1)
    assert collect(x*y + y*(x+1) + a*b + b*a, y) == y*(2*x + 1) + a*b + b*a


def test_collect_abs():
    s = abs(x) + abs(y)
    assert collect_abs(s) == s
    assert unchanged(Mul, abs(x), abs(y))
    ans = Abs(x*y)
    assert isinstance(ans, Abs)
    assert collect_abs(abs(x)*abs(y)) == ans
    assert collect_abs(1 + exp(abs(x)*abs(y))) == 1 + exp(ans)

    # See https://github.com/sympy/sympy/issues/12910
    p = Symbol('p', positive=True)
    assert collect_abs(p/abs(1-p)).is_commutative is True


def test_issue_19149():
    eq = exp(3*x/4)
    assert collect(eq, exp(x)) == eq

def test_issue_19719():
    a, b = symbols('a, b')
    expr = a**2 * (b + 1) + (7 + 1/b)/a
    collected = collect(expr, (a**2, 1/a), evaluate=False)
    # Would return {_Dummy_20**(-2): b + 1, 1/a: 7 + 1/b} without xreplace
    assert collected == {a**2: b + 1, 1/a: 7 + 1/b}


def test_issue_21355():
    assert radsimp(1/(x + sqrt(x**2))) == 1/(x + sqrt(x**2))
    assert radsimp(1/(x - sqrt(x**2))) == 1/(x - sqrt(x**2))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\tests\test_mrecords.py ===
"""Tests suite for mrecords.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu

"""
import pickle

import numpy as np
import numpy.ma as ma
from numpy._core.records import fromarrays as recfromarrays
from numpy._core.records import fromrecords as recfromrecords
from numpy._core.records import recarray
from numpy.ma import masked, nomask
from numpy.ma.mrecords import (
    MaskedRecords,
    addfield,
    fromarrays,
    fromrecords,
    fromtextfile,
    mrecarray,
)
from numpy.ma.testutils import (
    assert_,
    assert_equal,
    assert_equal_records,
)
from numpy.testing import temppath


class TestMRecords:

    ilist = [1, 2, 3, 4, 5]
    flist = [1.1, 2.2, 3.3, 4.4, 5.5]
    slist = [b'one', b'two', b'three', b'four', b'five']
    ddtype = [('a', int), ('b', float), ('c', '|S8')]
    mask = [0, 1, 0, 0, 1]
    base = ma.array(list(zip(ilist, flist, slist)), mask=mask, dtype=ddtype)

    def test_byview(self):
        # Test creation by view
        base = self.base
        mbase = base.view(mrecarray)
        assert_equal(mbase.recordmask, base.recordmask)
        assert_equal_records(mbase._mask, base._mask)
        assert_(isinstance(mbase._data, recarray))
        assert_equal_records(mbase._data, base._data.view(recarray))
        for field in ('a', 'b', 'c'):
            assert_equal(base[field], mbase[field])
        assert_equal_records(mbase.view(mrecarray), mbase)

    def test_get(self):
        # Tests fields retrieval
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # As fields..........
        for field in ('a', 'b', 'c'):
            assert_equal(getattr(mbase, field), mbase[field])
            assert_equal(base[field], mbase[field])
        # as elements .......
        mbase_first = mbase[0]
        assert_(isinstance(mbase_first, mrecarray))
        assert_equal(mbase_first.dtype, mbase.dtype)
        assert_equal(mbase_first.tolist(), (1, 1.1, b'one'))
        # Used to be mask, now it's recordmask
        assert_equal(mbase_first.recordmask, nomask)
        assert_equal(mbase_first._mask.item(), (False, False, False))
        assert_equal(mbase_first['a'], mbase['a'][0])
        mbase_last = mbase[-1]
        assert_(isinstance(mbase_last, mrecarray))
        assert_equal(mbase_last.dtype, mbase.dtype)
        assert_equal(mbase_last.tolist(), (None, None, None))
        # Used to be mask, now it's recordmask
        assert_equal(mbase_last.recordmask, True)
        assert_equal(mbase_last._mask.item(), (True, True, True))
        assert_equal(mbase_last['a'], mbase['a'][-1])
        assert_(mbase_last['a'] is masked)
        # as slice ..........
        mbase_sl = mbase[:2]
        assert_(isinstance(mbase_sl, mrecarray))
        assert_equal(mbase_sl.dtype, mbase.dtype)
        # Used to be mask, now it's recordmask
        assert_equal(mbase_sl.recordmask, [0, 1])
        assert_equal_records(mbase_sl.mask,
                             np.array([(False, False, False),
                                       (True, True, True)],
                                      dtype=mbase._mask.dtype))
        assert_equal_records(mbase_sl, base[:2].view(mrecarray))
        for field in ('a', 'b', 'c'):
            assert_equal(getattr(mbase_sl, field), base[:2][field])

    def test_set_fields(self):
        # Tests setting fields.
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase = mbase.copy()
        mbase.fill_value = (999999, 1e20, 'N/A')
        # Change the data, the mask should be conserved
        mbase.a._data[:] = 5
        assert_equal(mbase['a']._data, [5, 5, 5, 5, 5])
        assert_equal(mbase['a']._mask, [0, 1, 0, 0, 1])
        # Change the elements, and the mask will follow
        mbase.a = 1
        assert_equal(mbase['a']._data, [1] * 5)
        assert_equal(ma.getmaskarray(mbase['a']), [0] * 5)
        # Use to be _mask, now it's recordmask
        assert_equal(mbase.recordmask, [False] * 5)
        assert_equal(mbase._mask.tolist(),
                     np.array([(0, 0, 0),
                               (0, 1, 1),
                               (0, 0, 0),
                               (0, 0, 0),
                               (0, 1, 1)],
                              dtype=bool))
        # Set a field to mask ........................
        mbase.c = masked
        # Use to be mask, and now it's still mask !
        assert_equal(mbase.c.mask, [1] * 5)
        assert_equal(mbase.c.recordmask, [1] * 5)
        assert_equal(ma.getmaskarray(mbase['c']), [1] * 5)
        assert_equal(ma.getdata(mbase['c']), [b'N/A'] * 5)
        assert_equal(mbase._mask.tolist(),
                     np.array([(0, 0, 1),
                               (0, 1, 1),
                               (0, 0, 1),
                               (0, 0, 1),
                               (0, 1, 1)],
                              dtype=bool))
        # Set fields by slices .......................
        mbase = base.view(mrecarray).copy()
        mbase.a[3:] = 5
        assert_equal(mbase.a, [1, 2, 3, 5, 5])
        assert_equal(mbase.a._mask, [0, 1, 0, 0, 0])
        mbase.b[3:] = masked
        assert_equal(mbase.b, base['b'])
        assert_equal(mbase.b._mask, [0, 1, 0, 1, 1])
        # Set fields globally..........................
        ndtype = [('alpha', '|S1'), ('num', int)]
        data = ma.array([('a', 1), ('b', 2), ('c', 3)], dtype=ndtype)
        rdata = data.view(MaskedRecords)
        val = ma.array([10, 20, 30], mask=[1, 0, 0])

        rdata['num'] = val
        assert_equal(rdata.num, val)
        assert_equal(rdata.num.mask, [1, 0, 0])

    def test_set_fields_mask(self):
        # Tests setting the mask of a field.
        base = self.base.copy()
        # This one has already a mask....
        mbase = base.view(mrecarray)
        mbase['a'][-2] = masked
        assert_equal(mbase.a, [1, 2, 3, 4, 5])
        assert_equal(mbase.a._mask, [0, 1, 0, 1, 1])
        # This one has not yet
        mbase = fromarrays([np.arange(5), np.random.rand(5)],
                           dtype=[('a', int), ('b', float)])
        mbase['a'][-2] = masked
        assert_equal(mbase.a, [0, 1, 2, 3, 4])
        assert_equal(mbase.a._mask, [0, 0, 0, 1, 0])

    def test_set_mask(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # Set the mask to True .......................
        mbase.mask = masked
        assert_equal(ma.getmaskarray(mbase['b']), [1] * 5)
        assert_equal(mbase['a']._mask, mbase['b']._mask)
        assert_equal(mbase['a']._mask, mbase['c']._mask)
        assert_equal(mbase._mask.tolist(),
                     np.array([(1, 1, 1)] * 5, dtype=bool))
        # Delete the mask ............................
        mbase.mask = nomask
        assert_equal(ma.getmaskarray(mbase['c']), [0] * 5)
        assert_equal(mbase._mask.tolist(),
                     np.array([(0, 0, 0)] * 5, dtype=bool))

    def test_set_mask_fromarray(self):
        base = self.base.copy()
        mbase = base.view(mrecarray)
        # Sets the mask w/ an array
        mbase.mask = [1, 0, 0, 0, 1]
        assert_equal(mbase.a.mask, [1, 0, 0, 0, 1])
        assert_equal(mbase.b.mask, [1, 0, 0, 0, 1])
        assert_equal(mbase.c.mask, [1, 0, 0, 0, 1])
        # Yay, once more !
        mbase.mask = [0, 0, 0, 0, 1]
        assert_equal(mbase.a.mask, [0, 0, 0, 0, 1])
        assert_equal(mbase.b.mask, [0, 0, 0, 0, 1])
        assert_equal(mbase.c.mask, [0, 0, 0, 0, 1])

    def test_set_mask_fromfields(self):
        mbase = self.base.copy().view(mrecarray)

        nmask = np.array(
            [(0, 1, 0), (0, 1, 0), (1, 0, 1), (1, 0, 1), (0, 0, 0)],
            dtype=[('a', bool), ('b', bool), ('c', bool)])
        mbase.mask = nmask
        assert_equal(mbase.a.mask, [0, 0, 1, 1, 0])
        assert_equal(mbase.b.mask, [1, 1, 0, 0, 0])
        assert_equal(mbase.c.mask, [0, 0, 1, 1, 0])
        # Reinitialize and redo
        mbase.mask = False
        mbase.fieldmask = nmask
        assert_equal(mbase.a.mask, [0, 0, 1, 1, 0])
        assert_equal(mbase.b.mask, [1, 1, 0, 0, 0])
        assert_equal(mbase.c.mask, [0, 0, 1, 1, 0])

    def test_set_elements(self):
        base = self.base.copy()
        # Set an element to mask .....................
        mbase = base.view(mrecarray).copy()
        mbase[-2] = masked
        assert_equal(
            mbase._mask.tolist(),
            np.array([(0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1), (1, 1, 1)],
                     dtype=bool))
        # Used to be mask, now it's recordmask!
        assert_equal(mbase.recordmask, [0, 1, 0, 1, 1])
        # Set slices .................................
        mbase = base.view(mrecarray).copy()
        mbase[:2] = (5, 5, 5)
        assert_equal(mbase.a._data, [5, 5, 3, 4, 5])
        assert_equal(mbase.a._mask, [0, 0, 0, 0, 1])
        assert_equal(mbase.b._data, [5., 5., 3.3, 4.4, 5.5])
        assert_equal(mbase.b._mask, [0, 0, 0, 0, 1])
        assert_equal(mbase.c._data,
                     [b'5', b'5', b'three', b'four', b'five'])
        assert_equal(mbase.b._mask, [0, 0, 0, 0, 1])

        mbase = base.view(mrecarray).copy()
        mbase[:2] = masked
        assert_equal(mbase.a._data, [1, 2, 3, 4, 5])
        assert_equal(mbase.a._mask, [1, 1, 0, 0, 1])
        assert_equal(mbase.b._data, [1.1, 2.2, 3.3, 4.4, 5.5])
        assert_equal(mbase.b._mask, [1, 1, 0, 0, 1])
        assert_equal(mbase.c._data,
                     [b'one', b'two', b'three', b'four', b'five'])
        assert_equal(mbase.b._mask, [1, 1, 0, 0, 1])

    def test_setslices_hardmask(self):
        # Tests setting slices w/ hardmask.
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase.harden_mask()
        try:
            mbase[-2:] = (5, 5, 5)
            assert_equal(mbase.a._data, [1, 2, 3, 5, 5])
            assert_equal(mbase.b._data, [1.1, 2.2, 3.3, 5, 5.5])
            assert_equal(mbase.c._data,
                         [b'one', b'two', b'three', b'5', b'five'])
            assert_equal(mbase.a._mask, [0, 1, 0, 0, 1])
            assert_equal(mbase.b._mask, mbase.a._mask)
            assert_equal(mbase.b._mask, mbase.c._mask)
        except NotImplementedError:
            # OK, not implemented yet...
            pass
        except AssertionError:
            raise
        else:
            raise Exception("Flexible hard masks should be supported !")
        # Not using a tuple should crash
        try:
            mbase[-2:] = 3
        except (NotImplementedError, TypeError):
            pass
        else:
            raise TypeError("Should have expected a readable buffer object!")

    def test_hardmask(self):
        # Test hardmask
        base = self.base.copy()
        mbase = base.view(mrecarray)
        mbase.harden_mask()
        assert_(mbase._hardmask)
        mbase.mask = nomask
        assert_equal_records(mbase._mask, base._mask)
        mbase.soften_mask()
        assert_(not mbase._hardmask)
        mbase.mask = nomask
        # So, the mask of a field is no longer set to nomask...
        assert_equal_records(mbase._mask,
                             ma.make_mask_none(base.shape, base.dtype))
        assert_(ma.make_mask(mbase['b']._mask) is nomask)
        assert_equal(mbase['a']._mask, mbase['b']._mask)

    def test_pickling(self):
        # Test pickling
        base = self.base.copy()
        mrec = base.view(mrecarray)
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            _ = pickle.dumps(mrec, protocol=proto)
            mrec_ = pickle.loads(_)
            assert_equal(mrec_.dtype, mrec.dtype)
            assert_equal_records(mrec_._data, mrec._data)
            assert_equal(mrec_._mask, mrec._mask)
            assert_equal_records(mrec_._mask, mrec._mask)

    def test_filled(self):
        # Test filling the array
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[0, 0, 1], dtype='|S8')
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        mrec = fromarrays([_a, _b, _c], dtype=ddtype,
                          fill_value=(99999, 99999., 'N/A'))
        mrecfilled = mrec.filled()
        assert_equal(mrecfilled['a'], np.array((1, 2, 99999), dtype=int))
        assert_equal(mrecfilled['b'], np.array((1.1, 2.2, 99999.),
                                               dtype=float))
        assert_equal(mrecfilled['c'], np.array(('one', 'two', 'N/A'),
                                               dtype='|S8'))

    def test_tolist(self):
        # Test tolist.
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[1, 0, 0], dtype='|S8')
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        mrec = fromarrays([_a, _b, _c], dtype=ddtype,
                          fill_value=(99999, 99999., 'N/A'))

        assert_equal(mrec.tolist(),
                     [(1, 1.1, None), (2, 2.2, b'two'),
                      (None, None, b'three')])

    def test_withnames(self):
        # Test the creation w/ format and names
        x = mrecarray(1, formats=float, names='base')
        x[0]['base'] = 10
        assert_equal(x['base'][0], 10)

    def test_exotic_formats(self):
        # Test that 'exotic' formats are processed properly
        easy = mrecarray(1, dtype=[('i', int), ('s', '|S8'), ('f', float)])
        easy[0] = masked
        assert_equal(easy.filled(1).item(), (1, b'1', 1.))

        solo = mrecarray(1, dtype=[('f0', '<f8', (2, 2))])
        solo[0] = masked
        assert_equal(solo.filled(1).item(),
                     np.array((1,), dtype=solo.dtype).item())

        mult = mrecarray(2, dtype="i4, (2,3)float, float")
        mult[0] = masked
        mult[1] = (1, 1, 1)
        mult.filled(0)
        assert_equal_records(mult.filled(0),
                             np.array([(0, 0, 0), (1, 1, 1)],
                                      dtype=mult.dtype))


class TestView:

    def setup_method(self):
        (a, b) = (np.arange(10), np.random.rand(10))
        ndtype = [('a', float), ('b', float)]
        arr = np.array(list(zip(a, b)), dtype=ndtype)

        mrec = fromarrays([a, b], dtype=ndtype, fill_value=(-9., -99.))
        mrec.mask[3] = (False, True)
        self.data = (mrec, a, b, arr)

    def test_view_by_itself(self):
        (mrec, a, b, arr) = self.data
        test = mrec.view()
        assert_(isinstance(test, MaskedRecords))
        assert_equal_records(test, mrec)
        assert_equal_records(test._mask, mrec._mask)

    def test_view_simple_dtype(self):
        (mrec, a, b, arr) = self.data
        ntype = (float, 2)
        test = mrec.view(ntype)
        assert_(isinstance(test, ma.MaskedArray))
        assert_equal(test, np.array(list(zip(a, b)), dtype=float))
        assert_(test[3, 1] is ma.masked)

    def test_view_flexible_type(self):
        (mrec, a, b, arr) = self.data
        alttype = [('A', float), ('B', float)]
        test = mrec.view(alttype)
        assert_(isinstance(test, MaskedRecords))
        assert_equal_records(test, arr.view(alttype))
        assert_(test['B'][3] is masked)
        assert_equal(test.dtype, np.dtype(alttype))
        assert_(test._fill_value is None)


##############################################################################
class TestMRecordsImport:

    _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
    _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
    _c = ma.array([b'one', b'two', b'three'],
                  mask=[0, 0, 1], dtype='|S8')
    ddtype = [('a', int), ('b', float), ('c', '|S8')]
    mrec = fromarrays([_a, _b, _c], dtype=ddtype,
                      fill_value=(b'99999', b'99999.',
                                  b'N/A'))
    nrec = recfromarrays((_a._data, _b._data, _c._data), dtype=ddtype)
    data = (mrec, nrec, ddtype)

    def test_fromarrays(self):
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[0, 0, 1], dtype='|S8')
        (mrec, nrec, _) = self.data
        for (f, l) in zip(('a', 'b', 'c'), (_a, _b, _c)):
            assert_equal(getattr(mrec, f)._mask, l._mask)
        # One record only
        _x = ma.array([1, 1.1, 'one'], mask=[1, 0, 0], dtype=object)
        assert_equal_records(fromarrays(_x, dtype=mrec.dtype), mrec[0])

    def test_fromrecords(self):
        # Test construction from records.
        (mrec, nrec, ddtype) = self.data
        # ......
        palist = [(1, 'abc', 3.7000002861022949, 0),
                  (2, 'xy', 6.6999998092651367, 1),
                  (0, ' ', 0.40000000596046448, 0)]
        pa = recfromrecords(palist, names='c1, c2, c3, c4')
        mpa = fromrecords(palist, names='c1, c2, c3, c4')
        assert_equal_records(pa, mpa)
        # .....
        _mrec = fromrecords(nrec)
        assert_equal(_mrec.dtype, mrec.dtype)
        for field in _mrec.dtype.names:
            assert_equal(getattr(_mrec, field), getattr(mrec._data, field))

        _mrec = fromrecords(nrec.tolist(), names='c1,c2,c3')
        assert_equal(_mrec.dtype, [('c1', int), ('c2', float), ('c3', '|S5')])
        for (f, n) in zip(('c1', 'c2', 'c3'), ('a', 'b', 'c')):
            assert_equal(getattr(_mrec, f), getattr(mrec._data, n))

        _mrec = fromrecords(mrec)
        assert_equal(_mrec.dtype, mrec.dtype)
        assert_equal_records(_mrec._data, mrec.filled())
        assert_equal_records(_mrec._mask, mrec._mask)

    def test_fromrecords_wmask(self):
        # Tests construction from records w/ mask.
        (mrec, nrec, ddtype) = self.data

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=[0, 1, 0,])
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), [(0, 0, 0), (1, 1, 1), (0, 0, 0)])

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=True)
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), [(1, 1, 1), (1, 1, 1), (1, 1, 1)])

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=mrec._mask)
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), mrec._mask.tolist())

        _mrec = fromrecords(nrec.tolist(), dtype=ddtype,
                            mask=mrec._mask.tolist())
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), mrec._mask.tolist())

    def test_fromtextfile(self):
        # Tests reading from a text file.
        fcontent = (
"""#
'One (S)','Two (I)','Three (F)','Four (M)','Five (-)','Six (C)'
'strings',1,1.0,'mixed column',,1
'with embedded "double quotes"',2,2.0,1.0,,1
'strings',3,3.0E5,3,,1
'strings',4,-1e-10,,,1
""")
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(fcontent)
            mrectxt = fromtextfile(path, delimiter=',', varnames='ABCDEFG')
        assert_(isinstance(mrectxt, MaskedRecords))
        assert_equal(mrectxt.F, [1, 1, 1, 1])
        assert_equal(mrectxt.E._mask, [1, 1, 1, 1])
        assert_equal(mrectxt.C, [1, 2, 3.e+5, -1e-10])

    def test_addfield(self):
        # Tests addfield
        (mrec, nrec, ddtype) = self.data
        (d, m) = ([100, 200, 300], [1, 0, 0])
        mrec = addfield(mrec, ma.array(d, mask=m))
        assert_equal(mrec.f3, d)
        assert_equal(mrec.f3._mask, m)


def test_record_array_with_object_field():
    # Trac #1839
    y = ma.masked_array(
        [(1, '2'), (3, '4')],
        mask=[(0, 0), (0, 1)],
        dtype=[('a', int), ('b', object)])
    # getting an item used to fail
    y[1]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\tests\test_websocket.py ===
# -*- coding: utf-8 -*-
#
import os
import os.path
import socket
import unittest
from base64 import decodebytes as base64decode

import websocket as ws
from websocket._exceptions import WebSocketBadStatusException, WebSocketAddressException
from websocket._handshake import _create_sec_websocket_key
from websocket._handshake import _validate as _validate_header
from websocket._http import read_headers
from websocket._utils import validate_utf8

"""
test_websocket.py
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

try:
    import ssl
except ImportError:
    # dummy class of SSLError for ssl none-support environment.
    class SSLError(Exception):
        pass


# Skip test to access the internet unless TEST_WITH_INTERNET == 1
TEST_WITH_INTERNET = os.environ.get("TEST_WITH_INTERNET", "0") == "1"
# Skip tests relying on local websockets server unless LOCAL_WS_SERVER_PORT != -1
LOCAL_WS_SERVER_PORT = os.environ.get("LOCAL_WS_SERVER_PORT", "-1")
TEST_WITH_LOCAL_SERVER = LOCAL_WS_SERVER_PORT != "-1"
TRACEABLE = True


def create_mask_key(_):
    return "abcd"


class SockMock:
    def __init__(self):
        self.data = []
        self.sent = []

    def add_packet(self, data):
        self.data.append(data)

    def gettimeout(self):
        return None

    def recv(self, bufsize):
        if self.data:
            e = self.data.pop(0)
            if isinstance(e, Exception):
                raise e
            if len(e) > bufsize:
                self.data.insert(0, e[bufsize:])
            return e[:bufsize]

    def send(self, data):
        self.sent.append(data)
        return len(data)

    def close(self):
        pass


class HeaderSockMock(SockMock):
    def __init__(self, fname):
        SockMock.__init__(self)
        path = os.path.join(os.path.dirname(__file__), fname)
        with open(path, "rb") as f:
            self.add_packet(f.read())


class WebSocketTest(unittest.TestCase):
    def setUp(self):
        ws.enableTrace(TRACEABLE)

    def tearDown(self):
        pass

    def test_default_timeout(self):
        self.assertEqual(ws.getdefaulttimeout(), None)
        ws.setdefaulttimeout(10)
        self.assertEqual(ws.getdefaulttimeout(), 10)
        ws.setdefaulttimeout(None)

    def test_ws_key(self):
        key = _create_sec_websocket_key()
        self.assertTrue(key != 24)
        self.assertTrue("¥n" not in key)

    def test_nonce(self):
        """WebSocket key should be a random 16-byte nonce."""
        key = _create_sec_websocket_key()
        nonce = base64decode(key.encode("utf-8"))
        self.assertEqual(16, len(nonce))

    def test_ws_utils(self):
        key = "c6b8hTg4EeGb2gQMztV1/g=="
        required_header = {
            "upgrade": "websocket",
            "connection": "upgrade",
            "sec-websocket-accept": "Kxep+hNu9n51529fGidYu7a3wO0=",
        }
        self.assertEqual(_validate_header(required_header, key, None), (True, None))

        header = required_header.copy()
        header["upgrade"] = "http"
        self.assertEqual(_validate_header(header, key, None), (False, None))
        del header["upgrade"]
        self.assertEqual(_validate_header(header, key, None), (False, None))

        header = required_header.copy()
        header["connection"] = "something"
        self.assertEqual(_validate_header(header, key, None), (False, None))
        del header["connection"]
        self.assertEqual(_validate_header(header, key, None), (False, None))

        header = required_header.copy()
        header["sec-websocket-accept"] = "something"
        self.assertEqual(_validate_header(header, key, None), (False, None))
        del header["sec-websocket-accept"]
        self.assertEqual(_validate_header(header, key, None), (False, None))

        header = required_header.copy()
        header["sec-websocket-protocol"] = "sub1"
        self.assertEqual(
            _validate_header(header, key, ["sub1", "sub2"]), (True, "sub1")
        )
        # This case will print out a logging error using the error() function, but that is expected
        self.assertEqual(_validate_header(header, key, ["sub2", "sub3"]), (False, None))

        header = required_header.copy()
        header["sec-websocket-protocol"] = "sUb1"
        self.assertEqual(
            _validate_header(header, key, ["Sub1", "suB2"]), (True, "sub1")
        )

        header = required_header.copy()
        # This case will print out a logging error using the error() function, but that is expected
        self.assertEqual(_validate_header(header, key, ["Sub1", "suB2"]), (False, None))

    def test_read_header(self):
        status, header, _ = read_headers(HeaderSockMock("data/header01.txt"))
        self.assertEqual(status, 101)
        self.assertEqual(header["connection"], "Upgrade")

        status, header, _ = read_headers(HeaderSockMock("data/header03.txt"))
        self.assertEqual(status, 101)
        self.assertEqual(header["connection"], "Upgrade, Keep-Alive")

        HeaderSockMock("data/header02.txt")
        self.assertRaises(
            ws.WebSocketException, read_headers, HeaderSockMock("data/header02.txt")
        )

    def test_send(self):
        # TODO: add longer frame data
        sock = ws.WebSocket()
        sock.set_mask_key(create_mask_key)
        s = sock.sock = HeaderSockMock("data/header01.txt")
        sock.send("Hello")
        self.assertEqual(s.sent[0], b"\x81\x85abcd)\x07\x0f\x08\x0e")

        sock.send("こんにちは")
        self.assertEqual(
            s.sent[1],
            b"\x81\x8fabcd\x82\xe3\xf0\x87\xe3\xf1\x80\xe5\xca\x81\xe2\xc5\x82\xe3\xcc",
        )

        #        sock.send("x" * 5000)
        #        self.assertEqual(s.sent[1], b'\x81\x8fabcd\x82\xe3\xf0\x87\xe3\xf1\x80\xe5\xca\x81\xe2\xc5\x82\xe3\xcc")

        self.assertEqual(sock.send_binary(b"1111111111101"), 19)

    def test_recv(self):
        # TODO: add longer frame data
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        something = (
            b"\x81\x8fabcd\x82\xe3\xf0\x87\xe3\xf1\x80\xe5\xca\x81\xe2\xc5\x82\xe3\xcc"
        )
        s.add_packet(something)
        data = sock.recv()
        self.assertEqual(data, "こんにちは")

        s.add_packet(b"\x81\x85abcd)\x07\x0f\x08\x0e")
        data = sock.recv()
        self.assertEqual(data, "Hello")

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_iter(self):
        count = 2
        s = ws.create_connection("wss://api.bitfinex.com/ws/2")
        s.send('{"event": "subscribe", "channel": "ticker"}')
        for _ in s:
            count -= 1
            if count == 0:
                break

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_next(self):
        sock = ws.create_connection("wss://api.bitfinex.com/ws/2")
        self.assertEqual(str, type(next(sock)))

    def test_internal_recv_strict(self):
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        s.add_packet(b"foo")
        s.add_packet(socket.timeout())
        s.add_packet(b"bar")
        # s.add_packet(SSLError("The read operation timed out"))
        s.add_packet(b"baz")
        with self.assertRaises(ws.WebSocketTimeoutException):
            sock.frame_buffer.recv_strict(9)
        #     with self.assertRaises(SSLError):
        #         data = sock._recv_strict(9)
        data = sock.frame_buffer.recv_strict(9)
        self.assertEqual(data, b"foobarbaz")
        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.frame_buffer.recv_strict(1)

    def test_recv_timeout(self):
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        s.add_packet(b"\x81")
        s.add_packet(socket.timeout())
        s.add_packet(b"\x8dabcd\x29\x07\x0f\x08\x0e")
        s.add_packet(socket.timeout())
        s.add_packet(b"\x4e\x43\x33\x0e\x10\x0f\x00\x40")
        with self.assertRaises(ws.WebSocketTimeoutException):
            sock.recv()
        with self.assertRaises(ws.WebSocketTimeoutException):
            sock.recv()
        data = sock.recv()
        self.assertEqual(data, "Hello, World!")
        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.recv()

    def test_recv_with_simple_fragmentation(self):
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        # OPCODE=TEXT, FIN=0, MSG="Brevity is "
        s.add_packet(b"\x01\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C")
        # OPCODE=CONT, FIN=1, MSG="the soul of wit"
        s.add_packet(b"\x80\x8fabcd\x15\n\x06D\x12\r\x16\x08A\r\x05D\x16\x0b\x17")
        data = sock.recv()
        self.assertEqual(data, "Brevity is the soul of wit")
        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.recv()

    def test_recv_with_fire_event_of_fragmentation(self):
        sock = ws.WebSocket(fire_cont_frame=True)
        s = sock.sock = SockMock()
        # OPCODE=TEXT, FIN=0, MSG="Brevity is "
        s.add_packet(b"\x01\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C")
        # OPCODE=CONT, FIN=0, MSG="Brevity is "
        s.add_packet(b"\x00\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C")
        # OPCODE=CONT, FIN=1, MSG="the soul of wit"
        s.add_packet(b"\x80\x8fabcd\x15\n\x06D\x12\r\x16\x08A\r\x05D\x16\x0b\x17")

        _, data = sock.recv_data()
        self.assertEqual(data, b"Brevity is ")
        _, data = sock.recv_data()
        self.assertEqual(data, b"Brevity is ")
        _, data = sock.recv_data()
        self.assertEqual(data, b"the soul of wit")

        # OPCODE=CONT, FIN=0, MSG="Brevity is "
        s.add_packet(b"\x80\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C")

        with self.assertRaises(ws.WebSocketException):
            sock.recv_data()

        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.recv()

    def test_close(self):
        sock = ws.WebSocket()
        sock.connected = True
        sock.close

        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        sock.connected = True
        s.add_packet(b"\x88\x80\x17\x98p\x84")
        sock.recv()
        self.assertEqual(sock.connected, False)

    def test_recv_cont_fragmentation(self):
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        # OPCODE=CONT, FIN=1, MSG="the soul of wit"
        s.add_packet(b"\x80\x8fabcd\x15\n\x06D\x12\r\x16\x08A\r\x05D\x16\x0b\x17")
        self.assertRaises(ws.WebSocketException, sock.recv)

    def test_recv_with_prolonged_fragmentation(self):
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        # OPCODE=TEXT, FIN=0, MSG="Once more unto the breach, "
        s.add_packet(
            b"\x01\x9babcd.\x0c\x00\x01A\x0f\x0c\x16\x04B\x16\n\x15\rC\x10\t\x07C\x06\x13\x07\x02\x07\tNC"
        )
        # OPCODE=CONT, FIN=0, MSG="dear friends, "
        s.add_packet(b"\x00\x8eabcd\x05\x07\x02\x16A\x04\x11\r\x04\x0c\x07\x17MB")
        # OPCODE=CONT, FIN=1, MSG="once more"
        s.add_packet(b"\x80\x89abcd\x0e\x0c\x00\x01A\x0f\x0c\x16\x04")
        data = sock.recv()
        self.assertEqual(data, "Once more unto the breach, dear friends, once more")
        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.recv()

    def test_recv_with_fragmentation_and_control_frame(self):
        sock = ws.WebSocket()
        sock.set_mask_key(create_mask_key)
        s = sock.sock = SockMock()
        # OPCODE=TEXT, FIN=0, MSG="Too much "
        s.add_packet(b"\x01\x89abcd5\r\x0cD\x0c\x17\x00\x0cA")
        # OPCODE=PING, FIN=1, MSG="Please PONG this"
        s.add_packet(b"\x89\x90abcd1\x0e\x06\x05\x12\x07C4.,$D\x15\n\n\x17")
        # OPCODE=CONT, FIN=1, MSG="of a good thing"
        s.add_packet(b"\x80\x8fabcd\x0e\x04C\x05A\x05\x0c\x0b\x05B\x17\x0c\x08\x0c\x04")
        data = sock.recv()
        self.assertEqual(data, "Too much of a good thing")
        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.recv()
        self.assertEqual(
            s.sent[0], b"\x8a\x90abcd1\x0e\x06\x05\x12\x07C4.,$D\x15\n\n\x17"
        )

    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_websocket(self):
        s = ws.create_connection(f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}")
        self.assertNotEqual(s, None)
        s.send("Hello, World")
        result = s.next()
        s.fileno()
        self.assertEqual(result, "Hello, World")

        s.send("こにゃにゃちは、世界")
        result = s.recv()
        self.assertEqual(result, "こにゃにゃちは、世界")
        self.assertRaises(ValueError, s.send_close, -1, "")
        s.close()

    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_ping_pong(self):
        s = ws.create_connection(f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}")
        self.assertNotEqual(s, None)
        s.ping("Hello")
        s.pong("Hi")
        s.close()

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_support_redirect(self):
        s = ws.WebSocket()
        self.assertRaises(WebSocketBadStatusException, s.connect, "ws://google.com/")
        # Need to find a URL that has a redirect code leading to a websocket

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_secure_websocket(self):
        s = ws.create_connection("wss://api.bitfinex.com/ws/2")
        self.assertNotEqual(s, None)
        self.assertTrue(isinstance(s.sock, ssl.SSLSocket))
        self.assertEqual(s.getstatus(), 101)
        self.assertNotEqual(s.getheaders(), None)
        s.settimeout(10)
        self.assertEqual(s.gettimeout(), 10)
        self.assertEqual(s.getsubprotocol(), None)
        s.abort()

    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_websocket_with_custom_header(self):
        s = ws.create_connection(
            f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}",
            headers={"User-Agent": "PythonWebsocketClient"},
        )
        self.assertNotEqual(s, None)
        self.assertEqual(s.getsubprotocol(), None)
        s.send("Hello, World")
        result = s.recv()
        self.assertEqual(result, "Hello, World")
        self.assertRaises(ValueError, s.close, -1, "")
        s.close()

    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_after_close(self):
        s = ws.create_connection(f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}")
        self.assertNotEqual(s, None)
        s.close()
        self.assertRaises(ws.WebSocketConnectionClosedException, s.send, "Hello")
        self.assertRaises(ws.WebSocketConnectionClosedException, s.recv)


class SockOptTest(unittest.TestCase):
    @unittest.skipUnless(
        TEST_WITH_LOCAL_SERVER, "Tests using local websocket server are disabled"
    )
    def test_sockopt(self):
        sockopt = ((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),)
        s = ws.create_connection(
            f"ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}", sockopt=sockopt
        )
        self.assertNotEqual(
            s.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY), 0
        )
        s.close()


class UtilsTest(unittest.TestCase):
    def test_utf8_validator(self):
        state = validate_utf8(b"\xf0\x90\x80\x80")
        self.assertEqual(state, True)
        state = validate_utf8(
            b"\xce\xba\xe1\xbd\xb9\xcf\x83\xce\xbc\xce\xb5\xed\xa0\x80edited"
        )
        self.assertEqual(state, False)
        state = validate_utf8(b"")
        self.assertEqual(state, True)


class HandshakeTest(unittest.TestCase):
    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_http_ssl(self):
        websock1 = ws.WebSocket(
            sslopt={"cert_chain": ssl.get_default_verify_paths().capath},
            enable_multithread=False,
        )
        self.assertRaises(ValueError, websock1.connect, "wss://api.bitfinex.com/ws/2")
        websock2 = ws.WebSocket(sslopt={"certfile": "myNonexistentCertFile"})
        self.assertRaises(
            FileNotFoundError, websock2.connect, "wss://api.bitfinex.com/ws/2"
        )

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def test_manual_headers(self):
        websock3 = ws.WebSocket(
            sslopt={
                "ca_certs": ssl.get_default_verify_paths().cafile,
                "ca_cert_path": ssl.get_default_verify_paths().capath,
            }
        )
        self.assertRaises(
            WebSocketBadStatusException,
            websock3.connect,
            "wss://api.bitfinex.com/ws/2",
            cookie="chocolate",
            origin="testing_websockets.com",
            host="echo.websocket.events/websocket-client-test",
            subprotocols=["testproto"],
            connection="Upgrade",
            header={
                "CustomHeader1": "123",
                "Cookie": "TestValue",
                "Sec-WebSocket-Key": "k9kFAUWNAMmf5OEMfTlOEA==",
                "Sec-WebSocket-Protocol": "newprotocol",
            },
        )

    def test_ipv6(self):
        websock2 = ws.WebSocket()
        self.assertRaises(ValueError, websock2.connect, "2001:4860:4860::8888")

    def test_bad_urls(self):
        websock3 = ws.WebSocket()
        self.assertRaises(ValueError, websock3.connect, "ws//example.com")
        self.assertRaises(WebSocketAddressException, websock3.connect, "ws://example")
        self.assertRaises(ValueError, websock3.connect, "example.com")


if __name__ == "__main__":
    unittest.main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\methods\test_quantile.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize(
    "a_vals,b_vals",
    [
        # Ints
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),
        ([1, 2, 3, 4], [4, 3, 2, 1]),
        ([1, 2, 3, 4, 5], [4, 3, 2, 1]),
        # Floats
        ([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]),
        # Missing data
        ([1.0, np.nan, 3.0, np.nan, 5.0], [5.0, np.nan, 3.0, np.nan, 1.0]),
        ([np.nan, 4.0, np.nan, 2.0, np.nan], [np.nan, 4.0, np.nan, 2.0, np.nan]),
        # Timestamps
        (
            pd.date_range("1/1/18", freq="D", periods=5),
            pd.date_range("1/1/18", freq="D", periods=5)[::-1],
        ),
        (
            pd.date_range("1/1/18", freq="D", periods=5).as_unit("s"),
            pd.date_range("1/1/18", freq="D", periods=5)[::-1].as_unit("s"),
        ),
        # All NA
        ([np.nan] * 5, [np.nan] * 5),
    ],
)
@pytest.mark.parametrize("q", [0, 0.25, 0.5, 0.75, 1])
def test_quantile(interpolation, a_vals, b_vals, q, request):
    if (
        interpolation == "nearest"
        and q == 0.5
        and isinstance(b_vals, list)
        and b_vals == [4, 3, 2, 1]
    ):
        request.applymarker(
            pytest.mark.xfail(
                reason="Unclear numpy expectation for nearest "
                "result with equidistant data"
            )
        )
    all_vals = pd.concat([pd.Series(a_vals), pd.Series(b_vals)])

    a_expected = pd.Series(a_vals).quantile(q, interpolation=interpolation)
    b_expected = pd.Series(b_vals).quantile(q, interpolation=interpolation)

    df = DataFrame({"key": ["a"] * len(a_vals) + ["b"] * len(b_vals), "val": all_vals})

    expected = DataFrame(
        [a_expected, b_expected], columns=["val"], index=Index(["a", "b"], name="key")
    )
    if all_vals.dtype.kind == "M" and expected.dtypes.values[0].kind == "M":
        # TODO(non-nano): this should be unnecessary once array_to_datetime
        #  correctly infers non-nano from Timestamp.unit
        expected = expected.astype(all_vals.dtype)
    result = df.groupby("key").quantile(q, interpolation=interpolation)

    tm.assert_frame_equal(result, expected)


def test_quantile_array():
    # https://github.com/pandas-dev/pandas/issues/27526
    df = DataFrame({"A": [0, 1, 2, 3, 4]})
    key = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    result = df.groupby(key).quantile([0.25])

    index = pd.MultiIndex.from_product([[0, 1], [0.25]])
    expected = DataFrame({"A": [0.25, 2.50]}, index=index)
    tm.assert_frame_equal(result, expected)

    df = DataFrame({"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]})
    index = pd.MultiIndex.from_product([[0, 1], [0.25, 0.75]])

    key = np.array([0, 0, 1, 1], dtype=np.int64)
    result = df.groupby(key).quantile([0.25, 0.75])
    expected = DataFrame(
        {"A": [0.25, 0.75, 2.25, 2.75], "B": [4.25, 4.75, 6.25, 6.75]}, index=index
    )
    tm.assert_frame_equal(result, expected)


def test_quantile_array2():
    # https://github.com/pandas-dev/pandas/pull/28085#issuecomment-524066959
    arr = np.random.default_rng(2).integers(0, 5, size=(10, 3), dtype=np.int64)
    df = DataFrame(arr, columns=list("ABC"))
    result = df.groupby("A").quantile([0.3, 0.7])
    expected = DataFrame(
        {
            "B": [2.0, 2.0, 2.3, 2.7, 0.3, 0.7, 3.2, 4.0, 0.3, 0.7],
            "C": [1.0, 1.0, 1.9, 3.0999999999999996, 0.3, 0.7, 2.6, 3.0, 1.2, 2.8],
        },
        index=pd.MultiIndex.from_product(
            [[0, 1, 2, 3, 4], [0.3, 0.7]], names=["A", None]
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_quantile_array_no_sort():
    df = DataFrame({"A": [0, 1, 2], "B": [3, 4, 5]})
    key = np.array([1, 0, 1], dtype=np.int64)
    result = df.groupby(key, sort=False).quantile([0.25, 0.5, 0.75])
    expected = DataFrame(
        {"A": [0.5, 1.0, 1.5, 1.0, 1.0, 1.0], "B": [3.5, 4.0, 4.5, 4.0, 4.0, 4.0]},
        index=pd.MultiIndex.from_product([[1, 0], [0.25, 0.5, 0.75]]),
    )
    tm.assert_frame_equal(result, expected)

    result = df.groupby(key, sort=False).quantile([0.75, 0.25])
    expected = DataFrame(
        {"A": [1.5, 0.5, 1.0, 1.0], "B": [4.5, 3.5, 4.0, 4.0]},
        index=pd.MultiIndex.from_product([[1, 0], [0.75, 0.25]]),
    )
    tm.assert_frame_equal(result, expected)


def test_quantile_array_multiple_levels():
    df = DataFrame(
        {"A": [0, 1, 2], "B": [3, 4, 5], "c": ["a", "a", "a"], "d": ["a", "a", "b"]}
    )
    result = df.groupby(["c", "d"]).quantile([0.25, 0.75])
    index = pd.MultiIndex.from_tuples(
        [("a", "a", 0.25), ("a", "a", 0.75), ("a", "b", 0.25), ("a", "b", 0.75)],
        names=["c", "d", None],
    )
    expected = DataFrame(
        {"A": [0.25, 0.75, 2.0, 2.0], "B": [3.25, 3.75, 5.0, 5.0]}, index=index
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("frame_size", [(2, 3), (100, 10)])
@pytest.mark.parametrize("groupby", [[0], [0, 1]])
@pytest.mark.parametrize("q", [[0.5, 0.6]])
def test_groupby_quantile_with_arraylike_q_and_int_columns(frame_size, groupby, q):
    # GH30289
    nrow, ncol = frame_size
    df = DataFrame(np.array([ncol * [_ % 4] for _ in range(nrow)]), columns=range(ncol))

    idx_levels = [np.arange(min(nrow, 4))] * len(groupby) + [q]
    idx_codes = [[x for x in range(min(nrow, 4)) for _ in q]] * len(groupby) + [
        list(range(len(q))) * min(nrow, 4)
    ]
    expected_index = pd.MultiIndex(
        levels=idx_levels, codes=idx_codes, names=groupby + [None]
    )
    expected_values = [
        [float(x)] * (ncol - len(groupby)) for x in range(min(nrow, 4)) for _ in q
    ]
    expected_columns = [x for x in range(ncol) if x not in groupby]
    expected = DataFrame(
        expected_values, index=expected_index, columns=expected_columns
    )
    result = df.groupby(groupby).quantile(q)

    tm.assert_frame_equal(result, expected)


def test_quantile_raises():
    df = DataFrame([["foo", "a"], ["foo", "b"], ["foo", "c"]], columns=["key", "val"])

    msg = "dtype '(object|str)' does not support operation 'quantile'"
    with pytest.raises(TypeError, match=msg):
        df.groupby("key").quantile()


def test_quantile_out_of_bounds_q_raises():
    # https://github.com/pandas-dev/pandas/issues/27470
    df = DataFrame({"a": [0, 0, 0, 1, 1, 1], "b": range(6)})
    g = df.groupby([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="Got '50.0' instead"):
        g.quantile(50)

    with pytest.raises(ValueError, match="Got '-1.0' instead"):
        g.quantile(-1)


def test_quantile_missing_group_values_no_segfaults():
    # GH 28662
    data = np.array([1.0, np.nan, 1.0])
    df = DataFrame({"key": data, "val": range(3)})

    # Random segfaults; would have been guaranteed in loop
    grp = df.groupby("key")
    for _ in range(100):
        grp.quantile()


@pytest.mark.parametrize(
    "key, val, expected_key, expected_val",
    [
        ([1.0, np.nan, 3.0, np.nan], range(4), [1.0, 3.0], [0.0, 2.0]),
        ([1.0, np.nan, 2.0, 2.0], range(4), [1.0, 2.0], [0.0, 2.5]),
        (["a", "b", "b", np.nan], range(4), ["a", "b"], [0, 1.5]),
        ([0], [42], [0], [42.0]),
        ([], [], np.array([], dtype="float64"), np.array([], dtype="float64")),
    ],
)
def test_quantile_missing_group_values_correct_results(
    key, val, expected_key, expected_val
):
    # GH 28662, GH 33200, GH 33569
    df = DataFrame({"key": key, "val": val})

    expected = DataFrame(
        expected_val, index=Index(expected_key, name="key"), columns=["val"]
    )

    grp = df.groupby("key")

    result = grp.quantile(0.5)
    tm.assert_frame_equal(result, expected)

    result = grp.quantile()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        pd.array([1, 0, None] * 2, dtype="Int64"),
        pd.array([True, False, None] * 2, dtype="boolean"),
    ],
)
@pytest.mark.parametrize("q", [0.5, [0.0, 0.5, 1.0]])
def test_groupby_quantile_nullable_array(values, q):
    # https://github.com/pandas-dev/pandas/issues/33136
    df = DataFrame({"a": ["x"] * 3 + ["y"] * 3, "b": values})
    result = df.groupby("a")["b"].quantile(q)

    if isinstance(q, list):
        idx = pd.MultiIndex.from_product((["x", "y"], q), names=["a", None])
        true_quantiles = [0.0, 0.5, 1.0]
    else:
        idx = Index(["x", "y"], name="a")
        true_quantiles = [0.5]

    expected = pd.Series(true_quantiles * 2, index=idx, name="b", dtype="Float64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("q", [0.5, [0.0, 0.5, 1.0]])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_quantile_raises_on_invalid_dtype(q, numeric_only):
    df = DataFrame({"a": [1], "b": [2.0], "c": ["x"]})
    if numeric_only:
        result = df.groupby("a").quantile(q, numeric_only=numeric_only)
        expected = df.groupby("a")[["b"]].quantile(q)
        tm.assert_frame_equal(result, expected)
    else:
        msg = "dtype '.*' does not support operation 'quantile'"
        with pytest.raises(TypeError, match=msg):
            df.groupby("a").quantile(q, numeric_only=numeric_only)


def test_groupby_quantile_NA_float(any_float_dtype):
    # GH#42849
    df = DataFrame({"x": [1, 1], "y": [0.2, np.nan]}, dtype=any_float_dtype)
    result = df.groupby("x")["y"].quantile(0.5)
    exp_index = Index([1.0], dtype=any_float_dtype, name="x")

    if any_float_dtype in ["Float32", "Float64"]:
        expected_dtype = any_float_dtype
    else:
        expected_dtype = None

    expected = pd.Series([0.2], dtype=expected_dtype, index=exp_index, name="y")
    tm.assert_series_equal(result, expected)

    result = df.groupby("x")["y"].quantile([0.5, 0.75])
    expected = pd.Series(
        [0.2] * 2,
        index=pd.MultiIndex.from_product((exp_index, [0.5, 0.75]), names=["x", None]),
        name="y",
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)


def test_groupby_quantile_NA_int(any_int_ea_dtype):
    # GH#42849
    df = DataFrame({"x": [1, 1], "y": [2, 5]}, dtype=any_int_ea_dtype)
    result = df.groupby("x")["y"].quantile(0.5)
    expected = pd.Series(
        [3.5],
        dtype="Float64",
        index=Index([1], name="x", dtype=any_int_ea_dtype),
        name="y",
    )
    tm.assert_series_equal(expected, result)

    result = df.groupby("x").quantile(0.5)
    expected = DataFrame(
        {"y": 3.5}, dtype="Float64", index=Index([1], name="x", dtype=any_int_ea_dtype)
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "interpolation, val1, val2", [("lower", 2, 2), ("higher", 2, 3), ("nearest", 2, 2)]
)
def test_groupby_quantile_all_na_group_masked(
    interpolation, val1, val2, any_numeric_ea_dtype
):
    # GH#37493
    df = DataFrame(
        {"a": [1, 1, 1, 2], "b": [1, 2, 3, pd.NA]}, dtype=any_numeric_ea_dtype
    )
    result = df.groupby("a").quantile(q=[0.5, 0.7], interpolation=interpolation)
    expected = DataFrame(
        {"b": [val1, val2, pd.NA, pd.NA]},
        dtype=any_numeric_ea_dtype,
        index=pd.MultiIndex.from_arrays(
            [pd.Series([1, 1, 2, 2], dtype=any_numeric_ea_dtype), [0.5, 0.7, 0.5, 0.7]],
            names=["a", None],
        ),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("interpolation", ["midpoint", "linear"])
def test_groupby_quantile_all_na_group_masked_interp(
    interpolation, any_numeric_ea_dtype
):
    # GH#37493
    df = DataFrame(
        {"a": [1, 1, 1, 2], "b": [1, 2, 3, pd.NA]}, dtype=any_numeric_ea_dtype
    )
    result = df.groupby("a").quantile(q=[0.5, 0.75], interpolation=interpolation)

    if any_numeric_ea_dtype == "Float32":
        expected_dtype = any_numeric_ea_dtype
    else:
        expected_dtype = "Float64"

    expected = DataFrame(
        {"b": [2.0, 2.5, pd.NA, pd.NA]},
        dtype=expected_dtype,
        index=pd.MultiIndex.from_arrays(
            [
                pd.Series([1, 1, 2, 2], dtype=any_numeric_ea_dtype),
                [0.5, 0.75, 0.5, 0.75],
            ],
            names=["a", None],
        ),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Float64", "Float32"])
def test_groupby_quantile_allNA_column(dtype):
    # GH#42849
    df = DataFrame({"x": [1, 1], "y": [pd.NA] * 2}, dtype=dtype)
    result = df.groupby("x")["y"].quantile(0.5)
    expected = pd.Series(
        [np.nan], dtype=dtype, index=Index([1.0], dtype=dtype), name="y"
    )
    expected.index.name = "x"
    tm.assert_series_equal(expected, result)


def test_groupby_timedelta_quantile():
    # GH: 29485
    df = DataFrame(
        {"value": pd.to_timedelta(np.arange(4), unit="s"), "group": [1, 1, 2, 2]}
    )
    result = df.groupby("group").quantile(0.99)
    expected = DataFrame(
        {
            "value": [
                pd.Timedelta("0 days 00:00:00.990000"),
                pd.Timedelta("0 days 00:00:02.990000"),
            ]
        },
        index=Index([1, 2], name="group"),
    )
    tm.assert_frame_equal(result, expected)


def test_columns_groupby_quantile():
    # GH 33795
    df = DataFrame(
        np.arange(12).reshape(3, -1),
        index=list("XYZ"),
        columns=pd.Series(list("ABAB"), name="col"),
    )
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby("col", axis=1)
    result = gb.quantile(q=[0.8, 0.2])
    expected = DataFrame(
        [
            [1.6, 0.4, 2.6, 1.4],
            [5.6, 4.4, 6.6, 5.4],
            [9.6, 8.4, 10.6, 9.4],
        ],
        index=list("XYZ"),
        columns=pd.MultiIndex.from_tuples(
            [("A", 0.8), ("A", 0.2), ("B", 0.8), ("B", 0.2)], names=["col", None]
        ),
    )

    tm.assert_frame_equal(result, expected)


def test_timestamp_groupby_quantile(unit):
    # GH 33168
    dti = pd.date_range(
        start="2020-04-19 00:00:00", freq="1min", periods=100, tz="UTC", unit=unit
    ).floor("1h")
    df = DataFrame(
        {
            "timestamp": dti,
            "category": list(range(1, 101)),
            "value": list(range(101, 201)),
        }
    )

    result = df.groupby("timestamp").quantile([0.2, 0.8])

    mi = pd.MultiIndex.from_product([dti[::99], [0.2, 0.8]], names=("timestamp", None))
    expected = DataFrame(
        [
            {"category": 12.8, "value": 112.8},
            {"category": 48.2, "value": 148.2},
            {"category": 68.8, "value": 168.8},
            {"category": 92.2, "value": 192.2},
        ],
        index=mi,
    )

    tm.assert_frame_equal(result, expected)


def test_groupby_quantile_dt64tz_period():
    # GH#51373
    dti = pd.date_range("2016-01-01", periods=1000)
    df = pd.Series(dti).to_frame().copy()
    df[1] = dti.tz_localize("US/Pacific")
    df[2] = dti.to_period("D")
    df[3] = dti - dti[0]
    df.iloc[-1] = pd.NaT

    by = np.tile(np.arange(5), 200)
    gb = df.groupby(by)

    result = gb.quantile(0.5)

    # Check that we match the group-by-group result
    exp = {i: df.iloc[i::5].quantile(0.5) for i in range(5)}
    expected = DataFrame(exp).T.infer_objects()
    expected.index = expected.index.astype(int)

    tm.assert_frame_equal(result, expected)


def test_groupby_quantile_nonmulti_levels_order():
    # Non-regression test for GH #53009
    ind = pd.MultiIndex.from_tuples(
        [
            (0, "a", "B"),
            (0, "a", "A"),
            (0, "b", "B"),
            (0, "b", "A"),
            (1, "a", "B"),
            (1, "a", "A"),
            (1, "b", "B"),
            (1, "b", "A"),
        ],
        names=["sample", "cat0", "cat1"],
    )
    ser = pd.Series(range(8), index=ind)
    result = ser.groupby(level="cat1", sort=False).quantile([0.2, 0.8])

    qind = pd.MultiIndex.from_tuples(
        [("B", 0.2), ("B", 0.8), ("A", 0.2), ("A", 0.8)], names=["cat1", None]
    )
    expected = pd.Series([1.2, 4.8, 2.2, 5.8], index=qind)

    tm.assert_series_equal(result, expected)

    # We need to check that index levels are not sorted
    expected_levels = pd.core.indexes.frozen.FrozenList([["B", "A"], [0.2, 0.8]])
    tm.assert_equal(result.index.levels, expected_levels)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_symbolic.py ===
import pytest

from numpy.f2py.symbolic import (
    ArithOp,
    Expr,
    Language,
    Op,
    as_apply,
    as_array,
    as_complex,
    as_deref,
    as_eq,
    as_expr,
    as_factors,
    as_ge,
    as_gt,
    as_le,
    as_lt,
    as_ne,
    as_number,
    as_numer_denom,
    as_ref,
    as_string,
    as_symbol,
    as_terms,
    as_ternary,
    eliminate_quotes,
    fromstring,
    insert_quotes,
    normalize,
)

from . import util


class TestSymbolic(util.F2PyTest):
    def test_eliminate_quotes(self):
        def worker(s):
            r, d = eliminate_quotes(s)
            s1 = insert_quotes(r, d)
            assert s1 == s

        for kind in ["", "mykind_"]:
            worker(kind + '"1234" // "ABCD"')
            worker(kind + '"1234" // ' + kind + '"ABCD"')
            worker(kind + "\"1234\" // 'ABCD'")
            worker(kind + '"1234" // ' + kind + "'ABCD'")
            worker(kind + '"1\\"2\'AB\'34"')
            worker("a = " + kind + "'1\\'2\"AB\"34'")

    def test_sanity(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")

        assert x.op == Op.SYMBOL
        assert repr(x) == "Expr(Op.SYMBOL, 'x')"
        assert x == x
        assert x != y
        assert hash(x) is not None

        n = as_number(123)
        m = as_number(456)
        assert n.op == Op.INTEGER
        assert repr(n) == "Expr(Op.INTEGER, (123, 4))"
        assert n == n
        assert n != m
        assert hash(n) is not None

        fn = as_number(12.3)
        fm = as_number(45.6)
        assert fn.op == Op.REAL
        assert repr(fn) == "Expr(Op.REAL, (12.3, 4))"
        assert fn == fn
        assert fn != fm
        assert hash(fn) is not None

        c = as_complex(1, 2)
        c2 = as_complex(3, 4)
        assert c.op == Op.COMPLEX
        assert repr(c) == ("Expr(Op.COMPLEX, (Expr(Op.INTEGER, (1, 4)),"
                           " Expr(Op.INTEGER, (2, 4))))")
        assert c == c
        assert c != c2
        assert hash(c) is not None

        s = as_string("'123'")
        s2 = as_string('"ABC"')
        assert s.op == Op.STRING
        assert repr(s) == "Expr(Op.STRING, (\"'123'\", 1))", repr(s)
        assert s == s
        assert s != s2

        a = as_array((n, m))
        b = as_array((n, ))
        assert a.op == Op.ARRAY
        assert repr(a) == ("Expr(Op.ARRAY, (Expr(Op.INTEGER, (123, 4)),"
                           " Expr(Op.INTEGER, (456, 4))))")
        assert a == a
        assert a != b

        t = as_terms(x)
        u = as_terms(y)
        assert t.op == Op.TERMS
        assert repr(t) == "Expr(Op.TERMS, {Expr(Op.SYMBOL, 'x'): 1})"
        assert t == t
        assert t != u
        assert hash(t) is not None

        v = as_factors(x)
        w = as_factors(y)
        assert v.op == Op.FACTORS
        assert repr(v) == "Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'x'): 1})"
        assert v == v
        assert w != v
        assert hash(v) is not None

        t = as_ternary(x, y, z)
        u = as_ternary(x, z, y)
        assert t.op == Op.TERNARY
        assert t == t
        assert t != u
        assert hash(t) is not None

        e = as_eq(x, y)
        f = as_lt(x, y)
        assert e.op == Op.RELATIONAL
        assert e == e
        assert e != f
        assert hash(e) is not None

    def test_tostring_fortran(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        n = as_number(123)
        m = as_number(456)
        a = as_array((n, m))
        c = as_complex(n, m)

        assert str(x) == "x"
        assert str(n) == "123"
        assert str(a) == "[123, 456]"
        assert str(c) == "(123, 456)"

        assert str(Expr(Op.TERMS, {x: 1})) == "x"
        assert str(Expr(Op.TERMS, {x: 2})) == "2 * x"
        assert str(Expr(Op.TERMS, {x: -1})) == "-x"
        assert str(Expr(Op.TERMS, {x: -2})) == "-2 * x"
        assert str(Expr(Op.TERMS, {x: 1, y: 1})) == "x + y"
        assert str(Expr(Op.TERMS, {x: -1, y: -1})) == "-x - y"
        assert str(Expr(Op.TERMS, {x: 2, y: 3})) == "2 * x + 3 * y"
        assert str(Expr(Op.TERMS, {x: -2, y: 3})) == "-2 * x + 3 * y"
        assert str(Expr(Op.TERMS, {x: 2, y: -3})) == "2 * x - 3 * y"

        assert str(Expr(Op.FACTORS, {x: 1})) == "x"
        assert str(Expr(Op.FACTORS, {x: 2})) == "x ** 2"
        assert str(Expr(Op.FACTORS, {x: -1})) == "x ** -1"
        assert str(Expr(Op.FACTORS, {x: -2})) == "x ** -2"
        assert str(Expr(Op.FACTORS, {x: 1, y: 1})) == "x * y"
        assert str(Expr(Op.FACTORS, {x: 2, y: 3})) == "x ** 2 * y ** 3"

        v = Expr(Op.FACTORS, {x: 2, Expr(Op.TERMS, {x: 1, y: 1}): 3})
        assert str(v) == "x ** 2 * (x + y) ** 3", str(v)
        v = Expr(Op.FACTORS, {x: 2, Expr(Op.FACTORS, {x: 1, y: 1}): 3})
        assert str(v) == "x ** 2 * (x * y) ** 3", str(v)

        assert str(Expr(Op.APPLY, ("f", (), {}))) == "f()"
        assert str(Expr(Op.APPLY, ("f", (x, ), {}))) == "f(x)"
        assert str(Expr(Op.APPLY, ("f", (x, y), {}))) == "f(x, y)"
        assert str(Expr(Op.INDEXING, ("f", x))) == "f[x]"

        assert str(as_ternary(x, y, z)) == "merge(y, z, x)"
        assert str(as_eq(x, y)) == "x .eq. y"
        assert str(as_ne(x, y)) == "x .ne. y"
        assert str(as_lt(x, y)) == "x .lt. y"
        assert str(as_le(x, y)) == "x .le. y"
        assert str(as_gt(x, y)) == "x .gt. y"
        assert str(as_ge(x, y)) == "x .ge. y"

    def test_tostring_c(self):
        language = Language.C
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        n = as_number(123)

        assert Expr(Op.FACTORS, {x: 2}).tostring(language=language) == "x * x"
        assert (Expr(Op.FACTORS, {
            x + y: 2
        }).tostring(language=language) == "(x + y) * (x + y)")
        assert Expr(Op.FACTORS, {
            x: 12
        }).tostring(language=language) == "pow(x, 12)"

        assert as_apply(ArithOp.DIV, x,
                        y).tostring(language=language) == "x / y"
        assert (as_apply(ArithOp.DIV, x,
                         x + y).tostring(language=language) == "x / (x + y)")
        assert (as_apply(ArithOp.DIV, x - y, x +
                         y).tostring(language=language) == "(x - y) / (x + y)")
        assert (x + (x - y) / (x + y) +
                n).tostring(language=language) == "123 + x + (x - y) / (x + y)"

        assert as_ternary(x, y, z).tostring(language=language) == "(x?y:z)"
        assert as_eq(x, y).tostring(language=language) == "x == y"
        assert as_ne(x, y).tostring(language=language) == "x != y"
        assert as_lt(x, y).tostring(language=language) == "x < y"
        assert as_le(x, y).tostring(language=language) == "x <= y"
        assert as_gt(x, y).tostring(language=language) == "x > y"
        assert as_ge(x, y).tostring(language=language) == "x >= y"

    def test_operations(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")

        assert x + x == Expr(Op.TERMS, {x: 2})
        assert x - x == Expr(Op.INTEGER, (0, 4))
        assert x + y == Expr(Op.TERMS, {x: 1, y: 1})
        assert x - y == Expr(Op.TERMS, {x: 1, y: -1})
        assert x * x == Expr(Op.FACTORS, {x: 2})
        assert x * y == Expr(Op.FACTORS, {x: 1, y: 1})

        assert +x == x
        assert -x == Expr(Op.TERMS, {x: -1}), repr(-x)
        assert 2 * x == Expr(Op.TERMS, {x: 2})
        assert 2 + x == Expr(Op.TERMS, {x: 1, as_number(1): 2})
        assert 2 * x + 3 * y == Expr(Op.TERMS, {x: 2, y: 3})
        assert (x + y) * 2 == Expr(Op.TERMS, {x: 2, y: 2})

        assert x**2 == Expr(Op.FACTORS, {x: 2})
        assert (x + y)**2 == Expr(
            Op.TERMS,
            {
                Expr(Op.FACTORS, {x: 2}): 1,
                Expr(Op.FACTORS, {y: 2}): 1,
                Expr(Op.FACTORS, {
                    x: 1,
                    y: 1
                }): 2,
            },
        )
        assert (x + y) * x == x**2 + x * y
        assert (x + y)**2 == x**2 + 2 * x * y + y**2
        assert (x + y)**2 + (x - y)**2 == 2 * x**2 + 2 * y**2
        assert (x + y) * z == x * z + y * z
        assert z * (x + y) == x * z + y * z

        assert (x / 2) == as_apply(ArithOp.DIV, x, as_number(2))
        assert (2 * x / 2) == x
        assert (3 * x / 2) == as_apply(ArithOp.DIV, 3 * x, as_number(2))
        assert (4 * x / 2) == 2 * x
        assert (5 * x / 2) == as_apply(ArithOp.DIV, 5 * x, as_number(2))
        assert (6 * x / 2) == 3 * x
        assert ((3 * 5) * x / 6) == as_apply(ArithOp.DIV, 5 * x, as_number(2))
        assert (30 * x**2 * y**4 / (24 * x**3 * y**3)) == as_apply(
            ArithOp.DIV, 5 * y, 4 * x)
        assert ((15 * x / 6) / 5) == as_apply(ArithOp.DIV, x,
                                              as_number(2)), (15 * x / 6) / 5
        assert (x / (5 / x)) == as_apply(ArithOp.DIV, x**2, as_number(5))

        assert (x / 2.0) == Expr(Op.TERMS, {x: 0.5})

        s = as_string('"ABC"')
        t = as_string('"123"')

        assert s // t == Expr(Op.STRING, ('"ABC123"', 1))
        assert s // x == Expr(Op.CONCAT, (s, x))
        assert x // s == Expr(Op.CONCAT, (x, s))

        c = as_complex(1.0, 2.0)
        assert -c == as_complex(-1.0, -2.0)
        assert c + c == as_expr((1 + 2j) * 2)
        assert c * c == as_expr((1 + 2j)**2)

    def test_substitute(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        a = as_array((x, y))

        assert x.substitute({x: y}) == y
        assert (x + y).substitute({x: z}) == y + z
        assert (x * y).substitute({x: z}) == y * z
        assert (x**4).substitute({x: z}) == z**4
        assert (x / y).substitute({x: z}) == z / y
        assert x.substitute({x: y + z}) == y + z
        assert a.substitute({x: y + z}) == as_array((y + z, y))

        assert as_ternary(x, y,
                          z).substitute({x: y + z}) == as_ternary(y + z, y, z)
        assert as_eq(x, y).substitute({x: y + z}) == as_eq(y + z, y)

    def test_fromstring(self):

        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        f = as_symbol("f")
        s = as_string('"ABC"')
        t = as_string('"123"')
        a = as_array((x, y))

        assert fromstring("x") == x
        assert fromstring("+ x") == x
        assert fromstring("-  x") == -x
        assert fromstring("x + y") == x + y
        assert fromstring("x + 1") == x + 1
        assert fromstring("x * y") == x * y
        assert fromstring("x * 2") == x * 2
        assert fromstring("x / y") == x / y
        assert fromstring("x ** 2", language=Language.Python) == x**2
        assert fromstring("x ** 2 ** 3", language=Language.Python) == x**2**3
        assert fromstring("(x + y) * z") == (x + y) * z

        assert fromstring("f(x)") == f(x)
        assert fromstring("f(x,y)") == f(x, y)
        assert fromstring("f[x]") == f[x]
        assert fromstring("f[x][y]") == f[x][y]

        assert fromstring('"ABC"') == s
        assert (normalize(
            fromstring('"ABC" // "123" ',
                       language=Language.Fortran)) == s // t)
        assert fromstring('f("ABC")') == f(s)
        assert fromstring('MYSTRKIND_"ABC"') == as_string('"ABC"', "MYSTRKIND")

        assert fromstring("(/x, y/)") == a, fromstring("(/x, y/)")
        assert fromstring("f((/x, y/))") == f(a)
        assert fromstring("(/(x+y)*z/)") == as_array(((x + y) * z, ))

        assert fromstring("123") == as_number(123)
        assert fromstring("123_2") == as_number(123, 2)
        assert fromstring("123_myintkind") == as_number(123, "myintkind")

        assert fromstring("123.0") == as_number(123.0, 4)
        assert fromstring("123.0_4") == as_number(123.0, 4)
        assert fromstring("123.0_8") == as_number(123.0, 8)
        assert fromstring("123.0e0") == as_number(123.0, 4)
        assert fromstring("123.0d0") == as_number(123.0, 8)
        assert fromstring("123d0") == as_number(123.0, 8)
        assert fromstring("123e-0") == as_number(123.0, 4)
        assert fromstring("123d+0") == as_number(123.0, 8)
        assert fromstring("123.0_myrealkind") == as_number(123.0, "myrealkind")
        assert fromstring("3E4") == as_number(30000.0, 4)

        assert fromstring("(1, 2)") == as_complex(1, 2)
        assert fromstring("(1e2, PI)") == as_complex(as_number(100.0),
                                                     as_symbol("PI"))

        assert fromstring("[1, 2]") == as_array((as_number(1), as_number(2)))

        assert fromstring("POINT(x, y=1)") == as_apply(as_symbol("POINT"),
                                                       x,
                                                       y=as_number(1))
        assert fromstring(
            'PERSON(name="John", age=50, shape=(/34, 23/))') == as_apply(
                as_symbol("PERSON"),
                name=as_string('"John"'),
                age=as_number(50),
                shape=as_array((as_number(34), as_number(23))),
            )

        assert fromstring("x?y:z") == as_ternary(x, y, z)

        assert fromstring("*x") == as_deref(x)
        assert fromstring("**x") == as_deref(as_deref(x))
        assert fromstring("&x") == as_ref(x)
        assert fromstring("(*x) * (*y)") == as_deref(x) * as_deref(y)
        assert fromstring("(*x) * *y") == as_deref(x) * as_deref(y)
        assert fromstring("*x * *y") == as_deref(x) * as_deref(y)
        assert fromstring("*x**y") == as_deref(x) * as_deref(y)

        assert fromstring("x == y") == as_eq(x, y)
        assert fromstring("x != y") == as_ne(x, y)
        assert fromstring("x < y") == as_lt(x, y)
        assert fromstring("x > y") == as_gt(x, y)
        assert fromstring("x <= y") == as_le(x, y)
        assert fromstring("x >= y") == as_ge(x, y)

        assert fromstring("x .eq. y", language=Language.Fortran) == as_eq(x, y)
        assert fromstring("x .ne. y", language=Language.Fortran) == as_ne(x, y)
        assert fromstring("x .lt. y", language=Language.Fortran) == as_lt(x, y)
        assert fromstring("x .gt. y", language=Language.Fortran) == as_gt(x, y)
        assert fromstring("x .le. y", language=Language.Fortran) == as_le(x, y)
        assert fromstring("x .ge. y", language=Language.Fortran) == as_ge(x, y)

    def test_traverse(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        f = as_symbol("f")

        # Use traverse to substitute a symbol
        def replace_visit(s, r=z):
            if s == x:
                return r

        assert x.traverse(replace_visit) == z
        assert y.traverse(replace_visit) == y
        assert z.traverse(replace_visit) == z
        assert (f(y)).traverse(replace_visit) == f(y)
        assert (f(x)).traverse(replace_visit) == f(z)
        assert (f[y]).traverse(replace_visit) == f[y]
        assert (f[z]).traverse(replace_visit) == f[z]
        assert (x + y + z).traverse(replace_visit) == (2 * z + y)
        assert (x +
                f(y, x - z)).traverse(replace_visit) == (z +
                                                         f(y, as_number(0)))
        assert as_eq(x, y).traverse(replace_visit) == as_eq(z, y)

        # Use traverse to collect symbols, method 1
        function_symbols = set()
        symbols = set()

        def collect_symbols(s):
            if s.op is Op.APPLY:
                oper = s.data[0]
                function_symbols.add(oper)
                if oper in symbols:
                    symbols.remove(oper)
            elif s.op is Op.SYMBOL and s not in function_symbols:
                symbols.add(s)

        (x + f(y, x - z)).traverse(collect_symbols)
        assert function_symbols == {f}
        assert symbols == {x, y, z}

        # Use traverse to collect symbols, method 2
        def collect_symbols2(expr, symbols):
            if expr.op is Op.SYMBOL:
                symbols.add(expr)

        symbols = set()
        (x + f(y, x - z)).traverse(collect_symbols2, symbols)
        assert symbols == {x, y, z, f}

        # Use traverse to partially collect symbols
        def collect_symbols3(expr, symbols):
            if expr.op is Op.APPLY:
                # skip traversing function calls
                return expr
            if expr.op is Op.SYMBOL:
                symbols.add(expr)

        symbols = set()
        (x + f(y, x - z)).traverse(collect_symbols3, symbols)
        assert symbols == {x}

    def test_linear_solve(self):
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")

        assert x.linear_solve(x) == (as_number(1), as_number(0))
        assert (x + 1).linear_solve(x) == (as_number(1), as_number(1))
        assert (2 * x).linear_solve(x) == (as_number(2), as_number(0))
        assert (2 * x + 3).linear_solve(x) == (as_number(2), as_number(3))
        assert as_number(3).linear_solve(x) == (as_number(0), as_number(3))
        assert y.linear_solve(x) == (as_number(0), y)
        assert (y * z).linear_solve(x) == (as_number(0), y * z)

        assert (x + y).linear_solve(x) == (as_number(1), y)
        assert (z * x + y).linear_solve(x) == (z, y)
        assert ((z + y) * x + y).linear_solve(x) == (z + y, y)
        assert (z * y * x + y).linear_solve(x) == (z * y, y)

        pytest.raises(RuntimeError, lambda: (x * x).linear_solve(x))

    def test_as_numer_denom(self):
        x = as_symbol("x")
        y = as_symbol("y")
        n = as_number(123)

        assert as_numer_denom(x) == (x, as_number(1))
        assert as_numer_denom(x / n) == (x, n)
        assert as_numer_denom(n / x) == (n, x)
        assert as_numer_denom(x / y) == (x, y)
        assert as_numer_denom(x * y) == (x * y, as_number(1))
        assert as_numer_denom(n + x / y) == (x + n * y, y)
        assert as_numer_denom(n + x / (y - x / n)) == (y * n**2, y * n - x)

    def test_polynomial_atoms(self):
        x = as_symbol("x")
        y = as_symbol("y")
        n = as_number(123)

        assert x.polynomial_atoms() == {x}
        assert n.polynomial_atoms() == set()
        assert (y[x]).polynomial_atoms() == {y[x]}
        assert (y(x)).polynomial_atoms() == {y(x)}
        assert (y(x) + x).polynomial_atoms() == {y(x), x}
        assert (y(x) * x[y]).polynomial_atoms() == {y(x), x[y]}
        assert (y(x)**x).polynomial_atoms() == {y(x)}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\ranges\test_setops.py ===
from datetime import (
    datetime,
    timedelta,
)

from hypothesis import (
    assume,
    given,
    strategies as st,
)
import numpy as np
import pytest

from pandas import (
    Index,
    RangeIndex,
)
import pandas._testing as tm


class TestRangeIndexSetOps:
    @pytest.mark.parametrize("dtype", [None, "int64", "uint64"])
    def test_intersection_mismatched_dtype(self, dtype):
        # check that we cast to float, not object
        index = RangeIndex(start=0, stop=20, step=2, name="foo")
        index = Index(index, dtype=dtype)

        flt = index.astype(np.float64)

        # bc index.equals(flt), we go through fastpath and get RangeIndex back
        result = index.intersection(flt)
        tm.assert_index_equal(result, index, exact=True)

        result = flt.intersection(index)
        tm.assert_index_equal(result, flt, exact=True)

        # neither empty, not-equals
        result = index.intersection(flt[1:])
        tm.assert_index_equal(result, flt[1:], exact=True)

        result = flt[1:].intersection(index)
        tm.assert_index_equal(result, flt[1:], exact=True)

        # empty other
        result = index.intersection(flt[:0])
        tm.assert_index_equal(result, flt[:0], exact=True)

        result = flt[:0].intersection(index)
        tm.assert_index_equal(result, flt[:0], exact=True)

    def test_intersection_empty(self, sort, names):
        # name retention on empty intersections
        index = RangeIndex(start=0, stop=20, step=2, name=names[0])

        # empty other
        result = index.intersection(index[:0].rename(names[1]), sort=sort)
        tm.assert_index_equal(result, index[:0].rename(names[2]), exact=True)

        # empty self
        result = index[:0].intersection(index.rename(names[1]), sort=sort)
        tm.assert_index_equal(result, index[:0].rename(names[2]), exact=True)

    def test_intersection(self, sort):
        # intersect with Index with dtype int64
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index(np.arange(1, 6))
        result = index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected)

        result = other.intersection(index, sort=sort)
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

        # intersect with increasing RangeIndex
        other = RangeIndex(1, 6)
        result = index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected, exact="equiv")

        # intersect with decreasing RangeIndex
        other = RangeIndex(5, 0, -1)
        result = index.intersection(other, sort=sort)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected, exact="equiv")

        # reversed (GH 17296)
        result = other.intersection(index, sort=sort)
        tm.assert_index_equal(result, expected, exact="equiv")

        # GH 17296: intersect two decreasing RangeIndexes
        first = RangeIndex(10, -2, -2)
        other = RangeIndex(5, -4, -1)
        expected = first.astype(int).intersection(other.astype(int), sort=sort)
        result = first.intersection(other, sort=sort).astype(int)
        tm.assert_index_equal(result, expected)

        # reversed
        result = other.intersection(first, sort=sort).astype(int)
        tm.assert_index_equal(result, expected)

        index = RangeIndex(5, name="foo")

        # intersect of non-overlapping indices
        other = RangeIndex(5, 10, 1, name="foo")
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1, name="foo")
        tm.assert_index_equal(result, expected)

        other = RangeIndex(-1, -5, -1)
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected)

        # intersection of empty indices
        other = RangeIndex(0, 0, 1)
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1)
        tm.assert_index_equal(result, expected)

        result = other.intersection(index, sort=sort)
        tm.assert_index_equal(result, expected)

    def test_intersection_non_overlapping_gcd(self, sort, names):
        # intersection of non-overlapping values based on start value and gcd
        index = RangeIndex(1, 10, 2, name=names[0])
        other = RangeIndex(0, 10, 4, name=names[1])
        result = index.intersection(other, sort=sort)
        expected = RangeIndex(0, 0, 1, name=names[2])
        tm.assert_index_equal(result, expected)

    def test_union_noncomparable(self, sort):
        # corner case, Index with non-int64 dtype
        index = RangeIndex(start=0, stop=20, step=2)
        other = Index([datetime.now() + timedelta(i) for i in range(4)], dtype=object)
        result = index.union(other, sort=sort)
        expected = Index(np.concatenate((index, other)))
        tm.assert_index_equal(result, expected)

        result = other.union(index, sort=sort)
        expected = Index(np.concatenate((other, index)))
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "idx1, idx2, expected_sorted, expected_notsorted",
        [
            (
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
            ),
            (
                RangeIndex(0, 10, 1),
                RangeIndex(5, 20, 1),
                RangeIndex(0, 20, 1),
                RangeIndex(0, 20, 1),
            ),
            (
                RangeIndex(0, 10, 1),
                RangeIndex(10, 20, 1),
                RangeIndex(0, 20, 1),
                RangeIndex(0, 20, 1),
            ),
            (
                RangeIndex(0, -10, -1),
                RangeIndex(0, -10, -1),
                RangeIndex(0, -10, -1),
                RangeIndex(0, -10, -1),
            ),
            (
                RangeIndex(0, -10, -1),
                RangeIndex(-10, -20, -1),
                RangeIndex(-19, 1, 1),
                RangeIndex(0, -20, -1),
            ),
            (
                RangeIndex(0, 10, 2),
                RangeIndex(1, 10, 2),
                RangeIndex(0, 10, 1),
                Index(list(range(0, 10, 2)) + list(range(1, 10, 2))),
            ),
            (
                RangeIndex(0, 11, 2),
                RangeIndex(1, 12, 2),
                RangeIndex(0, 12, 1),
                Index(list(range(0, 11, 2)) + list(range(1, 12, 2))),
            ),
            (
                RangeIndex(0, 21, 4),
                RangeIndex(-2, 24, 4),
                RangeIndex(-2, 24, 2),
                Index(list(range(0, 21, 4)) + list(range(-2, 24, 4))),
            ),
            (
                RangeIndex(0, -20, -2),
                RangeIndex(-1, -21, -2),
                RangeIndex(-19, 1, 1),
                Index(list(range(0, -20, -2)) + list(range(-1, -21, -2))),
            ),
            (
                RangeIndex(0, 100, 5),
                RangeIndex(0, 100, 20),
                RangeIndex(0, 100, 5),
                RangeIndex(0, 100, 5),
            ),
            (
                RangeIndex(0, -100, -5),
                RangeIndex(5, -100, -20),
                RangeIndex(-95, 10, 5),
                Index(list(range(0, -100, -5)) + [5]),
            ),
            (
                RangeIndex(0, -11, -1),
                RangeIndex(1, -12, -4),
                RangeIndex(-11, 2, 1),
                Index(list(range(0, -11, -1)) + [1, -11]),
            ),
            (RangeIndex(0), RangeIndex(0), RangeIndex(0), RangeIndex(0)),
            (
                RangeIndex(0, -10, -2),
                RangeIndex(0),
                RangeIndex(0, -10, -2),
                RangeIndex(0, -10, -2),
            ),
            (
                RangeIndex(0, 100, 2),
                RangeIndex(100, 150, 200),
                RangeIndex(0, 102, 2),
                RangeIndex(0, 102, 2),
            ),
            (
                RangeIndex(0, -100, -2),
                RangeIndex(-100, 50, 102),
                RangeIndex(-100, 4, 2),
                Index(list(range(0, -100, -2)) + [-100, 2]),
            ),
            (
                RangeIndex(0, -100, -1),
                RangeIndex(0, -50, -3),
                RangeIndex(-99, 1, 1),
                RangeIndex(0, -100, -1),
            ),
            (
                RangeIndex(0, 1, 1),
                RangeIndex(5, 6, 10),
                RangeIndex(0, 6, 5),
                RangeIndex(0, 10, 5),
            ),
            (
                RangeIndex(0, 10, 5),
                RangeIndex(-5, -6, -20),
                RangeIndex(-5, 10, 5),
                Index([0, 5, -5]),
            ),
            (
                RangeIndex(0, 3, 1),
                RangeIndex(4, 5, 1),
                Index([0, 1, 2, 4]),
                Index([0, 1, 2, 4]),
            ),
            (
                RangeIndex(0, 10, 1),
                Index([], dtype=np.int64),
                RangeIndex(0, 10, 1),
                RangeIndex(0, 10, 1),
            ),
            (
                RangeIndex(0),
                Index([1, 5, 6]),
                Index([1, 5, 6]),
                Index([1, 5, 6]),
            ),
            # GH 43885
            (
                RangeIndex(0, 10),
                RangeIndex(0, 5),
                RangeIndex(0, 10),
                RangeIndex(0, 10),
            ),
        ],
        ids=lambda x: repr(x) if isinstance(x, RangeIndex) else x,
    )
    def test_union_sorted(self, idx1, idx2, expected_sorted, expected_notsorted):
        res1 = idx1.union(idx2, sort=None)
        tm.assert_index_equal(res1, expected_sorted, exact=True)

        res1 = idx1.union(idx2, sort=False)
        tm.assert_index_equal(res1, expected_notsorted, exact=True)

        res2 = idx2.union(idx1, sort=None)
        res3 = Index(idx1._values, name=idx1.name).union(idx2, sort=None)
        tm.assert_index_equal(res2, expected_sorted, exact=True)
        tm.assert_index_equal(res3, expected_sorted, exact="equiv")

    def test_union_same_step_misaligned(self):
        # GH#44019
        left = RangeIndex(range(0, 20, 4))
        right = RangeIndex(range(1, 21, 4))

        result = left.union(right)
        expected = Index([0, 1, 4, 5, 8, 9, 12, 13, 16, 17])
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference(self):
        # GH#12034 Cases where we operate against another RangeIndex and may
        #  get back another RangeIndex
        obj = RangeIndex.from_range(range(1, 10), name="foo")

        result = obj.difference(obj)
        expected = RangeIndex.from_range(range(0), name="foo")
        tm.assert_index_equal(result, expected, exact=True)

        result = obj.difference(expected.rename("bar"))
        tm.assert_index_equal(result, obj.rename(None), exact=True)

        result = obj.difference(obj[:3])
        tm.assert_index_equal(result, obj[3:], exact=True)

        result = obj.difference(obj[-3:])
        tm.assert_index_equal(result, obj[:-3], exact=True)

        # Flipping the step of 'other' doesn't affect the result, but
        #  flipping the stepof 'self' does when sort=None
        result = obj[::-1].difference(obj[-3:])
        tm.assert_index_equal(result, obj[:-3], exact=True)

        result = obj[::-1].difference(obj[-3:], sort=False)
        tm.assert_index_equal(result, obj[:-3][::-1], exact=True)

        result = obj[::-1].difference(obj[-3:][::-1])
        tm.assert_index_equal(result, obj[:-3], exact=True)

        result = obj[::-1].difference(obj[-3:][::-1], sort=False)
        tm.assert_index_equal(result, obj[:-3][::-1], exact=True)

        result = obj.difference(obj[2:6])
        expected = Index([1, 2, 7, 8, 9], name="foo")
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_sort(self):
        # GH#44085 ensure we respect the sort keyword

        idx = Index(range(4))[::-1]
        other = Index(range(3, 4))

        result = idx.difference(other)
        expected = Index(range(3))
        tm.assert_index_equal(result, expected, exact=True)

        result = idx.difference(other, sort=False)
        expected = expected[::-1]
        tm.assert_index_equal(result, expected, exact=True)

        # case where the intersection is empty
        other = range(10, 12)
        result = idx.difference(other, sort=None)
        expected = idx[::-1]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_mismatched_step(self):
        obj = RangeIndex.from_range(range(1, 10), name="foo")

        result = obj.difference(obj[::2])
        expected = obj[1::2]
        tm.assert_index_equal(result, expected, exact=True)

        result = obj[::-1].difference(obj[::2], sort=False)
        tm.assert_index_equal(result, expected[::-1], exact=True)

        result = obj.difference(obj[1::2])
        expected = obj[::2]
        tm.assert_index_equal(result, expected, exact=True)

        result = obj[::-1].difference(obj[1::2], sort=False)
        tm.assert_index_equal(result, expected[::-1], exact=True)

    def test_difference_interior_overlap_endpoints_preserved(self):
        left = RangeIndex(range(4))
        right = RangeIndex(range(1, 3))

        result = left.difference(right)
        expected = RangeIndex(0, 4, 3)
        assert expected.tolist() == [0, 3]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_endpoints_overlap_interior_preserved(self):
        left = RangeIndex(-8, 20, 7)
        right = RangeIndex(13, -9, -3)

        result = left.difference(right)
        expected = RangeIndex(-1, 13, 7)
        assert expected.tolist() == [-1, 6]
        tm.assert_index_equal(result, expected, exact=True)

    def test_difference_interior_non_preserving(self):
        # case with intersection of length 1 but RangeIndex is not preserved
        idx = Index(range(10))

        other = idx[3:4]
        result = idx.difference(other)
        expected = Index([0, 1, 2, 4, 5, 6, 7, 8, 9])
        tm.assert_index_equal(result, expected, exact=True)

        # case with other.step / self.step > 2
        other = idx[::3]
        result = idx.difference(other)
        expected = Index([1, 2, 4, 5, 7, 8])
        tm.assert_index_equal(result, expected, exact=True)

        # cases with only reaching one end of left
        obj = Index(range(20))
        other = obj[:10:2]
        result = obj.difference(other)
        expected = Index([1, 3, 5, 7, 9] + list(range(10, 20)))
        tm.assert_index_equal(result, expected, exact=True)

        other = obj[1:11:2]
        result = obj.difference(other)
        expected = Index([0, 2, 4, 6, 8, 10] + list(range(11, 20)))
        tm.assert_index_equal(result, expected, exact=True)

    def test_symmetric_difference(self):
        # GH#12034 Cases where we operate against another RangeIndex and may
        #  get back another RangeIndex
        left = RangeIndex.from_range(range(1, 10), name="foo")

        result = left.symmetric_difference(left)
        expected = RangeIndex.from_range(range(0), name="foo")
        tm.assert_index_equal(result, expected)

        result = left.symmetric_difference(expected.rename("bar"))
        tm.assert_index_equal(result, left.rename(None))

        result = left[:-2].symmetric_difference(left[2:])
        expected = Index([1, 2, 8, 9], name="foo")
        tm.assert_index_equal(result, expected, exact=True)

        right = RangeIndex.from_range(range(10, 15))

        result = left.symmetric_difference(right)
        expected = RangeIndex.from_range(range(1, 15))
        tm.assert_index_equal(result, expected)

        result = left.symmetric_difference(right[1:])
        expected = Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14])
        tm.assert_index_equal(result, expected, exact=True)


def assert_range_or_not_is_rangelike(index):
    """
    Check that we either have a RangeIndex or that this index *cannot*
    be represented as a RangeIndex.
    """
    if not isinstance(index, RangeIndex) and len(index) > 0:
        diff = index[:-1] - index[1:]
        assert not (diff == diff[0]).all()


@given(
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
    st.integers(-20, 20),
)
def test_range_difference(start1, stop1, step1, start2, stop2, step2):
    # test that
    #  a) we match Index[int64].difference and
    #  b) we return RangeIndex whenever it is possible to do so.
    assume(step1 != 0)
    assume(step2 != 0)

    left = RangeIndex(start1, stop1, step1)
    right = RangeIndex(start2, stop2, step2)

    result = left.difference(right, sort=None)
    assert_range_or_not_is_rangelike(result)

    left_int64 = Index(left.to_numpy())
    right_int64 = Index(right.to_numpy())

    alt = left_int64.difference(right_int64, sort=None)
    tm.assert_index_equal(result, alt, exact="equiv")

    result = left.difference(right, sort=False)
    assert_range_or_not_is_rangelike(result)

    alt = left_int64.difference(right_int64, sort=False)
    tm.assert_index_equal(result, alt, exact="equiv")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_exprtools.py ===
"""Tests for tools for manipulating of large commutative expressions. """

from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import simplify
from sympy.core.exprtools import (decompose_power, Factors, Term, _gcd_terms,
                                  gcd_terms, factor_terms, factor_nc, _mask_nc,
                                  _monotonic_sign)
from sympy.core.mul import _keep_coeff as _keep_coeff
from sympy.simplify.cse_opts import sub_pre
from sympy.testing.pytest import raises

from sympy.abc import a, b, t, x, y, z


def test_decompose_power():
    assert decompose_power(x) == (x, 1)
    assert decompose_power(x**2) == (x, 2)
    assert decompose_power(x**(2*y)) == (x**y, 2)
    assert decompose_power(x**(2*y/3)) == (x**(y/3), 2)
    assert decompose_power(x**(y*Rational(2, 3))) == (x**(y/3), 2)


def test_Factors():
    assert Factors() == Factors({}) == Factors(S.One)
    assert Factors().as_expr() is S.One
    assert Factors({x: 2, y: 3, sin(x): 4}).as_expr() == x**2*y**3*sin(x)**4
    assert Factors(S.Infinity) == Factors({oo: 1})
    assert Factors(S.NegativeInfinity) == Factors({oo: 1, -1: 1})
    # issue #18059:
    assert Factors((x**2)**S.Half).as_expr() == (x**2)**S.Half

    a = Factors({x: 5, y: 3, z: 7})
    b = Factors({      y: 4, z: 3, t: 10})

    assert a.mul(b) == a*b == Factors({x: 5, y: 7, z: 10, t: 10})

    assert a.div(b) == divmod(a, b) == \
        (Factors({x: 5, z: 4}), Factors({y: 1, t: 10}))
    assert a.quo(b) == a/b == Factors({x: 5, z: 4})
    assert a.rem(b) == a % b == Factors({y: 1, t: 10})

    assert a.pow(3) == a**3 == Factors({x: 15, y: 9, z: 21})
    assert b.pow(3) == b**3 == Factors({y: 12, z: 9, t: 30})

    assert a.gcd(b) == Factors({y: 3, z: 3})
    assert a.lcm(b) == Factors({x: 5, y: 4, z: 7, t: 10})

    a = Factors({x: 4, y: 7, t: 7})
    b = Factors({z: 1, t: 3})

    assert a.normal(b) == (Factors({x: 4, y: 7, t: 4}), Factors({z: 1}))

    assert Factors(sqrt(2)*x).as_expr() == sqrt(2)*x

    assert Factors(-I)*I == Factors()
    assert Factors({S.NegativeOne: S(3)})*Factors({S.NegativeOne: S.One, I: S(5)}) == \
        Factors(I)
    assert Factors(sqrt(I)*I) == Factors(I**(S(3)/2)) == Factors({I: S(3)/2})
    assert Factors({I: S(3)/2}).as_expr() == I**(S(3)/2)

    assert Factors(S(2)**x).div(S(3)**x) == \
        (Factors({S(2): x}), Factors({S(3): x}))
    assert Factors(2**(2*x + 2)).div(S(8)) == \
        (Factors({S(2): 2*x + 2}), Factors({S(8): S.One}))

    # coverage
    # /!\ things break if this is not True
    assert Factors({S.NegativeOne: Rational(3, 2)}) == Factors({I: S.One, S.NegativeOne: S.One})
    assert Factors({I: S.One, S.NegativeOne: Rational(1, 3)}).as_expr() == I*(-1)**Rational(1, 3)

    assert Factors(-1.) == Factors({S.NegativeOne: S.One, S(1.): 1})
    assert Factors(-2.) == Factors({S.NegativeOne: S.One, S(2.): 1})
    assert Factors((-2.)**x) == Factors({S(-2.): x})
    assert Factors(S(-2)) == Factors({S.NegativeOne: S.One, S(2): 1})
    assert Factors(S.Half) == Factors({S(2): -S.One})
    assert Factors(Rational(3, 2)) == Factors({S(3): S.One, S(2): S.NegativeOne})
    assert Factors({I: S.One}) == Factors(I)
    assert Factors({-1.0: 2, I: 1}) == Factors({S(1.0): 1, I: 1})
    assert Factors({S.NegativeOne: Rational(-3, 2)}).as_expr() == I
    A = symbols('A', commutative=False)
    assert Factors(2*A**2) == Factors({S(2): 1, A**2: 1})
    assert Factors(I) == Factors({I: S.One})
    assert Factors(x).normal(S(2)) == (Factors(x), Factors(S(2)))
    assert Factors(x).normal(S.Zero) == (Factors(), Factors(S.Zero))
    raises(ZeroDivisionError, lambda: Factors(x).div(S.Zero))
    assert Factors(x).mul(S(2)) == Factors(2*x)
    assert Factors(x).mul(S.Zero).is_zero
    assert Factors(x).mul(1/x).is_one
    assert Factors(x**sqrt(2)**3).as_expr() == x**(2*sqrt(2))
    assert Factors(x)**Factors(S(2)) == Factors(x**2)
    assert Factors(x).gcd(S.Zero) == Factors(x)
    assert Factors(x).lcm(S.Zero).is_zero
    assert Factors(S.Zero).div(x) == (Factors(S.Zero), Factors())
    assert Factors(x).div(x) == (Factors(), Factors())
    assert Factors({x: .2})/Factors({x: .2}) == Factors()
    assert Factors(x) != Factors()
    assert Factors(S.Zero).normal(x) == (Factors(S.Zero), Factors())
    n, d = x**(2 + y), x**2
    f = Factors(n)
    assert f.div(d) == f.normal(d) == (Factors(x**y), Factors())
    assert f.gcd(d) == Factors()
    d = x**y
    assert f.div(d) == f.normal(d) == (Factors(x**2), Factors())
    assert f.gcd(d) == Factors(d)
    n = d = 2**x
    f = Factors(n)
    assert f.div(d) == f.normal(d) == (Factors(), Factors())
    assert f.gcd(d) == Factors(d)
    n, d = 2**x, 2**y
    f = Factors(n)
    assert f.div(d) == f.normal(d) == (Factors({S(2): x}), Factors({S(2): y}))
    assert f.gcd(d) == Factors()

    # extraction of constant only
    n = x**(x + 3)
    assert Factors(n).normal(x**-3) == (Factors({x: x + 6}), Factors({}))
    assert Factors(n).normal(x**3) == (Factors({x: x}), Factors({}))
    assert Factors(n).normal(x**4) == (Factors({x: x}), Factors({x: 1}))
    assert Factors(n).normal(x**(y - 3)) == \
        (Factors({x: x + 6}), Factors({x: y}))
    assert Factors(n).normal(x**(y + 3)) == (Factors({x: x}), Factors({x: y}))
    assert Factors(n).normal(x**(y + 4)) == \
        (Factors({x: x}), Factors({x: y + 1}))

    assert Factors(n).div(x**-3) == (Factors({x: x + 6}), Factors({}))
    assert Factors(n).div(x**3) == (Factors({x: x}), Factors({}))
    assert Factors(n).div(x**4) == (Factors({x: x}), Factors({x: 1}))
    assert Factors(n).div(x**(y - 3)) == \
        (Factors({x: x + 6}), Factors({x: y}))
    assert Factors(n).div(x**(y + 3)) == (Factors({x: x}), Factors({x: y}))
    assert Factors(n).div(x**(y + 4)) == \
        (Factors({x: x}), Factors({x: y + 1}))

    assert Factors(3 * x / 2) == Factors({3: 1, 2: -1, x: 1})
    assert Factors(x * x / y) == Factors({x: 2, y: -1})
    assert Factors(27 * x / y**9) == Factors({27: 1, x: 1, y: -9})


def test_Term():
    a = Term(4*x*y**2/z/t**3)
    b = Term(2*x**3*y**5/t**3)

    assert a == Term(4, Factors({x: 1, y: 2}), Factors({z: 1, t: 3}))
    assert b == Term(2, Factors({x: 3, y: 5}), Factors({t: 3}))

    assert a.as_expr() == 4*x*y**2/z/t**3
    assert b.as_expr() == 2*x**3*y**5/t**3

    assert a.inv() == \
        Term(S.One/4, Factors({z: 1, t: 3}), Factors({x: 1, y: 2}))
    assert b.inv() == Term(S.Half, Factors({t: 3}), Factors({x: 3, y: 5}))

    assert a.mul(b) == a*b == \
        Term(8, Factors({x: 4, y: 7}), Factors({z: 1, t: 6}))
    assert a.quo(b) == a/b == Term(2, Factors({}), Factors({x: 2, y: 3, z: 1}))

    assert a.pow(3) == a**3 == \
        Term(64, Factors({x: 3, y: 6}), Factors({z: 3, t: 9}))
    assert b.pow(3) == b**3 == Term(8, Factors({x: 9, y: 15}), Factors({t: 9}))

    assert a.pow(-3) == a**(-3) == \
        Term(S.One/64, Factors({z: 3, t: 9}), Factors({x: 3, y: 6}))
    assert b.pow(-3) == b**(-3) == \
        Term(S.One/8, Factors({t: 9}), Factors({x: 9, y: 15}))

    assert a.gcd(b) == Term(2, Factors({x: 1, y: 2}), Factors({t: 3}))
    assert a.lcm(b) == Term(4, Factors({x: 3, y: 5}), Factors({z: 1, t: 3}))

    a = Term(4*x*y**2/z/t**3)
    b = Term(2*x**3*y**5*t**7)

    assert a.mul(b) == Term(8, Factors({x: 4, y: 7, t: 4}), Factors({z: 1}))

    assert Term((2*x + 2)**3) == Term(8, Factors({x + 1: 3}), Factors({}))
    assert Term((2*x + 2)*(3*x + 6)**2) == \
        Term(18, Factors({x + 1: 1, x + 2: 2}), Factors({}))


def test_gcd_terms():
    f = 2*(x + 1)*(x + 4)/(5*x**2 + 5) + (2*x + 2)*(x + 5)/(x**2 + 1)/5 + \
        (2*x + 2)*(x + 6)/(5*x**2 + 5)

    assert _gcd_terms(f) == ((Rational(6, 5))*((1 + x)/(1 + x**2)), 5 + x, 1)
    assert _gcd_terms(Add.make_args(f)) == \
        ((Rational(6, 5))*((1 + x)/(1 + x**2)), 5 + x, 1)

    newf = (Rational(6, 5))*((1 + x)*(5 + x)/(1 + x**2))
    assert gcd_terms(f) == newf
    args = Add.make_args(f)
    # non-Basic sequences of terms treated as terms of Add
    assert gcd_terms(list(args)) == newf
    assert gcd_terms(tuple(args)) == newf
    assert gcd_terms(set(args)) == newf
    # but a Basic sequence is treated as a container
    assert gcd_terms(Tuple(*args)) != newf
    assert gcd_terms(Basic(Tuple(S(1), 3*y + 3*x*y), Tuple(S(1), S(3)))) == \
        Basic(Tuple(S(1), 3*y*(x + 1)), Tuple(S(1), S(3)))
    # but we shouldn't change keys of a dictionary or some may be lost
    assert gcd_terms(Dict((x*(1 + y), S(2)), (x + x*y, y + x*y))) == \
        Dict({x*(y + 1): S(2), x + x*y: y*(1 + x)})

    assert gcd_terms((2*x + 2)**3 + (2*x + 2)**2) == 4*(x + 1)**2*(2*x + 3)

    assert gcd_terms(0) == 0
    assert gcd_terms(1) == 1
    assert gcd_terms(x) == x
    assert gcd_terms(2 + 2*x) == Mul(2, 1 + x, evaluate=False)
    arg = x*(2*x + 4*y)
    garg = 2*x*(x + 2*y)
    assert gcd_terms(arg) == garg
    assert gcd_terms(sin(arg)) == sin(garg)

    # issue 6139-like
    alpha, alpha1, alpha2, alpha3 = symbols('alpha:4')
    a = alpha**2 - alpha*x**2 + alpha + x**3 - x*(alpha + 1)
    rep = (alpha, (1 + sqrt(5))/2 + alpha1*x + alpha2*x**2 + alpha3*x**3)
    s = (a/(x - alpha)).subs(*rep).series(x, 0, 1)
    assert simplify(collect(s, x)) == -sqrt(5)/2 - Rational(3, 2) + O(x)

    # issue 5917
    assert _gcd_terms([S.Zero, S.Zero]) == (0, 0, 1)
    assert _gcd_terms([2*x + 4]) == (2, x + 2, 1)

    eq = x/(x + 1/x)
    assert gcd_terms(eq, fraction=False) == eq
    eq = x/2/y + 1/x/y
    assert gcd_terms(eq, fraction=True, clear=True) == \
        (x**2 + 2)/(2*x*y)
    assert gcd_terms(eq, fraction=True, clear=False) == \
        (x**2/2 + 1)/(x*y)
    assert gcd_terms(eq, fraction=False, clear=True) == \
        (x + 2/x)/(2*y)
    assert gcd_terms(eq, fraction=False, clear=False) == \
        (x/2 + 1/x)/y


def test_factor_terms():
    A = Symbol('A', commutative=False)
    assert factor_terms(9*(x + x*y + 1) + (3*x + 3)**(2 + 2*x)) == \
        9*x*y + 9*x + _keep_coeff(S(3), x + 1)**_keep_coeff(S(2), x + 1) + 9
    assert factor_terms(9*(x + x*y + 1) + (3)**(2 + 2*x)) == \
        _keep_coeff(S(9), 3**(2*x) + x*y + x + 1)
    assert factor_terms(3**(2 + 2*x) + a*3**(2 + 2*x)) == \
        9*3**(2*x)*(a + 1)
    assert factor_terms(x + x*A) == \
        x*(1 + A)
    assert factor_terms(sin(x + x*A)) == \
        sin(x*(1 + A))
    assert factor_terms((3*x + 3)**((2 + 2*x)/3)) == \
        _keep_coeff(S(3), x + 1)**_keep_coeff(Rational(2, 3), x + 1)
    assert factor_terms(x + (x*y + x)**(3*x + 3)) == \
        x + (x*(y + 1))**_keep_coeff(S(3), x + 1)
    assert factor_terms(a*(x + x*y) + b*(x*2 + y*x*2)) == \
        x*(a + 2*b)*(y + 1)
    i = Integral(x, (x, 0, oo))
    assert factor_terms(i) == i

    assert factor_terms(x/2 + y) == x/2 + y
    # fraction doesn't apply to integer denominators
    assert factor_terms(x/2 + y, fraction=True) == x/2 + y
    # clear *does* apply to the integer denominators
    assert factor_terms(x/2 + y, clear=True) == Mul(S.Half, x + 2*y, evaluate=False)

    # check radical extraction
    eq = sqrt(2) + sqrt(10)
    assert factor_terms(eq) == eq
    assert factor_terms(eq, radical=True) == sqrt(2)*(1 + sqrt(5))
    eq = root(-6, 3) + root(6, 3)
    assert factor_terms(eq, radical=True) == 6**(S.One/3)*(1 + (-1)**(S.One/3))

    eq = [x + x*y]
    ans = [x*(y + 1)]
    for c in [list, tuple, set]:
        assert factor_terms(c(eq)) == c(ans)
    assert factor_terms(Tuple(x + x*y)) == Tuple(x*(y + 1))
    assert factor_terms(Interval(0, 1)) == Interval(0, 1)
    e = 1/sqrt(a/2 + 1)
    assert factor_terms(e, clear=False) == 1/sqrt(a/2 + 1)
    assert factor_terms(e, clear=True) == sqrt(2)/sqrt(a + 2)

    eq = x/(x + 1/x) + 1/(x**2 + 1)
    assert factor_terms(eq, fraction=False) == eq
    assert factor_terms(eq, fraction=True) == 1

    assert factor_terms((1/(x**3 + x**2) + 2/x**2)*y) == \
        y*(2 + 1/(x + 1))/x**2

    # if not True, then processesing for this in factor_terms is not necessary
    assert gcd_terms(-x - y) == -x - y
    assert factor_terms(-x - y) == Mul(-1, x + y, evaluate=False)

    # if not True, then "special" processesing in factor_terms is not necessary
    assert gcd_terms(exp(Mul(-1, x + 1))) == exp(-x - 1)
    e = exp(-x - 2) + x
    assert factor_terms(e) == exp(Mul(-1, x + 2, evaluate=False)) + x
    assert factor_terms(e, sign=False) == e
    assert factor_terms(exp(-4*x - 2) - x) == -x + exp(Mul(-2, 2*x + 1, evaluate=False))

    # sum/integral tests
    for F in (Sum, Integral):
        assert factor_terms(F(x, (y, 1, 10))) == x * F(1, (y, 1, 10))
        assert factor_terms(F(x, (y, 1, 10)) + x) == x * (1 + F(1, (y, 1, 10)))
        assert factor_terms(F(x*y + x*y**2, (y, 1, 10))) == x*F(y*(y + 1), (y, 1, 10))

    # expressions involving Pow terms with base 0
    assert factor_terms(0**(x - 2) - 1) == 0**(x - 2) - 1
    assert factor_terms(0**(x + 2) - 1) == 0**(x + 2) - 1
    assert factor_terms((0**(x + 2) - 1).subs(x,-2)) == 0


def test_xreplace():
    e = Mul(2, 1 + x, evaluate=False)
    assert e.xreplace({}) == e
    assert e.xreplace({y: x}) == e


def test_factor_nc():
    x, y = symbols('x,y')
    k = symbols('k', integer=True)
    n, m, o = symbols('n,m,o', commutative=False)

    # mul and multinomial expansion is needed
    from sympy.core.function import _mexpand
    e = x*(1 + y)**2
    assert _mexpand(e) == x + x*2*y + x*y**2

    def factor_nc_test(e):
        ex = _mexpand(e)
        assert ex.is_Add
        f = factor_nc(ex)
        assert not f.is_Add and _mexpand(f) == ex

    factor_nc_test(x*(1 + y))
    factor_nc_test(n*(x + 1))
    factor_nc_test(n*(x + m))
    factor_nc_test((x + m)*n)
    factor_nc_test(n*m*(x*o + n*o*m)*n)
    s = Sum(x, (x, 1, 2))
    factor_nc_test(x*(1 + s))
    factor_nc_test(x*(1 + s)*s)
    factor_nc_test(x*(1 + sin(s)))
    factor_nc_test((1 + n)**2)

    factor_nc_test((x + n)*(x + m)*(x + y))
    factor_nc_test(x*(n*m + 1))
    factor_nc_test(x*(n*m + x))
    factor_nc_test(x*(x*n*m + 1))
    factor_nc_test(n*(m/x + o))
    factor_nc_test(m*(n + o/2))
    factor_nc_test(x*n*(x*m + 1))
    factor_nc_test(x*(m*n + x*n*m))
    factor_nc_test(n*(1 - m)*n**2)

    factor_nc_test((n + m)**2)
    factor_nc_test((n - m)*(n + m)**2)
    factor_nc_test((n + m)**2*(n - m))
    factor_nc_test((m - n)*(n + m)**2*(n - m))

    assert factor_nc(n*(n + n*m)) == n**2*(1 + m)
    assert factor_nc(m*(m*n + n*m*n**2)) == m*(m + n*m*n)*n
    eq = m*sin(n) - sin(n)*m
    assert factor_nc(eq) == eq

    # for coverage:
    from sympy.physics.secondquant import Commutator
    from sympy.polys.polytools import factor
    eq = 1 + x*Commutator(m, n)
    assert factor_nc(eq) == eq
    eq = x*Commutator(m, n) + x*Commutator(m, o)*Commutator(m, n)
    assert factor(eq) == x*(1 + Commutator(m, o))*Commutator(m, n)

    # issue 6534
    assert (2*n + 2*m).factor() == 2*(n + m)

    # issue 6701
    _n = symbols('nz', zero=False, commutative=False)
    assert factor_nc(_n**k + _n**(k + 1)) == _n**k*(1 + _n)
    assert factor_nc((m*n)**k + (m*n)**(k + 1)) == (1 + m*n)*(m*n)**k

    # issue 6918
    assert factor_nc(-n*(2*x**2 + 2*x)) == -2*n*x*(x + 1)


def test_issue_6360():
    a, b = symbols("a b")
    apb = a + b
    eq = apb + apb**2*(-2*a - 2*b)
    assert factor_terms(sub_pre(eq)) == a + b - 2*(a + b)**3


def test_issue_7903():
    a = symbols(r'a', real=True)
    t = exp(I*cos(a)) + exp(-I*sin(a))
    assert t.simplify()

def test_issue_8263():
    F, G = symbols('F, G', commutative=False, cls=Function)
    x, y = symbols('x, y')
    expr, dummies, _ = _mask_nc(F(x)*G(y) - G(y)*F(x))
    for v in dummies.values():
        assert not v.is_commutative
    assert not expr.is_zero

def test_monotonic_sign():
    F = _monotonic_sign
    x = symbols('x')
    assert F(x) is None
    assert F(-x) is None
    assert F(Dummy(prime=True)) == 2
    assert F(Dummy(prime=True, odd=True)) == 3
    assert F(Dummy(composite=True)) == 4
    assert F(Dummy(composite=True, odd=True)) == 9
    assert F(Dummy(positive=True, integer=True)) == 1
    assert F(Dummy(positive=True, even=True)) == 2
    assert F(Dummy(positive=True, even=True, prime=False)) == 4
    assert F(Dummy(negative=True, integer=True)) == -1
    assert F(Dummy(negative=True, even=True)) == -2
    assert F(Dummy(zero=True)) == 0
    assert F(Dummy(nonnegative=True)) == 0
    assert F(Dummy(nonpositive=True)) == 0

    assert F(Dummy(positive=True) + 1).is_positive
    assert F(Dummy(positive=True, integer=True) - 1).is_nonnegative
    assert F(Dummy(positive=True) - 1) is None
    assert F(Dummy(negative=True) + 1) is None
    assert F(Dummy(negative=True, integer=True) - 1).is_nonpositive
    assert F(Dummy(negative=True) - 1).is_negative
    assert F(-Dummy(positive=True) + 1) is None
    assert F(-Dummy(positive=True, integer=True) - 1).is_negative
    assert F(-Dummy(positive=True) - 1).is_negative
    assert F(-Dummy(negative=True) + 1).is_positive
    assert F(-Dummy(negative=True, integer=True) - 1).is_nonnegative
    assert F(-Dummy(negative=True) - 1) is None
    x = Dummy(negative=True)
    assert F(x**3).is_nonpositive
    assert F(x**3 + log(2)*x - 1).is_negative
    x = Dummy(positive=True)
    assert F(-x**3).is_nonpositive

    p = Dummy(positive=True)
    assert F(1/p).is_positive
    assert F(p/(p + 1)).is_positive
    p = Dummy(nonnegative=True)
    assert F(p/(p + 1)).is_nonnegative
    p = Dummy(positive=True)
    assert F(-1/p).is_negative
    p = Dummy(nonpositive=True)
    assert F(p/(-p + 1)).is_nonpositive

    p = Dummy(positive=True, integer=True)
    q = Dummy(positive=True, integer=True)
    assert F(-2/p/q).is_negative
    assert F(-2/(p - 1)/q) is None

    assert F((p - 1)*q + 1).is_positive
    assert F(-(p - 1)*q - 1).is_negative

def test_issue_17256():
    from sympy.sets.fancysets import Range
    x = Symbol('x')
    s1 = Sum(x + 1, (x, 1, 9))
    s2 = Sum(x + 1, (x, Range(1, 10)))
    a = Symbol('a')
    r1 = s1.xreplace({x:a})
    r2 = s2.xreplace({x:a})

    assert r1.doit() == r2.doit()
    s1 = Sum(x + 1, (x, 0, 9))
    s2 = Sum(x + 1, (x, Range(10)))
    a = Symbol('a')
    r1 = s1.xreplace({x:a})
    r2 = s2.xreplace({x:a})
    assert r1 == r2

def test_issue_21623():
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    M = MatrixSymbol('X', 2, 2)
    assert gcd_terms(M[0,0], 1) == M[0,0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_pycode.py ===
from sympy import Not
from sympy.codegen import Assignment
from sympy.codegen.ast import none
from sympy.codegen.cfunctions import expm1, log1p
from sympy.codegen.scipy_nodes import cosm1
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core import Expr, Mod, symbols, Eq, Le, Gt, zoo, oo, Rational, Pow
from sympy.core.function import Derivative
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions import acos, KroneckerDelta, Piecewise, sign, sqrt, Min, Max, cot, acsch, asec, coth, sec, log, sin, cos, tan, asin, atan, sinh, cosh, tanh, asinh, acosh, atanh
from sympy.functions.elementary.trigonometric import atan2
from sympy.logic import And, Or
from sympy.matrices import SparseMatrix, MatrixSymbol, Identity
from sympy.printing.codeprinter import PrintMethodNotImplementedError
from sympy.printing.pycode import (
    MpmathPrinter, CmathPrinter, PythonCodePrinter, pycode, SymPyPrinter
)
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter
from sympy.testing.pytest import raises, skip
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayDiagonal, ArrayContraction, ZeroArray, OneArray
from sympy.external import import_module
from sympy.functions.special.gamma_functions import loggamma



x, y, z = symbols('x y z')
p = IndexedBase("p")


def test_PythonCodePrinter():
    prntr = PythonCodePrinter()

    assert not prntr.module_imports

    assert prntr.doprint(x**y) == 'x**y'
    assert prntr.doprint(Mod(x, 2)) == 'x % 2'
    assert prntr.doprint(-Mod(x, y)) == '-(x % y)'
    assert prntr.doprint(Mod(-x, y)) == '(-x) % y'
    assert prntr.doprint(And(x, y)) == 'x and y'
    assert prntr.doprint(Or(x, y)) == 'x or y'
    assert prntr.doprint(1/(x+y)) == '1/(x + y)'
    assert prntr.doprint(Not(x)) == 'not x'
    assert not prntr.module_imports

    assert prntr.doprint(pi) == 'math.pi'
    assert prntr.module_imports == {'math': {'pi'}}

    assert prntr.doprint(x**Rational(1, 2)) == 'math.sqrt(x)'
    assert prntr.doprint(sqrt(x)) == 'math.sqrt(x)'
    assert prntr.module_imports == {'math': {'pi', 'sqrt'}}

    assert prntr.doprint(acos(x)) == 'math.acos(x)'
    assert prntr.doprint(cot(x)) == '(1/math.tan(x))'
    assert prntr.doprint(coth(x)) == '((math.exp(x) + math.exp(-x))/(math.exp(x) - math.exp(-x)))'
    assert prntr.doprint(asec(x)) == '(math.acos(1/x))'
    assert prntr.doprint(acsch(x)) == '(math.log(math.sqrt(1 + x**(-2)) + 1/x))'

    assert prntr.doprint(Assignment(x, 2)) == 'x = 2'
    assert prntr.doprint(Piecewise((1, Eq(x, 0)),
                        (2, x>6))) == '((1) if (x == 0) else (2) if (x > 6) else None)'
    assert prntr.doprint(Piecewise((2, Le(x, 0)),
                        (3, Gt(x, 0)), evaluate=False)) == '((2) if (x <= 0) else'\
                                                        ' (3) if (x > 0) else None)'
    assert prntr.doprint(sign(x)) == '(0.0 if x == 0 else math.copysign(1, x))'
    assert prntr.doprint(p[0, 1]) == 'p[0, 1]'
    assert prntr.doprint(KroneckerDelta(x,y)) == '(1 if x == y else 0)'

    assert prntr.doprint((2,3)) == "(2, 3)"
    assert prntr.doprint([2,3]) == "[2, 3]"

    assert prntr.doprint(Min(x, y)) == "min(x, y)"
    assert prntr.doprint(Max(x, y)) == "max(x, y)"


def test_PythonCodePrinter_standard():
    prntr = PythonCodePrinter()

    assert prntr.standard == 'python3'

    raises(ValueError, lambda: PythonCodePrinter({'standard':'python4'}))


def test_CmathPrinter():
    p = CmathPrinter()

    assert p.doprint(sqrt(x)) == 'cmath.sqrt(x)'
    assert p.doprint(log(x)) == 'cmath.log(x)'

    assert p.doprint(sin(x)) == 'cmath.sin(x)'
    assert p.doprint(cos(x)) == 'cmath.cos(x)'
    assert p.doprint(tan(x)) == 'cmath.tan(x)'

    assert p.doprint(asin(x)) == 'cmath.asin(x)'
    assert p.doprint(acos(x)) == 'cmath.acos(x)'
    assert p.doprint(atan(x)) == 'cmath.atan(x)'

    assert p.doprint(sinh(x)) == 'cmath.sinh(x)'
    assert p.doprint(cosh(x)) == 'cmath.cosh(x)'
    assert p.doprint(tanh(x)) == 'cmath.tanh(x)'

    assert p.doprint(asinh(x)) == 'cmath.asinh(x)'
    assert p.doprint(acosh(x)) == 'cmath.acosh(x)'
    assert p.doprint(atanh(x)) == 'cmath.atanh(x)'


def test_MpmathPrinter():
    p = MpmathPrinter()
    assert p.doprint(sign(x)) == 'mpmath.sign(x)'
    assert p.doprint(Rational(1, 2)) == 'mpmath.mpf(1)/mpmath.mpf(2)'

    assert p.doprint(S.Exp1) == 'mpmath.e'
    assert p.doprint(S.Pi) == 'mpmath.pi'
    assert p.doprint(S.GoldenRatio) == 'mpmath.phi'
    assert p.doprint(S.EulerGamma) == 'mpmath.euler'
    assert p.doprint(S.NaN) == 'mpmath.nan'
    assert p.doprint(S.Infinity) == 'mpmath.inf'
    assert p.doprint(S.NegativeInfinity) == 'mpmath.ninf'
    assert p.doprint(loggamma(x)) == 'mpmath.loggamma(x)'


def test_NumPyPrinter():
    from sympy.core.function import Lambda
    from sympy.matrices.expressions.adjoint import Adjoint
    from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix, DiagonalOf)
    from sympy.matrices.expressions.funcmatrix import FunctionMatrix
    from sympy.matrices.expressions.hadamard import HadamardProduct
    from sympy.matrices.expressions.kronecker import KroneckerProduct
    from sympy.matrices.expressions.special import (OneMatrix, ZeroMatrix)
    from sympy.abc import a, b
    p = NumPyPrinter()
    assert p.doprint(sign(x)) == 'numpy.sign(x)'
    A = MatrixSymbol("A", 2, 2)
    B = MatrixSymbol("B", 2, 2)
    C = MatrixSymbol("C", 1, 5)
    D = MatrixSymbol("D", 3, 4)
    assert p.doprint(A**(-1)) == "numpy.linalg.inv(A)"
    assert p.doprint(A**5) == "numpy.linalg.matrix_power(A, 5)"
    assert p.doprint(Identity(3)) == "numpy.eye(3)"

    u = MatrixSymbol('x', 2, 1)
    v = MatrixSymbol('y', 2, 1)
    assert p.doprint(MatrixSolve(A, u)) == 'numpy.linalg.solve(A, x)'
    assert p.doprint(MatrixSolve(A, u) + v) == 'numpy.linalg.solve(A, x) + y'

    assert p.doprint(ZeroMatrix(2, 3)) == "numpy.zeros((2, 3))"
    assert p.doprint(OneMatrix(2, 3)) == "numpy.ones((2, 3))"
    assert p.doprint(FunctionMatrix(4, 5, Lambda((a, b), a + b))) == \
        "numpy.fromfunction(lambda a, b: a + b, (4, 5))"
    assert p.doprint(HadamardProduct(A, B)) == "numpy.multiply(A, B)"
    assert p.doprint(KroneckerProduct(A, B)) == "numpy.kron(A, B)"
    assert p.doprint(Adjoint(A)) == "numpy.conjugate(numpy.transpose(A))"
    assert p.doprint(DiagonalOf(A)) == "numpy.reshape(numpy.diag(A), (-1, 1))"
    assert p.doprint(DiagMatrix(C)) == "numpy.diagflat(C)"
    assert p.doprint(DiagonalMatrix(D)) == "numpy.multiply(D, numpy.eye(3, 4))"

    # Workaround for numpy negative integer power errors
    assert p.doprint(x**-1) == 'x**(-1.0)'
    assert p.doprint(x**-2) == 'x**(-2.0)'

    expr = Pow(2, -1, evaluate=False)
    assert p.doprint(expr) == "2**(-1.0)"

    assert p.doprint(S.Exp1) == 'numpy.e'
    assert p.doprint(S.Pi) == 'numpy.pi'
    assert p.doprint(S.EulerGamma) == 'numpy.euler_gamma'
    assert p.doprint(S.NaN) == 'numpy.nan'
    assert p.doprint(S.Infinity) == 'numpy.inf'
    assert p.doprint(S.NegativeInfinity) == '-numpy.inf'

    # Function rewriting operator precedence fix
    assert p.doprint(sec(x)**2) == '(numpy.cos(x)**(-1.0))**2'


def test_issue_18770():
    numpy = import_module('numpy')
    if not numpy:
        skip("numpy not installed.")

    from sympy.functions.elementary.miscellaneous import (Max, Min)
    from sympy.utilities.lambdify import lambdify

    expr1 = Min(0.1*x + 3, x + 1, 0.5*x + 1)
    func = lambdify(x, expr1, "numpy")
    assert (func(numpy.linspace(0, 3, 3)) == [1.0, 1.75, 2.5 ]).all()
    assert  func(4) == 3

    expr1 = Max(x**2, x**3)
    func = lambdify(x,expr1, "numpy")
    assert (func(numpy.linspace(-1, 2, 4)) == [1, 0, 1, 8] ).all()
    assert func(4) == 64


def test_SciPyPrinter():
    p = SciPyPrinter()
    expr = acos(x)
    assert 'numpy' not in p.module_imports
    assert p.doprint(expr) == 'numpy.arccos(x)'
    assert 'numpy' in p.module_imports
    assert not any(m.startswith('scipy') for m in p.module_imports)
    smat = SparseMatrix(2, 5, {(0, 1): 3})
    assert p.doprint(smat) == \
        'scipy.sparse.coo_matrix(([3], ([0], [1])), shape=(2, 5))'
    assert 'scipy.sparse' in p.module_imports

    assert p.doprint(S.GoldenRatio) == 'scipy.constants.golden_ratio'
    assert p.doprint(S.Pi) == 'scipy.constants.pi'
    assert p.doprint(S.Exp1) == 'numpy.e'


def test_pycode_reserved_words():
    s1, s2 = symbols('if else')
    raises(ValueError, lambda: pycode(s1 + s2, error_on_reserved=True))
    py_str = pycode(s1 + s2)
    assert py_str in ('else_ + if_', 'if_ + else_')


def test_issue_20762():
    # Make sure pycode removes curly braces from subscripted variables
    a_b, b, a_11 = symbols('a_{b} b a_{11}')
    expr = a_b*b
    assert pycode(expr) == 'a_b*b'
    expr = a_11*b
    assert pycode(expr) == 'a_11*b'


def test_sqrt():
    prntr = PythonCodePrinter()
    assert prntr._print_Pow(sqrt(x), rational=False) == 'math.sqrt(x)'
    assert prntr._print_Pow(1/sqrt(x), rational=False) == '1/math.sqrt(x)'

    prntr = PythonCodePrinter({'standard' : 'python3'})
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'
    assert prntr._print_Pow(1/sqrt(x), rational=True) == 'x**(-1/2)'

    prntr = MpmathPrinter()
    assert prntr._print_Pow(sqrt(x), rational=False) == 'mpmath.sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == \
        "x**(mpmath.mpf(1)/mpmath.mpf(2))"

    prntr = NumPyPrinter()
    assert prntr._print_Pow(sqrt(x), rational=False) == 'numpy.sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'

    prntr = SciPyPrinter()
    assert prntr._print_Pow(sqrt(x), rational=False) == 'numpy.sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'

    prntr = SymPyPrinter()
    assert prntr._print_Pow(sqrt(x), rational=False) == 'sympy.sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'


def test_frac():
    from sympy.functions.elementary.integers import frac

    expr = frac(x)
    prntr = NumPyPrinter()
    assert prntr.doprint(expr) == 'numpy.mod(x, 1)'

    prntr = SciPyPrinter()
    assert prntr.doprint(expr) == 'numpy.mod(x, 1)'

    prntr = PythonCodePrinter()
    assert prntr.doprint(expr) == 'x % 1'

    prntr = MpmathPrinter()
    assert prntr.doprint(expr) == 'mpmath.frac(x)'

    prntr = SymPyPrinter()
    assert prntr.doprint(expr) == 'sympy.functions.elementary.integers.frac(x)'


class CustomPrintedObject(Expr):
    def _numpycode(self, printer):
        return 'numpy'

    def _mpmathcode(self, printer):
        return 'mpmath'


def test_printmethod():
    obj = CustomPrintedObject()
    assert NumPyPrinter().doprint(obj) == 'numpy'
    assert MpmathPrinter().doprint(obj) == 'mpmath'


def test_codegen_ast_nodes():
    assert pycode(none) == 'None'


def test_issue_14283():
    prntr = PythonCodePrinter()

    assert prntr.doprint(zoo) == "math.nan"
    assert prntr.doprint(-oo) == "float('-inf')"


def test_NumPyPrinter_print_seq():
    n = NumPyPrinter()

    assert n._print_seq(range(2)) == '(0, 1,)'


def test_issue_16535_16536():
    from sympy.functions.special.gamma_functions import (lowergamma, uppergamma)

    a = symbols('a')
    expr1 = lowergamma(a, x)
    expr2 = uppergamma(a, x)

    prntr = SciPyPrinter()
    assert prntr.doprint(expr1) == 'scipy.special.gamma(a)*scipy.special.gammainc(a, x)'
    assert prntr.doprint(expr2) == 'scipy.special.gamma(a)*scipy.special.gammaincc(a, x)'

    p_numpy = NumPyPrinter()
    p_pycode = PythonCodePrinter({'strict': False})

    for expr in [expr1, expr2]:
        with raises(NotImplementedError):
            p_numpy.doprint(expr1)
        assert "Not supported" in p_pycode.doprint(expr)


def test_Integral():
    from sympy.functions.elementary.exponential import exp
    from sympy.integrals.integrals import Integral

    single = Integral(exp(-x), (x, 0, oo))
    double = Integral(x**2*exp(x*y), (x, -z, z), (y, 0, z))
    indefinite = Integral(x**2, x)
    evaluateat = Integral(x**2, (x, 1))

    prntr = SciPyPrinter()
    assert prntr.doprint(single) == 'scipy.integrate.quad(lambda x: numpy.exp(-x), 0, numpy.inf)[0]'
    assert prntr.doprint(double) == 'scipy.integrate.nquad(lambda x, y: x**2*numpy.exp(x*y), ((-z, z), (0, z)))[0]'
    raises(NotImplementedError, lambda: prntr.doprint(indefinite))
    raises(NotImplementedError, lambda: prntr.doprint(evaluateat))

    prntr = MpmathPrinter()
    assert prntr.doprint(single) == 'mpmath.quad(lambda x: mpmath.exp(-x), (0, mpmath.inf))'
    assert prntr.doprint(double) == 'mpmath.quad(lambda x, y: x**2*mpmath.exp(x*y), (-z, z), (0, z))'
    raises(NotImplementedError, lambda: prntr.doprint(indefinite))
    raises(NotImplementedError, lambda: prntr.doprint(evaluateat))


def test_fresnel_integrals():
    from sympy.functions.special.error_functions import (fresnelc, fresnels)

    expr1 = fresnelc(x)
    expr2 = fresnels(x)

    prntr = SciPyPrinter()
    assert prntr.doprint(expr1) == 'scipy.special.fresnel(x)[1]'
    assert prntr.doprint(expr2) == 'scipy.special.fresnel(x)[0]'

    p_numpy = NumPyPrinter()
    p_pycode = PythonCodePrinter()
    p_mpmath = MpmathPrinter()
    for expr in [expr1, expr2]:
        with raises(NotImplementedError):
            p_numpy.doprint(expr)
        with raises(NotImplementedError):
            p_pycode.doprint(expr)

    assert p_mpmath.doprint(expr1) == 'mpmath.fresnelc(x)'
    assert p_mpmath.doprint(expr2) == 'mpmath.fresnels(x)'


def test_beta():
    from sympy.functions.special.beta_functions import beta

    expr = beta(x, y)

    prntr = SciPyPrinter()
    assert prntr.doprint(expr) == 'scipy.special.beta(x, y)'

    prntr = NumPyPrinter()
    assert prntr.doprint(expr) == '(math.gamma(x)*math.gamma(y)/math.gamma(x + y))'

    prntr = PythonCodePrinter()
    assert prntr.doprint(expr) == '(math.gamma(x)*math.gamma(y)/math.gamma(x + y))'

    prntr = PythonCodePrinter({'allow_unknown_functions': True})
    assert prntr.doprint(expr) == '(math.gamma(x)*math.gamma(y)/math.gamma(x + y))'

    prntr = MpmathPrinter()
    assert prntr.doprint(expr) ==  'mpmath.beta(x, y)'

def test_airy():
    from sympy.functions.special.bessel import (airyai, airybi)

    expr1 = airyai(x)
    expr2 = airybi(x)

    prntr = SciPyPrinter()
    assert prntr.doprint(expr1) == 'scipy.special.airy(x)[0]'
    assert prntr.doprint(expr2) == 'scipy.special.airy(x)[2]'

    prntr = NumPyPrinter({'strict': False})
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)

    prntr = PythonCodePrinter({'strict': False})
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)

def test_airy_prime():
    from sympy.functions.special.bessel import (airyaiprime, airybiprime)

    expr1 = airyaiprime(x)
    expr2 = airybiprime(x)

    prntr = SciPyPrinter()
    assert prntr.doprint(expr1) == 'scipy.special.airy(x)[1]'
    assert prntr.doprint(expr2) == 'scipy.special.airy(x)[3]'

    prntr = NumPyPrinter({'strict': False})
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)

    prntr = PythonCodePrinter({'strict': False})
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)


def test_numerical_accuracy_functions():
    prntr = SciPyPrinter()
    assert prntr.doprint(expm1(x)) == 'numpy.expm1(x)'
    assert prntr.doprint(log1p(x)) == 'numpy.log1p(x)'
    assert prntr.doprint(cosm1(x)) == 'scipy.special.cosm1(x)'

def test_array_printer():
    A = ArraySymbol('A', (4,4,6,6,6))
    I = IndexedBase('I')
    i,j,k = Idx('i', (0,1)), Idx('j', (2,3)), Idx('k', (4,5))

    prntr = NumPyPrinter()
    assert prntr.doprint(ZeroArray(5)) == 'numpy.zeros((5,))'
    assert prntr.doprint(OneArray(5)) == 'numpy.ones((5,))'
    assert prntr.doprint(ArrayContraction(A, [2,3])) == 'numpy.einsum("abccd->abd", A)'
    assert prntr.doprint(I) == 'I'
    assert prntr.doprint(ArrayDiagonal(A, [2,3,4])) == 'numpy.einsum("abccc->abc", A)'
    assert prntr.doprint(ArrayDiagonal(A, [0,1], [2,3])) == 'numpy.einsum("aabbc->cab", A)'
    assert prntr.doprint(ArrayContraction(A, [2], [3])) == 'numpy.einsum("abcde->abe", A)'
    assert prntr.doprint(Assignment(I[i,j,k], I[i,j,k])) == 'I = I'

    prntr = TensorflowPrinter()
    assert prntr.doprint(ZeroArray(5)) == 'tensorflow.zeros((5,))'
    assert prntr.doprint(OneArray(5)) == 'tensorflow.ones((5,))'
    assert prntr.doprint(ArrayContraction(A, [2,3])) == 'tensorflow.linalg.einsum("abccd->abd", A)'
    assert prntr.doprint(I) == 'I'
    assert prntr.doprint(ArrayDiagonal(A, [2,3,4])) == 'tensorflow.linalg.einsum("abccc->abc", A)'
    assert prntr.doprint(ArrayDiagonal(A, [0,1], [2,3])) == 'tensorflow.linalg.einsum("aabbc->cab", A)'
    assert prntr.doprint(ArrayContraction(A, [2], [3])) == 'tensorflow.linalg.einsum("abcde->abe", A)'
    assert prntr.doprint(Assignment(I[i,j,k], I[i,j,k])) == 'I = I'


def test_custom_Derivative_methods():
    class MyPrinter(SciPyPrinter):
        def _print_Derivative_cosm1(self, args, seq_orders):
            arg, = args
            order, = seq_orders
            return 'my_custom_cosm1(%s, deriv_order=%d)' % (self._print(arg), order)

        def _print_Derivative_atan2(self, args, seq_orders):
            arg1, arg2 = args
            ord1, ord2 = seq_orders
            return 'my_custom_atan2(%s, %s, deriv1=%d, deriv2=%d)' % (
                self._print(arg1), self._print(arg2), ord1, ord2
            )

    p = MyPrinter()
    cosm1_1 = cosm1(x).diff(x, evaluate=False)
    assert p.doprint(cosm1_1) == 'my_custom_cosm1(x, deriv_order=1)'
    atan2_2_3 = atan2(x, y).diff(x, 2, y, 3, evaluate=False)
    assert p.doprint(atan2_2_3) == 'my_custom_atan2(x, y, deriv1=2, deriv2=3)'

    try:
        p.doprint(expm1(x).diff(x, evaluate=False))
    except PrintMethodNotImplementedError as e:
        assert '_print_Derivative_expm1' in repr(e)
    else:
        assert False  # should have thrown

    try:
        p.doprint(Derivative(cosm1(x**2),x))
    except ValueError as e:
        assert '_print_Derivative(' in repr(e)
    else:
        assert False  # should have thrown

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_tensorflow.py ===
import random
from sympy.core.function import Derivative
from sympy.core.symbol import symbols
from sympy import Piecewise
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
    PermuteDims, ArrayDiagonal
from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
from sympy.external import import_module
from sympy.functions import \
    Abs, ceiling, exp, floor, sign, sin, asin, sqrt, cos, \
    acos, tan, atan, atan2, cosh, acosh, sinh, asinh, tanh, atanh, \
    re, im, arg, erf, loggamma, log
from sympy.codegen.cfunctions import isnan, isinf
from sympy.matrices import Matrix, MatrixBase, eye, randMatrix
from sympy.matrices.expressions import \
    Determinant, HadamardProduct, Inverse, MatrixSymbol, Trace
from sympy.printing.tensorflow import tensorflow_code
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import skip
from sympy.testing.pytest import XFAIL


tf = tensorflow = import_module("tensorflow")

if tensorflow:
    # Hide Tensorflow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


M = MatrixSymbol("M", 3, 3)
N = MatrixSymbol("N", 3, 3)
P = MatrixSymbol("P", 3, 3)
Q = MatrixSymbol("Q", 3, 3)

x, y, z, t = symbols("x y z t")

if tf is not None:
    llo = [list(range(i, i+3)) for i in range(0, 9, 3)]
    m3x3 = tf.constant(llo)
    m3x3sympy = Matrix(llo)


def _compare_tensorflow_matrix(variables, expr, use_float=False):
    f = lambdify(variables, expr, 'tensorflow')
    if not use_float:
        random_matrices = [randMatrix(v.rows, v.cols) for v in variables]
    else:
        random_matrices = [randMatrix(v.rows, v.cols)/100. for v in variables]

    graph = tf.Graph()
    r = None
    with graph.as_default():
        random_variables = [eval(tensorflow_code(i)) for i in random_matrices]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*random_variables))

    e = expr.subs(dict(zip(variables, random_matrices)))
    e = e.doit()
    if e.is_Matrix:
        if not isinstance(e, MatrixBase):
            e = e.as_explicit()
        e = e.tolist()

    if not use_float:
        assert (r == e).all()
    else:
        r = [i for row in r for i in row]
        e = [i for row in e for i in row]
        assert all(
            abs(a-b) < 10**-(4-int(log(abs(a), 10))) for a, b in zip(r, e))


# Creating a custom inverse test.
# See https://github.com/sympy/sympy/issues/18469
def _compare_tensorflow_matrix_inverse(variables, expr, use_float=False):
    f = lambdify(variables, expr, 'tensorflow')
    if not use_float:
        random_matrices = [eye(v.rows, v.cols)*4 for v in variables]
    else:
        random_matrices = [eye(v.rows, v.cols)*3.14 for v in variables]

    graph = tf.Graph()
    r = None
    with graph.as_default():
        random_variables = [eval(tensorflow_code(i)) for i in random_matrices]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*random_variables))

    e = expr.subs(dict(zip(variables, random_matrices)))
    e = e.doit()
    if e.is_Matrix:
        if not isinstance(e, MatrixBase):
            e = e.as_explicit()
        e = e.tolist()

    if not use_float:
        assert (r == e).all()
    else:
        r = [i for row in r for i in row]
        e = [i for row in e for i in row]
        assert all(
            abs(a-b) < 10**-(4-int(log(abs(a), 10))) for a, b in zip(r, e))


def _compare_tensorflow_matrix_scalar(variables, expr):
    f = lambdify(variables, expr, 'tensorflow')
    random_matrices = [
        randMatrix(v.rows, v.cols).evalf() / 100 for v in variables]

    graph = tf.Graph()
    r = None
    with graph.as_default():
        random_variables = [eval(tensorflow_code(i)) for i in random_matrices]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*random_variables))

    e = expr.subs(dict(zip(variables, random_matrices)))
    e = e.doit()
    assert abs(r-e) < 10**-6


def _compare_tensorflow_scalar(
    variables, expr, rng=lambda: random.randint(0, 10)):
    f = lambdify(variables, expr, 'tensorflow')
    rvs = [rng() for v in variables]

    graph = tf.Graph()
    r = None
    with graph.as_default():
        tf_rvs = [eval(tensorflow_code(i)) for i in rvs]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*tf_rvs))

    e = expr.subs(dict(zip(variables, rvs))).evalf().doit()
    assert abs(r-e) < 10**-6


def _compare_tensorflow_relational(
    variables, expr, rng=lambda: random.randint(0, 10)):
    f = lambdify(variables, expr, 'tensorflow')
    rvs = [rng() for v in variables]

    graph = tf.Graph()
    r = None
    with graph.as_default():
        tf_rvs = [eval(tensorflow_code(i)) for i in rvs]
        session = tf.compat.v1.Session(graph=graph)
        r = session.run(f(*tf_rvs))

    e = expr.subs(dict(zip(variables, rvs))).doit()
    assert r == e


def test_tensorflow_printing():
    assert tensorflow_code(eye(3)) == \
        "tensorflow.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])"

    expr = Matrix([[x, sin(y)], [exp(z), -t]])
    assert tensorflow_code(expr) == \
        "tensorflow.Variable(" \
            "[[x, tensorflow.math.sin(y)]," \
            " [tensorflow.math.exp(z), -t]])"


# This (random) test is XFAIL because it fails occasionally
# See https://github.com/sympy/sympy/issues/18469
@XFAIL
def test_tensorflow_math():
    if not tf:
        skip("TensorFlow not installed")

    expr = Abs(x)
    assert tensorflow_code(expr) == "tensorflow.math.abs(x)"
    _compare_tensorflow_scalar((x,), expr)

    expr = sign(x)
    assert tensorflow_code(expr) == "tensorflow.math.sign(x)"
    _compare_tensorflow_scalar((x,), expr)

    expr = ceiling(x)
    assert tensorflow_code(expr) == "tensorflow.math.ceil(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = floor(x)
    assert tensorflow_code(expr) == "tensorflow.math.floor(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = exp(x)
    assert tensorflow_code(expr) == "tensorflow.math.exp(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = sqrt(x)
    assert tensorflow_code(expr) == "tensorflow.math.sqrt(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = x ** 4
    assert tensorflow_code(expr) == "tensorflow.math.pow(x, 4)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = cos(x)
    assert tensorflow_code(expr) == "tensorflow.math.cos(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = acos(x)
    assert tensorflow_code(expr) == "tensorflow.math.acos(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(0, 0.95))

    expr = sin(x)
    assert tensorflow_code(expr) == "tensorflow.math.sin(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = asin(x)
    assert tensorflow_code(expr) == "tensorflow.math.asin(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = tan(x)
    assert tensorflow_code(expr) == "tensorflow.math.tan(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = atan(x)
    assert tensorflow_code(expr) == "tensorflow.math.atan(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = atan2(y, x)
    assert tensorflow_code(expr) == "tensorflow.math.atan2(y, x)"
    _compare_tensorflow_scalar((y, x), expr, rng=lambda: random.random())

    expr = cosh(x)
    assert tensorflow_code(expr) == "tensorflow.math.cosh(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.random())

    expr = acosh(x)
    assert tensorflow_code(expr) == "tensorflow.math.acosh(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    expr = sinh(x)
    assert tensorflow_code(expr) == "tensorflow.math.sinh(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    expr = asinh(x)
    assert tensorflow_code(expr) == "tensorflow.math.asinh(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    expr = tanh(x)
    assert tensorflow_code(expr) == "tensorflow.math.tanh(x)"
    _compare_tensorflow_scalar((x,), expr, rng=lambda: random.uniform(1, 2))

    expr = atanh(x)
    assert tensorflow_code(expr) == "tensorflow.math.atanh(x)"
    _compare_tensorflow_scalar(
        (x,), expr, rng=lambda: random.uniform(-.5, .5))

    expr = erf(x)
    assert tensorflow_code(expr) == "tensorflow.math.erf(x)"
    _compare_tensorflow_scalar(
        (x,), expr, rng=lambda: random.random())

    expr = loggamma(x)
    assert tensorflow_code(expr) == "tensorflow.math.lgamma(x)"
    _compare_tensorflow_scalar(
        (x,), expr, rng=lambda: random.random())


def test_tensorflow_complexes():
    assert tensorflow_code(re(x)) == "tensorflow.math.real(x)"
    assert tensorflow_code(im(x)) == "tensorflow.math.imag(x)"
    assert tensorflow_code(arg(x)) == "tensorflow.math.angle(x)"


def test_tensorflow_relational():
    if not tf:
        skip("TensorFlow not installed")

    expr = Eq(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.equal(x, y)"
    _compare_tensorflow_relational((x, y), expr)

    expr = Ne(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.not_equal(x, y)"
    _compare_tensorflow_relational((x, y), expr)

    expr = Ge(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.greater_equal(x, y)"
    _compare_tensorflow_relational((x, y), expr)

    expr = Gt(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.greater(x, y)"
    _compare_tensorflow_relational((x, y), expr)

    expr = Le(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.less_equal(x, y)"
    _compare_tensorflow_relational((x, y), expr)

    expr = Lt(x, y)
    assert tensorflow_code(expr) == "tensorflow.math.less(x, y)"
    _compare_tensorflow_relational((x, y), expr)


# This (random) test is XFAIL because it fails occasionally
# See https://github.com/sympy/sympy/issues/18469
@XFAIL
def test_tensorflow_matrices():
    if not tf:
        skip("TensorFlow not installed")

    expr = M
    assert tensorflow_code(expr) == "M"
    _compare_tensorflow_matrix((M,), expr)

    expr = M + N
    assert tensorflow_code(expr) == "tensorflow.math.add(M, N)"
    _compare_tensorflow_matrix((M, N), expr)

    expr = M * N
    assert tensorflow_code(expr) == "tensorflow.linalg.matmul(M, N)"
    _compare_tensorflow_matrix((M, N), expr)

    expr = HadamardProduct(M, N)
    assert tensorflow_code(expr) == "tensorflow.math.multiply(M, N)"
    _compare_tensorflow_matrix((M, N), expr)

    expr = M*N*P*Q
    assert tensorflow_code(expr) == \
        "tensorflow.linalg.matmul(" \
            "tensorflow.linalg.matmul(" \
                "tensorflow.linalg.matmul(M, N), P), Q)"
    _compare_tensorflow_matrix((M, N, P, Q), expr)

    expr = M**3
    assert tensorflow_code(expr) == \
        "tensorflow.linalg.matmul(tensorflow.linalg.matmul(M, M), M)"
    _compare_tensorflow_matrix((M,), expr)

    expr = Trace(M)
    assert tensorflow_code(expr) == "tensorflow.linalg.trace(M)"
    _compare_tensorflow_matrix((M,), expr)

    expr = Determinant(M)
    assert tensorflow_code(expr) == "tensorflow.linalg.det(M)"
    _compare_tensorflow_matrix_scalar((M,), expr)

    expr = Inverse(M)
    assert tensorflow_code(expr) == "tensorflow.linalg.inv(M)"
    _compare_tensorflow_matrix_inverse((M,), expr, use_float=True)

    expr = M.T
    assert tensorflow_code(expr, tensorflow_version='1.14') == \
        "tensorflow.linalg.matrix_transpose(M)"
    assert tensorflow_code(expr, tensorflow_version='1.13') == \
        "tensorflow.matrix_transpose(M)"

    _compare_tensorflow_matrix((M,), expr)


def test_codegen_einsum():
    if not tf:
        skip("TensorFlow not installed")

    graph = tf.Graph()
    with graph.as_default():
        session = tf.compat.v1.Session(graph=graph)

        M = MatrixSymbol("M", 2, 2)
        N = MatrixSymbol("N", 2, 2)

        cg = convert_matrix_to_array(M * N)
        f = lambdify((M, N), cg, 'tensorflow')

        ma = tf.constant([[1, 2], [3, 4]])
        mb = tf.constant([[1,-2], [-1, 3]])
        y = session.run(f(ma, mb))
        c = session.run(tf.matmul(ma, mb))
        assert (y == c).all()


def test_codegen_extra():
    if not tf:
        skip("TensorFlow not installed")

    graph = tf.Graph()
    with graph.as_default():
        session = tf.compat.v1.Session()

        M = MatrixSymbol("M", 2, 2)
        N = MatrixSymbol("N", 2, 2)
        P = MatrixSymbol("P", 2, 2)
        Q = MatrixSymbol("Q", 2, 2)
        ma = tf.constant([[1, 2], [3, 4]])
        mb = tf.constant([[1,-2], [-1, 3]])
        mc = tf.constant([[2, 0], [1, 2]])
        md = tf.constant([[1,-1], [4, 7]])

        cg = ArrayTensorProduct(M, N)
        assert tensorflow_code(cg) == \
            'tensorflow.linalg.einsum("ab,cd", M, N)'
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        c = session.run(tf.einsum("ij,kl", ma, mb))
        assert (y == c).all()

        cg = ArrayAdd(M, N)
        assert tensorflow_code(cg) == 'tensorflow.math.add(M, N)'
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        c = session.run(ma + mb)
        assert (y == c).all()

        cg = ArrayAdd(M, N, P)
        assert tensorflow_code(cg) == \
            'tensorflow.math.add(tensorflow.math.add(M, N), P)'
        f = lambdify((M, N, P), cg, 'tensorflow')
        y = session.run(f(ma, mb, mc))
        c = session.run(ma + mb + mc)
        assert (y == c).all()

        cg = ArrayAdd(M, N, P, Q)
        assert tensorflow_code(cg) == \
            'tensorflow.math.add(' \
                'tensorflow.math.add(tensorflow.math.add(M, N), P), Q)'
        f = lambdify((M, N, P, Q), cg, 'tensorflow')
        y = session.run(f(ma, mb, mc, md))
        c = session.run(ma + mb + mc + md)
        assert (y == c).all()

        cg = PermuteDims(M, [1, 0])
        assert tensorflow_code(cg) == 'tensorflow.transpose(M, [1, 0])'
        f = lambdify((M,), cg, 'tensorflow')
        y = session.run(f(ma))
        c = session.run(tf.transpose(ma))
        assert (y == c).all()

        cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
        assert tensorflow_code(cg) == \
            'tensorflow.transpose(' \
                'tensorflow.linalg.einsum("ab,cd", M, N), [1, 2, 3, 0])'
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        c = session.run(tf.transpose(tf.einsum("ab,cd", ma, mb), [1, 2, 3, 0]))
        assert (y == c).all()

        cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
        assert tensorflow_code(cg) == \
            'tensorflow.linalg.einsum("ab,bc->acb", M, N)'
        f = lambdify((M, N), cg, 'tensorflow')
        y = session.run(f(ma, mb))
        c = session.run(tf.einsum("ab,bc->acb", ma, mb))
        assert (y == c).all()


def test_MatrixElement_printing():
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert tensorflow_code(A[0, 0]) == "A[0, 0]"
    assert tensorflow_code(3 * A[0, 0]) == "3*A[0, 0]"

    F = C[0, 0].subs(C, A - B)
    assert tensorflow_code(F) == "(tensorflow.math.add((-1)*B, A))[0, 0]"


def test_tensorflow_Derivative():
    expr = Derivative(sin(x), x)
    assert tensorflow_code(expr) == \
        "tensorflow.gradients(tensorflow.math.sin(x), x)[0]"

def test_tensorflow_isnan_isinf():
    if not tf:
        skip("TensorFlow not installed")

    # Test for isnan
    x = symbols("x")
    # Return 0 if x is of nan value, and 1 otherwise
    expression = Piecewise((0.0, isnan(x)), (1.0, True))
    printed_code = tensorflow_code(expression)
    expected_printed_code = "tensorflow.where(tensorflow.math.is_nan(x), 0.0, 1.0)"
    assert tensorflow_code(expression) == expected_printed_code, f"Incorrect printed result {printed_code}, expected {expected_printed_code}"
    for _input, _expected in [(float('nan'), 0.0), (float('inf'), 1.0), (float('-inf'), 1.0), (1.0, 1.0)]:
        _output = lambdify((x), expression, modules="tensorflow")(x=tf.constant([_input]))
        assert (_output == _expected).numpy().all()

    # Test for isinf
    x = symbols("x")
    # Return 0 if x is of nan value, and 1 otherwise
    expression = Piecewise((0.0, isinf(x)), (1.0, True))
    printed_code = tensorflow_code(expression)
    expected_printed_code = "tensorflow.where(tensorflow.math.is_inf(x), 0.0, 1.0)"
    assert tensorflow_code(expression) == expected_printed_code, f"Incorrect printed result {printed_code}, expected {expected_printed_code}"
    for _input, _expected in [(float('inf'), 0.0), (float('-inf'), 0.0), (float('nan'), 1.0), (1.0, 1.0)]:
        _output = lambdify((x), expression, modules="tensorflow")(x=tf.constant([_input]))
        assert (_output == _expected).numpy().all()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tests\test_identitysearch.py ===
from sympy.external import import_module
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import (X, Y, Z, H, CNOT,
        IdentityGate, CGate, PhaseGate, TGate)
from sympy.physics.quantum.identitysearch import (generate_gate_rules,
        generate_equivalent_ids, GateIdentity, bfs_identity_search,
        is_scalar_sparse_matrix,
        is_scalar_nonsparse_matrix, is_degenerate, is_reducible)
from sympy.testing.pytest import skip


def create_gate_sequence(qubit=0):
    gates = (X(qubit), Y(qubit), Z(qubit), H(qubit))
    return gates


def test_generate_gate_rules_1():
    # Test with tuples
    (x, y, z, h) = create_gate_sequence()
    ph = PhaseGate(0)
    cgate_t = CGate(0, TGate(1))

    assert generate_gate_rules((x,)) == {((x,), ())}

    gate_rules = {((x, x), ()),
                      ((x,), (x,))}
    assert generate_gate_rules((x, x)) == gate_rules

    gate_rules = {((x, y, x), ()),
                      ((y, x, x), ()),
                      ((x, x, y), ()),
                      ((y, x), (x,)),
                      ((x, y), (x,)),
                      ((y,), (x, x))}
    assert generate_gate_rules((x, y, x)) == gate_rules

    gate_rules = {((x, y, z), ()), ((y, z, x), ()), ((z, x, y), ()),
                      ((), (x, z, y)), ((), (y, x, z)), ((), (z, y, x)),
                      ((x,), (z, y)), ((y, z), (x,)), ((y,), (x, z)),
                      ((z, x), (y,)), ((z,), (y, x)), ((x, y), (z,))}
    actual = generate_gate_rules((x, y, z))
    assert actual == gate_rules

    gate_rules = {
        ((), (h, z, y, x)), ((), (x, h, z, y)), ((), (y, x, h, z)),
         ((), (z, y, x, h)), ((h,), (z, y, x)), ((x,), (h, z, y)),
         ((y,), (x, h, z)), ((z,), (y, x, h)), ((h, x), (z, y)),
         ((x, y), (h, z)), ((y, z), (x, h)), ((z, h), (y, x)),
         ((h, x, y), (z,)), ((x, y, z), (h,)), ((y, z, h), (x,)),
         ((z, h, x), (y,)), ((h, x, y, z), ()), ((x, y, z, h), ()),
         ((y, z, h, x), ()), ((z, h, x, y), ())}
    actual = generate_gate_rules((x, y, z, h))
    assert actual == gate_rules

    gate_rules = {((), (cgate_t**(-1), ph**(-1), x)),
                      ((), (ph**(-1), x, cgate_t**(-1))),
                      ((), (x, cgate_t**(-1), ph**(-1))),
                      ((cgate_t,), (ph**(-1), x)),
                      ((ph,), (x, cgate_t**(-1))),
                      ((x,), (cgate_t**(-1), ph**(-1))),
                      ((cgate_t, x), (ph**(-1),)),
                      ((ph, cgate_t), (x,)),
                      ((x, ph), (cgate_t**(-1),)),
                      ((cgate_t, x, ph), ()),
                      ((ph, cgate_t, x), ()),
                      ((x, ph, cgate_t), ())}
    actual = generate_gate_rules((x, ph, cgate_t))
    assert actual == gate_rules

    gate_rules = {(Integer(1), cgate_t**(-1)*ph**(-1)*x),
                      (Integer(1), ph**(-1)*x*cgate_t**(-1)),
                      (Integer(1), x*cgate_t**(-1)*ph**(-1)),
                      (cgate_t, ph**(-1)*x),
                      (ph, x*cgate_t**(-1)),
                      (x, cgate_t**(-1)*ph**(-1)),
                      (cgate_t*x, ph**(-1)),
                      (ph*cgate_t, x),
                      (x*ph, cgate_t**(-1)),
                      (cgate_t*x*ph, Integer(1)),
                      (ph*cgate_t*x, Integer(1)),
                      (x*ph*cgate_t, Integer(1))}
    actual = generate_gate_rules((x, ph, cgate_t), return_as_muls=True)
    assert actual == gate_rules


def test_generate_gate_rules_2():
    # Test with Muls
    (x, y, z, h) = create_gate_sequence()
    ph = PhaseGate(0)
    cgate_t = CGate(0, TGate(1))

    # Note: 1 (type int) is not the same as 1 (type One)
    expected = {(x, Integer(1))}
    assert generate_gate_rules((x,), return_as_muls=True) == expected

    expected = {(Integer(1), Integer(1))}
    assert generate_gate_rules(x*x, return_as_muls=True) == expected

    expected = {((), ())}
    assert generate_gate_rules(x*x, return_as_muls=False) == expected

    gate_rules = {(x*y*x, Integer(1)),
                      (y, Integer(1)),
                      (y*x, x),
                      (x*y, x)}
    assert generate_gate_rules(x*y*x, return_as_muls=True) == gate_rules

    gate_rules = {(x*y*z, Integer(1)),
                      (y*z*x, Integer(1)),
                      (z*x*y, Integer(1)),
                      (Integer(1), x*z*y),
                      (Integer(1), y*x*z),
                      (Integer(1), z*y*x),
                      (x, z*y),
                      (y*z, x),
                      (y, x*z),
                      (z*x, y),
                      (z, y*x),
                      (x*y, z)}
    actual = generate_gate_rules(x*y*z, return_as_muls=True)
    assert actual == gate_rules

    gate_rules = {(Integer(1), h*z*y*x),
                      (Integer(1), x*h*z*y),
                      (Integer(1), y*x*h*z),
                      (Integer(1), z*y*x*h),
                      (h, z*y*x), (x, h*z*y),
                      (y, x*h*z), (z, y*x*h),
                      (h*x, z*y), (z*h, y*x),
                      (x*y, h*z), (y*z, x*h),
                      (h*x*y, z), (x*y*z, h),
                      (y*z*h, x), (z*h*x, y),
                      (h*x*y*z, Integer(1)),
                      (x*y*z*h, Integer(1)),
                      (y*z*h*x, Integer(1)),
                      (z*h*x*y, Integer(1))}
    actual = generate_gate_rules(x*y*z*h, return_as_muls=True)
    assert actual == gate_rules

    gate_rules = {(Integer(1), cgate_t**(-1)*ph**(-1)*x),
                      (Integer(1), ph**(-1)*x*cgate_t**(-1)),
                      (Integer(1), x*cgate_t**(-1)*ph**(-1)),
                      (cgate_t, ph**(-1)*x),
                      (ph, x*cgate_t**(-1)),
                      (x, cgate_t**(-1)*ph**(-1)),
                      (cgate_t*x, ph**(-1)),
                      (ph*cgate_t, x),
                      (x*ph, cgate_t**(-1)),
                      (cgate_t*x*ph, Integer(1)),
                      (ph*cgate_t*x, Integer(1)),
                      (x*ph*cgate_t, Integer(1))}
    actual = generate_gate_rules(x*ph*cgate_t, return_as_muls=True)
    assert actual == gate_rules

    gate_rules = {((), (cgate_t**(-1), ph**(-1), x)),
                      ((), (ph**(-1), x, cgate_t**(-1))),
                      ((), (x, cgate_t**(-1), ph**(-1))),
                      ((cgate_t,), (ph**(-1), x)),
                      ((ph,), (x, cgate_t**(-1))),
                      ((x,), (cgate_t**(-1), ph**(-1))),
                      ((cgate_t, x), (ph**(-1),)),
                      ((ph, cgate_t), (x,)),
                      ((x, ph), (cgate_t**(-1),)),
                      ((cgate_t, x, ph), ()),
                      ((ph, cgate_t, x), ()),
                      ((x, ph, cgate_t), ())}
    actual = generate_gate_rules(x*ph*cgate_t)
    assert actual == gate_rules


def test_generate_equivalent_ids_1():
    # Test with tuples
    (x, y, z, h) = create_gate_sequence()

    assert generate_equivalent_ids((x,)) == {(x,)}
    assert generate_equivalent_ids((x, x)) == {(x, x)}
    assert generate_equivalent_ids((x, y)) == {(x, y), (y, x)}

    gate_seq = (x, y, z)
    gate_ids = {(x, y, z), (y, z, x), (z, x, y), (z, y, x),
                    (y, x, z), (x, z, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    gate_ids = {Mul(x, y, z), Mul(y, z, x), Mul(z, x, y),
                    Mul(z, y, x), Mul(y, x, z), Mul(x, z, y)}
    assert generate_equivalent_ids(gate_seq, return_as_muls=True) == gate_ids

    gate_seq = (x, y, z, h)
    gate_ids = {(x, y, z, h), (y, z, h, x),
                    (h, x, y, z), (h, z, y, x),
                    (z, y, x, h), (y, x, h, z),
                    (z, h, x, y), (x, h, z, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    gate_seq = (x, y, x, y)
    gate_ids = {(x, y, x, y), (y, x, y, x)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    cgate_y = CGate((1,), y)
    gate_seq = (y, cgate_y, y, cgate_y)
    gate_ids = {(y, cgate_y, y, cgate_y), (cgate_y, y, cgate_y, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))
    gate_seq = (cnot, h, cgate_z, h)
    gate_ids = {(cnot, h, cgate_z, h), (h, cgate_z, h, cnot),
                    (h, cnot, h, cgate_z), (cgate_z, h, cnot, h)}
    assert generate_equivalent_ids(gate_seq) == gate_ids


def test_generate_equivalent_ids_2():
    # Test with Muls
    (x, y, z, h) = create_gate_sequence()

    assert generate_equivalent_ids((x,), return_as_muls=True) == {x}

    gate_ids = {Integer(1)}
    assert generate_equivalent_ids(x*x, return_as_muls=True) == gate_ids

    gate_ids = {x*y, y*x}
    assert generate_equivalent_ids(x*y, return_as_muls=True) == gate_ids

    gate_ids = {(x, y), (y, x)}
    assert generate_equivalent_ids(x*y) == gate_ids

    circuit = Mul(*(x, y, z))
    gate_ids = {x*y*z, y*z*x, z*x*y, z*y*x,
                    y*x*z, x*z*y}
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    circuit = Mul(*(x, y, z, h))
    gate_ids = {x*y*z*h, y*z*h*x,
                    h*x*y*z, h*z*y*x,
                    z*y*x*h, y*x*h*z,
                    z*h*x*y, x*h*z*y}
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    circuit = Mul(*(x, y, x, y))
    gate_ids = {x*y*x*y, y*x*y*x}
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    cgate_y = CGate((1,), y)
    circuit = Mul(*(y, cgate_y, y, cgate_y))
    gate_ids = {y*cgate_y*y*cgate_y, cgate_y*y*cgate_y*y}
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))
    circuit = Mul(*(cnot, h, cgate_z, h))
    gate_ids = {cnot*h*cgate_z*h, h*cgate_z*h*cnot,
                    h*cnot*h*cgate_z, cgate_z*h*cnot*h}
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids


def test_is_scalar_nonsparse_matrix():
    numqubits = 2
    id_only = False

    id_gate = (IdentityGate(1),)
    actual = is_scalar_nonsparse_matrix(id_gate, numqubits, id_only)
    assert actual is True

    x0 = X(0)
    xx_circuit = (x0, x0)
    actual = is_scalar_nonsparse_matrix(xx_circuit, numqubits, id_only)
    assert actual is True

    x1 = X(1)
    y1 = Y(1)
    xy_circuit = (x1, y1)
    actual = is_scalar_nonsparse_matrix(xy_circuit, numqubits, id_only)
    assert actual is False

    z1 = Z(1)
    xyz_circuit = (x1, y1, z1)
    actual = is_scalar_nonsparse_matrix(xyz_circuit, numqubits, id_only)
    assert actual is True

    cnot = CNOT(1, 0)
    cnot_circuit = (cnot, cnot)
    actual = is_scalar_nonsparse_matrix(cnot_circuit, numqubits, id_only)
    assert actual is True

    h = H(0)
    hh_circuit = (h, h)
    actual = is_scalar_nonsparse_matrix(hh_circuit, numqubits, id_only)
    assert actual is True

    h1 = H(1)
    xhzh_circuit = (x1, h1, z1, h1)
    actual = is_scalar_nonsparse_matrix(xhzh_circuit, numqubits, id_only)
    assert actual is True

    id_only = True
    actual = is_scalar_nonsparse_matrix(xhzh_circuit, numqubits, id_only)
    assert actual is True
    actual = is_scalar_nonsparse_matrix(xyz_circuit, numqubits, id_only)
    assert actual is False
    actual = is_scalar_nonsparse_matrix(cnot_circuit, numqubits, id_only)
    assert actual is True
    actual = is_scalar_nonsparse_matrix(hh_circuit, numqubits, id_only)
    assert actual is True


def test_is_scalar_sparse_matrix():
    np = import_module('numpy')
    if not np:
        skip("numpy not installed.")

    scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
    if not scipy:
        skip("scipy not installed.")

    numqubits = 2
    id_only = False

    id_gate = (IdentityGate(1),)
    assert is_scalar_sparse_matrix(id_gate, numqubits, id_only) is True

    x0 = X(0)
    xx_circuit = (x0, x0)
    assert is_scalar_sparse_matrix(xx_circuit, numqubits, id_only) is True

    x1 = X(1)
    y1 = Y(1)
    xy_circuit = (x1, y1)
    assert is_scalar_sparse_matrix(xy_circuit, numqubits, id_only) is False

    z1 = Z(1)
    xyz_circuit = (x1, y1, z1)
    assert is_scalar_sparse_matrix(xyz_circuit, numqubits, id_only) is True

    cnot = CNOT(1, 0)
    cnot_circuit = (cnot, cnot)
    assert is_scalar_sparse_matrix(cnot_circuit, numqubits, id_only) is True

    h = H(0)
    hh_circuit = (h, h)
    assert is_scalar_sparse_matrix(hh_circuit, numqubits, id_only) is True

    # NOTE:
    # The elements of the sparse matrix for the following circuit
    # is actually 1.0000000000000002+0.0j.
    h1 = H(1)
    xhzh_circuit = (x1, h1, z1, h1)
    assert is_scalar_sparse_matrix(xhzh_circuit, numqubits, id_only) is True

    id_only = True
    assert is_scalar_sparse_matrix(xhzh_circuit, numqubits, id_only) is True
    assert is_scalar_sparse_matrix(xyz_circuit, numqubits, id_only) is False
    assert is_scalar_sparse_matrix(cnot_circuit, numqubits, id_only) is True
    assert is_scalar_sparse_matrix(hh_circuit, numqubits, id_only) is True


def test_is_degenerate():
    (x, y, z, h) = create_gate_sequence()

    gate_id = GateIdentity(x, y, z)
    ids = {gate_id}

    another_id = (z, y, x)
    assert is_degenerate(ids, another_id) is True


def test_is_reducible():
    nqubits = 2
    (x, y, z, h) = create_gate_sequence()

    circuit = (x, y, y)
    assert is_reducible(circuit, nqubits, 1, 3) is True

    circuit = (x, y, x)
    assert is_reducible(circuit, nqubits, 1, 3) is False

    circuit = (x, y, y, x)
    assert is_reducible(circuit, nqubits, 0, 4) is True

    circuit = (x, y, y, x)
    assert is_reducible(circuit, nqubits, 1, 3) is True

    circuit = (x, y, z, y, y)
    assert is_reducible(circuit, nqubits, 1, 5) is True


def test_bfs_identity_search():
    assert bfs_identity_search([], 1) == set()

    (x, y, z, h) = create_gate_sequence()

    gate_list = [x]
    id_set = {GateIdentity(x, x)}
    assert bfs_identity_search(gate_list, 1, max_depth=2) == id_set

    # Set should not contain degenerate quantum circuits
    gate_list = [x, y, z]
    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(x, y, z)}
    assert bfs_identity_search(gate_list, 1) == id_set

    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(x, y, z),
                  GateIdentity(x, y, x, y),
                  GateIdentity(x, z, x, z),
                  GateIdentity(y, z, y, z)}
    assert bfs_identity_search(gate_list, 1, max_depth=4) == id_set
    assert bfs_identity_search(gate_list, 1, max_depth=5) == id_set

    gate_list = [x, y, z, h]
    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(h, h),
                  GateIdentity(x, y, z),
                  GateIdentity(x, y, x, y),
                  GateIdentity(x, z, x, z),
                  GateIdentity(x, h, z, h),
                  GateIdentity(y, z, y, z),
                  GateIdentity(y, h, y, h)}
    assert bfs_identity_search(gate_list, 1) == id_set

    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(h, h)}
    assert id_set == bfs_identity_search(gate_list, 1, max_depth=3,
                                         identity_only=True)

    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(h, h),
                  GateIdentity(x, y, z),
                  GateIdentity(x, y, x, y),
                  GateIdentity(x, z, x, z),
                  GateIdentity(x, h, z, h),
                  GateIdentity(y, z, y, z),
                  GateIdentity(y, h, y, h),
                  GateIdentity(x, y, h, x, h),
                  GateIdentity(x, z, h, y, h),
                  GateIdentity(y, z, h, z, h)}
    assert bfs_identity_search(gate_list, 1, max_depth=5) == id_set

    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(h, h),
                  GateIdentity(x, h, z, h)}
    assert id_set == bfs_identity_search(gate_list, 1, max_depth=4,
                                         identity_only=True)

    cnot = CNOT(1, 0)
    gate_list = [x, cnot]
    id_set = {GateIdentity(x, x),
                  GateIdentity(cnot, cnot),
                  GateIdentity(x, cnot, x, cnot)}
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set

    cgate_x = CGate((1,), x)
    gate_list = [x, cgate_x]
    id_set = {GateIdentity(x, x),
                  GateIdentity(cgate_x, cgate_x),
                  GateIdentity(x, cgate_x, x, cgate_x)}
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set

    cgate_z = CGate((0,), Z(1))
    gate_list = [cnot, cgate_z, h]
    id_set = {GateIdentity(h, h),
                  GateIdentity(cgate_z, cgate_z),
                  GateIdentity(cnot, cnot),
                  GateIdentity(cnot, h, cgate_z, h)}
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set

    s = PhaseGate(0)
    t = TGate(0)
    gate_list = [s, t]
    id_set = {GateIdentity(s, s, s, s)}
    assert bfs_identity_search(gate_list, 1, max_depth=4) == id_set


def test_bfs_identity_search_xfail():
    s = PhaseGate(0)
    t = TGate(0)
    gate_list = [Dagger(s), t]
    id_set = {GateIdentity(Dagger(s), t, t)}
    assert bfs_identity_search(gate_list, 1, max_depth=3) == id_set

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_fu.py ===
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.parameters import evaluate
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.hyperbolic import (cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, cot, csc, sec, sin, tan)
from sympy.simplify.powsimp import powsimp
from sympy.simplify.fu import (
    L, TR1, TR10, TR10i, TR11, _TR11, TR12, TR12i, TR13, TR14, TR15, TR16,
    TR111, TR2, TR2i, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TRmorrie, _TR56 as T,
    TRpower, hyper_as_trig, fu, process_common_addends, trig_split,
    as_f_sign_1)
from sympy.core.random import verify_numerically
from sympy.abc import a, b, c, x, y, z


def test_TR1():
    assert TR1(2*csc(x) + sec(x)) == 1/cos(x) + 2/sin(x)


def test_TR2():
    assert TR2(tan(x)) == sin(x)/cos(x)
    assert TR2(cot(x)) == cos(x)/sin(x)
    assert TR2(tan(tan(x) - sin(x)/cos(x))) == 0


def test_TR2i():
    # just a reminder that ratios of powers only simplify if both
    # numerator and denominator satisfy the condition that each
    # has a positive base or an integer exponent; e.g. the following,
    # at y=-1, x=1/2 gives sqrt(2)*I != -sqrt(2)*I
    assert powsimp(2**x/y**x) != (2/y)**x

    assert TR2i(sin(x)/cos(x)) == tan(x)
    assert TR2i(sin(x)*sin(y)/cos(x)) == tan(x)*sin(y)
    assert TR2i(1/(sin(x)/cos(x))) == 1/tan(x)
    assert TR2i(1/(sin(x)*sin(y)/cos(x))) == 1/tan(x)/sin(y)
    assert TR2i(sin(x)/2/(cos(x) + 1)) == sin(x)/(cos(x) + 1)/2

    assert TR2i(sin(x)/2/(cos(x) + 1), half=True) == tan(x/2)/2
    assert TR2i(sin(1)/(cos(1) + 1), half=True) == tan(S.Half)
    assert TR2i(sin(2)/(cos(2) + 1), half=True) == tan(1)
    assert TR2i(sin(4)/(cos(4) + 1), half=True) == tan(2)
    assert TR2i(sin(5)/(cos(5) + 1), half=True) == tan(5*S.Half)
    assert TR2i((cos(1) + 1)/sin(1), half=True) == 1/tan(S.Half)
    assert TR2i((cos(2) + 1)/sin(2), half=True) == 1/tan(1)
    assert TR2i((cos(4) + 1)/sin(4), half=True) == 1/tan(2)
    assert TR2i((cos(5) + 1)/sin(5), half=True) == 1/tan(5*S.Half)
    assert TR2i((cos(1) + 1)**(-a)*sin(1)**a, half=True) == tan(S.Half)**a
    assert TR2i((cos(2) + 1)**(-a)*sin(2)**a, half=True) == tan(1)**a
    assert TR2i((cos(4) + 1)**(-a)*sin(4)**a, half=True) == (cos(4) + 1)**(-a)*sin(4)**a
    assert TR2i((cos(5) + 1)**(-a)*sin(5)**a, half=True) == (cos(5) + 1)**(-a)*sin(5)**a
    assert TR2i((cos(1) + 1)**a*sin(1)**(-a), half=True) == tan(S.Half)**(-a)
    assert TR2i((cos(2) + 1)**a*sin(2)**(-a), half=True) == tan(1)**(-a)
    assert TR2i((cos(4) + 1)**a*sin(4)**(-a), half=True) == (cos(4) + 1)**a*sin(4)**(-a)
    assert TR2i((cos(5) + 1)**a*sin(5)**(-a), half=True) == (cos(5) + 1)**a*sin(5)**(-a)

    i = symbols('i', integer=True)
    assert TR2i(((cos(5) + 1)**i*sin(5)**(-i)), half=True) == tan(5*S.Half)**(-i)
    assert TR2i(1/((cos(5) + 1)**i*sin(5)**(-i)), half=True) == tan(5*S.Half)**i


def test_TR3():
    assert TR3(cos(y - x*(y - x))) == cos(x*(x - y) + y)
    assert cos(pi/2 + x) == -sin(x)
    assert cos(30*pi/2 + x) == -cos(x)

    for f in (cos, sin, tan, cot, csc, sec):
        i = f(pi*Rational(3, 7))
        j = TR3(i)
        assert verify_numerically(i, j) and i.func != j.func

    with evaluate(False):
        eq = cos(9*pi/22)
    assert eq.has(9*pi) and TR3(eq) == sin(pi/11)


def test_TR4():
    for i in [0, pi/6, pi/4, pi/3, pi/2]:
        with evaluate(False):
            eq = cos(i)
        assert isinstance(eq, cos) and TR4(eq) == cos(i)


def test__TR56():
    h = lambda x: 1 - x
    assert T(sin(x)**3, sin, cos, h, 4, False) == sin(x)*(-cos(x)**2 + 1)
    assert T(sin(x)**10, sin, cos, h, 4, False) == sin(x)**10
    assert T(sin(x)**6, sin, cos, h, 6, False) == (-cos(x)**2 + 1)**3
    assert T(sin(x)**6, sin, cos, h, 6, True) == sin(x)**6
    assert T(sin(x)**8, sin, cos, h, 10, True) == (-cos(x)**2 + 1)**4

    # issue 17137
    assert T(sin(x)**I, sin, cos, h, 4, True) == sin(x)**I
    assert T(sin(x)**(2*I + 1), sin, cos, h, 4, True) == sin(x)**(2*I + 1)


def test_TR5():
    assert TR5(sin(x)**2) == -cos(x)**2 + 1
    assert TR5(sin(x)**-2) == sin(x)**(-2)
    assert TR5(sin(x)**4) == (-cos(x)**2 + 1)**2


def test_TR6():
    assert TR6(cos(x)**2) == -sin(x)**2 + 1
    assert TR6(cos(x)**-2) == cos(x)**(-2)
    assert TR6(cos(x)**4) == (-sin(x)**2 + 1)**2


def test_TR7():
    assert TR7(cos(x)**2) == cos(2*x)/2 + S.Half
    assert TR7(cos(x)**2 + 1) == cos(2*x)/2 + Rational(3, 2)


def test_TR8():
    assert TR8(cos(2)*cos(3)) == cos(5)/2 + cos(1)/2
    assert TR8(cos(2)*sin(3)) == sin(5)/2 + sin(1)/2
    assert TR8(sin(2)*sin(3)) == -cos(5)/2 + cos(1)/2
    assert TR8(sin(1)*sin(2)*sin(3)) == sin(4)/4 - sin(6)/4 + sin(2)/4
    assert TR8(cos(2)*cos(3)*cos(4)*cos(5)) == \
        cos(4)/4 + cos(10)/8 + cos(2)/8 + cos(8)/8 + cos(14)/8 + \
        cos(6)/8 + Rational(1, 8)
    assert TR8(cos(2)*cos(3)*cos(4)*cos(5)*cos(6)) == \
        cos(10)/8 + cos(4)/8 + 3*cos(2)/16 + cos(16)/16 + cos(8)/8 + \
        cos(14)/16 + cos(20)/16 + cos(12)/16 + Rational(1, 16) + cos(6)/8
    assert TR8(sin(pi*Rational(3, 7))**2*cos(pi*Rational(3, 7))**2/(16*sin(pi/7)**2)) == Rational(1, 64)

def test_TR9():
    a = S.Half
    b = 3*a
    assert TR9(a) == a
    assert TR9(cos(1) + cos(2)) == 2*cos(a)*cos(b)
    assert TR9(cos(1) - cos(2)) == 2*sin(a)*sin(b)
    assert TR9(sin(1) - sin(2)) == -2*sin(a)*cos(b)
    assert TR9(sin(1) + sin(2)) == 2*sin(b)*cos(a)
    assert TR9(cos(1) + 2*sin(1) + 2*sin(2)) == cos(1) + 4*sin(b)*cos(a)
    assert TR9(cos(4) + cos(2) + 2*cos(1)*cos(3)) == 4*cos(1)*cos(3)
    assert TR9((cos(4) + cos(2))/cos(3)/2 + cos(3)) == 2*cos(1)*cos(2)
    assert TR9(cos(3) + cos(4) + cos(5) + cos(6)) == \
        4*cos(S.Half)*cos(1)*cos(Rational(9, 2))
    assert TR9(cos(3) + cos(3)*cos(2)) == cos(3) + cos(2)*cos(3)
    assert TR9(-cos(y) + cos(x*y)) == -2*sin(x*y/2 - y/2)*sin(x*y/2 + y/2)
    assert TR9(-sin(y) + sin(x*y)) == 2*sin(x*y/2 - y/2)*cos(x*y/2 + y/2)
    c = cos(x)
    s = sin(x)
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for a in ((c, s), (s, c), (cos(x), cos(x*y)), (sin(x), sin(x*y))):
            args = zip(si, a)
            ex = Add(*[Mul(*ai) for ai in args])
            t = TR9(ex)
            assert not (a[0].func == a[1].func and (
                not verify_numerically(ex, t.expand(trig=True)) or t.is_Add)
                or a[1].func != a[0].func and ex != t)


def test_TR10():
    assert TR10(cos(a + b)) == -sin(a)*sin(b) + cos(a)*cos(b)
    assert TR10(sin(a + b)) == sin(a)*cos(b) + sin(b)*cos(a)
    assert TR10(sin(a + b + c)) == \
        (-sin(a)*sin(b) + cos(a)*cos(b))*sin(c) + \
        (sin(a)*cos(b) + sin(b)*cos(a))*cos(c)
    assert TR10(cos(a + b + c)) == \
        (-sin(a)*sin(b) + cos(a)*cos(b))*cos(c) - \
        (sin(a)*cos(b) + sin(b)*cos(a))*sin(c)


def test_TR10i():
    assert TR10i(cos(1)*cos(3) + sin(1)*sin(3)) == cos(2)
    assert TR10i(cos(1)*cos(3) - sin(1)*sin(3)) == cos(4)
    assert TR10i(cos(1)*sin(3) - sin(1)*cos(3)) == sin(2)
    assert TR10i(cos(1)*sin(3) + sin(1)*cos(3)) == sin(4)
    assert TR10i(cos(1)*sin(3) + sin(1)*cos(3) + 7) == sin(4) + 7
    assert TR10i(cos(1)*sin(3) + sin(1)*cos(3) + cos(3)) == cos(3) + sin(4)
    assert TR10i(2*cos(1)*sin(3) + 2*sin(1)*cos(3) + cos(3)) == \
        2*sin(4) + cos(3)
    assert TR10i(cos(2)*cos(3) + sin(2)*(cos(1)*sin(2) + cos(2)*sin(1))) == \
        cos(1)
    eq = (cos(2)*cos(3) + sin(2)*(
        cos(1)*sin(2) + cos(2)*sin(1)))*cos(5) + sin(1)*sin(5)
    assert TR10i(eq) == TR10i(eq.expand()) == cos(4)
    assert TR10i(sqrt(2)*cos(x)*x + sqrt(6)*sin(x)*x) == \
        2*sqrt(2)*x*sin(x + pi/6)
    assert TR10i(cos(x)/sqrt(6) + sin(x)/sqrt(2) +
            cos(x)/sqrt(6)/3 + sin(x)/sqrt(2)/3) == 4*sqrt(6)*sin(x + pi/6)/9
    assert TR10i(cos(x)/sqrt(6) + sin(x)/sqrt(2) +
            cos(y)/sqrt(6)/3 + sin(y)/sqrt(2)/3) == \
        sqrt(6)*sin(x + pi/6)/3 + sqrt(6)*sin(y + pi/6)/9
    assert TR10i(cos(x) + sqrt(3)*sin(x) + 2*sqrt(3)*cos(x + pi/6)) == 4*cos(x)
    assert TR10i(cos(x) + sqrt(3)*sin(x) +
            2*sqrt(3)*cos(x + pi/6) + 4*sin(x)) == 4*sqrt(2)*sin(x + pi/4)
    assert TR10i(cos(2)*sin(3) + sin(2)*cos(4)) == \
        sin(2)*cos(4) + sin(3)*cos(2)

    A = Symbol('A', commutative=False)
    assert TR10i(sqrt(2)*cos(x)*A + sqrt(6)*sin(x)*A) == \
        2*sqrt(2)*sin(x + pi/6)*A


    c = cos(x)
    s = sin(x)
    h = sin(y)
    r = cos(y)
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for argsi in ((c*r, s*h), (c*h, s*r)): # explicit 2-args
            args = zip(si, argsi)
            ex = Add(*[Mul(*ai) for ai in args])
            t = TR10i(ex)
            assert not (ex - t.expand(trig=True) or t.is_Add)

    c = cos(x)
    s = sin(x)
    h = sin(pi/6)
    r = cos(pi/6)
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for argsi in ((c*r, s*h), (c*h, s*r)): # induced
            args = zip(si, argsi)
            ex = Add(*[Mul(*ai) for ai in args])
            t = TR10i(ex)
            assert not (ex - t.expand(trig=True) or t.is_Add)


def test_TR11():

    assert TR11(sin(2*x)) == 2*sin(x)*cos(x)
    assert TR11(sin(4*x)) == 4*((-sin(x)**2 + cos(x)**2)*sin(x)*cos(x))
    assert TR11(sin(x*Rational(4, 3))) == \
        4*((-sin(x/3)**2 + cos(x/3)**2)*sin(x/3)*cos(x/3))

    assert TR11(cos(2*x)) == -sin(x)**2 + cos(x)**2
    assert TR11(cos(4*x)) == \
        (-sin(x)**2 + cos(x)**2)**2 - 4*sin(x)**2*cos(x)**2

    assert TR11(cos(2)) == cos(2)

    assert TR11(cos(pi*Rational(3, 7)), pi*Rational(2, 7)) == -cos(pi*Rational(2, 7))**2 + sin(pi*Rational(2, 7))**2
    assert TR11(cos(4), 2) == -sin(2)**2 + cos(2)**2
    assert TR11(cos(6), 2) == cos(6)
    assert TR11(sin(x)/cos(x/2), x/2) == 2*sin(x/2)

def test__TR11():

    assert _TR11(sin(x/3)*sin(2*x)*sin(x/4)/(cos(x/6)*cos(x/8))) == \
        4*sin(x/8)*sin(x/6)*sin(2*x),_TR11(sin(x/3)*sin(2*x)*sin(x/4)/(cos(x/6)*cos(x/8)))
    assert _TR11(sin(x/3)/cos(x/6)) == 2*sin(x/6)

    assert _TR11(cos(x/6)/sin(x/3)) == 1/(2*sin(x/6))
    assert _TR11(sin(2*x)*cos(x/8)/sin(x/4)) == sin(2*x)/(2*sin(x/8)), _TR11(sin(2*x)*cos(x/8)/sin(x/4))
    assert _TR11(sin(x)/sin(x/2)) == 2*cos(x/2)


def test_TR12():
    assert TR12(tan(x + y)) == (tan(x) + tan(y))/(-tan(x)*tan(y) + 1)
    assert TR12(tan(x + y + z)) ==\
        (tan(z) + (tan(x) + tan(y))/(-tan(x)*tan(y) + 1))/(
        1 - (tan(x) + tan(y))*tan(z)/(-tan(x)*tan(y) + 1))
    assert TR12(tan(x*y)) == tan(x*y)


def test_TR13():
    assert TR13(tan(3)*tan(2)) == -tan(2)/tan(5) - tan(3)/tan(5) + 1
    assert TR13(cot(3)*cot(2)) == 1 + cot(3)*cot(5) + cot(2)*cot(5)
    assert TR13(tan(1)*tan(2)*tan(3)) == \
        (-tan(2)/tan(5) - tan(3)/tan(5) + 1)*tan(1)
    assert TR13(tan(1)*tan(2)*cot(3)) == \
        (-tan(2)/tan(3) + 1 - tan(1)/tan(3))*cot(3)


def test_L():
    assert L(cos(x) + sin(x)) == 2


def test_fu():

    assert fu(sin(50)**2 + cos(50)**2 + sin(pi/6)) == Rational(3, 2)
    assert fu(sqrt(6)*cos(x) + sqrt(2)*sin(x)) == 2*sqrt(2)*sin(x + pi/3)


    eq = sin(x)**4 - cos(y)**2 + sin(y)**2 + 2*cos(x)**2
    assert fu(eq) == cos(x)**4 - 2*cos(y)**2 + 2

    assert fu(S.Half - cos(2*x)/2) == sin(x)**2

    assert fu(sin(a)*(cos(b) - sin(b)) + cos(a)*(sin(b) + cos(b))) == \
        sqrt(2)*sin(a + b + pi/4)

    assert fu(sqrt(3)*cos(x)/2 + sin(x)/2) == sin(x + pi/3)

    assert fu(1 - sin(2*x)**2/4 - sin(y)**2 - cos(x)**4) == \
        -cos(x)**2 + cos(y)**2

    assert fu(cos(pi*Rational(4, 9))) == sin(pi/18)
    assert fu(cos(pi/9)*cos(pi*Rational(2, 9))*cos(pi*Rational(3, 9))*cos(pi*Rational(4, 9))) == Rational(1, 16)

    assert fu(
        tan(pi*Rational(7, 18)) + tan(pi*Rational(5, 18)) - sqrt(3)*tan(pi*Rational(5, 18))*tan(pi*Rational(7, 18))) == \
        -sqrt(3)

    assert fu(tan(1)*tan(2)) == tan(1)*tan(2)

    expr = Mul(*[cos(2**i) for i in range(10)])
    assert fu(expr) == sin(1024)/(1024*sin(1))

    # issue #18059:
    assert fu(cos(x) + sqrt(sin(x)**2)) == cos(x) + sqrt(sin(x)**2)

    assert fu((-14*sin(x)**3 + 35*sin(x) + 6*sqrt(3)*cos(x)**3 + 9*sqrt(3)*cos(x))/((cos(2*x) + 4))) == \
        7*sin(x) + 3*sqrt(3)*cos(x)


def test_objective():
    assert fu(sin(x)/cos(x), measure=lambda x: x.count_ops()) == \
            tan(x)
    assert fu(sin(x)/cos(x), measure=lambda x: -x.count_ops()) == \
            sin(x)/cos(x)


def test_process_common_addends():
    # this tests that the args are not evaluated as they are given to do
    # and that key2 works when key1 is False
    do = lambda x: Add(*[i**(i%2) for i in x.args])
    assert process_common_addends(Add(*[1, 2, 3, 4], evaluate=False), do,
        key2=lambda x: x%2, key1=False) == 1**1 + 3**1 + 2**0 + 4**0


def test_trig_split():
    assert trig_split(cos(x), cos(y)) == (1, 1, 1, x, y, True)
    assert trig_split(2*cos(x), -2*cos(y)) == (2, 1, -1, x, y, True)
    assert trig_split(cos(x)*sin(y), cos(y)*sin(y)) == \
        (sin(y), 1, 1, x, y, True)

    assert trig_split(cos(x), -sqrt(3)*sin(x), two=True) == \
        (2, 1, -1, x, pi/6, False)
    assert trig_split(cos(x), sin(x), two=True) == \
        (sqrt(2), 1, 1, x, pi/4, False)
    assert trig_split(cos(x), -sin(x), two=True) == \
        (sqrt(2), 1, -1, x, pi/4, False)
    assert trig_split(sqrt(2)*cos(x), -sqrt(6)*sin(x), two=True) == \
        (2*sqrt(2), 1, -1, x, pi/6, False)
    assert trig_split(-sqrt(6)*cos(x), -sqrt(2)*sin(x), two=True) == \
        (-2*sqrt(2), 1, 1, x, pi/3, False)
    assert trig_split(cos(x)/sqrt(6), sin(x)/sqrt(2), two=True) == \
        (sqrt(6)/3, 1, 1, x, pi/6, False)
    assert trig_split(-sqrt(6)*cos(x)*sin(y),
            -sqrt(2)*sin(x)*sin(y), two=True) == \
        (-2*sqrt(2)*sin(y), 1, 1, x, pi/3, False)

    assert trig_split(cos(x), sin(x)) is None
    assert trig_split(cos(x), sin(z)) is None
    assert trig_split(2*cos(x), -sin(x)) is None
    assert trig_split(cos(x), -sqrt(3)*sin(x)) is None
    assert trig_split(cos(x)*cos(y), sin(x)*sin(z)) is None
    assert trig_split(cos(x)*cos(y), sin(x)*sin(y)) is None
    assert trig_split(-sqrt(6)*cos(x), sqrt(2)*sin(x)*sin(y), two=True) is \
        None

    assert trig_split(sqrt(3)*sqrt(x), cos(3), two=True) is None
    assert trig_split(sqrt(3)*root(x, 3), sin(3)*cos(2), two=True) is None
    assert trig_split(cos(5)*cos(6), cos(7)*sin(5), two=True) is None


def test_TRmorrie():
    assert TRmorrie(7*Mul(*[cos(i) for i in range(10)])) == \
        7*sin(12)*sin(16)*cos(5)*cos(7)*cos(9)/(64*sin(1)*sin(3))
    assert TRmorrie(x) == x
    assert TRmorrie(2*x) == 2*x
    e = cos(pi/7)*cos(pi*Rational(2, 7))*cos(pi*Rational(4, 7))
    assert TR8(TRmorrie(e)) == Rational(-1, 8)
    e = Mul(*[cos(2**i*pi/17) for i in range(1, 17)])
    assert TR8(TR3(TRmorrie(e))) == Rational(1, 65536)
    # issue 17063
    eq = cos(x)/cos(x/2)
    assert TRmorrie(eq) == eq
    # issue #20430
    eq = cos(x/2)*sin(x/2)*cos(x)**3
    assert TRmorrie(eq) == sin(2*x)*cos(x)**2/4


def test_TRpower():
    assert TRpower(1/sin(x)**2) == 1/sin(x)**2
    assert TRpower(cos(x)**3*sin(x/2)**4) == \
        (3*cos(x)/4 + cos(3*x)/4)*(-cos(x)/2 + cos(2*x)/8 + Rational(3, 8))
    for k in range(2, 8):
        assert verify_numerically(sin(x)**k, TRpower(sin(x)**k))
        assert verify_numerically(cos(x)**k, TRpower(cos(x)**k))


def test_hyper_as_trig():
    from sympy.simplify.fu import _osborne, _osbornei

    eq = sinh(x)**2 + cosh(x)**2
    t, f = hyper_as_trig(eq)
    assert f(fu(t)) == cosh(2*x)
    e, f = hyper_as_trig(tanh(x + y))
    assert f(TR12(e)) == (tanh(x) + tanh(y))/(tanh(x)*tanh(y) + 1)

    d = Dummy()
    assert _osborne(sinh(x), d) == I*sin(x*d)
    assert _osborne(tanh(x), d) == I*tan(x*d)
    assert _osborne(coth(x), d) == cot(x*d)/I
    assert _osborne(cosh(x), d) == cos(x*d)
    assert _osborne(sech(x), d) == sec(x*d)
    assert _osborne(csch(x), d) == csc(x*d)/I
    for func in (sinh, cosh, tanh, coth, sech, csch):
        h = func(pi)
        assert _osbornei(_osborne(h, d), d) == h
    # /!\ the _osborne functions are not meant to work
    # in the o(i(trig, d), d) direction so we just check
    # that they work as they are supposed to work
    assert _osbornei(cos(x*y + z), y) == cosh(x + z*I)
    assert _osbornei(sin(x*y + z), y) == sinh(x + z*I)/I
    assert _osbornei(tan(x*y + z), y) == tanh(x + z*I)/I
    assert _osbornei(cot(x*y + z), y) == coth(x + z*I)*I
    assert _osbornei(sec(x*y + z), y) == sech(x + z*I)
    assert _osbornei(csc(x*y + z), y) == csch(x + z*I)*I


def test_TR12i():
    ta, tb, tc = [tan(i) for i in (a, b, c)]
    assert TR12i((ta + tb)/(-ta*tb + 1)) == tan(a + b)
    assert TR12i((ta + tb)/(ta*tb - 1)) == -tan(a + b)
    assert TR12i((-ta - tb)/(ta*tb - 1)) == tan(a + b)
    eq = (ta + tb)/(-ta*tb + 1)**2*(-3*ta - 3*tc)/(2*(ta*tc - 1))
    assert TR12i(eq.expand()) == \
        -3*tan(a + b)*tan(a + c)/(tan(a) + tan(b) - 1)/2
    assert TR12i(tan(x)/sin(x)) == tan(x)/sin(x)
    eq = (ta + cos(2))/(-ta*tb + 1)
    assert TR12i(eq) == eq
    eq = (ta + tb + 2)**2/(-ta*tb + 1)
    assert TR12i(eq) == eq
    eq = ta/(-ta*tb + 1)
    assert TR12i(eq) == eq
    eq = (((ta + tb)*(a + 1)).expand())**2/(ta*tb - 1)
    assert TR12i(eq) == -(a + 1)**2*tan(a + b)


def test_TR14():
    eq = (cos(x) - 1)*(cos(x) + 1)
    ans = -sin(x)**2
    assert TR14(eq) == ans
    assert TR14(1/eq) == 1/ans
    assert TR14((cos(x) - 1)**2*(cos(x) + 1)**2) == ans**2
    assert TR14((cos(x) - 1)**2*(cos(x) + 1)**3) == ans**2*(cos(x) + 1)
    assert TR14((cos(x) - 1)**3*(cos(x) + 1)**2) == ans**2*(cos(x) - 1)
    eq = (cos(x) - 1)**y*(cos(x) + 1)**y
    assert TR14(eq) == eq
    eq = (cos(x) - 2)**y*(cos(x) + 1)
    assert TR14(eq) == eq
    eq = (tan(x) - 2)**2*(cos(x) + 1)
    assert TR14(eq) == eq
    i = symbols('i', integer=True)
    assert TR14((cos(x) - 1)**i*(cos(x) + 1)**i) == ans**i
    assert TR14((sin(x) - 1)**i*(sin(x) + 1)**i) == (-cos(x)**2)**i
    # could use extraction in this case
    eq = (cos(x) - 1)**(i + 1)*(cos(x) + 1)**i
    assert TR14(eq) in [(cos(x) - 1)*ans**i, eq]

    assert TR14((sin(x) - 1)*(sin(x) + 1)) == -cos(x)**2
    p1 = (cos(x) + 1)*(cos(x) - 1)
    p2 = (cos(y) - 1)*2*(cos(y) + 1)
    p3 = (3*(cos(y) - 1))*(3*(cos(y) + 1))
    assert TR14(p1*p2*p3*(x - 1)) == -18*((x - 1)*sin(x)**2*sin(y)**4)


def test_TR15_16_17():
    assert TR15(1 - 1/sin(x)**2) == -cot(x)**2
    assert TR16(1 - 1/cos(x)**2) == -tan(x)**2
    assert TR111(1 - 1/tan(x)**2) == 1 - cot(x)**2


def test_as_f_sign_1():
    assert as_f_sign_1(x + 1) == (1, x, 1)
    assert as_f_sign_1(x - 1) == (1, x, -1)
    assert as_f_sign_1(-x + 1) == (-1, x, -1)
    assert as_f_sign_1(-x - 1) == (-1, x, 1)
    assert as_f_sign_1(2*x + 2) == (2, x, 1)
    assert as_f_sign_1(x*y - y) == (y, x, -1)
    assert as_f_sign_1(-x*y + y) == (-y, x, -1)


def test_issue_25590():
    A = Symbol('A', commutative=False)
    B = Symbol('B', commutative=False)

    assert TR8(2*cos(x)*sin(x)*B*A) == sin(2*x)*B*A
    assert TR13(tan(2)*tan(3)*B*A) == (-tan(2)/tan(5) - tan(3)/tan(5) + 1)*B*A

    # XXX The result may not be optimal than
    # sin(2*x)*B*A + cos(x)**2 and may change in the future
    assert (2*cos(x)*sin(x)*B*A + cos(x)**2).simplify() == sin(2*x)*B*A + cos(2*x)/2 + S.One/2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\copy_view\test_replace.py ===
import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    option_context,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


@pytest.mark.parametrize(
    "replace_kwargs",
    [
        {"to_replace": {"a": 1, "b": 4}, "value": -1},
        # Test CoW splits blocks to avoid copying unchanged columns
        {"to_replace": {"a": 1}, "value": -1},
        {"to_replace": {"b": 4}, "value": -1},
        {"to_replace": {"b": {4: 1}}},
        # TODO: Add these in a further optimization
        # We would need to see which columns got replaced in the mask
        # which could be expensive
        # {"to_replace": {"b": 1}},
        # 1
    ],
)
def test_replace(using_copy_on_write, replace_kwargs):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()

    df_replaced = df.replace(**replace_kwargs)

    if using_copy_on_write:
        if (df_replaced["b"] == df["b"]).all():
            assert np.shares_memory(get_array(df_replaced, "b"), get_array(df, "b"))
        assert tm.shares_memory(get_array(df_replaced, "c"), get_array(df, "c"))

    # mutating squeezed df triggers a copy-on-write for that column/block
    df_replaced.loc[0, "c"] = -1
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df_replaced, "c"), get_array(df, "c"))

    if "a" in replace_kwargs["to_replace"]:
        arr = get_array(df_replaced, "a")
        df_replaced.loc[0, "a"] = 100
        assert np.shares_memory(get_array(df_replaced, "a"), arr)
    tm.assert_frame_equal(df, df_orig)


def test_replace_regex_inplace_refs(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": ["aaa", "bbb"]})
    df_orig = df.copy()
    view = df[:]
    arr = get_array(df, "a")
    with tm.assert_cow_warning(warn_copy_on_write):
        df.replace(to_replace=r"^a.*$", value="new", inplace=True, regex=True)
    if using_copy_on_write:
        assert not tm.shares_memory(arr, get_array(df, "a"))
        assert df._mgr._has_no_reference(0)
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(arr, get_array(df, "a"))


def test_replace_regex_inplace(using_copy_on_write):
    df = DataFrame({"a": ["aaa", "bbb"]})
    arr = get_array(df, "a")
    df.replace(to_replace=r"^a.*$", value="new", inplace=True, regex=True)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)
    assert tm.shares_memory(arr, get_array(df, "a"))

    df_orig = df.copy()
    df2 = df.replace(to_replace=r"^b.*$", value="new", regex=True)
    tm.assert_frame_equal(df_orig, df)
    assert not tm.shares_memory(get_array(df2, "a"), get_array(df, "a"))


def test_replace_regex_inplace_no_op(using_copy_on_write):
    df = DataFrame({"a": [1, 2]})
    arr = get_array(df, "a")
    df.replace(to_replace=r"^a.$", value="new", inplace=True, regex=True)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)
    assert np.shares_memory(arr, get_array(df, "a"))

    df_orig = df.copy()
    df2 = df.replace(to_replace=r"^x.$", value="new", regex=True)
    tm.assert_frame_equal(df_orig, df)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))


def test_replace_mask_all_false_second_block(using_copy_on_write):
    df = DataFrame({"a": [1.5, 2, 3], "b": 100.5, "c": 1, "d": 2})
    df_orig = df.copy()

    df2 = df.replace(to_replace=1.5, value=55.5)

    if using_copy_on_write:
        # TODO: Block splitting would allow us to avoid copying b
        assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
        assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    else:
        assert not np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
        assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    df2.loc[0, "c"] = 1
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged

    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
        # TODO: This should split and not copy the whole block
        # assert np.shares_memory(get_array(df, "d"), get_array(df2, "d"))


def test_replace_coerce_single_column(using_copy_on_write, using_array_manager):
    df = DataFrame({"a": [1.5, 2, 3], "b": 100.5})
    df_orig = df.copy()

    df2 = df.replace(to_replace=1.5, value="a")

    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
        assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    elif not using_array_manager:
        assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
        assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    if using_copy_on_write:
        df2.loc[0, "b"] = 0.5
        tm.assert_frame_equal(df, df_orig)  # Original is unchanged
        assert not np.shares_memory(get_array(df, "b"), get_array(df2, "b"))


def test_replace_to_replace_wrong_dtype(using_copy_on_write):
    df = DataFrame({"a": [1.5, 2, 3], "b": 100.5})
    df_orig = df.copy()

    df2 = df.replace(to_replace="xxx", value=1.5)

    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
        assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    else:
        assert not np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
        assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    df2.loc[0, "b"] = 0.5
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged

    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "b"), get_array(df2, "b"))


def test_replace_list_categorical(using_copy_on_write):
    df = DataFrame({"a": ["a", "b", "c"]}, dtype="category")
    arr = get_array(df, "a")
    msg = (
        r"The behavior of Series\.replace \(and DataFrame.replace\) "
        "with CategoricalDtype"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.replace(["c"], value="a", inplace=True)
    assert np.shares_memory(arr.codes, get_array(df, "a").codes)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)

    df_orig = df.copy()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df2 = df.replace(["b"], value="a")
    assert not np.shares_memory(arr.codes, get_array(df2, "a").codes)

    tm.assert_frame_equal(df, df_orig)


def test_replace_list_inplace_refs_categorical(using_copy_on_write):
    df = DataFrame({"a": ["a", "b", "c"]}, dtype="category")
    view = df[:]
    df_orig = df.copy()
    msg = (
        r"The behavior of Series\.replace \(and DataFrame.replace\) "
        "with CategoricalDtype"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.replace(["c"], value="a", inplace=True)
    if using_copy_on_write:
        assert not np.shares_memory(
            get_array(view, "a").codes, get_array(df, "a").codes
        )
        tm.assert_frame_equal(df_orig, view)
    else:
        # This could be inplace
        assert not np.shares_memory(
            get_array(view, "a").codes, get_array(df, "a").codes
        )


@pytest.mark.parametrize("to_replace", [1.5, [1.5], []])
def test_replace_inplace(using_copy_on_write, to_replace):
    df = DataFrame({"a": [1.5, 2, 3]})
    arr_a = get_array(df, "a")
    df.replace(to_replace=1.5, value=15.5, inplace=True)

    assert np.shares_memory(get_array(df, "a"), arr_a)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)


@pytest.mark.parametrize("to_replace", [1.5, [1.5]])
def test_replace_inplace_reference(using_copy_on_write, to_replace, warn_copy_on_write):
    df = DataFrame({"a": [1.5, 2, 3]})
    arr_a = get_array(df, "a")
    view = df[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.replace(to_replace=to_replace, value=15.5, inplace=True)

    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "a"), arr_a)
        assert df._mgr._has_no_reference(0)
        assert view._mgr._has_no_reference(0)
    else:
        assert np.shares_memory(get_array(df, "a"), arr_a)


@pytest.mark.parametrize("to_replace", ["a", 100.5])
def test_replace_inplace_reference_no_op(using_copy_on_write, to_replace):
    df = DataFrame({"a": [1.5, 2, 3]})
    arr_a = get_array(df, "a")
    view = df[:]
    df.replace(to_replace=to_replace, value=15.5, inplace=True)

    assert np.shares_memory(get_array(df, "a"), arr_a)
    if using_copy_on_write:
        assert not df._mgr._has_no_reference(0)
        assert not view._mgr._has_no_reference(0)


@pytest.mark.parametrize("to_replace", [1, [1]])
@pytest.mark.parametrize("val", [1, 1.5])
def test_replace_categorical_inplace_reference(using_copy_on_write, val, to_replace):
    df = DataFrame({"a": Categorical([1, 2, 3])})
    df_orig = df.copy()
    arr_a = get_array(df, "a")
    view = df[:]
    msg = (
        r"The behavior of Series\.replace \(and DataFrame.replace\) "
        "with CategoricalDtype"
    )
    warn = FutureWarning if val == 1.5 else None
    with tm.assert_produces_warning(warn, match=msg):
        df.replace(to_replace=to_replace, value=val, inplace=True)

    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "a").codes, arr_a.codes)
        assert df._mgr._has_no_reference(0)
        assert view._mgr._has_no_reference(0)
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(get_array(df, "a").codes, arr_a.codes)


@pytest.mark.parametrize("val", [1, 1.5])
def test_replace_categorical_inplace(using_copy_on_write, val):
    df = DataFrame({"a": Categorical([1, 2, 3])})
    arr_a = get_array(df, "a")
    msg = (
        r"The behavior of Series\.replace \(and DataFrame.replace\) "
        "with CategoricalDtype"
    )
    warn = FutureWarning if val == 1.5 else None
    with tm.assert_produces_warning(warn, match=msg):
        df.replace(to_replace=1, value=val, inplace=True)

    assert np.shares_memory(get_array(df, "a").codes, arr_a.codes)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)

    expected = DataFrame({"a": Categorical([val, 2, 3])})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("val", [1, 1.5])
def test_replace_categorical(using_copy_on_write, val):
    df = DataFrame({"a": Categorical([1, 2, 3])})
    df_orig = df.copy()
    msg = (
        r"The behavior of Series\.replace \(and DataFrame.replace\) "
        "with CategoricalDtype"
    )
    warn = FutureWarning if val == 1.5 else None
    with tm.assert_produces_warning(warn, match=msg):
        df2 = df.replace(to_replace=1, value=val)

    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)
        assert df2._mgr._has_no_reference(0)
    assert not np.shares_memory(get_array(df, "a").codes, get_array(df2, "a").codes)
    tm.assert_frame_equal(df, df_orig)

    arr_a = get_array(df2, "a").codes
    df2.iloc[0, 0] = 2.0
    assert np.shares_memory(get_array(df2, "a").codes, arr_a)


@pytest.mark.parametrize("method", ["where", "mask"])
def test_masking_inplace(using_copy_on_write, method, warn_copy_on_write):
    df = DataFrame({"a": [1.5, 2, 3]})
    df_orig = df.copy()
    arr_a = get_array(df, "a")
    view = df[:]

    method = getattr(df, method)
    if warn_copy_on_write:
        with tm.assert_cow_warning():
            method(df["a"] > 1.6, -1, inplace=True)
    else:
        method(df["a"] > 1.6, -1, inplace=True)

    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "a"), arr_a)
        assert df._mgr._has_no_reference(0)
        assert view._mgr._has_no_reference(0)
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(get_array(df, "a"), arr_a)


def test_replace_empty_list(using_copy_on_write):
    df = DataFrame({"a": [1, 2]})

    df2 = df.replace([], [])
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
        assert not df._mgr._has_no_reference(0)
    else:
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    arr_a = get_array(df, "a")
    df.replace([], [])
    if using_copy_on_write:
        assert np.shares_memory(get_array(df, "a"), arr_a)
        assert not df._mgr._has_no_reference(0)
        assert not df2._mgr._has_no_reference(0)


@pytest.mark.parametrize("value", ["d", None])
def test_replace_object_list_inplace(using_copy_on_write, value):
    df = DataFrame({"a": ["a", "b", "c"]}, dtype=object)
    arr = get_array(df, "a")
    df.replace(["c"], value, inplace=True)
    if using_copy_on_write or value is None:
        assert tm.shares_memory(arr, get_array(df, "a"))
    else:
        # This could be inplace
        assert not np.shares_memory(arr, get_array(df, "a"))
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)


def test_replace_list_multiple_elements_inplace(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3]})
    arr = get_array(df, "a")
    df.replace([1, 2], 4, inplace=True)
    if using_copy_on_write:
        assert np.shares_memory(arr, get_array(df, "a"))
        assert df._mgr._has_no_reference(0)
    else:
        assert np.shares_memory(arr, get_array(df, "a"))


def test_replace_list_none(using_copy_on_write):
    df = DataFrame({"a": ["a", "b", "c"]})

    df_orig = df.copy()
    df2 = df.replace(["b"], value=None)
    tm.assert_frame_equal(df, df_orig)

    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    # replace multiple values that don't actually replace anything with None
    # https://github.com/pandas-dev/pandas/issues/59770
    df3 = df.replace(["d", "e", "f"], value=None)
    tm.assert_frame_equal(df3, df_orig)
    if using_copy_on_write:
        assert tm.shares_memory(get_array(df, "a"), get_array(df3, "a"))
    else:
        assert not tm.shares_memory(get_array(df, "a"), get_array(df3, "a"))


def test_replace_list_none_inplace_refs(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": ["a", "b", "c"]})
    arr = get_array(df, "a")
    df_orig = df.copy()
    view = df[:]
    with tm.assert_cow_warning(warn_copy_on_write):
        df.replace(["a"], value=None, inplace=True)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)
        assert not np.shares_memory(arr, get_array(df, "a"))
        tm.assert_frame_equal(df_orig, view)
    else:
        assert np.shares_memory(arr, get_array(df, "a"))


def test_replace_columnwise_no_op_inplace(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    view = df[:]
    df_orig = df.copy()
    df.replace({"a": 10}, 100, inplace=True)
    if using_copy_on_write:
        assert np.shares_memory(get_array(view, "a"), get_array(df, "a"))
        df.iloc[0, 0] = 100
        tm.assert_frame_equal(view, df_orig)


def test_replace_columnwise_no_op(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    df_orig = df.copy()
    df2 = df.replace({"a": 10}, 100)
    if using_copy_on_write:
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    df2.iloc[0, 0] = 100
    tm.assert_frame_equal(df, df_orig)


def test_replace_chained_assignment(using_copy_on_write):
    df = DataFrame({"a": [1, np.nan, 2], "b": 1})
    df_orig = df.copy()
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df["a"].replace(1, 100, inplace=True)
        tm.assert_frame_equal(df, df_orig)

        with tm.raises_chained_assignment_error():
            df[["a"]].replace(1, 100, inplace=True)
        tm.assert_frame_equal(df, df_orig)
    else:
        with tm.assert_produces_warning(None):
            with option_context("mode.chained_assignment", None):
                df[["a"]].replace(1, 100, inplace=True)

        with tm.assert_produces_warning(None):
            with option_context("mode.chained_assignment", None):
                df[df.a > 5].replace(1, 100, inplace=True)

        with tm.assert_produces_warning(FutureWarning, match="inplace method"):
            df["a"].replace(1, 100, inplace=True)


def test_replace_listlike(using_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    df_orig = df.copy()

    result = df.replace([200, 201], [11, 11])
    if using_copy_on_write:
        assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    else:
        assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))

    result.iloc[0, 0] = 100
    tm.assert_frame_equal(df, df)

    result = df.replace([200, 2], [10, 10])
    assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))
    tm.assert_frame_equal(df, df_orig)


def test_replace_listlike_inplace(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    arr = get_array(df, "a")
    df.replace([200, 2], [10, 11], inplace=True)
    assert np.shares_memory(get_array(df, "a"), arr)

    view = df[:]
    df_orig = df.copy()
    with tm.assert_cow_warning(warn_copy_on_write):
        df.replace([200, 3], [10, 11], inplace=True)
    if using_copy_on_write:
        assert not np.shares_memory(get_array(df, "a"), arr)
        tm.assert_frame_equal(view, df_orig)
    else:
        assert np.shares_memory(get_array(df, "a"), arr)
        tm.assert_frame_equal(df, view)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\json\test_json.py ===
import collections
import operator
import sys

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
    JSONArray,
    JSONDtype,
    make_data,
)

# We intentionally don't run base.BaseSetitemTests because pandas'
# internals has trouble setting sequences of values into scalar positions.
unhashable = pytest.mark.xfail(reason="Unhashable")


@pytest.fixture
def dtype():
    return JSONDtype()


@pytest.fixture
def data():
    """Length-100 PeriodArray for semantics test."""
    data = make_data()

    # Why the while loop? NumPy is unable to construct an ndarray from
    # equal-length ndarrays. Many of our operations involve coercing the
    # EA to an ndarray of objects. To avoid random test failures, we ensure
    # that our data is coercible to an ndarray. Several tests deal with only
    # the first two elements, so that's what we'll check.

    while len(data[0]) == len(data[1]):
        data = make_data()

    return JSONArray(data)


@pytest.fixture
def data_missing():
    """Length 2 array with [NA, Valid]"""
    return JSONArray([{}, {"a": 10}])


@pytest.fixture
def data_for_sorting():
    return JSONArray([{"b": 1}, {"c": 4}, {"a": 2, "c": 3}])


@pytest.fixture
def data_missing_for_sorting():
    return JSONArray([{"b": 1}, {}, {"a": 4}])


@pytest.fixture
def na_cmp():
    return operator.eq


@pytest.fixture
def data_for_grouping():
    return JSONArray(
        [
            {"b": 1},
            {"b": 1},
            {},
            {},
            {"a": 0, "c": 2},
            {"a": 0, "c": 2},
            {"b": 1},
            {"c": 2},
        ]
    )


class TestJSONArray(base.ExtensionTests):
    @pytest.mark.xfail(
        reason="comparison method not implemented for JSONArray (GH-37867)"
    )
    def test_contains(self, data):
        # GH-37867
        super().test_contains(data)

    @pytest.mark.xfail(reason="not implemented constructor from dtype")
    def test_from_dtype(self, data):
        # construct from our dtype & string dtype
        super().test_from_dtype(data)

    @pytest.mark.xfail(reason="RecursionError, GH-33900")
    def test_series_constructor_no_data_with_index(self, dtype, na_value):
        # RecursionError: maximum recursion depth exceeded in comparison
        rec_limit = sys.getrecursionlimit()
        try:
            # Limit to avoid stack overflow on Windows CI
            sys.setrecursionlimit(100)
            super().test_series_constructor_no_data_with_index(dtype, na_value)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason="RecursionError, GH-33900")
    def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
        # RecursionError: maximum recursion depth exceeded in comparison
        rec_limit = sys.getrecursionlimit()
        try:
            # Limit to avoid stack overflow on Windows CI
            sys.setrecursionlimit(100)
            super().test_series_constructor_scalar_na_with_index(dtype, na_value)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason="collection as scalar, GH-33901")
    def test_series_constructor_scalar_with_index(self, data, dtype):
        # TypeError: All values must be of type <class 'collections.abc.Mapping'>
        rec_limit = sys.getrecursionlimit()
        try:
            # Limit to avoid stack overflow on Windows CI
            sys.setrecursionlimit(100)
            super().test_series_constructor_scalar_with_index(data, dtype)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason="Different definitions of NA")
    def test_stack(self):
        """
        The test does .astype(object).stack(future_stack=True). If we happen to have
        any missing values in `data`, then we'll end up with different
        rows since we consider `{}` NA, but `.astype(object)` doesn't.
        """
        super().test_stack()

    @pytest.mark.xfail(reason="dict for NA")
    def test_unstack(self, data, index):
        # The base test has NaN for the expected NA value.
        # this matches otherwise
        return super().test_unstack(data, index)

    @pytest.mark.xfail(reason="Setting a dict as a scalar")
    def test_fillna_series(self):
        """We treat dictionaries as a mapping in fillna, not a scalar."""
        super().test_fillna_series()

    @pytest.mark.xfail(reason="Setting a dict as a scalar")
    def test_fillna_frame(self):
        """We treat dictionaries as a mapping in fillna, not a scalar."""
        super().test_fillna_frame()

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
        msg = "JSONArray does not implement limit_area"
        with pytest.raises(NotImplementedError, match=msg):
            super().test_ffill_limit_area(
                data_missing, limit_area, input_ilocs, expected_ilocs
            )

    @unhashable
    def test_value_counts(self, all_data, dropna):
        super().test_value_counts(all_data, dropna)

    @unhashable
    def test_value_counts_with_normalize(self, data):
        super().test_value_counts_with_normalize(data)

    @unhashable
    def test_sort_values_frame(self):
        # TODO (EA.factorize): see if _values_for_factorize allows this.
        super().test_sort_values_frame()

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(self, data_for_sorting, ascending, sort_by_key):
        super().test_sort_values(data_for_sorting, ascending, sort_by_key)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_missing(
        self, data_missing_for_sorting, ascending, sort_by_key
    ):
        super().test_sort_values_missing(
            data_missing_for_sorting, ascending, sort_by_key
        )

    @pytest.mark.xfail(reason="combine for JSONArray not supported")
    def test_combine_le(self, data_repeated):
        super().test_combine_le(data_repeated)

    @pytest.mark.xfail(
        reason="combine for JSONArray not supported - "
        "may pass depending on random data",
        strict=False,
        raises=AssertionError,
    )
    def test_combine_first(self, data):
        super().test_combine_first(data)

    @pytest.mark.xfail(reason="broadcasting error")
    def test_where_series(self, data, na_value):
        # Fails with
        # *** ValueError: operands could not be broadcast together
        # with shapes (4,) (4,) (0,)
        super().test_where_series(data, na_value)

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_searchsorted(self, data_for_sorting):
        super().test_searchsorted(data_for_sorting)

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_equals(self, data, na_value, as_series):
        super().test_equals(data, na_value, as_series)

    @pytest.mark.skip("fill-value is interpreted as a dict of values")
    def test_fillna_copy_frame(self, data_missing):
        super().test_fillna_copy_frame(data_missing)

    def test_equals_same_data_different_object(
        self, data, using_copy_on_write, request
    ):
        if using_copy_on_write:
            mark = pytest.mark.xfail(reason="Fails with CoW")
            request.applymarker(mark)
        super().test_equals_same_data_different_object(data)

    @pytest.mark.xfail(reason="failing on np.array(self, dtype=str)")
    def test_astype_str(self):
        """This currently fails in NumPy on np.array(self, dtype=str) with

        *** ValueError: setting an array element with a sequence
        """
        super().test_astype_str()

    @unhashable
    def test_groupby_extension_transform(self):
        """
        This currently fails in Series.name.setter, since the
        name must be hashable, but the value is a dictionary.
        I think this is what we want, i.e. `.name` should be the original
        values, and not the values for factorization.
        """
        super().test_groupby_extension_transform()

    @unhashable
    def test_groupby_extension_apply(self):
        """
        This fails in Index._do_unique_check with

        >   hash(val)
        E   TypeError: unhashable type: 'UserDict' with

        I suspect that once we support Index[ExtensionArray],
        we'll be able to dispatch unique.
        """
        super().test_groupby_extension_apply()

    @unhashable
    def test_groupby_extension_agg(self):
        """
        This fails when we get to tm.assert_series_equal when left.index
        contains dictionaries, which are not hashable.
        """
        super().test_groupby_extension_agg()

    @unhashable
    def test_groupby_extension_no_sort(self):
        """
        This fails when we get to tm.assert_series_equal when left.index
        contains dictionaries, which are not hashable.
        """
        super().test_groupby_extension_no_sort()

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        if len(data[0]) != 1:
            mark = pytest.mark.xfail(reason="raises in coercing to Series")
            request.applymarker(mark)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_compare_array(self, data, comparison_op, request):
        if comparison_op.__name__ in ["eq", "ne"]:
            mark = pytest.mark.xfail(reason="Comparison methods not implemented")
            request.applymarker(mark)
        super().test_compare_array(data, comparison_op)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_loc_scalar_mixed(self, data):
        super().test_setitem_loc_scalar_mixed(data)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        super().test_setitem_loc_scalar_multiple_homogoneous(data)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_iloc_scalar_mixed(self, data):
        super().test_setitem_iloc_scalar_mixed(data)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        super().test_setitem_iloc_scalar_multiple_homogoneous(data)

    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
            pd.array([True, True, True, pd.NA, pd.NA], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array", "boolean-array-na"],
    )
    def test_setitem_mask(self, data, mask, box_in_series, request):
        if box_in_series:
            mark = pytest.mark.xfail(
                reason="cannot set using a list-like indexer with a different length"
            )
            request.applymarker(mark)
        elif not isinstance(mask, np.ndarray):
            mark = pytest.mark.xfail(reason="Issues unwanted DeprecationWarning")
            request.applymarker(mark)
        super().test_setitem_mask(data, mask, box_in_series)

    def test_setitem_mask_raises(self, data, box_in_series, request):
        if not box_in_series:
            mark = pytest.mark.xfail(reason="Fails to raise")
            request.applymarker(mark)

        super().test_setitem_mask_raises(data, box_in_series)

    @pytest.mark.xfail(
        reason="cannot set using a list-like indexer with a different length"
    )
    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        super().test_setitem_mask_boolean_array_with_na(data, box_in_series)

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series, request):
        if box_in_series:
            mark = pytest.mark.xfail(
                reason="cannot set using a list-like indexer with a different length"
            )
            request.applymarker(mark)
        super().test_setitem_integer_array(data, idx, box_in_series)

    @pytest.mark.xfail(reason="list indices must be integers or slices, not NAType")
    @pytest.mark.parametrize(
        "idx, box_in_series",
        [
            ([0, 1, 2, pd.NA], False),
            pytest.param(
                [0, 1, 2, pd.NA], True, marks=pytest.mark.xfail(reason="GH-31948")
            ),
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), False),
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), False),
        ],
        ids=["list-False", "list-True", "integer-array-False", "integer-array-True"],
    )
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        super().test_setitem_integer_with_missing_raises(data, idx, box_in_series)

    @pytest.mark.xfail(reason="Fails to raise")
    def test_setitem_scalar_key_sequence_raise(self, data):
        super().test_setitem_scalar_key_sequence_raise(data)

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer, request):
        if "full_slice" in request.node.name:
            mark = pytest.mark.xfail(reason="slice is not iterable")
            request.applymarker(mark)
        super().test_setitem_with_expansion_dataframe_column(data, full_indexer)

    @pytest.mark.xfail(reason="slice is not iterable")
    def test_setitem_frame_2d_values(self, data):
        super().test_setitem_frame_2d_values(data)

    @pytest.mark.xfail(
        reason="cannot set using a list-like indexer with a different length"
    )
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter):
        super().test_setitem_mask_broadcast(data, setter)

    @pytest.mark.xfail(
        reason="cannot set using a slice indexer with a different length"
    )
    def test_setitem_slice(self, data, box_in_series):
        super().test_setitem_slice(data, box_in_series)

    @pytest.mark.xfail(reason="slice object is not iterable")
    def test_setitem_loc_iloc_slice(self, data):
        super().test_setitem_loc_iloc_slice(data)

    @pytest.mark.xfail(reason="slice object is not iterable")
    def test_setitem_slice_mismatch_length_raises(self, data):
        super().test_setitem_slice_mismatch_length_raises(data)

    @pytest.mark.xfail(reason="slice object is not iterable")
    def test_setitem_slice_array(self, data):
        super().test_setitem_slice_array(data)

    @pytest.mark.xfail(reason="Fail to raise")
    def test_setitem_invalid(self, data, invalid_scalar):
        super().test_setitem_invalid(data, invalid_scalar)

    @pytest.mark.xfail(reason="only integer scalar arrays can be converted")
    def test_setitem_2d_values(self, data):
        super().test_setitem_2d_values(data)

    @pytest.mark.xfail(reason="data type 'json' not understood")
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        super().test_EA_types(engine, data, request)


def custom_assert_series_equal(left, right, *args, **kwargs):
    # NumPy doesn't handle an array of equal-length UserDicts.
    # The default assert_series_equal eventually does a
    # Series.values, which raises. We work around it by
    # converting the UserDicts to dicts.
    if left.dtype.name == "json":
        assert left.dtype == right.dtype
        left = pd.Series(
            JSONArray(left.values.astype(object)), index=left.index, name=left.name
        )
        right = pd.Series(
            JSONArray(right.values.astype(object)),
            index=right.index,
            name=right.name,
        )
    tm.assert_series_equal(left, right, *args, **kwargs)


def custom_assert_frame_equal(left, right, *args, **kwargs):
    obj_type = kwargs.get("obj", "DataFrame")
    tm.assert_index_equal(
        left.columns,
        right.columns,
        exact=kwargs.get("check_column_type", "equiv"),
        check_names=kwargs.get("check_names", True),
        check_exact=kwargs.get("check_exact", False),
        check_categorical=kwargs.get("check_categorical", True),
        obj=f"{obj_type}.columns",
    )

    jsons = (left.dtypes == "json").index

    for col in jsons:
        custom_assert_series_equal(left[col], right[col], *args, **kwargs)

    left = left.drop(columns=jsons)
    right = right.drop(columns=jsons)
    tm.assert_frame_equal(left, right, *args, **kwargs)


def test_custom_asserts():
    # This would always trigger the KeyError from trying to put
    # an array of equal-length UserDicts inside an ndarray.
    data = JSONArray(
        [
            collections.UserDict({"a": 1}),
            collections.UserDict({"b": 2}),
            collections.UserDict({"c": 3}),
        ]
    )
    a = pd.Series(data)
    custom_assert_series_equal(a, a)
    custom_assert_frame_equal(a.to_frame(), a.to_frame())

    b = pd.Series(data.take([0, 0, 1]))
    msg = r"Series are different"
    with pytest.raises(AssertionError, match=msg):
        custom_assert_series_equal(a, b)

    with pytest.raises(AssertionError, match=msg):
        custom_assert_frame_equal(a.to_frame(), b.to_frame())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\tests\test_minpoly.py ===
"""Tests for minimal polynomials. """

from sympy.core.function import expand
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (AlgebraicNumber, I, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.ntheory.generate import nextprime
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.solvers.solveset import nonlinsolve
from sympy.geometry import Circle, intersection
from sympy.testing.pytest import raises, slow
from sympy.sets.sets import FiniteSet
from sympy.geometry.point import Point2D
from sympy.polys.numberfields.minpoly import (
    minimal_polynomial,
    _choose_factor,
    _minpoly_op_algebraic_element,
    _separate_sq,
    _minpoly_groebner,
)
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import (
    NotAlgebraic,
    GeneratorsError,
)

from sympy.polys.domains import QQ
from sympy.polys.rootoftools import rootof
from sympy.polys.polytools import degree

from sympy.abc import x, y, z

Q = Rational


def test_minimal_polynomial():
    assert minimal_polynomial(-7, x) == x + 7
    assert minimal_polynomial(-1, x) == x + 1
    assert minimal_polynomial( 0, x) == x
    assert minimal_polynomial( 1, x) == x - 1
    assert minimal_polynomial( 7, x) == x - 7

    assert minimal_polynomial(sqrt(2), x) == x**2 - 2
    assert minimal_polynomial(sqrt(5), x) == x**2 - 5
    assert minimal_polynomial(sqrt(6), x) == x**2 - 6

    assert minimal_polynomial(2*sqrt(2), x) == x**2 - 8
    assert minimal_polynomial(3*sqrt(5), x) == x**2 - 45
    assert minimal_polynomial(4*sqrt(6), x) == x**2 - 96

    assert minimal_polynomial(2*sqrt(2) + 3, x) == x**2 - 6*x + 1
    assert minimal_polynomial(3*sqrt(5) + 6, x) == x**2 - 12*x - 9
    assert minimal_polynomial(4*sqrt(6) + 7, x) == x**2 - 14*x - 47

    assert minimal_polynomial(2*sqrt(2) - 3, x) == x**2 + 6*x + 1
    assert minimal_polynomial(3*sqrt(5) - 6, x) == x**2 + 12*x - 9
    assert minimal_polynomial(4*sqrt(6) - 7, x) == x**2 + 14*x - 47

    assert minimal_polynomial(sqrt(1 + sqrt(6)), x) == x**4 - 2*x**2 - 5
    assert minimal_polynomial(sqrt(I + sqrt(6)), x) == x**8 - 10*x**4 + 49

    assert minimal_polynomial(2*I + sqrt(2 + I), x) == x**4 + 4*x**2 + 8*x + 37

    assert minimal_polynomial(sqrt(2) + sqrt(3), x) == x**4 - 10*x**2 + 1
    assert minimal_polynomial(
        sqrt(2) + sqrt(3) + sqrt(6), x) == x**4 - 22*x**2 - 48*x - 23

    a = 1 - 9*sqrt(2) + 7*sqrt(3)

    assert minimal_polynomial(
        1/a, x) == 392*x**4 - 1232*x**3 + 612*x**2 + 4*x - 1
    assert minimal_polynomial(
        1/sqrt(a), x) == 392*x**8 - 1232*x**6 + 612*x**4 + 4*x**2 - 1

    raises(NotAlgebraic, lambda: minimal_polynomial(oo, x))
    raises(NotAlgebraic, lambda: minimal_polynomial(2**y, x))
    raises(NotAlgebraic, lambda: minimal_polynomial(sin(1), x))

    assert minimal_polynomial(sqrt(2)).dummy_eq(x**2 - 2)
    assert minimal_polynomial(sqrt(2), x) == x**2 - 2

    assert minimal_polynomial(sqrt(2), polys=True) == Poly(x**2 - 2)
    assert minimal_polynomial(sqrt(2), x, polys=True) == Poly(x**2 - 2, domain='QQ')
    assert minimal_polynomial(sqrt(2), x, polys=True, compose=False) == Poly(x**2 - 2, domain='QQ')

    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(3))

    assert minimal_polynomial(a, x) == x**2 - 2
    assert minimal_polynomial(b, x) == x**2 - 3

    assert minimal_polynomial(a, x, polys=True) == Poly(x**2 - 2, domain='QQ')
    assert minimal_polynomial(b, x, polys=True) == Poly(x**2 - 3, domain='QQ')

    assert minimal_polynomial(sqrt(a/2 + 17), x) == 2*x**4 - 68*x**2 + 577
    assert minimal_polynomial(sqrt(b/2 + 17), x) == 4*x**4 - 136*x**2 + 1153

    a, b = sqrt(2)/3 + 7, AlgebraicNumber(sqrt(2)/3 + 7)

    f = 81*x**8 - 2268*x**6 - 4536*x**5 + 22644*x**4 + 63216*x**3 - \
        31608*x**2 - 189648*x + 141358

    assert minimal_polynomial(sqrt(a) + sqrt(sqrt(a)), x) == f
    assert minimal_polynomial(sqrt(b) + sqrt(sqrt(b)), x) == f

    assert minimal_polynomial(
        a**Q(3, 2), x) == 729*x**4 - 506898*x**2 + 84604519

    # issue 5994
    eq = S('''
        -1/(800*sqrt(-1/240 + 1/(18000*(-1/17280000 +
        sqrt(15)*I/28800000)**(1/3)) + 2*(-1/17280000 +
        sqrt(15)*I/28800000)**(1/3)))''')
    assert minimal_polynomial(eq, x) == 8000*x**2 - 1

    ex = (sqrt(5)*sqrt(I)/(5*sqrt(1 + 125*I))
            + 25*sqrt(5)/(I**Q(5,2)*(1 + 125*I)**Q(3,2))
            + 3125*sqrt(5)/(I**Q(11,2)*(1 + 125*I)**Q(3,2))
            + 5*I*sqrt(1 - I/125))
    mp = minimal_polynomial(ex, x)
    assert mp == 25*x**4 + 5000*x**2 + 250016

    ex = 1 + sqrt(2) + sqrt(3)
    mp = minimal_polynomial(ex, x)
    assert mp == x**4 - 4*x**3 - 4*x**2 + 16*x - 8

    ex = 1/(1 + sqrt(2) + sqrt(3))
    mp = minimal_polynomial(ex, x)
    assert mp == 8*x**4 - 16*x**3 + 4*x**2 + 4*x - 1

    p = (expand((1 + sqrt(2) - 2*sqrt(3) + sqrt(7))**3))**Rational(1, 3)
    mp = minimal_polynomial(p, x)
    assert mp == x**8 - 8*x**7 - 56*x**6 + 448*x**5 + 480*x**4 - 5056*x**3 + 1984*x**2 + 7424*x - 3008
    p = expand((1 + sqrt(2) - 2*sqrt(3) + sqrt(7))**3)
    mp = minimal_polynomial(p, x)
    assert mp == x**8 - 512*x**7 - 118208*x**6 + 31131136*x**5 + 647362560*x**4 - 56026611712*x**3 + 116994310144*x**2 + 404854931456*x - 27216576512

    assert minimal_polynomial(S("-sqrt(5)/2 - 1/2 + (-sqrt(5)/2 - 1/2)**2"), x) == x - 1
    a = 1 + sqrt(2)
    assert minimal_polynomial((a*sqrt(2) + a)**3, x) == x**2 - 198*x + 1

    p = 1/(1 + sqrt(2) + sqrt(3))
    assert minimal_polynomial(p, x, compose=False) == 8*x**4 - 16*x**3 + 4*x**2 + 4*x - 1

    p = 2/(1 + sqrt(2) + sqrt(3))
    assert minimal_polynomial(p, x, compose=False) == x**4 - 4*x**3 + 2*x**2 + 4*x - 2

    assert minimal_polynomial(1 + sqrt(2)*I, x, compose=False) == x**2 - 2*x + 3
    assert minimal_polynomial(1/(1 + sqrt(2)) + 1, x, compose=False) == x**2 - 2
    assert minimal_polynomial(sqrt(2)*I + I*(1 + sqrt(2)), x,
            compose=False) ==  x**4 + 18*x**2 + 49

    # minimal polynomial of I
    assert minimal_polynomial(I, x, domain=QQ.algebraic_field(I)) == x - I
    K = QQ.algebraic_field(I*(sqrt(2) + 1))
    assert minimal_polynomial(I, x, domain=K) == x - I
    assert minimal_polynomial(I, x, domain=QQ) == x**2 + 1
    assert minimal_polynomial(I, x, domain='QQ(y)') == x**2 + 1

    #issue 11553
    assert minimal_polynomial(GoldenRatio, x) == x**2 - x - 1
    assert minimal_polynomial(TribonacciConstant + 3, x) == x**3 - 10*x**2 + 32*x - 34
    assert minimal_polynomial(GoldenRatio, x, domain=QQ.algebraic_field(sqrt(5))) == \
            2*x - sqrt(5) - 1
    assert minimal_polynomial(TribonacciConstant, x, domain=QQ.algebraic_field(cbrt(19 - 3*sqrt(33)))) == \
    48*x - 19*(19 - 3*sqrt(33))**Rational(2, 3) - 3*sqrt(33)*(19 - 3*sqrt(33))**Rational(2, 3) \
    - 16*(19 - 3*sqrt(33))**Rational(1, 3) - 16

    # AlgebraicNumber with an alias.
    # Wester H24
    phi = AlgebraicNumber(S.GoldenRatio.expand(func=True), alias='phi')
    assert minimal_polynomial(phi, x) == x**2 - x - 1


def test_issue_26903():
    p1 = nextprime(10**16)  # greater than 10**15
    p2 = nextprime(p1)
    assert sqrt(p1**2*p2).is_Pow  # square not extracted
    zero = sqrt(p1**2*p2) - p1*sqrt(p2)
    assert minimal_polynomial(zero, x) == x
    assert minimal_polynomial(sqrt(2) - zero, x) == x**2 - 2


def test_issue_8353():
    assert minimal_polynomial(exp(3*I*pi, evaluate=False), x) == x + 1
    assert minimal_polynomial(Pow(8, S(1)/3, evaluate=False), x
        ) == x - 2


def test_minimal_polynomial_issue_19732():
    # https://github.com/sympy/sympy/issues/19732
    expr = (-280898097948878450887044002323982963174671632174995451265117559518123750720061943079105185551006003416773064305074191140286225850817291393988597615/(-488144716373031204149459129212782509078221364279079444636386844223983756114492222145074506571622290776245390771587888364089507840000000*sqrt(238368341569)*sqrt(S(11918417078450)/63568729
    - 24411360*sqrt(238368341569)/63568729) +
    238326799225996604451373809274348704114327860564921529846705817404208077866956345381951726531296652901169111729944612727047670549086208000000*sqrt(S(11918417078450)/63568729
        - 24411360*sqrt(238368341569)/63568729)) -
    180561807339168676696180573852937120123827201075968945871075967679148461189459480842956689723484024031016208588658753107/(-59358007109636562851035004992802812513575019937126272896569856090962677491318275291141463850327474176000000*sqrt(238368341569)*sqrt(S(11918417078450)/63568729
        - 24411360*sqrt(238368341569)/63568729) +
        28980348180319251787320809875930301310576055074938369007463004788921613896002936637780993064387310446267596800000*sqrt(S(11918417078450)/63568729
            - 24411360*sqrt(238368341569)/63568729)))
    poly = (2151288870990266634727173620565483054187142169311153766675688628985237817262915166497766867289157986631135400926544697981091151416655364879773546003475813114962656742744975460025956167152918469472166170500512008351638710934022160294849059721218824490226159355197136265032810944357335461128949781377875451881300105989490353140886315677977149440000000000000000000000*x**4
            - 5773274155644072033773937864114266313663195672820501581692669271302387257492905909558846459600429795784309388968498783843631580008547382703258503404023153694528041873101120067477617592651525155101107144042679962433039557235772239171616433004024998230222455940044709064078962397144550855715640331680262171410099614469231080995436488414164502751395405398078353242072696360734131090111239998110773292915337556205692674790561090109440000000000000*x**2
            + 211295968822207088328287206509522887719741955693091053353263782924470627623790749534705683380138972642560898936171035770539616881000369889020398551821767092685775598633794696371561234818461806577723412581353857653829324364446419444210520602157621008010129702779407422072249192199762604318993590841636967747488049176548615614290254356975376588506729604345612047361483789518445332415765213187893207704958013682516462853001964919444736320672860140355089)
    assert minimal_polynomial(expr, x) == poly


def test_minimal_polynomial_hi_prec():
    p = 1/sqrt(1 - 9*sqrt(2) + 7*sqrt(3) + Rational(1, 10)**30)
    mp = minimal_polynomial(p, x)
    # checked with Wolfram Alpha
    assert mp.coeff(x**6) == -1232000000000000000000000000001223999999999999999999999999999987999999999999999999999999999996000000000000000000000000000000


def test_minimal_polynomial_sq():
    from sympy.core.add import Add
    from sympy.core.function import expand_multinomial
    p = expand_multinomial((1 + 5*sqrt(2) + 2*sqrt(3))**3)
    mp = minimal_polynomial(p**Rational(1, 3), x)
    assert mp == x**4 - 4*x**3 - 118*x**2 + 244*x + 1321
    p = expand_multinomial((1 + sqrt(2) - 2*sqrt(3) + sqrt(7))**3)
    mp = minimal_polynomial(p**Rational(1, 3), x)
    assert mp == x**8 - 8*x**7 - 56*x**6 + 448*x**5 + 480*x**4 - 5056*x**3 + 1984*x**2 + 7424*x - 3008
    p = Add(*[sqrt(i) for i in range(1, 12)])
    mp = minimal_polynomial(p, x)
    assert mp.subs({x: 0}) == -71965773323122507776


def test_minpoly_compose():
    # issue 6868
    eq = S('''
        -1/(800*sqrt(-1/240 + 1/(18000*(-1/17280000 +
        sqrt(15)*I/28800000)**(1/3)) + 2*(-1/17280000 +
        sqrt(15)*I/28800000)**(1/3)))''')
    mp = minimal_polynomial(eq + 3, x)
    assert mp == 8000*x**2 - 48000*x + 71999

    # issue 5888
    assert minimal_polynomial(exp(I*pi/8), x) == x**8 + 1

    mp = minimal_polynomial(sin(pi/7) + sqrt(2), x)
    assert mp == 4096*x**12 - 63488*x**10 + 351488*x**8 - 826496*x**6 + \
        770912*x**4 - 268432*x**2 + 28561
    mp = minimal_polynomial(cos(pi/7) + sqrt(2), x)
    assert mp == 64*x**6 - 64*x**5 - 432*x**4 + 304*x**3 + 712*x**2 - \
            232*x - 239
    mp = minimal_polynomial(exp(I*pi/7) + sqrt(2), x)
    assert mp == x**12 - 2*x**11 - 9*x**10 + 16*x**9 + 43*x**8 - 70*x**7 - 97*x**6 + 126*x**5 + 211*x**4 - 212*x**3 - 37*x**2 + 142*x + 127

    mp = minimal_polynomial(sin(pi/7) + sqrt(2), x)
    assert mp == 4096*x**12 - 63488*x**10 + 351488*x**8 - 826496*x**6 + \
        770912*x**4 - 268432*x**2 + 28561
    mp = minimal_polynomial(cos(pi/7) + sqrt(2), x)
    assert mp == 64*x**6 - 64*x**5 - 432*x**4 + 304*x**3 + 712*x**2 - \
            232*x - 239
    mp = minimal_polynomial(exp(I*pi/7) + sqrt(2), x)
    assert mp == x**12 - 2*x**11 - 9*x**10 + 16*x**9 + 43*x**8 - 70*x**7 - 97*x**6 + 126*x**5 + 211*x**4 - 212*x**3 - 37*x**2 + 142*x + 127

    mp = minimal_polynomial(exp(I*pi*Rational(2, 7)), x)
    assert mp == x**6 + x**5 + x**4 + x**3 + x**2 + x + 1
    mp = minimal_polynomial(exp(I*pi*Rational(2, 15)), x)
    assert mp == x**8 - x**7 + x**5 - x**4 + x**3 - x + 1
    mp = minimal_polynomial(cos(pi*Rational(2, 7)), x)
    assert mp == 8*x**3 + 4*x**2 - 4*x - 1
    mp = minimal_polynomial(sin(pi*Rational(2, 7)), x)
    ex = (5*cos(pi*Rational(2, 7)) - 7)/(9*cos(pi/7) - 5*cos(pi*Rational(3, 7)))
    mp = minimal_polynomial(ex, x)
    assert mp == x**3 + 2*x**2 - x - 1
    assert minimal_polynomial(-1/(2*cos(pi/7)), x) == x**3 + 2*x**2 - x - 1
    assert minimal_polynomial(sin(pi*Rational(2, 15)), x) == \
            256*x**8 - 448*x**6 + 224*x**4 - 32*x**2 + 1
    assert minimal_polynomial(sin(pi*Rational(5, 14)), x) == 8*x**3 - 4*x**2 - 4*x + 1
    assert minimal_polynomial(cos(pi/15), x) == 16*x**4 + 8*x**3 - 16*x**2 - 8*x + 1

    ex = rootof(x**3 +x*4 + 1, 0)
    mp = minimal_polynomial(ex, x)
    assert mp == x**3 + 4*x + 1
    mp = minimal_polynomial(ex + 1, x)
    assert mp == x**3 - 3*x**2 + 7*x - 4
    assert minimal_polynomial(exp(I*pi/3), x) == x**2 - x + 1
    assert minimal_polynomial(exp(I*pi/4), x) == x**4 + 1
    assert minimal_polynomial(exp(I*pi/6), x) == x**4 - x**2 + 1
    assert minimal_polynomial(exp(I*pi/9), x) == x**6 - x**3 + 1
    assert minimal_polynomial(exp(I*pi/10), x) == x**8 - x**6 + x**4 - x**2 + 1
    assert minimal_polynomial(sin(pi/9), x) == 64*x**6 - 96*x**4 + 36*x**2 - 3
    assert minimal_polynomial(sin(pi/11), x) == 1024*x**10 - 2816*x**8 + \
            2816*x**6 - 1232*x**4 + 220*x**2 - 11
    assert minimal_polynomial(sin(pi/21), x) == 4096*x**12 - 11264*x**10 + \
           11264*x**8 - 4992*x**6 + 960*x**4 - 64*x**2 + 1
    assert minimal_polynomial(cos(pi/9), x) == 8*x**3 - 6*x - 1

    ex = 2**Rational(1, 3)*exp(2*I*pi/3)
    assert minimal_polynomial(ex, x) == x**3 - 2

    raises(NotAlgebraic, lambda: minimal_polynomial(cos(pi*sqrt(2)), x))
    raises(NotAlgebraic, lambda: minimal_polynomial(sin(pi*sqrt(2)), x))
    raises(NotAlgebraic, lambda: minimal_polynomial(exp(1.618*I*pi), x))
    raises(NotAlgebraic, lambda: minimal_polynomial(exp(I*pi*sqrt(2)), x))

    # issue 5934
    ex = 1/(-36000 - 7200*sqrt(5) + (12*sqrt(10)*sqrt(sqrt(5) + 5) +
        24*sqrt(10)*sqrt(-sqrt(5) + 5))**2) + 1
    raises(ZeroDivisionError, lambda: minimal_polynomial(ex, x))

    ex = sqrt(1 + 2**Rational(1,3)) + sqrt(1 + 2**Rational(1,4)) + sqrt(2)
    mp = minimal_polynomial(ex, x)
    assert degree(mp) == 48 and mp.subs({x:0}) == -16630256576

    ex = tan(pi/5, evaluate=False)
    mp = minimal_polynomial(ex, x)
    assert mp == x**4 - 10*x**2 + 5
    assert mp.subs(x, tan(pi/5)).is_zero

    ex = tan(pi/6, evaluate=False)
    mp = minimal_polynomial(ex, x)
    assert mp == 3*x**2 - 1
    assert mp.subs(x, tan(pi/6)).is_zero

    ex = tan(pi/10, evaluate=False)
    mp = minimal_polynomial(ex, x)
    assert mp == 5*x**4 - 10*x**2 + 1
    assert mp.subs(x, tan(pi/10)).is_zero

    raises(NotAlgebraic, lambda: minimal_polynomial(tan(pi*sqrt(2)), x))


def test_minpoly_issue_7113():
    # see discussion in https://github.com/sympy/sympy/pull/2234
    from sympy.simplify.simplify import nsimplify
    r = nsimplify(pi, tolerance=0.000000001)
    mp = minimal_polynomial(r, x)
    assert mp == 1768292677839237920489538677417507171630859375*x**109 - \
    2734577732179183863586489182929671773182898498218854181690460140337930774573792597743853652058046464


def test_minpoly_issue_23677():
    r1 = CRootOf(4000000*x**3 - 239960000*x**2 + 4782399900*x - 31663998001, 0)
    r2 = CRootOf(4000000*x**3 - 239960000*x**2 + 4782399900*x - 31663998001, 1)
    num = (7680000000000000000*r1**4*r2**4 - 614323200000000000000*r1**4*r2**3
            + 18458112576000000000000*r1**4*r2**2 - 246896663036160000000000*r1**4*r2
            + 1240473830323209600000000*r1**4 - 614323200000000000000*r1**3*r2**4
            - 1476464424954240000000000*r1**3*r2**2 - 99225501687553535904000000*r1**3
            + 18458112576000000000000*r1**2*r2**4 - 1476464424954240000000000*r1**2*r2**3
            - 593391458458356671712000000*r1**2*r2 + 2981354896834339226880720000*r1**2
            - 246896663036160000000000*r1*r2**4 - 593391458458356671712000000*r1*r2**2
            - 39878756418031796275267195200*r1 + 1240473830323209600000000*r2**4
            - 99225501687553535904000000*r2**3 + 2981354896834339226880720000*r2**2 -
            39878756418031796275267195200*r2 + 200361370275616536577343808012)
    mp = (x**3 + 59426520028417434406408556687919*x**2 +
        1161475464966574421163316896737773190861975156439163671112508400*x +
        7467465541178623874454517208254940823818304424383315270991298807299003671748074773558707779600)
    assert minimal_polynomial(num, x) == mp


def test_minpoly_issue_7574():
    ex = -(-1)**Rational(1, 3) + (-1)**Rational(2,3)
    assert minimal_polynomial(ex, x) == x + 1


def test_choose_factor():
    # Test that this does not enter an infinite loop:
    bad_factors = [Poly(x-2, x), Poly(x+2, x)]
    raises(NotImplementedError, lambda: _choose_factor(bad_factors, x, sqrt(3)))


def test_minpoly_fraction_field():
    assert minimal_polynomial(1/x, y) == -x*y + 1
    assert minimal_polynomial(1 / (x + 1), y) == (x + 1)*y - 1

    assert minimal_polynomial(sqrt(x), y) == y**2 - x
    assert minimal_polynomial(sqrt(x + 1), y) == y**2 - x - 1
    assert minimal_polynomial(sqrt(x) / x, y) == x*y**2 - 1
    assert minimal_polynomial(sqrt(2) * sqrt(x), y) == y**2 - 2 * x
    assert minimal_polynomial(sqrt(2) + sqrt(x), y) == \
        y**4 + (-2*x - 4)*y**2 + x**2 - 4*x + 4

    assert minimal_polynomial(x**Rational(1,3), y) == y**3 - x
    assert minimal_polynomial(x**Rational(1,3) + sqrt(x), y) == \
        y**6 - 3*x*y**4 - 2*x*y**3 + 3*x**2*y**2 - 6*x**2*y - x**3 + x**2

    assert minimal_polynomial(sqrt(x) / z, y) == z**2*y**2 - x
    assert minimal_polynomial(sqrt(x) / (z + 1), y) == (z**2 + 2*z + 1)*y**2 - x

    assert minimal_polynomial(1/x, y, polys=True) == Poly(-x*y + 1, y, domain='ZZ(x)')
    assert minimal_polynomial(1 / (x + 1), y, polys=True) == \
        Poly((x + 1)*y - 1, y, domain='ZZ(x)')
    assert minimal_polynomial(sqrt(x), y, polys=True) == Poly(y**2 - x, y, domain='ZZ(x)')
    assert minimal_polynomial(sqrt(x) / z, y, polys=True) == \
        Poly(z**2*y**2 - x, y, domain='ZZ(x, z)')

    # this is (sqrt(1 + x**3)/x).integrate(x).diff(x) - sqrt(1 + x**3)/x
    a = sqrt(x)/sqrt(1 + x**(-3)) - sqrt(x**3 + 1)/x + 1/(x**Rational(5, 2)* \
        (1 + x**(-3))**Rational(3, 2)) + 1/(x**Rational(11, 2)*(1 + x**(-3))**Rational(3, 2))

    assert minimal_polynomial(a, y) == y

    raises(NotAlgebraic, lambda: minimal_polynomial(exp(x), y))
    raises(GeneratorsError, lambda: minimal_polynomial(sqrt(x), x))
    raises(GeneratorsError, lambda: minimal_polynomial(sqrt(x) - y, x))
    raises(NotImplementedError, lambda: minimal_polynomial(sqrt(x), y, compose=False))

@slow
def test_minpoly_fraction_field_slow():
    assert minimal_polynomial(minimal_polynomial(sqrt(x**Rational(1,5) - 1),
        y).subs(y, sqrt(x**Rational(1,5) - 1)), z) == z

def test_minpoly_domain():
    assert minimal_polynomial(sqrt(2), x, domain=QQ.algebraic_field(sqrt(2))) == \
        x - sqrt(2)
    assert minimal_polynomial(sqrt(8), x, domain=QQ.algebraic_field(sqrt(2))) == \
        x - 2*sqrt(2)
    assert minimal_polynomial(sqrt(Rational(3,2)), x,
        domain=QQ.algebraic_field(sqrt(2))) == 2*x**2 - 3

    raises(NotAlgebraic, lambda: minimal_polynomial(y, x, domain=QQ))


def test_issue_14831():
    a = -2*sqrt(2)*sqrt(12*sqrt(2) + 17)
    assert minimal_polynomial(a, x) == x**2 + 16*x - 8
    e = (-3*sqrt(12*sqrt(2) + 17) + 12*sqrt(2) +
         17 - 2*sqrt(2)*sqrt(12*sqrt(2) + 17))
    assert minimal_polynomial(e, x) == x


def test_issue_18248():
    assert nonlinsolve([x*y**3-sqrt(2)/3, x*y**6-4/(9*(sqrt(3)))],x,y) == \
            FiniteSet((sqrt(3)/2, sqrt(6)/3), (sqrt(3)/2, -sqrt(6)/6 - sqrt(2)*I/2),
            (sqrt(3)/2, -sqrt(6)/6 + sqrt(2)*I/2))


def test_issue_13230():
    c1 = Circle(Point2D(3, sqrt(5)), 5)
    c2 = Circle(Point2D(4, sqrt(7)), 6)
    assert intersection(c1, c2) == [Point2D(-1 + (-sqrt(7) + sqrt(5))*(-2*sqrt(7)/29
    + 9*sqrt(5)/29 + sqrt(196*sqrt(35) + 1941)/29), -2*sqrt(7)/29 + 9*sqrt(5)/29
    + sqrt(196*sqrt(35) + 1941)/29), Point2D(-1 + (-sqrt(7) + sqrt(5))*(-sqrt(196*sqrt(35)
    + 1941)/29 - 2*sqrt(7)/29 + 9*sqrt(5)/29), -sqrt(196*sqrt(35) + 1941)/29 - 2*sqrt(7)/29 + 9*sqrt(5)/29)]

def test_issue_19760():
    e = 1/(sqrt(1 + sqrt(2)) - sqrt(2)*sqrt(1 + sqrt(2))) + 1
    mp_expected = x**4 - 4*x**3 + 4*x**2 - 2

    for comp in (True, False):
        mp = Poly(minimal_polynomial(e, compose=comp))
        assert mp(x) == mp_expected, "minimal_polynomial(e, compose=%s) = %s; %s expected" % (comp, mp(x), mp_expected)


def test_issue_20163():
    assert apart(1/(x**6+1), extension=[sqrt(3), I]) == \
        (sqrt(3) + I)/(2*x + sqrt(3) + I)/6 + \
        (sqrt(3) - I)/(2*x + sqrt(3) - I)/6 - \
        (sqrt(3) - I)/(2*x - sqrt(3) + I)/6 - \
        (sqrt(3) + I)/(2*x - sqrt(3) - I)/6 + \
        I/(x + I)/6 - I/(x - I)/6


def test_issue_22559():
    alpha = AlgebraicNumber(sqrt(2))
    assert minimal_polynomial(alpha**3, x) == x**2 - 8


def test_issue_22561():
    a = AlgebraicNumber(sqrt(2) + sqrt(3), [S(1) / 2, 0, S(-9) / 2, 0], gen=x)
    assert a.as_expr() == sqrt(2)
    assert minimal_polynomial(a, x) == x**2 - 2
    assert minimal_polynomial(a**3, x) == x**2 - 8


def test_separate_sq_not_impl():
    raises(NotImplementedError, lambda: _separate_sq(x**(S(1)/3) + x))


def test_minpoly_op_algebraic_element_not_impl():
    raises(NotImplementedError,
           lambda: _minpoly_op_algebraic_element(Pow, sqrt(2), sqrt(3), x, QQ))


def test_minpoly_groebner():
    assert _minpoly_groebner(S(2)/3, x, Poly) == 3*x - 2
    assert _minpoly_groebner(
        (sqrt(2) + 3)*(sqrt(2) + 1), x, Poly) == x**2 - 10*x - 7
    assert _minpoly_groebner((sqrt(2) + 3)**(S(1)/3)*(sqrt(2) + 1)**(S(1)/3),
                             x, Poly) == x**6 - 10*x**3 - 7
    assert _minpoly_groebner((sqrt(2) + 3)**(-S(1)/3)*(sqrt(2) + 1)**(S(1)/3),
                             x, Poly) == 7*x**6 - 2*x**3 - 1
    raises(NotAlgebraic, lambda: _minpoly_groebner(pi**2, x, Poly))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\tests\test_gruntz.py ===
from sympy.core import EulerGamma
from sympy.core.function import Function
from sympy.core.numbers import (E, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acot, atan, cos, sin)
from sympy.functions.special.error_functions import (Ei, erf)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.polys.polytools import cancel
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.series.gruntz import compare, mrv, rewrite, mrv_leadterm, gruntz, \
    sign
from sympy.testing.pytest import XFAIL, raises, skip, slow

"""
This test suite is testing the limit algorithm using the bottom up approach.
See the documentation in limits2.py. The algorithm itself is highly recursive
by nature, so "compare" is logically the lowest part of the algorithm, yet in
some sense it's the most complex part, because it needs to calculate a limit
to return the result.

Nevertheless, the rest of the algorithm depends on compare working correctly.
"""

x = Symbol('x', real=True)
m = Symbol('m', real=True)


runslow = False


def _sskip():
    if not runslow:
        skip("slow")


@slow
def test_gruntz_evaluation():
    # Gruntz' thesis pp. 122 to 123
    # 8.1
    assert gruntz(exp(x)*(exp(1/x - exp(-x)) - exp(1/x)), x, oo) == -1
    # 8.2
    assert gruntz(exp(x)*(exp(1/x + exp(-x) + exp(-x**2))
                  - exp(1/x - exp(-exp(x)))), x, oo) == 1
    # 8.3
    assert gruntz(exp(exp(x - exp(-x))/(1 - 1/x)) - exp(exp(x)), x, oo) is oo
    # 8.5
    assert gruntz(exp(exp(exp(x + exp(-x)))) / exp(exp(exp(x))), x, oo) is oo
    # 8.6
    assert gruntz(exp(exp(exp(x))) / exp(exp(exp(x - exp(-exp(x))))),
                  x, oo) is oo
    # 8.7
    assert gruntz(exp(exp(exp(x))) / exp(exp(exp(x - exp(-exp(exp(x)))))),
                  x, oo) == 1
    # 8.8
    assert gruntz(exp(exp(x)) / exp(exp(x - exp(-exp(exp(x))))), x, oo) == 1
    # 8.9
    assert gruntz(log(x)**2 * exp(sqrt(log(x))*(log(log(x)))**2
                  * exp(sqrt(log(log(x))) * (log(log(log(x))))**3)) / sqrt(x),
                  x, oo) == 0
    # 8.10
    assert gruntz((x*log(x)*(log(x*exp(x) - x**2))**2)
                  / (log(log(x**2 + 2*exp(exp(3*x**3*log(x)))))), x, oo) == Rational(1, 3)
    # 8.11
    assert gruntz((exp(x*exp(-x)/(exp(-x) + exp(-2*x**2/(x + 1)))) - exp(x))/x,
                  x, oo) == -exp(2)
    # 8.12
    assert gruntz((3**x + 5**x)**(1/x), x, oo) == 5
    # 8.13
    assert gruntz(x/log(x**(log(x**(log(2)/log(x))))), x, oo) is oo
    # 8.14
    assert gruntz(exp(exp(2*log(x**5 + x)*log(log(x))))
                  / exp(exp(10*log(x)*log(log(x)))), x, oo) is oo
    # 8.15
    assert gruntz(exp(exp(Rational(5, 2)*x**Rational(-5, 7) + Rational(21, 8)*x**Rational(6, 11)
                          + 2*x**(-8) + Rational(54, 17)*x**Rational(49, 45)))**8
                  / log(log(-log(Rational(4, 3)*x**Rational(-5, 14))))**Rational(7, 6), x, oo) is oo
    # 8.16
    assert gruntz((exp(4*x*exp(-x)/(1/exp(x) + 1/exp(2*x**2/(x + 1)))) - exp(x))
                  / exp(x)**4, x, oo) == 1
    # 8.17
    assert gruntz(exp(x*exp(-x)/(exp(-x) + exp(-2*x**2/(x + 1))))/exp(x), x, oo) \
        == 1
    # 8.19
    assert gruntz(log(x)*(log(log(x) + log(log(x))) - log(log(x)))
                  / (log(log(x) + log(log(log(x))))), x, oo) == 1
    # 8.20
    assert gruntz(exp((log(log(x + exp(log(x)*log(log(x))))))
                  / (log(log(log(exp(x) + x + log(x)))))), x, oo) == E
    # Another
    assert gruntz(exp(exp(exp(x + exp(-x)))) / exp(exp(x)), x, oo) is oo


def test_gruntz_evaluation_slow():
    _sskip()
    # 8.4
    assert gruntz(exp(exp(exp(x)/(1 - 1/x)))
                  - exp(exp(exp(x)/(1 - 1/x - log(x)**(-log(x))))), x, oo) is -oo
    # 8.18
    assert gruntz((exp(exp(-x/(1 + exp(-x))))*exp(-x/(1 + exp(-x/(1 + exp(-x)))))
                   *exp(exp(-x + exp(-x/(1 + exp(-x))))))
                  / (exp(-x/(1 + exp(-x))))**2 - exp(x) + x, x, oo) == 2


@slow
def test_gruntz_eval_special():
    # Gruntz, p. 126
    assert gruntz(exp(x)*(sin(1/x + exp(-x)) - sin(1/x + exp(-x**2))), x, oo) == 1
    assert gruntz((erf(x - exp(-exp(x))) - erf(x)) * exp(exp(x)) * exp(x**2),
                  x, oo) == -2/sqrt(pi)
    assert gruntz(exp(exp(x)) * (exp(sin(1/x + exp(-exp(x)))) - exp(sin(1/x))),
                  x, oo) == 1
    assert gruntz(exp(x)*(gamma(x + exp(-x)) - gamma(x)), x, oo) is oo
    assert gruntz(exp(exp(digamma(digamma(x))))/x, x, oo) == exp(Rational(-1, 2))
    assert gruntz(exp(exp(digamma(log(x))))/x, x, oo) == exp(Rational(-1, 2))
    assert gruntz(digamma(digamma(digamma(x))), x, oo) is oo
    assert gruntz(loggamma(loggamma(x)), x, oo) is oo
    assert gruntz(((gamma(x + 1/gamma(x)) - gamma(x))/log(x) - cos(1/x))
                  * x*log(x), x, oo) == Rational(-1, 2)
    assert gruntz(x * (gamma(x - 1/gamma(x)) - gamma(x) + log(x)), x, oo) \
        == S.Half
    assert gruntz((gamma(x + 1/gamma(x)) - gamma(x)) / log(x), x, oo) == 1


def test_gruntz_eval_special_slow():
    _sskip()
    assert gruntz(gamma(x + 1)/sqrt(2*pi)
                  - exp(-x)*(x**(x + S.Half) + x**(x - S.Half)/12), x, oo) is oo
    assert gruntz(exp(exp(exp(digamma(digamma(digamma(x))))))/x, x, oo) == 0


@XFAIL
def test_grunts_eval_special_slow_sometimes_fail():
    _sskip()
    # XXX This sometimes fails!!!
    assert gruntz(exp(gamma(x - exp(-x))*exp(1/x)) - exp(gamma(x)), x, oo) is oo


def test_gruntz_Ei():
    assert gruntz((Ei(x - exp(-exp(x))) - Ei(x)) *exp(-x)*exp(exp(x))*x, x, oo) == -1


@XFAIL
def test_gruntz_eval_special_fail():
    # TODO zeta function series
    assert gruntz(
        exp((log(2) + 1)*x) * (zeta(x + exp(-x)) - zeta(x)), x, oo) == -log(2)

    # TODO 8.35 - 8.37 (bessel, max-min)


def test_gruntz_hyperbolic():
    assert gruntz(cosh(x), x, oo) is oo
    assert gruntz(cosh(x), x, -oo) is oo
    assert gruntz(sinh(x), x, oo) is oo
    assert gruntz(sinh(x), x, -oo) is -oo
    assert gruntz(2*cosh(x)*exp(x), x, oo) is oo
    assert gruntz(2*cosh(x)*exp(x), x, -oo) == 1
    assert gruntz(2*sinh(x)*exp(x), x, oo) is oo
    assert gruntz(2*sinh(x)*exp(x), x, -oo) == -1
    assert gruntz(tanh(x), x, oo) == 1
    assert gruntz(tanh(x), x, -oo) == -1
    assert gruntz(coth(x), x, oo) == 1
    assert gruntz(coth(x), x, -oo) == -1


def test_compare1():
    assert compare(2, x, x) == "<"
    assert compare(x, exp(x), x) == "<"
    assert compare(exp(x), exp(x**2), x) == "<"
    assert compare(exp(x**2), exp(exp(x)), x) == "<"
    assert compare(1, exp(exp(x)), x) == "<"

    assert compare(x, 2, x) == ">"
    assert compare(exp(x), x, x) == ">"
    assert compare(exp(x**2), exp(x), x) == ">"
    assert compare(exp(exp(x)), exp(x**2), x) == ">"
    assert compare(exp(exp(x)), 1, x) == ">"

    assert compare(2, 3, x) == "="
    assert compare(3, -5, x) == "="
    assert compare(2, -5, x) == "="

    assert compare(x, x**2, x) == "="
    assert compare(x**2, x**3, x) == "="
    assert compare(x**3, 1/x, x) == "="
    assert compare(1/x, x**m, x) == "="
    assert compare(x**m, -x, x) == "="

    assert compare(exp(x), exp(-x), x) == "="
    assert compare(exp(-x), exp(2*x), x) == "="
    assert compare(exp(2*x), exp(x)**2, x) == "="
    assert compare(exp(x)**2, exp(x + exp(-x)), x) == "="
    assert compare(exp(x), exp(x + exp(-x)), x) == "="

    assert compare(exp(x**2), 1/exp(x**2), x) == "="


def test_compare2():
    assert compare(exp(x), x**5, x) == ">"
    assert compare(exp(x**2), exp(x)**2, x) == ">"
    assert compare(exp(x), exp(x + exp(-x)), x) == "="
    assert compare(exp(x + exp(-x)), exp(x), x) == "="
    assert compare(exp(x + exp(-x)), exp(-x), x) == "="
    assert compare(exp(-x), x, x) == ">"
    assert compare(x, exp(-x), x) == "<"
    assert compare(exp(x + 1/x), x, x) == ">"
    assert compare(exp(-exp(x)), exp(x), x) == ">"
    assert compare(exp(exp(-exp(x)) + x), exp(-exp(x)), x) == "<"


def test_compare3():
    assert compare(exp(exp(x)), exp(x + exp(-exp(x))), x) == ">"


def test_sign1():
    assert sign(Rational(0), x) == 0
    assert sign(Rational(3), x) == 1
    assert sign(Rational(-5), x) == -1
    assert sign(log(x), x) == 1
    assert sign(exp(-x), x) == 1
    assert sign(exp(x), x) == 1
    assert sign(-exp(x), x) == -1
    assert sign(3 - 1/x, x) == 1
    assert sign(-3 - 1/x, x) == -1
    assert sign(sin(1/x), x) == 1
    assert sign((x**Integer(2)), x) == 1
    assert sign(x**2, x) == 1
    assert sign(x**5, x) == 1


def test_sign2():
    assert sign(x, x) == 1
    assert sign(-x, x) == -1
    y = Symbol("y", positive=True)
    assert sign(y, x) == 1
    assert sign(-y, x) == -1
    assert sign(y*x, x) == 1
    assert sign(-y*x, x) == -1


def mmrv(a, b):
    return set(mrv(a, b)[0].keys())


def test_mrv1():
    assert mmrv(x, x) == {x}
    assert mmrv(x + 1/x, x) == {x}
    assert mmrv(x**2, x) == {x}
    assert mmrv(log(x), x) == {x}
    assert mmrv(exp(x), x) == {exp(x)}
    assert mmrv(exp(-x), x) == {exp(-x)}
    assert mmrv(exp(x**2), x) == {exp(x**2)}
    assert mmrv(-exp(1/x), x) == {x}
    assert mmrv(exp(x + 1/x), x) == {exp(x + 1/x)}


def test_mrv2a():
    assert mmrv(exp(x + exp(-exp(x))), x) == {exp(-exp(x))}
    assert mmrv(exp(x + exp(-x)), x) == {exp(x + exp(-x)), exp(-x)}
    assert mmrv(exp(1/x + exp(-x)), x) == {exp(-x)}

#sometimes infinite recursion due to log(exp(x**2)) not simplifying


def test_mrv2b():
    assert mmrv(exp(x + exp(-x**2)), x) == {exp(-x**2)}

#sometimes infinite recursion due to log(exp(x**2)) not simplifying


def test_mrv2c():
    assert mmrv(
        exp(-x + 1/x**2) - exp(x + 1/x), x) == {exp(x + 1/x), exp(1/x**2 - x)}

#sometimes infinite recursion due to log(exp(x**2)) not simplifying


def test_mrv3():
    assert mmrv(exp(x**2) + x*exp(x) + log(x)**x/x, x) == {exp(x**2)}
    assert mmrv(
        exp(x)*(exp(1/x + exp(-x)) - exp(1/x)), x) == {exp(x), exp(-x)}
    assert mmrv(log(
        x**2 + 2*exp(exp(3*x**3*log(x)))), x) == {exp(exp(3*x**3*log(x)))}
    assert mmrv(log(x - log(x))/log(x), x) == {x}
    assert mmrv(
        (exp(1/x - exp(-x)) - exp(1/x))*exp(x), x) == {exp(x), exp(-x)}
    assert mmrv(
        1/exp(-x + exp(-x)) - exp(x), x) == {exp(x), exp(-x), exp(x - exp(-x))}
    assert mmrv(log(log(x*exp(x*exp(x)) + 1)), x) == {exp(x*exp(x))}
    assert mmrv(exp(exp(log(log(x) + 1/x))), x) == {x}


def test_mrv4():
    ln = log
    assert mmrv((ln(ln(x) + ln(ln(x))) - ln(ln(x)))/ln(ln(x) + ln(ln(ln(x))))*ln(x),
            x) == {x}
    assert mmrv(log(log(x*exp(x*exp(x)) + 1)) - exp(exp(log(log(x) + 1/x))), x) == \
        {exp(x*exp(x))}


def mrewrite(a, b, c):
    return rewrite(a[1], a[0], b, c)


def test_rewrite1():
    e = exp(x)
    assert mrewrite(mrv(e, x), x, m) == (1/m, -x)
    e = exp(x**2)
    assert mrewrite(mrv(e, x), x, m) == (1/m, -x**2)
    e = exp(x + 1/x)
    assert mrewrite(mrv(e, x), x, m) == (1/m, -x - 1/x)
    e = 1/exp(-x + exp(-x)) - exp(x)
    assert mrewrite(mrv(e, x), x, m) == ((-m*exp(m) + m)*exp(-m)/m**2, -x)


def test_rewrite2():
    e = exp(x)*log(log(exp(x)))
    assert mmrv(e, x) == {exp(x)}
    assert mrewrite(mrv(e, x), x, m) == (1/m*log(x), -x)

#sometimes infinite recursion due to log(exp(x**2)) not simplifying


def test_rewrite3():
    e = exp(-x + 1/x**2) - exp(x + 1/x)
    #both of these are correct and should be equivalent:
    assert mrewrite(mrv(e, x), x, m) in [(-1/m + m*exp(
        (x**2 + x)/x**3), -x - 1/x), ((m**2 - exp((x**2 + x)/x**3))/m, x**(-2) - x)]


def test_mrv_leadterm1():
    assert mrv_leadterm(-exp(1/x), x) == (-1, 0)
    assert mrv_leadterm(1/exp(-x + exp(-x)) - exp(x), x) == (-1, 0)
    assert mrv_leadterm(
        (exp(1/x - exp(-x)) - exp(1/x))*exp(x), x) == (-exp(1/x), 0)


def test_mrv_leadterm2():
    #Gruntz: p51, 3.25
    assert mrv_leadterm((log(exp(x) + x) - x)/log(exp(x) + log(x))*exp(x), x) == \
        (1, 0)


def test_mrv_leadterm3():
    #Gruntz: p56, 3.27
    assert mmrv(exp(-x + exp(-x)*exp(-x*log(x))), x) == {exp(-x - x*log(x))}
    assert mrv_leadterm(exp(-x + exp(-x)*exp(-x*log(x))), x) == (exp(-x), 0)


def test_limit1():
    assert gruntz(x, x, oo) is oo
    assert gruntz(x, x, -oo) is -oo
    assert gruntz(-x, x, oo) is -oo
    assert gruntz(x**2, x, -oo) is oo
    assert gruntz(-x**2, x, oo) is -oo
    assert gruntz(x*log(x), x, 0, dir="+") == 0
    assert gruntz(1/x, x, oo) == 0
    assert gruntz(exp(x), x, oo) is oo
    assert gruntz(-exp(x), x, oo) is -oo
    assert gruntz(exp(x)/x, x, oo) is oo
    assert gruntz(1/x - exp(-x), x, oo) == 0
    assert gruntz(x + 1/x, x, oo) is oo


def test_limit2():
    assert gruntz(x**x, x, 0, dir="+") == 1
    assert gruntz((exp(x) - 1)/x, x, 0) == 1
    assert gruntz(1 + 1/x, x, oo) == 1
    assert gruntz(-exp(1/x), x, oo) == -1
    assert gruntz(x + exp(-x), x, oo) is oo
    assert gruntz(x + exp(-x**2), x, oo) is oo
    assert gruntz(x + exp(-exp(x)), x, oo) is oo
    assert gruntz(13 + 1/x - exp(-x), x, oo) == 13


def test_limit3():
    a = Symbol('a')
    assert gruntz(x - log(1 + exp(x)), x, oo) == 0
    assert gruntz(x - log(a + exp(x)), x, oo) == 0
    assert gruntz(exp(x)/(1 + exp(x)), x, oo) == 1
    assert gruntz(exp(x)/(a + exp(x)), x, oo) == 1


def test_limit4():
    #issue 3463
    assert gruntz((3**x + 5**x)**(1/x), x, oo) == 5
    #issue 3463
    assert gruntz((3**(1/x) + 5**(1/x))**x, x, 0) == 5


@XFAIL
def test_MrvTestCase_page47_ex3_21():
    h = exp(-x/(1 + exp(-x)))
    expr = exp(h)*exp(-x/(1 + h))*exp(exp(-x + h))/h**2 - exp(x) + x
    assert mmrv(expr, x) == {1/h, exp(-x), exp(x), exp(x - h), exp(x/(1 + h))}


def test_gruntz_I():
    y = Symbol("y")
    assert gruntz(I*x, x, oo) == I*oo
    assert gruntz(y*I*x, x, oo) == y*I*oo
    assert gruntz(y*3*I*x, x, oo) == y*I*oo
    assert gruntz(y*3*sin(I)*x, x, oo) == y*I*oo


def test_issue_4814():
    assert gruntz((x + 1)**(1/log(x + 1)), x, oo) == E


def test_intractable():
    assert gruntz(1/gamma(x), x, oo) == 0
    assert gruntz(1/loggamma(x), x, oo) == 0
    assert gruntz(gamma(x)/loggamma(x), x, oo) is oo
    assert gruntz(exp(gamma(x))/gamma(x), x, oo) is oo
    assert gruntz(gamma(x), x, 3) == 2
    assert gruntz(gamma(Rational(1, 7) + 1/x), x, oo) == gamma(Rational(1, 7))
    assert gruntz(log(x**x)/log(gamma(x)), x, oo) == 1
    assert gruntz(log(gamma(gamma(x)))/exp(x), x, oo) is oo


def test_aseries_trig():
    assert cancel(gruntz(1/log(atan(x)), x, oo)
           - 1/(log(pi) + log(S.Half))) == 0
    assert gruntz(1/acot(x), x, -oo) is -oo


def test_exp_log_series():
    assert gruntz(x/log(log(x*exp(x))), x, oo) is oo


def test_issue_3644():
    assert gruntz(((x**7 + x + 1)/(2**x + x**2))**(-1/x), x, oo) == 2


def test_issue_6843():
    n = Symbol('n', integer=True, positive=True)
    r = (n + 1)*x**(n + 1)/(x**(n + 1) - 1) - x/(x - 1)
    assert gruntz(r, x, 1).simplify() == n/2


def test_issue_4190():
    assert gruntz(x - gamma(1/x), x, oo) == S.EulerGamma


@XFAIL
def test_issue_5172():
    n = Symbol('n')
    r = Symbol('r', positive=True)
    c = Symbol('c')
    p = Symbol('p', positive=True)
    m = Symbol('m', negative=True)
    expr = ((2*n*(n - r + 1)/(n + r*(n - r + 1)))**c + \
        (r - 1)*(n*(n - r + 2)/(n + r*(n - r + 1)))**c - n)/(n**c - n)
    expr = expr.subs(c, c + 1)
    assert gruntz(expr.subs(c, m), n, oo) == 1
    # fail:
    assert gruntz(expr.subs(c, p), n, oo).simplify() == \
        (2**(p + 1) + r - 1)/(r + 1)**(p + 1)


def test_issue_4109():
    assert gruntz(1/gamma(x), x, 0) == 0
    assert gruntz(x*gamma(x), x, 0) == 1


def test_issue_6682():
    assert gruntz(exp(2*Ei(-x))/x**2, x, 0) == exp(2*EulerGamma)


def test_issue_7096():
    from sympy.functions import sign
    assert gruntz(x**-pi, x, 0, dir='-') == oo*sign((-1)**(-pi))


def test_issue_7391_8166():
    f = Function('f')
    # limit should depend on the continuity of the expression at the point passed
    raises(ValueError, lambda: gruntz(f(x), x, 4))
    raises(ValueError, lambda: gruntz(x*f(x)**2/(x**2 + f(x)**4), x, 0))


def test_issue_24210_25885():
    eq = exp(x)/(1+1/x)**x**2
    ans = sqrt(E)
    assert gruntz(eq, x, oo) == ans
    assert gruntz(1/eq, x, oo) == 1/ans

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\test_sorting.py ===
from collections import defaultdict
from datetime import datetime
from itertools import product

import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    MultiIndex,
    Series,
    array,
    concat,
    merge,
)
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
    _decons_group_index,
    get_group_index,
    is_int64_overflow_possible,
    lexsort_indexer,
    nargsort,
)


@pytest.fixture
def left_right():
    low, high, n = -1 << 10, 1 << 10, 1 << 20
    left = DataFrame(
        np.random.default_rng(2).integers(low, high, (n, 7)), columns=list("ABCDEFG")
    )
    left["left"] = left.sum(axis=1)

    # one-2-one match
    i = np.random.default_rng(2).permutation(len(left))
    right = left.iloc[i].copy()
    right.columns = right.columns[:-1].tolist() + ["right"]
    right.index = np.arange(len(right))
    right["right"] *= -1
    return left, right


class TestSorting:
    @pytest.mark.slow
    def test_int64_overflow(self):
        B = np.concatenate((np.arange(1000), np.arange(1000), np.arange(500)))
        A = np.arange(2500)
        df = DataFrame(
            {
                "A": A,
                "B": B,
                "C": A,
                "D": B,
                "E": A,
                "F": B,
                "G": A,
                "H": B,
                "values": np.random.default_rng(2).standard_normal(2500),
            }
        )

        lg = df.groupby(["A", "B", "C", "D", "E", "F", "G", "H"])
        rg = df.groupby(["H", "G", "F", "E", "D", "C", "B", "A"])

        left = lg.sum()["values"]
        right = rg.sum()["values"]

        exp_index, _ = left.index.sortlevel()
        tm.assert_index_equal(left.index, exp_index)

        exp_index, _ = right.index.sortlevel(0)
        tm.assert_index_equal(right.index, exp_index)

        tups = list(map(tuple, df[["A", "B", "C", "D", "E", "F", "G", "H"]].values))
        tups = com.asarray_tuplesafe(tups)

        expected = df.groupby(tups).sum()["values"]

        for k, v in expected.items():
            assert left[k] == right[k[::-1]]
            assert left[k] == v
        assert len(left) == len(right)

    def test_int64_overflow_groupby_large_range(self):
        # GH9096
        values = range(55109)
        data = DataFrame.from_dict({"a": values, "b": values, "c": values, "d": values})
        grouped = data.groupby(["a", "b", "c", "d"])
        assert len(grouped) == len(values)

    @pytest.mark.parametrize("agg", ["mean", "median"])
    def test_int64_overflow_groupby_large_df_shuffled(self, agg):
        rs = np.random.default_rng(2)
        arr = rs.integers(-1 << 12, 1 << 12, (1 << 15, 5))
        i = rs.choice(len(arr), len(arr) * 4)
        arr = np.vstack((arr, arr[i]))  # add some duplicate rows

        i = rs.permutation(len(arr))
        arr = arr[i]  # shuffle rows

        df = DataFrame(arr, columns=list("abcde"))
        df["jim"], df["joe"] = np.zeros((2, len(df)))
        gr = df.groupby(list("abcde"))

        # verify this is testing what it is supposed to test!
        assert is_int64_overflow_possible(gr._grouper.shape)

        mi = MultiIndex.from_arrays(
            [ar.ravel() for ar in np.array_split(np.unique(arr, axis=0), 5, axis=1)],
            names=list("abcde"),
        )

        res = DataFrame(
            np.zeros((len(mi), 2)), columns=["jim", "joe"], index=mi
        ).sort_index()

        tm.assert_frame_equal(getattr(gr, agg)(), res)

    @pytest.mark.parametrize(
        "order, na_position, exp",
        [
            [
                True,
                "last",
                list(range(5, 105)) + list(range(5)) + list(range(105, 110)),
            ],
            [
                True,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(5, 105)),
            ],
            [
                False,
                "last",
                list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110)),
            ],
            [
                False,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1)),
            ],
        ],
    )
    def test_lexsort_indexer(self, order, na_position, exp):
        keys = [[np.nan] * 5 + list(range(100)) + [np.nan] * 5]
        result = lexsort_indexer(keys, orders=order, na_position=na_position)
        tm.assert_numpy_array_equal(result, np.array(exp, dtype=np.intp))

    @pytest.mark.parametrize(
        "ascending, na_position, exp",
        [
            [
                True,
                "last",
                list(range(5, 105)) + list(range(5)) + list(range(105, 110)),
            ],
            [
                True,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(5, 105)),
            ],
            [
                False,
                "last",
                list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110)),
            ],
            [
                False,
                "first",
                list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1)),
            ],
        ],
    )
    def test_nargsort(self, ascending, na_position, exp):
        # list places NaNs last, np.array(..., dtype="O") may not place NaNs first
        items = np.array([np.nan] * 5 + list(range(100)) + [np.nan] * 5, dtype="O")

        # mergesort is the most difficult to get right because we want it to be
        # stable.

        # According to numpy/core/tests/test_multiarray, """The number of
        # sorted items must be greater than ~50 to check the actual algorithm
        # because quick and merge sort fall over to insertion sort for small
        # arrays."""

        result = nargsort(
            items, kind="mergesort", ascending=ascending, na_position=na_position
        )
        tm.assert_numpy_array_equal(result, np.array(exp), check_dtype=False)


class TestMerge:
    def test_int64_overflow_outer_merge(self):
        # #2690, combinatorial explosion
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 7)),
            columns=list("ABCDEF") + ["G1"],
        )
        df2 = DataFrame(
            np.random.default_rng(3).standard_normal((1000, 7)),
            columns=list("ABCDEF") + ["G2"],
        )
        result = merge(df1, df2, how="outer")
        assert len(result) == 2000

    @pytest.mark.slow
    def test_int64_overflow_check_sum_col(self, left_right):
        left, right = left_right

        out = merge(left, right, how="outer")
        assert len(out) == len(left)
        tm.assert_series_equal(out["left"], -out["right"], check_names=False)
        result = out.iloc[:, :-2].sum(axis=1)
        tm.assert_series_equal(out["left"], result, check_names=False)
        assert result.name is None

    @pytest.mark.slow
    @pytest.mark.parametrize("how", ["left", "right", "outer", "inner"])
    def test_int64_overflow_how_merge(self, left_right, how):
        left, right = left_right

        out = merge(left, right, how="outer")
        out.sort_values(out.columns.tolist(), inplace=True)
        out.index = np.arange(len(out))
        tm.assert_frame_equal(out, merge(left, right, how=how, sort=True))

    @pytest.mark.slow
    def test_int64_overflow_sort_false_order(self, left_right):
        left, right = left_right

        # check that left merge w/ sort=False maintains left frame order
        out = merge(left, right, how="left", sort=False)
        tm.assert_frame_equal(left, out[left.columns.tolist()])

        out = merge(right, left, how="left", sort=False)
        tm.assert_frame_equal(right, out[right.columns.tolist()])

    @pytest.mark.slow
    @pytest.mark.parametrize("how", ["left", "right", "outer", "inner"])
    @pytest.mark.parametrize("sort", [True, False])
    def test_int64_overflow_one_to_many_none_match(self, how, sort):
        # one-2-many/none match
        low, high, n = -1 << 10, 1 << 10, 1 << 11
        left = DataFrame(
            np.random.default_rng(2).integers(low, high, (n, 7)).astype("int64"),
            columns=list("ABCDEFG"),
        )

        # confirm that this is checking what it is supposed to check
        shape = left.apply(Series.nunique).values
        assert is_int64_overflow_possible(shape)

        # add duplicates to left frame
        left = concat([left, left], ignore_index=True)

        right = DataFrame(
            np.random.default_rng(3).integers(low, high, (n // 2, 7)).astype("int64"),
            columns=list("ABCDEFG"),
        )

        # add duplicates & overlap with left to the right frame
        i = np.random.default_rng(4).choice(len(left), n)
        right = concat([right, right, left.iloc[i]], ignore_index=True)

        left["left"] = np.random.default_rng(2).standard_normal(len(left))
        right["right"] = np.random.default_rng(2).standard_normal(len(right))

        # shuffle left & right frames
        i = np.random.default_rng(5).permutation(len(left))
        left = left.iloc[i].copy()
        left.index = np.arange(len(left))

        i = np.random.default_rng(6).permutation(len(right))
        right = right.iloc[i].copy()
        right.index = np.arange(len(right))

        # manually compute outer merge
        ldict, rdict = defaultdict(list), defaultdict(list)

        for idx, row in left.set_index(list("ABCDEFG")).iterrows():
            ldict[idx].append(row["left"])

        for idx, row in right.set_index(list("ABCDEFG")).iterrows():
            rdict[idx].append(row["right"])

        vals = []
        for k, lval in ldict.items():
            rval = rdict.get(k, [np.nan])
            for lv, rv in product(lval, rval):
                vals.append(
                    k
                    + (
                        lv,
                        rv,
                    )
                )

        for k, rval in rdict.items():
            if k not in ldict:
                vals.extend(
                    k
                    + (
                        np.nan,
                        rv,
                    )
                    for rv in rval
                )

        def align(df):
            df = df.sort_values(df.columns.tolist())
            df.index = np.arange(len(df))
            return df

        out = DataFrame(vals, columns=list("ABCDEFG") + ["left", "right"])
        out = align(out)

        jmask = {
            "left": out["left"].notna(),
            "right": out["right"].notna(),
            "inner": out["left"].notna() & out["right"].notna(),
            "outer": np.ones(len(out), dtype="bool"),
        }

        mask = jmask[how]
        frame = align(out[mask].copy())
        assert mask.all() ^ mask.any() or how == "outer"

        res = merge(left, right, how=how, sort=sort)
        if sort:
            kcols = list("ABCDEFG")
            tm.assert_frame_equal(
                res[kcols].copy(), res[kcols].sort_values(kcols, kind="mergesort")
            )

        # as in GH9092 dtypes break with outer/right join
        # 2021-12-18: dtype does not break anymore
        tm.assert_frame_equal(frame, align(res))


@pytest.mark.parametrize(
    "codes_list, shape",
    [
        [
            [
                np.tile([0, 1, 2, 3, 0, 1, 2, 3], 100).astype(np.int64),
                np.tile([0, 2, 4, 3, 0, 1, 2, 3], 100).astype(np.int64),
                np.tile([5, 1, 0, 2, 3, 0, 5, 4], 100).astype(np.int64),
            ],
            (4, 5, 6),
        ],
        [
            [
                np.tile(np.arange(10000, dtype=np.int64), 5),
                np.tile(np.arange(10000, dtype=np.int64), 5),
            ],
            (10000, 10000),
        ],
    ],
)
def test_decons(codes_list, shape):
    group_index = get_group_index(codes_list, shape, sort=True, xnull=True)
    codes_list2 = _decons_group_index(group_index, shape)

    for a, b in zip(codes_list, codes_list2):
        tm.assert_numpy_array_equal(a, b)


class TestSafeSort:
    @pytest.mark.parametrize(
        "arg, exp",
        [
            [[3, 1, 2, 0, 4], [0, 1, 2, 3, 4]],
            [
                np.array(list("baaacb"), dtype=object),
                np.array(list("aaabbc"), dtype=object),
            ],
            [[], []],
        ],
    )
    def test_basic_sort(self, arg, exp):
        result = safe_sort(np.array(arg))
        expected = np.array(exp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("verify", [True, False])
    @pytest.mark.parametrize(
        "codes, exp_codes",
        [
            [[0, 1, 1, 2, 3, 0, -1, 4], [3, 1, 1, 2, 0, 3, -1, 4]],
            [[], []],
        ],
    )
    def test_codes(self, verify, codes, exp_codes):
        values = np.array([3, 1, 2, 0, 4])
        expected = np.array([0, 1, 2, 3, 4])

        result, result_codes = safe_sort(
            values, codes, use_na_sentinel=True, verify=verify
        )
        expected_codes = np.array(exp_codes, dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_codes_out_of_bound(self):
        values = np.array([3, 1, 2, 0, 4])
        expected = np.array([0, 1, 2, 3, 4])

        # out of bound indices
        codes = [0, 101, 102, 2, 3, 0, 99, 4]
        result, result_codes = safe_sort(values, codes, use_na_sentinel=True)
        expected_codes = np.array([3, -1, -1, 2, 0, 3, -1, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_mixed_integer(self):
        values = np.array(["b", 1, 0, "a", 0, "b"], dtype=object)
        result = safe_sort(values)
        expected = np.array([0, 0, 1, "a", "b", "b"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_mixed_integer_with_codes(self):
        values = np.array(["b", 1, 0, "a"], dtype=object)
        codes = [0, 1, 2, 3, 0, -1, 1]
        result, result_codes = safe_sort(values, codes)
        expected = np.array([0, 1, "a", "b"], dtype=object)
        expected_codes = np.array([3, 1, 0, 2, 3, -1, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)
        tm.assert_numpy_array_equal(result_codes, expected_codes)

    def test_unsortable(self):
        # GH 13714
        arr = np.array([1, 2, datetime.now(), 0, 3], dtype=object)
        msg = "'[<>]' not supported between instances of .*"
        with pytest.raises(TypeError, match=msg):
            safe_sort(arr)

    @pytest.mark.parametrize(
        "arg, codes, err, msg",
        [
            [1, None, TypeError, "Only np.ndarray, ExtensionArray, and Index"],
            [np.array([0, 1, 2]), 1, TypeError, "Only list-like objects or None"],
            [np.array([0, 1, 2, 1]), [0, 1], ValueError, "values should be unique"],
        ],
    )
    def test_exceptions(self, arg, codes, err, msg):
        with pytest.raises(err, match=msg):
            safe_sort(values=arg, codes=codes)

    @pytest.mark.parametrize(
        "arg, exp", [[[1, 3, 2], [1, 2, 3]], [[1, 3, np.nan, 2], [1, 2, 3, np.nan]]]
    )
    def test_extension_array(self, arg, exp):
        a = array(arg, dtype="Int64")
        result = safe_sort(a)
        expected = array(exp, dtype="Int64")
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("verify", [True, False])
    def test_extension_array_codes(self, verify):
        a = array([1, 3, 2], dtype="Int64")
        result, codes = safe_sort(a, [0, 1, -1, 2], use_na_sentinel=True, verify=verify)
        expected_values = array([1, 2, 3], dtype="Int64")
        expected_codes = np.array([0, 2, -1, 1], dtype=np.intp)
        tm.assert_extension_array_equal(result, expected_values)
        tm.assert_numpy_array_equal(codes, expected_codes)


def test_mixed_str_null(nulls_fixture):
    values = np.array(["b", nulls_fixture, "a", "b"], dtype=object)
    result = safe_sort(values)
    expected = np.array(["a", "b", "b", nulls_fixture], dtype=object)
    tm.assert_numpy_array_equal(result, expected)


def test_safe_sort_multiindex():
    # GH#48412
    arr1 = Series([2, 1, NA, NA], dtype="Int64")
    arr2 = [2, 1, 3, 3]
    midx = MultiIndex.from_arrays([arr1, arr2])
    result = safe_sort(midx)
    expected = MultiIndex.from_arrays(
        [Series([1, 2, NA, NA], dtype="Int64"), [1, 2, 3, 3]]
    )
    tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\period\test_arithmetic.py ===
from datetime import timedelta

import numpy as np
import pytest

from pandas._libs.tslibs.period import IncompatibleFrequency

from pandas import (
    NaT,
    Period,
    Timedelta,
    Timestamp,
    offsets,
)


class TestPeriodArithmetic:
    def test_add_overflow_raises(self):
        # GH#55503
        per = Timestamp.max.to_period("ns")

        msg = "|".join(
            [
                "Python int too large to convert to C long",
                # windows, 32bit linux builds
                "int too big to convert",
            ]
        )
        with pytest.raises(OverflowError, match=msg):
            per + 1

        msg = "value too large"
        with pytest.raises(OverflowError, match=msg):
            per + Timedelta(1)
        with pytest.raises(OverflowError, match=msg):
            per + offsets.Nano(1)

    def test_period_add_integer(self):
        per1 = Period(freq="D", year=2008, month=1, day=1)
        per2 = Period(freq="D", year=2008, month=1, day=2)
        assert per1 + 1 == per2
        assert 1 + per1 == per2

    def test_period_add_invalid(self):
        # GH#4731
        per1 = Period(freq="D", year=2008, month=1, day=1)
        per2 = Period(freq="D", year=2008, month=1, day=2)

        msg = "|".join(
            [
                r"unsupported operand type\(s\)",
                "can only concatenate str",
                "must be str, not Period",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            per1 + "str"
        with pytest.raises(TypeError, match=msg):
            "str" + per1
        with pytest.raises(TypeError, match=msg):
            per1 + per2

    def test_period_sub_period_annual(self):
        left, right = Period("2011", freq="Y"), Period("2007", freq="Y")
        result = left - right
        assert result == 4 * right.freq

        msg = r"Input has different freq=M from Period\(freq=Y-DEC\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            left - Period("2007-01", freq="M")

    def test_period_sub_period(self):
        per1 = Period("2011-01-01", freq="D")
        per2 = Period("2011-01-15", freq="D")

        off = per1.freq
        assert per1 - per2 == -14 * off
        assert per2 - per1 == 14 * off

        msg = r"Input has different freq=M from Period\(freq=D\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            per1 - Period("2011-02", freq="M")

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_sub_n_gt_1_ticks(self, tick_classes, n):
        # GH#23878
        p1 = Period("19910905", freq=tick_classes(n))
        p2 = Period("19920406", freq=tick_classes(n))

        expected = Period(str(p2), freq=p2.freq.base) - Period(
            str(p1), freq=p1.freq.base
        )

        assert (p2 - p1) == expected

    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize(
        "offset, kwd_name",
        [
            (offsets.YearEnd, "month"),
            (offsets.QuarterEnd, "startingMonth"),
            (offsets.MonthEnd, None),
            (offsets.Week, "weekday"),
        ],
    )
    def test_sub_n_gt_1_offsets(self, offset, kwd_name, n, normalize):
        # GH#23878
        kwds = {kwd_name: 3} if kwd_name is not None else {}
        p1_d = "19910905"
        p2_d = "19920406"
        p1 = Period(p1_d, freq=offset(n, normalize, **kwds))
        p2 = Period(p2_d, freq=offset(n, normalize, **kwds))

        expected = Period(p2_d, freq=p2.freq.base) - Period(p1_d, freq=p1.freq.base)

        assert (p2 - p1) == expected

    def test_period_add_offset(self):
        # freq is DateOffset
        for freq in ["Y", "2Y", "3Y"]:
            per = Period("2011", freq=freq)
            exp = Period("2013", freq=freq)
            assert per + offsets.YearEnd(2) == exp
            assert offsets.YearEnd(2) + per == exp

            for off in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(365, "D"),
                timedelta(365),
            ]:
                msg = "Input has different freq|Input cannot be converted to Period"
                with pytest.raises(IncompatibleFrequency, match=msg):
                    per + off
                with pytest.raises(IncompatibleFrequency, match=msg):
                    off + per

        for freq in ["M", "2M", "3M"]:
            per = Period("2011-03", freq=freq)
            exp = Period("2011-05", freq=freq)
            assert per + offsets.MonthEnd(2) == exp
            assert offsets.MonthEnd(2) + per == exp

            exp = Period("2012-03", freq=freq)
            assert per + offsets.MonthEnd(12) == exp
            assert offsets.MonthEnd(12) + per == exp

            msg = "|".join(
                [
                    "Input has different freq",
                    "Input cannot be converted to Period",
                ]
            )

            for off in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(365, "D"),
                timedelta(365),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    per + off
                with pytest.raises(IncompatibleFrequency, match=msg):
                    off + per

        # freq is Tick
        for freq in ["D", "2D", "3D"]:
            per = Period("2011-04-01", freq=freq)

            exp = Period("2011-04-06", freq=freq)
            assert per + offsets.Day(5) == exp
            assert offsets.Day(5) + per == exp

            exp = Period("2011-04-02", freq=freq)
            assert per + offsets.Hour(24) == exp
            assert offsets.Hour(24) + per == exp

            exp = Period("2011-04-03", freq=freq)
            assert per + np.timedelta64(2, "D") == exp
            assert np.timedelta64(2, "D") + per == exp

            exp = Period("2011-04-02", freq=freq)
            assert per + np.timedelta64(3600 * 24, "s") == exp
            assert np.timedelta64(3600 * 24, "s") + per == exp

            exp = Period("2011-03-30", freq=freq)
            assert per + timedelta(-2) == exp
            assert timedelta(-2) + per == exp

            exp = Period("2011-04-03", freq=freq)
            assert per + timedelta(hours=48) == exp
            assert timedelta(hours=48) + per == exp

            msg = "|".join(
                [
                    "Input has different freq",
                    "Input cannot be converted to Period",
                ]
            )

            for off in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(4, "h"),
                timedelta(hours=23),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    per + off
                with pytest.raises(IncompatibleFrequency, match=msg):
                    off + per

        for freq in ["h", "2h", "3h"]:
            per = Period("2011-04-01 09:00", freq=freq)

            exp = Period("2011-04-03 09:00", freq=freq)
            assert per + offsets.Day(2) == exp
            assert offsets.Day(2) + per == exp

            exp = Period("2011-04-01 12:00", freq=freq)
            assert per + offsets.Hour(3) == exp
            assert offsets.Hour(3) + per == exp

            msg = "cannot use operands with types"
            exp = Period("2011-04-01 12:00", freq=freq)
            assert per + np.timedelta64(3, "h") == exp
            assert np.timedelta64(3, "h") + per == exp

            exp = Period("2011-04-01 10:00", freq=freq)
            assert per + np.timedelta64(3600, "s") == exp
            assert np.timedelta64(3600, "s") + per == exp

            exp = Period("2011-04-01 11:00", freq=freq)
            assert per + timedelta(minutes=120) == exp
            assert timedelta(minutes=120) + per == exp

            exp = Period("2011-04-05 12:00", freq=freq)
            assert per + timedelta(days=4, minutes=180) == exp
            assert timedelta(days=4, minutes=180) + per == exp

            msg = "|".join(
                [
                    "Input has different freq",
                    "Input cannot be converted to Period",
                ]
            )

            for off in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(3200, "s"),
                timedelta(hours=23, minutes=30),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    per + off
                with pytest.raises(IncompatibleFrequency, match=msg):
                    off + per

    def test_period_sub_offset(self):
        # freq is DateOffset
        msg = "|".join(
            [
                "Input has different freq",
                "Input cannot be converted to Period",
            ]
        )

        for freq in ["Y", "2Y", "3Y"]:
            per = Period("2011", freq=freq)
            assert per - offsets.YearEnd(2) == Period("2009", freq=freq)

            for off in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(365, "D"),
                timedelta(365),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    per - off

        for freq in ["M", "2M", "3M"]:
            per = Period("2011-03", freq=freq)
            assert per - offsets.MonthEnd(2) == Period("2011-01", freq=freq)
            assert per - offsets.MonthEnd(12) == Period("2010-03", freq=freq)

            for off in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(365, "D"),
                timedelta(365),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    per - off

        # freq is Tick
        for freq in ["D", "2D", "3D"]:
            per = Period("2011-04-01", freq=freq)
            assert per - offsets.Day(5) == Period("2011-03-27", freq=freq)
            assert per - offsets.Hour(24) == Period("2011-03-31", freq=freq)
            assert per - np.timedelta64(2, "D") == Period("2011-03-30", freq=freq)
            assert per - np.timedelta64(3600 * 24, "s") == Period(
                "2011-03-31", freq=freq
            )
            assert per - timedelta(-2) == Period("2011-04-03", freq=freq)
            assert per - timedelta(hours=48) == Period("2011-03-30", freq=freq)

            for off in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(4, "h"),
                timedelta(hours=23),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    per - off

        for freq in ["h", "2h", "3h"]:
            per = Period("2011-04-01 09:00", freq=freq)
            assert per - offsets.Day(2) == Period("2011-03-30 09:00", freq=freq)
            assert per - offsets.Hour(3) == Period("2011-04-01 06:00", freq=freq)
            assert per - np.timedelta64(3, "h") == Period("2011-04-01 06:00", freq=freq)
            assert per - np.timedelta64(3600, "s") == Period(
                "2011-04-01 08:00", freq=freq
            )
            assert per - timedelta(minutes=120) == Period("2011-04-01 07:00", freq=freq)
            assert per - timedelta(days=4, minutes=180) == Period(
                "2011-03-28 06:00", freq=freq
            )

            for off in [
                offsets.YearBegin(2),
                offsets.MonthBegin(1),
                offsets.Minute(),
                np.timedelta64(3200, "s"),
                timedelta(hours=23, minutes=30),
            ]:
                with pytest.raises(IncompatibleFrequency, match=msg):
                    per - off

    @pytest.mark.parametrize("freq", ["M", "2M", "3M"])
    def test_period_addsub_nat(self, freq):
        # GH#13071
        per = Period("2011-01", freq=freq)

        # For subtraction, NaT is treated as another Period object
        assert NaT - per is NaT
        assert per - NaT is NaT

        # For addition, NaT is treated as offset-like
        assert NaT + per is NaT
        assert per + NaT is NaT

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "m"])
    def test_period_add_sub_td64_nat(self, unit):
        # GH#47196
        per = Period("2022-06-01", "D")
        nat = np.timedelta64("NaT", unit)

        assert per + nat is NaT
        assert nat + per is NaT
        assert per - nat is NaT

        with pytest.raises(TypeError, match="unsupported operand"):
            nat - per

    def test_period_ops_offset(self):
        per = Period("2011-04-01", freq="D")
        result = per + offsets.Day()
        exp = Period("2011-04-02", freq="D")
        assert result == exp

        result = per - offsets.Day(2)
        exp = Period("2011-03-30", freq="D")
        assert result == exp

        msg = r"Input cannot be converted to Period\(freq=D\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            per + offsets.Hour(2)

        with pytest.raises(IncompatibleFrequency, match=msg):
            per - offsets.Hour(2)

    def test_period_add_timestamp_raises(self):
        # GH#17983
        ts = Timestamp("2017")
        per = Period("2017", freq="M")

        msg = r"unsupported operand type\(s\) for \+: 'Timestamp' and 'Period'"
        with pytest.raises(TypeError, match=msg):
            ts + per

        msg = r"unsupported operand type\(s\) for \+: 'Period' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            per + ts


class TestPeriodComparisons:
    def test_period_comparison_same_freq(self):
        jan = Period("2000-01", "M")
        feb = Period("2000-02", "M")

        assert not jan == feb
        assert jan != feb
        assert jan < feb
        assert jan <= feb
        assert not jan > feb
        assert not jan >= feb

    def test_period_comparison_same_period_different_object(self):
        # Separate Period objects for the same period
        left = Period("2000-01", "M")
        right = Period("2000-01", "M")

        assert left == right
        assert left >= right
        assert left <= right
        assert not left < right
        assert not left > right

    def test_period_comparison_mismatched_freq(self):
        jan = Period("2000-01", "M")
        day = Period("2012-01-01", "D")

        assert not jan == day
        assert jan != day
        msg = r"Input has different freq=D from Period\(freq=M\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan < day
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan <= day
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan > day
        with pytest.raises(IncompatibleFrequency, match=msg):
            jan >= day

    def test_period_comparison_invalid_type(self):
        jan = Period("2000-01", "M")

        assert not jan == 1
        assert jan != 1

        int_or_per = "'(Period|int)'"
        msg = f"not supported between instances of {int_or_per} and {int_or_per}"
        for left, right in [(jan, 1), (1, jan)]:
            with pytest.raises(TypeError, match=msg):
                left > right
            with pytest.raises(TypeError, match=msg):
                left >= right
            with pytest.raises(TypeError, match=msg):
                left < right
            with pytest.raises(TypeError, match=msg):
                left <= right

    def test_period_comparison_nat(self):
        per = Period("2011-01-01", freq="D")

        ts = Timestamp("2011-01-01")
        # confirm Period('NaT') work identical with Timestamp('NaT')
        for left, right in [
            (NaT, per),
            (per, NaT),
            (NaT, ts),
            (ts, NaT),
        ]:
            assert not left < right
            assert not left > right
            assert not left == right
            assert left != right
            assert not left <= right
            assert not left >= right

    @pytest.mark.parametrize(
        "zerodim_arr, expected",
        ((np.array(0), False), (np.array(Period("2000-01", "M")), True)),
    )
    def test_period_comparison_numpy_zerodim_arr(self, zerodim_arr, expected):
        per = Period("2000-01", "M")

        assert (per == zerodim_arr) is expected
        assert (zerodim_arr == per) is expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_select_dtypes.py ===
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import ExtensionDtype

import pandas as pd
from pandas import (
    DataFrame,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray


class DummyDtype(ExtensionDtype):
    type = int

    def __init__(self, numeric) -> None:
        self._numeric = numeric

    @property
    def name(self):
        return "Dummy"

    @property
    def _is_numeric(self):
        return self._numeric


class DummyArray(ExtensionArray):
    def __init__(self, data, dtype) -> None:
        self.data = data
        self._dtype = dtype

    def __array__(self, dtype=None, copy=None):
        return self.data

    @property
    def dtype(self):
        return self._dtype

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item):
        pass

    def copy(self):
        return self


class TestSelectDtypes:
    def test_select_dtypes_include_using_list_like(self, using_infer_string):
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        ri = df.select_dtypes(include=[np.number])
        ei = df[["b", "c", "d", "k"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=[np.number], exclude=["timedelta"])
        ei = df[["b", "c", "d"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=[np.number, "category"], exclude=["timedelta"])
        ei = df[["b", "c", "d", "f"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=["datetime"])
        ei = df[["g"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=["datetime64"])
        ei = df[["g"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=["datetimetz"])
        ei = df[["h", "i"]]
        tm.assert_frame_equal(ri, ei)

        with pytest.raises(NotImplementedError, match=r"^$"):
            df.select_dtypes(include=["period"])

        if using_infer_string:
            ri = df.select_dtypes(include=["str"])
            ei = df[["a"]]
            tm.assert_frame_equal(ri, ei)

            ri = df.select_dtypes(include=[str])
            tm.assert_frame_equal(ri, ei)

    def test_select_dtypes_exclude_using_list_like(self):
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
            }
        )
        re = df.select_dtypes(exclude=[np.number])
        ee = df[["a", "e"]]
        tm.assert_frame_equal(re, ee)

    def test_select_dtypes_exclude_include_using_list_like(self):
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6, dtype="u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        exclude = (np.datetime64,)
        include = np.bool_, "integer"
        r = df.select_dtypes(include=include, exclude=exclude)
        e = df[["b", "c", "e"]]
        tm.assert_frame_equal(r, e)

        exclude = ("datetime",)
        include = "bool", "int64", "int32"
        r = df.select_dtypes(include=include, exclude=exclude)
        e = df[["b", "e"]]
        tm.assert_frame_equal(r, e)

    @pytest.mark.parametrize(
        "include", [(np.bool_, "int"), (np.bool_, "integer"), ("bool", int)]
    )
    def test_select_dtypes_exclude_include_int(self, include):
        # Fix select_dtypes(include='int') for Windows, FYI #36596
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6, dtype="int32"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        exclude = (np.datetime64,)
        result = df.select_dtypes(include=include, exclude=exclude)
        expected = df[["b", "c", "e"]]
        tm.assert_frame_equal(result, expected)

    def test_select_dtypes_include_using_scalars(self, using_infer_string):
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        ri = df.select_dtypes(include=np.number)
        ei = df[["b", "c", "d", "k"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include="datetime")
        ei = df[["g"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include="datetime64")
        ei = df[["g"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include="category")
        ei = df[["f"]]
        tm.assert_frame_equal(ri, ei)

        with pytest.raises(NotImplementedError, match=r"^$"):
            df.select_dtypes(include="period")

        if using_infer_string:
            ri = df.select_dtypes(include="str")
            ei = df[["a"]]
            tm.assert_frame_equal(ri, ei)

    def test_select_dtypes_exclude_using_scalars(self):
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        ri = df.select_dtypes(exclude=np.number)
        ei = df[["a", "e", "f", "g", "h", "i", "j"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(exclude="category")
        ei = df[["a", "b", "c", "d", "e", "g", "h", "i", "j", "k"]]
        tm.assert_frame_equal(ri, ei)

        with pytest.raises(NotImplementedError, match=r"^$"):
            df.select_dtypes(exclude="period")

    def test_select_dtypes_include_exclude_using_scalars(self):
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        ri = df.select_dtypes(include=np.number, exclude="floating")
        ei = df[["b", "c", "k"]]
        tm.assert_frame_equal(ri, ei)

    def test_select_dtypes_include_exclude_mixed_scalars_lists(self):
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        ri = df.select_dtypes(include=np.number, exclude=["floating", "timedelta"])
        ei = df[["b", "c"]]
        tm.assert_frame_equal(ri, ei)

        ri = df.select_dtypes(include=[np.number, "category"], exclude="floating")
        ei = df[["b", "c", "f", "k"]]
        tm.assert_frame_equal(ri, ei)

    def test_select_dtypes_duplicate_columns(self):
        # GH20839
        df = DataFrame(
            {
                "a": ["a", "b", "c"],
                "b": [1, 2, 3],
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        df.columns = ["a", "a", "b", "b", "b", "c"]

        expected = DataFrame(
            {"a": list(range(1, 4)), "b": np.arange(3, 6).astype("u1")}
        )

        result = df.select_dtypes(include=[np.number], exclude=["floating"])
        tm.assert_frame_equal(result, expected)

    def test_select_dtypes_not_an_attr_but_still_valid_dtype(self, using_infer_string):
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        df["g"] = df.f.diff()
        assert not hasattr(np, "u8")
        r = df.select_dtypes(include=["i8", "O"], exclude=["timedelta"])
        if using_infer_string:
            e = df[["b"]]
        else:
            e = df[["a", "b"]]
        tm.assert_frame_equal(r, e)

        r = df.select_dtypes(include=["i8", "O", "timedelta64[ns]"])
        if using_infer_string:
            e = df[["b", "g"]]
        else:
            e = df[["a", "b", "g"]]
        tm.assert_frame_equal(r, e)

    def test_select_dtypes_empty(self):
        df = DataFrame({"a": list("abc"), "b": list(range(1, 4))})
        msg = "at least one of include or exclude must be nonempty"
        with pytest.raises(ValueError, match=msg):
            df.select_dtypes()

    def test_select_dtypes_bad_datetime64(self):
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        with pytest.raises(ValueError, match=".+ is too specific"):
            df.select_dtypes(include=["datetime64[D]"])

        with pytest.raises(ValueError, match=".+ is too specific"):
            df.select_dtypes(exclude=["datetime64[as]"])

    def test_select_dtypes_datetime_with_tz(self):
        df2 = DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
            },
            index=range(5),
        )
        df3 = pd.concat([df2.A.to_frame(), df2.B.to_frame()], axis=1)
        result = df3.select_dtypes(include=["datetime64[ns]"])
        expected = df3.reindex(columns=[])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", [str, "str", np.bytes_, "S1", np.str_, "U1"])
    @pytest.mark.parametrize("arg", ["include", "exclude"])
    def test_select_dtypes_str_raises(self, dtype, arg, using_infer_string):
        if using_infer_string and (dtype == "str" or dtype is str):
            # this is tested below
            pytest.skip("Selecting string columns works with future strings")
        df = DataFrame(
            {
                "a": list("abc"),
                "g": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        msg = "string dtypes are not allowed"
        kwargs = {arg: [dtype]}

        with pytest.raises(TypeError, match=msg):
            df.select_dtypes(**kwargs)

    def test_select_dtypes_bad_arg_raises(self):
        df = DataFrame(
            {
                "a": list("abc"),
                "g": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )

        msg = "data type.*not understood"
        with pytest.raises(TypeError, match=msg):
            df.select_dtypes(["blargy, blarg, blarg"])

    def test_select_dtypes_typecodes(self):
        # GH 11990
        df = DataFrame(np.random.default_rng(2).random((5, 3)))
        FLOAT_TYPES = list(np.typecodes["AllFloat"])
        tm.assert_frame_equal(df.select_dtypes(FLOAT_TYPES), df)

    @pytest.mark.parametrize(
        "arr,expected",
        (
            (np.array([1, 2], dtype=np.int32), True),
            (pd.array([1, 2], dtype="Int32"), True),
            (DummyArray([1, 2], dtype=DummyDtype(numeric=True)), True),
            (DummyArray([1, 2], dtype=DummyDtype(numeric=False)), False),
        ),
    )
    def test_select_dtypes_numeric(self, arr, expected):
        # GH 35340

        df = DataFrame(arr)
        is_selected = df.select_dtypes(np.number).shape == df.shape
        assert is_selected == expected

    def test_select_dtypes_numeric_nullable_string(self, nullable_string_dtype):
        arr = pd.array(["a", "b"], dtype=nullable_string_dtype)
        df = DataFrame(arr)
        is_selected = df.select_dtypes(np.number).shape == df.shape
        assert not is_selected

    @pytest.mark.parametrize(
        "expected, float_dtypes",
        [
            [
                DataFrame(
                    {"A": range(3), "B": range(5, 8), "C": range(10, 7, -1)}
                ).astype(dtype={"A": float, "B": np.float64, "C": np.float32}),
                float,
            ],
            [
                DataFrame(
                    {"A": range(3), "B": range(5, 8), "C": range(10, 7, -1)}
                ).astype(dtype={"A": float, "B": np.float64, "C": np.float32}),
                "float",
            ],
            [DataFrame({"C": range(10, 7, -1)}, dtype=np.float32), np.float32],
            [
                DataFrame({"A": range(3), "B": range(5, 8)}).astype(
                    dtype={"A": float, "B": np.float64}
                ),
                np.float64,
            ],
        ],
    )
    def test_select_dtypes_float_dtype(self, expected, float_dtypes):
        # GH#42452
        dtype_dict = {"A": float, "B": np.float64, "C": np.float32}
        df = DataFrame(
            {"A": range(3), "B": range(5, 8), "C": range(10, 7, -1)},
        )
        df = df.astype(dtype_dict)
        result = df.select_dtypes(include=float_dtypes)
        tm.assert_frame_equal(result, expected)

    def test_np_bool_ea_boolean_include_number(self):
        # GH 46870
        df = DataFrame(
            {
                "a": [1, 2, 3],
                "b": pd.Series([True, False, True], dtype="boolean"),
                "c": np.array([True, False, True]),
                "d": pd.Categorical([True, False, True]),
                "e": pd.arrays.SparseArray([True, False, True]),
            }
        )
        result = df.select_dtypes(include="number")
        expected = DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_select_dtypes_no_view(self):
        # https://github.com/pandas-dev/pandas/issues/48090
        # result of this method is not a view on the original dataframe
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_orig = df.copy()
        result = df.select_dtypes(include=["number"])
        result.iloc[0, 0] = 0
        tm.assert_frame_equal(df, df_orig)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\xml\test_xml_dtypes.py ===
from __future__ import annotations

from io import StringIO

import pytest

from pandas.errors import ParserWarning
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    to_datetime,
)
import pandas._testing as tm

from pandas.io.xml import read_xml


@pytest.fixture(params=[pytest.param("lxml", marks=td.skip_if_no("lxml")), "etree"])
def parser(request):
    return request.param


@pytest.fixture(
    params=[None, {"book": ["category", "title", "author", "year", "price"]}]
)
def iterparse(request):
    return request.param


def read_xml_iterparse(data, **kwargs):
    with tm.ensure_clean() as path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
        return read_xml(path, **kwargs)


xml_types = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <shape>square</shape>
    <degrees>00360</degrees>
    <sides>4.0</sides>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>00360</degrees>
    <sides/>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>00180</degrees>
    <sides>3.0</sides>
  </row>
</data>"""

xml_dates = """<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <shape>square</shape>
    <degrees>00360</degrees>
    <sides>4.0</sides>
    <date>2020-01-01</date>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>00360</degrees>
    <sides/>
    <date>2021-01-01</date>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>00180</degrees>
    <sides>3.0</sides>
    <date>2022-01-01</date>
  </row>
</data>"""


# DTYPE


def test_dtype_single_str(parser):
    df_result = read_xml(StringIO(xml_types), dtype={"degrees": "str"}, parser=parser)
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        dtype={"degrees": "str"},
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_dtypes_all_str(parser):
    df_result = read_xml(StringIO(xml_dates), dtype="string", parser=parser)
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        dtype="string",
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": ["4.0", None, "3.0"],
            "date": ["2020-01-01", "2021-01-01", "2022-01-01"],
        },
        dtype="string",
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_dtypes_with_names(parser):
    df_result = read_xml(
        StringIO(xml_dates),
        names=["Col1", "Col2", "Col3", "Col4"],
        dtype={"Col2": "string", "Col3": "Int64", "Col4": "datetime64[ns]"},
        parser=parser,
    )
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        names=["Col1", "Col2", "Col3", "Col4"],
        dtype={"Col2": "string", "Col3": "Int64", "Col4": "datetime64[ns]"},
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )

    df_expected = DataFrame(
        {
            "Col1": ["square", "circle", "triangle"],
            "Col2": Series(["00360", "00360", "00180"]).astype("string"),
            "Col3": Series([4.0, float("nan"), 3.0]).astype("Int64"),
            "Col4": DatetimeIndex(
                ["2020-01-01", "2021-01-01", "2022-01-01"], dtype="M8[ns]"
            ),
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_dtype_nullable_int(parser):
    df_result = read_xml(StringIO(xml_types), dtype={"sides": "Int64"}, parser=parser)
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        dtype={"sides": "Int64"},
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": Series([4.0, float("nan"), 3.0]).astype("Int64"),
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_dtype_float(parser):
    df_result = read_xml(StringIO(xml_types), dtype={"degrees": "float"}, parser=parser)
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        dtype={"degrees": "float"},
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": Series([360, 360, 180]).astype("float"),
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_wrong_dtype(xml_books, parser, iterparse):
    with pytest.raises(
        ValueError, match=('Unable to parse string "Everyday Italian" at position 0')
    ):
        read_xml(
            xml_books, dtype={"title": "Int64"}, parser=parser, iterparse=iterparse
        )


def test_both_dtype_converters(parser):
    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    with tm.assert_produces_warning(ParserWarning, match="Both a converter and dtype"):
        df_result = read_xml(
            StringIO(xml_types),
            dtype={"degrees": "str"},
            converters={"degrees": str},
            parser=parser,
        )
        df_iter = read_xml_iterparse(
            xml_types,
            dtype={"degrees": "str"},
            converters={"degrees": str},
            parser=parser,
            iterparse={"row": ["shape", "degrees", "sides"]},
        )

        tm.assert_frame_equal(df_result, df_expected)
        tm.assert_frame_equal(df_iter, df_expected)


# CONVERTERS


def test_converters_str(parser):
    df_result = read_xml(
        StringIO(xml_types), converters={"degrees": str}, parser=parser
    )
    df_iter = read_xml_iterparse(
        xml_types,
        parser=parser,
        converters={"degrees": str},
        iterparse={"row": ["shape", "degrees", "sides"]},
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": ["00360", "00360", "00180"],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_converters_date(parser):
    convert_to_datetime = lambda x: to_datetime(x)
    df_result = read_xml(
        StringIO(xml_dates), converters={"date": convert_to_datetime}, parser=parser
    )
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        converters={"date": convert_to_datetime},
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_wrong_converters_type(xml_books, parser, iterparse):
    with pytest.raises(TypeError, match=("Type converters must be a dict or subclass")):
        read_xml(
            xml_books, converters={"year", str}, parser=parser, iterparse=iterparse
        )


def test_callable_func_converters(xml_books, parser, iterparse):
    with pytest.raises(TypeError, match=("'float' object is not callable")):
        read_xml(
            xml_books, converters={"year": float()}, parser=parser, iterparse=iterparse
        )


def test_callable_str_converters(xml_books, parser, iterparse):
    with pytest.raises(TypeError, match=("'str' object is not callable")):
        read_xml(
            xml_books, converters={"year": "float"}, parser=parser, iterparse=iterparse
        )


# PARSE DATES


def test_parse_dates_column_name(parser):
    df_result = read_xml(StringIO(xml_dates), parse_dates=["date"], parser=parser)
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        parse_dates=["date"],
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_parse_dates_column_index(parser):
    df_result = read_xml(StringIO(xml_dates), parse_dates=[3], parser=parser)
    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        parse_dates=[3],
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_parse_dates_true(parser):
    df_result = read_xml(StringIO(xml_dates), parse_dates=True, parser=parser)

    df_iter = read_xml_iterparse(
        xml_dates,
        parser=parser,
        parse_dates=True,
        iterparse={"row": ["shape", "degrees", "sides", "date"]},
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": ["2020-01-01", "2021-01-01", "2022-01-01"],
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_parse_dates_dictionary(parser):
    xml = """<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4.0</sides>
    <year>2020</year>
    <month>12</month>
    <day>31</day>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
    <year>2021</year>
    <month>12</month>
    <day>31</day>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3.0</sides>
    <year>2022</year>
    <month>12</month>
    <day>31</day>
  </row>
</data>"""

    df_result = read_xml(
        StringIO(xml), parse_dates={"date_end": ["year", "month", "day"]}, parser=parser
    )
    df_iter = read_xml_iterparse(
        xml,
        parser=parser,
        parse_dates={"date_end": ["year", "month", "day"]},
        iterparse={"row": ["shape", "degrees", "sides", "year", "month", "day"]},
    )

    df_expected = DataFrame(
        {
            "date_end": to_datetime(["2020-12-31", "2021-12-31", "2022-12-31"]),
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)


def test_day_first_parse_dates(parser):
    xml = """\
<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <shape>square</shape>
    <degrees>00360</degrees>
    <sides>4.0</sides>
    <date>31/12/2020</date>
   </row>
  <row>
    <shape>circle</shape>
    <degrees>00360</degrees>
    <sides/>
    <date>31/12/2021</date>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>00180</degrees>
    <sides>3.0</sides>
    <date>31/12/2022</date>
  </row>
</data>"""

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
            "date": to_datetime(["2020-12-31", "2021-12-31", "2022-12-31"]),
        }
    )

    with tm.assert_produces_warning(
        UserWarning, match="Parsing dates in %d/%m/%Y format"
    ):
        df_result = read_xml(StringIO(xml), parse_dates=["date"], parser=parser)
        df_iter = read_xml_iterparse(
            xml,
            parse_dates=["date"],
            parser=parser,
            iterparse={"row": ["shape", "degrees", "sides", "date"]},
        )

        tm.assert_frame_equal(df_result, df_expected)
        tm.assert_frame_equal(df_iter, df_expected)


def test_wrong_parse_dates_type(xml_books, parser, iterparse):
    with pytest.raises(
        TypeError, match=("Only booleans, lists, and dictionaries are accepted")
    ):
        read_xml(xml_books, parse_dates={"date"}, parser=parser, iterparse=iterparse)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_polyoptions.py ===
"""Tests for options manager for :class:`Poly` and public API functions. """

from sympy.polys.polyoptions import (
    Options, Expand, Gens, Wrt, Sort, Order, Field, Greedy, Domain,
    Split, Gaussian, Extension, Modulus, Symmetric, Strict, Auto,
    Frac, Formal, Polys, Include, All, Gen, Symbols, Method)

from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX

from sympy.polys.polyerrors import OptionError, GeneratorsError

from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z


def test_Options_clone():
    opt = Options((x, y, z), {'domain': 'ZZ'})

    assert opt.gens == (x, y, z)
    assert opt.domain == ZZ
    assert ('order' in opt) is False

    new_opt = opt.clone({'gens': (x, y), 'order': 'lex'})

    assert opt.gens == (x, y, z)
    assert opt.domain == ZZ
    assert ('order' in opt) is False

    assert new_opt.gens == (x, y)
    assert new_opt.domain == ZZ
    assert ('order' in new_opt) is True


def test_Expand_preprocess():
    assert Expand.preprocess(False) is False
    assert Expand.preprocess(True) is True

    assert Expand.preprocess(0) is False
    assert Expand.preprocess(1) is True

    raises(OptionError, lambda: Expand.preprocess(x))


def test_Expand_postprocess():
    opt = {'expand': True}
    Expand.postprocess(opt)

    assert opt == {'expand': True}


def test_Gens_preprocess():
    assert Gens.preprocess((None,)) == ()
    assert Gens.preprocess((x, y, z)) == (x, y, z)
    assert Gens.preprocess(((x, y, z),)) == (x, y, z)

    a = Symbol('a', commutative=False)

    raises(GeneratorsError, lambda: Gens.preprocess((x, x, y)))
    raises(GeneratorsError, lambda: Gens.preprocess((x, y, a)))


def test_Gens_postprocess():
    opt = {'gens': (x, y)}
    Gens.postprocess(opt)

    assert opt == {'gens': (x, y)}


def test_Wrt_preprocess():
    assert Wrt.preprocess(x) == ['x']
    assert Wrt.preprocess('') == []
    assert Wrt.preprocess(' ') == []
    assert Wrt.preprocess('x,y') == ['x', 'y']
    assert Wrt.preprocess('x y') == ['x', 'y']
    assert Wrt.preprocess('x, y') == ['x', 'y']
    assert Wrt.preprocess('x , y') == ['x', 'y']
    assert Wrt.preprocess(' x, y') == ['x', 'y']
    assert Wrt.preprocess(' x,  y') == ['x', 'y']
    assert Wrt.preprocess([x, y]) == ['x', 'y']

    raises(OptionError, lambda: Wrt.preprocess(','))
    raises(OptionError, lambda: Wrt.preprocess(0))


def test_Wrt_postprocess():
    opt = {'wrt': ['x']}
    Wrt.postprocess(opt)

    assert opt == {'wrt': ['x']}


def test_Sort_preprocess():
    assert Sort.preprocess([x, y, z]) == ['x', 'y', 'z']
    assert Sort.preprocess((x, y, z)) == ['x', 'y', 'z']

    assert Sort.preprocess('x > y > z') == ['x', 'y', 'z']
    assert Sort.preprocess('x>y>z') == ['x', 'y', 'z']

    raises(OptionError, lambda: Sort.preprocess(0))
    raises(OptionError, lambda: Sort.preprocess({x, y, z}))


def test_Sort_postprocess():
    opt = {'sort': 'x > y'}
    Sort.postprocess(opt)

    assert opt == {'sort': 'x > y'}


def test_Order_preprocess():
    assert Order.preprocess('lex') == lex


def test_Order_postprocess():
    opt = {'order': True}
    Order.postprocess(opt)

    assert opt == {'order': True}


def test_Field_preprocess():
    assert Field.preprocess(False) is False
    assert Field.preprocess(True) is True

    assert Field.preprocess(0) is False
    assert Field.preprocess(1) is True

    raises(OptionError, lambda: Field.preprocess(x))


def test_Field_postprocess():
    opt = {'field': True}
    Field.postprocess(opt)

    assert opt == {'field': True}


def test_Greedy_preprocess():
    assert Greedy.preprocess(False) is False
    assert Greedy.preprocess(True) is True

    assert Greedy.preprocess(0) is False
    assert Greedy.preprocess(1) is True

    raises(OptionError, lambda: Greedy.preprocess(x))


def test_Greedy_postprocess():
    opt = {'greedy': True}
    Greedy.postprocess(opt)

    assert opt == {'greedy': True}


def test_Domain_preprocess():
    assert Domain.preprocess(ZZ) == ZZ
    assert Domain.preprocess(QQ) == QQ
    assert Domain.preprocess(EX) == EX
    assert Domain.preprocess(FF(2)) == FF(2)
    assert Domain.preprocess(ZZ[x, y]) == ZZ[x, y]

    assert Domain.preprocess('Z') == ZZ
    assert Domain.preprocess('Q') == QQ

    assert Domain.preprocess('ZZ') == ZZ
    assert Domain.preprocess('QQ') == QQ

    assert Domain.preprocess('EX') == EX

    assert Domain.preprocess('FF(23)') == FF(23)
    assert Domain.preprocess('GF(23)') == GF(23)

    raises(OptionError, lambda: Domain.preprocess('Z[]'))

    assert Domain.preprocess('Z[x]') == ZZ[x]
    assert Domain.preprocess('Q[x]') == QQ[x]
    assert Domain.preprocess('R[x]') == RR[x]
    assert Domain.preprocess('C[x]') == CC[x]

    assert Domain.preprocess('ZZ[x]') == ZZ[x]
    assert Domain.preprocess('QQ[x]') == QQ[x]
    assert Domain.preprocess('RR[x]') == RR[x]
    assert Domain.preprocess('CC[x]') == CC[x]

    assert Domain.preprocess('Z[x,y]') == ZZ[x, y]
    assert Domain.preprocess('Q[x,y]') == QQ[x, y]
    assert Domain.preprocess('R[x,y]') == RR[x, y]
    assert Domain.preprocess('C[x,y]') == CC[x, y]

    assert Domain.preprocess('ZZ[x,y]') == ZZ[x, y]
    assert Domain.preprocess('QQ[x,y]') == QQ[x, y]
    assert Domain.preprocess('RR[x,y]') == RR[x, y]
    assert Domain.preprocess('CC[x,y]') == CC[x, y]

    raises(OptionError, lambda: Domain.preprocess('Z()'))

    assert Domain.preprocess('Z(x)') == ZZ.frac_field(x)
    assert Domain.preprocess('Q(x)') == QQ.frac_field(x)

    assert Domain.preprocess('ZZ(x)') == ZZ.frac_field(x)
    assert Domain.preprocess('QQ(x)') == QQ.frac_field(x)

    assert Domain.preprocess('Z(x,y)') == ZZ.frac_field(x, y)
    assert Domain.preprocess('Q(x,y)') == QQ.frac_field(x, y)

    assert Domain.preprocess('ZZ(x,y)') == ZZ.frac_field(x, y)
    assert Domain.preprocess('QQ(x,y)') == QQ.frac_field(x, y)

    assert Domain.preprocess('Q<I>') == QQ.algebraic_field(I)
    assert Domain.preprocess('QQ<I>') == QQ.algebraic_field(I)

    assert Domain.preprocess('Q<sqrt(2), I>') == QQ.algebraic_field(sqrt(2), I)
    assert Domain.preprocess(
        'QQ<sqrt(2), I>') == QQ.algebraic_field(sqrt(2), I)

    raises(OptionError, lambda: Domain.preprocess('abc'))


def test_Domain_postprocess():
    raises(GeneratorsError, lambda: Domain.postprocess({'gens': (x, y),
           'domain': ZZ[y, z]}))

    raises(GeneratorsError, lambda: Domain.postprocess({'gens': (),
           'domain': EX}))
    raises(GeneratorsError, lambda: Domain.postprocess({'domain': EX}))


def test_Split_preprocess():
    assert Split.preprocess(False) is False
    assert Split.preprocess(True) is True

    assert Split.preprocess(0) is False
    assert Split.preprocess(1) is True

    raises(OptionError, lambda: Split.preprocess(x))


def test_Split_postprocess():
    raises(NotImplementedError, lambda: Split.postprocess({'split': True}))


def test_Gaussian_preprocess():
    assert Gaussian.preprocess(False) is False
    assert Gaussian.preprocess(True) is True

    assert Gaussian.preprocess(0) is False
    assert Gaussian.preprocess(1) is True

    raises(OptionError, lambda: Gaussian.preprocess(x))


def test_Gaussian_postprocess():
    opt = {'gaussian': True}
    Gaussian.postprocess(opt)

    assert opt == {
        'gaussian': True,
        'domain': QQ_I,
    }


def test_Extension_preprocess():
    assert Extension.preprocess(True) is True
    assert Extension.preprocess(1) is True

    assert Extension.preprocess([]) is None

    assert Extension.preprocess(sqrt(2)) == {sqrt(2)}
    assert Extension.preprocess([sqrt(2)]) == {sqrt(2)}

    assert Extension.preprocess([sqrt(2), I]) == {sqrt(2), I}

    raises(OptionError, lambda: Extension.preprocess(False))
    raises(OptionError, lambda: Extension.preprocess(0))


def test_Extension_postprocess():
    opt = {'extension': {sqrt(2)}}
    Extension.postprocess(opt)

    assert opt == {
        'extension': {sqrt(2)},
        'domain': QQ.algebraic_field(sqrt(2)),
    }

    opt = {'extension': True}
    Extension.postprocess(opt)

    assert opt == {'extension': True}


def test_Modulus_preprocess():
    assert Modulus.preprocess(23) == 23
    assert Modulus.preprocess(Integer(23)) == 23

    raises(OptionError, lambda: Modulus.preprocess(0))
    raises(OptionError, lambda: Modulus.preprocess(x))


def test_Modulus_postprocess():
    opt = {'modulus': 5}
    Modulus.postprocess(opt)

    assert opt == {
        'modulus': 5,
        'domain': FF(5),
    }

    opt = {'modulus': 5, 'symmetric': False}
    Modulus.postprocess(opt)

    assert opt == {
        'modulus': 5,
        'domain': FF(5, False),
        'symmetric': False,
    }


def test_Symmetric_preprocess():
    assert Symmetric.preprocess(False) is False
    assert Symmetric.preprocess(True) is True

    assert Symmetric.preprocess(0) is False
    assert Symmetric.preprocess(1) is True

    raises(OptionError, lambda: Symmetric.preprocess(x))


def test_Symmetric_postprocess():
    opt = {'symmetric': True}
    Symmetric.postprocess(opt)

    assert opt == {'symmetric': True}


def test_Strict_preprocess():
    assert Strict.preprocess(False) is False
    assert Strict.preprocess(True) is True

    assert Strict.preprocess(0) is False
    assert Strict.preprocess(1) is True

    raises(OptionError, lambda: Strict.preprocess(x))


def test_Strict_postprocess():
    opt = {'strict': True}
    Strict.postprocess(opt)

    assert opt == {'strict': True}


def test_Auto_preprocess():
    assert Auto.preprocess(False) is False
    assert Auto.preprocess(True) is True

    assert Auto.preprocess(0) is False
    assert Auto.preprocess(1) is True

    raises(OptionError, lambda: Auto.preprocess(x))


def test_Auto_postprocess():
    opt = {'auto': True}
    Auto.postprocess(opt)

    assert opt == {'auto': True}


def test_Frac_preprocess():
    assert Frac.preprocess(False) is False
    assert Frac.preprocess(True) is True

    assert Frac.preprocess(0) is False
    assert Frac.preprocess(1) is True

    raises(OptionError, lambda: Frac.preprocess(x))


def test_Frac_postprocess():
    opt = {'frac': True}
    Frac.postprocess(opt)

    assert opt == {'frac': True}


def test_Formal_preprocess():
    assert Formal.preprocess(False) is False
    assert Formal.preprocess(True) is True

    assert Formal.preprocess(0) is False
    assert Formal.preprocess(1) is True

    raises(OptionError, lambda: Formal.preprocess(x))


def test_Formal_postprocess():
    opt = {'formal': True}
    Formal.postprocess(opt)

    assert opt == {'formal': True}


def test_Polys_preprocess():
    assert Polys.preprocess(False) is False
    assert Polys.preprocess(True) is True

    assert Polys.preprocess(0) is False
    assert Polys.preprocess(1) is True

    raises(OptionError, lambda: Polys.preprocess(x))


def test_Polys_postprocess():
    opt = {'polys': True}
    Polys.postprocess(opt)

    assert opt == {'polys': True}


def test_Include_preprocess():
    assert Include.preprocess(False) is False
    assert Include.preprocess(True) is True

    assert Include.preprocess(0) is False
    assert Include.preprocess(1) is True

    raises(OptionError, lambda: Include.preprocess(x))


def test_Include_postprocess():
    opt = {'include': True}
    Include.postprocess(opt)

    assert opt == {'include': True}


def test_All_preprocess():
    assert All.preprocess(False) is False
    assert All.preprocess(True) is True

    assert All.preprocess(0) is False
    assert All.preprocess(1) is True

    raises(OptionError, lambda: All.preprocess(x))


def test_All_postprocess():
    opt = {'all': True}
    All.postprocess(opt)

    assert opt == {'all': True}


def test_Gen_postprocess():
    opt = {'gen': x}
    Gen.postprocess(opt)

    assert opt == {'gen': x}


def test_Symbols_preprocess():
    raises(OptionError, lambda: Symbols.preprocess(x))


def test_Symbols_postprocess():
    opt = {'symbols': [x, y, z]}
    Symbols.postprocess(opt)

    assert opt == {'symbols': [x, y, z]}


def test_Method_preprocess():
    raises(OptionError, lambda: Method.preprocess(10))


def test_Method_postprocess():
    opt = {'method': 'f5b'}
    Method.postprocess(opt)

    assert opt == {'method': 'f5b'}