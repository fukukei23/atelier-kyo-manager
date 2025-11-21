
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\tests\test_complexes.py ===
from sympy.core.function import (Derivative, Function, Lambda, expand, PoleError)
from sympy.core.numbers import (E, I, Rational, comp, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, sign, transpose)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, atan, atan2, cos, sin)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.funcmatrix import FunctionMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import (ImmutableMatrix, ImmutableSparseMatrix)
from sympy.matrices import SparseMatrix
from sympy.sets.sets import Interval
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.series.order import Order
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow


def N_equals(a, b):
    """Check whether two complex numbers are numerically close"""
    return comp(a.n(), b.n(), 1.e-6)


def test_re():
    x, y = symbols('x,y')
    a, b = symbols('a,b', real=True)

    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)

    assert re(nan) is nan

    assert re(oo) is oo
    assert re(-oo) is -oo

    assert re(0) == 0

    assert re(1) == 1
    assert re(-1) == -1

    assert re(E) == E
    assert re(-E) == -E

    assert unchanged(re, x)
    assert re(x*I) == -im(x)
    assert re(r*I) == 0
    assert re(r) == r
    assert re(i*I) == I * i
    assert re(i) == 0

    assert re(x + y) == re(x) + re(y)
    assert re(x + r) == re(x) + r

    assert re(re(x)) == re(x)

    assert re(2 + I) == 2
    assert re(x + I) == re(x)

    assert re(x + y*I) == re(x) - im(y)
    assert re(x + r*I) == re(x)

    assert re(log(2*I)) == log(2)

    assert re((2 + I)**2).expand(complex=True) == 3

    assert re(conjugate(x)) == re(x)
    assert conjugate(re(x)) == re(x)

    assert re(x).as_real_imag() == (re(x), 0)

    assert re(i*r*x).diff(r) == re(i*x)
    assert re(i*r*x).diff(i) == I*r*im(x)

    assert re(
        sqrt(a + b*I)) == (a**2 + b**2)**Rational(1, 4)*cos(atan2(b, a)/2)
    assert re(a * (2 + b*I)) == 2*a

    assert re((1 + sqrt(a + b*I))/2) == \
        (a**2 + b**2)**Rational(1, 4)*cos(atan2(b, a)/2)/2 + S.Half

    assert re(x).rewrite(im) == x - S.ImaginaryUnit*im(x)
    assert (x + re(y)).rewrite(re, im) == x + y - S.ImaginaryUnit*im(y)

    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    x = Symbol('x')
    assert re(a).is_algebraic
    assert re(x).is_algebraic is None
    assert re(t).is_algebraic is False

    assert re(S.ComplexInfinity) is S.NaN

    n, m, l = symbols('n m l')
    A = MatrixSymbol('A',n,m)
    assert re(A) == (S.Half) * (A + conjugate(A))

    A = Matrix([[1 + 4*I,2],[0, -3*I]])
    assert re(A) == Matrix([[1, 2],[0, 0]])

    A = ImmutableMatrix([[1 + 3*I, 3-2*I],[0, 2*I]])
    assert re(A) == ImmutableMatrix([[1, 3],[0, 0]])

    X = SparseMatrix([[2*j + i*I for i in range(5)] for j in range(5)])
    assert re(X) - Matrix([[0, 0, 0, 0, 0],
                           [2, 2, 2, 2, 2],
                           [4, 4, 4, 4, 4],
                           [6, 6, 6, 6, 6],
                           [8, 8, 8, 8, 8]]) == Matrix.zeros(5)

    assert im(X) - Matrix([[0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4]]) == Matrix.zeros(5)

    X = FunctionMatrix(3, 3, Lambda((n, m), n + m*I))
    assert re(X) == Matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2]])


def test_im():
    x, y = symbols('x,y')
    a, b = symbols('a,b', real=True)

    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)

    assert im(nan) is nan

    assert im(oo*I) is oo
    assert im(-oo*I) is -oo

    assert im(0) == 0

    assert im(1) == 0
    assert im(-1) == 0

    assert im(E*I) == E
    assert im(-E*I) == -E

    assert unchanged(im, x)
    assert im(x*I) == re(x)
    assert im(r*I) == r
    assert im(r) == 0
    assert im(i*I) == 0
    assert im(i) == -I * i

    assert im(x + y) == im(x) + im(y)
    assert im(x + r) == im(x)
    assert im(x + r*I) == im(x) + r

    assert im(im(x)*I) == im(x)

    assert im(2 + I) == 1
    assert im(x + I) == im(x) + 1

    assert im(x + y*I) == im(x) + re(y)
    assert im(x + r*I) == im(x) + r

    assert im(log(2*I)) == pi/2

    assert im((2 + I)**2).expand(complex=True) == 4

    assert im(conjugate(x)) == -im(x)
    assert conjugate(im(x)) == im(x)

    assert im(x).as_real_imag() == (im(x), 0)

    assert im(i*r*x).diff(r) == im(i*x)
    assert im(i*r*x).diff(i) == -I * re(r*x)

    assert im(
        sqrt(a + b*I)) == (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)
    assert im(a * (2 + b*I)) == a*b

    assert im((1 + sqrt(a + b*I))/2) == \
        (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)/2

    assert im(x).rewrite(re) == -S.ImaginaryUnit * (x - re(x))
    assert (x + im(y)).rewrite(im, re) == x - S.ImaginaryUnit * (y - re(y))

    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    x = Symbol('x')
    assert re(a).is_algebraic
    assert re(x).is_algebraic is None
    assert re(t).is_algebraic is False

    assert im(S.ComplexInfinity) is S.NaN

    n, m, l = symbols('n m l')
    A = MatrixSymbol('A',n,m)

    assert im(A) == (S.One/(2*I)) * (A - conjugate(A))

    A = Matrix([[1 + 4*I, 2],[0, -3*I]])
    assert im(A) == Matrix([[4, 0],[0, -3]])

    A = ImmutableMatrix([[1 + 3*I, 3-2*I],[0, 2*I]])
    assert im(A) == ImmutableMatrix([[3, -2],[0, 2]])

    X = ImmutableSparseMatrix(
            [[i*I + i for i in range(5)] for i in range(5)])
    Y = SparseMatrix([list(range(5)) for i in range(5)])
    assert im(X).as_immutable() == Y

    X = FunctionMatrix(3, 3, Lambda((n, m), n + m*I))
    assert im(X) == Matrix([[0, 1, 2], [0, 1, 2], [0, 1, 2]])

def test_sign():
    assert sign(1.2) == 1
    assert sign(-1.2) == -1
    assert sign(3*I) == I
    assert sign(-3*I) == -I
    assert sign(0) == 0
    assert sign(0, evaluate=False).doit() == 0
    assert sign(oo, evaluate=False).doit() == 1
    assert sign(nan) is nan
    assert sign(2 + 2*I).doit() == sqrt(2)*(2 + 2*I)/4
    assert sign(2 + 3*I).simplify() == sign(2 + 3*I)
    assert sign(2 + 2*I).simplify() == sign(1 + I)
    assert sign(im(sqrt(1 - sqrt(3)))) == 1
    assert sign(sqrt(1 - sqrt(3))) == I

    x = Symbol('x')
    assert sign(x).is_finite is True
    assert sign(x).is_complex is True
    assert sign(x).is_imaginary is None
    assert sign(x).is_integer is None
    assert sign(x).is_real is None
    assert sign(x).is_zero is None
    assert sign(x).doit() == sign(x)
    assert sign(1.2*x) == sign(x)
    assert sign(2*x) == sign(x)
    assert sign(I*x) == I*sign(x)
    assert sign(-2*I*x) == -I*sign(x)
    assert sign(conjugate(x)) == conjugate(sign(x))

    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    m = Symbol('m', negative=True)
    assert sign(2*p*x) == sign(x)
    assert sign(n*x) == -sign(x)
    assert sign(n*m*x) == sign(x)

    x = Symbol('x', imaginary=True)
    assert sign(x).is_imaginary is True
    assert sign(x).is_integer is False
    assert sign(x).is_real is False
    assert sign(x).is_zero is False
    assert sign(x).diff(x) == 2*DiracDelta(-I*x)
    assert sign(x).doit() == x / Abs(x)
    assert conjugate(sign(x)) == -sign(x)

    x = Symbol('x', real=True)
    assert sign(x).is_imaginary is False
    assert sign(x).is_integer is True
    assert sign(x).is_real is True
    assert sign(x).is_zero is None
    assert sign(x).diff(x) == 2*DiracDelta(x)
    assert sign(x).doit() == sign(x)
    assert conjugate(sign(x)) == sign(x)

    x = Symbol('x', nonzero=True)
    assert sign(x).is_imaginary is False
    assert sign(x).is_integer is True
    assert sign(x).is_real is True
    assert sign(x).is_zero is False
    assert sign(x).doit() == x / Abs(x)
    assert sign(Abs(x)) == 1
    assert Abs(sign(x)) == 1

    x = Symbol('x', positive=True)
    assert sign(x).is_imaginary is False
    assert sign(x).is_integer is True
    assert sign(x).is_real is True
    assert sign(x).is_zero is False
    assert sign(x).doit() == x / Abs(x)
    assert sign(Abs(x)) == 1
    assert Abs(sign(x)) == 1

    x = 0
    assert sign(x).is_imaginary is False
    assert sign(x).is_integer is True
    assert sign(x).is_real is True
    assert sign(x).is_zero is True
    assert sign(x).doit() == 0
    assert sign(Abs(x)) == 0
    assert Abs(sign(x)) == 0

    nz = Symbol('nz', nonzero=True, integer=True)
    assert sign(nz).is_imaginary is False
    assert sign(nz).is_integer is True
    assert sign(nz).is_real is True
    assert sign(nz).is_zero is False
    assert sign(nz)**2 == 1
    assert (sign(nz)**3).args == (sign(nz), 3)

    assert sign(Symbol('x', nonnegative=True)).is_nonnegative
    assert sign(Symbol('x', nonnegative=True)).is_nonpositive is None
    assert sign(Symbol('x', nonpositive=True)).is_nonnegative is None
    assert sign(Symbol('x', nonpositive=True)).is_nonpositive
    assert sign(Symbol('x', real=True)).is_nonnegative is None
    assert sign(Symbol('x', real=True)).is_nonpositive is None
    assert sign(Symbol('x', real=True, zero=False)).is_nonpositive is None

    x, y = Symbol('x', real=True), Symbol('y')
    f = Function('f')
    assert sign(x).rewrite(Piecewise) == \
        Piecewise((1, x > 0), (-1, x < 0), (0, True))
    assert sign(y).rewrite(Piecewise) == sign(y)
    assert sign(x).rewrite(Heaviside) == 2*Heaviside(x, H0=S(1)/2) - 1
    assert sign(y).rewrite(Heaviside) == sign(y)
    assert sign(y).rewrite(Abs) == Piecewise((0, Eq(y, 0)), (y/Abs(y), True))
    assert sign(f(y)).rewrite(Abs) == Piecewise((0, Eq(f(y), 0)), (f(y)/Abs(f(y)), True))

    # evaluate what can be evaluated
    assert sign(exp_polar(I*pi)*pi) is S.NegativeOne

    eq = -sqrt(10 + 6*sqrt(3)) + sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3))
    # if there is a fast way to know when and when you cannot prove an
    # expression like this is zero then the equality to zero is ok
    assert sign(eq).func is sign or sign(eq) == 0
    # but sometimes it's hard to do this so it's better not to load
    # abs down with tests that will be very slow
    q = 1 + sqrt(2) - 2*sqrt(3) + 1331*sqrt(6)
    p = expand(q**3)**Rational(1, 3)
    d = p - q
    assert sign(d).func is sign or sign(d) == 0


def test_as_real_imag():
    n = pi**1000
    # the special code for working out the real
    # and complex parts of a power with Integer exponent
    # should not run if there is no imaginary part, hence
    # this should not hang
    assert n.as_real_imag() == (n, 0)

    # issue 6261
    x = Symbol('x')
    assert sqrt(x).as_real_imag() == \
        ((re(x)**2 + im(x)**2)**Rational(1, 4)*cos(atan2(im(x), re(x))/2),
     (re(x)**2 + im(x)**2)**Rational(1, 4)*sin(atan2(im(x), re(x))/2))

    # issue 3853
    a, b = symbols('a,b', real=True)
    assert ((1 + sqrt(a + b*I))/2).as_real_imag() == \
           (
               (a**2 + b**2)**Rational(
                   1, 4)*cos(atan2(b, a)/2)/2 + S.Half,
               (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)/2)

    assert sqrt(a**2).as_real_imag() == (sqrt(a**2), 0)
    i = symbols('i', imaginary=True)
    assert sqrt(i**2).as_real_imag() == (0, abs(i))

    assert ((1 + I)/(1 - I)).as_real_imag() == (0, 1)
    assert ((1 + I)**3/(1 - I)).as_real_imag() == (-2, 0)


@XFAIL
def test_sign_issue_3068():
    n = pi**1000
    i = int(n)
    x = Symbol('x')
    assert (n - i).round() == 1  # doesn't hang
    assert sign(n - i) == 1
    # perhaps it's not possible to get the sign right when
    # only 1 digit is being requested for this situation;
    # 2 digits works
    assert (n - x).n(1, subs={x: i}) > 0
    assert (n - x).n(2, subs={x: i}) > 0


def test_Abs():
    raises(TypeError, lambda: Abs(Interval(2, 3)))  # issue 8717

    x, y = symbols('x,y')
    assert sign(sign(x)) == sign(x)
    assert sign(x*y).func is sign
    assert Abs(0) == 0
    assert Abs(1) == 1
    assert Abs(-1) == 1
    assert Abs(I) == 1
    assert Abs(-I) == 1
    assert Abs(nan) is nan
    assert Abs(zoo) is oo
    assert Abs(I * pi) == pi
    assert Abs(-I * pi) == pi
    assert Abs(I * x) == Abs(x)
    assert Abs(-I * x) == Abs(x)
    assert Abs(-2*x) == 2*Abs(x)
    assert Abs(-2.0*x) == 2.0*Abs(x)
    assert Abs(2*pi*x*y) == 2*pi*Abs(x*y)
    assert Abs(conjugate(x)) == Abs(x)
    assert conjugate(Abs(x)) == Abs(x)
    assert Abs(x).expand(complex=True) == sqrt(re(x)**2 + im(x)**2)

    a = Symbol('a', positive=True)
    assert Abs(2*pi*x*a) == 2*pi*a*Abs(x)
    assert Abs(2*pi*I*x*a) == 2*pi*a*Abs(x)

    x = Symbol('x', real=True)
    n = Symbol('n', integer=True)
    assert Abs((-1)**n) == 1
    assert x**(2*n) == Abs(x)**(2*n)
    assert Abs(x).diff(x) == sign(x)
    assert abs(x) == Abs(x)  # Python built-in
    assert Abs(x)**3 == x**2*Abs(x)
    assert Abs(x)**4 == x**4
    assert (
        Abs(x)**(3*n)).args == (Abs(x), 3*n)  # leave symbolic odd unchanged
    assert (1/Abs(x)).args == (Abs(x), -1)
    assert 1/Abs(x)**3 == 1/(x**2*Abs(x))
    assert Abs(x)**-3 == Abs(x)/(x**4)
    assert Abs(x**3) == x**2*Abs(x)
    assert Abs(I**I) == exp(-pi/2)
    assert Abs((4 + 5*I)**(6 + 7*I)) == 68921*exp(-7*atan(Rational(5, 4)))
    y = Symbol('y', real=True)
    assert Abs(I**y) == 1
    y = Symbol('y')
    assert Abs(I**y) == exp(-pi*im(y)/2)

    x = Symbol('x', imaginary=True)
    assert Abs(x).diff(x) == -sign(x)

    eq = -sqrt(10 + 6*sqrt(3)) + sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3))
    # if there is a fast way to know when you can and when you cannot prove an
    # expression like this is zero then the equality to zero is ok
    assert abs(eq).func is Abs or abs(eq) == 0
    # but sometimes it's hard to do this so it's better not to load
    # abs down with tests that will be very slow
    q = 1 + sqrt(2) - 2*sqrt(3) + 1331*sqrt(6)
    p = expand(q**3)**Rational(1, 3)
    d = p - q
    assert abs(d).func is Abs or abs(d) == 0

    assert Abs(4*exp(pi*I/4)) == 4
    assert Abs(3**(2 + I)) == 9
    assert Abs((-3)**(1 - I)) == 3*exp(pi)

    assert Abs(oo) is oo
    assert Abs(-oo) is oo
    assert Abs(oo + I) is oo
    assert Abs(oo + I*oo) is oo

    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    x = Symbol('x')
    assert re(a).is_algebraic
    assert re(x).is_algebraic is None
    assert re(t).is_algebraic is False
    assert Abs(x).fdiff() == sign(x)
    raises(ArgumentIndexError, lambda: Abs(x).fdiff(2))

    # doesn't have recursion error
    arg = sqrt(acos(1 - I)*acos(1 + I))
    assert abs(arg) == arg

    # special handling to put Abs in denom
    assert abs(1/x) == 1/Abs(x)
    e = abs(2/x**2)
    assert e.is_Mul and e == 2/Abs(x**2)
    assert unchanged(Abs, y/x)
    assert unchanged(Abs, x/(x + 1))
    assert unchanged(Abs, x*y)
    p = Symbol('p', positive=True)
    assert abs(x/p) == abs(x)/p

    # coverage
    assert unchanged(Abs, Symbol('x', real=True)**y)
    # issue 19627
    f = Function('f', positive=True)
    assert sqrt(f(x)**2) == f(x)
    # issue 21625
    assert unchanged(Abs, S("im(acos(-i + acosh(-g + i)))"))


def test_Abs_rewrite():
    x = Symbol('x', real=True)
    a = Abs(x).rewrite(Heaviside).expand()
    assert a == x*Heaviside(x) - x*Heaviside(-x)
    for i in [-2, -1, 0, 1, 2]:
        assert a.subs(x, i) == abs(i)
    y = Symbol('y')
    assert Abs(y).rewrite(Heaviside) == Abs(y)

    x, y = Symbol('x', real=True), Symbol('y')
    assert Abs(x).rewrite(Piecewise) == Piecewise((x, x >= 0), (-x, True))
    assert Abs(y).rewrite(Piecewise) == Abs(y)
    assert Abs(y).rewrite(sign) == y/sign(y)

    i = Symbol('i', imaginary=True)
    assert abs(i).rewrite(Piecewise) == Piecewise((I*i, I*i >= 0), (-I*i, True))


    assert Abs(y).rewrite(conjugate) == sqrt(y*conjugate(y))
    assert Abs(i).rewrite(conjugate) == sqrt(-i**2) #  == -I*i

    y = Symbol('y', extended_real=True)
    assert  (Abs(exp(-I*x)-exp(-I*y))**2).rewrite(conjugate) == \
        -exp(I*x)*exp(-I*y) + 2 - exp(-I*x)*exp(I*y)


def test_Abs_real():
    # test some properties of abs that only apply
    # to real numbers
    x = Symbol('x', complex=True)
    assert sqrt(x**2) != Abs(x)
    assert Abs(x**2) != x**2

    x = Symbol('x', real=True)
    assert sqrt(x**2) == Abs(x)
    assert Abs(x**2) == x**2

    # if the symbol is zero, the following will still apply
    nn = Symbol('nn', nonnegative=True, real=True)
    np = Symbol('np', nonpositive=True, real=True)
    assert Abs(nn) == nn
    assert Abs(np) == -np


def test_Abs_properties():
    x = Symbol('x')
    assert Abs(x).is_real is None
    assert Abs(x).is_extended_real is True
    assert Abs(x).is_rational is None
    assert Abs(x).is_positive is None
    assert Abs(x).is_nonnegative is None
    assert Abs(x).is_extended_positive is None
    assert Abs(x).is_extended_nonnegative is True

    f = Symbol('x', finite=True)
    assert Abs(f).is_real is True
    assert Abs(f).is_extended_real is True
    assert Abs(f).is_rational is None
    assert Abs(f).is_positive is None
    assert Abs(f).is_nonnegative is True
    assert Abs(f).is_extended_positive is None
    assert Abs(f).is_extended_nonnegative is True

    z = Symbol('z', complex=True, zero=False)
    assert Abs(z).is_real is True # since complex implies finite
    assert Abs(z).is_extended_real is True
    assert Abs(z).is_rational is None
    assert Abs(z).is_positive is True
    assert Abs(z).is_extended_positive is True
    assert Abs(z).is_zero is False

    p = Symbol('p', positive=True)
    assert Abs(p).is_real is True
    assert Abs(p).is_extended_real is True
    assert Abs(p).is_rational is None
    assert Abs(p).is_positive is True
    assert Abs(p).is_zero is False

    q = Symbol('q', rational=True)
    assert Abs(q).is_real is True
    assert Abs(q).is_rational is True
    assert Abs(q).is_integer is None
    assert Abs(q).is_positive is None
    assert Abs(q).is_nonnegative is True

    i = Symbol('i', integer=True)
    assert Abs(i).is_real is True
    assert Abs(i).is_integer is True
    assert Abs(i).is_positive is None
    assert Abs(i).is_nonnegative is True

    e = Symbol('n', even=True)
    ne = Symbol('ne', real=True, even=False)
    assert Abs(e).is_even is True
    assert Abs(ne).is_even is False
    assert Abs(i).is_even is None

    o = Symbol('n', odd=True)
    no = Symbol('no', real=True, odd=False)
    assert Abs(o).is_odd is True
    assert Abs(no).is_odd is False
    assert Abs(i).is_odd is None


def test_abs():
    # this tests that abs calls Abs; don't rename to
    # test_Abs since that test is already above
    a = Symbol('a', positive=True)
    assert abs(I*(1 + a)**2) == (1 + a)**2


def test_arg():
    assert arg(0) is nan
    assert arg(1) == 0
    assert arg(-1) == pi
    assert arg(I) == pi/2
    assert arg(-I) == -pi/2
    assert arg(1 + I) == pi/4
    assert arg(-1 + I) == pi*Rational(3, 4)
    assert arg(1 - I) == -pi/4
    assert arg(exp_polar(4*pi*I)) == 4*pi
    assert arg(exp_polar(-7*pi*I)) == -7*pi
    assert arg(exp_polar(5 - 3*pi*I/4)) == pi*Rational(-3, 4)

    assert arg(exp(I*pi/7)) == pi/7     # issue 17300
    assert arg(exp(16*I)) == 16 - 6*pi
    assert arg(exp(13*I*pi/12)) == -11*pi/12
    assert arg(exp(123 - 5*I)) == -5 + 2*pi
    assert arg(exp(sin(1 + 3*I))) == -2*pi + cos(1)*sinh(3)
    r = Symbol('r', real=True)
    assert arg(exp(r - 2*I)) == -2

    f = Function('f')
    assert not arg(f(0) + I*f(1)).atoms(re)

    # check nesting
    x = Symbol('x')
    assert arg(arg(arg(x))) is not S.NaN
    assert arg(arg(arg(arg(x)))) is S.NaN
    r = Symbol('r', extended_real=True)
    assert arg(arg(r)) is not S.NaN
    assert arg(arg(arg(r))) is S.NaN

    p = Function('p', extended_positive=True)
    assert arg(p(x)) == 0
    assert arg((3 + I)*p(x)) == arg(3  + I)

    p = Symbol('p', positive=True)
    assert arg(p) == 0
    assert arg(p*I) == pi/2

    n = Symbol('n', negative=True)
    assert arg(n) == pi
    assert arg(n*I) == -pi/2

    x = Symbol('x')
    assert conjugate(arg(x)) == arg(x)

    e = p + I*p**2
    assert arg(e) == arg(1 + p*I)
    # make sure sign doesn't swap
    e = -2*p + 4*I*p**2
    assert arg(e) == arg(-1 + 2*p*I)
    # make sure sign isn't lost
    x = symbols('x', real=True)  # could be zero
    e = x + I*x
    assert arg(e) == arg(x*(1 + I))
    assert arg(e/p) == arg(x*(1 + I))
    e = p*cos(p) + I*log(p)*exp(p)
    assert arg(e).args[0] == e
    # keep it simple -- let the user do more advanced cancellation
    e = (p + 1) + I*(p**2 - 1)
    assert arg(e).args[0] == e

    f = Function('f')
    e = 2*x*(f(0) - 1) - 2*x*f(0)
    assert arg(e) == arg(-2*x)
    assert arg(f(0)).func == arg and arg(f(0)).args == (f(0),)


def test_arg_rewrite():
    assert arg(1 + I) == atan2(1, 1)

    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert arg(x + I*y).rewrite(atan2) == atan2(y, x)


def test_arg_leading_term_and_series():
    x = Symbol('x')
    assert arg(x).as_leading_term(x, cdir = 1) == 0
    assert arg(x).as_leading_term(x, cdir = -1) == pi
    raises(PoleError, lambda: arg(x + I).as_leading_term(x, cdir = 1))
    raises(PoleError, lambda: arg(2*x).as_leading_term(x, cdir = I))

    assert arg(x).nseries(x) == 0
    assert arg(x).nseries(x, n=0) == Order(1)


def test_adjoint():
    a = Symbol('a', antihermitian=True)
    b = Symbol('b', hermitian=True)
    assert adjoint(a) == -a
    assert adjoint(I*a) == I*a
    assert adjoint(b) == b
    assert adjoint(I*b) == -I*b
    assert adjoint(a*b) == -b*a
    assert adjoint(I*a*b) == I*b*a

    x, y = symbols('x y')
    assert adjoint(adjoint(x)) == x
    assert adjoint(x + y) == conjugate(x) + conjugate(y)
    assert adjoint(x - y) == conjugate(x) - conjugate(y)
    assert adjoint(x * y) == conjugate(x) * conjugate(y)
    assert adjoint(x / y) == conjugate(x) / conjugate(y)
    assert adjoint(-x) == -conjugate(x)

    x, y = symbols('x y', commutative=False)
    assert adjoint(adjoint(x)) == x
    assert adjoint(x + y) == adjoint(x) + adjoint(y)
    assert adjoint(x - y) == adjoint(x) - adjoint(y)
    assert adjoint(x * y) == adjoint(y) * adjoint(x)
    assert adjoint(x / y) == 1 / adjoint(y) * adjoint(x)
    assert adjoint(-x) == -adjoint(x)


def test_conjugate():
    a = Symbol('a', real=True)
    b = Symbol('b', imaginary=True)
    assert conjugate(a) == a
    assert conjugate(I*a) == -I*a
    assert conjugate(b) == -b
    assert conjugate(I*b) == I*b
    assert conjugate(a*b) == -a*b
    assert conjugate(I*a*b) == I*a*b

    x, y = symbols('x y')
    assert conjugate(conjugate(x)) == x
    assert conjugate(x).inverse() == conjugate
    assert conjugate(x + y) == conjugate(x) + conjugate(y)
    assert conjugate(x - y) == conjugate(x) - conjugate(y)
    assert conjugate(x * y) == conjugate(x) * conjugate(y)
    assert conjugate(x / y) == conjugate(x) / conjugate(y)
    assert conjugate(-x) == -conjugate(x)

    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    assert re(a).is_algebraic
    assert re(x).is_algebraic is None
    assert re(t).is_algebraic is False


def test_conjugate_transpose():
    x = Symbol('x', commutative=False)
    assert conjugate(transpose(x)) == adjoint(x)
    assert transpose(conjugate(x)) == adjoint(x)
    assert adjoint(transpose(x)) == conjugate(x)
    assert transpose(adjoint(x)) == conjugate(x)
    assert adjoint(conjugate(x)) == transpose(x)
    assert conjugate(adjoint(x)) == transpose(x)

    x = Symbol('x')
    assert conjugate(x) == adjoint(x)
    assert transpose(x) == x


def test_transpose():
    a = Symbol('a', complex=True)
    assert transpose(a) == a
    assert transpose(I*a) == I*a

    x, y = symbols('x y')
    assert transpose(transpose(x)) == x
    assert transpose(x + y) == x + y
    assert transpose(x - y) == x - y
    assert transpose(x * y) == x * y
    assert transpose(x / y) == x / y
    assert transpose(-x) == -x

    x, y = symbols('x y', commutative=False)
    assert transpose(transpose(x)) == x
    assert transpose(x + y) == transpose(x) + transpose(y)
    assert transpose(x - y) == transpose(x) - transpose(y)
    assert transpose(x * y) == transpose(y) * transpose(x)
    assert transpose(x / y) == 1 / transpose(y) * transpose(x)
    assert transpose(-x) == -transpose(x)


@_both_exp_pow
def test_polarify():
    from sympy.functions.elementary.complexes import (polar_lift, polarify)
    x = Symbol('x')
    z = Symbol('z', polar=True)
    f = Function('f')
    ES = {}

    assert polarify(-1) == (polar_lift(-1), ES)
    assert polarify(1 + I) == (polar_lift(1 + I), ES)

    assert polarify(exp(x), subs=False) == exp(x)
    assert polarify(1 + x, subs=False) == 1 + x
    assert polarify(f(I) + x, subs=False) == f(polar_lift(I)) + x

    assert polarify(x, lift=True) == polar_lift(x)
    assert polarify(z, lift=True) == z
    assert polarify(f(x), lift=True) == f(polar_lift(x))
    assert polarify(1 + x, lift=True) == polar_lift(1 + x)
    assert polarify(1 + f(x), lift=True) == polar_lift(1 + f(polar_lift(x)))

    newex, subs = polarify(f(x) + z)
    assert newex.subs(subs) == f(x) + z

    mu = Symbol("mu")
    sigma = Symbol("sigma", positive=True)

    # Make sure polarify(lift=True) doesn't try to lift the integration
    # variable
    assert polarify(
        Integral(sqrt(2)*x*exp(-(-mu + x)**2/(2*sigma**2))/(2*sqrt(pi)*sigma),
        (x, -oo, oo)), lift=True) == Integral(sqrt(2)*(sigma*exp_polar(0))**exp_polar(I*pi)*
        exp((sigma*exp_polar(0))**(2*exp_polar(I*pi))*exp_polar(I*pi)*polar_lift(-mu + x)**
        (2*exp_polar(0))/2)*exp_polar(0)*polar_lift(x)/(2*sqrt(pi)), (x, -oo, oo))


def test_unpolarify():
    from sympy.functions.elementary.complexes import (polar_lift, principal_branch, unpolarify)
    from sympy.core.relational import Ne
    from sympy.functions.elementary.hyperbolic import tanh
    from sympy.functions.special.error_functions import erf
    from sympy.functions.special.gamma_functions import (gamma, uppergamma)
    from sympy.abc import x
    p = exp_polar(7*I) + 1
    u = exp(7*I) + 1

    assert unpolarify(1) == 1
    assert unpolarify(p) == u
    assert unpolarify(p**2) == u**2
    assert unpolarify(p**x) == p**x
    assert unpolarify(p*x) == u*x
    assert unpolarify(p + x) == u + x
    assert unpolarify(sqrt(sin(p))) == sqrt(sin(u))

    # Test reduction to principal branch 2*pi.
    t = principal_branch(x, 2*pi)
    assert unpolarify(t) == x
    assert unpolarify(sqrt(t)) == sqrt(t)

    # Test exponents_only.
    assert unpolarify(p**p, exponents_only=True) == p**u
    assert unpolarify(uppergamma(x, p**p)) == uppergamma(x, p**u)

    # Test functions.
    assert unpolarify(sin(p)) == sin(u)
    assert unpolarify(tanh(p)) == tanh(u)
    assert unpolarify(gamma(p)) == gamma(u)
    assert unpolarify(erf(p)) == erf(u)
    assert unpolarify(uppergamma(x, p)) == uppergamma(x, p)

    assert unpolarify(uppergamma(sin(p), sin(p + exp_polar(0)))) == \
        uppergamma(sin(u), sin(u + 1))
    assert unpolarify(uppergamma(polar_lift(0), 2*exp_polar(0))) == \
        uppergamma(0, 2)

    assert unpolarify(Eq(p, 0)) == Eq(u, 0)
    assert unpolarify(Ne(p, 0)) == Ne(u, 0)
    assert unpolarify(polar_lift(x) > 0) == (x > 0)

    # Test bools
    assert unpolarify(True) is True


def test_issue_4035():
    x = Symbol('x')
    assert Abs(x).expand(trig=True) == Abs(x)
    assert sign(x).expand(trig=True) == sign(x)
    assert arg(x).expand(trig=True) == arg(x)


def test_issue_3206():
    x = Symbol('x')
    assert Abs(Abs(x)) == Abs(x)


def test_issue_4754_derivative_conjugate():
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    f = Function('f')
    assert (f(x).conjugate()).diff(x) == (f(x).diff(x)).conjugate()
    assert (f(y).conjugate()).diff(y) == -(f(y).diff(y)).conjugate()


def test_derivatives_issue_4757():
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    f = Function('f')
    assert re(f(x)).diff(x) == re(f(x).diff(x))
    assert im(f(x)).diff(x) == im(f(x).diff(x))
    assert re(f(y)).diff(y) == -I*im(f(y).diff(y))
    assert im(f(y)).diff(y) == -I*re(f(y).diff(y))
    assert Abs(f(x)).diff(x).subs(f(x), 1 + I*x).doit() == x/sqrt(1 + x**2)
    assert arg(f(x)).diff(x).subs(f(x), 1 + I*x**2).doit() == 2*x/(1 + x**4)
    assert Abs(f(y)).diff(y).subs(f(y), 1 + y).doit() == -y/sqrt(1 - y**2)
    assert arg(f(y)).diff(y).subs(f(y), I + y**2).doit() == 2*y/(1 + y**4)


def test_issue_11413():
    from sympy.simplify.simplify import simplify
    v0 = Symbol('v0')
    v1 = Symbol('v1')
    v2 = Symbol('v2')
    V = Matrix([[v0],[v1],[v2]])
    U = V.normalized()
    assert U == Matrix([
    [v0/sqrt(Abs(v0)**2 + Abs(v1)**2 + Abs(v2)**2)],
    [v1/sqrt(Abs(v0)**2 + Abs(v1)**2 + Abs(v2)**2)],
    [v2/sqrt(Abs(v0)**2 + Abs(v1)**2 + Abs(v2)**2)]])
    U.norm = sqrt(v0**2/(v0**2 + v1**2 + v2**2) + v1**2/(v0**2 + v1**2 + v2**2) + v2**2/(v0**2 + v1**2 + v2**2))
    assert simplify(U.norm) == 1


def test_periodic_argument():
    from sympy.functions.elementary.complexes import (periodic_argument, polar_lift, principal_branch, unbranched_argument)
    x = Symbol('x')
    p = Symbol('p', positive=True)

    assert unbranched_argument(2 + I) == periodic_argument(2 + I, oo)
    assert unbranched_argument(1 + x) == periodic_argument(1 + x, oo)
    assert N_equals(unbranched_argument((1 + I)**2), pi/2)
    assert N_equals(unbranched_argument((1 - I)**2), -pi/2)
    assert N_equals(periodic_argument((1 + I)**2, 3*pi), pi/2)
    assert N_equals(periodic_argument((1 - I)**2, 3*pi), -pi/2)

    assert unbranched_argument(principal_branch(x, pi)) == \
        periodic_argument(x, pi)

    assert unbranched_argument(polar_lift(2 + I)) == unbranched_argument(2 + I)
    assert periodic_argument(polar_lift(2 + I), 2*pi) == \
        periodic_argument(2 + I, 2*pi)
    assert periodic_argument(polar_lift(2 + I), 3*pi) == \
        periodic_argument(2 + I, 3*pi)
    assert periodic_argument(polar_lift(2 + I), pi) == \
        periodic_argument(polar_lift(2 + I), pi)

    assert unbranched_argument(polar_lift(1 + I)) == pi/4
    assert periodic_argument(2*p, p) == periodic_argument(p, p)
    assert periodic_argument(pi*p, p) == periodic_argument(p, p)

    assert Abs(polar_lift(1 + I)) == Abs(1 + I)


@XFAIL
def test_principal_branch_fail():
    # TODO XXX why does abs(x)._eval_evalf() not fall back to global evalf?
    from sympy.functions.elementary.complexes import principal_branch
    assert N_equals(principal_branch((1 + I)**2, pi/2), 0)


def test_principal_branch():
    from sympy.functions.elementary.complexes import (polar_lift, principal_branch)
    p = Symbol('p', positive=True)
    x = Symbol('x')
    neg = Symbol('x', negative=True)

    assert principal_branch(polar_lift(x), p) == principal_branch(x, p)
    assert principal_branch(polar_lift(2 + I), p) == principal_branch(2 + I, p)
    assert principal_branch(2*x, p) == 2*principal_branch(x, p)
    assert principal_branch(1, pi) == exp_polar(0)
    assert principal_branch(-1, 2*pi) == exp_polar(I*pi)
    assert principal_branch(-1, pi) == exp_polar(0)
    assert principal_branch(exp_polar(3*pi*I)*x, 2*pi) == \
        principal_branch(exp_polar(I*pi)*x, 2*pi)
    assert principal_branch(neg*exp_polar(pi*I), 2*pi) == neg*exp_polar(-I*pi)
    # related to issue #14692
    assert principal_branch(exp_polar(-I*pi/2)/polar_lift(neg), 2*pi) == \
        exp_polar(-I*pi/2)/neg

    assert N_equals(principal_branch((1 + I)**2, 2*pi), 2*I)
    assert N_equals(principal_branch((1 + I)**2, 3*pi), 2*I)
    assert N_equals(principal_branch((1 + I)**2, 1*pi), 2*I)

    # test argument sanitization
    assert principal_branch(x, I).func is principal_branch
    assert principal_branch(x, -4).func is principal_branch
    assert principal_branch(x, -oo).func is principal_branch
    assert principal_branch(x, zoo).func is principal_branch


@XFAIL
def test_issue_6167_6151():
    n = pi**1000
    i = int(n)
    assert sign(n - i) == 1
    assert abs(n - i) == n - i
    x = Symbol('x')
    eps = pi**-1500
    big = pi**1000
    one = cos(x)**2 + sin(x)**2
    e = big*one - big + eps
    from sympy.simplify.simplify import simplify
    assert sign(simplify(e)) == 1
    for xi in (111, 11, 1, Rational(1, 10)):
        assert sign(e.subs(x, xi)) == 1


def test_issue_14216():
    from sympy.functions.elementary.complexes import unpolarify
    A = MatrixSymbol("A", 2, 2)
    assert unpolarify(A[0, 0]) == A[0, 0]
    assert unpolarify(A[0, 0]*A[1, 0]) == A[0, 0]*A[1, 0]


def test_issue_14238():
    # doesn't cause recursion error
    r = Symbol('r', real=True)
    assert Abs(r + Piecewise((0, r > 0), (1 - r, True)))


def test_issue_22189():
    x = Symbol('x')
    for a in (sqrt(7 - 2*x) - 2, 1 - x):
        assert Abs(a) - Abs(-a) == 0, a


def test_zero_assumptions():
    nr = Symbol('nonreal', real=False, finite=True)
    ni = Symbol('nonimaginary', imaginary=False)
    # imaginary implies not zero
    nzni = Symbol('nonzerononimaginary', zero=False, imaginary=False)

    assert re(nr).is_zero is None
    assert im(nr).is_zero is False

    assert re(ni).is_zero is None
    assert im(ni).is_zero is None

    assert re(nzni).is_zero is False
    assert im(nzni).is_zero is None


@_both_exp_pow
def test_issue_15893():
    f = Function('f', real=True)
    x = Symbol('x', real=True)
    eq = Derivative(Abs(f(x)), f(x))
    assert eq.doit() == sign(f(x))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_sort_index.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    IntervalIndex,
    MultiIndex,
    RangeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestDataFrameSortIndex:
    def test_sort_index_and_reconstruction_doc_example(self):
        # doc example
        df = DataFrame(
            {"value": [1, 2, 3, 4]},
            index=MultiIndex(
                levels=[["a", "b"], ["bb", "aa"]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]
            ),
        )
        assert df.index._is_lexsorted()
        assert not df.index.is_monotonic_increasing

        # sort it
        expected = DataFrame(
            {"value": [2, 1, 4, 3]},
            index=MultiIndex(
                levels=[["a", "b"], ["aa", "bb"]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]
            ),
        )
        result = df.sort_index()
        assert result.index.is_monotonic_increasing
        tm.assert_frame_equal(result, expected)

        # reconstruct
        result = df.sort_index().copy()
        result.index = result.index._sort_levels_monotonic()
        assert result.index.is_monotonic_increasing
        tm.assert_frame_equal(result, expected)

    def test_sort_index_non_existent_label_multiindex(self):
        # GH#12261
        df = DataFrame(0, columns=[], index=MultiIndex.from_product([[], []]))
        with tm.assert_produces_warning(None):
            df.loc["b", "2"] = 1
            df.loc["a", "3"] = 1
        result = df.sort_index().index.is_monotonic_increasing
        assert result is True

    def test_sort_index_reorder_on_ops(self):
        # GH#15687
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 2)),
            index=MultiIndex.from_product(
                [["a", "b"], ["big", "small"], ["red", "blu"]],
                names=["letter", "size", "color"],
            ),
            columns=["near", "far"],
        )
        df = df.sort_index()

        def my_func(group):
            group.index = ["newz", "newa"]
            return group

        result = df.groupby(level=["letter", "size"]).apply(my_func).sort_index()
        expected = MultiIndex.from_product(
            [["a", "b"], ["big", "small"], ["newa", "newz"]],
            names=["letter", "size", None],
        )

        tm.assert_index_equal(result.index, expected)

    def test_sort_index_nan_multiindex(self):
        # GH#14784
        # incorrect sorting w.r.t. nans
        tuples = [[12, 13], [np.nan, np.nan], [np.nan, 3], [1, 2]]
        mi = MultiIndex.from_tuples(tuples)

        df = DataFrame(np.arange(16).reshape(4, 4), index=mi, columns=list("ABCD"))
        s = Series(np.arange(4), index=mi)

        df2 = DataFrame(
            {
                "date": pd.DatetimeIndex(
                    [
                        "20121002",
                        "20121007",
                        "20130130",
                        "20130202",
                        "20130305",
                        "20121002",
                        "20121207",
                        "20130130",
                        "20130202",
                        "20130305",
                        "20130202",
                        "20130305",
                    ]
                ),
                "user_id": [1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5, 5],
                "whole_cost": [
                    1790,
                    np.nan,
                    280,
                    259,
                    np.nan,
                    623,
                    90,
                    312,
                    np.nan,
                    301,
                    359,
                    801,
                ],
                "cost": [12, 15, 10, 24, 39, 1, 0, np.nan, 45, 34, 1, 12],
            }
        ).set_index(["date", "user_id"])

        # sorting frame, default nan position is last
        result = df.sort_index()
        expected = df.iloc[[3, 0, 2, 1], :]
        tm.assert_frame_equal(result, expected)

        # sorting frame, nan position last
        result = df.sort_index(na_position="last")
        expected = df.iloc[[3, 0, 2, 1], :]
        tm.assert_frame_equal(result, expected)

        # sorting frame, nan position first
        result = df.sort_index(na_position="first")
        expected = df.iloc[[1, 2, 3, 0], :]
        tm.assert_frame_equal(result, expected)

        # sorting frame with removed rows
        result = df2.dropna().sort_index()
        expected = df2.sort_index().dropna()
        tm.assert_frame_equal(result, expected)

        # sorting series, default nan position is last
        result = s.sort_index()
        expected = s.iloc[[3, 0, 2, 1]]
        tm.assert_series_equal(result, expected)

        # sorting series, nan position last
        result = s.sort_index(na_position="last")
        expected = s.iloc[[3, 0, 2, 1]]
        tm.assert_series_equal(result, expected)

        # sorting series, nan position first
        result = s.sort_index(na_position="first")
        expected = s.iloc[[1, 2, 3, 0]]
        tm.assert_series_equal(result, expected)

    def test_sort_index_nan(self):
        # GH#3917

        # Test DataFrame with nan label
        df = DataFrame(
            {"A": [1, 2, np.nan, 1, 6, 8, 4], "B": [9, np.nan, 5, 2, 5, 4, 5]},
            index=[1, 2, 3, 4, 5, 6, np.nan],
        )

        # NaN label, ascending=True, na_position='last'
        sorted_df = df.sort_index(kind="quicksort", ascending=True, na_position="last")
        expected = DataFrame(
            {"A": [1, 2, np.nan, 1, 6, 8, 4], "B": [9, np.nan, 5, 2, 5, 4, 5]},
            index=[1, 2, 3, 4, 5, 6, np.nan],
        )
        tm.assert_frame_equal(sorted_df, expected)

        # NaN label, ascending=True, na_position='first'
        sorted_df = df.sort_index(na_position="first")
        expected = DataFrame(
            {"A": [4, 1, 2, np.nan, 1, 6, 8], "B": [5, 9, np.nan, 5, 2, 5, 4]},
            index=[np.nan, 1, 2, 3, 4, 5, 6],
        )
        tm.assert_frame_equal(sorted_df, expected)

        # NaN label, ascending=False, na_position='last'
        sorted_df = df.sort_index(kind="quicksort", ascending=False)
        expected = DataFrame(
            {"A": [8, 6, 1, np.nan, 2, 1, 4], "B": [4, 5, 2, 5, np.nan, 9, 5]},
            index=[6, 5, 4, 3, 2, 1, np.nan],
        )
        tm.assert_frame_equal(sorted_df, expected)

        # NaN label, ascending=False, na_position='first'
        sorted_df = df.sort_index(
            kind="quicksort", ascending=False, na_position="first"
        )
        expected = DataFrame(
            {"A": [4, 8, 6, 1, np.nan, 2, 1], "B": [5, 4, 5, 2, 5, np.nan, 9]},
            index=[np.nan, 6, 5, 4, 3, 2, 1],
        )
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_index_multi_index(self):
        # GH#25775, testing that sorting by index works with a multi-index.
        df = DataFrame(
            {"a": [3, 1, 2], "b": [0, 0, 0], "c": [0, 1, 2], "d": list("abc")}
        )
        result = df.set_index(list("abc")).sort_index(level=list("ba"))

        expected = DataFrame(
            {"a": [1, 2, 3], "b": [0, 0, 0], "c": [1, 2, 0], "d": list("bca")}
        )
        expected = expected.set_index(list("abc"))

        tm.assert_frame_equal(result, expected)

    def test_sort_index_inplace(self):
        frame = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        # axis=0
        unordered = frame.loc[[3, 2, 4, 1]]
        a_values = unordered["A"]
        df = unordered.copy()
        return_value = df.sort_index(inplace=True)
        assert return_value is None
        expected = frame
        tm.assert_frame_equal(df, expected)
        # GH 44153 related
        # Used to be a_id != id(df["A"]), but flaky in the CI
        assert a_values is not df["A"]

        df = unordered.copy()
        return_value = df.sort_index(ascending=False, inplace=True)
        assert return_value is None
        expected = frame[::-1]
        tm.assert_frame_equal(df, expected)

        # axis=1
        unordered = frame.loc[:, ["D", "B", "C", "A"]]
        df = unordered.copy()
        return_value = df.sort_index(axis=1, inplace=True)
        assert return_value is None
        expected = frame
        tm.assert_frame_equal(df, expected)

        df = unordered.copy()
        return_value = df.sort_index(axis=1, ascending=False, inplace=True)
        assert return_value is None
        expected = frame.iloc[:, ::-1]
        tm.assert_frame_equal(df, expected)

    def test_sort_index_different_sortorder(self):
        A = np.arange(20).repeat(5)
        B = np.tile(np.arange(5), 20)

        indexer = np.random.default_rng(2).permutation(100)
        A = A.take(indexer)
        B = B.take(indexer)

        df = DataFrame(
            {"A": A, "B": B, "C": np.random.default_rng(2).standard_normal(100)}
        )

        ex_indexer = np.lexsort((df.B.max() - df.B, df.A))
        expected = df.take(ex_indexer)

        # test with multiindex, too
        idf = df.set_index(["A", "B"])

        result = idf.sort_index(ascending=[1, 0])
        expected = idf.take(ex_indexer)
        tm.assert_frame_equal(result, expected)

        # also, Series!
        result = idf["C"].sort_index(ascending=[1, 0])
        tm.assert_series_equal(result, expected["C"])

    def test_sort_index_level(self):
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list("ABC"))
        df = DataFrame([[1, 2], [3, 4]], mi)

        result = df.sort_index(level="A", sort_remaining=False)
        expected = df
        tm.assert_frame_equal(result, expected)

        result = df.sort_index(level=["A", "B"], sort_remaining=False)
        expected = df
        tm.assert_frame_equal(result, expected)

        # Error thrown by sort_index when
        # first index is sorted last (GH#26053)
        result = df.sort_index(level=["C", "B", "A"])
        expected = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_index(level=["B", "C", "A"])
        expected = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_index(level=["C", "A"])
        expected = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)

    def test_sort_index_categorical_index(self):
        df = DataFrame(
            {
                "A": np.arange(6, dtype="int64"),
                "B": Series(list("aabbca")).astype(CategoricalDtype(list("cab"))),
            }
        ).set_index("B")

        result = df.sort_index()
        expected = df.iloc[[4, 0, 1, 5, 2, 3]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_index(ascending=False)
        expected = df.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_frame_equal(result, expected)

    def test_sort_index(self):
        # GH#13496

        frame = DataFrame(
            np.arange(16).reshape(4, 4),
            index=[1, 2, 3, 4],
            columns=["A", "B", "C", "D"],
        )

        # axis=0 : sort rows by index labels
        unordered = frame.loc[[3, 2, 4, 1]]
        result = unordered.sort_index(axis=0)
        expected = frame
        tm.assert_frame_equal(result, expected)

        result = unordered.sort_index(ascending=False)
        expected = frame[::-1]
        tm.assert_frame_equal(result, expected)

        # axis=1 : sort columns by column names
        unordered = frame.iloc[:, [2, 1, 3, 0]]
        result = unordered.sort_index(axis=1)
        tm.assert_frame_equal(result, frame)

        result = unordered.sort_index(axis=1, ascending=False)
        expected = frame.iloc[:, ::-1]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("level", ["A", 0])  # GH#21052
    def test_sort_index_multiindex(self, level):
        # GH#13496

        # sort rows by specified level of multi-index
        mi = MultiIndex.from_tuples(
            [[2, 1, 3], [2, 1, 2], [1, 1, 1]], names=list("ABC")
        )
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mi)

        expected_mi = MultiIndex.from_tuples(
            [[1, 1, 1], [2, 1, 2], [2, 1, 3]], names=list("ABC")
        )
        expected = DataFrame([[5, 6], [3, 4], [1, 2]], index=expected_mi)
        result = df.sort_index(level=level)
        tm.assert_frame_equal(result, expected)

        # sort_remaining=False
        expected_mi = MultiIndex.from_tuples(
            [[1, 1, 1], [2, 1, 3], [2, 1, 2]], names=list("ABC")
        )
        expected = DataFrame([[5, 6], [1, 2], [3, 4]], index=expected_mi)
        result = df.sort_index(level=level, sort_remaining=False)
        tm.assert_frame_equal(result, expected)

    def test_sort_index_intervalindex(self):
        # this is a de-facto sort via unstack
        # confirming that we sort in the order of the bins
        y = Series(np.random.default_rng(2).standard_normal(100))
        x1 = Series(np.sign(np.random.default_rng(2).standard_normal(100)))
        x2 = pd.cut(
            Series(np.random.default_rng(2).standard_normal(100)),
            bins=[-3, -0.5, 0, 0.5, 3],
        )
        model = pd.concat([y, x1, x2], axis=1, keys=["Y", "X1", "X2"])

        result = model.groupby(["X1", "X2"], observed=True).mean().unstack()
        expected = IntervalIndex.from_tuples(
            [(-3.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 3.0)], closed="right"
        )
        result = result.columns.levels[1].categories
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_dict, sorted_dict, ascending, ignore_index, output_index",
        [
            ({"A": [1, 2, 3]}, {"A": [2, 3, 1]}, False, True, [0, 1, 2]),
            ({"A": [1, 2, 3]}, {"A": [1, 3, 2]}, True, True, [0, 1, 2]),
            ({"A": [1, 2, 3]}, {"A": [2, 3, 1]}, False, False, [5, 3, 2]),
            ({"A": [1, 2, 3]}, {"A": [1, 3, 2]}, True, False, [2, 3, 5]),
        ],
    )
    def test_sort_index_ignore_index(
        self, inplace, original_dict, sorted_dict, ascending, ignore_index, output_index
    ):
        # GH 30114
        original_index = [2, 5, 3]
        df = DataFrame(original_dict, index=original_index)
        expected_df = DataFrame(sorted_dict, index=output_index)
        kwargs = {
            "ascending": ascending,
            "ignore_index": ignore_index,
            "inplace": inplace,
        }

        if inplace:
            result_df = df.copy()
            result_df.sort_index(**kwargs)
        else:
            result_df = df.sort_index(**kwargs)

        tm.assert_frame_equal(result_df, expected_df)
        tm.assert_frame_equal(df, DataFrame(original_dict, index=original_index))

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("ignore_index", [True, False])
    def test_respect_ignore_index(self, inplace, ignore_index):
        # GH 43591
        df = DataFrame({"a": [1, 2, 3]}, index=RangeIndex(4, -1, -2))
        result = df.sort_index(
            ascending=False, ignore_index=ignore_index, inplace=inplace
        )

        if inplace:
            result = df
        if ignore_index:
            expected = DataFrame({"a": [1, 2, 3]})
        else:
            expected = DataFrame({"a": [1, 2, 3]}, index=RangeIndex(4, -1, -2))

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize(
        "original_dict, sorted_dict, ascending, ignore_index, output_index",
        [
            (
                {"M1": [1, 2], "M2": [3, 4]},
                {"M1": [1, 2], "M2": [3, 4]},
                True,
                True,
                [0, 1],
            ),
            (
                {"M1": [1, 2], "M2": [3, 4]},
                {"M1": [2, 1], "M2": [4, 3]},
                False,
                True,
                [0, 1],
            ),
            (
                {"M1": [1, 2], "M2": [3, 4]},
                {"M1": [1, 2], "M2": [3, 4]},
                True,
                False,
                MultiIndex.from_tuples([(2, 1), (3, 4)], names=list("AB")),
            ),
            (
                {"M1": [1, 2], "M2": [3, 4]},
                {"M1": [2, 1], "M2": [4, 3]},
                False,
                False,
                MultiIndex.from_tuples([(3, 4), (2, 1)], names=list("AB")),
            ),
        ],
    )
    def test_sort_index_ignore_index_multi_index(
        self, inplace, original_dict, sorted_dict, ascending, ignore_index, output_index
    ):
        # GH 30114, this is to test ignore_index on MultiIndex of index
        mi = MultiIndex.from_tuples([(2, 1), (3, 4)], names=list("AB"))
        df = DataFrame(original_dict, index=mi)
        expected_df = DataFrame(sorted_dict, index=output_index)

        kwargs = {
            "ascending": ascending,
            "ignore_index": ignore_index,
            "inplace": inplace,
        }

        if inplace:
            result_df = df.copy()
            result_df.sort_index(**kwargs)
        else:
            result_df = df.sort_index(**kwargs)

        tm.assert_frame_equal(result_df, expected_df)
        tm.assert_frame_equal(df, DataFrame(original_dict, index=mi))

    def test_sort_index_categorical_multiindex(self):
        # GH#15058
        df = DataFrame(
            {
                "a": range(6),
                "l1": pd.Categorical(
                    ["a", "a", "b", "b", "c", "c"],
                    categories=["c", "a", "b"],
                    ordered=True,
                ),
                "l2": [0, 1, 0, 1, 0, 1],
            }
        )
        result = df.set_index(["l1", "l2"]).sort_index()
        expected = DataFrame(
            [4, 5, 0, 1, 2, 3],
            columns=["a"],
            index=MultiIndex(
                levels=[
                    CategoricalIndex(
                        ["c", "a", "b"],
                        categories=["c", "a", "b"],
                        ordered=True,
                        name="l1",
                        dtype="category",
                    ),
                    [0, 1],
                ],
                codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
                names=["l1", "l2"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_sort_index_and_reconstruction(self):
        # GH#15622
        # lexsortedness should be identical
        # across MultiIndex construction methods

        df = DataFrame([[1, 1], [2, 2]], index=list("ab"))
        expected = DataFrame(
            [[1, 1], [2, 2], [1, 1], [2, 2]],
            index=MultiIndex.from_tuples(
                [(0.5, "a"), (0.5, "b"), (0.8, "a"), (0.8, "b")]
            ),
        )
        assert expected.index._is_lexsorted()

        result = DataFrame(
            [[1, 1], [2, 2], [1, 1], [2, 2]],
            index=MultiIndex.from_product([[0.5, 0.8], list("ab")]),
        )
        result = result.sort_index()
        assert result.index.is_monotonic_increasing

        tm.assert_frame_equal(result, expected)

        result = DataFrame(
            [[1, 1], [2, 2], [1, 1], [2, 2]],
            index=MultiIndex(
                levels=[[0.5, 0.8], ["a", "b"]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]
            ),
        )
        result = result.sort_index()
        assert result.index._is_lexsorted()

        tm.assert_frame_equal(result, expected)

        concatted = pd.concat([df, df], keys=[0.8, 0.5])
        result = concatted.sort_index()

        assert result.index.is_monotonic_increasing

        tm.assert_frame_equal(result, expected)

        # GH#14015
        df = DataFrame(
            [[1, 2], [6, 7]],
            columns=MultiIndex.from_tuples(
                [(0, "20160811 12:00:00"), (0, "20160809 12:00:00")],
                names=["l1", "Date"],
            ),
        )

        df.columns = df.columns.set_levels(
            pd.to_datetime(df.columns.levels[1]), level=1
        )
        assert not df.columns.is_monotonic_increasing
        result = df.sort_index(axis=1)
        assert result.columns.is_monotonic_increasing
        result = df.sort_index(axis=1, level=1)
        assert result.columns.is_monotonic_increasing

    # TODO: better name, de-duplicate with test_sort_index_level above
    def test_sort_index_level2(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        df = frame.copy()
        df.index = np.arange(len(df))

        # axis=1

        # series
        a_sorted = frame["A"].sort_index(level=0)

        # preserve names
        assert a_sorted.index.names == frame.index.names

        # inplace
        rs = frame.copy()
        return_value = rs.sort_index(level=0, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(rs, frame.sort_index(level=0))

    def test_sort_index_level_large_cardinality(self):
        # GH#2684 (int64)
        index = MultiIndex.from_arrays([np.arange(4000)] * 3)
        df = DataFrame(
            np.random.default_rng(2).standard_normal(4000).astype("int64"), index=index
        )

        # it works!
        result = df.sort_index(level=0)
        assert result.index._lexsort_depth == 3

        # GH#2684 (int32)
        index = MultiIndex.from_arrays([np.arange(4000)] * 3)
        df = DataFrame(
            np.random.default_rng(2).standard_normal(4000).astype("int32"), index=index
        )

        # it works!
        result = df.sort_index(level=0)
        assert (result.dtypes.values == df.dtypes.values).all()
        assert result.index._lexsort_depth == 3

    def test_sort_index_level_by_name(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        frame.index.names = ["first", "second"]
        result = frame.sort_index(level="second")
        expected = frame.sort_index(level=1)
        tm.assert_frame_equal(result, expected)

    def test_sort_index_level_mixed(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        sorted_before = frame.sort_index(level=1)

        df = frame.copy()
        df["foo"] = "bar"
        sorted_after = df.sort_index(level=1)
        tm.assert_frame_equal(sorted_before, sorted_after.drop(["foo"], axis=1))

        dft = frame.T
        sorted_before = dft.sort_index(level=1, axis=1)
        dft["foo", "three"] = "bar"

        sorted_after = dft.sort_index(level=1, axis=1)
        tm.assert_frame_equal(
            sorted_before.drop([("foo", "three")], axis=1),
            sorted_after.drop([("foo", "three")], axis=1),
        )

    def test_sort_index_preserve_levels(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        result = frame.sort_index()
        assert result.index.names == frame.index.names

    @pytest.mark.parametrize(
        "gen,extra",
        [
            ([1.0, 3.0, 2.0, 5.0], 4.0),
            ([1, 3, 2, 5], 4),
            (
                [
                    Timestamp("20130101"),
                    Timestamp("20130103"),
                    Timestamp("20130102"),
                    Timestamp("20130105"),
                ],
                Timestamp("20130104"),
            ),
            (["1one", "3one", "2one", "5one"], "4one"),
        ],
    )
    def test_sort_index_multilevel_repr_8017(self, gen, extra):
        data = np.random.default_rng(2).standard_normal((3, 4))

        columns = MultiIndex.from_tuples([("red", i) for i in gen])
        df = DataFrame(data, index=list("def"), columns=columns)
        df2 = pd.concat(
            [
                df,
                DataFrame(
                    "world",
                    index=list("def"),
                    columns=MultiIndex.from_tuples([("red", extra)]),
                ),
            ],
            axis=1,
        )

        # check that the repr is good
        # make sure that we have a correct sparsified repr
        # e.g. only 1 header of read
        assert str(df2).splitlines()[0].split() == ["red"]

        # GH 8017
        # sorting fails after columns added

        # construct single-dtype then sort
        result = df.copy().sort_index(axis=1)
        expected = df.iloc[:, [0, 2, 1, 3]]
        tm.assert_frame_equal(result, expected)

        result = df2.sort_index(axis=1)
        expected = df2.iloc[:, [0, 2, 1, 4, 3]]
        tm.assert_frame_equal(result, expected)

        # setitem then sort
        result = df.copy()
        result[("red", extra)] = "world"

        result = result.sort_index(axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "categories",
        [
            pytest.param(["a", "b", "c"], id="str"),
            pytest.param(
                [pd.Interval(0, 1), pd.Interval(1, 2), pd.Interval(2, 3)],
                id="pd.Interval",
            ),
        ],
    )
    def test_sort_index_with_categories(self, categories):
        # GH#23452
        df = DataFrame(
            {"foo": range(len(categories))},
            index=CategoricalIndex(
                data=categories, categories=categories, ordered=True
            ),
        )
        df.index = df.index.reorder_categories(df.index.categories[::-1])
        result = df.sort_index()
        expected = DataFrame(
            {"foo": reversed(range(len(categories)))},
            index=CategoricalIndex(
                data=categories[::-1], categories=categories[::-1], ordered=True
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "ascending",
        [
            None,
            [True, None],
            [False, "True"],
        ],
    )
    def test_sort_index_ascending_bad_value_raises(self, ascending):
        # GH 39434
        df = DataFrame(np.arange(64))
        length = len(df.index)
        df.index = [(i - length / 2) % length for i in range(length)]
        match = 'For argument "ascending" expected type bool'
        with pytest.raises(ValueError, match=match):
            df.sort_index(axis=0, ascending=ascending, na_position="first")

    def test_sort_index_use_inf_as_na(self):
        # GH 29687
        expected = DataFrame(
            {"col1": [1, 2, 3], "col2": [3, 4, 5]},
            index=pd.date_range("2020", periods=3),
        )
        msg = "use_inf_as_na option is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with pd.option_context("mode.use_inf_as_na", True):
                result = expected.sort_index()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "ascending",
        [(True, False), [True, False]],
    )
    def test_sort_index_ascending_tuple(self, ascending):
        df = DataFrame(
            {
                "legs": [4, 2, 4, 2, 2],
            },
            index=MultiIndex.from_tuples(
                [
                    ("mammal", "dog"),
                    ("bird", "duck"),
                    ("mammal", "horse"),
                    ("bird", "penguin"),
                    ("mammal", "kangaroo"),
                ],
                names=["class", "animal"],
            ),
        )

        # parameter `ascending`` is a tuple
        result = df.sort_index(level=(0, 1), ascending=ascending)

        expected = DataFrame(
            {
                "legs": [2, 2, 2, 4, 4],
            },
            index=MultiIndex.from_tuples(
                [
                    ("bird", "penguin"),
                    ("bird", "duck"),
                    ("mammal", "kangaroo"),
                    ("mammal", "horse"),
                    ("mammal", "dog"),
                ],
                names=["class", "animal"],
            ),
        )

        tm.assert_frame_equal(result, expected)


class TestDataFrameSortIndexKey:
    def test_sort_multi_index_key(self):
        # GH 25775, testing that sorting by index works with a multi-index.
        df = DataFrame(
            {"a": [3, 1, 2], "b": [0, 0, 0], "c": [0, 1, 2], "d": list("abc")}
        ).set_index(list("abc"))

        result = df.sort_index(level=list("ac"), key=lambda x: x)

        expected = DataFrame(
            {"a": [1, 2, 3], "b": [0, 0, 0], "c": [1, 2, 0], "d": list("bca")}
        ).set_index(list("abc"))
        tm.assert_frame_equal(result, expected)

        result = df.sort_index(level=list("ac"), key=lambda x: -x)
        expected = DataFrame(
            {"a": [3, 2, 1], "b": [0, 0, 0], "c": [0, 2, 1], "d": list("acb")}
        ).set_index(list("abc"))

        tm.assert_frame_equal(result, expected)

    def test_sort_index_key(self):  # issue 27237
        df = DataFrame(np.arange(6, dtype="int64"), index=list("aaBBca"))

        result = df.sort_index()
        expected = df.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_index(key=lambda x: x.str.lower())
        expected = df.iloc[[0, 1, 5, 2, 3, 4]]
        tm.assert_frame_equal(result, expected)

        result = df.sort_index(key=lambda x: x.str.lower(), ascending=False)
        expected = df.iloc[[4, 2, 3, 0, 1, 5]]
        tm.assert_frame_equal(result, expected)

    def test_sort_index_key_int(self):
        df = DataFrame(np.arange(6, dtype="int64"), index=np.arange(6, dtype="int64"))

        result = df.sort_index()
        tm.assert_frame_equal(result, df)

        result = df.sort_index(key=lambda x: -x)
        expected = df.sort_index(ascending=False)
        tm.assert_frame_equal(result, expected)

        result = df.sort_index(key=lambda x: 2 * x)
        tm.assert_frame_equal(result, df)

    def test_sort_multi_index_key_str(self):
        # GH 25775, testing that sorting by index works with a multi-index.
        df = DataFrame(
            {"a": ["B", "a", "C"], "b": [0, 1, 0], "c": list("abc"), "d": [0, 1, 2]}
        ).set_index(list("abc"))

        result = df.sort_index(level="a", key=lambda x: x.str.lower())

        expected = DataFrame(
            {"a": ["a", "B", "C"], "b": [1, 0, 0], "c": list("bac"), "d": [1, 0, 2]}
        ).set_index(list("abc"))
        tm.assert_frame_equal(result, expected)

        result = df.sort_index(
            level=list("abc"),  # can refer to names
            key=lambda x: x.str.lower() if x.name in ["a", "c"] else -x,
        )

        expected = DataFrame(
            {"a": ["a", "B", "C"], "b": [1, 0, 0], "c": list("bac"), "d": [1, 0, 2]}
        ).set_index(list("abc"))
        tm.assert_frame_equal(result, expected)

    def test_changes_length_raises(self):
        df = DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ValueError, match="change the shape"):
            df.sort_index(key=lambda x: x[:1])

    def test_sort_index_multiindex_sparse_column(self):
        # GH 29735, testing that sort_index on a multiindexed frame with sparse
        # columns fills with 0.
        expected = DataFrame(
            {
                i: pd.array([0.0, 0.0, 0.0, 0.0], dtype=pd.SparseDtype("float64", 0.0))
                for i in range(4)
            },
            index=MultiIndex.from_product([[1, 2], [1, 2]]),
        )

        result = expected.sort_index(level=0)

        tm.assert_frame_equal(result, expected)

    def test_sort_index_na_position(self):
        # GH#51612
        df = DataFrame([1, 2], index=MultiIndex.from_tuples([(1, 1), (1, pd.NA)]))
        expected = df.copy()
        result = df.sort_index(level=[0, 1], na_position="last")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_index_multiindex_sort_remaining(self, ascending):
        # GH #24247
        df = DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]},
            index=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y"), ("c", "x")]
            ),
        )

        result = df.sort_index(level=1, sort_remaining=False, ascending=ascending)

        if ascending:
            expected = DataFrame(
                {"A": [1, 3, 5, 2, 4], "B": [10, 30, 50, 20, 40]},
                index=MultiIndex.from_tuples(
                    [("a", "x"), ("b", "x"), ("c", "x"), ("a", "y"), ("b", "y")]
                ),
            )
        else:
            expected = DataFrame(
                {"A": [2, 4, 1, 3, 5], "B": [20, 40, 10, 30, 50]},
                index=MultiIndex.from_tuples(
                    [("a", "y"), ("b", "y"), ("a", "x"), ("b", "x"), ("c", "x")]
                ),
            )

        tm.assert_frame_equal(result, expected)


def test_sort_index_with_sliced_multiindex():
    # GH 55379
    mi = MultiIndex.from_tuples(
        [
            ("a", "10"),
            ("a", "18"),
            ("a", "25"),
            ("b", "16"),
            ("b", "26"),
            ("a", "45"),
            ("b", "28"),
            ("a", "5"),
            ("a", "50"),
            ("a", "51"),
            ("b", "4"),
        ],
        names=["group", "str"],
    )

    df = DataFrame({"x": range(len(mi))}, index=mi)
    result = df.iloc[0:6].sort_index()

    expected = DataFrame(
        {"x": [0, 1, 2, 5, 3, 4]},
        index=MultiIndex.from_tuples(
            [
                ("a", "10"),
                ("a", "18"),
                ("a", "25"),
                ("a", "45"),
                ("b", "16"),
                ("b", "26"),
            ],
            names=["group", "str"],
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_axis_columns_ignore_index():
    # GH 56478
    df = DataFrame([[1, 2]], columns=["d", "c"])
    result = df.sort_index(axis="columns", ignore_index=True)
    expected = DataFrame([[2, 1]])
    tm.assert_frame_equal(result, expected)


def test_sort_index_stable_sort():
    # GH 57151
    df = DataFrame(
        data=[
            (Timestamp("2024-01-30 13:00:00"), 13.0),
            (Timestamp("2024-01-30 13:00:00"), 13.1),
            (Timestamp("2024-01-30 12:00:00"), 12.0),
            (Timestamp("2024-01-30 12:00:00"), 12.1),
        ],
        columns=["dt", "value"],
    ).set_index(["dt"])
    result = df.sort_index(level="dt", kind="stable")
    expected = DataFrame(
        data=[
            (Timestamp("2024-01-30 12:00:00"), 12.0),
            (Timestamp("2024-01-30 12:00:00"), 12.1),
            (Timestamp("2024-01-30 13:00:00"), 13.0),
            (Timestamp("2024-01-30 13:00:00"), 13.1),
        ],
        columns=["dt", "value"],
    ).set_index(["dt"])
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\_config.py ===
# SPDX-License-Identifier: MIT

__all__ = ["get_run_validators", "set_run_validators"]

_run_validators = True


def set_run_validators(run):
    """
    Set whether or not validators are run.  By default, they are run.

    .. deprecated:: 21.3.0 It will not be removed, but it also will not be
        moved to new ``attrs`` namespace. Use `attrs.validators.set_disabled()`
        instead.
    """
    if not isinstance(run, bool):
        msg = "'run' must be bool."
        raise TypeError(msg)
    global _run_validators
    _run_validators = run


def get_run_validators():
    """
    Return whether or not validators are run.

    .. deprecated:: 21.3.0 It will not be removed, but it also will not be
        moved to new ``attrs`` namespace. Use `attrs.validators.get_disabled()`
        instead.
    """
    return _run_validators

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\cffi\error.py ===

class FFIError(Exception):
    __module__ = 'cffi'

class CDefError(Exception):
    __module__ = 'cffi'
    def __str__(self):
        try:
            current_decl = self.args[1]
            filename = current_decl.coord.file
            linenum = current_decl.coord.line
            prefix = '%s:%d: ' % (filename, linenum)
        except (AttributeError, TypeError, IndexError):
            prefix = ''
        return '%s%s' % (prefix, self.args[0])

class VerificationError(Exception):
    """ An error raised when verification fails
    """
    __module__ = 'cffi'

class VerificationMissing(Exception):
    """ An error raised when incomplete structures are passed into
    cdef, but no verification has been done
    """
    __module__ = 'cffi'

class PkgConfigError(Exception):
    """ An error raised for missing modules in pkg-config
    """
    __module__ = 'cffi'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\__init__.py ===

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\update_onnx_opset.py ===
#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pathlib

from .onnx_model_utils import update_onnx_opset


def update_onnx_opset_helper():
    parser = argparse.ArgumentParser(
        f"{os.path.basename(__file__)}:{update_onnx_opset_helper.__name__}",
        description="""
                                     Update the ONNX opset of the model.
                                     New opset must be later than the existing one.
                                     If not specified will update to opset 15.
                                     """,
    )

    parser.add_argument("--opset", type=int, required=False, default=15, help="ONNX opset to update to.")
    parser.add_argument("input_model", type=pathlib.Path, help="Provide path to ONNX model to update.")
    parser.add_argument("output_model", type=pathlib.Path, help="Provide path to write updated ONNX model to.")

    args = parser.parse_args()
    update_onnx_opset(args.input_model, args.opset, args.output_model)


if __name__ == "__main__":
    update_onnx_opset_helper()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\updown_bars.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import Typed
from openpyxl.descriptors.excel import ExtensionList

from .shapes import GraphicalProperties
from .axis import ChartLines
from .descriptors import NestedGapAmount


class UpDownBars(Serialisable):

    tagname = "upbars"

    gapWidth = NestedGapAmount()
    upBars = Typed(expected_type=ChartLines, allow_none=True)
    downBars = Typed(expected_type=ChartLines, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('gapWidth', 'upBars', 'downBars')

    def __init__(self,
                 gapWidth=150,
                 upBars=None,
                 downBars=None,
                 extLst=None,
                ):
        self.gapWidth = gapWidth
        self.upBars = upBars
        self.downBars = downBars

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexers\__init__.py ===
from pandas.core.indexers.utils import (
    check_array_indexer,
    check_key_length,
    check_setitem_lengths,
    disallow_ndim_indexing,
    is_empty_indexer,
    is_list_like_indexer,
    is_scalar_indexer,
    is_valid_positional_slice,
    length_of_indexer,
    maybe_convert_indices,
    unpack_1tuple,
    unpack_tuple_and_ellipses,
    validate_indices,
)

__all__ = [
    "is_valid_positional_slice",
    "is_list_like_indexer",
    "is_scalar_indexer",
    "is_empty_indexer",
    "check_setitem_lengths",
    "validate_indices",
    "maybe_convert_indices",
    "length_of_indexer",
    "disallow_ndim_indexing",
    "unpack_1tuple",
    "check_key_length",
    "check_array_indexer",
    "unpack_tuple_and_ellipses",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pyproject_hooks\__init__.py ===
"""Wrappers to call pyproject.toml-based build backend hooks.
"""

from typing import TYPE_CHECKING

from ._impl import (
    BackendUnavailable,
    BuildBackendHookCaller,
    HookMissing,
    UnsupportedOperation,
    default_subprocess_runner,
    quiet_subprocess_runner,
)

__version__ = "1.2.0"
__all__ = [
    "BackendUnavailable",
    "BackendInvalid",
    "HookMissing",
    "UnsupportedOperation",
    "default_subprocess_runner",
    "quiet_subprocess_runner",
    "BuildBackendHookCaller",
]

BackendInvalid = BackendUnavailable  # Deprecated alias, previously a separate exception

if TYPE_CHECKING:
    from ._impl import SubprocessRunner

    __all__ += ["SubprocessRunner"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\truststore\_ssl_constants.py ===
import ssl
import sys
import typing

# Hold on to the original class so we can create it consistently
# even if we inject our own SSLContext into the ssl module.
_original_SSLContext = ssl.SSLContext
_original_super_SSLContext = super(_original_SSLContext, _original_SSLContext)

# CPython is known to be good, but non-CPython implementations
# may implement SSLContext differently so to be safe we don't
# subclass the SSLContext.

# This is returned by truststore.SSLContext.__class__()
_truststore_SSLContext_dunder_class: typing.Optional[type]

# This value is the superclass of truststore.SSLContext.
_truststore_SSLContext_super_class: type

if sys.implementation.name == "cpython":
    _truststore_SSLContext_super_class = _original_SSLContext
    _truststore_SSLContext_dunder_class = None
else:
    _truststore_SSLContext_super_class = object
    _truststore_SSLContext_dunder_class = _original_SSLContext


def _set_ssl_context_verify_mode(
    ssl_context: ssl.SSLContext, verify_mode: ssl.VerifyMode
) -> None:
    _original_super_SSLContext.verify_mode.__set__(ssl_context, verify_mode)  # type: ignore[attr-defined]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\clipboard\ironpython_clipboard.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import clr
import System.Windows.Forms.Clipboard as cb

clr.AddReferenceByPartialName("System.Windows.Forms")


def get_clipboard_text() -> str:
    text = ""
    if cb.ContainsText():
        text = cb.GetText()

    return text


def set_clipboard_text(text: str) -> None:
    cb.SetText(text)


if __name__ == "__main__":
    txt = get_clipboard_text()  # display last text clipped
    print(txt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\pydy-example-repo\mass_spring_damper.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

m, k, b, g = _sm.symbols('m k b g', real=True)
position, speed = _me.dynamicsymbols('position speed')
position_d, speed_d = _me.dynamicsymbols('position_ speed_', 1)
o = _me.dynamicsymbols('o')
force = o*_sm.sin(_me.dynamicsymbols._t)
frame_ceiling = _me.ReferenceFrame('ceiling')
point_origin = _me.Point('origin')
point_origin.set_vel(frame_ceiling, 0)
particle_block = _me.Particle('block', _me.Point('block_pt'), _sm.Symbol('m'))
particle_block.point.set_pos(point_origin, position*frame_ceiling.x)
particle_block.mass = m
particle_block.point.set_vel(frame_ceiling, speed*frame_ceiling.x)
force_magnitude = m*g-k*position-b*speed+force
force_block = (force_magnitude*frame_ceiling.x).subs({position_d:speed})
kd_eqs = [position_d - speed]
forceList = [(particle_block.point,(force_magnitude*frame_ceiling.x).subs({position_d:speed}))]
kane = _me.KanesMethod(frame_ceiling, q_ind=[position], u_ind=[speed], kd_eqs = kd_eqs)
fr, frstar = kane.kanes_equations([particle_block], forceList)
zero = fr+frstar
from pydy.system import System
sys = System(kane, constants = {m:1.0, k:1.0, b:0.2, g:9.8},
specifieds={_me.dynamicsymbols('t'):lambda x, t: t, o:2},
initial_conditions={position:0.1, speed:-1*1.0},
times = _np.linspace(0.0, 10.0, 10.0/0.01))

y=sys.integrate()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\tests\test_xxm.py ===
#
# Test basic features of DDM, SDM and DFM.
#
# These three types are supposed to be interchangeable, so we should use the
# same tests for all of them for the most part.
#
# The tests here cover the basic part of the interface that the three types
# should expose and that DomainMatrix should mostly rely on.
#
# More in-depth tests of the heavier algorithms like rref etc should go in
# their own test files.
#
# Any new methods added to the DDM, SDM or DFM classes should be tested here
# and added to all classes.
#

from sympy.external.gmpy import GROUND_TYPES

from sympy import ZZ, QQ, GF, ZZ_I, symbols

from sympy.polys.matrices.exceptions import (
    DMBadInputError,
    DMDomainError,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError,
    DMShapeError,
)

from sympy.polys.matrices.domainmatrix import DM, DomainMatrix, DDM, SDM, DFM

from sympy.testing.pytest import raises, skip
import pytest


def test_XXM_constructors():
    """Test the DDM, etc constructors."""

    lol = [
        [ZZ(1), ZZ(2)],
        [ZZ(3), ZZ(4)],
        [ZZ(5), ZZ(6)],
    ]
    dod = {
        0: {0: ZZ(1), 1: ZZ(2)},
        1: {0: ZZ(3), 1: ZZ(4)},
        2: {0: ZZ(5), 1: ZZ(6)},
    }

    lol_0x0 = []
    lol_0x2 = []
    lol_2x0 = [[], []]
    dod_0x0 = {}
    dod_0x2 = {}
    dod_2x0 = {}

    lol_bad = [
        [ZZ(1), ZZ(2)],
        [ZZ(3), ZZ(4)],
        [ZZ(5), ZZ(6), ZZ(7)],
    ]
    dod_bad = {
        0: {0: ZZ(1), 1: ZZ(2)},
        1: {0: ZZ(3), 1: ZZ(4)},
        2: {0: ZZ(5), 1: ZZ(6), 2: ZZ(7)},
    }

    XDM_dense = [DDM]
    XDM_sparse = [SDM]

    if GROUND_TYPES == 'flint':
        XDM_dense.append(DFM)

    for XDM in XDM_dense:

        A = XDM(lol, (3, 2), ZZ)
        assert A.rows == 3
        assert A.cols == 2
        assert A.domain == ZZ
        assert A.shape == (3, 2)
        if XDM is not DFM:
            assert ZZ.of_type(A[0][0]) is True
        else:
            assert ZZ.of_type(A.rep[0, 0]) is True

        Adm = DomainMatrix(lol, (3, 2), ZZ)
        if XDM is DFM:
            assert Adm.rep == A
            assert Adm.rep.to_ddm() != A
        elif GROUND_TYPES == 'flint':
            assert Adm.rep.to_ddm() == A
            assert Adm.rep != A
        else:
            assert Adm.rep == A
            assert Adm.rep.to_ddm() == A

        assert XDM(lol_0x0, (0, 0), ZZ).shape == (0, 0)
        assert XDM(lol_0x2, (0, 2), ZZ).shape == (0, 2)
        assert XDM(lol_2x0, (2, 0), ZZ).shape == (2, 0)
        raises(DMBadInputError, lambda: XDM(lol, (2, 3), ZZ))
        raises(DMBadInputError, lambda: XDM(lol_bad, (3, 2), ZZ))
        raises(DMBadInputError, lambda: XDM(dod, (3, 2), ZZ))

    for XDM in XDM_sparse:

        A = XDM(dod, (3, 2), ZZ)
        assert A.rows == 3
        assert A.cols == 2
        assert A.domain == ZZ
        assert A.shape == (3, 2)
        assert ZZ.of_type(A[0][0]) is True

        assert DomainMatrix(dod, (3, 2), ZZ).rep == A

        assert XDM(dod_0x0, (0, 0), ZZ).shape == (0, 0)
        assert XDM(dod_0x2, (0, 2), ZZ).shape == (0, 2)
        assert XDM(dod_2x0, (2, 0), ZZ).shape == (2, 0)
        raises(DMBadInputError, lambda: XDM(dod, (2, 3), ZZ))
        raises(DMBadInputError, lambda: XDM(lol, (3, 2), ZZ))
        raises(DMBadInputError, lambda: XDM(dod_bad, (3, 2), ZZ))

    raises(DMBadInputError, lambda: DomainMatrix(lol, (2, 3), ZZ))
    raises(DMBadInputError, lambda: DomainMatrix(lol_bad, (3, 2), ZZ))
    raises(DMBadInputError, lambda: DomainMatrix(dod_bad, (3, 2), ZZ))


def test_XXM_eq():
    """Test equality for DDM, SDM, DFM and DomainMatrix."""

    lol1 = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    dod1 = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}

    lol2 = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(5)]]
    dod2 = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(5)}}

    A1_ddm = DDM(lol1, (2, 2), ZZ)
    A1_sdm = SDM(dod1, (2, 2), ZZ)
    A1_dm_d = DomainMatrix(lol1, (2, 2), ZZ)
    A1_dm_s = DomainMatrix(dod1, (2, 2), ZZ)

    A2_ddm = DDM(lol2, (2, 2), ZZ)
    A2_sdm = SDM(dod2, (2, 2), ZZ)
    A2_dm_d = DomainMatrix(lol2, (2, 2), ZZ)
    A2_dm_s = DomainMatrix(dod2, (2, 2), ZZ)

    A1_all = [A1_ddm, A1_sdm, A1_dm_d, A1_dm_s]
    A2_all = [A2_ddm, A2_sdm, A2_dm_d, A2_dm_s]

    if GROUND_TYPES == 'flint':

        A1_dfm = DFM([[1, 2], [3, 4]], (2, 2), ZZ)
        A2_dfm = DFM([[1, 2], [3, 5]], (2, 2), ZZ)

        A1_all.append(A1_dfm)
        A2_all.append(A2_dfm)

    for n, An in enumerate(A1_all):
        for m, Am in enumerate(A1_all):
            if n == m:
                assert (An == Am) is True
                assert (An != Am) is False
            else:
                assert (An == Am) is False
                assert (An != Am) is True

    for n, An in enumerate(A2_all):
        for m, Am in enumerate(A2_all):
            if n == m:
                assert (An == Am) is True
                assert (An != Am) is False
            else:
                assert (An == Am) is False
                assert (An != Am) is True

    for n, A1 in enumerate(A1_all):
        for m, A2 in enumerate(A2_all):
            assert (A1 == A2) is False
            assert (A1 != A2) is True


def test_to_XXM():
    """Test to_ddm etc. for DDM, SDM, DFM and DomainMatrix."""

    lol = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    dod = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}

    A_ddm = DDM(lol, (2, 2), ZZ)
    A_sdm = SDM(dod, (2, 2), ZZ)
    A_dm_d = DomainMatrix(lol, (2, 2), ZZ)
    A_dm_s = DomainMatrix(dod, (2, 2), ZZ)

    A_all = [A_ddm, A_sdm, A_dm_d, A_dm_s]

    if GROUND_TYPES == 'flint':
        A_dfm = DFM(lol, (2, 2), ZZ)
        A_all.append(A_dfm)

    for A in A_all:
        assert A.to_ddm() == A_ddm
        assert A.to_sdm() == A_sdm
        if GROUND_TYPES != 'flint':
            raises(NotImplementedError, lambda: A.to_dfm())
            assert A.to_dfm_or_ddm() == A_ddm

        # Add e.g. DDM.to_DM()?
        # assert A.to_DM() == A_dm

    if GROUND_TYPES == 'flint':
        for A in A_all:
            assert A.to_dfm() == A_dfm
            for K in [ZZ, QQ, GF(5), ZZ_I]:
                if isinstance(A, DFM) and not DFM._supports_domain(K):
                    raises(NotImplementedError, lambda: A.convert_to(K))
                else:
                    A_K = A.convert_to(K)
                    if DFM._supports_domain(K):
                        A_dfm_K = A_dfm.convert_to(K)
                        assert A_K.to_dfm() == A_dfm_K
                        assert A_K.to_dfm_or_ddm() == A_dfm_K
                    else:
                        raises(NotImplementedError, lambda: A_K.to_dfm())
                        assert A_K.to_dfm_or_ddm() == A_ddm.convert_to(K)


def test_DFM_domains():
    """Test which domains are supported by DFM."""

    x, y = symbols('x, y')

    if GROUND_TYPES in ('python', 'gmpy'):

        supported = []
        flint_funcs = {}
        not_supported = [ZZ, QQ, GF(5), QQ[x], QQ[x,y]]

    elif GROUND_TYPES == 'flint':

        import flint
        supported = [ZZ, QQ]
        flint_funcs = {
            ZZ: flint.fmpz_mat,
            QQ: flint.fmpq_mat,
            GF(5): None,
        }
        not_supported = [
            # Other domains could be supported but not implemented as matrices
            # in python-flint:
            QQ[x],
            QQ[x,y],
            QQ.frac_field(x,y),
            # Others would potentially never be supported by python-flint:
            ZZ_I,
        ]

    else:
        assert False, "Unknown GROUND_TYPES: %s" % GROUND_TYPES

    for domain in supported:
        assert DFM._supports_domain(domain) is True
        if flint_funcs[domain] is not None:
            assert DFM._get_flint_func(domain) == flint_funcs[domain]
    for domain in not_supported:
        assert DFM._supports_domain(domain) is False
        raises(NotImplementedError, lambda: DFM._get_flint_func(domain))


def _DM(lol, typ, K):
    """Make a DM of type typ over K from lol."""
    A = DM(lol, K)

    if typ == 'DDM':
        return A.to_ddm()
    elif typ == 'SDM':
        return A.to_sdm()
    elif typ == 'DFM':
        if GROUND_TYPES != 'flint':
            skip("DFM not supported in this ground type")
        return A.to_dfm()
    else:
        assert False, "Unknown type %s" % typ


def _DMZ(lol, typ):
    """Make a DM of type typ over ZZ from lol."""
    return _DM(lol, typ, ZZ)


def _DMQ(lol, typ):
    """Make a DM of type typ over QQ from lol."""
    return _DM(lol, typ, QQ)


def DM_ddm(lol, K):
    """Make a DDM over K from lol."""
    return _DM(lol, 'DDM', K)


def DM_sdm(lol, K):
    """Make a SDM over K from lol."""
    return _DM(lol, 'SDM', K)


def DM_dfm(lol, K):
    """Make a DFM over K from lol."""
    return _DM(lol, 'DFM', K)


def DMZ_ddm(lol):
    """Make a DDM from lol."""
    return _DMZ(lol, 'DDM')


def DMZ_sdm(lol):
    """Make a SDM from lol."""
    return _DMZ(lol, 'SDM')


def DMZ_dfm(lol):
    """Make a DFM from lol."""
    return _DMZ(lol, 'DFM')


def DMQ_ddm(lol):
    """Make a DDM from lol."""
    return _DMQ(lol, 'DDM')


def DMQ_sdm(lol):
    """Make a SDM from lol."""
    return _DMQ(lol, 'SDM')


def DMQ_dfm(lol):
    """Make a DFM from lol."""
    return _DMQ(lol, 'DFM')


DM_all = [DM_ddm, DM_sdm, DM_dfm]
DMZ_all = [DMZ_ddm, DMZ_sdm, DMZ_dfm]
DMQ_all = [DMQ_ddm, DMQ_sdm, DMQ_dfm]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XDM_getitem(DM):
    """Test getitem for DDM, etc."""

    lol = [[0, 1], [2, 0]]
    A = DM(lol)
    m, n = A.shape

    indices = [-3, -2, -1, 0, 1, 2]

    for i in indices:
        for j in indices:
            if -2 <= i < m and -2 <= j < n:
                assert A.getitem(i, j) == ZZ(lol[i][j])
            else:
                raises(IndexError, lambda: A.getitem(i, j))


@pytest.mark.parametrize('DM', DMZ_all)
def test_XDM_setitem(DM):
    """Test setitem for DDM, etc."""

    A = DM([[0, 1, 2], [3, 4, 5]])

    A.setitem(0, 0, ZZ(6))
    assert A == DM([[6, 1, 2], [3, 4, 5]])

    A.setitem(0, 1, ZZ(7))
    assert A == DM([[6, 7, 2], [3, 4, 5]])

    A.setitem(0, 2, ZZ(8))
    assert A == DM([[6, 7, 8], [3, 4, 5]])

    A.setitem(0, -1, ZZ(9))
    assert A == DM([[6, 7, 9], [3, 4, 5]])

    A.setitem(0, -2, ZZ(10))
    assert A == DM([[6, 10, 9], [3, 4, 5]])

    A.setitem(0, -3, ZZ(11))
    assert A == DM([[11, 10, 9], [3, 4, 5]])

    raises(IndexError, lambda: A.setitem(0, 3, ZZ(12)))
    raises(IndexError, lambda: A.setitem(0, -4, ZZ(13)))

    A.setitem(1, 0, ZZ(14))
    assert A == DM([[11, 10, 9], [14, 4, 5]])

    A.setitem(1, 1, ZZ(15))
    assert A == DM([[11, 10, 9], [14, 15, 5]])

    A.setitem(-1, 1, ZZ(16))
    assert A == DM([[11, 10, 9], [14, 16, 5]])

    A.setitem(-2, 1, ZZ(17))
    assert A == DM([[11, 17, 9], [14, 16, 5]])

    raises(IndexError, lambda: A.setitem(2, 0, ZZ(18)))
    raises(IndexError, lambda: A.setitem(-3, 0, ZZ(19)))

    A.setitem(1, 2, ZZ(0))
    assert A == DM([[11, 17, 9], [14, 16, 0]])

    A.setitem(1, -2, ZZ(0))
    assert A == DM([[11, 17, 9], [14, 0, 0]])

    A.setitem(1, -3, ZZ(0))
    assert A == DM([[11, 17, 9], [0, 0, 0]])

    A.setitem(0, 0, ZZ(0))
    assert A == DM([[0, 17, 9], [0, 0, 0]])

    A.setitem(0, -1, ZZ(0))
    assert A == DM([[0, 17, 0], [0, 0, 0]])

    A.setitem(0, 0, ZZ(0))
    assert A == DM([[0, 17, 0], [0, 0, 0]])

    A.setitem(0, -2, ZZ(0))
    assert A == DM([[0, 0, 0], [0, 0, 0]])

    A.setitem(0, -3, ZZ(1))
    assert A == DM([[1, 0, 0], [0, 0, 0]])


class _Sliced:
    def __getitem__(self, item):
        return item


_slice = _Sliced()


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_extract_slice(DM):
    A = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert A.extract_slice(*_slice[:,:]) == A
    assert A.extract_slice(*_slice[1:,:]) == DM([[4, 5, 6], [7, 8, 9]])
    assert A.extract_slice(*_slice[1:,1:]) == DM([[5, 6], [8, 9]])
    assert A.extract_slice(*_slice[1:,:-1]) == DM([[4, 5], [7, 8]])
    assert A.extract_slice(*_slice[1:,:-1:2]) == DM([[4], [7]])
    assert A.extract_slice(*_slice[:,::2]) == DM([[1, 3], [4, 6], [7, 9]])
    assert A.extract_slice(*_slice[::2,:]) == DM([[1, 2, 3], [7, 8, 9]])
    assert A.extract_slice(*_slice[::2,::2]) == DM([[1, 3], [7, 9]])
    assert A.extract_slice(*_slice[::2,::-2]) == DM([[3, 1], [9, 7]])
    assert A.extract_slice(*_slice[::-2,::2]) == DM([[7, 9], [1, 3]])
    assert A.extract_slice(*_slice[::-2,::-2]) == DM([[9, 7], [3, 1]])
    assert A.extract_slice(*_slice[:,::-1]) == DM([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
    assert A.extract_slice(*_slice[::-1,:]) == DM([[7, 8, 9], [4, 5, 6], [1, 2, 3]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_extract(DM):

    A = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    assert A.extract([0, 1, 2], [0, 1, 2]) == A
    assert A.extract([1, 2], [1, 2]) == DM([[5, 6], [8, 9]])
    assert A.extract([1, 2], [0, 1]) == DM([[4, 5], [7, 8]])
    assert A.extract([1, 2], [0, 2]) == DM([[4, 6], [7, 9]])
    assert A.extract([1, 2], [0]) == DM([[4], [7]])
    assert A.extract([1, 2], []) == DM([[1]]).zeros((2, 0), ZZ)
    assert A.extract([], [0, 1, 2]) == DM([[1]]).zeros((0, 3), ZZ)

    raises(IndexError, lambda: A.extract([1, 2], [0, 3]))
    raises(IndexError, lambda: A.extract([1, 2], [0, -4]))
    raises(IndexError, lambda: A.extract([3, 1], [0, 1]))
    raises(IndexError, lambda: A.extract([-4, 2], [3, 1]))

    B = DM([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert B.extract([1, 2], [1, 2]) == DM([[0, 0], [0, 0]])


def test_XXM_str():

    A = DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)

    assert str(A) == \
        'DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
    assert str(A.to_ddm()) == \
        '[[1, 2, 3], [4, 5, 6], [7, 8, 9]]'
    assert str(A.to_sdm()) == \
        '{0: {0: 1, 1: 2, 2: 3}, 1: {0: 4, 1: 5, 2: 6}, 2: {0: 7, 1: 8, 2: 9}}'

    assert repr(A) == \
        'DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
    assert repr(A.to_ddm()) == \
        'DDM([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
    assert repr(A.to_sdm()) == \
        'SDM({0: {0: 1, 1: 2, 2: 3}, 1: {0: 4, 1: 5, 2: 6}, 2: {0: 7, 1: 8, 2: 9}}, (3, 3), ZZ)'

    B = DomainMatrix({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3)}}, (2, 2), ZZ)

    assert str(B) == \
        'DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3}}, (2, 2), ZZ)'
    assert str(B.to_ddm()) == \
        '[[1, 2], [3, 0]]'
    assert str(B.to_sdm()) == \
        '{0: {0: 1, 1: 2}, 1: {0: 3}}'

    assert repr(B) == \
        'DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3}}, (2, 2), ZZ)'

    if GROUND_TYPES != 'gmpy':
        assert repr(B.to_ddm()) == \
            'DDM([[1, 2], [3, 0]], (2, 2), ZZ)'
        assert repr(B.to_sdm()) == \
            'SDM({0: {0: 1, 1: 2}, 1: {0: 3}}, (2, 2), ZZ)'
    else:
        assert repr(B.to_ddm()) == \
            'DDM([[mpz(1), mpz(2)], [mpz(3), mpz(0)]], (2, 2), ZZ)'
        assert repr(B.to_sdm()) == \
            'SDM({0: {0: mpz(1), 1: mpz(2)}, 1: {0: mpz(3)}}, (2, 2), ZZ)'

    if GROUND_TYPES == 'flint':

        assert str(A.to_dfm()) == \
            '[[1, 2, 3], [4, 5, 6], [7, 8, 9]]'
        assert str(B.to_dfm()) == \
            '[[1, 2], [3, 0]]'

        assert repr(A.to_dfm()) == \
            'DFM([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3), ZZ)'
        assert repr(B.to_dfm()) == \
            'DFM([[1, 2], [3, 0]], (2, 2), ZZ)'


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_list(DM):
    T = type(DM([[0]]))

    lol = [[1, 2, 4], [4, 5, 6]]
    lol_ZZ = [[ZZ(1), ZZ(2), ZZ(4)], [ZZ(4), ZZ(5), ZZ(6)]]
    lol_ZZ_bad = [[ZZ(1), ZZ(2), ZZ(4)], [ZZ(4), ZZ(5), ZZ(6), ZZ(7)]]

    assert T.from_list(lol_ZZ, (2, 3), ZZ) == DM(lol)
    raises(DMBadInputError, lambda: T.from_list(lol_ZZ_bad, (3, 2), ZZ))


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_list(DM):
    lol = [[1, 2, 4], [4, 5, 6]]
    assert DM(lol).to_list() == [[ZZ(1), ZZ(2), ZZ(4)], [ZZ(4), ZZ(5), ZZ(6)]]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_list_flat(DM):
    lol = [[1, 2, 4], [4, 5, 6]]
    assert DM(lol).to_list_flat() == [ZZ(1), ZZ(2), ZZ(4), ZZ(4), ZZ(5), ZZ(6)]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_list_flat(DM):
    T = type(DM([[0]]))
    flat = [ZZ(1), ZZ(2), ZZ(4), ZZ(4), ZZ(5), ZZ(6)]
    assert T.from_list_flat(flat, (2, 3), ZZ) == DM([[1, 2, 4], [4, 5, 6]])
    raises(DMBadInputError, lambda: T.from_list_flat(flat, (3, 3), ZZ))


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_flat_nz(DM):
    M = DM([[1, 2, 0], [0, 0, 0], [0, 0, 3]])
    elements = [ZZ(1), ZZ(2), ZZ(3)]
    indices = ((0, 0), (0, 1), (2, 2))
    assert M.to_flat_nz() == (elements, (indices, M.shape))


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_flat_nz(DM):
    T = type(DM([[0]]))
    elements = [ZZ(1), ZZ(2), ZZ(3)]
    indices = ((0, 0), (0, 1), (2, 2))
    data = (indices, (3, 3))
    result = DM([[1, 2, 0], [0, 0, 0], [0, 0, 3]])
    assert T.from_flat_nz(elements, data, ZZ) == result
    raises(DMBadInputError, lambda: T.from_flat_nz(elements, (indices, (2, 3)), ZZ))


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_dod(DM):
    dod = {0: {0: ZZ(1), 2: ZZ(4)}, 1: {0: ZZ(4), 1: ZZ(5), 2: ZZ(6)}}
    assert DM([[1, 0, 4], [4, 5, 6]]).to_dod() == dod


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_dod(DM):
    T = type(DM([[0]]))
    dod = {0: {0: ZZ(1), 2: ZZ(4)}, 1: {0: ZZ(4), 1: ZZ(5), 2: ZZ(6)}}
    assert T.from_dod(dod, (2, 3), ZZ) == DM([[1, 0, 4], [4, 5, 6]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_to_dok(DM):
    dod = {(0, 0): ZZ(1), (0, 2): ZZ(4),
           (1, 0): ZZ(4), (1, 1): ZZ(5), (1, 2): ZZ(6)}
    assert DM([[1, 0, 4], [4, 5, 6]]).to_dok() == dod


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_dok(DM):
    T = type(DM([[0]]))
    dod = {(0, 0): ZZ(1), (0, 2): ZZ(4),
           (1, 0): ZZ(4), (1, 1): ZZ(5), (1, 2): ZZ(6)}
    assert T.from_dok(dod, (2, 3), ZZ) == DM([[1, 0, 4], [4, 5, 6]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_iter_values(DM):
    values = [ZZ(1), ZZ(4), ZZ(4), ZZ(5), ZZ(6)]
    assert sorted(DM([[1, 0, 4], [4, 5, 6]]).iter_values()) == values


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_iter_items(DM):
    items = [((0, 0), ZZ(1)), ((0, 2), ZZ(4)),
             ((1, 0), ZZ(4)), ((1, 1), ZZ(5)), ((1, 2), ZZ(6))]
    assert sorted(DM([[1, 0, 4], [4, 5, 6]]).iter_items()) == items


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_from_ddm(DM):
    T = type(DM([[0]]))
    ddm = DDM([[1, 2, 4], [4, 5, 6]], (2, 3), ZZ)
    assert T.from_ddm(ddm) == DM([[1, 2, 4], [4, 5, 6]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_zeros(DM):
    T = type(DM([[0]]))
    assert T.zeros((2, 3), ZZ) == DM([[0, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_ones(DM):
    T = type(DM([[0]]))
    assert T.ones((2, 3), ZZ) == DM([[1, 1, 1], [1, 1, 1]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_eye(DM):
    T = type(DM([[0]]))
    assert T.eye(3, ZZ) == DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert T.eye((3, 2), ZZ) == DM([[1, 0], [0, 1], [0, 0]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_diag(DM):
    T = type(DM([[0]]))
    assert T.diag([1, 2, 3], ZZ) == DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_transpose(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    assert A.transpose() == DM([[1, 4], [2, 5], [3, 6]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_add(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[2, 4, 6], [8, 10, 12]])
    assert A.add(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_sub(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[0, 0, 0], [0, 0, 0]])
    assert A.sub(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_mul(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    b = ZZ(2)
    assert A.mul(b) == DM([[2, 4, 6], [8, 10, 12]])
    assert A.rmul(b) == DM([[2, 4, 6], [8, 10, 12]])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_matmul(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2], [3, 4], [5, 6]])
    C = DM([[22, 28], [49, 64]])
    assert A.matmul(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_mul_elementwise(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[1, 4, 9], [16, 25, 36]])
    assert A.mul_elementwise(B) == C


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_neg(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    C = DM([[-1, -2, -3], [-4, -5, -6]])
    assert A.neg() == C


@pytest.mark.parametrize('DM', DM_all)
def test_XXM_convert_to(DM):
    A = DM([[1, 2, 3], [4, 5, 6]], ZZ)
    B = DM([[1, 2, 3], [4, 5, 6]], QQ)
    assert A.convert_to(QQ) == B
    assert B.convert_to(ZZ) == A


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_scc(DM):
    A = DM([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1]])
    assert A.scc() == [[0, 1], [2], [3, 5], [4]]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_hstack(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[7, 8], [9, 10]])
    C = DM([[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]])
    ABC = DM([[1, 2, 3, 7, 8, 1, 2, 3, 7, 8],
              [4, 5, 6, 9, 10, 4, 5, 6, 9, 10]])
    assert A.hstack(B) == C
    assert A.hstack(B, C) == ABC


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_vstack(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[7, 8, 9]])
    C = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ABC = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert A.vstack(B) == C
    assert A.vstack(B, C) == ABC


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_applyfunc(DM):
    A = DM([[1, 2, 3], [4, 5, 6]])
    B = DM([[2, 4, 6], [8, 10, 12]])
    assert A.applyfunc(lambda x: 2*x, ZZ) == B


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_is_upper(DM):
    assert DM([[1, 2, 3], [0, 5, 6]]).is_upper() is True
    assert DM([[1, 2, 3], [4, 5, 6]]).is_upper() is False
    assert DM([]).is_upper() is True
    assert DM([[], []]).is_upper() is True


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_is_lower(DM):
    assert DM([[1, 0, 0], [4, 5, 0]]).is_lower() is True
    assert DM([[1, 2, 3], [4, 5, 6]]).is_lower() is False


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_is_diagonal(DM):
    assert DM([[1, 0, 0], [0, 5, 0]]).is_diagonal() is True
    assert DM([[1, 2, 3], [4, 5, 6]]).is_diagonal() is False


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_diagonal(DM):
    assert DM([[1, 0, 0], [0, 5, 0]]).diagonal() == [1, 5]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_is_zero_matrix(DM):
    assert DM([[0, 0, 0], [0, 0, 0]]).is_zero_matrix() is True
    assert DM([[1, 0, 0], [0, 0, 0]]).is_zero_matrix() is False


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_det_ZZ(DM):
    assert DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).det() == 0
    assert DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).det() == -3


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_det_QQ(DM):
    dM1 = DM([[(1,2), (2,3)], [(3,4), (4,5)]])
    assert dM1.det() == QQ(-1,10)


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_inv_QQ(DM):
    dM1 = DM([[(1,2), (2,3)], [(3,4), (4,5)]])
    dM2 = DM([[(-8,1), (20,3)], [(15,2), (-5,1)]])
    assert dM1.inv() == dM2
    assert dM1.matmul(dM2) == DM([[1, 0], [0, 1]])

    dM3 = DM([[(1,2), (2,3)], [(1,4), (1,3)]])
    raises(DMNonInvertibleMatrixError, lambda: dM3.inv())

    dM4 = DM([[(1,2), (2,3), (3,4)], [(1,4), (1,3), (1,2)]])
    raises(DMNonSquareMatrixError, lambda: dM4.inv())


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_inv_ZZ(DM):
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    # XXX: Maybe this should return a DM over QQ instead?
    # XXX: Handle unimodular matrices?
    raises(DMDomainError, lambda: dM1.inv())


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_charpoly_ZZ(DM):
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    assert dM1.charpoly() == [1, -16, -12, 3]


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_charpoly_QQ(DM):
    dM1 = DM([[(1,2), (2,3)], [(3,4), (4,5)]])
    assert dM1.charpoly() == [QQ(1,1), QQ(-13,10), QQ(-1,10)]


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_lu_solve_ZZ(DM):
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    dM2 = DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    raises(DMDomainError, lambda: dM1.lu_solve(dM2))


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_lu_solve_QQ(DM):
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    dM2 = DM([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dM3 = DM([[(-2,3),(-4,3),(1,1)],[(-2,3),(11,3),(-2,1)],[(1,1),(-2,1),(1,1)]])
    assert dM1.lu_solve(dM2) == dM3 == dM1.inv()

    dM4 = DM([[1, 2, 3], [4, 5, 6]])
    dM5 = DM([[1, 0], [0, 1], [0, 0]])
    raises(DMShapeError, lambda: dM4.lu_solve(dM5))


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_nullspace_QQ(DM):
    dM1 = DM([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # XXX: Change the signature to just return the nullspace. Possibly
    # returning the rank or nullity makes sense but the list of nonpivots is
    # not useful.
    assert dM1.nullspace() == (DM([[1, -2, 1]]), [2])


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_lll(DM):
    M = DM([[1, 2, 3], [4, 5, 20]])
    M_lll = DM([[1, 2, 3], [-1, -5, 5]])
    T = DM([[1, 0], [-5, 1]])
    assert M.lll() == M_lll
    assert M.lll_transform() == (M_lll, T)
    assert T.matmul(M) == M_lll


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_mixed_signs(DM):
    lol = [[QQ(1), QQ(-2)], [QQ(-3), QQ(4)]]
    A = DM(lol)
    Q, R = A.qr()
    assert Q.matmul(R) == A
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_large_matrix(DM):
    lol = [[QQ(i + j) for j in range(10)] for i in range(10)]
    A = DM(lol)
    Q, R = A.qr()
    assert Q.matmul(R) == A
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_identity_matrix(DM):
    T = type(DM([[0]]))
    A = T.eye(3, QQ)
    Q, R = A.qr()
    assert Q == A
    assert R == A
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper
    assert Q.shape == (3, 3)
    assert R.shape == (3, 3)


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_square_matrix(DM):
    lol = [[QQ(3), QQ(1)], [QQ(4), QQ(3)]]
    A = DM(lol)
    Q, R = A.qr()
    assert Q.matmul(R) == A
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_matrix_with_zero_columns(DM):
    lol = [[QQ(3), QQ(0)], [QQ(4), QQ(0)]]
    A = DM(lol)
    Q, R = A.qr()
    assert Q.matmul(R) == A
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_linearly_dependent_columns(DM):
    lol = [[QQ(1), QQ(2)], [QQ(2), QQ(4)]]
    A = DM(lol)
    Q, R = A.qr()
    assert Q.matmul(R) == A
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_qr_non_field(DM):
    lol = [[ZZ(3), ZZ(1)], [ZZ(4), ZZ(3)]]
    A = DM(lol)
    with pytest.raises(DMDomainError):
        A.qr()


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_field(DM):
    lol = [[QQ(3), QQ(1)], [QQ(4), QQ(3)]]
    A = DM(lol)
    Q, R = A.qr()
    assert Q.matmul(R) == A
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_tall_matrix(DM):
    lol = [[QQ(1), QQ(2)], [QQ(3), QQ(4)], [QQ(5), QQ(6)]]
    A = DM(lol)
    Q, R = A.qr()
    assert Q.matmul(R) == A
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_wide_matrix(DM):
    lol = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]]
    A = DM(lol)
    Q, R = A.qr()
    assert Q.matmul(R) == A
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_empty_matrix_0x0(DM):
    T = type(DM([[0]]))
    A = T.zeros((0, 0), QQ)
    Q, R = A.qr()
    assert Q.matmul(R).shape == A.shape
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper
    assert Q.shape == (0, 0)
    assert R.shape == (0, 0)


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_empty_matrix_2x0(DM):
    T = type(DM([[0]]))
    A = T.zeros((2, 0), QQ)
    Q, R = A.qr()
    assert Q.matmul(R).shape == A.shape
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper
    assert Q.shape == (2, 0)
    assert R.shape == (0, 0)


@pytest.mark.parametrize('DM', DMQ_all)
def test_XXM_qr_empty_matrix_0x2(DM):
    T = type(DM([[0]]))
    A = T.zeros((0, 2), QQ)
    Q, R = A.qr()
    assert Q.matmul(R).shape == A.shape
    assert (Q.transpose().matmul(Q)).is_diagonal
    assert R.is_upper
    assert Q.shape == (0, 0)
    assert R.shape == (0, 2)


@pytest.mark.parametrize('DM', DMZ_all)
def test_XXM_fflu(DM):
    A = DM([[1, 2], [3, 4]])
    P, L, D, U = A.fflu()
    A_field = A.convert_to(QQ)
    P_field = P.convert_to(QQ)
    L_field = L.convert_to(QQ)
    D_field = D.convert_to(QQ)
    U_field = U.convert_to(QQ)
    assert P.shape == A.shape
    assert L.shape == A.shape
    assert D.shape == A.shape
    assert U.shape == A.shape
    assert P == DM([[1, 0], [0, 1]])
    assert L == DM([[1, 0], [3, -2]])
    assert D == DM([[1, 0], [0, -2]])
    assert U == DM([[1, 2], [0, -2]])
    assert L_field.matmul(D_field.inv()).matmul(U_field) == P_field.matmul(A_field)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_io_common.py ===
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import outcome

from .. import _core

if TYPE_CHECKING:
    from ._io_epoll import EpollWaiters
    from ._io_windows import AFDWaiters


# Utility function shared between _io_epoll and _io_windows
def wake_all(waiters: EpollWaiters | AFDWaiters, exc: BaseException) -> None:
    try:
        current_task = _core.current_task()
    except RuntimeError:
        current_task = None
    raise_at_end = False
    for attr_name in ["read_task", "write_task"]:
        task = getattr(waiters, attr_name)
        if task is not None:
            if task is current_task:
                raise_at_end = True
            else:
                _core.reschedule(task, outcome.Error(copy.copy(exc)))
            setattr(waiters, attr_name, None)
    if raise_at_end:
        raise exc

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\pytables\test_append.py ===
import datetime
from datetime import timedelta
import re

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    _testing as tm,
    concat,
    date_range,
    read_hdf,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)

pytestmark = [pytest.mark.single_cpu]

tables = pytest.importorskip("tables")


@pytest.mark.filterwarnings("ignore::tables.NaturalNameWarning")
def test_append(setup_path):
    with ensure_clean_store(setup_path) as store:
        # this is allowed by almost always don't want to do it
        # tables.NaturalNameWarning):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=20, freq="B"),
        )
        _maybe_remove(store, "df1")
        store.append("df1", df[:10])
        store.append("df1", df[10:])
        tm.assert_frame_equal(store["df1"], df)

        _maybe_remove(store, "df2")
        store.put("df2", df[:10], format="table")
        store.append("df2", df[10:])
        tm.assert_frame_equal(store["df2"], df)

        _maybe_remove(store, "df3")
        store.append("/df3", df[:10])
        store.append("/df3", df[10:])
        tm.assert_frame_equal(store["df3"], df)

        # this is allowed by almost always don't want to do it
        # tables.NaturalNameWarning
        _maybe_remove(store, "/df3 foo")
        store.append("/df3 foo", df[:10])
        store.append("/df3 foo", df[10:])
        tm.assert_frame_equal(store["df3 foo"], df)

        # dtype issues - mizxed type in a single object column
        df = DataFrame(data=[[1, 2], [0, 1], [1, 2], [0, 0]])
        df["mixed_column"] = "testing"
        df.loc[2, "mixed_column"] = np.nan
        _maybe_remove(store, "df")
        store.append("df", df)
        tm.assert_frame_equal(store["df"], df)

        # uints - test storage of uints
        uint_data = DataFrame(
            {
                "u08": Series(
                    np.random.default_rng(2).integers(0, high=255, size=5),
                    dtype=np.uint8,
                ),
                "u16": Series(
                    np.random.default_rng(2).integers(0, high=65535, size=5),
                    dtype=np.uint16,
                ),
                "u32": Series(
                    np.random.default_rng(2).integers(0, high=2**30, size=5),
                    dtype=np.uint32,
                ),
                "u64": Series(
                    [2**58, 2**59, 2**60, 2**61, 2**62],
                    dtype=np.uint64,
                ),
            },
            index=np.arange(5),
        )
        _maybe_remove(store, "uints")
        store.append("uints", uint_data)
        tm.assert_frame_equal(store["uints"], uint_data, check_index_type=True)

        # uints - test storage of uints in indexable columns
        _maybe_remove(store, "uints")
        # 64-bit indices not yet supported
        store.append("uints", uint_data, data_columns=["u08", "u16", "u32"])
        tm.assert_frame_equal(store["uints"], uint_data, check_index_type=True)


def test_append_series(setup_path):
    with ensure_clean_store(setup_path) as store:
        # basic
        ss = Series(range(20), dtype=np.float64, index=[f"i_{i}" for i in range(20)])
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        ns = Series(np.arange(100))

        store.append("ss", ss)
        result = store["ss"]
        tm.assert_series_equal(result, ss)
        assert result.name is None

        store.append("ts", ts)
        result = store["ts"]
        tm.assert_series_equal(result, ts)
        assert result.name is None

        ns.name = "foo"
        store.append("ns", ns)
        result = store["ns"]
        tm.assert_series_equal(result, ns)
        assert result.name == ns.name

        # select on the values
        expected = ns[ns > 60]
        result = store.select("ns", "foo>60")
        tm.assert_series_equal(result, expected)

        # select on the index and values
        expected = ns[(ns > 70) & (ns.index < 90)]
        result = store.select("ns", "foo>70 and index<90")
        tm.assert_series_equal(result, expected, check_index_type=True)

        # multi-index
        mi = DataFrame(np.random.default_rng(2).standard_normal((5, 1)), columns=["A"])
        mi["B"] = np.arange(len(mi))
        mi["C"] = "foo"
        mi.loc[3:5, "C"] = "bar"
        mi.set_index(["C", "B"], inplace=True)
        s = mi.stack(future_stack=True)
        s.index = s.index.droplevel(2)
        store.append("mi", s)
        tm.assert_series_equal(store["mi"], s, check_index_type=True)


def test_append_some_nans(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            {
                "A": Series(np.random.default_rng(2).standard_normal(20)).astype(
                    "int32"
                ),
                "A1": np.random.default_rng(2).standard_normal(20),
                "A2": np.random.default_rng(2).standard_normal(20),
                "B": "foo",
                "C": "bar",
                "D": Timestamp("2001-01-01").as_unit("ns"),
                "E": Timestamp("2001-01-02").as_unit("ns"),
            },
            index=np.arange(20),
        )
        # some nans
        _maybe_remove(store, "df1")
        df.loc[0:15, ["A1", "B", "D", "E"]] = np.nan
        store.append("df1", df[:10])
        store.append("df1", df[10:])
        tm.assert_frame_equal(store["df1"], df, check_index_type=True)

        # first column
        df1 = df.copy()
        df1["A1"] = np.nan
        _maybe_remove(store, "df1")
        store.append("df1", df1[:10])
        store.append("df1", df1[10:])
        tm.assert_frame_equal(store["df1"], df1, check_index_type=True)

        # 2nd column
        df2 = df.copy()
        df2["A2"] = np.nan
        _maybe_remove(store, "df2")
        store.append("df2", df2[:10])
        store.append("df2", df2[10:])
        tm.assert_frame_equal(store["df2"], df2, check_index_type=True)

        # datetimes
        df3 = df.copy()
        df3["E"] = np.nan
        _maybe_remove(store, "df3")
        store.append("df3", df3[:10])
        store.append("df3", df3[10:])
        tm.assert_frame_equal(store["df3"], df3, check_index_type=True)


def test_append_all_nans(setup_path, using_infer_string):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            {
                "A1": np.random.default_rng(2).standard_normal(20),
                "A2": np.random.default_rng(2).standard_normal(20),
            },
            index=np.arange(20),
        )
        df.loc[0:15, :] = np.nan

        # nan some entire rows (dropna=True)
        _maybe_remove(store, "df")
        store.append("df", df[:10], dropna=True)
        store.append("df", df[10:], dropna=True)
        tm.assert_frame_equal(store["df"], df[-4:], check_index_type=True)

        # nan some entire rows (dropna=False)
        _maybe_remove(store, "df2")
        store.append("df2", df[:10], dropna=False)
        store.append("df2", df[10:], dropna=False)
        tm.assert_frame_equal(store["df2"], df, check_index_type=True)

        # tests the option io.hdf.dropna_table
        with pd.option_context("io.hdf.dropna_table", False):
            _maybe_remove(store, "df3")
            store.append("df3", df[:10])
            store.append("df3", df[10:])
            tm.assert_frame_equal(store["df3"], df)

        with pd.option_context("io.hdf.dropna_table", True):
            _maybe_remove(store, "df4")
            store.append("df4", df[:10])
            store.append("df4", df[10:])
            tm.assert_frame_equal(store["df4"], df[-4:])

            # nan some entire rows (string are still written!)
            df = DataFrame(
                {
                    "A1": np.random.default_rng(2).standard_normal(20),
                    "A2": np.random.default_rng(2).standard_normal(20),
                    "B": "foo",
                    "C": "bar",
                },
                index=np.arange(20),
            )

            df.loc[0:15, :] = np.nan

            _maybe_remove(store, "df")
            store.append("df", df[:10], dropna=True)
            store.append("df", df[10:], dropna=True)
            result = store["df"]
            expected = df
            if using_infer_string:
                # TODO: Test is incorrect when not using_infer_string.
                #       Should take the last 4 rows uncondiationally.
                expected = expected[-4:]
            tm.assert_frame_equal(result, expected, check_index_type=True)

            _maybe_remove(store, "df2")
            store.append("df2", df[:10], dropna=False)
            store.append("df2", df[10:], dropna=False)
            tm.assert_frame_equal(store["df2"], df, check_index_type=True)

            # nan some entire rows (but since we have dates they are still
            # written!)
            df = DataFrame(
                {
                    "A1": np.random.default_rng(2).standard_normal(20),
                    "A2": np.random.default_rng(2).standard_normal(20),
                    "B": "foo",
                    "C": "bar",
                    "D": Timestamp("2001-01-01").as_unit("ns"),
                    "E": Timestamp("2001-01-02").as_unit("ns"),
                },
                index=np.arange(20),
            )

            df.loc[0:15, :] = np.nan

            _maybe_remove(store, "df")
            store.append("df", df[:10], dropna=True)
            store.append("df", df[10:], dropna=True)
            tm.assert_frame_equal(store["df"], df, check_index_type=True)

            _maybe_remove(store, "df2")
            store.append("df2", df[:10], dropna=False)
            store.append("df2", df[10:], dropna=False)
            tm.assert_frame_equal(store["df2"], df, check_index_type=True)


def test_append_frame_column_oriented(setup_path):
    with ensure_clean_store(setup_path) as store:
        # column oriented
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df.index = df.index._with_freq(None)  # freq doesn't round-trip

        _maybe_remove(store, "df1")
        store.append("df1", df.iloc[:, :2], axes=["columns"])
        store.append("df1", df.iloc[:, 2:])
        tm.assert_frame_equal(store["df1"], df)

        result = store.select("df1", "columns=A")
        expected = df.reindex(columns=["A"])
        tm.assert_frame_equal(expected, result)

        # selection on the non-indexable
        result = store.select("df1", ("columns=A", "index=df.index[0:4]"))
        expected = df.reindex(columns=["A"], index=df.index[0:4])
        tm.assert_frame_equal(expected, result)

        # this isn't supported
        msg = re.escape(
            "passing a filterable condition to a non-table indexer "
            "[Filter: Not Initialized]"
        )
        with pytest.raises(TypeError, match=msg):
            store.select("df1", "columns=A and index>df.index[4]")


def test_append_with_different_block_ordering(setup_path):
    # GH 4096; using same frames, but different block orderings
    with ensure_clean_store(setup_path) as store:
        for i in range(10):
            df = DataFrame(
                np.random.default_rng(2).standard_normal((10, 2)), columns=list("AB")
            )
            df["index"] = range(10)
            df["index"] += i * 10
            df["int64"] = Series([1] * len(df), dtype="int64")
            df["int16"] = Series([1] * len(df), dtype="int16")

            if i % 2 == 0:
                del df["int64"]
                df["int64"] = Series([1] * len(df), dtype="int64")
            if i % 3 == 0:
                a = df.pop("A")
                df["A"] = a

            df.set_index("index", inplace=True)

            store.append("df", df)

    # test a different ordering but with more fields (like invalid
    # combinations)
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            columns=list("AB"),
            dtype="float64",
        )
        df["int64"] = Series([1] * len(df), dtype="int64")
        df["int16"] = Series([1] * len(df), dtype="int16")
        store.append("df", df)

        # store additional fields in different blocks
        df["int16_2"] = Series([1] * len(df), dtype="int16")
        msg = re.escape(
            "cannot match existing table structure for [int16] on appending data"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)

        # store multiple additional fields in different blocks
        df["float_3"] = Series([1.0] * len(df), dtype="float64")
        msg = re.escape(
            "cannot match existing table structure for [A,B] on appending data"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)


def test_append_with_strings(setup_path):
    with ensure_clean_store(setup_path) as store:

        def check_col(key, name, size):
            assert (
                getattr(store.get_storer(key).table.description, name).itemsize == size
            )

        # avoid truncation on elements
        df = DataFrame([[123, "asdqwerty"], [345, "dggnhebbsdfbdfb"]])
        store.append("df_big", df)
        tm.assert_frame_equal(store.select("df_big"), df)
        check_col("df_big", "values_block_1", 15)

        # appending smaller string ok
        df2 = DataFrame([[124, "asdqy"], [346, "dggnhefbdfb"]])
        store.append("df_big", df2)
        expected = concat([df, df2])
        tm.assert_frame_equal(store.select("df_big"), expected)
        check_col("df_big", "values_block_1", 15)

        # avoid truncation on elements
        df = DataFrame([[123, "asdqwerty"], [345, "dggnhebbsdfbdfb"]])
        store.append("df_big2", df, min_itemsize={"values": 50})
        tm.assert_frame_equal(store.select("df_big2"), df)
        check_col("df_big2", "values_block_1", 50)

        # bigger string on next append
        store.append("df_new", df)
        df_new = DataFrame([[124, "abcdefqhij"], [346, "abcdefghijklmnopqrtsuvwxyz"]])
        msg = (
            r"Trying to store a string with len \[26\] in "
            r"\[values_block_1\] column but\n"
            r"this column has a limit of \[15\]!\n"
            "Consider using min_itemsize to preset the sizes on these "
            "columns"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df_new", df_new)

        # min_itemsize on Series index (GH 11412)
        df = DataFrame(
            {
                "A": [0.0, 1.0, 2.0, 3.0, 4.0],
                "B": [0.0, 1.0, 0.0, 1.0, 0.0],
                "C": Index(["foo1", "foo2", "foo3", "foo4", "foo5"]),
                "D": date_range("20130101", periods=5),
            }
        ).set_index("C")
        store.append("ss", df["B"], min_itemsize={"index": 4})
        tm.assert_series_equal(store.select("ss"), df["B"])

        # same as above, with data_columns=True
        store.append("ss2", df["B"], data_columns=True, min_itemsize={"index": 4})
        tm.assert_series_equal(store.select("ss2"), df["B"])

        # min_itemsize in index without appending (GH 10381)
        store.put("ss3", df, format="table", min_itemsize={"index": 6})
        # just make sure there is a longer string:
        df2 = df.copy().reset_index().assign(C="longer").set_index("C")
        store.append("ss3", df2)
        tm.assert_frame_equal(store.select("ss3"), concat([df, df2]))

        # same as above, with a Series
        store.put("ss4", df["B"], format="table", min_itemsize={"index": 6})
        store.append("ss4", df2["B"])
        tm.assert_series_equal(store.select("ss4"), concat([df["B"], df2["B"]]))

        # with nans
        _maybe_remove(store, "df")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df["string"] = "foo"
        df.loc[df.index[1:4], "string"] = np.nan
        df["string2"] = "bar"
        df.loc[df.index[4:8], "string2"] = np.nan
        df["string3"] = "bah"
        df.loc[df.index[1:], "string3"] = np.nan
        store.append("df", df)
        result = store.select("df")
        tm.assert_frame_equal(result, df)

    with ensure_clean_store(setup_path) as store:
        df = DataFrame({"A": "foo", "B": "bar"}, index=range(10))

        # a min_itemsize that creates a data_column
        _maybe_remove(store, "df")
        store.append("df", df, min_itemsize={"A": 200})
        check_col("df", "A", 200)
        assert store.get_storer("df").data_columns == ["A"]

        # a min_itemsize that creates a data_column2
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=["B"], min_itemsize={"A": 200})
        check_col("df", "A", 200)
        assert store.get_storer("df").data_columns == ["B", "A"]

        # a min_itemsize that creates a data_column2
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=["B"], min_itemsize={"values": 200})
        check_col("df", "B", 200)
        check_col("df", "values_block_0", 200)
        assert store.get_storer("df").data_columns == ["B"]

        # infer the .typ on subsequent appends
        _maybe_remove(store, "df")
        store.append("df", df[:5], min_itemsize=200)
        store.append("df", df[5:], min_itemsize=200)
        tm.assert_frame_equal(store["df"], df)

        # invalid min_itemsize keys
        df = DataFrame(["foo", "foo", "foo", "barh", "barh", "barh"], columns=["A"])
        _maybe_remove(store, "df")
        msg = re.escape(
            "min_itemsize has the key [foo] which is not an axis or data_column"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df", df, min_itemsize={"foo": 20, "foobar": 20})


def test_append_with_empty_string(setup_path):
    with ensure_clean_store(setup_path) as store:
        # with all empty strings (GH 12242)
        df = DataFrame({"x": ["a", "b", "c", "d", "e", "f", ""]})
        store.append("df", df[:-1], min_itemsize={"x": 1})
        store.append("df", df[-1:], min_itemsize={"x": 1})
        tm.assert_frame_equal(store.select("df"), df)


def test_append_with_data_columns(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df.iloc[0, df.columns.get_loc("B")] = 1.0
        _maybe_remove(store, "df")
        store.append("df", df[:2], data_columns=["B"])
        store.append("df", df[2:])
        tm.assert_frame_equal(store["df"], df)

        # check that we have indices created
        assert store._handle.root.df.table.cols.index.is_indexed is True
        assert store._handle.root.df.table.cols.B.is_indexed is True

        # data column searching
        result = store.select("df", "B>0")
        expected = df[df.B > 0]
        tm.assert_frame_equal(result, expected)

        # data column searching (with an indexable and a data_columns)
        result = store.select("df", "B>0 and index>df.index[3]")
        df_new = df.reindex(index=df.index[4:])
        expected = df_new[df_new.B > 0]
        tm.assert_frame_equal(result, expected)

        # data column selection with a string data_column
        df_new = df.copy()
        df_new["string"] = "foo"
        df_new.loc[df_new.index[1:4], "string"] = np.nan
        df_new.loc[df_new.index[5:6], "string"] = "bar"
        _maybe_remove(store, "df")
        store.append("df", df_new, data_columns=["string"])
        result = store.select("df", "string='foo'")
        expected = df_new[df_new.string == "foo"]
        tm.assert_frame_equal(result, expected)

        # using min_itemsize and a data column
        def check_col(key, name, size):
            assert (
                getattr(store.get_storer(key).table.description, name).itemsize == size
            )

    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, "df")
        store.append("df", df_new, data_columns=["string"], min_itemsize={"string": 30})
        check_col("df", "string", 30)
        _maybe_remove(store, "df")
        store.append("df", df_new, data_columns=["string"], min_itemsize=30)
        check_col("df", "string", 30)
        _maybe_remove(store, "df")
        store.append("df", df_new, data_columns=["string"], min_itemsize={"values": 30})
        check_col("df", "string", 30)

    with ensure_clean_store(setup_path) as store:
        df_new["string2"] = "foobarbah"
        df_new["string_block1"] = "foobarbah1"
        df_new["string_block2"] = "foobarbah2"
        _maybe_remove(store, "df")
        store.append(
            "df",
            df_new,
            data_columns=["string", "string2"],
            min_itemsize={"string": 30, "string2": 40, "values": 50},
        )
        check_col("df", "string", 30)
        check_col("df", "string2", 40)
        check_col("df", "values_block_1", 50)

    with ensure_clean_store(setup_path) as store:
        # multiple data columns
        df_new = df.copy()
        df_new.iloc[0, df_new.columns.get_loc("A")] = 1.0
        df_new.iloc[0, df_new.columns.get_loc("B")] = -1.0
        df_new["string"] = "foo"

        sl = df_new.columns.get_loc("string")
        df_new.iloc[1:4, sl] = np.nan
        df_new.iloc[5:6, sl] = "bar"

        df_new["string2"] = "foo"
        sl = df_new.columns.get_loc("string2")
        df_new.iloc[2:5, sl] = np.nan
        df_new.iloc[7:8, sl] = "bar"
        _maybe_remove(store, "df")
        store.append("df", df_new, data_columns=["A", "B", "string", "string2"])
        result = store.select("df", "string='foo' and string2='foo' and A>0 and B<0")
        expected = df_new[
            (df_new.string == "foo")
            & (df_new.string2 == "foo")
            & (df_new.A > 0)
            & (df_new.B < 0)
        ]
        tm.assert_frame_equal(result, expected, check_freq=False)
        # FIXME: 2020-05-07 freq check randomly fails in the CI

        # yield an empty frame
        result = store.select("df", "string='foo' and string2='cool'")
        expected = df_new[(df_new.string == "foo") & (df_new.string2 == "cool")]
        tm.assert_frame_equal(result, expected)

    with ensure_clean_store(setup_path) as store:
        # doc example
        df_dc = df.copy()
        df_dc["string"] = "foo"
        df_dc.loc[df_dc.index[4:6], "string"] = np.nan
        df_dc.loc[df_dc.index[7:9], "string"] = "bar"
        df_dc["string2"] = "cool"
        df_dc["datetime"] = Timestamp("20010102").as_unit("ns")
        df_dc.loc[df_dc.index[3:5], ["A", "B", "datetime"]] = np.nan

        _maybe_remove(store, "df_dc")
        store.append(
            "df_dc", df_dc, data_columns=["B", "C", "string", "string2", "datetime"]
        )
        result = store.select("df_dc", "B>0")

        expected = df_dc[df_dc.B > 0]
        tm.assert_frame_equal(result, expected)

        result = store.select("df_dc", ["B > 0", "C > 0", "string == foo"])
        expected = df_dc[(df_dc.B > 0) & (df_dc.C > 0) & (df_dc.string == "foo")]
        tm.assert_frame_equal(result, expected, check_freq=False)
        # FIXME: 2020-12-07 intermittent build failures here with freq of
        #  None instead of BDay(4)

    with ensure_clean_store(setup_path) as store:
        # doc example part 2

        index = date_range("1/1/2000", periods=8)
        df_dc = DataFrame(
            np.random.default_rng(2).standard_normal((8, 3)),
            index=index,
            columns=["A", "B", "C"],
        )
        df_dc["string"] = "foo"
        df_dc.loc[df_dc.index[4:6], "string"] = np.nan
        df_dc.loc[df_dc.index[7:9], "string"] = "bar"
        df_dc[["B", "C"]] = df_dc[["B", "C"]].abs()
        df_dc["string2"] = "cool"

        # on-disk operations
        store.append("df_dc", df_dc, data_columns=["B", "C", "string", "string2"])

        result = store.select("df_dc", "B>0")
        expected = df_dc[df_dc.B > 0]
        tm.assert_frame_equal(result, expected)

        result = store.select("df_dc", ["B > 0", "C > 0", 'string == "foo"'])
        expected = df_dc[(df_dc.B > 0) & (df_dc.C > 0) & (df_dc.string == "foo")]
        tm.assert_frame_equal(result, expected)


def test_append_hierarchical(tmp_path, setup_path, multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    df.columns.name = None

    with ensure_clean_store(setup_path) as store:
        store.append("mi", df)
        result = store.select("mi")
        tm.assert_frame_equal(result, df)

        # GH 3748
        result = store.select("mi", columns=["A", "B"])
        expected = df.reindex(columns=["A", "B"])
        tm.assert_frame_equal(result, expected)

    path = tmp_path / "test.hdf"
    df.to_hdf(path, key="df", format="table")
    result = read_hdf(path, "df", columns=["A", "B"])
    expected = df.reindex(columns=["A", "B"])
    tm.assert_frame_equal(result, expected)


def test_append_misc(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )
        store.append("df", df, chunksize=1)
        result = store.select("df")
        tm.assert_frame_equal(result, df)

        store.append("df1", df, expectedrows=10)
        result = store.select("df1")
        tm.assert_frame_equal(result, df)


@pytest.mark.parametrize("chunksize", [10, 200, 1000])
def test_append_misc_chunksize(setup_path, chunksize):
    # more chunksize in append tests
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    df["string"] = "foo"
    df["float322"] = 1.0
    df["float322"] = df["float322"].astype("float32")
    df["bool"] = df["float322"] > 0
    df["time1"] = Timestamp("20130101").as_unit("ns")
    df["time2"] = Timestamp("20130102").as_unit("ns")
    with ensure_clean_store(setup_path, mode="w") as store:
        store.append("obj", df, chunksize=chunksize)
        result = store.select("obj")
        tm.assert_frame_equal(result, df)


def test_append_misc_empty_frame(setup_path):
    # empty frame, GH4273
    with ensure_clean_store(setup_path) as store:
        # 0 len
        df_empty = DataFrame(columns=list("ABC"))
        store.append("df", df_empty)
        with pytest.raises(KeyError, match="'No object named df in the file'"):
            store.select("df")

        # repeated append of 0/non-zero frames
        df = DataFrame(np.random.default_rng(2).random((10, 3)), columns=list("ABC"))
        store.append("df", df)
        tm.assert_frame_equal(store.select("df"), df)
        store.append("df", df_empty)
        tm.assert_frame_equal(store.select("df"), df)

        # store
        df = DataFrame(columns=list("ABC"))
        store.put("df2", df)
        tm.assert_frame_equal(store.select("df2"), df)


# TODO(ArrayManager) currently we rely on falling back to BlockManager, but
# the conversion from AM->BM converts the invalid object dtype column into
# a datetime64 column no longer raising an error
@td.skip_array_manager_not_yet_implemented
def test_append_raise(setup_path, using_infer_string):
    with ensure_clean_store(setup_path) as store:
        # test append with invalid input to get good error messages

        # list in column
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )
        df["invalid"] = [["a"]] * len(df)
        assert df.dtypes["invalid"] == np.object_
        msg = re.escape(
            """Cannot serialize the column [invalid]
because its data contents are not [string] but [mixed] object dtype"""
        )
        with pytest.raises(TypeError, match=msg):
            store.append("df", df)

        # multiple invalid columns
        df["invalid2"] = [["a"]] * len(df)
        df["invalid3"] = [["a"]] * len(df)
        with pytest.raises(TypeError, match=msg):
            store.append("df", df)

        # datetime with embedded nans as object
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )
        s = Series(datetime.datetime(2001, 1, 2), index=df.index)
        s = s.astype(object)
        s[0:5] = np.nan
        df["invalid"] = s
        assert df.dtypes["invalid"] == np.object_
        msg = "too many timezones in this block, create separate data columns"
        with pytest.raises(TypeError, match=msg):
            store.append("df", df)

        # directly ndarray
        msg = "value must be None, Series, or DataFrame"
        with pytest.raises(TypeError, match=msg):
            store.append("df", np.arange(10))

        # series directly
        msg = re.escape(
            "cannot properly create the storer for: "
            "[group->df,value-><class 'pandas.core.series.Series'>]"
        )
        with pytest.raises(TypeError, match=msg):
            store.append("df", Series(np.arange(10)))

        # appending an incompatible table
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )
        store.append("df", df)

        df["foo"] = "foo"
        msg = re.escape(
            "invalid combination of [non_index_axes] on appending data "
            "[(1, ['A', 'B', 'C', 'D', 'foo'])] vs current table "
            "[(1, ['A', 'B', 'C', 'D'])]"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)

        # incompatible type (GH 41897)
        _maybe_remove(store, "df")
        df["foo"] = Timestamp("20130101")
        store.append("df", df)
        df["foo"] = "bar"
        msg = re.escape(
            "Cannot serialize the column [foo] "
            "because its data contents are not [string] "
            "but [datetime64[s]] object dtype"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)


def test_append_with_timedelta(setup_path):
    # GH 3577
    # append timedelta

    ts = Timestamp("20130101").as_unit("ns")
    df = DataFrame(
        {
            "A": ts,
            "B": [ts + timedelta(days=i, seconds=10) for i in range(10)],
        }
    )
    df["C"] = df["A"] - df["B"]
    df.loc[3:5, "C"] = np.nan

    with ensure_clean_store(setup_path) as store:
        # table
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=True)
        result = store.select("df")
        tm.assert_frame_equal(result, df)

        result = store.select("df", where="C<100000")
        tm.assert_frame_equal(result, df)

        result = store.select("df", where="C<pd.Timedelta('-3D')")
        tm.assert_frame_equal(result, df.iloc[3:])

        result = store.select("df", "C<'-3D'")
        tm.assert_frame_equal(result, df.iloc[3:])

        # a bit hacky here as we don't really deal with the NaT properly

        result = store.select("df", "C<'-500000s'")
        result = result.dropna(subset=["C"])
        tm.assert_frame_equal(result, df.iloc[6:])

        result = store.select("df", "C<'-3.5D'")
        result = result.iloc[1:]
        tm.assert_frame_equal(result, df.iloc[4:])

        # fixed
        _maybe_remove(store, "df2")
        store.put("df2", df)
        result = store.select("df2")
        tm.assert_frame_equal(result, df)


def test_append_to_multiple(setup_path):
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df2 = df1.copy().rename(columns="{}_2".format)
    df2["foo"] = "bar"
    df = concat([df1, df2], axis=1)

    with ensure_clean_store(setup_path) as store:
        # exceptions
        msg = "append_to_multiple requires a selector that is in passed dict"
        with pytest.raises(ValueError, match=msg):
            store.append_to_multiple(
                {"df1": ["A", "B"], "df2": None}, df, selector="df3"
            )

        with pytest.raises(ValueError, match=msg):
            store.append_to_multiple({"df1": None, "df2": None}, df, selector="df3")

        msg = (
            "append_to_multiple must have a dictionary specified as the way to "
            "split the value"
        )
        with pytest.raises(ValueError, match=msg):
            store.append_to_multiple("df1", df, "df1")

        # regular operation
        store.append_to_multiple({"df1": ["A", "B"], "df2": None}, df, selector="df1")
        result = store.select_as_multiple(
            ["df1", "df2"], where=["A>0", "B>0"], selector="df1"
        )
        expected = df[(df.A > 0) & (df.B > 0)]
        tm.assert_frame_equal(result, expected)


def test_append_to_multiple_dropna(setup_path):
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df2 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    ).rename(columns="{}_2".format)
    df1.iloc[1, df1.columns.get_indexer(["A", "B"])] = np.nan
    df = concat([df1, df2], axis=1)

    with ensure_clean_store(setup_path) as store:
        # dropna=True should guarantee rows are synchronized
        store.append_to_multiple(
            {"df1": ["A", "B"], "df2": None}, df, selector="df1", dropna=True
        )
        result = store.select_as_multiple(["df1", "df2"])
        expected = df.dropna()
        tm.assert_frame_equal(result, expected, check_index_type=True)
        tm.assert_index_equal(store.select("df1").index, store.select("df2").index)


def test_append_to_multiple_dropna_false(setup_path):
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df2 = df1.copy().rename(columns="{}_2".format)
    df1.iloc[1, df1.columns.get_indexer(["A", "B"])] = np.nan
    df = concat([df1, df2], axis=1)

    with ensure_clean_store(setup_path) as store, pd.option_context(
        "io.hdf.dropna_table", True
    ):
        # dropna=False shouldn't synchronize row indexes
        store.append_to_multiple(
            {"df1a": ["A", "B"], "df2a": None}, df, selector="df1a", dropna=False
        )

        msg = "all tables must have exactly the same nrows!"
        with pytest.raises(ValueError, match=msg):
            store.select_as_multiple(["df1a", "df2a"])

        assert not store.select("df1a").index.equals(store.select("df2a").index)


def test_append_to_multiple_min_itemsize(setup_path):
    # GH 11238
    df = DataFrame(
        {
            "IX": np.arange(1, 21),
            "Num": np.arange(1, 21),
            "BigNum": np.arange(1, 21) * 88,
            "Str": ["a" for _ in range(20)],
            "LongStr": ["abcde" for _ in range(20)],
        }
    )
    expected = df.iloc[[0]]

    with ensure_clean_store(setup_path) as store:
        store.append_to_multiple(
            {
                "index": ["IX"],
                "nums": ["Num", "BigNum"],
                "strs": ["Str", "LongStr"],
            },
            df.iloc[[0]],
            "index",
            min_itemsize={"Str": 10, "LongStr": 100, "Num": 2},
        )
        result = store.select_as_multiple(["index", "nums", "strs"])
        tm.assert_frame_equal(result, expected, check_index_type=True)


def test_append_string_nan_rep(setup_path):
    # GH 16300
    df = DataFrame({"A": "a", "B": "foo"}, index=np.arange(10))
    df_nan = df.copy()
    df_nan.loc[0:4, :] = np.nan
    msg = "NaN representation is too large for existing column size"

    with ensure_clean_store(setup_path) as store:
        # string column too small
        store.append("sa", df["A"])
        with pytest.raises(ValueError, match=msg):
            store.append("sa", df_nan["A"])

        # nan_rep too big
        store.append("sb", df["B"], nan_rep="bars")
        with pytest.raises(ValueError, match=msg):
            store.append("sb", df_nan["B"])

        # smaller modified nan_rep
        store.append("sc", df["A"], nan_rep="n")
        store.append("sc", df_nan["A"])
        result = store["sc"]
        expected = concat([df["A"], df_nan["A"]])
        tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\style\test_html.py ===
from textwrap import (
    dedent,
    indent,
)

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    option_context,
)

jinja2 = pytest.importorskip("jinja2")
from pandas.io.formats.style import Styler


@pytest.fixture
def env():
    loader = jinja2.PackageLoader("pandas", "io/formats/templates")
    env = jinja2.Environment(loader=loader, trim_blocks=True)
    return env


@pytest.fixture
def styler():
    return Styler(DataFrame([[2.61], [2.69]], index=["a", "b"], columns=["A"]))


@pytest.fixture
def styler_mi():
    midx = MultiIndex.from_product([["a", "b"], ["c", "d"]])
    return Styler(DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=midx))


@pytest.fixture
def tpl_style(env):
    return env.get_template("html_style.tpl")


@pytest.fixture
def tpl_table(env):
    return env.get_template("html_table.tpl")


def test_html_template_extends_options():
    # make sure if templates are edited tests are updated as are setup fixtures
    # to understand the dependency
    with open("pandas/io/formats/templates/html.tpl", encoding="utf-8") as file:
        result = file.read()
    assert "{% include html_style_tpl %}" in result
    assert "{% include html_table_tpl %}" in result


def test_exclude_styles(styler):
    result = styler.to_html(exclude_styles=True, doctype_html=True)
    expected = dedent(
        """\
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        </head>
        <body>
        <table>
          <thead>
            <tr>
              <th >&nbsp;</th>
              <th >A</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th >a</th>
              <td >2.610000</td>
            </tr>
            <tr>
              <th >b</th>
              <td >2.690000</td>
            </tr>
          </tbody>
        </table>
        </body>
        </html>
        """
    )
    assert result == expected


def test_w3_html_format(styler):
    styler.set_uuid("").set_table_styles([{"selector": "th", "props": "att2:v2;"}]).map(
        lambda x: "att1:v1;"
    ).set_table_attributes('class="my-cls1" style="attr3:v3;"').set_td_classes(
        DataFrame(["my-cls2"], index=["a"], columns=["A"])
    ).format(
        "{:.1f}"
    ).set_caption(
        "A comprehensive test"
    )
    expected = dedent(
        """\
        <style type="text/css">
        #T_ th {
          att2: v2;
        }
        #T__row0_col0, #T__row1_col0 {
          att1: v1;
        }
        </style>
        <table id="T_" class="my-cls1" style="attr3:v3;">
          <caption>A comprehensive test</caption>
          <thead>
            <tr>
              <th class="blank level0" >&nbsp;</th>
              <th id="T__level0_col0" class="col_heading level0 col0" >A</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th id="T__level0_row0" class="row_heading level0 row0" >a</th>
              <td id="T__row0_col0" class="data row0 col0 my-cls2" >2.6</td>
            </tr>
            <tr>
              <th id="T__level0_row1" class="row_heading level0 row1" >b</th>
              <td id="T__row1_col0" class="data row1 col0" >2.7</td>
            </tr>
          </tbody>
        </table>
        """
    )
    assert expected == styler.to_html()


def test_colspan_w3():
    # GH 36223
    df = DataFrame(data=[[1, 2]], columns=[["l0", "l0"], ["l1a", "l1b"]])
    styler = Styler(df, uuid="_", cell_ids=False)
    assert '<th class="col_heading level0 col0" colspan="2">l0</th>' in styler.to_html()


def test_rowspan_w3():
    # GH 38533
    df = DataFrame(data=[[1, 2]], index=[["l0", "l0"], ["l1a", "l1b"]])
    styler = Styler(df, uuid="_", cell_ids=False)
    assert '<th class="row_heading level0 row0" rowspan="2">l0</th>' in styler.to_html()


def test_styles(styler):
    styler.set_uuid("abc")
    styler.set_table_styles([{"selector": "td", "props": "color: red;"}])
    result = styler.to_html(doctype_html=True)
    expected = dedent(
        """\
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style type="text/css">
        #T_abc td {
          color: red;
        }
        </style>
        </head>
        <body>
        <table id="T_abc">
          <thead>
            <tr>
              <th class="blank level0" >&nbsp;</th>
              <th id="T_abc_level0_col0" class="col_heading level0 col0" >A</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th id="T_abc_level0_row0" class="row_heading level0 row0" >a</th>
              <td id="T_abc_row0_col0" class="data row0 col0" >2.610000</td>
            </tr>
            <tr>
              <th id="T_abc_level0_row1" class="row_heading level0 row1" >b</th>
              <td id="T_abc_row1_col0" class="data row1 col0" >2.690000</td>
            </tr>
          </tbody>
        </table>
        </body>
        </html>
        """
    )
    assert result == expected


def test_doctype(styler):
    result = styler.to_html(doctype_html=False)
    assert "<html>" not in result
    assert "<body>" not in result
    assert "<!DOCTYPE html>" not in result
    assert "<head>" not in result


def test_doctype_encoding(styler):
    with option_context("styler.render.encoding", "ASCII"):
        result = styler.to_html(doctype_html=True)
        assert '<meta charset="ASCII">' in result
        result = styler.to_html(doctype_html=True, encoding="ANSI")
        assert '<meta charset="ANSI">' in result


def test_bold_headers_arg(styler):
    result = styler.to_html(bold_headers=True)
    assert "th {\n  font-weight: bold;\n}" in result
    result = styler.to_html()
    assert "th {\n  font-weight: bold;\n}" not in result


def test_caption_arg(styler):
    result = styler.to_html(caption="foo bar")
    assert "<caption>foo bar</caption>" in result
    result = styler.to_html()
    assert "<caption>foo bar</caption>" not in result


def test_block_names(tpl_style, tpl_table):
    # catch accidental removal of a block
    expected_style = {
        "before_style",
        "style",
        "table_styles",
        "before_cellstyle",
        "cellstyle",
    }
    expected_table = {
        "before_table",
        "table",
        "caption",
        "thead",
        "tbody",
        "after_table",
        "before_head_rows",
        "head_tr",
        "after_head_rows",
        "before_rows",
        "tr",
        "after_rows",
    }
    result1 = set(tpl_style.blocks)
    assert result1 == expected_style

    result2 = set(tpl_table.blocks)
    assert result2 == expected_table


def test_from_custom_template_table(tmpdir):
    p = tmpdir.mkdir("tpl").join("myhtml_table.tpl")
    p.write(
        dedent(
            """\
            {% extends "html_table.tpl" %}
            {% block table %}
            <h1>{{custom_title}}</h1>
            {{ super() }}
            {% endblock table %}"""
        )
    )
    result = Styler.from_custom_template(str(tmpdir.join("tpl")), "myhtml_table.tpl")
    assert issubclass(result, Styler)
    assert result.env is not Styler.env
    assert result.template_html_table is not Styler.template_html_table
    styler = result(DataFrame({"A": [1, 2]}))
    assert "<h1>My Title</h1>\n\n\n<table" in styler.to_html(custom_title="My Title")


def test_from_custom_template_style(tmpdir):
    p = tmpdir.mkdir("tpl").join("myhtml_style.tpl")
    p.write(
        dedent(
            """\
            {% extends "html_style.tpl" %}
            {% block style %}
            <link rel="stylesheet" href="mystyle.css">
            {{ super() }}
            {% endblock style %}"""
        )
    )
    result = Styler.from_custom_template(
        str(tmpdir.join("tpl")), html_style="myhtml_style.tpl"
    )
    assert issubclass(result, Styler)
    assert result.env is not Styler.env
    assert result.template_html_style is not Styler.template_html_style
    styler = result(DataFrame({"A": [1, 2]}))
    assert '<link rel="stylesheet" href="mystyle.css">\n\n<style' in styler.to_html()


def test_caption_as_sequence(styler):
    styler.set_caption(("full cap", "short cap"))
    assert "<caption>full cap</caption>" in styler.to_html()


@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("index_name", [True, False])
def test_sticky_basic(styler, index, columns, index_name):
    if index_name:
        styler.index.name = "some text"
    if index:
        styler.set_sticky(axis=0)
    if columns:
        styler.set_sticky(axis=1)

    left_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  left: 0px;\n  z-index: {1};\n}}"
    )
    top_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  top: {1}px;\n  z-index: {2};\n{3}}}"
    )

    res = styler.set_uuid("").to_html()

    # test index stickys over thead and tbody
    assert (left_css.format("thead tr th:nth-child(1)", "3 !important") in res) is index
    assert (left_css.format("tbody tr th:nth-child(1)", "1") in res) is index

    # test column stickys including if name row
    assert (
        top_css.format("thead tr:nth-child(1) th", "0", "2", "  height: 25px;\n") in res
    ) is (columns and index_name)
    assert (
        top_css.format("thead tr:nth-child(2) th", "25", "2", "  height: 25px;\n")
        in res
    ) is (columns and index_name)
    assert (top_css.format("thead tr:nth-child(1) th", "0", "2", "") in res) is (
        columns and not index_name
    )


@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("columns", [False, True])
def test_sticky_mi(styler_mi, index, columns):
    if index:
        styler_mi.set_sticky(axis=0)
    if columns:
        styler_mi.set_sticky(axis=1)

    left_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  left: {1}px;\n  min-width: 75px;\n  max-width: 75px;\n  z-index: {2};\n}}"
    )
    top_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  top: {1}px;\n  height: 25px;\n  z-index: {2};\n}}"
    )

    res = styler_mi.set_uuid("").to_html()

    # test the index stickys for thead and tbody over both levels
    assert (
        left_css.format("thead tr th:nth-child(1)", "0", "3 !important") in res
    ) is index
    assert (left_css.format("tbody tr th.level0", "0", "1") in res) is index
    assert (
        left_css.format("thead tr th:nth-child(2)", "75", "3 !important") in res
    ) is index
    assert (left_css.format("tbody tr th.level1", "75", "1") in res) is index

    # test the column stickys for each level row
    assert (top_css.format("thead tr:nth-child(1) th", "0", "2") in res) is columns
    assert (top_css.format("thead tr:nth-child(2) th", "25", "2") in res) is columns


@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("levels", [[1], ["one"], "one"])
def test_sticky_levels(styler_mi, index, columns, levels):
    styler_mi.index.names, styler_mi.columns.names = ["zero", "one"], ["zero", "one"]
    if index:
        styler_mi.set_sticky(axis=0, levels=levels)
    if columns:
        styler_mi.set_sticky(axis=1, levels=levels)

    left_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  left: {1}px;\n  min-width: 75px;\n  max-width: 75px;\n  z-index: {2};\n}}"
    )
    top_css = (
        "#T_ {0} {{\n  position: sticky;\n  background-color: inherit;\n"
        "  top: {1}px;\n  height: 25px;\n  z-index: {2};\n}}"
    )

    res = styler_mi.set_uuid("").to_html()

    # test no sticking of level0
    assert "#T_ thead tr th:nth-child(1)" not in res
    assert "#T_ tbody tr th.level0" not in res
    assert "#T_ thead tr:nth-child(1) th" not in res

    # test sticking level1
    assert (
        left_css.format("thead tr th:nth-child(2)", "0", "3 !important") in res
    ) is index
    assert (left_css.format("tbody tr th.level1", "0", "1") in res) is index
    assert (top_css.format("thead tr:nth-child(2) th", "0", "2") in res) is columns


def test_sticky_raises(styler):
    with pytest.raises(ValueError, match="No axis named bad for object type DataFrame"):
        styler.set_sticky(axis="bad")


@pytest.mark.parametrize(
    "sparse_index, sparse_columns",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_sparse_options(sparse_index, sparse_columns):
    cidx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    ridx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=ridx, columns=cidx)
    styler = df.style

    default_html = styler.to_html()  # defaults under pd.options to (True , True)

    with option_context(
        "styler.sparse.index", sparse_index, "styler.sparse.columns", sparse_columns
    ):
        html1 = styler.to_html()
        assert (html1 == default_html) is (sparse_index and sparse_columns)
    html2 = styler.to_html(sparse_index=sparse_index, sparse_columns=sparse_columns)
    assert html1 == html2


@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("columns", [True, False])
def test_map_header_cell_ids(styler, index, columns):
    # GH 41893
    func = lambda v: "attr: val;"
    styler.uuid, styler.cell_ids = "", False
    if index:
        styler.map_index(func, axis="index")
    if columns:
        styler.map_index(func, axis="columns")

    result = styler.to_html()

    # test no data cell ids
    assert '<td class="data row0 col0" >2.610000</td>' in result
    assert '<td class="data row1 col0" >2.690000</td>' in result

    # test index header ids where needed and css styles
    assert (
        '<th id="T__level0_row0" class="row_heading level0 row0" >a</th>' in result
    ) is index
    assert (
        '<th id="T__level0_row1" class="row_heading level0 row1" >b</th>' in result
    ) is index
    assert ("#T__level0_row0, #T__level0_row1 {\n  attr: val;\n}" in result) is index

    # test column header ids where needed and css styles
    assert (
        '<th id="T__level0_col0" class="col_heading level0 col0" >A</th>' in result
    ) is columns
    assert ("#T__level0_col0 {\n  attr: val;\n}" in result) is columns


@pytest.mark.parametrize("rows", [True, False])
@pytest.mark.parametrize("cols", [True, False])
def test_maximums(styler_mi, rows, cols):
    result = styler_mi.to_html(
        max_rows=2 if rows else None,
        max_columns=2 if cols else None,
    )

    assert ">5</td>" in result  # [[0,1], [4,5]] always visible
    assert (">8</td>" in result) is not rows  # first trimmed vertical element
    assert (">2</td>" in result) is not cols  # first trimmed horizontal element


def test_replaced_css_class_names():
    css = {
        "row_heading": "ROWHEAD",
        # "col_heading": "COLHEAD",
        "index_name": "IDXNAME",
        # "col": "COL",
        "row": "ROW",
        # "col_trim": "COLTRIM",
        "row_trim": "ROWTRIM",
        "level": "LEVEL",
        "data": "DATA",
        "blank": "BLANK",
    }
    midx = MultiIndex.from_product([["a", "b"], ["c", "d"]])
    styler_mi = Styler(
        DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=midx),
        uuid_len=0,
    ).set_table_styles(css_class_names=css)
    styler_mi.index.names = ["n1", "n2"]
    styler_mi.hide(styler_mi.index[1:], axis=0)
    styler_mi.hide(styler_mi.columns[1:], axis=1)
    styler_mi.map_index(lambda v: "color: red;", axis=0)
    styler_mi.map_index(lambda v: "color: green;", axis=1)
    styler_mi.map(lambda v: "color: blue;")
    expected = dedent(
        """\
    <style type="text/css">
    #T__ROW0_col0 {
      color: blue;
    }
    #T__LEVEL0_ROW0, #T__LEVEL1_ROW0 {
      color: red;
    }
    #T__LEVEL0_col0, #T__LEVEL1_col0 {
      color: green;
    }
    </style>
    <table id="T_">
      <thead>
        <tr>
          <th class="BLANK" >&nbsp;</th>
          <th class="IDXNAME LEVEL0" >n1</th>
          <th id="T__LEVEL0_col0" class="col_heading LEVEL0 col0" >a</th>
        </tr>
        <tr>
          <th class="BLANK" >&nbsp;</th>
          <th class="IDXNAME LEVEL1" >n2</th>
          <th id="T__LEVEL1_col0" class="col_heading LEVEL1 col0" >c</th>
        </tr>
        <tr>
          <th class="IDXNAME LEVEL0" >n1</th>
          <th class="IDXNAME LEVEL1" >n2</th>
          <th class="BLANK col0" >&nbsp;</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T__LEVEL0_ROW0" class="ROWHEAD LEVEL0 ROW0" >a</th>
          <th id="T__LEVEL1_ROW0" class="ROWHEAD LEVEL1 ROW0" >c</th>
          <td id="T__ROW0_col0" class="DATA ROW0 col0" >0</td>
        </tr>
      </tbody>
    </table>
    """
    )
    result = styler_mi.to_html()
    assert result == expected


def test_include_css_style_rules_only_for_visible_cells(styler_mi):
    # GH 43619
    result = (
        styler_mi.set_uuid("")
        .map(lambda v: "color: blue;")
        .hide(styler_mi.data.columns[1:], axis="columns")
        .hide(styler_mi.data.index[1:], axis="index")
        .to_html()
    )
    expected_styles = dedent(
        """\
        <style type="text/css">
        #T__row0_col0 {
          color: blue;
        }
        </style>
        """
    )
    assert expected_styles in result


def test_include_css_style_rules_only_for_visible_index_labels(styler_mi):
    # GH 43619
    result = (
        styler_mi.set_uuid("")
        .map_index(lambda v: "color: blue;", axis="index")
        .hide(styler_mi.data.columns, axis="columns")
        .hide(styler_mi.data.index[1:], axis="index")
        .to_html()
    )
    expected_styles = dedent(
        """\
        <style type="text/css">
        #T__level0_row0, #T__level1_row0 {
          color: blue;
        }
        </style>
        """
    )
    assert expected_styles in result


def test_include_css_style_rules_only_for_visible_column_labels(styler_mi):
    # GH 43619
    result = (
        styler_mi.set_uuid("")
        .map_index(lambda v: "color: blue;", axis="columns")
        .hide(styler_mi.data.columns[1:], axis="columns")
        .hide(styler_mi.data.index, axis="index")
        .to_html()
    )
    expected_styles = dedent(
        """\
        <style type="text/css">
        #T__level0_col0, #T__level1_col0 {
          color: blue;
        }
        </style>
        """
    )
    assert expected_styles in result


def test_hiding_index_columns_multiindex_alignment():
    # gh 43644
    midx = MultiIndex.from_product(
        [["i0", "j0"], ["i1"], ["i2", "j2"]], names=["i-0", "i-1", "i-2"]
    )
    cidx = MultiIndex.from_product(
        [["c0"], ["c1", "d1"], ["c2", "d2"]], names=["c-0", "c-1", "c-2"]
    )
    df = DataFrame(np.arange(16).reshape(4, 4), index=midx, columns=cidx)
    styler = Styler(df, uuid_len=0)
    styler.hide(level=1, axis=0).hide(level=0, axis=1)
    styler.hide([("j0", "i1", "j2")], axis=0)
    styler.hide([("c0", "d1", "d2")], axis=1)
    result = styler.to_html()
    expected = dedent(
        """\
    <style type="text/css">
    </style>
    <table id="T_">
      <thead>
        <tr>
          <th class="blank" >&nbsp;</th>
          <th class="index_name level1" >c-1</th>
          <th id="T__level1_col0" class="col_heading level1 col0" colspan="2">c1</th>
          <th id="T__level1_col2" class="col_heading level1 col2" >d1</th>
        </tr>
        <tr>
          <th class="blank" >&nbsp;</th>
          <th class="index_name level2" >c-2</th>
          <th id="T__level2_col0" class="col_heading level2 col0" >c2</th>
          <th id="T__level2_col1" class="col_heading level2 col1" >d2</th>
          <th id="T__level2_col2" class="col_heading level2 col2" >c2</th>
        </tr>
        <tr>
          <th class="index_name level0" >i-0</th>
          <th class="index_name level2" >i-2</th>
          <th class="blank col0" >&nbsp;</th>
          <th class="blank col1" >&nbsp;</th>
          <th class="blank col2" >&nbsp;</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T__level0_row0" class="row_heading level0 row0" rowspan="2">i0</th>
          <th id="T__level2_row0" class="row_heading level2 row0" >i2</th>
          <td id="T__row0_col0" class="data row0 col0" >0</td>
          <td id="T__row0_col1" class="data row0 col1" >1</td>
          <td id="T__row0_col2" class="data row0 col2" >2</td>
        </tr>
        <tr>
          <th id="T__level2_row1" class="row_heading level2 row1" >j2</th>
          <td id="T__row1_col0" class="data row1 col0" >4</td>
          <td id="T__row1_col1" class="data row1 col1" >5</td>
          <td id="T__row1_col2" class="data row1 col2" >6</td>
        </tr>
        <tr>
          <th id="T__level0_row2" class="row_heading level0 row2" >j0</th>
          <th id="T__level2_row2" class="row_heading level2 row2" >i2</th>
          <td id="T__row2_col0" class="data row2 col0" >8</td>
          <td id="T__row2_col1" class="data row2 col1" >9</td>
          <td id="T__row2_col2" class="data row2 col2" >10</td>
        </tr>
      </tbody>
    </table>
    """
    )
    assert result == expected


def test_hiding_index_columns_multiindex_trimming():
    # gh 44272
    df = DataFrame(np.arange(64).reshape(8, 8))
    df.columns = MultiIndex.from_product([[0, 1, 2, 3], [0, 1]])
    df.index = MultiIndex.from_product([[0, 1, 2, 3], [0, 1]])
    df.index.names, df.columns.names = ["a", "b"], ["c", "d"]
    styler = Styler(df, cell_ids=False, uuid_len=0)
    styler.hide([(0, 0), (0, 1), (1, 0)], axis=1).hide([(0, 0), (0, 1), (1, 0)], axis=0)
    with option_context("styler.render.max_rows", 4, "styler.render.max_columns", 4):
        result = styler.to_html()

    expected = dedent(
        """\
    <style type="text/css">
    </style>
    <table id="T_">
      <thead>
        <tr>
          <th class="blank" >&nbsp;</th>
          <th class="index_name level0" >c</th>
          <th class="col_heading level0 col3" >1</th>
          <th class="col_heading level0 col4" colspan="2">2</th>
          <th class="col_heading level0 col6" >3</th>
        </tr>
        <tr>
          <th class="blank" >&nbsp;</th>
          <th class="index_name level1" >d</th>
          <th class="col_heading level1 col3" >1</th>
          <th class="col_heading level1 col4" >0</th>
          <th class="col_heading level1 col5" >1</th>
          <th class="col_heading level1 col6" >0</th>
          <th class="col_heading level1 col_trim" >...</th>
        </tr>
        <tr>
          <th class="index_name level0" >a</th>
          <th class="index_name level1" >b</th>
          <th class="blank col3" >&nbsp;</th>
          <th class="blank col4" >&nbsp;</th>
          <th class="blank col5" >&nbsp;</th>
          <th class="blank col6" >&nbsp;</th>
          <th class="blank col7 col_trim" >&nbsp;</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th class="row_heading level0 row3" >1</th>
          <th class="row_heading level1 row3" >1</th>
          <td class="data row3 col3" >27</td>
          <td class="data row3 col4" >28</td>
          <td class="data row3 col5" >29</td>
          <td class="data row3 col6" >30</td>
          <td class="data row3 col_trim" >...</td>
        </tr>
        <tr>
          <th class="row_heading level0 row4" rowspan="2">2</th>
          <th class="row_heading level1 row4" >0</th>
          <td class="data row4 col3" >35</td>
          <td class="data row4 col4" >36</td>
          <td class="data row4 col5" >37</td>
          <td class="data row4 col6" >38</td>
          <td class="data row4 col_trim" >...</td>
        </tr>
        <tr>
          <th class="row_heading level1 row5" >1</th>
          <td class="data row5 col3" >43</td>
          <td class="data row5 col4" >44</td>
          <td class="data row5 col5" >45</td>
          <td class="data row5 col6" >46</td>
          <td class="data row5 col_trim" >...</td>
        </tr>
        <tr>
          <th class="row_heading level0 row6" >3</th>
          <th class="row_heading level1 row6" >0</th>
          <td class="data row6 col3" >51</td>
          <td class="data row6 col4" >52</td>
          <td class="data row6 col5" >53</td>
          <td class="data row6 col6" >54</td>
          <td class="data row6 col_trim" >...</td>
        </tr>
        <tr>
          <th class="row_heading level0 row_trim" >...</th>
          <th class="row_heading level1 row_trim" >...</th>
          <td class="data col3 row_trim" >...</td>
          <td class="data col4 row_trim" >...</td>
          <td class="data col5 row_trim" >...</td>
          <td class="data col6 row_trim" >...</td>
          <td class="data row_trim col_trim" >...</td>
        </tr>
      </tbody>
    </table>
    """
    )

    assert result == expected


@pytest.mark.parametrize("type", ["data", "index"])
@pytest.mark.parametrize(
    "text, exp, found",
    [
        ("no link, just text", False, ""),
        ("subdomain not www: sub.web.com", False, ""),
        ("www subdomain: www.web.com other", True, "www.web.com"),
        ("scheme full structure: http://www.web.com", True, "http://www.web.com"),
        ("scheme no top-level: http://www.web", True, "http://www.web"),
        ("no scheme, no top-level: www.web", False, "www.web"),
        ("https scheme: https://www.web.com", True, "https://www.web.com"),
        ("ftp scheme: ftp://www.web", True, "ftp://www.web"),
        ("ftps scheme: ftps://www.web", True, "ftps://www.web"),
        ("subdirectories: www.web.com/directory", True, "www.web.com/directory"),
        ("Multiple domains: www.1.2.3.4", True, "www.1.2.3.4"),
        ("with port: http://web.com:80", True, "http://web.com:80"),
        (
            "full net_loc scheme: http://user:pass@web.com",
            True,
            "http://user:pass@web.com",
        ),
        (
            "with valid special chars: http://web.com/,.':;~!@#$*()[]",
            True,
            "http://web.com/,.':;~!@#$*()[]",
        ),
    ],
)
def test_rendered_links(type, text, exp, found):
    if type == "data":
        df = DataFrame([text])
        styler = df.style.format(hyperlinks="html")
    else:
        df = DataFrame([0], index=[text])
        styler = df.style.format_index(hyperlinks="html")

    rendered = f'<a href="{found}" target="_blank">{found}</a>'
    result = styler.to_html()
    assert (rendered in result) is exp
    assert (text in result) is not exp  # test conversion done when expected and not


def test_multiple_rendered_links():
    links = ("www.a.b", "http://a.c", "https://a.d", "ftp://a.e")
    # pylint: disable-next=consider-using-f-string
    df = DataFrame(["text {} {} text {} {}".format(*links)])
    result = df.style.format(hyperlinks="html").to_html()
    href = '<a href="{0}" target="_blank">{0}</a>'
    for link in links:
        assert href.format(link) in result
    assert href.format("text") not in result


def test_concat(styler):
    other = styler.data.agg(["mean"]).style
    styler.concat(other).set_uuid("X")
    result = styler.to_html()
    fp = "foot0_"
    expected = dedent(
        f"""\
    <tr>
      <th id="T_X_level0_row1" class="row_heading level0 row1" >b</th>
      <td id="T_X_row1_col0" class="data row1 col0" >2.690000</td>
    </tr>
    <tr>
      <th id="T_X_level0_{fp}row0" class="{fp}row_heading level0 {fp}row0" >mean</th>
      <td id="T_X_{fp}row0_col0" class="{fp}data {fp}row0 col0" >2.650000</td>
    </tr>
  </tbody>
</table>
    """
    )
    assert expected in result


def test_concat_recursion(styler):
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(["mean"]), precision=3)
    styler3 = Styler(df.agg(["mean"]), precision=4)
    styler1.concat(styler2.concat(styler3)).set_uuid("X")
    result = styler.to_html()
    # notice that the second concat (last <tr> of the output html),
    # there are two `foot_` in the id and class
    fp1 = "foot0_"
    fp2 = "foot0_foot0_"
    expected = dedent(
        f"""\
    <tr>
      <th id="T_X_level0_row1" class="row_heading level0 row1" >b</th>
      <td id="T_X_row1_col0" class="data row1 col0" >2.690000</td>
    </tr>
    <tr>
      <th id="T_X_level0_{fp1}row0" class="{fp1}row_heading level0 {fp1}row0" >mean</th>
      <td id="T_X_{fp1}row0_col0" class="{fp1}data {fp1}row0 col0" >2.650</td>
    </tr>
    <tr>
      <th id="T_X_level0_{fp2}row0" class="{fp2}row_heading level0 {fp2}row0" >mean</th>
      <td id="T_X_{fp2}row0_col0" class="{fp2}data {fp2}row0 col0" >2.6500</td>
    </tr>
  </tbody>
</table>
    """
    )
    assert expected in result


def test_concat_chain(styler):
    df = styler.data
    styler1 = styler
    styler2 = Styler(df.agg(["mean"]), precision=3)
    styler3 = Styler(df.agg(["mean"]), precision=4)
    styler1.concat(styler2).concat(styler3).set_uuid("X")
    result = styler.to_html()
    fp1 = "foot0_"
    fp2 = "foot1_"
    expected = dedent(
        f"""\
    <tr>
      <th id="T_X_level0_row1" class="row_heading level0 row1" >b</th>
      <td id="T_X_row1_col0" class="data row1 col0" >2.690000</td>
    </tr>
    <tr>
      <th id="T_X_level0_{fp1}row0" class="{fp1}row_heading level0 {fp1}row0" >mean</th>
      <td id="T_X_{fp1}row0_col0" class="{fp1}data {fp1}row0 col0" >2.650</td>
    </tr>
    <tr>
      <th id="T_X_level0_{fp2}row0" class="{fp2}row_heading level0 {fp2}row0" >mean</th>
      <td id="T_X_{fp2}row0_col0" class="{fp2}data {fp2}row0 col0" >2.6500</td>
    </tr>
  </tbody>
</table>
    """
    )
    assert expected in result


def test_concat_combined():
    def html_lines(foot_prefix: str):
        assert foot_prefix.endswith("_") or foot_prefix == ""
        fp = foot_prefix
        return indent(
            dedent(
                f"""\
        <tr>
          <th id="T_X_level0_{fp}row0" class="{fp}row_heading level0 {fp}row0" >a</th>
          <td id="T_X_{fp}row0_col0" class="{fp}data {fp}row0 col0" >2.610000</td>
        </tr>
        <tr>
          <th id="T_X_level0_{fp}row1" class="{fp}row_heading level0 {fp}row1" >b</th>
          <td id="T_X_{fp}row1_col0" class="{fp}data {fp}row1 col0" >2.690000</td>
        </tr>
        """
            ),
            prefix=" " * 4,
        )

    df = DataFrame([[2.61], [2.69]], index=["a", "b"], columns=["A"])
    s1 = df.style.highlight_max(color="red")
    s2 = df.style.highlight_max(color="green")
    s3 = df.style.highlight_max(color="blue")
    s4 = df.style.highlight_max(color="yellow")

    result = s1.concat(s2).concat(s3.concat(s4)).set_uuid("X").to_html()
    expected_css = dedent(
        """\
        <style type="text/css">
        #T_X_row1_col0 {
          background-color: red;
        }
        #T_X_foot0_row1_col0 {
          background-color: green;
        }
        #T_X_foot1_row1_col0 {
          background-color: blue;
        }
        #T_X_foot1_foot0_row1_col0 {
          background-color: yellow;
        }
        </style>
        """
    )
    expected_table = (
        dedent(
            """\
            <table id="T_X">
              <thead>
                <tr>
                  <th class="blank level0" >&nbsp;</th>
                  <th id="T_X_level0_col0" class="col_heading level0 col0" >A</th>
                </tr>
              </thead>
              <tbody>
            """
        )
        + html_lines("")
        + html_lines("foot0_")
        + html_lines("foot1_")
        + html_lines("foot1_foot0_")
        + dedent(
            """\
              </tbody>
            </table>
            """
        )
    )
    assert expected_css + expected_table == result


def test_to_html_na_rep_non_scalar_data(datapath):
    # GH47103
    df = DataFrame([{"a": 1, "b": [1, 2, 3], "c": np.nan}])
    result = df.style.format(na_rep="-").to_html(table_uuid="test")
    expected = """\
<style type="text/css">
</style>
<table id="T_test">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_test_level0_col0" class="col_heading level0 col0" >a</th>
      <th id="T_test_level0_col1" class="col_heading level0 col1" >b</th>
      <th id="T_test_level0_col2" class="col_heading level0 col2" >c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_test_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_test_row0_col0" class="data row0 col0" >1</td>
      <td id="T_test_row0_col1" class="data row0 col1" >[1, 2, 3]</td>
      <td id="T_test_row0_col2" class="data row0 col2" >-</td>
    </tr>
  </tbody>
</table>
"""
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_densearith.py ===
"""Tests for dense recursive polynomials' arithmetics. """

from sympy.external.gmpy import GROUND_TYPES

from sympy.polys.densebasic import (
    dup_normal, dmp_normal,
)

from sympy.polys.densearith import (
    dup_add_term, dmp_add_term,
    dup_sub_term, dmp_sub_term,
    dup_mul_term, dmp_mul_term,
    dup_add_ground, dmp_add_ground,
    dup_sub_ground, dmp_sub_ground,
    dup_mul_ground, dmp_mul_ground,
    dup_quo_ground, dmp_quo_ground,
    dup_exquo_ground, dmp_exquo_ground,
    dup_lshift, dup_rshift,
    dup_abs, dmp_abs,
    dup_neg, dmp_neg,
    dup_add, dmp_add,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_sqr, dmp_sqr,
    dup_pow, dmp_pow,
    dup_add_mul, dmp_add_mul,
    dup_sub_mul, dmp_sub_mul,
    dup_pdiv, dup_prem, dup_pquo, dup_pexquo,
    dmp_pdiv, dmp_prem, dmp_pquo, dmp_pexquo,
    dup_rr_div, dmp_rr_div,
    dup_ff_div, dmp_ff_div,
    dup_div, dup_rem, dup_quo, dup_exquo,
    dmp_div, dmp_rem, dmp_quo, dmp_exquo,
    dup_max_norm, dmp_max_norm,
    dup_l1_norm, dmp_l1_norm,
    dup_l2_norm_squared, dmp_l2_norm_squared,
    dup_expand, dmp_expand,
)

from sympy.polys.polyerrors import (
    ExactQuotientFailed,
)

from sympy.polys.specialpolys import f_polys, Symbol, Poly
from sympy.polys.domains import FF, ZZ, QQ, CC

from sympy.testing.pytest import raises

x = Symbol('x')

f_0, f_1, f_2, f_3, f_4, f_5, f_6 = [ f.to_dense() for f in f_polys() ]
F_0 = dmp_mul_ground(dmp_normal(f_0, 2, QQ), QQ(1, 7), 2, QQ)

def test_dup_add_term():
    f = dup_normal([], ZZ)

    assert dup_add_term(f, ZZ(0), 0, ZZ) == dup_normal([], ZZ)

    assert dup_add_term(f, ZZ(1), 0, ZZ) == dup_normal([1], ZZ)
    assert dup_add_term(f, ZZ(1), 1, ZZ) == dup_normal([1, 0], ZZ)
    assert dup_add_term(f, ZZ(1), 2, ZZ) == dup_normal([1, 0, 0], ZZ)

    f = dup_normal([1, 1, 1], ZZ)

    assert dup_add_term(f, ZZ(1), 0, ZZ) == dup_normal([1, 1, 2], ZZ)
    assert dup_add_term(f, ZZ(1), 1, ZZ) == dup_normal([1, 2, 1], ZZ)
    assert dup_add_term(f, ZZ(1), 2, ZZ) == dup_normal([2, 1, 1], ZZ)

    assert dup_add_term(f, ZZ(1), 3, ZZ) == dup_normal([1, 1, 1, 1], ZZ)
    assert dup_add_term(f, ZZ(1), 4, ZZ) == dup_normal([1, 0, 1, 1, 1], ZZ)
    assert dup_add_term(f, ZZ(1), 5, ZZ) == dup_normal([1, 0, 0, 1, 1, 1], ZZ)
    assert dup_add_term(
        f, ZZ(1), 6, ZZ) == dup_normal([1, 0, 0, 0, 1, 1, 1], ZZ)

    assert dup_add_term(f, ZZ(-1), 2, ZZ) == dup_normal([1, 1], ZZ)


def test_dmp_add_term():
    assert dmp_add_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, 0, ZZ) == \
        dup_add_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, ZZ)
    assert dmp_add_term(f_0, [[]], 3, 2, ZZ) == f_0
    assert dmp_add_term(F_0, [[]], 3, 2, QQ) == F_0


def test_dup_sub_term():
    f = dup_normal([], ZZ)

    assert dup_sub_term(f, ZZ(0), 0, ZZ) == dup_normal([], ZZ)

    assert dup_sub_term(f, ZZ(1), 0, ZZ) == dup_normal([-1], ZZ)
    assert dup_sub_term(f, ZZ(1), 1, ZZ) == dup_normal([-1, 0], ZZ)
    assert dup_sub_term(f, ZZ(1), 2, ZZ) == dup_normal([-1, 0, 0], ZZ)

    f = dup_normal([1, 1, 1], ZZ)

    assert dup_sub_term(f, ZZ(2), 0, ZZ) == dup_normal([ 1, 1, -1], ZZ)
    assert dup_sub_term(f, ZZ(2), 1, ZZ) == dup_normal([ 1, -1, 1], ZZ)
    assert dup_sub_term(f, ZZ(2), 2, ZZ) == dup_normal([-1, 1, 1], ZZ)

    assert dup_sub_term(f, ZZ(1), 3, ZZ) == dup_normal([-1, 1, 1, 1], ZZ)
    assert dup_sub_term(f, ZZ(1), 4, ZZ) == dup_normal([-1, 0, 1, 1, 1], ZZ)
    assert dup_sub_term(f, ZZ(1), 5, ZZ) == dup_normal([-1, 0, 0, 1, 1, 1], ZZ)
    assert dup_sub_term(
        f, ZZ(1), 6, ZZ) == dup_normal([-1, 0, 0, 0, 1, 1, 1], ZZ)

    assert dup_sub_term(f, ZZ(1), 2, ZZ) == dup_normal([1, 1], ZZ)


def test_dmp_sub_term():
    assert dmp_sub_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, 0, ZZ) == \
        dup_sub_term([ZZ(1), ZZ(1), ZZ(1)], ZZ(1), 2, ZZ)
    assert dmp_sub_term(f_0, [[]], 3, 2, ZZ) == f_0
    assert dmp_sub_term(F_0, [[]], 3, 2, QQ) == F_0


def test_dup_mul_term():
    f = dup_normal([], ZZ)

    assert dup_mul_term(f, ZZ(2), 3, ZZ) == dup_normal([], ZZ)

    f = dup_normal([1, 1], ZZ)

    assert dup_mul_term(f, ZZ(0), 3, ZZ) == dup_normal([], ZZ)

    f = dup_normal([1, 2, 3], ZZ)

    assert dup_mul_term(f, ZZ(2), 0, ZZ) == dup_normal([2, 4, 6], ZZ)
    assert dup_mul_term(f, ZZ(2), 1, ZZ) == dup_normal([2, 4, 6, 0], ZZ)
    assert dup_mul_term(f, ZZ(2), 2, ZZ) == dup_normal([2, 4, 6, 0, 0], ZZ)
    assert dup_mul_term(f, ZZ(2), 3, ZZ) == dup_normal([2, 4, 6, 0, 0, 0], ZZ)


def test_dmp_mul_term():
    assert dmp_mul_term([ZZ(1), ZZ(2), ZZ(3)], ZZ(2), 1, 0, ZZ) == \
        dup_mul_term([ZZ(1), ZZ(2), ZZ(3)], ZZ(2), 1, ZZ)

    assert dmp_mul_term([[]], [ZZ(2)], 3, 1, ZZ) == [[]]
    assert dmp_mul_term([[ZZ(1)]], [], 3, 1, ZZ) == [[]]

    assert dmp_mul_term([[ZZ(1), ZZ(2)], [ZZ(3)]], [ZZ(2)], 2, 1, ZZ) == \
        [[ZZ(2), ZZ(4)], [ZZ(6)], [], []]

    assert dmp_mul_term([[]], [QQ(2, 3)], 3, 1, QQ) == [[]]
    assert dmp_mul_term([[QQ(1, 2)]], [], 3, 1, QQ) == [[]]

    assert dmp_mul_term([[QQ(1, 5), QQ(2, 5)], [QQ(3, 5)]], [QQ(2, 3)], 2, 1, QQ) == \
        [[QQ(2, 15), QQ(4, 15)], [QQ(6, 15)], [], []]


def test_dup_add_ground():
    f = ZZ.map([1, 2, 3, 4])
    g = ZZ.map([1, 2, 3, 8])

    assert dup_add_ground(f, ZZ(4), ZZ) == g


def test_dmp_add_ground():
    f = ZZ.map([[1], [2], [3], [4]])
    g = ZZ.map([[1], [2], [3], [8]])

    assert dmp_add_ground(f, ZZ(4), 1, ZZ) == g


def test_dup_sub_ground():
    f = ZZ.map([1, 2, 3, 4])
    g = ZZ.map([1, 2, 3, 0])

    assert dup_sub_ground(f, ZZ(4), ZZ) == g


def test_dmp_sub_ground():
    f = ZZ.map([[1], [2], [3], [4]])
    g = ZZ.map([[1], [2], [3], []])

    assert dmp_sub_ground(f, ZZ(4), 1, ZZ) == g


def test_dup_mul_ground():
    f = dup_normal([], ZZ)

    assert dup_mul_ground(f, ZZ(2), ZZ) == dup_normal([], ZZ)

    f = dup_normal([1, 2, 3], ZZ)

    assert dup_mul_ground(f, ZZ(0), ZZ) == dup_normal([], ZZ)
    assert dup_mul_ground(f, ZZ(2), ZZ) == dup_normal([2, 4, 6], ZZ)


def test_dmp_mul_ground():
    assert dmp_mul_ground(f_0, ZZ(2), 2, ZZ) == [
        [[ZZ(2), ZZ(4), ZZ(6)], [ZZ(4)]],
        [[ZZ(6)]],
        [[ZZ(8), ZZ(10), ZZ(12)], [ZZ(2), ZZ(4), ZZ(2)], [ZZ(2)]]
    ]

    assert dmp_mul_ground(F_0, QQ(1, 2), 2, QQ) == [
        [[QQ(1, 14), QQ(2, 14), QQ(3, 14)], [QQ(2, 14)]],
        [[QQ(3, 14)]],
        [[QQ(4, 14), QQ(5, 14), QQ(6, 14)], [QQ(1, 14), QQ(2, 14),
             QQ(1, 14)], [QQ(1, 14)]]
    ]


def test_dup_quo_ground():
    raises(ZeroDivisionError, lambda: dup_quo_ground(dup_normal([1, 2,
           3], ZZ), ZZ(0), ZZ))

    f = dup_normal([], ZZ)

    assert dup_quo_ground(f, ZZ(3), ZZ) == dup_normal([], ZZ)

    f = dup_normal([6, 2, 8], ZZ)

    assert dup_quo_ground(f, ZZ(1), ZZ) == f
    assert dup_quo_ground(f, ZZ(2), ZZ) == dup_normal([3, 1, 4], ZZ)

    assert dup_quo_ground(f, ZZ(3), ZZ) == dup_normal([2, 0, 2], ZZ)

    f = dup_normal([6, 2, 8], QQ)

    assert dup_quo_ground(f, QQ(1), QQ) == f
    assert dup_quo_ground(f, QQ(2), QQ) == [QQ(3), QQ(1), QQ(4)]
    assert dup_quo_ground(f, QQ(7), QQ) == [QQ(6, 7), QQ(2, 7), QQ(8, 7)]


def test_dup_exquo_ground():
    raises(ZeroDivisionError, lambda: dup_exquo_ground(dup_normal([1,
           2, 3], ZZ), ZZ(0), ZZ))
    raises(ExactQuotientFailed, lambda: dup_exquo_ground(dup_normal([1,
           2, 3], ZZ), ZZ(3), ZZ))

    f = dup_normal([], ZZ)

    assert dup_exquo_ground(f, ZZ(3), ZZ) == dup_normal([], ZZ)

    f = dup_normal([6, 2, 8], ZZ)

    assert dup_exquo_ground(f, ZZ(1), ZZ) == f
    assert dup_exquo_ground(f, ZZ(2), ZZ) == dup_normal([3, 1, 4], ZZ)

    f = dup_normal([6, 2, 8], QQ)

    assert dup_exquo_ground(f, QQ(1), QQ) == f
    assert dup_exquo_ground(f, QQ(2), QQ) == [QQ(3), QQ(1), QQ(4)]
    assert dup_exquo_ground(f, QQ(7), QQ) == [QQ(6, 7), QQ(2, 7), QQ(8, 7)]


def test_dmp_quo_ground():
    f = dmp_normal([[6], [2], [8]], 1, ZZ)

    assert dmp_quo_ground(f, ZZ(1), 1, ZZ) == f
    assert dmp_quo_ground(
        f, ZZ(2), 1, ZZ) == dmp_normal([[3], [1], [4]], 1, ZZ)

    assert dmp_normal(dmp_quo_ground(
        f, ZZ(3), 1, ZZ), 1, ZZ) == dmp_normal([[2], [], [2]], 1, ZZ)


def test_dmp_exquo_ground():
    f = dmp_normal([[6], [2], [8]], 1, ZZ)

    assert dmp_exquo_ground(f, ZZ(1), 1, ZZ) == f
    assert dmp_exquo_ground(
        f, ZZ(2), 1, ZZ) == dmp_normal([[3], [1], [4]], 1, ZZ)


def test_dup_lshift():
    assert dup_lshift([], 3, ZZ) == []
    assert dup_lshift([1], 3, ZZ) == [1, 0, 0, 0]


def test_dup_rshift():
    assert dup_rshift([], 3, ZZ) == []
    assert dup_rshift([1, 0, 0, 0], 3, ZZ) == [1]


def test_dup_abs():
    assert dup_abs([], ZZ) == []
    assert dup_abs([ZZ( 1)], ZZ) == [ZZ(1)]
    assert dup_abs([ZZ(-7)], ZZ) == [ZZ(7)]
    assert dup_abs([ZZ(-1), ZZ(2), ZZ(3)], ZZ) == [ZZ(1), ZZ(2), ZZ(3)]

    assert dup_abs([], QQ) == []
    assert dup_abs([QQ( 1, 2)], QQ) == [QQ(1, 2)]
    assert dup_abs([QQ(-7, 3)], QQ) == [QQ(7, 3)]
    assert dup_abs(
        [QQ(-1, 7), QQ(2, 7), QQ(3, 7)], QQ) == [QQ(1, 7), QQ(2, 7), QQ(3, 7)]


def test_dmp_abs():
    assert dmp_abs([ZZ(-1)], 0, ZZ) == [ZZ(1)]
    assert dmp_abs([QQ(-1, 2)], 0, QQ) == [QQ(1, 2)]

    assert dmp_abs([[[]]], 2, ZZ) == [[[]]]
    assert dmp_abs([[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_abs([[[ZZ(-7)]]], 2, ZZ) == [[[ZZ(7)]]]

    assert dmp_abs([[[]]], 2, QQ) == [[[]]]
    assert dmp_abs([[[QQ(1, 2)]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_abs([[[QQ(-7, 9)]]], 2, QQ) == [[[QQ(7, 9)]]]


def test_dup_neg():
    assert dup_neg([], ZZ) == []
    assert dup_neg([ZZ(1)], ZZ) == [ZZ(-1)]
    assert dup_neg([ZZ(-7)], ZZ) == [ZZ(7)]
    assert dup_neg([ZZ(-1), ZZ(2), ZZ(3)], ZZ) == [ZZ(1), ZZ(-2), ZZ(-3)]

    assert dup_neg([], QQ) == []
    assert dup_neg([QQ(1, 2)], QQ) == [QQ(-1, 2)]
    assert dup_neg([QQ(-7, 9)], QQ) == [QQ(7, 9)]
    assert dup_neg([QQ(
        -1, 7), QQ(2, 7), QQ(3, 7)], QQ) == [QQ(1, 7), QQ(-2, 7), QQ(-3, 7)]


def test_dmp_neg():
    assert dmp_neg([ZZ(-1)], 0, ZZ) == [ZZ(1)]
    assert dmp_neg([QQ(-1, 2)], 0, QQ) == [QQ(1, 2)]

    assert dmp_neg([[[]]], 2, ZZ) == [[[]]]
    assert dmp_neg([[[ZZ(1)]]], 2, ZZ) == [[[ZZ(-1)]]]
    assert dmp_neg([[[ZZ(-7)]]], 2, ZZ) == [[[ZZ(7)]]]

    assert dmp_neg([[[]]], 2, QQ) == [[[]]]
    assert dmp_neg([[[QQ(1, 9)]]], 2, QQ) == [[[QQ(-1, 9)]]]
    assert dmp_neg([[[QQ(-7, 9)]]], 2, QQ) == [[[QQ(7, 9)]]]


def test_dup_add():
    assert dup_add([], [], ZZ) == []
    assert dup_add([ZZ(1)], [], ZZ) == [ZZ(1)]
    assert dup_add([], [ZZ(1)], ZZ) == [ZZ(1)]
    assert dup_add([ZZ(1)], [ZZ(1)], ZZ) == [ZZ(2)]
    assert dup_add([ZZ(1)], [ZZ(2)], ZZ) == [ZZ(3)]

    assert dup_add([ZZ(1), ZZ(2)], [ZZ(1)], ZZ) == [ZZ(1), ZZ(3)]
    assert dup_add([ZZ(1)], [ZZ(1), ZZ(2)], ZZ) == [ZZ(1), ZZ(3)]

    assert dup_add([ZZ(1), ZZ(
        2), ZZ(3)], [ZZ(8), ZZ(9), ZZ(10)], ZZ) == [ZZ(9), ZZ(11), ZZ(13)]

    assert dup_add([], [], QQ) == []
    assert dup_add([QQ(1, 2)], [], QQ) == [QQ(1, 2)]
    assert dup_add([], [QQ(1, 2)], QQ) == [QQ(1, 2)]
    assert dup_add([QQ(1, 4)], [QQ(1, 4)], QQ) == [QQ(1, 2)]
    assert dup_add([QQ(1, 4)], [QQ(1, 2)], QQ) == [QQ(3, 4)]

    assert dup_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ) == [QQ(1, 2), QQ(5, 3)]
    assert dup_add([QQ(1)], [QQ(1, 2), QQ(2, 3)], QQ) == [QQ(1, 2), QQ(5, 3)]

    assert dup_add([QQ(1, 7), QQ(2, 7), QQ(3, 7)], [QQ(
        8, 7), QQ(9, 7), QQ(10, 7)], QQ) == [QQ(9, 7), QQ(11, 7), QQ(13, 7)]


def test_dmp_add():
    assert dmp_add([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == \
        dup_add([ZZ(1), ZZ(2)], [ZZ(1)], ZZ)
    assert dmp_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], 0, QQ) == \
        dup_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ)

    assert dmp_add([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_add([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_add([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_add([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(3)]]]
    assert dmp_add([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(3)]]]

    assert dmp_add([[[]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_add([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_add([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_add([[[QQ(2, 7)]]], [[[QQ(1, 7)]]], 2, QQ) == [[[QQ(3, 7)]]]
    assert dmp_add([[[QQ(1, 7)]]], [[[QQ(2, 7)]]], 2, QQ) == [[[QQ(3, 7)]]]


def test_dup_sub():
    assert dup_sub([], [], ZZ) == []
    assert dup_sub([ZZ(1)], [], ZZ) == [ZZ(1)]
    assert dup_sub([], [ZZ(1)], ZZ) == [ZZ(-1)]
    assert dup_sub([ZZ(1)], [ZZ(1)], ZZ) == []
    assert dup_sub([ZZ(1)], [ZZ(2)], ZZ) == [ZZ(-1)]

    assert dup_sub([ZZ(1), ZZ(2)], [ZZ(1)], ZZ) == [ZZ(1), ZZ(1)]
    assert dup_sub([ZZ(1)], [ZZ(1), ZZ(2)], ZZ) == [ZZ(-1), ZZ(-1)]

    assert dup_sub([ZZ(3), ZZ(
        2), ZZ(1)], [ZZ(8), ZZ(9), ZZ(10)], ZZ) == [ZZ(-5), ZZ(-7), ZZ(-9)]

    assert dup_sub([], [], QQ) == []
    assert dup_sub([QQ(1, 2)], [], QQ) == [QQ(1, 2)]
    assert dup_sub([], [QQ(1, 2)], QQ) == [QQ(-1, 2)]
    assert dup_sub([QQ(1, 3)], [QQ(1, 3)], QQ) == []
    assert dup_sub([QQ(1, 3)], [QQ(2, 3)], QQ) == [QQ(-1, 3)]

    assert dup_sub([QQ(1, 7), QQ(2, 7)], [QQ(1)], QQ) == [QQ(1, 7), QQ(-5, 7)]
    assert dup_sub([QQ(1)], [QQ(1, 7), QQ(2, 7)], QQ) == [QQ(-1, 7), QQ(5, 7)]

    assert dup_sub([QQ(3, 7), QQ(2, 7), QQ(1, 7)], [QQ(
        8, 7), QQ(9, 7), QQ(10, 7)], QQ) == [QQ(-5, 7), QQ(-7, 7), QQ(-9, 7)]


def test_dmp_sub():
    assert dmp_sub([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == \
        dup_sub([ZZ(1), ZZ(2)], [ZZ(1)], ZZ)
    assert dmp_sub([QQ(1, 2), QQ(2, 3)], [QQ(1)], 0, QQ) == \
        dup_sub([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ)

    assert dmp_sub([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_sub([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_sub([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(-1)]]]
    assert dmp_sub([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_sub([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(-1)]]]

    assert dmp_sub([[[]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_sub([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_sub([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[QQ(-1, 2)]]]
    assert dmp_sub([[[QQ(2, 7)]]], [[[QQ(1, 7)]]], 2, QQ) == [[[QQ(1, 7)]]]
    assert dmp_sub([[[QQ(1, 7)]]], [[[QQ(2, 7)]]], 2, QQ) == [[[QQ(-1, 7)]]]


def test_dup_add_mul():
    assert dup_add_mul([ZZ(1), ZZ(2), ZZ(3)], [ZZ(3), ZZ(2), ZZ(1)],
               [ZZ(1), ZZ(2)], ZZ) == [ZZ(3), ZZ(9), ZZ(7), ZZ(5)]
    assert dmp_add_mul([[ZZ(1), ZZ(2)], [ZZ(3)]], [[ZZ(3)], [ZZ(2), ZZ(1)]],
               [[ZZ(1)], [ZZ(2)]], 1, ZZ) == [[ZZ(3)], [ZZ(3), ZZ(9)], [ZZ(4), ZZ(5)]]


def test_dup_sub_mul():
    assert dup_sub_mul([ZZ(1), ZZ(2), ZZ(3)], [ZZ(3), ZZ(2), ZZ(1)],
               [ZZ(1), ZZ(2)], ZZ) == [ZZ(-3), ZZ(-7), ZZ(-3), ZZ(1)]
    assert dmp_sub_mul([[ZZ(1), ZZ(2)], [ZZ(3)]], [[ZZ(3)], [ZZ(2), ZZ(1)]],
               [[ZZ(1)], [ZZ(2)]], 1, ZZ) == [[ZZ(-3)], [ZZ(-1), ZZ(-5)], [ZZ(-4), ZZ(1)]]


def test_dup_mul():
    assert dup_mul([], [], ZZ) == []
    assert dup_mul([], [ZZ(1)], ZZ) == []
    assert dup_mul([ZZ(1)], [], ZZ) == []
    assert dup_mul([ZZ(1)], [ZZ(1)], ZZ) == [ZZ(1)]
    assert dup_mul([ZZ(5)], [ZZ(7)], ZZ) == [ZZ(35)]

    assert dup_mul([], [], QQ) == []
    assert dup_mul([], [QQ(1, 2)], QQ) == []
    assert dup_mul([QQ(1, 2)], [], QQ) == []
    assert dup_mul([QQ(1, 2)], [QQ(4, 7)], QQ) == [QQ(2, 7)]
    assert dup_mul([QQ(5, 7)], [QQ(3, 7)], QQ) == [QQ(15, 49)]

    f = dup_normal([3, 0, 0, 6, 1, 2], ZZ)
    g = dup_normal([4, 0, 1, 0], ZZ)
    h = dup_normal([12, 0, 3, 24, 4, 14, 1, 2, 0], ZZ)

    assert dup_mul(f, g, ZZ) == h
    assert dup_mul(g, f, ZZ) == h

    f = dup_normal([2, 0, 0, 1, 7], ZZ)
    h = dup_normal([4, 0, 0, 4, 28, 0, 1, 14, 49], ZZ)

    assert dup_mul(f, f, ZZ) == h

    K = FF(6)

    assert dup_mul([K(2), K(1)], [K(3), K(4)], K) == [K(5), K(4)]

    p1 = dup_normal([79, -1, 78, -94, -10, 11, 32, -19, 78, 2, -89, 30, 73, 42,
        85, 77, 83, -30, -34, -2, 95, -81, 37, -49, -46, -58, -16, 37, 35, -11,
        -57, -15, -31, 67, -20, 27, 76, 2, 70, 67, -65, 65, -26, -93, -44, -12,
        -92, 57, -90, -57, -11, -67, -98, -69, 97, -41, 89, 33, 89, -50, 81,
        -31, 60, -27, 43, 29, -77, 44, 21, -91, 32, -57, 33, 3, 53, -51, -38,
        -99, -84, 23, -50, 66, -100, 1, -75, -25, 27, -60, 98, -51, -87, 6, 8,
        78, -28, -95, -88, 12, -35, 26, -9, 16, -92, 55, -7, -86, 68, -39, -46,
        84, 94, 45, 60, 92, 68, -75, -74, -19, 8, 75, 78, 91, 57, 34, 14, -3,
        -49, 65, 78, -18, 6, -29, -80, -98, 17, 13, 58, 21, 20, 9, 37, 7, -30,
        -53, -20, 34, 67, -42, 89, -22, 73, 43, -6, 5, 51, -8, -15, -52, -22,
        -58, -72, -3, 43, -92, 82, 83, -2, -13, -23, -60, 16, -94, -8, -28,
        -95, -72, 63, -90, 76, 6, -43, -100, -59, 76, 3, 3, 46, -85, 75, 62,
        -71, -76, 88, 97, -72, -1, 30, -64, 72, -48, 14, -78, 58, 63, -91, 24,
        -87, -27, -80, -100, -44, 98, 70, 100, -29, -38, 11, 77, 100, 52, 86,
        65, -5, -42, -81, -38, -42, 43, -2, -70, -63, -52], ZZ)
    p2 = dup_normal([65, -19, -47, 1, 90, 81, -15, -34, 25, -75, 9, -83, 50, -5,
        -44, 31, 1, 70, -7, 78, 74, 80, 85, 65, 21, 41, 66, 19, -40, 63, -21,
        -27, 32, 69, 83, 34, -35, 14, 81, 57, -75, 32, -67, -89, -100, -61, 46,
        84, -78, -29, -50, -94, -24, -32, -68, -16, 100, -7, -72, -89, 35, 82,
        58, 81, -92, 62, 5, -47, -39, -58, -72, -13, 84, 44, 55, -25, 48, -54,
        -31, -56, -11, -50, -84, 10, 67, 17, 13, -14, 61, 76, -64, -44, -40,
        -96, 11, -11, -94, 2, 6, 27, -6, 68, -54, 66, -74, -14, -1, -24, -73,
        96, 89, -11, -89, 56, -53, 72, -43, 96, 25, 63, -31, 29, 68, 83, 91,
        -93, -19, -38, -40, 40, -12, -19, -79, 44, 100, -66, -29, -77, 62, 39,
        -8, 11, -97, 14, 87, 64, 21, -18, 13, 15, -59, -75, -99, -88, 57, 54,
        56, -67, 6, -63, -59, -14, 28, 87, -20, -39, 84, -91, -2, 49, -75, 11,
        -24, -95, 36, 66, 5, 25, -72, -40, 86, 90, 37, -33, 57, -35, 29, -18,
        4, -79, 64, -17, -27, 21, 29, -5, -44, -87, -24, 52, 78, 11, -23, -53,
        36, 42, 21, -68, 94, -91, -51, -21, 51, -76, 72, 31, 24, -48, -80, -9,
        37, -47, -6, -8, -63, -91, 79, -79, -100, 38, -20, 38, 100, 83, -90,
        87, 63, -36, 82, -19, 18, -98, -38, 26, 98, -70, 79, 92, 12, 12, 70,
        74, 36, 48, -13, 31, 31, -47, -71, -12, -64, 36, -42, 32, -86, 60, 83,
        70, 55, 0, 1, 29, -35, 8, -82, 8, -73, -46, -50, 43, 48, -5, -86, -72,
        44, -90, 19, 19, 5, -20, 97, -13, -66, -5, 5, -69, 64, -30, 41, 51, 36,
        13, -99, -61, 94, -12, 74, 98, 68, 24, 46, -97, -87, -6, -27, 82, 62,
        -11, -77, 86, 66, -47, -49, -50, 13, 18, 89, -89, 46, -80, 13, 98, -35,
        -36, -25, 12, 20, 26, -52, 79, 27, 79, 100, 8, 62, -58, -28, 37], ZZ)
    res = dup_normal([5135, -1566, 1376, -7466, 4579, 11710, 8001, -7183,
        -3737, -7439, 345, -10084, 24522, -1201, 1070, -10245, 9582, 9264,
        1903, 23312, 18953, 10037, -15268, -5450, 6442, -6243, -3777, 5110,
        10936, -16649, -6022, 16255, 31300, 24818, 31922, 32760, 7854, 27080,
        15766, 29596, 7139, 31945, -19810, 465, -38026, -3971, 9641, 465,
        -19375, 5524, -30112, -11960, -12813, 13535, 30670, 5925, -43725,
        -14089, 11503, -22782, 6371, 43881, 37465, -33529, -33590, -39798,
        -37854, -18466, -7908, -35825, -26020, -36923, -11332, -5699, 25166,
        -3147, 19885, 12962, -20659, -1642, 27723, -56331, -24580, -11010,
        -20206, 20087, -23772, -16038, 38580, 20901, -50731, 32037, -4299,
        26508, 18038, -28357, 31846, -7405, -20172, -15894, 2096, 25110,
        -45786, 45918, -55333, -31928, -49428, -29824, -58796, -24609, -15408,
        69, -35415, -18439, 10123, -20360, -65949, 33356, -20333, 26476,
        -32073, 33621, 930, 28803, -42791, 44716, 38164, 12302, -1739, 11421,
        73385, -7613, 14297, 38155, -414, 77587, 24338, -21415, 29367, 42639,
        13901, -288, 51027, -11827, 91260, 43407, 88521, -15186, 70572, -12049,
        5090, -12208, -56374, 15520, -623, -7742, 50825, 11199, -14894, 40892,
        59591, -31356, -28696, -57842, -87751, -33744, -28436, -28945, -40287,
        37957, -35638, 33401, -61534, 14870, 40292, 70366, -10803, 102290,
        -71719, -85251, 7902, -22409, 75009, 99927, 35298, -1175, -762, -34744,
        -10587, -47574, -62629, -19581, -43659, -54369, -32250, -39545, 15225,
        -24454, 11241, -67308, -30148, 39929, 37639, 14383, -73475, -77636,
        -81048, -35992, 41601, -90143, 76937, -8112, 56588, 9124, -40094,
        -32340, 13253, 10898, -51639, 36390, 12086, -1885, 100714, -28561,
        -23784, -18735, 18916, 16286, 10742, -87360, -13697, 10689, -19477,
        -29770, 5060, 20189, -8297, 112407, 47071, 47743, 45519, -4109, 17468,
        -68831, 78325, -6481, -21641, -19459, 30919, 96115, 8607, 53341, 32105,
        -16211, 23538, 57259, -76272, -40583, 62093, 38511, -34255, -40665,
        -40604, -37606, -15274, 33156, -13885, 103636, 118678, -14101, -92682,
        -100791, 2634, 63791, 98266, 19286, -34590, -21067, -71130, 25380,
        -40839, -27614, -26060, 52358, -15537, 27138, -6749, 36269, -33306,
        13207, -91084, -5540, -57116, 69548, 44169, -57742, -41234, -103327,
        -62904, -8566, 41149, -12866, 71188, 23980, 1838, 58230, 73950, 5594,
        43113, -8159, -15925, 6911, 85598, -75016, -16214, -62726, -39016,
        8618, -63882, -4299, 23182, 49959, 49342, -3238, -24913, -37138, 78361,
        32451, 6337, -11438, -36241, -37737, 8169, -3077, -24829, 57953, 53016,
        -31511, -91168, 12599, -41849, 41576, 55275, -62539, 47814, -62319,
        12300, -32076, -55137, -84881, -27546, 4312, -3433, -54382, 113288,
        -30157, 74469, 18219, 79880, -2124, 98911, 17655, -33499, -32861,
        47242, -37393, 99765, 14831, -44483, 10800, -31617, -52710, 37406,
        22105, 29704, -20050, 13778, 43683, 36628, 8494, 60964, -22644, 31550,
        -17693, 33805, -124879, -12302, 19343, 20400, -30937, -21574, -34037,
        -33380, 56539, -24993, -75513, -1527, 53563, 65407, -101, 53577, 37991,
        18717, -23795, -8090, -47987, -94717, 41967, 5170, -14815, -94311,
        17896, -17734, -57718, -774, -38410, 24830, 29682, 76480, 58802,
        -46416, -20348, -61353, -68225, -68306, 23822, -31598, 42972, 36327,
        28968, -65638, -21638, 24354, -8356, 26777, 52982, -11783, -44051,
        -26467, -44721, -28435, -53265, -25574, -2669, 44155, 22946, -18454,
        -30718, -11252, 58420, 8711, 67447, 4425, 41749, 67543, 43162, 11793,
        -41907, 20477, -13080, 6559, -6104, -13244, 42853, 42935, 29793, 36730,
        -28087, 28657, 17946, 7503, 7204, 21491, -27450, -24241, -98156,
        -18082, -42613, -24928, 10775, -14842, -44127, 55910, 14777, 31151, -2194,
        39206, -2100, -4211, 11827, -8918, -19471, 72567, 36447, -65590, -34861,
        -17147, -45303, 9025, -7333, -35473, 11101, 11638, 3441, 6626, -41800,
        9416, 13679, 33508, 40502, -60542, 16358, 8392, -43242, -35864, -34127,
        -48721, 35878, 30598, 28630, 20279, -19983, -14638, -24455, -1851, -11344,
        45150, 42051, 26034, -28889, -32382, -3527, -14532, 22564, -22346, 477,
        11706, 28338, -25972, -9185, -22867, -12522, 32120, -4424, 11339, -33913,
        -7184, 5101, -23552, -17115, -31401, -6104, 21906, 25708, 8406, 6317,
        -7525, 5014, 20750, 20179, 22724, 11692, 13297, 2493, -253, -16841, -17339,
        -6753, -4808, 2976, -10881, -10228, -13816, -12686, 1385, 2316, 2190, -875,
        -1924], ZZ)

    assert dup_mul(p1, p2, ZZ) == res

    p1 = dup_normal([83, -61, -86, -24, 12, 43, -88, -9, 42, 55, -66, 74, 95,
        -25, -12, 68, -99, 4, 45, 6, -15, -19, 78, 65, -55, 47, -13, 17, 86,
        81, -58, -27, 50, -40, -24, 39, -41, -92, 75, 90, -1, 40, -15, -27,
        -35, 68, 70, -64, -40, 78, -88, -58, -39, 69, 46, 12, 28, -94, -37,
        -50, -80, -96, -61, 25, 1, 71, 4, 12, 48, 4, 34, -47, -75, 5, 48, 82,
        88, 23, 98, 35, 17, -10, 48, -61, -95, 47, 65, -19, -66, -57, -6, -51,
        -42, -89, 66, -13, 18, 37, 90, -23, 72, 96, -53, 0, 40, -73, -52, -68,
        32, -25, -53, 79, -52, 18, 44, 73, -81, 31, -90, 70, 3, 36, 48, 76,
        -24, -44, 23, 98, -4, 73, 69, 88, -70, 14, -68, 94, -78, -15, -64, -97,
        -70, -35, 65, 88, 49, -53, -7, 12, -45, -7, 59, -94, 99, -2, 67, -60,
        -71, 29, -62, -77, 1, 51, 17, 80, -20, -47, -19, 24, -9, 39, -23, 21,
        -84, 10, 84, 56, -17, -21, -66, 85, 70, 46, -51, -22, -95, 78, -60,
        -96, -97, -45, 72, 35, 30, -61, -92, -93, -60, -61, 4, -4, -81, -73,
        46, 53, -11, 26, 94, 45, 14, -78, 55, 84, -68, 98, 60, 23, 100, -63,
        68, 96, -16, 3, 56, 21, -58, 62, -67, 66, 85, 41, -79, -22, 97, -67,
        82, 82, -96, -20, -7, 48, -67, 48, -9, -39, 78], ZZ)
    p2 = dup_normal([52, 88, 76, 66, 9, -64, 46, -20, -28, 69, 60, 96, -36,
        -92, -30, -11, -35, 35, 55, 63, -92, -7, 25, -58, 74, 55, -6, 4, 47,
        -92, -65, 67, -45, 74, -76, 59, -6, 69, 39, 24, -71, -7, 39, -45, 60,
        -68, 98, 97, -79, 17, 4, 94, -64, 68, -100, -96, -2, 3, 22, 96, 54,
        -77, -86, 67, 6, 57, 37, 40, 89, -78, 64, -94, -45, -92, 57, 87, -26,
        36, 19, 97, 25, 77, -87, 24, 43, -5, 35, 57, 83, 71, 35, 63, 61, 96,
        -22, 8, -1, 96, 43, 45, 94, -93, 36, 71, -41, -99, 85, -48, 59, 52,
        -17, 5, 87, -16, -68, -54, 76, -18, 100, 91, -42, -70, -66, -88, -12,
        1, 95, -82, 52, 43, -29, 3, 12, 72, -99, -43, -32, -93, -51, 16, -20,
        -12, -11, 5, 33, -38, 93, -5, -74, 25, 74, -58, 93, 59, -63, -86, 63,
        -20, -4, -74, -73, -95, 29, -28, 93, -91, -2, -38, -62, 77, -58, -85,
        -28, 95, 38, 19, -69, 86, 94, 25, -2, -4, 47, 34, -59, 35, -48, 29,
        -63, -53, 34, 29, 66, 73, 6, 92, -84, 89, 15, 81, 93, 97, 51, -72, -78,
        25, 60, 90, -45, 39, 67, -84, -62, 57, 26, -32, -56, -14, -83, 76, 5,
        -2, 99, -100, 28, 46, 94, -7, 53, -25, 16, -23, -36, 89, -78, -63, 31,
        1, 84, -99, -52, 76, 48, 90, -76, 44, -19, 54, -36, -9, -73, -100, -69,
        31, 42, 25, -39, 76, -26, -8, -14, 51, 3, 37, 45, 2, -54, 13, -34, -92,
        17, -25, -65, 53, -63, 30, 4, -70, -67, 90, 52, 51, 18, -3, 31, -45,
        -9, 59, 63, -87, 22, -32, 29, -38, 21, 36, -82, 27, -11], ZZ)
    res = dup_normal([4316, 4132, -3532, -7974, -11303, -10069, 5484, -3330,
        -5874, 7734, 4673, 11327, -9884, -8031, 17343, 21035, -10570, -9285,
        15893, 3780, -14083, 8819, 17592, 10159, 7174, -11587, 8598, -16479,
        3602, 25596, 9781, 12163, 150, 18749, -21782, -12307, 27578, -2757,
        -12573, 12565, 6345, -18956, 19503, -15617, 1443, -16778, 36851, 23588,
        -28474, 5749, 40695, -7521, -53669, -2497, -18530, 6770, 57038, 3926,
        -6927, -15399, 1848, -64649, -27728, 3644, 49608, 15187, -8902, -9480,
        -7398, -40425, 4824, 23767, -7594, -6905, 33089, 18786, 12192, 24670,
        31114, 35334, -4501, -14676, 7107, -59018, -21352, 20777, 19661, 20653,
        33754, -885, -43758, 6269, 51897, -28719, -97488, -9527, 13746, 11644,
        17644, -21720, 23782, -10481, 47867, 20752, 33810, -1875, 39918, -7710,
        -40840, 19808, -47075, 23066, 46616, 25201, 9287, 35436, -1602, 9645,
        -11978, 13273, 15544, 33465, 20063, 44539, 11687, 27314, -6538, -37467,
        14031, 32970, -27086, 41323, 29551, 65910, -39027, -37800, -22232,
        8212, 46316, -28981, -55282, 50417, -44929, -44062, 73879, 37573,
        -2596, -10877, -21893, -133218, -33707, -25753, -9531, 17530, 61126,
        2748, -56235, 43874, -10872, -90459, -30387, 115267, -7264, -44452,
        122626, 14839, -599, 10337, 57166, -67467, -54957, 63669, 1202, 18488,
        52594, 7205, -97822, 612, 78069, -5403, -63562, 47236, 36873, -154827,
        -26188, 82427, -39521, 5628, 7416, 5276, -53095, 47050, 26121, -42207,
        79021, -13035, 2499, -66943, 29040, -72355, -23480, 23416, -12885,
        -44225, -42688, -4224, 19858, 55299, 15735, 11465, 101876, -39169,
        51786, 14723, 43280, -68697, 16410, 92295, 56767, 7183, 111850, 4550,
        115451, -38443, -19642, -35058, 10230, 93829, 8925, 63047, 3146, 29250,
        8530, 5255, -98117, -115517, -76817, -8724, 41044, 1312, -35974, 79333,
        -28567, 7547, -10580, -24559, -16238, 10794, -3867, 24848, 57770,
        -51536, -35040, 71033, 29853, 62029, -7125, -125585, -32169, -47907,
        156811, -65176, -58006, -15757, -57861, 11963, 30225, -41901, -41681,
        31310, 27982, 18613, 61760, 60746, -59096, 33499, 30097, -17997, 24032,
        56442, -83042, 23747, -20931, -21978, -158752, -9883, -73598, -7987,
        -7333, -125403, -116329, 30585, 53281, 51018, -29193, 88575, 8264,
        -40147, -16289, 113088, 12810, -6508, 101552, -13037, 34440, -41840,
        101643, 24263, 80532, 61748, 65574, 6423, -20672, 6591, -10834, -71716,
        86919, -92626, 39161, 28490, 81319, 46676, 106720, 43530, 26998, 57456,
        -8862, 60989, 13982, 3119, -2224, 14743, 55415, -49093, -29303, 28999,
        1789, 55953, -84043, -7780, -65013, 57129, -47251, 61484, 61994,
        -78361, -82778, 22487, -26894, 9756, -74637, -15519, -4360, 30115,
        42433, 35475, 15286, 69768, 21509, -20214, 78675, -21163, 13596, 11443,
        -10698, -53621, -53867, -24155, 64500, -42784, -33077, -16500, 873,
        -52788, 14546, -38011, 36974, -39849, -34029, -94311, 83068, -50437,
        -26169, -46746, 59185, 42259, -101379, -12943, 30089, -59086, 36271,
        22723, -30253, -52472, -70826, -23289, 3331, -31687, 14183, -857,
        -28627, 35246, -51284, 5636, -6933, 66539, 36654, 50927, 24783, 3457,
        33276, 45281, 45650, -4938, -9968, -22590, 47995, 69229, 5214, -58365,
        -17907, -14651, 18668, 18009, 12649, -11851, -13387, 20339, 52472,
        -1087, -21458, -68647, 52295, 15849, 40608, 15323, 25164, -29368,
        10352, -7055, 7159, 21695, -5373, -54849, 101103, -24963, -10511,
        33227, 7659, 41042, -69588, 26718, -20515, 6441, 38135, -63, 24088,
        -35364, -12785, -18709, 47843, 48533, -48575, 17251, -19394, 32878,
        -9010, -9050, 504, -12407, 28076, -3429, 25324, -4210, -26119, 752,
        -29203, 28251, -11324, -32140, -3366, -25135, 18702, -31588, -7047,
        -24267, 49987, -14975, -33169, 37744, -7720, -9035, 16964, -2807, -421,
        14114, -17097, -13662, 40628, -12139, -9427, 5369, 17551, -13232, -16211,
        9804, -7422, 2677, 28635, -8280, -4906, 2908, -22558, 5604, 12459, 8756,
        -3980, -4745, -18525, 7913, 5970, -16457, 20230, -6247, -13812, 2505,
        11899, 1409, -15094, 22540, -18863, 137, 11123, -4516, 2290, -8594, 12150,
        -10380, 3005, 5235, -7350, 2535, -858], ZZ)

    assert dup_mul(p1, p2, ZZ) == res


def test_dmp_mul():
    assert dmp_mul([ZZ(5)], [ZZ(7)], 0, ZZ) == \
        dup_mul([ZZ(5)], [ZZ(7)], ZZ)
    assert dmp_mul([QQ(5, 7)], [QQ(3, 7)], 0, QQ) == \
        dup_mul([QQ(5, 7)], [QQ(3, 7)], QQ)

    assert dmp_mul([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(2)]]]
    assert dmp_mul([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(2)]]]

    assert dmp_mul([[[]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[QQ(2, 7)]]], [[[QQ(1, 3)]]], 2, QQ) == [[[QQ(2, 21)]]]
    assert dmp_mul([[[QQ(1, 7)]]], [[[QQ(2, 3)]]], 2, QQ) == [[[QQ(2, 21)]]]

    K = FF(6)

    assert dmp_mul(
        [[K(2)], [K(1)]], [[K(3)], [K(4)]], 1, K) == [[K(5)], [K(4)]]


def test_dup_sqr():
    assert dup_sqr([], ZZ) == []
    assert dup_sqr([ZZ(2)], ZZ) == [ZZ(4)]
    assert dup_sqr([ZZ(1), ZZ(2)], ZZ) == [ZZ(1), ZZ(4), ZZ(4)]

    assert dup_sqr([], QQ) == []
    assert dup_sqr([QQ(2, 3)], QQ) == [QQ(4, 9)]
    assert dup_sqr([QQ(1, 3), QQ(2, 3)], QQ) == [QQ(1, 9), QQ(4, 9), QQ(4, 9)]

    f = dup_normal([2, 0, 0, 1, 7], ZZ)

    assert dup_sqr(f, ZZ) == dup_normal([4, 0, 0, 4, 28, 0, 1, 14, 49], ZZ)

    K = FF(9)

    assert dup_sqr([K(3), K(4)], K) == [K(6), K(7)]


def test_dmp_sqr():
    assert dmp_sqr([ZZ(1), ZZ(2)], 0, ZZ) == \
        dup_sqr([ZZ(1), ZZ(2)], ZZ)

    assert dmp_sqr([[[]]], 2, ZZ) == [[[]]]
    assert dmp_sqr([[[ZZ(2)]]], 2, ZZ) == [[[ZZ(4)]]]

    assert dmp_sqr([[[]]], 2, QQ) == [[[]]]
    assert dmp_sqr([[[QQ(2, 3)]]], 2, QQ) == [[[QQ(4, 9)]]]

    K = FF(9)

    assert dmp_sqr([[K(3)], [K(4)]], 1, K) == [[K(6)], [K(7)]]


def test_dup_pow():
    assert dup_pow([], 0, ZZ) == [ZZ(1)]
    assert dup_pow([], 0, QQ) == [QQ(1)]

    assert dup_pow([], 1, ZZ) == []
    assert dup_pow([], 7, ZZ) == []

    assert dup_pow([ZZ(1)], 0, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(1)], 1, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(1)], 7, ZZ) == [ZZ(1)]

    assert dup_pow([ZZ(3)], 0, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(3)], 1, ZZ) == [ZZ(3)]
    assert dup_pow([ZZ(3)], 7, ZZ) == [ZZ(2187)]

    assert dup_pow([QQ(1, 1)], 0, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(1, 1)], 1, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(1, 1)], 7, QQ) == [QQ(1, 1)]

    assert dup_pow([QQ(3, 7)], 0, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(3, 7)], 1, QQ) == [QQ(3, 7)]
    assert dup_pow([QQ(3, 7)], 7, QQ) == [QQ(2187, 823543)]

    f = dup_normal([2, 0, 0, 1, 7], ZZ)

    assert dup_pow(f, 0, ZZ) == dup_normal([1], ZZ)
    assert dup_pow(f, 1, ZZ) == dup_normal([2, 0, 0, 1, 7], ZZ)
    assert dup_pow(f, 2, ZZ) == dup_normal([4, 0, 0, 4, 28, 0, 1, 14, 49], ZZ)
    assert dup_pow(f, 3, ZZ) == dup_normal(
        [8, 0, 0, 12, 84, 0, 6, 84, 294, 1, 21, 147, 343], ZZ)


def test_dmp_pow():
    assert dmp_pow([[]], 0, 1, ZZ) == [[ZZ(1)]]
    assert dmp_pow([[]], 0, 1, QQ) == [[QQ(1)]]

    assert dmp_pow([[]], 1, 1, ZZ) == [[]]
    assert dmp_pow([[]], 7, 1, ZZ) == [[]]

    assert dmp_pow([[ZZ(1)]], 0, 1, ZZ) == [[ZZ(1)]]
    assert dmp_pow([[ZZ(1)]], 1, 1, ZZ) == [[ZZ(1)]]
    assert dmp_pow([[ZZ(1)]], 7, 1, ZZ) == [[ZZ(1)]]

    assert dmp_pow([[QQ(3, 7)]], 0, 1, QQ) == [[QQ(1, 1)]]
    assert dmp_pow([[QQ(3, 7)]], 1, 1, QQ) == [[QQ(3, 7)]]
    assert dmp_pow([[QQ(3, 7)]], 7, 1, QQ) == [[QQ(2187, 823543)]]

    f = dup_normal([2, 0, 0, 1, 7], ZZ)

    assert dmp_pow(f, 2, 0, ZZ) == dup_pow(f, 2, ZZ)


def test_dup_pdiv():
    f = dup_normal([3, 1, 1, 5], ZZ)
    g = dup_normal([5, -3, 1], ZZ)

    q = dup_normal([15, 14], ZZ)
    r = dup_normal([52, 111], ZZ)

    assert dup_pdiv(f, g, ZZ) == (q, r)
    assert dup_pquo(f, g, ZZ) == q
    assert dup_prem(f, g, ZZ) == r

    raises(ExactQuotientFailed, lambda: dup_pexquo(f, g, ZZ))

    f = dup_normal([3, 1, 1, 5], QQ)
    g = dup_normal([5, -3, 1], QQ)

    q = dup_normal([15, 14], QQ)
    r = dup_normal([52, 111], QQ)

    assert dup_pdiv(f, g, QQ) == (q, r)
    assert dup_pquo(f, g, QQ) == q
    assert dup_prem(f, g, QQ) == r

    raises(ExactQuotientFailed, lambda: dup_pexquo(f, g, QQ))


def test_dmp_pdiv():
    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[1], [-1, 0]], 1, ZZ)

    q = dmp_normal([[1], [1, 0]], 1, ZZ)
    r = dmp_normal([[2, 0, 0]], 1, ZZ)

    assert dmp_pdiv(f, g, 1, ZZ) == (q, r)
    assert dmp_pquo(f, g, 1, ZZ) == q
    assert dmp_prem(f, g, 1, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_pexquo(f, g, 1, ZZ))

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[2], [-2, 0]], 1, ZZ)

    q = dmp_normal([[2], [2, 0]], 1, ZZ)
    r = dmp_normal([[8, 0, 0]], 1, ZZ)

    assert dmp_pdiv(f, g, 1, ZZ) == (q, r)
    assert dmp_pquo(f, g, 1, ZZ) == q
    assert dmp_prem(f, g, 1, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_pexquo(f, g, 1, ZZ))


def test_dup_rr_div():
    raises(ZeroDivisionError, lambda: dup_rr_div([1, 2, 3], [], ZZ))

    f = dup_normal([3, 1, 1, 5], ZZ)
    g = dup_normal([5, -3, 1], ZZ)

    q, r = [], f

    assert dup_rr_div(f, g, ZZ) == (q, r)


def test_dmp_rr_div():
    raises(ZeroDivisionError, lambda: dmp_rr_div([[1, 2], [3]], [[]], 1, ZZ))

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[1], [-1, 0]], 1, ZZ)

    q = dmp_normal([[1], [1, 0]], 1, ZZ)
    r = dmp_normal([[2, 0, 0]], 1, ZZ)

    assert dmp_rr_div(f, g, 1, ZZ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[-1], [1, 0]], 1, ZZ)

    q = dmp_normal([[-1], [-1, 0]], 1, ZZ)
    r = dmp_normal([[2, 0, 0]], 1, ZZ)

    assert dmp_rr_div(f, g, 1, ZZ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, ZZ)
    g = dmp_normal([[2], [-2, 0]], 1, ZZ)

    q, r = [[]], f

    assert dmp_rr_div(f, g, 1, ZZ) == (q, r)


def test_dup_ff_div():
    raises(ZeroDivisionError, lambda: dup_ff_div([1, 2, 3], [], QQ))

    f = dup_normal([3, 1, 1, 5], QQ)
    g = dup_normal([5, -3, 1], QQ)

    q = [QQ(3, 5), QQ(14, 25)]
    r = [QQ(52, 25), QQ(111, 25)]

    assert dup_ff_div(f, g, QQ) == (q, r)

def test_dup_ff_div_gmpy2():
    if GROUND_TYPES != 'gmpy2':
        return

    from gmpy2 import mpq
    from sympy.polys.domains import GMPYRationalField
    K = GMPYRationalField()

    f = [mpq(1,3), mpq(3,2)]
    g = [mpq(2,1)]
    assert dmp_ff_div(f, g, 0, K) == ([mpq(1,6), mpq(3,4)], [])

    f = [mpq(1,2), mpq(1,3), mpq(1,4), mpq(1,5)]
    g = [mpq(-1,1), mpq(1,1), mpq(-1,1)]
    assert dmp_ff_div(f, g, 0, K) == ([mpq(-1,2), mpq(-5,6)], [mpq(7,12), mpq(-19,30)])

def test_dmp_ff_div():
    raises(ZeroDivisionError, lambda: dmp_ff_div([[1, 2], [3]], [[]], 1, QQ))

    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[1], [-1, 0]], 1, QQ)

    q = [[QQ(1, 1)], [QQ(1, 1), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]

    assert dmp_ff_div(f, g, 1, QQ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[-1], [1, 0]], 1, QQ)

    q = [[QQ(-1, 1)], [QQ(-1, 1), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]

    assert dmp_ff_div(f, g, 1, QQ) == (q, r)

    f = dmp_normal([[1], [], [1, 0, 0]], 1, QQ)
    g = dmp_normal([[2], [-2, 0]], 1, QQ)

    q = [[QQ(1, 2)], [QQ(1, 2), QQ(0, 1)]]
    r = [[QQ(2, 1), QQ(0, 1), QQ(0, 1)]]

    assert dmp_ff_div(f, g, 1, QQ) == (q, r)


def test_dup_div():
    f, g, q, r = [5, 4, 3, 2, 1], [1, 2, 3], [5, -6, 0], [20, 1]

    assert dup_div(f, g, ZZ) == (q, r)
    assert dup_quo(f, g, ZZ) == q
    assert dup_rem(f, g, ZZ) == r

    raises(ExactQuotientFailed, lambda: dup_exquo(f, g, ZZ))

    f, g, q, r = [5, 4, 3, 2, 1, 0], [1, 2, 0, 0, 9], [5, -6], [15, 2, -44, 54]

    assert dup_div(f, g, ZZ) == (q, r)
    assert dup_quo(f, g, ZZ) == q
    assert dup_rem(f, g, ZZ) == r

    raises(ExactQuotientFailed, lambda: dup_exquo(f, g, ZZ))


def test_dmp_div():
    f, g, q, r = [5, 4, 3, 2, 1], [1, 2, 3], [5, -6, 0], [20, 1]

    assert dmp_div(f, g, 0, ZZ) == (q, r)
    assert dmp_quo(f, g, 0, ZZ) == q
    assert dmp_rem(f, g, 0, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_exquo(f, g, 0, ZZ))

    f, g, q, r = [[[1]]], [[[2]], [1]], [[[]]], [[[1]]]

    assert dmp_div(f, g, 2, ZZ) == (q, r)
    assert dmp_quo(f, g, 2, ZZ) == q
    assert dmp_rem(f, g, 2, ZZ) == r

    raises(ExactQuotientFailed, lambda: dmp_exquo(f, g, 2, ZZ))


def test_dup_max_norm():
    assert dup_max_norm([], ZZ) == 0
    assert dup_max_norm([1], ZZ) == 1

    assert dup_max_norm([1, 4, 2, 3], ZZ) == 4


def test_dmp_max_norm():
    assert dmp_max_norm([[[]]], 2, ZZ) == 0
    assert dmp_max_norm([[[1]]], 2, ZZ) == 1

    assert dmp_max_norm(f_0, 2, ZZ) == 6


def test_dup_l1_norm():
    assert dup_l1_norm([], ZZ) == 0
    assert dup_l1_norm([1], ZZ) == 1
    assert dup_l1_norm([1, 4, 2, 3], ZZ) == 10


def test_dmp_l1_norm():
    assert dmp_l1_norm([[[]]], 2, ZZ) == 0
    assert dmp_l1_norm([[[1]]], 2, ZZ) == 1

    assert dmp_l1_norm(f_0, 2, ZZ) == 31


def test_dup_l2_norm_squared():
    assert dup_l2_norm_squared([], ZZ) == 0
    assert dup_l2_norm_squared([1], ZZ) == 1
    assert dup_l2_norm_squared([1, 4, 2, 3], ZZ) == 30


def test_dmp_l2_norm_squared():
    assert dmp_l2_norm_squared([[[]]], 2, ZZ) == 0
    assert dmp_l2_norm_squared([[[1]]], 2, ZZ) == 1
    assert dmp_l2_norm_squared(f_0, 2, ZZ) == 111


def test_dup_expand():
    assert dup_expand((), ZZ) == [1]
    assert dup_expand(([1, 2, 3], [1, 2], [7, 5, 4, 3]), ZZ) == \
        dup_mul([1, 2, 3], dup_mul([1, 2], [7, 5, 4, 3], ZZ), ZZ)


def test_dmp_expand():
    assert dmp_expand((), 1, ZZ) == [[1]]
    assert dmp_expand(([[1], [2], [3]], [[1], [2]], [[7], [5], [4], [3]]), 1, ZZ) == \
        dmp_mul([[1], [2], [3]], dmp_mul([[1], [2]], [[7], [5], [
                4], [3]], 1, ZZ), 1, ZZ)

def test_dup_mul_poly():
    p = Poly(18786186952704.0*x**165 + 9.31746684052255e+31*x**82, x, domain='RR')
    px = Poly(18786186952704.0*x**166 + 9.31746684052255e+31*x**83, x, domain='RR')

    assert p * x == px
    assert p.set_domain(QQ) * x == px.set_domain(QQ)
    assert p.set_domain(CC) * x == px.set_domain(CC)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_indexing.py ===
from datetime import timedelta
import re

import numpy as np
import pytest

from pandas._libs import index as libindex
from pandas.errors import (
    InvalidIndexError,
    PerformanceWarning,
)

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    date_range,
)
import pandas._testing as tm


class TestSliceLocs:
    def test_slice_locs_partial(self, idx):
        sorted_idx, _ = idx.sortlevel(0)

        result = sorted_idx.slice_locs(("foo", "two"), ("qux", "one"))
        assert result == (1, 5)

        result = sorted_idx.slice_locs(None, ("qux", "one"))
        assert result == (0, 5)

        result = sorted_idx.slice_locs(("foo", "two"), None)
        assert result == (1, len(sorted_idx))

        result = sorted_idx.slice_locs("bar", "baz")
        assert result == (2, 4)

    def test_slice_locs(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((50, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=50, freq="B"),
        )
        stacked = df.stack(future_stack=True)
        idx = stacked.index

        slob = slice(*idx.slice_locs(df.index[5], df.index[15]))
        sliced = stacked[slob]
        expected = df[5:16].stack(future_stack=True)
        tm.assert_almost_equal(sliced.values, expected.values)

        slob = slice(
            *idx.slice_locs(
                df.index[5] + timedelta(seconds=30),
                df.index[15] - timedelta(seconds=30),
            )
        )
        sliced = stacked[slob]
        expected = df[6:15].stack(future_stack=True)
        tm.assert_almost_equal(sliced.values, expected.values)

    def test_slice_locs_with_type_mismatch(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        stacked = df.stack(future_stack=True)
        idx = stacked.index
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs((1, 3))
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs(df.index[5] + timedelta(seconds=30), (5, 2))
        df = DataFrame(
            np.ones((5, 5)),
            index=Index([f"i-{i}" for i in range(5)], name="a"),
            columns=Index([f"i-{i}" for i in range(5)], name="a"),
        )
        stacked = df.stack(future_stack=True)
        idx = stacked.index
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs(timedelta(seconds=30))
        # TODO: Try creating a UnicodeDecodeError in exception message
        with pytest.raises(TypeError, match="^Level type mismatch"):
            idx.slice_locs(df.index[1], (16, "a"))

    def test_slice_locs_not_sorted(self):
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        msg = "[Kk]ey length.*greater than MultiIndex lexsort depth"
        with pytest.raises(KeyError, match=msg):
            index.slice_locs((1, 0, 1), (2, 1, 0))

        # works
        sorted_index, _ = index.sortlevel(0)
        # should there be a test case here???
        sorted_index.slice_locs((1, 0, 1), (2, 1, 0))

    def test_slice_locs_not_contained(self):
        # some searchsorted action

        index = MultiIndex(
            levels=[[0, 2, 4, 6], [0, 2, 4]],
            codes=[[0, 0, 0, 1, 1, 2, 3, 3, 3], [0, 1, 2, 1, 2, 2, 0, 1, 2]],
        )

        result = index.slice_locs((1, 0), (5, 2))
        assert result == (3, 6)

        result = index.slice_locs(1, 5)
        assert result == (3, 6)

        result = index.slice_locs((2, 2), (5, 2))
        assert result == (3, 6)

        result = index.slice_locs(2, 5)
        assert result == (3, 6)

        result = index.slice_locs((1, 0), (6, 3))
        assert result == (3, 8)

        result = index.slice_locs(-1, 10)
        assert result == (0, len(index))

    @pytest.mark.parametrize(
        "index_arr,expected,start_idx,end_idx",
        [
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, None),
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, "b"),
            ([[np.nan, "a", "b"], ["c", "d", "e"]], (0, 3), np.nan, ("b", "e")),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), None),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), "c"),
            ([["a", "b", "c"], ["d", np.nan, "e"]], (1, 3), ("b", np.nan), ("c", "e")),
        ],
    )
    def test_slice_locs_with_missing_value(
        self, index_arr, expected, start_idx, end_idx
    ):
        # issue 19132
        idx = MultiIndex.from_arrays(index_arr)
        result = idx.slice_locs(start=start_idx, end=end_idx)
        assert result == expected


class TestPutmask:
    def test_putmask_with_wrong_mask(self, idx):
        # GH18368

        msg = "putmask: mask and data must be the same size"
        with pytest.raises(ValueError, match=msg):
            idx.putmask(np.ones(len(idx) + 1, np.bool_), 1)

        with pytest.raises(ValueError, match=msg):
            idx.putmask(np.ones(len(idx) - 1, np.bool_), 1)

        with pytest.raises(ValueError, match=msg):
            idx.putmask("foo", 1)

    def test_putmask_multiindex_other(self):
        # GH#43212 `value` is also a MultiIndex

        left = MultiIndex.from_tuples([(np.nan, 6), (np.nan, 6), ("a", 4)])
        right = MultiIndex.from_tuples([("a", 1), ("a", 1), ("d", 1)])
        mask = np.array([True, True, False])

        result = left.putmask(mask, right)

        expected = MultiIndex.from_tuples([right[0], right[1], left[2]])
        tm.assert_index_equal(result, expected)

    def test_putmask_keep_dtype(self, any_numeric_ea_dtype):
        # GH#49830
        midx = MultiIndex.from_arrays(
            [pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]]
        )
        midx2 = MultiIndex.from_arrays(
            [pd.Series([5, 6, 7], dtype=any_numeric_ea_dtype), [-1, -2, -3]]
        )
        result = midx.putmask([True, False, False], midx2)
        expected = MultiIndex.from_arrays(
            [pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]]
        )
        tm.assert_index_equal(result, expected)

    def test_putmask_keep_dtype_shorter_value(self, any_numeric_ea_dtype):
        # GH#49830
        midx = MultiIndex.from_arrays(
            [pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]]
        )
        midx2 = MultiIndex.from_arrays(
            [pd.Series([5], dtype=any_numeric_ea_dtype), [-1]]
        )
        result = midx.putmask([True, False, False], midx2)
        expected = MultiIndex.from_arrays(
            [pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]]
        )
        tm.assert_index_equal(result, expected)


class TestGetIndexer:
    def test_get_indexer(self):
        major_axis = Index(np.arange(4))
        minor_axis = Index(np.arange(2))

        major_codes = np.array([0, 0, 1, 2, 2, 3, 3], dtype=np.intp)
        minor_codes = np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.intp)

        index = MultiIndex(
            levels=[major_axis, minor_axis], codes=[major_codes, minor_codes]
        )
        idx1 = index[:5]
        idx2 = index[[1, 3, 5]]

        r1 = idx1.get_indexer(idx2)
        tm.assert_almost_equal(r1, np.array([1, 3, -1], dtype=np.intp))

        r1 = idx2.get_indexer(idx1, method="pad")
        e1 = np.array([-1, 0, 0, 1, 1], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)

        r2 = idx2.get_indexer(idx1[::-1], method="pad")
        tm.assert_almost_equal(r2, e1[::-1])

        rffill1 = idx2.get_indexer(idx1, method="ffill")
        tm.assert_almost_equal(r1, rffill1)

        r1 = idx2.get_indexer(idx1, method="backfill")
        e1 = np.array([0, 0, 1, 1, 2], dtype=np.intp)
        tm.assert_almost_equal(r1, e1)

        r2 = idx2.get_indexer(idx1[::-1], method="backfill")
        tm.assert_almost_equal(r2, e1[::-1])

        rbfill1 = idx2.get_indexer(idx1, method="bfill")
        tm.assert_almost_equal(r1, rbfill1)

        # pass non-MultiIndex
        r1 = idx1.get_indexer(idx2.values)
        rexp1 = idx1.get_indexer(idx2)
        tm.assert_almost_equal(r1, rexp1)

        r1 = idx1.get_indexer([1, 2, 3])
        assert (r1 == [-1, -1, -1]).all()

        # create index with duplicates
        idx1 = Index(list(range(10)) + list(range(10)))
        idx2 = Index(list(range(20)))

        msg = "Reindexing only valid with uniquely valued Index objects"
        with pytest.raises(InvalidIndexError, match=msg):
            idx1.get_indexer(idx2)

    def test_get_indexer_nearest(self):
        midx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
        msg = (
            "method='nearest' not implemented yet for MultiIndex; "
            "see GitHub issue 9365"
        )
        with pytest.raises(NotImplementedError, match=msg):
            midx.get_indexer(["a"], method="nearest")
        msg = "tolerance not implemented yet for MultiIndex"
        with pytest.raises(NotImplementedError, match=msg):
            midx.get_indexer(["a"], method="pad", tolerance=2)

    def test_get_indexer_categorical_time(self):
        # https://github.com/pandas-dev/pandas/issues/21390
        midx = MultiIndex.from_product(
            [
                Categorical(["a", "b", "c"]),
                Categorical(date_range("2012-01-01", periods=3, freq="h")),
            ]
        )
        result = midx.get_indexer(midx)
        tm.assert_numpy_array_equal(result, np.arange(9, dtype=np.intp))

    @pytest.mark.parametrize(
        "index_arr,labels,expected",
        [
            (
                [[1, np.nan, 2], [3, 4, 5]],
                [1, np.nan, 2],
                np.array([-1, -1, -1], dtype=np.intp),
            ),
            ([[1, np.nan, 2], [3, 4, 5]], [(np.nan, 4)], np.array([1], dtype=np.intp)),
            ([[1, 2, 3], [np.nan, 4, 5]], [(1, np.nan)], np.array([0], dtype=np.intp)),
            (
                [[1, 2, 3], [np.nan, 4, 5]],
                [np.nan, 4, 5],
                np.array([-1, -1, -1], dtype=np.intp),
            ),
        ],
    )
    def test_get_indexer_with_missing_value(self, index_arr, labels, expected):
        # issue 19132
        idx = MultiIndex.from_arrays(index_arr)
        result = idx.get_indexer(labels)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_methods(self):
        # https://github.com/pandas-dev/pandas/issues/29896
        # test getting an indexer for another index with different methods
        # confirms that getting an indexer without a filling method, getting an
        # indexer and backfilling, and getting an indexer and padding all behave
        # correctly in the case where all of the target values fall in between
        # several levels in the MultiIndex into which they are getting an indexer
        #
        # visually, the MultiIndexes used in this test are:
        # mult_idx_1:
        #  0: -1 0
        #  1:    2
        #  2:    3
        #  3:    4
        #  4:  0 0
        #  5:    2
        #  6:    3
        #  7:    4
        #  8:  1 0
        #  9:    2
        # 10:    3
        # 11:    4
        #
        # mult_idx_2:
        #  0: 0 1
        #  1:   3
        #  2:   4
        mult_idx_1 = MultiIndex.from_product([[-1, 0, 1], [0, 2, 3, 4]])
        mult_idx_2 = MultiIndex.from_product([[0], [1, 3, 4]])

        indexer = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([-1, 6, 7], dtype=indexer.dtype)
        tm.assert_almost_equal(expected, indexer)

        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="backfill")
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)

        # ensure the legacy "bfill" option functions identically to "backfill"
        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="bfill")
        expected = np.array([5, 6, 7], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)

        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="pad")
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

        # ensure the legacy "ffill" option functions identically to "pad"
        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="ffill")
        expected = np.array([4, 6, 7], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

    @pytest.mark.parametrize("method", ["pad", "ffill", "backfill", "bfill", "nearest"])
    def test_get_indexer_methods_raise_for_non_monotonic(self, method):
        # 53452
        mi = MultiIndex.from_arrays([[0, 4, 2], [0, 4, 2]])
        if method == "nearest":
            err = NotImplementedError
            msg = "not implemented yet for MultiIndex"
        else:
            err = ValueError
            msg = "index must be monotonic increasing or decreasing"
        with pytest.raises(err, match=msg):
            mi.get_indexer([(1, 1)], method=method)

    def test_get_indexer_three_or_more_levels(self):
        # https://github.com/pandas-dev/pandas/issues/29896
        # tests get_indexer() on MultiIndexes with 3+ levels
        # visually, these are
        # mult_idx_1:
        #  0: 1 2 5
        #  1:     7
        #  2:   4 5
        #  3:     7
        #  4:   6 5
        #  5:     7
        #  6: 3 2 5
        #  7:     7
        #  8:   4 5
        #  9:     7
        # 10:   6 5
        # 11:     7
        #
        # mult_idx_2:
        #  0: 1 1 8
        #  1: 1 5 9
        #  2: 1 6 7
        #  3: 2 1 6
        #  4: 2 7 6
        #  5: 2 7 8
        #  6: 3 6 8
        mult_idx_1 = MultiIndex.from_product([[1, 3], [2, 4, 6], [5, 7]])
        mult_idx_2 = MultiIndex.from_tuples(
            [
                (1, 1, 8),
                (1, 5, 9),
                (1, 6, 7),
                (2, 1, 6),
                (2, 7, 7),
                (2, 7, 8),
                (3, 6, 8),
            ]
        )
        # sanity check
        assert mult_idx_1.is_monotonic_increasing
        assert mult_idx_1.is_unique
        assert mult_idx_2.is_monotonic_increasing
        assert mult_idx_2.is_unique

        # show the relationships between the two
        assert mult_idx_2[0] < mult_idx_1[0]
        assert mult_idx_1[3] < mult_idx_2[1] < mult_idx_1[4]
        assert mult_idx_1[5] == mult_idx_2[2]
        assert mult_idx_1[5] < mult_idx_2[3] < mult_idx_1[6]
        assert mult_idx_1[5] < mult_idx_2[4] < mult_idx_1[6]
        assert mult_idx_1[5] < mult_idx_2[5] < mult_idx_1[6]
        assert mult_idx_1[-1] < mult_idx_2[6]

        indexer_no_fill = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([-1, -1, 5, -1, -1, -1, -1], dtype=indexer_no_fill.dtype)
        tm.assert_almost_equal(expected, indexer_no_fill)

        # test with backfilling
        indexer_backfilled = mult_idx_1.get_indexer(mult_idx_2, method="backfill")
        expected = np.array([0, 4, 5, 6, 6, 6, -1], dtype=indexer_backfilled.dtype)
        tm.assert_almost_equal(expected, indexer_backfilled)

        # now, the same thing, but forward-filled (aka "padded")
        indexer_padded = mult_idx_1.get_indexer(mult_idx_2, method="pad")
        expected = np.array([-1, 3, 5, 5, 5, 5, 11], dtype=indexer_padded.dtype)
        tm.assert_almost_equal(expected, indexer_padded)

        # now, do the indexing in the other direction
        assert mult_idx_2[0] < mult_idx_1[0] < mult_idx_2[1]
        assert mult_idx_2[0] < mult_idx_1[1] < mult_idx_2[1]
        assert mult_idx_2[0] < mult_idx_1[2] < mult_idx_2[1]
        assert mult_idx_2[0] < mult_idx_1[3] < mult_idx_2[1]
        assert mult_idx_2[1] < mult_idx_1[4] < mult_idx_2[2]
        assert mult_idx_2[2] == mult_idx_1[5]
        assert mult_idx_2[5] < mult_idx_1[6] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[7] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[8] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[9] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[10] < mult_idx_2[6]
        assert mult_idx_2[5] < mult_idx_1[11] < mult_idx_2[6]

        indexer = mult_idx_2.get_indexer(mult_idx_1)
        expected = np.array(
            [-1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1], dtype=indexer.dtype
        )
        tm.assert_almost_equal(expected, indexer)

        backfill_indexer = mult_idx_2.get_indexer(mult_idx_1, method="bfill")
        expected = np.array(
            [1, 1, 1, 1, 2, 2, 6, 6, 6, 6, 6, 6], dtype=backfill_indexer.dtype
        )
        tm.assert_almost_equal(expected, backfill_indexer)

        pad_indexer = mult_idx_2.get_indexer(mult_idx_1, method="pad")
        expected = np.array(
            [0, 0, 0, 0, 1, 2, 5, 5, 5, 5, 5, 5], dtype=pad_indexer.dtype
        )
        tm.assert_almost_equal(expected, pad_indexer)

    def test_get_indexer_crossing_levels(self):
        # https://github.com/pandas-dev/pandas/issues/29896
        # tests a corner case with get_indexer() with MultiIndexes where, when we
        # need to "carry" across levels, proper tuple ordering is respected
        #
        # the MultiIndexes used in this test, visually, are:
        # mult_idx_1:
        #  0: 1 1 1 1
        #  1:       2
        #  2:     2 1
        #  3:       2
        #  4: 1 2 1 1
        #  5:       2
        #  6:     2 1
        #  7:       2
        #  8: 2 1 1 1
        #  9:       2
        # 10:     2 1
        # 11:       2
        # 12: 2 2 1 1
        # 13:       2
        # 14:     2 1
        # 15:       2
        #
        # mult_idx_2:
        #  0: 1 3 2 2
        #  1: 2 3 2 2
        mult_idx_1 = MultiIndex.from_product([[1, 2]] * 4)
        mult_idx_2 = MultiIndex.from_tuples([(1, 3, 2, 2), (2, 3, 2, 2)])

        # show the tuple orderings, which get_indexer() should respect
        assert mult_idx_1[7] < mult_idx_2[0] < mult_idx_1[8]
        assert mult_idx_1[-1] < mult_idx_2[1]

        indexer = mult_idx_1.get_indexer(mult_idx_2)
        expected = np.array([-1, -1], dtype=indexer.dtype)
        tm.assert_almost_equal(expected, indexer)

        backfill_indexer = mult_idx_1.get_indexer(mult_idx_2, method="bfill")
        expected = np.array([8, -1], dtype=backfill_indexer.dtype)
        tm.assert_almost_equal(expected, backfill_indexer)

        pad_indexer = mult_idx_1.get_indexer(mult_idx_2, method="ffill")
        expected = np.array([7, 15], dtype=pad_indexer.dtype)
        tm.assert_almost_equal(expected, pad_indexer)

    def test_get_indexer_kwarg_validation(self):
        # GH#41918
        mi = MultiIndex.from_product([range(3), ["A", "B"]])

        msg = "limit argument only valid if doing pad, backfill or nearest"
        with pytest.raises(ValueError, match=msg):
            mi.get_indexer(mi[:-1], limit=4)

        msg = "tolerance argument only valid if doing pad, backfill or nearest"
        with pytest.raises(ValueError, match=msg):
            mi.get_indexer(mi[:-1], tolerance="piano")

    def test_get_indexer_nan(self):
        # GH#37222
        idx1 = MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"])
        idx2 = MultiIndex.from_product([["A"], [np.nan, 2.0]], names=["id1", "id2"])
        expected = np.array([-1, 1])
        result = idx2.get_indexer(idx1)
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)
        result = idx1.get_indexer(idx2)
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)


def test_getitem(idx):
    # scalar
    assert idx[2] == ("bar", "one")

    # slice
    result = idx[2:5]
    expected = idx[[2, 3, 4]]
    assert result.equals(expected)

    # boolean
    result = idx[[True, False, True, False, True, True]]
    result2 = idx[np.array([True, False, True, False, True, True])]
    expected = idx[[0, 2, 4, 5]]
    assert result.equals(expected)
    assert result2.equals(expected)


def test_getitem_group_select(idx):
    sorted_idx, _ = idx.sortlevel(0)
    assert sorted_idx.get_loc("baz") == slice(3, 4)
    assert sorted_idx.get_loc("foo") == slice(0, 2)


@pytest.mark.parametrize("ind1", [[True] * 5, Index([True] * 5)])
@pytest.mark.parametrize(
    "ind2",
    [[True, False, True, False, False], Index([True, False, True, False, False])],
)
def test_getitem_bool_index_all(ind1, ind2):
    # GH#22533
    idx = MultiIndex.from_tuples([(10, 1), (20, 2), (30, 3), (40, 4), (50, 5)])
    tm.assert_index_equal(idx[ind1], idx)

    expected = MultiIndex.from_tuples([(10, 1), (30, 3)])
    tm.assert_index_equal(idx[ind2], expected)


@pytest.mark.parametrize("ind1", [[True], Index([True])])
@pytest.mark.parametrize("ind2", [[False], Index([False])])
def test_getitem_bool_index_single(ind1, ind2):
    # GH#22533
    idx = MultiIndex.from_tuples([(10, 1)])
    tm.assert_index_equal(idx[ind1], idx)

    expected = MultiIndex(
        levels=[np.array([], dtype=np.int64), np.array([], dtype=np.int64)],
        codes=[[], []],
    )
    tm.assert_index_equal(idx[ind2], expected)


class TestGetLoc:
    def test_get_loc(self, idx):
        assert idx.get_loc(("foo", "two")) == 1
        assert idx.get_loc(("baz", "two")) == 3
        with pytest.raises(KeyError, match=r"^\('bar', 'two'\)$"):
            idx.get_loc(("bar", "two"))
        with pytest.raises(KeyError, match=r"^'quux'$"):
            idx.get_loc("quux")

        # 3 levels
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        with pytest.raises(KeyError, match=r"^\(1, 1\)$"):
            index.get_loc((1, 1))
        assert index.get_loc((2, 0)) == slice(3, 5)

    def test_get_loc_duplicates(self):
        index = Index([2, 2, 2, 2])
        result = index.get_loc(2)
        expected = slice(0, 4)
        assert result == expected

        index = Index(["c", "a", "a", "b", "b"])
        rs = index.get_loc("c")
        xp = 0
        assert rs == xp

        with pytest.raises(KeyError, match="2"):
            index.get_loc(2)

    def test_get_loc_level(self):
        index = MultiIndex(
            levels=[Index(np.arange(4)), Index(np.arange(4)), Index(np.arange(4))],
            codes=[
                np.array([0, 0, 1, 2, 2, 2, 3, 3]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1]),
                np.array([1, 0, 1, 1, 0, 0, 1, 0]),
            ],
        )
        loc, new_index = index.get_loc_level((0, 1))
        expected = slice(1, 2)
        exp_index = index[expected].droplevel(0).droplevel(0)
        assert loc == expected
        assert new_index.equals(exp_index)

        loc, new_index = index.get_loc_level((0, 1, 0))
        expected = 1
        assert loc == expected
        assert new_index is None

        with pytest.raises(KeyError, match=r"^\(2, 2\)$"):
            index.get_loc_level((2, 2))
        # GH 22221: unused label
        with pytest.raises(KeyError, match=r"^2$"):
            index.drop(2).get_loc_level(2)
        # Unused label on unsorted level:
        with pytest.raises(KeyError, match=r"^2$"):
            index.drop(1, level=2).get_loc_level(2, level=2)

        index = MultiIndex(
            levels=[[2000], list(range(4))],
            codes=[np.array([0, 0, 0, 0]), np.array([0, 1, 2, 3])],
        )
        result, new_index = index.get_loc_level((2000, slice(None, None)))
        expected = slice(None, None)
        assert result == expected
        assert new_index.equals(index.droplevel(0))

    @pytest.mark.parametrize("dtype1", [int, float, bool, str])
    @pytest.mark.parametrize("dtype2", [int, float, bool, str])
    def test_get_loc_multiple_dtypes(self, dtype1, dtype2):
        # GH 18520
        levels = [np.array([0, 1]).astype(dtype1), np.array([0, 1]).astype(dtype2)]
        idx = MultiIndex.from_product(levels)
        assert idx.get_loc(idx[2]) == 2

    @pytest.mark.parametrize("level", [0, 1])
    @pytest.mark.parametrize("dtypes", [[int, float], [float, int]])
    def test_get_loc_implicit_cast(self, level, dtypes):
        # GH 18818, GH 15994 : as flat index, cast int to float and vice-versa
        levels = [["a", "b"], ["c", "d"]]
        key = ["b", "d"]
        lev_dtype, key_dtype = dtypes
        levels[level] = np.array([0, 1], dtype=lev_dtype)
        key[level] = key_dtype(1)
        idx = MultiIndex.from_product(levels)
        assert idx.get_loc(tuple(key)) == 3

    @pytest.mark.parametrize("dtype", [bool, object])
    def test_get_loc_cast_bool(self, dtype):
        # GH 19086 : int is casted to bool, but not vice-versa (for object dtype)
        #  With bool dtype, we don't cast in either direction.
        levels = [Index([False, True], dtype=dtype), np.arange(2, dtype="int64")]
        idx = MultiIndex.from_product(levels)

        if dtype is bool:
            with pytest.raises(KeyError, match=r"^\(0, 1\)$"):
                assert idx.get_loc((0, 1)) == 1
            with pytest.raises(KeyError, match=r"^\(1, 0\)$"):
                assert idx.get_loc((1, 0)) == 2
        else:
            # We use python object comparisons, which treat 0 == False and 1 == True
            assert idx.get_loc((0, 1)) == 1
            assert idx.get_loc((1, 0)) == 2

        with pytest.raises(KeyError, match=r"^\(False, True\)$"):
            idx.get_loc((False, True))
        with pytest.raises(KeyError, match=r"^\(True, False\)$"):
            idx.get_loc((True, False))

    @pytest.mark.parametrize("level", [0, 1])
    def test_get_loc_nan(self, level, nulls_fixture):
        # GH 18485 : NaN in MultiIndex
        levels = [["a", "b"], ["c", "d"]]
        key = ["b", "d"]
        levels[level] = np.array([0, nulls_fixture], dtype=type(nulls_fixture))
        key[level] = nulls_fixture
        idx = MultiIndex.from_product(levels)
        assert idx.get_loc(tuple(key)) == 3

    def test_get_loc_missing_nan(self):
        # GH 8569
        idx = MultiIndex.from_arrays([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(idx.get_loc(1), slice)
        with pytest.raises(KeyError, match=r"^3$"):
            idx.get_loc(3)
        with pytest.raises(KeyError, match=r"^nan$"):
            idx.get_loc(np.nan)
        with pytest.raises(InvalidIndexError, match=r"\[nan\]"):
            # listlike/non-hashable raises TypeError
            idx.get_loc([np.nan])

    def test_get_loc_with_values_including_missing_values(self):
        # issue 19132
        idx = MultiIndex.from_product([[np.nan, 1]] * 2)
        expected = slice(0, 2, None)
        assert idx.get_loc(np.nan) == expected

        idx = MultiIndex.from_arrays([[np.nan, 1, 2, np.nan]])
        expected = np.array([True, False, False, True])
        tm.assert_numpy_array_equal(idx.get_loc(np.nan), expected)

        idx = MultiIndex.from_product([[np.nan, 1]] * 3)
        expected = slice(2, 4, None)
        assert idx.get_loc((np.nan, 1)) == expected

    def test_get_loc_duplicates2(self):
        # TODO: de-duplicate with test_get_loc_duplicates above?
        index = MultiIndex(
            levels=[["D", "B", "C"], [0, 26, 27, 37, 57, 67, 75, 82]],
            codes=[[0, 0, 0, 1, 2, 2, 2, 2, 2, 2], [1, 3, 4, 6, 0, 2, 2, 3, 5, 7]],
            names=["tag", "day"],
        )

        assert index.get_loc("D") == slice(0, 3)

    def test_get_loc_past_lexsort_depth(self):
        # GH#30053
        idx = MultiIndex(
            levels=[["a"], [0, 7], [1]],
            codes=[[0, 0], [1, 0], [0, 0]],
            names=["x", "y", "z"],
            sortorder=0,
        )
        key = ("a", 7)

        with tm.assert_produces_warning(PerformanceWarning):
            # PerformanceWarning: indexing past lexsort depth may impact performance
            result = idx.get_loc(key)

        assert result == slice(0, 1, None)

    def test_multiindex_get_loc_list_raises(self):
        # GH#35878
        idx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
        msg = r"\[\]"
        with pytest.raises(InvalidIndexError, match=msg):
            idx.get_loc([])

    def test_get_loc_nested_tuple_raises_keyerror(self):
        # raise KeyError, not TypeError
        mi = MultiIndex.from_product([range(3), range(4), range(5), range(6)])
        key = ((2, 3, 4), "foo")

        with pytest.raises(KeyError, match=re.escape(str(key))):
            mi.get_loc(key)


class TestWhere:
    def test_where(self):
        i = MultiIndex.from_tuples([("A", 1), ("A", 2)])

        msg = r"\.where is not supported for MultiIndex operations"
        with pytest.raises(NotImplementedError, match=msg):
            i.where(True)

    def test_where_array_like(self, listlike_box):
        mi = MultiIndex.from_tuples([("A", 1), ("A", 2)])
        cond = [False, True]
        msg = r"\.where is not supported for MultiIndex operations"
        with pytest.raises(NotImplementedError, match=msg):
            mi.where(listlike_box(cond))


class TestContains:
    def test_contains_top_level(self):
        midx = MultiIndex.from_product([["A", "B"], [1, 2]])
        assert "A" in midx
        assert "A" not in midx._engine

    def test_contains_with_nat(self):
        # MI with a NaT
        mi = MultiIndex(
            levels=[["C"], date_range("2012-01-01", periods=5)],
            codes=[[0, 0, 0, 0, 0, 0], [-1, 0, 1, 2, 3, 4]],
            names=[None, "B"],
        )
        assert ("C", pd.Timestamp("2012-01-01")) in mi
        for val in mi.values:
            assert val in mi

    def test_contains(self, idx):
        assert ("foo", "two") in idx
        assert ("bar", "two") not in idx
        assert None not in idx

    def test_contains_with_missing_value(self):
        # GH#19132
        idx = MultiIndex.from_arrays([[1, np.nan, 2]])
        assert np.nan in idx

        idx = MultiIndex.from_arrays([[1, 2], [np.nan, 3]])
        assert np.nan not in idx
        assert (1, np.nan) in idx

    def test_multiindex_contains_dropped(self):
        # GH#19027
        # test that dropped MultiIndex levels are not in the MultiIndex
        # despite continuing to be in the MultiIndex's levels
        idx = MultiIndex.from_product([[1, 2], [3, 4]])
        assert 2 in idx
        idx = idx.drop(2)

        # drop implementation keeps 2 in the levels
        assert 2 in idx.levels[0]
        # but it should no longer be in the index itself
        assert 2 not in idx

        # also applies to strings
        idx = MultiIndex.from_product([["a", "b"], ["c", "d"]])
        assert "a" in idx
        idx = idx.drop("a")
        assert "a" in idx.levels[0]
        assert "a" not in idx

    def test_contains_td64_level(self):
        # GH#24570
        tx = pd.timedelta_range("09:30:00", "16:00:00", freq="30 min")
        idx = MultiIndex.from_arrays([tx, np.arange(len(tx))])
        assert tx[0] in idx
        assert "element_not_exit" not in idx
        assert "0 day 09:30:00" in idx

    def test_large_mi_contains(self, monkeypatch):
        # GH#10645
        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 10)
            result = MultiIndex.from_arrays([range(10), range(10)])
            assert (10, 0) not in result


def test_timestamp_multiindex_indexer():
    # https://github.com/pandas-dev/pandas/issues/26944
    idx = MultiIndex.from_product(
        [
            date_range("2019-01-01T00:15:33", periods=100, freq="h", name="date"),
            ["x"],
            [3],
        ]
    )
    df = DataFrame({"foo": np.arange(len(idx))}, idx)
    result = df.loc[pd.IndexSlice["2019-1-2":, "x", :], "foo"]
    qidx = MultiIndex.from_product(
        [
            date_range(
                start="2019-01-02T00:15:33",
                end="2019-01-05T03:15:33",
                freq="h",
                name="date",
            ),
            ["x"],
            [3],
        ]
    )
    should_be = pd.Series(data=np.arange(24, len(qidx) + 24), index=qidx, name="foo")
    tm.assert_series_equal(result, should_be)


@pytest.mark.parametrize(
    "index_arr,expected,target,algo",
    [
        ([[np.nan, "a", "b"], ["c", "d", "e"]], 0, np.nan, "left"),
        ([[np.nan, "a", "b"], ["c", "d", "e"]], 1, (np.nan, "c"), "right"),
        ([["a", "b", "c"], ["d", np.nan, "d"]], 1, ("b", np.nan), "left"),
    ],
)
def test_get_slice_bound_with_missing_value(index_arr, expected, target, algo):
    # issue 19132
    idx = MultiIndex.from_arrays(index_arr)
    result = idx.get_slice_bound(target, side=algo)
    assert result == expected


@pytest.mark.parametrize(
    "index_arr,expected,start_idx,end_idx",
    [
        ([[np.nan, 1, 2], [3, 4, 5]], slice(0, 2, None), np.nan, 1),
        ([[np.nan, 1, 2], [3, 4, 5]], slice(0, 3, None), np.nan, (2, 5)),
        ([[1, 2, 3], [4, np.nan, 5]], slice(1, 3, None), (2, np.nan), 3),
        ([[1, 2, 3], [4, np.nan, 5]], slice(1, 3, None), (2, np.nan), (3, 5)),
    ],
)
def test_slice_indexer_with_missing_value(index_arr, expected, start_idx, end_idx):
    # issue 19132
    idx = MultiIndex.from_arrays(index_arr)
    result = idx.slice_indexer(start=start_idx, end=end_idx)
    assert result == expected


def test_pyint_engine():
    # GH#18519 : when combinations of codes cannot be represented in 64
    # bits, the index underlying the MultiIndex engine works with Python
    # integers, rather than uint64.
    N = 5
    keys = [
        tuple(arr)
        for arr in [
            [0] * 10 * N,
            [1] * 10 * N,
            [2] * 10 * N,
            [np.nan] * N + [2] * 9 * N,
            [0] * N + [2] * 9 * N,
            [np.nan] * N + [2] * 8 * N + [0] * N,
        ]
    ]
    # Each level contains 4 elements (including NaN), so it is represented
    # in 2 bits, for a total of 2*N*10 = 100 > 64 bits. If we were using a
    # 64 bit engine and truncating the first levels, the fourth and fifth
    # keys would collide; if truncating the last levels, the fifth and
    # sixth; if rotating bits rather than shifting, the third and fifth.

    for idx, key_value in enumerate(keys):
        index = MultiIndex.from_tuples(keys)
        assert index.get_loc(key_value) == idx

        expected = np.arange(idx + 1, dtype=np.intp)
        result = index.get_indexer([keys[i] for i in expected])
        tm.assert_numpy_array_equal(result, expected)

    # With missing key:
    idces = range(len(keys))
    expected = np.array([-1] + list(idces), dtype=np.intp)
    missing = tuple([0, 1] * 5 * N)
    result = index.get_indexer([missing] + [keys[i] for i in idces])
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "keys,expected",
    [
        ((slice(None), [5, 4]), [1, 0]),
        ((slice(None), [4, 5]), [0, 1]),
        (([True, False, True], [4, 6]), [0, 2]),
        (([True, False, True], [6, 4]), [0, 2]),
        ((2, [4, 5]), [0, 1]),
        ((2, [5, 4]), [1, 0]),
        (([2], [4, 5]), [0, 1]),
        (([2], [5, 4]), [1, 0]),
    ],
)
def test_get_locs_reordering(keys, expected):
    # GH48384
    idx = MultiIndex.from_arrays(
        [
            [2, 2, 1],
            [4, 5, 6],
        ]
    )
    result = idx.get_locs(keys)
    expected = np.array(expected, dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)


def test_get_indexer_for_multiindex_with_nans(nulls_fixture):
    # GH37222
    idx1 = MultiIndex.from_product([["A"], [1.0, 2.0]], names=["id1", "id2"])
    idx2 = MultiIndex.from_product([["A"], [nulls_fixture, 2.0]], names=["id1", "id2"])

    result = idx2.get_indexer(idx1)
    expected = np.array([-1, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)

    result = idx1.get_indexer(idx2)
    expected = np.array([-1, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tests\test_glsl.py ===
from sympy.core import (pi, symbols, Rational, Integer, GoldenRatio, EulerGamma,
                        Catalan, Lambda, Dummy, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.functions import Piecewise, sin, cos, Abs, exp, ceiling, sqrt
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.printing.glsl import GLSLPrinter
from sympy.printing.str import StrPrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.core import Tuple
from sympy.printing.glsl import glsl_code
import textwrap

x, y, z = symbols('x,y,z')


def test_printmethod():
    assert glsl_code(Abs(x)) == "abs(x)"

def test_print_without_operators():
    assert glsl_code(x*y,use_operators = False) == 'mul(x, y)'
    assert glsl_code(x**y+z,use_operators = False) == 'add(pow(x, y), z)'
    assert glsl_code(x*(y+z),use_operators = False) == 'mul(x, add(y, z))'
    assert glsl_code(x*(y+z),use_operators = False) == 'mul(x, add(y, z))'
    assert glsl_code(x*(y+z**y**0.5),use_operators = False) == 'mul(x, add(y, pow(z, sqrt(y))))'
    assert glsl_code(-x-y, use_operators=False, zero='zero()') == 'sub(zero(), add(x, y))'
    assert glsl_code(-x-y, use_operators=False) == 'sub(0.0, add(x, y))'

def test_glsl_code_sqrt():
    assert glsl_code(sqrt(x)) == "sqrt(x)"
    assert glsl_code(x**0.5) == "sqrt(x)"
    assert glsl_code(sqrt(x)) == "sqrt(x)"


def test_glsl_code_Pow():
    g = implemented_function('g', Lambda(x, 2*x))
    assert glsl_code(x**3) == "pow(x, 3.0)"
    assert glsl_code(x**(y**3)) == "pow(x, pow(y, 3.0))"
    assert glsl_code(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "pow(3.5*2*x, -x + pow(y, x))/(pow(x, 2.0) + y)"
    assert glsl_code(x**-1.0) == '1.0/x'


def test_glsl_code_Relational():
    assert glsl_code(Eq(x, y)) == "x == y"
    assert glsl_code(Ne(x, y)) == "x != y"
    assert glsl_code(Le(x, y)) == "x <= y"
    assert glsl_code(Lt(x, y)) == "x < y"
    assert glsl_code(Gt(x, y)) == "x > y"
    assert glsl_code(Ge(x, y)) == "x >= y"


def test_glsl_code_constants_mathh():
    assert glsl_code(exp(1)) == "float E = 2.71828183;\nE"
    assert glsl_code(pi) == "float pi = 3.14159265;\npi"
    # assert glsl_code(oo) == "Number.POSITIVE_INFINITY"
    # assert glsl_code(-oo) == "Number.NEGATIVE_INFINITY"


def test_glsl_code_constants_other():
    assert glsl_code(2*GoldenRatio) == "float GoldenRatio = 1.61803399;\n2*GoldenRatio"
    assert glsl_code(2*Catalan) == "float Catalan = 0.915965594;\n2*Catalan"
    assert glsl_code(2*EulerGamma) == "float EulerGamma = 0.577215665;\n2*EulerGamma"


def test_glsl_code_Rational():
    assert glsl_code(Rational(3, 7)) == "3.0/7.0"
    assert glsl_code(Rational(18, 9)) == "2"
    assert glsl_code(Rational(3, -7)) == "-3.0/7.0"
    assert glsl_code(Rational(-3, -7)) == "3.0/7.0"


def test_glsl_code_Integer():
    assert glsl_code(Integer(67)) == "67"
    assert glsl_code(Integer(-1)) == "-1"


def test_glsl_code_functions():
    assert glsl_code(sin(x) ** cos(x)) == "pow(sin(x), cos(x))"


def test_glsl_code_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert glsl_code(g(x)) == "2*x"
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    assert glsl_code(g(x)) == "float Catalan = 0.915965594;\n2*x/Catalan"
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    assert glsl_code(g(A[i]), assign_to=A[i]) == (
        "for (int i=0; i<n; i++){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )


def test_glsl_code_exceptions():
    assert glsl_code(ceiling(x)) == "ceil(x)"
    assert glsl_code(Abs(x)) == "abs(x)"


def test_glsl_code_boolean():
    assert glsl_code(x & y) == "x && y"
    assert glsl_code(x | y) == "x || y"
    assert glsl_code(~x) == "!x"
    assert glsl_code(x & y & z) == "x && y && z"
    assert glsl_code(x | y | z) == "x || y || z"
    assert glsl_code((x & y) | z) == "z || x && y"
    assert glsl_code((x | y) & z) == "z && (x || y)"


def test_glsl_code_Piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    p = glsl_code(expr)
    s = \
"""\
((x < 1) ? (
   x
)
: (
   pow(x, 2.0)
))\
"""
    assert p == s
    assert glsl_code(expr, assign_to="c") == (
    "if (x < 1) {\n"
    "   c = x;\n"
    "}\n"
    "else {\n"
    "   c = pow(x, 2.0);\n"
    "}")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: glsl_code(expr))


def test_glsl_code_Piecewise_deep():
    p = glsl_code(2*Piecewise((x, x < 1), (x**2, True)))
    s = \
"""\
2*((x < 1) ? (
   x
)
: (
   pow(x, 2.0)
))\
"""
    assert p == s


def test_glsl_code_settings():
    raises(TypeError, lambda: glsl_code(sin(x), method="garbage"))


def test_glsl_code_Indexed():
    n, m, o = symbols('n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
    p = GLSLPrinter()
    p._not_c = set()

    x = IndexedBase('x')[j]
    assert p._print_Indexed(x) == 'x[j]'
    A = IndexedBase('A')[i, j]
    assert p._print_Indexed(A) == 'A[%s]' % (m*i+j)
    B = IndexedBase('B')[i, j, k]
    assert p._print_Indexed(B) == 'B[%s]' % (i*o*m+j*o+k)

    assert p._not_c == set()

def test_glsl_code_list_tuple_Tuple():
    assert glsl_code([1,2,3,4]) == 'vec4(1, 2, 3, 4)'
    assert glsl_code([1,2,3],glsl_types=False) == 'float[3](1, 2, 3)'
    assert glsl_code([1,2,3]) == glsl_code((1,2,3))
    assert glsl_code([1,2,3]) == glsl_code(Tuple(1,2,3))

    m = MatrixSymbol('A',3,4)
    assert glsl_code([m[0],m[1]])

def test_glsl_code_loops_matrix_vector():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0.0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = A[n*i + j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )

    c = glsl_code(A[i, j]*x[j], assign_to=y[i])
    assert c == s


def test_dummy_loops():
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)

    expected = (
        'for (int i_%(icount)i=0; i_%(icount)i<m_%(mcount)i; i_%(icount)i++){\n'
        '   y[i_%(icount)i] = x[i_%(icount)i];\n'
        '}'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    code = glsl_code(x[i], assign_to=y[i])
    assert code == expected


def test_glsl_code_loops_add():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = x[i] + z[i];\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = A[n*i + j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    c = glsl_code(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i])
    assert c == s


def test_glsl_code_loops_multiple_contractions():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0.0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         for (int l=0; l<p; l++){\n'
        '            y[i] = a[%s]*b[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    c = glsl_code(b[j, k, l]*a[i, j, k, l], assign_to=y[i])
    assert c == s


def test_glsl_code_loops_addfactor():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0.0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         for (int l=0; l<p; l++){\n'
        '            y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    c = glsl_code((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    assert c == s


def test_glsl_code_loops_multiple_terms():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)

    s0 = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0.0;\n'
        '}\n'
    )
    s1 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         y[i] = b[j]*b[k]*c[%s] + y[i];\n' % (i*n*o + j*o + k) +\
        '      }\n'
        '   }\n'
        '}\n'
    )
    s2 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int k=0; k<o; k++){\n'
        '      y[i] = a[%s]*b[k] + y[i];\n' % (i*o + k) +\
        '   }\n'
        '}\n'
    )
    s3 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = a[%s]*b[j] + y[i];\n' % (i*n + j) +\
        '   }\n'
        '}\n'
    )
    c = glsl_code(
        b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i])
    assert (c == s0 + s1 + s2 + s3[:-1] or
            c == s0 + s1 + s3 + s2[:-1] or
            c == s0 + s2 + s1 + s3[:-1] or
            c == s0 + s2 + s3 + s1[:-1] or
            c == s0 + s3 + s1 + s2[:-1] or
            c == s0 + s3 + s2 + s1[:-1])


def test_Matrix_printing():
    # Test returning a Matrix

    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    assert glsl_code(mat, assign_to=A) == (
'''A[0][0] = x*y;
if (y > 0) {
   A[1][0] = x + 2;
}
else {
   A[1][0] = y;
}
A[2][0] = sin(z);''' )
    assert glsl_code(Matrix([A[0],A[1]]))
    # Test using MatrixElements in expressions
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    assert glsl_code(expr) == (
'''((x > 0) ? (
   2*A[2][0]
)
: (
   A[2][0]
)) + sin(A[1][0]) + A[0][0]''' )

    # Test using MatrixElements in a Matrix
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    assert glsl_code(m,M) == (
'''M[0][0] = sin(q[1]);
M[0][1] = 0;
M[0][2] = cos(q[2]);
M[1][0] = q[1] + q[2];
M[1][1] = q[3];
M[1][2] = 5;
M[2][0] = 2*q[4]/q[1];
M[2][1] = sqrt(q[0]) + 4;
M[2][2] = 0;'''
        )

def test_Matrices_1x7():
    gl = glsl_code
    A = Matrix([1,2,3,4,5,6,7])
    assert gl(A) == 'float[7](1, 2, 3, 4, 5, 6, 7)'
    assert gl(A.transpose()) == 'float[7](1, 2, 3, 4, 5, 6, 7)'

def test_Matrices_1x7_array_type_int():
    gl = glsl_code
    A = Matrix([1,2,3,4,5,6,7])
    assert gl(A, array_type='int') == 'int[7](1, 2, 3, 4, 5, 6, 7)'

def test_Tuple_array_type_custom():
    gl = glsl_code
    A = symbols('a b c')
    assert gl(A, array_type='AbcType', glsl_types=False) == 'AbcType[3](a, b, c)'

def test_Matrices_1x7_spread_assign_to_symbols():
    gl = glsl_code
    A = Matrix([1,2,3,4,5,6,7])
    assign_to = symbols('x.a x.b x.c x.d x.e x.f x.g')
    assert gl(A, assign_to=assign_to) == textwrap.dedent('''\
        x.a = 1;
        x.b = 2;
        x.c = 3;
        x.d = 4;
        x.e = 5;
        x.f = 6;
        x.g = 7;'''
    )

def test_spread_assign_to_nested_symbols():
    gl = glsl_code
    expr = ((1,2,3), (1,2,3))
    assign_to = (symbols('a b c'), symbols('x y z'))
    assert gl(expr, assign_to=assign_to) == textwrap.dedent('''\
        a = 1;
        b = 2;
        c = 3;
        x = 1;
        y = 2;
        z = 3;'''
    )

def test_spread_assign_to_deeply_nested_symbols():
    gl = glsl_code
    a, b, c, x, y, z = symbols('a b c x y z')
    expr = (((1,2),3), ((1,2),3))
    assign_to = (((a, b), c), ((x, y), z))
    assert gl(expr, assign_to=assign_to) == textwrap.dedent('''\
        a = 1;
        b = 2;
        c = 3;
        x = 1;
        y = 2;
        z = 3;'''
    )

def test_matrix_of_tuples_spread_assign_to_symbols():
    gl = glsl_code
    with warns_deprecated_sympy():
        expr = Matrix([[(1,2),(3,4)],[(5,6),(7,8)]])
    assign_to = (symbols('a b'), symbols('c d'), symbols('e f'), symbols('g h'))
    assert gl(expr, assign_to) == textwrap.dedent('''\
        a = 1;
        b = 2;
        c = 3;
        d = 4;
        e = 5;
        f = 6;
        g = 7;
        h = 8;'''
    )

def test_cannot_assign_to_cause_mismatched_length():
    expr = (1, 2)
    assign_to = symbols('x y z')
    raises(ValueError, lambda: glsl_code(expr, assign_to))

def test_matrix_4x4_assign():
    gl = glsl_code
    expr = MatrixSymbol('A',4,4) * MatrixSymbol('B',4,4) + MatrixSymbol('C',4,4)
    assign_to = MatrixSymbol('X',4,4)
    assert gl(expr, assign_to=assign_to) == textwrap.dedent('''\
        X[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0] + C[0][0];
        X[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1] + A[0][3]*B[3][1] + C[0][1];
        X[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2] + A[0][3]*B[3][2] + C[0][2];
        X[0][3] = A[0][0]*B[0][3] + A[0][1]*B[1][3] + A[0][2]*B[2][3] + A[0][3]*B[3][3] + C[0][3];
        X[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0] + A[1][3]*B[3][0] + C[1][0];
        X[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1] + A[1][3]*B[3][1] + C[1][1];
        X[1][2] = A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2] + A[1][3]*B[3][2] + C[1][2];
        X[1][3] = A[1][0]*B[0][3] + A[1][1]*B[1][3] + A[1][2]*B[2][3] + A[1][3]*B[3][3] + C[1][3];
        X[2][0] = A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0] + A[2][3]*B[3][0] + C[2][0];
        X[2][1] = A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1] + A[2][3]*B[3][1] + C[2][1];
        X[2][2] = A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2] + A[2][3]*B[3][2] + C[2][2];
        X[2][3] = A[2][0]*B[0][3] + A[2][1]*B[1][3] + A[2][2]*B[2][3] + A[2][3]*B[3][3] + C[2][3];
        X[3][0] = A[3][0]*B[0][0] + A[3][1]*B[1][0] + A[3][2]*B[2][0] + A[3][3]*B[3][0] + C[3][0];
        X[3][1] = A[3][0]*B[0][1] + A[3][1]*B[1][1] + A[3][2]*B[2][1] + A[3][3]*B[3][1] + C[3][1];
        X[3][2] = A[3][0]*B[0][2] + A[3][1]*B[1][2] + A[3][2]*B[2][2] + A[3][3]*B[3][2] + C[3][2];
        X[3][3] = A[3][0]*B[0][3] + A[3][1]*B[1][3] + A[3][2]*B[2][3] + A[3][3]*B[3][3] + C[3][3];'''
    )

def test_1xN_vecs():
    gl = glsl_code
    for i in range(1,10):
        A = Matrix(range(i))
        assert gl(A.transpose()) == gl(A)
        assert gl(A,mat_transpose=True) == gl(A)
        if i > 1:
            if i <= 4:
                assert gl(A) == 'vec%s(%s)' % (i,', '.join(str(s) for s in range(i)))
            else:
                assert gl(A) == 'float[%s](%s)' % (i,', '.join(str(s) for s in range(i)))

def test_MxN_mats():
    generatedAssertions='def test_misc_mats():\n'
    for i in range(1,6):
        for j in range(1,6):
            A = Matrix([[x + y*j for x in range(j)] for y in range(i)])
            gl = glsl_code(A)
            glTransposed = glsl_code(A,mat_transpose=True)
            generatedAssertions+='    mat = '+StrPrinter()._print(A)+'\n\n'
            generatedAssertions+='    gl = \'\'\''+gl+'\'\'\'\n'
            generatedAssertions+='    glTransposed = \'\'\''+glTransposed+'\'\'\'\n\n'
            generatedAssertions+='    assert glsl_code(mat) == gl\n'
            generatedAssertions+='    assert glsl_code(mat,mat_transpose=True) == glTransposed\n'
            if i == 1 and j == 1:
                assert gl == '0'
            elif i <= 4 and j <= 4 and i>1 and j>1:
                assert gl.startswith('mat%s' % j)
                assert glTransposed.startswith('mat%s' % i)
            elif i == 1 and j <= 4:
                assert gl.startswith('vec')
            elif j == 1 and i <= 4:
                assert gl.startswith('vec')
            elif i == 1:
                assert gl.startswith('float[%s]('% j*i)
                assert glTransposed.startswith('float[%s]('% j*i)
            elif j == 1:
                assert gl.startswith('float[%s]('% i*j)
                assert glTransposed.startswith('float[%s]('% i*j)
            else:
                assert gl.startswith('float[%s](' % (i*j))
                assert glTransposed.startswith('float[%s](' % (i*j))
                glNested = glsl_code(A,mat_nested=True)
                glNestedTransposed = glsl_code(A,mat_transpose=True,mat_nested=True)
                assert glNested.startswith('float[%s][%s]' % (i,j))
                assert glNestedTransposed.startswith('float[%s][%s]' % (j,i))
                generatedAssertions+='    glNested = \'\'\''+glNested+'\'\'\'\n'
                generatedAssertions+='    glNestedTransposed = \'\'\''+glNestedTransposed+'\'\'\'\n\n'
                generatedAssertions+='    assert glsl_code(mat,mat_nested=True) == glNested\n'
                generatedAssertions+='    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed\n\n'
    generateAssertions = False # set this to true to write bake these generated tests to a file
    if generateAssertions:
        gen = open('test_glsl_generated_matrices.py','w')
        gen.write(generatedAssertions)
        gen.close()


# these assertions were generated from the previous function
# glsl has complicated rules and this makes it easier to look over all the cases
def test_misc_mats():

    mat = Matrix([[0]])

    gl = '''0'''
    glTransposed = '''0'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([[0, 1]])

    gl = '''vec2(0, 1)'''
    glTransposed = '''vec2(0, 1)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([[0, 1, 2]])

    gl = '''vec3(0, 1, 2)'''
    glTransposed = '''vec3(0, 1, 2)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([[0, 1, 2, 3]])

    gl = '''vec4(0, 1, 2, 3)'''
    glTransposed = '''vec4(0, 1, 2, 3)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([[0, 1, 2, 3, 4]])

    gl = '''float[5](0, 1, 2, 3, 4)'''
    glTransposed = '''float[5](0, 1, 2, 3, 4)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0],
[1]])

    gl = '''vec2(0, 1)'''
    glTransposed = '''vec2(0, 1)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1],
[2, 3]])

    gl = '''mat2(0, 1, 2, 3)'''
    glTransposed = '''mat2(0, 2, 1, 3)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1, 2],
[3, 4, 5]])

    gl = '''mat3x2(0, 1, 2, 3, 4, 5)'''
    glTransposed = '''mat2x3(0, 3, 1, 4, 2, 5)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1, 2, 3],
[4, 5, 6, 7]])

    gl = '''mat4x2(0, 1, 2, 3, 4, 5, 6, 7)'''
    glTransposed = '''mat2x4(0, 4, 1, 5, 2, 6, 3, 7)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1, 2, 3, 4],
[5, 6, 7, 8, 9]])

    gl = '''float[10](
   0, 1, 2, 3, 4,
   5, 6, 7, 8, 9
) /* a 2x5 matrix */'''
    glTransposed = '''float[10](
   0, 5,
   1, 6,
   2, 7,
   3, 8,
   4, 9
) /* a 5x2 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[2][5](
   float[](0, 1, 2, 3, 4),
   float[](5, 6, 7, 8, 9)
)'''
    glNestedTransposed = '''float[5][2](
   float[](0, 5),
   float[](1, 6),
   float[](2, 7),
   float[](3, 8),
   float[](4, 9)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[0],
[1],
[2]])

    gl = '''vec3(0, 1, 2)'''
    glTransposed = '''vec3(0, 1, 2)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1],
[2, 3],
[4, 5]])

    gl = '''mat2x3(0, 1, 2, 3, 4, 5)'''
    glTransposed = '''mat3x2(0, 2, 4, 1, 3, 5)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1, 2],
[3, 4, 5],
[6, 7, 8]])

    gl = '''mat3(0, 1, 2, 3, 4, 5, 6, 7, 8)'''
    glTransposed = '''mat3(0, 3, 6, 1, 4, 7, 2, 5, 8)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1,  2,  3],
[4, 5,  6,  7],
[8, 9, 10, 11]])

    gl = '''mat4x3(0, 1,  2,  3, 4, 5,  6,  7, 8, 9, 10, 11)'''
    glTransposed = '''mat3x4(0, 4,  8, 1, 5,  9, 2, 6, 10, 3, 7, 11)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[ 0,  1,  2,  3,  4],
[ 5,  6,  7,  8,  9],
[10, 11, 12, 13, 14]])

    gl = '''float[15](
   0,  1,  2,  3,  4,
   5,  6,  7,  8,  9,
   10, 11, 12, 13, 14
) /* a 3x5 matrix */'''
    glTransposed = '''float[15](
   0, 5, 10,
   1, 6, 11,
   2, 7, 12,
   3, 8, 13,
   4, 9, 14
) /* a 5x3 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[3][5](
   float[]( 0,  1,  2,  3,  4),
   float[]( 5,  6,  7,  8,  9),
   float[](10, 11, 12, 13, 14)
)'''
    glNestedTransposed = '''float[5][3](
   float[](0, 5, 10),
   float[](1, 6, 11),
   float[](2, 7, 12),
   float[](3, 8, 13),
   float[](4, 9, 14)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[0],
[1],
[2],
[3]])

    gl = '''vec4(0, 1, 2, 3)'''
    glTransposed = '''vec4(0, 1, 2, 3)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1],
[2, 3],
[4, 5],
[6, 7]])

    gl = '''mat2x4(0, 1, 2, 3, 4, 5, 6, 7)'''
    glTransposed = '''mat4x2(0, 2, 4, 6, 1, 3, 5, 7)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0,  1,  2],
[3,  4,  5],
[6,  7,  8],
[9, 10, 11]])

    gl = '''mat3x4(0,  1,  2, 3,  4,  5, 6,  7,  8, 9, 10, 11)'''
    glTransposed = '''mat4x3(0, 3, 6,  9, 1, 4, 7, 10, 2, 5, 8, 11)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11],
[12, 13, 14, 15]])

    gl = '''mat4( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15)'''
    glTransposed = '''mat4(0, 4,  8, 12, 1, 5,  9, 13, 2, 6, 10, 14, 3, 7, 11, 15)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[ 0,  1,  2,  3,  4],
[ 5,  6,  7,  8,  9],
[10, 11, 12, 13, 14],
[15, 16, 17, 18, 19]])

    gl = '''float[20](
   0,  1,  2,  3,  4,
   5,  6,  7,  8,  9,
   10, 11, 12, 13, 14,
   15, 16, 17, 18, 19
) /* a 4x5 matrix */'''
    glTransposed = '''float[20](
   0, 5, 10, 15,
   1, 6, 11, 16,
   2, 7, 12, 17,
   3, 8, 13, 18,
   4, 9, 14, 19
) /* a 5x4 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[4][5](
   float[]( 0,  1,  2,  3,  4),
   float[]( 5,  6,  7,  8,  9),
   float[](10, 11, 12, 13, 14),
   float[](15, 16, 17, 18, 19)
)'''
    glNestedTransposed = '''float[5][4](
   float[](0, 5, 10, 15),
   float[](1, 6, 11, 16),
   float[](2, 7, 12, 17),
   float[](3, 8, 13, 18),
   float[](4, 9, 14, 19)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[0],
[1],
[2],
[3],
[4]])

    gl = '''float[5](0, 1, 2, 3, 4)'''
    glTransposed = '''float[5](0, 1, 2, 3, 4)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1],
[2, 3],
[4, 5],
[6, 7],
[8, 9]])

    gl = '''float[10](
   0, 1,
   2, 3,
   4, 5,
   6, 7,
   8, 9
) /* a 5x2 matrix */'''
    glTransposed = '''float[10](
   0, 2, 4, 6, 8,
   1, 3, 5, 7, 9
) /* a 2x5 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[5][2](
   float[](0, 1),
   float[](2, 3),
   float[](4, 5),
   float[](6, 7),
   float[](8, 9)
)'''
    glNestedTransposed = '''float[2][5](
   float[](0, 2, 4, 6, 8),
   float[](1, 3, 5, 7, 9)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[ 0,  1,  2],
[ 3,  4,  5],
[ 6,  7,  8],
[ 9, 10, 11],
[12, 13, 14]])

    gl = '''float[15](
   0,  1,  2,
   3,  4,  5,
   6,  7,  8,
   9, 10, 11,
   12, 13, 14
) /* a 5x3 matrix */'''
    glTransposed = '''float[15](
   0, 3, 6,  9, 12,
   1, 4, 7, 10, 13,
   2, 5, 8, 11, 14
) /* a 3x5 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[5][3](
   float[]( 0,  1,  2),
   float[]( 3,  4,  5),
   float[]( 6,  7,  8),
   float[]( 9, 10, 11),
   float[](12, 13, 14)
)'''
    glNestedTransposed = '''float[3][5](
   float[](0, 3, 6,  9, 12),
   float[](1, 4, 7, 10, 13),
   float[](2, 5, 8, 11, 14)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11],
[12, 13, 14, 15],
[16, 17, 18, 19]])

    gl = '''float[20](
   0,  1,  2,  3,
   4,  5,  6,  7,
   8,  9, 10, 11,
   12, 13, 14, 15,
   16, 17, 18, 19
) /* a 5x4 matrix */'''
    glTransposed = '''float[20](
   0, 4,  8, 12, 16,
   1, 5,  9, 13, 17,
   2, 6, 10, 14, 18,
   3, 7, 11, 15, 19
) /* a 4x5 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[5][4](
   float[]( 0,  1,  2,  3),
   float[]( 4,  5,  6,  7),
   float[]( 8,  9, 10, 11),
   float[](12, 13, 14, 15),
   float[](16, 17, 18, 19)
)'''
    glNestedTransposed = '''float[4][5](
   float[](0, 4,  8, 12, 16),
   float[](1, 5,  9, 13, 17),
   float[](2, 6, 10, 14, 18),
   float[](3, 7, 11, 15, 19)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[ 0,  1,  2,  3,  4],
[ 5,  6,  7,  8,  9],
[10, 11, 12, 13, 14],
[15, 16, 17, 18, 19],
[20, 21, 22, 23, 24]])

    gl = '''float[25](
   0,  1,  2,  3,  4,
   5,  6,  7,  8,  9,
   10, 11, 12, 13, 14,
   15, 16, 17, 18, 19,
   20, 21, 22, 23, 24
) /* a 5x5 matrix */'''
    glTransposed = '''float[25](
   0, 5, 10, 15, 20,
   1, 6, 11, 16, 21,
   2, 7, 12, 17, 22,
   3, 8, 13, 18, 23,
   4, 9, 14, 19, 24
) /* a 5x5 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[5][5](
   float[]( 0,  1,  2,  3,  4),
   float[]( 5,  6,  7,  8,  9),
   float[](10, 11, 12, 13, 14),
   float[](15, 16, 17, 18, 19),
   float[](20, 21, 22, 23, 24)
)'''
    glNestedTransposed = '''float[5][5](
   float[](0, 5, 10, 15, 20),
   float[](1, 6, 11, 16, 21),
   float[](2, 7, 12, 17, 22),
   float[](3, 8, 13, 18, 23),
   float[](4, 9, 14, 19, 24)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\multiindex\test_loc.py ===
import numpy as np
import pytest

from pandas.errors import (
    IndexingError,
    PerformanceWarning,
)

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


@pytest.fixture
def single_level_multiindex():
    """single level MultiIndex"""
    return MultiIndex(
        levels=[["foo", "bar", "baz", "qux"]], codes=[[0, 1, 2, 3]], names=["first"]
    )


@pytest.fixture
def frame_random_data_integer_multi_index():
    levels = [[0, 1], [0, 1, 2]]
    codes = [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]]
    index = MultiIndex(levels=levels, codes=codes)
    return DataFrame(np.random.default_rng(2).standard_normal((6, 2)), index=index)


class TestMultiIndexLoc:
    def test_loc_setitem_frame_with_multiindex(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data
        frame.loc[("bar", "two"), "B"] = 5
        assert frame.loc[("bar", "two"), "B"] == 5

        # with integer labels
        df = frame.copy()
        df.columns = list(range(3))
        df.loc[("bar", "two"), 1] = 7
        assert df.loc[("bar", "two"), 1] == 7

    def test_loc_getitem_general(self, any_real_numpy_dtype):
        # GH#2817
        dtype = any_real_numpy_dtype
        data = {
            "amount": {0: 700, 1: 600, 2: 222, 3: 333, 4: 444},
            "col": {0: 3.5, 1: 3.5, 2: 4.0, 3: 4.0, 4: 4.0},
            "num": {0: 12, 1: 11, 2: 12, 3: 12, 4: 12},
        }
        df = DataFrame(data)
        df = df.astype({"col": dtype, "num": dtype})
        df = df.set_index(keys=["col", "num"])
        key = 4.0, 12

        # emits a PerformanceWarning, ok
        with tm.assert_produces_warning(PerformanceWarning):
            tm.assert_frame_equal(df.loc[key], df.iloc[2:])

        # this is ok
        return_value = df.sort_index(inplace=True)
        assert return_value is None
        res = df.loc[key]

        # col has float dtype, result should be float64 Index
        col_arr = np.array([4.0] * 3, dtype=dtype)
        year_arr = np.array([12] * 3, dtype=dtype)
        index = MultiIndex.from_arrays([col_arr, year_arr], names=["col", "num"])
        expected = DataFrame({"amount": [222, 333, 444]}, index=index)
        tm.assert_frame_equal(res, expected)

    def test_loc_getitem_multiindex_missing_label_raises(self):
        # GH#21593
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=[[2, 2, 4], [6, 8, 10]],
            index=[[4, 4, 8], [8, 10, 12]],
        )

        with pytest.raises(KeyError, match=r"^2$"):
            df.loc[2]

    def test_loc_getitem_list_of_tuples_with_multiindex(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        ser = multiindex_year_month_day_dataframe_random_data["A"]
        expected = ser.reindex(ser.index[49:51])
        result = ser.loc[[(2000, 3, 10), (2000, 3, 13)]]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_series(self):
        # GH14730
        # passing a series as a key with a MultiIndex
        index = MultiIndex.from_product([[1, 2, 3], ["A", "B", "C"]])
        x = Series(index=index, data=range(9), dtype=np.float64)
        y = Series([1, 3])
        expected = Series(
            data=[0, 1, 2, 6, 7, 8],
            index=MultiIndex.from_product([[1, 3], ["A", "B", "C"]]),
            dtype=np.float64,
        )
        result = x.loc[y]
        tm.assert_series_equal(result, expected)

        result = x.loc[[1, 3]]
        tm.assert_series_equal(result, expected)

        # GH15424
        y1 = Series([1, 3], index=[1, 2])
        result = x.loc[y1]
        tm.assert_series_equal(result, expected)

        empty = Series(data=[], dtype=np.float64)
        expected = Series(
            [],
            index=MultiIndex(levels=index.levels, codes=[[], []], dtype=np.float64),
            dtype=np.float64,
        )
        result = x.loc[empty]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_array(self):
        # GH15434
        # passing an array as a key with a MultiIndex
        index = MultiIndex.from_product([[1, 2, 3], ["A", "B", "C"]])
        x = Series(index=index, data=range(9), dtype=np.float64)
        y = np.array([1, 3])
        expected = Series(
            data=[0, 1, 2, 6, 7, 8],
            index=MultiIndex.from_product([[1, 3], ["A", "B", "C"]]),
            dtype=np.float64,
        )
        result = x.loc[y]
        tm.assert_series_equal(result, expected)

        # empty array:
        empty = np.array([])
        expected = Series(
            [],
            index=MultiIndex(levels=index.levels, codes=[[], []], dtype=np.float64),
            dtype="float64",
        )
        result = x.loc[empty]
        tm.assert_series_equal(result, expected)

        # 0-dim array (scalar):
        scalar = np.int64(1)
        expected = Series(data=[0, 1, 2], index=["A", "B", "C"], dtype=np.float64)
        result = x.loc[scalar]
        tm.assert_series_equal(result, expected)

    def test_loc_multiindex_labels(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=[["i", "i", "j"], ["A", "A", "B"]],
            index=[["i", "i", "j"], ["X", "X", "Y"]],
        )

        # the first 2 rows
        expected = df.iloc[[0, 1]].droplevel(0)
        result = df.loc["i"]
        tm.assert_frame_equal(result, expected)

        # 2nd (last) column
        expected = df.iloc[:, [2]].droplevel(0, axis=1)
        result = df.loc[:, "j"]
        tm.assert_frame_equal(result, expected)

        # bottom right corner
        expected = df.iloc[[2], [2]].droplevel(0).droplevel(0, axis=1)
        result = df.loc["j"].loc[:, "j"]
        tm.assert_frame_equal(result, expected)

        # with a tuple
        expected = df.iloc[[0, 1]]
        result = df.loc[("i", "X")]
        tm.assert_frame_equal(result, expected)

    def test_loc_multiindex_ints(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=[[2, 2, 4], [6, 8, 10]],
            index=[[4, 4, 8], [8, 10, 12]],
        )
        expected = df.iloc[[0, 1]].droplevel(0)
        result = df.loc[4]
        tm.assert_frame_equal(result, expected)

    def test_loc_multiindex_missing_label_raises(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=[[2, 2, 4], [6, 8, 10]],
            index=[[4, 4, 8], [8, 10, 12]],
        )

        with pytest.raises(KeyError, match=r"^2$"):
            df.loc[2]

    @pytest.mark.parametrize("key, pos", [([2, 4], [0, 1]), ([2], []), ([2, 3], [])])
    def test_loc_multiindex_list_missing_label(self, key, pos):
        # GH 27148 - lists with missing labels _do_ raise
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=[[2, 2, 4], [6, 8, 10]],
            index=[[4, 4, 8], [8, 10, 12]],
        )

        with pytest.raises(KeyError, match="not in index"):
            df.loc[key]

    def test_loc_multiindex_too_many_dims_raises(self):
        # GH 14885
        s = Series(
            range(8),
            index=MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]]),
        )

        with pytest.raises(KeyError, match=r"^\('a', 'b'\)$"):
            s.loc["a", "b"]
        with pytest.raises(KeyError, match=r"^\('a', 'd', 'g'\)$"):
            s.loc["a", "d", "g"]
        with pytest.raises(IndexingError, match="Too many indexers"):
            s.loc["a", "d", "g", "j"]

    def test_loc_multiindex_indexer_none(self):
        # GH6788
        # multi-index indexer is None (meaning take all)
        attributes = ["Attribute" + str(i) for i in range(1)]
        attribute_values = ["Value" + str(i) for i in range(5)]

        index = MultiIndex.from_product([attributes, attribute_values])
        df = 0.1 * np.random.default_rng(2).standard_normal((10, 1 * 5)) + 0.5
        df = DataFrame(df, columns=index)
        result = df[attributes]
        tm.assert_frame_equal(result, df)

        # GH 7349
        # loc with a multi-index seems to be doing fallback
        df = DataFrame(
            np.arange(12).reshape(-1, 1),
            index=MultiIndex.from_product([[1, 2, 3, 4], [1, 2, 3]]),
        )

        expected = df.loc[([1, 2],), :]
        result = df.loc[[1, 2]]
        tm.assert_frame_equal(result, expected)

    def test_loc_multiindex_incomplete(self):
        # GH 7399
        # incomplete indexers
        s = Series(
            np.arange(15, dtype="int64"),
            MultiIndex.from_product([range(5), ["a", "b", "c"]]),
        )
        expected = s.loc[:, "a":"c"]

        result = s.loc[0:4, "a":"c"]
        tm.assert_series_equal(result, expected)

        result = s.loc[:4, "a":"c"]
        tm.assert_series_equal(result, expected)

        result = s.loc[0:, "a":"c"]
        tm.assert_series_equal(result, expected)

        # GH 7400
        # multiindexer getitem with list of indexers skips wrong element
        s = Series(
            np.arange(15, dtype="int64"),
            MultiIndex.from_product([range(5), ["a", "b", "c"]]),
        )
        expected = s.iloc[[6, 7, 8, 12, 13, 14]]
        result = s.loc[2:4:2, "a":"c"]
        tm.assert_series_equal(result, expected)

    def test_get_loc_single_level(self, single_level_multiindex):
        single_level = single_level_multiindex
        s = Series(
            np.random.default_rng(2).standard_normal(len(single_level)),
            index=single_level,
        )
        for k in single_level.values:
            s[k]

    def test_loc_getitem_int_slice(self):
        # GH 3053
        # loc should treat integer slices like label slices

        index = MultiIndex.from_product([[6, 7, 8], ["a", "b"]])
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 6)), index, index)
        result = df.loc[6:8, :]
        expected = df
        tm.assert_frame_equal(result, expected)

        index = MultiIndex.from_product([[10, 20, 30], ["a", "b"]])
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 6)), index, index)
        result = df.loc[20:30, :]
        expected = df.iloc[2:]
        tm.assert_frame_equal(result, expected)

        # doc examples
        result = df.loc[10, :]
        expected = df.iloc[0:2]
        expected.index = ["a", "b"]
        tm.assert_frame_equal(result, expected)

        result = df.loc[:, 10]
        expected = df[10]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "indexer_type_1", (list, tuple, set, slice, np.ndarray, Series, Index)
    )
    @pytest.mark.parametrize(
        "indexer_type_2", (list, tuple, set, slice, np.ndarray, Series, Index)
    )
    def test_loc_getitem_nested_indexer(self, indexer_type_1, indexer_type_2):
        # GH #19686
        # .loc should work with nested indexers which can be
        # any list-like objects (see `is_list_like` (`pandas.api.types`)) or slices

        def convert_nested_indexer(indexer_type, keys):
            if indexer_type == np.ndarray:
                return np.array(keys)
            if indexer_type == slice:
                return slice(*keys)
            return indexer_type(keys)

        a = [10, 20, 30]
        b = [1, 2, 3]
        index = MultiIndex.from_product([a, b])
        df = DataFrame(
            np.arange(len(index), dtype="int64"), index=index, columns=["Data"]
        )

        keys = ([10, 20], [2, 3])
        types = (indexer_type_1, indexer_type_2)

        # check indexers with all the combinations of nested objects
        # of all the valid types
        indexer = tuple(
            convert_nested_indexer(indexer_type, k)
            for indexer_type, k in zip(types, keys)
        )
        if indexer_type_1 is set or indexer_type_2 is set:
            with pytest.raises(TypeError, match="as an indexer is not supported"):
                df.loc[indexer, "Data"]

            return
        else:
            result = df.loc[indexer, "Data"]
        expected = Series(
            [1, 2, 4, 5], name="Data", index=MultiIndex.from_product(keys)
        )

        tm.assert_series_equal(result, expected)

    def test_multiindex_loc_one_dimensional_tuple(self, frame_or_series):
        # GH#37711
        mi = MultiIndex.from_tuples([("a", "A"), ("b", "A")])
        obj = frame_or_series([1, 2], index=mi)
        obj.loc[("a",)] = 0
        expected = frame_or_series([0, 2], index=mi)
        tm.assert_equal(obj, expected)

    @pytest.mark.parametrize("indexer", [("a",), ("a")])
    def test_multiindex_one_dimensional_tuple_columns(self, indexer):
        # GH#37711
        mi = MultiIndex.from_tuples([("a", "A"), ("b", "A")])
        obj = DataFrame([1, 2], index=mi)
        obj.loc[indexer, :] = 0
        expected = DataFrame([0, 2], index=mi)
        tm.assert_frame_equal(obj, expected)

    @pytest.mark.parametrize(
        "indexer, exp_value", [(slice(None), 1.0), ((1, 2), np.nan)]
    )
    def test_multiindex_setitem_columns_enlarging(self, indexer, exp_value):
        # GH#39147
        mi = MultiIndex.from_tuples([(1, 2), (3, 4)])
        df = DataFrame([[1, 2], [3, 4]], index=mi, columns=["a", "b"])
        df.loc[indexer, ["c", "d"]] = 1.0
        expected = DataFrame(
            [[1, 2, 1.0, 1.0], [3, 4, exp_value, exp_value]],
            index=mi,
            columns=["a", "b", "c", "d"],
        )
        tm.assert_frame_equal(df, expected)

    def test_sorted_multiindex_after_union(self):
        # GH#44752
        midx = MultiIndex.from_product(
            [pd.date_range("20110101", periods=2), Index(["a", "b"])]
        )
        ser1 = Series(1, index=midx)
        ser2 = Series(1, index=midx[:2])
        df = pd.concat([ser1, ser2], axis=1)
        expected = df.copy()
        result = df.loc["2011-01-01":"2011-01-02"]
        tm.assert_frame_equal(result, expected)

        df = DataFrame({0: ser1, 1: ser2})
        result = df.loc["2011-01-01":"2011-01-02"]
        tm.assert_frame_equal(result, expected)

        df = pd.concat([ser1, ser2.reindex(ser1.index)], axis=1)
        result = df.loc["2011-01-01":"2011-01-02"]
        tm.assert_frame_equal(result, expected)

    def test_loc_no_second_level_index(self):
        # GH#43599
        df = DataFrame(
            index=MultiIndex.from_product([list("ab"), list("cd"), list("e")]),
            columns=["Val"],
        )
        res = df.loc[np.s_[:, "c", :]]
        expected = DataFrame(
            index=MultiIndex.from_product([list("ab"), list("e")]), columns=["Val"]
        )
        tm.assert_frame_equal(res, expected)

    def test_loc_multi_index_key_error(self):
        # GH 51892
        df = DataFrame(
            {
                (1, 2): ["a", "b", "c"],
                (1, 3): ["d", "e", "f"],
                (2, 2): ["g", "h", "i"],
                (2, 4): ["j", "k", "l"],
            }
        )
        with pytest.raises(KeyError, match=r"(1, 4)"):
            df.loc[0, (1, 4)]


@pytest.mark.parametrize(
    "indexer, pos",
    [
        ([], []),  # empty ok
        (["A"], slice(3)),
        (["A", "D"], []),  # "D" isn't present -> raise
        (["D", "E"], []),  # no values found -> raise
        (["D"], []),  # same, with single item list: GH 27148
        (pd.IndexSlice[:, ["foo"]], slice(2, None, 3)),
        (pd.IndexSlice[:, ["foo", "bah"]], slice(2, None, 3)),
    ],
)
def test_loc_getitem_duplicates_multiindex_missing_indexers(indexer, pos):
    # GH 7866
    # multi-index slicing with missing indexers
    idx = MultiIndex.from_product(
        [["A", "B", "C"], ["foo", "bar", "baz"]], names=["one", "two"]
    )
    ser = Series(np.arange(9, dtype="int64"), index=idx).sort_index()
    expected = ser.iloc[pos]

    if expected.size == 0 and indexer != []:
        with pytest.raises(KeyError, match=str(indexer)):
            ser.loc[indexer]
    elif indexer == (slice(None), ["foo", "bah"]):
        # "bah" is not in idx.levels[1], raising KeyError enforced in 2.0
        with pytest.raises(KeyError, match="'bah'"):
            ser.loc[indexer]
    else:
        result = ser.loc[indexer]
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns_indexer", [([], slice(None)), (["foo"], [])])
def test_loc_getitem_duplicates_multiindex_empty_indexer(columns_indexer):
    # GH 8737
    # empty indexer
    multi_index = MultiIndex.from_product((["foo", "bar", "baz"], ["alpha", "beta"]))
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 6)),
        index=range(5),
        columns=multi_index,
    )
    df = df.sort_index(level=0, axis=1)

    expected = DataFrame(index=range(5), columns=multi_index.reindex([])[0])
    result = df.loc[:, columns_indexer]
    tm.assert_frame_equal(result, expected)


def test_loc_getitem_duplicates_multiindex_non_scalar_type_object():
    # regression from < 0.14.0
    # GH 7914
    df = DataFrame(
        [[np.mean, np.median], ["mean", "median"]],
        columns=MultiIndex.from_tuples([("functs", "mean"), ("functs", "median")]),
        index=["function", "name"],
    )
    result = df.loc["function", ("functs", "mean")]
    expected = np.mean
    assert result == expected


def test_loc_getitem_tuple_plus_slice():
    # GH 671
    df = DataFrame(
        {
            "a": np.arange(10),
            "b": np.arange(10),
            "c": np.random.default_rng(2).standard_normal(10),
            "d": np.random.default_rng(2).standard_normal(10),
        }
    ).set_index(["a", "b"])
    expected = df.loc[0, 0]
    result = df.loc[(0, 0), :]
    tm.assert_series_equal(result, expected)


def test_loc_getitem_int(frame_random_data_integer_multi_index):
    df = frame_random_data_integer_multi_index
    result = df.loc[1]
    expected = df[-3:]
    expected.index = expected.index.droplevel(0)
    tm.assert_frame_equal(result, expected)


def test_loc_getitem_int_raises_exception(frame_random_data_integer_multi_index):
    df = frame_random_data_integer_multi_index
    with pytest.raises(KeyError, match=r"^3$"):
        df.loc[3]


def test_loc_getitem_lowerdim_corner(multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data

    # test setup - check key not in dataframe
    with pytest.raises(KeyError, match=r"^\('bar', 'three'\)$"):
        df.loc[("bar", "three"), "B"]

    # in theory should be inserting in a sorted space????
    df.loc[("bar", "three"), "B"] = 0
    expected = 0
    result = df.sort_index().loc[("bar", "three"), "B"]
    assert result == expected


def test_loc_setitem_single_column_slice():
    # case from https://github.com/pandas-dev/pandas/issues/27841
    df = DataFrame(
        "string",
        index=list("abcd"),
        columns=MultiIndex.from_product([["Main"], ("another", "one")]),
    )
    df["labels"] = "a"
    df.loc[:, "labels"] = df.index
    tm.assert_numpy_array_equal(np.asarray(df["labels"]), np.asarray(df.index))

    # test with non-object block
    df = DataFrame(
        np.nan,
        index=range(4),
        columns=MultiIndex.from_tuples([("A", "1"), ("A", "2"), ("B", "1")]),
    )
    expected = df.copy()
    df.loc[:, "B"] = np.arange(4)
    expected.iloc[:, 2] = np.arange(4)
    tm.assert_frame_equal(df, expected)


def test_loc_nan_multiindex(using_infer_string):
    # GH 5286
    tups = [
        ("Good Things", "C", np.nan),
        ("Good Things", "R", np.nan),
        ("Bad Things", "C", np.nan),
        ("Bad Things", "T", np.nan),
        ("Okay Things", "N", "B"),
        ("Okay Things", "N", "D"),
        ("Okay Things", "B", np.nan),
        ("Okay Things", "D", np.nan),
    ]
    df = DataFrame(
        np.ones((8, 4)),
        columns=Index(["d1", "d2", "d3", "d4"]),
        index=MultiIndex.from_tuples(tups, names=["u1", "u2", "u3"]),
    )
    result = df.loc["Good Things"].loc["C"]
    expected = DataFrame(
        np.ones((1, 4)),
        index=Index(
            [np.nan],
            dtype="object" if not using_infer_string else "str",
            name="u3",
        ),
        columns=Index(["d1", "d2", "d3", "d4"]),
    )
    tm.assert_frame_equal(result, expected)


def test_loc_period_string_indexing():
    # GH 9892
    a = pd.period_range("2013Q1", "2013Q4", freq="Q")
    i = (1111, 2222, 3333)
    idx = MultiIndex.from_product((a, i), names=("Period", "CVR"))
    df = DataFrame(
        index=idx,
        columns=(
            "OMS",
            "OMK",
            "RES",
            "DRIFT_IND",
            "OEVRIG_IND",
            "FIN_IND",
            "VARE_UD",
            "LOEN_UD",
            "FIN_UD",
        ),
    )
    result = df.loc[("2013Q1", 1111), "OMS"]

    alt = df.loc[(a[0], 1111), "OMS"]
    assert np.isnan(alt)

    # Because the resolution of the string matches, it is an exact lookup,
    #  not a slice
    assert np.isnan(result)

    alt = df.loc[("2013Q1", 1111), "OMS"]
    assert np.isnan(alt)


def test_loc_datetime_mask_slicing():
    # GH 16699
    dt_idx = pd.to_datetime(["2017-05-04", "2017-05-05"])
    m_idx = MultiIndex.from_product([dt_idx, dt_idx], names=["Idx1", "Idx2"])
    df = DataFrame(
        data=[[1, 2], [3, 4], [5, 6], [7, 6]], index=m_idx, columns=["C1", "C2"]
    )
    result = df.loc[(dt_idx[0], (df.index.get_level_values(1) > "2017-05-04")), "C1"]
    expected = Series(
        [3],
        name="C1",
        index=MultiIndex.from_tuples(
            [(pd.Timestamp("2017-05-04"), pd.Timestamp("2017-05-05"))],
            names=["Idx1", "Idx2"],
        ),
    )
    tm.assert_series_equal(result, expected)


def test_loc_datetime_series_tuple_slicing():
    # https://github.com/pandas-dev/pandas/issues/35858
    date = pd.Timestamp("2000")
    ser = Series(
        1,
        index=MultiIndex.from_tuples([("a", date)], names=["a", "b"]),
        name="c",
    )
    result = ser.loc[:, [date]]
    tm.assert_series_equal(result, ser)


def test_loc_with_mi_indexer():
    # https://github.com/pandas-dev/pandas/issues/35351
    df = DataFrame(
        data=[["a", 1], ["a", 0], ["b", 1], ["c", 2]],
        index=MultiIndex.from_tuples(
            [(0, 1), (1, 0), (1, 1), (1, 1)], names=["index", "date"]
        ),
        columns=["author", "price"],
    )
    idx = MultiIndex.from_tuples([(0, 1), (1, 1)], names=["index", "date"])
    result = df.loc[idx, :]
    expected = DataFrame(
        [["a", 1], ["b", 1], ["c", 2]],
        index=MultiIndex.from_tuples([(0, 1), (1, 1), (1, 1)], names=["index", "date"]),
        columns=["author", "price"],
    )
    tm.assert_frame_equal(result, expected)


def test_loc_mi_with_level1_named_0():
    # GH#37194
    dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")

    ser = Series(range(3), index=dti)
    df = ser.to_frame()
    df[1] = dti

    df2 = df.set_index(0, append=True)
    assert df2.index.names == (None, 0)
    df2.index.get_loc(dti[0])  # smoke test

    result = df2.loc[dti[0]]
    expected = df2.iloc[[0]].droplevel(None)
    tm.assert_frame_equal(result, expected)

    ser2 = df2[1]
    assert ser2.index.names == (None, 0)

    result = ser2.loc[dti[0]]
    expected = ser2.iloc[[0]].droplevel(None)
    tm.assert_series_equal(result, expected)


def test_getitem_str_slice():
    # GH#15928
    df = DataFrame(
        [
            ["20160525 13:30:00.023", "MSFT", "51.95", "51.95"],
            ["20160525 13:30:00.048", "GOOG", "720.50", "720.93"],
            ["20160525 13:30:00.076", "AAPL", "98.55", "98.56"],
            ["20160525 13:30:00.131", "AAPL", "98.61", "98.62"],
            ["20160525 13:30:00.135", "MSFT", "51.92", "51.95"],
            ["20160525 13:30:00.135", "AAPL", "98.61", "98.62"],
        ],
        columns="time,ticker,bid,ask".split(","),
    )
    df2 = df.set_index(["ticker", "time"]).sort_index()

    res = df2.loc[("AAPL", slice("2016-05-25 13:30:00")), :].droplevel(0)
    expected = df2.loc["AAPL"].loc[slice("2016-05-25 13:30:00"), :]
    tm.assert_frame_equal(res, expected)


def test_3levels_leading_period_index():
    # GH#24091
    pi = pd.PeriodIndex(
        ["20181101 1100", "20181101 1200", "20181102 1300", "20181102 1400"],
        name="datetime",
        freq="D",
    )
    lev2 = ["A", "A", "Z", "W"]
    lev3 = ["B", "C", "Q", "F"]
    mi = MultiIndex.from_arrays([pi, lev2, lev3])

    ser = Series(range(4), index=mi, dtype=np.float64)
    result = ser.loc[(pi[0], "A", "B")]
    assert result == 0.0


class TestKeyErrorsWithMultiIndex:
    def test_missing_keys_raises_keyerror(self):
        # GH#27420 KeyError, not TypeError
        df = DataFrame(np.arange(12).reshape(4, 3), columns=["A", "B", "C"])
        df2 = df.set_index(["A", "B"])

        with pytest.raises(KeyError, match="1"):
            df2.loc[(1, 6)]

    def test_missing_key_raises_keyerror2(self):
        # GH#21168 KeyError, not "IndexingError: Too many indexers"
        ser = Series(-1, index=MultiIndex.from_product([[0, 1]] * 2))

        with pytest.raises(KeyError, match=r"\(0, 3\)"):
            ser.loc[0, 3]

    def test_missing_key_combination(self):
        # GH: 19556
        mi = MultiIndex.from_arrays(
            [
                np.array(["a", "a", "b", "b"]),
                np.array(["1", "2", "2", "3"]),
                np.array(["c", "d", "c", "d"]),
            ],
            names=["one", "two", "three"],
        )
        df = DataFrame(np.random.default_rng(2).random((4, 3)), index=mi)
        msg = r"\('b', '1', slice\(None, None, None\)\)"
        with pytest.raises(KeyError, match=msg):
            df.loc[("b", "1", slice(None)), :]
        with pytest.raises(KeyError, match=msg):
            df.index.get_locs(("b", "1", slice(None)))
        with pytest.raises(KeyError, match=r"\('b', '1'\)"):
            df.loc[("b", "1"), :]


def test_getitem_loc_commutability(multiindex_year_month_day_dataframe_random_data):
    df = multiindex_year_month_day_dataframe_random_data
    ser = df["A"]
    result = ser[2000, 5]
    expected = df.loc[2000, 5]["A"]
    tm.assert_series_equal(result, expected)


def test_loc_with_nan():
    # GH: 27104
    df = DataFrame(
        {"col": [1, 2, 5], "ind1": ["a", "d", np.nan], "ind2": [1, 4, 5]}
    ).set_index(["ind1", "ind2"])
    result = df.loc[["a"]]
    expected = DataFrame(
        {"col": [1]}, index=MultiIndex.from_tuples([("a", 1)], names=["ind1", "ind2"])
    )
    tm.assert_frame_equal(result, expected)

    result = df.loc["a"]
    expected = DataFrame({"col": [1]}, index=Index([1], name="ind2"))
    tm.assert_frame_equal(result, expected)


def test_getitem_non_found_tuple():
    # GH: 25236
    df = DataFrame([[1, 2, 3, 4]], columns=["a", "b", "c", "d"]).set_index(
        ["a", "b", "c"]
    )
    with pytest.raises(KeyError, match=r"\(2\.0, 2\.0, 3\.0\)"):
        df.loc[(2.0, 2.0, 3.0)]


def test_get_loc_datetime_index():
    # GH#24263
    index = pd.date_range("2001-01-01", periods=100)
    mi = MultiIndex.from_arrays([index])
    # Check if get_loc matches for Index and MultiIndex
    assert mi.get_loc("2001-01") == slice(0, 31, None)
    assert index.get_loc("2001-01") == slice(0, 31, None)

    loc = mi[::2].get_loc("2001-01")
    expected = index[::2].get_loc("2001-01")
    assert loc == expected

    loc = mi.repeat(2).get_loc("2001-01")
    expected = index.repeat(2).get_loc("2001-01")
    assert loc == expected

    loc = mi.append(mi).get_loc("2001-01")
    expected = index.append(index).get_loc("2001-01")
    # TODO: standardize return type for MultiIndex.get_loc
    tm.assert_numpy_array_equal(loc.nonzero()[0], expected)


def test_loc_setitem_indexer_differently_ordered():
    # GH#34603
    mi = MultiIndex.from_product([["a", "b"], [0, 1]])
    df = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=mi)

    indexer = ("a", [1, 0])
    df.loc[indexer, :] = np.array([[9, 10], [11, 12]])
    expected = DataFrame([[11, 12], [9, 10], [5, 6], [7, 8]], index=mi)
    tm.assert_frame_equal(df, expected)


def test_loc_getitem_index_differently_ordered_slice_none():
    # GH#31330
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
        columns=["a", "b"],
    )
    result = df.loc[(slice(None), [2, 1]), :]
    expected = DataFrame(
        [[3, 4], [7, 8], [1, 2], [5, 6]],
        index=[["a", "b", "a", "b"], [2, 2, 1, 1]],
        columns=["a", "b"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("indexer", [[1, 2, 7, 6, 2, 3, 8, 7], [1, 2, 7, 6, 3, 8]])
def test_loc_getitem_index_differently_ordered_slice_none_duplicates(indexer):
    # GH#40978
    df = DataFrame(
        [1] * 8,
        index=MultiIndex.from_tuples(
            [(1, 1), (1, 2), (1, 7), (1, 6), (2, 2), (2, 3), (2, 8), (2, 7)]
        ),
        columns=["a"],
    )
    result = df.loc[(slice(None), indexer), :]
    expected = DataFrame(
        [1] * 8,
        index=[[1, 1, 2, 1, 2, 1, 2, 2], [1, 2, 2, 7, 7, 6, 3, 8]],
        columns=["a"],
    )
    tm.assert_frame_equal(result, expected)

    result = df.loc[df.index.isin(indexer, level=1), :]
    tm.assert_frame_equal(result, df)


def test_loc_getitem_drops_levels_for_one_row_dataframe():
    # GH#10521 "x" and "z" are both scalar indexing, so those levels are dropped
    mi = MultiIndex.from_arrays([["x"], ["y"], ["z"]], names=["a", "b", "c"])
    df = DataFrame({"d": [0]}, index=mi)
    expected = df.droplevel([0, 2])
    result = df.loc["x", :, "z"]
    tm.assert_frame_equal(result, expected)

    ser = Series([0], index=mi)
    result = ser.loc["x", :, "z"]
    expected = Series([0], index=Index(["y"], name="b"))
    tm.assert_series_equal(result, expected)


def test_mi_columns_loc_list_label_order():
    # GH 10710
    cols = MultiIndex.from_product([["A", "B", "C"], [1, 2]])
    df = DataFrame(np.zeros((5, 6)), columns=cols)
    result = df.loc[:, ["B", "A"]]
    expected = DataFrame(
        np.zeros((5, 4)),
        columns=MultiIndex.from_tuples([("B", 1), ("B", 2), ("A", 1), ("A", 2)]),
    )
    tm.assert_frame_equal(result, expected)


def test_mi_partial_indexing_list_raises():
    # GH 13501
    frame = DataFrame(
        np.arange(12).reshape((4, 3)),
        index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
        columns=[["Ohio", "Ohio", "Colorado"], ["Green", "Red", "Green"]],
    )
    frame.index.names = ["key1", "key2"]
    frame.columns.names = ["state", "color"]
    with pytest.raises(KeyError, match="\\[2\\] not in index"):
        frame.loc[["b", 2], "Colorado"]


def test_mi_indexing_list_nonexistent_raises():
    # GH 15452
    s = Series(range(4), index=MultiIndex.from_product([[1, 2], ["a", "b"]]))
    with pytest.raises(KeyError, match="\\['not' 'found'\\] not in index"):
        s.loc[["not", "found"]]


def test_mi_add_cell_missing_row_non_unique():
    # GH 16018
    result = DataFrame(
        [[1, 2, 5, 6], [3, 4, 7, 8]],
        index=["a", "a"],
        columns=MultiIndex.from_product([[1, 2], ["A", "B"]]),
    )
    result.loc["c"] = -1
    result.loc["c", (1, "A")] = 3
    result.loc["d", (1, "A")] = 3
    expected = DataFrame(
        [
            [1.0, 2.0, 5.0, 6.0],
            [3.0, 4.0, 7.0, 8.0],
            [3.0, -1.0, -1, -1],
            [3.0, np.nan, np.nan, np.nan],
        ],
        index=["a", "a", "c", "d"],
        columns=MultiIndex.from_product([[1, 2], ["A", "B"]]),
    )
    tm.assert_frame_equal(result, expected)


def test_loc_get_scalar_casting_to_float():
    # GH#41369
    df = DataFrame(
        {"a": 1.0, "b": 2}, index=MultiIndex.from_arrays([[3], [4]], names=["c", "d"])
    )
    result = df.loc[(3, 4), "b"]
    assert result == 2
    assert isinstance(result, np.int64)
    result = df.loc[[(3, 4)], "b"].iloc[0]
    assert result == 2
    assert isinstance(result, np.int64)


def test_loc_empty_single_selector_with_names():
    # GH 19517
    idx = MultiIndex.from_product([["a", "b"], ["A", "B"]], names=[1, 0])
    s2 = Series(index=idx, dtype=np.float64)
    result = s2.loc["a"]
    expected = Series([np.nan, np.nan], index=Index(["A", "B"], name=0))
    tm.assert_series_equal(result, expected)


def test_loc_keyerror_rightmost_key_missing():
    # GH 20951

    df = DataFrame(
        {
            "A": [100, 100, 200, 200, 300, 300],
            "B": [10, 10, 20, 21, 31, 33],
            "C": range(6),
        }
    )
    df = df.set_index(["A", "B"])
    with pytest.raises(KeyError, match="^1$"):
        df.loc[(100, 1)]


def test_multindex_series_loc_with_tuple_label():
    # GH#43908
    mi = MultiIndex.from_tuples([(1, 2), (3, (4, 5))])
    ser = Series([1, 2], index=mi)
    result = ser.loc[(3, (4, 5))]
    assert result == 2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\cffi\lock.py ===
import sys

if sys.version_info < (3,):
    try:
        from thread import allocate_lock
    except ImportError:
        from dummy_thread import allocate_lock
else:
    try:
        from _thread import allocate_lock
    except ImportError:
        from _dummy_thread import allocate_lock


##import sys
##l1 = allocate_lock

##class allocate_lock(object):
##    def __init__(self):
##        self._real = l1()
##    def __enter__(self):
##        for i in range(4, 0, -1):
##            print sys._getframe(i).f_code
##        print
##        return self._real.__enter__()
##    def __exit__(self, *args):
##        return self._real.__exit__(*args)
##    def acquire(self, f):
##        assert f is False
##        return self._real.acquire(f)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\compat\_constants.py ===
"""
_constants
======

Constants relevant for the Python implementation.
"""

from __future__ import annotations

import platform
import sys
import sysconfig

IS64 = sys.maxsize > 2**32

PY310 = sys.version_info >= (3, 10)
PY311 = sys.version_info >= (3, 11)
PY312 = sys.version_info >= (3, 12)
PYPY = platform.python_implementation() == "PyPy"
ISMUSL = "musl" in (sysconfig.get_config_var("HOST_GNU_TYPE") or "")
REF_COUNT = 2 if PY311 else 3

__all__ = [
    "IS64",
    "ISMUSL",
    "PY310",
    "PY311",
    "PY312",
    "PYPY",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\ops\dispatch.py ===
"""
Functions for defining unary operations.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.dtypes.generic import ABCExtensionArray

if TYPE_CHECKING:
    from pandas._typing import ArrayLike


def should_extension_dispatch(left: ArrayLike, right: Any) -> bool:
    """
    Identify cases where Series operation should dispatch to ExtensionArray method.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object

    Returns
    -------
    bool
    """
    return isinstance(left, ABCExtensionArray) or isinstance(right, ABCExtensionArray)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\clipboard\obsolete.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from typing import Any, List

from .api import set_clipboard_text


def _make_tab(lists: List[Any]) -> str:
    if hasattr(lists, "tolist"):
        lists = lists.tolist()

    ut = []
    for rad in lists:
        if type(rad) in [list, tuple]:
            ut.append("\t".join(["%s" % x for x in rad]))
        else:
            ut.append("%s" % rad)

    return "\n".join(ut)


def _send_data(lists: List[Any]) -> None:
    set_clipboard_text(_make_tab(lists))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\_typing.py ===
# dialects/_typing.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
from __future__ import annotations

from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Union

from ..sql import roles
from ..sql.base import ColumnCollection
from ..sql.schema import Column
from ..sql.schema import ColumnCollectionConstraint
from ..sql.schema import Index


_OnConflictConstraintT = Union[str, ColumnCollectionConstraint, Index, None]
_OnConflictIndexElementsT = Optional[
    Iterable[Union[Column[Any], str, roles.DDLConstraintColumnRole]]
]
_OnConflictIndexWhereT = Optional[roles.WhereHavingRole]
_OnConflictSetT = Optional[
    Union[Mapping[Any, Any], ColumnCollection[Any, Any]]
]
_OnConflictWhereT = Optional[roles.WhereHavingRole]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\random_matrix.py ===
from sympy.core.basic import Basic
from sympy.stats.rv import PSpace, _symbol_converter, RandomMatrixSymbol

class RandomMatrixPSpace(PSpace):
    """
    Represents probability space for
    random matrices. It contains the mechanics
    for handling the API calls for random matrices.
    """
    def __new__(cls, sym, model=None):
        sym = _symbol_converter(sym)
        if model:
            return Basic.__new__(cls, sym, model)
        else:
            return Basic.__new__(cls, sym)

    @property
    def model(self):
        try:
            return self.args[1]
        except IndexError:
            return None

    def compute_density(self, expr, *args):
        rms = expr.atoms(RandomMatrixSymbol)
        if len(rms) > 2 or (not isinstance(expr, RandomMatrixSymbol)):
            raise NotImplementedError("Currently, no algorithm has been "
                    "implemented to handle general expressions containing "
                    "multiple random matrices.")
        return self.model.density(expr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\__init__.py ===
"""This module contains some general purpose utilities that are used across
SymPy.
"""
from .iterables import (flatten, group, take, subsets,
    variations, numbered_symbols, cartes, capture, dict_merge,
    prefixes, postfixes, sift, topological_sort, unflatten,
    has_dups, has_variety, reshape, rotations)

from .misc import filldedent

from .lambdify import lambdify

from .decorator import threaded, xthreaded, public, memoize_property

from .timeutils import timed

__all__ = [
    'flatten', 'group', 'take', 'subsets', 'variations', 'numbered_symbols',
    'cartes', 'capture', 'dict_merge', 'prefixes', 'postfixes', 'sift',
    'topological_sort', 'unflatten', 'has_dups', 'has_variety', 'reshape',
    'rotations',

    'filldedent',

    'lambdify',

    'threaded', 'xthreaded', 'public', 'memoize_property',

    'timed',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\test_series.py ===
""" Test cases for Series.plot """
from datetime import datetime
from itertools import chain

import numpy as np
import pytest

from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
    period_range,
    plotting,
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_ax_scales,
    _check_axes_shape,
    _check_colors,
    _check_grid_settings,
    _check_has_errorbars,
    _check_legend_labels,
    _check_plot_works,
    _check_text_labels,
    _check_ticks_props,
    _unpack_cycler,
    get_y_axis,
)

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture
def ts():
    return Series(
        np.arange(10, dtype=np.float64),
        index=date_range("2020-01-01", periods=10),
        name="ts",
    )


@pytest.fixture
def series():
    return Series(
        range(20), dtype=np.float64, name="series", index=[f"i_{i}" for i in range(20)]
    )


class TestSeriesPlots:
    @pytest.mark.slow
    @pytest.mark.parametrize("kwargs", [{"label": "foo"}, {"use_index": False}])
    def test_plot(self, ts, kwargs):
        _check_plot_works(ts.plot, **kwargs)

    @pytest.mark.slow
    def test_plot_tick_props(self, ts):
        axes = _check_plot_works(ts.plot, rot=0)
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "scale, exp_scale",
        [
            [{"logy": True}, {"yaxis": "log"}],
            [{"logx": True}, {"xaxis": "log"}],
            [{"loglog": True}, {"xaxis": "log", "yaxis": "log"}],
        ],
    )
    def test_plot_scales(self, ts, scale, exp_scale):
        ax = _check_plot_works(ts.plot, style=".", **scale)
        _check_ax_scales(ax, **exp_scale)

    @pytest.mark.slow
    def test_plot_ts_bar(self, ts):
        _check_plot_works(ts[:10].plot.bar)

    @pytest.mark.slow
    def test_plot_ts_area_stacked(self, ts):
        _check_plot_works(ts.plot.area, stacked=False)

    def test_plot_iseries(self):
        ser = Series(range(5), period_range("2020-01-01", periods=5))
        _check_plot_works(ser.plot)

    @pytest.mark.parametrize(
        "kind",
        [
            "line",
            "bar",
            "barh",
            pytest.param("kde", marks=td.skip_if_no("scipy")),
            "hist",
            "box",
        ],
    )
    def test_plot_series_kinds(self, series, kind):
        _check_plot_works(series[:5].plot, kind=kind)

    def test_plot_series_barh(self, series):
        _check_plot_works(series[:10].plot.barh)

    def test_plot_series_bar_ax(self):
        ax = _check_plot_works(
            Series(np.random.default_rng(2).standard_normal(10)).plot.bar, color="black"
        )
        _check_colors([ax.patches[0]], facecolors=["black"])

    @pytest.mark.parametrize("kwargs", [{}, {"layout": (-1, 1)}, {"layout": (1, -1)}])
    def test_plot_6951(self, ts, kwargs):
        # GH 6951
        ax = _check_plot_works(ts.plot, subplots=True, **kwargs)
        _check_axes_shape(ax, axes_num=1, layout=(1, 1))

    def test_plot_figsize_and_title(self, series):
        # figsize and title
        _, ax = mpl.pyplot.subplots()
        ax = series.plot(title="Test", figsize=(16, 8), ax=ax)
        _check_text_labels(ax.title, "Test")
        _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16, 8))

    def test_dont_modify_rcParams(self):
        # GH 8242
        key = "axes.prop_cycle"
        colors = mpl.pyplot.rcParams[key]
        _, ax = mpl.pyplot.subplots()
        Series([1, 2, 3]).plot(ax=ax)
        assert colors == mpl.pyplot.rcParams[key]

    @pytest.mark.parametrize("kwargs", [{}, {"secondary_y": True}])
    def test_ts_line_lim(self, ts, kwargs):
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot(ax=ax, **kwargs)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data(orig=False)[0][0]
        assert xmax >= lines[0].get_data(orig=False)[0][-1]

    def test_ts_area_lim(self, ts):
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.area(stacked=False, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_ts_area_lim_xcompat(self, ts):
        # GH 7471
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=30)

    def test_ts_tz_area_lim_xcompat(self, ts):
        tz_ts = ts.copy()
        tz_ts.index = tz_ts.tz_localize("GMT").tz_convert("CET")
        _, ax = mpl.pyplot.subplots()
        ax = tz_ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_ts_tz_area_lim_xcompat_secondary_y(self, ts):
        tz_ts = ts.copy()
        tz_ts.index = tz_ts.tz_localize("GMT").tz_convert("CET")
        _, ax = mpl.pyplot.subplots()
        ax = tz_ts.plot.area(stacked=False, secondary_y=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        _check_ticks_props(ax, xrot=0)

    def test_area_sharey_dont_overwrite(self, ts):
        # GH37942
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2, sharey=True)

        abs(ts).plot(ax=ax1, kind="area")
        abs(ts).plot(ax=ax2, kind="area")

        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)
        plt.close(fig)

    def test_label(self):
        s = Series([1, 2])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(label="LABEL", legend=True, ax=ax)
        _check_legend_labels(ax, labels=["LABEL"])
        mpl.pyplot.close("all")

    def test_label_none(self):
        s = Series([1, 2])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, ax=ax)
        _check_legend_labels(ax, labels=[""])
        mpl.pyplot.close("all")

    def test_label_ser_name(self):
        s = Series([1, 2], name="NAME")
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, ax=ax)
        _check_legend_labels(ax, labels=["NAME"])
        mpl.pyplot.close("all")

    def test_label_ser_name_override(self):
        s = Series([1, 2], name="NAME")
        # override the default
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=True, label="LABEL", ax=ax)
        _check_legend_labels(ax, labels=["LABEL"])
        mpl.pyplot.close("all")

    def test_label_ser_name_override_dont_draw(self):
        s = Series([1, 2], name="NAME")
        # Add lebel info, but don't draw
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(legend=False, label="LABEL", ax=ax)
        assert ax.get_legend() is None  # Hasn't been drawn
        ax.legend()  # draw it
        _check_legend_labels(ax, labels=["LABEL"])
        mpl.pyplot.close("all")

    def test_boolean(self):
        # GH 23719
        s = Series([False, False, True])
        _check_plot_works(s.plot, include_bool=True)

        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            _check_plot_works(s.plot)

    @pytest.mark.parametrize("index", [None, date_range("2020-01-01", periods=4)])
    def test_line_area_nan_series(self, index):
        values = [1, 2, np.nan, 3]
        d = Series(values, index=index)
        ax = _check_plot_works(d.plot)
        masked = ax.lines[0].get_ydata()
        # remove nan for comparison purpose
        exp = np.array([1, 2, 3], dtype=np.float64)
        tm.assert_numpy_array_equal(np.delete(masked.data, 2), exp)
        tm.assert_numpy_array_equal(masked.mask, np.array([False, False, True, False]))

        expected = np.array([1, 2, 0, 3], dtype=np.float64)
        ax = _check_plot_works(d.plot, stacked=True)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
        ax = _check_plot_works(d.plot.area)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
        ax = _check_plot_works(d.plot.area, stacked=False)
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)

    def test_line_use_index_false(self):
        s = Series([1, 2, 3], index=["a", "b", "c"])
        s.index.name = "The Index"
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(use_index=False, ax=ax)
        label = ax.get_xlabel()
        assert label == ""

    def test_line_use_index_false_diff_var(self):
        s = Series([1, 2, 3], index=["a", "b", "c"])
        s.index.name = "The Index"
        _, ax = mpl.pyplot.subplots()
        ax2 = s.plot.bar(use_index=False, ax=ax)
        label2 = ax2.get_xlabel()
        assert label2 == ""

    @pytest.mark.xfail(
        np_version_gte1p24 and is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    @pytest.mark.parametrize("axis, meth", [("yaxis", "bar"), ("xaxis", "barh")])
    def test_bar_log(self, axis, meth):
        expected = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

        _, ax = mpl.pyplot.subplots()
        ax = getattr(Series([200, 500]).plot, meth)(log=True, ax=ax)
        tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)

    @pytest.mark.xfail(
        np_version_gte1p24 and is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    @pytest.mark.parametrize(
        "axis, kind, res_meth",
        [["yaxis", "bar", "get_ylim"], ["xaxis", "barh", "get_xlim"]],
    )
    def test_bar_log_kind_bar(self, axis, kind, res_meth):
        # GH 9905
        expected = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])

        _, ax = mpl.pyplot.subplots()
        ax = Series([0.1, 0.01, 0.001]).plot(log=True, kind=kind, ax=ax)
        ymin = 0.0007943282347242822
        ymax = 0.12589254117941673
        res = getattr(ax, res_meth)()
        tm.assert_almost_equal(res[0], ymin)
        tm.assert_almost_equal(res[1], ymax)
        tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)

    def test_bar_ignore_index(self):
        df = Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.bar(use_index=False, ax=ax)
        _check_text_labels(ax.get_xticklabels(), ["0", "1", "2", "3"])

    def test_bar_user_colors(self):
        s = Series([1, 2, 3, 4])
        ax = s.plot.bar(color=["red", "blue", "blue", "red"])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        assert result == expected

    def test_rotation_default(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # Default rot 0
        _, ax = mpl.pyplot.subplots()
        axes = df.plot(ax=ax)
        _check_ticks_props(axes, xrot=0)

    def test_rotation_30(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        _, ax = mpl.pyplot.subplots()
        axes = df.plot(rot=30, ax=ax)
        _check_ticks_props(axes, xrot=30)

    def test_irregular_datetime(self):
        from pandas.plotting._matplotlib.converter import DatetimeConverter

        rng = date_range("1/1/2000", "3/1/2000")
        rng = rng[[0, 1, 2, 3, 5, 9, 10, 11, 12]]
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        _, ax = mpl.pyplot.subplots()
        ax = ser.plot(ax=ax)
        xp = DatetimeConverter.convert(datetime(1999, 1, 1), "", ax)
        ax.set_xlim("1/1/1999", "1/1/2001")
        assert xp == ax.get_xlim()[0]
        _check_ticks_props(ax, xrot=30)

    def test_unsorted_index_xlim(self):
        ser = Series(
            [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0],
        )
        _, ax = mpl.pyplot.subplots()
        ax = ser.plot(ax=ax)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data(orig=False)[0])
        assert xmax >= np.nanmax(lines[0].get_data(orig=False)[0])

    def test_pie_series(self):
        # if sum of values is less than 1.0, pie handle them as rate and draw
        # semicircle.
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        ax = _check_plot_works(series.plot.pie)
        _check_text_labels(ax.texts, series.index)
        assert ax.get_ylabel() == "YLABEL"

    def test_pie_series_no_label(self):
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        ax = _check_plot_works(series.plot.pie, labels=None)
        _check_text_labels(ax.texts, [""] * 5)

    def test_pie_series_less_colors_than_elements(self):
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        color_args = ["r", "g", "b"]
        ax = _check_plot_works(series.plot.pie, colors=color_args)

        color_expected = ["r", "g", "b", "r", "g"]
        _check_colors(ax.patches, facecolors=color_expected)

    def test_pie_series_labels_and_colors(self):
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        # with labels and colors
        labels = ["A", "B", "C", "D", "E"]
        color_args = ["r", "g", "b", "c", "m"]
        ax = _check_plot_works(series.plot.pie, labels=labels, colors=color_args)
        _check_text_labels(ax.texts, labels)
        _check_colors(ax.patches, facecolors=color_args)

    def test_pie_series_autopct_and_fontsize(self):
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        color_args = ["r", "g", "b", "c", "m"]
        ax = _check_plot_works(
            series.plot.pie, colors=color_args, autopct="%.2f", fontsize=7
        )
        pcts = [f"{s*100:.2f}" for s in series.values / series.sum()]
        expected_texts = list(chain.from_iterable(zip(series.index, pcts)))
        _check_text_labels(ax.texts, expected_texts)
        for t in ax.texts:
            assert t.get_fontsize() == 7

    def test_pie_series_negative_raises(self):
        # includes negative value
        series = Series([1, 2, 0, 4, -1], index=["a", "b", "c", "d", "e"])
        with pytest.raises(ValueError, match="pie plot doesn't allow negative values"):
            series.plot.pie()

    def test_pie_series_nan(self):
        # includes nan
        series = Series([1, 2, np.nan, 4], index=["a", "b", "c", "d"], name="YLABEL")
        ax = _check_plot_works(series.plot.pie)
        _check_text_labels(ax.texts, ["a", "b", "", "d"])

    def test_pie_nan(self):
        s = Series([1, np.nan, 1, 1])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot.pie(legend=True, ax=ax)
        expected = ["0", "", "2", "3"]
        result = [x.get_text() for x in ax.texts]
        assert result == expected

    def test_df_series_secondary_legend(self):
        # GH 9779
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 3)), columns=list("abc")
        )
        s = Series(np.random.default_rng(2).standard_normal(30), name="x")

        # primary -> secondary (without passing ax)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(ax=ax)
        s.plot(legend=True, secondary_y=True, ax=ax)
        # both legends are drawn on left ax
        # left and right axis must be visible
        _check_legend_labels(ax, labels=["a", "b", "c", "x (right)"])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()

    def test_df_series_secondary_legend_with_axes(self):
        # GH 9779
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 3)), columns=list("abc")
        )
        s = Series(np.random.default_rng(2).standard_normal(30), name="x")
        # primary -> secondary (with passing ax)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        # both legends are drawn on left ax
        # left and right axis must be visible
        _check_legend_labels(ax, labels=["a", "b", "c", "x (right)"])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()

    def test_df_series_secondary_legend_both(self):
        # GH 9779
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 3)), columns=list("abc")
        )
        s = Series(np.random.default_rng(2).standard_normal(30), name="x")
        # secondary -> secondary (without passing ax)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(secondary_y=True, ax=ax)
        s.plot(legend=True, secondary_y=True, ax=ax)
        # both legends are drawn on left ax
        # left axis must be invisible and right axis must be visible
        expected = ["a (right)", "b (right)", "c (right)", "x (right)"]
        _check_legend_labels(ax.left_ax, labels=expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    def test_df_series_secondary_legend_both_with_axis(self):
        # GH 9779
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 3)), columns=list("abc")
        )
        s = Series(np.random.default_rng(2).standard_normal(30), name="x")
        # secondary -> secondary (with passing ax)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(secondary_y=True, ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        # both legends are drawn on left ax
        # left axis must be invisible and right axis must be visible
        expected = ["a (right)", "b (right)", "c (right)", "x (right)"]
        _check_legend_labels(ax.left_ax, expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    def test_df_series_secondary_legend_both_with_axis_2(self):
        # GH 9779
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 3)), columns=list("abc")
        )
        s = Series(np.random.default_rng(2).standard_normal(30), name="x")
        # secondary -> secondary (with passing ax)
        _, ax = mpl.pyplot.subplots()
        ax = df.plot(secondary_y=True, mark_right=False, ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        # both legends are drawn on left ax
        # left axis must be invisible and right axis must be visible
        expected = ["a", "b", "c", "x (right)"]
        _check_legend_labels(ax.left_ax, expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    @pytest.mark.parametrize(
        "input_logy, expected_scale", [(True, "log"), ("sym", "symlog")]
    )
    def test_secondary_logy(self, input_logy, expected_scale):
        # GH 25545
        s1 = Series(np.random.default_rng(2).standard_normal(100))
        s2 = Series(np.random.default_rng(2).standard_normal(100))

        # GH 24980
        ax1 = s1.plot(logy=input_logy)
        ax2 = s2.plot(secondary_y=True, logy=input_logy)

        assert ax1.get_yscale() == expected_scale
        assert ax2.get_yscale() == expected_scale

    def test_plot_fails_with_dupe_color_and_style(self):
        x = Series(np.random.default_rng(2).standard_normal(2))
        _, ax = mpl.pyplot.subplots()
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' keyword "
            "argument. Please use one or the other or pass 'style' without a color "
            "symbol"
        )
        with pytest.raises(ValueError, match=msg):
            x.plot(style="k--", color="k", ax=ax)

    @pytest.mark.parametrize(
        "bw_method, ind",
        [
            ["scott", 20],
            [None, 20],
            [None, np.int_(20)],
            [0.5, np.linspace(-100, 100, 20)],
        ],
    )
    def test_kde_kwargs(self, ts, bw_method, ind):
        pytest.importorskip("scipy")
        _check_plot_works(ts.plot.kde, bw_method=bw_method, ind=ind)

    def test_density_kwargs(self, ts):
        pytest.importorskip("scipy")
        sample_points = np.linspace(-100, 100, 20)
        _check_plot_works(ts.plot.density, bw_method=0.5, ind=sample_points)

    def test_kde_kwargs_check_axes(self, ts):
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        sample_points = np.linspace(-100, 100, 20)
        ax = ts.plot.kde(logy=True, bw_method=0.5, ind=sample_points, ax=ax)
        _check_ax_scales(ax, yaxis="log")
        _check_text_labels(ax.yaxis.get_label(), "Density")

    def test_kde_missing_vals(self):
        pytest.importorskip("scipy")
        s = Series(np.random.default_rng(2).uniform(size=50))
        s[0] = np.nan
        axes = _check_plot_works(s.plot.kde)

        # gh-14821: check if the values have any missing values
        assert any(~np.isnan(axes.lines[0].get_xdata()))

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_boxplot_series(self, ts):
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.box(logy=True, ax=ax)
        _check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        _check_text_labels(xlabels, [ts.name])
        ylabels = ax.get_yticklabels()
        _check_text_labels(ylabels, [""] * len(ylabels))

    @pytest.mark.parametrize(
        "kind",
        plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds,
    )
    def test_kind_kwarg(self, kind):
        pytest.importorskip("scipy")
        s = Series(range(3))
        _, ax = mpl.pyplot.subplots()
        s.plot(kind=kind, ax=ax)
        mpl.pyplot.close()

    @pytest.mark.parametrize(
        "kind",
        plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds,
    )
    def test_kind_attr(self, kind):
        pytest.importorskip("scipy")
        s = Series(range(3))
        _, ax = mpl.pyplot.subplots()
        getattr(s.plot, kind)()
        mpl.pyplot.close()

    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    def test_invalid_plot_data(self, kind):
        s = Series(list("abcd"))
        _, ax = mpl.pyplot.subplots()
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            s.plot(kind=kind, ax=ax)

    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    def test_valid_object_plot(self, kind):
        pytest.importorskip("scipy")
        s = Series(range(10), dtype=object)
        _check_plot_works(s.plot, kind=kind)

    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    def test_partially_invalid_plot_data(self, kind):
        s = Series(["a", "b", 1.0, 2])
        _, ax = mpl.pyplot.subplots()
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            s.plot(kind=kind, ax=ax)

    def test_invalid_kind(self):
        s = Series([1, 2])
        with pytest.raises(ValueError, match="invalid_kind is not a valid plot kind"):
            s.plot(kind="invalid_kind")

    def test_dup_datetime_index_plot(self):
        dr1 = date_range("1/1/2009", periods=4)
        dr2 = date_range("1/2/2009", periods=4)
        index = dr1.append(dr2)
        values = np.random.default_rng(2).standard_normal(index.size)
        s = Series(values, index=index)
        _check_plot_works(s.plot)

    def test_errorbar_asymmetrical(self):
        # GH9536
        s = Series(np.arange(10), name="x")
        err = np.random.default_rng(2).random((2, 10))

        ax = s.plot(yerr=err, xerr=err)

        result = np.vstack([i.vertices[:, 1] for i in ax.collections[1].get_paths()])
        expected = (err.T * np.array([-1, 1])) + s.to_numpy().reshape(-1, 1)
        tm.assert_numpy_array_equal(result, expected)

        msg = (
            "Asymmetrical error bars should be provided "
            f"with the shape \\(2, {len(s)}\\)"
        )
        with pytest.raises(ValueError, match=msg):
            s.plot(yerr=np.random.default_rng(2).random((2, 11)))

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["line", "bar"])
    @pytest.mark.parametrize(
        "yerr",
        [
            Series(np.abs(np.random.default_rng(2).standard_normal(10))),
            np.abs(np.random.default_rng(2).standard_normal(10)),
            list(np.abs(np.random.default_rng(2).standard_normal(10))),
            DataFrame(
                np.abs(np.random.default_rng(2).standard_normal((10, 2))),
                columns=["x", "y"],
            ),
        ],
    )
    def test_errorbar_plot(self, kind, yerr):
        s = Series(np.arange(10), name="x")
        ax = _check_plot_works(s.plot, yerr=yerr, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    def test_errorbar_plot_yerr_0(self):
        s = Series(np.arange(10), name="x")
        s_err = np.abs(np.random.default_rng(2).standard_normal(10))
        ax = _check_plot_works(s.plot, xerr=s_err)
        _check_has_errorbars(ax, xerr=1, yerr=0)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "yerr",
        [
            Series(np.abs(np.random.default_rng(2).standard_normal(12))),
            DataFrame(
                np.abs(np.random.default_rng(2).standard_normal((12, 2))),
                columns=["x", "y"],
            ),
        ],
    )
    def test_errorbar_plot_ts(self, yerr):
        # test time series plotting
        ix = date_range("1/1/2000", "1/1/2001", freq="ME")
        ts = Series(np.arange(12), index=ix, name="x")
        yerr.index = ix

        ax = _check_plot_works(ts.plot, yerr=yerr)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    def test_errorbar_plot_invalid_yerr_shape(self):
        s = Series(np.arange(10), name="x")
        # check incorrect lengths and types
        with tm.external_error_raised(ValueError):
            s.plot(yerr=np.arange(11))

    @pytest.mark.slow
    def test_errorbar_plot_invalid_yerr(self):
        s = Series(np.arange(10), name="x")
        s_err = ["zzz"] * 10
        with tm.external_error_raised(TypeError):
            s.plot(yerr=s_err)

    @pytest.mark.slow
    def test_table_true(self, series):
        _check_plot_works(series.plot, table=True)

    @pytest.mark.slow
    def test_table_self(self, series):
        _check_plot_works(series.plot, table=series)

    @pytest.mark.slow
    def test_series_grid_settings(self):
        # Make sure plot defaults to rcParams['axes.grid'] setting, GH 9792
        pytest.importorskip("scipy")
        _check_grid_settings(
            Series([1, 2, 3]),
            plotting.PlotAccessor._series_kinds + plotting.PlotAccessor._common_kinds,
        )

    @pytest.mark.parametrize("c", ["r", "red", "green", "#FF0000"])
    def test_standard_colors(self, c):
        from pandas.plotting._matplotlib.style import get_standard_colors

        result = get_standard_colors(1, color=c)
        assert result == [c]

        result = get_standard_colors(1, color=[c])
        assert result == [c]

        result = get_standard_colors(3, color=c)
        assert result == [c] * 3

        result = get_standard_colors(3, color=[c])
        assert result == [c] * 3

    def test_standard_colors_all(self):
        from matplotlib import colors

        from pandas.plotting._matplotlib.style import get_standard_colors

        # multiple colors like mediumaquamarine
        for c in colors.cnames:
            result = get_standard_colors(num_colors=1, color=c)
            assert result == [c]

            result = get_standard_colors(num_colors=1, color=[c])
            assert result == [c]

            result = get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3

            result = get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

        # single letter colors like k
        for c in colors.ColorConverter.colors:
            result = get_standard_colors(num_colors=1, color=c)
            assert result == [c]

            result = get_standard_colors(num_colors=1, color=[c])
            assert result == [c]

            result = get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3

            result = get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

    def test_series_plot_color_kwargs(self):
        # GH1890
        _, ax = mpl.pyplot.subplots()
        ax = Series(np.arange(12) + 1).plot(color="green", ax=ax)
        _check_colors(ax.get_lines(), linecolors=["green"])

    def test_time_series_plot_color_kwargs(self):
        # #1890
        _, ax = mpl.pyplot.subplots()
        ax = Series(np.arange(12) + 1, index=date_range("1/1/2000", periods=12)).plot(
            color="green", ax=ax
        )
        _check_colors(ax.get_lines(), linecolors=["green"])

    def test_time_series_plot_color_with_empty_kwargs(self):
        import matplotlib as mpl

        def_colors = _unpack_cycler(mpl.rcParams)
        index = date_range("1/1/2000", periods=12)
        s = Series(np.arange(1, 13), index=index)

        ncolors = 3

        _, ax = mpl.pyplot.subplots()
        for i in range(ncolors):
            ax = s.plot(ax=ax)
        _check_colors(ax.get_lines(), linecolors=def_colors[:ncolors])

    def test_xticklabels(self):
        # GH11529
        s = Series(np.arange(10), index=[f"P{i:02d}" for i in range(10)])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(xticks=[0, 3, 5, 9], ax=ax)
        exp = [f"P{i:02d}" for i in [0, 3, 5, 9]]
        _check_text_labels(ax.get_xticklabels(), exp)

    def test_xtick_barPlot(self):
        # GH28172
        s = Series(range(10), index=[f"P{i:02d}" for i in range(10)])
        ax = s.plot.bar(xticks=range(0, 11, 2))
        exp = np.array(list(range(0, 11, 2)))
        tm.assert_numpy_array_equal(exp, ax.get_xticks())

    def test_custom_business_day_freq(self):
        # GH7222
        from pandas.tseries.offsets import CustomBusinessDay

        s = Series(
            range(100, 121),
            index=pd.bdate_range(
                start="2014-05-01",
                end="2014-06-01",
                freq=CustomBusinessDay(holidays=["2014-05-26"]),
            ),
        )

        _check_plot_works(s.plot)

    @pytest.mark.xfail(
        reason="GH#24426, see also "
        "github.com/pandas-dev/pandas/commit/"
        "ef1bd69fa42bbed5d09dd17f08c44fc8bfc2b685#r61470674"
    )
    def test_plot_accessor_updates_on_inplace(self):
        ser = Series([1, 2, 3, 4])
        _, ax = mpl.pyplot.subplots()
        ax = ser.plot(ax=ax)
        before = ax.xaxis.get_ticklocs()

        ser.drop([0, 1], inplace=True)
        _, ax = mpl.pyplot.subplots()
        after = ax.xaxis.get_ticklocs()
        tm.assert_numpy_array_equal(before, after)

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_plot_xlim_for_series(self, kind):
        # test if xlim is also correctly plotted in Series for line and area
        # GH 27686
        s = Series([2, 3])
        _, ax = mpl.pyplot.subplots()
        s.plot(kind=kind, ax=ax)
        xlims = ax.get_xlim()

        assert xlims[0] < 0
        assert xlims[1] > 1

    def test_plot_no_rows(self):
        # GH 27758
        df = Series(dtype=int)
        assert df.empty
        ax = df.plot()
        assert len(ax.get_lines()) == 1
        line = ax.get_lines()[0]
        assert len(line.get_xdata()) == 0
        assert len(line.get_ydata()) == 0

    def test_plot_no_numeric_data(self):
        df = Series(["a", "b", "c"])
        with pytest.raises(TypeError, match="no numeric data to plot"):
            df.plot()

    @pytest.mark.parametrize(
        "data, index",
        [
            ([1, 2, 3, 4], [3, 2, 1, 0]),
            ([10, 50, 20, 30], [1910, 1920, 1980, 1950]),
        ],
    )
    def test_plot_order(self, data, index):
        # GH38865 Verify plot order of a Series
        ser = Series(data=data, index=index)
        ax = ser.plot(kind="bar")

        expected = ser.tolist()
        result = [
            patch.get_bbox().ymax
            for patch in sorted(ax.patches, key=lambda patch: patch.get_bbox().xmax)
        ]
        assert expected == result

    def test_style_single_ok(self):
        s = Series([1, 2])
        ax = s.plot(style="s", color="C3")
        assert ax.lines[0].get_color() == "C3"

    @pytest.mark.parametrize(
        "index_name, old_label, new_label",
        [(None, "", "new"), ("old", "old", "new"), (None, "", "")],
    )
    @pytest.mark.parametrize("kind", ["line", "area", "bar", "barh", "hist"])
    def test_xlabel_ylabel_series(self, kind, index_name, old_label, new_label):
        # GH 9093
        ser = Series([1, 2, 3, 4])
        ser.index.name = index_name

        # default is the ylabel is not shown and xlabel is index name (reverse for barh)
        ax = ser.plot(kind=kind)
        if kind == "barh":
            assert ax.get_xlabel() == ""
            assert ax.get_ylabel() == old_label
        elif kind == "hist":
            assert ax.get_xlabel() == ""
            assert ax.get_ylabel() == "Frequency"
        else:
            assert ax.get_ylabel() == ""
            assert ax.get_xlabel() == old_label

        # old xlabel will be overridden and assigned ylabel will be used as ylabel
        ax = ser.plot(kind=kind, ylabel=new_label, xlabel=new_label)
        assert ax.get_ylabel() == new_label
        assert ax.get_xlabel() == new_label

    @pytest.mark.parametrize(
        "index",
        [
            pd.timedelta_range(start=0, periods=2, freq="D"),
            [pd.Timedelta(days=1), pd.Timedelta(days=2)],
        ],
    )
    def test_timedelta_index(self, index):
        # GH37454
        xlims = (3, 1)
        ax = Series([1, 2], index=index).plot(xlim=(xlims))
        assert ax.get_xlim() == (3, 1)

    def test_series_none_color(self):
        # GH51953
        series = Series([1, 2, 3])
        ax = series.plot(color=None)
        expected = _unpack_cycler(mpl.pyplot.rcParams)[:1]
        _check_colors(ax.get_lines(), linecolors=expected)

    @pytest.mark.slow
    def test_plot_no_warning(self, ts):
        # GH 55138
        # TODO(3.0): this can be removed once Period[B] deprecation is enforced
        with tm.assert_produces_warning(False):
            _ = ts.plot()