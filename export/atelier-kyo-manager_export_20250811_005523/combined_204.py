
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\tests\test_integers.py ===
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.numbers import (E, Float, I, Rational, Integer, nan, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, tan, asin
from sympy.polys.rootoftools import RootOf, CRootOf
from sympy import Integers
from sympy.sets.sets import Interval
from sympy.sets.fancysets import ImageSet
from sympy.core.function import Lambda

from sympy.core.expr import unchanged
from sympy.testing.pytest import XFAIL, raises

x = Symbol('x')
i = Symbol('i', imaginary=True)
y = Symbol('y', real=True)
k, n = symbols('k,n', integer=True)
b = Symbol('b', real=True, noninteger=True)
m = Symbol('m', positive=True)


def test_floor():

    assert floor(nan) is nan

    assert floor(oo) is oo
    assert floor(-oo) is -oo
    assert floor(zoo) is zoo

    assert floor(0) == 0

    assert floor(1) == 1
    assert floor(-1) == -1

    assert floor(I*log(asin(5)/abs(asin(5)))) == 0
    assert floor(-I*log(asin(7)/abs(asin(7)))) == -2

    assert floor(E) == 2
    assert floor(-E) == -3

    assert floor(2*E) == 5
    assert floor(-2*E) == -6

    assert floor(pi) == 3
    assert floor(-pi) == -4

    assert floor(S.Half) == 0
    assert floor(Rational(-1, 2)) == -1

    assert floor(Rational(7, 3)) == 2
    assert floor(Rational(-7, 3)) == -3
    assert floor(-Rational(7, 3)) == -3

    assert floor(Float(17.0)) == 17
    assert floor(-Float(17.0)) == -17

    assert floor(Float(7.69)) == 7
    assert floor(-Float(7.69)) == -8

    assert floor(1/(m+1)) == S.Zero
    assert floor((m+2)/(m+1)) == S.One
    assert floor(-1/(m+1)) == S.NegativeOne
    assert floor((m+2)/(-m-1)) == Integer(-2)

    assert floor(I) == I
    assert floor(-I) == -I
    e = floor(i)
    assert e.func is floor and e.args[0] == i

    assert floor(oo*I) == oo*I
    assert floor(-oo*I) == -oo*I
    assert floor(exp(I*pi/4)*oo) == exp(I*pi/4)*oo

    assert floor(2*I) == 2*I
    assert floor(-2*I) == -2*I

    assert floor(I/2) == 0
    assert floor(-I/2) == -I

    assert floor(E + 17) == 19
    assert floor(pi + 2) == 5

    assert floor(E + pi) == 5
    assert floor(I + pi) == 3 + I

    assert floor(floor(pi)) == 3
    assert floor(floor(y)) == floor(y)
    assert floor(floor(x)) == floor(x)

    assert unchanged(floor, x)
    assert unchanged(floor, 2*x)
    assert unchanged(floor, k*x)

    assert floor(k) == k
    assert floor(2*k) == 2*k
    assert floor(k*n) == k*n

    assert unchanged(floor, k/2)

    assert unchanged(floor, x + y)

    assert floor(x + 3) == floor(x) + 3
    assert floor(x + k) == floor(x) + k

    assert floor(y + 3) == floor(y) + 3
    assert floor(y + k) == floor(y) + k

    assert floor(3 + I*y + pi) == 6 + floor(y)*I

    assert floor(k + n) == k + n

    assert unchanged(floor, x*I)
    assert floor(k*I) == k*I

    assert floor(Rational(23, 10) - E*I) == 2 - 3*I

    assert floor(sin(1)) == 0
    assert floor(sin(-1)) == -1

    assert floor(exp(2)) == 7

    assert floor(log(8)/log(2)) != 2
    assert int(floor(log(8)/log(2)).evalf(chop=True)) == 3

    assert floor(factorial(50)/exp(1)) == \
        11188719610782480504630258070757734324011354208865721592720336800

    assert (floor(y) < y).is_Relational
    assert (floor(y) <= y) == True
    assert (floor(y) > y) == False
    assert (floor(y) >= y).is_Relational
    assert (floor(x) <= x).is_Relational  # x could be non-real
    assert (floor(x) > x).is_Relational
    assert (floor(x) <= y).is_Relational  # arg is not same as rhs
    assert (floor(x) > y).is_Relational
    assert (floor(y) <= oo) == True
    assert (floor(y) < oo) == True
    assert (floor(y) >= -oo) == True
    assert (floor(y) > -oo) == True
    assert (floor(b) < b) == True
    assert (floor(b) <= b) == True
    assert (floor(b) > b) == False
    assert (floor(b) >= b) == False

    assert floor(y).rewrite(frac) == y - frac(y)
    assert floor(y).rewrite(ceiling) == -ceiling(-y)
    assert floor(y).rewrite(frac).subs(y, -pi) == floor(-pi)
    assert floor(y).rewrite(frac).subs(y, E) == floor(E)
    assert floor(y).rewrite(ceiling).subs(y, E) == -ceiling(-E)
    assert floor(y).rewrite(ceiling).subs(y, -pi) == -ceiling(pi)

    assert Eq(floor(y), y - frac(y))
    assert Eq(floor(y), -ceiling(-y))

    neg = Symbol('neg', negative=True)
    nn = Symbol('nn', nonnegative=True)
    pos = Symbol('pos', positive=True)
    np = Symbol('np', nonpositive=True)

    assert (floor(neg) < 0) == True
    assert (floor(neg) <= 0) == True
    assert (floor(neg) > 0) == False
    assert (floor(neg) >= 0) == False
    assert (floor(neg) <= -1) == True
    assert (floor(neg) >= -3) == (neg >= -3)
    assert (floor(neg) < 5) == (neg < 5)

    assert (floor(nn) < 0) == False
    assert (floor(nn) >= 0) == True

    assert (floor(pos) < 0) == False
    assert (floor(pos) <= 0) == (pos < 1)
    assert (floor(pos) > 0) == (pos >= 1)
    assert (floor(pos) >= 0) == True
    assert (floor(pos) >= 3) == (pos >= 3)

    assert (floor(np) <= 0) == True
    assert (floor(np) > 0) == False

    assert floor(neg).is_negative == True
    assert floor(neg).is_nonnegative == False
    assert floor(nn).is_negative == False
    assert floor(nn).is_nonnegative == True
    assert floor(pos).is_negative == False
    assert floor(pos).is_nonnegative == True
    assert floor(np).is_negative is None
    assert floor(np).is_nonnegative is None

    assert (floor(7, evaluate=False) >= 7) == True
    assert (floor(7, evaluate=False) > 7) == False
    assert (floor(7, evaluate=False) <= 7) == True
    assert (floor(7, evaluate=False) < 7) == False

    assert (floor(7, evaluate=False) >= 6) == True
    assert (floor(7, evaluate=False) > 6) == True
    assert (floor(7, evaluate=False) <= 6) == False
    assert (floor(7, evaluate=False) < 6) == False

    assert (floor(7, evaluate=False) >= 8) == False
    assert (floor(7, evaluate=False) > 8) == False
    assert (floor(7, evaluate=False) <= 8) == True
    assert (floor(7, evaluate=False) < 8) == True

    assert (floor(x) <= 5.5) == Le(floor(x), 5.5, evaluate=False)
    assert (floor(x) >= -3.2) == Ge(floor(x), -3.2, evaluate=False)
    assert (floor(x) < 2.9) == Lt(floor(x), 2.9, evaluate=False)
    assert (floor(x) > -1.7) == Gt(floor(x), -1.7, evaluate=False)

    assert (floor(y) <= 5.5) == (y < 6)
    assert (floor(y) >= -3.2) == (y >= -3)
    assert (floor(y) < 2.9) == (y < 3)
    assert (floor(y) > -1.7) == (y >= -1)

    assert (floor(y) <= n) == (y < n + 1)
    assert (floor(y) >= n) == (y >= n)
    assert (floor(y) < n) == (y < n)
    assert (floor(y) > n) == (y >= n + 1)

    assert floor(RootOf(x**3 - 27*x, 2)) == 5


def test_ceiling():

    assert ceiling(nan) is nan

    assert ceiling(oo) is oo
    assert ceiling(-oo) is -oo
    assert ceiling(zoo) is zoo

    assert ceiling(0) == 0

    assert ceiling(1) == 1
    assert ceiling(-1) == -1

    assert ceiling(I*log(asin(5)/abs(asin(5)))) == 1
    assert ceiling(-I*log(asin(7)/abs(asin(7)))) == -1

    assert ceiling(E) == 3
    assert ceiling(-E) == -2

    assert ceiling(2*E) == 6
    assert ceiling(-2*E) == -5

    assert ceiling(pi) == 4
    assert ceiling(-pi) == -3

    assert ceiling(S.Half) == 1
    assert ceiling(Rational(-1, 2)) == 0

    assert ceiling(Rational(7, 3)) == 3
    assert ceiling(-Rational(7, 3)) == -2

    assert ceiling(Float(17.0)) == 17
    assert ceiling(-Float(17.0)) == -17

    assert ceiling(Float(7.69)) == 8
    assert ceiling(-Float(7.69)) == -7

    assert ceiling(1/(m+1)) == S.One
    assert ceiling((m+2)/(m+1)) == Integer(2)
    assert ceiling(-1/(m+1)) == S.Zero
    assert ceiling((m+2)/(-m-1)) == S.NegativeOne

    assert ceiling(I) == I
    assert ceiling(-I) == -I
    e = ceiling(i)
    assert e.func is ceiling and e.args[0] == i

    assert ceiling(oo*I) == oo*I
    assert ceiling(-oo*I) == -oo*I
    assert ceiling(exp(I*pi/4)*oo) == exp(I*pi/4)*oo

    assert ceiling(2*I) == 2*I
    assert ceiling(-2*I) == -2*I

    assert ceiling(I/2) == I
    assert ceiling(-I/2) == 0

    assert ceiling(E + 17) == 20
    assert ceiling(pi + 2) == 6

    assert ceiling(E + pi) == 6
    assert ceiling(I + pi) == I + 4

    assert ceiling(ceiling(pi)) == 4
    assert ceiling(ceiling(y)) == ceiling(y)
    assert ceiling(ceiling(x)) == ceiling(x)

    assert unchanged(ceiling, x)
    assert unchanged(ceiling, 2*x)
    assert unchanged(ceiling, k*x)

    assert ceiling(k) == k
    assert ceiling(2*k) == 2*k
    assert ceiling(k*n) == k*n

    assert unchanged(ceiling, k/2)

    assert unchanged(ceiling, x + y)

    assert ceiling(x + 3) == ceiling(x) + 3
    assert ceiling(x + 3.0) == ceiling(x) + 3
    assert ceiling(x + 3.0*I) == ceiling(x) + 3*I
    assert ceiling(x + k) == ceiling(x) + k

    assert ceiling(y + 3) == ceiling(y) + 3
    assert ceiling(y + k) == ceiling(y) + k

    assert ceiling(3 + pi + y*I) == 7 + ceiling(y)*I

    assert ceiling(k + n) == k + n

    assert unchanged(ceiling, x*I)
    assert ceiling(k*I) == k*I

    assert ceiling(Rational(23, 10) - E*I) == 3 - 2*I

    assert ceiling(sin(1)) == 1
    assert ceiling(sin(-1)) == 0

    assert ceiling(exp(2)) == 8

    assert ceiling(-log(8)/log(2)) != -2
    assert int(ceiling(-log(8)/log(2)).evalf(chop=True)) == -3

    assert ceiling(factorial(50)/exp(1)) == \
        11188719610782480504630258070757734324011354208865721592720336801

    assert (ceiling(y) >= y) == True
    assert (ceiling(y) > y).is_Relational
    assert (ceiling(y) < y) == False
    assert (ceiling(y) <= y).is_Relational
    assert (ceiling(x) >= x).is_Relational  # x could be non-real
    assert (ceiling(x) < x).is_Relational
    assert (ceiling(x) >= y).is_Relational  # arg is not same as rhs
    assert (ceiling(x) < y).is_Relational
    assert (ceiling(y) >= -oo) == True
    assert (ceiling(y) > -oo) == True
    assert (ceiling(y) <= oo) == True
    assert (ceiling(y) < oo) == True
    assert (ceiling(b) < b) == False
    assert (ceiling(b) <= b) == False
    assert (ceiling(b) > b) == True
    assert (ceiling(b) >= b) == True

    assert ceiling(y).rewrite(floor) == -floor(-y)
    assert ceiling(y).rewrite(frac) == y + frac(-y)
    assert ceiling(y).rewrite(floor).subs(y, -pi) == -floor(pi)
    assert ceiling(y).rewrite(floor).subs(y, E) == -floor(-E)
    assert ceiling(y).rewrite(frac).subs(y, pi) == ceiling(pi)
    assert ceiling(y).rewrite(frac).subs(y, -E) == ceiling(-E)

    assert Eq(ceiling(y), y + frac(-y))
    assert Eq(ceiling(y), -floor(-y))

    neg = Symbol('neg', negative=True)
    nn = Symbol('nn', nonnegative=True)
    pos = Symbol('pos', positive=True)
    np = Symbol('np', nonpositive=True)

    assert (ceiling(neg) <= 0) == True
    assert (ceiling(neg) < 0) == (neg <= -1)
    assert (ceiling(neg) > 0) == False
    assert (ceiling(neg) >= 0) == (neg > -1)
    assert (ceiling(neg) > -3) == (neg > -3)
    assert (ceiling(neg) <= 10) == (neg <= 10)

    assert (ceiling(nn) < 0) == False
    assert (ceiling(nn) >= 0) == True

    assert (ceiling(pos) < 0) == False
    assert (ceiling(pos) <= 0) == False
    assert (ceiling(pos) > 0) == True
    assert (ceiling(pos) >= 0) == True
    assert (ceiling(pos) >= 1) == True
    assert (ceiling(pos) > 5) == (pos > 5)

    assert (ceiling(np) <= 0) == True
    assert (ceiling(np) > 0) == False

    assert ceiling(neg).is_positive == False
    assert ceiling(neg).is_nonpositive == True
    assert ceiling(nn).is_positive is None
    assert ceiling(nn).is_nonpositive is None
    assert ceiling(pos).is_positive == True
    assert ceiling(pos).is_nonpositive == False
    assert ceiling(np).is_positive == False
    assert ceiling(np).is_nonpositive == True

    assert (ceiling(7, evaluate=False) >= 7) == True
    assert (ceiling(7, evaluate=False) > 7) == False
    assert (ceiling(7, evaluate=False) <= 7) == True
    assert (ceiling(7, evaluate=False) < 7) == False

    assert (ceiling(7, evaluate=False) >= 6) == True
    assert (ceiling(7, evaluate=False) > 6) == True
    assert (ceiling(7, evaluate=False) <= 6) == False
    assert (ceiling(7, evaluate=False) < 6) == False

    assert (ceiling(7, evaluate=False) >= 8) == False
    assert (ceiling(7, evaluate=False) > 8) == False
    assert (ceiling(7, evaluate=False) <= 8) == True
    assert (ceiling(7, evaluate=False) < 8) == True

    assert (ceiling(x) <= 5.5) == Le(ceiling(x), 5.5, evaluate=False)
    assert (ceiling(x) >= -3.2) == Ge(ceiling(x), -3.2, evaluate=False)
    assert (ceiling(x) < 2.9) == Lt(ceiling(x), 2.9, evaluate=False)
    assert (ceiling(x) > -1.7) == Gt(ceiling(x), -1.7, evaluate=False)

    assert (ceiling(y) <= 5.5) == (y <= 5)
    assert (ceiling(y) >= -3.2) == (y > -4)
    assert (ceiling(y) < 2.9) == (y <= 2)
    assert (ceiling(y) > -1.7) == (y > -2)

    assert (ceiling(y) <= n) == (y <= n)
    assert (ceiling(y) >= n) == (y > n - 1)
    assert (ceiling(y) < n) == (y <= n - 1)
    assert (ceiling(y) > n) == (y > n)

    assert ceiling(RootOf(x**3 - 27*x, 2)) == 6
    s = ImageSet(Lambda(n, n + (CRootOf(x**5 - x**2 + 1, 0))), Integers)
    f = CRootOf(x**5 - x**2 + 1, 0)
    s = ImageSet(Lambda(n, n + f), Integers)
    assert s.intersect(Interval(-10, 10)) == {i + f for i in range(-9, 11)}


def test_frac():
    assert isinstance(frac(x), frac)
    assert frac(oo) == AccumBounds(0, 1)
    assert frac(-oo) == AccumBounds(0, 1)
    assert frac(zoo) is nan

    assert frac(n) == 0
    assert frac(nan) is nan
    assert frac(Rational(4, 3)) == Rational(1, 3)
    assert frac(-Rational(4, 3)) == Rational(2, 3)
    assert frac(Rational(-4, 3)) == Rational(2, 3)

    r = Symbol('r', real=True)
    assert frac(I*r) == I*frac(r)
    assert frac(1 + I*r) == I*frac(r)
    assert frac(0.5 + I*r) == 0.5 + I*frac(r)
    assert frac(n + I*r) == I*frac(r)
    assert frac(n + I*k) == 0
    assert unchanged(frac, x + I*x)
    assert frac(x + I*n) == frac(x)

    assert frac(x).rewrite(floor) == x - floor(x)
    assert frac(x).rewrite(ceiling) == x + ceiling(-x)
    assert frac(y).rewrite(floor).subs(y, pi) == frac(pi)
    assert frac(y).rewrite(floor).subs(y, -E) == frac(-E)
    assert frac(y).rewrite(ceiling).subs(y, -pi) == frac(-pi)
    assert frac(y).rewrite(ceiling).subs(y, E) == frac(E)

    assert Eq(frac(y), y - floor(y))
    assert Eq(frac(y), y + ceiling(-y))

    r = Symbol('r', real=True)
    p_i = Symbol('p_i', integer=True, positive=True)
    n_i = Symbol('p_i', integer=True, negative=True)
    np_i = Symbol('np_i', integer=True, nonpositive=True)
    nn_i = Symbol('nn_i', integer=True, nonnegative=True)
    p_r = Symbol('p_r', positive=True)
    n_r = Symbol('n_r', negative=True)
    np_r = Symbol('np_r', real=True, nonpositive=True)
    nn_r = Symbol('nn_r', real=True, nonnegative=True)

    # Real frac argument, integer rhs
    assert frac(r) <= p_i
    assert not frac(r) <= n_i
    assert (frac(r) <= np_i).has(Le)
    assert (frac(r) <= nn_i).has(Le)
    assert frac(r) < p_i
    assert not frac(r) < n_i
    assert not frac(r) < np_i
    assert (frac(r) < nn_i).has(Lt)
    assert not frac(r) >= p_i
    assert frac(r) >= n_i
    assert frac(r) >= np_i
    assert (frac(r) >= nn_i).has(Ge)
    assert not frac(r) > p_i
    assert frac(r) > n_i
    assert (frac(r) > np_i).has(Gt)
    assert (frac(r) > nn_i).has(Gt)

    assert not Eq(frac(r), p_i)
    assert not Eq(frac(r), n_i)
    assert Eq(frac(r), np_i).has(Eq)
    assert Eq(frac(r), nn_i).has(Eq)

    assert Ne(frac(r), p_i)
    assert Ne(frac(r), n_i)
    assert Ne(frac(r), np_i).has(Ne)
    assert Ne(frac(r), nn_i).has(Ne)


    # Real frac argument, real rhs
    assert (frac(r) <= p_r).has(Le)
    assert not frac(r) <= n_r
    assert (frac(r) <= np_r).has(Le)
    assert (frac(r) <= nn_r).has(Le)
    assert (frac(r) < p_r).has(Lt)
    assert not frac(r) < n_r
    assert not frac(r) < np_r
    assert (frac(r) < nn_r).has(Lt)
    assert (frac(r) >= p_r).has(Ge)
    assert frac(r) >= n_r
    assert frac(r) >= np_r
    assert (frac(r) >= nn_r).has(Ge)
    assert (frac(r) > p_r).has(Gt)
    assert frac(r) > n_r
    assert (frac(r) > np_r).has(Gt)
    assert (frac(r) > nn_r).has(Gt)

    assert not Eq(frac(r), n_r)
    assert Eq(frac(r), p_r).has(Eq)
    assert Eq(frac(r), np_r).has(Eq)
    assert Eq(frac(r), nn_r).has(Eq)

    assert Ne(frac(r), p_r).has(Ne)
    assert Ne(frac(r), n_r)
    assert Ne(frac(r), np_r).has(Ne)
    assert Ne(frac(r), nn_r).has(Ne)

    # Real frac argument, +/- oo rhs
    assert frac(r) < oo
    assert frac(r) <= oo
    assert not frac(r) > oo
    assert not frac(r) >= oo

    assert not frac(r) < -oo
    assert not frac(r) <= -oo
    assert frac(r) > -oo
    assert frac(r) >= -oo

    assert frac(r) < 1
    assert frac(r) <= 1
    assert not frac(r) > 1
    assert not frac(r) >= 1

    assert not frac(r) < 0
    assert (frac(r) <= 0).has(Le)
    assert (frac(r) > 0).has(Gt)
    assert frac(r) >= 0

    # Some test for numbers
    assert frac(r) <= sqrt(2)
    assert (frac(r) <= sqrt(3) - sqrt(2)).has(Le)
    assert not frac(r) <= sqrt(2) - sqrt(3)
    assert not frac(r) >= sqrt(2)
    assert (frac(r) >= sqrt(3) - sqrt(2)).has(Ge)
    assert frac(r) >= sqrt(2) - sqrt(3)

    assert not Eq(frac(r), sqrt(2))
    assert Eq(frac(r), sqrt(3) - sqrt(2)).has(Eq)
    assert not Eq(frac(r), sqrt(2) - sqrt(3))
    assert Ne(frac(r), sqrt(2))
    assert Ne(frac(r), sqrt(3) - sqrt(2)).has(Ne)
    assert Ne(frac(r), sqrt(2) - sqrt(3))

    assert frac(p_i, evaluate=False).is_zero
    assert frac(p_i, evaluate=False).is_finite
    assert frac(p_i, evaluate=False).is_integer
    assert frac(p_i, evaluate=False).is_real
    assert frac(r).is_finite
    assert frac(r).is_real
    assert frac(r).is_zero is None
    assert frac(r).is_integer is None

    assert frac(oo).is_finite
    assert frac(oo).is_real


def test_series():
    x, y = symbols('x,y')
    assert floor(x).nseries(x, y, 100) == floor(y)
    assert ceiling(x).nseries(x, y, 100) == ceiling(y)
    assert floor(x).nseries(x, pi, 100) == 3
    assert ceiling(x).nseries(x, pi, 100) == 4
    assert floor(x).nseries(x, 0, 100) == 0
    assert ceiling(x).nseries(x, 0, 100) == 1
    assert floor(-x).nseries(x, 0, 100) == -1
    assert ceiling(-x).nseries(x, 0, 100) == 0


def test_issue_14355():
    # This test checks the leading term and series for the floor and ceil
    # function when arg0 evaluates to S.NaN.
    assert floor((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir = 1) == -2
    assert floor((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir = -1) == -1
    assert floor((cos(x) - 1)/x).as_leading_term(x, cdir = 1) == -1
    assert floor((cos(x) - 1)/x).as_leading_term(x, cdir = -1) == 0
    assert floor(sin(x)/x).as_leading_term(x, cdir = 1) == 0
    assert floor(sin(x)/x).as_leading_term(x, cdir = -1) == 0
    assert floor(-tan(x)/x).as_leading_term(x, cdir = 1) == -2
    assert floor(-tan(x)/x).as_leading_term(x, cdir = -1) == -2
    assert floor(sin(x)/x/3).as_leading_term(x, cdir = 1) == 0
    assert floor(sin(x)/x/3).as_leading_term(x, cdir = -1) == 0
    assert ceiling((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir = 1) == -1
    assert ceiling((x**3 + x)/(x**2 - x)).as_leading_term(x, cdir = -1) == 0
    assert ceiling((cos(x) - 1)/x).as_leading_term(x, cdir = 1) == 0
    assert ceiling((cos(x) - 1)/x).as_leading_term(x, cdir = -1) == 1
    assert ceiling(sin(x)/x).as_leading_term(x, cdir = 1) == 1
    assert ceiling(sin(x)/x).as_leading_term(x, cdir = -1) == 1
    assert ceiling(-tan(x)/x).as_leading_term(x, cdir = 1) == -1
    assert ceiling(-tan(x)/x).as_leading_term(x, cdir = 1) == -1
    assert ceiling(sin(x)/x/3).as_leading_term(x, cdir = 1) == 1
    assert ceiling(sin(x)/x/3).as_leading_term(x, cdir = -1) == 1
    # test for series
    assert floor(sin(x)/x).series(x, 0, 100, cdir = 1) == 0
    assert floor(sin(x)/x).series(x, 0, 100, cdir = 1) == 0
    assert floor((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir = 1) == -2
    assert floor((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir = -1) == -1
    assert ceiling(sin(x)/x).series(x, 0, 100, cdir = 1) == 1
    assert ceiling(sin(x)/x).series(x, 0, 100, cdir = -1) == 1
    assert ceiling((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir = 1) == -1
    assert ceiling((x**3 + x)/(x**2 - x)).series(x, 0, 100, cdir = -1) == 0


def test_frac_leading_term():
    assert frac(x).as_leading_term(x) == x
    assert frac(x).as_leading_term(x, cdir = 1) == x
    assert frac(x).as_leading_term(x, cdir = -1) == 1
    assert frac(x + S.Half).as_leading_term(x, cdir = 1) == S.Half
    assert frac(x + S.Half).as_leading_term(x, cdir = -1) == S.Half
    assert frac(-2*x + 1).as_leading_term(x, cdir = 1) == S.One
    assert frac(-2*x + 1).as_leading_term(x, cdir = -1) == -2*x
    assert frac(sin(x) + 5).as_leading_term(x, cdir = 1) == x
    assert frac(sin(x) + 5).as_leading_term(x, cdir = -1) == S.One
    assert frac(sin(x**2) + 5).as_leading_term(x, cdir = 1) == x**2
    assert frac(sin(x**2) + 5).as_leading_term(x, cdir = -1) == x**2


@XFAIL
def test_issue_4149():
    assert floor(3 + pi*I + y*I) == 3 + floor(pi + y)*I
    assert floor(3*I + pi*I + y*I) == floor(3 + pi + y)*I
    assert floor(3 + E + pi*I + y*I) == 5 + floor(pi + y)*I


def test_issue_21651():
    k = Symbol('k', positive=True, integer=True)
    exp = 2*2**(-k)
    assert isinstance(floor(exp), floor)


def test_issue_11207():
    assert floor(floor(x)) == floor(x)
    assert floor(ceiling(x)) == ceiling(x)
    assert ceiling(floor(x)) == floor(x)
    assert ceiling(ceiling(x)) == ceiling(x)


def test_nested_floor_ceiling():
    assert floor(-floor(ceiling(x**3)/y)) == -floor(ceiling(x**3)/y)
    assert ceiling(-floor(ceiling(x**3)/y)) == -floor(ceiling(x**3)/y)
    assert floor(ceiling(-floor(x**Rational(7, 2)/y))) == -floor(x**Rational(7, 2)/y)
    assert -ceiling(-ceiling(floor(x)/y)) == ceiling(floor(x)/y)

def test_issue_18689():
    assert floor(floor(floor(x)) + 3) == floor(x) + 3
    assert ceiling(ceiling(ceiling(x)) + 1) == ceiling(x) + 1
    assert ceiling(ceiling(floor(x)) + 3) == floor(x) + 3

def test_issue_18421():
    assert floor(float(0)) is S.Zero
    assert ceiling(float(0)) is S.Zero

def test_issue_25230():
    a = Symbol('a', real = True)
    b = Symbol('b', positive = True)
    c = Symbol('c', negative = True)
    raises(NotImplementedError, lambda: floor(x/a).as_leading_term(x, cdir = 1))
    raises(NotImplementedError, lambda: ceiling(x/a).as_leading_term(x, cdir = 1))
    assert floor(x/b).as_leading_term(x, cdir = 1) == 0
    assert floor(x/b).as_leading_term(x, cdir = -1) == -1
    assert floor(x/c).as_leading_term(x, cdir = 1) == -1
    assert floor(x/c).as_leading_term(x, cdir = -1) == 0
    assert ceiling(x/b).as_leading_term(x, cdir = 1) == 1
    assert ceiling(x/b).as_leading_term(x, cdir = -1) == 0
    assert ceiling(x/c).as_leading_term(x, cdir = 1) == 0
    assert ceiling(x/c).as_leading_term(x, cdir = -1) == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\tests\test_filter.py ===
import pytest
import re
import warnings

from . import (
    SoupTest,
)
from typing import (
    Callable,
    Optional,
    Tuple,
)
from bs4.element import Tag
from bs4.filter import (
    AttributeValueMatchRule,
    ElementFilter,
    MatchRule,
    SoupStrainer,
    StringMatchRule,
    TagNameMatchRule,
)
from bs4._typing import _RawAttributeValues


class TestElementFilter(SoupTest):
    def test_default_behavior(self):
        # An unconfigured ElementFilter matches absolutely everything.
        selector = ElementFilter()
        assert not selector.excludes_everything
        assert selector.includes_everything
        soup = self.soup("<a>text</a>")
        tag = soup.a
        string = tag.string
        assert True is selector.match(soup)
        assert True is selector.match(tag)
        assert True is selector.match(string)
        assert soup.find(selector).name == "a"

        # And allows any incoming markup to be turned into PageElements.
        assert True is selector.allow_tag_creation(None, "tag", None)
        assert True is selector.allow_string_creation("some string")

    def test_setup_with_match_function(self):
        # Configure an ElementFilter with a match function and
        # we can no longer state with certainty that it includes everything.
        selector = ElementFilter(lambda x: False)
        assert not selector.includes_everything

    def test_match(self):
        def m(pe):
            return pe.string == "allow" or (isinstance(pe, Tag) and pe.name == "allow")

        soup = self.soup("<allow>deny</allow>allow<deny>deny</deny>")
        allow_tag = soup.allow
        allow_string = soup.find(string="allow")
        deny_tag = soup.deny
        deny_string = soup.find(string="deny")

        selector = ElementFilter(match_function=m)
        assert True is selector.match(allow_tag)
        assert True is selector.match(allow_string)
        assert False is selector.match(deny_tag)
        assert False is selector.match(deny_string)

        # Since only the match function was provided, there is
        # no effect on tag or string creation.
        soup = self.soup("<a>text</a>", parse_only=selector)
        assert "text" == soup.a.string

    def test_allow_tag_creation(self):
        # By default, ElementFilter.allow_tag_creation allows everything.
        filter = ElementFilter()
        f = filter.allow_tag_creation
        assert True is f("allow", "ignore", {})
        assert True is f("ignore", "allow", {})
        assert True is f(None, "ignore", {"allow": "1"})
        assert True is f("no", "no", {"no": "nope"})

        # You can customize this behavior by overriding
        # allow_tag_creation in a subclass.
        class MyFilter(ElementFilter):
            def allow_tag_creation(
                self,
                nsprefix: Optional[str],
                name: str,
                attrs: Optional[_RawAttributeValues],
            ):
                return (
                    nsprefix == "allow"
                    or name == "allow"
                    or (attrs is not None and "allow" in attrs)
                )

        filter = MyFilter()
        f = filter.allow_tag_creation
        assert True is f("allow", "ignore", {})
        assert True is f("ignore", "allow", {})
        assert True is f(None, "ignore", {"allow": "1"})
        assert False is f("no", "no", {"no": "nope"})

        # Test the customized ElementFilter as a value for parse_only.
        soup = self.soup(
            "<deny>deny</deny> <allow>deny</allow> allow", parse_only=filter
        )

        # The <deny> tag was filtered out, but there was no effect on
        # the strings, since only allow_tag_creation_function was
        # overridden.
        assert "deny <allow>deny</allow> allow" == soup.decode()

        # Similarly, since match_function was not defined, this
        # ElementFilter matches everything.
        assert soup.find(filter) == "deny"

    def test_allow_string_creation(self):
        # By default, ElementFilter.allow_string_creation allows everything.
        filter = ElementFilter()
        f = filter.allow_string_creation
        assert True is f("allow")
        assert True is f("deny")
        assert True is f("please allow")

        # You can customize this behavior by overriding allow_string_creation
        # in a subclass.
        class MyFilter(ElementFilter):
            def allow_string_creation(self, s: str):
                return s == "allow"

        filter = MyFilter()
        f = filter.allow_string_creation
        assert True is f("allow")
        assert False is f("deny")
        assert False is f("please allow")

        # Test the customized ElementFilter as a value for parse_only.
        soup = self.soup(
            "<deny>deny</deny> <allow>deny</allow> allow", parse_only=filter
        )

        # All incoming strings other than "allow" (even whitespace)
        # were filtered out, but there was no effect on the tags,
        # since only allow_string_creation_function was defined.
        assert "<deny>deny</deny><allow>deny</allow>" == soup.decode()

        # Similarly, since match_function was not defined, this
        # ElementFilter matches everything.
        assert soup.find(filter).name == "deny"


class TestMatchRule(SoupTest):
    def _tuple(
        self, rule: MatchRule
    ) -> Tuple[Optional[str], Optional[str], Optional[Callable], Optional[bool]]:
        return (
            rule.string,
            rule.pattern.pattern if rule.pattern else None,
            rule.function,
            rule.present,
        )

    @staticmethod
    def tag_function(x: Tag) -> bool:
        return False

    @staticmethod
    def string_function(x: str) -> bool:
        return False

    @pytest.mark.parametrize(
        "constructor_args, constructor_kwargs, result",
        [
            # String
            ([], dict(string="a"), ("a", None, None, None)),
            (
                [],
                dict(string="\N{SNOWMAN}".encode("utf8")),
                ("\N{SNOWMAN}", None, None, None),
            ),
            # Regular expression
            ([], dict(pattern=re.compile("a")), (None, "a", None, None)),
            ([], dict(pattern="b"), (None, "b", None, None)),
            ([], dict(pattern=b"c"), (None, "c", None, None)),
            # Function
            ([], dict(function=tag_function), (None, None, tag_function, None)),
            ([], dict(function=string_function), (None, None, string_function, None)),
            # Boolean
            ([], dict(present=True), (None, None, None, True)),
            # With positional arguments rather than keywords
            (("a", None, None, None), {}, ("a", None, None, None)),
            ((None, "b", None, None), {}, (None, "b", None, None)),
            ((None, None, tag_function, None), {}, (None, None, tag_function, None)),
            ((None, None, None, True), {}, (None, None, None, True)),
        ],
    )
    def test_constructor(self, constructor_args, constructor_kwargs, result):
        rule = MatchRule(*constructor_args, **constructor_kwargs)
        assert result == self._tuple(rule)

    def test_empty_match_not_allowed(self):
        with pytest.raises(
            ValueError,
            match="Either string, pattern, function, present, or exclude_everything must be provided.",
        ):
            MatchRule()

    def test_full_match_not_allowed(self):
        with pytest.raises(
            ValueError,
            match="At most one of string, pattern, function, present, and exclude_everything must be provided.",
        ):
            MatchRule("a", "b", self.tag_function, True)

    @pytest.mark.parametrize(
        "rule_kwargs, match_against, result",
        [
            (dict(string="a"), "a", True),
            (dict(string="a"), "ab", False),
            (dict(pattern="a"), "a", True),
            (dict(pattern="a"), "ab", True),
            (dict(pattern="^a$"), "a", True),
            (dict(pattern="^a$"), "ab", False),
            (dict(present=True), "any random value", True),
            (dict(present=True), None, False),
            (dict(present=False), "any random value", False),
            (dict(present=False), None, True),
            (dict(function=lambda x: x.upper() == x), "UPPERCASE", True),
            (dict(function=lambda x: x.upper() == x), "lowercase", False),
            (dict(function=lambda x: x.lower() == x), "UPPERCASE", False),
            (dict(function=lambda x: x.lower() == x), "lowercase", True),
        ],
    )
    def test_matches_string(self, rule_kwargs, match_against, result):
        rule = MatchRule(**rule_kwargs)
        assert rule.matches_string(match_against) == result


class TestTagNameMatchRule(SoupTest):
    @pytest.mark.parametrize(
        "rule_kwargs, tag_kwargs, result",
        [
            (dict(string="a"), dict(name="a"), True),
            (dict(string="a"), dict(name="ab"), False),
            (dict(pattern="a"), dict(name="a"), True),
            (dict(pattern="a"), dict(name="ab"), True),
            (dict(pattern="^a$"), dict(name="a"), True),
            (dict(pattern="^a$"), dict(name="ab"), False),
            # This isn't very useful, but it will work.
            (dict(present=True), dict(name="any random value"), True),
            (dict(present=False), dict(name="any random value"), False),
            (
                dict(function=lambda t: t.name in t.attrs),
                dict(name="id", attrs=dict(id="a")),
                True,
            ),
            (
                dict(function=lambda t: t.name in t.attrs),
                dict(name="id", attrs={"class": "a"}),
                False,
            ),
        ],
    )
    def test_matches_tag(self, rule_kwargs, tag_kwargs, result):
        rule = TagNameMatchRule(**rule_kwargs)
        tag = Tag(**tag_kwargs)
        assert rule.matches_tag(tag) == result

    def test_matches_tag_only_passes_tag_to_function(self):
        def arg1_must_be_tag(t):
            if not isinstance(t, Tag):
                raise ValueError("Non-tag passed into tag name match rule function")
            return True

        rule = TagNameMatchRule(function=arg1_must_be_tag)
        assert rule.matches_tag(Tag(name="tagname")) is True


# AttributeValueMatchRule and StringMatchRule have the same
# logic as MatchRule.


class TestSoupStrainer(SoupTest):

    def test_constructor_string_deprecated_text_argument(self):
        with warnings.catch_warnings(record=True) as w:
            strainer = SoupStrainer(text="text")
            assert strainer.text == "text"
            [w1, w2] = w
            msg = str(w1.message)
            assert w1.filename == __file__
            assert (
                msg
                == "As of version 4.11.0, the 'text' argument to the SoupStrainer constructor is deprecated. Use 'string' instead."
            )

            msg = str(w2.message)
            assert w2.filename == __file__
            assert (
                msg
                == "Access to deprecated property text. (Look at .string_rules instead) -- Deprecated since version 4.13.0."
            )

    def test_search_tag_deprecated(self):
        strainer = SoupStrainer(name="a")
        with warnings.catch_warnings(record=True) as w:
            assert False is strainer.search_tag("b", {})
            [w1] = w
            msg = str(w1.message)
            assert w1.filename == __file__
            assert (
                msg
                == "Call to deprecated method search_tag. (Replaced by allow_tag_creation) -- Deprecated since version 4.13.0."
            )

    def test_search_deprecated(self):
        strainer = SoupStrainer(name="a")
        soup = self.soup("<a></a><b></b>")
        with warnings.catch_warnings(record=True) as w:
            assert soup.a == strainer.search(soup.a)
            assert None is strainer.search(soup.b)
            [w1, w2] = w
            msg = str(w1.message)
            assert msg == str(w2.message)
            assert w1.filename == __file__
            assert (
                msg
                == "Call to deprecated method search. (Replaced by match) -- Deprecated since version 4.13.0."
            )

    # Dummy function used within tests.
    def _match_function(x):
        pass

    def test_constructor_default(self):
        # The default SoupStrainer matches all tags, and only tags.
        strainer = SoupStrainer()
        [name_rule] = strainer.name_rules
        assert True == name_rule.present
        assert 0 == len(strainer.attribute_rules)
        assert 0 == len(strainer.string_rules)

    def test_constructor(self):
        strainer = SoupStrainer(
            "tagname",
            {"attr1": "value"},
            string=self._match_function,
            attr2=["value1", False],
        )
        [name_rule] = strainer.name_rules
        assert name_rule == TagNameMatchRule(string="tagname")

        [attr1_rule] = strainer.attribute_rules.pop("attr1")
        assert attr1_rule == AttributeValueMatchRule(string="value")

        [attr2_rule1, attr2_rule2] = strainer.attribute_rules.pop("attr2")
        assert attr2_rule1 == AttributeValueMatchRule(string="value1")
        assert attr2_rule2 == AttributeValueMatchRule(present=False)

        assert not strainer.attribute_rules

        [string_rule] = strainer.string_rules
        assert string_rule == StringMatchRule(function=self._match_function)

    def test_scalar_attrs_becomes_class_restriction(self):
        # For the sake of convenience, passing a scalar value as
        # ``args`` results in a restriction on the 'class' attribute.
        strainer = SoupStrainer(attrs="mainbody")
        assert [] == strainer.name_rules
        assert [] == strainer.string_rules
        assert {"class": [AttributeValueMatchRule(string="mainbody")]} == (
            strainer.attribute_rules
        )

    def test_constructor_class_attribute(self):
        # The 'class' HTML attribute is also treated specially because
        # it's a Python reserved word. Passing in "class_" as a
        # keyword argument results in a restriction on the 'class'
        # attribute.
        strainer = SoupStrainer(class_="mainbody")
        assert [] == strainer.name_rules
        assert [] == strainer.string_rules
        assert {"class": [AttributeValueMatchRule(string="mainbody")]} == (
            strainer.attribute_rules
        )

        # But if you pass in "class_" as part of the ``attrs`` dict
        # it's not changed. (Otherwise there'd be no way to actually put
        # a restriction on an attribute called "class_".)
        strainer = SoupStrainer(attrs=dict(class_="mainbody"))
        assert [] == strainer.name_rules
        assert [] == strainer.string_rules
        assert {"class_": [AttributeValueMatchRule(string="mainbody")]} == (
            strainer.attribute_rules
        )

    def test_constructor_with_overlapping_attributes(self):
        # If you specify the same attribute in args and **kwargs, you end up
        # with two different AttributeValueMatchRule objects.

        # This happens whether you use the 'class' shortcut on attrs...
        strainer = SoupStrainer(attrs="class1", class_="class2")
        rule1, rule2 = strainer.attribute_rules["class"]
        assert rule1.string == "class1"
        assert rule2.string == "class2"

        # Or explicitly specify the same attribute twice.
        strainer = SoupStrainer(attrs={"id": "id1"}, id="id2")
        rule1, rule2 = strainer.attribute_rules["id"]
        assert rule1.string == "id1"
        assert rule2.string == "id2"

    @pytest.mark.parametrize(
        "obj, result",
        [
            ("a", MatchRule(string="a")),
            (b"a", MatchRule(string="a")),
            (True, MatchRule(present=True)),
            (False, MatchRule(present=False)),
            (re.compile("a"), MatchRule(pattern=re.compile("a"))),
            (_match_function, MatchRule(function=_match_function)),
            # Pass in a list and get back a list of rules.
            (["a", b"b"], [MatchRule(string="a"), MatchRule(string="b")]),
            (
                [re.compile("a"), _match_function],
                [
                    MatchRule(pattern=re.compile("a")),
                    MatchRule(function=_match_function),
                ],
            ),
            # Anything that doesn't fit is converted to a string.
            (100, MatchRule(string="100")),
        ],
    )
    def test__make_match_rules(self, obj, result):
        actual = list(SoupStrainer._make_match_rules(obj, MatchRule))
        # Helper to reduce the number of single-item lists in the
        # parameters.
        if len(actual) == 1:
            [actual] = actual
        assert result == actual

    @pytest.mark.parametrize(
        "cls, result",
        [
            (AttributeValueMatchRule, AttributeValueMatchRule(string="a")),
            (StringMatchRule, StringMatchRule(string="a")),
        ],
    )
    def test__make_match_rules_different_classes(self, cls, result):
        actual = cls(string="a")
        assert actual == result

    def test__make_match_rules_nested_list(self):
        # If you pass a nested list into _make_match_rules, it's
        # turned into a restriction that excludes everything, to avoid the
        # possibility of an infinite recursion.

        # Create a self-referential object.
        selfref = []
        selfref.append(selfref)

        with warnings.catch_warnings(record=True) as w:
            rules = SoupStrainer._make_match_rules(["a", selfref, "b"], MatchRule)
            assert list(rules) == [MatchRule(string="a"), MatchRule(exclude_everything=True), MatchRule(string="b")]

            [warning] = w
            # Don't check the filename because the stacklevel is
            # designed for normal use and we're testing the private
            # method directly.
            msg = str(warning.message)
            assert (
                msg
                == "Ignoring nested list [[...]] to avoid the possibility of infinite recursion."
            )

    def tag_matches(
        self,
        strainer: SoupStrainer,
        name: str,
        attrs: Optional[_RawAttributeValues] = None,
        string: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> bool:
        # Create a Tag with the given prefix, name and attributes,
        # then make sure that strainer.matches_tag and allow_tag_creation
        # both approve it.
        tag = Tag(prefix=prefix, name=name, attrs=attrs)
        if string:
            tag.string = string
        return strainer.matches_tag(tag) and strainer.allow_tag_creation(
            prefix, name, attrs
        )

    def test_matches_tag_with_only_string(self):
        # A SoupStrainer that only has StringMatchRules won't ever
        # match a Tag.
        strainer = SoupStrainer(string=["a string", re.compile("string")])
        tag = Tag(name="b", attrs=dict(id="1"))
        tag.string = "a string"
        assert not strainer.matches_tag(tag)

        # There has to be a TagNameMatchRule or an
        # AttributeValueMatchRule as well.
        strainer.name_rules.append(TagNameMatchRule(string="b"))
        assert strainer.matches_tag(tag)

        strainer.name_rules = []
        strainer.attribute_rules["id"] = [AttributeValueMatchRule("1")]
        assert strainer.matches_tag(tag)

    def test_matches_tag_with_prefix(self):
        # If a tag has an attached namespace prefix, the tag's name is
        # tested both with and without the prefix.
        kwargs = dict(name="a", prefix="ns")

        assert self.tag_matches(SoupStrainer(name="a"), **kwargs)
        assert self.tag_matches(SoupStrainer(name="ns:a"), **kwargs)
        assert not self.tag_matches(SoupStrainer(name="ns2:a"), **kwargs)

    def test_one_name_rule_must_match(self):
        # If there are TagNameMatchRule, at least one must match.
        kwargs = dict(name="b")

        assert self.tag_matches(SoupStrainer(name="b"), **kwargs)
        assert not self.tag_matches(SoupStrainer(name="c"), **kwargs)
        assert self.tag_matches(SoupStrainer(name=["c", "d", "d", "b"]), **kwargs)
        assert self.tag_matches(
            SoupStrainer(name=[re.compile("c-f"), re.compile("[ab]$")]), **kwargs
        )

    def test_one_attribute_rule_must_match_for_each_attribute(self):
        # If there is one or more AttributeValueMatchRule for a given
        # attribute, at least one must match that attribute's
        # value. This is true for *every* attribute -- just matching one
        # attribute isn't enough.
        kwargs = dict(name="b", attrs={"class": "main", "id": "1"})

        # 'class' and 'id' match
        assert self.tag_matches(
            SoupStrainer(
                class_=["other", "main"], id=["20", "a", re.compile("^[0-9]")]
            ),
            **kwargs,
        )

        # 'class' and 'id' are present and 'data' attribute is missing
        assert self.tag_matches(
            SoupStrainer(class_=True, id=True, data=False), **kwargs
        )

        # 'id' matches, 'class' does not.
        assert not self.tag_matches(SoupStrainer(class_=["other"], id=["2"]), **kwargs)

        # 'class' matches, 'id' does not
        assert not self.tag_matches(SoupStrainer(class_=["main"], id=["2"]), **kwargs)

        # 'class' and 'id' match but 'data' attribute is missing
        assert not self.tag_matches(
            SoupStrainer(class_=["main"], id=["1"], data=True), **kwargs
        )

    def test_match_against_multi_valued_attribute(self):
        # If an attribute has multiple values, only one of them
        # has to match the AttributeValueMatchRule.
        kwargs = dict(name="b", attrs={"class": ["main", "big"]})
        assert self.tag_matches(SoupStrainer(attrs="main"), **kwargs)
        assert self.tag_matches(SoupStrainer(attrs="big"), **kwargs)
        assert self.tag_matches(SoupStrainer(attrs=["main", "big"]), **kwargs)
        assert self.tag_matches(SoupStrainer(attrs=["big", "small"]), **kwargs)
        assert not self.tag_matches(SoupStrainer(attrs=["small", "smaller"]), **kwargs)

    def test_match_against_multi_valued_attribute_as_string(self):
        # If an attribute has multiple values, you can treat the entire
        # thing as one string during a match.
        kwargs = dict(name="b", attrs={"class": ["main", "big"]})
        assert self.tag_matches(SoupStrainer(attrs="main big"), **kwargs)

        # But you can't put them in any order; it's got to be the
        # order they are present in the Tag, which basically means the
        # order they were originally present in the document.
        assert not self.tag_matches(SoupStrainer(attrs=["big main"]), **kwargs)

    def test_one_string_rule_must_match(self):
        # If there's a TagNameMatchRule and/or an
        # AttributeValueMatchRule, then the StringMatchRule is _not_
        # ignored, and must match as well.
        tag = Tag(name="b", attrs=dict(id="1"))
        tag.string = "A string"

        assert SoupStrainer(name="b", string="A string").matches_tag(tag)
        assert not SoupStrainer(name="a", string="A string").matches_tag(tag)
        assert not SoupStrainer(name="a", string="Wrong string").matches_tag(tag)
        assert SoupStrainer(id="1", string="A string").matches_tag(tag)
        assert not SoupStrainer(id="2", string="A string").matches_tag(tag)
        assert not SoupStrainer(id="1", string="Wrong string").matches_tag(tag)

        assert SoupStrainer(name="b", id="1", string="A string").matches_tag(tag)

        # If there are multiple string rules, only one needs to match.
        assert SoupStrainer(
            name="b",
            id="1",
            string=["Wrong string", "Also wrong", re.compile("string")],
        ).matches_tag(tag)

    def test_allowing_tag_implies_allowing_its_contents(self):
        markup = "<a><b>one string<div>another string</div></b></a>"

        # Letting the <b> tag through implies parsing the <div> tag
        # and both strings, even though they wouldn't match the
        # SoupStrainer on their own.
        assert (
            "<b>one string<div>another string</div></b>"
            == self.soup(markup, parse_only=SoupStrainer(name="b")).decode()
        )

    @pytest.mark.parametrize(
        "soupstrainer",
        [
            SoupStrainer(name="b", string="one string"),
            SoupStrainer(name="div", string="another string"),
        ],
    )
    def test_parse_only_combining_tag_and_string(self, soupstrainer):
        # If you pass parse_only a SoupStrainer that contains both tag
        # restrictions and string restrictions, you get no results,
        # because the string restrictions can't be evaluated during
        # the parsing process, and the tag restrictions eliminate
        # any strings from consideration.
        #
        # We can detect this ahead of time, and warn about it,
        # thanks to SoupStrainer.excludes_everything
        markup = "<a><b>one string<div>another string</div></b></a>"

        with warnings.catch_warnings(record=True) as w:
            assert True, soupstrainer.excludes_everything
            assert "" == self.soup(markup, parse_only=soupstrainer).decode()
            [warning] = w
            str(warning.message)
            assert warning.filename == __file__
            assert str(warning.message).startswith(
                "The given value for parse_only will exclude everything:"
            )

        # The average SoupStrainer has excludes_everything=False
        assert not SoupStrainer().excludes_everything

    def test_documentation_examples(self):
        """Medium-weight real-world tests based on the Beautiful Soup
        documentation.
        """
        html_doc = """<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""
        only_a_tags = SoupStrainer("a")
        only_tags_with_id_link2 = SoupStrainer(id="link2")

        def is_short_string(string):
            return string is not None and len(string) < 10

        only_short_strings = SoupStrainer(string=is_short_string)

        a_soup = self.soup(html_doc, parse_only=only_a_tags)
        assert (
            '<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a><a class="sister" href="http://example.com/lacie" id="link2">Lacie</a><a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>'
            == a_soup.decode()
        )

        id_soup = self.soup(html_doc, parse_only=only_tags_with_id_link2)
        assert (
            '<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>'
            == id_soup.decode()
        )
        string_soup = self.soup(html_doc, parse_only=only_short_strings)
        assert "\n\n\nElsie,\nLacie and\nTillie\n...\n" == string_soup.decode()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\tests\test_array_from_pyobj.py ===
import copy
import platform
import sys
from pathlib import Path

import pytest

import numpy as np
from numpy._core._type_aliases import c_names_dict as _c_names_dict

from . import util

wrap = None

# Extend core typeinfo with CHARACTER to test dtype('c')
c_names_dict = dict(
    CHARACTER=np.dtype("c"),
    **_c_names_dict
)


def get_testdir():
    testroot = Path(__file__).resolve().parent / "src"
    return testroot / "array_from_pyobj"

def setup_module():
    """
    Build the required testing extension module

    """
    global wrap

    if wrap is None:
        src = [
            get_testdir() / "wrapmodule.c",
        ]
        wrap = util.build_meson(src, module_name="test_array_from_pyobj_ext")


def flags_info(arr):
    flags = wrap.array_attrs(arr)[6]
    return flags2names(flags)


def flags2names(flags):
    info = []
    for flagname in [
            "CONTIGUOUS",
            "FORTRAN",
            "OWNDATA",
            "ENSURECOPY",
            "ENSUREARRAY",
            "ALIGNED",
            "NOTSWAPPED",
            "WRITEABLE",
            "WRITEBACKIFCOPY",
            "UPDATEIFCOPY",
            "BEHAVED",
            "BEHAVED_RO",
            "CARRAY",
            "FARRAY",
    ]:
        if abs(flags) & getattr(wrap, flagname, 0):
            info.append(flagname)
    return info


class Intent:
    def __init__(self, intent_list=[]):
        self.intent_list = intent_list[:]
        flags = 0
        for i in intent_list:
            if i == "optional":
                flags |= wrap.F2PY_OPTIONAL
            else:
                flags |= getattr(wrap, "F2PY_INTENT_" + i.upper())
        self.flags = flags

    def __getattr__(self, name):
        name = name.lower()
        if name == "in_":
            name = "in"
        return self.__class__(self.intent_list + [name])

    def __str__(self):
        return f"intent({','.join(self.intent_list)})"

    def __repr__(self):
        return f"Intent({self.intent_list!r})"

    def is_intent(self, *names):
        return all(name in self.intent_list for name in names)

    def is_intent_exact(self, *names):
        return len(self.intent_list) == len(names) and self.is_intent(*names)


intent = Intent()

_type_names = [
    "BOOL",
    "BYTE",
    "UBYTE",
    "SHORT",
    "USHORT",
    "INT",
    "UINT",
    "LONG",
    "ULONG",
    "LONGLONG",
    "ULONGLONG",
    "FLOAT",
    "DOUBLE",
    "CFLOAT",
    "STRING1",
    "STRING5",
    "CHARACTER",
]

_cast_dict = {"BOOL": ["BOOL"]}
_cast_dict["BYTE"] = _cast_dict["BOOL"] + ["BYTE"]
_cast_dict["UBYTE"] = _cast_dict["BOOL"] + ["UBYTE"]
_cast_dict["BYTE"] = ["BYTE"]
_cast_dict["UBYTE"] = ["UBYTE"]
_cast_dict["SHORT"] = _cast_dict["BYTE"] + ["UBYTE", "SHORT"]
_cast_dict["USHORT"] = _cast_dict["UBYTE"] + ["BYTE", "USHORT"]
_cast_dict["INT"] = _cast_dict["SHORT"] + ["USHORT", "INT"]
_cast_dict["UINT"] = _cast_dict["USHORT"] + ["SHORT", "UINT"]

_cast_dict["LONG"] = _cast_dict["INT"] + ["LONG"]
_cast_dict["ULONG"] = _cast_dict["UINT"] + ["ULONG"]

_cast_dict["LONGLONG"] = _cast_dict["LONG"] + ["LONGLONG"]
_cast_dict["ULONGLONG"] = _cast_dict["ULONG"] + ["ULONGLONG"]

_cast_dict["FLOAT"] = _cast_dict["SHORT"] + ["USHORT", "FLOAT"]
_cast_dict["DOUBLE"] = _cast_dict["INT"] + ["UINT", "FLOAT", "DOUBLE"]

_cast_dict["CFLOAT"] = _cast_dict["FLOAT"] + ["CFLOAT"]

_cast_dict['STRING1'] = ['STRING1']
_cast_dict['STRING5'] = ['STRING5']
_cast_dict['CHARACTER'] = ['CHARACTER']

# 32 bit system malloc typically does not provide the alignment required by
# 16 byte long double types this means the inout intent cannot be satisfied
# and several tests fail as the alignment flag can be randomly true or false
# when numpy gains an aligned allocator the tests could be enabled again
#
# Furthermore, on macOS ARM64, LONGDOUBLE is an alias for DOUBLE.
if ((np.intp().dtype.itemsize != 4 or np.clongdouble().dtype.alignment <= 8)
        and sys.platform != "win32"
        and (platform.system(), platform.processor()) != ("Darwin", "arm")):
    _type_names.extend(["LONGDOUBLE", "CDOUBLE", "CLONGDOUBLE"])
    _cast_dict["LONGDOUBLE"] = _cast_dict["LONG"] + [
        "ULONG",
        "FLOAT",
        "DOUBLE",
        "LONGDOUBLE",
    ]
    _cast_dict["CLONGDOUBLE"] = _cast_dict["LONGDOUBLE"] + [
        "CFLOAT",
        "CDOUBLE",
        "CLONGDOUBLE",
    ]
    _cast_dict["CDOUBLE"] = _cast_dict["DOUBLE"] + ["CFLOAT", "CDOUBLE"]


class Type:
    _type_cache = {}

    def __new__(cls, name):
        if isinstance(name, np.dtype):
            dtype0 = name
            name = None
            for n, i in c_names_dict.items():
                if not isinstance(i, type) and dtype0.type is i.type:
                    name = n
                    break
        obj = cls._type_cache.get(name.upper(), None)
        if obj is not None:
            return obj
        obj = object.__new__(cls)
        obj._init(name)
        cls._type_cache[name.upper()] = obj
        return obj

    def _init(self, name):
        self.NAME = name.upper()

        if self.NAME == 'CHARACTER':
            info = c_names_dict[self.NAME]
            self.type_num = wrap.NPY_STRING
            self.elsize = 1
            self.dtype = np.dtype('c')
        elif self.NAME.startswith('STRING'):
            info = c_names_dict[self.NAME[:6]]
            self.type_num = wrap.NPY_STRING
            self.elsize = int(self.NAME[6:] or 0)
            self.dtype = np.dtype(f'S{self.elsize}')
        else:
            info = c_names_dict[self.NAME]
            self.type_num = getattr(wrap, 'NPY_' + self.NAME)
            self.elsize = info.itemsize
            self.dtype = np.dtype(info.type)

        assert self.type_num == info.num
        self.type = info.type
        self.dtypechar = info.char

    def __repr__(self):
        return (f"Type({self.NAME})|type_num={self.type_num},"
                f" dtype={self.dtype},"
                f" type={self.type}, elsize={self.elsize},"
                f" dtypechar={self.dtypechar}")

    def cast_types(self):
        return [self.__class__(_m) for _m in _cast_dict[self.NAME]]

    def all_types(self):
        return [self.__class__(_m) for _m in _type_names]

    def smaller_types(self):
        bits = c_names_dict[self.NAME].alignment
        types = []
        for name in _type_names:
            if c_names_dict[name].alignment < bits:
                types.append(Type(name))
        return types

    def equal_types(self):
        bits = c_names_dict[self.NAME].alignment
        types = []
        for name in _type_names:
            if name == self.NAME:
                continue
            if c_names_dict[name].alignment == bits:
                types.append(Type(name))
        return types

    def larger_types(self):
        bits = c_names_dict[self.NAME].alignment
        types = []
        for name in _type_names:
            if c_names_dict[name].alignment > bits:
                types.append(Type(name))
        return types


class Array:

    def __repr__(self):
        return (f'Array({self.type}, {self.dims}, {self.intent},'
                f' {self.obj})|arr={self.arr}')

    def __init__(self, typ, dims, intent, obj):
        self.type = typ
        self.dims = dims
        self.intent = intent
        self.obj_copy = copy.deepcopy(obj)
        self.obj = obj

        # arr.dtypechar may be different from typ.dtypechar
        self.arr = wrap.call(typ.type_num,
                             typ.elsize,
                             dims, intent.flags, obj)

        assert isinstance(self.arr, np.ndarray)

        self.arr_attr = wrap.array_attrs(self.arr)

        if len(dims) > 1:
            if self.intent.is_intent("c"):
                assert (intent.flags & wrap.F2PY_INTENT_C)
                assert not self.arr.flags["FORTRAN"]
                assert self.arr.flags["CONTIGUOUS"]
                assert (not self.arr_attr[6] & wrap.FORTRAN)
            else:
                assert (not intent.flags & wrap.F2PY_INTENT_C)
                assert self.arr.flags["FORTRAN"]
                assert not self.arr.flags["CONTIGUOUS"]
                assert (self.arr_attr[6] & wrap.FORTRAN)

        if obj is None:
            self.pyarr = None
            self.pyarr_attr = None
            return

        if intent.is_intent("cache"):
            assert isinstance(obj, np.ndarray), repr(type(obj))
            self.pyarr = np.array(obj).reshape(*dims).copy()
        else:
            self.pyarr = np.array(
                np.array(obj, dtype=typ.dtypechar).reshape(*dims),
                order=(self.intent.is_intent("c") and "C") or "F",
            )
            assert self.pyarr.dtype == typ
        self.pyarr.setflags(write=self.arr.flags["WRITEABLE"])
        assert self.pyarr.flags["OWNDATA"], (obj, intent)
        self.pyarr_attr = wrap.array_attrs(self.pyarr)

        if len(dims) > 1:
            if self.intent.is_intent("c"):
                assert not self.pyarr.flags["FORTRAN"]
                assert self.pyarr.flags["CONTIGUOUS"]
                assert (not self.pyarr_attr[6] & wrap.FORTRAN)
            else:
                assert self.pyarr.flags["FORTRAN"]
                assert not self.pyarr.flags["CONTIGUOUS"]
                assert (self.pyarr_attr[6] & wrap.FORTRAN)

        assert self.arr_attr[1] == self.pyarr_attr[1]  # nd
        assert self.arr_attr[2] == self.pyarr_attr[2]  # dimensions
        if self.arr_attr[1] <= 1:
            assert self.arr_attr[3] == self.pyarr_attr[3], repr((
                self.arr_attr[3],
                self.pyarr_attr[3],
                self.arr.tobytes(),
                self.pyarr.tobytes(),
            ))  # strides
        assert self.arr_attr[5][-2:] == self.pyarr_attr[5][-2:], repr((
            self.arr_attr[5], self.pyarr_attr[5]
            ))  # descr
        assert self.arr_attr[6] == self.pyarr_attr[6], repr((
            self.arr_attr[6],
            self.pyarr_attr[6],
            flags2names(0 * self.arr_attr[6] - self.pyarr_attr[6]),
            flags2names(self.arr_attr[6]),
            intent,
        ))  # flags

        if intent.is_intent("cache"):
            assert self.arr_attr[5][3] >= self.type.elsize
        else:
            assert self.arr_attr[5][3] == self.type.elsize
            assert (self.arr_equal(self.pyarr, self.arr))

        if isinstance(self.obj, np.ndarray):
            if typ.elsize == Type(obj.dtype).elsize:
                if not intent.is_intent("copy") and self.arr_attr[1] <= 1:
                    assert self.has_shared_memory()

    def arr_equal(self, arr1, arr2):
        if arr1.shape != arr2.shape:
            return False
        return (arr1 == arr2).all()

    def __str__(self):
        return str(self.arr)

    def has_shared_memory(self):
        """Check that created array shares data with input array."""
        if self.obj is self.arr:
            return True
        if not isinstance(self.obj, np.ndarray):
            return False
        obj_attr = wrap.array_attrs(self.obj)
        return obj_attr[0] == self.arr_attr[0]


class TestIntent:
    def test_in_out(self):
        assert str(intent.in_.out) == "intent(in,out)"
        assert intent.in_.c.is_intent("c")
        assert not intent.in_.c.is_intent_exact("c")
        assert intent.in_.c.is_intent_exact("c", "in")
        assert intent.in_.c.is_intent_exact("in", "c")
        assert not intent.in_.is_intent("c")


class TestSharedMemory:

    @pytest.fixture(autouse=True, scope="class", params=_type_names)
    def setup_type(self, request):
        request.cls.type = Type(request.param)
        request.cls.array = lambda self, dims, intent, obj: Array(
            Type(request.param), dims, intent, obj)

    @property
    def num2seq(self):
        if self.type.NAME.startswith('STRING'):
            elsize = self.type.elsize
            return ['1' * elsize, '2' * elsize]
        return [1, 2]

    @property
    def num23seq(self):
        if self.type.NAME.startswith('STRING'):
            elsize = self.type.elsize
            return [['1' * elsize, '2' * elsize, '3' * elsize],
                    ['4' * elsize, '5' * elsize, '6' * elsize]]
        return [[1, 2, 3], [4, 5, 6]]

    def test_in_from_2seq(self):
        a = self.array([2], intent.in_, self.num2seq)
        assert not a.has_shared_memory()

    def test_in_from_2casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num2seq, dtype=t.dtype)
            a = self.array([len(self.num2seq)], intent.in_, obj)
            if t.elsize == self.type.elsize:
                assert a.has_shared_memory(), repr((self.type.dtype, t.dtype))
            else:
                assert not a.has_shared_memory()

    @pytest.mark.parametrize("write", ["w", "ro"])
    @pytest.mark.parametrize("order", ["C", "F"])
    @pytest.mark.parametrize("inp", ["2seq", "23seq"])
    def test_in_nocopy(self, write, order, inp):
        """Test if intent(in) array can be passed without copies"""
        seq = getattr(self, "num" + inp)
        obj = np.array(seq, dtype=self.type.dtype, order=order)
        obj.setflags(write=(write == 'w'))
        a = self.array(obj.shape,
                       ((order == 'C' and intent.in_.c) or intent.in_), obj)
        assert a.has_shared_memory()

    def test_inout_2seq(self):
        obj = np.array(self.num2seq, dtype=self.type.dtype)
        a = self.array([len(self.num2seq)], intent.inout, obj)
        assert a.has_shared_memory()

        try:
            a = self.array([2], intent.in_.inout, self.num2seq)
        except TypeError as msg:
            if not str(msg).startswith(
                    "failed to initialize intent(inout|inplace|cache) array"):
                raise
        else:
            raise SystemError("intent(inout) should have failed on sequence")

    def test_f_inout_23seq(self):
        obj = np.array(self.num23seq, dtype=self.type.dtype, order="F")
        shape = (len(self.num23seq), len(self.num23seq[0]))
        a = self.array(shape, intent.in_.inout, obj)
        assert a.has_shared_memory()

        obj = np.array(self.num23seq, dtype=self.type.dtype, order="C")
        shape = (len(self.num23seq), len(self.num23seq[0]))
        try:
            a = self.array(shape, intent.in_.inout, obj)
        except ValueError as msg:
            if not str(msg).startswith(
                    "failed to initialize intent(inout) array"):
                raise
        else:
            raise SystemError(
                "intent(inout) should have failed on improper array")

    def test_c_inout_23seq(self):
        obj = np.array(self.num23seq, dtype=self.type.dtype)
        shape = (len(self.num23seq), len(self.num23seq[0]))
        a = self.array(shape, intent.in_.c.inout, obj)
        assert a.has_shared_memory()

    def test_in_copy_from_2casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num2seq, dtype=t.dtype)
            a = self.array([len(self.num2seq)], intent.in_.copy, obj)
            assert not a.has_shared_memory()

    def test_c_in_from_23seq(self):
        a = self.array(
            [len(self.num23seq), len(self.num23seq[0])], intent.in_,
            self.num23seq)
        assert not a.has_shared_memory()

    def test_in_from_23casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num23seq, dtype=t.dtype)
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_, obj)
            assert not a.has_shared_memory()

    def test_f_in_from_23casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num23seq, dtype=t.dtype, order="F")
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_, obj)
            if t.elsize == self.type.elsize:
                assert a.has_shared_memory()
            else:
                assert not a.has_shared_memory()

    def test_c_in_from_23casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num23seq, dtype=t.dtype)
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_.c, obj)
            if t.elsize == self.type.elsize:
                assert a.has_shared_memory()
            else:
                assert not a.has_shared_memory()

    def test_f_copy_in_from_23casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num23seq, dtype=t.dtype, order="F")
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_.copy,
                obj)
            assert not a.has_shared_memory()

    def test_c_copy_in_from_23casttype(self):
        for t in self.type.cast_types():
            obj = np.array(self.num23seq, dtype=t.dtype)
            a = self.array(
                [len(self.num23seq), len(self.num23seq[0])], intent.in_.c.copy,
                obj)
            assert not a.has_shared_memory()

    def test_in_cache_from_2casttype(self):
        for t in self.type.all_types():
            if t.elsize != self.type.elsize:
                continue
            obj = np.array(self.num2seq, dtype=t.dtype)
            shape = (len(self.num2seq), )
            a = self.array(shape, intent.in_.c.cache, obj)
            assert a.has_shared_memory()

            a = self.array(shape, intent.in_.cache, obj)
            assert a.has_shared_memory()

            obj = np.array(self.num2seq, dtype=t.dtype, order="F")
            a = self.array(shape, intent.in_.c.cache, obj)
            assert a.has_shared_memory()

            a = self.array(shape, intent.in_.cache, obj)
            assert a.has_shared_memory(), repr(t.dtype)

            try:
                a = self.array(shape, intent.in_.cache, obj[::-1])
            except ValueError as msg:
                if not str(msg).startswith(
                        "failed to initialize intent(cache) array"):
                    raise
            else:
                raise SystemError(
                    "intent(cache) should have failed on multisegmented array")

    def test_in_cache_from_2casttype_failure(self):
        for t in self.type.all_types():
            if t.NAME == 'STRING':
                # string elsize is 0, so skipping the test
                continue
            if t.elsize >= self.type.elsize:
                continue
            is_int = np.issubdtype(t.dtype, np.integer)
            if is_int and int(self.num2seq[0]) > np.iinfo(t.dtype).max:
                # skip test if num2seq would trigger an overflow error
                continue
            obj = np.array(self.num2seq, dtype=t.dtype)
            shape = (len(self.num2seq), )
            try:
                self.array(shape, intent.in_.cache, obj)  # Should succeed
            except ValueError as msg:
                if not str(msg).startswith(
                        "failed to initialize intent(cache) array"):
                    raise
            else:
                raise SystemError(
                    "intent(cache) should have failed on smaller array")

    def test_cache_hidden(self):
        shape = (2, )
        a = self.array(shape, intent.cache.hide, None)
        assert a.arr.shape == shape

        shape = (2, 3)
        a = self.array(shape, intent.cache.hide, None)
        assert a.arr.shape == shape

        shape = (-1, 3)
        try:
            a = self.array(shape, intent.cache.hide, None)
        except ValueError as msg:
            if not str(msg).startswith(
                    "failed to create intent(cache|hide)|optional array"):
                raise
        else:
            raise SystemError(
                "intent(cache) should have failed on undefined dimensions")

    def test_hidden(self):
        shape = (2, )
        a = self.array(shape, intent.hide, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))

        shape = (2, 3)
        a = self.array(shape, intent.hide, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        assert a.arr.flags["FORTRAN"] and not a.arr.flags["CONTIGUOUS"]

        shape = (2, 3)
        a = self.array(shape, intent.c.hide, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        assert not a.arr.flags["FORTRAN"] and a.arr.flags["CONTIGUOUS"]

        shape = (-1, 3)
        try:
            a = self.array(shape, intent.hide, None)
        except ValueError as msg:
            if not str(msg).startswith(
                    "failed to create intent(cache|hide)|optional array"):
                raise
        else:
            raise SystemError(
                "intent(hide) should have failed on undefined dimensions")

    def test_optional_none(self):
        shape = (2, )
        a = self.array(shape, intent.optional, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))

        shape = (2, 3)
        a = self.array(shape, intent.optional, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        assert a.arr.flags["FORTRAN"] and not a.arr.flags["CONTIGUOUS"]

        shape = (2, 3)
        a = self.array(shape, intent.c.optional, None)
        assert a.arr.shape == shape
        assert a.arr_equal(a.arr, np.zeros(shape, dtype=self.type.dtype))
        assert not a.arr.flags["FORTRAN"] and a.arr.flags["CONTIGUOUS"]

    def test_optional_from_2seq(self):
        obj = self.num2seq
        shape = (len(obj), )
        a = self.array(shape, intent.optional, obj)
        assert a.arr.shape == shape
        assert not a.has_shared_memory()

    def test_optional_from_23seq(self):
        obj = self.num23seq
        shape = (len(obj), len(obj[0]))
        a = self.array(shape, intent.optional, obj)
        assert a.arr.shape == shape
        assert not a.has_shared_memory()

        a = self.array(shape, intent.optional.c, obj)
        assert a.arr.shape == shape
        assert not a.has_shared_memory()

    def test_inplace(self):
        obj = np.array(self.num23seq, dtype=self.type.dtype)
        assert not obj.flags["FORTRAN"] and obj.flags["CONTIGUOUS"]
        shape = obj.shape
        a = self.array(shape, intent.inplace, obj)
        assert obj[1][2] == a.arr[1][2], repr((obj, a.arr))
        a.arr[1][2] = 54
        assert obj[1][2] == a.arr[1][2] == np.array(54, dtype=self.type.dtype)
        assert a.arr is obj
        assert obj.flags["FORTRAN"]  # obj attributes are changed inplace!
        assert not obj.flags["CONTIGUOUS"]

    def test_inplace_from_casttype(self):
        for t in self.type.cast_types():
            if t is self.type:
                continue
            obj = np.array(self.num23seq, dtype=t.dtype)
            assert obj.dtype.type == t.type
            assert obj.dtype.type is not self.type.type
            assert not obj.flags["FORTRAN"] and obj.flags["CONTIGUOUS"]
            shape = obj.shape
            a = self.array(shape, intent.inplace, obj)
            assert obj[1][2] == a.arr[1][2], repr((obj, a.arr))
            a.arr[1][2] = 54
            assert obj[1][2] == a.arr[1][2] == np.array(54,
                                                        dtype=self.type.dtype)
            assert a.arr is obj
            assert obj.flags["FORTRAN"]  # obj attributes changed inplace!
            assert not obj.flags["CONTIGUOUS"]
            assert obj.dtype.type is self.type.type  # obj changed inplace!

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\aggregate\test_other.py ===
"""
test all other .agg behavior
"""

import datetime as dt
from functools import partial

import numpy as np
import pytest

from pandas.errors import SpecificationError

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    PeriodIndex,
    Series,
    date_range,
    period_range,
)
import pandas._testing as tm

from pandas.io.formats.printing import pprint_thing


def test_agg_partial_failure_raises():
    # GH#43741

    df = DataFrame(
        {
            "data1": np.random.default_rng(2).standard_normal(5),
            "data2": np.random.default_rng(2).standard_normal(5),
            "key1": ["a", "a", "b", "b", "a"],
            "key2": ["one", "two", "one", "two", "one"],
        }
    )
    grouped = df.groupby("key1")

    def peak_to_peak(arr):
        return arr.max() - arr.min()

    with pytest.raises(TypeError, match="unsupported operand type"):
        grouped.agg([peak_to_peak])

    with pytest.raises(TypeError, match="unsupported operand type"):
        grouped.agg(peak_to_peak)


def test_agg_datetimes_mixed():
    data = [[1, "2012-01-01", 1.0], [2, "2012-01-02", 2.0], [3, None, 3.0]]

    df1 = DataFrame(
        {
            "key": [x[0] for x in data],
            "date": [x[1] for x in data],
            "value": [x[2] for x in data],
        }
    )

    data = [
        [
            row[0],
            (dt.datetime.strptime(row[1], "%Y-%m-%d").date() if row[1] else None),
            row[2],
        ]
        for row in data
    ]

    df2 = DataFrame(
        {
            "key": [x[0] for x in data],
            "date": [x[1] for x in data],
            "value": [x[2] for x in data],
        }
    )

    df1["weights"] = df1["value"] / df1["value"].sum()
    gb1 = df1.groupby("date").aggregate("sum")

    df2["weights"] = df1["value"] / df1["value"].sum()
    gb2 = df2.groupby("date").aggregate("sum")

    assert len(gb1) == len(gb2)


def test_agg_period_index():
    prng = period_range("2012-1-1", freq="M", periods=3)
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)), index=prng)
    rs = df.groupby(level=0).sum()
    assert isinstance(rs.index, PeriodIndex)

    # GH 3579
    index = period_range(start="1999-01", periods=5, freq="M")
    s1 = Series(np.random.default_rng(2).random(len(index)), index=index)
    s2 = Series(np.random.default_rng(2).random(len(index)), index=index)
    df = DataFrame.from_dict({"s1": s1, "s2": s2})
    grouped = df.groupby(df.index.month)
    list(grouped)


def test_agg_dict_parameter_cast_result_dtypes():
    # GH 12821

    df = DataFrame(
        {
            "class": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "time": date_range("1/1/2011", periods=8, freq="h"),
        }
    )
    df.loc[[0, 1, 2, 5], "time"] = None

    # test for `first` function
    exp = df.loc[[0, 3, 4, 6]].set_index("class")
    grouped = df.groupby("class")
    tm.assert_frame_equal(grouped.first(), exp)
    tm.assert_frame_equal(grouped.agg("first"), exp)
    tm.assert_frame_equal(grouped.agg({"time": "first"}), exp)
    tm.assert_series_equal(grouped.time.first(), exp["time"])
    tm.assert_series_equal(grouped.time.agg("first"), exp["time"])

    # test for `last` function
    exp = df.loc[[0, 3, 4, 7]].set_index("class")
    grouped = df.groupby("class")
    tm.assert_frame_equal(grouped.last(), exp)
    tm.assert_frame_equal(grouped.agg("last"), exp)
    tm.assert_frame_equal(grouped.agg({"time": "last"}), exp)
    tm.assert_series_equal(grouped.time.last(), exp["time"])
    tm.assert_series_equal(grouped.time.agg("last"), exp["time"])

    # count
    exp = Series([2, 2, 2, 2], index=Index(list("ABCD"), name="class"), name="time")
    tm.assert_series_equal(grouped.time.agg(len), exp)
    tm.assert_series_equal(grouped.time.size(), exp)

    exp = Series([0, 1, 1, 2], index=Index(list("ABCD"), name="class"), name="time")
    tm.assert_series_equal(grouped.time.count(), exp)


def test_agg_cast_results_dtypes():
    # similar to GH12821
    # xref #11444
    u = [dt.datetime(2015, x + 1, 1) for x in range(12)]
    v = list("aaabbbbbbccd")
    df = DataFrame({"X": v, "Y": u})

    result = df.groupby("X")["Y"].agg(len)
    expected = df.groupby("X")["Y"].count()
    tm.assert_series_equal(result, expected)


def test_aggregate_float64_no_int64():
    # see gh-11199
    df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 2, 4, 5], "c": [1, 2, 3, 4, 5]})

    expected = DataFrame({"a": [1, 2.5, 4, 5]}, index=[1, 2, 4, 5])
    expected.index.name = "b"

    result = df.groupby("b")[["a"]].mean()
    tm.assert_frame_equal(result, expected)

    expected = DataFrame({"a": [1, 2.5, 4, 5], "c": [1, 2.5, 4, 5]}, index=[1, 2, 4, 5])
    expected.index.name = "b"

    result = df.groupby("b")[["a", "c"]].mean()
    tm.assert_frame_equal(result, expected)


def test_aggregate_api_consistency():
    # GH 9052
    # make sure that the aggregates via dict
    # are consistent
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )

    grouped = df.groupby(["A", "B"])
    c_mean = grouped["C"].mean()
    c_sum = grouped["C"].sum()
    d_mean = grouped["D"].mean()
    d_sum = grouped["D"].sum()

    result = grouped["D"].agg(["sum", "mean"])
    expected = pd.concat([d_sum, d_mean], axis=1)
    expected.columns = ["sum", "mean"]
    tm.assert_frame_equal(result, expected, check_like=True)

    result = grouped.agg(["sum", "mean"])
    expected = pd.concat([c_sum, c_mean, d_sum, d_mean], axis=1)
    expected.columns = MultiIndex.from_product([["C", "D"], ["sum", "mean"]])
    tm.assert_frame_equal(result, expected, check_like=True)

    result = grouped[["D", "C"]].agg(["sum", "mean"])
    expected = pd.concat([d_sum, d_mean, c_sum, c_mean], axis=1)
    expected.columns = MultiIndex.from_product([["D", "C"], ["sum", "mean"]])
    tm.assert_frame_equal(result, expected, check_like=True)

    result = grouped.agg({"C": "mean", "D": "sum"})
    expected = pd.concat([d_sum, c_mean], axis=1)
    tm.assert_frame_equal(result, expected, check_like=True)

    result = grouped.agg({"C": ["mean", "sum"], "D": ["mean", "sum"]})
    expected = pd.concat([c_mean, c_sum, d_mean, d_sum], axis=1)
    expected.columns = MultiIndex.from_product([["C", "D"], ["mean", "sum"]])

    msg = r"Column\(s\) \['r', 'r2'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        grouped[["D", "C"]].agg({"r": "sum", "r2": "mean"})


def test_agg_dict_renaming_deprecation():
    # 15931
    df = DataFrame({"A": [1, 1, 1, 2, 2], "B": range(5), "C": range(5)})

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        df.groupby("A").agg(
            {"B": {"foo": ["sum", "max"]}, "C": {"bar": ["count", "min"]}}
        )

    msg = r"Column\(s\) \['ma'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.groupby("A")[["B", "C"]].agg({"ma": "max"})

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        df.groupby("A").B.agg({"foo": "count"})


def test_agg_compat():
    # GH 12334
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )

    g = df.groupby(["A", "B"])

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"C": ["sum", "std"]})

    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"C": "sum", "D": "std"})


def test_agg_nested_dicts():
    # API change for disallowing these types of nested dicts
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )

    g = df.groupby(["A", "B"])

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        g.aggregate({"r1": {"C": ["mean", "sum"]}, "r2": {"D": ["mean", "sum"]}})

    with pytest.raises(SpecificationError, match=msg):
        g.agg({"C": {"ra": ["mean", "std"]}, "D": {"rb": ["mean", "std"]}})

    # same name as the original column
    # GH9052
    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"result1": np.sum, "result2": np.mean})

    with pytest.raises(SpecificationError, match=msg):
        g["D"].agg({"D": np.sum, "result2": np.mean})


def test_agg_item_by_item_raise_typeerror():
    df = DataFrame(np.random.default_rng(2).integers(10, size=(20, 10)))

    def raiseException(df):
        pprint_thing("----------------------------------------")
        pprint_thing(df.to_string())
        raise TypeError("test")

    with pytest.raises(TypeError, match="test"):
        df.groupby(0).agg(raiseException)


def test_series_agg_multikey():
    ts = Series(
        np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
    )
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])

    result = grouped.agg("sum")
    expected = grouped.sum()
    tm.assert_series_equal(result, expected)


def test_series_agg_multi_pure_python():
    data = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )

    def bad(x):
        if isinstance(x.values, np.ndarray):
            assert len(x.values.base) > 0
        return "foo"

    result = data.groupby(["A", "B"]).agg(bad)
    expected = data.groupby(["A", "B"]).agg(lambda x: "foo")
    tm.assert_frame_equal(result, expected)


def test_agg_consistency():
    # agg with ([]) and () not consistent
    # GH 6715
    def P1(a):
        return np.percentile(a.dropna(), q=1)

    df = DataFrame(
        {
            "col1": [1, 2, 3, 4],
            "col2": [10, 25, 26, 31],
            "date": [
                dt.date(2013, 2, 10),
                dt.date(2013, 2, 10),
                dt.date(2013, 2, 11),
                dt.date(2013, 2, 11),
            ],
        }
    )

    g = df.groupby("date")

    expected = g.agg([P1])
    expected.columns = expected.columns.levels[0]

    result = g.agg(P1)
    tm.assert_frame_equal(result, expected)


def test_agg_callables():
    # GH 7929
    df = DataFrame({"foo": [1, 2], "bar": [3, 4]}).astype(np.int64)

    class fn_class:
        def __call__(self, x):
            return sum(x)

    equiv_callables = [
        sum,
        np.sum,
        lambda x: sum(x),
        lambda x: x.sum(),
        partial(sum),
        fn_class(),
    ]

    expected = df.groupby("foo").agg("sum")
    for ecall in equiv_callables:
        warn = FutureWarning if ecall is sum or ecall is np.sum else None
        msg = "using DataFrameGroupBy.sum"
        with tm.assert_produces_warning(warn, match=msg):
            result = df.groupby("foo").agg(ecall)
        tm.assert_frame_equal(result, expected)


def test_agg_over_numpy_arrays():
    # GH 3788
    df = DataFrame(
        [
            [1, np.array([10, 20, 30])],
            [1, np.array([40, 50, 60])],
            [2, np.array([20, 30, 40])],
        ],
        columns=["category", "arraydata"],
    )
    gb = df.groupby("category")

    expected_data = [[np.array([50, 70, 90])], [np.array([20, 30, 40])]]
    expected_index = Index([1, 2], name="category")
    expected_column = ["arraydata"]
    expected = DataFrame(expected_data, index=expected_index, columns=expected_column)

    alt = gb.sum(numeric_only=False)
    tm.assert_frame_equal(alt, expected)

    result = gb.agg("sum", numeric_only=False)
    tm.assert_frame_equal(result, expected)

    # FIXME: the original version of this test called `gb.agg(sum)`
    #  and that raises TypeError if `numeric_only=False` is passed


@pytest.mark.parametrize("as_period", [True, False])
def test_agg_tzaware_non_datetime_result(as_period):
    # discussed in GH#29589, fixed in GH#29641, operating on tzaware values
    #  with function that is not dtype-preserving
    dti = date_range("2012-01-01", periods=4, tz="UTC")
    if as_period:
        dti = dti.tz_localize(None).to_period("D")

    df = DataFrame({"a": [0, 0, 1, 1], "b": dti})
    gb = df.groupby("a")

    # Case that _does_ preserve the dtype
    result = gb["b"].agg(lambda x: x.iloc[0])
    expected = Series(dti[::2], name="b")
    expected.index.name = "a"
    tm.assert_series_equal(result, expected)

    # Cases that do _not_ preserve the dtype
    result = gb["b"].agg(lambda x: x.iloc[0].year)
    expected = Series([2012, 2012], name="b")
    expected.index.name = "a"
    tm.assert_series_equal(result, expected)

    result = gb["b"].agg(lambda x: x.iloc[-1] - x.iloc[0])
    expected = Series([pd.Timedelta(days=1), pd.Timedelta(days=1)], name="b")
    expected.index.name = "a"
    if as_period:
        expected = Series([pd.offsets.Day(1), pd.offsets.Day(1)], name="b")
        expected.index.name = "a"
    tm.assert_series_equal(result, expected)


def test_agg_timezone_round_trip():
    # GH 15426
    ts = pd.Timestamp("2016-01-01 12:00:00", tz="US/Pacific")
    df = DataFrame({"a": 1, "b": [ts + dt.timedelta(minutes=nn) for nn in range(10)]})

    result1 = df.groupby("a")["b"].agg("min").iloc[0]
    result2 = df.groupby("a")["b"].agg(lambda x: np.min(x)).iloc[0]
    result3 = df.groupby("a")["b"].min().iloc[0]

    assert result1 == ts
    assert result2 == ts
    assert result3 == ts

    dates = [
        pd.Timestamp(f"2016-01-0{i:d} 12:00:00", tz="US/Pacific") for i in range(1, 5)
    ]
    df = DataFrame({"A": ["a", "b"] * 2, "B": dates})
    grouped = df.groupby("A")

    ts = df["B"].iloc[0]
    assert ts == grouped.nth(0)["B"].iloc[0]
    assert ts == grouped.head(1)["B"].iloc[0]
    assert ts == grouped.first()["B"].iloc[0]

    # GH#27110 applying iloc should return a DataFrame
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ts == grouped.apply(lambda x: x.iloc[0]).iloc[0, 1]

    ts = df["B"].iloc[2]
    assert ts == grouped.last()["B"].iloc[0]

    # GH#27110 applying iloc should return a DataFrame
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert ts == grouped.apply(lambda x: x.iloc[-1]).iloc[0, 1]


def test_sum_uint64_overflow():
    # see gh-14758
    # Convert to uint64 and don't overflow
    df = DataFrame([[1, 2], [3, 4], [5, 6]], dtype=object)
    df = df + 9223372036854775807

    index = Index(
        [9223372036854775808, 9223372036854775810, 9223372036854775812], dtype=np.uint64
    )
    expected = DataFrame(
        {1: [9223372036854775809, 9223372036854775811, 9223372036854775813]},
        index=index,
        dtype=object,
    )

    expected.index.name = 0
    result = df.groupby(0).sum(numeric_only=False)
    tm.assert_frame_equal(result, expected)

    # out column is non-numeric, so with numeric_only=True it is dropped
    result2 = df.groupby(0).sum(numeric_only=True)
    expected2 = expected[[]]
    tm.assert_frame_equal(result2, expected2)


@pytest.mark.parametrize(
    "structure, expected",
    [
        (tuple, DataFrame({"C": {(1, 1): (1, 1, 1), (3, 4): (3, 4, 4)}})),
        (list, DataFrame({"C": {(1, 1): [1, 1, 1], (3, 4): [3, 4, 4]}})),
        (
            lambda x: tuple(x),
            DataFrame({"C": {(1, 1): (1, 1, 1), (3, 4): (3, 4, 4)}}),
        ),
        (
            lambda x: list(x),
            DataFrame({"C": {(1, 1): [1, 1, 1], (3, 4): [3, 4, 4]}}),
        ),
    ],
)
def test_agg_structs_dataframe(structure, expected):
    df = DataFrame(
        {"A": [1, 1, 1, 3, 3, 3], "B": [1, 1, 1, 4, 4, 4], "C": [1, 1, 1, 3, 4, 4]}
    )

    result = df.groupby(["A", "B"]).aggregate(structure)
    expected.index.names = ["A", "B"]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "structure, expected",
    [
        (tuple, Series([(1, 1, 1), (3, 4, 4)], index=[1, 3], name="C")),
        (list, Series([[1, 1, 1], [3, 4, 4]], index=[1, 3], name="C")),
        (lambda x: tuple(x), Series([(1, 1, 1), (3, 4, 4)], index=[1, 3], name="C")),
        (lambda x: list(x), Series([[1, 1, 1], [3, 4, 4]], index=[1, 3], name="C")),
    ],
)
def test_agg_structs_series(structure, expected):
    # Issue #18079
    df = DataFrame(
        {"A": [1, 1, 1, 3, 3, 3], "B": [1, 1, 1, 4, 4, 4], "C": [1, 1, 1, 3, 4, 4]}
    )

    result = df.groupby("A")["C"].aggregate(structure)
    expected.index.name = "A"
    tm.assert_series_equal(result, expected)


def test_agg_category_nansum(observed):
    categories = ["a", "b", "c"]
    df = DataFrame(
        {"A": pd.Categorical(["a", "a", "b"], categories=categories), "B": [1, 2, 3]}
    )
    msg = "using SeriesGroupBy.sum"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby("A", observed=observed).B.agg(np.nansum)
    expected = Series(
        [3, 3, 0],
        index=pd.CategoricalIndex(["a", "b", "c"], categories=categories, name="A"),
        name="B",
    )
    if observed:
        expected = expected[expected != 0]
    tm.assert_series_equal(result, expected)


def test_agg_list_like_func():
    # GH 18473
    df = DataFrame({"A": [str(x) for x in range(3)], "B": [str(x) for x in range(3)]})
    grouped = df.groupby("A", as_index=False, sort=False)
    result = grouped.agg({"B": lambda x: list(x)})
    expected = DataFrame(
        {"A": [str(x) for x in range(3)], "B": [[str(x)] for x in range(3)]}
    )
    tm.assert_frame_equal(result, expected)


def test_agg_lambda_with_timezone():
    # GH 23683
    df = DataFrame(
        {
            "tag": [1, 1],
            "date": [
                pd.Timestamp("2018-01-01", tz="UTC"),
                pd.Timestamp("2018-01-02", tz="UTC"),
            ],
        }
    )
    result = df.groupby("tag").agg({"date": lambda e: e.head(1)})
    expected = DataFrame(
        [pd.Timestamp("2018-01-01", tz="UTC")],
        index=Index([1], name="tag"),
        columns=["date"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "err_cls",
    [
        NotImplementedError,
        RuntimeError,
        KeyError,
        IndexError,
        OSError,
        ValueError,
        ArithmeticError,
        AttributeError,
    ],
)
def test_groupby_agg_err_catching(err_cls):
    # make sure we suppress anything other than TypeError or AssertionError
    #  in _python_agg_general

    # Use a non-standard EA to make sure we don't go down ndarray paths
    from pandas.tests.extension.decimal.array import (
        DecimalArray,
        make_data,
        to_decimal,
    )

    data = make_data()[:5]
    df = DataFrame(
        {"id1": [0, 0, 0, 1, 1], "id2": [0, 1, 0, 1, 1], "decimals": DecimalArray(data)}
    )

    expected = Series(to_decimal([data[0], data[3]]))

    def weird_func(x):
        # weird function that raise something other than TypeError or IndexError
        #  in _python_agg_general
        if len(x) == 0:
            raise err_cls
        return x.iloc[0]

    result = df["decimals"].groupby(df["id1"]).agg(weird_func)
    tm.assert_series_equal(result, expected, check_names=False)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\tests\test_polygon.py ===
from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.functions.elementary.trigonometric import tan
from sympy.geometry import (Circle, Ellipse, GeometryError, Point, Point2D,
                            Polygon, Ray, RegularPolygon, Segment, Triangle,
                            are_similar, convex_hull, intersection, Line, Ray2D)
from sympy.testing.pytest import raises, slow, warns
from sympy.core.random import verify_numerically
from sympy.geometry.polygon import rad, deg
from sympy.integrals.integrals import integrate
from sympy.utilities.iterables import rotate_left


def feq(a, b):
    """Test if two floating point values are 'equal'."""
    t_float = Float("1.0E-10")
    return -t_float < a - b < t_float

@slow
def test_polygon():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    q = Symbol('q', real=True)
    u = Symbol('u', real=True)
    v = Symbol('v', real=True)
    w = Symbol('w', real=True)
    x1 = Symbol('x1', real=True)
    half = S.Half
    a, b, c = Point(0, 0), Point(2, 0), Point(3, 3)
    t = Triangle(a, b, c)
    assert Polygon(Point(0, 0)) == Point(0, 0)
    assert Polygon(a, Point(1, 0), b, c) == t
    assert Polygon(Point(1, 0), b, c, a) == t
    assert Polygon(b, c, a, Point(1, 0)) == t
    # 2 "remove folded" tests
    assert Polygon(a, Point(3, 0), b, c) == t
    assert Polygon(a, b, Point(3, -1), b, c) == t
    # remove multiple collinear points
    assert Polygon(Point(-4, 15), Point(-11, 15), Point(-15, 15),
        Point(-15, 33/5), Point(-15, -87/10), Point(-15, -15),
        Point(-42/5, -15), Point(-2, -15), Point(7, -15), Point(15, -15),
        Point(15, -3), Point(15, 10), Point(15, 15)) == \
        Polygon(Point(-15, -15), Point(15, -15), Point(15, 15), Point(-15, 15))

    p1 = Polygon(
        Point(0, 0), Point(3, -1),
        Point(6, 0), Point(4, 5),
        Point(2, 3), Point(0, 3))
    p2 = Polygon(
        Point(6, 0), Point(3, -1),
        Point(0, 0), Point(0, 3),
        Point(2, 3), Point(4, 5))
    p3 = Polygon(
        Point(0, 0), Point(3, 0),
        Point(5, 2), Point(4, 4))
    p4 = Polygon(
        Point(0, 0), Point(4, 4),
        Point(5, 2), Point(3, 0))
    p5 = Polygon(
        Point(0, 0), Point(4, 4),
        Point(0, 4))
    p6 = Polygon(
        Point(-11, 1), Point(-9, 6.6),
        Point(-4, -3), Point(-8.4, -8.7))
    p7 = Polygon(
        Point(x, y), Point(q, u),
        Point(v, w))
    p8 = Polygon(
        Point(x, y), Point(v, w),
        Point(q, u))
    p9 = Polygon(
        Point(0, 0), Point(4, 4),
        Point(3, 0), Point(5, 2))
    p10 = Polygon(
        Point(0, 2), Point(2, 2),
        Point(0, 0), Point(2, 0))
    p11 = Polygon(Point(0, 0), 1, n=3)
    p12 = Polygon(Point(0, 0), 1, 0, n=3)
    p13 = Polygon(
        Point(0, 0),Point(8, 8),
        Point(23, 20),Point(0, 20))
    p14 = Polygon(*rotate_left(p13.args, 1))


    r = Ray(Point(-9, 6.6), Point(-9, 5.5))
    #
    # General polygon
    #
    assert p1 == p2
    assert len(p1.args) == 6
    assert len(p1.sides) == 6
    assert p1.perimeter == 5 + 2*sqrt(10) + sqrt(29) + sqrt(8)
    assert p1.area == 22
    assert not p1.is_convex()
    assert Polygon((-1, 1), (2, -1), (2, 1), (-1, -1), (3, 0)
        ).is_convex() is False
    # ensure convex for both CW and CCW point specification
    assert p3.is_convex()
    assert p4.is_convex()
    dict5 = p5.angles
    assert dict5[Point(0, 0)] == pi / 4
    assert dict5[Point(0, 4)] == pi / 2
    assert p5.encloses_point(Point(x, y)) is None
    assert p5.encloses_point(Point(1, 3))
    assert p5.encloses_point(Point(0, 0)) is False
    assert p5.encloses_point(Point(4, 0)) is False
    assert p1.encloses(Circle(Point(2.5, 2.5), 5)) is False
    assert p1.encloses(Ellipse(Point(2.5, 2), 5, 6)) is False
    assert p5.plot_interval('x') == [x, 0, 1]
    assert p5.distance(
        Polygon(Point(10, 10), Point(14, 14), Point(10, 14))) == 6 * sqrt(2)
    assert p5.distance(
        Polygon(Point(1, 8), Point(5, 8), Point(8, 12), Point(1, 12))) == 4
    with warns(UserWarning, \
               match="Polygons may intersect producing erroneous output"):
        Polygon(Point(0, 0), Point(1, 0), Point(1, 1)).distance(
                Polygon(Point(0, 0), Point(0, 1), Point(1, 1)))
    assert hash(p5) == hash(Polygon(Point(0, 0), Point(4, 4), Point(0, 4)))
    assert hash(p1) == hash(p2)
    assert hash(p7) == hash(p8)
    assert hash(p3) != hash(p9)
    assert p5 == Polygon(Point(4, 4), Point(0, 4), Point(0, 0))
    assert Polygon(Point(4, 4), Point(0, 4), Point(0, 0)) in p5
    assert p5 != Point(0, 4)
    assert Point(0, 1) in p5
    assert p5.arbitrary_point('t').subs(Symbol('t', real=True), 0) == \
        Point(0, 0)
    raises(ValueError, lambda: Polygon(
        Point(x, 0), Point(0, y), Point(x, y)).arbitrary_point('x'))
    assert p6.intersection(r) == [Point(-9, Rational(-84, 13)), Point(-9, Rational(33, 5))]
    assert p10.area == 0
    assert p11 == RegularPolygon(Point(0, 0), 1, 3, 0)
    assert p11 == p12
    assert p11.vertices[0] == Point(1, 0)
    assert p11.args[0] == Point(0, 0)
    p11.spin(pi/2)
    assert p11.vertices[0] == Point(0, 1)
    #
    # Regular polygon
    #
    p1 = RegularPolygon(Point(0, 0), 10, 5)
    p2 = RegularPolygon(Point(0, 0), 5, 5)
    raises(GeometryError, lambda: RegularPolygon(Point(0, 0), Point(0,
           1), Point(1, 1)))
    raises(GeometryError, lambda: RegularPolygon(Point(0, 0), 1, 2))
    raises(ValueError, lambda: RegularPolygon(Point(0, 0), 1, 2.5))

    assert p1 != p2
    assert p1.interior_angle == pi*Rational(3, 5)
    assert p1.exterior_angle == pi*Rational(2, 5)
    assert p2.apothem == 5*cos(pi/5)
    assert p2.circumcenter == p1.circumcenter == Point(0, 0)
    assert p1.circumradius == p1.radius == 10
    assert p2.circumcircle == Circle(Point(0, 0), 5)
    assert p2.incircle == Circle(Point(0, 0), p2.apothem)
    assert p2.inradius == p2.apothem == (5 * (1 + sqrt(5)) / 4)
    p2.spin(pi / 10)
    dict1 = p2.angles
    assert dict1[Point(0, 5)] == 3 * pi / 5
    assert p1.is_convex()
    assert p1.rotation == 0
    assert p1.encloses_point(Point(0, 0))
    assert p1.encloses_point(Point(11, 0)) is False
    assert p2.encloses_point(Point(0, 4.9))
    p1.spin(pi/3)
    assert p1.rotation == pi/3
    assert p1.vertices[0] == Point(5, 5*sqrt(3))
    for var in p1.args:
        if isinstance(var, Point):
            assert var == Point(0, 0)
        else:
            assert var in (5, 10, pi / 3)
    assert p1 != Point(0, 0)
    assert p1 != p5

    # while spin works in place (notice that rotation is 2pi/3 below)
    # rotate returns a new object
    p1_old = p1
    assert p1.rotate(pi/3) == RegularPolygon(Point(0, 0), 10, 5, pi*Rational(2, 3))
    assert p1 == p1_old

    assert p1.area == (-250*sqrt(5) + 1250)/(4*tan(pi/5))
    assert p1.length == 20*sqrt(-sqrt(5)/8 + Rational(5, 8))
    assert p1.scale(2, 2) == \
        RegularPolygon(p1.center, p1.radius*2, p1._n, p1.rotation)
    assert RegularPolygon((0, 0), 1, 4).scale(2, 3) == \
        Polygon(Point(2, 0), Point(0, 3), Point(-2, 0), Point(0, -3))

    assert repr(p1) == str(p1)

    #
    # Angles
    #
    angles = p4.angles
    assert feq(angles[Point(0, 0)].evalf(), Float("0.7853981633974483"))
    assert feq(angles[Point(4, 4)].evalf(), Float("1.2490457723982544"))
    assert feq(angles[Point(5, 2)].evalf(), Float("1.8925468811915388"))
    assert feq(angles[Point(3, 0)].evalf(), Float("2.3561944901923449"))

    angles = p3.angles
    assert feq(angles[Point(0, 0)].evalf(), Float("0.7853981633974483"))
    assert feq(angles[Point(4, 4)].evalf(), Float("1.2490457723982544"))
    assert feq(angles[Point(5, 2)].evalf(), Float("1.8925468811915388"))
    assert feq(angles[Point(3, 0)].evalf(), Float("2.3561944901923449"))

    # https://github.com/sympy/sympy/issues/24885
    interior_angles_sum = sum(p13.angles.values())
    assert feq(interior_angles_sum, (len(p13.angles) - 2)*pi )
    interior_angles_sum = sum(p14.angles.values())
    assert feq(interior_angles_sum, (len(p14.angles) - 2)*pi )

    #
    # Triangle
    #
    p1 = Point(0, 0)
    p2 = Point(5, 0)
    p3 = Point(0, 5)
    t1 = Triangle(p1, p2, p3)
    t2 = Triangle(p1, p2, Point(Rational(5, 2), sqrt(Rational(75, 4))))
    t3 = Triangle(p1, Point(x1, 0), Point(0, x1))
    s1 = t1.sides
    assert Triangle(p1, p2, p1) == Polygon(p1, p2, p1) == Segment(p1, p2)
    raises(GeometryError, lambda: Triangle(Point(0, 0)))

    # Basic stuff
    assert Triangle(p1, p1, p1) == p1
    assert Triangle(p2, p2*2, p2*3) == Segment(p2, p2*3)
    assert t1.area == Rational(25, 2)
    assert t1.is_right()
    assert t2.is_right() is False
    assert t3.is_right()
    assert p1 in t1
    assert t1.sides[0] in t1
    assert Segment((0, 0), (1, 0)) in t1
    assert Point(5, 5) not in t2
    assert t1.is_convex()
    assert feq(t1.angles[p1].evalf(), pi.evalf()/2)

    assert t1.is_equilateral() is False
    assert t2.is_equilateral()
    assert t3.is_equilateral() is False
    assert are_similar(t1, t2) is False
    assert are_similar(t1, t3)
    assert are_similar(t2, t3) is False
    assert t1.is_similar(Point(0, 0)) is False
    assert t1.is_similar(t2) is False

    # Bisectors
    bisectors = t1.bisectors()
    assert bisectors[p1] == Segment(
        p1, Point(Rational(5, 2), Rational(5, 2)))
    assert t2.bisectors()[p2] == Segment(
        Point(5, 0), Point(Rational(5, 4), 5*sqrt(3)/4))
    p4 = Point(0, x1)
    assert t3.bisectors()[p4] == Segment(p4, Point(x1*(sqrt(2) - 1), 0))
    ic = (250 - 125*sqrt(2))/50
    assert t1.incenter == Point(ic, ic)

    # Inradius
    assert t1.inradius == t1.incircle.radius == 5 - 5*sqrt(2)/2
    assert t2.inradius == t2.incircle.radius == 5*sqrt(3)/6
    assert t3.inradius == t3.incircle.radius == x1**2/((2 + sqrt(2))*Abs(x1))

    # Exradius
    assert t1.exradii[t1.sides[2]] == 5*sqrt(2)/2

    # Excenters
    assert t1.excenters[t1.sides[2]] == Point2D(25*sqrt(2), -5*sqrt(2)/2)

    # Circumcircle
    assert t1.circumcircle.center == Point(2.5, 2.5)

    # Medians + Centroid
    m = t1.medians
    assert t1.centroid == Point(Rational(5, 3), Rational(5, 3))
    assert m[p1] == Segment(p1, Point(Rational(5, 2), Rational(5, 2)))
    assert t3.medians[p1] == Segment(p1, Point(x1/2, x1/2))
    assert intersection(m[p1], m[p2], m[p3]) == [t1.centroid]
    assert t1.medial == Triangle(Point(2.5, 0), Point(0, 2.5), Point(2.5, 2.5))

    # Nine-point circle
    assert t1.nine_point_circle == Circle(Point(2.5, 0),
                                          Point(0, 2.5), Point(2.5, 2.5))
    assert t1.nine_point_circle == Circle(Point(0, 0),
                                          Point(0, 2.5), Point(2.5, 2.5))

    # Perpendicular
    altitudes = t1.altitudes
    assert altitudes[p1] == Segment(p1, Point(Rational(5, 2), Rational(5, 2)))
    assert altitudes[p2].equals(s1[0])
    assert altitudes[p3] == s1[2]
    assert t1.orthocenter == p1
    t = S('''Triangle(
    Point(100080156402737/5000000000000, 79782624633431/500000000000),
    Point(39223884078253/2000000000000, 156345163124289/1000000000000),
    Point(31241359188437/1250000000000, 338338270939941/1000000000000000))''')
    assert t.orthocenter == S('''Point(-780660869050599840216997'''
    '''79471538701955848721853/80368430960602242240789074233100000000000000,'''
    '''20151573611150265741278060334545897615974257/16073686192120448448157'''
    '''8148466200000000000)''')

    # Ensure
    assert len(intersection(*bisectors.values())) == 1
    assert len(intersection(*altitudes.values())) == 1
    assert len(intersection(*m.values())) == 1

    # Distance
    p1 = Polygon(
        Point(0, 0), Point(1, 0),
        Point(1, 1), Point(0, 1))
    p2 = Polygon(
        Point(0, Rational(5)/4), Point(1, Rational(5)/4),
        Point(1, Rational(9)/4), Point(0, Rational(9)/4))
    p3 = Polygon(
        Point(1, 2), Point(2, 2),
        Point(2, 1))
    p4 = Polygon(
        Point(1, 1), Point(Rational(6)/5, 1),
        Point(1, Rational(6)/5))
    pt1 = Point(half, half)
    pt2 = Point(1, 1)

    '''Polygon to Point'''
    assert p1.distance(pt1) == half
    assert p1.distance(pt2) == 0
    assert p2.distance(pt1) == Rational(3)/4
    assert p3.distance(pt2) == sqrt(2)/2

    '''Polygon to Polygon'''
    # p1.distance(p2) emits a warning
    with warns(UserWarning, \
               match="Polygons may intersect producing erroneous output"):
        assert p1.distance(p2) == half/2

    assert p1.distance(p3) == sqrt(2)/2

    # p3.distance(p4) emits a warning
    with warns(UserWarning, \
               match="Polygons may intersect producing erroneous output"):
        assert p3.distance(p4) == (sqrt(2)/2 - sqrt(Rational(2)/25)/2)


def test_convex_hull():
    p = [Point(-5, -1), Point(-2, 1), Point(-2, -1), Point(-1, -3), \
         Point(0, 0), Point(1, 1), Point(2, 2), Point(2, -1), Point(3, 1), \
         Point(4, -1), Point(6, 2)]
    ch = Polygon(p[0], p[3], p[9], p[10], p[6], p[1])
    #test handling of duplicate points
    p.append(p[3])

    #more than 3 collinear points
    another_p = [Point(-45, -85), Point(-45, 85), Point(-45, 26), \
                 Point(-45, -24)]
    ch2 = Segment(another_p[0], another_p[1])

    assert convex_hull(*another_p) == ch2
    assert convex_hull(*p) == ch
    assert convex_hull(p[0]) == p[0]
    assert convex_hull(p[0], p[1]) == Segment(p[0], p[1])

    # no unique points
    assert convex_hull(*[p[-1]]*3) == p[-1]

    # collection of items
    assert convex_hull(*[Point(0, 0), \
                        Segment(Point(1, 0), Point(1, 1)), \
                        RegularPolygon(Point(2, 0), 2, 4)]) == \
        Polygon(Point(0, 0), Point(2, -2), Point(4, 0), Point(2, 2))


def test_encloses():
    # square with a dimpled left side
    s = Polygon(Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1), \
        Point(S.Half, S.Half))
    # the following is True if the polygon isn't treated as closing on itself
    assert s.encloses(Point(0, S.Half)) is False
    assert s.encloses(Point(S.Half, S.Half)) is False  # it's a vertex
    assert s.encloses(Point(Rational(3, 4), S.Half)) is True


def test_triangle_kwargs():
    assert Triangle(sss=(3, 4, 5)) == \
        Triangle(Point(0, 0), Point(3, 0), Point(3, 4))
    assert Triangle(asa=(30, 2, 30)) == \
        Triangle(Point(0, 0), Point(2, 0), Point(1, sqrt(3)/3))
    assert Triangle(sas=(1, 45, 2)) == \
        Triangle(Point(0, 0), Point(2, 0), Point(sqrt(2)/2, sqrt(2)/2))
    assert Triangle(sss=(1, 2, 5)) is None
    assert deg(rad(180)) == 180


def test_transform():
    pts = [Point(0, 0), Point(S.Half, Rational(1, 4)), Point(1, 1)]
    pts_out = [Point(-4, -10), Point(-3, Rational(-37, 4)), Point(-2, -7)]
    assert Triangle(*pts).scale(2, 3, (4, 5)) == Triangle(*pts_out)
    assert RegularPolygon((0, 0), 1, 4).scale(2, 3, (4, 5)) == \
        Polygon(Point(-2, -10), Point(-4, -7), Point(-6, -10), Point(-4, -13))
    # Checks for symmetric scaling
    assert RegularPolygon((0, 0), 1, 4).scale(2, 2) == \
        RegularPolygon(Point2D(0, 0), 2, 4, 0)

def test_reflect():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    b = Symbol('b')
    m = Symbol('m')
    l = Line((0, b), slope=m)
    p = Point(x, y)
    r = p.reflect(l)
    dp = l.perpendicular_segment(p).length
    dr = l.perpendicular_segment(r).length

    assert verify_numerically(dp, dr)

    assert Polygon((1, 0), (2, 0), (2, 2)).reflect(Line((3, 0), slope=oo)) \
        == Triangle(Point(5, 0), Point(4, 0), Point(4, 2))
    assert Polygon((1, 0), (2, 0), (2, 2)).reflect(Line((0, 3), slope=oo)) \
        == Triangle(Point(-1, 0), Point(-2, 0), Point(-2, 2))
    assert Polygon((1, 0), (2, 0), (2, 2)).reflect(Line((0, 3), slope=0)) \
        == Triangle(Point(1, 6), Point(2, 6), Point(2, 4))
    assert Polygon((1, 0), (2, 0), (2, 2)).reflect(Line((3, 0), slope=0)) \
        == Triangle(Point(1, 0), Point(2, 0), Point(2, -2))

def test_bisectors():
    p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
    p = Polygon(Point(0, 0), Point(2, 0), Point(1, 1), Point(0, 3))
    q = Polygon(Point(1, 0), Point(2, 0), Point(3, 3), Point(-1, 5))
    poly = Polygon(Point(3, 4), Point(0, 0), Point(8, 7), Point(-1, 1), Point(19, -19))
    t = Triangle(p1, p2, p3)
    assert t.bisectors()[p2] == Segment(Point(1, 0), Point(0, sqrt(2) - 1))
    assert p.bisectors()[Point2D(0, 3)] == Ray2D(Point2D(0, 3), \
        Point2D(sin(acos(2*sqrt(5)/5)/2), 3 - cos(acos(2*sqrt(5)/5)/2)))
    assert q.bisectors()[Point2D(-1, 5)] == \
        Ray2D(Point2D(-1, 5), Point2D(-1 + sqrt(29)*(5*sin(acos(9*sqrt(145)/145)/2) + \
        2*cos(acos(9*sqrt(145)/145)/2))/29, sqrt(29)*(-5*cos(acos(9*sqrt(145)/145)/2) + \
        2*sin(acos(9*sqrt(145)/145)/2))/29 + 5))
    assert poly.bisectors()[Point2D(-1, 1)] == Ray2D(Point2D(-1, 1), \
        Point2D(-1 + sin(acos(sqrt(26)/26)/2 + pi/4), 1 - sin(-acos(sqrt(26)/26)/2 + pi/4)))

def test_incenter():
    assert Triangle(Point(0, 0), Point(1, 0), Point(0, 1)).incenter \
        == Point(1 - sqrt(2)/2, 1 - sqrt(2)/2)

def test_inradius():
    assert Triangle(Point(0, 0), Point(4, 0), Point(0, 3)).inradius == 1

def test_incircle():
    assert Triangle(Point(0, 0), Point(2, 0), Point(0, 2)).incircle \
        == Circle(Point(2 - sqrt(2), 2 - sqrt(2)), 2 - sqrt(2))

def test_exradii():
    t = Triangle(Point(0, 0), Point(6, 0), Point(0, 2))
    assert t.exradii[t.sides[2]] == (-2 + sqrt(10))

def test_medians():
    t = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
    assert t.medians[Point(0, 0)] == Segment(Point(0, 0), Point(S.Half, S.Half))

def test_medial():
    assert Triangle(Point(0, 0), Point(1, 0), Point(0, 1)).medial \
        == Triangle(Point(S.Half, 0), Point(S.Half, S.Half), Point(0, S.Half))

def test_nine_point_circle():
    assert Triangle(Point(0, 0), Point(1, 0), Point(0, 1)).nine_point_circle \
        == Circle(Point2D(Rational(1, 4), Rational(1, 4)), sqrt(2)/4)

def test_eulerline():
    assert Triangle(Point(0, 0), Point(1, 0), Point(0, 1)).eulerline \
        == Line(Point2D(0, 0), Point2D(S.Half, S.Half))
    assert Triangle(Point(0, 0), Point(10, 0), Point(5, 5*sqrt(3))).eulerline \
        == Point2D(5, 5*sqrt(3)/3)
    assert Triangle(Point(4, -6), Point(4, -1), Point(-3, 3)).eulerline \
        == Line(Point2D(Rational(64, 7), 3), Point2D(Rational(-29, 14), Rational(-7, 2)))

def test_intersection():
    poly1 = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
    poly2 = Polygon(Point(0, 1), Point(-5, 0),
                    Point(0, -4), Point(0, Rational(1, 5)),
                    Point(S.Half, -0.1), Point(1, 0), Point(0, 1))

    assert poly1.intersection(poly2) == [Point2D(Rational(1, 3), 0),
        Segment(Point(0, Rational(1, 5)), Point(0, 0)),
        Segment(Point(1, 0), Point(0, 1))]
    assert poly2.intersection(poly1) == [Point(Rational(1, 3), 0),
        Segment(Point(0, 0), Point(0, Rational(1, 5))),
        Segment(Point(1, 0), Point(0, 1))]
    assert poly1.intersection(Point(0, 0)) == [Point(0, 0)]
    assert poly1.intersection(Point(-12,  -43)) == []
    assert poly2.intersection(Line((-12, 0), (12, 0))) == [Point(-5, 0),
        Point(0, 0), Point(Rational(1, 3), 0), Point(1, 0)]
    assert poly2.intersection(Line((-12, 12), (12, 12))) == []
    assert poly2.intersection(Ray((-3, 4), (1, 0))) == [Segment(Point(1, 0),
        Point(0, 1))]
    assert poly2.intersection(Circle((0, -1), 1)) == [Point(0, -2),
        Point(0, 0)]
    assert poly1.intersection(poly1) == [Segment(Point(0, 0), Point(1, 0)),
        Segment(Point(0, 1), Point(0, 0)), Segment(Point(1, 0), Point(0, 1))]
    assert poly2.intersection(poly2) == [Segment(Point(-5, 0), Point(0, -4)),
        Segment(Point(0, -4), Point(0, Rational(1, 5))),
        Segment(Point(0, Rational(1, 5)), Point(S.Half, Rational(-1, 10))),
        Segment(Point(0, 1), Point(-5, 0)),
        Segment(Point(S.Half, Rational(-1, 10)), Point(1, 0)),
        Segment(Point(1, 0), Point(0, 1))]
    assert poly2.intersection(Triangle(Point(0, 1), Point(1, 0), Point(-1, 1))) \
        == [Point(Rational(-5, 7), Rational(6, 7)), Segment(Point2D(0, 1), Point(1, 0))]
    assert poly1.intersection(RegularPolygon((-12, -15), 3, 3)) == []


def test_parameter_value():
    t = Symbol('t')
    sq = Polygon((0, 0), (0, 1), (1, 1), (1, 0))
    assert sq.parameter_value((0.5, 1), t) == {t: Rational(3, 8)}
    q = Polygon((0, 0), (2, 1), (2, 4), (4, 0))
    assert q.parameter_value((4, 0), t) == {t: -6 + 3*sqrt(5)} # ~= 0.708

    raises(ValueError, lambda: sq.parameter_value((5, 6), t))
    raises(ValueError, lambda: sq.parameter_value(Circle(Point(0, 0), 1), t))


def test_issue_12966():
    poly = Polygon(Point(0, 0), Point(0, 10), Point(5, 10), Point(5, 5),
        Point(10, 5), Point(10, 0))
    t = Symbol('t')
    pt = poly.arbitrary_point(t)
    DELTA = 5/poly.perimeter
    assert [pt.subs(t, DELTA*i) for i in range(int(1/DELTA))] == [
        Point(0, 0), Point(0, 5), Point(0, 10), Point(5, 10),
        Point(5, 5), Point(10, 5), Point(10, 0), Point(5, 0)]


def test_second_moment_of_area():
    x, y = symbols('x, y')
    # triangle
    p1, p2, p3 = [(0, 0), (4, 0), (0, 2)]
    p = (0, 0)
    # equation of hypotenuse
    eq_y = (1-x/4)*2
    I_yy = integrate((x**2) * (integrate(1, (y, 0, eq_y))), (x, 0, 4))
    I_xx = integrate(1 * (integrate(y**2, (y, 0, eq_y))), (x, 0, 4))
    I_xy = integrate(x * (integrate(y, (y, 0, eq_y))), (x, 0, 4))

    triangle = Polygon(p1, p2, p3)

    assert (I_xx - triangle.second_moment_of_area(p)[0]) == 0
    assert (I_yy - triangle.second_moment_of_area(p)[1]) == 0
    assert (I_xy - triangle.second_moment_of_area(p)[2]) == 0

    # rectangle
    p1, p2, p3, p4=[(0, 0), (4, 0), (4, 2), (0, 2)]
    I_yy = integrate((x**2) * integrate(1, (y, 0, 2)), (x, 0, 4))
    I_xx = integrate(1 * integrate(y**2, (y, 0, 2)), (x, 0, 4))
    I_xy = integrate(x * integrate(y, (y, 0, 2)), (x, 0, 4))

    rectangle = Polygon(p1, p2, p3, p4)

    assert (I_xx - rectangle.second_moment_of_area(p)[0]) == 0
    assert (I_yy - rectangle.second_moment_of_area(p)[1]) == 0
    assert (I_xy - rectangle.second_moment_of_area(p)[2]) == 0


    r = RegularPolygon(Point(0, 0), 5, 3)
    assert r.second_moment_of_area() == (1875*sqrt(3)/S(32), 1875*sqrt(3)/S(32), 0)


def test_first_moment():
    a, b  = symbols('a, b', positive=True)
    # rectangle
    p1 = Polygon((0, 0), (a, 0), (a, b), (0, b))
    assert p1.first_moment_of_area() == (a*b**2/8, a**2*b/8)
    assert p1.first_moment_of_area((a/3, b/4)) == (-3*a*b**2/32, -a**2*b/9)

    p1 = Polygon((0, 0), (40, 0), (40, 30), (0, 30))
    assert p1.first_moment_of_area() == (4500, 6000)

    # triangle
    p2 = Polygon((0, 0), (a, 0), (a/2, b))
    assert p2.first_moment_of_area() == (4*a*b**2/81, a**2*b/24)
    assert p2.first_moment_of_area((a/8, b/6)) == (-25*a*b**2/648, -5*a**2*b/768)

    p2 = Polygon((0, 0), (12, 0), (12, 30))
    assert p2.first_moment_of_area() == (S(1600)/3, -S(640)/3)


def test_section_modulus_and_polar_second_moment_of_area():
    a, b = symbols('a, b', positive=True)
    x, y = symbols('x, y')
    rectangle = Polygon((0, b), (0, 0), (a, 0), (a, b))
    assert rectangle.section_modulus(Point(x, y)) == (a*b**3/12/(-b/2 + y), a**3*b/12/(-a/2 + x))
    assert rectangle.polar_second_moment_of_area() == a**3*b/12 + a*b**3/12

    convex = RegularPolygon((0, 0), 1, 6)
    assert convex.section_modulus() == (Rational(5, 8), sqrt(3)*Rational(5, 16))
    assert convex.polar_second_moment_of_area() == 5*sqrt(3)/S(8)

    concave = Polygon((0, 0), (1, 8), (3, 4), (4, 6), (7, 1))
    assert concave.section_modulus() == (Rational(-6371, 429), Rational(-9778, 519))
    assert concave.polar_second_moment_of_area() == Rational(-38669, 252)


def test_cut_section():
    # concave polygon
    p = Polygon((-1, -1), (1, Rational(5, 2)), (2, 1), (3, Rational(5, 2)), (4, 2), (5, 3), (-1, 3))
    l = Line((0, 0), (Rational(9, 2), 3))
    p1 = p.cut_section(l)[0]
    p2 = p.cut_section(l)[1]
    assert p1 == Polygon(
        Point2D(Rational(-9, 13), Rational(-6, 13)), Point2D(1, Rational(5, 2)), Point2D(Rational(24, 13), Rational(16, 13)),
        Point2D(Rational(12, 5), Rational(8, 5)), Point2D(3, Rational(5, 2)), Point2D(Rational(24, 7), Rational(16, 7)),
        Point2D(Rational(9, 2), 3), Point2D(-1, 3), Point2D(-1, Rational(-2, 3)))
    assert p2 == Polygon(Point2D(-1, -1), Point2D(Rational(-9, 13), Rational(-6, 13)), Point2D(Rational(24, 13), Rational(16, 13)),
        Point2D(2, 1), Point2D(Rational(12, 5), Rational(8, 5)), Point2D(Rational(24, 7), Rational(16, 7)), Point2D(4, 2), Point2D(5, 3),
        Point2D(Rational(9, 2), 3), Point2D(-1, Rational(-2, 3)))

    # convex polygon
    p = RegularPolygon(Point2D(0, 0), 6, 6)
    s = p.cut_section(Line((0, 0), slope=1))
    assert s[0] == Polygon(Point2D(-3*sqrt(3) + 9, -3*sqrt(3) + 9), Point2D(3, 3*sqrt(3)),
        Point2D(-3, 3*sqrt(3)), Point2D(-6, 0), Point2D(-9 + 3*sqrt(3), -9 + 3*sqrt(3)))
    assert s[1] == Polygon(Point2D(6, 0), Point2D(-3*sqrt(3) + 9, -3*sqrt(3) + 9),
        Point2D(-9 + 3*sqrt(3), -9 + 3*sqrt(3)), Point2D(-3, -3*sqrt(3)), Point2D(3, -3*sqrt(3)))

    # case where line does not intersects but coincides with the edge of polygon
    a, b = 20, 10
    t1, t2, t3, t4 = [(0, b), (0, 0), (a, 0), (a, b)]
    p = Polygon(t1, t2, t3, t4)
    p1, p2 = p.cut_section(Line((0, b), slope=0))
    assert p1 == None
    assert p2 == Polygon(Point2D(0, 10), Point2D(0, 0), Point2D(20, 0), Point2D(20, 10))

    p3, p4 = p.cut_section(Line((0, 0), slope=0))
    assert p3 == Polygon(Point2D(0, 10), Point2D(0, 0), Point2D(20, 0), Point2D(20, 10))
    assert p4 == None

    # case where the line does not intersect with a polygon at all
    raises(ValueError, lambda: p.cut_section(Line((0, a), slope=0)))

def test_type_of_triangle():
    # Isoceles triangle
    p1 = Polygon(Point(0, 0), Point(5, 0), Point(2, 4))
    assert p1.is_isosceles() == True
    assert p1.is_scalene() == False
    assert p1.is_equilateral() == False

    # Scalene triangle
    p2 = Polygon (Point(0, 0), Point(0, 2), Point(4, 0))
    assert p2.is_isosceles() == False
    assert p2.is_scalene() == True
    assert p2.is_equilateral() == False

    # Equilateral triangle
    p3 = Polygon(Point(0, 0), Point(6, 0), Point(3, sqrt(27)))
    assert p3.is_isosceles() == True
    assert p3.is_scalene() == False
    assert p3.is_equilateral() == True

def test_do_poly_distance():
    # Non-intersecting polygons
    square1 = Polygon (Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    triangle1 = Polygon(Point(1, 2), Point(2, 2), Point(2, 1))
    assert square1._do_poly_distance(triangle1) == sqrt(2)/2

    # Polygons which sides intersect
    square2 = Polygon(Point(1, 0), Point(2, 0), Point(2, 1), Point(1, 1))
    with warns(UserWarning, \
               match="Polygons may intersect producing erroneous output", test_stacklevel=False):
        assert square1._do_poly_distance(square2) == 0

    # Polygons which bodies intersect
    triangle2 = Polygon(Point(0, -1), Point(2, -1), Point(S.Half, S.Half))
    with warns(UserWarning, \
               match="Polygons may intersect producing erroneous output", test_stacklevel=False):
        assert triangle2._do_poly_distance(square1) == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\interval\test_indexing.py ===
import re

import numpy as np
import pytest

from pandas.errors import InvalidIndexError

from pandas import (
    NA,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Timedelta,
    Timestamp,
    array,
    date_range,
    interval_range,
    isna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class TestGetItem:
    def test_getitem(self, closed):
        idx = IntervalIndex.from_arrays((0, 1, np.nan), (1, 2, np.nan), closed=closed)
        assert idx[0] == Interval(0.0, 1.0, closed=closed)
        assert idx[1] == Interval(1.0, 2.0, closed=closed)
        assert isna(idx[2])

        result = idx[0:1]
        expected = IntervalIndex.from_arrays((0.0,), (1.0,), closed=closed)
        tm.assert_index_equal(result, expected)

        result = idx[0:2]
        expected = IntervalIndex.from_arrays((0.0, 1), (1.0, 2.0), closed=closed)
        tm.assert_index_equal(result, expected)

        result = idx[1:3]
        expected = IntervalIndex.from_arrays(
            (1.0, np.nan), (2.0, np.nan), closed=closed
        )
        tm.assert_index_equal(result, expected)

    def test_getitem_2d_deprecated(self):
        # GH#30588 multi-dim indexing is deprecated, but raising is also acceptable
        idx = IntervalIndex.from_breaks(range(11), closed="right")
        with pytest.raises(ValueError, match="multi-dimensional indexing not allowed"):
            idx[:, None]
        with pytest.raises(ValueError, match="multi-dimensional indexing not allowed"):
            # GH#44051
            idx[True]
        with pytest.raises(ValueError, match="multi-dimensional indexing not allowed"):
            # GH#44051
            idx[False]


class TestWhere:
    def test_where(self, listlike_box):
        klass = listlike_box

        idx = IntervalIndex.from_breaks(range(11), closed="right")
        cond = [True] * len(idx)
        expected = idx
        result = expected.where(klass(cond))
        tm.assert_index_equal(result, expected)

        cond = [False] + [True] * len(idx[1:])
        expected = IntervalIndex([np.nan] + idx[1:].tolist())
        result = idx.where(klass(cond))
        tm.assert_index_equal(result, expected)


class TestTake:
    def test_take(self, closed):
        index = IntervalIndex.from_breaks(range(11), closed=closed)

        result = index.take(range(10))
        tm.assert_index_equal(result, index)

        result = index.take([0, 0, 1])
        expected = IntervalIndex.from_arrays([0, 0, 1], [1, 1, 2], closed=closed)
        tm.assert_index_equal(result, expected)


class TestGetLoc:
    @pytest.mark.parametrize("side", ["right", "left", "both", "neither"])
    def test_get_loc_interval(self, closed, side):
        idx = IntervalIndex.from_tuples([(0, 1), (2, 3)], closed=closed)

        for bound in [[0, 1], [1, 2], [2, 3], [3, 4], [0, 2], [2.5, 3], [-1, 4]]:
            # if get_loc is supplied an interval, it should only search
            # for exact matches, not overlaps or covers, else KeyError.
            msg = re.escape(f"Interval({bound[0]}, {bound[1]}, closed='{side}')")
            if closed == side:
                if bound == [0, 1]:
                    assert idx.get_loc(Interval(0, 1, closed=side)) == 0
                elif bound == [2, 3]:
                    assert idx.get_loc(Interval(2, 3, closed=side)) == 1
                else:
                    with pytest.raises(KeyError, match=msg):
                        idx.get_loc(Interval(*bound, closed=side))
            else:
                with pytest.raises(KeyError, match=msg):
                    idx.get_loc(Interval(*bound, closed=side))

    @pytest.mark.parametrize("scalar", [-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    def test_get_loc_scalar(self, closed, scalar):
        # correct = {side: {query: answer}}.
        # If query is not in the dict, that query should raise a KeyError
        correct = {
            "right": {0.5: 0, 1: 0, 2.5: 1, 3: 1},
            "left": {0: 0, 0.5: 0, 2: 1, 2.5: 1},
            "both": {0: 0, 0.5: 0, 1: 0, 2: 1, 2.5: 1, 3: 1},
            "neither": {0.5: 0, 2.5: 1},
        }

        idx = IntervalIndex.from_tuples([(0, 1), (2, 3)], closed=closed)

        # if get_loc is supplied a scalar, it should return the index of
        # the interval which contains the scalar, or KeyError.
        if scalar in correct[closed].keys():
            assert idx.get_loc(scalar) == correct[closed][scalar]
        else:
            with pytest.raises(KeyError, match=str(scalar)):
                idx.get_loc(scalar)

    @pytest.mark.parametrize("scalar", [-1, 0, 0.5, 3, 4.5, 5, 6])
    def test_get_loc_length_one_scalar(self, scalar, closed):
        # GH 20921
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        if scalar in index[0]:
            result = index.get_loc(scalar)
            assert result == 0
        else:
            with pytest.raises(KeyError, match=str(scalar)):
                index.get_loc(scalar)

    @pytest.mark.parametrize("other_closed", ["left", "right", "both", "neither"])
    @pytest.mark.parametrize("left, right", [(0, 5), (-1, 4), (-1, 6), (6, 7)])
    def test_get_loc_length_one_interval(self, left, right, closed, other_closed):
        # GH 20921
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        interval = Interval(left, right, closed=other_closed)
        if interval == index[0]:
            result = index.get_loc(interval)
            assert result == 0
        else:
            with pytest.raises(
                KeyError,
                match=re.escape(f"Interval({left}, {right}, closed='{other_closed}')"),
            ):
                index.get_loc(interval)

    # Make consistent with test_interval_new.py (see #16316, #16386)
    @pytest.mark.parametrize(
        "breaks",
        [
            date_range("20180101", periods=4),
            date_range("20180101", periods=4, tz="US/Eastern"),
            timedelta_range("0 days", periods=4),
        ],
        ids=lambda x: str(x.dtype),
    )
    def test_get_loc_datetimelike_nonoverlapping(self, breaks):
        # GH 20636
        # nonoverlapping = IntervalIndex method and no i8 conversion
        index = IntervalIndex.from_breaks(breaks)

        value = index[0].mid
        result = index.get_loc(value)
        expected = 0
        assert result == expected

        interval = Interval(index[0].left, index[0].right)
        result = index.get_loc(interval)
        expected = 0
        assert result == expected

    @pytest.mark.parametrize(
        "arrays",
        [
            (date_range("20180101", periods=4), date_range("20180103", periods=4)),
            (
                date_range("20180101", periods=4, tz="US/Eastern"),
                date_range("20180103", periods=4, tz="US/Eastern"),
            ),
            (
                timedelta_range("0 days", periods=4),
                timedelta_range("2 days", periods=4),
            ),
        ],
        ids=lambda x: str(x[0].dtype),
    )
    def test_get_loc_datetimelike_overlapping(self, arrays):
        # GH 20636
        index = IntervalIndex.from_arrays(*arrays)

        value = index[0].mid + Timedelta("12 hours")
        result = index.get_loc(value)
        expected = slice(0, 2, None)
        assert result == expected

        interval = Interval(index[0].left, index[0].right)
        result = index.get_loc(interval)
        expected = 0
        assert result == expected

    @pytest.mark.parametrize(
        "values",
        [
            date_range("2018-01-04", periods=4, freq="-1D"),
            date_range("2018-01-04", periods=4, freq="-1D", tz="US/Eastern"),
            timedelta_range("3 days", periods=4, freq="-1D"),
            np.arange(3.0, -1.0, -1.0),
            np.arange(3, -1, -1),
        ],
        ids=lambda x: str(x.dtype),
    )
    def test_get_loc_decreasing(self, values):
        # GH 25860
        index = IntervalIndex.from_arrays(values[1:], values[:-1])
        result = index.get_loc(index[0])
        expected = 0
        assert result == expected

    @pytest.mark.parametrize("key", [[5], (2, 3)])
    def test_get_loc_non_scalar_errors(self, key):
        # GH 31117
        idx = IntervalIndex.from_tuples([(1, 3), (2, 4), (3, 5), (7, 10), (3, 10)])

        msg = str(key)
        with pytest.raises(InvalidIndexError, match=msg):
            idx.get_loc(key)

    def test_get_indexer_with_nans(self):
        # GH#41831
        index = IntervalIndex([np.nan, Interval(1, 2), np.nan])

        expected = np.array([True, False, True])
        for key in [None, np.nan, NA]:
            assert key in index
            result = index.get_loc(key)
            tm.assert_numpy_array_equal(result, expected)

        for key in [NaT, np.timedelta64("NaT", "ns"), np.datetime64("NaT", "ns")]:
            with pytest.raises(KeyError, match=str(key)):
                index.get_loc(key)


class TestGetIndexer:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ([Interval(2, 4, closed="right")], [1]),
            ([Interval(2, 4, closed="left")], [-1]),
            ([Interval(2, 4, closed="both")], [-1]),
            ([Interval(2, 4, closed="neither")], [-1]),
            ([Interval(1, 4, closed="right")], [-1]),
            ([Interval(0, 4, closed="right")], [-1]),
            ([Interval(0.5, 1.5, closed="right")], [-1]),
            ([Interval(2, 4, closed="right"), Interval(0, 1, closed="right")], [1, -1]),
            ([Interval(2, 4, closed="right"), Interval(2, 4, closed="right")], [1, 1]),
            ([Interval(5, 7, closed="right"), Interval(2, 4, closed="right")], [2, 1]),
            ([Interval(2, 4, closed="right"), Interval(2, 4, closed="left")], [1, -1]),
        ],
    )
    def test_get_indexer_with_interval(self, query, expected):
        tuples = [(0, 2), (2, 4), (5, 7)]
        index = IntervalIndex.from_tuples(tuples, closed="right")

        result = index.get_indexer(query)
        expected = np.array(expected, dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "query, expected",
        [
            ([-0.5], [-1]),
            ([0], [-1]),
            ([0.5], [0]),
            ([1], [0]),
            ([1.5], [1]),
            ([2], [1]),
            ([2.5], [-1]),
            ([3], [-1]),
            ([3.5], [2]),
            ([4], [2]),
            ([4.5], [-1]),
            ([1, 2], [0, 1]),
            ([1, 2, 3], [0, 1, -1]),
            ([1, 2, 3, 4], [0, 1, -1, 2]),
            ([1, 2, 3, 4, 2], [0, 1, -1, 2, 1]),
        ],
    )
    def test_get_indexer_with_int_and_float(self, query, expected):
        tuples = [(0, 1), (1, 2), (3, 4)]
        index = IntervalIndex.from_tuples(tuples, closed="right")

        result = index.get_indexer(query)
        expected = np.array(expected, dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("item", [[3], np.arange(0.5, 5, 0.5)])
    def test_get_indexer_length_one(self, item, closed):
        # GH 17284
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        result = index.get_indexer(item)
        expected = np.array([0] * len(item), dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("size", [1, 5])
    def test_get_indexer_length_one_interval(self, size, closed):
        # GH 17284
        index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
        result = index.get_indexer([Interval(0, 5, closed)] * size)
        expected = np.array([0] * size, dtype="intp")
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "target",
        [
            IntervalIndex.from_tuples([(7, 8), (1, 2), (3, 4), (0, 1)]),
            IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4), np.nan]),
            IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)], closed="both"),
            [-1, 0, 0.5, 1, 2, 2.5, np.nan],
            ["foo", "foo", "bar", "baz"],
        ],
    )
    def test_get_indexer_categorical(self, target, ordered):
        # GH 30063: categorical and non-categorical results should be consistent
        index = IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)])
        categorical_target = CategoricalIndex(target, ordered=ordered)

        result = index.get_indexer(categorical_target)
        expected = index.get_indexer(target)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_get_indexer_categorical_with_nans(self):
        # GH#41934 nans in both index and in target
        ii = IntervalIndex.from_breaks(range(5))
        ii2 = ii.append(IntervalIndex([np.nan]))
        ci2 = CategoricalIndex(ii2)

        result = ii2.get_indexer(ci2)
        expected = np.arange(5, dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        # not-all-matches
        result = ii2[1:].get_indexer(ci2[::-1])
        expected = np.array([3, 2, 1, 0, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        # non-unique target, non-unique nans
        result = ii2.get_indexer(ci2.append(ci2))
        expected = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_datetime(self):
        ii = IntervalIndex.from_breaks(date_range("2018-01-01", periods=4))
        # TODO: with mismatched resolution get_indexer currently raises;
        #  this should probably coerce?
        target = DatetimeIndex(["2018-01-02"], dtype="M8[ns]")
        result = ii.get_indexer(target)
        expected = np.array([0], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        result = ii.get_indexer(target.astype(str))
        tm.assert_numpy_array_equal(result, expected)

        # https://github.com/pandas-dev/pandas/issues/47772
        result = ii.get_indexer(target.asi8)
        expected = np.array([-1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "tuples, closed",
        [
            ([(0, 2), (1, 3), (3, 4)], "neither"),
            ([(0, 5), (1, 4), (6, 7)], "left"),
            ([(0, 1), (0, 1), (1, 2)], "right"),
            ([(0, 1), (2, 3), (3, 4)], "both"),
        ],
    )
    def test_get_indexer_errors(self, tuples, closed):
        # IntervalIndex needs non-overlapping for uniqueness when querying
        index = IntervalIndex.from_tuples(tuples, closed=closed)

        msg = (
            "cannot handle overlapping indices; use "
            "IntervalIndex.get_indexer_non_unique"
        )
        with pytest.raises(InvalidIndexError, match=msg):
            index.get_indexer([0, 2])

    @pytest.mark.parametrize(
        "query, expected",
        [
            ([-0.5], ([-1], [0])),
            ([0], ([0], [])),
            ([0.5], ([0], [])),
            ([1], ([0, 1], [])),
            ([1.5], ([0, 1], [])),
            ([2], ([0, 1, 2], [])),
            ([2.5], ([1, 2], [])),
            ([3], ([2], [])),
            ([3.5], ([2], [])),
            ([4], ([-1], [0])),
            ([4.5], ([-1], [0])),
            ([1, 2], ([0, 1, 0, 1, 2], [])),
            ([1, 2, 3], ([0, 1, 0, 1, 2, 2], [])),
            ([1, 2, 3, 4], ([0, 1, 0, 1, 2, 2, -1], [3])),
            ([1, 2, 3, 4, 2], ([0, 1, 0, 1, 2, 2, -1, 0, 1, 2], [3])),
        ],
    )
    def test_get_indexer_non_unique_with_int_and_float(self, query, expected):
        tuples = [(0, 2.5), (1, 3), (2, 4)]
        index = IntervalIndex.from_tuples(tuples, closed="left")

        result_indexer, result_missing = index.get_indexer_non_unique(query)
        expected_indexer = np.array(expected[0], dtype="intp")
        expected_missing = np.array(expected[1], dtype="intp")

        tm.assert_numpy_array_equal(result_indexer, expected_indexer)
        tm.assert_numpy_array_equal(result_missing, expected_missing)

        # TODO we may also want to test get_indexer for the case when
        # the intervals are duplicated, decreasing, non-monotonic, etc..

    def test_get_indexer_non_monotonic(self):
        # GH 16410
        idx1 = IntervalIndex.from_tuples([(2, 3), (4, 5), (0, 1)])
        idx2 = IntervalIndex.from_tuples([(0, 1), (2, 3), (6, 7), (8, 9)])
        result = idx1.get_indexer(idx2)
        expected = np.array([2, 0, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        result = idx1.get_indexer(idx1[1:])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_with_nans(self):
        # GH#41831
        index = IntervalIndex([np.nan, np.nan])
        other = IntervalIndex([np.nan])

        assert not index._index_as_unique

        result = index.get_indexer_for(other)
        expected = np.array([0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_index_non_unique_non_monotonic(self):
        # GH#44084 (root cause)
        index = IntervalIndex.from_tuples(
            [(0.0, 1.0), (1.0, 2.0), (0.0, 1.0), (1.0, 2.0)]
        )

        result, _ = index.get_indexer_non_unique([Interval(1.0, 2.0)])
        expected = np.array([1, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_get_indexer_multiindex_with_intervals(self):
        # GH#44084 (MultiIndex case as reported)
        interval_index = IntervalIndex.from_tuples(
            [(2.0, 3.0), (0.0, 1.0), (1.0, 2.0)], name="interval"
        )
        foo_index = Index([1, 2, 3], name="foo")

        multi_index = MultiIndex.from_product([foo_index, interval_index])

        result = multi_index.get_level_values("interval").get_indexer_for(
            [Interval(0.0, 1.0)]
        )
        expected = np.array([1, 4, 7], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("box", [IntervalIndex, array, list])
    def test_get_indexer_interval_index(self, box):
        # GH#30178
        rng = period_range("2022-07-01", freq="D", periods=3)
        idx = box(interval_range(Timestamp("2022-07-01"), freq="3D", periods=3))

        actual = rng.get_indexer(idx)
        expected = np.array([-1, -1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(actual, expected)

    def test_get_indexer_read_only(self):
        idx = interval_range(start=0, end=5)
        arr = np.array([1, 2])
        arr.flags.writeable = False
        result = idx.get_indexer(arr)
        expected = np.array([0, 1])
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

        result = idx.get_indexer_non_unique(arr)[0]
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)


class TestSliceLocs:
    def test_slice_locs_with_interval(self):
        # increasing monotonically
        index = IntervalIndex.from_tuples([(0, 2), (1, 3), (2, 4)])

        assert index.slice_locs(start=Interval(0, 2), end=Interval(2, 4)) == (0, 3)
        assert index.slice_locs(start=Interval(0, 2)) == (0, 3)
        assert index.slice_locs(end=Interval(2, 4)) == (0, 3)
        assert index.slice_locs(end=Interval(0, 2)) == (0, 1)
        assert index.slice_locs(start=Interval(2, 4), end=Interval(0, 2)) == (2, 1)

        # decreasing monotonically
        index = IntervalIndex.from_tuples([(2, 4), (1, 3), (0, 2)])

        assert index.slice_locs(start=Interval(0, 2), end=Interval(2, 4)) == (2, 1)
        assert index.slice_locs(start=Interval(0, 2)) == (2, 3)
        assert index.slice_locs(end=Interval(2, 4)) == (0, 1)
        assert index.slice_locs(end=Interval(0, 2)) == (0, 3)
        assert index.slice_locs(start=Interval(2, 4), end=Interval(0, 2)) == (0, 3)

        # sorted duplicates
        index = IntervalIndex.from_tuples([(0, 2), (0, 2), (2, 4)])

        assert index.slice_locs(start=Interval(0, 2), end=Interval(2, 4)) == (0, 3)
        assert index.slice_locs(start=Interval(0, 2)) == (0, 3)
        assert index.slice_locs(end=Interval(2, 4)) == (0, 3)
        assert index.slice_locs(end=Interval(0, 2)) == (0, 2)
        assert index.slice_locs(start=Interval(2, 4), end=Interval(0, 2)) == (2, 2)

        # unsorted duplicates
        index = IntervalIndex.from_tuples([(0, 2), (2, 4), (0, 2)])

        with pytest.raises(
            KeyError,
            match=re.escape(
                '"Cannot get left slice bound for non-unique label: '
                "Interval(0, 2, closed='right')\""
            ),
        ):
            index.slice_locs(start=Interval(0, 2), end=Interval(2, 4))

        with pytest.raises(
            KeyError,
            match=re.escape(
                '"Cannot get left slice bound for non-unique label: '
                "Interval(0, 2, closed='right')\""
            ),
        ):
            index.slice_locs(start=Interval(0, 2))

        assert index.slice_locs(end=Interval(2, 4)) == (0, 2)

        with pytest.raises(
            KeyError,
            match=re.escape(
                '"Cannot get right slice bound for non-unique label: '
                "Interval(0, 2, closed='right')\""
            ),
        ):
            index.slice_locs(end=Interval(0, 2))

        with pytest.raises(
            KeyError,
            match=re.escape(
                '"Cannot get right slice bound for non-unique label: '
                "Interval(0, 2, closed='right')\""
            ),
        ):
            index.slice_locs(start=Interval(2, 4), end=Interval(0, 2))

        # another unsorted duplicates
        index = IntervalIndex.from_tuples([(0, 2), (0, 2), (2, 4), (1, 3)])

        assert index.slice_locs(start=Interval(0, 2), end=Interval(2, 4)) == (0, 3)
        assert index.slice_locs(start=Interval(0, 2)) == (0, 4)
        assert index.slice_locs(end=Interval(2, 4)) == (0, 3)
        assert index.slice_locs(end=Interval(0, 2)) == (0, 2)
        assert index.slice_locs(start=Interval(2, 4), end=Interval(0, 2)) == (2, 2)

    def test_slice_locs_with_ints_and_floats_succeeds(self):
        # increasing non-overlapping
        index = IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)])

        assert index.slice_locs(0, 1) == (0, 1)
        assert index.slice_locs(0, 2) == (0, 2)
        assert index.slice_locs(0, 3) == (0, 2)
        assert index.slice_locs(3, 1) == (2, 1)
        assert index.slice_locs(3, 4) == (2, 3)
        assert index.slice_locs(0, 4) == (0, 3)

        # decreasing non-overlapping
        index = IntervalIndex.from_tuples([(3, 4), (1, 2), (0, 1)])
        assert index.slice_locs(0, 1) == (3, 3)
        assert index.slice_locs(0, 2) == (3, 2)
        assert index.slice_locs(0, 3) == (3, 1)
        assert index.slice_locs(3, 1) == (1, 3)
        assert index.slice_locs(3, 4) == (1, 1)
        assert index.slice_locs(0, 4) == (3, 1)

    @pytest.mark.parametrize("query", [[0, 1], [0, 2], [0, 3], [0, 4]])
    @pytest.mark.parametrize(
        "tuples",
        [
            [(0, 2), (1, 3), (2, 4)],
            [(2, 4), (1, 3), (0, 2)],
            [(0, 2), (0, 2), (2, 4)],
            [(0, 2), (2, 4), (0, 2)],
            [(0, 2), (0, 2), (2, 4), (1, 3)],
        ],
    )
    def test_slice_locs_with_ints_and_floats_errors(self, tuples, query):
        start, stop = query
        index = IntervalIndex.from_tuples(tuples)
        with pytest.raises(
            KeyError,
            match=(
                "'can only get slices from an IntervalIndex if bounds are "
                "non-overlapping and all monotonic increasing or decreasing'"
            ),
        ):
            index.slice_locs(start, stop)


class TestPutmask:
    @pytest.mark.parametrize("tz", ["US/Pacific", None])
    def test_putmask_dt64(self, tz):
        # GH#37968
        dti = date_range("2016-01-01", periods=9, tz=tz)
        idx = IntervalIndex.from_breaks(dti)
        mask = np.zeros(idx.shape, dtype=bool)
        mask[0:3] = True

        result = idx.putmask(mask, idx[-1])
        expected = IntervalIndex([idx[-1]] * 3 + list(idx[3:]))
        tm.assert_index_equal(result, expected)

    def test_putmask_td64(self):
        # GH#37968
        dti = date_range("2016-01-01", periods=9)
        tdi = dti - dti[0]
        idx = IntervalIndex.from_breaks(tdi)
        mask = np.zeros(idx.shape, dtype=bool)
        mask[0:3] = True

        result = idx.putmask(mask, idx[-1])
        expected = IntervalIndex([idx[-1]] * 3 + list(idx[3:]))
        tm.assert_index_equal(result, expected)


class TestContains:
    # .__contains__, not .contains

    def test_contains_dunder(self):
        index = IntervalIndex.from_arrays([0, 1], [1, 2], closed="right")

        # __contains__ requires perfect matches to intervals.
        assert 0 not in index
        assert 1 not in index
        assert 2 not in index

        assert Interval(0, 1, closed="right") in index
        assert Interval(0, 2, closed="right") not in index
        assert Interval(0, 0.5, closed="right") not in index
        assert Interval(3, 5, closed="right") not in index
        assert Interval(-1, 0, closed="left") not in index
        assert Interval(0, 1, closed="left") not in index
        assert Interval(0, 1, closed="both") not in index

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\constants.py ===
#: list of lorem ipsum words used by the lipsum() helper function
LOREM_IPSUM_WORDS = """\
a ac accumsan ad adipiscing aenean aliquam aliquet amet ante aptent arcu at
auctor augue bibendum blandit class commodo condimentum congue consectetuer
consequat conubia convallis cras cubilia cum curabitur curae cursus dapibus
diam dictum dictumst dignissim dis dolor donec dui duis egestas eget eleifend
elementum elit enim erat eros est et etiam eu euismod facilisi facilisis fames
faucibus felis fermentum feugiat fringilla fusce gravida habitant habitasse hac
hendrerit hymenaeos iaculis id imperdiet in inceptos integer interdum ipsum
justo lacinia lacus laoreet lectus leo libero ligula litora lobortis lorem
luctus maecenas magna magnis malesuada massa mattis mauris metus mi molestie
mollis montes morbi mus nam nascetur natoque nec neque netus nibh nisi nisl non
nonummy nostra nulla nullam nunc odio orci ornare parturient pede pellentesque
penatibus per pharetra phasellus placerat platea porta porttitor posuere
potenti praesent pretium primis proin pulvinar purus quam quis quisque rhoncus
ridiculus risus rutrum sagittis sapien scelerisque sed sem semper senectus sit
sociis sociosqu sodales sollicitudin suscipit suspendisse taciti tellus tempor
tempus tincidunt torquent tortor tristique turpis ullamcorper ultrices
ultricies urna ut varius vehicula vel velit venenatis vestibulum vitae vivamus
viverra volutpat vulputate"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_typing\_scalars.py ===
from typing import Any, TypeAlias

import numpy as np

# NOTE: `_StrLike_co` and `_BytesLike_co` are pointless, as `np.str_` and
# `np.bytes_` are already subclasses of their builtin counterpart
_CharLike_co: TypeAlias = str | bytes

# The `<X>Like_co` type-aliases below represent all scalars that can be
# coerced into `<X>` (with the casting rule `same_kind`)
_BoolLike_co: TypeAlias = bool | np.bool
_UIntLike_co: TypeAlias = bool | np.unsignedinteger | np.bool
_IntLike_co: TypeAlias = int | np.integer | np.bool
_FloatLike_co: TypeAlias = float | np.floating | np.integer | np.bool
_ComplexLike_co: TypeAlias = complex | np.number | np.bool
_NumberLike_co: TypeAlias = _ComplexLike_co
_TD64Like_co: TypeAlias = int | np.timedelta64 | np.integer | np.bool
# `_VoidLike_co` is technically not a scalar, but it's close enough
_VoidLike_co: TypeAlias = tuple[Any, ...] | np.void
_ScalarLike_co: TypeAlias = complex | str | bytes | np.generic

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\import_utils.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib.metadata
import importlib.util


def is_installed(package):
    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            return False

        return spec is not None

    return dist is not None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\outcome\__init__.py ===
"""Top-level package for outcome."""

from ._impl import (
    Error as Error,
    Maybe as Maybe,
    Outcome as Outcome,
    Value as Value,
    acapture as acapture,
    capture as capture,
)
from ._util import AlreadyUsedError as AlreadyUsedError, fixup_module_metadata
from ._version import __version__ as __version__

__all__ = (
    'Error', 'Outcome', 'Value', 'Maybe', 'acapture', 'capture',
    'AlreadyUsedError'
)

fixup_module_metadata(__name__, globals())
del fixup_module_metadata

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\resolution\base.py ===
from typing import Callable, List, Optional

from pip._internal.req.req_install import InstallRequirement
from pip._internal.req.req_set import RequirementSet

InstallRequirementProvider = Callable[
    [str, Optional[InstallRequirement]], InstallRequirement
]


class BaseResolver:
    def resolve(
        self, root_reqs: List[InstallRequirement], check_supported_wheels: bool
    ) -> RequirementSet:
        raise NotImplementedError()

    def get_installation_order(
        self, req_set: RequirementSet
    ) -> List[InstallRequirement]:
        raise NotImplementedError()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\logger\logger.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

import logging
import os

from .null_handler import NULLHandler

_default_log_level = os.environ.get("PYREADLINE_LOG", "DEBUG")

LOGGER = logging.getLogger("PYREADLINE")
LOGGER.setLevel(_default_log_level)
LOGGER.propagate = False
LOGGER.addHandler(NULLHandler())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\util.py ===

import typing


T_JSON_DICT = typing.Dict[str, typing.Any]
_event_parsers = dict()


def event_class(method):
    ''' A decorator that registers a class as an event class. '''
    def decorate(cls):
        _event_parsers[method] = cls
        cls.event_class = method
        return cls
    return decorate


def parse_json_event(json: T_JSON_DICT) -> typing.Any:
    ''' Parse a JSON dictionary into a CDP event. '''
    return _event_parsers[json['method']].from_json(json['params'])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\util.py ===

import typing


T_JSON_DICT = typing.Dict[str, typing.Any]
_event_parsers = dict()


def event_class(method):
    ''' A decorator that registers a class as an event class. '''
    def decorate(cls):
        _event_parsers[method] = cls
        cls.event_class = method
        return cls
    return decorate


def parse_json_event(json: T_JSON_DICT) -> typing.Any:
    ''' Parse a JSON dictionary into a CDP event. '''
    return _event_parsers[json['method']].from_json(json['params'])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\util.py ===

import typing


T_JSON_DICT = typing.Dict[str, typing.Any]
_event_parsers = dict()


def event_class(method):
    ''' A decorator that registers a class as an event class. '''
    def decorate(cls):
        _event_parsers[method] = cls
        cls.event_class = method
        return cls
    return decorate


def parse_json_event(json: T_JSON_DICT) -> typing.Any:
    ''' Parse a JSON dictionary into a CDP event. '''
    return _event_parsers[json['method']].from_json(json['params'])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\_orm_types.py ===
# sql/_orm_types.py
# Copyright (C) 2022-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""ORM types that need to present specifically for **documentation only** of
the Executable.execution_options() method, which includes options that
are meaningful to the ORM.

"""


from __future__ import annotations

from ..util.typing import Literal

SynchronizeSessionArgument = Literal[False, "auto", "evaluate", "fetch"]
DMLStrategyArgument = Literal["bulk", "raw", "orm", "auto"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\discrete\__init__.py ===
"""This module contains functions which operate on discrete sequences.

Transforms - ``fft``, ``ifft``, ``ntt``, ``intt``, ``fwht``, ``ifwht``,
            ``mobius_transform``, ``inverse_mobius_transform``

Convolutions - ``convolution``, ``convolution_fft``, ``convolution_ntt``,
            ``convolution_fwht``, ``convolution_subset``,
            ``covering_product``, ``intersecting_product``
"""

from .transforms import (fft, ifft, ntt, intt, fwht, ifwht,
    mobius_transform, inverse_mobius_transform)
from .convolutions import convolution, covering_product, intersecting_product

__all__ = [
    'fft', 'ifft', 'ntt', 'intt', 'fwht', 'ifwht', 'mobius_transform',
    'inverse_mobius_transform',

    'convolution', 'covering_product', 'intersecting_product',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\__init__.py ===
"""
Unified place for determining if external dependencies are installed or not.

You should import all external modules using the import_module() function.

For example

>>> from sympy.external import import_module
>>> numpy = import_module('numpy')

If the resulting library is not installed, or if the installed version
is less than a given minimum version, the function will return None.
Otherwise, it will return the library. See the docstring of
import_module() for more information.

"""

from sympy.external.importtools import import_module

__all__ = ['import_module']

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\test-examples\ruletest4.py ===
import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
q1, q2, q3 = _me.dynamicsymbols('q1 q2 q3')
frame_b.orient(frame_a, 'Axis', [q3, frame_a.x])
dcm = frame_a.dcm(frame_b)
m = dcm*3-frame_a.dcm(frame_b)
r = _me.dynamicsymbols('r')
circle_area = _sm.pi*r**2
u, a = _me.dynamicsymbols('u a')
x, y = _me.dynamicsymbols('x y')
s = u*_me.dynamicsymbols._t-1/2*a*_me.dynamicsymbols._t**2
expr1 = 2*a*0.5-1.25+0.25
expr2 = -1*x**2+y**2+0.25*(x+y)**2
expr3 = 0.5*10**(-10)
dyadic = _me.outer(frame_a.x, frame_a.x)+_me.outer(frame_a.y, frame_a.y)+_me.outer(frame_a.z, frame_a.z)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\tests\test_elliptic.py ===
"""
Limited tests of the elliptic functions module.  A full suite of
extensive testing can be found in elliptic_torture_tests.py

Author of the first version: M.T. Taschuk

References:

[1] Abramowitz & Stegun. 'Handbook of Mathematical Functions, 9th Ed.',
    (Dover duplicate of 1972 edition)
[2] Whittaker 'A Course of Modern Analysis, 4th Ed.', 1946,
    Cambridge University Press

"""

import mpmath
import random
import pytest

from mpmath import *

def mpc_ae(a, b, eps=eps):
    res = True
    res = res and a.real.ae(b.real, eps)
    res = res and a.imag.ae(b.imag, eps)
    return res

zero = mpf(0)
one = mpf(1)

jsn = ellipfun('sn')
jcn = ellipfun('cn')
jdn = ellipfun('dn')

calculate_nome = lambda k: qfrom(k=k)

def test_ellipfun():
    mp.dps = 15
    assert ellipfun('ss', 0, 0) == 1
    assert ellipfun('cc', 0, 0) == 1
    assert ellipfun('dd', 0, 0) == 1
    assert ellipfun('nn', 0, 0) == 1
    assert ellipfun('sn', 0.25, 0).ae(sin(0.25))
    assert ellipfun('cn', 0.25, 0).ae(cos(0.25))
    assert ellipfun('dn', 0.25, 0).ae(1)
    assert ellipfun('ns', 0.25, 0).ae(csc(0.25))
    assert ellipfun('nc', 0.25, 0).ae(sec(0.25))
    assert ellipfun('nd', 0.25, 0).ae(1)
    assert ellipfun('sc', 0.25, 0).ae(tan(0.25))
    assert ellipfun('sd', 0.25, 0).ae(sin(0.25))
    assert ellipfun('cd', 0.25, 0).ae(cos(0.25))
    assert ellipfun('cs', 0.25, 0).ae(cot(0.25))
    assert ellipfun('dc', 0.25, 0).ae(sec(0.25))
    assert ellipfun('ds', 0.25, 0).ae(csc(0.25))
    assert ellipfun('sn', 0.25, 1).ae(tanh(0.25))
    assert ellipfun('cn', 0.25, 1).ae(sech(0.25))
    assert ellipfun('dn', 0.25, 1).ae(sech(0.25))
    assert ellipfun('ns', 0.25, 1).ae(coth(0.25))
    assert ellipfun('nc', 0.25, 1).ae(cosh(0.25))
    assert ellipfun('nd', 0.25, 1).ae(cosh(0.25))
    assert ellipfun('sc', 0.25, 1).ae(sinh(0.25))
    assert ellipfun('sd', 0.25, 1).ae(sinh(0.25))
    assert ellipfun('cd', 0.25, 1).ae(1)
    assert ellipfun('cs', 0.25, 1).ae(csch(0.25))
    assert ellipfun('dc', 0.25, 1).ae(1)
    assert ellipfun('ds', 0.25, 1).ae(csch(0.25))
    assert ellipfun('sn', 0.25, 0.5).ae(0.24615967096986145833)
    assert ellipfun('cn', 0.25, 0.5).ae(0.96922928989378439337)
    assert ellipfun('dn', 0.25, 0.5).ae(0.98473484156599474563)
    assert ellipfun('ns', 0.25, 0.5).ae(4.0624038700573130369)
    assert ellipfun('nc', 0.25, 0.5).ae(1.0317476065024692949)
    assert ellipfun('nd', 0.25, 0.5).ae(1.0155017958029488665)
    assert ellipfun('sc', 0.25, 0.5).ae(0.25397465134058993408)
    assert ellipfun('sd', 0.25, 0.5).ae(0.24997558792415733063)
    assert ellipfun('cd', 0.25, 0.5).ae(0.98425408443195497052)
    assert ellipfun('cs', 0.25, 0.5).ae(3.9374008182374110826)
    assert ellipfun('dc', 0.25, 0.5).ae(1.0159978158253033913)
    assert ellipfun('ds', 0.25, 0.5).ae(4.0003906313579720593)




def test_calculate_nome():
    mp.dps = 100

    q = calculate_nome(zero)
    assert(q == zero)

    mp.dps = 25
    # used Mathematica's EllipticNomeQ[m]
    math1 = [(mpf(1)/10, mpf('0.006584651553858370274473060')),
             (mpf(2)/10, mpf('0.01394285727531826872146409')),
             (mpf(3)/10, mpf('0.02227743615715350822901627')),
             (mpf(4)/10, mpf('0.03188334731336317755064299')),
             (mpf(5)/10, mpf('0.04321391826377224977441774')),
             (mpf(6)/10, mpf('0.05702025781460967637754953')),
             (mpf(7)/10, mpf('0.07468994353717944761143751')),
             (mpf(8)/10, mpf('0.09927369733882489703607378')),
             (mpf(9)/10, mpf('0.1401731269542615524091055')),
             (mpf(9)/10, mpf('0.1401731269542615524091055'))]

    for i in math1:
        m = i[0]
        q = calculate_nome(sqrt(m))
        assert q.ae(i[1])

    mp.dps = 15

def test_jtheta():
    mp.dps = 25

    z = q = zero
    for n in range(1,5):
        value = jtheta(n, z, q)
        assert(value == (n-1)//2)

    for q in [one, mpf(2)]:
        for n in range(1,5):
            pytest.raises(ValueError, lambda: jtheta(n, z, q))

    z = one/10
    q = one/11

    # Mathematical N[EllipticTheta[1, 1/10, 1/11], 25]
    res = mpf('0.1069552990104042681962096')
    result = jtheta(1, z, q)
    assert(result.ae(res))

    # Mathematica N[EllipticTheta[2, 1/10, 1/11], 25]
    res = mpf('1.101385760258855791140606')
    result = jtheta(2, z, q)
    assert(result.ae(res))

    # Mathematica N[EllipticTheta[3, 1/10, 1/11], 25]
    res = mpf('1.178319743354331061795905')
    result = jtheta(3, z, q)
    assert(result.ae(res))

    # Mathematica N[EllipticTheta[4, 1/10, 1/11], 25]
    res = mpf('0.8219318954665153577314573')
    result = jtheta(4, z, q)
    assert(result.ae(res))

    # test for sin zeros for jtheta(1, z, q)
    # test for cos zeros for jtheta(2, z, q)
    z1 = pi
    z2 = pi/2
    for i in range(10):
        qstring = str(random.random())
        q = mpf(qstring)
        result = jtheta(1, z1, q)
        assert(result.ae(0))
        result = jtheta(2, z2, q)
        assert(result.ae(0))
    mp.dps = 15


def test_jtheta_issue_79():
    # near the circle of covergence |q| = 1 the convergence slows
    # down; for |q| > Q_LIM the theta functions raise ValueError
    mp.dps = 30
    mp.dps += 30
    q = mpf(6)/10 - one/10**6 - mpf(8)/10 * j
    mp.dps -= 30
    # Mathematica run first
    # N[EllipticTheta[3, 1, 6/10 - 10^-6 - 8/10*I], 2000]
    # then it works:
    # N[EllipticTheta[3, 1, 6/10 - 10^-6 - 8/10*I], 30]
    res = mpf('32.0031009628901652627099524264') + \
          mpf('16.6153027998236087899308935624') * j
    result = jtheta(3, 1, q)
    # check that for abs(q) > Q_LIM a ValueError exception is raised
    mp.dps += 30
    q = mpf(6)/10 - one/10**7 - mpf(8)/10 * j
    mp.dps -= 30
    pytest.raises(ValueError, lambda: jtheta(3, 1, q))

    # bug reported in issue 79
    mp.dps = 100
    z = (1+j)/3
    q = mpf(368983957219251)/10**15 + mpf(636363636363636)/10**15 * j
    # Mathematica N[EllipticTheta[1, z, q], 35]
    res = mpf('2.4439389177990737589761828991467471') + \
          mpf('0.5446453005688226915290954851851490') *j
    mp.dps = 30
    result = jtheta(1, z, q)
    assert(result.ae(res))
    mp.dps = 80
    z = 3 + 4*j
    q = 0.5 + 0.5*j
    r1 = jtheta(1, z, q)
    mp.dps = 15
    r2 = jtheta(1, z, q)
    assert r1.ae(r2)
    mp.dps = 80
    z = 3 + j
    q1 = exp(j*3)
    # longer test
    # for n in range(1, 6)
    for n in range(1, 2):
        mp.dps = 80
        q = q1*(1 - mpf(1)/10**n)
        r1 = jtheta(1, z, q)
        mp.dps = 15
        r2 = jtheta(1, z, q)
    assert r1.ae(r2)
    mp.dps = 15
    # issue 79 about high derivatives
    assert jtheta(3, 4.5, 0.25, 9).ae(1359.04892680683)
    assert jtheta(3, 4.5, 0.25, 50).ae(-6.14832772630905e+33)
    mp.dps = 50
    r = jtheta(3, 4.5, 0.25, 9)
    assert r.ae('1359.048926806828939547859396600218966947753213803')
    r = jtheta(3, 4.5, 0.25, 50)
    assert r.ae('-6148327726309051673317975084654262.4119215720343656')

def test_jtheta_identities():
    """
    Tests the some of the jacobi identidies found in Abramowitz,
    Sec. 16.28, Pg. 576. The identities are tested to 1 part in 10^98.
    """
    mp.dps = 110
    eps1 = ldexp(eps, 30)

    for i in range(10):
        qstring = str(random.random())
        q = mpf(qstring)

        zstring = str(10*random.random())
        z = mpf(zstring)
        # Abramowitz 16.28.1
        # v_1(z, q)**2 * v_4(0, q)**2 =   v_3(z, q)**2 * v_2(0, q)**2
        #                               - v_2(z, q)**2 * v_3(0, q)**2
        term1 = (jtheta(1, z, q)**2) * (jtheta(4, zero, q)**2)
        term2 = (jtheta(3, z, q)**2) * (jtheta(2, zero, q)**2)
        term3 = (jtheta(2, z, q)**2) * (jtheta(3, zero, q)**2)
        equality = term1 - term2 + term3
        assert(equality.ae(0, eps1))

        zstring = str(100*random.random())
        z = mpf(zstring)
        # Abramowitz 16.28.2
        # v_2(z, q)**2 * v_4(0, q)**2 =   v_4(z, q)**2 * v_2(0, q)**2
        #                               - v_1(z, q)**2 * v_3(0, q)**2
        term1 = (jtheta(2, z, q)**2) * (jtheta(4, zero, q)**2)
        term2 = (jtheta(4, z, q)**2) * (jtheta(2, zero, q)**2)
        term3 = (jtheta(1, z, q)**2) * (jtheta(3, zero, q)**2)
        equality = term1 - term2 + term3
        assert(equality.ae(0, eps1))

        # Abramowitz 16.28.3
        # v_3(z, q)**2 * v_4(0, q)**2 =   v_4(z, q)**2 * v_3(0, q)**2
        #                               - v_1(z, q)**2 * v_2(0, q)**2
        term1 = (jtheta(3, z, q)**2) * (jtheta(4, zero, q)**2)
        term2 = (jtheta(4, z, q)**2) * (jtheta(3, zero, q)**2)
        term3 = (jtheta(1, z, q)**2) * (jtheta(2, zero, q)**2)
        equality = term1 - term2 + term3
        assert(equality.ae(0, eps1))

        # Abramowitz 16.28.4
        # v_4(z, q)**2 * v_4(0, q)**2 =   v_3(z, q)**2 * v_3(0, q)**2
        #                               - v_2(z, q)**2 * v_2(0, q)**2
        term1 = (jtheta(4, z, q)**2) * (jtheta(4, zero, q)**2)
        term2 = (jtheta(3, z, q)**2) * (jtheta(3, zero, q)**2)
        term3 = (jtheta(2, z, q)**2) * (jtheta(2, zero, q)**2)
        equality = term1 - term2 + term3
        assert(equality.ae(0, eps1))

        # Abramowitz 16.28.5
        # v_2(0, q)**4 + v_4(0, q)**4 == v_3(0, q)**4
        term1 = (jtheta(2, zero, q))**4
        term2 = (jtheta(4, zero, q))**4
        term3 = (jtheta(3, zero, q))**4
        equality = term1 + term2 - term3
        assert(equality.ae(0, eps1))
    mp.dps = 15

def test_jtheta_complex():
    mp.dps = 30
    z = mpf(1)/4 + j/8
    q = mpf(1)/3 + j/7
    # Mathematica N[EllipticTheta[1, 1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('0.31618034835986160705729105731678285') + \
          mpf('0.07542013825835103435142515194358975') * j
    r = jtheta(1, z, q)
    assert(mpc_ae(r, res))

    # Mathematica N[EllipticTheta[2, 1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('1.6530986428239765928634711417951828') + \
          mpf('0.2015344864707197230526742145361455') * j
    r = jtheta(2, z, q)
    assert(mpc_ae(r, res))

    # Mathematica N[EllipticTheta[3, 1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('1.6520564411784228184326012700348340') + \
          mpf('0.1998129119671271328684690067401823') * j
    r = jtheta(3, z, q)
    assert(mpc_ae(r, res))

    # Mathematica N[EllipticTheta[4, 1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('0.37619082382228348252047624089973824') - \
          mpf('0.15623022130983652972686227200681074') * j
    r = jtheta(4, z, q)
    assert(mpc_ae(r, res))

    # check some theta function identities
    mp.dos = 100
    z = mpf(1)/4 + j/8
    q = mpf(1)/3 + j/7
    mp.dps += 10
    a = [0,0, jtheta(2, 0, q), jtheta(3, 0, q), jtheta(4, 0, q)]
    t = [0, jtheta(1, z, q), jtheta(2, z, q), jtheta(3, z, q), jtheta(4, z, q)]
    r = [(t[2]*a[4])**2 - (t[4]*a[2])**2 + (t[1] *a[3])**2,
        (t[3]*a[4])**2 - (t[4]*a[3])**2 + (t[1] *a[2])**2,
        (t[1]*a[4])**2 - (t[3]*a[2])**2 + (t[2] *a[3])**2,
        (t[4]*a[4])**2 - (t[3]*a[3])**2 + (t[2] *a[2])**2,
        a[2]**4 + a[4]**4 - a[3]**4]
    mp.dps -= 10
    for x in r:
        assert(mpc_ae(x, mpc(0)))
    mp.dps = 15

def test_djtheta():
    mp.dps = 30

    z = one/7 + j/3
    q = one/8 + j/5
    # Mathematica N[EllipticThetaPrime[1, 1/7 + I/3, 1/8 + I/5], 35]
    res = mpf('1.5555195883277196036090928995803201') - \
          mpf('0.02439761276895463494054149673076275') * j
    result = jtheta(1, z, q, 1)
    assert(mpc_ae(result, res))

    # Mathematica N[EllipticThetaPrime[2, 1/7 + I/3, 1/8 + I/5], 35]
    res = mpf('0.19825296689470982332701283509685662') - \
          mpf('0.46038135182282106983251742935250009') * j
    result = jtheta(2, z, q, 1)
    assert(mpc_ae(result, res))

    # Mathematica N[EllipticThetaPrime[3, 1/7 + I/3, 1/8 + I/5], 35]
    res = mpf('0.36492498415476212680896699407390026') - \
          mpf('0.57743812698666990209897034525640369') * j
    result = jtheta(3, z, q, 1)
    assert(mpc_ae(result, res))

    # Mathematica N[EllipticThetaPrime[4, 1/7 + I/3, 1/8 + I/5], 35]
    res = mpf('-0.38936892528126996010818803742007352') + \
          mpf('0.66549886179739128256269617407313625') * j
    result = jtheta(4, z, q, 1)
    assert(mpc_ae(result, res))

    for i in range(10):
        q = (one*random.random() + j*random.random())/2
        # identity in Wittaker, Watson &21.41
        a = jtheta(1, 0, q, 1)
        b = jtheta(2, 0, q)*jtheta(3, 0, q)*jtheta(4, 0, q)
        assert(a.ae(b))

    # test higher derivatives
    mp.dps = 20
    for q,z in [(one/3, one/5), (one/3 + j/8, one/5),
        (one/3, one/5 + j/8), (one/3 + j/7, one/5 + j/8)]:
        for n in [1, 2, 3, 4]:
            r = jtheta(n, z, q, 2)
            r1 = diff(lambda zz: jtheta(n, zz, q), z, n=2)
            assert r.ae(r1)
            r = jtheta(n, z, q, 3)
            r1 = diff(lambda zz: jtheta(n, zz, q), z, n=3)
            assert r.ae(r1)

    # identity in Wittaker, Watson &21.41
    q = one/3
    z = zero
    a = [0]*5
    a[1] = jtheta(1, z, q, 3)/jtheta(1, z, q, 1)
    for n in [2,3,4]:
        a[n] = jtheta(n, z, q, 2)/jtheta(n, z, q)
    equality = a[2] + a[3] + a[4] - a[1]
    assert(equality.ae(0))
    mp.dps = 15

def test_jsn():
    """
    Test some special cases of the sn(z, q) function.
    """
    mp.dps = 100

    # trival case
    result = jsn(zero, zero)
    assert(result == zero)

    # Abramowitz Table 16.5
    #
    # sn(0, m) = 0

    for i in range(10):
        qstring = str(random.random())
        q = mpf(qstring)

        equality = jsn(zero, q)
        assert(equality.ae(0))

    # Abramowitz Table 16.6.1
    #
    # sn(z, 0) = sin(z), m == 0
    #
    # sn(z, 1) = tanh(z), m == 1
    #
    # It would be nice to test these, but I find that they run
    # in to numerical trouble.  I'm currently treating as a boundary
    # case for sn function.

    mp.dps = 25
    arg = one/10
    #N[JacobiSN[1/10, 2^-100], 25]
    res = mpf('0.09983341664682815230681420')
    m = ldexp(one, -100)
    result = jsn(arg, m)
    assert(result.ae(res))

    # N[JacobiSN[1/10, 1/10], 25]
    res = mpf('0.09981686718599080096451168')
    result = jsn(arg, arg)
    assert(result.ae(res))
    mp.dps = 15

def test_jcn():
    """
    Test some special cases of the cn(z, q) function.
    """
    mp.dps = 100

    # Abramowitz Table 16.5
    # cn(0, q) = 1
    qstring = str(random.random())
    q = mpf(qstring)
    cn = jcn(zero, q)
    assert(cn.ae(one))

    # Abramowitz Table 16.6.2
    #
    # cn(u, 0) = cos(u), m == 0
    #
    # cn(u, 1) = sech(z), m == 1
    #
    # It would be nice to test these, but I find that they run
    # in to numerical trouble.  I'm currently treating as a boundary
    # case for cn function.

    mp.dps = 25
    arg = one/10
    m = ldexp(one, -100)
    #N[JacobiCN[1/10, 2^-100], 25]
    res = mpf('0.9950041652780257660955620')
    result = jcn(arg, m)
    assert(result.ae(res))

    # N[JacobiCN[1/10, 1/10], 25]
    res = mpf('0.9950058256237368748520459')
    result = jcn(arg, arg)
    assert(result.ae(res))
    mp.dps = 15

def test_jdn():
    """
    Test some special cases of the dn(z, q) function.
    """
    mp.dps = 100

    # Abramowitz Table 16.5
    # dn(0, q) = 1
    mstring = str(random.random())
    m = mpf(mstring)

    dn = jdn(zero, m)
    assert(dn.ae(one))

    mp.dps = 25
    # N[JacobiDN[1/10, 1/10], 25]
    res = mpf('0.9995017055025556219713297')
    arg = one/10
    result = jdn(arg, arg)
    assert(result.ae(res))
    mp.dps = 15


def test_sn_cn_dn_identities():
    """
    Tests the some of the jacobi elliptic function identities found
    on Mathworld. Haven't found in Abramowitz.
    """
    mp.dps = 100
    N = 5
    for i in range(N):
        qstring = str(random.random())
        q = mpf(qstring)
        zstring = str(100*random.random())
        z = mpf(zstring)

        # MathWorld
        # sn(z, q)**2 + cn(z, q)**2 == 1
        term1 = jsn(z, q)**2
        term2 = jcn(z, q)**2
        equality = one - term1 - term2
        assert(equality.ae(0))

    # MathWorld
    # k**2 * sn(z, m)**2 + dn(z, m)**2 == 1
    for i in range(N):
        mstring = str(random.random())
        m = mpf(qstring)
        k = m.sqrt()
        zstring = str(10*random.random())
        z = mpf(zstring)
        term1 = k**2 * jsn(z, m)**2
        term2 = jdn(z, m)**2
        equality = one - term1 - term2
        assert(equality.ae(0))


    for i in range(N):
        mstring = str(random.random())
        m = mpf(mstring)
        k = m.sqrt()
        zstring = str(random.random())
        z = mpf(zstring)

        # MathWorld
        # k**2 * cn(z, m)**2 + (1 - k**2) = dn(z, m)**2
        term1 = k**2 * jcn(z, m)**2
        term2 = 1 - k**2
        term3 = jdn(z, m)**2
        equality = term3 - term1 - term2
        assert(equality.ae(0))

        K = ellipk(k**2)
        # Abramowitz Table 16.5
        # sn(K, m) = 1; K is K(k), first complete elliptic integral
        r = jsn(K, m)
        assert(r.ae(one))

        # Abramowitz Table 16.5
        # cn(K, q) = 0; K is K(k), first complete elliptic integral
        equality = jcn(K, m)
        assert(equality.ae(0))

        # Abramowitz Table 16.6.3
        # dn(z, 0) = 1, m == 0
        z = m
        value = jdn(z, zero)
        assert(value.ae(one))

    mp.dps = 15

def test_sn_cn_dn_complex():
    mp.dps = 30
    # N[JacobiSN[1/4 + I/8, 1/3 + I/7], 35] in Mathematica
    res = mpf('0.2495674401066275492326652143537') + \
          mpf('0.12017344422863833381301051702823') * j
    u = mpf(1)/4 + j/8
    m = mpf(1)/3 + j/7
    r = jsn(u, m)
    assert(mpc_ae(r, res))

    #N[JacobiCN[1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('0.9762691700944007312693721148331') - \
          mpf('0.0307203994181623243583169154824')*j
    r = jcn(u, m)
    #assert r.real.ae(res.real)
    #assert r.imag.ae(res.imag)
    assert(mpc_ae(r, res))

    #N[JacobiDN[1/4 + I/8, 1/3 + I/7], 35]
    res = mpf('0.99639490163039577560547478589753039') - \
          mpf('0.01346296520008176393432491077244994')*j
    r = jdn(u, m)
    assert(mpc_ae(r, res))
    mp.dps = 15

def test_elliptic_integrals():
    # Test cases from Carlson's paper
    mp.dps = 15
    assert elliprd(0,2,1).ae(1.7972103521033883112)
    assert elliprd(2,3,4).ae(0.16510527294261053349)
    assert elliprd(j,-j,2).ae(0.65933854154219768919)
    assert elliprd(0,j,-j).ae(1.2708196271909686299 + 2.7811120159520578777j)
    assert elliprd(0,j-1,j).ae(-1.8577235439239060056 - 0.96193450888838559989j)
    assert elliprd(-2-j,-j,-1+j).ae(1.8249027393703805305 - 1.2218475784827035855j)
    # extra test cases
    assert elliprg(0,0,0) == 0
    assert elliprg(0,0,16).ae(2)
    assert elliprg(0,16,0).ae(2)
    assert elliprg(16,0,0).ae(2)
    assert elliprg(1,4,0).ae(1.2110560275684595248036)
    assert elliprg(1,0,4).ae(1.2110560275684595248036)
    assert elliprg(0,4,1).ae(1.2110560275684595248036)
    # should be symmetric -- fixes a bug present in the paper
    x,y,z = 1,1j,-1+1j
    assert elliprg(x,y,z).ae(0.64139146875812627545 + 0.58085463774808290907j)
    assert elliprg(x,z,y).ae(0.64139146875812627545 + 0.58085463774808290907j)
    assert elliprg(y,x,z).ae(0.64139146875812627545 + 0.58085463774808290907j)
    assert elliprg(y,z,x).ae(0.64139146875812627545 + 0.58085463774808290907j)
    assert elliprg(z,x,y).ae(0.64139146875812627545 + 0.58085463774808290907j)
    assert elliprg(z,y,x).ae(0.64139146875812627545 + 0.58085463774808290907j)

    for n in [5, 15, 30, 60, 100]:
        mp.dps = n
        assert elliprf(1,2,0).ae('1.3110287771460599052324197949455597068413774757158115814084108519003952935352071251151477664807145467230678763')
        assert elliprf(0.5,1,0).ae('1.854074677301371918433850347195260046217598823521766905585928045056021776838119978357271861650371897277771871')
        assert elliprf(j,-j,0).ae('1.854074677301371918433850347195260046217598823521766905585928045056021776838119978357271861650371897277771871')
        assert elliprf(j-1,j,0).ae(mpc('0.79612586584233913293056938229563057846592264089185680214929401744498956943287031832657642790719940442165621412',
            '-1.2138566698364959864300942567386038975419875860741507618279563735753073152507112254567291141460317931258599889'))
        assert elliprf(2,3,4).ae('0.58408284167715170669284916892566789240351359699303216166309375305508295130412919665541330837704050454472379308')
        assert elliprf(j,-j,2).ae('1.0441445654064360931078658361850779139591660747973017593275012615517220315993723776182276555339288363064476126')
        assert elliprf(j-1,j,1-j).ae(mpc('0.93912050218619371196624617169781141161485651998254431830645241993282941057500174238125105410055253623847335313',
            '-0.53296252018635269264859303449447908970360344322834582313172115220559316331271520508208025270300138589669326136'))
        assert elliprc(0,0.25).ae(+pi)
        assert elliprc(2.25,2).ae(+ln2)
        assert elliprc(0,j).ae(mpc('1.1107207345395915617539702475151734246536554223439225557713489017391086982748684776438317336911913093408525532',
            '-1.1107207345395915617539702475151734246536554223439225557713489017391086982748684776438317336911913093408525532'))
        assert elliprc(-j,j).ae(mpc('1.2260849569072198222319655083097718755633725139745941606203839524036426936825652935738621522906572884239069297',
            '-0.34471136988767679699935618332997956653521218571295874986708834375026550946053920574015526038040124556716711353'))
        assert elliprc(0.25,-2).ae(ln2/3)
        assert elliprc(j,-1).ae(mpc('0.77778596920447389875196055840799837589537035343923012237628610795937014001905822029050288316217145443865649819',
            '0.1983248499342877364755170948292130095921681309577950696116251029742793455964385947473103628983664877025779304'))
        assert elliprj(0,1,2,3).ae('0.77688623778582332014190282640545501102298064276022952731669118325952563819813258230708177398475643634103990878')
        assert elliprj(2,3,4,5).ae('0.14297579667156753833233879421985774801466647854232626336218889885463800128817976132826443904216546421431528308')
        assert elliprj(2,3,4,-1+j).ae(mpc('0.13613945827770535203521374457913768360237593025944342652613569368333226052158214183059386307242563164036672709',
            '-0.38207561624427164249600936454845112611060375760094156571007648297226090050927156176977091273224510621553615189'))
        assert elliprj(j,-j,0,2).ae('1.6490011662710884518243257224860232300246792717163891216346170272567376981346412066066050103935109581019055806')
        assert elliprj(-1+j,-1-j,1,2).ae('0.94148358841220238083044612133767270187474673547917988681610772381758628963408843935027667916713866133196845063')
        assert elliprj(j,-j,0,1-j).ae(mpc('1.8260115229009316249372594065790946657011067182850435297162034335356430755397401849070610280860044610878657501',
            '1.2290661908643471500163617732957042849283739403009556715926326841959667290840290081010472716420690899886276961'))
        assert elliprj(-1+j,-1-j,1,-3+j).ae(mpc('-0.61127970812028172123588152373622636829986597243716610650831553882054127570542477508023027578037045504958619422',
            '-1.0684038390006807880182112972232562745485871763154040245065581157751693730095703406209466903752930797510491155'))
        assert elliprj(-1+j,-2-j,-j,-1+j).ae(mpc('1.8249027393703805304622013339009022294368078659619988943515764258335975852685224202567854526307030593012768954',
            '-1.2218475784827035854568450371590419833166777535029296025352291308244564398645467465067845461070602841312456831'))

        assert elliprg(0,16,16).ae(+pi)
        assert elliprg(2,3,4).ae('1.7255030280692277601061148835701141842692457170470456590515892070736643637303053506944907685301315299153040991')
        assert elliprg(0,j,-j).ae('0.42360654239698954330324956174109581824072295516347109253028968632986700241706737986160014699730561497106114281')
        assert elliprg(j-1,j,0).ae(mpc('0.44660591677018372656731970402124510811555212083508861036067729944477855594654762496407405328607219895053798354',
            '0.70768352357515390073102719507612395221369717586839400605901402910893345301718731499237159587077682267374159282'))
        assert elliprg(-j,j-1,j).ae(mpc('0.36023392184473309033675652092928695596803358846377334894215349632203382573844427952830064383286995172598964266',
            '0.40348623401722113740956336997761033878615232917480045914551915169013722542827052849476969199578321834819903921'))
        assert elliprg(0, mpf('0.0796'), 4).ae('1.0284758090288040009838871385180217366569777284430590125081211090574701293154645750017813190805144572673802094')
    mp.dps = 15

    # more test cases for the branch of ellippi / elliprj
    assert elliprj(-1-0.5j, -10-6j, -10-3j, -5+10j).ae(0.128470516743927699 + 0.102175950778504625j, abs_eps=1e-8)
    assert elliprj(1.987, 4.463 - 1.614j, 0, -3.965).ae(-0.341575118513811305 - 0.394703757004268486j, abs_eps=1e-8)
    assert elliprj(0.3068, -4.037+0.632j, 1.654, -0.9609).ae(-1.14735199581485639 - 0.134450158867472264j, abs_eps=1e-8)
    assert elliprj(0.3068, -4.037-0.632j, 1.654, -0.9609).ae(1.758765901861727 - 0.161002343366626892j, abs_eps=1e-5)
    assert elliprj(0.3068, -4.037+0.0632j, 1.654, -0.9609).ae(-1.17157627949475577 - 0.069182614173988811j, abs_eps=1e-8)
    assert elliprj(0.3068, -4.037+0.00632j, 1.654, -0.9609).ae(-1.17337595670549633 - 0.0623069224526925j, abs_eps=1e-8)

    # these require accurate integration
    assert elliprj(0.3068, -4.037-0.0632j, 1.654, -0.9609).ae(1.77940452391261626 + 0.0388711305592447234j)
    assert elliprj(0.3068, -4.037-0.00632j, 1.654, -0.9609).ae(1.77806722756403055 + 0.0592749824572262329j)
    # issue #571
    assert ellippi(2.1 + 0.94j, 2.3 + 0.98j, 2.5 + 0.01j).ae(-0.40652414240811963438 + 2.1547659461404749309j)

    assert ellippi(2.0-1.0j, 2.0+1.0j).ae(1.8578723151271115 - 1.18642180609983531j)
    assert ellippi(2.0-0.5j, 0.5+1.0j).ae(0.936761970766645807 - 1.61876787838890786j)
    assert ellippi(2.0, 1.0+1.0j).ae(0.999881420735506708 - 2.4139272867045391j)
    assert ellippi(2.0+1.0j, 2.0-1.0j).ae(1.8578723151271115 + 1.18642180609983531j)
    assert ellippi(2.0+1.0j, 2.0).ae(2.78474654927885845 + 2.02204728966993314j)

def test_issue_238():
    assert isnan(qfrom(m=nan))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\plotting\frame\test_frame_color.py ===
""" Test cases for DataFrame.plot """
import re

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_colors,
    _check_plot_works,
    _unpack_cycler,
)
from pandas.util.version import Version

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
cm = pytest.importorskip("matplotlib.cm")


def _check_colors_box(bp, box_c, whiskers_c, medians_c, caps_c="k", fliers_c=None):
    if fliers_c is None:
        fliers_c = "k"
    _check_colors(bp["boxes"], linecolors=[box_c] * len(bp["boxes"]))
    _check_colors(bp["whiskers"], linecolors=[whiskers_c] * len(bp["whiskers"]))
    _check_colors(bp["medians"], linecolors=[medians_c] * len(bp["medians"]))
    _check_colors(bp["fliers"], linecolors=[fliers_c] * len(bp["fliers"]))
    _check_colors(bp["caps"], linecolors=[caps_c] * len(bp["caps"]))


class TestDataFrameColor:
    @pytest.mark.parametrize(
        "color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    )
    def test_mpl2_color_cycle_str(self, color):
        # GH 15516
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )
        _check_plot_works(df.plot, color=color)

    def test_color_single_series_list(self):
        # GH 3486
        df = DataFrame({"A": [1, 2, 3]})
        _check_plot_works(df.plot, color=["red"])

    @pytest.mark.parametrize("color", [(1, 0, 0), (1, 0, 0, 0.5)])
    def test_rgb_tuple_color(self, color):
        # GH 16695
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        _check_plot_works(df.plot, x="x", y="y", color=color)

    def test_color_empty_string(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        with pytest.raises(ValueError, match="Invalid color argument:"):
            df.plot(color="")

    def test_color_and_style_arguments(self):
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        # passing both 'color' and 'style' arguments should be allowed
        # if there is no color symbol in the style strings:
        ax = df.plot(color=["red", "black"], style=["-", "--"])
        # check that the linestyles are correctly set:
        linestyle = [line.get_linestyle() for line in ax.lines]
        assert linestyle == ["-", "--"]
        # check that the colors are correctly set:
        color = [line.get_color() for line in ax.lines]
        assert color == ["red", "black"]
        # passing both 'color' and 'style' arguments should not be allowed
        # if there is a color symbol in the style strings:
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' keyword "
            "argument. Please use one or the other or pass 'style' without a color "
            "symbol"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(color=["red", "black"], style=["k-", "r--"])

    @pytest.mark.parametrize(
        "color, expected",
        [
            ("green", ["green"] * 4),
            (["yellow", "red", "green", "blue"], ["yellow", "red", "green", "blue"]),
        ],
    )
    def test_color_and_marker(self, color, expected):
        # GH 21003
        df = DataFrame(np.random.default_rng(2).random((7, 4)))
        ax = df.plot(color=color, style="d--")
        # check colors
        result = [i.get_color() for i in ax.lines]
        assert result == expected
        # check markers and linestyles
        assert all(i.get_linestyle() == "--" for i in ax.lines)
        assert all(i.get_marker() == "d" for i in ax.lines)

    def test_bar_colors(self):
        default_colors = _unpack_cycler(plt.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar()
        _check_colors(ax.patches[::5], facecolors=default_colors[:5])

    def test_bar_colors_custom(self):
        custom_colors = "rgcby"
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.bar(color=custom_colors)
        _check_colors(ax.patches[::5], facecolors=custom_colors)

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_bar_colors_cmap(self, colormap):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        ax = df.plot.bar(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        _check_colors(ax.patches[::5], facecolors=rgba_colors)

    def test_bar_colors_single_col(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot.bar(color="DodgerBlue")
        _check_colors([ax.patches[0]], facecolors=["DodgerBlue"])

    def test_bar_colors_green(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(kind="bar", color="green")
        _check_colors(ax.patches[::5], facecolors=["green"] * 5)

    def test_bar_user_colors(self):
        df = DataFrame(
            {"A": range(4), "B": range(1, 5), "color": ["red", "blue", "blue", "red"]}
        )
        # This should *only* work when `y` is specified, else
        # we use one color per column
        ax = df.plot.bar(y="A", color=df["color"])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        assert result == expected

    def test_if_scatterplot_colorbar_affects_xaxis_visibility(self):
        # addressing issue #10611, to ensure colobar does not
        # interfere with x-axis label and ticklabels with
        # ipython inline backend.
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        ax1 = df.plot.scatter(x="A label", y="B label")
        ax2 = df.plot.scatter(x="A label", y="B label", c="C label")

        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_minorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_minorticklabels()]
        assert vis1 == vis2

        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_majorticklabels()]
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_majorticklabels()]
        assert vis1 == vis2

        assert (
            ax1.xaxis.get_label().get_visible() == ax2.xaxis.get_label().get_visible()
        )

    def test_if_hexbin_xaxis_label_is_visible(self):
        # addressing issue #10678, to ensure colobar does not
        # interfere with x-axis label and ticklabels with
        # ipython inline backend.
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        ax = df.plot.hexbin("A label", "B label", gridsize=12)
        assert all(vis.get_visible() for vis in ax.xaxis.get_minorticklabels())
        assert all(vis.get_visible() for vis in ax.xaxis.get_majorticklabels())
        assert ax.xaxis.get_label().get_visible()

    def test_if_scatterplot_colorbars_are_next_to_parent_axes(self):
        random_array = np.random.default_rng(2).random((10, 3))
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        fig, axes = plt.subplots(1, 2)
        df.plot.scatter("A label", "B label", c="C label", ax=axes[0])
        df.plot.scatter("A label", "B label", c="C label", ax=axes[1])
        plt.tight_layout()

        points = np.array([ax.get_position().get_points() for ax in fig.axes])
        axes_x_coords = points[:, :, 0]
        parent_distance = axes_x_coords[1, :] - axes_x_coords[0, :]
        colorbar_distance = axes_x_coords[3, :] - axes_x_coords[2, :]
        assert np.isclose(parent_distance, colorbar_distance, atol=1e-7).all()

    @pytest.mark.parametrize("cmap", [None, "Greys"])
    def test_scatter_with_c_column_name_with_colors(self, cmap):
        # https://github.com/pandas-dev/pandas/issues/34316

        df = DataFrame(
            [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]],
            columns=["length", "width"],
        )
        df["species"] = ["r", "r", "g", "g", "b"]
        if cmap is not None:
            with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
                ax = df.plot.scatter(x=0, y=1, cmap=cmap, c="species")
        else:
            ax = df.plot.scatter(x=0, y=1, c="species", cmap=cmap)
        assert ax.collections[0].colorbar is None

    def test_scatter_colors(self):
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        with pytest.raises(TypeError, match="Specify exactly one of `c` and `color`"):
            df.plot.scatter(x="a", y="b", c="c", color="green")

    def test_scatter_colors_not_raising_warnings(self):
        # GH-53908. Do not raise UserWarning: No data for colormapping
        # provided via 'c'. Parameters 'cmap' will be ignored
        df = DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        with tm.assert_produces_warning(None):
            df.plot.scatter(x="x", y="y", c="b")

    def test_scatter_colors_default(self):
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        ax = df.plot.scatter(x="a", y="b", c="c")
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array(mpl.colors.ColorConverter.to_rgba(default_colors[0])),
        )

    def test_scatter_colors_white(self):
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        ax = df.plot.scatter(x="a", y="b", color="white")
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array([1, 1, 1, 1], dtype=np.float64),
        )

    def test_scatter_colorbar_different_cmap(self):
        # GH 33389
        df = DataFrame({"x": [1, 2, 3], "y": [1, 3, 2], "c": [1, 2, 3]})
        df["x2"] = df["x"] + 1

        _, ax = plt.subplots()
        df.plot("x", "y", c="c", kind="scatter", cmap="cividis", ax=ax)
        df.plot("x2", "y", c="c", kind="scatter", cmap="magma", ax=ax)

        assert ax.collections[0].cmap.name == "cividis"
        assert ax.collections[1].cmap.name == "magma"

    def test_line_colors(self):
        custom_colors = "rgcby"
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        ax = df.plot(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)

        plt.close("all")

        ax2 = df.plot(color=custom_colors)
        lines2 = ax2.get_lines()

        for l1, l2 in zip(ax.get_lines(), lines2):
            assert l1.get_color() == l2.get_color()

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_line_colors_cmap(self, colormap):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        _check_colors(ax.get_lines(), linecolors=rgba_colors)

    def test_line_colors_single_col(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # make color a list if plotting one column frame
        # handles cases like df.plot(color='DodgerBlue')
        ax = df.loc[:, [0]].plot(color="DodgerBlue")
        _check_colors(ax.lines, linecolors=["DodgerBlue"])

    def test_line_colors_single_color(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(color="red")
        _check_colors(ax.get_lines(), linecolors=["red"] * 5)

    def test_line_colors_hex(self):
        # GH 10299
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors = ["#FF0000", "#0000FF", "#FFFF00", "#000000", "#FFFFFF"]
        ax = df.plot(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)

    def test_dont_modify_colors(self):
        colors = ["r", "g", "b"]
        DataFrame(np.random.default_rng(2).random((10, 2))).plot(color=colors)
        assert len(colors) == 3

    def test_line_colors_and_styles_subplots(self):
        # GH 9894
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        axes = df.plot(subplots=True)
        for ax, c in zip(axes, list(default_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize("color", ["k", "green"])
    def test_line_colors_and_styles_subplots_single_color_str(self, color):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(subplots=True, color=color)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=[color])

    @pytest.mark.parametrize("color", ["rgcby", list("rgcby")])
    def test_line_colors_and_styles_subplots_custom_colors(self, color):
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(color=color, subplots=True)
        for ax, c in zip(axes, list(color)):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_line_colors_and_styles_subplots_colormap_hex(self):
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # GH 10299
        custom_colors = ["#FF0000", "#0000FF", "#FFFF00", "#000000", "#FFFFFF"]
        axes = df.plot(color=custom_colors, subplots=True)
        for ax, c in zip(axes, list(custom_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize("cmap", ["jet", cm.jet])
    def test_line_colors_and_styles_subplots_colormap_subplot(self, cmap):
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        axes = df.plot(colormap=cmap, subplots=True)
        for ax, c in zip(axes, rgba_colors):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_line_colors_and_styles_subplots_single_col(self):
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # make color a list if plotting one column frame
        # handles cases like df.plot(color='DodgerBlue')
        axes = df.loc[:, [0]].plot(color="DodgerBlue", subplots=True)
        _check_colors(axes[0].lines, linecolors=["DodgerBlue"])

    def test_line_colors_and_styles_subplots_single_char(self):
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # single character style
        axes = df.plot(style="r", subplots=True)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=["r"])

    def test_line_colors_and_styles_subplots_list_styles(self):
        # GH 9894
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # list of styles
        styles = list("rgcby")
        axes = df.plot(style=styles, subplots=True)
        for ax, c in zip(axes, styles):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_area_colors(self):
        from matplotlib.collections import PolyCollection

        custom_colors = "rgcby"
        df = DataFrame(np.random.default_rng(2).random((5, 5)))

        ax = df.plot.area(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)
        poly = [o for o in ax.get_children() if isinstance(o, PolyCollection)]
        _check_colors(poly, facecolors=custom_colors)

        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, facecolors=custom_colors)

        for h in handles:
            assert h.get_alpha() is None

    def test_area_colors_poly(self):
        from matplotlib import cm
        from matplotlib.collections import PolyCollection

        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        ax = df.plot.area(colormap="jet")
        jet_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        _check_colors(ax.get_lines(), linecolors=jet_colors)
        poly = [o for o in ax.get_children() if isinstance(o, PolyCollection)]
        _check_colors(poly, facecolors=jet_colors)

        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, facecolors=jet_colors)
        for h in handles:
            assert h.get_alpha() is None

    def test_area_colors_stacked_false(self):
        from matplotlib import cm
        from matplotlib.collections import PolyCollection

        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        jet_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        # When stacked=False, alpha is set to 0.5
        ax = df.plot.area(colormap=cm.jet, stacked=False)
        _check_colors(ax.get_lines(), linecolors=jet_colors)
        poly = [o for o in ax.get_children() if isinstance(o, PolyCollection)]
        jet_with_alpha = [(c[0], c[1], c[2], 0.5) for c in jet_colors]
        _check_colors(poly, facecolors=jet_with_alpha)

        handles, _ = ax.get_legend_handles_labels()
        linecolors = jet_with_alpha
        _check_colors(handles[: len(jet_colors)], linecolors=linecolors)
        for h in handles:
            assert h.get_alpha() == 0.5

    def test_hist_colors(self):
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.hist()
        _check_colors(ax.patches[::10], facecolors=default_colors[:5])

    def test_hist_colors_single_custom(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors = "rgcby"
        ax = df.plot.hist(color=custom_colors)
        _check_colors(ax.patches[::10], facecolors=custom_colors)

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_hist_colors_cmap(self, colormap):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.hist(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        _check_colors(ax.patches[::10], facecolors=rgba_colors)

    def test_hist_colors_single_col(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot.hist(color="DodgerBlue")
        _check_colors([ax.patches[0]], facecolors=["DodgerBlue"])

    def test_hist_colors_single_color(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(kind="hist", color="green")
        _check_colors(ax.patches[::10], facecolors=["green"] * 5)

    def test_kde_colors(self):
        pytest.importorskip("scipy")
        custom_colors = "rgcby"
        df = DataFrame(np.random.default_rng(2).random((5, 5)))

        ax = df.plot.kde(color=custom_colors)
        _check_colors(ax.get_lines(), linecolors=custom_colors)

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_kde_colors_cmap(self, colormap):
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot.kde(colormap=colormap)
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        _check_colors(ax.get_lines(), linecolors=rgba_colors)

    def test_kde_colors_and_styles_subplots(self):
        pytest.importorskip("scipy")
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        axes = df.plot(kind="kde", subplots=True)
        for ax, c in zip(axes, list(default_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize("colormap", ["k", "red"])
    def test_kde_colors_and_styles_subplots_single_col_str(self, colormap):
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        axes = df.plot(kind="kde", color=colormap, subplots=True)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=[colormap])

    def test_kde_colors_and_styles_subplots_custom_color(self):
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        custom_colors = "rgcby"
        axes = df.plot(kind="kde", color=custom_colors, subplots=True)
        for ax, c in zip(axes, list(custom_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_kde_colors_and_styles_subplots_cmap(self, colormap):
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        axes = df.plot(kind="kde", colormap=colormap, subplots=True)
        for ax, c in zip(axes, rgba_colors):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_kde_colors_and_styles_subplots_single_col(self):
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # make color a list if plotting one column frame
        # handles cases like df.plot(color='DodgerBlue')
        axes = df.loc[:, [0]].plot(kind="kde", color="DodgerBlue", subplots=True)
        _check_colors(axes[0].lines, linecolors=["DodgerBlue"])

    def test_kde_colors_and_styles_subplots_single_char(self):
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # list of styles
        # single character style
        axes = df.plot(kind="kde", style="r", subplots=True)
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=["r"])

    def test_kde_colors_and_styles_subplots_list(self):
        pytest.importorskip("scipy")
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # list of styles
        styles = list("rgcby")
        axes = df.plot(kind="kde", style=styles, subplots=True)
        for ax, c in zip(axes, styles):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_boxplot_colors(self):
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        bp = df.plot.box(return_type="dict")
        _check_colors_box(
            bp,
            default_colors[0],
            default_colors[0],
            default_colors[2],
            default_colors[0],
        )

    def test_boxplot_colors_dict_colors(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        dict_colors = {
            "boxes": "#572923",
            "whiskers": "#982042",
            "medians": "#804823",
            "caps": "#123456",
        }
        bp = df.plot.box(color=dict_colors, sym="r+", return_type="dict")
        _check_colors_box(
            bp,
            dict_colors["boxes"],
            dict_colors["whiskers"],
            dict_colors["medians"],
            dict_colors["caps"],
            "r",
        )

    def test_boxplot_colors_default_color(self):
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # partial colors
        dict_colors = {"whiskers": "c", "medians": "m"}
        bp = df.plot.box(color=dict_colors, return_type="dict")
        _check_colors_box(bp, default_colors[0], "c", "m", default_colors[0])

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_boxplot_colors_cmap(self, colormap):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        bp = df.plot.box(colormap=colormap, return_type="dict")
        jet_colors = [cm.jet(n) for n in np.linspace(0, 1, 3)]
        _check_colors_box(
            bp, jet_colors[0], jet_colors[0], jet_colors[2], jet_colors[0]
        )

    def test_boxplot_colors_single(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # string color is applied to all artists except fliers
        bp = df.plot.box(color="DodgerBlue", return_type="dict")
        _check_colors_box(bp, "DodgerBlue", "DodgerBlue", "DodgerBlue", "DodgerBlue")

    def test_boxplot_colors_tuple(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # tuple is also applied to all artists except fliers
        bp = df.plot.box(color=(0, 1, 0), sym="#123456", return_type="dict")
        _check_colors_box(bp, (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), "#123456")

    def test_boxplot_colors_invalid(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        msg = re.escape(
            "color dict contains invalid key 'xxxx'. The key must be either "
            "['boxes', 'whiskers', 'medians', 'caps']"
        )
        with pytest.raises(ValueError, match=msg):
            # Color contains invalid key results in ValueError
            df.plot.box(color={"boxes": "red", "xxxx": "blue"})

    def test_default_color_cycle(self):
        import cycler

        colors = list("rgbk")
        plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", colors)

        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        ax = df.plot()

        expected = _unpack_cycler(plt.rcParams)[:3]
        _check_colors(ax.get_lines(), linecolors=expected)

    def test_no_color_bar(self):
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )
        ax = df.plot.hexbin(x="A", y="B", colorbar=None)
        assert ax.collections[0].colorbar is None

    def test_mixing_cmap_and_colormap_raises(self):
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )
        msg = "Only specify one of `cmap` and `colormap`"
        with pytest.raises(TypeError, match=msg):
            df.plot.hexbin(x="A", y="B", cmap="YlGn", colormap="BuGn")

    def test_passed_bar_colors(self):
        color_tuples = [(0.9, 0, 0, 1), (0, 0.9, 0, 1), (0, 0, 0.9, 1)]
        colormap = mpl.colors.ListedColormap(color_tuples)
        barplot = DataFrame([[1, 2, 3]]).plot(kind="bar", cmap=colormap)
        assert color_tuples == [c.get_facecolor() for c in barplot.patches]

    def test_rcParams_bar_colors(self):
        color_tuples = [(0.9, 0, 0, 1), (0, 0.9, 0, 1), (0, 0, 0.9, 1)]
        with mpl.rc_context(rc={"axes.prop_cycle": mpl.cycler("color", color_tuples)}):
            barplot = DataFrame([[1, 2, 3]]).plot(kind="bar")
        assert color_tuples == [c.get_facecolor() for c in barplot.patches]

    def test_colors_of_columns_with_same_name(self):
        # ISSUE 11136 -> https://github.com/pandas-dev/pandas/issues/11136
        # Creating a DataFrame with duplicate column labels and testing colors of them.
        df = DataFrame({"b": [0, 1, 0], "a": [1, 2, 3]})
        df1 = DataFrame({"a": [2, 4, 6]})
        df_concat = pd.concat([df, df1], axis=1)
        result = df_concat.plot()
        legend = result.get_legend()
        if Version(mpl.__version__) < Version("3.7"):
            handles = legend.legendHandles
        else:
            handles = legend.legend_handles
        for legend, line in zip(handles, result.lines):
            assert legend.get_color() == line.get_color()

    def test_invalid_colormap(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 2)), columns=["A", "B"]
        )
        msg = "(is not a valid value)|(is not a known colormap)"
        with pytest.raises((ValueError, KeyError), match=msg):
            df.plot(colormap="invalid_colormap")

    def test_dataframe_none_color(self):
        # GH51953
        df = DataFrame([[1, 2, 3]])
        ax = df.plot(color=None)
        expected = _unpack_cycler(mpl.pyplot.rcParams)[:3]
        _check_colors(ax.get_lines(), linecolors=expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_power.py ===
from sympy.core import (
    Basic, Rational, Symbol, S, Float, Integer, Mul, Number, Pow,
    Expr, I, nan, pi, symbols, oo, zoo, N)
from sympy.core.parameters import global_parameters
from sympy.core.tests.test_evalf import NS
from sympy.core.function import expand_multinomial
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.error_functions import erf
from sympy.functions.elementary.trigonometric import (
    sin, cos, tan, sec, csc, atan)
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.polys import Poly
from sympy.series.order import O
from sympy.sets import FiniteSet
from sympy.core.power import power
from sympy.core.intfunc import integer_nthroot
from sympy.testing.pytest import warns, _both_exp_pow
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.abc import a, b, c, x, y
from sympy.core.numbers import all_close

def test_rational():
    a = Rational(1, 5)

    r = sqrt(5)/5
    assert sqrt(a) == r
    assert 2*sqrt(a) == 2*r

    r = a*a**S.Half
    assert a**Rational(3, 2) == r
    assert 2*a**Rational(3, 2) == 2*r

    r = a**5*a**Rational(2, 3)
    assert a**Rational(17, 3) == r
    assert 2 * a**Rational(17, 3) == 2*r


def test_large_rational():
    e = (Rational(123712**12 - 1, 7) + Rational(1, 7))**Rational(1, 3)
    assert e == 234232585392159195136 * (Rational(1, 7)**Rational(1, 3))


def test_negative_real():
    def feq(a, b):
        return abs(a - b) < 1E-10

    assert feq(S.One / Float(-0.5), -Integer(2))


def test_expand():
    assert (2**(-1 - x)).expand() == S.Half*2**(-x)


def test_issue_3449():
    #test if powers are simplified correctly
    #see also issue 3995
    assert ((x**Rational(1, 3))**Rational(2)) == x**Rational(2, 3)
    assert (
        (x**Rational(3))**Rational(2, 5)) == (x**Rational(3))**Rational(2, 5)

    a = Symbol('a', real=True)
    b = Symbol('b', real=True)
    assert (a**2)**b == (abs(a)**b)**2
    assert sqrt(1/a) != 1/sqrt(a)  # e.g. for a = -1
    assert (a**3)**Rational(1, 3) != a
    assert (x**a)**b != x**(a*b)  # e.g. x = -1, a=2, b=1/2
    assert (x**.5)**b == x**(.5*b)
    assert (x**.5)**.5 == x**.25
    assert (x**2.5)**.5 != x**1.25  # e.g. for x = 5*I

    k = Symbol('k', integer=True)
    m = Symbol('m', integer=True)
    assert (x**k)**m == x**(k*m)
    assert Number(5)**Rational(2, 3) == Number(25)**Rational(1, 3)

    assert (x**.5)**2 == x**1.0
    assert (x**2)**k == (x**k)**2 == x**(2*k)

    a = Symbol('a', positive=True)
    assert (a**3)**Rational(2, 5) == a**Rational(6, 5)
    assert (a**2)**b == (a**b)**2
    assert (a**Rational(2, 3))**x == a**(x*Rational(2, 3)) != (a**x)**Rational(2, 3)


def test_issue_3866():
    assert --sqrt(sqrt(5) - 1) == sqrt(sqrt(5) - 1)


def test_negative_one():
    x = Symbol('x', complex=True)
    y = Symbol('y', complex=True)
    assert 1/x**y == x**(-y)


def test_issue_4362():
    neg = Symbol('neg', negative=True)
    nonneg = Symbol('nonneg', nonnegative=True)
    any = Symbol('any')
    num, den = sqrt(1/neg).as_numer_denom()
    assert num == sqrt(-1)
    assert den == sqrt(-neg)
    num, den = sqrt(1/nonneg).as_numer_denom()
    assert num == 1
    assert den == sqrt(nonneg)
    num, den = sqrt(1/any).as_numer_denom()
    assert num == sqrt(1/any)
    assert den == 1

    def eqn(num, den, pow):
        return (num/den)**pow
    npos = 1
    nneg = -1
    dpos = 2 - sqrt(3)
    dneg = 1 - sqrt(3)
    assert dpos > 0 and dneg < 0 and npos > 0 and nneg < 0
    # pos or neg integer
    eq = eqn(npos, dpos, 2)
    assert eq.is_Pow and eq.as_numer_denom() == (1, dpos**2)
    eq = eqn(npos, dneg, 2)
    assert eq.is_Pow and eq.as_numer_denom() == (1, dneg**2)
    eq = eqn(nneg, dpos, 2)
    assert eq.is_Pow and eq.as_numer_denom() == (1, dpos**2)
    eq = eqn(nneg, dneg, 2)
    assert eq.is_Pow and eq.as_numer_denom() == (1, dneg**2)
    eq = eqn(npos, dpos, -2)
    assert eq.is_Pow and eq.as_numer_denom() == (dpos**2, 1)
    eq = eqn(npos, dneg, -2)
    assert eq.is_Pow and eq.as_numer_denom() == (dneg**2, 1)
    eq = eqn(nneg, dpos, -2)
    assert eq.is_Pow and eq.as_numer_denom() == (dpos**2, 1)
    eq = eqn(nneg, dneg, -2)
    assert eq.is_Pow and eq.as_numer_denom() == (dneg**2, 1)
    # pos or neg rational
    pow = S.Half
    eq = eqn(npos, dpos, pow)
    assert eq.is_Pow and eq.as_numer_denom() == (npos**pow, dpos**pow)
    eq = eqn(npos, dneg, pow)
    assert eq.is_Pow is False and eq.as_numer_denom() == ((-npos)**pow, (-dneg)**pow)
    eq = eqn(nneg, dpos, pow)
    assert not eq.is_Pow or eq.as_numer_denom() == (nneg**pow, dpos**pow)
    eq = eqn(nneg, dneg, pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-nneg)**pow, (-dneg)**pow)
    eq = eqn(npos, dpos, -pow)
    assert eq.is_Pow and eq.as_numer_denom() == (dpos**pow, npos**pow)
    eq = eqn(npos, dneg, -pow)
    assert eq.is_Pow is False and eq.as_numer_denom() == (-(-npos)**pow*(-dneg)**pow, npos)
    eq = eqn(nneg, dpos, -pow)
    assert not eq.is_Pow or eq.as_numer_denom() == (dpos**pow, nneg**pow)
    eq = eqn(nneg, dneg, -pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-dneg)**pow, (-nneg)**pow)
    # unknown exponent
    pow = 2*any
    eq = eqn(npos, dpos, pow)
    assert eq.is_Pow and eq.as_numer_denom() == (npos**pow, dpos**pow)
    eq = eqn(npos, dneg, pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-npos)**pow, (-dneg)**pow)
    eq = eqn(nneg, dpos, pow)
    assert eq.is_Pow and eq.as_numer_denom() == (nneg**pow, dpos**pow)
    eq = eqn(nneg, dneg, pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-nneg)**pow, (-dneg)**pow)
    eq = eqn(npos, dpos, -pow)
    assert eq.as_numer_denom() == (dpos**pow, npos**pow)
    eq = eqn(npos, dneg, -pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-dneg)**pow, (-npos)**pow)
    eq = eqn(nneg, dpos, -pow)
    assert eq.is_Pow and eq.as_numer_denom() == (dpos**pow, nneg**pow)
    eq = eqn(nneg, dneg, -pow)
    assert eq.is_Pow and eq.as_numer_denom() == ((-dneg)**pow, (-nneg)**pow)

    assert ((1/(1 + x/3))**(-S.One)).as_numer_denom() == (3 + x, 3)
    notp = Symbol('notp', positive=False)  # not positive does not imply real
    b = ((1 + x/notp)**-2)
    assert (b**(-y)).as_numer_denom() == (1, b**y)
    assert (b**(-S.One)).as_numer_denom() == ((notp + x)**2, notp**2)
    nonp = Symbol('nonp', nonpositive=True)
    assert (((1 + x/nonp)**-2)**(-S.One)).as_numer_denom() == ((-nonp -
            x)**2, nonp**2)

    n = Symbol('n', negative=True)
    assert (x**n).as_numer_denom() == (1, x**-n)
    assert sqrt(1/n).as_numer_denom() == (S.ImaginaryUnit, sqrt(-n))
    n = Symbol('0 or neg', nonpositive=True)
    # if x and n are split up without negating each term and n is negative
    # then the answer might be wrong; if n is 0 it won't matter since
    # 1/oo and 1/zoo are both zero as is sqrt(0)/sqrt(-x) unless x is also
    # zero (in which case the negative sign doesn't matter):
    # 1/sqrt(1/-1) = -I but sqrt(-1)/sqrt(1) = I
    assert (1/sqrt(x/n)).as_numer_denom() == (sqrt(-n), sqrt(-x))
    c = Symbol('c', complex=True)
    e = sqrt(1/c)
    assert e.as_numer_denom() == (e, 1)
    i = Symbol('i', integer=True)
    assert ((1 + x/y)**i).as_numer_denom() == ((x + y)**i, y**i)


def test_Pow_Expr_args():
    bases = [Basic(), Poly(x, x), FiniteSet(x)]
    for base in bases:
        # The cache can mess with the stacklevel test
        with warns(SymPyDeprecationWarning, test_stacklevel=False):
            Pow(base, S.One)


def test_Pow_signs():
    """Cf. issues 4595 and 5250"""
    n = Symbol('n', even=True)
    assert (3 - y)**2 != (y - 3)**2
    assert (3 - y)**n != (y - 3)**n
    assert (-3 + y - x)**2 != (3 - y + x)**2
    assert (y - 3)**3 != -(3 - y)**3


def test_power_with_noncommutative_mul_as_base():
    x = Symbol('x', commutative=False)
    y = Symbol('y', commutative=False)
    assert not (x*y)**3 == x**3*y**3
    assert (2*x*y)**3 == 8*(x*y)**3


@_both_exp_pow
def test_power_rewrite_exp():
    assert (I**I).rewrite(exp) == exp(-pi/2)

    expr = (2 + 3*I)**(4 + 5*I)
    assert expr.rewrite(exp) == exp((4 + 5*I)*(log(sqrt(13)) + I*atan(Rational(3, 2))))
    assert expr.rewrite(exp).expand() == \
        169*exp(5*I*log(13)/2)*exp(4*I*atan(Rational(3, 2)))*exp(-5*atan(Rational(3, 2)))

    assert ((6 + 7*I)**5).rewrite(exp) == 7225*sqrt(85)*exp(5*I*atan(Rational(7, 6)))

    expr = 5**(6 + 7*I)
    assert expr.rewrite(exp) == exp((6 + 7*I)*log(5))
    assert expr.rewrite(exp).expand() == 15625*exp(7*I*log(5))

    assert Pow(123, 789, evaluate=False).rewrite(exp) == 123**789
    assert (1**I).rewrite(exp) == 1**I
    assert (0**I).rewrite(exp) == 0**I

    expr = (-2)**(2 + 5*I)
    assert expr.rewrite(exp) == exp((2 + 5*I)*(log(2) + I*pi))
    assert expr.rewrite(exp).expand() == 4*exp(-5*pi)*exp(5*I*log(2))

    assert ((-2)**S(-5)).rewrite(exp) == (-2)**S(-5)

    x, y = symbols('x y')
    assert (x**y).rewrite(exp) == exp(y*log(x))
    if global_parameters.exp_is_pow:
        assert (7**x).rewrite(exp) == Pow(S.Exp1, x*log(7), evaluate=False)
    else:
        assert (7**x).rewrite(exp) == exp(x*log(7), evaluate=False)
    assert ((2 + 3*I)**x).rewrite(exp) == exp(x*(log(sqrt(13)) + I*atan(Rational(3, 2))))
    assert (y**(5 + 6*I)).rewrite(exp) == exp(log(y)*(5 + 6*I))

    assert all((1/func(x)).rewrite(exp) == 1/(func(x).rewrite(exp)) for func in
                    (sin, cos, tan, sec, csc, sinh, cosh, tanh))


def test_zero():
    assert 0**x != 0
    assert 0**(2*x) == 0**x
    assert 0**(1.0*x) == 0**x
    assert 0**(2.0*x) == 0**x
    assert (0**(2 - x)).as_base_exp() == (0, 2 - x)
    assert 0**(x - 2) != S.Infinity**(2 - x)
    assert 0**(2*x*y) == 0**(x*y)
    assert 0**(-2*x*y) == S.ComplexInfinity**(x*y)
    assert Float(0)**2 is not S.Zero
    assert Float(0)**2 == 0.0
    assert Float(0)**-2 is zoo
    assert Float(0)**oo is S.Zero

    #Test issue 19572
    assert 0 ** -oo is zoo
    assert power(0, -oo) is zoo
    assert Float(0)**-oo is zoo

def test_pow_as_base_exp():
    assert (S.Infinity**(2 - x)).as_base_exp() == (S.Infinity, 2 - x)
    assert (S.Infinity**(x - 2)).as_base_exp() == (S.Infinity, x - 2)
    p = S.Half**x
    assert p.base, p.exp == p.as_base_exp() == (S(2), -x)
    p = (S(3)/2)**x
    assert p.base, p.exp == p.as_base_exp() == (3*S.Half, x)
    p = (S(2)/3)**x
    assert p.as_base_exp() == (S(2)/3, x)
    assert p.base, p.exp == (S(2)/3, x)
    # issue 8344:
    assert Pow(1, 2, evaluate=False).as_base_exp() == (S.One, S(2))


def test_nseries():
    assert sqrt(I*x - 1)._eval_nseries(x, 4, None, 1) == I + x/2 + I*x**2/8 - x**3/16 + O(x**4)
    assert sqrt(I*x - 1)._eval_nseries(x, 4, None, -1) == -I - x/2 - I*x**2/8 + x**3/16 + O(x**4)
    assert cbrt(I*x - 1)._eval_nseries(x, 4, None, 1) == (-1)**(S(1)/3) - (-1)**(S(5)/6)*x/3 + \
    (-1)**(S(1)/3)*x**2/9 + 5*(-1)**(S(5)/6)*x**3/81 + O(x**4)
    assert cbrt(I*x - 1)._eval_nseries(x, 4, None, -1) == -(-1)**(S(2)/3) - (-1)**(S(1)/6)*x/3 - \
    (-1)**(S(2)/3)*x**2/9 + 5*(-1)**(S(1)/6)*x**3/81 + O(x**4)
    assert (1 / (exp(-1/x) + 1/x))._eval_nseries(x, 2, None) == x + O(x**2)
    # test issue 23752
    assert sqrt(-I*x**2 + x - 3)._eval_nseries(x, 4, None, 1) == -sqrt(3)*I + sqrt(3)*I*x/6 - \
    sqrt(3)*I*x**2*(-S(1)/72 + I/6) - sqrt(3)*I*x**3*(-S(1)/432 + I/36) + O(x**4)
    assert sqrt(-I*x**2 + x - 3)._eval_nseries(x, 4, None, -1) == -sqrt(3)*I + sqrt(3)*I*x/6 - \
    sqrt(3)*I*x**2*(-S(1)/72 + I/6) - sqrt(3)*I*x**3*(-S(1)/432 + I/36) + O(x**4)
    assert cbrt(-I*x**2 + x - 3)._eval_nseries(x, 4, None, 1) == -(-1)**(S(2)/3)*3**(S(1)/3) + \
    (-1)**(S(2)/3)*3**(S(1)/3)*x/9 - (-1)**(S(2)/3)*3**(S(1)/3)*x**2*(-S(1)/81 + I/9) - \
    (-1)**(S(2)/3)*3**(S(1)/3)*x**3*(-S(5)/2187 + 2*I/81) + O(x**4)
    assert cbrt(-I*x**2 + x - 3)._eval_nseries(x, 4, None, -1) == -(-1)**(S(2)/3)*3**(S(1)/3) + \
    (-1)**(S(2)/3)*3**(S(1)/3)*x/9 - (-1)**(S(2)/3)*3**(S(1)/3)*x**2*(-S(1)/81 + I/9) - \
    (-1)**(S(2)/3)*3**(S(1)/3)*x**3*(-S(5)/2187 + 2*I/81) + O(x**4)


def test_issue_6100_12942_4473():
    assert x**1.0 != x
    assert x != x**1.0
    assert True != x**1.0
    assert x**1.0 is not True
    assert x is not True
    assert x*y != (x*y)**1.0
    # Pow != Symbol
    assert (x**1.0)**1.0 != x
    assert (x**1.0)**2.0 != x**2
    b = Expr()
    assert Pow(b, 1.0, evaluate=False) != b
    # if the following gets distributed as a Mul (x**1.0*y**1.0 then
    # __eq__ methods could be added to Symbol and Pow to detect the
    # power-of-1.0 case.
    assert ((x*y)**1.0).func is Pow


def test_issue_6208():
    from sympy.functions.elementary.miscellaneous import root
    assert sqrt(33**(I*9/10)) == -33**(I*9/20)
    assert root((6*I)**(2*I), 3).as_base_exp()[1] == Rational(1, 3)  # != 2*I/3
    assert root((6*I)**(I/3), 3).as_base_exp()[1] == I/9
    assert sqrt(exp(3*I)) == exp(3*I/2)
    assert sqrt(-sqrt(3)*(1 + 2*I)) == sqrt(sqrt(3))*sqrt(-1 - 2*I)
    assert sqrt(exp(5*I)) == -exp(5*I/2)
    assert root(exp(5*I), 3).exp == Rational(1, 3)


def test_issue_6990():
    assert (sqrt(a + b*x + x**2)).series(x, 0, 3).removeO() == \
        sqrt(a)*x**2*(1/(2*a) - b**2/(8*a**2)) + sqrt(a) + b*x/(2*sqrt(a))


def test_issue_6068():
    assert sqrt(sin(x)).series(x, 0, 7) == \
        sqrt(x) - x**Rational(5, 2)/12 + x**Rational(9, 2)/1440 - \
        x**Rational(13, 2)/24192 + O(x**7)
    assert sqrt(sin(x)).series(x, 0, 9) == \
        sqrt(x) - x**Rational(5, 2)/12 + x**Rational(9, 2)/1440 - \
        x**Rational(13, 2)/24192 - 67*x**Rational(17, 2)/29030400 + O(x**9)
    assert sqrt(sin(x**3)).series(x, 0, 19) == \
        x**Rational(3, 2) - x**Rational(15, 2)/12 + x**Rational(27, 2)/1440 + O(x**19)
    assert sqrt(sin(x**3)).series(x, 0, 20) == \
        x**Rational(3, 2) - x**Rational(15, 2)/12 + x**Rational(27, 2)/1440 - \
        x**Rational(39, 2)/24192 + O(x**20)


def test_issue_6782():
    assert sqrt(sin(x**3)).series(x, 0, 7) == x**Rational(3, 2) + O(x**7)
    assert sqrt(sin(x**4)).series(x, 0, 3) == x**2 + O(x**3)


def test_issue_6653():
    assert (1 / sqrt(1 + sin(x**2))).series(x, 0, 3) == 1 - x**2/2 + O(x**3)


def test_issue_6429():
    f = (c**2 + x)**(0.5)
    assert f.series(x, x0=0, n=1) == (c**2)**0.5 + O(x)
    assert f.taylor_term(0, x) == (c**2)**0.5
    assert f.taylor_term(1, x) == 0.5*x*(c**2)**(-0.5)
    assert f.taylor_term(2, x) == -0.125*x**2*(c**2)**(-1.5)


def test_issue_7638():
    f = pi/log(sqrt(2))
    assert ((1 + I)**(I*f/2))**0.3 == (1 + I)**(0.15*I*f)
    # if 1/3 -> 1.0/3 this should fail since it cannot be shown that the
    # sign will be +/-1; for the previous "small arg" case, it didn't matter
    # that this could not be proved
    assert (1 + I)**(4*I*f) == ((1 + I)**(12*I*f))**Rational(1, 3)

    assert (((1 + I)**(I*(1 + 7*f)))**Rational(1, 3)).exp == Rational(1, 3)
    r = symbols('r', real=True)
    assert sqrt(r**2) == abs(r)
    assert cbrt(r**3) != r
    assert sqrt(Pow(2*I, 5*S.Half)) != (2*I)**Rational(5, 4)
    p = symbols('p', positive=True)
    assert cbrt(p**2) == p**Rational(2, 3)
    assert NS(((0.2 + 0.7*I)**(0.7 + 1.0*I))**(0.5 - 0.1*I), 1) == '0.4 + 0.2*I'
    assert sqrt(1/(1 + I)) == sqrt(1 - I)/sqrt(2)  # or 1/sqrt(1 + I)
    e = 1/(1 - sqrt(2))
    assert sqrt(e) == I/sqrt(-1 + sqrt(2))
    assert e**Rational(-1, 2) == -I*sqrt(-1 + sqrt(2))
    assert sqrt((cos(1)**2 + sin(1)**2 - 1)**(3 + I)).exp in [S.Half,
                                                              Rational(3, 2) + I/2]
    assert sqrt(r**Rational(4, 3)) != r**Rational(2, 3)
    assert sqrt((p + I)**Rational(4, 3)) == (p + I)**Rational(2, 3)

    for q in 1+I, 1-I:
        assert sqrt(q**2) == q
    for q in -1+I, -1-I:
        assert sqrt(q**2) == -q

    assert sqrt((p + r*I)**2) != p + r*I
    e = (1 + I/5)
    assert sqrt(e**5) == e**(5*S.Half)
    assert sqrt(e**6) == e**3
    assert sqrt((1 + I*r)**6) != (1 + I*r)**3


def test_issue_8582():
    assert 1**oo is nan
    assert 1**(-oo) is nan
    assert 1**zoo is nan
    assert 1**(oo + I) is nan
    assert 1**(1 + I*oo) is nan
    assert 1**(oo + I*oo) is nan


def test_issue_8650():
    n = Symbol('n', integer=True, nonnegative=True)
    assert (n**n).is_positive is True
    x = 5*n + 5
    assert (x**(5*(n + 1))).is_positive is True


def test_issue_13914():
    b = Symbol('b')
    assert (-1)**zoo is nan
    assert 2**zoo is nan
    assert (S.Half)**(1 + zoo) is nan
    assert I**(zoo + I) is nan
    assert b**(I + zoo) is nan


def test_better_sqrt():
    n = Symbol('n', integer=True, nonnegative=True)
    assert sqrt(3 + 4*I) == 2 + I
    assert sqrt(3 - 4*I) == 2 - I
    assert sqrt(-3 - 4*I) == 1 - 2*I
    assert sqrt(-3 + 4*I) == 1 + 2*I
    assert sqrt(32 + 24*I) == 6 + 2*I
    assert sqrt(32 - 24*I) == 6 - 2*I
    assert sqrt(-32 - 24*I) == 2 - 6*I
    assert sqrt(-32 + 24*I) == 2 + 6*I

    # triple (3, 4, 5):
    # parity of 3 matches parity of 5 and
    # den, 4, is a square
    assert sqrt((3 + 4*I)/4) == 1 + I/2
    # triple (8, 15, 17)
    # parity of 8 doesn't match parity of 17 but
    # den/2, 8/2, is a square
    assert sqrt((8 + 15*I)/8) == (5 + 3*I)/4
    # handle the denominator
    assert sqrt((3 - 4*I)/25) == (2 - I)/5
    assert sqrt((3 - 4*I)/26) == (2 - I)/sqrt(26)
    # mul
    #  issue #12739
    assert sqrt((3 + 4*I)/(3 - 4*I)) == (3 + 4*I)/5
    assert sqrt(2/(3 + 4*I)) == sqrt(2)/5*(2 - I)
    assert sqrt(n/(3 + 4*I)).subs(n, 2) == sqrt(2)/5*(2 - I)
    assert sqrt(-2/(3 + 4*I)) == sqrt(2)/5*(1 + 2*I)
    assert sqrt(-n/(3 + 4*I)).subs(n, 2) == sqrt(2)/5*(1 + 2*I)
    # power
    assert sqrt(1/(3 + I*4)) == (2 - I)/5
    assert sqrt(1/(3 - I)) == sqrt(10)*sqrt(3 + I)/10
    # symbolic
    i = symbols('i', imaginary=True)
    assert sqrt(3/i) == Mul(sqrt(3), 1/sqrt(i), evaluate=False)
    # multiples of 1/2; don't make this too automatic
    assert sqrt(3 + 4*I)**3 == (2 + I)**3
    assert Pow(3 + 4*I, Rational(3, 2)) == 2 + 11*I
    assert Pow(6 + 8*I, Rational(3, 2)) == 2*sqrt(2)*(2 + 11*I)
    n, d = (3 + 4*I), (3 - 4*I)**3
    a = n/d
    assert a.args == (1/d, n)
    eq = sqrt(a)
    assert eq.args == (a, S.Half)
    assert expand_multinomial(eq) == sqrt((-117 + 44*I)*(3 + 4*I))/125
    assert eq.expand() == (7 - 24*I)/125

    # issue 12775
    # pos im part
    assert sqrt(2*I) == (1 + I)
    assert sqrt(2*9*I) == Mul(3, 1 + I, evaluate=False)
    assert Pow(2*I, 3*S.Half) == (1 + I)**3
    # neg im part
    assert sqrt(-I/2) == Mul(S.Half, 1 - I, evaluate=False)
    # fractional im part
    assert Pow(Rational(-9, 2)*I, Rational(3, 2)) == 27*(1 - I)**3/8


def test_issue_2993():
    assert str((2.3*x - 4)**0.3) == '(2.3*x - 4)**0.3'
    assert str((2.3*x + 4)**0.3) == '(2.3*x + 4)**0.3'
    assert str((-2.3*x + 4)**0.3) == '(4 - 2.3*x)**0.3'
    assert str((-2.3*x - 4)**0.3) == '(-2.3*x - 4)**0.3'
    assert str((2.3*x - 2)**0.3) == '(2.3*x - 2)**0.3'
    assert str((-2.3*x - 2)**0.3) == '(-2.3*x - 2)**0.3'
    assert str((-2.3*x + 2)**0.3) == '(2 - 2.3*x)**0.3'
    assert str((2.3*x + 2)**0.3) == '(2.3*x + 2)**0.3'
    assert str((2.3*x - 4)**Rational(1, 3)) == '(2.3*x - 4)**(1/3)'
    eq = (2.3*x + 4)
    assert str(eq**2) == '(2.3*x + 4)**2'
    assert (1/eq).args == (eq, -1)  # don't change trivial power
    # issue 17735
    q=.5*exp(x) - .5*exp(-x) + 0.1
    assert int((q**2).subs(x, 1)) == 1
    # issue 17756
    y = Symbol('y')
    assert len(sqrt(x/(x + y)**2 + Float('0.008', 30)).subs(y, pi.n(25)).atoms(Float)) == 2
    # issue 17756
    a, b, c, d, e, f, g = symbols('a:g')
    expr = sqrt(1 + a*(c**4 + g*d - 2*g*e - f*(-g + d))**2/
        (c**3*b**2*(d - 3*e + 2*f)**2))/2
    r = [
    (a, N('0.0170992456333788667034850458615', 30)),
    (b, N('0.0966594956075474769169134801223', 30)),
    (c, N('0.390911862903463913632151616184', 30)),
    (d, N('0.152812084558656566271750185933', 30)),
    (e, N('0.137562344465103337106561623432', 30)),
    (f, N('0.174259178881496659302933610355', 30)),
    (g, N('0.220745448491223779615401870086', 30))]
    tru = expr.n(30, subs=dict(r))
    seq = expr.subs(r)
    # although `tru` is the right way to evaluate
    # expr with numerical values, `seq` will have
    # significant loss of precision if extraction of
    # the largest coefficient of a power's base's terms
    # is done improperly
    assert seq == tru

def test_issue_17450():
    assert (erf(cosh(1)**7)**I).is_real is None
    assert (erf(cosh(1)**7)**I).is_imaginary is False
    assert (Pow(exp(1+sqrt(2)), ((1-sqrt(2))*I*pi), evaluate=False)).is_real is None
    assert ((-10)**(10*I*pi/3)).is_real is False
    assert ((-5)**(4*I*pi)).is_real is False


def test_issue_18190():
    assert sqrt(1 / tan(1 + I)) == 1 / sqrt(tan(1 + I))


def test_issue_14815():
    x = Symbol('x', real=True)
    assert sqrt(x).is_extended_negative is False
    x = Symbol('x', real=False)
    assert sqrt(x).is_extended_negative is None
    x = Symbol('x', complex=True)
    assert sqrt(x).is_extended_negative is False
    x = Symbol('x', extended_real=True)
    assert sqrt(x).is_extended_negative is False
    assert sqrt(zoo, evaluate=False).is_extended_negative is False
    assert sqrt(nan, evaluate=False).is_extended_negative is None


def test_issue_18509():
    x = Symbol('x', prime=True)
    assert x**oo is oo
    assert (1/x)**oo is S.Zero
    assert (-1/x)**oo is S.Zero
    assert (-x)**oo is zoo
    assert (-oo)**(-1 + I) is S.Zero
    assert (-oo)**(1 + I) is zoo
    assert (oo)**(-1 + I) is S.Zero
    assert (oo)**(1 + I) is zoo


def test_issue_18762():
    e, p = symbols('e p')
    g0 = sqrt(1 + e**2 - 2*e*cos(p))
    assert len(g0.series(e, 1, 3).args) == 4


def test_issue_21860():
    e = 3*2**Rational(66666666667,200000000000)*3**Rational(16666666667,50000000000)*x**Rational(66666666667, 200000000000)
    ans = Mul(Rational(3, 2),
              Pow(Integer(2), Rational(33333333333, 100000000000)),
              Pow(Integer(3), Rational(26666666667, 40000000000)))
    assert e.xreplace({x: Rational(3,8)}) == ans


def test_issue_21647():
    e = log((Integer(567)/500)**(811*(Integer(567)/500)**x/100))
    ans = log(Mul(Rational(64701150190720499096094005280169087619821081527,
                           76293945312500000000000000000000000000000000000),
                  Pow(Integer(2), Rational(396204892125479941, 781250000000000000)),
                  Pow(Integer(3), Rational(385045107874520059, 390625000000000000)),
                  Pow(Integer(5), Rational(407364676376439823, 1562500000000000000)),
                  Pow(Integer(7), Rational(385045107874520059, 1562500000000000000))))
    assert e.xreplace({x: 6}) == ans


def test_issue_21762():
    e = (x**2 + 6)**(Integer(33333333333333333)/50000000000000000)
    ans = Mul(Rational(5, 4),
              Pow(Integer(2), Rational(16666666666666667, 25000000000000000)),
              Pow(Integer(5), Rational(8333333333333333, 25000000000000000)))
    assert e.xreplace({x: S.Half}) == ans


def test_issue_14704():
    a = 144**144
    x, xexact = integer_nthroot(a,a)
    assert x == 1 and xexact is False


def test_rational_powers_larger_than_one():
    assert Rational(2, 3)**Rational(3, 2) == 2*sqrt(6)/9
    assert Rational(1, 6)**Rational(9, 4) == 6**Rational(3, 4)/216
    assert Rational(3, 7)**Rational(7, 3) == 9*3**Rational(1, 3)*7**Rational(2, 3)/343


def test_power_dispatcher():

    class NewBase(Expr):
        pass
    class NewPow(NewBase, Pow):
        pass
    a, b = Symbol('a'), NewBase()

    @power.register(Expr, NewBase)
    @power.register(NewBase, Expr)
    @power.register(NewBase, NewBase)
    def _(a, b):
        return NewPow(a, b)

    # Pow called as fallback
    assert power(2, 3) == 8*S.One
    assert power(a, 2) == Pow(a, 2)
    assert power(a, a) == Pow(a, a)

    # NewPow called by dispatch
    assert power(a, b) == NewPow(a, b)
    assert power(b, a) == NewPow(b, a)
    assert power(b, b) == NewPow(b, b)


def test_powers_of_I():
    assert [sqrt(I)**i for i in range(13)] == [
        1, sqrt(I), I, sqrt(I)**3, -1, -sqrt(I), -I, -sqrt(I)**3,
        1, sqrt(I), I, sqrt(I)**3, -1]
    assert sqrt(I)**(S(9)/2) == -I**(S(1)/4)


def test_issue_23918():
    b = S(2)/3
    assert (b**x).as_base_exp() == (b, x)


def test_issue_26546():
    x = Symbol('x', real=True)
    assert x.is_extended_real is True
    assert sqrt(x+I).is_extended_real is False
    assert Pow(x+I, S.Half).is_extended_real is False
    assert Pow(x+I, Rational(1,2)).is_extended_real is False
    assert Pow(x+I, Rational(1,13)).is_extended_real is False
    assert Pow(x+I, Rational(2,3)).is_extended_real is None


def test_issue_25165():
    e1 = (1/sqrt(( - x + 1)**2 + (x - 0.23)**4)).series(x, 0, 2)
    e2 = 0.998603724830355 + 1.02004923189934*x + O(x**2)
    assert all_close(e1, e2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\tests\test_polynomial.py ===
"""Tests for polynomial module.

"""
import pickle
from copy import deepcopy
from fractions import Fraction
from functools import reduce

import numpy as np
import numpy.polynomial.polynomial as poly
import numpy.polynomial.polyutils as pu
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
)


def trim(x):
    return poly.polytrim(x, tol=1e-6)


T0 = [1]
T1 = [0, 1]
T2 = [-1, 0, 2]
T3 = [0, -3, 0, 4]
T4 = [1, 0, -8, 0, 8]
T5 = [0, 5, 0, -20, 0, 16]
T6 = [-1, 0, 18, 0, -48, 0, 32]
T7 = [0, -7, 0, 56, 0, -112, 0, 64]
T8 = [1, 0, -32, 0, 160, 0, -256, 0, 128]
T9 = [0, 9, 0, -120, 0, 432, 0, -576, 0, 256]

Tlist = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]


class TestConstants:

    def test_polydomain(self):
        assert_equal(poly.polydomain, [-1, 1])

    def test_polyzero(self):
        assert_equal(poly.polyzero, [0])

    def test_polyone(self):
        assert_equal(poly.polyone, [1])

    def test_polyx(self):
        assert_equal(poly.polyx, [0, 1])

    def test_copy(self):
        x = poly.Polynomial([1, 2, 3])
        y = deepcopy(x)
        assert_equal(x, y)

    def test_pickle(self):
        x = poly.Polynomial([1, 2, 3])
        y = pickle.loads(pickle.dumps(x))
        assert_equal(x, y)

class TestArithmetic:

    def test_polyadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = poly.polyadd([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polysub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = poly.polysub([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polymulx(self):
        assert_equal(poly.polymulx([0]), [0])
        assert_equal(poly.polymulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i + 1) + [1]
            assert_equal(poly.polymulx(ser), tgt)

    def test_polymul(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(i + j + 1)
                tgt[i + j] += 1
                res = poly.polymul([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polydiv(self):
        # check zero division
        assert_raises(ZeroDivisionError, poly.polydiv, [1], [0])

        # check scalar division
        quo, rem = poly.polydiv([2], [2])
        assert_equal((quo, rem), (1, 0))
        quo, rem = poly.polydiv([2, 2], [2])
        assert_equal((quo, rem), ((1, 1), 0))

        # check rest.
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0] * i + [1, 2]
                cj = [0] * j + [1, 2]
                tgt = poly.polyadd(ci, cj)
                quo, rem = poly.polydiv(tgt, ci)
                res = poly.polyadd(poly.polymul(quo, ci), rem)
                assert_equal(res, tgt, err_msg=msg)

    def test_polypow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(poly.polymul, [c] * j, np.array([1]))
                res = poly.polypow(c, j)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

class TestFraction:

    def test_Fraction(self):
        # assert we can use Polynomials with coefficients of object dtype
        f = Fraction(2, 3)
        one = Fraction(1, 1)
        zero = Fraction(0, 1)
        p = poly.Polynomial([f, f], domain=[zero, one], window=[zero, one])

        x = 2 * p + p ** 2
        assert_equal(x.coef, np.array([Fraction(16, 9), Fraction(20, 9),
                                       Fraction(4, 9)], dtype=object))
        assert_equal(p.domain, [zero, one])
        assert_equal(p.coef.dtype, np.dtypes.ObjectDType())
        assert_(isinstance(p(f), Fraction))
        assert_equal(p(f), Fraction(10, 9))
        p_deriv = poly.Polynomial([Fraction(2, 3)], domain=[zero, one],
                                  window=[zero, one])
        assert_equal(p.deriv(), p_deriv)

class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([1., 2., 3.])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1
    y = poly.polyval(x, [1., 2., 3.])

    def test_polyval(self):
        # check empty input
        assert_equal(poly.polyval([], [1]).size, 0)

        # check normal input)
        x = np.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(5):
            tgt = y[i]
            res = poly.polyval(x, [0] * i + [1])
            assert_almost_equal(res, tgt)
        tgt = x * (x**2 - 1)
        res = poly.polyval(x, [0, -1, 0, 1])
        assert_almost_equal(res, tgt)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = np.zeros(dims)
            assert_equal(poly.polyval(x, [1]).shape, dims)
            assert_equal(poly.polyval(x, [1, 0]).shape, dims)
            assert_equal(poly.polyval(x, [1, 0, 0]).shape, dims)

        # check masked arrays are processed correctly
        mask = [False, True, False]
        mx = np.ma.array([1, 2, 3], mask=mask)
        res = np.polyval([7, 5, 3], mx)
        assert_array_equal(res.mask, mask)

        # check subtypes of ndarray are preserved
        class C(np.ndarray):
            pass

        cx = np.array([1, 2, 3]).view(C)
        assert_equal(type(np.polyval([2, 3, 4], cx)), C)

    def test_polyvalfromroots(self):
        # check exception for broadcasting x values over root array with
        # too few dimensions
        assert_raises(ValueError, poly.polyvalfromroots,
                      [1], [1], tensor=False)

        # check empty input
        assert_equal(poly.polyvalfromroots([], [1]).size, 0)
        assert_(poly.polyvalfromroots([], [1]).shape == (0,))

        # check empty input + multidimensional roots
        assert_equal(poly.polyvalfromroots([], [[1] * 5]).size, 0)
        assert_(poly.polyvalfromroots([], [[1] * 5]).shape == (5, 0))

        # check scalar input
        assert_equal(poly.polyvalfromroots(1, 1), 0)
        assert_(poly.polyvalfromroots(1, np.ones((3, 3))).shape == (3,))

        # check normal input)
        x = np.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(1, 5):
            tgt = y[i]
            res = poly.polyvalfromroots(x, [0] * i)
            assert_almost_equal(res, tgt)
        tgt = x * (x - 1) * (x + 1)
        res = poly.polyvalfromroots(x, [-1, 0, 1])
        assert_almost_equal(res, tgt)

        # check that shape is preserved
        for i in range(3):
            dims = [2] * i
            x = np.zeros(dims)
            assert_equal(poly.polyvalfromroots(x, [1]).shape, dims)
            assert_equal(poly.polyvalfromroots(x, [1, 0]).shape, dims)
            assert_equal(poly.polyvalfromroots(x, [1, 0, 0]).shape, dims)

        # check compatibility with factorization
        ptest = [15, 2, -16, -2, 1]
        r = poly.polyroots(ptest)
        x = np.linspace(-1, 1)
        assert_almost_equal(poly.polyval(x, ptest),
                            poly.polyvalfromroots(x, r))

        # check multidimensional arrays of roots and values
        # check tensor=False
        rshape = (3, 5)
        x = np.arange(-3, 2)
        r = np.random.randint(-5, 5, size=rshape)
        res = poly.polyvalfromroots(x, r, tensor=False)
        tgt = np.empty(r.shape[1:])
        for ii in range(tgt.size):
            tgt[ii] = poly.polyvalfromroots(x[ii], r[:, ii])
        assert_equal(res, tgt)

        # check tensor=True
        x = np.vstack([x, 2 * x])
        res = poly.polyvalfromroots(x, r, tensor=True)
        tgt = np.empty(r.shape[1:] + x.shape)
        for ii in range(r.shape[1]):
            for jj in range(x.shape[0]):
                tgt[ii, jj, :] = poly.polyvalfromroots(x[jj], r[:, ii])
        assert_equal(res, tgt)

    def test_polyval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises_regex(ValueError, 'incompatible',
                            poly.polyval2d, x1, x2[:2], self.c2d)

        # test values
        tgt = y1 * y2
        res = poly.polyval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = poly.polyval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_polyval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test exceptions
        assert_raises_regex(ValueError, 'incompatible',
                      poly.polyval3d, x1, x2, x3[:2], self.c3d)

        # test values
        tgt = y1 * y2 * y3
        res = poly.polyval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = poly.polyval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_polygrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = poly.polygrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = poly.polygrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3) * 2)

    def test_polygrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        # test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = poly.polygrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        # test shape
        z = np.ones((2, 3))
        res = poly.polygrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3) * 3)


class TestIntegral:

    def test_polyint(self):
        # check exceptions
        assert_raises(TypeError, poly.polyint, [0], .5)
        assert_raises(ValueError, poly.polyint, [0], -1)
        assert_raises(ValueError, poly.polyint, [0], 1, [0, 0])
        assert_raises(ValueError, poly.polyint, [0], lbnd=[0])
        assert_raises(ValueError, poly.polyint, [0], scl=[0])
        assert_raises(TypeError, poly.polyint, [0], axis=.5)
        assert_raises(TypeError, poly.polyint, [1, 1], 1.)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = poly.polyint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            res = poly.polyint(pol, m=1, k=[i])
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            res = poly.polyint(pol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(poly.polyval(-1, res), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            res = poly.polyint(pol, m=1, k=[i], scl=2)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = poly.polyint(tgt, m=1)
                res = poly.polyint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = poly.polyint(tgt, m=1, k=[k])
                res = poly.polyint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = poly.polyint(tgt, m=1, k=[k], lbnd=-1)
                res = poly.polyint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = poly.polyint(tgt, m=1, k=[k], scl=2)
                res = poly.polyint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_polyint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([poly.polyint(c) for c in c2d.T]).T
        res = poly.polyint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([poly.polyint(c) for c in c2d])
        res = poly.polyint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([poly.polyint(c, k=3) for c in c2d])
        res = poly.polyint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)


class TestDerivative:

    def test_polyder(self):
        # check exceptions
        assert_raises(TypeError, poly.polyder, [0], .5)
        assert_raises(ValueError, poly.polyder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0] * i + [1]
            res = poly.polyder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = poly.polyder(poly.polyint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = poly.polyder(poly.polyint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_polyder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([poly.polyder(c) for c in c2d.T]).T
        res = poly.polyder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([poly.polyder(c) for c in c2d])
        res = poly.polyder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5)) * 2 - 1

    def test_polyvander(self):
        # check for 1d x
        x = np.arange(3)
        v = poly.polyvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], poly.polyval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = poly.polyvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], poly.polyval(x, coef))

    def test_polyvander2d(self):
        # also tests polyval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = poly.polyvander2d(x1, x2, [1, 2])
        tgt = poly.polyval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = poly.polyvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_polyvander3d(self):
        # also tests polyval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = poly.polyvander3d(x1, x2, x3, [1, 2, 3])
        tgt = poly.polyval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = poly.polyvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))

    def test_polyvandernegdeg(self):
        x = np.arange(3)
        assert_raises(ValueError, poly.polyvander, x, -1)


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, poly.polycompanion, [])
        assert_raises(ValueError, poly.polycompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0] * i + [1]
            assert_(poly.polycompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(poly.polycompanion([1, 2])[0, 0] == -.5)


class TestMisc:

    def test_polyfromroots(self):
        res = poly.polyfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
            tgt = Tlist[i]
            res = poly.polyfromroots(roots) * 2**(i - 1)
            assert_almost_equal(trim(res), trim(tgt))

    def test_polyroots(self):
        assert_almost_equal(poly.polyroots([1]), [])
        assert_almost_equal(poly.polyroots([1, 2]), [-.5])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = poly.polyroots(poly.polyfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

        # Testing for larger root values
        for i in np.logspace(10, 25, num=1000, base=10):
            tgt = np.array([-1, 1, i])
            res = poly.polyroots(poly.polyfromroots(tgt))
            # Adapting the expected precision according to the root value,
            # to take into account numerical calculation error.
            assert_almost_equal(res, tgt, 15 - int(np.log10(i)))
        for i in np.logspace(10, 25, num=1000, base=10):
            tgt = np.array([-1, 1.01, i])
            res = poly.polyroots(poly.polyfromroots(tgt))
            # Adapting the expected precision according to the root value,
            # to take into account numerical calculation error.
            assert_almost_equal(res, tgt, 14 - int(np.log10(i)))

    def test_polyfit(self):
        def f(x):
            return x * (x - 1) * (x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        assert_raises(ValueError, poly.polyfit, [1], [1], -1)
        assert_raises(TypeError, poly.polyfit, [[1]], [1], 0)
        assert_raises(TypeError, poly.polyfit, [], [1], 0)
        assert_raises(TypeError, poly.polyfit, [1], [[[1]]], 0)
        assert_raises(TypeError, poly.polyfit, [1, 2], [1], 0)
        assert_raises(TypeError, poly.polyfit, [1], [1, 2], 0)
        assert_raises(TypeError, poly.polyfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, poly.polyfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, poly.polyfit, [1], [1], [-1,])
        assert_raises(ValueError, poly.polyfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, poly.polyfit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = poly.polyfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(poly.polyval(x, coef3), y)
        coef3 = poly.polyfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(poly.polyval(x, coef3), y)
        #
        coef4 = poly.polyfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(poly.polyval(x, coef4), y)
        coef4 = poly.polyfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(poly.polyval(x, coef4), y)
        #
        coef2d = poly.polyfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = poly.polyfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        wcoef3 = poly.polyfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = poly.polyfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = poly.polyfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = poly.polyfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(poly.polyfit(x, x, 1), [0, 1])
        assert_almost_equal(poly.polyfit(x, x, [0, 1]), [0, 1])
        # test fitting only even Polyendre polynomials
        x = np.linspace(-1, 1)
        y = f2(x)
        coef1 = poly.polyfit(x, y, 4)
        assert_almost_equal(poly.polyval(x, coef1), y)
        coef2 = poly.polyfit(x, y, [0, 2, 4])
        assert_almost_equal(poly.polyval(x, coef2), y)
        assert_almost_equal(coef1, coef2)

    def test_polytrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, poly.polytrim, coef, -1)

        # Test results
        assert_equal(poly.polytrim(coef), coef[:-1])
        assert_equal(poly.polytrim(coef, 1), coef[:-3])
        assert_equal(poly.polytrim(coef, 2), [0])

    def test_polyline(self):
        assert_equal(poly.polyline(3, 4), [3, 4])

    def test_polyline_zero(self):
        assert_equal(poly.polyline(3, 0), [3])

    def test_fit_degenerate_domain(self):
        p = poly.Polynomial.fit([1], [2], deg=0)
        assert_equal(p.coef, [2.])
        p = poly.Polynomial.fit([1, 1], [2, 2.1], deg=0)
        assert_almost_equal(p.coef, [2.05])
        with assert_warns(pu.RankWarning):
            p = poly.Polynomial.fit([1, 1], [2, 2.1], deg=1)

    def test_result_type(self):
        w = np.array([-1, 1], dtype=np.float32)
        p = np.polynomial.Polynomial(w, domain=w, window=w)
        v = p(2)
        assert_equal(v.dtype, np.float32)

        arr = np.polydiv(1, np.float32(1))
        assert_equal(arr[0].dtype, np.float64)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\datetimes\test_setops.py ===
from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest
import pytz

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    Timestamp,
    bdate_range,
    date_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    BMonthEnd,
    Minute,
    MonthEnd,
)

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)


class TestDatetimeIndexSetOps:
    tz = [
        None,
        "UTC",
        "Asia/Tokyo",
        "US/Eastern",
        "dateutil/Asia/Singapore",
        "dateutil/US/Pacific",
    ]

    # TODO: moved from test_datetimelike; dedup with version below
    def test_union2(self, sort):
        everything = date_range("2020-01-01", periods=10)
        first = everything[:5]
        second = everything[5:]
        union = first.union(second, sort=sort)
        tm.assert_index_equal(union, everything)

    @pytest.mark.parametrize("box", [np.array, Series, list])
    def test_union3(self, sort, box):
        everything = date_range("2020-01-01", periods=10)
        first = everything[:5]
        second = everything[5:]

        # GH 10149 support listlike inputs other than Index objects
        expected = first.union(second, sort=sort)
        case = box(second.values)
        result = first.union(case, sort=sort)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tz", tz)
    def test_union(self, tz, sort):
        rng1 = date_range("1/1/2000", freq="D", periods=5, tz=tz)
        other1 = date_range("1/6/2000", freq="D", periods=5, tz=tz)
        expected1 = date_range("1/1/2000", freq="D", periods=10, tz=tz)
        expected1_notsorted = DatetimeIndex(list(other1) + list(rng1))

        rng2 = date_range("1/1/2000", freq="D", periods=5, tz=tz)
        other2 = date_range("1/4/2000", freq="D", periods=5, tz=tz)
        expected2 = date_range("1/1/2000", freq="D", periods=8, tz=tz)
        expected2_notsorted = DatetimeIndex(list(other2) + list(rng2[:3]))

        rng3 = date_range("1/1/2000", freq="D", periods=5, tz=tz)
        other3 = DatetimeIndex([], tz=tz).as_unit("ns")
        expected3 = date_range("1/1/2000", freq="D", periods=5, tz=tz)
        expected3_notsorted = rng3

        for rng, other, exp, exp_notsorted in [
            (rng1, other1, expected1, expected1_notsorted),
            (rng2, other2, expected2, expected2_notsorted),
            (rng3, other3, expected3, expected3_notsorted),
        ]:
            result_union = rng.union(other, sort=sort)
            tm.assert_index_equal(result_union, exp)

            result_union = other.union(rng, sort=sort)
            if sort is None:
                tm.assert_index_equal(result_union, exp)
            else:
                tm.assert_index_equal(result_union, exp_notsorted)

    def test_union_coverage(self, sort):
        idx = DatetimeIndex(["2000-01-03", "2000-01-01", "2000-01-02"])
        ordered = DatetimeIndex(idx.sort_values(), freq="infer")
        result = ordered.union(idx, sort=sort)
        tm.assert_index_equal(result, ordered)

        result = ordered[:0].union(ordered, sort=sort)
        tm.assert_index_equal(result, ordered)
        assert result.freq == ordered.freq

    def test_union_bug_1730(self, sort):
        rng_a = date_range("1/1/2012", periods=4, freq="3h")
        rng_b = date_range("1/1/2012", periods=4, freq="4h")

        result = rng_a.union(rng_b, sort=sort)
        exp = list(rng_a) + list(rng_b[1:])
        if sort is None:
            exp = DatetimeIndex(sorted(exp))
        else:
            exp = DatetimeIndex(exp)
        tm.assert_index_equal(result, exp)

    def test_union_bug_1745(self, sort):
        left = DatetimeIndex(["2012-05-11 15:19:49.695000"])
        right = DatetimeIndex(
            [
                "2012-05-29 13:04:21.322000",
                "2012-05-11 15:27:24.873000",
                "2012-05-11 15:31:05.350000",
            ]
        )

        result = left.union(right, sort=sort)
        exp = DatetimeIndex(
            [
                "2012-05-11 15:19:49.695000",
                "2012-05-29 13:04:21.322000",
                "2012-05-11 15:27:24.873000",
                "2012-05-11 15:31:05.350000",
            ]
        )
        if sort is None:
            exp = exp.sort_values()
        tm.assert_index_equal(result, exp)

    def test_union_bug_4564(self, sort):
        from pandas import DateOffset

        left = date_range("2013-01-01", "2013-02-01")
        right = left + DateOffset(minutes=15)

        result = left.union(right, sort=sort)
        exp = list(left) + list(right)
        if sort is None:
            exp = DatetimeIndex(sorted(exp))
        else:
            exp = DatetimeIndex(exp)
        tm.assert_index_equal(result, exp)

    def test_union_freq_both_none(self, sort):
        # GH11086
        expected = bdate_range("20150101", periods=10)
        expected._data.freq = None

        result = expected.union(expected, sort=sort)
        tm.assert_index_equal(result, expected)
        assert result.freq is None

    def test_union_freq_infer(self):
        # When taking the union of two DatetimeIndexes, we infer
        #  a freq even if the arguments don't have freq.  This matches
        #  TimedeltaIndex behavior.
        dti = date_range("2016-01-01", periods=5)
        left = dti[[0, 1, 3, 4]]
        right = dti[[2, 3, 1]]

        assert left.freq is None
        assert right.freq is None

        result = left.union(right)
        tm.assert_index_equal(result, dti)
        assert result.freq == "D"

    def test_union_dataframe_index(self):
        rng1 = date_range("1/1/1999", "1/1/2012", freq="MS")
        s1 = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)

        rng2 = date_range("1/1/1980", "12/1/2001", freq="MS")
        s2 = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)
        df = DataFrame({"s1": s1, "s2": s2})

        exp = date_range("1/1/1980", "1/1/2012", freq="MS")
        tm.assert_index_equal(df.index, exp)

    def test_union_with_DatetimeIndex(self, sort):
        i1 = Index(np.arange(0, 20, 2, dtype=np.int64))
        i2 = date_range(start="2012-01-03 00:00:00", periods=10, freq="D")
        # Works
        i1.union(i2, sort=sort)
        # Fails with "AttributeError: can't set attribute"
        i2.union(i1, sort=sort)

    def test_union_same_timezone_different_units(self):
        # GH 55238
        idx1 = date_range("2000-01-01", periods=3, tz="UTC").as_unit("ms")
        idx2 = date_range("2000-01-01", periods=3, tz="UTC").as_unit("us")
        result = idx1.union(idx2)
        expected = date_range("2000-01-01", periods=3, tz="UTC").as_unit("us")
        tm.assert_index_equal(result, expected)

    # TODO: moved from test_datetimelike; de-duplicate with version below
    def test_intersection2(self):
        first = date_range("2020-01-01", periods=10)
        second = first[5:]
        intersect = first.intersection(second)
        tm.assert_index_equal(intersect, second)

        # GH 10149
        cases = [klass(second.values) for klass in [np.array, Series, list]]
        for case in cases:
            result = first.intersection(case)
            tm.assert_index_equal(result, second)

        third = Index(["a", "b", "c"])
        result = first.intersection(third)
        expected = Index([], dtype=object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "tz", [None, "Asia/Tokyo", "US/Eastern", "dateutil/US/Pacific"]
    )
    def test_intersection(self, tz, sort):
        # GH 4690 (with tz)
        base = date_range("6/1/2000", "6/30/2000", freq="D", name="idx")

        # if target has the same name, it is preserved
        rng2 = date_range("5/15/2000", "6/20/2000", freq="D", name="idx")
        expected2 = date_range("6/1/2000", "6/20/2000", freq="D", name="idx")

        # if target name is different, it will be reset
        rng3 = date_range("5/15/2000", "6/20/2000", freq="D", name="other")
        expected3 = date_range("6/1/2000", "6/20/2000", freq="D", name=None)

        rng4 = date_range("7/1/2000", "7/31/2000", freq="D", name="idx")
        expected4 = DatetimeIndex([], freq="D", name="idx", dtype="M8[ns]")

        for rng, expected in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            result = base.intersection(rng)
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

        # non-monotonic
        base = DatetimeIndex(
            ["2011-01-05", "2011-01-04", "2011-01-02", "2011-01-03"], tz=tz, name="idx"
        ).as_unit("ns")

        rng2 = DatetimeIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"], tz=tz, name="idx"
        ).as_unit("ns")
        expected2 = DatetimeIndex(
            ["2011-01-04", "2011-01-02"], tz=tz, name="idx"
        ).as_unit("ns")

        rng3 = DatetimeIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"],
            tz=tz,
            name="other",
        ).as_unit("ns")
        expected3 = DatetimeIndex(
            ["2011-01-04", "2011-01-02"], tz=tz, name=None
        ).as_unit("ns")

        # GH 7880
        rng4 = date_range("7/1/2000", "7/31/2000", freq="D", tz=tz, name="idx")
        expected4 = DatetimeIndex([], tz=tz, name="idx").as_unit("ns")
        assert expected4.freq is None

        for rng, expected in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            result = base.intersection(rng, sort=sort)
            if sort is None:
                expected = expected.sort_values()
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

    # parametrize over both anchored and non-anchored freqs, as they
    #  have different code paths
    @pytest.mark.parametrize("freq", ["min", "B"])
    def test_intersection_empty(self, tz_aware_fixture, freq):
        # empty same freq GH2129
        tz = tz_aware_fixture
        rng = date_range("6/1/2000", "6/15/2000", freq=freq, tz=tz)
        result = rng[0:0].intersection(rng)
        assert len(result) == 0
        assert result.freq == rng.freq

        result = rng.intersection(rng[0:0])
        assert len(result) == 0
        assert result.freq == rng.freq

        # no overlap GH#33604
        check_freq = freq != "min"  # We don't preserve freq on non-anchored offsets
        result = rng[:3].intersection(rng[-3:])
        tm.assert_index_equal(result, rng[:0])
        if check_freq:
            # We don't preserve freq on non-anchored offsets
            assert result.freq == rng.freq

        # swapped left and right
        result = rng[-3:].intersection(rng[:3])
        tm.assert_index_equal(result, rng[:0])
        if check_freq:
            # We don't preserve freq on non-anchored offsets
            assert result.freq == rng.freq

    def test_intersection_bug_1708(self):
        from pandas import DateOffset

        index_1 = date_range("1/1/2012", periods=4, freq="12h")
        index_2 = index_1 + DateOffset(hours=1)

        result = index_1.intersection(index_2)
        assert len(result) == 0

    @pytest.mark.parametrize("tz", tz)
    def test_difference(self, tz, sort):
        rng_dates = ["1/2/2000", "1/3/2000", "1/1/2000", "1/4/2000", "1/5/2000"]

        rng1 = DatetimeIndex(rng_dates, tz=tz)
        other1 = date_range("1/6/2000", freq="D", periods=5, tz=tz)
        expected1 = DatetimeIndex(rng_dates, tz=tz)

        rng2 = DatetimeIndex(rng_dates, tz=tz)
        other2 = date_range("1/4/2000", freq="D", periods=5, tz=tz)
        expected2 = DatetimeIndex(rng_dates[:3], tz=tz)

        rng3 = DatetimeIndex(rng_dates, tz=tz)
        other3 = DatetimeIndex([], tz=tz)
        expected3 = DatetimeIndex(rng_dates, tz=tz)

        for rng, other, expected in [
            (rng1, other1, expected1),
            (rng2, other2, expected2),
            (rng3, other3, expected3),
        ]:
            result_diff = rng.difference(other, sort)
            if sort is None and len(other):
                # We dont sort (yet?) when empty GH#24959
                expected = expected.sort_values()
            tm.assert_index_equal(result_diff, expected)

    def test_difference_freq(self, sort):
        # GH14323: difference of DatetimeIndex should not preserve frequency

        index = date_range("20160920", "20160925", freq="D")
        other = date_range("20160921", "20160924", freq="D")
        expected = DatetimeIndex(["20160920", "20160925"], dtype="M8[ns]", freq=None)
        idx_diff = index.difference(other, sort)
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

        # preserve frequency when the difference is a contiguous
        # subset of the original range
        other = date_range("20160922", "20160925", freq="D")
        idx_diff = index.difference(other, sort)
        expected = DatetimeIndex(["20160920", "20160921"], dtype="M8[ns]", freq="D")
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

    def test_datetimeindex_diff(self, sort):
        dti1 = date_range(freq="QE-JAN", start=datetime(1997, 12, 31), periods=100)
        dti2 = date_range(freq="QE-JAN", start=datetime(1997, 12, 31), periods=98)
        assert len(dti1.difference(dti2, sort)) == 2

    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo", "US/Eastern"])
    def test_setops_preserve_freq(self, tz):
        rng = date_range("1/1/2000", "1/1/2002", name="idx", tz=tz)

        result = rng[:50].union(rng[50:100])
        assert result.name == rng.name
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].union(rng[30:100])
        assert result.name == rng.name
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].union(rng[60:100])
        assert result.name == rng.name
        assert result.freq is None
        assert result.tz == rng.tz

        result = rng[:50].intersection(rng[25:75])
        assert result.name == rng.name
        assert result.freqstr == "D"
        assert result.tz == rng.tz

        nofreq = DatetimeIndex(list(rng[25:75]), name="other")
        result = rng[:50].union(nofreq)
        assert result.name is None
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].intersection(nofreq)
        assert result.name is None
        assert result.freq == rng.freq
        assert result.tz == rng.tz

    def test_intersection_non_tick_no_fastpath(self):
        # GH#42104
        dti = DatetimeIndex(
            [
                "2018-12-31",
                "2019-03-31",
                "2019-06-30",
                "2019-09-30",
                "2019-12-31",
                "2020-03-31",
            ],
            freq="QE-DEC",
        )
        result = dti[::2].intersection(dti[1::2])
        expected = dti[:0]
        tm.assert_index_equal(result, expected)

    def test_dti_intersection(self):
        rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")

        left = rng[10:90][::-1]
        right = rng[20:80][::-1]

        assert left.tz == rng.tz
        result = left.intersection(right)
        assert result.tz == left.tz

    # Note: not difference, as there is no symmetry requirement there
    @pytest.mark.parametrize("setop", ["union", "intersection", "symmetric_difference"])
    def test_dti_setop_aware(self, setop):
        # non-overlapping
        # GH#39328 as of 2.0 we cast these to UTC instead of object
        rng = date_range("2012-11-15 00:00:00", periods=6, freq="h", tz="US/Central")

        rng2 = date_range("2012-11-15 12:00:00", periods=6, freq="h", tz="US/Eastern")

        result = getattr(rng, setop)(rng2)

        left = rng.tz_convert("UTC")
        right = rng2.tz_convert("UTC")
        expected = getattr(left, setop)(right)
        tm.assert_index_equal(result, expected)
        assert result.tz == left.tz
        if len(result):
            assert result[0].tz is timezone.utc
            assert result[-1].tz is timezone.utc

    def test_dti_union_mixed(self):
        # GH#21671
        rng = DatetimeIndex([Timestamp("2011-01-01"), pd.NaT])
        rng2 = DatetimeIndex(["2012-01-01", "2012-01-02"], tz="Asia/Tokyo")
        result = rng.union(rng2)
        expected = Index(
            [
                Timestamp("2011-01-01"),
                pd.NaT,
                Timestamp("2012-01-01", tz="Asia/Tokyo"),
                Timestamp("2012-01-02", tz="Asia/Tokyo"),
            ],
            dtype=object,
        )
        tm.assert_index_equal(result, expected)


class TestBusinessDatetimeIndex:
    def test_union(self, sort):
        rng = bdate_range(START, END)
        # overlapping
        left = rng[:10]
        right = rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # non-overlapping, gap in middle
        left = rng[:5]
        right = rng[10:]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, Index)

        # non-overlapping, no gap
        left = rng[:5]
        right = rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # order does not matter
        if sort is None:
            tm.assert_index_equal(right.union(left, sort=sort), the_union)
        else:
            expected = DatetimeIndex(list(right) + list(left))
            tm.assert_index_equal(right.union(left, sort=sort), expected)

        # overlapping, but different offset
        rng = date_range(START, END, freq=BMonthEnd())

        the_union = rng.union(rng, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

    def test_union_not_cacheable(self, sort):
        rng = date_range("1/1/2000", periods=50, freq=Minute())
        rng1 = rng[10:]
        rng2 = rng[:25]
        the_union = rng1.union(rng2, sort=sort)
        if sort is None:
            tm.assert_index_equal(the_union, rng)
        else:
            expected = DatetimeIndex(list(rng[10:]) + list(rng[:10]))
            tm.assert_index_equal(the_union, expected)

        rng1 = rng[10:]
        rng2 = rng[15:35]
        the_union = rng1.union(rng2, sort=sort)
        expected = rng[10:]
        tm.assert_index_equal(the_union, expected)

    def test_intersection(self):
        rng = date_range("1/1/2000", periods=50, freq=Minute())
        rng1 = rng[10:]
        rng2 = rng[:25]
        the_int = rng1.intersection(rng2)
        expected = rng[10:25]
        tm.assert_index_equal(the_int, expected)
        assert isinstance(the_int, DatetimeIndex)
        assert the_int.freq == rng.freq

        the_int = rng1.intersection(rng2)
        tm.assert_index_equal(the_int, expected)

        # non-overlapping
        the_int = rng[:10].intersection(rng[10:])
        expected = DatetimeIndex([]).as_unit("ns")
        tm.assert_index_equal(the_int, expected)

    def test_intersection_bug(self):
        # GH #771
        a = bdate_range("11/30/2011", "12/31/2011")
        b = bdate_range("12/10/2011", "12/20/2011")
        result = a.intersection(b)
        tm.assert_index_equal(result, b)
        assert result.freq == b.freq

    def test_intersection_list(self):
        # GH#35876
        # values is not an Index -> no name -> retain "a"
        values = [Timestamp("2020-01-01"), Timestamp("2020-02-01")]
        idx = DatetimeIndex(values, name="a")
        res = idx.intersection(values)
        tm.assert_index_equal(res, idx)

    def test_month_range_union_tz_pytz(self, sort):
        tz = pytz.timezone("US/Eastern")

        early_start = datetime(2011, 1, 1)
        early_end = datetime(2011, 3, 1)

        late_start = datetime(2011, 3, 1)
        late_end = datetime(2011, 5, 1)

        early_dr = date_range(start=early_start, end=early_end, tz=tz, freq=MonthEnd())
        late_dr = date_range(start=late_start, end=late_end, tz=tz, freq=MonthEnd())

        early_dr.union(late_dr, sort=sort)

    @td.skip_if_windows
    def test_month_range_union_tz_dateutil(self, sort):
        from pandas._libs.tslibs.timezones import dateutil_gettz

        tz = dateutil_gettz("US/Eastern")

        early_start = datetime(2011, 1, 1)
        early_end = datetime(2011, 3, 1)

        late_start = datetime(2011, 3, 1)
        late_end = datetime(2011, 5, 1)

        early_dr = date_range(start=early_start, end=early_end, tz=tz, freq=MonthEnd())
        late_dr = date_range(start=late_start, end=late_end, tz=tz, freq=MonthEnd())

        early_dr.union(late_dr, sort=sort)

    @pytest.mark.parametrize("sort", [False, None])
    def test_intersection_duplicates(self, sort):
        # GH#38196
        idx1 = Index(
            [
                Timestamp("2019-12-13"),
                Timestamp("2019-12-12"),
                Timestamp("2019-12-12"),
            ]
        )
        result = idx1.intersection(idx1, sort=sort)
        expected = Index([Timestamp("2019-12-13"), Timestamp("2019-12-12")])
        tm.assert_index_equal(result, expected)


class TestCustomDatetimeIndex:
    def test_union(self, sort):
        # overlapping
        rng = bdate_range(START, END, freq="C")
        left = rng[:10]
        right = rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # non-overlapping, gap in middle
        left = rng[:5]
        right = rng[10:]

        the_union = left.union(right, sort)
        assert isinstance(the_union, Index)

        # non-overlapping, no gap
        left = rng[:5]
        right = rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # order does not matter
        if sort is None:
            tm.assert_index_equal(right.union(left, sort=sort), the_union)

        # overlapping, but different offset
        rng = date_range(START, END, freq=BMonthEnd())

        the_union = rng.union(rng, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

    def test_intersection_bug(self):
        # GH #771
        a = bdate_range("11/30/2011", "12/31/2011", freq="C")
        b = bdate_range("12/10/2011", "12/20/2011", freq="C")
        result = a.intersection(b)
        tm.assert_index_equal(result, b)
        assert result.freq == b.freq

    @pytest.mark.parametrize(
        "tz", [None, "UTC", "Europe/Berlin", pytz.FixedOffset(-60)]
    )
    def test_intersection_dst_transition(self, tz):
        # GH 46702: Europe/Berlin has DST transition
        idx1 = date_range("2020-03-27", periods=5, freq="D", tz=tz)
        idx2 = date_range("2020-03-30", periods=5, freq="D", tz=tz)
        result = idx1.intersection(idx2)
        expected = date_range("2020-03-30", periods=2, freq="D", tz=tz)
        tm.assert_index_equal(result, expected)

        # GH#45863 same problem for union
        index1 = date_range("2021-10-28", periods=3, freq="D", tz="Europe/London")
        index2 = date_range("2021-10-30", periods=4, freq="D", tz="Europe/London")
        result = index1.union(index2)
        expected = date_range("2021-10-28", periods=6, freq="D", tz="Europe/London")
        tm.assert_index_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\scalar\timedelta\test_timedelta.py ===
""" test the scalar Timedelta """
from datetime import timedelta
import sys

from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    iNaT,
)
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta

from pandas import (
    Timedelta,
    to_timedelta,
)
import pandas._testing as tm


class TestNonNano:
    @pytest.fixture(params=["s", "ms", "us"])
    def unit_str(self, request):
        return request.param

    @pytest.fixture
    def unit(self, unit_str):
        # 7, 8, 9 correspond to second, millisecond, and microsecond, respectively
        attr = f"NPY_FR_{unit_str}"
        return getattr(NpyDatetimeUnit, attr).value

    @pytest.fixture
    def val(self, unit):
        # microsecond that would be just out of bounds for nano
        us = 9223372800000000
        if unit == NpyDatetimeUnit.NPY_FR_us.value:
            value = us
        elif unit == NpyDatetimeUnit.NPY_FR_ms.value:
            value = us // 1000
        else:
            value = us // 1_000_000
        return value

    @pytest.fixture
    def td(self, unit, val):
        return Timedelta._from_value_and_reso(val, unit)

    def test_from_value_and_reso(self, unit, val):
        # Just checking that the fixture is giving us what we asked for
        td = Timedelta._from_value_and_reso(val, unit)
        assert td._value == val
        assert td._creso == unit
        assert td.days == 106752

    def test_unary_non_nano(self, td, unit):
        assert abs(td)._creso == unit
        assert (-td)._creso == unit
        assert (+td)._creso == unit

    def test_sub_preserves_reso(self, td, unit):
        res = td - td
        expected = Timedelta._from_value_and_reso(0, unit)
        assert res == expected
        assert res._creso == unit

    def test_mul_preserves_reso(self, td, unit):
        # The td fixture should always be far from the implementation
        #  bound, so doubling does not risk overflow.
        res = td * 2
        assert res._value == td._value * 2
        assert res._creso == unit

    def test_cmp_cross_reso(self, td):
        # numpy gets this wrong because of silent overflow
        other = Timedelta(days=106751, unit="ns")
        assert other < td
        assert td > other
        assert not other == td
        assert td != other

    def test_to_pytimedelta(self, td):
        res = td.to_pytimedelta()
        expected = timedelta(days=106752)
        assert type(res) is timedelta
        assert res == expected

    def test_to_timedelta64(self, td, unit):
        for res in [td.to_timedelta64(), td.to_numpy(), td.asm8]:
            assert isinstance(res, np.timedelta64)
            assert res.view("i8") == td._value
            if unit == NpyDatetimeUnit.NPY_FR_s.value:
                assert res.dtype == "m8[s]"
            elif unit == NpyDatetimeUnit.NPY_FR_ms.value:
                assert res.dtype == "m8[ms]"
            elif unit == NpyDatetimeUnit.NPY_FR_us.value:
                assert res.dtype == "m8[us]"

    def test_truediv_timedeltalike(self, td):
        assert td / td == 1
        assert (2.5 * td) / td == 2.5

        other = Timedelta(td._value)
        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow."
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td / other

        # Timedelta(other.to_pytimedelta()) has microsecond resolution,
        #  so the division doesn't require casting all the way to nanos,
        #  so succeeds
        res = other.to_pytimedelta() / td
        expected = other.to_pytimedelta() / td.to_pytimedelta()
        assert res == expected

        # if there's no overflow, we cast to the higher reso
        left = Timedelta._from_value_and_reso(50, NpyDatetimeUnit.NPY_FR_us.value)
        right = Timedelta._from_value_and_reso(50, NpyDatetimeUnit.NPY_FR_ms.value)
        result = left / right
        assert result == 0.001

        result = right / left
        assert result == 1000

    def test_truediv_numeric(self, td):
        assert td / np.nan is NaT

        res = td / 2
        assert res._value == td._value / 2
        assert res._creso == td._creso

        res = td / 2.0
        assert res._value == td._value / 2
        assert res._creso == td._creso

    def test_floordiv_timedeltalike(self, td):
        assert td // td == 1
        assert (2.5 * td) // td == 2

        other = Timedelta(td._value)
        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td // other

        # Timedelta(other.to_pytimedelta()) has microsecond resolution,
        #  so the floordiv doesn't require casting all the way to nanos,
        #  so succeeds
        res = other.to_pytimedelta() // td
        assert res == 0

        # if there's no overflow, we cast to the higher reso
        left = Timedelta._from_value_and_reso(50050, NpyDatetimeUnit.NPY_FR_us.value)
        right = Timedelta._from_value_and_reso(50, NpyDatetimeUnit.NPY_FR_ms.value)
        result = left // right
        assert result == 1
        result = right // left
        assert result == 0

    def test_floordiv_numeric(self, td):
        assert td // np.nan is NaT

        res = td // 2
        assert res._value == td._value // 2
        assert res._creso == td._creso

        res = td // 2.0
        assert res._value == td._value // 2
        assert res._creso == td._creso

        assert td // np.array(np.nan) is NaT

        res = td // np.array(2)
        assert res._value == td._value // 2
        assert res._creso == td._creso

        res = td // np.array(2.0)
        assert res._value == td._value // 2
        assert res._creso == td._creso

    def test_addsub_mismatched_reso(self, td):
        # need to cast to since td is out of bounds for ns, so
        #  so we would raise OverflowError without casting
        other = Timedelta(days=1).as_unit("us")

        # td is out of bounds for ns
        result = td + other
        assert result._creso == other._creso
        assert result.days == td.days + 1

        result = other + td
        assert result._creso == other._creso
        assert result.days == td.days + 1

        result = td - other
        assert result._creso == other._creso
        assert result.days == td.days - 1

        result = other - td
        assert result._creso == other._creso
        assert result.days == 1 - td.days

        other2 = Timedelta(500)
        msg = "Cannot cast 106752 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td + other2
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            other2 + td
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            td - other2
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            other2 - td

    def test_min(self, td):
        assert td.min <= td
        assert td.min._creso == td._creso
        assert td.min._value == NaT._value + 1

    def test_max(self, td):
        assert td.max >= td
        assert td.max._creso == td._creso
        assert td.max._value == np.iinfo(np.int64).max

    def test_resolution(self, td):
        expected = Timedelta._from_value_and_reso(1, td._creso)
        result = td.resolution
        assert result == expected
        assert result._creso == expected._creso

    def test_hash(self) -> None:
        # GH#54037
        second_resolution_max = Timedelta(0).as_unit("s").max

        assert hash(second_resolution_max)


def test_timedelta_class_min_max_resolution():
    # when accessed on the class (as opposed to an instance), we default
    #  to nanoseconds
    assert Timedelta.min == Timedelta(NaT._value + 1)
    assert Timedelta.min._creso == NpyDatetimeUnit.NPY_FR_ns.value

    assert Timedelta.max == Timedelta(np.iinfo(np.int64).max)
    assert Timedelta.max._creso == NpyDatetimeUnit.NPY_FR_ns.value

    assert Timedelta.resolution == Timedelta(1)
    assert Timedelta.resolution._creso == NpyDatetimeUnit.NPY_FR_ns.value


class TestTimedeltaUnaryOps:
    def test_invert(self):
        td = Timedelta(10, unit="d")

        msg = "bad operand type for unary ~"
        with pytest.raises(TypeError, match=msg):
            ~td

        # check this matches pytimedelta and timedelta64
        with pytest.raises(TypeError, match=msg):
            ~(td.to_pytimedelta())

        umsg = "ufunc 'invert' not supported for the input types"
        with pytest.raises(TypeError, match=umsg):
            ~(td.to_timedelta64())

    def test_unary_ops(self):
        td = Timedelta(10, unit="d")

        # __neg__, __pos__
        assert -td == Timedelta(-10, unit="d")
        assert -td == Timedelta("-10d")
        assert +td == Timedelta(10, unit="d")

        # __abs__, __abs__(__neg__)
        assert abs(td) == td
        assert abs(-td) == td
        assert abs(-td) == Timedelta("10d")


class TestTimedeltas:
    @pytest.mark.parametrize(
        "unit, value, expected",
        [
            ("us", 9.999, 9999),
            ("ms", 9.999999, 9999999),
            ("s", 9.999999999, 9999999999),
        ],
    )
    def test_rounding_on_int_unit_construction(self, unit, value, expected):
        # GH 12690
        result = Timedelta(value, unit=unit)
        assert result._value == expected
        result = Timedelta(str(value) + unit)
        assert result._value == expected

    def test_total_seconds_scalar(self):
        # see gh-10939
        rng = Timedelta("1 days, 10:11:12.100123456")
        expt = 1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1e9
        tm.assert_almost_equal(rng.total_seconds(), expt)

        rng = Timedelta(np.nan)
        assert np.isnan(rng.total_seconds())

    def test_conversion(self):
        for td in [Timedelta(10, unit="d"), Timedelta("1 days, 10:11:12.012345")]:
            pydt = td.to_pytimedelta()
            assert td == Timedelta(pydt)
            assert td == pydt
            assert isinstance(pydt, timedelta) and not isinstance(pydt, Timedelta)

            assert td == np.timedelta64(td._value, "ns")
            td64 = td.to_timedelta64()

            assert td64 == np.timedelta64(td._value, "ns")
            assert td == td64

            assert isinstance(td64, np.timedelta64)

        # this is NOT equal and cannot be roundtripped (because of the nanos)
        td = Timedelta("1 days, 10:11:12.012345678")
        assert td != td.to_pytimedelta()

    def test_fields(self):
        def check(value):
            # that we are int
            assert isinstance(value, int)

        # compat to datetime.timedelta
        rng = to_timedelta("1 days, 10:11:12")
        assert rng.days == 1
        assert rng.seconds == 10 * 3600 + 11 * 60 + 12
        assert rng.microseconds == 0
        assert rng.nanoseconds == 0

        msg = "'Timedelta' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # GH 10050
        check(rng.days)
        check(rng.seconds)
        check(rng.microseconds)
        check(rng.nanoseconds)

        td = Timedelta("-1 days, 10:11:12")
        assert abs(td) == Timedelta("13:48:48")
        assert str(td) == "-1 days +10:11:12"
        assert -td == Timedelta("0 days 13:48:48")
        assert -Timedelta("-1 days, 10:11:12")._value == 49728000000000
        assert Timedelta("-1 days, 10:11:12")._value == -49728000000000

        rng = to_timedelta("-1 days, 10:11:12.100123456")
        assert rng.days == -1
        assert rng.seconds == 10 * 3600 + 11 * 60 + 12
        assert rng.microseconds == 100 * 1000 + 123
        assert rng.nanoseconds == 456
        msg = "'Timedelta' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # components
        tup = to_timedelta(-1, "us").components
        assert tup.days == -1
        assert tup.hours == 23
        assert tup.minutes == 59
        assert tup.seconds == 59
        assert tup.milliseconds == 999
        assert tup.microseconds == 999
        assert tup.nanoseconds == 0

        # GH 10050
        check(tup.days)
        check(tup.hours)
        check(tup.minutes)
        check(tup.seconds)
        check(tup.milliseconds)
        check(tup.microseconds)
        check(tup.nanoseconds)

        tup = Timedelta("-1 days 1 us").components
        assert tup.days == -2
        assert tup.hours == 23
        assert tup.minutes == 59
        assert tup.seconds == 59
        assert tup.milliseconds == 999
        assert tup.microseconds == 999
        assert tup.nanoseconds == 0

    # TODO: this is a test of to_timedelta string parsing
    def test_iso_conversion(self):
        # GH #21877
        expected = Timedelta(1, unit="s")
        assert to_timedelta("P0DT0H0M1S") == expected

    # TODO: this is a test of to_timedelta returning NaT
    def test_nat_converters(self):
        result = to_timedelta("nat").to_numpy()
        assert result.dtype.kind == "M"
        assert result.astype("int64") == iNaT

        result = to_timedelta("nan").to_numpy()
        assert result.dtype.kind == "M"
        assert result.astype("int64") == iNaT

    def test_numeric_conversions(self):
        assert Timedelta(0) == np.timedelta64(0, "ns")
        assert Timedelta(10) == np.timedelta64(10, "ns")
        assert Timedelta(10, unit="ns") == np.timedelta64(10, "ns")

        assert Timedelta(10, unit="us") == np.timedelta64(10, "us")
        assert Timedelta(10, unit="ms") == np.timedelta64(10, "ms")
        assert Timedelta(10, unit="s") == np.timedelta64(10, "s")
        assert Timedelta(10, unit="d") == np.timedelta64(10, "D")

    def test_timedelta_conversions(self):
        assert Timedelta(timedelta(seconds=1)) == np.timedelta64(1, "s").astype(
            "m8[ns]"
        )
        assert Timedelta(timedelta(microseconds=1)) == np.timedelta64(1, "us").astype(
            "m8[ns]"
        )
        assert Timedelta(timedelta(days=1)) == np.timedelta64(1, "D").astype("m8[ns]")

    def test_to_numpy_alias(self):
        # GH 24653: alias .to_numpy() for scalars
        td = Timedelta("10m7s")
        assert td.to_timedelta64() == td.to_numpy()

        # GH#44460
        msg = "dtype and copy arguments are ignored"
        with pytest.raises(ValueError, match=msg):
            td.to_numpy("m8[s]")
        with pytest.raises(ValueError, match=msg):
            td.to_numpy(copy=True)

    def test_identity(self):
        td = Timedelta(10, unit="d")
        assert isinstance(td, Timedelta)
        assert isinstance(td, timedelta)

    def test_short_format_converters(self):
        def conv(v):
            return v.astype("m8[ns]")

        assert Timedelta("10") == np.timedelta64(10, "ns")
        assert Timedelta("10ns") == np.timedelta64(10, "ns")
        assert Timedelta("100") == np.timedelta64(100, "ns")
        assert Timedelta("100ns") == np.timedelta64(100, "ns")

        assert Timedelta("1000") == np.timedelta64(1000, "ns")
        assert Timedelta("1000ns") == np.timedelta64(1000, "ns")
        assert Timedelta("1000NS") == np.timedelta64(1000, "ns")

        assert Timedelta("10us") == np.timedelta64(10000, "ns")
        assert Timedelta("100us") == np.timedelta64(100000, "ns")
        assert Timedelta("1000us") == np.timedelta64(1000000, "ns")
        assert Timedelta("1000Us") == np.timedelta64(1000000, "ns")
        assert Timedelta("1000uS") == np.timedelta64(1000000, "ns")

        assert Timedelta("1ms") == np.timedelta64(1000000, "ns")
        assert Timedelta("10ms") == np.timedelta64(10000000, "ns")
        assert Timedelta("100ms") == np.timedelta64(100000000, "ns")
        assert Timedelta("1000ms") == np.timedelta64(1000000000, "ns")

        assert Timedelta("-1s") == -np.timedelta64(1000000000, "ns")
        assert Timedelta("1s") == np.timedelta64(1000000000, "ns")
        assert Timedelta("10s") == np.timedelta64(10000000000, "ns")
        assert Timedelta("100s") == np.timedelta64(100000000000, "ns")
        assert Timedelta("1000s") == np.timedelta64(1000000000000, "ns")

        assert Timedelta("1d") == conv(np.timedelta64(1, "D"))
        assert Timedelta("-1d") == -conv(np.timedelta64(1, "D"))
        assert Timedelta("1D") == conv(np.timedelta64(1, "D"))
        assert Timedelta("10D") == conv(np.timedelta64(10, "D"))
        assert Timedelta("100D") == conv(np.timedelta64(100, "D"))
        assert Timedelta("1000D") == conv(np.timedelta64(1000, "D"))
        assert Timedelta("10000D") == conv(np.timedelta64(10000, "D"))

        # space
        assert Timedelta(" 10000D ") == conv(np.timedelta64(10000, "D"))
        assert Timedelta(" - 10000D ") == -conv(np.timedelta64(10000, "D"))

        # invalid
        msg = "invalid unit abbreviation"
        with pytest.raises(ValueError, match=msg):
            Timedelta("1foo")
        msg = "unit abbreviation w/o a number"
        with pytest.raises(ValueError, match=msg):
            Timedelta("foo")

    def test_full_format_converters(self):
        def conv(v):
            return v.astype("m8[ns]")

        d1 = np.timedelta64(1, "D")

        assert Timedelta("1days") == conv(d1)
        assert Timedelta("1days,") == conv(d1)
        assert Timedelta("- 1days,") == -conv(d1)

        assert Timedelta("00:00:01") == conv(np.timedelta64(1, "s"))
        assert Timedelta("06:00:01") == conv(np.timedelta64(6 * 3600 + 1, "s"))
        assert Timedelta("06:00:01.0") == conv(np.timedelta64(6 * 3600 + 1, "s"))
        assert Timedelta("06:00:01.01") == conv(
            np.timedelta64(1000 * (6 * 3600 + 1) + 10, "ms")
        )

        assert Timedelta("- 1days, 00:00:01") == conv(-d1 + np.timedelta64(1, "s"))
        assert Timedelta("1days, 06:00:01") == conv(
            d1 + np.timedelta64(6 * 3600 + 1, "s")
        )
        assert Timedelta("1days, 06:00:01.01") == conv(
            d1 + np.timedelta64(1000 * (6 * 3600 + 1) + 10, "ms")
        )

        # invalid
        msg = "have leftover units"
        with pytest.raises(ValueError, match=msg):
            Timedelta("- 1days, 00")

    def test_pickle(self):
        v = Timedelta("1 days 10:11:12.0123456")
        v_p = tm.round_trip_pickle(v)
        assert v == v_p

    def test_timedelta_hash_equality(self):
        # GH 11129
        v = Timedelta(1, "D")
        td = timedelta(days=1)
        assert hash(v) == hash(td)

        d = {td: 2}
        assert d[v] == 2

        tds = [Timedelta(seconds=1) + Timedelta(days=n) for n in range(20)]
        assert all(hash(td) == hash(td.to_pytimedelta()) for td in tds)

        # python timedeltas drop ns resolution
        ns_td = Timedelta(1, "ns")
        assert hash(ns_td) != hash(ns_td.to_pytimedelta())

    @pytest.mark.skip_ubsan
    @pytest.mark.xfail(
        reason="pd.Timedelta violates the Python hash invariant (GH#44504).",
    )
    @given(
        st.integers(
            min_value=(-sys.maxsize - 1) // 500,
            max_value=sys.maxsize // 500,
        )
    )
    def test_hash_equality_invariance(self, half_microseconds: int) -> None:
        # GH#44504

        nanoseconds = half_microseconds * 500

        pandas_timedelta = Timedelta(nanoseconds)
        numpy_timedelta = np.timedelta64(nanoseconds)

        # See: https://docs.python.org/3/glossary.html#term-hashable
        # Hashable objects which compare equal must have the same hash value.
        assert pandas_timedelta != numpy_timedelta or hash(pandas_timedelta) == hash(
            numpy_timedelta
        )

    def test_implementation_limits(self):
        min_td = Timedelta(Timedelta.min)
        max_td = Timedelta(Timedelta.max)

        # GH 12727
        # timedelta limits correspond to int64 boundaries
        assert min_td._value == iNaT + 1
        assert max_td._value == lib.i8max

        # Beyond lower limit, a NAT before the Overflow
        assert (min_td - Timedelta(1, "ns")) is NaT

        msg = "int too (large|big) to convert"
        with pytest.raises(OverflowError, match=msg):
            min_td - Timedelta(2, "ns")

        with pytest.raises(OverflowError, match=msg):
            max_td + Timedelta(1, "ns")

        # Same tests using the internal nanosecond values
        td = Timedelta(min_td._value - 1, "ns")
        assert td is NaT

        msg = "Cannot cast -9223372036854775809 from ns to 'ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta(min_td._value - 2, "ns")

        msg = "Cannot cast 9223372036854775808 from ns to 'ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            Timedelta(max_td._value + 1, "ns")

    def test_total_seconds_precision(self):
        # GH 19458
        assert Timedelta("30s").total_seconds() == 30.0
        assert Timedelta("0").total_seconds() == 0.0
        assert Timedelta("-2s").total_seconds() == -2.0
        assert Timedelta("5.324s").total_seconds() == 5.324
        assert (Timedelta("30s").total_seconds() - 30.0) < 1e-20
        assert (30.0 - Timedelta("30s").total_seconds()) < 1e-20

    def test_resolution_string(self):
        assert Timedelta(days=1).resolution_string == "D"
        assert Timedelta(days=1, hours=6).resolution_string == "h"
        assert Timedelta(days=1, minutes=6).resolution_string == "min"
        assert Timedelta(days=1, seconds=6).resolution_string == "s"
        assert Timedelta(days=1, milliseconds=6).resolution_string == "ms"
        assert Timedelta(days=1, microseconds=6).resolution_string == "us"
        assert Timedelta(days=1, nanoseconds=6).resolution_string == "ns"

    def test_resolution_deprecated(self):
        # GH#21344
        td = Timedelta(days=4, hours=3)
        result = td.resolution
        assert result == Timedelta(nanoseconds=1)

        # Check that the attribute is available on the class, mirroring
        #  the stdlib timedelta behavior
        result = Timedelta.resolution
        assert result == Timedelta(nanoseconds=1)


@pytest.mark.parametrize(
    "value, expected",
    [
        (Timedelta("10s"), True),
        (Timedelta("-10s"), True),
        (Timedelta(10, unit="ns"), True),
        (Timedelta(0, unit="ns"), False),
        (Timedelta(-10, unit="ns"), True),
        (Timedelta(None), True),
        (NaT, True),
    ],
)
def test_truthiness(value, expected):
    # https://github.com/pandas-dev/pandas/issues/21484
    assert bool(value) is expected


def test_timedelta_attribute_precision():
    # GH 31354
    td = Timedelta(1552211999999999872, unit="ns")
    result = td.days * 86400
    result += td.seconds
    result *= 1000000
    result += td.microseconds
    result *= 1000
    result += td.nanoseconds
    expected = td._value
    assert result == expected

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\tseries\offsets\test_month.py ===
"""
Tests for the following offsets:
- SemiMonthBegin
- SemiMonthEnd
- MonthBegin
- MonthEnd
"""
from __future__ import annotations

from datetime import datetime

import pytest

from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
    MonthBegin,
    MonthEnd,
    SemiMonthBegin,
    SemiMonthEnd,
)

from pandas import (
    DatetimeIndex,
    Series,
    _testing as tm,
)
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,
    assert_offset_equal,
)


class TestSemiMonthEnd:
    def test_offset_whole_year(self):
        dates = (
            datetime(2007, 12, 31),
            datetime(2008, 1, 15),
            datetime(2008, 1, 31),
            datetime(2008, 2, 15),
            datetime(2008, 2, 29),
            datetime(2008, 3, 15),
            datetime(2008, 3, 31),
            datetime(2008, 4, 15),
            datetime(2008, 4, 30),
            datetime(2008, 5, 15),
            datetime(2008, 5, 31),
            datetime(2008, 6, 15),
            datetime(2008, 6, 30),
            datetime(2008, 7, 15),
            datetime(2008, 7, 31),
            datetime(2008, 8, 15),
            datetime(2008, 8, 31),
            datetime(2008, 9, 15),
            datetime(2008, 9, 30),
            datetime(2008, 10, 15),
            datetime(2008, 10, 31),
            datetime(2008, 11, 15),
            datetime(2008, 11, 30),
            datetime(2008, 12, 15),
            datetime(2008, 12, 31),
        )

        for base, exp_date in zip(dates[:-1], dates[1:]):
            assert_offset_equal(SemiMonthEnd(), base, exp_date)

        # ensure .apply_index works as expected
        shift = DatetimeIndex(dates[:-1])
        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = SemiMonthEnd() + shift

        exp = DatetimeIndex(dates[1:])
        tm.assert_index_equal(result, exp)

    offset_cases = []
    offset_cases.append(
        (
            SemiMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 15),
                datetime(2008, 1, 15): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 15),
                datetime(2006, 12, 14): datetime(2006, 12, 15),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 15),
                datetime(2007, 1, 1): datetime(2007, 1, 15),
                datetime(2006, 12, 1): datetime(2006, 12, 15),
                datetime(2006, 12, 15): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(day_of_month=20),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 20),
                datetime(2008, 1, 15): datetime(2008, 1, 20),
                datetime(2008, 1, 21): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 20),
                datetime(2006, 12, 14): datetime(2006, 12, 20),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 20),
                datetime(2007, 1, 1): datetime(2007, 1, 20),
                datetime(2006, 12, 1): datetime(2006, 12, 20),
                datetime(2006, 12, 15): datetime(2006, 12, 20),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 15),
                datetime(2008, 1, 16): datetime(2008, 1, 31),
                datetime(2008, 1, 15): datetime(2008, 1, 15),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2006, 12, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 15),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(0, day_of_month=16),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 16),
                datetime(2008, 1, 16): datetime(2008, 1, 16),
                datetime(2008, 1, 15): datetime(2008, 1, 16),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2006, 12, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 16),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(2),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2007, 1, 15),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2007, 1, 16): datetime(2007, 2, 15),
                datetime(2006, 11, 1): datetime(2006, 11, 30),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 6, 15),
                datetime(2008, 12, 31): datetime(2008, 12, 15),
                datetime(2006, 12, 29): datetime(2006, 12, 15),
                datetime(2006, 12, 30): datetime(2006, 12, 15),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(-1, day_of_month=4),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2007, 1, 4): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 6, 4),
                datetime(2008, 12, 31): datetime(2008, 12, 4),
                datetime(2006, 12, 5): datetime(2006, 12, 4),
                datetime(2006, 12, 30): datetime(2006, 12, 4),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            SemiMonthEnd(-2),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 15),
                datetime(2008, 6, 30): datetime(2008, 5, 31),
                datetime(2008, 3, 15): datetime(2008, 2, 15),
                datetime(2008, 12, 31): datetime(2008, 11, 30),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 14): datetime(2006, 11, 15),
                datetime(2007, 1, 1): datetime(2006, 12, 15),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize("case", offset_cases)
    def test_apply_index(self, case):
        # https://github.com/pandas-dev/pandas/issues/34580
        offset, cases = case
        shift = DatetimeIndex(cases.keys())
        exp = DatetimeIndex(cases.values())

        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = offset + shift
        tm.assert_index_equal(result, exp)

    on_offset_cases = [
        (datetime(2007, 12, 31), True),
        (datetime(2007, 12, 15), True),
        (datetime(2007, 12, 14), False),
        (datetime(2007, 12, 1), False),
        (datetime(2008, 2, 29), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        dt, expected = case
        assert_is_on_offset(SemiMonthEnd(), dt, expected)

    @pytest.mark.parametrize("klass", [Series, DatetimeIndex])
    def test_vectorized_offset_addition(self, klass):
        shift = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )

        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = shift + SemiMonthEnd()
            result2 = SemiMonthEnd() + shift

        exp = klass(
            [
                Timestamp("2000-01-31 00:15:00", tz="US/Central"),
                Timestamp("2000-02-29", tz="US/Central"),
            ],
            name="a",
        )
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)

        shift = klass(
            [
                Timestamp("2000-01-01 00:15:00", tz="US/Central"),
                Timestamp("2000-02-01", tz="US/Central"),
            ],
            name="a",
        )

        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = shift + SemiMonthEnd()
            result2 = SemiMonthEnd() + shift

        exp = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)


class TestSemiMonthBegin:
    def test_offset_whole_year(self):
        dates = (
            datetime(2007, 12, 15),
            datetime(2008, 1, 1),
            datetime(2008, 1, 15),
            datetime(2008, 2, 1),
            datetime(2008, 2, 15),
            datetime(2008, 3, 1),
            datetime(2008, 3, 15),
            datetime(2008, 4, 1),
            datetime(2008, 4, 15),
            datetime(2008, 5, 1),
            datetime(2008, 5, 15),
            datetime(2008, 6, 1),
            datetime(2008, 6, 15),
            datetime(2008, 7, 1),
            datetime(2008, 7, 15),
            datetime(2008, 8, 1),
            datetime(2008, 8, 15),
            datetime(2008, 9, 1),
            datetime(2008, 9, 15),
            datetime(2008, 10, 1),
            datetime(2008, 10, 15),
            datetime(2008, 11, 1),
            datetime(2008, 11, 15),
            datetime(2008, 12, 1),
            datetime(2008, 12, 15),
        )

        for base, exp_date in zip(dates[:-1], dates[1:]):
            assert_offset_equal(SemiMonthBegin(), base, exp_date)

        # ensure .apply_index works as expected
        shift = DatetimeIndex(dates[:-1])
        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = SemiMonthBegin() + shift

        exp = DatetimeIndex(dates[1:])
        tm.assert_index_equal(result, exp)

    offset_cases = [
        (
            SemiMonthBegin(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 15),
                datetime(2008, 1, 15): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 14): datetime(2006, 12, 15),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2007, 1, 1): datetime(2007, 1, 15),
                datetime(2006, 12, 1): datetime(2006, 12, 15),
                datetime(2006, 12, 15): datetime(2007, 1, 1),
            },
        ),
        (
            SemiMonthBegin(day_of_month=20),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 20),
                datetime(2008, 1, 15): datetime(2008, 1, 20),
                datetime(2008, 1, 21): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 14): datetime(2006, 12, 20),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2007, 1, 1): datetime(2007, 1, 20),
                datetime(2006, 12, 1): datetime(2006, 12, 20),
                datetime(2006, 12, 15): datetime(2006, 12, 20),
            },
        ),
        (
            SemiMonthBegin(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 16): datetime(2008, 2, 1),
                datetime(2008, 1, 15): datetime(2008, 1, 15),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 2): datetime(2006, 12, 15),
                datetime(2007, 1, 1): datetime(2007, 1, 1),
            },
        ),
        (
            SemiMonthBegin(0, day_of_month=16),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 16): datetime(2008, 1, 16),
                datetime(2008, 1, 15): datetime(2008, 1, 16),
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2007, 1, 5): datetime(2007, 1, 16),
                datetime(2007, 1, 1): datetime(2007, 1, 1),
            },
        ),
        (
            SemiMonthBegin(2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 1, 31): datetime(2008, 2, 15),
                datetime(2006, 12, 1): datetime(2007, 1, 1),
                datetime(2006, 12, 29): datetime(2007, 1, 15),
                datetime(2006, 12, 15): datetime(2007, 1, 15),
                datetime(2007, 1, 1): datetime(2007, 2, 1),
                datetime(2007, 1, 16): datetime(2007, 2, 15),
                datetime(2006, 11, 1): datetime(2006, 12, 1),
            },
        ),
        (
            SemiMonthBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 15),
                datetime(2008, 6, 30): datetime(2008, 6, 15),
                datetime(2008, 6, 14): datetime(2008, 6, 1),
                datetime(2008, 12, 31): datetime(2008, 12, 15),
                datetime(2006, 12, 29): datetime(2006, 12, 15),
                datetime(2006, 12, 15): datetime(2006, 12, 1),
                datetime(2007, 1, 1): datetime(2006, 12, 15),
            },
        ),
        (
            SemiMonthBegin(-1, day_of_month=4),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 4),
                datetime(2007, 1, 4): datetime(2007, 1, 1),
                datetime(2008, 6, 30): datetime(2008, 6, 4),
                datetime(2008, 12, 31): datetime(2008, 12, 4),
                datetime(2006, 12, 5): datetime(2006, 12, 4),
                datetime(2006, 12, 30): datetime(2006, 12, 4),
                datetime(2006, 12, 2): datetime(2006, 12, 1),
                datetime(2007, 1, 1): datetime(2006, 12, 4),
            },
        ),
        (
            SemiMonthBegin(-2),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 1),
                datetime(2008, 6, 30): datetime(2008, 6, 1),
                datetime(2008, 6, 14): datetime(2008, 5, 15),
                datetime(2008, 12, 31): datetime(2008, 12, 1),
                datetime(2006, 12, 29): datetime(2006, 12, 1),
                datetime(2006, 12, 15): datetime(2006, 11, 15),
                datetime(2007, 1, 1): datetime(2006, 12, 1),
            },
        ),
    ]

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    @pytest.mark.parametrize("case", offset_cases)
    def test_apply_index(self, case):
        offset, cases = case
        shift = DatetimeIndex(cases.keys())

        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = offset + shift

        exp = DatetimeIndex(cases.values())
        tm.assert_index_equal(result, exp)

    on_offset_cases = [
        (datetime(2007, 12, 1), True),
        (datetime(2007, 12, 15), True),
        (datetime(2007, 12, 14), False),
        (datetime(2007, 12, 31), False),
        (datetime(2008, 2, 15), True),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        dt, expected = case
        assert_is_on_offset(SemiMonthBegin(), dt, expected)

    @pytest.mark.parametrize("klass", [Series, DatetimeIndex])
    def test_vectorized_offset_addition(self, klass):
        shift = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = shift + SemiMonthBegin()
            result2 = SemiMonthBegin() + shift

        exp = klass(
            [
                Timestamp("2000-02-01 00:15:00", tz="US/Central"),
                Timestamp("2000-03-01", tz="US/Central"),
            ],
            name="a",
        )
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)

        shift = klass(
            [
                Timestamp("2000-01-01 00:15:00", tz="US/Central"),
                Timestamp("2000-02-01", tz="US/Central"),
            ],
            name="a",
        )
        with tm.assert_produces_warning(None):
            # GH#22535 check that we don't get a FutureWarning from adding
            # an integer array to PeriodIndex
            result = shift + SemiMonthBegin()
            result2 = SemiMonthBegin() + shift

        exp = klass(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)


class TestMonthBegin:
    offset_cases = []
    # NOTE: I'm not entirely happy with the logic here for Begin -ss
    # see thread 'offset conventions' on the ML
    offset_cases.append(
        (
            MonthBegin(),
            {
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 2, 1): datetime(2008, 3, 1),
                datetime(2006, 12, 31): datetime(2007, 1, 1),
                datetime(2006, 12, 1): datetime(2007, 1, 1),
                datetime(2007, 1, 31): datetime(2007, 2, 1),
            },
        )
    )

    offset_cases.append(
        (
            MonthBegin(0),
            {
                datetime(2008, 1, 31): datetime(2008, 2, 1),
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2006, 12, 3): datetime(2007, 1, 1),
                datetime(2007, 1, 31): datetime(2007, 2, 1),
            },
        )
    )

    offset_cases.append(
        (
            MonthBegin(2),
            {
                datetime(2008, 2, 29): datetime(2008, 4, 1),
                datetime(2008, 1, 31): datetime(2008, 3, 1),
                datetime(2006, 12, 31): datetime(2007, 2, 1),
                datetime(2007, 12, 28): datetime(2008, 2, 1),
                datetime(2007, 1, 1): datetime(2007, 3, 1),
                datetime(2006, 11, 1): datetime(2007, 1, 1),
            },
        )
    )

    offset_cases.append(
        (
            MonthBegin(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 1),
                datetime(2008, 5, 31): datetime(2008, 5, 1),
                datetime(2008, 12, 31): datetime(2008, 12, 1),
                datetime(2006, 12, 29): datetime(2006, 12, 1),
                datetime(2006, 1, 2): datetime(2006, 1, 1),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)


class TestMonthEnd:
    def test_day_of_month(self):
        dt = datetime(2007, 1, 1)
        offset = MonthEnd()

        result = dt + offset
        assert result == Timestamp(2007, 1, 31)

        result = result + offset
        assert result == Timestamp(2007, 2, 28)

    def test_normalize(self):
        dt = datetime(2007, 1, 1, 3)

        result = dt + MonthEnd(normalize=True)
        expected = dt.replace(hour=0) + MonthEnd()
        assert result == expected

    offset_cases = []
    offset_cases.append(
        (
            MonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 2, 29),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2007, 1, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
                datetime(2006, 12, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            MonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 1, 31): datetime(2008, 1, 31),
                datetime(2006, 12, 29): datetime(2006, 12, 31),
                datetime(2006, 12, 31): datetime(2006, 12, 31),
                datetime(2007, 1, 1): datetime(2007, 1, 31),
            },
        )
    )

    offset_cases.append(
        (
            MonthEnd(2),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 1, 31): datetime(2008, 3, 31),
                datetime(2006, 12, 29): datetime(2007, 1, 31),
                datetime(2006, 12, 31): datetime(2007, 2, 28),
                datetime(2007, 1, 1): datetime(2007, 2, 28),
                datetime(2006, 11, 1): datetime(2006, 12, 31),
            },
        )
    )

    offset_cases.append(
        (
            MonthEnd(-1),
            {
                datetime(2007, 1, 1): datetime(2006, 12, 31),
                datetime(2008, 6, 30): datetime(2008, 5, 31),
                datetime(2008, 12, 31): datetime(2008, 11, 30),
                datetime(2006, 12, 29): datetime(2006, 11, 30),
                datetime(2006, 12, 30): datetime(2006, 11, 30),
                datetime(2007, 1, 1): datetime(2006, 12, 31),
            },
        )
    )

    @pytest.mark.parametrize("case", offset_cases)
    def test_offset(self, case):
        offset, cases = case
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    on_offset_cases = [
        (MonthEnd(), datetime(2007, 12, 31), True),
        (MonthEnd(), datetime(2008, 1, 1), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        assert_is_on_offset(offset, dt, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\tests\test_ast.py ===
import math
from sympy.core.containers import Tuple
from sympy.core.numbers import nan, oo, Float, Integer
from sympy.core.relational import Lt
from sympy.core.symbol import symbols, Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.testing.pytest import raises


from sympy.codegen.ast import (
    Assignment, Attribute, aug_assign, CodeBlock, For, Type, Variable, Pointer, Declaration,
    AddAugmentedAssignment, SubAugmentedAssignment, MulAugmentedAssignment,
    DivAugmentedAssignment, ModAugmentedAssignment, value_const, pointer_const,
    integer, real, complex_, int8, uint8, float16 as f16, float32 as f32,
    float64 as f64, float80 as f80, float128 as f128, complex64 as c64, complex128 as c128,
    While, Scope, String, Print, QuotedString, FunctionPrototype, FunctionDefinition, Return,
    FunctionCall, untyped, IntBaseType, intc, Node, none, NoneToken, Token, Comment
)

x, y, z, t, x0, x1, x2, a, b = symbols("x, y, z, t, x0, x1, x2, a, b")
n = symbols("n", integer=True)
A = MatrixSymbol('A', 3, 1)
mat = Matrix([1, 2, 3])
B = IndexedBase('B')
i = Idx("i", n)
A22 = MatrixSymbol('A22',2,2)
B22 = MatrixSymbol('B22',2,2)


def test_Assignment():
    # Here we just do things to show they don't error
    Assignment(x, y)
    Assignment(x, 0)
    Assignment(A, mat)
    Assignment(A[1,0], 0)
    Assignment(A[1,0], x)
    Assignment(B[i], x)
    Assignment(B[i], 0)
    a = Assignment(x, y)
    assert a.func(*a.args) == a
    assert a.op == ':='
    # Here we test things to show that they error
    # Matrix to scalar
    raises(ValueError, lambda: Assignment(B[i], A))
    raises(ValueError, lambda: Assignment(B[i], mat))
    raises(ValueError, lambda: Assignment(x, mat))
    raises(ValueError, lambda: Assignment(x, A))
    raises(ValueError, lambda: Assignment(A[1,0], mat))
    # Scalar to matrix
    raises(ValueError, lambda: Assignment(A, x))
    raises(ValueError, lambda: Assignment(A, 0))
    # Non-atomic lhs
    raises(TypeError, lambda: Assignment(mat, A))
    raises(TypeError, lambda: Assignment(0, x))
    raises(TypeError, lambda: Assignment(x*x, 1))
    raises(TypeError, lambda: Assignment(A + A, mat))
    raises(TypeError, lambda: Assignment(B, 0))


def test_AugAssign():
    # Here we just do things to show they don't error
    aug_assign(x, '+', y)
    aug_assign(x, '+', 0)
    aug_assign(A, '+', mat)
    aug_assign(A[1, 0], '+', 0)
    aug_assign(A[1, 0], '+', x)
    aug_assign(B[i], '+', x)
    aug_assign(B[i], '+', 0)

    # Check creation via aug_assign vs constructor
    for binop, cls in [
            ('+', AddAugmentedAssignment),
            ('-', SubAugmentedAssignment),
            ('*', MulAugmentedAssignment),
            ('/', DivAugmentedAssignment),
            ('%', ModAugmentedAssignment),
        ]:
        a = aug_assign(x, binop, y)
        b = cls(x, y)
        assert a.func(*a.args) == a == b
        assert a.binop == binop
        assert a.op == binop + '='

    # Here we test things to show that they error
    # Matrix to scalar
    raises(ValueError, lambda: aug_assign(B[i], '+', A))
    raises(ValueError, lambda: aug_assign(B[i], '+', mat))
    raises(ValueError, lambda: aug_assign(x, '+', mat))
    raises(ValueError, lambda: aug_assign(x, '+', A))
    raises(ValueError, lambda: aug_assign(A[1, 0], '+', mat))
    # Scalar to matrix
    raises(ValueError, lambda: aug_assign(A, '+', x))
    raises(ValueError, lambda: aug_assign(A, '+', 0))
    # Non-atomic lhs
    raises(TypeError, lambda: aug_assign(mat, '+', A))
    raises(TypeError, lambda: aug_assign(0, '+', x))
    raises(TypeError, lambda: aug_assign(x * x, '+', 1))
    raises(TypeError, lambda: aug_assign(A + A, '+', mat))
    raises(TypeError, lambda: aug_assign(B, '+', 0))


def test_Assignment_printing():
    assignment_classes = [
        Assignment,
        AddAugmentedAssignment,
        SubAugmentedAssignment,
        MulAugmentedAssignment,
        DivAugmentedAssignment,
        ModAugmentedAssignment,
    ]
    pairs = [
        (x, 2 * y + 2),
        (B[i], x),
        (A22, B22),
        (A[0, 0], x),
    ]

    for cls in assignment_classes:
        for lhs, rhs in pairs:
            a = cls(lhs, rhs)
            assert repr(a) == '%s(%s, %s)' % (cls.__name__, repr(lhs), repr(rhs))


def test_CodeBlock():
    c = CodeBlock(Assignment(x, 1), Assignment(y, x + 1))
    assert c.func(*c.args) == c

    assert c.left_hand_sides == Tuple(x, y)
    assert c.right_hand_sides == Tuple(1, x + 1)

def test_CodeBlock_topological_sort():
    assignments = [
        Assignment(x, y + z),
        Assignment(z, 1),
        Assignment(t, x),
        Assignment(y, 2),
        ]

    ordered_assignments = [
        # Note that the unrelated z=1 and y=2 are kept in that order
        Assignment(z, 1),
        Assignment(y, 2),
        Assignment(x, y + z),
        Assignment(t, x),
        ]
    c1 = CodeBlock.topological_sort(assignments)
    assert c1 == CodeBlock(*ordered_assignments)

    # Cycle
    invalid_assignments = [
        Assignment(x, y + z),
        Assignment(z, 1),
        Assignment(y, x),
        Assignment(y, 2),
        ]

    raises(ValueError, lambda: CodeBlock.topological_sort(invalid_assignments))

    # Free symbols
    free_assignments = [
        Assignment(x, y + z),
        Assignment(z, a * b),
        Assignment(t, x),
        Assignment(y, b + 3),
        ]

    free_assignments_ordered = [
        Assignment(z, a * b),
        Assignment(y, b + 3),
        Assignment(x, y + z),
        Assignment(t, x),
        ]

    c2 = CodeBlock.topological_sort(free_assignments)
    assert c2 == CodeBlock(*free_assignments_ordered)

def test_CodeBlock_free_symbols():
    c1 = CodeBlock(
        Assignment(x, y + z),
        Assignment(z, 1),
        Assignment(t, x),
        Assignment(y, 2),
        )
    assert c1.free_symbols == set()

    c2 = CodeBlock(
        Assignment(x, y + z),
        Assignment(z, a * b),
        Assignment(t, x),
        Assignment(y, b + 3),
    )
    assert c2.free_symbols == {a, b}

def test_CodeBlock_cse():
    c1 = CodeBlock(
        Assignment(y, 1),
        Assignment(x, sin(y)),
        Assignment(z, sin(y)),
        Assignment(t, x*z),
        )
    assert c1.cse() == CodeBlock(
        Assignment(y, 1),
        Assignment(x0, sin(y)),
        Assignment(x, x0),
        Assignment(z, x0),
        Assignment(t, x*z),
    )

    # Multiple assignments to same symbol not supported
    raises(NotImplementedError, lambda: CodeBlock(
        Assignment(x, 1),
        Assignment(y, 1), Assignment(y, 2)
    ).cse())

    # Check auto-generated symbols do not collide with existing ones
    c2 = CodeBlock(
        Assignment(x0, sin(y) + 1),
        Assignment(x1, 2 * sin(y)),
        Assignment(z, x * y),
        )
    assert c2.cse() == CodeBlock(
        Assignment(x2, sin(y)),
        Assignment(x0, x2 + 1),
        Assignment(x1, 2 * x2),
        Assignment(z, x * y),
        )


def test_CodeBlock_cse__issue_14118():
    # see https://github.com/sympy/sympy/issues/14118
    c = CodeBlock(
        Assignment(A22, Matrix([[x, sin(y)],[3, 4]])),
        Assignment(B22, Matrix([[sin(y), 2*sin(y)], [sin(y)**2, 7]]))
    )
    assert c.cse() == CodeBlock(
        Assignment(x0, sin(y)),
        Assignment(A22, Matrix([[x, x0],[3, 4]])),
        Assignment(B22, Matrix([[x0, 2*x0], [x0**2, 7]]))
    )

def test_For():
    f = For(n, Range(0, 3), (Assignment(A[n, 0], x + n), aug_assign(x, '+', y)))
    f = For(n, (1, 2, 3, 4, 5), (Assignment(A[n, 0], x + n),))
    assert f.func(*f.args) == f
    raises(TypeError, lambda: For(n, x, (x + y,)))


def test_none():
    assert none.is_Atom
    assert none == none
    class Foo(Token):
        pass
    foo = Foo()
    assert foo != none
    assert none == None
    assert none == NoneToken()
    assert none.func(*none.args) == none


def test_String():
    st = String('foobar')
    assert st.is_Atom
    assert st == String('foobar')
    assert st.text == 'foobar'
    assert st.func(**st.kwargs()) == st
    assert st.func(*st.args) == st


    class Signifier(String):
        pass

    si = Signifier('foobar')
    assert si != st
    assert si.text == st.text
    s = String('foo')
    assert str(s) == 'foo'
    assert repr(s) == "String('foo')"

def test_Comment():
    c = Comment('foobar')
    assert c.text == 'foobar'
    assert str(c) == 'foobar'

def test_Node():
    n = Node()
    assert n == Node()
    assert n.func(*n.args) == n


def test_Type():
    t = Type('MyType')
    assert len(t.args) == 1
    assert t.name == String('MyType')
    assert str(t) == 'MyType'
    assert repr(t) == "Type(String('MyType'))"
    assert Type(t) == t
    assert t.func(*t.args) == t
    t1 = Type('t1')
    t2 = Type('t2')
    assert t1 != t2
    assert t1 == t1 and t2 == t2
    t1b = Type('t1')
    assert t1 == t1b
    assert t2 != t1b


def test_Type__from_expr():
    assert Type.from_expr(i) == integer
    u = symbols('u', real=True)
    assert Type.from_expr(u) == real
    assert Type.from_expr(n) == integer
    assert Type.from_expr(3) == integer
    assert Type.from_expr(3.0) == real
    assert Type.from_expr(3+1j) == complex_
    raises(ValueError, lambda: Type.from_expr(sum))


def test_Type__cast_check__integers():
    # Rounding
    raises(ValueError, lambda: integer.cast_check(3.5))
    assert integer.cast_check('3') == 3
    assert integer.cast_check(Float('3.0000000000000000000')) == 3
    assert integer.cast_check(Float('3.0000000000000000001')) == 3  # unintuitive maybe?

    # Range
    assert int8.cast_check(127.0) == 127
    raises(ValueError, lambda: int8.cast_check(128))
    assert int8.cast_check(-128) == -128
    raises(ValueError, lambda: int8.cast_check(-129))

    assert uint8.cast_check(0) == 0
    assert uint8.cast_check(128) == 128
    raises(ValueError, lambda: uint8.cast_check(256.0))
    raises(ValueError, lambda: uint8.cast_check(-1))

def test_Attribute():
    noexcept = Attribute('noexcept')
    assert noexcept == Attribute('noexcept')
    alignas16 = Attribute('alignas', [16])
    alignas32 = Attribute('alignas', [32])
    assert alignas16 != alignas32
    assert alignas16.func(*alignas16.args) == alignas16


def test_Variable():
    v = Variable(x, type=real)
    assert v == Variable(v)
    assert v == Variable('x', type=real)
    assert v.symbol == x
    assert v.type == real
    assert value_const not in v.attrs
    assert v.func(*v.args) == v
    assert str(v) == 'Variable(x, type=real)'

    w = Variable(y, f32, attrs={value_const})
    assert w.symbol == y
    assert w.type == f32
    assert value_const in w.attrs
    assert w.func(*w.args) == w

    v_n = Variable(n, type=Type.from_expr(n))
    assert v_n.type == integer
    assert v_n.func(*v_n.args) == v_n
    v_i = Variable(i, type=Type.from_expr(n))
    assert v_i.type == integer
    assert v_i != v_n

    a_i = Variable.deduced(i)
    assert a_i.type == integer
    assert Variable.deduced(Symbol('x', real=True)).type == real
    assert a_i.func(*a_i.args) == a_i

    v_n2 = Variable.deduced(n, value=3.5, cast_check=False)
    assert v_n2.func(*v_n2.args) == v_n2
    assert abs(v_n2.value - 3.5) < 1e-15
    raises(ValueError, lambda: Variable.deduced(n, value=3.5, cast_check=True))

    v_n3 = Variable.deduced(n)
    assert v_n3.type == integer
    assert str(v_n3) == 'Variable(n, type=integer)'
    assert Variable.deduced(z, value=3).type == integer
    assert Variable.deduced(z, value=3.0).type == real
    assert Variable.deduced(z, value=3.0+1j).type == complex_


def test_Pointer():
    p = Pointer(x)
    assert p.symbol == x
    assert p.type == untyped
    assert value_const not in p.attrs
    assert pointer_const not in p.attrs
    assert p.func(*p.args) == p

    u = symbols('u', real=True)
    pu = Pointer(u, type=Type.from_expr(u), attrs={value_const, pointer_const})
    assert pu.symbol is u
    assert pu.type == real
    assert value_const in pu.attrs
    assert pointer_const in pu.attrs
    assert pu.func(*pu.args) == pu

    i = symbols('i', integer=True)
    deref = pu[i]
    assert deref.indices == (i,)


def test_Declaration():
    u = symbols('u', real=True)
    vu = Variable(u, type=Type.from_expr(u))
    assert Declaration(vu).variable.type == real
    vn = Variable(n, type=Type.from_expr(n))
    assert Declaration(vn).variable.type == integer

    # PR 19107, does not allow comparison between expressions and Basic
    # lt = StrictLessThan(vu, vn)
    # assert isinstance(lt, StrictLessThan)

    vuc = Variable(u, Type.from_expr(u), value=3.0, attrs={value_const})
    assert value_const in vuc.attrs
    assert pointer_const not in vuc.attrs
    decl = Declaration(vuc)
    assert decl.variable == vuc
    assert isinstance(decl.variable.value, Float)
    assert decl.variable.value == 3.0
    assert decl.func(*decl.args) == decl
    assert vuc.as_Declaration() == decl
    assert vuc.as_Declaration(value=None, attrs=None) == Declaration(vu)

    vy = Variable(y, type=integer, value=3)
    decl2 = Declaration(vy)
    assert decl2.variable == vy
    assert decl2.variable.value == Integer(3)

    vi = Variable(i, type=Type.from_expr(i), value=3.0)
    decl3 = Declaration(vi)
    assert decl3.variable.type == integer
    assert decl3.variable.value == 3.0

    raises(ValueError, lambda: Declaration(vi, 42))


def test_IntBaseType():
    assert intc.name == String('intc')
    assert intc.args == (intc.name,)
    assert str(IntBaseType('a').name) == 'a'


def test_FloatType():
    assert f16.dig == 3
    assert f32.dig == 6
    assert f64.dig == 15
    assert f80.dig == 18
    assert f128.dig == 33

    assert f16.decimal_dig == 5
    assert f32.decimal_dig == 9
    assert f64.decimal_dig == 17
    assert f80.decimal_dig == 21
    assert f128.decimal_dig == 36

    assert f16.max_exponent == 16
    assert f32.max_exponent == 128
    assert f64.max_exponent == 1024
    assert f80.max_exponent == 16384
    assert f128.max_exponent == 16384

    assert f16.min_exponent == -13
    assert f32.min_exponent == -125
    assert f64.min_exponent == -1021
    assert f80.min_exponent == -16381
    assert f128.min_exponent == -16381

    assert abs(f16.eps / Float('0.00097656', precision=16) - 1) < 0.1*10**-f16.dig
    assert abs(f32.eps / Float('1.1920929e-07', precision=32) - 1) < 0.1*10**-f32.dig
    assert abs(f64.eps / Float('2.2204460492503131e-16', precision=64) - 1) < 0.1*10**-f64.dig
    assert abs(f80.eps / Float('1.08420217248550443401e-19', precision=80) - 1) < 0.1*10**-f80.dig
    assert abs(f128.eps / Float(' 1.92592994438723585305597794258492732e-34', precision=128) - 1) < 0.1*10**-f128.dig

    assert abs(f16.max / Float('65504', precision=16) - 1) < .1*10**-f16.dig
    assert abs(f32.max / Float('3.40282347e+38', precision=32) - 1) < 0.1*10**-f32.dig
    assert abs(f64.max / Float('1.79769313486231571e+308', precision=64) - 1) < 0.1*10**-f64.dig  # cf. np.finfo(np.float64).max
    assert abs(f80.max / Float('1.18973149535723176502e+4932', precision=80) - 1) < 0.1*10**-f80.dig
    assert abs(f128.max / Float('1.18973149535723176508575932662800702e+4932', precision=128) - 1) < 0.1*10**-f128.dig

    # cf. np.finfo(np.float32).tiny
    assert abs(f16.tiny / Float('6.1035e-05', precision=16) - 1) < 0.1*10**-f16.dig
    assert abs(f32.tiny / Float('1.17549435e-38', precision=32) - 1) < 0.1*10**-f32.dig
    assert abs(f64.tiny / Float('2.22507385850720138e-308', precision=64) - 1) < 0.1*10**-f64.dig
    assert abs(f80.tiny / Float('3.36210314311209350626e-4932', precision=80) - 1) < 0.1*10**-f80.dig
    assert abs(f128.tiny / Float('3.3621031431120935062626778173217526e-4932', precision=128) - 1) < 0.1*10**-f128.dig

    assert f64.cast_check(0.5) == Float(0.5, 17)
    assert abs(f64.cast_check(3.7) - 3.7) < 3e-17
    assert isinstance(f64.cast_check(3), (Float, float))

    assert f64.cast_nocheck(oo) == float('inf')
    assert f64.cast_nocheck(-oo) == float('-inf')
    assert f64.cast_nocheck(float(oo)) == float('inf')
    assert f64.cast_nocheck(float(-oo)) == float('-inf')
    assert math.isnan(f64.cast_nocheck(nan))

    assert f32 != f64
    assert f64 == f64.func(*f64.args)


def test_Type__cast_check__floating_point():
    raises(ValueError, lambda: f32.cast_check(123.45678949))
    raises(ValueError, lambda: f32.cast_check(12.345678949))
    raises(ValueError, lambda: f32.cast_check(1.2345678949))
    raises(ValueError, lambda: f32.cast_check(.12345678949))
    assert abs(123.456789049 - f32.cast_check(123.456789049) - 4.9e-8) < 1e-8
    assert abs(0.12345678904 - f32.cast_check(0.12345678904) - 4e-11) < 1e-11

    dcm21 = Float('0.123456789012345670499')  # 21 decimals
    assert abs(dcm21 - f64.cast_check(dcm21) - 4.99e-19) < 1e-19

    f80.cast_check(Float('0.12345678901234567890103', precision=88))
    raises(ValueError, lambda: f80.cast_check(Float('0.12345678901234567890149', precision=88)))

    v10 = 12345.67894
    raises(ValueError, lambda: f32.cast_check(v10))
    assert abs(Float(str(v10), precision=64+8) - f64.cast_check(v10)) < v10*1e-16

    assert abs(f32.cast_check(2147483647) - 2147483650) < 1


def test_Type__cast_check__complex_floating_point():
    val9_11 = 123.456789049 + 0.123456789049j
    raises(ValueError, lambda: c64.cast_check(.12345678949 + .12345678949j))
    assert abs(val9_11 - c64.cast_check(val9_11) - 4.9e-8) < 1e-8

    dcm21 = Float('0.123456789012345670499') + 1e-20j  # 21 decimals
    assert abs(dcm21 - c128.cast_check(dcm21) - 4.99e-19) < 1e-19
    v19 = Float('0.1234567890123456749') + 1j*Float('0.1234567890123456749')
    raises(ValueError, lambda: c128.cast_check(v19))


def test_While():
    xpp = AddAugmentedAssignment(x, 1)
    whl1 = While(x < 2, [xpp])
    assert whl1.condition.args[0] == x
    assert whl1.condition.args[1] == 2
    assert whl1.condition == Lt(x, 2, evaluate=False)
    assert whl1.body.args == (xpp,)
    assert whl1.func(*whl1.args) == whl1

    cblk = CodeBlock(AddAugmentedAssignment(x, 1))
    whl2 = While(x < 2, cblk)
    assert whl1 == whl2
    assert whl1 != While(x < 3, [xpp])


def test_Scope():
    assign = Assignment(x, y)
    incr = AddAugmentedAssignment(x, 1)
    scp = Scope([assign, incr])
    cblk = CodeBlock(assign, incr)
    assert scp.body == cblk
    assert scp == Scope(cblk)
    assert scp != Scope([incr, assign])
    assert scp.func(*scp.args) == scp


def test_Print():
    fmt = "%d %.3f"
    ps = Print([n, x], fmt)
    assert str(ps.format_string) == fmt
    assert ps.print_args == Tuple(n, x)
    assert ps.args == (Tuple(n, x), QuotedString(fmt), none)
    assert ps == Print((n, x), fmt)
    assert ps != Print([x, n], fmt)
    assert ps.func(*ps.args) == ps

    ps2 = Print([n, x])
    assert ps2 == Print([n, x])
    assert ps2 != ps
    assert ps2.format_string == None


def test_FunctionPrototype_and_FunctionDefinition():
    vx = Variable(x, type=real)
    vn = Variable(n, type=integer)
    fp1 = FunctionPrototype(real, 'power', [vx, vn])
    assert fp1.return_type == real
    assert fp1.name == String('power')
    assert fp1.parameters == Tuple(vx, vn)
    assert fp1 == FunctionPrototype(real, 'power', [vx, vn])
    assert fp1 != FunctionPrototype(real, 'power', [vn, vx])
    assert fp1.func(*fp1.args) == fp1


    body = [Assignment(x, x**n), Return(x)]
    fd1 = FunctionDefinition(real, 'power', [vx, vn], body)
    assert fd1.return_type == real
    assert str(fd1.name) == 'power'
    assert fd1.parameters == Tuple(vx, vn)
    assert fd1.body == CodeBlock(*body)
    assert fd1 == FunctionDefinition(real, 'power', [vx, vn], body)
    assert fd1 != FunctionDefinition(real, 'power', [vx, vn], body[::-1])
    assert fd1.func(*fd1.args) == fd1

    fp2 = FunctionPrototype.from_FunctionDefinition(fd1)
    assert fp2 == fp1

    fd2 = FunctionDefinition.from_FunctionPrototype(fp1, body)
    assert fd2 == fd1


def test_Return():
    rs = Return(x)
    assert rs.args == (x,)
    assert rs == Return(x)
    assert rs != Return(y)
    assert rs.func(*rs.args) == rs


def test_FunctionCall():
    fc = FunctionCall('power', (x, 3))
    assert fc.function_args[0] == x
    assert fc.function_args[1] == 3
    assert len(fc.function_args) == 2
    assert isinstance(fc.function_args[1], Integer)
    assert fc == FunctionCall('power', (x, 3))
    assert fc != FunctionCall('power', (3, x))
    assert fc != FunctionCall('Power', (x, 3))
    assert fc.func(*fc.args) == fc

    fc2 = FunctionCall('fma', [2, 3, 4])
    assert len(fc2.function_args) == 3
    assert fc2.function_args[0] == 2
    assert fc2.function_args[1] == 3
    assert fc2.function_args[2] == 4
    assert str(fc2) in ( # not sure if QuotedString is a better default...
        'FunctionCall(fma, function_args=(2, 3, 4))',
        'FunctionCall("fma", function_args=(2, 3, 4))',
    )

def test_ast_replace():
    x = Variable('x', real)
    y = Variable('y', real)
    n = Variable('n', integer)

    pwer = FunctionDefinition(real, 'pwer', [x, n], [pow(x.symbol, n.symbol)])
    pname = pwer.name
    pcall = FunctionCall('pwer', [y, 3])

    tree1 = CodeBlock(pwer, pcall)
    assert str(tree1.args[0].name) == 'pwer'
    assert str(tree1.args[1].name) == 'pwer'
    for a, b in zip(tree1, [pwer, pcall]):
        assert a == b

    tree2 = tree1.replace(pname, String('power'))
    assert str(tree1.args[0].name) == 'pwer'
    assert str(tree1.args[1].name) == 'pwer'
    assert str(tree2.args[0].name) == 'power'
    assert str(tree2.args[1].name) == 'power'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\test_common.py ===
"""
Tests for the pandas.io.common functionalities
"""
import codecs
import errno
from functools import partial
from io import (
    BytesIO,
    StringIO,
    UnsupportedOperation,
)
import mmap
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pytest

from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import pa_version_under19p0
import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm

import pandas.io.common as icom

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


class CustomFSPath:
    """For testing fspath on unknown objects"""

    def __init__(self, path) -> None:
        self.path = path

    def __fspath__(self):
        return self.path


# Functions that consume a string path and return a string or path-like object
path_types = [str, CustomFSPath, Path]

try:
    from py.path import local as LocalPath

    path_types.append(LocalPath)
except ImportError:
    pass

HERE = os.path.abspath(os.path.dirname(__file__))


# https://github.com/cython/cython/issues/1720
class TestCommonIOCapabilities:
    data1 = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    def test_expand_user(self):
        filename = "~/sometest"
        expanded_name = icom._expand_user(filename)

        assert expanded_name != filename
        assert os.path.isabs(expanded_name)
        assert os.path.expanduser(filename) == expanded_name

    def test_expand_user_normal_path(self):
        filename = "/somefolder/sometest"
        expanded_name = icom._expand_user(filename)

        assert expanded_name == filename
        assert os.path.expanduser(filename) == expanded_name

    def test_stringify_path_pathlib(self):
        rel_path = icom.stringify_path(Path("."))
        assert rel_path == "."
        redundant_path = icom.stringify_path(Path("foo//bar"))
        assert redundant_path == os.path.join("foo", "bar")

    @td.skip_if_no("py.path")
    def test_stringify_path_localpath(self):
        path = os.path.join("foo", "bar")
        abs_path = os.path.abspath(path)
        lpath = LocalPath(path)
        assert icom.stringify_path(lpath) == abs_path

    def test_stringify_path_fspath(self):
        p = CustomFSPath("foo/bar.csv")
        result = icom.stringify_path(p)
        assert result == "foo/bar.csv"

    def test_stringify_file_and_path_like(self):
        # GH 38125: do not stringify file objects that are also path-like
        fsspec = pytest.importorskip("fsspec")
        with tm.ensure_clean() as path:
            with fsspec.open(f"file://{path}", mode="wb") as fsspec_obj:
                assert fsspec_obj == icom.stringify_path(fsspec_obj)

    @pytest.mark.parametrize("path_type", path_types)
    def test_infer_compression_from_path(self, compression_format, path_type):
        extension, expected = compression_format
        path = path_type("foo/bar.csv" + extension)
        compression = icom.infer_compression(path, compression="infer")
        assert compression == expected

    @pytest.mark.parametrize("path_type", [str, CustomFSPath, Path])
    def test_get_handle_with_path(self, path_type):
        # ignore LocalPath: it creates strange paths: /absolute/~/sometest
        with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
            filename = path_type("~/" + Path(tmp).name + "/sometest")
            with icom.get_handle(filename, "w") as handles:
                assert Path(handles.handle.name).is_absolute()
                assert os.path.expanduser(filename) == handles.handle.name

    def test_get_handle_with_buffer(self):
        with StringIO() as input_buffer:
            with icom.get_handle(input_buffer, "r") as handles:
                assert handles.handle == input_buffer
            assert not input_buffer.closed
        assert input_buffer.closed

    # Test that BytesIOWrapper(get_handle) returns correct amount of bytes every time
    def test_bytesiowrapper_returns_correct_bytes(self):
        # Test latin1, ucs-2, and ucs-4 chars
        data = """a,b,c
1,2,3
©,®,®
Look,a snake,🐍"""
        with icom.get_handle(StringIO(data), "rb", is_text=False) as handles:
            result = b""
            chunksize = 5
            while True:
                chunk = handles.handle.read(chunksize)
                # Make sure each chunk is correct amount of bytes
                assert len(chunk) <= chunksize
                if len(chunk) < chunksize:
                    # Can be less amount of bytes, but only at EOF
                    # which happens when read returns empty
                    assert len(handles.handle.read()) == 0
                    result += chunk
                    break
                result += chunk
            assert result == data.encode("utf-8")

    # Test that pyarrow can handle a file opened with get_handle
    def test_get_handle_pyarrow_compat(self):
        pa_csv = pytest.importorskip("pyarrow.csv")

        # Test latin1, ucs-2, and ucs-4 chars
        data = """a,b,c
1,2,3
©,®,®
Look,a snake,🐍"""
        expected = pd.DataFrame(
            {"a": ["1", "©", "Look"], "b": ["2", "®", "a snake"], "c": ["3", "®", "🐍"]}
        )
        s = StringIO(data)
        with icom.get_handle(s, "rb", is_text=False) as handles:
            df = pa_csv.read_csv(handles.handle).to_pandas()
            if pa_version_under19p0:
                expected = expected.astype("object")
            tm.assert_frame_equal(df, expected)
            assert not s.closed

    def test_iterator(self):
        with pd.read_csv(StringIO(self.data1), chunksize=1) as reader:
            result = pd.concat(reader, ignore_index=True)
        expected = pd.read_csv(StringIO(self.data1))
        tm.assert_frame_equal(result, expected)

        # GH12153
        with pd.read_csv(StringIO(self.data1), chunksize=1) as it:
            first = next(it)
            tm.assert_frame_equal(first, expected.iloc[[0]])
            tm.assert_frame_equal(pd.concat(it), expected.iloc[1:])

    @pytest.mark.parametrize(
        "reader, module, error_class, fn_ext",
        [
            (pd.read_csv, "os", FileNotFoundError, "csv"),
            (pd.read_fwf, "os", FileNotFoundError, "txt"),
            (pd.read_excel, "xlrd", FileNotFoundError, "xlsx"),
            (pd.read_feather, "pyarrow", OSError, "feather"),
            (pd.read_hdf, "tables", FileNotFoundError, "h5"),
            (pd.read_stata, "os", FileNotFoundError, "dta"),
            (pd.read_sas, "os", FileNotFoundError, "sas7bdat"),
            (pd.read_json, "os", FileNotFoundError, "json"),
            (pd.read_pickle, "os", FileNotFoundError, "pickle"),
        ],
    )
    def test_read_non_existent(self, reader, module, error_class, fn_ext):
        pytest.importorskip(module)

        path = os.path.join(HERE, "data", "does_not_exist." + fn_ext)
        msg1 = rf"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2 = rf"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3 = "Expected object or value"
        msg4 = "path_or_buf needs to be a string file path or file-like"
        msg5 = (
            rf"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: "
            rf"'.+does_not_exist\.{fn_ext}'"
        )
        msg6 = rf"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7 = (
            rf"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        )
        msg8 = rf"Failed to open local file.+does_not_exist\.{fn_ext}"

        with pytest.raises(
            error_class,
            match=rf"({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})",
        ):
            reader(path)

    @pytest.mark.parametrize(
        "method, module, error_class, fn_ext",
        [
            (pd.DataFrame.to_csv, "os", OSError, "csv"),
            (pd.DataFrame.to_html, "os", OSError, "html"),
            (pd.DataFrame.to_excel, "xlrd", OSError, "xlsx"),
            (pd.DataFrame.to_feather, "pyarrow", OSError, "feather"),
            (pd.DataFrame.to_parquet, "pyarrow", OSError, "parquet"),
            (pd.DataFrame.to_stata, "os", OSError, "dta"),
            (pd.DataFrame.to_json, "os", OSError, "json"),
            (pd.DataFrame.to_pickle, "os", OSError, "pickle"),
        ],
    )
    # NOTE: Missing parent directory for pd.DataFrame.to_hdf is handled by PyTables
    def test_write_missing_parent_directory(self, method, module, error_class, fn_ext):
        pytest.importorskip(module)

        dummy_frame = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5]})

        path = os.path.join(HERE, "data", "missing_folder", "does_not_exist." + fn_ext)

        with pytest.raises(
            error_class,
            match=r"Cannot save file into a non-existent directory: .*missing_folder",
        ):
            method(dummy_frame, path)

    @pytest.mark.parametrize(
        "reader, module, error_class, fn_ext",
        [
            (pd.read_csv, "os", FileNotFoundError, "csv"),
            (pd.read_table, "os", FileNotFoundError, "csv"),
            (pd.read_fwf, "os", FileNotFoundError, "txt"),
            (pd.read_excel, "xlrd", FileNotFoundError, "xlsx"),
            (pd.read_feather, "pyarrow", OSError, "feather"),
            (pd.read_hdf, "tables", FileNotFoundError, "h5"),
            (pd.read_stata, "os", FileNotFoundError, "dta"),
            (pd.read_sas, "os", FileNotFoundError, "sas7bdat"),
            (pd.read_json, "os", FileNotFoundError, "json"),
            (pd.read_pickle, "os", FileNotFoundError, "pickle"),
        ],
    )
    def test_read_expands_user_home_dir(
        self, reader, module, error_class, fn_ext, monkeypatch
    ):
        pytest.importorskip(module)

        path = os.path.join("~", "does_not_exist." + fn_ext)
        monkeypatch.setattr(icom, "_expand_user", lambda x: os.path.join("foo", x))

        msg1 = rf"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2 = rf"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3 = "Unexpected character found when decoding 'false'"
        msg4 = "path_or_buf needs to be a string file path or file-like"
        msg5 = (
            rf"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: "
            rf"'.+does_not_exist\.{fn_ext}'"
        )
        msg6 = rf"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7 = (
            rf"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        )
        msg8 = rf"Failed to open local file.+does_not_exist\.{fn_ext}"

        with pytest.raises(
            error_class,
            match=rf"({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})",
        ):
            reader(path)

    @pytest.mark.parametrize(
        "reader, module, path",
        [
            (pd.read_csv, "os", ("io", "data", "csv", "iris.csv")),
            (pd.read_table, "os", ("io", "data", "csv", "iris.csv")),
            (
                pd.read_fwf,
                "os",
                ("io", "data", "fixed_width", "fixed_width_format.txt"),
            ),
            (pd.read_excel, "xlrd", ("io", "data", "excel", "test1.xlsx")),
            (
                pd.read_feather,
                "pyarrow",
                ("io", "data", "feather", "feather-0_3_1.feather"),
            ),
            pytest.param(
                pd.read_hdf,
                "tables",
                ("io", "data", "legacy_hdf", "datetimetz_object.h5"),
                # cleaned-up in https://github.com/pandas-dev/pandas/pull/57387 on main
                marks=pytest.mark.xfail(reason="TODO(infer_string)", strict=False),
            ),
            (pd.read_stata, "os", ("io", "data", "stata", "stata10_115.dta")),
            (pd.read_sas, "os", ("io", "sas", "data", "test1.sas7bdat")),
            (pd.read_json, "os", ("io", "json", "data", "tsframe_v012.json")),
            (
                pd.read_pickle,
                "os",
                ("io", "data", "pickle", "categorical.0.25.0.pickle"),
            ),
        ],
    )
    def test_read_fspath_all(self, reader, module, path, datapath):
        pytest.importorskip(module)
        path = datapath(*path)

        mypath = CustomFSPath(path)
        result = reader(mypath)
        expected = reader(path)

        if path.endswith(".pickle"):
            # categorical
            tm.assert_categorical_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "writer_name, writer_kwargs, module",
        [
            ("to_csv", {}, "os"),
            ("to_excel", {"engine": "openpyxl"}, "openpyxl"),
            ("to_feather", {}, "pyarrow"),
            ("to_html", {}, "os"),
            ("to_json", {}, "os"),
            ("to_latex", {}, "os"),
            ("to_pickle", {}, "os"),
            ("to_stata", {"time_stamp": pd.to_datetime("2019-01-01 00:00")}, "os"),
        ],
    )
    def test_write_fspath_all(self, writer_name, writer_kwargs, module):
        if writer_name in ["to_latex"]:  # uses Styler implementation
            pytest.importorskip("jinja2")
        p1 = tm.ensure_clean("string")
        p2 = tm.ensure_clean("fspath")
        df = pd.DataFrame({"A": [1, 2]})

        with p1 as string, p2 as fspath:
            pytest.importorskip(module)
            mypath = CustomFSPath(fspath)
            writer = getattr(df, writer_name)

            writer(string, **writer_kwargs)
            writer(mypath, **writer_kwargs)
            with open(string, "rb") as f_str, open(fspath, "rb") as f_path:
                if writer_name == "to_excel":
                    # binary representation of excel contains time creation
                    # data that causes flaky CI failures
                    result = pd.read_excel(f_str, **writer_kwargs)
                    expected = pd.read_excel(f_path, **writer_kwargs)
                    tm.assert_frame_equal(result, expected)
                else:
                    result = f_str.read()
                    expected = f_path.read()
                    assert result == expected

    def test_write_fspath_hdf5(self):
        # Same test as write_fspath_all, except HDF5 files aren't
        # necessarily byte-for-byte identical for a given dataframe, so we'll
        # have to read and compare equality
        pytest.importorskip("tables")

        df = pd.DataFrame({"A": [1, 2]})
        p1 = tm.ensure_clean("string")
        p2 = tm.ensure_clean("fspath")

        with p1 as string, p2 as fspath:
            mypath = CustomFSPath(fspath)
            df.to_hdf(mypath, key="bar")
            df.to_hdf(string, key="bar")

            result = pd.read_hdf(fspath, key="bar")
            expected = pd.read_hdf(string, key="bar")

        tm.assert_frame_equal(result, expected)


@pytest.fixture
def mmap_file(datapath):
    return datapath("io", "data", "csv", "test_mmap.csv")


class TestMMapWrapper:
    def test_constructor_bad_file(self, mmap_file):
        non_file = StringIO("I am not a file")
        non_file.fileno = lambda: -1

        # the error raised is different on Windows
        if is_platform_windows():
            msg = "The parameter is incorrect"
            err = OSError
        else:
            msg = "[Errno 22]"
            err = mmap.error

        with pytest.raises(err, match=msg):
            icom._maybe_memory_map(non_file, True)

        with open(mmap_file, encoding="utf-8") as target:
            pass

        msg = "I/O operation on closed file"
        with pytest.raises(ValueError, match=msg):
            icom._maybe_memory_map(target, True)

    def test_next(self, mmap_file):
        with open(mmap_file, encoding="utf-8") as target:
            lines = target.readlines()

            with icom.get_handle(
                target, "r", is_text=True, memory_map=True
            ) as wrappers:
                wrapper = wrappers.handle
                assert isinstance(wrapper.buffer.buffer, mmap.mmap)

                for line in lines:
                    next_line = next(wrapper)
                    assert next_line.strip() == line.strip()

                with pytest.raises(StopIteration, match=r"^$"):
                    next(wrapper)

    def test_unknown_engine(self):
        with tm.ensure_clean() as path:
            df = pd.DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=pd.Index(list("ABCD")),
                index=pd.Index([f"i-{i}" for i in range(30)]),
            )
            df.to_csv(path)
            with pytest.raises(ValueError, match="Unknown engine"):
                pd.read_csv(path, engine="pyt")

    def test_binary_mode(self):
        """
        'encoding' shouldn't be passed to 'open' in binary mode.

        GH 35058
        """
        with tm.ensure_clean() as path:
            df = pd.DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=pd.Index(list("ABCD")),
                index=pd.Index([f"i-{i}" for i in range(30)]),
            )
            df.to_csv(path, mode="w+b")
            tm.assert_frame_equal(df, pd.read_csv(path, index_col=0))

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
    @pytest.mark.parametrize("compression_", ["bz2", "xz"])
    def test_warning_missing_utf_bom(self, encoding, compression_):
        """
        bz2 and xz do not write the byte order mark (BOM) for utf-16/32.

        https://stackoverflow.com/questions/55171439

        GH 35681
        """
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD")),
            index=pd.Index([f"i-{i}" for i in range(30)]),
        )
        with tm.ensure_clean() as path:
            with tm.assert_produces_warning(UnicodeWarning):
                df.to_csv(path, compression=compression_, encoding=encoding)

            # reading should fail (otherwise we wouldn't need the warning)
            msg = (
                r"UTF-\d+ stream does not start with BOM|"
                r"'utf-\d+' codec can't decode byte"
            )
            with pytest.raises(UnicodeError, match=msg):
                pd.read_csv(path, compression=compression_, encoding=encoding)


def test_is_fsspec_url():
    assert icom.is_fsspec_url("gcs://pandas/somethingelse.com")
    assert icom.is_fsspec_url("gs://pandas/somethingelse.com")
    # the following is the only remote URL that is handled without fsspec
    assert not icom.is_fsspec_url("http://pandas/somethingelse.com")
    assert not icom.is_fsspec_url("random:pandas/somethingelse.com")
    assert not icom.is_fsspec_url("/local/path")
    assert not icom.is_fsspec_url("relative/local/path")
    # fsspec URL in string should not be recognized
    assert not icom.is_fsspec_url("this is not fsspec://url")
    assert not icom.is_fsspec_url("{'url': 'gs://pandas/somethingelse.com'}")
    # accept everything that conforms to RFC 3986 schema
    assert icom.is_fsspec_url("RFC-3986+compliant.spec://something")


@pytest.mark.parametrize("encoding", [None, "utf-8"])
@pytest.mark.parametrize("format", ["csv", "json"])
def test_codecs_encoding(encoding, format):
    # GH39247
    expected = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )
    with tm.ensure_clean() as path:
        with codecs.open(path, mode="w", encoding=encoding) as handle:
            getattr(expected, f"to_{format}")(handle)
        with codecs.open(path, mode="r", encoding=encoding) as handle:
            if format == "csv":
                df = pd.read_csv(handle, index_col=0)
            else:
                df = pd.read_json(handle)
    tm.assert_frame_equal(expected, df)


def test_codecs_get_writer_reader():
    # GH39247
    expected = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )
    with tm.ensure_clean() as path:
        with open(path, "wb") as handle:
            with codecs.getwriter("utf-8")(handle) as encoded:
                expected.to_csv(encoded)
        with open(path, "rb") as handle:
            with codecs.getreader("utf-8")(handle) as encoded:
                df = pd.read_csv(encoded, index_col=0)
    tm.assert_frame_equal(expected, df)


@pytest.mark.parametrize(
    "io_class,mode,msg",
    [
        (BytesIO, "t", "a bytes-like object is required, not 'str'"),
        (StringIO, "b", "string argument expected, got 'bytes'"),
    ],
)
def test_explicit_encoding(io_class, mode, msg):
    # GH39247; this test makes sure that if a user provides mode="*t" or "*b",
    # it is used. In the case of this test it leads to an error as intentionally the
    # wrong mode is requested
    expected = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD")),
        index=pd.Index([f"i-{i}" for i in range(30)]),
    )
    with io_class() as buffer:
        with pytest.raises(TypeError, match=msg):
            expected.to_csv(buffer, mode=f"w{mode}")


@pytest.mark.parametrize("encoding_errors", [None, "strict", "replace"])
@pytest.mark.parametrize("format", ["csv", "json"])
def test_encoding_errors(encoding_errors, format):
    # GH39450
    msg = "'utf-8' codec can't decode byte"
    bad_encoding = b"\xe4"

    if format == "csv":
        content = b"," + bad_encoding + b"\n" + bad_encoding * 2 + b"," + bad_encoding
        reader = partial(pd.read_csv, index_col=0)
    else:
        content = (
            b'{"'
            + bad_encoding * 2
            + b'": {"'
            + bad_encoding
            + b'":"'
            + bad_encoding
            + b'"}}'
        )
        reader = partial(pd.read_json, orient="index")
    with tm.ensure_clean() as path:
        file = Path(path)
        file.write_bytes(content)

        if encoding_errors != "replace":
            with pytest.raises(UnicodeDecodeError, match=msg):
                reader(path, encoding_errors=encoding_errors)
        else:
            df = reader(path, encoding_errors=encoding_errors)
            decoded = bad_encoding.decode(errors=encoding_errors)
            expected = pd.DataFrame({decoded: [decoded]}, index=[decoded * 2])
            tm.assert_frame_equal(df, expected)


def test_bad_encdoing_errors():
    # GH 39777
    with tm.ensure_clean() as path:
        with pytest.raises(LookupError, match="unknown error handler name"):
            icom.get_handle(path, "w", errors="bad")


def test_errno_attribute():
    # GH 13872
    with pytest.raises(FileNotFoundError, match="\\[Errno 2\\]") as err:
        pd.read_csv("doesnt_exist")
        assert err.errno == errno.ENOENT


def test_fail_mmap():
    with pytest.raises(UnsupportedOperation, match="fileno"):
        with BytesIO() as buffer:
            icom.get_handle(buffer, "rb", memory_map=True)


def test_close_on_error():
    # GH 47136
    class TestError:
        def close(self):
            raise OSError("test")

    with pytest.raises(OSError, match="test"):
        with BytesIO() as buffer:
            with icom.get_handle(buffer, "rb") as handles:
                handles.created_handles.append(TestError())


@pytest.mark.parametrize(
    "reader",
    [
        pd.read_csv,
        pd.read_fwf,
        pd.read_excel,
        pd.read_feather,
        pd.read_hdf,
        pd.read_stata,
        pd.read_sas,
        pd.read_json,
        pd.read_pickle,
    ],
)
def test_pickle_reader(reader):
    # GH 22265
    with BytesIO() as buffer:
        pickle.dump(reader, buffer)