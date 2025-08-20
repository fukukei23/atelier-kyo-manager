
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_laplace.py ===
from sympy.integrals.laplace import (
    laplace_transform, inverse_laplace_transform,
    LaplaceTransform, InverseLaplaceTransform,
    _laplace_deep_collect, laplace_correspondence,
    laplace_initial_conds)
from sympy.core.function import Function, expand_mul
from sympy.core import EulerGamma, Subs, Derivative, diff
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I, oo, pi
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.simplify.simplify import simplify
from sympy.functions.elementary.complexes import Abs, re
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh, coth, asinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import atan, cos, sin
from sympy.logic.boolalg import And
from sympy.functions.special.gamma_functions import (
    lowergamma, gamma, uppergamma)
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import lerchphi
from sympy.functions.special.error_functions import (
    fresnelc, fresnels, erf, erfc, Ei, Ci, expint, E1)
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.testing.pytest import slow, warns_deprecated_sympy
from sympy.matrices import Matrix, eye
from sympy.abc import s


@slow
def test_laplace_transform():
    LT = laplace_transform
    ILT = inverse_laplace_transform
    a, b, c = symbols('a, b, c', positive=True)
    np = symbols('np', integer=True, positive=True)
    t, w, x = symbols('t, w, x')
    f = Function('f')
    F = Function('F')
    g = Function('g')
    y = Function('y')
    Y = Function('Y')

    # Test helper functions
    assert (
        _laplace_deep_collect(exp((t+a)*(t+b)) +
                              besselj(2, exp((t+a)*(t+b)-t**2)), t) ==
        exp(a*b + t**2 + t*(a + b)) + besselj(2, exp(a*b + t*(a + b))))
    L = laplace_transform(diff(y(t), t, 3), t, s, noconds=True)
    L = laplace_correspondence(L, {y: Y})
    L = laplace_initial_conds(L, t, {y: [2, 4, 8, 16, 32]})
    assert L == s**3*Y(s) - 2*s**2 - 4*s - 8
    # Test whether `noconds=True` in `doit`:
    assert (2*LaplaceTransform(exp(t), t, s) - 1).doit() == -1 + 2/(s - 1)
    assert (LT(a*t+t**2+t**(S(5)/2), t, s) ==
            (a/s**2 + 2/s**3 + 15*sqrt(pi)/(8*s**(S(7)/2)), 0, True))
    assert LT(b/(t+a), t, s) == (-b*exp(-a*s)*Ei(-a*s), 0, True)
    assert (LT(1/sqrt(t+a), t, s) ==
            (sqrt(pi)*sqrt(1/s)*exp(a*s)*erfc(sqrt(a)*sqrt(s)), 0, True))
    assert (LT(sqrt(t)/(t+a), t, s) ==
            (-pi*sqrt(a)*exp(a*s)*erfc(sqrt(a)*sqrt(s)) + sqrt(pi)*sqrt(1/s),
             0, True))
    assert (LT((t+a)**(-S(3)/2), t, s) ==
            (-2*sqrt(pi)*sqrt(s)*exp(a*s)*erfc(sqrt(a)*sqrt(s)) + 2/sqrt(a),
             0, True))
    assert (LT(t**(S(1)/2)*(t+a)**(-1), t, s) ==
            (-pi*sqrt(a)*exp(a*s)*erfc(sqrt(a)*sqrt(s)) + sqrt(pi)*sqrt(1/s),
             0, True))
    assert (LT(1/(a*sqrt(t) + t**(3/2)), t, s) ==
            (pi*sqrt(a)*exp(a*s)*erfc(sqrt(a)*sqrt(s)), 0, True))
    assert (LT((t+a)**b, t, s) ==
            (s**(-b - 1)*exp(-a*s)*uppergamma(b + 1, a*s), 0, True))
    assert LT(t**5/(t+a), t, s) == (120*a**5*uppergamma(-5, a*s), 0, True)
    assert LT(exp(t), t, s) == (1/(s - 1), 1, True)
    assert LT(exp(2*t), t, s) == (1/(s - 2), 2, True)
    assert LT(exp(a*t), t, s) == (1/(s - a), a, True)
    assert LT(exp(a*(t-b)), t, s) == (exp(-a*b)/(-a + s), a, True)
    assert LT(t*exp(-a*(t)), t, s) == ((a + s)**(-2), -a, True)
    assert LT(t*exp(-a*(t-b)), t, s) == (exp(a*b)/(a + s)**2, -a, True)
    assert LT(b*t*exp(-a*t), t, s) == (b/(a + s)**2, -a, True)
    assert LT(exp(-a*exp(-t)), t, s) == (lowergamma(s, a)/a**s, 0, True)
    assert LT(exp(-a*exp(t)), t, s) == (a**s*uppergamma(-s, a), 0, True)
    assert (LT(t**(S(7)/4)*exp(-8*t)/gamma(S(11)/4), t, s) ==
            ((s + 8)**(-S(11)/4), -8, True))
    assert (LT(t**(S(3)/2)*exp(-8*t), t, s) ==
            (3*sqrt(pi)/(4*(s + 8)**(S(5)/2)), -8, True))
    assert LT(t**a*exp(-a*t), t, s) == ((a+s)**(-a-1)*gamma(a+1), -a, True)
    assert (LT(b*exp(-a*t**2), t, s) ==
            (sqrt(pi)*b*exp(s**2/(4*a))*erfc(s/(2*sqrt(a)))/(2*sqrt(a)),
             0, True))
    assert (LT(exp(-2*t**2), t, s) ==
            (sqrt(2)*sqrt(pi)*exp(s**2/8)*erfc(sqrt(2)*s/4)/4, 0, True))
    assert (LT(b*exp(2*t**2), t, s) ==
            (b*LaplaceTransform(exp(2*t**2), t, s), -oo, True))
    assert (LT(t*exp(-a*t**2), t, s) ==
            (1/(2*a) - s*erfc(s/(2*sqrt(a)))/(4*sqrt(pi)*a**(S(3)/2)),
             0, True))
    assert (LT(exp(-a/t), t, s) ==
            (2*sqrt(a)*sqrt(1/s)*besselk(1, 2*sqrt(a)*sqrt(s)), 0, True))
    assert LT(sqrt(t)*exp(-a/t), t, s, simplify=True) == (
        sqrt(pi)*(sqrt(a)*sqrt(s) + 1/S(2))*sqrt(s**(-3)) *
        exp(-2*sqrt(a)*sqrt(s)), 0, True)
    assert (LT(exp(-a/t)/sqrt(t), t, s) ==
            (sqrt(pi)*sqrt(1/s)*exp(-2*sqrt(a)*sqrt(s)), 0, True))
    assert (LT(exp(-a/t)/(t*sqrt(t)), t, s) ==
            (sqrt(pi)*sqrt(1/a)*exp(-2*sqrt(a)*sqrt(s)), 0, True))
    # TODO: rules with sqrt(a*t) and sqrt(a/t) have stopped working after
    #       changes to as_base_exp
    # assert (
    #     LT(exp(-2*sqrt(a*t)), t, s) ==
    #     (1/s - sqrt(pi)*sqrt(a) * exp(a/s)*erfc(sqrt(a)*sqrt(1/s)) /
    #      s**(S(3)/2), 0, True))
    # assert LT(exp(-2*sqrt(a*t))/sqrt(t), t, s) == (
    #     exp(a/s)*erfc(sqrt(a) * sqrt(1/s))*(sqrt(pi)*sqrt(1/s)), 0, True)
    assert (LT(t**4*exp(-2/t), t, s) ==
            (8*sqrt(2)*(1/s)**(S(5)/2)*besselk(5, 2*sqrt(2)*sqrt(s)),
             0, True))
    assert LT(sinh(a*t), t, s) == (a/(-a**2 + s**2), a, True)
    assert (LT(b*sinh(a*t)**2, t, s) ==
            (2*a**2*b/(-4*a**2*s + s**3), 2*a, True))
    assert (LT(b*sinh(a*t)**2, t, s, simplify=True) ==
            (2*a**2*b/(s*(-4*a**2 + s**2)), 2*a, True))
    # The following line confirms that issue #21202 is solved
    assert LT(cosh(2*t), t, s) == (s/(-4 + s**2), 2, True)
    assert LT(cosh(a*t), t, s) == (s/(-a**2 + s**2), a, True)
    assert (LT(cosh(a*t)**2, t, s, simplify=True) ==
            ((2*a**2 - s**2)/(s*(4*a**2 - s**2)), 2*a, True))
    assert (LT(sinh(x+3), x, s, simplify=True) ==
            ((s*sinh(3) + cosh(3))/(s**2 - 1), 1, True))
    L, _, _ = LT(42*sin(w*t+x)**2, t, s)
    assert (
        L -
        21*(s**2 + s*(-s*cos(2*x) + 2*w*sin(2*x)) +
            4*w**2)/(s*(s**2 + 4*w**2))).simplify() == 0
    # The following line replaces the old test test_issue_7173()
    assert LT(sinh(a*t)*cosh(a*t), t, s, simplify=True) == (a/(-4*a**2 + s**2),
                                                            2*a, True)
    assert LT(sinh(a*t)/t, t, s) == (log((a + s)/(-a + s))/2, a, True)
    assert (LT(t**(-S(3)/2)*sinh(a*t), t, s) ==
            (-sqrt(pi)*(sqrt(-a + s) - sqrt(a + s)), a, True))
    # assert (LT(sinh(2*sqrt(a*t)), t, s) ==
    #         (sqrt(pi)*sqrt(a)*exp(a/s)/s**(S(3)/2), 0, True))
    # assert (LT(sqrt(t)*sinh(2*sqrt(a*t)), t, s, simplify=True) ==
    #         ((-sqrt(a)*s**(S(5)/2) + sqrt(pi)*s**2*(2*a + s)*exp(a/s) *
    #           erf(sqrt(a)*sqrt(1/s))/2)/s**(S(9)/2), 0, True))
    # assert (LT(sinh(2*sqrt(a*t))/sqrt(t), t, s) ==
    #         (sqrt(pi)*exp(a/s)*erf(sqrt(a)*sqrt(1/s))/sqrt(s), 0, True))
    # assert (LT(sinh(sqrt(a*t))**2/sqrt(t), t, s) ==
    #         (sqrt(pi)*(exp(a/s) - 1)/(2*sqrt(s)), 0, True))
    assert (LT(t**(S(3)/7)*cosh(a*t), t, s) ==
            (((a + s)**(-S(10)/7) + (-a+s)**(-S(10)/7))*gamma(S(10)/7)/2,
             a, True))
    # assert (LT(cosh(2*sqrt(a*t)), t, s) ==
    #         (sqrt(pi)*sqrt(a)*exp(a/s)*erf(sqrt(a)*sqrt(1/s))/s**(S(3)/2) +
    #          1/s, 0, True))
    # assert (LT(sqrt(t)*cosh(2*sqrt(a*t)), t, s) ==
    #         (sqrt(pi)*(a + s/2)*exp(a/s)/s**(S(5)/2), 0, True))
    # assert (LT(cosh(2*sqrt(a*t))/sqrt(t), t, s) ==
    #         (sqrt(pi)*exp(a/s)/sqrt(s), 0, True))
    # assert (LT(cosh(sqrt(a*t))**2/sqrt(t), t, s) ==
    #         (sqrt(pi)*(exp(a/s) + 1)/(2*sqrt(s)), 0, True))
    assert LT(log(t), t, s, simplify=True) == (
        (-log(s) - EulerGamma)/s, 0, True)
    assert (LT(-log(t/a), t, s, simplify=True) ==
            ((log(a) + log(s) + EulerGamma)/s, 0, True))
    assert LT(log(1+a*t), t, s) == (-exp(s/a)*Ei(-s/a)/s, 0, True)
    assert (LT(log(t+a), t, s, simplify=True) ==
            ((s*log(a) - exp(s/a)*Ei(-s/a))/s**2, 0, True))
    assert (LT(log(t)/sqrt(t), t, s, simplify=True) ==
            (sqrt(pi)*(-log(s) - log(4) - EulerGamma)/sqrt(s), 0, True))
    assert (LT(t**(S(5)/2)*log(t), t, s, simplify=True) ==
            (sqrt(pi)*(-15*log(s) - log(1073741824) - 15*EulerGamma + 46) /
             (8*s**(S(7)/2)), 0, True))
    assert (LT(t**3*log(t), t, s, noconds=True, simplify=True) -
            6*(-log(s) - S.EulerGamma + S(11)/6)/s**4).simplify() == S.Zero
    assert (LT(log(t)**2, t, s, simplify=True) ==
            (((log(s) + EulerGamma)**2 + pi**2/6)/s, 0, True))
    assert (LT(exp(-a*t)*log(t), t, s, simplify=True) ==
            ((-log(a + s) - EulerGamma)/(a + s), -a, True))
    assert LT(sin(a*t), t, s) == (a/(a**2 + s**2), 0, True)
    assert (LT(Abs(sin(a*t)), t, s) ==
            (a*coth(pi*s/(2*a))/(a**2 + s**2), 0, True))
    assert LT(sin(a*t)/t, t, s) == (atan(a/s), 0, True)
    assert LT(sin(a*t)**2/t, t, s) == (log(4*a**2/s**2 + 1)/4, 0, True)
    assert (LT(sin(a*t)**2/t**2, t, s) ==
            (a*atan(2*a/s) - s*log(4*a**2/s**2 + 1)/4, 0, True))
    # assert (LT(sin(2*sqrt(a*t)), t, s) ==
    #         (sqrt(pi)*sqrt(a)*exp(-a/s)/s**(S(3)/2), 0, True))
    # assert LT(sin(2*sqrt(a*t))/t, t, s) == (pi*erf(sqrt(a)*sqrt(1/s)), 0, True)
    assert LT(cos(a*t), t, s) == (s/(a**2 + s**2), 0, True)
    assert (LT(cos(a*t)**2, t, s) ==
            ((2*a**2 + s**2)/(s*(4*a**2 + s**2)), 0, True))
    # assert (LT(sqrt(t)*cos(2*sqrt(a*t)), t, s, simplify=True) ==
    #         (sqrt(pi)*(-a + s/2)*exp(-a/s)/s**(S(5)/2), 0, True))
    # assert (LT(cos(2*sqrt(a*t))/sqrt(t), t, s) ==
    #         (sqrt(pi)*sqrt(1/s)*exp(-a/s), 0, True))
    assert (LT(sin(a*t)*sin(b*t), t, s) ==
            (2*a*b*s/((s**2 + (a - b)**2)*(s**2 + (a + b)**2)), 0, True))
    assert (LT(cos(a*t)*sin(b*t), t, s) ==
            (b*(-a**2 + b**2 + s**2)/((s**2 + (a - b)**2)*(s**2 + (a + b)**2)),
             0, True))
    assert (LT(cos(a*t)*cos(b*t), t, s) ==
            (s*(a**2 + b**2 + s**2)/((s**2 + (a - b)**2)*(s**2 + (a + b)**2)),
             0, True))
    assert (LT(-a*t*cos(a*t) + sin(a*t), t, s, simplify=True) ==
            (2*a**3/(a**4 + 2*a**2*s**2 + s**4), 0, True))
    assert LT(c*exp(-b*t)*sin(a*t), t, s) == (a *
                                              c/(a**2 + (b + s)**2), -b, True)
    assert LT(c*exp(-b*t)*cos(a*t), t, s) == (c*(b + s)/(a**2 + (b + s)**2),
                                              -b, True)
    L, plane, cond = LT(cos(x + 3), x, s, simplify=True)
    assert plane == 0
    assert L - (s*cos(3) - sin(3))/(s**2 + 1) == 0
    # Error functions (laplace7.pdf)
    assert LT(erf(a*t), t, s) == (exp(s**2/(4*a**2))*erfc(s/(2*a))/s, 0, True)
    # assert LT(erf(sqrt(a*t)), t, s) == (sqrt(a)/(s*sqrt(a + s)), 0, True)
    # assert (LT(exp(a*t)*erf(sqrt(a*t)), t, s, simplify=True) ==
    #         (-sqrt(a)/(sqrt(s)*(a - s)), a, True))
    # assert (LT(erf(sqrt(a/t)/2), t, s, simplify=True) ==
    #         (1/s - exp(-sqrt(a)*sqrt(s))/s, 0, True))
    # assert (LT(erfc(sqrt(a*t)), t, s, simplify=True) ==
    #         (-sqrt(a)/(s*sqrt(a + s)) + 1/s, -a, True))
    # assert (LT(exp(a*t)*erfc(sqrt(a*t)), t, s) ==
    #         (1/(sqrt(a)*sqrt(s) + s), 0, True))
    # assert LT(erfc(sqrt(a/t)/2), t, s) == (exp(-sqrt(a)*sqrt(s))/s, 0, True)
    # Bessel functions (laplace8.pdf)
    assert LT(besselj(0, a*t), t, s) == (1/sqrt(a**2 + s**2), 0, True)
    assert (LT(besselj(1, a*t), t, s, simplify=True) ==
            (a/(a**2 + s**2 + s*sqrt(a**2 + s**2)), 0, True))
    assert (LT(besselj(2, a*t), t, s, simplify=True) ==
            (a**2/(sqrt(a**2 + s**2)*(s + sqrt(a**2 + s**2))**2), 0, True))
    assert (LT(t*besselj(0, a*t), t, s) ==
            (s/(a**2 + s**2)**(S(3)/2), 0, True))
    assert (LT(t*besselj(1, a*t), t, s) ==
            (a/(a**2 + s**2)**(S(3)/2), 0, True))
    assert (LT(t**2*besselj(2, a*t), t, s) ==
            (3*a**2/(a**2 + s**2)**(S(5)/2), 0, True))
    # assert LT(besselj(0, 2*sqrt(a*t)), t, s) == (exp(-a/s)/s, 0, True)
    # assert (LT(t**(S(3)/2)*besselj(3, 2*sqrt(a*t)), t, s) ==
    #         (a**(S(3)/2)*exp(-a/s)/s**4, 0, True))
    assert (LT(besselj(0, a*sqrt(t**2+b*t)), t, s, simplify=True) ==
            (exp(b*(s - sqrt(a**2 + s**2)))/sqrt(a**2 + s**2), 0, True))
    assert LT(besseli(0, a*t), t, s) == (1/sqrt(-a**2 + s**2), a, True)
    assert (LT(besseli(1, a*t), t, s, simplify=True) ==
            (a/(-a**2 + s**2 + s*sqrt(-a**2 + s**2)), a, True))
    assert (LT(besseli(2, a*t), t, s, simplify=True) ==
            (a**2/(sqrt(-a**2 + s**2)*(s + sqrt(-a**2 + s**2))**2), a, True))
    assert LT(t*besseli(0, a*t), t, s) == (s/(-a**2 + s**2)**(S(3)/2), a, True)
    assert LT(t*besseli(1, a*t), t, s) == (a/(-a**2 + s**2)**(S(3)/2), a, True)
    assert (LT(t**2*besseli(2, a*t), t, s) ==
            (3*a**2/(-a**2 + s**2)**(S(5)/2), a, True))
    # assert (LT(t**(S(3)/2)*besseli(3, 2*sqrt(a*t)), t, s) ==
    #         (a**(S(3)/2)*exp(a/s)/s**4, 0, True))
    assert (LT(bessely(0, a*t), t, s) ==
            (-2*asinh(s/a)/(pi*sqrt(a**2 + s**2)), 0, True))
    assert (LT(besselk(0, a*t), t, s) ==
            (log((s + sqrt(-a**2 + s**2))/a)/sqrt(-a**2 + s**2), -a, True))
    assert (LT(sin(a*t)**4, t, s, simplify=True) ==
            (24*a**4/(s*(64*a**4 + 20*a**2*s**2 + s**4)), 0, True))
    # Test general rules and unevaluated forms
    # These all also test whether issue #7219 is solved.
    assert LT(Heaviside(t-1)*cos(t-1), t, s) == (s*exp(-s)/(s**2 + 1), 0, True)
    assert LT(a*f(t), t, w) == (a*LaplaceTransform(f(t), t, w), -oo, True)
    assert (LT(a*Heaviside(t+1)*f(t+1), t, s) ==
            (a*LaplaceTransform(f(t + 1), t, s), -oo, True))
    assert (LT(a*Heaviside(t-1)*f(t-1), t, s) ==
            (a*LaplaceTransform(f(t), t, s)*exp(-s), -oo, True))
    assert (LT(b*f(t/a), t, s) ==
            (a*b*LaplaceTransform(f(t), t, a*s), -oo, True))
    assert LT(exp(-f(x)*t), t, s) == (1/(s + f(x)), -re(f(x)), True)
    assert (LT(exp(-a*t)*f(t), t, s) ==
            (LaplaceTransform(f(t), t, a + s), -oo, True))
    # assert (LT(exp(-a*t)*erfc(sqrt(b/t)/2), t, s) ==
    #         (exp(-sqrt(b)*sqrt(a + s))/(a + s), -a, True))
    assert (LT(sinh(a*t)*f(t), t, s) ==
            (LaplaceTransform(f(t), t, -a + s)/2 -
             LaplaceTransform(f(t), t, a + s)/2, -oo, True))
    assert (LT(sinh(a*t)*t, t, s, simplify=True) ==
            (2*a*s/(a**4 - 2*a**2*s**2 + s**4), a, True))
    assert (LT(cosh(a*t)*f(t), t, s) ==
            (LaplaceTransform(f(t), t, -a + s)/2 +
             LaplaceTransform(f(t), t, a + s)/2, -oo, True))
    assert (LT(cosh(a*t)*t, t, s, simplify=True) ==
            (1/(2*(a + s)**2) + 1/(2*(a - s)**2), a, True))
    assert (LT(sin(a*t)*f(t), t, s, simplify=True) ==
            (I*(-LaplaceTransform(f(t), t, -I*a + s) +
                LaplaceTransform(f(t), t, I*a + s))/2, -oo, True))
    assert (LT(sin(f(t)), t, s) ==
            (LaplaceTransform(sin(f(t)), t, s), -oo, True))
    assert (LT(sin(a*t)*t, t, s, simplify=True) ==
            (2*a*s/(a**4 + 2*a**2*s**2 + s**4), 0, True))
    assert (LT(cos(a*t)*f(t), t, s) ==
            (LaplaceTransform(f(t), t, -I*a + s)/2 +
             LaplaceTransform(f(t), t, I*a + s)/2, -oo, True))
    assert (LT(cos(a*t)*t, t, s, simplify=True) ==
            ((-a**2 + s**2)/(a**4 + 2*a**2*s**2 + s**4), 0, True))
    L, plane, _ = LT(sin(a*t+b)**2*f(t), t, s)
    assert plane == -oo
    assert (
        -L + (
            LaplaceTransform(f(t), t, s)/2 -
            LaplaceTransform(f(t), t, -2*I*a + s)*exp(2*I*b)/4 -
            LaplaceTransform(f(t), t, 2*I*a + s)*exp(-2*I*b)/4)) == 0
    L = LT(sin(a*t+b)**2*f(t), t, s, noconds=True)
    assert (
        laplace_correspondence(L, {f: F}) ==
        F(s)/2 - F(-2*I*a + s)*exp(2*I*b)/4 -
        F(2*I*a + s)*exp(-2*I*b)/4)
    L, plane, _ = LT(sin(a*t)**3*cosh(b*t), t, s)
    assert plane == b
    assert (
        -L - 3*a/(8*(9*a**2 + b**2 + 2*b*s + s**2)) -
        3*a/(8*(9*a**2 + b**2 - 2*b*s + s**2)) +
        3*a/(8*(a**2 + b**2 + 2*b*s + s**2)) +
        3*a/(8*(a**2 + b**2 - 2*b*s + s**2))).simplify() == 0
    assert (LT(t**2*exp(-t**2), t, s) ==
            (sqrt(pi)*s**2*exp(s**2/4)*erfc(s/2)/8 - s/4 +
             sqrt(pi)*exp(s**2/4)*erfc(s/2)/4, 0, True))
    assert (LT((a*t**2 + b*t + c)*f(t), t, s) ==
            (a*Derivative(LaplaceTransform(f(t), t, s), (s, 2)) -
             b*Derivative(LaplaceTransform(f(t), t, s), s) +
            c*LaplaceTransform(f(t), t, s), -oo, True))
    assert (LT(t**np*g(t), t, s) ==
            ((-1)**np*Derivative(LaplaceTransform(g(t), t, s), (s, np)),
             -oo, True))
    # The following tests check whether _piecewise_to_heaviside works:
    x1 = Piecewise((0, t <= 0), (1, t <= 1), (0, True))
    X1 = LT(x1, t, s)[0]
    assert X1 == 1/s - exp(-s)/s
    y1 = ILT(X1, s, t)
    assert y1 == Heaviside(t) - Heaviside(t - 1)
    x1 = Piecewise((0, t <= 0), (t, t <= 1), (2-t, t <= 2), (0, True))
    X1 = LT(x1, t, s)[0].simplify()
    assert X1 == (exp(2*s) - 2*exp(s) + 1)*exp(-2*s)/s**2
    y1 = ILT(X1, s, t)
    assert (
        -y1 + t*Heaviside(t) + (t - 2)*Heaviside(t - 2) -
        2*(t - 1)*Heaviside(t - 1)).simplify() == 0
    x1 = Piecewise((exp(t), t <= 0), (1, t <= 1), (exp(-(t)), True))
    X1 = LT(x1, t, s)[0]
    assert X1 == exp(-1)*exp(-s)/(s + 1) + 1/s - exp(-s)/s
    y1 = ILT(X1, s, t)
    assert y1 == (
        exp(-1)*exp(1 - t)*Heaviside(t - 1) + Heaviside(t) - Heaviside(t - 1))
    x1 = Piecewise((0, x <= 0), (1, x <= 1), (0, True))
    X1 = LT(x1, t, s)[0]
    assert X1 == Piecewise((0, x <= 0), (1, x <= 1), (0, True))/s
    x1 = [
        a*Piecewise((1, And(t > 1, t <= 3)), (2, True)),
        a*Piecewise((1, And(t >= 1, t <= 3)), (2, True)),
        a*Piecewise((1, And(t >= 1, t < 3)), (2, True)),
        a*Piecewise((1, And(t > 1, t < 3)), (2, True))]
    for x2 in x1:
        assert LT(x2, t, s)[0].expand() == 2*a/s - a*exp(-s)/s + a*exp(-3*s)/s
    assert (
        LT(Piecewise((1, Eq(t, 1)), (2, True)), t, s)[0] ==
        LaplaceTransform(Piecewise((1, Eq(t, 1)), (2, True)), t, s))
    # The following lines test whether _laplace_transform successfully
    # removes Heaviside(1) before processing espressions. It fails if
    # Heaviside(t) remains because then meijerg functions will appear.
    X1 = 1/sqrt(a*s**2-b)
    x1 = ILT(X1, s, t)
    Y1 = LT(x1, t, s)[0]
    Z1 = (Y1**2/X1**2).simplify()
    assert Z1 == 1
    # The following two lines test whether issues #5813 and #7176 are solved.
    assert (LT(diff(f(t), (t, 1)), t, s, noconds=True) ==
            s*LaplaceTransform(f(t), t, s) - f(0))
    assert (LT(diff(f(t), (t, 3)), t, s, noconds=True) ==
            s**3*LaplaceTransform(f(t), t, s) - s**2*f(0) -
            s*Subs(Derivative(f(t), t), t, 0) -
            Subs(Derivative(f(t), (t, 2)), t, 0))
    # Issue #7219
    assert (LT(diff(f(x, t, w), t, 2), t, s) ==
            (s**2*LaplaceTransform(f(x, t, w), t, s) - s*f(x, 0, w) -
             Subs(Derivative(f(x, t, w), t), t, 0), -oo, True))
    # Issue #23307
    assert (LT(10*diff(f(t), (t, 1)), t, s, noconds=True) ==
            10*s*LaplaceTransform(f(t), t, s) - 10*f(0))
    assert (LT(a*f(b*t)+g(c*t), t, s, noconds=True) ==
            a*LaplaceTransform(f(t), t, s/b)/b +
            LaplaceTransform(g(t), t, s/c)/c)
    assert inverse_laplace_transform(
        f(w), w, t, plane=0) == InverseLaplaceTransform(f(w), w, t, 0)
    assert (LT(f(t)*g(t), t, s, noconds=True) ==
            LaplaceTransform(f(t)*g(t), t, s))
    # Issue #24294
    assert (LT(b*f(a*t), t, s, noconds=True) ==
            b*LaplaceTransform(f(t), t, s/a)/a)
    assert LT(3*exp(t)*Heaviside(t), t, s) == (3/(s - 1), 1, True)
    assert (LT(2*sin(t)*Heaviside(t), t, s, simplify=True) ==
            (2/(s**2 + 1), 0, True))
    # Issue #25293
    assert (
        LT((1/(t-1))*sin(4*pi*(t-1))*DiracDelta(t-1) *
           (Heaviside(t-1/4) - Heaviside(t-2)), t, s)[0] == 4*pi*exp(-s))
    # additional basic tests from wikipedia
    assert (LT((t - a)**b*exp(-c*(t - a))*Heaviside(t - a), t, s) ==
            ((c + s)**(-b - 1)*exp(-a*s)*gamma(b + 1), -c, True))
    assert (
        LT((exp(2*t)-1)*exp(-b-t)*Heaviside(t)/2, t, s, noconds=True,
           simplify=True) ==
        exp(-b)/(s**2 - 1))
    # DiracDelta function: standard cases
    assert LT(DiracDelta(t), t, s) == (1, -oo, True)
    assert LT(DiracDelta(a*t), t, s) == (1/a, -oo, True)
    assert LT(DiracDelta(t/42), t, s) == (42, -oo, True)
    assert LT(DiracDelta(t+42), t, s) == (0, -oo, True)
    assert (LT(DiracDelta(t)+DiracDelta(t-42), t, s) ==
            (1 + exp(-42*s), -oo, True))
    assert (LT(DiracDelta(t)-a*exp(-a*t), t, s, simplify=True) ==
            (s/(a + s), -a, True))
    assert (
        LT(exp(-t)*(DiracDelta(t)+DiracDelta(t-42)), t, s, simplify=True) ==
        (exp(-42*s - 42) + 1, -oo, True))
    assert LT(f(t)*DiracDelta(t-42), t, s) == (f(42)*exp(-42*s), -oo, True)
    assert LT(f(t)*DiracDelta(b*t-a), t, s) == (f(a/b)*exp(-a*s/b)/b,
                                                -oo, True)
    assert LT(f(t)*DiracDelta(b*t+a), t, s) == (0, -oo, True)
    # SingularityFunction
    assert LT(SingularityFunction(t, a, -1), t, s)[0] == exp(-a*s)
    assert LT(SingularityFunction(t, a, 1), t, s)[0] == exp(-a*s)/s**2
    assert LT(SingularityFunction(t, a, x), t, s)[0] == (
        LaplaceTransform(SingularityFunction(t, a, x), t, s))
    # Collection of cases that cannot be fully evaluated and/or would catch
    # some common implementation errors
    assert (LT(DiracDelta(t**2), t, s, noconds=True) ==
            LaplaceTransform(DiracDelta(t**2), t, s))
    assert LT(DiracDelta(t**2 - 1), t, s) == (exp(-s)/2, -oo, True)
    assert LT(DiracDelta(t*(1 - t)), t, s) == (1 - exp(-s), -oo, True)
    assert (LT((DiracDelta(t) + 1)*(DiracDelta(t - 1) + 1), t, s) ==
            (LaplaceTransform(DiracDelta(t)*DiracDelta(t - 1), t, s) +
             1 + exp(-s) + 1/s, 0, True))
    assert LT(DiracDelta(2*t-2*exp(a)), t, s) == (exp(-s*exp(a))/2, -oo, True)
    assert LT(DiracDelta(-2*t+2*exp(a)), t, s) == (exp(-s*exp(a))/2, -oo, True)
    # Heaviside tests
    assert LT(Heaviside(t), t, s) == (1/s, 0, True)
    assert LT(Heaviside(t - a), t, s) == (exp(-a*s)/s, 0, True)
    assert LT(Heaviside(t-1), t, s) == (exp(-s)/s, 0, True)
    assert LT(Heaviside(2*t-4), t, s) == (exp(-2*s)/s, 0, True)
    assert LT(Heaviside(2*t+4), t, s) == (1/s, 0, True)
    assert (LT(Heaviside(-2*t+4), t, s, simplify=True) ==
            (1/s - exp(-2*s)/s, 0, True))
    assert (LT(g(t)*Heaviside(t - w), t, s) ==
            (LaplaceTransform(g(t)*Heaviside(t - w), t, s), -oo, True))
    assert (
        LT(Heaviside(t-a)*g(t), t, s) ==
        (LaplaceTransform(g(a + t), t, s)*exp(-a*s), -oo, True))
    assert (
        LT(Heaviside(t+a)*g(t), t, s) ==
        (LaplaceTransform(g(t), t, s), -oo, True))
    assert (
        LT(Heaviside(-t+a)*g(t), t, s) ==
        (LaplaceTransform(g(t), t, s) -
         LaplaceTransform(g(a + t), t, s)*exp(-a*s), -oo, True))
    assert (
        LT(Heaviside(-t-a)*g(t), t, s) == (0, 0, True))
    # Fresnel functions
    assert (laplace_transform(fresnels(t), t, s, simplify=True) ==
            ((-sin(s**2/(2*pi))*fresnels(s/pi) +
              sqrt(2)*sin(s**2/(2*pi) + pi/4)/2 -
              cos(s**2/(2*pi))*fresnelc(s/pi))/s, 0, True))
    assert (laplace_transform(fresnelc(t), t, s, simplify=True) ==
            ((sin(s**2/(2*pi))*fresnelc(s/pi) -
              cos(s**2/(2*pi))*fresnels(s/pi) +
              sqrt(2)*cos(s**2/(2*pi) + pi/4)/2)/s, 0, True))
    # Matrix tests
    Mt = Matrix([[exp(t), t*exp(-t)], [t*exp(-t), exp(t)]])
    Ms = Matrix([[1/(s - 1), (s + 1)**(-2)],
                 [(s + 1)**(-2),     1/(s - 1)]])
    # The default behaviour for Laplace transform of a Matrix returns a Matrix
    # of Tuples and is deprecated:
    with warns_deprecated_sympy():
        Ms_conds = Matrix(
            [[(1/(s - 1), 1, True), ((s + 1)**(-2), -1, True)],
             [((s + 1)**(-2), -1, True), (1/(s - 1), 1, True)]])
    with warns_deprecated_sympy():
        assert LT(Mt, t, s) == Ms_conds
    # The new behavior is to return a tuple of a Matrix and the convergence
    # conditions for the matrix as a whole:
    assert LT(Mt, t, s, legacy_matrix=False) == (Ms, 1, True)
    # With noconds=True the transformed matrix is returned without conditions
    # either way:
    assert LT(Mt, t, s, noconds=True) == Ms
    assert LT(Mt, t, s, legacy_matrix=False, noconds=True) == Ms


@slow
def test_inverse_laplace_transform():
    s = symbols('s')
    k, n, t = symbols('k, n, t', real=True)
    a, b, c, d = symbols('a, b, c, d', positive=True)
    f = Function('f')
    F = Function('F')

    def ILT(g):
        return inverse_laplace_transform(g, s, t)

    def ILTS(g):
        return inverse_laplace_transform(g, s, t, simplify=True)

    def ILTF(g):
        return laplace_correspondence(
            inverse_laplace_transform(g, s, t), {f: F})

    # Tests for the rules in Bateman54.

    # Section 4.1: Some of the Laplace transform rules can also be used well
    #     in the inverse transform.
    assert ILTF(exp(-a*s)*F(s)) == f(-a + t)
    assert ILTF(k*F(s-a)) == k*f(t)*exp(-a*t)
    assert ILTF(diff(F(s), s, 3)) == -t**3*f(t)
    assert ILTF(diff(F(s), s, 4)) == t**4*f(t)

    # Section 5.1: Most rules are impractical for a computer algebra system.

    # Section 5.2: Rational functions
    assert ILT(2) == 2*DiracDelta(t)
    assert ILT(1/s) == Heaviside(t)
    assert ILT(1/s**2) == t*Heaviside(t)
    assert ILT(1/s**5) == t**4*Heaviside(t)/24
    assert ILT(1/s**n) == t**(n - 1)*Heaviside(t)/gamma(n)
    assert ILT(a/(a + s)) == a*exp(-a*t)*Heaviside(t)
    assert ILT(s/(a + s)) == -a*exp(-a*t)*Heaviside(t) + DiracDelta(t)
    assert (ILT(b*s/(s+a)**2) ==
            b*(-a*t*exp(-a*t)*Heaviside(t) + exp(-a*t)*Heaviside(t)))
    assert (ILTS(c/((s+a)*(s+b))) ==
            c*(exp(a*t) - exp(b*t))*exp(-t*(a + b))*Heaviside(t)/(a - b))
    assert (ILTS(c*s/((s+a)*(s+b))) ==
            c*(a*exp(b*t) - b*exp(a*t))*exp(-t*(a + b))*Heaviside(t)/(a - b))
    assert ILTS(s/(a + s)**3) == t*(-a*t + 2)*exp(-a*t)*Heaviside(t)/2
    assert ILTS(1/(s*(a + s)**3)) == (
        -a**2*t**2 - 2*a*t + 2*exp(a*t) - 2)*exp(-a*t)*Heaviside(t)/(2*a**3)
    assert ILT(1/(s*(a + s)**n)) == (
        Heaviside(t)*lowergamma(n, a*t)/(a**n*gamma(n)))
    assert ILT((s-a)**(-b)) == t**(b - 1)*exp(a*t)*Heaviside(t)/gamma(b)
    assert ILT((a + s)**(-2)) == t*exp(-a*t)*Heaviside(t)
    assert ILT((a + s)**(-5)) == t**4*exp(-a*t)*Heaviside(t)/24
    assert ILT(s**2/(s**2 + 1)) == -sin(t)*Heaviside(t) + DiracDelta(t)
    assert ILT(1 - 1/(s**2 + 1)) == -sin(t)*Heaviside(t) + DiracDelta(t)
    assert ILT(a/(a**2 + s**2)) == sin(a*t)*Heaviside(t)
    assert ILT(s/(s**2 + a**2)) == cos(a*t)*Heaviside(t)
    assert ILT(b/(b**2 + (a + s)**2)) == exp(-a*t)*sin(b*t)*Heaviside(t)
    assert (ILT(b*s/(b**2 + (a + s)**2)) ==
            b*(-a*exp(-a*t)*sin(b*t)/b + exp(-a*t)*cos(b*t))*Heaviside(t))
    assert ILT(1/(s**2*(s**2 + 1))) == t*Heaviside(t) - sin(t)*Heaviside(t)
    assert (ILTS(c*s/(d**2*(s+a)**2+b**2)) ==
            c*(-a*d*sin(b*t/d) + b*cos(b*t/d))*exp(-a*t)*Heaviside(t)/(b*d**2))
    assert ILTS((b*s**2 + d)/(a**2 + s**2)**2) == (
        2*a**2*b*sin(a*t) + (a**2*b - d)*(a*t*cos(a*t) -
                                          sin(a*t)))*Heaviside(t)/(2*a**3)
    assert ILTS(b/(s**2-a**2)) == b*sinh(a*t)*Heaviside(t)/a
    assert (ILT(b/(s**2-a**2)) ==
            b*(exp(a*t)*Heaviside(t)/(2*a) - exp(-a*t)*Heaviside(t)/(2*a)))
    assert ILTS(b*s/(s**2-a**2)) == b*cosh(a*t)*Heaviside(t)
    assert (ILT(b/(s*(s+a))) ==
            b*(Heaviside(t)/a - exp(-a*t)*Heaviside(t)/a))
    # Issue #24424
    assert (ILTS((s + 8)/((s + 2)*(s**2 + 2*s + 10))) ==
            ((8*sin(3*t) - 9*cos(3*t))*exp(t) + 9)*exp(-2*t)*Heaviside(t)/15)
    # Issue #8514; this is not important anymore, since this function
    # is not solved by integration anymore
    assert (ILT(1/(a*s**2+b*s+c)) ==
            2*exp(-b*t/(2*a))*sin(t*sqrt(4*a*c - b**2)/(2*a)) *
            Heaviside(t)/sqrt(4*a*c - b**2))

    # Section 5.3: Irrational algebraic functions
    assert (  # (1)
        ILT(1/sqrt(s)/(b*s-a)) ==
        exp(a*t/b)*Heaviside(t)*erf(sqrt(a)*sqrt(t)/sqrt(b))/(sqrt(a)*sqrt(b)))
    assert (  # (2)
        ILT(1/sqrt(k*s)/(c*s-a)/s) ==
        (-2*c*sqrt(t)/(sqrt(pi)*a) +
         c**(S(3)/2)*exp(a*t/c)*erf(sqrt(a)*sqrt(t)/sqrt(c))/a**(S(3)/2)) *
        Heaviside(t)/(c*sqrt(k)))
    assert (  # (4)
        ILT(1/(sqrt(c*s)+a)) == (-a*exp(a**2*t/c)*erfc(a*sqrt(t)/sqrt(c))/c +
                                 1/(sqrt(pi)*sqrt(c)*sqrt(t)))*Heaviside(t))
    assert (  # (5)
        ILT(a/s/(b*sqrt(s)+a)) ==
        (-exp(a**2*t/b**2)*erfc(a*sqrt(t)/b) + 1)*Heaviside(t))
    assert (  # (6)
            ILT((a-b)*sqrt(s)/(sqrt(s)+sqrt(a))/(s-b)) ==
            (sqrt(a)*sqrt(b)*exp(b*t)*erfc(sqrt(b)*sqrt(t)) +
             a*exp(a*t)*erfc(sqrt(a)*sqrt(t)) - b*exp(b*t))*Heaviside(t))
    assert (  # (7)
            ILT(1/sqrt(s)/(sqrt(b*s)+a)) ==
            exp(a**2*t/b)*Heaviside(t)*erfc(a*sqrt(t)/sqrt(b))/sqrt(b))
    assert (  # (8)
            ILT(a**2/(sqrt(s)+a)/s**(S(3)/2)) ==
            (2*a*sqrt(t)/sqrt(pi) + exp(a**2*t)*erfc(a*sqrt(t)) - 1) *
            Heaviside(t))
    assert (  # (9)
            ILT((a-b)*sqrt(b)/(s-b)/sqrt(s)/(sqrt(s)+sqrt(a))) ==
            (sqrt(a)*exp(b*t)*erf(sqrt(b)*sqrt(t)) +
             sqrt(b)*exp(a*t)*erfc(sqrt(a)*sqrt(t)) -
             sqrt(b)*exp(b*t))*Heaviside(t))
    assert (  # (10)
            ILT(1/(sqrt(s)+sqrt(a))**2) ==
            (-2*sqrt(a)*sqrt(t)/sqrt(pi) +
             (-2*a*t + 1)*(erf(sqrt(a)*sqrt(t)) -
                           1)*exp(a*t) + 1)*Heaviside(t))
    assert (  # (11)
            ILT(1/(sqrt(s)+sqrt(a))**2/s) ==
            ((2*t - 1/a)*exp(a*t)*erfc(sqrt(a)*sqrt(t)) + 1/a -
             2*sqrt(t)/(sqrt(pi)*sqrt(a)))*Heaviside(t))
    assert (  # (12)
            ILT(1/(sqrt(s)+a)**2/sqrt(s)) ==
            (-2*a*t*exp(a**2*t)*erfc(a*sqrt(t)) +
             2*sqrt(t)/sqrt(pi))*Heaviside(t))
    assert (  # (13)
            ILT(1/(sqrt(s)+a)**3) ==
            (-a*t*(2*a**2*t + 3)*exp(a**2*t)*erfc(a*sqrt(t)) +
             2*sqrt(t)*(a**2*t + 1)/sqrt(pi))*Heaviside(t))
    x = (
        - ILT(sqrt(s)/(sqrt(s)+a)**3) +
        2*(sqrt(pi)*a**2*t*(-2*sqrt(pi)*erfc(a*sqrt(t)) +
                            2*exp(-a**2*t)/(a*sqrt(t))) *
           (-a**4*t**2 - 5*a**2*t/2 - S.Half) * exp(a**2*t)/2 +
           sqrt(pi)*a*sqrt(t)*(a**2*t + 1)/2) *
        Heaviside(t)/(pi*a**2*t)).simplify()
    assert (  # (14)
            x == 0)
    x = (
        - ILT(1/sqrt(s)/(sqrt(s)+a)**3) +
        Heaviside(t)*(sqrt(t)*((2*a**2*t + 1) *
                               (sqrt(pi)*a*sqrt(t)*exp(a**2*t) *
                                erfc(a*sqrt(t)) - 1) + 1) /
                      (sqrt(pi)*a))).simplify()
    assert (  # (15)
            x == 0)
    assert (  # (16)
            factor_terms(ILT(3/(sqrt(s)+a)**4)) ==
            3*(-2*a**3*t**(S(5)/2)*(2*a**2*t + 5)/(3*sqrt(pi)) +
               t*(4*a**4*t**2 + 12*a**2*t + 3)*exp(a**2*t) *
               erfc(a*sqrt(t))/3)*Heaviside(t))
    assert (  # (17)
            ILT((sqrt(s)-a)/(s*(sqrt(s)+a))) ==
            (2*exp(a**2*t)*erfc(a*sqrt(t))-1)*Heaviside(t))
    assert (  # (18)
            ILT((sqrt(s)-a)**2/(s*(sqrt(s)+a)**2)) == (
                1 + 8*a**2*t*exp(a**2*t)*erfc(a*sqrt(t)) -
                8/sqrt(pi)*a*sqrt(t))*Heaviside(t))
    assert (  # (19)
            ILT((sqrt(s)-a)**3/(s*(sqrt(s)+a)**3)) == Heaviside(t)*(
                2*(8*a**4*t**2+8*a**2*t+1)*exp(a**2*t) *
                erfc(a*sqrt(t))-8/sqrt(pi)*a*sqrt(t)*(2*a**2*t+1)-1))
    assert (  # (22)
            ILT(sqrt(s+a)/(s+b)) == Heaviside(t)*(
                exp(-a*t)/sqrt(t)/sqrt(pi) +
                sqrt(a-b)*exp(-b*t)*erf(sqrt(a-b)*sqrt(t))))
    assert (  # (23)
            ILT(1/sqrt(s+b)/(s+a)) == Heaviside(t)*(
                1/sqrt(b-a)*exp(-a*t)*erf(sqrt(b-a)*sqrt(t))))
    assert (  # (35)
            ILT(1/sqrt(s**2+a**2)) == Heaviside(t)*(
                besselj(0, a*t)))
    assert (  # (44)
            ILT(1/sqrt(s**2-a**2)) == Heaviside(t)*(
                besseli(0, a*t)))

    # Miscellaneous tests
    # Can _inverse_laplace_time_shift deal with positive exponents?
    assert (
        - ILT((s**2*exp(2*s) + 4*exp(s) - 4)*exp(-2*s)/(s*(s**2 + 1))) +
        cos(t)*Heaviside(t) + 4*cos(t - 2)*Heaviside(t - 2) -
        4*cos(t - 1)*Heaviside(t - 1) - 4*Heaviside(t - 2) +
        4*Heaviside(t - 1)).simplify() == 0


@slow
def test_inverse_laplace_transform_old():
    from sympy.functions.special.delta_functions import DiracDelta
    ILT = inverse_laplace_transform
    a, b, c, d = symbols('a b c d', positive=True)
    n, r = symbols('n, r', real=True)
    t, z = symbols('t z')
    f = Function('f')
    F = Function('F')

    def simp_hyp(expr):
        return factor_terms(expand_mul(expr)).rewrite(sin)

    L = ILT(F(s), s, t)
    assert laplace_correspondence(L, {f: F}) == f(t)
    assert ILT(exp(-a*s)/s, s, t) == Heaviside(-a + t)
    assert ILT(exp(-a*s)/(b + s), s, t) == exp(-b*(-a + t))*Heaviside(-a + t)
    assert (ILT((b + s)/(a**2 + (b + s)**2), s, t) ==
            exp(-b*t)*cos(a*t)*Heaviside(t))
    assert (ILT(exp(-a*s)/s**b, s, t) ==
            (-a + t)**(b - 1)*Heaviside(-a + t)/gamma(b))
    assert (ILT(exp(-a*s)/sqrt(s**2 + 1), s, t) ==
            Heaviside(-a + t)*besselj(0, a - t))
    assert ILT(1/(s*sqrt(s + 1)), s, t) == Heaviside(t)*erf(sqrt(t))
    # TODO sinh/cosh shifted come out a mess. also delayed trig is a mess
    # TODO should this simplify further?
    assert (ILT(exp(-a*s)/s**b, s, t) ==
            (t - a)**(b - 1)*Heaviside(t - a)/gamma(b))
    assert (ILT(exp(-a*s)/sqrt(1 + s**2), s, t) ==
            Heaviside(t - a)*besselj(0, a - t))  # note: besselj(0, x) is even
    # XXX ILT turns these branch factor into trig functions ...
    assert (
        simplify(ILT(a**b*(s + sqrt(s**2 - a**2))**(-b)/sqrt(s**2 - a**2),
                     s, t).rewrite(exp)) ==
        Heaviside(t)*besseli(b, a*t))
    assert (
        ILT(a**b*(s + sqrt(s**2 + a**2))**(-b)/sqrt(s**2 + a**2),
            s, t, simplify=True).rewrite(exp) ==
        Heaviside(t)*besselj(b, a*t))
    assert ILT(1/(s*sqrt(s + 1)), s, t) == Heaviside(t)*erf(sqrt(t))
    # TODO can we make erf(t) work?
    assert (ILT((s * eye(2) - Matrix([[1, 0], [0, 2]])).inv(), s, t) ==
            Matrix([[exp(t)*Heaviside(t), 0], [0, exp(2*t)*Heaviside(t)]]))
    # Test time_diff rule
    assert (ILT(s**42*f(s), s, t) ==
            Derivative(InverseLaplaceTransform(f(s), s, t, None), (t, 42)))
    assert ILT(cos(s), s, t) == InverseLaplaceTransform(cos(s), s, t, None)
    # Rules for testing different DiracDelta cases
    assert (
        ILT(1 + 2*s + 3*s**2 + 5*s**3, s, t) == DiracDelta(t) +
        2*DiracDelta(t, 1) + 3*DiracDelta(t, 2) + 5*DiracDelta(t, 3))
    assert (ILT(2*exp(3*s) - 5*exp(-7*s), s, t) ==
            2*InverseLaplaceTransform(exp(3*s), s, t, None) -
            5*DiracDelta(t - 7))
    a = cos(sin(7)/2)
    assert ILT(a*exp(-3*s), s, t) == a*DiracDelta(t - 3)
    assert ILT(exp(2*s), s, t) == InverseLaplaceTransform(exp(2*s), s, t, None)
    r = Symbol('r', real=True)
    assert ILT(exp(r*s), s, t) == InverseLaplaceTransform(exp(r*s), s, t, None)
    # Rules for testing whether Heaviside(t) is treated properly in diff rule
    assert ILT(s**2/(a**2 + s**2), s, t) == (
        -a*sin(a*t)*Heaviside(t) + DiracDelta(t))
    assert ILT(s**2*(f(s) + 1/(a**2 + s**2)), s, t) == (
        -a*sin(a*t)*Heaviside(t) + DiracDelta(t) +
        Derivative(InverseLaplaceTransform(f(s), s, t, None), (t, 2)))
    # Rules from the previous test_inverse_laplace_transform_delta_cond():
    assert (ILT(exp(r*s), s, t, noconds=False) ==
            (InverseLaplaceTransform(exp(r*s), s, t, None), True))
    # inversion does not exist: verify it doesn't evaluate to DiracDelta
    for z in (Symbol('z', extended_real=False),
              Symbol('z', imaginary=True, zero=False)):
        f = ILT(exp(z*s), s, t, noconds=False)
        f = f[0] if isinstance(f, tuple) else f
        assert f.func != DiracDelta


@slow
def test_expint():
    x = Symbol('x')
    a = Symbol('a')
    u = Symbol('u', polar=True)

    # TODO LT of Si, Shi, Chi is a mess ...
    assert laplace_transform(Ci(x), x, s) == (-log(1 + s**2)/2/s, 0, True)
    assert (laplace_transform(expint(a, x), x, s, simplify=True) ==
            (lerchphi(s*exp_polar(I*pi), 1, a), 0, re(a) > S.Zero))
    assert (laplace_transform(expint(1, x), x, s, simplify=True) ==
            (log(s + 1)/s, 0, True))
    assert (laplace_transform(expint(2, x), x, s, simplify=True) ==
            ((s - log(s + 1))/s**2, 0, True))
    assert (inverse_laplace_transform(-log(1 + s**2)/2/s, s, u).expand() ==
            Heaviside(u)*Ci(u))
    assert (
        inverse_laplace_transform(log(s + 1)/s, s, x,
                                  simplify=True).rewrite(expint) ==
        Heaviside(x)*E1(x))
    assert (
        inverse_laplace_transform(
            (s - log(s + 1))/s**2, s, x,
            simplify=True).rewrite(expint).expand() ==
        (expint(2, x)*Heaviside(x)).rewrite(Ei).rewrite(expint).expand())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_meijerint.py ===
from sympy.core.function import expand_func
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.complexes import Abs, arg, re, unpolarify
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import cosh, acosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, asin)
from sympy.functions.special.error_functions import (erf, erfc)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import simplify
from sympy.integrals.meijerint import (_rewrite_single, _rewrite1,
    meijerint_indefinite, _inflate_g, _create_lookup_table,
    meijerint_definite, meijerint_inversion)
from sympy.testing.pytest import slow
from sympy.core.random import (verify_numerically,
        random_complex_number as randcplx)
from sympy.abc import x, y, a, b, c, d, s, t, z


def test_rewrite_single():
    def t(expr, c, m):
        e = _rewrite_single(meijerg([a], [b], [c], [d], expr), x)
        assert e is not None
        assert isinstance(e[0][0][2], meijerg)
        assert e[0][0][2].argument.as_coeff_mul(x) == (c, (m,))

    def tn(expr):
        assert _rewrite_single(meijerg([a], [b], [c], [d], expr), x) is None

    t(x, 1, x)
    t(x**2, 1, x**2)
    t(x**2 + y*x**2, y + 1, x**2)
    tn(x**2 + x)
    tn(x**y)

    def u(expr, x):
        from sympy.core.add import Add
        r = _rewrite_single(expr, x)
        e = Add(*[res[0]*res[2] for res in r[0]]).replace(
            exp_polar, exp)  # XXX Hack?
        assert verify_numerically(e, expr, x)

    u(exp(-x)*sin(x), x)

    # The following has stopped working because hyperexpand changed slightly.
    # It is probably not worth fixing
    #u(exp(-x)*sin(x)*cos(x), x)

    # This one cannot be done numerically, since it comes out as a g-function
    # of argument 4*pi
    # NOTE This also tests a bug in inverse mellin transform (which used to
    #      turn exp(4*pi*I*t) into a factor of exp(4*pi*I)**t instead of
    #      exp_polar).
    #u(exp(x)*sin(x), x)
    assert _rewrite_single(exp(x)*sin(x), x) == \
        ([(-sqrt(2)/(2*sqrt(pi)), 0,
           meijerg(((Rational(-1, 2), 0, Rational(1, 4), S.Half, Rational(3, 4)), (1,)),
                   ((), (Rational(-1, 2), 0)), 64*exp_polar(-4*I*pi)/x**4))], True)


def test_rewrite1():
    assert _rewrite1(x**3*meijerg([a], [b], [c], [d], x**2 + y*x**2)*5, x) == \
        (5, x**3, [(1, 0, meijerg([a], [b], [c], [d], x**2*(y + 1)))], True)


def test_meijerint_indefinite_numerically():
    def t(fac, arg):
        g = meijerg([a], [b], [c], [d], arg)*fac
        subs = {a: randcplx()/10, b: randcplx()/10 + I,
                c: randcplx(), d: randcplx()}
        integral = meijerint_indefinite(g, x)
        assert integral is not None
        assert verify_numerically(g.subs(subs), integral.diff(x).subs(subs), x)
    t(1, x)
    t(2, x)
    t(1, 2*x)
    t(1, x**2)
    t(5, x**S('3/2'))
    t(x**3, x)
    t(3*x**S('3/2'), 4*x**S('7/3'))


def test_meijerint_definite():
    v, b = meijerint_definite(x, x, 0, 0)
    assert v.is_zero and b is True
    v, b = meijerint_definite(x, x, oo, oo)
    assert v.is_zero and b is True


def test_inflate():
    subs = {a: randcplx()/10, b: randcplx()/10 + I, c: randcplx(),
            d: randcplx(), y: randcplx()/10}

    def t(a, b, arg, n):
        from sympy.core.mul import Mul
        m1 = meijerg(a, b, arg)
        m2 = Mul(*_inflate_g(m1, n))
        # NOTE: (the random number)**9 must still be on the principal sheet.
        # Thus make b&d small to create random numbers of small imaginary part.
        return verify_numerically(m1.subs(subs), m2.subs(subs), x, b=0.1, d=-0.1)
    assert t([[a], [b]], [[c], [d]], x, 3)
    assert t([[a, y], [b]], [[c], [d]], x, 3)
    assert t([[a], [b]], [[c, y], [d]], 2*x**3, 3)


def test_recursive():
    from sympy.core.symbol import symbols
    a, b, c = symbols('a b c', positive=True)
    r = exp(-(x - a)**2)*exp(-(x - b)**2)
    e = integrate(r, (x, 0, oo), meijerg=True)
    assert simplify(e.expand()) == (
        sqrt(2)*sqrt(pi)*(
        (erf(sqrt(2)*(a + b)/2) + 1)*exp(-a**2/2 + a*b - b**2/2))/4)
    e = integrate(exp(-(x - a)**2)*exp(-(x - b)**2)*exp(c*x), (x, 0, oo), meijerg=True)
    assert simplify(e) == (
        sqrt(2)*sqrt(pi)*(erf(sqrt(2)*(2*a + 2*b + c)/4) + 1)*exp(-a**2 - b**2
        + (2*a + 2*b + c)**2/8)/4)
    assert simplify(integrate(exp(-(x - a - b - c)**2), (x, 0, oo), meijerg=True)) == \
        sqrt(pi)/2*(1 + erf(a + b + c))
    assert simplify(integrate(exp(-(x + a + b + c)**2), (x, 0, oo), meijerg=True)) == \
        sqrt(pi)/2*(1 - erf(a + b + c))


@slow
def test_meijerint():
    from sympy.core.function import expand
    from sympy.core.symbol import symbols
    s, t, mu = symbols('s t mu', real=True)
    assert integrate(meijerg([], [], [0], [], s*t)
                     *meijerg([], [], [mu/2], [-mu/2], t**2/4),
                     (t, 0, oo)).is_Piecewise
    s = symbols('s', positive=True)
    assert integrate(x**s*meijerg([[], []], [[0], []], x), (x, 0, oo)) == \
        gamma(s + 1)
    assert integrate(x**s*meijerg([[], []], [[0], []], x), (x, 0, oo),
                     meijerg=True) == gamma(s + 1)
    assert isinstance(integrate(x**s*meijerg([[], []], [[0], []], x),
                                (x, 0, oo), meijerg=False),
                      Integral)

    assert meijerint_indefinite(exp(x), x) == exp(x)

    # TODO what simplifications should be done automatically?
    # This tests "extra case" for antecedents_1.
    a, b = symbols('a b', positive=True)
    assert simplify(meijerint_definite(x**a, x, 0, b)[0]) == \
        b**(a + 1)/(a + 1)

    # This tests various conditions and expansions:
    assert meijerint_definite((x + 1)**3*exp(-x), x, 0, oo) == (16, True)

    # Again, how about simplifications?
    sigma, mu = symbols('sigma mu', positive=True)
    i, c = meijerint_definite(exp(-((x - mu)/(2*sigma))**2), x, 0, oo)
    assert simplify(i) == sqrt(pi)*sigma*(2 - erfc(mu/(2*sigma)))
    assert c == True

    i, _ = meijerint_definite(exp(-mu*x)*exp(sigma*x), x, 0, oo)
    # TODO it would be nice to test the condition
    assert simplify(i) == 1/(mu - sigma)

    # Test substitutions to change limits
    assert meijerint_definite(exp(x), x, -oo, 2) == (exp(2), True)
    # Note: causes a NaN in _check_antecedents
    assert expand(meijerint_definite(exp(x), x, 0, I)[0]) == exp(I) - 1
    assert expand(meijerint_definite(exp(-x), x, 0, x)[0]) == \
        1 - exp(-exp(I*arg(x))*abs(x))

    # Test -oo to oo
    assert meijerint_definite(exp(-x**2), x, -oo, oo) == (sqrt(pi), True)
    assert meijerint_definite(exp(-abs(x)), x, -oo, oo) == (2, True)
    assert meijerint_definite(exp(-(2*x - 3)**2), x, -oo, oo) == \
        (sqrt(pi)/2, True)
    assert meijerint_definite(exp(-abs(2*x - 3)), x, -oo, oo) == (1, True)
    assert meijerint_definite(exp(-((x - mu)/sigma)**2/2)/sqrt(2*pi*sigma**2),
                              x, -oo, oo) == (1, True)
    assert meijerint_definite(sinc(x)**2, x, -oo, oo) == (pi, True)

    # Test one of the extra conditions for 2 g-functinos
    assert meijerint_definite(exp(-x)*sin(x), x, 0, oo) == (S.Half, True)

    # Test a bug
    def res(n):
        return (1/(1 + x**2)).diff(x, n).subs(x, 1)*(-1)**n
    for n in range(6):
        assert integrate(exp(-x)*sin(x)*x**n, (x, 0, oo), meijerg=True) == \
            res(n)

    # This used to test trigexpand... now it is done by linear substitution
    assert simplify(integrate(exp(-x)*sin(x + a), (x, 0, oo), meijerg=True)
                    ) == sqrt(2)*sin(a + pi/4)/2

    # Test the condition 14 from prudnikov.
    # (This is besselj*besselj in disguise, to stop the product from being
    #  recognised in the tables.)
    a, b, s = symbols('a b s')
    assert meijerint_definite(meijerg([], [], [a/2], [-a/2], x/4)
                  *meijerg([], [], [b/2], [-b/2], x/4)*x**(s - 1), x, 0, oo
        ) == (
        (4*2**(2*s - 2)*gamma(-2*s + 1)*gamma(a/2 + b/2 + s)
         /(gamma(-a/2 + b/2 - s + 1)*gamma(a/2 - b/2 - s + 1)
           *gamma(a/2 + b/2 - s + 1)),
            (re(s) < 1) & (re(s) < S(1)/2) & (re(a)/2 + re(b)/2 + re(s) > 0)))

    # test a bug
    assert integrate(sin(x**a)*sin(x**b), (x, 0, oo), meijerg=True) == \
        Integral(sin(x**a)*sin(x**b), (x, 0, oo))

    # test better hyperexpand
    assert integrate(exp(-x**2)*log(x), (x, 0, oo), meijerg=True) == \
        (sqrt(pi)*polygamma(0, S.Half)/4).expand()

    # Test hyperexpand bug.
    from sympy.functions.special.gamma_functions import lowergamma
    n = symbols('n', integer=True)
    assert simplify(integrate(exp(-x)*x**n, x, meijerg=True)) == \
        lowergamma(n + 1, x)

    # Test a bug with argument 1/x
    alpha = symbols('alpha', positive=True)
    assert meijerint_definite((2 - x)**alpha*sin(alpha/x), x, 0, 2) == \
        (sqrt(pi)*alpha*gamma(alpha + 1)*meijerg(((), (alpha/2 + S.Half,
        alpha/2 + 1)), ((0, 0, S.Half), (Rational(-1, 2),)), alpha**2/16)/4, True)

    # test a bug related to 3016
    a, s = symbols('a s', positive=True)
    assert simplify(integrate(x**s*exp(-a*x**2), (x, -oo, oo))) == \
        a**(-s/2 - S.Half)*((-1)**s + 1)*gamma(s/2 + S.Half)/2


def test_bessel():
    from sympy.functions.special.bessel import (besseli, besselj)
    assert simplify(integrate(besselj(a, z)*besselj(b, z)/z, (z, 0, oo),
                     meijerg=True, conds='none')) == \
        2*sin(pi*(a/2 - b/2))/(pi*(a - b)*(a + b))
    assert simplify(integrate(besselj(a, z)*besselj(a, z)/z, (z, 0, oo),
                     meijerg=True, conds='none')) == 1/(2*a)

    # TODO more orthogonality integrals

    assert simplify(integrate(sin(z*x)*(x**2 - 1)**(-(y + S.Half)),
                              (x, 1, oo), meijerg=True, conds='none')
                    *2/((z/2)**y*sqrt(pi)*gamma(S.Half - y))) == \
        besselj(y, z)

    # Werner Rosenheinrich
    # SOME INDEFINITE INTEGRALS OF BESSEL FUNCTIONS

    assert integrate(x*besselj(0, x), x, meijerg=True) == x*besselj(1, x)
    assert integrate(x*besseli(0, x), x, meijerg=True) == x*besseli(1, x)
    # TODO can do higher powers, but come out as high order ... should they be
    #      reduced to order 0, 1?
    assert integrate(besselj(1, x), x, meijerg=True) == -besselj(0, x)
    assert integrate(besselj(1, x)**2/x, x, meijerg=True) == \
        -(besselj(0, x)**2 + besselj(1, x)**2)/2
    # TODO more besseli when tables are extended or recursive mellin works
    assert integrate(besselj(0, x)**2/x**2, x, meijerg=True) == \
        -2*x*besselj(0, x)**2 - 2*x*besselj(1, x)**2 \
        + 2*besselj(0, x)*besselj(1, x) - besselj(0, x)**2/x
    assert integrate(besselj(0, x)*besselj(1, x), x, meijerg=True) == \
        -besselj(0, x)**2/2
    assert integrate(x**2*besselj(0, x)*besselj(1, x), x, meijerg=True) == \
        x**2*besselj(1, x)**2/2
    assert integrate(besselj(0, x)*besselj(1, x)/x, x, meijerg=True) == \
        (x*besselj(0, x)**2 + x*besselj(1, x)**2 -
            besselj(0, x)*besselj(1, x))
    # TODO how does besselj(0, a*x)*besselj(0, b*x) work?
    # TODO how does besselj(0, x)**2*besselj(1, x)**2 work?
    # TODO sin(x)*besselj(0, x) etc come out a mess
    # TODO can x*log(x)*besselj(0, x) be done?
    # TODO how does besselj(1, x)*besselj(0, x+a) work?
    # TODO more indefinite integrals when struve functions etc are implemented

    # test a substitution
    assert integrate(besselj(1, x**2)*x, x, meijerg=True) == \
        -besselj(0, x**2)/2


def test_inversion():
    from sympy.functions.special.bessel import besselj
    from sympy.functions.special.delta_functions import Heaviside

    def inv(f):
        return piecewise_fold(meijerint_inversion(f, s, t))
    assert inv(1/(s**2 + 1)) == sin(t)*Heaviside(t)
    assert inv(s/(s**2 + 1)) == cos(t)*Heaviside(t)
    assert inv(exp(-s)/s) == Heaviside(t - 1)
    assert inv(1/sqrt(1 + s**2)) == besselj(0, t)*Heaviside(t)

    # Test some antcedents checking.
    assert meijerint_inversion(sqrt(s)/sqrt(1 + s**2), s, t) is None
    assert inv(exp(s**2)) is None
    assert meijerint_inversion(exp(-s**2), s, t) is None


def test_inversion_conditional_output():
    from sympy.core.symbol import Symbol
    from sympy.integrals.transforms import InverseLaplaceTransform

    a = Symbol('a', positive=True)
    F = sqrt(pi/a)*exp(-2*sqrt(a)*sqrt(s))
    f = meijerint_inversion(F, s, t)
    assert not f.is_Piecewise

    b = Symbol('b', real=True)
    F = F.subs(a, b)
    f2 = meijerint_inversion(F, s, t)
    assert f2.is_Piecewise
    # first piece is same as f
    assert f2.args[0][0] == f.subs(a, b)
    # last piece is an unevaluated transform
    assert f2.args[-1][1]
    ILT = InverseLaplaceTransform(F, s, t, None)
    assert f2.args[-1][0] == ILT or f2.args[-1][0] == ILT.as_integral


def test_inversion_exp_real_nonreal_shift():
    from sympy.core.symbol import Symbol
    from sympy.functions.special.delta_functions import DiracDelta
    r = Symbol('r', real=True)
    c = Symbol('c', extended_real=False)
    a = 1 + 2*I
    z = Symbol('z')
    assert not meijerint_inversion(exp(r*s), s, t).is_Piecewise
    assert meijerint_inversion(exp(a*s), s, t) is None
    assert meijerint_inversion(exp(c*s), s, t) is None
    f = meijerint_inversion(exp(z*s), s, t)
    assert f.is_Piecewise
    assert isinstance(f.args[0][0], DiracDelta)


@slow
def test_lookup_table():
    from sympy.core.random import uniform, randrange
    from sympy.core.add import Add
    from sympy.integrals.meijerint import z as z_dummy
    table = {}
    _create_lookup_table(table)
    for l in table.values():
        for formula, terms, cond, hint in sorted(l, key=default_sort_key):
            subs = {}
            for ai in list(formula.free_symbols) + [z_dummy]:
                if hasattr(ai, 'properties') and ai.properties:
                    # these Wilds match positive integers
                    subs[ai] = randrange(1, 10)
                else:
                    subs[ai] = uniform(1.5, 2.0)
            if not isinstance(terms, list):
                terms = terms(subs)

            # First test that hyperexpand can do this.
            expanded = [hyperexpand(g) for (_, g) in terms]
            assert all(x.is_Piecewise or not x.has(meijerg) for x in expanded)

            # Now test that the meijer g-function is indeed as advertised.
            expanded = Add(*[f*x for (f, x) in terms])
            a, b = formula.n(subs=subs), expanded.n(subs=subs)
            r = min(abs(a), abs(b))
            if r < 1:
                assert abs(a - b).n() <= 1e-10
            else:
                assert (abs(a - b)/r).n() <= 1e-10


def test_branch_bug():
    from sympy.functions.special.gamma_functions import lowergamma
    from sympy.simplify.powsimp import powdenest
    # TODO gammasimp cannot prove that the factor is unity
    assert powdenest(integrate(erf(x**3), x, meijerg=True).diff(x),
           polar=True) == 2*erf(x**3)*gamma(Rational(2, 3))/3/gamma(Rational(5, 3))
    assert integrate(erf(x**3), x, meijerg=True) == \
        2*x*erf(x**3)*gamma(Rational(2, 3))/(3*gamma(Rational(5, 3))) \
        - 2*gamma(Rational(2, 3))*lowergamma(Rational(2, 3), x**6)/(3*sqrt(pi)*gamma(Rational(5, 3)))


def test_linear_subs():
    from sympy.functions.special.bessel import besselj
    assert integrate(sin(x - 1), x, meijerg=True) == -cos(1 - x)
    assert integrate(besselj(1, x - 1), x, meijerg=True) == -besselj(0, 1 - x)


@slow
def test_probability():
    # various integrals from probability theory
    from sympy.core.function import expand_mul
    from sympy.core.symbol import (Symbol, symbols)
    from sympy.simplify.gammasimp import gammasimp
    from sympy.simplify.powsimp import powsimp
    mu1, mu2 = symbols('mu1 mu2', nonzero=True)
    sigma1, sigma2 = symbols('sigma1 sigma2', positive=True)
    rate = Symbol('lambda', positive=True)

    def normal(x, mu, sigma):
        return 1/sqrt(2*pi*sigma**2)*exp(-(x - mu)**2/2/sigma**2)

    def exponential(x, rate):
        return rate*exp(-rate*x)

    assert integrate(normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) == 1
    assert integrate(x*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) == \
        mu1
    assert integrate(x**2*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) \
        == mu1**2 + sigma1**2
    assert integrate(x**3*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) \
        == mu1**3 + 3*mu1*sigma1**2
    assert integrate(normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == 1
    assert integrate(x*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == mu1
    assert integrate(y*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == mu2
    assert integrate(x*y*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == mu1*mu2
    assert integrate((x + y + 1)*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == 1 + mu1 + mu2
    assert integrate((x + y - 1)*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == \
        -1 + mu1 + mu2

    i = integrate(x**2*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                  (x, -oo, oo), (y, -oo, oo), meijerg=True)
    assert not i.has(Abs)
    assert simplify(i) == mu1**2 + sigma1**2
    assert integrate(y**2*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),
                     (x, -oo, oo), (y, -oo, oo), meijerg=True) == \
        sigma2**2 + mu2**2

    assert integrate(exponential(x, rate), (x, 0, oo), meijerg=True) == 1
    assert integrate(x*exponential(x, rate), (x, 0, oo), meijerg=True) == \
        1/rate
    assert integrate(x**2*exponential(x, rate), (x, 0, oo), meijerg=True) == \
        2/rate**2

    def E(expr):
        res1 = integrate(expr*exponential(x, rate)*normal(y, mu1, sigma1),
                         (x, 0, oo), (y, -oo, oo), meijerg=True)
        res2 = integrate(expr*exponential(x, rate)*normal(y, mu1, sigma1),
                        (y, -oo, oo), (x, 0, oo), meijerg=True)
        assert expand_mul(res1) == expand_mul(res2)
        return res1

    assert E(1) == 1
    assert E(x*y) == mu1/rate
    assert E(x*y**2) == mu1**2/rate + sigma1**2/rate
    ans = sigma1**2 + 1/rate**2
    assert simplify(E((x + y + 1)**2) - E(x + y + 1)**2) == ans
    assert simplify(E((x + y - 1)**2) - E(x + y - 1)**2) == ans
    assert simplify(E((x + y)**2) - E(x + y)**2) == ans

    # Beta' distribution
    alpha, beta = symbols('alpha beta', positive=True)
    betadist = x**(alpha - 1)*(1 + x)**(-alpha - beta)*gamma(alpha + beta) \
        /gamma(alpha)/gamma(beta)
    assert integrate(betadist, (x, 0, oo), meijerg=True) == 1
    i = integrate(x*betadist, (x, 0, oo), meijerg=True, conds='separate')
    assert (gammasimp(i[0]), i[1]) == (alpha/(beta - 1), 1 < beta)
    j = integrate(x**2*betadist, (x, 0, oo), meijerg=True, conds='separate')
    assert j[1] == (beta > 2)
    assert gammasimp(j[0] - i[0]**2) == (alpha + beta - 1)*alpha \
        /(beta - 2)/(beta - 1)**2

    # Beta distribution
    # NOTE: this is evaluated using antiderivatives. It also tests that
    #       meijerint_indefinite returns the simplest possible answer.
    a, b = symbols('a b', positive=True)
    betadist = x**(a - 1)*(-x + 1)**(b - 1)*gamma(a + b)/(gamma(a)*gamma(b))
    assert simplify(integrate(betadist, (x, 0, 1), meijerg=True)) == 1
    assert simplify(integrate(x*betadist, (x, 0, 1), meijerg=True)) == \
        a/(a + b)
    assert simplify(integrate(x**2*betadist, (x, 0, 1), meijerg=True)) == \
        a*(a + 1)/(a + b)/(a + b + 1)
    assert simplify(integrate(x**y*betadist, (x, 0, 1), meijerg=True)) == \
        gamma(a + b)*gamma(a + y)/gamma(a)/gamma(a + b + y)

    # Chi distribution
    k = Symbol('k', integer=True, positive=True)
    chi = 2**(1 - k/2)*x**(k - 1)*exp(-x**2/2)/gamma(k/2)
    assert powsimp(integrate(chi, (x, 0, oo), meijerg=True)) == 1
    assert simplify(integrate(x*chi, (x, 0, oo), meijerg=True)) == \
        sqrt(2)*gamma((k + 1)/2)/gamma(k/2)
    assert simplify(integrate(x**2*chi, (x, 0, oo), meijerg=True)) == k

    # Chi^2 distribution
    chisquared = 2**(-k/2)/gamma(k/2)*x**(k/2 - 1)*exp(-x/2)
    assert powsimp(integrate(chisquared, (x, 0, oo), meijerg=True)) == 1
    assert simplify(integrate(x*chisquared, (x, 0, oo), meijerg=True)) == k
    assert simplify(integrate(x**2*chisquared, (x, 0, oo), meijerg=True)) == \
        k*(k + 2)
    assert gammasimp(integrate(((x - k)/sqrt(2*k))**3*chisquared, (x, 0, oo),
                    meijerg=True)) == 2*sqrt(2)/sqrt(k)

    # Dagum distribution
    a, b, p = symbols('a b p', positive=True)
    # XXX (x/b)**a does not work
    dagum = a*p/x*(x/b)**(a*p)/(1 + x**a/b**a)**(p + 1)
    assert simplify(integrate(dagum, (x, 0, oo), meijerg=True)) == 1
    # XXX conditions are a mess
    arg = x*dagum
    assert simplify(integrate(arg, (x, 0, oo), meijerg=True, conds='none')
                    ) == a*b*gamma(1 - 1/a)*gamma(p + 1 + 1/a)/(
                    (a*p + 1)*gamma(p))
    assert simplify(integrate(x*arg, (x, 0, oo), meijerg=True, conds='none')
                    ) == a*b**2*gamma(1 - 2/a)*gamma(p + 1 + 2/a)/(
                    (a*p + 2)*gamma(p))

    # F-distribution
    d1, d2 = symbols('d1 d2', positive=True)
    f = sqrt(((d1*x)**d1 * d2**d2)/(d1*x + d2)**(d1 + d2))/x \
        /gamma(d1/2)/gamma(d2/2)*gamma((d1 + d2)/2)
    assert simplify(integrate(f, (x, 0, oo), meijerg=True)) == 1
    # TODO conditions are a mess
    assert simplify(integrate(x*f, (x, 0, oo), meijerg=True, conds='none')
                    ) == d2/(d2 - 2)
    assert simplify(integrate(x**2*f, (x, 0, oo), meijerg=True, conds='none')
                    ) == d2**2*(d1 + 2)/d1/(d2 - 4)/(d2 - 2)

    # TODO gamma, rayleigh

    # inverse gaussian
    lamda, mu = symbols('lamda mu', positive=True)
    dist = sqrt(lamda/2/pi)*x**(Rational(-3, 2))*exp(-lamda*(x - mu)**2/x/2/mu**2)
    mysimp = lambda expr: simplify(expr.rewrite(exp))
    assert mysimp(integrate(dist, (x, 0, oo))) == 1
    assert mysimp(integrate(x*dist, (x, 0, oo))) == mu
    assert mysimp(integrate((x - mu)**2*dist, (x, 0, oo))) == mu**3/lamda
    assert mysimp(integrate((x - mu)**3*dist, (x, 0, oo))) == 3*mu**5/lamda**2

    # Levi
    c = Symbol('c', positive=True)
    assert integrate(sqrt(c/2/pi)*exp(-c/2/(x - mu))/(x - mu)**S('3/2'),
                    (x, mu, oo)) == 1
    # higher moments oo

    # log-logistic
    alpha, beta = symbols('alpha beta', positive=True)
    distn = (beta/alpha)*x**(beta - 1)/alpha**(beta - 1)/ \
        (1 + x**beta/alpha**beta)**2
    # FIXME: If alpha, beta are not declared as finite the line below hangs
    # after the changes in:
    #    https://github.com/sympy/sympy/pull/16603
    assert simplify(integrate(distn, (x, 0, oo))) == 1
    # NOTE the conditions are a mess, but correctly state beta > 1
    assert simplify(integrate(x*distn, (x, 0, oo), conds='none')) == \
        pi*alpha/beta/sin(pi/beta)
    # (similar comment for conditions applies)
    assert simplify(integrate(x**y*distn, (x, 0, oo), conds='none')) == \
        pi*alpha**y*y/beta/sin(pi*y/beta)

    # weibull
    k = Symbol('k', positive=True)
    n = Symbol('n', positive=True)
    distn = k/lamda*(x/lamda)**(k - 1)*exp(-(x/lamda)**k)
    assert simplify(integrate(distn, (x, 0, oo))) == 1
    assert simplify(integrate(x**n*distn, (x, 0, oo))) == \
        lamda**n*gamma(1 + n/k)

    # rice distribution
    from sympy.functions.special.bessel import besseli
    nu, sigma = symbols('nu sigma', positive=True)
    rice = x/sigma**2*exp(-(x**2 + nu**2)/2/sigma**2)*besseli(0, x*nu/sigma**2)
    assert integrate(rice, (x, 0, oo), meijerg=True) == 1
    # can someone verify higher moments?

    # Laplace distribution
    mu = Symbol('mu', real=True)
    b = Symbol('b', positive=True)
    laplace = exp(-abs(x - mu)/b)/2/b
    assert integrate(laplace, (x, -oo, oo), meijerg=True) == 1
    assert integrate(x*laplace, (x, -oo, oo), meijerg=True) == mu
    assert integrate(x**2*laplace, (x, -oo, oo), meijerg=True) == \
        2*b**2 + mu**2

    # TODO are there other distributions supported on (-oo, oo) that we can do?

    # misc tests
    k = Symbol('k', positive=True)
    assert gammasimp(expand_mul(integrate(log(x)*x**(k - 1)*exp(-x)/gamma(k),
                              (x, 0, oo)))) == polygamma(0, k)


@slow
def test_expint():
    """ Test various exponential integrals. """
    from sympy.core.symbol import Symbol
    from sympy.functions.elementary.hyperbolic import sinh
    from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, expint)
    assert simplify(unpolarify(integrate(exp(-z*x)/x**y, (x, 1, oo),
                meijerg=True, conds='none'
                ).rewrite(expint).expand(func=True))) == expint(y, z)

    assert integrate(exp(-z*x)/x, (x, 1, oo), meijerg=True,
                     conds='none').rewrite(expint).expand() == \
        expint(1, z)
    assert integrate(exp(-z*x)/x**2, (x, 1, oo), meijerg=True,
                     conds='none').rewrite(expint).expand() == \
        expint(2, z).rewrite(Ei).rewrite(expint)
    assert integrate(exp(-z*x)/x**3, (x, 1, oo), meijerg=True,
                     conds='none').rewrite(expint).expand() == \
        expint(3, z).rewrite(Ei).rewrite(expint).expand()

    t = Symbol('t', positive=True)
    assert integrate(-cos(x)/x, (x, t, oo), meijerg=True).expand() == Ci(t)
    assert integrate(-sin(x)/x, (x, t, oo), meijerg=True).expand() == \
        Si(t) - pi/2
    assert integrate(sin(x)/x, (x, 0, z), meijerg=True) == Si(z)
    assert integrate(sinh(x)/x, (x, 0, z), meijerg=True) == Shi(z)
    assert integrate(exp(-x)/x, x, meijerg=True).expand().rewrite(expint) == \
        I*pi - expint(1, x)
    assert integrate(exp(-x)/x**2, x, meijerg=True).rewrite(expint).expand() \
        == expint(1, x) - exp(-x)/x - I*pi

    u = Symbol('u', polar=True)
    assert integrate(cos(u)/u, u, meijerg=True).expand().as_independent(u)[1] \
        == Ci(u)
    assert integrate(cosh(u)/u, u, meijerg=True).expand().as_independent(u)[1] \
        == Chi(u)

    assert integrate(expint(1, x), x, meijerg=True
            ).rewrite(expint).expand() == x*expint(1, x) - exp(-x)
    assert integrate(expint(2, x), x, meijerg=True
            ).rewrite(expint).expand() == \
        -x**2*expint(1, x)/2 + x*exp(-x)/2 - exp(-x)/2
    assert simplify(unpolarify(integrate(expint(y, x), x,
                 meijerg=True).rewrite(expint).expand(func=True))) == \
        -expint(y + 1, x)

    assert integrate(Si(x), x, meijerg=True) == x*Si(x) + cos(x)
    assert integrate(Ci(u), u, meijerg=True).expand() == u*Ci(u) - sin(u)
    assert integrate(Shi(x), x, meijerg=True) == x*Shi(x) - cosh(x)
    assert integrate(Chi(u), u, meijerg=True).expand() == u*Chi(u) - sinh(u)

    assert integrate(Si(x)*exp(-x), (x, 0, oo), meijerg=True) == pi/4
    assert integrate(expint(1, x)*sin(x), (x, 0, oo), meijerg=True) == log(2)/2


def test_messy():
    from sympy.functions.elementary.hyperbolic import (acosh, acoth)
    from sympy.functions.elementary.trigonometric import (asin, atan)
    from sympy.functions.special.bessel import besselj
    from sympy.functions.special.error_functions import (Chi, E1, Shi, Si)
    from sympy.integrals.transforms import (fourier_transform, laplace_transform)
    assert (laplace_transform(Si(x), x, s, simplify=True) ==
            ((-atan(s) + pi/2)/s, 0, True))

    assert laplace_transform(Shi(x), x, s, simplify=True) == (
        acoth(s)/s, -oo, s**2 > 1)

    # where should the logs be simplified?
    assert laplace_transform(Chi(x), x, s, simplify=True) == (
        (log(s**(-2)) - log(1 - 1/s**2))/(2*s), -oo, s**2 > 1)

    # TODO maybe simplify the inequalities? when the simplification
    # allows for generators instead of symbols this will work
    assert laplace_transform(besselj(a, x), x, s)[1:] == \
        (0, (re(a) > -2) & (re(a) > -1))

    # NOTE s < 0 can be done, but argument reduction is not good enough yet
    ans = fourier_transform(besselj(1, x)/x, x, s, noconds=False)
    assert (ans[0].factor(deep=True).expand(), ans[1]) == \
        (Piecewise((0, (s > 1/(2*pi)) | (s < -1/(2*pi))),
                   (2*sqrt(-4*pi**2*s**2 + 1), True)), s > 0)
    # TODO FT(besselj(0,x)) - conditions are messy (but for acceptable reasons)
    #                       - folding could be better

    assert integrate(E1(x)*besselj(0, x), (x, 0, oo), meijerg=True) == \
        log(1 + sqrt(2))
    assert integrate(E1(x)*besselj(1, x), (x, 0, oo), meijerg=True) == \
        log(S.Half + sqrt(2)/2)

    assert integrate(1/x/sqrt(1 - x**2), x, meijerg=True) == \
        Piecewise((-acosh(1/x), abs(x**(-2)) > 1), (I*asin(1/x), True))


def test_issue_6122():
    assert integrate(exp(-I*x**2), (x, -oo, oo), meijerg=True) == \
        -I*sqrt(pi)*exp(I*pi/4)


def test_issue_6252():
    expr = 1/x/(a + b*x)**Rational(1, 3)
    anti = integrate(expr, x, meijerg=True)
    assert not anti.has(hyper)
    # XXX the expression is a mess, but actually upon differentiation and
    # putting in numerical values seems to work...


def test_issue_6348():
    assert integrate(exp(I*x)/(1 + x**2), (x, -oo, oo)).simplify().rewrite(exp) \
        == pi*exp(-1)


def test_fresnel():
    from sympy.functions.special.error_functions import (fresnelc, fresnels)

    assert expand_func(integrate(sin(pi*x**2/2), x)) == fresnels(x)
    assert expand_func(integrate(cos(pi*x**2/2), x)) == fresnelc(x)


def test_issue_6860():
    assert meijerint_indefinite(x**x**x, x) is None


def test_issue_7337():
    f = meijerint_indefinite(x*sqrt(2*x + 3), x).together()
    assert f == sqrt(2*x + 3)*(2*x**2 + x - 3)/5
    assert f._eval_interval(x, S.NegativeOne, S.One) == Rational(2, 5)


def test_issue_8368():
    assert meijerint_indefinite(cosh(x)*exp(-x*t), x) == (
        (-t - 1)*exp(x) + (-t + 1)*exp(-x))*exp(-t*x)/2/(t**2 - 1)


def test_issue_10211():
    from sympy.abc import h, w
    assert integrate((1/sqrt((y-x)**2 + h**2)**3), (x,0,w), (y,0,w)) == \
        2*sqrt(1 + w**2/h**2)/h - 2/h


def test_issue_11806():
    from sympy.core.symbol import symbols
    y, L = symbols('y L', positive=True)
    assert integrate(1/sqrt(x**2 + y**2)**3, (x, -L, L)) == \
        2*L/(y**2*sqrt(L**2 + y**2))

def test_issue_10681():
    from sympy.polys.domains.realfield import RR
    from sympy.abc import R, r
    f = integrate(r**2*(R**2-r**2)**0.5, r, meijerg=True)
    g = (1.0/3)*R**1.0*r**3*hyper((-0.5, Rational(3, 2)), (Rational(5, 2),),
                                  r**2*exp_polar(2*I*pi)/R**2)
    assert RR.almosteq((f/g).n(), 1.0, 1e-12)

def test_issue_13536():
    from sympy.core.symbol import Symbol
    a = Symbol('a', positive=True)
    assert integrate(1/x**2, (x, oo, a)) == -1/a


def test_issue_6462():
    from sympy.core.symbol import Symbol
    x = Symbol('x')
    n = Symbol('n')
    # Not the actual issue, still wrong answer for n = 1, but that there is no
    # exception
    assert integrate(cos(x**n)/x**n, x, meijerg=True).subs(n, 2).equals(
            integrate(cos(x**2)/x**2, x, meijerg=True))


def test_indefinite_1_bug():
    assert integrate((b + t)**(-a), t, meijerg=True) == -b*(1 + t/b)**(1 - a)/(a*b**a - b**a)


def test_pr_23583():
    # This result is wrong. Check whether new result is correct when this test fail.
    assert integrate(1/sqrt((x - I)**2-1), meijerg=True) == \
           Piecewise((acosh(x - I), Abs((x - I)**2) > 1), (-I*asin(x - I), True))


# 25786
def test_integrate_function_of_square_over_negatives():
    assert integrate(exp(-x**2), (x,-5,0), meijerg=True) == sqrt(pi)/2 * erf(5)


def test_issue_25949():
    from sympy.core.symbol import symbols
    y = symbols("y", nonzero=True)
    assert integrate(cosh(y*(x + 1)), (x, -1, -0.25), meijerg=True) == sinh(0.75*y)/y

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexes\multi\test_setops.py ===
import numpy as np
import pytest

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    IntervalIndex,
    MultiIndex,
    Series,
)
import pandas._testing as tm
from pandas.api.types import (
    is_float_dtype,
    is_unsigned_integer_dtype,
)


@pytest.mark.parametrize("case", [0.5, "xxx"])
@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_set_ops_error_cases(idx, case, sort, method):
    # non-iterable input
    msg = "Input must be Index or array-like"
    with pytest.raises(TypeError, match=msg):
        getattr(idx, method)(case, sort=sort)


@pytest.mark.parametrize("klass", [MultiIndex, np.array, Series, list])
def test_intersection_base(idx, sort, klass):
    first = idx[2::-1]  # first 3 elements reversed
    second = idx[:5]

    if klass is not MultiIndex:
        second = klass(second.values)

    intersect = first.intersection(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(intersect, expected)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.intersection([1, 2, 3], sort=sort)


@pytest.mark.arm_slow
@pytest.mark.parametrize("klass", [MultiIndex, np.array, Series, list])
def test_union_base(idx, sort, klass):
    first = idx[::-1]
    second = idx[:5]

    if klass is not MultiIndex:
        second = klass(second.values)

    union = first.union(second, sort=sort)
    if sort is None:
        expected = first.sort_values()
    else:
        expected = first
    tm.assert_index_equal(union, expected)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.union([1, 2, 3], sort=sort)


def test_difference_base(idx, sort):
    second = idx[4:]
    answer = idx[:4]
    result = idx.difference(second, sort=sort)

    if sort is None:
        answer = answer.sort_values()

    assert result.equals(answer)
    tm.assert_index_equal(result, answer)

    # GH 10149
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = idx.difference(case, sort=sort)
        tm.assert_index_equal(result, answer)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        idx.difference([1, 2, 3], sort=sort)


def test_symmetric_difference(idx, sort):
    first = idx[1:]
    second = idx[:-1]
    answer = idx[[-1, 0]]
    result = first.symmetric_difference(second, sort=sort)

    if sort is None:
        answer = answer.sort_values()

    tm.assert_index_equal(result, answer)

    # GH 10149
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = first.symmetric_difference(case, sort=sort)
        tm.assert_index_equal(result, answer)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.symmetric_difference([1, 2, 3], sort=sort)


def test_multiindex_symmetric_difference():
    # GH 13490
    idx = MultiIndex.from_product([["a", "b"], ["A", "B"]], names=["a", "b"])
    result = idx.symmetric_difference(idx)
    assert result.names == idx.names

    idx2 = idx.copy().rename(["A", "B"])
    result = idx.symmetric_difference(idx2)
    assert result.names == [None, None]


def test_empty(idx):
    # GH 15270
    assert not idx.empty
    assert idx[:0].empty


def test_difference(idx, sort):
    first = idx
    result = first.difference(idx[-3:], sort=sort)
    vals = idx[:-3].values

    if sort is None:
        vals = sorted(vals)

    expected = MultiIndex.from_tuples(vals, sortorder=0, names=idx.names)

    assert isinstance(result, MultiIndex)
    assert result.equals(expected)
    assert result.names == idx.names
    tm.assert_index_equal(result, expected)

    # empty difference: reflexive
    result = idx.difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # empty difference: superset
    result = idx[-3:].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # empty difference: degenerate
    result = idx[:0].difference(idx, sort=sort)
    expected = idx[:0]
    assert result.equals(expected)
    assert result.names == idx.names

    # names not the same
    chunklet = idx[-3:]
    chunklet.names = ["foo", "baz"]
    result = first.difference(chunklet, sort=sort)
    assert result.names == (None, None)

    # empty, but non-equal
    result = idx.difference(idx.sortlevel(1)[0], sort=sort)
    assert len(result) == 0

    # raise Exception called with non-MultiIndex
    result = first.difference(first.values, sort=sort)
    assert result.equals(first[:0])

    # name from empty array
    result = first.difference([], sort=sort)
    assert first.equals(result)
    assert first.names == result.names

    # name from non-empty array
    result = first.difference([("foo", "one")], sort=sort)
    expected = MultiIndex.from_tuples(
        [("bar", "one"), ("baz", "two"), ("foo", "two"), ("qux", "one"), ("qux", "two")]
    )
    expected.names = first.names
    assert first.names == result.names

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        first.difference([1, 2, 3, 4, 5], sort=sort)


def test_difference_sort_special():
    # GH-24959
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    # sort=None, the default
    result = idx.difference([])
    tm.assert_index_equal(result, idx)


def test_difference_sort_special_true():
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    result = idx.difference([], sort=True)
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    tm.assert_index_equal(result, expected)


def test_difference_sort_incomparable():
    # GH-24959
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000"), 2], ["a", "b"]])

    other = MultiIndex.from_product([[3, pd.Timestamp("2000"), 4], ["c", "d"]])
    # sort=None, the default
    msg = "sort order is undefined for incomparable objects"
    with tm.assert_produces_warning(RuntimeWarning, match=msg):
        result = idx.difference(other)
    tm.assert_index_equal(result, idx)

    # sort=False
    result = idx.difference(other, sort=False)
    tm.assert_index_equal(result, idx)


def test_difference_sort_incomparable_true():
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000"), 2], ["a", "b"]])
    other = MultiIndex.from_product([[3, pd.Timestamp("2000"), 4], ["c", "d"]])

    # TODO: this is raising in constructing a Categorical when calling
    #  algos.safe_sort. Should we catch and re-raise with a better message?
    msg = "'values' is not ordered, please explicitly specify the categories order "
    with pytest.raises(TypeError, match=msg):
        idx.difference(other, sort=True)


def test_union(idx, sort):
    piece1 = idx[:5][::-1]
    piece2 = idx[3:]

    the_union = piece1.union(piece2, sort=sort)

    if sort in (None, False):
        tm.assert_index_equal(the_union.sort_values(), idx.sort_values())
    else:
        tm.assert_index_equal(the_union, idx)

    # corner case, pass self or empty thing:
    the_union = idx.union(idx, sort=sort)
    tm.assert_index_equal(the_union, idx)

    the_union = idx.union(idx[:0], sort=sort)
    tm.assert_index_equal(the_union, idx)

    tuples = idx.values
    result = idx[:4].union(tuples[4:], sort=sort)
    if sort is None:
        tm.assert_index_equal(result.sort_values(), idx.sort_values())
    else:
        assert result.equals(idx)


def test_union_with_regular_index(idx, using_infer_string):
    other = Index(["A", "B", "C"])

    result = other.union(idx)
    assert ("foo", "one") in result
    assert "B" in result

    if using_infer_string:
        with pytest.raises(NotImplementedError, match="Can only union"):
            idx.union(other)
    else:
        msg = "The values in the array are unorderable"
        with tm.assert_produces_warning(RuntimeWarning, match=msg):
            result2 = idx.union(other)
        # This is more consistent now, if sorting fails then we don't sort at all
        # in the MultiIndex case.
        assert not result.equals(result2)


def test_intersection(idx, sort):
    piece1 = idx[:5][::-1]
    piece2 = idx[3:]

    the_int = piece1.intersection(piece2, sort=sort)

    if sort in (None, True):
        tm.assert_index_equal(the_int, idx[3:5])
    else:
        tm.assert_index_equal(the_int.sort_values(), idx[3:5])

    # corner case, pass self
    the_int = idx.intersection(idx, sort=sort)
    tm.assert_index_equal(the_int, idx)

    # empty intersection: disjoint
    empty = idx[:2].intersection(idx[2:], sort=sort)
    expected = idx[:0]
    assert empty.equals(expected)

    tuples = idx.values
    result = idx.intersection(tuples)
    assert result.equals(idx)


@pytest.mark.parametrize(
    "method", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_setop_with_categorical(idx, sort, method):
    other = idx.to_flat_index().astype("category")
    res_names = [None] * idx.nlevels

    result = getattr(idx, method)(other, sort=sort)
    expected = getattr(idx, method)(idx, sort=sort).rename(res_names)
    tm.assert_index_equal(result, expected)

    result = getattr(idx, method)(other[:5], sort=sort)
    expected = getattr(idx, method)(idx[:5], sort=sort).rename(res_names)
    tm.assert_index_equal(result, expected)


def test_intersection_non_object(idx, sort):
    other = Index(range(3), name="foo")

    result = idx.intersection(other, sort=sort)
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=None)
    tm.assert_index_equal(result, expected, exact=True)

    # if we pass a length-0 ndarray (i.e. no name, we retain our idx.name)
    result = idx.intersection(np.asarray(other)[:0], sort=sort)
    expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=idx.names)
    tm.assert_index_equal(result, expected, exact=True)

    msg = "other must be a MultiIndex or a list of tuples"
    with pytest.raises(TypeError, match=msg):
        # With non-zero length non-index, we try and fail to convert to tuples
        idx.intersection(np.asarray(other), sort=sort)


def test_intersect_equal_sort():
    # GH-24959
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    tm.assert_index_equal(idx.intersection(idx, sort=False), idx)
    tm.assert_index_equal(idx.intersection(idx, sort=None), idx)


def test_intersect_equal_sort_true():
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    result = idx.intersection(idx, sort=True)
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("slice_", [slice(None), slice(0)])
def test_union_sort_other_empty(slice_):
    # https://github.com/pandas-dev/pandas/issues/24959
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])

    # default, sort=None
    other = idx[slice_]
    tm.assert_index_equal(idx.union(other), idx)
    tm.assert_index_equal(other.union(idx), idx)

    # sort=False
    tm.assert_index_equal(idx.union(other, sort=False), idx)


def test_union_sort_other_empty_sort():
    idx = MultiIndex.from_product([[1, 0], ["a", "b"]])
    other = idx[:0]
    result = idx.union(other, sort=True)
    expected = MultiIndex.from_product([[0, 1], ["a", "b"]])
    tm.assert_index_equal(result, expected)


def test_union_sort_other_incomparable():
    # https://github.com/pandas-dev/pandas/issues/24959
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000")], ["a", "b"]])

    # default, sort=None
    with tm.assert_produces_warning(RuntimeWarning):
        result = idx.union(idx[:1])
    tm.assert_index_equal(result, idx)

    # sort=False
    result = idx.union(idx[:1], sort=False)
    tm.assert_index_equal(result, idx)


def test_union_sort_other_incomparable_sort():
    idx = MultiIndex.from_product([[1, pd.Timestamp("2000")], ["a", "b"]])
    msg = "'<' not supported between instances of 'Timestamp' and 'int'"
    with pytest.raises(TypeError, match=msg):
        idx.union(idx[:1], sort=True)


def test_union_non_object_dtype_raises():
    # GH#32646 raise NotImplementedError instead of less-informative error
    mi = MultiIndex.from_product([["a", "b"], [1, 2]])

    idx = mi.levels[1]

    msg = "Can only union MultiIndex with MultiIndex or Index of tuples"
    with pytest.raises(NotImplementedError, match=msg):
        mi.union(idx)


def test_union_empty_self_different_names():
    # GH#38423
    mi = MultiIndex.from_arrays([[]])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    result = mi.union(mi2)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]])
    tm.assert_index_equal(result, expected)


def test_union_multiindex_empty_rangeindex():
    # GH#41234
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])
    ri = pd.RangeIndex(0)

    result_left = mi.union(ri)
    tm.assert_index_equal(mi, result_left, check_names=False)

    result_right = ri.union(mi)
    tm.assert_index_equal(mi, result_right, check_names=False)


@pytest.mark.parametrize(
    "method", ["union", "intersection", "difference", "symmetric_difference"]
)
def test_setops_sort_validation(method):
    idx1 = MultiIndex.from_product([["a", "b"], [1, 2]])
    idx2 = MultiIndex.from_product([["b", "c"], [1, 2]])

    with pytest.raises(ValueError, match="The 'sort' keyword only takes"):
        getattr(idx1, method)(idx2, sort=2)

    # sort=True is supported as of GH#?
    getattr(idx1, method)(idx2, sort=True)


@pytest.mark.parametrize("val", [pd.NA, 100])
def test_difference_keep_ea_dtypes(any_numeric_ea_dtype, val):
    # GH#48606
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    result = midx.difference(midx2)
    expected = MultiIndex.from_arrays([Series([1], dtype=any_numeric_ea_dtype), [2]])
    tm.assert_index_equal(result, expected)

    result = midx.difference(midx.sort_values(ascending=False))
    expected = MultiIndex.from_arrays(
        [Series([], dtype=any_numeric_ea_dtype), Series([], dtype=np.int64)],
        names=["a", None],
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("val", [pd.NA, 5])
def test_symmetric_difference_keeping_ea_dtype(any_numeric_ea_dtype, val):
    # GH#48607
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    result = midx.symmetric_difference(midx2)
    expected = MultiIndex.from_arrays(
        [Series([1, 1, val], dtype=any_numeric_ea_dtype), [1, 2, 3]]
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    ("tuples", "exp_tuples"),
    [
        ([("val1", "test1")], [("val1", "test1")]),
        ([("val1", "test1"), ("val1", "test1")], [("val1", "test1")]),
        (
            [("val2", "test2"), ("val1", "test1")],
            [("val2", "test2"), ("val1", "test1")],
        ),
    ],
)
def test_intersect_with_duplicates(tuples, exp_tuples):
    # GH#36915
    left = MultiIndex.from_tuples(tuples, names=["first", "second"])
    right = MultiIndex.from_tuples(
        [("val1", "test1"), ("val1", "test1"), ("val2", "test2")],
        names=["first", "second"],
    )
    result = left.intersection(right)
    expected = MultiIndex.from_tuples(exp_tuples, names=["first", "second"])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "data, names, expected",
    [
        ((1,), None, [None, None]),
        ((1,), ["a"], [None, None]),
        ((1,), ["b"], [None, None]),
        ((1, 2), ["c", "d"], [None, None]),
        ((1, 2), ["b", "a"], [None, None]),
        ((1, 2, 3), ["a", "b", "c"], [None, None]),
        ((1, 2), ["a", "c"], ["a", None]),
        ((1, 2), ["c", "b"], [None, "b"]),
        ((1, 2), ["a", "b"], ["a", "b"]),
        ((1, 2), [None, "b"], [None, "b"]),
    ],
)
def test_maybe_match_names(data, names, expected):
    # GH#38323
    mi = MultiIndex.from_tuples([], names=["a", "b"])
    mi2 = MultiIndex.from_tuples([data], names=names)
    result = mi._maybe_match_names(mi2)
    assert result == expected


def test_intersection_equal_different_names():
    # GH#30302
    mi1 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["c", "b"])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["a", "b"])

    result = mi1.intersection(mi2)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]], names=[None, "b"])
    tm.assert_index_equal(result, expected)


def test_intersection_different_names():
    # GH#38323
    mi = MultiIndex.from_arrays([[1], [3]], names=["c", "b"])
    mi2 = MultiIndex.from_arrays([[1], [3]])
    result = mi.intersection(mi2)
    tm.assert_index_equal(result, mi2)


def test_intersection_with_missing_values_on_both_sides(nulls_fixture):
    # GH#38623
    mi1 = MultiIndex.from_arrays([[3, nulls_fixture, 4, nulls_fixture], [1, 2, 4, 2]])
    mi2 = MultiIndex.from_arrays([[3, nulls_fixture, 3], [1, 2, 4]])
    result = mi1.intersection(mi2)
    expected = MultiIndex.from_arrays([[3, nulls_fixture], [1, 2]])
    tm.assert_index_equal(result, expected)


def test_union_with_missing_values_on_both_sides(nulls_fixture):
    # GH#38623
    mi1 = MultiIndex.from_arrays([[1, nulls_fixture]])
    mi2 = MultiIndex.from_arrays([[1, nulls_fixture, 3]])
    result = mi1.union(mi2)
    expected = MultiIndex.from_arrays([[1, 3, nulls_fixture]])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("sort", [None, False])
def test_union_nan_got_duplicated(dtype, sort):
    # GH#38977, GH#49010
    mi1 = MultiIndex.from_arrays([pd.array([1.0, np.nan], dtype=dtype), [2, 3]])
    mi2 = MultiIndex.from_arrays([pd.array([1.0, np.nan, 3.0], dtype=dtype), [2, 3, 4]])
    result = mi1.union(mi2, sort=sort)
    if sort is None:
        expected = MultiIndex.from_arrays(
            [pd.array([1.0, 3.0, np.nan], dtype=dtype), [2, 4, 3]]
        )
    else:
        expected = mi2
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("val", [4, 1])
def test_union_keep_ea_dtype(any_numeric_ea_dtype, val):
    # GH#48505

    arr1 = Series([val, 2], dtype=any_numeric_ea_dtype)
    arr2 = Series([2, 1], dtype=any_numeric_ea_dtype)
    midx = MultiIndex.from_arrays([arr1, [1, 2]], names=["a", None])
    midx2 = MultiIndex.from_arrays([arr2, [2, 1]])
    result = midx.union(midx2)
    if val == 4:
        expected = MultiIndex.from_arrays(
            [Series([1, 2, 4], dtype=any_numeric_ea_dtype), [1, 2, 1]]
        )
    else:
        expected = MultiIndex.from_arrays(
            [Series([1, 2], dtype=any_numeric_ea_dtype), [1, 2]]
        )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dupe_val", [3, pd.NA])
def test_union_with_duplicates_keep_ea_dtype(dupe_val, any_numeric_ea_dtype):
    # GH48900
    mi1 = MultiIndex.from_arrays(
        [
            Series([1, dupe_val, 2], dtype=any_numeric_ea_dtype),
            Series([1, dupe_val, 2], dtype=any_numeric_ea_dtype),
        ]
    )
    mi2 = MultiIndex.from_arrays(
        [
            Series([2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
            Series([2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
        ]
    )
    result = mi1.union(mi2)
    expected = MultiIndex.from_arrays(
        [
            Series([1, 2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
            Series([1, 2, dupe_val, dupe_val], dtype=any_numeric_ea_dtype),
        ]
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_union_duplicates(index, request):
    # GH#38977
    if index.empty or isinstance(index, (IntervalIndex, CategoricalIndex)):
        pytest.skip(f"No duplicates in an empty {type(index).__name__}")

    values = index.unique().values.tolist()
    mi1 = MultiIndex.from_arrays([values, [1] * len(values)])
    mi2 = MultiIndex.from_arrays([[values[0]] + values, [1] * (len(values) + 1)])
    result = mi2.union(mi1)
    expected = mi2.sort_values()
    tm.assert_index_equal(result, expected)

    if (
        is_unsigned_integer_dtype(mi2.levels[0])
        and (mi2.get_level_values(0) < 2**63).all()
    ):
        # GH#47294 - union uses lib.fast_zip, converting data to Python integers
        # and loses type information. Result is then unsigned only when values are
        # sufficiently large to require unsigned dtype. This happens only if other
        # has dups or one of both have missing values
        expected = expected.set_levels(
            [expected.levels[0].astype(np.int64), expected.levels[1]]
        )
    elif is_float_dtype(mi2.levels[0]):
        # mi2 has duplicates witch is a different path than above, Fix that path
        # to use correct float dtype?
        expected = expected.set_levels(
            [expected.levels[0].astype(float), expected.levels[1]]
        )

    result = mi1.union(mi2)
    tm.assert_index_equal(result, expected)


def test_union_keep_dtype_precision(any_real_numeric_dtype):
    # GH#48498
    arr1 = Series([4, 1, 1], dtype=any_real_numeric_dtype)
    arr2 = Series([1, 4], dtype=any_real_numeric_dtype)
    midx = MultiIndex.from_arrays([arr1, [2, 1, 1]], names=["a", None])
    midx2 = MultiIndex.from_arrays([arr2, [1, 2]], names=["a", None])

    result = midx.union(midx2)
    expected = MultiIndex.from_arrays(
        ([Series([1, 1, 4], dtype=any_real_numeric_dtype), [1, 1, 2]]),
        names=["a", None],
    )
    tm.assert_index_equal(result, expected)


def test_union_keep_ea_dtype_with_na(any_numeric_ea_dtype):
    # GH#48498
    arr1 = Series([4, pd.NA], dtype=any_numeric_ea_dtype)
    arr2 = Series([1, pd.NA], dtype=any_numeric_ea_dtype)
    midx = MultiIndex.from_arrays([arr1, [2, 1]], names=["a", None])
    midx2 = MultiIndex.from_arrays([arr2, [1, 2]])
    result = midx.union(midx2)
    expected = MultiIndex.from_arrays(
        [Series([1, 4, pd.NA, pd.NA], dtype=any_numeric_ea_dtype), [1, 2, 1, 2]]
    )
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "levels1, levels2, codes1, codes2, names",
    [
        (
            [["a", "b", "c"], [0, ""]],
            [["c", "d", "b"], [""]],
            [[0, 1, 2], [1, 1, 1]],
            [[0, 1, 2], [0, 0, 0]],
            ["name1", "name2"],
        ),
    ],
)
def test_intersection_lexsort_depth(levels1, levels2, codes1, codes2, names):
    # GH#25169
    mi1 = MultiIndex(levels=levels1, codes=codes1, names=names)
    mi2 = MultiIndex(levels=levels2, codes=codes2, names=names)
    mi_int = mi1.intersection(mi2)
    assert mi_int._lexsort_depth == 2


@pytest.mark.parametrize(
    "a",
    [pd.Categorical(["a", "b"], categories=["a", "b"]), ["a", "b"]],
)
@pytest.mark.parametrize(
    "b",
    [
        pd.Categorical(["a", "b"], categories=["b", "a"], ordered=True),
        pd.Categorical(["a", "b"], categories=["b", "a"]),
    ],
)
def test_intersection_with_non_lex_sorted_categories(a, b):
    # GH#49974
    other = ["1", "2"]

    df1 = DataFrame({"x": a, "y": other})
    df2 = DataFrame({"x": b, "y": other})

    expected = MultiIndex.from_arrays([a, other], names=["x", "y"])

    res1 = MultiIndex.from_frame(df1).intersection(
        MultiIndex.from_frame(df2.sort_values(["x", "y"]))
    )
    res2 = MultiIndex.from_frame(df1).intersection(MultiIndex.from_frame(df2))
    res3 = MultiIndex.from_frame(df1.sort_values(["x", "y"])).intersection(
        MultiIndex.from_frame(df2)
    )
    res4 = MultiIndex.from_frame(df1.sort_values(["x", "y"])).intersection(
        MultiIndex.from_frame(df2.sort_values(["x", "y"]))
    )

    tm.assert_index_equal(res1, expected)
    tm.assert_index_equal(res2, expected)
    tm.assert_index_equal(res3, expected)
    tm.assert_index_equal(res4, expected)


@pytest.mark.parametrize("val", [pd.NA, 100])
def test_intersection_keep_ea_dtypes(val, any_numeric_ea_dtype):
    # GH#48604
    midx = MultiIndex.from_arrays(
        [Series([1, 2], dtype=any_numeric_ea_dtype), [2, 1]], names=["a", None]
    )
    midx2 = MultiIndex.from_arrays(
        [Series([1, 2, val], dtype=any_numeric_ea_dtype), [1, 1, 3]]
    )
    result = midx.intersection(midx2)
    expected = MultiIndex.from_arrays([Series([2], dtype=any_numeric_ea_dtype), [1]])
    tm.assert_index_equal(result, expected)


def test_union_with_na_when_constructing_dataframe():
    # GH43222
    series1 = Series(
        (1,),
        index=MultiIndex.from_arrays(
            [Series([None], dtype="str"), Series([None], dtype="str")]
        ),
    )
    series2 = Series((10, 20), index=MultiIndex.from_tuples(((None, None), ("a", "b"))))
    result = DataFrame([series1, series2])
    expected = DataFrame({(np.nan, np.nan): [1.0, 10.0], ("a", "b"): [np.nan, 20.0]})
    tm.assert_frame_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\api\types\__init__.py ===
"""
Public toolkit API.
"""

from pandas._libs.lib import infer_dtype

from pandas.core.dtypes.api import *  # noqa: F403
from pandas.core.dtypes.concat import union_categoricals
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)

__all__ = [
    "infer_dtype",
    "union_categoricals",
    "CategoricalDtype",
    "DatetimeTZDtype",
    "IntervalDtype",
    "PeriodDtype",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\window\__init__.py ===
from pandas.core.window.ewm import (
    ExponentialMovingWindow,
    ExponentialMovingWindowGroupby,
)
from pandas.core.window.expanding import (
    Expanding,
    ExpandingGroupby,
)
from pandas.core.window.rolling import (
    Rolling,
    RollingGroupby,
    Window,
)

__all__ = [
    "Expanding",
    "ExpandingGroupby",
    "ExponentialMovingWindow",
    "ExponentialMovingWindowGroupby",
    "Rolling",
    "RollingGroupby",
    "Window",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\generic\test_finalize.py ===
"""
An exhaustive list of pandas methods exercising NDFrame.__finalize__.
"""
import operator
import re

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

# TODO:
# * Binary methods (mul, div, etc.)
# * Binary outputs (align, etc.)
# * top-level methods (concat, merge, get_dummies, etc.)
# * window
# * cumulative reductions

not_implemented_mark = pytest.mark.xfail(reason="not implemented")

mi = pd.MultiIndex.from_product([["a", "b"], [0, 1]], names=["A", "B"])

frame_data = ({"A": [1]},)
frame_mi_data = ({"A": [1, 2, 3, 4]}, mi)


# Tuple of
# - Callable: Constructor (Series, DataFrame)
# - Tuple: Constructor args
# - Callable: pass the constructed value with attrs set to this.

_all_methods = [
    (pd.Series, ([0],), operator.methodcaller("take", [])),
    (pd.Series, ([0],), operator.methodcaller("__getitem__", [True])),
    (pd.Series, ([0],), operator.methodcaller("repeat", 2)),
    (pd.Series, ([0],), operator.methodcaller("reset_index")),
    (pd.Series, ([0],), operator.methodcaller("reset_index", drop=True)),
    (pd.Series, ([0],), operator.methodcaller("to_frame")),
    (pd.Series, ([0, 0],), operator.methodcaller("drop_duplicates")),
    (pd.Series, ([0, 0],), operator.methodcaller("duplicated")),
    (pd.Series, ([0, 0],), operator.methodcaller("round")),
    (pd.Series, ([0, 0],), operator.methodcaller("rename", lambda x: x + 1)),
    (pd.Series, ([0, 0],), operator.methodcaller("rename", "name")),
    (pd.Series, ([0, 0],), operator.methodcaller("set_axis", ["a", "b"])),
    (pd.Series, ([0, 0],), operator.methodcaller("reindex", [1, 0])),
    (pd.Series, ([0, 0],), operator.methodcaller("drop", [0])),
    (pd.Series, (pd.array([0, pd.NA]),), operator.methodcaller("fillna", 0)),
    (pd.Series, ([0, 0],), operator.methodcaller("replace", {0: 1})),
    (pd.Series, ([0, 0],), operator.methodcaller("shift")),
    (pd.Series, ([0, 0],), operator.methodcaller("isin", [0, 1])),
    (pd.Series, ([0, 0],), operator.methodcaller("between", 0, 2)),
    (pd.Series, ([0, 0],), operator.methodcaller("isna")),
    (pd.Series, ([0, 0],), operator.methodcaller("isnull")),
    (pd.Series, ([0, 0],), operator.methodcaller("notna")),
    (pd.Series, ([0, 0],), operator.methodcaller("notnull")),
    (pd.Series, ([1],), operator.methodcaller("add", pd.Series([1]))),
    # TODO: mul, div, etc.
    (
        pd.Series,
        ([0], pd.period_range("2000", periods=1)),
        operator.methodcaller("to_timestamp"),
    ),
    (
        pd.Series,
        ([0], pd.date_range("2000", periods=1)),
        operator.methodcaller("to_period"),
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("dot", pd.DataFrame(index=["A"])),
        ),
        marks=pytest.mark.xfail(reason="Implement binary finalize"),
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("transpose")),
    (pd.DataFrame, frame_data, operator.methodcaller("__getitem__", "A")),
    (pd.DataFrame, frame_data, operator.methodcaller("__getitem__", ["A"])),
    (pd.DataFrame, frame_data, operator.methodcaller("__getitem__", np.array([True]))),
    (pd.DataFrame, ({("A", "a"): [1]},), operator.methodcaller("__getitem__", ["A"])),
    (pd.DataFrame, frame_data, operator.methodcaller("query", "A == 1")),
    (pd.DataFrame, frame_data, operator.methodcaller("eval", "A + 1", engine="python")),
    (pd.DataFrame, frame_data, operator.methodcaller("select_dtypes", include="int")),
    (pd.DataFrame, frame_data, operator.methodcaller("assign", b=1)),
    (pd.DataFrame, frame_data, operator.methodcaller("set_axis", ["A"])),
    (pd.DataFrame, frame_data, operator.methodcaller("reindex", [0, 1])),
    (pd.DataFrame, frame_data, operator.methodcaller("drop", columns=["A"])),
    (pd.DataFrame, frame_data, operator.methodcaller("drop", index=[0])),
    (pd.DataFrame, frame_data, operator.methodcaller("rename", columns={"A": "a"})),
    (pd.DataFrame, frame_data, operator.methodcaller("rename", index=lambda x: x)),
    (pd.DataFrame, frame_data, operator.methodcaller("fillna", "A")),
    (pd.DataFrame, frame_data, operator.methodcaller("fillna", method="ffill")),
    (pd.DataFrame, frame_data, operator.methodcaller("set_index", "A")),
    (pd.DataFrame, frame_data, operator.methodcaller("reset_index")),
    (pd.DataFrame, frame_data, operator.methodcaller("isna")),
    (pd.DataFrame, frame_data, operator.methodcaller("isnull")),
    (pd.DataFrame, frame_data, operator.methodcaller("notna")),
    (pd.DataFrame, frame_data, operator.methodcaller("notnull")),
    (pd.DataFrame, frame_data, operator.methodcaller("dropna")),
    (pd.DataFrame, frame_data, operator.methodcaller("drop_duplicates")),
    (pd.DataFrame, frame_data, operator.methodcaller("duplicated")),
    (pd.DataFrame, frame_data, operator.methodcaller("sort_values", by="A")),
    (pd.DataFrame, frame_data, operator.methodcaller("sort_index")),
    (pd.DataFrame, frame_data, operator.methodcaller("nlargest", 1, "A")),
    (pd.DataFrame, frame_data, operator.methodcaller("nsmallest", 1, "A")),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("swaplevel")),
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("add", pd.DataFrame(*frame_data)),
    ),
    # TODO: div, mul, etc.
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("combine", pd.DataFrame(*frame_data), operator.add),
    ),
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("combine_first", pd.DataFrame(*frame_data)),
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("update", pd.DataFrame(*frame_data)),
        ),
        marks=not_implemented_mark,
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("pivot", columns="A")),
    (
        pd.DataFrame,
        ({"A": [1], "B": [1]},),
        operator.methodcaller("pivot_table", columns="A"),
    ),
    (
        pd.DataFrame,
        ({"A": [1], "B": [1]},),
        operator.methodcaller("pivot_table", columns="A", aggfunc=["mean", "sum"]),
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("stack")),
    (pd.DataFrame, frame_data, operator.methodcaller("explode", "A")),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("unstack")),
    (
        pd.DataFrame,
        ({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]},),
        operator.methodcaller("melt", id_vars=["A"], value_vars=["B"]),
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("map", lambda x: x)),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("merge", pd.DataFrame({"A": [1]})),
        ),
        marks=not_implemented_mark,
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("round", 2)),
    (pd.DataFrame, frame_data, operator.methodcaller("corr")),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("cov")),
        marks=[
            pytest.mark.filterwarnings("ignore::RuntimeWarning"),
        ],
    ),
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("corrwith", pd.DataFrame(*frame_data)),
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("count")),
    (pd.DataFrame, frame_data, operator.methodcaller("nunique")),
    (pd.DataFrame, frame_data, operator.methodcaller("idxmin")),
    (pd.DataFrame, frame_data, operator.methodcaller("idxmax")),
    (pd.DataFrame, frame_data, operator.methodcaller("mode")),
    (pd.Series, [0], operator.methodcaller("mode")),
    (pd.DataFrame, frame_data, operator.methodcaller("median")),
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("quantile", numeric_only=True),
    ),
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("quantile", q=[0.25, 0.75], numeric_only=True),
    ),
    (
        pd.DataFrame,
        ({"A": [pd.Timedelta(days=1), pd.Timedelta(days=2)]},),
        operator.methodcaller("quantile", numeric_only=False),
    ),
    (
        pd.DataFrame,
        ({"A": [np.datetime64("2022-01-01"), np.datetime64("2022-01-02")]},),
        operator.methodcaller("quantile", numeric_only=True),
    ),
    (
        pd.DataFrame,
        ({"A": [1]}, [pd.Period("2000", "D")]),
        operator.methodcaller("to_timestamp"),
    ),
    (
        pd.DataFrame,
        ({"A": [1]}, [pd.Timestamp("2000")]),
        operator.methodcaller("to_period", freq="D"),
    ),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("isin", [1])),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("isin", pd.Series([1]))),
    (
        pd.DataFrame,
        frame_mi_data,
        operator.methodcaller("isin", pd.DataFrame({"A": [1]})),
    ),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("droplevel", "A")),
    (pd.DataFrame, frame_data, operator.methodcaller("pop", "A")),
    # Squeeze on columns, otherwise we'll end up with a scalar
    (pd.DataFrame, frame_data, operator.methodcaller("squeeze", axis="columns")),
    (pd.Series, ([1, 2],), operator.methodcaller("squeeze")),
    (pd.Series, ([1, 2],), operator.methodcaller("rename_axis", index="a")),
    (pd.DataFrame, frame_data, operator.methodcaller("rename_axis", columns="a")),
    # Unary ops
    (pd.DataFrame, frame_data, operator.neg),
    (pd.Series, [1], operator.neg),
    (pd.DataFrame, frame_data, operator.pos),
    (pd.Series, [1], operator.pos),
    (pd.DataFrame, frame_data, operator.inv),
    (pd.Series, [1], operator.inv),
    (pd.DataFrame, frame_data, abs),
    (pd.Series, [1], abs),
    (pd.DataFrame, frame_data, round),
    (pd.Series, [1], round),
    (pd.DataFrame, frame_data, operator.methodcaller("take", [0, 0])),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("xs", "a")),
    (pd.Series, (1, mi), operator.methodcaller("xs", "a")),
    (pd.DataFrame, frame_data, operator.methodcaller("get", "A")),
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("reindex_like", pd.DataFrame({"A": [1, 2, 3]})),
    ),
    (
        pd.Series,
        frame_data,
        operator.methodcaller("reindex_like", pd.Series([0, 1, 2])),
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("add_prefix", "_")),
    (pd.DataFrame, frame_data, operator.methodcaller("add_suffix", "_")),
    (pd.Series, (1, ["a", "b"]), operator.methodcaller("add_prefix", "_")),
    (pd.Series, (1, ["a", "b"]), operator.methodcaller("add_suffix", "_")),
    (pd.Series, ([3, 2],), operator.methodcaller("sort_values")),
    (pd.Series, ([1] * 10,), operator.methodcaller("head")),
    (pd.DataFrame, ({"A": [1] * 10},), operator.methodcaller("head")),
    (pd.Series, ([1] * 10,), operator.methodcaller("tail")),
    (pd.DataFrame, ({"A": [1] * 10},), operator.methodcaller("tail")),
    (pd.Series, ([1, 2],), operator.methodcaller("sample", n=2, replace=True)),
    (pd.DataFrame, (frame_data,), operator.methodcaller("sample", n=2, replace=True)),
    (pd.Series, ([1, 2],), operator.methodcaller("astype", float)),
    (pd.DataFrame, frame_data, operator.methodcaller("astype", float)),
    (pd.Series, ([1, 2],), operator.methodcaller("copy")),
    (pd.DataFrame, frame_data, operator.methodcaller("copy")),
    (pd.Series, ([1, 2], None, object), operator.methodcaller("infer_objects")),
    (
        pd.DataFrame,
        ({"A": np.array([1, 2], dtype=object)},),
        operator.methodcaller("infer_objects"),
    ),
    (pd.Series, ([1, 2],), operator.methodcaller("convert_dtypes")),
    (pd.DataFrame, frame_data, operator.methodcaller("convert_dtypes")),
    (pd.Series, ([1, None, 3],), operator.methodcaller("interpolate")),
    (pd.DataFrame, ({"A": [1, None, 3]},), operator.methodcaller("interpolate")),
    (pd.Series, ([1, 2],), operator.methodcaller("clip", lower=1)),
    (pd.DataFrame, frame_data, operator.methodcaller("clip", lower=1)),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("asfreq", "h"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("asfreq", "h"),
    ),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("at_time", "12:00"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("at_time", "12:00"),
    ),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("between_time", "12:00", "13:00"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("between_time", "12:00", "13:00"),
    ),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("last", "3D"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("last", "3D"),
    ),
    (pd.Series, ([1, 2],), operator.methodcaller("rank")),
    (pd.DataFrame, frame_data, operator.methodcaller("rank")),
    (pd.Series, ([1, 2],), operator.methodcaller("where", np.array([True, False]))),
    (pd.DataFrame, frame_data, operator.methodcaller("where", np.array([[True]]))),
    (pd.Series, ([1, 2],), operator.methodcaller("mask", np.array([True, False]))),
    (pd.DataFrame, frame_data, operator.methodcaller("mask", np.array([[True]]))),
    (pd.Series, ([1, 2],), operator.methodcaller("truncate", before=0)),
    (pd.DataFrame, frame_data, operator.methodcaller("truncate", before=0)),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4, tz="UTC")),
        operator.methodcaller("tz_convert", "CET"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4, tz="UTC")),
        operator.methodcaller("tz_convert", "CET"),
    ),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("tz_localize", "CET"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("tz_localize", "CET"),
    ),
    (pd.Series, ([1, 2],), operator.methodcaller("describe")),
    (pd.DataFrame, frame_data, operator.methodcaller("describe")),
    (pd.Series, ([1, 2],), operator.methodcaller("pct_change")),
    (pd.DataFrame, frame_data, operator.methodcaller("pct_change")),
    (pd.Series, ([1],), operator.methodcaller("transform", lambda x: x - x.min())),
    (
        pd.DataFrame,
        frame_mi_data,
        operator.methodcaller("transform", lambda x: x - x.min()),
    ),
    (pd.Series, ([1],), operator.methodcaller("apply", lambda x: x)),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("apply", lambda x: x)),
    # Cumulative reductions
    (pd.Series, ([1],), operator.methodcaller("cumsum")),
    (pd.DataFrame, frame_data, operator.methodcaller("cumsum")),
    (pd.Series, ([1],), operator.methodcaller("cummin")),
    (pd.DataFrame, frame_data, operator.methodcaller("cummin")),
    (pd.Series, ([1],), operator.methodcaller("cummax")),
    (pd.DataFrame, frame_data, operator.methodcaller("cummax")),
    (pd.Series, ([1],), operator.methodcaller("cumprod")),
    (pd.DataFrame, frame_data, operator.methodcaller("cumprod")),
    # Reductions
    (pd.DataFrame, frame_data, operator.methodcaller("any")),
    (pd.DataFrame, frame_data, operator.methodcaller("all")),
    (pd.DataFrame, frame_data, operator.methodcaller("min")),
    (pd.DataFrame, frame_data, operator.methodcaller("max")),
    (pd.DataFrame, frame_data, operator.methodcaller("sum")),
    (pd.DataFrame, frame_data, operator.methodcaller("std")),
    (pd.DataFrame, frame_data, operator.methodcaller("mean")),
    (pd.DataFrame, frame_data, operator.methodcaller("prod")),
    (pd.DataFrame, frame_data, operator.methodcaller("sem")),
    (pd.DataFrame, frame_data, operator.methodcaller("skew")),
    (pd.DataFrame, frame_data, operator.methodcaller("kurt")),
]


def idfn(x):
    xpr = re.compile(r"'(.*)?'")
    m = xpr.search(str(x))
    if m:
        return m.group(1)
    else:
        return str(x)


@pytest.fixture(params=_all_methods, ids=lambda x: idfn(x[-1]))
def ndframe_method(request):
    """
    An NDFrame method returning an NDFrame.
    """
    return request.param


@pytest.mark.filterwarnings(
    "ignore:DataFrame.fillna with 'method' is deprecated:FutureWarning",
    "ignore:last is deprecated:FutureWarning",
)
def test_finalize_called(ndframe_method):
    cls, init_args, method = ndframe_method
    ndframe = cls(*init_args)

    ndframe.attrs = {"a": 1}
    result = method(ndframe)

    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "data",
    [
        pd.Series(1, pd.date_range("2000", periods=4)),
        pd.DataFrame({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
    ],
)
def test_finalize_first(data):
    deprecated_msg = "first is deprecated"

    data.attrs = {"a": 1}
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = data.first("3D")
        assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "data",
    [
        pd.Series(1, pd.date_range("2000", periods=4)),
        pd.DataFrame({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
    ],
)
def test_finalize_last(data):
    # GH 53710
    deprecated_msg = "last is deprecated"

    data.attrs = {"a": 1}
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = data.last("3D")
        assert result.attrs == {"a": 1}


@not_implemented_mark
def test_finalize_called_eval_numexpr():
    pytest.importorskip("numexpr")
    df = pd.DataFrame({"A": [1, 2]})
    df.attrs["A"] = 1
    result = df.eval("A + 1", engine="numexpr")
    assert result.attrs == {"A": 1}


# ----------------------------------------------------------------------------
# Binary operations


@pytest.mark.parametrize("annotate", ["left", "right", "both"])
@pytest.mark.parametrize(
    "args",
    [
        (1, pd.Series([1])),
        (1, pd.DataFrame({"A": [1]})),
        (pd.Series([1]), 1),
        (pd.DataFrame({"A": [1]}), 1),
        (pd.Series([1]), pd.Series([1])),
        (pd.DataFrame({"A": [1]}), pd.DataFrame({"A": [1]})),
        (pd.Series([1]), pd.DataFrame({"A": [1]})),
        (pd.DataFrame({"A": [1]}), pd.Series([1])),
    ],
    ids=lambda x: f"({type(x[0]).__name__},{type(x[1]).__name__})",
)
def test_binops(request, args, annotate, all_binary_operators):
    # This generates 624 tests... Is that needed?
    left, right = args
    if isinstance(left, (pd.DataFrame, pd.Series)):
        left.attrs = {}
    if isinstance(right, (pd.DataFrame, pd.Series)):
        right.attrs = {}

    if annotate == "left" and isinstance(left, int):
        pytest.skip("left is an int and doesn't support .attrs")
    if annotate == "right" and isinstance(right, int):
        pytest.skip("right is an int and doesn't support .attrs")

    if not (isinstance(left, int) or isinstance(right, int)) and annotate != "both":
        if not all_binary_operators.__name__.startswith("r"):
            if annotate == "right" and isinstance(left, type(right)):
                request.applymarker(
                    pytest.mark.xfail(
                        reason=f"{all_binary_operators} doesn't work when right has "
                        f"attrs and both are {type(left)}"
                    )
                )
            if not isinstance(left, type(right)):
                if annotate == "left" and isinstance(left, pd.Series):
                    request.applymarker(
                        pytest.mark.xfail(
                            reason=f"{all_binary_operators} doesn't work when the "
                            "objects are different Series has attrs"
                        )
                    )
                elif annotate == "right" and isinstance(right, pd.Series):
                    request.applymarker(
                        pytest.mark.xfail(
                            reason=f"{all_binary_operators} doesn't work when the "
                            "objects are different Series has attrs"
                        )
                    )
        else:
            if annotate == "left" and isinstance(left, type(right)):
                request.applymarker(
                    pytest.mark.xfail(
                        reason=f"{all_binary_operators} doesn't work when left has "
                        f"attrs and both are {type(left)}"
                    )
                )
            if not isinstance(left, type(right)):
                if annotate == "right" and isinstance(right, pd.Series):
                    request.applymarker(
                        pytest.mark.xfail(
                            reason=f"{all_binary_operators} doesn't work when the "
                            "objects are different Series has attrs"
                        )
                    )
                elif annotate == "left" and isinstance(left, pd.Series):
                    request.applymarker(
                        pytest.mark.xfail(
                            reason=f"{all_binary_operators} doesn't work when the "
                            "objects are different Series has attrs"
                        )
                    )
    if annotate in {"left", "both"} and not isinstance(left, int):
        left.attrs = {"a": 1}
    if annotate in {"right", "both"} and not isinstance(right, int):
        right.attrs = {"a": 1}

    is_cmp = all_binary_operators in [
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
    ]
    if is_cmp and isinstance(left, pd.DataFrame) and isinstance(right, pd.Series):
        # in 2.0 silent alignment on comparisons was removed xref GH#28759
        left, right = left.align(right, axis=1, copy=False)
    elif is_cmp and isinstance(left, pd.Series) and isinstance(right, pd.DataFrame):
        right, left = right.align(left, axis=1, copy=False)

    result = all_binary_operators(left, right)
    assert result.attrs == {"a": 1}


# ----------------------------------------------------------------------------
# Accessors


@pytest.mark.parametrize(
    "method",
    [
        operator.methodcaller("capitalize"),
        operator.methodcaller("casefold"),
        operator.methodcaller("cat", ["a"]),
        operator.methodcaller("contains", "a"),
        operator.methodcaller("count", "a"),
        operator.methodcaller("encode", "utf-8"),
        operator.methodcaller("endswith", "a"),
        operator.methodcaller("extract", r"(\w)(\d)"),
        operator.methodcaller("extract", r"(\w)(\d)", expand=False),
        operator.methodcaller("find", "a"),
        operator.methodcaller("findall", "a"),
        operator.methodcaller("get", 0),
        operator.methodcaller("index", "a"),
        operator.methodcaller("len"),
        operator.methodcaller("ljust", 4),
        operator.methodcaller("lower"),
        operator.methodcaller("lstrip"),
        operator.methodcaller("match", r"\w"),
        operator.methodcaller("normalize", "NFC"),
        operator.methodcaller("pad", 4),
        operator.methodcaller("partition", "a"),
        operator.methodcaller("repeat", 2),
        operator.methodcaller("replace", "a", "b"),
        operator.methodcaller("rfind", "a"),
        operator.methodcaller("rindex", "a"),
        operator.methodcaller("rjust", 4),
        operator.methodcaller("rpartition", "a"),
        operator.methodcaller("rstrip"),
        operator.methodcaller("slice", 4),
        operator.methodcaller("slice_replace", 1, repl="a"),
        operator.methodcaller("startswith", "a"),
        operator.methodcaller("strip"),
        operator.methodcaller("swapcase"),
        operator.methodcaller("translate", {"a": "b"}),
        operator.methodcaller("upper"),
        operator.methodcaller("wrap", 4),
        operator.methodcaller("zfill", 4),
        operator.methodcaller("isalnum"),
        operator.methodcaller("isalpha"),
        operator.methodcaller("isdigit"),
        operator.methodcaller("isspace"),
        operator.methodcaller("islower"),
        operator.methodcaller("isupper"),
        operator.methodcaller("istitle"),
        operator.methodcaller("isnumeric"),
        operator.methodcaller("isdecimal"),
        operator.methodcaller("get_dummies"),
    ],
    ids=idfn,
)
def test_string_method(method):
    s = pd.Series(["a1"])
    s.attrs = {"a": 1}
    result = method(s.str)
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "method",
    [
        operator.methodcaller("to_period"),
        operator.methodcaller("tz_localize", "CET"),
        operator.methodcaller("normalize"),
        operator.methodcaller("strftime", "%Y"),
        operator.methodcaller("round", "h"),
        operator.methodcaller("floor", "h"),
        operator.methodcaller("ceil", "h"),
        operator.methodcaller("month_name"),
        operator.methodcaller("day_name"),
    ],
    ids=idfn,
)
def test_datetime_method(method):
    s = pd.Series(pd.date_range("2000", periods=4))
    s.attrs = {"a": 1}
    result = method(s.dt)
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "attr",
    [
        "date",
        "time",
        "timetz",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_leap_year",
        "daysinmonth",
        "days_in_month",
    ],
)
def test_datetime_property(attr):
    s = pd.Series(pd.date_range("2000", periods=4))
    s.attrs = {"a": 1}
    result = getattr(s.dt, attr)
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "attr", ["days", "seconds", "microseconds", "nanoseconds", "components"]
)
def test_timedelta_property(attr):
    s = pd.Series(pd.timedelta_range("2000", periods=4))
    s.attrs = {"a": 1}
    result = getattr(s.dt, attr)
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize("method", [operator.methodcaller("total_seconds")])
def test_timedelta_methods(method):
    s = pd.Series(pd.timedelta_range("2000", periods=4))
    s.attrs = {"a": 1}
    result = method(s.dt)
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "method",
    [
        operator.methodcaller("add_categories", ["c"]),
        operator.methodcaller("as_ordered"),
        operator.methodcaller("as_unordered"),
        lambda x: getattr(x, "codes"),
        operator.methodcaller("remove_categories", "a"),
        operator.methodcaller("remove_unused_categories"),
        operator.methodcaller("rename_categories", {"a": "A", "b": "B"}),
        operator.methodcaller("reorder_categories", ["b", "a"]),
        operator.methodcaller("set_categories", ["A", "B"]),
    ],
)
@not_implemented_mark
def test_categorical_accessor(method):
    s = pd.Series(["a", "b"], dtype="category")
    s.attrs = {"a": 1}
    result = method(s.cat)
    assert result.attrs == {"a": 1}


# ----------------------------------------------------------------------------
# Groupby


@pytest.mark.parametrize(
    "obj", [pd.Series([0, 0]), pd.DataFrame({"A": [0, 1], "B": [1, 2]})]
)
@pytest.mark.parametrize(
    "method",
    [
        operator.methodcaller("sum"),
        lambda x: x.apply(lambda y: y),
        lambda x: x.agg("sum"),
        lambda x: x.agg("mean"),
        lambda x: x.agg("median"),
    ],
)
def test_groupby_finalize(obj, method):
    obj.attrs = {"a": 1}
    result = method(obj.groupby([0, 0], group_keys=False))
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "obj", [pd.Series([0, 0]), pd.DataFrame({"A": [0, 1], "B": [1, 2]})]
)
@pytest.mark.parametrize(
    "method",
    [
        lambda x: x.agg(["sum", "count"]),
        lambda x: x.agg("std"),
        lambda x: x.agg("var"),
        lambda x: x.agg("sem"),
        lambda x: x.agg("size"),
        lambda x: x.agg("ohlc"),
    ],
)
@not_implemented_mark
def test_groupby_finalize_not_implemented(obj, method):
    obj.attrs = {"a": 1}
    result = method(obj.groupby([0, 0]))
    assert result.attrs == {"a": 1}


def test_finalize_frame_series_name():
    # https://github.com/pandas-dev/pandas/pull/37186/files#r506978889
    # ensure we don't copy the column `name` to the Series.
    df = pd.DataFrame({"name": [1, 2]})
    result = pd.Series([1, 2]).__finalize__(df)
    assert result.name is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\formatters\_mapping.py ===
# Automatically generated by scripts/gen_mapfiles.py.
# DO NOT EDIT BY HAND; run `tox -e mapfiles` instead.

FORMATTERS = {
    'BBCodeFormatter': ('pygments.formatters.bbcode', 'BBCode', ('bbcode', 'bb'), (), 'Format tokens with BBcodes. These formatting codes are used by many bulletin boards, so you can highlight your sourcecode with pygments before posting it there.'),
    'BmpImageFormatter': ('pygments.formatters.img', 'img_bmp', ('bmp', 'bitmap'), ('*.bmp',), 'Create a bitmap image from source code. This uses the Python Imaging Library to generate a pixmap from the source code.'),
    'GifImageFormatter': ('pygments.formatters.img', 'img_gif', ('gif',), ('*.gif',), 'Create a GIF image from source code. This uses the Python Imaging Library to generate a pixmap from the source code.'),
    'GroffFormatter': ('pygments.formatters.groff', 'groff', ('groff', 'troff', 'roff'), (), 'Format tokens with groff escapes to change their color and font style.'),
    'HtmlFormatter': ('pygments.formatters.html', 'HTML', ('html',), ('*.html', '*.htm'), "Format tokens as HTML 4 ``<span>`` tags. By default, the content is enclosed in a ``<pre>`` tag, itself wrapped in a ``<div>`` tag (but see the `nowrap` option). The ``<div>``'s CSS class can be set by the `cssclass` option."),
    'IRCFormatter': ('pygments.formatters.irc', 'IRC', ('irc', 'IRC'), (), 'Format tokens with IRC color sequences'),
    'ImageFormatter': ('pygments.formatters.img', 'img', ('img', 'IMG', 'png'), ('*.png',), 'Create a PNG image from source code. This uses the Python Imaging Library to generate a pixmap from the source code.'),
    'JpgImageFormatter': ('pygments.formatters.img', 'img_jpg', ('jpg', 'jpeg'), ('*.jpg',), 'Create a JPEG image from source code. This uses the Python Imaging Library to generate a pixmap from the source code.'),
    'LatexFormatter': ('pygments.formatters.latex', 'LaTeX', ('latex', 'tex'), ('*.tex',), 'Format tokens as LaTeX code. This needs the `fancyvrb` and `color` standard packages.'),
    'NullFormatter': ('pygments.formatters.other', 'Text only', ('text', 'null'), ('*.txt',), 'Output the text unchanged without any formatting.'),
    'PangoMarkupFormatter': ('pygments.formatters.pangomarkup', 'Pango Markup', ('pango', 'pangomarkup'), (), 'Format tokens as Pango Markup code. It can then be rendered to an SVG.'),
    'RawTokenFormatter': ('pygments.formatters.other', 'Raw tokens', ('raw', 'tokens'), ('*.raw',), 'Format tokens as a raw representation for storing token streams.'),
    'RtfFormatter': ('pygments.formatters.rtf', 'RTF', ('rtf',), ('*.rtf',), 'Format tokens as RTF markup. This formatter automatically outputs full RTF documents with color information and other useful stuff. Perfect for Copy and Paste into Microsoft(R) Word(R) documents.'),
    'SvgFormatter': ('pygments.formatters.svg', 'SVG', ('svg',), ('*.svg',), 'Format tokens as an SVG graphics file.  This formatter is still experimental. Each line of code is a ``<text>`` element with explicit ``x`` and ``y`` coordinates containing ``<tspan>`` elements with the individual token styles.'),
    'Terminal256Formatter': ('pygments.formatters.terminal256', 'Terminal256', ('terminal256', 'console256', '256'), (), 'Format tokens with ANSI color sequences, for output in a 256-color terminal or console.  Like in `TerminalFormatter` color sequences are terminated at newlines, so that paging the output works correctly.'),
    'TerminalFormatter': ('pygments.formatters.terminal', 'Terminal', ('terminal', 'console'), (), 'Format tokens with ANSI color sequences, for output in a text console. Color sequences are terminated at newlines, so that paging the output works correctly.'),
    'TerminalTrueColorFormatter': ('pygments.formatters.terminal256', 'TerminalTrueColor', ('terminal16m', 'console16m', '16m'), (), 'Format tokens with ANSI color sequences, for output in a true-color terminal or console.  Like in `TerminalFormatter` color sequences are terminated at newlines, so that paging the output works correctly.'),
    'TestcaseFormatter': ('pygments.formatters.other', 'Testcase', ('testcase',), (), 'Format tokens as appropriate for a new testcase.'),
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\packages.py ===
import sys

from .compat import chardet

# This code exists for backwards compatibility reasons.
# I don't like it either. Just look the other way. :)

for package in ("urllib3", "idna"):
    locals()[package] = __import__(package)
    # This traversal is apparently necessary such that the identities are
    # preserved (requests.packages.urllib3.* is urllib3.*)
    for mod in list(sys.modules):
        if mod == package or mod.startswith(f"{package}."):
            sys.modules[f"requests.packages.{mod}"] = sys.modules[mod]

if chardet is not None:
    target = chardet.__name__
    for mod in list(sys.modules):
        if mod == target or mod.startswith(f"{target}."):
            imported_mod = sys.modules[mod]
            sys.modules[f"requests.packages.{mod}"] = imported_mod
            mod = mod.replace(target, "chardet")
            sys.modules[f"requests.packages.{mod}"] = imported_mod

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\bidi_connection.py ===
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


class BidiConnection:
    def __init__(self, session, cdp, devtools_import) -> None:
        self.session = session
        self.cdp = cdp
        self.devtools = devtools_import

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\safari\permissions.py ===
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
"""The Permission implementation."""


class Permission:
    """Set of supported permissions."""

    GET_USER_MEDIA = "getUserMedia"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\coreerrors.py ===
"""Definitions of common exceptions for :mod:`sympy.core` module. """

from typing import Callable


class BaseCoreError(Exception):
    """Base class for core related exceptions. """


class NonCommutativeExpression(BaseCoreError):
    """Raised when expression didn't have commutative property. """


class LazyExceptionMessage:
    """Wrapper class that lets you specify an expensive to compute
    error message that is only evaluated if the error is rendered."""
    callback: Callable[[], str]

    def __init__(self, callback: Callable[[], str]):
        self.callback = callback

    def __str__(self):
        return self.callback()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\benchmarks\bench_expand.py ===
from sympy.core import symbols, I

x, y, z = symbols('x,y,z')

p = 3*x**2*y*z**7 + 7*x*y*z**2 + 4*x + x*y**4
e = (x + y + z + 1)**32


def timeit_expand_nothing_todo():
    p.expand()


def bench_expand_32():
    """(x+y+z+1)**32  -> expand"""
    e.expand()


def timeit_expand_complex_number_1():
    ((2 + 3*I)**1000).expand(complex=True)


def timeit_expand_complex_number_2():
    ((2 + 3*I/4)**1000).expand(complex=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\__init__.py ===
"""A module that handles series: find a limit, order the series etc.
"""
from .order import Order
from .limits import limit, Limit
from .gruntz import gruntz
from .series import series
from .approximants import approximants
from .residues import residue
from .sequences import SeqPer, SeqFormula, sequence, SeqAdd, SeqMul
from .fourier import fourier_series
from .formal import fps
from .limitseq import difference_delta, limit_seq

from sympy.core.singleton import S
EmptySequence = S.EmptySequence

O = Order

__all__ = ['Order', 'O', 'limit', 'Limit', 'gruntz', 'series', 'approximants',
        'residue', 'EmptySequence', 'SeqPer', 'SeqFormula', 'sequence',
        'SeqAdd', 'SeqMul', 'fourier_series', 'fps', 'difference_delta',
        'limit_seq'
        ]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\__init__.py ===
"""A module to manipulate symbolic objects with indices including tensors

"""
from .indexed import IndexedBase, Idx, Indexed
from .index_methods import get_contraction_structure, get_indices
from .functions import shape
from .array import (MutableDenseNDimArray, ImmutableDenseNDimArray,
    MutableSparseNDimArray, ImmutableSparseNDimArray, NDimArray, tensorproduct,
    tensorcontraction, tensordiagonal, derive_by_array, permutedims, Array,
    DenseNDimArray, SparseNDimArray,)

__all__ = [
    'IndexedBase', 'Idx', 'Indexed',

    'get_contraction_structure', 'get_indices',

    'shape',

    'MutableDenseNDimArray', 'ImmutableDenseNDimArray',
    'MutableSparseNDimArray', 'ImmutableSparseNDimArray', 'NDimArray',
    'tensorproduct', 'tensorcontraction', 'tensordiagonal', 'derive_by_array', 'permutedims',
    'Array', 'DenseNDimArray', 'SparseNDimArray',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\abc.py ===
# This is a public namespace, so we don't want to expose any non-underscored
# attributes that aren't actually part of our public API. But it's very
# annoying to carefully always use underscored names for module-level
# temporaries, imports, etc. when implementing the module. So we put the
# implementation in an underscored module, and then re-export the public parts
# here.

# Uses `from x import y as y` for compatibility with `pyright --verifytypes` (#2625)
from ._abc import (
    AsyncResource as AsyncResource,
    Channel as Channel,
    Clock as Clock,
    HalfCloseableStream as HalfCloseableStream,
    HostnameResolver as HostnameResolver,
    Instrument as Instrument,
    Listener as Listener,
    ReceiveChannel as ReceiveChannel,
    ReceiveStream as ReceiveStream,
    SendChannel as SendChannel,
    SendStream as SendStream,
    SocketFactory as SocketFactory,
    Stream as Stream,
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\type_tests\subprocesses.py ===
import sys

import trio


async def test() -> None:
    # this could test more by using platform checks, but currently this
    # is just regression tests + sanity checks.
    await trio.run_process("python", executable="ls")
    await trio.lowlevel.open_process("python", executable="ls")

    # note: there's no error code on the type ignore as it varies
    # between platforms.
    await trio.run_process("python", capture_stdout=True)
    await trio.lowlevel.open_process("python", capture_stdout=True)  # type: ignore

    if sys.platform != "win32" and sys.version_info >= (3, 9):
        await trio.run_process("python", extra_groups=[5])
        await trio.lowlevel.open_process("python", extra_groups=[5])

        # 3.11+:
        await trio.run_process("python", process_group=5)  # type: ignore
        await trio.lowlevel.open_process("python", process_group=5)  # type: ignore

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\archive.py ===
import os
import zipfile


class LinuxZipFileWithPermissions(zipfile.ZipFile):
    """Class for extract files in linux with right permissions"""

    def extract(self, member, path=None, pwd=None):
        if not isinstance(member, zipfile.ZipInfo):
            member = self.getinfo(member)

        if path is None:
            path = os.getcwd()

        ret_val = self._extract_member(member, path, pwd)  # noqa
        attr = member.external_attr >> 16
        os.chmod(ret_val, attr)
        return ret_val


class Archive(object):
    def __init__(self, path: str):
        self.file_path = path

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_match.py ===
from sympy import abc
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import signsimp

from sympy.testing.pytest import XFAIL


def test_symbol():
    x = Symbol('x')
    a, b, c, p, q = map(Wild, 'abcpq')

    e = x
    assert e.match(x) == {}
    assert e.matches(x) == {}
    assert e.match(a) == {a: x}

    e = Rational(5)
    assert e.match(c) == {c: 5}
    assert e.match(e) == {}
    assert e.match(e + 1) is None


def test_add():
    x, y, a, b, c = map(Symbol, 'xyabc')
    p, q, r = map(Wild, 'pqr')

    e = a + b
    assert e.match(p + b) == {p: a}
    assert e.match(p + a) == {p: b}

    e = 1 + b
    assert e.match(p + b) == {p: 1}

    e = a + b + c
    assert e.match(a + p + c) == {p: b}
    assert e.match(b + p + c) == {p: a}

    e = a + b + c + x
    assert e.match(a + p + x + c) == {p: b}
    assert e.match(b + p + c + x) == {p: a}
    assert e.match(b) is None
    assert e.match(b + p) == {p: a + c + x}
    assert e.match(a + p + c) == {p: b + x}
    assert e.match(b + p + c) == {p: a + x}

    e = 4*x + 5
    assert e.match(4*x + p) == {p: 5}
    assert e.match(3*x + p) == {p: x + 5}
    assert e.match(p*x + 5) == {p: 4}


def test_power():
    x, y, a, b, c = map(Symbol, 'xyabc')
    p, q, r = map(Wild, 'pqr')

    e = (x + y)**a
    assert e.match(p**q) == {p: x + y, q: a}
    assert e.match(p**p) is None

    e = (x + y)**(x + y)
    assert e.match(p**p) == {p: x + y}
    assert e.match(p**q) == {p: x + y, q: x + y}

    e = (2*x)**2
    assert e.match(p*q**r) == {p: 4, q: x, r: 2}

    e = Integer(1)
    assert e.match(x**p) == {p: 0}


def test_match_exclude():
    x = Symbol('x')
    y = Symbol('y')
    p = Wild("p")
    q = Wild("q")
    r = Wild("r")

    e = Rational(6)
    assert e.match(2*p) == {p: 3}

    e = 3/(4*x + 5)
    assert e.match(3/(p*x + q)) == {p: 4, q: 5}

    e = 3/(4*x + 5)
    assert e.match(p/(q*x + r)) == {p: 3, q: 4, r: 5}

    e = 2/(x + 1)
    assert e.match(p/(q*x + r)) == {p: 2, q: 1, r: 1}

    e = 1/(x + 1)
    assert e.match(p/(q*x + r)) == {p: 1, q: 1, r: 1}

    e = 4*x + 5
    assert e.match(p*x + q) == {p: 4, q: 5}

    e = 4*x + 5*y + 6
    assert e.match(p*x + q*y + r) == {p: 4, q: 5, r: 6}

    a = Wild('a', exclude=[x])

    e = 3*x
    assert e.match(p*x) == {p: 3}
    assert e.match(a*x) == {a: 3}

    e = 3*x**2
    assert e.match(p*x) == {p: 3*x}
    assert e.match(a*x) is None

    e = 3*x + 3 + 6/x
    assert e.match(p*x**2 + p*x + 2*p) == {p: 3/x}
    assert e.match(a*x**2 + a*x + 2*a) is None


def test_mul():
    x, y, a, b, c = map(Symbol, 'xyabc')
    p, q = map(Wild, 'pq')

    e = 4*x
    assert e.match(p*x) == {p: 4}
    assert e.match(p*y) is None
    assert e.match(e + p*y) == {p: 0}

    e = a*x*b*c
    assert e.match(p*x) == {p: a*b*c}
    assert e.match(c*p*x) == {p: a*b}

    e = (a + b)*(a + c)
    assert e.match((p + b)*(p + c)) == {p: a}

    e = x
    assert e.match(p*x) == {p: 1}

    e = exp(x)
    assert e.match(x**p*exp(x*q)) == {p: 0, q: 1}

    e = I*Poly(x, x)
    assert e.match(I*p) == {p: x}


def test_mul_noncommutative():
    x, y = symbols('x y')
    A, B, C = symbols('A B C', commutative=False)
    u, v = symbols('u v', cls=Wild)
    w, z = symbols('w z', cls=Wild, commutative=False)

    assert (u*v).matches(x) in ({v: x, u: 1}, {u: x, v: 1})
    assert (u*v).matches(x*y) in ({v: y, u: x}, {u: y, v: x})
    assert (u*v).matches(A) is None
    assert (u*v).matches(A*B) is None
    assert (u*v).matches(x*A) is None
    assert (u*v).matches(x*y*A) is None
    assert (u*v).matches(x*A*B) is None
    assert (u*v).matches(x*y*A*B) is None

    assert (v*w).matches(x) is None
    assert (v*w).matches(x*y) is None
    assert (v*w).matches(A) == {w: A, v: 1}
    assert (v*w).matches(A*B) == {w: A*B, v: 1}
    assert (v*w).matches(x*A) == {w: A, v: x}
    assert (v*w).matches(x*y*A) == {w: A, v: x*y}
    assert (v*w).matches(x*A*B) == {w: A*B, v: x}
    assert (v*w).matches(x*y*A*B) == {w: A*B, v: x*y}

    assert (v*w).matches(-x) is None
    assert (v*w).matches(-x*y) is None
    assert (v*w).matches(-A) == {w: A, v: -1}
    assert (v*w).matches(-A*B) == {w: A*B, v: -1}
    assert (v*w).matches(-x*A) == {w: A, v: -x}
    assert (v*w).matches(-x*y*A) == {w: A, v: -x*y}
    assert (v*w).matches(-x*A*B) == {w: A*B, v: -x}
    assert (v*w).matches(-x*y*A*B) == {w: A*B, v: -x*y}

    assert (w*z).matches(x) is None
    assert (w*z).matches(x*y) is None
    assert (w*z).matches(A) is None
    assert (w*z).matches(A*B) == {w: A, z: B}
    assert (w*z).matches(B*A) == {w: B, z: A}
    assert (w*z).matches(A*B*C) in [{w: A, z: B*C}, {w: A*B, z: C}]
    assert (w*z).matches(x*A) is None
    assert (w*z).matches(x*y*A) is None
    assert (w*z).matches(x*A*B) is None
    assert (w*z).matches(x*y*A*B) is None

    assert (w*A).matches(A) is None
    assert (A*w*B).matches(A*B) is None

    assert (u*w*z).matches(x) is None
    assert (u*w*z).matches(x*y) is None
    assert (u*w*z).matches(A) is None
    assert (u*w*z).matches(A*B) == {u: 1, w: A, z: B}
    assert (u*w*z).matches(B*A) == {u: 1, w: B, z: A}
    assert (u*w*z).matches(x*A) is None
    assert (u*w*z).matches(x*y*A) is None
    assert (u*w*z).matches(x*A*B) == {u: x, w: A, z: B}
    assert (u*w*z).matches(x*B*A) == {u: x, w: B, z: A}
    assert (u*w*z).matches(x*y*A*B) == {u: x*y, w: A, z: B}
    assert (u*w*z).matches(x*y*B*A) == {u: x*y, w: B, z: A}

    assert (u*A).matches(x*A) == {u: x}
    assert (u*A).matches(x*A*B) is None
    assert (u*B).matches(x*A) is None
    assert (u*A*B).matches(x*A*B) == {u: x}
    assert (u*A*B).matches(x*B*A) is None
    assert (u*A*B).matches(x*A) is None

    assert (u*w*A).matches(x*A*B) is None
    assert (u*w*B).matches(x*A*B) == {u: x, w: A}

    assert (u*v*A*B).matches(x*A*B) in [{u: x, v: 1}, {v: x, u: 1}]
    assert (u*v*A*B).matches(x*B*A) is None
    assert (u*v*A*B).matches(u*v*A*C) is None


def test_mul_noncommutative_mismatch():
    A, B, C = symbols('A B C', commutative=False)
    w = symbols('w', cls=Wild, commutative=False)

    assert (w*B*w).matches(A*B*A) == {w: A}
    assert (w*B*w).matches(A*C*B*A*C) == {w: A*C}
    assert (w*B*w).matches(A*C*B*A*B) is None
    assert (w*B*w).matches(A*B*C) is None
    assert (w*w*C).matches(A*B*C) is None


def test_mul_noncommutative_pow():
    A, B, C = symbols('A B C', commutative=False)
    w = symbols('w', cls=Wild, commutative=False)

    assert (A*B*w).matches(A*B**2) == {w: B}
    assert (A*(B**2)*w*(B**3)).matches(A*B**8) == {w: B**3}
    assert (A*B*w*C).matches(A*(B**4)*C) == {w: B**3}

    assert (A*B*(w**(-1))).matches(A*B*(C**(-1))) == {w: C}
    assert (A*(B*w)**(-1)*C).matches(A*(B*C)**(-1)*C) == {w: C}

    assert ((w**2)*B*C).matches((A**2)*B*C) == {w: A}
    assert ((w**2)*B*(w**3)).matches((A**2)*B*(A**3)) == {w: A}
    assert ((w**2)*B*(w**4)).matches((A**2)*B*(A**2)) is None

def test_complex():
    a, b, c = map(Symbol, 'abc')
    x, y = map(Wild, 'xy')

    assert (1 + I).match(x + I) == {x: 1}
    assert (a + I).match(x + I) == {x: a}
    assert (2*I).match(x*I) == {x: 2}
    assert (a*I).match(x*I) == {x: a}
    assert (a*I).match(x*y) == {x: I, y: a}
    assert (2*I).match(x*y) == {x: 2, y: I}
    assert (a + b*I).match(x + y*I) == {x: a, y: b}


def test_functions():
    from sympy.core.function import WildFunction
    x = Symbol('x')
    g = WildFunction('g')
    p = Wild('p')
    q = Wild('q')

    f = cos(5*x)
    notf = x
    assert f.match(p*cos(q*x)) == {p: 1, q: 5}
    assert f.match(p*g) == {p: 1, g: cos(5*x)}
    assert notf.match(g) is None


@XFAIL
def test_functions_X1():
    from sympy.core.function import WildFunction
    x = Symbol('x')
    g = WildFunction('g')
    p = Wild('p')
    q = Wild('q')

    f = cos(5*x)
    assert f.match(p*g(q*x)) == {p: 1, g: cos, q: 5}


def test_interface():
    x, y = map(Symbol, 'xy')
    p, q = map(Wild, 'pq')

    assert (x + 1).match(p + 1) == {p: x}
    assert (x*3).match(p*3) == {p: x}
    assert (x**3).match(p**3) == {p: x}
    assert (x*cos(y)).match(p*cos(q)) == {p: x, q: y}

    assert (x*y).match(p*q) in [{p:x, q:y}, {p:y, q:x}]
    assert (x + y).match(p + q) in [{p:x, q:y}, {p:y, q:x}]
    assert (x*y + 1).match(p*q) in [{p:1, q:1 + x*y}, {p:1 + x*y, q:1}]


def test_derivative1():
    x, y = map(Symbol, 'xy')
    p, q = map(Wild, 'pq')

    f = Function('f', nargs=1)
    fd = Derivative(f(x), x)

    assert fd.match(p) == {p: fd}
    assert (fd + 1).match(p + 1) == {p: fd}
    assert (fd).match(fd) == {}
    assert (3*fd).match(p*fd) is not None
    assert (3*fd - 1).match(p*fd + q) == {p: 3, q: -1}


def test_derivative_bug1():
    f = Function("f")
    x = Symbol("x")
    a = Wild("a", exclude=[f, x])
    b = Wild("b", exclude=[f])
    pattern = a * Derivative(f(x), x, x) + b
    expr = Derivative(f(x), x) + x**2
    d1 = {b: x**2}
    d2 = pattern.xreplace(d1).matches(expr, d1)
    assert d2 is None


def test_derivative2():
    f = Function("f")
    x = Symbol("x")
    a = Wild("a", exclude=[f, x])
    b = Wild("b", exclude=[f])
    e = Derivative(f(x), x)
    assert e.match(Derivative(f(x), x)) == {}
    assert e.match(Derivative(f(x), x, x)) is None
    e = Derivative(f(x), x, x)
    assert e.match(Derivative(f(x), x)) is None
    assert e.match(Derivative(f(x), x, x)) == {}
    e = Derivative(f(x), x) + x**2
    assert e.match(a*Derivative(f(x), x) + b) == {a: 1, b: x**2}
    assert e.match(a*Derivative(f(x), x, x) + b) is None
    e = Derivative(f(x), x, x) + x**2
    assert e.match(a*Derivative(f(x), x) + b) is None
    assert e.match(a*Derivative(f(x), x, x) + b) == {a: 1, b: x**2}


def test_match_deriv_bug1():
    n = Function('n')
    l = Function('l')

    x = Symbol('x')
    p = Wild('p')

    e = diff(l(x), x)/x - diff(diff(n(x), x), x)/2 - \
        diff(n(x), x)**2/4 + diff(n(x), x)*diff(l(x), x)/4
    e = e.subs(n(x), -l(x)).doit()
    t = x*exp(-l(x))
    t2 = t.diff(x, x)/t
    assert e.match( (p*t2).expand() ) == {p: Rational(-1, 2)}


def test_match_bug2():
    x, y = map(Symbol, 'xy')
    p, q, r = map(Wild, 'pqr')
    res = (x + y).match(p + q + r)
    assert (p + q + r).subs(res) == x + y


def test_match_bug3():
    x, a, b = map(Symbol, 'xab')
    p = Wild('p')
    assert (b*x*exp(a*x)).match(x*exp(p*x)) is None


def test_match_bug4():
    x = Symbol('x')
    p = Wild('p')
    e = x
    assert e.match(-p*x) == {p: -1}


def test_match_bug5():
    x = Symbol('x')
    p = Wild('p')
    e = -x
    assert e.match(-p*x) == {p: 1}


def test_match_bug6():
    x = Symbol('x')
    p = Wild('p')
    e = x
    assert e.match(3*p*x) == {p: Rational(1)/3}


def test_match_polynomial():
    x = Symbol('x')
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x])
    c = Wild('c', exclude=[x])
    d = Wild('d', exclude=[x])

    eq = 4*x**3 + 3*x**2 + 2*x + 1
    pattern = a*x**3 + b*x**2 + c*x + d
    assert eq.match(pattern) == {a: 4, b: 3, c: 2, d: 1}
    assert (eq - 3*x**2).match(pattern) == {a: 4, b: 0, c: 2, d: 1}
    assert (x + sqrt(2) + 3).match(a + b*x + c*x**2) == \
        {b: 1, a: sqrt(2) + 3, c: 0}


def test_exclude():
    x, y, a = map(Symbol, 'xya')
    p = Wild('p', exclude=[1, x])
    q = Wild('q')
    r = Wild('r', exclude=[sin, y])

    assert sin(x).match(r) is None
    assert cos(y).match(r) is None

    e = 3*x**2 + y*x + a
    assert e.match(p*x**2 + q*x + r) == {p: 3, q: y, r: a}

    e = x + 1
    assert e.match(x + p) is None
    assert e.match(p + 1) is None
    assert e.match(x + 1 + p) == {p: 0}

    e = cos(x) + 5*sin(y)
    assert e.match(r) is None
    assert e.match(cos(y) + r) is None
    assert e.match(r + p*sin(q)) == {r: cos(x), p: 5, q: y}


def test_floats():
    a, b = map(Wild, 'ab')

    e = cos(0.12345, evaluate=False)**2
    r = e.match(a*cos(b)**2)
    assert r == {a: 1, b: Float(0.12345)}


def test_Derivative_bug1():
    f = Function("f")
    x = abc.x
    a = Wild("a", exclude=[f(x)])
    b = Wild("b", exclude=[f(x)])
    eq = f(x).diff(x)
    assert eq.match(a*Derivative(f(x), x) + b) == {a: 1, b: 0}


def test_match_wild_wild():
    p = Wild('p')
    q = Wild('q')
    r = Wild('r')

    assert p.match(q + r) in [ {q: p, r: 0}, {q: 0, r: p} ]
    assert p.match(q*r) in [ {q: p, r: 1}, {q: 1, r: p} ]

    p = Wild('p')
    q = Wild('q', exclude=[p])
    r = Wild('r')

    assert p.match(q + r) == {q: 0, r: p}
    assert p.match(q*r) == {q: 1, r: p}

    p = Wild('p')
    q = Wild('q', exclude=[p])
    r = Wild('r', exclude=[p])

    assert p.match(q + r) is None
    assert p.match(q*r) is None


def test__combine_inverse():
    x, y = symbols("x y")
    assert Mul._combine_inverse(x*I*y, x*I) == y
    assert Mul._combine_inverse(x*x**(1 + y), x**(1 + y)) == x
    assert Mul._combine_inverse(x*I*y, y*I) == x
    assert Mul._combine_inverse(oo*I*y, y*I) is oo
    assert Mul._combine_inverse(oo*I*y, oo*I) == y
    assert Mul._combine_inverse(oo*I*y, oo*I) == y
    assert Mul._combine_inverse(oo*y, -oo) == -y
    assert Mul._combine_inverse(-oo*y, oo) == -y
    assert Mul._combine_inverse((1-exp(x/y)),(exp(x/y)-1)) == -1
    assert Add._combine_inverse(oo, oo) is S.Zero
    assert Add._combine_inverse(oo*I, oo*I) is S.Zero
    assert Add._combine_inverse(x*oo, x*oo) is S.Zero
    assert Add._combine_inverse(-x*oo, -x*oo) is S.Zero
    assert Add._combine_inverse((x - oo)*(x + oo), -oo)


def test_issue_3773():
    x = symbols('x')
    z, phi, r = symbols('z phi r')
    c, A, B, N = symbols('c A B N', cls=Wild)
    l = Wild('l', exclude=(0,))

    eq = z * sin(2*phi) * r**7
    matcher = c * sin(phi*N)**l * r**A * log(r)**B

    assert eq.match(matcher) == {c: z, l: 1, N: 2, A: 7, B: 0}
    assert (-eq).match(matcher) == {c: -z, l: 1, N: 2, A: 7, B: 0}
    assert (x*eq).match(matcher) == {c: x*z, l: 1, N: 2, A: 7, B: 0}
    assert (-7*x*eq).match(matcher) == {c: -7*x*z, l: 1, N: 2, A: 7, B: 0}

    matcher = c*sin(phi*N)**l * r**A

    assert eq.match(matcher) == {c: z, l: 1, N: 2, A: 7}
    assert (-eq).match(matcher) == {c: -z, l: 1, N: 2, A: 7}
    assert (x*eq).match(matcher) == {c: x*z, l: 1, N: 2, A: 7}
    assert (-7*x*eq).match(matcher) == {c: -7*x*z, l: 1, N: 2, A: 7}


def test_issue_3883():
    from sympy.abc import gamma, mu, x
    f = (-gamma * (x - mu)**2 - log(gamma) + log(2*pi))/2
    a, b, c = symbols('a b c', cls=Wild, exclude=(gamma,))

    assert f.match(a * log(gamma) + b * gamma + c) == \
        {a: Rational(-1, 2), b: -(-mu + x)**2/2, c: log(2*pi)/2}
    assert f.expand().collect(gamma).match(a * log(gamma) + b * gamma + c) == \
        {a: Rational(-1, 2), b: (-(x - mu)**2/2).expand(), c: (log(2*pi)/2).expand()}
    g1 = Wild('g1', exclude=[gamma])
    g2 = Wild('g2', exclude=[gamma])
    g3 = Wild('g3', exclude=[gamma])
    assert f.expand().match(g1 * log(gamma) + g2 * gamma + g3) == \
    {g3: log(2)/2 + log(pi)/2, g1: Rational(-1, 2), g2: -mu**2/2 + mu*x - x**2/2}


def test_issue_4418():
    x = Symbol('x')
    a, b, c = symbols('a b c', cls=Wild, exclude=(x,))
    f, g = symbols('f g', cls=Function)

    eq = diff(g(x)*f(x).diff(x), x)

    assert eq.match(
        g(x).diff(x)*f(x).diff(x) + g(x)*f(x).diff(x, x) + c) == {c: 0}
    assert eq.match(a*g(x).diff(
        x)*f(x).diff(x) + b*g(x)*f(x).diff(x, x) + c) == {a: 1, b: 1, c: 0}


def test_issue_4700():
    f = Function('f')
    x = Symbol('x')
    a, b = symbols('a b', cls=Wild, exclude=(f(x),))

    p = a*f(x) + b
    eq1 = sin(x)
    eq2 = f(x) + sin(x)
    eq3 = f(x) + x + sin(x)
    eq4 = x + sin(x)

    assert eq1.match(p) == {a: 0, b: sin(x)}
    assert eq2.match(p) == {a: 1, b: sin(x)}
    assert eq3.match(p) == {a: 1, b: x + sin(x)}
    assert eq4.match(p) == {a: 0, b: x + sin(x)}


def test_issue_5168():
    a, b, c = symbols('a b c', cls=Wild)
    x = Symbol('x')
    f = Function('f')

    assert x.match(a) == {a: x}
    assert x.match(a*f(x)**c) == {a: x, c: 0}
    assert x.match(a*b) == {a: 1, b: x}
    assert x.match(a*b*f(x)**c) == {a: 1, b: x, c: 0}

    assert (-x).match(a) == {a: -x}
    assert (-x).match(a*f(x)**c) == {a: -x, c: 0}
    assert (-x).match(a*b) == {a: -1, b: x}
    assert (-x).match(a*b*f(x)**c) == {a: -1, b: x, c: 0}

    assert (2*x).match(a) == {a: 2*x}
    assert (2*x).match(a*f(x)**c) == {a: 2*x, c: 0}
    assert (2*x).match(a*b) == {a: 2, b: x}
    assert (2*x).match(a*b*f(x)**c) == {a: 2, b: x, c: 0}

    assert (-2*x).match(a) == {a: -2*x}
    assert (-2*x).match(a*f(x)**c) == {a: -2*x, c: 0}
    assert (-2*x).match(a*b) == {a: -2, b: x}
    assert (-2*x).match(a*b*f(x)**c) == {a: -2, b: x, c: 0}


def test_issue_4559():
    x = Symbol('x')
    e = Symbol('e')
    w = Wild('w', exclude=[x])
    y = Wild('y')

    # this is as it should be

    assert (3/x).match(w/y) == {w: 3, y: x}
    assert (3*x).match(w*y) == {w: 3, y: x}
    assert (x/3).match(y/w) == {w: 3, y: x}
    assert (3*x).match(y/w) == {w: S.One/3, y: x}
    assert (3*x).match(y/w) == {w: Rational(1, 3), y: x}

    # these could be allowed to fail

    assert (x/3).match(w/y) == {w: S.One/3, y: 1/x}
    assert (3*x).match(w/y) == {w: 3, y: 1/x}
    assert (3/x).match(w*y) == {w: 3, y: 1/x}

    # Note that solve will give
    # multiple roots but match only gives one:
    #
    # >>> solve(x**r-y**2,y)
    # [-x**(r/2), x**(r/2)]

    r = Symbol('r', rational=True)
    assert (x**r).match(y**2) == {y: x**(r/2)}
    assert (x**e).match(y**2) == {y: sqrt(x**e)}

    # since (x**i = y) -> x = y**(1/i) where i is an integer
    # the following should also be valid as long as y is not
    # zero when i is negative.

    a = Wild('a')

    e = S.Zero
    assert e.match(a) == {a: e}
    assert e.match(1/a) is None
    assert e.match(a**.3) is None

    e = S(3)
    assert e.match(1/a) == {a: 1/e}
    assert e.match(1/a**2) == {a: 1/sqrt(e)}
    e = pi
    assert e.match(1/a) == {a: 1/e}
    assert e.match(1/a**2) == {a: 1/sqrt(e)}
    assert (-e).match(sqrt(a)) is None
    assert (-e).match(a**2) == {a: I*sqrt(pi)}

# The pattern matcher doesn't know how to handle (x - a)**2 == (a - x)**2. To
# avoid ambiguity in actual applications, don't put a coefficient (including a
# minus sign) in front of a wild.
@XFAIL
def test_issue_4883():
    a = Wild('a')
    x = Symbol('x')

    e = [i**2 for i in (x - 2, 2 - x)]
    p = [i**2 for i in (x - a, a- x)]
    for eq in e:
        for pat in p:
            assert eq.match(pat) == {a: 2}


def test_issue_4319():
    x, y = symbols('x y')

    p = -x*(S.One/8 - y)
    ans = {S.Zero, y - S.One/8}

    def ok(pat):
        assert set(p.match(pat).values()) == ans

    ok(Wild("coeff", exclude=[x])*x + Wild("rest"))
    ok(Wild("w", exclude=[x])*x + Wild("rest"))
    ok(Wild("coeff", exclude=[x])*x + Wild("rest"))
    ok(Wild("w", exclude=[x])*x + Wild("rest"))
    ok(Wild("e", exclude=[x])*x + Wild("rest"))
    ok(Wild("ress", exclude=[x])*x + Wild("rest"))
    ok(Wild("resu", exclude=[x])*x + Wild("rest"))


def test_issue_3778():
    p, c, q = symbols('p c q', cls=Wild)
    x = Symbol('x')

    assert (sin(x)**2).match(sin(p)*sin(q)*c) == {q: x, c: 1, p: x}
    assert (2*sin(x)).match(sin(p) + sin(q) + c) == {q: x, c: 0, p: x}


def test_issue_6103():
    x = Symbol('x')
    a = Wild('a')
    assert (-I*x*oo).match(I*a*oo) == {a: -x}


def test_issue_3539():
    a = Wild('a')
    x = Symbol('x')
    assert (x - 2).match(a - x) is None
    assert (6/x).match(a*x) is None
    assert (6/x**2).match(a/x) == {a: 6/x}

def test_gh_issue_2711():
    x = Symbol('x')
    f = meijerg(((), ()), ((0,), ()), x)
    a = Wild('a')
    b = Wild('b')

    assert f.find(a) == {(S.Zero,), ((), ()), ((S.Zero,), ()), x, S.Zero,
                             (), meijerg(((), ()), ((S.Zero,), ()), x)}
    assert f.find(a + b) == \
        {meijerg(((), ()), ((S.Zero,), ()), x), x, S.Zero}
    assert f.find(a**2) == {meijerg(((), ()), ((S.Zero,), ()), x), x}


def test_issue_17354():
    from sympy.core.symbol import (Wild, symbols)
    x, y = symbols("x y", real=True)
    a, b = symbols("a b", cls=Wild)
    assert ((0 <= x).reversed | (y <= x)).match((1/a <= b) | (a <= b)) is None


def test_match_issue_17397():
    f = Function("f")
    x = Symbol("x")
    a3 = Wild('a3', exclude=[f(x), f(x).diff(x), f(x).diff(x, 2)])
    b3 = Wild('b3', exclude=[f(x), f(x).diff(x), f(x).diff(x, 2)])
    c3 = Wild('c3', exclude=[f(x), f(x).diff(x), f(x).diff(x, 2)])
    deq = a3*(f(x).diff(x, 2)) + b3*f(x).diff(x) + c3*f(x)

    eq = (x-2)**2*(f(x).diff(x, 2)) + (x-2)*(f(x).diff(x)) + ((x-2)**2 - 4)*f(x)
    r = collect(eq, [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
    assert r == {a3: (x - 2)**2, c3: (x - 2)**2 - 4, b3: x - 2}

    eq =x*f(x) + x*Derivative(f(x), (x, 2)) - 4*f(x) + Derivative(f(x), x) \
        - 4*Derivative(f(x), (x, 2)) - 2*Derivative(f(x), x)/x + 4*Derivative(f(x), (x, 2))/x
    r = collect(eq, [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
    assert r == {a3: x - 4 + 4/x, b3: 1 - 2/x, c3: x - 4}


def test_match_issue_21942():
    a, r, w = symbols('a, r, w', nonnegative=True)
    p = symbols('p', positive=True)
    g_ = Wild('g')
    pattern = g_ ** (1 / (1 - p))
    eq = (a * r ** (1 - p) + w ** (1 - p) * (1 - a)) ** (1 / (1 - p))
    m = {g_: a * r ** (1 - p) + w ** (1 - p) * (1 - a)}
    assert pattern.matches(eq) == m
    assert (-pattern).matches(-eq) == m
    assert pattern.matches(signsimp(eq)) is None


def test_match_terms():
    X, Y = map(Wild, "XY")
    x, y, z = symbols('x y z')
    assert (5*y - x).match(5*X - Y) == {X: y, Y: x}
    # 15907
    assert (x + (y - 1)*z).match(x + X*z) == {X: y - 1}
    # 20747
    assert (x - log(x/y)*(1-exp(x/y))).match(x - log(X/y)*(1-exp(x/y))) == {X: x}


def test_match_bound():
    V, W = map(Wild, "VW")
    x, y = symbols('x y')
    assert Sum(x, (x, 1, 2)).match(Sum(y, (y, 1, W))) is None
    assert Sum(x, (x, 1, 2)).match(Sum(V, (V, 1, W))) == {W: 2, V:x}
    assert Sum(x, (x, 1, 2)).match(Sum(V, (V, 1, 2))) == {V:x}


def test_issue_22462():
    x, f = symbols('x'), Function('f')
    n, Q = symbols('n Q', cls=Wild)
    pattern = -Q*f(x)**n
    eq = 5*f(x)**2
    assert pattern.matches(eq) == {n: 2, Q: -5}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\methods\test_shift.py ===
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Index,
    NaT,
    Series,
    date_range,
    offsets,
)
import pandas._testing as tm


class TestDataFrameShift:
    def test_shift_axis1_with_valid_fill_value_one_array(self):
        # Case with axis=1 that does not go through the "len(arrays)>1" path
        #  in DataFrame.shift
        data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)
        res = df.shift(axis=1, periods=1, fill_value=12345)
        expected = df.T.shift(periods=1, fill_value=12345).T
        tm.assert_frame_equal(res, expected)

        # same but with an 1D ExtensionArray backing it
        df2 = df[[0]].astype("Float64")
        res2 = df2.shift(axis=1, periods=1, fill_value=12345)
        expected2 = DataFrame([12345] * 5, dtype="Float64")
        tm.assert_frame_equal(res2, expected2)

    def test_shift_deprecate_freq_and_fill_value(self, frame_or_series):
        # Can't pass both!
        obj = frame_or_series(
            np.random.default_rng(2).standard_normal(5),
            index=date_range("1/1/2000", periods=5, freq="h"),
        )

        msg = (
            "Passing a 'freq' together with a 'fill_value' silently ignores the "
            "fill_value"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            obj.shift(1, fill_value=1, freq="h")

        if frame_or_series is DataFrame:
            obj.columns = date_range("1/1/2000", periods=1, freq="h")
            with tm.assert_produces_warning(FutureWarning, match=msg):
                obj.shift(1, axis=1, fill_value=1, freq="h")

    @pytest.mark.parametrize(
        "input_data, output_data",
        [(np.empty(shape=(0,)), []), (np.ones(shape=(2,)), [np.nan, 1.0])],
    )
    def test_shift_non_writable_array(self, input_data, output_data, frame_or_series):
        # GH21049 Verify whether non writable numpy array is shiftable
        input_data.setflags(write=False)

        result = frame_or_series(input_data).shift(1)
        if frame_or_series is not Series:
            # need to explicitly specify columns in the empty case
            expected = frame_or_series(
                output_data,
                index=range(len(output_data)),
                columns=range(1),
                dtype="float64",
            )
        else:
            expected = frame_or_series(output_data, dtype="float64")

        tm.assert_equal(result, expected)

    def test_shift_mismatched_freq(self, frame_or_series):
        ts = frame_or_series(
            np.random.default_rng(2).standard_normal(5),
            index=date_range("1/1/2000", periods=5, freq="h"),
        )

        result = ts.shift(1, freq="5min")
        exp_index = ts.index.shift(1, freq="5min")
        tm.assert_index_equal(result.index, exp_index)

        # GH#1063, multiple of same base
        result = ts.shift(1, freq="4h")
        exp_index = ts.index + offsets.Hour(4)
        tm.assert_index_equal(result.index, exp_index)

    @pytest.mark.parametrize(
        "obj",
        [
            Series([np.arange(5)]),
            date_range("1/1/2011", periods=24, freq="h"),
            Series(range(5), index=date_range("2017", periods=5)),
        ],
    )
    @pytest.mark.parametrize("shift_size", [0, 1, 2])
    def test_shift_always_copy(self, obj, shift_size, frame_or_series):
        # GH#22397
        if frame_or_series is not Series:
            obj = obj.to_frame()
        assert obj.shift(shift_size) is not obj

    def test_shift_object_non_scalar_fill(self):
        # shift requires scalar fill_value except for object dtype
        ser = Series(range(3))
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            ser.shift(1, fill_value=[])

        df = ser.to_frame()
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            df.shift(1, fill_value=np.arange(3))

        obj_ser = ser.astype(object)
        result = obj_ser.shift(1, fill_value={})
        assert result[0] == {}

        obj_df = obj_ser.to_frame()
        result = obj_df.shift(1, fill_value={})
        assert result.iloc[0, 0] == {}

    def test_shift_int(self, datetime_frame, frame_or_series):
        ts = tm.get_obj(datetime_frame, frame_or_series).astype(int)
        shifted = ts.shift(1)
        expected = ts.astype(float).shift(1)
        tm.assert_equal(shifted, expected)

    @pytest.mark.parametrize("dtype", ["int32", "int64"])
    def test_shift_32bit_take(self, frame_or_series, dtype):
        # 32-bit taking
        # GH#8129
        index = date_range("2000-01-01", periods=5)
        arr = np.arange(5, dtype=dtype)
        s1 = frame_or_series(arr, index=index)
        p = arr[1]
        result = s1.shift(periods=p)
        expected = frame_or_series([np.nan, 0, 1, 2, 3], index=index)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("periods", [1, 2, 3, 4])
    def test_shift_preserve_freqstr(self, periods, frame_or_series):
        # GH#21275
        obj = frame_or_series(
            range(periods),
            index=date_range("2016-1-1 00:00:00", periods=periods, freq="h"),
        )

        result = obj.shift(1, "2h")

        expected = frame_or_series(
            range(periods),
            index=date_range("2016-1-1 02:00:00", periods=periods, freq="h"),
        )
        tm.assert_equal(result, expected)

    def test_shift_dst(self, frame_or_series):
        # GH#13926
        dates = date_range("2016-11-06", freq="h", periods=10, tz="US/Eastern")
        obj = frame_or_series(dates)

        res = obj.shift(0)
        tm.assert_equal(res, obj)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

        res = obj.shift(1)
        exp_vals = [NaT] + dates.astype(object).values.tolist()[:9]
        exp = frame_or_series(exp_vals)
        tm.assert_equal(res, exp)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

        res = obj.shift(-2)
        exp_vals = dates.astype(object).values.tolist()[2:] + [NaT, NaT]
        exp = frame_or_series(exp_vals)
        tm.assert_equal(res, exp)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

    @pytest.mark.parametrize("ex", [10, -10, 20, -20])
    def test_shift_dst_beyond(self, frame_or_series, ex):
        # GH#13926
        dates = date_range("2016-11-06", freq="h", periods=10, tz="US/Eastern")
        obj = frame_or_series(dates)
        res = obj.shift(ex)
        exp = frame_or_series([NaT] * 10, dtype="datetime64[ns, US/Eastern]")
        tm.assert_equal(res, exp)
        assert tm.get_dtype(res) == "datetime64[ns, US/Eastern]"

    def test_shift_by_zero(self, datetime_frame, frame_or_series):
        # shift by 0
        obj = tm.get_obj(datetime_frame, frame_or_series)
        unshifted = obj.shift(0)
        tm.assert_equal(unshifted, obj)

    def test_shift(self, datetime_frame):
        # naive shift
        ser = datetime_frame["A"]

        shifted = datetime_frame.shift(5)
        tm.assert_index_equal(shifted.index, datetime_frame.index)

        shifted_ser = ser.shift(5)
        tm.assert_series_equal(shifted["A"], shifted_ser)

        shifted = datetime_frame.shift(-5)
        tm.assert_index_equal(shifted.index, datetime_frame.index)

        shifted_ser = ser.shift(-5)
        tm.assert_series_equal(shifted["A"], shifted_ser)

        unshifted = datetime_frame.shift(5).shift(-5)
        tm.assert_numpy_array_equal(
            unshifted.dropna().values, datetime_frame.values[:-5]
        )

        unshifted_ser = ser.shift(5).shift(-5)
        tm.assert_numpy_array_equal(unshifted_ser.dropna().values, ser.values[:-5])

    def test_shift_by_offset(self, datetime_frame, frame_or_series):
        # shift by DateOffset
        obj = tm.get_obj(datetime_frame, frame_or_series)
        offset = offsets.BDay()

        shifted = obj.shift(5, freq=offset)
        assert len(shifted) == len(obj)
        unshifted = shifted.shift(-5, freq=offset)
        tm.assert_equal(unshifted, obj)

        shifted2 = obj.shift(5, freq="B")
        tm.assert_equal(shifted, shifted2)

        unshifted = obj.shift(0, freq=offset)
        tm.assert_equal(unshifted, obj)

        d = obj.index[0]
        shifted_d = d + offset * 5
        if frame_or_series is DataFrame:
            tm.assert_series_equal(obj.xs(d), shifted.xs(shifted_d), check_names=False)
        else:
            tm.assert_almost_equal(obj.at[d], shifted.at[shifted_d])

    def test_shift_with_periodindex(self, frame_or_series):
        # Shifting with PeriodIndex
        ps = DataFrame(
            np.arange(4, dtype=float), index=pd.period_range("2020-01-01", periods=4)
        )
        ps = tm.get_obj(ps, frame_or_series)

        shifted = ps.shift(1)
        unshifted = shifted.shift(-1)
        tm.assert_index_equal(shifted.index, ps.index)
        tm.assert_index_equal(unshifted.index, ps.index)
        if frame_or_series is DataFrame:
            tm.assert_numpy_array_equal(
                unshifted.iloc[:, 0].dropna().values, ps.iloc[:-1, 0].values
            )
        else:
            tm.assert_numpy_array_equal(unshifted.dropna().values, ps.values[:-1])

        shifted2 = ps.shift(1, "D")
        shifted3 = ps.shift(1, offsets.Day())
        tm.assert_equal(shifted2, shifted3)
        tm.assert_equal(ps, shifted2.shift(-1, "D"))

        msg = "does not match PeriodIndex freq"
        with pytest.raises(ValueError, match=msg):
            ps.shift(freq="W")

        # legacy support
        shifted4 = ps.shift(1, freq="D")
        tm.assert_equal(shifted2, shifted4)

        shifted5 = ps.shift(1, freq=offsets.Day())
        tm.assert_equal(shifted5, shifted4)

    def test_shift_other_axis(self):
        # shift other axis
        # GH#6371
        df = DataFrame(np.random.default_rng(2).random((10, 5)))
        expected = pd.concat(
            [DataFrame(np.nan, index=df.index, columns=[0]), df.iloc[:, 0:-1]],
            ignore_index=True,
            axis=1,
        )
        result = df.shift(1, axis=1)
        tm.assert_frame_equal(result, expected)

    def test_shift_named_axis(self):
        # shift named axis
        df = DataFrame(np.random.default_rng(2).random((10, 5)))
        expected = pd.concat(
            [DataFrame(np.nan, index=df.index, columns=[0]), df.iloc[:, 0:-1]],
            ignore_index=True,
            axis=1,
        )
        result = df.shift(1, axis="columns")
        tm.assert_frame_equal(result, expected)

    def test_shift_other_axis_with_freq(self, datetime_frame):
        obj = datetime_frame.T
        offset = offsets.BDay()

        # GH#47039
        shifted = obj.shift(5, freq=offset, axis=1)
        assert len(shifted) == len(obj)
        unshifted = shifted.shift(-5, freq=offset, axis=1)
        tm.assert_equal(unshifted, obj)

    def test_shift_bool(self):
        df = DataFrame({"high": [True, False], "low": [False, False]})
        rs = df.shift(1)
        xp = DataFrame(
            np.array([[np.nan, np.nan], [True, False]], dtype=object),
            columns=["high", "low"],
        )
        tm.assert_frame_equal(rs, xp)

    def test_shift_categorical1(self, frame_or_series):
        # GH#9416
        obj = frame_or_series(["a", "b", "c", "d"], dtype="category")

        rt = obj.shift(1).shift(-1)
        tm.assert_equal(obj.iloc[:-1], rt.dropna())

        def get_cat_values(ndframe):
            # For Series we could just do ._values; for DataFrame
            #  we may be able to do this if we ever have 2D Categoricals
            return ndframe._mgr.arrays[0]

        cat = get_cat_values(obj)

        sp1 = obj.shift(1)
        tm.assert_index_equal(obj.index, sp1.index)
        assert np.all(get_cat_values(sp1).codes[:1] == -1)
        assert np.all(cat.codes[:-1] == get_cat_values(sp1).codes[1:])

        sn2 = obj.shift(-2)
        tm.assert_index_equal(obj.index, sn2.index)
        assert np.all(get_cat_values(sn2).codes[-2:] == -1)
        assert np.all(cat.codes[2:] == get_cat_values(sn2).codes[:-2])

        tm.assert_index_equal(cat.categories, get_cat_values(sp1).categories)
        tm.assert_index_equal(cat.categories, get_cat_values(sn2).categories)

    def test_shift_categorical(self):
        # GH#9416
        s1 = Series(["a", "b", "c"], dtype="category")
        s2 = Series(["A", "B", "C"], dtype="category")
        df = DataFrame({"one": s1, "two": s2})
        rs = df.shift(1)
        xp = DataFrame({"one": s1.shift(1), "two": s2.shift(1)})
        tm.assert_frame_equal(rs, xp)

    def test_shift_categorical_fill_value(self, frame_or_series):
        ts = frame_or_series(["a", "b", "c", "d"], dtype="category")
        res = ts.shift(1, fill_value="a")
        expected = frame_or_series(
            pd.Categorical(
                ["a", "a", "b", "c"], categories=["a", "b", "c", "d"], ordered=False
            )
        )
        tm.assert_equal(res, expected)

        # check for incorrect fill_value
        msg = r"Cannot setitem on a Categorical with a new category \(f\)"
        with pytest.raises(TypeError, match=msg):
            ts.shift(1, fill_value="f")

    def test_shift_fill_value(self, frame_or_series):
        # GH#24128
        dti = date_range("1/1/2000", periods=5, freq="h")

        ts = frame_or_series([1.0, 2.0, 3.0, 4.0, 5.0], index=dti)
        exp = frame_or_series([0.0, 1.0, 2.0, 3.0, 4.0], index=dti)
        # check that fill value works
        result = ts.shift(1, fill_value=0.0)
        tm.assert_equal(result, exp)

        exp = frame_or_series([0.0, 0.0, 1.0, 2.0, 3.0], index=dti)
        result = ts.shift(2, fill_value=0.0)
        tm.assert_equal(result, exp)

        ts = frame_or_series([1, 2, 3])
        res = ts.shift(2, fill_value=0)
        assert tm.get_dtype(res) == tm.get_dtype(ts)

        # retain integer dtype
        obj = frame_or_series([1, 2, 3, 4, 5], index=dti)
        exp = frame_or_series([0, 1, 2, 3, 4], index=dti)
        result = obj.shift(1, fill_value=0)
        tm.assert_equal(result, exp)

        exp = frame_or_series([0, 0, 1, 2, 3], index=dti)
        result = obj.shift(2, fill_value=0)
        tm.assert_equal(result, exp)

    def test_shift_empty(self):
        # Regression test for GH#8019
        df = DataFrame({"foo": []})
        rs = df.shift(-1)

        tm.assert_frame_equal(df, rs)

    def test_shift_duplicate_columns(self):
        # GH#9092; verify that position-based shifting works
        # in the presence of duplicate columns
        column_lists = [list(range(5)), [1] * 5, [1, 1, 2, 2, 1]]
        data = np.random.default_rng(2).standard_normal((20, 5))

        shifted = []
        for columns in column_lists:
            df = DataFrame(data.copy(), columns=columns)
            for s in range(5):
                df.iloc[:, s] = df.iloc[:, s].shift(s + 1)
            df.columns = range(5)
            shifted.append(df)

        # sanity check the base case
        nulls = shifted[0].isna().sum()
        tm.assert_series_equal(nulls, Series(range(1, 6), dtype="int64"))

        # check all answers are the same
        tm.assert_frame_equal(shifted[0], shifted[1])
        tm.assert_frame_equal(shifted[0], shifted[2])

    def test_shift_axis1_multiple_blocks(self, using_array_manager):
        # GH#35488
        df1 = DataFrame(np.random.default_rng(2).integers(1000, size=(5, 3)))
        df2 = DataFrame(np.random.default_rng(2).integers(1000, size=(5, 2)))
        df3 = pd.concat([df1, df2], axis=1)
        if not using_array_manager:
            assert len(df3._mgr.blocks) == 2

        result = df3.shift(2, axis=1)

        expected = df3.take([-1, -1, 0, 1, 2], axis=1)
        # Explicit cast to float to avoid implicit cast when setting nan.
        # Column names aren't unique, so directly calling `expected.astype` won't work.
        expected = expected.pipe(
            lambda df: df.set_axis(range(df.shape[1]), axis=1)
            .astype({0: "float", 1: "float"})
            .set_axis(df.columns, axis=1)
        )
        expected.iloc[:, :2] = np.nan
        expected.columns = df3.columns

        tm.assert_frame_equal(result, expected)

        # Case with periods < 0
        # rebuild df3 because `take` call above consolidated
        df3 = pd.concat([df1, df2], axis=1)
        if not using_array_manager:
            assert len(df3._mgr.blocks) == 2
        result = df3.shift(-2, axis=1)

        expected = df3.take([2, 3, 4, -1, -1], axis=1)
        # Explicit cast to float to avoid implicit cast when setting nan.
        # Column names aren't unique, so directly calling `expected.astype` won't work.
        expected = expected.pipe(
            lambda df: df.set_axis(range(df.shape[1]), axis=1)
            .astype({3: "float", 4: "float"})
            .set_axis(df.columns, axis=1)
        )
        expected.iloc[:, -2:] = np.nan
        expected.columns = df3.columns

        tm.assert_frame_equal(result, expected)

    @td.skip_array_manager_not_yet_implemented  # TODO(ArrayManager) axis=1 support
    def test_shift_axis1_multiple_blocks_with_int_fill(self):
        # GH#42719
        rng = np.random.default_rng(2)
        df1 = DataFrame(rng.integers(1000, size=(5, 3), dtype=int))
        df2 = DataFrame(rng.integers(1000, size=(5, 2), dtype=int))
        df3 = pd.concat([df1.iloc[:4, 1:3], df2.iloc[:4, :]], axis=1)
        result = df3.shift(2, axis=1, fill_value=np.int_(0))
        assert len(df3._mgr.blocks) == 2

        expected = df3.take([-1, -1, 0, 1], axis=1)
        expected.iloc[:, :2] = np.int_(0)
        expected.columns = df3.columns

        tm.assert_frame_equal(result, expected)

        # Case with periods < 0
        df3 = pd.concat([df1.iloc[:4, 1:3], df2.iloc[:4, :]], axis=1)
        result = df3.shift(-2, axis=1, fill_value=np.int_(0))
        assert len(df3._mgr.blocks) == 2

        expected = df3.take([2, 3, -1, -1], axis=1)
        expected.iloc[:, -2:] = np.int_(0)
        expected.columns = df3.columns

        tm.assert_frame_equal(result, expected)

    def test_period_index_frame_shift_with_freq(self, frame_or_series):
        ps = DataFrame(range(4), index=pd.period_range("2020-01-01", periods=4))
        ps = tm.get_obj(ps, frame_or_series)

        shifted = ps.shift(1, freq="infer")
        unshifted = shifted.shift(-1, freq="infer")
        tm.assert_equal(unshifted, ps)

        shifted2 = ps.shift(freq="D")
        tm.assert_equal(shifted, shifted2)

        shifted3 = ps.shift(freq=offsets.Day())
        tm.assert_equal(shifted, shifted3)

    def test_datetime_frame_shift_with_freq(self, datetime_frame, frame_or_series):
        dtobj = tm.get_obj(datetime_frame, frame_or_series)
        shifted = dtobj.shift(1, freq="infer")
        unshifted = shifted.shift(-1, freq="infer")
        tm.assert_equal(dtobj, unshifted)

        shifted2 = dtobj.shift(freq=dtobj.index.freq)
        tm.assert_equal(shifted, shifted2)

        inferred_ts = DataFrame(
            datetime_frame.values,
            Index(np.asarray(datetime_frame.index)),
            columns=datetime_frame.columns,
        )
        inferred_ts = tm.get_obj(inferred_ts, frame_or_series)
        shifted = inferred_ts.shift(1, freq="infer")
        expected = dtobj.shift(1, freq="infer")
        expected.index = expected.index._with_freq(None)
        tm.assert_equal(shifted, expected)

        unshifted = shifted.shift(-1, freq="infer")
        tm.assert_equal(unshifted, inferred_ts)

    def test_period_index_frame_shift_with_freq_error(self, frame_or_series):
        ps = DataFrame(range(4), index=pd.period_range("2020-01-01", periods=4))
        ps = tm.get_obj(ps, frame_or_series)
        msg = "Given freq M does not match PeriodIndex freq D"
        with pytest.raises(ValueError, match=msg):
            ps.shift(freq="M")

    def test_datetime_frame_shift_with_freq_error(
        self, datetime_frame, frame_or_series
    ):
        dtobj = tm.get_obj(datetime_frame, frame_or_series)
        no_freq = dtobj.iloc[[0, 5, 7]]
        msg = "Freq was not set in the index hence cannot be inferred"
        with pytest.raises(ValueError, match=msg):
            no_freq.shift(freq="infer")

    def test_shift_dt64values_int_fill_deprecated(self):
        # GH#31971
        ser = Series([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")])

        with pytest.raises(TypeError, match="value should be a"):
            ser.shift(1, fill_value=0)

        df = ser.to_frame()
        with pytest.raises(TypeError, match="value should be a"):
            df.shift(1, fill_value=0)

        # axis = 1
        df2 = DataFrame({"A": ser, "B": ser})
        df2._consolidate_inplace()

        result = df2.shift(1, axis=1, fill_value=0)
        expected = DataFrame({"A": [0, 0], "B": df2["A"]})
        tm.assert_frame_equal(result, expected)

        # same thing but not consolidated; pre-2.0 we got different behavior
        df3 = DataFrame({"A": ser})
        df3["B"] = ser
        assert len(df3._mgr.arrays) == 2
        result = df3.shift(1, axis=1, fill_value=0)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "as_cat",
        [
            pytest.param(
                True,
                marks=pytest.mark.xfail(
                    reason="_can_hold_element incorrectly always returns True"
                ),
            ),
            False,
        ],
    )
    @pytest.mark.parametrize(
        "vals",
        [
            date_range("2020-01-01", periods=2),
            date_range("2020-01-01", periods=2, tz="US/Pacific"),
            pd.period_range("2020-01-01", periods=2, freq="D"),
            pd.timedelta_range("2020 Days", periods=2, freq="D"),
            pd.interval_range(0, 3, periods=2),
            pytest.param(
                pd.array([1, 2], dtype="Int64"),
                marks=pytest.mark.xfail(
                    reason="_can_hold_element incorrectly always returns True"
                ),
            ),
            pytest.param(
                pd.array([1, 2], dtype="Float32"),
                marks=pytest.mark.xfail(
                    reason="_can_hold_element incorrectly always returns True"
                ),
            ),
        ],
        ids=lambda x: str(x.dtype),
    )
    def test_shift_dt64values_axis1_invalid_fill(self, vals, as_cat):
        # GH#44564
        ser = Series(vals)
        if as_cat:
            ser = ser.astype("category")

        df = DataFrame({"A": ser})
        result = df.shift(-1, axis=1, fill_value="foo")
        expected = DataFrame({"A": ["foo", "foo"]})
        tm.assert_frame_equal(result, expected)

        # same thing but multiple blocks
        df2 = DataFrame({"A": ser, "B": ser})
        df2._consolidate_inplace()

        result = df2.shift(-1, axis=1, fill_value="foo")
        expected = DataFrame({"A": df2["B"], "B": ["foo", "foo"]})
        tm.assert_frame_equal(result, expected)

        # same thing but not consolidated
        df3 = DataFrame({"A": ser})
        df3["B"] = ser
        assert len(df3._mgr.arrays) == 2
        result = df3.shift(-1, axis=1, fill_value="foo")
        tm.assert_frame_equal(result, expected)

    def test_shift_axis1_categorical_columns(self):
        # GH#38434
        ci = CategoricalIndex(["a", "b", "c"])
        df = DataFrame(
            {"a": [1, 3], "b": [2, 4], "c": [5, 6]}, index=ci[:-1], columns=ci
        )
        result = df.shift(axis=1)

        expected = DataFrame(
            {"a": [np.nan, np.nan], "b": [1, 3], "c": [2, 4]}, index=ci[:-1], columns=ci
        )
        tm.assert_frame_equal(result, expected)

        # periods != 1
        result = df.shift(2, axis=1)
        expected = DataFrame(
            {"a": [np.nan, np.nan], "b": [np.nan, np.nan], "c": [1, 3]},
            index=ci[:-1],
            columns=ci,
        )
        tm.assert_frame_equal(result, expected)

    def test_shift_axis1_many_periods(self):
        # GH#44978 periods > len(columns)
        df = DataFrame(np.random.default_rng(2).random((5, 3)))
        shifted = df.shift(6, axis=1, fill_value=None)

        expected = df * np.nan
        tm.assert_frame_equal(shifted, expected)

        shifted2 = df.shift(-6, axis=1, fill_value=None)
        tm.assert_frame_equal(shifted2, expected)

    def test_shift_with_offsets_freq(self):
        df = DataFrame({"x": [1, 2, 3]}, index=date_range("2000", periods=3))
        shifted = df.shift(freq="1MS")
        expected = DataFrame(
            {"x": [1, 2, 3]},
            index=date_range(start="02/01/2000", end="02/01/2000", periods=3),
        )
        tm.assert_frame_equal(shifted, expected)

    def test_shift_with_iterable_basic_functionality(self):
        # GH#44424
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        shifts = [0, 1, 2]

        df = DataFrame(data)
        shifted = df.shift(shifts)

        expected = DataFrame(
            {
                "a_0": [1, 2, 3],
                "b_0": [4, 5, 6],
                "a_1": [np.nan, 1.0, 2.0],
                "b_1": [np.nan, 4.0, 5.0],
                "a_2": [np.nan, np.nan, 1.0],
                "b_2": [np.nan, np.nan, 4.0],
            }
        )
        tm.assert_frame_equal(expected, shifted)

    def test_shift_with_iterable_series(self):
        # GH#44424
        data = {"a": [1, 2, 3]}
        shifts = [0, 1, 2]

        df = DataFrame(data)
        s = df["a"]
        tm.assert_frame_equal(s.shift(shifts), df.shift(shifts))

    def test_shift_with_iterable_freq_and_fill_value(self):
        # GH#44424
        df = DataFrame(
            np.random.default_rng(2).standard_normal(5),
            index=date_range("1/1/2000", periods=5, freq="h"),
        )

        tm.assert_frame_equal(
            # rename because shift with an iterable leads to str column names
            df.shift([1], fill_value=1).rename(columns=lambda x: int(x[0])),
            df.shift(1, fill_value=1),
        )

        tm.assert_frame_equal(
            df.shift([1], freq="h").rename(columns=lambda x: int(x[0])),
            df.shift(1, freq="h"),
        )

        msg = (
            "Passing a 'freq' together with a 'fill_value' silently ignores the "
            "fill_value"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.shift([1, 2], fill_value=1, freq="h")

    def test_shift_with_iterable_check_other_arguments(self):
        # GH#44424
        data = {"a": [1, 2], "b": [4, 5]}
        shifts = [0, 1]
        df = DataFrame(data)

        # test suffix
        shifted = df[["a"]].shift(shifts, suffix="_suffix")
        expected = DataFrame({"a_suffix_0": [1, 2], "a_suffix_1": [np.nan, 1.0]})
        tm.assert_frame_equal(shifted, expected)

        # check bad inputs when doing multiple shifts
        msg = "If `periods` contains multiple shifts, `axis` cannot be 1."
        with pytest.raises(ValueError, match=msg):
            df.shift(shifts, axis=1)

        msg = "Periods must be integer, but s is <class 'str'>."
        with pytest.raises(TypeError, match=msg):
            df.shift(["s"])

        msg = "If `periods` is an iterable, it cannot be empty."
        with pytest.raises(ValueError, match=msg):
            df.shift([])

        msg = "Cannot specify `suffix` if `periods` is an int."
        with pytest.raises(ValueError, match=msg):
            df.shift(1, suffix="fails")

    def test_shift_axis_one_empty(self):
        # GH#57301
        df = DataFrame()
        result = df.shift(1, axis=1)
        tm.assert_frame_equal(result, df)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\tests\test_risch.py ===
"""Most of these tests come from the examples in Bronstein's book."""
from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
    derivation, splitfactor, splitfactor_sqf, canonical_representation,
    hermite_reduce, polynomial_reduce, residue_reduce, residue_reduce_to_basic,
    integrate_primitive, integrate_hyperexponential_polynomial,
    integrate_hyperexponential, integrate_hypertangent_polynomial,
    integrate_nonlinear_no_specials, integer_powers, DifferentialExtension,
    risch_integrate, DecrementLevel, NonElementaryIntegral, recognize_log_derivative,
    recognize_derivative, laurent_series)
from sympy.testing.pytest import raises

from sympy.abc import x, t, nu, z, a, y
t0, t1, t2 = symbols('t:3')
i = Symbol('i')

def test_gcdex_diophantine():
    assert gcdex_diophantine(Poly(x**4 - 2*x**3 - 6*x**2 + 12*x + 15),
    Poly(x**3 + x**2 - 4*x - 4), Poly(x**2 - 1)) == \
        (Poly((-x**2 + 4*x - 3)/5), Poly((x**3 - 7*x**2 + 16*x - 10)/5))
    assert gcdex_diophantine(Poly(x**3 + 6*x + 7), Poly(x**2 + 3*x + 2), Poly(x + 1)) == \
        (Poly(1/13, x, domain='QQ'), Poly(-1/13*x + 3/13, x, domain='QQ'))


def test_frac_in():
    assert frac_in(Poly((x + 1)/x*t, t), x) == \
        (Poly(t*x + t, x), Poly(x, x))
    assert frac_in((x + 1)/x*t, x) == \
        (Poly(t*x + t, x), Poly(x, x))
    assert frac_in((Poly((x + 1)/x*t, t), Poly(t + 1, t)), x) == \
        (Poly(t*x + t, x), Poly((1 + t)*x, x))
    raises(ValueError, lambda: frac_in((x + 1)/log(x)*t, x))
    assert frac_in(Poly((2 + 2*x + x*(1 + x))/(1 + x)**2, t), x, cancel=True) == \
        (Poly(x + 2, x), Poly(x + 1, x))


def test_as_poly_1t():
    assert as_poly_1t(2/t + t, t, z) in [
        Poly(t + 2*z, t, z), Poly(t + 2*z, z, t)]
    assert as_poly_1t(2/t + 3/t**2, t, z) in [
        Poly(2*z + 3*z**2, t, z), Poly(2*z + 3*z**2, z, t)]
    assert as_poly_1t(2/((exp(2) + 1)*t), t, z) in [
        Poly(2/(exp(2) + 1)*z, t, z), Poly(2/(exp(2) + 1)*z, z, t)]
    assert as_poly_1t(2/((exp(2) + 1)*t) + t, t, z) in [
        Poly(t + 2/(exp(2) + 1)*z, t, z), Poly(t + 2/(exp(2) + 1)*z, z, t)]
    assert as_poly_1t(S.Zero, t, z) == Poly(0, t, z)


def test_derivation():
    p = Poly(4*x**4*t**5 + (-4*x**3 - 4*x**4)*t**4 + (-3*x**2 + 2*x**3)*t**3 +
        (2*x + 7*x**2 + 2*x**3)*t**2 + (1 - 4*x - 4*x**2)*t - 1 + 2*x, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - 3/(2*x)*t + 1/(2*x), t)]})
    assert derivation(p, DE) == Poly(-20*x**4*t**6 + (2*x**3 + 16*x**4)*t**5 +
        (21*x**2 + 12*x**3)*t**4 + (x*Rational(7, 2) - 25*x**2 - 12*x**3)*t**3 +
        (-5 - x*Rational(15, 2) + 7*x**2)*t**2 - (3 - 8*x - 10*x**2 - 4*x**3)/(2*x)*t +
        (1 - 4*x**2)/(2*x), t)
    assert derivation(Poly(1, t), DE) == Poly(0, t)
    assert derivation(Poly(t, t), DE) == DE.d
    assert derivation(Poly(t**2 + 1/x*t + (1 - 2*x)/(4*x**2), t), DE) == \
        Poly(-2*t**3 - 4/x*t**2 - (5 - 2*x)/(2*x**2)*t - (1 - 2*x)/(2*x**3), t, domain='ZZ(x)')
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(t, t)]})
    assert derivation(Poly(x*t*t1, t), DE) == Poly(t*t1 + x*t*t1 + t, t)
    assert derivation(Poly(x*t*t1, t), DE, coefficientD=True) == \
        Poly((1 + t1)*t, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert derivation(Poly(x, x), DE) == Poly(1, x)
    # Test basic option
    assert derivation((x + 1)/(x - 1), DE, basic=True) == -2/(1 - 2*x + x**2)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert derivation((t + 1)/(t - 1), DE, basic=True) == -2*t/(1 - 2*t + t**2)
    assert derivation(t + 1, DE, basic=True) == t


def test_splitfactor():
    p = Poly(4*x**4*t**5 + (-4*x**3 - 4*x**4)*t**4 + (-3*x**2 + 2*x**3)*t**3 +
        (2*x + 7*x**2 + 2*x**3)*t**2 + (1 - 4*x - 4*x**2)*t - 1 + 2*x, t, field=True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - 3/(2*x)*t + 1/(2*x), t)]})
    assert splitfactor(p, DE) == (Poly(4*x**4*t**3 + (-8*x**3 - 4*x**4)*t**2 +
        (4*x**2 + 8*x**3)*t - 4*x**2, t, domain='ZZ(x)'),
        Poly(t**2 + 1/x*t + (1 - 2*x)/(4*x**2), t, domain='ZZ(x)'))
    assert splitfactor(Poly(x, t), DE) == (Poly(x, t), Poly(1, t))
    r = Poly(-4*x**4*z**2 + 4*x**6*z**2 - z*x**3 - 4*x**5*z**3 + 4*x**3*z**3 + x**4 + z*x**5 - x**6, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    assert splitfactor(r, DE, coefficientD=True) == \
        (Poly(x*z - x**2 - z*x**3 + x**4, t), Poly(-x**2 + 4*x**2*z**2, t))
    assert splitfactor_sqf(r, DE, coefficientD=True) == \
        (((Poly(x*z - x**2 - z*x**3 + x**4, t), 1),), ((Poly(-x**2 + 4*x**2*z**2, t), 1),))
    assert splitfactor(Poly(0, t), DE) == (Poly(0, t), Poly(1, t))
    assert splitfactor_sqf(Poly(0, t), DE) == (((Poly(0, t), 1),), ())


def test_canonical_representation():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    assert canonical_representation(Poly(x - t, t), Poly(t**2, t), DE) == \
        (Poly(0, t, domain='ZZ[x]'), (Poly(0, t, domain='QQ[x]'),
        Poly(1, t, domain='ZZ')), (Poly(-t + x, t, domain='QQ[x]'),
        Poly(t**2, t)))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert canonical_representation(Poly(t**5 + t**3 + x**2*t + 1, t),
    Poly((t**2 + 1)**3, t), DE) == \
        (Poly(0, t, domain='ZZ[x]'), (Poly(t**5 + t**3 + x**2*t + 1, t, domain='QQ[x]'),
         Poly(t**6 + 3*t**4 + 3*t**2 + 1, t, domain='QQ')),
        (Poly(0, t, domain='QQ[x]'), Poly(1, t, domain='QQ')))


def test_hermite_reduce():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})

    assert hermite_reduce(Poly(x - t, t), Poly(t**2, t), DE) == \
        ((Poly(-x, t, domain='QQ[x]'), Poly(t, t, domain='QQ[x]')),
         (Poly(0, t, domain='QQ[x]'), Poly(1, t, domain='QQ[x]')),
         (Poly(-x, t, domain='QQ[x]'), Poly(1, t, domain='QQ[x]')))

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - t/x - (1 - nu**2/x**2), t)]})

    assert hermite_reduce(
            Poly(x**2*t**5 + x*t**4 - nu**2*t**3 - x*(x**2 + 1)*t**2 - (x**2 - nu**2)*t - x**5/4, t),
            Poly(x**2*t**4 + x**2*(x**2 + 2)*t**2 + x**2 + x**4 + x**6/4, t), DE) == \
        ((Poly(-x**2 - 4, t, domain='ZZ(x,nu)'), Poly(4*t**2 + 2*x**2 + 4, t, domain='ZZ(x,nu)')),
         (Poly((-2*nu**2 - x**4)*t - (2*x**3 + 2*x), t, domain='ZZ(x,nu)'),
          Poly(2*x**2*t**2 + x**4 + 2*x**2, t, domain='ZZ(x,nu)')),
         (Poly(x*t + 1, t, domain='ZZ(x,nu)'), Poly(x, t, domain='ZZ(x,nu)')))

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})

    a = Poly((-2 + 3*x)*t**3 + (-1 + x)*t**2 + (-4*x + 2*x**2)*t + x**2, t)
    d = Poly(x*t**6 - 4*x**2*t**5 + 6*x**3*t**4 - 4*x**4*t**3 + x**5*t**2, t)

    assert hermite_reduce(a, d, DE) == \
        ((Poly(3*t**2 + t + 3*x, t, domain='ZZ(x)'),
          Poly(3*t**4 - 9*x*t**3 + 9*x**2*t**2 - 3*x**3*t, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')))

    assert hermite_reduce(
            Poly(-t**2 + 2*t + 2, t, domain='ZZ(x)'),
            Poly(-x*t**2 + 2*x*t - x, t, domain='ZZ(x)'), DE) == \
        ((Poly(3, t, domain='ZZ(x)'), Poly(t - 1, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(1, t, domain='ZZ(x)'), Poly(x, t, domain='ZZ(x)')))

    assert hermite_reduce(
            Poly(-x**2*t**6 + (-1 - 2*x**3 + x**4)*t**3 + (-3 - 3*x**4)*t**2 -
                2*x*t - x - 3*x**2, t, domain='ZZ(x)'),
            Poly(x**4*t**6 - 2*x**2*t**3 + 1, t, domain='ZZ(x)'), DE) == \
        ((Poly(x**3*t + x**4 + 1, t, domain='ZZ(x)'), Poly(x**3*t**3 - x, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(-1, t, domain='ZZ(x)'), Poly(x**2, t, domain='ZZ(x)')))

    assert hermite_reduce(
            Poly((-2 + 3*x)*t**3 + (-1 + x)*t**2 + (-4*x + 2*x**2)*t + x**2, t),
            Poly(x*t**6 - 4*x**2*t**5 + 6*x**3*t**4 - 4*x**4*t**3 + x**5*t**2, t), DE) == \
        ((Poly(3*t**2 + t + 3*x, t, domain='ZZ(x)'),
          Poly(3*t**4 - 9*x*t**3 + 9*x**2*t**2 - 3*x**3*t, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')))


def test_polynomial_reduce():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    assert polynomial_reduce(Poly(1 + x*t + t**2, t), DE) == \
        (Poly(t, t), Poly(x*t, t))
    assert polynomial_reduce(Poly(0, t), DE) == \
        (Poly(0, t), Poly(0, t))


def test_laurent_series():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1, t)]})
    a = Poly(36, t)
    d = Poly((t - 2)*(t**2 - 1)**2, t)
    F = Poly(t**2 - 1, t)
    n = 2
    assert laurent_series(a, d, F, n, DE) == \
        (Poly(-3*t**3 + 3*t**2 - 6*t - 8, t), Poly(t**5 + t**4 - 2*t**3 - 2*t**2 + t + 1, t),
        [Poly(-3*t**3 - 6*t**2, t, domain='QQ'), Poly(2*t**6 + 6*t**5 - 8*t**3, t, domain='QQ')])


def test_recognize_derivative():
    DE = DifferentialExtension(extension={'D': [Poly(1, t)]})
    a = Poly(36, t)
    d = Poly((t - 2)*(t**2 - 1)**2, t)
    assert recognize_derivative(a, d, DE) == False
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    a = Poly(2, t)
    d = Poly(t**2 - 1, t)
    assert recognize_derivative(a, d, DE) == False
    assert recognize_derivative(Poly(x*t, t), Poly(1, t), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert recognize_derivative(Poly(t, t), Poly(1, t), DE) == True


def test_recognize_log_derivative():

    a = Poly(2*x**2 + 4*x*t - 2*t - x**2*t, t)
    d = Poly((2*x + t)*(t + x**2), t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert recognize_log_derivative(a, d, DE, z) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    assert recognize_log_derivative(Poly(t + 1, t), Poly(t + x, t), DE) == True
    assert recognize_log_derivative(Poly(2, t), Poly(t**2 - 1, t), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert recognize_log_derivative(Poly(1, x), Poly(x**2 - 2, x), DE) == False
    assert recognize_log_derivative(Poly(1, x), Poly(x**2 + x, x), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert recognize_log_derivative(Poly(1, t), Poly(t**2 - 2, t), DE) == False
    assert recognize_log_derivative(Poly(1, t), Poly(t**2 + t, t), DE) == False


def test_residue_reduce():
    a = Poly(2*t**2 - t - x**2, t)
    d = Poly(t**3 - x**2*t, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)], 'Tfuncs': [log]})
    assert residue_reduce(a, d, DE, z, invert=False) == \
        ([(Poly(z**2 - Rational(1, 4), z, domain='ZZ(x)'),
          Poly((1 + 3*x*z - 6*z**2 - 2*x**2 + 4*x**2*z**2)*t - x*z + x**2 +
              2*x**2*z**2 - 2*z*x**3, t, domain='ZZ(z, x)'))], False)
    assert residue_reduce(a, d, DE, z, invert=True) == \
        ([(Poly(z**2 - Rational(1, 4), z, domain='ZZ(x)'), Poly(t + 2*x*z, t))], False)
    assert residue_reduce(Poly(-2/x, t), Poly(t**2 - 1, t,), DE, z, invert=False) == \
        ([(Poly(z**2 - 1, z, domain='QQ'), Poly(-2*z*t/x - 2/x, t, domain='ZZ(z,x)'))], True)
    ans = residue_reduce(Poly(-2/x, t), Poly(t**2 - 1, t), DE, z, invert=True)
    assert ans == ([(Poly(z**2 - 1, z, domain='QQ'), Poly(t + z, t))], True)
    assert residue_reduce_to_basic(ans[0], DE, z) == -log(-1 + log(x)) + log(1 + log(x))

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - t/x - (1 - nu**2/x**2), t)]})
    # TODO: Skip or make faster
    assert residue_reduce(Poly((-2*nu**2 - x**4)/(2*x**2)*t - (1 + x**2)/x, t),
    Poly(t**2 + 1 + x**2/2, t), DE, z) == \
        ([(Poly(z + S.Half, z, domain='QQ'), Poly(t**2 + 1 + x**2/2, t,
            domain='ZZ(x,nu)'))], True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    assert residue_reduce(Poly(-2*x*t + 1 - x**2, t),
    Poly(t**2 + 2*x*t + 1 + x**2, t), DE, z) == \
        ([(Poly(z**2 + Rational(1, 4), z), Poly(t + x + 2*z, t))], True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert residue_reduce(Poly(t, t), Poly(t + sqrt(2), t), DE, z) == \
        ([(Poly(z - 1, z, domain='QQ'), Poly(t + sqrt(2), t))], True)


def test_integrate_hyperexponential():
    # TODO: Add tests for integrate_hyperexponential() from the book
    a = Poly((1 + 2*t1 + t1**2 + 2*t1**3)*t**2 + (1 + t1**2)*t + 1 + t1**2, t)
    d = Poly(1, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t1**2, t1),
        Poly(t*(1 + t1**2), t)], 'Tfuncs': [tan, Lambda(i, exp(tan(i)))]})
    assert integrate_hyperexponential(a, d, DE) == \
        (exp(2*tan(x))*tan(x) + exp(tan(x)), 1 + t1**2, True)
    a = Poly((t1**3 + (x + 1)*t1**2 + t1 + x + 2)*t, t)
    assert integrate_hyperexponential(a, d, DE) == \
        ((x + tan(x))*exp(tan(x)), 0, True)

    a = Poly(t, t)
    d = Poly(1, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2*x*t, t)],
        'Tfuncs': [Lambda(i, exp(x**2))]})

    assert integrate_hyperexponential(a, d, DE) == \
        (0, NonElementaryIntegral(exp(x**2), x), False)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)], 'Tfuncs': [exp]})
    assert integrate_hyperexponential(a, d, DE) == (exp(x), 0, True)

    a = Poly(25*t**6 - 10*t**5 + 7*t**4 - 8*t**3 + 13*t**2 + 2*t - 1, t)
    d = Poly(25*t**6 + 35*t**4 + 11*t**2 + 1, t)
    assert integrate_hyperexponential(a, d, DE) == \
        (-(11 - 10*exp(x))/(5 + 25*exp(2*x)) + log(1 + exp(2*x)), -1, True)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(t0*t, t)],
        'Tfuncs': [exp, Lambda(i, exp(exp(i)))]})
    assert integrate_hyperexponential(Poly(2*t0*t**2, t), Poly(1, t), DE) == (exp(2*exp(x)), 0, True)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(-t0*t, t)],
        'Tfuncs': [exp, Lambda(i, exp(-exp(i)))]})
    assert integrate_hyperexponential(Poly(-27*exp(9) - 162*t0*exp(9) +
    27*x*t0*exp(9), t), Poly((36*exp(18) + x**2*exp(18) - 12*x*exp(18))*t, t), DE) == \
        (27*exp(exp(x))/(-6*exp(9) + x*exp(9)), 0, True)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)], 'Tfuncs': [exp]})
    assert integrate_hyperexponential(Poly(x**2/2*t, t), Poly(1, t), DE) == \
        ((2 - 2*x + x**2)*exp(x)/2, 0, True)
    assert integrate_hyperexponential(Poly(1 + t, t), Poly(t, t), DE) == \
        (-exp(-x), 1, True)  # x - exp(-x)
    assert integrate_hyperexponential(Poly(x, t), Poly(t + 1, t), DE) == \
        (0, NonElementaryIntegral(x/(1 + exp(x)), x), False)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t0), Poly(2*x*t1, t1)],
        'Tfuncs': [log, Lambda(i, exp(i**2))]})

    elem, nonelem, b = integrate_hyperexponential(Poly((8*x**7 - 12*x**5 + 6*x**3 - x)*t1**4 +
        (8*t0*x**7 - 8*t0*x**6 - 4*t0*x**5 + 2*t0*x**3 + 2*t0*x**2 - t0*x +
        24*x**8 - 36*x**6 - 4*x**5 + 22*x**4 + 4*x**3 - 7*x**2 - x + 1)*t1**3
        + (8*t0*x**8 - 4*t0*x**6 - 16*t0*x**5 - 2*t0*x**4 + 12*t0*x**3 +
        t0*x**2 - 2*t0*x + 24*x**9 - 36*x**7 - 8*x**6 + 22*x**5 + 12*x**4 -
        7*x**3 - 6*x**2 + x + 1)*t1**2 + (8*t0*x**8 - 8*t0*x**6 - 16*t0*x**5 +
        6*t0*x**4 + 10*t0*x**3 - 2*t0*x**2 - t0*x + 8*x**10 - 12*x**8 - 4*x**7
        + 2*x**6 + 12*x**5 + 3*x**4 - 9*x**3 - x**2 + 2*x)*t1 + 8*t0*x**7 -
        12*t0*x**6 - 4*t0*x**5 + 8*t0*x**4 - t0*x**2 - 4*x**7 + 4*x**6 +
        4*x**5 - 4*x**4 - x**3 + x**2, t1), Poly((8*x**7 - 12*x**5 + 6*x**3 -
        x)*t1**4 + (24*x**8 + 8*x**7 - 36*x**6 - 12*x**5 + 18*x**4 + 6*x**3 -
        3*x**2 - x)*t1**3 + (24*x**9 + 24*x**8 - 36*x**7 - 36*x**6 + 18*x**5 +
        18*x**4 - 3*x**3 - 3*x**2)*t1**2 + (8*x**10 + 24*x**9 - 12*x**8 -
        36*x**7 + 6*x**6 + 18*x**5 - x**4 - 3*x**3)*t1 + 8*x**10 - 12*x**8 +
        6*x**6 - x**4, t1), DE)

    assert factor(elem) == -((x - 1)*log(x)/((x + exp(x**2))*(2*x**2 - 1)))
    assert (nonelem, b) == (NonElementaryIntegral(exp(x**2)/(exp(x**2) + 1), x), False)

def test_integrate_hyperexponential_polynomial():
    # Without proper cancellation within integrate_hyperexponential_polynomial(),
    # this will take a long time to complete, and will return a complicated
    # expression
    p = Poly((-28*x**11*t0 - 6*x**8*t0 + 6*x**9*t0 - 15*x**8*t0**2 +
        15*x**7*t0**2 + 84*x**10*t0**2 - 140*x**9*t0**3 - 20*x**6*t0**3 +
        20*x**7*t0**3 - 15*x**6*t0**4 + 15*x**5*t0**4 + 140*x**8*t0**4 -
        84*x**7*t0**5 - 6*x**4*t0**5 + 6*x**5*t0**5 + x**3*t0**6 - x**4*t0**6 +
        28*x**6*t0**6 - 4*x**5*t0**7 + x**9 - x**10 + 4*x**12)/(-8*x**11*t0 +
        28*x**10*t0**2 - 56*x**9*t0**3 + 70*x**8*t0**4 - 56*x**7*t0**5 +
        28*x**6*t0**6 - 8*x**5*t0**7 + x**4*t0**8 + x**12)*t1**2 +
        (-28*x**11*t0 - 12*x**8*t0 + 12*x**9*t0 - 30*x**8*t0**2 +
        30*x**7*t0**2 + 84*x**10*t0**2 - 140*x**9*t0**3 - 40*x**6*t0**3 +
        40*x**7*t0**3 - 30*x**6*t0**4 + 30*x**5*t0**4 + 140*x**8*t0**4 -
        84*x**7*t0**5 - 12*x**4*t0**5 + 12*x**5*t0**5 - 2*x**4*t0**6 +
        2*x**3*t0**6 + 28*x**6*t0**6 - 4*x**5*t0**7 + 2*x**9 - 2*x**10 +
        4*x**12)/(-8*x**11*t0 + 28*x**10*t0**2 - 56*x**9*t0**3 +
        70*x**8*t0**4 - 56*x**7*t0**5 + 28*x**6*t0**6 - 8*x**5*t0**7 +
        x**4*t0**8 + x**12)*t1 + (-2*x**2*t0 + 2*x**3*t0 + x*t0**2 -
        x**2*t0**2 + x**3 - x**4)/(-4*x**5*t0 + 6*x**4*t0**2 - 4*x**3*t0**3 +
        x**2*t0**4 + x**6), t1, z, expand=False)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t0), Poly(2*x*t1, t1)]})
    assert integrate_hyperexponential_polynomial(p, DE, z) == (
        Poly((x - t0)*t1**2 + (-2*t0 + 2*x)*t1, t1), Poly(-2*x*t0 + x**2 +
        t0**2, t1), True)

    DE = DifferentialExtension(extension={'D':[Poly(1, x), Poly(t0, t0)]})
    assert integrate_hyperexponential_polynomial(Poly(0, t0), DE, z) == (
        Poly(0, t0), Poly(1, t0), True)


def test_integrate_hyperexponential_returns_piecewise():
    a, b = symbols('a b')
    DE = DifferentialExtension(a**x, x)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(x*log(a))/log(a), Ne(log(a), 0)), (x, True)), 0, True)
    DE = DifferentialExtension(a**(b*x), x)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(b*x*log(a))/(b*log(a)), Ne(b*log(a), 0)), (x, True)), 0, True)
    DE = DifferentialExtension(exp(a*x), x)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(a*x)/a, Ne(a, 0)), (x, True)), 0, True)
    DE = DifferentialExtension(x*exp(a*x), x)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        ((a*x - 1)*exp(a*x)/a**2, Ne(a**2, 0)), (x**2/2, True)), 0, True)
    DE = DifferentialExtension(x**2*exp(a*x), x)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        ((x**2*a**2 - 2*a*x + 2)*exp(a*x)/a**3, Ne(a**3, 0)),
        (x**3/3, True)), 0, True)
    DE = DifferentialExtension(x**y + z, y)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(log(x)*y)/log(x), Ne(log(x), 0)), (y, True)), z, True)
    DE = DifferentialExtension(x**y + z + x**(2*y), y)
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        ((exp(2*log(x)*y)*log(x) +
            2*exp(log(x)*y)*log(x))/(2*log(x)**2), Ne(2*log(x)**2, 0)),
            (2*y, True),
        ), z, True)
    # TODO: Add a test where two different parts of the extension use a
    # Piecewise, like y**x + z**x.


def test_issue_13947():
    a, t, s = symbols('a t s')
    assert risch_integrate(2**(-pi)/(2**t + 1), t) == \
        2**(-pi)*t - 2**(-pi)*log(2**t + 1)/log(2)
    assert risch_integrate(a**(t - s)/(a**t + 1), t) == \
        exp(-s*log(a))*log(a**t + 1)/log(a)


def test_integrate_primitive():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)],
        'Tfuncs': [log]})
    assert integrate_primitive(Poly(t, t), Poly(1, t), DE) == (x*log(x), -1, True)
    assert integrate_primitive(Poly(x, t), Poly(t, t), DE) == (0, NonElementaryIntegral(x/log(x), x), False)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(1/(x + 1), t2)],
        'Tfuncs': [log, Lambda(i, log(i + 1))]})
    assert integrate_primitive(Poly(t1, t2), Poly(t2, t2), DE) == \
        (0, NonElementaryIntegral(log(x)/log(1 + x), x), False)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(1/(x*t1), t2)],
        'Tfuncs': [log, Lambda(i, log(log(i)))]})
    assert integrate_primitive(Poly(t2, t2), Poly(t1, t2), DE) == \
        (0, NonElementaryIntegral(log(log(x))/log(x), x), False)

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t0)],
        'Tfuncs': [log]})
    assert integrate_primitive(Poly(x**2*t0**3 + (3*x**2 + x)*t0**2 + (3*x**2
    + 2*x)*t0 + x**2 + x, t0), Poly(x**2*t0**4 + 4*x**2*t0**3 + 6*x**2*t0**2 +
    4*x**2*t0 + x**2, t0), DE) == \
        (-1/(log(x) + 1), NonElementaryIntegral(1/(log(x) + 1), x), False)

def test_integrate_hypertangent_polynomial():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert integrate_hypertangent_polynomial(Poly(t**2 + x*t + 1, t), DE) == \
        (Poly(t, t), Poly(x/2, t))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(a*(t**2 + 1), t)]})
    assert integrate_hypertangent_polynomial(Poly(t**5, t), DE) == \
        (Poly(1/(4*a)*t**4 - 1/(2*a)*t**2, t), Poly(1/(2*a), t))


def test_integrate_nonlinear_no_specials():
    a, d, = Poly(x**2*t**5 + x*t**4 - nu**2*t**3 - x*(x**2 + 1)*t**2 - (x**2 -
    nu**2)*t - x**5/4, t), Poly(x**2*t**4 + x**2*(x**2 + 2)*t**2 + x**2 + x**4 + x**6/4, t)
    # f(x) == phi_nu(x), the logarithmic derivative of J_v, the Bessel function,
    # which has no specials (see Chapter 5, note 4 of Bronstein's book).
    f = Function('phi_nu')
    DE = DifferentialExtension(extension={'D': [Poly(1, x),
        Poly(-t**2 - t/x - (1 - nu**2/x**2), t)], 'Tfuncs': [f]})
    assert integrate_nonlinear_no_specials(a, d, DE) == \
        (-log(1 + f(x)**2 + x**2/2)/2 + (- 4 - x**2)/(4 + 2*x**2 + 4*f(x)**2), True)
    assert integrate_nonlinear_no_specials(Poly(t, t), Poly(1, t), DE) == \
        (0, False)


def test_integer_powers():
    assert integer_powers([x, x/2, x**2 + 1, x*Rational(2, 3)]) == [
            (x/6, [(x, 6), (x/2, 3), (x*Rational(2, 3), 4)]),
            (1 + x**2, [(1 + x**2, 1)])]


def test_DifferentialExtension_exp():
    assert DifferentialExtension(exp(x) + exp(x**2), x)._important_attrs == \
        (Poly(t1 + t0, t1), Poly(1, t1), [Poly(1, x,), Poly(t0, t0),
        Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i)),
        Lambda(i, exp(i**2))], [], [None, 'exp', 'exp'], [None, x, x**2])
    assert DifferentialExtension(exp(x) + exp(2*x), x)._important_attrs == \
        (Poly(t0**2 + t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0, t0)], [x, t0],
        [Lambda(i, exp(i))], [], [None, 'exp'], [None, x])
    assert DifferentialExtension(exp(x) + exp(x/2), x)._important_attrs == \
        (Poly(t0**2 + t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0/2, t0)],
        [x, t0], [Lambda(i, exp(i/2))], [], [None, 'exp'], [None, x/2])
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x + x**2), x)._important_attrs == \
        (Poly((1 + t0)*t1 + t0, t1), Poly(1, t1), [Poly(1, x), Poly(t0, t0),
        Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i)),
        Lambda(i, exp(i**2))], [], [None, 'exp', 'exp'], [None, x, x**2])
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x + x**2 + 1), x)._important_attrs == \
        (Poly((1 + S.Exp1*t0)*t1 + t0, t1), Poly(1, t1), [Poly(1, x),
        Poly(t0, t0), Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i)),
        Lambda(i, exp(i**2))], [], [None, 'exp', 'exp'], [None, x, x**2])
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x/2 + x**2), x)._important_attrs == \
        (Poly((t0 + 1)*t1 + t0**2, t1), Poly(1, t1), [Poly(1, x),
        Poly(t0/2, t0), Poly(2*x*t1, t1)], [x, t0, t1],
        [Lambda(i, exp(i/2)), Lambda(i, exp(i**2))],
        [(exp(x/2), sqrt(exp(x)))], [None, 'exp', 'exp'], [None, x/2, x**2])
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x/2 + x**2 + 3), x)._important_attrs == \
        (Poly((t0*exp(3) + 1)*t1 + t0**2, t1), Poly(1, t1), [Poly(1, x),
        Poly(t0/2, t0), Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i/2)),
        Lambda(i, exp(i**2))], [(exp(x/2), sqrt(exp(x)))], [None, 'exp', 'exp'],
        [None, x/2, x**2])
    assert DifferentialExtension(sqrt(exp(x)), x)._important_attrs == \
        (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0/2, t0)], [x, t0],
        [Lambda(i, exp(i/2))], [(exp(x/2), sqrt(exp(x)))], [None, 'exp'], [None, x/2])

    assert DifferentialExtension(exp(x/2), x)._important_attrs == \
        (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0/2, t0)], [x, t0],
        [Lambda(i, exp(i/2))], [], [None, 'exp'], [None, x/2])


def test_DifferentialExtension_log():
    assert DifferentialExtension(log(x)*log(x + 1)*log(2*x**2 + 2*x), x)._important_attrs == \
        (Poly(t0*t1**2 + (t0*log(2) + t0**2)*t1, t1), Poly(1, t1),
        [Poly(1, x), Poly(1/x, t0),
        Poly(1/(x + 1), t1, expand=False)], [x, t0, t1],
        [Lambda(i, log(i)), Lambda(i, log(i + 1))], [], [None, 'log', 'log'],
        [None, x, x + 1])
    assert DifferentialExtension(x**x*log(x), x)._important_attrs == \
        (Poly(t0*t1, t1), Poly(1, t1), [Poly(1, x), Poly(1/x, t0),
        Poly((1 + t0)*t1, t1)], [x, t0, t1], [Lambda(i, log(i)),
        Lambda(i, exp(t0*i))], [(exp(x*log(x)), x**x)], [None, 'log', 'exp'],
        [None, x, t0*x])


def test_DifferentialExtension_symlog():
    # See comment on test_risch_integrate below
    assert DifferentialExtension(log(x**x), x)._important_attrs == \
        (Poly(t0*x, t1), Poly(1, t1), [Poly(1, x), Poly(1/x, t0), Poly((t0 +
            1)*t1, t1)], [x, t0, t1], [Lambda(i, log(i)), Lambda(i, exp(i*t0))],
            [(exp(x*log(x)), x**x)], [None, 'log', 'exp'], [None, x, t0*x])
    assert DifferentialExtension(log(x**y), x)._important_attrs == \
        (Poly(y*t0, t0), Poly(1, t0), [Poly(1, x), Poly(1/x, t0)], [x, t0],
        [Lambda(i, log(i))], [(y*log(x), log(x**y))], [None, 'log'],
        [None, x])
    assert DifferentialExtension(log(sqrt(x)), x)._important_attrs == \
        (Poly(t0, t0), Poly(2, t0), [Poly(1, x), Poly(1/x, t0)], [x, t0],
        [Lambda(i, log(i))], [(log(x)/2, log(sqrt(x)))], [None, 'log'],
        [None, x])


def test_DifferentialExtension_handle_first():
    assert DifferentialExtension(exp(x)*log(x), x, handle_first='log')._important_attrs == \
        (Poly(t0*t1, t1), Poly(1, t1), [Poly(1, x), Poly(1/x, t0),
        Poly(t1, t1)], [x, t0, t1], [Lambda(i, log(i)), Lambda(i, exp(i))],
        [], [None, 'log', 'exp'], [None, x, x])
    assert DifferentialExtension(exp(x)*log(x), x, handle_first='exp')._important_attrs == \
        (Poly(t0*t1, t1), Poly(1, t1), [Poly(1, x), Poly(t0, t0),
        Poly(1/x, t1)], [x, t0, t1], [Lambda(i, exp(i)), Lambda(i, log(i))],
        [], [None, 'exp', 'log'], [None, x, x])

    # This one must have the log first, regardless of what we set it to
    # (because the log is inside of the exponential: x**x == exp(x*log(x)))
    assert DifferentialExtension(-x**x*log(x)**2 + x**x - x**x/x, x,
    handle_first='exp')._important_attrs == \
        DifferentialExtension(-x**x*log(x)**2 + x**x - x**x/x, x,
        handle_first='log')._important_attrs == \
        (Poly((-1 + x - x*t0**2)*t1, t1), Poly(x, t1),
            [Poly(1, x), Poly(1/x, t0), Poly((1 + t0)*t1, t1)], [x, t0, t1],
            [Lambda(i, log(i)), Lambda(i, exp(t0*i))], [(exp(x*log(x)), x**x)],
            [None, 'log', 'exp'], [None, x, t0*x])


def test_DifferentialExtension_all_attrs():
    # Test 'unimportant' attributes
    DE = DifferentialExtension(exp(x)*log(x), x, handle_first='exp')
    assert DE.f == exp(x)*log(x)
    assert DE.newf == t0*t1
    assert DE.x == x
    assert DE.cases == ['base', 'exp', 'primitive']
    assert DE.case == 'primitive'

    assert DE.level == -1
    assert DE.t == t1 == DE.T[DE.level]
    assert DE.d == Poly(1/x, t1) == DE.D[DE.level]
    raises(ValueError, lambda: DE.increment_level())
    DE.decrement_level()
    assert DE.level == -2
    assert DE.t == t0 == DE.T[DE.level]
    assert DE.d == Poly(t0, t0) == DE.D[DE.level]
    assert DE.case == 'exp'
    DE.decrement_level()
    assert DE.level == -3
    assert DE.t == x == DE.T[DE.level] == DE.x
    assert DE.d == Poly(1, x) == DE.D[DE.level]
    assert DE.case == 'base'
    raises(ValueError, lambda: DE.decrement_level())
    DE.increment_level()
    DE.increment_level()
    assert DE.level == -1
    assert DE.t == t1 == DE.T[DE.level]
    assert DE.d == Poly(1/x, t1) == DE.D[DE.level]
    assert DE.case == 'primitive'

    # Test methods
    assert DE.indices('log') == [2]
    assert DE.indices('exp') == [1]


def test_DifferentialExtension_extension_flag():
    raises(ValueError, lambda: DifferentialExtension(extension={'T': [x, t]}))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert DE._important_attrs == (None, None, [Poly(1, x), Poly(t, t)], [x, t],
        None, None, None, None)
    assert DE.d == Poly(t, t)
    assert DE.t == t
    assert DE.level == -1
    assert DE.cases == ['base', 'exp']
    assert DE.x == x
    assert DE.case == 'exp'

    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)],
        'exts': [None, 'exp'], 'extargs': [None, x]})
    assert DE._important_attrs == (None, None, [Poly(1, x), Poly(t, t)], [x, t],
        None, None, [None, 'exp'], [None, x])
    raises(ValueError, lambda: DifferentialExtension())


def test_DifferentialExtension_misc():
    # Odd ends
    assert DifferentialExtension(sin(y)*exp(x), x)._important_attrs == \
        (Poly(sin(y)*t0, t0, domain='ZZ[sin(y)]'), Poly(1, t0, domain='ZZ'),
        [Poly(1, x, domain='ZZ'), Poly(t0, t0, domain='ZZ')], [x, t0],
        [Lambda(i, exp(i))], [], [None, 'exp'], [None, x])
    raises(NotImplementedError, lambda: DifferentialExtension(sin(x), x))
    assert DifferentialExtension(10**x, x)._important_attrs == \
        (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(log(10)*t0, t0)], [x, t0],
        [Lambda(i, exp(i*log(10)))], [(exp(x*log(10)), 10**x)], [None, 'exp'],
        [None, x*log(10)])
    assert DifferentialExtension(log(x) + log(x**2), x)._important_attrs in [
        (Poly(3*t0, t0), Poly(2, t0), [Poly(1, x), Poly(2/x, t0)], [x, t0],
        [Lambda(i, log(i**2))], [], [None, ], [], [1], [x**2]),
        (Poly(3*t0, t0), Poly(1, t0), [Poly(1, x), Poly(1/x, t0)], [x, t0],
        [Lambda(i, log(i))], [], [None, 'log'], [None, x])]
    assert DifferentialExtension(S.Zero, x)._important_attrs == \
        (Poly(0, x), Poly(1, x), [Poly(1, x)], [x], [], [], [None], [None])
    assert DifferentialExtension(tan(atan(x).rewrite(log)), x)._important_attrs == \
        (Poly(x, x), Poly(1, x), [Poly(1, x)], [x], [], [], [None], [None])


def test_DifferentialExtension_Rothstein():
    # Rothstein's integral
    f = (2581284541*exp(x) + 1757211400)/(39916800*exp(3*x) +
    119750400*exp(x)**2 + 119750400*exp(x) + 39916800)*exp(1/(exp(x) + 1) - 10*x)
    assert DifferentialExtension(f, x)._important_attrs == \
        (Poly((1757211400 + 2581284541*t0)*t1, t1), Poly(39916800 +
        119750400*t0 + 119750400*t0**2 + 39916800*t0**3, t1),
        [Poly(1, x), Poly(t0, t0), Poly(-(10 + 21*t0 + 10*t0**2)/(1 + 2*t0 +
        t0**2)*t1, t1, domain='ZZ(t0)')], [x, t0, t1],
        [Lambda(i, exp(i)), Lambda(i, exp(1/(t0 + 1) - 10*i))], [],
        [None, 'exp', 'exp'], [None, x, 1/(t0 + 1) - 10*x])


class _TestingException(Exception):
    """Dummy Exception class for testing."""
    pass


def test_DecrementLevel():
    DE = DifferentialExtension(x*log(exp(x) + 1), x)
    assert DE.level == -1
    assert DE.t == t1
    assert DE.d == Poly(t0/(t0 + 1), t1)
    assert DE.case == 'primitive'

    with DecrementLevel(DE):
        assert DE.level == -2
        assert DE.t == t0
        assert DE.d == Poly(t0, t0)
        assert DE.case == 'exp'

        with DecrementLevel(DE):
            assert DE.level == -3
            assert DE.t == x
            assert DE.d == Poly(1, x)
            assert DE.case == 'base'

        assert DE.level == -2
        assert DE.t == t0
        assert DE.d == Poly(t0, t0)
        assert DE.case == 'exp'

    assert DE.level == -1
    assert DE.t == t1
    assert DE.d == Poly(t0/(t0 + 1), t1)
    assert DE.case == 'primitive'

    # Test that __exit__ is called after an exception correctly
    try:
        with DecrementLevel(DE):
            raise _TestingException
    except _TestingException:
        pass
    else:
        raise AssertionError("Did not raise.")

    assert DE.level == -1
    assert DE.t == t1
    assert DE.d == Poly(t0/(t0 + 1), t1)
    assert DE.case == 'primitive'


def test_risch_integrate():
    assert risch_integrate(t0*exp(x), x) == t0*exp(x)
    assert risch_integrate(sin(x), x, rewrite_complex=True) == -exp(I*x)/2 - exp(-I*x)/2

    # From my GSoC writeup
    assert risch_integrate((1 + 2*x**2 + x**4 + 2*x**3*exp(2*x**2))/
    (x**4*exp(x**2) + 2*x**2*exp(x**2) + exp(x**2)), x) == \
        NonElementaryIntegral(exp(-x**2), x) + exp(x**2)/(1 + x**2)


    assert risch_integrate(0, x) == 0

    # also tests prde_cancel()
    e1 = log(x/exp(x) + 1)
    ans1 = risch_integrate(e1, x)
    assert ans1 == (x*log(x*exp(-x) + 1) + NonElementaryIntegral((x**2 - x)/(x + exp(x)), x))
    assert cancel(diff(ans1, x) - e1) == 0

    # also tests issue #10798
    e2 = (log(-1/y)/2 - log(1/y)/2)/y - (log(1 - 1/y)/2 - log(1 + 1/y)/2)/y
    ans2 = risch_integrate(e2, y)
    assert ans2 == log(1/y)*log(1 - 1/y)/2 - log(1/y)*log(1 + 1/y)/2 + \
            NonElementaryIntegral((I*pi*y**2 - 2*y*log(1/y) - I*pi)/(2*y**3 - 2*y), y)
    assert expand_log(cancel(diff(ans2, y) - e2), force=True) == 0

    # These are tested here in addition to in test_DifferentialExtension above
    # (symlogs) to test that backsubs works correctly.  The integrals should be
    # written in terms of the original logarithms in the integrands.

    # XXX: Unfortunately, making backsubs work on this one is a little
    # trickier, because x**x is converted to exp(x*log(x)), and so log(x**x)
    # is converted to x*log(x). (x**2*log(x)).subs(x*log(x), log(x**x)) is
    # smart enough, the issue is that these splits happen at different places
    # in the algorithm.  Maybe a heuristic is in order
    assert risch_integrate(log(x**x), x) == x**2*log(x)/2 - x**2/4

    assert risch_integrate(log(x**y), x) == x*log(x**y) - x*y
    assert risch_integrate(log(sqrt(x)), x) == x*log(sqrt(x)) - x/2


def test_risch_integrate_float():
    assert risch_integrate((-60*exp(x) - 19.2*exp(4*x))*exp(4*x), x) == -2.4*exp(8*x) - 12.0*exp(5*x)


def test_NonElementaryIntegral():
    assert isinstance(risch_integrate(exp(x**2), x), NonElementaryIntegral)
    assert isinstance(risch_integrate(x**x*log(x), x), NonElementaryIntegral)
    # Make sure methods of Integral still give back a NonElementaryIntegral
    assert isinstance(NonElementaryIntegral(x**x*t0, x).subs(t0, log(x)), NonElementaryIntegral)


def test_xtothex():
    a = risch_integrate(x**x, x)
    assert a == NonElementaryIntegral(x**x, x)
    assert isinstance(a, NonElementaryIntegral)


def test_DifferentialExtension_equality():
    DE1 = DE2 = DifferentialExtension(log(x), x)
    assert DE1 == DE2


def test_DifferentialExtension_printing():
    DE = DifferentialExtension(exp(2*x**2) + log(exp(x**2) + 1), x)
    assert repr(DE) == ("DifferentialExtension(dict([('f', exp(2*x**2) + log(exp(x**2) + 1)), "
        "('x', x), ('T', [x, t0, t1]), ('D', [Poly(1, x, domain='ZZ'), Poly(2*x*t0, t0, domain='ZZ[x]'), "
        "Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')]), ('fa', Poly(t1 + t0**2, t1, domain='ZZ[t0]')), "
        "('fd', Poly(1, t1, domain='ZZ')), ('Tfuncs', [Lambda(i, exp(i**2)), Lambda(i, log(t0 + 1))]), "
        "('backsubs', []), ('exts', [None, 'exp', 'log']), ('extargs', [None, x**2, t0 + 1]), "
        "('cases', ['base', 'exp', 'primitive']), ('case', 'primitive'), ('t', t1), "
        "('d', Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')), ('newf', t0**2 + t1), ('level', -1), "
        "('dummy', False)]))")

    assert str(DE) == ("DifferentialExtension({fa=Poly(t1 + t0**2, t1, domain='ZZ[t0]'), "
        "fd=Poly(1, t1, domain='ZZ'), D=[Poly(1, x, domain='ZZ'), Poly(2*x*t0, t0, domain='ZZ[x]'), "
        "Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')]})")


def test_issue_23948():
    f = (
        ( (-2*x**5 + 28*x**4 - 144*x**3 + 324*x**2 - 270*x)*log(x)**2
         +(-4*x**6 + 56*x**5 - 288*x**4 + 648*x**3 - 540*x**2)*log(x)
         +(2*x**5 - 28*x**4 + 144*x**3 - 324*x**2 + 270*x)*exp(x)
         +(2*x**5 - 28*x**4 + 144*x**3 - 324*x**2 + 270*x)*log(5)
         -2*x**7 + 26*x**6 - 116*x**5 + 180*x**4 + 54*x**3 - 270*x**2
        )*log(-log(x)**2 - 2*x*log(x) + exp(x) + log(5) - x**2 - x)**2
       +( (4*x**5 - 44*x**4 + 168*x**3 - 216*x**2 - 108*x + 324)*log(x)
         +(-2*x**5 + 24*x**4 - 108*x**3 + 216*x**2 - 162*x)*exp(x)
         +4*x**6 - 42*x**5 + 144*x**4 - 108*x**3 - 324*x**2 + 486*x
        )*log(-log(x)**2 - 2*x*log(x) + exp(x) + log(5) - x**2 - x)
    )/(x*exp(x)**2*log(x)**2 + 2*x**2*exp(x)**2*log(x) - x*exp(x)**3
       +(-x*log(5) + x**3 + x**2)*exp(x)**2)

    F = ((x**4 - 12*x**3 + 54*x**2 - 108*x + 81)*exp(-2*x)
        *log(-x**2 - 2*x*log(x) - x + exp(x) - log(x)**2 + log(5))**2)

    assert risch_integrate(f, x) == F

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\tests\test_stochastic_process.py ===
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.function import Lambda
from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
from sympy.logic.boolalg import (And, Not)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.stats import (DiscreteMarkovChain, P, TransitionMatrixOf, E,
                         StochasticStateSpaceOf, variance, ContinuousMarkovChain,
                         BernoulliProcess, PoissonProcess, WienerProcess,
                         GammaProcess, sample_stochastic_process)
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import RandomIndexedSymbol
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.testing.pytest import (raises, skip, ignore_warnings,
                                  warns_deprecated_sympy)
from sympy.external import import_module
from sympy.stats.frv_types import BernoulliDistribution
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.crv_types import NormalDistribution, GammaDistribution
from sympy.core.symbol import Str


def test_DiscreteMarkovChain():

    # pass only the name
    X = DiscreteMarkovChain("X")
    assert isinstance(X.state_space, Range)
    assert X.index_set == S.Naturals0
    assert isinstance(X.transition_probabilities, MatrixSymbol)
    t = symbols('t', positive=True, integer=True)
    assert isinstance(X[t], RandomIndexedSymbol)
    assert E(X[0]) == Expectation(X[0])
    raises(TypeError, lambda: DiscreteMarkovChain(1))
    raises(NotImplementedError, lambda: X(t))
    raises(NotImplementedError, lambda: X.communication_classes())
    raises(NotImplementedError, lambda: X.canonical_form())
    raises(NotImplementedError, lambda: X.decompose())

    nz = Symbol('n', integer=True)
    TZ = MatrixSymbol('M', nz, nz)
    SZ = Range(nz)
    YZ = DiscreteMarkovChain('Y', SZ, TZ)
    assert P(Eq(YZ[2], 1), Eq(YZ[1], 0)) == TZ[0, 1]

    raises(ValueError, lambda: sample_stochastic_process(t))
    raises(ValueError, lambda: next(sample_stochastic_process(X)))
    # pass name and state_space
    # any hashable object should be a valid state
    # states should be valid as a tuple/set/list/Tuple/Range
    sym, rainy, cloudy, sunny = symbols('a Rainy Cloudy Sunny', real=True)
    state_spaces = [(1, 2, 3), [Str('Hello'), sym, DiscreteMarkovChain("Y", (1,2,3))],
                    Tuple(S(1), exp(sym), Str('World'), sympify=False), Range(-1, 5, 2),
                    [rainy, cloudy, sunny]]
    chains = [DiscreteMarkovChain("Y", state_space) for state_space in state_spaces]

    for i, Y in enumerate(chains):
        assert isinstance(Y.transition_probabilities, MatrixSymbol)
        assert Y.state_space == state_spaces[i] or Y.state_space == FiniteSet(*state_spaces[i])
        assert Y.number_of_states == 3

        with ignore_warnings(UserWarning):  # TODO: Restore tests once warnings are removed
            assert P(Eq(Y[2], 1), Eq(Y[0], 2), evaluate=False) == Probability(Eq(Y[2], 1), Eq(Y[0], 2))
        assert E(Y[0]) == Expectation(Y[0])

        raises(ValueError, lambda: next(sample_stochastic_process(Y)))

    raises(TypeError, lambda: DiscreteMarkovChain("Y", {1: 1}))
    Y = DiscreteMarkovChain("Y", Range(1, t, 2))
    assert Y.number_of_states == ceiling((t-1)/2)

    # pass name and transition_probabilities
    chains = [DiscreteMarkovChain("Y", trans_probs=Matrix([])),
              DiscreteMarkovChain("Y", trans_probs=Matrix([[0, 1], [1, 0]])),
              DiscreteMarkovChain("Y", trans_probs=Matrix([[pi, 1-pi], [sym, 1-sym]]))]
    for Z in chains:
        assert Z.number_of_states == Z.transition_probabilities.shape[0]
        assert isinstance(Z.transition_probabilities, ImmutableMatrix)

    # pass name, state_space and transition_probabilities
    T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    TS = MatrixSymbol('T', 3, 3)
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    YS = DiscreteMarkovChain("Y", ['One', 'Two', 3], TS)
    assert Y.joint_distribution(1, Y[2], 3) == JointDistribution(Y[1], Y[2], Y[3])
    raises(ValueError, lambda: Y.joint_distribution(Y[1].symbol, Y[2].symbol))
    assert P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2) == Float(0.36, 2)
    assert (P(Eq(YS[3], 2), Eq(YS[1], 1)) -
            (TS[0, 2]*TS[1, 0] + TS[1, 1]*TS[1, 2] + TS[1, 2]*TS[2, 2])).simplify() == 0
    assert P(Eq(YS[1], 1), Eq(YS[2], 2)) == Probability(Eq(YS[1], 1))
    assert P(Eq(YS[3], 3), Eq(YS[1], 1)) == TS[0, 2]*TS[1, 0] + TS[1, 1]*TS[1, 2] + TS[1, 2]*TS[2, 2]
    TO = Matrix([[0.25, 0.75, 0],[0, 0.25, 0.75],[0.75, 0, 0.25]])
    assert P(Eq(Y[3], 2), Eq(Y[1], 1) & TransitionMatrixOf(Y, TO)).round(3) == Float(0.375, 3)
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert E(Y[3], evaluate=False) == Expectation(Y[3])
        assert E(Y[3], Eq(Y[2], 1)).round(2) == Float(1.1, 3)
    TSO = MatrixSymbol('T', 4, 4)
    raises(ValueError, lambda: str(P(Eq(YS[3], 2), Eq(YS[1], 1) & TransitionMatrixOf(YS, TSO))))
    raises(TypeError, lambda: DiscreteMarkovChain("Z", [0, 1, 2], symbols('M')))
    raises(ValueError, lambda: DiscreteMarkovChain("Z", [0, 1, 2], MatrixSymbol('T', 3, 4)))
    raises(ValueError, lambda: E(Y[3], Eq(Y[2], 6)))
    raises(ValueError, lambda: E(Y[2], Eq(Y[3], 1)))


    # extended tests for probability queries
    TO1 = Matrix([[Rational(1, 4), Rational(3, 4), 0],[Rational(1, 3), Rational(1, 3), Rational(1, 3)],[0, Rational(1, 4), Rational(3, 4)]])
    assert P(And(Eq(Y[2], 1), Eq(Y[1], 1), Eq(Y[0], 0)),
            Eq(Probability(Eq(Y[0], 0)), Rational(1, 4)) & TransitionMatrixOf(Y, TO1)) == Rational(1, 16)
    assert P(And(Eq(Y[2], 1), Eq(Y[1], 1), Eq(Y[0], 0)), TransitionMatrixOf(Y, TO1)) == \
            Probability(Eq(Y[0], 0))/4
    assert P(Lt(X[1], 2) & Gt(X[1], 0), Eq(X[0], 2) &
        StochasticStateSpaceOf(X, [0, 1, 2]) & TransitionMatrixOf(X, TO1)) == Rational(1, 4)
    assert P(Lt(X[1], 2) & Gt(X[1], 0), Eq(X[0], 2) &
             StochasticStateSpaceOf(X, [S(0), '0', 1]) & TransitionMatrixOf(X, TO1)) == Rational(1, 4)
    assert P(Ne(X[1], 2) & Ne(X[1], 1), Eq(X[0], 2) &
        StochasticStateSpaceOf(X, [0, 1, 2]) & TransitionMatrixOf(X, TO1)) is S.Zero
    assert P(Ne(X[1], 2) & Ne(X[1], 1), Eq(X[0], 2) &
             StochasticStateSpaceOf(X, [S(0), '0', 1]) & TransitionMatrixOf(X, TO1)) is S.Zero
    assert P(And(Eq(Y[2], 1), Eq(Y[1], 1), Eq(Y[0], 0)), Eq(Y[1], 1)) == 0.1*Probability(Eq(Y[0], 0))

    # testing properties of Markov chain
    TO2 = Matrix([[S.One, 0, 0],[Rational(1, 3), Rational(1, 3), Rational(1, 3)],[0, Rational(1, 4), Rational(3, 4)]])
    TO3 = Matrix([[Rational(1, 4), Rational(3, 4), 0],[Rational(1, 3), Rational(1, 3), Rational(1, 3)], [0, Rational(1, 4), Rational(3, 4)]])
    Y2 = DiscreteMarkovChain('Y', trans_probs=TO2)
    Y3 = DiscreteMarkovChain('Y', trans_probs=TO3)
    assert Y3.fundamental_matrix() == ImmutableMatrix([[176, 81, -132], [36, 141, -52], [-44, -39, 208]])/125
    assert Y2.is_absorbing_chain() == True
    assert Y3.is_absorbing_chain() == False
    assert Y2.canonical_form() == ([0, 1, 2], TO2)
    assert Y3.canonical_form() == ([0, 1, 2], TO3)
    assert Y2.decompose() == ([0, 1, 2], TO2[0:1, 0:1], TO2[1:3, 0:1], TO2[1:3, 1:3])
    assert Y3.decompose() == ([0, 1, 2], TO3, Matrix(0, 3, []), Matrix(0, 0, []))
    TO4 = Matrix([[Rational(1, 5), Rational(2, 5), Rational(2, 5)], [Rational(1, 10), S.Half, Rational(2, 5)], [Rational(3, 5), Rational(3, 10), Rational(1, 10)]])
    Y4 = DiscreteMarkovChain('Y', trans_probs=TO4)
    w = ImmutableMatrix([[Rational(11, 39), Rational(16, 39), Rational(4, 13)]])
    assert Y4.limiting_distribution == w
    assert Y4.is_regular() == True
    assert Y4.is_ergodic() == True
    TS1 = MatrixSymbol('T', 3, 3)
    Y5 = DiscreteMarkovChain('Y', trans_probs=TS1)
    assert Y5.limiting_distribution(w, TO4).doit() == True
    assert Y5.stationary_distribution(condition_set=True).subs(TS1, TO4).contains(w).doit() == S.true
    TO6 = Matrix([[S.One, 0, 0, 0, 0],[S.Half, 0, S.Half, 0, 0],[0, S.Half, 0, S.Half, 0], [0, 0, S.Half, 0, S.Half], [0, 0, 0, 0, 1]])
    Y6 = DiscreteMarkovChain('Y', trans_probs=TO6)
    assert Y6.fundamental_matrix() == ImmutableMatrix([[Rational(3, 2), S.One, S.Half], [S.One, S(2), S.One], [S.Half, S.One, Rational(3, 2)]])
    assert Y6.absorbing_probabilities() == ImmutableMatrix([[Rational(3, 4), Rational(1, 4)], [S.Half, S.Half], [Rational(1, 4), Rational(3, 4)]])
    with warns_deprecated_sympy():
        Y6.absorbing_probabilites()
    TO7 = Matrix([[Rational(1, 2), Rational(1, 4), Rational(1, 4)], [Rational(1, 2), 0, Rational(1, 2)], [Rational(1, 4), Rational(1, 4), Rational(1, 2)]])
    Y7 = DiscreteMarkovChain('Y', trans_probs=TO7)
    assert Y7.is_absorbing_chain() == False
    assert Y7.fundamental_matrix() == ImmutableMatrix([[Rational(86, 75), Rational(1, 25), Rational(-14, 75)],
                                                            [Rational(2, 25), Rational(21, 25), Rational(2, 25)],
                                                            [Rational(-14, 75), Rational(1, 25), Rational(86, 75)]])

    # test for zero-sized matrix functionality
    X = DiscreteMarkovChain('X', trans_probs=Matrix([]))
    assert X.number_of_states == 0
    assert X.stationary_distribution() == Matrix([[]])
    assert X.communication_classes() == []
    assert X.canonical_form() == ([], Matrix([]))
    assert X.decompose() == ([], Matrix([]), Matrix([]), Matrix([]))
    assert X.is_regular() == False
    assert X.is_ergodic() == False

    # test communication_class
    # see https://drive.google.com/drive/folders/1HbxLlwwn2b3U8Lj7eb_ASIUb5vYaNIjg?usp=sharing
    # tutorial 2.pdf
    TO7 = Matrix([[0, 5, 5, 0, 0],
                  [0, 0, 0, 10, 0],
                  [5, 0, 5, 0, 0],
                  [0, 10, 0, 0, 0],
                  [0, 3, 0, 3, 4]])/10
    Y7 = DiscreteMarkovChain('Y', trans_probs=TO7)
    tuples = Y7.communication_classes()
    classes, recurrence, periods = list(zip(*tuples))
    assert classes == ([1, 3], [0, 2], [4])
    assert recurrence == (True, False, False)
    assert periods == (2, 1, 1)

    TO8 = Matrix([[0, 0, 0, 10, 0, 0],
                  [5, 0, 5, 0, 0, 0],
                  [0, 4, 0, 0, 0, 6],
                  [10, 0, 0, 0, 0, 0],
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 0, 5, 5, 0]])/10
    Y8 = DiscreteMarkovChain('Y', trans_probs=TO8)
    tuples = Y8.communication_classes()
    classes, recurrence, periods = list(zip(*tuples))
    assert classes == ([0, 3], [1, 2, 5, 4])
    assert recurrence == (True, False)
    assert periods == (2, 2)

    TO9 = Matrix([[2, 0, 0, 3, 0, 0, 3, 2, 0, 0],
                  [0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 2, 2, 0, 0, 0, 0, 0, 3, 3],
                  [0, 0, 0, 3, 0, 0, 6, 1, 0, 0],
                  [0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 10, 0, 0, 0, 0],
                  [4, 0, 0, 5, 0, 0, 1, 0, 0, 0],
                  [2, 0, 0, 4, 0, 0, 2, 2, 0, 0],
                  [3, 0, 1, 0, 0, 0, 0, 0, 4, 2],
                  [0, 0, 4, 0, 0, 0, 0, 0, 3, 3]])/10
    Y9 = DiscreteMarkovChain('Y', trans_probs=TO9)
    tuples = Y9.communication_classes()
    classes, recurrence, periods = list(zip(*tuples))
    assert classes == ([0, 3, 6, 7], [1], [2, 8, 9], [5], [4])
    assert recurrence == (True, True, False, True, False)
    assert periods == (1, 1, 1, 1, 1)

    # test canonical form
    # see https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    # example 11.13
    T = Matrix([[1, 0, 0, 0, 0],
                [S(1) / 2, 0, S(1) / 2, 0, 0],
                [0, S(1) / 2, 0, S(1) / 2, 0],
                [0, 0, S(1) / 2, 0, S(1) / 2],
                [0, 0, 0, 0, S(1)]])
    DW = DiscreteMarkovChain('DW', [0, 1, 2, 3, 4], T)
    states, A, B, C = DW.decompose()
    assert states == [0, 4, 1, 2, 3]
    assert A == Matrix([[1, 0], [0, 1]])
    assert B == Matrix([[S(1)/2, 0], [0, 0], [0, S(1)/2]])
    assert C == Matrix([[0, S(1)/2, 0], [S(1)/2, 0, S(1)/2], [0, S(1)/2, 0]])
    states, new_matrix = DW.canonical_form()
    assert states == [0, 4, 1, 2, 3]
    assert new_matrix == Matrix([[1, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0],
                                 [S(1)/2, 0, 0, S(1)/2, 0],
                                 [0, 0, S(1)/2, 0, S(1)/2],
                                 [0, S(1)/2, 0, S(1)/2, 0]])

    # test regular and ergodic
    # https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    T = Matrix([[0, 4, 0, 0, 0],
                [1, 0, 3, 0, 0],
                [0, 2, 0, 2, 0],
                [0, 0, 3, 0, 1],
                [0, 0, 0, 4, 0]])/4
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert not X.is_regular()
    assert X.is_ergodic()
    T = Matrix([[0, 1], [1, 0]])
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert not X.is_regular()
    assert X.is_ergodic()
    # http://www.math.wisc.edu/~valko/courses/331/MC2.pdf
    T = Matrix([[2, 1, 1],
                [2, 0, 2],
                [1, 1, 2]])/4
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert X.is_regular()
    assert X.is_ergodic()
    # https://docs.ufpr.br/~lucambio/CE222/1S2014/Kemeny-Snell1976.pdf
    T = Matrix([[1, 1], [1, 1]])/2
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert X.is_regular()
    assert X.is_ergodic()

    # test is_absorbing_chain
    T = Matrix([[0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]])
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert not X.is_absorbing_chain()
    # https://en.wikipedia.org/wiki/Absorbing_Markov_chain
    T = Matrix([[1, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 0, 0, 2]])/2
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert X.is_absorbing_chain()
    T = Matrix([[2, 0, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 0, 2]])/2
    X = DiscreteMarkovChain('X', trans_probs=T)
    assert X.is_absorbing_chain()

    # test custom state space
    Y10 = DiscreteMarkovChain('Y', [1, 2, 3], TO2)
    tuples = Y10.communication_classes()
    classes, recurrence, periods = list(zip(*tuples))
    assert classes == ([1], [2, 3])
    assert recurrence == (True, False)
    assert periods == (1, 1)
    assert Y10.canonical_form() == ([1, 2, 3], TO2)
    assert Y10.decompose() == ([1, 2, 3], TO2[0:1, 0:1], TO2[1:3, 0:1], TO2[1:3, 1:3])

    # testing miscellaneous queries
    T = Matrix([[S.Half, Rational(1, 4), Rational(1, 4)],
                [Rational(1, 3), 0, Rational(2, 3)],
                [S.Half, S.Half, 0]])
    X = DiscreteMarkovChain('X', [0, 1, 2], T)
    assert P(Eq(X[1], 2) & Eq(X[2], 1) & Eq(X[3], 0),
    Eq(P(Eq(X[1], 0)), Rational(1, 4)) & Eq(P(Eq(X[1], 1)), Rational(1, 4))) == Rational(1, 12)
    assert P(Eq(X[2], 1) | Eq(X[2], 2), Eq(X[1], 1)) == Rational(2, 3)
    assert P(Eq(X[2], 1) & Eq(X[2], 2), Eq(X[1], 1)) is S.Zero
    assert P(Ne(X[2], 2), Eq(X[1], 1)) == Rational(1, 3)
    assert E(X[1]**2, Eq(X[0], 1)) == Rational(8, 3)
    assert variance(X[1], Eq(X[0], 1)) == Rational(8, 9)
    raises(ValueError, lambda: E(X[1], Eq(X[2], 1)))
    raises(ValueError, lambda: DiscreteMarkovChain('X', [0, 1], T))

    # testing miscellaneous queries with different state space
    X = DiscreteMarkovChain('X', ['A', 'B', 'C'], T)
    assert P(Eq(X[1], 2) & Eq(X[2], 1) & Eq(X[3], 0),
    Eq(P(Eq(X[1], 0)), Rational(1, 4)) & Eq(P(Eq(X[1], 1)), Rational(1, 4))) == Rational(1, 12)
    assert P(Eq(X[2], 1) | Eq(X[2], 2), Eq(X[1], 1)) == Rational(2, 3)
    assert P(Eq(X[2], 1) & Eq(X[2], 2), Eq(X[1], 1)) is S.Zero
    assert P(Ne(X[2], 2), Eq(X[1], 1)) == Rational(1, 3)
    a = X.state_space.args[0]
    c = X.state_space.args[2]
    assert (E(X[1] ** 2, Eq(X[0], 1)) - (a**2/3 + 2*c**2/3)).simplify() == 0
    assert (variance(X[1], Eq(X[0], 1)) - (2*(-a/3 + c/3)**2/3 + (2*a/3 - 2*c/3)**2/3)).simplify() == 0
    raises(ValueError, lambda: E(X[1], Eq(X[2], 1)))

    #testing queries with multiple RandomIndexedSymbols
    T = Matrix([[Rational(5, 10), Rational(3, 10), Rational(2, 10)], [Rational(2, 10), Rational(7, 10), Rational(1, 10)], [Rational(3, 10), Rational(3, 10), Rational(4, 10)]])
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    assert P(Eq(Y[7], Y[5]), Eq(Y[2], 0)).round(5) == Float(0.44428, 5)
    assert P(Gt(Y[3], Y[1]), Eq(Y[0], 0)).round(2) == Float(0.36, 2)
    assert P(Le(Y[5], Y[10]), Eq(Y[4], 2)).round(6) == Float(0.583120, 6)
    assert Float(P(Eq(Y[10], Y[5]), Eq(Y[4], 1)), 14) == Float(1 - P(Ne(Y[10], Y[5]), Eq(Y[4], 1)), 14)
    assert Float(P(Gt(Y[8], Y[9]), Eq(Y[3], 2)), 14) == Float(1 - P(Le(Y[8], Y[9]), Eq(Y[3], 2)), 14)
    assert Float(P(Lt(Y[1], Y[4]), Eq(Y[0], 0)), 14) == Float(1 - P(Ge(Y[1], Y[4]), Eq(Y[0], 0)), 14)
    assert P(Eq(Y[5], Y[10]), Eq(Y[2], 1)) == P(Eq(Y[10], Y[5]), Eq(Y[2], 1))
    assert P(Gt(Y[1], Y[2]), Eq(Y[0], 1)) == P(Lt(Y[2], Y[1]), Eq(Y[0], 1))
    assert P(Ge(Y[7], Y[6]), Eq(Y[4], 1)) == P(Le(Y[6], Y[7]), Eq(Y[4], 1))

    #test symbolic queries
    a, b, c, d = symbols('a b c d')
    T = Matrix([[Rational(1, 10), Rational(4, 10), Rational(5, 10)], [Rational(3, 10), Rational(4, 10), Rational(3, 10)], [Rational(7, 10), Rational(2, 10), Rational(1, 10)]])
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    query = P(Eq(Y[a], b), Eq(Y[c], d))
    assert query.subs({a:10, b:2, c:5, d:1}).evalf().round(4) == P(Eq(Y[10], 2), Eq(Y[5], 1)).round(4)
    assert query.subs({a:15, b:0, c:10, d:1}).evalf().round(4) == P(Eq(Y[15], 0), Eq(Y[10], 1)).round(4)
    query_gt = P(Gt(Y[a], b), Eq(Y[c], d))
    query_le = P(Le(Y[a], b), Eq(Y[c], d))
    assert query_gt.subs({a:5, b:2, c:1, d:0}).evalf() + query_le.subs({a:5, b:2, c:1, d:0}).evalf() == 1.0
    query_ge = P(Ge(Y[a], b), Eq(Y[c], d))
    query_lt = P(Lt(Y[a], b), Eq(Y[c], d))
    assert query_ge.subs({a:4, b:1, c:0, d:2}).evalf() + query_lt.subs({a:4, b:1, c:0, d:2}).evalf() == 1.0

    #test issue 20078
    assert (2*Y[1] + 3*Y[1]).simplify() == 5*Y[1]
    assert (2*Y[1] - 3*Y[1]).simplify() == -Y[1]
    assert (2*(0.25*Y[1])).simplify() == 0.5*Y[1]
    assert ((2*Y[1]) * (0.25*Y[1])).simplify() == 0.5*Y[1]**2
    assert (Y[1]**2 + Y[1]**3).simplify() == (Y[1] + 1)*Y[1]**2

def test_sample_stochastic_process():
    if not import_module('scipy'):
        skip('SciPy Not installed. Skip sampling tests')
    import random
    random.seed(0)
    numpy = import_module('numpy')
    if numpy:
        numpy.random.seed(0) # scipy uses numpy to sample so to set its seed
    T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    for samps in range(10):
        assert next(sample_stochastic_process(Y)) in Y.state_space
    Z = DiscreteMarkovChain("Z", ['1', 1, 0], T)
    for samps in range(10):
        assert next(sample_stochastic_process(Z)) in Z.state_space

    T = Matrix([[S.Half, Rational(1, 4), Rational(1, 4)],
                [Rational(1, 3), 0, Rational(2, 3)],
                [S.Half, S.Half, 0]])
    X = DiscreteMarkovChain('X', [0, 1, 2], T)
    for samps in range(10):
        assert next(sample_stochastic_process(X)) in X.state_space
    W = DiscreteMarkovChain('W', [1, pi, oo], T)
    for samps in range(10):
        assert next(sample_stochastic_process(W)) in W.state_space


def test_ContinuousMarkovChain():
    T1 = Matrix([[S(-2), S(2), S.Zero],
                 [S.Zero, S.NegativeOne, S.One],
                 [Rational(3, 2), Rational(3, 2), S(-3)]])
    C1 = ContinuousMarkovChain('C', [0, 1, 2], T1)
    assert C1.limiting_distribution() == ImmutableMatrix([[Rational(3, 19), Rational(12, 19), Rational(4, 19)]])

    T2 = Matrix([[-S.One, S.One, S.Zero], [S.One, -S.One, S.Zero], [S.Zero, S.One, -S.One]])
    C2 = ContinuousMarkovChain('C', [0, 1, 2], T2)
    A, t = C2.generator_matrix, symbols('t', positive=True)
    assert C2.transition_probabilities(A)(t) == Matrix([[S.Half + exp(-2*t)/2, S.Half - exp(-2*t)/2, 0],
                                                       [S.Half - exp(-2*t)/2, S.Half + exp(-2*t)/2, 0],
                                                       [S.Half - exp(-t) + exp(-2*t)/2, S.Half - exp(-2*t)/2, exp(-t)]])
    with ignore_warnings(UserWarning): ### TODO: Restore tests once warnings are removed
        assert P(Eq(C2(1), 1), Eq(C2(0), 1), evaluate=False) == Probability(Eq(C2(1), 1), Eq(C2(0), 1))
    assert P(Eq(C2(1), 1), Eq(C2(0), 1)) == exp(-2)/2 + S.Half
    assert P(Eq(C2(1), 0) & Eq(C2(2), 1) & Eq(C2(3), 1),
                Eq(P(Eq(C2(1), 0)), S.Half)) == (Rational(1, 4) - exp(-2)/4)*(exp(-2)/2 + S.Half)
    assert P(Not(Eq(C2(1), 0) & Eq(C2(2), 1) & Eq(C2(3), 2)) |
                (Eq(C2(1), 0) & Eq(C2(2), 1) & Eq(C2(3), 2)),
                Eq(P(Eq(C2(1), 0)), Rational(1, 4)) & Eq(P(Eq(C2(1), 1)), Rational(1, 4))) is S.One
    assert E(C2(Rational(3, 2)), Eq(C2(0), 2)) == -exp(-3)/2 + 2*exp(Rational(-3, 2)) + S.Half
    assert variance(C2(Rational(3, 2)), Eq(C2(0), 1)) == ((S.Half - exp(-3)/2)**2*(exp(-3)/2 + S.Half)
                                                    + (Rational(-1, 2) - exp(-3)/2)**2*(S.Half - exp(-3)/2))
    raises(KeyError, lambda: P(Eq(C2(1), 0), Eq(P(Eq(C2(1), 1)), S.Half)))
    assert P(Eq(C2(1), 0), Eq(P(Eq(C2(5), 1)), S.Half)) == Probability(Eq(C2(1), 0))
    TS1 = MatrixSymbol('G', 3, 3)
    CS1 = ContinuousMarkovChain('C', [0, 1, 2], TS1)
    A = CS1.generator_matrix
    assert CS1.transition_probabilities(A)(t) == exp(t*A)

    C3 = ContinuousMarkovChain('C', [Symbol('0'), Symbol('1'), Symbol('2')], T2)
    assert P(Eq(C3(1), 1), Eq(C3(0), 1)) == exp(-2)/2 + S.Half
    assert P(Eq(C3(1), Symbol('1')), Eq(C3(0), Symbol('1'))) == exp(-2)/2 + S.Half

    #test probability queries
    G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    assert P(Eq(C(7.385), C(3.19)), Eq(C(0.862), 0)).round(5) == Float(0.35469, 5)
    assert P(Gt(C(98.715), C(19.807)), Eq(C(11.314), 2)).round(5) == Float(0.32452, 5)
    assert P(Le(C(5.9), C(10.112)), Eq(C(4), 1)).round(6) == Float(0.675214, 6)
    assert Float(P(Eq(C(7.32), C(2.91)), Eq(C(2.63), 1)), 14) == Float(1 - P(Ne(C(7.32), C(2.91)), Eq(C(2.63), 1)), 14)
    assert Float(P(Gt(C(3.36), C(1.101)), Eq(C(0.8), 2)), 14) == Float(1 - P(Le(C(3.36), C(1.101)), Eq(C(0.8), 2)), 14)
    assert Float(P(Lt(C(4.9), C(2.79)), Eq(C(1.61), 0)), 14) == Float(1 - P(Ge(C(4.9), C(2.79)), Eq(C(1.61), 0)), 14)
    assert P(Eq(C(5.243), C(10.912)), Eq(C(2.174), 1)) == P(Eq(C(10.912), C(5.243)), Eq(C(2.174), 1))
    assert P(Gt(C(2.344), C(9.9)), Eq(C(1.102), 1)) == P(Lt(C(9.9), C(2.344)), Eq(C(1.102), 1))
    assert P(Ge(C(7.87), C(1.008)), Eq(C(0.153), 1)) == P(Le(C(1.008), C(7.87)), Eq(C(0.153), 1))

    #test symbolic queries
    a, b, c, d = symbols('a b c d')
    query = P(Eq(C(a), b), Eq(C(c), d))
    assert query.subs({a:3.65, b:2, c:1.78, d:1}).evalf().round(10) == P(Eq(C(3.65), 2), Eq(C(1.78), 1)).round(10)
    query_gt = P(Gt(C(a), b), Eq(C(c), d))
    query_le = P(Le(C(a), b), Eq(C(c), d))
    assert query_gt.subs({a:13.2, b:0, c:3.29, d:2}).evalf() + query_le.subs({a:13.2, b:0, c:3.29, d:2}).evalf() == 1.0
    query_ge = P(Ge(C(a), b), Eq(C(c), d))
    query_lt = P(Lt(C(a), b), Eq(C(c), d))
    assert query_ge.subs({a:7.43, b:1, c:1.45, d:0}).evalf() + query_lt.subs({a:7.43, b:1, c:1.45, d:0}).evalf() == 1.0

    #test issue 20078
    assert (2*C(1) + 3*C(1)).simplify() == 5*C(1)
    assert (2*C(1) - 3*C(1)).simplify() == -C(1)
    assert (2*(0.25*C(1))).simplify() == 0.5*C(1)
    assert (2*C(1) * 0.25*C(1)).simplify() == 0.5*C(1)**2
    assert (C(1)**2 + C(1)**3).simplify() == (C(1) + 1)*C(1)**2

def test_BernoulliProcess():

    B = BernoulliProcess("B", p=0.6, success=1, failure=0)
    assert B.state_space == FiniteSet(0, 1)
    assert B.index_set == S.Naturals0
    assert B.success == 1
    assert B.failure == 0

    X = BernoulliProcess("X", p=Rational(1,3), success='H', failure='T')
    assert X.state_space == FiniteSet('H', 'T')
    H, T = symbols("H,T")
    assert E(X[1]+X[2]*X[3]) == H**2/9 + 4*H*T/9 + H/3 + 4*T**2/9 + 2*T/3

    t, x = symbols('t, x', positive=True, integer=True)
    assert isinstance(B[t], RandomIndexedSymbol)

    raises(ValueError, lambda: BernoulliProcess("X", p=1.1, success=1, failure=0))
    raises(NotImplementedError, lambda: B(t))

    raises(IndexError, lambda: B[-3])
    assert B.joint_distribution(B[3], B[9]) == JointDistributionHandmade(Lambda((B[3], B[9]),
                Piecewise((0.6, Eq(B[3], 1)), (0.4, Eq(B[3], 0)), (0, True))
                *Piecewise((0.6, Eq(B[9], 1)), (0.4, Eq(B[9], 0)), (0, True))))

    assert B.joint_distribution(2, B[4]) == JointDistributionHandmade(Lambda((B[2], B[4]),
                Piecewise((0.6, Eq(B[2], 1)), (0.4, Eq(B[2], 0)), (0, True))
                *Piecewise((0.6, Eq(B[4], 1)), (0.4, Eq(B[4], 0)), (0, True))))

    # Test for the sum distribution of Bernoulli Process RVs
    Y = B[1] + B[2] + B[3]
    assert P(Eq(Y, 0)).round(2) == Float(0.06, 1)
    assert P(Eq(Y, 2)).round(2) == Float(0.43, 2)
    assert P(Eq(Y, 4)).round(2) == 0
    assert P(Gt(Y, 1)).round(2) == Float(0.65, 2)
    # Test for independency of each Random Indexed variable
    assert P(Eq(B[1], 0) & Eq(B[2], 1) & Eq(B[3], 0) & Eq(B[4], 1)).round(2) == Float(0.06, 1)

    assert E(2 * B[1] + B[2]).round(2) == Float(1.80, 3)
    assert E(2 * B[1] + B[2] + 5).round(2) == Float(6.80, 3)
    assert E(B[2] * B[4] + B[10]).round(2) == Float(0.96, 2)
    assert E(B[2] > 0, Eq(B[1],1) & Eq(B[2],1)).round(2) == Float(0.60,2)
    assert E(B[1]) == 0.6
    assert P(B[1] > 0).round(2) == Float(0.60, 2)
    assert P(B[1] < 1).round(2) == Float(0.40, 2)
    assert P(B[1] > 0, B[2] <= 1).round(2) == Float(0.60, 2)
    assert P(B[12] * B[5] > 0).round(2) == Float(0.36, 2)
    assert P(B[12] * B[5] > 0, B[4] < 1).round(2) == Float(0.36, 2)
    assert P(Eq(B[2], 1), B[2] > 0) == 1.0
    assert P(Eq(B[5], 3)) == 0
    assert P(Eq(B[1], 1), B[1] < 0) == 0
    assert P(B[2] > 0, Eq(B[2], 1)) == 1
    assert P(B[2] < 0, Eq(B[2], 1)) == 0
    assert P(B[2] > 0, B[2]==7) == 0
    assert P(B[5] > 0, B[5]) == BernoulliDistribution(0.6, 0, 1)
    raises(ValueError, lambda: P(3))
    raises(ValueError, lambda: P(B[3] > 0, 3))

    # test issue 19456
    expr = Sum(B[t], (t, 0, 4))
    expr2 = Sum(B[t], (t, 1, 3))
    expr3 = Sum(B[t]**2, (t, 1, 3))
    assert expr.doit() == B[0] + B[1] + B[2] + B[3] + B[4]
    assert expr2.doit() == Y
    assert expr3.doit() == B[1]**2 + B[2]**2 + B[3]**2
    assert B[2*t].free_symbols == {B[2*t], t}
    assert B[4].free_symbols == {B[4]}
    assert B[x*t].free_symbols == {B[x*t], x, t}

    #test issue 20078
    assert (2*B[t] + 3*B[t]).simplify() == 5*B[t]
    assert (2*B[t] - 3*B[t]).simplify() == -B[t]
    assert (2*(0.25*B[t])).simplify() == 0.5*B[t]
    assert (2*B[t] * 0.25*B[t]).simplify() == 0.5*B[t]**2
    assert (B[t]**2 + B[t]**3).simplify() == (B[t] + 1)*B[t]**2

def test_PoissonProcess():
    X = PoissonProcess("X", 3)
    assert X.state_space == S.Naturals0
    assert X.index_set == Interval(0, oo)
    assert X.lamda == 3

    t, d, x, y = symbols('t d x y', positive=True)
    assert isinstance(X(t), RandomIndexedSymbol)
    assert X.distribution(t) == PoissonDistribution(3*t)
    with warns_deprecated_sympy():
        X.distribution(X(t))
    raises(ValueError, lambda: PoissonProcess("X", -1))
    raises(NotImplementedError, lambda: X[t])
    raises(IndexError, lambda: X(-5))

    assert X.joint_distribution(X(2), X(3)) == JointDistributionHandmade(Lambda((X(2), X(3)),
                6**X(2)*9**X(3)*exp(-15)/(factorial(X(2))*factorial(X(3)))))

    assert X.joint_distribution(4, 6) == JointDistributionHandmade(Lambda((X(4), X(6)),
                12**X(4)*18**X(6)*exp(-30)/(factorial(X(4))*factorial(X(6)))))

    assert P(X(t) < 1) == exp(-3*t)
    assert P(Eq(X(t), 0), Contains(t, Interval.Lopen(3, 5))) == exp(-6) # exp(-2*lamda)
    res = P(Eq(X(t), 1), Contains(t, Interval.Lopen(3, 4)))
    assert res == 3*exp(-3)

    # Equivalent to P(Eq(X(t), 1))**4 because of non-overlapping intervals
    assert P(Eq(X(t), 1) & Eq(X(d), 1) & Eq(X(x), 1) & Eq(X(y), 1), Contains(t, Interval.Lopen(0, 1))
    & Contains(d, Interval.Lopen(1, 2)) & Contains(x, Interval.Lopen(2, 3))
    & Contains(y, Interval.Lopen(3, 4))) == res**4

    # Return Probability because of overlapping intervals
    assert P(Eq(X(t), 2) & Eq(X(d), 3), Contains(t, Interval.Lopen(0, 2))
    & Contains(d, Interval.Ropen(2, 4))) == \
                Probability(Eq(X(d), 3) & Eq(X(t), 2), Contains(t, Interval.Lopen(0, 2))
                & Contains(d, Interval.Ropen(2, 4)))

    raises(ValueError, lambda: P(Eq(X(t), 2) & Eq(X(d), 3),
    Contains(t, Interval.Lopen(0, 4)) & Contains(d, Interval.Lopen(3, oo)))) # no bound on d
    assert P(Eq(X(3), 2)) == 81*exp(-9)/2
    assert P(Eq(X(t), 2), Contains(t, Interval.Lopen(0, 5))) == 225*exp(-15)/2

    # Check that probability works correctly by adding it to 1
    res1 = P(X(t) <= 3, Contains(t, Interval.Lopen(0, 5)))
    res2 = P(X(t) > 3, Contains(t, Interval.Lopen(0, 5)))
    assert res1 == 691*exp(-15)
    assert (res1 + res2).simplify() == 1

    # Check Not and  Or
    assert P(Not(Eq(X(t), 2) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) & \
            Contains(d, Interval.Lopen(7, 8))).simplify() == -18*exp(-6) + 234*exp(-9) + 1
    assert P(Eq(X(t), 2) | Ne(X(t), 4), Contains(t, Interval.Ropen(2, 4))) == 1 - 36*exp(-6)
    raises(ValueError, lambda: P(X(t) > 2, X(t) + X(d)))
    assert E(X(t)) == 3*t  # property of the distribution at a given timestamp
    assert E(X(t)**2 + X(d)*2 + X(y)**3, Contains(t, Interval.Lopen(0, 1))
        & Contains(d, Interval.Lopen(1, 2)) & Contains(y, Interval.Ropen(3, 4))) == 75
    assert E(X(t)**2, Contains(t, Interval.Lopen(0, 1))) == 12
    assert E(x*(X(t) + X(d))*(X(t)**2+X(d)**2), Contains(t, Interval.Lopen(0, 1))
    & Contains(d, Interval.Ropen(1, 2))) == \
            Expectation(x*(X(d) + X(t))*(X(d)**2 + X(t)**2), Contains(t, Interval.Lopen(0, 1))
            & Contains(d, Interval.Ropen(1, 2)))

    # Value Error because of infinite time bound
    raises(ValueError, lambda: E(X(t)**3, Contains(t, Interval.Lopen(1, oo))))

    # Equivalent to E(X(t)**2) - E(X(d)**2) == E(X(1)**2) - E(X(1)**2) == 0
    assert E((X(t) + X(d))*(X(t) - X(d)), Contains(t, Interval.Lopen(0, 1))
        & Contains(d, Interval.Lopen(1, 2))) == 0
    assert E(X(2) + x*E(X(5))) == 15*x + 6
    assert E(x*X(1) + y) == 3*x + y
    assert P(Eq(X(1), 2) & Eq(X(t), 3), Contains(t, Interval.Lopen(1, 2))) == 81*exp(-6)/4
    Y = PoissonProcess("Y", 6)
    Z = X + Y
    assert Z.lamda == X.lamda + Y.lamda == 9
    raises(ValueError, lambda: X + 5) # should be added be only PoissonProcess instance
    N, M = Z.split(4, 5)
    assert N.lamda == 4
    assert M.lamda == 5
    raises(ValueError, lambda: Z.split(3, 2)) # 2+3 != 9

    raises(ValueError, lambda :P(Eq(X(t), 0), Contains(t, Interval.Lopen(1, 3)) & Eq(X(1), 0)))
    # check if it handles queries with two random variables in one args
    res1 = P(Eq(N(3), N(5)))
    assert res1 == P(Eq(N(t), 0), Contains(t, Interval(3, 5)))
    res2 = P(N(3) > N(1))
    assert res2 == P((N(t) > 0), Contains(t, Interval(1, 3)))
    assert P(N(3) < N(1)) == 0 # condition is not possible
    res3 = P(N(3) <= N(1)) # holds only for Eq(N(3), N(1))
    assert res3 == P(Eq(N(t), 0), Contains(t, Interval(1, 3)))

    # tests from https://www.probabilitycourse.com/chapter11/11_1_2_basic_concepts_of_the_poisson_process.php
    X = PoissonProcess('X', 10) # 11.1
    assert P(Eq(X(S(1)/3), 3) & Eq(X(1), 10)) == exp(-10)*Rational(8000000000, 11160261)
    assert P(Eq(X(1), 1), Eq(X(S(1)/3), 3)) == 0
    assert P(Eq(X(1), 10), Eq(X(S(1)/3), 3)) == P(Eq(X(S(2)/3), 7))

    X = PoissonProcess('X', 2) # 11.2
    assert P(X(S(1)/2) < 1) == exp(-1)
    assert P(X(3) < 1, Eq(X(1), 0)) == exp(-4)
    assert P(Eq(X(4), 3), Eq(X(2), 3)) == exp(-4)

    X = PoissonProcess('X', 3)
    assert P(Eq(X(2), 5) & Eq(X(1), 2)) == Rational(81, 4)*exp(-6)

    # check few properties
    assert P(X(2) <= 3, X(1)>=1) == 3*P(Eq(X(1), 0)) + 2*P(Eq(X(1), 1)) + P(Eq(X(1), 2))
    assert P(X(2) <= 3, X(1) > 1) == 2*P(Eq(X(1), 0)) + 1*P(Eq(X(1), 1))
    assert P(Eq(X(2), 5) & Eq(X(1), 2)) == P(Eq(X(1), 3))*P(Eq(X(1), 2))
    assert P(Eq(X(3), 4), Eq(X(1), 3)) == P(Eq(X(2), 1))

    #test issue 20078
    assert (2*X(t) + 3*X(t)).simplify() == 5*X(t)
    assert (2*X(t) - 3*X(t)).simplify() == -X(t)
    assert (2*(0.25*X(t))).simplify() == 0.5*X(t)
    assert (2*X(t) * 0.25*X(t)).simplify() == 0.5*X(t)**2
    assert (X(t)**2 + X(t)**3).simplify() == (X(t) + 1)*X(t)**2

def test_WienerProcess():
    X = WienerProcess("X")
    assert X.state_space == S.Reals
    assert X.index_set == Interval(0, oo)

    t, d, x, y = symbols('t d x y', positive=True)
    assert isinstance(X(t), RandomIndexedSymbol)
    assert X.distribution(t) == NormalDistribution(0, sqrt(t))
    with warns_deprecated_sympy():
        X.distribution(X(t))
    raises(ValueError, lambda: PoissonProcess("X", -1))
    raises(NotImplementedError, lambda: X[t])
    raises(IndexError, lambda: X(-2))

    assert X.joint_distribution(X(2), X(3)) == JointDistributionHandmade(
        Lambda((X(2), X(3)), sqrt(6)*exp(-X(2)**2/4)*exp(-X(3)**2/6)/(12*pi)))
    assert X.joint_distribution(4, 6) == JointDistributionHandmade(
        Lambda((X(4), X(6)), sqrt(6)*exp(-X(4)**2/8)*exp(-X(6)**2/12)/(24*pi)))

    assert P(X(t) < 3).simplify() == erf(3*sqrt(2)/(2*sqrt(t)))/2 + S(1)/2
    assert P(X(t) > 2, Contains(t, Interval.Lopen(3, 7))).simplify() == S(1)/2 -\
                erf(sqrt(2)/2)/2

    # Equivalent to P(X(1)>1)**4
    assert P((X(t) > 4) & (X(d) > 3) & (X(x) > 2) & (X(y) > 1),
        Contains(t, Interval.Lopen(0, 1)) & Contains(d, Interval.Lopen(1, 2))
        & Contains(x, Interval.Lopen(2, 3)) & Contains(y, Interval.Lopen(3, 4))).simplify() ==\
        (1 - erf(sqrt(2)/2))*(1 - erf(sqrt(2)))*(1 - erf(3*sqrt(2)/2))*(1 - erf(2*sqrt(2)))/16

    # Contains an overlapping interval so, return Probability
    assert P((X(t)< 2) & (X(d)> 3), Contains(t, Interval.Lopen(0, 2))
        & Contains(d, Interval.Ropen(2, 4))) == Probability((X(d) > 3) & (X(t) < 2),
        Contains(d, Interval.Ropen(2, 4)) & Contains(t, Interval.Lopen(0, 2)))

    assert str(P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
        Contains(d, Interval.Lopen(7, 8))).simplify()) == \
                '-(1 - erf(3*sqrt(2)/2))*(2 - erfc(5/2))/4 + 1'
    # Distribution has mean 0 at each timestamp
    assert E(X(t)) == 0
    assert E(x*(X(t) + X(d))*(X(t)**2+X(d)**2), Contains(t, Interval.Lopen(0, 1))
    & Contains(d, Interval.Ropen(1, 2))) == Expectation(x*(X(d) + X(t))*(X(d)**2 + X(t)**2),
    Contains(d, Interval.Ropen(1, 2)) & Contains(t, Interval.Lopen(0, 1)))
    assert E(X(t) + x*E(X(3))) == 0

    #test issue 20078
    assert (2*X(t) + 3*X(t)).simplify() == 5*X(t)
    assert (2*X(t) - 3*X(t)).simplify() == -X(t)
    assert (2*(0.25*X(t))).simplify() == 0.5*X(t)
    assert (2*X(t) * 0.25*X(t)).simplify() == 0.5*X(t)**2
    assert (X(t)**2 + X(t)**3).simplify() == (X(t) + 1)*X(t)**2


def test_GammaProcess_symbolic():
    t, d, x, y, g, l = symbols('t d x y g l', positive=True)
    X = GammaProcess("X", l, g)

    raises(NotImplementedError, lambda: X[t])
    raises(IndexError, lambda: X(-1))
    assert isinstance(X(t), RandomIndexedSymbol)
    assert X.state_space == Interval(0, oo)
    assert X.distribution(t) == GammaDistribution(g*t, 1/l)
    with warns_deprecated_sympy():
        X.distribution(X(t))
    assert X.joint_distribution(5, X(3)) == JointDistributionHandmade(Lambda(
        (X(5), X(3)), l**(8*g)*exp(-l*X(3))*exp(-l*X(5))*X(3)**(3*g - 1)*X(5)**(5*g
        - 1)/(gamma(3*g)*gamma(5*g))))
    # property of the gamma process at any given timestamp
    assert E(X(t)) == g*t/l
    assert variance(X(t)).simplify() == g*t/l**2

    # Equivalent to E(2*X(1)) + E(X(1)**2) + E(X(1)**3), where E(X(1)) == g/l
    assert E(X(t)**2 + X(d)*2 + X(y)**3, Contains(t, Interval.Lopen(0, 1))
        & Contains(d, Interval.Lopen(1, 2)) & Contains(y, Interval.Ropen(3, 4))) == \
            2*g/l + (g**2 + g)/l**2 + (g**3 + 3*g**2 + 2*g)/l**3

    assert P(X(t) > 3, Contains(t, Interval.Lopen(3, 4))).simplify() == \
                                1 - lowergamma(g, 3*l)/gamma(g) # equivalent to P(X(1)>3)


    #test issue 20078
    assert (2*X(t) + 3*X(t)).simplify() == 5*X(t)
    assert (2*X(t) - 3*X(t)).simplify() == -X(t)
    assert (2*(0.25*X(t))).simplify() == 0.5*X(t)
    assert (2*X(t) * 0.25*X(t)).simplify() == 0.5*X(t)**2
    assert (X(t)**2 + X(t)**3).simplify() == (X(t) + 1)*X(t)**2
def test_GammaProcess_numeric():
    t, d, x, y = symbols('t d x y', positive=True)
    X = GammaProcess("X", 1, 2)
    assert X.state_space == Interval(0, oo)
    assert X.index_set == Interval(0, oo)
    assert X.lamda == 1
    assert X.gamma == 2

    raises(ValueError, lambda: GammaProcess("X", -1, 2))
    raises(ValueError, lambda: GammaProcess("X", 0, -2))
    raises(ValueError, lambda: GammaProcess("X", -1, -2))

    # all are independent because of non-overlapping intervals
    assert P((X(t) > 4) & (X(d) > 3) & (X(x) > 2) & (X(y) > 1), Contains(t,
        Interval.Lopen(0, 1)) & Contains(d, Interval.Lopen(1, 2)) & Contains(x,
        Interval.Lopen(2, 3)) & Contains(y, Interval.Lopen(3, 4))).simplify() == \
                                                            120*exp(-10)

    # Check working with Not and Or
    assert P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
        Contains(d, Interval.Lopen(7, 8))).simplify() == -4*exp(-3) + 472*exp(-8)/3 + 1
    assert P((X(t) > 2) | (X(t) < 4), Contains(t, Interval.Ropen(1, 4))).simplify() == \
                                            -643*exp(-4)/15 + 109*exp(-2)/15 + 1

    assert E(X(t)) == 2*t # E(X(t)) == gamma*t/l
    assert E(X(2) + x*E(X(5))) == 10*x + 4

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\tests\test_frame.py ===
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.simplify import trigsimp
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, zeros)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.vector import (ReferenceFrame, Vector, CoordinateSym,
                                  dynamicsymbols, time_derivative, express,
                                  dot)
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import VectorTypeError
from sympy.testing.pytest import raises
import warnings
import pickle


def test_dict_list():

    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    D = ReferenceFrame('D')
    E = ReferenceFrame('E')
    F = ReferenceFrame('F')

    B.orient_axis(A, A.x, 1.0)
    C.orient_axis(B, B.x, 1.0)
    D.orient_axis(C, C.x, 1.0)

    assert D._dict_list(A, 0) == [D, C, B, A]

    E.orient_axis(D, D.x, 1.0)

    assert C._dict_list(A, 0) == [C, B, A]
    assert C._dict_list(E, 0) == [C, D, E]

    # only 0, 1, 2 permitted for second argument
    raises(ValueError, lambda: C._dict_list(E, 5))
    # no connecting path
    raises(ValueError, lambda: F._dict_list(A, 0))


def test_coordinate_vars():
    """Tests the coordinate variables functionality"""
    A = ReferenceFrame('A')
    assert CoordinateSym('Ax', A, 0) == A[0]
    assert CoordinateSym('Ax', A, 1) == A[1]
    assert CoordinateSym('Ax', A, 2) == A[2]
    raises(ValueError, lambda: CoordinateSym('Ax', A, 3))
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    assert isinstance(A[0], CoordinateSym) and \
           isinstance(A[0], CoordinateSym) and \
           isinstance(A[0], CoordinateSym)
    assert A.variable_map(A) == {A[0]:A[0], A[1]:A[1], A[2]:A[2]}
    assert A[0].frame == A
    B = A.orientnew('B', 'Axis', [q, A.z])
    assert B.variable_map(A) == {B[2]: A[2], B[1]: -A[0]*sin(q) + A[1]*cos(q),
                                 B[0]: A[0]*cos(q) + A[1]*sin(q)}
    assert A.variable_map(B) == {A[0]: B[0]*cos(q) - B[1]*sin(q),
                                 A[1]: B[0]*sin(q) + B[1]*cos(q), A[2]: B[2]}
    assert time_derivative(B[0], A) == -A[0]*sin(q)*qd + A[1]*cos(q)*qd
    assert time_derivative(B[1], A) == -A[0]*cos(q)*qd - A[1]*sin(q)*qd
    assert time_derivative(B[2], A) == 0
    assert express(B[0], A, variables=True) == A[0]*cos(q) + A[1]*sin(q)
    assert express(B[1], A, variables=True) == -A[0]*sin(q) + A[1]*cos(q)
    assert express(B[2], A, variables=True) == A[2]
    assert time_derivative(A[0]*A.x + A[1]*A.y + A[2]*A.z, B) == A[1]*qd*A.x - A[0]*qd*A.y
    assert time_derivative(B[0]*B.x + B[1]*B.y + B[2]*B.z, A) == - B[1]*qd*B.x + B[0]*qd*B.y
    assert express(B[0]*B[1]*B[2], A, variables=True) == \
           A[2]*(-A[0]*sin(q) + A[1]*cos(q))*(A[0]*cos(q) + A[1]*sin(q))
    assert (time_derivative(B[0]*B[1]*B[2], A) -
            (A[2]*(-A[0]**2*cos(2*q) -
             2*A[0]*A[1]*sin(2*q) +
             A[1]**2*cos(2*q))*qd)).trigsimp() == 0
    assert express(B[0]*B.x + B[1]*B.y + B[2]*B.z, A) == \
           (B[0]*cos(q) - B[1]*sin(q))*A.x + (B[0]*sin(q) + \
           B[1]*cos(q))*A.y + B[2]*A.z
    assert express(B[0]*B.x + B[1]*B.y + B[2]*B.z, A,
                   variables=True).simplify() == A[0]*A.x + A[1]*A.y + A[2]*A.z
    assert express(A[0]*A.x + A[1]*A.y + A[2]*A.z, B) == \
           (A[0]*cos(q) + A[1]*sin(q))*B.x + \
           (-A[0]*sin(q) + A[1]*cos(q))*B.y + A[2]*B.z
    assert express(A[0]*A.x + A[1]*A.y + A[2]*A.z, B,
                   variables=True).simplify() == B[0]*B.x + B[1]*B.y + B[2]*B.z
    N = B.orientnew('N', 'Axis', [-q, B.z])
    assert ({k: v.simplify() for k, v in N.variable_map(A).items()} ==
            {N[0]: A[0], N[2]: A[2], N[1]: A[1]})
    C = A.orientnew('C', 'Axis', [q, A.x + A.y + A.z])
    mapping = A.variable_map(C)
    assert trigsimp(mapping[A[0]]) == (2*C[0]*cos(q)/3 + C[0]/3 -
                                       2*C[1]*sin(q + pi/6)/3 +
                                       C[1]/3 - 2*C[2]*cos(q + pi/3)/3 +
                                       C[2]/3)
    assert trigsimp(mapping[A[1]]) == -2*C[0]*cos(q + pi/3)/3 + \
           C[0]/3 + 2*C[1]*cos(q)/3 + C[1]/3 - 2*C[2]*sin(q + pi/6)/3 + C[2]/3
    assert trigsimp(mapping[A[2]]) == -2*C[0]*sin(q + pi/6)/3 + C[0]/3 - \
           2*C[1]*cos(q + pi/3)/3 + C[1]/3 + 2*C[2]*cos(q)/3 + C[2]/3


def test_ang_vel():
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = A.orientnew('B', 'Axis', [q2, A.x])
    C = B.orientnew('C', 'Axis', [q3, B.y])
    D = N.orientnew('D', 'Axis', [q4, N.y])
    u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    assert A.ang_vel_in(N) == (q1d)*A.z
    assert B.ang_vel_in(N) == (q2d)*B.x + (q1d)*A.z
    assert C.ang_vel_in(N) == (q3d)*C.y + (q2d)*B.x + (q1d)*A.z

    A2 = N.orientnew('A2', 'Axis', [q4, N.y])
    assert N.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == -q1d*N.z
    assert N.ang_vel_in(B) == -q1d*A.z - q2d*B.x
    assert N.ang_vel_in(C) == -q1d*A.z - q2d*B.x - q3d*B.y
    assert N.ang_vel_in(A2) == -q4d*N.y

    assert A.ang_vel_in(N) == q1d*N.z
    assert A.ang_vel_in(A) == 0
    assert A.ang_vel_in(B) == - q2d*B.x
    assert A.ang_vel_in(C) == - q2d*B.x - q3d*B.y
    assert A.ang_vel_in(A2) == q1d*N.z - q4d*N.y

    assert B.ang_vel_in(N) == q1d*A.z + q2d*A.x
    assert B.ang_vel_in(A) == q2d*A.x
    assert B.ang_vel_in(B) == 0
    assert B.ang_vel_in(C) == -q3d*B.y
    assert B.ang_vel_in(A2) == q1d*A.z + q2d*A.x - q4d*N.y

    assert C.ang_vel_in(N) == q1d*A.z + q2d*A.x + q3d*B.y
    assert C.ang_vel_in(A) == q2d*A.x + q3d*C.y
    assert C.ang_vel_in(B) == q3d*B.y
    assert C.ang_vel_in(C) == 0
    assert C.ang_vel_in(A2) == q1d*A.z + q2d*A.x + q3d*B.y - q4d*N.y

    assert A2.ang_vel_in(N) == q4d*A2.y
    assert A2.ang_vel_in(A) == q4d*A2.y - q1d*N.z
    assert A2.ang_vel_in(B) == q4d*N.y - q1d*A.z - q2d*A.x
    assert A2.ang_vel_in(C) == q4d*N.y - q1d*A.z - q2d*A.x - q3d*B.y
    assert A2.ang_vel_in(A2) == 0

    C.set_ang_vel(N, u1*C.x + u2*C.y + u3*C.z)
    assert C.ang_vel_in(N) == (u1)*C.x + (u2)*C.y + (u3)*C.z
    assert N.ang_vel_in(C) == (-u1)*C.x + (-u2)*C.y + (-u3)*C.z
    assert C.ang_vel_in(D) == (u1)*C.x + (u2)*C.y + (u3)*C.z + (-q4d)*D.y
    assert D.ang_vel_in(C) == (-u1)*C.x + (-u2)*C.y + (-u3)*C.z + (q4d)*D.y

    q0 = dynamicsymbols('q0')
    q0d = dynamicsymbols('q0', 1)
    E = N.orientnew('E', 'Quaternion', (q0, q1, q2, q3))
    assert E.ang_vel_in(N) == (
        2 * (q1d * q0 + q2d * q3 - q3d * q2 - q0d * q1) * E.x +
        2 * (q2d * q0 + q3d * q1 - q1d * q3 - q0d * q2) * E.y +
        2 * (q3d * q0 + q1d * q2 - q2d * q1 - q0d * q3) * E.z)

    F = N.orientnew('F', 'Body', (q1, q2, q3), 313)
    assert F.ang_vel_in(N) == ((sin(q2)*sin(q3)*q1d + cos(q3)*q2d)*F.x +
        (sin(q2)*cos(q3)*q1d - sin(q3)*q2d)*F.y + (cos(q2)*q1d + q3d)*F.z)
    G = N.orientnew('G', 'Axis', (q1, N.x + N.y))
    assert G.ang_vel_in(N) == q1d * (N.x + N.y).normalize()
    assert N.ang_vel_in(G) == -q1d * (N.x + N.y).normalize()


def test_dcm():
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = A.orientnew('B', 'Axis', [q2, A.x])
    C = B.orientnew('C', 'Axis', [q3, B.y])
    D = N.orientnew('D', 'Axis', [q4, N.y])
    E = N.orientnew('E', 'Space', [q1, q2, q3], '123')
    assert N.dcm(C) == Matrix([
        [- sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), - sin(q1) *
        cos(q2), sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)], [sin(q1) *
        cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2), sin(q1) *
            sin(q3) - sin(q2) * cos(q1) * cos(q3)], [- sin(q3) * cos(q2), sin(q2),
        cos(q2) * cos(q3)]])
    # This is a little touchy.  Is it ok to use simplify in assert?
    test_mat = D.dcm(C) - Matrix(
        [[cos(q1) * cos(q3) * cos(q4) - sin(q3) * (- sin(q4) * cos(q2) +
        sin(q1) * sin(q2) * cos(q4)), - sin(q2) * sin(q4) - sin(q1) *
            cos(q2) * cos(q4), sin(q3) * cos(q1) * cos(q4) + cos(q3) * (- sin(q4) *
        cos(q2) + sin(q1) * sin(q2) * cos(q4))], [sin(q1) * cos(q3) +
        sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2), sin(q1) * sin(q3) -
            sin(q2) * cos(q1) * cos(q3)], [sin(q4) * cos(q1) * cos(q3) -
        sin(q3) * (cos(q2) * cos(q4) + sin(q1) * sin(q2) * sin(q4)), sin(q2) *
                cos(q4) - sin(q1) * sin(q4) * cos(q2), sin(q3) * sin(q4) * cos(q1) +
                cos(q3) * (cos(q2) * cos(q4) + sin(q1) * sin(q2) * sin(q4))]])
    assert test_mat.expand() == zeros(3, 3)
    assert E.dcm(N) == Matrix(
        [[cos(q2)*cos(q3), sin(q3)*cos(q2), -sin(q2)],
        [sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1), sin(q1)*sin(q2)*sin(q3) +
        cos(q1)*cos(q3), sin(q1)*cos(q2)], [sin(q1)*sin(q3) +
        sin(q2)*cos(q1)*cos(q3), - sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1),
         cos(q1)*cos(q2)]])

def test_w_diff_dcm1():
    # Ref:
    # Dynamics Theory and Applications, Kane 1985
    # Sec. 2.1 ANGULAR VELOCITY
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    c11, c12, c13 = dynamicsymbols('C11 C12 C13')
    c21, c22, c23 = dynamicsymbols('C21 C22 C23')
    c31, c32, c33 = dynamicsymbols('C31 C32 C33')

    c11d, c12d, c13d = dynamicsymbols('C11 C12 C13', level=1)
    c21d, c22d, c23d = dynamicsymbols('C21 C22 C23', level=1)
    c31d, c32d, c33d = dynamicsymbols('C31 C32 C33', level=1)

    DCM = Matrix([
        [c11, c12, c13],
        [c21, c22, c23],
        [c31, c32, c33]
    ])

    B.orient(A, 'DCM', DCM)
    b1a = (B.x).express(A)
    b2a = (B.y).express(A)
    b3a = (B.z).express(A)

    # Equation (2.1.1)
    B.set_ang_vel(A, B.x*(dot((b3a).dt(A), B.y))
                   + B.y*(dot((b1a).dt(A), B.z))
                   + B.z*(dot((b2a).dt(A), B.x)))

    # Equation (2.1.21)
    expr = (  (c12*c13d + c22*c23d + c32*c33d)*B.x
            + (c13*c11d + c23*c21d + c33*c31d)*B.y
            + (c11*c12d + c21*c22d + c31*c32d)*B.z)
    assert B.ang_vel_in(A) - expr == 0

def test_w_diff_dcm2():
    q1, q2, q3 = dynamicsymbols('q1:4')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'axis', [q1, N.x])
    B = A.orientnew('B', 'axis', [q2, A.y])
    C = B.orientnew('C', 'axis', [q3, B.z])

    DCM = C.dcm(N).T
    D = N.orientnew('D', 'DCM', DCM)

    # Frames D and C are the same ReferenceFrame,
    # since they have equal DCM respect to frame N.
    # Therefore, D and C should have same angle velocity in N.
    assert D.dcm(N) == C.dcm(N) == Matrix([
        [cos(q2)*cos(q3), sin(q1)*sin(q2)*cos(q3) +
        sin(q3)*cos(q1), sin(q1)*sin(q3) -
        sin(q2)*cos(q1)*cos(q3)], [-sin(q3)*cos(q2),
        -sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3),
        sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],
        [sin(q2), -sin(q1)*cos(q2), cos(q1)*cos(q2)]])
    assert (D.ang_vel_in(N) - C.ang_vel_in(N)).express(N).simplify() == 0

def test_orientnew_respects_parent_class():
    class MyReferenceFrame(ReferenceFrame):
        pass
    B = MyReferenceFrame('B')
    C = B.orientnew('C', 'Axis', [0, B.x])
    assert isinstance(C, MyReferenceFrame)


def test_orientnew_respects_input_indices():
    N = ReferenceFrame('N')
    q1 = dynamicsymbols('q1')
    A = N.orientnew('a', 'Axis', [q1, N.z])
    #modify default indices:
    minds = [x+'1' for x in N.indices]
    B = N.orientnew('b', 'Axis', [q1, N.z], indices=minds)

    assert N.indices == A.indices
    assert B.indices == minds

def test_orientnew_respects_input_latexs():
    N = ReferenceFrame('N')
    q1 = dynamicsymbols('q1')
    A = N.orientnew('a', 'Axis', [q1, N.z])

    #build default and alternate latex_vecs:
    def_latex_vecs = [(r"\mathbf{\hat{%s}_%s}" % (A.name.lower(),
                      A.indices[0])), (r"\mathbf{\hat{%s}_%s}" %
                      (A.name.lower(), A.indices[1])),
                      (r"\mathbf{\hat{%s}_%s}" % (A.name.lower(),
                      A.indices[2]))]

    name = 'b'
    indices = [x+'1' for x in N.indices]
    new_latex_vecs = [(r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                      indices[0])), (r"\mathbf{\hat{%s}_{%s}}" %
                      (name.lower(), indices[1])),
                      (r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                      indices[2]))]

    B = N.orientnew(name, 'Axis', [q1, N.z], latexs=new_latex_vecs)

    assert A.latex_vecs == def_latex_vecs
    assert B.latex_vecs == new_latex_vecs
    assert B.indices != indices

def test_orientnew_respects_input_variables():
    N = ReferenceFrame('N')
    q1 = dynamicsymbols('q1')
    A = N.orientnew('a', 'Axis', [q1, N.z])

    #build non-standard variable names
    name = 'b'
    new_variables = ['notb_'+x+'1' for x in N.indices]
    B = N.orientnew(name, 'Axis', [q1, N.z], variables=new_variables)

    for j,var in enumerate(A.varlist):
        assert var.name == A.name + '_' + A.indices[j]

    for j,var in enumerate(B.varlist):
        assert var.name == new_variables[j]

def test_issue_10348():
    u = dynamicsymbols('u:3')
    I = ReferenceFrame('I')
    I.orientnew('A', 'space', u, 'XYZ')


def test_issue_11503():
    A = ReferenceFrame("A")
    A.orientnew("B", "Axis", [35, A.y])
    C = ReferenceFrame("C")
    A.orient(C, "Axis", [70, C.z])


def test_partial_velocity():

    N = ReferenceFrame('N')
    A = ReferenceFrame('A')

    u1, u2 = dynamicsymbols('u1, u2')

    A.set_ang_vel(N, u1 * A.x + u2 * N.y)

    assert N.partial_velocity(A, u1) == -A.x
    assert N.partial_velocity(A, u1, u2) == (-A.x, -N.y)

    assert A.partial_velocity(N, u1) == A.x
    assert A.partial_velocity(N, u1, u2) == (A.x, N.y)

    assert N.partial_velocity(N, u1) == 0
    assert A.partial_velocity(A, u1) == 0


def test_issue_11498():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    # Identity transformation
    A.orient(B, 'DCM', eye(3))
    assert A.dcm(B) == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert B.dcm(A) == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # x -> y
    # y -> -z
    # z -> -x
    A.orient(B, 'DCM', Matrix([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
    assert B.dcm(A) == Matrix([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    assert A.dcm(B) == Matrix([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    assert B.dcm(A).T == A.dcm(B)


def test_reference_frame():
    raises(TypeError, lambda: ReferenceFrame(0))
    raises(TypeError, lambda: ReferenceFrame('N', 0))
    raises(ValueError, lambda: ReferenceFrame('N', [0, 1]))
    raises(TypeError, lambda: ReferenceFrame('N', [0, 1, 2]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], 0))
    raises(ValueError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], [0, 1]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], [0, 1, 2]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'],
                                                 ['a', 'b', 'c'], 0))
    raises(ValueError, lambda: ReferenceFrame('N', ['a', 'b', 'c'],
                                              ['a', 'b', 'c'], [0, 1]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'],
                                             ['a', 'b', 'c'], [0, 1, 2]))
    N = ReferenceFrame('N')
    assert N[0] == CoordinateSym('N_x', N, 0)
    assert N[1] == CoordinateSym('N_y', N, 1)
    assert N[2] == CoordinateSym('N_z', N, 2)
    raises(ValueError, lambda: N[3])
    N = ReferenceFrame('N', ['a', 'b', 'c'])
    assert N['a'] == N.x
    assert N['b'] == N.y
    assert N['c'] == N.z
    raises(ValueError, lambda: N['d'])
    assert str(N) == 'N'

    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
    raises(TypeError, lambda: A.orient(B, 'DCM', 0))
    raises(TypeError, lambda: B.orient(N, 'Space', [q1, q2, q3], '222'))
    raises(TypeError, lambda: B.orient(N, 'Axis', [q1, N.x + 2 * N.y], '222'))
    raises(TypeError, lambda: B.orient(N, 'Axis', q1))
    raises(IndexError, lambda: B.orient(N, 'Axis', [q1]))
    raises(TypeError, lambda: B.orient(N, 'Quaternion', [q0, q1, q2, q3], '222'))
    raises(TypeError, lambda: B.orient(N, 'Quaternion', q0))
    raises(TypeError, lambda: B.orient(N, 'Quaternion', [q0, q1, q2]))
    raises(NotImplementedError, lambda: B.orient(N, 'Foo', [q0, q1, q2]))
    raises(TypeError, lambda: B.orient(N, 'Body', [q1, q2], '232'))
    raises(TypeError, lambda: B.orient(N, 'Space', [q1, q2], '232'))

    N.set_ang_acc(B, 0)
    assert N.ang_acc_in(B) == Vector(0)
    N.set_ang_vel(B, 0)
    assert N.ang_vel_in(B) == Vector(0)


def test_check_frame():
    raises(VectorTypeError, lambda: _check_frame(0))


def test_dcm_diff_16824():
    # NOTE : This is a regression test for the bug introduced in PR 14758,
    # identified in 16824, and solved by PR 16828.

    # This is the solution to Problem 2.2 on page 264 in Kane & Lenvinson's
    # 1985 book.

    q1, q2, q3 = dynamicsymbols('q1:4')

    s1 = sin(q1)
    c1 = cos(q1)
    s2 = sin(q2)
    c2 = cos(q2)
    s3 = sin(q3)
    c3 = cos(q3)

    dcm = Matrix([[c2*c3, s1*s2*c3 - s3*c1, c1*s2*c3 + s3*s1],
                  [c2*s3, s1*s2*s3 + c3*c1, c1*s2*s3 - c3*s1],
                  [-s2,   s1*c2,            c1*c2]])

    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient(A, 'DCM', dcm)

    AwB = B.ang_vel_in(A)

    alpha2 = s3*c2*q1.diff() + c3*q2.diff()
    beta2 = s1*c2*q3.diff() + c1*q2.diff()

    assert simplify(AwB.dot(A.y) - alpha2) == 0
    assert simplify(AwB.dot(B.y) - beta2) == 0

def test_orient_explicit():
    cxx, cyy, czz = dynamicsymbols('c_{xx}, c_{yy}, c_{zz}')
    cxy, cxz, cyx = dynamicsymbols('c_{xy}, c_{xz}, c_{yx}')
    cyz, czx, czy = dynamicsymbols('c_{yz}, c_{zx}, c_{zy}')
    dcxx, dcyy, dczz = dynamicsymbols('c_{xx}, c_{yy}, c_{zz}', 1)
    dcxy, dcxz, dcyx = dynamicsymbols('c_{xy}, c_{xz}, c_{yx}', 1)
    dcyz, dczx, dczy = dynamicsymbols('c_{yz}, c_{zx}, c_{zy}', 1)
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B_C_A = Matrix([[cxx, cxy, cxz],
                    [cyx, cyy, cyz],
                    [czx, czy, czz]])
    B_w_A = ((cyx*dczx + cyy*dczy + cyz*dczz)*B.x +
            (czx*dcxx + czy*dcxy + czz*dcxz)*B.y +
            (cxx*dcyx + cxy*dcyy + cxz*dcyz)*B.z)
    A.orient_explicit(B, B_C_A)
    assert B.dcm(A) == B_C_A
    assert A.ang_vel_in(B) == B_w_A
    assert B.ang_vel_in(A) == -B_w_A

def test_orient_dcm():
    cxx, cyy, czz = dynamicsymbols('c_{xx}, c_{yy}, c_{zz}')
    cxy, cxz, cyx = dynamicsymbols('c_{xy}, c_{xz}, c_{yx}')
    cyz, czx, czy = dynamicsymbols('c_{yz}, c_{zx}, c_{zy}')
    B_C_A = Matrix([[cxx, cxy, cxz],
                    [cyx, cyy, cyz],
                    [czx, czy, czz]])
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_dcm(A, B_C_A)
    assert B.dcm(A) == Matrix([[cxx, cxy, cxz],
                               [cyx, cyy, cyz],
                               [czx, czy, czz]])

def test_orient_axis():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    A.orient_axis(B,-B.x, 1)
    A1 = A.dcm(B)
    A.orient_axis(B, B.x, -1)
    A2 = A.dcm(B)
    A.orient_axis(B, 1, -B.x)
    A3 = A.dcm(B)
    assert A1 == A2
    assert A2 == A3
    raises(TypeError, lambda: A.orient_axis(B, 1, 1))

def test_orient_body():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_body_fixed(A, (1,1,0), 'XYX')
    assert B.dcm(A) == Matrix([[cos(1), sin(1)**2, -sin(1)*cos(1)], [0, cos(1), sin(1)], [sin(1), -sin(1)*cos(1), cos(1)**2]])


def test_orient_body_advanced():
    q1, q2, q3 = dynamicsymbols('q1:4')
    c1, c2, c3 = symbols('c1:4')
    u1, u2, u3 = dynamicsymbols('q1:4', 1)

    # Test with everything as dynamicsymbols
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_body_fixed(A, (q1, q2, q3), 'zxy')
    assert A.dcm(B) == Matrix([
        [-sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), -sin(q1) * cos(q2),
         sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)],
        [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2),
         sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)],
        [-sin(q3) * cos(q2), sin(q2), cos(q2) * cos(q3)]])
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [-sin(q3) * cos(q2) * u1 + cos(q3) * u2],
        [sin(q2) * u1 + u3],
        [sin(q3) * u2 + cos(q2) * cos(q3) * u1]])

    # Test with constant symbol
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_body_fixed(A, (q1, c2, q3), 131)
    assert A.dcm(B) == Matrix([
        [cos(c2), -sin(c2) * cos(q3), sin(c2) * sin(q3)],
        [sin(c2) * cos(q1), -sin(q1) * sin(q3) + cos(c2) * cos(q1) * cos(q3),
         -sin(q1) * cos(q3) - sin(q3) * cos(c2) * cos(q1)],
        [sin(c2) * sin(q1), sin(q1) * cos(c2) * cos(q3) + sin(q3) * cos(q1),
         -sin(q1) * sin(q3) * cos(c2) + cos(q1) * cos(q3)]])
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [cos(c2) * u1 + u3],
        [-sin(c2) * cos(q3) * u1],
        [sin(c2) * sin(q3) * u1]])

    # Test all symbols not time dependent
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_body_fixed(A, (c1, c2, c3), 123)
    assert B.ang_vel_in(A) == Vector(0)


def test_orient_space_advanced():
    # space fixed is in the end like body fixed only in opposite order
    q1, q2, q3 = dynamicsymbols('q1:4')
    c1, c2, c3 = symbols('c1:4')
    u1, u2, u3 = dynamicsymbols('q1:4', 1)

    # Test with everything as dynamicsymbols
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_space_fixed(A, (q3, q2, q1), 'yxz')
    assert A.dcm(B) == Matrix([
        [-sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), -sin(q1) * cos(q2),
         sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)],
        [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2),
         sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)],
        [-sin(q3) * cos(q2), sin(q2), cos(q2) * cos(q3)]])
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [-sin(q3) * cos(q2) * u1 + cos(q3) * u2],
        [sin(q2) * u1 + u3],
        [sin(q3) * u2 + cos(q2) * cos(q3) * u1]])

    # Test with constant symbol
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_space_fixed(A, (q3, c2, q1), 131)
    assert A.dcm(B) == Matrix([
        [cos(c2), -sin(c2) * cos(q3), sin(c2) * sin(q3)],
        [sin(c2) * cos(q1), -sin(q1) * sin(q3) + cos(c2) * cos(q1) * cos(q3),
         -sin(q1) * cos(q3) - sin(q3) * cos(c2) * cos(q1)],
        [sin(c2) * sin(q1), sin(q1) * cos(c2) * cos(q3) + sin(q3) * cos(q1),
         -sin(q1) * sin(q3) * cos(c2) + cos(q1) * cos(q3)]])
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [cos(c2) * u1 + u3],
        [-sin(c2) * cos(q3) * u1],
        [sin(c2) * sin(q3) * u1]])

    # Test all symbols not time dependent
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_space_fixed(A, (c1, c2, c3), 123)
    assert B.ang_vel_in(A) == Vector(0)


def test_orient_body_simple_ang_vel():
    """This test ensures that the simplest form of that linear system solution
    is returned, thus the == for the expression comparison."""

    psi, theta, phi = dynamicsymbols('psi, theta, varphi')
    t = dynamicsymbols._t
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_body_fixed(A, (psi, theta, phi), 'ZXZ')
    A_w_B = B.ang_vel_in(A)
    assert A_w_B.args[0][1] == B
    assert A_w_B.args[0][0][0] == (sin(theta)*sin(phi)*psi.diff(t) +
                                   cos(phi)*theta.diff(t))
    assert A_w_B.args[0][0][1] == (sin(theta)*cos(phi)*psi.diff(t) -
                                   sin(phi)*theta.diff(t))
    assert A_w_B.args[0][0][2] == cos(theta)*psi.diff(t) + phi.diff(t)


def test_orient_space():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_space_fixed(A, (0,0,0), '123')
    assert B.dcm(A) == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def test_orient_quaternion():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_quaternion(A, (0,0,0,0))
    assert B.dcm(A) == Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

def test_looped_frame_warning():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')

    a, b, c = symbols('a b c')
    B.orient_axis(A, A.x, a)
    C.orient_axis(B, B.x, b)

    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        A.orient_axis(C, C.x, c)
        assert issubclass(w[-1].category, UserWarning)
        assert 'Loops are defined among the orientation of frames. ' + \
            'This is likely not desired and may cause errors in your calculations.' in str(w[-1].message)

def test_frame_dict():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')

    a, b, c = symbols('a b c')

    B.orient_axis(A, A.x, a)
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(a), -sin(a)],[0, sin(a),  cos(a)]])}
    assert B._dcm_dict == {A: Matrix([[1, 0, 0],[0,  cos(a), sin(a)],[0, -sin(a), cos(a)]])}
    assert C._dcm_dict == {}

    B.orient_axis(C, C.x, b)
    # Previous relation is not wiped
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(a), -sin(a)],[0, sin(a),  cos(a)]])}
    assert B._dcm_dict == {A: Matrix([[1, 0, 0],[0,  cos(a), sin(a)],[0, -sin(a), cos(a)]]), \
        C: Matrix([[1, 0, 0],[0,  cos(b), sin(b)],[0, -sin(b), cos(b)]])}
    assert C._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b),  cos(b)]])}

    A.orient_axis(B, B.x, c)
    # Previous relation is updated
    assert B._dcm_dict == {C: Matrix([[1, 0, 0],[0,  cos(b), sin(b)],[0, -sin(b), cos(b)]]),\
        A: Matrix([[1, 0, 0],[0, cos(c), -sin(c)],[0, sin(c),  cos(c)]])}
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0,  cos(c), sin(c)],[0, -sin(c), cos(c)]])}
    assert C._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b),  cos(b)]])}

def test_dcm_cache_dict():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    D = ReferenceFrame('D')

    a, b, c = symbols('a b c')

    B.orient_axis(A, A.x, a)
    C.orient_axis(B, B.x, b)
    D.orient_axis(C, C.x, c)

    assert D._dcm_dict == {C: Matrix([[1, 0, 0],[0,  cos(c), sin(c)],[0, -sin(c), cos(c)]])}
    assert C._dcm_dict == {B: Matrix([[1, 0, 0],[0,  cos(b), sin(b)],[0, -sin(b), cos(b)]]), \
        D: Matrix([[1, 0, 0],[0, cos(c), -sin(c)],[0, sin(c),  cos(c)]])}
    assert B._dcm_dict == {A: Matrix([[1, 0, 0],[0,  cos(a), sin(a)],[0, -sin(a), cos(a)]]), \
        C: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b),  cos(b)]])}
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(a), -sin(a)],[0, sin(a),  cos(a)]])}

    assert D._dcm_dict == D._dcm_cache

    D.dcm(A) # Check calculated dcm relation is stored in _dcm_cache and not in _dcm_dict
    assert list(A._dcm_cache.keys()) == [A, B, D]
    assert list(D._dcm_cache.keys()) == [C, A]
    assert list(A._dcm_dict.keys()) == [B]
    assert list(D._dcm_dict.keys()) == [C]
    assert A._dcm_dict != A._dcm_cache

    A.orient_axis(B, B.x, b) # _dcm_cache of A is wiped out and new relation is stored.
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0,  cos(b), sin(b)],[0, -sin(b), cos(b)]])}
    assert A._dcm_dict == A._dcm_cache
    assert B._dcm_dict == {C: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b),  cos(b)]]), \
        A: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b),  cos(b)]])}

def test_xx_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.xx == Vector.outer(N.x, N.x)
    assert F.xx == Vector.outer(F.x, F.x)

def test_xy_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.xy == Vector.outer(N.x, N.y)
    assert F.xy == Vector.outer(F.x, F.y)

def test_xz_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.xz == Vector.outer(N.x, N.z)
    assert F.xz == Vector.outer(F.x, F.z)

def test_yx_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.yx == Vector.outer(N.y, N.x)
    assert F.yx == Vector.outer(F.y, F.x)

def test_yy_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.yy == Vector.outer(N.y, N.y)
    assert F.yy == Vector.outer(F.y, F.y)

def test_yz_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.yz == Vector.outer(N.y, N.z)
    assert F.yz == Vector.outer(F.y, F.z)

def test_zx_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.zx == Vector.outer(N.z, N.x)
    assert F.zx == Vector.outer(F.z, F.x)

def test_zy_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.zy == Vector.outer(N.z, N.y)
    assert F.zy == Vector.outer(F.z, F.y)

def test_zz_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.zz == Vector.outer(N.z, N.z)
    assert F.zz == Vector.outer(F.z, F.z)

def test_unit_dyadic():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.u == N.xx + N.yy + N.zz
    assert F.u == F.xx + F.yy + F.zz


def test_pickle_frame():
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    A.orient_axis(N, N.x, 1)
    A_C_N = A.dcm(N)
    N1 = pickle.loads(pickle.dumps(N))
    A1 = tuple(N1._dcm_dict.keys())[0]
    assert A1.dcm(N1) == A_C_N

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\tests\test_cse.py ===
from functools import reduce
import itertools
from operator import add

from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import Inverse, MatAdd, MatMul, Transpose
from sympy.polys.rootoftools import CRootOf
from sympy.series.order import O
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import signsimp
from sympy.tensor.indexed import (Idx, IndexedBase)

from sympy.core.function import count_ops
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.functions.special.hyper import meijerg
from sympy.simplify import cse_main, cse_opts
from sympy.utilities.iterables import subsets
from sympy.testing.pytest import XFAIL, raises
from sympy.matrices import (MutableDenseMatrix, MutableSparseMatrix,
        ImmutableDenseMatrix, ImmutableSparseMatrix)
from sympy.matrices.expressions import MatrixSymbol


w, x, y, z = symbols('w,x,y,z')
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = symbols('x:13')


def test_numbered_symbols():
    ns = cse_main.numbered_symbols(prefix='y')
    assert list(itertools.islice(
        ns, 0, 10)) == [Symbol('y%s' % i) for i in range(0, 10)]
    ns = cse_main.numbered_symbols(prefix='y')
    assert list(itertools.islice(
        ns, 10, 20)) == [Symbol('y%s' % i) for i in range(10, 20)]
    ns = cse_main.numbered_symbols()
    assert list(itertools.islice(
        ns, 0, 10)) == [Symbol('x%s' % i) for i in range(0, 10)]

# Dummy "optimization" functions for testing.


def opt1(expr):
    return expr + y


def opt2(expr):
    return expr*z


def test_preprocess_for_cse():
    assert cse_main.preprocess_for_cse(x, [(opt1, None)]) == x + y
    assert cse_main.preprocess_for_cse(x, [(None, opt1)]) == x
    assert cse_main.preprocess_for_cse(x, [(None, None)]) == x
    assert cse_main.preprocess_for_cse(x, [(opt1, opt2)]) == x + y
    assert cse_main.preprocess_for_cse(
        x, [(opt1, None), (opt2, None)]) == (x + y)*z


def test_postprocess_for_cse():
    assert cse_main.postprocess_for_cse(x, [(opt1, None)]) == x
    assert cse_main.postprocess_for_cse(x, [(None, opt1)]) == x + y
    assert cse_main.postprocess_for_cse(x, [(None, None)]) == x
    assert cse_main.postprocess_for_cse(x, [(opt1, opt2)]) == x*z
    # Note the reverse order of application.
    assert cse_main.postprocess_for_cse(
        x, [(None, opt1), (None, opt2)]) == x*z + y


def test_cse_single():
    # Simple substitution.
    e = Add(Pow(x + y, 2), sqrt(x + y))
    substs, reduced = cse([e])
    assert substs == [(x0, x + y)]
    assert reduced == [sqrt(x0) + x0**2]

    subst42, (red42,) = cse([42])  # issue_15082
    assert len(subst42) == 0 and red42 == 42
    subst_half, (red_half,) = cse([0.5])
    assert len(subst_half) == 0 and red_half == 0.5


def test_cse_single2():
    # Simple substitution, test for being able to pass the expression directly
    e = Add(Pow(x + y, 2), sqrt(x + y))
    substs, reduced = cse(e)
    assert substs == [(x0, x + y)]
    assert reduced == [sqrt(x0) + x0**2]
    substs, reduced = cse(Matrix([[1]]))
    assert isinstance(reduced[0], Matrix)

    subst42, (red42,) = cse(42)  # issue 15082
    assert len(subst42) == 0 and red42 == 42
    subst_half, (red_half,) = cse(0.5)  # issue 15082
    assert len(subst_half) == 0 and red_half == 0.5


def test_cse_not_possible():
    # No substitution possible.
    e = Add(x, y)
    substs, reduced = cse([e])
    assert substs == []
    assert reduced == [x + y]
    # issue 6329
    eq = (meijerg((1, 2), (y, 4), (5,), [], x) +
          meijerg((1, 3), (y, 4), (5,), [], x))
    assert cse(eq) == ([], [eq])


def test_nested_substitution():
    # Substitution within a substitution.
    e = Add(Pow(w*x + y, 2), sqrt(w*x + y))
    substs, reduced = cse([e])
    assert substs == [(x0, w*x + y)]
    assert reduced == [sqrt(x0) + x0**2]


def test_subtraction_opt():
    # Make sure subtraction is optimized.
    e = (x - y)*(z - y) + exp((x - y)*(z - y))
    substs, reduced = cse(
        [e], optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])
    assert substs == [(x0, (x - y)*(y - z))]
    assert reduced == [-x0 + exp(-x0)]
    e = -(x - y)*(z - y) + exp(-(x - y)*(z - y))
    substs, reduced = cse(
        [e], optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])
    assert substs == [(x0, (x - y)*(y - z))]
    assert reduced == [x0 + exp(x0)]
    # issue 4077
    n = -1 + 1/x
    e = n/x/(-n)**2 - 1/n/x
    assert cse(e, optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)]) == \
        ([], [0])
    assert cse(((w + x + y + z)*(w - y - z))/(w + x)**3) == \
        ([(x0, w + x), (x1, y + z)], [(w - x1)*(x0 + x1)/x0**3])


def test_multiple_expressions():
    e1 = (x + y)*z
    e2 = (x + y)*w
    substs, reduced = cse([e1, e2])
    assert substs == [(x0, x + y)]
    assert reduced == [x0*z, x0*w]
    l = [w*x*y + z, w*y]
    substs, reduced = cse(l)
    rsubsts, _ = cse(reversed(l))
    assert substs == rsubsts
    assert reduced == [z + x*x0, x0]
    l = [w*x*y, w*x*y + z, w*y]
    substs, reduced = cse(l)
    rsubsts, _ = cse(reversed(l))
    assert substs == rsubsts
    assert reduced == [x1, x1 + z, x0]
    l = [(x - z)*(y - z), x - z, y - z]
    substs, reduced = cse(l)
    rsubsts, _ = cse(reversed(l))
    assert substs == [(x0, -z), (x1, x + x0), (x2, x0 + y)]
    assert rsubsts == [(x0, -z), (x1, x0 + y), (x2, x + x0)]
    assert reduced == [x1*x2, x1, x2]
    l = [w*y + w + x + y + z, w*x*y]
    assert cse(l) == ([(x0, w*y)], [w + x + x0 + y + z, x*x0])
    assert cse([x + y, x + y + z]) == ([(x0, x + y)], [x0, z + x0])
    assert cse([x + y, x + z]) == ([], [x + y, x + z])
    assert cse([x*y, z + x*y, x*y*z + 3]) == \
        ([(x0, x*y)], [x0, z + x0, 3 + x0*z])


@XFAIL # CSE of non-commutative Mul terms is disabled
def test_non_commutative_cse():
    A, B, C = symbols('A B C', commutative=False)
    l = [A*B*C, A*C]
    assert cse(l) == ([], l)
    l = [A*B*C, A*B]
    assert cse(l) == ([(x0, A*B)], [x0*C, x0])


# Test if CSE of non-commutative Mul terms is disabled
def test_bypass_non_commutatives():
    A, B, C = symbols('A B C', commutative=False)
    l = [A*B*C, A*C]
    assert cse(l) == ([], l)
    l = [A*B*C, A*B]
    assert cse(l) == ([], l)
    l = [B*C, A*B*C]
    assert cse(l) == ([], l)


@XFAIL # CSE fails when replacing non-commutative sub-expressions
def test_non_commutative_order():
    A, B, C = symbols('A B C', commutative=False)
    x0 = symbols('x0', commutative=False)
    l = [B+C, A*(B+C)]
    assert cse(l) == ([(x0, B+C)], [x0, A*x0])


@XFAIL # Worked in gh-11232, but was reverted due to performance considerations
def test_issue_10228():
    assert cse([x*y**2 + x*y]) == ([(x0, x*y)], [x0*y + x0])
    assert cse([x + y, 2*x + y]) == ([(x0, x + y)], [x0, x + x0])
    assert cse((w + 2*x + y + z, w + x + 1)) == (
        [(x0, w + x)], [x0 + x + y + z, x0 + 1])
    assert cse(((w + x + y + z)*(w - x))/(w + x)) == (
        [(x0, w + x)], [(x0 + y + z)*(w - x)/x0])
    a, b, c, d, f, g, j, m = symbols('a, b, c, d, f, g, j, m')
    exprs = (d*g**2*j*m, 4*a*f*g*m, a*b*c*f**2)
    assert cse(exprs) == (
        [(x0, g*m), (x1, a*f)], [d*g*j*x0, 4*x0*x1, b*c*f*x1]
)

@XFAIL
def test_powers():
    assert cse(x*y**2 + x*y) == ([(x0, x*y)], [x0*y + x0])


def test_issue_4498():
    assert cse(w/(x - y) + z/(y - x), optimizations='basic') == \
        ([], [(w - z)/(x - y)])


def test_issue_4020():
    assert cse(x**5 + x**4 + x**3 + x**2, optimizations='basic') \
        == ([(x0, x**2)], [x0*(x**3 + x + x0 + 1)])


def test_issue_4203():
    assert cse(sin(x**x)/x**x) == ([(x0, x**x)], [sin(x0)/x0])


def test_issue_6263():
    e = Eq(x*(-x + 1) + x*(x - 1), 0)
    assert cse(e, optimizations='basic') == ([], [True])


def test_issue_25043():
    c = symbols("c")
    x = symbols("x0", real=True)
    cse_expr = cse(c*x**2 + c*(x**4 - x**2))[-1][-1]
    free = cse_expr.free_symbols
    assert len(free) == len({i.name for i in free})


def test_dont_cse_tuples():
    from sympy.core.function import Subs
    f = Function("f")
    g = Function("g")

    name_val, (expr,) = cse(
        Subs(f(x, y), (x, y), (0, 1))
        + Subs(g(x, y), (x, y), (0, 1)))

    assert name_val == []
    assert expr == (Subs(f(x, y), (x, y), (0, 1))
            + Subs(g(x, y), (x, y), (0, 1)))

    name_val, (expr,) = cse(
        Subs(f(x, y), (x, y), (0, x + y))
        + Subs(g(x, y), (x, y), (0, x + y)))

    assert name_val == [(x0, x + y)]
    assert expr == Subs(f(x, y), (x, y), (0, x0)) + \
        Subs(g(x, y), (x, y), (0, x0))


def test_pow_invpow():
    assert cse(1/x**2 + x**2) == \
        ([(x0, x**2)], [x0 + 1/x0])
    assert cse(x**2 + (1 + 1/x**2)/x**2) == \
        ([(x0, x**2), (x1, 1/x0)], [x0 + x1*(x1 + 1)])
    assert cse(1/x**2 + (1 + 1/x**2)*x**2) == \
        ([(x0, x**2), (x1, 1/x0)], [x0*(x1 + 1) + x1])
    assert cse(cos(1/x**2) + sin(1/x**2)) == \
        ([(x0, x**(-2))], [sin(x0) + cos(x0)])
    assert cse(cos(x**2) + sin(x**2)) == \
        ([(x0, x**2)], [sin(x0) + cos(x0)])
    assert cse(y/(2 + x**2) + z/x**2/y) == \
        ([(x0, x**2)], [y/(x0 + 2) + z/(x0*y)])
    assert cse(exp(x**2) + x**2*cos(1/x**2)) == \
        ([(x0, x**2)], [x0*cos(1/x0) + exp(x0)])
    assert cse((1 + 1/x**2)/x**2) == \
        ([(x0, x**(-2))], [x0*(x0 + 1)])
    assert cse(x**(2*y) + x**(-2*y)) == \
        ([(x0, x**(2*y))], [x0 + 1/x0])


def test_postprocess():
    eq = (x + 1 + exp((x + 1)/(y + 1)) + cos(y + 1))
    assert cse([eq, Eq(x, z + 1), z - 2, (z + 1)*(x + 1)],
        postprocess=cse_main.cse_separate) == \
        [[(x0, y + 1), (x2, z + 1), (x, x2), (x1, x + 1)],
        [x1 + exp(x1/x0) + cos(x0), z - 2, x1*x2]]


def test_issue_4499():
    # previously, this gave 16 constants
    from sympy.abc import a, b
    B = Function('B')
    G = Function('G')
    t = Tuple(*
        (a, a + S.Half, 2*a, b, 2*a - b + 1, (sqrt(z)/2)**(-2*a + 1)*B(2*a -
        b, sqrt(z))*B(b - 1, sqrt(z))*G(b)*G(2*a - b + 1),
        sqrt(z)*(sqrt(z)/2)**(-2*a + 1)*B(b, sqrt(z))*B(2*a - b,
        sqrt(z))*G(b)*G(2*a - b + 1), sqrt(z)*(sqrt(z)/2)**(-2*a + 1)*B(b - 1,
        sqrt(z))*B(2*a - b + 1, sqrt(z))*G(b)*G(2*a - b + 1),
        (sqrt(z)/2)**(-2*a + 1)*B(b, sqrt(z))*B(2*a - b + 1,
        sqrt(z))*G(b)*G(2*a - b + 1), 1, 0, S.Half, z/2, -b + 1, -2*a + b,
        -2*a))
    c = cse(t)
    ans = (
        [(x0, 2*a), (x1, -b + x0), (x2, x1 + 1), (x3, b - 1), (x4, sqrt(z)),
         (x5, B(x3, x4)), (x6, (x4/2)**(1 - x0)*G(b)*G(x2)), (x7, x6*B(x1, x4)),
         (x8, B(b, x4)), (x9, x6*B(x2, x4))],
        [(a, a + S.Half, x0, b, x2, x5*x7, x4*x7*x8, x4*x5*x9, x8*x9,
          1, 0, S.Half, z/2, -x3, -x1, -x0)])
    assert ans == c


def test_issue_6169():
    r = CRootOf(x**6 - 4*x**5 - 2, 1)
    assert cse(r) == ([], [r])
    # and a check that the right thing is done with the new
    # mechanism
    assert sub_post(sub_pre((-x - y)*z - x - y)) == -z*(x + y) - x - y


def test_cse_Indexed():
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    i = Idx('i', len_y-1)

    expr1 = (y[i+1]-y[i])/(x[i+1]-x[i])
    expr2 = 1/(x[i+1]-x[i])
    replacements, reduced_exprs = cse([expr1, expr2])
    assert len(replacements) > 0


def test_cse_MatrixSymbol():
    # MatrixSymbols have non-Basic args, so make sure that works
    A = MatrixSymbol("A", 3, 3)
    assert cse(A) == ([], [A])

    n = symbols('n', integer=True)
    B = MatrixSymbol("B", n, n)
    assert cse(B) == ([], [B])

    assert cse(A[0] * A[0]) == ([], [A[0]*A[0]])

    assert cse(A[0,0]*A[0,1] + A[0,0]*A[0,1]*A[0,2]) == ([(x0, A[0, 0]*A[0, 1])], [x0*A[0, 2] + x0])

def test_cse_MatrixExpr():
    A = MatrixSymbol('A', 3, 3)
    y = MatrixSymbol('y', 3, 1)

    expr1 = (A.T*A).I * A * y
    expr2 = (A.T*A) * A * y
    replacements, reduced_exprs = cse([expr1, expr2])
    assert len(replacements) > 0

    replacements, reduced_exprs = cse([expr1 + expr2, expr1])
    assert replacements

    replacements, reduced_exprs = cse([A**2, A + A**2])
    assert replacements


def test_Piecewise():
    f = Piecewise((-z + x*y, Eq(y, 0)), (-z - x*y, True))
    ans = cse(f)
    actual_ans = ([(x0, x*y)],
        [Piecewise((x0 - z, Eq(y, 0)), (-z - x0, True))])
    assert ans == actual_ans


def test_ignore_order_terms():
    eq = exp(x).series(x,0,3) + sin(y+x**3) - 1
    assert cse(eq) == ([], [sin(x**3 + y) + x + x**2/2 + O(x**3)])


def test_name_conflict():
    z1 = x0 + y
    z2 = x2 + x3
    l = [cos(z1) + z1, cos(z2) + z2, x0 + x2]
    substs, reduced = cse(l)
    assert [e.subs(reversed(substs)) for e in reduced] == l


def test_name_conflict_cust_symbols():
    z1 = x0 + y
    z2 = x2 + x3
    l = [cos(z1) + z1, cos(z2) + z2, x0 + x2]
    substs, reduced = cse(l, symbols("x:10"))
    assert [e.subs(reversed(substs)) for e in reduced] == l


def test_symbols_exhausted_error():
    l = cos(x+y)+x+y+cos(w+y)+sin(w+y)
    sym = [x, y, z]
    with raises(ValueError):
        cse(l, symbols=sym)


def test_issue_7840():
    # daveknippers' example
    C393 = sympify( \
        'Piecewise((C391 - 1.65, C390 < 0.5), (Piecewise((C391 - 1.65, \
        C391 > 2.35), (C392, True)), True))'
    )
    C391 = sympify( \
        'Piecewise((2.05*C390**(-1.03), C390 < 0.5), (2.5*C390**(-0.625), True))'
    )
    C393 = C393.subs('C391',C391)
    # simple substitution
    sub = {}
    sub['C390'] = 0.703451854
    sub['C392'] = 1.01417794
    ss_answer = C393.subs(sub)
    # cse
    substitutions,new_eqn = cse(C393)
    for pair in substitutions:
        sub[pair[0].name] = pair[1].subs(sub)
    cse_answer = new_eqn[0].subs(sub)
    # both methods should be the same
    assert ss_answer == cse_answer

    # GitRay's example
    expr = sympify(
        "Piecewise((Symbol('ON'), Equality(Symbol('mode'), Symbol('ON'))), \
        (Piecewise((Piecewise((Symbol('OFF'), StrictLessThan(Symbol('x'), \
        Symbol('threshold'))), (Symbol('ON'), true)), Equality(Symbol('mode'), \
        Symbol('AUTO'))), (Symbol('OFF'), true)), true))"
    )
    substitutions, new_eqn = cse(expr)
    # this Piecewise should be exactly the same
    assert new_eqn[0] == expr
    # there should not be any replacements
    assert len(substitutions) < 1


def test_issue_8891():
    for cls in (MutableDenseMatrix, MutableSparseMatrix,
            ImmutableDenseMatrix, ImmutableSparseMatrix):
        m = cls(2, 2, [x + y, 0, 0, 0])
        res = cse([x + y, m])
        ans = ([(x0, x + y)], [x0, cls([[x0, 0], [0, 0]])])
        assert res == ans
        assert isinstance(res[1][-1], cls)


def test_issue_11230():
    # a specific test that always failed
    a, b, f, k, l, i = symbols('a b f k l i')
    p = [a*b*f*k*l, a*i*k**2*l, f*i*k**2*l]
    R, C = cse(p)
    assert not any(i.is_Mul for a in C for i in a.args)

    # random tests for the issue
    from sympy.core.random import choice
    from sympy.core.function import expand_mul
    s = symbols('a:m')
    # 35 Mul tests, none of which should ever fail
    ex = [Mul(*[choice(s) for i in range(5)]) for i in range(7)]
    for p in subsets(ex, 3):
        p = list(p)
        R, C = cse(p)
        assert not any(i.is_Mul for a in C for i in a.args)
        for ri in reversed(R):
            for i in range(len(C)):
                C[i] = C[i].subs(*ri)
        assert p == C
    # 35 Add tests, none of which should ever fail
    ex = [Add(*[choice(s[:7]) for i in range(5)]) for i in range(7)]
    for p in subsets(ex, 3):
        p = list(p)
        R, C = cse(p)
        assert not any(i.is_Add for a in C for i in a.args)
        for ri in reversed(R):
            for i in range(len(C)):
                C[i] = C[i].subs(*ri)
        # use expand_mul to handle cases like this:
        # p = [a + 2*b + 2*e, 2*b + c + 2*e, b + 2*c + 2*g]
        # x0 = 2*(b + e) is identified giving a rebuilt p that
        # is now `[a + 2*(b + e), c + 2*(b + e), b + 2*c + 2*g]`
        assert p == [expand_mul(i) for i in C]


@XFAIL
def test_issue_11577():
    def check(eq):
        r, c = cse(eq)
        assert eq.count_ops() >= \
            len(r) + sum(i[1].count_ops() for i in r) + \
            count_ops(c)

    eq = x**5*y**2 + x**5*y + x**5
    assert cse(eq) == (
        [(x0, x**4), (x1, x*y)], [x**5 + x0*x1*y + x0*x1])
        # ([(x0, x**5*y)], [x0*y + x0 + x**5]) or
        # ([(x0, x**5)], [x0*y**2 + x0*y + x0])
    check(eq)

    eq = x**2/(y + 1)**2 + x/(y + 1)
    assert cse(eq) == (
        [(x0, y + 1)], [x**2/x0**2 + x/x0])
        # ([(x0, x/(y + 1))], [x0**2 + x0])
    check(eq)


def test_hollow_rejection():
    eq = [x + 3, x + 4]
    assert cse(eq) == ([], eq)


def test_cse_ignore():
    exprs = [exp(y)*(3*y + 3*sqrt(x+1)), exp(y)*(5*y + 5*sqrt(x+1))]
    subst1, red1 = cse(exprs)
    assert any(y in sub.free_symbols for _, sub in subst1), "cse failed to identify any term with y"

    subst2, red2 = cse(exprs, ignore=(y,))  # y is not allowed in substitutions
    assert not any(y in sub.free_symbols for _, sub in subst2), "Sub-expressions containing y must be ignored"
    assert any(sub - sqrt(x + 1) == 0 for _, sub in subst2), "cse failed to identify sqrt(x + 1) as sub-expression"


def test_cse_ignore_issue_15002():
    l = [
        w*exp(x)*exp(-z),
        exp(y)*exp(x)*exp(-z)
    ]
    substs, reduced = cse(l, ignore=(x,))
    rl = [e.subs(reversed(substs)) for e in reduced]
    assert rl == l


def test_cse_unevaluated():
    xp1 = UnevaluatedExpr(x + 1)
    # This used to cause RecursionError
    [(x0, ue)], [red] = cse([(-1 - xp1) / (1 - xp1)])
    if ue == xp1:
        assert red == (-1 - x0) / (1 - x0)
    elif ue == -xp1:
        assert red == (-1 + x0) / (1 + x0)
    else:
        msg = f'Expected common subexpression {xp1} or {-xp1}, instead got {ue}'
        assert False, msg


def test_cse__performance():
    nexprs, nterms = 3, 20
    x = symbols('x:%d' % nterms)
    exprs = [
        reduce(add, [x[j]*(-1)**(i+j) for j in range(nterms)])
        for i in range(nexprs)
    ]
    assert (exprs[0] + exprs[1]).simplify() == 0
    subst, red = cse(exprs)
    assert len(subst) > 0, "exprs[0] == -exprs[2], i.e. a CSE"
    for i, e in enumerate(red):
        assert (e.subs(reversed(subst)) - exprs[i]).simplify() == 0


def test_issue_12070():
    exprs = [x + y, 2 + x + y, x + y + z, 3 + x + y + z]
    subst, red = cse(exprs)
    assert 6 >= (len(subst) + sum(v.count_ops() for k, v in subst) +
                 count_ops(red))


def test_issue_13000():
    eq = x/(-4*x**2 + y**2)
    cse_eq = cse(eq)[1][0]
    assert cse_eq == eq


def test_issue_18203():
    eq = CRootOf(x**5 + 11*x - 2, 0) + CRootOf(x**5 + 11*x - 2, 1)
    assert cse(eq) == ([], [eq])


def test_unevaluated_mul():
    eq = Mul(x + y, x + y, evaluate=False)
    assert cse(eq) == ([(x0, x + y)], [x0**2])


def test_cse_release_variables():
    from sympy.simplify.cse_main import cse_release_variables
    _0, _1, _2, _3, _4 = symbols('_:5')
    eqs = [(x + y - 1)**2, x,
        x + y, (x + y)/(2*x + 1) + (x + y - 1)**2,
        (2*x + 1)**(x + y)]
    r, e = cse(eqs, postprocess=cse_release_variables)
    # this can change in keeping with the intention of the function
    assert r, e == ([
    (x0, x + y), (x1, (x0 - 1)**2), (x2, 2*x + 1),
    (_3, x0/x2 + x1), (_4, x2**x0), (x2, None), (_0, x1),
    (x1, None), (_2, x0), (x0, None), (_1, x)], (_0, _1, _2, _3, _4))
    r.reverse()
    r = [(s, v) for s, v in r if v is not None]
    assert eqs == [i.subs(r) for i in e]


def test_cse_list():
    _cse = lambda x: cse(x, list=False)
    assert _cse(x) == ([], x)
    assert _cse('x') == ([], 'x')
    it = [x]
    for c in (list, tuple, set):
        assert _cse(c(it)) == ([], c(it))
    #Tuple works different from tuple:
    assert _cse(Tuple(*it)) == ([], Tuple(*it))
    d = {x: 1}
    assert _cse(d) == ([], d)

def test_issue_18991():
    A = MatrixSymbol('A', 2, 2)
    assert signsimp(-A * A - A) == -A * A - A


def test_unevaluated_Mul():
    m = [Mul(1, 2, evaluate=False)]
    assert cse(m) == ([], m)


def test_cse_matrix_expression_inverse():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = Inverse(A)
    cse_expr = cse(x)
    assert cse_expr == ([], [Inverse(A)])


def test_cse_matrix_expression_matmul_inverse():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    b = ImmutableDenseMatrix(symbols('b:2'))
    x = MatMul(Inverse(A), b)
    cse_expr = cse(x)
    assert cse_expr == ([], [x])


def test_cse_matrix_negate_matrix():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = MatMul(S.NegativeOne, A)
    cse_expr = cse(x)
    assert cse_expr == ([], [x])


def test_cse_matrix_negate_matmul_not_extracted():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    B = ImmutableDenseMatrix(symbols('B:4')).reshape(2, 2)
    x = MatMul(S.NegativeOne, A, B)
    cse_expr = cse(x)
    assert cse_expr == ([], [x])


@XFAIL  # No simplification rule for nested associative operations
def test_cse_matrix_nested_matmul_collapsed():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    B = ImmutableDenseMatrix(symbols('B:4')).reshape(2, 2)
    x = MatMul(S.NegativeOne, MatMul(A, B))
    cse_expr = cse(x)
    assert cse_expr == ([], [MatMul(S.NegativeOne, A, B)])


def test_cse_matrix_optimize_out_single_argument_mul():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = MatMul(MatMul(MatMul(A)))
    cse_expr = cse(x)
    assert cse_expr == ([], [A])


@XFAIL  # Multiple simplification passed not supported in CSE
def test_cse_matrix_optimize_out_single_argument_mul_combined():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = MatAdd(MatMul(MatMul(MatMul(A))), MatMul(MatMul(A)), MatMul(A), A)
    cse_expr = cse(x)
    assert cse_expr == ([], [MatMul(4, A)])


def test_cse_matrix_optimize_out_single_argument_add():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = MatAdd(MatAdd(MatAdd(MatAdd(A))))
    cse_expr = cse(x)
    assert cse_expr == ([], [A])


@XFAIL  # Multiple simplification passed not supported in CSE
def test_cse_matrix_optimize_out_single_argument_add_combined():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = MatMul(MatAdd(MatAdd(MatAdd(A))), MatAdd(MatAdd(A)), MatAdd(A), A)
    cse_expr = cse(x)
    assert cse_expr == ([], [MatMul(4, A)])


def test_cse_matrix_expression_matrix_solve():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    b = ImmutableDenseMatrix(symbols('b:2'))
    x = MatrixSolve(A, b)
    cse_expr = cse(x)
    assert cse_expr == ([], [x])


def test_cse_matrix_matrix_expression():
    X = ImmutableDenseMatrix(symbols('X:4')).reshape(2, 2)
    y = ImmutableDenseMatrix(symbols('y:2'))
    b = MatMul(Inverse(MatMul(Transpose(X), X)), Transpose(X), y)
    cse_expr = cse(b)
    x0 = MatrixSymbol('x0', 2, 2)
    reduced_expr_expected = MatMul(Inverse(MatMul(x0, X)), x0, y)
    assert cse_expr == ([(x0, Transpose(X))], [reduced_expr_expected])


def test_cse_matrix_kalman_filter():
    """Kalman Filter example from Matthew Rocklin's SciPy 2013 talk.

    Talk titled: "Matrix Expressions and BLAS/LAPACK; SciPy 2013 Presentation"

    Video: https://pyvideo.org/scipy-2013/matrix-expressions-and-blaslapack-scipy-2013-pr.html

    Notes
    =====

    Equations are:

    new_mu = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
           = MatAdd(mu, MatMul(Sigma, Transpose(H), Inverse(MatAdd(R, MatMul(H, Sigma, Transpose(H)))), MatAdd(MatMul(H, mu), MatMul(S.NegativeOne, data))))
    new_Sigma = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma
              = MatAdd(Sigma, MatMul(S.NegativeOne, Sigma, Transpose(H)), Inverse(MatAdd(R, MatMul(H*Sigma*Transpose(H)))), H, Sigma))

    """
    N = 2
    mu = ImmutableDenseMatrix(symbols(f'mu:{N}'))
    Sigma = ImmutableDenseMatrix(symbols(f'Sigma:{N * N}')).reshape(N, N)
    H = ImmutableDenseMatrix(symbols(f'H:{N * N}')).reshape(N, N)
    R = ImmutableDenseMatrix(symbols(f'R:{N * N}')).reshape(N, N)
    data = ImmutableDenseMatrix(symbols(f'data:{N}'))
    new_mu = MatAdd(mu, MatMul(Sigma, Transpose(H), Inverse(MatAdd(R, MatMul(H, Sigma, Transpose(H)))), MatAdd(MatMul(H, mu), MatMul(S.NegativeOne, data))))
    new_Sigma = MatAdd(Sigma, MatMul(S.NegativeOne, Sigma, Transpose(H), Inverse(MatAdd(R, MatMul(H, Sigma, Transpose(H)))), H, Sigma))
    cse_expr = cse([new_mu, new_Sigma])
    x0 = MatrixSymbol('x0', N, N)
    x1 = MatrixSymbol('x1', N, N)
    replacements_expected = [
        (x0, Transpose(H)),
        (x1, Inverse(MatAdd(R, MatMul(H, Sigma, x0)))),
    ]
    reduced_exprs_expected = [
        MatAdd(mu, MatMul(Sigma, x0, x1, MatAdd(MatMul(H, mu), MatMul(S.NegativeOne, data)))),
        MatAdd(Sigma, MatMul(S.NegativeOne, Sigma, x0, x1, H, Sigma)),
    ]
    assert cse_expr == (replacements_expected, reduced_exprs_expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\io\formats\test_to_csv.py ===
import io
import os
import sys
from zipfile import ZipFile

from _csv import Error
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    compat,
)
import pandas._testing as tm


class TestToCSV:
    def test_to_csv_with_single_column(self):
        # see gh-18676, https://bugs.python.org/issue32255
        #
        # Python's CSV library adds an extraneous '""'
        # before the newline when the NaN-value is in
        # the first row. Otherwise, only the newline
        # character is added. This behavior is inconsistent
        # and was patched in https://bugs.python.org/pull_request4672.
        df1 = DataFrame([None, 1])
        expected1 = """\
""
1.0
"""
        with tm.ensure_clean("test.csv") as path:
            df1.to_csv(path, header=None, index=None)
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected1

        df2 = DataFrame([1, None])
        expected2 = """\
1.0
""
"""
        with tm.ensure_clean("test.csv") as path:
            df2.to_csv(path, header=None, index=None)
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected2

    def test_to_csv_default_encoding(self):
        # GH17097
        df = DataFrame({"col": ["AAAAA", "ÄÄÄÄÄ", "ßßßßß", "聞聞聞聞聞"]})

        with tm.ensure_clean("test.csv") as path:
            # the default to_csv encoding is uft-8.
            df.to_csv(path)
            tm.assert_frame_equal(pd.read_csv(path, index_col=0), df)

    def test_to_csv_quotechar(self):
        df = DataFrame({"col": [1, 2]})
        expected = """\
"","col"
"0","1"
"1","2"
"""

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path, quoting=1)  # 1=QUOTE_ALL
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

        expected = """\
$$,$col$
$0$,$1$
$1$,$2$
"""

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path, quoting=1, quotechar="$")
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

        with tm.ensure_clean("test.csv") as path:
            with pytest.raises(TypeError, match="quotechar"):
                df.to_csv(path, quoting=1, quotechar=None)

    def test_to_csv_doublequote(self):
        df = DataFrame({"col": ['a"a', '"bb"']})
        expected = '''\
"","col"
"0","a""a"
"1","""bb"""
'''

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path, quoting=1, doublequote=True)  # QUOTE_ALL
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

        with tm.ensure_clean("test.csv") as path:
            with pytest.raises(Error, match="escapechar"):
                df.to_csv(path, doublequote=False)  # no escapechar set

    def test_to_csv_escapechar(self):
        df = DataFrame({"col": ['a"a', '"bb"']})
        expected = """\
"","col"
"0","a\\"a"
"1","\\"bb\\""
"""

        with tm.ensure_clean("test.csv") as path:  # QUOTE_ALL
            df.to_csv(path, quoting=1, doublequote=False, escapechar="\\")
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

        df = DataFrame({"col": ["a,a", ",bb,"]})
        expected = """\
,col
0,a\\,a
1,\\,bb\\,
"""

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path, quoting=3, escapechar="\\")  # QUOTE_NONE
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

    def test_csv_to_string(self):
        df = DataFrame({"col": [1, 2]})
        expected_rows = [",col", "0,1", "1,2"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv() == expected

    def test_to_csv_decimal(self):
        # see gh-781
        df = DataFrame({"col1": [1], "col2": ["a"], "col3": [10.1]})

        expected_rows = [",col1,col2,col3", "0,1,a,10.1"]
        expected_default = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv() == expected_default

        expected_rows = [";col1;col2;col3", "0;1;a;10,1"]
        expected_european_excel = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(decimal=",", sep=";") == expected_european_excel

        expected_rows = [",col1,col2,col3", "0,1,a,10.10"]
        expected_float_format_default = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(float_format="%.2f") == expected_float_format_default

        expected_rows = [";col1;col2;col3", "0;1;a;10,10"]
        expected_float_format = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (
            df.to_csv(decimal=",", sep=";", float_format="%.2f")
            == expected_float_format
        )

        # see gh-11553: testing if decimal is taken into account for '0.0'
        df = DataFrame({"a": [0, 1.1], "b": [2.2, 3.3], "c": 1})

        expected_rows = ["a,b,c", "0^0,2^2,1", "1^1,3^3,1"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(index=False, decimal="^") == expected

        # same but for an index
        assert df.set_index("a").to_csv(decimal="^") == expected

        # same for a multi-index
        assert df.set_index(["a", "b"]).to_csv(decimal="^") == expected

    def test_to_csv_float_format(self):
        # testing if float_format is taken into account for the index
        # GH 11553
        df = DataFrame({"a": [0, 1], "b": [2.2, 3.3], "c": 1})

        expected_rows = ["a,b,c", "0,2.20,1", "1,3.30,1"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.set_index("a").to_csv(float_format="%.2f") == expected

        # same for a multi-index
        assert df.set_index(["a", "b"]).to_csv(float_format="%.2f") == expected

    def test_to_csv_na_rep(self):
        # see gh-11553
        #
        # Testing if NaN values are correctly represented in the index.
        df = DataFrame({"a": [0, np.nan], "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "0.0,0,2", "_,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        # now with an index containing only NaNs
        df = DataFrame({"a": np.nan, "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "_,0,2", "_,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        # check if na_rep parameter does not break anything when no NaN
        df = DataFrame({"a": 0, "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "0,0,2", "0,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        csv = pd.Series(["a", pd.NA, "c"]).to_csv(na_rep="ZZZZZ")
        expected = tm.convert_rows_list_to_csv_str([",0", "0,a", "1,ZZZZZ", "2,c"])
        assert expected == csv

    def test_to_csv_na_rep_nullable_string(self, nullable_string_dtype):
        # GH 29975
        # Make sure full na_rep shows up when a dtype is provided
        expected = tm.convert_rows_list_to_csv_str([",0", "0,a", "1,ZZZZZ", "2,c"])
        csv = pd.Series(["a", pd.NA, "c"], dtype=nullable_string_dtype).to_csv(
            na_rep="ZZZZZ"
        )
        assert expected == csv

    def test_to_csv_date_format(self):
        # GH 10209
        df_sec = DataFrame({"A": pd.date_range("20130101", periods=5, freq="s")})
        df_day = DataFrame({"A": pd.date_range("20130101", periods=5, freq="d")})

        expected_rows = [
            ",A",
            "0,2013-01-01 00:00:00",
            "1,2013-01-01 00:00:01",
            "2,2013-01-01 00:00:02",
            "3,2013-01-01 00:00:03",
            "4,2013-01-01 00:00:04",
        ]
        expected_default_sec = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_sec.to_csv() == expected_default_sec

        expected_rows = [
            ",A",
            "0,2013-01-01 00:00:00",
            "1,2013-01-02 00:00:00",
            "2,2013-01-03 00:00:00",
            "3,2013-01-04 00:00:00",
            "4,2013-01-05 00:00:00",
        ]
        expected_ymdhms_day = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_day.to_csv(date_format="%Y-%m-%d %H:%M:%S") == expected_ymdhms_day

        expected_rows = [
            ",A",
            "0,2013-01-01",
            "1,2013-01-01",
            "2,2013-01-01",
            "3,2013-01-01",
            "4,2013-01-01",
        ]
        expected_ymd_sec = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_sec.to_csv(date_format="%Y-%m-%d") == expected_ymd_sec

        expected_rows = [
            ",A",
            "0,2013-01-01",
            "1,2013-01-02",
            "2,2013-01-03",
            "3,2013-01-04",
            "4,2013-01-05",
        ]
        expected_default_day = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df_day.to_csv() == expected_default_day
        assert df_day.to_csv(date_format="%Y-%m-%d") == expected_default_day

        # see gh-7791
        #
        # Testing if date_format parameter is taken into account
        # for multi-indexed DataFrames.
        df_sec["B"] = 0
        df_sec["C"] = 1

        expected_rows = ["A,B,C", "2013-01-01,0,1.0"]
        expected_ymd_sec = tm.convert_rows_list_to_csv_str(expected_rows)

        df_sec_grouped = df_sec.groupby([pd.Grouper(key="A", freq="1h"), "B"])
        assert df_sec_grouped.mean().to_csv(date_format="%Y-%m-%d") == expected_ymd_sec

    def test_to_csv_different_datetime_formats(self):
        # GH#21734
        df = DataFrame(
            {
                "date": pd.to_datetime("1970-01-01"),
                "datetime": pd.date_range("1970-01-01", periods=2, freq="h"),
            }
        )
        expected_rows = [
            "date,datetime",
            "1970-01-01,1970-01-01 00:00:00",
            "1970-01-01,1970-01-01 01:00:00",
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(index=False) == expected

    def test_to_csv_date_format_in_categorical(self):
        # GH#40754
        ser = pd.Series(pd.to_datetime(["2021-03-27", pd.NaT], format="%Y-%m-%d"))
        ser = ser.astype("category")
        expected = tm.convert_rows_list_to_csv_str(["0", "2021-03-27", '""'])
        assert ser.to_csv(index=False) == expected

        ser = pd.Series(
            pd.date_range(
                start="2021-03-27", freq="D", periods=1, tz="Europe/Berlin"
            ).append(pd.DatetimeIndex([pd.NaT]))
        )
        ser = ser.astype("category")
        assert ser.to_csv(index=False, date_format="%Y-%m-%d") == expected

    def test_to_csv_float_ea_float_format(self):
        # GH#45991
        df = DataFrame({"a": [1.1, 2.02, pd.NA, 6.000006], "b": "c"})
        df["a"] = df["a"].astype("Float64")
        result = df.to_csv(index=False, float_format="%.5f")
        expected = tm.convert_rows_list_to_csv_str(
            ["a,b", "1.10000,c", "2.02000,c", ",c", "6.00001,c"]
        )
        assert result == expected

    def test_to_csv_float_ea_no_float_format(self):
        # GH#45991
        df = DataFrame({"a": [1.1, 2.02, pd.NA, 6.000006], "b": "c"})
        df["a"] = df["a"].astype("Float64")
        result = df.to_csv(index=False)
        expected = tm.convert_rows_list_to_csv_str(
            ["a,b", "1.1,c", "2.02,c", ",c", "6.000006,c"]
        )
        assert result == expected

    def test_to_csv_multi_index(self):
        # see gh-6618
        df = DataFrame([1], columns=pd.MultiIndex.from_arrays([[1], [2]]))

        exp_rows = [",1", ",2", "0,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv() == exp

        exp_rows = ["1", "2", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv(index=False) == exp

        df = DataFrame(
            [1],
            columns=pd.MultiIndex.from_arrays([[1], [2]]),
            index=pd.MultiIndex.from_arrays([[1], [2]]),
        )

        exp_rows = [",,1", ",,2", "1,2,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv() == exp

        exp_rows = ["1", "2", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv(index=False) == exp

        df = DataFrame([1], columns=pd.MultiIndex.from_arrays([["foo"], ["bar"]]))

        exp_rows = [",foo", ",bar", "0,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv() == exp

        exp_rows = ["foo", "bar", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        assert df.to_csv(index=False) == exp

    @pytest.mark.parametrize(
        "ind,expected",
        [
            (
                pd.MultiIndex(levels=[[1.0]], codes=[[0]], names=["x"]),
                "x,data\n1.0,1\n",
            ),
            (
                pd.MultiIndex(
                    levels=[[1.0], [2.0]], codes=[[0], [0]], names=["x", "y"]
                ),
                "x,y,data\n1.0,2.0,1\n",
            ),
        ],
    )
    def test_to_csv_single_level_multi_index(self, ind, expected, frame_or_series):
        # see gh-19589
        obj = frame_or_series(pd.Series([1], ind, name="data"))

        result = obj.to_csv(lineterminator="\n", header=True)
        assert result == expected

    def test_to_csv_string_array_ascii(self):
        # GH 10813
        str_array = [{"names": ["foo", "bar"]}, {"names": ["baz", "qux"]}]
        df = DataFrame(str_array)
        expected_ascii = """\
,names
0,"['foo', 'bar']"
1,"['baz', 'qux']"
"""
        with tm.ensure_clean("str_test.csv") as path:
            df.to_csv(path, encoding="ascii")
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected_ascii

    def test_to_csv_string_array_utf8(self):
        # GH 10813
        str_array = [{"names": ["foo", "bar"]}, {"names": ["baz", "qux"]}]
        df = DataFrame(str_array)
        expected_utf8 = """\
,names
0,"['foo', 'bar']"
1,"['baz', 'qux']"
"""
        with tm.ensure_clean("unicode_test.csv") as path:
            df.to_csv(path, encoding="utf-8")
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected_utf8

    def test_to_csv_string_with_lf(self):
        # GH 20353
        data = {"int": [1, 2, 3], "str_lf": ["abc", "d\nef", "g\nh\n\ni"]}
        df = DataFrame(data)
        with tm.ensure_clean("lf_test.csv") as path:
            # case 1: The default line terminator(=os.linesep)(PR 21406)
            os_linesep = os.linesep.encode("utf-8")
            expected_noarg = (
                b"int,str_lf"
                + os_linesep
                + b"1,abc"
                + os_linesep
                + b'2,"d\nef"'
                + os_linesep
                + b'3,"g\nh\n\ni"'
                + os_linesep
            )
            df.to_csv(path, index=False)
            with open(path, "rb") as f:
                assert f.read() == expected_noarg
        with tm.ensure_clean("lf_test.csv") as path:
            # case 2: LF as line terminator
            expected_lf = b'int,str_lf\n1,abc\n2,"d\nef"\n3,"g\nh\n\ni"\n'
            df.to_csv(path, lineterminator="\n", index=False)
            with open(path, "rb") as f:
                assert f.read() == expected_lf
        with tm.ensure_clean("lf_test.csv") as path:
            # case 3: CRLF as line terminator
            # 'lineterminator' should not change inner element
            expected_crlf = b'int,str_lf\r\n1,abc\r\n2,"d\nef"\r\n3,"g\nh\n\ni"\r\n'
            df.to_csv(path, lineterminator="\r\n", index=False)
            with open(path, "rb") as f:
                assert f.read() == expected_crlf

    def test_to_csv_string_with_crlf(self):
        # GH 20353
        data = {"int": [1, 2, 3], "str_crlf": ["abc", "d\r\nef", "g\r\nh\r\n\r\ni"]}
        df = DataFrame(data)
        with tm.ensure_clean("crlf_test.csv") as path:
            # case 1: The default line terminator(=os.linesep)(PR 21406)
            os_linesep = os.linesep.encode("utf-8")
            expected_noarg = (
                b"int,str_crlf"
                + os_linesep
                + b"1,abc"
                + os_linesep
                + b'2,"d\r\nef"'
                + os_linesep
                + b'3,"g\r\nh\r\n\r\ni"'
                + os_linesep
            )
            df.to_csv(path, index=False)
            with open(path, "rb") as f:
                assert f.read() == expected_noarg
        with tm.ensure_clean("crlf_test.csv") as path:
            # case 2: LF as line terminator
            expected_lf = b'int,str_crlf\n1,abc\n2,"d\r\nef"\n3,"g\r\nh\r\n\r\ni"\n'
            df.to_csv(path, lineterminator="\n", index=False)
            with open(path, "rb") as f:
                assert f.read() == expected_lf
        with tm.ensure_clean("crlf_test.csv") as path:
            # case 3: CRLF as line terminator
            # 'lineterminator' should not change inner element
            expected_crlf = (
                b"int,str_crlf\r\n"
                b"1,abc\r\n"
                b'2,"d\r\nef"\r\n'
                b'3,"g\r\nh\r\n\r\ni"\r\n'
            )
            df.to_csv(path, lineterminator="\r\n", index=False)
            with open(path, "rb") as f:
                assert f.read() == expected_crlf

    def test_to_csv_stdout_file(self, capsys):
        # GH 21561
        df = DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["name_1", "name_2"])
        expected_rows = [",name_1,name_2", "0,foo,bar", "1,baz,qux"]
        expected_ascii = tm.convert_rows_list_to_csv_str(expected_rows)

        df.to_csv(sys.stdout, encoding="ascii")
        captured = capsys.readouterr()

        assert captured.out == expected_ascii
        assert not sys.stdout.closed

    @pytest.mark.xfail(
        compat.is_platform_windows(),
        reason=(
            "Especially in Windows, file stream should not be passed"
            "to csv writer without newline='' option."
            "(https://docs.python.org/3/library/csv.html#csv.writer)"
        ),
    )
    def test_to_csv_write_to_open_file(self):
        # GH 21696
        df = DataFrame({"a": ["x", "y", "z"]})
        expected = """\
manual header
x
y
z
"""
        with tm.ensure_clean("test.txt") as path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("manual header\n")
                df.to_csv(f, header=None, index=None)
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

    def test_to_csv_write_to_open_file_with_newline_py3(self):
        # see gh-21696
        # see gh-20353
        df = DataFrame({"a": ["x", "y", "z"]})
        expected_rows = ["x", "y", "z"]
        expected = "manual header\n" + tm.convert_rows_list_to_csv_str(expected_rows)
        with tm.ensure_clean("test.txt") as path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write("manual header\n")
                df.to_csv(f, header=None, index=None)

            with open(path, "rb") as f:
                assert f.read() == bytes(expected, "utf-8")

    @pytest.mark.parametrize("to_infer", [True, False])
    @pytest.mark.parametrize("read_infer", [True, False])
    def test_to_csv_compression(
        self, compression_only, read_infer, to_infer, compression_to_extension
    ):
        # see gh-15008
        compression = compression_only

        # We'll complete file extension subsequently.
        filename = "test."
        filename += compression_to_extension[compression]

        df = DataFrame({"A": [1]})

        to_compression = "infer" if to_infer else compression
        read_compression = "infer" if read_infer else compression

        with tm.ensure_clean(filename) as path:
            df.to_csv(path, compression=to_compression)
            result = pd.read_csv(path, index_col=0, compression=read_compression)
            tm.assert_frame_equal(result, df)

    def test_to_csv_compression_dict(self, compression_only):
        # GH 26023
        method = compression_only
        df = DataFrame({"ABC": [1]})
        filename = "to_csv_compress_as_dict."
        extension = {
            "gzip": "gz",
            "zstd": "zst",
        }.get(method, method)
        filename += extension
        with tm.ensure_clean(filename) as path:
            df.to_csv(path, compression={"method": method})
            read_df = pd.read_csv(path, index_col=0)
            tm.assert_frame_equal(read_df, df)

    def test_to_csv_compression_dict_no_method_raises(self):
        # GH 26023
        df = DataFrame({"ABC": [1]})
        compression = {"some_option": True}
        msg = "must have key 'method'"

        with tm.ensure_clean("out.zip") as path:
            with pytest.raises(ValueError, match=msg):
                df.to_csv(path, compression=compression)

    @pytest.mark.parametrize("compression", ["zip", "infer"])
    @pytest.mark.parametrize("archive_name", ["test_to_csv.csv", "test_to_csv.zip"])
    def test_to_csv_zip_arguments(self, compression, archive_name):
        # GH 26023
        df = DataFrame({"ABC": [1]})
        with tm.ensure_clean("to_csv_archive_name.zip") as path:
            df.to_csv(
                path, compression={"method": compression, "archive_name": archive_name}
            )
            with ZipFile(path) as zp:
                assert len(zp.filelist) == 1
                archived_file = zp.filelist[0].filename
                assert archived_file == archive_name

    @pytest.mark.parametrize(
        "filename,expected_arcname",
        [
            ("archive.csv", "archive.csv"),
            ("archive.tsv", "archive.tsv"),
            ("archive.csv.zip", "archive.csv"),
            ("archive.tsv.zip", "archive.tsv"),
            ("archive.zip", "archive"),
        ],
    )
    def test_to_csv_zip_infer_name(self, tmp_path, filename, expected_arcname):
        # GH 39465
        df = DataFrame({"ABC": [1]})
        path = tmp_path / filename
        df.to_csv(path, compression="zip")
        with ZipFile(path) as zp:
            assert len(zp.filelist) == 1
            archived_file = zp.filelist[0].filename
            assert archived_file == expected_arcname

    @pytest.mark.parametrize("df_new_type", ["Int64"])
    def test_to_csv_na_rep_long_string(self, df_new_type):
        # see gh-25099
        df = DataFrame({"c": [float("nan")] * 3})
        df = df.astype(df_new_type)
        expected_rows = ["c", "mynull", "mynull", "mynull"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        result = df.to_csv(index=False, na_rep="mynull", encoding="ascii")

        assert expected == result

    def test_to_csv_timedelta_precision(self):
        # GH 6783
        s = pd.Series([1, 1]).astype("timedelta64[ns]")
        buf = io.StringIO()
        s.to_csv(buf)
        result = buf.getvalue()
        expected_rows = [
            ",0",
            "0,0 days 00:00:00.000000001",
            "1,0 days 00:00:00.000000001",
        ]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_na_rep_truncated(self):
        # https://github.com/pandas-dev/pandas/issues/31447
        result = pd.Series(range(8, 12)).to_csv(na_rep="-")
        expected = tm.convert_rows_list_to_csv_str([",0", "0,8", "1,9", "2,10", "3,11"])
        assert result == expected

        result = pd.Series([True, False]).to_csv(na_rep="nan")
        expected = tm.convert_rows_list_to_csv_str([",0", "0,True", "1,False"])
        assert result == expected

        result = pd.Series([1.1, 2.2]).to_csv(na_rep=".")
        expected = tm.convert_rows_list_to_csv_str([",0", "0,1.1", "1,2.2"])
        assert result == expected

    @pytest.mark.parametrize("errors", ["surrogatepass", "ignore", "replace"])
    def test_to_csv_errors(self, errors):
        # GH 22610
        data = ["\ud800foo"]
        ser = pd.Series(data, index=Index(data, dtype=object), dtype=object)
        with tm.ensure_clean("test.csv") as path:
            ser.to_csv(path, errors=errors)
        # No use in reading back the data as it is not the same anymore
        # due to the error handling

    @pytest.mark.parametrize("mode", ["wb", "w"])
    def test_to_csv_binary_handle(self, mode):
        """
        Binary file objects should work (if 'mode' contains a 'b') or even without
        it in most cases.

        GH 35058 and GH 19827
        """
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )
        with tm.ensure_clean() as path:
            with open(path, mode="w+b") as handle:
                df.to_csv(handle, mode=mode)
            tm.assert_frame_equal(df, pd.read_csv(path, index_col=0))

    @pytest.mark.parametrize("mode", ["wb", "w"])
    def test_to_csv_encoding_binary_handle(self, mode):
        """
        Binary file objects should honor a specified encoding.

        GH 23854 and GH 13068 with binary handles
        """
        # example from GH 23854
        content = "a, b, 🐟".encode("utf-8-sig")
        buffer = io.BytesIO(content)
        df = pd.read_csv(buffer, encoding="utf-8-sig")

        buffer = io.BytesIO()
        df.to_csv(buffer, mode=mode, encoding="utf-8-sig", index=False)
        buffer.seek(0)  # tests whether file handle wasn't closed
        assert buffer.getvalue().startswith(content)

        # example from GH 13068
        with tm.ensure_clean() as path:
            with open(path, "w+b") as handle:
                DataFrame().to_csv(handle, mode=mode, encoding="utf-8-sig")

                handle.seek(0)
                assert handle.read().startswith(b'\xef\xbb\xbf""')


def test_to_csv_iterative_compression_name(compression):
    # GH 38714
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    with tm.ensure_clean() as path:
        df.to_csv(path, compression=compression, chunksize=1)
        tm.assert_frame_equal(
            pd.read_csv(path, compression=compression, index_col=0), df
        )


def test_to_csv_iterative_compression_buffer(compression):
    # GH 38714
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )
    with io.BytesIO() as buffer:
        df.to_csv(buffer, compression=compression, chunksize=1)
        buffer.seek(0)
        tm.assert_frame_equal(
            pd.read_csv(buffer, compression=compression, index_col=0), df
        )
        assert not buffer.closed


def test_to_csv_pos_args_deprecation():
    # GH-54229
    df = DataFrame({"a": [1, 2, 3]})
    msg = (
        r"Starting with pandas version 3.0 all arguments of to_csv except for the "
        r"argument 'path_or_buf' will be keyword-only."
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        buffer = io.BytesIO()
        df.to_csv(buffer, ";")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\tests\test_polyroots.py ===
"""Tests for algorithms for computing symbolic roots of polynomials. """

from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import (conjugate, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.polys.domains.integerring import ZZ
from sympy.sets.sets import Interval
from sympy.simplify.powsimp import powsimp

from sympy.polys import Poly, cyclotomic_poly, intervals, nroots, rootof

from sympy.polys.polyroots import (root_factors, roots_linear,
    roots_quadratic, roots_cubic, roots_quartic, roots_quintic,
    roots_cyclotomic, roots_binomial, preprocess_roots, roots)

from sympy.polys.orthopolys import legendre_poly
from sympy.polys.polyerrors import PolynomialError, \
    UnsolvableFactorError
from sympy.polys.polyutils import _nsort

from sympy.testing.pytest import raises, slow
from sympy.core.random import verify_numerically
import mpmath
from itertools import product



a, b, c, d, e, q, t, x, y, z = symbols('a,b,c,d,e,q,t,x,y,z')


def _check(roots):
    # this is the desired invariant for roots returned
    # by all_roots. It is trivially true for linear
    # polynomials.
    nreal = sum(1 if i.is_real else 0 for i in roots)
    assert sorted(roots[:nreal]) == list(roots[:nreal])
    for ix in range(nreal, len(roots), 2):
        if not (
                roots[ix + 1] == roots[ix] or
                roots[ix + 1] == conjugate(roots[ix])):
            return False
    return True


def test_roots_linear():
    assert roots_linear(Poly(2*x + 1, x)) == [Rational(-1, 2)]


def test_roots_quadratic():
    assert roots_quadratic(Poly(2*x**2, x)) == [0, 0]
    assert roots_quadratic(Poly(2*x**2 + 3*x, x)) == [Rational(-3, 2), 0]
    assert roots_quadratic(Poly(2*x**2 + 3, x)) == [-I*sqrt(6)/2, I*sqrt(6)/2]
    assert roots_quadratic(Poly(2*x**2 + 4*x + 3, x)) == [-1 - I*sqrt(2)/2, -1 + I*sqrt(2)/2]
    _check(Poly(2*x**2 + 4*x + 3, x).all_roots())

    f = x**2 + (2*a*e + 2*c*e)/(a - c)*x + (d - b + a*e**2 - c*e**2)/(a - c)
    assert roots_quadratic(Poly(f, x)) == \
        [-e*(a + c)/(a - c) - sqrt(a*b + c*d - a*d - b*c + 4*a*c*e**2)/(a - c),
         -e*(a + c)/(a - c) + sqrt(a*b + c*d - a*d - b*c + 4*a*c*e**2)/(a - c)]

    # check for simplification
    f = Poly(y*x**2 - 2*x - 2*y, x)
    assert roots_quadratic(f) == \
        [-sqrt(2*y**2 + 1)/y + 1/y, sqrt(2*y**2 + 1)/y + 1/y]
    f = Poly(x**2 + (-y**2 - 2)*x + y**2 + 1, x)
    assert roots_quadratic(f) == \
        [1,y**2 + 1]

    f = Poly(sqrt(2)*x**2 - 1, x)
    r = roots_quadratic(f)
    assert r == _nsort(r)

    # issue 8255
    f = Poly(-24*x**2 - 180*x + 264)
    assert [w.n(2) for w in f.all_roots(radicals=True)] == \
           [w.n(2) for w in f.all_roots(radicals=False)]
    for _a, _b, _c in product((-2, 2), (-2, 2), (0, -1)):
        f = Poly(_a*x**2 + _b*x + _c)
        roots = roots_quadratic(f)
        assert roots == _nsort(roots)


def test_issue_7724():
    eq = Poly(x**4*I + x**2 + I, x)
    assert roots(eq) == {
        sqrt(I/2 + sqrt(5)*I/2): 1,
        sqrt(-sqrt(5)*I/2 + I/2): 1,
        -sqrt(I/2 + sqrt(5)*I/2): 1,
        -sqrt(-sqrt(5)*I/2 + I/2): 1}


def test_issue_8438():
    p = Poly([1, y, -2, -3], x).as_expr()
    roots = roots_cubic(Poly(p, x), x)
    z = Rational(-3, 2) - I*7/2  # this will fail in code given in commit msg
    post = [r.subs(y, z) for r in roots]
    assert set(post) == \
    set(roots_cubic(Poly(p.subs(y, z), x)))
    # /!\ if p is not made an expression, this is *very* slow
    assert all(p.subs({y: z, x: i}).n(2, chop=True) == 0 for i in post)


def test_issue_8285():
    roots = (Poly(4*x**8 - 1, x)*Poly(x**2 + 1)).all_roots()
    assert _check(roots)
    f = Poly(x**4 + 5*x**2 + 6, x)
    ro = [rootof(f, i) for i in range(4)]
    roots = Poly(x**4 + 5*x**2 + 6, x).all_roots()
    assert roots == ro
    assert _check(roots)
    # more than 2 complex roots from which to identify the
    # imaginary ones
    roots = Poly(2*x**8 - 1).all_roots()
    assert _check(roots)
    assert len(Poly(2*x**10 - 1).all_roots()) == 10  # doesn't fail


def test_issue_8289():
    roots = (Poly(x**2 + 2)*Poly(x**4 + 2)).all_roots()
    assert _check(roots)
    roots = Poly(x**6 + 3*x**3 + 2, x).all_roots()
    assert _check(roots)
    roots = Poly(x**6 - x + 1).all_roots()
    assert _check(roots)
    # all imaginary roots with multiplicity of 2
    roots = Poly(x**4 + 4*x**2 + 4, x).all_roots()
    assert _check(roots)


def test_issue_14291():
    assert Poly(((x - 1)**2 + 1)*((x - 1)**2 + 2)*(x - 1)
        ).all_roots() == [1, 1 - I, 1 + I, 1 - sqrt(2)*I, 1 + sqrt(2)*I]
    p = x**4 + 10*x**2 + 1
    ans = [rootof(p, i) for i in range(4)]
    assert Poly(p).all_roots() == ans
    _check(ans)


def test_issue_13340():
    eq = Poly(y**3 + exp(x)*y + x, y, domain='EX')
    roots_d = roots(eq)
    assert len(roots_d) == 3


def test_issue_14522():
    eq = Poly(x**4 + x**3*(16 + 32*I) + x**2*(-285 + 386*I) + x*(-2824 - 448*I) - 2058 - 6053*I, x)
    roots_eq = roots(eq)
    assert all(eq(r) == 0 for r in roots_eq)


def test_issue_15076():
    sol = roots_quartic(Poly(t**4 - 6*t**2 + t/x - 3, t))
    assert sol[0].has(x)


def test_issue_16589():
    eq = Poly(x**4 - 8*sqrt(2)*x**3 + 4*x**3 - 64*sqrt(2)*x**2 + 1024*x, x)
    roots_eq = roots(eq)
    assert 0 in roots_eq


def test_roots_cubic():
    assert roots_cubic(Poly(2*x**3, x)) == [0, 0, 0]
    assert roots_cubic(Poly(x**3 - 3*x**2 + 3*x - 1, x)) == [1, 1, 1]

    # valid for arbitrary y (issue 21263)
    r = root(y, 3)
    assert roots_cubic(Poly(x**3 - y, x)) == [r,
        r*(-S.Half + sqrt(3)*I/2),
        r*(-S.Half - sqrt(3)*I/2)]
    # simpler form when y is negative
    assert roots_cubic(Poly(x**3 - -1, x)) == \
        [-1, S.Half - I*sqrt(3)/2, S.Half + I*sqrt(3)/2]
    assert roots_cubic(Poly(2*x**3 - 3*x**2 - 3*x - 1, x))[0] == \
         S.Half + 3**Rational(1, 3)/2 + 3**Rational(2, 3)/2
    eq = -x**3 + 2*x**2 + 3*x - 2
    assert roots(eq, trig=True, multiple=True) == \
           roots_cubic(Poly(eq, x), trig=True) == [
        Rational(2, 3) + 2*sqrt(13)*cos(acos(8*sqrt(13)/169)/3)/3,
        -2*sqrt(13)*sin(-acos(8*sqrt(13)/169)/3 + pi/6)/3 + Rational(2, 3),
        -2*sqrt(13)*cos(-acos(8*sqrt(13)/169)/3 + pi/3)/3 + Rational(2, 3),
        ]


def test_roots_quartic():
    assert roots_quartic(Poly(x**4, x)) == [0, 0, 0, 0]
    assert roots_quartic(Poly(x**4 + x**3, x)) in [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1]
    ]
    assert roots_quartic(Poly(x**4 - x**3, x)) in [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    lhs = roots_quartic(Poly(x**4 + x, x))
    rhs = [S.Half + I*sqrt(3)/2, S.Half - I*sqrt(3)/2, S.Zero, -S.One]

    assert sorted(lhs, key=hash) == sorted(rhs, key=hash)

    # test of all branches of roots quartic
    for i, (a, b, c, d) in enumerate([(1, 2, 3, 0),
                                      (3, -7, -9, 9),
                                      (1, 2, 3, 4),
                                      (1, 2, 3, 4),
                                      (-7, -3, 3, -6),
                                      (-3, 5, -6, -4),
                                      (6, -5, -10, -3)]):
        if i == 2:
            c = -a*(a**2/S(8) - b/S(2))
        elif i == 3:
            d = a*(a*(a**2*Rational(3, 256) - b/S(16)) + c/S(4))
        eq = x**4 + a*x**3 + b*x**2 + c*x + d
        ans = roots_quartic(Poly(eq, x))
        assert all(eq.subs(x, ai).n(chop=True) == 0 for ai in ans)

    # not all symbolic quartics are unresolvable
    eq = Poly(q*x + q/4 + x**4 + x**3 + 2*x**2 - Rational(1, 3), x)
    sol = roots_quartic(eq)
    assert all(verify_numerically(eq.subs(x, i), 0) for i in sol)
    z = symbols('z', negative=True)
    eq = x**4 + 2*x**3 + 3*x**2 + x*(z + 11) + 5
    zans = roots_quartic(Poly(eq, x))
    assert all(verify_numerically(eq.subs(((x, i), (z, -1))), 0) for i in zans)
    # but some are (see also issue 4989)
    # it's ok if the solution is not Piecewise, but the tests below should pass
    eq = Poly(y*x**4 + x**3 - x + z, x)
    ans = roots_quartic(eq)
    assert all(type(i) == Piecewise for i in ans)
    reps = (
        {"y": Rational(-1, 3), "z": Rational(-1, 4)},  # 4 real
        {"y": Rational(-1, 3), "z": Rational(-1, 2)},  # 2 real
        {"y": Rational(-1, 3), "z": -2})  # 0 real
    for rep in reps:
        sol = roots_quartic(Poly(eq.subs(rep), x))
        assert all(verify_numerically(w.subs(rep) - s, 0) for w, s in zip(ans, sol))


def test_issue_21287():
    assert not any(isinstance(i, Piecewise) for i in roots_quartic(
        Poly(x**4 - x**2*(3 + 5*I) + 2*x*(-1 + I) - 1 + 3*I, x)))


def test_roots_quintic():
    eqs = (x**5 - 2,
            (x/2 + 1)**5 - 5*(x/2 + 1) + 12,
            x**5 - 110*x**3 - 55*x**2 + 2310*x + 979)
    for eq in eqs:
        roots = roots_quintic(Poly(eq))
        assert len(roots) == 5
        assert all(eq.subs(x, r.n(10)).n(chop = 1e-5) == 0 for r in roots)


def test_roots_cyclotomic():
    assert roots_cyclotomic(cyclotomic_poly(1, x, polys=True)) == [1]
    assert roots_cyclotomic(cyclotomic_poly(2, x, polys=True)) == [-1]
    assert roots_cyclotomic(cyclotomic_poly(
        3, x, polys=True)) == [Rational(-1, 2) - I*sqrt(3)/2, Rational(-1, 2) + I*sqrt(3)/2]
    assert roots_cyclotomic(cyclotomic_poly(4, x, polys=True)) == [-I, I]
    assert roots_cyclotomic(cyclotomic_poly(
        6, x, polys=True)) == [S.Half - I*sqrt(3)/2, S.Half + I*sqrt(3)/2]

    assert roots_cyclotomic(cyclotomic_poly(7, x, polys=True)) == [
        -cos(pi/7) - I*sin(pi/7),
        -cos(pi/7) + I*sin(pi/7),
        -cos(pi*Rational(3, 7)) - I*sin(pi*Rational(3, 7)),
        -cos(pi*Rational(3, 7)) + I*sin(pi*Rational(3, 7)),
        cos(pi*Rational(2, 7)) - I*sin(pi*Rational(2, 7)),
        cos(pi*Rational(2, 7)) + I*sin(pi*Rational(2, 7)),
    ]

    assert roots_cyclotomic(cyclotomic_poly(8, x, polys=True)) == [
        -sqrt(2)/2 - I*sqrt(2)/2,
        -sqrt(2)/2 + I*sqrt(2)/2,
        sqrt(2)/2 - I*sqrt(2)/2,
        sqrt(2)/2 + I*sqrt(2)/2,
    ]

    assert roots_cyclotomic(cyclotomic_poly(12, x, polys=True)) == [
        -sqrt(3)/2 - I/2,
        -sqrt(3)/2 + I/2,
        sqrt(3)/2 - I/2,
        sqrt(3)/2 + I/2,
    ]

    assert roots_cyclotomic(
        cyclotomic_poly(1, x, polys=True), factor=True) == [1]
    assert roots_cyclotomic(
        cyclotomic_poly(2, x, polys=True), factor=True) == [-1]

    assert roots_cyclotomic(cyclotomic_poly(3, x, polys=True), factor=True) == \
        [-root(-1, 3), -1 + root(-1, 3)]
    assert roots_cyclotomic(cyclotomic_poly(4, x, polys=True), factor=True) == \
        [-I, I]
    assert roots_cyclotomic(cyclotomic_poly(5, x, polys=True), factor=True) == \
        [-root(-1, 5), -root(-1, 5)**3, root(-1, 5)**2, -1 - root(-1, 5)**2 + root(-1, 5) + root(-1, 5)**3]

    assert roots_cyclotomic(cyclotomic_poly(6, x, polys=True), factor=True) == \
        [1 - root(-1, 3), root(-1, 3)]


def test_roots_binomial():
    assert roots_binomial(Poly(5*x, x)) == [0]
    assert roots_binomial(Poly(5*x**4, x)) == [0, 0, 0, 0]
    assert roots_binomial(Poly(5*x + 2, x)) == [Rational(-2, 5)]

    A = 10**Rational(3, 4)/10

    assert roots_binomial(Poly(5*x**4 + 2, x)) == \
        [-A - A*I, -A + A*I, A - A*I, A + A*I]
    _check(roots_binomial(Poly(x**8 - 2)))

    a1 = Symbol('a1', nonnegative=True)
    b1 = Symbol('b1', nonnegative=True)

    r0 = roots_quadratic(Poly(a1*x**2 + b1, x))
    r1 = roots_binomial(Poly(a1*x**2 + b1, x))

    assert powsimp(r0[0]) == powsimp(r1[0])
    assert powsimp(r0[1]) == powsimp(r1[1])
    for a, b, s, n in product((1, 2), (1, 2), (-1, 1), (2, 3, 4, 5)):
        if a == b and a != 1:  # a == b == 1 is sufficient
            continue
        p = Poly(a*x**n + s*b)
        ans = roots_binomial(p)
        assert ans == _nsort(ans)

    # issue 8813
    assert roots(Poly(2*x**3 - 16*y**3, x)) == {
        2*y*(Rational(-1, 2) - sqrt(3)*I/2): 1,
        2*y: 1,
        2*y*(Rational(-1, 2) + sqrt(3)*I/2): 1}


def test_roots_preprocessing():
    f = a*y*x**2 + y - b

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 1
    assert poly == Poly(a*y*x**2 + y - b, x)

    f = c**3*x**3 + c**2*x**2 + c*x + a

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 1/c
    assert poly == Poly(x**3 + x**2 + x + a, x)

    f = c**3*x**3 + c**2*x**2 + a

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 1/c
    assert poly == Poly(x**3 + x**2 + a, x)

    f = c**3*x**3 + c*x + a

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 1/c
    assert poly == Poly(x**3 + x + a, x)

    f = c**3*x**3 + a

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 1/c
    assert poly == Poly(x**3 + a, x)

    E, F, J, L = symbols("E,F,J,L")

    f = -21601054687500000000*E**8*J**8/L**16 + \
        508232812500000000*F*x*E**7*J**7/L**14 - \
        4269543750000000*E**6*F**2*J**6*x**2/L**12 + \
        16194716250000*E**5*F**3*J**5*x**3/L**10 - \
        27633173750*E**4*F**4*J**4*x**4/L**8 + \
        14840215*E**3*F**5*J**3*x**5/L**6 + \
        54794*E**2*F**6*J**2*x**6/(5*L**4) - \
        1153*E*J*F**7*x**7/(80*L**2) + \
        633*F**8*x**8/160000

    coeff, poly = preprocess_roots(Poly(f, x))

    assert coeff == 20*E*J/(F*L**2)
    assert poly == 633*x**8 - 115300*x**7 + 4383520*x**6 + 296804300*x**5 - 27633173750*x**4 + \
        809735812500*x**3 - 10673859375000*x**2 + 63529101562500*x - 135006591796875

    f = Poly(-y**2 + x**2*exp(x), y, domain=ZZ[x, exp(x)])
    g = Poly(-y**2 + exp(x), y, domain=ZZ[exp(x)])

    assert preprocess_roots(f) == (x, g)


def test_roots0():
    assert roots(1, x) == {}
    assert roots(x, x) == {S.Zero: 1}
    assert roots(x**9, x) == {S.Zero: 9}
    assert roots(((x - 2)*(x + 3)*(x - 4)).expand(), x) == {-S(3): 1, S(2): 1, S(4): 1}

    assert roots(2*x + 1, x) == {Rational(-1, 2): 1}
    assert roots((2*x + 1)**2, x) == {Rational(-1, 2): 2}
    assert roots((2*x + 1)**5, x) == {Rational(-1, 2): 5}
    assert roots((2*x + 1)**10, x) == {Rational(-1, 2): 10}

    assert roots(x**4 - 1, x) == {I: 1, S.One: 1, -S.One: 1, -I: 1}
    assert roots((x**4 - 1)**2, x) == {I: 2, S.One: 2, -S.One: 2, -I: 2}

    assert roots(((2*x - 3)**2).expand(), x) == {Rational( 3, 2): 2}
    assert roots(((2*x + 3)**2).expand(), x) == {Rational(-3, 2): 2}

    assert roots(((2*x - 3)**3).expand(), x) == {Rational( 3, 2): 3}
    assert roots(((2*x + 3)**3).expand(), x) == {Rational(-3, 2): 3}

    assert roots(((2*x - 3)**5).expand(), x) == {Rational( 3, 2): 5}
    assert roots(((2*x + 3)**5).expand(), x) == {Rational(-3, 2): 5}

    assert roots(((a*x - b)**5).expand(), x) == { b/a: 5}
    assert roots(((a*x + b)**5).expand(), x) == {-b/a: 5}

    assert roots(x**2 + (-a - 1)*x + a, x) == {a: 1, S.One: 1}

    assert roots(x**4 - 2*x**2 + 1, x) == {S.One: 2, S.NegativeOne: 2}

    assert roots(x**6 - 4*x**4 + 4*x**3 - x**2, x) == \
        {S.One: 2, -1 - sqrt(2): 1, S.Zero: 2, -1 + sqrt(2): 1}

    assert roots(x**8 - 1, x) == {
        sqrt(2)/2 + I*sqrt(2)/2: 1,
        sqrt(2)/2 - I*sqrt(2)/2: 1,
        -sqrt(2)/2 + I*sqrt(2)/2: 1,
        -sqrt(2)/2 - I*sqrt(2)/2: 1,
        S.One: 1, -S.One: 1, I: 1, -I: 1
    }

    f = -2016*x**2 - 5616*x**3 - 2056*x**4 + 3324*x**5 + 2176*x**6 - \
        224*x**7 - 384*x**8 - 64*x**9

    assert roots(f) == {S.Zero: 2, -S(2): 2, S(2): 1, Rational(-7, 2): 1,
                Rational(-3, 2): 1, Rational(-1, 2): 1, Rational(3, 2): 1}

    assert roots((a + b + c)*x - (a + b + c + d), x) == {(a + b + c + d)/(a + b + c): 1}

    assert roots(x**3 + x**2 - x + 1, x, cubics=False) == {}
    assert roots(((x - 2)*(
        x + 3)*(x - 4)).expand(), x, cubics=False) == {-S(3): 1, S(2): 1, S(4): 1}
    assert roots(((x - 2)*(x + 3)*(x - 4)*(x - 5)).expand(), x, cubics=False) == \
        {-S(3): 1, S(2): 1, S(4): 1, S(5): 1}
    assert roots(x**3 + 2*x**2 + 4*x + 8, x) == {-S(2): 1, -2*I: 1, 2*I: 1}
    assert roots(x**3 + 2*x**2 + 4*x + 8, x, cubics=True) == \
        {-2*I: 1, 2*I: 1, -S(2): 1}
    assert roots((x**2 - x)*(x**3 + 2*x**2 + 4*x + 8), x ) == \
        {S.One: 1, S.Zero: 1, -S(2): 1, -2*I: 1, 2*I: 1}

    r1_2, r1_3 = S.Half, Rational(1, 3)

    x0 = (3*sqrt(33) + 19)**r1_3
    x1 = 4/x0/3
    x2 = x0/3
    x3 = sqrt(3)*I/2
    x4 = x3 - r1_2
    x5 = -x3 - r1_2
    assert roots(x**3 + x**2 - x + 1, x, cubics=True) == {
        -x1 - x2 - r1_3: 1,
        -x1/x4 - x2*x4 - r1_3: 1,
        -x1/x5 - x2*x5 - r1_3: 1,
    }

    f = (x**2 + 2*x + 3).subs(x, 2*x**2 + 3*x).subs(x, 5*x - 4)

    r13_20, r1_20 = [ Rational(*r)
        for r in ((13, 20), (1, 20)) ]

    s2 = sqrt(2)
    assert roots(f, x) == {
        r13_20 + r1_20*sqrt(1 - 8*I*s2): 1,
        r13_20 - r1_20*sqrt(1 - 8*I*s2): 1,
        r13_20 + r1_20*sqrt(1 + 8*I*s2): 1,
        r13_20 - r1_20*sqrt(1 + 8*I*s2): 1,
    }

    f = x**4 + x**3 + x**2 + x + 1

    r1_4, r1_8, r5_8 = [ Rational(*r) for r in ((1, 4), (1, 8), (5, 8)) ]

    assert roots(f, x) == {
        -r1_4 + r1_4*5**r1_2 + I*(r5_8 + r1_8*5**r1_2)**r1_2: 1,
        -r1_4 + r1_4*5**r1_2 - I*(r5_8 + r1_8*5**r1_2)**r1_2: 1,
        -r1_4 - r1_4*5**r1_2 + I*(r5_8 - r1_8*5**r1_2)**r1_2: 1,
        -r1_4 - r1_4*5**r1_2 - I*(r5_8 - r1_8*5**r1_2)**r1_2: 1,
    }

    f = z**3 + (-2 - y)*z**2 + (1 + 2*y - 2*x**2)*z - y + 2*x**2

    assert roots(f, z) == {
        S.One: 1,
        S.Half + S.Half*y + S.Half*sqrt(1 - 2*y + y**2 + 8*x**2): 1,
        S.Half + S.Half*y - S.Half*sqrt(1 - 2*y + y**2 + 8*x**2): 1,
    }

    assert roots(a*b*c*x**3 + 2*x**2 + 4*x + 8, x, cubics=False) == {}
    assert roots(a*b*c*x**3 + 2*x**2 + 4*x + 8, x, cubics=True) != {}

    assert roots(x**4 - 1, x, filter='Z') == {S.One: 1, -S.One: 1}
    assert roots(x**4 - 1, x, filter='I') == {I: 1, -I: 1}

    assert roots((x - 1)*(x + 1), x) == {S.One: 1, -S.One: 1}
    assert roots(
        (x - 1)*(x + 1), x, predicate=lambda r: r.is_positive) == {S.One: 1}

    assert roots(x**4 - 1, x, filter='Z', multiple=True) == [-S.One, S.One]
    assert roots(x**4 - 1, x, filter='I', multiple=True) == [I, -I]

    ar, br = symbols('a, b', real=True)
    p = x**2*(ar-br)**2 + 2*x*(br-ar) + 1
    assert roots(p, x, filter='R') == {1/(ar - br): 2}

    assert roots(x**3, x, multiple=True) == [S.Zero, S.Zero, S.Zero]
    assert roots(1234, x, multiple=True) == []

    f = x**6 - x**5 + x**4 - x**3 + x**2 - x + 1

    assert roots(f) == {
        -I*sin(pi/7) + cos(pi/7): 1,
        -I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 1,
        -I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 1,
        I*sin(pi/7) + cos(pi/7): 1,
        I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 1,
        I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 1,
    }

    g = ((x**2 + 1)*f**2).expand()

    assert roots(g) == {
        -I*sin(pi/7) + cos(pi/7): 2,
        -I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 2,
        -I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 2,
        I*sin(pi/7) + cos(pi/7): 2,
        I*sin(pi*Rational(2, 7)) - cos(pi*Rational(2, 7)): 2,
        I*sin(pi*Rational(3, 7)) + cos(pi*Rational(3, 7)): 2,
        -I: 1, I: 1,
    }

    r = roots(x**3 + 40*x + 64)
    real_root = [rx for rx in r if rx.is_real][0]
    cr = 108 + 6*sqrt(1074)
    assert real_root == -2*root(cr, 3)/3 + 20/root(cr, 3)

    eq = Poly((7 + 5*sqrt(2))*x**3 + (-6 - 4*sqrt(2))*x**2 + (-sqrt(2) - 1)*x + 2, x, domain='EX')
    assert roots(eq) == {-1 + sqrt(2): 1, -2 + 2*sqrt(2): 1, -sqrt(2) + 1: 1}

    eq = Poly(41*x**5 + 29*sqrt(2)*x**5 - 153*x**4 - 108*sqrt(2)*x**4 +
    175*x**3 + 125*sqrt(2)*x**3 - 45*x**2 - 30*sqrt(2)*x**2 - 26*sqrt(2)*x -
    26*x + 24, x, domain='EX')
    assert roots(eq) == {-sqrt(2) + 1: 1, -2 + 2*sqrt(2): 1, -1 + sqrt(2): 1,
                         -4 + 4*sqrt(2): 1, -3 + 3*sqrt(2): 1}

    eq = Poly(x**3 - 2*x**2 + 6*sqrt(2)*x**2 - 8*sqrt(2)*x + 23*x - 14 +
            14*sqrt(2), x, domain='EX')
    assert roots(eq) == {-2*sqrt(2) + 2: 1, -2*sqrt(2) + 1: 1, -2*sqrt(2) - 1: 1}

    assert roots(Poly((x + sqrt(2))**3 - 7, x, domain='EX')) == \
        {-sqrt(2) + root(7, 3)*(-S.Half - sqrt(3)*I/2): 1,
         -sqrt(2) + root(7, 3)*(-S.Half + sqrt(3)*I/2): 1,
         -sqrt(2) + root(7, 3): 1}

def test_roots_slow():
    """Just test that calculating these roots does not hang. """
    a, b, c, d, x = symbols("a,b,c,d,x")

    f1 = x**2*c + (a/b) + x*c*d - a
    f2 = x**2*(a + b*(c - d)*a) + x*a*b*c/(b*d - d) + (a*d - c/d)

    assert list(roots(f1, x).values()) == [1, 1]
    assert list(roots(f2, x).values()) == [1, 1]

    (zz, yy, xx, zy, zx, yx, k) = symbols("zz,yy,xx,zy,zx,yx,k")

    e1 = (zz - k)*(yy - k)*(xx - k) + zy*yx*zx + zx - zy - yx
    e2 = (zz - k)*yx*yx + zx*(yy - k)*zx + zy*zy*(xx - k)

    assert list(roots(e1 - e2, k).values()) == [1, 1, 1]

    f = x**3 + 2*x**2 + 8
    R = list(roots(f).keys())

    assert not any(i for i in [f.subs(x, ri).n(chop=True) for ri in R])


def test_roots_inexact():
    R1 = roots(x**2 + x + 1, x, multiple=True)
    R2 = roots(x**2 + x + 1.0, x, multiple=True)

    for r1, r2 in zip(R1, R2):
        assert abs(r1 - r2) < 1e-12

    f = x**4 + 3.0*sqrt(2.0)*x**3 - (78.0 + 24.0*sqrt(3.0))*x**2 \
        + 144.0*(2*sqrt(3.0) + 9.0)

    R1 = roots(f, multiple=True)
    R2 = (-12.7530479110482, -3.85012393732929,
          4.89897948556636, 7.46155167569183)

    for r1, r2 in zip(R1, R2):
        assert abs(r1 - r2) < 1e-10


def test_roots_preprocessed():
    E, F, J, L = symbols("E,F,J,L")

    f = -21601054687500000000*E**8*J**8/L**16 + \
        508232812500000000*F*x*E**7*J**7/L**14 - \
        4269543750000000*E**6*F**2*J**6*x**2/L**12 + \
        16194716250000*E**5*F**3*J**5*x**3/L**10 - \
        27633173750*E**4*F**4*J**4*x**4/L**8 + \
        14840215*E**3*F**5*J**3*x**5/L**6 + \
        54794*E**2*F**6*J**2*x**6/(5*L**4) - \
        1153*E*J*F**7*x**7/(80*L**2) + \
        633*F**8*x**8/160000

    assert roots(f, x) == {}

    R1 = roots(f.evalf(), x, multiple=True)
    R2 = [-1304.88375606366, 97.1168816800648, 186.946430171876, 245.526792947065,
          503.441004174773, 791.549343830097, 1273.16678129348, 1850.10650616851]

    w = Wild('w')
    p = w*E*J/(F*L**2)

    assert len(R1) == len(R2)

    for r1, r2 in zip(R1, R2):
        match = r1.match(p)
        assert match is not None and abs(match[w] - r2) < 1e-10


def test_roots_strict():
    assert roots(x**2 - 2*x + 1, strict=False) == {1: 2}
    assert roots(x**2 - 2*x + 1, strict=True) == {1: 2}

    assert roots(x**6 - 2*x**5 - x**2 + 3*x - 2, strict=False) == {2: 1}
    raises(UnsolvableFactorError, lambda: roots(x**6 - 2*x**5 - x**2 + 3*x - 2, strict=True))


def test_roots_mixed():
    f = -1936 - 5056*x - 7592*x**2 + 2704*x**3 - 49*x**4

    _re, _im = intervals(f, all=True)
    _nroots = nroots(f)
    _sroots = roots(f, multiple=True)

    _re = [ Interval(a, b) for (a, b), _ in _re ]
    _im = [ Interval(re(a), re(b))*Interval(im(a), im(b)) for (a, b),
            _ in _im ]

    _intervals = _re + _im
    _sroots = [ r.evalf() for r in _sroots ]

    _nroots = sorted(_nroots, key=lambda x: x.sort_key())
    _sroots = sorted(_sroots, key=lambda x: x.sort_key())

    for _roots in (_nroots, _sroots):
        for i, r in zip(_intervals, _roots):
            if r.is_real:
                assert r in i
            else:
                assert (re(r), im(r)) in i


def test_root_factors():
    assert root_factors(Poly(1, x)) == [Poly(1, x)]
    assert root_factors(Poly(x, x)) == [Poly(x, x)]

    assert root_factors(x**2 - 1, x) == [x + 1, x - 1]
    assert root_factors(x**2 - y, x) == [x - sqrt(y), x + sqrt(y)]

    assert root_factors((x**4 - 1)**2) == \
        [x + 1, x + 1, x - 1, x - 1, x - I, x - I, x + I, x + I]

    assert root_factors(Poly(x**4 - 1, x), filter='Z') == \
        [Poly(x + 1, x), Poly(x - 1, x), Poly(x**2 + 1, x)]
    assert root_factors(8*x**2 + 12*x**4 + 6*x**6 + x**8, x, filter='Q') == \
        [x, x, x**6 + 6*x**4 + 12*x**2 + 8]


@slow
def test_nroots1():
    n = 64
    p = legendre_poly(n, x, polys=True)

    raises(mpmath.mp.NoConvergence, lambda: p.nroots(n=3, maxsteps=5))

    roots = p.nroots(n=3)
    # The order of roots matters. They are ordered from smallest to the
    # largest.
    assert [str(r) for r in roots] == \
            ['-0.999', '-0.996', '-0.991', '-0.983', '-0.973', '-0.961',
            '-0.946', '-0.930', '-0.911', '-0.889', '-0.866', '-0.841',
            '-0.813', '-0.784', '-0.753', '-0.720', '-0.685', '-0.649',
            '-0.611', '-0.572', '-0.531', '-0.489', '-0.446', '-0.402',
            '-0.357', '-0.311', '-0.265', '-0.217', '-0.170', '-0.121',
            '-0.0730', '-0.0243', '0.0243', '0.0730', '0.121', '0.170',
            '0.217', '0.265', '0.311', '0.357', '0.402', '0.446', '0.489',
            '0.531', '0.572', '0.611', '0.649', '0.685', '0.720', '0.753',
            '0.784', '0.813', '0.841', '0.866', '0.889', '0.911', '0.930',
            '0.946', '0.961', '0.973', '0.983', '0.991', '0.996', '0.999']

def test_nroots2():
    p = Poly(x**5 + 3*x + 1, x)

    roots = p.nroots(n=3)
    # The order of roots matters. The roots are ordered by their real
    # components (if they agree, then by their imaginary components),
    # with real roots appearing first.
    assert [str(r) for r in roots] == \
            ['-0.332', '-0.839 - 0.944*I', '-0.839 + 0.944*I',
                '1.01 - 0.937*I', '1.01 + 0.937*I']

    roots = p.nroots(n=5)
    assert [str(r) for r in roots] == \
            ['-0.33199', '-0.83907 - 0.94385*I', '-0.83907 + 0.94385*I',
              '1.0051 - 0.93726*I', '1.0051 + 0.93726*I']


def test_roots_composite():
    assert len(roots(Poly(y**3 + y**2*sqrt(x) + y + x, y, composite=True))) == 3


def test_issue_19113():
    eq = cos(x)**3 - cos(x) + 1
    raises(PolynomialError, lambda: roots(eq))


def test_issue_17454():
    assert roots([1, -3*(-4 - 4*I)**2/8 + 12*I, 0], multiple=True) == [0, 0]


def test_issue_20913():
    assert Poly(x + 9671406556917067856609794, x).real_roots() == [-9671406556917067856609794]
    assert Poly(x**3 + 4, x).real_roots() == [-2**(S(2)/3)]


def test_issue_22768():
    e = Rational(1, 3)
    r = (-1/a)**e*(a + 1)**(5*e)
    assert roots(Poly(a*x**3 + (a + 1)**5, x)) == {
        r: 1,
        -r*(1 + sqrt(3)*I)/2: 1,
        r*(-1 + sqrt(3)*I)/2: 1}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\groupby\test_raises.py ===
# Only tests that raise an error and have no better location should go here.
# Tests for specific groupby methods should go in their respective
# test file.

import datetime
import re

import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    Grouper,
    Series,
)
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args


@pytest.fixture(
    params=[
        "a",
        ["a"],
        ["a", "b"],
        Grouper(key="a"),
        lambda x: x % 2,
        [0, 0, 0, 1, 2, 2, 2, 3, 3],
        np.array([0, 0, 0, 1, 2, 2, 2, 3, 3]),
        dict(zip(range(9), [0, 0, 0, 1, 2, 2, 2, 3, 3])),
        Series([1, 1, 1, 1, 1, 2, 2, 2, 2]),
        [Series([1, 1, 1, 1, 1, 2, 2, 2, 2]), Series([3, 3, 4, 4, 4, 4, 4, 3, 3])],
    ]
)
def by(request):
    return request.param


@pytest.fixture(params=[True, False])
def groupby_series(request):
    return request.param


@pytest.fixture
def df_with_string_col():
    df = DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],
            "c": range(9),
            "d": list("xyzwtyuio"),
        }
    )
    return df


@pytest.fixture
def df_with_datetime_col():
    df = DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],
            "c": range(9),
            "d": datetime.datetime(2005, 1, 1, 10, 30, 23, 540000),
        }
    )
    return df


@pytest.fixture
def df_with_timedelta_col():
    df = DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],
            "c": range(9),
            "d": datetime.timedelta(days=1),
        }
    )
    return df


@pytest.fixture
def df_with_cat_col():
    df = DataFrame(
        {
            "a": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "b": [3, 3, 4, 4, 4, 4, 4, 3, 3],
            "c": range(9),
            "d": Categorical(
                ["a", "a", "a", "a", "b", "b", "b", "b", "c"],
                categories=["a", "b", "c", "d"],
                ordered=True,
            ),
        }
    )
    return df


def _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg=""):
    warn_klass = None if warn_msg == "" else FutureWarning
    with tm.assert_produces_warning(warn_klass, match=warn_msg):
        if klass is None:
            if how == "method":
                getattr(gb, groupby_func)(*args)
            elif how == "agg":
                gb.agg(groupby_func, *args)
            else:
                gb.transform(groupby_func, *args)
        else:
            with pytest.raises(klass, match=msg):
                if how == "method":
                    getattr(gb, groupby_func)(*args)
                elif how == "agg":
                    gb.agg(groupby_func, *args)
                else:
                    gb.transform(groupby_func, *args)


@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_string(
    how, by, groupby_series, groupby_func, df_with_string_col, using_infer_string
):
    df = df_with_string_col
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

        if groupby_func == "corrwith":
            assert not hasattr(gb, "corrwith")
            return

    klass, msg = {
        "all": (None, ""),
        "any": (None, ""),
        "bfill": (None, ""),
        "corrwith": (TypeError, "Could not convert"),
        "count": (None, ""),
        "cumcount": (None, ""),
        "cummax": (
            (NotImplementedError, TypeError),
            "(function|cummax) is not (implemented|supported) for (this|object) dtype",
        ),
        "cummin": (
            (NotImplementedError, TypeError),
            "(function|cummin) is not (implemented|supported) for (this|object) dtype",
        ),
        "cumprod": (
            (NotImplementedError, TypeError),
            "(function|cumprod) is not (implemented|supported) for (this|object) dtype",
        ),
        "cumsum": (
            (NotImplementedError, TypeError),
            "(function|cumsum) is not (implemented|supported) for (this|object) dtype",
        ),
        "diff": (TypeError, "unsupported operand type"),
        "ffill": (None, ""),
        "fillna": (None, ""),
        "first": (None, ""),
        "idxmax": (None, ""),
        "idxmin": (None, ""),
        "last": (None, ""),
        "max": (None, ""),
        "mean": (
            TypeError,
            re.escape("agg function failed [how->mean,dtype->object]"),
        ),
        "median": (
            TypeError,
            re.escape("agg function failed [how->median,dtype->object]"),
        ),
        "min": (None, ""),
        "ngroup": (None, ""),
        "nunique": (None, ""),
        "pct_change": (TypeError, "unsupported operand type"),
        "prod": (
            TypeError,
            re.escape("agg function failed [how->prod,dtype->object]"),
        ),
        "quantile": (TypeError, "dtype 'object' does not support operation 'quantile'"),
        "rank": (None, ""),
        "sem": (ValueError, "could not convert string to float"),
        "shift": (None, ""),
        "size": (None, ""),
        "skew": (ValueError, "could not convert string to float"),
        "std": (ValueError, "could not convert string to float"),
        "sum": (None, ""),
        "var": (
            TypeError,
            re.escape("agg function failed [how->var,dtype->"),
        ),
    }[groupby_func]

    if using_infer_string:
        if groupby_func in [
            "prod",
            "mean",
            "median",
            "cumsum",
            "cumprod",
            "std",
            "sem",
            "var",
            "skew",
            "quantile",
        ]:
            msg = f"dtype 'str' does not support operation '{groupby_func}'"
            if groupby_func in ["sem", "std", "skew"]:
                # The object-dtype raises ValueError when trying to convert to numeric.
                klass = TypeError
        elif groupby_func == "pct_change" and df["d"].dtype.storage == "pyarrow":
            # This doesn't go through EA._groupby_op so the message isn't controlled
            #  there.
            msg = "operation 'truediv' not supported for dtype 'str' with dtype 'str'"
        elif groupby_func == "diff" and df["d"].dtype.storage == "pyarrow":
            # This doesn't go through EA._groupby_op so the message isn't controlled
            #  there.
            msg = "operation 'sub' not supported for dtype 'str' with dtype 'str'"

        elif groupby_func in ["cummin", "cummax"]:
            msg = msg.replace("object", "str")
        elif groupby_func == "corrwith":
            msg = "Cannot perform reduction 'mean' with string dtype"

    if groupby_func == "fillna":
        kind = "Series" if groupby_series else "DataFrame"
        warn_msg = f"{kind}GroupBy.fillna is deprecated"
    else:
        warn_msg = ""
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg)


@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_string_udf(how, by, groupby_series, df_with_string_col):
    df = df_with_string_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    def func(x):
        raise TypeError("Test error message")

    with pytest.raises(TypeError, match="Test error message"):
        getattr(gb, how)(func)


@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_string_np(
    how,
    by,
    groupby_series,
    groupby_func_np,
    df_with_string_col,
    using_infer_string,
):
    # GH#50749
    df = df_with_string_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    klass, msg = {
        np.sum: (None, ""),
        np.mean: (
            TypeError,
            "agg function failed|Cannot perform reduction 'mean' with string dtype",
        ),
    }[groupby_func_np]

    if using_infer_string:
        if groupby_func_np is np.mean:
            klass = TypeError
        msg = "dtype 'str' does not support operation 'mean'"

    if groupby_series:
        warn_msg = "using SeriesGroupBy.[sum|mean]"
    else:
        warn_msg = "using DataFrameGroupBy.[sum|mean]"
    _call_and_check(klass, msg, how, gb, groupby_func_np, (), warn_msg=warn_msg)


@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_datetime(
    how, by, groupby_series, groupby_func, df_with_datetime_col
):
    df = df_with_datetime_col
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

        if groupby_func == "corrwith":
            assert not hasattr(gb, "corrwith")
            return

    klass, msg = {
        "all": (None, ""),
        "any": (None, ""),
        "bfill": (None, ""),
        "corrwith": (TypeError, "cannot perform __mul__ with this index type"),
        "count": (None, ""),
        "cumcount": (None, ""),
        "cummax": (None, ""),
        "cummin": (None, ""),
        "cumprod": (TypeError, "datetime64 type does not support cumprod operations"),
        "cumsum": (TypeError, "datetime64 type does not support cumsum operations"),
        "diff": (None, ""),
        "ffill": (None, ""),
        "fillna": (None, ""),
        "first": (None, ""),
        "idxmax": (None, ""),
        "idxmin": (None, ""),
        "last": (None, ""),
        "max": (None, ""),
        "mean": (None, ""),
        "median": (None, ""),
        "min": (None, ""),
        "ngroup": (None, ""),
        "nunique": (None, ""),
        "pct_change": (TypeError, "cannot perform __truediv__ with this index type"),
        "prod": (TypeError, "datetime64 type does not support prod"),
        "quantile": (None, ""),
        "rank": (None, ""),
        "sem": (None, ""),
        "shift": (None, ""),
        "size": (None, ""),
        "skew": (
            TypeError,
            "|".join(
                [
                    r"dtype datetime64\[ns\] does not support reduction",
                    "datetime64 type does not support skew operations",
                ]
            ),
        ),
        "std": (None, ""),
        "sum": (TypeError, "datetime64 type does not support sum operations"),
        "var": (TypeError, "datetime64 type does not support var operations"),
    }[groupby_func]

    if groupby_func in ["any", "all"]:
        warn_msg = f"'{groupby_func}' with datetime64 dtypes is deprecated"
    elif groupby_func == "fillna":
        kind = "Series" if groupby_series else "DataFrame"
        warn_msg = f"{kind}GroupBy.fillna is deprecated"
    else:
        warn_msg = ""
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg=warn_msg)


@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_datetime_udf(how, by, groupby_series, df_with_datetime_col):
    df = df_with_datetime_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    def func(x):
        raise TypeError("Test error message")

    with pytest.raises(TypeError, match="Test error message"):
        getattr(gb, how)(func)


@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_datetime_np(
    how, by, groupby_series, groupby_func_np, df_with_datetime_col
):
    # GH#50749
    df = df_with_datetime_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    klass, msg = {
        np.sum: (TypeError, "datetime64 type does not support sum operations"),
        np.mean: (None, ""),
    }[groupby_func_np]

    if groupby_series:
        warn_msg = "using SeriesGroupBy.[sum|mean]"
    else:
        warn_msg = "using DataFrameGroupBy.[sum|mean]"
    _call_and_check(klass, msg, how, gb, groupby_func_np, (), warn_msg=warn_msg)


@pytest.mark.parametrize("func", ["prod", "cumprod", "skew", "var"])
def test_groupby_raises_timedelta(func, df_with_timedelta_col):
    df = df_with_timedelta_col
    gb = df.groupby(by="a")

    _call_and_check(
        TypeError,
        "timedelta64 type does not support .* operations",
        "method",
        gb,
        func,
        [],
    )


@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_category(
    how, by, groupby_series, groupby_func, using_copy_on_write, df_with_cat_col
):
    # GH#50749
    df = df_with_cat_col
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

        if groupby_func == "corrwith":
            assert not hasattr(gb, "corrwith")
            return

    klass, msg = {
        "all": (None, ""),
        "any": (None, ""),
        "bfill": (None, ""),
        "corrwith": (
            TypeError,
            r"unsupported operand type\(s\) for \*: 'Categorical' and 'int'",
        ),
        "count": (None, ""),
        "cumcount": (None, ""),
        "cummax": (
            (NotImplementedError, TypeError),
            "(category type does not support cummax operations|"
            "category dtype not supported|"
            "cummax is not supported for category dtype)",
        ),
        "cummin": (
            (NotImplementedError, TypeError),
            "(category type does not support cummin operations|"
            "category dtype not supported|"
            "cummin is not supported for category dtype)",
        ),
        "cumprod": (
            (NotImplementedError, TypeError),
            "(category type does not support cumprod operations|"
            "category dtype not supported|"
            "cumprod is not supported for category dtype)",
        ),
        "cumsum": (
            (NotImplementedError, TypeError),
            "(category type does not support cumsum operations|"
            "category dtype not supported|"
            "cumsum is not supported for category dtype)",
        ),
        "diff": (
            TypeError,
            r"unsupported operand type\(s\) for -: 'Categorical' and 'Categorical'",
        ),
        "ffill": (None, ""),
        "fillna": (
            TypeError,
            r"Cannot setitem on a Categorical with a new category \(0\), "
            "set the categories first",
        )
        if not using_copy_on_write
        else (None, ""),  # no-op with CoW
        "first": (None, ""),
        "idxmax": (None, ""),
        "idxmin": (None, ""),
        "last": (None, ""),
        "max": (None, ""),
        "mean": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support reduction 'mean'",
                    "category dtype does not support aggregation 'mean'",
                ]
            ),
        ),
        "median": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support reduction 'median'",
                    "category dtype does not support aggregation 'median'",
                ]
            ),
        ),
        "min": (None, ""),
        "ngroup": (None, ""),
        "nunique": (None, ""),
        "pct_change": (
            TypeError,
            r"unsupported operand type\(s\) for /: 'Categorical' and 'Categorical'",
        ),
        "prod": (TypeError, "category type does not support prod operations"),
        "quantile": (TypeError, "No matching signature found"),
        "rank": (None, ""),
        "sem": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support reduction 'sem'",
                    "category dtype does not support aggregation 'sem'",
                ]
            ),
        ),
        "shift": (None, ""),
        "size": (None, ""),
        "skew": (
            TypeError,
            "|".join(
                [
                    "dtype category does not support reduction 'skew'",
                    "category type does not support skew operations",
                ]
            ),
        ),
        "std": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support reduction 'std'",
                    "category dtype does not support aggregation 'std'",
                ]
            ),
        ),
        "sum": (TypeError, "category type does not support sum operations"),
        "var": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support reduction 'var'",
                    "category dtype does not support aggregation 'var'",
                ]
            ),
        ),
    }[groupby_func]

    if groupby_func == "fillna":
        kind = "Series" if groupby_series else "DataFrame"
        warn_msg = f"{kind}GroupBy.fillna is deprecated"
    else:
        warn_msg = ""
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg)


@pytest.mark.parametrize("how", ["agg", "transform"])
def test_groupby_raises_category_udf(how, by, groupby_series, df_with_cat_col):
    # GH#50749
    df = df_with_cat_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    def func(x):
        raise TypeError("Test error message")

    with pytest.raises(TypeError, match="Test error message"):
        getattr(gb, how)(func)


@pytest.mark.parametrize("how", ["agg", "transform"])
@pytest.mark.parametrize("groupby_func_np", [np.sum, np.mean])
def test_groupby_raises_category_np(
    how, by, groupby_series, groupby_func_np, df_with_cat_col
):
    # GH#50749
    df = df_with_cat_col
    gb = df.groupby(by=by)

    if groupby_series:
        gb = gb["d"]

    klass, msg = {
        np.sum: (TypeError, "category type does not support sum operations"),
        np.mean: (
            TypeError,
            "category dtype does not support aggregation 'mean'",
        ),
    }[groupby_func_np]

    if groupby_series:
        warn_msg = "using SeriesGroupBy.[sum|mean]"
    else:
        warn_msg = "using DataFrameGroupBy.[sum|mean]"
    _call_and_check(klass, msg, how, gb, groupby_func_np, (), warn_msg=warn_msg)


@pytest.mark.parametrize("how", ["method", "agg", "transform"])
def test_groupby_raises_category_on_category(
    how,
    by,
    groupby_series,
    groupby_func,
    observed,
    using_copy_on_write,
    df_with_cat_col,
):
    # GH#50749
    df = df_with_cat_col
    df["a"] = Categorical(
        ["a", "a", "a", "a", "b", "b", "b", "b", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby(by=by, observed=observed)

    if groupby_series:
        gb = gb["d"]

        if groupby_func == "corrwith":
            assert not hasattr(gb, "corrwith")
            return

    empty_groups = not observed and any(group.empty for group in gb.groups.values())
    if (
        not observed
        and how != "transform"
        and isinstance(by, list)
        and isinstance(by[0], str)
        and by == ["a", "b"]
    ):
        assert not empty_groups
        # TODO: empty_groups should be true due to unobserved categorical combinations
        empty_groups = True
    if how == "transform":
        # empty groups will be ignored
        empty_groups = False

    klass, msg = {
        "all": (None, ""),
        "any": (None, ""),
        "bfill": (None, ""),
        "corrwith": (
            TypeError,
            r"unsupported operand type\(s\) for \*: 'Categorical' and 'int'",
        ),
        "count": (None, ""),
        "cumcount": (None, ""),
        "cummax": (
            (NotImplementedError, TypeError),
            "(cummax is not supported for category dtype|"
            "category dtype not supported|"
            "category type does not support cummax operations)",
        ),
        "cummin": (
            (NotImplementedError, TypeError),
            "(cummin is not supported for category dtype|"
            "category dtype not supported|"
            "category type does not support cummin operations)",
        ),
        "cumprod": (
            (NotImplementedError, TypeError),
            "(cumprod is not supported for category dtype|"
            "category dtype not supported|"
            "category type does not support cumprod operations)",
        ),
        "cumsum": (
            (NotImplementedError, TypeError),
            "(cumsum is not supported for category dtype|"
            "category dtype not supported|"
            "category type does not support cumsum operations)",
        ),
        "diff": (TypeError, "unsupported operand type"),
        "ffill": (None, ""),
        "fillna": (
            TypeError,
            r"Cannot setitem on a Categorical with a new category \(0\), "
            "set the categories first",
        )
        if not using_copy_on_write
        else (None, ""),  # no-op with CoW
        "first": (None, ""),
        "idxmax": (ValueError, "empty group due to unobserved categories")
        if empty_groups
        else (None, ""),
        "idxmin": (ValueError, "empty group due to unobserved categories")
        if empty_groups
        else (None, ""),
        "last": (None, ""),
        "max": (None, ""),
        "mean": (TypeError, "category dtype does not support aggregation 'mean'"),
        "median": (TypeError, "category dtype does not support aggregation 'median'"),
        "min": (None, ""),
        "ngroup": (None, ""),
        "nunique": (None, ""),
        "pct_change": (TypeError, "unsupported operand type"),
        "prod": (TypeError, "category type does not support prod operations"),
        "quantile": (TypeError, "No matching signature found"),
        "rank": (None, ""),
        "sem": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support reduction 'sem'",
                    "category dtype does not support aggregation 'sem'",
                ]
            ),
        ),
        "shift": (None, ""),
        "size": (None, ""),
        "skew": (
            TypeError,
            "|".join(
                [
                    "category type does not support skew operations",
                    "dtype category does not support reduction 'skew'",
                ]
            ),
        ),
        "std": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support reduction 'std'",
                    "category dtype does not support aggregation 'std'",
                ]
            ),
        ),
        "sum": (TypeError, "category type does not support sum operations"),
        "var": (
            TypeError,
            "|".join(
                [
                    "'Categorical' .* does not support reduction 'var'",
                    "category dtype does not support aggregation 'var'",
                ]
            ),
        ),
    }[groupby_func]

    if groupby_func == "fillna":
        kind = "Series" if groupby_series else "DataFrame"
        warn_msg = f"{kind}GroupBy.fillna is deprecated"
    else:
        warn_msg = ""
    _call_and_check(klass, msg, how, gb, groupby_func, args, warn_msg)


def test_subsetting_columns_axis_1_raises():
    # GH 35443
    df = DataFrame({"a": [1], "b": [2], "c": [3]})
    msg = "DataFrame.groupby with axis=1 is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby("a", axis=1)
    with pytest.raises(ValueError, match="Cannot subset columns when using axis=1"):
        gb["b"]