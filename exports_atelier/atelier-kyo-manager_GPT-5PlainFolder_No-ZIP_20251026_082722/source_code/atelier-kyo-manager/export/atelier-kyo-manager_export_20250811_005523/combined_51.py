
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\bessel.py ===
from functools import wraps

from sympy.core import S
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import DefinedFunction, ArgumentIndexError, _mexpand
from sympy.core.logic import fuzzy_or, fuzzy_not
from sympy.core.numbers import Rational, pi, I
from sympy.core.power import Pow
from sympy.core.symbol import Dummy, uniquely_named_symbol, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.elementary.trigonometric import sin, cos, csc, cot
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import cbrt, sqrt, root
from sympy.functions.elementary.complexes import (Abs, re, im, polar_lift, unpolarify)
from sympy.functions.special.gamma_functions import gamma, digamma, uppergamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import spherical_bessel_fn

from mpmath import mp, workprec

# TODO
# o Scorer functions G1 and G2
# o Asymptotic expansions
#   These are possible, e.g. for fixed order, but since the bessel type
#   functions are oscillatory they are not actually tractable at
#   infinity, so this is not particularly useful right now.
# o Nicer series expansions.
# o More rewriting.
# o Add solvers to ode.py (or rather add solvers for the hypergeometric equation).


class BesselBase(DefinedFunction):
    """
    Abstract base class for Bessel-type functions.

    This class is meant to reduce code duplication.
    All Bessel-type functions can 1) be differentiated, with the derivatives
    expressed in terms of similar functions, and 2) be rewritten in terms
    of other Bessel-type functions.

    Here, Bessel-type functions are assumed to have one complex parameter.

    To use this base class, define class attributes ``_a`` and ``_b`` such that
    ``2*F_n' = -_a*F_{n+1} + b*F_{n-1}``.

    """

    @property
    def order(self):
        """ The order of the Bessel-type function. """
        return self.args[0]

    @property
    def argument(self):
        """ The argument of the Bessel-type function. """
        return self.args[1]

    @classmethod
    def eval(cls, nu, z):
        return

    def fdiff(self, argindex=2):
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        return (self._b/2 * self.__class__(self.order - 1, self.argument) -
                self._a/2 * self.__class__(self.order + 1, self.argument))

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return self.__class__(self.order.conjugate(), z.conjugate())

    def _eval_is_meromorphic(self, x, a):
        nu, z = self.order, self.argument

        if nu.has(x):
            return False
        if not z._eval_is_meromorphic(x, a):
            return None
        z0 = z.subs(x, a)
        if nu.is_integer:
            if isinstance(self, (besselj, besseli, hn1, hn2, jn, yn)) or not nu.is_zero:
                return fuzzy_not(z0.is_infinite)
        return fuzzy_not(fuzzy_or([z0.is_zero, z0.is_infinite]))

    def _eval_expand_func(self, **hints):
        nu, z, f = self.order, self.argument, self.__class__
        if nu.is_real:
            if (nu - 1).is_positive:
                return (-self._a*self._b*f(nu - 2, z)._eval_expand_func() +
                        2*self._a*(nu - 1)*f(nu - 1, z)._eval_expand_func()/z)
            elif (nu + 1).is_negative:
                return (2*self._b*(nu + 1)*f(nu + 1, z)._eval_expand_func()/z -
                        self._a*self._b*f(nu + 2, z)._eval_expand_func())
        return self

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import besselsimp
        return besselsimp(self)


class besselj(BesselBase):
    r"""
    Bessel function of the first kind.

    Explanation
    ===========

    The Bessel $J$ function of order $\nu$ is defined to be the function
    satisfying Bessel's differential equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
        + z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 - \nu^2) w = 0,

    with Laurent expansion

    .. math ::
        J_\nu(z) = z^\nu \left(\frac{1}{\Gamma(\nu + 1) 2^\nu} + O(z^2) \right),

    if $\nu$ is not a negative integer. If $\nu=-n \in \mathbb{Z}_{<0}$
    *is* a negative integer, then the definition is

    .. math ::
        J_{-n}(z) = (-1)^n J_n(z).

    Examples
    ========

    Create a Bessel function object:

    >>> from sympy import besselj, jn
    >>> from sympy.abc import z, n
    >>> b = besselj(n, z)

    Differentiate it:

    >>> b.diff(z)
    besselj(n - 1, z)/2 - besselj(n + 1, z)/2

    Rewrite in terms of spherical Bessel functions:

    >>> b.rewrite(jn)
    sqrt(2)*sqrt(z)*jn(n - 1/2, z)/sqrt(pi)

    Access the parameter and argument:

    >>> b.order
    n
    >>> b.argument
    z

    See Also
    ========

    bessely, besseli, besselk

    References
    ==========

    .. [1] Abramowitz, Milton; Stegun, Irene A., eds. (1965), "Chapter 9",
           Handbook of Mathematical Functions with Formulas, Graphs, and
           Mathematical Tables
    .. [2] Luke, Y. L. (1969), The Special Functions and Their
           Approximations, Volume 1
    .. [3] https://en.wikipedia.org/wiki/Bessel_function
    .. [4] https://functions.wolfram.com/Bessel-TypeFunctions/BesselJ/

    """

    _a = S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif (nu.is_integer and nu.is_zero is False) or re(nu).is_positive:
                return S.Zero
            elif re(nu).is_negative and not (nu.is_integer is True):
                return S.ComplexInfinity
            elif nu.is_imaginary:
                return S.NaN
        if z in (S.Infinity, S.NegativeInfinity):
            return S.Zero

        if z.could_extract_minus_sign():
            return (z)**nu*(-z)**(-nu)*besselj(nu, -z)
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return S.NegativeOne**(-nu)*besselj(-nu, z)
            newz = z.extract_multiplicatively(I)
            if newz:  # NOTE we don't want to change the function if z==0
                return I**(nu)*besseli(nu, newz)

        # branch handling:
        if nu.is_integer:
            newz = unpolarify(z)
            if newz != z:
                return besselj(nu, newz)
        else:
            newz, n = z.extract_branch_factor()
            if n != 0:
                return exp(2*n*pi*nu*I)*besselj(nu, newz)
        nnu = unpolarify(nu)
        if nu != nnu:
            return besselj(nnu, z)

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        return exp(I*pi*nu/2)*besseli(nu, polar_lift(-I)*z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return csc(pi*nu)*bessely(-nu, z) - cot(pi*nu)*bessely(nu, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return sqrt(2*z/pi)*jn(nu - S.Half, self.argument)

    def _eval_as_leading_term(self, x, logx, cdir):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            return arg**nu/(2**nu*gamma(nu + 1))
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c*cdir**e
            if not sign.is_negative:
                # Refer Abramowitz and Stegun 1965, p. 364 for more information on
                # asymptotic approximation of besselj function.
                return sqrt(2)*cos(z - pi*(2*nu + 1)/4)/sqrt(pi*z)
            return self

        return super(besselj, self)._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_extended_real:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        # Refer https://functions.wolfram.com/Bessel-TypeFunctions/BesselJ/06/01/04/01/01/0003/
        # for more information on nseries expansion of besselj function.
        from sympy.series.order import Order
        nu, z = self.args

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive:
            newn = ceiling(n/exp)
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            term = r**nu/gamma(nu + 1)
            s = [term]
            for k in range(1, (newn + 1)//2):
                term *= -t/(k*(nu + k))
                term = (_mexpand(term) + o).removeO()
                s.append(term)
            return Add(*s) + o

        return super(besselj, self)._eval_nseries(x, n, logx, cdir)


class bessely(BesselBase):
    r"""
    Bessel function of the second kind.

    Explanation
    ===========

    The Bessel $Y$ function of order $\nu$ is defined as

    .. math ::
        Y_\nu(z) = \lim_{\mu \to \nu} \frac{J_\mu(z) \cos(\pi \mu)
                                            - J_{-\mu}(z)}{\sin(\pi \mu)},

    where $J_\mu(z)$ is the Bessel function of the first kind.

    It is a solution to Bessel's equation, and linearly independent from
    $J_\nu$.

    Examples
    ========

    >>> from sympy import bessely, yn
    >>> from sympy.abc import z, n
    >>> b = bessely(n, z)
    >>> b.diff(z)
    bessely(n - 1, z)/2 - bessely(n + 1, z)/2
    >>> b.rewrite(yn)
    sqrt(2)*sqrt(z)*yn(n - 1/2, z)/sqrt(pi)

    See Also
    ========

    besselj, besseli, besselk

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselY/

    """

    _a = S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.NegativeInfinity
            elif re(nu).is_zero is False:
                return S.ComplexInfinity
            elif re(nu).is_zero:
                return S.NaN
        if z in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        if z == I*S.Infinity:
            return exp(I*pi*(nu + 1)/2) * S.Infinity
        if z == I*S.NegativeInfinity:
            return exp(-I*pi*(nu + 1)/2) * S.Infinity

        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return S.NegativeOne**(-nu)*bessely(-nu, z)

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return csc(pi*nu)*(cos(pi*nu)*besselj(nu, z) - besselj(-nu, z))

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(besseli)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        return sqrt(2*z/pi) * yn(nu - S.Half, self.argument)

    def _eval_as_leading_term(self, x, logx, cdir):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            term_one = ((2/pi)*log(z/2)*besselj(nu, z))
            term_two = -(z/2)**(-nu)*factorial(nu - 1)/pi if (nu).is_positive else S.Zero
            term_three = -(z/2)**nu/(pi*factorial(nu))*(digamma(nu + 1) - S.EulerGamma)
            arg = Add(*[term_one, term_two, term_three]).as_leading_term(x, logx=logx)
            return arg
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c*cdir**e
            if not sign.is_negative:
                # Refer Abramowitz and Stegun 1965, p. 364 for more information on
                # asymptotic approximation of bessely function.
                return sqrt(2)*(-sin(pi*nu/2 - z + pi/4) + 3*cos(pi*nu/2 - z + pi/4)/(8*z))*sqrt(1/z)/sqrt(pi)
            return self

        return super(bessely, self)._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_positive:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        # Refer https://functions.wolfram.com/Bessel-TypeFunctions/BesselY/06/01/04/01/02/0008/
        # for more information on nseries expansion of bessely function.
        from sympy.series.order import Order
        nu, z = self.args

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive and nu.is_integer:
            newn = ceiling(n/exp)
            bn = besselj(nu, z)
            a = ((2/pi)*log(z/2)*bn)._eval_nseries(x, n, logx, cdir)

            b, c = [], []
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            if nu > S.Zero:
                term = r**(-nu)*factorial(nu - 1)/pi
                b.append(term)
                for k in range(1, nu):
                    denom = (nu - k)*k
                    if denom == S.Zero:
                        term *= t/k
                    else:
                        term *= t/denom
                    term = (_mexpand(term) + o).removeO()
                    b.append(term)

            p = r**nu/(pi*factorial(nu))
            term = p*(digamma(nu + 1) - S.EulerGamma)
            c.append(term)
            for k in range(1, (newn + 1)//2):
                p *= -t/(k*(k + nu))
                p = (_mexpand(p) + o).removeO()
                term = p*(digamma(k + nu + 1) + digamma(k + 1))
                c.append(term)
            return a - Add(*b) - Add(*c) # Order term comes from a

        return super(bessely, self)._eval_nseries(x, n, logx, cdir)


class besseli(BesselBase):
    r"""
    Modified Bessel function of the first kind.

    Explanation
    ===========

    The Bessel $I$ function is a solution to the modified Bessel equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
        + z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 + \nu^2)^2 w = 0.

    It can be defined as

    .. math ::
        I_\nu(z) = i^{-\nu} J_\nu(iz),

    where $J_\nu(z)$ is the Bessel function of the first kind.

    Examples
    ========

    >>> from sympy import besseli
    >>> from sympy.abc import z, n
    >>> besseli(n, z).diff(z)
    besseli(n - 1, z)/2 + besseli(n + 1, z)/2

    See Also
    ========

    besselj, bessely, besselk

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/

    """

    _a = -S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif (nu.is_integer and nu.is_zero is False) or re(nu).is_positive:
                return S.Zero
            elif re(nu).is_negative and not (nu.is_integer is True):
                return S.ComplexInfinity
            elif nu.is_imaginary:
                return S.NaN
        if im(z) in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        if z is S.Infinity:
            return S.Infinity
        if z is S.NegativeInfinity:
            return (-1)**nu*S.Infinity

        if z.could_extract_minus_sign():
            return (z)**nu*(-z)**(-nu)*besseli(nu, -z)
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besseli(-nu, z)
            newz = z.extract_multiplicatively(I)
            if newz:  # NOTE we don't want to change the function if z==0
                return I**(-nu)*besselj(nu, -newz)

        # branch handling:
        if nu.is_integer:
            newz = unpolarify(z)
            if newz != z:
                return besseli(nu, newz)
        else:
            newz, n = z.extract_branch_factor()
            if n != 0:
                return exp(2*n*pi*nu*I)*besseli(nu, newz)
        nnu = unpolarify(nu)
        if nu != nnu:
            return besseli(nnu, z)

    def _eval_rewrite_as_tractable(self, nu, z, limitvar=None, **kwargs):
        if z.is_extended_real:
            return exp(z)*_besseli(nu, z)

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return exp(-I*pi*nu/2)*besselj(nu, polar_lift(I)*z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(bessely)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return self._eval_rewrite_as_besselj(*self.args).rewrite(jn)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_extended_real:
            return True

    def _eval_as_leading_term(self, x, logx, cdir):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            return arg**nu/(2**nu*gamma(nu + 1))
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c*cdir**e
            if not sign.is_negative:
                # Refer Abramowitz and Stegun 1965, p. 377 for more information on
                # asymptotic approximation of besseli function.
                return exp(z)/sqrt(2*pi*z)
            return self

        return super(besseli, self)._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # Refer https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/06/01/04/01/01/0003/
        # for more information on nseries expansion of besseli function.
        from sympy.series.order import Order
        nu, z = self.args

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive:
            newn = ceiling(n/exp)
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            term = r**nu/gamma(nu + 1)
            s = [term]
            for k in range(1, (newn + 1)//2):
                term *= t/(k*(nu + k))
                term = (_mexpand(term) + o).removeO()
                s.append(term)
            return Add(*s) + o

        return super(besseli, self)._eval_nseries(x, n, logx, cdir)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.functions.combinatorial.factorials import RisingFactorial
        from sympy.series.order import Order
        point = args0[1]

        if point in [S.Infinity, S.NegativeInfinity]:
            nu, z = self.args
            s = [(RisingFactorial(Rational(2*nu - 1, 2), k)*RisingFactorial(Rational(2*nu + 1, 2), k))/\
            ((2)**(k)*z**(Rational(2*k + 1, 2))*factorial(k)) for k in range(n)] + [Order(1/z**(Rational(2*n + 1, 2)), x)]
            return exp(z)/sqrt(2*pi) * (Add(*s))

        return super()._eval_aseries(n, args0, x, logx)


class besselk(BesselBase):
    r"""
    Modified Bessel function of the second kind.

    Explanation
    ===========

    The Bessel $K$ function of order $\nu$ is defined as

    .. math ::
        K_\nu(z) = \lim_{\mu \to \nu} \frac{\pi}{2}
                   \frac{I_{-\mu}(z) -I_\mu(z)}{\sin(\pi \mu)},

    where $I_\mu(z)$ is the modified Bessel function of the first kind.

    It is a solution of the modified Bessel equation, and linearly independent
    from $Y_\nu$.

    Examples
    ========

    >>> from sympy import besselk
    >>> from sympy.abc import z, n
    >>> besselk(n, z).diff(z)
    -besselk(n - 1, z)/2 - besselk(n + 1, z)/2

    See Also
    ========

    besselj, besseli, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/

    """

    _a = S.One
    _b = -S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.Infinity
            elif re(nu).is_zero is False:
                return S.ComplexInfinity
            elif re(nu).is_zero:
                return S.NaN
        if z in (S.Infinity, I*S.Infinity, I*S.NegativeInfinity):
            return S.Zero

        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besselk(-nu, z)

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return pi*csc(pi*nu)*(besseli(-nu, z) - besseli(nu, z))/2

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        ai = self._eval_rewrite_as_besseli(*self.args)
        if ai:
            return ai.rewrite(besselj)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(bessely)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        ay = self._eval_rewrite_as_bessely(*self.args)
        if ay:
            return ay.rewrite(yn)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_positive:
            return True

    def _eval_rewrite_as_tractable(self, nu, z, limitvar=None, **kwargs):
        if z.is_extended_real:
            return exp(-z)*_besselk(nu, z)

    def _eval_as_leading_term(self, x, logx, cdir):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        _, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            if nu.is_zero:
                # Equation 9.6.8 of Abramowitz and Stegun (10th ed, 1972).
                term = -log(z) - S.EulerGamma + log(2)
            elif nu.is_nonzero:
                # Equation 9.6.9 of Abramowitz and Stegun (10th ed, 1972).
                term = gamma(Abs(nu))*(z/2)**(-Abs(nu))/2
            else:
                raise NotImplementedError(f"Cannot proceed without knowing if {nu} is zero or not.")

            return term.as_leading_term(x, logx=logx)
        elif e.is_negative:
            # Equation 9.7.2 of Abramowitz and Stegun (10th ed, 1972).
            return sqrt(pi)*exp(-arg)/sqrt(2*arg)
        else:
            return self.func(nu, arg)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        nu, z = self.args

        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        if exp.is_positive:
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return Order(z**(-nu) + z**nu, x)

            o = Order(x**n, x)
            if nu.is_integer:
                # Reference: https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/06/01/04/01/02/0008/ (only for integer order)
                newn = ceiling(n/exp)
                bn = besseli(nu, z)
                a = ((-1)**(nu - 1)*log(z/2)*bn)._eval_nseries(x, n, logx, cdir)

                b, c = [], []
                t = _mexpand(r**2)

                if nu > S.Zero:
                    term = r**(-nu)*factorial(nu - 1)/2
                    b.append(term)
                    for k in range(1, nu):
                        term *= t/((k - nu)*k)
                        term = (_mexpand(term) + o).removeO()
                        b.append(term)

                p = r**nu*(-1)**nu/(2*factorial(nu))
                term = p*(digamma(nu + 1) - S.EulerGamma)
                c.append(term)
                for k in range(1, (newn + 1)//2):
                    p *= t/(k*(k + nu))
                    p = (_mexpand(p) + o).removeO()
                    term = p*(digamma(k + nu + 1) + digamma(k + 1))
                    c.append(term)
                return a + Add(*b) + Add(*c) + o
            elif nu.is_noninteger:
                # Reference: https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/06/01/04/01/01/0003/
                # (only for non-integer order).
                # While the expression in the reference above seems correct
                # for non-real order as well, it would need some manipulation
                # (not implemented) to be written as a power series in x with
                # real exponents [e.g. Dunster 1990. "Bessel functions
                # of purely imaginary order, with an application to second-order
                # linear differential equations having a large parameter".
                # SIAM J. Math. Anal. Vol 21, No. 4, pp 995-1018.].
                newn_a = ceiling((n+nu)/exp)
                newn_b = ceiling((n-nu)/exp)

                a, b = [], []
                for k in range((newn_a+1)//2):
                    term = gamma(nu)*r**(2*k-nu)/(2*RisingFactorial(1-nu, k)*factorial(k))
                    a.append(_mexpand(term))
                for k in range((newn_b+1)//2):
                    term = gamma(-nu)*r**(2*k+nu)/(2*RisingFactorial(nu+1, k)*factorial(k))
                    b.append(_mexpand(term))
                return Add(*a) + Add(*b) + o
            else:
                raise NotImplementedError("besselk expansion is only implemented for real order")

        return super(besselk, self)._eval_nseries(x, n, logx, cdir)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.functions.combinatorial.factorials import RisingFactorial
        from sympy.series.order import Order
        point = args0[1]

        if point in [S.Infinity, S.NegativeInfinity]:
            nu, z = self.args
            s = [(RisingFactorial(Rational(2*nu - 1, 2), k)*RisingFactorial(Rational(2*nu + 1, 2), k))/\
            ((-2)**(k)*z**(Rational(2*k + 1, 2))*factorial(k)) for k in range(n)] +[Order(1/z**(Rational(2*n + 1, 2)), x)]
            return (exp(-z)*sqrt(pi/2))*Add(*s)

        return super()._eval_aseries(n, args0, x, logx)


class hankel1(BesselBase):
    r"""
    Hankel function of the first kind.

    Explanation
    ===========

    This function is defined as

    .. math ::
        H_\nu^{(1)} = J_\nu(z) + iY_\nu(z),

    where $J_\nu(z)$ is the Bessel function of the first kind, and
    $Y_\nu(z)$ is the Bessel function of the second kind.

    It is a solution to Bessel's equation.

    Examples
    ========

    >>> from sympy import hankel1
    >>> from sympy.abc import z, n
    >>> hankel1(n, z).diff(z)
    hankel1(n - 1, z)/2 - hankel1(n + 1, z)/2

    See Also
    ========

    hankel2, besselj, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/HankelH1/

    """

    _a = S.One
    _b = S.One

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return hankel2(self.order.conjugate(), z.conjugate())


class hankel2(BesselBase):
    r"""
    Hankel function of the second kind.

    Explanation
    ===========

    This function is defined as

    .. math ::
        H_\nu^{(2)} = J_\nu(z) - iY_\nu(z),

    where $J_\nu(z)$ is the Bessel function of the first kind, and
    $Y_\nu(z)$ is the Bessel function of the second kind.

    It is a solution to Bessel's equation, and linearly independent from
    $H_\nu^{(1)}$.

    Examples
    ========

    >>> from sympy import hankel2
    >>> from sympy.abc import z, n
    >>> hankel2(n, z).diff(z)
    hankel2(n - 1, z)/2 - hankel2(n + 1, z)/2

    See Also
    ========

    hankel1, besselj, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/HankelH2/

    """

    _a = S.One
    _b = S.One

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return hankel1(self.order.conjugate(), z.conjugate())


def assume_integer_order(fn):
    @wraps(fn)
    def g(self, nu, z):
        if nu.is_integer:
            return fn(self, nu, z)
    return g


class SphericalBesselBase(BesselBase):
    """
    Base class for spherical Bessel functions.

    These are thin wrappers around ordinary Bessel functions,
    since spherical Bessel functions differ from the ordinary
    ones just by a slight change in order.

    To use this class, define the ``_eval_evalf()`` and ``_expand()`` methods.

    """

    def _expand(self, **hints):
        """ Expand self into a polynomial. Nu is guaranteed to be Integer. """
        raise NotImplementedError('expansion')

    def _eval_expand_func(self, **hints):
        if self.order.is_Integer:
            return self._expand(**hints)
        return self

    def fdiff(self, argindex=2):
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        return self.__class__(self.order - 1, self.argument) - \
            self * (self.order + 1)/self.argument


def _jn(n, z):
    return (spherical_bessel_fn(n, z)*sin(z) +
            S.NegativeOne**(n + 1)*spherical_bessel_fn(-n - 1, z)*cos(z))


def _yn(n, z):
    # (-1)**(n + 1) * _jn(-n - 1, z)
    return (S.NegativeOne**(n + 1) * spherical_bessel_fn(-n - 1, z)*sin(z) -
            spherical_bessel_fn(n, z)*cos(z))


class jn(SphericalBesselBase):
    r"""
    Spherical Bessel function of the first kind.

    Explanation
    ===========

    This function is a solution to the spherical Bessel equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
          + 2z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 - \nu(\nu + 1)) w = 0.

    It can be defined as

    .. math ::
        j_\nu(z) = \sqrt{\frac{\pi}{2z}} J_{\nu + \frac{1}{2}}(z),

    where $J_\nu(z)$ is the Bessel function of the first kind.

    The spherical Bessel functions of integral order are
    calculated using the formula:

    .. math:: j_n(z) = f_n(z) \sin{z} + (-1)^{n+1} f_{-n-1}(z) \cos{z},

    where the coefficients $f_n(z)$ are available as
    :func:`sympy.polys.orthopolys.spherical_bessel_fn`.

    Examples
    ========

    >>> from sympy import Symbol, jn, sin, cos, expand_func, besselj, bessely
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(jn(0, z)))
    sin(z)/z
    >>> expand_func(jn(1, z)) == sin(z)/z**2 - cos(z)/z
    True
    >>> expand_func(jn(3, z))
    (-6/z**2 + 15/z**4)*sin(z) + (1/z - 15/z**3)*cos(z)
    >>> jn(nu, z).rewrite(besselj)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(nu + 1/2, z)/2
    >>> jn(nu, z).rewrite(bessely)
    (-1)**nu*sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(-nu - 1/2, z)/2
    >>> jn(2, 5.2+0.3j).evalf(20)
    0.099419756723640344491 - 0.054525080242173562897*I

    See Also
    ========

    besselj, bessely, besselk, yn

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """
    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif nu.is_integer:
                if nu.is_positive:
                    return S.Zero
                else:
                    return S.ComplexInfinity
        if z in (S.NegativeInfinity, S.Infinity):
            return S.Zero

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return sqrt(pi/(2*z)) * besselj(nu + S.Half, z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        return S.NegativeOne**nu * sqrt(pi/(2*z)) * bessely(-nu - S.Half, z)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        return S.NegativeOne**(nu) * yn(-nu - 1, z)

    def _expand(self, **hints):
        return _jn(self.order, self.argument)

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(besselj)._eval_evalf(prec)


class yn(SphericalBesselBase):
    r"""
    Spherical Bessel function of the second kind.

    Explanation
    ===========

    This function is another solution to the spherical Bessel equation, and
    linearly independent from $j_n$. It can be defined as

    .. math ::
        y_\nu(z) = \sqrt{\frac{\pi}{2z}} Y_{\nu + \frac{1}{2}}(z),

    where $Y_\nu(z)$ is the Bessel function of the second kind.

    For integral orders $n$, $y_n$ is calculated using the formula:

    .. math:: y_n(z) = (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, yn, sin, cos, expand_func, besselj, bessely
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(yn(0, z)))
    -cos(z)/z
    >>> expand_func(yn(1, z)) == -cos(z)/z**2-sin(z)/z
    True
    >>> yn(nu, z).rewrite(besselj)
    (-1)**(nu + 1)*sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(-nu - 1/2, z)/2
    >>> yn(nu, z).rewrite(bessely)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(nu + 1/2, z)/2
    >>> yn(2, 5.2+0.3j).evalf(20)
    0.18525034196069722536 + 0.014895573969924817587*I

    See Also
    ========

    besselj, bessely, besselk, jn

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """
    @assume_integer_order
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return S.NegativeOne**(nu+1) * sqrt(pi/(2*z)) * besselj(-nu - S.Half, z)

    @assume_integer_order
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        return sqrt(pi/(2*z)) * bessely(nu + S.Half, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return S.NegativeOne**(nu + 1) * jn(-nu - 1, z)

    def _expand(self, **hints):
        return _yn(self.order, self.argument)

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(bessely)._eval_evalf(prec)


class SphericalHankelBase(SphericalBesselBase):

    @assume_integer_order
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        # jn +- I*yn
        # jn as beeselj: sqrt(pi/(2*z)) * besselj(nu + S.Half, z)
        # yn as besselj: (-1)**(nu+1) * sqrt(pi/(2*z)) * besselj(-nu - S.Half, z)
        hks = self._hankel_kind_sign
        return sqrt(pi/(2*z))*(besselj(nu + S.Half, z) +
                               hks*I*S.NegativeOne**(nu+1)*besselj(-nu - S.Half, z))

    @assume_integer_order
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        # jn +- I*yn
        # jn as bessely: (-1)**nu * sqrt(pi/(2*z)) * bessely(-nu - S.Half, z)
        # yn as bessely: sqrt(pi/(2*z)) * bessely(nu + S.Half, z)
        hks = self._hankel_kind_sign
        return sqrt(pi/(2*z))*(S.NegativeOne**nu*bessely(-nu - S.Half, z) +
                               hks*I*bessely(nu + S.Half, z))

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        hks = self._hankel_kind_sign
        return jn(nu, z).rewrite(yn) + hks*I*yn(nu, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        hks = self._hankel_kind_sign
        return jn(nu, z) + hks*I*yn(nu, z).rewrite(jn)

    def _eval_expand_func(self, **hints):
        if self.order.is_Integer:
            return self._expand(**hints)
        else:
            nu = self.order
            z = self.argument
            hks = self._hankel_kind_sign
            return jn(nu, z) + hks*I*yn(nu, z)

    def _expand(self, **hints):
        n = self.order
        z = self.argument
        hks = self._hankel_kind_sign

        # fully expanded version
        # return ((fn(n, z) * sin(z) +
        #          (-1)**(n + 1) * fn(-n - 1, z) * cos(z)) +  # jn
        #         (hks * I * (-1)**(n + 1) *
        #          (fn(-n - 1, z) * hk * I * sin(z) +
        #           (-1)**(-n) * fn(n, z) * I * cos(z)))  # +-I*yn
        #         )

        return (_jn(n, z) + hks*I*_yn(n, z)).expand()

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(besselj)._eval_evalf(prec)


class hn1(SphericalHankelBase):
    r"""
    Spherical Hankel function of the first kind.

    Explanation
    ===========

    This function is defined as

    .. math:: h_\nu^(1)(z) = j_\nu(z) + i y_\nu(z),

    where $j_\nu(z)$ and $y_\nu(z)$ are the spherical
    Bessel function of the first and second kinds.

    For integral orders $n$, $h_n^(1)$ is calculated using the formula:

    .. math:: h_n^(1)(z) = j_{n}(z) + i (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, hn1, hankel1, expand_func, yn, jn
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(hn1(nu, z)))
    jn(nu, z) + I*yn(nu, z)
    >>> print(expand_func(hn1(0, z)))
    sin(z)/z - I*cos(z)/z
    >>> print(expand_func(hn1(1, z)))
    -I*sin(z)/z - cos(z)/z + sin(z)/z**2 - I*cos(z)/z**2
    >>> hn1(nu, z).rewrite(jn)
    (-1)**(nu + 1)*I*jn(-nu - 1, z) + jn(nu, z)
    >>> hn1(nu, z).rewrite(yn)
    (-1)**nu*yn(-nu - 1, z) + I*yn(nu, z)
    >>> hn1(nu, z).rewrite(hankel1)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*hankel1(nu, z)/2

    See Also
    ========

    hn2, jn, yn, hankel1, hankel2

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """

    _hankel_kind_sign = S.One

    @assume_integer_order
    def _eval_rewrite_as_hankel1(self, nu, z, **kwargs):
        return sqrt(pi/(2*z))*hankel1(nu, z)


class hn2(SphericalHankelBase):
    r"""
    Spherical Hankel function of the second kind.

    Explanation
    ===========

    This function is defined as

    .. math:: h_\nu^(2)(z) = j_\nu(z) - i y_\nu(z),

    where $j_\nu(z)$ and $y_\nu(z)$ are the spherical
    Bessel function of the first and second kinds.

    For integral orders $n$, $h_n^(2)$ is calculated using the formula:

    .. math:: h_n^(2)(z) = j_{n} - i (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, hn2, hankel2, expand_func, jn, yn
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(hn2(nu, z)))
    jn(nu, z) - I*yn(nu, z)
    >>> print(expand_func(hn2(0, z)))
    sin(z)/z + I*cos(z)/z
    >>> print(expand_func(hn2(1, z)))
    I*sin(z)/z - cos(z)/z + sin(z)/z**2 + I*cos(z)/z**2
    >>> hn2(nu, z).rewrite(hankel2)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*hankel2(nu, z)/2
    >>> hn2(nu, z).rewrite(jn)
    -(-1)**(nu + 1)*I*jn(-nu - 1, z) + jn(nu, z)
    >>> hn2(nu, z).rewrite(yn)
    (-1)**nu*yn(-nu - 1, z) - I*yn(nu, z)

    See Also
    ========

    hn1, jn, yn, hankel1, hankel2

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """

    _hankel_kind_sign = -S.One

    @assume_integer_order
    def _eval_rewrite_as_hankel2(self, nu, z, **kwargs):
        return sqrt(pi/(2*z))*hankel2(nu, z)


def jn_zeros(n, k, method="sympy", dps=15):
    """
    Zeros of the spherical Bessel function of the first kind.

    Explanation
    ===========

    This returns an array of zeros of $jn$ up to the $k$-th zero.

    * method = "sympy": uses `mpmath.besseljzero
      <https://mpmath.org/doc/current/functions/bessel.html#mpmath.besseljzero>`_
    * method = "scipy": uses the
      `SciPy's sph_jn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jn_zeros.html>`_
      and
      `newton <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html>`_
      to find all
      roots, which is faster than computing the zeros using a general
      numerical solver, but it requires SciPy and only works with low
      precision floating point numbers. (The function used with
      method="sympy" is a recent addition to mpmath; before that a general
      solver was used.)

    Examples
    ========

    >>> from sympy import jn_zeros
    >>> jn_zeros(2, 4, dps=5)
    [5.7635, 9.095, 12.323, 15.515]

    See Also
    ========

    jn, yn, besselj, besselk, bessely

    Parameters
    ==========

    n : integer
        order of Bessel function

    k : integer
        number of zeros to return


    """
    from math import pi as math_pi

    if method == "sympy":
        from mpmath import besseljzero
        from mpmath.libmp.libmpf import dps_to_prec
        prec = dps_to_prec(dps)
        return [Expr._from_mpmath(besseljzero(S(n + 0.5)._to_mpmath(prec),
                                              int(l)), prec)
                for l in range(1, k + 1)]
    elif method == "scipy":
        from scipy.optimize import newton
        try:
            from scipy.special import spherical_jn
            f = lambda x: spherical_jn(n, x)
        except ImportError:
            from scipy.special import sph_jn
            f = lambda x: sph_jn(n, x)[0][-1]
    else:
        raise NotImplementedError("Unknown method.")

    def solver(f, x):
        if method == "scipy":
            root = newton(f, x)
        else:
            raise NotImplementedError("Unknown method.")
        return root

    # we need to approximate the position of the first root:
    root = n + math_pi
    # determine the first root exactly:
    root = solver(f, root)
    roots = [root]
    for i in range(k - 1):
        # estimate the position of the next root using the last root + pi:
        root = solver(f, root + math_pi)
        roots.append(root)
    return roots


class AiryBase(DefinedFunction):
    """
    Abstract base class for Airy functions.

    This class is meant to reduce code duplication.

    """

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def as_real_imag(self, deep=True, **hints):
        z = self.args[0]
        zc = z.conjugate()
        f = self.func
        u = (f(z)+f(zc))/2
        v = I*(f(zc)-f(z))/2
        return u, v

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I


class airyai(AiryBase):
    r"""
    The Airy function $\operatorname{Ai}$ of the first kind.

    Explanation
    ===========

    The Airy function $\operatorname{Ai}(z)$ is defined to be the function
    satisfying Airy's differential equation

    .. math::
        \frac{\mathrm{d}^2 w(z)}{\mathrm{d}z^2} - z w(z) = 0.

    Equivalently, for real $z$

    .. math::
        \operatorname{Ai}(z) := \frac{1}{\pi}
        \int_0^\infty \cos\left(\frac{t^3}{3} + z t\right) \mathrm{d}t.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airyai
    >>> from sympy.abc import z

    >>> airyai(z)
    airyai(z)

    Several special values are known:

    >>> airyai(0)
    3**(1/3)/(3*gamma(2/3))
    >>> from sympy import oo
    >>> airyai(oo)
    0
    >>> airyai(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airyai(z))
    airyai(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airyai(z), z)
    airyaiprime(z)
    >>> diff(airyai(z), z, 2)
    z*airyai(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airyai(z), z, 0, 3)
    3**(5/6)*gamma(1/3)/(6*pi) - 3**(1/6)*z*gamma(2/3)/(2*pi) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airyai(-2).evalf(50)
    0.22740742820168557599192443603787379946077222541710

    Rewrite $\operatorname{Ai}(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airyai(z).rewrite(hyper)
    -3**(2/3)*z*hyper((), (4/3,), z**3/9)/(3*gamma(1/3)) + 3**(1/3)*hyper((), (2/3,), z**3/9)/(3*gamma(2/3))

    See Also
    ========

    airybi: Airy function of the second kind.
    airyaiprime: Derivative of the Airy function of the first kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return S.One / (3**Rational(2, 3) * gamma(Rational(2, 3)))
        if arg.is_zero:
            return S.One / (3**Rational(2, 3) * gamma(Rational(2, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return airyaiprime(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return ((cbrt(3)*x)**(-n)*(cbrt(3)*x)**(n + 1)*sin(pi*(n*Rational(2, 3) + Rational(4, 3)))*factorial(n) *
                        gamma(n/3 + Rational(2, 3))/(sin(pi*(n*Rational(2, 3) + Rational(2, 3)))*factorial(n + 1)*gamma(n/3 + Rational(1, 3))) * p)
            else:
                return (S.One/(3**Rational(2, 3)*pi) * gamma((n+S.One)/S(3)) * sin(Rational(2, 3)*pi*(n+S.One)) /
                        factorial(n) * (cbrt(3)*x)**n)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return ot*sqrt(-z) * (besselj(-ot, tt*a) + besselj(ot, tt*a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return ot*sqrt(z) * (besseli(-ot, tt*a) - besseli(ot, tt*a))
        else:
            return ot*(Pow(a, ot)*besseli(-ot, tt*a) - z*Pow(a, -ot)*besseli(ot, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = S.One / (3**Rational(2, 3)*gamma(Rational(2, 3)))
        pf2 = z / (root(3, 3)*gamma(Rational(1, 3)))
        return pf1 * hyper([], [Rational(2, 3)], z**3/9) - pf2 * hyper([], [Rational(4, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is given by 03.05.16.0001.01
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryAi/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d * z**n)**m / (d**m * z**(m*n))
                    newarg = c * d**m * z**(m*n)
                    return S.Half * ((pf + S.One)*airyai(newarg) - (pf - S.One)/sqrt(3)*airybi(newarg))


class airybi(AiryBase):
    r"""
    The Airy function $\operatorname{Bi}$ of the second kind.

    Explanation
    ===========

    The Airy function $\operatorname{Bi}(z)$ is defined to be the function
    satisfying Airy's differential equation

    .. math::
        \frac{\mathrm{d}^2 w(z)}{\mathrm{d}z^2} - z w(z) = 0.

    Equivalently, for real $z$

    .. math::
        \operatorname{Bi}(z) := \frac{1}{\pi}
                 \int_0^\infty
                   \exp\left(-\frac{t^3}{3} + z t\right)
                   + \sin\left(\frac{t^3}{3} + z t\right) \mathrm{d}t.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airybi
    >>> from sympy.abc import z

    >>> airybi(z)
    airybi(z)

    Several special values are known:

    >>> airybi(0)
    3**(5/6)/(3*gamma(2/3))
    >>> from sympy import oo
    >>> airybi(oo)
    oo
    >>> airybi(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airybi(z))
    airybi(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airybi(z), z)
    airybiprime(z)
    >>> diff(airybi(z), z, 2)
    z*airybi(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airybi(z), z, 0, 3)
    3**(1/3)*gamma(1/3)/(2*pi) + 3**(2/3)*z*gamma(2/3)/(2*pi) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airybi(-2).evalf(50)
    -0.41230258795639848808323405461146104203453483447240

    Rewrite $\operatorname{Bi}(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airybi(z).rewrite(hyper)
    3**(1/6)*z*hyper((), (4/3,), z**3/9)/gamma(1/3) + 3**(5/6)*hyper((), (2/3,), z**3/9)/(3*gamma(2/3))

    See Also
    ========

    airyai: Airy function of the first kind.
    airyaiprime: Derivative of the Airy function of the first kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return S.One / (3**Rational(1, 6) * gamma(Rational(2, 3)))

        if arg.is_zero:
            return S.One / (3**Rational(1, 6) * gamma(Rational(2, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return airybiprime(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return (cbrt(3)*x * Abs(sin(Rational(2, 3)*pi*(n + S.One))) * factorial((n - S.One)/S(3)) /
                        ((n + S.One) * Abs(cos(Rational(2, 3)*pi*(n + S.Half))) * factorial((n - 2)/S(3))) * p)
            else:
                return (S.One/(root(3, 6)*pi) * gamma((n + S.One)/S(3)) * Abs(sin(Rational(2, 3)*pi*(n + S.One))) /
                        factorial(n) * (cbrt(3)*x)**n)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return sqrt(-z/3) * (besselj(-ot, tt*a) - besselj(ot, tt*a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return sqrt(z)/sqrt(3) * (besseli(-ot, tt*a) + besseli(ot, tt*a))
        else:
            b = Pow(a, ot)
            c = Pow(a, -ot)
            return sqrt(ot)*(b*besseli(-ot, tt*a) + z*c*besseli(ot, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = S.One / (root(3, 6)*gamma(Rational(2, 3)))
        pf2 = z*root(3, 6) / gamma(Rational(1, 3))
        return pf1 * hyper([], [Rational(2, 3)], z**3/9) + pf2 * hyper([], [Rational(4, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is given by 03.06.16.0001.01
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryBi/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d * z**n)**m / (d**m * z**(m*n))
                    newarg = c * d**m * z**(m*n)
                    return S.Half * (sqrt(3)*(S.One - pf)*airyai(newarg) + (S.One + pf)*airybi(newarg))


class airyaiprime(AiryBase):
    r"""
    The derivative $\operatorname{Ai}^\prime$ of the Airy function of the first
    kind.

    Explanation
    ===========

    The Airy function $\operatorname{Ai}^\prime(z)$ is defined to be the
    function

    .. math::
        \operatorname{Ai}^\prime(z) := \frac{\mathrm{d} \operatorname{Ai}(z)}{\mathrm{d} z}.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airyaiprime
    >>> from sympy.abc import z

    >>> airyaiprime(z)
    airyaiprime(z)

    Several special values are known:

    >>> airyaiprime(0)
    -3**(2/3)/(3*gamma(1/3))
    >>> from sympy import oo
    >>> airyaiprime(oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airyaiprime(z))
    airyaiprime(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airyaiprime(z), z)
    z*airyai(z)
    >>> diff(airyaiprime(z), z, 2)
    z*airyaiprime(z) + airyai(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airyaiprime(z), z, 0, 3)
    -3**(2/3)/(3*gamma(1/3)) + 3**(1/3)*z**2/(6*gamma(2/3)) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airyaiprime(-2).evalf(50)
    0.61825902074169104140626429133247528291577794512415

    Rewrite $\operatorname{Ai}^\prime(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airyaiprime(z).rewrite(hyper)
    3**(1/3)*z**2*hyper((), (5/3,), z**3/9)/(6*gamma(2/3)) - 3**(2/3)*hyper((), (1/3,), z**3/9)/(3*gamma(1/3))

    See Also
    ========

    airyai: Airy function of the first kind.
    airybi: Airy function of the second kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero

        if arg.is_zero:
            return S.NegativeOne / (3**Rational(1, 3) * gamma(Rational(1, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.args[0]*airyai(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        z = self.args[0]._to_mpmath(prec)
        with workprec(prec):
            res = mp.airyai(z, derivative=1)
        return Expr._from_mpmath(res, prec)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return z/3 * (besselj(-tt, tt*a) - besselj(tt, tt*a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = tt * Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return z/3 * (besseli(tt, a) - besseli(-tt, a))
        else:
            a = Pow(z, Rational(3, 2))
            b = Pow(a, tt)
            c = Pow(a, -tt)
            return ot * (z**2*c*besseli(tt, tt*a) - b*besseli(-ot, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = z**2 / (2*3**Rational(2, 3)*gamma(Rational(2, 3)))
        pf2 = 1 / (root(3, 3)*gamma(Rational(1, 3)))
        return pf1 * hyper([], [Rational(5, 3)], z**3/9) - pf2 * hyper([], [Rational(1, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is in principle
                # given by 03.07.16.0001.01 but note
                # that there is an error in this formula.
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryAiPrime/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d**m * z**(n*m)) / (d * z**n)**m
                    newarg = c * d**m * z**(n*m)
                    return S.Half * ((pf + S.One)*airyaiprime(newarg) + (pf - S.One)/sqrt(3)*airybiprime(newarg))


class airybiprime(AiryBase):
    r"""
    The derivative $\operatorname{Bi}^\prime$ of the Airy function of the first
    kind.

    Explanation
    ===========

    The Airy function $\operatorname{Bi}^\prime(z)$ is defined to be the
    function

    .. math::
        \operatorname{Bi}^\prime(z) := \frac{\mathrm{d} \operatorname{Bi}(z)}{\mathrm{d} z}.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airybiprime
    >>> from sympy.abc import z

    >>> airybiprime(z)
    airybiprime(z)

    Several special values are known:

    >>> airybiprime(0)
    3**(1/6)/gamma(1/3)
    >>> from sympy import oo
    >>> airybiprime(oo)
    oo
    >>> airybiprime(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airybiprime(z))
    airybiprime(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airybiprime(z), z)
    z*airybi(z)
    >>> diff(airybiprime(z), z, 2)
    z*airybiprime(z) + airybi(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airybiprime(z), z, 0, 3)
    3**(1/6)/gamma(1/3) + 3**(5/6)*z**2/(6*gamma(2/3)) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airybiprime(-2).evalf(50)
    0.27879516692116952268509756941098324140300059345163

    Rewrite $\operatorname{Bi}^\prime(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airybiprime(z).rewrite(hyper)
    3**(5/6)*z**2*hyper((), (5/3,), z**3/9)/(6*gamma(2/3)) + 3**(1/6)*hyper((), (1/3,), z**3/9)/gamma(1/3)

    See Also
    ========

    airyai: Airy function of the first kind.
    airybi: Airy function of the second kind.
    airyaiprime: Derivative of the Airy function of the first kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return 3**Rational(1, 6) / gamma(Rational(1, 3))

        if arg.is_zero:
            return 3**Rational(1, 6) / gamma(Rational(1, 3))


    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.args[0]*airybi(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        z = self.args[0]._to_mpmath(prec)
        with workprec(prec):
            res = mp.airybi(z, derivative=1)
        return Expr._from_mpmath(res, prec)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        tt = Rational(2, 3)
        a = tt * Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return -z/sqrt(3) * (besselj(-tt, a) + besselj(tt, a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = tt * Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return z/sqrt(3) * (besseli(-tt, a) + besseli(tt, a))
        else:
            a = Pow(z, Rational(3, 2))
            b = Pow(a, tt)
            c = Pow(a, -tt)
            return sqrt(ot) * (b*besseli(-tt, tt*a) + z**2*c*besseli(tt, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = z**2 / (2*root(3, 6)*gamma(Rational(2, 3)))
        pf2 = root(3, 6) / gamma(Rational(1, 3))
        return pf1 * hyper([], [Rational(5, 3)], z**3/9) + pf2 * hyper([], [Rational(1, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is in principle
                # given by 03.08.16.0001.01 but note
                # that there is an error in this formula.
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryBiPrime/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d**m * z**(n*m)) / (d * z**n)**m
                    newarg = c * d**m * z**(n*m)
                    return S.Half * (sqrt(3)*(pf - S.One)*airyaiprime(newarg) + (pf + S.One)*airybiprime(newarg))


class marcumq(DefinedFunction):
    r"""
    The Marcum Q-function.

    Explanation
    ===========

    The Marcum Q-function is defined by the meromorphic continuation of

    .. math::
        Q_m(a, b) = a^{- m + 1} \int_{b}^{\infty} x^{m} e^{- \frac{a^{2}}{2} - \frac{x^{2}}{2}} I_{m - 1}\left(a x\right)\, dx

    Examples
    ========

    >>> from sympy import marcumq
    >>> from sympy.abc import m, a, b
    >>> marcumq(m, a, b)
    marcumq(m, a, b)

    Special values:

    >>> marcumq(m, 0, b)
    uppergamma(m, b**2/2)/gamma(m)
    >>> marcumq(0, 0, 0)
    0
    >>> marcumq(0, a, 0)
    1 - exp(-a**2/2)
    >>> marcumq(1, a, a)
    1/2 + exp(-a**2)*besseli(0, a**2)/2
    >>> marcumq(2, a, a)
    1/2 + exp(-a**2)*besseli(0, a**2)/2 + exp(-a**2)*besseli(1, a**2)

    Differentiation with respect to $a$ and $b$ is supported:

    >>> from sympy import diff
    >>> diff(marcumq(m, a, b), a)
    a*(-marcumq(m, a, b) + marcumq(m + 1, a, b))
    >>> diff(marcumq(m, a, b), b)
    -a**(1 - m)*b**m*exp(-a**2/2 - b**2/2)*besseli(m - 1, a*b)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Marcum_Q-function
    .. [2] https://mathworld.wolfram.com/MarcumQ-Function.html

    """

    @classmethod
    def eval(cls, m, a, b):
        if a is S.Zero:
            if m is S.Zero and b is S.Zero:
                return S.Zero
            return uppergamma(m, b**2 * S.Half) / gamma(m)

        if m is S.Zero and b is S.Zero:
            return 1 - 1 / exp(a**2 * S.Half)

        if a == b:
            if m is S.One:
                return (1 + exp(-a**2) * besseli(0, a**2))*S.Half
            if m == 2:
                return S.Half + S.Half * exp(-a**2) * besseli(0, a**2) + exp(-a**2) * besseli(1, a**2)

        if a.is_zero:
            if m.is_zero and b.is_zero:
                return S.Zero
            return uppergamma(m, b**2*S.Half) / gamma(m)

        if m.is_zero and b.is_zero:
            return 1 - 1 / exp(a**2*S.Half)

    def fdiff(self, argindex=2):
        m, a, b = self.args
        if argindex == 2:
            return a * (-marcumq(m, a, b) + marcumq(1+m, a, b))
        elif argindex == 3:
            return (-b**m / a**(m-1)) * exp(-(a**2 + b**2)/2) * besseli(m-1, a*b)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Integral(self, m, a, b, **kwargs):
        from sympy.integrals.integrals import Integral
        x = kwargs.get('x', Dummy(uniquely_named_symbol('x').name))
        return a ** (1 - m) * \
               Integral(x**m * exp(-(x**2 + a**2)/2) * besseli(m-1, a*x), [x, b, S.Infinity])

    def _eval_rewrite_as_Sum(self, m, a, b, **kwargs):
        from sympy.concrete.summations import Sum
        k = kwargs.get('k', Dummy('k'))
        return exp(-(a**2 + b**2) / 2) * Sum((a/b)**k * besseli(k, a*b), [k, 1-m, S.Infinity])

    def _eval_rewrite_as_besseli(self, m, a, b, **kwargs):
        if a == b:
            if m == 1:
                return (1 + exp(-a**2) * besseli(0, a**2)) / 2
            if m.is_Integer and m >= 2:
                s = sum(besseli(i, a**2) for i in range(1, m))
                return S.Half + exp(-a**2) * besseli(0, a**2) / 2 + exp(-a**2) * s

    def _eval_is_zero(self):
        if all(arg.is_zero for arg in self.args):
            return True

class _besseli(DefinedFunction):
    """
    Helper function to make the $\\mathrm{besseli}(nu, z)$
    function tractable for the Gruntz algorithm.

    """

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.functions.combinatorial.factorials import RisingFactorial
        from sympy.series.order import Order
        point = args0[1]

        if point in [S.Infinity, S.NegativeInfinity]:
            nu, z = self.args
            l = [((RisingFactorial(Rational(2*nu - 1, 2), k)*RisingFactorial(
                    Rational(2*nu + 1, 2), k))/((2)**(k)*z**(Rational(2*k + 1, 2))*factorial(k))) for k in range(n)]
            return sqrt(pi/(2))*(Add(*l)) + Order(1/z**(Rational(2*n + 1, 2)), x)

        return super()._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_intractable(self, nu, z, **kwargs):
        return exp(-z)*besseli(nu, z)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)


class _besselk(DefinedFunction):
    """
    Helper function to make the $\\mathrm{besselk}(nu, z)$
    function tractable for the Gruntz algorithm.

    """

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.functions.combinatorial.factorials import RisingFactorial
        from sympy.series.order import Order
        point = args0[1]

        if point in [S.Infinity, S.NegativeInfinity]:
            nu, z = self.args
            l = [((RisingFactorial(Rational(2*nu - 1, 2), k)*RisingFactorial(
                    Rational(2*nu + 1, 2), k))/((-2)**(k)*z**(Rational(2*k + 1, 2))*factorial(k))) for k in range(n)]
            return sqrt(pi/(2))*(Add(*l)) + Order(1/z**(Rational(2*n + 1, 2)), x)

        return super()._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_intractable(self,nu, z, **kwargs):
        return exp(z)*besselk(nu, z)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\base.py ===
# sql/base.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Foundational utilities common to many sql modules.

"""


from __future__ import annotations

import collections
from enum import Enum
import itertools
from itertools import zip_longest
import operator
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import roles
from . import visitors
from .cache_key import HasCacheKey  # noqa
from .cache_key import MemoizedHasCacheKey  # noqa
from .traversals import HasCopyInternals  # noqa
from .visitors import ClauseVisitor
from .visitors import ExtendedInternalTraversal
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import util
from ..util import HasMemoized as HasMemoized
from ..util import hybridmethod
from ..util import typing as compat_typing
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeGuard

if TYPE_CHECKING:
    from . import coercions
    from . import elements
    from . import type_api
    from ._orm_types import DMLStrategyArgument
    from ._orm_types import SynchronizeSessionArgument
    from ._typing import _CLE
    from .compiler import SQLCompiler
    from .elements import BindParameter
    from .elements import ClauseList
    from .elements import ColumnClause  # noqa
    from .elements import ColumnElement
    from .elements import NamedColumn
    from .elements import SQLCoreOperations
    from .elements import TextClause
    from .schema import Column
    from .schema import DefaultGenerator
    from .selectable import _JoinTargetElement
    from .selectable import _SelectIterable
    from .selectable import FromClause
    from ..engine import Connection
    from ..engine import CursorResult
    from ..engine.interfaces import _CoreMultiExecuteParams
    from ..engine.interfaces import _ExecuteOptions
    from ..engine.interfaces import _ImmutableExecuteOptions
    from ..engine.interfaces import CacheStats
    from ..engine.interfaces import Compiled
    from ..engine.interfaces import CompiledCacheType
    from ..engine.interfaces import CoreExecuteOptionsParameter
    from ..engine.interfaces import Dialect
    from ..engine.interfaces import IsolationLevel
    from ..engine.interfaces import SchemaTranslateMapType
    from ..event import dispatcher

if not TYPE_CHECKING:
    coercions = None  # noqa
    elements = None  # noqa
    type_api = None  # noqa


class _NoArg(Enum):
    NO_ARG = 0

    def __repr__(self):
        return f"_NoArg.{self.name}"


NO_ARG = _NoArg.NO_ARG


class _NoneName(Enum):
    NONE_NAME = 0
    """indicate a 'deferred' name that was ultimately the value None."""


_NONE_NAME = _NoneName.NONE_NAME

_T = TypeVar("_T", bound=Any)

_Fn = TypeVar("_Fn", bound=Callable[..., Any])

_AmbiguousTableNameMap = MutableMapping[str, str]


class _DefaultDescriptionTuple(NamedTuple):
    arg: Any
    is_scalar: Optional[bool]
    is_callable: Optional[bool]
    is_sentinel: Optional[bool]

    @classmethod
    def _from_column_default(
        cls, default: Optional[DefaultGenerator]
    ) -> _DefaultDescriptionTuple:
        return (
            _DefaultDescriptionTuple(
                default.arg,  # type: ignore
                default.is_scalar,
                default.is_callable,
                default.is_sentinel,
            )
            if default
            and (
                default.has_arg
                or (not default.for_update and default.is_sentinel)
            )
            else _DefaultDescriptionTuple(None, None, None, None)
        )


_never_select_column = operator.attrgetter("_omit_from_statements")


class _EntityNamespace(Protocol):
    def __getattr__(self, key: str) -> SQLCoreOperations[Any]: ...


class _HasEntityNamespace(Protocol):
    @util.ro_non_memoized_property
    def entity_namespace(self) -> _EntityNamespace: ...


def _is_has_entity_namespace(element: Any) -> TypeGuard[_HasEntityNamespace]:
    return hasattr(element, "entity_namespace")


# Remove when https://github.com/python/mypy/issues/14640 will be fixed
_Self = TypeVar("_Self", bound=Any)


class Immutable:
    """mark a ClauseElement as 'immutable' when expressions are cloned.

    "immutable" objects refers to the "mutability" of an object in the
    context of SQL DQL and DML generation.   Such as, in DQL, one can
    compose a SELECT or subquery of varied forms, but one cannot modify
    the structure of a specific table or column within DQL.
    :class:`.Immutable` is mostly intended to follow this concept, and as
    such the primary "immutable" objects are :class:`.ColumnClause`,
    :class:`.Column`, :class:`.TableClause`, :class:`.Table`.

    """

    __slots__ = ()

    _is_immutable = True

    def unique_params(self, *optionaldict, **kwargs):
        raise NotImplementedError("Immutable objects do not support copying")

    def params(self, *optionaldict, **kwargs):
        raise NotImplementedError("Immutable objects do not support copying")

    def _clone(self: _Self, **kw: Any) -> _Self:
        return self

    def _copy_internals(
        self, *, omit_attrs: Iterable[str] = (), **kw: Any
    ) -> None:
        pass


class SingletonConstant(Immutable):
    """Represent SQL constants like NULL, TRUE, FALSE"""

    _is_singleton_constant = True

    _singleton: SingletonConstant

    def __new__(cls: _T, *arg: Any, **kw: Any) -> _T:
        return cast(_T, cls._singleton)

    @util.non_memoized_property
    def proxy_set(self) -> FrozenSet[ColumnElement[Any]]:
        raise NotImplementedError()

    @classmethod
    def _create_singleton(cls):
        obj = object.__new__(cls)
        obj.__init__()  # type: ignore

        # for a long time this was an empty frozenset, meaning
        # a SingletonConstant would never be a "corresponding column" in
        # a statement.  This referred to #6259.  However, in #7154 we see
        # that we do in fact need "correspondence" to work when matching cols
        # in result sets, so the non-correspondence was moved to a more
        # specific level when we are actually adapting expressions for SQL
        # render only.
        obj.proxy_set = frozenset([obj])
        cls._singleton = obj


def _from_objects(
    *elements: Union[
        ColumnElement[Any], FromClause, TextClause, _JoinTargetElement
    ]
) -> Iterator[FromClause]:
    return itertools.chain.from_iterable(
        [element._from_objects for element in elements]
    )


def _select_iterables(
    elements: Iterable[roles.ColumnsClauseRole],
) -> _SelectIterable:
    """expand tables into individual columns in the
    given list of column expressions.

    """
    return itertools.chain.from_iterable(
        [c._select_iterable for c in elements]
    )


_SelfGenerativeType = TypeVar("_SelfGenerativeType", bound="_GenerativeType")


class _GenerativeType(compat_typing.Protocol):
    def _generate(self) -> Self: ...


def _generative(fn: _Fn) -> _Fn:
    """non-caching _generative() decorator.

    This is basically the legacy decorator that copies the object and
    runs a method on the new copy.

    """

    @util.decorator
    def _generative(
        fn: _Fn, self: _SelfGenerativeType, *args: Any, **kw: Any
    ) -> _SelfGenerativeType:
        """Mark a method as generative."""

        self = self._generate()
        x = fn(self, *args, **kw)
        assert x is self, "generative methods must return self"
        return self

    decorated = _generative(fn)
    decorated.non_generative = fn  # type: ignore
    return decorated


def _exclusive_against(*names: str, **kw: Any) -> Callable[[_Fn], _Fn]:
    msgs = kw.pop("msgs", {})

    defaults = kw.pop("defaults", {})

    getters = [
        (name, operator.attrgetter(name), defaults.get(name, None))
        for name in names
    ]

    @util.decorator
    def check(fn, *args, **kw):
        # make pylance happy by not including "self" in the argument
        # list
        self = args[0]
        args = args[1:]
        for name, getter, default_ in getters:
            if getter(self) is not default_:
                msg = msgs.get(
                    name,
                    "Method %s() has already been invoked on this %s construct"
                    % (fn.__name__, self.__class__),
                )
                raise exc.InvalidRequestError(msg)
        return fn(self, *args, **kw)

    return check


def _clone(element, **kw):
    return element._clone(**kw)


def _expand_cloned(
    elements: Iterable[_CLE],
) -> Iterable[_CLE]:
    """expand the given set of ClauseElements to be the set of all 'cloned'
    predecessors.

    """
    # TODO: cython candidate
    return itertools.chain(*[x._cloned_set for x in elements])


def _de_clone(
    elements: Iterable[_CLE],
) -> Iterable[_CLE]:
    for x in elements:
        while x._is_clone_of is not None:
            x = x._is_clone_of
        yield x


def _cloned_intersection(a: Iterable[_CLE], b: Iterable[_CLE]) -> Set[_CLE]:
    """return the intersection of sets a and b, counting
    any overlap between 'cloned' predecessors.

    The returned set is in terms of the entities present within 'a'.

    """
    all_overlap = set(_expand_cloned(a)).intersection(_expand_cloned(b))
    return {elem for elem in a if all_overlap.intersection(elem._cloned_set)}


def _cloned_difference(a: Iterable[_CLE], b: Iterable[_CLE]) -> Set[_CLE]:
    all_overlap = set(_expand_cloned(a)).intersection(_expand_cloned(b))
    return {
        elem for elem in a if not all_overlap.intersection(elem._cloned_set)
    }


class _DialectArgView(MutableMapping[str, Any]):
    """A dictionary view of dialect-level arguments in the form
    <dialectname>_<argument_name>.

    """

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def _key(self, key):
        try:
            dialect, value_key = key.split("_", 1)
        except ValueError as err:
            raise KeyError(key) from err
        else:
            return dialect, value_key

    def __getitem__(self, key):
        dialect, value_key = self._key(key)

        try:
            opt = self.obj.dialect_options[dialect]
        except exc.NoSuchModuleError as err:
            raise KeyError(key) from err
        else:
            return opt[value_key]

    def __setitem__(self, key, value):
        try:
            dialect, value_key = self._key(key)
        except KeyError as err:
            raise exc.ArgumentError(
                "Keys must be of the form <dialectname>_<argname>"
            ) from err
        else:
            self.obj.dialect_options[dialect][value_key] = value

    def __delitem__(self, key):
        dialect, value_key = self._key(key)
        del self.obj.dialect_options[dialect][value_key]

    def __len__(self):
        return sum(
            len(args._non_defaults)
            for args in self.obj.dialect_options.values()
        )

    def __iter__(self):
        return (
            "%s_%s" % (dialect_name, value_name)
            for dialect_name in self.obj.dialect_options
            for value_name in self.obj.dialect_options[
                dialect_name
            ]._non_defaults
        )


class _DialectArgDict(MutableMapping[str, Any]):
    """A dictionary view of dialect-level arguments for a specific
    dialect.

    Maintains a separate collection of user-specified arguments
    and dialect-specified default arguments.

    """

    def __init__(self):
        self._non_defaults = {}
        self._defaults = {}

    def __len__(self):
        return len(set(self._non_defaults).union(self._defaults))

    def __iter__(self):
        return iter(set(self._non_defaults).union(self._defaults))

    def __getitem__(self, key):
        if key in self._non_defaults:
            return self._non_defaults[key]
        else:
            return self._defaults[key]

    def __setitem__(self, key, value):
        self._non_defaults[key] = value

    def __delitem__(self, key):
        del self._non_defaults[key]


@util.preload_module("sqlalchemy.dialects")
def _kw_reg_for_dialect(dialect_name):
    dialect_cls = util.preloaded.dialects.registry.load(dialect_name)
    if dialect_cls.construct_arguments is None:
        return None
    return dict(dialect_cls.construct_arguments)


class DialectKWArgs:
    """Establish the ability for a class to have dialect-specific arguments
    with defaults and constructor validation.

    The :class:`.DialectKWArgs` interacts with the
    :attr:`.DefaultDialect.construct_arguments` present on a dialect.

    .. seealso::

        :attr:`.DefaultDialect.construct_arguments`

    """

    __slots__ = ()

    _dialect_kwargs_traverse_internals = [
        ("dialect_options", InternalTraversal.dp_dialect_options)
    ]

    @classmethod
    def argument_for(cls, dialect_name, argument_name, default):
        """Add a new kind of dialect-specific keyword argument for this class.

        E.g.::

            Index.argument_for("mydialect", "length", None)

            some_index = Index("a", "b", mydialect_length=5)

        The :meth:`.DialectKWArgs.argument_for` method is a per-argument
        way adding extra arguments to the
        :attr:`.DefaultDialect.construct_arguments` dictionary. This
        dictionary provides a list of argument names accepted by various
        schema-level constructs on behalf of a dialect.

        New dialects should typically specify this dictionary all at once as a
        data member of the dialect class.  The use case for ad-hoc addition of
        argument names is typically for end-user code that is also using
        a custom compilation scheme which consumes the additional arguments.

        :param dialect_name: name of a dialect.  The dialect must be
         locatable, else a :class:`.NoSuchModuleError` is raised.   The
         dialect must also include an existing
         :attr:`.DefaultDialect.construct_arguments` collection, indicating
         that it participates in the keyword-argument validation and default
         system, else :class:`.ArgumentError` is raised.  If the dialect does
         not include this collection, then any keyword argument can be
         specified on behalf of this dialect already.  All dialects packaged
         within SQLAlchemy include this collection, however for third party
         dialects, support may vary.

        :param argument_name: name of the parameter.

        :param default: default value of the parameter.

        """

        construct_arg_dictionary = DialectKWArgs._kw_registry[dialect_name]
        if construct_arg_dictionary is None:
            raise exc.ArgumentError(
                "Dialect '%s' does have keyword-argument "
                "validation and defaults enabled configured" % dialect_name
            )
        if cls not in construct_arg_dictionary:
            construct_arg_dictionary[cls] = {}
        construct_arg_dictionary[cls][argument_name] = default

    @property
    def dialect_kwargs(self):
        """A collection of keyword arguments specified as dialect-specific
        options to this construct.

        The arguments are present here in their original ``<dialect>_<kwarg>``
        format.  Only arguments that were actually passed are included;
        unlike the :attr:`.DialectKWArgs.dialect_options` collection, which
        contains all options known by this dialect including defaults.

        The collection is also writable; keys are accepted of the
        form ``<dialect>_<kwarg>`` where the value will be assembled
        into the list of options.

        .. seealso::

            :attr:`.DialectKWArgs.dialect_options` - nested dictionary form

        """
        return _DialectArgView(self)

    @property
    def kwargs(self):
        """A synonym for :attr:`.DialectKWArgs.dialect_kwargs`."""
        return self.dialect_kwargs

    _kw_registry = util.PopulateDict(_kw_reg_for_dialect)

    @classmethod
    def _kw_reg_for_dialect_cls(cls, dialect_name):
        construct_arg_dictionary = DialectKWArgs._kw_registry[dialect_name]
        d = _DialectArgDict()

        if construct_arg_dictionary is None:
            d._defaults.update({"*": None})
        else:
            for cls in reversed(cls.__mro__):
                if cls in construct_arg_dictionary:
                    d._defaults.update(construct_arg_dictionary[cls])
        return d

    @util.memoized_property
    def dialect_options(self):
        """A collection of keyword arguments specified as dialect-specific
        options to this construct.

        This is a two-level nested registry, keyed to ``<dialect_name>``
        and ``<argument_name>``.  For example, the ``postgresql_where``
        argument would be locatable as::

            arg = my_object.dialect_options["postgresql"]["where"]

        .. versionadded:: 0.9.2

        .. seealso::

            :attr:`.DialectKWArgs.dialect_kwargs` - flat dictionary form

        """

        return util.PopulateDict(self._kw_reg_for_dialect_cls)

    def _validate_dialect_kwargs(self, kwargs: Dict[str, Any]) -> None:
        # validate remaining kwargs that they all specify DB prefixes

        if not kwargs:
            return

        for k in kwargs:
            m = re.match("^(.+?)_(.+)$", k)
            if not m:
                raise TypeError(
                    "Additional arguments should be "
                    "named <dialectname>_<argument>, got '%s'" % k
                )
            dialect_name, arg_name = m.group(1, 2)

            try:
                construct_arg_dictionary = self.dialect_options[dialect_name]
            except exc.NoSuchModuleError:
                util.warn(
                    "Can't validate argument %r; can't "
                    "locate any SQLAlchemy dialect named %r"
                    % (k, dialect_name)
                )
                self.dialect_options[dialect_name] = d = _DialectArgDict()
                d._defaults.update({"*": None})
                d._non_defaults[arg_name] = kwargs[k]
            else:
                if (
                    "*" not in construct_arg_dictionary
                    and arg_name not in construct_arg_dictionary
                ):
                    raise exc.ArgumentError(
                        "Argument %r is not accepted by "
                        "dialect %r on behalf of %r"
                        % (k, dialect_name, self.__class__)
                    )
                else:
                    construct_arg_dictionary[arg_name] = kwargs[k]


class CompileState:
    """Produces additional object state necessary for a statement to be
    compiled.

    the :class:`.CompileState` class is at the base of classes that assemble
    state for a particular statement object that is then used by the
    compiler.   This process is essentially an extension of the process that
    the SQLCompiler.visit_XYZ() method takes, however there is an emphasis
    on converting raw user intent into more organized structures rather than
    producing string output.   The top-level :class:`.CompileState` for the
    statement being executed is also accessible when the execution context
    works with invoking the statement and collecting results.

    The production of :class:`.CompileState` is specific to the compiler,  such
    as within the :meth:`.SQLCompiler.visit_insert`,
    :meth:`.SQLCompiler.visit_select` etc. methods.  These methods are also
    responsible for associating the :class:`.CompileState` with the
    :class:`.SQLCompiler` itself, if the statement is the "toplevel" statement,
    i.e. the outermost SQL statement that's actually being executed.
    There can be other :class:`.CompileState` objects that are not the
    toplevel, such as when a SELECT subquery or CTE-nested
    INSERT/UPDATE/DELETE is generated.

    .. versionadded:: 1.4

    """

    __slots__ = ("statement", "_ambiguous_table_name_map")

    plugins: Dict[Tuple[str, str], Type[CompileState]] = {}

    _ambiguous_table_name_map: Optional[_AmbiguousTableNameMap]

    @classmethod
    def create_for_statement(
        cls, statement: Executable, compiler: SQLCompiler, **kw: Any
    ) -> CompileState:
        # factory construction.

        if statement._propagate_attrs:
            plugin_name = statement._propagate_attrs.get(
                "compile_state_plugin", "default"
            )
            klass = cls.plugins.get(
                (plugin_name, statement._effective_plugin_target), None
            )
            if klass is None:
                klass = cls.plugins[
                    ("default", statement._effective_plugin_target)
                ]

        else:
            klass = cls.plugins[
                ("default", statement._effective_plugin_target)
            ]

        if klass is cls:
            return cls(statement, compiler, **kw)
        else:
            return klass.create_for_statement(statement, compiler, **kw)

    def __init__(self, statement, compiler, **kw):
        self.statement = statement

    @classmethod
    def get_plugin_class(
        cls, statement: Executable
    ) -> Optional[Type[CompileState]]:
        plugin_name = statement._propagate_attrs.get(
            "compile_state_plugin", None
        )

        if plugin_name:
            key = (plugin_name, statement._effective_plugin_target)
            if key in cls.plugins:
                return cls.plugins[key]

        # there's no case where we call upon get_plugin_class() and want
        # to get None back, there should always be a default.  return that
        # if there was no plugin-specific class  (e.g. Insert with "orm"
        # plugin)
        try:
            return cls.plugins[("default", statement._effective_plugin_target)]
        except KeyError:
            return None

    @classmethod
    def _get_plugin_class_for_plugin(
        cls, statement: Executable, plugin_name: str
    ) -> Optional[Type[CompileState]]:
        try:
            return cls.plugins[
                (plugin_name, statement._effective_plugin_target)
            ]
        except KeyError:
            return None

    @classmethod
    def plugin_for(
        cls, plugin_name: str, visit_name: str
    ) -> Callable[[_Fn], _Fn]:
        def decorate(cls_to_decorate):
            cls.plugins[(plugin_name, visit_name)] = cls_to_decorate
            return cls_to_decorate

        return decorate


class Generative(HasMemoized):
    """Provide a method-chaining pattern in conjunction with the
    @_generative decorator."""

    def _generate(self) -> Self:
        skip = self._memoized_keys
        cls = self.__class__
        s = cls.__new__(cls)
        if skip:
            # ensure this iteration remains atomic
            s.__dict__ = {
                k: v for k, v in self.__dict__.copy().items() if k not in skip
            }
        else:
            s.__dict__ = self.__dict__.copy()
        return s


class InPlaceGenerative(HasMemoized):
    """Provide a method-chaining pattern in conjunction with the
    @_generative decorator that mutates in place."""

    __slots__ = ()

    def _generate(self):
        skip = self._memoized_keys
        # note __dict__ needs to be in __slots__ if this is used
        for k in skip:
            self.__dict__.pop(k, None)
        return self


class HasCompileState(Generative):
    """A class that has a :class:`.CompileState` associated with it."""

    _compile_state_plugin: Optional[Type[CompileState]] = None

    _attributes: util.immutabledict[str, Any] = util.EMPTY_DICT

    _compile_state_factory = CompileState.create_for_statement


class _MetaOptions(type):
    """metaclass for the Options class.

    This metaclass is actually necessary despite the availability of the
    ``__init_subclass__()`` hook as this type also provides custom class-level
    behavior for the ``__add__()`` method.

    """

    _cache_attrs: Tuple[str, ...]

    def __add__(self, other):
        o1 = self()

        if set(other).difference(self._cache_attrs):
            raise TypeError(
                "dictionary contains attributes not covered by "
                "Options class %s: %r"
                % (self, set(other).difference(self._cache_attrs))
            )

        o1.__dict__.update(other)
        return o1

    if TYPE_CHECKING:

        def __getattr__(self, key: str) -> Any: ...

        def __setattr__(self, key: str, value: Any) -> None: ...

        def __delattr__(self, key: str) -> None: ...


class Options(metaclass=_MetaOptions):
    """A cacheable option dictionary with defaults."""

    __slots__ = ()

    _cache_attrs: Tuple[str, ...]

    def __init_subclass__(cls) -> None:
        dict_ = cls.__dict__
        cls._cache_attrs = tuple(
            sorted(
                d
                for d in dict_
                if not d.startswith("__")
                and d not in ("_cache_key_traversal",)
            )
        )
        super().__init_subclass__()

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __add__(self, other):
        o1 = self.__class__.__new__(self.__class__)
        o1.__dict__.update(self.__dict__)

        if set(other).difference(self._cache_attrs):
            raise TypeError(
                "dictionary contains attributes not covered by "
                "Options class %s: %r"
                % (self, set(other).difference(self._cache_attrs))
            )

        o1.__dict__.update(other)
        return o1

    def __eq__(self, other):
        # TODO: very inefficient.  This is used only in test suites
        # right now.
        for a, b in zip_longest(self._cache_attrs, other._cache_attrs):
            if getattr(self, a) != getattr(other, b):
                return False
        return True

    def __repr__(self):
        # TODO: fairly inefficient, used only in debugging right now.

        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(
                "%s=%r" % (k, self.__dict__[k])
                for k in self._cache_attrs
                if k in self.__dict__
            ),
        )

    @classmethod
    def isinstance(cls, klass: Type[Any]) -> bool:
        return issubclass(cls, klass)

    @hybridmethod
    def add_to_element(self, name, value):
        return self + {name: getattr(self, name) + value}

    @hybridmethod
    def _state_dict_inst(self) -> Mapping[str, Any]:
        return self.__dict__

    _state_dict_const: util.immutabledict[str, Any] = util.EMPTY_DICT

    @_state_dict_inst.classlevel
    def _state_dict(cls) -> Mapping[str, Any]:
        return cls._state_dict_const

    @classmethod
    def safe_merge(cls, other):
        d = other._state_dict()

        # only support a merge with another object of our class
        # and which does not have attrs that we don't.   otherwise
        # we risk having state that might not be part of our cache
        # key strategy

        if (
            cls is not other.__class__
            and other._cache_attrs
            and set(other._cache_attrs).difference(cls._cache_attrs)
        ):
            raise TypeError(
                "other element %r is not empty, is not of type %s, "
                "and contains attributes not covered here %r"
                % (
                    other,
                    cls,
                    set(other._cache_attrs).difference(cls._cache_attrs),
                )
            )
        return cls + d

    @classmethod
    def from_execution_options(
        cls, key, attrs, exec_options, statement_exec_options
    ):
        """process Options argument in terms of execution options.


        e.g.::

            (
                load_options,
                execution_options,
            ) = QueryContext.default_load_options.from_execution_options(
                "_sa_orm_load_options",
                {"populate_existing", "autoflush", "yield_per"},
                execution_options,
                statement._execution_options,
            )

        get back the Options and refresh "_sa_orm_load_options" in the
        exec options dict w/ the Options as well

        """

        # common case is that no options we are looking for are
        # in either dictionary, so cancel for that first
        check_argnames = attrs.intersection(
            set(exec_options).union(statement_exec_options)
        )

        existing_options = exec_options.get(key, cls)

        if check_argnames:
            result = {}
            for argname in check_argnames:
                local = "_" + argname
                if argname in exec_options:
                    result[local] = exec_options[argname]
                elif argname in statement_exec_options:
                    result[local] = statement_exec_options[argname]

            new_options = existing_options + result
            exec_options = util.immutabledict().merge_with(
                exec_options, {key: new_options}
            )
            return new_options, exec_options

        else:
            return existing_options, exec_options

    if TYPE_CHECKING:

        def __getattr__(self, key: str) -> Any: ...

        def __setattr__(self, key: str, value: Any) -> None: ...

        def __delattr__(self, key: str) -> None: ...


class CacheableOptions(Options, HasCacheKey):
    __slots__ = ()

    @hybridmethod
    def _gen_cache_key_inst(self, anon_map, bindparams):
        return HasCacheKey._gen_cache_key(self, anon_map, bindparams)

    @_gen_cache_key_inst.classlevel
    def _gen_cache_key(cls, anon_map, bindparams):
        return (cls, ())

    @hybridmethod
    def _generate_cache_key(self):
        return HasCacheKey._generate_cache_key_for_object(self)


class ExecutableOption(HasCopyInternals):
    __slots__ = ()

    _annotations = util.EMPTY_DICT

    __visit_name__ = "executable_option"

    _is_has_cache_key = False

    _is_core = True

    def _clone(self, **kw):
        """Create a shallow copy of this ExecutableOption."""
        c = self.__class__.__new__(self.__class__)
        c.__dict__ = dict(self.__dict__)  # type: ignore
        return c


class Executable(roles.StatementRole):
    """Mark a :class:`_expression.ClauseElement` as supporting execution.

    :class:`.Executable` is a superclass for all "statement" types
    of objects, including :func:`select`, :func:`delete`, :func:`update`,
    :func:`insert`, :func:`text`.

    """

    supports_execution: bool = True
    _execution_options: _ImmutableExecuteOptions = util.EMPTY_DICT
    _is_default_generator = False
    _with_options: Tuple[ExecutableOption, ...] = ()
    _with_context_options: Tuple[
        Tuple[Callable[[CompileState], None], Any], ...
    ] = ()
    _compile_options: Optional[Union[Type[CacheableOptions], CacheableOptions]]

    _executable_traverse_internals = [
        ("_with_options", InternalTraversal.dp_executable_options),
        (
            "_with_context_options",
            ExtendedInternalTraversal.dp_with_context_options,
        ),
        ("_propagate_attrs", ExtendedInternalTraversal.dp_propagate_attrs),
    ]

    is_select = False
    is_from_statement = False
    is_update = False
    is_insert = False
    is_text = False
    is_delete = False
    is_dml = False

    if TYPE_CHECKING:
        __visit_name__: str

        def _compile_w_cache(
            self,
            dialect: Dialect,
            *,
            compiled_cache: Optional[CompiledCacheType],
            column_keys: List[str],
            for_executemany: bool = False,
            schema_translate_map: Optional[SchemaTranslateMapType] = None,
            **kw: Any,
        ) -> Tuple[
            Compiled, Optional[Sequence[BindParameter[Any]]], CacheStats
        ]: ...

        def _execute_on_connection(
            self,
            connection: Connection,
            distilled_params: _CoreMultiExecuteParams,
            execution_options: CoreExecuteOptionsParameter,
        ) -> CursorResult[Any]: ...

        def _execute_on_scalar(
            self,
            connection: Connection,
            distilled_params: _CoreMultiExecuteParams,
            execution_options: CoreExecuteOptionsParameter,
        ) -> Any: ...

    @util.ro_non_memoized_property
    def _all_selected_columns(self):
        raise NotImplementedError()

    @property
    def _effective_plugin_target(self) -> str:
        return self.__visit_name__

    @_generative
    def options(self, *options: ExecutableOption) -> Self:
        """Apply options to this statement.

        In the general sense, options are any kind of Python object
        that can be interpreted by the SQL compiler for the statement.
        These options can be consumed by specific dialects or specific kinds
        of compilers.

        The most commonly known kind of option are the ORM level options
        that apply "eager load" and other loading behaviors to an ORM
        query.   However, options can theoretically be used for many other
        purposes.

        For background on specific kinds of options for specific kinds of
        statements, refer to the documentation for those option objects.

        .. versionchanged:: 1.4 - added :meth:`.Executable.options` to
           Core statement objects towards the goal of allowing unified
           Core / ORM querying capabilities.

        .. seealso::

            :ref:`loading_columns` - refers to options specific to the usage
            of ORM queries

            :ref:`relationship_loader_options` - refers to options specific
            to the usage of ORM queries

        """
        self._with_options += tuple(
            coercions.expect(roles.ExecutableOptionRole, opt)
            for opt in options
        )
        return self

    @_generative
    def _set_compile_options(self, compile_options: CacheableOptions) -> Self:
        """Assign the compile options to a new value.

        :param compile_options: appropriate CacheableOptions structure

        """

        self._compile_options = compile_options
        return self

    @_generative
    def _update_compile_options(self, options: CacheableOptions) -> Self:
        """update the _compile_options with new keys."""

        assert self._compile_options is not None
        self._compile_options += options
        return self

    @_generative
    def _add_context_option(
        self,
        callable_: Callable[[CompileState], None],
        cache_args: Any,
    ) -> Self:
        """Add a context option to this statement.

        These are callable functions that will
        be given the CompileState object upon compilation.

        A second argument cache_args is required, which will be combined with
        the ``__code__`` identity of the function itself in order to produce a
        cache key.

        """
        self._with_context_options += ((callable_, cache_args),)
        return self

    @overload
    def execution_options(
        self,
        *,
        compiled_cache: Optional[CompiledCacheType] = ...,
        logging_token: str = ...,
        isolation_level: IsolationLevel = ...,
        no_parameters: bool = False,
        stream_results: bool = False,
        max_row_buffer: int = ...,
        yield_per: int = ...,
        insertmanyvalues_page_size: int = ...,
        schema_translate_map: Optional[SchemaTranslateMapType] = ...,
        populate_existing: bool = False,
        autoflush: bool = False,
        synchronize_session: SynchronizeSessionArgument = ...,
        dml_strategy: DMLStrategyArgument = ...,
        render_nulls: bool = ...,
        is_delete_using: bool = ...,
        is_update_from: bool = ...,
        preserve_rowcount: bool = False,
        **opt: Any,
    ) -> Self: ...

    @overload
    def execution_options(self, **opt: Any) -> Self: ...

    @_generative
    def execution_options(self, **kw: Any) -> Self:
        """Set non-SQL options for the statement which take effect during
        execution.

        Execution options can be set at many scopes, including per-statement,
        per-connection, or per execution, using methods such as
        :meth:`_engine.Connection.execution_options` and parameters which
        accept a dictionary of options such as
        :paramref:`_engine.Connection.execute.execution_options` and
        :paramref:`_orm.Session.execute.execution_options`.

        The primary characteristic of an execution option, as opposed to
        other kinds of options such as ORM loader options, is that
        **execution options never affect the compiled SQL of a query, only
        things that affect how the SQL statement itself is invoked or how
        results are fetched**.  That is, execution options are not part of
        what's accommodated by SQL compilation nor are they considered part of
        the cached state of a statement.

        The :meth:`_sql.Executable.execution_options` method is
        :term:`generative`, as
        is the case for the method as applied to the :class:`_engine.Engine`
        and :class:`_orm.Query` objects, which means when the method is called,
        a copy of the object is returned, which applies the given parameters to
        that new copy, but leaves the original unchanged::

            statement = select(table.c.x, table.c.y)
            new_statement = statement.execution_options(my_option=True)

        An exception to this behavior is the :class:`_engine.Connection`
        object, where the :meth:`_engine.Connection.execution_options` method
        is explicitly **not** generative.

        The kinds of options that may be passed to
        :meth:`_sql.Executable.execution_options` and other related methods and
        parameter dictionaries include parameters that are explicitly consumed
        by SQLAlchemy Core or ORM, as well as arbitrary keyword arguments not
        defined by SQLAlchemy, which means the methods and/or parameter
        dictionaries may be used for user-defined parameters that interact with
        custom code, which may access the parameters using methods such as
        :meth:`_sql.Executable.get_execution_options` and
        :meth:`_engine.Connection.get_execution_options`, or within selected
        event hooks using a dedicated ``execution_options`` event parameter
        such as
        :paramref:`_events.ConnectionEvents.before_execute.execution_options`
        or :attr:`_orm.ORMExecuteState.execution_options`, e.g.::

             from sqlalchemy import event


             @event.listens_for(some_engine, "before_execute")
             def _process_opt(conn, statement, multiparams, params, execution_options):
                 "run a SQL function before invoking a statement"

                 if execution_options.get("do_special_thing", False):
                     conn.exec_driver_sql("run_special_function()")

        Within the scope of options that are explicitly recognized by
        SQLAlchemy, most apply to specific classes of objects and not others.
        The most common execution options include:

        * :paramref:`_engine.Connection.execution_options.isolation_level` -
          sets the isolation level for a connection or a class of connections
          via an :class:`_engine.Engine`.  This option is accepted only
          by :class:`_engine.Connection` or :class:`_engine.Engine`.

        * :paramref:`_engine.Connection.execution_options.stream_results` -
          indicates results should be fetched using a server side cursor;
          this option is accepted by :class:`_engine.Connection`, by the
          :paramref:`_engine.Connection.execute.execution_options` parameter
          on :meth:`_engine.Connection.execute`, and additionally by
          :meth:`_sql.Executable.execution_options` on a SQL statement object,
          as well as by ORM constructs like :meth:`_orm.Session.execute`.

        * :paramref:`_engine.Connection.execution_options.compiled_cache` -
          indicates a dictionary that will serve as the
          :ref:`SQL compilation cache <sql_caching>`
          for a :class:`_engine.Connection` or :class:`_engine.Engine`, as
          well as for ORM methods like :meth:`_orm.Session.execute`.
          Can be passed as ``None`` to disable caching for statements.
          This option is not accepted by
          :meth:`_sql.Executable.execution_options` as it is inadvisable to
          carry along a compilation cache within a statement object.

        * :paramref:`_engine.Connection.execution_options.schema_translate_map`
          - a mapping of schema names used by the
          :ref:`Schema Translate Map <schema_translating>` feature, accepted
          by :class:`_engine.Connection`, :class:`_engine.Engine`,
          :class:`_sql.Executable`, as well as by ORM constructs
          like :meth:`_orm.Session.execute`.

        .. seealso::

            :meth:`_engine.Connection.execution_options`

            :paramref:`_engine.Connection.execute.execution_options`

            :paramref:`_orm.Session.execute.execution_options`

            :ref:`orm_queryguide_execution_options` - documentation on all
            ORM-specific execution options

        """  # noqa: E501
        if "isolation_level" in kw:
            raise exc.ArgumentError(
                "'isolation_level' execution option may only be specified "
                "on Connection.execution_options(), or "
                "per-engine using the isolation_level "
                "argument to create_engine()."
            )
        if "compiled_cache" in kw:
            raise exc.ArgumentError(
                "'compiled_cache' execution option may only be specified "
                "on Connection.execution_options(), not per statement."
            )
        self._execution_options = self._execution_options.union(kw)
        return self

    def get_execution_options(self) -> _ExecuteOptions:
        """Get the non-SQL options which will take effect during execution.

        .. versionadded:: 1.3

        .. seealso::

            :meth:`.Executable.execution_options`
        """
        return self._execution_options


class SchemaEventTarget(event.EventTarget):
    """Base class for elements that are the targets of :class:`.DDLEvents`
    events.

    This includes :class:`.SchemaItem` as well as :class:`.SchemaType`.

    """

    dispatch: dispatcher[SchemaEventTarget]

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        """Associate with this SchemaEvent's parent object."""

    def _set_parent_with_dispatch(
        self, parent: SchemaEventTarget, **kw: Any
    ) -> None:
        self.dispatch.before_parent_attach(self, parent)
        self._set_parent(parent, **kw)
        self.dispatch.after_parent_attach(self, parent)


class SchemaVisitable(SchemaEventTarget, visitors.Visitable):
    """Base class for elements that are targets of a :class:`.SchemaVisitor`.

    .. versionadded:: 2.0.41

    """


class SchemaVisitor(ClauseVisitor):
    """Define the visiting for ``SchemaItem`` and more
    generally ``SchemaVisitable`` objects.

    """

    __traverse_options__ = {"schema_visitor": True}


class _SentinelDefaultCharacterization(Enum):
    NONE = "none"
    UNKNOWN = "unknown"
    CLIENTSIDE = "clientside"
    SENTINEL_DEFAULT = "sentinel_default"
    SERVERSIDE = "serverside"
    IDENTITY = "identity"
    SEQUENCE = "sequence"


class _SentinelColumnCharacterization(NamedTuple):
    columns: Optional[Sequence[Column[Any]]] = None
    is_explicit: bool = False
    is_autoinc: bool = False
    default_characterization: _SentinelDefaultCharacterization = (
        _SentinelDefaultCharacterization.NONE
    )


_COLKEY = TypeVar("_COLKEY", Union[None, str], str)

_COL_co = TypeVar("_COL_co", bound="ColumnElement[Any]", covariant=True)
_COL = TypeVar("_COL", bound="ColumnElement[Any]")


class _ColumnMetrics(Generic[_COL_co]):
    __slots__ = ("column",)

    column: _COL_co

    def __init__(
        self, collection: ColumnCollection[Any, _COL_co], col: _COL_co
    ):
        self.column = col

        # proxy_index being non-empty means it was initialized.
        # so we need to update it
        pi = collection._proxy_index
        if pi:
            for eps_col in col._expanded_proxy_set:
                pi[eps_col].add(self)

    def get_expanded_proxy_set(self):
        return self.column._expanded_proxy_set

    def dispose(self, collection):
        pi = collection._proxy_index
        if not pi:
            return
        for col in self.column._expanded_proxy_set:
            colset = pi.get(col, None)
            if colset:
                colset.discard(self)
            if colset is not None and not colset:
                del pi[col]

    def embedded(
        self,
        target_set: Union[
            Set[ColumnElement[Any]], FrozenSet[ColumnElement[Any]]
        ],
    ) -> bool:
        expanded_proxy_set = self.column._expanded_proxy_set
        for t in target_set.difference(expanded_proxy_set):
            if not expanded_proxy_set.intersection(_expand_cloned([t])):
                return False
        return True


class ColumnCollection(Generic[_COLKEY, _COL_co]):
    """Collection of :class:`_expression.ColumnElement` instances,
    typically for
    :class:`_sql.FromClause` objects.

    The :class:`_sql.ColumnCollection` object is most commonly available
    as the :attr:`_schema.Table.c` or :attr:`_schema.Table.columns` collection
    on the :class:`_schema.Table` object, introduced at
    :ref:`metadata_tables_and_columns`.

    The :class:`_expression.ColumnCollection` has both mapping- and sequence-
    like behaviors. A :class:`_expression.ColumnCollection` usually stores
    :class:`_schema.Column` objects, which are then accessible both via mapping
    style access as well as attribute access style.

    To access :class:`_schema.Column` objects using ordinary attribute-style
    access, specify the name like any other object attribute, such as below
    a column named ``employee_name`` is accessed::

        >>> employee_table.c.employee_name

    To access columns that have names with special characters or spaces,
    index-style access is used, such as below which illustrates a column named
    ``employee ' payment`` is accessed::

        >>> employee_table.c["employee ' payment"]

    As the :class:`_sql.ColumnCollection` object provides a Python dictionary
    interface, common dictionary method names like
    :meth:`_sql.ColumnCollection.keys`, :meth:`_sql.ColumnCollection.values`,
    and :meth:`_sql.ColumnCollection.items` are available, which means that
    database columns that are keyed under these names also need to use indexed
    access::

        >>> employee_table.c["values"]


    The name for which a :class:`_schema.Column` would be present is normally
    that of the :paramref:`_schema.Column.key` parameter.  In some contexts,
    such as a :class:`_sql.Select` object that uses a label style set
    using the :meth:`_sql.Select.set_label_style` method, a column of a certain
    key may instead be represented under a particular label name such
    as ``tablename_columnname``::

        >>> from sqlalchemy import select, column, table
        >>> from sqlalchemy import LABEL_STYLE_TABLENAME_PLUS_COL
        >>> t = table("t", column("c"))
        >>> stmt = select(t).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        >>> subq = stmt.subquery()
        >>> subq.c.t_c
        <sqlalchemy.sql.elements.ColumnClause at 0x7f59dcf04fa0; t_c>

    :class:`.ColumnCollection` also indexes the columns in order and allows
    them to be accessible by their integer position::

        >>> cc[0]
        Column('x', Integer(), table=None)
        >>> cc[1]
        Column('y', Integer(), table=None)

    .. versionadded:: 1.4 :class:`_expression.ColumnCollection`
       allows integer-based
       index access to the collection.

    Iterating the collection yields the column expressions in order::

        >>> list(cc)
        [Column('x', Integer(), table=None),
         Column('y', Integer(), table=None)]

    The base :class:`_expression.ColumnCollection` object can store
    duplicates, which can
    mean either two columns with the same key, in which case the column
    returned by key  access is **arbitrary**::

        >>> x1, x2 = Column("x", Integer), Column("x", Integer)
        >>> cc = ColumnCollection(columns=[(x1.name, x1), (x2.name, x2)])
        >>> list(cc)
        [Column('x', Integer(), table=None),
         Column('x', Integer(), table=None)]
        >>> cc["x"] is x1
        False
        >>> cc["x"] is x2
        True

    Or it can also mean the same column multiple times.   These cases are
    supported as :class:`_expression.ColumnCollection`
    is used to represent the columns in
    a SELECT statement which may include duplicates.

    A special subclass :class:`.DedupeColumnCollection` exists which instead
    maintains SQLAlchemy's older behavior of not allowing duplicates; this
    collection is used for schema level objects like :class:`_schema.Table`
    and
    :class:`.PrimaryKeyConstraint` where this deduping is helpful.  The
    :class:`.DedupeColumnCollection` class also has additional mutation methods
    as the schema constructs have more use cases that require removal and
    replacement of columns.

    .. versionchanged:: 1.4 :class:`_expression.ColumnCollection`
       now stores duplicate
       column keys as well as the same column in multiple positions.  The
       :class:`.DedupeColumnCollection` class is added to maintain the
       former behavior in those cases where deduplication as well as
       additional replace/remove operations are needed.


    """

    __slots__ = "_collection", "_index", "_colset", "_proxy_index"

    _collection: List[Tuple[_COLKEY, _COL_co, _ColumnMetrics[_COL_co]]]
    _index: Dict[Union[None, str, int], Tuple[_COLKEY, _COL_co]]
    _proxy_index: Dict[ColumnElement[Any], Set[_ColumnMetrics[_COL_co]]]
    _colset: Set[_COL_co]

    def __init__(
        self, columns: Optional[Iterable[Tuple[_COLKEY, _COL_co]]] = None
    ):
        object.__setattr__(self, "_colset", set())
        object.__setattr__(self, "_index", {})
        object.__setattr__(
            self, "_proxy_index", collections.defaultdict(util.OrderedSet)
        )
        object.__setattr__(self, "_collection", [])
        if columns:
            self._initial_populate(columns)

    @util.preload_module("sqlalchemy.sql.elements")
    def __clause_element__(self) -> ClauseList:
        elements = util.preloaded.sql_elements

        return elements.ClauseList(
            _literal_as_text_role=roles.ColumnsClauseRole,
            group=False,
            *self._all_columns,
        )

    def _initial_populate(
        self, iter_: Iterable[Tuple[_COLKEY, _COL_co]]
    ) -> None:
        self._populate_separate_keys(iter_)

    @property
    def _all_columns(self) -> List[_COL_co]:
        return [col for (_, col, _) in self._collection]

    def keys(self) -> List[_COLKEY]:
        """Return a sequence of string key names for all columns in this
        collection."""
        return [k for (k, _, _) in self._collection]

    def values(self) -> List[_COL_co]:
        """Return a sequence of :class:`_sql.ColumnClause` or
        :class:`_schema.Column` objects for all columns in this
        collection."""
        return [col for (_, col, _) in self._collection]

    def items(self) -> List[Tuple[_COLKEY, _COL_co]]:
        """Return a sequence of (key, column) tuples for all columns in this
        collection each consisting of a string key name and a
        :class:`_sql.ColumnClause` or
        :class:`_schema.Column` object.
        """

        return [(k, col) for (k, col, _) in self._collection]

    def __bool__(self) -> bool:
        return bool(self._collection)

    def __len__(self) -> int:
        return len(self._collection)

    def __iter__(self) -> Iterator[_COL_co]:
        # turn to a list first to maintain over a course of changes
        return iter([col for _, col, _ in self._collection])

    @overload
    def __getitem__(self, key: Union[str, int]) -> _COL_co: ...

    @overload
    def __getitem__(
        self, key: Tuple[Union[str, int], ...]
    ) -> ReadOnlyColumnCollection[_COLKEY, _COL_co]: ...

    @overload
    def __getitem__(
        self, key: slice
    ) -> ReadOnlyColumnCollection[_COLKEY, _COL_co]: ...

    def __getitem__(
        self, key: Union[str, int, slice, Tuple[Union[str, int], ...]]
    ) -> Union[ReadOnlyColumnCollection[_COLKEY, _COL_co], _COL_co]:
        try:
            if isinstance(key, (tuple, slice)):
                if isinstance(key, slice):
                    cols = (
                        (sub_key, col)
                        for (sub_key, col, _) in self._collection[key]
                    )
                else:
                    cols = (self._index[sub_key] for sub_key in key)

                return ColumnCollection(cols).as_readonly()
            else:
                return self._index[key][1]
        except KeyError as err:
            if isinstance(err.args[0], int):
                raise IndexError(err.args[0]) from err
            else:
                raise

    def __getattr__(self, key: str) -> _COL_co:
        try:
            return self._index[key][1]
        except KeyError as err:
            raise AttributeError(key) from err

    def __contains__(self, key: str) -> bool:
        if key not in self._index:
            if not isinstance(key, str):
                raise exc.ArgumentError(
                    "__contains__ requires a string argument"
                )
            return False
        else:
            return True

    def compare(self, other: ColumnCollection[Any, Any]) -> bool:
        """Compare this :class:`_expression.ColumnCollection` to another
        based on the names of the keys"""

        for l, r in zip_longest(self, other):
            if l is not r:
                return False
        else:
            return True

    def __eq__(self, other: Any) -> bool:
        return self.compare(other)

    @overload
    def get(self, key: str, default: None = None) -> Optional[_COL_co]: ...

    @overload
    def get(self, key: str, default: _COL) -> Union[_COL_co, _COL]: ...

    def get(
        self, key: str, default: Optional[_COL] = None
    ) -> Optional[Union[_COL_co, _COL]]:
        """Get a :class:`_sql.ColumnClause` or :class:`_schema.Column` object
        based on a string key name from this
        :class:`_expression.ColumnCollection`."""

        if key in self._index:
            return self._index[key][1]
        else:
            return default

    def __str__(self) -> str:
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(str(c) for c in self),
        )

    def __setitem__(self, key: str, value: Any) -> NoReturn:
        raise NotImplementedError()

    def __delitem__(self, key: str) -> NoReturn:
        raise NotImplementedError()

    def __setattr__(self, key: str, obj: Any) -> NoReturn:
        raise NotImplementedError()

    def clear(self) -> NoReturn:
        """Dictionary clear() is not implemented for
        :class:`_sql.ColumnCollection`."""
        raise NotImplementedError()

    def remove(self, column: Any) -> None:
        raise NotImplementedError()

    def update(self, iter_: Any) -> NoReturn:
        """Dictionary update() is not implemented for
        :class:`_sql.ColumnCollection`."""
        raise NotImplementedError()

    # https://github.com/python/mypy/issues/4266
    __hash__ = None  # type: ignore

    def _populate_separate_keys(
        self, iter_: Iterable[Tuple[_COLKEY, _COL_co]]
    ) -> None:
        """populate from an iterator of (key, column)"""

        self._collection[:] = collection = [
            (k, c, _ColumnMetrics(self, c)) for k, c in iter_
        ]
        self._colset.update(c._deannotate() for _, c, _ in collection)
        self._index.update(
            {idx: (k, c) for idx, (k, c, _) in enumerate(collection)}
        )
        self._index.update({k: (k, col) for k, col, _ in reversed(collection)})

    def add(
        self, column: ColumnElement[Any], key: Optional[_COLKEY] = None
    ) -> None:
        """Add a column to this :class:`_sql.ColumnCollection`.

        .. note::

            This method is **not normally used by user-facing code**, as the
            :class:`_sql.ColumnCollection` is usually part of an existing
            object such as a :class:`_schema.Table`. To add a
            :class:`_schema.Column` to an existing :class:`_schema.Table`
            object, use the :meth:`_schema.Table.append_column` method.

        """
        colkey: _COLKEY

        if key is None:
            colkey = column.key  # type: ignore
        else:
            colkey = key

        l = len(self._collection)

        # don't really know how this part is supposed to work w/ the
        # covariant thing

        _column = cast(_COL_co, column)

        self._collection.append(
            (colkey, _column, _ColumnMetrics(self, _column))
        )
        self._colset.add(_column._deannotate())
        self._index[l] = (colkey, _column)
        if colkey not in self._index:
            self._index[colkey] = (colkey, _column)

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "_collection": [(k, c) for k, c, _ in self._collection],
            "_index": self._index,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        object.__setattr__(self, "_index", state["_index"])
        object.__setattr__(
            self, "_proxy_index", collections.defaultdict(util.OrderedSet)
        )
        object.__setattr__(
            self,
            "_collection",
            [
                (k, c, _ColumnMetrics(self, c))
                for (k, c) in state["_collection"]
            ],
        )
        object.__setattr__(
            self, "_colset", {col for k, col, _ in self._collection}
        )

    def contains_column(self, col: ColumnElement[Any]) -> bool:
        """Checks if a column object exists in this collection"""
        if col not in self._colset:
            if isinstance(col, str):
                raise exc.ArgumentError(
                    "contains_column cannot be used with string arguments. "
                    "Use ``col_name in table.c`` instead."
                )
            return False
        else:
            return True

    def as_readonly(self) -> ReadOnlyColumnCollection[_COLKEY, _COL_co]:
        """Return a "read only" form of this
        :class:`_sql.ColumnCollection`."""

        return ReadOnlyColumnCollection(self)

    def _init_proxy_index(self):
        """populate the "proxy index", if empty.

        proxy index is added in 2.0 to provide more efficient operation
        for the corresponding_column() method.

        For reasons of both time to construct new .c collections as well as
        memory conservation for large numbers of large .c collections, the
        proxy_index is only filled if corresponding_column() is called. once
        filled it stays that way, and new _ColumnMetrics objects created after
        that point will populate it with new data. Note this case would be
        unusual, if not nonexistent, as it means a .c collection is being
        mutated after corresponding_column() were used, however it is tested in
        test/base/test_utils.py.

        """
        pi = self._proxy_index
        if pi:
            return

        for _, _, metrics in self._collection:
            eps = metrics.column._expanded_proxy_set

            for eps_col in eps:
                pi[eps_col].add(metrics)

    def corresponding_column(
        self, column: _COL, require_embedded: bool = False
    ) -> Optional[Union[_COL, _COL_co]]:
        """Given a :class:`_expression.ColumnElement`, return the exported
        :class:`_expression.ColumnElement` object from this
        :class:`_expression.ColumnCollection`
        which corresponds to that original :class:`_expression.ColumnElement`
        via a common
        ancestor column.

        :param column: the target :class:`_expression.ColumnElement`
                      to be matched.

        :param require_embedded: only return corresponding columns for
         the given :class:`_expression.ColumnElement`, if the given
         :class:`_expression.ColumnElement`
         is actually present within a sub-element
         of this :class:`_expression.Selectable`.
         Normally the column will match if
         it merely shares a common ancestor with one of the exported
         columns of this :class:`_expression.Selectable`.

        .. seealso::

            :meth:`_expression.Selectable.corresponding_column`
            - invokes this method
            against the collection returned by
            :attr:`_expression.Selectable.exported_columns`.

        .. versionchanged:: 1.4 the implementation for ``corresponding_column``
           was moved onto the :class:`_expression.ColumnCollection` itself.

        """
        # TODO: cython candidate

        # don't dig around if the column is locally present
        if column in self._colset:
            return column

        selected_intersection, selected_metrics = None, None
        target_set = column.proxy_set

        pi = self._proxy_index
        if not pi:
            self._init_proxy_index()

        for current_metrics in (
            mm for ts in target_set if ts in pi for mm in pi[ts]
        ):
            if not require_embedded or current_metrics.embedded(target_set):
                if selected_metrics is None:
                    # no corresponding column yet, pick this one.
                    selected_metrics = current_metrics
                    continue

                current_intersection = target_set.intersection(
                    current_metrics.column._expanded_proxy_set
                )
                if selected_intersection is None:
                    selected_intersection = target_set.intersection(
                        selected_metrics.column._expanded_proxy_set
                    )

                if len(current_intersection) > len(selected_intersection):
                    # 'current' has a larger field of correspondence than
                    # 'selected'. i.e. selectable.c.a1_x->a1.c.x->table.c.x
                    # matches a1.c.x->table.c.x better than
                    # selectable.c.x->table.c.x does.

                    selected_metrics = current_metrics
                    selected_intersection = current_intersection
                elif current_intersection == selected_intersection:
                    # they have the same field of correspondence. see
                    # which proxy_set has fewer columns in it, which
                    # indicates a closer relationship with the root
                    # column. Also take into account the "weight"
                    # attribute which CompoundSelect() uses to give
                    # higher precedence to columns based on vertical
                    # position in the compound statement, and discard
                    # columns that have no reference to the target
                    # column (also occurs with CompoundSelect)

                    selected_col_distance = sum(
                        [
                            sc._annotations.get("weight", 1)
                            for sc in (
                                selected_metrics.column._uncached_proxy_list()
                            )
                            if sc.shares_lineage(column)
                        ],
                    )
                    current_col_distance = sum(
                        [
                            sc._annotations.get("weight", 1)
                            for sc in (
                                current_metrics.column._uncached_proxy_list()
                            )
                            if sc.shares_lineage(column)
                        ],
                    )
                    if current_col_distance < selected_col_distance:
                        selected_metrics = current_metrics
                        selected_intersection = current_intersection

        return selected_metrics.column if selected_metrics else None


_NAMEDCOL = TypeVar("_NAMEDCOL", bound="NamedColumn[Any]")


class DedupeColumnCollection(ColumnCollection[str, _NAMEDCOL]):
    """A :class:`_expression.ColumnCollection`
    that maintains deduplicating behavior.

    This is useful by schema level objects such as :class:`_schema.Table` and
    :class:`.PrimaryKeyConstraint`.    The collection includes more
    sophisticated mutator methods as well to suit schema objects which
    require mutable column collections.

    .. versionadded:: 1.4

    """

    def add(  # type: ignore[override]
        self, column: _NAMEDCOL, key: Optional[str] = None
    ) -> None:
        if key is not None and column.key != key:
            raise exc.ArgumentError(
                "DedupeColumnCollection requires columns be under "
                "the same key as their .key"
            )
        key = column.key

        if key is None:
            raise exc.ArgumentError(
                "Can't add unnamed column to column collection"
            )

        if key in self._index:
            existing = self._index[key][1]

            if existing is column:
                return

            self.replace(column)

            # pop out memoized proxy_set as this
            # operation may very well be occurring
            # in a _make_proxy operation
            util.memoized_property.reset(column, "proxy_set")
        else:
            self._append_new_column(key, column)

    def _append_new_column(self, key: str, named_column: _NAMEDCOL) -> None:
        l = len(self._collection)
        self._collection.append(
            (key, named_column, _ColumnMetrics(self, named_column))
        )
        self._colset.add(named_column._deannotate())
        self._index[l] = (key, named_column)
        self._index[key] = (key, named_column)

    def _populate_separate_keys(
        self, iter_: Iterable[Tuple[str, _NAMEDCOL]]
    ) -> None:
        """populate from an iterator of (key, column)"""
        cols = list(iter_)

        replace_col = []
        for k, col in cols:
            if col.key != k:
                raise exc.ArgumentError(
                    "DedupeColumnCollection requires columns be under "
                    "the same key as their .key"
                )
            if col.name in self._index and col.key != col.name:
                replace_col.append(col)
            elif col.key in self._index:
                replace_col.append(col)
            else:
                self._index[k] = (k, col)
                self._collection.append((k, col, _ColumnMetrics(self, col)))
        self._colset.update(c._deannotate() for (k, c, _) in self._collection)

        self._index.update(
            (idx, (k, c)) for idx, (k, c, _) in enumerate(self._collection)
        )
        for col in replace_col:
            self.replace(col)

    def extend(self, iter_: Iterable[_NAMEDCOL]) -> None:
        self._populate_separate_keys((col.key, col) for col in iter_)

    def remove(self, column: _NAMEDCOL) -> None:
        if column not in self._colset:
            raise ValueError(
                "Can't remove column %r; column is not in this collection"
                % column
            )
        del self._index[column.key]
        self._colset.remove(column)
        self._collection[:] = [
            (k, c, metrics)
            for (k, c, metrics) in self._collection
            if c is not column
        ]
        for metrics in self._proxy_index.get(column, ()):
            metrics.dispose(self)

        self._index.update(
            {idx: (k, col) for idx, (k, col, _) in enumerate(self._collection)}
        )
        # delete higher index
        del self._index[len(self._collection)]

    def replace(
        self,
        column: _NAMEDCOL,
        extra_remove: Optional[Iterable[_NAMEDCOL]] = None,
    ) -> None:
        """add the given column to this collection, removing unaliased
        versions of this column  as well as existing columns with the
        same key.

        e.g.::

            t = Table("sometable", metadata, Column("col1", Integer))
            t.columns.replace(Column("col1", Integer, key="columnone"))

        will remove the original 'col1' from the collection, and add
        the new column under the name 'columnname'.

        Used by schema.Column to override columns during table reflection.

        """

        if extra_remove:
            remove_col = set(extra_remove)
        else:
            remove_col = set()
        # remove up to two columns based on matches of name as well as key
        if column.name in self._index and column.key != column.name:
            other = self._index[column.name][1]
            if other.name == other.key:
                remove_col.add(other)

        if column.key in self._index:
            remove_col.add(self._index[column.key][1])

        if not remove_col:
            self._append_new_column(column.key, column)
            return
        new_cols: List[Tuple[str, _NAMEDCOL, _ColumnMetrics[_NAMEDCOL]]] = []
        replaced = False
        for k, col, metrics in self._collection:
            if col in remove_col:
                if not replaced:
                    replaced = True
                    new_cols.append(
                        (column.key, column, _ColumnMetrics(self, column))
                    )
            else:
                new_cols.append((k, col, metrics))

        if remove_col:
            self._colset.difference_update(remove_col)

            for rc in remove_col:
                for metrics in self._proxy_index.get(rc, ()):
                    metrics.dispose(self)

        if not replaced:
            new_cols.append((column.key, column, _ColumnMetrics(self, column)))

        self._colset.add(column._deannotate())
        self._collection[:] = new_cols

        self._index.clear()

        self._index.update(
            {idx: (k, col) for idx, (k, col, _) in enumerate(self._collection)}
        )
        self._index.update({k: (k, col) for (k, col, _) in self._collection})


class ReadOnlyColumnCollection(
    util.ReadOnlyContainer, ColumnCollection[_COLKEY, _COL_co]
):
    __slots__ = ("_parent",)

    def __init__(self, collection):
        object.__setattr__(self, "_parent", collection)
        object.__setattr__(self, "_colset", collection._colset)
        object.__setattr__(self, "_index", collection._index)
        object.__setattr__(self, "_collection", collection._collection)
        object.__setattr__(self, "_proxy_index", collection._proxy_index)

    def __getstate__(self):
        return {"_parent": self._parent}

    def __setstate__(self, state):
        parent = state["_parent"]
        self.__init__(parent)  # type: ignore

    def add(self, column: Any, key: Any = ...) -> Any:
        self._readonly()

    def extend(self, elements: Any) -> NoReturn:
        self._readonly()

    def remove(self, item: Any) -> NoReturn:
        self._readonly()


class ColumnSet(util.OrderedSet["ColumnClause[Any]"]):
    def contains_column(self, col):
        return col in self

    def extend(self, cols):
        for col in cols:
            self.add(col)

    def __eq__(self, other):
        l = []
        for c in other:
            for local in self:
                if c.shares_lineage(local):
                    l.append(c == local)
        return elements.and_(*l)

    def __hash__(self):  # type: ignore[override]
        return hash(tuple(x for x in self))


def _entity_namespace(
    entity: Union[_HasEntityNamespace, ExternallyTraversible]
) -> _EntityNamespace:
    """Return the nearest .entity_namespace for the given entity.

    If not immediately available, does an iterate to find a sub-element
    that has one, if any.

    """
    try:
        return cast(_HasEntityNamespace, entity).entity_namespace
    except AttributeError:
        for elem in visitors.iterate(cast(ExternallyTraversible, entity)):
            if _is_has_entity_namespace(elem):
                return elem.entity_namespace
        else:
            raise


def _entity_namespace_key(
    entity: Union[_HasEntityNamespace, ExternallyTraversible],
    key: str,
    default: Union[SQLCoreOperations[Any], _NoArg] = NO_ARG,
) -> SQLCoreOperations[Any]:
    """Return an entry from an entity_namespace.


    Raises :class:`_exc.InvalidRequestError` rather than attribute error
    on not found.

    """

    try:
        ns = _entity_namespace(entity)
        if default is not NO_ARG:
            return getattr(ns, key, default)
        else:
            return getattr(ns, key)  # type: ignore
    except AttributeError as err:
        raise exc.InvalidRequestError(
            'Entity namespace for "%s" has no property "%s"' % (entity, key)
        ) from err

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\sdm.py ===
"""

Module for the SDM class.

"""

from operator import add, neg, pos, sub, mul
from collections import defaultdict

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import _strongly_connected_components

from .exceptions import DMBadInputError, DMDomainError, DMShapeError

from sympy.polys.domains import QQ

from .ddm import DDM


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['SDM.to_dfm', 'SDM.to_dfm_or_ddm']


class SDM(dict):
    r"""Sparse matrix based on polys domain elements

    This is a dict subclass and is a wrapper for a dict of dicts that supports
    basic matrix arithmetic +, -, *, **.


    In order to create a new :py:class:`~.SDM`, a dict
    of dicts mapping non-zero elements to their
    corresponding row and column in the matrix is needed.

    We also need to specify the shape and :py:class:`~.Domain`
    of our :py:class:`~.SDM` object.

    We declare a 2x2 :py:class:`~.SDM` matrix belonging
    to QQ domain as shown below.
    The 2x2 Matrix in the example is

    .. math::
           A = \left[\begin{array}{ccc}
                0 & \frac{1}{2} \\
                0 & 0 \end{array} \right]


    >>> from sympy.polys.matrices.sdm import SDM
    >>> from sympy import QQ
    >>> elemsdict = {0:{1:QQ(1, 2)}}
    >>> A = SDM(elemsdict, (2, 2), QQ)
    >>> A
    {0: {1: 1/2}}

    We can manipulate :py:class:`~.SDM` the same way
    as a Matrix class

    >>> from sympy import ZZ
    >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
    >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
    >>> A + B
    {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

    Multiplication

    >>> A*B
    {0: {1: 8}, 1: {0: 3}}
    >>> A*ZZ(2)
    {0: {1: 4}, 1: {0: 2}}

    """

    fmt = 'sparse'
    is_DFM = False
    is_DDM = False

    def __init__(self, elemsdict, shape, domain):
        super().__init__(elemsdict)
        self.shape = self.rows, self.cols = m, n = shape
        self.domain = domain

        if not all(0 <= r < m for r in self):
            raise DMBadInputError("Row out of range")
        if not all(0 <= c < n for row in self.values() for c in row):
            raise DMBadInputError("Column out of range")

    def getitem(self, i, j):
        try:
            return self[i][j]
        except KeyError:
            m, n = self.shape
            if -m <= i < m and -n <= j < n:
                try:
                    return self[i % m][j % n]
                except KeyError:
                    return self.domain.zero
            else:
                raise IndexError("index out of range")

    def setitem(self, i, j, value):
        m, n = self.shape
        if not (-m <= i < m and -n <= j < n):
            raise IndexError("index out of range")
        i, j = i % m, j % n
        if value:
            try:
                self[i][j] = value
            except KeyError:
                self[i] = {j: value}
        else:
            rowi = self.get(i, None)
            if rowi is not None:
                try:
                    del rowi[j]
                except KeyError:
                    pass
                else:
                    if not rowi:
                        del self[i]

    def extract_slice(self, slice1, slice2):
        m, n = self.shape
        ri = range(m)[slice1]
        ci = range(n)[slice2]

        sdm = {}
        for i, row in self.items():
            if i in ri:
                row = {ci.index(j): e for j, e in row.items() if j in ci}
                if row:
                    sdm[ri.index(i)] = row

        return self.new(sdm, (len(ri), len(ci)), self.domain)

    def extract(self, rows, cols):
        if not (self and rows and cols):
            return self.zeros((len(rows), len(cols)), self.domain)

        m, n = self.shape
        if not (-m <= min(rows) <= max(rows) < m):
            raise IndexError('Row index out of range')
        if not (-n <= min(cols) <= max(cols) < n):
            raise IndexError('Column index out of range')

        # rows and cols can contain duplicates e.g. M[[1, 2, 2], [0, 1]]
        # Build a map from row/col in self to list of rows/cols in output
        rowmap = defaultdict(list)
        colmap = defaultdict(list)
        for i2, i1 in enumerate(rows):
            rowmap[i1 % m].append(i2)
        for j2, j1 in enumerate(cols):
            colmap[j1 % n].append(j2)

        # Used to efficiently skip zero rows/cols
        rowset = set(rowmap)
        colset = set(colmap)

        sdm1 = self
        sdm2 = {}
        for i1 in rowset & sdm1.keys():
            row1 = sdm1[i1]
            row2 = {}
            for j1 in colset & row1.keys():
                row1_j1 = row1[j1]
                for j2 in colmap[j1]:
                    row2[j2] = row1_j1
            if row2:
                for i2 in rowmap[i1]:
                    sdm2[i2] = row2.copy()

        return self.new(sdm2, (len(rows), len(cols)), self.domain)

    def __str__(self):
        rowsstr = []
        for i, row in self.items():
            elemsstr = ', '.join('%s: %s' % (j, elem) for j, elem in row.items())
            rowsstr.append('%s: {%s}' % (i, elemsstr))
        return '{%s}' % ', '.join(rowsstr)

    def __repr__(self):
        cls = type(self).__name__
        rows = dict.__repr__(self)
        return '%s(%s, %s, %s)' % (cls, rows, self.shape, self.domain)

    @classmethod
    def new(cls, sdm, shape, domain):
        """

        Parameters
        ==========

        sdm: A dict of dicts for non-zero elements in SDM
        shape: tuple representing dimension of SDM
        domain: Represents :py:class:`~.Domain` of SDM

        Returns
        =======

        An :py:class:`~.SDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1: QQ(2)}}
        >>> A = SDM.new(elemsdict, (2, 2), QQ)
        >>> A
        {0: {1: 2}}

        """
        return cls(sdm, shape, domain)

    def copy(A):
        """
        Returns the copy of a :py:class:`~.SDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}
        >>> A = SDM(elemsdict, (2, 2), QQ)
        >>> B = A.copy()
        >>> B
        {0: {1: 2}, 1: {}}

        """
        Ac = {i: Ai.copy() for i, Ai in A.items()}
        return A.new(Ac, A.shape, A.domain)

    @classmethod
    def from_list(cls, ddm, shape, domain):
        """
        Create :py:class:`~.SDM` object from a list of lists.

        Parameters
        ==========

        ddm:
            list of lists containing domain elements
        shape:
            Dimensions of :py:class:`~.SDM` matrix
        domain:
            Represents :py:class:`~.Domain` of :py:class:`~.SDM` object

        Returns
        =======

        :py:class:`~.SDM` containing elements of ddm

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = [[QQ(1, 2), QQ(0)], [QQ(0), QQ(3, 4)]]
        >>> A = SDM.from_list(ddm, (2, 2), QQ)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}

        See Also
        ========

        to_list
        from_list_flat
        from_dok
        from_ddm
        """

        m, n = shape
        if not (len(ddm) == m and all(len(row) == n for row in ddm)):
            raise DMBadInputError("Inconsistent row-list/shape")
        getrow = lambda i: {j:ddm[i][j] for j in range(n) if ddm[i][j]}
        irows = ((i, getrow(i)) for i in range(m))
        sdm = {i: row for i, row in irows if row}
        return cls(sdm, shape, domain)

    @classmethod
    def from_ddm(cls, ddm):
        """
        Create :py:class:`~.SDM` from a :py:class:`~.DDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = DDM( [[QQ(1, 2), 0], [0, QQ(3, 4)]], (2, 2), QQ)
        >>> A = SDM.from_ddm(ddm)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}
        >>> SDM.from_ddm(ddm).to_ddm() == ddm
        True

        See Also
        ========

        to_ddm
        from_list
        from_list_flat
        from_dok
        """
        return cls.from_list(ddm, ddm.shape, ddm.domain)

    def to_list(M):
        """
        Convert a :py:class:`~.SDM` object to a list of lists.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}
        >>> A = SDM(elemsdict, (2, 2), QQ)
        >>> A.to_list()
        [[0, 2], [0, 0]]


        """
        m, n = M.shape
        zero = M.domain.zero
        ddm = [[zero] * n for _ in range(m)]
        for i, row in M.items():
            for j, e in row.items():
                ddm[i][j] = e
        return ddm

    def to_list_flat(M):
        """
        Convert :py:class:`~.SDM` to a flat list.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_list_flat()
        [0, 2, 3, 0]
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        from_list_flat
        to_list
        to_dok
        to_ddm
        """
        m, n = M.shape
        zero = M.domain.zero
        flat = [zero] * (m * n)
        for i, row in M.items():
            for j, e in row.items():
                flat[i*n + j] = e
        return flat

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """
        Create :py:class:`~.SDM` from a flat list of elements.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM.from_list_flat([QQ(0), QQ(2), QQ(0), QQ(0)], (2, 2), QQ)
        >>> A
        {0: {1: 2}}
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        from_list
        from_dok
        from_ddm
        """
        m, n = shape
        if len(elements) != m * n:
            raise DMBadInputError("Inconsistent flat-list shape")
        sdm = defaultdict(dict)
        for inj, element in enumerate(elements):
            if element:
                i, j = divmod(inj, n)
                sdm[i][j] = element
        return cls(sdm, shape, domain)

    def to_flat_nz(M):
        """
        Convert :class:`SDM` to a flat list of nonzero elements and data.

        Explanation
        ===========

        This is used to operate on a list of the elements of a matrix and then
        reconstruct a modified matrix with elements in the same positions using
        :meth:`from_flat_nz`. Zero elements are omitted from the list.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [2, 3]
        >>> A == A.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        from_flat_nz
        to_list_flat
        sympy.polys.matrices.ddm.DDM.to_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz
        """
        dok = M.to_dok()
        indices = tuple(dok)
        elements = list(dok.values())
        data = (indices, M.shape)
        return elements, data

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        Reconstruct a :class:`~.SDM` after calling :meth:`to_flat_nz`.

        See :meth:`to_flat_nz` for explanation.

        See Also
        ========

        to_flat_nz
        from_list_flat
        sympy.polys.matrices.ddm.DDM.from_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz
        """
        indices, shape = data
        dok = dict(zip(indices, elements))
        return cls.from_dok(dok, shape, domain)

    def to_dod(M):
        """
        Convert to dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_dod()
        {0: {1: 2}, 1: {0: 3}}

        See Also
        ========

        from_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        return {i: row.copy() for i, row in M.items()}

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create :py:class:`~.SDM` from dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> dod = {0: {1: QQ(2)}, 1: {0: QQ(3)}}
        >>> A = SDM.from_dod(dod, (2, 2), QQ)
        >>> A
        {0: {1: 2}, 1: {0: 3}}
        >>> A == SDM.from_dod(A.to_dod(), A.shape, A.domain)
        True

        See Also
        ========

        to_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
        sdm = defaultdict(dict)
        for i, row in dod.items():
            for j, e in row.items():
                if e:
                    sdm[i][j] = e
        return cls(sdm, shape, domain)

    def to_dok(M):
        """
        Convert to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_dok()
        {(0, 1): 2, (1, 0): 3}

        See Also
        ========

        from_dok
        to_list
        to_list_flat
        to_ddm
        """
        return {(i, j): e for i, row in M.items() for j, e in row.items()}

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create :py:class:`~.SDM` from dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> dok = {(0, 1): QQ(2), (1, 0): QQ(3)}
        >>> A = SDM.from_dok(dok, (2, 2), QQ)
        >>> A
        {0: {1: 2}, 1: {0: 3}}
        >>> A == SDM.from_dok(A.to_dok(), A.shape, A.domain)
        True

        See Also
        ========

        to_dok
        from_list
        from_list_flat
        from_ddm
        """
        sdm = defaultdict(dict)
        for (i, j), e in dok.items():
            if e:
                sdm[i][j] = e
        return cls(sdm, shape, domain)

    def iter_values(M):
        """
        Iterate over the nonzero values of a :py:class:`~.SDM` matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> list(A.iter_values())
        [2, 3]

        """
        for row in M.values():
            yield from row.values()

    def iter_items(M):
        """
        Iterate over indices and values of the nonzero elements.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> list(A.iter_items())
        [((0, 1), 2), ((1, 0), 3)]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_items
        """
        for i, row in M.items():
            for j, e in row.items():
                yield (i, j), e

    def to_ddm(M):
        """
        Convert a :py:class:`~.SDM` object to a :py:class:`~.DDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_ddm()
        [[0, 2], [0, 0]]

        """
        return DDM(M.to_list(), M.shape, M.domain)

    def to_sdm(M):
        """
        Convert to :py:class:`~.SDM` format (returns self).
        """
        return M

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(M):
        """
        Convert a :py:class:`~.SDM` object to a :py:class:`~.DFM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_dfm()
        [[0, 2], [0, 0]]

        See Also
        ========

        to_ddm
        to_dfm_or_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm
        """
        return M.to_ddm().to_dfm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(M):
        """
        Convert to :py:class:`~.DFM` if possible, else :py:class:`~.DDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_dfm_or_ddm()
        [[0, 2], [0, 0]]
        >>> type(A.to_dfm_or_ddm())  # depends on the ground types
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm
        to_dfm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
        return M.to_ddm().to_dfm_or_ddm()

    @classmethod
    def zeros(cls, shape, domain):
        r"""

        Returns a :py:class:`~.SDM` of size shape,
        belonging to the specified domain

        In the example below we declare a matrix A where,

        .. math::
            A := \left[\begin{array}{ccc}
            0 & 0 & 0 \\
            0 & 0 & 0 \end{array} \right]

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM.zeros((2, 3), QQ)
        >>> A
        {}

        """
        return cls({}, shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        one = domain.one
        m, n = shape
        row = dict(zip(range(n), [one]*n))
        sdm = {i: row.copy() for i in range(m)}
        return cls(sdm, shape, domain)

    @classmethod
    def eye(cls, shape, domain):
        """

        Returns a identity :py:class:`~.SDM` matrix of dimensions
        size x size, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> I = SDM.eye((2, 2), QQ)
        >>> I
        {0: {0: 1}, 1: {1: 1}}

        """
        if isinstance(shape, int):
            rows, cols = shape, shape
        else:
            rows, cols = shape
        one = domain.one
        sdm = {i: {i: one} for i in range(min(rows, cols))}
        return cls(sdm, (rows, cols), domain)

    @classmethod
    def diag(cls, diagonal, domain, shape=None):
        if shape is None:
            shape = (len(diagonal), len(diagonal))
        sdm = {i: {i: v} for i, v in enumerate(diagonal) if v}
        return cls(sdm, shape, domain)

    def transpose(M):
        """

        Returns the transpose of a :py:class:`~.SDM` matrix

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.transpose()
        {1: {0: 2}}

        """
        MT = sdm_transpose(M)
        return M.new(MT, M.shape[::-1], M.domain)

    def __add__(A, B):
        if not isinstance(B, SDM):
            return NotImplemented
        elif A.shape != B.shape:
            raise DMShapeError("Matrix size mismatch: %s + %s" % (A.shape, B.shape))
        return A.add(B)

    def __sub__(A, B):
        if not isinstance(B, SDM):
            return NotImplemented
        elif A.shape != B.shape:
            raise DMShapeError("Matrix size mismatch: %s - %s" % (A.shape, B.shape))
        return A.sub(B)

    def __neg__(A):
        return A.neg()

    def __mul__(A, B):
        """A * B"""
        if isinstance(B, SDM):
            return A.matmul(B)
        elif B in A.domain:
            return A.mul(B)
        else:
            return NotImplemented

    def __rmul__(a, b):
        if b in a.domain:
            return a.rmul(b)
        else:
            return NotImplemented

    def matmul(A, B):
        """
        Performs matrix multiplication of two SDM matrices

        Parameters
        ==========

        A, B: SDM to multiply

        Returns
        =======

        SDM
            SDM after multiplication

        Raises
        ======

        DomainError
            If domain of A does not match
            with that of B

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B = SDM({0:{0:ZZ(2), 1:ZZ(3)}, 1:{0:ZZ(4)}}, (2, 2), ZZ)
        >>> A.matmul(B)
        {0: {0: 8}, 1: {0: 2, 1: 3}}

        """
        if A.domain != B.domain:
            raise DMDomainError
        m, n = A.shape
        n2, o = B.shape
        if n != n2:
            raise DMShapeError
        C = sdm_matmul(A, B, A.domain, m, o)
        return A.new(C, (m, o), A.domain)

    def mul(A, b):
        """
        Multiplies each element of A with a scalar b

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.mul(ZZ(3))
        {0: {1: 6}, 1: {0: 3}}

        """
        Csdm = unop_dict(A, lambda aij: aij*b)
        return A.new(Csdm, A.shape, A.domain)

    def rmul(A, b):
        Csdm = unop_dict(A, lambda aij: b*aij)
        return A.new(Csdm, A.shape, A.domain)

    def mul_elementwise(A, B):
        if A.domain != B.domain:
            raise DMDomainError
        if A.shape != B.shape:
            raise DMShapeError
        zero = A.domain.zero
        fzero = lambda e: zero
        Csdm = binop_dict(A, B, mul, fzero, fzero)
        return A.new(Csdm, A.shape, A.domain)

    def add(A, B):
        """

        Adds two :py:class:`~.SDM` matrices

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
        >>> A.add(B)
        {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

        """
        Csdm = binop_dict(A, B, add, pos, pos)
        return A.new(Csdm, A.shape, A.domain)

    def sub(A, B):
        """

        Subtracts two :py:class:`~.SDM` matrices

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
        >>> A.sub(B)
        {0: {0: -3, 1: 2}, 1: {0: 1, 1: -4}}

        """
        Csdm = binop_dict(A, B, sub, pos, neg)
        return A.new(Csdm, A.shape, A.domain)

    def neg(A):
        """

        Returns the negative of a :py:class:`~.SDM` matrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.neg()
        {0: {1: -2}, 1: {0: -1}}

        """
        Csdm = unop_dict(A, neg)
        return A.new(Csdm, A.shape, A.domain)

    def convert_to(A, K):
        """
        Converts the :py:class:`~.Domain` of a :py:class:`~.SDM` matrix to K

        Examples
        ========

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.convert_to(QQ)
        {0: {1: 2}, 1: {0: 1}}

        """
        Kold = A.domain
        if K == Kold:
            return A.copy()
        Ak = unop_dict(A, lambda e: K.convert_from(e, Kold))
        return A.new(Ak, A.shape, K)

    def nnz(A):
        """Number of non-zero elements in the :py:class:`~.SDM` matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.nnz()
        2

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz
        """
        return sum(map(len, A.values()))

    def scc(A):
        """Strongly connected components of a square matrix *A*.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0: ZZ(2)}, 1:{1:ZZ(1)}}, (2, 2), ZZ)
        >>> A.scc()
        [[0], [1]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.scc
        """
        rows, cols = A.shape
        assert rows == cols
        V = range(rows)
        Emap = {v: list(A.get(v, [])) for v in V}
        return _strongly_connected_components(V, Emap)

    def rref(A):
        """

        Returns reduced-row echelon form and list of pivots for the :py:class:`~.SDM`

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.rref()
        ({0: {0: 1, 1: 2}}, [0])

        """
        B, pivots, _ = sdm_irref(A)
        return A.new(B, A.shape, A.domain), pivots

    def rref_den(A):
        """

        Returns reduced-row echelon form (RREF) with denominator and pivots.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.rref_den()
        ({0: {0: 1, 1: 2}}, 1, [0])

        """
        K = A.domain
        A_rref_sdm, denom, pivots = sdm_rref_den(A, K)
        A_rref = A.new(A_rref_sdm, A.shape, A.domain)
        return A_rref, denom, pivots

    def inv(A):
        """

        Returns inverse of a matrix A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.inv()
        {0: {0: -2, 1: 1}, 1: {0: 3/2, 1: -1/2}}

        """
        return A.to_dfm_or_ddm().inv().to_sdm()

    def det(A):
        """
        Returns determinant of A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.det()
        -2

        """
        # It would be better to have a sparse implementation of det for use
        # with very sparse matrices. Extremely sparse matrices probably just
        # have determinant zero and we could probably detect that very quickly.
        # In the meantime, we convert to a dense matrix and use ddm_idet.
        #
        # If GROUND_TYPES=flint though then we will use Flint's implementation
        # if possible (dfm).
        return A.to_dfm_or_ddm().det()

    def lu(A):
        """

        Returns LU decomposition for a matrix A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.lu()
        ({0: {0: 1}, 1: {0: 3, 1: 1}}, {0: {0: 1, 1: 2}, 1: {1: -2}}, [])

        """
        L, U, swaps = A.to_ddm().lu()
        return A.from_ddm(L), A.from_ddm(U), swaps

    def qr(self):
        """
        QR decomposition for SDM (Sparse Domain Matrix).

        Returns:
            - Q: Orthogonal matrix as a SDM.
            - R: Upper triangular matrix as a SDM.
        """
        ddm_q, ddm_r = self.to_ddm().qr()
        Q = ddm_q.to_sdm()
        R = ddm_r.to_sdm()
        return Q, R

    def lu_solve(A, b):
        """

        Uses LU decomposition to solve Ax = b,

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> b = SDM({0:{0:QQ(1)}, 1:{0:QQ(2)}}, (2, 1), QQ)
        >>> A.lu_solve(b)
        {1: {0: 1/2}}

        """
        return A.from_ddm(A.to_ddm().lu_solve(b.to_ddm()))

    def fflu(self):
        """
        Fraction free LU decomposition of SDM.

        Uses DDM implementation.

        See Also
        ========

        sympy.polys.matrices.ddm.DDM.fflu
        """
        ddm_p, ddm_l, ddm_d, ddm_u = self.to_dfm_or_ddm().fflu()
        P = ddm_p.to_sdm()
        L = ddm_l.to_sdm()
        D = ddm_d.to_sdm()
        U = ddm_u.to_sdm()
        return P, L, D, U

    def nullspace(A):
        """
        Nullspace of a :py:class:`~.SDM` matrix A.

        The domain of the matrix must be a field.

        It is better to use the :meth:`~.DomainMatrix.nullspace` method rather
        than this method which is otherwise no longer used.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)
        >>> A.nullspace()
        ({0: {0: -2, 1: 1}}, [1])


        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The preferred way to get the nullspace of a matrix.

        """
        ncols = A.shape[1]
        one = A.domain.one
        B, pivots, nzcols = sdm_irref(A)
        K, nonpivots = sdm_nullspace_from_rref(B, one, ncols, pivots, nzcols)
        K = dict(enumerate(K))
        shape = (len(K), ncols)
        return A.new(K, shape, A.domain), nonpivots

    def nullspace_from_rref(A, pivots=None):
        """
        Returns nullspace for a :py:class:`~.SDM` matrix ``A`` in RREF.

        The domain of the matrix can be any domain.

        The matrix must already be in reduced row echelon form (RREF).

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)
        >>> A_rref, pivots = A.rref()
        >>> A_null, nonpivots = A_rref.nullspace_from_rref(pivots)
        >>> A_null
        {0: {0: -2, 1: 1}}
        >>> pivots
        [0]
        >>> nonpivots
        [1]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The higher-level function that would usually be called instead of
            calling this one directly.

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace_from_rref
            The higher-level direct equivalent of this function.

        sympy.polys.matrices.ddm.DDM.nullspace_from_rref
            The equivalent function for dense :py:class:`~.DDM` matrices.

        """
        m, n = A.shape
        K = A.domain

        if pivots is None:
            pivots = sorted(map(min, A.values()))

        if not pivots:
            return A.eye((n, n), K), list(range(n))
        elif len(pivots) == n:
            return A.zeros((0, n), K), []

        # In fraction-free RREF the nonzero entry inserted for the pivots is
        # not necessarily 1.
        pivot_val = A[0][pivots[0]]
        assert not K.is_zero(pivot_val)

        pivots_set = set(pivots)

        # Loop once over all nonzero entries making a map from column indices
        # to the nonzero entries in that column along with the row index of the
        # nonzero entry. This is basically the transpose of the matrix.
        nonzero_cols = defaultdict(list)
        for i, Ai in A.items():
            for j, Aij in Ai.items():
                nonzero_cols[j].append((i, Aij))

        # Usually in SDM we want to avoid looping over the dimensions of the
        # matrix because it is optimised to support extremely sparse matrices.
        # Here in nullspace though every zero column becomes a nonzero column
        # so we need to loop once over the columns at least (range(n)) rather
        # than just the nonzero entries of the matrix. We can still avoid
        # an inner loop over the rows though by using the nonzero_cols map.
        basis = []
        nonpivots = []
        for j in range(n):
            if j in pivots_set:
                continue
            nonpivots.append(j)

            vec = {j: pivot_val}
            for ip, Aij in nonzero_cols[j]:
                vec[pivots[ip]] = -Aij

            basis.append(vec)

        sdm = dict(enumerate(basis))
        A_null = A.new(sdm, (len(basis), n), K)

        return (A_null, nonpivots)

    def particular(A):
        ncols = A.shape[1]
        B, pivots, nzcols = sdm_irref(A)
        P = sdm_particular_from_rref(B, ncols, pivots)
        rep = {0:P} if P else {}
        return A.new(rep, (1, ncols-1), A.domain)

    def hstack(A, *B):
        """Horizontally stacks :py:class:`~.SDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM

        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)
        >>> A.hstack(B)
        {0: {0: 1, 1: 2, 2: 5, 3: 6}, 1: {0: 3, 1: 4, 2: 7, 3: 8}}

        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)
        >>> A.hstack(B, C)
        {0: {0: 1, 1: 2, 2: 5, 3: 6, 4: 9, 5: 10}, 1: {0: 3, 1: 4, 2: 7, 3: 8, 4: 11, 5: 12}}
        """
        Anew = dict(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkrows == rows
            assert Bk.domain == domain

            for i, Bki in Bk.items():
                Ai = Anew.get(i, None)
                if Ai is None:
                    Anew[i] = Ai = {}
                for j, Bkij in Bki.items():
                    Ai[j + cols] = Bkij
            cols += Bkcols

        return A.new(Anew, (rows, cols), A.domain)

    def vstack(A, *B):
        """Vertically stacks :py:class:`~.SDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM

        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)
        >>> A.vstack(B)
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}}

        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)
        >>> A.vstack(B, C)
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}, 4: {0: 9, 1: 10}, 5: {0: 11, 1: 12}}
        """
        Anew = dict(A.copy())
        rows, cols = A.shape
        domain = A.domain

        for Bk in B:
            Bkrows, Bkcols = Bk.shape
            assert Bkcols == cols
            assert Bk.domain == domain

            for i, Bki in Bk.items():
                Anew[i + rows] = Bki
            rows += Bkrows

        return A.new(Anew, (rows, cols), A.domain)

    def applyfunc(self, func, domain):
        sdm = {i: {j: func(e) for j, e in row.items()} for i, row in self.items()}
        return self.new(sdm, self.shape, domain)

    def charpoly(A):
        """
        Returns the coefficients of the characteristic polynomial
        of the :py:class:`~.SDM` matrix. These elements will be domain elements.
        The domain of the elements will be same as domain of the :py:class:`~.SDM`.

        Examples
        ========

        >>> from sympy import QQ, Symbol
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy.polys import Poly
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.charpoly()
        [1, -5, -2]

        We can create a polynomial using the
        coefficients using :py:class:`~.Poly`

        >>> x = Symbol('x')
        >>> p = Poly(A.charpoly(), x, domain=A.domain)
        >>> p
        Poly(x**2 - 5*x - 2, x, domain='QQ')

        """
        K = A.domain
        n, _ = A.shape
        pdict = sdm_berk(A, n, K)
        plist = [K.zero] * (n + 1)
        for i, pi in pdict.items():
            plist[i] = pi
        return plist

    def is_zero_matrix(self):
        """
        Says whether this matrix has all zero entries.
        """
        return not self

    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        return all(i <= j for i, row in self.items() for j in row)

    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        return all(i >= j for i, row in self.items() for j in row)

    def is_diagonal(self):
        """
        Says whether this matrix is diagonal. True can be returned
        even if the matrix is not square.
        """
        return all(i == j for i, row in self.items() for j in row)

    def diagonal(self):
        """
        Returns the diagonal of the matrix as a list.
        """
        m, n = self.shape
        zero = self.domain.zero
        return [row.get(i, zero) for i, row in self.items() if i < n]

    def lll(A, delta=QQ(3, 4)):
        """
        Returns the LLL-reduced basis for the :py:class:`~.SDM` matrix.
        """
        return A.to_dfm_or_ddm().lll(delta=delta).to_sdm()

    def lll_transform(A, delta=QQ(3, 4)):
        """
        Returns the LLL-reduced basis and transformation matrix.
        """
        reduced, transform = A.to_dfm_or_ddm().lll_transform(delta=delta)
        return reduced.to_sdm(), transform.to_sdm()


def binop_dict(A, B, fab, fa, fb):
    Anz, Bnz = set(A), set(B)
    C = {}

    for i in Anz & Bnz:
        Ai, Bi = A[i], B[i]
        Ci = {}
        Anzi, Bnzi = set(Ai), set(Bi)
        for j in Anzi & Bnzi:
            Cij = fab(Ai[j], Bi[j])
            if Cij:
                Ci[j] = Cij
        for j in Anzi - Bnzi:
            Cij = fa(Ai[j])
            if Cij:
                Ci[j] = Cij
        for j in Bnzi - Anzi:
            Cij = fb(Bi[j])
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    for i in Anz - Bnz:
        Ai = A[i]
        Ci = {}
        for j, Aij in Ai.items():
            Cij = fa(Aij)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    for i in Bnz - Anz:
        Bi = B[i]
        Ci = {}
        for j, Bij in Bi.items():
            Cij = fb(Bij)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    return C


def unop_dict(A, f):
    B = {}
    for i, Ai in A.items():
        Bi = {}
        for j, Aij in Ai.items():
            Bij = f(Aij)
            if Bij:
                Bi[j] = Bij
        if Bi:
            B[i] = Bi
    return B


def sdm_transpose(M):
    MT = {}
    for i, Mi in M.items():
        for j, Mij in Mi.items():
            try:
                MT[j][i] = Mij
            except KeyError:
                MT[j] = {i: Mij}
    return MT


def sdm_dotvec(A, B, K):
    return K.sum(A[j] * B[j] for j in A.keys() & B.keys())


def sdm_matvecmul(A, B, K):
    C = {}
    for i, Ai in A.items():
        Ci = sdm_dotvec(Ai, B, K)
        if Ci:
            C[i] = Ci
    return C


def sdm_matmul(A, B, K, m, o):
    #
    # Should be fast if A and B are very sparse.
    # Consider e.g. A = B = eye(1000).
    #
    # The idea here is that we compute C = A*B in terms of the rows of C and
    # B since the dict of dicts representation naturally stores the matrix as
    # rows. The ith row of C (Ci) is equal to the sum of Aik * Bk where Bk is
    # the kth row of B. The algorithm below loops over each nonzero element
    # Aik of A and if the corresponding row Bj is nonzero then we do
    #    Ci += Aik * Bk.
    # To make this more efficient we don't need to loop over all elements Aik.
    # Instead for each row Ai we compute the intersection of the nonzero
    # columns in Ai with the nonzero rows in B. That gives the k such that
    # Aik and Bk are both nonzero. In Python the intersection of two sets
    # of int can be computed very efficiently.
    #
    if K.is_EXRAW:
        return sdm_matmul_exraw(A, B, K, m, o)

    C = {}
    B_knz = set(B)
    for i, Ai in A.items():
        Ci = {}
        Ai_knz = set(Ai)
        for k in Ai_knz & B_knz:
            Aik = Ai[k]
            for j, Bkj in B[k].items():
                Cij = Ci.get(j, None)
                if Cij is not None:
                    Cij = Cij + Aik * Bkj
                    if Cij:
                        Ci[j] = Cij
                    else:
                        Ci.pop(j)
                else:
                    Cij = Aik * Bkj
                    if Cij:
                        Ci[j] = Cij
        if Ci:
            C[i] = Ci
    return C


def sdm_matmul_exraw(A, B, K, m, o):
    #
    # Like sdm_matmul above except that:
    #
    # - Handles cases like 0*oo -> nan (sdm_matmul skips multiplication by zero)
    # - Uses K.sum (Add(*items)) for efficient addition of Expr
    #
    zero = K.zero
    C = {}
    B_knz = set(B)
    for i, Ai in A.items():
        Ci_list = defaultdict(list)
        Ai_knz = set(Ai)

        # Nonzero row/column pair
        for k in Ai_knz & B_knz:
            Aik = Ai[k]
            if zero * Aik == zero:
                # This is the main inner loop:
                for j, Bkj in B[k].items():
                    Ci_list[j].append(Aik * Bkj)
            else:
                for j in range(o):
                    Ci_list[j].append(Aik * B[k].get(j, zero))

        # Zero row in B, check for infinities in A
        for k in Ai_knz - B_knz:
            zAik = zero * Ai[k]
            if zAik != zero:
                for j in range(o):
                    Ci_list[j].append(zAik)

        # Add terms using K.sum (Add(*terms)) for efficiency
        Ci = {}
        for j, Cij_list in Ci_list.items():
            Cij = K.sum(Cij_list)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci

    # Find all infinities in B
    for k, Bk in B.items():
        for j, Bkj in Bk.items():
            if zero * Bkj != zero:
                for i in range(m):
                    Aik = A.get(i, {}).get(k, zero)
                    # If Aik is not zero then this was handled above
                    if Aik == zero:
                        Ci = C.get(i, {})
                        Cij = Ci.get(j, zero) + Aik * Bkj
                        if Cij != zero:
                            Ci[j] = Cij
                            C[i] = Ci
                        else:
                            Ci.pop(j, None)
                            if Ci:
                                C[i] = Ci
                            else:
                                C.pop(i, None)

    return C


def sdm_irref(A):
    """RREF and pivots of a sparse matrix *A*.

    Compute the reduced row echelon form (RREF) of the matrix *A* and return a
    list of the pivot columns. This routine does not work in place and leaves
    the original matrix *A* unmodified.

    The domain of the matrix must be a field.

    Examples
    ========

    This routine works with a dict of dicts sparse representation of a matrix:

    >>> from sympy import QQ
    >>> from sympy.polys.matrices.sdm import sdm_irref
    >>> A = {0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}
    >>> Arref, pivots, _ = sdm_irref(A)
    >>> Arref
    {0: {0: 1}, 1: {1: 1}}
    >>> pivots
    [0, 1]

    The analogous calculation with :py:class:`~.MutableDenseMatrix` would be

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> Mrref, pivots = M.rref()
    >>> Mrref
    Matrix([
    [1, 0],
    [0, 1]])
    >>> pivots
    (0, 1)

    Notes
    =====

    The cost of this algorithm is determined purely by the nonzero elements of
    the matrix. No part of the cost of any step in this algorithm depends on
    the number of rows or columns in the matrix. No step depends even on the
    number of nonzero rows apart from the primary loop over those rows. The
    implementation is much faster than ddm_rref for sparse matrices. In fact
    at the time of writing it is also (slightly) faster than the dense
    implementation even if the input is a fully dense matrix so it seems to be
    faster in all cases.

    The elements of the matrix should support exact division with ``/``. For
    example elements of any domain that is a field (e.g. ``QQ``) should be
    fine. No attempt is made to handle inexact arithmetic.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        The higher-level function that would normally be used to call this
        routine.
    sympy.polys.matrices.dense.ddm_irref
        The dense equivalent of this routine.
    sdm_rref_den
        Fraction-free version of this routine.
    """
    #
    # Any zeros in the matrix are not stored at all so an element is zero if
    # its row dict has no index at that key. A row is entirely zero if its
    # row index is not in the outer dict. Since rref reorders the rows and
    # removes zero rows we can completely discard the row indices. The first
    # step then copies the row dicts into a list sorted by the index of the
    # first nonzero column in each row.
    #
    # The algorithm then processes each row Ai one at a time. Previously seen
    # rows are used to cancel their pivot columns from Ai. Then a pivot from
    # Ai is chosen and is cancelled from all previously seen rows. At this
    # point Ai joins the previously seen rows. Once all rows are seen all
    # elimination has occurred and the rows are sorted by pivot column index.
    #
    # The previously seen rows are stored in two separate groups. The reduced
    # group consists of all rows that have been reduced to a single nonzero
    # element (the pivot). There is no need to attempt any further reduction
    # with these. Rows that still have other nonzeros need to be considered
    # when Ai is cancelled from the previously seen rows.
    #
    # A dict nonzerocolumns is used to map from a column index to a set of
    # previously seen rows that still have a nonzero element in that column.
    # This means that we can cancel the pivot from Ai into the previously seen
    # rows without needing to loop over each row that might have a zero in
    # that column.
    #

    # Row dicts sorted by index of first nonzero column
    # (Maybe sorting is not needed/useful.)
    Arows = sorted((Ai.copy() for Ai in A.values()), key=min)

    # Each processed row has an associated pivot column.
    # pivot_row_map maps from the pivot column index to the row dict.
    # This means that we can represent a set of rows purely as a set of their
    # pivot indices.
    pivot_row_map = {}

    # Set of pivot indices for rows that are fully reduced to a single nonzero.
    reduced_pivots = set()

    # Set of pivot indices for rows not fully reduced
    nonreduced_pivots = set()

    # Map from column index to a set of pivot indices representing the rows
    # that have a nonzero at that column.
    nonzero_columns = defaultdict(set)

    while Arows:
        # Select pivot element and row
        Ai = Arows.pop()

        # Nonzero columns from fully reduced pivot rows can be removed
        Ai = {j: Aij for j, Aij in Ai.items() if j not in reduced_pivots}

        # Others require full row cancellation
        for j in nonreduced_pivots & set(Ai):
            Aj = pivot_row_map[j]
            Aij = Ai[j]
            Ainz = set(Ai)
            Ajnz = set(Aj)
            for k in Ajnz - Ainz:
                Ai[k] = - Aij * Aj[k]
            Ai.pop(j)
            Ainz.remove(j)
            for k in Ajnz & Ainz:
                Aik = Ai[k] - Aij * Aj[k]
                if Aik:
                    Ai[k] = Aik
                else:
                    Ai.pop(k)

        # We have now cancelled previously seen pivots from Ai.
        # If it is zero then discard it.
        if not Ai:
            continue

        # Choose a pivot from Ai:
        j = min(Ai)
        Aij = Ai[j]
        pivot_row_map[j] = Ai
        Ainz = set(Ai)

        # Normalise the pivot row to make the pivot 1.
        #
        # This approach is slow for some domains. Cross cancellation might be
        # better for e.g. QQ(x) with division delayed to the final steps.
        Aijinv = Aij**-1
        for l in Ai:
            Ai[l] *= Aijinv

        # Use Aij to cancel column j from all previously seen rows
        for k in nonzero_columns.pop(j, ()):
            Ak = pivot_row_map[k]
            Akj = Ak[j]
            Aknz = set(Ak)
            for l in Ainz - Aknz:
                Ak[l] = - Akj * Ai[l]
                nonzero_columns[l].add(k)
            Ak.pop(j)
            Aknz.remove(j)
            for l in Ainz & Aknz:
                Akl = Ak[l] - Akj * Ai[l]
                if Akl:
                    Ak[l] = Akl
                else:
                    # Drop nonzero elements
                    Ak.pop(l)
                    if l != j:
                        nonzero_columns[l].remove(k)
            if len(Ak) == 1:
                reduced_pivots.add(k)
                nonreduced_pivots.remove(k)

        if len(Ai) == 1:
            reduced_pivots.add(j)
        else:
            nonreduced_pivots.add(j)
            for l in Ai:
                if l != j:
                    nonzero_columns[l].add(j)

    # All done!
    pivots = sorted(reduced_pivots | nonreduced_pivots)
    pivot2row = {p: n for n, p in enumerate(pivots)}
    nonzero_columns = {c: {pivot2row[p] for p in s} for c, s in nonzero_columns.items()}
    rows = [pivot_row_map[i] for i in pivots]
    rref = dict(enumerate(rows))
    return rref, pivots, nonzero_columns


def sdm_rref_den(A, K):
    """
    Return the reduced row echelon form (RREF) of A with denominator.

    The RREF is computed using fraction-free Gauss-Jordan elimination.

    Explanation
    ===========

    The algorithm used is the fraction-free version of Gauss-Jordan elimination
    described as FFGJ in [1]_. Here it is modified to handle zero or missing
    pivots and to avoid redundant arithmetic. This implementation is also
    optimized for sparse matrices.

    The domain $K$ must support exact division (``K.exquo``) but does not need
    to be a field. This method is suitable for most exact rings and fields like
    :ref:`ZZ`, :ref:`QQ` and :ref:`QQ(a)`. In the case of :ref:`QQ` or
    :ref:`K(x)` it might be more efficient to clear denominators and use
    :ref:`ZZ` or :ref:`K[x]` instead.

    For inexact domains like :ref:`RR` and :ref:`CC` use ``ddm_irref`` instead.

    Examples
    ========

    >>> from sympy.polys.matrices.sdm import sdm_rref_den
    >>> from sympy.polys.domains import ZZ
    >>> A = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}
    >>> A_rref, den, pivots = sdm_rref_den(A, ZZ)
    >>> A_rref
    {0: {0: -2}, 1: {1: -2}}
    >>> den
    -2
    >>> pivots
    [0, 1]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        Higher-level interface to ``sdm_rref_den`` that would usually be used
        instead of calling this function directly.
    sympy.polys.matrices.sdm.sdm_rref_den
        The ``SDM`` method that uses this function.
    sdm_irref
        Computes RREF using field division.
    ddm_irref_den
        The dense version of this algorithm.

    References
    ==========

    .. [1] Fraction-free algorithms for linear and polynomial equations.
        George C. Nakos , Peter R. Turner , Robert M. Williams.
        https://dl.acm.org/doi/10.1145/271130.271133
    """
    #
    # We represent each row of the matrix as a dict mapping column indices to
    # nonzero elements. We will build the RREF matrix starting from an empty
    # matrix and appending one row at a time. At each step we will have the
    # RREF of the rows we have processed so far.
    #
    # Our representation of the RREF divides it into three parts:
    #
    # 1. Fully reduced rows having only a single nonzero element (the pivot).
    # 2. Partially reduced rows having nonzeros after the pivot.
    # 3. The current denominator and divisor.
    #
    # For example if the incremental RREF might be:
    #
    #   [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #   [0, 0, 2, 0, 0, 0, 7, 0, 0, 0]
    #   [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    #   [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    #   [0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
    #
    # Here the second row is partially reduced and the other rows are fully
    # reduced. The denominator would be 2 in this case. We distinguish the
    # fully reduced rows because we can handle them more efficiently when
    # adding a new row.
    #
    # When adding a new row we need to multiply it by the current denominator.
    # Then we reduce the new row by cross cancellation with the previous rows.
    # Then if it is not reduced to zero we take its leading entry as the new
    # pivot, cross cancel the new row from the previous rows and update the
    # denominator. In the fraction-free version this last step requires
    # multiplying and dividing the whole matrix by the new pivot and the
    # current divisor. The advantage of building the RREF one row at a time is
    # that in the sparse case we only need to work with the relatively sparse
    # upper rows of the matrix. The simple version of FFGJ in [1] would
    # multiply and divide all the dense lower rows at each step.

    # Handle the trivial cases.
    if not A:
        return ({}, K.one, [])
    elif len(A) == 1:
        Ai, = A.values()
        j = min(Ai)
        Aij = Ai[j]
        return ({0: Ai.copy()}, Aij, [j])

    # For inexact domains like RR[x] we use quo and discard the remainder.
    # Maybe it would be better for K.exquo to do this automatically.
    if K.is_Exact:
        exquo = K.exquo
    else:
        exquo = K.quo

    # Make sure we have the rows in order to make this deterministic from the
    # outset.
    _, rows_in_order = zip(*sorted(A.items()))

    col_to_row_reduced = {}
    col_to_row_unreduced = {}
    reduced = col_to_row_reduced.keys()
    unreduced = col_to_row_unreduced.keys()

    # Our representation of the RREF so far.
    A_rref_rows = []
    denom = None
    divisor = None

    # The rows that remain to be added to the RREF. These are sorted by the
    # column index of their leading entry. Note that sorted() is stable so the
    # previous sort by unique row index is still needed to make this
    # deterministic (there may be multiple rows with the same leading column).
    A_rows = sorted(rows_in_order, key=min)

    for Ai in A_rows:

        # All fully reduced columns can be immediately discarded.
        Ai = {j: Aij for j, Aij in Ai.items() if j not in reduced}

        # We need to multiply the new row by the current denominator to bring
        # it into the same scale as the previous rows and then cross-cancel to
        # reduce it wrt the previous unreduced rows. All pivots in the previous
        # rows are equal to denom so the coefficients we need to make a linear
        # combination of the previous rows to cancel into the new row are just
        # the ones that are already in the new row *before* we multiply by
        # denom. We compute that linear combination first and then multiply the
        # new row by denom before subtraction.
        Ai_cancel = {}

        for j in unreduced & Ai.keys():
            # Remove the pivot column from the new row since it would become
            # zero anyway.
            Aij = Ai.pop(j)

            Aj = A_rref_rows[col_to_row_unreduced[j]]

            for k, Ajk in Aj.items():
                Aik_cancel = Ai_cancel.get(k)
                if Aik_cancel is None:
                    Ai_cancel[k] = Aij * Ajk
                else:
                    Aik_cancel = Aik_cancel + Aij * Ajk
                    if Aik_cancel:
                        Ai_cancel[k] = Aik_cancel
                    else:
                        Ai_cancel.pop(k)

        # Multiply the new row by the current denominator and subtract.
        Ai_nz = set(Ai)
        Ai_cancel_nz = set(Ai_cancel)

        d = denom or K.one

        for k in Ai_cancel_nz - Ai_nz:
            Ai[k] = -Ai_cancel[k]

        for k in Ai_nz - Ai_cancel_nz:
            Ai[k] = Ai[k] * d

        for k in Ai_cancel_nz & Ai_nz:
            Aik = Ai[k] * d - Ai_cancel[k]
            if Aik:
                Ai[k] = Aik
            else:
                Ai.pop(k)

        # Now Ai has the same scale as the other rows and is reduced wrt the
        # unreduced rows.

        # If the row is reduced to zero then discard it.
        if not Ai:
            continue

        # Choose a pivot for this row.
        j = min(Ai)
        Aij = Ai.pop(j)

        # Cross cancel the unreduced rows by the new row.
        #     a[k][l] = (a[i][j]*a[k][l] - a[k][j]*a[i][l]) / divisor
        for pk, k in list(col_to_row_unreduced.items()):

            Ak = A_rref_rows[k]

            if j not in Ak:
                # This row is already reduced wrt the new row but we need to
                # bring it to the same scale as the new denominator. This step
                # is not needed in sdm_irref.
                for l, Akl in Ak.items():
                    Akl = Akl * Aij
                    if divisor is not None:
                        Akl = exquo(Akl, divisor)
                    Ak[l] = Akl
                continue

            Akj = Ak.pop(j)
            Ai_nz = set(Ai)
            Ak_nz = set(Ak)

            for l in Ai_nz - Ak_nz:
                Ak[l] = - Akj * Ai[l]
                if divisor is not None:
                    Ak[l] = exquo(Ak[l], divisor)

            # This loop also not needed in sdm_irref.
            for l in Ak_nz - Ai_nz:
                Ak[l] = Aij * Ak[l]
                if divisor is not None:
                    Ak[l] = exquo(Ak[l], divisor)

            for l in Ai_nz & Ak_nz:
                Akl = Aij * Ak[l] - Akj * Ai[l]
                if Akl:
                    if divisor is not None:
                        Akl = exquo(Akl, divisor)
                    Ak[l] = Akl
                else:
                    Ak.pop(l)

            if not Ak:
                col_to_row_unreduced.pop(pk)
                col_to_row_reduced[pk] = k

        i = len(A_rref_rows)
        A_rref_rows.append(Ai)
        if Ai:
            col_to_row_unreduced[j] = i
        else:
            col_to_row_reduced[j] = i

        # Update the denominator.
        if not K.is_one(Aij):
            if denom is None:
                denom = Aij
            else:
                denom *= Aij

        if divisor is not None:
            denom = exquo(denom, divisor)

        # Update the divisor.
        divisor = denom

    if denom is None:
        denom = K.one

    # Sort the rows by their leading column index.
    col_to_row = {**col_to_row_reduced, **col_to_row_unreduced}
    row_to_col = {i: j for j, i in col_to_row.items()}
    A_rref_rows_col = [(row_to_col[i], Ai) for i, Ai in enumerate(A_rref_rows)]
    pivots, A_rref = zip(*sorted(A_rref_rows_col))
    pivots = list(pivots)

    # Insert the pivot values
    for i, Ai in enumerate(A_rref):
        Ai[pivots[i]] = denom

    A_rref_sdm = dict(enumerate(A_rref))

    return A_rref_sdm, denom, pivots


def sdm_nullspace_from_rref(A, one, ncols, pivots, nonzero_cols):
    """Get nullspace from A which is in RREF"""
    nonpivots = sorted(set(range(ncols)) - set(pivots))

    K = []
    for j in nonpivots:
        Kj = {j:one}
        for i in nonzero_cols.get(j, ()):
            Kj[pivots[i]] = -A[i][j]
        K.append(Kj)

    return K, nonpivots


def sdm_particular_from_rref(A, ncols, pivots):
    """Get a particular solution from A which is in RREF"""
    P = {}
    for i, j in enumerate(pivots):
        Ain = A[i].get(ncols-1, None)
        if Ain is not None:
            P[j] = Ain / A[i][j]
    return P


def sdm_berk(M, n, K):
    """
    Berkowitz algorithm for computing the characteristic polynomial.

    Explanation
    ===========

    The Berkowitz algorithm is a division-free algorithm for computing the
    characteristic polynomial of a matrix over any commutative ring using only
    arithmetic in the coefficient ring. This implementation is for sparse
    matrices represented in a dict-of-dicts format (like :class:`SDM`).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices.sdm import sdm_berk
    >>> from sympy.polys.domains import ZZ
    >>> M = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    >>> sdm_berk(M, 2, ZZ)
    {0: 1, 1: -5, 2: -2}
    >>> Matrix([[1, 2], [3, 4]]).charpoly()
    PurePoly(lambda**2 - 5*lambda - 2, lambda, domain='ZZ')

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
        The high-level interface to this function.
    sympy.polys.matrices.dense.ddm_berk
        The dense version of this function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Samuelson%E2%80%93Berkowitz_algorithm
    """
    zero = K.zero
    one = K.one

    if n == 0:
        return {0: one}
    elif n == 1:
        pdict = {0: one}
        if M00 := M.get(0, {}).get(0, zero):
            pdict[1] = -M00

    # M = [[a, R],
    #      [C, A]]
    a, R, C, A = K.zero, {}, {}, defaultdict(dict)
    for i, Mi in M.items():
        for j, Mij in Mi.items():
            if i and j:
                A[i-1][j-1] = Mij
            elif i:
                C[i-1] = Mij
            elif j:
                R[j-1] = Mij
            else:
                a = Mij

    # T = [       1,      0,    0,  0, 0, ... ]
    #     [      -a,      1,    0,  0, 0, ... ]
    #     [    -R*C,     -a,    1,  0, 0, ... ]
    #     [  -R*A*C,   -R*C,   -a,  1, 0, ... ]
    #     [-R*A^2*C, -R*A*C, -R*C, -a, 1, ... ]
    #     [ ...                               ]
    # T is (n+1) x n
    #
    # In the sparse case we might have A^m*C = 0 for some m making T banded
    # rather than triangular so we just compute the nonzero entries of the
    # first column rather than constructing the matrix explicitly.

    AnC = C
    RC = sdm_dotvec(R, C, K)

    Tvals = [one, -a, -RC]
    for i in range(3, n+1):
        AnC = sdm_matvecmul(A, AnC, K)
        if not AnC:
            break
        RAnC = sdm_dotvec(R, AnC, K)
        Tvals.append(-RAnC)

    # Strip trailing zeros
    while Tvals and not Tvals[-1]:
        Tvals.pop()

    q = sdm_berk(A, n-1, K)

    # This would be the explicit multiplication T*q but we can do better:
    #
    #  T = {}
    #  for i in range(n+1):
    #      Ti = {}
    #      for j in range(max(0, i-len(Tvals)+1), min(i+1, n)):
    #          Ti[j] = Tvals[i-j]
    #      T[i] = Ti
    #  Tq = sdm_matvecmul(T, q, K)
    #
    # In the sparse case q might be mostly zero. We know that T[i,j] is nonzero
    # for i <= j < i + len(Tvals) so if q does not have a nonzero entry in that
    # range then Tq[j] must be zero. We exploit this potential banded
    # structure and the potential sparsity of q to compute Tq more efficiently.

    Tvals = Tvals[::-1]

    Tq = {}

    for i in range(min(q), min(max(q)+len(Tvals), n+1)):
        Ti = dict(enumerate(Tvals, i-len(Tvals)+1))
        if Tqi := sdm_dotvec(Ti, q, K):
            Tq[i] = Tqi

    return Tq

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\meijerint.py ===
"""
Integrate functions by rewriting them as Meijer G-functions.

There are three user-visible functions that can be used by other parts of the
sympy library to solve various integration problems:

- meijerint_indefinite
- meijerint_definite
- meijerint_inversion

They can be used to compute, respectively, indefinite integrals, definite
integrals over intervals of the real line, and inverse laplace-type integrals
(from c-I*oo to c+I*oo). See the respective docstrings for details.

The main references for this are:

[L] Luke, Y. L. (1969), The Special Functions and Their Approximations,
    Volume 1

[R] Kelly B. Roach.  Meijer G Function Representations.
    In: Proceedings of the 1997 International Symposium on Symbolic and
    Algebraic Computation, pages 205-211, New York, 1997. ACM.

[P] A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev (1990).
    Integrals and Series: More Special Functions, Vol. 3,.
    Gordon and Breach Science Publisher
"""

from __future__ import annotations
import itertools

from sympy import SYMPY_DEBUG
from sympy.core import S, Expr
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand, expand_mul, expand_power_base,
                                 expand_trig, Function)
from sympy.core.mul import Mul
from sympy.core.intfunc import ilcm
from sympy.core.numbers import Rational, pi
from sympy.core.relational import Eq, Ne, _canonical_coeff
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, symbols, Wild, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (re, im, arg, Abs, sign,
        unpolarify, polarify, polar_lift, principal_branch, unbranched_argument,
        periodic_argument)
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.hyperbolic import (cosh, sinh,
        _rewrite_hyperbolics_as_exp, HyperbolicFunction)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc,
        TrigonometricFunction)
from sympy.functions.special.bessel import besselj, bessely, besseli, besselk
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e
from sympy.functions.special.error_functions import (erf, erfc, erfi, Ei,
        expint, Si, Ci, Shi, Chi, fresnels, fresnelc)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.functions.special.singularity_functions import SingularityFunction
from .integrals import Integral
from sympy.logic.boolalg import And, Or, BooleanAtom, Not, BooleanFunction
from sympy.polys import cancel, factor
from sympy.utilities.iterables import multiset_partitions
from sympy.utilities.misc import debug as _debug
from sympy.utilities.misc import debugf as _debugf

# keep this at top for easy reference
z = Dummy('z')


def _has(res, *f):
    # return True if res has f; in the case of Piecewise
    # only return True if *all* pieces have f
    res = piecewise_fold(res)
    if getattr(res, 'is_Piecewise', False):
        return all(_has(i, *f) for i in res.args)
    return res.has(*f)


def _create_lookup_table(table):
    """ Add formulae for the function -> meijerg lookup table. """
    def wild(n):
        return Wild(n, exclude=[z])
    p, q, a, b, c = list(map(wild, 'pqabc'))
    n = Wild('n', properties=[lambda x: x.is_Integer and x > 0])
    t = p*z**q

    def add(formula, an, ap, bm, bq, arg=t, fac=S.One, cond=True, hint=True):
        table.setdefault(_mytype(formula, z), []).append((formula,
                                     [(fac, meijerg(an, ap, bm, bq, arg))], cond, hint))

    def addi(formula, inst, cond, hint=True):
        table.setdefault(
            _mytype(formula, z), []).append((formula, inst, cond, hint))

    def constant(a):
        return [(a, meijerg([1], [], [], [0], z)),
                (a, meijerg([], [1], [0], [], z))]
    table[()] = [(a, constant(a), True, True)]

    # [P], Section 8.
    class IsNonPositiveInteger(Function):

        @classmethod
        def eval(cls, arg):
            arg = unpolarify(arg)
            if arg.is_Integer is True:
                return arg <= 0

    # Section 8.4.2
    # TODO this needs more polar_lift (c/f entry for exp)
    add(Heaviside(t - b)*(t - b)**(a - 1), [a], [], [], [0], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside(b - t)*(b - t)**(a - 1), [], [a], [0], [], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside(z - (b/p)**(1/q))*(t - b)**(a - 1), [a], [], [], [0], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside((b/p)**(1/q) - z)*(b - t)**(a - 1), [], [a], [0], [], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add((b + t)**(-a), [1 - a], [], [0], [], t/b, b**(-a)/gamma(a),
        hint=Not(IsNonPositiveInteger(a)))
    add(Abs(b - t)**(-a), [1 - a], [(1 - a)/2], [0], [(1 - a)/2], t/b,
        2*sin(pi*a/2)*gamma(1 - a)*Abs(b)**(-a), re(a) < 1)
    add((t**a - b**a)/(t - b), [0, a], [], [0, a], [], t/b,
        b**(a - 1)*sin(a*pi)/pi)

    # 12
    def A1(r, sign, nu):
        return pi**Rational(-1, 2)*(-sign*nu/2)**(1 - 2*r)

    def tmpadd(r, sgn):
        # XXX the a**2 is bad for matching
        add((sqrt(a**2 + t) + sgn*a)**b/(a**2 + t)**r,
            [(1 + b)/2, 1 - 2*r + b/2], [],
            [(b - sgn*b)/2], [(b + sgn*b)/2], t/a**2,
            a**(b - 2*r)*A1(r, sgn, b))
    tmpadd(0, 1)
    tmpadd(0, -1)
    tmpadd(S.Half, 1)
    tmpadd(S.Half, -1)

    # 13
    def tmpadd(r, sgn):
        add((sqrt(a + p*z**q) + sgn*sqrt(p)*z**(q/2))**b/(a + p*z**q)**r,
            [1 - r + sgn*b/2], [1 - r - sgn*b/2], [0, S.Half], [],
            p*z**q/a, a**(b/2 - r)*A1(r, sgn, b))
    tmpadd(0, 1)
    tmpadd(0, -1)
    tmpadd(S.Half, 1)
    tmpadd(S.Half, -1)
    # (those after look obscure)

    # Section 8.4.3
    add(exp(polar_lift(-1)*t), [], [], [0], [])

    # TODO can do sin^n, sinh^n by expansion ... where?
    # 8.4.4 (hyperbolic functions)
    add(sinh(t), [], [1], [S.Half], [1, 0], t**2/4, pi**Rational(3, 2))
    add(cosh(t), [], [S.Half], [0], [S.Half, S.Half], t**2/4, pi**Rational(3, 2))

    # Section 8.4.5
    # TODO can do t + a. but can also do by expansion... (XXX not really)
    add(sin(t), [], [], [S.Half], [0], t**2/4, sqrt(pi))
    add(cos(t), [], [], [0], [S.Half], t**2/4, sqrt(pi))

    # Section 8.4.6 (sinc function)
    add(sinc(t), [], [], [0], [Rational(-1, 2)], t**2/4, sqrt(pi)/2)

    # Section 8.5.5
    def make_log1(subs):
        N = subs[n]
        return [(S.NegativeOne**N*factorial(N),
                 meijerg([], [1]*(N + 1), [0]*(N + 1), [], t))]

    def make_log2(subs):
        N = subs[n]
        return [(factorial(N),
                 meijerg([1]*(N + 1), [], [], [0]*(N + 1), t))]
    # TODO these only hold for positive p, and can be made more general
    #      but who uses log(x)*Heaviside(a-x) anyway ...
    # TODO also it would be nice to derive them recursively ...
    addi(log(t)**n*Heaviside(1 - t), make_log1, True)
    addi(log(t)**n*Heaviside(t - 1), make_log2, True)

    def make_log3(subs):
        return make_log1(subs) + make_log2(subs)
    addi(log(t)**n, make_log3, True)
    addi(log(t + a),
         constant(log(a)) + [(S.One, meijerg([1, 1], [], [1], [0], t/a))],
         True)
    addi(log(Abs(t - a)), constant(log(Abs(a))) +
         [(pi, meijerg([1, 1], [S.Half], [1], [0, S.Half], t/a))],
         True)
    # TODO log(x)/(x+a) and log(x)/(x-1) can also be done. should they
    #      be derivable?
    # TODO further formulae in this section seem obscure

    # Sections 8.4.9-10
    # TODO

    # Section 8.4.11
    addi(Ei(t),
         constant(-S.ImaginaryUnit*pi) + [(S.NegativeOne, meijerg([], [1], [0, 0], [],
                  t*polar_lift(-1)))],
         True)

    # Section 8.4.12
    add(Si(t), [1], [], [S.Half], [0, 0], t**2/4, sqrt(pi)/2)
    add(Ci(t), [], [1], [0, 0], [S.Half], t**2/4, -sqrt(pi)/2)

    # Section 8.4.13
    add(Shi(t), [S.Half], [], [0], [Rational(-1, 2), Rational(-1, 2)], polar_lift(-1)*t**2/4,
        t*sqrt(pi)/4)
    add(Chi(t), [], [S.Half, 1], [0, 0], [S.Half, S.Half], t**2/4, -
        pi**S('3/2')/2)

    # generalized exponential integral
    add(expint(a, t), [], [a], [a - 1, 0], [], t)

    # Section 8.4.14
    add(erf(t), [1], [], [S.Half], [0], t**2, 1/sqrt(pi))
    # TODO exp(-x)*erf(I*x) does not work
    add(erfc(t), [], [1], [0, S.Half], [], t**2, 1/sqrt(pi))
    # This formula for erfi(z) yields a wrong(?) minus sign
    #add(erfi(t), [1], [], [S.Half], [0], -t**2, I/sqrt(pi))
    add(erfi(t), [S.Half], [], [0], [Rational(-1, 2)], -t**2, t/sqrt(pi))

    # Fresnel Integrals
    add(fresnels(t), [1], [], [Rational(3, 4)], [0, Rational(1, 4)], pi**2*t**4/16, S.Half)
    add(fresnelc(t), [1], [], [Rational(1, 4)], [0, Rational(3, 4)], pi**2*t**4/16, S.Half)

    ##### bessel-type functions #####
    # Section 8.4.19
    add(besselj(a, t), [], [], [a/2], [-a/2], t**2/4)

    # all of the following are derivable
    #add(sin(t)*besselj(a, t), [Rational(1, 4), Rational(3, 4)], [], [(1+a)/2],
    #    [-a/2, a/2, (1-a)/2], t**2, 1/sqrt(2))
    #add(cos(t)*besselj(a, t), [Rational(1, 4), Rational(3, 4)], [], [a/2],
    #    [-a/2, (1+a)/2, (1-a)/2], t**2, 1/sqrt(2))
    #add(besselj(a, t)**2, [S.Half], [], [a], [-a, 0], t**2, 1/sqrt(pi))
    #add(besselj(a, t)*besselj(b, t), [0, S.Half], [], [(a + b)/2],
    #    [-(a+b)/2, (a - b)/2, (b - a)/2], t**2, 1/sqrt(pi))

    # Section 8.4.20
    add(bessely(a, t), [], [-(a + 1)/2], [a/2, -a/2], [-(a + 1)/2], t**2/4)

    # TODO all of the following should be derivable
    #add(sin(t)*bessely(a, t), [Rational(1, 4), Rational(3, 4)], [(1 - a - 1)/2],
    #    [(1 + a)/2, (1 - a)/2], [(1 - a - 1)/2, (1 - 1 - a)/2, (1 - 1 + a)/2],
    #    t**2, 1/sqrt(2))
    #add(cos(t)*bessely(a, t), [Rational(1, 4), Rational(3, 4)], [(0 - a - 1)/2],
    #    [(0 + a)/2, (0 - a)/2], [(0 - a - 1)/2, (1 - 0 - a)/2, (1 - 0 + a)/2],
    #    t**2, 1/sqrt(2))
    #add(besselj(a, t)*bessely(b, t), [0, S.Half], [(a - b - 1)/2],
    #    [(a + b)/2, (a - b)/2], [(a - b - 1)/2, -(a + b)/2, (b - a)/2],
    #    t**2, 1/sqrt(pi))
    #addi(bessely(a, t)**2,
    #     [(2/sqrt(pi), meijerg([], [S.Half, S.Half - a], [0, a, -a],
    #                           [S.Half - a], t**2)),
    #      (1/sqrt(pi), meijerg([S.Half], [], [a], [-a, 0], t**2))],
    #     True)
    #addi(bessely(a, t)*bessely(b, t),
    #     [(2/sqrt(pi), meijerg([], [0, S.Half, (1 - a - b)/2],
    #                           [(a + b)/2, (a - b)/2, (b - a)/2, -(a + b)/2],
    #                           [(1 - a - b)/2], t**2)),
    #      (1/sqrt(pi), meijerg([0, S.Half], [], [(a + b)/2],
    #                           [-(a + b)/2, (a - b)/2, (b - a)/2], t**2))],
    #     True)

    # Section 8.4.21 ?
    # Section 8.4.22
    add(besseli(a, t), [], [(1 + a)/2], [a/2], [-a/2, (1 + a)/2], t**2/4, pi)
    # TODO many more formulas. should all be derivable

    # Section 8.4.23
    add(besselk(a, t), [], [], [a/2, -a/2], [], t**2/4, S.Half)
    # TODO many more formulas. should all be derivable

    # Complete elliptic integrals K(z) and E(z)
    add(elliptic_k(t), [S.Half, S.Half], [], [0], [0], -t, S.Half)
    add(elliptic_e(t), [S.Half, 3*S.Half], [], [0], [0], -t, Rational(-1, 2)/2)


####################################################################
# First some helper functions.
####################################################################

from sympy.utilities.timeutils import timethis
timeit = timethis('meijerg')


def _mytype(f: Basic, x: Symbol) -> tuple[type[Basic], ...]:
    """ Create a hashable entity describing the type of f. """
    def key(x: type[Basic]) -> tuple[int, int, str]:
        return x.class_key()

    if x not in f.free_symbols:
        return ()
    elif f.is_Function:
        return type(f),
    return tuple(sorted((t for a in f.args for t in _mytype(a, x)), key=key))


class _CoeffExpValueError(ValueError):
    """
    Exception raised by _get_coeff_exp, for internal use only.
    """
    pass


def _get_coeff_exp(expr, x):
    """
    When expr is known to be of the form c*x**b, with c and/or b possibly 1,
    return c, b.

    Examples
    ========

    >>> from sympy.abc import x, a, b
    >>> from sympy.integrals.meijerint import _get_coeff_exp
    >>> _get_coeff_exp(a*x**b, x)
    (a, b)
    >>> _get_coeff_exp(x, x)
    (1, 1)
    >>> _get_coeff_exp(2*x, x)
    (2, 1)
    >>> _get_coeff_exp(x**3, x)
    (1, 3)
    """
    from sympy.simplify import powsimp
    (c, m) = expand_power_base(powsimp(expr)).as_coeff_mul(x)
    if not m:
        return c, S.Zero
    [m] = m
    if m.is_Pow:
        if m.base != x:
            raise _CoeffExpValueError('expr not of form a*x**b')
        return c, m.exp
    elif m == x:
        return c, S.One
    else:
        raise _CoeffExpValueError('expr not of form a*x**b: %s' % expr)


def _exponents(expr, x):
    """
    Find the exponents of ``x`` (not including zero) in ``expr``.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _exponents
    >>> from sympy.abc import x, y
    >>> from sympy import sin
    >>> _exponents(x, x)
    {1}
    >>> _exponents(x**2, x)
    {2}
    >>> _exponents(x**2 + x, x)
    {1, 2}
    >>> _exponents(x**3*sin(x + x**y) + 1/x, x)
    {-1, 1, 3, y}
    """
    def _exponents_(expr, x, res):
        if expr == x:
            res.update([1])
            return
        if expr.is_Pow and expr.base == x:
            res.update([expr.exp])
            return
        for argument in expr.args:
            _exponents_(argument, x, res)
    res = set()
    _exponents_(expr, x, res)
    return res


def _functions(expr, x):
    """ Find the types of functions in expr, to estimate the complexity. """
    return {e.func for e in expr.atoms(Function) if x in e.free_symbols}


def _find_splitting_points(expr, x):
    """
    Find numbers a such that a linear substitution x -> x + a would
    (hopefully) simplify expr.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _find_splitting_points as fsp
    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> fsp(x, x)
    {0}
    >>> fsp((x-1)**3, x)
    {1}
    >>> fsp(sin(x+3)*x, x)
    {-3, 0}
    """
    p, q = [Wild(n, exclude=[x]) for n in 'pq']

    def compute_innermost(expr, res):
        if not isinstance(expr, Expr):
            return
        m = expr.match(p*x + q)
        if m and m[p] != 0:
            res.add(-m[q]/m[p])
            return
        if expr.is_Atom:
            return
        for argument in expr.args:
            compute_innermost(argument, res)
    innermost = set()
    compute_innermost(expr, innermost)
    return innermost


def _split_mul(f, x):
    """
    Split expression ``f`` into fac, po, g, where fac is a constant factor,
    po = x**s for some s independent of s, and g is "the rest".

    Examples
    ========

    >>> from sympy.integrals.meijerint import _split_mul
    >>> from sympy import sin
    >>> from sympy.abc import s, x
    >>> _split_mul((3*x)**s*sin(x**2)*x, x)
    (3**s, x*x**s, sin(x**2))
    """
    fac = S.One
    po = S.One
    g = S.One
    f = expand_power_base(f)

    args = Mul.make_args(f)
    for a in args:
        if a == x:
            po *= x
        elif x not in a.free_symbols:
            fac *= a
        else:
            if a.is_Pow and x not in a.exp.free_symbols:
                c, t = a.base.as_coeff_mul(x)
                if t != (x,):
                    c, t = expand_mul(a.base).as_coeff_mul(x)
                if t == (x,):
                    po *= x**a.exp
                    fac *= unpolarify(polarify(c**a.exp, subs=False))
                    continue
            g *= a

    return fac, po, g


def _mul_args(f):
    """
    Return a list ``L`` such that ``Mul(*L) == f``.

    If ``f`` is not a ``Mul`` or ``Pow``, ``L=[f]``.
    If ``f=g**n`` for an integer ``n``, ``L=[g]*n``.
    If ``f`` is a ``Mul``, ``L`` comes from applying ``_mul_args`` to all factors of ``f``.
    """
    args = Mul.make_args(f)
    gs = []
    for g in args:
        if g.is_Pow and g.exp.is_Integer:
            n = g.exp
            base = g.base
            if n < 0:
                n = -n
                base = 1/base
            gs += [base]*n
        else:
            gs.append(g)
    return gs


def _mul_as_two_parts(f):
    """
    Find all the ways to split ``f`` into a product of two terms.
    Return None on failure.

    Explanation
    ===========

    Although the order is canonical from multiset_partitions, this is
    not necessarily the best order to process the terms. For example,
    if the case of len(gs) == 2 is removed and multiset is allowed to
    sort the terms, some tests fail.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _mul_as_two_parts
    >>> from sympy import sin, exp, ordered
    >>> from sympy.abc import x
    >>> list(ordered(_mul_as_two_parts(x*sin(x)*exp(x))))
    [(x, exp(x)*sin(x)), (x*exp(x), sin(x)), (x*sin(x), exp(x))]
    """

    gs = _mul_args(f)
    if len(gs) < 2:
        return None
    if len(gs) == 2:
        return [tuple(gs)]
    return [(Mul(*x), Mul(*y)) for (x, y) in multiset_partitions(gs, 2)]


def _inflate_g(g, n):
    """ Return C, h such that h is a G function of argument z**n and
        g = C*h. """
    # TODO should this be a method of meijerg?
    # See: [L, page 150, equation (5)]
    def inflate(params, n):
        """ (a1, .., ak) -> (a1/n, (a1+1)/n, ..., (ak + n-1)/n) """
        return [(a + i)/n for a, i in itertools.product(params, range(n))]
    v = S(len(g.ap) - len(g.bq))
    C = n**(1 + g.nu + v/2)
    C /= (2*pi)**((n - 1)*g.delta)
    return C, meijerg(inflate(g.an, n), inflate(g.aother, n),
                      inflate(g.bm, n), inflate(g.bother, n),
                      g.argument**n * n**(n*v))


def _flip_g(g):
    """ Turn the G function into one of inverse argument
        (i.e. G(1/x) -> G'(x)) """
    # See [L], section 5.2
    def tr(l):
        return [1 - a for a in l]
    return meijerg(tr(g.bm), tr(g.bother), tr(g.an), tr(g.aother), 1/g.argument)


def _inflate_fox_h(g, a):
    r"""
    Let d denote the integrand in the definition of the G function ``g``.
    Consider the function H which is defined in the same way, but with
    integrand d/Gamma(a*s) (contour conventions as usual).

    If ``a`` is rational, the function H can be written as C*G, for a constant C
    and a G-function G.

    This function returns C, G.
    """
    if a < 0:
        return _inflate_fox_h(_flip_g(g), -a)
    p = S(a.p)
    q = S(a.q)
    # We use the substitution s->qs, i.e. inflate g by q. We are left with an
    # extra factor of Gamma(p*s), for which we use Gauss' multiplication
    # theorem.
    D, g = _inflate_g(g, q)
    z = g.argument
    D /= (2*pi)**((1 - p)/2)*p**Rational(-1, 2)
    z /= p**p
    bs = [(n + 1)/p for n in range(p)]
    return D, meijerg(g.an, g.aother, g.bm, list(g.bother) + bs, z)


_dummies: dict[tuple[str, str], Dummy]  = {}


def _dummy(name, token, expr, **kwargs):
    """
    Return a dummy. This will return the same dummy if the same token+name is
    requested more than once, and it is not already in expr.
    This is for being cache-friendly.
    """
    d = _dummy_(name, token, **kwargs)
    if d in expr.free_symbols:
        return Dummy(name, **kwargs)
    return d


def _dummy_(name, token, **kwargs):
    """
    Return a dummy associated to name and token. Same effect as declaring
    it globally.
    """
    if not (name, token) in _dummies:
        _dummies[(name, token)] = Dummy(name, **kwargs)
    return _dummies[(name, token)]


def _is_analytic(f, x):
    """ Check if f(x), when expressed using G functions on the positive reals,
        will in fact agree with the G functions almost everywhere """
    return not any(x in expr.free_symbols for expr in f.atoms(Heaviside, Abs))


def _condsimp(cond, first=True):
    """
    Do naive simplifications on ``cond``.

    Explanation
    ===========

    Note that this routine is completely ad-hoc, simplification rules being
    added as need arises rather than following any logical pattern.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _condsimp as simp
    >>> from sympy import Or, Eq
    >>> from sympy.abc import x, y
    >>> simp(Or(x < y, Eq(x, y)))
    x <= y
    """
    if first:
        cond = cond.replace(lambda _: _.is_Relational, _canonical_coeff)
        first = False
    if not isinstance(cond, BooleanFunction):
        return cond
    p, q, r = symbols('p q r', cls=Wild)
    # transforms tests use 0, 4, 5 and 11-14
    # meijer tests use 0, 2, 11, 14
    # joint_rv uses 6, 7
    rules = [
        (Or(p < q, Eq(p, q)), p <= q),  # 0
        # The next two obviously are instances of a general pattern, but it is
        # easier to spell out the few cases we care about.
        (And(Abs(arg(p)) <= pi, Abs(arg(p) - 2*pi) <= pi),
         Eq(arg(p) - pi, 0)),  # 1
        (And(Abs(2*arg(p) + pi) <= pi, Abs(2*arg(p) - pi) <= pi),
         Eq(arg(p), 0)), # 2
        (And(Abs(2*arg(p) + pi) < pi, Abs(2*arg(p) - pi) <= pi),
         S.false),  # 3
        (And(Abs(arg(p) - pi/2) <= pi/2, Abs(arg(p) + pi/2) <= pi/2),
         Eq(arg(p), 0)),  # 4
        (And(Abs(arg(p) - pi/2) <= pi/2, Abs(arg(p) + pi/2) < pi/2),
         S.false),  # 5
        (And(Abs(arg(p**2/2 + 1)) < pi, Ne(Abs(arg(p**2/2 + 1)), pi)),
         S.true),  # 6
        (Or(Abs(arg(p**2/2 + 1)) < pi, Ne(1/(p**2/2 + 1), 0)),
         S.true),  # 7
        (And(Abs(unbranched_argument(p)) <= pi,
           Abs(unbranched_argument(exp_polar(-2*pi*S.ImaginaryUnit)*p)) <= pi),
         Eq(unbranched_argument(exp_polar(-S.ImaginaryUnit*pi)*p), 0)),  # 8
        (And(Abs(unbranched_argument(p)) <= pi/2,
           Abs(unbranched_argument(exp_polar(-pi*S.ImaginaryUnit)*p)) <= pi/2),
         Eq(unbranched_argument(exp_polar(-S.ImaginaryUnit*pi/2)*p), 0)),  # 9
        (Or(p <= q, And(p < q, r)), p <= q),  # 10
        (Ne(p**2, 1) & (p**2 > 1), p**2 > 1),  # 11
        (Ne(1/p, 1) & (cos(Abs(arg(p)))*Abs(p) > 1), Abs(p) > 1),  # 12
        (Ne(p, 2) & (cos(Abs(arg(p)))*Abs(p) > 2), Abs(p) > 2),  # 13
        ((Abs(arg(p)) < pi/2) & (cos(Abs(arg(p)))*sqrt(Abs(p**2)) > 1), p**2 > 1),  # 14
    ]
    cond = cond.func(*[_condsimp(_, first) for _ in cond.args])
    change = True
    while change:
        change = False
        for irule, (fro, to) in enumerate(rules):
            if fro.func != cond.func:
                continue
            for n, arg1 in enumerate(cond.args):
                if r in fro.args[0].free_symbols:
                    m = arg1.match(fro.args[1])
                    num = 1
                else:
                    num = 0
                    m = arg1.match(fro.args[0])
                if not m:
                    continue
                otherargs = [x.subs(m) for x in fro.args[:num] + fro.args[num + 1:]]
                otherlist = [n]
                for arg2 in otherargs:
                    for k, arg3 in enumerate(cond.args):
                        if k in otherlist:
                            continue
                        if arg2 == arg3:
                            otherlist += [k]
                            break
                        if isinstance(arg3, And) and arg2.args[1] == r and \
                                isinstance(arg2, And) and arg2.args[0] in arg3.args:
                            otherlist += [k]
                            break
                        if isinstance(arg3, And) and arg2.args[0] == r and \
                                isinstance(arg2, And) and arg2.args[1] in arg3.args:
                            otherlist += [k]
                            break
                if len(otherlist) != len(otherargs) + 1:
                    continue
                newargs = [arg_ for (k, arg_) in enumerate(cond.args)
                           if k not in otherlist] + [to.subs(m)]
                if SYMPY_DEBUG:
                    if irule not in (0, 2, 4, 5, 6, 7, 11, 12, 13, 14):
                        print('used new rule:', irule)
                cond = cond.func(*newargs)
                change = True
                break

    # final tweak
    def rel_touchup(rel):
        if rel.rel_op != '==' or rel.rhs != 0:
            return rel

        # handle Eq(*, 0)
        LHS = rel.lhs
        m = LHS.match(arg(p)**q)
        if not m:
            m = LHS.match(unbranched_argument(polar_lift(p)**q))
        if not m:
            if isinstance(LHS, periodic_argument) and not LHS.args[0].is_polar \
                    and LHS.args[1] is S.Infinity:
                return (LHS.args[0] > 0)
            return rel
        return (m[p] > 0)
    cond = cond.replace(lambda _: _.is_Relational, rel_touchup)
    if SYMPY_DEBUG:
        print('_condsimp: ', cond)
    return cond

def _eval_cond(cond):
    """ Re-evaluate the conditions. """
    if isinstance(cond, bool):
        return cond
    return _condsimp(cond.doit())

####################################################################
# Now the "backbone" functions to do actual integration.
####################################################################


def _my_principal_branch(expr, period, full_pb=False):
    """ Bring expr nearer to its principal branch by removing superfluous
        factors.
        This function does *not* guarantee to yield the principal branch,
        to avoid introducing opaque principal_branch() objects,
        unless full_pb=True. """
    res = principal_branch(expr, period)
    if not full_pb:
        res = res.replace(principal_branch, lambda x, y: x)
    return res


def _rewrite_saxena_1(fac, po, g, x):
    """
    Rewrite the integral fac*po*g dx, from zero to infinity, as
    integral fac*G, where G has argument a*x. Note po=x**s.
    Return fac, G.
    """
    _, s = _get_coeff_exp(po, x)
    a, b = _get_coeff_exp(g.argument, x)
    period = g.get_period()
    a = _my_principal_branch(a, period)

    # We substitute t = x**b.
    C = fac/(Abs(b)*a**((s + 1)/b - 1))
    # Absorb a factor of (at)**((1 + s)/b - 1).

    def tr(l):
        return [a + (1 + s)/b - 1 for a in l]
    return C, meijerg(tr(g.an), tr(g.aother), tr(g.bm), tr(g.bother),
                      a*x)


def _check_antecedents_1(g, x, helper=False):
    r"""
    Return a condition under which the mellin transform of g exists.
    Any power of x has already been absorbed into the G function,
    so this is just $\int_0^\infty g\, dx$.

    See [L, section 5.6.1]. (Note that s=1.)

    If ``helper`` is True, only check if the MT exists at infinity, i.e. if
    $\int_1^\infty g\, dx$ exists.
    """
    # NOTE if you update these conditions, please update the documentation as well
    delta = g.delta
    eta, _ = _get_coeff_exp(g.argument, x)
    m, n, p, q = S([len(g.bm), len(g.an), len(g.ap), len(g.bq)])

    if p > q:
        def tr(l):
            return [1 - x for x in l]
        return _check_antecedents_1(meijerg(tr(g.bm), tr(g.bother),
                                            tr(g.an), tr(g.aother), x/eta),
                                    x)

    tmp = [-re(b) < 1 for b in g.bm] + [1 < 1 - re(a) for a in g.an]
    cond_3 = And(*tmp)

    tmp += [-re(b) < 1 for b in g.bother]
    tmp += [1 < 1 - re(a) for a in g.aother]
    cond_3_star = And(*tmp)

    cond_4 = (-re(g.nu) + (q + 1 - p)/2 > q - p)

    def debug(*msg):
        _debug(*msg)

    def debugf(string, arg):
        _debugf(string, arg)

    debug('Checking antecedents for 1 function:')
    debugf('  delta=%s, eta=%s, m=%s, n=%s, p=%s, q=%s',
           (delta, eta, m, n, p, q))
    debugf('  ap = %s, %s', (list(g.an), list(g.aother)))
    debugf('  bq = %s, %s', (list(g.bm), list(g.bother)))
    debugf('  cond_3=%s, cond_3*=%s, cond_4=%s', (cond_3, cond_3_star, cond_4))

    conds = []

    # case 1
    case1 = []
    tmp1 = [1 <= n, p < q, 1 <= m]
    tmp2 = [1 <= p, 1 <= m, Eq(q, p + 1), Not(And(Eq(n, 0), Eq(m, p + 1)))]
    tmp3 = [1 <= p, Eq(q, p)]
    for k in range(ceiling(delta/2) + 1):
        tmp3 += [Ne(Abs(unbranched_argument(eta)), (delta - 2*k)*pi)]
    tmp = [delta > 0, Abs(unbranched_argument(eta)) < delta*pi]
    extra = [Ne(eta, 0), cond_3]
    if helper:
        extra = []
    for t in [tmp1, tmp2, tmp3]:
        case1 += [And(*(t + tmp + extra))]
    conds += case1
    debug('  case 1:', case1)

    # case 2
    extra = [cond_3]
    if helper:
        extra = []
    case2 = [And(Eq(n, 0), p + 1 <= m, m <= q,
                 Abs(unbranched_argument(eta)) < delta*pi, *extra)]
    conds += case2
    debug('  case 2:', case2)

    # case 3
    extra = [cond_3, cond_4]
    if helper:
        extra = []
    case3 = [And(p < q, 1 <= m, delta > 0, Eq(Abs(unbranched_argument(eta)), delta*pi),
                 *extra)]
    case3 += [And(p <= q - 2, Eq(delta, 0), Eq(Abs(unbranched_argument(eta)), 0), *extra)]
    conds += case3
    debug('  case 3:', case3)

    # TODO altered cases 4-7

    # extra case from wofram functions site:
    # (reproduced verbatim from Prudnikov, section 2.24.2)
    # https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/01/
    case_extra = []
    case_extra += [Eq(p, q), Eq(delta, 0), Eq(unbranched_argument(eta), 0), Ne(eta, 0)]
    if not helper:
        case_extra += [cond_3]
    s = []
    for a, b in zip(g.ap, g.bq):
        s += [b - a]
    case_extra += [re(Add(*s)) < 0]
    case_extra = And(*case_extra)
    conds += [case_extra]
    debug('  extra case:', [case_extra])

    case_extra_2 = [And(delta > 0, Abs(unbranched_argument(eta)) < delta*pi)]
    if not helper:
        case_extra_2 += [cond_3]
    case_extra_2 = And(*case_extra_2)
    conds += [case_extra_2]
    debug('  second extra case:', [case_extra_2])

    # TODO This leaves only one case from the three listed by Prudnikov.
    #      Investigate if these indeed cover everything; if so, remove the rest.

    return Or(*conds)


def _int0oo_1(g, x):
    r"""
    Evaluate $\int_0^\infty g\, dx$ using G functions,
    assuming the necessary conditions are fulfilled.

    Examples
    ========

    >>> from sympy.abc import a, b, c, d, x, y
    >>> from sympy import meijerg
    >>> from sympy.integrals.meijerint import _int0oo_1
    >>> _int0oo_1(meijerg([a], [b], [c], [d], x*y), x)
    gamma(-a)*gamma(c + 1)/(y*gamma(-d)*gamma(b + 1))
    """
    from sympy.simplify import gammasimp
    # See [L, section 5.6.1]. Note that s=1.
    eta, _ = _get_coeff_exp(g.argument, x)
    res = 1/eta
    # XXX TODO we should reduce order first
    for b in g.bm:
        res *= gamma(b + 1)
    for a in g.an:
        res *= gamma(1 - a - 1)
    for b in g.bother:
        res /= gamma(1 - b - 1)
    for a in g.aother:
        res /= gamma(a + 1)
    return gammasimp(unpolarify(res))


def _rewrite_saxena(fac, po, g1, g2, x, full_pb=False):
    """
    Rewrite the integral ``fac*po*g1*g2`` from 0 to oo in terms of G
    functions with argument ``c*x``.

    Explanation
    ===========

    Return C, f1, f2 such that integral C f1 f2 from 0 to infinity equals
    integral fac ``po``, ``g1``, ``g2`` from 0 to infinity.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _rewrite_saxena
    >>> from sympy.abc import s, t, m
    >>> from sympy import meijerg
    >>> g1 = meijerg([], [], [0], [], s*t)
    >>> g2 = meijerg([], [], [m/2], [-m/2], t**2/4)
    >>> r = _rewrite_saxena(1, t**0, g1, g2, t)
    >>> r[0]
    s/(4*sqrt(pi))
    >>> r[1]
    meijerg(((), ()), ((-1/2, 0), ()), s**2*t/4)
    >>> r[2]
    meijerg(((), ()), ((m/2,), (-m/2,)), t/4)
    """
    def pb(g):
        a, b = _get_coeff_exp(g.argument, x)
        per = g.get_period()
        return meijerg(g.an, g.aother, g.bm, g.bother,
                       _my_principal_branch(a, per, full_pb)*x**b)

    _, s = _get_coeff_exp(po, x)
    _, b1 = _get_coeff_exp(g1.argument, x)
    _, b2 = _get_coeff_exp(g2.argument, x)
    if (b1 < 0) == True:
        b1 = -b1
        g1 = _flip_g(g1)
    if (b2 < 0) == True:
        b2 = -b2
        g2 = _flip_g(g2)
    if not b1.is_Rational or not b2.is_Rational:
        return
    m1, n1 = b1.p, b1.q
    m2, n2 = b2.p, b2.q
    tau = ilcm(m1*n2, m2*n1)
    r1 = tau//(m1*n2)
    r2 = tau//(m2*n1)

    C1, g1 = _inflate_g(g1, r1)
    C2, g2 = _inflate_g(g2, r2)
    g1 = pb(g1)
    g2 = pb(g2)

    fac *= C1*C2
    a1, b = _get_coeff_exp(g1.argument, x)
    a2, _ = _get_coeff_exp(g2.argument, x)

    # arbitrarily tack on the x**s part to g1
    # TODO should we try both?
    exp = (s + 1)/b - 1
    fac = fac/(Abs(b) * a1**exp)

    def tr(l):
        return [a + exp for a in l]
    g1 = meijerg(tr(g1.an), tr(g1.aother), tr(g1.bm), tr(g1.bother), a1*x)
    g2 = meijerg(g2.an, g2.aother, g2.bm, g2.bother, a2*x)

    from sympy.simplify import powdenest
    return powdenest(fac, polar=True), g1, g2


def _check_antecedents(g1, g2, x):
    """ Return a condition under which the integral theorem applies. """
    #  Yes, this is madness.
    # XXX TODO this is a testing *nightmare*
    # NOTE if you update these conditions, please update the documentation as well

    # The following conditions are found in
    # [P], Section 2.24.1
    #
    # They are also reproduced (verbatim!) at
    # https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/03/
    #
    # Note: k=l=r=alpha=1
    sigma, _ = _get_coeff_exp(g1.argument, x)
    omega, _ = _get_coeff_exp(g2.argument, x)
    s, t, u, v = S([len(g1.bm), len(g1.an), len(g1.ap), len(g1.bq)])
    m, n, p, q = S([len(g2.bm), len(g2.an), len(g2.ap), len(g2.bq)])
    bstar = s + t - (u + v)/2
    cstar = m + n - (p + q)/2
    rho = g1.nu + (u - v)/2 + 1
    mu = g2.nu + (p - q)/2 + 1
    phi = q - p - (v - u)
    eta = 1 - (v - u) - mu - rho
    psi = (pi*(q - m - n) + Abs(unbranched_argument(omega)))/(q - p)
    theta = (pi*(v - s - t) + Abs(unbranched_argument(sigma)))/(v - u)

    _debug('Checking antecedents:')
    _debugf('  sigma=%s, s=%s, t=%s, u=%s, v=%s, b*=%s, rho=%s',
            (sigma, s, t, u, v, bstar, rho))
    _debugf('  omega=%s, m=%s, n=%s, p=%s, q=%s, c*=%s, mu=%s,',
            (omega, m, n, p, q, cstar, mu))
    _debugf('  phi=%s, eta=%s, psi=%s, theta=%s', (phi, eta, psi, theta))

    def _c1():
        for g in [g1, g2]:
            for i, j in itertools.product(g.an, g.bm):
                diff = i - j
                if diff.is_integer and diff.is_positive:
                    return False
        return True
    c1 = _c1()
    c2 = And(*[re(1 + i + j) > 0 for i in g1.bm for j in g2.bm])
    c3 = And(*[re(1 + i + j) < 1 + 1 for i in g1.an for j in g2.an])
    c4 = And(*[(p - q)*re(1 + i - 1) - re(mu) > Rational(-3, 2) for i in g1.an])
    c5 = And(*[(p - q)*re(1 + i) - re(mu) > Rational(-3, 2) for i in g1.bm])
    c6 = And(*[(u - v)*re(1 + i - 1) - re(rho) > Rational(-3, 2) for i in g2.an])
    c7 = And(*[(u - v)*re(1 + i) - re(rho) > Rational(-3, 2) for i in g2.bm])
    c8 = (Abs(phi) + 2*re((rho - 1)*(q - p) + (v - u)*(q - p) + (mu -
          1)*(v - u)) > 0)
    c9 = (Abs(phi) - 2*re((rho - 1)*(q - p) + (v - u)*(q - p) + (mu -
          1)*(v - u)) > 0)
    c10 = (Abs(unbranched_argument(sigma)) < bstar*pi)
    c11 = Eq(Abs(unbranched_argument(sigma)), bstar*pi)
    c12 = (Abs(unbranched_argument(omega)) < cstar*pi)
    c13 = Eq(Abs(unbranched_argument(omega)), cstar*pi)

    # The following condition is *not* implemented as stated on the wolfram
    # function site. In the book of Prudnikov there is an additional part
    # (the And involving re()). However, I only have this book in russian, and
    # I don't read any russian. The following condition is what other people
    # have told me it means.
    # Worryingly, it is different from the condition implemented in REDUCE.
    # The REDUCE implementation:
    #   https://reduce-algebra.svn.sourceforge.net/svnroot/reduce-algebra/trunk/packages/defint/definta.red
    #   (search for tst14)
    # The Wolfram alpha version:
    #   https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/03/03/0014/
    z0 = exp(-(bstar + cstar)*pi*S.ImaginaryUnit)
    zos = unpolarify(z0*omega/sigma)
    zso = unpolarify(z0*sigma/omega)
    if zos == 1/zso:
        c14 = And(Eq(phi, 0), bstar + cstar <= 1,
                  Or(Ne(zos, 1), re(mu + rho + v - u) < 1,
                     re(mu + rho + q - p) < 1))
    else:
        def _cond(z):
            '''Returns True if abs(arg(1-z)) < pi, avoiding arg(0).

            Explanation
            ===========

            If ``z`` is 1 then arg is NaN. This raises a
            TypeError on `NaN < pi`. Previously this gave `False` so
            this behavior has been hardcoded here but someone should
            check if this NaN is more serious! This NaN is triggered by
            test_meijerint() in test_meijerint.py:
            `meijerint_definite(exp(x), x, 0, I)`
            '''
            return z != 1 and Abs(arg(1 - z)) < pi

        c14 = And(Eq(phi, 0), bstar - 1 + cstar <= 0,
                  Or(And(Ne(zos, 1), _cond(zos)),
                     And(re(mu + rho + v - u) < 1, Eq(zos, 1))))

        c14_alt = And(Eq(phi, 0), cstar - 1 + bstar <= 0,
                  Or(And(Ne(zso, 1), _cond(zso)),
                     And(re(mu + rho + q - p) < 1, Eq(zso, 1))))

        # Since r=k=l=1, in our case there is c14_alt which is the same as calling
        # us with (g1, g2) = (g2, g1). The conditions below enumerate all cases
        # (i.e. we don't have to try arguments reversed by hand), and indeed try
        # all symmetric cases. (i.e. whenever there is a condition involving c14,
        # there is also a dual condition which is exactly what we would get when g1,
        # g2 were interchanged, *but c14 was unaltered*).
        # Hence the following seems correct:
        c14 = Or(c14, c14_alt)

    '''
    When `c15` is NaN (e.g. from `psi` being NaN as happens during
    'test_issue_4992' and/or `theta` is NaN as in 'test_issue_6253',
    both in `test_integrals.py`) the comparison to 0 formerly gave False
    whereas now an error is raised. To keep the old behavior, the value
    of NaN is replaced with False but perhaps a closer look at this condition
    should be made: XXX how should conditions leading to c15=NaN be handled?
    '''
    try:
        lambda_c = (q - p)*Abs(omega)**(1/(q - p))*cos(psi) \
            + (v - u)*Abs(sigma)**(1/(v - u))*cos(theta)
        # the TypeError might be raised here, e.g. if lambda_c is NaN
        if _eval_cond(lambda_c > 0) != False:
            c15 = (lambda_c > 0)
        else:
            def lambda_s0(c1, c2):
                return c1*(q - p)*Abs(omega)**(1/(q - p))*sin(psi) \
                    + c2*(v - u)*Abs(sigma)**(1/(v - u))*sin(theta)
            lambda_s = Piecewise(
                ((lambda_s0(+1, +1)*lambda_s0(-1, -1)),
                 And(Eq(unbranched_argument(sigma), 0), Eq(unbranched_argument(omega), 0))),
                (lambda_s0(sign(unbranched_argument(omega)), +1)*lambda_s0(sign(unbranched_argument(omega)), -1),
                 And(Eq(unbranched_argument(sigma), 0), Ne(unbranched_argument(omega), 0))),
                (lambda_s0(+1, sign(unbranched_argument(sigma)))*lambda_s0(-1, sign(unbranched_argument(sigma))),
                 And(Ne(unbranched_argument(sigma), 0), Eq(unbranched_argument(omega), 0))),
                (lambda_s0(sign(unbranched_argument(omega)), sign(unbranched_argument(sigma))), True))
            tmp = [lambda_c > 0,
                   And(Eq(lambda_c, 0), Ne(lambda_s, 0), re(eta) > -1),
                   And(Eq(lambda_c, 0), Eq(lambda_s, 0), re(eta) > 0)]
            c15 = Or(*tmp)
    except TypeError:
        c15 = False
    for cond, i in [(c1, 1), (c2, 2), (c3, 3), (c4, 4), (c5, 5), (c6, 6),
                    (c7, 7), (c8, 8), (c9, 9), (c10, 10), (c11, 11),
                    (c12, 12), (c13, 13), (c14, 14), (c15, 15)]:
        _debugf('  c%s: %s', (i, cond))

    # We will return Or(*conds)
    conds = []

    def pr(count):
        _debugf('  case %s: %s', (count, conds[-1]))
    conds += [And(m*n*s*t != 0, bstar.is_positive is True, cstar.is_positive is True, c1, c2, c3, c10,
                  c12)]  # 1
    pr(1)
    conds += [And(Eq(u, v), Eq(bstar, 0), cstar.is_positive is True, sigma.is_positive is True, re(rho) < 1,
                  c1, c2, c3, c12)]  # 2
    pr(2)
    conds += [And(Eq(p, q), Eq(cstar, 0), bstar.is_positive is True, omega.is_positive is True, re(mu) < 1,
                  c1, c2, c3, c10)]  # 3
    pr(3)
    conds += [And(Eq(p, q), Eq(u, v), Eq(bstar, 0), Eq(cstar, 0),
                  sigma.is_positive is True, omega.is_positive is True, re(mu) < 1, re(rho) < 1,
                  Ne(sigma, omega), c1, c2, c3)]  # 4
    pr(4)
    conds += [And(Eq(p, q), Eq(u, v), Eq(bstar, 0), Eq(cstar, 0),
                  sigma.is_positive is True, omega.is_positive is True, re(mu + rho) < 1,
                  Ne(omega, sigma), c1, c2, c3)]  # 5
    pr(5)
    conds += [And(p > q, s.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  c1, c2, c3, c5, c10, c13)]  # 6
    pr(6)
    conds += [And(p < q, t.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  c1, c2, c3, c4, c10, c13)]  # 7
    pr(7)
    conds += [And(u > v, m.is_positive is True, cstar.is_positive is True, bstar >= 0,
                  c1, c2, c3, c7, c11, c12)]  # 8
    pr(8)
    conds += [And(u < v, n.is_positive is True, cstar.is_positive is True, bstar >= 0,
                  c1, c2, c3, c6, c11, c12)]  # 9
    pr(9)
    conds += [And(p > q, Eq(u, v), Eq(bstar, 0), cstar >= 0, sigma.is_positive is True,
                  re(rho) < 1, c1, c2, c3, c5, c13)]  # 10
    pr(10)
    conds += [And(p < q, Eq(u, v), Eq(bstar, 0), cstar >= 0, sigma.is_positive is True,
                  re(rho) < 1, c1, c2, c3, c4, c13)]  # 11
    pr(11)
    conds += [And(Eq(p, q), u > v, bstar >= 0, Eq(cstar, 0), omega.is_positive is True,
                  re(mu) < 1, c1, c2, c3, c7, c11)]  # 12
    pr(12)
    conds += [And(Eq(p, q), u < v, bstar >= 0, Eq(cstar, 0), omega.is_positive is True,
                  re(mu) < 1, c1, c2, c3, c6, c11)]  # 13
    pr(13)
    conds += [And(p < q, u > v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c4, c7, c11, c13)]  # 14
    pr(14)
    conds += [And(p > q, u < v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c5, c6, c11, c13)]  # 15
    pr(15)
    conds += [And(p > q, u > v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c5, c7, c8, c11, c13, c14)]  # 16
    pr(16)
    conds += [And(p < q, u < v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c4, c6, c9, c11, c13, c14)]  # 17
    pr(17)
    conds += [And(Eq(t, 0), s.is_positive is True, bstar.is_positive is True, phi.is_positive is True, c1, c2, c10)]  # 18
    pr(18)
    conds += [And(Eq(s, 0), t.is_positive is True, bstar.is_positive is True, phi.is_negative is True, c1, c3, c10)]  # 19
    pr(19)
    conds += [And(Eq(n, 0), m.is_positive is True, cstar.is_positive is True, phi.is_negative is True, c1, c2, c12)]  # 20
    pr(20)
    conds += [And(Eq(m, 0), n.is_positive is True, cstar.is_positive is True, phi.is_positive is True, c1, c3, c12)]  # 21
    pr(21)
    conds += [And(Eq(s*t, 0), bstar.is_positive is True, cstar.is_positive is True,
                  c1, c2, c3, c10, c12)]  # 22
    pr(22)
    conds += [And(Eq(m*n, 0), bstar.is_positive is True, cstar.is_positive is True,
                  c1, c2, c3, c10, c12)]  # 23
    pr(23)

    # The following case is from [Luke1969]. As far as I can tell, it is *not*
    # covered by Prudnikov's.
    # Let G1 and G2 be the two G-functions. Suppose the integral exists from
    # 0 to a > 0 (this is easy the easy part), that G1 is exponential decay at
    # infinity, and that the mellin transform of G2 exists.
    # Then the integral exists.
    mt1_exists = _check_antecedents_1(g1, x, helper=True)
    mt2_exists = _check_antecedents_1(g2, x, helper=True)
    conds += [And(mt2_exists, Eq(t, 0), u < s, bstar.is_positive is True, c10, c1, c2, c3)]
    pr('E1')
    conds += [And(mt2_exists, Eq(s, 0), v < t, bstar.is_positive is True, c10, c1, c2, c3)]
    pr('E2')
    conds += [And(mt1_exists, Eq(n, 0), p < m, cstar.is_positive is True, c12, c1, c2, c3)]
    pr('E3')
    conds += [And(mt1_exists, Eq(m, 0), q < n, cstar.is_positive is True, c12, c1, c2, c3)]
    pr('E4')

    # Let's short-circuit if this worked ...
    # the rest is corner-cases and terrible to read.
    r = Or(*conds)
    if _eval_cond(r) != False:
        return r

    conds += [And(m + n > p, Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True, cstar.is_negative is True,
                  Abs(unbranched_argument(omega)) < (m + n - p + 1)*pi,
                  c1, c2, c10, c14, c15)]  # 24
    pr(24)
    conds += [And(m + n > q, Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True, cstar.is_negative is True,
                  Abs(unbranched_argument(omega)) < (m + n - q + 1)*pi,
                  c1, c3, c10, c14, c15)]  # 25
    pr(25)
    conds += [And(Eq(p, q - 1), Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  c1, c2, c10, c14, c15)]  # 26
    pr(26)
    conds += [And(Eq(p, q + 1), Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  c1, c3, c10, c14, c15)]  # 27
    pr(27)
    conds += [And(p < q - 1, Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  Abs(unbranched_argument(omega)) < (m + n - p + 1)*pi,
                  c1, c2, c10, c14, c15)]  # 28
    pr(28)
    conds += [And(
        p > q + 1, Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  cstar*pi < Abs(unbranched_argument(omega)),
                  Abs(unbranched_argument(omega)) < (m + n - q + 1)*pi,
                  c1, c3, c10, c14, c15)]  # 29
    pr(29)
    conds += [And(Eq(n, 0), Eq(phi, 0), s + t > 0, m.is_positive is True, cstar.is_positive is True, bstar.is_negative is True,
                  Abs(unbranched_argument(sigma)) < (s + t - u + 1)*pi,
                  c1, c2, c12, c14, c15)]  # 30
    pr(30)
    conds += [And(Eq(m, 0), Eq(phi, 0), s + t > v, n.is_positive is True, cstar.is_positive is True, bstar.is_negative is True,
                  Abs(unbranched_argument(sigma)) < (s + t - v + 1)*pi,
                  c1, c3, c12, c14, c15)]  # 31
    pr(31)
    conds += [And(Eq(n, 0), Eq(phi, 0), Eq(u, v - 1), m.is_positive is True, cstar.is_positive is True,
                  bstar >= 0, bstar*pi < Abs(unbranched_argument(sigma)),
                  Abs(unbranched_argument(sigma)) < (bstar + 1)*pi,
                  c1, c2, c12, c14, c15)]  # 32
    pr(32)
    conds += [And(Eq(m, 0), Eq(phi, 0), Eq(u, v + 1), n.is_positive is True, cstar.is_positive is True,
                  bstar >= 0, bstar*pi < Abs(unbranched_argument(sigma)),
                  Abs(unbranched_argument(sigma)) < (bstar + 1)*pi,
                  c1, c3, c12, c14, c15)]  # 33
    pr(33)
    conds += [And(
        Eq(n, 0), Eq(phi, 0), u < v - 1, m.is_positive is True, cstar.is_positive is True, bstar >= 0,
        bstar*pi < Abs(unbranched_argument(sigma)),
        Abs(unbranched_argument(sigma)) < (s + t - u + 1)*pi,
        c1, c2, c12, c14, c15)]  # 34
    pr(34)
    conds += [And(
        Eq(m, 0), Eq(phi, 0), u > v + 1, n.is_positive is True, cstar.is_positive is True, bstar >= 0,
        bstar*pi < Abs(unbranched_argument(sigma)),
        Abs(unbranched_argument(sigma)) < (s + t - v + 1)*pi,
        c1, c3, c12, c14, c15)]  # 35
    pr(35)

    return Or(*conds)

    # NOTE An alternative, but as far as I can tell weaker, set of conditions
    #      can be found in [L, section 5.6.2].


def _int0oo(g1, g2, x):
    """
    Express integral from zero to infinity g1*g2 using a G function,
    assuming the necessary conditions are fulfilled.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _int0oo
    >>> from sympy.abc import s, t, m
    >>> from sympy import meijerg, S
    >>> g1 = meijerg([], [], [-S(1)/2, 0], [], s**2*t/4)
    >>> g2 = meijerg([], [], [m/2], [-m/2], t/4)
    >>> _int0oo(g1, g2, t)
    4*meijerg(((0, 1/2), ()), ((m/2,), (-m/2,)), s**(-2))/s**2
    """
    # See: [L, section 5.6.2, equation (1)]
    eta, _ = _get_coeff_exp(g1.argument, x)
    omega, _ = _get_coeff_exp(g2.argument, x)

    def neg(l):
        return [-x for x in l]
    a1 = neg(g1.bm) + list(g2.an)
    a2 = list(g2.aother) + neg(g1.bother)
    b1 = neg(g1.an) + list(g2.bm)
    b2 = list(g2.bother) + neg(g1.aother)
    return meijerg(a1, a2, b1, b2, omega/eta)/eta


def _rewrite_inversion(fac, po, g, x):
    """ Absorb ``po`` == x**s into g. """
    _, s = _get_coeff_exp(po, x)
    a, b = _get_coeff_exp(g.argument, x)

    def tr(l):
        return [t + s/b for t in l]
    from sympy.simplify import powdenest
    return (powdenest(fac/a**(s/b), polar=True),
            meijerg(tr(g.an), tr(g.aother), tr(g.bm), tr(g.bother), g.argument))


def _check_antecedents_inversion(g, x):
    """ Check antecedents for the laplace inversion integral. """
    _debug('Checking antecedents for inversion:')
    z = g.argument
    _, e = _get_coeff_exp(z, x)
    if e < 0:
        _debug('  Flipping G.')
        # We want to assume that argument gets large as |x| -> oo
        return _check_antecedents_inversion(_flip_g(g), x)

    def statement_half(a, b, c, z, plus):
        coeff, exponent = _get_coeff_exp(z, x)
        a *= exponent
        b *= coeff**c
        c *= exponent
        conds = []
        wp = b*exp(S.ImaginaryUnit*re(c)*pi/2)
        wm = b*exp(-S.ImaginaryUnit*re(c)*pi/2)
        if plus:
            w = wp
        else:
            w = wm
        conds += [And(Or(Eq(b, 0), re(c) <= 0), re(a) <= -1)]
        conds += [And(Ne(b, 0), Eq(im(c), 0), re(c) > 0, re(w) < 0)]
        conds += [And(Ne(b, 0), Eq(im(c), 0), re(c) > 0, re(w) <= 0,
                      re(a) <= -1)]
        return Or(*conds)

    def statement(a, b, c, z):
        """ Provide a convergence statement for z**a * exp(b*z**c),
             c/f sphinx docs. """
        return And(statement_half(a, b, c, z, True),
                   statement_half(a, b, c, z, False))

    # Notations from [L], section 5.7-10
    m, n, p, q = S([len(g.bm), len(g.an), len(g.ap), len(g.bq)])
    tau = m + n - p
    nu = q - m - n
    rho = (tau - nu)/2
    sigma = q - p
    if sigma == 1:
        epsilon = S.Half
    elif sigma > 1:
        epsilon = 1
    else:
        epsilon = S.NaN
    theta = ((1 - sigma)/2 + Add(*g.bq) - Add(*g.ap))/sigma
    delta = g.delta
    _debugf('  m=%s, n=%s, p=%s, q=%s, tau=%s, nu=%s, rho=%s, sigma=%s',
            (m, n, p, q, tau, nu, rho, sigma))
    _debugf('  epsilon=%s, theta=%s, delta=%s', (epsilon, theta, delta))

    # First check if the computation is valid.
    if not (g.delta >= e/2 or (p >= 1 and p >= q)):
        _debug('  Computation not valid for these parameters.')
        return False

    # Now check if the inversion integral exists.

    # Test "condition A"
    for a, b in itertools.product(g.an, g.bm):
        if (a - b).is_integer and a > b:
            _debug('  Not a valid G function.')
            return False

    # There are two cases. If p >= q, we can directly use a slater expansion
    # like [L], 5.2 (11). Note in particular that the asymptotics of such an
    # expansion even hold when some of the parameters differ by integers, i.e.
    # the formula itself would not be valid! (b/c G functions are cts. in their
    # parameters)
    # When p < q, we need to use the theorems of [L], 5.10.

    if p >= q:
        _debug('  Using asymptotic Slater expansion.')
        return And(*[statement(a - 1, 0, 0, z) for a in g.an])

    def E(z):
        return And(*[statement(a - 1, 0, 0, z) for a in g.an])

    def H(z):
        return statement(theta, -sigma, 1/sigma, z)

    def Hp(z):
        return statement_half(theta, -sigma, 1/sigma, z, True)

    def Hm(z):
        return statement_half(theta, -sigma, 1/sigma, z, False)

    # [L], section 5.10
    conds = []
    # Theorem 1 -- p < q from test above
    conds += [And(1 <= n, 1 <= m, rho*pi - delta >= pi/2, delta > 0,
                  E(z*exp(S.ImaginaryUnit*pi*(nu + 1))))]
    # Theorem 2, statements (2) and (3)
    conds += [And(p + 1 <= m, m + 1 <= q, delta > 0, delta < pi/2, n == 0,
                  (m - p + 1)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*(q - m))),
                  Hm(z*exp(-S.ImaginaryUnit*pi*(q - m))))]
    # Theorem 2, statement (5)  -- p < q from test above
    conds += [And(m == q, n == 0, delta > 0,
                  (sigma + epsilon)*pi - delta >= pi/2, H(z))]
    # Theorem 3, statements (6) and (7)
    conds += [And(Or(And(p <= q - 2, 1 <= tau, tau <= sigma/2),
                     And(p + 1 <= m + n, m + n <= (p + q)/2)),
                  delta > 0, delta < pi/2, (tau + 1)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*nu)),
                  Hm(z*exp(-S.ImaginaryUnit*pi*nu)))]
    # Theorem 4, statements (10) and (11)  -- p < q from test above
    conds += [And(1 <= m, rho > 0, delta > 0, delta + rho*pi < pi/2,
                  (tau + epsilon)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*nu)),
                  Hm(z*exp(-S.ImaginaryUnit*pi*nu)))]
    # Trivial case
    conds += [m == 0]

    # TODO
    # Theorem 5 is quite general
    # Theorem 6 contains special cases for q=p+1

    return Or(*conds)


def _int_inversion(g, x, t):
    """
    Compute the laplace inversion integral, assuming the formula applies.
    """
    b, a = _get_coeff_exp(g.argument, x)
    C, g = _inflate_fox_h(meijerg(g.an, g.aother, g.bm, g.bother, b/t**a), -a)
    return C/t*g


####################################################################
# Finally, the real meat.
####################################################################

_lookup_table = None


@cacheit
@timeit
def _rewrite_single(f, x, recursive=True):
    """
    Try to rewrite f as a sum of single G functions of the form
    C*x**s*G(a*x**b), where b is a rational number and C is independent of x.
    We guarantee that result.argument.as_coeff_mul(x) returns (a, (x**b,))
    or (a, ()).
    Returns a list of tuples (C, s, G) and a condition cond.
    Returns None on failure.
    """
    from .transforms import (mellin_transform, inverse_mellin_transform,
        IntegralTransformError, MellinTransformStripError)

    global _lookup_table
    if not _lookup_table:
        _lookup_table = {}
        _create_lookup_table(_lookup_table)

    if isinstance(f, meijerg):
        coeff, m = factor(f.argument, x).as_coeff_mul(x)
        if len(m) > 1:
            return None
        m = m[0]
        if m.is_Pow:
            if m.base != x or not m.exp.is_Rational:
                return None
        elif m != x:
            return None
        return [(1, 0, meijerg(f.an, f.aother, f.bm, f.bother, coeff*m))], True

    f_ = f
    f = f.subs(x, z)
    t = _mytype(f, z)
    if t in _lookup_table:
        l = _lookup_table[t]
        for formula, terms, cond, hint in l:
            subs = f.match(formula, old=True)
            if subs:
                subs_ = {}
                for fro, to in subs.items():
                    subs_[fro] = unpolarify(polarify(to, lift=True),
                                            exponents_only=True)
                subs = subs_
                if not isinstance(hint, bool):
                    hint = hint.subs(subs)
                if hint == False:
                    continue
                if not isinstance(cond, (bool, BooleanAtom)):
                    cond = unpolarify(cond.subs(subs))
                if _eval_cond(cond) == False:
                    continue
                if not isinstance(terms, list):
                    terms = terms(subs)
                res = []
                for fac, g in terms:
                    r1 = _get_coeff_exp(unpolarify(fac.subs(subs).subs(z, x),
                                                   exponents_only=True), x)
                    try:
                        g = g.subs(subs).subs(z, x)
                    except ValueError:
                        continue
                    # NOTE these substitutions can in principle introduce oo,
                    #      zoo and other absurdities. It shouldn't matter,
                    #      but better be safe.
                    if Tuple(*(r1 + (g,))).has(S.Infinity, S.ComplexInfinity, S.NegativeInfinity):
                        continue
                    g = meijerg(g.an, g.aother, g.bm, g.bother,
                                unpolarify(g.argument, exponents_only=True))
                    res.append(r1 + (g,))
                if res:
                    return res, cond

    # try recursive mellin transform
    if not recursive:
        return None
    _debug('Trying recursive Mellin transform method.')

    def my_imt(F, s, x, strip):
        """ Calling simplify() all the time is slow and not helpful, since
            most of the time it only factors things in a way that has to be
            un-done anyway. But sometimes it can remove apparent poles. """
        # XXX should this be in inverse_mellin_transform?
        try:
            return inverse_mellin_transform(F, s, x, strip,
                                            as_meijerg=True, needeval=True)
        except MellinTransformStripError:
            from sympy.simplify import simplify
            return inverse_mellin_transform(
                simplify(cancel(expand(F))), s, x, strip,
                as_meijerg=True, needeval=True)
    f = f_
    s = _dummy('s', 'rewrite-single', f)
    # to avoid infinite recursion, we have to force the two g functions case

    def my_integrator(f, x):
        r = _meijerint_definite_4(f, x, only_double=True)
        if r is not None:
            from sympy.simplify import hyperexpand
            res, cond = r
            res = _my_unpolarify(hyperexpand(res, rewrite='nonrepsmall'))
            return Piecewise((res, cond),
                             (Integral(f, (x, S.Zero, S.Infinity)), True))
        return Integral(f, (x, S.Zero, S.Infinity))
    try:
        F, strip, _ = mellin_transform(f, x, s, integrator=my_integrator,
                                       simplify=False, needeval=True)
        g = my_imt(F, s, x, strip)
    except IntegralTransformError:
        g = None
    if g is None:
        # We try to find an expression by analytic continuation.
        # (also if the dummy is already in the expression, there is no point in
        #  putting in another one)
        a = _dummy_('a', 'rewrite-single')
        if a not in f.free_symbols and _is_analytic(f, x):
            try:
                F, strip, _ = mellin_transform(f.subs(x, a*x), x, s,
                                               integrator=my_integrator,
                                               needeval=True, simplify=False)
                g = my_imt(F, s, x, strip).subs(a, 1)
            except IntegralTransformError:
                g = None
    if g is None or g.has(S.Infinity, S.NaN, S.ComplexInfinity):
        _debug('Recursive Mellin transform failed.')
        return None
    args = Add.make_args(g)
    res = []
    for f in args:
        c, m = f.as_coeff_mul(x)
        if len(m) > 1:
            raise NotImplementedError('Unexpected form...')
        g = m[0]
        a, b = _get_coeff_exp(g.argument, x)
        res += [(c, 0, meijerg(g.an, g.aother, g.bm, g.bother,
                               unpolarify(polarify(
                                   a, lift=True), exponents_only=True)
                               *x**b))]
    _debug('Recursive Mellin transform worked:', g)
    return res, True


def _rewrite1(f, x, recursive=True):
    """
    Try to rewrite ``f`` using a (sum of) single G functions with argument a*x**b.
    Return fac, po, g such that f = fac*po*g, fac is independent of ``x``.
    and po = x**s.
    Here g is a result from _rewrite_single.
    Return None on failure.
    """
    fac, po, g = _split_mul(f, x)
    g = _rewrite_single(g, x, recursive)
    if g:
        return fac, po, g[0], g[1]


def _rewrite2(f, x):
    """
    Try to rewrite ``f`` as a product of two G functions of arguments a*x**b.
    Return fac, po, g1, g2 such that f = fac*po*g1*g2, where fac is
    independent of x and po is x**s.
    Here g1 and g2 are results of _rewrite_single.
    Returns None on failure.
    """
    fac, po, g = _split_mul(f, x)
    if any(_rewrite_single(expr, x, False) is None for expr in _mul_args(g)):
        return None
    l = _mul_as_two_parts(g)
    if not l:
        return None
    l = list(ordered(l, [
        lambda p: max(len(_exponents(p[0], x)), len(_exponents(p[1], x))),
        lambda p: max(len(_functions(p[0], x)), len(_functions(p[1], x))),
        lambda p: max(len(_find_splitting_points(p[0], x)),
                      len(_find_splitting_points(p[1], x)))]))

    for recursive, (fac1, fac2) in itertools.product((False, True), l):
        g1 = _rewrite_single(fac1, x, recursive)
        g2 = _rewrite_single(fac2, x, recursive)
        if g1 and g2:
            cond = And(g1[1], g2[1])
            if cond != False:
                return fac, po, g1[0], g2[0], cond


def meijerint_indefinite(f, x):
    """
    Compute an indefinite integral of ``f`` by rewriting it as a G function.

    Examples
    ========

    >>> from sympy.integrals.meijerint import meijerint_indefinite
    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> meijerint_indefinite(sin(x), x)
    -cos(x)
    """
    f = sympify(f)
    results = []
    for a in sorted(_find_splitting_points(f, x) | {S.Zero}, key=default_sort_key):
        res = _meijerint_indefinite_1(f.subs(x, x + a), x)
        if not res:
            continue
        res = res.subs(x, x - a)
        if _has(res, hyper, meijerg):
            results.append(res)
        else:
            return res
    if f.has(HyperbolicFunction):
        _debug('Try rewriting hyperbolics in terms of exp.')
        rv = meijerint_indefinite(
            _rewrite_hyperbolics_as_exp(f), x)
        if rv:
            if not isinstance(rv, list):
                from sympy.simplify.radsimp import collect
                return collect(factor_terms(rv), rv.atoms(exp))
            results.extend(rv)
    if results:
        return next(ordered(results))


def _meijerint_indefinite_1(f, x):
    """ Helper that does not attempt any substitution. """
    _debug('Trying to compute the indefinite integral of', f, 'wrt', x)
    from sympy.simplify import hyperexpand, powdenest

    gs = _rewrite1(f, x)
    if gs is None:
        # Note: the code that calls us will do expand() and try again
        return None

    fac, po, gl, cond = gs
    _debug(' could rewrite:', gs)
    res = S.Zero
    for C, s, g in gl:
        a, b = _get_coeff_exp(g.argument, x)
        _, c = _get_coeff_exp(po, x)
        c += s

        # we do a substitution t=a*x**b, get integrand fac*t**rho*g
        fac_ = fac * C * x**(1 + c) / b
        rho = (c + 1)/b

        # we now use t**rho*G(params, t) = G(params + rho, t)
        # [L, page 150, equation (4)]
        # and integral G(params, t) dt = G(1, params+1, 0, t)
        #   (or a similar expression with 1 and 0 exchanged ... pick the one
        #    which yields a well-defined function)
        # [R, section 5]
        # (Note that this dummy will immediately go away again, so we
        #  can safely pass S.One for ``expr``.)
        t = _dummy('t', 'meijerint-indefinite', S.One)

        def tr(p):
            return [a + rho for a in p]
        if any(b.is_integer and (b <= 0) == True for b in tr(g.bm)):
            r = -meijerg(
                list(g.an), list(g.aother) + [1-rho], list(g.bm) + [-rho], list(g.bother), t)
        else:
            r = meijerg(
                list(g.an) + [1-rho], list(g.aother), list(g.bm), list(g.bother) + [-rho], t)
        # The antiderivative is most often expected to be defined
        # in the neighborhood of  x = 0.
        if b.is_extended_nonnegative and not f.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            place = 0  # Assume we can expand at zero
        else:
            place = None
        r = hyperexpand(r.subs(t, a*x**b), place=place)

        # now substitute back
        # Note: we really do want the powers of x to combine.
        res += powdenest(fac_*r, polar=True)

    def _clean(res):
        """This multiplies out superfluous powers of x we created, and chops off
        constants:

            >> _clean(x*(exp(x)/x - 1/x) + 3)
            exp(x)

        cancel is used before mul_expand since it is possible for an
        expression to have an additive constant that does not become isolated
        with simple expansion. Such a situation was identified in issue 6369:

        Examples
        ========

        >>> from sympy import sqrt, cancel
        >>> from sympy.abc import x
        >>> a = sqrt(2*x + 1)
        >>> bad = (3*x*a**5 + 2*x - a**5 + 1)/a**2
        >>> bad.expand().as_independent(x)[0]
        0
        >>> cancel(bad).expand().as_independent(x)[0]
        1
        """
        res = expand_mul(cancel(res), deep=False)
        return Add._from_args(res.as_coeff_add(x)[1])

    res = piecewise_fold(res, evaluate=None)
    if res.is_Piecewise:
        newargs = []
        for e, c in res.args:
            e = _my_unpolarify(_clean(e))
            newargs += [(e, c)]
        res = Piecewise(*newargs, evaluate=False)
    else:
        res = _my_unpolarify(_clean(res))
    return Piecewise((res, _my_unpolarify(cond)), (Integral(f, x), True))


@timeit
def meijerint_definite(f, x, a, b):
    """
    Integrate ``f`` over the interval [``a``, ``b``], by rewriting it as a product
    of two G functions, or as a single G function.

    Return res, cond, where cond are convergence conditions.

    Examples
    ========

    >>> from sympy.integrals.meijerint import meijerint_definite
    >>> from sympy import exp, oo
    >>> from sympy.abc import x
    >>> meijerint_definite(exp(-x**2), x, -oo, oo)
    (sqrt(pi), True)

    This function is implemented as a succession of functions
    meijerint_definite, _meijerint_definite_2, _meijerint_definite_3,
    _meijerint_definite_4. Each function in the list calls the next one
    (presumably) several times. This means that calling meijerint_definite
    can be very costly.
    """
    # This consists of three steps:
    # 1) Change the integration limits to 0, oo
    # 2) Rewrite in terms of G functions
    # 3) Evaluate the integral
    #
    # There are usually several ways of doing this, and we want to try all.
    # This function does (1), calls _meijerint_definite_2 for step (2).
    _debugf('Integrating %s wrt %s from %s to %s.', (f, x, a, b))
    f = sympify(f)
    if f.has(DiracDelta):
        _debug('Integrand has DiracDelta terms - giving up.')
        return None

    if f.has(SingularityFunction):
        _debug('Integrand has Singularity Function terms - giving up.')
        return None

    f_, x_, a_, b_ = f, x, a, b

    # Let's use a dummy in case any of the boundaries has x.
    d = Dummy('x')
    f = f.subs(x, d)
    x = d

    if a == b:
        return (S.Zero, True)

    results = []
    if a is S.NegativeInfinity and b is not S.Infinity:
        return meijerint_definite(f.subs(x, -x), x, -b, -a)

    elif a is S.NegativeInfinity:
        # Integrating -oo to oo. We need to find a place to split the integral.
        _debug('  Integrating -oo to +oo.')
        innermost = _find_splitting_points(f, x)
        _debug('  Sensible splitting points:', innermost)
        for c in sorted(innermost, key=default_sort_key, reverse=True) + [S.Zero]:
            _debug('  Trying to split at', c)
            if not c.is_extended_real:
                _debug('  Non-real splitting point.')
                continue
            res1 = _meijerint_definite_2(f.subs(x, x + c), x)
            if res1 is None:
                _debug('  But could not compute first integral.')
                continue
            res2 = _meijerint_definite_2(f.subs(x, c - x), x)
            if res2 is None:
                _debug('  But could not compute second integral.')
                continue
            res1, cond1 = res1
            res2, cond2 = res2
            cond = _condsimp(And(cond1, cond2))
            if cond == False:
                _debug('  But combined condition is always false.')
                continue
            res = res1 + res2
            return res, cond

    elif a is S.Infinity:
        res = meijerint_definite(f, x, b, S.Infinity)
        return -res[0], res[1]

    elif (a, b) == (S.Zero, S.Infinity):
        # This is a common case - try it directly first.
        res = _meijerint_definite_2(f, x)
        if res:
            if _has(res[0], meijerg):
                results.append(res)
            else:
                return res

    else:
        if b is S.Infinity:
            for split in _find_splitting_points(f, x):
                if (a - split >= 0) == True:
                    _debugf('Trying x -> x + %s', split)
                    res = _meijerint_definite_2(f.subs(x, x + split)
                                                *Heaviside(x + split - a), x)
                    if res:
                        if _has(res[0], meijerg):
                            results.append(res)
                        else:
                            return res

        f = f.subs(x, x + a)
        b = b - a
        a = 0
        if b is not S.Infinity:
            phi = exp(S.ImaginaryUnit*arg(b))
            b = Abs(b)
            f = f.subs(x, phi*x)
            f *= Heaviside(b - x)*phi
            b = S.Infinity

        _debug('Changed limits to', a, b)
        _debug('Changed function to', f)
        res = _meijerint_definite_2(f, x)
        if res:
            if _has(res[0], meijerg):
                results.append(res)
            else:
                return res
    if f_.has(HyperbolicFunction):
        _debug('Try rewriting hyperbolics in terms of exp.')
        rv = meijerint_definite(
            _rewrite_hyperbolics_as_exp(f_), x_, a_, b_)
        if rv:
            if not isinstance(rv, list):
                from sympy.simplify.radsimp import collect
                rv = (collect(factor_terms(rv[0]), rv[0].atoms(exp)),) + rv[1:]
                return rv
            results.extend(rv)
    if results:
        return next(ordered(results))


def _guess_expansion(f, x):
    """ Try to guess sensible rewritings for integrand f(x). """
    res = [(f, 'original integrand')]

    orig = res[-1][0]
    saw = {orig}
    expanded = expand_mul(orig)
    if expanded not in saw:
        res += [(expanded, 'expand_mul')]
        saw.add(expanded)

    expanded = expand(orig)
    if expanded not in saw:
        res += [(expanded, 'expand')]
        saw.add(expanded)

    if orig.has(TrigonometricFunction, HyperbolicFunction):
        expanded = expand_mul(expand_trig(orig))
        if expanded not in saw:
            res += [(expanded, 'expand_trig, expand_mul')]
            saw.add(expanded)

    if orig.has(cos, sin):
        from sympy.simplify.fu import sincos_to_sum
        reduced = sincos_to_sum(orig)
        if reduced not in saw:
            res += [(reduced, 'trig power reduction')]
            saw.add(reduced)

    return res


def _meijerint_definite_2(f, x):
    """
    Try to integrate f dx from zero to infinity.

    The body of this function computes various 'simplifications'
    f1, f2, ... of f (e.g. by calling expand_mul(), trigexpand()
    - see _guess_expansion) and calls _meijerint_definite_3 with each of
    these in succession.
    If _meijerint_definite_3 succeeds with any of the simplified functions,
    returns this result.
    """
    # This function does preparation for (2), calls
    # _meijerint_definite_3 for (2) and (3) combined.

    # use a positive dummy - we integrate from 0 to oo
    # XXX if a nonnegative symbol is used there will be test failures
    dummy = _dummy('x', 'meijerint-definite2', f, positive=True)
    f = f.subs(x, dummy)
    x = dummy

    if f == 0:
        return S.Zero, True

    for g, explanation in _guess_expansion(f, x):
        _debug('Trying', explanation)
        res = _meijerint_definite_3(g, x)
        if res:
            return res


def _meijerint_definite_3(f, x):
    """
    Try to integrate f dx from zero to infinity.

    This function calls _meijerint_definite_4 to try to compute the
    integral. If this fails, it tries using linearity.
    """
    res = _meijerint_definite_4(f, x)
    if res and res[1] != False:
        return res
    if f.is_Add:
        _debug('Expanding and evaluating all terms.')
        ress = [_meijerint_definite_4(g, x) for g in f.args]
        if all(r is not None for r in ress):
            conds = []
            res = S.Zero
            for r, c in ress:
                res += r
                conds += [c]
            c = And(*conds)
            if c != False:
                return res, c


def _my_unpolarify(f):
    return _eval_cond(unpolarify(f))


@timeit
def _meijerint_definite_4(f, x, only_double=False):
    """
    Try to integrate f dx from zero to infinity.

    Explanation
    ===========

    This function tries to apply the integration theorems found in literature,
    i.e. it tries to rewrite f as either one or a product of two G-functions.

    The parameter ``only_double`` is used internally in the recursive algorithm
    to disable trying to rewrite f as a single G-function.
    """
    from sympy.simplify import hyperexpand
    # This function does (2) and (3)
    _debug('Integrating', f)
    # Try single G function.
    if not only_double:
        gs = _rewrite1(f, x, recursive=False)
        if gs is not None:
            fac, po, g, cond = gs
            _debug('Could rewrite as single G function:', fac, po, g)
            res = S.Zero
            for C, s, f in g:
                if C == 0:
                    continue
                C, f = _rewrite_saxena_1(fac*C, po*x**s, f, x)
                res += C*_int0oo_1(f, x)
                cond = And(cond, _check_antecedents_1(f, x))
                if cond == False:
                    break
            cond = _my_unpolarify(cond)
            if cond == False:
                _debug('But cond is always False.')
            else:
                _debug('Result before branch substitutions is:', res)
                return _my_unpolarify(hyperexpand(res)), cond

    # Try two G functions.
    gs = _rewrite2(f, x)
    if gs is not None:
        for full_pb in [False, True]:
            fac, po, g1, g2, cond = gs
            _debug('Could rewrite as two G functions:', fac, po, g1, g2)
            res = S.Zero
            for C1, s1, f1 in g1:
                for C2, s2, f2 in g2:
                    r = _rewrite_saxena(fac*C1*C2, po*x**(s1 + s2),
                                        f1, f2, x, full_pb)
                    if r is None:
                        _debug('Non-rational exponents.')
                        return
                    C, f1_, f2_ = r
                    _debug('Saxena subst for yielded:', C, f1_, f2_)
                    cond = And(cond, _check_antecedents(f1_, f2_, x))
                    if cond == False:
                        break
                    res += C*_int0oo(f1_, f2_, x)
                else:
                    continue
                break
            cond = _my_unpolarify(cond)
            if cond == False:
                _debugf('But cond is always False (full_pb=%s).', full_pb)
            else:
                _debugf('Result before branch substitutions is: %s', (res, ))
                if only_double:
                    return res, cond
                return _my_unpolarify(hyperexpand(res)), cond


def meijerint_inversion(f, x, t):
    r"""
    Compute the inverse laplace transform
    $\int_{c+i\infty}^{c-i\infty} f(x) e^{tx}\, dx$,
    for real c larger than the real part of all singularities of ``f``.

    Note that ``t`` is always assumed real and positive.

    Return None if the integral does not exist or could not be evaluated.

    Examples
    ========

    >>> from sympy.abc import x, t
    >>> from sympy.integrals.meijerint import meijerint_inversion
    >>> meijerint_inversion(1/x, x, t)
    Heaviside(t)
    """
    f_ = f
    t_ = t
    t = Dummy('t', polar=True)  # We don't want sqrt(t**2) = abs(t) etc
    f = f.subs(t_, t)
    _debug('Laplace-inverting', f)
    if not _is_analytic(f, x):
        _debug('But expression is not analytic.')
        return None
    # Exponentials correspond to shifts; we filter them out and then
    # shift the result later.  If we are given an Add this will not
    # work, but the calling code will take care of that.
    shift = S.Zero

    if f.is_Mul:
        args = list(f.args)
    elif isinstance(f, exp):
        args = [f]
    else:
        args = None

    if args:
        newargs = []
        exponentials = []
        while args:
            arg = args.pop()
            if isinstance(arg, exp):
                arg2 = expand(arg)
                if arg2.is_Mul:
                    args += arg2.args
                    continue
                try:
                    a, b = _get_coeff_exp(arg.args[0], x)
                except _CoeffExpValueError:
                    b = 0
                if b == 1:
                    exponentials.append(a)
                else:
                    newargs.append(arg)
            elif arg.is_Pow:
                arg2 = expand(arg)
                if arg2.is_Mul:
                    args += arg2.args
                    continue
                if x not in arg.base.free_symbols:
                    try:
                        a, b = _get_coeff_exp(arg.exp, x)
                    except _CoeffExpValueError:
                        b = 0
                    if b == 1:
                        exponentials.append(a*log(arg.base))
                newargs.append(arg)
            else:
                newargs.append(arg)
        shift = Add(*exponentials)
        f = Mul(*newargs)

    if x not in f.free_symbols:
        _debug('Expression consists of constant and exp shift:', f, shift)
        cond = Eq(im(shift), 0)
        if cond == False:
            _debug('but shift is nonreal, cannot be a Laplace transform')
            return None
        res = f*DiracDelta(t + shift)
        _debug('Result is a delta function, possibly conditional:', res, cond)
        # cond is True or Eq
        return Piecewise((res.subs(t, t_), cond))

    gs = _rewrite1(f, x)
    if gs is not None:
        fac, po, g, cond = gs
        _debug('Could rewrite as single G function:', fac, po, g)
        res = S.Zero
        for C, s, f in g:
            C, f = _rewrite_inversion(fac*C, po*x**s, f, x)
            res += C*_int_inversion(f, x, t)
            cond = And(cond, _check_antecedents_inversion(f, x))
            if cond == False:
                break
        cond = _my_unpolarify(cond)
        if cond == False:
            _debug('But cond is always False.')
        else:
            _debug('Result before branch substitution:', res)
            from sympy.simplify import hyperexpand
            res = _my_unpolarify(hyperexpand(res))
            if not res.has(Heaviside):
                res *= Heaviside(t)
            res = res.subs(t, t + shift)
            if not isinstance(cond, bool):
                cond = cond.subs(t, t + shift)
            from .transforms import InverseLaplaceTransform
            return Piecewise((res.subs(t, t_), cond),
                             (InverseLaplaceTransform(f_.subs(t, t_), x, t_, None), True))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\rootisolation.py ===
"""Real and complex root isolation and refinement algorithms. """


from sympy.polys.densearith import (
    dup_neg, dup_rshift, dup_rem,
    dup_l2_norm_squared)
from sympy.polys.densebasic import (
    dup_LC, dup_TC, dup_degree,
    dup_strip, dup_reverse,
    dup_convert,
    dup_terms_gcd)
from sympy.polys.densetools import (
    dup_clear_denoms,
    dup_mirror, dup_scale, dup_shift,
    dup_transform,
    dup_diff,
    dup_eval, dmp_eval_in,
    dup_sign_variations,
    dup_real_imag)
from sympy.polys.euclidtools import (
    dup_discriminant)
from sympy.polys.factortools import (
    dup_factor_list)
from sympy.polys.polyerrors import (
    RefinementFailed,
    DomainError,
    PolynomialError)
from sympy.polys.sqfreetools import (
    dup_sqf_part, dup_sqf_list)


def dup_sturm(f, K):
    """
    Computes the Sturm sequence of ``f`` in ``F[x]``.

    Given a univariate, square-free polynomial ``f(x)`` returns the
    associated Sturm sequence ``f_0(x), ..., f_n(x)`` defined by::

       f_0(x), f_1(x) = f(x), f'(x)
       f_n = -rem(f_{n-2}(x), f_{n-1}(x))

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_sturm(x**3 - 2*x**2 + x - 3)
    [x**3 - 2*x**2 + x - 3, 3*x**2 - 4*x + 1, 2/9*x + 25/9, -2079/4]

    References
    ==========

    .. [1] [Davenport88]_

    """
    if not K.is_Field:
        raise DomainError("Cannot compute Sturm sequence over %s" % K)

    f = dup_sqf_part(f, K)

    sturm = [f, dup_diff(f, 1, K)]

    while sturm[-1]:
        s = dup_rem(sturm[-2], sturm[-1], K)
        sturm.append(dup_neg(s, K))

    return sturm[:-1]

def dup_root_upper_bound(f, K):
    """Compute the LMQ upper bound for the positive roots of `f`;
       LMQ (Local Max Quadratic) was developed by Akritas-Strzebonski-Vigklas.

    References
    ==========
    .. [1] Alkiviadis G. Akritas: "Linear and Quadratic Complexity Bounds on the
        Values of the Positive Roots of Polynomials"
        Journal of Universal Computer Science, Vol. 15, No. 3, 523-537, 2009.
    """
    n, P = len(f), []
    t = n * [K.one]
    if dup_LC(f, K) < 0:
        f = dup_neg(f, K)
    f = list(reversed(f))

    for i in range(0, n):
        if f[i] >= 0:
            continue

        a, QL = K.log(-f[i], 2), []

        for j in range(i + 1, n):

            if f[j] <= 0:
                continue

            q = t[j] + a - K.log(f[j], 2)
            QL.append([q // (j - i), j])

        if not QL:
            continue

        q = min(QL)

        t[q[1]] = t[q[1]] + 1

        P.append(q[0])

    if not P:
        return None
    else:
        return K.get_field()(2)**(max(P) + 1)

def dup_root_lower_bound(f, K):
    """Compute the LMQ lower bound for the positive roots of `f`;
       LMQ (Local Max Quadratic) was developed by Akritas-Strzebonski-Vigklas.

       References
       ==========
       .. [1] Alkiviadis G. Akritas: "Linear and Quadratic Complexity Bounds on the
              Values of the Positive Roots of Polynomials"
              Journal of Universal Computer Science, Vol. 15, No. 3, 523-537, 2009.
    """
    bound = dup_root_upper_bound(dup_reverse(f), K)

    if bound is not None:
        return 1/bound
    else:
        return None

def dup_cauchy_upper_bound(f, K):
    """
    Compute the Cauchy upper bound on the absolute value of all roots of f,
    real or complex.

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Geometrical_properties_of_polynomial_roots#Lagrange's_and_Cauchy's_bounds
    """
    n = dup_degree(f)
    if n < 1:
        raise PolynomialError('Polynomial has no roots.')

    if K.is_ZZ:
        L = K.get_field()
        f, K = dup_convert(f, K, L), L
    elif not K.is_QQ or K.is_RR or K.is_CC:
        # We need to compute absolute value, and we are not supporting cases
        # where this would take us outside the domain (or its quotient field).
        raise DomainError('Cauchy bound not supported over %s' % K)
    else:
        f = f[:]

    while K.is_zero(f[-1]):
        f.pop()
    if len(f) == 1:
        # Monomial. All roots are zero.
        return K.zero

    lc = f[0]
    return K.one + max(abs(n / lc) for n in f[1:])

def dup_cauchy_lower_bound(f, K):
    """Compute the Cauchy lower bound on the absolute value of all non-zero
       roots of f, real or complex."""
    g = dup_reverse(f)
    if len(g) < 2:
        raise PolynomialError('Polynomial has no non-zero roots.')
    if K.is_ZZ:
        K = K.get_field()
    b = dup_cauchy_upper_bound(g, K)
    return K.one / b

def dup_mignotte_sep_bound_squared(f, K):
    """
    Return the square of the Mignotte lower bound on separation between
    distinct roots of f. The square is returned so that the bound lies in
    K or its quotient field.

    References
    ==========

    .. [1] Mignotte, Maurice. "Some useful bounds." Computer algebra.
        Springer, Vienna, 1982. 259-263.
        https://people.dm.unipi.it/gianni/AC-EAG/Mignotte.pdf
    """
    n = dup_degree(f)
    if n < 2:
        raise PolynomialError('Polynomials of degree < 2 have no distinct roots.')

    if K.is_ZZ:
        L = K.get_field()
        f, K = dup_convert(f, K, L), L
    elif not K.is_QQ or K.is_RR or K.is_CC:
        # We need to compute absolute value, and we are not supporting cases
        # where this would take us outside the domain (or its quotient field).
        raise DomainError('Mignotte bound not supported over %s' % K)

    D = dup_discriminant(f, K)
    l2sq = dup_l2_norm_squared(f, K)
    return K(3)*K.abs(D) / ( K(n)**(n+1) * l2sq**(n-1) )

def _mobius_from_interval(I, field):
    """Convert an open interval to a Mobius transform. """
    s, t = I

    a, c = field.numer(s), field.denom(s)
    b, d = field.numer(t), field.denom(t)

    return a, b, c, d

def _mobius_to_interval(M, field):
    """Convert a Mobius transform to an open interval. """
    a, b, c, d = M

    s, t = field(a, c), field(b, d)

    if s <= t:
        return (s, t)
    else:
        return (t, s)

def dup_step_refine_real_root(f, M, K, fast=False):
    """One step of positive real root refinement algorithm. """
    a, b, c, d = M

    if a == b and c == d:
        return f, (a, b, c, d)

    A = dup_root_lower_bound(f, K)

    if A is not None:
        A = K(int(A))
    else:
        A = K.zero

    if fast and A > 16:
        f = dup_scale(f, A, K)
        a, c, A = A*a, A*c, K.one

    if A >= K.one:
        f = dup_shift(f, A, K)
        b, d = A*a + b, A*c + d

        if not dup_eval(f, K.zero, K):
            return f, (b, b, d, d)

    f, g = dup_shift(f, K.one, K), f

    a1, b1, c1, d1 = a, a + b, c, c + d

    if not dup_eval(f, K.zero, K):
        return f, (b1, b1, d1, d1)

    k = dup_sign_variations(f, K)

    if k == 1:
        a, b, c, d = a1, b1, c1, d1
    else:
        f = dup_shift(dup_reverse(g), K.one, K)

        if not dup_eval(f, K.zero, K):
            f = dup_rshift(f, 1, K)

        a, b, c, d = b, a + b, d, c + d

    return f, (a, b, c, d)

def dup_inner_refine_real_root(f, M, K, eps=None, steps=None, disjoint=None, fast=False, mobius=False):
    """Refine a positive root of `f` given a Mobius transform or an interval. """
    F = K.get_field()

    if len(M) == 2:
        a, b, c, d = _mobius_from_interval(M, F)
    else:
        a, b, c, d = M

    while not c:
        f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c,
            d), K, fast=fast)

    if eps is not None and steps is not None:
        for i in range(0, steps):
            if abs(F(a, c) - F(b, d)) >= eps:
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)
            else:
                break
    else:
        if eps is not None:
            while abs(F(a, c) - F(b, d)) >= eps:
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)

        if steps is not None:
            for i in range(0, steps):
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)

    if disjoint is not None:
        while True:
            u, v = _mobius_to_interval((a, b, c, d), F)

            if v <= disjoint or disjoint <= u:
                break
            else:
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)

    if not mobius:
        return _mobius_to_interval((a, b, c, d), F)
    else:
        return f, (a, b, c, d)

def dup_outer_refine_real_root(f, s, t, K, eps=None, steps=None, disjoint=None, fast=False):
    """Refine a positive root of `f` given an interval `(s, t)`. """
    a, b, c, d = _mobius_from_interval((s, t), K.get_field())

    f = dup_transform(f, dup_strip([a, b]),
                         dup_strip([c, d]), K)

    if dup_sign_variations(f, K) != 1:
        raise RefinementFailed("there should be exactly one root in (%s, %s) interval" % (s, t))

    return dup_inner_refine_real_root(f, (a, b, c, d), K, eps=eps, steps=steps, disjoint=disjoint, fast=fast)

def dup_refine_real_root(f, s, t, K, eps=None, steps=None, disjoint=None, fast=False):
    """Refine real root's approximating interval to the given precision. """
    if K.is_QQ:
        (_, f), K = dup_clear_denoms(f, K, convert=True), K.get_ring()
    elif not K.is_ZZ:
        raise DomainError("real root refinement not supported over %s" % K)

    if s == t:
        return (s, t)

    if s > t:
        s, t = t, s

    negative = False

    if s < 0:
        if t <= 0:
            f, s, t, negative = dup_mirror(f, K), -t, -s, True
        else:
            raise ValueError("Cannot refine a real root in (%s, %s)" % (s, t))

    if negative and disjoint is not None:
        if disjoint < 0:
            disjoint = -disjoint
        else:
            disjoint = None

    s, t = dup_outer_refine_real_root(
        f, s, t, K, eps=eps, steps=steps, disjoint=disjoint, fast=fast)

    if negative:
        return (-t, -s)
    else:
        return ( s, t)

def dup_inner_isolate_real_roots(f, K, eps=None, fast=False):
    """Internal function for isolation positive roots up to given precision.

       References
       ==========
           1. Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative Study of Two Real Root
           Isolation Methods . Nonlinear Analysis: Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
           2. Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S. Vigklas: Improving the
           Performance of the Continued Fractions Method Using new Bounds of Positive Roots. Nonlinear
           Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.
    """
    a, b, c, d = K.one, K.zero, K.zero, K.one

    k = dup_sign_variations(f, K)

    if k == 0:
        return []
    if k == 1:
        roots = [dup_inner_refine_real_root(
            f, (a, b, c, d), K, eps=eps, fast=fast, mobius=True)]
    else:
        roots, stack = [], [(a, b, c, d, f, k)]

        while stack:
            a, b, c, d, f, k = stack.pop()

            A = dup_root_lower_bound(f, K)

            if A is not None:
                A = K(int(A))
            else:
                A = K.zero

            if fast and A > 16:
                f = dup_scale(f, A, K)
                a, c, A = A*a, A*c, K.one

            if A >= K.one:
                f = dup_shift(f, A, K)
                b, d = A*a + b, A*c + d

                if not dup_TC(f, K):
                    roots.append((f, (b, b, d, d)))
                    f = dup_rshift(f, 1, K)

                k = dup_sign_variations(f, K)

                if k == 0:
                    continue
                if k == 1:
                    roots.append(dup_inner_refine_real_root(
                        f, (a, b, c, d), K, eps=eps, fast=fast, mobius=True))
                    continue

            f1 = dup_shift(f, K.one, K)

            a1, b1, c1, d1, r = a, a + b, c, c + d, 0

            if not dup_TC(f1, K):
                roots.append((f1, (b1, b1, d1, d1)))
                f1, r = dup_rshift(f1, 1, K), 1

            k1 = dup_sign_variations(f1, K)
            k2 = k - k1 - r

            a2, b2, c2, d2 = b, a + b, d, c + d

            if k2 > 1:
                f2 = dup_shift(dup_reverse(f), K.one, K)

                if not dup_TC(f2, K):
                    f2 = dup_rshift(f2, 1, K)

                k2 = dup_sign_variations(f2, K)
            else:
                f2 = None

            if k1 < k2:
                a1, a2, b1, b2 = a2, a1, b2, b1
                c1, c2, d1, d2 = c2, c1, d2, d1
                f1, f2, k1, k2 = f2, f1, k2, k1

            if not k1:
                continue

            if f1 is None:
                f1 = dup_shift(dup_reverse(f), K.one, K)

                if not dup_TC(f1, K):
                    f1 = dup_rshift(f1, 1, K)

            if k1 == 1:
                roots.append(dup_inner_refine_real_root(
                    f1, (a1, b1, c1, d1), K, eps=eps, fast=fast, mobius=True))
            else:
                stack.append((a1, b1, c1, d1, f1, k1))

            if not k2:
                continue

            if f2 is None:
                f2 = dup_shift(dup_reverse(f), K.one, K)

                if not dup_TC(f2, K):
                    f2 = dup_rshift(f2, 1, K)

            if k2 == 1:
                roots.append(dup_inner_refine_real_root(
                    f2, (a2, b2, c2, d2), K, eps=eps, fast=fast, mobius=True))
            else:
                stack.append((a2, b2, c2, d2, f2, k2))

    return roots

def _discard_if_outside_interval(f, M, inf, sup, K, negative, fast, mobius):
    """Discard an isolating interval if outside ``(inf, sup)``. """
    F = K.get_field()

    while True:
        u, v = _mobius_to_interval(M, F)

        if negative:
            u, v = -v, -u

        if (inf is None or u >= inf) and (sup is None or v <= sup):
            if not mobius:
                return u, v
            else:
                return f, M
        elif (sup is not None and u > sup) or (inf is not None and v < inf):
            return None
        else:
            f, M = dup_step_refine_real_root(f, M, K, fast=fast)

def dup_inner_isolate_positive_roots(f, K, eps=None, inf=None, sup=None, fast=False, mobius=False):
    """Iteratively compute disjoint positive root isolation intervals. """
    if sup is not None and sup < 0:
        return []

    roots = dup_inner_isolate_real_roots(f, K, eps=eps, fast=fast)

    F, results = K.get_field(), []

    if inf is not None or sup is not None:
        for f, M in roots:
            result = _discard_if_outside_interval(f, M, inf, sup, K, False, fast, mobius)

            if result is not None:
                results.append(result)
    elif not mobius:
        results.extend(_mobius_to_interval(M, F) for _, M in roots)
    else:
        results = roots

    return results

def dup_inner_isolate_negative_roots(f, K, inf=None, sup=None, eps=None, fast=False, mobius=False):
    """Iteratively compute disjoint negative root isolation intervals. """
    if inf is not None and inf >= 0:
        return []

    roots = dup_inner_isolate_real_roots(dup_mirror(f, K), K, eps=eps, fast=fast)

    F, results = K.get_field(), []

    if inf is not None or sup is not None:
        for f, M in roots:
            result = _discard_if_outside_interval(f, M, inf, sup, K, True, fast, mobius)

            if result is not None:
                results.append(result)
    elif not mobius:
        for f, M in roots:
            u, v = _mobius_to_interval(M, F)
            results.append((-v, -u))
    else:
        results = roots

    return results

def _isolate_zero(f, K, inf, sup, basis=False, sqf=False):
    """Handle special case of CF algorithm when ``f`` is homogeneous. """
    j, f = dup_terms_gcd(f, K)

    if j > 0:
        F = K.get_field()

        if (inf is None or inf <= 0) and (sup is None or 0 <= sup):
            if not sqf:
                if not basis:
                    return [((F.zero, F.zero), j)], f
                else:
                    return [((F.zero, F.zero), j, [K.one, K.zero])], f
            else:
                return [(F.zero, F.zero)], f

    return [], f

def dup_isolate_real_roots_sqf(f, K, eps=None, inf=None, sup=None, fast=False, blackbox=False):
    """Isolate real roots of a square-free polynomial using the Vincent-Akritas-Strzebonski (VAS) CF approach.

       References
       ==========
       .. [1] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative
              Study of Two Real Root Isolation Methods. Nonlinear Analysis:
              Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
       .. [2] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S.
              Vigklas: Improving the Performance of the Continued Fractions
              Method Using New Bounds of Positive Roots. Nonlinear Analysis:
              Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

    """
    if K.is_QQ:
        (_, f), K = dup_clear_denoms(f, K, convert=True), K.get_ring()
    elif not K.is_ZZ:
        raise DomainError("isolation of real roots not supported over %s" % K)

    if dup_degree(f) <= 0:
        return []

    I_zero, f = _isolate_zero(f, K, inf, sup, basis=False, sqf=True)

    I_neg = dup_inner_isolate_negative_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)
    I_pos = dup_inner_isolate_positive_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)

    roots = sorted(I_neg + I_zero + I_pos)

    if not blackbox:
        return roots
    else:
        return [ RealInterval((a, b), f, K) for (a, b) in roots ]

def dup_isolate_real_roots(f, K, eps=None, inf=None, sup=None, basis=False, fast=False):
    """Isolate real roots using Vincent-Akritas-Strzebonski (VAS) continued fractions approach.

       References
       ==========

       .. [1] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative
              Study of Two Real Root Isolation Methods. Nonlinear Analysis:
              Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
       .. [2] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S.
              Vigklas: Improving the Performance of the Continued Fractions
              Method Using New Bounds of Positive Roots.
              Nonlinear Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

    """
    if K.is_QQ:
        (_, f), K = dup_clear_denoms(f, K, convert=True), K.get_ring()
    elif not K.is_ZZ:
        raise DomainError("isolation of real roots not supported over %s" % K)

    if dup_degree(f) <= 0:
        return []

    I_zero, f = _isolate_zero(f, K, inf, sup, basis=basis, sqf=False)

    _, factors = dup_sqf_list(f, K)

    if len(factors) == 1:
        ((f, k),) = factors

        I_neg = dup_inner_isolate_negative_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)
        I_pos = dup_inner_isolate_positive_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)

        I_neg = [ ((u, v), k) for u, v in I_neg ]
        I_pos = [ ((u, v), k) for u, v in I_pos ]
    else:
        I_neg, I_pos = _real_isolate_and_disjoin(factors, K,
            eps=eps, inf=inf, sup=sup, basis=basis, fast=fast)

    return sorted(I_neg + I_zero + I_pos)

def dup_isolate_real_roots_list(polys, K, eps=None, inf=None, sup=None, strict=False, basis=False, fast=False):
    """Isolate real roots of a list of polynomial using Vincent-Akritas-Strzebonski (VAS) CF approach.

       References
       ==========

       .. [1] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative
              Study of Two Real Root Isolation Methods. Nonlinear Analysis:
              Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
       .. [2] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S.
              Vigklas: Improving the Performance of the Continued Fractions
              Method Using New Bounds of Positive Roots.
              Nonlinear Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

    """
    if K.is_QQ:
        K, F, polys = K.get_ring(), K, polys[:]

        for i, p in enumerate(polys):
            polys[i] = dup_clear_denoms(p, F, K, convert=True)[1]
    elif not K.is_ZZ:
        raise DomainError("isolation of real roots not supported over %s" % K)

    zeros, factors_dict = False, {}

    if (inf is None or inf <= 0) and (sup is None or 0 <= sup):
        zeros, zero_indices = True, {}

    for i, p in enumerate(polys):
        j, p = dup_terms_gcd(p, K)

        if zeros and j > 0:
            zero_indices[i] = j

        for f, k in dup_factor_list(p, K)[1]:
            f = tuple(f)

            if f not in factors_dict:
                factors_dict[f] = {i: k}
            else:
                factors_dict[f][i] = k

    factors_list = [(list(f), indices) for f, indices in factors_dict.items()]
    I_neg, I_pos = _real_isolate_and_disjoin(factors_list, K, eps=eps,
        inf=inf, sup=sup, strict=strict, basis=basis, fast=fast)

    F = K.get_field()

    if not zeros or not zero_indices:
        I_zero = []
    else:
        if not basis:
            I_zero = [((F.zero, F.zero), zero_indices)]
        else:
            I_zero = [((F.zero, F.zero), zero_indices, [K.one, K.zero])]

    return sorted(I_neg + I_zero + I_pos)

def _disjoint_p(M, N, strict=False):
    """Check if Mobius transforms define disjoint intervals. """
    a1, b1, c1, d1 = M
    a2, b2, c2, d2 = N

    a1d1, b1c1 = a1*d1, b1*c1
    a2d2, b2c2 = a2*d2, b2*c2

    if a1d1 == b1c1 and a2d2 == b2c2:
        return True

    if a1d1 > b1c1:
        a1, c1, b1, d1 = b1, d1, a1, c1

    if a2d2 > b2c2:
        a2, c2, b2, d2 = b2, d2, a2, c2

    if not strict:
        return a2*d1 >= c2*b1 or b2*c1 <= d2*a1
    else:
        return a2*d1 > c2*b1 or b2*c1 < d2*a1

def _real_isolate_and_disjoin(factors, K, eps=None, inf=None, sup=None, strict=False, basis=False, fast=False):
    """Isolate real roots of a list of polynomials and disjoin intervals. """
    I_pos, I_neg = [], []

    for i, (f, k) in enumerate(factors):
        for F, M in dup_inner_isolate_positive_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast, mobius=True):
            I_pos.append((F, M, k, f))

        for G, N in dup_inner_isolate_negative_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast, mobius=True):
            I_neg.append((G, N, k, f))

    for i, (f, M, k, F) in enumerate(I_pos):
        for j, (g, N, m, G) in enumerate(I_pos[i + 1:]):
            while not _disjoint_p(M, N, strict=strict):
                f, M = dup_inner_refine_real_root(f, M, K, steps=1, fast=fast, mobius=True)
                g, N = dup_inner_refine_real_root(g, N, K, steps=1, fast=fast, mobius=True)

            I_pos[i + j + 1] = (g, N, m, G)

        I_pos[i] = (f, M, k, F)

    for i, (f, M, k, F) in enumerate(I_neg):
        for j, (g, N, m, G) in enumerate(I_neg[i + 1:]):
            while not _disjoint_p(M, N, strict=strict):
                f, M = dup_inner_refine_real_root(f, M, K, steps=1, fast=fast, mobius=True)
                g, N = dup_inner_refine_real_root(g, N, K, steps=1, fast=fast, mobius=True)

            I_neg[i + j + 1] = (g, N, m, G)

        I_neg[i] = (f, M, k, F)

    if strict:
        for i, (f, M, k, F) in enumerate(I_neg):
            if not M[0]:
                while not M[0]:
                    f, M = dup_inner_refine_real_root(f, M, K, steps=1, fast=fast, mobius=True)

                I_neg[i] = (f, M, k, F)
                break

        for j, (g, N, m, G) in enumerate(I_pos):
            if not N[0]:
                while not N[0]:
                    g, N = dup_inner_refine_real_root(g, N, K, steps=1, fast=fast, mobius=True)

                I_pos[j] = (g, N, m, G)
                break

    field = K.get_field()

    I_neg = [ (_mobius_to_interval(M, field), k, f) for (_, M, k, f) in I_neg ]
    I_pos = [ (_mobius_to_interval(M, field), k, f) for (_, M, k, f) in I_pos ]

    I_neg = [((-v, -u), k, f) for ((u, v), k, f) in I_neg]

    if not basis:
        I_neg = [((u, v), k) for ((u, v), k, _) in I_neg]
        I_pos = [((u, v), k) for ((u, v), k, _) in I_pos]

    return I_neg, I_pos

def dup_count_real_roots(f, K, inf=None, sup=None):
    """Returns the number of distinct real roots of ``f`` in ``[inf, sup]``. """
    if dup_degree(f) <= 0:
        return 0

    if not K.is_Field:
        R, K = K, K.get_field()
        f = dup_convert(f, R, K)

    sturm = dup_sturm(f, K)

    if inf is None:
        signs_inf = dup_sign_variations([ dup_LC(s, K)*(-1)**dup_degree(s) for s in sturm ], K)
    else:
        signs_inf = dup_sign_variations([ dup_eval(s, inf, K) for s in sturm ], K)

    if sup is None:
        signs_sup = dup_sign_variations([ dup_LC(s, K) for s in sturm ], K)
    else:
        signs_sup = dup_sign_variations([ dup_eval(s, sup, K) for s in sturm ], K)

    count = abs(signs_inf - signs_sup)

    if inf is not None and not dup_eval(f, inf, K):
        count += 1

    return count

OO = 'OO'  # Origin of (re, im) coordinate system

Q1 = 'Q1'  # Quadrant #1 (++): re > 0 and im > 0
Q2 = 'Q2'  # Quadrant #2 (-+): re < 0 and im > 0
Q3 = 'Q3'  # Quadrant #3 (--): re < 0 and im < 0
Q4 = 'Q4'  # Quadrant #4 (+-): re > 0 and im < 0

A1 = 'A1'  # Axis #1 (+0): re > 0 and im = 0
A2 = 'A2'  # Axis #2 (0+): re = 0 and im > 0
A3 = 'A3'  # Axis #3 (-0): re < 0 and im = 0
A4 = 'A4'  # Axis #4 (0-): re = 0 and im < 0

_rules_simple = {
    # Q --> Q (same) => no change
    (Q1, Q1): 0,
    (Q2, Q2): 0,
    (Q3, Q3): 0,
    (Q4, Q4): 0,

    # A -- CCW --> Q => +1/4 (CCW)
    (A1, Q1): 1,
    (A2, Q2): 1,
    (A3, Q3): 1,
    (A4, Q4): 1,

    # A --  CW --> Q => -1/4 (CCW)
    (A1, Q4): 2,
    (A2, Q1): 2,
    (A3, Q2): 2,
    (A4, Q3): 2,

    # Q -- CCW --> A => +1/4 (CCW)
    (Q1, A2): 3,
    (Q2, A3): 3,
    (Q3, A4): 3,
    (Q4, A1): 3,

    # Q --  CW --> A => -1/4 (CCW)
    (Q1, A1): 4,
    (Q2, A2): 4,
    (Q3, A3): 4,
    (Q4, A4): 4,

    # Q -- CCW --> Q => +1/2 (CCW)
    (Q1, Q2): +5,
    (Q2, Q3): +5,
    (Q3, Q4): +5,
    (Q4, Q1): +5,

    # Q --  CW --> Q => -1/2 (CW)
    (Q1, Q4): -5,
    (Q2, Q1): -5,
    (Q3, Q2): -5,
    (Q4, Q3): -5,
}

_rules_ambiguous = {
    # A -- CCW --> Q => { +1/4 (CCW), -9/4 (CW) }
    (A1, OO, Q1): -1,
    (A2, OO, Q2): -1,
    (A3, OO, Q3): -1,
    (A4, OO, Q4): -1,

    # A --  CW --> Q => { -1/4 (CCW), +7/4 (CW) }
    (A1, OO, Q4): -2,
    (A2, OO, Q1): -2,
    (A3, OO, Q2): -2,
    (A4, OO, Q3): -2,

    # Q -- CCW --> A => { +1/4 (CCW), -9/4 (CW) }
    (Q1, OO, A2): -3,
    (Q2, OO, A3): -3,
    (Q3, OO, A4): -3,
    (Q4, OO, A1): -3,

    # Q --  CW --> A => { -1/4 (CCW), +7/4 (CW) }
    (Q1, OO, A1): -4,
    (Q2, OO, A2): -4,
    (Q3, OO, A3): -4,
    (Q4, OO, A4): -4,

    # A --  OO --> A => { +1 (CCW), -1 (CW) }
    (A1, A3): 7,
    (A2, A4): 7,
    (A3, A1): 7,
    (A4, A2): 7,

    (A1, OO, A3): 7,
    (A2, OO, A4): 7,
    (A3, OO, A1): 7,
    (A4, OO, A2): 7,

    # Q -- DIA --> Q => { +1 (CCW), -1 (CW) }
    (Q1, Q3): 8,
    (Q2, Q4): 8,
    (Q3, Q1): 8,
    (Q4, Q2): 8,

    (Q1, OO, Q3): 8,
    (Q2, OO, Q4): 8,
    (Q3, OO, Q1): 8,
    (Q4, OO, Q2): 8,

    # A --- R ---> A => { +1/2 (CCW), -3/2 (CW) }
    (A1, A2): 9,
    (A2, A3): 9,
    (A3, A4): 9,
    (A4, A1): 9,

    (A1, OO, A2): 9,
    (A2, OO, A3): 9,
    (A3, OO, A4): 9,
    (A4, OO, A1): 9,

    # A --- L ---> A => { +3/2 (CCW), -1/2 (CW) }
    (A1, A4): 10,
    (A2, A1): 10,
    (A3, A2): 10,
    (A4, A3): 10,

    (A1, OO, A4): 10,
    (A2, OO, A1): 10,
    (A3, OO, A2): 10,
    (A4, OO, A3): 10,

    # Q --- 1 ---> A => { +3/4 (CCW), -5/4 (CW) }
    (Q1, A3): 11,
    (Q2, A4): 11,
    (Q3, A1): 11,
    (Q4, A2): 11,

    (Q1, OO, A3): 11,
    (Q2, OO, A4): 11,
    (Q3, OO, A1): 11,
    (Q4, OO, A2): 11,

    # Q --- 2 ---> A => { +5/4 (CCW), -3/4 (CW) }
    (Q1, A4): 12,
    (Q2, A1): 12,
    (Q3, A2): 12,
    (Q4, A3): 12,

    (Q1, OO, A4): 12,
    (Q2, OO, A1): 12,
    (Q3, OO, A2): 12,
    (Q4, OO, A3): 12,

    # A --- 1 ---> Q => { +5/4 (CCW), -3/4 (CW) }
    (A1, Q3): 13,
    (A2, Q4): 13,
    (A3, Q1): 13,
    (A4, Q2): 13,

    (A1, OO, Q3): 13,
    (A2, OO, Q4): 13,
    (A3, OO, Q1): 13,
    (A4, OO, Q2): 13,

    # A --- 2 ---> Q => { +3/4 (CCW), -5/4 (CW) }
    (A1, Q2): 14,
    (A2, Q3): 14,
    (A3, Q4): 14,
    (A4, Q1): 14,

    (A1, OO, Q2): 14,
    (A2, OO, Q3): 14,
    (A3, OO, Q4): 14,
    (A4, OO, Q1): 14,

    # Q --> OO --> Q => { +1/2 (CCW), -3/2 (CW) }
    (Q1, OO, Q2): 15,
    (Q2, OO, Q3): 15,
    (Q3, OO, Q4): 15,
    (Q4, OO, Q1): 15,

    # Q --> OO --> Q => { +3/2 (CCW), -1/2 (CW) }
    (Q1, OO, Q4): 16,
    (Q2, OO, Q1): 16,
    (Q3, OO, Q2): 16,
    (Q4, OO, Q3): 16,

    # A --> OO --> A => { +2 (CCW), 0 (CW) }
    (A1, OO, A1): 17,
    (A2, OO, A2): 17,
    (A3, OO, A3): 17,
    (A4, OO, A4): 17,

    # Q --> OO --> Q => { +2 (CCW), 0 (CW) }
    (Q1, OO, Q1): 18,
    (Q2, OO, Q2): 18,
    (Q3, OO, Q3): 18,
    (Q4, OO, Q4): 18,
}

_values = {
    0: [( 0, 1)],
    1: [(+1, 4)],
    2: [(-1, 4)],
    3: [(+1, 4)],
    4: [(-1, 4)],
    -1: [(+9, 4), (+1, 4)],
    -2: [(+7, 4), (-1, 4)],
    -3: [(+9, 4), (+1, 4)],
    -4: [(+7, 4), (-1, 4)],
    +5: [(+1, 2)],
    -5: [(-1, 2)],
    7: [(+1, 1), (-1, 1)],
    8: [(+1, 1), (-1, 1)],
    9: [(+1, 2), (-3, 2)],
    10: [(+3, 2), (-1, 2)],
    11: [(+3, 4), (-5, 4)],
    12: [(+5, 4), (-3, 4)],
    13: [(+5, 4), (-3, 4)],
    14: [(+3, 4), (-5, 4)],
    15: [(+1, 2), (-3, 2)],
    16: [(+3, 2), (-1, 2)],
    17: [(+2, 1), ( 0, 1)],
    18: [(+2, 1), ( 0, 1)],
}

def _classify_point(re, im):
    """Return the half-axis (or origin) on which (re, im) point is located. """
    if not re and not im:
        return OO

    if not re:
        if im > 0:
            return A2
        else:
            return A4
    elif not im:
        if re > 0:
            return A1
        else:
            return A3

def _intervals_to_quadrants(intervals, f1, f2, s, t, F):
    """Generate a sequence of extended quadrants from a list of critical points. """
    if not intervals:
        return []

    Q = []

    if not f1:
        (a, b), _, _ = intervals[0]

        if a == b == s:
            if len(intervals) == 1:
                if dup_eval(f2, t, F) > 0:
                    return [OO, A2]
                else:
                    return [OO, A4]
            else:
                (a, _), _, _ = intervals[1]

                if dup_eval(f2, (s + a)/2, F) > 0:
                    Q.extend([OO, A2])
                    f2_sgn = +1
                else:
                    Q.extend([OO, A4])
                    f2_sgn = -1

                intervals = intervals[1:]
        else:
            if dup_eval(f2, s, F) > 0:
                Q.append(A2)
                f2_sgn = +1
            else:
                Q.append(A4)
                f2_sgn = -1

        for (a, _), indices, _ in intervals:
            Q.append(OO)

            if indices[1] % 2 == 1:
                f2_sgn = -f2_sgn

            if a != t:
                if f2_sgn > 0:
                    Q.append(A2)
                else:
                    Q.append(A4)

        return Q

    if not f2:
        (a, b), _, _ = intervals[0]

        if a == b == s:
            if len(intervals) == 1:
                if dup_eval(f1, t, F) > 0:
                    return [OO, A1]
                else:
                    return [OO, A3]
            else:
                (a, _), _, _ = intervals[1]

                if dup_eval(f1, (s + a)/2, F) > 0:
                    Q.extend([OO, A1])
                    f1_sgn = +1
                else:
                    Q.extend([OO, A3])
                    f1_sgn = -1

                intervals = intervals[1:]
        else:
            if dup_eval(f1, s, F) > 0:
                Q.append(A1)
                f1_sgn = +1
            else:
                Q.append(A3)
                f1_sgn = -1

        for (a, _), indices, _ in intervals:
            Q.append(OO)

            if indices[0] % 2 == 1:
                f1_sgn = -f1_sgn

            if a != t:
                if f1_sgn > 0:
                    Q.append(A1)
                else:
                    Q.append(A3)

        return Q

    re = dup_eval(f1, s, F)
    im = dup_eval(f2, s, F)

    if not re or not im:
        Q.append(_classify_point(re, im))

        if len(intervals) == 1:
            re = dup_eval(f1, t, F)
            im = dup_eval(f2, t, F)
        else:
            (a, _), _, _ = intervals[1]

            re = dup_eval(f1, (s + a)/2, F)
            im = dup_eval(f2, (s + a)/2, F)

        intervals = intervals[1:]

    if re > 0:
        f1_sgn = +1
    else:
        f1_sgn = -1

    if im > 0:
        f2_sgn = +1
    else:
        f2_sgn = -1

    sgn = {
        (+1, +1): Q1,
        (-1, +1): Q2,
        (-1, -1): Q3,
        (+1, -1): Q4,
    }

    Q.append(sgn[(f1_sgn, f2_sgn)])

    for (a, b), indices, _ in intervals:
        if a == b:
            re = dup_eval(f1, a, F)
            im = dup_eval(f2, a, F)

            cls = _classify_point(re, im)

            if cls is not None:
                Q.append(cls)

        if 0 in indices:
            if indices[0] % 2 == 1:
                f1_sgn = -f1_sgn

        if 1 in indices:
            if indices[1] % 2 == 1:
                f2_sgn = -f2_sgn

        if not (a == b and b == t):
            Q.append(sgn[(f1_sgn, f2_sgn)])

    return Q

def _traverse_quadrants(Q_L1, Q_L2, Q_L3, Q_L4, exclude=None):
    """Transform sequences of quadrants to a sequence of rules. """
    if exclude is True:
        edges = [1, 1, 0, 0]

        corners = {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 0,
            (3, 0): 1,
        }
    else:
        edges = [0, 0, 0, 0]

        corners = {
            (0, 1): 0,
            (1, 2): 0,
            (2, 3): 0,
            (3, 0): 0,
        }

    if exclude is not None and exclude is not True:
        exclude = set(exclude)

        for i, edge in enumerate(['S', 'E', 'N', 'W']):
            if edge in exclude:
                edges[i] = 1

        for i, corner in enumerate(['SW', 'SE', 'NE', 'NW']):
            if corner in exclude:
                corners[((i - 1) % 4, i)] = 1

    QQ, rules = [Q_L1, Q_L2, Q_L3, Q_L4], []

    for i, Q in enumerate(QQ):
        if not Q:
            continue

        if Q[-1] == OO:
            Q = Q[:-1]

        if Q[0] == OO:
            j, Q = (i - 1) % 4, Q[1:]
            qq = (QQ[j][-2], OO, Q[0])

            if qq in _rules_ambiguous:
                rules.append((_rules_ambiguous[qq], corners[(j, i)]))
            else:
                raise NotImplementedError("3 element rule (corner): " + str(qq))

        q1, k = Q[0], 1

        while k < len(Q):
            q2, k = Q[k], k + 1

            if q2 != OO:
                qq = (q1, q2)

                if qq in _rules_simple:
                    rules.append((_rules_simple[qq], 0))
                elif qq in _rules_ambiguous:
                    rules.append((_rules_ambiguous[qq], edges[i]))
                else:
                    raise NotImplementedError("2 element rule (inside): " + str(qq))
            else:
                qq, k = (q1, q2, Q[k]), k + 1

                if qq in _rules_ambiguous:
                    rules.append((_rules_ambiguous[qq], edges[i]))
                else:
                    raise NotImplementedError("3 element rule (edge): " + str(qq))

            q1 = qq[-1]

    return rules

def _reverse_intervals(intervals):
    """Reverse intervals for traversal from right to left and from top to bottom. """
    return [ ((b, a), indices, f) for (a, b), indices, f in reversed(intervals) ]

def _winding_number(T, field):
    """Compute the winding number of the input polynomial, i.e. the number of roots. """
    return int(sum(field(*_values[t][i]) for t, i in T) / field(2))

def dup_count_complex_roots(f, K, inf=None, sup=None, exclude=None):
    """Count all roots in [u + v*I, s + t*I] rectangle using Collins-Krandick algorithm. """
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("complex root counting is not supported over %s" % K)

    if K.is_ZZ:
        R, F = K, K.get_field()
    else:
        R, F = K.get_ring(), K

    f = dup_convert(f, K, F)

    if inf is None or sup is None:
        _, lc = dup_degree(f), abs(dup_LC(f, F))
        B = 2*max(F.quo(abs(c), lc) for c in f)

    if inf is None:
        (u, v) = (-B, -B)
    else:
        (u, v) = inf

    if sup is None:
        (s, t) = (+B, +B)
    else:
        (s, t) = sup

    f1, f2 = dup_real_imag(f, F)

    f1L1F = dmp_eval_in(f1, v, 1, 1, F)
    f2L1F = dmp_eval_in(f2, v, 1, 1, F)

    _, f1L1R = dup_clear_denoms(f1L1F, F, R, convert=True)
    _, f2L1R = dup_clear_denoms(f2L1F, F, R, convert=True)

    f1L2F = dmp_eval_in(f1, s, 0, 1, F)
    f2L2F = dmp_eval_in(f2, s, 0, 1, F)

    _, f1L2R = dup_clear_denoms(f1L2F, F, R, convert=True)
    _, f2L2R = dup_clear_denoms(f2L2F, F, R, convert=True)

    f1L3F = dmp_eval_in(f1, t, 1, 1, F)
    f2L3F = dmp_eval_in(f2, t, 1, 1, F)

    _, f1L3R = dup_clear_denoms(f1L3F, F, R, convert=True)
    _, f2L3R = dup_clear_denoms(f2L3F, F, R, convert=True)

    f1L4F = dmp_eval_in(f1, u, 0, 1, F)
    f2L4F = dmp_eval_in(f2, u, 0, 1, F)

    _, f1L4R = dup_clear_denoms(f1L4F, F, R, convert=True)
    _, f2L4R = dup_clear_denoms(f2L4F, F, R, convert=True)

    S_L1 = [f1L1R, f2L1R]
    S_L2 = [f1L2R, f2L2R]
    S_L3 = [f1L3R, f2L3R]
    S_L4 = [f1L4R, f2L4R]

    I_L1 = dup_isolate_real_roots_list(S_L1, R, inf=u, sup=s, fast=True, basis=True, strict=True)
    I_L2 = dup_isolate_real_roots_list(S_L2, R, inf=v, sup=t, fast=True, basis=True, strict=True)
    I_L3 = dup_isolate_real_roots_list(S_L3, R, inf=u, sup=s, fast=True, basis=True, strict=True)
    I_L4 = dup_isolate_real_roots_list(S_L4, R, inf=v, sup=t, fast=True, basis=True, strict=True)

    I_L3 = _reverse_intervals(I_L3)
    I_L4 = _reverse_intervals(I_L4)

    Q_L1 = _intervals_to_quadrants(I_L1, f1L1F, f2L1F, u, s, F)
    Q_L2 = _intervals_to_quadrants(I_L2, f1L2F, f2L2F, v, t, F)
    Q_L3 = _intervals_to_quadrants(I_L3, f1L3F, f2L3F, s, u, F)
    Q_L4 = _intervals_to_quadrants(I_L4, f1L4F, f2L4F, t, v, F)

    T = _traverse_quadrants(Q_L1, Q_L2, Q_L3, Q_L4, exclude=exclude)

    return _winding_number(T, F)

def _vertical_bisection(N, a, b, I, Q, F1, F2, f1, f2, F):
    """Vertical bisection step in Collins-Krandick root isolation algorithm. """
    (u, v), (s, t) = a, b

    I_L1, I_L2, I_L3, I_L4 = I
    Q_L1, Q_L2, Q_L3, Q_L4 = Q

    f1L1F, f1L2F, f1L3F, f1L4F = F1
    f2L1F, f2L2F, f2L3F, f2L4F = F2

    x = (u + s) / 2

    f1V = dmp_eval_in(f1, x, 0, 1, F)
    f2V = dmp_eval_in(f2, x, 0, 1, F)

    I_V = dup_isolate_real_roots_list([f1V, f2V], F, inf=v, sup=t, fast=True, strict=True, basis=True)

    I_L1_L, I_L1_R = [], []
    I_L2_L, I_L2_R = I_V, I_L2
    I_L3_L, I_L3_R = [], []
    I_L4_L, I_L4_R = I_L4, _reverse_intervals(I_V)

    for I in I_L1:
        (a, b), indices, h = I

        if a == b:
            if a == x:
                I_L1_L.append(I)
                I_L1_R.append(I)
            elif a < x:
                I_L1_L.append(I)
            else:
                I_L1_R.append(I)
        else:
            if b <= x:
                I_L1_L.append(I)
            elif a >= x:
                I_L1_R.append(I)
            else:
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=x, fast=True)

                if b <= x:
                    I_L1_L.append(((a, b), indices, h))
                if a >= x:
                    I_L1_R.append(((a, b), indices, h))

    for I in I_L3:
        (b, a), indices, h = I

        if a == b:
            if a == x:
                I_L3_L.append(I)
                I_L3_R.append(I)
            elif a < x:
                I_L3_L.append(I)
            else:
                I_L3_R.append(I)
        else:
            if b <= x:
                I_L3_L.append(I)
            elif a >= x:
                I_L3_R.append(I)
            else:
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=x, fast=True)

                if b <= x:
                    I_L3_L.append(((b, a), indices, h))
                if a >= x:
                    I_L3_R.append(((b, a), indices, h))

    Q_L1_L = _intervals_to_quadrants(I_L1_L, f1L1F, f2L1F, u, x, F)
    Q_L2_L = _intervals_to_quadrants(I_L2_L, f1V, f2V, v, t, F)
    Q_L3_L = _intervals_to_quadrants(I_L3_L, f1L3F, f2L3F, x, u, F)
    Q_L4_L = Q_L4

    Q_L1_R = _intervals_to_quadrants(I_L1_R, f1L1F, f2L1F, x, s, F)
    Q_L2_R = Q_L2
    Q_L3_R = _intervals_to_quadrants(I_L3_R, f1L3F, f2L3F, s, x, F)
    Q_L4_R = _intervals_to_quadrants(I_L4_R, f1V, f2V, t, v, F)

    T_L = _traverse_quadrants(Q_L1_L, Q_L2_L, Q_L3_L, Q_L4_L, exclude=True)
    T_R = _traverse_quadrants(Q_L1_R, Q_L2_R, Q_L3_R, Q_L4_R, exclude=True)

    N_L = _winding_number(T_L, F)
    N_R = _winding_number(T_R, F)

    I_L = (I_L1_L, I_L2_L, I_L3_L, I_L4_L)
    Q_L = (Q_L1_L, Q_L2_L, Q_L3_L, Q_L4_L)

    I_R = (I_L1_R, I_L2_R, I_L3_R, I_L4_R)
    Q_R = (Q_L1_R, Q_L2_R, Q_L3_R, Q_L4_R)

    F1_L = (f1L1F, f1V, f1L3F, f1L4F)
    F2_L = (f2L1F, f2V, f2L3F, f2L4F)

    F1_R = (f1L1F, f1L2F, f1L3F, f1V)
    F2_R = (f2L1F, f2L2F, f2L3F, f2V)

    a, b = (u, v), (x, t)
    c, d = (x, v), (s, t)

    D_L = (N_L, a, b, I_L, Q_L, F1_L, F2_L)
    D_R = (N_R, c, d, I_R, Q_R, F1_R, F2_R)

    return D_L, D_R

def _horizontal_bisection(N, a, b, I, Q, F1, F2, f1, f2, F):
    """Horizontal bisection step in Collins-Krandick root isolation algorithm. """
    (u, v), (s, t) = a, b

    I_L1, I_L2, I_L3, I_L4 = I
    Q_L1, Q_L2, Q_L3, Q_L4 = Q

    f1L1F, f1L2F, f1L3F, f1L4F = F1
    f2L1F, f2L2F, f2L3F, f2L4F = F2

    y = (v + t) / 2

    f1H = dmp_eval_in(f1, y, 1, 1, F)
    f2H = dmp_eval_in(f2, y, 1, 1, F)

    I_H = dup_isolate_real_roots_list([f1H, f2H], F, inf=u, sup=s, fast=True, strict=True, basis=True)

    I_L1_B, I_L1_U = I_L1, I_H
    I_L2_B, I_L2_U = [], []
    I_L3_B, I_L3_U = _reverse_intervals(I_H), I_L3
    I_L4_B, I_L4_U = [], []

    for I in I_L2:
        (a, b), indices, h = I

        if a == b:
            if a == y:
                I_L2_B.append(I)
                I_L2_U.append(I)
            elif a < y:
                I_L2_B.append(I)
            else:
                I_L2_U.append(I)
        else:
            if b <= y:
                I_L2_B.append(I)
            elif a >= y:
                I_L2_U.append(I)
            else:
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=y, fast=True)

                if b <= y:
                    I_L2_B.append(((a, b), indices, h))
                if a >= y:
                    I_L2_U.append(((a, b), indices, h))

    for I in I_L4:
        (b, a), indices, h = I

        if a == b:
            if a == y:
                I_L4_B.append(I)
                I_L4_U.append(I)
            elif a < y:
                I_L4_B.append(I)
            else:
                I_L4_U.append(I)
        else:
            if b <= y:
                I_L4_B.append(I)
            elif a >= y:
                I_L4_U.append(I)
            else:
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=y, fast=True)

                if b <= y:
                    I_L4_B.append(((b, a), indices, h))
                if a >= y:
                    I_L4_U.append(((b, a), indices, h))

    Q_L1_B = Q_L1
    Q_L2_B = _intervals_to_quadrants(I_L2_B, f1L2F, f2L2F, v, y, F)
    Q_L3_B = _intervals_to_quadrants(I_L3_B, f1H, f2H, s, u, F)
    Q_L4_B = _intervals_to_quadrants(I_L4_B, f1L4F, f2L4F, y, v, F)

    Q_L1_U = _intervals_to_quadrants(I_L1_U, f1H, f2H, u, s, F)
    Q_L2_U = _intervals_to_quadrants(I_L2_U, f1L2F, f2L2F, y, t, F)
    Q_L3_U = Q_L3
    Q_L4_U = _intervals_to_quadrants(I_L4_U, f1L4F, f2L4F, t, y, F)

    T_B = _traverse_quadrants(Q_L1_B, Q_L2_B, Q_L3_B, Q_L4_B, exclude=True)
    T_U = _traverse_quadrants(Q_L1_U, Q_L2_U, Q_L3_U, Q_L4_U, exclude=True)

    N_B = _winding_number(T_B, F)
    N_U = _winding_number(T_U, F)

    I_B = (I_L1_B, I_L2_B, I_L3_B, I_L4_B)
    Q_B = (Q_L1_B, Q_L2_B, Q_L3_B, Q_L4_B)

    I_U = (I_L1_U, I_L2_U, I_L3_U, I_L4_U)
    Q_U = (Q_L1_U, Q_L2_U, Q_L3_U, Q_L4_U)

    F1_B = (f1L1F, f1L2F, f1H, f1L4F)
    F2_B = (f2L1F, f2L2F, f2H, f2L4F)

    F1_U = (f1H, f1L2F, f1L3F, f1L4F)
    F2_U = (f2H, f2L2F, f2L3F, f2L4F)

    a, b = (u, v), (s, y)
    c, d = (u, y), (s, t)

    D_B = (N_B, a, b, I_B, Q_B, F1_B, F2_B)
    D_U = (N_U, c, d, I_U, Q_U, F1_U, F2_U)

    return D_B, D_U

def _depth_first_select(rectangles):
    """Find a rectangle of minimum area for bisection. """
    min_area, j = None, None

    for i, (_, (u, v), (s, t), _, _, _, _) in enumerate(rectangles):
        area = (s - u)*(t - v)

        if min_area is None or area < min_area:
            min_area, j = area, i

    return rectangles.pop(j)

def _rectangle_small_p(a, b, eps):
    """Return ``True`` if the given rectangle is small enough. """
    (u, v), (s, t) = a, b

    if eps is not None:
        return s - u < eps and t - v < eps
    else:
        return True

def dup_isolate_complex_roots_sqf(f, K, eps=None, inf=None, sup=None, blackbox=False):
    """Isolate complex roots of a square-free polynomial using Collins-Krandick algorithm. """
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("isolation of complex roots is not supported over %s" % K)

    if dup_degree(f) <= 0:
        return []

    if K.is_ZZ:
        F = K.get_field()
    else:
        F = K

    f = dup_convert(f, K, F)

    lc = abs(dup_LC(f, F))
    B = 2*max(F.quo(abs(c), lc) for c in f)

    (u, v), (s, t) = (-B, F.zero), (B, B)

    if inf is not None:
        u = inf

    if sup is not None:
        s = sup

    if v < 0 or t <= v or s <= u:
        raise ValueError("not a valid complex isolation rectangle")

    f1, f2 = dup_real_imag(f, F)

    f1L1 = dmp_eval_in(f1, v, 1, 1, F)
    f2L1 = dmp_eval_in(f2, v, 1, 1, F)

    f1L2 = dmp_eval_in(f1, s, 0, 1, F)
    f2L2 = dmp_eval_in(f2, s, 0, 1, F)

    f1L3 = dmp_eval_in(f1, t, 1, 1, F)
    f2L3 = dmp_eval_in(f2, t, 1, 1, F)

    f1L4 = dmp_eval_in(f1, u, 0, 1, F)
    f2L4 = dmp_eval_in(f2, u, 0, 1, F)

    S_L1 = [f1L1, f2L1]
    S_L2 = [f1L2, f2L2]
    S_L3 = [f1L3, f2L3]
    S_L4 = [f1L4, f2L4]

    I_L1 = dup_isolate_real_roots_list(S_L1, F, inf=u, sup=s, fast=True, strict=True, basis=True)
    I_L2 = dup_isolate_real_roots_list(S_L2, F, inf=v, sup=t, fast=True, strict=True, basis=True)
    I_L3 = dup_isolate_real_roots_list(S_L3, F, inf=u, sup=s, fast=True, strict=True, basis=True)
    I_L4 = dup_isolate_real_roots_list(S_L4, F, inf=v, sup=t, fast=True, strict=True, basis=True)

    I_L3 = _reverse_intervals(I_L3)
    I_L4 = _reverse_intervals(I_L4)

    Q_L1 = _intervals_to_quadrants(I_L1, f1L1, f2L1, u, s, F)
    Q_L2 = _intervals_to_quadrants(I_L2, f1L2, f2L2, v, t, F)
    Q_L3 = _intervals_to_quadrants(I_L3, f1L3, f2L3, s, u, F)
    Q_L4 = _intervals_to_quadrants(I_L4, f1L4, f2L4, t, v, F)

    T = _traverse_quadrants(Q_L1, Q_L2, Q_L3, Q_L4)
    N = _winding_number(T, F)

    if not N:
        return []

    I = (I_L1, I_L2, I_L3, I_L4)
    Q = (Q_L1, Q_L2, Q_L3, Q_L4)

    F1 = (f1L1, f1L2, f1L3, f1L4)
    F2 = (f2L1, f2L2, f2L3, f2L4)

    rectangles, roots = [(N, (u, v), (s, t), I, Q, F1, F2)], []

    while rectangles:
        N, (u, v), (s, t), I, Q, F1, F2 = _depth_first_select(rectangles)

        if s - u > t - v:
            D_L, D_R = _vertical_bisection(N, (u, v), (s, t), I, Q, F1, F2, f1, f2, F)

            N_L, a, b, I_L, Q_L, F1_L, F2_L = D_L
            N_R, c, d, I_R, Q_R, F1_R, F2_R = D_R

            if N_L >= 1:
                if N_L == 1 and _rectangle_small_p(a, b, eps):
                    roots.append(ComplexInterval(a, b, I_L, Q_L, F1_L, F2_L, f1, f2, F))
                else:
                    rectangles.append(D_L)

            if N_R >= 1:
                if N_R == 1 and _rectangle_small_p(c, d, eps):
                    roots.append(ComplexInterval(c, d, I_R, Q_R, F1_R, F2_R, f1, f2, F))
                else:
                    rectangles.append(D_R)
        else:
            D_B, D_U = _horizontal_bisection(N, (u, v), (s, t), I, Q, F1, F2, f1, f2, F)

            N_B, a, b, I_B, Q_B, F1_B, F2_B = D_B
            N_U, c, d, I_U, Q_U, F1_U, F2_U = D_U

            if N_B >= 1:
                if N_B == 1 and _rectangle_small_p(a, b, eps):
                    roots.append(ComplexInterval(
                        a, b, I_B, Q_B, F1_B, F2_B, f1, f2, F))
                else:
                    rectangles.append(D_B)

            if N_U >= 1:
                if N_U == 1 and _rectangle_small_p(c, d, eps):
                    roots.append(ComplexInterval(
                        c, d, I_U, Q_U, F1_U, F2_U, f1, f2, F))
                else:
                    rectangles.append(D_U)

    _roots, roots = sorted(roots, key=lambda r: (r.ax, r.ay)), []

    for root in _roots:
        roots.extend([root.conjugate(), root])

    if blackbox:
        return roots
    else:
        return [ r.as_tuple() for r in roots ]

def dup_isolate_all_roots_sqf(f, K, eps=None, inf=None, sup=None, fast=False, blackbox=False):
    """Isolate real and complex roots of a square-free polynomial ``f``. """
    return (
        dup_isolate_real_roots_sqf( f, K, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox),
        dup_isolate_complex_roots_sqf(f, K, eps=eps, inf=inf, sup=sup, blackbox=blackbox))

def dup_isolate_all_roots(f, K, eps=None, inf=None, sup=None, fast=False):
    """Isolate real and complex roots of a non-square-free polynomial ``f``. """
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("isolation of real and complex roots is not supported over %s" % K)

    _, factors = dup_sqf_list(f, K)

    if len(factors) == 1:
        ((f, k),) = factors

        real_part, complex_part = dup_isolate_all_roots_sqf(
            f, K, eps=eps, inf=inf, sup=sup, fast=fast)

        real_part = [ ((a, b), k) for (a, b) in real_part ]
        complex_part = [ ((a, b), k) for (a, b) in complex_part ]

        return real_part, complex_part
    else:
        raise NotImplementedError( "only trivial square-free polynomials are supported")

class RealInterval:
    """A fully qualified representation of a real isolation interval. """

    def __init__(self, data, f, dom):
        """Initialize new real interval with complete information. """
        if len(data) == 2:
            s, t = data

            self.neg = False

            if s < 0:
                if t <= 0:
                    f, s, t, self.neg = dup_mirror(f, dom), -t, -s, True
                else:
                    raise ValueError("Cannot refine a real root in (%s, %s)" % (s, t))

            a, b, c, d = _mobius_from_interval((s, t), dom.get_field())

            f = dup_transform(f, dup_strip([a, b]),
                                 dup_strip([c, d]), dom)

            self.mobius = a, b, c, d
        else:
            self.mobius = data[:-1]
            self.neg = data[-1]

        self.f, self.dom = f, dom

    @property
    def func(self):
        return RealInterval

    @property
    def args(self):
        i = self
        return (i.mobius + (i.neg,), i.f, i.dom)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return self.args == other.args

    @property
    def a(self):
        """Return the position of the left end. """
        field = self.dom.get_field()
        a, b, c, d = self.mobius

        if not self.neg:
            if a*d < b*c:
                return field(a, c)
            return field(b, d)
        else:
            if a*d > b*c:
                return -field(a, c)
            return -field(b, d)

    @property
    def b(self):
        """Return the position of the right end. """
        was = self.neg
        self.neg = not was
        rv = -self.a
        self.neg = was
        return rv

    @property
    def dx(self):
        """Return width of the real isolating interval. """
        return self.b - self.a

    @property
    def center(self):
        """Return the center of the real isolating interval. """
        return (self.a + self.b)/2

    @property
    def max_denom(self):
        """Return the largest denominator occurring in either endpoint. """
        return max(self.a.denominator, self.b.denominator)

    def as_tuple(self):
        """Return tuple representation of real isolating interval. """
        return (self.a, self.b)

    def __repr__(self):
        return "(%s, %s)" % (self.a, self.b)

    def __contains__(self, item):
        """
        Say whether a complex number belongs to this real interval.

        Parameters
        ==========

        item : pair (re, im) or number re
            Either a pair giving the real and imaginary parts of the number,
            or else a real number.

        """
        if isinstance(item, tuple):
            re, im = item
        else:
            re, im = item, 0
        return im == 0 and self.a <= re <= self.b

    def is_disjoint(self, other):
        """Return ``True`` if two isolation intervals are disjoint. """
        if isinstance(other, RealInterval):
            return (self.b < other.a or other.b < self.a)
        assert isinstance(other, ComplexInterval)
        return (self.b < other.ax or other.bx < self.a
            or other.ay*other.by > 0)

    def _inner_refine(self):
        """Internal one step real root refinement procedure. """
        if self.mobius is None:
            return self

        f, mobius = dup_inner_refine_real_root(
            self.f, self.mobius, self.dom, steps=1, mobius=True)

        return RealInterval(mobius + (self.neg,), f, self.dom)

    def refine_disjoint(self, other):
        """Refine an isolating interval until it is disjoint with another one. """
        expr = self
        while not expr.is_disjoint(other):
            expr, other = expr._inner_refine(), other._inner_refine()

        return expr, other

    def refine_size(self, dx):
        """Refine an isolating interval until it is of sufficiently small size. """
        expr = self
        while not (expr.dx < dx):
            expr = expr._inner_refine()

        return expr

    def refine_step(self, steps=1):
        """Perform several steps of real root refinement algorithm. """
        expr = self
        for _ in range(steps):
            expr = expr._inner_refine()

        return expr

    def refine(self):
        """Perform one step of real root refinement algorithm. """
        return self._inner_refine()


class ComplexInterval:
    """A fully qualified representation of a complex isolation interval.
    The printed form is shown as (ax, bx) x (ay, by) where (ax, ay)
    and (bx, by) are the coordinates of the southwest and northeast
    corners of the interval's rectangle, respectively.

    Examples
    ========

    >>> from sympy import CRootOf, S
    >>> from sympy.abc import x
    >>> CRootOf.clear_cache()  # for doctest reproducibility
    >>> root = CRootOf(x**10 - 2*x + 3, 9)
    >>> i = root._get_interval(); i
    (3/64, 3/32) x (9/8, 75/64)

    The real part of the root lies within the range [0, 3/4] while
    the imaginary part lies within the range [9/8, 3/2]:

    >>> root.n(3)
    0.0766 + 1.14*I

    The width of the ranges in the x and y directions on the complex
    plane are:

    >>> i.dx, i.dy
    (3/64, 3/64)

    The center of the range is

    >>> i.center
    (9/128, 147/128)

    The northeast coordinate of the rectangle bounding the root in the
    complex plane is given by attribute b and the x and y components
    are accessed by bx and by:

    >>> i.b, i.bx, i.by
    ((3/32, 75/64), 3/32, 75/64)

    The southwest coordinate is similarly given by i.a

    >>> i.a, i.ax, i.ay
    ((3/64, 9/8), 3/64, 9/8)

    Although the interval prints to show only the real and imaginary
    range of the root, all the information of the underlying root
    is contained as properties of the interval.

    For example, an interval with a nonpositive imaginary range is
    considered to be the conjugate. Since the y values of y are in the
    range [0, 1/4] it is not the conjugate:

    >>> i.conj
    False

    The conjugate's interval is

    >>> ic = i.conjugate(); ic
    (3/64, 3/32) x (-75/64, -9/8)

        NOTE: the values printed still represent the x and y range
        in which the root -- conjugate, in this case -- is located,
        but the underlying a and b values of a root and its conjugate
        are the same:

        >>> assert i.a == ic.a and i.b == ic.b

        What changes are the reported coordinates of the bounding rectangle:

        >>> (i.ax, i.ay), (i.bx, i.by)
        ((3/64, 9/8), (3/32, 75/64))
        >>> (ic.ax, ic.ay), (ic.bx, ic.by)
        ((3/64, -75/64), (3/32, -9/8))

    The interval can be refined once:

    >>> i  # for reference, this is the current interval
    (3/64, 3/32) x (9/8, 75/64)

    >>> i.refine()
    (3/64, 3/32) x (9/8, 147/128)

    Several refinement steps can be taken:

    >>> i.refine_step(2)  # 2 steps
    (9/128, 3/32) x (9/8, 147/128)

    It is also possible to refine to a given tolerance:

    >>> tol = min(i.dx, i.dy)/2
    >>> i.refine_size(tol)
    (9/128, 21/256) x (9/8, 291/256)

    A disjoint interval is one whose bounding rectangle does not
    overlap with another. An interval, necessarily, is not disjoint with
    itself, but any interval is disjoint with a conjugate since the
    conjugate rectangle will always be in the lower half of the complex
    plane and the non-conjugate in the upper half:

    >>> i.is_disjoint(i), i.is_disjoint(i.conjugate())
    (False, True)

    The following interval j is not disjoint from i:

    >>> close = CRootOf(x**10 - 2*x + 300/S(101), 9)
    >>> j = close._get_interval(); j
    (75/1616, 75/808) x (225/202, 1875/1616)
    >>> i.is_disjoint(j)
    False

    The two can be made disjoint, however:

    >>> newi, newj = i.refine_disjoint(j)
    >>> newi
    (39/512, 159/2048) x (2325/2048, 4653/4096)
    >>> newj
    (3975/51712, 2025/25856) x (29325/25856, 117375/103424)

    Even though the real ranges overlap, the imaginary do not, so
    the roots have been resolved as distinct. Intervals are disjoint
    when either the real or imaginary component of the intervals is
    distinct. In the case above, the real components have not been
    resolved (so we do not know, yet, which root has the smaller real
    part) but the imaginary part of ``close`` is larger than ``root``:

    >>> close.n(3)
    0.0771 + 1.13*I
    >>> root.n(3)
    0.0766 + 1.14*I
    """

    def __init__(self, a, b, I, Q, F1, F2, f1, f2, dom, conj=False):
        """Initialize new complex interval with complete information. """
        # a and b are the SW and NE corner of the bounding interval,
        # (ax, ay) and (bx, by), respectively, for the NON-CONJUGATE
        # root (the one with the positive imaginary part); when working
        # with the conjugate, the a and b value are still non-negative
        # but the ay, by are reversed and have oppositite sign
        self.a, self.b = a, b
        self.I, self.Q = I, Q

        self.f1, self.F1 = f1, F1
        self.f2, self.F2 = f2, F2

        self.dom = dom
        self.conj = conj

    @property
    def func(self):
        return ComplexInterval

    @property
    def args(self):
        i = self
        return (i.a, i.b, i.I, i.Q, i.F1, i.F2, i.f1, i.f2, i.dom, i.conj)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return self.args == other.args

    @property
    def ax(self):
        """Return ``x`` coordinate of south-western corner. """
        return self.a[0]

    @property
    def ay(self):
        """Return ``y`` coordinate of south-western corner. """
        if not self.conj:
            return self.a[1]
        else:
            return -self.b[1]

    @property
    def bx(self):
        """Return ``x`` coordinate of north-eastern corner. """
        return self.b[0]

    @property
    def by(self):
        """Return ``y`` coordinate of north-eastern corner. """
        if not self.conj:
            return self.b[1]
        else:
            return -self.a[1]

    @property
    def dx(self):
        """Return width of the complex isolating interval. """
        return self.b[0] - self.a[0]

    @property
    def dy(self):
        """Return height of the complex isolating interval. """
        return self.b[1] - self.a[1]

    @property
    def center(self):
        """Return the center of the complex isolating interval. """
        return ((self.ax + self.bx)/2, (self.ay + self.by)/2)

    @property
    def max_denom(self):
        """Return the largest denominator occurring in either endpoint. """
        return max(self.ax.denominator, self.bx.denominator,
                   self.ay.denominator, self.by.denominator)

    def as_tuple(self):
        """Return tuple representation of the complex isolating
        interval's SW and NE corners, respectively. """
        return ((self.ax, self.ay), (self.bx, self.by))

    def __repr__(self):
        return "(%s, %s) x (%s, %s)" % (self.ax, self.bx, self.ay, self.by)

    def conjugate(self):
        """This complex interval really is located in lower half-plane. """
        return ComplexInterval(self.a, self.b, self.I, self.Q,
            self.F1, self.F2, self.f1, self.f2, self.dom, conj=True)

    def __contains__(self, item):
        """
        Say whether a complex number belongs to this complex rectangular
        region.

        Parameters
        ==========

        item : pair (re, im) or number re
            Either a pair giving the real and imaginary parts of the number,
            or else a real number.

        """
        if isinstance(item, tuple):
            re, im = item
        else:
            re, im = item, 0
        return self.ax <= re <= self.bx and self.ay <= im <= self.by

    def is_disjoint(self, other):
        """Return ``True`` if two isolation intervals are disjoint. """
        if isinstance(other, RealInterval):
            return other.is_disjoint(self)
        if self.conj != other.conj:  # above and below real axis
            return True
        re_distinct = (self.bx < other.ax or other.bx < self.ax)
        if re_distinct:
            return True
        im_distinct = (self.by < other.ay or other.by < self.ay)
        return im_distinct

    def _inner_refine(self):
        """Internal one step complex root refinement procedure. """
        (u, v), (s, t) = self.a, self.b

        I, Q = self.I, self.Q

        f1, F1 = self.f1, self.F1
        f2, F2 = self.f2, self.F2

        dom = self.dom

        if s - u > t - v:
            D_L, D_R = _vertical_bisection(1, (u, v), (s, t), I, Q, F1, F2, f1, f2, dom)

            if D_L[0] == 1:
                _, a, b, I, Q, F1, F2 = D_L
            else:
                _, a, b, I, Q, F1, F2 = D_R
        else:
            D_B, D_U = _horizontal_bisection(1, (u, v), (s, t), I, Q, F1, F2, f1, f2, dom)

            if D_B[0] == 1:
                _, a, b, I, Q, F1, F2 = D_B
            else:
                _, a, b, I, Q, F1, F2 = D_U

        return ComplexInterval(a, b, I, Q, F1, F2, f1, f2, dom, self.conj)

    def refine_disjoint(self, other):
        """Refine an isolating interval until it is disjoint with another one. """
        expr = self
        while not expr.is_disjoint(other):
            expr, other = expr._inner_refine(), other._inner_refine()

        return expr, other

    def refine_size(self, dx, dy=None):
        """Refine an isolating interval until it is of sufficiently small size. """
        if dy is None:
            dy = dx
        expr = self
        while not (expr.dx < dx and expr.dy < dy):
            expr = expr._inner_refine()

        return expr

    def refine_step(self, steps=1):
        """Perform several steps of complex root refinement algorithm. """
        expr = self
        for _ in range(steps):
            expr = expr._inner_refine()

        return expr

    def refine(self):
        """Perform one step of complex root refinement algorithm. """
        return self._inner_refine()