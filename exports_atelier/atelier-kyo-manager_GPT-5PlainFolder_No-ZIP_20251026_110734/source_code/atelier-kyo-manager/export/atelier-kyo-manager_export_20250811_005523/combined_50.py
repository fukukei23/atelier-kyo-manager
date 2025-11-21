
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\hyperbolic.py ===
from sympy.core import S, sympify, cacheit
from sympy.core.add import Add
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.logic import fuzzy_or, fuzzy_and, fuzzy_not, FuzzyBool
from sympy.core.numbers import I, pi, Rational
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import (binomial, factorial,
                                                      RisingFactorial)
from sympy.functions.combinatorial.numbers import bernoulli, euler, nC
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
    acos, acot, asin, atan, cos, cot, csc, sec, sin, tan,
    _imaginary_unit_as_coefficient)
from sympy.polys.specialpolys import symmetric_poly


def _rewrite_hyperbolics_as_exp(expr):
    return expr.xreplace({h: h.rewrite(exp)
        for h in expr.atoms(HyperbolicFunction)})


@cacheit
def _acosh_table():
    return {
        I: log(I*(1 + sqrt(2))),
        -I: log(-I*(1 + sqrt(2))),
        S.Half: pi/3,
        Rational(-1, 2): pi*Rational(2, 3),
        sqrt(2)/2: pi/4,
        -sqrt(2)/2: pi*Rational(3, 4),
        1/sqrt(2): pi/4,
        -1/sqrt(2): pi*Rational(3, 4),
        sqrt(3)/2: pi/6,
        -sqrt(3)/2: pi*Rational(5, 6),
        (sqrt(3) - 1)/sqrt(2**3): pi*Rational(5, 12),
        -(sqrt(3) - 1)/sqrt(2**3): pi*Rational(7, 12),
        sqrt(2 + sqrt(2))/2: pi/8,
        -sqrt(2 + sqrt(2))/2: pi*Rational(7, 8),
        sqrt(2 - sqrt(2))/2: pi*Rational(3, 8),
        -sqrt(2 - sqrt(2))/2: pi*Rational(5, 8),
        (1 + sqrt(3))/(2*sqrt(2)): pi/12,
        -(1 + sqrt(3))/(2*sqrt(2)): pi*Rational(11, 12),
        (sqrt(5) + 1)/4: pi/5,
        -(sqrt(5) + 1)/4: pi*Rational(4, 5)
    }


@cacheit
def _acsch_table():
    return {
            I: -pi / 2,
            I*(sqrt(2) + sqrt(6)): -pi / 12,
            I*(1 + sqrt(5)): -pi / 10,
            I*2 / sqrt(2 - sqrt(2)): -pi / 8,
            I*2: -pi / 6,
            I*sqrt(2 + 2/sqrt(5)): -pi / 5,
            I*sqrt(2): -pi / 4,
            I*(sqrt(5)-1): -3*pi / 10,
            I*2 / sqrt(3): -pi / 3,
            I*2 / sqrt(2 + sqrt(2)): -3*pi / 8,
            I*sqrt(2 - 2/sqrt(5)): -2*pi / 5,
            I*(sqrt(6) - sqrt(2)): -5*pi / 12,
            S(2): -I*log((1+sqrt(5))/2),
        }


@cacheit
def _asech_table():
        return {
            I: - (pi*I / 2) + log(1 + sqrt(2)),
            -I: (pi*I / 2) + log(1 + sqrt(2)),
            (sqrt(6) - sqrt(2)): pi / 12,
            (sqrt(2) - sqrt(6)): 11*pi / 12,
            sqrt(2 - 2/sqrt(5)): pi / 10,
            -sqrt(2 - 2/sqrt(5)): 9*pi / 10,
            2 / sqrt(2 + sqrt(2)): pi / 8,
            -2 / sqrt(2 + sqrt(2)): 7*pi / 8,
            2 / sqrt(3): pi / 6,
            -2 / sqrt(3): 5*pi / 6,
            (sqrt(5) - 1): pi / 5,
            (1 - sqrt(5)): 4*pi / 5,
            sqrt(2): pi / 4,
            -sqrt(2): 3*pi / 4,
            sqrt(2 + 2/sqrt(5)): 3*pi / 10,
            -sqrt(2 + 2/sqrt(5)): 7*pi / 10,
            S(2): pi / 3,
            -S(2): 2*pi / 3,
            sqrt(2*(2 + sqrt(2))): 3*pi / 8,
            -sqrt(2*(2 + sqrt(2))): 5*pi / 8,
            (1 + sqrt(5)): 2*pi / 5,
            (-1 - sqrt(5)): 3*pi / 5,
            (sqrt(6) + sqrt(2)): 5*pi / 12,
            (-sqrt(6) - sqrt(2)): 7*pi / 12,
            I*S.Infinity: -pi*I / 2,
            I*S.NegativeInfinity: pi*I / 2,
        }

###############################################################################
########################### HYPERBOLIC FUNCTIONS ##############################
###############################################################################


class HyperbolicFunction(DefinedFunction):
    """
    Base class for hyperbolic functions.

    See Also
    ========

    sinh, cosh, tanh, coth
    """

    unbranched = True


def _peeloff_ipi(arg):
    r"""
    Split ARG into two parts, a "rest" and a multiple of $I\pi$.
    This assumes ARG to be an ``Add``.
    The multiple of $I\pi$ returned in the second position is always a ``Rational``.

    Examples
    ========

    >>> from sympy.functions.elementary.hyperbolic import _peeloff_ipi as peel
    >>> from sympy import pi, I
    >>> from sympy.abc import x, y
    >>> peel(x + I*pi/2)
    (x, 1/2)
    >>> peel(x + I*2*pi/3 + I*pi*y)
    (x + I*pi*y + I*pi/6, 1/2)
    """
    ipi = pi*I
    for a in Add.make_args(arg):
        if a == ipi:
            K = S.One
            break
        elif a.is_Mul:
            K, p = a.as_two_terms()
            if p == ipi and K.is_Rational:
                break
    else:
        return arg, S.Zero

    m1 = (K % S.Half)
    m2 = K - m1
    return arg - m2*ipi, m2


class sinh(HyperbolicFunction):
    r"""
    ``sinh(x)`` is the hyperbolic sine of ``x``.

    The hyperbolic sine function is $\frac{e^x - e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import sinh
    >>> from sympy.abc import x
    >>> sinh(x)
    sinh(x)

    See Also
    ========

    cosh, tanh, asinh
    """

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return cosh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return asinh

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg.is_zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return I * sin(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    m = m*pi*I
                    return sinh(m)*cosh(x) + cosh(m)*sinh(x)

            if arg.is_zero:
                return S.Zero

            if arg.func == asinh:
                return arg.args[0]

            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x - 1) * sqrt(x + 1)

            if arg.func == atanh:
                x = arg.args[0]
                return x/sqrt(1 - x**2)

            if arg.func == acoth:
                x = arg.args[0]
                return 1/(sqrt(x - 1) * sqrt(x + 1))

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the next term in the Taylor series expansion.
        """
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return p * x**2 / (n*(n - 1))
            else:
                return x**(n) / factorial(n)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.
        """
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (sinh(re)*cos(im), cosh(re)*sin(im))

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I

    def _eval_expand_trig(self, deep=True, **hints):
        if deep:
            arg = self.args[0].expand(deep, **hints)
        else:
            arg = self.args[0]
        x = None
        if arg.is_Add: # TODO, implement more if deep stuff here
            x, y = arg.as_two_terms()
        else:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff is not S.One and coeff.is_Integer and terms is not S.One:
                x = terms
                y = (coeff - 1)*x
        if x is not None:
            return (sinh(x)*cosh(y) + sinh(y)*cosh(x)).expand(trig=True)
        return sinh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return (exp(arg) - exp(-arg)) / 2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return (exp(arg) - exp(-arg)) / 2

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return -I * sin(I * arg)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return -I / csc(I * arg)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return -I*cosh(arg + pi*I/2)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        tanh_half = tanh(S.Half*arg)
        return 2*tanh_half/(1 - tanh_half**2)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        coth_half = coth(S.Half*arg)
        return 2*coth_half/(coth_half**2 - 1)

    def _eval_rewrite_as_csch(self, arg, **kwargs):
        return 1 / csch(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if cdir.is_negative else '+')
        if arg0.is_zero:
            return arg
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real:
            return True

        # if `im` is of the form n*pi
        # else, check if it is a number
        re, im = arg.as_real_imag()
        return (im%pi).is_zero

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_is_finite(self):
        arg = self.args[0]
        return arg.is_finite

    def _eval_is_zero(self):
        rest, ipi_mult = _peeloff_ipi(self.args[0])
        if rest.is_zero:
            return ipi_mult.is_integer


class cosh(HyperbolicFunction):
    r"""
    ``cosh(x)`` is the hyperbolic cosine of ``x``.

    The hyperbolic cosine function is $\frac{e^x + e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import cosh
    >>> from sympy.abc import x
    >>> cosh(x)
    cosh(x)

    See Also
    ========

    sinh, tanh, acosh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return sinh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.functions.elementary.trigonometric import cos
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg.is_zero:
                return S.One
            elif arg.is_negative:
                return cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return cos(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    m = m*pi*I
                    return cosh(m)*cosh(x) + sinh(m)*sinh(x)

            if arg.is_zero:
                return S.One

            if arg.func == asinh:
                return sqrt(1 + arg.args[0]**2)

            if arg.func == acosh:
                return arg.args[0]

            if arg.func == atanh:
                return 1/sqrt(1 - arg.args[0]**2)

            if arg.func == acoth:
                x = arg.args[0]
                return x/(sqrt(x - 1) * sqrt(x + 1))

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return p * x**2 / (n*(n - 1))
            else:
                return x**(n)/factorial(n)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()

        return (cosh(re)*cos(im), sinh(re)*sin(im))

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I

    def _eval_expand_trig(self, deep=True, **hints):
        if deep:
            arg = self.args[0].expand(deep, **hints)
        else:
            arg = self.args[0]
        x = None
        if arg.is_Add: # TODO, implement more if deep stuff here
            x, y = arg.as_two_terms()
        else:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff is not S.One and coeff.is_Integer and terms is not S.One:
                x = terms
                y = (coeff - 1)*x
        if x is not None:
            return (cosh(x)*cosh(y) + sinh(x)*sinh(y)).expand(trig=True)
        return cosh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return (exp(arg) + exp(-arg)) / 2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return (exp(arg) + exp(-arg)) / 2

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return cos(I * arg, evaluate=False)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1 / sec(I * arg, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return -I*sinh(arg + pi*I/2, evaluate=False)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        tanh_half = tanh(S.Half*arg)**2
        return (1 + tanh_half)/(1 - tanh_half)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        coth_half = coth(S.Half*arg)**2
        return (coth_half + 1)/(coth_half - 1)

    def _eval_rewrite_as_sech(self, arg, **kwargs):
        return 1 / sech(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if cdir.is_negative else '+')
        if arg0.is_zero:
            return S.One
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_is_real(self):
        arg = self.args[0]

        # `cosh(x)` is real for real OR purely imaginary `x`
        if arg.is_real or arg.is_imaginary:
            return True

        # cosh(a+ib) = cos(b)*cosh(a) + i*sin(b)*sinh(a)
        # the imaginary part can be an expression like n*pi
        # if not, check if the imaginary part is a number
        re, im = arg.as_real_imag()
        return (im%pi).is_zero

    def _eval_is_positive(self):
        # cosh(x+I*y) = cos(y)*cosh(x) + I*sin(y)*sinh(x)
        # cosh(z) is positive iff it is real and the real part is positive.
        # So we need sin(y)*sinh(x) = 0 which gives x=0 or y=n*pi
        # Case 1 (y=n*pi): cosh(z) = (-1)**n * cosh(x) -> positive for n even
        # Case 2 (x=0): cosh(z) = cos(y) -> positive when cos(y) is positive
        z = self.args[0]

        x, y = z.as_real_imag()
        ymod = y % (2*pi)

        yzero = ymod.is_zero
        # shortcut if ymod is zero
        if yzero:
            return True

        xzero = x.is_zero
        # shortcut x is not zero
        if xzero is False:
            return yzero

        return fuzzy_or([
                # Case 1:
                yzero,
                # Case 2:
                fuzzy_and([
                    xzero,
                    fuzzy_or([ymod < pi/2, ymod > 3*pi/2])
                ])
            ])


    def _eval_is_nonnegative(self):
        z = self.args[0]

        x, y = z.as_real_imag()
        ymod = y % (2*pi)

        yzero = ymod.is_zero
        # shortcut if ymod is zero
        if yzero:
            return True

        xzero = x.is_zero
        # shortcut x is not zero
        if xzero is False:
            return yzero

        return fuzzy_or([
                # Case 1:
                yzero,
                # Case 2:
                fuzzy_and([
                    xzero,
                    fuzzy_or([ymod <= pi/2, ymod >= 3*pi/2])
                ])
            ])

    def _eval_is_finite(self):
        arg = self.args[0]
        return arg.is_finite

    def _eval_is_zero(self):
        rest, ipi_mult = _peeloff_ipi(self.args[0])
        if ipi_mult and rest.is_zero:
            return (ipi_mult - S.Half).is_integer


class tanh(HyperbolicFunction):
    r"""
    ``tanh(x)`` is the hyperbolic tangent of ``x``.

    The hyperbolic tangent function is $\frac{\sinh(x)}{\cosh(x)}$.

    Examples
    ========

    >>> from sympy import tanh
    >>> from sympy.abc import x
    >>> tanh(x)
    tanh(x)

    See Also
    ========

    sinh, cosh, atanh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return S.One - tanh(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return atanh

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                if i_coeff.could_extract_minus_sign():
                    return -I * tan(-i_coeff)
                return I * tan(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    tanhm = tanh(m*pi*I)
                    if tanhm is S.ComplexInfinity:
                        return coth(x)
                    else: # tanhm == 0
                        return tanh(x)

            if arg.is_zero:
                return S.Zero

            if arg.func == asinh:
                x = arg.args[0]
                return x/sqrt(1 + x**2)

            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x - 1) * sqrt(x + 1) / x

            if arg.func == atanh:
                return arg.args[0]

            if arg.func == acoth:
                return 1/arg.args[0]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            a = 2**(n + 1)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return a*(a - 1) * B/F * x**n

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = sinh(re)**2 + cos(im)**2
        return (sinh(re)*cosh(re)/denom, sin(im)*cos(im)/denom)

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        if arg.is_Add:
            n = len(arg.args)
            TX = [tanh(x, evaluate=False)._eval_expand_trig()
                for x in arg.args]
            p = [0, 0]  # [den, num]
            for i in range(n + 1):
                p[i % 2] += symmetric_poly(i, TX)
            return p[1]/p[0]
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul()
            if coeff.is_Integer and coeff > 1:
                T = tanh(terms)
                n = [nC(range(coeff), k)*T**k for k in range(1, coeff + 1, 2)]
                d = [nC(range(coeff), k)*T**k for k in range(0, coeff + 1, 2)]
                return Add(*n)/Add(*d)
        return tanh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp - neg_exp)/(pos_exp + neg_exp)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp - neg_exp)/(pos_exp + neg_exp)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return -I * tan(I * arg, evaluate=False)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        return -I / cot(I * arg, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return I*sinh(arg)/sinh(pi*I/2 - arg, evaluate=False)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return I*cosh(pi*I/2 - arg, evaluate=False)/cosh(arg)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        return 1/coth(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x)

        if x in arg.free_symbols and Order(1, x).contains(arg):
            return arg
        else:
            return self.func(arg)

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real:
            return True

        re, im = arg.as_real_imag()

        # if denom = 0, tanh(arg) = zoo
        if re == 0 and im % pi == pi/2:
            return None

        # check if im is of the form n*pi/2 to make sin(2*im) = 0
        # if not, im could be a number, return False in that case
        return (im % (pi/2)).is_zero

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_is_finite(self):
        arg = self.args[0]

        re, im = arg.as_real_imag()
        denom = cos(im)**2 + sinh(re)**2
        if denom == 0:
            return False
        elif denom.is_number:
            return True
        if arg.is_extended_real:
            return True

    def _eval_is_zero(self):
        arg = self.args[0]
        if arg.is_zero:
            return True


class coth(HyperbolicFunction):
    r"""
    ``coth(x)`` is the hyperbolic cotangent of ``x``.

    The hyperbolic cotangent function is $\frac{\cosh(x)}{\sinh(x)}$.

    Examples
    ========

    >>> from sympy import coth
    >>> from sympy.abc import x
    >>> coth(x)
    coth(x)

    See Also
    ========

    sinh, cosh, acoth
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/sinh(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return acoth

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.ComplexInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                if i_coeff.could_extract_minus_sign():
                    return I * cot(-i_coeff)
                return -I * cot(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    cothm = coth(m*pi*I)
                    if cothm is S.ComplexInfinity:
                        return coth(x)
                    else: # cothm == 0
                        return tanh(x)

            if arg.is_zero:
                return S.ComplexInfinity

            if arg.func == asinh:
                x = arg.args[0]
                return sqrt(1 + x**2)/x

            if arg.func == acosh:
                x = arg.args[0]
                return x/(sqrt(x - 1) * sqrt(x + 1))

            if arg.func == atanh:
                return 1/arg.args[0]

            if arg.func == acoth:
                return arg.args[0]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1 / sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return 2**(n + 1) * B/F * x**n

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.trigonometric import (cos, sin)
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = sinh(re)**2 + sin(im)**2
        return (sinh(re)*cosh(re)/denom, -sin(im)*cos(im)/denom)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp + neg_exp)/(pos_exp - neg_exp)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp + neg_exp)/(pos_exp - neg_exp)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return -I*sinh(pi*I/2 - arg, evaluate=False)/sinh(arg)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return -I*cosh(arg)/cosh(pi*I/2 - arg, evaluate=False)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        return 1/tanh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x)

        if x in arg.free_symbols and Order(1, x).contains(arg):
            return 1/arg
        else:
            return self.func(arg)

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        if arg.is_Add:
            CX = [coth(x, evaluate=False)._eval_expand_trig() for x in arg.args]
            p = [[], []]
            n = len(arg.args)
            for i in range(n, -1, -1):
                p[(n - i) % 2].append(symmetric_poly(i, CX))
            return Add(*p[0])/Add(*p[1])
        elif arg.is_Mul:
            coeff, x = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer and coeff > 1:
                c = coth(x, evaluate=False)
                p = [[], []]
                for i in range(coeff, -1, -1):
                    p[(coeff - i) % 2].append(binomial(coeff, i)*c**i)
                return Add(*p[0])/Add(*p[1])
        return coth(arg)


class ReciprocalHyperbolicFunction(HyperbolicFunction):
    """Base class for reciprocal functions of hyperbolic functions. """

    #To be defined in class
    _reciprocal_of = None
    _is_even: FuzzyBool = None
    _is_odd: FuzzyBool = None

    @classmethod
    def eval(cls, arg):
        if arg.could_extract_minus_sign():
            if cls._is_even:
                return cls(-arg)
            if cls._is_odd:
                return -cls(-arg)

        t = cls._reciprocal_of.eval(arg)
        if hasattr(arg, 'inverse') and arg.inverse() == cls:
            return arg.args[0]
        return 1/t if t is not None else t

    def _call_reciprocal(self, method_name, *args, **kwargs):
        # Calls method_name on _reciprocal_of
        o = self._reciprocal_of(self.args[0])
        return getattr(o, method_name)(*args, **kwargs)

    def _calculate_reciprocal(self, method_name, *args, **kwargs):
        # If calling method_name on _reciprocal_of returns a value != None
        # then return the reciprocal of that value
        t = self._call_reciprocal(method_name, *args, **kwargs)
        return 1/t if t is not None else t

    def _rewrite_reciprocal(self, method_name, arg):
        # Special handling for rewrite functions. If reciprocal rewrite returns
        # unmodified expression, then return None
        t = self._call_reciprocal(method_name, arg)
        if t is not None and t != self._reciprocal_of(arg):
            return 1/t

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_exp", arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_tractable", arg)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_tanh", arg)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_coth", arg)

    def as_real_imag(self, deep = True, **hints):
        return (1 / self._reciprocal_of(self.args[0])).as_real_imag(deep, **hints)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=True, **hints)
        return re_part + I*im_part

    def _eval_expand_trig(self, **hints):
        return self._calculate_reciprocal("_eval_expand_trig", **hints)

    def _eval_as_leading_term(self, x, logx, cdir):
        return (1/self._reciprocal_of(self.args[0]))._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_is_extended_real(self):
        return self._reciprocal_of(self.args[0]).is_extended_real

    def _eval_is_finite(self):
        return (1/self._reciprocal_of(self.args[0])).is_finite


class csch(ReciprocalHyperbolicFunction):
    r"""
    ``csch(x)`` is the hyperbolic cosecant of ``x``.

    The hyperbolic cosecant function is $\frac{2}{e^x - e^{-x}}$

    Examples
    ========

    >>> from sympy import csch
    >>> from sympy.abc import x
    >>> csch(x)
    csch(x)

    See Also
    ========

    sinh, cosh, tanh, sech, asinh, acosh
    """

    _reciprocal_of = sinh
    _is_odd = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function
        """
        if argindex == 1:
            return -coth(self.args[0]) * csch(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the next term in the Taylor series expansion
        """
        if n == 0:
            return 1/sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return 2 * (1 - 2**n) * B/F * x**n

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return I / sin(I * arg, evaluate=False)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return I * csc(I * arg, evaluate=False)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return I / cosh(arg + I * pi / 2, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return 1 / sinh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative


class sech(ReciprocalHyperbolicFunction):
    r"""
    ``sech(x)`` is the hyperbolic secant of ``x``.

    The hyperbolic secant function is $\frac{2}{e^x + e^{-x}}$

    Examples
    ========

    >>> from sympy import sech
    >>> from sympy.abc import x
    >>> sech(x)
    sech(x)

    See Also
    ========

    sinh, cosh, tanh, coth, csch, asinh, acosh
    """

    _reciprocal_of = cosh
    _is_even = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return - tanh(self.args[0])*sech(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            return euler(n) / factorial(n) * x**(n)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return 1 / cos(I * arg, evaluate=False)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return sec(I * arg, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return I / sinh(arg + I * pi /2, evaluate=False)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return 1 / cosh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return True


###############################################################################
############################# HYPERBOLIC INVERSES #############################
###############################################################################

class InverseHyperbolicFunction(DefinedFunction):
    """Base class for inverse hyperbolic functions."""

    pass


class asinh(InverseHyperbolicFunction):
    """
    ``asinh(x)`` is the inverse hyperbolic sine of ``x``.

    The inverse hyperbolic sine function.

    Examples
    ========

    >>> from sympy import asinh
    >>> from sympy.abc import x
    >>> asinh(x).diff(x)
    1/sqrt(x**2 + 1)
    >>> asinh(1)
    log(1 + sqrt(2))

    See Also
    ========

    acosh, atanh, sinh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/sqrt(self.args[0]**2 + 1)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return log(sqrt(2) + 1)
            elif arg is S.NegativeOne:
                return log(sqrt(2) - 1)
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.ComplexInfinity

            if arg.is_zero:
                return S.Zero

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return I * asin(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        if isinstance(arg, sinh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return z
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor((i + pi/2)/pi)
                m = z - I*pi*f
                even = f.is_even
                if even is True:
                    return m
                elif even is False:
                    return -m

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return -p * (n - 2)**2/(n*(n - 1)) * x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return S.NegativeOne**k * R / F * x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0.is_zero:
            return arg.as_leading_term(x)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling branch points
        if x0 in (-I, I, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts (-I*oo, -I) U (I, I*oo)
        if (1 + x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_negative:
                    return -self.func(x0) - I*pi
            elif re(ndir).is_negative:
                if im(x0).is_positive:
                    return -self.func(x0) + I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asinh
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (I, -I):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-I*oo, -I) U (I, I*oo)
        if (1 + arg0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(arg0).is_negative:
                    return -res - I*pi
            elif re(ndir).is_negative:
                if im(arg0).is_positive:
                    return -res + I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return log(x + sqrt(x**2 + 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return atanh(x/sqrt(1 + x**2))

    def _eval_rewrite_as_acosh(self, x, **kwargs):
        ix = I*x
        return I*(sqrt(1 - ix)/sqrt(ix - 1) * acosh(ix) - pi/2)

    def _eval_rewrite_as_asin(self, x, **kwargs):
        return -I * asin(I * x, evaluate=False)

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return I * acos(I * x, evaluate=False) - I*pi/2

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sinh

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        return self.args[0].is_finite


class acosh(InverseHyperbolicFunction):
    """
    ``acosh(x)`` is the inverse hyperbolic cosine of ``x``.

    The inverse hyperbolic cosine function.

    Examples
    ========

    >>> from sympy import acosh
    >>> from sympy.abc import x
    >>> acosh(x).diff(x)
    1/(sqrt(x - 1)*sqrt(x + 1))
    >>> acosh(1)
    0

    See Also
    ========

    asinh, atanh, cosh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            arg = self.args[0]
            return 1/(sqrt(arg - 1)*sqrt(arg + 1))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg.is_zero:
                return pi*I / 2
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi*I

        if arg.is_number:
            cst_table = _acosh_table()

            if arg in cst_table:
                if arg.is_extended_real:
                    return cst_table[arg]*I
                return cst_table[arg]

        if arg is S.ComplexInfinity:
            return S.ComplexInfinity
        if arg == I*S.Infinity:
            return S.Infinity + I*pi/2
        if arg == -I*S.Infinity:
            return S.Infinity - I*pi/2

        if arg.is_zero:
            return pi*I*S.Half

        if isinstance(arg, cosh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return Abs(z)
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor(i/pi)
                m = z - I*pi*f
                even = f.is_even
                if even is True:
                    if r.is_nonnegative:
                        return m
                    elif r.is_negative:
                        return -m
                elif even is False:
                    m -= I*pi
                    if r.is_nonpositive:
                        return -m
                    elif r.is_positive:
                        return m

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return I*pi/2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p * (n - 2)**2/(n*(n - 1)) * x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return -R / F * I * x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        # Handling branch points
        if x0 in (-S.One, S.Zero, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling points lying on branch cuts (-oo, 1)
        if (x0 - 1).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if (x0 + 1).is_negative:
                    return self.func(x0) - 2*I*pi
                return -self.func(x0)
            elif not im(ndir).is_positive:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acosh
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (S.One, S.NegativeOne):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-oo, 1)
        if (arg0 - 1).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if (arg0 + 1).is_negative:
                    return res - 2*I*pi
                return -res
            elif not im(ndir).is_positive:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return log(x + sqrt(x + 1) * sqrt(x - 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return sqrt(x - 1)/sqrt(1 - x) * acos(x)

    def _eval_rewrite_as_asin(self, x, **kwargs):
        return sqrt(x - 1)/sqrt(1 - x) * (pi/2 - asin(x))

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        return sqrt(x - 1)/sqrt(1 - x) * (pi/2 + I*asinh(I*x, evaluate=False))

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        sxm1 = sqrt(x - 1)
        s1mx = sqrt(1 - x)
        sx2m1 = sqrt(x**2 - 1)
        return (pi/2*sxm1/s1mx*(1 - x * sqrt(1/x**2)) +
                sxm1*sqrt(x + 1)/sx2m1 * atanh(sx2m1/x))

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return cosh

    def _eval_is_zero(self):
        if (self.args[0] - 1).is_zero:
            return True

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, (self.args[0] - 1).is_extended_nonnegative])

    def _eval_is_finite(self):
        return self.args[0].is_finite


class atanh(InverseHyperbolicFunction):
    """
    ``atanh(x)`` is the inverse hyperbolic tangent of ``x``.

    The inverse hyperbolic tangent function.

    Examples
    ========

    >>> from sympy import atanh
    >>> from sympy.abc import x
    >>> atanh(x).diff(x)
    1/(1 - x**2)

    See Also
    ========

    asinh, acosh, tanh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg is S.Infinity:
                return -I * atan(arg)
            elif arg is S.NegativeInfinity:
                return I * atan(-arg)
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                from sympy.calculus.accumulationbounds import AccumBounds
                return I*AccumBounds(-pi/2, pi/2)

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return I * atan(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, tanh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return z
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor(2*i/pi)
                even = f.is_even
                m = z - I*f*pi/2
                if even is True:
                    return m
                elif even is False:
                    return m - I*pi/2

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0.is_zero:
            return arg.as_leading_term(x)
        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling branch points
        if x0 in (-S.One, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts (-oo, -1] U [1, oo)
        if (1 - x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return self.func(x0) - I*pi
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return self.func(x0) + I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # atanh
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (S.One, S.NegativeOne):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-oo, -1] U [1, oo)
        if (1 - arg0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_negative:
                    return res - I*pi
            elif im(ndir).is_positive:
                if arg0.is_positive:
                    return res + I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return (log(1 + x) - log(1 - x)) / 2

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        f = sqrt(1/(x**2 - 1))
        return (pi*x/(2*sqrt(-x**2)) -
                sqrt(-x)*sqrt(1 - x**2)/sqrt(x)*f*asinh(f))

    def _eval_is_zero(self):
        if self.args[0].is_zero:
            return True

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, (1 - self.args[0]).is_nonnegative, (self.args[0] + 1).is_nonnegative])

    def _eval_is_finite(self):
        return fuzzy_not(fuzzy_or([(self.args[0] - 1).is_zero, (self.args[0] + 1).is_zero]))

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return tanh


class acoth(InverseHyperbolicFunction):
    """
    ``acoth(x)`` is the inverse hyperbolic cotangent of ``x``.

    The inverse hyperbolic cotangent function.

    Examples
    ========

    >>> from sympy import acoth
    >>> from sympy.abc import x
    >>> acoth(x).diff(x)
    1/(1 - x**2)

    See Also
    ========

    asinh, acosh, coth
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

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
                return pi*I / 2
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.Zero

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return -I * acot(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        if arg.is_zero:
            return pi*I*S.Half

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return -I*pi/2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling branch points
        if x0 in (-S.One, S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts [-1, 1]
        if x0.is_real and (1 - x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_positive:
                    return self.func(x0) + I*pi
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return self.func(x0) - I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acoth
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (S.One, S.NegativeOne):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts [-1, 1]
        if arg0.is_real and (1 - arg0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_positive:
                    return res + I*pi
            elif im(ndir).is_positive:
                if arg0.is_negative:
                    return res - I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return (log(1 + 1/x) - log(1 - 1/x)) / 2

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return atanh(1/x)

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        return (pi*I/2*(sqrt((x - 1)/x)*sqrt(x/(x - 1)) - sqrt(1 + 1/x)*sqrt(x/(x + 1))) +
                x*sqrt(1/x**2)*asinh(sqrt(1/(x**2 - 1))))

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return coth

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, fuzzy_or([(self.args[0] - 1).is_extended_nonnegative, (self.args[0] + 1).is_extended_nonpositive])])

    def _eval_is_finite(self):
        return fuzzy_not(fuzzy_or([(self.args[0] - 1).is_zero, (self.args[0] + 1).is_zero]))


class asech(InverseHyperbolicFunction):
    """
    ``asech(x)`` is the inverse hyperbolic secant of ``x``.

    The inverse hyperbolic secant function.

    Examples
    ========

    >>> from sympy import asech, sqrt, S
    >>> from sympy.abc import x
    >>> asech(x).diff(x)
    -1/(x*sqrt(1 - x**2))
    >>> asech(1).diff(x)
    0
    >>> asech(1)
    0
    >>> asech(S(2))
    I*pi/3
    >>> asech(-sqrt(2))
    3*I*pi/4
    >>> asech((sqrt(6) - sqrt(2)))
    I*pi/12

    See Also
    ========

    asinh, atanh, cosh, acoth

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    .. [2] https://dlmf.nist.gov/4.37
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSech/

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -1/(z*sqrt(1 - z**2))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return pi*I / 2
            elif arg is S.NegativeInfinity:
                return pi*I / 2
            elif arg.is_zero:
                return S.Infinity
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi*I

        if arg.is_number:
            cst_table = _asech_table()

            if arg in cst_table:
                if arg.is_extended_real:
                    return cst_table[arg]*I
                return cst_table[arg]

        if arg is S.ComplexInfinity:
            from sympy.calculus.accumulationbounds import AccumBounds
            return I*AccumBounds(-pi/2, pi/2)

        if arg.is_zero:
            return S.Infinity

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) * n
                F = factorial(k) * n // 2 * n // 2
                return -1 * R / F * x**n / 4

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        # Handling branch points
        if x0 in (-S.One, S.Zero, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling points lying on branch cuts (-oo, 0] U (1, oo)
        if x0.is_negative or (1 - x0).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_positive:
                if x0.is_positive or (x0 + 1).is_negative:
                    return -self.func(x0)
                return self.func(x0) - 2*I*pi
            elif not im(ndir).is_negative:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asech
        from sympy.series.order import O
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = asech(S.One - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = asech(S.NegativeOne + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else I*pi + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-oo, 0] U (1, oo)
        if arg0.is_negative or (1 - arg0).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_positive:
                if arg0.is_positive or (arg0 + 1).is_negative:
                    return -res
                return res - 2*I*pi
            elif not im(ndir).is_negative:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sech

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return log(1/arg + sqrt(1/arg - 1) * sqrt(1/arg + 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acosh(self, arg, **kwargs):
        return acosh(1/arg)

    def _eval_rewrite_as_asinh(self, arg, **kwargs):
        return sqrt(1/arg - 1)/sqrt(1 - 1/arg)*(I*asinh(I/arg, evaluate=False)
                                                + pi*S.Half)

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return (I*pi*(1 - sqrt(x)*sqrt(1/x) - I/2*sqrt(-x)/sqrt(x) - I/2*sqrt(x**2)/sqrt(-x**2))
                + sqrt(1/(x + 1))*sqrt(x + 1)*atanh(sqrt(1 - x**2)))

    def _eval_rewrite_as_acsch(self, x, **kwargs):
        return sqrt(1/x - 1)/sqrt(1 - 1/x)*(pi/2 - I*acsch(I*x, evaluate=False))

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, self.args[0].is_nonnegative, (1 - self.args[0]).is_nonnegative])

    def _eval_is_finite(self):
        return fuzzy_not(self.args[0].is_zero)


class acsch(InverseHyperbolicFunction):
    """
    ``acsch(x)`` is the inverse hyperbolic cosecant of ``x``.

    The inverse hyperbolic cosecant function.

    Examples
    ========

    >>> from sympy import acsch, sqrt, I
    >>> from sympy.abc import x
    >>> acsch(x).diff(x)
    -1/(x**2*sqrt(1 + x**(-2)))
    >>> acsch(1).diff(x)
    0
    >>> acsch(1)
    log(1 + sqrt(2))
    >>> acsch(I)
    -I*pi/2
    >>> acsch(-2*I)
    I*pi/6
    >>> acsch(I*(sqrt(6) - sqrt(2)))
    -5*I*pi/12

    See Also
    ========

    asinh

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    .. [2] https://dlmf.nist.gov/4.37
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCsch/

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -1/(z**2*sqrt(1 + 1/z**2))
        else:
            raise ArgumentIndexError(self, argindex)

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
                return S.ComplexInfinity
            elif arg is S.One:
                return log(1 + sqrt(2))
            elif arg is S.NegativeOne:
                return - log(1 + sqrt(2))

        if arg.is_number:
            cst_table = _acsch_table()

            if arg in cst_table:
                return cst_table[arg]*I

        if arg is S.ComplexInfinity:
            return S.Zero

        if arg.is_infinite:
            return S.Zero

        if arg.is_zero:
            return S.ComplexInfinity

        if arg.could_extract_minus_sign():
            return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return -p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) *  n
                F = factorial(k) * n // 2 * n // 2
                return S.NegativeOne**(k +1) * R / F * x**n / 4

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        # Handling branch points
        if x0 in (-I, I, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        # Handling points lying on branch cuts (-I, I)
        if x0.is_imaginary and (1 + x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_positive:
                    return -self.func(x0) - I*pi
            elif re(ndir).is_negative:
                if im(x0).is_negative:
                    return -self.func(x0) + I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acsch
        from sympy.series.order import O
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 is I:
            t = Dummy('t', positive=True)
            ser = acsch(I + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = -I + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else -I*pi/2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            res = ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)
            return res

        if arg0 == S.NegativeOne*I:
            t = Dummy('t', positive=True)
            ser = acsch(-I + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = I + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else I*pi/2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-I, I)
        if arg0.is_imaginary and (1 + arg0**2).is_positive:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(arg0).is_positive:
                    return -res - I*pi
            elif re(ndir).is_negative:
                if im(arg0).is_negative:
                    return -res + I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return csch

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return log(1/arg + sqrt(1/arg**2 + 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asinh(self, arg, **kwargs):
        return asinh(1/arg)

    def _eval_rewrite_as_acosh(self, arg, **kwargs):
        return I*(sqrt(1 - I/arg)/sqrt(I/arg - 1)*
                                acosh(I/arg, evaluate=False) - pi*S.Half)

    def _eval_rewrite_as_atanh(self, arg, **kwargs):
        arg2 = arg**2
        arg2p1 = arg2 + 1
        return sqrt(-arg2)/arg*(pi*S.Half -
                                sqrt(-arg2p1**2)/arg2p1*atanh(sqrt(arg2p1)))

    def _eval_is_zero(self):
        return self.args[0].is_infinite

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        return fuzzy_not(self.args[0].is_zero)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\modulargcd.py ===
from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
    gf_gcd, gf_from_dict, gf_gcdex, gf_div, gf_lcm)
from sympy.polys.polyerrors import ModularGCDFailed

from mpmath import sqrt
import random


def _trivial_gcd(f, g):
    """
    Compute the GCD of two polynomials in trivial cases, i.e. when one
    or both polynomials are zero.
    """
    ring = f.ring

    if not (f or g):
        return ring.zero, ring.zero, ring.zero
    elif not f:
        if g.LC < ring.domain.zero:
            return -g, ring.zero, -ring.one
        else:
            return g, ring.zero, ring.one
    elif not g:
        if f.LC < ring.domain.zero:
            return -f, -ring.one, ring.zero
        else:
            return f, ring.one, ring.zero
    return None


def _gf_gcd(fp, gp, p):
    r"""
    Compute the GCD of two univariate polynomials in `\mathbb{Z}_p[x]`.
    """
    dom = fp.ring.domain

    while gp:
        rem = fp
        deg = gp.degree()
        lcinv = dom.invert(gp.LC, p)

        while True:
            degrem = rem.degree()
            if degrem < deg:
                break
            rem = (rem - gp.mul_monom((degrem - deg,)).mul_ground(lcinv * rem.LC)).trunc_ground(p)

        fp = gp
        gp = rem

    return fp.mul_ground(dom.invert(fp.LC, p)).trunc_ground(p)


def _degree_bound_univariate(f, g):
    r"""
    Compute an upper bound for the degree of the GCD of two univariate
    integer polynomials `f` and `g`.

    The function chooses a suitable prime `p` and computes the GCD of
    `f` and `g` in `\mathbb{Z}_p[x]`. The choice of `p` guarantees that
    the degree in `\mathbb{Z}_p[x]` is greater than or equal to the degree
    in `\mathbb{Z}[x]`.

    Parameters
    ==========

    f : PolyElement
        univariate integer polynomial
    g : PolyElement
        univariate integer polynomial

    """
    gamma = f.ring.domain.gcd(f.LC, g.LC)
    p = 1

    p = nextprime(p)
    while gamma % p == 0:
        p = nextprime(p)

    fp = f.trunc_ground(p)
    gp = g.trunc_ground(p)
    hp = _gf_gcd(fp, gp, p)
    deghp = hp.degree()
    return deghp


def _chinese_remainder_reconstruction_univariate(hp, hq, p, q):
    r"""
    Construct a polynomial `h_{pq}` in `\mathbb{Z}_{p q}[x]` such that

    .. math ::

        h_{pq} = h_p \; \mathrm{mod} \, p

        h_{pq} = h_q \; \mathrm{mod} \, q

    for relatively prime integers `p` and `q` and polynomials
    `h_p` and `h_q` in `\mathbb{Z}_p[x]` and `\mathbb{Z}_q[x]`
    respectively.

    The coefficients of the polynomial `h_{pq}` are computed with the
    Chinese Remainder Theorem. The symmetric representation in
    `\mathbb{Z}_p[x]`, `\mathbb{Z}_q[x]` and `\mathbb{Z}_{p q}[x]` is used.
    It is assumed that `h_p` and `h_q` have the same degree.

    Parameters
    ==========

    hp : PolyElement
        univariate integer polynomial with coefficients in `\mathbb{Z}_p`
    hq : PolyElement
        univariate integer polynomial with coefficients in `\mathbb{Z}_q`
    p : Integer
        modulus of `h_p`, relatively prime to `q`
    q : Integer
        modulus of `h_q`, relatively prime to `p`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _chinese_remainder_reconstruction_univariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x = ring("x", ZZ)
    >>> p = 3
    >>> q = 5

    >>> hp = -x**3 - 1
    >>> hq = 2*x**3 - 2*x**2 + x

    >>> hpq = _chinese_remainder_reconstruction_univariate(hp, hq, p, q)
    >>> hpq
    2*x**3 + 3*x**2 + 6*x + 5

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    """
    n = hp.degree()
    x = hp.ring.gens[0]
    hpq = hp.ring.zero

    for i in range(n+1):
        hpq[(i,)] = crt([p, q], [hp.coeff(x**i), hq.coeff(x**i)], symmetric=True)[0]

    hpq.strip_zero()
    return hpq


def modgcd_univariate(f, g):
    r"""
    Computes the GCD of two polynomials in `\mathbb{Z}[x]` using a modular
    algorithm.

    The algorithm computes the GCD of two univariate integer polynomials
    `f` and `g` by computing the GCD in `\mathbb{Z}_p[x]` for suitable
    primes `p` and then reconstructing the coefficients with the Chinese
    Remainder Theorem. Trial division is only made for candidates which
    are very likely the desired GCD.

    Parameters
    ==========

    f : PolyElement
        univariate integer polynomial
    g : PolyElement
        univariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_univariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x = ring("x", ZZ)

    >>> f = x**5 - 1
    >>> g = x - 1

    >>> h, cff, cfg = modgcd_univariate(f, g)
    >>> h, cff, cfg
    (x - 1, x**4 + x**3 + x**2 + x + 1, 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = 6*x**2 - 6
    >>> g = 2*x**2 + 4*x + 2

    >>> h, cff, cfg = modgcd_univariate(f, g)
    >>> h, cff, cfg
    (2*x + 2, 3*x - 3, x + 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    ring = f.ring

    cf, f = f.primitive()
    cg, g = g.primitive()
    ch = ring.domain.gcd(cf, cg)

    bound = _degree_bound_univariate(f, g)
    if bound == 0:
        return ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch)

    gamma = ring.domain.gcd(f.LC, g.LC)
    m = 1
    p = 1

    while True:
        p = nextprime(p)
        while gamma % p == 0:
            p = nextprime(p)

        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        hp = _gf_gcd(fp, gp, p)
        deghp = hp.degree()

        if deghp > bound:
            continue
        elif deghp < bound:
            m = 1
            bound = deghp
            continue

        hp = hp.mul_ground(gamma).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue

        hm = _chinese_remainder_reconstruction_univariate(hp, hlastm, p, m)
        m *= p

        if not hm == hlastm:
            hlastm = hm
            continue

        h = hm.quo_ground(hm.content())
        fquo, frem = f.div(h)
        gquo, grem = g.div(h)
        if not frem and not grem:
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return h, cff, cfg


def _primitive(f, p):
    r"""
    Compute the content and the primitive part of a polynomial in
    `\mathbb{Z}_p[x_0, \ldots, x_{k-2}, y] \cong \mathbb{Z}_p[y][x_0, \ldots, x_{k-2}]`.

    Parameters
    ==========

    f : PolyElement
        integer polynomial in `\mathbb{Z}_p[x0, \ldots, x{k-2}, y]`
    p : Integer
        modulus of `f`

    Returns
    =======

    contf : PolyElement
        integer polynomial in `\mathbb{Z}_p[y]`, content of `f`
    ppf : PolyElement
        primitive part of `f`, i.e. `\frac{f}{contf}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _primitive
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)
    >>> p = 3

    >>> f = x**2*y**2 + x**2*y - y**2 - y
    >>> _primitive(f, p)
    (y**2 + y, x**2 - 1)

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x*y*z - y**2*z**2
    >>> _primitive(f, p)
    (z, x*y - y**2*z)

    """
    ring = f.ring
    dom = ring.domain
    k = ring.ngens

    coeffs = {}
    for monom, coeff in f.iterterms():
        if monom[:-1] not in coeffs:
            coeffs[monom[:-1]] = {}
        coeffs[monom[:-1]][monom[-1]] = coeff

    cont = []
    for coeff in iter(coeffs.values()):
        cont = gf_gcd(cont, gf_from_dict(coeff, p, dom), p, dom)

    yring = ring.clone(symbols=ring.symbols[k-1])
    contf = yring.from_dense(cont).trunc_ground(p)

    return contf, f.quo(contf.set_ring(ring))


def _deg(f):
    r"""
    Compute the degree of a multivariate polynomial
    `f \in K[x_0, \ldots, x_{k-2}, y] \cong K[y][x_0, \ldots, x_{k-2}]`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `K[x_0, \ldots, x_{k-2}, y]`

    Returns
    =======

    degf : Integer tuple
        degree of `f` in `x_0, \ldots, x_{k-2}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _deg
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _deg(f)
    (2,)

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _deg(f)
    (2, 2)

    >>> f = x*y*z - y**2*z**2
    >>> _deg(f)
    (1, 1)

    """
    k = f.ring.ngens
    degf = (0,) * (k-1)
    for monom in f.itermonoms():
        if monom[:-1] > degf:
            degf = monom[:-1]
    return degf


def _LC(f):
    r"""
    Compute the leading coefficient of a multivariate polynomial
    `f \in K[x_0, \ldots, x_{k-2}, y] \cong K[y][x_0, \ldots, x_{k-2}]`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `K[x_0, \ldots, x_{k-2}, y]`

    Returns
    =======

    lcf : PolyElement
        polynomial in `K[y]`, leading coefficient of `f`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _LC
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _LC(f)
    y**2 + y

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _LC(f)
    1

    >>> f = x*y*z - y**2*z**2
    >>> _LC(f)
    z

    """
    ring = f.ring
    k = ring.ngens
    yring = ring.clone(symbols=ring.symbols[k-1])
    y = yring.gens[0]
    degf = _deg(f)

    lcf = yring.zero
    for monom, coeff in f.iterterms():
        if monom[:-1] == degf:
            lcf += coeff*y**monom[-1]
    return lcf


def _swap(f, i):
    """
    Make the variable `x_i` the leading one in a multivariate polynomial `f`.
    """
    ring = f.ring
    fswap = ring.zero
    for monom, coeff in f.iterterms():
        monomswap = (monom[i],) + monom[:i] + monom[i+1:]
        fswap[monomswap] = coeff
    return fswap


def _degree_bound_bivariate(f, g):
    r"""
    Compute upper degree bounds for the GCD of two bivariate
    integer polynomials `f` and `g`.

    The GCD is viewed as a polynomial in `\mathbb{Z}[y][x]` and the
    function returns an upper bound for its degree and one for the degree
    of its content. This is done by choosing a suitable prime `p` and
    computing the GCD of the contents of `f \; \mathrm{mod} \, p` and
    `g \; \mathrm{mod} \, p`. The choice of `p` guarantees that the degree
    of the content in `\mathbb{Z}_p[y]` is greater than or equal to the
    degree in `\mathbb{Z}[y]`. To obtain the degree bound in the variable
    `x`, the polynomials are evaluated at `y = a` for a suitable
    `a \in \mathbb{Z}_p` and then their GCD in `\mathbb{Z}_p[x]` is
    computed. If no such `a` exists, i.e. the degree in `\mathbb{Z}_p[x]`
    is always smaller than the one in `\mathbb{Z}[y][x]`, then the bound is
    set to the minimum of the degrees of `f` and `g` in `x`.

    Parameters
    ==========

    f : PolyElement
        bivariate integer polynomial
    g : PolyElement
        bivariate integer polynomial

    Returns
    =======

    xbound : Integer
        upper bound for the degree of the GCD of the polynomials `f` and
        `g` in the variable `x`
    ycontbound : Integer
        upper bound for the degree of the content of the GCD of the
        polynomials `f` and `g` in the variable `y`

    References
    ==========

    1. [Monagan00]_

    """
    ring = f.ring

    gamma1 = ring.domain.gcd(f.LC, g.LC)
    gamma2 = ring.domain.gcd(_swap(f, 1).LC, _swap(g, 1).LC)
    badprimes = gamma1 * gamma2
    p = 1

    p = nextprime(p)
    while badprimes % p == 0:
        p = nextprime(p)

    fp = f.trunc_ground(p)
    gp = g.trunc_ground(p)
    contfp, fp = _primitive(fp, p)
    contgp, gp = _primitive(gp, p)
    conthp = _gf_gcd(contfp, contgp, p) # polynomial in Z_p[y]
    ycontbound = conthp.degree()

    # polynomial in Z_p[y]
    delta = _gf_gcd(_LC(fp), _LC(gp), p)

    for a in range(p):
        if not delta.evaluate(0, a) % p:
            continue
        fpa = fp.evaluate(1, a).trunc_ground(p)
        gpa = gp.evaluate(1, a).trunc_ground(p)
        hpa = _gf_gcd(fpa, gpa, p)
        xbound = hpa.degree()
        return xbound, ycontbound

    return min(fp.degree(), gp.degree()), ycontbound


def _chinese_remainder_reconstruction_multivariate(hp, hq, p, q):
    r"""
    Construct a polynomial `h_{pq}` in
    `\mathbb{Z}_{p q}[x_0, \ldots, x_{k-1}]` such that

    .. math ::

        h_{pq} = h_p \; \mathrm{mod} \, p

        h_{pq} = h_q \; \mathrm{mod} \, q

    for relatively prime integers `p` and `q` and polynomials
    `h_p` and `h_q` in `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]` and
    `\mathbb{Z}_q[x_0, \ldots, x_{k-1}]` respectively.

    The coefficients of the polynomial `h_{pq}` are computed with the
    Chinese Remainder Theorem. The symmetric representation in
    `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`,
    `\mathbb{Z}_q[x_0, \ldots, x_{k-1}]` and
    `\mathbb{Z}_{p q}[x_0, \ldots, x_{k-1}]` is used.

    Parameters
    ==========

    hp : PolyElement
        multivariate integer polynomial with coefficients in `\mathbb{Z}_p`
    hq : PolyElement
        multivariate integer polynomial with coefficients in `\mathbb{Z}_q`
    p : Integer
        modulus of `h_p`, relatively prime to `q`
    q : Integer
        modulus of `h_q`, relatively prime to `p`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _chinese_remainder_reconstruction_multivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)
    >>> p = 3
    >>> q = 5

    >>> hp = x**3*y - x**2 - 1
    >>> hq = -x**3*y - 2*x*y**2 + 2

    >>> hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)
    >>> hpq
    4*x**3*y + 5*x**2 + 3*x*y**2 + 2

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    >>> R, x, y, z = ring("x, y, z", ZZ)
    >>> p = 6
    >>> q = 5

    >>> hp = 3*x**4 - y**3*z + z
    >>> hq = -2*x**4 + z

    >>> hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)
    >>> hpq
    3*x**4 + 5*y**3*z + z

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    """
    hpmonoms = set(hp.monoms())
    hqmonoms = set(hq.monoms())
    monoms = hpmonoms.intersection(hqmonoms)
    hpmonoms.difference_update(monoms)
    hqmonoms.difference_update(monoms)

    domain = hp.ring.domain
    zero = domain.zero

    hpq = hp.ring.zero

    if isinstance(hp.ring.domain, PolynomialRing):
        crt_ = _chinese_remainder_reconstruction_multivariate
    else:
        def crt_(cp, cq, p, q):
            return domain(crt([p, q], [cp, cq], symmetric=True)[0])

    for monom in monoms:
        hpq[monom] = crt_(hp[monom], hq[monom], p, q)
    for monom in hpmonoms:
        hpq[monom] = crt_(hp[monom], zero, p, q)
    for monom in hqmonoms:
        hpq[monom] = crt_(zero, hq[monom], p, q)

    return hpq


def _interpolate_multivariate(evalpoints, hpeval, ring, i, p, ground=False):
    r"""
    Reconstruct a polynomial `h_p` in `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`
    from a list of evaluation points in `\mathbb{Z}_p` and a list of
    polynomials in
    `\mathbb{Z}_p[x_0, \ldots, x_{i-1}, x_{i+1}, \ldots, x_{k-1}]`, which
    are the images of `h_p` evaluated in the variable `x_i`.

    It is also possible to reconstruct a parameter of the ground domain,
    i.e. if `h_p` is a polynomial over `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`.
    In this case, one has to set ``ground=True``.

    Parameters
    ==========

    evalpoints : list of Integer objects
        list of evaluation points in `\mathbb{Z}_p`
    hpeval : list of PolyElement objects
        list of polynomials in (resp. over)
        `\mathbb{Z}_p[x_0, \ldots, x_{i-1}, x_{i+1}, \ldots, x_{k-1}]`,
        images of `h_p` evaluated in the variable `x_i`
    ring : PolyRing
        `h_p` will be an element of this ring
    i : Integer
        index of the variable which has to be reconstructed
    p : Integer
        prime number, modulus of `h_p`
    ground : Boolean
        indicates whether `x_i` is in the ground domain, default is
        ``False``

    Returns
    =======

    hp : PolyElement
        interpolated polynomial in (resp. over)
        `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`

    """
    hp = ring.zero

    if ground:
        domain = ring.domain.domain
        y = ring.domain.gens[i]
    else:
        domain = ring.domain
        y = ring.gens[i]

    for a, hpa in zip(evalpoints, hpeval):
        numer = ring.one
        denom = domain.one
        for b in evalpoints:
            if b == a:
                continue

            numer *= y - b
            denom *= a - b

        denom = domain.invert(denom, p)
        coeff = numer.mul_ground(denom)
        hp += hpa.set_ring(ring) * coeff

    return hp.trunc_ground(p)


def modgcd_bivariate(f, g):
    r"""
    Computes the GCD of two polynomials in `\mathbb{Z}[x, y]` using a
    modular algorithm.

    The algorithm computes the GCD of two bivariate integer polynomials
    `f` and `g` by calculating the GCD in `\mathbb{Z}_p[x, y]` for
    suitable primes `p` and then reconstructing the coefficients with the
    Chinese Remainder Theorem. To compute the bivariate GCD over
    `\mathbb{Z}_p`, the polynomials `f \; \mathrm{mod} \, p` and
    `g \; \mathrm{mod} \, p` are evaluated at `y = a` for certain
    `a \in \mathbb{Z}_p` and then their univariate GCD in `\mathbb{Z}_p[x]`
    is computed. Interpolating those yields the bivariate GCD in
    `\mathbb{Z}_p[x, y]`. To verify the result in `\mathbb{Z}[x, y]`, trial
    division is done, but only for candidates which are very likely the
    desired GCD.

    Parameters
    ==========

    f : PolyElement
        bivariate integer polynomial
    g : PolyElement
        bivariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_bivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2 - y**2
    >>> g = x**2 + 2*x*y + y**2

    >>> h, cff, cfg = modgcd_bivariate(f, g)
    >>> h, cff, cfg
    (x + y, x - y, x + y)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = x**2*y - x**2 - 4*y + 4
    >>> g = x + 2

    >>> h, cff, cfg = modgcd_bivariate(f, g)
    >>> h, cff, cfg
    (x + 2, x*y - x - 2*y + 2, 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    ring = f.ring

    cf, f = f.primitive()
    cg, g = g.primitive()
    ch = ring.domain.gcd(cf, cg)

    xbound, ycontbound = _degree_bound_bivariate(f, g)
    if xbound == ycontbound == 0:
        return ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch)

    fswap = _swap(f, 1)
    gswap = _swap(g, 1)
    degyf = fswap.degree()
    degyg = gswap.degree()

    ybound, xcontbound = _degree_bound_bivariate(fswap, gswap)
    if ybound == xcontbound == 0:
        return ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch)

    # TODO: to improve performance, choose the main variable here

    gamma1 = ring.domain.gcd(f.LC, g.LC)
    gamma2 = ring.domain.gcd(fswap.LC, gswap.LC)
    badprimes = gamma1 * gamma2
    m = 1
    p = 1

    while True:
        p = nextprime(p)
        while badprimes % p == 0:
            p = nextprime(p)

        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        contfp, fp = _primitive(fp, p)
        contgp, gp = _primitive(gp, p)
        conthp = _gf_gcd(contfp, contgp, p) # monic polynomial in Z_p[y]
        degconthp = conthp.degree()

        if degconthp > ycontbound:
            continue
        elif degconthp < ycontbound:
            m = 1
            ycontbound = degconthp
            continue

        # polynomial in Z_p[y]
        delta = _gf_gcd(_LC(fp), _LC(gp), p)

        degcontfp = contfp.degree()
        degcontgp = contgp.degree()
        degdelta = delta.degree()

        N = min(degyf - degcontfp, degyg - degcontgp,
            ybound - ycontbound + degdelta) + 1

        if p < N:
            continue

        n = 0
        evalpoints = []
        hpeval = []
        unlucky = False

        for a in range(p):
            deltaa = delta.evaluate(0, a)
            if not deltaa % p:
                continue

            fpa = fp.evaluate(1, a).trunc_ground(p)
            gpa = gp.evaluate(1, a).trunc_ground(p)
            hpa = _gf_gcd(fpa, gpa, p) # monic polynomial in Z_p[x]
            deghpa = hpa.degree()

            if deghpa > xbound:
                continue
            elif deghpa < xbound:
                m = 1
                xbound = deghpa
                unlucky = True
                break

            hpa = hpa.mul_ground(deltaa).trunc_ground(p)
            evalpoints.append(a)
            hpeval.append(hpa)
            n += 1

            if n == N:
                break

        if unlucky:
            continue
        if n < N:
            continue

        hp = _interpolate_multivariate(evalpoints, hpeval, ring, 1, p)

        hp = _primitive(hp, p)[1]
        hp = hp * conthp.set_ring(ring)
        degyhp = hp.degree(1)

        if degyhp > ybound:
            continue
        if degyhp < ybound:
            m = 1
            ybound = degyhp
            continue

        hp = hp.mul_ground(gamma1).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue

        hm = _chinese_remainder_reconstruction_multivariate(hp, hlastm, p, m)
        m *= p

        if not hm == hlastm:
            hlastm = hm
            continue

        h = hm.quo_ground(hm.content())
        fquo, frem = f.div(h)
        gquo, grem = g.div(h)
        if not frem and not grem:
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return h, cff, cfg


def _modgcd_multivariate_p(f, g, p, degbound, contbound):
    r"""
    Compute the GCD of two polynomials in
    `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`.

    The algorithm reduces the problem step by step by evaluating the
    polynomials `f` and `g` at `x_{k-1} = a` for suitable
    `a \in \mathbb{Z}_p` and then calls itself recursively to compute the GCD
    in `\mathbb{Z}_p[x_0, \ldots, x_{k-2}]`. If these recursive calls are
    successful for enough evaluation points, the GCD in `k` variables is
    interpolated, otherwise the algorithm returns ``None``. Every time a GCD
    or a content is computed, their degrees are compared with the bounds. If
    a degree greater then the bound is encountered, then the current call
    returns ``None`` and a new evaluation point has to be chosen. If at some
    point the degree is smaller, the correspondent bound is updated and the
    algorithm fails.

    Parameters
    ==========

    f : PolyElement
        multivariate integer polynomial with coefficients in `\mathbb{Z}_p`
    g : PolyElement
        multivariate integer polynomial with coefficients in `\mathbb{Z}_p`
    p : Integer
        prime number, modulus of `f` and `g`
    degbound : list of Integer objects
        ``degbound[i]`` is an upper bound for the degree of the GCD of `f`
        and `g` in the variable `x_i`
    contbound : list of Integer objects
        ``contbound[i]`` is an upper bound for the degree of the content of
        the GCD in `\mathbb{Z}_p[x_i][x_0, \ldots, x_{i-1}]`,
        ``contbound[0]`` is not used can therefore be chosen
        arbitrarily.

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g` or ``None``

    References
    ==========

    1. [Monagan00]_
    2. [Brown71]_

    """
    ring = f.ring
    k = ring.ngens

    if k == 1:
        h = _gf_gcd(f, g, p).trunc_ground(p)
        degh = h.degree()

        if degh > degbound[0]:
            return None
        if degh < degbound[0]:
            degbound[0] = degh
            raise ModularGCDFailed

        return h

    degyf = f.degree(k-1)
    degyg = g.degree(k-1)

    contf, f = _primitive(f, p)
    contg, g = _primitive(g, p)

    conth = _gf_gcd(contf, contg, p) # polynomial in Z_p[y]

    degcontf = contf.degree()
    degcontg = contg.degree()
    degconth = conth.degree()

    if degconth > contbound[k-1]:
        return None
    if degconth < contbound[k-1]:
        contbound[k-1] = degconth
        raise ModularGCDFailed

    lcf = _LC(f)
    lcg = _LC(g)

    delta = _gf_gcd(lcf, lcg, p) # polynomial in Z_p[y]

    evaltest = delta

    for i in range(k-1):
        evaltest *= _gf_gcd(_LC(_swap(f, i)), _LC(_swap(g, i)), p)

    degdelta = delta.degree()

    N = min(degyf - degcontf, degyg - degcontg,
            degbound[k-1] - contbound[k-1] + degdelta) + 1

    if p < N:
        return None

    n = 0
    d = 0
    evalpoints = []
    heval = []
    points = list(range(p))

    while points:
        a = random.sample(points, 1)[0]
        points.remove(a)

        if not evaltest.evaluate(0, a) % p:
            continue

        deltaa = delta.evaluate(0, a) % p

        fa = f.evaluate(k-1, a).trunc_ground(p)
        ga = g.evaluate(k-1, a).trunc_ground(p)

        # polynomials in Z_p[x_0, ..., x_{k-2}]
        ha = _modgcd_multivariate_p(fa, ga, p, degbound, contbound)

        if ha is None:
            d += 1
            if d > n:
                return None
            continue

        if ha.is_ground:
            h = conth.set_ring(ring).trunc_ground(p)
            return h

        ha = ha.mul_ground(deltaa).trunc_ground(p)

        evalpoints.append(a)
        heval.append(ha)
        n += 1

        if n == N:
            h = _interpolate_multivariate(evalpoints, heval, ring, k-1, p)

            h = _primitive(h, p)[1] * conth.set_ring(ring)
            degyh = h.degree(k-1)

            if degyh > degbound[k-1]:
                return None
            if degyh < degbound[k-1]:
                degbound[k-1] = degyh
                raise ModularGCDFailed

            return h

    return None


def modgcd_multivariate(f, g):
    r"""
    Compute the GCD of two polynomials in `\mathbb{Z}[x_0, \ldots, x_{k-1}]`
    using a modular algorithm.

    The algorithm computes the GCD of two multivariate integer polynomials
    `f` and `g` by calculating the GCD in
    `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]` for suitable primes `p` and then
    reconstructing the coefficients with the Chinese Remainder Theorem. To
    compute the multivariate GCD over `\mathbb{Z}_p` the recursive
    subroutine :func:`_modgcd_multivariate_p` is used. To verify the result in
    `\mathbb{Z}[x_0, \ldots, x_{k-1}]`, trial division is done, but only for
    candidates which are very likely the desired GCD.

    Parameters
    ==========

    f : PolyElement
        multivariate integer polynomial
    g : PolyElement
        multivariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_multivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2 - y**2
    >>> g = x**2 + 2*x*y + y**2

    >>> h, cff, cfg = modgcd_multivariate(f, g)
    >>> h, cff, cfg
    (x + y, x - y, x + y)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x*z**2 - y*z**2
    >>> g = x**2*z + z

    >>> h, cff, cfg = modgcd_multivariate(f, g)
    >>> h, cff, cfg
    (z, x*z - y*z, x**2 + 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_
    2. [Brown71]_

    See also
    ========

    _modgcd_multivariate_p

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    ring = f.ring
    k = ring.ngens

    # divide out integer content
    cf, f = f.primitive()
    cg, g = g.primitive()
    ch = ring.domain.gcd(cf, cg)

    gamma = ring.domain.gcd(f.LC, g.LC)

    badprimes = ring.domain.one
    for i in range(k):
        badprimes *= ring.domain.gcd(_swap(f, i).LC, _swap(g, i).LC)

    degbound = [min(fdeg, gdeg) for fdeg, gdeg in zip(f.degrees(), g.degrees())]
    contbound = list(degbound)

    m = 1
    p = 1

    while True:
        p = nextprime(p)
        while badprimes % p == 0:
            p = nextprime(p)

        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)

        try:
            # monic GCD of fp, gp in Z_p[x_0, ..., x_{k-2}, y]
            hp = _modgcd_multivariate_p(fp, gp, p, degbound, contbound)
        except ModularGCDFailed:
            m = 1
            continue

        if hp is None:
            continue

        hp = hp.mul_ground(gamma).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue

        hm = _chinese_remainder_reconstruction_multivariate(hp, hlastm, p, m)
        m *= p

        if not hm == hlastm:
            hlastm = hm
            continue

        h = hm.primitive()[1]
        fquo, frem = f.div(h)
        gquo, grem = g.div(h)
        if not frem and not grem:
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return h, cff, cfg


def _gf_div(f, g, p):
    r"""
    Compute `\frac f g` modulo `p` for two univariate polynomials over
    `\mathbb Z_p`.
    """
    ring = f.ring
    densequo, denserem = gf_div(f.to_dense(), g.to_dense(), p, ring.domain)
    return ring.from_dense(densequo), ring.from_dense(denserem)


def _rational_function_reconstruction(c, p, m):
    r"""
    Reconstruct a rational function `\frac a b` in `\mathbb Z_p(t)` from

    .. math::

        c = \frac a b \; \mathrm{mod} \, m,

    where `c` and `m` are polynomials in `\mathbb Z_p[t]` and `m` has
    positive degree.

    The algorithm is based on the Euclidean Algorithm. In general, `m` is
    not irreducible, so it is possible that `b` is not invertible modulo
    `m`. In that case ``None`` is returned.

    Parameters
    ==========

    c : PolyElement
        univariate polynomial in `\mathbb Z[t]`
    p : Integer
        prime number
    m : PolyElement
        modulus, not necessarily irreducible

    Returns
    =======

    frac : FracElement
        either `\frac a b` in `\mathbb Z(t)` or ``None``

    References
    ==========

    1. [Hoeij04]_

    """
    ring = c.ring
    domain = ring.domain
    M = m.degree()
    N = M // 2
    D = M - N - 1

    r0, s0 = m, ring.zero
    r1, s1 = c, ring.one

    while r1.degree() > N:
        quo = _gf_div(r0, r1, p)[0]
        r0, r1 = r1, (r0 - quo*r1).trunc_ground(p)
        s0, s1 = s1, (s0 - quo*s1).trunc_ground(p)

    a, b = r1, s1
    if b.degree() > D or _gf_gcd(b, m, p) != 1:
        return None

    lc = b.LC
    if lc != 1:
        lcinv = domain.invert(lc, p)
        a = a.mul_ground(lcinv).trunc_ground(p)
        b = b.mul_ground(lcinv).trunc_ground(p)

    field = ring.to_field()

    return field(a) / field(b)


def _rational_reconstruction_func_coeffs(hm, p, m, ring, k):
    r"""
    Reconstruct every coefficient `c_h` of a polynomial `h` in
    `\mathbb Z_p(t_k)[t_1, \ldots, t_{k-1}][x, z]` from the corresponding
    coefficient `c_{h_m}` of a polynomial `h_m` in
    `\mathbb Z_p[t_1, \ldots, t_k][x, z] \cong \mathbb Z_p[t_k][t_1, \ldots, t_{k-1}][x, z]`
    such that

    .. math::

        c_{h_m} = c_h \; \mathrm{mod} \, m,

    where `m \in \mathbb Z_p[t]`.

    The reconstruction is based on the Euclidean Algorithm. In general, `m`
    is not irreducible, so it is possible that this fails for some
    coefficient. In that case ``None`` is returned.

    Parameters
    ==========

    hm : PolyElement
        polynomial in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    p : Integer
        prime number, modulus of `\mathbb Z_p`
    m : PolyElement
        modulus, polynomial in `\mathbb Z[t]`, not necessarily irreducible
    ring : PolyRing
        `\mathbb Z(t_k)[t_1, \ldots, t_{k-1}][x, z]`, `h` will be an
        element of this ring
    k : Integer
        index of the parameter `t_k` which will be reconstructed

    Returns
    =======

    h : PolyElement
        reconstructed polynomial in
        `\mathbb Z(t_k)[t_1, \ldots, t_{k-1}][x, z]` or ``None``

    See also
    ========

    _rational_function_reconstruction

    """
    h = ring.zero

    for monom, coeff in hm.iterterms():
        if k == 0:
            coeffh = _rational_function_reconstruction(coeff, p, m)

            if not coeffh:
                return None

        else:
            coeffh = ring.domain.zero
            for mon, c in coeff.drop_to_ground(k).iterterms():
                ch = _rational_function_reconstruction(c, p, m)

                if not ch:
                    return None

                coeffh[mon] = ch

        h[monom] = coeffh

    return h


def _gf_gcdex(f, g, p):
    r"""
    Extended Euclidean Algorithm for two univariate polynomials over
    `\mathbb Z_p`.

    Returns polynomials `s, t` and `h`, such that `h` is the GCD of `f` and
    `g` and `sf + tg = h \; \mathrm{mod} \, p`.

    """
    ring = f.ring
    s, t, h = gf_gcdex(f.to_dense(), g.to_dense(), p, ring.domain)
    return ring.from_dense(s), ring.from_dense(t), ring.from_dense(h)


def _trunc(f, minpoly, p):
    r"""
    Compute the reduced representation of a polynomial `f` in
    `\mathbb Z_p[z] / (\check m_{\alpha}(z))[x]`

    Parameters
    ==========

    f : PolyElement
        polynomial in `\mathbb Z[x, z]`
    minpoly : PolyElement
        polynomial `\check m_{\alpha} \in \mathbb Z[z]`, not necessarily
        irreducible
    p : Integer
        prime number, modulus of `\mathbb Z_p`

    Returns
    =======

    ftrunc : PolyElement
        polynomial in `\mathbb Z[x, z]`, reduced modulo
        `\check m_{\alpha}(z)` and `p`

    """
    ring = f.ring
    minpoly = minpoly.set_ring(ring)
    p_ = ring.ground_new(p)

    return f.trunc_ground(p).rem([minpoly, p_]).trunc_ground(p)


def _euclidean_algorithm(f, g, minpoly, p):
    r"""
    Compute the monic GCD of two univariate polynomials in
    `\mathbb{Z}_p[z]/(\check m_{\alpha}(z))[x]` with the Euclidean
    Algorithm.

    In general, `\check m_{\alpha}(z)` is not irreducible, so it is possible
    that some leading coefficient is not invertible modulo
    `\check m_{\alpha}(z)`. In that case ``None`` is returned.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\mathbb Z[x, z]`
    minpoly : PolyElement
        polynomial in `\mathbb Z[z]`, not necessarily irreducible
    p : Integer
        prime number, modulus of `\mathbb Z_p`

    Returns
    =======

    h : PolyElement
        GCD of `f` and `g` in `\mathbb Z[z, x]` or ``None``, coefficients
        are in `\left[ -\frac{p-1} 2, \frac{p-1} 2 \right]`

    """
    ring = f.ring

    f = _trunc(f, minpoly, p)
    g = _trunc(g, minpoly, p)

    while g:
        rem = f
        deg = g.degree(0) # degree in x
        lcinv, _, gcd = _gf_gcdex(ring.dmp_LC(g), minpoly, p)

        if not gcd == 1:
            return None

        while True:
            degrem = rem.degree(0) # degree in x
            if degrem < deg:
                break
            quo = (lcinv * ring.dmp_LC(rem)).set_ring(ring)
            rem = _trunc(rem - g.mul_monom((degrem - deg, 0))*quo, minpoly, p)

        f = g
        g = rem

    lcfinv = _gf_gcdex(ring.dmp_LC(f), minpoly, p)[0].set_ring(ring)

    return _trunc(f * lcfinv, minpoly, p)


def _trial_division(f, h, minpoly, p=None):
    r"""
    Check if `h` divides `f` in
    `\mathbb K[t_1, \ldots, t_k][z]/(m_{\alpha}(z))`, where `\mathbb K` is
    either `\mathbb Q` or `\mathbb Z_p`.

    This algorithm is based on pseudo division and does not use any
    fractions. By default `\mathbb K` is `\mathbb Q`, if a prime number `p`
    is given, `\mathbb Z_p` is chosen instead.

    Parameters
    ==========

    f, h : PolyElement
        polynomials in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    minpoly : PolyElement
        polynomial `m_{\alpha}(z)` in `\mathbb Z[t_1, \ldots, t_k][z]`
    p : Integer or None
        if `p` is given, `\mathbb K` is set to `\mathbb Z_p` instead of
        `\mathbb Q`, default is ``None``

    Returns
    =======

    rem : PolyElement
        remainder of `\frac f h`

    References
    ==========

    .. [1] [Hoeij02]_

    """
    ring = f.ring

    zxring = ring.clone(symbols=(ring.symbols[1], ring.symbols[0]))

    minpoly = minpoly.set_ring(ring)

    rem = f

    degrem = rem.degree()
    degh = h.degree()
    degm = minpoly.degree(1)

    lch = _LC(h).set_ring(ring)
    lcm = minpoly.LC

    while rem and degrem >= degh:
        # polynomial in Z[t_1, ..., t_k][z]
        lcrem = _LC(rem).set_ring(ring)
        rem = rem*lch - h.mul_monom((degrem - degh, 0))*lcrem
        if p:
            rem = rem.trunc_ground(p)
        degrem = rem.degree(1)

        while rem and degrem >= degm:
            # polynomial in Z[t_1, ..., t_k][x]
            lcrem = _LC(rem.set_ring(zxring)).set_ring(ring)
            rem = rem.mul_ground(lcm) - minpoly.mul_monom((0, degrem - degm))*lcrem
            if p:
                rem = rem.trunc_ground(p)
            degrem = rem.degree(1)

        degrem = rem.degree()

    return rem


def _evaluate_ground(f, i, a):
    r"""
    Evaluate a polynomial `f` at `a` in the `i`-th variable of the ground
    domain.
    """
    ring = f.ring.clone(domain=f.ring.domain.ring.drop(i))
    fa = ring.zero

    for monom, coeff in f.iterterms():
        fa[monom] = coeff.evaluate(i, a)

    return fa


def _func_field_modgcd_p(f, g, minpoly, p):
    r"""
    Compute the GCD of two polynomials `f` and `g` in
    `\mathbb Z_p(t_1, \ldots, t_k)[z]/(\check m_\alpha(z))[x]`.

    The algorithm reduces the problem step by step by evaluating the
    polynomials `f` and `g` at `t_k = a` for suitable `a \in \mathbb Z_p`
    and then calls itself recursively to compute the GCD in
    `\mathbb Z_p(t_1, \ldots, t_{k-1})[z]/(\check m_\alpha(z))[x]`. If these
    recursive calls are successful, the GCD over `k` variables is
    interpolated, otherwise the algorithm returns ``None``. After
    interpolation, Rational Function Reconstruction is used to obtain the
    correct coefficients. If this fails, a new evaluation point has to be
    chosen, otherwise the desired polynomial is obtained by clearing
    denominators. The result is verified with a fraction free trial
    division.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    minpoly : PolyElement
        polynomial in `\mathbb Z[t_1, \ldots, t_k][z]`, not necessarily
        irreducible
    p : Integer
        prime number, modulus of `\mathbb Z_p`

    Returns
    =======

    h : PolyElement
        primitive associate in `\mathbb Z[t_1, \ldots, t_k][x, z]` of the
        GCD of the polynomials `f` and `g`  or ``None``, coefficients are
        in `\left[ -\frac{p-1} 2, \frac{p-1} 2 \right]`

    References
    ==========

    1. [Hoeij04]_

    """
    ring = f.ring
    domain = ring.domain # Z[t_1, ..., t_k]

    if isinstance(domain, PolynomialRing):
        k = domain.ngens
    else:
        return _euclidean_algorithm(f, g, minpoly, p)

    if k == 1:
        qdomain = domain.ring.to_field()
    else:
        qdomain = domain.ring.drop_to_ground(k - 1)
        qdomain = qdomain.clone(domain=qdomain.domain.ring.to_field())

    qring = ring.clone(domain=qdomain) # = Z(t_k)[t_1, ..., t_{k-1}][x, z]

    n = 1
    d = 1

    # polynomial in Z_p[t_1, ..., t_k][z]
    gamma = ring.dmp_LC(f) * ring.dmp_LC(g)
    # polynomial in Z_p[t_1, ..., t_k]
    delta = minpoly.LC

    evalpoints = []
    heval = []
    LMlist = []
    points = list(range(p))

    while points:
        a = random.sample(points, 1)[0]
        points.remove(a)

        if k == 1:
            test = delta.evaluate(k-1, a) % p == 0
        else:
            test = delta.evaluate(k-1, a).trunc_ground(p) == 0

        if test:
            continue

        gammaa = _evaluate_ground(gamma, k-1, a)
        minpolya = _evaluate_ground(minpoly, k-1, a)

        if gammaa.rem([minpolya, gammaa.ring(p)]) == 0:
            continue

        fa = _evaluate_ground(f, k-1, a)
        ga = _evaluate_ground(g, k-1, a)

        # polynomial in Z_p[x, t_1, ..., t_{k-1}, z]/(minpoly)
        ha = _func_field_modgcd_p(fa, ga, minpolya, p)

        if ha is None:
            d += 1
            if d > n:
                return None
            continue

        if ha == 1:
            return ha

        LM = [ha.degree()] + [0]*(k-1)
        if k > 1:
            for monom, coeff in ha.iterterms():
                if monom[0] == LM[0] and coeff.LM > tuple(LM[1:]):
                    LM[1:] = coeff.LM

        evalpoints_a = [a]
        heval_a = [ha]
        if k == 1:
            m = qring.domain.get_ring().one
        else:
            m = qring.domain.domain.get_ring().one

        t = m.ring.gens[0]

        for b, hb, LMhb in zip(evalpoints, heval, LMlist):
            if LMhb == LM:
                evalpoints_a.append(b)
                heval_a.append(hb)
                m *= (t - b)

        m = m.trunc_ground(p)
        evalpoints.append(a)
        heval.append(ha)
        LMlist.append(LM)
        n += 1

        # polynomial in Z_p[t_1, ..., t_k][x, z]
        h = _interpolate_multivariate(evalpoints_a, heval_a, ring, k-1, p, ground=True)

        # polynomial in Z_p(t_k)[t_1, ..., t_{k-1}][x, z]
        h = _rational_reconstruction_func_coeffs(h, p, m, qring, k-1)

        if h is None:
            continue

        if k == 1:
            dom = qring.domain.field
            den = dom.ring.one

            for coeff in h.itercoeffs():
                den = dom.ring.from_dense(gf_lcm(den.to_dense(), coeff.denom.to_dense(),
                        p, dom.domain))

        else:
            dom = qring.domain.domain.field
            den = dom.ring.one

            for coeff in h.itercoeffs():
                for c in coeff.itercoeffs():
                    den = dom.ring.from_dense(gf_lcm(den.to_dense(), c.denom.to_dense(),
                            p, dom.domain))

        den = qring.domain_new(den.trunc_ground(p))
        h = ring(h.mul_ground(den).as_expr()).trunc_ground(p)

        if not _trial_division(f, h, minpoly, p) and not _trial_division(g, h, minpoly, p):
            return h

    return None


def _integer_rational_reconstruction(c, m, domain):
    r"""
    Reconstruct a rational number `\frac a b` from

    .. math::

        c = \frac a b \; \mathrm{mod} \, m,

    where `c` and `m` are integers.

    The algorithm is based on the Euclidean Algorithm. In general, `m` is
    not a prime number, so it is possible that `b` is not invertible modulo
    `m`. In that case ``None`` is returned.

    Parameters
    ==========

    c : Integer
        `c = \frac a b \; \mathrm{mod} \, m`
    m : Integer
        modulus, not necessarily prime
    domain : IntegerRing
        `a, b, c` are elements of ``domain``

    Returns
    =======

    frac : Rational
        either `\frac a b` in `\mathbb Q` or ``None``

    References
    ==========

    1. [Wang81]_

    """
    if c < 0:
        c += m

    r0, s0 = m, domain.zero
    r1, s1 = c, domain.one

    bound = sqrt(m / 2) # still correct if replaced by ZZ.sqrt(m // 2) ?

    while int(r1) >= bound:
        quo = r0 // r1
        r0, r1 = r1, r0 - quo*r1
        s0, s1 = s1, s0 - quo*s1

    if abs(int(s1)) >= bound:
        return None

    if s1 < 0:
        a, b = -r1, -s1
    elif s1 > 0:
        a, b = r1, s1
    else:
        return None

    field = domain.get_field()

    return field(a) / field(b)


def _rational_reconstruction_int_coeffs(hm, m, ring):
    r"""
    Reconstruct every rational coefficient `c_h` of a polynomial `h` in
    `\mathbb Q[t_1, \ldots, t_k][x, z]` from the corresponding integer
    coefficient `c_{h_m}` of a polynomial `h_m` in
    `\mathbb Z[t_1, \ldots, t_k][x, z]` such that

    .. math::

        c_{h_m} = c_h \; \mathrm{mod} \, m,

    where `m \in \mathbb Z`.

    The reconstruction is based on the Euclidean Algorithm. In general,
    `m` is not a prime number, so it is possible that this fails for some
    coefficient. In that case ``None`` is returned.

    Parameters
    ==========

    hm : PolyElement
        polynomial in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    m : Integer
        modulus, not necessarily prime
    ring : PolyRing
        `\mathbb Q[t_1, \ldots, t_k][x, z]`, `h` will be an element of this
        ring

    Returns
    =======

    h : PolyElement
        reconstructed polynomial in `\mathbb Q[t_1, \ldots, t_k][x, z]` or
        ``None``

    See also
    ========

    _integer_rational_reconstruction

    """
    h = ring.zero

    if isinstance(ring.domain, PolynomialRing):
        reconstruction = _rational_reconstruction_int_coeffs
        domain = ring.domain.ring
    else:
        reconstruction = _integer_rational_reconstruction
        domain = hm.ring.domain

    for monom, coeff in hm.iterterms():
        coeffh = reconstruction(coeff, m, domain)

        if not coeffh:
            return None

        h[monom] = coeffh

    return h


def _func_field_modgcd_m(f, g, minpoly):
    r"""
    Compute the GCD of two polynomials in
    `\mathbb Q(t_1, \ldots, t_k)[z]/(m_{\alpha}(z))[x]` using a modular
    algorithm.

    The algorithm computes the GCD of two polynomials `f` and `g` by
    calculating the GCD in
    `\mathbb Z_p(t_1, \ldots, t_k)[z] / (\check m_{\alpha}(z))[x]` for
    suitable primes `p` and the primitive associate `\check m_{\alpha}(z)`
    of `m_{\alpha}(z)`. Then the coefficients are reconstructed with the
    Chinese Remainder Theorem and Rational Reconstruction. To compute the
    GCD over `\mathbb Z_p(t_1, \ldots, t_k)[z] / (\check m_{\alpha})[x]`,
    the recursive subroutine ``_func_field_modgcd_p`` is used. To verify the
    result in `\mathbb Q(t_1, \ldots, t_k)[z] / (m_{\alpha}(z))[x]`, a
    fraction free trial division is used.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    minpoly : PolyElement
        irreducible polynomial in `\mathbb Z[t_1, \ldots, t_k][z]`

    Returns
    =======

    h : PolyElement
        the primitive associate in `\mathbb Z[t_1, \ldots, t_k][x, z]` of
        the GCD of `f` and `g`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _func_field_modgcd_m
    >>> from sympy.polys import ring, ZZ

    >>> R, x, z = ring('x, z', ZZ)
    >>> minpoly = (z**2 - 2).drop(0)

    >>> f = x**2 + 2*x*z + 2
    >>> g = x + z
    >>> _func_field_modgcd_m(f, g, minpoly)
    x + z

    >>> D, t = ring('t', ZZ)
    >>> R, x, z = ring('x, z', D)
    >>> minpoly = (z**2-3).drop(0)

    >>> f = x**2 + (t + 1)*x*z + 3*t
    >>> g = x*z + 3*t
    >>> _func_field_modgcd_m(f, g, minpoly)
    x + t*z

    References
    ==========

    1. [Hoeij04]_

    See also
    ========

    _func_field_modgcd_p

    """
    ring = f.ring
    domain = ring.domain

    if isinstance(domain, PolynomialRing):
        k = domain.ngens
        QQdomain = domain.ring.clone(domain=domain.domain.get_field())
        QQring = ring.clone(domain=QQdomain)
    else:
        k = 0
        QQring = ring.clone(domain=ring.domain.get_field())

    cf, f = f.primitive()
    cg, g = g.primitive()

    # polynomial in Z[t_1, ..., t_k][z]
    gamma = ring.dmp_LC(f) * ring.dmp_LC(g)
    # polynomial in Z[t_1, ..., t_k]
    delta = minpoly.LC

    p = 1
    primes = []
    hplist = []
    LMlist = []

    while True:
        p = nextprime(p)

        if gamma.trunc_ground(p) == 0:
            continue

        if k == 0:
            test = (delta % p == 0)
        else:
            test = (delta.trunc_ground(p) == 0)

        if test:
            continue

        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        minpolyp = minpoly.trunc_ground(p)

        hp = _func_field_modgcd_p(fp, gp, minpolyp, p)

        if hp is None:
            continue

        if hp == 1:
            return ring.one

        LM = [hp.degree()] + [0]*k
        if k > 0:
            for monom, coeff in hp.iterterms():
                if monom[0] == LM[0] and coeff.LM > tuple(LM[1:]):
                    LM[1:] = coeff.LM

        hm = hp
        m = p

        for q, hq, LMhq in zip(primes, hplist, LMlist):
            if LMhq == LM:
                hm = _chinese_remainder_reconstruction_multivariate(hq, hm, q, m)
                m *= q

        primes.append(p)
        hplist.append(hp)
        LMlist.append(LM)

        hm = _rational_reconstruction_int_coeffs(hm, m, QQring)

        if hm is None:
            continue

        if k == 0:
            h = hm.clear_denoms()[1]
        else:
            den = domain.domain.one
            for coeff in hm.itercoeffs():
                den = domain.domain.lcm(den, coeff.clear_denoms()[0])
            h = hm.mul_ground(den)

        # convert back to Z[t_1, ..., t_k][x, z] from Q[t_1, ..., t_k][x, z]
        h = h.set_ring(ring)
        h = h.primitive()[1]

        if not (_trial_division(f.mul_ground(cf), h, minpoly) or
            _trial_division(g.mul_ground(cg), h, minpoly)):
            return h


def _to_ZZ_poly(f, ring):
    r"""
    Compute an associate of a polynomial
    `f \in \mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]` in
    `\mathbb Z[x_1, \ldots, x_{n-1}][z] / (\check m_{\alpha}(z))[x_0]`,
    where `\check m_{\alpha}(z) \in \mathbb Z[z]` is the primitive associate
    of the minimal polynomial `m_{\alpha}(z)` of `\alpha` over
    `\mathbb Q`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`
    ring : PolyRing
        `\mathbb Z[x_1, \ldots, x_{n-1}][x_0, z]`

    Returns
    =======

    f_ : PolyElement
        associate of `f` in
        `\mathbb Z[x_1, \ldots, x_{n-1}][x_0, z]`

    """
    f_ = ring.zero

    if isinstance(ring.domain, PolynomialRing):
        domain = ring.domain.domain
    else:
        domain = ring.domain

    den = domain.one

    for coeff in f.itercoeffs():
        for c in coeff.to_list():
            if c:
                den = domain.lcm(den, c.denominator)

    for monom, coeff in f.iterterms():
        coeff = coeff.to_list()
        m = ring.domain.one
        if isinstance(ring.domain, PolynomialRing):
            m = m.mul_monom(monom[1:])
        n = len(coeff)

        for i in range(n):
            if coeff[i]:
                c = domain.convert(coeff[i] * den) * m

                if (monom[0], n-i-1) not in f_:
                    f_[(monom[0], n-i-1)] = c
                else:
                    f_[(monom[0], n-i-1)] += c

    return f_


def _to_ANP_poly(f, ring):
    r"""
    Convert a polynomial
    `f \in \mathbb Z[x_1, \ldots, x_{n-1}][z]/(\check m_{\alpha}(z))[x_0]`
    to a polynomial in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`,
    where `\check m_{\alpha}(z) \in \mathbb Z[z]` is the primitive associate
    of the minimal polynomial `m_{\alpha}(z)` of `\alpha` over
    `\mathbb Q`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `\mathbb Z[x_1, \ldots, x_{n-1}][x_0, z]`
    ring : PolyRing
        `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`

    Returns
    =======

    f_ : PolyElement
        polynomial in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`

    """
    domain = ring.domain
    f_ = ring.zero

    if isinstance(f.ring.domain, PolynomialRing):
        for monom, coeff in f.iterterms():
            for mon, coef in coeff.iterterms():
                m = (monom[0],) + mon
                c = domain([domain.domain(coef)] + [0]*monom[1])

                if m not in f_:
                    f_[m] = c
                else:
                    f_[m] += c

    else:
        for monom, coeff in f.iterterms():
            m = (monom[0],)
            c = domain([domain.domain(coeff)] + [0]*monom[1])

            if m not in f_:
                f_[m] = c
            else:
                f_[m] += c

    return f_


def _minpoly_from_dense(minpoly, ring):
    r"""
    Change representation of the minimal polynomial from ``DMP`` to
    ``PolyElement`` for a given ring.
    """
    minpoly_ = ring.zero

    for monom, coeff in minpoly.terms():
        minpoly_[monom] = ring.domain(coeff)

    return minpoly_


def _primitive_in_x0(f):
    r"""
    Compute the content in `x_0` and the primitive part of a polynomial `f`
    in
    `\mathbb Q(\alpha)[x_0, x_1, \ldots, x_{n-1}] \cong \mathbb Q(\alpha)[x_1, \ldots, x_{n-1}][x_0]`.
    """
    fring = f.ring
    ring = fring.drop_to_ground(*range(1, fring.ngens))
    dom = ring.domain.ring
    f_ = ring(f.as_expr())
    cont = dom.zero

    for coeff in f_.itercoeffs():
        cont = func_field_modgcd(cont, coeff)[0]
        if cont == dom.one:
            return cont, f

    return cont, f.quo(cont.set_ring(fring))


# TODO: add support for algebraic function fields
def func_field_modgcd(f, g):
    r"""
    Compute the GCD of two polynomials `f` and `g` in
    `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]` using a modular algorithm.

    The algorithm first computes the primitive associate
    `\check m_{\alpha}(z)` of the minimal polynomial `m_{\alpha}` in
    `\mathbb{Z}[z]` and the primitive associates of `f` and `g` in
    `\mathbb{Z}[x_1, \ldots, x_{n-1}][z]/(\check m_{\alpha})[x_0]`. Then it
    computes the GCD in
    `\mathbb Q(x_1, \ldots, x_{n-1})[z]/(m_{\alpha}(z))[x_0]`.
    This is done by calculating the GCD in
    `\mathbb{Z}_p(x_1, \ldots, x_{n-1})[z]/(\check m_{\alpha}(z))[x_0]` for
    suitable primes `p` and then reconstructing the coefficients with the
    Chinese Remainder Theorem and Rational Reconstruction. The GCD over
    `\mathbb{Z}_p(x_1, \ldots, x_{n-1})[z]/(\check m_{\alpha}(z))[x_0]` is
    computed with a recursive subroutine, which evaluates the polynomials at
    `x_{n-1} = a` for suitable evaluation points `a \in \mathbb Z_p` and
    then calls itself recursively until the ground domain does no longer
    contain any parameters. For
    `\mathbb{Z}_p[z]/(\check m_{\alpha}(z))[x_0]` the Euclidean Algorithm is
    used. The results of those recursive calls are then interpolated and
    Rational Function Reconstruction is used to obtain the correct
    coefficients. The results, both in
    `\mathbb Q(x_1, \ldots, x_{n-1})[z]/(m_{\alpha}(z))[x_0]` and
    `\mathbb{Z}_p(x_1, \ldots, x_{n-1})[z]/(\check m_{\alpha}(z))[x_0]`, are
    verified by a fraction free trial division.

    Apart from the above GCD computation some GCDs in
    `\mathbb Q(\alpha)[x_1, \ldots, x_{n-1}]` have to be calculated,
    because treating the polynomials as univariate ones can result in
    a spurious content of the GCD. For this ``func_field_modgcd`` is
    called recursively.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`

    Returns
    =======

    h : PolyElement
        monic GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\frac f h`
    cfg : PolyElement
        cofactor of `g`, i.e. `\frac g h`

    Examples
    ========

    >>> from sympy.polys.modulargcd import func_field_modgcd
    >>> from sympy.polys import AlgebraicField, QQ, ring
    >>> from sympy import sqrt

    >>> A = AlgebraicField(QQ, sqrt(2))
    >>> R, x = ring('x', A)

    >>> f = x**2 - 2
    >>> g = x + sqrt(2)

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == x + sqrt(2)
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> R, x, y = ring('x, y', A)

    >>> f = x**2 + 2*sqrt(2)*x*y + 2*y**2
    >>> g = x + sqrt(2)*y

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == x + sqrt(2)*y
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = x + sqrt(2)*y
    >>> g = x + y

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == R.one
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Hoeij04]_

    """
    ring = f.ring
    domain = ring.domain
    n = ring.ngens

    assert ring == g.ring and domain.is_Algebraic

    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    z = Dummy('z')

    ZZring = ring.clone(symbols=ring.symbols + (z,), domain=domain.domain.get_ring())

    if n == 1:
        f_ = _to_ZZ_poly(f, ZZring)
        g_ = _to_ZZ_poly(g, ZZring)
        minpoly = ZZring.drop(0).from_dense(domain.mod.to_list())

        h = _func_field_modgcd_m(f_, g_, minpoly)
        h = _to_ANP_poly(h, ring)

    else:
        # contx0f in Q(a)[x_1, ..., x_{n-1}], f in Q(a)[x_0, ..., x_{n-1}]
        contx0f, f = _primitive_in_x0(f)
        contx0g, g = _primitive_in_x0(g)
        contx0h = func_field_modgcd(contx0f, contx0g)[0]

        ZZring_ = ZZring.drop_to_ground(*range(1, n))

        f_ = _to_ZZ_poly(f, ZZring_)
        g_ = _to_ZZ_poly(g, ZZring_)
        minpoly = _minpoly_from_dense(domain.mod, ZZring_.drop(0))

        h = _func_field_modgcd_m(f_, g_, minpoly)
        h = _to_ANP_poly(h, ring)

        contx0h_, h = _primitive_in_x0(h)
        h *= contx0h.set_ring(ring)
        f *= contx0f.set_ring(ring)
        g *= contx0g.set_ring(ring)

    h = h.quo_ground(h.LC)

    return h, f.quo(h), g.quo(h)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\diffgeom\diffgeom.py ===
from __future__ import annotations
from typing import Any

from functools import reduce
from itertools import permutations

from sympy.combinatorics import Permutation
from sympy.core import (
    Basic, Expr, Function, diff,
    Pow, Mul, Add, Lambda, S, Tuple, Dict
)
from sympy.core.cache import cacheit

from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve

from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning,
                                        ignore_warnings)


# TODO you are a bit excessive in the use of Dummies
# TODO dummy point, literal field
# TODO too often one needs to call doit or simplify on the output, check the
# tests and find out why
from sympy.tensor.array import ImmutableDenseNDimArray


class Manifold(Basic):
    """
    A mathematical manifold.

    Explanation
    ===========

    A manifold is a topological space that locally resembles
    Euclidean space near each point [1].
    This class does not provide any means to study the topological
    characteristics of the manifold that it represents, though.

    Parameters
    ==========

    name : str
        The name of the manifold.

    dim : int
        The dimension of the manifold.

    Examples
    ========

    >>> from sympy.diffgeom import Manifold
    >>> m = Manifold('M', 2)
    >>> m
    M
    >>> m.dim
    2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Manifold
    """

    def __new__(cls, name, dim, **kwargs):
        if not isinstance(name, Str):
            name = Str(name)
        dim = _sympify(dim)
        obj = super().__new__(cls, name, dim)

        obj.patches = _deprecated_list(
            """
            Manifold.patches is deprecated. The Manifold object is now
            immutable. Instead use a separate list to keep track of the
            patches.
            """, [])
        return obj

    @property
    def name(self):
        return self.args[0]

    @property
    def dim(self):
        return self.args[1]


class Patch(Basic):
    """
    A patch on a manifold.

    Explanation
    ===========

    Coordinate patch, or patch in short, is a simply-connected open set around
    a point in the manifold [1]. On a manifold one can have many patches that
    do not always include the whole manifold. On these patches coordinate
    charts can be defined that permit the parameterization of any point on the
    patch in terms of a tuple of real numbers (the coordinates).

    This class does not provide any means to study the topological
    characteristics of the patch that it represents.

    Parameters
    ==========

    name : str
        The name of the patch.

    manifold : Manifold
        The manifold on which the patch is defined.

    Examples
    ========

    >>> from sympy.diffgeom import Manifold, Patch
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> p
    P
    >>> p.dim
    2

    References
    ==========

    .. [1] G. Sussman, J. Wisdom, W. Farr, Functional Differential Geometry
           (2013)

    """
    def __new__(cls, name, manifold, **kwargs):
        if not isinstance(name, Str):
            name = Str(name)
        obj = super().__new__(cls, name, manifold)

        obj.manifold.patches.append(obj) # deprecated
        obj.coord_systems = _deprecated_list(
            """
            Patch.coord_systms is deprecated. The Patch class is now
            immutable. Instead use a separate list to keep track of coordinate
            systems.
            """, [])
        return obj

    @property
    def name(self):
        return self.args[0]

    @property
    def manifold(self):
        return self.args[1]

    @property
    def dim(self):
        return self.manifold.dim


class CoordSystem(Basic):
    """
    A coordinate system defined on the patch.

    Explanation
    ===========

    Coordinate system is a system that uses one or more coordinates to uniquely
    determine the position of the points or other geometric elements on a
    manifold [1].

    By passing ``Symbols`` to *symbols* parameter, user can define the name and
    assumptions of coordinate symbols of the coordinate system. If not passed,
    these symbols are generated automatically and are assumed to be real valued.

    By passing *relations* parameter, user can define the transform relations of
    coordinate systems. Inverse transformation and indirect transformation can
    be found automatically. If this parameter is not passed, coordinate
    transformation cannot be done.

    Parameters
    ==========

    name : str
        The name of the coordinate system.

    patch : Patch
        The patch where the coordinate system is defined.

    symbols : list of Symbols, optional
        Defines the names and assumptions of coordinate symbols.

    relations : dict, optional
        Key is a tuple of two strings, who are the names of the systems where
        the coordinates transform from and transform to.
        Value is a tuple of the symbols before transformation and a tuple of
        the expressions after transformation.

    Examples
    ========

    We define two-dimensional Cartesian coordinate system and polar coordinate
    system.

    >>> from sympy import symbols, pi, sqrt, atan2, cos, sin
    >>> from sympy.diffgeom import Manifold, Patch, CoordSystem
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> x, y = symbols('x y', real=True)
    >>> r, theta = symbols('r theta', nonnegative=True)
    >>> relation_dict = {
    ... ('Car2D', 'Pol'): [(x, y), (sqrt(x**2 + y**2), atan2(y, x))],
    ... ('Pol', 'Car2D'): [(r, theta), (r*cos(theta), r*sin(theta))]
    ... }
    >>> Car2D = CoordSystem('Car2D', p, (x, y), relation_dict)
    >>> Pol = CoordSystem('Pol', p, (r, theta), relation_dict)

    ``symbols`` property returns ``CoordinateSymbol`` instances. These symbols
    are not same with the symbols used to construct the coordinate system.

    >>> Car2D
    Car2D
    >>> Car2D.dim
    2
    >>> Car2D.symbols
    (x, y)
    >>> _[0].func
    <class 'sympy.diffgeom.diffgeom.CoordinateSymbol'>

    ``transformation()`` method returns the transformation function from
    one coordinate system to another. ``transform()`` method returns the
    transformed coordinates.

    >>> Car2D.transformation(Pol)
    Lambda((x, y), Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]]))
    >>> Car2D.transform(Pol)
    Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]])
    >>> Car2D.transform(Pol, [1, 2])
    Matrix([
    [sqrt(5)],
    [atan(2)]])

    ``jacobian()`` method returns the Jacobian matrix of coordinate
    transformation between two systems. ``jacobian_determinant()`` method
    returns the Jacobian determinant of coordinate transformation between two
    systems.

    >>> Pol.jacobian(Car2D)
    Matrix([
    [cos(theta), -r*sin(theta)],
    [sin(theta),  r*cos(theta)]])
    >>> Pol.jacobian(Car2D, [1, pi/2])
    Matrix([
    [0, -1],
    [1,  0]])
    >>> Car2D.jacobian_determinant(Pol)
    1/sqrt(x**2 + y**2)
    >>> Car2D.jacobian_determinant(Pol, [1,0])
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Coordinate_system

    """
    def __new__(cls, name, patch, symbols=None, relations={}, **kwargs):
        if not isinstance(name, Str):
            name = Str(name)

        # canonicallize the symbols
        if symbols is None:
            names = kwargs.get('names', None)
            if names is None:
                symbols = Tuple(
                    *[Symbol('%s_%s' % (name.name, i), real=True)
                      for i in range(patch.dim)]
                )
            else:
                sympy_deprecation_warning(
                    f"""
The 'names' argument to CoordSystem is deprecated. Use 'symbols' instead. That
is, replace

    CoordSystem(..., names={names})

with

    CoordSystem(..., symbols=[{', '.join(["Symbol(" + repr(n) + ", real=True)" for n in names])}])
                    """,
                    deprecated_since_version="1.7",
                    active_deprecations_target="deprecated-diffgeom-mutable",
                )
                symbols = Tuple(
                    *[Symbol(n, real=True) for n in names]
                )
        else:
            syms = []
            for s in symbols:
                if isinstance(s, Symbol):
                    syms.append(Symbol(s.name, **s._assumptions.generator))
                elif isinstance(s, str):
                    sympy_deprecation_warning(
                        f"""

Passing a string as the coordinate symbol name to CoordSystem is deprecated.
Pass a Symbol with the appropriate name and assumptions instead.

That is, replace {s} with Symbol({s!r}, real=True).
                        """,

                        deprecated_since_version="1.7",
                        active_deprecations_target="deprecated-diffgeom-mutable",
                    )
                    syms.append(Symbol(s, real=True))
            symbols = Tuple(*syms)

        # canonicallize the relations
        rel_temp = {}
        for k,v in relations.items():
            s1, s2 = k
            if not isinstance(s1, Str):
                s1 = Str(s1)
            if not isinstance(s2, Str):
                s2 = Str(s2)
            key = Tuple(s1, s2)

            # Old version used Lambda as a value.
            if isinstance(v, Lambda):
                v = (tuple(v.signature), tuple(v.expr))
            else:
                v = (tuple(v[0]), tuple(v[1]))
            rel_temp[key] = v
        relations = Dict(rel_temp)

        # construct the object
        obj = super().__new__(cls, name, patch, symbols, relations)

        # Add deprecated attributes
        obj.transforms = _deprecated_dict(
            """
            CoordSystem.transforms is deprecated. The CoordSystem class is now
            immutable. Use the 'relations' keyword argument to the
            CoordSystems() constructor to specify relations.
            """, {})
        obj._names = [str(n) for n in symbols]
        obj.patch.coord_systems.append(obj) # deprecated
        obj._dummies = [Dummy(str(n)) for n in symbols] # deprecated
        obj._dummy = Dummy()

        return obj

    @property
    def name(self):
        return self.args[0]

    @property
    def patch(self):
        return self.args[1]

    @property
    def manifold(self):
        return self.patch.manifold

    @property
    def symbols(self):
        return tuple(CoordinateSymbol(self, i, **s._assumptions.generator)
            for i,s in enumerate(self.args[2]))

    @property
    def relations(self):
        return self.args[3]

    @property
    def dim(self):
        return self.patch.dim

    ##########################################################################
    # Finding transformation relation
    ##########################################################################

    def transformation(self, sys):
        """
        Return coordinate transformation function from *self* to *sys*.

        Parameters
        ==========

        sys : CoordSystem

        Returns
        =======

        sympy.Lambda

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.transformation(R2_p)
        Lambda((x, y), Matrix([
        [sqrt(x**2 + y**2)],
        [      atan2(y, x)]]))

        """
        signature = self.args[2]

        key = Tuple(self.name, sys.name)
        if self == sys:
            expr = Matrix(self.symbols)
        elif key in self.relations:
            expr = Matrix(self.relations[key][1])
        elif key[::-1] in self.relations:
            expr = Matrix(self._inverse_transformation(sys, self))
        else:
            expr = Matrix(self._indirect_transformation(self, sys))
        return Lambda(signature, expr)

    @staticmethod
    def _solve_inverse(sym1, sym2, exprs, sys1_name, sys2_name):
        ret = solve(
            [t[0] - t[1] for t in zip(sym2, exprs)],
            list(sym1), dict=True)

        if len(ret) == 0:
            temp = "Cannot solve inverse relation from {} to {}."
            raise NotImplementedError(temp.format(sys1_name, sys2_name))
        elif len(ret) > 1:
            temp = "Obtained multiple inverse relation from {} to {}."
            raise ValueError(temp.format(sys1_name, sys2_name))

        return ret[0]

    @classmethod
    def _inverse_transformation(cls, sys1, sys2):
        # Find the transformation relation from sys2 to sys1
        forward = sys1.transform(sys2)
        inv_results = cls._solve_inverse(sys1.symbols, sys2.symbols, forward,
                                         sys1.name, sys2.name)
        signature = tuple(sys1.symbols)
        return [inv_results[s] for s in signature]

    @classmethod
    @cacheit
    def _indirect_transformation(cls, sys1, sys2):
        # Find the transformation relation between two indirectly connected
        # coordinate systems
        rel = sys1.relations
        path = cls._dijkstra(sys1, sys2)

        transforms = []
        for s1, s2 in zip(path, path[1:]):
            if (s1, s2) in rel:
                transforms.append(rel[(s1, s2)])
            else:
                sym2, inv_exprs = rel[(s2, s1)]
                sym1 = tuple(Dummy() for i in sym2)
                ret = cls._solve_inverse(sym2, sym1, inv_exprs, s2, s1)
                ret = tuple(ret[s] for s in sym2)
                transforms.append((sym1, ret))
        syms = sys1.args[2]
        exprs = syms
        for newsyms, newexprs in transforms:
            exprs = tuple(e.subs(zip(newsyms, exprs)) for e in newexprs)
        return exprs

    @staticmethod
    def _dijkstra(sys1, sys2):
        # Use Dijkstra algorithm to find the shortest path between two indirectly-connected
        # coordinate systems
        # return value is the list of the names of the systems.
        relations = sys1.relations
        graph = {}
        for s1, s2 in relations.keys():
            if s1 not in graph:
                graph[s1] = {s2}
            else:
                graph[s1].add(s2)
            if s2 not in graph:
                graph[s2] = {s1}
            else:
                graph[s2].add(s1)

        path_dict = {sys:[0, [], 0] for sys in graph} # minimum distance, path, times of visited

        def visit(sys):
            path_dict[sys][2] = 1
            for newsys in graph[sys]:
                distance = path_dict[sys][0] + 1
                if path_dict[newsys][0] >= distance or not path_dict[newsys][1]:
                    path_dict[newsys][0] = distance
                    path_dict[newsys][1] = list(path_dict[sys][1])
                    path_dict[newsys][1].append(sys)

        visit(sys1.name)

        while True:
            min_distance = max(path_dict.values(), key=lambda x:x[0])[0]
            newsys = None
            for sys, lst in path_dict.items():
                if 0 < lst[0] <= min_distance and not lst[2]:
                    min_distance = lst[0]
                    newsys = sys
            if newsys is None:
                break
            visit(newsys)

        result = path_dict[sys2.name][1]
        result.append(sys2.name)

        if result == [sys2.name]:
            raise KeyError("Two coordinate systems are not connected.")
        return result

    def connect_to(self, to_sys, from_coords, to_exprs, inverse=True, fill_in_gaps=False):
        sympy_deprecation_warning(
            """
            The CoordSystem.connect_to() method is deprecated. Instead,
            generate a new instance of CoordSystem with the 'relations'
            keyword argument (CoordSystem classes are now immutable).
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-diffgeom-mutable",
        )

        from_coords, to_exprs = dummyfy(from_coords, to_exprs)
        self.transforms[to_sys] = Matrix(from_coords), Matrix(to_exprs)

        if inverse:
            to_sys.transforms[self] = self._inv_transf(from_coords, to_exprs)

        if fill_in_gaps:
            self._fill_gaps_in_transformations()

    @staticmethod
    def _inv_transf(from_coords, to_exprs):
        # Will be removed when connect_to is removed
        inv_from = [i.as_dummy() for i in from_coords]
        inv_to = solve(
            [t[0] - t[1] for t in zip(inv_from, to_exprs)],
            list(from_coords), dict=True)[0]
        inv_to = [inv_to[fc] for fc in from_coords]
        return Matrix(inv_from), Matrix(inv_to)

    @staticmethod
    def _fill_gaps_in_transformations():
        # Will be removed when connect_to is removed
        raise NotImplementedError

    ##########################################################################
    # Coordinate transformations
    ##########################################################################

    def transform(self, sys, coordinates=None):
        """
        Return the result of coordinate transformation from *self* to *sys*.
        If coordinates are not given, coordinate symbols of *self* are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.ImmutableDenseMatrix containing CoordinateSymbol

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.transform(R2_p)
        Matrix([
        [sqrt(x**2 + y**2)],
        [      atan2(y, x)]])
        >>> R2_r.transform(R2_p, [0, 1])
        Matrix([
        [   1],
        [pi/2]])

        """
        if coordinates is None:
            coordinates = self.symbols
        if self != sys:
            transf = self.transformation(sys)
            coordinates = transf(*coordinates)
        else:
            coordinates = Matrix(coordinates)
        return coordinates

    def coord_tuple_transform_to(self, to_sys, coords):
        """Transform ``coords`` to coord system ``to_sys``."""
        sympy_deprecation_warning(
            """
            The CoordSystem.coord_tuple_transform_to() method is deprecated.
            Use the CoordSystem.transform() method instead.
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-diffgeom-mutable",
        )

        coords = Matrix(coords)
        if self != to_sys:
            with ignore_warnings(SymPyDeprecationWarning):
                transf = self.transforms[to_sys]
            coords = transf[1].subs(list(zip(transf[0], coords)))
        return coords

    def jacobian(self, sys, coordinates=None):
        """
        Return the jacobian matrix of a transformation on given coordinates.
        If coordinates are not given, coordinate symbols of *self* are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.ImmutableDenseMatrix

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_p.jacobian(R2_r)
        Matrix([
        [cos(theta), -rho*sin(theta)],
        [sin(theta),  rho*cos(theta)]])
        >>> R2_p.jacobian(R2_r, [1, 0])
        Matrix([
        [1, 0],
        [0, 1]])

        """
        result = self.transform(sys).jacobian(self.symbols)
        if coordinates is not None:
            result = result.subs(list(zip(self.symbols, coordinates)))
        return result
    jacobian_matrix = jacobian

    def jacobian_determinant(self, sys, coordinates=None):
        """
        Return the jacobian determinant of a transformation on given
        coordinates. If coordinates are not given, coordinate symbols of *self*
        are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.Expr

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.jacobian_determinant(R2_p)
        1/sqrt(x**2 + y**2)
        >>> R2_r.jacobian_determinant(R2_p, [1, 0])
        1

        """
        return self.jacobian(sys, coordinates).det()


    ##########################################################################
    # Points
    ##########################################################################

    def point(self, coords):
        """Create a ``Point`` with coordinates given in this coord system."""
        return Point(self, coords)

    def point_to_coords(self, point):
        """Calculate the coordinates of a point in this coord system."""
        return point.coords(self)

    ##########################################################################
    # Base fields.
    ##########################################################################

    def base_scalar(self, coord_index):
        """Return ``BaseScalarField`` that takes a point and returns one of the coordinates."""
        return BaseScalarField(self, coord_index)
    coord_function = base_scalar

    def base_scalars(self):
        """Returns a list of all coordinate functions.
        For more details see the ``base_scalar`` method of this class."""
        return [self.base_scalar(i) for i in range(self.dim)]
    coord_functions = base_scalars

    def base_vector(self, coord_index):
        """Return a basis vector field.
        The basis vector field for this coordinate system. It is also an
        operator on scalar fields."""
        return BaseVectorField(self, coord_index)

    def base_vectors(self):
        """Returns a list of all base vectors.
        For more details see the ``base_vector`` method of this class."""
        return [self.base_vector(i) for i in range(self.dim)]

    def base_oneform(self, coord_index):
        """Return a basis 1-form field.
        The basis one-form field for this coordinate system. It is also an
        operator on vector fields."""
        return Differential(self.coord_function(coord_index))

    def base_oneforms(self):
        """Returns a list of all base oneforms.
        For more details see the ``base_oneform`` method of this class."""
        return [self.base_oneform(i) for i in range(self.dim)]


class CoordinateSymbol(Symbol):
    """A symbol which denotes an abstract value of i-th coordinate of
    the coordinate system with given context.

    Explanation
    ===========

    Each coordinates in coordinate system are represented by unique symbol,
    such as x, y, z in Cartesian coordinate system.

    You may not construct this class directly. Instead, use `symbols` method
    of CoordSystem.

    Parameters
    ==========

    coord_sys : CoordSystem

    index : integer

    Examples
    ========

    >>> from sympy import symbols, Lambda, Matrix, sqrt, atan2, cos, sin
    >>> from sympy.diffgeom import Manifold, Patch, CoordSystem
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> x, y = symbols('x y', real=True)
    >>> r, theta = symbols('r theta', nonnegative=True)
    >>> relation_dict = {
    ... ('Car2D', 'Pol'): Lambda((x, y), Matrix([sqrt(x**2 + y**2), atan2(y, x)])),
    ... ('Pol', 'Car2D'): Lambda((r, theta), Matrix([r*cos(theta), r*sin(theta)]))
    ... }
    >>> Car2D = CoordSystem('Car2D', p, [x, y], relation_dict)
    >>> Pol = CoordSystem('Pol', p, [r, theta], relation_dict)
    >>> x, y = Car2D.symbols

    ``CoordinateSymbol`` contains its coordinate symbol and index.

    >>> x.name
    'x'
    >>> x.coord_sys == Car2D
    True
    >>> x.index
    0
    >>> x.is_real
    True

    You can transform ``CoordinateSymbol`` into other coordinate system using
    ``rewrite()`` method.

    >>> x.rewrite(Pol)
    r*cos(theta)
    >>> sqrt(x**2 + y**2).rewrite(Pol).simplify()
    r

    """
    def __new__(cls, coord_sys, index, **assumptions):
        name = coord_sys.args[2][index].name
        obj = super().__new__(cls, name, **assumptions)
        obj.coord_sys = coord_sys
        obj.index = index
        return obj

    def __getnewargs__(self):
        return (self.coord_sys, self.index)

    def _hashable_content(self):
        return (
            self.coord_sys, self.index
        ) + tuple(sorted(self.assumptions0.items()))

    def _eval_rewrite(self, rule, args, **hints):
        if isinstance(rule, CoordSystem):
            return rule.transform(self.coord_sys)[self.index]
        return super()._eval_rewrite(rule, args, **hints)


class Point(Basic):
    """Point defined in a coordinate system.

    Explanation
    ===========

    Mathematically, point is defined in the manifold and does not have any coordinates
    by itself. Coordinate system is what imbues the coordinates to the point by coordinate
    chart. However, due to the difficulty of realizing such logic, you must supply
    a coordinate system and coordinates to define a Point here.

    The usage of this object after its definition is independent of the
    coordinate system that was used in order to define it, however due to
    limitations in the simplification routines you can arrive at complicated
    expressions if you use inappropriate coordinate systems.

    Parameters
    ==========

    coord_sys : CoordSystem

    coords : list
        The coordinates of the point.

    Examples
    ========

    >>> from sympy import pi
    >>> from sympy.diffgeom import Point
    >>> from sympy.diffgeom.rn import R2, R2_r, R2_p
    >>> rho, theta = R2_p.symbols

    >>> p = Point(R2_p, [rho, 3*pi/4])

    >>> p.manifold == R2
    True

    >>> p.coords()
    Matrix([
    [   rho],
    [3*pi/4]])
    >>> p.coords(R2_r)
    Matrix([
    [-sqrt(2)*rho/2],
    [ sqrt(2)*rho/2]])

    """

    def __new__(cls, coord_sys, coords, **kwargs):
        coords = Matrix(coords)
        obj = super().__new__(cls, coord_sys, coords)
        obj._coord_sys = coord_sys
        obj._coords = coords
        return obj

    @property
    def patch(self):
        return self._coord_sys.patch

    @property
    def manifold(self):
        return self._coord_sys.manifold

    @property
    def dim(self):
        return self.manifold.dim

    def coords(self, sys=None):
        """
        Coordinates of the point in given coordinate system. If coordinate system
        is not passed, it returns the coordinates in the coordinate system in which
        the point was defined.
        """
        if sys is None:
            return self._coords
        else:
            return self._coord_sys.transform(sys, self._coords)

    @property
    def free_symbols(self):
        return self._coords.free_symbols


class BaseScalarField(Expr):
    """Base scalar field over a manifold for a given coordinate system.

    Explanation
    ===========

    A scalar field takes a point as an argument and returns a scalar.
    A base scalar field of a coordinate system takes a point and returns one of
    the coordinates of that point in the coordinate system in question.

    To define a scalar field you need to choose the coordinate system and the
    index of the coordinate.

    The use of the scalar field after its definition is independent of the
    coordinate system in which it was defined, however due to limitations in
    the simplification routines you may arrive at more complicated
    expression if you use unappropriate coordinate systems.
    You can build complicated scalar fields by just building up SymPy
    expressions containing ``BaseScalarField`` instances.

    Parameters
    ==========

    coord_sys : CoordSystem

    index : integer

    Examples
    ========

    >>> from sympy import Function, pi
    >>> from sympy.diffgeom import BaseScalarField
    >>> from sympy.diffgeom.rn import R2_r, R2_p
    >>> rho, _ = R2_p.symbols
    >>> point = R2_p.point([rho, 0])
    >>> fx, fy = R2_r.base_scalars()
    >>> ftheta = BaseScalarField(R2_r, 1)

    >>> fx(point)
    rho
    >>> fy(point)
    0

    >>> (fx**2+fy**2).rcall(point)
    rho**2

    >>> g = Function('g')
    >>> fg = g(ftheta-pi)
    >>> fg.rcall(point)
    g(-pi)

    """

    is_commutative = True

    def __new__(cls, coord_sys, index, **kwargs):
        index = _sympify(index)
        obj = super().__new__(cls, coord_sys, index)
        obj._coord_sys = coord_sys
        obj._index = index
        return obj

    @property
    def coord_sys(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    @property
    def patch(self):
        return self.coord_sys.patch

    @property
    def manifold(self):
        return self.coord_sys.manifold

    @property
    def dim(self):
        return self.manifold.dim

    def __call__(self, *args):
        """Evaluating the field at a point or doing nothing.
        If the argument is a ``Point`` instance, the field is evaluated at that
        point. The field is returned itself if the argument is any other
        object. It is so in order to have working recursive calling mechanics
        for all fields (check the ``__call__`` method of ``Expr``).
        """
        point = args[0]
        if len(args) != 1 or not isinstance(point, Point):
            return self
        coords = point.coords(self._coord_sys)
        # XXX Calling doit  is necessary with all the Subs expressions
        # XXX Calling simplify is necessary with all the trig expressions
        return simplify(coords[self._index]).doit()

    # XXX Workaround for limitations on the content of args
    free_symbols: set[Any] = set()


class BaseVectorField(Expr):
    r"""Base vector field over a manifold for a given coordinate system.

    Explanation
    ===========

    A vector field is an operator taking a scalar field and returning a
    directional derivative (which is also a scalar field).
    A base vector field is the same type of operator, however the derivation is
    specifically done with respect to a chosen coordinate.

    To define a base vector field you need to choose the coordinate system and
    the index of the coordinate.

    The use of the vector field after its definition is independent of the
    coordinate system in which it was defined, however due to limitations in the
    simplification routines you may arrive at more complicated expression if you
    use unappropriate coordinate systems.

    Parameters
    ==========
    coord_sys : CoordSystem

    index : integer

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import BaseVectorField
    >>> from sympy import pprint

    >>> x, y = R2_r.symbols
    >>> rho, theta = R2_p.symbols
    >>> fx, fy = R2_r.base_scalars()
    >>> point_p = R2_p.point([rho, theta])
    >>> point_r = R2_r.point([x, y])

    >>> g = Function('g')
    >>> s_field = g(fx, fy)

    >>> v = BaseVectorField(R2_r, 1)
    >>> pprint(v(s_field))
    / d           \|
    |---(g(x, xi))||
    \dxi          /|xi=y
    >>> pprint(v(s_field).rcall(point_r).doit())
    d
    --(g(x, y))
    dy
    >>> pprint(v(s_field).rcall(point_p))
    / d                        \|
    |---(g(rho*cos(theta), xi))||
    \dxi                       /|xi=rho*sin(theta)

    """

    is_commutative = False

    def __new__(cls, coord_sys, index, **kwargs):
        index = _sympify(index)
        obj = super().__new__(cls, coord_sys, index)
        obj._coord_sys = coord_sys
        obj._index = index
        return obj

    @property
    def coord_sys(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    @property
    def patch(self):
        return self.coord_sys.patch

    @property
    def manifold(self):
        return self.coord_sys.manifold

    @property
    def dim(self):
        return self.manifold.dim

    def __call__(self, scalar_field):
        """Apply on a scalar field.
        The action of a vector field on a scalar field is a directional
        differentiation.
        If the argument is not a scalar field an error is raised.
        """
        if covariant_order(scalar_field) or contravariant_order(scalar_field):
            raise ValueError('Only scalar fields can be supplied as arguments to vector fields.')

        if scalar_field is None:
            return self

        base_scalars = list(scalar_field.atoms(BaseScalarField))

        # First step: e_x(x+r**2) -> e_x(x) + 2*r*e_x(r)
        d_var = self._coord_sys._dummy
        # TODO: you need a real dummy function for the next line
        d_funcs = [Function('_#_%s' % i)(d_var) for i,
                   b in enumerate(base_scalars)]
        d_result = scalar_field.subs(list(zip(base_scalars, d_funcs)))
        d_result = d_result.diff(d_var)

        # Second step: e_x(x) -> 1 and e_x(r) -> cos(atan2(x, y))
        coords = self._coord_sys.symbols
        d_funcs_deriv = [f.diff(d_var) for f in d_funcs]
        d_funcs_deriv_sub = []
        for b in base_scalars:
            jac = self._coord_sys.jacobian(b._coord_sys, coords)
            d_funcs_deriv_sub.append(jac[b._index, self._index])
        d_result = d_result.subs(list(zip(d_funcs_deriv, d_funcs_deriv_sub)))

        # Remove the dummies
        result = d_result.subs(list(zip(d_funcs, base_scalars)))
        result = result.subs(list(zip(coords, self._coord_sys.coord_functions())))
        return result.doit()


def _find_coords(expr):
    # Finds CoordinateSystems existing in expr
    fields = expr.atoms(BaseScalarField, BaseVectorField)
    return {f._coord_sys for f in fields}


class Commutator(Expr):
    r"""Commutator of two vector fields.

    Explanation
    ===========

    The commutator of two vector fields `v_1` and `v_2` is defined as the
    vector field `[v_1, v_2]` that evaluated on each scalar field `f` is equal
    to `v_1(v_2(f)) - v_2(v_1(f))`.

    Examples
    ========


    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import Commutator
    >>> from sympy import simplify

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> e_r = R2_p.base_vector(0)

    >>> c_xy = Commutator(e_x, e_y)
    >>> c_xr = Commutator(e_x, e_r)
    >>> c_xy
    0

    Unfortunately, the current code is not able to compute everything:

    >>> c_xr
    Commutator(e_x, e_rho)
    >>> simplify(c_xr(fy**2))
    -2*cos(theta)*y**2/(x**2 + y**2)

    """
    def __new__(cls, v1, v2):
        if (covariant_order(v1) or contravariant_order(v1) != 1
                or covariant_order(v2) or contravariant_order(v2) != 1):
            raise ValueError(
                'Only commutators of vector fields are supported.')
        if v1 == v2:
            return S.Zero
        coord_sys = set().union(*[_find_coords(v) for v in (v1, v2)])
        if len(coord_sys) == 1:
            # Only one coordinate systems is used, hence it is easy enough to
            # actually evaluate the commutator.
            if all(isinstance(v, BaseVectorField) for v in (v1, v2)):
                return S.Zero
            bases_1, bases_2 = [list(v.atoms(BaseVectorField))
                                for v in (v1, v2)]
            coeffs_1 = [v1.expand().coeff(b) for b in bases_1]
            coeffs_2 = [v2.expand().coeff(b) for b in bases_2]
            res = 0
            for c1, b1 in zip(coeffs_1, bases_1):
                for c2, b2 in zip(coeffs_2, bases_2):
                    res += c1*b1(c2)*b2 - c2*b2(c1)*b1
            return res
        else:
            obj = super().__new__(cls, v1, v2)
            obj._v1 = v1 # deprecated assignment
            obj._v2 = v2 # deprecated assignment
            return obj

    @property
    def v1(self):
        return self.args[0]

    @property
    def v2(self):
        return self.args[1]

    def __call__(self, scalar_field):
        """Apply on a scalar field.
        If the argument is not a scalar field an error is raised.
        """
        return self.v1(self.v2(scalar_field)) - self.v2(self.v1(scalar_field))


class Differential(Expr):
    r"""Return the differential (exterior derivative) of a form field.

    Explanation
    ===========

    The differential of a form (i.e. the exterior derivative) has a complicated
    definition in the general case.
    The differential `df` of the 0-form `f` is defined for any vector field `v`
    as `df(v) = v(f)`.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import Differential
    >>> from sympy import pprint

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> g = Function('g')
    >>> s_field = g(fx, fy)
    >>> dg = Differential(s_field)

    >>> dg
    d(g(x, y))
    >>> pprint(dg(e_x))
    / d           \|
    |---(g(xi, y))||
    \dxi          /|xi=x
    >>> pprint(dg(e_y))
    / d           \|
    |---(g(x, xi))||
    \dxi          /|xi=y

    Applying the exterior derivative operator twice always results in:

    >>> Differential(dg)
    0
    """

    is_commutative = False

    def __new__(cls, form_field):
        if contravariant_order(form_field):
            raise ValueError(
                'A vector field was supplied as an argument to Differential.')
        if isinstance(form_field, Differential):
            return S.Zero
        else:
            obj = super().__new__(cls, form_field)
            obj._form_field = form_field # deprecated assignment
            return obj

    @property
    def form_field(self):
        return self.args[0]

    def __call__(self, *vector_fields):
        """Apply on a list of vector_fields.

        Explanation
        ===========

        If the number of vector fields supplied is not equal to 1 + the order of
        the form field inside the differential the result is undefined.

        For 1-forms (i.e. differentials of scalar fields) the evaluation is
        done as `df(v)=v(f)`. However if `v` is ``None`` instead of a vector
        field, the differential is returned unchanged. This is done in order to
        permit partial contractions for higher forms.

        In the general case the evaluation is done by applying the form field
        inside the differential on a list with one less elements than the number
        of elements in the original list. Lowering the number of vector fields
        is achieved through replacing each pair of fields by their
        commutator.

        If the arguments are not vectors or ``None``s an error is raised.
        """
        if any((contravariant_order(a) != 1 or covariant_order(a)) and a is not None
                for a in vector_fields):
            raise ValueError('The arguments supplied to Differential should be vector fields or Nones.')
        k = len(vector_fields)
        if k == 1:
            if vector_fields[0]:
                return vector_fields[0].rcall(self._form_field)
            return self
        else:
            # For higher form it is more complicated:
            # Invariant formula:
            # https://en.wikipedia.org/wiki/Exterior_derivative#Invariant_formula
            # df(v1, ... vn) = +/- vi(f(v1..no i..vn))
            #                  +/- f([vi,vj],v1..no i, no j..vn)
            f = self._form_field
            v = vector_fields
            ret = 0
            for i in range(k):
                t = v[i].rcall(f.rcall(*v[:i] + v[i + 1:]))
                ret += (-1)**i*t
                for j in range(i + 1, k):
                    c = Commutator(v[i], v[j])
                    if c:  # TODO this is ugly - the Commutator can be Zero and
                        # this causes the next line to fail
                        t = f.rcall(*(c,) + v[:i] + v[i + 1:j] + v[j + 1:])
                        ret += (-1)**(i + j)*t
            return ret


class TensorProduct(Expr):
    """Tensor product of forms.

    Explanation
    ===========

    The tensor product permits the creation of multilinear functionals (i.e.
    higher order tensors) out of lower order fields (e.g. 1-forms and vector
    fields). However, the higher tensors thus created lack the interesting
    features provided by the other type of product, the wedge product, namely
    they are not antisymmetric and hence are not form fields.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import TensorProduct

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> TensorProduct(dx, dy)(e_x, e_y)
    1
    >>> TensorProduct(dx, dy)(e_y, e_x)
    0
    >>> TensorProduct(dx, fx*dy)(fx*e_x, e_y)
    x**2
    >>> TensorProduct(e_x, e_y)(fx**2, fy**2)
    4*x*y
    >>> TensorProduct(e_y, dx)(fy)
    dx

    You can nest tensor products.

    >>> tp1 = TensorProduct(dx, dy)
    >>> TensorProduct(tp1, dx)(e_x, e_y, e_x)
    1

    You can make partial contraction for instance when 'raising an index'.
    Putting ``None`` in the second argument of ``rcall`` means that the
    respective position in the tensor product is left as it is.

    >>> TP = TensorProduct
    >>> metric = TP(dx, dx) + 3*TP(dy, dy)
    >>> metric.rcall(e_y, None)
    3*dy

    Or automatically pad the args with ``None`` without specifying them.

    >>> metric.rcall(e_y)
    3*dy

    """
    def __new__(cls, *args):
        scalar = Mul(*[m for m in args if covariant_order(m) + contravariant_order(m) == 0])
        multifields = [m for m in args if covariant_order(m) + contravariant_order(m)]
        if multifields:
            if len(multifields) == 1:
                return scalar*multifields[0]
            return scalar*super().__new__(cls, *multifields)
        else:
            return scalar

    def __call__(self, *fields):
        """Apply on a list of fields.

        If the number of input fields supplied is not equal to the order of
        the tensor product field, the list of arguments is padded with ``None``'s.

        The list of arguments is divided in sublists depending on the order of
        the forms inside the tensor product. The sublists are provided as
        arguments to these forms and the resulting expressions are given to the
        constructor of ``TensorProduct``.

        """
        tot_order = covariant_order(self) + contravariant_order(self)
        tot_args = len(fields)
        if tot_args != tot_order:
            fields = list(fields) + [None]*(tot_order - tot_args)
        orders = [covariant_order(f) + contravariant_order(f) for f in self._args]
        indices = [sum(orders[:i + 1]) for i in range(len(orders) - 1)]
        fields = [fields[i:j] for i, j in zip([0] + indices, indices + [None])]
        multipliers = [t[0].rcall(*t[1]) for t in zip(self._args, fields)]
        return TensorProduct(*multipliers)


class WedgeProduct(TensorProduct):
    """Wedge product of forms.

    Explanation
    ===========

    In the context of integration only completely antisymmetric forms make
    sense. The wedge product permits the creation of such forms.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import WedgeProduct

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> WedgeProduct(dx, dy)(e_x, e_y)
    1
    >>> WedgeProduct(dx, dy)(e_y, e_x)
    -1
    >>> WedgeProduct(dx, fx*dy)(fx*e_x, e_y)
    x**2
    >>> WedgeProduct(e_x, e_y)(fy, None)
    -e_x

    You can nest wedge products.

    >>> wp1 = WedgeProduct(dx, dy)
    >>> WedgeProduct(wp1, dx)(e_x, e_y, e_x)
    0

    """
    # TODO the calculation of signatures is slow
    # TODO you do not need all these permutations (neither the prefactor)
    def __call__(self, *fields):
        """Apply on a list of vector_fields.
        The expression is rewritten internally in terms of tensor products and evaluated."""
        orders = (covariant_order(e) + contravariant_order(e) for e in self.args)
        mul = 1/Mul(*(factorial(o) for o in orders))
        perms = permutations(fields)
        perms_par = (Permutation(
            p).signature() for p in permutations(range(len(fields))))
        tensor_prod = TensorProduct(*self.args)
        return mul*Add(*[tensor_prod(*p[0])*p[1] for p in zip(perms, perms_par)])


class LieDerivative(Expr):
    """Lie derivative with respect to a vector field.

    Explanation
    ===========

    The transport operator that defines the Lie derivative is the pushforward of
    the field to be derived along the integral curve of the field with respect
    to which one derives.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r, R2_p
    >>> from sympy.diffgeom import (LieDerivative, TensorProduct)

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> e_rho, e_theta = R2_p.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> LieDerivative(e_x, fy)
    0
    >>> LieDerivative(e_x, fx)
    1
    >>> LieDerivative(e_x, e_x)
    0

    The Lie derivative of a tensor field by another tensor field is equal to
    their commutator:

    >>> LieDerivative(e_x, e_rho)
    Commutator(e_x, e_rho)
    >>> LieDerivative(e_x + e_y, fx)
    1

    >>> tp = TensorProduct(dx, dy)
    >>> LieDerivative(e_x, tp)
    LieDerivative(e_x, TensorProduct(dx, dy))
    >>> LieDerivative(e_x, tp)
    LieDerivative(e_x, TensorProduct(dx, dy))

    """
    def __new__(cls, v_field, expr):
        expr_form_ord = covariant_order(expr)
        if contravariant_order(v_field) != 1 or covariant_order(v_field):
            raise ValueError('Lie derivatives are defined only with respect to'
                             ' vector fields. The supplied argument was not a '
                             'vector field.')
        if expr_form_ord > 0:
            obj = super().__new__(cls, v_field, expr)
            # deprecated assignments
            obj._v_field = v_field
            obj._expr = expr
            return obj
        if expr.atoms(BaseVectorField):
            return Commutator(v_field, expr)
        else:
            return v_field.rcall(expr)

    @property
    def v_field(self):
        return self.args[0]

    @property
    def expr(self):
        return self.args[1]

    def __call__(self, *args):
        v = self.v_field
        expr = self.expr
        lead_term = v(expr(*args))
        rest = Add(*[Mul(*args[:i] + (Commutator(v, args[i]),) + args[i + 1:])
                     for i in range(len(args))])
        return lead_term - rest


class BaseCovarDerivativeOp(Expr):
    """Covariant derivative operator with respect to a base vector.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import BaseCovarDerivativeOp
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct

    >>> TP = TensorProduct
    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> ch = metric_to_Christoffel_2nd(TP(dx, dx) + TP(dy, dy))
    >>> ch
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> cvd = BaseCovarDerivativeOp(R2_r, 0, ch)
    >>> cvd(fx)
    1
    >>> cvd(fx*e_x)
    e_x
    """

    def __new__(cls, coord_sys, index, christoffel):
        index = _sympify(index)
        christoffel = ImmutableDenseNDimArray(christoffel)
        obj = super().__new__(cls, coord_sys, index, christoffel)
        # deprecated assignments
        obj._coord_sys = coord_sys
        obj._index = index
        obj._christoffel = christoffel
        return obj

    @property
    def coord_sys(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    @property
    def christoffel(self):
        return self.args[2]

    def __call__(self, field):
        """Apply on a scalar field.

        The action of a vector field on a scalar field is a directional
        differentiation.
        If the argument is not a scalar field the behaviour is undefined.
        """
        if covariant_order(field) != 0:
            raise NotImplementedError()

        field = vectors_in_basis(field, self._coord_sys)

        wrt_vector = self._coord_sys.base_vector(self._index)
        wrt_scalar = self._coord_sys.coord_function(self._index)
        vectors = list(field.atoms(BaseVectorField))

        # First step: replace all vectors with something susceptible to
        # derivation and do the derivation
        # TODO: you need a real dummy function for the next line
        d_funcs = [Function('_#_%s' % i)(wrt_scalar) for i,
                   b in enumerate(vectors)]
        d_result = field.subs(list(zip(vectors, d_funcs)))
        d_result = wrt_vector(d_result)

        # Second step: backsubstitute the vectors in
        d_result = d_result.subs(list(zip(d_funcs, vectors)))

        # Third step: evaluate the derivatives of the vectors
        derivs = []
        for v in vectors:
            d = Add(*[(self._christoffel[k, wrt_vector._index, v._index]
                       *v._coord_sys.base_vector(k))
                      for k in range(v._coord_sys.dim)])
            derivs.append(d)
        to_subs = [wrt_vector(d) for d in d_funcs]
        # XXX: This substitution can fail when there are Dummy symbols and the
        # cache is disabled: https://github.com/sympy/sympy/issues/17794
        result = d_result.subs(list(zip(to_subs, derivs)))

        # Remove the dummies
        result = result.subs(list(zip(d_funcs, vectors)))
        return result.doit()


class CovarDerivativeOp(Expr):
    """Covariant derivative operator.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import CovarDerivativeOp
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct
    >>> TP = TensorProduct
    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()
    >>> ch = metric_to_Christoffel_2nd(TP(dx, dx) + TP(dy, dy))

    >>> ch
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> cvd = CovarDerivativeOp(fx*e_x, ch)
    >>> cvd(fx)
    x
    >>> cvd(fx*e_x)
    x*e_x

    """

    def __new__(cls, wrt, christoffel):
        if len({v._coord_sys for v in wrt.atoms(BaseVectorField)}) > 1:
            raise NotImplementedError()
        if contravariant_order(wrt) != 1 or covariant_order(wrt):
            raise ValueError('Covariant derivatives are defined only with '
                             'respect to vector fields. The supplied argument '
                             'was not a vector field.')
        christoffel = ImmutableDenseNDimArray(christoffel)
        obj = super().__new__(cls, wrt, christoffel)
        # deprecated assignments
        obj._wrt = wrt
        obj._christoffel = christoffel
        return obj

    @property
    def wrt(self):
        return self.args[0]

    @property
    def christoffel(self):
        return self.args[1]

    def __call__(self, field):
        vectors = list(self._wrt.atoms(BaseVectorField))
        base_ops = [BaseCovarDerivativeOp(v._coord_sys, v._index, self._christoffel)
                    for v in vectors]
        return self._wrt.subs(list(zip(vectors, base_ops))).rcall(field)


###############################################################################
# Integral curves on vector fields
###############################################################################
def intcurve_series(vector_field, param, start_point, n=6, coord_sys=None, coeffs=False):
    r"""Return the series expansion for an integral curve of the field.

    Explanation
    ===========

    Integral curve is a function `\gamma` taking a parameter in `R` to a point
    in the manifold. It verifies the equation:

    `V(f)\big(\gamma(t)\big) = \frac{d}{dt}f\big(\gamma(t)\big)`

    where the given ``vector_field`` is denoted as `V`. This holds for any
    value `t` for the parameter and any scalar field `f`.

    This equation can also be decomposed of a basis of coordinate functions
    `V(f_i)\big(\gamma(t)\big) = \frac{d}{dt}f_i\big(\gamma(t)\big) \quad \forall i`

    This function returns a series expansion of `\gamma(t)` in terms of the
    coordinate system ``coord_sys``. The equations and expansions are necessarily
    done in coordinate-system-dependent way as there is no other way to
    represent movement between points on the manifold (i.e. there is no such
    thing as a difference of points for a general manifold).

    Parameters
    ==========
    vector_field
        the vector field for which an integral curve will be given

    param
        the argument of the function `\gamma` from R to the curve

    start_point
        the point which corresponds to `\gamma(0)`

    n
        the order to which to expand

    coord_sys
        the coordinate system in which to expand
        coeffs (default False) - if True return a list of elements of the expansion

    Examples
    ========

    Use the predefined R2 manifold:

    >>> from sympy.abc import t, x, y
    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import intcurve_series

    Specify a starting point and a vector field:

    >>> start_point = R2_r.point([x, y])
    >>> vector_field = R2_r.e_x

    Calculate the series:

    >>> intcurve_series(vector_field, t, start_point, n=3)
    Matrix([
    [t + x],
    [    y]])

    Or get the elements of the expansion in a list:

    >>> series = intcurve_series(vector_field, t, start_point, n=3, coeffs=True)
    >>> series[0]
    Matrix([
    [x],
    [y]])
    >>> series[1]
    Matrix([
    [t],
    [0]])
    >>> series[2]
    Matrix([
    [0],
    [0]])

    The series in the polar coordinate system:

    >>> series = intcurve_series(vector_field, t, start_point,
    ...             n=3, coord_sys=R2_p, coeffs=True)
    >>> series[0]
    Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]])
    >>> series[1]
    Matrix([
    [t*x/sqrt(x**2 + y**2)],
    [   -t*y/(x**2 + y**2)]])
    >>> series[2]
    Matrix([
    [t**2*(-x**2/(x**2 + y**2)**(3/2) + 1/sqrt(x**2 + y**2))/2],
    [                                t**2*x*y/(x**2 + y**2)**2]])

    See Also
    ========

    intcurve_diffequ

    """
    if contravariant_order(vector_field) != 1 or covariant_order(vector_field):
        raise ValueError('The supplied field was not a vector field.')

    def iter_vfield(scalar_field, i):
        """Return ``vector_field`` called `i` times on ``scalar_field``."""
        return reduce(lambda s, v: v.rcall(s), [vector_field, ]*i, scalar_field)

    def taylor_terms_per_coord(coord_function):
        """Return the series for one of the coordinates."""
        return [param**i*iter_vfield(coord_function, i).rcall(start_point)/factorial(i)
                for i in range(n)]
    coord_sys = coord_sys if coord_sys else start_point._coord_sys
    coord_functions = coord_sys.coord_functions()
    taylor_terms = [taylor_terms_per_coord(f) for f in coord_functions]
    if coeffs:
        return [Matrix(t) for t in zip(*taylor_terms)]
    else:
        return Matrix([sum(c) for c in taylor_terms])


def intcurve_diffequ(vector_field, param, start_point, coord_sys=None):
    r"""Return the differential equation for an integral curve of the field.

    Explanation
    ===========

    Integral curve is a function `\gamma` taking a parameter in `R` to a point
    in the manifold. It verifies the equation:

    `V(f)\big(\gamma(t)\big) = \frac{d}{dt}f\big(\gamma(t)\big)`

    where the given ``vector_field`` is denoted as `V`. This holds for any
    value `t` for the parameter and any scalar field `f`.

    This function returns the differential equation of `\gamma(t)` in terms of the
    coordinate system ``coord_sys``. The equations and expansions are necessarily
    done in coordinate-system-dependent way as there is no other way to
    represent movement between points on the manifold (i.e. there is no such
    thing as a difference of points for a general manifold).

    Parameters
    ==========

    vector_field
        the vector field for which an integral curve will be given

    param
        the argument of the function `\gamma` from R to the curve

    start_point
        the point which corresponds to `\gamma(0)`

    coord_sys
        the coordinate system in which to give the equations

    Returns
    =======

    a tuple of (equations, initial conditions)

    Examples
    ========

    Use the predefined R2 manifold:

    >>> from sympy.abc import t
    >>> from sympy.diffgeom.rn import R2, R2_p, R2_r
    >>> from sympy.diffgeom import intcurve_diffequ

    Specify a starting point and a vector field:

    >>> start_point = R2_r.point([0, 1])
    >>> vector_field = -R2.y*R2.e_x + R2.x*R2.e_y

    Get the equation:

    >>> equations, init_cond = intcurve_diffequ(vector_field, t, start_point)
    >>> equations
    [f_1(t) + Derivative(f_0(t), t), -f_0(t) + Derivative(f_1(t), t)]
    >>> init_cond
    [f_0(0), f_1(0) - 1]

    The series in the polar coordinate system:

    >>> equations, init_cond = intcurve_diffequ(vector_field, t, start_point, R2_p)
    >>> equations
    [Derivative(f_0(t), t), Derivative(f_1(t), t) - 1]
    >>> init_cond
    [f_0(0) - 1, f_1(0) - pi/2]

    See Also
    ========

    intcurve_series

    """
    if contravariant_order(vector_field) != 1 or covariant_order(vector_field):
        raise ValueError('The supplied field was not a vector field.')
    coord_sys = coord_sys if coord_sys else start_point._coord_sys
    gammas = [Function('f_%d' % i)(param) for i in range(
        start_point._coord_sys.dim)]
    arbitrary_p = Point(coord_sys, gammas)
    coord_functions = coord_sys.coord_functions()
    equations = [simplify(diff(cf.rcall(arbitrary_p), param) - vector_field.rcall(cf).rcall(arbitrary_p))
                 for cf in coord_functions]
    init_cond = [simplify(cf.rcall(arbitrary_p).subs(param, 0) - cf.rcall(start_point))
                 for cf in coord_functions]
    return equations, init_cond


###############################################################################
# Helpers
###############################################################################
def dummyfy(args, exprs):
    # TODO Is this a good idea?
    d_args = Matrix([s.as_dummy() for s in args])
    reps = dict(zip(args, d_args))
    d_exprs = Matrix([_sympify(expr).subs(reps) for expr in exprs])
    return d_args, d_exprs

###############################################################################
# Helpers
###############################################################################
def contravariant_order(expr, _strict=False):
    """Return the contravariant order of an expression.

    Examples
    ========

    >>> from sympy.diffgeom import contravariant_order
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.abc import a

    >>> contravariant_order(a)
    0
    >>> contravariant_order(a*R2.x + 2)
    0
    >>> contravariant_order(a*R2.x*R2.e_y + R2.e_x)
    1

    """
    # TODO move some of this to class methods.
    # TODO rewrite using the .as_blah_blah methods
    if isinstance(expr, Add):
        orders = [contravariant_order(e) for e in expr.args]
        if len(set(orders)) != 1:
            raise ValueError('Misformed expression containing contravariant fields of varying order.')
        return orders[0]
    elif isinstance(expr, Mul):
        orders = [contravariant_order(e) for e in expr.args]
        not_zero = [o for o in orders if o != 0]
        if len(not_zero) > 1:
            raise ValueError('Misformed expression containing multiplication between vectors.')
        return 0 if not not_zero else not_zero[0]
    elif isinstance(expr, Pow):
        if covariant_order(expr.base) or covariant_order(expr.exp):
            raise ValueError(
                'Misformed expression containing a power of a vector.')
        return 0
    elif isinstance(expr, BaseVectorField):
        return 1
    elif isinstance(expr, TensorProduct):
        return sum(contravariant_order(a) for a in expr.args)
    elif not _strict or expr.atoms(BaseScalarField):
        return 0
    else:  # If it does not contain anything related to the diffgeom module and it is _strict
        return -1


def covariant_order(expr, _strict=False):
    """Return the covariant order of an expression.

    Examples
    ========

    >>> from sympy.diffgeom import covariant_order
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.abc import a

    >>> covariant_order(a)
    0
    >>> covariant_order(a*R2.x + 2)
    0
    >>> covariant_order(a*R2.x*R2.dy + R2.dx)
    1

    """
    # TODO move some of this to class methods.
    # TODO rewrite using the .as_blah_blah methods
    if isinstance(expr, Add):
        orders = [covariant_order(e) for e in expr.args]
        if len(set(orders)) != 1:
            raise ValueError('Misformed expression containing form fields of varying order.')
        return orders[0]
    elif isinstance(expr, Mul):
        orders = [covariant_order(e) for e in expr.args]
        not_zero = [o for o in orders if o != 0]
        if len(not_zero) > 1:
            raise ValueError('Misformed expression containing multiplication between forms.')
        return 0 if not not_zero else not_zero[0]
    elif isinstance(expr, Pow):
        if covariant_order(expr.base) or covariant_order(expr.exp):
            raise ValueError(
                'Misformed expression containing a power of a form.')
        return 0
    elif isinstance(expr, Differential):
        return covariant_order(*expr.args) + 1
    elif isinstance(expr, TensorProduct):
        return sum(covariant_order(a) for a in expr.args)
    elif not _strict or expr.atoms(BaseScalarField):
        return 0
    else:  # If it does not contain anything related to the diffgeom module and it is _strict
        return -1


###############################################################################
# Coordinate transformation functions
###############################################################################
def vectors_in_basis(expr, to_sys):
    """Transform all base vectors in base vectors of a specified coord basis.
    While the new base vectors are in the new coordinate system basis, any
    coefficients are kept in the old system.

    Examples
    ========

    >>> from sympy.diffgeom import vectors_in_basis
    >>> from sympy.diffgeom.rn import R2_r, R2_p

    >>> vectors_in_basis(R2_r.e_x, R2_p)
    -y*e_theta/(x**2 + y**2) + x*e_rho/sqrt(x**2 + y**2)
    >>> vectors_in_basis(R2_p.e_r, R2_r)
    sin(theta)*e_y + cos(theta)*e_x

    """
    vectors = list(expr.atoms(BaseVectorField))
    new_vectors = []
    for v in vectors:
        cs = v._coord_sys
        jac = cs.jacobian(to_sys, cs.coord_functions())
        new = (jac.T*Matrix(to_sys.base_vectors()))[v._index]
        new_vectors.append(new)
    return expr.subs(list(zip(vectors, new_vectors)))


###############################################################################
# Coordinate-dependent functions
###############################################################################
def twoform_to_matrix(expr):
    """Return the matrix representing the twoform.

    For the twoform `w` return the matrix `M` such that `M[i,j]=w(e_i, e_j)`,
    where `e_i` is the i-th base vector field for the coordinate system in
    which the expression of `w` is given.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import twoform_to_matrix, TensorProduct
    >>> TP = TensorProduct

    >>> twoform_to_matrix(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    Matrix([
    [1, 0],
    [0, 1]])
    >>> twoform_to_matrix(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    Matrix([
    [x, 0],
    [0, 1]])
    >>> twoform_to_matrix(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy) - TP(R2.dx, R2.dy)/2)
    Matrix([
    [   1, 0],
    [-1/2, 1]])

    """
    if covariant_order(expr) != 2 or contravariant_order(expr):
        raise ValueError('The input expression is not a two-form.')
    coord_sys = _find_coords(expr)
    if len(coord_sys) != 1:
        raise ValueError('The input expression concerns more than one '
                         'coordinate systems, hence there is no unambiguous '
                         'way to choose a coordinate system for the matrix.')
    coord_sys = coord_sys.pop()
    vectors = coord_sys.base_vectors()
    expr = expr.expand()
    matrix_content = [[expr.rcall(v1, v2) for v1 in vectors]
                      for v2 in vectors]
    return Matrix(matrix_content)


def metric_to_Christoffel_1st(expr):
    """Return the nested list of Christoffel symbols for the given metric.
    This returns the Christoffel symbol of first kind that represents the
    Levi-Civita connection for the given metric.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import metric_to_Christoffel_1st, TensorProduct
    >>> TP = TensorProduct

    >>> metric_to_Christoffel_1st(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> metric_to_Christoffel_1st(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[1/2, 0], [0, 0]], [[0, 0], [0, 0]]]

    """
    matrix = twoform_to_matrix(expr)
    if not matrix.is_symmetric():
        raise ValueError(
            'The two-form representing the metric is not symmetric.')
    coord_sys = _find_coords(expr).pop()
    deriv_matrices = [matrix.applyfunc(d) for d in coord_sys.base_vectors()]
    indices = list(range(coord_sys.dim))
    christoffel = [[[(deriv_matrices[k][i, j] + deriv_matrices[j][i, k] - deriv_matrices[i][j, k])/2
                     for k in indices]
                    for j in indices]
                   for i in indices]
    return ImmutableDenseNDimArray(christoffel)


def metric_to_Christoffel_2nd(expr):
    """Return the nested list of Christoffel symbols for the given metric.
    This returns the Christoffel symbol of second kind that represents the
    Levi-Civita connection for the given metric.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct
    >>> TP = TensorProduct

    >>> metric_to_Christoffel_2nd(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> metric_to_Christoffel_2nd(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[1/(2*x), 0], [0, 0]], [[0, 0], [0, 0]]]

    """
    ch_1st = metric_to_Christoffel_1st(expr)
    coord_sys = _find_coords(expr).pop()
    indices = list(range(coord_sys.dim))
    # XXX workaround, inverting a matrix does not work if it contains non
    # symbols
    #matrix = twoform_to_matrix(expr).inv()
    matrix = twoform_to_matrix(expr)
    s_fields = set()
    for e in matrix:
        s_fields.update(e.atoms(BaseScalarField))
    s_fields = list(s_fields)
    dums = coord_sys.symbols
    matrix = matrix.subs(list(zip(s_fields, dums))).inv().subs(list(zip(dums, s_fields)))
    # XXX end of workaround
    christoffel = [[[Add(*[matrix[i, l]*ch_1st[l, j, k] for l in indices])
                     for k in indices]
                    for j in indices]
                   for i in indices]
    return ImmutableDenseNDimArray(christoffel)


def metric_to_Riemann_components(expr):
    """Return the components of the Riemann tensor expressed in a given basis.

    Given a metric it calculates the components of the Riemann tensor in the
    canonical basis of the coordinate system in which the metric expression is
    given.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import metric_to_Riemann_components, TensorProduct
    >>> TP = TensorProduct

    >>> metric_to_Riemann_components(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]
    >>> non_trivial_metric = exp(2*R2.r)*TP(R2.dr, R2.dr) + \
        R2.r**2*TP(R2.dtheta, R2.dtheta)
    >>> non_trivial_metric
    exp(2*rho)*TensorProduct(drho, drho) + rho**2*TensorProduct(dtheta, dtheta)
    >>> riemann = metric_to_Riemann_components(non_trivial_metric)
    >>> riemann[0, :, :, :]
    [[[0, 0], [0, 0]], [[0, exp(-2*rho)*rho], [-exp(-2*rho)*rho, 0]]]
    >>> riemann[1, :, :, :]
    [[[0, -1/rho], [1/rho, 0]], [[0, 0], [0, 0]]]

    """
    ch_2nd = metric_to_Christoffel_2nd(expr)
    coord_sys = _find_coords(expr).pop()
    indices = list(range(coord_sys.dim))
    deriv_ch = [[[[d(ch_2nd[i, j, k])
                   for d in coord_sys.base_vectors()]
                  for k in indices]
                 for j in indices]
                for i in indices]
    riemann_a = [[[[deriv_ch[rho][sig][nu][mu] - deriv_ch[rho][sig][mu][nu]
                    for nu in indices]
                   for mu in indices]
                  for sig in indices]
                     for rho in indices]
    riemann_b = [[[[Add(*[ch_2nd[rho, l, mu]*ch_2nd[l, sig, nu] - ch_2nd[rho, l, nu]*ch_2nd[l, sig, mu] for l in indices])
                    for nu in indices]
                   for mu in indices]
                  for sig in indices]
                 for rho in indices]
    riemann = [[[[riemann_a[rho][sig][mu][nu] + riemann_b[rho][sig][mu][nu]
                  for nu in indices]
                     for mu in indices]
                for sig in indices]
               for rho in indices]
    return ImmutableDenseNDimArray(riemann)


def metric_to_Ricci_components(expr):

    """Return the components of the Ricci tensor expressed in a given basis.

    Given a metric it calculates the components of the Ricci tensor in the
    canonical basis of the coordinate system in which the metric expression is
    given.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import metric_to_Ricci_components, TensorProduct
    >>> TP = TensorProduct

    >>> metric_to_Ricci_components(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    [[0, 0], [0, 0]]
    >>> non_trivial_metric = exp(2*R2.r)*TP(R2.dr, R2.dr) + \
                             R2.r**2*TP(R2.dtheta, R2.dtheta)
    >>> non_trivial_metric
    exp(2*rho)*TensorProduct(drho, drho) + rho**2*TensorProduct(dtheta, dtheta)
    >>> metric_to_Ricci_components(non_trivial_metric)
    [[1/rho, 0], [0, exp(-2*rho)*rho]]

    """
    riemann = metric_to_Riemann_components(expr)
    coord_sys = _find_coords(expr).pop()
    indices = list(range(coord_sys.dim))
    ricci = [[Add(*[riemann[k, i, k, j] for k in indices])
              for j in indices]
             for i in indices]
    return ImmutableDenseNDimArray(ricci)

###############################################################################
# Classes for deprecation
###############################################################################

class _deprecated_container:
    # This class gives deprecation warning.
    # When deprecated features are completely deleted, this should be removed as well.
    # See https://github.com/sympy/sympy/pull/19368
    def __init__(self, message, data):
        super().__init__(data)
        self.message = message

    def warn(self):
        sympy_deprecation_warning(
            self.message,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-diffgeom-mutable",
            stacklevel=4
        )

    def __iter__(self):
        self.warn()
        return super().__iter__()

    def __getitem__(self, key):
        self.warn()
        return super().__getitem__(key)

    def __contains__(self, key):
        self.warn()
        return super().__contains__(key)


class _deprecated_list(_deprecated_container, list):
    pass


class _deprecated_dict(_deprecated_container, dict):
    pass


# Import at end to avoid cyclic imports
from sympy.simplify.simplify import simplify

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\codegen.py ===
"""
module for generating C, C++, Fortran77, Fortran90, Julia, Rust
and Octave/Matlab routines that evaluate SymPy expressions.
This module is work in progress.
Only the milestones with a '+' character in the list below have been completed.

--- How is sympy.utilities.codegen different from sympy.printing.ccode? ---

We considered the idea to extend the printing routines for SymPy functions in
such a way that it prints complete compilable code, but this leads to a few
unsurmountable issues that can only be tackled with dedicated code generator:

- For C, one needs both a code and a header file, while the printing routines
  generate just one string. This code generator can be extended to support
  .pyf files for f2py.

- SymPy functions are not concerned with programming-technical issues, such
  as input, output and input-output arguments. Other examples are contiguous
  or non-contiguous arrays, including headers of other libraries such as gsl
  or others.

- It is highly interesting to evaluate several SymPy functions in one C
  routine, eventually sharing common intermediate results with the help
  of the cse routine. This is more than just printing.

- From the programming perspective, expressions with constants should be
  evaluated in the code generator as much as possible. This is different
  for printing.

--- Basic assumptions ---

* A generic Routine data structure describes the routine that must be
  translated into C/Fortran/... code. This data structure covers all
  features present in one or more of the supported languages.

* Descendants from the CodeGen class transform multiple Routine instances
  into compilable code. Each derived class translates into a specific
  language.

* In many cases, one wants a simple workflow. The friendly functions in the
  last part are a simple api on top of the Routine/CodeGen stuff. They are
  easier to use, but are less powerful.

--- Milestones ---

+ First working version with scalar input arguments, generating C code,
  tests
+ Friendly functions that are easier to use than the rigorous
  Routine/CodeGen workflow.
+ Integer and Real numbers as input and output
+ Output arguments
+ InputOutput arguments
+ Sort input/output arguments properly
+ Contiguous array arguments (numpy matrices)
+ Also generate .pyf code for f2py (in autowrap module)
+ Isolate constants and evaluate them beforehand in double precision
+ Fortran 90
+ Octave/Matlab

- Common Subexpression Elimination
- User defined comments in the generated code
- Optional extra include lines for libraries/objects that can eval special
  functions
- Test other C compilers and libraries: gcc, tcc, libtcc, gcc+gsl, ...
- Contiguous array arguments (SymPy matrices)
- Non-contiguous array arguments (SymPy matrices)
- ccode must raise an error when it encounters something that cannot be
  translated into c. ccode(integrate(sin(x)/x, x)) does not make sense.
- Complex numbers as input and output
- A default complex datatype
- Include extra information in the header: date, user, hostname, sha1
  hash, ...
- Fortran 77
- C++
- Python
- Julia
- Rust
- ...

"""

import os
import textwrap
from io import StringIO

from sympy import __version__ as sympy_version
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic
from sympy.printing.c import c_code_printers
from sympy.printing.codeprinter import AssignmentError
from sympy.printing.fortran import FCodePrinter
from sympy.printing.julia import JuliaCodePrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.printing.rust import RustCodePrinter
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
                            MatrixExpr, MatrixSlice)
from sympy.utilities.iterables import is_sequence


__all__ = [
    # description of routines
    "Routine", "DataType", "default_datatypes", "get_default_datatype",
    "Argument", "InputArgument", "OutputArgument", "Result",
    # routines -> code
    "CodeGen", "CCodeGen", "FCodeGen", "JuliaCodeGen", "OctaveCodeGen",
    "RustCodeGen",
    # friendly functions
    "codegen", "make_routine",
]


#
# Description of routines
#


class Routine:
    """Generic description of evaluation routine for set of expressions.

    A CodeGen class can translate instances of this class into code in a
    particular language.  The routine specification covers all the features
    present in these languages.  The CodeGen part must raise an exception
    when certain features are not present in the target language.  For
    example, multiple return values are possible in Python, but not in C or
    Fortran.  Another example: Fortran and Python support complex numbers,
    while C does not.

    """

    def __init__(self, name, arguments, results, local_vars, global_vars):
        """Initialize a Routine instance.

        Parameters
        ==========

        name : string
            Name of the routine.

        arguments : list of Arguments
            These are things that appear in arguments of a routine, often
            appearing on the right-hand side of a function call.  These are
            commonly InputArguments but in some languages, they can also be
            OutputArguments or InOutArguments (e.g., pass-by-reference in C
            code).

        results : list of Results
            These are the return values of the routine, often appearing on
            the left-hand side of a function call.  The difference between
            Results and OutputArguments and when you should use each is
            language-specific.

        local_vars : list of Results
            These are variables that will be defined at the beginning of the
            function.

        global_vars : list of Symbols
            Variables which will not be passed into the function.

        """

        # extract all input symbols and all symbols appearing in an expression
        input_symbols = set()
        symbols = set()
        for arg in arguments:
            if isinstance(arg, OutputArgument):
                symbols.update(arg.expr.free_symbols - arg.expr.atoms(Indexed))
            elif isinstance(arg, InputArgument):
                input_symbols.add(arg.name)
            elif isinstance(arg, InOutArgument):
                input_symbols.add(arg.name)
                symbols.update(arg.expr.free_symbols - arg.expr.atoms(Indexed))
            else:
                raise ValueError("Unknown Routine argument: %s" % arg)

        for r in results:
            if not isinstance(r, Result):
                raise ValueError("Unknown Routine result: %s" % r)
            symbols.update(r.expr.free_symbols - r.expr.atoms(Indexed))

        local_symbols = set()
        for r in local_vars:
            if isinstance(r, Result):
                symbols.update(r.expr.free_symbols - r.expr.atoms(Indexed))
                local_symbols.add(r.name)
            else:
                local_symbols.add(r)

        symbols = {s.label if isinstance(s, Idx) else s for s in symbols}

        # Check that all symbols in the expressions are covered by
        # InputArguments/InOutArguments---subset because user could
        # specify additional (unused) InputArguments or local_vars.
        notcovered = symbols.difference(
            input_symbols.union(local_symbols).union(global_vars))
        if notcovered != set():
            raise ValueError("Symbols needed for output are not in input " +
                             ", ".join([str(x) for x in notcovered]))

        self.name = name
        self.arguments = arguments
        self.results = results
        self.local_vars = local_vars
        self.global_vars = global_vars

    def __str__(self):
        return self.__class__.__name__ + "({name!r}, {arguments}, {results}, {local_vars}, {global_vars})".format(**self.__dict__)

    __repr__ = __str__

    @property
    def variables(self):
        """Returns a set of all variables possibly used in the routine.

        For routines with unnamed return values, the dummies that may or
        may not be used will be included in the set.

        """
        v = set(self.local_vars)
        v.update(arg.name for arg in self.arguments)
        v.update(res.result_var for res in self.results)
        return v

    @property
    def result_variables(self):
        """Returns a list of OutputArgument, InOutArgument and Result.

        If return values are present, they are at the end of the list.
        """
        args = [arg for arg in self.arguments if isinstance(
            arg, (OutputArgument, InOutArgument))]
        args.extend(self.results)
        return args


class DataType:
    """Holds strings for a certain datatype in different languages."""
    def __init__(self, cname, fname, pyname, jlname, octname, rsname):
        self.cname = cname
        self.fname = fname
        self.pyname = pyname
        self.jlname = jlname
        self.octname = octname
        self.rsname = rsname


default_datatypes = {
    "int": DataType("int", "INTEGER*4", "int", "", "", "i32"),
    "float": DataType("double", "REAL*8", "float", "", "", "f64"),
    "complex": DataType("double", "COMPLEX*16", "complex", "", "", "float") #FIXME:
       # complex is only supported in fortran, python, julia, and octave.
       # So to not break c or rust code generation, we stick with double or
       # float, respectively (but actually should raise an exception for
       # explicitly complex variables (x.is_complex==True))
}


COMPLEX_ALLOWED = False
def get_default_datatype(expr, complex_allowed=None):
    """Derives an appropriate datatype based on the expression."""
    if complex_allowed is None:
        complex_allowed = COMPLEX_ALLOWED
    if complex_allowed:
        final_dtype = "complex"
    else:
        final_dtype = "float"
    if expr.is_integer:
        return default_datatypes["int"]
    elif expr.is_real:
        return default_datatypes["float"]
    elif isinstance(expr, MatrixBase):
        #check all entries
        dt = "int"
        for element in expr:
            if dt == "int" and not element.is_integer:
                dt = "float"
            if dt == "float" and not element.is_real:
                return default_datatypes[final_dtype]
        return default_datatypes[dt]
    else:
        return default_datatypes[final_dtype]


class Variable:
    """Represents a typed variable."""

    def __init__(self, name, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol or MatrixSymbol

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimensions : sequence containing tuples, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        if not isinstance(name, (Symbol, MatrixSymbol)):
            raise TypeError("The first argument must be a SymPy symbol.")
        if datatype is None:
            datatype = get_default_datatype(name)
        elif not isinstance(datatype, DataType):
            raise TypeError("The (optional) `datatype' argument must be an "
                            "instance of the DataType class.")
        if dimensions and not isinstance(dimensions, (tuple, list)):
            raise TypeError(
                "The dimensions argument must be a sequence of tuples")

        self._name = name
        self._datatype = {
            'C': datatype.cname,
            'FORTRAN': datatype.fname,
            'JULIA': datatype.jlname,
            'OCTAVE': datatype.octname,
            'PYTHON': datatype.pyname,
            'RUST': datatype.rsname,
        }
        self.dimensions = dimensions
        self.precision = precision

    def __str__(self):
        return "%s(%r)" % (self.__class__.__name__, self.name)

    __repr__ = __str__

    @property
    def name(self):
        return self._name

    def get_datatype(self, language):
        """Returns the datatype string for the requested language.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.utilities.codegen import Variable
        >>> x = Variable(Symbol('x'))
        >>> x.get_datatype('c')
        'double'
        >>> x.get_datatype('fortran')
        'REAL*8'

        """
        try:
            return self._datatype[language.upper()]
        except KeyError:
            raise CodeGenError("Has datatypes for languages: %s" %
                    ", ".join(self._datatype))


class Argument(Variable):
    """An abstract Argument data structure: a name and a data type.

    This structure is refined in the descendants below.

    """
    pass


class InputArgument(Argument):
    pass


class ResultBase:
    """Base class for all "outgoing" information from a routine.

    Objects of this class stores a SymPy expression, and a SymPy object
    representing a result variable that will be used in the generated code
    only if necessary.

    """
    def __init__(self, expr, result_var):
        self.expr = expr
        self.result_var = result_var

    def __str__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.expr,
            self.result_var)

    __repr__ = __str__


class OutputArgument(Argument, ResultBase):
    """OutputArgument are always initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol, MatrixSymbol
            The name of this variable.  When used for code generation, this
            might appear, for example, in the prototype of function in the
            argument list.

        result_var : Symbol, Indexed
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".

        expr : object
            The expression that should be output, typically a SymPy
            expression.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimensions : sequence containing tuples, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """

        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.result_var, self.expr)

    __repr__ = __str__


class InOutArgument(Argument, ResultBase):
    """InOutArgument are never initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        if not datatype:
            datatype = get_default_datatype(expr)
        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)
    __init__.__doc__ = OutputArgument.__init__.__doc__


    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.expr,
            self.result_var)

    __repr__ = __str__


class Result(Variable, ResultBase):
    """An expression for a return value.

    The name result is used to avoid conflicts with the reserved word
    "return" in the Python language.  It is also shorter than ReturnValue.

    These may or may not need a name in the destination (e.g., "return(x*y)"
    might return a value without ever naming it).

    """

    def __init__(self, expr, name=None, result_var=None, datatype=None,
                 dimensions=None, precision=None):
        """Initialize a return value.

        Parameters
        ==========

        expr : SymPy expression

        name : Symbol, MatrixSymbol, optional
            The name of this return variable.  When used for code generation,
            this might appear, for example, in the prototype of function in a
            list of return values.  A dummy name is generated if omitted.

        result_var : Symbol, Indexed, optional
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".  Defaults to
            `name` if omitted.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the expr argument.

        dimensions : sequence containing tuples, optional
            If present, this variable is interpreted as an array,
            where this sequence of tuples specifies (lower, upper)
            bounds for each index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        # Basic because it is the base class for all types of expressions
        if not isinstance(expr, (Basic, MatrixBase)):
            raise TypeError("The first argument must be a SymPy expression.")

        if name is None:
            name = 'result_%d' % abs(hash(expr))

        if datatype is None:
            #try to infer data type from the expression
            datatype = get_default_datatype(expr)

        if isinstance(name, str):
            if isinstance(expr, (MatrixBase, MatrixExpr)):
                name = MatrixSymbol(name, *expr.shape)
            else:
                name = Symbol(name)

        if result_var is None:
            result_var = name

        Variable.__init__(self, name, datatype=datatype,
                          dimensions=dimensions, precision=precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.expr, self.name,
            self.result_var)

    __repr__ = __str__


#
# Transformation of routine objects into code
#

class CodeGen:
    """Abstract class for the code generators."""

    printer = None  # will be set to an instance of a CodePrinter subclass

    def _indent_code(self, codelines):
        return self.printer.indent_code(codelines)

    def _printer_method_with_settings(self, method, settings=None, *args, **kwargs):
        settings = settings or {}
        ori = {k: self.printer._settings[k] for k in settings}
        for k, v in settings.items():
            self.printer._settings[k] = v
        result = getattr(self.printer, method)(*args, **kwargs)
        for k, v in ori.items():
            self.printer._settings[k] = v
        return result

    def _get_symbol(self, s):
        """Returns the symbol as fcode prints it."""
        if self.printer._settings['human']:
            expr_str = self.printer.doprint(s)
        else:
            constants, not_supported, expr_str = self.printer.doprint(s)
            if constants or not_supported:
                raise ValueError("Failed to print %s" % str(s))
        return expr_str.strip()

    def __init__(self, project="project", cse=False):
        """Initialize a code generator.

        Derived classes will offer more options that affect the generated
        code.

        """
        self.project = project
        self.cse = cse

    def routine(self, name, expr, argument_sequence=None, global_vars=None):
        """Creates an Routine object that is appropriate for this language.

        This implementation is appropriate for at least C/Fortran.  Subclasses
        can override this if necessary.

        Here, we assume at most one return value (the l-value) which must be
        scalar.  Additional outputs are OutputArguments (e.g., pointers on
        right-hand-side or pass-by-reference).  Matrices are always returned
        via OutputArguments.  If ``argument_sequence`` is None, arguments will
        be ordered alphabetically, but with all InputArguments first, and then
        OutputArgument and InOutArguments.

        """

        if self.cse:
            from sympy.simplify.cse_main import cse

            if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
                if not expr:
                    raise ValueError("No expression given")
                for e in expr:
                    if not e.is_Equality:
                        raise CodeGenError("Lists of expressions must all be Equalities. {} is not.".format(e))

                # create a list of right hand sides and simplify them
                rhs = [e.rhs for e in expr]
                common, simplified = cse(rhs)

                # pack the simplified expressions back up with their left hand sides
                expr = [Equality(e.lhs, rhs) for e, rhs in zip(expr, simplified)]
            else:
                if isinstance(expr, Equality):
                    common, simplified = cse(expr.rhs) #, ignore=in_out_args)
                    expr = Equality(expr.lhs, simplified[0])
                else:
                    common, simplified = cse(expr)
                    expr = simplified

            local_vars = [Result(b,a) for a,b in common]
            local_symbols = {a for a,_ in common}
            local_expressions = Tuple(*[b for _,b in common])
        else:
            local_expressions = Tuple()

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        if self.cse:
            if {i.label for i in expressions.atoms(Idx)} != set():
                raise CodeGenError("CSE and Indexed expressions do not play well together yet")
        else:
            # local variables for indexed expressions
            local_vars = {i.label for i in expressions.atoms(Idx)}
            local_symbols = local_vars

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        symbols = (expressions.free_symbols | local_expressions.free_symbols) - local_symbols - global_vars
        new_symbols = set()
        new_symbols.update(symbols)

        for symbol in symbols:
            if isinstance(symbol, Idx):
                new_symbols.remove(symbol)
                new_symbols.update(symbol.args[1].free_symbols)
            if isinstance(symbol, Indexed):
                new_symbols.remove(symbol)
        symbols = new_symbols

        # Decide whether to use output argument or return value
        return_val = []
        output_args = []
        for expr in expressions:
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                elif isinstance(out_arg, Symbol):
                    dims = []
                    symbol = out_arg
                elif isinstance(out_arg, MatrixSymbol):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg
                else:
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                if expr.has(symbol):
                    output_args.append(
                        InOutArgument(symbol, out_arg, expr, dimensions=dims))
                else:
                    output_args.append(
                        OutputArgument(symbol, out_arg, expr, dimensions=dims))

                # remove duplicate arguments when they are not local variables
                if symbol not in local_vars:
                    # avoid duplicate arguments
                    symbols.remove(symbol)
            elif isinstance(expr, (ImmutableMatrix, MatrixSlice)):
                # Create a "dummy" MatrixSymbol to use as the Output arg
                out_arg = MatrixSymbol('out_%s' % abs(hash(expr)), *expr.shape)
                dims = tuple([(S.Zero, dim - 1) for dim in out_arg.shape])
                output_args.append(
                    OutputArgument(out_arg, out_arg, expr, dimensions=dims))
            else:
                return_val.append(Result(expr))

        arg_list = []

        # setup input argument list

        # helper to get dimensions for data for array-like args
        def dimensions(s):
            return [(S.Zero, dim - 1) for dim in s.shape]

        array_symbols = {}
        for array in expressions.atoms(Indexed) | local_expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol) | local_expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            if symbol in array_symbols:
                array = array_symbols[symbol]
                metadata = {'dimensions': dimensions(array)}
            else:
                metadata = {}

            arg_list.append(InputArgument(symbol, **metadata))

        output_args.sort(key=lambda x: str(x.name))
        arg_list.extend(output_args)

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    if isinstance(symbol, (IndexedBase, MatrixSymbol)):
                        metadata = {'dimensions': dimensions(symbol)}
                    else:
                        metadata = {}
                    new_args.append(InputArgument(symbol, **metadata))
            arg_list = new_args

        return Routine(name, arg_list, return_val, local_vars, global_vars)

    def write(self, routines, prefix, to_files=False, header=True, empty=True):
        """Writes all the source code files for the given routines.

        The generated source is returned as a list of (filename, contents)
        tuples, or is written to files (see below).  Each filename consists
        of the given prefix, appended with an appropriate extension.

        Parameters
        ==========

        routines : list
            A list of Routine instances to be written

        prefix : string
            The prefix for the output files

        to_files : bool, optional
            When True, the output is written to files.  Otherwise, a list
            of (filename, contents) tuples is returned.  [default: False]

        header : bool, optional
            When True, a header comment is included on top of each source
            file. [default: True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files. [default: True]

        """
        if to_files:
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                with open(filename, "w") as f:
                    dump_fn(self, routines, f, prefix, header, empty)
        else:
            result = []
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                contents = StringIO()
                dump_fn(self, routines, contents, prefix, header, empty)
                result.append((filename, contents.getvalue()))
            return result

    def dump_code(self, routines, f, prefix, header=True, empty=True):
        """Write the code by calling language specific methods.

        The generated file contains all the definitions of the routines in
        low-level code and refers to the header file if appropriate.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """

        code_lines = self._preprocessor_statements(prefix)

        for routine in routines:
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_opening(routine))
            code_lines.extend(self._declare_arguments(routine))
            code_lines.extend(self._declare_globals(routine))
            code_lines.extend(self._declare_locals(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._call_printer(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_ending(routine))

        code_lines = self._indent_code(''.join(code_lines))

        if header:
            code_lines = ''.join(self._get_header() + [code_lines])

        if code_lines:
            f.write(code_lines)


class CodeGenError(Exception):
    pass


class CodeGenArgumentListError(Exception):
    @property
    def missing_args(self):
        return self.args[1]


header_comment = """Code generated with SymPy %(version)s

See http://www.sympy.org/ for more information.

This file is part of '%(project)s'
"""


class CCodeGen(CodeGen):
    """Generator for C code.

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.c and <prefix>.h respectively.

    """

    code_extension = "c"
    interface_extension = "h"
    standard = 'c99'

    def __init__(self, project="project", printer=None,
                 preprocessor_statements=None, cse=False):
        super().__init__(project=project, cse=cse)
        self.printer = printer or c_code_printers[self.standard.lower()]()

        self.preprocessor_statements = preprocessor_statements
        if preprocessor_statements is None:
            self.preprocessor_statements = ['#include <math.h>']

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("/" + "*"*78 + '\n')
        tmp = header_comment % {"version": sympy_version,
                                "project": self.project}
        for line in tmp.splitlines():
            code_lines.append(" *%s*\n" % line.center(76))
        code_lines.append(" " + "*"*78 + "/\n")
        return code_lines

    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        if len(routine.results) > 1:
            raise CodeGenError("C only supports a single or no return value.")
        elif len(routine.results) == 1:
            ctype = routine.results[0].get_datatype('C')
        else:
            ctype = "void"

        type_args = []
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append((arg.get_datatype('C'), "*%s" % name))
            else:
                type_args.append((arg.get_datatype('C'), name))
        arguments = ", ".join([ "%s %s" % t for t in type_args])
        return "%s %s(%s)" % (ctype, routine.name, arguments)

    def _preprocessor_statements(self, prefix):
        code_lines = []
        code_lines.append('#include "{}.h"'.format(os.path.basename(prefix)))
        code_lines.extend(self.preprocessor_statements)
        code_lines = ['{}\n'.format(l) for l in code_lines]
        return code_lines

    def _get_routine_opening(self, routine):
        prototype = self.get_prototype(routine)
        return ["%s {\n" % prototype]

    def _declare_arguments(self, routine):
        # arguments are declared in prototype
        return []

    def _declare_globals(self, routine):
        # global variables are not explicitly declared within C functions
        return []

    def _declare_locals(self, routine):

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        code_lines = []
        for result in routine.local_vars:

            # local variables that are simple symbols such as those used as indices into
            # for loops are defined declared elsewhere.
            if not isinstance(result, Result):
                continue

            if result.name != result.result_var:
                raise CodeGen("Result variable and name should match: {}".format(result))
            assign_to = result.name
            t = result.get_datatype('c')
            if isinstance(result.expr, (MatrixBase, MatrixExpr)):
                dims = result.expr.shape
                code_lines.append("{} {}[{}];\n".format(t, str(assign_to), dims[0]*dims[1]))
                prefix = ""
            else:
                prefix = "const {} ".format(t)

            constants, not_c, c_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "dereference": dereference, "strict": False},
                result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))

            code_lines.append("{}{}\n".format(prefix, c_expr))

        return code_lines

    def _call_printer(self, routine):
        code_lines = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        return_val = None
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name + "_result"
                t = result.get_datatype('c')
                code_lines.append("{} {};\n".format(t, str(assign_to)))
                return_val = assign_to
            else:
                assign_to = result.result_var

            try:
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', {"human": False, "dereference": dereference, "strict": False},
                    result.expr, assign_to=assign_to)
            except AssignmentError:
                assign_to = result.result_var
                code_lines.append(
                    "%s %s;\n" % (result.get_datatype('c'), str(assign_to)))
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', {"human": False, "dereference": dereference, "strict": False},
                    result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))
            code_lines.append("%s\n" % c_expr)

        if return_val:
            code_lines.append("   return %s;\n" % return_val)
        return code_lines

    def _get_routine_ending(self, routine):
        return ["}\n"]

    def dump_c(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)
    dump_c.extension = code_extension  # type: ignore
    dump_c.__doc__ = CodeGen.dump_code.__doc__

    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the C header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to construct the include guards.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        if header:
            print(''.join(self._get_header()), file=f)
        guard_name = "%s__%s__H" % (self.project.replace(
            " ", "_").upper(), prefix.replace("/", "_").upper())
        # include guards
        if empty:
            print(file=f)
        print("#ifndef %s" % guard_name, file=f)
        print("#define %s" % guard_name, file=f)
        if empty:
            print(file=f)
        # declaration of the function prototypes
        for routine in routines:
            prototype = self.get_prototype(routine)
            print("%s;" % prototype, file=f)
        # end if include guards
        if empty:
            print(file=f)
        print("#endif", file=f)
        if empty:
            print(file=f)
    dump_h.extension = interface_extension  # type: ignore

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_c, dump_h]

class C89CodeGen(CCodeGen):
    standard = 'C89'

class C99CodeGen(CCodeGen):
    standard = 'C99'

class FCodeGen(CodeGen):
    """Generator for Fortran 95 code

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.f90 and <prefix>.h respectively.

    """

    code_extension = "f90"
    interface_extension = "h"

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or FCodePrinter()

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("!" + "*"*78 + '\n')
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            code_lines.append("!*%s*\n" % line.center(76))
        code_lines.append("!" + "*"*78 + '\n')
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the fortran routine."""
        code_list = []
        if len(routine.results) > 1:
            raise CodeGenError(
                "Fortran only supports a single or no return value.")
        elif len(routine.results) == 1:
            result = routine.results[0]
            code_list.append(result.get_datatype('fortran'))
            code_list.append("function")
        else:
            code_list.append("subroutine")

        args = ", ".join("%s" % self._get_symbol(arg.name)
                        for arg in routine.arguments)

        call_sig = "{}({})\n".format(routine.name, args)
        # Fortran 95 requires all lines be less than 132 characters, so wrap
        # this line before appending.
        call_sig = ' &\n'.join(textwrap.wrap(call_sig,
                                             width=60,
                                             break_long_words=False)) + '\n'
        code_list.append(call_sig)
        code_list = [' '.join(code_list)]
        code_list.append('implicit none\n')
        return code_list

    def _declare_arguments(self, routine):
        # argument type declarations
        code_list = []
        array_list = []
        scalar_list = []
        for arg in routine.arguments:

            if isinstance(arg, InputArgument):
                typeinfo = "%s, intent(in)" % arg.get_datatype('fortran')
            elif isinstance(arg, InOutArgument):
                typeinfo = "%s, intent(inout)" % arg.get_datatype('fortran')
            elif isinstance(arg, OutputArgument):
                typeinfo = "%s, intent(out)" % arg.get_datatype('fortran')
            else:
                raise CodeGenError("Unknown Argument type: %s" % type(arg))

            fprint = self._get_symbol

            if arg.dimensions:
                # fortran arrays start at 1
                dimstr = ", ".join(["%s:%s" % (
                    fprint(dim[0] + 1), fprint(dim[1] + 1))
                    for dim in arg.dimensions])
                typeinfo += ", dimension(%s)" % dimstr
                array_list.append("%s :: %s\n" % (typeinfo, fprint(arg.name)))
            else:
                scalar_list.append("%s :: %s\n" % (typeinfo, fprint(arg.name)))

        # scalars first, because they can be used in array declarations
        code_list.extend(scalar_list)
        code_list.extend(array_list)

        return code_list

    def _declare_globals(self, routine):
        # Global variables not explicitly declared within Fortran 90 functions.
        # Note: a future F77 mode may need to generate "common" blocks.
        return []

    def _declare_locals(self, routine):
        code_list = []
        for var in sorted(routine.local_vars, key=str):
            typeinfo = get_default_datatype(var)
            code_list.append("%s :: %s\n" % (
                typeinfo.fname, self._get_symbol(var)))
        return code_list

    def _get_routine_ending(self, routine):
        """Returns the closing statements of the fortran routine."""
        if len(routine.results) == 1:
            return ["end function\n"]
        else:
            return ["end subroutine\n"]

    def get_interface(self, routine):
        """Returns a string for the function interface.

        The routine should have a single result object, which can be None.
        If the routine has multiple result objects, a CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        prototype = [ "interface\n" ]
        prototype.extend(self._get_routine_opening(routine))
        prototype.extend(self._declare_arguments(routine))
        prototype.extend(self._get_routine_ending(routine))
        prototype.append("end interface\n")

        return "".join(prototype)

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name
            elif isinstance(result, (OutputArgument, InOutArgument)):
                assign_to = result.result_var

            constants, not_fortran, f_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "source_format": 'free', "standard": 95, "strict": False},
                result.expr, assign_to=assign_to)

            for obj, v in sorted(constants, key=str):
                t = get_default_datatype(obj)
                declarations.append(
                    "%s, parameter :: %s = %s\n" % (t.fname, obj, v))
            for obj in sorted(not_fortran, key=str):
                t = get_default_datatype(obj)
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append("%s :: %s\n" % (t.fname, name))

            code_lines.append("%s\n" % f_expr)
        return declarations + code_lines

    def _indent_code(self, codelines):
        return self._printer_method_with_settings(
            'indent_code', {"human": False, "source_format": 'free', "strict": False}, codelines)

    def dump_f95(self, routines, f, prefix, header=True, empty=True):
        # check that symbols are unique with ignorecase
        for r in routines:
            lowercase = {str(x).lower() for x in r.variables}
            orig_case = {str(x) for x in r.variables}
            if len(lowercase) < len(orig_case):
                raise CodeGenError("Fortran ignores case. Got symbols: %s" %
                        (", ".join([str(var) for var in r.variables])))
        self.dump_code(routines, f, prefix, header, empty)
    dump_f95.extension = code_extension  # type: ignore
    dump_f95.__doc__ = CodeGen.dump_code.__doc__

    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the interface to a header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        if header:
            print(''.join(self._get_header()), file=f)
        if empty:
            print(file=f)
        # declaration of the function prototypes
        for routine in routines:
            prototype = self.get_interface(routine)
            f.write(prototype)
        if empty:
            print(file=f)
    dump_h.extension = interface_extension  # type: ignore

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_f95, dump_h]


class JuliaCodeGen(CodeGen):
    """Generator for Julia code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.jl.

    """

    code_extension = "jl"

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or JuliaCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Julia."""

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # local variables
        local_vars = {i.label for i in expressions.atoms(Idx)}

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        old_symbols = expressions.free_symbols - local_vars - global_vars
        symbols = set()
        for s in old_symbols:
            if isinstance(s, Idx):
                symbols.update(s.args[1].free_symbols)
            elif not isinstance(s, Indexed):
                symbols.add(s)

        # Julia supports multiple return values
        return_vals = []
        output_args = []
        for (i, expr) in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.One, dim) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                    output_args.append(InOutArgument(symbol, out_arg, expr, dimensions=dims))
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    # this is a pure output: remove from the symbols list, so
                    # it doesn't become an input.
                    symbols.remove(symbol)

            else:
                # we have no name for this output
                return_vals.append(Result(expr, name='out%d' % (i+1)))

        # setup input argument list
        output_args.sort(key=lambda x: str(x.name))
        arg_list = list(output_args)
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        return Routine(name, arg_list, return_vals, local_vars, global_vars)

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            if line == '':
                code_lines.append("#\n")
            else:
                code_lines.append("#   %s\n" % line)
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        code_list = []
        code_list.append("function ")

        # Inputs
        args = []
        for arg in routine.arguments:
            if isinstance(arg, OutputArgument):
                raise CodeGenError("Julia: invalid argument of type %s" %
                                   str(type(arg)))
            if isinstance(arg, (InputArgument, InOutArgument)):
                args.append("%s" % self._get_symbol(arg.name))
        args = ", ".join(args)
        code_list.append("%s(%s)\n" % (routine.name, args))
        code_list = [ "".join(code_list) ]

        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        return []

    def _declare_locals(self, routine):
        return []

    def _get_routine_ending(self, routine):
        outs = []
        for result in routine.results:
            if isinstance(result, Result):
                # Note: name not result_var; want `y` not `y[i]` for Indexed
                s = self._get_symbol(result.name)
            else:
                raise CodeGenError("unexpected object in Routine results")
            outs.append(s)
        return ["return " + ", ".join(outs) + "\nend\n"]

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for result in routine.results:
            if isinstance(result, Result):
                assign_to = result.result_var
            else:
                raise CodeGenError("unexpected object in Routine results")

            constants, not_supported, jl_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            for obj, v in sorted(constants, key=str):
                declarations.append(
                    "%s = %s\n" % (obj, v))
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append(
                    "# unsupported: %s\n" % (name))
            code_lines.append("%s\n" % (jl_expr))
        return declarations + code_lines

    def _indent_code(self, codelines):
        # Note that indenting seems to happen twice, first
        # statement-by-statement by JuliaPrinter then again here.
        p = JuliaCodePrinter({'human': False, "strict": False})
        return p.indent_code(codelines)

    def dump_jl(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_jl.extension = code_extension  # type: ignore
    dump_jl.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_jl]


class OctaveCodeGen(CodeGen):
    """Generator for Octave code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.m.

    Octave .m files usually contain one function.  That function name should
    match the filename (``prefix``).  If you pass multiple ``name_expr`` pairs,
    the latter ones are presumed to be private functions accessed by the
    primary function.

    You should only pass inputs to ``argument_sequence``: outputs are ordered
    according to their order in ``name_expr``.

    """

    code_extension = "m"

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or OctaveCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Octave."""

        # FIXME: this is probably general enough for other high-level
        # languages, perhaps its the C/Fortran one that is specialized!

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # local variables
        local_vars = {i.label for i in expressions.atoms(Idx)}

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        old_symbols = expressions.free_symbols - local_vars - global_vars
        symbols = set()
        for s in old_symbols:
            if isinstance(s, Idx):
                symbols.update(s.args[1].free_symbols)
            elif not isinstance(s, Indexed):
                symbols.add(s)

        # Octave supports multiple return values
        return_vals = []
        for (i, expr) in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    symbol = out_arg.base.label
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    # this is a pure output: remove from the symbols list, so
                    # it doesn't become an input.
                    symbols.remove(symbol)

            else:
                # we have no name for this output
                return_vals.append(Result(expr, name='out%d' % (i+1)))

        # setup input argument list
        arg_list = []
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        return Routine(name, arg_list, return_vals, local_vars, global_vars)

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            if line == '':
                code_lines.append("%\n")
            else:
                code_lines.append("%%   %s\n" % line)
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        code_list = []
        code_list.append("function ")

        # Outputs
        outs = []
        for result in routine.results:
            if isinstance(result, Result):
                # Note: name not result_var; want `y` not `y(i)` for Indexed
                s = self._get_symbol(result.name)
            else:
                raise CodeGenError("unexpected object in Routine results")
            outs.append(s)
        if len(outs) > 1:
            code_list.append("[" + (", ".join(outs)) + "]")
        else:
            code_list.append("".join(outs))
        code_list.append(" = ")

        # Inputs
        args = []
        for arg in routine.arguments:
            if isinstance(arg, (OutputArgument, InOutArgument)):
                raise CodeGenError("Octave: invalid argument of type %s" %
                                   str(type(arg)))
            if isinstance(arg, InputArgument):
                args.append("%s" % self._get_symbol(arg.name))
        args = ", ".join(args)
        code_list.append("%s(%s)\n" % (routine.name, args))
        code_list = [ "".join(code_list) ]

        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        if not routine.global_vars:
            return []
        s = " ".join(sorted([self._get_symbol(g) for g in routine.global_vars]))
        return ["global " + s + "\n"]

    def _declare_locals(self, routine):
        return []

    def _get_routine_ending(self, routine):
        return ["end\n"]

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for result in routine.results:
            if isinstance(result, Result):
                assign_to = result.result_var
            else:
                raise CodeGenError("unexpected object in Routine results")

            constants, not_supported, oct_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            for obj, v in sorted(constants, key=str):
                declarations.append(
                    "  %s = %s;  %% constant\n" % (obj, v))
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append(
                    "  %% unsupported: %s\n" % (name))
            code_lines.append("%s\n" % (oct_expr))
        return declarations + code_lines

    def _indent_code(self, codelines):
        return self._printer_method_with_settings(
            'indent_code', {"human": False, "strict": False}, codelines)

    def dump_m(self, routines, f, prefix, header=True, empty=True, inline=True):
        # Note used to call self.dump_code() but we need more control for header

        code_lines = self._preprocessor_statements(prefix)

        for i, routine in enumerate(routines):
            if i > 0:
                if empty:
                    code_lines.append("\n")
            code_lines.extend(self._get_routine_opening(routine))
            if i == 0:
                if routine.name != prefix:
                    raise ValueError('Octave function name should match prefix')
                if header:
                    code_lines.append("%" + prefix.upper() +
                                      "  Autogenerated by SymPy\n")
                    code_lines.append(''.join(self._get_header()))
            code_lines.extend(self._declare_arguments(routine))
            code_lines.extend(self._declare_globals(routine))
            code_lines.extend(self._declare_locals(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._call_printer(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_ending(routine))

        code_lines = self._indent_code(''.join(code_lines))

        if code_lines:
            f.write(code_lines)

    dump_m.extension = code_extension  # type: ignore
    dump_m.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_m]

class RustCodeGen(CodeGen):
    """Generator for Rust code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.rs

    """

    code_extension = "rs"

    def __init__(self, project="project", printer=None):
        super().__init__(project=project)
        self.printer = printer or RustCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Rust."""

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # local variables
        local_vars = {i.label for i in expressions.atoms(Idx)}

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        symbols = expressions.free_symbols - local_vars - global_vars - expressions.atoms(Indexed)

        # Rust supports multiple return values
        return_vals = []
        output_args = []
        for (i, expr) in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.One, dim) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                    output_args.append(InOutArgument(symbol, out_arg, expr, dimensions=dims))
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    # this is a pure output: remove from the symbols list, so
                    # it doesn't become an input.
                    symbols.remove(symbol)

            else:
                # we have no name for this output
                return_vals.append(Result(expr, name='out%d' % (i+1)))

        # setup input argument list
        output_args.sort(key=lambda x: str(x.name))
        arg_list = list(output_args)
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        return Routine(name, arg_list, return_vals, local_vars, global_vars)


    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("/*\n")
        tmp = header_comment % {"version": sympy_version,
                                "project": self.project}
        for line in tmp.splitlines():
            code_lines.append((" *%s" % line.center(76)).rstrip() + "\n")
        code_lines.append(" */\n")
        return code_lines

    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        results = [i.get_datatype('Rust') for i in routine.results]

        if len(results) == 1:
            rstype = " -> " + results[0]
        elif len(routine.results) > 1:
            rstype = " -> (" + ", ".join(results) + ")"
        else:
            rstype = ""

        type_args = []
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append(("*%s" % name, arg.get_datatype('Rust')))
            else:
                type_args.append((name, arg.get_datatype('Rust')))
        arguments = ", ".join([ "%s: %s" % t for t in type_args])
        return "fn %s(%s)%s" % (routine.name, arguments, rstype)

    def _preprocessor_statements(self, prefix):
        code_lines = []
        # code_lines.append("use std::f64::consts::*;\n")
        return code_lines

    def _get_routine_opening(self, routine):
        prototype = self.get_prototype(routine)
        return ["%s {\n" % prototype]

    def _declare_arguments(self, routine):
        # arguments are declared in prototype
        return []

    def _declare_globals(self, routine):
        # global variables are not explicitly declared within C functions
        return []

    def _declare_locals(self, routine):
        # loop variables are declared in loop statement
        return []

    def _call_printer(self, routine):

        code_lines = []
        declarations = []
        returns = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        for result in routine.results:
            if isinstance(result, Result):
                assign_to = result.result_var
                returns.append(str(result.result_var))
            else:
                raise CodeGenError("unexpected object in Routine results")

            constants, not_supported, rs_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "strict": False}, result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                declarations.append("const %s: f64 = %s;\n" % (name, value))

            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append("// unsupported: %s\n" % (name))

            code_lines.append("let %s\n" % rs_expr)

        if len(returns) > 1:
            returns = ['(' + ', '.join(returns) + ')']

        returns.append('\n')

        return declarations + code_lines + returns

    def _get_routine_ending(self, routine):
        return ["}\n"]

    def dump_rs(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_rs.extension = code_extension  # type: ignore
    dump_rs.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_rs]




def get_code_generator(language, project=None, standard=None, printer = None):
    if language == 'C':
        if standard is None:
            pass
        elif standard.lower() == 'c89':
            language = 'C89'
        elif standard.lower() == 'c99':
            language = 'C99'
    CodeGenClass = {"C": CCodeGen, "C89": C89CodeGen, "C99": C99CodeGen,
                    "F95": FCodeGen, "JULIA": JuliaCodeGen,
                    "OCTAVE": OctaveCodeGen,
                    "RUST": RustCodeGen}.get(language.upper())
    if CodeGenClass is None:
        raise ValueError("Language '%s' is not supported." % language)
    return CodeGenClass(project, printer)


#
# Friendly functions
#


def codegen(name_expr, language=None, prefix=None, project="project",
            to_files=False, header=True, empty=True, argument_sequence=None,
            global_vars=None, standard=None, code_gen=None, printer=None):
    """Generate source code for expressions in a given language.

    Parameters
    ==========

    name_expr : tuple, or list of tuples
        A single (name, expression) tuple or a list of (name, expression)
        tuples.  Each tuple corresponds to a routine.  If the expression is
        an equality (an instance of class Equality) the left hand side is
        considered an output argument.  If expression is an iterable, then
        the routine will have multiple outputs.

    language : string,
        A string that indicates the source code language.  This is case
        insensitive.  Currently, 'C', 'F95' and 'Octave' are supported.
        'Octave' generates code compatible with both Octave and Matlab.

    prefix : string, optional
        A prefix for the names of the files that contain the source code.
        Language-dependent suffixes will be appended.  If omitted, the name
        of the first name_expr tuple is used.

    project : string, optional
        A project name, used for making unique preprocessor instructions.
        [default: "project"]

    to_files : bool, optional
        When True, the code will be written to one or more files with the
        given prefix, otherwise strings with the names and contents of
        these files are returned. [default: False]

    header : bool, optional
        When True, a header is written on top of each source file.
        [default: True]

    empty : bool, optional
        When True, empty lines are used to structure the code.
        [default: True]

    argument_sequence : iterable, optional
        Sequence of arguments for the routine in a preferred order.  A
        CodeGenError is raised if required arguments are missing.
        Redundant arguments are used without warning.  If omitted,
        arguments will be ordered alphabetically, but with all input
        arguments first, and then output or in-out arguments.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    standard : string, optional

    code_gen : CodeGen instance, optional
        An instance of a CodeGen subclass. Overrides ``language``.

    printer : Printer instance, optional
        An instance of a Printer subclass.

    Examples
    ========

    >>> from sympy.utilities.codegen import codegen
    >>> from sympy.abc import x, y, z
    >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    ...     ("f", x+y*z), "C89", "test", header=False, empty=False)
    >>> print(c_name)
    test.c
    >>> print(c_code)
    #include "test.h"
    #include <math.h>
    double f(double x, double y, double z) {
       double f_result;
       f_result = x + y*z;
       return f_result;
    }
    <BLANKLINE>
    >>> print(h_name)
    test.h
    >>> print(c_header)
    #ifndef PROJECT__TEST__H
    #define PROJECT__TEST__H
    double f(double x, double y, double z);
    #endif
    <BLANKLINE>

    Another example using Equality objects to give named outputs.  Here the
    filename (prefix) is taken from the first (name, expr) pair.

    >>> from sympy.abc import f, g
    >>> from sympy import Eq
    >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    ...      [("myfcn", x + y), ("fcn2", [Eq(f, 2*x), Eq(g, y)])],
    ...      "C99", header=False, empty=False)
    >>> print(c_name)
    myfcn.c
    >>> print(c_code)
    #include "myfcn.h"
    #include <math.h>
    double myfcn(double x, double y) {
       double myfcn_result;
       myfcn_result = x + y;
       return myfcn_result;
    }
    void fcn2(double x, double y, double *f, double *g) {
       (*f) = 2*x;
       (*g) = y;
    }
    <BLANKLINE>

    If the generated function(s) will be part of a larger project where various
    global variables have been defined, the 'global_vars' option can be used
    to remove the specified variables from the function signature

    >>> from sympy.utilities.codegen import codegen
    >>> from sympy.abc import x, y, z
    >>> [(f_name, f_code), header] = codegen(
    ...     ("f", x+y*z), "F95", header=False, empty=False,
    ...     argument_sequence=(x, y), global_vars=(z,))
    >>> print(f_code)
    REAL*8 function f(x, y)
    implicit none
    REAL*8, intent(in) :: x
    REAL*8, intent(in) :: y
    f = x + y*z
    end function
    <BLANKLINE>

    """

    # Initialize the code generator.
    if language is None:
        if code_gen is None:
            raise ValueError("Need either language or code_gen")
    else:
        if code_gen is not None:
            raise ValueError("You cannot specify both language and code_gen.")
        code_gen = get_code_generator(language, project, standard, printer)

    if isinstance(name_expr[0], str):
        # single tuple is given, turn it into a singleton list with a tuple.
        name_expr = [name_expr]

    if prefix is None:
        prefix = name_expr[0][0]

    # Construct Routines appropriate for this code_gen from (name, expr) pairs.
    routines = []
    for name, expr in name_expr:
        routines.append(code_gen.routine(name, expr, argument_sequence,
                                         global_vars))

    # Write the code.
    return code_gen.write(routines, prefix, to_files, header, empty)


def make_routine(name, expr, argument_sequence=None,
                 global_vars=None, language="F95"):
    """A factory that makes an appropriate Routine from an expression.

    Parameters
    ==========

    name : string
        The name of this routine in the generated code.

    expr : expression or list/tuple of expressions
        A SymPy expression that the Routine instance will represent.  If
        given a list or tuple of expressions, the routine will be
        considered to have multiple return values and/or output arguments.

    argument_sequence : list or tuple, optional
        List arguments for the routine in a preferred order.  If omitted,
        the results are language dependent, for example, alphabetical order
        or in the same order as the given expressions.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    language : string, optional
        Specify a target language.  The Routine itself should be
        language-agnostic but the precise way one is created, error
        checking, etc depend on the language.  [default: "F95"].

    Notes
    =====

    A decision about whether to use output arguments or return values is made
    depending on both the language and the particular mathematical expressions.
    For an expression of type Equality, the left hand side is typically made
    into an OutputArgument (or perhaps an InOutArgument if appropriate).
    Otherwise, typically, the calculated expression is made a return values of
    the routine.

    Examples
    ========

    >>> from sympy.utilities.codegen import make_routine
    >>> from sympy.abc import x, y, f, g
    >>> from sympy import Eq
    >>> r = make_routine('test', [Eq(f, 2*x), Eq(g, x + y)])
    >>> [arg.result_var for arg in r.results]
    []
    >>> [arg.name for arg in r.arguments]
    [x, y, f, g]
    >>> [arg.name for arg in r.result_variables]
    [f, g]
    >>> r.local_vars
    set()

    Another more complicated example with a mixture of specified and
    automatically-assigned names.  Also has Matrix output.

    >>> from sympy import Matrix
    >>> r = make_routine('fcn', [x*y, Eq(f, 1), Eq(g, x + g), Matrix([[x, 2]])])
    >>> [arg.result_var for arg in r.results]  # doctest: +SKIP
    [result_5397460570204848505]
    >>> [arg.expr for arg in r.results]
    [x*y]
    >>> [arg.name for arg in r.arguments]  # doctest: +SKIP
    [x, y, f, g, out_8598435338387848786]

    We can examine the various arguments more closely:

    >>> from sympy.utilities.codegen import (InputArgument, OutputArgument,
    ...                                      InOutArgument)
    >>> [a.name for a in r.arguments if isinstance(a, InputArgument)]
    [x, y]

    >>> [a.name for a in r.arguments if isinstance(a, OutputArgument)]  # doctest: +SKIP
    [f, out_8598435338387848786]
    >>> [a.expr for a in r.arguments if isinstance(a, OutputArgument)]
    [1, Matrix([[x, 2]])]

    >>> [a.name for a in r.arguments if isinstance(a, InOutArgument)]
    [g]
    >>> [a.expr for a in r.arguments if isinstance(a, InOutArgument)]
    [g + x]

    """

    # initialize a new code generator
    code_gen = get_code_generator(language)

    return code_gen.routine(name, expr, argument_sequence, global_vars)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\mul.py ===
from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar

from collections import defaultdict
from functools import reduce
from itertools import product
import operator

from .sympify import sympify
from .basic import Basic, _args_sortkey
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .intfunc import integer_nthroot, trailing
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up
from sympy.utilities.iterables import sift


# internal marker to indicate:
#   "there are still non-commutative objects -- don't forget to process them"
class NC_Marker:
    is_Order = False
    is_Mul = False
    is_Number = False
    is_Poly = False

    is_commutative = False


def _mulsort(args):
    # in-place sorting of args
    args.sort(key=_args_sortkey)


def _unevaluated_Mul(*args):
    """Return a well-formed unevaluated Mul: Numbers are collected and
    put in slot 0, any arguments that are Muls will be flattened, and args
    are sorted. Use this when args have changed but you still want to return
    an unevaluated Mul.

    Examples
    ========

    >>> from sympy.core.mul import _unevaluated_Mul as uMul
    >>> from sympy import S, sqrt, Mul
    >>> from sympy.abc import x
    >>> a = uMul(*[S(3.0), x, S(2)])
    >>> a.args[0]
    6.00000000000000
    >>> a.args[1]
    x

    Two unevaluated Muls with the same arguments will
    always compare as equal during testing:

    >>> m = uMul(sqrt(2), sqrt(3))
    >>> m == uMul(sqrt(3), sqrt(2))
    True
    >>> u = Mul(sqrt(3), sqrt(2), evaluate=False)
    >>> m == uMul(u)
    True
    >>> m == Mul(*m.args)
    False

    """
    cargs = []
    ncargs = []
    args = list(args)
    co = S.One
    for a in args:
        if a.is_Mul:
            a_c, a_nc = a.args_cnc()
            args.extend(a_c)  # grow args
            ncargs.extend(a_nc)
        elif a.is_Number:
            co *= a
        elif a.is_commutative:
            cargs.append(a)
        else:
            ncargs.append(a)
    _mulsort(cargs)
    if co is not S.One:
        cargs.insert(0, co)
    return Mul._from_args(cargs+ncargs)


class Mul(Expr, AssocOp):
    """
    Expression representing multiplication operation for algebraic field.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Mul()`` must be ``Expr``. Infix operator ``*``
    on most scalar objects in SymPy calls this class.

    Another use of ``Mul()`` is to represent the structure of abstract
    multiplication so that its arguments can be substituted to return
    different class. Refer to examples section for this.

    ``Mul()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Mul(x, Mul(y, z))`` -> ``Mul(x, y, z)``

    2. Identity removing
        ``Mul(x, 1, y)`` -> ``Mul(x, y)``

    3. Exponent collecting by ``.as_base_exp()``
        ``Mul(x, x**2)`` -> ``Pow(x, 3)``

    4. Term sorting
        ``Mul(y, x, 2)`` -> ``Mul(2, x, y)``

    Since multiplication can be vector space operation, arguments may
    have the different :obj:`sympy.core.kind.Kind()`. Kind of the
    resulting object is automatically inferred.

    Examples
    ========

    >>> from sympy import Mul
    >>> from sympy.abc import x, y
    >>> Mul(x, 1)
    x
    >>> Mul(x, x)
    x**2

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Mul(1, 2, evaluate=False)
    1*2
    >>> Mul(x, x, evaluate=False)
    x*x

    ``Mul()`` also represents the general structure of multiplication
    operation.

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 2,2)
    >>> expr = Mul(x,y).subs({y:A})
    >>> expr
    x*A
    >>> type(expr)
    <class 'sympy.matrices.expressions.matmul.MatMul'>

    See Also
    ========

    MatMul

    """
    __slots__ = ()

    is_Mul = True

    _args_type = Expr
    _kind_dispatcher = KindDispatcher("Mul_kind_dispatcher", commutative=True)

    identity: ClassVar[Expr]

    @property
    def kind(self):
        arg_kinds = (a.kind for a in self.args)
        return self._kind_dispatcher(*arg_kinds)

    if TYPE_CHECKING:

        def __new__(cls, *args: Expr | complex, evaluate: bool=True) -> Expr: # type: ignore
            ...

        @property
        def args(self) -> tuple[Expr, ...]:
            ...

    def could_extract_minus_sign(self):
        if self == (-self):
            return False  # e.g. zoo*x == -zoo*x
        c = self.args[0]
        return c.is_Number and c.is_extended_negative

    def __neg__(self):
        c, args = self.as_coeff_mul()
        if args[0] is not S.ComplexInfinity:
            c = -c
        if c is not S.One:
            if args[0].is_Number:
                args = list(args)
                if c is S.NegativeOne:
                    args[0] = -args[0]
                else:
                    args[0] *= c
            else:
                args = (c,) + args
        return self._from_args(args, self.is_commutative)

    @classmethod
    def flatten(cls, seq):
        """Return commutative, noncommutative and order arguments by
        combining related terms.

        Notes
        =====
            * In an expression like ``a*b*c``, Python process this through SymPy
              as ``Mul(Mul(a, b), c)``. This can have undesirable consequences.

              -  Sometimes terms are not combined as one would like:
                 {c.f. https://github.com/sympy/sympy/issues/4596}

                >>> from sympy import Mul, sqrt
                >>> from sympy.abc import x, y, z
                >>> 2*(x + 1) # this is the 2-arg Mul behavior
                2*x + 2
                >>> y*(x + 1)*2
                2*y*(x + 1)
                >>> 2*(x + 1)*y # 2-arg result will be obtained first
                y*(2*x + 2)
                >>> Mul(2, x + 1, y) # all 3 args simultaneously processed
                2*y*(x + 1)
                >>> 2*((x + 1)*y) # parentheses can control this behavior
                2*y*(x + 1)

                Powers with compound bases may not find a single base to
                combine with unless all arguments are processed at once.
                Post-processing may be necessary in such cases.
                {c.f. https://github.com/sympy/sympy/issues/5728}

                >>> a = sqrt(x*sqrt(y))
                >>> a**3
                (x*sqrt(y))**(3/2)
                >>> Mul(a,a,a)
                (x*sqrt(y))**(3/2)
                >>> a*a*a
                x*sqrt(y)*sqrt(x*sqrt(y))
                >>> _.subs(a.base, z).subs(z, a.base)
                (x*sqrt(y))**(3/2)

              -  If more than two terms are being multiplied then all the
                 previous terms will be re-processed for each new argument.
                 So if each of ``a``, ``b`` and ``c`` were :class:`Mul`
                 expression, then ``a*b*c`` (or building up the product
                 with ``*=``) will process all the arguments of ``a`` and
                 ``b`` twice: once when ``a*b`` is computed and again when
                 ``c`` is multiplied.

                 Using ``Mul(a, b, c)`` will process all arguments once.

            * The results of Mul are cached according to arguments, so flatten
              will only be called once for ``Mul(a, b, c)``. If you can
              structure a calculation so the arguments are most likely to be
              repeats then this can save time in computing the answer. For
              example, say you had a Mul, M, that you wished to divide by ``d[i]``
              and multiply by ``n[i]`` and you suspect there are many repeats
              in ``n``. It would be better to compute ``M*n[i]/d[i]`` rather
              than ``M/d[i]*n[i]`` since every time n[i] is a repeat, the
              product, ``M*n[i]`` will be returned without flattening -- the
              cached value will be returned. If you divide by the ``d[i]``
              first (and those are more unique than the ``n[i]``) then that will
              create a new Mul, ``M/d[i]`` the args of which will be traversed
              again when it is multiplied by ``n[i]``.

              {c.f. https://github.com/sympy/sympy/issues/5706}

              This consideration is moot if the cache is turned off.

            NB
            --
              The validity of the above notes depends on the implementation
              details of Mul and flatten which may change at any time. Therefore,
              you should only consider them when your code is highly performance
              sensitive.

              Removal of 1 from the sequence is already handled by AssocOp.__new__.
        """

        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.matrices.expressions import MatrixExpr
        rv = None
        if len(seq) == 2:
            a, b = seq
            if b.is_Rational:
                a, b = b, a
                seq = [a, b]
            assert a is not S.One
            if a.is_Rational and not a.is_zero:
                r, b = b.as_coeff_Mul()
                if b.is_Add:
                    if r is not S.One:  # 2-arg hack
                        # leave the Mul as a Mul?
                        ar = a*r
                        if ar is S.One:
                            arb = b
                        else:
                            arb = cls(a*r, b, evaluate=False)
                        rv = [arb], [], None
                    elif global_parameters.distribute and b.is_commutative:
                        newb = Add(*[_keep_coeff(a, bi) for bi in b.args])
                        rv = [newb], [], None
            if rv:
                return rv

        # apply associativity, separate commutative part of seq
        c_part = []         # out: commutative factors
        nc_part = []        # out: non-commutative factors

        nc_seq = []

        coeff = S.One       # standalone term
                            # e.g. 3 * ...

        c_powers = []       # (base,exp)      n
                            # e.g. (x,n) for x

        num_exp = []        # (num-base, exp)           y
                            # e.g.  (3, y)  for  ... * 3  * ...

        neg1e = S.Zero      # exponent on -1 extracted from Number-based Pow and I

        pnum_rat = {}       # (num-base, Rat-exp)          1/2
                            # e.g.  (3, 1/2)  for  ... * 3     * ...

        order_symbols = None

        # --- PART 1 ---
        #
        # "collect powers and coeff":
        #
        # o coeff
        # o c_powers
        # o num_exp
        # o neg1e
        # o pnum_rat
        #
        # NOTE: this is optimized for all-objects-are-commutative case
        for o in seq:
            # O(x)
            if o.is_Order:
                o, order_symbols = o.as_expr_variables(order_symbols)

            # Mul([...])
            if o.is_Mul:
                if o.is_commutative:
                    seq.extend(o.args)    # XXX zerocopy?

                else:
                    # NCMul can have commutative parts as well
                    for q in o.args:
                        if q.is_commutative:
                            seq.append(q)
                        else:
                            nc_seq.append(q)

                    # append non-commutative marker, so we don't forget to
                    # process scheduled non-commutative objects
                    seq.append(NC_Marker)

                continue

            # 3
            elif o.is_Number:
                if o is S.NaN or coeff is S.ComplexInfinity and o.is_zero:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
                elif coeff.is_Number or isinstance(coeff, AccumBounds):  # it could be zoo
                    coeff *= o
                    if coeff is S.NaN:
                        # we know for sure the result will be nan
                        return [S.NaN], [], None
                continue

            elif isinstance(o, AccumBounds):
                coeff = o.__mul__(coeff)
                continue

            elif o is S.ComplexInfinity:
                if not coeff:
                    # 0 * zoo = NaN
                    return [S.NaN], [], None
                coeff = S.ComplexInfinity
                continue

            elif not coeff and isinstance(o, Add) and any(
                    _ in (S.NegativeInfinity, S.ComplexInfinity, S.Infinity)
                    for __ in o.args for _ in Mul.make_args(__)):
                # e.g 0 * (x + oo) = NaN but not
                # 0 * (1 + Integral(x, (x, 0, oo))) which is
                # treated like 0 * x -> 0
                return [S.NaN], [], None

            elif o is S.ImaginaryUnit:
                neg1e += S.Half
                continue

            elif o.is_commutative:
                #      e
                # o = b
                b, e = o.as_base_exp()

                #  y
                # 3
                if o.is_Pow:
                    if b.is_Number:

                        # get all the factors with numeric base so they can be
                        # combined below, but don't combine negatives unless
                        # the exponent is an integer
                        if e.is_Rational:
                            if e.is_Integer:
                                coeff *= Pow(b, e)  # it is an unevaluated power
                                continue
                            elif e.is_negative:    # also a sign of an unevaluated power
                                seq.append(Pow(b, e))
                                continue
                            elif b.is_negative:
                                neg1e += e
                                b = -b
                            if b is not S.One:
                                pnum_rat.setdefault(b, []).append(e)
                            continue
                        elif b.is_positive or e.is_integer:
                            num_exp.append((b, e))
                            continue

                c_powers.append((b, e))

            # NON-COMMUTATIVE
            # TODO: Make non-commutative exponents not combine automatically
            else:
                if o is not NC_Marker:
                    nc_seq.append(o)

                # process nc_seq (if any)
                while nc_seq:
                    o = nc_seq.pop(0)
                    if not nc_part:
                        nc_part.append(o)
                        continue

                    #                             b    c       b+c
                    # try to combine last terms: a  * a   ->  a
                    o1 = nc_part.pop()
                    b1, e1 = o1.as_base_exp()
                    b2, e2 = o.as_base_exp()
                    new_exp = e1 + e2
                    # Only allow powers to combine if the new exponent is
                    # not an Add. This allow things like a**2*b**3 == a**5
                    # if a.is_commutative == False, but prohibits
                    # a**x*a**y and x**a*x**b from combining (x,y commute).
                    if b1 == b2 and (not new_exp.is_Add):
                        o12 = b1 ** new_exp

                        # now o12 could be a commutative object
                        if o12.is_commutative:
                            seq.append(o12)
                            continue
                        else:
                            nc_seq.insert(0, o12)

                    else:
                        nc_part.extend([o1, o])

        # We do want a combined exponent if it would not be an Add, such as
        #  y    2y     3y
        # x  * x   -> x
        # We determine if two exponents have the same term by using
        # as_coeff_Mul.
        #
        # Unfortunately, this isn't smart enough to consider combining into
        # exponents that might already be adds, so things like:
        #  z - y    y
        # x      * x  will be left alone.  This is because checking every possible
        # combination can slow things down.

        # gather exponents of common bases...
        def _gather(c_powers):
            common_b = {}  # b:e
            for b, e in c_powers:
                co = e.as_coeff_Mul()
                common_b.setdefault(b, {}).setdefault(
                    co[1], []).append(co[0])
            for b, d in common_b.items():
                for di, li in d.items():
                    d[di] = Add(*li)
            new_c_powers = []
            for b, e in common_b.items():
                new_c_powers.extend([(b, c*t) for t, c in e.items()])
            return new_c_powers

        # in c_powers
        c_powers = _gather(c_powers)

        # and in num_exp
        num_exp = _gather(num_exp)

        # --- PART 2 ---
        #
        # o process collected powers  (x**0 -> 1; x**1 -> x; otherwise Pow)
        # o combine collected powers  (2**x * 3**x -> 6**x)
        #   with numeric base

        # ................................
        # now we have:
        # - coeff:
        # - c_powers:    (b, e)
        # - num_exp:     (2, e)
        # - pnum_rat:    {(1/3, [1/3, 2/3, 1/4])}

        #  0             1
        # x  -> 1       x  -> x

        # this should only need to run twice; if it fails because
        # it needs to be run more times, perhaps this should be
        # changed to a "while True" loop -- the only reason it
        # isn't such now is to allow a less-than-perfect result to
        # be obtained rather than raising an error or entering an
        # infinite loop
        for i in range(2):
            new_c_powers = []
            changed = False
            for b, e in c_powers:
                if e.is_zero:
                    # canceling out infinities yields NaN
                    if (b.is_Add or b.is_Mul) and any(infty in b.args
                        for infty in (S.ComplexInfinity, S.Infinity,
                                      S.NegativeInfinity)):
                        return [S.NaN], [], None
                    continue
                if e is S.One:
                    if b.is_Number:
                        coeff *= b
                        continue
                    p = b
                if e is not S.One:
                    p = Pow(b, e)
                    # check to make sure that the base doesn't change
                    # after exponentiation; to allow for unevaluated
                    # Pow, we only do so if b is not already a Pow
                    if p.is_Pow and not b.is_Pow:
                        bi = b
                        b, e = p.as_base_exp()
                        if b != bi:
                            changed = True
                c_part.append(p)
                new_c_powers.append((b, e))
            # there might have been a change, but unless the base
            # matches some other base, there is nothing to do
            if changed and len({
                    b for b, e in new_c_powers}) != len(new_c_powers):
                # start over again
                c_part = []
                c_powers = _gather(new_c_powers)
            else:
                break

        #  x    x     x
        # 2  * 3  -> 6
        inv_exp_dict = {}   # exp:Mul(num-bases)     x    x
                            # e.g.  x:6  for  ... * 2  * 3  * ...
        for b, e in num_exp:
            inv_exp_dict.setdefault(e, []).append(b)
        for e, b in inv_exp_dict.items():
            inv_exp_dict[e] = cls(*b)
        c_part.extend([Pow(b, e) for e, b in inv_exp_dict.items() if e])

        # b, e -> e' = sum(e), b
        # {(1/5, [1/3]), (1/2, [1/12, 1/4]} -> {(1/3, [1/5, 1/2])}
        comb_e = {}
        for b, e in pnum_rat.items():
            comb_e.setdefault(Add(*e), []).append(b)
        del pnum_rat
        # process them, reducing exponents to values less than 1
        # and updating coeff if necessary else adding them to
        # num_rat for further processing
        num_rat = []
        for e, b in comb_e.items():
            b = cls(*b)
            if e.q == 1:
                coeff *= Pow(b, e)
                continue
            if e.p > e.q:
                e_i, ep = divmod(e.p, e.q)
                coeff *= Pow(b, e_i)
                e = Rational(ep, e.q)
            num_rat.append((b, e))
        del comb_e

        # extract gcd of bases in num_rat
        # 2**(1/3)*6**(1/4) -> 2**(1/3+1/4)*3**(1/4)
        pnew = defaultdict(list)
        i = 0  # steps through num_rat which may grow
        while i < len(num_rat):
            bi, ei = num_rat[i]
            if bi == 1:
                i += 1
                continue
            grow = []
            for j in range(i + 1, len(num_rat)):
                bj, ej = num_rat[j]
                g = bi.gcd(bj)
                if g is not S.One:
                    # 4**r1*6**r2 -> 2**(r1+r2)  *  2**r1 *  3**r2
                    # this might have a gcd with something else
                    e = ei + ej
                    if e.q == 1:
                        coeff *= Pow(g, e)
                    else:
                        if e.p > e.q:
                            e_i, ep = divmod(e.p, e.q)  # change e in place
                            coeff *= Pow(g, e_i)
                            e = Rational(ep, e.q)
                        grow.append((g, e))
                    # update the jth item
                    num_rat[j] = (bj/g, ej)
                    # update bi that we are checking with
                    bi = bi/g
                    if bi is S.One:
                        break
            if bi is not S.One:
                obj = Pow(bi, ei)
                if obj.is_Number:
                    coeff *= obj
                else:
                    # changes like sqrt(12) -> 2*sqrt(3)
                    for obj in Mul.make_args(obj):
                        if obj.is_Number:
                            coeff *= obj
                        else:
                            assert obj.is_Pow
                            bi, ei = obj.args
                            pnew[ei].append(bi)

            num_rat.extend(grow)
            i += 1

        # combine bases of the new powers
        for e, b in pnew.items():
            pnew[e] = cls(*b)

        # handle -1 and I
        if neg1e:
            # treat I as (-1)**(1/2) and compute -1's total exponent
            p, q =  neg1e.as_numer_denom()
            # if the integer part is odd, extract -1
            n, p = divmod(p, q)
            if n % 2:
                coeff = -coeff
            # if it's a multiple of 1/2 extract I
            if q == 2:
                c_part.append(S.ImaginaryUnit)
            elif p:
                # see if there is any positive base this power of
                # -1 can join
                neg1e = Rational(p, q)
                for e, b in pnew.items():
                    if e == neg1e and b.is_positive:
                        pnew[e] = -b
                        break
                else:
                    # keep it separate; we've already evaluated it as
                    # much as possible so evaluate=False
                    c_part.append(Pow(S.NegativeOne, neg1e, evaluate=False))

        # add all the pnew powers
        c_part.extend([Pow(b, e) for e, b in pnew.items()])

        # oo, -oo
        if coeff in (S.Infinity, S.NegativeInfinity):
            def _handle_for_oo(c_part, coeff_sign):
                new_c_part = []
                for t in c_part:
                    if t.is_extended_positive:
                        continue
                    if t.is_extended_negative:
                        coeff_sign *= -1
                        continue
                    new_c_part.append(t)
                return new_c_part, coeff_sign
            c_part, coeff_sign = _handle_for_oo(c_part, 1)
            nc_part, coeff_sign = _handle_for_oo(nc_part, coeff_sign)
            coeff *= coeff_sign

        # zoo
        if coeff is S.ComplexInfinity:
            # zoo might be
            #   infinite_real + bounded_im
            #   bounded_real + infinite_im
            #   infinite_real + infinite_im
            # and non-zero real or imaginary will not change that status.
            c_part = [c for c in c_part if not (fuzzy_not(c.is_zero) and
                                                c.is_extended_real is not None)]
            nc_part = [c for c in nc_part if not (fuzzy_not(c.is_zero) and
                                                  c.is_extended_real is not None)]

        # 0
        elif coeff.is_zero:
            # we know for sure the result will be 0 except the multiplicand
            # is infinity or a matrix
            if any(isinstance(c, MatrixExpr) for c in nc_part):
                return [coeff], nc_part, order_symbols
            if any(c.is_finite == False for c in c_part):
                return [S.NaN], [], order_symbols
            return [coeff], [], order_symbols

        # check for straggling Numbers that were produced
        _new = []
        for i in c_part:
            if i.is_Number:
                coeff *= i
            else:
                _new.append(i)
        c_part = _new

        # order commutative part canonically
        _mulsort(c_part)

        # current code expects coeff to be always in slot-0
        if coeff is not S.One:
            c_part.insert(0, coeff)

        # we are done
        if (global_parameters.distribute and not nc_part and len(c_part) == 2 and
                c_part[0].is_Number and c_part[0].is_finite and c_part[1].is_Add):
            # 2*(1+a) -> 2 + 2 * a
            coeff = c_part[0]
            c_part = [Add(*[coeff*f for f in c_part[1].args])]

        return c_part, nc_part, order_symbols

    def _eval_power(self, expt):

        # don't break up NC terms: (A*B)**3 != A**3*B**3, it is A*B*A*B*A*B
        cargs, nc = self.args_cnc(split_1=False)

        if expt.is_Integer:
            return Mul(*[Pow(b, expt, evaluate=False) for b in cargs]) * \
                Pow(Mul._from_args(nc), expt, evaluate=False)
        if expt.is_Rational and expt.q == 2:
            if self.is_imaginary:
                a = self.as_real_imag()[1]
                if a.is_Rational:
                    n, d = abs(a/2).as_numer_denom()
                    n, t = integer_nthroot(n, 2)
                    if t:
                        d, t = integer_nthroot(d, 2)
                        if t:
                            from sympy.functions.elementary.complexes import sign
                            r = sympify(n)/d
                            return _unevaluated_Mul(r**expt.p, (1 + sign(a)*S.ImaginaryUnit)**expt.p)

        p = Pow(self, expt, evaluate=False)

        if expt.is_Rational or expt.is_Float:
            return p._eval_expand_power_base()

        return p

    @classmethod
    def class_key(cls):
        return 3, 0, cls.__name__

    def _eval_evalf(self, prec):
        c, m = self.as_coeff_Mul()
        if c is S.NegativeOne:
            if m.is_Mul:
                rv = -AssocOp._eval_evalf(m, prec)
            else:
                mnew = m._eval_evalf(prec)
                if mnew is not None:
                    m = mnew
                rv = -m
        else:
            rv = AssocOp._eval_evalf(self, prec)
        if rv.is_number:
            return rv.expand()
        return rv

    @property
    def _mpc_(self):
        """
        Convert self to an mpmath mpc if possible
        """
        from .numbers import Float
        im_part, imag_unit = self.as_coeff_Mul()
        if imag_unit is not S.ImaginaryUnit:
            # ValueError may seem more reasonable but since it's a @property,
            # we need to use AttributeError to keep from confusing things like
            # hasattr.
            raise AttributeError("Cannot convert Mul to mpc. Must be of the form Number*I")

        return (Float(0)._mpf_, Float(im_part)._mpf_)

    @cacheit
    def as_two_terms(self):
        """Return head and tail of self.

        This is the most efficient way to get the head and tail of an
        expression.

        - if you want only the head, use self.args[0];
        - if you want to process the arguments of the tail then use
          self.as_coef_mul() which gives the head and a tuple containing
          the arguments of the tail when treated as a Mul.
        - if you want the coefficient when self is treated as an Add
          then use self.as_coeff_add()[0]

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> (3*x*y).as_two_terms()
        (3, x*y)
        """
        args = self.args

        if len(args) == 1:
            return S.One, self
        elif len(args) == 2:
            return args

        else:
            return args[0], self._new_rawargs(*args[1:])

    @cacheit
    def as_coeff_mul(self, *deps, rational=True, **kwargs):
        if deps:
            l1, l2 = sift(self.args, lambda x: x.has(*deps), binary=True)
            return self._new_rawargs(*l2), tuple(l1)
        args = self.args
        if args[0].is_Number:
            if not rational or args[0].is_Rational:
                return args[0], args[1:]
            elif args[0].is_extended_negative:
                return S.NegativeOne, (-args[0],) + args[1:]
        return S.One, args

    def as_coeff_Mul(self, rational=False):
        """
        Efficiently extract the coefficient of a product.
        """
        coeff, args = self.args[0], self.args[1:]

        if coeff.is_Number:
            if not rational or coeff.is_Rational:
                if len(args) == 1:
                    return coeff, args[0]
                else:
                    return coeff, self._new_rawargs(*args)
            elif coeff.is_extended_negative:
                return S.NegativeOne, self._new_rawargs(*((-coeff,) + args))
        return S.One, self

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.complexes import Abs, im, re
        other = []
        coeffr = []
        coeffi = []
        addterms = S.One
        for a in self.args:
            r, i = a.as_real_imag()
            if i.is_zero:
                coeffr.append(r)
            elif r.is_zero:
                coeffi.append(i*S.ImaginaryUnit)
            elif a.is_commutative:
                aconj = a.conjugate() if other else None
                # search for complex conjugate pairs:
                for i, x in enumerate(other):
                    if x == aconj:
                        coeffr.append(Abs(x)**2)
                        del other[i]
                        break
                else:
                    if a.is_Add:
                        addterms *= a
                    else:
                        other.append(a)
            else:
                other.append(a)
        m = self.func(*other)
        if hints.get('ignore') == m:
            return
        if len(coeffi) % 2:
            imco = im(coeffi.pop(0))
            # all other pairs make a real factor; they will be
            # put into reco below
        else:
            imco = S.Zero
        reco = self.func(*(coeffr + coeffi))
        r, i = (reco*re(m), reco*im(m))
        if addterms == 1:
            if m == 1:
                if imco.is_zero:
                    return (reco, S.Zero)
                else:
                    return (S.Zero, reco*imco)
            if imco is S.Zero:
                return (r, i)
            return (-imco*i, imco*r)
        from .function import expand_mul
        addre, addim = expand_mul(addterms, deep=False).as_real_imag()
        if imco is S.Zero:
            return (r*addre - i*addim, i*addre + r*addim)
        else:
            r, i = -imco*i, imco*r
            return (r*addre - i*addim, r*addim + i*addre)

    @staticmethod
    def _expandsums(sums):
        """
        Helper function for _eval_expand_mul.

        sums must be a list of instances of Basic.
        """

        L = len(sums)
        if L == 1:
            return sums[0].args
        terms = []
        left = Mul._expandsums(sums[:L//2])
        right = Mul._expandsums(sums[L//2:])

        terms = [Mul(a, b) for a in left for b in right]
        added = Add(*terms)
        return Add.make_args(added)  # it may have collapsed down to one term

    def _eval_expand_mul(self, **hints):
        from sympy.simplify.radsimp import fraction

        # Handle things like 1/(x*(x + 1)), which are automatically converted
        # to 1/x*1/(x + 1)
        expr = self
        # default matches fraction's default
        n, d = fraction(expr, hints.get('exact', False))
        if d.is_Mul:
            n, d = [i._eval_expand_mul(**hints) if i.is_Mul else i
                for i in (n, d)]
        expr = n/d
        if not expr.is_Mul:
            return expr

        plain, sums, rewrite = [], [], False
        for factor in expr.args:
            if factor.is_Add:
                sums.append(factor)
                rewrite = True
            else:
                if factor.is_commutative:
                    plain.append(factor)
                else:
                    sums.append(Basic(factor))  # Wrapper

        if not rewrite:
            return expr
        else:
            plain = self.func(*plain)
            if sums:
                deep = hints.get("deep", False)
                terms = self.func._expandsums(sums)
                args = []
                for term in terms:
                    t = self.func(plain, term)
                    if t.is_Mul and any(a.is_Add for a in t.args) and deep:
                        t = t._eval_expand_mul()
                    args.append(t)
                return Add(*args)
            else:
                return plain

    @cacheit
    def _eval_derivative(self, s):
        args = list(self.args)
        terms = []
        for i in range(len(args)):
            d = args[i].diff(s)
            if d:
                # Note: reduce is used in step of Mul as Mul is unable to
                # handle subtypes and operation priority:
                terms.append(reduce(lambda x, y: x*y, (args[:i] + [d] + args[i + 1:]), S.One))
        return Add.fromiter(terms)

    @cacheit
    def _eval_derivative_n_times(self, s, n):
        from .function import AppliedUndef
        from .symbol import Symbol, symbols, Dummy
        if not isinstance(s, (AppliedUndef, Symbol)):
            # other types of s may not be well behaved, e.g.
            # (cos(x)*sin(y)).diff([[x, y, z]])
            return super()._eval_derivative_n_times(s, n)
        from .numbers import Integer
        args = self.args
        m = len(args)
        if isinstance(n, (int, Integer)):
            # https://en.wikipedia.org/wiki/General_Leibniz_rule#More_than_two_factors
            terms = []
            from sympy.ntheory.multinomial import multinomial_coefficients_iterator
            for kvals, c in multinomial_coefficients_iterator(m, n):
                p = Mul(*[arg.diff((s, k)) for k, arg in zip(kvals, args)])
                terms.append(c * p)
            return Add(*terms)
        from sympy.concrete.summations import Sum
        from sympy.functions.combinatorial.factorials import factorial
        from sympy.functions.elementary.miscellaneous import Max
        kvals = symbols("k1:%i" % m, cls=Dummy)
        klast = n - sum(kvals)
        nfact = factorial(n)
        e, l = (# better to use the multinomial?
            nfact/prod(map(factorial, kvals))/factorial(klast)*\
            Mul(*[args[t].diff((s, kvals[t])) for t in range(m-1)])*\
            args[-1].diff((s, Max(0, klast))),
            [(k, 0, n) for k in kvals])
        return Sum(e, *l)

    def _eval_difference_delta(self, n, step):
        from sympy.series.limitseq import difference_delta as dd
        arg0 = self.args[0]
        rest = Mul(*self.args[1:])
        return (arg0.subs(n, n + step) * dd(rest, n, step) + dd(arg0, n, step) *
                rest)

    def _matches_simple(self, expr, repl_dict):
        # handle (w*3).matches('x*5') -> {w: x*5/3}
        coeff, terms = self.as_coeff_Mul()
        terms = Mul.make_args(terms)
        if len(terms) == 1:
            newexpr = self.__class__._combine_inverse(expr, coeff)
            return terms[0].matches(newexpr, repl_dict)
        return

    def matches(self, expr, repl_dict=None, old=False):
        expr = sympify(expr)
        if self.is_commutative and expr.is_commutative:
            return self._matches_commutative(expr, repl_dict, old)
        elif self.is_commutative is not expr.is_commutative:
            return None

        # Proceed only if both both expressions are non-commutative
        c1, nc1 = self.args_cnc()
        c2, nc2 = expr.args_cnc()
        c1, c2 = [c or [1] for c in [c1, c2]]

        # TODO: Should these be self.func?
        comm_mul_self = Mul(*c1)
        comm_mul_expr = Mul(*c2)

        repl_dict = comm_mul_self.matches(comm_mul_expr, repl_dict, old)

        # If the commutative arguments didn't match and aren't equal, then
        # then the expression as a whole doesn't match
        if not repl_dict and c1 != c2:
            return None

        # Now match the non-commutative arguments, expanding powers to
        # multiplications
        nc1 = Mul._matches_expand_pows(nc1)
        nc2 = Mul._matches_expand_pows(nc2)

        repl_dict = Mul._matches_noncomm(nc1, nc2, repl_dict)

        return repl_dict or None

    @staticmethod
    def _matches_expand_pows(arg_list):
        new_args = []
        for arg in arg_list:
            if arg.is_Pow and arg.exp > 0:
                new_args.extend([arg.base] * arg.exp)
            else:
                new_args.append(arg)
        return new_args

    @staticmethod
    def _matches_noncomm(nodes, targets, repl_dict=None):
        """Non-commutative multiplication matcher.

        `nodes` is a list of symbols within the matcher multiplication
        expression, while `targets` is a list of arguments in the
        multiplication expression being matched against.
        """
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        # List of possible future states to be considered
        agenda = []
        # The current matching state, storing index in nodes and targets
        state = (0, 0)
        node_ind, target_ind = state
        # Mapping between wildcard indices and the index ranges they match
        wildcard_dict = {}

        while target_ind < len(targets) and node_ind < len(nodes):
            node = nodes[node_ind]

            if node.is_Wild:
                Mul._matches_add_wildcard(wildcard_dict, state)

            states_matches = Mul._matches_new_states(wildcard_dict, state,
                                                     nodes, targets)
            if states_matches:
                new_states, new_matches = states_matches
                agenda.extend(new_states)
                if new_matches:
                    for match in new_matches:
                        repl_dict[match] = new_matches[match]
            if not agenda:
                return None
            else:
                state = agenda.pop()
                node_ind, target_ind = state

        return repl_dict

    @staticmethod
    def _matches_add_wildcard(dictionary, state):
        node_ind, target_ind = state
        if node_ind in dictionary:
            begin, end = dictionary[node_ind]
            dictionary[node_ind] = (begin, target_ind)
        else:
            dictionary[node_ind] = (target_ind, target_ind)

    @staticmethod
    def _matches_new_states(dictionary, state, nodes, targets):
        node_ind, target_ind = state
        node = nodes[node_ind]
        target = targets[target_ind]

        # Don't advance at all if we've exhausted the targets but not the nodes
        if target_ind >= len(targets) - 1 and node_ind < len(nodes) - 1:
            return None

        if node.is_Wild:
            match_attempt = Mul._matches_match_wilds(dictionary, node_ind,
                                                     nodes, targets)
            if match_attempt:
                # If the same node has been matched before, don't return
                # anything if the current match is diverging from the previous
                # match
                other_node_inds = Mul._matches_get_other_nodes(dictionary,
                                                               nodes, node_ind)
                for ind in other_node_inds:
                    other_begin, other_end = dictionary[ind]
                    curr_begin, curr_end = dictionary[node_ind]

                    other_targets = targets[other_begin:other_end + 1]
                    current_targets = targets[curr_begin:curr_end + 1]

                    for curr, other in zip(current_targets, other_targets):
                        if curr != other:
                            return None

                # A wildcard node can match more than one target, so only the
                # target index is advanced
                new_state = [(node_ind, target_ind + 1)]
                # Only move on to the next node if there is one
                if node_ind < len(nodes) - 1:
                    new_state.append((node_ind + 1, target_ind + 1))
                return new_state, match_attempt
        else:
            # If we're not at a wildcard, then make sure we haven't exhausted
            # nodes but not targets, since in this case one node can only match
            # one target
            if node_ind >= len(nodes) - 1 and target_ind < len(targets) - 1:
                return None

            match_attempt = node.matches(target)

            if match_attempt:
                return [(node_ind + 1, target_ind + 1)], match_attempt
            elif node == target:
                return [(node_ind + 1, target_ind + 1)], None
            else:
                return None

    @staticmethod
    def _matches_match_wilds(dictionary, wildcard_ind, nodes, targets):
        """Determine matches of a wildcard with sub-expression in `target`."""
        wildcard = nodes[wildcard_ind]
        begin, end = dictionary[wildcard_ind]
        terms = targets[begin:end + 1]
        # TODO: Should this be self.func?
        mult = Mul(*terms) if len(terms) > 1 else terms[0]
        return wildcard.matches(mult)

    @staticmethod
    def _matches_get_other_nodes(dictionary, nodes, node_ind):
        """Find other wildcards that may have already been matched."""
        ind_node = nodes[node_ind]
        return [ind for ind in dictionary if nodes[ind] == ind_node]

    @staticmethod
    def _combine_inverse(lhs, rhs):
        """
        Returns lhs/rhs, but treats arguments like symbols, so things
        like oo/oo return 1 (instead of a nan) and ``I`` behaves like
        a symbol instead of sqrt(-1).
        """
        from sympy.simplify.simplify import signsimp
        from .symbol import Dummy
        if lhs == rhs:
            return S.One

        def check(l, r):
            if l.is_Float and r.is_comparable:
                # if both objects are added to 0 they will share the same "normalization"
                # and are more likely to compare the same. Since Add(foo, 0) will not allow
                # the 0 to pass, we use __add__ directly.
                return l.__add__(0) == r.evalf().__add__(0)
            return False
        if check(lhs, rhs) or check(rhs, lhs):
            return S.One
        if any(i.is_Pow or i.is_Mul for i in (lhs, rhs)):
            # gruntz and limit wants a literal I to not combine
            # with a power of -1
            d = Dummy('I')
            _i = {S.ImaginaryUnit: d}
            i_ = {d: S.ImaginaryUnit}
            a = lhs.xreplace(_i).as_powers_dict()
            b = rhs.xreplace(_i).as_powers_dict()
            blen = len(b)
            for bi in tuple(b.keys()):
                if bi in a:
                    a[bi] -= b.pop(bi)
                    if not a[bi]:
                        a.pop(bi)
            if len(b) != blen:
                lhs = Mul(*[k**v for k, v in a.items()]).xreplace(i_)
                rhs = Mul(*[k**v for k, v in b.items()]).xreplace(i_)
        rv = lhs/rhs
        srv = signsimp(rv)
        return srv if srv.is_Number else rv

    def as_powers_dict(self):
        d = defaultdict(int)
        for term in self.args:
            for b, e in term.as_powers_dict().items():
                d[b] += e
        return d

    def as_numer_denom(self):
        # don't use _from_args to rebuild the numerators and denominators
        # as the order is not guaranteed to be the same once they have
        # been separated from each other
        numers, denoms = list(zip(*[f.as_numer_denom() for f in self.args]))
        return self.func(*numers), self.func(*denoms)

    def as_base_exp(self):
        e1 = None
        bases = []
        nc = 0
        for m in self.args:
            b, e = m.as_base_exp()
            if not b.is_commutative:
                nc += 1
            if e1 is None:
                e1 = e
            elif e != e1 or nc > 1 or not e.is_Integer:
                return self, S.One
            bases.append(b)
        return self.func(*bases), e1

    def _eval_is_polynomial(self, syms):
        return all(term._eval_is_polynomial(syms) for term in self.args)

    def _eval_is_rational_function(self, syms):
        return all(term._eval_is_rational_function(syms) for term in self.args)

    def _eval_is_meromorphic(self, x, a):
        return _fuzzy_group((arg.is_meromorphic(x, a) for arg in self.args),
                            quick_exit=True)

    def _eval_is_algebraic_expr(self, syms):
        return all(term._eval_is_algebraic_expr(syms) for term in self.args)

    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)

    def _eval_is_complex(self):
        comp = _fuzzy_group(a.is_complex for a in self.args)
        if comp is False:
            if any(a.is_infinite for a in self.args):
                if any(a.is_zero is not False for a in self.args):
                    return None
                return False
        return comp

    def _eval_is_zero_infinite_helper(self):
        #
        # Helper used by _eval_is_zero and _eval_is_infinite.
        #
        # Three-valued logic is tricky so let us reason this carefully. It
        # would be nice to say that we just check is_zero/is_infinite in all
        # args but we need to be careful about the case that one arg is zero
        # and another is infinite like Mul(0, oo) or more importantly a case
        # where it is not known if the arguments are zero or infinite like
        # Mul(y, 1/x). If either y or x could be zero then there is a
        # *possibility* that we have Mul(0, oo) which should give None for both
        # is_zero and is_infinite.
        #
        # We keep track of whether we have seen a zero or infinity but we also
        # need to keep track of whether we have *possibly* seen one which
        # would be indicated by None.
        #
        # For each argument there is the possibility that is_zero might give
        # True, False or None and likewise that is_infinite might give True,
        # False or None, giving 9 combinations. The True cases for is_zero and
        # is_infinite are mutually exclusive though so there are 3 main cases:
        #
        # - is_zero = True
        # - is_infinite = True
        # - is_zero and is_infinite are both either False or None
        #
        # At the end seen_zero and seen_infinite can be any of 9 combinations
        # of True/False/None. Unless one is False though we cannot return
        # anything except None:
        #
        # - is_zero=True needs seen_zero=True and seen_infinite=False
        # - is_zero=False needs seen_zero=False
        # - is_infinite=True needs seen_infinite=True and seen_zero=False
        # - is_infinite=False needs seen_infinite=False
        # - anything else gives both is_zero=None and is_infinite=None
        #
        # The loop only sets the flags to True or None and never back to False.
        # Hence as soon as neither flag is False we exit early returning None.
        # In particular as soon as we encounter a single arg that has
        # is_zero=is_infinite=None we exit. This is a common case since it is
        # the default assumptions for a Symbol and also the case for most
        # expressions containing such a symbol. The early exit gives a big
        # speedup for something like Mul(*symbols('x:1000')).is_zero.
        #
        seen_zero = seen_infinite = False

        for a in self.args:
            if a.is_zero:
                if seen_infinite is not False:
                    return None, None
                seen_zero = True
            elif a.is_infinite:
                if seen_zero is not False:
                    return None, None
                seen_infinite = True
            else:
                if seen_zero is False and a.is_zero is None:
                    if seen_infinite is not False:
                        return None, None
                    seen_zero = None
                if seen_infinite is False and a.is_infinite is None:
                    if seen_zero is not False:
                        return None, None
                    seen_infinite = None

        return seen_zero, seen_infinite

    def _eval_is_zero(self):
        # True iff any arg is zero and no arg is infinite but need to handle
        # three valued logic carefully.
        seen_zero, seen_infinite = self._eval_is_zero_infinite_helper()

        if seen_zero is False:
            return False
        elif seen_zero is True and seen_infinite is False:
            return True
        else:
            return None

    def _eval_is_infinite(self):
        # True iff any arg is infinite and no arg is zero but need to handle
        # three valued logic carefully.
        seen_zero, seen_infinite = self._eval_is_zero_infinite_helper()

        if seen_infinite is True and seen_zero is False:
            return True
        elif seen_infinite is False:
            return False
        else:
            return None

    # We do not need to implement _eval_is_finite because the assumptions
    # system can infer it from finite = not infinite.

    def _eval_is_rational(self):
        r = _fuzzy_group((a.is_rational for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            # All args except one are rational
            if all(a.is_zero is False for a in self.args):
                return False

    def _eval_is_algebraic(self):
        r = _fuzzy_group((a.is_algebraic for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            # All args except one are algebraic
            if all(a.is_zero is False for a in self.args):
                return False

    # without involving odd/even checks this code would suffice:
    #_eval_is_integer = lambda self: _fuzzy_group(
    #    (a.is_integer for a in self.args), quick_exit=True)
    def _eval_is_integer(self):
        is_rational = self._eval_is_rational()
        if is_rational is False:
            return False

        numerators = []
        denominators = []
        unknown = False
        for a in self.args:
            hit = False
            if a.is_integer:
                if abs(a) is not S.One:
                    numerators.append(a)
            elif a.is_Rational:
                n, d = a.as_numer_denom()
                if abs(n) is not S.One:
                    numerators.append(n)
                if d is not S.One:
                    denominators.append(d)
            elif a.is_Pow:
                b, e = a.as_base_exp()
                if not b.is_integer or not e.is_integer:
                    hit = unknown = True
                if e.is_negative:
                    denominators.append(2 if a is S.Half else
                        Pow(a, S.NegativeOne))
                elif not hit:
                    # int b and pos int e: a = b**e is integer
                    assert not e.is_positive
                    # for rational self and e equal to zero: a = b**e is 1
                    assert not e.is_zero
                    return # sign of e unknown -> self.is_integer unknown
                else:
                    # x**2, 2**x, or x**y with x and y int-unknown -> unknown
                    return
            else:
                return

        if not denominators and not unknown:
            return True

        allodd = lambda x: all(i.is_odd for i in x)
        alleven = lambda x: all(i.is_even for i in x)
        anyeven = lambda x: any(i.is_even for i in x)

        from .relational import is_gt
        if not numerators and denominators and all(
                is_gt(_, S.One) for _ in denominators):
            return False
        elif unknown:
            return
        elif allodd(numerators) and anyeven(denominators):
            return False
        elif anyeven(numerators) and denominators == [2]:
            return True
        elif alleven(numerators) and allodd(denominators
                ) and (Mul(*denominators, evaluate=False) - 1
                ).is_positive:
            return False
        if len(denominators) == 1:
            d = denominators[0]
            if d.is_Integer and d.is_even:
                # if minimal power of 2 in num vs den is not
                # negative then we have an integer
                if (Add(*[i.as_base_exp()[1] for i in
                        numerators if i.is_even]) - trailing(d.p)
                        ).is_nonnegative:
                    return True
        if len(numerators) == 1:
            n = numerators[0]
            if n.is_Integer and n.is_even:
                # if minimal power of 2 in den vs num is positive
                # then we have have a non-integer
                if (Add(*[i.as_base_exp()[1] for i in
                        denominators if i.is_even]) - trailing(n.p)
                        ).is_positive:
                    return False

    def _eval_is_polar(self):
        has_polar = any(arg.is_polar for arg in self.args)
        return has_polar and \
            all(arg.is_polar or arg.is_positive for arg in self.args)

    def _eval_is_extended_real(self):
        return self._eval_real_imag(True)

    def _eval_real_imag(self, real):
        zero = False
        t_not_re_im = None

        for t in self.args:
            if (t.is_complex or t.is_infinite) is False and t.is_extended_real is False:
                return False
            elif t.is_imaginary:  # I
                real = not real
            elif t.is_extended_real:  # 2
                if not zero:
                    z = t.is_zero
                    if not z and zero is False:
                        zero = z
                    elif z:
                        if all(a.is_finite for a in self.args):
                            return True
                        return
            elif t.is_extended_real is False:
                # symbolic or literal like `2 + I` or symbolic imaginary
                if t_not_re_im:
                    return  # complex terms might cancel
                t_not_re_im = t
            elif t.is_imaginary is False:  # symbolic like `2` or `2 + I`
                if t_not_re_im:
                    return  # complex terms might cancel
                t_not_re_im = t
            else:
                return

        if t_not_re_im:
            if t_not_re_im.is_extended_real is False:
                if real:  # like 3
                    return zero  # 3*(smthng like 2 + I or i) is not real
            if t_not_re_im.is_imaginary is False:  # symbolic 2 or 2 + I
                if not real:  # like I
                    return zero  # I*(smthng like 2 or 2 + I) is not real
        elif zero is False:
            return real  # can't be trumped by 0
        elif real:
            return real  # doesn't matter what zero is

    def _eval_is_imaginary(self):
        if all(a.is_zero is False and a.is_finite for a in self.args):
            return self._eval_real_imag(False)

    def _eval_is_hermitian(self):
        return self._eval_herm_antiherm(True)

    def _eval_is_antihermitian(self):
        return self._eval_herm_antiherm(False)

    def _eval_herm_antiherm(self, herm):
        for t in self.args:
            if t.is_hermitian is None or t.is_antihermitian is None:
                return
            if t.is_hermitian:
                continue
            elif t.is_antihermitian:
                herm = not herm
            else:
                return

        if herm is not False:
            return herm

        is_zero = self._eval_is_zero()
        if is_zero:
            return True
        elif is_zero is False:
            return herm

    def _eval_is_irrational(self):
        for t in self.args:
            a = t.is_irrational
            if a:
                others = list(self.args)
                others.remove(t)
                if all((x.is_rational and fuzzy_not(x.is_zero)) is True for x in others):
                    return True
                return
            if a is None:
                return
        if all(x.is_real for x in self.args):
            return False

    def _eval_is_extended_positive(self):
        """Return True if self is positive, False if not, and None if it
        cannot be determined.

        Explanation
        ===========

        This algorithm is non-recursive and works by keeping track of the
        sign which changes when a negative or nonpositive is encountered.
        Whether a nonpositive or nonnegative is seen is also tracked since
        the presence of these makes it impossible to return True, but
        possible to return False if the end result is nonpositive. e.g.

            pos * neg * nonpositive -> pos or zero -> None is returned
            pos * neg * nonnegative -> neg or zero -> False is returned
        """
        return self._eval_pos_neg(1)

    def _eval_pos_neg(self, sign):
        saw_NON = saw_NOT = False
        for t in self.args:
            if t.is_extended_positive:
                continue
            elif t.is_extended_negative:
                sign = -sign
            elif t.is_zero:
                if all(a.is_finite for a in self.args):
                    return False
                return
            elif t.is_extended_nonpositive:
                sign = -sign
                saw_NON = True
            elif t.is_extended_nonnegative:
                saw_NON = True
            # FIXME: is_positive/is_negative is False doesn't take account of
            # Symbol('x', infinite=True, extended_real=True) which has
            # e.g. is_positive is False but has uncertain sign.
            elif t.is_positive is False:
                sign = -sign
                if saw_NOT:
                    return
                saw_NOT = True
            elif t.is_negative is False:
                if saw_NOT:
                    return
                saw_NOT = True
            else:
                return
        if sign == 1 and saw_NON is False and saw_NOT is False:
            return True
        if sign < 0:
            return False

    def _eval_is_extended_negative(self):
        return self._eval_pos_neg(-1)

    def _eval_is_odd(self):
        is_integer = self._eval_is_integer()
        if is_integer is not True:
            return is_integer

        from sympy.simplify.radsimp import fraction
        n, d = fraction(self)
        if d.is_Integer and d.is_even:
            # if minimal power of 2 in num vs den is
            # positive then we have an even number
            if (Add(*[i.as_base_exp()[1] for i in
                    Mul.make_args(n) if i.is_even]) - trailing(d.p)
                    ).is_positive:
                return False
            return
        r, acc = True, 1
        for t in self.args:
            if abs(t) is S.One:
                continue
            if t.is_even:
                return False
            if r is False:
                pass
            elif acc != 1 and (acc + t).is_odd:
                r = False
            elif t.is_even is None:
                r = None
            acc = t
        return r

    def _eval_is_even(self):
        from sympy.simplify.radsimp import fraction
        n, d = fraction(self)
        if n.is_Integer and n.is_even:
            # if minimal power of 2 in den vs num is not
            # negative then this is not an integer and
            # can't be even
            if (Add(*[i.as_base_exp()[1] for i in
                    Mul.make_args(d) if i.is_even]) - trailing(n.p)
                    ).is_nonnegative:
                return False

    def _eval_is_composite(self):
        """
        Here we count the number of arguments that have a minimum value
        greater than two.
        If there are more than one of such a symbol then the result is composite.
        Else, the result cannot be determined.
        """
        number_of_args = 0 # count of symbols with minimum value greater than one
        for arg in self.args:
            if not (arg.is_integer and arg.is_positive):
                return None
            if (arg-1).is_positive:
                number_of_args += 1

        if number_of_args > 1:
            return True

    def _eval_subs(self, old, new):
        from sympy.functions.elementary.complexes import sign
        from sympy.ntheory.factor_ import multiplicity
        from sympy.simplify.powsimp import powdenest
        from sympy.simplify.radsimp import fraction

        if not old.is_Mul:
            return None

        # try keep replacement literal so -2*x doesn't replace 4*x
        if old.args[0].is_Number and old.args[0] < 0:
            if self.args[0].is_Number:
                if self.args[0] < 0:
                    return self._subs(-old, -new)
                return None

        def base_exp(a):
            # if I and -1 are in a Mul, they get both end up with
            # a -1 base (see issue 6421); all we want here are the
            # true Pow or exp separated into base and exponent
            from sympy.functions.elementary.exponential import exp
            if a.is_Pow or isinstance(a, exp):
                return a.as_base_exp()
            return a, S.One

        def breakup(eq):
            """break up powers of eq when treated as a Mul:
                   b**(Rational*e) -> b**e, Rational
                commutatives come back as a dictionary {b**e: Rational}
                noncommutatives come back as a list [(b**e, Rational)]
            """

            (c, nc) = (defaultdict(int), [])
            for a in Mul.make_args(eq):
                a = powdenest(a)
                (b, e) = base_exp(a)
                if e is not S.One:
                    (co, _) = e.as_coeff_mul()
                    b = Pow(b, e/co)
                    e = co
                if a.is_commutative:
                    c[b] += e
                else:
                    nc.append([b, e])
            return (c, nc)

        def rejoin(b, co):
            """
            Put rational back with exponent; in general this is not ok, but
            since we took it from the exponent for analysis, it's ok to put
            it back.
            """

            (b, e) = base_exp(b)
            return Pow(b, e*co)

        def ndiv(a, b):
            """if b divides a in an extractive way (like 1/4 divides 1/2
            but not vice versa, and 2/5 does not divide 1/3) then return
            the integer number of times it divides, else return 0.
            """
            if not b.q % a.q or not a.q % b.q:
                return int(a/b)
            return 0

        # give Muls in the denominator a chance to be changed (see issue 5651)
        # rv will be the default return value
        rv = None
        n, d = fraction(self)
        self2 = self
        if d is not S.One:
            self2 = n._subs(old, new)/d._subs(old, new)
            if not self2.is_Mul:
                return self2._subs(old, new)
            if self2 != self:
                rv = self2

        # Now continue with regular substitution.

        # handle the leading coefficient and use it to decide if anything
        # should even be started; we always know where to find the Rational
        # so it's a quick test

        co_self = self2.args[0]
        co_old = old.args[0]
        co_xmul = None
        if co_old.is_Rational and co_self.is_Rational:
            # if coeffs are the same there will be no updating to do
            # below after breakup() step; so skip (and keep co_xmul=None)
            if co_old != co_self:
                co_xmul = co_self.extract_multiplicatively(co_old)
        elif co_old.is_Rational:
            return rv

        # break self and old into factors

        (c, nc) = breakup(self2)
        (old_c, old_nc) = breakup(old)

        # update the coefficients if we had an extraction
        # e.g. if co_self were 2*(3/35*x)**2 and co_old = 3/5
        # then co_self in c is replaced by (3/5)**2 and co_residual
        # is 2*(1/7)**2

        if co_xmul and co_xmul.is_Rational and abs(co_old) != 1:
            mult = S(multiplicity(abs(co_old), co_self))
            c.pop(co_self)
            if co_old in c:
                c[co_old] += mult
            else:
                c[co_old] = mult
            co_residual = co_self/co_old**mult
        else:
            co_residual = 1

        # do quick tests to see if we can't succeed

        ok = True
        if len(old_nc) > len(nc):
            # more non-commutative terms
            ok = False
        elif len(old_c) > len(c):
            # more commutative terms
            ok = False
        elif {i[0] for i in old_nc}.difference({i[0] for i in nc}):
            # unmatched non-commutative bases
            ok = False
        elif set(old_c).difference(set(c)):
            # unmatched commutative terms
            ok = False
        elif any(sign(c[b]) != sign(old_c[b]) for b in old_c):
            # differences in sign
            ok = False
        if not ok:
            return rv

        if not old_c:
            cdid = None
        else:
            rat = []
            for (b, old_e) in old_c.items():
                c_e = c[b]
                rat.append(ndiv(c_e, old_e))
                if not rat[-1]:
                    return rv
            cdid = min(rat)

        if not old_nc:
            ncdid = None
            for i in range(len(nc)):
                nc[i] = rejoin(*nc[i])
        else:
            ncdid = 0  # number of nc replacements we did
            take = len(old_nc)  # how much to look at each time
            limit = cdid or S.Infinity  # max number that we can take
            failed = []  # failed terms will need subs if other terms pass
            i = 0
            while limit and i + take <= len(nc):
                hit = False

                # the bases must be equivalent in succession, and
                # the powers must be extractively compatible on the
                # first and last factor but equal in between.

                rat = []
                for j in range(take):
                    if nc[i + j][0] != old_nc[j][0]:
                        break
                    elif j == 0:
                        rat.append(ndiv(nc[i + j][1], old_nc[j][1]))
                    elif j == take - 1:
                        rat.append(ndiv(nc[i + j][1], old_nc[j][1]))
                    elif nc[i + j][1] != old_nc[j][1]:
                        break
                    else:
                        rat.append(1)
                    j += 1
                else:
                    ndo = min(rat)
                    if ndo:
                        if take == 1:
                            if cdid:
                                ndo = min(cdid, ndo)
                            nc[i] = Pow(new, ndo)*rejoin(nc[i][0],
                                    nc[i][1] - ndo*old_nc[0][1])
                        else:
                            ndo = 1

                            # the left residual

                            l = rejoin(nc[i][0], nc[i][1] - ndo*
                                    old_nc[0][1])

                            # eliminate all middle terms

                            mid = new

                            # the right residual (which may be the same as the middle if take == 2)

                            ir = i + take - 1
                            r = (nc[ir][0], nc[ir][1] - ndo*
                                 old_nc[-1][1])
                            if r[1]:
                                if i + take < len(nc):
                                    nc[i:i + take] = [l*mid, r]
                                else:
                                    r = rejoin(*r)
                                    nc[i:i + take] = [l*mid*r]
                            else:

                                # there was nothing left on the right

                                nc[i:i + take] = [l*mid]

                        limit -= ndo
                        ncdid += ndo
                        hit = True
                if not hit:

                    # do the subs on this failing factor

                    failed.append(i)
                i += 1
            else:

                if not ncdid:
                    return rv

                # although we didn't fail, certain nc terms may have
                # failed so we rebuild them after attempting a partial
                # subs on them

                failed.extend(range(i, len(nc)))
                for i in failed:
                    nc[i] = rejoin(*nc[i]).subs(old, new)

        # rebuild the expression

        if cdid is None:
            do = ncdid
        elif ncdid is None:
            do = cdid
        else:
            do = min(ncdid, cdid)

        margs = []
        for b in c:
            if b in old_c:

                # calculate the new exponent

                e = c[b] - old_c[b]*do
                margs.append(rejoin(b, e))
            else:
                margs.append(rejoin(b.subs(old, new), c[b]))
        if cdid and not ncdid:

            # in case we are replacing commutative with non-commutative,
            # we want the new term to come at the front just like the
            # rest of this routine

            margs = [Pow(new, cdid)] + margs
        return co_residual*self2.func(*margs)*self2.func(*nc)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from .function import PoleError
        from sympy.functions.elementary.integers import ceiling
        from sympy.series.order import Order

        def coeff_exp(term, x):
            lt = term.as_coeff_exponent(x)
            if lt[0].has(x):
                try:
                    lt = term.leadterm(x)
                except ValueError:
                    return term, S.Zero
            return lt

        ords = []

        try:
            for t in self.args:
                coeff, exp = t.leadterm(x)
                if not coeff.has(x):
                    ords.append((t, exp))
                else:
                    raise ValueError

            n0 = sum(t[1] for t in ords if t[1].is_number)
            facs = []
            for t, m in ords:
                n1 = ceiling(n - n0 + (m if m.is_number else 0))
                s = t.nseries(x, n=n1, logx=logx, cdir=cdir)
                ns = s.getn()
                if ns is not None:
                    if ns < n1:  # less than expected
                        n -= n1 - ns    # reduce n
                facs.append(s)

        except (ValueError, NotImplementedError, TypeError, PoleError):
            # XXX: Catching so many generic exceptions around a large block of
            # code will mask bugs. Whatever purpose catching these exceptions
            # serves should be handled in a different way.
            n0 = sympify(sum(t[1] for t in ords if t[1].is_number))
            if n0.is_nonnegative:
                n0 = S.Zero
            facs = [t.nseries(x, n=ceiling(n-n0), logx=logx, cdir=cdir) for t in self.args]
            from sympy.simplify.powsimp import powsimp
            res = powsimp(self.func(*facs).expand(), combine='exp', deep=True)
            if res.has(Order):
                res += Order(x**n, x)
            return res

        res = S.Zero
        ords2 = [Add.make_args(factor) for factor in facs]

        for fac in product(*ords2):
            ords3 = [coeff_exp(term, x) for term in fac]
            coeffs, powers = zip(*ords3)
            power = sum(powers)
            if (power - n).is_negative:
                res += Mul(*coeffs)*(x**power)

        def max_degree(e, x):
            if e is x:
                return S.One
            if e.is_Atom:
                return S.Zero
            if e.is_Add:
                return max(max_degree(a, x) for a in e.args)
            if e.is_Mul:
                return Add(*[max_degree(a, x) for a in e.args])
            if e.is_Pow:
                return max_degree(e.base, x)*e.exp
            return S.Zero

        if self.is_polynomial(x):
            from sympy.polys.polyerrors import PolynomialError
            from sympy.polys.polytools import degree
            try:
                if max_degree(self, x) >= n or degree(self, x) != degree(res, x):
                    res += Order(x**n, x)
            except PolynomialError:
                pass
            else:
                return res

        if res != self:
            if (self - res).subs(x, 0) == S.Zero and n > 0:
                lt = self._eval_as_leading_term(x, logx=logx, cdir=cdir)
                if lt == S.Zero:
                    return res
            res += Order(x**n, x)
        return res

    def _eval_as_leading_term(self, x, logx, cdir):
        return self.func(*[t.as_leading_term(x, logx=logx, cdir=cdir) for t in self.args])

    def _eval_conjugate(self):
        return self.func(*[t.conjugate() for t in self.args])

    def _eval_transpose(self):
        return self.func(*[t.transpose() for t in self.args[::-1]])

    def _eval_adjoint(self):
        return self.func(*[t.adjoint() for t in self.args[::-1]])

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import sqrt
        >>> (-3*sqrt(2)*(2 - 2*sqrt(2))).as_content_primitive()
        (6, -sqrt(2)*(1 - sqrt(2)))

        See docstring of Expr.as_content_primitive for more examples.
        """

        coef = S.One
        args = []
        for a in self.args:
            c, p = a.as_content_primitive(radical=radical, clear=clear)
            coef *= c
            if p is not S.One:
                args.append(p)
        # don't use self._from_args here to reconstruct args
        # since there may be identical args now that should be combined
        # e.g. (2+2*x)*(3+3*x) should be (6, (1 + x)**2) not (6, (1+x)*(1+x))
        return coef, self.func(*args)

    def as_ordered_factors(self, order=None):
        """Transform an expression into an ordered list of factors.

        Examples
        ========

        >>> from sympy import sin, cos
        >>> from sympy.abc import x, y

        >>> (2*x*y*sin(x)*cos(x)).as_ordered_factors()
        [2, x, y, sin(x), cos(x)]

        """
        cpart, ncpart = self.args_cnc()
        cpart.sort(key=lambda expr: expr.sort_key(order=order))
        return cpart + ncpart

    @property
    def _sorted_args(self):
        return tuple(self.as_ordered_factors())

mul = AssocOpDispatcher('mul')


def prod(a, start=1):
    """Return product of elements of a. Start with int 1 so if only
       ints are included then an int result is returned.

    Examples
    ========

    >>> from sympy import prod, S
    >>> prod(range(3))
    0
    >>> type(_) is int
    True
    >>> prod([S(2), 3])
    6
    >>> _.is_Integer
    True

    You can start the product at something other than 1:

    >>> prod([1, 2], 3)
    6

    """
    return reduce(operator.mul, a, start)


def _keep_coeff(coeff, factors, clear=True, sign=False):
    """Return ``coeff*factors`` unevaluated if necessary.

    If ``clear`` is False, do not keep the coefficient as a factor
    if it can be distributed on a single factor such that one or
    more terms will still have integer coefficients.

    If ``sign`` is True, allow a coefficient of -1 to remain factored out.

    Examples
    ========

    >>> from sympy.core.mul import _keep_coeff
    >>> from sympy.abc import x, y
    >>> from sympy import S

    >>> _keep_coeff(S.Half, x + 2)
    (x + 2)/2
    >>> _keep_coeff(S.Half, x + 2, clear=False)
    x/2 + 1
    >>> _keep_coeff(S.Half, (x + 2)*y, clear=False)
    y*(x + 2)/2
    >>> _keep_coeff(S(-1), x + y)
    -x - y
    >>> _keep_coeff(S(-1), x + y, sign=True)
    -(x + y)
    """
    if not coeff.is_Number:
        if factors.is_Number:
            factors, coeff = coeff, factors
        else:
            return coeff*factors
    if factors is S.One:
        return coeff
    if coeff is S.One:
        return factors
    elif coeff is S.NegativeOne and not sign:
        return -factors
    elif factors.is_Add:
        if not clear and coeff.is_Rational and coeff.q != 1:
            args = [i.as_coeff_Mul() for i in factors.args]
            args = [(_keep_coeff(c, coeff), m) for c, m in args]
            if any(c.is_Integer for c, _ in args):
                return Add._from_args([Mul._from_args(
                    i[1:] if i[0] == 1 else i) for i in args])
        return Mul(coeff, factors, evaluate=False)
    elif factors.is_Mul:
        margs = list(factors.args)
        if margs[0].is_Number:
            margs[0] *= coeff
            if margs[0] == 1:
                margs.pop(0)
        else:
            margs.insert(0, coeff)
        return Mul._from_args(margs)
    else:
        m = coeff*factors
        if m.is_Number and not factors.is_Number:
            m = Mul._from_args((coeff, factors))
        return m

def expand_2arg(e):
    def do(e):
        if e.is_Mul:
            c, r = e.as_coeff_Mul()
            if c.is_Number and r.is_Add:
                return _unevaluated_Add(*[c*ri for ri in r.args])
        return e
    return bottom_up(e, do)


from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add