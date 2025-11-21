
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\control\lti.py ===
from typing import Type
from sympy import Interval, numer, Rational, solveset
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions import Abs
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import Matrix, ImmutableMatrix, ImmutableDenseMatrix, eye, ShapeError, zeros
from sympy.functions.elementary.exponential import (exp, log)
from sympy.matrices.expressions import MatMul, MatAdd
from sympy.polys import Poly, rootof
from sympy.polys.polyroots import roots
from sympy.polys.polytools import (cancel, degree)
from sympy.series import limit
from sympy.utilities.misc import filldedent
from sympy.solvers.ode.systems import linodesolve
from sympy.solvers.solveset import linsolve, linear_eq_to_matrix

from mpmath.libmp.libmpf import prec_to_dps

__all__ = ['TransferFunction', 'PIDController', 'Series', 'MIMOSeries', 'Parallel', 'MIMOParallel',
    'Feedback', 'MIMOFeedback', 'TransferFunctionMatrix', 'StateSpace', 'gbt', 'bilinear', 'forward_diff', 'backward_diff',
    'phase_margin', 'gain_margin']

def _roots(poly, var):
    """ like roots, but works on higher-order polynomials. """
    r = roots(poly, var, multiple=True)
    n = degree(poly)
    if len(r) != n:
        r = [rootof(poly, var, k) for k in range(n)]
    return r

def gbt(tf, sample_per, alpha):
    r"""
    Returns falling coefficients of H(z) from numerator and denominator.

    Explanation
    ===========

    Where H(z) is the corresponding discretized transfer function,
    discretized with the generalised bilinear transformation method.
    H(z) is obtained from the continuous transfer function H(s)
    by substituting $s(z) = \frac{z-1}{T(\alpha z + (1-\alpha))}$ into H(s), where T is the
    sample period.
    Coefficients are falling, i.e. $H(z) = \frac{az+b}{cz+d}$ is returned
    as [a, b], [c, d].

    Examples
    ========

    >>> from sympy.physics.control.lti import TransferFunction, gbt
    >>> from sympy.abc import s, L, R, T

    >>> tf = TransferFunction(1, s*L + R, s)
    >>> numZ, denZ = gbt(tf, T, 0.5)
    >>> numZ
    [T/(2*(L + R*T/2)), T/(2*(L + R*T/2))]
    >>> denZ
    [1, (-L + R*T/2)/(L + R*T/2)]

    >>> numZ, denZ = gbt(tf, T, 0)
    >>> numZ
    [T/L]
    >>> denZ
    [1, (-L + R*T)/L]

    >>> numZ, denZ = gbt(tf, T, 1)
    >>> numZ
    [T/(L + R*T), 0]
    >>> denZ
    [1, -L/(L + R*T)]

    >>> numZ, denZ = gbt(tf, T, 0.3)
    >>> numZ
    [3*T/(10*(L + 3*R*T/10)), 7*T/(10*(L + 3*R*T/10))]
    >>> denZ
    [1, (-L + 7*R*T/10)/(L + 3*R*T/10)]

    References
    ==========

    .. [1] https://www.polyu.edu.hk/ama/profile/gfzhang/Research/ZCC09_IJC.pdf
    """
    if not tf.is_SISO:
        raise NotImplementedError("Not implemented for MIMO systems.")

    T = sample_per  # and sample period T
    s = tf.var
    z =  s         # dummy discrete variable z

    np = tf.num.as_poly(s).all_coeffs()
    dp = tf.den.as_poly(s).all_coeffs()
    alpha = Rational(alpha).limit_denominator(1000)

    # The next line results from multiplying H(z) with z^N/z^N
    N = max(len(np), len(dp)) - 1
    num = Add(*[ T**(N-i) * c * (z-1)**i * (alpha * z + 1 - alpha)**(N-i) for c, i in zip(np[::-1], range(len(np))) ])
    den = Add(*[ T**(N-i) * c * (z-1)**i * (alpha * z + 1 - alpha)**(N-i) for c, i in zip(dp[::-1], range(len(dp))) ])

    num_coefs = num.as_poly(z).all_coeffs()
    den_coefs = den.as_poly(z).all_coeffs()

    para = den_coefs[0]
    num_coefs = [coef/para for coef in num_coefs]
    den_coefs = [coef/para for coef in den_coefs]

    return num_coefs, den_coefs

def bilinear(tf, sample_per):
    r"""
    Returns falling coefficients of H(z) from numerator and denominator.

    Explanation
    ===========

    Where H(z) is the corresponding discretized transfer function,
    discretized with the bilinear transform method.
    H(z) is obtained from the continuous transfer function H(s)
    by substituting $s(z) = \frac{2}{T}\frac{z-1}{z+1}$ into H(s), where T is the
    sample period.
    Coefficients are falling, i.e. $H(z) = \frac{az+b}{cz+d}$ is returned
    as [a, b], [c, d].

    Examples
    ========

    >>> from sympy.physics.control.lti import TransferFunction, bilinear
    >>> from sympy.abc import s, L, R, T

    >>> tf = TransferFunction(1, s*L + R, s)
    >>> numZ, denZ = bilinear(tf, T)
    >>> numZ
    [T/(2*(L + R*T/2)), T/(2*(L + R*T/2))]
    >>> denZ
    [1, (-L + R*T/2)/(L + R*T/2)]
    """
    return gbt(tf, sample_per, S.Half)

def forward_diff(tf, sample_per):
    r"""
    Returns falling coefficients of H(z) from numerator and denominator.

    Explanation
    ===========

    Where H(z) is the corresponding discretized transfer function,
    discretized with the forward difference transform method.
    H(z) is obtained from the continuous transfer function H(s)
    by substituting $s(z) = \frac{z-1}{T}$ into H(s), where T is the
    sample period.
    Coefficients are falling, i.e. $H(z) = \frac{az+b}{cz+d}$ is returned
    as [a, b], [c, d].

    Examples
    ========

    >>> from sympy.physics.control.lti import TransferFunction, forward_diff
    >>> from sympy.abc import s, L, R, T

    >>> tf = TransferFunction(1, s*L + R, s)
    >>> numZ, denZ = forward_diff(tf, T)
    >>> numZ
    [T/L]
    >>> denZ
    [1, (-L + R*T)/L]
    """
    return gbt(tf, sample_per, S.Zero)

def backward_diff(tf, sample_per):
    r"""
    Returns falling coefficients of H(z) from numerator and denominator.

    Explanation
    ===========

    Where H(z) is the corresponding discretized transfer function,
    discretized with the backward difference transform method.
    H(z) is obtained from the continuous transfer function H(s)
    by substituting $s(z) =  \frac{z-1}{Tz}$ into H(s), where T is the
    sample period.
    Coefficients are falling, i.e. $H(z) = \frac{az+b}{cz+d}$ is returned
    as [a, b], [c, d].

    Examples
    ========

    >>> from sympy.physics.control.lti import TransferFunction, backward_diff
    >>> from sympy.abc import s, L, R, T

    >>> tf = TransferFunction(1, s*L + R, s)
    >>> numZ, denZ = backward_diff(tf, T)
    >>> numZ
    [T/(L + R*T), 0]
    >>> denZ
    [1, -L/(L + R*T)]
    """
    return gbt(tf, sample_per, S.One)

def phase_margin(system):
    r"""
    Returns the phase margin of a continuous time system.
    Only applicable to Transfer Functions which can generate valid bode plots.

    Raises
    ======

    NotImplementedError
        When time delay terms are present in the system.

    ValueError
        When a SISO LTI system is not passed.

        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

    Examples
    ========

    >>> from sympy.physics.control import TransferFunction, phase_margin
    >>> from sympy.abc import s

    >>> tf = TransferFunction(1, s**3 + 2*s**2 + s, s)
    >>> phase_margin(tf)
    180*(-pi + atan((-1 + (-2*18**(1/3)/(9 + sqrt(93))**(1/3) + 12**(1/3)*(9 + sqrt(93))**(1/3))**2/36)/(-12**(1/3)*(9 + sqrt(93))**(1/3)/3 + 2*18**(1/3)/(3*(9 + sqrt(93))**(1/3)))))/pi + 180
    >>> phase_margin(tf).n()
    21.3863897518751

    >>> tf1 = TransferFunction(s**3, s**2 + 5*s, s)
    >>> phase_margin(tf1)
    -180 + 180*(atan(sqrt(2)*(-51/10 - sqrt(101)/10)*sqrt(1 + sqrt(101))/(2*(sqrt(101)/2 + 51/2))) + pi)/pi
    >>> phase_margin(tf1).n()
    -25.1783920627277

    >>> tf2 = TransferFunction(1, s + 1, s)
    >>> phase_margin(tf2)
    -180

    See Also
    ========

    gain_margin

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Phase_margin

    """
    from sympy.functions import arg

    if not isinstance(system, SISOLinearTimeInvariant):
        raise ValueError("Margins are only applicable for SISO LTI systems.")

    _w = Dummy("w", real=True)
    repl = I*_w
    expr = system.to_expr()
    len_free_symbols = len(expr.free_symbols)
    if expr.has(exp):
        raise NotImplementedError("Margins for systems with Time delay terms are not supported.")
    elif len_free_symbols > 1:
        raise ValueError("Extra degree of freedom found. Make sure"
            " that there are no free symbols in the dynamical system other"
            " than the variable of Laplace transform.")

    w_expr = expr.subs({system.var: repl})

    mag = 20*log(Abs(w_expr), 10)
    mag_sol = list(solveset(mag, _w, Interval(0, oo, left_open=True)))

    if (len(mag_sol) == 0):
        pm = S(-180)
    else:
        wcp = mag_sol[0]
        pm = ((arg(w_expr)*S(180)/pi).subs({_w:wcp}) + S(180)) % 360

    if(pm >= 180):
        pm = pm - 360

    return pm

def gain_margin(system):
    r"""
    Returns the gain margin of a continuous time system.
    Only applicable to Transfer Functions which can generate valid bode plots.

    Raises
    ======

    NotImplementedError
        When time delay terms are present in the system.

    ValueError
        When a SISO LTI system is not passed.

        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

    Examples
    ========

    >>> from sympy.physics.control import TransferFunction, gain_margin
    >>> from sympy.abc import s

    >>> tf = TransferFunction(1, s**3 + 2*s**2 + s, s)
    >>> gain_margin(tf)
    20*log(2)/log(10)
    >>> gain_margin(tf).n()
    6.02059991327962

    >>> tf1 = TransferFunction(s**3, s**2 + 5*s, s)
    >>> gain_margin(tf1)
    oo

    See Also
    ========

    phase_margin

    References
    ==========

    https://en.wikipedia.org/wiki/Bode_plot

    """
    if not isinstance(system, SISOLinearTimeInvariant):
        raise ValueError("Margins are only applicable for SISO LTI systems.")

    _w = Dummy("w", real=True)
    repl = I*_w
    expr = system.to_expr()
    len_free_symbols = len(expr.free_symbols)
    if expr.has(exp):
        raise NotImplementedError("Margins for systems with Time delay terms are not supported.")
    elif len_free_symbols > 1:
        raise ValueError("Extra degree of freedom found. Make sure"
            " that there are no free symbols in the dynamical system other"
            " than the variable of Laplace transform.")

    w_expr = expr.subs({system.var: repl})

    mag = 20*log(Abs(w_expr), 10)
    phase = w_expr
    phase_sol = list(solveset(numer(phase.as_real_imag()[1].cancel()),_w, Interval(0, oo, left_open = True)))

    if (len(phase_sol) == 0):
        gm = oo
    else:
        wcg = phase_sol[0]
        gm = -mag.subs({_w:wcg})

    return gm

class LinearTimeInvariant(Basic, EvalfMixin):
    """A common class for all the Linear Time-Invariant Dynamical Systems."""

    _clstype: Type

    # Users should not directly interact with this class.
    def __new__(cls, *system, **kwargs):
        if cls is LinearTimeInvariant:
            raise NotImplementedError('The LTICommon class is not meant to be used directly.')
        return super(LinearTimeInvariant, cls).__new__(cls, *system, **kwargs)

    @classmethod
    def _check_args(cls, args):
        if not args:
            raise ValueError("At least 1 argument must be passed.")
        if not all(isinstance(arg, cls._clstype) for arg in args):
            raise TypeError(f"All arguments must be of type {cls._clstype}.")
        var_set = {arg.var for arg in args}
        if len(var_set) != 1:
            raise ValueError(filldedent(f"""
                All transfer functions should use the same complex variable
                of the Laplace transform. {len(var_set)} different
                values found."""))

    @property
    def is_SISO(self):
        """Returns `True` if the passed LTI system is SISO else returns False."""
        return self._is_SISO


class SISOLinearTimeInvariant(LinearTimeInvariant):
    """A common class for all the SISO Linear Time-Invariant Dynamical Systems."""
    # Users should not directly interact with this class.

    @property
    def num_inputs(self):
        """Return the number of inputs for SISOLinearTimeInvariant."""
        return 1

    @property
    def num_outputs(self):
        """Return the number of outputs for SISOLinearTimeInvariant."""
        return 1

    _is_SISO = True


class MIMOLinearTimeInvariant(LinearTimeInvariant):
    """A common class for all the MIMO Linear Time-Invariant Dynamical Systems."""
    # Users should not directly interact with this class.
    _is_SISO = False


SISOLinearTimeInvariant._clstype = SISOLinearTimeInvariant
MIMOLinearTimeInvariant._clstype = MIMOLinearTimeInvariant


def _check_other_SISO(func):
    def wrapper(*args, **kwargs):
        if not isinstance(args[-1], SISOLinearTimeInvariant):
            return NotImplemented
        else:
            return func(*args, **kwargs)
    return wrapper


def _check_other_MIMO(func):
    def wrapper(*args, **kwargs):
        if not isinstance(args[-1], MIMOLinearTimeInvariant):
            return NotImplemented
        else:
            return func(*args, **kwargs)
    return wrapper


class TransferFunction(SISOLinearTimeInvariant):
    r"""
    A class for representing LTI (Linear, time-invariant) systems that can be strictly described
    by ratio of polynomials in the Laplace transform complex variable. The arguments
    are ``num``, ``den``, and ``var``, where ``num`` and ``den`` are numerator and
    denominator polynomials of the ``TransferFunction`` respectively, and the third argument is
    a complex variable of the Laplace transform used by these polynomials of the transfer function.
    ``num`` and ``den`` can be either polynomials or numbers, whereas ``var``
    has to be a :py:class:`~.Symbol`.

    Explanation
    ===========

    Generally, a dynamical system representing a physical model can be described in terms of Linear
    Ordinary Differential Equations like -

            $b_{m}y^{\left(m\right)}+b_{m-1}y^{\left(m-1\right)}+\dots+b_{1}y^{\left(1\right)}+b_{0}y=
            a_{n}x^{\left(n\right)}+a_{n-1}x^{\left(n-1\right)}+\dots+a_{1}x^{\left(1\right)}+a_{0}x$

    Here, $x$ is the input signal and $y$ is the output signal and superscript on both is the order of derivative
    (not exponent). Derivative is taken with respect to the independent variable, $t$. Also, generally $m$ is greater
    than $n$.

    It is not feasible to analyse the properties of such systems in their native form therefore, we use
    mathematical tools like Laplace transform to get a better perspective. Taking the Laplace transform
    of both the sides in the equation (at zero initial conditions), we get -

            $\mathcal{L}[b_{m}y^{\left(m\right)}+b_{m-1}y^{\left(m-1\right)}+\dots+b_{1}y^{\left(1\right)}+b_{0}y]=
            \mathcal{L}[a_{n}x^{\left(n\right)}+a_{n-1}x^{\left(n-1\right)}+\dots+a_{1}x^{\left(1\right)}+a_{0}x]$

    Using the linearity property of Laplace transform and also considering zero initial conditions
    (i.e. $y(0^{-}) = 0$, $y'(0^{-}) = 0$ and so on), the equation
    above gets translated to -

            $b_{m}\mathcal{L}[y^{\left(m\right)}]+\dots+b_{1}\mathcal{L}[y^{\left(1\right)}]+b_{0}\mathcal{L}[y]=
            a_{n}\mathcal{L}[x^{\left(n\right)}]+\dots+a_{1}\mathcal{L}[x^{\left(1\right)}]+a_{0}\mathcal{L}[x]$

    Now, applying Derivative property of Laplace transform,

            $b_{m}s^{m}\mathcal{L}[y]+\dots+b_{1}s\mathcal{L}[y]+b_{0}\mathcal{L}[y]=
            a_{n}s^{n}\mathcal{L}[x]+\dots+a_{1}s\mathcal{L}[x]+a_{0}\mathcal{L}[x]$

    Here, the superscript on $s$ is **exponent**. Note that the zero initial conditions assumption, mentioned above, is very important
    and cannot be ignored otherwise the dynamical system cannot be considered time-independent and the simplified equation above
    cannot be reached.

    Collecting $\mathcal{L}[y]$ and $\mathcal{L}[x]$ terms from both the sides and taking the ratio
    $\frac{ \mathcal{L}\left\{y\right\} }{ \mathcal{L}\left\{x\right\} }$, we get the typical rational form of transfer
    function.

    The numerator of the transfer function is, therefore, the Laplace transform of the output signal
    (The signals are represented as functions of time) and similarly, the denominator
    of the transfer function is the Laplace transform of the input signal. It is also a convention
    to denote the input and output signal's Laplace transform with capital alphabets like shown below.

            $H(s) = \frac{Y(s)}{X(s)} = \frac{ \mathcal{L}\left\{y(t)\right\} }{ \mathcal{L}\left\{x(t)\right\} }$

    $s$, also known as complex frequency, is a complex variable in the Laplace domain. It corresponds to the
    equivalent variable $t$, in the time domain. Transfer functions are sometimes also referred to as the Laplace
    transform of the system's impulse response. Transfer function, $H$, is represented as a rational
    function in $s$ like,

            $H(s) =\ \frac{a_{n}s^{n}+a_{n-1}s^{n-1}+\dots+a_{1}s+a_{0}}{b_{m}s^{m}+b_{m-1}s^{m-1}+\dots+b_{1}s+b_{0}}$

    Parameters
    ==========

    num : Expr, Number
        The numerator polynomial of the transfer function.
    den : Expr, Number
        The denominator polynomial of the transfer function.
    var : Symbol
        Complex variable of the Laplace transform used by the
        polynomials of the transfer function.

    Raises
    ======

    TypeError
        When ``var`` is not a Symbol or when ``num`` or ``den`` is not a
        number or a polynomial.
    ValueError
        When ``den`` is zero.

    Examples
    ========

    >>> from sympy.abc import s, p, a
    >>> from sympy.physics.control.lti import TransferFunction
    >>> tf1 = TransferFunction(s + a, s**2 + s + 1, s)
    >>> tf1
    TransferFunction(a + s, s**2 + s + 1, s)
    >>> tf1.num
    a + s
    >>> tf1.den
    s**2 + s + 1
    >>> tf1.var
    s
    >>> tf1.args
    (a + s, s**2 + s + 1, s)

    Any complex variable can be used for ``var``.

    >>> tf2 = TransferFunction(a*p**3 - a*p**2 + s*p, p + a**2, p)
    >>> tf2
    TransferFunction(a*p**3 - a*p**2 + p*s, a**2 + p, p)
    >>> tf3 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)
    >>> tf3
    TransferFunction((p - 1)*(p + 3), (p - 1)*(p + 5), p)

    To negate a transfer function the ``-`` operator can be prepended:

    >>> tf4 = TransferFunction(-a + s, p**2 + s, p)
    >>> -tf4
    TransferFunction(a - s, p**2 + s, p)
    >>> tf5 = TransferFunction(s**4 - 2*s**3 + 5*s + 4, s + 4, s)
    >>> -tf5
    TransferFunction(-s**4 + 2*s**3 - 5*s - 4, s + 4, s)

    You can use a float or an integer (or other constants) as numerator and denominator:

    >>> tf6 = TransferFunction(1/2, 4, s)
    >>> tf6.num
    0.500000000000000
    >>> tf6.den
    4
    >>> tf6.var
    s
    >>> tf6.args
    (0.5, 4, s)

    You can take the integer power of a transfer function using the ``**`` operator:

    >>> tf7 = TransferFunction(s + a, s - a, s)
    >>> tf7**3
    TransferFunction((a + s)**3, (-a + s)**3, s)
    >>> tf7**0
    TransferFunction(1, 1, s)
    >>> tf8 = TransferFunction(p + 4, p - 3, p)
    >>> tf8**-1
    TransferFunction(p - 3, p + 4, p)

    Addition, subtraction, and multiplication of transfer functions can form
    unevaluated ``Series`` or ``Parallel`` objects.

    >>> tf9 = TransferFunction(s + 1, s**2 + s + 1, s)
    >>> tf10 = TransferFunction(s - p, s + 3, s)
    >>> tf11 = TransferFunction(4*s**2 + 2*s - 4, s - 1, s)
    >>> tf12 = TransferFunction(1 - s, s**2 + 4, s)
    >>> tf9 + tf10
    Parallel(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(-p + s, s + 3, s))
    >>> tf10 - tf11
    Parallel(TransferFunction(-p + s, s + 3, s), TransferFunction(-4*s**2 - 2*s + 4, s - 1, s))
    >>> tf9 * tf10
    Series(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(-p + s, s + 3, s))
    >>> tf10 - (tf9 + tf12)
    Parallel(TransferFunction(-p + s, s + 3, s), TransferFunction(-s - 1, s**2 + s + 1, s), TransferFunction(s - 1, s**2 + 4, s))
    >>> tf10 - (tf9 * tf12)
    Parallel(TransferFunction(-p + s, s + 3, s), Series(TransferFunction(-1, 1, s), TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(1 - s, s**2 + 4, s)))
    >>> tf11 * tf10 * tf9
    Series(TransferFunction(4*s**2 + 2*s - 4, s - 1, s), TransferFunction(-p + s, s + 3, s), TransferFunction(s + 1, s**2 + s + 1, s))
    >>> tf9 * tf11 + tf10 * tf12
    Parallel(Series(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(4*s**2 + 2*s - 4, s - 1, s)), Series(TransferFunction(-p + s, s + 3, s), TransferFunction(1 - s, s**2 + 4, s)))
    >>> (tf9 + tf12) * (tf10 + tf11)
    Series(Parallel(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(1 - s, s**2 + 4, s)), Parallel(TransferFunction(-p + s, s + 3, s), TransferFunction(4*s**2 + 2*s - 4, s - 1, s)))

    These unevaluated ``Series`` or ``Parallel`` objects can convert into the
    resultant transfer function using ``.doit()`` method or by ``.rewrite(TransferFunction)``.

    >>> ((tf9 + tf10) * tf12).doit()
    TransferFunction((1 - s)*((-p + s)*(s**2 + s + 1) + (s + 1)*(s + 3)), (s + 3)*(s**2 + 4)*(s**2 + s + 1), s)
    >>> (tf9 * tf10 - tf11 * tf12).rewrite(TransferFunction)
    TransferFunction(-(1 - s)*(s + 3)*(s**2 + s + 1)*(4*s**2 + 2*s - 4) + (-p + s)*(s - 1)*(s + 1)*(s**2 + 4), (s - 1)*(s + 3)*(s**2 + 4)*(s**2 + s + 1), s)

    See Also
    ========

    Feedback, Series, Parallel

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Transfer_function
    .. [2] https://en.wikipedia.org/wiki/Laplace_transform

    """
    def __new__(cls, num, den, var):
        num, den = _sympify(num), _sympify(den)

        if not isinstance(var, Symbol):
            raise TypeError("Variable input must be a Symbol.")

        if den == 0:
            raise ValueError("TransferFunction cannot have a zero denominator.")

        if (((isinstance(num, (Expr, TransferFunction, Series, Parallel)) and num.has(Symbol)) or num.is_number) and
            ((isinstance(den, (Expr, TransferFunction, Series, Parallel)) and den.has(Symbol)) or den.is_number)):
            cls.is_StateSpace_object = False
            return super(TransferFunction, cls).__new__(cls, num, den, var)

        else:
            raise TypeError("Unsupported type for numerator or denominator of TransferFunction.")

    @classmethod
    def from_rational_expression(cls, expr, var=None):
        r"""
        Creates a new ``TransferFunction`` efficiently from a rational expression.

        Parameters
        ==========

        expr : Expr, Number
            The rational expression representing the ``TransferFunction``.
        var : Symbol, optional
            Complex variable of the Laplace transform used by the
            polynomials of the transfer function.

        Raises
        ======

        ValueError
            When ``expr`` is of type ``Number`` and optional parameter ``var``
            is not passed.

            When ``expr`` has more than one variables and an optional parameter
            ``var`` is not passed.
        ZeroDivisionError
            When denominator of ``expr`` is zero or it has ``ComplexInfinity``
            in its numerator.

        Examples
        ========

        >>> from sympy.abc import s, p, a
        >>> from sympy.physics.control.lti import TransferFunction
        >>> expr1 = (s + 5)/(3*s**2 + 2*s + 1)
        >>> tf1 = TransferFunction.from_rational_expression(expr1)
        >>> tf1
        TransferFunction(s + 5, 3*s**2 + 2*s + 1, s)
        >>> expr2 = (a*p**3 - a*p**2 + s*p)/(p + a**2)  # Expr with more than one variables
        >>> tf2 = TransferFunction.from_rational_expression(expr2, p)
        >>> tf2
        TransferFunction(a*p**3 - a*p**2 + p*s, a**2 + p, p)

        In case of conflict between two or more variables in a expression, SymPy will
        raise a ``ValueError``, if ``var`` is not passed by the user.

        >>> tf = TransferFunction.from_rational_expression((a + a*s)/(s**2 + s + 1))
        Traceback (most recent call last):
        ...
        ValueError: Conflicting values found for positional argument `var` ({a, s}). Specify it manually.

        This can be corrected by specifying the ``var`` parameter manually.

        >>> tf = TransferFunction.from_rational_expression((a + a*s)/(s**2 + s + 1), s)
        >>> tf
        TransferFunction(a*s + a, s**2 + s + 1, s)

        ``var`` also need to be specified when ``expr`` is a ``Number``

        >>> tf3 = TransferFunction.from_rational_expression(10, s)
        >>> tf3
        TransferFunction(10, 1, s)

        """
        expr = _sympify(expr)
        if var is None:
            _free_symbols = expr.free_symbols
            _len_free_symbols = len(_free_symbols)
            if _len_free_symbols == 1:
                var = list(_free_symbols)[0]
            elif _len_free_symbols == 0:
                raise ValueError(filldedent("""
                    Positional argument `var` not found in the
                    TransferFunction defined. Specify it manually."""))
            else:
                raise ValueError(filldedent("""
                    Conflicting values found for positional argument `var` ({}).
                    Specify it manually.""".format(_free_symbols)))

        _num, _den = expr.as_numer_denom()
        if _den == 0 or _num.has(S.ComplexInfinity):
            raise ZeroDivisionError("TransferFunction cannot have a zero denominator.")
        return cls(_num, _den, var)

    @classmethod
    def from_coeff_lists(cls, num_list, den_list, var):
        r"""
        Creates a new ``TransferFunction`` efficiently from a list of coefficients.

        Parameters
        ==========

        num_list : Sequence
            Sequence comprising of numerator coefficients.
        den_list : Sequence
            Sequence comprising of denominator coefficients.
        var : Symbol
            Complex variable of the Laplace transform used by the
            polynomials of the transfer function.

        Raises
        ======

        ZeroDivisionError
            When the constructed denominator is zero.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction
        >>> num = [1, 0, 2]
        >>> den = [3, 2, 2, 1]
        >>> tf = TransferFunction.from_coeff_lists(num, den, s)
        >>> tf
        TransferFunction(s**2 + 2, 3*s**3 + 2*s**2 + 2*s + 1, s)
        >>> #Create a Transfer Function with more than one variable
        >>> tf1 = TransferFunction.from_coeff_lists([p, 1], [2*p, 0, 4], s)
        >>> tf1
        TransferFunction(p*s + 1, 2*p*s**2 + 4, s)

        """
        num_list = num_list[::-1]
        den_list = den_list[::-1]
        num_var_powers = [var**i for i in range(len(num_list))]
        den_var_powers = [var**i for i in range(len(den_list))]

        _num = sum(coeff * var_power for coeff, var_power in zip(num_list, num_var_powers))
        _den = sum(coeff * var_power for coeff, var_power in zip(den_list, den_var_powers))

        if _den == 0:
            raise ZeroDivisionError("TransferFunction cannot have a zero denominator.")

        return cls(_num, _den, var)

    @classmethod
    def from_zpk(cls, zeros, poles, gain, var):
        r"""
        Creates a new ``TransferFunction`` from given zeros, poles and gain.

        Parameters
        ==========

        zeros : Sequence
            Sequence comprising of zeros of transfer function.
        poles : Sequence
            Sequence comprising of poles of transfer function.
        gain : Number, Symbol, Expression
            A scalar value specifying gain of the model.
        var : Symbol
            Complex variable of the Laplace transform used by the
            polynomials of the transfer function.

        Examples
        ========

        >>> from sympy.abc import s, p, k
        >>> from sympy.physics.control.lti import TransferFunction
        >>> zeros = [1, 2, 3]
        >>> poles = [6, 5, 4]
        >>> gain = 7
        >>> tf = TransferFunction.from_zpk(zeros, poles, gain, s)
        >>> tf
        TransferFunction(7*(s - 3)*(s - 2)*(s - 1), (s - 6)*(s - 5)*(s - 4), s)
        >>> #Create a Transfer Function with variable poles and zeros
        >>> tf1 = TransferFunction.from_zpk([p, k], [p + k, p - k], 2, s)
        >>> tf1
        TransferFunction(2*(-k + s)*(-p + s), (-k - p + s)*(k - p + s), s)
        >>> #Complex poles or zeros are acceptable
        >>> tf2 = TransferFunction.from_zpk([0], [1-1j, 1+1j, 2], -2, s)
        >>> tf2
        TransferFunction(-2*s, (s - 2)*(s - 1.0 - 1.0*I)*(s - 1.0 + 1.0*I), s)

        """
        num_poly = 1
        den_poly = 1
        for zero in zeros:
            num_poly *= var - zero
        for pole in poles:
            den_poly *= var - pole

        return cls(gain*num_poly, den_poly, var)

    @property
    def num(self):
        """
        Returns the numerator polynomial of the transfer function.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction
        >>> G1 = TransferFunction(s**2 + p*s + 3, s - 4, s)
        >>> G1.num
        p*s + s**2 + 3
        >>> G2 = TransferFunction((p + 5)*(p - 3), (p - 3)*(p + 1), p)
        >>> G2.num
        (p - 3)*(p + 5)

        """
        return self.args[0]

    @property
    def den(self):
        """
        Returns the denominator polynomial of the transfer function.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction
        >>> G1 = TransferFunction(s + 4, p**3 - 2*p + 4, s)
        >>> G1.den
        p**3 - 2*p + 4
        >>> G2 = TransferFunction(3, 4, s)
        >>> G2.den
        4

        """
        return self.args[1]

    @property
    def var(self):
        """
        Returns the complex variable of the Laplace transform used by the polynomials of
        the transfer function.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G1.var
        p
        >>> G2 = TransferFunction(0, s - 5, s)
        >>> G2.var
        s

        """
        return self.args[2]

    def _eval_subs(self, old, new):
        arg_num = self.num.subs(old, new)
        arg_den = self.den.subs(old, new)
        argnew = TransferFunction(arg_num, arg_den, self.var)
        return self if old == self.var else argnew

    def _eval_evalf(self, prec):
        return TransferFunction(
            self.num._eval_evalf(prec),
            self.den._eval_evalf(prec),
            self.var)

    def _eval_simplify(self, **kwargs):
        tf = cancel(Mul(self.num, 1/self.den, evaluate=False), expand=False).as_numer_denom()
        num_, den_ = tf[0], tf[1]
        return TransferFunction(num_, den_, self.var)

    def _eval_rewrite_as_StateSpace(self, *args):
        """
        Returns the equivalent space model of the transfer function model.
        The state space model will be returned in the controllable canonical form.

        Unlike the space state to transfer function model conversion, the transfer function
        to state space model conversion is not unique. There can be multiple state space
        representations of a given transfer function model.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control import TransferFunction, StateSpace
        >>> tf = TransferFunction(s**2 + 1, s**3 + 2*s + 10, s)
        >>> tf.rewrite(StateSpace)
        StateSpace(Matrix([
        [  0,  1, 0],
        [  0,  0, 1],
        [-10, -2, 0]]), Matrix([
        [0],
        [0],
        [1]]), Matrix([[1, 0, 1]]), Matrix([[0]]))

        """
        if not self.is_proper:
            raise ValueError("Transfer Function must be proper.")

        num_poly = Poly(self.num, self.var)
        den_poly = Poly(self.den, self.var)
        n = den_poly.degree()

        num_coeffs = num_poly.all_coeffs()
        den_coeffs = den_poly.all_coeffs()
        diff = n - num_poly.degree()
        num_coeffs = [0]*diff + num_coeffs

        a = den_coeffs[1:]
        a_mat = Matrix([[(-1)*coefficient/den_coeffs[0] for coefficient in reversed(a)]])
        vert = zeros(n-1, 1)
        mat = eye(n-1)
        A = vert.row_join(mat)
        A = A.col_join(a_mat)

        B = zeros(n, 1)
        B[n-1] = 1

        i = n
        C = []
        while(i > 0):
            C.append(num_coeffs[i] - den_coeffs[i]*num_coeffs[0])
            i -= 1
        C = Matrix([C])

        D = Matrix([num_coeffs[0]])

        return StateSpace(A, B, C, D)

    def expand(self):
        """
        Returns the transfer function with numerator and denominator
        in expanded form.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> G1 = TransferFunction((a - s)**2, (s**2 + a)**2, s)
        >>> G1.expand()
        TransferFunction(a**2 - 2*a*s + s**2, a**2 + 2*a*s**2 + s**4, s)
        >>> G2 = TransferFunction((p + 3*b)*(p - b), (p - b)*(p + 2*b), p)
        >>> G2.expand()
        TransferFunction(-3*b**2 + 2*b*p + p**2, -2*b**2 + b*p + p**2, p)

        """
        return TransferFunction(expand(self.num), expand(self.den), self.var)

    def dc_gain(self):
        """
        Computes the gain of the response as the frequency approaches zero.

        The DC gain is infinite for systems with pure integrators.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction(s + 3, s**2 - 9, s)
        >>> tf1.dc_gain()
        -1/3
        >>> tf2 = TransferFunction(p**2, p - 3 + p**3, p)
        >>> tf2.dc_gain()
        0
        >>> tf3 = TransferFunction(a*p**2 - b, s + b, s)
        >>> tf3.dc_gain()
        (a*p**2 - b)/b
        >>> tf4 = TransferFunction(1, s, s)
        >>> tf4.dc_gain()
        oo

        """
        m = Mul(self.num, Pow(self.den, -1, evaluate=False), evaluate=False)
        return limit(m, self.var, 0)

    def poles(self):
        """
        Returns the poles of a transfer function.

        Examples
        ========

        >>> from sympy.abc import s, p, a
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)
        >>> tf1.poles()
        [-5, 1]
        >>> tf2 = TransferFunction((1 - s)**2, (s**2 + 1)**2, s)
        >>> tf2.poles()
        [I, I, -I, -I]
        >>> tf3 = TransferFunction(s**2, a*s + p, s)
        >>> tf3.poles()
        [-p/a]

        """
        return _roots(Poly(self.den, self.var), self.var)

    def zeros(self):
        """
        Returns the zeros of a transfer function.

        Examples
        ========

        >>> from sympy.abc import s, p, a
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)
        >>> tf1.zeros()
        [-3, 1]
        >>> tf2 = TransferFunction((1 - s)**2, (s**2 + 1)**2, s)
        >>> tf2.zeros()
        [1, 1]
        >>> tf3 = TransferFunction(s**2, a*s + p, s)
        >>> tf3.zeros()
        [0, 0]

        """
        return _roots(Poly(self.num, self.var), self.var)

    def eval_frequency(self, other):
        """
        Returns the system response at any point in the real or complex plane.

        Examples
        ========

        >>> from sympy.abc import s, p, a
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy import I
        >>> tf1 = TransferFunction(1, s**2 + 2*s + 1, s)
        >>> omega = 0.1
        >>> tf1.eval_frequency(I*omega)
        1/(0.99 + 0.2*I)
        >>> tf2 = TransferFunction(s**2, a*s + p, s)
        >>> tf2.eval_frequency(2)
        4/(2*a + p)
        >>> tf2.eval_frequency(I*2)
        -4/(2*I*a + p)
        """
        arg_num = self.num.subs(self.var, other)
        arg_den = self.den.subs(self.var, other)
        argnew = TransferFunction(arg_num, arg_den, self.var).to_expr()
        return argnew.expand()

    def is_stable(self):
        """
        Returns True if the transfer function is asymptotically stable; else False.

        This would not check the marginal or conditional stability of the system.

        Examples
        ========

        >>> from sympy.abc import s, p, a
        >>> from sympy import symbols
        >>> from sympy.physics.control.lti import TransferFunction
        >>> q, r = symbols('q, r', negative=True)
        >>> tf1 = TransferFunction((1 - s)**2, (s + 1)**2, s)
        >>> tf1.is_stable()
        True
        >>> tf2 = TransferFunction((1 - p)**2, (s**2 + 1)**2, s)
        >>> tf2.is_stable()
        False
        >>> tf3 = TransferFunction(4, q*s - r, s)
        >>> tf3.is_stable()
        False
        >>> tf4 = TransferFunction(p + 1, a*p - s**2, p)
        >>> tf4.is_stable() is None   # Not enough info about the symbols to determine stability
        True

        """
        return fuzzy_and(pole.as_real_imag()[0].is_negative for pole in self.poles())

    def __add__(self, other):
        if hasattr(other, "is_StateSpace_object") and other.is_StateSpace_object:
            return Parallel(self, other)
        elif isinstance(other, (TransferFunction, Series, Feedback)):
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            return Parallel(self, other)
        elif isinstance(other, Parallel):
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            arg_list = list(other.args)
            return Parallel(self, *arg_list)
        else:
            raise ValueError("TransferFunction cannot be added with {}.".
                format(type(other)))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if hasattr(other, "is_StateSpace_object") and other.is_StateSpace_object:
            return Parallel(self, -other)
        elif isinstance(other, (TransferFunction, Series)):
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            return Parallel(self, -other)
        elif isinstance(other, Parallel):
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            arg_list = [-i for i in list(other.args)]
            return Parallel(self, *arg_list)
        else:
            raise ValueError("{} cannot be subtracted from a TransferFunction."
                .format(type(other)))

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if hasattr(other, "is_StateSpace_object") and other.is_StateSpace_object:
            return Series(self, other)
        elif isinstance(other, (TransferFunction, Parallel, Feedback)):
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            return Series(self, other)
        elif isinstance(other, Series):
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            arg_list = list(other.args)
            return Series(self, *arg_list)
        else:
            raise ValueError("TransferFunction cannot be multiplied with {}."
                .format(type(other)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, TransferFunction):
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            return Series(self, TransferFunction(other.den, other.num, self.var))
        elif (isinstance(other, Parallel) and len(other.args
                ) == 2 and isinstance(other.args[0], TransferFunction)
            and isinstance(other.args[1], (Series, TransferFunction))):

            if not self.var == other.var:
                raise ValueError(filldedent("""
                    Both TransferFunction and Parallel should use the
                    same complex variable of the Laplace transform."""))
            if other.args[1] == self:
                # plant and controller with unit feedback.
                return Feedback(self, other.args[0])
            other_arg_list = list(other.args[1].args) if isinstance(
                other.args[1], Series) else other.args[1]
            if other_arg_list == other.args[1]:
                return Feedback(self, other_arg_list)
            elif self in other_arg_list:
                other_arg_list.remove(self)
            else:
                return Feedback(self, Series(*other_arg_list))

            if len(other_arg_list) == 1:
                return Feedback(self, *other_arg_list)
            else:
                return Feedback(self, Series(*other_arg_list))
        else:
            raise ValueError("TransferFunction cannot be divided by {}.".
                format(type(other)))

    __rtruediv__ = __truediv__

    def __pow__(self, p):
        p = sympify(p)
        if not p.is_Integer:
            raise ValueError("Exponent must be an integer.")
        if p is S.Zero:
            return TransferFunction(1, 1, self.var)
        elif p > 0:
            num_, den_ = self.num**p, self.den**p
        else:
            p = abs(p)
            num_, den_ = self.den**p, self.num**p

        return TransferFunction(num_, den_, self.var)

    def __neg__(self):
        return TransferFunction(-self.num, self.den, self.var)

    @property
    def is_proper(self):
        """
        Returns True if degree of the numerator polynomial is less than
        or equal to degree of the denominator polynomial, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
        >>> tf1.is_proper
        False
        >>> tf2 = TransferFunction(p**2 - 4*p, p**3 + 3*p + 2, p)
        >>> tf2.is_proper
        True

        """
        return degree(self.num, self.var) <= degree(self.den, self.var)

    @property
    def is_strictly_proper(self):
        """
        Returns True if degree of the numerator polynomial is strictly less
        than degree of the denominator polynomial, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf1.is_strictly_proper
        False
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> tf2.is_strictly_proper
        True

        """
        return degree(self.num, self.var) < degree(self.den, self.var)

    @property
    def is_biproper(self):
        """
        Returns True if degree of the numerator polynomial is equal to
        degree of the denominator polynomial, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf1.is_biproper
        True
        >>> tf2 = TransferFunction(p**2, p + a, p)
        >>> tf2.is_biproper
        False

        """
        return degree(self.num, self.var) == degree(self.den, self.var)

    def to_expr(self):
        """
        Converts a ``TransferFunction`` object to SymPy Expr.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy import Expr
        >>> tf1 = TransferFunction(s, a*s**2 + 1, s)
        >>> tf1.to_expr()
        s/(a*s**2 + 1)
        >>> isinstance(_, Expr)
        True
        >>> tf2 = TransferFunction(1, (p + 3*b)*(b - p), p)
        >>> tf2.to_expr()
        1/((b - p)*(3*b + p))
        >>> tf3 = TransferFunction((s - 2)*(s - 3), (s - 1)*(s - 2)*(s - 3), s)
        >>> tf3.to_expr()
        ((s - 3)*(s - 2))/(((s - 3)*(s - 2)*(s - 1)))

        """

        if self.num != 1:
            return Mul(self.num, Pow(self.den, -1, evaluate=False), evaluate=False)
        else:
            return Pow(self.den, -1, evaluate=False)


class PIDController(TransferFunction):
    r"""
    A class for representing PID (Proportional-Integral-Derivative)
    controllers in control systems. The PIDController class is a subclass
    of TransferFunction, representing the controller's transfer function
    in the Laplace domain. The arguments are ``kp``, ``ki``, ``kd``,
    ``tf``, and ``var``, where ``kp``, ``ki``, and ``kd`` are the
    proportional, integral, and derivative gains respectively.``tf``
    is the derivative filter time constant, which can be used to
    filter out the noise and ``var`` is the complex variable used in
    the transfer function.

    Parameters
    ==========

    kp : Expr, Number
        Proportional gain. Defaults to ``Symbol('kp')`` if not specified.
    ki : Expr, Number
        Integral gain. Defaults to ``Symbol('ki')`` if not specified.
    kd : Expr, Number
        Derivative gain. Defaults to ``Symbol('kd')`` if not specified.
    tf : Expr, Number
        Derivative filter time constant.  Defaults to ``0`` if not specified.
    var : Symbol
        The complex frequency variable.  Defaults to ``s`` if not specified.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.control.lti import PIDController
    >>> kp, ki, kd = symbols('kp ki kd')
    >>> p1 = PIDController(kp, ki, kd)
    >>> print(p1)
    PIDController(kp, ki, kd, 0, s)
    >>> p1.doit()
    TransferFunction(kd*s**2 + ki + kp*s, s, s)
    >>> p1.kp
    kp
    >>> p1.ki
    ki
    >>> p1.kd
    kd
    >>> p1.tf
    0
    >>> p1.var
    s
    >>> p1.to_expr()
    (kd*s**2 + ki + kp*s)/s

    See Also
    ========

    TransferFunction

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/PID_controller
    .. [2] https://in.mathworks.com/help/control/ug/proportional-integral-derivative-pid-controllers.html

    """
    def __new__(cls, kp=Symbol('kp'), ki=Symbol('ki'), kd=Symbol('kd'), tf=0, var=Symbol('s')):
        kp, ki, kd, tf = _sympify(kp), _sympify(ki), _sympify(kd), _sympify(tf)
        num = kp*tf*var**2 + kp*var + ki*tf*var + ki + kd*var**2
        den = tf*var**2 + var
        obj = TransferFunction.__new__(cls, num, den, var)
        obj._kp, obj._ki, obj._kd, obj._tf = kp, ki, kd, tf
        return obj

    def __repr__(self):
        return f"PIDController({self.kp}, {self.ki}, {self.kd}, {self.tf}, {self.var})"

    __str__ = __repr__

    @property
    def kp(self):
        """
        Returns the Proportional gain (kp) of the PIDController.
        """
        return self._kp

    @property
    def ki(self):
        """
        Returns the Integral gain (ki) of the PIDController.
        """
        return self._ki

    @property
    def kd(self):
        """
        Returns the Derivative gain (kd) of the PIDController.
        """
        return self._kd

    @property
    def tf(self):
        """
        Returns the Derivative filter time constant (tf) of the PIDController.
        """
        return self._tf

    def doit(self):
        """
        Convert the PIDController into TransferFunction.
        """
        return TransferFunction(self.num, self.den, self.var)


def _flatten_args(args, _cls):
    temp_args = []
    for arg in args:
        if isinstance(arg, _cls):
            temp_args.extend(arg.args)
        else:
            temp_args.append(arg)
    return tuple(temp_args)


def _dummify_args(_arg, var):
    dummy_dict = {}
    dummy_arg_list = []

    for arg in _arg:
        _s = Dummy()
        dummy_dict[_s] = var
        dummy_arg = arg.subs({var: _s})
        dummy_arg_list.append(dummy_arg)

    return dummy_arg_list, dummy_dict


class Series(SISOLinearTimeInvariant):
    r"""
    A class for representing a series configuration of SISO systems.

    Parameters
    ==========

    args : SISOLinearTimeInvariant
        SISO systems in a series configuration.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``Series(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed, SISO in this case.

    Examples
    ========

    >>> from sympy.abc import s, p, a, b
    >>> from sympy import Matrix
    >>> from sympy.physics.control.lti import TransferFunction, Series, Parallel, StateSpace
    >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
    >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
    >>> tf3 = TransferFunction(p**2, p + s, s)
    >>> S1 = Series(tf1, tf2)
    >>> S1
    Series(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s))
    >>> S1.var
    s
    >>> S2 = Series(tf2, Parallel(tf3, -tf1))
    >>> S2
    Series(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), Parallel(TransferFunction(p**2, p + s, s), TransferFunction(-a*p**2 - b*s, -p + s, s)))
    >>> S2.var
    s
    >>> S3 = Series(Parallel(tf1, tf2), Parallel(tf2, tf3))
    >>> S3
    Series(Parallel(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)), Parallel(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), TransferFunction(p**2, p + s, s)))
    >>> S3.var
    s

    You can get the resultant transfer function by using ``.doit()`` method:

    >>> S3 = Series(tf1, tf2, -tf3)
    >>> S3.doit()
    TransferFunction(-p**2*(s**3 - 2)*(a*p**2 + b*s), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)
    >>> S4 = Series(tf2, Parallel(tf1, -tf3))
    >>> S4.doit()
    TransferFunction((s**3 - 2)*(-p**2*(-p + s) + (p + s)*(a*p**2 + b*s)), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)

    You can also connect StateSpace which results in SISO

    >>> A1 = Matrix([[-1]])
    >>> B1 = Matrix([[1]])
    >>> C1 = Matrix([[-1]])
    >>> D1 = Matrix([1])
    >>> A2 = Matrix([[0]])
    >>> B2 = Matrix([[1]])
    >>> C2 = Matrix([[1]])
    >>> D2 = Matrix([[0]])
    >>> ss1 = StateSpace(A1, B1, C1, D1)
    >>> ss2 = StateSpace(A2, B2, C2, D2)
    >>> S5 = Series(ss1, ss2)
    >>> S5
    Series(StateSpace(Matrix([[-1]]), Matrix([[1]]), Matrix([[-1]]), Matrix([[1]])), StateSpace(Matrix([[0]]), Matrix([[1]]), Matrix([[1]]), Matrix([[0]])))
    >>> S5.doit()
    StateSpace(Matrix([
    [-1,  0],
    [-1, 0]]), Matrix([
    [1],
    [1]]), Matrix([[0, 1]]), Matrix([[0]]))

    Notes
    =====

    All the transfer functions should use the same complex variable
    ``var`` of the Laplace transform.

    See Also
    ========

    MIMOSeries, Parallel, TransferFunction, Feedback

    """
    def __new__(cls, *args, evaluate=False):

        args = _flatten_args(args, Series)
        # For StateSpace series connection
        if args and any(isinstance(arg, StateSpace) or (hasattr(arg, 'is_StateSpace_object')
                                            and arg.is_StateSpace_object)for arg in args):
            # Check for SISO
            if (args[0].num_inputs == 1) and (args[-1].num_outputs == 1):
                # Check the interconnection
                for i in range(1, len(args)):
                    if args[i].num_inputs != args[i-1].num_outputs:
                        raise ValueError(filldedent("""Systems with incompatible inputs and outputs
                            cannot be connected in Series."""))
                cls._is_series_StateSpace = True
            else:
                raise ValueError("To use Series connection for MIMO systems use MIMOSeries instead.")
        else:
            cls._is_series_StateSpace = False
            cls._check_args(args)

        obj = super().__new__(cls, *args)

        return obj.doit() if evaluate else obj

    def __repr__(self):
        systems_repr = ', '.join(repr(system) for system in self.args)
        return f"Series({systems_repr})"

    __str__ = __repr__

    @property
    def var(self):
        """
        Returns the complex variable used by all the transfer functions.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, Series, Parallel
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> Series(G1, G2).var
        p
        >>> Series(-G3, Parallel(G1, G2)).var
        p

        """
        return self.args[0].var

    def doit(self, **hints):
        """
        Returns the resultant transfer function or StateSpace obtained after evaluating
        the series interconnection.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Series
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> Series(tf2, tf1).doit()
        TransferFunction((s**3 - 2)*(a*p**2 + b*s), (-p + s)*(s**4 + 5*s + 6), s)
        >>> Series(-tf1, -tf2).doit()
        TransferFunction((2 - s**3)*(-a*p**2 - b*s), (-p + s)*(s**4 + 5*s + 6), s)

        Notes
        =====

        If a series connection contains only TransferFunction components, the equivalent system returned
        will be a TransferFunction. However, if a StateSpace object is used in any of the arguments,
        the output will be a StateSpace object.

        """
        # Check if the system is a StateSpace
        if self._is_series_StateSpace:
            # Return the equivalent StateSpace model
            res = self.args[0]
            if not isinstance(res, StateSpace):
                res = res.doit().rewrite(StateSpace)
            for arg in self.args[1:]:
                if not isinstance(arg, StateSpace):
                    arg = arg.doit().rewrite(StateSpace)
                else:
                    arg = arg.doit()
                arg = arg.doit()
                res = arg * res
                return res

        _num_arg = (arg.doit().num for arg in self.args)
        _den_arg = (arg.doit().den for arg in self.args)
        res_num = Mul(*_num_arg, evaluate=True)
        res_den = Mul(*_den_arg, evaluate=True)
        return TransferFunction(res_num, res_den, self.var)

    def _eval_rewrite_as_TransferFunction(self, *args, **kwargs):
        if self._is_series_StateSpace:
            return self.doit().rewrite(TransferFunction)[0][0]
        return self.doit()

    @_check_other_SISO
    def __add__(self, other):

        if isinstance(other, Parallel):
            arg_list = list(other.args)
            return Parallel(self, *arg_list)

        return Parallel(self, other)

    __radd__ = __add__

    @_check_other_SISO
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    @_check_other_SISO
    def __mul__(self, other):

        arg_list = list(self.args)
        return Series(*arg_list, other)

    def __truediv__(self, other):
        if isinstance(other, TransferFunction):
            return Series(*self.args, TransferFunction(other.den, other.num, other.var))
        elif isinstance(other, Series):
            tf_self = self.rewrite(TransferFunction)
            tf_other = other.rewrite(TransferFunction)
            return tf_self / tf_other
        elif (isinstance(other, Parallel) and len(other.args) == 2
            and isinstance(other.args[0], TransferFunction) and isinstance(other.args[1], Series)):

            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            self_arg_list = set(self.args)
            other_arg_list = set(other.args[1].args)
            res = list(self_arg_list ^ other_arg_list)
            if len(res) == 0:
                return Feedback(self, other.args[0])
            elif len(res) == 1:
                return Feedback(self, *res)
            else:
                return Feedback(self, Series(*res))
        else:
            raise ValueError("This transfer function expression is invalid.")

    def __neg__(self):
        return Series(TransferFunction(-1, 1, self.var), self)

    def to_expr(self):
        """Returns the equivalent ``Expr`` object."""
        return Mul(*(arg.to_expr() for arg in self.args), evaluate=False)

    @property
    def is_proper(self):
        """
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is less than or equal to degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Series
        >>> tf1 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
        >>> tf2 = TransferFunction(p**2 - 4*p, p**3 + 3*s + 2, s)
        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)
        >>> S1 = Series(-tf2, tf1)
        >>> S1.is_proper
        False
        >>> S2 = Series(tf1, tf2, tf3)
        >>> S2.is_proper
        True

        """
        return self.doit().is_proper

    @property
    def is_strictly_proper(self):
        """
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is strictly less than degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Series
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**2 + 5*s + 6, s)
        >>> tf3 = TransferFunction(1, s**2 + s + 1, s)
        >>> S1 = Series(tf1, tf2)
        >>> S1.is_strictly_proper
        False
        >>> S2 = Series(tf1, tf2, tf3)
        >>> S2.is_strictly_proper
        True

        """
        return self.doit().is_strictly_proper

    @property
    def is_biproper(self):
        r"""
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is equal to degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Series
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(p, s**2, s)
        >>> tf3 = TransferFunction(s**2, 1, s)
        >>> S1 = Series(tf1, -tf2)
        >>> S1.is_biproper
        False
        >>> S2 = Series(tf2, tf3)
        >>> S2.is_biproper
        True

        """
        return self.doit().is_biproper

    @property
    def is_StateSpace_object(self):
        return self._is_series_StateSpace

def _mat_mul_compatible(*args):
    """To check whether shapes are compatible for matrix mul."""
    return all(args[i].num_outputs == args[i+1].num_inputs for i in range(len(args)-1))


class MIMOSeries(MIMOLinearTimeInvariant):
    r"""
    A class for representing a series configuration of MIMO systems.

    Parameters
    ==========

    args : MIMOLinearTimeInvariant
        MIMO systems in a series configuration.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``MIMOSeries(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.

        ``num_outputs`` of the MIMO system is not equal to the
        ``num_inputs`` of its adjacent MIMO system. (Matrix
        multiplication constraint, basically)
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed, MIMO in this case.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import MIMOSeries, TransferFunctionMatrix, StateSpace
    >>> from sympy import Matrix, pprint
    >>> mat_a = Matrix([[5*s], [5]])  # 2 Outputs 1 Input
    >>> mat_b = Matrix([[5, 1/(6*s**2)]])  # 1 Output 2 Inputs
    >>> mat_c = Matrix([[1, s], [5/s, 1]])  # 2 Outputs 2 Inputs
    >>> tfm_a = TransferFunctionMatrix.from_Matrix(mat_a, s)
    >>> tfm_b = TransferFunctionMatrix.from_Matrix(mat_b, s)
    >>> tfm_c = TransferFunctionMatrix.from_Matrix(mat_c, s)
    >>> MIMOSeries(tfm_c, tfm_b, tfm_a)
    MIMOSeries(TransferFunctionMatrix(((TransferFunction(1, 1, s), TransferFunction(s, 1, s)), (TransferFunction(5, s, s), TransferFunction(1, 1, s)))), TransferFunctionMatrix(((TransferFunction(5, 1, s), TransferFunction(1, 6*s**2, s)),)), TransferFunctionMatrix(((TransferFunction(5*s, 1, s),), (TransferFunction(5, 1, s),))))
    >>> pprint(_, use_unicode=False)  #  For Better Visualization
    [5*s]                 [1  s]
    [---]    [5   1  ]    [-  -]
    [ 1 ]    [-  ----]    [1  1]
    [   ]   *[1     2]   *[    ]
    [ 5 ]    [   6*s ]{t} [5  1]
    [ - ]                 [-  -]
    [ 1 ]{t}              [s  1]{t}
    >>> MIMOSeries(tfm_c, tfm_b, tfm_a).doit()
    TransferFunctionMatrix(((TransferFunction(150*s**4 + 25*s, 6*s**3, s), TransferFunction(150*s**4 + 5*s, 6*s**2, s)), (TransferFunction(150*s**3 + 25, 6*s**3, s), TransferFunction(150*s**3 + 5, 6*s**2, s))))
    >>> pprint(_, use_unicode=False)  # (2 Inputs -A-> 2 Outputs) -> (2 Inputs -B-> 1 Output) -> (1 Input -C-> 2 Outputs) is equivalent to (2 Inputs -Series Equivalent-> 2 Outputs).
    [     4              4      ]
    [150*s  + 25*s  150*s  + 5*s]
    [-------------  ------------]
    [        3             2    ]
    [     6*s           6*s     ]
    [                           ]
    [      3              3     ]
    [ 150*s  + 25    150*s  + 5 ]
    [ -----------    ---------- ]
    [        3             2    ]
    [     6*s           6*s     ]{t}
    >>> a1 = Matrix([[4, 1], [2, -3]])
    >>> b1 = Matrix([[5, 2], [-3, -3]])
    >>> c1 = Matrix([[2, -4], [0, 1]])
    >>> d1 = Matrix([[3, 2], [1, -1]])
    >>> a2 = Matrix([[-3, 4, 2], [-1, -3, 0], [2, 5, 3]])
    >>> b2 = Matrix([[1, 4], [-3, -3], [-2, 1]])
    >>> c2 = Matrix([[4, 2, -3], [1, 4, 3]])
    >>> d2 = Matrix([[-2, 4], [0, 1]])
    >>> ss1 = StateSpace(a1, b1, c1, d1) #2 inputs, 2 outputs
    >>> ss2 = StateSpace(a2, b2, c2, d2) #2 inputs, 2 outputs
    >>> S1 = MIMOSeries(ss1, ss2) #(2 inputs, 2 outputs) -> (2 inputs, 2 outputs)
    >>> S1
    MIMOSeries(StateSpace(Matrix([
            [4,  1],
            [2, -3]]), Matrix([
            [ 5,  2],
            [-3, -3]]), Matrix([
            [2, -4],
            [0,  1]]), Matrix([
            [3,  2],
            [1, -1]])), StateSpace(Matrix([
            [-3,  4, 2],
            [-1, -3, 0],
            [ 2,  5, 3]]), Matrix([
            [ 1,  4],
            [-3, -3],
            [-2,  1]]), Matrix([
            [4, 2, -3],
            [1, 4,  3]]), Matrix([
            [-2, 4],
            [ 0, 1]])))
    >>> S1.doit()
    StateSpace(Matrix([
    [ 4,  1,  0,  0, 0],
    [ 2, -3,  0,  0, 0],
    [ 2,  0, -3,  4, 2],
    [-6,  9, -1, -3, 0],
    [-4,  9,  2,  5, 3]]), Matrix([
    [  5,  2],
    [ -3, -3],
    [  7, -2],
    [-12, -3],
    [ -5, -5]]), Matrix([
    [-4, 12, 4, 2, -3],
    [ 0,  1, 1, 4,  3]]), Matrix([
    [-2, -8],
    [ 1, -1]]))

    Notes
    =====

    All the transfer function matrices should use the same complex variable ``var`` of the Laplace transform.

    ``MIMOSeries(A, B)`` is not equivalent to ``A*B``. It is always in the reverse order, that is ``B*A``.

    See Also
    ========

    Series, MIMOParallel

    """
    def __new__(cls, *args, evaluate=False):

        if args and any(isinstance(arg, StateSpace) or (hasattr(arg, 'is_StateSpace_object')
                                            and arg.is_StateSpace_object) for arg in args):
            # Check compatibility
            for i in range(1, len(args)):
                if args[i].num_inputs != args[i - 1].num_outputs:
                    raise ValueError(filldedent("""Systems with incompatible inputs and outputs
                        cannot be connected in MIMOSeries."""))
            obj = super().__new__(cls, *args)
            cls._is_series_StateSpace = True
        else:
            cls._check_args(args)
            cls._is_series_StateSpace = False

            if _mat_mul_compatible(*args):
                obj = super().__new__(cls, *args)

            else:
                raise ValueError(filldedent("""
                    Number of input signals do not match the number
                    of output signals of adjacent systems for some args."""))

        return obj.doit() if evaluate else obj

    @property
    def var(self):
        """
        Returns the complex variable used by all the transfer functions.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, MIMOSeries, TransferFunctionMatrix
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> tfm_1 = TransferFunctionMatrix([[G1, G2, G3]])
        >>> tfm_2 = TransferFunctionMatrix([[G1], [G2], [G3]])
        >>> MIMOSeries(tfm_2, tfm_1).var
        p

        """
        return self.args[0].var

    @property
    def num_inputs(self):
        """Returns the number of input signals of the series system."""
        return self.args[0].num_inputs

    @property
    def num_outputs(self):
        """Returns the number of output signals of the series system."""
        return self.args[-1].num_outputs

    @property
    def shape(self):
        """Returns the shape of the equivalent MIMO system."""
        return self.num_outputs, self.num_inputs

    @property
    def is_StateSpace_object(self):
        return self._is_series_StateSpace

    def doit(self, cancel=False, **kwargs):
        """
        Returns the resultant obtained after evaluating the MIMO systems arranged
        in a series configuration. For TransferFunction systems it returns a TransferFunctionMatrix
        and for StateSpace systems it returns the resultant StateSpace system.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, MIMOSeries, TransferFunctionMatrix
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> tfm1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf2]])
        >>> tfm2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf1]])
        >>> MIMOSeries(tfm2, tfm1).doit()
        TransferFunctionMatrix(((TransferFunction(2*(-p + s)*(s**3 - 2)*(a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)**2*(s**4 + 5*s + 6)**2, s), TransferFunction((-p + s)**2*(s**3 - 2)*(a*p**2 + b*s) + (-p + s)*(a*p**2 + b*s)**2*(s**4 + 5*s + 6), (-p + s)**3*(s**4 + 5*s + 6), s)), (TransferFunction((-p + s)*(s**3 - 2)**2*(s**4 + 5*s + 6) + (s**3 - 2)*(a*p**2 + b*s)*(s**4 + 5*s + 6)**2, (-p + s)*(s**4 + 5*s + 6)**3, s), TransferFunction(2*(s**3 - 2)*(a*p**2 + b*s), (-p + s)*(s**4 + 5*s + 6), s))))

        """
        if self._is_series_StateSpace:
            # Return the equivalent StateSpace model
            res = self.args[0]
            if not isinstance(res, StateSpace):
                res = res.doit().rewrite(StateSpace)
            for arg in self.args[1:]:
                if not isinstance(arg, StateSpace):
                    arg = arg.doit().rewrite(StateSpace)
                else:
                    arg = arg.doit()
                res = arg * res
            return res

        _arg = (arg.doit()._expr_mat for arg in reversed(self.args))

        if cancel:
            res = MatMul(*_arg, evaluate=True)
            return TransferFunctionMatrix.from_Matrix(res, self.var)

        _dummy_args, _dummy_dict = _dummify_args(_arg, self.var)
        res = MatMul(*_dummy_args, evaluate=True)
        temp_tfm = TransferFunctionMatrix.from_Matrix(res, self.var)
        return temp_tfm.subs(_dummy_dict)

    def _eval_rewrite_as_TransferFunctionMatrix(self, *args, **kwargs):
        if self._is_series_StateSpace:
            return self.doit().rewrite(TransferFunction)
        return self.doit()

    @_check_other_MIMO
    def __add__(self, other):

        if isinstance(other, MIMOParallel):
            arg_list = list(other.args)
            return MIMOParallel(self, *arg_list)

        return MIMOParallel(self, other)

    __radd__ = __add__

    @_check_other_MIMO
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    @_check_other_MIMO
    def __mul__(self, other):

        if isinstance(other, MIMOSeries):
            self_arg_list = list(self.args)
            other_arg_list = list(other.args)
            return MIMOSeries(*other_arg_list, *self_arg_list)  # A*B = MIMOSeries(B, A)

        arg_list = list(self.args)
        return MIMOSeries(other, *arg_list)

    def __neg__(self):
        arg_list = list(self.args)
        arg_list[0] = -arg_list[0]
        return MIMOSeries(*arg_list)


class Parallel(SISOLinearTimeInvariant):
    r"""
    A class for representing a parallel configuration of SISO systems.

    Parameters
    ==========

    args : SISOLinearTimeInvariant
        SISO systems in a parallel arrangement.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``Parallel(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import s, p, a, b
    >>> from sympy.physics.control.lti import TransferFunction, Parallel, Series, StateSpace
    >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
    >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
    >>> tf3 = TransferFunction(p**2, p + s, s)
    >>> P1 = Parallel(tf1, tf2)
    >>> P1
    Parallel(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s))
    >>> P1.var
    s
    >>> P2 = Parallel(tf2, Series(tf3, -tf1))
    >>> P2
    Parallel(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), Series(TransferFunction(p**2, p + s, s), TransferFunction(-a*p**2 - b*s, -p + s, s)))
    >>> P2.var
    s
    >>> P3 = Parallel(Series(tf1, tf2), Series(tf2, tf3))
    >>> P3
    Parallel(Series(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)), Series(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), TransferFunction(p**2, p + s, s)))
    >>> P3.var
    s

    You can get the resultant transfer function by using ``.doit()`` method:

    >>> Parallel(tf1, tf2, -tf3).doit()
    TransferFunction(-p**2*(-p + s)*(s**4 + 5*s + 6) + (-p + s)*(p + s)*(s**3 - 2) + (p + s)*(a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)
    >>> Parallel(tf2, Series(tf1, -tf3)).doit()
    TransferFunction(-p**2*(a*p**2 + b*s)*(s**4 + 5*s + 6) + (-p + s)*(p + s)*(s**3 - 2), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)

    Parallel can be used to connect SISO ``StateSpace`` systems together.

    >>> A1 = Matrix([[-1]])
    >>> B1 = Matrix([[1]])
    >>> C1 = Matrix([[-1]])
    >>> D1 = Matrix([1])
    >>> A2 = Matrix([[0]])
    >>> B2 = Matrix([[1]])
    >>> C2 = Matrix([[1]])
    >>> D2 = Matrix([[0]])
    >>> ss1 = StateSpace(A1, B1, C1, D1)
    >>> ss2 = StateSpace(A2, B2, C2, D2)
    >>> P4 = Parallel(ss1, ss2)
    >>> P4
    Parallel(StateSpace(Matrix([[-1]]), Matrix([[1]]), Matrix([[-1]]), Matrix([[1]])), StateSpace(Matrix([[0]]), Matrix([[1]]), Matrix([[1]]), Matrix([[0]])))

    ``doit()`` can be used to find ``StateSpace`` equivalent for the system containing ``StateSpace`` objects.

    >>> P4.doit()
    StateSpace(Matrix([
    [-1, 0],
    [ 0, 0]]), Matrix([
    [1],
    [1]]), Matrix([[-1, 1]]), Matrix([[1]]))
    >>> P4.rewrite(TransferFunction)
    TransferFunction(s*(s + 1) + 1, s*(s + 1), s)

    Notes
    =====

    All the transfer functions should use the same complex variable
    ``var`` of the Laplace transform.

    See Also
    ========

    Series, TransferFunction, Feedback

    """

    def __new__(cls, *args, evaluate=False):

        args = _flatten_args(args, Parallel)
        # For StateSpace parallel connection
        if args and any(isinstance(arg, StateSpace) or (hasattr(arg, 'is_StateSpace_object')
                                                        and arg.is_StateSpace_object) for arg in args):
            # Check for SISO
            if all(arg.is_SISO for arg in args):
                cls._is_parallel_StateSpace = True
            else:
                raise ValueError("To use Parallel connection for MIMO systems use MIMOParallel instead.")
        else:
            cls._is_parallel_StateSpace = False
            cls._check_args(args)
        obj = super().__new__(cls, *args)

        return obj.doit() if evaluate else obj

    def __repr__(self):
        systems_repr = ', '.join(repr(system) for system in self.args)
        return f"Parallel({systems_repr})"

    __str__ = __repr__

    @property
    def var(self):
        """
        Returns the complex variable used by all the transfer functions.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, Parallel, Series
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> Parallel(G1, G2).var
        p
        >>> Parallel(-G3, Series(G1, G2)).var
        p

        """
        return self.args[0].var

    def doit(self, **hints):
        """
        Returns the resultant transfer function or state space obtained by
        parallel connection of transfer functions or state space objects.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Parallel
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> Parallel(tf2, tf1).doit()
        TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s)
        >>> Parallel(-tf1, -tf2).doit()
        TransferFunction((2 - s**3)*(-p + s) + (-a*p**2 - b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s)

        """
        if self._is_parallel_StateSpace:
            # Return the equivalent StateSpace model
            res = self.args[0].doit()
            if not isinstance(res, StateSpace):
                res = res.rewrite(StateSpace)
            for arg in self.args[1:]:
                if not isinstance(arg, StateSpace):
                    arg = arg.doit().rewrite(StateSpace)
                res += arg
            return res

        _arg = (arg.doit().to_expr() for arg in self.args)
        res = Add(*_arg).as_numer_denom()
        return TransferFunction(*res, self.var)

    def _eval_rewrite_as_TransferFunction(self, *args, **kwargs):
        if self._is_parallel_StateSpace:
            return self.doit().rewrite(TransferFunction)[0][0]
        return self.doit()

    @_check_other_SISO
    def __add__(self, other):

        self_arg_list = list(self.args)
        return Parallel(*self_arg_list, other)

    __radd__ = __add__

    @_check_other_SISO
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    @_check_other_SISO
    def __mul__(self, other):

        if isinstance(other, Series):
            arg_list = list(other.args)
            return Series(self, *arg_list)

        return Series(self, other)

    def __neg__(self):
        return Series(TransferFunction(-1, 1, self.var), self)

    def to_expr(self):
        """Returns the equivalent ``Expr`` object."""
        return Add(*(arg.to_expr() for arg in self.args), evaluate=False)

    @property
    def is_proper(self):
        """
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is less than or equal to degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Parallel
        >>> tf1 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
        >>> tf2 = TransferFunction(p**2 - 4*p, p**3 + 3*s + 2, s)
        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)
        >>> P1 = Parallel(-tf2, tf1)
        >>> P1.is_proper
        False
        >>> P2 = Parallel(tf2, tf3)
        >>> P2.is_proper
        True

        """
        return self.doit().is_proper

    @property
    def is_strictly_proper(self):
        """
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is strictly less than degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Parallel
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)
        >>> P1 = Parallel(tf1, tf2)
        >>> P1.is_strictly_proper
        False
        >>> P2 = Parallel(tf2, tf3)
        >>> P2.is_strictly_proper
        True

        """
        return self.doit().is_strictly_proper

    @property
    def is_biproper(self):
        """
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is equal to degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Parallel
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(p**2, p + s, s)
        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)
        >>> P1 = Parallel(tf1, -tf2)
        >>> P1.is_biproper
        True
        >>> P2 = Parallel(tf2, tf3)
        >>> P2.is_biproper
        False

        """
        return self.doit().is_biproper

    @property
    def is_StateSpace_object(self):
        return self._is_parallel_StateSpace


class MIMOParallel(MIMOLinearTimeInvariant):
    r"""
    A class for representing a parallel configuration of MIMO systems.

    Parameters
    ==========

    args : MIMOLinearTimeInvariant
        MIMO Systems in a parallel arrangement.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``MIMOParallel(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.

        All MIMO systems passed do not have same shape.
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed, MIMO in this case.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunctionMatrix, MIMOParallel, StateSpace
    >>> from sympy import Matrix, pprint
    >>> expr_1 = 1/s
    >>> expr_2 = s/(s**2-1)
    >>> expr_3 = (2 + s)/(s**2 - 1)
    >>> expr_4 = 5
    >>> tfm_a = TransferFunctionMatrix.from_Matrix(Matrix([[expr_1, expr_2], [expr_3, expr_4]]), s)
    >>> tfm_b = TransferFunctionMatrix.from_Matrix(Matrix([[expr_2, expr_1], [expr_4, expr_3]]), s)
    >>> tfm_c = TransferFunctionMatrix.from_Matrix(Matrix([[expr_3, expr_4], [expr_1, expr_2]]), s)
    >>> MIMOParallel(tfm_a, tfm_b, tfm_c)
    MIMOParallel(TransferFunctionMatrix(((TransferFunction(1, s, s), TransferFunction(s, s**2 - 1, s)), (TransferFunction(s + 2, s**2 - 1, s), TransferFunction(5, 1, s)))), TransferFunctionMatrix(((TransferFunction(s, s**2 - 1, s), TransferFunction(1, s, s)), (TransferFunction(5, 1, s), TransferFunction(s + 2, s**2 - 1, s)))), TransferFunctionMatrix(((TransferFunction(s + 2, s**2 - 1, s), TransferFunction(5, 1, s)), (TransferFunction(1, s, s), TransferFunction(s, s**2 - 1, s)))))
    >>> pprint(_, use_unicode=False)  #  For Better Visualization
    [  1       s   ]      [  s       1   ]      [s + 2     5   ]
    [  -     ------]      [------    -   ]      [------    -   ]
    [  s      2    ]      [ 2        s   ]      [ 2        1   ]
    [        s  - 1]      [s  - 1        ]      [s  - 1        ]
    [              ]    + [              ]    + [              ]
    [s + 2     5   ]      [  5     s + 2 ]      [  1       s   ]
    [------    -   ]      [  -     ------]      [  -     ------]
    [ 2        1   ]      [  1      2    ]      [  s      2    ]
    [s  - 1        ]{t}   [        s  - 1]{t}   [        s  - 1]{t}
    >>> MIMOParallel(tfm_a, tfm_b, tfm_c).doit()
    TransferFunctionMatrix(((TransferFunction(s**2 + s*(2*s + 2) - 1, s*(s**2 - 1), s), TransferFunction(2*s**2 + 5*s*(s**2 - 1) - 1, s*(s**2 - 1), s)), (TransferFunction(s**2 + s*(s + 2) + 5*s*(s**2 - 1) - 1, s*(s**2 - 1), s), TransferFunction(5*s**2 + 2*s - 3, s**2 - 1, s))))
    >>> pprint(_, use_unicode=False)
    [       2                              2       / 2    \    ]
    [      s  + s*(2*s + 2) - 1         2*s  + 5*s*\s  - 1/ - 1]
    [      --------------------         -----------------------]
    [             / 2    \                       / 2    \      ]
    [           s*\s  - 1/                     s*\s  - 1/      ]
    [                                                          ]
    [ 2                   / 2    \             2               ]
    [s  + s*(s + 2) + 5*s*\s  - 1/ - 1      5*s  + 2*s - 3     ]
    [---------------------------------      --------------     ]
    [              / 2    \                      2             ]
    [            s*\s  - 1/                     s  - 1         ]{t}

    ``MIMOParallel`` can also be used to connect MIMO ``StateSpace`` systems.

    >>> A1 = Matrix([[4, 1], [2, -3]])
    >>> B1 = Matrix([[5, 2], [-3, -3]])
    >>> C1 = Matrix([[2, -4], [0, 1]])
    >>> D1 = Matrix([[3, 2], [1, -1]])
    >>> A2 = Matrix([[-3, 4, 2], [-1, -3, 0], [2, 5, 3]])
    >>> B2 = Matrix([[1, 4], [-3, -3], [-2, 1]])
    >>> C2 = Matrix([[4, 2, -3], [1, 4, 3]])
    >>> D2 = Matrix([[-2, 4], [0, 1]])
    >>> ss1 = StateSpace(A1, B1, C1, D1)
    >>> ss2 = StateSpace(A2, B2, C2, D2)
    >>> p1 = MIMOParallel(ss1, ss2)
    >>> p1
    MIMOParallel(StateSpace(Matrix([
    [4,  1],
    [2, -3]]), Matrix([
    [ 5,  2],
    [-3, -3]]), Matrix([
    [2, -4],
    [0,  1]]), Matrix([
    [3,  2],
    [1, -1]])), StateSpace(Matrix([
    [-3,  4, 2],
    [-1, -3, 0],
    [ 2,  5, 3]]), Matrix([
    [ 1,  4],
    [-3, -3],
    [-2,  1]]), Matrix([
    [4, 2, -3],
    [1, 4,  3]]), Matrix([
    [-2, 4],
    [ 0, 1]])))

    ``doit()`` can be used to find ``StateSpace`` equivalent for the system containing ``StateSpace`` objects.

    >>> p1.doit()
    StateSpace(Matrix([
    [4,  1,  0,  0, 0],
    [2, -3,  0,  0, 0],
    [0,  0, -3,  4, 2],
    [0,  0, -1, -3, 0],
    [0,  0,  2,  5, 3]]), Matrix([
    [ 5,  2],
    [-3, -3],
    [ 1,  4],
    [-3, -3],
    [-2,  1]]), Matrix([
    [2, -4, 4, 2, -3],
    [0,  1, 1, 4,  3]]), Matrix([
    [1, 6],
    [1, 0]]))

    Notes
    =====

    All the transfer function matrices should use the same complex variable
    ``var`` of the Laplace transform.

    See Also
    ========

    Parallel, MIMOSeries

    """

    def __new__(cls, *args, evaluate=False):

        args = _flatten_args(args, MIMOParallel)

        # For StateSpace Parallel connection
        if args and any(isinstance(arg, StateSpace) or (hasattr(arg, 'is_StateSpace_object')
                                    and arg.is_StateSpace_object) for arg in args):
            if any(arg.num_inputs != args[0].num_inputs or arg.num_outputs != args[0].num_outputs
                   for arg in args[1:]):
                raise ShapeError("Systems with incompatible inputs and outputs cannot be "
                                 "connected in MIMOParallel.")
            cls._is_parallel_StateSpace = True
        else:
            cls._check_args(args)
            if any(arg.shape != args[0].shape for arg in args):
                raise TypeError("Shape of all the args is not equal.")
            cls._is_parallel_StateSpace = False
        obj = super().__new__(cls, *args)

        return obj.doit() if evaluate else obj

    @property
    def var(self):
        """
        Returns the complex variable used by all the systems.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOParallel
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> G4 = TransferFunction(p**2, p**2 - 1, p)
        >>> tfm_a = TransferFunctionMatrix([[G1, G2], [G3, G4]])
        >>> tfm_b = TransferFunctionMatrix([[G2, G1], [G4, G3]])
        >>> MIMOParallel(tfm_a, tfm_b).var
        p

        """
        return self.args[0].var

    @property
    def num_inputs(self):
        """Returns the number of input signals of the parallel system."""
        return self.args[0].num_inputs

    @property
    def num_outputs(self):
        """Returns the number of output signals of the parallel system."""
        return self.args[0].num_outputs

    @property
    def shape(self):
        """Returns the shape of the equivalent MIMO system."""
        return self.num_outputs, self.num_inputs

    @property
    def is_StateSpace_object(self):
        return self._is_parallel_StateSpace

    def doit(self, **hints):
        """
        Returns the resultant transfer function matrix or StateSpace obtained after evaluating
        the MIMO systems arranged in a parallel configuration.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, MIMOParallel, TransferFunctionMatrix
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
        >>> MIMOParallel(tfm_1, tfm_2).doit()
        TransferFunctionMatrix(((TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s), TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s)), (TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s), TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s))))

        """
        if self._is_parallel_StateSpace:
            # Return the equivalent StateSpace model.
            res = self.args[0]
            if not isinstance(res, StateSpace):
                res = res.doit().rewrite(StateSpace)
            for arg in self.args[1:]:
                if not isinstance(arg, StateSpace):
                    arg = arg.doit().rewrite(StateSpace)
                else:
                    arg = arg.doit()
                res += arg
            return res
        _arg = (arg.doit()._expr_mat for arg in self.args)
        res = MatAdd(*_arg, evaluate=True)
        return TransferFunctionMatrix.from_Matrix(res, self.var)

    def _eval_rewrite_as_TransferFunctionMatrix(self, *args, **kwargs):
        if self._is_parallel_StateSpace:
            return self.doit().rewrite(TransferFunction)
        return self.doit()

    @_check_other_MIMO
    def __add__(self, other):

        self_arg_list = list(self.args)
        return MIMOParallel(*self_arg_list, other)

    __radd__ = __add__

    @_check_other_MIMO
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    @_check_other_MIMO
    def __mul__(self, other):

        if isinstance(other, MIMOSeries):
            arg_list = list(other.args)
            return MIMOSeries(*arg_list, self)

        return MIMOSeries(other, self)

    def __neg__(self):
        arg_list = [-arg for arg in list(self.args)]
        return MIMOParallel(*arg_list)


class Feedback(SISOLinearTimeInvariant):
    r"""
    A class for representing closed-loop feedback interconnection between two
    SISO input/output systems.

    The first argument, ``sys1``, is the feedforward part of the closed-loop
    system or in simple words, the dynamical model representing the process
    to be controlled. The second argument, ``sys2``, is the feedback system
    and controls the fed back signal to ``sys1``. Both ``sys1`` and ``sys2``
    can either be ``Series``, ``StateSpace`` or ``TransferFunction`` objects.

    Parameters
    ==========

    sys1 : Series, StateSpace, TransferFunction
        The feedforward path system.
    sys2 : Series, StateSpace, TransferFunction, optional
        The feedback path system (often a feedback controller).
        It is the model sitting on the feedback path.

        If not specified explicitly, the sys2 is
        assumed to be unit (1.0) transfer function.
    sign : int, optional
        The sign of feedback. Can either be ``1``
        (for positive feedback) or ``-1`` (for negative feedback).
        Default value is `-1`.

    Raises
    ======

    ValueError
        When ``sys1`` and ``sys2`` are not using the
        same complex variable of the Laplace transform.

        When a combination of ``sys1`` and ``sys2`` yields
        zero denominator.

    TypeError
        When either ``sys1`` or ``sys2`` is not a ``Series``, ``StateSpace`` or
        ``TransferFunction`` object.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import StateSpace, TransferFunction, Feedback
    >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
    >>> controller = TransferFunction(5*s - 10, s + 7, s)
    >>> F1 = Feedback(plant, controller)
    >>> F1
    Feedback(TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s), TransferFunction(5*s - 10, s + 7, s), -1)
    >>> F1.var
    s
    >>> F1.args
    (TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s), TransferFunction(5*s - 10, s + 7, s), -1)

    You can get the feedforward and feedback path systems by using ``.sys1`` and ``.sys2`` respectively.

    >>> F1.sys1
    TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
    >>> F1.sys2
    TransferFunction(5*s - 10, s + 7, s)

    You can get the resultant closed loop transfer function obtained by negative feedback
    interconnection using ``.doit()`` method.

    >>> F1.doit()
    TransferFunction((s + 7)*(s**2 - 4*s + 2)*(3*s**2 + 7*s - 3), ((s + 7)*(s**2 - 4*s + 2) + (5*s - 10)*(3*s**2 + 7*s - 3))*(s**2 - 4*s + 2), s)
    >>> G = TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s)
    >>> C = TransferFunction(5*s + 10, s + 10, s)
    >>> F2 = Feedback(G*C, TransferFunction(1, 1, s))
    >>> F2.doit()
    TransferFunction((s + 10)*(5*s + 10)*(s**2 + 2*s + 3)*(2*s**2 + 5*s + 1), (s + 10)*((s + 10)*(s**2 + 2*s + 3) + (5*s + 10)*(2*s**2 + 5*s + 1))*(s**2 + 2*s + 3), s)

    To negate a ``Feedback`` object, the ``-`` operator can be prepended:

    >>> -F1
    Feedback(TransferFunction(-3*s**2 - 7*s + 3, s**2 - 4*s + 2, s), TransferFunction(10 - 5*s, s + 7, s), -1)
    >>> -F2
    Feedback(Series(TransferFunction(-1, 1, s), TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s), TransferFunction(5*s + 10, s + 10, s)), TransferFunction(-1, 1, s), -1)

    ``Feedback`` can also be used to connect SISO ``StateSpace`` systems together.

    >>> A1 = Matrix([[-1]])
    >>> B1 = Matrix([[1]])
    >>> C1 = Matrix([[-1]])
    >>> D1 = Matrix([1])
    >>> A2 = Matrix([[0]])
    >>> B2 = Matrix([[1]])
    >>> C2 = Matrix([[1]])
    >>> D2 = Matrix([[0]])
    >>> ss1 = StateSpace(A1, B1, C1, D1)
    >>> ss2 = StateSpace(A2, B2, C2, D2)
    >>> F3 = Feedback(ss1, ss2)
    >>> F3
    Feedback(StateSpace(Matrix([[-1]]), Matrix([[1]]), Matrix([[-1]]), Matrix([[1]])), StateSpace(Matrix([[0]]), Matrix([[1]]), Matrix([[1]]), Matrix([[0]])), -1)

    ``doit()`` can be used to find ``StateSpace`` equivalent for the system containing ``StateSpace`` objects.

    >>> F3.doit()
    StateSpace(Matrix([
    [-1, -1],
    [-1, -1]]), Matrix([
    [1],
    [1]]), Matrix([[-1, -1]]), Matrix([[1]]))

    We can also find the equivalent ``TransferFunction`` by using ``rewrite(TransferFunction)`` method.

    >>> F3.rewrite(TransferFunction)
    TransferFunction(s, s + 2, s)

    See Also
    ========

    MIMOFeedback, Series, Parallel

    """
    def __new__(cls, sys1, sys2=None, sign=-1):
        if not sys2:
            sys2 = TransferFunction(1, 1, sys1.var)

        if not isinstance(sys1, (TransferFunction, Series, StateSpace, Feedback)):
            raise TypeError("Unsupported type for `sys1` in Feedback.")

        if not isinstance(sys2, (TransferFunction, Series, StateSpace, Feedback)):
            raise TypeError("Unsupported type for `sys2` in Feedback.")

        if not (sys1.num_inputs == sys1.num_outputs == sys2.num_inputs ==
                sys2.num_outputs == 1):
            raise ValueError("""To use Feedback connection for MIMO systems
                            use MIMOFeedback instead.""")

        if sign not in [-1, 1]:
            raise ValueError(filldedent("""
                Unsupported type for feedback. `sign` arg should
                either be 1 (positive feedback loop) or -1
                (negative feedback loop)."""))

        if sys1.is_StateSpace_object or sys2.is_StateSpace_object:
            cls.is_StateSpace_object = True
        else:
            if Mul(sys1.to_expr(), sys2.to_expr()).simplify() == sign:
                raise ValueError("The equivalent system will have zero denominator.")
            if sys1.var != sys2.var:
                raise ValueError(filldedent("""Both `sys1` and `sys2` should be using the
                same complex variable."""))
            cls.is_StateSpace_object = False

        return super(SISOLinearTimeInvariant, cls).__new__(cls, sys1, sys2, _sympify(sign))

    def __repr__(self):
        return f"Feedback({self.sys1}, {self.sys2}, {self.sign})"

    __str__ = __repr__

    @property
    def sys1(self):
        """
        Returns the feedforward system of the feedback interconnection.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
        >>> controller = TransferFunction(5*s - 10, s + 7, s)
        >>> F1 = Feedback(plant, controller)
        >>> F1.sys1
        TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
        >>> G = TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p)
        >>> C = TransferFunction(5*p + 10, p + 10, p)
        >>> P = TransferFunction(1 - s, p + 2, p)
        >>> F2 = Feedback(TransferFunction(1, 1, p), G*C*P)
        >>> F2.sys1
        TransferFunction(1, 1, p)

        """
        return self.args[0]

    @property
    def sys2(self):
        """
        Returns the feedback controller of the feedback interconnection.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
        >>> controller = TransferFunction(5*s - 10, s + 7, s)
        >>> F1 = Feedback(plant, controller)
        >>> F1.sys2
        TransferFunction(5*s - 10, s + 7, s)
        >>> G = TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p)
        >>> C = TransferFunction(5*p + 10, p + 10, p)
        >>> P = TransferFunction(1 - s, p + 2, p)
        >>> F2 = Feedback(TransferFunction(1, 1, p), G*C*P)
        >>> F2.sys2
        Series(TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p), TransferFunction(5*p + 10, p + 10, p), TransferFunction(1 - s, p + 2, p))

        """
        return self.args[1]

    @property
    def var(self):
        """
        Returns the complex variable of the Laplace transform used by all
        the transfer functions involved in the feedback interconnection.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
        >>> controller = TransferFunction(5*s - 10, s + 7, s)
        >>> F1 = Feedback(plant, controller)
        >>> F1.var
        s
        >>> G = TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p)
        >>> C = TransferFunction(5*p + 10, p + 10, p)
        >>> P = TransferFunction(1 - s, p + 2, p)
        >>> F2 = Feedback(TransferFunction(1, 1, p), G*C*P)
        >>> F2.var
        p

        """
        return self.sys1.var

    @property
    def sign(self):
        """
        Returns the type of MIMO Feedback model. ``1``
        for Positive and ``-1`` for Negative.
        """
        return self.args[2]

    @property
    def num(self):
        """
        Returns the numerator of the closed loop feedback system.
        """
        return self.sys1

    @property
    def den(self):
        """
        Returns the denominator of the closed loop feedback model.
        """
        unit = TransferFunction(1, 1, self.var)
        arg_list = list(self.sys1.args) if isinstance(self.sys1, Series) else [self.sys1]
        if self.sign == 1:
            return Parallel(unit, -Series(self.sys2, *arg_list))
        return Parallel(unit, Series(self.sys2, *arg_list))

    @property
    def sensitivity(self):
        """
        Returns the sensitivity function of the feedback loop.

        Sensitivity of a Feedback system is the ratio
        of change in the open loop gain to the change in
        the closed loop gain.

        .. note::
            This method would not return the complementary
            sensitivity function.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> C = TransferFunction(5*p + 10, p + 10, p)
        >>> P = TransferFunction(1 - p, p + 2, p)
        >>> F_1 = Feedback(P, C)
        >>> F_1.sensitivity
        1/((1 - p)*(5*p + 10)/((p + 2)*(p + 10)) + 1)

        """

        return 1/(1 - self.sign*self.sys1.to_expr()*self.sys2.to_expr())

    def doit(self, cancel=False, expand=False, **hints):
        """
        Returns the resultant transfer function or state space obtained by
        feedback connection of transfer functions or state space objects.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy import Matrix
        >>> from sympy.physics.control.lti import TransferFunction, Feedback, StateSpace
        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
        >>> controller = TransferFunction(5*s - 10, s + 7, s)
        >>> F1 = Feedback(plant, controller)
        >>> F1.doit()
        TransferFunction((s + 7)*(s**2 - 4*s + 2)*(3*s**2 + 7*s - 3), ((s + 7)*(s**2 - 4*s + 2) + (5*s - 10)*(3*s**2 + 7*s - 3))*(s**2 - 4*s + 2), s)
        >>> G = TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s)
        >>> F2 = Feedback(G, TransferFunction(1, 1, s))
        >>> F2.doit()
        TransferFunction((s**2 + 2*s + 3)*(2*s**2 + 5*s + 1), (s**2 + 2*s + 3)*(3*s**2 + 7*s + 4), s)

        Use kwarg ``expand=True`` to expand the resultant transfer function.
        Use ``cancel=True`` to cancel out the common terms in numerator and
        denominator.

        >>> F2.doit(cancel=True, expand=True)
        TransferFunction(2*s**2 + 5*s + 1, 3*s**2 + 7*s + 4, s)
        >>> F2.doit(expand=True)
        TransferFunction(2*s**4 + 9*s**3 + 17*s**2 + 17*s + 3, 3*s**4 + 13*s**3 + 27*s**2 + 29*s + 12, s)

        If the connection contain any ``StateSpace`` object then ``doit()``
        will return the equivalent ``StateSpace`` object.

        >>> A1 = Matrix([[-1.5, -2], [1, 0]])
        >>> B1 = Matrix([0.5, 0])
        >>> C1 = Matrix([[0, 1]])
        >>> A2 = Matrix([[0, 1], [-5, -2]])
        >>> B2 = Matrix([0, 3])
        >>> C2 = Matrix([[0, 1]])
        >>> ss1 = StateSpace(A1, B1, C1)
        >>> ss2 = StateSpace(A2, B2, C2)
        >>> F3 = Feedback(ss1, ss2)
        >>> F3.doit()
        StateSpace(Matrix([
        [-1.5, -2,  0, -0.5],
        [   1,  0,  0,    0],
        [   0,  0,  0,    1],
        [   0,  3, -5,   -2]]), Matrix([
        [0.5],
        [  0],
        [  0],
        [  0]]), Matrix([[0, 1, 0, 0]]), Matrix([[0]]))

        """
        if self.is_StateSpace_object:
            sys1_ss = self.sys1.doit().rewrite(StateSpace)
            sys2_ss = self.sys2.doit().rewrite(StateSpace)
            A1, B1, C1, D1 = sys1_ss.A, sys1_ss.B, sys1_ss.C, sys1_ss.D
            A2, B2, C2, D2 = sys2_ss.A, sys2_ss.B, sys2_ss.C, sys2_ss.D

            # Create identity matrices
            I_inputs = eye(self.num_inputs)
            I_outputs = eye(self.num_outputs)

            # Compute F and its inverse
            F = I_inputs - self.sign * D2 * D1
            E = F.inv()

            # Compute intermediate matrices
            E_D2 = E * D2
            E_C2 = E * C2
            T1 = I_outputs + self.sign * D1 * E_D2
            T2 = I_inputs + self.sign * E_D2 * D1
            A = Matrix.vstack(
            Matrix.hstack(A1 + self.sign * B1 * E_D2 * C1, self.sign * B1 * E_C2),
            Matrix.hstack(B2 * T1 * C1, A2 + self.sign * B2 * D1 * E_C2)
            )
            B = Matrix.vstack(B1 * T2, B2 * D1 * T2)
            C = Matrix.hstack(T1 * C1, self.sign * D1 * E_C2)
            D = D1 * T2
            return StateSpace(A, B, C, D)

        arg_list = list(self.sys1.args) if isinstance(self.sys1, Series) else [self.sys1]
        # F_n and F_d are resultant TFs of num and den of Feedback.
        F_n, unit = self.sys1.doit(), TransferFunction(1, 1, self.sys1.var)
        if self.sign == -1:
            F_d = Parallel(unit, Series(self.sys2, *arg_list)).doit()
        else:
            F_d = Parallel(unit, -Series(self.sys2, *arg_list)).doit()

        _resultant_tf = TransferFunction(F_n.num * F_d.den, F_n.den * F_d.num, F_n.var)

        if cancel:
            _resultant_tf = _resultant_tf.simplify()

        if expand:
            _resultant_tf = _resultant_tf.expand()

        return _resultant_tf

    def _eval_rewrite_as_TransferFunction(self, num, den, sign, **kwargs):
        if self.is_StateSpace_object:
            return self.doit().rewrite(TransferFunction)[0][0]
        return self.doit()

    def to_expr(self):
        """
        Converts a ``Feedback`` object to SymPy Expr.

        Examples
        ========

        >>> from sympy.abc import s, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> from sympy import Expr
        >>> tf1 = TransferFunction(a+s, 1, s)
        >>> tf2 = TransferFunction(b+s, 1, s)
        >>> fd1 = Feedback(tf1, tf2)
        >>> fd1.to_expr()
        (a + s)/((a + s)*(b + s) + 1)
        >>> isinstance(_, Expr)
        True
        """

        return self.doit().to_expr()

    def __neg__(self):
        return Feedback(-self.sys1, -self.sys2, self.sign)


def _is_invertible(a, b, sign):
    """
    Checks whether a given pair of MIMO
    systems passed is invertible or not.
    """
    _mat = eye(a.num_outputs) - sign*(a.doit()._expr_mat)*(b.doit()._expr_mat)
    _det = _mat.det()

    return _det != 0


class MIMOFeedback(MIMOLinearTimeInvariant):
    r"""
    A class for representing closed-loop feedback interconnection between two
    MIMO input/output systems.

    Parameters
    ==========

    sys1 : MIMOSeries, TransferFunctionMatrix, StateSpace
        The MIMO system placed on the feedforward path.
    sys2 : MIMOSeries, TransferFunctionMatrix, StateSpace
        The system placed on the feedback path
        (often a feedback controller).
    sign : int, optional
        The sign of feedback. Can either be ``1``
        (for positive feedback) or ``-1`` (for negative feedback).
        Default value is `-1`.

    Raises
    ======

    ValueError
        When ``sys1`` and ``sys2`` are not using the
        same complex variable of the Laplace transform.

        Forward path model should have an equal number of inputs/outputs
        to the feedback path outputs/inputs.

        When product of ``sys1`` and ``sys2`` is not a square matrix.

        When the equivalent MIMO system is not invertible.

    TypeError
        When either ``sys1`` or ``sys2`` is not a ``MIMOSeries``,
        ``TransferFunctionMatrix`` or a ``StateSpace`` object.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import StateSpace, TransferFunctionMatrix, MIMOFeedback
    >>> plant_mat = Matrix([[1, 1/s], [0, 1]])
    >>> controller_mat = Matrix([[10, 0], [0, 10]])  # Constant Gain
    >>> plant = TransferFunctionMatrix.from_Matrix(plant_mat, s)
    >>> controller = TransferFunctionMatrix.from_Matrix(controller_mat, s)
    >>> feedback = MIMOFeedback(plant, controller)  # Negative Feedback (default)
    >>> pprint(feedback, use_unicode=False)
    /    [1  1]    [10  0 ]   \-1   [1  1]
    |    [-  -]    [--  - ]   |     [-  -]
    |    [1  s]    [1   1 ]   |     [1  s]
    |I + [    ]   *[      ]   |   * [    ]
    |    [0  1]    [0   10]   |     [0  1]
    |    [-  -]    [-   --]   |     [-  -]
    \    [1  1]{t} [1   1 ]{t}/     [1  1]{t}

    To get the equivalent system matrix, use either ``doit`` or ``rewrite`` method.

    >>> pprint(feedback.doit(), use_unicode=False)
    [1     1  ]
    [--  -----]
    [11  121*s]
    [         ]
    [0    1   ]
    [-    --  ]
    [1    11  ]{t}

    To negate the ``MIMOFeedback`` object, use ``-`` operator.

    >>> neg_feedback = -feedback
    >>> pprint(neg_feedback.doit(), use_unicode=False)
    [-1    -1  ]
    [---  -----]
    [11   121*s]
    [          ]
    [ 0    -1  ]
    [ -    --- ]
    [ 1    11  ]{t}

    ``MIMOFeedback`` can also be used to connect MIMO ``StateSpace`` systems.

    >>> A1 = Matrix([[4, 1], [2, -3]])
    >>> B1 = Matrix([[5, 2], [-3, -3]])
    >>> C1 = Matrix([[2, -4], [0, 1]])
    >>> D1 = Matrix([[3, 2], [1, -1]])
    >>> A2 = Matrix([[-3, 4, 2], [-1, -3, 0], [2, 5, 3]])
    >>> B2 = Matrix([[1, 4], [-3, -3], [-2, 1]])
    >>> C2 = Matrix([[4, 2, -3], [1, 4, 3]])
    >>> D2 = Matrix([[-2, 4], [0, 1]])
    >>> ss1 = StateSpace(A1, B1, C1, D1)
    >>> ss2 = StateSpace(A2, B2, C2, D2)
    >>> F1 = MIMOFeedback(ss1, ss2)
    >>> F1
    MIMOFeedback(StateSpace(Matrix([
    [4,  1],
    [2, -3]]), Matrix([
    [ 5,  2],
    [-3, -3]]), Matrix([
    [2, -4],
    [0,  1]]), Matrix([
    [3,  2],
    [1, -1]])), StateSpace(Matrix([
    [-3,  4, 2],
    [-1, -3, 0],
    [ 2,  5, 3]]), Matrix([
    [ 1,  4],
    [-3, -3],
    [-2,  1]]), Matrix([
    [4, 2, -3],
    [1, 4,  3]]), Matrix([
    [-2, 4],
    [ 0, 1]])), -1)

    ``doit()`` can be used to find ``StateSpace`` equivalent for the system containing ``StateSpace`` objects.

    >>> F1.doit()
    StateSpace(Matrix([
    [   3,  -3/4, -15/4, -37/2, -15],
    [ 7/2, -39/8,   9/8,  39/4,   9],
    [   3, -41/4, -45/4, -51/2, -19],
    [-9/2, 129/8,  73/8, 171/4,  36],
    [-3/2,  47/8,  31/8,  85/4,  18]]), Matrix([
    [-1/4,  19/4],
    [ 3/8, -21/8],
    [ 1/4,  29/4],
    [ 3/8, -93/8],
    [ 5/8, -35/8]]), Matrix([
    [  1, -15/4,  -7/4, -21/2, -9],
    [1/2, -13/8, -13/8, -19/4, -3]]), Matrix([
    [-1/4, 11/4],
    [ 1/8,  9/8]]))

    See Also
    ========

    Feedback, MIMOSeries, MIMOParallel

    """
    def __new__(cls, sys1, sys2, sign=-1):
        if not isinstance(sys1, (TransferFunctionMatrix, MIMOSeries, StateSpace)):
            raise TypeError("Unsupported type for `sys1` in MIMO Feedback.")

        if not isinstance(sys2, (TransferFunctionMatrix, MIMOSeries, StateSpace)):
            raise TypeError("Unsupported type for `sys2` in MIMO Feedback.")

        if sys1.num_inputs != sys2.num_outputs or \
            sys1.num_outputs != sys2.num_inputs:
            raise ValueError(filldedent("""
                Product of `sys1` and `sys2` must
                yield a square matrix."""))

        if sign not in (-1, 1):
            raise ValueError(filldedent("""
                Unsupported type for feedback. `sign` arg should
                either be 1 (positive feedback loop) or -1
                (negative feedback loop)."""))

        if sys1.is_StateSpace_object or sys2.is_StateSpace_object:
            cls.is_StateSpace_object = True
        else:
            if not _is_invertible(sys1, sys2, sign):
                raise ValueError("Non-Invertible system inputted.")
            cls.is_StateSpace_object = False

        if not cls.is_StateSpace_object and sys1.var != sys2.var:
            raise ValueError(filldedent("""
                Both `sys1` and `sys2` should be using the
                same complex variable."""))

        return super().__new__(cls, sys1, sys2, _sympify(sign))

    @property
    def sys1(self):
        r"""
        Returns the system placed on the feedforward path of the MIMO feedback interconnection.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(s**2 + s + 1, s**2 - s + 1, s)
        >>> tf2 = TransferFunction(1, s, s)
        >>> tf3 = TransferFunction(1, 1, s)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf3, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)
        >>> F_1.sys1
        TransferFunctionMatrix(((TransferFunction(s**2 + s + 1, s**2 - s + 1, s), TransferFunction(1, s, s)), (TransferFunction(1, s, s), TransferFunction(s**2 + s + 1, s**2 - s + 1, s))))
        >>> pprint(_, use_unicode=False)
        [ 2                    ]
        [s  + s + 1      1     ]
        [----------      -     ]
        [ 2              s     ]
        [s  - s + 1            ]
        [                      ]
        [             2        ]
        [    1       s  + s + 1]
        [    -       ----------]
        [    s        2        ]
        [            s  - s + 1]{t}

        """
        return self.args[0]

    @property
    def sys2(self):
        r"""
        Returns the feedback controller of the MIMO feedback interconnection.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(s**2, s**3 - s + 1, s)
        >>> tf2 = TransferFunction(1, s, s)
        >>> tf3 = TransferFunction(1, 1, s)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2)
        >>> F_1.sys2
        TransferFunctionMatrix(((TransferFunction(s**2, s**3 - s + 1, s), TransferFunction(1, 1, s)), (TransferFunction(1, 1, s), TransferFunction(1, s, s))))
        >>> pprint(_, use_unicode=False)
        [     2       ]
        [    s       1]
        [----------  -]
        [ 3          1]
        [s  - s + 1   ]
        [             ]
        [    1       1]
        [    -       -]
        [    1       s]{t}

        """
        return self.args[1]

    @property
    def var(self):
        r"""
        Returns the complex variable of the Laplace transform used by all
        the transfer functions involved in the MIMO feedback loop.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(p, 1 - p, p)
        >>> tf2 = TransferFunction(1, p, p)
        >>> tf3 = TransferFunction(1, 1, p)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)  # Positive feedback
        >>> F_1.var
        p

        """
        return self.sys1.var

    @property
    def sign(self):
        r"""
        Returns the type of feedback interconnection of two models. ``1``
        for Positive and ``-1`` for Negative.
        """
        return self.args[2]

    @property
    def sensitivity(self):
        r"""
        Returns the sensitivity function matrix of the feedback loop.

        Sensitivity of a closed-loop system is the ratio of change
        in the open loop gain to the change in the closed loop gain.

        .. note::
            This method would not return the complementary
            sensitivity function.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(p, 1 - p, p)
        >>> tf2 = TransferFunction(1, p, p)
        >>> tf3 = TransferFunction(1, 1, p)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)  # Positive feedback
        >>> F_2 = MIMOFeedback(sys1, sys2)  # Negative feedback
        >>> pprint(F_1.sensitivity, use_unicode=False)
        [   4      3      2               5      4      2           ]
        [- p  + 3*p  - 4*p  + 3*p - 1    p  - 2*p  + 3*p  - 3*p + 1 ]
        [----------------------------  -----------------------------]
        [  4      3      2              5      4      3      2      ]
        [ p  + 3*p  - 8*p  + 8*p - 3   p  + 3*p  - 8*p  + 8*p  - 3*p]
        [                                                           ]
        [       4    3    2                  3      2               ]
        [      p  - p  - p  + p           3*p  - 6*p  + 4*p - 1     ]
        [ --------------------------    --------------------------  ]
        [  4      3      2               4      3      2            ]
        [ p  + 3*p  - 8*p  + 8*p - 3    p  + 3*p  - 8*p  + 8*p - 3  ]
        >>> pprint(F_2.sensitivity, use_unicode=False)
        [ 4      3      2           5      4      2          ]
        [p  - 3*p  + 2*p  + p - 1  p  - 2*p  + 3*p  - 3*p + 1]
        [------------------------  --------------------------]
        [   4      3                   5      4      2       ]
        [  p  - 3*p  + 2*p - 1        p  - 3*p  + 2*p  - p   ]
        [                                                    ]
        [     4    3    2               4      3             ]
        [    p  - p  - p  + p        2*p  - 3*p  + 2*p - 1   ]
        [  -------------------       ---------------------   ]
        [   4      3                   4      3              ]
        [  p  - 3*p  + 2*p - 1        p  - 3*p  + 2*p - 1    ]

        """
        _sys1_mat = self.sys1.doit()._expr_mat
        _sys2_mat = self.sys2.doit()._expr_mat

        return (eye(self.sys1.num_inputs) - \
            self.sign*_sys1_mat*_sys2_mat).inv()

    @property
    def num_inputs(self):
        """Returns the number of inputs of the system."""
        return self.sys1.num_inputs

    @property
    def num_outputs(self):
        """Returns the number of outputs of the system."""
        return self.sys1.num_outputs

    def doit(self, cancel=True, expand=False, **hints):
        r"""
        Returns the resultant transfer function matrix obtained by the
        feedback interconnection.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(s, 1 - s, s)
        >>> tf2 = TransferFunction(1, s, s)
        >>> tf3 = TransferFunction(5, 1, s)
        >>> tf4 = TransferFunction(s - 1, s, s)
        >>> tf5 = TransferFunction(0, 1, s)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
        >>> sys2 = TransferFunctionMatrix([[tf3, tf5], [tf5, tf5]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)
        >>> pprint(F_1, use_unicode=False)
        /    [  s      1  ]    [5  0]   \-1   [  s      1  ]
        |    [-----    -  ]    [-  -]   |     [-----    -  ]
        |    [1 - s    s  ]    [1  1]   |     [1 - s    s  ]
        |I - [            ]   *[    ]   |   * [            ]
        |    [  5    s - 1]    [0  0]   |     [  5    s - 1]
        |    [  -    -----]    [-  -]   |     [  -    -----]
        \    [  1      s  ]{t} [1  1]{t}/     [  1      s  ]{t}
        >>> pprint(F_1.doit(), use_unicode=False)
        [  -s           s - 1       ]
        [-------     -----------    ]
        [6*s - 1     s*(6*s - 1)    ]
        [                           ]
        [5*s - 5  (s - 1)*(6*s + 24)]
        [-------  ------------------]
        [6*s - 1     s*(6*s - 1)    ]{t}

        If the user wants the resultant ``TransferFunctionMatrix`` object without
        canceling the common factors then the ``cancel`` kwarg should be passed ``False``.

        >>> pprint(F_1.doit(cancel=False), use_unicode=False)
        [             s*(s - 1)                              s - 1               ]
        [         -----------------                       -----------            ]
        [         (1 - s)*(6*s - 1)                       s*(6*s - 1)            ]
        [                                                                        ]
        [s*(25*s - 25) + 5*(1 - s)*(6*s - 1)  s*(s - 1)*(6*s - 1) + s*(25*s - 25)]
        [-----------------------------------  -----------------------------------]
        [         (1 - s)*(6*s - 1)                        2                     ]
        [                                                 s *(6*s - 1)           ]{t}

        If the user wants the expanded form of the resultant transfer function matrix,
        the ``expand`` kwarg should be passed as ``True``.

        >>> pprint(F_1.doit(expand=True), use_unicode=False)
        [  -s          s - 1      ]
        [-------      --------    ]
        [6*s - 1         2        ]
        [             6*s  - s    ]
        [                         ]
        [            2            ]
        [5*s - 5  6*s  + 18*s - 24]
        [-------  ----------------]
        [6*s - 1         2        ]
        [             6*s  - s    ]{t}

        """
        if self.is_StateSpace_object:
            sys1_ss = self.sys1.doit().rewrite(StateSpace)
            sys2_ss = self.sys2.doit().rewrite(StateSpace)
            A1, B1, C1, D1 = sys1_ss.A, sys1_ss.B, sys1_ss.C, sys1_ss.D
            A2, B2, C2, D2 = sys2_ss.A, sys2_ss.B, sys2_ss.C, sys2_ss.D

            # Create identity matrices
            I_inputs = eye(self.num_inputs)
            I_outputs = eye(self.num_outputs)

            # Compute F and its inverse
            F = I_inputs - self.sign * D2 * D1
            E = F.inv()

            # Compute intermediate matrices
            E_D2 = E * D2
            E_C2 = E * C2
            T1 = I_outputs + self.sign * D1 * E_D2
            T2 = I_inputs + self.sign * E_D2 * D1
            A = Matrix.vstack(
            Matrix.hstack(A1 + self.sign * B1 * E_D2 * C1, self.sign * B1 * E_C2),
            Matrix.hstack(B2 * T1 * C1, A2 + self.sign * B2 * D1 * E_C2)
            )
            B = Matrix.vstack(B1 * T2, B2 * D1 * T2)
            C = Matrix.hstack(T1 * C1, self.sign * D1 * E_C2)
            D = D1 * T2
            return StateSpace(A, B, C, D)

        _mat = self.sensitivity * self.sys1.doit()._expr_mat

        _resultant_tfm = _to_TFM(_mat, self.var)

        if cancel:
            _resultant_tfm = _resultant_tfm.simplify()

        if expand:
            _resultant_tfm = _resultant_tfm.expand()

        return _resultant_tfm

    def _eval_rewrite_as_TransferFunctionMatrix(self, sys1, sys2, sign, **kwargs):
        return self.doit()

    def __neg__(self):
        return MIMOFeedback(-self.sys1, -self.sys2, self.sign)


def _to_TFM(mat, var):
    """Private method to convert ImmutableMatrix to TransferFunctionMatrix efficiently"""
    to_tf = lambda expr: TransferFunction.from_rational_expression(expr, var)
    arg = [[to_tf(expr) for expr in row] for row in mat.tolist()]
    return TransferFunctionMatrix(arg)


class TransferFunctionMatrix(MIMOLinearTimeInvariant):
    r"""
    A class for representing the MIMO (multiple-input and multiple-output)
    generalization of the SISO (single-input and single-output) transfer function.

    It is a matrix of transfer functions (``TransferFunction``, SISO-``Series`` or SISO-``Parallel``).
    There is only one argument, ``arg`` which is also the compulsory argument.
    ``arg`` is expected to be strictly of the type list of lists
    which holds the transfer functions or reducible to transfer functions.

    Parameters
    ==========

    arg : Nested ``List`` (strictly).
        Users are expected to input a nested list of ``TransferFunction``, ``Series``
        and/or ``Parallel`` objects.

    Examples
    ========

    .. note::
        ``pprint()`` can be used for better visualization of ``TransferFunctionMatrix`` objects.

    >>> from sympy.abc import s, p, a
    >>> from sympy import pprint
    >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, Series, Parallel
    >>> tf_1 = TransferFunction(s + a, s**2 + s + 1, s)
    >>> tf_2 = TransferFunction(p**4 - 3*p + 2, s + p, s)
    >>> tf_3 = TransferFunction(3, s + 2, s)
    >>> tf_4 = TransferFunction(-a + p, 9*s - 9, s)
    >>> tfm_1 = TransferFunctionMatrix([[tf_1], [tf_2], [tf_3]])
    >>> tfm_1
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(3, s + 2, s),)))
    >>> tfm_1.var
    s
    >>> tfm_1.num_inputs
    1
    >>> tfm_1.num_outputs
    3
    >>> tfm_1.shape
    (3, 1)
    >>> tfm_1.args
    (((TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(3, s + 2, s),)),)
    >>> tfm_2 = TransferFunctionMatrix([[tf_1, -tf_3], [tf_2, -tf_1], [tf_3, -tf_2]])
    >>> tfm_2
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s), TransferFunction(-3, s + 2, s)), (TransferFunction(p**4 - 3*p + 2, p + s, s), TransferFunction(-a - s, s**2 + s + 1, s)), (TransferFunction(3, s + 2, s), TransferFunction(-p**4 + 3*p - 2, p + s, s))))
    >>> pprint(tfm_2, use_unicode=False)  # pretty-printing for better visualization
    [   a + s           -3       ]
    [ ----------       -----     ]
    [  2               s + 2     ]
    [ s  + s + 1                 ]
    [                            ]
    [ 4                          ]
    [p  - 3*p + 2      -a - s    ]
    [------------    ----------  ]
    [   p + s         2          ]
    [                s  + s + 1  ]
    [                            ]
    [                 4          ]
    [     3        - p  + 3*p - 2]
    [   -----      --------------]
    [   s + 2          p + s     ]{t}

    TransferFunctionMatrix can be transposed, if user wants to switch the input and output transfer functions

    >>> tfm_2.transpose()
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s), TransferFunction(p**4 - 3*p + 2, p + s, s), TransferFunction(3, s + 2, s)), (TransferFunction(-3, s + 2, s), TransferFunction(-a - s, s**2 + s + 1, s), TransferFunction(-p**4 + 3*p - 2, p + s, s))))
    >>> pprint(_, use_unicode=False)
    [             4                          ]
    [  a + s     p  - 3*p + 2        3       ]
    [----------  ------------      -----     ]
    [ 2             p + s          s + 2     ]
    [s  + s + 1                              ]
    [                                        ]
    [                             4          ]
    [   -3          -a - s     - p  + 3*p - 2]
    [  -----      ----------   --------------]
    [  s + 2       2               p + s     ]
    [             s  + s + 1                 ]{t}

    >>> tf_5 = TransferFunction(5, s, s)
    >>> tf_6 = TransferFunction(5*s, (2 + s**2), s)
    >>> tf_7 = TransferFunction(5, (s*(2 + s**2)), s)
    >>> tf_8 = TransferFunction(5, 1, s)
    >>> tfm_3 = TransferFunctionMatrix([[tf_5, tf_6], [tf_7, tf_8]])
    >>> tfm_3
    TransferFunctionMatrix(((TransferFunction(5, s, s), TransferFunction(5*s, s**2 + 2, s)), (TransferFunction(5, s*(s**2 + 2), s), TransferFunction(5, 1, s))))
    >>> pprint(tfm_3, use_unicode=False)
    [    5        5*s  ]
    [    -       ------]
    [    s        2    ]
    [            s  + 2]
    [                  ]
    [    5         5   ]
    [----------    -   ]
    [  / 2    \    1   ]
    [s*\s  + 2/        ]{t}
    >>> tfm_3.var
    s
    >>> tfm_3.shape
    (2, 2)
    >>> tfm_3.num_outputs
    2
    >>> tfm_3.num_inputs
    2
    >>> tfm_3.args
    (((TransferFunction(5, s, s), TransferFunction(5*s, s**2 + 2, s)), (TransferFunction(5, s*(s**2 + 2), s), TransferFunction(5, 1, s))),)

    To access the ``TransferFunction`` at any index in the ``TransferFunctionMatrix``, use the index notation.

    >>> tfm_3[1, 0]  # gives the TransferFunction present at 2nd Row and 1st Col. Similar to that in Matrix classes
    TransferFunction(5, s*(s**2 + 2), s)
    >>> tfm_3[0, 0]  # gives the TransferFunction present at 1st Row and 1st Col.
    TransferFunction(5, s, s)
    >>> tfm_3[:, 0]  # gives the first column
    TransferFunctionMatrix(((TransferFunction(5, s, s),), (TransferFunction(5, s*(s**2 + 2), s),)))
    >>> pprint(_, use_unicode=False)
    [    5     ]
    [    -     ]
    [    s     ]
    [          ]
    [    5     ]
    [----------]
    [  / 2    \]
    [s*\s  + 2/]{t}
    >>> tfm_3[0, :]  # gives the first row
    TransferFunctionMatrix(((TransferFunction(5, s, s), TransferFunction(5*s, s**2 + 2, s)),))
    >>> pprint(_, use_unicode=False)
    [5   5*s  ]
    [-  ------]
    [s   2    ]
    [   s  + 2]{t}

    To negate a transfer function matrix, ``-`` operator can be prepended:

    >>> tfm_4 = TransferFunctionMatrix([[tf_2], [-tf_1], [tf_3]])
    >>> -tfm_4
    TransferFunctionMatrix(((TransferFunction(-p**4 + 3*p - 2, p + s, s),), (TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(-3, s + 2, s),)))
    >>> tfm_5 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, -tf_1]])
    >>> -tfm_5
    TransferFunctionMatrix(((TransferFunction(-a - s, s**2 + s + 1, s), TransferFunction(-p**4 + 3*p - 2, p + s, s)), (TransferFunction(-3, s + 2, s), TransferFunction(a + s, s**2 + s + 1, s))))

    ``subs()`` returns the ``TransferFunctionMatrix`` object with the value substituted in the expression. This will not
    mutate your original ``TransferFunctionMatrix``.

    >>> tfm_2.subs(p, 2)  #  substituting p everywhere in tfm_2 with 2.
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s), TransferFunction(-3, s + 2, s)), (TransferFunction(12, s + 2, s), TransferFunction(-a - s, s**2 + s + 1, s)), (TransferFunction(3, s + 2, s), TransferFunction(-12, s + 2, s))))
    >>> pprint(_, use_unicode=False)
    [  a + s        -3     ]
    [----------    -----   ]
    [ 2            s + 2   ]
    [s  + s + 1            ]
    [                      ]
    [    12        -a - s  ]
    [  -----     ----------]
    [  s + 2      2        ]
    [            s  + s + 1]
    [                      ]
    [    3          -12    ]
    [  -----       -----   ]
    [  s + 2       s + 2   ]{t}
    >>> pprint(tfm_2, use_unicode=False) # State of tfm_2 is unchanged after substitution
    [   a + s           -3       ]
    [ ----------       -----     ]
    [  2               s + 2     ]
    [ s  + s + 1                 ]
    [                            ]
    [ 4                          ]
    [p  - 3*p + 2      -a - s    ]
    [------------    ----------  ]
    [   p + s         2          ]
    [                s  + s + 1  ]
    [                            ]
    [                 4          ]
    [     3        - p  + 3*p - 2]
    [   -----      --------------]
    [   s + 2          p + s     ]{t}

    ``subs()`` also supports multiple substitutions.

    >>> tfm_2.subs({p: 2, a: 1})  # substituting p with 2 and a with 1
    TransferFunctionMatrix(((TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(-3, s + 2, s)), (TransferFunction(12, s + 2, s), TransferFunction(-s - 1, s**2 + s + 1, s)), (TransferFunction(3, s + 2, s), TransferFunction(-12, s + 2, s))))
    >>> pprint(_, use_unicode=False)
    [  s + 1        -3     ]
    [----------    -----   ]
    [ 2            s + 2   ]
    [s  + s + 1            ]
    [                      ]
    [    12        -s - 1  ]
    [  -----     ----------]
    [  s + 2      2        ]
    [            s  + s + 1]
    [                      ]
    [    3          -12    ]
    [  -----       -----   ]
    [  s + 2       s + 2   ]{t}

    Users can reduce the ``Series`` and ``Parallel`` elements of the matrix to ``TransferFunction`` by using
    ``doit()``.

    >>> tfm_6 = TransferFunctionMatrix([[Series(tf_3, tf_4), Parallel(tf_3, tf_4)]])
    >>> tfm_6
    TransferFunctionMatrix(((Series(TransferFunction(3, s + 2, s), TransferFunction(-a + p, 9*s - 9, s)), Parallel(TransferFunction(3, s + 2, s), TransferFunction(-a + p, 9*s - 9, s))),))
    >>> pprint(tfm_6, use_unicode=False)
    [-a + p    3    -a + p      3  ]
    [-------*-----  ------- + -----]
    [9*s - 9 s + 2  9*s - 9   s + 2]{t}
    >>> tfm_6.doit()
    TransferFunctionMatrix(((TransferFunction(-3*a + 3*p, (s + 2)*(9*s - 9), s), TransferFunction(27*s + (-a + p)*(s + 2) - 27, (s + 2)*(9*s - 9), s)),))
    >>> pprint(_, use_unicode=False)
    [    -3*a + 3*p     27*s + (-a + p)*(s + 2) - 27]
    [-----------------  ----------------------------]
    [(s + 2)*(9*s - 9)       (s + 2)*(9*s - 9)      ]{t}
    >>> tf_9 = TransferFunction(1, s, s)
    >>> tf_10 = TransferFunction(1, s**2, s)
    >>> tfm_7 = TransferFunctionMatrix([[Series(tf_9, tf_10), tf_9], [tf_10, Parallel(tf_9, tf_10)]])
    >>> tfm_7
    TransferFunctionMatrix(((Series(TransferFunction(1, s, s), TransferFunction(1, s**2, s)), TransferFunction(1, s, s)), (TransferFunction(1, s**2, s), Parallel(TransferFunction(1, s, s), TransferFunction(1, s**2, s)))))
    >>> pprint(tfm_7, use_unicode=False)
    [ 1      1   ]
    [----    -   ]
    [   2    s   ]
    [s*s         ]
    [            ]
    [ 1    1    1]
    [ --   -- + -]
    [  2    2   s]
    [ s    s     ]{t}
    >>> tfm_7.doit()
    TransferFunctionMatrix(((TransferFunction(1, s**3, s), TransferFunction(1, s, s)), (TransferFunction(1, s**2, s), TransferFunction(s**2 + s, s**3, s))))
    >>> pprint(_, use_unicode=False)
    [1     1   ]
    [--    -   ]
    [ 3    s   ]
    [s         ]
    [          ]
    [     2    ]
    [1   s  + s]
    [--  ------]
    [ 2     3  ]
    [s     s   ]{t}

    Addition, subtraction, and multiplication of transfer function matrices can form
    unevaluated ``Series`` or ``Parallel`` objects.

    - For addition and subtraction:
      All the transfer function matrices must have the same shape.

    - For multiplication (C = A * B):
      The number of inputs of the first transfer function matrix (A) must be equal to the
      number of outputs of the second transfer function matrix (B).

    Also, use pretty-printing (``pprint``) to analyse better.

    >>> tfm_8 = TransferFunctionMatrix([[tf_3], [tf_2], [-tf_1]])
    >>> tfm_9 = TransferFunctionMatrix([[-tf_3]])
    >>> tfm_10 = TransferFunctionMatrix([[tf_1], [tf_2], [tf_4]])
    >>> tfm_11 = TransferFunctionMatrix([[tf_4], [-tf_1]])
    >>> tfm_12 = TransferFunctionMatrix([[tf_4, -tf_1, tf_3], [-tf_2, -tf_4, -tf_3]])
    >>> tfm_8 + tfm_10
    MIMOParallel(TransferFunctionMatrix(((TransferFunction(3, s + 2, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a - s, s**2 + s + 1, s),))), TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a + p, 9*s - 9, s),))))
    >>> pprint(_, use_unicode=False)
    [     3      ]      [   a + s    ]
    [   -----    ]      [ ---------- ]
    [   s + 2    ]      [  2         ]
    [            ]      [ s  + s + 1 ]
    [ 4          ]      [            ]
    [p  - 3*p + 2]      [ 4          ]
    [------------]    + [p  - 3*p + 2]
    [   p + s    ]      [------------]
    [            ]      [   p + s    ]
    [   -a - s   ]      [            ]
    [ ---------- ]      [   -a + p   ]
    [  2         ]      [  -------   ]
    [ s  + s + 1 ]{t}   [  9*s - 9   ]{t}
    >>> -tfm_10 - tfm_8
    MIMOParallel(TransferFunctionMatrix(((TransferFunction(-a - s, s**2 + s + 1, s),), (TransferFunction(-p**4 + 3*p - 2, p + s, s),), (TransferFunction(a - p, 9*s - 9, s),))), TransferFunctionMatrix(((TransferFunction(-3, s + 2, s),), (TransferFunction(-p**4 + 3*p - 2, p + s, s),), (TransferFunction(a + s, s**2 + s + 1, s),))))
    >>> pprint(_, use_unicode=False)
    [    -a - s    ]      [     -3       ]
    [  ----------  ]      [    -----     ]
    [   2          ]      [    s + 2     ]
    [  s  + s + 1  ]      [              ]
    [              ]      [   4          ]
    [   4          ]      [- p  + 3*p - 2]
    [- p  + 3*p - 2]    + [--------------]
    [--------------]      [    p + s     ]
    [    p + s     ]      [              ]
    [              ]      [    a + s     ]
    [    a - p     ]      [  ----------  ]
    [   -------    ]      [   2          ]
    [   9*s - 9    ]{t}   [  s  + s + 1  ]{t}
    >>> tfm_12 * tfm_8
    MIMOSeries(TransferFunctionMatrix(((TransferFunction(3, s + 2, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a - s, s**2 + s + 1, s),))), TransferFunctionMatrix(((TransferFunction(-a + p, 9*s - 9, s), TransferFunction(-a - s, s**2 + s + 1, s), TransferFunction(3, s + 2, s)), (TransferFunction(-p**4 + 3*p - 2, p + s, s), TransferFunction(a - p, 9*s - 9, s), TransferFunction(-3, s + 2, s)))))
    >>> pprint(_, use_unicode=False)
                                           [     3      ]
                                           [   -----    ]
    [    -a + p        -a - s      3  ]    [   s + 2    ]
    [   -------      ----------  -----]    [            ]
    [   9*s - 9       2          s + 2]    [ 4          ]
    [                s  + s + 1       ]    [p  - 3*p + 2]
    [                                 ]   *[------------]
    [   4                             ]    [   p + s    ]
    [- p  + 3*p - 2    a - p      -3  ]    [            ]
    [--------------   -------    -----]    [   -a - s   ]
    [    p + s        9*s - 9    s + 2]{t} [ ---------- ]
                                           [  2         ]
                                           [ s  + s + 1 ]{t}
    >>> tfm_12 * tfm_8 * tfm_9
    MIMOSeries(TransferFunctionMatrix(((TransferFunction(-3, s + 2, s),),)), TransferFunctionMatrix(((TransferFunction(3, s + 2, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a - s, s**2 + s + 1, s),))), TransferFunctionMatrix(((TransferFunction(-a + p, 9*s - 9, s), TransferFunction(-a - s, s**2 + s + 1, s), TransferFunction(3, s + 2, s)), (TransferFunction(-p**4 + 3*p - 2, p + s, s), TransferFunction(a - p, 9*s - 9, s), TransferFunction(-3, s + 2, s)))))
    >>> pprint(_, use_unicode=False)
                                           [     3      ]
                                           [   -----    ]
    [    -a + p        -a - s      3  ]    [   s + 2    ]
    [   -------      ----------  -----]    [            ]
    [   9*s - 9       2          s + 2]    [ 4          ]
    [                s  + s + 1       ]    [p  - 3*p + 2]    [ -3  ]
    [                                 ]   *[------------]   *[-----]
    [   4                             ]    [   p + s    ]    [s + 2]{t}
    [- p  + 3*p - 2    a - p      -3  ]    [            ]
    [--------------   -------    -----]    [   -a - s   ]
    [    p + s        9*s - 9    s + 2]{t} [ ---------- ]
                                           [  2         ]
                                           [ s  + s + 1 ]{t}
    >>> tfm_10 + tfm_8*tfm_9
    MIMOParallel(TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a + p, 9*s - 9, s),))), MIMOSeries(TransferFunctionMatrix(((TransferFunction(-3, s + 2, s),),)), TransferFunctionMatrix(((TransferFunction(3, s + 2, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a - s, s**2 + s + 1, s),)))))
    >>> pprint(_, use_unicode=False)
    [   a + s    ]      [     3      ]
    [ ---------- ]      [   -----    ]
    [  2         ]      [   s + 2    ]
    [ s  + s + 1 ]      [            ]
    [            ]      [ 4          ]
    [ 4          ]      [p  - 3*p + 2]    [ -3  ]
    [p  - 3*p + 2]    + [------------]   *[-----]
    [------------]      [   p + s    ]    [s + 2]{t}
    [   p + s    ]      [            ]
    [            ]      [   -a - s   ]
    [   -a + p   ]      [ ---------- ]
    [  -------   ]      [  2         ]
    [  9*s - 9   ]{t}   [ s  + s + 1 ]{t}

    These unevaluated ``Series`` or ``Parallel`` objects can convert into the
    resultant transfer function matrix using ``.doit()`` method or by
    ``.rewrite(TransferFunctionMatrix)``.

    >>> (-tfm_8 + tfm_10 + tfm_8*tfm_9).doit()
    TransferFunctionMatrix(((TransferFunction((a + s)*(s + 2)**3 - 3*(s + 2)**2*(s**2 + s + 1) - 9*(s + 2)*(s**2 + s + 1), (s + 2)**3*(s**2 + s + 1), s),), (TransferFunction((p + s)*(-3*p**4 + 9*p - 6), (p + s)**2*(s + 2), s),), (TransferFunction((-a + p)*(s + 2)*(s**2 + s + 1)**2 + (a + s)*(s + 2)*(9*s - 9)*(s**2 + s + 1) + (3*a + 3*s)*(9*s - 9)*(s**2 + s + 1), (s + 2)*(9*s - 9)*(s**2 + s + 1)**2, s),)))
    >>> (-tfm_12 * -tfm_8 * -tfm_9).rewrite(TransferFunctionMatrix)
    TransferFunctionMatrix(((TransferFunction(3*(-3*a + 3*p)*(p + s)*(s + 2)*(s**2 + s + 1)**2 + 3*(-3*a - 3*s)*(p + s)*(s + 2)*(9*s - 9)*(s**2 + s + 1) + 3*(a + s)*(s + 2)**2*(9*s - 9)*(-p**4 + 3*p - 2)*(s**2 + s + 1), (p + s)*(s + 2)**3*(9*s - 9)*(s**2 + s + 1)**2, s),), (TransferFunction(3*(-a + p)*(p + s)*(s + 2)**2*(-p**4 + 3*p - 2)*(s**2 + s + 1) + 3*(3*a + 3*s)*(p + s)**2*(s + 2)*(9*s - 9) + 3*(p + s)*(s + 2)*(9*s - 9)*(-3*p**4 + 9*p - 6)*(s**2 + s + 1), (p + s)**2*(s + 2)**3*(9*s - 9)*(s**2 + s + 1), s),)))

    See Also
    ========

    TransferFunction, MIMOSeries, MIMOParallel, Feedback

    """
    def __new__(cls, arg):

        expr_mat_arg = []
        try:
            var = arg[0][0].var
        except TypeError:
            raise ValueError(filldedent("""
                `arg` param in TransferFunctionMatrix should
                strictly be a nested list containing TransferFunction
                objects."""))
        for row in arg:
            temp = []
            for element in row:
                if not isinstance(element, SISOLinearTimeInvariant):
                    raise TypeError(filldedent("""
                        Each element is expected to be of
                        type `SISOLinearTimeInvariant`."""))

                if var != element.var:
                    raise ValueError(filldedent("""
                        Conflicting value(s) found for `var`. All TransferFunction
                        instances in TransferFunctionMatrix should use the same
                        complex variable in Laplace domain."""))

                temp.append(element.to_expr())
            expr_mat_arg.append(temp)

        if isinstance(arg, (tuple, list, Tuple)):
            # Making nested Tuple (sympy.core.containers.Tuple) from nested list or nested Python tuple
            arg = Tuple(*(Tuple(*r, sympify=False) for r in arg), sympify=False)

        obj = super(TransferFunctionMatrix, cls).__new__(cls, arg)
        obj._expr_mat = ImmutableMatrix(expr_mat_arg)
        obj.is_StateSpace_object = False
        return obj

    @classmethod
    def from_Matrix(cls, matrix, var):
        """
        Creates a new ``TransferFunctionMatrix`` efficiently from a SymPy Matrix of ``Expr`` objects.

        Parameters
        ==========

        matrix : ``ImmutableMatrix`` having ``Expr``/``Number`` elements.
        var : Symbol
            Complex variable of the Laplace transform which will be used by the
            all the ``TransferFunction`` objects in the ``TransferFunctionMatrix``.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunctionMatrix
        >>> from sympy import Matrix, pprint
        >>> M = Matrix([[s, 1/s], [1/(s+1), s]])
        >>> M_tf = TransferFunctionMatrix.from_Matrix(M, s)
        >>> pprint(M_tf, use_unicode=False)
        [  s    1]
        [  -    -]
        [  1    s]
        [        ]
        [  1    s]
        [-----  -]
        [s + 1  1]{t}
        >>> M_tf.elem_poles()
        [[[], [0]], [[-1], []]]
        >>> M_tf.elem_zeros()
        [[[0], []], [[], [0]]]

        """
        return _to_TFM(matrix, var)

    @property
    def var(self):
        """
        Returns the complex variable used by all the transfer functions or
        ``Series``/``Parallel`` objects in a transfer function matrix.

        Examples
        ========

        >>> from sympy.abc import p, s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, Series, Parallel
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> G4 = TransferFunction(s + 1, s**2 + s + 1, s)
        >>> S1 = Series(G1, G2)
        >>> S2 = Series(-G3, Parallel(G2, -G1))
        >>> tfm1 = TransferFunctionMatrix([[G1], [G2], [G3]])
        >>> tfm1.var
        p
        >>> tfm2 = TransferFunctionMatrix([[-S1, -S2], [S1, S2]])
        >>> tfm2.var
        p
        >>> tfm3 = TransferFunctionMatrix([[G4]])
        >>> tfm3.var
        s

        """
        return self.args[0][0][0].var

    @property
    def num_inputs(self):
        """
        Returns the number of inputs of the system.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> G1 = TransferFunction(s + 3, s**2 - 3, s)
        >>> G2 = TransferFunction(4, s**2, s)
        >>> G3 = TransferFunction(p**2 + s**2, p - 3, s)
        >>> tfm_1 = TransferFunctionMatrix([[G2, -G1, G3], [-G2, -G1, -G3]])
        >>> tfm_1.num_inputs
        3

        See Also
        ========

        num_outputs

        """
        return self._expr_mat.shape[1]

    @property
    def num_outputs(self):
        """
        Returns the number of outputs of the system.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunctionMatrix
        >>> from sympy import Matrix
        >>> M_1 = Matrix([[s], [1/s]])
        >>> TFM = TransferFunctionMatrix.from_Matrix(M_1, s)
        >>> print(TFM)
        TransferFunctionMatrix(((TransferFunction(s, 1, s),), (TransferFunction(1, s, s),)))
        >>> TFM.num_outputs
        2

        See Also
        ========

        num_inputs

        """
        return self._expr_mat.shape[0]

    @property
    def shape(self):
        """
        Returns the shape of the transfer function matrix, that is, ``(# of outputs, # of inputs)``.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> tf1 = TransferFunction(p**2 - 1, s**4 + s**3 - p, p)
        >>> tf2 = TransferFunction(1 - p, p**2 - 3*p + 7, p)
        >>> tf3 = TransferFunction(3, 4, p)
        >>> tfm1 = TransferFunctionMatrix([[tf1, -tf2]])
        >>> tfm1.shape
        (1, 2)
        >>> tfm2 = TransferFunctionMatrix([[-tf2, tf3], [tf1, -tf1]])
        >>> tfm2.shape
        (2, 2)

        """
        return self._expr_mat.shape

    def __neg__(self):
        neg = -self._expr_mat
        return _to_TFM(neg, self.var)

    @_check_other_MIMO
    def __add__(self, other):

        if not isinstance(other, MIMOParallel):
            return MIMOParallel(self, other)
        other_arg_list = list(other.args)
        return MIMOParallel(self, *other_arg_list)

    @_check_other_MIMO
    def __sub__(self, other):
        return self + (-other)

    @_check_other_MIMO
    def __mul__(self, other):

        if not isinstance(other, MIMOSeries):
            return MIMOSeries(other, self)
        other_arg_list = list(other.args)
        return MIMOSeries(*other_arg_list, self)

    def __getitem__(self, key):
        trunc = self._expr_mat.__getitem__(key)
        if isinstance(trunc, ImmutableMatrix):
            return _to_TFM(trunc, self.var)
        return TransferFunction.from_rational_expression(trunc, self.var)

    def transpose(self):
        """Returns the transpose of the ``TransferFunctionMatrix`` (switched input and output layers)."""
        transposed_mat = self._expr_mat.transpose()
        return _to_TFM(transposed_mat, self.var)

    def elem_poles(self):
        """
        Returns the poles of each element of the ``TransferFunctionMatrix``.

        .. note::
            Actual poles of a MIMO system are NOT the poles of individual elements.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> tf_1 = TransferFunction(3, (s + 1), s)
        >>> tf_2 = TransferFunction(s + 6, (s + 1)*(s + 2), s)
        >>> tf_3 = TransferFunction(s + 3, s**2 + 3*s + 2, s)
        >>> tf_4 = TransferFunction(s + 2, s**2 + 5*s - 10, s)
        >>> tfm_1 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, tf_4]])
        >>> tfm_1
        TransferFunctionMatrix(((TransferFunction(3, s + 1, s), TransferFunction(s + 6, (s + 1)*(s + 2), s)), (TransferFunction(s + 3, s**2 + 3*s + 2, s), TransferFunction(s + 2, s**2 + 5*s - 10, s))))
        >>> tfm_1.elem_poles()
        [[[-1], [-2, -1]], [[-2, -1], [-5/2 + sqrt(65)/2, -sqrt(65)/2 - 5/2]]]

        See Also
        ========

        elem_zeros

        """
        return [[element.poles() for element in row] for row in self.doit().args[0]]

    def elem_zeros(self):
        """
        Returns the zeros of each element of the ``TransferFunctionMatrix``.

        .. note::
            Actual zeros of a MIMO system are NOT the zeros of individual elements.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> tf_1 = TransferFunction(3, (s + 1), s)
        >>> tf_2 = TransferFunction(s + 6, (s + 1)*(s + 2), s)
        >>> tf_3 = TransferFunction(s + 3, s**2 + 3*s + 2, s)
        >>> tf_4 = TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s)
        >>> tfm_1 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, tf_4]])
        >>> tfm_1
        TransferFunctionMatrix(((TransferFunction(3, s + 1, s), TransferFunction(s + 6, (s + 1)*(s + 2), s)), (TransferFunction(s + 3, s**2 + 3*s + 2, s), TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s))))
        >>> tfm_1.elem_zeros()
        [[[], [-6]], [[-3], [4, 5]]]

        See Also
        ========

        elem_poles

        """
        return [[element.zeros() for element in row] for row in self.doit().args[0]]

    def eval_frequency(self, other):
        """
        Evaluates system response of each transfer function in the ``TransferFunctionMatrix`` at any point in the real or complex plane.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> from sympy import I
        >>> tf_1 = TransferFunction(3, (s + 1), s)
        >>> tf_2 = TransferFunction(s + 6, (s + 1)*(s + 2), s)
        >>> tf_3 = TransferFunction(s + 3, s**2 + 3*s + 2, s)
        >>> tf_4 = TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s)
        >>> tfm_1 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, tf_4]])
        >>> tfm_1
        TransferFunctionMatrix(((TransferFunction(3, s + 1, s), TransferFunction(s + 6, (s + 1)*(s + 2), s)), (TransferFunction(s + 3, s**2 + 3*s + 2, s), TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s))))
        >>> tfm_1.eval_frequency(2)
        Matrix([
        [   1, 2/3],
        [5/12, 3/2]])
        >>> tfm_1.eval_frequency(I*2)
        Matrix([
        [   3/5 - 6*I/5,                -I],
        [3/20 - 11*I/20, -101/74 + 23*I/74]])
        """
        mat = self._expr_mat.subs(self.var, other)
        return mat.expand()

    def _flat(self):
        """Returns flattened list of args in TransferFunctionMatrix"""
        return [elem for tup in self.args[0] for elem in tup]

    def _eval_evalf(self, prec):
        """Calls evalf() on each transfer function in the transfer function matrix"""
        dps = prec_to_dps(prec)
        mat = self._expr_mat.applyfunc(lambda a: a.evalf(n=dps))
        return _to_TFM(mat, self.var)

    def _eval_simplify(self, **kwargs):
        """Simplifies the transfer function matrix"""
        simp_mat = self._expr_mat.applyfunc(lambda a: cancel(a, expand=False))
        return _to_TFM(simp_mat, self.var)

    def expand(self, **hints):
        """Expands the transfer function matrix"""
        expand_mat = self._expr_mat.expand(**hints)
        return _to_TFM(expand_mat, self.var)

class StateSpace(LinearTimeInvariant):
    r"""
    State space model (ssm) of a linear, time invariant control system.

    Represents the standard state-space model with A, B, C, D as state-space matrices.
    This makes the linear control system:

        (1) x'(t) = A * x(t) + B * u(t);    x in R^n , u in R^k
        (2) y(t)  = C * x(t) + D * u(t);    y in R^m

    where u(t) is any input signal, y(t) the corresponding output, and x(t) the system's state.

    Parameters
    ==========

    A : Matrix
        The State matrix of the state space model.
    B : Matrix
        The Input-to-State matrix of the state space model.
    C : Matrix
        The State-to-Output matrix of the state space model.
    D : Matrix
        The Feedthrough matrix of the state space model.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.physics.control import StateSpace

    The easiest way to create a StateSpaceModel is via four matrices:

    >>> A = Matrix([[1, 2], [1, 0]])
    >>> B = Matrix([1, 1])
    >>> C = Matrix([[0, 1]])
    >>> D = Matrix([0])
    >>> StateSpace(A, B, C, D)
    StateSpace(Matrix([
    [1, 2],
    [1, 0]]), Matrix([
    [1],
    [1]]), Matrix([[0, 1]]), Matrix([[0]]))

    One can use less matrices. The rest will be filled with a minimum of zeros:

    >>> StateSpace(A, B)
    StateSpace(Matrix([
    [1, 2],
    [1, 0]]), Matrix([
    [1],
    [1]]), Matrix([[0, 0]]), Matrix([[0]]))

    See Also
    ========

    TransferFunction, TransferFunctionMatrix

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/State-space_representation
    .. [2] https://in.mathworks.com/help/control/ref/ss.html

    """
    def __new__(cls, A=None, B=None, C=None, D=None):
        if A is None:
            A = zeros(1)
        if B is None:
            B = zeros(A.rows, 1)
        if C is None:
            C = zeros(1, A.cols)
        if D is None:
            D = zeros(C.rows, B.cols)

        A = _sympify(A)
        B = _sympify(B)
        C = _sympify(C)
        D = _sympify(D)

        if (isinstance(A, ImmutableDenseMatrix) and isinstance(B, ImmutableDenseMatrix) and
            isinstance(C, ImmutableDenseMatrix) and isinstance(D, ImmutableDenseMatrix)):
            # Check State Matrix is square
            if A.rows != A.cols:
                raise ShapeError("Matrix A must be a square matrix.")

            # Check State and Input matrices have same rows
            if A.rows != B.rows:
                raise ShapeError("Matrices A and B must have the same number of rows.")

            # Check Output and Feedthrough matrices have same rows
            if C.rows != D.rows:
                raise ShapeError("Matrices C and D must have the same number of rows.")

            # Check State and Output matrices have same columns
            if A.cols != C.cols:
                raise ShapeError("Matrices A and C must have the same number of columns.")

            # Check Input and Feedthrough matrices have same columns
            if B.cols != D.cols:
                raise ShapeError("Matrices B and D must have the same number of columns.")

            obj = super(StateSpace, cls).__new__(cls, A, B, C, D)
            obj._A = A
            obj._B = B
            obj._C = C
            obj._D = D

            # Determine if the system is SISO or MIMO
            num_outputs = D.rows
            num_inputs = D.cols
            if num_inputs == 1 and num_outputs == 1:
                obj._is_SISO = True
                obj._clstype = SISOLinearTimeInvariant
            else:
                obj._is_SISO = False
                obj._clstype = MIMOLinearTimeInvariant
            obj.is_StateSpace_object = True
            return obj

        else:
            raise TypeError("A, B, C and D inputs must all be sympy Matrices.")

    @property
    def state_matrix(self):
        """
        Returns the state matrix of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.state_matrix
        Matrix([
        [1, 2],
        [1, 0]])

        """
        return self._A

    @property
    def input_matrix(self):
        """
        Returns the input matrix of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.input_matrix
        Matrix([
        [1],
        [1]])

        """
        return self._B

    @property
    def output_matrix(self):
        """
        Returns the output matrix of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.output_matrix
        Matrix([[0, 1]])

        """
        return self._C

    @property
    def feedforward_matrix(self):
        """
        Returns the feedforward matrix of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.feedforward_matrix
        Matrix([[0]])

        """
        return self._D

    A = state_matrix
    B = input_matrix
    C = output_matrix
    D = feedforward_matrix

    @property
    def num_states(self):
        """
        Returns the number of states of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.num_states
        2

        """
        return self._A.rows

    @property
    def num_inputs(self):
        """
        Returns the number of inputs of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.num_inputs
        1

        """
        return self._D.cols

    @property
    def num_outputs(self):
        """
        Returns the number of outputs of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.num_outputs
        1

        """
        return self._D.rows


    @property
    def shape(self):
        """Returns the shape of the equivalent StateSpace system."""
        return self.num_outputs, self.num_inputs

    def dsolve(self, initial_conditions=None, input_vector=None, var=Symbol('t')):
        r"""
        Returns `y(t)` or output of StateSpace given by the solution of equations:
            x'(t) = A * x(t) + B * u(t)
            y(t)  = C * x(t) + D * u(t)

        Parameters
        ============

        initial_conditions : Matrix
            The initial conditions of `x` state vector. If not provided, it defaults to a zero vector.
        input_vector : Matrix
            The input vector for state space. If not provided, it defaults to a zero vector.
        var : Symbol
            The symbol representing time. If not provided, it defaults to `t`.

        Examples
        ==========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-2, 0], [1, -1]])
        >>> B = Matrix([[1], [0]])
        >>> C = Matrix([[2, 1]])
        >>> ip = Matrix([5])
        >>> i = Matrix([0, 0])
        >>> ss = StateSpace(A, B, C)
        >>> ss.dsolve(input_vector=ip, initial_conditions=i).simplify()
        Matrix([[15/2 - 5*exp(-t) - 5*exp(-2*t)/2]])

        If no input is provided it defaults to solving the system with zero initial conditions and zero input.

        >>> ss.dsolve()
        Matrix([[0]])

        References
        ==========
        .. [1] https://web.mit.edu/2.14/www/Handouts/StateSpaceResponse.pdf
        .. [2] https://docs.sympy.org/latest/modules/solvers/ode.html#sympy.solvers.ode.systems.linodesolve

        """

        if not isinstance(var, Symbol):
            raise ValueError("Variable for representing time must be a Symbol.")
        if not initial_conditions:
            initial_conditions = zeros(self._A.shape[0], 1)
        elif initial_conditions.shape != (self._A.shape[0], 1):
            raise ShapeError("Initial condition vector should have the same number of "
                             "rows as the state matrix.")
        if not input_vector:
            input_vector = zeros(self._B.shape[1], 1)
        elif input_vector.shape != (self._B.shape[1], 1):
            raise ShapeError("Input vector should have the same number of "
                             "columns as the input matrix.")
        sol = linodesolve(A=self._A, t=var, b=self._B*input_vector, type='type2', doit=True)
        mat1 = Matrix(sol)
        mat2 = mat1.replace(var, 0)
        free1 = self._A.free_symbols | self._B.free_symbols | input_vector.free_symbols
        free2 = mat2.free_symbols
        # Get all the free symbols form the matrix
        dummy_symbols = list(free2-free1)
        # Convert the matrix to a Coefficient matrix
        r1, r2 = linear_eq_to_matrix(mat2, dummy_symbols)
        s = linsolve((r1, initial_conditions+r2))
        res_tuple = next(iter(s))
        for ind, v in enumerate(res_tuple):
            mat1 = mat1.replace(dummy_symbols[ind], v)
        res = self._C*mat1 + self._D*input_vector
        return res

    def _eval_evalf(self, prec):
        """
        Returns state space model where numerical expressions are evaluated into floating point numbers.
        """
        dps = prec_to_dps(prec)
        return StateSpace(
            self._A.evalf(n = dps),
            self._B.evalf(n = dps),
            self._C.evalf(n = dps),
            self._D.evalf(n = dps))

    def _eval_rewrite_as_TransferFunction(self, *args):
        """
        Returns the equivalent Transfer Function of the state space model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import TransferFunction, StateSpace
        >>> A = Matrix([[-5, -1], [3, -1]])
        >>> B = Matrix([2, 5])
        >>> C = Matrix([[1, 2]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.rewrite(TransferFunction)
        [[TransferFunction(12*s + 59, s**2 + 6*s + 8, s)]]

        """
        s = Symbol('s')
        n = self._A.shape[0]
        I = eye(n)
        G = self._C*(s*I - self._A).solve(self._B) + self._D
        G = G.simplify()
        to_tf = lambda expr: TransferFunction.from_rational_expression(expr, s)
        tf_mat = [[to_tf(expr) for expr in sublist] for sublist in G.tolist()]
        return tf_mat

    def __add__(self, other):
        """
        Add two State Space systems (parallel connection).

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A1 = Matrix([[1]])
        >>> B1 = Matrix([[2]])
        >>> C1 = Matrix([[-1]])
        >>> D1 = Matrix([[-2]])
        >>> A2 = Matrix([[-1]])
        >>> B2 = Matrix([[-2]])
        >>> C2 = Matrix([[1]])
        >>> D2 = Matrix([[2]])
        >>> ss1 = StateSpace(A1, B1, C1, D1)
        >>> ss2 = StateSpace(A2, B2, C2, D2)
        >>> ss1 + ss2
        StateSpace(Matrix([
        [1,  0],
        [0, -1]]), Matrix([
        [ 2],
        [-2]]), Matrix([[-1, 1]]), Matrix([[0]]))

        """
        # Check for scalars
        if isinstance(other, (int, float, complex, Symbol)):
            A = self._A
            B = self._B
            C = self._C
            D = self._D.applyfunc(lambda element: element + other)

        else:
            # Check nature of system
            if not isinstance(other, StateSpace):
                raise ValueError("Addition is only supported for 2 State Space models.")
            # Check dimensions of system
            elif ((self.num_inputs != other.num_inputs) or (self.num_outputs != other.num_outputs)):
                raise ShapeError("Systems with incompatible inputs and outputs cannot be added.")

            m1 = (self._A).row_join(zeros(self._A.shape[0], other._A.shape[-1]))
            m2 = zeros(other._A.shape[0], self._A.shape[-1]).row_join(other._A)

            A = m1.col_join(m2)
            B = self._B.col_join(other._B)
            C = self._C.row_join(other._C)
            D = self._D + other._D

        return StateSpace(A, B, C, D)

    def __radd__(self, other):
        """
        Right add two State Space systems.

        Examples
        ========

        >>> from sympy.physics.control import StateSpace
        >>> s = StateSpace()
        >>> 5 + s
        StateSpace(Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[5]]))

        """
        return self + other

    def __sub__(self, other):
        """
        Subtract two State Space systems.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A1 = Matrix([[1]])
        >>> B1 = Matrix([[2]])
        >>> C1 = Matrix([[-1]])
        >>> D1 = Matrix([[-2]])
        >>> A2 = Matrix([[-1]])
        >>> B2 = Matrix([[-2]])
        >>> C2 = Matrix([[1]])
        >>> D2 = Matrix([[2]])
        >>> ss1 = StateSpace(A1, B1, C1, D1)
        >>> ss2 = StateSpace(A2, B2, C2, D2)
        >>> ss1 - ss2
        StateSpace(Matrix([
        [1,  0],
        [0, -1]]), Matrix([
        [ 2],
        [-2]]), Matrix([[-1, -1]]), Matrix([[-4]]))

        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Right subtract two tate Space systems.

        Examples
        ========

        >>> from sympy.physics.control import StateSpace
        >>> s = StateSpace()
        >>> 5 - s
        StateSpace(Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[5]]))

        """
        return other + (-self)

    def __neg__(self):
        """
        Returns the negation of the state space model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-5, -1], [3, -1]])
        >>> B = Matrix([2, 5])
        >>> C = Matrix([[1, 2]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> -ss
        StateSpace(Matrix([
        [-5, -1],
        [ 3, -1]]), Matrix([
        [2],
        [5]]), Matrix([[-1, -2]]), Matrix([[0]]))

        """
        return StateSpace(self._A, self._B, -self._C, -self._D)

    def __mul__(self, other):
        """
        Multiplication of two State Space systems (serial connection).

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-5, -1], [3, -1]])
        >>> B = Matrix([2, 5])
        >>> C = Matrix([[1, 2]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss*5
        StateSpace(Matrix([
        [-5, -1],
        [ 3, -1]]), Matrix([
        [2],
        [5]]), Matrix([[5, 10]]), Matrix([[0]]))

        """
        # Check for scalars
        if isinstance(other, (int, float, complex, Symbol)):
            A = self._A
            B = self._B
            C = self._C.applyfunc(lambda element: element*other)
            D = self._D.applyfunc(lambda element: element*other)

        else:
            # Check nature of system
            if not isinstance(other, StateSpace):
                raise ValueError("Multiplication is only supported for 2 State Space models.")
            # Check dimensions of system
            elif self.num_inputs != other.num_outputs:
                raise ShapeError("Systems with incompatible inputs and outputs cannot be multiplied.")

            m1 = (other._A).row_join(zeros(other._A.shape[0], self._A.shape[1]))
            m2 = (self._B * other._C).row_join(self._A)

            A = m1.col_join(m2)
            B = (other._B).col_join(self._B * other._D)
            C = (self._D * other._C).row_join(self._C)
            D = self._D * other._D

        return StateSpace(A, B, C, D)

    def __rmul__(self, other):
        """
        Right multiply two tate Space systems.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-5, -1], [3, -1]])
        >>> B = Matrix([2, 5])
        >>> C = Matrix([[1, 2]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> 5*ss
        StateSpace(Matrix([
        [-5, -1],
        [ 3, -1]]), Matrix([
        [10],
        [25]]), Matrix([[1, 2]]), Matrix([[0]]))

        """
        if isinstance(other, (int, float, complex, Symbol)):
            A = self._A
            C = self._C
            B = self._B.applyfunc(lambda element: element*other)
            D = self._D.applyfunc(lambda element: element*other)
            return StateSpace(A, B, C, D)
        else:
            return self*other

    def __repr__(self):
        A_str = self._A.__repr__()
        B_str = self._B.__repr__()
        C_str = self._C.__repr__()
        D_str = self._D.__repr__()

        return f"StateSpace(\n{A_str},\n\n{B_str},\n\n{C_str},\n\n{D_str})"


    def append(self, other):
        """
        Returns the first model appended with the second model. The order is preserved.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A1 = Matrix([[1]])
        >>> B1 = Matrix([[2]])
        >>> C1 = Matrix([[-1]])
        >>> D1 = Matrix([[-2]])
        >>> A2 = Matrix([[-1]])
        >>> B2 = Matrix([[-2]])
        >>> C2 = Matrix([[1]])
        >>> D2 = Matrix([[2]])
        >>> ss1 = StateSpace(A1, B1, C1, D1)
        >>> ss2 = StateSpace(A2, B2, C2, D2)
        >>> ss1.append(ss2)
        StateSpace(Matrix([
        [1,  0],
        [0, -1]]), Matrix([
        [2,  0],
        [0, -2]]), Matrix([
        [-1, 0],
        [ 0, 1]]), Matrix([
        [-2, 0],
        [ 0, 2]]))

        """
        n = self.num_states + other.num_states
        m = self.num_inputs + other.num_inputs
        p = self.num_outputs + other.num_outputs

        A = zeros(n, n)
        B = zeros(n, m)
        C = zeros(p, n)
        D = zeros(p, m)

        A[:self.num_states, :self.num_states] = self._A
        A[self.num_states:, self.num_states:] = other._A
        B[:self.num_states, :self.num_inputs] = self._B
        B[self.num_states:, self.num_inputs:] = other._B
        C[:self.num_outputs, :self.num_states] = self._C
        C[self.num_outputs:, self.num_states:] = other._C
        D[:self.num_outputs, :self.num_inputs] = self._D
        D[self.num_outputs:, self.num_inputs:] = other._D
        return StateSpace(A, B, C, D)

    def observability_matrix(self):
        """
        Returns the observability matrix of the state space model:
            [C, C * A^1, C * A^2, .. , C * A^(n-1)]; A in R^(n x n), C in R^(m x k)

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-1.5, -2], [1, 0]])
        >>> B = Matrix([0.5, 0])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([1])
        >>> ss = StateSpace(A, B, C, D)
        >>> ob = ss.observability_matrix()
        >>> ob
        Matrix([
        [0, 1],
        [1, 0]])

        References
        ==========
        .. [1] https://in.mathworks.com/help/control/ref/statespacemodel.obsv.html

        """
        n = self.num_states
        ob = self._C
        for i in range(1,n):
            ob = ob.col_join(self._C * self._A**i)

        return ob

    def observable_subspace(self):
        """
        Returns the observable subspace of the state space model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-1.5, -2], [1, 0]])
        >>> B = Matrix([0.5, 0])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([1])
        >>> ss = StateSpace(A, B, C, D)
        >>> ob_subspace = ss.observable_subspace()
        >>> ob_subspace
        [Matrix([
        [0],
        [1]]), Matrix([
        [1],
        [0]])]

        """
        return self.observability_matrix().columnspace()

    def is_observable(self):
        """
        Returns if the state space model is observable.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-1.5, -2], [1, 0]])
        >>> B = Matrix([0.5, 0])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([1])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.is_observable()
        True

        """
        return self.observability_matrix().rank() == self.num_states

    def controllability_matrix(self):
        """
        Returns the controllability matrix of the system:
            [B, A * B, A^2 * B, .. , A^(n-1) * B]; A in R^(n x n), B in R^(n x m)

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-1.5, -2], [1, 0]])
        >>> B = Matrix([0.5, 0])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([1])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.controllability_matrix()
        Matrix([
        [0.5, -0.75],
        [  0,   0.5]])

        References
        ==========
        .. [1] https://in.mathworks.com/help/control/ref/statespacemodel.ctrb.html

        """
        co = self._B
        n = self._A.shape[0]
        for i in range(1, n):
            co = co.row_join(((self._A)**i) * self._B)

        return co

    def controllable_subspace(self):
        """
        Returns the controllable subspace of the state space model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-1.5, -2], [1, 0]])
        >>> B = Matrix([0.5, 0])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([1])
        >>> ss = StateSpace(A, B, C, D)
        >>> co_subspace = ss.controllable_subspace()
        >>> co_subspace
        [Matrix([
        [0.5],
        [  0]]), Matrix([
        [-0.75],
        [  0.5]])]

        """
        return self.controllability_matrix().columnspace()

    def is_controllable(self):
        """
        Returns if the state space model is controllable.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-1.5, -2], [1, 0]])
        >>> B = Matrix([0.5, 0])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([1])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.is_controllable()
        True

        """
        return self.controllability_matrix().rank() == self.num_states

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\_version.py ===
"""Utility to compare (NumPy) version strings.

The NumpyVersion class allows properly comparing numpy version strings.
The LooseVersion and StrictVersion classes that distutils provides don't
work; they don't recognize anything like alpha/beta/rc/dev versions.

"""
import re

__all__ = ['NumpyVersion']


class NumpyVersion:
    """Parse and compare numpy version strings.

    NumPy has the following versioning scheme (numbers given are examples; they
    can be > 9 in principle):

    - Released version: '1.8.0', '1.8.1', etc.
    - Alpha: '1.8.0a1', '1.8.0a2', etc.
    - Beta: '1.8.0b1', '1.8.0b2', etc.
    - Release candidates: '1.8.0rc1', '1.8.0rc2', etc.
    - Development versions: '1.8.0.dev-f1234afa' (git commit hash appended)
    - Development versions after a1: '1.8.0a1.dev-f1234afa',
                                     '1.8.0b2.dev-f1234afa',
                                     '1.8.1rc1.dev-f1234afa', etc.
    - Development versions (no git hash available): '1.8.0.dev-Unknown'

    Comparing needs to be done against a valid version string or other
    `NumpyVersion` instance. Note that all development versions of the same
    (pre-)release compare equal.

    Parameters
    ----------
    vstring : str
        NumPy version string (``np.__version__``).

    Examples
    --------
    >>> from numpy.lib import NumpyVersion
    >>> if NumpyVersion(np.__version__) < '1.7.0':
    ...     print('skip')
    >>> # skip

    >>> NumpyVersion('1.7')  # raises ValueError, add ".0"
    Traceback (most recent call last):
        ...
    ValueError: Not a valid numpy version string

    """

    __module__ = "numpy.lib"

    def __init__(self, vstring):
        self.vstring = vstring
        ver_main = re.match(r'\d+\.\d+\.\d+', vstring)
        if not ver_main:
            raise ValueError("Not a valid numpy version string")

        self.version = ver_main.group()
        self.major, self.minor, self.bugfix = [int(x) for x in
            self.version.split('.')]
        if len(vstring) == ver_main.end():
            self.pre_release = 'final'
        else:
            alpha = re.match(r'a\d', vstring[ver_main.end():])
            beta = re.match(r'b\d', vstring[ver_main.end():])
            rc = re.match(r'rc\d', vstring[ver_main.end():])
            pre_rel = [m for m in [alpha, beta, rc] if m is not None]
            if pre_rel:
                self.pre_release = pre_rel[0].group()
            else:
                self.pre_release = ''

        self.is_devversion = bool(re.search(r'.dev', vstring))

    def _compare_version(self, other):
        """Compare major.minor.bugfix"""
        if self.major == other.major:
            if self.minor == other.minor:
                if self.bugfix == other.bugfix:
                    vercmp = 0
                elif self.bugfix > other.bugfix:
                    vercmp = 1
                else:
                    vercmp = -1
            elif self.minor > other.minor:
                vercmp = 1
            else:
                vercmp = -1
        elif self.major > other.major:
            vercmp = 1
        else:
            vercmp = -1

        return vercmp

    def _compare_pre_release(self, other):
        """Compare alpha/beta/rc/final."""
        if self.pre_release == other.pre_release:
            vercmp = 0
        elif self.pre_release == 'final':
            vercmp = 1
        elif other.pre_release == 'final':
            vercmp = -1
        elif self.pre_release > other.pre_release:
            vercmp = 1
        else:
            vercmp = -1

        return vercmp

    def _compare(self, other):
        if not isinstance(other, (str, NumpyVersion)):
            raise ValueError("Invalid object to compare with NumpyVersion.")

        if isinstance(other, str):
            other = NumpyVersion(other)

        vercmp = self._compare_version(other)
        if vercmp == 0:
            # Same x.y.z version, check for alpha/beta/rc
            vercmp = self._compare_pre_release(other)
            if vercmp == 0:
                # Same version and same pre-release, check if dev version
                if self.is_devversion is other.is_devversion:
                    vercmp = 0
                elif self.is_devversion:
                    vercmp = -1
                else:
                    vercmp = 1

        return vercmp

    def __lt__(self, other):
        return self._compare(other) < 0

    def __le__(self, other):
        return self._compare(other) <= 0

    def __eq__(self, other):
        return self._compare(other) == 0

    def __ne__(self, other):
        return self._compare(other) != 0

    def __gt__(self, other):
        return self._compare(other) > 0

    def __ge__(self, other):
        return self._compare(other) >= 0

    def __repr__(self):
        return f"NumpyVersion({self.vstring})"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\crv_types.py ===
"""
Continuous Random Variables - Prebuilt variables

Contains
========
Arcsin
Benini
Beta
BetaNoncentral
BetaPrime
BoundedPareto
Cauchy
Chi
ChiNoncentral
ChiSquared
Dagum
Davis
Erlang
ExGaussian
Exponential
ExponentialPower
FDistribution
FisherZ
Frechet
Gamma
GammaInverse
Gumbel
Gompertz
Kumaraswamy
Laplace
Levy
LogCauchy
Logistic
LogLogistic
LogitNormal
LogNormal
Lomax
Maxwell
Moyal
Nakagami
Normal
Pareto
PowerFunction
QuadraticU
RaisedCosine
Rayleigh
Reciprocal
ShiftedGompertz
StudentT
Trapezoidal
Triangular
Uniform
UniformSum
VonMises
Wald
Weibull
WignerSemicircle
"""



from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import asin
from sympy.functions.special.error_functions import (erf, erfc, erfi, erfinv, expint)
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.matrices import MatrixBase
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDistribution
from sympy.stats.rv import _value_check, is_random

oo = S.Infinity

__all__ = ['ContinuousRV',
'Arcsin',
'Benini',
'Beta',
'BetaNoncentral',
'BetaPrime',
'BoundedPareto',
'Cauchy',
'Chi',
'ChiNoncentral',
'ChiSquared',
'Dagum',
'Davis',
'Erlang',
'ExGaussian',
'Exponential',
'ExponentialPower',
'FDistribution',
'FisherZ',
'Frechet',
'Gamma',
'GammaInverse',
'Gompertz',
'Gumbel',
'Kumaraswamy',
'Laplace',
'Levy',
'LogCauchy',
'Logistic',
'LogLogistic',
'LogitNormal',
'LogNormal',
'Lomax',
'Maxwell',
'Moyal',
'Nakagami',
'Normal',
'GaussianInverse',
'Pareto',
'PowerFunction',
'QuadraticU',
'RaisedCosine',
'Rayleigh',
'Reciprocal',
'StudentT',
'ShiftedGompertz',
'Trapezoidal',
'Triangular',
'Uniform',
'UniformSum',
'VonMises',
'Wald',
'Weibull',
'WignerSemicircle',
]


@is_random.register(MatrixBase)
def _(x):
    return any(is_random(i) for i in x)

def rv(symbol, cls, args, **kwargs):
    args = list(map(sympify, args))
    dist = cls(*args)
    if kwargs.pop('check', True):
        dist.check(*args)
    pspace = SingleContinuousPSpace(symbol, dist)
    if any(is_random(arg) for arg in args):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(symbol, CompoundDistribution(dist))
    return pspace.value


class ContinuousDistributionHandmade(SingleContinuousDistribution):
    _argnames = ('pdf',)

    def __new__(cls, pdf, set=Interval(-oo, oo)):
        return Basic.__new__(cls, pdf, set)

    @property
    def set(self):
        return self.args[1]

    @staticmethod
    def check(pdf, set):
        x = Dummy('x')
        val = integrate(pdf(x), (x, set))
        _value_check(Eq(val, 1) != S.false, "The pdf on the given set is incorrect.")


def ContinuousRV(symbol, density, set=Interval(-oo, oo), **kwargs):
    """
    Create a Continuous Random Variable given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents name of the random variable.
    density : Expression containing symbol
        Represents probability density function.
    set : set/Interval
        Represents the region where the pdf is valid, by default is real line.
    check : bool
        If True, it will check whether the given density
        integrates to 1 over the given set. If False, it
        will not perform this check. Default is False.


    Returns
    =======

    RandomSymbol

    Many common continuous random variable types are already implemented.
    This function should be necessary only very rarely.


    Examples
    ========

    >>> from sympy import Symbol, sqrt, exp, pi
    >>> from sympy.stats import ContinuousRV, P, E

    >>> x = Symbol("x")

    >>> pdf = sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)) # Normal distribution
    >>> X = ContinuousRV(x, pdf)

    >>> E(X)
    0
    >>> P(X>0)
    1/2
    """
    pdf = Piecewise((density, set.as_relational(symbol)), (0, True))
    pdf = Lambda(symbol, pdf)
    # have a default of False while `rv` should have a default of True
    kwargs['check'] = kwargs.pop('check', False)
    return rv(symbol.name, ContinuousDistributionHandmade, (pdf, set), **kwargs)

########################################
# Continuous Probability Distributions #
########################################

#-------------------------------------------------------------------------------
# Arcsin distribution ----------------------------------------------------------


class ArcsinDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    def pdf(self, x):
        a, b = self.a, self.b
        return 1/(pi*sqrt((x - a)*(b - x)))

    def _cdf(self, x):
        a, b = self.a, self.b
        return Piecewise(
            (S.Zero, x < a),
            (2*asin(sqrt((x - a)/(b - a)))/pi, x <= b),
            (S.One, True))


def Arcsin(name, a=0, b=1):
    r"""
    Create a Continuous Random Variable with an arcsin distribution.

    The density of the arcsin distribution is given by

    .. math::
        f(x) := \frac{1}{\pi\sqrt{(x-a)(b-x)}}

    with :math:`x \in (a,b)`. It must hold that :math:`-\infty < a < b < \infty`.

    Parameters
    ==========

    a : Real number, the left interval boundary
    b : Real number, the right interval boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Arcsin, density, cdf
    >>> from sympy import Symbol

    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = Arcsin("x", a, b)

    >>> density(X)(z)
    1/(pi*sqrt((-a + z)*(b - z)))

    >>> cdf(X)(z)
    Piecewise((0, a > z),
            (2*asin(sqrt((-a + z)/(-a + b)))/pi, b >= z),
            (1, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Arcsine_distribution

    """

    return rv(name, ArcsinDistribution, (a, b))

#-------------------------------------------------------------------------------
# Benini distribution ----------------------------------------------------------


class BeniniDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta', 'sigma')

    @staticmethod
    def check(alpha, beta, sigma):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")
        _value_check(sigma > 0, "Scale parameter Sigma must be positive.")

    @property
    def set(self):
        return Interval(self.sigma, oo)

    def pdf(self, x):
        alpha, beta, sigma = self.alpha, self.beta, self.sigma
        return (exp(-alpha*log(x/sigma) - beta*log(x/sigma)**2)
               *(alpha/x + 2*beta*log(x/sigma)/x))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function of the '
                                  'Benini distribution does not exist.')

def Benini(name, alpha, beta, sigma):
    r"""
    Create a Continuous Random Variable with a Benini distribution.

    The density of the Benini distribution is given by

    .. math::
        f(x) := e^{-\alpha\log{\frac{x}{\sigma}}
                -\beta\log^2\left[{\frac{x}{\sigma}}\right]}
                \left(\frac{\alpha}{x}+\frac{2\beta\log{\frac{x}{\sigma}}}{x}\right)

    This is a heavy-tailed distribution and is also known as the log-Rayleigh
    distribution.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
    beta : Real number, `\beta > 0`, a shape
    sigma : Real number, `\sigma > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Benini, density, cdf
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = Benini("x", alpha, beta, sigma)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    /                  /  z  \\             /  z  \            2/  z  \
    |        2*beta*log|-----||  - alpha*log|-----| - beta*log  |-----|
    |alpha             \sigma/|             \sigma/             \sigma/
    |----- + -----------------|*e
    \  z             z        /

    >>> cdf(X)(z)
    Piecewise((1 - exp(-alpha*log(z/sigma) - beta*log(z/sigma)**2), sigma <= z),
            (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Benini_distribution
    .. [2] https://reference.wolfram.com/legacy/v8/ref/BeniniDistribution.html

    """

    return rv(name, BeniniDistribution, (alpha, beta, sigma))

#-------------------------------------------------------------------------------
# Beta distribution ------------------------------------------------------------


class BetaDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    set = Interval(0, 1)

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return x**(alpha - 1) * (1 - x)**(beta - 1) / beta_fn(alpha, beta)

    def _characteristic_function(self, t):
        return hyper((self.alpha,), (self.alpha + self.beta,), I*t)

    def _moment_generating_function(self, t):
        return hyper((self.alpha,), (self.alpha + self.beta,), t)


def Beta(name, alpha, beta):
    r"""
    Create a Continuous Random Variable with a Beta distribution.

    The density of the Beta distribution is given by

    .. math::
        f(x) := \frac{x^{\alpha-1}(1-x)^{\beta-1}} {\mathrm{B}(\alpha,\beta)}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
    beta : Real number, `\beta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Beta, density, E, variance
    >>> from sympy import Symbol, simplify, pprint, factor

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> z = Symbol("z")

    >>> X = Beta("x", alpha, beta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
     alpha - 1        beta - 1
    z         *(1 - z)
    --------------------------
          B(alpha, beta)

    >>> simplify(E(X))
    alpha/(alpha + beta)

    >>> factor(simplify(variance(X)))
    alpha*beta/((alpha + beta)**2*(alpha + beta + 1))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_distribution
    .. [2] https://mathworld.wolfram.com/BetaDistribution.html

    """

    return rv(name, BetaDistribution, (alpha, beta))

#-------------------------------------------------------------------------------
# Noncentral Beta distribution ------------------------------------------------------------


class BetaNoncentralDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta', 'lamda')

    set = Interval(0, 1)

    @staticmethod
    def check(alpha, beta, lamda):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")
        _value_check(lamda >= 0, "Noncentrality parameter Lambda must be positive")

    def pdf(self, x):
        alpha, beta, lamda = self.alpha, self.beta, self.lamda
        k = Dummy("k")
        return Sum(exp(-lamda / 2) * (lamda / 2)**k * x**(alpha + k - 1) *(
            1 - x)**(beta - 1) / (factorial(k) * beta_fn(alpha + k, beta)), (k, 0, oo))

def BetaNoncentral(name, alpha, beta, lamda):
    r"""
    Create a Continuous Random Variable with a Type I Noncentral Beta distribution.

    The density of the Noncentral Beta distribution is given by

    .. math::
        f(x) := \sum_{k=0}^\infty e^{-\lambda/2}\frac{(\lambda/2)^k}{k!}
                \frac{x^{\alpha+k-1}(1-x)^{\beta-1}}{\mathrm{B}(\alpha+k,\beta)}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
    beta : Real number, `\beta > 0`, a shape
    lamda : Real number, `\lambda \geq 0`, noncentrality parameter

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import BetaNoncentral, density, cdf
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> lamda = Symbol("lamda", nonnegative=True)
    >>> z = Symbol("z")

    >>> X = BetaNoncentral("x", alpha, beta, lamda)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
      oo
    _____
    \    `
     \                                              -lamda
      \                          k                  -------
       \    k + alpha - 1 /lamda\         beta - 1     2
        )  z             *|-----| *(1 - z)        *e
       /                  \  2  /
      /    ------------------------------------------------
     /                  B(k + alpha, beta)*k!
    /____,
    k = 0

    Compute cdf with specific 'x', 'alpha', 'beta' and 'lamda' values as follows:

    >>> cdf(BetaNoncentral("x", 1, 1, 1), evaluate=False)(2).doit()
    2*exp(1/2)

    The argument evaluate=False prevents an attempt at evaluation
    of the sum for general x, before the argument 2 is passed.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Noncentral_beta_distribution
    .. [2] https://reference.wolfram.com/language/ref/NoncentralBetaDistribution.html

    """

    return rv(name, BetaNoncentralDistribution, (alpha, beta, lamda))


#-------------------------------------------------------------------------------
# Beta prime distribution ------------------------------------------------------


class BetaPrimeDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")

    set = Interval(0, oo)

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return x**(alpha - 1)*(1 + x)**(-alpha - beta)/beta_fn(alpha, beta)

def BetaPrime(name, alpha, beta):
    r"""
    Create a continuous random variable with a Beta prime distribution.

    The density of the Beta prime distribution is given by

    .. math::
        f(x) := \frac{x^{\alpha-1} (1+x)^{-\alpha -\beta}}{B(\alpha,\beta)}

    with :math:`x > 0`.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
    beta : Real number, `\beta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import BetaPrime, density
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> z = Symbol("z")

    >>> X = BetaPrime("x", alpha, beta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
     alpha - 1        -alpha - beta
    z         *(z + 1)
    -------------------------------
             B(alpha, beta)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_prime_distribution
    .. [2] https://mathworld.wolfram.com/BetaPrimeDistribution.html

    """

    return rv(name, BetaPrimeDistribution, (alpha, beta))

#-------------------------------------------------------------------------------
# Bounded Pareto Distribution --------------------------------------------------
class BoundedParetoDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'left', 'right')

    @property
    def set(self):
        return Interval(self.left, self.right)

    @staticmethod
    def check(alpha, left, right):
        _value_check (alpha.is_positive, "Shape must be positive.")
        _value_check (left.is_positive, "Left value should be positive.")
        _value_check (right > left, "Right should be greater than left.")

    def pdf(self, x):
        alpha, left, right = self.alpha, self.left, self.right
        num = alpha * (left**alpha) * x**(- alpha -1)
        den = 1 - (left/right)**alpha
        return num/den

def BoundedPareto(name, alpha, left, right):
    r"""
    Create a continuous random variable with a Bounded Pareto distribution.

    The density of the Bounded Pareto distribution is given by

    .. math::
        f(x) := \frac{\alpha L^{\alpha}x^{-\alpha-1}}{1-(\frac{L}{H})^{\alpha}}

    Parameters
    ==========

    alpha : Real Number, `\alpha > 0`
        Shape parameter
    left : Real Number, `left > 0`
        Location parameter
    right : Real Number, `right > left`
        Location parameter

    Examples
    ========

    >>> from sympy.stats import BoundedPareto, density, cdf, E
    >>> from sympy import symbols
    >>> L, H = symbols('L, H', positive=True)
    >>> X = BoundedPareto('X', 2, L, H)
    >>> x = symbols('x')
    >>> density(X)(x)
    2*L**2/(x**3*(1 - L**2/H**2))
    >>> cdf(X)(x)
    Piecewise((-H**2*L**2/(x**2*(H**2 - L**2)) + H**2/(H**2 - L**2), L <= x), (0, True))
    >>> E(X).simplify()
    2*H*L/(H + L)

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution

    """
    return rv (name, BoundedParetoDistribution, (alpha, left, right))

# ------------------------------------------------------------------------------
# Cauchy distribution ----------------------------------------------------------


class CauchyDistribution(SingleContinuousDistribution):
    _argnames = ('x0', 'gamma')

    @staticmethod
    def check(x0, gamma):
        _value_check(gamma > 0, "Scale parameter Gamma must be positive.")
        _value_check(x0.is_real, "Location parameter must be real.")

    def pdf(self, x):
        return 1/(pi*self.gamma*(1 + ((x - self.x0)/self.gamma)**2))

    def _cdf(self, x):
        x0, gamma = self.x0, self.gamma
        return (1/pi)*atan((x - x0)/gamma) + S.Half

    def _characteristic_function(self, t):
        return exp(self.x0 * I * t -  self.gamma * Abs(t))

    def _moment_generating_function(self, t):
        raise NotImplementedError("The moment generating function for the "
                                  "Cauchy distribution does not exist.")

    def _quantile(self, p):
        return self.x0 + self.gamma*tan(pi*(p - S.Half))


def Cauchy(name, x0, gamma):
    r"""
    Create a continuous random variable with a Cauchy distribution.

    The density of the Cauchy distribution is given by

    .. math::
        f(x) := \frac{1}{\pi \gamma [1 + {(\frac{x-x_0}{\gamma})}^2]}

    Parameters
    ==========

    x0 : Real number, the location
    gamma : Real number, `\gamma > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Cauchy, density
    >>> from sympy import Symbol

    >>> x0 = Symbol("x0")
    >>> gamma = Symbol("gamma", positive=True)
    >>> z = Symbol("z")

    >>> X = Cauchy("x", x0, gamma)

    >>> density(X)(z)
    1/(pi*gamma*(1 + (-x0 + z)**2/gamma**2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cauchy_distribution
    .. [2] https://mathworld.wolfram.com/CauchyDistribution.html

    """

    return rv(name, CauchyDistribution, (x0, gamma))

#-------------------------------------------------------------------------------
# Chi distribution -------------------------------------------------------------


class ChiDistribution(SingleContinuousDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")

    set = Interval(0, oo)

    def pdf(self, x):
        return 2**(1 - self.k/2)*x**(self.k - 1)*exp(-x**2/2)/gamma(self.k/2)

    def _characteristic_function(self, t):
        k = self.k

        part_1 = hyper((k/2,), (S.Half,), -t**2/2)
        part_2 = I*t*sqrt(2)*gamma((k+1)/2)/gamma(k/2)
        part_3 = hyper(((k+1)/2,), (Rational(3, 2),), -t**2/2)
        return part_1 + part_2*part_3

    def _moment_generating_function(self, t):
        k = self.k

        part_1 = hyper((k / 2,), (S.Half,), t ** 2 / 2)
        part_2 = t * sqrt(2) * gamma((k + 1) / 2) / gamma(k / 2)
        part_3 = hyper(((k + 1) / 2,), (S(3) / 2,), t ** 2 / 2)
        return part_1 + part_2 * part_3

def Chi(name, k):
    r"""
    Create a continuous random variable with a Chi distribution.

    The density of the Chi distribution is given by

    .. math::
        f(x) := \frac{2^{1-k/2}x^{k-1}e^{-x^2/2}}{\Gamma(k/2)}

    with :math:`x \geq 0`.

    Parameters
    ==========

    k : Positive integer, The number of degrees of freedom

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Chi, density, E
    >>> from sympy import Symbol, simplify

    >>> k = Symbol("k", integer=True)
    >>> z = Symbol("z")

    >>> X = Chi("x", k)

    >>> density(X)(z)
    2**(1 - k/2)*z**(k - 1)*exp(-z**2/2)/gamma(k/2)

    >>> simplify(E(X))
    sqrt(2)*gamma(k/2 + 1/2)/gamma(k/2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chi_distribution
    .. [2] https://mathworld.wolfram.com/ChiDistribution.html

    """

    return rv(name, ChiDistribution, (k,))

#-------------------------------------------------------------------------------
# Non-central Chi distribution -------------------------------------------------


class ChiNoncentralDistribution(SingleContinuousDistribution):
    _argnames = ('k', 'l')

    @staticmethod
    def check(k, l):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")
        _value_check(l > 0, "Shift parameter Lambda must be positive.")

    set = Interval(0, oo)

    def pdf(self, x):
        k, l = self.k, self.l
        return exp(-(x**2+l**2)/2)*x**k*l / (l*x)**(k/2) * besseli(k/2-1, l*x)

def ChiNoncentral(name, k, l):
    r"""
    Create a continuous random variable with a non-central Chi distribution.

    Explanation
    ===========

    The density of the non-central Chi distribution is given by

    .. math::
        f(x) := \frac{e^{-(x^2+\lambda^2)/2} x^k\lambda}
                {(\lambda x)^{k/2}} I_{k/2-1}(\lambda x)

    with `x \geq 0`. Here, `I_\nu (x)` is the
    :ref:`modified Bessel function of the first kind <besseli>`.

    Parameters
    ==========

    k : A positive Integer, $k > 0$
        The number of degrees of freedom.
    lambda : Real number, `\lambda > 0`
        Shift parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ChiNoncentral, density
    >>> from sympy import Symbol

    >>> k = Symbol("k", integer=True)
    >>> l = Symbol("l")
    >>> z = Symbol("z")

    >>> X = ChiNoncentral("x", k, l)

    >>> density(X)(z)
    l*z**k*exp(-l**2/2 - z**2/2)*besseli(k/2 - 1, l*z)/(l*z)**(k/2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Noncentral_chi_distribution
    """

    return rv(name, ChiNoncentralDistribution, (k, l))

#-------------------------------------------------------------------------------
# Chi squared distribution -----------------------------------------------------


class ChiSquaredDistribution(SingleContinuousDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")

    set = Interval(0, oo)

    def pdf(self, x):
        k = self.k
        return 1/(2**(k/2)*gamma(k/2))*x**(k/2 - 1)*exp(-x/2)

    def _cdf(self, x):
        k = self.k
        return Piecewise(
                (S.One/gamma(k/2)*lowergamma(k/2, x/2), x >= 0),
                (0, True)
        )

    def _characteristic_function(self, t):
        return (1 - 2*I*t)**(-self.k/2)

    def  _moment_generating_function(self, t):
        return (1 - 2*t)**(-self.k/2)


def ChiSquared(name, k):
    r"""
    Create a continuous random variable with a Chi-squared distribution.

    Explanation
    ===========

    The density of the Chi-squared distribution is given by

    .. math::
        f(x) := \frac{1}{2^{\frac{k}{2}}\Gamma\left(\frac{k}{2}\right)}
                x^{\frac{k}{2}-1} e^{-\frac{x}{2}}

    with :math:`x \geq 0`.

    Parameters
    ==========

    k : Positive integer
        The number of degrees of freedom.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ChiSquared, density, E, variance, moment
    >>> from sympy import Symbol

    >>> k = Symbol("k", integer=True, positive=True)
    >>> z = Symbol("z")

    >>> X = ChiSquared("x", k)

    >>> density(X)(z)
    z**(k/2 - 1)*exp(-z/2)/(2**(k/2)*gamma(k/2))

    >>> E(X)
    k

    >>> variance(X)
    2*k

    >>> moment(X, 3)
    k**3 + 6*k**2 + 8*k

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chi_squared_distribution
    .. [2] https://mathworld.wolfram.com/Chi-SquaredDistribution.html
    """

    return rv(name, ChiSquaredDistribution, (k, ))

#-------------------------------------------------------------------------------
# Dagum distribution -----------------------------------------------------------


class DagumDistribution(SingleContinuousDistribution):
    _argnames = ('p', 'a', 'b')

    set = Interval(0, oo)

    @staticmethod
    def check(p, a, b):
        _value_check(p > 0, "Shape parameter p must be positive.")
        _value_check(a > 0, "Shape parameter a must be positive.")
        _value_check(b > 0, "Scale parameter b must be positive.")

    def pdf(self, x):
        p, a, b = self.p, self.a, self.b
        return a*p/x*((x/b)**(a*p)/(((x/b)**a + 1)**(p + 1)))

    def _cdf(self, x):
        p, a, b = self.p, self.a, self.b
        return Piecewise(((S.One + (S(x)/b)**-a)**-p, x>=0),
                    (S.Zero, True))

def Dagum(name, p, a, b):
    r"""
    Create a continuous random variable with a Dagum distribution.

    Explanation
    ===========

    The density of the Dagum distribution is given by

    .. math::
        f(x) := \frac{a p}{x} \left( \frac{\left(\tfrac{x}{b}\right)^{a p}}
                {\left(\left(\tfrac{x}{b}\right)^a + 1 \right)^{p+1}} \right)

    with :math:`x > 0`.

    Parameters
    ==========

    p : Real number
        `p > 0`, a shape.
    a : Real number
        `a > 0`, a shape.
    b : Real number
        `b > 0`, a scale.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Dagum, density, cdf
    >>> from sympy import Symbol

    >>> p = Symbol("p", positive=True)
    >>> a = Symbol("a", positive=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Dagum("x", p, a, b)

    >>> density(X)(z)
    a*p*(z/b)**(a*p)*((z/b)**a + 1)**(-p - 1)/z

    >>> cdf(X)(z)
    Piecewise(((1 + (z/b)**(-a))**(-p), z >= 0), (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dagum_distribution

    """

    return rv(name, DagumDistribution, (p, a, b))

#-------------------------------------------------------------------------------
# Davis distribution -----------------------------------------------------------

class DavisDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'n', 'mu')

    set = Interval(0, oo)

    @staticmethod
    def check(b, n, mu):
        _value_check(b > 0, "Scale parameter b must be positive.")
        _value_check(n > 1, "Shape parameter n must be above 1.")
        _value_check(mu > 0, "Location parameter mu must be positive.")

    def pdf(self, x):
        b, n, mu = self.b, self.n, self.mu
        dividend = b**n*(x - mu)**(-1-n)
        divisor = (exp(b/(x-mu))-1)*(gamma(n)*zeta(n))
        return dividend/divisor


def Davis(name, b, n, mu):
    r""" Create a continuous random variable with Davis distribution.

    Explanation
    ===========

    The density of Davis distribution is given by

    .. math::
        f(x; \mu; b, n) := \frac{b^{n}(x - \mu)^{1-n}}{ \left( e^{\frac{b}{x-\mu}} - 1 \right) \Gamma(n)\zeta(n)}

    with :math:`x \in [0,\infty]`.

    Davis distribution is a generalization of the Planck's law of radiation from statistical physics. It is used for modeling income distribution.

    Parameters
    ==========
    b : Real number
        `p > 0`, a scale.
    n : Real number
        `n > 1`, a shape.
    mu : Real number
        `mu > 0`, a location.

    Returns
    =======

    RandomSymbol

    Examples
    ========
    >>> from sympy.stats import Davis, density
    >>> from sympy import Symbol
    >>> b = Symbol("b", positive=True)
    >>> n = Symbol("n", positive=True)
    >>> mu = Symbol("mu", positive=True)
    >>> z = Symbol("z")
    >>> X = Davis("x", b, n, mu)
    >>> density(X)(z)
    b**n*(-mu + z)**(-n - 1)/((exp(b/(-mu + z)) - 1)*gamma(n)*zeta(n))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Davis_distribution
    .. [2] https://reference.wolfram.com/language/ref/DavisDistribution.html

    """
    return rv(name, DavisDistribution, (b, n, mu))


#-------------------------------------------------------------------------------
# Erlang distribution ----------------------------------------------------------


def Erlang(name, k, l):
    r"""
    Create a continuous random variable with an Erlang distribution.

    Explanation
    ===========

    The density of the Erlang distribution is given by

    .. math::
        f(x) := \frac{\lambda^k x^{k-1} e^{-\lambda x}}{(k-1)!}

    with :math:`x \in [0,\infty]`.

    Parameters
    ==========

    k : Positive integer
    l : Real number, `\lambda > 0`, the rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Erlang, density, cdf, E, variance
    >>> from sympy import Symbol, simplify, pprint

    >>> k = Symbol("k", integer=True, positive=True)
    >>> l = Symbol("l", positive=True)
    >>> z = Symbol("z")

    >>> X = Erlang("x", k, l)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
     k  k - 1  -l*z
    l *z     *e
    ---------------
        Gamma(k)

    >>> C = cdf(X)(z)
    >>> pprint(C, use_unicode=False)
    /lowergamma(k, l*z)
    |------------------  for z > 0
    <     Gamma(k)
    |
    \        0           otherwise


    >>> E(X)
    k/l

    >>> simplify(variance(X))
    k/l**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Erlang_distribution
    .. [2] https://mathworld.wolfram.com/ErlangDistribution.html

    """

    return rv(name, GammaDistribution, (k, S.One/l))

# -------------------------------------------------------------------------------
# ExGaussian distribution -----------------------------------------------------


class ExGaussianDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std', 'rate')

    set = Interval(-oo, oo)

    @staticmethod
    def check(mean, std, rate):
        _value_check(
            std > 0, "Standard deviation of ExGaussian must be positive.")
        _value_check(rate > 0, "Rate of ExGaussian must be positive.")

    def pdf(self, x):
        mean, std, rate = self.mean, self.std, self.rate
        term1 = rate/2
        term2 = exp(rate * (2 * mean + rate * std**2 - 2*x)/2)
        term3 = erfc((mean + rate*std**2 - x)/(sqrt(2)*std))
        return term1*term2*term3

    def _cdf(self, x):
        from sympy.stats import cdf
        mean, std, rate = self.mean, self.std, self.rate
        u = rate*(x - mean)
        v = rate*std
        GaussianCDF1 = cdf(Normal('x', 0, v))(u)
        GaussianCDF2 = cdf(Normal('x', v**2, v))(u)

        return GaussianCDF1 - exp(-u + (v**2/2) + log(GaussianCDF2))

    def _characteristic_function(self, t):
        mean, std, rate = self.mean, self.std, self.rate
        term1 = (1 - I*t/rate)**(-1)
        term2 = exp(I*mean*t - std**2*t**2/2)
        return term1 * term2

    def _moment_generating_function(self, t):
        mean, std, rate = self.mean, self.std, self.rate
        term1 = (1 - t/rate)**(-1)
        term2 = exp(mean*t + std**2*t**2/2)
        return term1*term2


def ExGaussian(name, mean, std, rate):
    r"""
    Create a continuous random variable with an Exponentially modified
    Gaussian (EMG) distribution.

    Explanation
    ===========

    The density of the exponentially modified Gaussian distribution is given by

    .. math::
        f(x) := \frac{\lambda}{2}e^{\frac{\lambda}{2}(2\mu+\lambda\sigma^2-2x)}
            \text{erfc}(\frac{\mu + \lambda\sigma^2 - x}{\sqrt{2}\sigma})

    with $x > 0$. Note that the expected value is `1/\lambda`.

    Parameters
    ==========

    name : A string giving a name for this distribution
    mean : A Real number, the mean of Gaussian component
    std : A positive Real number,
        :math: `\sigma^2 > 0` the variance of Gaussian component
    rate : A positive Real number,
        :math: `\lambda > 0` the rate of Exponential component

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ExGaussian, density, cdf, E
    >>> from sympy.stats import variance, skewness
    >>> from sympy import Symbol, pprint, simplify

    >>> mean = Symbol("mu")
    >>> std = Symbol("sigma", positive=True)
    >>> rate = Symbol("lamda", positive=True)
    >>> z = Symbol("z")
    >>> X = ExGaussian("x", mean, std, rate)

    >>> pprint(density(X)(z), use_unicode=False)
                 /           2             \
           lamda*\lamda*sigma  + 2*mu - 2*z/
           ---------------------------------     /  ___ /           2         \\
                           2                     |\/ 2 *\lamda*sigma  + mu - z/|
    lamda*e                                 *erfc|-----------------------------|
                                                 \           2*sigma           /
    ----------------------------------------------------------------------------
                                         2

    >>> cdf(X)(z)
    -(erf(sqrt(2)*(-lamda**2*sigma**2 + lamda*(-mu + z))/(2*lamda*sigma))/2 + 1/2)*exp(lamda**2*sigma**2/2 - lamda*(-mu + z)) + erf(sqrt(2)*(-mu + z)/(2*sigma))/2 + 1/2

    >>> E(X)
    (lamda*mu + 1)/lamda

    >>> simplify(variance(X))
    sigma**2 + lamda**(-2)

    >>> simplify(skewness(X))
    2/(lamda**2*sigma**2 + 1)**(3/2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    """
    return rv(name, ExGaussianDistribution, (mean, std, rate))

#-------------------------------------------------------------------------------
# Exponential distribution -----------------------------------------------------


class ExponentialDistribution(SingleContinuousDistribution):
    _argnames = ('rate',)

    set  = Interval(0, oo)

    @staticmethod
    def check(rate):
        _value_check(rate > 0, "Rate must be positive.")

    def pdf(self, x):
        return self.rate * exp(-self.rate*x)

    def _cdf(self, x):
        return Piecewise(
                (S.One - exp(-self.rate*x), x >= 0),
                (0, True),
        )

    def _characteristic_function(self, t):
        rate = self.rate
        return rate / (rate - I*t)

    def _moment_generating_function(self, t):
        rate = self.rate
        return rate / (rate - t)

    def _quantile(self, p):
        return -log(1-p)/self.rate


def Exponential(name, rate):
    r"""
    Create a continuous random variable with an Exponential distribution.

    Explanation
    ===========

    The density of the exponential distribution is given by

    .. math::
        f(x) := \lambda \exp(-\lambda x)

    with $x > 0$. Note that the expected value is `1/\lambda`.

    Parameters
    ==========

    rate : A positive Real number, `\lambda > 0`, the rate (or inverse scale/inverse mean)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Exponential, density, cdf, E
    >>> from sympy.stats import variance, std, skewness, quantile
    >>> from sympy import Symbol

    >>> l = Symbol("lambda", positive=True)
    >>> z = Symbol("z")
    >>> p = Symbol("p")
    >>> X = Exponential("x", l)

    >>> density(X)(z)
    lambda*exp(-lambda*z)

    >>> cdf(X)(z)
    Piecewise((1 - exp(-lambda*z), z >= 0), (0, True))

    >>> quantile(X)(p)
    -log(1 - p)/lambda

    >>> E(X)
    1/lambda

    >>> variance(X)
    lambda**(-2)

    >>> skewness(X)
    2

    >>> X = Exponential('x', 10)

    >>> density(X)(z)
    10*exp(-10*z)

    >>> E(X)
    1/10

    >>> std(X)
    1/10

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponential_distribution
    .. [2] https://mathworld.wolfram.com/ExponentialDistribution.html

    """

    return rv(name, ExponentialDistribution, (rate, ))


# -------------------------------------------------------------------------------
# Exponential Power distribution -----------------------------------------------------

class ExponentialPowerDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'alpha', 'beta')

    set = Interval(-oo, oo)

    @staticmethod
    def check(mu, alpha, beta):
        _value_check(alpha > 0, "Scale parameter alpha must be positive.")
        _value_check(beta > 0, "Shape parameter beta must be positive.")

    def pdf(self, x):
        mu, alpha, beta = self.mu, self.alpha, self.beta
        num = beta*exp(-(Abs(x - mu)/alpha)**beta)
        den = 2*alpha*gamma(1/beta)
        return num/den

    def _cdf(self, x):
        mu, alpha, beta = self.mu, self.alpha, self.beta
        num = lowergamma(1/beta, (Abs(x - mu) / alpha)**beta)
        den = 2*gamma(1/beta)
        return sign(x - mu)*num/den + S.Half


def ExponentialPower(name, mu, alpha, beta):
    r"""
    Create a Continuous Random Variable with Exponential Power distribution.
    This distribution is known also as Generalized Normal
    distribution version 1.

    Explanation
    ===========

    The density of the Exponential Power distribution is given by

    .. math::
        f(x) := \frac{\beta}{2\alpha\Gamma(\frac{1}{\beta})}
            e^{{-(\frac{|x - \mu|}{\alpha})^{\beta}}}

    with :math:`x \in [ - \infty, \infty ]`.

    Parameters
    ==========

    mu : Real number
        A location.
    alpha : Real number,`\alpha > 0`
        A  scale.
    beta : Real number, `\beta > 0`
        A shape.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ExponentialPower, density, cdf
    >>> from sympy import Symbol, pprint
    >>> z = Symbol("z")
    >>> mu = Symbol("mu")
    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> X = ExponentialPower("x", mu, alpha, beta)
    >>> pprint(density(X)(z), use_unicode=False)
                     beta
           /|mu - z|\
          -|--------|
           \ alpha  /
    beta*e
    ---------------------
                  / 1  \
     2*alpha*Gamma|----|
                  \beta/
    >>> cdf(X)(z)
    1/2 + lowergamma(1/beta, (Abs(mu - z)/alpha)**beta)*sign(-mu + z)/(2*gamma(1/beta))

    References
    ==========

    .. [1] https://reference.wolfram.com/language/ref/ExponentialPowerDistribution.html
    .. [2] https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1

    """
    return rv(name, ExponentialPowerDistribution, (mu, alpha, beta))


#-------------------------------------------------------------------------------
# F distribution ---------------------------------------------------------------


class FDistributionDistribution(SingleContinuousDistribution):
    _argnames = ('d1', 'd2')

    set = Interval(0, oo)

    @staticmethod
    def check(d1, d2):
        _value_check((d1 > 0, d1.is_integer),
            "Degrees of freedom d1 must be positive integer.")
        _value_check((d2 > 0, d2.is_integer),
            "Degrees of freedom d2 must be positive integer.")

    def pdf(self, x):
        d1, d2 = self.d1, self.d2
        return (sqrt((d1*x)**d1*d2**d2 / (d1*x+d2)**(d1+d2))
               / (x * beta_fn(d1/2, d2/2)))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the '
                                  'F-distribution does not exist.')

def FDistribution(name, d1, d2):
    r"""
    Create a continuous random variable with a F distribution.

    Explanation
    ===========

    The density of the F distribution is given by

    .. math::
        f(x) := \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}
                {(d_1 x + d_2)^{d_1 + d_2}}}}
                {x \mathrm{B} \left(\frac{d_1}{2}, \frac{d_2}{2}\right)}

    with :math:`x > 0`.

    Parameters
    ==========

    d1 : `d_1 > 0`, where `d_1` is the degrees of freedom (`n_1 - 1`)
    d2 : `d_2 > 0`, where `d_2` is the degrees of freedom (`n_2 - 1`)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import FDistribution, density
    >>> from sympy import Symbol, pprint

    >>> d1 = Symbol("d1", positive=True)
    >>> d2 = Symbol("d2", positive=True)
    >>> z = Symbol("z")

    >>> X = FDistribution("x", d1, d2)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
      d2
      --    ______________________________
      2    /       d1            -d1 - d2
    d2  *\/  (d1*z)  *(d1*z + d2)
    --------------------------------------
                    /d1  d2\
                 z*B|--, --|
                    \2   2 /

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/F-distribution
    .. [2] https://mathworld.wolfram.com/F-Distribution.html

    """

    return rv(name, FDistributionDistribution, (d1, d2))

#-------------------------------------------------------------------------------
# Fisher Z distribution --------------------------------------------------------

class FisherZDistribution(SingleContinuousDistribution):
    _argnames = ('d1', 'd2')

    set = Interval(-oo, oo)

    @staticmethod
    def check(d1, d2):
        _value_check(d1 > 0, "Degree of freedom d1 must be positive.")
        _value_check(d2 > 0, "Degree of freedom d2 must be positive.")

    def pdf(self, x):
        d1, d2 = self.d1, self.d2
        return (2*d1**(d1/2)*d2**(d2/2) / beta_fn(d1/2, d2/2) *
               exp(d1*x) / (d1*exp(2*x)+d2)**((d1+d2)/2))

def FisherZ(name, d1, d2):
    r"""
    Create a Continuous Random Variable with an Fisher's Z distribution.

    Explanation
    ===========

    The density of the Fisher's Z distribution is given by

    .. math::
        f(x) := \frac{2d_1^{d_1/2} d_2^{d_2/2}} {\mathrm{B}(d_1/2, d_2/2)}
                \frac{e^{d_1z}}{\left(d_1e^{2z}+d_2\right)^{\left(d_1+d_2\right)/2}}


    .. TODO - What is the difference between these degrees of freedom?

    Parameters
    ==========

    d1 : `d_1 > 0`
        Degree of freedom.
    d2 : `d_2 > 0`
        Degree of freedom.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import FisherZ, density
    >>> from sympy import Symbol, pprint

    >>> d1 = Symbol("d1", positive=True)
    >>> d2 = Symbol("d2", positive=True)
    >>> z = Symbol("z")

    >>> X = FisherZ("x", d1, d2)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                d1   d2
        d1   d2               - -- - --
        --   --                 2    2
        2    2  /    2*z     \           d1*z
    2*d1  *d2  *\d1*e    + d2/         *e
    -----------------------------------------
                     /d1  d2\
                    B|--, --|
                     \2   2 /

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fisher%27s_z-distribution
    .. [2] https://mathworld.wolfram.com/Fishersz-Distribution.html

    """

    return rv(name, FisherZDistribution, (d1, d2))

#-------------------------------------------------------------------------------
# Frechet distribution ---------------------------------------------------------

class FrechetDistribution(SingleContinuousDistribution):
    _argnames = ('a', 's', 'm')

    set = Interval(0, oo)

    @staticmethod
    def check(a, s, m):
        _value_check(a > 0, "Shape parameter alpha must be positive.")
        _value_check(s > 0, "Scale parameter s must be positive.")

    def __new__(cls, a, s=1, m=0):
        a, s, m = list(map(sympify, (a, s, m)))
        return Basic.__new__(cls, a, s, m)

    def pdf(self, x):
        a, s, m = self.a, self.s, self.m
        return a/s * ((x-m)/s)**(-1-a) * exp(-((x-m)/s)**(-a))

    def _cdf(self, x):
        a, s, m = self.a, self.s, self.m
        return Piecewise((exp(-((x-m)/s)**(-a)), x >= m),
                        (S.Zero, True))

def Frechet(name, a, s=1, m=0):
    r"""
    Create a continuous random variable with a Frechet distribution.

    Explanation
    ===========

    The density of the Frechet distribution is given by

    .. math::
        f(x) := \frac{\alpha}{s} \left(\frac{x-m}{s}\right)^{-1-\alpha}
                 e^{-(\frac{x-m}{s})^{-\alpha}}

    with :math:`x \geq m`.

    Parameters
    ==========

    a : Real number, :math:`a \in \left(0, \infty\right)` the shape
    s : Real number, :math:`s \in \left(0, \infty\right)` the scale
    m : Real number, :math:`m \in \left(-\infty, \infty\right)` the minimum

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Frechet, density, cdf
    >>> from sympy import Symbol

    >>> a = Symbol("a", positive=True)
    >>> s = Symbol("s", positive=True)
    >>> m = Symbol("m", real=True)
    >>> z = Symbol("z")

    >>> X = Frechet("x", a, s, m)

    >>> density(X)(z)
    a*((-m + z)/s)**(-a - 1)*exp(-1/((-m + z)/s)**a)/s

    >>> cdf(X)(z)
    Piecewise((exp(-1/((-m + z)/s)**a), m <= z), (0, True))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution

    """

    return rv(name, FrechetDistribution, (a, s, m))

#-------------------------------------------------------------------------------
# Gamma distribution -----------------------------------------------------------


class GammaDistribution(SingleContinuousDistribution):
    _argnames = ('k', 'theta')

    set = Interval(0, oo)

    @staticmethod
    def check(k, theta):
        _value_check(k > 0, "k must be positive")
        _value_check(theta > 0, "Theta must be positive")

    def pdf(self, x):
        k, theta = self.k, self.theta
        return x**(k - 1) * exp(-x/theta) / (gamma(k)*theta**k)

    def _cdf(self, x):
        k, theta = self.k, self.theta
        return Piecewise(
                    (lowergamma(k, S(x)/theta)/gamma(k), x > 0),
                    (S.Zero, True))

    def _characteristic_function(self, t):
        return (1 - self.theta*I*t)**(-self.k)

    def _moment_generating_function(self, t):
        return (1- self.theta*t)**(-self.k)


def Gamma(name, k, theta):
    r"""
    Create a continuous random variable with a Gamma distribution.

    Explanation
    ===========

    The density of the Gamma distribution is given by

    .. math::
        f(x) := \frac{1}{\Gamma(k) \theta^k} x^{k - 1} e^{-\frac{x}{\theta}}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    k : Real number, `k > 0`, a shape
    theta : Real number, `\theta > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gamma, density, cdf, E, variance
    >>> from sympy import Symbol, pprint, simplify

    >>> k = Symbol("k", positive=True)
    >>> theta = Symbol("theta", positive=True)
    >>> z = Symbol("z")

    >>> X = Gamma("x", k, theta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                      -z
                    -----
         -k  k - 1  theta
    theta  *z     *e
    ---------------------
           Gamma(k)

    >>> C = cdf(X, meijerg=True)(z)
    >>> pprint(C, use_unicode=False)
    /            /     z  \
    |k*lowergamma|k, -----|
    |            \   theta/
    <----------------------  for z >= 0
    |     Gamma(k + 1)
    |
    \          0             otherwise

    >>> E(X)
    k*theta

    >>> V = simplify(variance(X))
    >>> pprint(V, use_unicode=False)
           2
    k*theta


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_distribution
    .. [2] https://mathworld.wolfram.com/GammaDistribution.html

    """

    return rv(name, GammaDistribution, (k, theta))

#-------------------------------------------------------------------------------
# Inverse Gamma distribution ---------------------------------------------------


class GammaInverseDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    set = Interval(0, oo)

    @staticmethod
    def check(a, b):
        _value_check(a > 0, "alpha must be positive")
        _value_check(b > 0, "beta must be positive")

    def pdf(self, x):
        a, b = self.a, self.b
        return b**a/gamma(a) * x**(-a-1) * exp(-b/x)

    def _cdf(self, x):
        a, b = self.a, self.b
        return Piecewise((uppergamma(a,b/x)/gamma(a), x > 0),
                        (S.Zero, True))

    def _characteristic_function(self, t):
        a, b = self.a, self.b
        return 2 * (-I*b*t)**(a/2) * besselk(a, sqrt(-4*I*b*t)) / gamma(a)

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the '
                                  'gamma inverse distribution does not exist.')

def GammaInverse(name, a, b):
    r"""
    Create a continuous random variable with an inverse Gamma distribution.

    Explanation
    ===========

    The density of the inverse Gamma distribution is given by

    .. math::
        f(x) := \frac{\beta^\alpha}{\Gamma(\alpha)} x^{-\alpha - 1}
                \exp\left(\frac{-\beta}{x}\right)

    with :math:`x > 0`.

    Parameters
    ==========

    a : Real number, `a > 0`, a shape
    b : Real number, `b > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import GammaInverse, density, cdf
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a", positive=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = GammaInverse("x", a, b)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                -b
                ---
     a  -a - 1   z
    b *z      *e
    ---------------
       Gamma(a)

    >>> cdf(X)(z)
    Piecewise((uppergamma(a, b/z)/gamma(a), z > 0), (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse-gamma_distribution

    """

    return rv(name, GammaInverseDistribution, (a, b))


#-------------------------------------------------------------------------------
# Gumbel distribution (Maximum and Minimum) --------------------------------------------------------


class GumbelDistribution(SingleContinuousDistribution):
    _argnames = ('beta', 'mu', 'minimum')

    set = Interval(-oo, oo)

    @staticmethod
    def check(beta, mu, minimum):
        _value_check(beta > 0, "Scale parameter beta must be positive.")

    def pdf(self, x):
        beta, mu = self.beta, self.mu
        z = (x - mu)/beta
        f_max = (1/beta)*exp(-z - exp(-z))
        f_min = (1/beta)*exp(z - exp(z))
        return Piecewise((f_min, self.minimum), (f_max, not self.minimum))

    def _cdf(self, x):
        beta, mu = self.beta, self.mu
        z = (x - mu)/beta
        F_max = exp(-exp(-z))
        F_min = 1 - exp(-exp(z))
        return Piecewise((F_min, self.minimum), (F_max, not self.minimum))

    def _characteristic_function(self, t):
        cf_max = gamma(1 - I*self.beta*t) * exp(I*self.mu*t)
        cf_min = gamma(1 + I*self.beta*t) * exp(I*self.mu*t)
        return Piecewise((cf_min, self.minimum), (cf_max, not self.minimum))

    def _moment_generating_function(self, t):
        mgf_max = gamma(1 - self.beta*t) * exp(self.mu*t)
        mgf_min = gamma(1 + self.beta*t) * exp(self.mu*t)
        return Piecewise((mgf_min, self.minimum), (mgf_max, not self.minimum))

def Gumbel(name, beta, mu, minimum=False):
    r"""
    Create a Continuous Random Variable with Gumbel distribution.

    Explanation
    ===========

    The density of the Gumbel distribution is given by

    For Maximum

    .. math::
        f(x) := \dfrac{1}{\beta} \exp \left( -\dfrac{x-\mu}{\beta}
                - \exp \left( -\dfrac{x - \mu}{\beta} \right) \right)

    with :math:`x \in [ - \infty, \infty ]`.

    For Minimum

    .. math::
        f(x) := \frac{e^{- e^{\frac{- \mu + x}{\beta}} + \frac{- \mu + x}{\beta}}}{\beta}

    with :math:`x \in [ - \infty, \infty ]`.

    Parameters
    ==========

    mu : Real number, `\mu`, a location
    beta : Real number, `\beta > 0`, a scale
    minimum : Boolean, by default ``False``, set to ``True`` for enabling minimum distribution

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gumbel, density, cdf
    >>> from sympy import Symbol
    >>> x = Symbol("x")
    >>> mu = Symbol("mu")
    >>> beta = Symbol("beta", positive=True)
    >>> X = Gumbel("x", beta, mu)
    >>> density(X)(x)
    exp(-exp(-(-mu + x)/beta) - (-mu + x)/beta)/beta
    >>> cdf(X)(x)
    exp(-exp(-(-mu + x)/beta))

    References
    ==========

    .. [1] https://mathworld.wolfram.com/GumbelDistribution.html
    .. [2] https://en.wikipedia.org/wiki/Gumbel_distribution
    .. [3] https://web.archive.org/web/20200628222206/http://www.mathwave.com/help/easyfit/html/analyses/distributions/gumbel_max.html
    .. [4] https://web.archive.org/web/20200628222212/http://www.mathwave.com/help/easyfit/html/analyses/distributions/gumbel_min.html

    """
    return rv(name, GumbelDistribution, (beta, mu, minimum))

#-------------------------------------------------------------------------------
# Gompertz distribution --------------------------------------------------------

class GompertzDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'eta')

    set = Interval(0, oo)

    @staticmethod
    def check(b, eta):
        _value_check(b > 0, "b must be positive")
        _value_check(eta > 0, "eta must be positive")

    def pdf(self, x):
        eta, b = self.eta, self.b
        return b*eta*exp(b*x)*exp(eta)*exp(-eta*exp(b*x))

    def _cdf(self, x):
        eta, b = self.eta, self.b
        return 1 - exp(eta)*exp(-eta*exp(b*x))

    def _moment_generating_function(self, t):
        eta, b = self.eta, self.b
        return eta * exp(eta) * expint(t/b, eta)

def Gompertz(name, b, eta):
    r"""
    Create a Continuous Random Variable with Gompertz distribution.

    Explanation
    ===========

    The density of the Gompertz distribution is given by

    .. math::
        f(x) := b \eta e^{b x} e^{\eta} \exp \left(-\eta e^{bx} \right)

    with :math:`x \in [0, \infty)`.

    Parameters
    ==========

    b : Real number, `b > 0`, a scale
    eta : Real number, `\eta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gompertz, density
    >>> from sympy import Symbol

    >>> b = Symbol("b", positive=True)
    >>> eta = Symbol("eta", positive=True)
    >>> z = Symbol("z")

    >>> X = Gompertz("x", b, eta)

    >>> density(X)(z)
    b*eta*exp(eta)*exp(b*z)*exp(-eta*exp(b*z))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gompertz_distribution

    """
    return rv(name, GompertzDistribution, (b, eta))

#-------------------------------------------------------------------------------
# Kumaraswamy distribution -----------------------------------------------------


class KumaraswamyDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    set = Interval(0, oo)

    @staticmethod
    def check(a, b):
        _value_check(a > 0, "a must be positive")
        _value_check(b > 0, "b must be positive")

    def pdf(self, x):
        a, b = self.a, self.b
        return a * b * x**(a-1) * (1-x**a)**(b-1)

    def _cdf(self, x):
        a, b = self.a, self.b
        return Piecewise(
            (S.Zero, x < S.Zero),
            (1 - (1 - x**a)**b, x <= S.One),
            (S.One, True))

def Kumaraswamy(name, a, b):
    r"""
    Create a Continuous Random Variable with a Kumaraswamy distribution.

    Explanation
    ===========

    The density of the Kumaraswamy distribution is given by

    .. math::
        f(x) := a b x^{a-1} (1-x^a)^{b-1}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    a : Real number, `a > 0`, a shape
    b : Real number, `b > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Kumaraswamy, density, cdf
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a", positive=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Kumaraswamy("x", a, b)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                       b - 1
         a - 1 /     a\
    a*b*z     *\1 - z /

    >>> cdf(X)(z)
    Piecewise((0, z < 0), (1 - (1 - z**a)**b, z <= 1), (1, True))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kumaraswamy_distribution

    """

    return rv(name, KumaraswamyDistribution, (a, b))

#-------------------------------------------------------------------------------
# Laplace distribution ---------------------------------------------------------


class LaplaceDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'b')

    set = Interval(-oo, oo)

    @staticmethod
    def check(mu, b):
        _value_check(b > 0, "Scale parameter b must be positive.")
        _value_check(mu.is_real, "Location parameter mu should be real")

    def pdf(self, x):
        mu, b = self.mu, self.b
        return 1/(2*b)*exp(-Abs(x - mu)/b)

    def _cdf(self, x):
        mu, b = self.mu, self.b
        return Piecewise(
                    (S.Half*exp((x - mu)/b), x < mu),
                    (S.One - S.Half*exp(-(x - mu)/b), x >= mu)
                        )

    def _characteristic_function(self, t):
        return exp(self.mu*I*t) / (1 + self.b**2*t**2)

    def _moment_generating_function(self, t):
        return exp(self.mu*t) / (1 - self.b**2*t**2)

def Laplace(name, mu, b):
    r"""
    Create a continuous random variable with a Laplace distribution.

    Explanation
    ===========

    The density of the Laplace distribution is given by

    .. math::
        f(x) := \frac{1}{2 b} \exp \left(-\frac{|x-\mu|}b \right)

    Parameters
    ==========

    mu : Real number or a list/matrix, the location (mean) or the
        location vector
    b : Real number or a positive definite matrix, representing a scale
        or the covariance matrix.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Laplace, density, cdf
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu")
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Laplace("x", mu, b)

    >>> density(X)(z)
    exp(-Abs(mu - z)/b)/(2*b)

    >>> cdf(X)(z)
    Piecewise((exp((-mu + z)/b)/2, mu > z), (1 - exp((mu - z)/b)/2, True))

    >>> L = Laplace('L', [1, 2], [[1, 0], [0, 1]])
    >>> pprint(density(L)(1, 2), use_unicode=False)
     5        /     ____\
    e *besselk\0, \/ 35 /
    ---------------------
              pi

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laplace_distribution
    .. [2] https://mathworld.wolfram.com/LaplaceDistribution.html

    """

    if isinstance(mu, (list, MatrixBase)) and\
        isinstance(b, (list, MatrixBase)):
        from sympy.stats.joint_rv_types import MultivariateLaplace
        return MultivariateLaplace(name, mu, b)

    return rv(name, LaplaceDistribution, (mu, b))

#-------------------------------------------------------------------------------
# Levy distribution ---------------------------------------------------------


class LevyDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'c')

    @property
    def set(self):
        return Interval(self.mu, oo)

    @staticmethod
    def check(mu, c):
        _value_check(c > 0, "c (scale parameter) must be positive")
        _value_check(mu.is_real, "mu (location parameter) must be real")

    def pdf(self, x):
        mu, c = self.mu, self.c
        return sqrt(c/(2*pi))*exp(-c/(2*(x - mu)))/((x - mu)**(S.One + S.Half))

    def _cdf(self, x):
        mu, c = self.mu, self.c
        return erfc(sqrt(c/(2*(x - mu))))

    def _characteristic_function(self, t):
        mu, c = self.mu, self.c
        return exp(I * mu * t - sqrt(-2 * I * c * t))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function of Levy distribution does not exist.')

def Levy(name, mu, c):
    r"""
    Create a continuous random variable with a Levy distribution.

    The density of the Levy distribution is given by

    .. math::
        f(x) := \sqrt(\frac{c}{2 \pi}) \frac{\exp -\frac{c}{2 (x - \mu)}}{(x - \mu)^{3/2}}

    Parameters
    ==========

    mu : Real number
        The location parameter.
    c : Real number, `c > 0`
        A scale parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Levy, density, cdf
    >>> from sympy import Symbol

    >>> mu = Symbol("mu", real=True)
    >>> c = Symbol("c", positive=True)
    >>> z = Symbol("z")

    >>> X = Levy("x", mu, c)

    >>> density(X)(z)
    sqrt(2)*sqrt(c)*exp(-c/(-2*mu + 2*z))/(2*sqrt(pi)*(-mu + z)**(3/2))

    >>> cdf(X)(z)
    erfc(sqrt(c)*sqrt(1/(-2*mu + 2*z)))

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/L%C3%A9vy_distribution
    .. [2] https://mathworld.wolfram.com/LevyDistribution.html
    """

    return rv(name, LevyDistribution, (mu, c))

#-------------------------------------------------------------------------------
# Log-Cauchy distribution --------------------------------------------------------


class LogCauchyDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'sigma')

    set = Interval.open(0, oo)

    @staticmethod
    def check(mu, sigma):
        _value_check((sigma > 0) != False, "Scale parameter Gamma must be positive.")
        _value_check(mu.is_real != False, "Location parameter must be real.")

    def pdf(self, x):
        mu, sigma = self.mu, self.sigma
        return 1/(x*pi)*(sigma/((log(x) - mu)**2 + sigma**2))

    def _cdf(self, x):
        mu, sigma = self.mu, self.sigma
        return (1/pi)*atan((log(x) - mu)/sigma) + S.Half

    def _characteristic_function(self, t):
        raise NotImplementedError("The characteristic function for the "
                                  "Log-Cauchy distribution does not exist.")

    def _moment_generating_function(self, t):
        raise NotImplementedError("The moment generating function for the "
                                  "Log-Cauchy distribution does not exist.")

def LogCauchy(name, mu, sigma):
    r"""
    Create a continuous random variable with a Log-Cauchy distribution.
    The density of the Log-Cauchy distribution is given by

    .. math::
        f(x) := \frac{1}{\pi x} \frac{\sigma}{(log(x)-\mu^2) + \sigma^2}

    Parameters
    ==========

    mu : Real number, the location

    sigma : Real number, `\sigma > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogCauchy, density, cdf
    >>> from sympy import Symbol, S

    >>> mu = 2
    >>> sigma = S.One / 5
    >>> z = Symbol("z")

    >>> X = LogCauchy("x", mu, sigma)

    >>> density(X)(z)
    1/(5*pi*z*((log(z) - 2)**2 + 1/25))

    >>> cdf(X)(z)
    atan(5*log(z) - 10)/pi + 1/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Log-Cauchy_distribution
    """

    return rv(name, LogCauchyDistribution, (mu, sigma))


#-------------------------------------------------------------------------------
# Logistic distribution --------------------------------------------------------


class LogisticDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')

    set = Interval(-oo, oo)

    @staticmethod
    def check(mu, s):
        _value_check(s > 0, "Scale parameter s must be positive.")

    def pdf(self, x):
        mu, s = self.mu, self.s
        return exp(-(x - mu)/s)/(s*(1 + exp(-(x - mu)/s))**2)

    def _cdf(self, x):
        mu, s = self.mu, self.s
        return S.One/(1 + exp(-(x - mu)/s))

    def _characteristic_function(self, t):
        return Piecewise((exp(I*t*self.mu) * pi*self.s*t / sinh(pi*self.s*t), Ne(t, 0)), (S.One, True))

    def _moment_generating_function(self, t):
        return exp(self.mu*t) * beta_fn(1 - self.s*t, 1 + self.s*t)

    def _quantile(self, p):
        return self.mu - self.s*log(-S.One + S.One/p)

def Logistic(name, mu, s):
    r"""
    Create a continuous random variable with a logistic distribution.

    Explanation
    ===========

    The density of the logistic distribution is given by

    .. math::
        f(x) := \frac{e^{-(x-\mu)/s}} {s\left(1+e^{-(x-\mu)/s}\right)^2}

    Parameters
    ==========

    mu : Real number, the location (mean)
    s : Real number, `s > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Logistic, density, cdf
    >>> from sympy import Symbol

    >>> mu = Symbol("mu", real=True)
    >>> s = Symbol("s", positive=True)
    >>> z = Symbol("z")

    >>> X = Logistic("x", mu, s)

    >>> density(X)(z)
    exp((mu - z)/s)/(s*(exp((mu - z)/s) + 1)**2)

    >>> cdf(X)(z)
    1/(exp((mu - z)/s) + 1)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logistic_distribution
    .. [2] https://mathworld.wolfram.com/LogisticDistribution.html

    """

    return rv(name, LogisticDistribution, (mu, s))

#-------------------------------------------------------------------------------
# Log-logistic distribution --------------------------------------------------------


class LogLogisticDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    set = Interval(0, oo)

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Scale parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")

    def pdf(self, x):
        a, b = self.alpha, self.beta
        return ((b/a)*(x/a)**(b - 1))/(1 + (x/a)**b)**2

    def _cdf(self, x):
        a, b = self.alpha, self.beta
        return 1/(1 + (x/a)**(-b))

    def _quantile(self, p):
        a, b = self.alpha, self.beta
        return a*((p/(1 - p))**(1/b))

    def expectation(self, expr, var, **kwargs):
        a, b = self.args
        return Piecewise((S.NaN, b <= 1), (pi*a/(b*sin(pi/b)), True))

def LogLogistic(name, alpha, beta):
    r"""
    Create a continuous random variable with a log-logistic distribution.
    The distribution is unimodal when ``beta > 1``.

    Explanation
    ===========

    The density of the log-logistic distribution is given by

    .. math::
        f(x) := \frac{(\frac{\beta}{\alpha})(\frac{x}{\alpha})^{\beta - 1}}
                {(1 + (\frac{x}{\alpha})^{\beta})^2}

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, scale parameter and median of distribution
    beta : Real number, `\beta > 0`, a shape parameter

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogLogistic, density, cdf, quantile
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> p = Symbol("p")
    >>> z = Symbol("z", positive=True)

    >>> X = LogLogistic("x", alpha, beta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                  beta - 1
           /  z  \
      beta*|-----|
           \alpha/
    ------------------------
                           2
          /       beta    \
          |/  z  \        |
    alpha*||-----|     + 1|
          \\alpha/        /

    >>> cdf(X)(z)
    1/(1 + (z/alpha)**(-beta))

    >>> quantile(X)(p)
    alpha*(p/(1 - p))**(1/beta)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Log-logistic_distribution

    """

    return rv(name, LogLogisticDistribution, (alpha, beta))

#-------------------------------------------------------------------------------
#Logit-Normal distribution------------------------------------------------------

class LogitNormalDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')
    set = Interval.open(0, 1)

    @staticmethod
    def check(mu, s):
        _value_check((s ** 2).is_real is not False and s ** 2 > 0, "Squared scale parameter s must be positive.")
        _value_check(mu.is_real is not False, "Location parameter must be real")

    def _logit(self, x):
        return log(x / (1 - x))

    def pdf(self, x):
        mu, s = self.mu, self.s
        return exp(-(self._logit(x) - mu)**2/(2*s**2))*(S.One/sqrt(2*pi*(s**2)))*(1/(x*(1 - x)))

    def _cdf(self, x):
        mu, s = self.mu, self.s
        return (S.One/2)*(1 + erf((self._logit(x) - mu)/(sqrt(2*s**2))))


def LogitNormal(name, mu, s):
    r"""
    Create a continuous random variable with a Logit-Normal distribution.

    The density of the logistic distribution is given by

    .. math::
        f(x) := \frac{1}{s \sqrt{2 \pi}} \frac{1}{x(1 - x)} e^{- \frac{(logit(x)  - \mu)^2}{s^2}}
        where logit(x) = \log(\frac{x}{1 - x})
    Parameters
    ==========

    mu : Real number, the location (mean)
    s : Real number, `s > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogitNormal, density, cdf
    >>> from sympy import Symbol,pprint

    >>> mu = Symbol("mu", real=True)
    >>> s = Symbol("s", positive=True)
    >>> z = Symbol("z")
    >>> X = LogitNormal("x",mu,s)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                              2
            /         /  z  \\
           -|-mu + log|-----||
            \         \1 - z//
           ---------------------
                       2
      ___           2*s
    \/ 2 *e
    ----------------------------
            ____
        2*\/ pi *s*z*(1 - z)

    >>> density(X)(z)
    sqrt(2)*exp(-(-mu + log(z/(1 - z)))**2/(2*s**2))/(2*sqrt(pi)*s*z*(1 - z))

    >>> cdf(X)(z)
    erf(sqrt(2)*(-mu + log(z/(1 - z)))/(2*s))/2 + 1/2


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logit-normal_distribution

    """

    return rv(name, LogitNormalDistribution, (mu, s))

#-------------------------------------------------------------------------------
# Log Normal distribution ------------------------------------------------------


class LogNormalDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std')

    set = Interval(0, oo)

    @staticmethod
    def check(mean, std):
        _value_check(std > 0, "Parameter std must be positive.")

    def pdf(self, x):
        mean, std = self.mean, self.std
        return exp(-(log(x) - mean)**2 / (2*std**2)) / (x*sqrt(2*pi)*std)

    def _cdf(self, x):
        mean, std = self.mean, self.std
        return Piecewise(
                (S.Half + S.Half*erf((log(x) - mean)/sqrt(2)/std), x > 0),
                (S.Zero, True)
        )

    def _moment_generating_function(self, t):
        raise NotImplementedError('Moment generating function of the log-normal distribution is not defined.')


def LogNormal(name, mean, std):
    r"""
    Create a continuous random variable with a log-normal distribution.

    Explanation
    ===========

    The density of the log-normal distribution is given by

    .. math::
        f(x) := \frac{1}{x\sqrt{2\pi\sigma^2}}
                e^{-\frac{\left(\ln x-\mu\right)^2}{2\sigma^2}}

    with :math:`x \geq 0`.

    Parameters
    ==========

    mu : Real number
        The log-scale.
    sigma : Real number
        A shape. ($\sigma^2 > 0$)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogNormal, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", real=True)
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = LogNormal("x", mu, sigma)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                          2
           -(-mu + log(z))
           -----------------
                      2
      ___      2*sigma
    \/ 2 *e
    ------------------------
            ____
        2*\/ pi *sigma*z


    >>> X = LogNormal('x', 0, 1) # Mean 0, standard deviation 1

    >>> density(X)(z)
    sqrt(2)*exp(-log(z)**2/2)/(2*sqrt(pi)*z)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lognormal
    .. [2] https://mathworld.wolfram.com/LogNormalDistribution.html

    """

    return rv(name, LogNormalDistribution, (mean, std))

#-------------------------------------------------------------------------------
# Lomax Distribution -----------------------------------------------------------

class LomaxDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'lamda',)
    set = Interval(0, oo)

    @staticmethod
    def check(alpha, lamda):
        _value_check(alpha.is_real, "Shape parameter should be real.")
        _value_check(lamda.is_real, "Scale parameter should be real.")
        _value_check(alpha.is_positive, "Shape parameter should be positive.")
        _value_check(lamda.is_positive, "Scale parameter should be positive.")

    def pdf(self, x):
        lamba, alpha = self.lamda, self.alpha
        return (alpha/lamba) * (S.One + x/lamba)**(-alpha-1)

def Lomax(name, alpha, lamda):
    r"""
    Create a continuous random variable with a Lomax distribution.

    Explanation
    ===========

    The density of the Lomax distribution is given by

    .. math::
        f(x) := \frac{\alpha}{\lambda}\left[1+\frac{x}{\lambda}\right]^{-(\alpha+1)}

    Parameters
    ==========

    alpha : Real Number, `\alpha > 0`
        Shape parameter
    lamda : Real Number, `\lambda > 0`
        Scale parameter

    Examples
    ========

    >>> from sympy.stats import Lomax, density, cdf, E
    >>> from sympy import symbols
    >>> a, l = symbols('a, l', positive=True)
    >>> X = Lomax('X', a, l)
    >>> x = symbols('x')
    >>> density(X)(x)
    a*(1 + x/l)**(-a - 1)/l
    >>> cdf(X)(x)
    Piecewise((1 - 1/(1 + x/l)**a, x >= 0), (0, True))
    >>> a = 2
    >>> X = Lomax('X', a, l)
    >>> E(X)
    l

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lomax_distribution

    """
    return rv(name, LomaxDistribution, (alpha, lamda))

#-------------------------------------------------------------------------------
# Maxwell distribution ---------------------------------------------------------


class MaxwellDistribution(SingleContinuousDistribution):
    _argnames = ('a',)

    set = Interval(0, oo)

    @staticmethod
    def check(a):
        _value_check(a > 0, "Parameter a must be positive.")

    def pdf(self, x):
        a = self.a
        return sqrt(2/pi)*x**2*exp(-x**2/(2*a**2))/a**3

    def _cdf(self, x):
        a = self.a
        return erf(sqrt(2)*x/(2*a)) - sqrt(2)*x*exp(-x**2/(2*a**2))/(sqrt(pi)*a)

def Maxwell(name, a):
    r"""
    Create a continuous random variable with a Maxwell distribution.

    Explanation
    ===========

    The density of the Maxwell distribution is given by

    .. math::
        f(x) := \sqrt{\frac{2}{\pi}} \frac{x^2 e^{-x^2/(2a^2)}}{a^3}

    with :math:`x \geq 0`.

    .. TODO - what does the parameter mean?

    Parameters
    ==========

    a : Real number, `a > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Maxwell, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> a = Symbol("a", positive=True)
    >>> z = Symbol("z")

    >>> X = Maxwell("x", a)

    >>> density(X)(z)
    sqrt(2)*z**2*exp(-z**2/(2*a**2))/(sqrt(pi)*a**3)

    >>> E(X)
    2*sqrt(2)*a/sqrt(pi)

    >>> simplify(variance(X))
    a**2*(-8 + 3*pi)/pi

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Maxwell_distribution
    .. [2] https://mathworld.wolfram.com/MaxwellDistribution.html

    """

    return rv(name, MaxwellDistribution, (a, ))

#-------------------------------------------------------------------------------
# Moyal Distribution -----------------------------------------------------------
class MoyalDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'sigma')

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.is_real, "Location parameter must be real.")
        _value_check(sigma.is_real and sigma > 0, "Scale parameter must be real\
        and positive.")

    def pdf(self, x):
        mu, sigma = self.mu, self.sigma
        num = exp(-(exp(-(x - mu)/sigma) + (x - mu)/(sigma))/2)
        den = (sqrt(2*pi) * sigma)
        return num/den

    def _characteristic_function(self, t):
        mu, sigma = self.mu, self.sigma
        term1 = exp(I*t*mu)
        term2 = (2**(-I*sigma*t) * gamma(Rational(1, 2) - I*t*sigma))
        return (term1 * term2)/sqrt(pi)

    def _moment_generating_function(self, t):
        mu, sigma = self.mu, self.sigma
        term1 = exp(t*mu)
        term2 = (2**(-1*sigma*t) * gamma(Rational(1, 2) - t*sigma))
        return (term1 * term2)/sqrt(pi)

def Moyal(name, mu, sigma):
    r"""
    Create a continuous random variable with a Moyal distribution.

    Explanation
    ===========

    The density of the Moyal distribution is given by

    .. math::
        f(x) := \frac{\exp-\frac{1}{2}\exp-\frac{x-\mu}{\sigma}-\frac{x-\mu}{2\sigma}}{\sqrt{2\pi}\sigma}

    with :math:`x \in \mathbb{R}`.

    Parameters
    ==========

    mu : Real number
        Location parameter
    sigma : Real positive number
        Scale parameter

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Moyal, density, cdf
    >>> from sympy import Symbol, simplify
    >>> mu = Symbol("mu", real=True)
    >>> sigma = Symbol("sigma", positive=True, real=True)
    >>> z = Symbol("z")
    >>> X = Moyal("x", mu, sigma)
    >>> density(X)(z)
    sqrt(2)*exp(-exp((mu - z)/sigma)/2 - (-mu + z)/(2*sigma))/(2*sqrt(pi)*sigma)
    >>> simplify(cdf(X)(z))
    1 - erf(sqrt(2)*exp((mu - z)/(2*sigma))/2)

    References
    ==========

    .. [1] https://reference.wolfram.com/language/ref/MoyalDistribution.html
    .. [2] https://www.stat.rice.edu/~dobelman/textfiles/DistributionsHandbook.pdf

    """

    return rv(name, MoyalDistribution, (mu, sigma))

#-------------------------------------------------------------------------------
# Nakagami distribution --------------------------------------------------------


class NakagamiDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'omega')

    set = Interval(0, oo)

    @staticmethod
    def check(mu, omega):
        _value_check(mu >= S.Half, "Shape parameter mu must be greater than equal to 1/2.")
        _value_check(omega > 0, "Spread parameter omega must be positive.")

    def pdf(self, x):
        mu, omega = self.mu, self.omega
        return 2*mu**mu/(gamma(mu)*omega**mu)*x**(2*mu - 1)*exp(-mu/omega*x**2)

    def _cdf(self, x):
        mu, omega = self.mu, self.omega
        return Piecewise(
                    (lowergamma(mu, (mu/omega)*x**2)/gamma(mu), x > 0),
                    (S.Zero, True))

def Nakagami(name, mu, omega):
    r"""
    Create a continuous random variable with a Nakagami distribution.

    Explanation
    ===========

    The density of the Nakagami distribution is given by

    .. math::
        f(x) := \frac{2\mu^\mu}{\Gamma(\mu)\omega^\mu} x^{2\mu-1}
                \exp\left(-\frac{\mu}{\omega}x^2 \right)

    with :math:`x > 0`.

    Parameters
    ==========

    mu : Real number, `\mu \geq \frac{1}{2}`, a shape
    omega : Real number, `\omega > 0`, the spread

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Nakagami, density, E, variance, cdf
    >>> from sympy import Symbol, simplify, pprint

    >>> mu = Symbol("mu", positive=True)
    >>> omega = Symbol("omega", positive=True)
    >>> z = Symbol("z")

    >>> X = Nakagami("x", mu, omega)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                    2
                               -mu*z
                               -------
        mu      -mu  2*mu - 1  omega
    2*mu  *omega   *z        *e
    ----------------------------------
                Gamma(mu)

    >>> simplify(E(X))
    sqrt(mu)*sqrt(omega)*gamma(mu + 1/2)/gamma(mu + 1)

    >>> V = simplify(variance(X))
    >>> pprint(V, use_unicode=False)
                        2
             omega*Gamma (mu + 1/2)
    omega - -----------------------
            Gamma(mu)*Gamma(mu + 1)

    >>> cdf(X)(z)
    Piecewise((lowergamma(mu, mu*z**2/omega)/gamma(mu), z > 0),
            (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Nakagami_distribution

    """

    return rv(name, NakagamiDistribution, (mu, omega))

#-------------------------------------------------------------------------------
# Normal distribution ----------------------------------------------------------


class NormalDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std')

    @staticmethod
    def check(mean, std):
        _value_check(std > 0, "Standard deviation must be positive")

    def pdf(self, x):
        return exp(-(x - self.mean)**2 / (2*self.std**2)) / (sqrt(2*pi)*self.std)

    def _cdf(self, x):
        mean, std = self.mean, self.std
        return erf(sqrt(2)*(-mean + x)/(2*std))/2 + S.Half

    def _characteristic_function(self, t):
        mean, std = self.mean, self.std
        return exp(I*mean*t - std**2*t**2/2)

    def _moment_generating_function(self, t):
        mean, std = self.mean, self.std
        return exp(mean*t + std**2*t**2/2)

    def _quantile(self, p):
        mean, std = self.mean, self.std
        return mean + std*sqrt(2)*erfinv(2*p - 1)


def Normal(name, mean, std):
    r"""
    Create a continuous random variable with a Normal distribution.

    Explanation
    ===========

    The density of the Normal distribution is given by

    .. math::
        f(x) := \frac{1}{\sigma\sqrt{2\pi}} e^{ -\frac{(x-\mu)^2}{2\sigma^2} }

    Parameters
    ==========

    mu : Real number or a list representing the mean or the mean vector
    sigma : Real number or a positive definite square matrix,
         :math:`\sigma^2 > 0`, the variance

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Normal, density, E, std, cdf, skewness, quantile, marginal_distribution
    >>> from sympy import Symbol, simplify, pprint

    >>> mu = Symbol("mu")
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")
    >>> y = Symbol("y")
    >>> p = Symbol("p")
    >>> X = Normal("x", mu, sigma)

    >>> density(X)(z)
    sqrt(2)*exp(-(-mu + z)**2/(2*sigma**2))/(2*sqrt(pi)*sigma)

    >>> C = simplify(cdf(X))(z) # it needs a little more help...
    >>> pprint(C, use_unicode=False)
       /  ___          \
       |\/ 2 *(-mu + z)|
    erf|---------------|
       \    2*sigma    /   1
    -------------------- + -
             2             2

    >>> quantile(X)(p)
    mu + sqrt(2)*sigma*erfinv(2*p - 1)

    >>> simplify(skewness(X))
    0

    >>> X = Normal("x", 0, 1) # Mean 0, standard deviation 1
    >>> density(X)(z)
    sqrt(2)*exp(-z**2/2)/(2*sqrt(pi))

    >>> E(2*X + 1)
    1

    >>> simplify(std(2*X + 1))
    2

    >>> m = Normal('X', [1, 2], [[2, 1], [1, 2]])
    >>> pprint(density(m)(y, z), use_unicode=False)
              2          2
             y    y*z   z
           - -- + --- - -- + z - 1
      ___    3     3    3
    \/ 3 *e
    ------------------------------
                 6*pi

    >>> marginal_distribution(m, m[0])(1)
     1/(2*sqrt(pi))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal_distribution
    .. [2] https://mathworld.wolfram.com/NormalDistributionFunction.html

    """

    if isinstance(mean, list) or getattr(mean, 'is_Matrix', False) and\
        isinstance(std, list) or getattr(std, 'is_Matrix', False):
        from sympy.stats.joint_rv_types import MultivariateNormal
        return MultivariateNormal(name, mean, std)
    return rv(name, NormalDistribution, (mean, std))


#-------------------------------------------------------------------------------
# Inverse Gaussian distribution ----------------------------------------------------------


class GaussianInverseDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'shape')

    @property
    def set(self):
        return Interval(0, oo)

    @staticmethod
    def check(mean, shape):
        _value_check(shape > 0, "Shape parameter must be positive")
        _value_check(mean > 0, "Mean must be positive")

    def pdf(self, x):
        mu, s = self.mean, self.shape
        return exp(-s*(x - mu)**2 / (2*x*mu**2)) * sqrt(s/(2*pi*x**3))

    def _cdf(self, x):
        from sympy.stats import cdf
        mu, s = self.mean, self.shape
        stdNormalcdf = cdf(Normal('x', 0, 1))

        first_term = stdNormalcdf(sqrt(s/x) * ((x/mu) - S.One))
        second_term = exp(2*s/mu) * stdNormalcdf(-sqrt(s/x)*(x/mu + S.One))

        return  first_term + second_term

    def _characteristic_function(self, t):
        mu, s = self.mean, self.shape
        return exp((s/mu)*(1 - sqrt(1 - (2*mu**2*I*t)/s)))

    def _moment_generating_function(self, t):
        mu, s = self.mean, self.shape
        return exp((s/mu)*(1 - sqrt(1 - (2*mu**2*t)/s)))


def GaussianInverse(name, mean, shape):
    r"""
    Create a continuous random variable with an Inverse Gaussian distribution.
    Inverse Gaussian distribution is also known as Wald distribution.

    Explanation
    ===========

    The density of the Inverse Gaussian distribution is given by

    .. math::
        f(x) := \sqrt{\frac{\lambda}{2\pi x^3}} e^{-\frac{\lambda(x-\mu)^2}{2x\mu^2}}

    Parameters
    ==========

    mu :
        Positive number representing the mean.
    lambda :
        Positive number representing the shape parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import GaussianInverse, density, E, std, skewness
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", positive=True)
    >>> lamda = Symbol("lambda", positive=True)
    >>> z = Symbol("z", positive=True)
    >>> X = GaussianInverse("x", mu, lamda)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                       2
                      -lambda*(-mu + z)
                      -------------------
                                2
      ___   ________        2*mu *z
    \/ 2 *\/ lambda *e
    -------------------------------------
                    ____  3/2
                2*\/ pi *z

    >>> E(X)
    mu

    >>> std(X).expand()
    mu**(3/2)/sqrt(lambda)

    >>> skewness(X).expand()
    3*sqrt(mu)/sqrt(lambda)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
    .. [2] https://mathworld.wolfram.com/InverseGaussianDistribution.html

    """

    return rv(name, GaussianInverseDistribution, (mean, shape))

Wald = GaussianInverse

#-------------------------------------------------------------------------------
# Pareto distribution ----------------------------------------------------------


class ParetoDistribution(SingleContinuousDistribution):
    _argnames = ('xm', 'alpha')

    @property
    def set(self):
        return Interval(self.xm, oo)

    @staticmethod
    def check(xm, alpha):
        _value_check(xm > 0, "Xm must be positive")
        _value_check(alpha > 0, "Alpha must be positive")

    def pdf(self, x):
        xm, alpha = self.xm, self.alpha
        return alpha * xm**alpha / x**(alpha + 1)

    def _cdf(self, x):
        xm, alpha = self.xm, self.alpha
        return Piecewise(
                (S.One - xm**alpha/x**alpha, x>=xm),
                (0, True),
        )

    def _moment_generating_function(self, t):
        xm, alpha = self.xm, self.alpha
        return alpha * (-xm*t)**alpha * uppergamma(-alpha, -xm*t)

    def _characteristic_function(self, t):
        xm, alpha = self.xm, self.alpha
        return alpha * (-I * xm * t) ** alpha * uppergamma(-alpha, -I * xm * t)


def Pareto(name, xm, alpha):
    r"""
    Create a continuous random variable with the Pareto distribution.

    Explanation
    ===========

    The density of the Pareto distribution is given by

    .. math::
        f(x) := \frac{\alpha\,x_m^\alpha}{x^{\alpha+1}}

    with :math:`x \in [x_m,\infty]`.

    Parameters
    ==========

    xm : Real number, `x_m > 0`, a scale
    alpha : Real number, `\alpha > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Pareto, density
    >>> from sympy import Symbol

    >>> xm = Symbol("xm", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> z = Symbol("z")

    >>> X = Pareto("x", xm, beta)

    >>> density(X)(z)
    beta*xm**beta*z**(-beta - 1)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pareto_distribution
    .. [2] https://mathworld.wolfram.com/ParetoDistribution.html

    """

    return rv(name, ParetoDistribution, (xm, alpha))

#-------------------------------------------------------------------------------
# PowerFunction distribution ---------------------------------------------------


class PowerFunctionDistribution(SingleContinuousDistribution):
    _argnames=('alpha','a','b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(alpha, a, b):
        _value_check(a.is_real, "Continuous Boundary parameter should be real.")
        _value_check(b.is_real, "Continuous Boundary parameter should be real.")
        _value_check(a < b, " 'a' the left Boundary must be smaller than 'b' the right Boundary." )
        _value_check(alpha.is_positive, "Continuous Shape parameter should be positive.")

    def pdf(self, x):
        alpha, a, b = self.alpha, self.a, self.b
        num = alpha*(x - a)**(alpha - 1)
        den = (b - a)**alpha
        return num/den

def PowerFunction(name, alpha, a, b):
    r"""
    Creates a continuous random variable with a Power Function Distribution.

    Explanation
    ===========

    The density of PowerFunction distribution is given by

    .. math::
        f(x) := \frac{{\alpha}(x - a)^{\alpha - 1}}{(b - a)^{\alpha}}

    with :math:`x \in [a,b]`.

    Parameters
    ==========

    alpha : Positive number, `0 < \alpha`, the shape parameter
    a : Real number, :math:`-\infty < a`, the left boundary
    b : Real number, :math:`a < b < \infty`, the right boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import PowerFunction, density, cdf, E, variance
    >>> from sympy import Symbol
    >>> alpha = Symbol("alpha", positive=True)
    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = PowerFunction("X", 2, a, b)

    >>> density(X)(z)
    (-2*a + 2*z)/(-a + b)**2

    >>> cdf(X)(z)
    Piecewise((a**2/(a**2 - 2*a*b + b**2) - 2*a*z/(a**2 - 2*a*b + b**2) +
    z**2/(a**2 - 2*a*b + b**2), a <= z), (0, True))

    >>> alpha = 2
    >>> a = 0
    >>> b = 1
    >>> Y = PowerFunction("Y", alpha, a, b)

    >>> E(Y)
    2/3

    >>> variance(Y)
    1/18

    References
    ==========

    .. [1] https://web.archive.org/web/20200204081320/http://www.mathwave.com/help/easyfit/html/analyses/distributions/power_func.html

    """
    return rv(name, PowerFunctionDistribution, (alpha, a, b))

#-------------------------------------------------------------------------------
# QuadraticU distribution ------------------------------------------------------


class QuadraticUDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b):
        _value_check(b > a, "Parameter b must be in range (%s, oo)."%(a))

    def pdf(self, x):
        a, b = self.a, self.b
        alpha = 12 / (b-a)**3
        beta = (a+b) / 2
        return Piecewise(
                  (alpha * (x-beta)**2, And(a<=x, x<=b)),
                  (S.Zero, True))

    def _moment_generating_function(self, t):
        a, b = self.a, self.b
        return -3 * (exp(a*t) * (4  + (a**2 + 2*a*(-2 + b) + b**2) * t) \
        - exp(b*t) * (4 + (-4*b + (a + b)**2) * t)) / ((a-b)**3 * t**2)

    def _characteristic_function(self, t):
        a, b = self.a, self.b
        return -3*I*(exp(I*a*t*exp(I*b*t)) * (4*I - (-4*b + (a+b)**2)*t)) \
                / ((a-b)**3 * t**2)


def QuadraticU(name, a, b):
    r"""
    Create a Continuous Random Variable with a U-quadratic distribution.

    Explanation
    ===========

    The density of the U-quadratic distribution is given by

    .. math::
        f(x) := \alpha (x-\beta)^2

    with :math:`x \in [a,b]`.

    Parameters
    ==========

    a : Real number
    b : Real number, :math:`a < b`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import QuadraticU, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = QuadraticU("x", a, b)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    /                2
    |   /  a   b    \
    |12*|- - - - + z|
    |   \  2   2    /
    <-----------------  for And(b >= z, a <= z)
    |            3
    |    (-a + b)
    |
    \        0                 otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/U-quadratic_distribution

    """

    return rv(name, QuadraticUDistribution, (a, b))

#-------------------------------------------------------------------------------
# RaisedCosine distribution ----------------------------------------------------


class RaisedCosineDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')

    @property
    def set(self):
        return Interval(self.mu - self.s, self.mu + self.s)

    @staticmethod
    def check(mu, s):
        _value_check(s > 0, "s must be positive")

    def pdf(self, x):
        mu, s = self.mu, self.s
        return Piecewise(
                ((1+cos(pi*(x-mu)/s)) / (2*s), And(mu-s<=x, x<=mu+s)),
                (S.Zero, True))

    def _characteristic_function(self, t):
        mu, s = self.mu, self.s
        return Piecewise((exp(-I*pi*mu/s)/2, Eq(t, -pi/s)),
                         (exp(I*pi*mu/s)/2, Eq(t, pi/s)),
                         (pi**2*sin(s*t)*exp(I*mu*t) / (s*t*(pi**2 - s**2*t**2)), True))

    def _moment_generating_function(self, t):
        mu, s = self.mu, self.s
        return pi**2 * sinh(s*t) * exp(mu*t) /  (s*t*(pi**2 + s**2*t**2))

def RaisedCosine(name, mu, s):
    r"""
    Create a Continuous Random Variable with a raised cosine distribution.

    Explanation
    ===========

    The density of the raised cosine distribution is given by

    .. math::
        f(x) := \frac{1}{2s}\left(1+\cos\left(\frac{x-\mu}{s}\pi\right)\right)

    with :math:`x \in [\mu-s,\mu+s]`.

    Parameters
    ==========

    mu : Real number
    s : Real number, `s > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import RaisedCosine, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", real=True)
    >>> s = Symbol("s", positive=True)
    >>> z = Symbol("z")

    >>> X = RaisedCosine("x", mu, s)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    /   /pi*(-mu + z)\
    |cos|------------| + 1
    |   \     s      /
    <---------------------  for And(z >= mu - s, z <= mu + s)
    |         2*s
    |
    \          0                        otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Raised_cosine_distribution

    """

    return rv(name, RaisedCosineDistribution, (mu, s))

#-------------------------------------------------------------------------------
# Rayleigh distribution --------------------------------------------------------


class RayleighDistribution(SingleContinuousDistribution):
    _argnames = ('sigma',)

    set = Interval(0, oo)

    @staticmethod
    def check(sigma):
        _value_check(sigma > 0, "Scale parameter sigma must be positive.")

    def pdf(self, x):
        sigma = self.sigma
        return x/sigma**2*exp(-x**2/(2*sigma**2))

    def _cdf(self, x):
        sigma = self.sigma
        return 1 - exp(-(x**2/(2*sigma**2)))

    def _characteristic_function(self, t):
        sigma = self.sigma
        return 1 - sigma*t*exp(-sigma**2*t**2/2) * sqrt(pi/2) * (erfi(sigma*t/sqrt(2)) - I)

    def _moment_generating_function(self, t):
        sigma = self.sigma
        return 1 + sigma*t*exp(sigma**2*t**2/2) * sqrt(pi/2) * (erf(sigma*t/sqrt(2)) + 1)


def Rayleigh(name, sigma):
    r"""
    Create a continuous random variable with a Rayleigh distribution.

    Explanation
    ===========

    The density of the Rayleigh distribution is given by

    .. math ::
        f(x) := \frac{x}{\sigma^2} e^{-x^2/2\sigma^2}

    with :math:`x > 0`.

    Parameters
    ==========

    sigma : Real number, `\sigma > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Rayleigh, density, E, variance
    >>> from sympy import Symbol

    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = Rayleigh("x", sigma)

    >>> density(X)(z)
    z*exp(-z**2/(2*sigma**2))/sigma**2

    >>> E(X)
    sqrt(2)*sqrt(pi)*sigma/2

    >>> variance(X)
    -pi*sigma**2/2 + 2*sigma**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rayleigh_distribution
    .. [2] https://mathworld.wolfram.com/RayleighDistribution.html

    """

    return rv(name, RayleighDistribution, (sigma, ))

#-------------------------------------------------------------------------------
# Reciprocal distribution --------------------------------------------------------

class ReciprocalDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b):
        _value_check(a > 0, "Parameter > 0. a = %s"%a)
        _value_check((a < b),
        "Parameter b must be in range (%s, +oo]. b = %s"%(a, b))

    def pdf(self, x):
        a, b = self.a, self.b
        return 1/(x*(log(b) - log(a)))


def Reciprocal(name, a, b):
    r"""Creates a continuous random variable with a reciprocal distribution.


    Parameters
    ==========

    a : Real number, :math:`0 < a`
    b : Real number, :math:`a < b`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Reciprocal, density, cdf
    >>> from sympy import symbols
    >>> a, b, x = symbols('a, b, x', positive=True)
    >>> R = Reciprocal('R', a, b)

    >>> density(R)(x)
    1/(x*(-log(a) + log(b)))
    >>> cdf(R)(x)
    Piecewise((log(a)/(log(a) - log(b)) - log(x)/(log(a) - log(b)), a <= x), (0, True))

    Reference
    =========

    .. [1] https://en.wikipedia.org/wiki/Reciprocal_distribution

    """
    return rv(name, ReciprocalDistribution, (a, b))


#-------------------------------------------------------------------------------
# Shifted Gompertz distribution ------------------------------------------------


class ShiftedGompertzDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'eta')

    set = Interval(0, oo)

    @staticmethod
    def check(b, eta):
        _value_check(b > 0, "b must be positive")
        _value_check(eta > 0, "eta must be positive")

    def pdf(self, x):
        b, eta = self.b, self.eta
        return b*exp(-b*x)*exp(-eta*exp(-b*x))*(1+eta*(1-exp(-b*x)))

def ShiftedGompertz(name, b, eta):
    r"""
    Create a continuous random variable with a Shifted Gompertz distribution.

    Explanation
    ===========

    The density of the Shifted Gompertz distribution is given by

    .. math::
        f(x) := b e^{-b x} e^{-\eta \exp(-b x)} \left[1 + \eta(1 - e^(-bx)) \right]

    with :math:`x \in [0, \infty)`.

    Parameters
    ==========

    b : Real number, `b > 0`, a scale
    eta : Real number, `\eta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========
    >>> from sympy.stats import ShiftedGompertz, density
    >>> from sympy import Symbol

    >>> b = Symbol("b", positive=True)
    >>> eta = Symbol("eta", positive=True)
    >>> x = Symbol("x")

    >>> X = ShiftedGompertz("x", b, eta)

    >>> density(X)(x)
    b*(eta*(1 - exp(-b*x)) + 1)*exp(-b*x)*exp(-eta*exp(-b*x))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Shifted_Gompertz_distribution

    """
    return rv(name, ShiftedGompertzDistribution, (b, eta))

#-------------------------------------------------------------------------------
# StudentT distribution --------------------------------------------------------


class StudentTDistribution(SingleContinuousDistribution):
    _argnames = ('nu',)

    set = Interval(-oo, oo)

    @staticmethod
    def check(nu):
        _value_check(nu > 0, "Degrees of freedom nu must be positive.")

    def pdf(self, x):
        nu = self.nu
        return 1/(sqrt(nu)*beta_fn(S.Half, nu/2))*(1 + x**2/nu)**(-(nu + 1)/2)

    def _cdf(self, x):
        nu = self.nu
        return S.Half + x*gamma((nu+1)/2)*hyper((S.Half, (nu+1)/2),
                                (Rational(3, 2),), -x**2/nu)/(sqrt(pi*nu)*gamma(nu/2))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the Student-T distribution is undefined.')


def StudentT(name, nu):
    r"""
    Create a continuous random variable with a student's t distribution.

    Explanation
    ===========

    The density of the student's t distribution is given by

    .. math::
        f(x) := \frac{\Gamma \left(\frac{\nu+1}{2} \right)}
                {\sqrt{\nu\pi}\Gamma \left(\frac{\nu}{2} \right)}
                \left(1+\frac{x^2}{\nu} \right)^{-\frac{\nu+1}{2}}

    Parameters
    ==========

    nu : Real number, `\nu > 0`, the degrees of freedom

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import StudentT, density, cdf
    >>> from sympy import Symbol, pprint

    >>> nu = Symbol("nu", positive=True)
    >>> z = Symbol("z")

    >>> X = StudentT("x", nu)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
               nu   1
             - -- - -
               2    2
     /     2\
     |    z |
     |1 + --|
     \    nu/
    -----------------
      ____  /     nu\
    \/ nu *B|1/2, --|
            \     2 /

    >>> cdf(X)(z)
    1/2 + z*gamma(nu/2 + 1/2)*hyper((1/2, nu/2 + 1/2), (3/2,),
                                -z**2/nu)/(sqrt(pi)*sqrt(nu)*gamma(nu/2))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Student_t-distribution
    .. [2] https://mathworld.wolfram.com/Studentst-Distribution.html

    """

    return rv(name, StudentTDistribution, (nu, ))

#-------------------------------------------------------------------------------
# Trapezoidal distribution ------------------------------------------------------


class TrapezoidalDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b', 'c', 'd')

    @property
    def set(self):
        return Interval(self.a, self.d)

    @staticmethod
    def check(a, b, c, d):
        _value_check(a < d, "Lower bound parameter a < %s. a = %s"%(d, a))
        _value_check((a <= b, b < c),
        "Level start parameter b must be in range [%s, %s). b = %s"%(a, c, b))
        _value_check((b < c, c <= d),
        "Level end parameter c must be in range (%s, %s]. c = %s"%(b, d, c))
        _value_check(d >= c, "Upper bound parameter d > %s. d = %s"%(c, d))

    def pdf(self, x):
        a, b, c, d = self.a, self.b, self.c, self.d
        return Piecewise(
            (2*(x-a) / ((b-a)*(d+c-a-b)), And(a <= x, x < b)),
            (2 / (d+c-a-b), And(b <= x, x < c)),
            (2*(d-x) / ((d-c)*(d+c-a-b)), And(c <= x, x <= d)),
            (S.Zero, True))

def Trapezoidal(name, a, b, c, d):
    r"""
    Create a continuous random variable with a trapezoidal distribution.

    Explanation
    ===========

    The density of the trapezoidal distribution is given by

    .. math::
        f(x) := \begin{cases}
                  0 & \mathrm{for\ } x < a, \\
                  \frac{2(x-a)}{(b-a)(d+c-a-b)} & \mathrm{for\ } a \le x < b, \\
                  \frac{2}{d+c-a-b} & \mathrm{for\ } b \le x < c, \\
                  \frac{2(d-x)}{(d-c)(d+c-a-b)} & \mathrm{for\ } c \le x < d, \\
                  0 & \mathrm{for\ } d < x.
                \end{cases}

    Parameters
    ==========

    a : Real number, :math:`a < d`
    b : Real number, :math:`a \le b < c`
    c : Real number, :math:`b < c \le d`
    d : Real number

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Trapezoidal, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a")
    >>> b = Symbol("b")
    >>> c = Symbol("c")
    >>> d = Symbol("d")
    >>> z = Symbol("z")

    >>> X = Trapezoidal("x", a,b,c,d)

    >>> pprint(density(X)(z), use_unicode=False)
    /        -2*a + 2*z
    |-------------------------  for And(a <= z, b > z)
    |(-a + b)*(-a - b + c + d)
    |
    |           2
    |     --------------        for And(b <= z, c > z)
    <     -a - b + c + d
    |
    |        2*d - 2*z
    |-------------------------  for And(d >= z, c <= z)
    |(-c + d)*(-a - b + c + d)
    |
    \            0                     otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trapezoidal_distribution

    """
    return rv(name, TrapezoidalDistribution, (a, b, c, d))

#-------------------------------------------------------------------------------
# Triangular distribution ------------------------------------------------------


class TriangularDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b', 'c')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b, c):
        _value_check(b > a, "Parameter b > %s. b = %s"%(a, b))
        _value_check((a <= c, c <= b),
        "Parameter c must be in range [%s, %s]. c = %s"%(a, b, c))

    def pdf(self, x):
        a, b, c = self.a, self.b, self.c
        return Piecewise(
            (2*(x - a)/((b - a)*(c - a)), And(a <= x, x < c)),
            (2/(b - a), Eq(x, c)),
            (2*(b - x)/((b - a)*(b - c)), And(c < x, x <= b)),
            (S.Zero, True))

    def _characteristic_function(self, t):
        a, b, c = self.a, self.b, self.c
        return -2 *((b-c) * exp(I*a*t) - (b-a) * exp(I*c*t) + (c-a) * exp(I*b*t)) / ((b-a)*(c-a)*(b-c)*t**2)

    def _moment_generating_function(self, t):
        a, b, c = self.a, self.b, self.c
        return 2 * ((b - c) * exp(a * t) - (b - a) * exp(c * t) + (c - a) * exp(b * t)) / (
        (b - a) * (c - a) * (b - c) * t ** 2)


def Triangular(name, a, b, c):
    r"""
    Create a continuous random variable with a triangular distribution.

    Explanation
    ===========

    The density of the triangular distribution is given by

    .. math::
        f(x) := \begin{cases}
                  0 & \mathrm{for\ } x < a, \\
                  \frac{2(x-a)}{(b-a)(c-a)} & \mathrm{for\ } a \le x < c, \\
                  \frac{2}{b-a} & \mathrm{for\ } x = c, \\
                  \frac{2(b-x)}{(b-a)(b-c)} & \mathrm{for\ } c < x \le b, \\
                  0 & \mathrm{for\ } b < x.
                \end{cases}

    Parameters
    ==========

    a : Real number, :math:`a \in \left(-\infty, \infty\right)`
    b : Real number, :math:`a < b`
    c : Real number, :math:`a \leq c \leq b`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Triangular, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a")
    >>> b = Symbol("b")
    >>> c = Symbol("c")
    >>> z = Symbol("z")

    >>> X = Triangular("x", a,b,c)

    >>> pprint(density(X)(z), use_unicode=False)
    /    -2*a + 2*z
    |-----------------  for And(a <= z, c > z)
    |(-a + b)*(-a + c)
    |
    |       2
    |     ------              for c = z
    <     -a + b
    |
    |   2*b - 2*z
    |----------------   for And(b >= z, c < z)
    |(-a + b)*(b - c)
    |
    \        0                otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Triangular_distribution
    .. [2] https://mathworld.wolfram.com/TriangularDistribution.html

    """

    return rv(name, TriangularDistribution, (a, b, c))

#-------------------------------------------------------------------------------
# Uniform distribution ---------------------------------------------------------


class UniformDistribution(SingleContinuousDistribution):
    _argnames = ('left', 'right')

    @property
    def set(self):
        return Interval(self.left, self.right)

    @staticmethod
    def check(left, right):
        _value_check(left < right, "Lower limit should be less than Upper limit.")

    def pdf(self, x):
        left, right = self.left, self.right
        return Piecewise(
            (S.One/(right - left), And(left <= x, x <= right)),
            (S.Zero, True)
        )

    def _cdf(self, x):
        left, right = self.left, self.right
        return Piecewise(
            (S.Zero, x < left),
            ((x - left)/(right - left), x <= right),
            (S.One, True)
        )

    def _characteristic_function(self, t):
        left, right = self.left, self.right
        return Piecewise(((exp(I*t*right) - exp(I*t*left)) / (I*t*(right - left)), Ne(t, 0)),
                         (S.One, True))

    def _moment_generating_function(self, t):
        left, right = self.left, self.right
        return Piecewise(((exp(t*right) - exp(t*left)) / (t * (right - left)), Ne(t, 0)),
                         (S.One, True))

    def expectation(self, expr, var, **kwargs):
        kwargs['evaluate'] = True
        result = SingleContinuousDistribution.expectation(self, expr, var, **kwargs)
        result = result.subs({Max(self.left, self.right): self.right,
                              Min(self.left, self.right): self.left})
        return result


def Uniform(name, left, right):
    r"""
    Create a continuous random variable with a uniform distribution.

    Explanation
    ===========

    The density of the uniform distribution is given by

    .. math::
        f(x) := \begin{cases}
                  \frac{1}{b - a} & \text{for } x \in [a,b]  \\
                  0               & \text{otherwise}
                \end{cases}

    with :math:`x \in [a,b]`.

    Parameters
    ==========

    a : Real number, :math:`-\infty < a`, the left boundary
    b : Real number, :math:`a < b < \infty`, the right boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Uniform, density, cdf, E, variance
    >>> from sympy import Symbol, simplify

    >>> a = Symbol("a", negative=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Uniform("x", a, b)

    >>> density(X)(z)
    Piecewise((1/(-a + b), (b >= z) & (a <= z)), (0, True))

    >>> cdf(X)(z)
    Piecewise((0, a > z), ((-a + z)/(-a + b), b >= z), (1, True))

    >>> E(X)
    a/2 + b/2

    >>> simplify(variance(X))
    a**2/12 - a*b/6 + b**2/12

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Uniform_distribution_%28continuous%29
    .. [2] https://mathworld.wolfram.com/UniformDistribution.html

    """

    return rv(name, UniformDistribution, (left, right))

#-------------------------------------------------------------------------------
# UniformSum distribution ------------------------------------------------------


class UniformSumDistribution(SingleContinuousDistribution):
    _argnames = ('n',)

    @property
    def set(self):
        return Interval(0, self.n)

    @staticmethod
    def check(n):
        _value_check((n > 0, n.is_integer),
        "Parameter n must be positive integer.")

    def pdf(self, x):
        n = self.n
        k = Dummy("k")
        return 1/factorial(
            n - 1)*Sum((-1)**k*binomial(n, k)*(x - k)**(n - 1), (k, 0, floor(x)))

    def _cdf(self, x):
        n = self.n
        k = Dummy("k")
        return Piecewise((S.Zero, x < 0),
                        (1/factorial(n)*Sum((-1)**k*binomial(n, k)*(x - k)**(n),
                        (k, 0, floor(x))), x <= n),
                        (S.One, True))

    def _characteristic_function(self, t):
        return ((exp(I*t) - 1) / (I*t))**self.n

    def _moment_generating_function(self, t):
        return ((exp(t) - 1) / t)**self.n

def UniformSum(name, n):
    r"""
    Create a continuous random variable with an Irwin-Hall distribution.

    Explanation
    ===========

    The probability distribution function depends on a single parameter
    $n$ which is an integer.

    The density of the Irwin-Hall distribution is given by

    .. math ::
        f(x) := \frac{1}{(n-1)!}\sum_{k=0}^{\left\lfloor x\right\rfloor}(-1)^k
                \binom{n}{k}(x-k)^{n-1}

    Parameters
    ==========

    n : A positive integer, `n > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import UniformSum, density, cdf
    >>> from sympy import Symbol, pprint

    >>> n = Symbol("n", integer=True)
    >>> z = Symbol("z")

    >>> X = UniformSum("x", n)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    floor(z)
      ___
      \  `
       \         k         n - 1 /n\
        )    (-1) *(-k + z)     *| |
       /                         \k/
      /__,
     k = 0
    --------------------------------
                (n - 1)!

    >>> cdf(X)(z)
    Piecewise((0, z < 0), (Sum((-1)**_k*(-_k + z)**n*binomial(n, _k),
                    (_k, 0, floor(z)))/factorial(n), n >= z), (1, True))


    Compute cdf with specific 'x' and 'n' values as follows :
    >>> cdf(UniformSum("x", 5), evaluate=False)(2).doit()
    9/40

    The argument evaluate=False prevents an attempt at evaluation
    of the sum for general n, before the argument 2 is passed.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Uniform_sum_distribution
    .. [2] https://mathworld.wolfram.com/UniformSumDistribution.html

    """

    return rv(name, UniformSumDistribution, (n, ))

#-------------------------------------------------------------------------------
# VonMises distribution --------------------------------------------------------


class VonMisesDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'k')

    set = Interval(0, 2*pi)

    @staticmethod
    def check(mu, k):
        _value_check(k > 0, "k must be positive")

    def pdf(self, x):
        mu, k = self.mu, self.k
        return exp(k*cos(x-mu)) / (2*pi*besseli(0, k))

def VonMises(name, mu, k):
    r"""
    Create a Continuous Random Variable with a von Mises distribution.

    Explanation
    ===========

    The density of the von Mises distribution is given by

    .. math::
        f(x) := \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    with :math:`x \in [0,2\pi]`.

    Parameters
    ==========

    mu : Real number
        Measure of location.
    k : Real number
        Measure of concentration.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import VonMises, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu")
    >>> k = Symbol("k", positive=True)
    >>> z = Symbol("z")

    >>> X = VonMises("x", mu, k)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
         k*cos(mu - z)
        e
    ------------------
    2*pi*besseli(0, k)


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Von_Mises_distribution
    .. [2] https://mathworld.wolfram.com/vonMisesDistribution.html

    """

    return rv(name, VonMisesDistribution, (mu, k))

#-------------------------------------------------------------------------------
# Weibull distribution ---------------------------------------------------------


class WeibullDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    set = Interval(0, oo)

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Alpha must be positive")
        _value_check(beta > 0, "Beta must be positive")

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return beta * (x/alpha)**(beta - 1) * exp(-(x/alpha)**beta) / alpha


def Weibull(name, alpha, beta):
    r"""
    Create a continuous random variable with a Weibull distribution.

    Explanation
    ===========

    The density of the Weibull distribution is given by

    .. math::
        f(x) := \begin{cases}
                  \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}
                  e^{-(x/\lambda)^{k}} & x\geq0\\
                  0 & x<0
                \end{cases}

    Parameters
    ==========

    lambda : Real number, $\lambda > 0$, a scale
    k : Real number, $k > 0$, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Weibull, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> l = Symbol("lambda", positive=True)
    >>> k = Symbol("k", positive=True)
    >>> z = Symbol("z")

    >>> X = Weibull("x", l, k)

    >>> density(X)(z)
    k*(z/lambda)**(k - 1)*exp(-(z/lambda)**k)/lambda

    >>> simplify(E(X))
    lambda*gamma(1 + 1/k)

    >>> simplify(variance(X))
    lambda**2*(-gamma(1 + 1/k)**2 + gamma(1 + 2/k))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Weibull_distribution
    .. [2] https://mathworld.wolfram.com/WeibullDistribution.html

    """

    return rv(name, WeibullDistribution, (alpha, beta))

#-------------------------------------------------------------------------------
# Wigner semicircle distribution -----------------------------------------------


class WignerSemicircleDistribution(SingleContinuousDistribution):
    _argnames = ('R',)

    @property
    def set(self):
        return Interval(-self.R, self.R)

    @staticmethod
    def check(R):
        _value_check(R > 0, "Radius R must be positive.")

    def pdf(self, x):
        R = self.R
        return 2/(pi*R**2)*sqrt(R**2 - x**2)

    def _characteristic_function(self, t):
        return Piecewise((2 * besselj(1, self.R*t) / (self.R*t), Ne(t, 0)),
                         (S.One, True))

    def _moment_generating_function(self, t):
        return Piecewise((2 * besseli(1, self.R*t) / (self.R*t), Ne(t, 0)),
                         (S.One, True))

def WignerSemicircle(name, R):
    r"""
    Create a continuous random variable with a Wigner semicircle distribution.

    Explanation
    ===========

    The density of the Wigner semicircle distribution is given by

    .. math::
        f(x) := \frac2{\pi R^2}\,\sqrt{R^2-x^2}

    with :math:`x \in [-R,R]`.

    Parameters
    ==========

    R : Real number, `R > 0`, the radius

    Returns
    =======

    A RandomSymbol.

    Examples
    ========

    >>> from sympy.stats import WignerSemicircle, density, E
    >>> from sympy import Symbol

    >>> R = Symbol("R", positive=True)
    >>> z = Symbol("z")

    >>> X = WignerSemicircle("x", R)

    >>> density(X)(z)
    2*sqrt(R**2 - z**2)/(pi*R**2)

    >>> E(X)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Wigner_semicircle_distribution
    .. [2] https://mathworld.wolfram.com/WignersSemicircleLaw.html

    """

    return rv(name, WignerSemicircleDistribution, (R,))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\sansio\app.py ===
from __future__ import annotations

import logging
import os
import sys
import typing as t
from datetime import timedelta
from itertools import chain

from werkzeug.exceptions import Aborter
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import BadRequestKeyError
from werkzeug.routing import BuildError
from werkzeug.routing import Map
from werkzeug.routing import Rule
from werkzeug.sansio.response import Response
from werkzeug.utils import cached_property
from werkzeug.utils import redirect as _wz_redirect

from .. import typing as ft
from ..config import Config
from ..config import ConfigAttribute
from ..ctx import _AppCtxGlobals
from ..helpers import _split_blueprint_path
from ..helpers import get_debug_flag
from ..json.provider import DefaultJSONProvider
from ..json.provider import JSONProvider
from ..logging import create_logger
from ..templating import DispatchingJinjaLoader
from ..templating import Environment
from .scaffold import _endpoint_from_view_func
from .scaffold import find_package
from .scaffold import Scaffold
from .scaffold import setupmethod

if t.TYPE_CHECKING:  # pragma: no cover
    from werkzeug.wrappers import Response as BaseResponse

    from ..testing import FlaskClient
    from ..testing import FlaskCliRunner
    from .blueprints import Blueprint

T_shell_context_processor = t.TypeVar(
    "T_shell_context_processor", bound=ft.ShellContextProcessorCallable
)
T_teardown = t.TypeVar("T_teardown", bound=ft.TeardownCallable)
T_template_filter = t.TypeVar("T_template_filter", bound=ft.TemplateFilterCallable)
T_template_global = t.TypeVar("T_template_global", bound=ft.TemplateGlobalCallable)
T_template_test = t.TypeVar("T_template_test", bound=ft.TemplateTestCallable)


def _make_timedelta(value: timedelta | int | None) -> timedelta | None:
    if value is None or isinstance(value, timedelta):
        return value

    return timedelta(seconds=value)


class App(Scaffold):
    """The flask object implements a WSGI application and acts as the central
    object.  It is passed the name of the module or package of the
    application.  Once it is created it will act as a central registry for
    the view functions, the URL rules, template configuration and much more.

    The name of the package is used to resolve resources from inside the
    package or the folder the module is contained in depending on if the
    package parameter resolves to an actual python package (a folder with
    an :file:`__init__.py` file inside) or a standard module (just a ``.py`` file).

    For more information about resource loading, see :func:`open_resource`.

    Usually you create a :class:`Flask` instance in your main module or
    in the :file:`__init__.py` file of your package like this::

        from flask import Flask
        app = Flask(__name__)

    .. admonition:: About the First Parameter

        The idea of the first parameter is to give Flask an idea of what
        belongs to your application.  This name is used to find resources
        on the filesystem, can be used by extensions to improve debugging
        information and a lot more.

        So it's important what you provide there.  If you are using a single
        module, `__name__` is always the correct value.  If you however are
        using a package, it's usually recommended to hardcode the name of
        your package there.

        For example if your application is defined in :file:`yourapplication/app.py`
        you should create it with one of the two versions below::

            app = Flask('yourapplication')
            app = Flask(__name__.split('.')[0])

        Why is that?  The application will work even with `__name__`, thanks
        to how resources are looked up.  However it will make debugging more
        painful.  Certain extensions can make assumptions based on the
        import name of your application.  For example the Flask-SQLAlchemy
        extension will look for the code in your application that triggered
        an SQL query in debug mode.  If the import name is not properly set
        up, that debugging information is lost.  (For example it would only
        pick up SQL queries in `yourapplication.app` and not
        `yourapplication.views.frontend`)

    .. versionadded:: 0.7
       The `static_url_path`, `static_folder`, and `template_folder`
       parameters were added.

    .. versionadded:: 0.8
       The `instance_path` and `instance_relative_config` parameters were
       added.

    .. versionadded:: 0.11
       The `root_path` parameter was added.

    .. versionadded:: 1.0
       The ``host_matching`` and ``static_host`` parameters were added.

    .. versionadded:: 1.0
       The ``subdomain_matching`` parameter was added. Subdomain
       matching needs to be enabled manually now. Setting
       :data:`SERVER_NAME` does not implicitly enable it.

    :param import_name: the name of the application package
    :param static_url_path: can be used to specify a different path for the
                            static files on the web.  Defaults to the name
                            of the `static_folder` folder.
    :param static_folder: The folder with static files that is served at
        ``static_url_path``. Relative to the application ``root_path``
        or an absolute path. Defaults to ``'static'``.
    :param static_host: the host to use when adding the static route.
        Defaults to None. Required when using ``host_matching=True``
        with a ``static_folder`` configured.
    :param host_matching: set ``url_map.host_matching`` attribute.
        Defaults to False.
    :param subdomain_matching: consider the subdomain relative to
        :data:`SERVER_NAME` when matching routes. Defaults to False.
    :param template_folder: the folder that contains the templates that should
                            be used by the application.  Defaults to
                            ``'templates'`` folder in the root path of the
                            application.
    :param instance_path: An alternative instance path for the application.
                          By default the folder ``'instance'`` next to the
                          package or module is assumed to be the instance
                          path.
    :param instance_relative_config: if set to ``True`` relative filenames
                                     for loading the config are assumed to
                                     be relative to the instance path instead
                                     of the application root.
    :param root_path: The path to the root of the application files.
        This should only be set manually when it can't be detected
        automatically, such as for namespace packages.
    """

    #: The class of the object assigned to :attr:`aborter`, created by
    #: :meth:`create_aborter`. That object is called by
    #: :func:`flask.abort` to raise HTTP errors, and can be
    #: called directly as well.
    #:
    #: Defaults to :class:`werkzeug.exceptions.Aborter`.
    #:
    #: .. versionadded:: 2.2
    aborter_class = Aborter

    #: The class that is used for the Jinja environment.
    #:
    #: .. versionadded:: 0.11
    jinja_environment = Environment

    #: The class that is used for the :data:`~flask.g` instance.
    #:
    #: Example use cases for a custom class:
    #:
    #: 1. Store arbitrary attributes on flask.g.
    #: 2. Add a property for lazy per-request database connectors.
    #: 3. Return None instead of AttributeError on unexpected attributes.
    #: 4. Raise exception if an unexpected attr is set, a "controlled" flask.g.
    #:
    #: In Flask 0.9 this property was called `request_globals_class` but it
    #: was changed in 0.10 to :attr:`app_ctx_globals_class` because the
    #: flask.g object is now application context scoped.
    #:
    #: .. versionadded:: 0.10
    app_ctx_globals_class = _AppCtxGlobals

    #: The class that is used for the ``config`` attribute of this app.
    #: Defaults to :class:`~flask.Config`.
    #:
    #: Example use cases for a custom class:
    #:
    #: 1. Default values for certain config options.
    #: 2. Access to config values through attributes in addition to keys.
    #:
    #: .. versionadded:: 0.11
    config_class = Config

    #: The testing flag.  Set this to ``True`` to enable the test mode of
    #: Flask extensions (and in the future probably also Flask itself).
    #: For example this might activate test helpers that have an
    #: additional runtime cost which should not be enabled by default.
    #:
    #: If this is enabled and PROPAGATE_EXCEPTIONS is not changed from the
    #: default it's implicitly enabled.
    #:
    #: This attribute can also be configured from the config with the
    #: ``TESTING`` configuration key.  Defaults to ``False``.
    testing = ConfigAttribute[bool]("TESTING")

    #: If a secret key is set, cryptographic components can use this to
    #: sign cookies and other things. Set this to a complex random value
    #: when you want to use the secure cookie for instance.
    #:
    #: This attribute can also be configured from the config with the
    #: :data:`SECRET_KEY` configuration key. Defaults to ``None``.
    secret_key = ConfigAttribute[t.Union[str, bytes, None]]("SECRET_KEY")

    #: A :class:`~datetime.timedelta` which is used to set the expiration
    #: date of a permanent session.  The default is 31 days which makes a
    #: permanent session survive for roughly one month.
    #:
    #: This attribute can also be configured from the config with the
    #: ``PERMANENT_SESSION_LIFETIME`` configuration key.  Defaults to
    #: ``timedelta(days=31)``
    permanent_session_lifetime = ConfigAttribute[timedelta](
        "PERMANENT_SESSION_LIFETIME",
        get_converter=_make_timedelta,  # type: ignore[arg-type]
    )

    json_provider_class: type[JSONProvider] = DefaultJSONProvider
    """A subclass of :class:`~flask.json.provider.JSONProvider`. An
    instance is created and assigned to :attr:`app.json` when creating
    the app.

    The default, :class:`~flask.json.provider.DefaultJSONProvider`, uses
    Python's built-in :mod:`json` library. A different provider can use
    a different JSON library.

    .. versionadded:: 2.2
    """

    #: Options that are passed to the Jinja environment in
    #: :meth:`create_jinja_environment`. Changing these options after
    #: the environment is created (accessing :attr:`jinja_env`) will
    #: have no effect.
    #:
    #: .. versionchanged:: 1.1.0
    #:     This is a ``dict`` instead of an ``ImmutableDict`` to allow
    #:     easier configuration.
    #:
    jinja_options: dict[str, t.Any] = {}

    #: The rule object to use for URL rules created.  This is used by
    #: :meth:`add_url_rule`.  Defaults to :class:`werkzeug.routing.Rule`.
    #:
    #: .. versionadded:: 0.7
    url_rule_class = Rule

    #: The map object to use for storing the URL rules and routing
    #: configuration parameters. Defaults to :class:`werkzeug.routing.Map`.
    #:
    #: .. versionadded:: 1.1.0
    url_map_class = Map

    #: The :meth:`test_client` method creates an instance of this test
    #: client class. Defaults to :class:`~flask.testing.FlaskClient`.
    #:
    #: .. versionadded:: 0.7
    test_client_class: type[FlaskClient] | None = None

    #: The :class:`~click.testing.CliRunner` subclass, by default
    #: :class:`~flask.testing.FlaskCliRunner` that is used by
    #: :meth:`test_cli_runner`. Its ``__init__`` method should take a
    #: Flask app object as the first argument.
    #:
    #: .. versionadded:: 1.0
    test_cli_runner_class: type[FlaskCliRunner] | None = None

    default_config: dict[str, t.Any]
    response_class: type[Response]

    def __init__(
        self,
        import_name: str,
        static_url_path: str | None = None,
        static_folder: str | os.PathLike[str] | None = "static",
        static_host: str | None = None,
        host_matching: bool = False,
        subdomain_matching: bool = False,
        template_folder: str | os.PathLike[str] | None = "templates",
        instance_path: str | None = None,
        instance_relative_config: bool = False,
        root_path: str | None = None,
    ) -> None:
        super().__init__(
            import_name=import_name,
            static_folder=static_folder,
            static_url_path=static_url_path,
            template_folder=template_folder,
            root_path=root_path,
        )

        if instance_path is None:
            instance_path = self.auto_find_instance_path()
        elif not os.path.isabs(instance_path):
            raise ValueError(
                "If an instance path is provided it must be absolute."
                " A relative path was given instead."
            )

        #: Holds the path to the instance folder.
        #:
        #: .. versionadded:: 0.8
        self.instance_path = instance_path

        #: The configuration dictionary as :class:`Config`.  This behaves
        #: exactly like a regular dictionary but supports additional methods
        #: to load a config from files.
        self.config = self.make_config(instance_relative_config)

        #: An instance of :attr:`aborter_class` created by
        #: :meth:`make_aborter`. This is called by :func:`flask.abort`
        #: to raise HTTP errors, and can be called directly as well.
        #:
        #: .. versionadded:: 2.2
        #:     Moved from ``flask.abort``, which calls this object.
        self.aborter = self.make_aborter()

        self.json: JSONProvider = self.json_provider_class(self)
        """Provides access to JSON methods. Functions in ``flask.json``
        will call methods on this provider when the application context
        is active. Used for handling JSON requests and responses.

        An instance of :attr:`json_provider_class`. Can be customized by
        changing that attribute on a subclass, or by assigning to this
        attribute afterwards.

        The default, :class:`~flask.json.provider.DefaultJSONProvider`,
        uses Python's built-in :mod:`json` library. A different provider
        can use a different JSON library.

        .. versionadded:: 2.2
        """

        #: A list of functions that are called by
        #: :meth:`handle_url_build_error` when :meth:`.url_for` raises a
        #: :exc:`~werkzeug.routing.BuildError`. Each function is called
        #: with ``error``, ``endpoint`` and ``values``. If a function
        #: returns ``None`` or raises a ``BuildError``, it is skipped.
        #: Otherwise, its return value is returned by ``url_for``.
        #:
        #: .. versionadded:: 0.9
        self.url_build_error_handlers: list[
            t.Callable[[Exception, str, dict[str, t.Any]], str]
        ] = []

        #: A list of functions that are called when the application context
        #: is destroyed.  Since the application context is also torn down
        #: if the request ends this is the place to store code that disconnects
        #: from databases.
        #:
        #: .. versionadded:: 0.9
        self.teardown_appcontext_funcs: list[ft.TeardownCallable] = []

        #: A list of shell context processor functions that should be run
        #: when a shell context is created.
        #:
        #: .. versionadded:: 0.11
        self.shell_context_processors: list[ft.ShellContextProcessorCallable] = []

        #: Maps registered blueprint names to blueprint objects. The
        #: dict retains the order the blueprints were registered in.
        #: Blueprints can be registered multiple times, this dict does
        #: not track how often they were attached.
        #:
        #: .. versionadded:: 0.7
        self.blueprints: dict[str, Blueprint] = {}

        #: a place where extensions can store application specific state.  For
        #: example this is where an extension could store database engines and
        #: similar things.
        #:
        #: The key must match the name of the extension module. For example in
        #: case of a "Flask-Foo" extension in `flask_foo`, the key would be
        #: ``'foo'``.
        #:
        #: .. versionadded:: 0.7
        self.extensions: dict[str, t.Any] = {}

        #: The :class:`~werkzeug.routing.Map` for this instance.  You can use
        #: this to change the routing converters after the class was created
        #: but before any routes are connected.  Example::
        #:
        #:    from werkzeug.routing import BaseConverter
        #:
        #:    class ListConverter(BaseConverter):
        #:        def to_python(self, value):
        #:            return value.split(',')
        #:        def to_url(self, values):
        #:            return ','.join(super(ListConverter, self).to_url(value)
        #:                            for value in values)
        #:
        #:    app = Flask(__name__)
        #:    app.url_map.converters['list'] = ListConverter
        self.url_map = self.url_map_class(host_matching=host_matching)

        self.subdomain_matching = subdomain_matching

        # tracks internally if the application already handled at least one
        # request.
        self._got_first_request = False

    def _check_setup_finished(self, f_name: str) -> None:
        if self._got_first_request:
            raise AssertionError(
                f"The setup method '{f_name}' can no longer be called"
                " on the application. It has already handled its first"
                " request, any changes will not be applied"
                " consistently.\n"
                "Make sure all imports, decorators, functions, etc."
                " needed to set up the application are done before"
                " running it."
            )

    @cached_property
    def name(self) -> str:  # type: ignore
        """The name of the application.  This is usually the import name
        with the difference that it's guessed from the run file if the
        import name is main.  This name is used as a display name when
        Flask needs the name of the application.  It can be set and overridden
        to change the value.

        .. versionadded:: 0.8
        """
        if self.import_name == "__main__":
            fn: str | None = getattr(sys.modules["__main__"], "__file__", None)
            if fn is None:
                return "__main__"
            return os.path.splitext(os.path.basename(fn))[0]
        return self.import_name

    @cached_property
    def logger(self) -> logging.Logger:
        """A standard Python :class:`~logging.Logger` for the app, with
        the same name as :attr:`name`.

        In debug mode, the logger's :attr:`~logging.Logger.level` will
        be set to :data:`~logging.DEBUG`.

        If there are no handlers configured, a default handler will be
        added. See :doc:`/logging` for more information.

        .. versionchanged:: 1.1.0
            The logger takes the same name as :attr:`name` rather than
            hard-coding ``"flask.app"``.

        .. versionchanged:: 1.0.0
            Behavior was simplified. The logger is always named
            ``"flask.app"``. The level is only set during configuration,
            it doesn't check ``app.debug`` each time. Only one format is
            used, not different ones depending on ``app.debug``. No
            handlers are removed, and a handler is only added if no
            handlers are already configured.

        .. versionadded:: 0.3
        """
        return create_logger(self)

    @cached_property
    def jinja_env(self) -> Environment:
        """The Jinja environment used to load templates.

        The environment is created the first time this property is
        accessed. Changing :attr:`jinja_options` after that will have no
        effect.
        """
        return self.create_jinja_environment()

    def create_jinja_environment(self) -> Environment:
        raise NotImplementedError()

    def make_config(self, instance_relative: bool = False) -> Config:
        """Used to create the config attribute by the Flask constructor.
        The `instance_relative` parameter is passed in from the constructor
        of Flask (there named `instance_relative_config`) and indicates if
        the config should be relative to the instance path or the root path
        of the application.

        .. versionadded:: 0.8
        """
        root_path = self.root_path
        if instance_relative:
            root_path = self.instance_path
        defaults = dict(self.default_config)
        defaults["DEBUG"] = get_debug_flag()
        return self.config_class(root_path, defaults)

    def make_aborter(self) -> Aborter:
        """Create the object to assign to :attr:`aborter`. That object
        is called by :func:`flask.abort` to raise HTTP errors, and can
        be called directly as well.

        By default, this creates an instance of :attr:`aborter_class`,
        which defaults to :class:`werkzeug.exceptions.Aborter`.

        .. versionadded:: 2.2
        """
        return self.aborter_class()

    def auto_find_instance_path(self) -> str:
        """Tries to locate the instance path if it was not provided to the
        constructor of the application class.  It will basically calculate
        the path to a folder named ``instance`` next to your main file or
        the package.

        .. versionadded:: 0.8
        """
        prefix, package_path = find_package(self.import_name)
        if prefix is None:
            return os.path.join(package_path, "instance")
        return os.path.join(prefix, "var", f"{self.name}-instance")

    def create_global_jinja_loader(self) -> DispatchingJinjaLoader:
        """Creates the loader for the Jinja2 environment.  Can be used to
        override just the loader and keeping the rest unchanged.  It's
        discouraged to override this function.  Instead one should override
        the :meth:`jinja_loader` function instead.

        The global loader dispatches between the loaders of the application
        and the individual blueprints.

        .. versionadded:: 0.7
        """
        return DispatchingJinjaLoader(self)

    def select_jinja_autoescape(self, filename: str) -> bool:
        """Returns ``True`` if autoescaping should be active for the given
        template name. If no template name is given, returns `True`.

        .. versionchanged:: 2.2
            Autoescaping is now enabled by default for ``.svg`` files.

        .. versionadded:: 0.5
        """
        if filename is None:
            return True
        return filename.endswith((".html", ".htm", ".xml", ".xhtml", ".svg"))

    @property
    def debug(self) -> bool:
        """Whether debug mode is enabled. When using ``flask run`` to start the
        development server, an interactive debugger will be shown for unhandled
        exceptions, and the server will be reloaded when code changes. This maps to the
        :data:`DEBUG` config key. It may not behave as expected if set late.

        **Do not enable debug mode when deploying in production.**

        Default: ``False``
        """
        return self.config["DEBUG"]  # type: ignore[no-any-return]

    @debug.setter
    def debug(self, value: bool) -> None:
        self.config["DEBUG"] = value

        if self.config["TEMPLATES_AUTO_RELOAD"] is None:
            self.jinja_env.auto_reload = value

    @setupmethod
    def register_blueprint(self, blueprint: Blueprint, **options: t.Any) -> None:
        """Register a :class:`~flask.Blueprint` on the application. Keyword
        arguments passed to this method will override the defaults set on the
        blueprint.

        Calls the blueprint's :meth:`~flask.Blueprint.register` method after
        recording the blueprint in the application's :attr:`blueprints`.

        :param blueprint: The blueprint to register.
        :param url_prefix: Blueprint routes will be prefixed with this.
        :param subdomain: Blueprint routes will match on this subdomain.
        :param url_defaults: Blueprint routes will use these default values for
            view arguments.
        :param options: Additional keyword arguments are passed to
            :class:`~flask.blueprints.BlueprintSetupState`. They can be
            accessed in :meth:`~flask.Blueprint.record` callbacks.

        .. versionchanged:: 2.0.1
            The ``name`` option can be used to change the (pre-dotted)
            name the blueprint is registered with. This allows the same
            blueprint to be registered multiple times with unique names
            for ``url_for``.

        .. versionadded:: 0.7
        """
        blueprint.register(self, options)

    def iter_blueprints(self) -> t.ValuesView[Blueprint]:
        """Iterates over all blueprints by the order they were registered.

        .. versionadded:: 0.11
        """
        return self.blueprints.values()

    @setupmethod
    def add_url_rule(
        self,
        rule: str,
        endpoint: str | None = None,
        view_func: ft.RouteCallable | None = None,
        provide_automatic_options: bool | None = None,
        **options: t.Any,
    ) -> None:
        if endpoint is None:
            endpoint = _endpoint_from_view_func(view_func)  # type: ignore
        options["endpoint"] = endpoint
        methods = options.pop("methods", None)

        # if the methods are not given and the view_func object knows its
        # methods we can use that instead.  If neither exists, we go with
        # a tuple of only ``GET`` as default.
        if methods is None:
            methods = getattr(view_func, "methods", None) or ("GET",)
        if isinstance(methods, str):
            raise TypeError(
                "Allowed methods must be a list of strings, for"
                ' example: @app.route(..., methods=["POST"])'
            )
        methods = {item.upper() for item in methods}

        # Methods that should always be added
        required_methods: set[str] = set(getattr(view_func, "required_methods", ()))

        # starting with Flask 0.8 the view_func object can disable and
        # force-enable the automatic options handling.
        if provide_automatic_options is None:
            provide_automatic_options = getattr(
                view_func, "provide_automatic_options", None
            )

        if provide_automatic_options is None:
            if "OPTIONS" not in methods and self.config["PROVIDE_AUTOMATIC_OPTIONS"]:
                provide_automatic_options = True
                required_methods.add("OPTIONS")
            else:
                provide_automatic_options = False

        # Add the required methods now.
        methods |= required_methods

        rule_obj = self.url_rule_class(rule, methods=methods, **options)
        rule_obj.provide_automatic_options = provide_automatic_options  # type: ignore[attr-defined]

        self.url_map.add(rule_obj)
        if view_func is not None:
            old_func = self.view_functions.get(endpoint)
            if old_func is not None and old_func != view_func:
                raise AssertionError(
                    "View function mapping is overwriting an existing"
                    f" endpoint function: {endpoint}"
                )
            self.view_functions[endpoint] = view_func

    @setupmethod
    def template_filter(
        self, name: str | None = None
    ) -> t.Callable[[T_template_filter], T_template_filter]:
        """A decorator that is used to register custom template filter.
        You can specify a name for the filter, otherwise the function
        name will be used. Example::

          @app.template_filter()
          def reverse(s):
              return s[::-1]

        :param name: the optional name of the filter, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_filter) -> T_template_filter:
            self.add_template_filter(f, name=name)
            return f

        return decorator

    @setupmethod
    def add_template_filter(
        self, f: ft.TemplateFilterCallable, name: str | None = None
    ) -> None:
        """Register a custom template filter.  Works exactly like the
        :meth:`template_filter` decorator.

        :param name: the optional name of the filter, otherwise the
                     function name will be used.
        """
        self.jinja_env.filters[name or f.__name__] = f

    @setupmethod
    def template_test(
        self, name: str | None = None
    ) -> t.Callable[[T_template_test], T_template_test]:
        """A decorator that is used to register custom template test.
        You can specify a name for the test, otherwise the function
        name will be used. Example::

          @app.template_test()
          def is_prime(n):
              if n == 2:
                  return True
              for i in range(2, int(math.ceil(math.sqrt(n))) + 1):
                  if n % i == 0:
                      return False
              return True

        .. versionadded:: 0.10

        :param name: the optional name of the test, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_test) -> T_template_test:
            self.add_template_test(f, name=name)
            return f

        return decorator

    @setupmethod
    def add_template_test(
        self, f: ft.TemplateTestCallable, name: str | None = None
    ) -> None:
        """Register a custom template test.  Works exactly like the
        :meth:`template_test` decorator.

        .. versionadded:: 0.10

        :param name: the optional name of the test, otherwise the
                     function name will be used.
        """
        self.jinja_env.tests[name or f.__name__] = f

    @setupmethod
    def template_global(
        self, name: str | None = None
    ) -> t.Callable[[T_template_global], T_template_global]:
        """A decorator that is used to register a custom template global function.
        You can specify a name for the global function, otherwise the function
        name will be used. Example::

            @app.template_global()
            def double(n):
                return 2 * n

        .. versionadded:: 0.10

        :param name: the optional name of the global function, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_global) -> T_template_global:
            self.add_template_global(f, name=name)
            return f

        return decorator

    @setupmethod
    def add_template_global(
        self, f: ft.TemplateGlobalCallable, name: str | None = None
    ) -> None:
        """Register a custom template global function. Works exactly like the
        :meth:`template_global` decorator.

        .. versionadded:: 0.10

        :param name: the optional name of the global function, otherwise the
                     function name will be used.
        """
        self.jinja_env.globals[name or f.__name__] = f

    @setupmethod
    def teardown_appcontext(self, f: T_teardown) -> T_teardown:
        """Registers a function to be called when the application
        context is popped. The application context is typically popped
        after the request context for each request, at the end of CLI
        commands, or after a manually pushed context ends.

        .. code-block:: python

            with app.app_context():
                ...

        When the ``with`` block exits (or ``ctx.pop()`` is called), the
        teardown functions are called just before the app context is
        made inactive. Since a request context typically also manages an
        application context it would also be called when you pop a
        request context.

        When a teardown function was called because of an unhandled
        exception it will be passed an error object. If an
        :meth:`errorhandler` is registered, it will handle the exception
        and the teardown will not receive it.

        Teardown functions must avoid raising exceptions. If they
        execute code that might fail they must surround that code with a
        ``try``/``except`` block and log any errors.

        The return values of teardown functions are ignored.

        .. versionadded:: 0.9
        """
        self.teardown_appcontext_funcs.append(f)
        return f

    @setupmethod
    def shell_context_processor(
        self, f: T_shell_context_processor
    ) -> T_shell_context_processor:
        """Registers a shell context processor function.

        .. versionadded:: 0.11
        """
        self.shell_context_processors.append(f)
        return f

    def _find_error_handler(
        self, e: Exception, blueprints: list[str]
    ) -> ft.ErrorHandlerCallable | None:
        """Return a registered error handler for an exception in this order:
        blueprint handler for a specific code, app handler for a specific code,
        blueprint handler for an exception class, app handler for an exception
        class, or ``None`` if a suitable handler is not found.
        """
        exc_class, code = self._get_exc_class_and_code(type(e))
        names = (*blueprints, None)

        for c in (code, None) if code is not None else (None,):
            for name in names:
                handler_map = self.error_handler_spec[name][c]

                if not handler_map:
                    continue

                for cls in exc_class.__mro__:
                    handler = handler_map.get(cls)

                    if handler is not None:
                        return handler
        return None

    def trap_http_exception(self, e: Exception) -> bool:
        """Checks if an HTTP exception should be trapped or not.  By default
        this will return ``False`` for all exceptions except for a bad request
        key error if ``TRAP_BAD_REQUEST_ERRORS`` is set to ``True``.  It
        also returns ``True`` if ``TRAP_HTTP_EXCEPTIONS`` is set to ``True``.

        This is called for all HTTP exceptions raised by a view function.
        If it returns ``True`` for any exception the error handler for this
        exception is not called and it shows up as regular exception in the
        traceback.  This is helpful for debugging implicitly raised HTTP
        exceptions.

        .. versionchanged:: 1.0
            Bad request errors are not trapped by default in debug mode.

        .. versionadded:: 0.8
        """
        if self.config["TRAP_HTTP_EXCEPTIONS"]:
            return True

        trap_bad_request = self.config["TRAP_BAD_REQUEST_ERRORS"]

        # if unset, trap key errors in debug mode
        if (
            trap_bad_request is None
            and self.debug
            and isinstance(e, BadRequestKeyError)
        ):
            return True

        if trap_bad_request:
            return isinstance(e, BadRequest)

        return False

    def should_ignore_error(self, error: BaseException | None) -> bool:
        """This is called to figure out if an error should be ignored
        or not as far as the teardown system is concerned.  If this
        function returns ``True`` then the teardown handlers will not be
        passed the error.

        .. versionadded:: 0.10
        """
        return False

    def redirect(self, location: str, code: int = 302) -> BaseResponse:
        """Create a redirect response object.

        This is called by :func:`flask.redirect`, and can be called
        directly as well.

        :param location: The URL to redirect to.
        :param code: The status code for the redirect.

        .. versionadded:: 2.2
            Moved from ``flask.redirect``, which calls this method.
        """
        return _wz_redirect(
            location,
            code=code,
            Response=self.response_class,  # type: ignore[arg-type]
        )

    def inject_url_defaults(self, endpoint: str, values: dict[str, t.Any]) -> None:
        """Injects the URL defaults for the given endpoint directly into
        the values dictionary passed.  This is used internally and
        automatically called on URL building.

        .. versionadded:: 0.7
        """
        names: t.Iterable[str | None] = (None,)

        # url_for may be called outside a request context, parse the
        # passed endpoint instead of using request.blueprints.
        if "." in endpoint:
            names = chain(
                names, reversed(_split_blueprint_path(endpoint.rpartition(".")[0]))
            )

        for name in names:
            if name in self.url_default_functions:
                for func in self.url_default_functions[name]:
                    func(endpoint, values)

    def handle_url_build_error(
        self, error: BuildError, endpoint: str, values: dict[str, t.Any]
    ) -> str:
        """Called by :meth:`.url_for` if a
        :exc:`~werkzeug.routing.BuildError` was raised. If this returns
        a value, it will be returned by ``url_for``, otherwise the error
        will be re-raised.

        Each function in :attr:`url_build_error_handlers` is called with
        ``error``, ``endpoint`` and ``values``. If a function returns
        ``None`` or raises a ``BuildError``, it is skipped. Otherwise,
        its return value is returned by ``url_for``.

        :param error: The active ``BuildError`` being handled.
        :param endpoint: The endpoint being built.
        :param values: The keyword arguments passed to ``url_for``.
        """
        for handler in self.url_build_error_handlers:
            try:
                rv = handler(error, endpoint, values)
            except BuildError as e:
                # make error available outside except block
                error = e
            else:
                if rv is not None:
                    return rv

        # Re-raise if called with an active exception, otherwise raise
        # the passed in exception.
        if error is sys.exc_info()[1]:
            raise

        raise error