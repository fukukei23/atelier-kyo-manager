
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\ctx_mp.py ===
"""
This module defines the mpf, mpc classes, and standard functions for
operating with them.
"""
__docformat__ = 'plaintext'

import functools

import re

from .ctx_base import StandardBaseContext

from .libmp.backend import basestring, BACKEND

from . import libmp

from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
    round_floor, round_ceiling, dps_to_prec, round_nearest, prec_to_dps,
    ComplexResult, to_pickable, from_pickable, normalize,
    from_int, from_float, from_str, to_int, to_float, to_str,
    from_rational, from_man_exp,
    fone, fzero, finf, fninf, fnan,
    mpf_abs, mpf_pos, mpf_neg, mpf_add, mpf_sub, mpf_mul, mpf_mul_int,
    mpf_div, mpf_rdiv_int, mpf_pow_int, mpf_mod,
    mpf_eq, mpf_cmp, mpf_lt, mpf_gt, mpf_le, mpf_ge,
    mpf_hash, mpf_rand,
    mpf_sum,
    bitcount, to_fixed,
    mpc_to_str,
    mpc_to_complex, mpc_hash, mpc_pos, mpc_is_nonzero, mpc_neg, mpc_conjugate,
    mpc_abs, mpc_add, mpc_add_mpf, mpc_sub, mpc_sub_mpf, mpc_mul, mpc_mul_mpf,
    mpc_mul_int, mpc_div, mpc_div_mpf, mpc_pow, mpc_pow_mpf, mpc_pow_int,
    mpc_mpf_div,
    mpf_pow,
    mpf_pi, mpf_degree, mpf_e, mpf_phi, mpf_ln2, mpf_ln10,
    mpf_euler, mpf_catalan, mpf_apery, mpf_khinchin,
    mpf_glaisher, mpf_twinprime, mpf_mertens,
    int_types)

from . import function_docs
from . import rational

new = object.__new__

get_complex = re.compile(r'^\(?(?P<re>[\+\-]?\d*(\.\d*)?(e[\+\-]?\d+)?)??'
                         r'(?P<im>[\+\-]?\d*(\.\d*)?(e[\+\-]?\d+)?j)?\)?$')

if BACKEND == 'sage':
    from sage.libs.mpmath.ext_main import Context as BaseMPContext
    # pickle hack
    import sage.libs.mpmath.ext_main as _mpf_module
else:
    from .ctx_mp_python import PythonMPContext as BaseMPContext
    from . import ctx_mp_python as _mpf_module

from .ctx_mp_python import _mpf, _mpc, mpnumeric

class MPContext(BaseMPContext, StandardBaseContext):
    """
    Context for multiprecision arithmetic with a global precision.
    """

    def __init__(ctx):
        BaseMPContext.__init__(ctx)
        ctx.trap_complex = False
        ctx.pretty = False
        ctx.types = [ctx.mpf, ctx.mpc, ctx.constant]
        ctx._mpq = rational.mpq
        ctx.default()
        StandardBaseContext.__init__(ctx)

        ctx.mpq = rational.mpq
        ctx.init_builtins()

        ctx.hyp_summators = {}

        ctx._init_aliases()

        # XXX: automate
        try:
            ctx.bernoulli.im_func.func_doc = function_docs.bernoulli
            ctx.primepi.im_func.func_doc = function_docs.primepi
            ctx.psi.im_func.func_doc = function_docs.psi
            ctx.atan2.im_func.func_doc = function_docs.atan2
        except AttributeError:
            # python 3
            ctx.bernoulli.__func__.func_doc = function_docs.bernoulli
            ctx.primepi.__func__.func_doc = function_docs.primepi
            ctx.psi.__func__.func_doc = function_docs.psi
            ctx.atan2.__func__.func_doc = function_docs.atan2

        ctx.digamma.func_doc = function_docs.digamma
        ctx.cospi.func_doc = function_docs.cospi
        ctx.sinpi.func_doc = function_docs.sinpi

    def init_builtins(ctx):

        mpf = ctx.mpf
        mpc = ctx.mpc

        # Exact constants
        ctx.one = ctx.make_mpf(fone)
        ctx.zero = ctx.make_mpf(fzero)
        ctx.j = ctx.make_mpc((fzero,fone))
        ctx.inf = ctx.make_mpf(finf)
        ctx.ninf = ctx.make_mpf(fninf)
        ctx.nan = ctx.make_mpf(fnan)

        eps = ctx.constant(lambda prec, rnd: (0, MPZ_ONE, 1-prec, 1),
            "epsilon of working precision", "eps")
        ctx.eps = eps

        # Approximate constants
        ctx.pi = ctx.constant(mpf_pi, "pi", "pi")
        ctx.ln2 = ctx.constant(mpf_ln2, "ln(2)", "ln2")
        ctx.ln10 = ctx.constant(mpf_ln10, "ln(10)", "ln10")
        ctx.phi = ctx.constant(mpf_phi, "Golden ratio phi", "phi")
        ctx.e = ctx.constant(mpf_e, "e = exp(1)", "e")
        ctx.euler = ctx.constant(mpf_euler, "Euler's constant", "euler")
        ctx.catalan = ctx.constant(mpf_catalan, "Catalan's constant", "catalan")
        ctx.khinchin = ctx.constant(mpf_khinchin, "Khinchin's constant", "khinchin")
        ctx.glaisher = ctx.constant(mpf_glaisher, "Glaisher's constant", "glaisher")
        ctx.apery = ctx.constant(mpf_apery, "Apery's constant", "apery")
        ctx.degree = ctx.constant(mpf_degree, "1 deg = pi / 180", "degree")
        ctx.twinprime = ctx.constant(mpf_twinprime, "Twin prime constant", "twinprime")
        ctx.mertens = ctx.constant(mpf_mertens, "Mertens' constant", "mertens")

        # Standard functions
        ctx.sqrt = ctx._wrap_libmp_function(libmp.mpf_sqrt, libmp.mpc_sqrt)
        ctx.cbrt = ctx._wrap_libmp_function(libmp.mpf_cbrt, libmp.mpc_cbrt)
        ctx.ln = ctx._wrap_libmp_function(libmp.mpf_log, libmp.mpc_log)
        ctx.atan = ctx._wrap_libmp_function(libmp.mpf_atan, libmp.mpc_atan)
        ctx.exp = ctx._wrap_libmp_function(libmp.mpf_exp, libmp.mpc_exp)
        ctx.expj = ctx._wrap_libmp_function(libmp.mpf_expj, libmp.mpc_expj)
        ctx.expjpi = ctx._wrap_libmp_function(libmp.mpf_expjpi, libmp.mpc_expjpi)
        ctx.sin = ctx._wrap_libmp_function(libmp.mpf_sin, libmp.mpc_sin)
        ctx.cos = ctx._wrap_libmp_function(libmp.mpf_cos, libmp.mpc_cos)
        ctx.tan = ctx._wrap_libmp_function(libmp.mpf_tan, libmp.mpc_tan)
        ctx.sinh = ctx._wrap_libmp_function(libmp.mpf_sinh, libmp.mpc_sinh)
        ctx.cosh = ctx._wrap_libmp_function(libmp.mpf_cosh, libmp.mpc_cosh)
        ctx.tanh = ctx._wrap_libmp_function(libmp.mpf_tanh, libmp.mpc_tanh)
        ctx.asin = ctx._wrap_libmp_function(libmp.mpf_asin, libmp.mpc_asin)
        ctx.acos = ctx._wrap_libmp_function(libmp.mpf_acos, libmp.mpc_acos)
        ctx.atan = ctx._wrap_libmp_function(libmp.mpf_atan, libmp.mpc_atan)
        ctx.asinh = ctx._wrap_libmp_function(libmp.mpf_asinh, libmp.mpc_asinh)
        ctx.acosh = ctx._wrap_libmp_function(libmp.mpf_acosh, libmp.mpc_acosh)
        ctx.atanh = ctx._wrap_libmp_function(libmp.mpf_atanh, libmp.mpc_atanh)
        ctx.sinpi = ctx._wrap_libmp_function(libmp.mpf_sin_pi, libmp.mpc_sin_pi)
        ctx.cospi = ctx._wrap_libmp_function(libmp.mpf_cos_pi, libmp.mpc_cos_pi)
        ctx.floor = ctx._wrap_libmp_function(libmp.mpf_floor, libmp.mpc_floor)
        ctx.ceil = ctx._wrap_libmp_function(libmp.mpf_ceil, libmp.mpc_ceil)
        ctx.nint = ctx._wrap_libmp_function(libmp.mpf_nint, libmp.mpc_nint)
        ctx.frac = ctx._wrap_libmp_function(libmp.mpf_frac, libmp.mpc_frac)
        ctx.fib = ctx.fibonacci = ctx._wrap_libmp_function(libmp.mpf_fibonacci, libmp.mpc_fibonacci)

        ctx.gamma = ctx._wrap_libmp_function(libmp.mpf_gamma, libmp.mpc_gamma)
        ctx.rgamma = ctx._wrap_libmp_function(libmp.mpf_rgamma, libmp.mpc_rgamma)
        ctx.loggamma = ctx._wrap_libmp_function(libmp.mpf_loggamma, libmp.mpc_loggamma)
        ctx.fac = ctx.factorial = ctx._wrap_libmp_function(libmp.mpf_factorial, libmp.mpc_factorial)

        ctx.digamma = ctx._wrap_libmp_function(libmp.mpf_psi0, libmp.mpc_psi0)
        ctx.harmonic = ctx._wrap_libmp_function(libmp.mpf_harmonic, libmp.mpc_harmonic)
        ctx.ei = ctx._wrap_libmp_function(libmp.mpf_ei, libmp.mpc_ei)
        ctx.e1 = ctx._wrap_libmp_function(libmp.mpf_e1, libmp.mpc_e1)
        ctx._ci = ctx._wrap_libmp_function(libmp.mpf_ci, libmp.mpc_ci)
        ctx._si = ctx._wrap_libmp_function(libmp.mpf_si, libmp.mpc_si)
        ctx.ellipk = ctx._wrap_libmp_function(libmp.mpf_ellipk, libmp.mpc_ellipk)
        ctx._ellipe = ctx._wrap_libmp_function(libmp.mpf_ellipe, libmp.mpc_ellipe)
        ctx.agm1 = ctx._wrap_libmp_function(libmp.mpf_agm1, libmp.mpc_agm1)
        ctx._erf = ctx._wrap_libmp_function(libmp.mpf_erf, None)
        ctx._erfc = ctx._wrap_libmp_function(libmp.mpf_erfc, None)
        ctx._zeta = ctx._wrap_libmp_function(libmp.mpf_zeta, libmp.mpc_zeta)
        ctx._altzeta = ctx._wrap_libmp_function(libmp.mpf_altzeta, libmp.mpc_altzeta)

        # Faster versions
        ctx.sqrt = getattr(ctx, "_sage_sqrt", ctx.sqrt)
        ctx.exp = getattr(ctx, "_sage_exp", ctx.exp)
        ctx.ln = getattr(ctx, "_sage_ln", ctx.ln)
        ctx.cos = getattr(ctx, "_sage_cos", ctx.cos)
        ctx.sin = getattr(ctx, "_sage_sin", ctx.sin)

    def to_fixed(ctx, x, prec):
        return x.to_fixed(prec)

    def hypot(ctx, x, y):
        r"""
        Computes the Euclidean norm of the vector `(x, y)`, equal
        to `\sqrt{x^2 + y^2}`. Both `x` and `y` must be real."""
        x = ctx.convert(x)
        y = ctx.convert(y)
        return ctx.make_mpf(libmp.mpf_hypot(x._mpf_, y._mpf_, *ctx._prec_rounding))

    def _gamma_upper_int(ctx, n, z):
        n = int(ctx._re(n))
        if n == 0:
            return ctx.e1(z)
        if not hasattr(z, '_mpf_'):
            raise NotImplementedError
        prec, rounding = ctx._prec_rounding
        real, imag = libmp.mpf_expint(n, z._mpf_, prec, rounding, gamma=True)
        if imag is None:
            return ctx.make_mpf(real)
        else:
            return ctx.make_mpc((real, imag))

    def _expint_int(ctx, n, z):
        n = int(n)
        if n == 1:
            return ctx.e1(z)
        if not hasattr(z, '_mpf_'):
            raise NotImplementedError
        prec, rounding = ctx._prec_rounding
        real, imag = libmp.mpf_expint(n, z._mpf_, prec, rounding)
        if imag is None:
            return ctx.make_mpf(real)
        else:
            return ctx.make_mpc((real, imag))

    def _nthroot(ctx, x, n):
        if hasattr(x, '_mpf_'):
            try:
                return ctx.make_mpf(libmp.mpf_nthroot(x._mpf_, n, *ctx._prec_rounding))
            except ComplexResult:
                if ctx.trap_complex:
                    raise
                x = (x._mpf_, libmp.fzero)
        else:
            x = x._mpc_
        return ctx.make_mpc(libmp.mpc_nthroot(x, n, *ctx._prec_rounding))

    def _besselj(ctx, n, z):
        prec, rounding = ctx._prec_rounding
        if hasattr(z, '_mpf_'):
            return ctx.make_mpf(libmp.mpf_besseljn(n, z._mpf_, prec, rounding))
        elif hasattr(z, '_mpc_'):
            return ctx.make_mpc(libmp.mpc_besseljn(n, z._mpc_, prec, rounding))

    def _agm(ctx, a, b=1):
        prec, rounding = ctx._prec_rounding
        if hasattr(a, '_mpf_') and hasattr(b, '_mpf_'):
            try:
                v = libmp.mpf_agm(a._mpf_, b._mpf_, prec, rounding)
                return ctx.make_mpf(v)
            except ComplexResult:
                pass
        if hasattr(a, '_mpf_'): a = (a._mpf_, libmp.fzero)
        else: a = a._mpc_
        if hasattr(b, '_mpf_'): b = (b._mpf_, libmp.fzero)
        else: b = b._mpc_
        return ctx.make_mpc(libmp.mpc_agm(a, b, prec, rounding))

    def bernoulli(ctx, n):
        return ctx.make_mpf(libmp.mpf_bernoulli(int(n), *ctx._prec_rounding))

    def _zeta_int(ctx, n):
        return ctx.make_mpf(libmp.mpf_zeta_int(int(n), *ctx._prec_rounding))

    def atan2(ctx, y, x):
        x = ctx.convert(x)
        y = ctx.convert(y)
        return ctx.make_mpf(libmp.mpf_atan2(y._mpf_, x._mpf_, *ctx._prec_rounding))

    def psi(ctx, m, z):
        z = ctx.convert(z)
        m = int(m)
        if ctx._is_real_type(z):
            return ctx.make_mpf(libmp.mpf_psi(m, z._mpf_, *ctx._prec_rounding))
        else:
            return ctx.make_mpc(libmp.mpc_psi(m, z._mpc_, *ctx._prec_rounding))

    def cos_sin(ctx, x, **kwargs):
        if type(x) not in ctx.types:
            x = ctx.convert(x)
        prec, rounding = ctx._parse_prec(kwargs)
        if hasattr(x, '_mpf_'):
            c, s = libmp.mpf_cos_sin(x._mpf_, prec, rounding)
            return ctx.make_mpf(c), ctx.make_mpf(s)
        elif hasattr(x, '_mpc_'):
            c, s = libmp.mpc_cos_sin(x._mpc_, prec, rounding)
            return ctx.make_mpc(c), ctx.make_mpc(s)
        else:
            return ctx.cos(x, **kwargs), ctx.sin(x, **kwargs)

    def cospi_sinpi(ctx, x, **kwargs):
        if type(x) not in ctx.types:
            x = ctx.convert(x)
        prec, rounding = ctx._parse_prec(kwargs)
        if hasattr(x, '_mpf_'):
            c, s = libmp.mpf_cos_sin_pi(x._mpf_, prec, rounding)
            return ctx.make_mpf(c), ctx.make_mpf(s)
        elif hasattr(x, '_mpc_'):
            c, s = libmp.mpc_cos_sin_pi(x._mpc_, prec, rounding)
            return ctx.make_mpc(c), ctx.make_mpc(s)
        else:
            return ctx.cos(x, **kwargs), ctx.sin(x, **kwargs)

    def clone(ctx):
        """
        Create a copy of the context, with the same working precision.
        """
        a = ctx.__class__()
        a.prec = ctx.prec
        return a

    # Several helper methods
    # TODO: add more of these, make consistent, write docstrings, ...

    def _is_real_type(ctx, x):
        if hasattr(x, '_mpc_') or type(x) is complex:
            return False
        return True

    def _is_complex_type(ctx, x):
        if hasattr(x, '_mpc_') or type(x) is complex:
            return True
        return False

    def isnan(ctx, x):
        """
        Return *True* if *x* is a NaN (not-a-number), or for a complex
        number, whether either the real or complex part is NaN;
        otherwise return *False*::

            >>> from mpmath import *
            >>> isnan(3.14)
            False
            >>> isnan(nan)
            True
            >>> isnan(mpc(3.14,2.72))
            False
            >>> isnan(mpc(3.14,nan))
            True

        """
        if hasattr(x, "_mpf_"):
            return x._mpf_ == fnan
        if hasattr(x, "_mpc_"):
            return fnan in x._mpc_
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return False
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isnan(x)
        raise TypeError("isnan() needs a number as input")

    def isfinite(ctx, x):
        """
        Return *True* if *x* is a finite number, i.e. neither
        an infinity or a NaN.

            >>> from mpmath import *
            >>> isfinite(inf)
            False
            >>> isfinite(-inf)
            False
            >>> isfinite(3)
            True
            >>> isfinite(nan)
            False
            >>> isfinite(3+4j)
            True
            >>> isfinite(mpc(3,inf))
            False
            >>> isfinite(mpc(nan,3))
            False

        """
        if ctx.isinf(x) or ctx.isnan(x):
            return False
        return True

    def isnpint(ctx, x):
        """
        Determine if *x* is a nonpositive integer.
        """
        if not x:
            return True
        if hasattr(x, '_mpf_'):
            sign, man, exp, bc = x._mpf_
            return sign and exp >= 0
        if hasattr(x, '_mpc_'):
            return not x.imag and ctx.isnpint(x.real)
        if type(x) in int_types:
            return x <= 0
        if isinstance(x, ctx.mpq):
            p, q = x._mpq_
            if not p:
                return True
            return q == 1 and p <= 0
        return ctx.isnpint(ctx.convert(x))

    def __str__(ctx):
        lines = ["Mpmath settings:",
            ("  mp.prec = %s" % ctx.prec).ljust(30) + "[default: 53]",
            ("  mp.dps = %s" % ctx.dps).ljust(30) + "[default: 15]",
            ("  mp.trap_complex = %s" % ctx.trap_complex).ljust(30) + "[default: False]",
        ]
        return "\n".join(lines)

    @property
    def _repr_digits(ctx):
        return repr_dps(ctx._prec)

    @property
    def _str_digits(ctx):
        return ctx._dps

    def extraprec(ctx, n, normalize_output=False):
        """
        The block

            with extraprec(n):
                <code>

        increases the precision n bits, executes <code>, and then
        restores the precision.

        extraprec(n)(f) returns a decorated version of the function f
        that increases the working precision by n bits before execution,
        and restores the parent precision afterwards. With
        normalize_output=True, it rounds the return value to the parent
        precision.
        """
        return PrecisionManager(ctx, lambda p: p + n, None, normalize_output)

    def extradps(ctx, n, normalize_output=False):
        """
        This function is analogous to extraprec (see documentation)
        but changes the decimal precision instead of the number of bits.
        """
        return PrecisionManager(ctx, None, lambda d: d + n, normalize_output)

    def workprec(ctx, n, normalize_output=False):
        """
        The block

            with workprec(n):
                <code>

        sets the precision to n bits, executes <code>, and then restores
        the precision.

        workprec(n)(f) returns a decorated version of the function f
        that sets the precision to n bits before execution,
        and restores the precision afterwards. With normalize_output=True,
        it rounds the return value to the parent precision.
        """
        return PrecisionManager(ctx, lambda p: n, None, normalize_output)

    def workdps(ctx, n, normalize_output=False):
        """
        This function is analogous to workprec (see documentation)
        but changes the decimal precision instead of the number of bits.
        """
        return PrecisionManager(ctx, None, lambda d: n, normalize_output)

    def autoprec(ctx, f, maxprec=None, catch=(), verbose=False):
        r"""
        Return a wrapped copy of *f* that repeatedly evaluates *f*
        with increasing precision until the result converges to the
        full precision used at the point of the call.

        This heuristically protects against rounding errors, at the cost of
        roughly a 2x slowdown compared to manually setting the optimal
        precision. This method can, however, easily be fooled if the results
        from *f* depend "discontinuously" on the precision, for instance
        if catastrophic cancellation can occur. Therefore, :func:`~mpmath.autoprec`
        should be used judiciously.

        **Examples**

        Many functions are sensitive to perturbations of the input arguments.
        If the arguments are decimal numbers, they may have to be converted
        to binary at a much higher precision. If the amount of required
        extra precision is unknown, :func:`~mpmath.autoprec` is convenient::

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> mp.pretty = True
            >>> besselj(5, 125 * 10**28)    # Exact input
            -8.03284785591801e-17
            >>> besselj(5, '1.25e30')   # Bad
            7.12954868316652e-16
            >>> autoprec(besselj)(5, '1.25e30')   # Good
            -8.03284785591801e-17

        The following fails to converge because `\sin(\pi) = 0` whereas all
        finite-precision approximations of `\pi` give nonzero values::

            >>> autoprec(sin)(pi) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: autoprec: prec increased to 2910 without convergence

        As the following example shows, :func:`~mpmath.autoprec` can protect against
        cancellation, but is fooled by too severe cancellation::

            >>> x = 1e-10
            >>> exp(x)-1; expm1(x); autoprec(lambda t: exp(t)-1)(x)
            1.00000008274037e-10
            1.00000000005e-10
            1.00000000005e-10
            >>> x = 1e-50
            >>> exp(x)-1; expm1(x); autoprec(lambda t: exp(t)-1)(x)
            0.0
            1.0e-50
            0.0

        With *catch*, an exception or list of exceptions to intercept
        may be specified. The raised exception is interpreted
        as signaling insufficient precision. This permits, for example,
        evaluating a function where a too low precision results in a
        division by zero::

            >>> f = lambda x: 1/(exp(x)-1)
            >>> f(1e-30)
            Traceback (most recent call last):
              ...
            ZeroDivisionError
            >>> autoprec(f, catch=ZeroDivisionError)(1e-30)
            1.0e+30


        """
        def f_autoprec_wrapped(*args, **kwargs):
            prec = ctx.prec
            if maxprec is None:
                maxprec2 = ctx._default_hyper_maxprec(prec)
            else:
                maxprec2 = maxprec
            try:
                ctx.prec = prec + 10
                try:
                    v1 = f(*args, **kwargs)
                except catch:
                    v1 = ctx.nan
                prec2 = prec + 20
                while 1:
                    ctx.prec = prec2
                    try:
                        v2 = f(*args, **kwargs)
                    except catch:
                        v2 = ctx.nan
                    if v1 == v2:
                        break
                    err = ctx.mag(v2-v1) - ctx.mag(v2)
                    if err < (-prec):
                        break
                    if verbose:
                        print("autoprec: target=%s, prec=%s, accuracy=%s" \
                            % (prec, prec2, -err))
                    v1 = v2
                    if prec2 >= maxprec2:
                        raise ctx.NoConvergence(\
                        "autoprec: prec increased to %i without convergence"\
                        % prec2)
                    prec2 += int(prec2*2)
                    prec2 = min(prec2, maxprec2)
            finally:
                ctx.prec = prec
            return +v2
        return f_autoprec_wrapped

    def nstr(ctx, x, n=6, **kwargs):
        """
        Convert an ``mpf`` or ``mpc`` to a decimal string literal with *n*
        significant digits. The small default value for *n* is chosen to
        make this function useful for printing collections of numbers
        (lists, matrices, etc).

        If *x* is a list or tuple, :func:`~mpmath.nstr` is applied recursively
        to each element. For unrecognized classes, :func:`~mpmath.nstr`
        simply returns ``str(x)``.

        The companion function :func:`~mpmath.nprint` prints the result
        instead of returning it.

        The keyword arguments *strip_zeros*, *min_fixed*, *max_fixed*
        and *show_zero_exponent* are forwarded to :func:`~mpmath.libmp.to_str`.

        The number will be printed in fixed-point format if the position
        of the leading digit is strictly between min_fixed
        (default = min(-dps/3,-5)) and max_fixed (default = dps).

        To force fixed-point format always, set min_fixed = -inf,
        max_fixed = +inf. To force floating-point format, set
        min_fixed >= max_fixed.

            >>> from mpmath import *
            >>> nstr([+pi, ldexp(1,-500)])
            '[3.14159, 3.05494e-151]'
            >>> nprint([+pi, ldexp(1,-500)])
            [3.14159, 3.05494e-151]
            >>> nstr(mpf("5e-10"), 5)
            '5.0e-10'
            >>> nstr(mpf("5e-10"), 5, strip_zeros=False)
            '5.0000e-10'
            >>> nstr(mpf("5e-10"), 5, strip_zeros=False, min_fixed=-11)
            '0.00000000050000'
            >>> nstr(mpf(0), 5, show_zero_exponent=True)
            '0.0e+0'

        """
        if isinstance(x, list):
            return "[%s]" % (", ".join(ctx.nstr(c, n, **kwargs) for c in x))
        if isinstance(x, tuple):
            return "(%s)" % (", ".join(ctx.nstr(c, n, **kwargs) for c in x))
        if hasattr(x, '_mpf_'):
            return to_str(x._mpf_, n, **kwargs)
        if hasattr(x, '_mpc_'):
            return "(" + mpc_to_str(x._mpc_, n, **kwargs)  + ")"
        if isinstance(x, basestring):
            return repr(x)
        if isinstance(x, ctx.matrix):
            return x.__nstr__(n, **kwargs)
        return str(x)

    def _convert_fallback(ctx, x, strings):
        if strings and isinstance(x, basestring):
            if 'j' in x.lower():
                x = x.lower().replace(' ', '')
                match = get_complex.match(x)
                re = match.group('re')
                if not re:
                    re = 0
                im = match.group('im').rstrip('j')
                return ctx.mpc(ctx.convert(re), ctx.convert(im))
        if hasattr(x, "_mpi_"):
            a, b = x._mpi_
            if a == b:
                return ctx.make_mpf(a)
            else:
                raise ValueError("can only create mpf from zero-width interval")
        raise TypeError("cannot create mpf from " + repr(x))

    def mpmathify(ctx, *args, **kwargs):
        return ctx.convert(*args, **kwargs)

    def _parse_prec(ctx, kwargs):
        if kwargs:
            if kwargs.get('exact'):
                return 0, 'f'
            prec, rounding = ctx._prec_rounding
            if 'rounding' in kwargs:
                rounding = kwargs['rounding']
            if 'prec' in kwargs:
                prec = kwargs['prec']
                if prec == ctx.inf:
                    return 0, 'f'
                else:
                    prec = int(prec)
            elif 'dps' in kwargs:
                dps = kwargs['dps']
                if dps == ctx.inf:
                    return 0, 'f'
                prec = dps_to_prec(dps)
            return prec, rounding
        return ctx._prec_rounding

    _exact_overflow_msg = "the exact result does not fit in memory"

    _hypsum_msg = """hypsum() failed to converge to the requested %i bits of accuracy
using a working precision of %i bits. Try with a higher maxprec,
maxterms, or set zeroprec."""

    def hypsum(ctx, p, q, flags, coeffs, z, accurate_small=True, **kwargs):
        if hasattr(z, "_mpf_"):
            key = p, q, flags, 'R'
            v = z._mpf_
        elif hasattr(z, "_mpc_"):
            key = p, q, flags, 'C'
            v = z._mpc_
        if key not in ctx.hyp_summators:
            ctx.hyp_summators[key] = libmp.make_hyp_summator(key)[1]
        summator = ctx.hyp_summators[key]
        prec = ctx.prec
        maxprec = kwargs.get('maxprec', ctx._default_hyper_maxprec(prec))
        extraprec = 50
        epsshift = 25
        # Jumps in magnitude occur when parameters are close to negative
        # integers. We must ensure that these terms are included in
        # the sum and added accurately
        magnitude_check = {}
        max_total_jump = 0
        for i, c in enumerate(coeffs):
            if flags[i] == 'Z':
                if i >= p and c <= 0:
                    ok = False
                    for ii, cc in enumerate(coeffs[:p]):
                        # Note: c <= cc or c < cc, depending on convention
                        if flags[ii] == 'Z' and cc <= 0 and c <= cc:
                            ok = True
                    if not ok:
                        raise ZeroDivisionError("pole in hypergeometric series")
                continue
            n, d = ctx.nint_distance(c)
            n = -int(n)
            d = -d
            if i >= p and n >= 0 and d > 4:
                if n in magnitude_check:
                    magnitude_check[n] += d
                else:
                    magnitude_check[n] = d
                extraprec = max(extraprec, d - prec + 60)
            max_total_jump += abs(d)
        while 1:
            if extraprec > maxprec:
                raise ValueError(ctx._hypsum_msg % (prec, prec+extraprec))
            wp = prec + extraprec
            if magnitude_check:
                mag_dict = dict((n,None) for n in magnitude_check)
            else:
                mag_dict = {}
            zv, have_complex, magnitude = summator(coeffs, v, prec, wp, \
                epsshift, mag_dict, **kwargs)
            cancel = -magnitude
            jumps_resolved = True
            if extraprec < max_total_jump:
                for n in mag_dict.values():
                    if (n is None) or (n < prec):
                        jumps_resolved = False
                        break
            accurate = (cancel < extraprec-25-5 or not accurate_small)
            if jumps_resolved:
                if accurate:
                    break
                # zero?
                zeroprec = kwargs.get('zeroprec')
                if zeroprec is not None:
                    if cancel > zeroprec:
                        if have_complex:
                            return ctx.mpc(0)
                        else:
                            return ctx.zero

            # Some near-singularities were not included, so increase
            # precision and repeat until they are
            extraprec *= 2
            # Possible workaround for bad roundoff in fixed-point arithmetic
            epsshift += 5
            extraprec += 5

        if type(zv) is tuple:
            if have_complex:
                return ctx.make_mpc(zv)
            else:
                return ctx.make_mpf(zv)
        else:
            return zv

    def ldexp(ctx, x, n):
        r"""
        Computes `x 2^n` efficiently. No rounding is performed.
        The argument `x` must be a real floating-point number (or
        possible to convert into one) and `n` must be a Python ``int``.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> ldexp(1, 10)
            mpf('1024.0')
            >>> ldexp(1, -3)
            mpf('0.125')

        """
        x = ctx.convert(x)
        return ctx.make_mpf(libmp.mpf_shift(x._mpf_, n))

    def frexp(ctx, x):
        r"""
        Given a real number `x`, returns `(y, n)` with `y \in [0.5, 1)`,
        `n` a Python integer, and such that `x = y 2^n`. No rounding is
        performed.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> frexp(7.5)
            (mpf('0.9375'), 3)

        """
        x = ctx.convert(x)
        y, n = libmp.mpf_frexp(x._mpf_)
        return ctx.make_mpf(y), n

    def fneg(ctx, x, **kwargs):
        """
        Negates the number *x*, giving a floating-point result, optionally
        using a custom precision and rounding mode.

        See the documentation of :func:`~mpmath.fadd` for a detailed description
        of how to specify precision and rounding.

        **Examples**

        An mpmath number is returned::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fneg(2.5)
            mpf('-2.5')
            >>> fneg(-5+2j)
            mpc(real='5.0', imag='-2.0')

        Precise control over rounding is possible::

            >>> x = fadd(2, 1e-100, exact=True)
            >>> fneg(x)
            mpf('-2.0')
            >>> fneg(x, rounding='f')
            mpf('-2.0000000000000004')

        Negating with and without roundoff::

            >>> n = 200000000000000000000001
            >>> print(int(-mpf(n)))
            -200000000000000016777216
            >>> print(int(fneg(n)))
            -200000000000000016777216
            >>> print(int(fneg(n, prec=log(n,2)+1)))
            -200000000000000000000001
            >>> print(int(fneg(n, dps=log(n,10)+1)))
            -200000000000000000000001
            >>> print(int(fneg(n, prec=inf)))
            -200000000000000000000001
            >>> print(int(fneg(n, dps=inf)))
            -200000000000000000000001
            >>> print(int(fneg(n, exact=True)))
            -200000000000000000000001

        """
        prec, rounding = ctx._parse_prec(kwargs)
        x = ctx.convert(x)
        if hasattr(x, '_mpf_'):
            return ctx.make_mpf(mpf_neg(x._mpf_, prec, rounding))
        if hasattr(x, '_mpc_'):
            return ctx.make_mpc(mpc_neg(x._mpc_, prec, rounding))
        raise ValueError("Arguments need to be mpf or mpc compatible numbers")

    def fadd(ctx, x, y, **kwargs):
        """
        Adds the numbers *x* and *y*, giving a floating-point result,
        optionally using a custom precision and rounding mode.

        The default precision is the working precision of the context.
        You can specify a custom precision in bits by passing the *prec* keyword
        argument, or by providing an equivalent decimal precision with the *dps*
        keyword argument. If the precision is set to ``+inf``, or if the flag
        *exact=True* is passed, an exact addition with no rounding is performed.

        When the precision is finite, the optional *rounding* keyword argument
        specifies the direction of rounding. Valid options are ``'n'`` for
        nearest (default), ``'f'`` for floor, ``'c'`` for ceiling, ``'d'``
        for down, ``'u'`` for up.

        **Examples**

        Using :func:`~mpmath.fadd` with precision and rounding control::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fadd(2, 1e-20)
            mpf('2.0')
            >>> fadd(2, 1e-20, rounding='u')
            mpf('2.0000000000000004')
            >>> nprint(fadd(2, 1e-20, prec=100), 25)
            2.00000000000000000001
            >>> nprint(fadd(2, 1e-20, dps=15), 25)
            2.0
            >>> nprint(fadd(2, 1e-20, dps=25), 25)
            2.00000000000000000001
            >>> nprint(fadd(2, 1e-20, exact=True), 25)
            2.00000000000000000001

        Exact addition avoids cancellation errors, enforcing familiar laws
        of numbers such as `x+y-x = y`, which don't hold in floating-point
        arithmetic with finite precision::

            >>> x, y = mpf(2), mpf('1e-1000')
            >>> print(x + y - x)
            0.0
            >>> print(fadd(x, y, prec=inf) - x)
            1.0e-1000
            >>> print(fadd(x, y, exact=True) - x)
            1.0e-1000

        Exact addition can be inefficient and may be impossible to perform
        with large magnitude differences::

            >>> fadd(1, '1e-100000000000000000000', prec=inf)
            Traceback (most recent call last):
              ...
            OverflowError: the exact result does not fit in memory

        """
        prec, rounding = ctx._parse_prec(kwargs)
        x = ctx.convert(x)
        y = ctx.convert(y)
        try:
            if hasattr(x, '_mpf_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpf(mpf_add(x._mpf_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_add_mpf(y._mpc_, x._mpf_, prec, rounding))
            if hasattr(x, '_mpc_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpc(mpc_add_mpf(x._mpc_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_add(x._mpc_, y._mpc_, prec, rounding))
        except (ValueError, OverflowError):
            raise OverflowError(ctx._exact_overflow_msg)
        raise ValueError("Arguments need to be mpf or mpc compatible numbers")

    def fsub(ctx, x, y, **kwargs):
        """
        Subtracts the numbers *x* and *y*, giving a floating-point result,
        optionally using a custom precision and rounding mode.

        See the documentation of :func:`~mpmath.fadd` for a detailed description
        of how to specify precision and rounding.

        **Examples**

        Using :func:`~mpmath.fsub` with precision and rounding control::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fsub(2, 1e-20)
            mpf('2.0')
            >>> fsub(2, 1e-20, rounding='d')
            mpf('1.9999999999999998')
            >>> nprint(fsub(2, 1e-20, prec=100), 25)
            1.99999999999999999999
            >>> nprint(fsub(2, 1e-20, dps=15), 25)
            2.0
            >>> nprint(fsub(2, 1e-20, dps=25), 25)
            1.99999999999999999999
            >>> nprint(fsub(2, 1e-20, exact=True), 25)
            1.99999999999999999999

        Exact subtraction avoids cancellation errors, enforcing familiar laws
        of numbers such as `x-y+y = x`, which don't hold in floating-point
        arithmetic with finite precision::

            >>> x, y = mpf(2), mpf('1e1000')
            >>> print(x - y + y)
            0.0
            >>> print(fsub(x, y, prec=inf) + y)
            2.0
            >>> print(fsub(x, y, exact=True) + y)
            2.0

        Exact addition can be inefficient and may be impossible to perform
        with large magnitude differences::

            >>> fsub(1, '1e-100000000000000000000', prec=inf)
            Traceback (most recent call last):
              ...
            OverflowError: the exact result does not fit in memory

        """
        prec, rounding = ctx._parse_prec(kwargs)
        x = ctx.convert(x)
        y = ctx.convert(y)
        try:
            if hasattr(x, '_mpf_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpf(mpf_sub(x._mpf_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_sub((x._mpf_, fzero), y._mpc_, prec, rounding))
            if hasattr(x, '_mpc_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpc(mpc_sub_mpf(x._mpc_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_sub(x._mpc_, y._mpc_, prec, rounding))
        except (ValueError, OverflowError):
            raise OverflowError(ctx._exact_overflow_msg)
        raise ValueError("Arguments need to be mpf or mpc compatible numbers")

    def fmul(ctx, x, y, **kwargs):
        """
        Multiplies the numbers *x* and *y*, giving a floating-point result,
        optionally using a custom precision and rounding mode.

        See the documentation of :func:`~mpmath.fadd` for a detailed description
        of how to specify precision and rounding.

        **Examples**

        The result is an mpmath number::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fmul(2, 5.0)
            mpf('10.0')
            >>> fmul(0.5j, 0.5)
            mpc(real='0.0', imag='0.25')

        Avoiding roundoff::

            >>> x, y = 10**10+1, 10**15+1
            >>> print(x*y)
            10000000001000010000000001
            >>> print(mpf(x) * mpf(y))
            1.0000000001e+25
            >>> print(int(mpf(x) * mpf(y)))
            10000000001000011026399232
            >>> print(int(fmul(x, y)))
            10000000001000011026399232
            >>> print(int(fmul(x, y, dps=25)))
            10000000001000010000000001
            >>> print(int(fmul(x, y, exact=True)))
            10000000001000010000000001

        Exact multiplication with complex numbers can be inefficient and may
        be impossible to perform with large magnitude differences between
        real and imaginary parts::

            >>> x = 1+2j
            >>> y = mpc(2, '1e-100000000000000000000')
            >>> fmul(x, y)
            mpc(real='2.0', imag='4.0')
            >>> fmul(x, y, rounding='u')
            mpc(real='2.0', imag='4.0000000000000009')
            >>> fmul(x, y, exact=True)
            Traceback (most recent call last):
              ...
            OverflowError: the exact result does not fit in memory

        """
        prec, rounding = ctx._parse_prec(kwargs)
        x = ctx.convert(x)
        y = ctx.convert(y)
        try:
            if hasattr(x, '_mpf_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpf(mpf_mul(x._mpf_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_mul_mpf(y._mpc_, x._mpf_, prec, rounding))
            if hasattr(x, '_mpc_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpc(mpc_mul_mpf(x._mpc_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_mul(x._mpc_, y._mpc_, prec, rounding))
        except (ValueError, OverflowError):
            raise OverflowError(ctx._exact_overflow_msg)
        raise ValueError("Arguments need to be mpf or mpc compatible numbers")

    def fdiv(ctx, x, y, **kwargs):
        """
        Divides the numbers *x* and *y*, giving a floating-point result,
        optionally using a custom precision and rounding mode.

        See the documentation of :func:`~mpmath.fadd` for a detailed description
        of how to specify precision and rounding.

        **Examples**

        The result is an mpmath number::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fdiv(3, 2)
            mpf('1.5')
            >>> fdiv(2, 3)
            mpf('0.66666666666666663')
            >>> fdiv(2+4j, 0.5)
            mpc(real='4.0', imag='8.0')

        The rounding direction and precision can be controlled::

            >>> fdiv(2, 3, dps=3)    # Should be accurate to at least 3 digits
            mpf('0.6666259765625')
            >>> fdiv(2, 3, rounding='d')
            mpf('0.66666666666666663')
            >>> fdiv(2, 3, prec=60)
            mpf('0.66666666666666667')
            >>> fdiv(2, 3, rounding='u')
            mpf('0.66666666666666674')

        Checking the error of a division by performing it at higher precision::

            >>> fdiv(2, 3) - fdiv(2, 3, prec=100)
            mpf('-3.7007434154172148e-17')

        Unlike :func:`~mpmath.fadd`, :func:`~mpmath.fmul`, etc., exact division is not
        allowed since the quotient of two floating-point numbers generally
        does not have an exact floating-point representation. (In the
        future this might be changed to allow the case where the division
        is actually exact.)

            >>> fdiv(2, 3, exact=True)
            Traceback (most recent call last):
              ...
            ValueError: division is not an exact operation

        """
        prec, rounding = ctx._parse_prec(kwargs)
        if not prec:
            raise ValueError("division is not an exact operation")
        x = ctx.convert(x)
        y = ctx.convert(y)
        if hasattr(x, '_mpf_'):
            if hasattr(y, '_mpf_'):
                return ctx.make_mpf(mpf_div(x._mpf_, y._mpf_, prec, rounding))
            if hasattr(y, '_mpc_'):
                return ctx.make_mpc(mpc_div((x._mpf_, fzero), y._mpc_, prec, rounding))
        if hasattr(x, '_mpc_'):
            if hasattr(y, '_mpf_'):
                return ctx.make_mpc(mpc_div_mpf(x._mpc_, y._mpf_, prec, rounding))
            if hasattr(y, '_mpc_'):
                return ctx.make_mpc(mpc_div(x._mpc_, y._mpc_, prec, rounding))
        raise ValueError("Arguments need to be mpf or mpc compatible numbers")

    def nint_distance(ctx, x):
        r"""
        Return `(n,d)` where `n` is the nearest integer to `x` and `d` is
        an estimate of `\log_2(|x-n|)`. If `d < 0`, `-d` gives the precision
        (measured in bits) lost to cancellation when computing `x-n`.

            >>> from mpmath import *
            >>> n, d = nint_distance(5)
            >>> print(n); print(d)
            5
            -inf
            >>> n, d = nint_distance(mpf(5))
            >>> print(n); print(d)
            5
            -inf
            >>> n, d = nint_distance(mpf(5.00000001))
            >>> print(n); print(d)
            5
            -26
            >>> n, d = nint_distance(mpf(4.99999999))
            >>> print(n); print(d)
            5
            -26
            >>> n, d = nint_distance(mpc(5,10))
            >>> print(n); print(d)
            5
            4
            >>> n, d = nint_distance(mpc(5,0.000001))
            >>> print(n); print(d)
            5
            -19

        """
        typx = type(x)
        if typx in int_types:
            return int(x), ctx.ninf
        elif typx is rational.mpq:
            p, q = x._mpq_
            n, r = divmod(p, q)
            if 2*r >= q:
                n += 1
            elif not r:
                return n, ctx.ninf
            # log(p/q-n) = log((p-nq)/q) = log(p-nq) - log(q)
            d = bitcount(abs(p-n*q)) - bitcount(q)
            return n, d
        if hasattr(x, "_mpf_"):
            re = x._mpf_
            im_dist = ctx.ninf
        elif hasattr(x, "_mpc_"):
            re, im = x._mpc_
            isign, iman, iexp, ibc = im
            if iman:
                im_dist = iexp + ibc
            elif im == fzero:
                im_dist = ctx.ninf
            else:
                raise ValueError("requires a finite number")
        else:
            x = ctx.convert(x)
            if hasattr(x, "_mpf_") or hasattr(x, "_mpc_"):
                return ctx.nint_distance(x)
            else:
                raise TypeError("requires an mpf/mpc")
        sign, man, exp, bc = re
        mag = exp+bc
        # |x| < 0.5
        if mag < 0:
            n = 0
            re_dist = mag
        elif man:
            # exact integer
            if exp >= 0:
                n = man << exp
                re_dist = ctx.ninf
            # exact half-integer
            elif exp == -1:
                n = (man>>1)+1
                re_dist = 0
            else:
                d = (-exp-1)
                t = man >> d
                if t & 1:
                    t += 1
                    man = (t<<d) - man
                else:
                    man -= (t<<d)
                n = t>>1   # int(t)>>1
                re_dist = exp+bitcount(man)
            if sign:
                n = -n
        elif re == fzero:
            re_dist = ctx.ninf
            n = 0
        else:
            raise ValueError("requires a finite number")
        return n, max(re_dist, im_dist)

    def fprod(ctx, factors):
        r"""
        Calculates a product containing a finite number of factors (for
        infinite products, see :func:`~mpmath.nprod`). The factors will be
        converted to mpmath numbers.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fprod([1, 2, 0.5, 7])
            mpf('7.0')

        """
        orig = ctx.prec
        try:
            v = ctx.one
            for p in factors:
                v *= p
        finally:
            ctx.prec = orig
        return +v

    def rand(ctx):
        """
        Returns an ``mpf`` with value chosen randomly from `[0, 1)`.
        The number of randomly generated bits in the mantissa is equal
        to the working precision.
        """
        return ctx.make_mpf(mpf_rand(ctx._prec))

    def fraction(ctx, p, q):
        """
        Given Python integers `(p, q)`, returns a lazy ``mpf`` representing
        the fraction `p/q`. The value is updated with the precision.

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> a = fraction(1,100)
            >>> b = mpf(1)/100
            >>> print(a); print(b)
            0.01
            0.01
            >>> mp.dps = 30
            >>> print(a); print(b)      # a will be accurate
            0.01
            0.0100000000000000002081668171172
            >>> mp.dps = 15
        """
        return ctx.constant(lambda prec, rnd: from_rational(p, q, prec, rnd),
            '%s/%s' % (p, q))

    def absmin(ctx, x):
        return abs(ctx.convert(x))

    def absmax(ctx, x):
        return abs(ctx.convert(x))

    def _as_points(ctx, x):
        # XXX: remove this?
        if hasattr(x, '_mpi_'):
            a, b = x._mpi_
            return [ctx.make_mpf(a), ctx.make_mpf(b)]
        return x

    '''
    def _zetasum(ctx, s, a, b):
        """
        Computes sum of k^(-s) for k = a, a+1, ..., b with a, b both small
        integers.
        """
        a = int(a)
        b = int(b)
        s = ctx.convert(s)
        prec, rounding = ctx._prec_rounding
        if hasattr(s, '_mpf_'):
            v = ctx.make_mpf(libmp.mpf_zetasum(s._mpf_, a, b, prec))
        elif hasattr(s, '_mpc_'):
            v = ctx.make_mpc(libmp.mpc_zetasum(s._mpc_, a, b, prec))
        return v
    '''

    def _zetasum_fast(ctx, s, a, n, derivatives=[0], reflect=False):
        if not (ctx.isint(a) and hasattr(s, "_mpc_")):
            raise NotImplementedError
        a = int(a)
        prec = ctx._prec
        xs, ys = libmp.mpc_zetasum(s._mpc_, a, n, derivatives, reflect, prec)
        xs = [ctx.make_mpc(x) for x in xs]
        ys = [ctx.make_mpc(y) for y in ys]
        return xs, ys

class PrecisionManager:
    def __init__(self, ctx, precfun, dpsfun, normalize_output=False):
        self.ctx = ctx
        self.precfun = precfun
        self.dpsfun = dpsfun
        self.normalize_output = normalize_output
    def __call__(self, f):
        @functools.wraps(f)
        def g(*args, **kwargs):
            orig = self.ctx.prec
            try:
                if self.precfun:
                    self.ctx.prec = self.precfun(self.ctx.prec)
                else:
                    self.ctx.dps = self.dpsfun(self.ctx.dps)
                if self.normalize_output:
                    v = f(*args, **kwargs)
                    if type(v) is tuple:
                        return tuple([+a for a in v])
                    return +v
                else:
                    return f(*args, **kwargs)
            finally:
                self.ctx.prec = orig
        return g
    def __enter__(self):
        self.origp = self.ctx.prec
        if self.precfun:
            self.ctx.prec = self.precfun(self.ctx.prec)
        else:
            self.ctx.dps = self.dpsfun(self.ctx.dps)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx.prec = self.origp
        return False


if __name__ == '__main__':
    import doctest
    doctest.testmod()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\prde.py ===
"""
Algorithms for solving Parametric Risch Differential Equations.

The methods used for solving Parametric Risch Differential Equations parallel
those for solving Risch Differential Equations.  See the outline in the
docstring of rde.py for more information.

The Parametric Risch Differential Equation problem is, given f, g1, ..., gm in
K(t), to determine if there exist y in K(t) and c1, ..., cm in Const(K) such
that Dy + f*y == Sum(ci*gi, (i, 1, m)), and to find such y and ci if they exist.

For the algorithms here G is a list of tuples of factions of the terms on the
right hand side of the equation (i.e., gi in k(t)), and Q is a list of terms on
the right hand side of the equation (i.e., qi in k[t]).  See the docstring of
each function for more information.
"""
import itertools
from functools import reduce

from sympy.core.intfunc import ilcm
from sympy.core import Dummy, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
    bound_degree)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
    residue_reduce, splitfactor, residue_reduce_derivation, DecrementLevel,
    recognize_log_derivative)
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve

zeros = Matrix.zeros
eye = Matrix.eye


def prde_normal_denom(fa, fd, G, DE):
    """
    Parametric Risch Differential Equation - Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g1, ..., gm in k(t) with f weakly
    normalized with respect to t, return the tuple (a, b, G, h) such that
    a, h in k[t], b in k<t>, G = [g1, ..., gm] in k(t)^m, and for any solution
    c1, ..., cm in Const(k) and y in k(t) of Dy + f*y == Sum(ci*gi, (i, 1, m)),
    q == y*h in k<t> satisfies a*Dq + b*q == Sum(ci*Gi, (i, 1, m)).
    """
    dn, ds = splitfactor(fd, DE)
    Gas, Gds = list(zip(*G))
    gd = reduce(lambda i, j: i.lcm(j), Gds, Poly(1, DE.t))
    en, es = splitfactor(gd, DE)

    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))

    a = dn*h
    c = a*h

    ba = a*fa - dn*derivation(h, DE)*fd
    ba, bd = ba.cancel(fd, include=True)

    G = [(c*A).cancel(D, include=True) for A, D in G]

    return (a, (ba, bd), G, h)

def real_imag(ba, bd, gen):
    """
    Helper function, to get the real and imaginary part of a rational function
    evaluated at sqrt(-1) without actually evaluating it at sqrt(-1).

    Explanation
    ===========

    Separates the even and odd power terms by checking the degree of terms wrt
    mod 4. Returns a tuple (ba[0], ba[1], bd) where ba[0] is real part
    of the numerator ba[1] is the imaginary part and bd is the denominator
    of the rational function.
    """
    bd = bd.as_poly(gen).as_dict()
    ba = ba.as_poly(gen).as_dict()
    denom_real = [value if key[0] % 4 == 0 else -value if key[0] % 4 == 2 else 0 for key, value in bd.items()]
    denom_imag = [value if key[0] % 4 == 1 else -value if key[0] % 4 == 3 else 0 for key, value in bd.items()]
    bd_real = sum(r for r in denom_real)
    bd_imag = sum(r for r in denom_imag)
    num_real = [value if key[0] % 4 == 0 else -value if key[0] % 4 == 2 else 0 for key, value in ba.items()]
    num_imag = [value if key[0] % 4 == 1 else -value if key[0] % 4 == 3 else 0 for key, value in ba.items()]
    ba_real = sum(r for r in num_real)
    ba_imag = sum(r for r in num_imag)
    ba = ((ba_real*bd_real + ba_imag*bd_imag).as_poly(gen), (ba_imag*bd_real - ba_real*bd_imag).as_poly(gen))
    bd = (bd_real*bd_real + bd_imag*bd_imag).as_poly(gen)
    return (ba[0], ba[1], bd)


def prde_special_denom(a, ba, bd, G, DE, case='auto'):
    """
    Parametric Risch Differential Equation - Special part of the denominator.

    Explanation
    ===========

    Case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the hyperexponential
    (resp. hypertangent) case, given a derivation D on k[t] and a in k[t],
    b in k<t>, and g1, ..., gm in k(t) with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the tuple (A, B, GG, h) such that A, B, h in
    k[t], GG = [gg1, ..., ggm] in k(t)^m, and for any solution c1, ..., cm in
    Const(k) and q in k<t> of a*Dq + b*q == Sum(ci*gi, (i, 1, m)), r == q*h in
    k[t] satisfies A*Dr + B*r == Sum(ci*ggi, (i, 1, m)).

    For case == 'primitive', k<t> == k[t], so it returns (a, b, G, 1) in this
    case.
    """
    # TODO: Merge this with the very similar special_denom() in rde.py
    if case == 'auto':
        case = DE.case

    if case == 'exp':
        p = Poly(DE.t, DE.t)
    elif case == 'tan':
        p = Poly(DE.t**2 + 1, DE.t)
    elif case in ('primitive', 'base'):
        B = ba.quo(bd)
        return (a, B, G, Poly(1, DE.t))
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'base'}, not %s." % case)

    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = min(order_at(Ga, p, DE.t) - order_at(Gd, p, DE.t) for Ga, Gd in G)
    n = min(0, nc - min(0, nb))
    if not nb:
        # Possible cancellation.
        if case == 'exp':
            dcoeff = DE.d.quo(Poly(DE.t, DE.t))
            with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                      # because case != 'base'.
                alphaa, alphad = frac_in(-ba.eval(0)/bd.eval(0)/a.eval(0), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    Q, m, z = A
                    if Q == 1:
                        n = min(n, m)

        elif case == 'tan':
            dcoeff = DE.d.quo(Poly(DE.t**2 + 1, DE.t))
            with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                      # because case != 'base'.
                betaa, alphaa, alphad =  real_imag(ba, bd*a, DE.t)
                betad = alphad
                etaa, etad = frac_in(dcoeff, DE.t)
                if recognize_log_derivative(Poly(2, DE.t)*betaa, betad, DE):
                    A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                    B = parametric_log_deriv(betaa, betad, etaa, etad, DE)
                    if A is not None and B is not None:
                        Q, s, z = A
                        # TODO: Add test
                        if Q == 1:
                            n = min(n, s/2)

    N = max(0, -nb)
    pN = p**N
    pn = p**-n  # This is 1/h

    A = a*pN
    B = ba*pN.quo(bd) + Poly(n, DE.t)*a*derivation(p, DE).quo(p)*pN
    G = [(Ga*pN*pn).cancel(Gd, include=True) for Ga, Gd in G]
    h = pn

    # (a*p**N, (b + n*a*Dp/p)*p**N, g1*p**(N - n), ..., gm*p**(N - n), p**-n)
    return (A, B, G, h)


def prde_linear_constraints(a, b, G, DE):
    """
    Parametric Risch Differential Equation - Generate linear constraints on the constants.

    Explanation
    ===========

    Given a derivation D on k[t], a, b, in k[t] with gcd(a, b) == 1, and
    G = [g1, ..., gm] in k(t)^m, return Q = [q1, ..., qm] in k[t]^m and a
    matrix M with entries in k(t) such that for any solution c1, ..., cm in
    Const(k) and p in k[t] of a*Dp + b*p == Sum(ci*gi, (i, 1, m)),
    (c1, ..., cm) is a solution of Mx == 0, and p and the ci satisfy
    a*Dp + b*p == Sum(ci*qi, (i, 1, m)).

    Because M has entries in k(t), and because Matrix does not play well with
    Poly, M will be a Matrix of Basic expressions.
    """
    m = len(G)

    Gns, Gds = list(zip(*G))
    d = reduce(lambda i, j: i.lcm(j), Gds)
    d = Poly(d, field=True)
    Q = [(ga*(d).quo(gd)).div(d) for ga, gd in G]

    if not all(ri.is_zero for _, ri in Q):
        N = max(ri.degree(DE.t) for _, ri in Q)
        M = Matrix(N + 1, m, lambda i, j: Q[j][1].nth(i), DE.t)
    else:
        M = Matrix(0, m, [], DE.t)  # No constraints, return the empty matrix.

    qs, _ = list(zip(*Q))
    return (qs, M)

def poly_linear_constraints(p, d):
    """
    Given p = [p1, ..., pm] in k[t]^m and d in k[t], return
    q = [q1, ..., qm] in k[t]^m and a matrix M with entries in k such
    that Sum(ci*pi, (i, 1, m)), for c1, ..., cm in k, is divisible
    by d if and only if (c1, ..., cm) is a solution of Mx = 0, in
    which case the quotient is Sum(ci*qi, (i, 1, m)).
    """
    m = len(p)
    q, r = zip(*[pi.div(d) for pi in p])

    if not all(ri.is_zero for ri in r):
        n = max(ri.degree() for ri in r)
        M = Matrix(n + 1, m, lambda i, j: r[j].nth(i), d.gens)
    else:
        M = Matrix(0, m, [], d.gens)  # No constraints.

    return q, M

def constant_system(A, u, DE):
    """
    Generate a system for the constant solutions.

    Explanation
    ===========

    Given a differential field (K, D) with constant field C = Const(K), a Matrix
    A, and a vector (Matrix) u with coefficients in K, returns the tuple
    (B, v, s), where B is a Matrix with coefficients in C and v is a vector
    (Matrix) such that either v has coefficients in C, in which case s is True
    and the solutions in C of Ax == u are exactly all the solutions of Bx == v,
    or v has a non-constant coefficient, in which case s is False Ax == u has no
    constant solution.

    This algorithm is used both in solving parametric problems and in
    determining if an element a of K is a derivative of an element of K or the
    logarithmic derivative of a K-radical using the structure theorem approach.

    Because Poly does not play well with Matrix yet, this algorithm assumes that
    all matrix entries are Basic expressions.
    """
    if not A:
        return A, u
    Au = A.row_join(u)
    Au, _ = Au.rref()
    # Warning: This will NOT return correct results if cancel() cannot reduce
    # an identically zero expression to 0.  The danger is that we might
    # incorrectly prove that an integral is nonelementary (such as
    # risch_integrate(exp((sin(x)**2 + cos(x)**2 - 1)*x**2), x).
    # But this is a limitation in computer algebra in general, and implicit
    # in the correctness of the Risch Algorithm is the computability of the
    # constant field (actually, this same correctness problem exists in any
    # algorithm that uses rref()).
    #
    # We therefore limit ourselves to constant fields that are computable
    # via the cancel() function, in order to prevent a speed bottleneck from
    # calling some more complex simplification function (rational function
    # coefficients will fall into this class).  Furthermore, (I believe) this
    # problem will only crop up if the integral explicitly contains an
    # expression in the constant field that is identically zero, but cannot
    # be reduced to such by cancel().  Therefore, a careful user can avoid this
    # problem entirely by being careful with the sorts of expressions that
    # appear in his integrand in the variables other than the integration
    # variable (the structure theorems should be able to completely decide these
    # problems in the integration variable).

    A, u = Au[:, :-1], Au[:, -1]

    D = lambda x: derivation(x, DE, basic=True)

    for j, i in itertools.product(range(A.cols), range(A.rows)):
        if A[i, j].expr.has(*DE.T):
            # This assumes that const(F(t0, ..., tn) == const(K) == F
            Ri = A[i, :]
            # Rm+1; m = A.rows
            DAij = D(A[i, j])
            Rm1 = Ri.applyfunc(lambda x: D(x) / DAij)
            um1 = D(u[i]) / DAij

            Aj = A[:, j]
            A = A - Aj * Rm1
            u = u - Aj * um1

            A = A.col_join(Rm1)
            u = u.col_join(Matrix([um1], u.gens))

    return (A, u)


def prde_spde(a, b, Q, n, DE):
    """
    Special Polynomial Differential Equation algorithm: Parametric Version.

    Explanation
    ===========

    Given a derivation D on k[t], an integer n, and a, b, q1, ..., qm in k[t]
    with deg(a) > 0 and gcd(a, b) == 1, return (A, B, Q, R, n1), with
    Qq = [q1, ..., qm] and R = [r1, ..., rm], such that for any solution
    c1, ..., cm in Const(k) and q in k[t] of degree at most n of
    a*Dq + b*q == Sum(ci*gi, (i, 1, m)), p = (q - Sum(ci*ri, (i, 1, m)))/a has
    degree at most n1 and satisfies A*Dp + B*p == Sum(ci*qi, (i, 1, m))
    """
    R, Z = list(zip(*[gcdex_diophantine(b, a, qi) for qi in Q]))

    A = a
    B = b + derivation(a, DE)
    Qq = [zi - derivation(ri, DE) for ri, zi in zip(R, Z)]
    R = list(R)
    n1 = n - a.degree(DE.t)

    return (A, B, Qq, R, n1)


def prde_no_cancel_b_large(b, Q, n, DE):
    """
    Parametric Poly Risch Differential Equation - No cancellation: deg(b) large enough.

    Explanation
    ===========

    Given a derivation D on k[t], n in ZZ, and b, q1, ..., qm in k[t] with
    b != 0 and either D == d/dt or deg(b) > max(0, deg(D) - 1), returns
    h1, ..., hr in k[t] and a matrix A with coefficients in Const(k) such that
    if c1, ..., cm in Const(k) and q in k[t] satisfy deg(q) <= n and
    Dq + b*q == Sum(ci*qi, (i, 1, m)), then q = Sum(dj*hj, (j, 1, r)), where
    d1, ..., dr in Const(k) and A*Matrix([[c1, ..., cm, d1, ..., dr]]).T == 0.
    """
    db = b.degree(DE.t)
    m = len(Q)
    H = [Poly(0, DE.t)]*m

    for N, i in itertools.product(range(n, -1, -1), range(m)):  # [n, ..., 0]
        si = Q[i].nth(N + db)/b.LC()
        sitn = Poly(si*DE.t**N, DE.t)
        H[i] = H[i] + sitn
        Q[i] = Q[i] - derivation(sitn, DE) - b*sitn

    if all(qi.is_zero for qi in Q):
        dc = -1
    else:
        dc = max(qi.degree(DE.t) for qi in Q)
    M = Matrix(dc + 1, m, lambda i, j: Q[j].nth(i), DE.t)
    A, u = constant_system(M, zeros(dc + 1, 1, DE.t), DE)
    c = eye(m, DE.t)
    A = A.row_join(zeros(A.rows, m, DE.t)).col_join(c.row_join(-c))

    return (H, A)


def prde_no_cancel_b_small(b, Q, n, DE):
    """
    Parametric Poly Risch Differential Equation - No cancellation: deg(b) small enough.

    Explanation
    ===========

    Given a derivation D on k[t], n in ZZ, and b, q1, ..., qm in k[t] with
    deg(b) < deg(D) - 1 and either D == d/dt or deg(D) >= 2, returns
    h1, ..., hr in k[t] and a matrix A with coefficients in Const(k) such that
    if c1, ..., cm in Const(k) and q in k[t] satisfy deg(q) <= n and
    Dq + b*q == Sum(ci*qi, (i, 1, m)) then q = Sum(dj*hj, (j, 1, r)) where
    d1, ..., dr in Const(k) and A*Matrix([[c1, ..., cm, d1, ..., dr]]).T == 0.
    """
    m = len(Q)
    H = [Poly(0, DE.t)]*m

    for N, i in itertools.product(range(n, 0, -1), range(m)):  # [n, ..., 1]
        si = Q[i].nth(N + DE.d.degree(DE.t) - 1)/(N*DE.d.LC())
        sitn = Poly(si*DE.t**N, DE.t)
        H[i] = H[i] + sitn
        Q[i] = Q[i] - derivation(sitn, DE) - b*sitn

    if b.degree(DE.t) > 0:
        for i in range(m):
            si = Poly(Q[i].nth(b.degree(DE.t))/b.LC(), DE.t)
            H[i] = H[i] + si
            Q[i] = Q[i] - derivation(si, DE) - b*si
        if all(qi.is_zero for qi in Q):
            dc = -1
        else:
            dc = max(qi.degree(DE.t) for qi in Q)
        M = Matrix(dc + 1, m, lambda i, j: Q[j].nth(i), DE.t)
        A, u = constant_system(M, zeros(dc + 1, 1, DE.t), DE)
        c = eye(m, DE.t)
        A = A.row_join(zeros(A.rows, m, DE.t)).col_join(c.row_join(-c))
        return (H, A)

    # else: b is in k, deg(qi) < deg(Dt)

    t = DE.t
    if DE.case != 'base':
        with DecrementLevel(DE):
            t0 = DE.t  # k = k0(t0)
            ba, bd = frac_in(b, t0, field=True)
            Q0 = [frac_in(qi.TC(), t0, field=True) for qi in Q]
            f, B = param_rischDE(ba, bd, Q0, DE)

            # f = [f1, ..., fr] in k^r and B is a matrix with
            # m + r columns and entries in Const(k) = Const(k0)
            # such that Dy0 + b*y0 = Sum(ci*qi, (i, 1, m)) has
            # a solution y0 in k with c1, ..., cm in Const(k)
            # if and only y0 = Sum(dj*fj, (j, 1, r)) where
            # d1, ..., dr ar in Const(k) and
            # B*Matrix([c1, ..., cm, d1, ..., dr]) == 0.

        # Transform fractions (fa, fd) in f into constant
        # polynomials fa/fd in k[t].
        # (Is there a better way?)
        f = [Poly(fa.as_expr()/fd.as_expr(), t, field=True)
             for fa, fd in f]
        B = Matrix.from_Matrix(B.to_Matrix(), t)
    else:
        # Base case. Dy == 0 for all y in k and b == 0.
        # Dy + b*y = Sum(ci*qi) is solvable if and only if
        # Sum(ci*qi) == 0 in which case the solutions are
        # y = d1*f1 for f1 = 1 and any d1 in Const(k) = k.

        f = [Poly(1, t, field=True)]  # r = 1
        B = Matrix([[qi.TC() for qi in Q] + [S.Zero]], DE.t)
        # The condition for solvability is
        # B*Matrix([c1, ..., cm, d1]) == 0
        # There are no constraints on d1.

    # Coefficients of t^j (j > 0) in Sum(ci*qi) must be zero.
    d = max(qi.degree(DE.t) for qi in Q)
    if d > 0:
        M = Matrix(d, m, lambda i, j: Q[j].nth(i + 1), DE.t)
        A, _ = constant_system(M, zeros(d, 1, DE.t), DE)
    else:
        # No constraints on the hj.
        A = Matrix(0, m, [], DE.t)

    # Solutions of the original equation are
    #    y = Sum(dj*fj, (j, 1, r) + Sum(ei*hi, (i, 1, m)),
    # where  ei == ci  (i = 1, ..., m),  when
    # A*Matrix([c1, ..., cm]) == 0 and
    # B*Matrix([c1, ..., cm, d1, ..., dr]) == 0

    # Build combined constraint matrix with m + r + m columns.

    r = len(f)
    I = eye(m, DE.t)
    A = A.row_join(zeros(A.rows, r + m, DE.t))
    B = B.row_join(zeros(B.rows, m, DE.t))
    C = I.row_join(zeros(m, r, DE.t)).row_join(-I)

    return f + H, A.col_join(B).col_join(C)


def prde_cancel_liouvillian(b, Q, n, DE):
    """
    Pg, 237.
    """
    H = []

    # Why use DecrementLevel? Below line answers that:
    # Assuming that we can solve such problems over 'k' (not k[t])
    if DE.case == 'primitive':
        with DecrementLevel(DE):
            ba, bd = frac_in(b, DE.t, field=True)

    for i in range(n, -1, -1):
        if DE.case == 'exp': # this re-checking can be avoided
            with DecrementLevel(DE):
                ba, bd = frac_in(b + (i*(derivation(DE.t, DE)/DE.t)).as_poly(b.gens),
                                DE.t, field=True)
        with DecrementLevel(DE):
            Qy = [frac_in(q.nth(i), DE.t, field=True) for q in Q]
            fi, Ai = param_rischDE(ba, bd, Qy, DE)
        fi = [Poly(fa.as_expr()/fd.as_expr(), DE.t, field=True)
                for fa, fd in fi]
        Ai = Ai.set_gens(DE.t)

        ri = len(fi)

        if i == n:
            M = Ai
        else:
            M = Ai.col_join(M.row_join(zeros(M.rows, ri, DE.t)))

        Fi, hi = [None]*ri, [None]*ri

        # from eq. on top of p.238 (unnumbered)
        for j in range(ri):
            hji = fi[j] * (DE.t**i).as_poly(fi[j].gens)
            hi[j] = hji
            # building up Sum(djn*(D(fjn*t^n) - b*fjnt^n))
            Fi[j] = -(derivation(hji, DE) - b*hji)

        H += hi
        # in the next loop instead of Q it has
        # to be Q + Fi taking its place
        Q = Q + Fi

    return (H, M)


def param_poly_rischDE(a, b, q, n, DE):
    """Polynomial solutions of a parametric Risch differential equation.

    Explanation
    ===========

    Given a derivation D in k[t], a, b in k[t] relatively prime, and q
    = [q1, ..., qm] in k[t]^m, return h = [h1, ..., hr] in k[t]^r and
    a matrix A with m + r columns and entries in Const(k) such that
    a*Dp + b*p = Sum(ci*qi, (i, 1, m)) has a solution p of degree <= n
    in k[t] with c1, ..., cm in Const(k) if and only if p = Sum(dj*hj,
    (j, 1, r)) where d1, ..., dr are in Const(k) and (c1, ..., cm,
    d1, ..., dr) is a solution of Ax == 0.
    """
    m = len(q)
    if n < 0:
        # Only the trivial zero solution is possible.
        # Find relations between the qi.
        if all(qi.is_zero for qi in q):
            return [], zeros(1, m, DE.t)  # No constraints.

        N = max(qi.degree(DE.t) for qi in q)
        M = Matrix(N + 1, m, lambda i, j: q[j].nth(i), DE.t)
        A, _ = constant_system(M, zeros(M.rows, 1, DE.t), DE)

        return [], A

    if a.is_ground:
        # Normalization: a = 1.
        a = a.LC()
        b, q = b.to_field().exquo_ground(a), [qi.to_field().exquo_ground(a) for qi in q]

        if not b.is_zero and (DE.case == 'base' or
                b.degree() > max(0, DE.d.degree() - 1)):
            return prde_no_cancel_b_large(b, q, n, DE)

        elif ((b.is_zero or b.degree() < DE.d.degree() - 1)
                and (DE.case == 'base' or DE.d.degree() >= 2)):
            return prde_no_cancel_b_small(b, q, n, DE)

        elif (DE.d.degree() >= 2 and
              b.degree() == DE.d.degree() - 1 and
              n > -b.as_poly().LC()/DE.d.as_poly().LC()):
            raise NotImplementedError("prde_no_cancel_b_equal() is "
                "not yet implemented.")

        else:
            # Liouvillian cases
            if DE.case in ('primitive', 'exp'):
                return prde_cancel_liouvillian(b, q, n, DE)
            else:
                raise NotImplementedError("non-linear and hypertangent "
                        "cases have not yet been implemented")

    # else: deg(a) > 0

    # Iterate SPDE as long as possible cumulating coefficient
    # and terms for the recovery of original solutions.
    alpha, beta = a.one, [a.zero]*m
    while n >= 0:  # and a, b relatively prime
        a, b, q, r, n = prde_spde(a, b, q, n, DE)
        beta = [betai + alpha*ri for betai, ri in zip(beta, r)]
        alpha *= a
        # Solutions p of a*Dp + b*p = Sum(ci*qi) correspond to
        # solutions alpha*p + Sum(ci*betai) of the initial equation.
        d = a.gcd(b)
        if not d.is_ground:
            break

    # a*Dp + b*p = Sum(ci*qi) may have a polynomial solution
    # only if the sum is divisible by d.

    qq, M = poly_linear_constraints(q, d)
    # qq = [qq1, ..., qqm] where qqi = qi.quo(d).
    # M is a matrix with m columns an entries in k.
    # Sum(fi*qi, (i, 1, m)), where f1, ..., fm are elements of k, is
    # divisible by d if and only if M*Matrix([f1, ..., fm]) == 0,
    # in which case the quotient is Sum(fi*qqi).

    A, _ = constant_system(M, zeros(M.rows, 1, DE.t), DE)
    # A is a matrix with m columns and entries in Const(k).
    # Sum(ci*qqi) is Sum(ci*qi).quo(d), and the remainder is zero
    # for c1, ..., cm in Const(k) if and only if
    # A*Matrix([c1, ...,cm]) == 0.

    V = A.nullspace()
    # V = [v1, ..., vu] where each vj is a column matrix with
    # entries aj1, ..., ajm in Const(k).
    # Sum(aji*qi) is divisible by d with exact quotient Sum(aji*qqi).
    # Sum(ci*qi) is divisible by d if and only if ci = Sum(dj*aji)
    # (i = 1, ..., m) for some d1, ..., du in Const(k).
    # In that case, solutions of
    #     a*Dp + b*p = Sum(ci*qi) = Sum(dj*Sum(aji*qi))
    # are the same as those of
    #     (a/d)*Dp + (b/d)*p = Sum(dj*rj)
    # where rj = Sum(aji*qqi).

    if not V:  # No non-trivial solution.
        return [], eye(m, DE.t)  # Could return A, but this has
                                 # the minimum number of rows.

    Mqq = Matrix([qq])  # A single row.
    r = [(Mqq*vj)[0] for vj in V]  # [r1, ..., ru]

    # Solutions of (a/d)*Dp + (b/d)*p = Sum(dj*rj) correspond to
    # solutions alpha*p + Sum(Sum(dj*aji)*betai) of the initial
    # equation. These are equal to alpha*p + Sum(dj*fj) where
    # fj = Sum(aji*betai).
    Mbeta = Matrix([beta])
    f = [(Mbeta*vj)[0] for vj in V]  # [f1, ..., fu]

    #
    # Solve the reduced equation recursively.
    #
    g, B = param_poly_rischDE(a.quo(d), b.quo(d), r, n, DE)

    # g = [g1, ..., gv] in k[t]^v and and B is a matrix with u + v
    # columns and entries in Const(k) such that
    # (a/d)*Dp + (b/d)*p = Sum(dj*rj) has a solution p of degree <= n
    # in k[t] if and only if p = Sum(ek*gk) where e1, ..., ev are in
    # Const(k) and B*Matrix([d1, ..., du, e1, ..., ev]) == 0.
    # The solutions of the original equation are then
    # Sum(dj*fj, (j, 1, u)) + alpha*Sum(ek*gk, (k, 1, v)).

    # Collect solution components.
    h = f + [alpha*gk for gk in g]

    # Build combined relation matrix.
    A = -eye(m, DE.t)
    for vj in V:
        A = A.row_join(vj)
    A = A.row_join(zeros(m, len(g), DE.t))
    A = A.col_join(zeros(B.rows, m, DE.t).row_join(B))

    return h, A


def param_rischDE(fa, fd, G, DE):
    """
    Solve a Parametric Risch Differential Equation: Dy + f*y == Sum(ci*Gi, (i, 1, m)).

    Explanation
    ===========

    Given a derivation D in k(t), f in k(t), and G
    = [G1, ..., Gm] in k(t)^m, return h = [h1, ..., hr] in k(t)^r and
    a matrix A with m + r columns and entries in Const(k) such that
    Dy + f*y = Sum(ci*Gi, (i, 1, m)) has a solution y
    in k(t) with c1, ..., cm in Const(k) if and only if y = Sum(dj*hj,
    (j, 1, r)) where d1, ..., dr are in Const(k) and (c1, ..., cm,
    d1, ..., dr) is a solution of Ax == 0.

    Elements of k(t) are tuples (a, d) with a and d in k[t].
    """
    m = len(G)
    q, (fa, fd) = weak_normalizer(fa, fd, DE)
    # Solutions of the weakly normalized equation Dz + f*z = q*Sum(ci*Gi)
    # correspond to solutions y = z/q of the original equation.
    gamma = q
    G = [(q*ga).cancel(gd, include=True) for ga, gd in G]

    a, (ba, bd), G, hn = prde_normal_denom(fa, fd, G, DE)
    # Solutions q in k<t> of  a*Dq + b*q = Sum(ci*Gi) correspond
    # to solutions z = q/hn of the weakly normalized equation.
    gamma *= hn

    A, B, G, hs = prde_special_denom(a, ba, bd, G, DE)
    # Solutions p in k[t] of  A*Dp + B*p = Sum(ci*Gi) correspond
    # to solutions q = p/hs of the previous equation.
    gamma *= hs

    g = A.gcd(B)
    a, b, g = A.quo(g), B.quo(g), [gia.cancel(gid*g, include=True) for
        gia, gid in G]

    # a*Dp + b*p = Sum(ci*gi)  may have a polynomial solution
    # only if the sum is in k[t].

    q, M = prde_linear_constraints(a, b, g, DE)

    # q = [q1, ..., qm] where qi in k[t] is the polynomial component
    # of the partial fraction expansion of gi.
    # M is a matrix with m columns and entries in k.
    # Sum(fi*gi, (i, 1, m)), where f1, ..., fm are elements of k,
    # is a polynomial if and only if M*Matrix([f1, ..., fm]) == 0,
    # in which case the sum is equal to Sum(fi*qi).

    M, _ = constant_system(M, zeros(M.rows, 1, DE.t), DE)
    # M is a matrix with m columns and entries in Const(k).
    # Sum(ci*gi) is in k[t] for c1, ..., cm in Const(k)
    # if and only if M*Matrix([c1, ..., cm]) == 0,
    # in which case the sum is Sum(ci*qi).

    ## Reduce number of constants at this point

    V = M.nullspace()
    # V = [v1, ..., vu] where each vj is a column matrix with
    # entries aj1, ..., ajm in Const(k).
    # Sum(aji*gi) is in k[t] and equal to Sum(aji*qi) (j = 1, ..., u).
    # Sum(ci*gi) is in k[t] if and only is ci = Sum(dj*aji)
    # (i = 1, ..., m) for some d1, ..., du in Const(k).
    # In that case,
    #     Sum(ci*gi) = Sum(ci*qi) = Sum(dj*Sum(aji*qi)) = Sum(dj*rj)
    # where rj = Sum(aji*qi) (j = 1, ..., u) in k[t].

    if not V:  # No non-trivial solution
        return [], eye(m, DE.t)

    Mq = Matrix([q])  # A single row.
    r = [(Mq*vj)[0] for vj in V]  # [r1, ..., ru]

    # Solutions of a*Dp + b*p = Sum(dj*rj) correspond to solutions
    # y = p/gamma of the initial equation with ci = Sum(dj*aji).

    try:
        # We try n=5. At least for prde_spde, it will always
        # terminate no matter what n is.
        n = bound_degree(a, b, r, DE, parametric=True)
    except NotImplementedError:
        # A temporary bound is set. Eventually, it will be removed.
        # the currently added test case takes large time
        # even with n=5, and much longer with large n's.
        n = 5

    h, B = param_poly_rischDE(a, b, r, n, DE)

    # h = [h1, ..., hv] in k[t]^v and and B is a matrix with u + v
    # columns and entries in Const(k) such that
    # a*Dp + b*p = Sum(dj*rj) has a solution p of degree <= n
    # in k[t] if and only if p = Sum(ek*hk) where e1, ..., ev are in
    # Const(k) and B*Matrix([d1, ..., du, e1, ..., ev]) == 0.
    # The solutions of the original equation for ci = Sum(dj*aji)
    # (i = 1, ..., m) are then y = Sum(ek*hk, (k, 1, v))/gamma.

    ## Build combined relation matrix with m + u + v columns.

    A = -eye(m, DE.t)
    for vj in V:
        A = A.row_join(vj)
    A = A.row_join(zeros(m, len(h), DE.t))
    A = A.col_join(zeros(B.rows, m, DE.t).row_join(B))

    ## Eliminate d1, ..., du.

    W = A.nullspace()

    # W = [w1, ..., wt] where each wl is a column matrix with
    # entries blk (k = 1, ..., m + u + v) in Const(k).
    # The vectors (bl1, ..., blm) generate the space of those
    # constant families (c1, ..., cm) for which a solution of
    # the equation Dy + f*y == Sum(ci*Gi) exists. They generate
    # the space and form a basis except possibly when Dy + f*y == 0
    # is solvable in k(t}. The corresponding solutions are
    # y = Sum(blk'*hk, (k, 1, v))/gamma, where k' = k + m + u.

    v = len(h)
    shape = (len(W), m+v)
    elements = [wl[:m] + wl[-v:] for wl in W] # excise dj's.
    items = [e for row in elements for e in row]

    # Need to set the shape in case W is empty
    M = Matrix(*shape, items, DE.t)
    N = M.nullspace()

    # N = [n1, ..., ns] where the ni in Const(k)^(m + v) are column
    # vectors generating the space of linear relations between
    # c1, ..., cm, e1, ..., ev.

    C = Matrix([ni[:] for ni in N], DE.t)  # rows n1, ..., ns.

    return [hk.cancel(gamma, include=True) for hk in h], C


def limited_integrate_reduce(fa, fd, G, DE):
    """
    Simpler version of step 1 & 2 for the limited integration problem.

    Explanation
    ===========

    Given a derivation D on k(t) and f, g1, ..., gn in k(t), return
    (a, b, h, N, g, V) such that a, b, h in k[t], N is a non-negative integer,
    g in k(t), V == [v1, ..., vm] in k(t)^m, and for any solution v in k(t),
    c1, ..., cm in C of f == Dv + Sum(ci*wi, (i, 1, m)), p = v*h is in k<t>, and
    p and the ci satisfy a*Dp + b*p == g + Sum(ci*vi, (i, 1, m)).  Furthermore,
    if S1irr == Sirr, then p is in k[t], and if t is nonlinear or Liouvillian
    over k, then deg(p) <= N.

    So that the special part is always computed, this function calls the more
    general prde_special_denom() automatically if it cannot determine that
    S1irr == Sirr.  Furthermore, it will automatically call bound_degree() when
    t is linear and non-Liouvillian, which for the transcendental case, implies
    that Dt == a*t + b with for some a, b in k*.
    """
    dn, ds = splitfactor(fd, DE)
    E = [splitfactor(gd, DE) for _, gd in G]
    En, Es = list(zip(*E))
    c = reduce(lambda i, j: i.lcm(j), (dn,) + En)  # lcm(dn, en1, ..., enm)
    hn = c.gcd(c.diff(DE.t))
    a = hn
    b = -derivation(hn, DE)
    N = 0

    # These are the cases where we know that S1irr = Sirr, but there could be
    # others, and this algorithm will need to be extended to handle them.
    if DE.case in ('base', 'primitive', 'exp', 'tan'):
        hs = reduce(lambda i, j: i.lcm(j), (ds,) + Es)  # lcm(ds, es1, ..., esm)
        a = hn*hs
        b -= (hn*derivation(hs, DE)).quo(hs)
        mu = min(order_at_oo(fa, fd, DE.t), min(order_at_oo(ga, gd, DE.t) for
            ga, gd in G))
        # So far, all the above are also nonlinear or Liouvillian, but if this
        # changes, then this will need to be updated to call bound_degree()
        # as per the docstring of this function (DE.case == 'other_linear').
        N = hn.degree(DE.t) + hs.degree(DE.t) + max(0, 1 - DE.d.degree(DE.t) - mu)
    else:
        # TODO: implement this
        raise NotImplementedError

    V = [(-a*hn*ga).cancel(gd, include=True) for ga, gd in G]
    return (a, b, a, N, (a*hn*fa).cancel(fd, include=True), V)


def limited_integrate(fa, fd, G, DE):
    """
    Solves the limited integration problem:  f = Dv + Sum(ci*wi, (i, 1, n))
    """
    fa, fd = fa*Poly(1/fd.LC(), DE.t), fd.monic()
    # interpreting limited integration problem as a
    # parametric Risch DE problem
    Fa = Poly(0, DE.t)
    Fd = Poly(1, DE.t)
    G = [(fa, fd)] + G
    h, A = param_rischDE(Fa, Fd, G, DE)
    V = A.nullspace()
    V = [v for v in V if v[0] != 0]
    if not V:
        return None
    else:
        # we can take any vector from V, we take V[0]
        c0 = V[0][0]
        # v = [-1, c1, ..., cm, d1, ..., dr]
        v = V[0]/(-c0)
        r = len(h)
        m = len(v) - r - 1
        C = list(v[1: m + 1])
        y = -sum(v[m + 1 + i]*h[i][0].as_expr()/h[i][1].as_expr() \
                for i in range(r))
        y_num, y_den = y.as_numer_denom()
        Ya, Yd = Poly(y_num, DE.t), Poly(y_den, DE.t)
        Y = Ya*Poly(1/Yd.LC(), DE.t), Yd.monic()
        return Y, C


def parametric_log_deriv_heu(fa, fd, wa, wd, DE, c1=None):
    """
    Parametric logarithmic derivative heuristic.

    Explanation
    ===========

    Given a derivation D on k[t], f in k(t), and a hyperexponential monomial
    theta over k(t), raises either NotImplementedError, in which case the
    heuristic failed, or returns None, in which case it has proven that no
    solution exists, or returns a solution (n, m, v) of the equation
    n*f == Dv/v + m*Dtheta/theta, with v in k(t)* and n, m in ZZ with n != 0.

    If this heuristic fails, the structure theorem approach will need to be
    used.

    The argument w == Dtheta/theta
    """
    # TODO: finish writing this and write tests
    c1 = c1 or Dummy('c1')

    p, a = fa.div(fd)
    q, b = wa.div(wd)

    B = max(0, derivation(DE.t, DE).degree(DE.t) - 1)
    C = max(p.degree(DE.t), q.degree(DE.t))

    if q.degree(DE.t) > B:
        eqs = [p.nth(i) - c1*q.nth(i) for i in range(B + 1, C + 1)]
        s = solve(eqs, c1)
        if not s or not s[c1].is_Rational:
            # deg(q) > B, no solution for c.
            return None

        M, N = s[c1].as_numer_denom()
        M_poly = M.as_poly(q.gens)
        N_poly = N.as_poly(q.gens)

        nfmwa = N_poly*fa*wd - M_poly*wa*fd
        nfmwd = fd*wd
        Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE, 'auto')
        if Qv is None:
            # (N*f - M*w) is not the logarithmic derivative of a k(t)-radical.
            return None

        Q, v = Qv

        if Q.is_zero or v.is_zero:
            return None

        return (Q*N, Q*M, v)

    if p.degree(DE.t) > B:
        return None

    c = lcm(fd.as_poly(DE.t).LC(), wd.as_poly(DE.t).LC())
    l = fd.monic().lcm(wd.monic())*Poly(c, DE.t)
    ln, ls = splitfactor(l, DE)
    z = ls*ln.gcd(ln.diff(DE.t))

    if not z.has(DE.t):
        # TODO: We treat this as 'no solution', until the structure
        # theorem version of parametric_log_deriv is implemented.
        return None

    u1, r1 = (fa*l.quo(fd)).div(z)  # (l*f).div(z)
    u2, r2 = (wa*l.quo(wd)).div(z)  # (l*w).div(z)

    eqs = [r1.nth(i) - c1*r2.nth(i) for i in range(z.degree(DE.t))]
    s = solve(eqs, c1)
    if not s or not s[c1].is_Rational:
        # deg(q) <= B, no solution for c.
        return None

    M, N = s[c1].as_numer_denom()

    nfmwa = N.as_poly(DE.t)*fa*wd - M.as_poly(DE.t)*wa*fd
    nfmwd = fd*wd
    Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE)
    if Qv is None:
        # (N*f - M*w) is not the logarithmic derivative of a k(t)-radical.
        return None

    Q, v = Qv

    if Q.is_zero or v.is_zero:
        return None

    return (Q*N, Q*M, v)


def parametric_log_deriv(fa, fd, wa, wd, DE):
    # TODO: Write the full algorithm using the structure theorems.
#    try:
    A = parametric_log_deriv_heu(fa, fd, wa, wd, DE)
#    except NotImplementedError:
        # Heuristic failed, we have to use the full method.
        # TODO: This could be implemented more efficiently.
        # It isn't too worrisome, because the heuristic handles most difficult
        # cases.
    return A


def is_deriv_k(fa, fd, DE):
    r"""
    Checks if Df/f is the derivative of an element of k(t).

    Explanation
    ===========

    a in k(t) is the derivative of an element of k(t) if there exists b in k(t)
    such that a = Db.  Either returns (ans, u), such that Df/f == Du, or None,
    which means that Df/f is not the derivative of an element of k(t).  ans is
    a list of tuples such that Add(*[i*j for i, j in ans]) == u.  This is useful
    for seeing exactly which elements of k(t) produce u.

    This function uses the structure theorem approach, which says that for any
    f in K, Df/f is the derivative of a element of K if and only if there are ri
    in QQ such that::

            ---               ---       Dt
            \    r  * Dt   +  \    r  *   i      Df
            /     i     i     /     i   ---   =  --.
            ---               ---        t        f
         i in L            i in E         i
               K/C(x)            K/C(x)


    Where C = Const(K), L_K/C(x) = { i in {1, ..., n} such that t_i is
    transcendental over C(x)(t_1, ..., t_i-1) and Dt_i = Da_i/a_i, for some a_i
    in C(x)(t_1, ..., t_i-1)* } (i.e., the set of all indices of logarithmic
    monomials of K over C(x)), and E_K/C(x) = { i in {1, ..., n} such that t_i
    is transcendental over C(x)(t_1, ..., t_i-1) and Dt_i/t_i = Da_i, for some
    a_i in C(x)(t_1, ..., t_i-1) } (i.e., the set of all indices of
    hyperexponential monomials of K over C(x)).  If K is an elementary extension
    over C(x), then the cardinality of L_K/C(x) U E_K/C(x) is exactly the
    transcendence degree of K over C(x).  Furthermore, because Const_D(K) ==
    Const_D(C(x)) == C, deg(Dt_i) == 1 when t_i is in E_K/C(x) and
    deg(Dt_i) == 0 when t_i is in L_K/C(x), implying in particular that E_K/C(x)
    and L_K/C(x) are disjoint.

    The sets L_K/C(x) and E_K/C(x) must, by their nature, be computed
    recursively using this same function.  Therefore, it is required to pass
    them as indices to D (or T).  E_args are the arguments of the
    hyperexponentials indexed by E_K (i.e., if i is in E_K, then T[i] ==
    exp(E_args[i])).  This is needed to compute the final answer u such that
    Df/f == Du.

    log(f) will be the same as u up to a additive constant.  This is because
    they will both behave the same as monomials. For example, both log(x) and
    log(2*x) == log(x) + log(2) satisfy Dt == 1/x, because log(2) is constant.
    Therefore, the term const is returned.  const is such that
    log(const) + f == u.  This is calculated by dividing the arguments of one
    logarithm from the other.  Therefore, it is necessary to pass the arguments
    of the logarithmic terms in L_args.

    To handle the case where we are given Df/f, not f, use is_deriv_k_in_field().

    See also
    ========
    is_log_deriv_k_t_radical_in_field, is_log_deriv_k_t_radical

    """
    # Compute Df/f
    dfa, dfd = (fd*derivation(fa, DE) - fa*derivation(fd, DE)), fd*fa
    dfa, dfd = dfa.cancel(dfd, include=True)

    # Our assumption here is that each monomial is recursively transcendental
    if len(DE.exts) != len(DE.D):
        if [i for i in DE.cases if i == 'tan'] or \
                ({i for i in DE.cases if i == 'primitive'} -
                        set(DE.indices('log'))):
            raise NotImplementedError("Real version of the structure "
                "theorems with hypertangent support is not yet implemented.")

        # TODO: What should really be done in this case?
        raise NotImplementedError("Nonelementary extensions not supported "
            "in the structure theorems.")

    E_part = [DE.D[i].quo(Poly(DE.T[i], DE.T[i])).as_expr() for i in DE.indices('exp')]
    L_part = [DE.D[i].as_expr() for i in DE.indices('log')]

    # The expression dfa/dfd might not be polynomial in any of its symbols so we
    # use a Dummy as the generator for PolyMatrix.
    dum = Dummy()
    lhs = Matrix([E_part + L_part], dum)
    rhs = Matrix([dfa.as_expr()/dfd.as_expr()], dum)

    A, u = constant_system(lhs, rhs, DE)

    u = u.to_Matrix()  # Poly to Expr

    if not A or not all(derivation(i, DE, basic=True).is_zero for i in u):
        # If the elements of u are not all constant
        # Note: See comment in constant_system

        # Also note: derivation(basic=True) calls cancel()
        return None
    else:
        if not all(i.is_Rational for i in u):
            raise NotImplementedError("Cannot work with non-rational "
                "coefficients in this case.")
        else:
            terms = ([DE.extargs[i] for i in DE.indices('exp')] +
                    [DE.T[i] for i in DE.indices('log')])
            ans = list(zip(terms, u))
            result = Add(*[Mul(i, j) for i, j in ans])
            argterms = ([DE.T[i] for i in DE.indices('exp')] +
                    [DE.extargs[i] for i in DE.indices('log')])
            l = []
            ld = []
            for i, j in zip(argterms, u):
                # We need to get around things like sqrt(x**2) != x
                # and also sqrt(x**2 + 2*x + 1) != x + 1
                # Issue 10798: i need not be a polynomial
                i, d = i.as_numer_denom()
                icoeff, iterms = sqf_list(i)
                l.append(Mul(*([Pow(icoeff, j)] + [Pow(b, e*j) for b, e in iterms])))
                dcoeff, dterms = sqf_list(d)
                ld.append(Mul(*([Pow(dcoeff, j)] + [Pow(b, e*j) for b, e in dterms])))
            const = cancel(fa.as_expr()/fd.as_expr()/Mul(*l)*Mul(*ld))

            return (ans, result, const)


def is_log_deriv_k_t_radical(fa, fd, DE, Df=True):
    r"""
    Checks if Df is the logarithmic derivative of a k(t)-radical.

    Explanation
    ===========

    b in k(t) can be written as the logarithmic derivative of a k(t) radical if
    there exist n in ZZ and u in k(t) with n, u != 0 such that n*b == Du/u.
    Either returns (ans, u, n, const) or None, which means that Df cannot be
    written as the logarithmic derivative of a k(t)-radical.  ans is a list of
    tuples such that Mul(*[i**j for i, j in ans]) == u.  This is useful for
    seeing exactly what elements of k(t) produce u.

    This function uses the structure theorem approach, which says that for any
    f in K, Df is the logarithmic derivative of a K-radical if and only if there
    are ri in QQ such that::

            ---               ---       Dt
            \    r  * Dt   +  \    r  *   i
            /     i     i     /     i   ---   =  Df.
            ---               ---        t
         i in L            i in E         i
               K/C(x)            K/C(x)


    Where C = Const(K), L_K/C(x) = { i in {1, ..., n} such that t_i is
    transcendental over C(x)(t_1, ..., t_i-1) and Dt_i = Da_i/a_i, for some a_i
    in C(x)(t_1, ..., t_i-1)* } (i.e., the set of all indices of logarithmic
    monomials of K over C(x)), and E_K/C(x) = { i in {1, ..., n} such that t_i
    is transcendental over C(x)(t_1, ..., t_i-1) and Dt_i/t_i = Da_i, for some
    a_i in C(x)(t_1, ..., t_i-1) } (i.e., the set of all indices of
    hyperexponential monomials of K over C(x)).  If K is an elementary extension
    over C(x), then the cardinality of L_K/C(x) U E_K/C(x) is exactly the
    transcendence degree of K over C(x).  Furthermore, because Const_D(K) ==
    Const_D(C(x)) == C, deg(Dt_i) == 1 when t_i is in E_K/C(x) and
    deg(Dt_i) == 0 when t_i is in L_K/C(x), implying in particular that E_K/C(x)
    and L_K/C(x) are disjoint.

    The sets L_K/C(x) and E_K/C(x) must, by their nature, be computed
    recursively using this same function.  Therefore, it is required to pass
    them as indices to D (or T).  L_args are the arguments of the logarithms
    indexed by L_K (i.e., if i is in L_K, then T[i] == log(L_args[i])).  This is
    needed to compute the final answer u such that n*f == Du/u.

    exp(f) will be the same as u up to a multiplicative constant.  This is
    because they will both behave the same as monomials.  For example, both
    exp(x) and exp(x + 1) == E*exp(x) satisfy Dt == t. Therefore, the term const
    is returned.  const is such that exp(const)*f == u.  This is calculated by
    subtracting the arguments of one exponential from the other.  Therefore, it
    is necessary to pass the arguments of the exponential terms in E_args.

    To handle the case where we are given Df, not f, use
    is_log_deriv_k_t_radical_in_field().

    See also
    ========

    is_log_deriv_k_t_radical_in_field, is_deriv_k

    """
    if Df:
        dfa, dfd = (fd*derivation(fa, DE) - fa*derivation(fd, DE)).cancel(fd**2,
            include=True)
    else:
        dfa, dfd = fa, fd

    # Our assumption here is that each monomial is recursively transcendental
    if len(DE.exts) != len(DE.D):
        if [i for i in DE.cases if i == 'tan'] or \
                ({i for i in DE.cases if i == 'primitive'} -
                        set(DE.indices('log'))):
            raise NotImplementedError("Real version of the structure "
                "theorems with hypertangent support is not yet implemented.")

        # TODO: What should really be done in this case?
        raise NotImplementedError("Nonelementary extensions not supported "
            "in the structure theorems.")

    E_part = [DE.D[i].quo(Poly(DE.T[i], DE.T[i])).as_expr() for i in DE.indices('exp')]
    L_part = [DE.D[i].as_expr() for i in DE.indices('log')]

    # The expression dfa/dfd might not be polynomial in any of its symbols so we
    # use a Dummy as the generator for PolyMatrix.
    dum = Dummy()
    lhs = Matrix([E_part + L_part], dum)
    rhs = Matrix([dfa.as_expr()/dfd.as_expr()], dum)

    A, u = constant_system(lhs, rhs, DE)

    u = u.to_Matrix()  # Poly to Expr

    if not A or not all(derivation(i, DE, basic=True).is_zero for i in u):
        # If the elements of u are not all constant
        # Note: See comment in constant_system

        # Also note: derivation(basic=True) calls cancel()
        return None
    else:
        if not all(i.is_Rational for i in u):
            # TODO: But maybe we can tell if they're not rational, like
            # log(2)/log(3). Also, there should be an option to continue
            # anyway, even if the result might potentially be wrong.
            raise NotImplementedError("Cannot work with non-rational "
                "coefficients in this case.")
        else:
            n = S.One*reduce(ilcm, [i.as_numer_denom()[1] for i in u])
            u *= n
            terms = ([DE.T[i] for i in DE.indices('exp')] +
                    [DE.extargs[i] for i in DE.indices('log')])
            ans = list(zip(terms, u))
            result = Mul(*[Pow(i, j) for i, j in ans])

            # exp(f) will be the same as result up to a multiplicative
            # constant.  We now find the log of that constant.
            argterms = ([DE.extargs[i] for i in DE.indices('exp')] +
                    [DE.T[i] for i in DE.indices('log')])
            const = cancel(fa.as_expr()/fd.as_expr() -
                Add(*[Mul(i, j/n) for i, j in zip(argterms, u)]))

            return (ans, result, n, const)


def is_log_deriv_k_t_radical_in_field(fa, fd, DE, case='auto', z=None):
    """
    Checks if f can be written as the logarithmic derivative of a k(t)-radical.

    Explanation
    ===========

    It differs from is_log_deriv_k_t_radical(fa, fd, DE, Df=False)
    for any given fa, fd, DE in that it finds the solution in the
    given field not in some (possibly unspecified extension) and
    "in_field" with the function name is used to indicate that.

    f in k(t) can be written as the logarithmic derivative of a k(t) radical if
    there exist n in ZZ and u in k(t) with n, u != 0 such that n*f == Du/u.
    Either returns (n, u) or None, which means that f cannot be written as the
    logarithmic derivative of a k(t)-radical.

    case is one of {'primitive', 'exp', 'tan', 'auto'} for the primitive,
    hyperexponential, and hypertangent cases, respectively.  If case is 'auto',
    it will attempt to determine the type of the derivation automatically.

    See also
    ========
    is_log_deriv_k_t_radical, is_deriv_k

    """
    fa, fd = fa.cancel(fd, include=True)

    # f must be simple
    n, s = splitfactor(fd, DE)
    if not s.is_one:
        pass

    z = z or Dummy('z')
    H, b = residue_reduce(fa, fd, DE, z=z)
    if not b:
        # I will have to verify, but I believe that the answer should be
        # None in this case. This should never happen for the
        # functions given when solving the parametric logarithmic
        # derivative problem when integration elementary functions (see
        # Bronstein's book, page 255), so most likely this indicates a bug.
        return None

    roots = [(i, i.real_roots()) for i, _ in H]
    if not all(len(j) == i.degree() and all(k.is_Rational for k in j) for
               i, j in roots):
        # If f is the logarithmic derivative of a k(t)-radical, then all the
        # roots of the resultant must be rational numbers.
        return None

    # [(a, i), ...], where i*log(a) is a term in the log-part of the integral
    # of f
    respolys, residues = list(zip(*roots)) or [[], []]
    # Note: this might be empty, but everything below should work find in that
    # case (it should be the same as if it were [[1, 1]])
    residueterms = [(H[j][1].subs(z, i), i) for j in range(len(H)) for
        i in residues[j]]

    # TODO: finish writing this and write tests

    p = cancel(fa.as_expr()/fd.as_expr() - residue_reduce_derivation(H, DE, z))

    p = p.as_poly(DE.t)
    if p is None:
        # f - Dg will be in k[t] if f is the logarithmic derivative of a k(t)-radical
        return None

    if p.degree(DE.t) >= max(1, DE.d.degree(DE.t)):
        return None

    if case == 'auto':
        case = DE.case

    if case == 'exp':
        wa, wd = derivation(DE.t, DE).cancel(Poly(DE.t, DE.t), include=True)
        with DecrementLevel(DE):
            pa, pd = frac_in(p, DE.t, cancel=True)
            wa, wd = frac_in((wa, wd), DE.t)
            A = parametric_log_deriv(pa, pd, wa, wd, DE)
        if A is None:
            return None
        n, e, u = A
        u *= DE.t**e

    elif case == 'primitive':
        with DecrementLevel(DE):
            pa, pd = frac_in(p, DE.t)
            A = is_log_deriv_k_t_radical_in_field(pa, pd, DE, case='auto')
        if A is None:
            return None
        n, u = A

    elif case == 'base':
        # TODO: we can use more efficient residue reduction from ratint()
        if not fd.is_sqf or fa.degree() >= fd.degree():
            # f is the logarithmic derivative in the base case if and only if
            # f = fa/fd, fd is square-free, deg(fa) < deg(fd), and
            # gcd(fa, fd) == 1.  The last condition is handled by cancel() above.
            return None
        # Note: if residueterms = [], returns (1, 1)
        # f had better be 0 in that case.
        n = S.One*reduce(ilcm, [i.as_numer_denom()[1] for _, i in residueterms], 1)
        u = Mul(*[Pow(i, j*n) for i, j in residueterms])
        return (n, u)

    elif case == 'tan':
        raise NotImplementedError("The hypertangent case is "
        "not yet implemented for is_log_deriv_k_t_radical_in_field()")

    elif case in ('other_linear', 'other_nonlinear'):
        # XXX: If these are supported by the structure theorems, change to NotImplementedError.
        raise ValueError("The %s case is not supported in this function." % case)

    else:
        raise ValueError("case must be one of {'primitive', 'exp', 'tan', "
        "'base', 'auto'}, not %s" % case)

    common_denom = S.One*reduce(ilcm, [i.as_numer_denom()[1] for i in [j for _, j in
        residueterms]] + [n], 1)
    residueterms = [(i, j*common_denom) for i, j in residueterms]
    m = common_denom//n
    if common_denom != n*m:  # Verify exact division
        raise ValueError("Inexact division")
    u = cancel(u**m*Mul(*[Pow(i, j) for i, j in residueterms]))

    return (common_denom, u)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\period.py ===
from __future__ import annotations

from datetime import timedelta
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    TypeVar,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    algos as libalgos,
    lib,
)
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Timedelta,
    add_overflowsafe,
    astype_overflowsafe,
    dt64arr_to_periodarr as c_dt64arr_to_periodarr,
    get_unit_from_dtype,
    iNaT,
    parsing,
    period as libperiod,
    to_offset,
)
from pandas._libs.tslibs.dtypes import (
    FreqGroup,
    PeriodDtypeBase,
    freq_to_period_freqstr,
)
from pandas._libs.tslibs.fields import isleapyear_arr
from pandas._libs.tslibs.offsets import (
    Tick,
    delta_to_tick,
)
from pandas._libs.tslibs.period import (
    DIFFERENT_FREQ,
    IncompatibleFrequency,
    Period,
    get_period_field_arr,
    period_asfreq_arr,
)
from pandas.util._decorators import (
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_object,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCPeriodIndex,
    ABCSeries,
    ABCTimedeltaArray,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays import datetimelike as dtl
import pandas.core.common as com

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        AnyArrayLike,
        Dtype,
        FillnaOptions,
        NpDtype,
        NumpySorter,
        NumpyValueArrayLike,
        Self,
        npt,
    )

    from pandas.core.arrays import (
        DatetimeArray,
        TimedeltaArray,
    )
    from pandas.core.arrays.base import ExtensionArray


BaseOffsetT = TypeVar("BaseOffsetT", bound=BaseOffset)


_shared_doc_kwargs = {
    "klass": "PeriodArray",
}


def _field_accessor(name: str, docstring: str | None = None):
    def f(self):
        base = self.dtype._dtype_code
        result = get_period_field_arr(name, self.asi8, base)
        return result

    f.__name__ = name
    f.__doc__ = docstring
    return property(f)


# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class PeriodArray(dtl.DatelikeOps, libperiod.PeriodMixin):  # type: ignore[misc]
    """
    Pandas ExtensionArray for storing Period data.

    Users should use :func:`~pandas.array` to create new instances.

    Parameters
    ----------
    values : Union[PeriodArray, Series[period], ndarray[int], PeriodIndex]
        The data to store. These should be arrays that can be directly
        converted to ordinals without inference or copy (PeriodArray,
        ndarray[int64]), or a box around such an array (Series[period],
        PeriodIndex).
    dtype : PeriodDtype, optional
        A PeriodDtype instance from which to extract a `freq`. If both
        `freq` and `dtype` are specified, then the frequencies must match.
    freq : str or DateOffset
        The `freq` to use for the array. Mostly applicable when `values`
        is an ndarray of integers, when `freq` is required. When `values`
        is a PeriodArray (or box around), it's checked that ``values.freq``
        matches `freq`.
    copy : bool, default False
        Whether to copy the ordinals before storing.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    Period: Represents a period of time.
    PeriodIndex : Immutable Index for period data.
    period_range: Create a fixed-frequency PeriodArray.
    array: Construct a pandas array.

    Notes
    -----
    There are two components to a PeriodArray

    - ordinals : integer ndarray
    - freq : pd.tseries.offsets.Offset

    The values are physically stored as a 1-D ndarray of integers. These are
    called "ordinals" and represent some kind of offset from a base.

    The `freq` indicates the span covered by each element of the array.
    All elements in the PeriodArray have the same `freq`.

    Examples
    --------
    >>> pd.arrays.PeriodArray(pd.PeriodIndex(['2023-01-01',
    ...                                       '2023-01-02'], freq='D'))
    <PeriodArray>
    ['2023-01-01', '2023-01-02']
    Length: 2, dtype: period[D]
    """

    # array priority higher than numpy scalars
    __array_priority__ = 1000
    _typ = "periodarray"  # ABCPeriodArray
    _internal_fill_value = np.int64(iNaT)
    _recognized_scalars = (Period,)
    _is_recognized_dtype = lambda x: isinstance(
        x, PeriodDtype
    )  # check_compatible_with checks freq match
    _infer_matches = ("period",)

    @property
    def _scalar_type(self) -> type[Period]:
        return Period

    # Names others delegate to us
    _other_ops: list[str] = []
    _bool_ops: list[str] = ["is_leap_year"]
    _object_ops: list[str] = ["start_time", "end_time", "freq"]
    _field_ops: list[str] = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "weekofyear",
        "weekday",
        "week",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
        "qyear",
        "days_in_month",
        "daysinmonth",
    ]
    _datetimelike_ops: list[str] = _field_ops + _object_ops + _bool_ops
    _datetimelike_methods: list[str] = ["strftime", "to_timestamp", "asfreq"]

    _dtype: PeriodDtype

    # --------------------------------------------------------------------
    # Constructors

    def __init__(
        self, values, dtype: Dtype | None = None, freq=None, copy: bool = False
    ) -> None:
        if freq is not None:
            # GH#52462
            warnings.warn(
                "The 'freq' keyword in the PeriodArray constructor is deprecated "
                "and will be removed in a future version. Pass 'dtype' instead",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            freq = validate_dtype_freq(dtype, freq)
            dtype = PeriodDtype(freq)

        if dtype is not None:
            dtype = pandas_dtype(dtype)
            if not isinstance(dtype, PeriodDtype):
                raise ValueError(f"Invalid dtype {dtype} for PeriodArray")

        if isinstance(values, ABCSeries):
            values = values._values
            if not isinstance(values, type(self)):
                raise TypeError("Incorrect dtype")

        elif isinstance(values, ABCPeriodIndex):
            values = values._values

        if isinstance(values, type(self)):
            if dtype is not None and dtype != values.dtype:
                raise raise_on_incompatible(values, dtype.freq)
            values, dtype = values._ndarray, values.dtype

        if not copy:
            values = np.asarray(values, dtype="int64")
        else:
            values = np.array(values, dtype="int64", copy=copy)
        if dtype is None:
            raise ValueError("dtype is not specified and cannot be inferred")
        dtype = cast(PeriodDtype, dtype)
        NDArrayBacked.__init__(self, values, dtype)

    # error: Signature of "_simple_new" incompatible with supertype "NDArrayBacked"
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls,
        values: npt.NDArray[np.int64],
        dtype: PeriodDtype,
    ) -> Self:
        # alias for PeriodArray.__init__
        assertion_msg = "Should be numpy array of type i8"
        assert isinstance(values, np.ndarray) and values.dtype == "i8", assertion_msg
        return cls(values, dtype=dtype)

    @classmethod
    def _from_sequence(
        cls,
        scalars,
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> Self:
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        if dtype and isinstance(dtype, PeriodDtype):
            freq = dtype.freq
        else:
            freq = None

        if isinstance(scalars, cls):
            validate_dtype_freq(scalars.dtype, freq)
            if copy:
                scalars = scalars.copy()
            return scalars

        periods = np.asarray(scalars, dtype=object)

        freq = freq or libperiod.extract_freq(periods)
        ordinals = libperiod.extract_ordinals(periods, freq)
        dtype = PeriodDtype(freq)
        return cls(ordinals, dtype=dtype)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_datetime64(cls, data, freq, tz=None) -> Self:
        """
        Construct a PeriodArray from a datetime64 array

        Parameters
        ----------
        data : ndarray[datetime64[ns], datetime64[ns, tz]]
        freq : str or Tick
        tz : tzinfo, optional

        Returns
        -------
        PeriodArray[freq]
        """
        if isinstance(freq, BaseOffset):
            freq = freq_to_period_freqstr(freq.n, freq.name)
        data, freq = dt64arr_to_periodarr(data, freq, tz)
        dtype = PeriodDtype(freq)
        return cls(data, dtype=dtype)

    @classmethod
    def _generate_range(cls, start, end, periods, freq):
        periods = dtl.validate_periods(periods)

        if freq is not None:
            freq = Period._maybe_convert_freq(freq)

        if start is not None or end is not None:
            subarr, freq = _get_ordinal_range(start, end, periods, freq)
        else:
            raise ValueError("Not enough parameters to construct Period range")

        return subarr, freq

    @classmethod
    def _from_fields(cls, *, fields: dict, freq) -> Self:
        subarr, freq = _range_from_fields(freq=freq, **fields)
        dtype = PeriodDtype(freq)
        return cls._simple_new(subarr, dtype=dtype)

    # -----------------------------------------------------------------
    # DatetimeLike Interface

    # error: Argument 1 of "_unbox_scalar" is incompatible with supertype
    # "DatetimeLikeArrayMixin"; supertype defines the argument type as
    # "Union[Union[Period, Any, Timedelta], NaTType]"
    def _unbox_scalar(  # type: ignore[override]
        self,
        value: Period | NaTType,
    ) -> np.int64:
        if value is NaT:
            # error: Item "Period" of "Union[Period, NaTType]" has no attribute "value"
            return np.int64(value._value)  # type: ignore[union-attr]
        elif isinstance(value, self._scalar_type):
            self._check_compatible_with(value)
            return np.int64(value.ordinal)
        else:
            raise ValueError(f"'value' should be a Period. Got '{value}' instead.")

    def _scalar_from_string(self, value: str) -> Period:
        return Period(value, freq=self.freq)

    # error: Argument 1 of "_check_compatible_with" is incompatible with
    # supertype "DatetimeLikeArrayMixin"; supertype defines the argument type
    # as "Period | Timestamp | Timedelta | NaTType"
    def _check_compatible_with(self, other: Period | NaTType | PeriodArray) -> None:  # type: ignore[override]
        if other is NaT:
            return
        # error: Item "NaTType" of "Period | NaTType | PeriodArray" has no
        # attribute "freq"
        self._require_matching_freq(other.freq)  # type: ignore[union-attr]

    # --------------------------------------------------------------------
    # Data / Attributes

    @cache_readonly
    def dtype(self) -> PeriodDtype:
        return self._dtype

    # error: Cannot override writeable attribute with read-only property
    @property  # type: ignore[override]
    def freq(self) -> BaseOffset:
        """
        Return the frequency object for this PeriodArray.
        """
        return self.dtype.freq

    @property
    def freqstr(self) -> str:
        return freq_to_period_freqstr(self.freq.n, self.freq.name)

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if dtype == "i8":
            # For NumPy 1.x compatibility we cannot use copy=None.  And
            # `copy=False` has the meaning of `copy=None` here:
            if not copy:
                return np.asarray(self.asi8, dtype=dtype)
            else:
                return np.array(self.asi8, dtype=dtype)

        if copy is False:
            warnings.warn(
                "Starting with NumPy 2.0, the behavior of the 'copy' keyword has "
                "changed and passing 'copy=False' raises an error when returning "
                "a zero-copy NumPy array is not possible. pandas will follow "
                "this behavior starting with pandas 3.0.\nThis conversion to "
                "NumPy requires a copy, but 'copy=False' was passed. Consider "
                "using 'np.asarray(..)' instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        if dtype == bool:
            return ~self._isnan

        # This will raise TypeError for non-object dtypes
        return np.array(list(self), dtype=object)

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow

        from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

        if type is not None:
            if pyarrow.types.is_integer(type):
                return pyarrow.array(self._ndarray, mask=self.isna(), type=type)
            elif isinstance(type, ArrowPeriodType):
                # ensure we have the same freq
                if self.freqstr != type.freq:
                    raise TypeError(
                        "Not supported to convert PeriodArray to array with different "
                        f"'freq' ({self.freqstr} vs {type.freq})"
                    )
            else:
                raise TypeError(
                    f"Not supported to convert PeriodArray to '{type}' type"
                )

        period_type = ArrowPeriodType(self.freqstr)
        storage_array = pyarrow.array(self._ndarray, mask=self.isna(), type="int64")
        return pyarrow.ExtensionArray.from_storage(period_type, storage_array)

    # --------------------------------------------------------------------
    # Vectorized analogues of Period properties

    year = _field_accessor(
        "year",
        """
        The year of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx.year
        Index([2023, 2024, 2025], dtype='int64')
        """,
    )
    month = _field_accessor(
        "month",
        """
        The month as January=1, December=12.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.month
        Index([1, 2, 3], dtype='int64')
        """,
    )
    day = _field_accessor(
        "day",
        """
        The days of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(['2020-01-31', '2020-02-28'], freq='D')
        >>> idx.day
        Index([31, 28], dtype='int64')
        """,
    )
    hour = _field_accessor(
        "hour",
        """
        The hour of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:00", "2023-01-01 11:00"], freq='h')
        >>> idx.hour
        Index([10, 11], dtype='int64')
        """,
    )
    minute = _field_accessor(
        "minute",
        """
        The minute of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:30:00",
        ...                       "2023-01-01 11:50:00"], freq='min')
        >>> idx.minute
        Index([30, 50], dtype='int64')
        """,
    )
    second = _field_accessor(
        "second",
        """
        The second of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:00:30",
        ...                       "2023-01-01 10:00:31"], freq='s')
        >>> idx.second
        Index([30, 31], dtype='int64')
        """,
    )
    weekofyear = _field_accessor(
        "week",
        """
        The week ordinal of the year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.week  # It can be written `weekofyear`
        Index([5, 9, 13], dtype='int64')
        """,
    )
    week = weekofyear
    day_of_week = _field_accessor(
        "day_of_week",
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01", "2023-01-02", "2023-01-03"], freq="D")
        >>> idx.weekday
        Index([6, 0, 1], dtype='int64')
        """,
    )
    dayofweek = day_of_week
    weekday = dayofweek
    dayofyear = day_of_year = _field_accessor(
        "day_of_year",
        """
        The ordinal day of the year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-10", "2023-02-01", "2023-03-01"], freq="D")
        >>> idx.dayofyear
        Index([10, 32, 60], dtype='int64')

        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx
        PeriodIndex(['2023', '2024', '2025'], dtype='period[Y-DEC]')
        >>> idx.dayofyear
        Index([365, 366, 365], dtype='int64')
        """,
    )
    quarter = _field_accessor(
        "quarter",
        """
        The quarter of the date.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.quarter
        Index([1, 1, 1], dtype='int64')
        """,
    )
    qyear = _field_accessor("qyear")
    days_in_month = _field_accessor(
        "days_in_month",
        """
        The number of days in the month.

        Examples
        --------
        For Series:

        >>> period = pd.period_range('2020-1-1 00:00', '2020-3-1 00:00', freq='M')
        >>> s = pd.Series(period)
        >>> s
        0   2020-01
        1   2020-02
        2   2020-03
        dtype: period[M]
        >>> s.dt.days_in_month
        0    31
        1    29
        2    31
        dtype: int64

        For PeriodIndex:

        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.days_in_month   # It can be also entered as `daysinmonth`
        Index([31, 28, 31], dtype='int64')
        """,
    )
    daysinmonth = days_in_month

    @property
    def is_leap_year(self) -> npt.NDArray[np.bool_]:
        """
        Logical indicating if the date belongs to a leap year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx.is_leap_year
        array([False,  True, False])
        """
        return isleapyear_arr(np.asarray(self.year))

    def to_timestamp(self, freq=None, how: str = "start") -> DatetimeArray:
        """
        Cast to DatetimeArray/Index.

        Parameters
        ----------
        freq : str or DateOffset, optional
            Target frequency. The default is 'D' for week or longer,
            's' otherwise.
        how : {'s', 'e', 'start', 'end'}
            Whether to use the start or end of the time period being converted.

        Returns
        -------
        DatetimeArray/Index

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.to_timestamp()
        DatetimeIndex(['2023-01-01', '2023-02-01', '2023-03-01'],
        dtype='datetime64[ns]', freq='MS')
        """
        from pandas.core.arrays import DatetimeArray

        how = libperiod.validate_end_alias(how)

        end = how == "E"
        if end:
            if freq == "B" or self.freq == "B":
                # roll forward to ensure we land on B date
                adjust = Timedelta(1, "D") - Timedelta(1, "ns")
                return self.to_timestamp(how="start") + adjust
            else:
                adjust = Timedelta(1, "ns")
                return (self + self.freq).to_timestamp(how="start") - adjust

        if freq is None:
            freq_code = self._dtype._get_to_timestamp_base()
            dtype = PeriodDtypeBase(freq_code, 1)
            freq = dtype._freqstr
            base = freq_code
        else:
            freq = Period._maybe_convert_freq(freq)
            base = freq._period_dtype_code

        new_parr = self.asfreq(freq, how=how)

        new_data = libperiod.periodarr_to_dt64arr(new_parr.asi8, base)
        dta = DatetimeArray._from_sequence(new_data)

        if self.freq.name == "B":
            # See if we can retain BDay instead of Day in cases where
            #  len(self) is too small for infer_freq to distinguish between them
            diffs = libalgos.unique_deltas(self.asi8)
            if len(diffs) == 1:
                diff = diffs[0]
                if diff == self.dtype._n:
                    dta._freq = self.freq
                elif diff == 1:
                    dta._freq = self.freq.base
                # TODO: other cases?
            return dta
        else:
            return dta._with_freq("infer")

    # --------------------------------------------------------------------

    def _box_func(self, x) -> Period | NaTType:
        return Period._from_ordinal(ordinal=x, freq=self.freq)

    @doc(**_shared_doc_kwargs, other="PeriodIndex", other_name="PeriodIndex")
    def asfreq(self, freq=None, how: str = "E") -> Self:
        """
        Convert the {klass} to the specified frequency `freq`.

        Equivalent to applying :meth:`pandas.Period.asfreq` with the given arguments
        to each :class:`~pandas.Period` in this {klass}.

        Parameters
        ----------
        freq : str
            A frequency.
        how : str {{'E', 'S'}}, default 'E'
            Whether the elements should be aligned to the end
            or start within pa period.

            * 'E', 'END', or 'FINISH' for end,
            * 'S', 'START', or 'BEGIN' for start.

            January 31st ('END') vs. January 1st ('START') for example.

        Returns
        -------
        {klass}
            The transformed {klass} with the new frequency.

        See Also
        --------
        {other}.asfreq: Convert each Period in a {other_name} to the given frequency.
        Period.asfreq : Convert a :class:`~pandas.Period` object to the given frequency.

        Examples
        --------
        >>> pidx = pd.period_range('2010-01-01', '2015-01-01', freq='Y')
        >>> pidx
        PeriodIndex(['2010', '2011', '2012', '2013', '2014', '2015'],
        dtype='period[Y-DEC]')

        >>> pidx.asfreq('M')
        PeriodIndex(['2010-12', '2011-12', '2012-12', '2013-12', '2014-12',
        '2015-12'], dtype='period[M]')

        >>> pidx.asfreq('M', how='S')
        PeriodIndex(['2010-01', '2011-01', '2012-01', '2013-01', '2014-01',
        '2015-01'], dtype='period[M]')
        """
        how = libperiod.validate_end_alias(how)
        if isinstance(freq, BaseOffset) and hasattr(freq, "_period_dtype_code"):
            freq = PeriodDtype(freq)._freqstr
        freq = Period._maybe_convert_freq(freq)

        base1 = self._dtype._dtype_code
        base2 = freq._period_dtype_code

        asi8 = self.asi8
        # self.freq.n can't be negative or 0
        end = how == "E"
        if end:
            ordinal = asi8 + self.dtype._n - 1
        else:
            ordinal = asi8

        new_data = period_asfreq_arr(ordinal, base1, base2, end)

        if self._hasna:
            new_data[self._isnan] = iNaT

        dtype = PeriodDtype(freq)
        return type(self)(new_data, dtype=dtype)

    # ------------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        if boxed:
            return str
        return "'{}'".format

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None, **kwargs
    ) -> npt.NDArray[np.object_]:
        """
        actually format my specific types
        """
        return libperiod.period_array_strftime(
            self.asi8, self.dtype._dtype_code, na_rep, date_format
        )

    # ------------------------------------------------------------------

    def astype(self, dtype, copy: bool = True):
        # We handle Period[T] -> Period[U]
        # Our parent handles everything else.
        dtype = pandas_dtype(dtype)
        if dtype == self._dtype:
            if not copy:
                return self
            else:
                return self.copy()
        if isinstance(dtype, PeriodDtype):
            return self.asfreq(dtype.freq)

        if lib.is_np_dtype(dtype, "M") or isinstance(dtype, DatetimeTZDtype):
            # GH#45038 match PeriodIndex behavior.
            tz = getattr(dtype, "tz", None)
            unit = dtl.dtype_to_unit(dtype)
            return self.to_timestamp().tz_localize(tz).as_unit(unit)

        return super().astype(dtype, copy=copy)

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        npvalue = self._validate_setitem_value(value).view("M8[ns]")

        # Cast to M8 to get datetime-like NaT placement,
        #  similar to dtl._period_dispatch
        m8arr = self._ndarray.view("M8[ns]")
        return m8arr.searchsorted(npvalue, side=side, sorter=sorter)

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # view as dt64 so we get treated as timelike in core.missing,
        #  similar to dtl._period_dispatch
        dta = self.view("M8[ns]")
        result = dta._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )
        if copy:
            return cast("Self", result.view(self.dtype))
        else:
            return self

    def fillna(
        self, value=None, method=None, limit: int | None = None, copy: bool = True
    ) -> Self:
        if method is not None:
            # view as dt64 so we get treated as timelike in core.missing,
            #  similar to dtl._period_dispatch
            dta = self.view("M8[ns]")
            result = dta.fillna(value=value, method=method, limit=limit, copy=copy)
            # error: Incompatible return value type (got "Union[ExtensionArray,
            # ndarray[Any, Any]]", expected "PeriodArray")
            return result.view(self.dtype)  # type: ignore[return-value]
        return super().fillna(value=value, method=method, limit=limit, copy=copy)

    # ------------------------------------------------------------------
    # Arithmetic Methods

    def _addsub_int_array_or_scalar(
        self, other: np.ndarray | int, op: Callable[[Any, Any], Any]
    ) -> Self:
        """
        Add or subtract array of integers.

        Parameters
        ----------
        other : np.ndarray[int64] or int
        op : {operator.add, operator.sub}

        Returns
        -------
        result : PeriodArray
        """
        assert op in [operator.add, operator.sub]
        if op is operator.sub:
            other = -other
        res_values = add_overflowsafe(self.asi8, np.asarray(other, dtype="i8"))
        return type(self)(res_values, dtype=self.dtype)

    def _add_offset(self, other: BaseOffset):
        assert not isinstance(other, Tick)

        self._require_matching_freq(other, base=True)
        return self._addsub_int_array_or_scalar(other.n, operator.add)

    # TODO: can we de-duplicate with Period._add_timedeltalike_scalar?
    def _add_timedeltalike_scalar(self, other):
        """
        Parameters
        ----------
        other : timedelta, Tick, np.timedelta64

        Returns
        -------
        PeriodArray
        """
        if not isinstance(self.freq, Tick):
            # We cannot add timedelta-like to non-tick PeriodArray
            raise raise_on_incompatible(self, other)

        if isna(other):
            # i.e. np.timedelta64("NaT")
            return super()._add_timedeltalike_scalar(other)

        td = np.asarray(Timedelta(other).asm8)
        return self._add_timedelta_arraylike(td)

    def _add_timedelta_arraylike(
        self, other: TimedeltaArray | npt.NDArray[np.timedelta64]
    ) -> Self:
        """
        Parameters
        ----------
        other : TimedeltaArray or ndarray[timedelta64]

        Returns
        -------
        PeriodArray
        """
        if not self.dtype._is_tick_like():
            # We cannot add timedelta-like to non-tick PeriodArray
            raise TypeError(
                f"Cannot add or subtract timedelta64[ns] dtype from {self.dtype}"
            )

        dtype = np.dtype(f"m8[{self.dtype._td64_unit}]")

        # Similar to _check_timedeltalike_freq_compat, but we raise with a
        #  more specific exception message if necessary.
        try:
            delta = astype_overflowsafe(
                np.asarray(other), dtype=dtype, copy=False, round_ok=False
            )
        except ValueError as err:
            # e.g. if we have minutes freq and try to add 30s
            # "Cannot losslessly convert units"
            raise IncompatibleFrequency(
                "Cannot add/subtract timedelta-like from PeriodArray that is "
                "not an integer multiple of the PeriodArray's freq."
            ) from err

        res_values = add_overflowsafe(self.asi8, np.asarray(delta.view("i8")))
        return type(self)(res_values, dtype=self.dtype)

    def _check_timedeltalike_freq_compat(self, other):
        """
        Arithmetic operations with timedelta-like scalars or array `other`
        are only valid if `other` is an integer multiple of `self.freq`.
        If the operation is valid, find that integer multiple.  Otherwise,
        raise because the operation is invalid.

        Parameters
        ----------
        other : timedelta, np.timedelta64, Tick,
                ndarray[timedelta64], TimedeltaArray, TimedeltaIndex

        Returns
        -------
        multiple : int or ndarray[int64]

        Raises
        ------
        IncompatibleFrequency
        """
        assert self.dtype._is_tick_like()  # checked by calling function

        dtype = np.dtype(f"m8[{self.dtype._td64_unit}]")

        if isinstance(other, (timedelta, np.timedelta64, Tick)):
            td = np.asarray(Timedelta(other).asm8)
        else:
            td = np.asarray(other)

        try:
            delta = astype_overflowsafe(td, dtype=dtype, copy=False, round_ok=False)
        except ValueError as err:
            raise raise_on_incompatible(self, other) from err

        delta = delta.view("i8")
        return lib.item_from_zerodim(delta)


def raise_on_incompatible(left, right) -> IncompatibleFrequency:
    """
    Helper function to render a consistent error message when raising
    IncompatibleFrequency.

    Parameters
    ----------
    left : PeriodArray
    right : None, DateOffset, Period, ndarray, or timedelta-like

    Returns
    -------
    IncompatibleFrequency
        Exception to be raised by the caller.
    """
    # GH#24283 error message format depends on whether right is scalar
    if isinstance(right, (np.ndarray, ABCTimedeltaArray)) or right is None:
        other_freq = None
    elif isinstance(right, BaseOffset):
        other_freq = freq_to_period_freqstr(right.n, right.name)
    elif isinstance(right, (ABCPeriodIndex, PeriodArray, Period)):
        other_freq = right.freqstr
    else:
        other_freq = delta_to_tick(Timedelta(right)).freqstr

    own_freq = freq_to_period_freqstr(left.freq.n, left.freq.name)
    msg = DIFFERENT_FREQ.format(
        cls=type(left).__name__, own_freq=own_freq, other_freq=other_freq
    )
    return IncompatibleFrequency(msg)


# -------------------------------------------------------------------
# Constructor Helpers


def period_array(
    data: Sequence[Period | str | None] | AnyArrayLike,
    freq: str | Tick | BaseOffset | None = None,
    copy: bool = False,
) -> PeriodArray:
    """
    Construct a new PeriodArray from a sequence of Period scalars.

    Parameters
    ----------
    data : Sequence of Period objects
        A sequence of Period objects. These are required to all have
        the same ``freq.`` Missing values can be indicated by ``None``
        or ``pandas.NaT``.
    freq : str, Tick, or Offset
        The frequency of every element of the array. This can be specified
        to avoid inferring the `freq` from `data`.
    copy : bool, default False
        Whether to ensure a copy of the data is made.

    Returns
    -------
    PeriodArray

    See Also
    --------
    PeriodArray
    pandas.PeriodIndex

    Examples
    --------
    >>> period_array([pd.Period('2017', freq='Y'),
    ...               pd.Period('2018', freq='Y')])
    <PeriodArray>
    ['2017', '2018']
    Length: 2, dtype: period[Y-DEC]

    >>> period_array([pd.Period('2017', freq='Y'),
    ...               pd.Period('2018', freq='Y'),
    ...               pd.NaT])
    <PeriodArray>
    ['2017', '2018', 'NaT']
    Length: 3, dtype: period[Y-DEC]

    Integers that look like years are handled

    >>> period_array([2000, 2001, 2002], freq='D')
    <PeriodArray>
    ['2000-01-01', '2001-01-01', '2002-01-01']
    Length: 3, dtype: period[D]

    Datetime-like strings may also be passed

    >>> period_array(['2000-Q1', '2000-Q2', '2000-Q3', '2000-Q4'], freq='Q')
    <PeriodArray>
    ['2000Q1', '2000Q2', '2000Q3', '2000Q4']
    Length: 4, dtype: period[Q-DEC]
    """
    data_dtype = getattr(data, "dtype", None)

    if lib.is_np_dtype(data_dtype, "M"):
        return PeriodArray._from_datetime64(data, freq)
    if isinstance(data_dtype, PeriodDtype):
        out = PeriodArray(data)
        if freq is not None:
            if freq == data_dtype.freq:
                return out
            return out.asfreq(freq)
        return out

    # other iterable of some kind
    if not isinstance(data, (np.ndarray, list, tuple, ABCSeries)):
        data = list(data)

    arrdata = np.asarray(data)

    dtype: PeriodDtype | None
    if freq:
        dtype = PeriodDtype(freq)
    else:
        dtype = None

    if arrdata.dtype.kind == "f" and len(arrdata) > 0:
        raise TypeError("PeriodIndex does not allow floating point in construction")

    if arrdata.dtype.kind in "iu":
        arr = arrdata.astype(np.int64, copy=False)
        # error: Argument 2 to "from_ordinals" has incompatible type "Union[str,
        # Tick, None]"; expected "Union[timedelta, BaseOffset, str]"
        ordinals = libperiod.from_ordinals(arr, freq)  # type: ignore[arg-type]
        return PeriodArray(ordinals, dtype=dtype)

    data = ensure_object(arrdata)
    if freq is None:
        freq = libperiod.extract_freq(data)
    dtype = PeriodDtype(freq)
    return PeriodArray._from_sequence(data, dtype=dtype)


@overload
def validate_dtype_freq(dtype, freq: BaseOffsetT) -> BaseOffsetT:
    ...


@overload
def validate_dtype_freq(dtype, freq: timedelta | str | None) -> BaseOffset:
    ...


def validate_dtype_freq(
    dtype, freq: BaseOffsetT | BaseOffset | timedelta | str | None
) -> BaseOffsetT:
    """
    If both a dtype and a freq are available, ensure they match.  If only
    dtype is available, extract the implied freq.

    Parameters
    ----------
    dtype : dtype
    freq : DateOffset or None

    Returns
    -------
    freq : DateOffset

    Raises
    ------
    ValueError : non-period dtype
    IncompatibleFrequency : mismatch between dtype and freq
    """
    if freq is not None:
        freq = to_offset(freq, is_period=True)

    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if not isinstance(dtype, PeriodDtype):
            raise ValueError("dtype must be PeriodDtype")
        if freq is None:
            freq = dtype.freq
        elif freq != dtype.freq:
            raise IncompatibleFrequency("specified freq and dtype are different")
    # error: Incompatible return value type (got "Union[BaseOffset, Any, None]",
    # expected "BaseOffset")
    return freq  # type: ignore[return-value]


def dt64arr_to_periodarr(
    data, freq, tz=None
) -> tuple[npt.NDArray[np.int64], BaseOffset]:
    """
    Convert an datetime-like array to values Period ordinals.

    Parameters
    ----------
    data : Union[Series[datetime64[ns]], DatetimeIndex, ndarray[datetime64ns]]
    freq : Optional[Union[str, Tick]]
        Must match the `freq` on the `data` if `data` is a DatetimeIndex
        or Series.
    tz : Optional[tzinfo]

    Returns
    -------
    ordinals : ndarray[int64]
    freq : Tick
        The frequency extracted from the Series or DatetimeIndex if that's
        used.

    """
    if not isinstance(data.dtype, np.dtype) or data.dtype.kind != "M":
        raise ValueError(f"Wrong dtype: {data.dtype}")

    if freq is None:
        if isinstance(data, ABCIndex):
            data, freq = data._values, data.freq
        elif isinstance(data, ABCSeries):
            data, freq = data._values, data.dt.freq

    elif isinstance(data, (ABCIndex, ABCSeries)):
        data = data._values

    reso = get_unit_from_dtype(data.dtype)
    freq = Period._maybe_convert_freq(freq)
    base = freq._period_dtype_code
    return c_dt64arr_to_periodarr(data.view("i8"), base, tz, reso=reso), freq


def _get_ordinal_range(start, end, periods, freq, mult: int = 1):
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )

    if freq is not None:
        freq = to_offset(freq, is_period=True)
        mult = freq.n

    if start is not None:
        start = Period(start, freq)
    if end is not None:
        end = Period(end, freq)

    is_start_per = isinstance(start, Period)
    is_end_per = isinstance(end, Period)

    if is_start_per and is_end_per and start.freq != end.freq:
        raise ValueError("start and end must have same freq")
    if start is NaT or end is NaT:
        raise ValueError("start and end must not be NaT")

    if freq is None:
        if is_start_per:
            freq = start.freq
        elif is_end_per:
            freq = end.freq
        else:  # pragma: no cover
            raise ValueError("Could not infer freq from start/end")
        mult = freq.n

    if periods is not None:
        periods = periods * mult
        if start is None:
            data = np.arange(
                end.ordinal - periods + mult, end.ordinal + 1, mult, dtype=np.int64
            )
        else:
            data = np.arange(
                start.ordinal, start.ordinal + periods, mult, dtype=np.int64
            )
    else:
        data = np.arange(start.ordinal, end.ordinal + 1, mult, dtype=np.int64)

    return data, freq


def _range_from_fields(
    year=None,
    month=None,
    quarter=None,
    day=None,
    hour=None,
    minute=None,
    second=None,
    freq=None,
) -> tuple[np.ndarray, BaseOffset]:
    if hour is None:
        hour = 0
    if minute is None:
        minute = 0
    if second is None:
        second = 0
    if day is None:
        day = 1

    ordinals = []

    if quarter is not None:
        if freq is None:
            freq = to_offset("Q", is_period=True)
            base = FreqGroup.FR_QTR.value
        else:
            freq = to_offset(freq, is_period=True)
            base = libperiod.freq_to_dtype_code(freq)
            if base != FreqGroup.FR_QTR.value:
                raise AssertionError("base must equal FR_QTR")

        freqstr = freq.freqstr
        year, quarter = _make_field_arrays(year, quarter)
        for y, q in zip(year, quarter):
            calendar_year, calendar_month = parsing.quarter_to_myear(y, q, freqstr)
            val = libperiod.period_ordinal(
                calendar_year, calendar_month, 1, 1, 1, 1, 0, 0, base
            )
            ordinals.append(val)
    else:
        freq = to_offset(freq, is_period=True)
        base = libperiod.freq_to_dtype_code(freq)
        arrays = _make_field_arrays(year, month, day, hour, minute, second)
        for y, mth, d, h, mn, s in zip(*arrays):
            ordinals.append(libperiod.period_ordinal(y, mth, d, h, mn, s, 0, 0, base))

    return np.array(ordinals, dtype=np.int64), freq


def _make_field_arrays(*fields) -> list[np.ndarray]:
    length = None
    for x in fields:
        if isinstance(x, (list, np.ndarray, ABCSeries)):
            if length is not None and len(x) != length:
                raise ValueError("Mismatched Period array lengths")
            if length is None:
                length = len(x)

    # error: Argument 2 to "repeat" has incompatible type "Optional[int]"; expected
    # "Union[Union[int, integer[Any]], Union[bool, bool_], ndarray, Sequence[Union[int,
    # integer[Any]]], Sequence[Union[bool, bool_]], Sequence[Sequence[Any]]]"
    return [
        np.asarray(x)
        if isinstance(x, (np.ndarray, list, ABCSeries))
        else np.repeat(x, length)  # type: ignore[arg-type]
        for x in fields
    ]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\database.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012-2023 The Python Software Foundation.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
"""PEP 376 implementation."""

from __future__ import unicode_literals

import base64
import codecs
import contextlib
import hashlib
import logging
import os
import posixpath
import sys
import zipimport

from . import DistlibException, resources
from .compat import StringIO
from .version import get_scheme, UnsupportedVersionError
from .metadata import (Metadata, METADATA_FILENAME, WHEEL_METADATA_FILENAME, LEGACY_METADATA_FILENAME)
from .util import (parse_requirement, cached_property, parse_name_and_version, read_exports, write_exports, CSVReader,
                   CSVWriter)

__all__ = [
    'Distribution', 'BaseInstalledDistribution', 'InstalledDistribution', 'EggInfoDistribution', 'DistributionPath'
]

logger = logging.getLogger(__name__)

EXPORTS_FILENAME = 'pydist-exports.json'
COMMANDS_FILENAME = 'pydist-commands.json'

DIST_FILES = ('INSTALLER', METADATA_FILENAME, 'RECORD', 'REQUESTED', 'RESOURCES', EXPORTS_FILENAME, 'SHARED')

DISTINFO_EXT = '.dist-info'


class _Cache(object):
    """
    A simple cache mapping names and .dist-info paths to distributions
    """

    def __init__(self):
        """
        Initialise an instance. There is normally one for each DistributionPath.
        """
        self.name = {}
        self.path = {}
        self.generated = False

    def clear(self):
        """
        Clear the cache, setting it to its initial state.
        """
        self.name.clear()
        self.path.clear()
        self.generated = False

    def add(self, dist):
        """
        Add a distribution to the cache.
        :param dist: The distribution to add.
        """
        if dist.path not in self.path:
            self.path[dist.path] = dist
            self.name.setdefault(dist.key, []).append(dist)


class DistributionPath(object):
    """
    Represents a set of distributions installed on a path (typically sys.path).
    """

    def __init__(self, path=None, include_egg=False):
        """
        Create an instance from a path, optionally including legacy (distutils/
        setuptools/distribute) distributions.
        :param path: The path to use, as a list of directories. If not specified,
                     sys.path is used.
        :param include_egg: If True, this instance will look for and return legacy
                            distributions as well as those based on PEP 376.
        """
        if path is None:
            path = sys.path
        self.path = path
        self._include_dist = True
        self._include_egg = include_egg

        self._cache = _Cache()
        self._cache_egg = _Cache()
        self._cache_enabled = True
        self._scheme = get_scheme('default')

    def _get_cache_enabled(self):
        return self._cache_enabled

    def _set_cache_enabled(self, value):
        self._cache_enabled = value

    cache_enabled = property(_get_cache_enabled, _set_cache_enabled)

    def clear_cache(self):
        """
        Clears the internal cache.
        """
        self._cache.clear()
        self._cache_egg.clear()

    def _yield_distributions(self):
        """
        Yield .dist-info and/or .egg(-info) distributions.
        """
        # We need to check if we've seen some resources already, because on
        # some Linux systems (e.g. some Debian/Ubuntu variants) there are
        # symlinks which alias other files in the environment.
        seen = set()
        for path in self.path:
            finder = resources.finder_for_path(path)
            if finder is None:
                continue
            r = finder.find('')
            if not r or not r.is_container:
                continue
            rset = sorted(r.resources)
            for entry in rset:
                r = finder.find(entry)
                if not r or r.path in seen:
                    continue
                try:
                    if self._include_dist and entry.endswith(DISTINFO_EXT):
                        possible_filenames = [METADATA_FILENAME, WHEEL_METADATA_FILENAME, LEGACY_METADATA_FILENAME]
                        for metadata_filename in possible_filenames:
                            metadata_path = posixpath.join(entry, metadata_filename)
                            pydist = finder.find(metadata_path)
                            if pydist:
                                break
                        else:
                            continue

                        with contextlib.closing(pydist.as_stream()) as stream:
                            metadata = Metadata(fileobj=stream, scheme='legacy')
                        logger.debug('Found %s', r.path)
                        seen.add(r.path)
                        yield new_dist_class(r.path, metadata=metadata, env=self)
                    elif self._include_egg and entry.endswith(('.egg-info', '.egg')):
                        logger.debug('Found %s', r.path)
                        seen.add(r.path)
                        yield old_dist_class(r.path, self)
                except Exception as e:
                    msg = 'Unable to read distribution at %s, perhaps due to bad metadata: %s'
                    logger.warning(msg, r.path, e)
                    import warnings
                    warnings.warn(msg % (r.path, e), stacklevel=2)

    def _generate_cache(self):
        """
        Scan the path for distributions and populate the cache with
        those that are found.
        """
        gen_dist = not self._cache.generated
        gen_egg = self._include_egg and not self._cache_egg.generated
        if gen_dist or gen_egg:
            for dist in self._yield_distributions():
                if isinstance(dist, InstalledDistribution):
                    self._cache.add(dist)
                else:
                    self._cache_egg.add(dist)

            if gen_dist:
                self._cache.generated = True
            if gen_egg:
                self._cache_egg.generated = True

    @classmethod
    def distinfo_dirname(cls, name, version):
        """
        The *name* and *version* parameters are converted into their
        filename-escaped form, i.e. any ``'-'`` characters are replaced
        with ``'_'`` other than the one in ``'dist-info'`` and the one
        separating the name from the version number.

        :parameter name: is converted to a standard distribution name by replacing
                         any runs of non- alphanumeric characters with a single
                         ``'-'``.
        :type name: string
        :parameter version: is converted to a standard version string. Spaces
                            become dots, and all other non-alphanumeric characters
                            (except dots) become dashes, with runs of multiple
                            dashes condensed to a single dash.
        :type version: string
        :returns: directory name
        :rtype: string"""
        name = name.replace('-', '_')
        return '-'.join([name, version]) + DISTINFO_EXT

    def get_distributions(self):
        """
        Provides an iterator that looks for distributions and returns
        :class:`InstalledDistribution` or
        :class:`EggInfoDistribution` instances for each one of them.

        :rtype: iterator of :class:`InstalledDistribution` and
                :class:`EggInfoDistribution` instances
        """
        if not self._cache_enabled:
            for dist in self._yield_distributions():
                yield dist
        else:
            self._generate_cache()

            for dist in self._cache.path.values():
                yield dist

            if self._include_egg:
                for dist in self._cache_egg.path.values():
                    yield dist

    def get_distribution(self, name):
        """
        Looks for a named distribution on the path.

        This function only returns the first result found, as no more than one
        value is expected. If nothing is found, ``None`` is returned.

        :rtype: :class:`InstalledDistribution`, :class:`EggInfoDistribution`
                or ``None``
        """
        result = None
        name = name.lower()
        if not self._cache_enabled:
            for dist in self._yield_distributions():
                if dist.key == name:
                    result = dist
                    break
        else:
            self._generate_cache()

            if name in self._cache.name:
                result = self._cache.name[name][0]
            elif self._include_egg and name in self._cache_egg.name:
                result = self._cache_egg.name[name][0]
        return result

    def provides_distribution(self, name, version=None):
        """
        Iterates over all distributions to find which distributions provide *name*.
        If a *version* is provided, it will be used to filter the results.

        This function only returns the first result found, since no more than
        one values are expected. If the directory is not found, returns ``None``.

        :parameter version: a version specifier that indicates the version
                            required, conforming to the format in ``PEP-345``

        :type name: string
        :type version: string
        """
        matcher = None
        if version is not None:
            try:
                matcher = self._scheme.matcher('%s (%s)' % (name, version))
            except ValueError:
                raise DistlibException('invalid name or version: %r, %r' % (name, version))

        for dist in self.get_distributions():
            # We hit a problem on Travis where enum34 was installed and doesn't
            # have a provides attribute ...
            if not hasattr(dist, 'provides'):
                logger.debug('No "provides": %s', dist)
            else:
                provided = dist.provides

                for p in provided:
                    p_name, p_ver = parse_name_and_version(p)
                    if matcher is None:
                        if p_name == name:
                            yield dist
                            break
                    else:
                        if p_name == name and matcher.match(p_ver):
                            yield dist
                            break

    def get_file_path(self, name, relative_path):
        """
        Return the path to a resource file.
        """
        dist = self.get_distribution(name)
        if dist is None:
            raise LookupError('no distribution named %r found' % name)
        return dist.get_resource_path(relative_path)

    def get_exported_entries(self, category, name=None):
        """
        Return all of the exported entries in a particular category.

        :param category: The category to search for entries.
        :param name: If specified, only entries with that name are returned.
        """
        for dist in self.get_distributions():
            r = dist.exports
            if category in r:
                d = r[category]
                if name is not None:
                    if name in d:
                        yield d[name]
                else:
                    for v in d.values():
                        yield v


class Distribution(object):
    """
    A base class for distributions, whether installed or from indexes.
    Either way, it must have some metadata, so that's all that's needed
    for construction.
    """

    build_time_dependency = False
    """
    Set to True if it's known to be only a build-time dependency (i.e.
    not needed after installation).
    """

    requested = False
    """A boolean that indicates whether the ``REQUESTED`` metadata file is
    present (in other words, whether the package was installed by user
    request or it was installed as a dependency)."""

    def __init__(self, metadata):
        """
        Initialise an instance.
        :param metadata: The instance of :class:`Metadata` describing this
        distribution.
        """
        self.metadata = metadata
        self.name = metadata.name
        self.key = self.name.lower()  # for case-insensitive comparisons
        self.version = metadata.version
        self.locator = None
        self.digest = None
        self.extras = None  # additional features requested
        self.context = None  # environment marker overrides
        self.download_urls = set()
        self.digests = {}

    @property
    def source_url(self):
        """
        The source archive download URL for this distribution.
        """
        return self.metadata.source_url

    download_url = source_url  # Backward compatibility

    @property
    def name_and_version(self):
        """
        A utility property which displays the name and version in parentheses.
        """
        return '%s (%s)' % (self.name, self.version)

    @property
    def provides(self):
        """
        A set of distribution names and versions provided by this distribution.
        :return: A set of "name (version)" strings.
        """
        plist = self.metadata.provides
        s = '%s (%s)' % (self.name, self.version)
        if s not in plist:
            plist.append(s)
        return plist

    def _get_requirements(self, req_attr):
        md = self.metadata
        reqts = getattr(md, req_attr)
        logger.debug('%s: got requirements %r from metadata: %r', self.name, req_attr, reqts)
        return set(md.get_requirements(reqts, extras=self.extras, env=self.context))

    @property
    def run_requires(self):
        return self._get_requirements('run_requires')

    @property
    def meta_requires(self):
        return self._get_requirements('meta_requires')

    @property
    def build_requires(self):
        return self._get_requirements('build_requires')

    @property
    def test_requires(self):
        return self._get_requirements('test_requires')

    @property
    def dev_requires(self):
        return self._get_requirements('dev_requires')

    def matches_requirement(self, req):
        """
        Say if this instance matches (fulfills) a requirement.
        :param req: The requirement to match.
        :rtype req: str
        :return: True if it matches, else False.
        """
        # Requirement may contain extras - parse to lose those
        # from what's passed to the matcher
        r = parse_requirement(req)
        scheme = get_scheme(self.metadata.scheme)
        try:
            matcher = scheme.matcher(r.requirement)
        except UnsupportedVersionError:
            # XXX compat-mode if cannot read the version
            logger.warning('could not read version %r - using name only', req)
            name = req.split()[0]
            matcher = scheme.matcher(name)

        name = matcher.key  # case-insensitive

        result = False
        for p in self.provides:
            p_name, p_ver = parse_name_and_version(p)
            if p_name != name:
                continue
            try:
                result = matcher.match(p_ver)
                break
            except UnsupportedVersionError:
                pass
        return result

    def __repr__(self):
        """
        Return a textual representation of this instance,
        """
        if self.source_url:
            suffix = ' [%s]' % self.source_url
        else:
            suffix = ''
        return '<Distribution %s (%s)%s>' % (self.name, self.version, suffix)

    def __eq__(self, other):
        """
        See if this distribution is the same as another.
        :param other: The distribution to compare with. To be equal to one
                      another. distributions must have the same type, name,
                      version and source_url.
        :return: True if it is the same, else False.
        """
        if type(other) is not type(self):
            result = False
        else:
            result = (self.name == other.name and self.version == other.version and self.source_url == other.source_url)
        return result

    def __hash__(self):
        """
        Compute hash in a way which matches the equality test.
        """
        return hash(self.name) + hash(self.version) + hash(self.source_url)


class BaseInstalledDistribution(Distribution):
    """
    This is the base class for installed distributions (whether PEP 376 or
    legacy).
    """

    hasher = None

    def __init__(self, metadata, path, env=None):
        """
        Initialise an instance.
        :param metadata: An instance of :class:`Metadata` which describes the
                         distribution. This will normally have been initialised
                         from a metadata file in the ``path``.
        :param path:     The path of the ``.dist-info`` or ``.egg-info``
                         directory for the distribution.
        :param env:      This is normally the :class:`DistributionPath`
                         instance where this distribution was found.
        """
        super(BaseInstalledDistribution, self).__init__(metadata)
        self.path = path
        self.dist_path = env

    def get_hash(self, data, hasher=None):
        """
        Get the hash of some data, using a particular hash algorithm, if
        specified.

        :param data: The data to be hashed.
        :type data: bytes
        :param hasher: The name of a hash implementation, supported by hashlib,
                       or ``None``. Examples of valid values are ``'sha1'``,
                       ``'sha224'``, ``'sha384'``, '``sha256'``, ``'md5'`` and
                       ``'sha512'``. If no hasher is specified, the ``hasher``
                       attribute of the :class:`InstalledDistribution` instance
                       is used. If the hasher is determined to be ``None``, MD5
                       is used as the hashing algorithm.
        :returns: The hash of the data. If a hasher was explicitly specified,
                  the returned hash will be prefixed with the specified hasher
                  followed by '='.
        :rtype: str
        """
        if hasher is None:
            hasher = self.hasher
        if hasher is None:
            hasher = hashlib.md5
            prefix = ''
        else:
            hasher = getattr(hashlib, hasher)
            prefix = '%s=' % self.hasher
        digest = hasher(data).digest()
        digest = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
        return '%s%s' % (prefix, digest)


class InstalledDistribution(BaseInstalledDistribution):
    """
    Created with the *path* of the ``.dist-info`` directory provided to the
    constructor. It reads the metadata contained in ``pydist.json`` when it is
    instantiated., or uses a passed in Metadata instance (useful for when
    dry-run mode is being used).
    """

    hasher = 'sha256'

    def __init__(self, path, metadata=None, env=None):
        self.modules = []
        self.finder = finder = resources.finder_for_path(path)
        if finder is None:
            raise ValueError('finder unavailable for %s' % path)
        if env and env._cache_enabled and path in env._cache.path:
            metadata = env._cache.path[path].metadata
        elif metadata is None:
            r = finder.find(METADATA_FILENAME)
            # Temporary - for Wheel 0.23 support
            if r is None:
                r = finder.find(WHEEL_METADATA_FILENAME)
            # Temporary - for legacy support
            if r is None:
                r = finder.find(LEGACY_METADATA_FILENAME)
            if r is None:
                raise ValueError('no %s found in %s' % (METADATA_FILENAME, path))
            with contextlib.closing(r.as_stream()) as stream:
                metadata = Metadata(fileobj=stream, scheme='legacy')

        super(InstalledDistribution, self).__init__(metadata, path, env)

        if env and env._cache_enabled:
            env._cache.add(self)

        r = finder.find('REQUESTED')
        self.requested = r is not None
        p = os.path.join(path, 'top_level.txt')
        if os.path.exists(p):
            with open(p, 'rb') as f:
                data = f.read().decode('utf-8')
            self.modules = data.splitlines()

    def __repr__(self):
        return '<InstalledDistribution %r %s at %r>' % (self.name, self.version, self.path)

    def __str__(self):
        return "%s %s" % (self.name, self.version)

    def _get_records(self):
        """
        Get the list of installed files for the distribution
        :return: A list of tuples of path, hash and size. Note that hash and
                 size might be ``None`` for some entries. The path is exactly
                 as stored in the file (which is as in PEP 376).
        """
        results = []
        r = self.get_distinfo_resource('RECORD')
        with contextlib.closing(r.as_stream()) as stream:
            with CSVReader(stream=stream) as record_reader:
                # Base location is parent dir of .dist-info dir
                # base_location = os.path.dirname(self.path)
                # base_location = os.path.abspath(base_location)
                for row in record_reader:
                    missing = [None for i in range(len(row), 3)]
                    path, checksum, size = row + missing
                    # if not os.path.isabs(path):
                    #     path = path.replace('/', os.sep)
                    #     path = os.path.join(base_location, path)
                    results.append((path, checksum, size))
        return results

    @cached_property
    def exports(self):
        """
        Return the information exported by this distribution.
        :return: A dictionary of exports, mapping an export category to a dict
                 of :class:`ExportEntry` instances describing the individual
                 export entries, and keyed by name.
        """
        result = {}
        r = self.get_distinfo_resource(EXPORTS_FILENAME)
        if r:
            result = self.read_exports()
        return result

    def read_exports(self):
        """
        Read exports data from a file in .ini format.

        :return: A dictionary of exports, mapping an export category to a list
                 of :class:`ExportEntry` instances describing the individual
                 export entries.
        """
        result = {}
        r = self.get_distinfo_resource(EXPORTS_FILENAME)
        if r:
            with contextlib.closing(r.as_stream()) as stream:
                result = read_exports(stream)
        return result

    def write_exports(self, exports):
        """
        Write a dictionary of exports to a file in .ini format.
        :param exports: A dictionary of exports, mapping an export category to
                        a list of :class:`ExportEntry` instances describing the
                        individual export entries.
        """
        rf = self.get_distinfo_file(EXPORTS_FILENAME)
        with open(rf, 'w') as f:
            write_exports(exports, f)

    def get_resource_path(self, relative_path):
        """
        NOTE: This API may change in the future.

        Return the absolute path to a resource file with the given relative
        path.

        :param relative_path: The path, relative to .dist-info, of the resource
                              of interest.
        :return: The absolute path where the resource is to be found.
        """
        r = self.get_distinfo_resource('RESOURCES')
        with contextlib.closing(r.as_stream()) as stream:
            with CSVReader(stream=stream) as resources_reader:
                for relative, destination in resources_reader:
                    if relative == relative_path:
                        return destination
        raise KeyError('no resource file with relative path %r '
                       'is installed' % relative_path)

    def list_installed_files(self):
        """
        Iterates over the ``RECORD`` entries and returns a tuple
        ``(path, hash, size)`` for each line.

        :returns: iterator of (path, hash, size)
        """
        for result in self._get_records():
            yield result

    def write_installed_files(self, paths, prefix, dry_run=False):
        """
        Writes the ``RECORD`` file, using the ``paths`` iterable passed in. Any
        existing ``RECORD`` file is silently overwritten.

        prefix is used to determine when to write absolute paths.
        """
        prefix = os.path.join(prefix, '')
        base = os.path.dirname(self.path)
        base_under_prefix = base.startswith(prefix)
        base = os.path.join(base, '')
        record_path = self.get_distinfo_file('RECORD')
        logger.info('creating %s', record_path)
        if dry_run:
            return None
        with CSVWriter(record_path) as writer:
            for path in paths:
                if os.path.isdir(path) or path.endswith(('.pyc', '.pyo')):
                    # do not put size and hash, as in PEP-376
                    hash_value = size = ''
                else:
                    size = '%d' % os.path.getsize(path)
                    with open(path, 'rb') as fp:
                        hash_value = self.get_hash(fp.read())
                if path.startswith(base) or (base_under_prefix and path.startswith(prefix)):
                    path = os.path.relpath(path, base)
                writer.writerow((path, hash_value, size))

            # add the RECORD file itself
            if record_path.startswith(base):
                record_path = os.path.relpath(record_path, base)
            writer.writerow((record_path, '', ''))
        return record_path

    def check_installed_files(self):
        """
        Checks that the hashes and sizes of the files in ``RECORD`` are
        matched by the files themselves. Returns a (possibly empty) list of
        mismatches. Each entry in the mismatch list will be a tuple consisting
        of the path, 'exists', 'size' or 'hash' according to what didn't match
        (existence is checked first, then size, then hash), the expected
        value and the actual value.
        """
        mismatches = []
        base = os.path.dirname(self.path)
        record_path = self.get_distinfo_file('RECORD')
        for path, hash_value, size in self.list_installed_files():
            if not os.path.isabs(path):
                path = os.path.join(base, path)
            if path == record_path:
                continue
            if not os.path.exists(path):
                mismatches.append((path, 'exists', True, False))
            elif os.path.isfile(path):
                actual_size = str(os.path.getsize(path))
                if size and actual_size != size:
                    mismatches.append((path, 'size', size, actual_size))
                elif hash_value:
                    if '=' in hash_value:
                        hasher = hash_value.split('=', 1)[0]
                    else:
                        hasher = None

                    with open(path, 'rb') as f:
                        actual_hash = self.get_hash(f.read(), hasher)
                        if actual_hash != hash_value:
                            mismatches.append((path, 'hash', hash_value, actual_hash))
        return mismatches

    @cached_property
    def shared_locations(self):
        """
        A dictionary of shared locations whose keys are in the set 'prefix',
        'purelib', 'platlib', 'scripts', 'headers', 'data' and 'namespace'.
        The corresponding value is the absolute path of that category for
        this distribution, and takes into account any paths selected by the
        user at installation time (e.g. via command-line arguments). In the
        case of the 'namespace' key, this would be a list of absolute paths
        for the roots of namespace packages in this distribution.

        The first time this property is accessed, the relevant information is
        read from the SHARED file in the .dist-info directory.
        """
        result = {}
        shared_path = os.path.join(self.path, 'SHARED')
        if os.path.isfile(shared_path):
            with codecs.open(shared_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
            for line in lines:
                key, value = line.split('=', 1)
                if key == 'namespace':
                    result.setdefault(key, []).append(value)
                else:
                    result[key] = value
        return result

    def write_shared_locations(self, paths, dry_run=False):
        """
        Write shared location information to the SHARED file in .dist-info.
        :param paths: A dictionary as described in the documentation for
        :meth:`shared_locations`.
        :param dry_run: If True, the action is logged but no file is actually
                        written.
        :return: The path of the file written to.
        """
        shared_path = os.path.join(self.path, 'SHARED')
        logger.info('creating %s', shared_path)
        if dry_run:
            return None
        lines = []
        for key in ('prefix', 'lib', 'headers', 'scripts', 'data'):
            path = paths[key]
            if os.path.isdir(paths[key]):
                lines.append('%s=%s' % (key, path))
        for ns in paths.get('namespace', ()):
            lines.append('namespace=%s' % ns)

        with codecs.open(shared_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return shared_path

    def get_distinfo_resource(self, path):
        if path not in DIST_FILES:
            raise DistlibException('invalid path for a dist-info file: '
                                   '%r at %r' % (path, self.path))
        finder = resources.finder_for_path(self.path)
        if finder is None:
            raise DistlibException('Unable to get a finder for %s' % self.path)
        return finder.find(path)

    def get_distinfo_file(self, path):
        """
        Returns a path located under the ``.dist-info`` directory. Returns a
        string representing the path.

        :parameter path: a ``'/'``-separated path relative to the
                         ``.dist-info`` directory or an absolute path;
                         If *path* is an absolute path and doesn't start
                         with the ``.dist-info`` directory path,
                         a :class:`DistlibException` is raised
        :type path: str
        :rtype: str
        """
        # Check if it is an absolute path  # XXX use relpath, add tests
        if path.find(os.sep) >= 0:
            # it's an absolute path?
            distinfo_dirname, path = path.split(os.sep)[-2:]
            if distinfo_dirname != self.path.split(os.sep)[-1]:
                raise DistlibException('dist-info file %r does not belong to the %r %s '
                                       'distribution' % (path, self.name, self.version))

        # The file must be relative
        if path not in DIST_FILES:
            raise DistlibException('invalid path for a dist-info file: '
                                   '%r at %r' % (path, self.path))

        return os.path.join(self.path, path)

    def list_distinfo_files(self):
        """
        Iterates over the ``RECORD`` entries and returns paths for each line if
        the path is pointing to a file located in the ``.dist-info`` directory
        or one of its subdirectories.

        :returns: iterator of paths
        """
        base = os.path.dirname(self.path)
        for path, checksum, size in self._get_records():
            # XXX add separator or use real relpath algo
            if not os.path.isabs(path):
                path = os.path.join(base, path)
            if path.startswith(self.path):
                yield path

    def __eq__(self, other):
        return (isinstance(other, InstalledDistribution) and self.path == other.path)

    # See http://docs.python.org/reference/datamodel#object.__hash__
    __hash__ = object.__hash__


class EggInfoDistribution(BaseInstalledDistribution):
    """Created with the *path* of the ``.egg-info`` directory or file provided
    to the constructor. It reads the metadata contained in the file itself, or
    if the given path happens to be a directory, the metadata is read from the
    file ``PKG-INFO`` under that directory."""

    requested = True  # as we have no way of knowing, assume it was
    shared_locations = {}

    def __init__(self, path, env=None):

        def set_name_and_version(s, n, v):
            s.name = n
            s.key = n.lower()  # for case-insensitive comparisons
            s.version = v

        self.path = path
        self.dist_path = env
        if env and env._cache_enabled and path in env._cache_egg.path:
            metadata = env._cache_egg.path[path].metadata
            set_name_and_version(self, metadata.name, metadata.version)
        else:
            metadata = self._get_metadata(path)

            # Need to be set before caching
            set_name_and_version(self, metadata.name, metadata.version)

            if env and env._cache_enabled:
                env._cache_egg.add(self)
        super(EggInfoDistribution, self).__init__(metadata, path, env)

    def _get_metadata(self, path):
        requires = None

        def parse_requires_data(data):
            """Create a list of dependencies from a requires.txt file.

            *data*: the contents of a setuptools-produced requires.txt file.
            """
            reqs = []
            lines = data.splitlines()
            for line in lines:
                line = line.strip()
                # sectioned files have bare newlines (separating sections)
                if not line:  # pragma: no cover
                    continue
                if line.startswith('['):  # pragma: no cover
                    logger.warning('Unexpected line: quitting requirement scan: %r', line)
                    break
                r = parse_requirement(line)
                if not r:  # pragma: no cover
                    logger.warning('Not recognised as a requirement: %r', line)
                    continue
                if r.extras:  # pragma: no cover
                    logger.warning('extra requirements in requires.txt are '
                                   'not supported')
                if not r.constraints:
                    reqs.append(r.name)
                else:
                    cons = ', '.join('%s%s' % c for c in r.constraints)
                    reqs.append('%s (%s)' % (r.name, cons))
            return reqs

        def parse_requires_path(req_path):
            """Create a list of dependencies from a requires.txt file.

            *req_path*: the path to a setuptools-produced requires.txt file.
            """

            reqs = []
            try:
                with codecs.open(req_path, 'r', 'utf-8') as fp:
                    reqs = parse_requires_data(fp.read())
            except IOError:
                pass
            return reqs

        tl_path = tl_data = None
        if path.endswith('.egg'):
            if os.path.isdir(path):
                p = os.path.join(path, 'EGG-INFO')
                meta_path = os.path.join(p, 'PKG-INFO')
                metadata = Metadata(path=meta_path, scheme='legacy')
                req_path = os.path.join(p, 'requires.txt')
                tl_path = os.path.join(p, 'top_level.txt')
                requires = parse_requires_path(req_path)
            else:
                # FIXME handle the case where zipfile is not available
                zipf = zipimport.zipimporter(path)
                fileobj = StringIO(zipf.get_data('EGG-INFO/PKG-INFO').decode('utf8'))
                metadata = Metadata(fileobj=fileobj, scheme='legacy')
                try:
                    data = zipf.get_data('EGG-INFO/requires.txt')
                    tl_data = zipf.get_data('EGG-INFO/top_level.txt').decode('utf-8')
                    requires = parse_requires_data(data.decode('utf-8'))
                except IOError:
                    requires = None
        elif path.endswith('.egg-info'):
            if os.path.isdir(path):
                req_path = os.path.join(path, 'requires.txt')
                requires = parse_requires_path(req_path)
                path = os.path.join(path, 'PKG-INFO')
                tl_path = os.path.join(path, 'top_level.txt')
            metadata = Metadata(path=path, scheme='legacy')
        else:
            raise DistlibException('path must end with .egg-info or .egg, '
                                   'got %r' % path)

        if requires:
            metadata.add_requirements(requires)
        # look for top-level modules in top_level.txt, if present
        if tl_data is None:
            if tl_path is not None and os.path.exists(tl_path):
                with open(tl_path, 'rb') as f:
                    tl_data = f.read().decode('utf-8')
        if not tl_data:
            tl_data = []
        else:
            tl_data = tl_data.splitlines()
        self.modules = tl_data
        return metadata

    def __repr__(self):
        return '<EggInfoDistribution %r %s at %r>' % (self.name, self.version, self.path)

    def __str__(self):
        return "%s %s" % (self.name, self.version)

    def check_installed_files(self):
        """
        Checks that the hashes and sizes of the files in ``RECORD`` are
        matched by the files themselves. Returns a (possibly empty) list of
        mismatches. Each entry in the mismatch list will be a tuple consisting
        of the path, 'exists', 'size' or 'hash' according to what didn't match
        (existence is checked first, then size, then hash), the expected
        value and the actual value.
        """
        mismatches = []
        record_path = os.path.join(self.path, 'installed-files.txt')
        if os.path.exists(record_path):
            for path, _, _ in self.list_installed_files():
                if path == record_path:
                    continue
                if not os.path.exists(path):
                    mismatches.append((path, 'exists', True, False))
        return mismatches

    def list_installed_files(self):
        """
        Iterates over the ``installed-files.txt`` entries and returns a tuple
        ``(path, hash, size)`` for each line.

        :returns: a list of (path, hash, size)
        """

        def _md5(path):
            f = open(path, 'rb')
            try:
                content = f.read()
            finally:
                f.close()
            return hashlib.md5(content).hexdigest()

        def _size(path):
            return os.stat(path).st_size

        record_path = os.path.join(self.path, 'installed-files.txt')
        result = []
        if os.path.exists(record_path):
            with codecs.open(record_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    p = os.path.normpath(os.path.join(self.path, line))
                    # "./" is present as a marker between installed files
                    # and installation metadata files
                    if not os.path.exists(p):
                        logger.warning('Non-existent file: %s', p)
                        if p.endswith(('.pyc', '.pyo')):
                            continue
                        # otherwise fall through and fail
                    if not os.path.isdir(p):
                        result.append((p, _md5(p), _size(p)))
            result.append((record_path, None, None))
        return result

    def list_distinfo_files(self, absolute=False):
        """
        Iterates over the ``installed-files.txt`` entries and returns paths for
        each line if the path is pointing to a file located in the
        ``.egg-info`` directory or one of its subdirectories.

        :parameter absolute: If *absolute* is ``True``, each returned path is
                          transformed into a local absolute path. Otherwise the
                          raw value from ``installed-files.txt`` is returned.
        :type absolute: boolean
        :returns: iterator of paths
        """
        record_path = os.path.join(self.path, 'installed-files.txt')
        if os.path.exists(record_path):
            skip = True
            with codecs.open(record_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == './':
                        skip = False
                        continue
                    if not skip:
                        p = os.path.normpath(os.path.join(self.path, line))
                        if p.startswith(self.path):
                            if absolute:
                                yield p
                            else:
                                yield line

    def __eq__(self, other):
        return (isinstance(other, EggInfoDistribution) and self.path == other.path)

    # See http://docs.python.org/reference/datamodel#object.__hash__
    __hash__ = object.__hash__


new_dist_class = InstalledDistribution
old_dist_class = EggInfoDistribution


class DependencyGraph(object):
    """
    Represents a dependency graph between distributions.

    The dependency relationships are stored in an ``adjacency_list`` that maps
    distributions to a list of ``(other, label)`` tuples where  ``other``
    is a distribution and the edge is labeled with ``label`` (i.e. the version
    specifier, if such was provided). Also, for more efficient traversal, for
    every distribution ``x``, a list of predecessors is kept in
    ``reverse_list[x]``. An edge from distribution ``a`` to
    distribution ``b`` means that ``a`` depends on ``b``. If any missing
    dependencies are found, they are stored in ``missing``, which is a
    dictionary that maps distributions to a list of requirements that were not
    provided by any other distributions.
    """

    def __init__(self):
        self.adjacency_list = {}
        self.reverse_list = {}
        self.missing = {}

    def add_distribution(self, distribution):
        """Add the *distribution* to the graph.

        :type distribution: :class:`distutils2.database.InstalledDistribution`
                            or :class:`distutils2.database.EggInfoDistribution`
        """
        self.adjacency_list[distribution] = []
        self.reverse_list[distribution] = []
        # self.missing[distribution] = []

    def add_edge(self, x, y, label=None):
        """Add an edge from distribution *x* to distribution *y* with the given
        *label*.

        :type x: :class:`distutils2.database.InstalledDistribution` or
                 :class:`distutils2.database.EggInfoDistribution`
        :type y: :class:`distutils2.database.InstalledDistribution` or
                 :class:`distutils2.database.EggInfoDistribution`
        :type label: ``str`` or ``None``
        """
        self.adjacency_list[x].append((y, label))
        # multiple edges are allowed, so be careful
        if x not in self.reverse_list[y]:
            self.reverse_list[y].append(x)

    def add_missing(self, distribution, requirement):
        """
        Add a missing *requirement* for the given *distribution*.

        :type distribution: :class:`distutils2.database.InstalledDistribution`
                            or :class:`distutils2.database.EggInfoDistribution`
        :type requirement: ``str``
        """
        logger.debug('%s missing %r', distribution, requirement)
        self.missing.setdefault(distribution, []).append(requirement)

    def _repr_dist(self, dist):
        return '%s %s' % (dist.name, dist.version)

    def repr_node(self, dist, level=1):
        """Prints only a subgraph"""
        output = [self._repr_dist(dist)]
        for other, label in self.adjacency_list[dist]:
            dist = self._repr_dist(other)
            if label is not None:
                dist = '%s [%s]' % (dist, label)
            output.append('    ' * level + str(dist))
            suboutput = self.repr_node(other, level + 1)
            subs = suboutput.split('\n')
            output.extend(subs[1:])
        return '\n'.join(output)

    def to_dot(self, f, skip_disconnected=True):
        """Writes a DOT output for the graph to the provided file *f*.

        If *skip_disconnected* is set to ``True``, then all distributions
        that are not dependent on any other distribution are skipped.

        :type f: has to support ``file``-like operations
        :type skip_disconnected: ``bool``
        """
        disconnected = []

        f.write("digraph dependencies {\n")
        for dist, adjs in self.adjacency_list.items():
            if len(adjs) == 0 and not skip_disconnected:
                disconnected.append(dist)
            for other, label in adjs:
                if label is not None:
                    f.write('"%s" -> "%s" [label="%s"]\n' % (dist.name, other.name, label))
                else:
                    f.write('"%s" -> "%s"\n' % (dist.name, other.name))
        if not skip_disconnected and len(disconnected) > 0:
            f.write('subgraph disconnected {\n')
            f.write('label = "Disconnected"\n')
            f.write('bgcolor = red\n')

            for dist in disconnected:
                f.write('"%s"' % dist.name)
                f.write('\n')
            f.write('}\n')
        f.write('}\n')

    def topological_sort(self):
        """
        Perform a topological sort of the graph.
        :return: A tuple, the first element of which is a topologically sorted
                 list of distributions, and the second element of which is a
                 list of distributions that cannot be sorted because they have
                 circular dependencies and so form a cycle.
        """
        result = []
        # Make a shallow copy of the adjacency list
        alist = {}
        for k, v in self.adjacency_list.items():
            alist[k] = v[:]
        while True:
            # See what we can remove in this run
            to_remove = []
            for k, v in list(alist.items())[:]:
                if not v:
                    to_remove.append(k)
                    del alist[k]
            if not to_remove:
                # What's left in alist (if anything) is a cycle.
                break
            # Remove from the adjacency list of others
            for k, v in alist.items():
                alist[k] = [(d, r) for d, r in v if d not in to_remove]
            logger.debug('Moving to result: %s', ['%s (%s)' % (d.name, d.version) for d in to_remove])
            result.extend(to_remove)
        return result, list(alist.keys())

    def __repr__(self):
        """Representation of the graph"""
        output = []
        for dist, adjs in self.adjacency_list.items():
            output.append(self.repr_node(dist))
        return '\n'.join(output)


def make_graph(dists, scheme='default'):
    """Makes a dependency graph from the given distributions.

    :parameter dists: a list of distributions
    :type dists: list of :class:`distutils2.database.InstalledDistribution` and
                 :class:`distutils2.database.EggInfoDistribution` instances
    :rtype: a :class:`DependencyGraph` instance
    """
    scheme = get_scheme(scheme)
    graph = DependencyGraph()
    provided = {}  # maps names to lists of (version, dist) tuples

    # first, build the graph and find out what's provided
    for dist in dists:
        graph.add_distribution(dist)

        for p in dist.provides:
            name, version = parse_name_and_version(p)
            logger.debug('Add to provided: %s, %s, %s', name, version, dist)
            provided.setdefault(name, []).append((version, dist))

    # now make the edges
    for dist in dists:
        requires = (dist.run_requires | dist.meta_requires | dist.build_requires | dist.dev_requires)
        for req in requires:
            try:
                matcher = scheme.matcher(req)
            except UnsupportedVersionError:
                # XXX compat-mode if cannot read the version
                logger.warning('could not read version %r - using name only', req)
                name = req.split()[0]
                matcher = scheme.matcher(name)

            name = matcher.key  # case-insensitive

            matched = False
            if name in provided:
                for version, provider in provided[name]:
                    try:
                        match = matcher.match(version)
                    except UnsupportedVersionError:
                        match = False

                    if match:
                        graph.add_edge(dist, provider, req)
                        matched = True
                        break
            if not matched:
                graph.add_missing(dist, req)
    return graph


def get_dependent_dists(dists, dist):
    """Recursively generate a list of distributions from *dists* that are
    dependent on *dist*.

    :param dists: a list of distributions
    :param dist: a distribution, member of *dists* for which we are interested
    """
    if dist not in dists:
        raise DistlibException('given distribution %r is not a member '
                               'of the list' % dist.name)
    graph = make_graph(dists)

    dep = [dist]  # dependent distributions
    todo = graph.reverse_list[dist]  # list of nodes we should inspect

    while todo:
        d = todo.pop()
        dep.append(d)
        for succ in graph.reverse_list[d]:
            if succ not in dep:
                todo.append(succ)

    dep.pop(0)  # remove dist from dep, was there to prevent infinite loops
    return dep


def get_required_dists(dists, dist):
    """Recursively generate a list of distributions from *dists* that are
    required by *dist*.

    :param dists: a list of distributions
    :param dist: a distribution, member of *dists* for which we are interested
                 in finding the dependencies.
    """
    if dist not in dists:
        raise DistlibException('given distribution %r is not a member '
                               'of the list' % dist.name)
    graph = make_graph(dists)

    req = set()  # required distributions
    todo = graph.adjacency_list[dist]  # list of nodes we should inspect
    seen = set(t[0] for t in todo)  # already added to todo

    while todo:
        d = todo.pop()[0]
        req.add(d)
        pred_list = graph.adjacency_list[d]
        for pred in pred_list:
            d = pred[0]
            if d not in req and d not in seen:
                seen.add(d)
                todo.append(pred)
    return req


def make_dist(name, version, **kwargs):
    """
    A convenience method for making a dist given just a name and version.
    """
    summary = kwargs.pop('summary', 'Placeholder for summary')
    md = Metadata(**kwargs)
    md.name = name
    md.version = version
    md.summary = summary or 'Placeholder for summary'
    return Distribution(md)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_socket.py ===
from __future__ import annotations

import os
import select
import socket as _stdlib_socket
import sys
from operator import index
from socket import AddressFamily, SocketKind
from typing import (
    TYPE_CHECKING,
    Any,
    SupportsIndex,
    TypeVar,
    Union,
    overload,
)

import idna as _idna

import trio
from trio._util import wraps as _wraps

from . import _core

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from types import TracebackType

    from typing_extensions import Buffer, Concatenate, ParamSpec, Self, TypeAlias

    from ._abc import HostnameResolver, SocketFactory

    P = ParamSpec("P")


T = TypeVar("T")

# _stdlib_socket.socket supports 13 different socket families, see
# https://docs.python.org/3/library/socket.html#socket-families
# and the return type of several methods in SocketType will depend on those. Typeshed
# has ended up typing those return types as `Any` in most cases, but for users that
# know which family/families they're working in we could make SocketType a generic type,
# where you specify the return values you expect from those methods depending on the
# protocol the socket will be handling.
# But without the ability to default the value to `Any` it will be overly cumbersome for
# most users, so currently we just specify it as `Any`. Otherwise we would write:
# `AddressFormat = TypeVar("AddressFormat")`
# but instead we simply do:
AddressFormat: TypeAlias = Any  # type: ignore[explicit-any]


# Usage:
#
#   async with _try_sync():
#       return sync_call_that_might_fail_with_exception()
#   # we only get here if the sync call in fact did fail with a
#   # BlockingIOError
#   return await do_it_properly_with_a_check_point()
#
class _try_sync:
    def __init__(
        self,
        blocking_exc_override: Callable[[BaseException], bool] | None = None,
    ) -> None:
        self._blocking_exc_override = blocking_exc_override

    def _is_blocking_io_error(self, exc: BaseException) -> bool:
        if self._blocking_exc_override is None:
            return isinstance(exc, BlockingIOError)
        else:
            return self._blocking_exc_override(exc)

    async def __aenter__(self) -> None:
        await trio.lowlevel.checkpoint_if_cancelled()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc_value is not None and self._is_blocking_io_error(exc_value):
            # Discard the exception and fall through to the code below the
            # block
            return True
        else:
            await trio.lowlevel.cancel_shielded_checkpoint()
            # Let the return or exception propagate
            return False


################################################################
# Overrides
################################################################

_resolver: _core.RunVar[HostnameResolver | None] = _core.RunVar("hostname_resolver")
_socket_factory: _core.RunVar[SocketFactory | None] = _core.RunVar("socket_factory")


def set_custom_hostname_resolver(
    hostname_resolver: HostnameResolver | None,
) -> HostnameResolver | None:
    """Set a custom hostname resolver.

    By default, Trio's :func:`getaddrinfo` and :func:`getnameinfo` functions
    use the standard system resolver functions. This function allows you to
    customize that behavior. The main intended use case is for testing, but it
    might also be useful for using third-party resolvers like `c-ares
    <https://c-ares.haxx.se/>`__ (though be warned that these rarely make
    perfect drop-in replacements for the system resolver). See
    :class:`trio.abc.HostnameResolver` for more details.

    Setting a custom hostname resolver affects all future calls to
    :func:`getaddrinfo` and :func:`getnameinfo` within the enclosing call to
    :func:`trio.run`. All other hostname resolution in Trio is implemented in
    terms of these functions.

    Generally you should call this function just once, right at the beginning
    of your program.

    Args:
      hostname_resolver (trio.abc.HostnameResolver or None): The new custom
          hostname resolver, or None to restore the default behavior.

    Returns:
      The previous hostname resolver (which may be None).

    """
    old = _resolver.get(None)
    _resolver.set(hostname_resolver)
    return old


def set_custom_socket_factory(
    socket_factory: SocketFactory | None,
) -> SocketFactory | None:
    """Set a custom socket object factory.

    This function allows you to replace Trio's normal socket class with a
    custom class. This is very useful for testing, and probably a bad idea in
    any other circumstance. See :class:`trio.abc.HostnameResolver` for more
    details.

    Setting a custom socket factory affects all future calls to :func:`socket`
    within the enclosing call to :func:`trio.run`.

    Generally you should call this function just once, right at the beginning
    of your program.

    Args:
      socket_factory (trio.abc.SocketFactory or None): The new custom
          socket factory, or None to restore the default behavior.

    Returns:
      The previous socket factory (which may be None).

    """
    old = _socket_factory.get(None)
    _socket_factory.set(socket_factory)
    return old


################################################################
# getaddrinfo and friends
################################################################

# AI_NUMERICSERV may be missing on some older platforms, so use it when available.
# See: https://github.com/python-trio/trio/issues/3133
_NUMERIC_ONLY = _stdlib_socket.AI_NUMERICHOST
_NUMERIC_ONLY |= getattr(_stdlib_socket, "AI_NUMERICSERV", 0)


# It would be possible to @overload the return value depending on Literal[AddressFamily.INET/6], but should probably be added in typeshed first
async def getaddrinfo(
    host: bytes | str | None,
    port: bytes | str | int | None,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
) -> list[
    tuple[
        AddressFamily,
        SocketKind,
        int,
        str,
        tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes],
    ]
]:
    """Look up a numeric address given a name.

    Arguments and return values are identical to :func:`socket.getaddrinfo`,
    except that this version is async.

    Also, :func:`trio.socket.getaddrinfo` correctly uses IDNA 2008 to process
    non-ASCII domain names. (:func:`socket.getaddrinfo` uses IDNA 2003, which
    can give the wrong result in some cases and cause you to connect to a
    different host than the one you intended; see `bpo-17305
    <https://bugs.python.org/issue17305>`__.)

    This function's behavior can be customized using
    :func:`set_custom_hostname_resolver`.

    """

    # If host and port are numeric, then getaddrinfo doesn't block and we can
    # skip the whole thread thing, which seems worthwhile. So we try first
    # with the _NUMERIC_ONLY flags set, and then only spawn a thread if that
    # fails with EAI_NONAME:
    def numeric_only_failure(exc: BaseException) -> bool:
        return (
            isinstance(exc, _stdlib_socket.gaierror)
            and exc.errno == _stdlib_socket.EAI_NONAME
        )

    async with _try_sync(numeric_only_failure):
        return _stdlib_socket.getaddrinfo(
            host,
            port,
            family,
            type,
            proto,
            flags | _NUMERIC_ONLY,
        )
    # That failed; it's a real hostname. We better use a thread.
    #
    # Also, it might be a unicode hostname, in which case we want to do our
    # own encoding using the idna module, rather than letting Python do
    # it. (Python will use the old IDNA 2003 standard, and possibly get the
    # wrong answer - see bpo-17305). However, the idna module is picky, and
    # will refuse to process some valid hostname strings, like "::1". So if
    # it's already ascii, we pass it through; otherwise, we encode it to.
    if isinstance(host, str):
        try:
            host = host.encode("ascii")
        except UnicodeEncodeError:
            # UTS-46 defines various normalizations; in particular, by default
            # idna.encode will error out if the hostname has Capital Letters
            # in it; with uts46=True it will lowercase them instead.
            host = _idna.encode(host, uts46=True)
    hr = _resolver.get(None)
    if hr is not None:
        return await hr.getaddrinfo(host, port, family, type, proto, flags)
    else:
        return await trio.to_thread.run_sync(
            _stdlib_socket.getaddrinfo,
            host,
            port,
            family,
            type,
            proto,
            flags,
            abandon_on_cancel=True,
        )


async def getnameinfo(
    sockaddr: tuple[str, int] | tuple[str, int, int, int],
    flags: int,
) -> tuple[str, str]:
    """Look up a name given a numeric address.

    Arguments and return values are identical to :func:`socket.getnameinfo`,
    except that this version is async.

    This function's behavior can be customized using
    :func:`set_custom_hostname_resolver`.

    """
    hr = _resolver.get(None)
    if hr is not None:
        return await hr.getnameinfo(sockaddr, flags)
    else:
        return await trio.to_thread.run_sync(
            _stdlib_socket.getnameinfo,
            sockaddr,
            flags,
            abandon_on_cancel=True,
        )


async def getprotobyname(name: str) -> int:
    """Look up a protocol number by name. (Rarely used.)

    Like :func:`socket.getprotobyname`, but async.

    """
    return await trio.to_thread.run_sync(
        _stdlib_socket.getprotobyname,
        name,
        abandon_on_cancel=True,
    )


# obsolete gethostbyname etc. intentionally omitted
# likewise for create_connection (use open_tcp_stream instead)

################################################################
# Socket "constructors"
################################################################


def from_stdlib_socket(sock: _stdlib_socket.socket) -> SocketType:
    """Convert a standard library :class:`socket.socket` object into a Trio
    socket object.

    """
    return _SocketType(sock)


@_wraps(_stdlib_socket.fromfd, assigned=(), updated=())
def fromfd(
    fd: SupportsIndex,
    family: AddressFamily | int = _stdlib_socket.AF_INET,
    type: SocketKind | int = _stdlib_socket.SOCK_STREAM,
    proto: int = 0,
) -> SocketType:
    """Like :func:`socket.fromfd`, but returns a Trio socket object."""
    family, type_, proto = _sniff_sockopts_for_fileno(family, type, proto, index(fd))
    return from_stdlib_socket(_stdlib_socket.fromfd(fd, family, type_, proto))


if sys.platform == "win32" or (
    not TYPE_CHECKING and hasattr(_stdlib_socket, "fromshare")
):

    @_wraps(_stdlib_socket.fromshare, assigned=(), updated=())
    def fromshare(info: bytes) -> SocketType:
        """Like :func:`socket.fromshare`, but returns a Trio socket object."""
        return from_stdlib_socket(_stdlib_socket.fromshare(info))


if sys.platform == "win32":
    FamilyT: TypeAlias = int
    TypeT: TypeAlias = int
    FamilyDefault = _stdlib_socket.AF_INET
else:
    FamilyDefault: None = None
    FamilyT: TypeAlias = Union[int, AddressFamily, None]
    TypeT: TypeAlias = Union[_stdlib_socket.socket, int]


@_wraps(_stdlib_socket.socketpair, assigned=(), updated=())
def socketpair(
    family: FamilyT = FamilyDefault,
    type: TypeT = SocketKind.SOCK_STREAM,
    proto: int = 0,
) -> tuple[SocketType, SocketType]:
    """Like :func:`socket.socketpair`, but returns a pair of Trio socket
    objects.

    """
    left, right = _stdlib_socket.socketpair(family, type, proto)
    return (from_stdlib_socket(left), from_stdlib_socket(right))


@_wraps(_stdlib_socket.socket, assigned=(), updated=())
def socket(
    family: AddressFamily | int = _stdlib_socket.AF_INET,
    type: SocketKind | int = _stdlib_socket.SOCK_STREAM,
    proto: int = 0,
    fileno: int | None = None,
) -> SocketType:
    """Create a new Trio socket, like :class:`socket.socket`.

    This function's behavior can be customized using
    :func:`set_custom_socket_factory`.

    """
    if fileno is None:
        sf = _socket_factory.get(None)
        if sf is not None:
            return sf.socket(family, type, proto)
    else:
        family, type, proto = _sniff_sockopts_for_fileno(  # noqa: A001
            family,
            type,
            proto,
            fileno,
        )
    stdlib_socket = _stdlib_socket.socket(family, type, proto, fileno)
    return from_stdlib_socket(stdlib_socket)


def _sniff_sockopts_for_fileno(
    family: AddressFamily | int,
    type_: SocketKind | int,
    proto: int,
    fileno: int | None,
) -> tuple[AddressFamily | int, SocketKind | int, int]:
    """Correct SOCKOPTS for given fileno, falling back to provided values."""
    # Wrap the raw fileno into a Python socket object
    # This object might have the wrong metadata, but it lets us easily call getsockopt
    # and then we'll throw it away and construct a new one with the correct metadata.
    if sys.platform != "linux":
        return family, type_, proto
    from socket import (  # type: ignore[attr-defined,unused-ignore]
        SO_DOMAIN,
        SO_PROTOCOL,
        SO_TYPE,
        SOL_SOCKET,
    )

    sockobj = _stdlib_socket.socket(family, type_, proto, fileno=fileno)
    try:
        family = sockobj.getsockopt(SOL_SOCKET, SO_DOMAIN)
        proto = sockobj.getsockopt(SOL_SOCKET, SO_PROTOCOL)
        type_ = sockobj.getsockopt(SOL_SOCKET, SO_TYPE)
    finally:
        # Unwrap it again, so that sockobj.__del__ doesn't try to close our socket
        sockobj.detach()
    return family, type_, proto


################################################################
# SocketType
################################################################

# sock.type gets weird stuff set in it, in particular on Linux:
#
#   https://bugs.python.org/issue21327
#
# But on other platforms (e.g. Windows) SOCK_NONBLOCK and SOCK_CLOEXEC aren't
# even defined. To recover the actual socket type (e.g. SOCK_STREAM) from a
# socket.type attribute, mask with this:
_SOCK_TYPE_MASK = ~(
    getattr(_stdlib_socket, "SOCK_NONBLOCK", 0)
    | getattr(_stdlib_socket, "SOCK_CLOEXEC", 0)
)


def _make_simple_sock_method_wrapper(
    fn: Callable[Concatenate[_stdlib_socket.socket, P], T],
    wait_fn: Callable[[_stdlib_socket.socket], Awaitable[None]],
    maybe_avail: bool = False,
) -> Callable[Concatenate[_SocketType, P], Awaitable[T]]:
    @_wraps(fn, assigned=("__name__",), updated=())
    async def wrapper(self: _SocketType, *args: P.args, **kwargs: P.kwargs) -> T:
        return await self._nonblocking_helper(wait_fn, fn, *args, **kwargs)

    wrapper.__doc__ = f"""Like :meth:`socket.socket.{fn.__name__}`, but async.

            """
    if maybe_avail:
        wrapper.__doc__ += (
            f"Only available on platforms where :meth:`socket.socket.{fn.__name__}` is "
            "available."
        )
    return wrapper


# Helpers to work with the (hostname, port) language that Python uses for socket
# addresses everywhere. Split out into a standalone function so it can be reused by
# FakeNet.


# Take an address in Python's representation, and returns a new address in
# the same representation, but with names resolved to numbers,
# etc.
#
# local=True means that the address is being used with bind() or similar
# local=False means that the address is being used with connect() or sendto() or
# similar.
#


# Using a TypeVar to indicate we return the same type of address appears to give errors
# when passed a union of address types.
# @overload likely works, but is extremely verbose.
# NOTE: this function does not always checkpoint
async def _resolve_address_nocp(
    type_: int,
    family: AddressFamily,
    proto: int,
    *,
    ipv6_v6only: bool | int,
    address: AddressFormat,
    local: bool,
) -> AddressFormat:
    # Do some pre-checking (or exit early for non-IP sockets)
    if family == _stdlib_socket.AF_INET:
        if not isinstance(address, tuple) or not len(address) == 2:
            raise ValueError("address should be a (host, port) tuple")
    elif family == _stdlib_socket.AF_INET6:
        if not isinstance(address, tuple) or not 2 <= len(address) <= 4:
            raise ValueError(
                "address should be a (host, port, [flowinfo, [scopeid]]) tuple",
            )
    elif hasattr(_stdlib_socket, "AF_UNIX") and family == _stdlib_socket.AF_UNIX:
        # unwrap path-likes
        assert isinstance(address, (str, bytes, os.PathLike))
        return os.fspath(address)
    else:
        return address

    # -- From here on we know we have IPv4 or IPV6 --
    host: str | None
    host, port, *_ = address
    # Fast path for the simple case: already-resolved IP address,
    # already-resolved port. This is particularly important for UDP, since
    # every sendto call goes through here.
    if isinstance(port, int) and host is not None:
        try:
            _stdlib_socket.inet_pton(family, host)
        except (OSError, TypeError):
            pass
        else:
            return address
    # Special cases to match the stdlib, see gh-277
    if host == "":
        host = None
    if host == "<broadcast>":
        host = "255.255.255.255"
    flags = 0
    if local:
        flags |= _stdlib_socket.AI_PASSIVE
    # Since we always pass in an explicit family here, AI_ADDRCONFIG
    # doesn't add any value -- if we have no ipv6 connectivity and are
    # working with an ipv6 socket, then things will break soon enough! And
    # if we do enable it, then it makes it impossible to even run tests
    # for ipv6 address resolution on travis-ci, which as of 2017-03-07 has
    # no ipv6.
    # flags |= AI_ADDRCONFIG
    if family == _stdlib_socket.AF_INET6 and not ipv6_v6only:
        flags |= _stdlib_socket.AI_V4MAPPED
    gai_res = await getaddrinfo(host, port, family, type_, proto, flags)
    # AFAICT from the spec it's not possible for getaddrinfo to return an
    # empty list.
    assert len(gai_res) >= 1
    # Address is the last item in the first entry
    (*_, normed), *_ = gai_res
    # The above ignored any flowid and scopeid in the passed-in address,
    # so restore them if present:
    if family == _stdlib_socket.AF_INET6:
        list_normed = list(normed)
        assert len(normed) == 4
        if len(address) >= 3:
            list_normed[2] = address[2]
        if len(address) >= 4:
            list_normed[3] = address[3]
        return tuple(list_normed)
    return normed


class SocketType:
    def __init__(self) -> None:
        # make sure this __init__ works with multiple inheritance
        super().__init__()
        # and only raises error if it's directly constructed
        if type(self) is SocketType:
            raise TypeError(
                "SocketType is an abstract class; use trio.socket.socket if you "
                "want to construct a socket object",
            )

    def detach(self) -> int:
        raise NotImplementedError

    def fileno(self) -> int:
        raise NotImplementedError

    def getpeername(self) -> AddressFormat:
        raise NotImplementedError

    def getsockname(self) -> AddressFormat:
        raise NotImplementedError

    @overload
    def getsockopt(self, level: int, optname: int) -> int: ...

    @overload
    def getsockopt(self, level: int, optname: int, buflen: int) -> bytes: ...

    def getsockopt(
        self,
        level: int,
        optname: int,
        buflen: int | None = None,
    ) -> int | bytes:
        raise NotImplementedError

    @overload
    def setsockopt(self, level: int, optname: int, value: int | Buffer) -> None: ...

    @overload
    def setsockopt(
        self,
        level: int,
        optname: int,
        value: None,
        optlen: int,
    ) -> None: ...

    def setsockopt(
        self,
        level: int,
        optname: int,
        value: int | Buffer | None,
        optlen: int | None = None,
    ) -> None:
        raise NotImplementedError

    def listen(self, backlog: int = min(_stdlib_socket.SOMAXCONN, 128)) -> None:
        raise NotImplementedError

    def get_inheritable(self) -> bool:
        raise NotImplementedError

    def set_inheritable(self, inheritable: bool) -> None:
        raise NotImplementedError

    if sys.platform == "win32" or (
        not TYPE_CHECKING and hasattr(_stdlib_socket.socket, "share")
    ):

        def share(self, process_id: int) -> bytes:
            raise NotImplementedError

    def __enter__(self) -> Self:
        raise NotImplementedError

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        raise NotImplementedError

    @property
    def family(self) -> AddressFamily:
        raise NotImplementedError

    @property
    def type(self) -> SocketKind:
        raise NotImplementedError

    @property
    def proto(self) -> int:
        raise NotImplementedError

    @property
    def did_shutdown_SHUT_WR(self) -> bool:
        """Return True if the socket has been shut down with the SHUT_WR flag"""
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def dup(self) -> SocketType:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    async def bind(self, address: AddressFormat) -> None:
        raise NotImplementedError

    def shutdown(self, flag: int) -> None:
        raise NotImplementedError

    def is_readable(self) -> bool:
        """Return True if the socket is readable. This is checked with `select.select` on Windows, otherwise `select.poll`."""
        raise NotImplementedError

    async def wait_writable(self) -> None:
        """Convenience method that calls trio.lowlevel.wait_writable for the object."""
        raise NotImplementedError

    async def accept(self) -> tuple[SocketType, AddressFormat]:
        raise NotImplementedError

    async def connect(self, address: AddressFormat) -> None:
        raise NotImplementedError

    def recv(self, buflen: int, flags: int = 0, /) -> Awaitable[bytes]:
        raise NotImplementedError

    def recv_into(
        self,
        buffer: Buffer,
        nbytes: int = 0,
        flags: int = 0,
    ) -> Awaitable[int]:
        raise NotImplementedError

    # return type of socket.socket.recvfrom in typeshed is tuple[bytes, Any]
    def recvfrom(
        self,
        bufsize: int,
        flags: int = 0,
        /,
    ) -> Awaitable[tuple[bytes, AddressFormat]]:
        raise NotImplementedError

    # return type of socket.socket.recvfrom_into in typeshed is tuple[bytes, Any]
    def recvfrom_into(
        self,
        buffer: Buffer,
        nbytes: int = 0,
        flags: int = 0,
    ) -> Awaitable[tuple[int, AddressFormat]]:
        raise NotImplementedError

    if sys.platform != "win32" or (
        not TYPE_CHECKING and hasattr(_stdlib_socket.socket, "recvmsg")
    ):

        def recvmsg(
            self,
            bufsize: int,
            ancbufsize: int = 0,
            flags: int = 0,
            /,
        ) -> Awaitable[tuple[bytes, list[tuple[int, int, bytes]], int, object]]:
            raise NotImplementedError

    if sys.platform != "win32" or (
        not TYPE_CHECKING and hasattr(_stdlib_socket.socket, "recvmsg_into")
    ):

        def recvmsg_into(
            self,
            buffers: Iterable[Buffer],
            ancbufsize: int = 0,
            flags: int = 0,
            /,
        ) -> Awaitable[tuple[int, list[tuple[int, int, bytes]], int, object]]:
            raise NotImplementedError

    def send(self, bytes: Buffer, flags: int = 0, /) -> Awaitable[int]:
        raise NotImplementedError

    @overload
    async def sendto(
        self,
        data: Buffer,
        address: tuple[object, ...] | str | Buffer,
        /,
    ) -> int: ...

    @overload
    async def sendto(
        self,
        data: Buffer,
        flags: int,
        address: tuple[object, ...] | str | Buffer,
        /,
    ) -> int: ...

    async def sendto(self, *args: object) -> int:
        raise NotImplementedError

    if sys.platform != "win32" or (
        not TYPE_CHECKING and hasattr(_stdlib_socket.socket, "sendmsg")
    ):

        @_wraps(_stdlib_socket.socket.sendmsg, assigned=(), updated=())
        async def sendmsg(
            self,
            buffers: Iterable[Buffer],
            ancdata: Iterable[tuple[int, int, Buffer]] = (),
            flags: int = 0,
            address: AddressFormat | None = None,
            /,
        ) -> int:
            raise NotImplementedError


# copy docstrings from socket.SocketType / socket.socket
for name, obj in SocketType.__dict__.items():
    # skip dunders and already defined docstrings
    if name.startswith("__") or obj.__doc__:
        continue
    # try both socket.socket and socket.SocketType
    for stdlib_type in _stdlib_socket.socket, _stdlib_socket.SocketType:
        stdlib_obj = getattr(stdlib_type, name, None)
        if stdlib_obj and stdlib_obj.__doc__:
            break
    else:
        continue
    obj.__doc__ = stdlib_obj.__doc__


class _SocketType(SocketType):
    def __init__(self, sock: _stdlib_socket.socket) -> None:
        if type(sock) is not _stdlib_socket.socket:
            # For example, ssl.SSLSocket subclasses socket.socket, but we
            # certainly don't want to blindly wrap one of those.
            raise TypeError(
                f"expected object of type 'socket.socket', not '{type(sock).__name__}'",
            )
        self._sock = sock
        self._sock.setblocking(False)
        self._did_shutdown_SHUT_WR = False

    ################################################################
    # Simple + portable methods and attributes
    ################################################################

    # forwarded methods
    def detach(self) -> int:
        return self._sock.detach()

    def fileno(self) -> int:
        return self._sock.fileno()

    def getpeername(self) -> AddressFormat:
        return self._sock.getpeername()

    def getsockname(self) -> AddressFormat:
        return self._sock.getsockname()

    @overload
    def getsockopt(self, level: int, optname: int) -> int: ...

    @overload
    def getsockopt(self, level: int, optname: int, buflen: int) -> bytes: ...

    def getsockopt(
        self,
        level: int,
        optname: int,
        buflen: int | None = None,
    ) -> int | bytes:
        if buflen is None:
            return self._sock.getsockopt(level, optname)
        return self._sock.getsockopt(level, optname, buflen)

    @overload
    def setsockopt(self, level: int, optname: int, value: int | Buffer) -> None: ...

    @overload
    def setsockopt(
        self,
        level: int,
        optname: int,
        value: None,
        optlen: int,
    ) -> None: ...

    def setsockopt(
        self,
        level: int,
        optname: int,
        value: int | Buffer | None,
        optlen: int | None = None,
    ) -> None:
        if optlen is None:
            if value is None:
                raise TypeError(
                    "invalid value for argument 'value', must not be None when specifying optlen",
                )
            return self._sock.setsockopt(level, optname, value)
        if value is not None:
            raise TypeError(
                f"invalid value for argument 'value': {value!r}, must be None when specifying optlen",
            )

        # Note: PyPy may crash here due to setsockopt only supporting
        # four parameters.
        return self._sock.setsockopt(level, optname, value, optlen)

    def listen(self, backlog: int = min(_stdlib_socket.SOMAXCONN, 128)) -> None:
        return self._sock.listen(backlog)

    def get_inheritable(self) -> bool:
        return self._sock.get_inheritable()

    def set_inheritable(self, inheritable: bool) -> None:
        return self._sock.set_inheritable(inheritable)

    if sys.platform == "win32" or (
        not TYPE_CHECKING and hasattr(_stdlib_socket.socket, "share")
    ):

        def share(self, process_id: int) -> bytes:
            return self._sock.share(process_id)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return self._sock.__exit__(exc_type, exc_value, traceback)

    @property
    def family(self) -> AddressFamily:
        return self._sock.family

    @property
    def type(self) -> SocketKind:
        return self._sock.type

    @property
    def proto(self) -> int:
        return self._sock.proto

    @property
    def did_shutdown_SHUT_WR(self) -> bool:
        return self._did_shutdown_SHUT_WR

    def __repr__(self) -> str:
        return repr(self._sock).replace("socket.socket", "trio.socket.socket")

    def dup(self) -> SocketType:
        """Same as :meth:`socket.socket.dup`."""
        return _SocketType(self._sock.dup())

    def close(self) -> None:
        if self._sock.fileno() != -1:
            trio.lowlevel.notify_closing(self._sock)
            self._sock.close()

    async def bind(self, address: AddressFormat) -> None:
        address = await self._resolve_address_nocp(address, local=True)
        if (
            hasattr(_stdlib_socket, "AF_UNIX")
            and self.family == _stdlib_socket.AF_UNIX
            and address[0]
        ):
            # Use a thread for the filesystem traversal (unless it's an
            # abstract domain socket)
            return await trio.to_thread.run_sync(self._sock.bind, address)
        else:
            # POSIX actually says that bind can return EWOULDBLOCK and
            # complete asynchronously, like connect. But in practice AFAICT
            # there aren't yet any real systems that do this, so we'll worry
            # about it when it happens.
            await trio.lowlevel.checkpoint()
            return self._sock.bind(address)

    def shutdown(self, flag: int) -> None:
        # no need to worry about return value b/c always returns None:
        self._sock.shutdown(flag)
        # only do this if the call succeeded:
        if flag in [_stdlib_socket.SHUT_WR, _stdlib_socket.SHUT_RDWR]:
            self._did_shutdown_SHUT_WR = True

    def is_readable(self) -> bool:
        # use select.select on Windows, and select.poll everywhere else
        if sys.platform == "win32":
            rready, _, _ = select.select([self._sock], [], [], 0)
            return bool(rready)
        p = select.poll()
        p.register(self._sock, select.POLLIN)
        return bool(p.poll(0))

    async def wait_writable(self) -> None:
        await _core.wait_writable(self._sock)

    async def _resolve_address_nocp(
        self,
        address: AddressFormat,
        *,
        local: bool,
    ) -> AddressFormat:
        if self.family == _stdlib_socket.AF_INET6:
            ipv6_v6only = self._sock.getsockopt(
                _stdlib_socket.IPPROTO_IPV6,
                _stdlib_socket.IPV6_V6ONLY,
            )
        else:
            ipv6_v6only = False
        return await _resolve_address_nocp(
            self.type,
            self.family,
            self.proto,
            ipv6_v6only=ipv6_v6only,
            address=address,
            local=local,
        )

    # args and kwargs must be starred, otherwise pyright complains:
    # '"args" member of ParamSpec is valid only when used with *args parameter'
    # '"kwargs" member of ParamSpec is valid only when used with **kwargs parameter'
    # wait_fn and fn must also be first in the signature
    # 'Keyword parameter cannot appear in signature after ParamSpec args parameter'

    async def _nonblocking_helper(
        self,
        wait_fn: Callable[[_stdlib_socket.socket], Awaitable[None]],
        fn: Callable[Concatenate[_stdlib_socket.socket, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        # We have to reconcile two conflicting goals:
        # - We want to make it look like we always blocked in doing these
        #   operations. The obvious way is to always do an IO wait before
        #   calling the function.
        # - But, we also want to provide the correct semantics, and part
        #   of that means giving correct errors. So, for example, if you
        #   haven't called .listen(), then .accept() raises an error
        #   immediately. But in this same circumstance, then on macOS, the
        #   socket does not register as readable. So if we block waiting
        #   for read *before* we call accept, then we'll be waiting
        #   forever instead of properly raising an error. (On Linux,
        #   interestingly, AFAICT a socket that can't possible read/write
        #   *does* count as readable/writable for select() purposes. But
        #   not on macOS.)
        #
        # So, we have to call the function once, with the appropriate
        # cancellation/yielding sandwich if it succeeds, and if it gives
        # BlockingIOError *then* we fall back to IO wait.
        #
        # XX think if this can be combined with the similar logic for IOCP
        # submission...
        async with _try_sync():
            return fn(self._sock, *args, **kwargs)
        # First attempt raised BlockingIOError:
        while True:
            await wait_fn(self._sock)
            try:
                return fn(self._sock, *args, **kwargs)
            except BlockingIOError:
                pass

    ################################################################
    # accept
    ################################################################

    _accept = _make_simple_sock_method_wrapper(
        _stdlib_socket.socket.accept,
        _core.wait_readable,
    )

    async def accept(self) -> tuple[SocketType, AddressFormat]:
        """Like :meth:`socket.socket.accept`, but async."""
        sock, addr = await self._accept()
        return from_stdlib_socket(sock), addr

    ################################################################
    # connect
    ################################################################

    async def connect(self, address: AddressFormat) -> None:
        # nonblocking connect is weird -- you call it to start things
        # off, then the socket becomes writable as a completion
        # notification. This means it isn't really cancellable... we close the
        # socket if cancelled, to avoid confusion.
        try:
            address = await self._resolve_address_nocp(address, local=False)
            async with _try_sync():
                # An interesting puzzle: can a non-blocking connect() return EINTR
                # (= raise InterruptedError)? PEP 475 specifically left this as
                # the one place where it lets an InterruptedError escape instead
                # of automatically retrying. This is based on the idea that EINTR
                # from connect means that the connection was already started, and
                # will continue in the background. For a blocking connect, this
                # sort of makes sense: if it returns EINTR then the connection
                # attempt is continuing in the background, and on many system you
                # can't then call connect() again because there is already a
                # connect happening. See:
                #
                #   http://www.madore.org/~david/computers/connect-intr.html
                #
                # For a non-blocking connect, it doesn't make as much sense --
                # surely the interrupt didn't happen after we successfully
                # initiated the connect and are just waiting for it to complete,
                # because a non-blocking connect does not wait! And the spec
                # describes the interaction between EINTR/blocking connect, but
                # doesn't have anything useful to say about non-blocking connect:
                #
                #   http://pubs.opengroup.org/onlinepubs/007904975/functions/connect.html
                #
                # So we have a conundrum: if EINTR means that the connect() hasn't
                # happened (like it does for essentially every other syscall),
                # then InterruptedError should be caught and retried. If EINTR
                # means that the connect() has successfully started, then
                # InterruptedError should be caught and ignored. Which should we
                # do?
                #
                # In practice, the resolution is probably that non-blocking
                # connect simply never returns EINTR, so the question of how to
                # handle it is moot.  Someone spelunked macOS/FreeBSD and
                # confirmed this is true there:
                #
                #   https://stackoverflow.com/questions/14134440/eintr-and-non-blocking-calls
                #
                # and exarkun seems to think it's true in general of non-blocking
                # calls:
                #
                #   https://twistedmatrix.com/pipermail/twisted-python/2010-September/022864.html
                # (and indeed, AFAICT twisted doesn't try to handle
                # InterruptedError).
                #
                # So we don't try to catch InterruptedError. This way if it
                # happens, someone will hopefully tell us, and then hopefully we
                # can investigate their system to figure out what its semantics
                # are.
                return self._sock.connect(address)
            # It raised BlockingIOError, meaning that it's started the
            # connection attempt. We wait for it to complete:
            await _core.wait_writable(self._sock)
        except trio.Cancelled:
            # We can't really cancel a connect, and the socket is in an
            # indeterminate state. Better to close it so we don't get
            # confused.
            self._sock.close()
            raise
        # Okay, the connect finished, but it might have failed:
        err = self._sock.getsockopt(_stdlib_socket.SOL_SOCKET, _stdlib_socket.SO_ERROR)
        if err != 0:
            raise OSError(err, f"Error connecting to {address!r}: {os.strerror(err)}")

    ################################################################
    # recv
    ################################################################

    # Not possible to typecheck with a Callable (due to DefaultArg), nor with a
    # callback Protocol (https://github.com/python/typing/discussions/1040)
    # but this seems to work. If not explicitly defined then pyright --verifytypes will
    # complain about AmbiguousType
    if TYPE_CHECKING:

        def recv(self, buflen: int, flags: int = 0, /) -> Awaitable[bytes]: ...

    recv = _make_simple_sock_method_wrapper(
        _stdlib_socket.socket.recv,
        _core.wait_readable,
    )

    ################################################################
    # recv_into
    ################################################################

    if TYPE_CHECKING:

        def recv_into(
            self,
            /,
            buffer: Buffer,
            nbytes: int = 0,
            flags: int = 0,
        ) -> Awaitable[int]: ...

    recv_into = _make_simple_sock_method_wrapper(
        _stdlib_socket.socket.recv_into,
        _core.wait_readable,
    )

    ################################################################
    # recvfrom
    ################################################################

    if TYPE_CHECKING:
        # return type of socket.socket.recvfrom in typeshed is tuple[bytes, Any]
        def recvfrom(
            self,
            bufsize: int,
            flags: int = 0,
            /,
        ) -> Awaitable[tuple[bytes, AddressFormat]]: ...

    recvfrom = _make_simple_sock_method_wrapper(
        _stdlib_socket.socket.recvfrom,
        _core.wait_readable,
    )

    ################################################################
    # recvfrom_into
    ################################################################

    if TYPE_CHECKING:
        # return type of socket.socket.recvfrom_into in typeshed is tuple[bytes, Any]
        def recvfrom_into(
            self,
            /,
            buffer: Buffer,
            nbytes: int = 0,
            flags: int = 0,
        ) -> Awaitable[tuple[int, AddressFormat]]: ...

    recvfrom_into = _make_simple_sock_method_wrapper(
        _stdlib_socket.socket.recvfrom_into,
        _core.wait_readable,
    )

    ################################################################
    # recvmsg
    ################################################################

    if sys.platform != "win32" or (
        not TYPE_CHECKING and hasattr(_stdlib_socket.socket, "recvmsg")
    ):
        if TYPE_CHECKING:

            def recvmsg(
                self,
                bufsize: int,
                ancbufsize: int = 0,
                flags: int = 0,
                /,
            ) -> Awaitable[tuple[bytes, list[tuple[int, int, bytes]], int, object]]: ...

        recvmsg = _make_simple_sock_method_wrapper(
            _stdlib_socket.socket.recvmsg,
            _core.wait_readable,
            maybe_avail=True,
        )

    ################################################################
    # recvmsg_into
    ################################################################

    if sys.platform != "win32" or (
        not TYPE_CHECKING and hasattr(_stdlib_socket.socket, "recvmsg_into")
    ):
        if TYPE_CHECKING:

            def recvmsg_into(
                self,
                buffers: Iterable[Buffer],
                ancbufsize: int = 0,
                flags: int = 0,
                /,
            ) -> Awaitable[tuple[int, list[tuple[int, int, bytes]], int, object]]: ...

        recvmsg_into = _make_simple_sock_method_wrapper(
            _stdlib_socket.socket.recvmsg_into,
            _core.wait_readable,
            maybe_avail=True,
        )

    ################################################################
    # send
    ################################################################

    if TYPE_CHECKING:

        def send(self, bytes: Buffer, flags: int = 0, /) -> Awaitable[int]: ...

    send = _make_simple_sock_method_wrapper(
        _stdlib_socket.socket.send,
        _core.wait_writable,
    )

    ################################################################
    # sendto
    ################################################################

    @overload
    async def sendto(
        self,
        data: Buffer,
        address: tuple[object, ...] | str | Buffer,
        /,
    ) -> int: ...

    @overload
    async def sendto(
        self,
        data: Buffer,
        flags: int,
        address: tuple[object, ...] | str | Buffer,
        /,
    ) -> int: ...

    @_wraps(_stdlib_socket.socket.sendto, assigned=(), updated=())
    async def sendto(self, *args: object) -> int:
        """Similar to :meth:`socket.socket.sendto`, but async."""
        # args is: data[, flags], address
        # and kwargs are not accepted
        args_list = list(args)
        args_list[-1] = await self._resolve_address_nocp(args[-1], local=False)
        # args_list is Any, which isn't the signature of sendto().
        # We don't care about invalid types, sendto() will do the checking.
        return await self._nonblocking_helper(
            _core.wait_writable,
            _stdlib_socket.socket.sendto,  # type: ignore[arg-type]
            *args_list,
        )

    ################################################################
    # sendmsg
    ################################################################

    if sys.platform != "win32" or (
        not TYPE_CHECKING and hasattr(_stdlib_socket.socket, "sendmsg")
    ):

        @_wraps(_stdlib_socket.socket.sendmsg, assigned=(), updated=())
        async def sendmsg(
            self,
            buffers: Iterable[Buffer],
            ancdata: Iterable[tuple[int, int, Buffer]] = (),
            flags: int = 0,
            address: AddressFormat | None = None,
            /,
        ) -> int:
            """Similar to :meth:`socket.socket.sendmsg`, but async.

            Only available on platforms where :meth:`socket.socket.sendmsg` is
            available.

            """
            if address is not None:
                address = await self._resolve_address_nocp(address, local=False)
            return await self._nonblocking_helper(
                _core.wait_writable,
                _stdlib_socket.socket.sendmsg,
                buffers,
                ancdata,
                flags,
                address,
            )

    ################################################################
    # sendfile
    ################################################################

    # Not implemented yet:
    # async def sendfile(self, file, offset=0, count=None):
    #     XX

    # Intentionally omitted:
    #   sendall
    #   makefile
    #   setblocking/getblocking
    #   settimeout/gettimeout
    #   timeout

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\diffusion_models.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modified from stable_diffusion_tensorrt_txt2img.py in diffusers and TensorRT demo diffusion,
# which has the following license:
#
# Copyright 2023 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import tempfile

import onnx
import onnx_graphsurgeon as gs
import torch
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from onnx import GraphProto, ModelProto, shape_inference
from ort_optimizer import OrtStableDiffusionOptimizer
from polygraphy.backend.onnx.loader import fold_constants
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from onnxruntime.transformers.onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class TrtOptimizer:
    def __init__(self, onnx_graph):
        self.graph = gs.import_onnx(onnx_graph)

    def cleanup(self):
        self.graph.cleanup().toposort()

    def get_optimized_onnx_graph(self):
        return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)

    def infer_shapes(self):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_onnx_path = os.path.join(temp_dir, "model.onnx")
                onnx.save_model(
                    onnx_graph,
                    input_onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False,
                )
                output_onnx_path = os.path.join(temp_dir, "model_with_shape.onnx")
                onnx.shape_inference.infer_shapes_path(input_onnx_path, output_onnx_path)
                onnx_graph = onnx.load(output_onnx_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)


class PipelineInfo:
    def __init__(
        self,
        version: str,
        is_inpaint: bool = False,
        is_refiner: bool = False,
        use_vae=True,  # TODO: this has couple with output type of pipeline
        min_image_size=256,
        max_image_size=1024,
        use_fp16_vae=True,
        use_lcm=False,
        do_classifier_free_guidance=True,
        controlnet=None,
        lora_weights=None,
        lora_scale=1.0,
    ):
        self.version = version
        self._is_inpaint = is_inpaint
        self._is_refiner = is_refiner
        self._use_vae = use_vae
        self._min_image_size = min_image_size
        self._max_image_size = max_image_size
        self._use_fp16_vae = use_fp16_vae
        self._use_lcm = use_lcm
        self.do_classifier_free_guidance = do_classifier_free_guidance and not use_lcm
        self.controlnet = controlnet  # A list of control net type
        self.lora_weights = lora_weights
        self.lora_scale = lora_scale

        if is_refiner:
            assert not use_lcm
            assert self.is_xl()

    def is_inpaint(self) -> bool:
        return self._is_inpaint

    def is_xl(self) -> bool:
        return "xl" in self.version

    def is_xl_turbo(self) -> bool:
        return self.version == "xl-turbo"

    def is_xl_base(self) -> bool:
        return self.version == "xl-1.0" and not self._is_refiner

    def is_xl_base_or_turbo(self) -> bool:
        return self.is_xl_base() or self.is_xl_turbo()

    def is_xl_refiner(self) -> bool:
        return self.version == "xl-1.0" and self._is_refiner

    def use_safetensors(self) -> bool:
        return self.is_xl() or self.version in ["sd-turbo"]

    def stages(self) -> list[str]:
        if self.is_xl_base_or_turbo():
            return ["clip", "clip2", "unetxl"] + (["vae"] if self._use_vae else [])

        if self.is_xl_refiner():
            return ["clip2", "unetxl", "vae"]

        return ["clip", "unet", "vae"]

    def vae_scaling_factor(self) -> float:
        return 0.13025 if self.is_xl() else 0.18215

    def vae_torch_fallback(self) -> bool:
        return self.is_xl() and not self._use_fp16_vae

    def custom_fp16_vae(self) -> str | None:
        # For SD XL, use a VAE that fine-tuned to run in fp16 precision without generating NaNs
        return "madebyollin/sdxl-vae-fp16-fix" if self._use_fp16_vae and self.is_xl() else None

    def custom_unet(self) -> str | None:
        return "latent-consistency/lcm-sdxl" if self._use_lcm and self.is_xl_base() else None

    @staticmethod
    def supported_versions(is_xl: bool):
        return ["xl-1.0", "xl-turbo"] if is_xl else ["1.4", "1.5", "2.0-base", "2.0", "2.1", "2.1-base", "sd-turbo"]

    @staticmethod
    def supported_models():
        return {
            "CompVis/stable-diffusion-v1-4": "1.4",
            "runwayml/stable-diffusion-v1-5": "1.5",
            "stabilityai/stable-diffusion-2-base": "2.0-base",
            "stabilityai/stable-diffusion-2": "2.0",
            "stabilityai/stable-diffusion-2-1": "2.1",
            "stabilityai/stable-diffusion-2-1-base": "2.1",
            "stabilityai/stable-diffusion-xl-base-1.0": "xl-1.0",
            "stabilityai/stable-diffusion-xl-refiner-1.0": "xl-1.0",
            "stabilityai/sdxl-turbo": "xl-turbo",
            "stabilityai/sd-turbo": "sd-turbo",
            # "runwayml/stable-diffusion-inpainting": "1.5",
            # "stabilityai/stable-diffusion-2-inpainting": "2.0",
        }

    def name(self) -> str:
        if self.version == "1.4":
            if self.is_inpaint():
                return "runwayml/stable-diffusion-inpainting"
            else:
                return "CompVis/stable-diffusion-v1-4"
        elif self.version == "1.5":
            if self.is_inpaint():
                return "runwayml/stable-diffusion-inpainting"
            else:
                return "runwayml/stable-diffusion-v1-5"
        elif self.version == "2.0-base":
            if self.is_inpaint():
                return "stabilityai/stable-diffusion-2-inpainting"
            else:
                return "stabilityai/stable-diffusion-2-base"
        elif self.version == "2.0":
            if self.is_inpaint():
                return "stabilityai/stable-diffusion-2-inpainting"
            else:
                return "stabilityai/stable-diffusion-2"
        elif self.version == "2.1":
            return "stabilityai/stable-diffusion-2-1"
        elif self.version == "2.1-base":
            return "stabilityai/stable-diffusion-2-1-base"
        elif self.version == "xl-1.0":
            if self.is_xl_refiner():
                return "stabilityai/stable-diffusion-xl-refiner-1.0"
            else:
                return "stabilityai/stable-diffusion-xl-base-1.0"
        elif self.version == "xl-turbo":
            return "stabilityai/sdxl-turbo"
        elif self.version == "sd-turbo":
            return "stabilityai/sd-turbo"

        raise ValueError(f"Incorrect version {self.version}")

    def short_name(self) -> str:
        return self.name().split("/")[-1].replace("stable-diffusion", "sd")

    def clip_embedding_dim(self):
        # TODO: can we read from config instead
        if self.version in ("1.4", "1.5"):
            return 768
        elif self.version in ("2.0", "2.0-base", "2.1", "2.1-base", "sd-turbo"):
            return 1024
        elif self.is_xl_base_or_turbo():
            return 768
        else:
            raise ValueError(f"Invalid version {self.version}")

    def clipwithproj_embedding_dim(self):
        if self.is_xl():
            return 1280
        else:
            raise ValueError(f"Invalid version {self.version}")

    def unet_embedding_dim(self):
        if self.version in ("1.4", "1.5"):
            return 768
        elif self.version in ("2.0", "2.0-base", "2.1", "2.1-base", "sd-turbo"):
            return 1024
        elif self.is_xl_base_or_turbo():
            return 2048
        elif self.is_xl_refiner():
            return 1280
        else:
            raise ValueError(f"Invalid version {self.version}")

    def min_image_size(self):
        return self._min_image_size

    def max_image_size(self):
        return self._max_image_size

    @staticmethod
    def default_resolution(version: str) -> int:
        if version == "xl-1.0":
            return 1024
        if version in ("2.0", "2.1"):
            return 768
        return 512

    def default_image_size(self) -> int:
        return PipelineInfo.default_resolution(self.version)

    @staticmethod
    def supported_controlnet(version="1.5"):
        if version in ("xl-1.0", "xl-turbo"):
            return {
                "canny": "diffusers/controlnet-canny-sdxl-1.0",
                "depth": "diffusers/controlnet-depth-sdxl-1.0",
            }
        elif version == "1.5":
            return {
                "canny": "lllyasviel/control_v11p_sd15_canny",
                "depth": "lllyasviel/control_v11f1p_sd15_depth",
                "openpose": "lllyasviel/control_v11p_sd15_openpose",
                # "tile": "lllyasviel/control_v11f1e_sd15_tile",
                # "lineart": "lllyasviel/control_v11p_sd15_lineart",
                # "inpaint": "lllyasviel/control_v11p_sd15_inpaint",
                # "softedge": "lllyasviel/control_v11p_sd15_softedge",
                "mlsd": "lllyasviel/control_v11p_sd15_mlsd",
                "scribble": "lllyasviel/control_v11p_sd15_scribble",
                # "ip2p": "lllyasviel/control_v11e_sd15_ip2p",
                "normalbae": "lllyasviel/control_v11p_sd15_normalbae",
                "seg": "lllyasviel/control_v11p_sd15_seg",
                # "shuffle": "lllyasviel/control_v11e_sd15_shuffle",
                # "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
            }
        return None

    def controlnet_name(self):
        """Return a list of controlnet name"""
        if not self.controlnet:
            return None
        controlnet_map = PipelineInfo.supported_controlnet(self.version)
        if controlnet_map is None:
            return None
        return [controlnet_map[controlnet] for controlnet in self.controlnet]


class BaseModel:
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        fp16: bool = False,
        max_batch_size: int = 16,
        embedding_dim: int = 768,
        text_maxlen: int = 77,
    ):
        self.name = self.__class__.__name__

        self.pipeline_info = pipeline_info

        self.model = model
        self.fp16 = fp16
        self.device = device

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = pipeline_info.min_image_size()
        self.max_image_shape = pipeline_info.max_image_size()
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_batch_multiplier(self):
        return 2 if self.pipeline_info.do_classifier_free_guidance else 1

    def get_ort_optimizer(self):
        model_name_to_model_type = {
            "CLIP": "clip",
            "UNet": "unet",
            "VAE": "vae",
            "UNetXL": "unet",
            "CLIPWithProj": "clip",
        }
        model_type = model_name_to_model_type[self.name]
        return OrtStableDiffusionOptimizer(model_type)

    def get_model(self):
        return self.model

    def from_pretrained(self, model_class, framework_model_dir, subfolder=None, model_name=None, **kwargs):
        if model_name is None:
            model_name = self.pipeline_info.name()

        if subfolder:
            model_dir = os.path.join(framework_model_dir, model_name, subfolder)
        else:
            model_dir = os.path.join(framework_model_dir, model_name)

        if not os.path.exists(model_dir):
            model = model_class.from_pretrained(
                model_name,
                subfolder=subfolder,
                use_safetensors=self.pipeline_info.use_safetensors(),
                **kwargs,
            ).to(self.device)
            model.save_pretrained(model_dir)
        else:
            print(f"Load {self.name} pytorch model from: {model_dir}")

            model = model_class.from_pretrained(model_dir).to(self.device)
        return model

    def load_model(self, framework_model_dir: str, subfolder: str):
        pass

    def get_input_names(self) -> list[str]:
        pass

    def get_output_names(self) -> list[str]:
        pass

    def get_dynamic_axes(self) -> dict[str, dict[int, str]]:
        pass

    def get_sample_input(self, batch_size, image_height, image_width) -> tuple:
        pass

    def get_profile_id(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        """For TensorRT EP"""
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)

        if (self.name in ["UNet", "UNetXL"]) and (self.get_batch_multiplier() == 1):
            profile_id = f"_b1_{batch_size}" if static_batch else f"_b1_{min_batch}_{max_batch}"
        else:
            profile_id = f"_b_{batch_size}" if static_batch else f"_b_{min_batch}_{max_batch}"

        if self.name != "CLIP":
            if static_image_shape:
                profile_id += f"_h_{image_height}_w_{image_width}"
            else:
                profile_id += f"_h_{min_image_height}_{max_image_height}_w_{min_image_width}_{max_image_width}"

        return profile_id

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        """For TensorRT"""

    def get_shape_dict(self, batch_size, image_height, image_width):
        pass

    def fp32_input_output_names(self) -> list[str]:
        """For CUDA EP, we export ONNX model with FP32 first, then convert it to mixed precision model.
        This is a list of input or output names that are kept as float32 in optimized model.
        """
        return []

    def optimize_ort(
        self,
        input_onnx_path,
        optimized_onnx_path,
        to_fp16=True,
        fp32_op_list=None,
        optimize_by_ort=True,
        optimize_by_fusion=True,
        tmp_dir=None,
    ):
        optimizer = self.get_ort_optimizer()
        optimizer.optimize(
            input_onnx_path,
            optimized_onnx_path,
            float16=to_fp16,
            keep_io_types=self.fp32_input_output_names(),
            fp32_op_list=fp32_op_list,
            optimize_by_ort=optimize_by_ort,
            optimize_by_fusion=optimize_by_fusion,
            tmp_dir=tmp_dir,
        )

    def optimize_trt(self, input_onnx_path, optimized_onnx_path):
        onnx_graph = onnx.load(input_onnx_path)
        opt = TrtOptimizer(onnx_graph)
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        opt.cleanup()
        onnx_opt_graph = opt.get_optimized_onnx_graph()

        if onnx_opt_graph.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
            onnx.save_model(
                onnx_opt_graph,
                optimized_onnx_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )
        else:
            onnx.save(onnx_opt_graph, optimized_onnx_path)

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_image_shape else self.min_image_shape
        max_image_height = image_height if static_image_shape else self.max_image_shape
        min_image_width = image_width if static_image_shape else self.min_image_shape
        max_image_width = image_width if static_image_shape else self.max_image_shape
        min_latent_height = latent_height if static_image_shape else self.min_latent_shape
        max_latent_height = latent_height if static_image_shape else self.max_latent_shape
        min_latent_width = latent_width if static_image_shape else self.min_latent_shape
        max_latent_width = latent_width if static_image_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


class CLIP(BaseModel):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        max_batch_size,
        embedding_dim: int = 0,
        clip_skip=0,
    ):
        super().__init__(
            pipeline_info,
            model=model,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim if embedding_dim > 0 else pipeline_info.clip_embedding_dim(),
        )
        self.output_hidden_state = pipeline_info.is_xl()

        # see https://github.com/huggingface/diffusers/pull/5057 for more information of clip_skip.
        # Clip_skip=1 means that the output of the pre-final layer will be used for computing the prompt embeddings.
        self.clip_skip = clip_skip

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        # The exported onnx model has no hidden_state. For SD-XL, We will add hidden_state to optimized onnx model.
        return ["text_embeddings"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B", 1: "S"}, "text_embeddings": {0: "B", 1: "S"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_image_shape
        )
        return {
            "input_ids": [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

        if self.output_hidden_state:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)

        return output

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return (torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device),)

    def add_hidden_states_graph_output(self, model: ModelProto, optimized_onnx_path, use_external_data_format=False):
        graph: GraphProto = model.graph
        hidden_layers = -1
        for i in range(len(graph.node)):
            for j in range(len(graph.node[i].output)):
                name = graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)

        assert self.clip_skip >= 0 and self.clip_skip < hidden_layers

        node_output_name = f"/text_model/encoder/layers.{hidden_layers - 1 - self.clip_skip}/Add_1_output_0"

        # search the name in outputs of all node
        found = False
        for i in range(len(graph.node)):
            for j in range(len(graph.node[i].output)):
                if graph.node[i].output[j] == node_output_name:
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError("Failed to find hidden_states graph output in clip")

        # Insert a Cast  (fp32 -> fp16) node so that hidden_states has same data type as the first graph output.
        graph_output_name = "hidden_states"
        cast_node = onnx.helper.make_node("Cast", inputs=[node_output_name], outputs=[graph_output_name])
        cast_node.attribute.extend([onnx.helper.make_attribute("to", graph.output[0].type.tensor_type.elem_type)])

        hidden_state = graph.output.add()
        hidden_state.CopyFrom(
            onnx.helper.make_tensor_value_info(
                graph_output_name,
                graph.output[0].type.tensor_type.elem_type,
                ["B", "S", self.embedding_dim],
            )
        )

        onnx_model = OnnxModel(model)
        onnx_model.add_node(cast_node)
        onnx_model.save_model_to_file(optimized_onnx_path, use_external_data_format=use_external_data_format)

    def optimize_ort(
        self,
        input_onnx_path,
        optimized_onnx_path,
        to_fp16=True,
        fp32_op_list=None,
        optimize_by_ort=True,
        optimize_by_fusion=True,
        tmp_dir=None,
    ):
        optimizer = self.get_ort_optimizer()

        if not self.output_hidden_state:
            optimizer.optimize(
                input_onnx_path,
                optimized_onnx_path,
                float16=to_fp16,
                keep_io_types=[],
                fp32_op_list=fp32_op_list,
                keep_outputs=["text_embeddings"],
                optimize_by_ort=optimize_by_ort,
                optimize_by_fusion=optimize_by_fusion,
                tmp_dir=tmp_dir,
            )
        elif optimize_by_fusion:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save to a temporary file so that we can load it with Onnx Runtime.
                logger.info("Saving a temporary model to add hidden_states to graph output ...")
                tmp_model_path = os.path.join(tmp_dir, "model.onnx")

                model = onnx.load(input_onnx_path)
                self.add_hidden_states_graph_output(model, tmp_model_path, use_external_data_format=True)
                optimizer.optimize(
                    tmp_model_path,
                    optimized_onnx_path,
                    float16=to_fp16,
                    keep_io_types=[],
                    fp32_op_list=fp32_op_list,
                    keep_outputs=["text_embeddings", "hidden_states"],
                    optimize_by_ort=optimize_by_ort,
                    optimize_by_fusion=optimize_by_fusion,
                    tmp_dir=tmp_dir,
                )
        else:  # input is optimized model, there is no need to add hidden states.
            optimizer.optimize(
                input_onnx_path,
                optimized_onnx_path,
                float16=to_fp16,
                keep_io_types=[],
                fp32_op_list=fp32_op_list,
                keep_outputs=["text_embeddings", "hidden_states"],
                optimize_by_ort=optimize_by_ort,
                optimize_by_fusion=optimize_by_fusion,
                tmp_dir=tmp_dir,
            )

    def optimize_trt(self, input_onnx_path, optimized_onnx_path):
        onnx_graph = onnx.load(input_onnx_path)
        opt = TrtOptimizer(onnx_graph)
        opt.select_outputs([0])  # delete graph output#1
        opt.cleanup()
        opt.fold_constants()
        opt.infer_shapes()
        opt.select_outputs([0], names=["text_embeddings"])  # rename network output
        opt.cleanup()
        onnx_opt_graph = opt.get_optimized_onnx_graph()
        if self.output_hidden_state:
            self.add_hidden_states_graph_output(onnx_opt_graph, optimized_onnx_path)
        else:
            onnx.save(onnx_opt_graph, optimized_onnx_path)

    def load_model(self, framework_model_dir, subfolder="text_encoder"):
        return self.from_pretrained(CLIPTextModel, framework_model_dir, subfolder)


class CLIPWithProj(CLIP):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        max_batch_size=16,
        clip_skip=0,
    ):
        super().__init__(
            pipeline_info,
            model,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=pipeline_info.clipwithproj_embedding_dim(),
            clip_skip=clip_skip,
        )

    def load_model(self, framework_model_dir, subfolder="text_encoder_2"):
        return self.from_pretrained(CLIPTextModelWithProjection, framework_model_dir, subfolder)

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.embedding_dim),
        }

        if self.output_hidden_state:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)

        return output


class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnets: ControlNetModel):
        super().__init__()
        self.unet = unet
        self.controlnets = controlnets

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_images, controlnet_scales):
        for i, (controlnet_image, conditioning_scale, controlnet) in enumerate(
            zip(controlnet_images, controlnet_scales, self.controlnets, strict=False)
        ):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

            down_samples = [down_sample * conditioning_scale for down_sample in down_samples]
            mid_sample *= conditioning_scale

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples, strict=False)
                ]
                mid_block_res_sample += mid_sample

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )
        return noise_pred[0]


# Modified from convert_stable_diffusion_controlnet_to_onnx.py in diffusers
class UNet2DConditionXLControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnets: ControlNetModel):
        super().__init__()
        self.unet = unet
        self.controlnets = controlnets

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        text_embeds,
        time_ids,
        controlnet_images,
        controlnet_scales,
    ):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        for i, (controlnet_image, conditioning_scale, controlnet) in enumerate(
            zip(controlnet_images, controlnet_scales, self.controlnets, strict=False)
        ):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_image,
                conditioning_scale=conditioning_scale,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples, strict=False)
                ]
                mid_block_res_sample += mid_sample

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return noise_pred[0]


class UNet(BaseModel):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        fp16=False,  # used by TRT
        max_batch_size=16,
        text_maxlen=77,
        unet_dim=4,
    ):
        super().__init__(
            pipeline_info,
            model=model,
            device=device,
            fp16=fp16,
            max_batch_size=max_batch_size,
            embedding_dim=pipeline_info.unet_embedding_dim(),
            text_maxlen=text_maxlen,
        )

        self.unet_dim = unet_dim
        self.controlnet = pipeline_info.controlnet_name()

    def load_model(self, framework_model_dir, subfolder="unet"):
        options = {"variant": "fp16", "torch_dtype": torch.float16}

        model = self.from_pretrained(UNet2DConditionModel, framework_model_dir, subfolder, **options)

        if self.controlnet:
            controlnet_list = []
            for name in self.controlnet:
                controlnet = self.from_pretrained(
                    ControlNetModel,
                    framework_model_dir,
                    subfolder=None,
                    model_name=name,
                    torch_dtype=torch.float16,
                )
                controlnet_list.append(controlnet)

            model = UNet2DConditionControlNetModel(model, torch.nn.ModuleList(controlnet_list))

        if not self.fp16:
            model = model.to(torch.float32)

        return model

    def get_input_names(self):
        if not self.controlnet:
            return ["sample", "timestep", "encoder_hidden_states"]
        else:
            return ["sample", "timestep", "encoder_hidden_states", "controlnet_images", "controlnet_scales"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        b = "2B" if self.get_batch_multiplier() == 2 else "B"
        output = {
            "sample": {0: b, 2: "H", 3: "W"},
            "encoder_hidden_states": {0: b},
            "latent": {0: b, 2: "H", 3: "W"},
        }
        if self.controlnet:
            output.update(
                {
                    "controlnet_images": {1: b, 3: "8H", 4: "8W"},
                }
            )
        return output

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)
        m = self.get_batch_multiplier()
        output = {
            "sample": [
                (m * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (m * batch_size, self.unet_dim, latent_height, latent_width),
                (m * max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (m * min_batch, self.text_maxlen, self.embedding_dim),
                (m * batch_size, self.text_maxlen, self.embedding_dim),
                (m * max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }

        if self.controlnet:
            output.update(
                {
                    "controlnet_images": [
                        (len(self.controlnet), m * min_batch, 3, min_image_height, min_image_width),
                        (len(self.controlnet), m * batch_size, 3, image_height, image_width),
                        (len(self.controlnet), m * max_batch, 3, max_image_height, max_image_width),
                    ]
                }
            )
        return output

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        m = self.get_batch_multiplier()
        output = {
            "sample": (m * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": [1],
            "encoder_hidden_states": (m * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (m * batch_size, 4, latent_height, latent_width),
        }

        if self.controlnet:
            output.update(
                {
                    "controlnet_images": (len(self.controlnet), m * batch_size, 3, image_height, image_width),
                    "controlnet_scales": [len(self.controlnet)],
                }
            )
        return output

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        m = self.get_batch_multiplier()
        output = (
            torch.randn(m * batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device),
            torch.tensor([1.0], dtype=dtype, device=self.device),
            torch.randn(m * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        )

        if self.controlnet:
            output = (
                *output,
                torch.randn(
                    len(self.controlnet), m * batch_size, 3, image_height, image_width, dtype=dtype, device=self.device
                ),
                torch.randn(len(self.controlnet), dtype=dtype, device=self.device),
            )
        return output


class UNetXL(BaseModel):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        fp16=False,  # used by TRT
        max_batch_size=16,
        text_maxlen=77,
        unet_dim=4,
        time_dim=6,
    ):
        super().__init__(
            pipeline_info,
            model,
            device=device,
            fp16=fp16,
            max_batch_size=max_batch_size,
            embedding_dim=pipeline_info.unet_embedding_dim(),
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim
        self.time_dim = time_dim

        self.custom_unet = pipeline_info.custom_unet()
        self.controlnet = pipeline_info.controlnet_name()

    def load_model(self, framework_model_dir, subfolder="unet", always_download_fp16=True):
        options = {"variant": "fp16", "torch_dtype": torch.float16} if self.fp16 or always_download_fp16 else {}

        if self.custom_unet:
            model_dir = os.path.join(framework_model_dir, self.custom_unet, subfolder)
            if not os.path.exists(model_dir):
                unet = UNet2DConditionModel.from_pretrained(self.custom_unet, **options)
                unet.save_pretrained(model_dir)
            else:
                unet = UNet2DConditionModel.from_pretrained(model_dir, **options)
            model = unet.to(self.device)
        else:
            model = self.from_pretrained(UNet2DConditionModel, framework_model_dir, subfolder, **options)

        if always_download_fp16 and not self.fp16:
            model = model.to(torch.float32)

        if self.controlnet:
            cnet_model_opts = {"torch_dtype": torch.float16} if self.fp16 or always_download_fp16 else {}
            controlnets = torch.nn.ModuleList(
                [ControlNetModel.from_pretrained(path, **cnet_model_opts).to(self.device) for path in self.controlnet]
            )
            model = UNet2DConditionXLControlNetModel(model, controlnets)

        if always_download_fp16 and not self.fp16:
            model = model.to(torch.float32)

        return model

    def get_input_names(self):
        input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
        if self.controlnet:
            return [*input_names, "controlnet_images", "controlnet_scales"]
        return input_names

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        b = "2B" if self.get_batch_multiplier() == 2 else "B"
        output = {
            "sample": {0: b, 2: "H", 3: "W"},
            "encoder_hidden_states": {0: b},
            "text_embeds": {0: b},
            "time_ids": {0: b},
            "latent": {0: b, 2: "H", 3: "W"},
        }

        if self.controlnet:
            output.update(
                {
                    "controlnet_images": {1: b, 3: "8H", 4: "8W"},
                }
            )
        return output

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)
        m = self.get_batch_multiplier()
        output = {
            "sample": [
                (m * min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (m * batch_size, self.unet_dim, latent_height, latent_width),
                (m * max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "encoder_hidden_states": [
                (m * min_batch, self.text_maxlen, self.embedding_dim),
                (m * batch_size, self.text_maxlen, self.embedding_dim),
                (m * max_batch, self.text_maxlen, self.embedding_dim),
            ],
            "text_embeds": [(m * min_batch, 1280), (m * batch_size, 1280), (m * max_batch, 1280)],
            "time_ids": [
                (m * min_batch, self.time_dim),
                (m * batch_size, self.time_dim),
                (m * max_batch, self.time_dim),
            ],
        }

        if self.controlnet:
            output.update(
                {
                    "controlnet_images": [
                        (len(self.controlnet), m * min_batch, 3, min_image_height, min_image_width),
                        (len(self.controlnet), m * batch_size, 3, image_height, image_width),
                        (len(self.controlnet), m * max_batch, 3, max_image_height, max_image_width),
                    ],
                }
            )
        return output

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        m = self.get_batch_multiplier()
        output = {
            "sample": (m * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (1,),
            "encoder_hidden_states": (m * batch_size, self.text_maxlen, self.embedding_dim),
            "text_embeds": (m * batch_size, 1280),
            "time_ids": (m * batch_size, self.time_dim),
            "latent": (m * batch_size, 4, latent_height, latent_width),
        }

        if self.controlnet:
            output.update(
                {
                    "controlnet_images": (len(self.controlnet), m * batch_size, 3, image_height, image_width),
                    "controlnet_scales": [len(self.controlnet)],
                }
            )
        return output

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        m = self.get_batch_multiplier()
        if not self.controlnet:
            return (
                torch.randn(
                    m * batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device
                ),
                torch.tensor([1.0], dtype=dtype, device=self.device),
                torch.randn(m * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
                {
                    "added_cond_kwargs": {
                        "text_embeds": torch.randn(m * batch_size, 1280, dtype=dtype, device=self.device),
                        "time_ids": torch.randn(m * batch_size, self.time_dim, dtype=dtype, device=self.device),
                    }
                },
            )
        else:
            # sample, timestep, encoder_hidden_states, text_embeds, time_ids, controlnet_images, controlnet_scales,
            return (
                torch.randn(
                    m * batch_size, self.unet_dim, latent_height, latent_width, dtype=dtype, device=self.device
                ),
                torch.tensor([1.0], dtype=dtype, device=self.device),
                torch.randn(m * batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
                torch.randn(m * batch_size, 1280, dtype=dtype, device=self.device),
                torch.randn(m * batch_size, self.time_dim, dtype=dtype, device=self.device),
                torch.randn(
                    len(self.controlnet), m * batch_size, 3, image_height, image_width, dtype=dtype, device=self.device
                ),
                torch.randn(len(self.controlnet), dtype=dtype, device=self.device),
            )


# VAE Decoder
class VAE(BaseModel):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        model,
        device,
        max_batch_size,
        fp16: bool = False,
        custom_fp16_vae: str | None = None,
    ):
        super().__init__(
            pipeline_info,
            model=model,
            device=device,
            fp16=fp16,
            max_batch_size=max_batch_size,
        )

        # For SD XL, need custom trained fp16 model to speed up, and avoid overflow at the same time.
        self.custom_fp16_vae = custom_fp16_vae

    def load_model(self, framework_model_dir, subfolder: str = "vae_decoder"):
        model_name = self.custom_fp16_vae or self.pipeline_info.name()

        model_dir = os.path.join(framework_model_dir, model_name, subfolder)
        if not os.path.exists(model_dir):
            if self.custom_fp16_vae:
                vae = AutoencoderKL.from_pretrained(self.custom_fp16_vae, torch_dtype=torch.float16).to(self.device)
            else:
                vae = AutoencoderKL.from_pretrained(
                    self.pipeline_info.name(),
                    subfolder="vae",
                    use_safetensors=self.pipeline_info.use_safetensors(),
                ).to(self.device)
            vae.save_pretrained(model_dir)
        else:
            print(f"Load {self.name} pytorch model from: {model_dir}")
            if self.custom_fp16_vae:
                vae = AutoencoderKL.from_pretrained(model_dir, torch_dtype=torch.float16).to(self.device)
            else:
                vae = AutoencoderKL.from_pretrained(model_dir).to(self.device)

        vae.forward = vae.decode
        return vae

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {"latent": {0: "B", 2: "H", 3: "W"}, "images": {0: "B", 2: "8H", 3: "8W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (torch.randn(batch_size, 4, latent_height, latent_width, dtype=dtype, device=self.device),)

    def fp32_input_output_names(self) -> list[str]:
        return []


def get_tokenizer(pipeline_info: PipelineInfo, framework_model_dir, subfolder="tokenizer"):
    tokenizer_dir = os.path.join(framework_model_dir, pipeline_info.name(), subfolder)

    if not os.path.exists(tokenizer_dir):
        model = CLIPTokenizer.from_pretrained(
            pipeline_info.name(),
            subfolder=subfolder,
            use_safetensors=pipeline_info.is_xl(),
        )
        model.save_pretrained(tokenizer_dir)
    else:
        print(f"[I] Load tokenizer pytorch model from: {tokenizer_dir}")
        model = CLIPTokenizer.from_pretrained(tokenizer_dir)
    return model


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae_encoder):
        super().__init__()
        self.vae_encoder = vae_encoder

    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()


class VAEEncoder(BaseModel):
    def __init__(self, pipeline_info: PipelineInfo, model, device, max_batch_size):
        super().__init__(
            pipeline_info,
            model=model,
            device=device,
            max_batch_size=max_batch_size,
        )

    def load_model(self, framework_model_dir, subfolder="vae_encoder"):
        vae = self.from_pretrained(AutoencoderKL, framework_model_dir, subfolder)
        return TorchVAEEncoder(vae)

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {"images": {0: "B", 2: "8H", 3: "8W"}, "latent": {0: "B", 2: "H", 3: "W"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        self.check_dims(batch_size, image_height, image_width)

        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_image_shape)

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\soupsieve\css_parser.py ===
"""CSS selector parser."""
from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast

UNICODE_REPLACEMENT_CHAR = 0xFFFD

# Simple pseudo classes that take no parameters
PSEUDO_SIMPLE = {
    ":any-link",
    ":empty",
    ":first-child",
    ":first-of-type",
    ":in-range",
    ":open",
    ":out-of-range",
    ":last-child",
    ":last-of-type",
    ":link",
    ":only-child",
    ":only-of-type",
    ":root",
    ':checked',
    ':default',
    ':disabled',
    ':enabled',
    ':indeterminate',
    ':optional',
    ':placeholder-shown',
    ':read-only',
    ':read-write',
    ':required',
    ':scope',
    ':defined',
    ':muted'
}

# Supported, simple pseudo classes that match nothing in the Soup Sieve environment
PSEUDO_SIMPLE_NO_MATCH = {
    ':active',
    ':autofill',
    ':buffering',
    ':current',
    ':focus',
    ':focus-visible',
    ':focus-within',
    ':fullscreen',
    ':future',
    ':host',
    ':hover',
    ':local-link',
    ':past',
    ':paused',
    ':picture-in-picture',
    ':playing',
    ':popover-open',
    ':seeking',
    ':stalled',
    ':target',
    ':target-within',
    ':user-invalid',
    ':volume-locked',
    ':visited'
}

# Complex pseudo classes that take selector lists
PSEUDO_COMPLEX = {
    ':contains',
    ':-soup-contains',
    ':-soup-contains-own',
    ':has',
    ':is',
    ':matches',
    ':not',
    ':where'
}

PSEUDO_COMPLEX_NO_MATCH = {
    ':current',
    ':host',
    ':host-context'
}

# Complex pseudo classes that take very specific parameters and are handled special
PSEUDO_SPECIAL = {
    ':dir',
    ':lang',
    ':nth-child',
    ':nth-last-child',
    ':nth-last-of-type',
    ':nth-of-type'
}

PSEUDO_SUPPORTED = PSEUDO_SIMPLE | PSEUDO_SIMPLE_NO_MATCH | PSEUDO_COMPLEX | PSEUDO_COMPLEX_NO_MATCH | PSEUDO_SPECIAL

# Sub-patterns parts
# Whitespace
NEWLINE = r'(?:\r\n|(?!\r\n)[\n\f\r])'
WS = fr'(?:[ \t]|{NEWLINE})'
# Comments
COMMENTS = r'(?:/\*[^*]*\*+(?:[^/*][^*]*\*+)*/)'
# Whitespace with comments included
WSC = fr'(?:{WS}|{COMMENTS})'
# CSS escapes
CSS_ESCAPES = fr'(?:\\(?:[a-f0-9]{{1,6}}{WS}?|[^\r\n\f]|$))'
CSS_STRING_ESCAPES = fr'(?:\\(?:[a-f0-9]{{1,6}}{WS}?|[^\r\n\f]|$|{NEWLINE}))'
# CSS Identifier
IDENTIFIER = fr'''
(?:(?:-?(?:[^\x00-\x2f\x30-\x40\x5B-\x5E\x60\x7B-\x9f]|{CSS_ESCAPES})+|--)
(?:[^\x00-\x2c\x2e\x2f\x3A-\x40\x5B-\x5E\x60\x7B-\x9f]|{CSS_ESCAPES})*)
'''
# `nth` content
NTH = fr'(?:[-+])?(?:[0-9]+n?|n)(?:(?<=n){WSC}*(?:[-+]){WSC}*(?:[0-9]+))?'
# Value: quoted string or identifier
VALUE = fr'''(?:"(?:\\(?:.|{NEWLINE})|[^\\"\r\n\f]+)*?"|'(?:\\(?:.|{NEWLINE})|[^\\'\r\n\f]+)*?'|{IDENTIFIER}+)'''
# Attribute value comparison. `!=` is handled special as it is non-standard.
ATTR = fr'(?:{WSC}*(?P<cmp>[!~^|*$]?=){WSC}*(?P<value>{VALUE})(?:{WSC}*(?P<case>[is]))?)?{WSC}*\]'

# Selector patterns
# IDs (`#id`)
PAT_ID = fr'\#{IDENTIFIER}'
# Classes (`.class`)
PAT_CLASS = fr'\.{IDENTIFIER}'
# Prefix:Tag (`prefix|tag`)
PAT_TAG = fr'(?P<tag_ns>(?:{IDENTIFIER}|\*)?\|)?(?P<tag_name>{IDENTIFIER}|\*)'
# Attributes (`[attr]`, `[attr=value]`, etc.)
PAT_ATTR = fr'\[{WSC}*(?P<attr_ns>(?:{IDENTIFIER}|\*)?\|)?(?P<attr_name>{IDENTIFIER}){ATTR}'
# Pseudo class (`:pseudo-class`, `:pseudo-class(`)
PAT_PSEUDO_CLASS = fr'(?P<name>:{IDENTIFIER})(?P<open>\({WSC}*)?'
# Pseudo class special patterns. Matches `:pseudo-class(` for special case pseudo classes.
PAT_PSEUDO_CLASS_SPECIAL = fr'(?P<name>:{IDENTIFIER})(?P<open>\({WSC}*)'
# Custom pseudo class (`:--custom-pseudo`)
PAT_PSEUDO_CLASS_CUSTOM = fr'(?P<name>:(?=--){IDENTIFIER})'
# Nesting ampersand selector. Matches `&`
PAT_AMP = r'&'
# Closing pseudo group (`)`)
PAT_PSEUDO_CLOSE = fr'{WSC}*\)'
# Pseudo element (`::pseudo-element`)
PAT_PSEUDO_ELEMENT = fr':{PAT_PSEUDO_CLASS}'
# At rule (`@page`, etc.) (not supported)
PAT_AT_RULE = fr'@P{IDENTIFIER}'
# Pseudo class `nth-child` (`:nth-child(an+b [of S]?)`, `:first-child`, etc.)
PAT_PSEUDO_NTH_CHILD = fr'''
(?P<pseudo_nth_child>{PAT_PSEUDO_CLASS_SPECIAL}
(?P<nth_child>{NTH}|even|odd))(?:{WSC}*\)|(?P<of>{COMMENTS}*{WS}{WSC}*of{COMMENTS}*{WS}{WSC}*))
'''
# Pseudo class `nth-of-type` (`:nth-of-type(an+b)`, `:first-of-type`, etc.)
PAT_PSEUDO_NTH_TYPE = fr'''
(?P<pseudo_nth_type>{PAT_PSEUDO_CLASS_SPECIAL}
(?P<nth_type>{NTH}|even|odd)){WSC}*\)
'''
# Pseudo class language (`:lang("*-de", en)`)
PAT_PSEUDO_LANG = fr'{PAT_PSEUDO_CLASS_SPECIAL}(?P<values>{VALUE}(?:{WSC}*,{WSC}*{VALUE})*){WSC}*\)'
# Pseudo class direction (`:dir(ltr)`)
PAT_PSEUDO_DIR = fr'{PAT_PSEUDO_CLASS_SPECIAL}(?P<dir>ltr|rtl){WSC}*\)'
# Combining characters (`>`, `~`, ` `, `+`, `,`)
PAT_COMBINE = fr'{WSC}*?(?P<relation>[,+>~]|{WS}(?![,+>~])){WSC}*'
# Extra: Contains (`:contains(text)`)
PAT_PSEUDO_CONTAINS = fr'{PAT_PSEUDO_CLASS_SPECIAL}(?P<values>{VALUE}(?:{WSC}*,{WSC}*{VALUE})*){WSC}*\)'

# Regular expressions
# CSS escape pattern
RE_CSS_ESC = re.compile(fr'(?:(\\[a-f0-9]{{1,6}}{WSC}?)|(\\[^\r\n\f])|(\\$))', re.I)
RE_CSS_STR_ESC = re.compile(fr'(?:(\\[a-f0-9]{{1,6}}{WS}?)|(\\[^\r\n\f])|(\\$)|(\\{NEWLINE}))', re.I)
# Pattern to break up `nth` specifiers
RE_NTH = re.compile(fr'(?P<s1>[-+])?(?P<a>[0-9]+n?|n)(?:(?<=n){WSC}*(?P<s2>[-+]){WSC}*(?P<b>[0-9]+))?', re.I)
# Pattern to iterate multiple values.
RE_VALUES = re.compile(fr'(?:(?P<value>{VALUE})|(?P<split>{WSC}*,{WSC}*))', re.X)
# Whitespace checks
RE_WS = re.compile(WS)
RE_WS_BEGIN = re.compile(fr'^{WSC}*')
RE_WS_END = re.compile(fr'{WSC}*$')
RE_CUSTOM = re.compile(fr'^{PAT_PSEUDO_CLASS_CUSTOM}$', re.X)

# Constants
# List split token
COMMA_COMBINATOR = ','
# Relation token for descendant
WS_COMBINATOR = " "

# Parse flags
FLG_PSEUDO = 0x01
FLG_NOT = 0x02
FLG_RELATIVE = 0x04
FLG_DEFAULT = 0x08
FLG_HTML = 0x10
FLG_INDETERMINATE = 0x20
FLG_OPEN = 0x40
FLG_IN_RANGE = 0x80
FLG_OUT_OF_RANGE = 0x100
FLG_PLACEHOLDER_SHOWN = 0x200
FLG_FORGIVE = 0x400

# Maximum cached patterns to store
_MAXCACHE = 500


@lru_cache(maxsize=_MAXCACHE)
def _cached_css_compile(
    pattern: str,
    namespaces: ct.Namespaces | None,
    custom: ct.CustomSelectors | None,
    flags: int
) -> cm.SoupSieve:
    """Cached CSS compile."""

    custom_selectors = process_custom(custom)
    return cm.SoupSieve(
        pattern,
        CSSParser(
            pattern,
            custom=custom_selectors,
            flags=flags
        ).process_selectors(),
        namespaces,
        custom,
        flags
    )


def _purge_cache() -> None:
    """Purge the cache."""

    _cached_css_compile.cache_clear()


def process_custom(custom: ct.CustomSelectors | None) -> dict[str, str | ct.SelectorList]:
    """Process custom."""

    custom_selectors = {}
    if custom is not None:
        for key, value in custom.items():
            name = util.lower(key)
            if RE_CUSTOM.match(name) is None:
                raise SelectorSyntaxError(f"The name '{name}' is not a valid custom pseudo-class name")
            if name in custom_selectors:
                raise KeyError(f"The custom selector '{name}' has already been registered")
            custom_selectors[css_unescape(name)] = value
    return custom_selectors


def css_unescape(content: str, string: bool = False) -> str:
    """
    Unescape CSS value.

    Strings allow for spanning the value on multiple strings by escaping a new line.
    """

    def replace(m: Match[str]) -> str:
        """Replace with the appropriate substitute."""

        if m.group(1):
            codepoint = int(m.group(1)[1:], 16)
            if codepoint == 0:
                codepoint = UNICODE_REPLACEMENT_CHAR
            value = chr(codepoint)
        elif m.group(2):
            value = m.group(2)[1:]
        elif m.group(3):
            value = '\ufffd'
        else:
            value = ''

        return value

    return (RE_CSS_ESC if not string else RE_CSS_STR_ESC).sub(replace, content)


def escape(ident: str) -> str:
    """Escape identifier."""

    string = []
    length = len(ident)
    start_dash = length > 0 and ident[0] == '-'
    if length == 1 and start_dash:
        # Need to escape identifier that is a single `-` with no other characters
        string.append(f'\\{ident}')
    else:
        for index, c in enumerate(ident):
            codepoint = ord(c)
            if codepoint == 0x00:
                string.append('\ufffd')
            elif (0x01 <= codepoint <= 0x1F) or codepoint == 0x7F:
                string.append(f'\\{codepoint:x} ')
            elif (index == 0 or (start_dash and index == 1)) and (0x30 <= codepoint <= 0x39):
                string.append(f'\\{codepoint:x} ')
            elif (
                codepoint in (0x2D, 0x5F) or codepoint >= 0x80 or (0x30 <= codepoint <= 0x39) or
                (0x30 <= codepoint <= 0x39) or (0x41 <= codepoint <= 0x5A) or (0x61 <= codepoint <= 0x7A)
            ):
                string.append(c)
            else:
                string.append(f'\\{c}')
    return ''.join(string)


class SelectorPattern:
    """Selector pattern."""

    def __init__(self, name: str, pattern: str) -> None:
        """Initialize."""

        self.name = name
        self.re_pattern = re.compile(pattern, re.I | re.X | re.U)

    def get_name(self) -> str:
        """Get name."""

        return self.name

    def match(self, selector: str, index: int, flags: int) -> Match[str] | None:
        """Match the selector."""

        return self.re_pattern.match(selector, index)


class SpecialPseudoPattern(SelectorPattern):
    """Selector pattern."""

    def __init__(self, patterns: tuple[tuple[str, tuple[str, ...], str, type[SelectorPattern]], ...]) -> None:
        """Initialize."""

        self.patterns = {}
        for p in patterns:
            name = p[0]
            pattern = p[3](name, p[2])
            for pseudo in p[1]:
                self.patterns[pseudo] = pattern

        self.matched_name = None  # type: SelectorPattern | None
        self.re_pseudo_name = re.compile(PAT_PSEUDO_CLASS_SPECIAL, re.I | re.X | re.U)

    def get_name(self) -> str:
        """Get name."""

        return '' if self.matched_name is None else self.matched_name.get_name()

    def match(self, selector: str, index: int, flags: int) -> Match[str] | None:
        """Match the selector."""

        pseudo = None
        m = self.re_pseudo_name.match(selector, index)
        if m:
            name = util.lower(css_unescape(m.group('name')))
            pattern = self.patterns.get(name)
            if pattern:
                pseudo = pattern.match(selector, index, flags)
                if pseudo:
                    self.matched_name = pattern

        return pseudo


class _Selector:
    """
    Intermediate selector class.

    This stores selector data for a compound selector as we are acquiring them.
    Once we are done collecting the data for a compound selector, we freeze
    the data in an object that can be pickled and hashed.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize."""

        self.tag = kwargs.get('tag', None)  # type: ct.SelectorTag | None
        self.ids = kwargs.get('ids', [])  # type: list[str]
        self.classes = kwargs.get('classes', [])  # type: list[str]
        self.attributes = kwargs.get('attributes', [])  # type: list[ct.SelectorAttribute]
        self.nth = kwargs.get('nth', [])  # type: list[ct.SelectorNth]
        self.selectors = kwargs.get('selectors', [])  # type: list[ct.SelectorList]
        self.relations = kwargs.get('relations', [])  # type: list[_Selector]
        self.rel_type = kwargs.get('rel_type', None)  # type: str | None
        self.contains = kwargs.get('contains', [])  # type: list[ct.SelectorContains]
        self.lang = kwargs.get('lang', [])  # type: list[ct.SelectorLang]
        self.flags = kwargs.get('flags', 0)  # type: int
        self.no_match = kwargs.get('no_match', False)  # type: bool

    def _freeze_relations(self, relations: list[_Selector]) -> ct.SelectorList:
        """Freeze relation."""

        if relations:
            sel = relations[0]
            sel.relations.extend(relations[1:])
            return ct.SelectorList([sel.freeze()])
        else:
            return ct.SelectorList()

    def freeze(self) -> ct.Selector | ct.SelectorNull:
        """Freeze self."""

        if self.no_match:
            return ct.SelectorNull()
        else:
            return ct.Selector(
                self.tag,
                tuple(self.ids),
                tuple(self.classes),
                tuple(self.attributes),
                tuple(self.nth),
                tuple(self.selectors),
                self._freeze_relations(self.relations),
                self.rel_type,
                tuple(self.contains),
                tuple(self.lang),
                self.flags
            )

    def __str__(self) -> str:  # pragma: no cover
        """String representation."""

        return (
            f'_Selector(tag={self.tag!r}, ids={self.ids!r}, classes={self.classes!r}, attributes={self.attributes!r}, '
            f'nth={self.nth!r}, selectors={self.selectors!r}, relations={self.relations!r}, '
            f'rel_type={self.rel_type!r}, contains={self.contains!r}, lang={self.lang!r}, flags={self.flags!r}, '
            f'no_match={self.no_match!r})'
        )

    __repr__ = __str__


class CSSParser:
    """Parse CSS selectors."""

    css_tokens = (
        SelectorPattern("pseudo_close", PAT_PSEUDO_CLOSE),
        SpecialPseudoPattern(
            (
                (
                    "pseudo_contains",
                    (':contains', ':-soup-contains', ':-soup-contains-own'),
                    PAT_PSEUDO_CONTAINS,
                    SelectorPattern
                ),
                ("pseudo_nth_child", (':nth-child', ':nth-last-child'), PAT_PSEUDO_NTH_CHILD, SelectorPattern),
                ("pseudo_nth_type", (':nth-of-type', ':nth-last-of-type'), PAT_PSEUDO_NTH_TYPE, SelectorPattern),
                ("pseudo_lang", (':lang',), PAT_PSEUDO_LANG, SelectorPattern),
                ("pseudo_dir", (':dir',), PAT_PSEUDO_DIR, SelectorPattern)
            )
        ),
        SelectorPattern("pseudo_class_custom", PAT_PSEUDO_CLASS_CUSTOM),
        SelectorPattern("pseudo_class", PAT_PSEUDO_CLASS),
        SelectorPattern("pseudo_element", PAT_PSEUDO_ELEMENT),
        SelectorPattern("amp", PAT_AMP),
        SelectorPattern("at_rule", PAT_AT_RULE),
        SelectorPattern("id", PAT_ID),
        SelectorPattern("class", PAT_CLASS),
        SelectorPattern("tag", PAT_TAG),
        SelectorPattern("attribute", PAT_ATTR),
        SelectorPattern("combine", PAT_COMBINE)
    )

    def __init__(
        self,
        selector: str,
        custom: dict[str, str | ct.SelectorList] | None = None,
        flags: int = 0
    ) -> None:
        """Initialize."""

        self.pattern = selector.replace('\x00', '\ufffd')
        self.flags = flags
        self.debug = self.flags & util.DEBUG
        self.custom = {} if custom is None else custom

    def parse_attribute_selector(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Create attribute selector from the returned regex match."""

        inverse = False
        op = m.group('cmp')
        case = util.lower(m.group('case')) if m.group('case') else None
        ns = css_unescape(m.group('attr_ns')[:-1]) if m.group('attr_ns') else ''
        attr = css_unescape(m.group('attr_name'))
        is_type = False
        pattern2 = None
        value = ''

        if case:
            flags = (re.I if case == 'i' else 0) | re.DOTALL
        elif util.lower(attr) == 'type':
            flags = re.I | re.DOTALL
            is_type = True
        else:
            flags = re.DOTALL

        if op:
            if m.group('value').startswith(('"', "'")):
                value = css_unescape(m.group('value')[1:-1], True)
            else:
                value = css_unescape(m.group('value'))

        if not op:
            # Attribute name
            pattern = None
        elif op.startswith('^'):
            # Value start with
            pattern = re.compile(r'^%s.*' % re.escape(value), flags)
        elif op.startswith('$'):
            # Value ends with
            pattern = re.compile(r'.*?%s$' % re.escape(value), flags)
        elif op.startswith('*'):
            # Value contains
            pattern = re.compile(r'.*?%s.*' % re.escape(value), flags)
        elif op.startswith('~'):
            # Value contains word within space separated list
            # `~=` should match nothing if it is empty or contains whitespace,
            # so if either of these cases is present, use `[^\s\S]` which cannot be matched.
            value = r'[^\s\S]' if not value or RE_WS.search(value) else re.escape(value)
            pattern = re.compile(r'.*?(?:(?<=^)|(?<=[ \t\r\n\f]))%s(?=(?:[ \t\r\n\f]|$)).*' % value, flags)
        elif op.startswith('|'):
            # Value starts with word in dash separated list
            pattern = re.compile(r'^%s(?:-.*)?$' % re.escape(value), flags)
        else:
            # Value matches
            pattern = re.compile(r'^%s$' % re.escape(value), flags)
            if op.startswith('!'):
                # Equivalent to `:not([attr=value])`
                inverse = True
        if is_type and pattern:
            pattern2 = re.compile(pattern.pattern)

        # Append the attribute selector
        sel_attr = ct.SelectorAttribute(attr, ns, pattern, pattern2)
        if inverse:
            # If we are using `!=`, we need to nest the pattern under a `:not()`.
            sub_sel = _Selector()
            sub_sel.attributes.append(sel_attr)
            not_list = ct.SelectorList([sub_sel.freeze()], True, False)
            sel.selectors.append(not_list)
        else:
            sel.attributes.append(sel_attr)

        has_selector = True
        return has_selector

    def parse_tag_pattern(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Parse tag pattern from regex match."""

        prefix = css_unescape(m.group('tag_ns')[:-1]) if m.group('tag_ns') else None
        tag = css_unescape(m.group('tag_name'))
        sel.tag = ct.SelectorTag(tag, prefix)
        has_selector = True
        return has_selector

    def parse_pseudo_class_custom(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """
        Parse custom pseudo class alias.

        Compile custom selectors as we need them. When compiling a custom selector,
        set it to `None` in the dictionary so we can avoid an infinite loop.
        """

        pseudo = util.lower(css_unescape(m.group('name')))
        selector = self.custom.get(pseudo)
        if selector is None:
            raise SelectorSyntaxError(
                f"Undefined custom selector '{pseudo}' found at position {m.end(0)}",
                self.pattern,
                m.end(0)
            )

        if not isinstance(selector, ct.SelectorList):
            del self.custom[pseudo]
            selector = CSSParser(
                selector, custom=self.custom, flags=self.flags
            ).process_selectors(flags=FLG_PSEUDO)
            self.custom[pseudo] = selector

        sel.selectors.append(selector)
        has_selector = True
        return has_selector

    def parse_pseudo_class(
        self,
        sel: _Selector,
        m: Match[str],
        has_selector: bool,
        iselector: Iterator[tuple[str, Match[str]]],
        is_html: bool
    ) -> tuple[bool, bool]:
        """Parse pseudo class."""

        complex_pseudo = False
        pseudo = util.lower(css_unescape(m.group('name')))
        if m.group('open'):
            complex_pseudo = True
        if complex_pseudo and pseudo in PSEUDO_COMPLEX:
            has_selector = self.parse_pseudo_open(sel, pseudo, has_selector, iselector, m.end(0))
        elif not complex_pseudo and pseudo in PSEUDO_SIMPLE:
            if pseudo == ':root':
                sel.flags |= ct.SEL_ROOT
            elif pseudo == ':defined':
                sel.flags |= ct.SEL_DEFINED
                is_html = True
            elif pseudo == ':scope':
                sel.flags |= ct.SEL_SCOPE
            elif pseudo == ':empty':
                sel.flags |= ct.SEL_EMPTY
            elif pseudo in (':link', ':any-link'):
                sel.selectors.append(CSS_LINK)
            elif pseudo == ':checked':
                sel.selectors.append(CSS_CHECKED)
            elif pseudo == ':default':
                sel.selectors.append(CSS_DEFAULT)
            elif pseudo == ':indeterminate':
                sel.selectors.append(CSS_INDETERMINATE)
            elif pseudo == ":disabled":
                sel.selectors.append(CSS_DISABLED)
            elif pseudo == ":enabled":
                sel.selectors.append(CSS_ENABLED)
            elif pseudo == ":required":
                sel.selectors.append(CSS_REQUIRED)
            elif pseudo == ":muted":
                sel.selectors.append(CSS_MUTED)
            elif pseudo == ":open":
                sel.selectors.append(CSS_OPEN)
            elif pseudo == ":optional":
                sel.selectors.append(CSS_OPTIONAL)
            elif pseudo == ":read-only":
                sel.selectors.append(CSS_READ_ONLY)
            elif pseudo == ":read-write":
                sel.selectors.append(CSS_READ_WRITE)
            elif pseudo == ":in-range":
                sel.selectors.append(CSS_IN_RANGE)
            elif pseudo == ":out-of-range":
                sel.selectors.append(CSS_OUT_OF_RANGE)
            elif pseudo == ":placeholder-shown":
                sel.selectors.append(CSS_PLACEHOLDER_SHOWN)
            elif pseudo == ':first-child':
                sel.nth.append(ct.SelectorNth(1, False, 0, False, False, ct.SelectorList()))
            elif pseudo == ':last-child':
                sel.nth.append(ct.SelectorNth(1, False, 0, False, True, ct.SelectorList()))
            elif pseudo == ':first-of-type':
                sel.nth.append(ct.SelectorNth(1, False, 0, True, False, ct.SelectorList()))
            elif pseudo == ':last-of-type':
                sel.nth.append(ct.SelectorNth(1, False, 0, True, True, ct.SelectorList()))
            elif pseudo == ':only-child':
                sel.nth.extend(
                    [
                        ct.SelectorNth(1, False, 0, False, False, ct.SelectorList()),
                        ct.SelectorNth(1, False, 0, False, True, ct.SelectorList())
                    ]
                )
            elif pseudo == ':only-of-type':
                sel.nth.extend(
                    [
                        ct.SelectorNth(1, False, 0, True, False, ct.SelectorList()),
                        ct.SelectorNth(1, False, 0, True, True, ct.SelectorList())
                    ]
                )
            has_selector = True
        elif complex_pseudo and pseudo in PSEUDO_COMPLEX_NO_MATCH:
            self.parse_selectors(iselector, m.end(0), FLG_PSEUDO | FLG_OPEN)
            sel.no_match = True
            has_selector = True
        elif not complex_pseudo and pseudo in PSEUDO_SIMPLE_NO_MATCH:
            sel.no_match = True
            has_selector = True
        elif pseudo in PSEUDO_SUPPORTED:
            raise SelectorSyntaxError(
                f"Invalid syntax for pseudo class '{pseudo}'",
                self.pattern,
                m.start(0)
            )
        else:
            raise SelectorSyntaxError(
                f"'{pseudo}' was detected as a pseudo-class and is either unsupported or invalid. "
                "If the syntax was not intended to be recognized as a pseudo-class, please escape the colon.",
                self.pattern,
                m.start(0)
            )

        return has_selector, is_html

    def parse_pseudo_nth(
        self,
        sel: _Selector,
        m: Match[str],
        has_selector: bool,
        iselector: Iterator[tuple[str, Match[str]]]
    ) -> bool:
        """Parse `nth` pseudo."""

        mdict = m.groupdict()
        if mdict.get('pseudo_nth_child'):
            postfix = '_child'
        else:
            postfix = '_type'
        mdict['name'] = util.lower(css_unescape(mdict['name']))
        content = util.lower(mdict.get('nth' + postfix))
        if content == 'even':
            # 2n
            s1 = 2
            s2 = 0
            var = True
        elif content == 'odd':
            # 2n+1
            s1 = 2
            s2 = 1
            var = True
        else:
            nth_parts = cast(Match[str], RE_NTH.match(content))
            _s1 = '-' if nth_parts.group('s1') and nth_parts.group('s1') == '-' else ''
            a = nth_parts.group('a')
            var = a.endswith('n')
            if a.startswith('n'):
                _s1 += '1'
            elif var:
                _s1 += a[:-1]
            else:
                _s1 += a
            _s2 = '-' if nth_parts.group('s2') and nth_parts.group('s2') == '-' else ''
            if nth_parts.group('b'):
                _s2 += nth_parts.group('b')
            else:
                _s2 = '0'
            s1 = int(_s1, 10)
            s2 = int(_s2, 10)

        pseudo_sel = mdict['name']
        if postfix == '_child':
            if m.group('of'):
                # Parse the rest of `of S`.
                nth_sel = self.parse_selectors(iselector, m.end(0), FLG_PSEUDO | FLG_OPEN)
            else:
                # Use default `*|*` for `of S`.
                nth_sel = CSS_NTH_OF_S_DEFAULT
            if pseudo_sel == ':nth-child':
                sel.nth.append(ct.SelectorNth(s1, var, s2, False, False, nth_sel))
            elif pseudo_sel == ':nth-last-child':
                sel.nth.append(ct.SelectorNth(s1, var, s2, False, True, nth_sel))
        else:
            if pseudo_sel == ':nth-of-type':
                sel.nth.append(ct.SelectorNth(s1, var, s2, True, False, ct.SelectorList()))
            elif pseudo_sel == ':nth-last-of-type':
                sel.nth.append(ct.SelectorNth(s1, var, s2, True, True, ct.SelectorList()))
        has_selector = True
        return has_selector

    def parse_pseudo_open(
        self,
        sel: _Selector,
        name: str,
        has_selector: bool,
        iselector: Iterator[tuple[str, Match[str]]],
        index: int
    ) -> bool:
        """Parse pseudo with opening bracket."""

        flags = FLG_PSEUDO | FLG_OPEN
        if name == ':not':
            flags |= FLG_NOT
        elif name == ':has':
            flags |= FLG_RELATIVE
        elif name in (':where', ':is'):
            flags |= FLG_FORGIVE

        sel.selectors.append(self.parse_selectors(iselector, index, flags))
        has_selector = True

        return has_selector

    def parse_has_combinator(
        self,
        sel: _Selector,
        m: Match[str],
        has_selector: bool,
        selectors: list[_Selector],
        rel_type: str,
        index: int
    ) -> tuple[bool, _Selector, str]:
        """Parse combinator tokens."""

        combinator = m.group('relation').strip()
        if not combinator:
            combinator = WS_COMBINATOR
        if combinator == COMMA_COMBINATOR:
            sel.rel_type = rel_type
            selectors[-1].relations.append(sel)
            rel_type = ":" + WS_COMBINATOR
            selectors.append(_Selector())
        else:
            if has_selector:
                # End the current selector and associate the leading combinator with this selector.
                sel.rel_type = rel_type
                selectors[-1].relations.append(sel)
            elif rel_type[1:] != WS_COMBINATOR:
                # It's impossible to have two whitespace combinators after each other as the patterns
                # will gobble up trailing whitespace. It is also impossible to have a whitespace
                # combinator after any other kind for the same reason. But we could have
                # multiple non-whitespace combinators. So if the current combinator is not a whitespace,
                # then we've hit the multiple combinator case, so we should fail.
                raise SelectorSyntaxError(
                    f'The multiple combinators at position {index}',
                    self.pattern,
                    index
                )

            # Set the leading combinator for the next selector.
            rel_type = ':' + combinator

        sel = _Selector()
        has_selector = False
        return has_selector, sel, rel_type

    def parse_combinator(
        self,
        sel: _Selector,
        m: Match[str],
        has_selector: bool,
        selectors: list[_Selector],
        relations: list[_Selector],
        is_pseudo: bool,
        is_forgive: bool,
        index: int
    ) -> tuple[bool, _Selector]:
        """Parse combinator tokens."""

        combinator = m.group('relation').strip()
        if not combinator:
            combinator = WS_COMBINATOR
        if not has_selector:
            if not is_forgive or combinator != COMMA_COMBINATOR:
                raise SelectorSyntaxError(
                    f"The combinator '{combinator}' at position {index}, must have a selector before it",
                    self.pattern,
                    index
                )

            # If we are in a forgiving pseudo class, just make the selector a "no match"
            if combinator == COMMA_COMBINATOR:
                sel.no_match = True
                del relations[:]
                selectors.append(sel)
        else:
            if combinator == COMMA_COMBINATOR:
                if not sel.tag and not is_pseudo:
                    # Implied `*`
                    sel.tag = ct.SelectorTag('*', None)
                sel.relations.extend(relations)
                selectors.append(sel)
                del relations[:]
            else:
                sel.relations.extend(relations)
                sel.rel_type = combinator
                del relations[:]
                relations.append(sel)

        sel = _Selector()
        has_selector = False

        return has_selector, sel

    def parse_class_id(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Parse HTML classes and ids."""

        selector = m.group(0)
        if selector.startswith('.'):
            sel.classes.append(css_unescape(selector[1:]))
        else:
            sel.ids.append(css_unescape(selector[1:]))
        has_selector = True
        return has_selector

    def parse_pseudo_contains(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Parse contains."""

        pseudo = util.lower(css_unescape(m.group('name')))
        if pseudo == ":contains":
            warnings.warn(  # noqa: B028
                "The pseudo class ':contains' is deprecated, ':-soup-contains' should be used moving forward.",
                FutureWarning
            )
        contains_own = pseudo == ":-soup-contains-own"
        values = css_unescape(m.group('values'))
        patterns = []
        for token in RE_VALUES.finditer(values):
            if token.group('split'):
                continue
            value = token.group('value')
            if value.startswith(("'", '"')):
                value = css_unescape(value[1:-1], True)
            else:
                value = css_unescape(value)
            patterns.append(value)
        sel.contains.append(ct.SelectorContains(patterns, contains_own))
        has_selector = True
        return has_selector

    def parse_pseudo_lang(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Parse pseudo language."""

        values = m.group('values')
        patterns = []
        for token in RE_VALUES.finditer(values):
            if token.group('split'):
                continue
            value = token.group('value')
            if value.startswith(('"', "'")):
                value = css_unescape(value[1:-1], True)
            else:
                value = css_unescape(value)

            patterns.append(value)

        sel.lang.append(ct.SelectorLang(patterns))
        has_selector = True

        return has_selector

    def parse_pseudo_dir(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
        """Parse pseudo direction."""

        value = ct.SEL_DIR_LTR if util.lower(m.group('dir')) == 'ltr' else ct.SEL_DIR_RTL
        sel.flags |= value
        has_selector = True
        return has_selector

    def parse_selectors(
        self,
        iselector: Iterator[tuple[str, Match[str]]],
        index: int = 0,
        flags: int = 0
    ) -> ct.SelectorList:
        """Parse selectors."""

        # Initialize important variables
        sel = _Selector()
        selectors = []
        has_selector = False
        closed = False
        relations = []  # type: list[_Selector]
        rel_type = ":" + WS_COMBINATOR

        # Setup various flags
        is_open = bool(flags & FLG_OPEN)
        is_pseudo = bool(flags & FLG_PSEUDO)
        is_relative = bool(flags & FLG_RELATIVE)
        is_not = bool(flags & FLG_NOT)
        is_html = bool(flags & FLG_HTML)
        is_default = bool(flags & FLG_DEFAULT)
        is_indeterminate = bool(flags & FLG_INDETERMINATE)
        is_in_range = bool(flags & FLG_IN_RANGE)
        is_out_of_range = bool(flags & FLG_OUT_OF_RANGE)
        is_placeholder_shown = bool(flags & FLG_PLACEHOLDER_SHOWN)
        is_forgive = bool(flags & FLG_FORGIVE)

        # Print out useful debug stuff
        if self.debug:  # pragma: no cover
            if is_pseudo:
                print('    is_pseudo: True')
            if is_open:
                print('    is_open: True')
            if is_relative:
                print('    is_relative: True')
            if is_not:
                print('    is_not: True')
            if is_html:
                print('    is_html: True')
            if is_default:
                print('    is_default: True')
            if is_indeterminate:
                print('    is_indeterminate: True')
            if is_in_range:
                print('    is_in_range: True')
            if is_out_of_range:
                print('    is_out_of_range: True')
            if is_placeholder_shown:
                print('    is_placeholder_shown: True')
            if is_forgive:
                print('    is_forgive: True')

        # The algorithm for relative selectors require an initial selector in the selector list
        if is_relative:
            selectors.append(_Selector())

        try:
            while True:
                key, m = next(iselector)

                # Handle parts
                if key == "at_rule":
                    raise NotImplementedError(f"At-rules found at position {m.start(0)}")
                elif key == "amp":
                    sel.flags |= ct.SEL_SCOPE
                    has_selector = True
                elif key == 'pseudo_class_custom':
                    has_selector = self.parse_pseudo_class_custom(sel, m, has_selector)
                elif key == 'pseudo_class':
                    has_selector, is_html = self.parse_pseudo_class(sel, m, has_selector, iselector, is_html)
                elif key == 'pseudo_element':
                    raise NotImplementedError(f"Pseudo-element found at position {m.start(0)}")
                elif key == 'pseudo_contains':
                    has_selector = self.parse_pseudo_contains(sel, m, has_selector)
                elif key in ('pseudo_nth_type', 'pseudo_nth_child'):
                    has_selector = self.parse_pseudo_nth(sel, m, has_selector, iselector)
                elif key == 'pseudo_lang':
                    has_selector = self.parse_pseudo_lang(sel, m, has_selector)
                elif key == 'pseudo_dir':
                    has_selector = self.parse_pseudo_dir(sel, m, has_selector)
                    # Currently only supports HTML
                    is_html = True
                elif key == 'pseudo_close':
                    if not has_selector:
                        if not is_forgive:
                            raise SelectorSyntaxError(
                                f"Expected a selector at position {m.start(0)}",
                                self.pattern,
                                m.start(0)
                            )
                        sel.no_match = True
                    if is_open:
                        closed = True
                        break
                    else:
                        raise SelectorSyntaxError(
                            f"Unmatched pseudo-class close at position {m.start(0)}",
                            self.pattern,
                            m.start(0)
                        )
                elif key == 'combine':
                    if is_relative:
                        has_selector, sel, rel_type = self.parse_has_combinator(
                            sel, m, has_selector, selectors, rel_type, index
                        )
                    else:
                        has_selector, sel = self.parse_combinator(
                            sel, m, has_selector, selectors, relations, is_pseudo, is_forgive, index
                        )
                elif key == 'attribute':
                    has_selector = self.parse_attribute_selector(sel, m, has_selector)
                elif key == 'tag':
                    if has_selector:
                        raise SelectorSyntaxError(
                            f"Tag name found at position {m.start(0)} instead of at the start",
                            self.pattern,
                            m.start(0)
                        )
                    has_selector = self.parse_tag_pattern(sel, m, has_selector)
                elif key in ('class', 'id'):
                    has_selector = self.parse_class_id(sel, m, has_selector)

                index = m.end(0)
        except StopIteration:
            pass

        # Handle selectors that are not closed
        if is_open and not closed:
            raise SelectorSyntaxError(
                f"Unclosed pseudo-class at position {index}",
                self.pattern,
                index
            )

        # Cleanup completed selector piece
        if has_selector:
            if not sel.tag and not is_pseudo:
                # Implied `*`
                sel.tag = ct.SelectorTag('*', None)
            if is_relative:
                sel.rel_type = rel_type
                selectors[-1].relations.append(sel)
            else:
                sel.relations.extend(relations)
                del relations[:]
                selectors.append(sel)

        # Forgive empty slots in pseudo-classes that have lists (and are forgiving)
        elif is_forgive and (not selectors or not relations):
            # Handle normal pseudo-classes with empty slots like `:is()` etc.
            sel.no_match = True
            del relations[:]
            selectors.append(sel)
            has_selector = True

        if not has_selector:
            # We will always need to finish a selector when `:has()` is used as it leads with combining.
            # May apply to others as well.
            raise SelectorSyntaxError(
                f'Expected a selector at position {index}',
                self.pattern,
                index
            )

        # Some patterns require additional logic, such as default. We try to make these the
        # last pattern, and append the appropriate flag to that selector which communicates
        # to the matcher what additional logic is required.
        if is_default:
            selectors[-1].flags = ct.SEL_DEFAULT
        if is_indeterminate:
            selectors[-1].flags = ct.SEL_INDETERMINATE
        if is_in_range:
            selectors[-1].flags = ct.SEL_IN_RANGE
        if is_out_of_range:
            selectors[-1].flags = ct.SEL_OUT_OF_RANGE
        if is_placeholder_shown:
            selectors[-1].flags = ct.SEL_PLACEHOLDER_SHOWN

        # Return selector list
        return ct.SelectorList([s.freeze() for s in selectors], is_not, is_html)

    def selector_iter(self, pattern: str) -> Iterator[tuple[str, Match[str]]]:
        """Iterate selector tokens."""

        # Ignore whitespace and comments at start and end of pattern
        m = RE_WS_BEGIN.search(pattern)
        index = m.end(0) if m else 0
        m = RE_WS_END.search(pattern)
        end = (m.start(0) - 1) if m else (len(pattern) - 1)

        if self.debug:  # pragma: no cover
            print(f'## PARSING: {pattern!r}')
        while index <= end:
            m = None
            for v in self.css_tokens:
                m = v.match(pattern, index, self.flags)
                if m:
                    name = v.get_name()
                    if self.debug:  # pragma: no cover
                        print(f"TOKEN: '{name}' --> {m.group(0)!r} at position {m.start(0)}")
                    index = m.end(0)
                    yield name, m
                    break
            if m is None:
                c = pattern[index]
                # If the character represents the start of one of the known selector types,
                # throw an exception mentioning that the known selector type is in error;
                # otherwise, report the invalid character.
                if c == '[':
                    msg = f"Malformed attribute selector at position {index}"
                elif c == '.':
                    msg = f"Malformed class selector at position {index}"
                elif c == '#':
                    msg = f"Malformed id selector at position {index}"
                elif c == ':':
                    msg = f"Malformed pseudo-class selector at position {index}"
                else:
                    msg = f"Invalid character {c!r} position {index}"
                raise SelectorSyntaxError(msg, self.pattern, index)
        if self.debug:  # pragma: no cover
            print('## END PARSING')

    def process_selectors(self, index: int = 0, flags: int = 0) -> ct.SelectorList:
        """Process selectors."""

        return self.parse_selectors(self.selector_iter(self.pattern), index, flags)


# Precompile CSS selector lists for pseudo-classes (additional logic may be required beyond the pattern)
# A few patterns are order dependent as they use patterns previous compiled.

# CSS pattern for `:link` and `:any-link`
CSS_LINK = CSSParser(
    'html|*:is(a, area)[href]'
).process_selectors(flags=FLG_PSEUDO | FLG_HTML)
# CSS pattern for `:checked`
CSS_CHECKED = CSSParser(
    '''
    html|*:is(input[type=checkbox], input[type=radio])[checked], html|option[selected]
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_HTML)
# CSS pattern for `:default` (must compile CSS_CHECKED first)
CSS_DEFAULT = CSSParser(
    '''
    :checked,

    /*
    This pattern must be at the end.
    Special logic is applied to the last selector.
    */
    html|form html|*:is(button, input)[type="submit"]
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_HTML | FLG_DEFAULT)
# CSS pattern for `:indeterminate`
CSS_INDETERMINATE = CSSParser(
    '''
    html|input[type="checkbox"][indeterminate],
    html|input[type="radio"]:is(:not([name]), [name=""]):not([checked]),
    html|progress:not([value]),

    /*
    This pattern must be at the end.
    Special logic is applied to the last selector.
    */
    html|input[type="radio"][name]:not([name='']):not([checked])
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_HTML | FLG_INDETERMINATE)
# CSS pattern for `:disabled`
CSS_DISABLED = CSSParser(
    '''
    html|*:is(input:not([type=hidden]), button, select, textarea, fieldset, optgroup, option, fieldset)[disabled],
    html|optgroup[disabled] > html|option,
    html|fieldset[disabled] > html|*:is(input:not([type=hidden]), button, select, textarea, fieldset),
    html|fieldset[disabled] >
        html|*:not(legend:nth-of-type(1)) html|*:is(input:not([type=hidden]), button, select, textarea, fieldset)
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_HTML)
# CSS pattern for `:enabled`
CSS_ENABLED = CSSParser(
    '''
    html|*:is(input:not([type=hidden]), button, select, textarea, fieldset, optgroup, option, fieldset):not(:disabled)
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_HTML)
# CSS pattern for `:required`
CSS_REQUIRED = CSSParser(
    'html|*:is(input, textarea, select)[required]'
).process_selectors(flags=FLG_PSEUDO | FLG_HTML)
# CSS pattern for `:optional`
CSS_OPTIONAL = CSSParser(
    'html|*:is(input, textarea, select):not([required])'
).process_selectors(flags=FLG_PSEUDO | FLG_HTML)
# CSS pattern for `:placeholder-shown`
CSS_PLACEHOLDER_SHOWN = CSSParser(
    '''
    html|input:is(
        :not([type]),
        [type=""],
        [type=text],
        [type=search],
        [type=url],
        [type=tel],
        [type=email],
        [type=password],
        [type=number]
    )[placeholder]:not([placeholder='']):is(:not([value]), [value=""]),
    html|textarea[placeholder]:not([placeholder=''])
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_HTML | FLG_PLACEHOLDER_SHOWN)
# CSS pattern default for `:nth-child` "of S" feature
CSS_NTH_OF_S_DEFAULT = CSSParser(
    '*|*'
).process_selectors(flags=FLG_PSEUDO)
# CSS pattern for `:read-write` (CSS_DISABLED must be compiled first)
CSS_READ_WRITE = CSSParser(
    '''
    html|*:is(
        textarea,
        input:is(
            :not([type]),
            [type=""],
            [type=text],
            [type=search],
            [type=url],
            [type=tel],
            [type=email],
            [type=number],
            [type=password],
            [type=date],
            [type=datetime-local],
            [type=month],
            [type=time],
            [type=week]
        )
    ):not([readonly], :disabled),
    html|*:is([contenteditable=""], [contenteditable="true" i])
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_HTML)
# CSS pattern for `:read-only`
CSS_READ_ONLY = CSSParser(
    '''
    html|*:not(:read-write)
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_HTML)
# CSS pattern for `:in-range`
CSS_IN_RANGE = CSSParser(
    '''
    html|input:is(
        [type="date"],
        [type="month"],
        [type="week"],
        [type="time"],
        [type="datetime-local"],
        [type="number"],
        [type="range"]
    ):is(
        [min],
        [max]
    )
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_IN_RANGE | FLG_HTML)
# CSS pattern for `:out-of-range`
CSS_OUT_OF_RANGE = CSSParser(
    '''
    html|input:is(
        [type="date"],
        [type="month"],
        [type="week"],
        [type="time"],
        [type="datetime-local"],
        [type="number"],
        [type="range"]
    ):is(
        [min],
        [max]
    )
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_OUT_OF_RANGE | FLG_HTML)

# CSS pattern for :open
CSS_OPEN = CSSParser(
    '''
    html|*:is(details, dialog)[open]
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_HTML)


# CSS pattern for :muted
CSS_MUTED = CSSParser(
    '''
    html|*:is(video, audio)[muted]
    '''
).process_selectors(flags=FLG_PSEUDO | FLG_HTML)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\emulation.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Emulation
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page


@dataclass
class SafeAreaInsets:
    #: Overrides safe-area-inset-top.
    top: typing.Optional[int] = None

    #: Overrides safe-area-max-inset-top.
    top_max: typing.Optional[int] = None

    #: Overrides safe-area-inset-left.
    left: typing.Optional[int] = None

    #: Overrides safe-area-max-inset-left.
    left_max: typing.Optional[int] = None

    #: Overrides safe-area-inset-bottom.
    bottom: typing.Optional[int] = None

    #: Overrides safe-area-max-inset-bottom.
    bottom_max: typing.Optional[int] = None

    #: Overrides safe-area-inset-right.
    right: typing.Optional[int] = None

    #: Overrides safe-area-max-inset-right.
    right_max: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        if self.top is not None:
            json['top'] = self.top
        if self.top_max is not None:
            json['topMax'] = self.top_max
        if self.left is not None:
            json['left'] = self.left
        if self.left_max is not None:
            json['leftMax'] = self.left_max
        if self.bottom is not None:
            json['bottom'] = self.bottom
        if self.bottom_max is not None:
            json['bottomMax'] = self.bottom_max
        if self.right is not None:
            json['right'] = self.right
        if self.right_max is not None:
            json['rightMax'] = self.right_max
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            top=int(json['top']) if 'top' in json else None,
            top_max=int(json['topMax']) if 'topMax' in json else None,
            left=int(json['left']) if 'left' in json else None,
            left_max=int(json['leftMax']) if 'leftMax' in json else None,
            bottom=int(json['bottom']) if 'bottom' in json else None,
            bottom_max=int(json['bottomMax']) if 'bottomMax' in json else None,
            right=int(json['right']) if 'right' in json else None,
            right_max=int(json['rightMax']) if 'rightMax' in json else None,
        )


@dataclass
class ScreenOrientation:
    '''
    Screen orientation.
    '''
    #: Orientation type.
    type_: str

    #: Orientation angle.
    angle: int

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['angle'] = self.angle
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            angle=int(json['angle']),
        )


@dataclass
class DisplayFeature:
    #: Orientation of a display feature in relation to screen
    orientation: str

    #: The offset from the screen origin in either the x (for vertical
    #: orientation) or y (for horizontal orientation) direction.
    offset: int

    #: A display feature may mask content such that it is not physically
    #: displayed - this length along with the offset describes this area.
    #: A display feature that only splits content will have a 0 mask_length.
    mask_length: int

    def to_json(self):
        json = dict()
        json['orientation'] = self.orientation
        json['offset'] = self.offset
        json['maskLength'] = self.mask_length
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            orientation=str(json['orientation']),
            offset=int(json['offset']),
            mask_length=int(json['maskLength']),
        )


@dataclass
class DevicePosture:
    #: Current posture of the device
    type_: str

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
        )


@dataclass
class MediaFeature:
    name: str

    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


class VirtualTimePolicy(enum.Enum):
    '''
    advance: If the scheduler runs out of immediate work, the virtual time base may fast forward to
    allow the next delayed task (if any) to run; pause: The virtual time base may not advance;
    pauseIfNetworkFetchesPending: The virtual time base may not advance if there are any pending
    resource fetches.
    '''
    ADVANCE = "advance"
    PAUSE = "pause"
    PAUSE_IF_NETWORK_FETCHES_PENDING = "pauseIfNetworkFetchesPending"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class UserAgentBrandVersion:
    '''
    Used to specify User Agent Client Hints to emulate. See https://wicg.github.io/ua-client-hints
    '''
    brand: str

    version: str

    def to_json(self):
        json = dict()
        json['brand'] = self.brand
        json['version'] = self.version
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            brand=str(json['brand']),
            version=str(json['version']),
        )


@dataclass
class UserAgentMetadata:
    '''
    Used to specify User Agent Client Hints to emulate. See https://wicg.github.io/ua-client-hints
    Missing optional values will be filled in by the target with what it would normally use.
    '''
    platform: str

    platform_version: str

    architecture: str

    model: str

    mobile: bool

    #: Brands appearing in Sec-CH-UA.
    brands: typing.Optional[typing.List[UserAgentBrandVersion]] = None

    #: Brands appearing in Sec-CH-UA-Full-Version-List.
    full_version_list: typing.Optional[typing.List[UserAgentBrandVersion]] = None

    full_version: typing.Optional[str] = None

    bitness: typing.Optional[str] = None

    wow64: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['platform'] = self.platform
        json['platformVersion'] = self.platform_version
        json['architecture'] = self.architecture
        json['model'] = self.model
        json['mobile'] = self.mobile
        if self.brands is not None:
            json['brands'] = [i.to_json() for i in self.brands]
        if self.full_version_list is not None:
            json['fullVersionList'] = [i.to_json() for i in self.full_version_list]
        if self.full_version is not None:
            json['fullVersion'] = self.full_version
        if self.bitness is not None:
            json['bitness'] = self.bitness
        if self.wow64 is not None:
            json['wow64'] = self.wow64
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            platform=str(json['platform']),
            platform_version=str(json['platformVersion']),
            architecture=str(json['architecture']),
            model=str(json['model']),
            mobile=bool(json['mobile']),
            brands=[UserAgentBrandVersion.from_json(i) for i in json['brands']] if 'brands' in json else None,
            full_version_list=[UserAgentBrandVersion.from_json(i) for i in json['fullVersionList']] if 'fullVersionList' in json else None,
            full_version=str(json['fullVersion']) if 'fullVersion' in json else None,
            bitness=str(json['bitness']) if 'bitness' in json else None,
            wow64=bool(json['wow64']) if 'wow64' in json else None,
        )


class SensorType(enum.Enum):
    '''
    Used to specify sensor types to emulate.
    See https://w3c.github.io/sensors/#automation for more information.
    '''
    ABSOLUTE_ORIENTATION = "absolute-orientation"
    ACCELEROMETER = "accelerometer"
    AMBIENT_LIGHT = "ambient-light"
    GRAVITY = "gravity"
    GYROSCOPE = "gyroscope"
    LINEAR_ACCELERATION = "linear-acceleration"
    MAGNETOMETER = "magnetometer"
    RELATIVE_ORIENTATION = "relative-orientation"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SensorMetadata:
    available: typing.Optional[bool] = None

    minimum_frequency: typing.Optional[float] = None

    maximum_frequency: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        if self.available is not None:
            json['available'] = self.available
        if self.minimum_frequency is not None:
            json['minimumFrequency'] = self.minimum_frequency
        if self.maximum_frequency is not None:
            json['maximumFrequency'] = self.maximum_frequency
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            available=bool(json['available']) if 'available' in json else None,
            minimum_frequency=float(json['minimumFrequency']) if 'minimumFrequency' in json else None,
            maximum_frequency=float(json['maximumFrequency']) if 'maximumFrequency' in json else None,
        )


@dataclass
class SensorReadingSingle:
    value: float

    def to_json(self):
        json = dict()
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=float(json['value']),
        )


@dataclass
class SensorReadingXYZ:
    x: float

    y: float

    z: float

    def to_json(self):
        json = dict()
        json['x'] = self.x
        json['y'] = self.y
        json['z'] = self.z
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            x=float(json['x']),
            y=float(json['y']),
            z=float(json['z']),
        )


@dataclass
class SensorReadingQuaternion:
    x: float

    y: float

    z: float

    w: float

    def to_json(self):
        json = dict()
        json['x'] = self.x
        json['y'] = self.y
        json['z'] = self.z
        json['w'] = self.w
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            x=float(json['x']),
            y=float(json['y']),
            z=float(json['z']),
            w=float(json['w']),
        )


@dataclass
class SensorReading:
    single: typing.Optional[SensorReadingSingle] = None

    xyz: typing.Optional[SensorReadingXYZ] = None

    quaternion: typing.Optional[SensorReadingQuaternion] = None

    def to_json(self):
        json = dict()
        if self.single is not None:
            json['single'] = self.single.to_json()
        if self.xyz is not None:
            json['xyz'] = self.xyz.to_json()
        if self.quaternion is not None:
            json['quaternion'] = self.quaternion.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            single=SensorReadingSingle.from_json(json['single']) if 'single' in json else None,
            xyz=SensorReadingXYZ.from_json(json['xyz']) if 'xyz' in json else None,
            quaternion=SensorReadingQuaternion.from_json(json['quaternion']) if 'quaternion' in json else None,
        )


class PressureSource(enum.Enum):
    CPU = "cpu"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PressureState(enum.Enum):
    NOMINAL = "nominal"
    FAIR = "fair"
    SERIOUS = "serious"
    CRITICAL = "critical"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PressureMetadata:
    available: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        if self.available is not None:
            json['available'] = self.available
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            available=bool(json['available']) if 'available' in json else None,
        )


class DisabledImageType(enum.Enum):
    '''
    Enum of image types that can be disabled.
    '''
    AVIF = "avif"
    WEBP = "webp"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def can_emulate() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Tells whether emulation is supported.

    :returns: True if emulation is supported.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.canEmulate',
    }
    json = yield cmd_dict
    return bool(json['result'])


def clear_device_metrics_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears the overridden device metrics.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.clearDeviceMetricsOverride',
    }
    json = yield cmd_dict


def clear_geolocation_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears the overridden Geolocation Position and Error.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.clearGeolocationOverride',
    }
    json = yield cmd_dict


def reset_page_scale_factor() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Requests that page scale factor is reset to initial values.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.resetPageScaleFactor',
    }
    json = yield cmd_dict


def set_focus_emulation_enabled(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables or disables simulating a focused and active page.

    **EXPERIMENTAL**

    :param enabled: Whether to enable to disable focus emulation.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setFocusEmulationEnabled',
        'params': params,
    }
    json = yield cmd_dict


def set_auto_dark_mode_override(
        enabled: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Automatically render all web contents using a dark theme.

    **EXPERIMENTAL**

    :param enabled: *(Optional)* Whether to enable or disable automatic dark mode. If not specified, any existing override will be cleared.
    '''
    params: T_JSON_DICT = dict()
    if enabled is not None:
        params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setAutoDarkModeOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_cpu_throttling_rate(
        rate: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables CPU throttling to emulate slow CPUs.

    :param rate: Throttling rate as a slowdown factor (1 is no throttle, 2 is 2x slowdown, etc).
    '''
    params: T_JSON_DICT = dict()
    params['rate'] = rate
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setCPUThrottlingRate',
        'params': params,
    }
    json = yield cmd_dict


def set_default_background_color_override(
        color: typing.Optional[dom.RGBA] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets or clears an override of the default background color of the frame. This override is used
    if the content does not specify one.

    :param color: *(Optional)* RGBA of the default background color. If not specified, any existing override will be cleared.
    '''
    params: T_JSON_DICT = dict()
    if color is not None:
        params['color'] = color.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setDefaultBackgroundColorOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_safe_area_insets_override(
        insets: SafeAreaInsets
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides the values for env(safe-area-inset-*) and env(safe-area-max-inset-*). Unset values will cause the
    respective variables to be undefined, even if previously overridden.

    **EXPERIMENTAL**

    :param insets:
    '''
    params: T_JSON_DICT = dict()
    params['insets'] = insets.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setSafeAreaInsetsOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_device_metrics_override(
        width: int,
        height: int,
        device_scale_factor: float,
        mobile: bool,
        scale: typing.Optional[float] = None,
        screen_width: typing.Optional[int] = None,
        screen_height: typing.Optional[int] = None,
        position_x: typing.Optional[int] = None,
        position_y: typing.Optional[int] = None,
        dont_set_visible_size: typing.Optional[bool] = None,
        screen_orientation: typing.Optional[ScreenOrientation] = None,
        viewport: typing.Optional[page.Viewport] = None,
        display_feature: typing.Optional[DisplayFeature] = None,
        device_posture: typing.Optional[DevicePosture] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides the values of device screen dimensions (window.screen.width, window.screen.height,
    window.innerWidth, window.innerHeight, and "device-width"/"device-height"-related CSS media
    query results).

    :param width: Overriding width value in pixels (minimum 0, maximum 10000000). 0 disables the override.
    :param height: Overriding height value in pixels (minimum 0, maximum 10000000). 0 disables the override.
    :param device_scale_factor: Overriding device scale factor value. 0 disables the override.
    :param mobile: Whether to emulate mobile device. This includes viewport meta tag, overlay scrollbars, text autosizing and more.
    :param scale: **(EXPERIMENTAL)** *(Optional)* Scale to apply to resulting view image.
    :param screen_width: **(EXPERIMENTAL)** *(Optional)* Overriding screen width value in pixels (minimum 0, maximum 10000000).
    :param screen_height: **(EXPERIMENTAL)** *(Optional)* Overriding screen height value in pixels (minimum 0, maximum 10000000).
    :param position_x: **(EXPERIMENTAL)** *(Optional)* Overriding view X position on screen in pixels (minimum 0, maximum 10000000).
    :param position_y: **(EXPERIMENTAL)** *(Optional)* Overriding view Y position on screen in pixels (minimum 0, maximum 10000000).
    :param dont_set_visible_size: **(EXPERIMENTAL)** *(Optional)* Do not set visible view size, rely upon explicit setVisibleSize call.
    :param screen_orientation: *(Optional)* Screen orientation override.
    :param viewport: **(EXPERIMENTAL)** *(Optional)* If set, the visible area of the page will be overridden to this viewport. This viewport change is not observed by the page, e.g. viewport-relative elements do not change positions.
    :param display_feature: **(EXPERIMENTAL)** *(Optional)* If set, the display feature of a multi-segment screen. If not set, multi-segment support is turned-off. Deprecated, use Emulation.setDisplayFeaturesOverride.
    :param device_posture: **(EXPERIMENTAL)** *(Optional)* If set, the posture of a foldable device. If not set the posture is set to continuous. Deprecated, use Emulation.setDevicePostureOverride.
    '''
    params: T_JSON_DICT = dict()
    params['width'] = width
    params['height'] = height
    params['deviceScaleFactor'] = device_scale_factor
    params['mobile'] = mobile
    if scale is not None:
        params['scale'] = scale
    if screen_width is not None:
        params['screenWidth'] = screen_width
    if screen_height is not None:
        params['screenHeight'] = screen_height
    if position_x is not None:
        params['positionX'] = position_x
    if position_y is not None:
        params['positionY'] = position_y
    if dont_set_visible_size is not None:
        params['dontSetVisibleSize'] = dont_set_visible_size
    if screen_orientation is not None:
        params['screenOrientation'] = screen_orientation.to_json()
    if viewport is not None:
        params['viewport'] = viewport.to_json()
    if display_feature is not None:
        params['displayFeature'] = display_feature.to_json()
    if device_posture is not None:
        params['devicePosture'] = device_posture.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setDeviceMetricsOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_device_posture_override(
        posture: DevicePosture
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Start reporting the given posture value to the Device Posture API.
    This override can also be set in setDeviceMetricsOverride().

    **EXPERIMENTAL**

    :param posture:
    '''
    params: T_JSON_DICT = dict()
    params['posture'] = posture.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setDevicePostureOverride',
        'params': params,
    }
    json = yield cmd_dict


def clear_device_posture_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears a device posture override set with either setDeviceMetricsOverride()
    or setDevicePostureOverride() and starts using posture information from the
    platform again.
    Does nothing if no override is set.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.clearDevicePostureOverride',
    }
    json = yield cmd_dict


def set_display_features_override(
        features: typing.List[DisplayFeature]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Start using the given display features to pupulate the Viewport Segments API.
    This override can also be set in setDeviceMetricsOverride().

    **EXPERIMENTAL**

    :param features:
    '''
    params: T_JSON_DICT = dict()
    params['features'] = [i.to_json() for i in features]
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setDisplayFeaturesOverride',
        'params': params,
    }
    json = yield cmd_dict


def clear_display_features_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears the display features override set with either setDeviceMetricsOverride()
    or setDisplayFeaturesOverride() and starts using display features from the
    platform again.
    Does nothing if no override is set.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.clearDisplayFeaturesOverride',
    }
    json = yield cmd_dict


def set_scrollbars_hidden(
        hidden: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param hidden: Whether scrollbars should be always hidden.
    '''
    params: T_JSON_DICT = dict()
    params['hidden'] = hidden
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setScrollbarsHidden',
        'params': params,
    }
    json = yield cmd_dict


def set_document_cookie_disabled(
        disabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param disabled: Whether document.coookie API should be disabled.
    '''
    params: T_JSON_DICT = dict()
    params['disabled'] = disabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setDocumentCookieDisabled',
        'params': params,
    }
    json = yield cmd_dict


def set_emit_touch_events_for_mouse(
        enabled: bool,
        configuration: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param enabled: Whether touch emulation based on mouse input should be enabled.
    :param configuration: *(Optional)* Touch/gesture events configuration. Default: current platform.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    if configuration is not None:
        params['configuration'] = configuration
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setEmitTouchEventsForMouse',
        'params': params,
    }
    json = yield cmd_dict


def set_emulated_media(
        media: typing.Optional[str] = None,
        features: typing.Optional[typing.List[MediaFeature]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Emulates the given media type or media feature for CSS media queries.

    :param media: *(Optional)* Media type to emulate. Empty string disables the override.
    :param features: *(Optional)* Media features to emulate.
    '''
    params: T_JSON_DICT = dict()
    if media is not None:
        params['media'] = media
    if features is not None:
        params['features'] = [i.to_json() for i in features]
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setEmulatedMedia',
        'params': params,
    }
    json = yield cmd_dict


def set_emulated_vision_deficiency(
        type_: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Emulates the given vision deficiency.

    :param type_: Vision deficiency to emulate. Order: best-effort emulations come first, followed by any physiologically accurate emulations for medically recognized color vision deficiencies.
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setEmulatedVisionDeficiency',
        'params': params,
    }
    json = yield cmd_dict


def set_geolocation_override(
        latitude: typing.Optional[float] = None,
        longitude: typing.Optional[float] = None,
        accuracy: typing.Optional[float] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides the Geolocation Position or Error. Omitting any of the parameters emulates position
    unavailable.

    :param latitude: *(Optional)* Mock latitude
    :param longitude: *(Optional)* Mock longitude
    :param accuracy: *(Optional)* Mock accuracy
    '''
    params: T_JSON_DICT = dict()
    if latitude is not None:
        params['latitude'] = latitude
    if longitude is not None:
        params['longitude'] = longitude
    if accuracy is not None:
        params['accuracy'] = accuracy
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setGeolocationOverride',
        'params': params,
    }
    json = yield cmd_dict


def get_overridden_sensor_information(
        type_: SensorType
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''


    **EXPERIMENTAL**

    :param type_:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.getOverriddenSensorInformation',
        'params': params,
    }
    json = yield cmd_dict
    return float(json['requestedSamplingFrequency'])


def set_sensor_override_enabled(
        enabled: bool,
        type_: SensorType,
        metadata: typing.Optional[SensorMetadata] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides a platform sensor of a given type. If ``enabled`` is true, calls to
    Sensor.start() will use a virtual sensor as backend rather than fetching
    data from a real hardware sensor. Otherwise, existing virtual
    sensor-backend Sensor objects will fire an error event and new calls to
    Sensor.start() will attempt to use a real sensor instead.

    **EXPERIMENTAL**

    :param enabled:
    :param type_:
    :param metadata: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    params['type'] = type_.to_json()
    if metadata is not None:
        params['metadata'] = metadata.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setSensorOverrideEnabled',
        'params': params,
    }
    json = yield cmd_dict


def set_sensor_override_readings(
        type_: SensorType,
        reading: SensorReading
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Updates the sensor readings reported by a sensor type previously overridden
    by setSensorOverrideEnabled.

    **EXPERIMENTAL**

    :param type_:
    :param reading:
    '''
    params: T_JSON_DICT = dict()
    params['type'] = type_.to_json()
    params['reading'] = reading.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setSensorOverrideReadings',
        'params': params,
    }
    json = yield cmd_dict


def set_pressure_source_override_enabled(
        enabled: bool,
        source: PressureSource,
        metadata: typing.Optional[PressureMetadata] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides a pressure source of a given type, as used by the Compute
    Pressure API, so that updates to PressureObserver.observe() are provided
    via setPressureStateOverride instead of being retrieved from
    platform-provided telemetry data.

    **EXPERIMENTAL**

    :param enabled:
    :param source:
    :param metadata: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    params['source'] = source.to_json()
    if metadata is not None:
        params['metadata'] = metadata.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setPressureSourceOverrideEnabled',
        'params': params,
    }
    json = yield cmd_dict


def set_pressure_state_override(
        source: PressureSource,
        state: PressureState
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Provides a given pressure state that will be processed and eventually be
    delivered to PressureObserver users. ``source`` must have been previously
    overridden by setPressureSourceOverrideEnabled.

    **EXPERIMENTAL**

    :param source:
    :param state:
    '''
    params: T_JSON_DICT = dict()
    params['source'] = source.to_json()
    params['state'] = state.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setPressureStateOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_idle_override(
        is_user_active: bool,
        is_screen_unlocked: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides the Idle state.

    :param is_user_active: Mock isUserActive
    :param is_screen_unlocked: Mock isScreenUnlocked
    '''
    params: T_JSON_DICT = dict()
    params['isUserActive'] = is_user_active
    params['isScreenUnlocked'] = is_screen_unlocked
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setIdleOverride',
        'params': params,
    }
    json = yield cmd_dict


def clear_idle_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears Idle state overrides.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.clearIdleOverride',
    }
    json = yield cmd_dict


def set_navigator_overrides(
        platform: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides value returned by the javascript navigator object.

    **EXPERIMENTAL**

    :param platform: The platform navigator.platform should return.
    '''
    params: T_JSON_DICT = dict()
    params['platform'] = platform
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setNavigatorOverrides',
        'params': params,
    }
    json = yield cmd_dict


def set_page_scale_factor(
        page_scale_factor: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets a specified page scale factor.

    **EXPERIMENTAL**

    :param page_scale_factor: Page scale factor.
    '''
    params: T_JSON_DICT = dict()
    params['pageScaleFactor'] = page_scale_factor
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setPageScaleFactor',
        'params': params,
    }
    json = yield cmd_dict


def set_script_execution_disabled(
        value: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Switches script execution in the page.

    :param value: Whether script execution should be disabled in the page.
    '''
    params: T_JSON_DICT = dict()
    params['value'] = value
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setScriptExecutionDisabled',
        'params': params,
    }
    json = yield cmd_dict


def set_touch_emulation_enabled(
        enabled: bool,
        max_touch_points: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables touch on platforms which do not support them.

    :param enabled: Whether the touch event emulation should be enabled.
    :param max_touch_points: *(Optional)* Maximum touch points supported. Defaults to one.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    if max_touch_points is not None:
        params['maxTouchPoints'] = max_touch_points
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setTouchEmulationEnabled',
        'params': params,
    }
    json = yield cmd_dict


def set_virtual_time_policy(
        policy: VirtualTimePolicy,
        budget: typing.Optional[float] = None,
        max_virtual_time_task_starvation_count: typing.Optional[int] = None,
        initial_virtual_time: typing.Optional[network.TimeSinceEpoch] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''
    Turns on virtual time for all frames (replacing real-time with a synthetic time source) and sets
    the current virtual time policy.  Note this supersedes any previous time budget.

    **EXPERIMENTAL**

    :param policy:
    :param budget: *(Optional)* If set, after this many virtual milliseconds have elapsed virtual time will be paused and a virtualTimeBudgetExpired event is sent.
    :param max_virtual_time_task_starvation_count: *(Optional)* If set this specifies the maximum number of tasks that can be run before virtual is forced forwards to prevent deadlock.
    :param initial_virtual_time: *(Optional)* If set, base::Time::Now will be overridden to initially return this value.
    :returns: Absolute timestamp at which virtual time was first enabled (up time in milliseconds).
    '''
    params: T_JSON_DICT = dict()
    params['policy'] = policy.to_json()
    if budget is not None:
        params['budget'] = budget
    if max_virtual_time_task_starvation_count is not None:
        params['maxVirtualTimeTaskStarvationCount'] = max_virtual_time_task_starvation_count
    if initial_virtual_time is not None:
        params['initialVirtualTime'] = initial_virtual_time.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setVirtualTimePolicy',
        'params': params,
    }
    json = yield cmd_dict
    return float(json['virtualTimeTicksBase'])


def set_locale_override(
        locale: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides default host system locale with the specified one.

    **EXPERIMENTAL**

    :param locale: *(Optional)* ICU style C locale (e.g. "en_US"). If not specified or empty, disables the override and restores default host system locale.
    '''
    params: T_JSON_DICT = dict()
    if locale is not None:
        params['locale'] = locale
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setLocaleOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_timezone_override(
        timezone_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Overrides default host system timezone with the specified one.

    :param timezone_id: The timezone identifier. List of supported timezones: https://source.chromium.org/chromium/chromium/deps/icu.git/+/faee8bc70570192d82d2978a71e2a615788597d1:source/data/misc/metaZones.txt If empty, disables the override and restores default host system timezone.
    '''
    params: T_JSON_DICT = dict()
    params['timezoneId'] = timezone_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setTimezoneOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_visible_size(
        width: int,
        height: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resizes the frame/viewport of the page. Note that this does not affect the frame's container
    (e.g. browser window). Can be used to produce screenshots of the specified size. Not supported
    on Android.

    **EXPERIMENTAL**

    :param width: Frame width (DIP).
    :param height: Frame height (DIP).
    '''
    params: T_JSON_DICT = dict()
    params['width'] = width
    params['height'] = height
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setVisibleSize',
        'params': params,
    }
    json = yield cmd_dict


def set_disabled_image_types(
        image_types: typing.List[DisabledImageType]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param image_types: Image types to disable.
    '''
    params: T_JSON_DICT = dict()
    params['imageTypes'] = [i.to_json() for i in image_types]
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setDisabledImageTypes',
        'params': params,
    }
    json = yield cmd_dict


def set_hardware_concurrency_override(
        hardware_concurrency: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param hardware_concurrency: Hardware concurrency to report
    '''
    params: T_JSON_DICT = dict()
    params['hardwareConcurrency'] = hardware_concurrency
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setHardwareConcurrencyOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_user_agent_override(
        user_agent: str,
        accept_language: typing.Optional[str] = None,
        platform: typing.Optional[str] = None,
        user_agent_metadata: typing.Optional[UserAgentMetadata] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Allows overriding user agent with the given string.
    ``userAgentMetadata`` must be set for Client Hint headers to be sent.

    :param user_agent: User agent to use.
    :param accept_language: *(Optional)* Browser language to emulate.
    :param platform: *(Optional)* The platform navigator.platform should return.
    :param user_agent_metadata: **(EXPERIMENTAL)** *(Optional)* To be sent in Sec-CH-UA-* headers and returned in navigator.userAgentData
    '''
    params: T_JSON_DICT = dict()
    params['userAgent'] = user_agent
    if accept_language is not None:
        params['acceptLanguage'] = accept_language
    if platform is not None:
        params['platform'] = platform
    if user_agent_metadata is not None:
        params['userAgentMetadata'] = user_agent_metadata.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setUserAgentOverride',
        'params': params,
    }
    json = yield cmd_dict


def set_automation_override(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Allows overriding the automation flag.

    **EXPERIMENTAL**

    :param enabled: Whether the override should be enabled.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Emulation.setAutomationOverride',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Emulation.virtualTimeBudgetExpired')
@dataclass
class VirtualTimeBudgetExpired:
    '''
    **EXPERIMENTAL**

    Notification sent after the virtual time budget for the current VirtualTimePolicy has run out.
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> VirtualTimeBudgetExpired:
        return cls(

        )