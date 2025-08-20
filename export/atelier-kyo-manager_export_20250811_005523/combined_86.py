
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\ctx_mp_python.py ===
#from ctx_base import StandardBaseContext

from .libmp.backend import basestring, exec_

from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
    round_floor, round_ceiling, dps_to_prec, round_nearest, prec_to_dps,
    ComplexResult, to_pickable, from_pickable, normalize,
    from_int, from_float, from_npfloat, from_Decimal, from_str, to_int, to_float, to_str,
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

from . import rational
from . import function_docs

new = object.__new__

class mpnumeric(object):
    """Base class for mpf and mpc."""
    __slots__ = []
    def __new__(cls, val):
        raise NotImplementedError

class _mpf(mpnumeric):
    """
    An mpf instance holds a real-valued floating-point number. mpf:s
    work analogously to Python floats, but support arbitrary-precision
    arithmetic.
    """
    __slots__ = ['_mpf_']

    def __new__(cls, val=fzero, **kwargs):
        """A new mpf can be created from a Python float, an int, a
        or a decimal string representing a number in floating-point
        format."""
        prec, rounding = cls.context._prec_rounding
        if kwargs:
            prec = kwargs.get('prec', prec)
            if 'dps' in kwargs:
                prec = dps_to_prec(kwargs['dps'])
            rounding = kwargs.get('rounding', rounding)
        if type(val) is cls:
            sign, man, exp, bc = val._mpf_
            if (not man) and exp:
                return val
            v = new(cls)
            v._mpf_ = normalize(sign, man, exp, bc, prec, rounding)
            return v
        elif type(val) is tuple:
            if len(val) == 2:
                v = new(cls)
                v._mpf_ = from_man_exp(val[0], val[1], prec, rounding)
                return v
            if len(val) == 4:
                if val not in (finf, fninf, fnan):
                    sign, man, exp, bc = val
                    val = normalize(sign, MPZ(man), exp, bc, prec, rounding)
                v = new(cls)
                v._mpf_ = val
                return v
            raise ValueError
        else:
            v = new(cls)
            v._mpf_ = mpf_pos(cls.mpf_convert_arg(val, prec, rounding), prec, rounding)
            return v

    @classmethod
    def mpf_convert_arg(cls, x, prec, rounding):
        if isinstance(x, int_types): return from_int(x)
        if isinstance(x, float): return from_float(x)
        if isinstance(x, basestring): return from_str(x, prec, rounding)
        if isinstance(x, cls.context.constant): return x.func(prec, rounding)
        if hasattr(x, '_mpf_'): return x._mpf_
        if hasattr(x, '_mpmath_'):
            t = cls.context.convert(x._mpmath_(prec, rounding))
            if hasattr(t, '_mpf_'):
                return t._mpf_
        if hasattr(x, '_mpi_'):
            a, b = x._mpi_
            if a == b:
                return a
            raise ValueError("can only create mpf from zero-width interval")
        raise TypeError("cannot create mpf from " + repr(x))

    @classmethod
    def mpf_convert_rhs(cls, x):
        if isinstance(x, int_types): return from_int(x)
        if isinstance(x, float): return from_float(x)
        if isinstance(x, complex_types): return cls.context.mpc(x)
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return from_rational(p, q, cls.context.prec)
        if hasattr(x, '_mpf_'): return x._mpf_
        if hasattr(x, '_mpmath_'):
            t = cls.context.convert(x._mpmath_(*cls.context._prec_rounding))
            if hasattr(t, '_mpf_'):
                return t._mpf_
            return t
        return NotImplemented

    @classmethod
    def mpf_convert_lhs(cls, x):
        x = cls.mpf_convert_rhs(x)
        if type(x) is tuple:
            return cls.context.make_mpf(x)
        return x

    man_exp = property(lambda self: self._mpf_[1:3])
    man = property(lambda self: self._mpf_[1])
    exp = property(lambda self: self._mpf_[2])
    bc = property(lambda self: self._mpf_[3])

    real = property(lambda self: self)
    imag = property(lambda self: self.context.zero)

    conjugate = lambda self: self

    def __getstate__(self): return to_pickable(self._mpf_)
    def __setstate__(self, val): self._mpf_ = from_pickable(val)

    def __repr__(s):
        if s.context.pretty:
            return str(s)
        return "mpf('%s')" % to_str(s._mpf_, s.context._repr_digits)

    def __str__(s): return to_str(s._mpf_, s.context._str_digits)
    def __hash__(s): return mpf_hash(s._mpf_)
    def __int__(s): return int(to_int(s._mpf_))
    def __long__(s): return long(to_int(s._mpf_))
    def __float__(s): return to_float(s._mpf_, rnd=s.context._prec_rounding[1])
    def __complex__(s): return complex(float(s))
    def __nonzero__(s): return s._mpf_ != fzero

    __bool__ = __nonzero__

    def __abs__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_abs(s._mpf_, prec, rounding)
        return v

    def __pos__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_pos(s._mpf_, prec, rounding)
        return v

    def __neg__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_neg(s._mpf_, prec, rounding)
        return v

    def _cmp(s, t, func):
        if hasattr(t, '_mpf_'):
            t = t._mpf_
        else:
            t = s.mpf_convert_rhs(t)
            if t is NotImplemented:
                return t
        return func(s._mpf_, t)

    def __cmp__(s, t): return s._cmp(t, mpf_cmp)
    def __lt__(s, t): return s._cmp(t, mpf_lt)
    def __gt__(s, t): return s._cmp(t, mpf_gt)
    def __le__(s, t): return s._cmp(t, mpf_le)
    def __ge__(s, t): return s._cmp(t, mpf_ge)

    def __ne__(s, t):
        v = s.__eq__(t)
        if v is NotImplemented:
            return v
        return not v

    def __rsub__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if type(t) in int_types:
            v = new(cls)
            v._mpf_ = mpf_sub(from_int(t), s._mpf_, prec, rounding)
            return v
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t - s

    def __rdiv__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if isinstance(t, int_types):
            v = new(cls)
            v._mpf_ = mpf_rdiv_int(t, s._mpf_, prec, rounding)
            return v
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t / s

    def __rpow__(s, t):
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t ** s

    def __rmod__(s, t):
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t % s

    def sqrt(s):
        return s.context.sqrt(s)

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.context.almosteq(s, t, rel_eps, abs_eps)

    def to_fixed(self, prec):
        return to_fixed(self._mpf_, prec)

    def __round__(self, *args):
        return round(float(self), *args)

mpf_binary_op = """
def %NAME%(self, other):
    mpf, new, (prec, rounding) = self._ctxdata
    sval = self._mpf_
    if hasattr(other, '_mpf_'):
        tval = other._mpf_
        %WITH_MPF%
    ttype = type(other)
    if ttype in int_types:
        %WITH_INT%
    elif ttype is float:
        tval = from_float(other)
        %WITH_MPF%
    elif hasattr(other, '_mpc_'):
        tval = other._mpc_
        mpc = type(other)
        %WITH_MPC%
    elif ttype is complex:
        tval = from_float(other.real), from_float(other.imag)
        mpc = self.context.mpc
        %WITH_MPC%
    if isinstance(other, mpnumeric):
        return NotImplemented
    try:
        other = mpf.context.convert(other, strings=False)
    except TypeError:
        return NotImplemented
    return self.%NAME%(other)
"""

return_mpf = "; obj = new(mpf); obj._mpf_ = val; return obj"
return_mpc = "; obj = new(mpc); obj._mpc_ = val; return obj"

mpf_pow_same = """
        try:
            val = mpf_pow(sval, tval, prec, rounding) %s
        except ComplexResult:
            if mpf.context.trap_complex:
                raise
            mpc = mpf.context.mpc
            val = mpc_pow((sval, fzero), (tval, fzero), prec, rounding) %s
""" % (return_mpf, return_mpc)

def binary_op(name, with_mpf='', with_int='', with_mpc=''):
    code = mpf_binary_op
    code = code.replace("%WITH_INT%", with_int)
    code = code.replace("%WITH_MPC%", with_mpc)
    code = code.replace("%WITH_MPF%", with_mpf)
    code = code.replace("%NAME%", name)
    np = {}
    exec_(code, globals(), np)
    return np[name]

_mpf.__eq__ = binary_op('__eq__',
    'return mpf_eq(sval, tval)',
    'return mpf_eq(sval, from_int(other))',
    'return (tval[1] == fzero) and mpf_eq(tval[0], sval)')

_mpf.__add__ = binary_op('__add__',
    'val = mpf_add(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_add(sval, from_int(other), prec, rounding)' + return_mpf,
    'val = mpc_add_mpf(tval, sval, prec, rounding)' + return_mpc)

_mpf.__sub__ = binary_op('__sub__',
    'val = mpf_sub(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_sub(sval, from_int(other), prec, rounding)' + return_mpf,
    'val = mpc_sub((sval, fzero), tval, prec, rounding)' + return_mpc)

_mpf.__mul__ = binary_op('__mul__',
    'val = mpf_mul(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_mul_int(sval, other, prec, rounding)' + return_mpf,
    'val = mpc_mul_mpf(tval, sval, prec, rounding)' + return_mpc)

_mpf.__div__ = binary_op('__div__',
    'val = mpf_div(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_div(sval, from_int(other), prec, rounding)' + return_mpf,
    'val = mpc_mpf_div(sval, tval, prec, rounding)' + return_mpc)

_mpf.__mod__ = binary_op('__mod__',
    'val = mpf_mod(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_mod(sval, from_int(other), prec, rounding)' + return_mpf,
    'raise NotImplementedError("complex modulo")')

_mpf.__pow__ = binary_op('__pow__',
    mpf_pow_same,
    'val = mpf_pow_int(sval, other, prec, rounding)' + return_mpf,
    'val = mpc_pow((sval, fzero), tval, prec, rounding)' + return_mpc)

_mpf.__radd__ = _mpf.__add__
_mpf.__rmul__ = _mpf.__mul__
_mpf.__truediv__ = _mpf.__div__
_mpf.__rtruediv__ = _mpf.__rdiv__


class _constant(_mpf):
    """Represents a mathematical constant with dynamic precision.
    When printed or used in an arithmetic operation, a constant
    is converted to a regular mpf at the working precision. A
    regular mpf can also be obtained using the operation +x."""

    def __new__(cls, func, name, docname=''):
        a = object.__new__(cls)
        a.name = name
        a.func = func
        a.__doc__ = getattr(function_docs, docname, '')
        return a

    def __call__(self, prec=None, dps=None, rounding=None):
        prec2, rounding2 = self.context._prec_rounding
        if not prec: prec = prec2
        if not rounding: rounding = rounding2
        if dps: prec = dps_to_prec(dps)
        return self.context.make_mpf(self.func(prec, rounding))

    @property
    def _mpf_(self):
        prec, rounding = self.context._prec_rounding
        return self.func(prec, rounding)

    def __repr__(self):
        return "<%s: %s~>" % (self.name, self.context.nstr(self(dps=15)))


class _mpc(mpnumeric):
    """
    An mpc represents a complex number using a pair of mpf:s (one
    for the real part and another for the imaginary part.) The mpc
    class behaves fairly similarly to Python's complex type.
    """

    __slots__ = ['_mpc_']

    def __new__(cls, real=0, imag=0):
        s = object.__new__(cls)
        if isinstance(real, complex_types):
            real, imag = real.real, real.imag
        elif hasattr(real, '_mpc_'):
            s._mpc_ = real._mpc_
            return s
        real = cls.context.mpf(real)
        imag = cls.context.mpf(imag)
        s._mpc_ = (real._mpf_, imag._mpf_)
        return s

    real = property(lambda self: self.context.make_mpf(self._mpc_[0]))
    imag = property(lambda self: self.context.make_mpf(self._mpc_[1]))

    def __getstate__(self):
        return to_pickable(self._mpc_[0]), to_pickable(self._mpc_[1])

    def __setstate__(self, val):
        self._mpc_ = from_pickable(val[0]), from_pickable(val[1])

    def __repr__(s):
        if s.context.pretty:
            return str(s)
        r = repr(s.real)[4:-1]
        i = repr(s.imag)[4:-1]
        return "%s(real=%s, imag=%s)" % (type(s).__name__, r, i)

    def __str__(s):
        return "(%s)" % mpc_to_str(s._mpc_, s.context._str_digits)

    def __complex__(s):
        return mpc_to_complex(s._mpc_, rnd=s.context._prec_rounding[1])

    def __pos__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpc_ = mpc_pos(s._mpc_, prec, rounding)
        return v

    def __abs__(s):
        prec, rounding = s.context._prec_rounding
        v = new(s.context.mpf)
        v._mpf_ = mpc_abs(s._mpc_, prec, rounding)
        return v

    def __neg__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpc_ = mpc_neg(s._mpc_, prec, rounding)
        return v

    def conjugate(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpc_ = mpc_conjugate(s._mpc_, prec, rounding)
        return v

    def __nonzero__(s):
        return mpc_is_nonzero(s._mpc_)

    __bool__ = __nonzero__

    def __hash__(s):
        return mpc_hash(s._mpc_)

    @classmethod
    def mpc_convert_lhs(cls, x):
        try:
            y = cls.context.convert(x)
            return y
        except TypeError:
            return NotImplemented

    def __eq__(s, t):
        if not hasattr(t, '_mpc_'):
            if isinstance(t, str):
                return False
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
        return s.real == t.real and s.imag == t.imag

    def __ne__(s, t):
        b = s.__eq__(t)
        if b is NotImplemented:
            return b
        return not b

    def _compare(*args):
        raise TypeError("no ordering relation is defined for complex numbers")

    __gt__ = _compare
    __le__ = _compare
    __gt__ = _compare
    __ge__ = _compare

    def __add__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_add_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
        v = new(cls)
        v._mpc_ = mpc_add(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __sub__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_sub_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
        v = new(cls)
        v._mpc_ = mpc_sub(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __mul__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            if isinstance(t, int_types):
                v = new(cls)
                v._mpc_ = mpc_mul_int(s._mpc_, t, prec, rounding)
                return v
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_mul_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
            t = s.mpc_convert_lhs(t)
        v = new(cls)
        v._mpc_ = mpc_mul(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __div__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_div_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
        v = new(cls)
        v._mpc_ = mpc_div(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __pow__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if isinstance(t, int_types):
            v = new(cls)
            v._mpc_ = mpc_pow_int(s._mpc_, t, prec, rounding)
            return v
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        v = new(cls)
        if hasattr(t, '_mpf_'):
            v._mpc_ = mpc_pow_mpf(s._mpc_, t._mpf_, prec, rounding)
        else:
            v._mpc_ = mpc_pow(s._mpc_, t._mpc_, prec, rounding)
        return v

    __radd__ = __add__

    def __rsub__(s, t):
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t - s

    def __rmul__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if isinstance(t, int_types):
            v = new(cls)
            v._mpc_ = mpc_mul_int(s._mpc_, t, prec, rounding)
            return v
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t * s

    def __rdiv__(s, t):
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t / s

    def __rpow__(s, t):
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t ** s

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.context.almosteq(s, t, rel_eps, abs_eps)


complex_types = (complex, _mpc)


class PythonMPContext(object):

    def __init__(ctx):
        ctx._prec_rounding = [53, round_nearest]
        ctx.mpf = type('mpf', (_mpf,), {})
        ctx.mpc = type('mpc', (_mpc,), {})
        ctx.mpf._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.mpc._ctxdata = [ctx.mpc, new, ctx._prec_rounding]
        ctx.mpf.context = ctx
        ctx.mpc.context = ctx
        ctx.constant = type('constant', (_constant,), {})
        ctx.constant._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.constant.context = ctx

    def make_mpf(ctx, v):
        a = new(ctx.mpf)
        a._mpf_ = v
        return a

    def make_mpc(ctx, v):
        a = new(ctx.mpc)
        a._mpc_ = v
        return a

    def default(ctx):
        ctx._prec = ctx._prec_rounding[0] = 53
        ctx._dps = 15
        ctx.trap_complex = False

    def _set_prec(ctx, n):
        ctx._prec = ctx._prec_rounding[0] = max(1, int(n))
        ctx._dps = prec_to_dps(n)

    def _set_dps(ctx, n):
        ctx._prec = ctx._prec_rounding[0] = dps_to_prec(n)
        ctx._dps = max(1, int(n))

    prec = property(lambda ctx: ctx._prec, _set_prec)
    dps = property(lambda ctx: ctx._dps, _set_dps)

    def convert(ctx, x, strings=True):
        """
        Converts *x* to an ``mpf`` or ``mpc``. If *x* is of type ``mpf``,
        ``mpc``, ``int``, ``float``, ``complex``, the conversion
        will be performed losslessly.

        If *x* is a string, the result will be rounded to the present
        working precision. Strings representing fractions or complex
        numbers are permitted.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> mpmathify(3.5)
            mpf('3.5')
            >>> mpmathify('2.1')
            mpf('2.1000000000000001')
            >>> mpmathify('3/4')
            mpf('0.75')
            >>> mpmathify('2+3j')
            mpc(real='2.0', imag='3.0')

        """
        if type(x) in ctx.types: return x
        if isinstance(x, int_types): return ctx.make_mpf(from_int(x))
        if isinstance(x, float): return ctx.make_mpf(from_float(x))
        if isinstance(x, complex):
            return ctx.make_mpc((from_float(x.real), from_float(x.imag)))
        if type(x).__module__ == 'numpy': return ctx.npconvert(x)
        if isinstance(x, numbers.Rational): # e.g. Fraction
            try: x = rational.mpq(int(x.numerator), int(x.denominator))
            except: pass
        prec, rounding = ctx._prec_rounding
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return ctx.make_mpf(from_rational(p, q, prec))
        if strings and isinstance(x, basestring):
            try:
                _mpf_ = from_str(x, prec, rounding)
                return ctx.make_mpf(_mpf_)
            except ValueError:
                pass
        if hasattr(x, '_mpf_'): return ctx.make_mpf(x._mpf_)
        if hasattr(x, '_mpc_'): return ctx.make_mpc(x._mpc_)
        if hasattr(x, '_mpmath_'):
            return ctx.convert(x._mpmath_(prec, rounding))
        if type(x).__module__ == 'decimal':
            try: return ctx.make_mpf(from_Decimal(x, prec, rounding))
            except: pass
        return ctx._convert_fallback(x, strings)

    def npconvert(ctx, x):
        """
        Converts *x* to an ``mpf`` or ``mpc``. *x* should be a numpy
        scalar.
        """
        import numpy as np
        if isinstance(x, np.integer): return ctx.make_mpf(from_int(int(x)))
        if isinstance(x, np.floating): return ctx.make_mpf(from_npfloat(x))
        if isinstance(x, np.complexfloating):
            return ctx.make_mpc((from_npfloat(x.real), from_npfloat(x.imag)))
        raise TypeError("cannot create mpf from " + repr(x))

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

    def isinf(ctx, x):
        """
        Return *True* if the absolute value of *x* is infinite;
        otherwise return *False*::

            >>> from mpmath import *
            >>> isinf(inf)
            True
            >>> isinf(-inf)
            True
            >>> isinf(3)
            False
            >>> isinf(3+4j)
            False
            >>> isinf(mpc(3,inf))
            True
            >>> isinf(mpc(inf,3))
            True

        """
        if hasattr(x, "_mpf_"):
            return x._mpf_ in (finf, fninf)
        if hasattr(x, "_mpc_"):
            re, im = x._mpc_
            return re in (finf, fninf) or im in (finf, fninf)
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return False
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isinf(x)
        raise TypeError("isinf() needs a number as input")

    def isnormal(ctx, x):
        """
        Determine whether *x* is "normal" in the sense of floating-point
        representation; that is, return *False* if *x* is zero, an
        infinity or NaN; otherwise return *True*. By extension, a
        complex number *x* is considered "normal" if its magnitude is
        normal::

            >>> from mpmath import *
            >>> isnormal(3)
            True
            >>> isnormal(0)
            False
            >>> isnormal(inf); isnormal(-inf); isnormal(nan)
            False
            False
            False
            >>> isnormal(0+0j)
            False
            >>> isnormal(0+3j)
            True
            >>> isnormal(mpc(2,nan))
            False
        """
        if hasattr(x, "_mpf_"):
            return bool(x._mpf_[1])
        if hasattr(x, "_mpc_"):
            re, im = x._mpc_
            re_normal = bool(re[1])
            im_normal = bool(im[1])
            if re == fzero: return im_normal
            if im == fzero: return re_normal
            return re_normal and im_normal
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return bool(x)
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isnormal(x)
        raise TypeError("isnormal() needs a number as input")

    def isint(ctx, x, gaussian=False):
        """
        Return *True* if *x* is integer-valued; otherwise return
        *False*::

            >>> from mpmath import *
            >>> isint(3)
            True
            >>> isint(mpf(3))
            True
            >>> isint(3.2)
            False
            >>> isint(inf)
            False

        Optionally, Gaussian integers can be checked for::

            >>> isint(3+0j)
            True
            >>> isint(3+2j)
            False
            >>> isint(3+2j, gaussian=True)
            True

        """
        if isinstance(x, int_types):
            return True
        if hasattr(x, "_mpf_"):
            sign, man, exp, bc = xval = x._mpf_
            return bool((man and exp >= 0) or xval == fzero)
        if hasattr(x, "_mpc_"):
            re, im = x._mpc_
            rsign, rman, rexp, rbc = re
            isign, iman, iexp, ibc = im
            re_isint = (rman and rexp >= 0) or re == fzero
            if gaussian:
                im_isint = (iman and iexp >= 0) or im == fzero
                return re_isint and im_isint
            return re_isint and im == fzero
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return p % q == 0
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isint(x, gaussian)
        raise TypeError("isint() needs a number as input")

    def fsum(ctx, terms, absolute=False, squared=False):
        """
        Calculates a sum containing a finite number of terms (for infinite
        series, see :func:`~mpmath.nsum`). The terms will be converted to
        mpmath numbers. For len(terms) > 2, this function is generally
        faster and produces more accurate results than the builtin
        Python function :func:`sum`.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fsum([1, 2, 0.5, 7])
            mpf('10.5')

        With squared=True each term is squared, and with absolute=True
        the absolute value of each term is used.
        """
        prec, rnd = ctx._prec_rounding
        real = []
        imag = []
        for term in terms:
            reval = imval = 0
            if hasattr(term, "_mpf_"):
                reval = term._mpf_
            elif hasattr(term, "_mpc_"):
                reval, imval = term._mpc_
            else:
                term = ctx.convert(term)
                if hasattr(term, "_mpf_"):
                    reval = term._mpf_
                elif hasattr(term, "_mpc_"):
                    reval, imval = term._mpc_
                else:
                    raise NotImplementedError
            if imval:
                if squared:
                    if absolute:
                        real.append(mpf_mul(reval,reval))
                        real.append(mpf_mul(imval,imval))
                    else:
                        reval, imval = mpc_pow_int((reval,imval),2,prec+10)
                        real.append(reval)
                        imag.append(imval)
                elif absolute:
                    real.append(mpc_abs((reval,imval), prec))
                else:
                    real.append(reval)
                    imag.append(imval)
            else:
                if squared:
                    reval = mpf_mul(reval, reval)
                elif absolute:
                    reval = mpf_abs(reval)
                real.append(reval)
        s = mpf_sum(real, prec, rnd, absolute)
        if imag:
            s = ctx.make_mpc((s, mpf_sum(imag, prec, rnd)))
        else:
            s = ctx.make_mpf(s)
        return s

    def fdot(ctx, A, B=None, conjugate=False):
        r"""
        Computes the dot product of the iterables `A` and `B`,

        .. math ::

            \sum_{k=0} A_k B_k.

        Alternatively, :func:`~mpmath.fdot` accepts a single iterable of pairs.
        In other words, ``fdot(A,B)`` and ``fdot(zip(A,B))`` are equivalent.
        The elements are automatically converted to mpmath numbers.

        With ``conjugate=True``, the elements in the second vector
        will be conjugated:

        .. math ::

            \sum_{k=0} A_k \overline{B_k}

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> A = [2, 1.5, 3]
            >>> B = [1, -1, 2]
            >>> fdot(A, B)
            mpf('6.5')
            >>> list(zip(A, B))
            [(2, 1), (1.5, -1), (3, 2)]
            >>> fdot(_)
            mpf('6.5')
            >>> A = [2, 1.5, 3j]
            >>> B = [1+j, 3, -1-j]
            >>> fdot(A, B)
            mpc(real='9.5', imag='-1.0')
            >>> fdot(A, B, conjugate=True)
            mpc(real='3.5', imag='-5.0')

        """
        if B is not None:
            A = zip(A, B)
        prec, rnd = ctx._prec_rounding
        real = []
        imag = []
        hasattr_ = hasattr
        types = (ctx.mpf, ctx.mpc)
        for a, b in A:
            if type(a) not in types: a = ctx.convert(a)
            if type(b) not in types: b = ctx.convert(b)
            a_real = hasattr_(a, "_mpf_")
            b_real = hasattr_(b, "_mpf_")
            if a_real and b_real:
                real.append(mpf_mul(a._mpf_, b._mpf_))
                continue
            a_complex = hasattr_(a, "_mpc_")
            b_complex = hasattr_(b, "_mpc_")
            if a_real and b_complex:
                aval = a._mpf_
                bre, bim = b._mpc_
                if conjugate:
                    bim = mpf_neg(bim)
                real.append(mpf_mul(aval, bre))
                imag.append(mpf_mul(aval, bim))
            elif b_real and a_complex:
                are, aim = a._mpc_
                bval = b._mpf_
                real.append(mpf_mul(are, bval))
                imag.append(mpf_mul(aim, bval))
            elif a_complex and b_complex:
                #re, im = mpc_mul(a._mpc_, b._mpc_, prec+20)
                are, aim = a._mpc_
                bre, bim = b._mpc_
                if conjugate:
                    bim = mpf_neg(bim)
                real.append(mpf_mul(are, bre))
                real.append(mpf_neg(mpf_mul(aim, bim)))
                imag.append(mpf_mul(are, bim))
                imag.append(mpf_mul(aim, bre))
            else:
                raise NotImplementedError
        s = mpf_sum(real, prec, rnd)
        if imag:
            s = ctx.make_mpc((s, mpf_sum(imag, prec, rnd)))
        else:
            s = ctx.make_mpf(s)
        return s

    def _wrap_libmp_function(ctx, mpf_f, mpc_f=None, mpi_f=None, doc="<no doc>"):
        """
        Given a low-level mpf_ function, and optionally similar functions
        for mpc_ and mpi_, defines the function as a context method.

        It is assumed that the return type is the same as that of
        the input; the exception is that propagation from mpf to mpc is possible
        by raising ComplexResult.

        """
        def f(x, **kwargs):
            if type(x) not in ctx.types:
                x = ctx.convert(x)
            prec, rounding = ctx._prec_rounding
            if kwargs:
                prec = kwargs.get('prec', prec)
                if 'dps' in kwargs:
                    prec = dps_to_prec(kwargs['dps'])
                rounding = kwargs.get('rounding', rounding)
            if hasattr(x, '_mpf_'):
                try:
                    return ctx.make_mpf(mpf_f(x._mpf_, prec, rounding))
                except ComplexResult:
                    # Handle propagation to complex
                    if ctx.trap_complex:
                        raise
                    return ctx.make_mpc(mpc_f((x._mpf_, fzero), prec, rounding))
            elif hasattr(x, '_mpc_'):
                return ctx.make_mpc(mpc_f(x._mpc_, prec, rounding))
            raise NotImplementedError("%s of a %s" % (name, type(x)))
        name = mpf_f.__name__[4:]
        f.__doc__ = function_docs.__dict__.get(name, "Computes the %s of x" % doc)
        return f

    # Called by SpecialFunctions.__init__()
    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        if wrap:
            def f_wrapped(ctx, *args, **kwargs):
                convert = ctx.convert
                args = [convert(a) for a in args]
                prec = ctx.prec
                try:
                    ctx.prec += 10
                    retval = f(ctx, *args, **kwargs)
                finally:
                    ctx.prec = prec
                return +retval
        else:
            f_wrapped = f
        f_wrapped.__doc__ = function_docs.__dict__.get(name, f.__doc__)
        setattr(cls, name, f_wrapped)

    def _convert_param(ctx, x):
        if hasattr(x, "_mpc_"):
            v, im = x._mpc_
            if im != fzero:
                return x, 'C'
        elif hasattr(x, "_mpf_"):
            v = x._mpf_
        else:
            if type(x) in int_types:
                return int(x), 'Z'
            p = None
            if isinstance(x, tuple):
                p, q = x
            elif hasattr(x, '_mpq_'):
                p, q = x._mpq_
            elif isinstance(x, basestring) and '/' in x:
                p, q = x.split('/')
                p = int(p)
                q = int(q)
            if p is not None:
                if not p % q:
                    return p // q, 'Z'
                return ctx.mpq(p,q), 'Q'
            x = ctx.convert(x)
            if hasattr(x, "_mpc_"):
                v, im = x._mpc_
                if im != fzero:
                    return x, 'C'
            elif hasattr(x, "_mpf_"):
                v = x._mpf_
            else:
                return x, 'U'
        sign, man, exp, bc = v
        if man:
            if exp >= -4:
                if sign:
                    man = -man
                if exp >= 0:
                    return int(man) << exp, 'Z'
                if exp >= -4:
                    p, q = int(man), (1<<(-exp))
                    return ctx.mpq(p,q), 'Q'
            x = ctx.make_mpf(v)
            return x, 'R'
        elif not exp:
            return 0, 'Z'
        else:
            return x, 'U'

    def _mpf_mag(ctx, x):
        sign, man, exp, bc = x
        if man:
            return exp+bc
        if x == fzero:
            return ctx.ninf
        if x == finf or x == fninf:
            return ctx.inf
        return ctx.nan

    def mag(ctx, x):
        """
        Quick logarithmic magnitude estimate of a number. Returns an
        integer or infinity `m` such that `|x| <= 2^m`. It is not
        guaranteed that `m` is an optimal bound, but it will never
        be too large by more than 2 (and probably not more than 1).

        **Examples**

            >>> from mpmath import *
            >>> mp.pretty = True
            >>> mag(10), mag(10.0), mag(mpf(10)), int(ceil(log(10,2)))
            (4, 4, 4, 4)
            >>> mag(10j), mag(10+10j)
            (4, 5)
            >>> mag(0.01), int(ceil(log(0.01,2)))
            (-6, -6)
            >>> mag(0), mag(inf), mag(-inf), mag(nan)
            (-inf, +inf, +inf, nan)

        """
        if hasattr(x, "_mpf_"):
            return ctx._mpf_mag(x._mpf_)
        elif hasattr(x, "_mpc_"):
            r, i = x._mpc_
            if r == fzero:
                return ctx._mpf_mag(i)
            if i == fzero:
                return ctx._mpf_mag(r)
            return 1+max(ctx._mpf_mag(r), ctx._mpf_mag(i))
        elif isinstance(x, int_types):
            if x:
                return bitcount(abs(x))
            return ctx.ninf
        elif isinstance(x, rational.mpq):
            p, q = x._mpq_
            if p:
                return 1 + bitcount(abs(p)) - bitcount(q)
            return ctx.ninf
        else:
            x = ctx.convert(x)
            if hasattr(x, "_mpf_") or hasattr(x, "_mpc_"):
                return ctx.mag(x)
            else:
                raise TypeError("requires an mpf/mpc")


# Register with "numbers" ABC
#     We do not subclass, hence we do not use the @abstractmethod checks. While
#     this is less invasive it may turn out that we do not actually support
#     parts of the expected interfaces.  See
#     http://docs.python.org/2/library/numbers.html for list of abstract
#     methods.
try:
    import numbers
    numbers.Complex.register(_mpc)
    numbers.Real.register(_mpf)
except ImportError:
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\actuator.py ===
"""Implementations of actuators for linked force and torque application."""

from abc import ABC, abstractmethod

from sympy import S, sympify, exp, sign
from sympy.physics.mechanics.joint import PinJoint
from sympy.physics.mechanics.loads import Torque
from sympy.physics.mechanics.pathway import PathwayBase
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.vector import ReferenceFrame, Vector


__all__ = [
    'ActuatorBase',
    'ForceActuator',
    'LinearDamper',
    'LinearSpring',
    'TorqueActuator',
    'DuffingSpring',
    'CoulombKineticFriction',
]


class ActuatorBase(ABC):
    """Abstract base class for all actuator classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom actuator types through subclassing.

    """

    def __init__(self):
        """Initializer for ``ActuatorBase``."""
        pass

    @abstractmethod
    def to_loads(self):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        """
        pass

    def __repr__(self):
        """Default representation of an actuator."""
        return f'{self.__class__.__name__}()'


class ForceActuator(ActuatorBase):
    """Force-producing actuator.

    Explanation
    ===========

    A ``ForceActuator`` is an actuator that produces a (expansile) force along
    its length.

    A force actuator uses a pathway instance to determine the direction and
    number of forces that it applies to a system. Consider the simplest case
    where a ``LinearPathway`` instance is used. This pathway is made up of two
    points that can move relative to each other, and results in a pair of equal
    and opposite forces acting on the endpoints. If the positive time-varying
    Euclidean distance between the two points is defined, then the "extension
    velocity" is the time derivative of this distance. The extension velocity
    is positive when the two points are moving away from each other and
    negative when moving closer to each other. The direction for the force
    acting on either point is determined by constructing a unit vector directed
    from the other point to this point. This establishes a sign convention such
    that a positive force magnitude tends to push the points apart, this is the
    meaning of "expansile" in this context. The following diagram shows the
    positive force sense and the distance between the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct an actuator, an expression (or symbol) must be supplied to
    represent the force it can produce, alongside a pathway specifying its line
    of action. Let's also create a global reference frame and spatially fix one
    of the points in it while setting the other to be positioned such that it
    can freely move in the frame's x direction specified by the coordinate
    ``q``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (ForceActuator, LinearPathway,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> force = symbols('F')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> actuator = ForceActuator(force, linear_pathway)
    >>> actuator
    ForceActuator(F, LinearPathway(pA, pB))

    Parameters
    ==========

    force : Expr
        The scalar expression defining the (expansile) force that the actuator
        produces.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

    """

    def __init__(self, force, pathway):
        """Initializer for ``ForceActuator``.

        Parameters
        ==========

        force : Expr
            The scalar expression defining the (expansile) force that the
            actuator produces.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

        """
        self.force = force
        self.pathway = pathway

    @property
    def force(self):
        """The magnitude of the force produced by the actuator."""
        return self._force

    @force.setter
    def force(self, force):
        if hasattr(self, '_force'):
            msg = (
                f'Can\'t set attribute `force` to {repr(force)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._force = sympify(force, strict=True)

    @property
    def pathway(self):
        """The ``Pathway`` defining the actuator's line of action."""
        return self._pathway

    @pathway.setter
    def pathway(self, pathway):
        if hasattr(self, '_pathway'):
            msg = (
                f'Can\'t set attribute `pathway` to {repr(pathway)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(pathway, PathwayBase):
            msg = (
                f'Value {repr(pathway)} passed to `pathway` was of type '
                f'{type(pathway)}, must be {PathwayBase}.'
            )
            raise TypeError(msg)
        self._pathway = pathway

    def to_loads(self):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        Examples
        ========

        The below example shows how to generate the loads produced by a force
        actuator that follows a linear pathway. In this example we'll assume
        that the force actuator is being used to model a simple linear spring.
        First, create a linear pathway between two points separated by the
        coordinate ``q`` in the ``x`` direction of the global frame ``N``.

        >>> from sympy.physics.mechanics import (LinearPathway, Point,
        ...     ReferenceFrame)
        >>> from sympy.physics.vector import dynamicsymbols
        >>> q = dynamicsymbols('q')
        >>> N = ReferenceFrame('N')
        >>> pA, pB = Point('pA'), Point('pB')
        >>> pB.set_pos(pA, q*N.x)
        >>> pathway = LinearPathway(pA, pB)

        Now create a symbol ``k`` to describe the spring's stiffness and
        instantiate a force actuator that produces a (contractile) force
        proportional to both the spring's stiffness and the pathway's length.
        Note that actuator classes use the sign convention that expansile
        forces are positive, so for a spring to produce a contractile force the
        spring force needs to be calculated as the negative for the stiffness
        multiplied by the length.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import ForceActuator
        >>> stiffness = symbols('k')
        >>> spring_force = -stiffness*pathway.length
        >>> spring = ForceActuator(spring_force, pathway)

        The forces produced by the spring can be generated in the list of loads
        form that ``KanesMethod`` (and other equations of motion methods)
        requires by calling the ``to_loads`` method.

        >>> spring.to_loads()
        [(pA, k*q(t)*N.x), (pB, - k*q(t)*N.x)]

        A simple linear damper can be modeled in a similar way. Create another
        symbol ``c`` to describe the dampers damping coefficient. This time
        instantiate a force actuator that produces a force proportional to both
        the damper's damping coefficient and the pathway's extension velocity.
        Note that the damping force is negative as it acts in the opposite
        direction to which the damper is changing in length.

        >>> damping_coefficient = symbols('c')
        >>> damping_force = -damping_coefficient*pathway.extension_velocity
        >>> damper = ForceActuator(damping_force, pathway)

        Again, the forces produces by the damper can be generated by calling
        the ``to_loads`` method.

        >>> damper.to_loads()
        [(pA, c*Derivative(q(t), t)*N.x), (pB, - c*Derivative(q(t), t)*N.x)]

        """
        return self.pathway.to_loads(self.force)

    def __repr__(self):
        """Representation of a ``ForceActuator``."""
        return f'{self.__class__.__name__}({self.force}, {self.pathway})'


class LinearSpring(ForceActuator):
    """A spring with its spring force as a linear function of its length.

    Explanation
    ===========

    Note that the "linear" in the name ``LinearSpring`` refers to the fact that
    the spring force is a linear function of the springs length. I.e. for a
    linear spring with stiffness ``k``, distance between its ends of ``x``, and
    an equilibrium length of ``0``, the spring force will be ``-k*x``, which is
    a linear function in ``x``. To create a spring that follows a linear, or
    straight, pathway between its two ends, a ``LinearPathway`` instance needs
    to be passed to the ``pathway`` parameter.

    A ``LinearSpring`` is a subclass of ``ForceActuator`` and so follows the
    same sign conventions for length, extension velocity, and the direction of
    the forces it applies to its points of attachment on bodies. The sign
    convention for the direction of forces is such that, for the case where a
    linear spring is instantiated with a ``LinearPathway`` instance as its
    pathway, they act to push the two ends of the spring away from one another.
    Because springs produces a contractile force and acts to pull the two ends
    together towards the equilibrium length when stretched, the scalar portion
    of the forces on the endpoint are negative in order to flip the sign of the
    forces on the endpoints when converted into vector quantities. The
    following diagram shows the positive force sense and the distance between
    the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct a linear spring, an expression (or symbol) must be supplied to
    represent the stiffness (spring constant) of the spring, alongside a
    pathway specifying its line of action. Let's also create a global reference
    frame and spatially fix one of the points in it while setting the other to
    be positioned such that it can freely move in the frame's x direction
    specified by the coordinate ``q``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearPathway, LinearSpring,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> stiffness = symbols('k')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> spring = LinearSpring(stiffness, linear_pathway)
    >>> spring
    LinearSpring(k, LinearPathway(pA, pB))

    This spring will produce a force that is proportional to both its stiffness
    and the pathway's length. Note that this force is negative as SymPy's sign
    convention for actuators is that negative forces are contractile.

    >>> spring.force
    -k*sqrt(q(t)**2)

    To create a linear spring with a non-zero equilibrium length, an expression
    (or symbol) can be passed to the ``equilibrium_length`` parameter on
    construction on a ``LinearSpring`` instance. Let's create a symbol ``l``
    to denote a non-zero equilibrium length and create another linear spring.

    >>> l = symbols('l')
    >>> spring = LinearSpring(stiffness, linear_pathway, equilibrium_length=l)
    >>> spring
    LinearSpring(k, LinearPathway(pA, pB), equilibrium_length=l)

    The spring force of this new spring is again proportional to both its
    stiffness and the pathway's length. However, the spring will not produce
    any force when ``q(t)`` equals ``l``. Note that the force will become
    expansile when ``q(t)`` is less than ``l``, as expected.

    >>> spring.force
    -k*(-l + sqrt(q(t)**2))

    Parameters
    ==========

    stiffness : Expr
        The spring constant.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    equilibrium_length : Expr, optional
        The length at which the spring is in equilibrium, i.e. it produces no
        force. The default value is 0, i.e. the spring force is a linear
        function of the pathway's length with no constant offset.

    See Also
    ========

    ForceActuator: force-producing actuator (superclass of ``LinearSpring``).
    LinearPathway: straight-line pathway between a pair of points.

    """

    def __init__(self, stiffness, pathway, equilibrium_length=S.Zero):
        """Initializer for ``LinearSpring``.

        Parameters
        ==========

        stiffness : Expr
            The spring constant.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
        equilibrium_length : Expr, optional
            The length at which the spring is in equilibrium, i.e. it produces
            no force. The default value is 0, i.e. the spring force is a linear
            function of the pathway's length with no constant offset.

        """
        self.stiffness = stiffness
        self.pathway = pathway
        self.equilibrium_length = equilibrium_length

    @property
    def force(self):
        """The spring force produced by the linear spring."""
        return -self.stiffness*(self.pathway.length - self.equilibrium_length)

    @force.setter
    def force(self, force):
        raise AttributeError('Can\'t set computed attribute `force`.')

    @property
    def stiffness(self):
        """The spring constant for the linear spring."""
        return self._stiffness

    @stiffness.setter
    def stiffness(self, stiffness):
        if hasattr(self, '_stiffness'):
            msg = (
                f'Can\'t set attribute `stiffness` to {repr(stiffness)} as it '
                f'is immutable.'
            )
            raise AttributeError(msg)
        self._stiffness = sympify(stiffness, strict=True)

    @property
    def equilibrium_length(self):
        """The length of the spring at which it produces no force."""
        return self._equilibrium_length

    @equilibrium_length.setter
    def equilibrium_length(self, equilibrium_length):
        if hasattr(self, '_equilibrium_length'):
            msg = (
                f'Can\'t set attribute `equilibrium_length` to '
                f'{repr(equilibrium_length)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._equilibrium_length = sympify(equilibrium_length, strict=True)

    def __repr__(self):
        """Representation of a ``LinearSpring``."""
        string = f'{self.__class__.__name__}({self.stiffness}, {self.pathway}'
        if self.equilibrium_length == S.Zero:
            string += ')'
        else:
            string += f', equilibrium_length={self.equilibrium_length})'
        return string


class LinearDamper(ForceActuator):
    """A damper whose force is a linear function of its extension velocity.

    Explanation
    ===========

    Note that the "linear" in the name ``LinearDamper`` refers to the fact that
    the damping force is a linear function of the damper's rate of change in
    its length. I.e. for a linear damper with damping ``c`` and extension
    velocity ``v``, the damping force will be ``-c*v``, which is a linear
    function in ``v``. To create a damper that follows a linear, or straight,
    pathway between its two ends, a ``LinearPathway`` instance needs to be
    passed to the ``pathway`` parameter.

    A ``LinearDamper`` is a subclass of ``ForceActuator`` and so follows the
    same sign conventions for length, extension velocity, and the direction of
    the forces it applies to its points of attachment on bodies. The sign
    convention for the direction of forces is such that, for the case where a
    linear damper is instantiated with a ``LinearPathway`` instance as its
    pathway, they act to push the two ends of the damper away from one another.
    Because dampers produce a force that opposes the direction of change in
    length, when extension velocity is positive the scalar portions of the
    forces applied at the two endpoints are negative in order to flip the sign
    of the forces on the endpoints wen converted into vector quantities. When
    extension velocity is negative (i.e. when the damper is shortening), the
    scalar portions of the fofces applied are also negative so that the signs
    cancel producing forces on the endpoints that are in the same direction as
    the positive sign convention for the forces at the endpoints of the pathway
    (i.e. they act to push the endpoints away from one another). The following
    diagram shows the positive force sense and the distance between the
    points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct a linear damper, an expression (or symbol) must be supplied to
    represent the damping coefficient of the damper (we'll use the symbol
    ``c``), alongside a pathway specifying its line of action. Let's also
    create a global reference frame and spatially fix one of the points in it
    while setting the other to be positioned such that it can freely move in
    the frame's x direction specified by the coordinate ``q``. The velocity
    that the two points move away from one another can be specified by the
    coordinate ``u`` where ``u`` is the first time derivative of ``q``
    (i.e., ``u = Derivative(q(t), t)``).

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearDamper, LinearPathway,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> damping = symbols('c')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> pB.vel(N)
    Derivative(q(t), t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> damper = LinearDamper(damping, linear_pathway)
    >>> damper
    LinearDamper(c, LinearPathway(pA, pB))

    This damper will produce a force that is proportional to both its damping
    coefficient and the pathway's extension length. Note that this force is
    negative as SymPy's sign convention for actuators is that negative forces
    are contractile and the damping force of the damper will oppose the
    direction of length change.

    >>> damper.force
    -c*sqrt(q(t)**2)*Derivative(q(t), t)/q(t)

    Parameters
    ==========

    damping : Expr
        The damping constant.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

    See Also
    ========

    ForceActuator: force-producing actuator (superclass of ``LinearDamper``).
    LinearPathway: straight-line pathway between a pair of points.

    """

    def __init__(self, damping, pathway):
        """Initializer for ``LinearDamper``.

        Parameters
        ==========

        damping : Expr
            The damping constant.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

        """
        self.damping = damping
        self.pathway = pathway

    @property
    def force(self):
        """The damping force produced by the linear damper."""
        return -self.damping*self.pathway.extension_velocity

    @force.setter
    def force(self, force):
        raise AttributeError('Can\'t set computed attribute `force`.')

    @property
    def damping(self):
        """The damping constant for the linear damper."""
        return self._damping

    @damping.setter
    def damping(self, damping):
        if hasattr(self, '_damping'):
            msg = (
                f'Can\'t set attribute `damping` to {repr(damping)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._damping = sympify(damping, strict=True)

    def __repr__(self):
        """Representation of a ``LinearDamper``."""
        return f'{self.__class__.__name__}({self.damping}, {self.pathway})'


class TorqueActuator(ActuatorBase):
    """Torque-producing actuator.

    Explanation
    ===========

    A ``TorqueActuator`` is an actuator that produces a pair of equal and
    opposite torques on a pair of bodies.

    Examples
    ========

    To construct a torque actuator, an expression (or symbol) must be supplied
    to represent the torque it can produce, alongside a vector specifying the
    axis about which the torque will act, and a pair of frames on which the
    torque will act.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (ReferenceFrame, RigidBody,
    ...     TorqueActuator)
    >>> N = ReferenceFrame('N')
    >>> A = ReferenceFrame('A')
    >>> torque = symbols('T')
    >>> axis = N.z
    >>> parent = RigidBody('parent', frame=N)
    >>> child = RigidBody('child', frame=A)
    >>> bodies = (child, parent)
    >>> actuator = TorqueActuator(torque, axis, *bodies)
    >>> actuator
    TorqueActuator(T, axis=N.z, target_frame=A, reaction_frame=N)

    Note that because torques actually act on frames, not bodies,
    ``TorqueActuator`` will extract the frame associated with a ``RigidBody``
    when one is passed instead of a ``ReferenceFrame``.

    Parameters
    ==========

    torque : Expr
        The scalar expression defining the torque that the actuator produces.
    axis : Vector
        The axis about which the actuator applies torques.
    target_frame : ReferenceFrame | RigidBody
        The primary frame on which the actuator will apply the torque.
    reaction_frame : ReferenceFrame | RigidBody | None
        The secondary frame on which the actuator will apply the torque. Note
        that the (equal and opposite) reaction torque is applied to this frame.

    """

    def __init__(self, torque, axis, target_frame, reaction_frame=None):
        """Initializer for ``TorqueActuator``.

        Parameters
        ==========

        torque : Expr
            The scalar expression defining the torque that the actuator
            produces.
        axis : Vector
            The axis about which the actuator applies torques.
        target_frame : ReferenceFrame | RigidBody
            The primary frame on which the actuator will apply the torque.
        reaction_frame : ReferenceFrame | RigidBody | None
           The secondary frame on which the actuator will apply the torque.
           Note that the (equal and opposite) reaction torque is applied to
           this frame.

        """
        self.torque = torque
        self.axis = axis
        self.target_frame = target_frame
        self.reaction_frame = reaction_frame

    @classmethod
    def at_pin_joint(cls, torque, pin_joint):
        """Alternate constructor to instantiate from a ``PinJoint`` instance.

        Examples
        ========

        To create a pin joint the ``PinJoint`` class requires a name, parent
        body, and child body to be passed to its constructor. It is also
        possible to control the joint axis using the ``joint_axis`` keyword
        argument. In this example let's use the parent body's reference frame's
        z-axis as the joint axis.

        >>> from sympy.physics.mechanics import (PinJoint, ReferenceFrame,
        ...     RigidBody, TorqueActuator)
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> parent = RigidBody('parent', frame=N)
        >>> child = RigidBody('child', frame=A)
        >>> pin_joint = PinJoint(
        ...     'pin',
        ...     parent,
        ...     child,
        ...     joint_axis=N.z,
        ... )

        Let's also create a symbol ``T`` that will represent the torque applied
        by the torque actuator.

        >>> from sympy import symbols
        >>> torque = symbols('T')

        To create the torque actuator from the ``torque`` and ``pin_joint``
        variables previously instantiated, these can be passed to the alternate
        constructor class method ``at_pin_joint`` of the ``TorqueActuator``
        class. It should be noted that a positive torque will cause a positive
        displacement of the joint coordinate or that the torque is applied on
        the child body with a reaction torque on the parent.

        >>> actuator = TorqueActuator.at_pin_joint(torque, pin_joint)
        >>> actuator
        TorqueActuator(T, axis=N.z, target_frame=A, reaction_frame=N)

        Parameters
        ==========

        torque : Expr
            The scalar expression defining the torque that the actuator
            produces.
        pin_joint : PinJoint
            The pin joint, and by association the parent and child bodies, on
            which the torque actuator will act. The pair of bodies acted upon
            by the torque actuator are the parent and child bodies of the pin
            joint, with the child acting as the reaction body. The pin joint's
            axis is used as the axis about which the torque actuator will apply
            its torque.

        """
        if not isinstance(pin_joint, PinJoint):
            msg = (
                f'Value {repr(pin_joint)} passed to `pin_joint` was of type '
                f'{type(pin_joint)}, must be {PinJoint}.'
            )
            raise TypeError(msg)
        return cls(
            torque,
            pin_joint.joint_axis,
            pin_joint.child_interframe,
            pin_joint.parent_interframe,
        )

    @property
    def torque(self):
        """The magnitude of the torque produced by the actuator."""
        return self._torque

    @torque.setter
    def torque(self, torque):
        if hasattr(self, '_torque'):
            msg = (
                f'Can\'t set attribute `torque` to {repr(torque)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._torque = sympify(torque, strict=True)

    @property
    def axis(self):
        """The axis about which the torque acts."""
        return self._axis

    @axis.setter
    def axis(self, axis):
        if hasattr(self, '_axis'):
            msg = (
                f'Can\'t set attribute `axis` to {repr(axis)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(axis, Vector):
            msg = (
                f'Value {repr(axis)} passed to `axis` was of type '
                f'{type(axis)}, must be {Vector}.'
            )
            raise TypeError(msg)
        self._axis = axis

    @property
    def target_frame(self):
        """The primary reference frames on which the torque will act."""
        return self._target_frame

    @target_frame.setter
    def target_frame(self, target_frame):
        if hasattr(self, '_target_frame'):
            msg = (
                f'Can\'t set attribute `target_frame` to {repr(target_frame)} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        if isinstance(target_frame, RigidBody):
            target_frame = target_frame.frame
        elif not isinstance(target_frame, ReferenceFrame):
            msg = (
                f'Value {repr(target_frame)} passed to `target_frame` was of '
                f'type {type(target_frame)}, must be {ReferenceFrame}.'
            )
            raise TypeError(msg)
        self._target_frame = target_frame

    @property
    def reaction_frame(self):
        """The primary reference frames on which the torque will act."""
        return self._reaction_frame

    @reaction_frame.setter
    def reaction_frame(self, reaction_frame):
        if hasattr(self, '_reaction_frame'):
            msg = (
                f'Can\'t set attribute `reaction_frame` to '
                f'{repr(reaction_frame)} as it is immutable.'
            )
            raise AttributeError(msg)
        if isinstance(reaction_frame, RigidBody):
            reaction_frame = reaction_frame.frame
        elif (
            not isinstance(reaction_frame, ReferenceFrame)
            and reaction_frame is not None
        ):
            msg = (
                f'Value {repr(reaction_frame)} passed to `reaction_frame` was '
                f'of type {type(reaction_frame)}, must be {ReferenceFrame}.'
            )
            raise TypeError(msg)
        self._reaction_frame = reaction_frame

    def to_loads(self):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        Examples
        ========

        The below example shows how to generate the loads produced by a torque
        actuator that acts on a pair of bodies attached by a pin joint.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import (PinJoint, ReferenceFrame,
        ...     RigidBody, TorqueActuator)
        >>> torque = symbols('T')
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> parent = RigidBody('parent', frame=N)
        >>> child = RigidBody('child', frame=A)
        >>> pin_joint = PinJoint(
        ...     'pin',
        ...     parent,
        ...     child,
        ...     joint_axis=N.z,
        ... )
        >>> actuator = TorqueActuator.at_pin_joint(torque, pin_joint)

        The forces produces by the damper can be generated by calling the
        ``to_loads`` method.

        >>> actuator.to_loads()
        [(A, T*N.z), (N, - T*N.z)]

        Alternatively, if a torque actuator is created without a reaction frame
        then the loads returned by the ``to_loads`` method will contain just
        the single load acting on the target frame.

        >>> actuator = TorqueActuator(torque, N.z, N)
        >>> actuator.to_loads()
        [(N, T*N.z)]

        """
        loads = [
            Torque(self.target_frame, self.torque*self.axis),
        ]
        if self.reaction_frame is not None:
            loads.append(Torque(self.reaction_frame, -self.torque*self.axis))
        return loads

    def __repr__(self):
        """Representation of a ``TorqueActuator``."""
        string = (
            f'{self.__class__.__name__}({self.torque}, axis={self.axis}, '
            f'target_frame={self.target_frame}'
        )
        if self.reaction_frame is not None:
            string += f', reaction_frame={self.reaction_frame})'
        else:
            string += ')'
        return string


class DuffingSpring(ForceActuator):
    """A nonlinear spring based on the Duffing equation.

    Explanation
    ===========

    Here, ``DuffingSpring`` represents the force exerted by a nonlinear spring based on the Duffing equation:
    F = -beta*x-alpha*x**3, where x is the displacement from the equilibrium position, beta is the linear spring constant,
    and alpha is the coefficient for the nonlinear cubic term.

    Parameters
    ==========

    linear_stiffness : Expr
        The linear stiffness coefficient (beta).
    nonlinear_stiffness : Expr
        The nonlinear stiffness coefficient (alpha).
    pathway : PathwayBase
        The pathway that the actuator follows.
    equilibrium_length : Expr, optional
        The length at which the spring is in equilibrium (x).
    """

    def __init__(self, linear_stiffness, nonlinear_stiffness, pathway, equilibrium_length=S.Zero):
        self.linear_stiffness = sympify(linear_stiffness, strict=True)
        self.nonlinear_stiffness = sympify(nonlinear_stiffness, strict=True)
        self.equilibrium_length = sympify(equilibrium_length, strict=True)

        if not isinstance(pathway, PathwayBase):
            raise TypeError("pathway must be an instance of PathwayBase.")
        self._pathway = pathway

    @property
    def linear_stiffness(self):
        return self._linear_stiffness

    @linear_stiffness.setter
    def linear_stiffness(self, linear_stiffness):
        if hasattr(self, '_linear_stiffness'):
            msg = (
                f'Can\'t set attribute `linear_stiffness` to '
                f'{repr(linear_stiffness)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._linear_stiffness = sympify(linear_stiffness, strict=True)

    @property
    def nonlinear_stiffness(self):
        return self._nonlinear_stiffness

    @nonlinear_stiffness.setter
    def nonlinear_stiffness(self, nonlinear_stiffness):
        if hasattr(self, '_nonlinear_stiffness'):
            msg = (
                f'Can\'t set attribute `nonlinear_stiffness` to '
                f'{repr(nonlinear_stiffness)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._nonlinear_stiffness = sympify(nonlinear_stiffness, strict=True)

    @property
    def pathway(self):
        return self._pathway

    @pathway.setter
    def pathway(self, pathway):
        if hasattr(self, '_pathway'):
            msg = (
                f'Can\'t set attribute `pathway` to {repr(pathway)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(pathway, PathwayBase):
            msg = (
                f'Value {repr(pathway)} passed to `pathway` was of type '
                f'{type(pathway)}, must be {PathwayBase}.'
            )
            raise TypeError(msg)
        self._pathway = pathway

    @property
    def equilibrium_length(self):
        return self._equilibrium_length

    @equilibrium_length.setter
    def equilibrium_length(self, equilibrium_length):
        if hasattr(self, '_equilibrium_length'):
            msg = (
                f'Can\'t set attribute `equilibrium_length` to '
                f'{repr(equilibrium_length)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._equilibrium_length = sympify(equilibrium_length, strict=True)

    @property
    def force(self):
        """The force produced by the Duffing spring."""
        displacement = self.pathway.length - self.equilibrium_length
        return -self.linear_stiffness * displacement - self.nonlinear_stiffness * displacement**3

    @force.setter
    def force(self, force):
        if hasattr(self, '_force'):
            msg = (
                f'Can\'t set attribute `force` to {repr(force)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._force = sympify(force, strict=True)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self.linear_stiffness}, {self.nonlinear_stiffness}, {self.pathway}, "
                f"equilibrium_length={self.equilibrium_length})")

class CoulombKineticFriction(ForceActuator):
    r"""Coulomb kinetic friction with Stribeck and viscous effects.

    Explanation
    ===========

    This represents a Coulomb kinetic friction with the Stribeck and viscous effect,
    described by the function:

    .. math::
        F = (\mu_k f_n + (\mu_s - \mu_k) f_n e^{-(\frac{v}{v_s})^2}) \text{sign}(v) + \sigma  v

    where :math:`\mu_k` is the coefficient of kinetic friction, :math:`\mu_s` is the
    coefficient of static friction, :math:`f_n` is the normal force, :math:`v` is the
    relative velocity, :math:`v_s` is the Stribeck friction coefficient, and
    :math:`\sigma` is the viscous friction constant.

    The default friction force is :math:`F = \mu_k f_n`.
    When specified, the actuator includes:

    - Stribeck effect: :math:`(\mu_s - \mu_k) f_n e^{-(\frac{v}{v_s})^2}`
    - Viscous effect: :math:`\sigma v`

    Notes
    =====

    The actuator makes the following assumptions:

    - The actuator assumes relative motion is non-zero.
    - The normal force is assumed to be a non-negative scalar.
    - The resultant friction force is opposite to the velocity direction.
    - Each point in the pathway is fixed within separate objects that are sliding relative to each other. In other words, these two points are fixed in the mutually sliding objects.

    This actuator has been tested for straightforward motions, like a block sliding
    on a surface.

    The friction force is defined to always oppose the direction of relative velocity :math:`v`.
    Specifically:

    - The default Coulomb friction force :math:`\mu_k f_n \text{sign}(v)` is opposite to :math:`v`.
    - The Stribeck effect :math:`(\mu_s - \mu_k) f_n e^{-(\frac{v}{v_s})^2} \text{sign}(v)` is also opposite to :math:`v`.
    - The viscous friction term :math:`\sigma v` is opposite to :math:`v`.

    Examples
    ========

    The below example shows how to generate the loads produced by a Coulomb kinetic
    friction actuator in a mass-spring system with friction.

    >>> import sympy as sm
    >>> from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
    ...     LinearPathway, CoulombKineticFriction, LinearSpring, KanesMethod, Particle)

    >>> x, v = dynamicsymbols('x, v', real=True)
    >>> m, g, k, mu_k, mu_s, v_s, sigma = sm.symbols('m, g, k, mu_k, mu_s, v_s, sigma')

    >>> N = ReferenceFrame('N')
    >>> O, P = Point('O'), Point('P')
    >>> O.set_vel(N, 0)
    >>> P.set_pos(O, x*N.x)

    >>> pathway = LinearPathway(O, P)
    >>> friction = CoulombKineticFriction(mu_k, m*g, pathway, v_s=v_s, sigma=sigma, mu_s=mu_k)
    >>> spring = LinearSpring(k, pathway)
    >>> block = Particle('block', point=P, mass=m)

    >>> kane = KanesMethod(N, (x,), (v,), kd_eqs=(x.diff() - v,))
    >>> friction.to_loads()
        [(O, (g*m*mu_k*sign(sign(x(t))*Derivative(x(t), t)) + sigma*sign(x(t))*Derivative(x(t), t))*x(t)/Abs(x(t))*N.x), (P, (-g*m*mu_k*sign(sign(x(t))*Derivative(x(t), t)) - sigma*sign(x(t))*Derivative(x(t), t))*x(t)/Abs(x(t))*N.x)]
    >>> loads = friction.to_loads() + spring.to_loads()
    >>> fr, frstar = kane.kanes_equations([block], loads)
    >>> eom = fr + frstar
    >>> eom
        Matrix([[-k*x(t) - m*Derivative(v(t), t) + (-g*m*mu_k*sign(v(t)*sign(x(t))) - sigma*v(t)*sign(x(t)))*x(t)/Abs(x(t))]])

    Parameters
    ==========

    f_n : sympifiable
        The normal force between the surfaces. It should always be a non-negative scalar.
    mu_k : sympifiable
        The coefficient of kinetic friction.
    pathway : PathwayBase
        The pathway that the actuator follows.
    v_s : sympifiable, optional
        The Stribeck friction coefficient.
    sigma : sympifiable, optional
        The viscous friction coefficient.
    mu_s : sympifiable, optional
        The coefficient of static friction. Defaults to mu_k, meaning the Stribeck effect evaluates to 0 by default.

    References
    ==========

    .. [Moore2022] https://moorepants.github.io/learn-multibody-dynamics/loads.html#friction.
    .. [Flores2023] Paulo Flores, Jorge Ambrosio, Hamid M. Lankarani,
            "Contact-impact events with friction in multibody dynamics: Back to basics",
            Mechanism and Machine Theory, vol. 184, 2023. https://doi.org/10.1016/j.mechmachtheory.2023.105305.
    .. [Rogner2017] I. Rogner, "Friction modelling for robotic applications with planar motion",
            Chalmers University of Technology, Department of Electrical Engineering, 2017.

    """

    def __init__(self, mu_k, f_n, pathway, *, v_s=None, sigma=None, mu_s=None):
        self._mu_k = sympify(mu_k, strict=True) if mu_k is not None else 1
        self._mu_s = sympify(mu_s, strict=True) if mu_s is not None else self._mu_k
        self._f_n = sympify(f_n, strict=True)
        self._sigma = sympify(sigma, strict=True) if sigma is not None else 0
        self._v_s = sympify(v_s, strict=True) if v_s is not None or v_s == 0 else 0.01
        self.pathway = pathway

    @property
    def mu_k(self):
        """The coefficient of kinetic friction."""
        return self._mu_k

    @property
    def mu_s(self):
        """The coefficient of static friction."""
        return self._mu_s

    @property
    def f_n(self):
        """The normal force between the surfaces."""
        return self._f_n

    @property
    def sigma(self):
        """The viscous friction coefficient."""
        return self._sigma

    @property
    def v_s(self):
        """The Stribeck friction coefficient."""
        return self._v_s

    @property
    def force(self):
        v = self.pathway.extension_velocity
        f_c = self.mu_k * self.f_n
        f_max = self.mu_s * self.f_n
        stribeck_term = (f_max - f_c) * exp(-(v / self.v_s)**2) if self.v_s is not None else 0
        viscous_term = self.sigma * v if self.sigma is not None else 0
        return (f_c + stribeck_term) * -sign(v) - viscous_term

    @force.setter
    def force(self, force):
        raise AttributeError('Can\'t set computed attribute `force`.')

    def __repr__(self):
        return (f'{self.__class__.__name__}({self._mu_k}, {self._mu_s} '
                f'{self._f_n}, {self.pathway}, {self._v_s}, '
                f'{self._sigma})')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\state.py ===
# orm/state.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Defines instrumentation of instances.

This module is usually not directly visible to user applications, but
defines a large part of the ORM's interactivity.

"""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref

from . import base
from . import exc as orm_exc
from . import interfaces
from ._typing import _O
from ._typing import is_collection_impl
from .base import ATTR_WAS_SET
from .base import INIT_OK
from .base import LoaderCallableStatus
from .base import NEVER_SET
from .base import NO_VALUE
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import SQL_OK
from .path_registry import PathRegistry
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol

if TYPE_CHECKING:
    from ._typing import _IdentityKeyType
    from ._typing import _InstanceDict
    from ._typing import _LoaderCallable
    from .attributes import AttributeImpl
    from .attributes import History
    from .base import PassiveFlag
    from .collections import _AdaptedCollectionProtocol
    from .identity import IdentityMap
    from .instrumentation import ClassManager
    from .interfaces import ORMOption
    from .mapper import Mapper
    from .session import Session
    from ..engine import Row
    from ..ext.asyncio.session import async_session as _async_provider
    from ..ext.asyncio.session import AsyncSession

if TYPE_CHECKING:
    _sessions: weakref.WeakValueDictionary[int, Session]
else:
    # late-populated by session.py
    _sessions = None


if not TYPE_CHECKING:
    # optionally late-provided by sqlalchemy.ext.asyncio.session

    _async_provider = None  # noqa


class _InstanceDictProto(Protocol):
    def __call__(self) -> Optional[IdentityMap]: ...


class _InstallLoaderCallableProto(Protocol[_O]):
    """used at result loading time to install a _LoaderCallable callable
    upon a specific InstanceState, which will be used to populate an
    attribute when that attribute is accessed.

    Concrete examples are per-instance deferred column loaders and
    relationship lazy loaders.

    """

    def __call__(
        self, state: InstanceState[_O], dict_: _InstanceDict, row: Row[Any]
    ) -> None: ...


@inspection._self_inspects
class InstanceState(interfaces.InspectionAttrInfo, Generic[_O]):
    """Tracks state information at the instance level.

    The :class:`.InstanceState` is a key object used by the
    SQLAlchemy ORM in order to track the state of an object;
    it is created the moment an object is instantiated, typically
    as a result of :term:`instrumentation` which SQLAlchemy applies
    to the ``__init__()`` method of the class.

    :class:`.InstanceState` is also a semi-public object,
    available for runtime inspection as to the state of a
    mapped instance, including information such as its current
    status within a particular :class:`.Session` and details
    about data on individual attributes.  The public API
    in order to acquire a :class:`.InstanceState` object
    is to use the :func:`_sa.inspect` system::

        >>> from sqlalchemy import inspect
        >>> insp = inspect(some_mapped_object)
        >>> insp.attrs.nickname.history
        History(added=['new nickname'], unchanged=(), deleted=['nickname'])

    .. seealso::

        :ref:`orm_mapper_inspection_instancestate`

    """

    __slots__ = (
        "__dict__",
        "__weakref__",
        "class_",
        "manager",
        "obj",
        "committed_state",
        "expired_attributes",
    )

    manager: ClassManager[_O]
    session_id: Optional[int] = None
    key: Optional[_IdentityKeyType[_O]] = None
    runid: Optional[int] = None
    load_options: Tuple[ORMOption, ...] = ()
    load_path: PathRegistry = PathRegistry.root
    insert_order: Optional[int] = None
    _strong_obj: Optional[object] = None
    obj: weakref.ref[_O]

    committed_state: Dict[str, Any]

    modified: bool = False
    """When ``True`` the object was modified."""
    expired: bool = False
    """When ``True`` the object is :term:`expired`.

    .. seealso::

        :ref:`session_expire`
    """
    _deleted: bool = False
    _load_pending: bool = False
    _orphaned_outside_of_session: bool = False
    is_instance: bool = True
    identity_token: object = None
    _last_known_values: Optional[Dict[str, Any]] = None

    _instance_dict: _InstanceDictProto
    """A weak reference, or in the default case a plain callable, that
    returns a reference to the current :class:`.IdentityMap`, if any.

    """
    if not TYPE_CHECKING:

        def _instance_dict(self):
            """default 'weak reference' for _instance_dict"""
            return None

    expired_attributes: Set[str]
    """The set of keys which are 'expired' to be loaded by
    the manager's deferred scalar loader, assuming no pending
    changes.

    See also the ``unmodified`` collection which is intersected
    against this set when a refresh operation occurs.
    """

    callables: Dict[str, Callable[[InstanceState[_O], PassiveFlag], Any]]
    """A namespace where a per-state loader callable can be associated.

    In SQLAlchemy 1.0, this is only used for lazy loaders / deferred
    loaders that were set up via query option.

    Previously, callables was used also to indicate expired attributes
    by storing a link to the InstanceState itself in this dictionary.
    This role is now handled by the expired_attributes set.

    """

    if not TYPE_CHECKING:
        callables = util.EMPTY_DICT

    def __init__(self, obj: _O, manager: ClassManager[_O]):
        self.class_ = obj.__class__
        self.manager = manager
        self.obj = weakref.ref(obj, self._cleanup)
        self.committed_state = {}
        self.expired_attributes = set()

    @util.memoized_property
    def attrs(self) -> util.ReadOnlyProperties[AttributeState]:
        """Return a namespace representing each attribute on
        the mapped object, including its current value
        and history.

        The returned object is an instance of :class:`.AttributeState`.
        This object allows inspection of the current data
        within an attribute as well as attribute history
        since the last flush.

        """
        return util.ReadOnlyProperties(
            {key: AttributeState(self, key) for key in self.manager}
        )

    @property
    def transient(self) -> bool:
        """Return ``True`` if the object is :term:`transient`.

        .. seealso::

            :ref:`session_object_states`

        """
        return self.key is None and not self._attached

    @property
    def pending(self) -> bool:
        """Return ``True`` if the object is :term:`pending`.

        .. seealso::

            :ref:`session_object_states`

        """
        return self.key is None and self._attached

    @property
    def deleted(self) -> bool:
        """Return ``True`` if the object is :term:`deleted`.

        An object that is in the deleted state is guaranteed to
        not be within the :attr:`.Session.identity_map` of its parent
        :class:`.Session`; however if the session's transaction is rolled
        back, the object will be restored to the persistent state and
        the identity map.

        .. note::

            The :attr:`.InstanceState.deleted` attribute refers to a specific
            state of the object that occurs between the "persistent" and
            "detached" states; once the object is :term:`detached`, the
            :attr:`.InstanceState.deleted` attribute **no longer returns
            True**; in order to detect that a state was deleted, regardless
            of whether or not the object is associated with a
            :class:`.Session`, use the :attr:`.InstanceState.was_deleted`
            accessor.

        .. versionadded: 1.1

        .. seealso::

            :ref:`session_object_states`

        """
        return self.key is not None and self._attached and self._deleted

    @property
    def was_deleted(self) -> bool:
        """Return True if this object is or was previously in the
        "deleted" state and has not been reverted to persistent.

        This flag returns True once the object was deleted in flush.
        When the object is expunged from the session either explicitly
        or via transaction commit and enters the "detached" state,
        this flag will continue to report True.

        .. seealso::

            :attr:`.InstanceState.deleted` - refers to the "deleted" state

            :func:`.orm.util.was_deleted` - standalone function

            :ref:`session_object_states`

        """
        return self._deleted

    @property
    def persistent(self) -> bool:
        """Return ``True`` if the object is :term:`persistent`.

        An object that is in the persistent state is guaranteed to
        be within the :attr:`.Session.identity_map` of its parent
        :class:`.Session`.

        .. seealso::

            :ref:`session_object_states`

        """
        return self.key is not None and self._attached and not self._deleted

    @property
    def detached(self) -> bool:
        """Return ``True`` if the object is :term:`detached`.

        .. seealso::

            :ref:`session_object_states`

        """
        return self.key is not None and not self._attached

    @util.non_memoized_property
    @util.preload_module("sqlalchemy.orm.session")
    def _attached(self) -> bool:
        return (
            self.session_id is not None
            and self.session_id in util.preloaded.orm_session._sessions
        )

    def _track_last_known_value(self, key: str) -> None:
        """Track the last known value of a particular key after expiration
        operations.

        .. versionadded:: 1.3

        """

        lkv = self._last_known_values
        if lkv is None:
            self._last_known_values = lkv = {}
        if key not in lkv:
            lkv[key] = NO_VALUE

    @property
    def session(self) -> Optional[Session]:
        """Return the owning :class:`.Session` for this instance,
        or ``None`` if none available.

        Note that the result here can in some cases be *different*
        from that of ``obj in session``; an object that's been deleted
        will report as not ``in session``, however if the transaction is
        still in progress, this attribute will still refer to that session.
        Only when the transaction is completed does the object become
        fully detached under normal circumstances.

        .. seealso::

            :attr:`_orm.InstanceState.async_session`

        """
        if self.session_id:
            try:
                return _sessions[self.session_id]
            except KeyError:
                pass
        return None

    @property
    def async_session(self) -> Optional[AsyncSession]:
        """Return the owning :class:`_asyncio.AsyncSession` for this instance,
        or ``None`` if none available.

        This attribute is only non-None when the :mod:`sqlalchemy.ext.asyncio`
        API is in use for this ORM object. The returned
        :class:`_asyncio.AsyncSession` object will be a proxy for the
        :class:`_orm.Session` object that would be returned from the
        :attr:`_orm.InstanceState.session` attribute for this
        :class:`_orm.InstanceState`.

        .. versionadded:: 1.4.18

        .. seealso::

            :ref:`asyncio_toplevel`

        """
        if _async_provider is None:
            return None

        sess = self.session
        if sess is not None:
            return _async_provider(sess)
        else:
            return None

    @property
    def object(self) -> Optional[_O]:
        """Return the mapped object represented by this
        :class:`.InstanceState`.

        Returns None if the object has been garbage collected

        """
        return self.obj()

    @property
    def identity(self) -> Optional[Tuple[Any, ...]]:
        """Return the mapped identity of the mapped object.
        This is the primary key identity as persisted by the ORM
        which can always be passed directly to
        :meth:`_query.Query.get`.

        Returns ``None`` if the object has no primary key identity.

        .. note::
            An object which is :term:`transient` or :term:`pending`
            does **not** have a mapped identity until it is flushed,
            even if its attributes include primary key values.

        """
        if self.key is None:
            return None
        else:
            return self.key[1]

    @property
    def identity_key(self) -> Optional[_IdentityKeyType[_O]]:
        """Return the identity key for the mapped object.

        This is the key used to locate the object within
        the :attr:`.Session.identity_map` mapping.   It contains
        the identity as returned by :attr:`.identity` within it.


        """
        return self.key

    @util.memoized_property
    def parents(self) -> Dict[int, Union[Literal[False], InstanceState[Any]]]:
        return {}

    @util.memoized_property
    def _pending_mutations(self) -> Dict[str, PendingCollection]:
        return {}

    @util.memoized_property
    def _empty_collections(self) -> Dict[str, _AdaptedCollectionProtocol]:
        return {}

    @util.memoized_property
    def mapper(self) -> Mapper[_O]:
        """Return the :class:`_orm.Mapper` used for this mapped object."""
        return self.manager.mapper

    @property
    def has_identity(self) -> bool:
        """Return ``True`` if this object has an identity key.

        This should always have the same value as the
        expression ``state.persistent`` or ``state.detached``.

        """
        return bool(self.key)

    @classmethod
    def _detach_states(
        self,
        states: Iterable[InstanceState[_O]],
        session: Session,
        to_transient: bool = False,
    ) -> None:
        persistent_to_detached = (
            session.dispatch.persistent_to_detached or None
        )
        deleted_to_detached = session.dispatch.deleted_to_detached or None
        pending_to_transient = session.dispatch.pending_to_transient or None
        persistent_to_transient = (
            session.dispatch.persistent_to_transient or None
        )

        for state in states:
            deleted = state._deleted
            pending = state.key is None
            persistent = not pending and not deleted

            state.session_id = None

            if to_transient and state.key:
                del state.key
            if persistent:
                if to_transient:
                    if persistent_to_transient is not None:
                        persistent_to_transient(session, state)
                elif persistent_to_detached is not None:
                    persistent_to_detached(session, state)
            elif deleted and deleted_to_detached is not None:
                deleted_to_detached(session, state)
            elif pending and pending_to_transient is not None:
                pending_to_transient(session, state)

            state._strong_obj = None

    def _detach(self, session: Optional[Session] = None) -> None:
        if session:
            InstanceState._detach_states([self], session)
        else:
            self.session_id = self._strong_obj = None

    def _dispose(self) -> None:
        # used by the test suite, apparently
        self._detach()

    def _cleanup(self, ref: weakref.ref[_O]) -> None:
        """Weakref callback cleanup.

        This callable cleans out the state when it is being garbage
        collected.

        this _cleanup **assumes** that there are no strong refs to us!
        Will not work otherwise!

        """

        # Python builtins become undefined during interpreter shutdown.
        # Guard against exceptions during this phase, as the method cannot
        # proceed in any case if builtins have been undefined.
        if dict is None:
            return

        instance_dict = self._instance_dict()
        if instance_dict is not None:
            instance_dict._fast_discard(self)
            del self._instance_dict

            # we can't possibly be in instance_dict._modified
            # b.c. this is weakref cleanup only, that set
            # is strong referencing!
            # assert self not in instance_dict._modified

        self.session_id = self._strong_obj = None

    @property
    def dict(self) -> _InstanceDict:
        """Return the instance dict used by the object.

        Under normal circumstances, this is always synonymous
        with the ``__dict__`` attribute of the mapped object,
        unless an alternative instrumentation system has been
        configured.

        In the case that the actual object has been garbage
        collected, this accessor returns a blank dictionary.

        """
        o = self.obj()
        if o is not None:
            return base.instance_dict(o)
        else:
            return {}

    def _initialize_instance(*mixed: Any, **kwargs: Any) -> None:
        self, instance, args = mixed[0], mixed[1], mixed[2:]  # noqa
        manager = self.manager

        manager.dispatch.init(self, args, kwargs)

        try:
            manager.original_init(*mixed[1:], **kwargs)
        except:
            with util.safe_reraise():
                manager.dispatch.init_failure(self, args, kwargs)

    def get_history(self, key: str, passive: PassiveFlag) -> History:
        return self.manager[key].impl.get_history(self, self.dict, passive)

    def get_impl(self, key: str) -> AttributeImpl:
        return self.manager[key].impl

    def _get_pending_mutation(self, key: str) -> PendingCollection:
        if key not in self._pending_mutations:
            self._pending_mutations[key] = PendingCollection()
        return self._pending_mutations[key]

    def __getstate__(self) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = {
            "instance": self.obj(),
            "class_": self.class_,
            "committed_state": self.committed_state,
            "expired_attributes": self.expired_attributes,
        }
        state_dict.update(
            (k, self.__dict__[k])
            for k in (
                "_pending_mutations",
                "modified",
                "expired",
                "callables",
                "key",
                "parents",
                "load_options",
                "class_",
                "expired_attributes",
                "info",
            )
            if k in self.__dict__
        )
        if self.load_path:
            state_dict["load_path"] = self.load_path.serialize()

        state_dict["manager"] = self.manager._serialize(self, state_dict)

        return state_dict

    def __setstate__(self, state_dict: Dict[str, Any]) -> None:
        inst = state_dict["instance"]
        if inst is not None:
            self.obj = weakref.ref(inst, self._cleanup)
            self.class_ = inst.__class__
        else:
            self.obj = lambda: None  # type: ignore
            self.class_ = state_dict["class_"]

        self.committed_state = state_dict.get("committed_state", {})
        self._pending_mutations = state_dict.get("_pending_mutations", {})
        self.parents = state_dict.get("parents", {})
        self.modified = state_dict.get("modified", False)
        self.expired = state_dict.get("expired", False)
        if "info" in state_dict:
            self.info.update(state_dict["info"])
        if "callables" in state_dict:
            self.callables = state_dict["callables"]

            self.expired_attributes = state_dict["expired_attributes"]
        else:
            if "expired_attributes" in state_dict:
                self.expired_attributes = state_dict["expired_attributes"]
            else:
                self.expired_attributes = set()

        self.__dict__.update(
            [
                (k, state_dict[k])
                for k in ("key", "load_options")
                if k in state_dict
            ]
        )
        if self.key:
            self.identity_token = self.key[2]

        if "load_path" in state_dict:
            self.load_path = PathRegistry.deserialize(state_dict["load_path"])

        state_dict["manager"](self, inst, state_dict)

    def _reset(self, dict_: _InstanceDict, key: str) -> None:
        """Remove the given attribute and any
        callables associated with it."""

        old = dict_.pop(key, None)
        manager_impl = self.manager[key].impl
        if old is not None and is_collection_impl(manager_impl):
            manager_impl._invalidate_collection(old)
        self.expired_attributes.discard(key)
        if self.callables:
            self.callables.pop(key, None)

    def _copy_callables(self, from_: InstanceState[Any]) -> None:
        if "callables" in from_.__dict__:
            self.callables = dict(from_.callables)

    @classmethod
    def _instance_level_callable_processor(
        cls, manager: ClassManager[_O], fn: _LoaderCallable, key: Any
    ) -> _InstallLoaderCallableProto[_O]:
        impl = manager[key].impl
        if is_collection_impl(impl):
            fixed_impl = impl

            def _set_callable(
                state: InstanceState[_O], dict_: _InstanceDict, row: Row[Any]
            ) -> None:
                if "callables" not in state.__dict__:
                    state.callables = {}
                old = dict_.pop(key, None)
                if old is not None:
                    fixed_impl._invalidate_collection(old)
                state.callables[key] = fn

        else:

            def _set_callable(
                state: InstanceState[_O], dict_: _InstanceDict, row: Row[Any]
            ) -> None:
                if "callables" not in state.__dict__:
                    state.callables = {}
                state.callables[key] = fn

        return _set_callable

    def _expire(
        self, dict_: _InstanceDict, modified_set: Set[InstanceState[Any]]
    ) -> None:
        self.expired = True
        if self.modified:
            modified_set.discard(self)
            self.committed_state.clear()
            self.modified = False

        self._strong_obj = None

        if "_pending_mutations" in self.__dict__:
            del self.__dict__["_pending_mutations"]

        if "parents" in self.__dict__:
            del self.__dict__["parents"]

        self.expired_attributes.update(
            [impl.key for impl in self.manager._loader_impls]
        )

        if self.callables:
            # the per state loader callables we can remove here are
            # LoadDeferredColumns, which undefers a column at the instance
            # level that is mapped with deferred, and LoadLazyAttribute,
            # which lazy loads a relationship at the instance level that
            # is mapped with "noload" or perhaps "immediateload".
            # Before 1.4, only column-based
            # attributes could be considered to be "expired", so here they
            # were the only ones "unexpired", which means to make them deferred
            # again.   For the moment, as of 1.4 we also apply the same
            # treatment relationships now, that is, an instance level lazy
            # loader is reset in the same way as a column loader.
            for k in self.expired_attributes.intersection(self.callables):
                del self.callables[k]

        for k in self.manager._collection_impl_keys.intersection(dict_):
            collection = dict_.pop(k)
            collection._sa_adapter.invalidated = True

        if self._last_known_values:
            self._last_known_values.update(
                {k: dict_[k] for k in self._last_known_values if k in dict_}
            )

        for key in self.manager._all_key_set.intersection(dict_):
            del dict_[key]

        self.manager.dispatch.expire(self, None)

    def _expire_attributes(
        self,
        dict_: _InstanceDict,
        attribute_names: Iterable[str],
        no_loader: bool = False,
    ) -> None:
        pending = self.__dict__.get("_pending_mutations", None)

        callables = self.callables

        for key in attribute_names:
            impl = self.manager[key].impl
            if impl.accepts_scalar_loader:
                if no_loader and (impl.callable_ or key in callables):
                    continue

                self.expired_attributes.add(key)
                if callables and key in callables:
                    del callables[key]
            old = dict_.pop(key, NO_VALUE)
            if is_collection_impl(impl) and old is not NO_VALUE:
                impl._invalidate_collection(old)

            lkv = self._last_known_values
            if lkv is not None and key in lkv and old is not NO_VALUE:
                lkv[key] = old

            self.committed_state.pop(key, None)
            if pending:
                pending.pop(key, None)

        self.manager.dispatch.expire(self, attribute_names)

    def _load_expired(
        self, state: InstanceState[_O], passive: PassiveFlag
    ) -> LoaderCallableStatus:
        """__call__ allows the InstanceState to act as a deferred
        callable for loading expired attributes, which is also
        serializable (picklable).

        """

        if not passive & SQL_OK:
            return PASSIVE_NO_RESULT

        toload = self.expired_attributes.intersection(self.unmodified)
        toload = toload.difference(
            attr
            for attr in toload
            if not self.manager[attr].impl.load_on_unexpire
        )

        self.manager.expired_attribute_loader(self, toload, passive)

        # if the loader failed, or this
        # instance state didn't have an identity,
        # the attributes still might be in the callables
        # dict.  ensure they are removed.
        self.expired_attributes.clear()

        return ATTR_WAS_SET

    @property
    def unmodified(self) -> Set[str]:
        """Return the set of keys which have no uncommitted changes"""

        return set(self.manager).difference(self.committed_state)

    def unmodified_intersection(self, keys: Iterable[str]) -> Set[str]:
        """Return self.unmodified.intersection(keys)."""

        return (
            set(keys)
            .intersection(self.manager)
            .difference(self.committed_state)
        )

    @property
    def unloaded(self) -> Set[str]:
        """Return the set of keys which do not have a loaded value.

        This includes expired attributes and any other attribute that was never
        populated or modified.

        """
        return (
            set(self.manager)
            .difference(self.committed_state)
            .difference(self.dict)
        )

    @property
    @util.deprecated(
        "2.0",
        "The :attr:`.InstanceState.unloaded_expirable` attribute is "
        "deprecated.  Please use :attr:`.InstanceState.unloaded`.",
    )
    def unloaded_expirable(self) -> Set[str]:
        """Synonymous with :attr:`.InstanceState.unloaded`.

        This attribute was added as an implementation-specific detail at some
        point and should be considered to be private.

        """
        return self.unloaded

    @property
    def _unloaded_non_object(self) -> Set[str]:
        return self.unloaded.intersection(
            attr
            for attr in self.manager
            if self.manager[attr].impl.accepts_scalar_loader
        )

    def _modified_event(
        self,
        dict_: _InstanceDict,
        attr: Optional[AttributeImpl],
        previous: Any,
        collection: bool = False,
        is_userland: bool = False,
    ) -> None:
        if attr:
            if not attr.send_modified_events:
                return
            if is_userland and attr.key not in dict_:
                raise sa_exc.InvalidRequestError(
                    "Can't flag attribute '%s' modified; it's not present in "
                    "the object state" % attr.key
                )
            if attr.key not in self.committed_state or is_userland:
                if collection:
                    if TYPE_CHECKING:
                        assert is_collection_impl(attr)
                    if previous is NEVER_SET:
                        if attr.key in dict_:
                            previous = dict_[attr.key]

                    if previous not in (None, NO_VALUE, NEVER_SET):
                        previous = attr.copy(previous)
                self.committed_state[attr.key] = previous

            lkv = self._last_known_values
            if lkv is not None and attr.key in lkv:
                lkv[attr.key] = NO_VALUE

        # assert self._strong_obj is None or self.modified

        if (self.session_id and self._strong_obj is None) or not self.modified:
            self.modified = True
            instance_dict = self._instance_dict()
            if instance_dict:
                has_modified = bool(instance_dict._modified)
                instance_dict._modified.add(self)
            else:
                has_modified = False

            # only create _strong_obj link if attached
            # to a session

            inst = self.obj()
            if self.session_id:
                self._strong_obj = inst

                # if identity map already had modified objects,
                # assume autobegin already occurred, else check
                # for autobegin
                if not has_modified:
                    # inline of autobegin, to ensure session transaction
                    # snapshot is established
                    try:
                        session = _sessions[self.session_id]
                    except KeyError:
                        pass
                    else:
                        if session._transaction is None:
                            session._autobegin_t()

            if inst is None and attr:
                raise orm_exc.ObjectDereferencedError(
                    "Can't emit change event for attribute '%s' - "
                    "parent object of type %s has been garbage "
                    "collected."
                    % (self.manager[attr.key], base.state_class_str(self))
                )

    def _commit(self, dict_: _InstanceDict, keys: Iterable[str]) -> None:
        """Commit attributes.

        This is used by a partial-attribute load operation to mark committed
        those attributes which were refreshed from the database.

        Attributes marked as "expired" can potentially remain "expired" after
        this step if a value was not populated in state.dict.

        """
        for key in keys:
            self.committed_state.pop(key, None)

        self.expired = False

        self.expired_attributes.difference_update(
            set(keys).intersection(dict_)
        )

        # the per-keys commit removes object-level callables,
        # while that of commit_all does not.  it's not clear
        # if this behavior has a clear rationale, however tests do
        # ensure this is what it does.
        if self.callables:
            for key in (
                set(self.callables).intersection(keys).intersection(dict_)
            ):
                del self.callables[key]

    def _commit_all(
        self, dict_: _InstanceDict, instance_dict: Optional[IdentityMap] = None
    ) -> None:
        """commit all attributes unconditionally.

        This is used after a flush() or a full load/refresh
        to remove all pending state from the instance.

         - all attributes are marked as "committed"
         - the "strong dirty reference" is removed
         - the "modified" flag is set to False
         - any "expired" markers for scalar attributes loaded are removed.
         - lazy load callables for objects / collections *stay*

        Attributes marked as "expired" can potentially remain
        "expired" after this step if a value was not populated in state.dict.

        """
        self._commit_all_states([(self, dict_)], instance_dict)

    @classmethod
    def _commit_all_states(
        self,
        iter_: Iterable[Tuple[InstanceState[Any], _InstanceDict]],
        instance_dict: Optional[IdentityMap] = None,
    ) -> None:
        """Mass / highly inlined version of commit_all()."""

        for state, dict_ in iter_:
            state_dict = state.__dict__

            state.committed_state.clear()

            if "_pending_mutations" in state_dict:
                del state_dict["_pending_mutations"]

            state.expired_attributes.difference_update(dict_)

            if instance_dict and state.modified:
                instance_dict._modified.discard(state)

            state.modified = state.expired = False
            state._strong_obj = None


class AttributeState:
    """Provide an inspection interface corresponding
    to a particular attribute on a particular mapped object.

    The :class:`.AttributeState` object is accessed
    via the :attr:`.InstanceState.attrs` collection
    of a particular :class:`.InstanceState`::

        from sqlalchemy import inspect

        insp = inspect(some_mapped_object)
        attr_state = insp.attrs.some_attribute

    """

    __slots__ = ("state", "key")

    state: InstanceState[Any]
    key: str

    def __init__(self, state: InstanceState[Any], key: str):
        self.state = state
        self.key = key

    @property
    def loaded_value(self) -> Any:
        """The current value of this attribute as loaded from the database.

        If the value has not been loaded, or is otherwise not present
        in the object's dictionary, returns NO_VALUE.

        """
        return self.state.dict.get(self.key, NO_VALUE)

    @property
    def value(self) -> Any:
        """Return the value of this attribute.

        This operation is equivalent to accessing the object's
        attribute directly or via ``getattr()``, and will fire
        off any pending loader callables if needed.

        """
        return self.state.manager[self.key].__get__(
            self.state.obj(), self.state.class_
        )

    @property
    def history(self) -> History:
        """Return the current **pre-flush** change history for
        this attribute, via the :class:`.History` interface.

        This method will **not** emit loader callables if the value of the
        attribute is unloaded.

        .. note::

            The attribute history system tracks changes on a **per flush
            basis**. Each time the :class:`.Session` is flushed, the history
            of each attribute is reset to empty.   The :class:`.Session` by
            default autoflushes each time a :class:`_query.Query` is invoked.
            For
            options on how to control this, see :ref:`session_flushing`.


        .. seealso::

            :meth:`.AttributeState.load_history` - retrieve history
            using loader callables if the value is not locally present.

            :func:`.attributes.get_history` - underlying function

        """
        return self.state.get_history(self.key, PASSIVE_NO_INITIALIZE)

    def load_history(self) -> History:
        """Return the current **pre-flush** change history for
        this attribute, via the :class:`.History` interface.

        This method **will** emit loader callables if the value of the
        attribute is unloaded.

        .. note::

            The attribute history system tracks changes on a **per flush
            basis**. Each time the :class:`.Session` is flushed, the history
            of each attribute is reset to empty.   The :class:`.Session` by
            default autoflushes each time a :class:`_query.Query` is invoked.
            For
            options on how to control this, see :ref:`session_flushing`.

        .. seealso::

            :attr:`.AttributeState.history`

            :func:`.attributes.get_history` - underlying function

        """
        return self.state.get_history(self.key, PASSIVE_OFF ^ INIT_OK)


class PendingCollection:
    """A writable placeholder for an unloaded collection.

    Stores items appended to and removed from a collection that has not yet
    been loaded. When the collection is loaded, the changes stored in
    PendingCollection are applied to it to produce the final result.

    """

    __slots__ = ("deleted_items", "added_items")

    deleted_items: util.IdentitySet
    added_items: util.OrderedIdentitySet

    def __init__(self) -> None:
        self.deleted_items = util.IdentitySet()
        self.added_items = util.OrderedIdentitySet()

    def merge_with_history(self, history: History) -> History:
        return history._merge(self.added_items, self.deleted_items)

    def append(self, value: Any) -> None:
        if value in self.deleted_items:
            self.deleted_items.remove(value)
        else:
            self.added_items.add(value)

    def remove(self, value: Any) -> None:
        if value in self.added_items:
            self.added_items.remove(value)
        else:
            self.deleted_items.add(value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_threads.py ===
from __future__ import annotations

import contextvars
import queue as stdlib_queue
import re
import sys
import threading
import time
import weakref
from functools import partial
from typing import (
    TYPE_CHECKING,
    NoReturn,
    TypeVar,
    Union,
)

import pytest
import sniffio

from .. import (
    CancelScope,
    CapacityLimiter,
    Event,
    _core,
    fail_after,
    move_on_after,
    sleep,
    sleep_forever,
)
from .._core._tests.test_ki import ki_self
from .._core._tests.tutil import slow
from .._threads import (
    active_thread_count,
    current_default_thread_limiter,
    from_thread_check_cancelled,
    from_thread_run,
    from_thread_run_sync,
    to_thread_run_sync,
    wait_all_threads_completed,
)
from ..testing import wait_all_tasks_blocked

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from outcome import Outcome

    from ..lowlevel import Task

RecordType = list[tuple[str, Union[threading.Thread, type[BaseException]]]]
T = TypeVar("T")


async def test_do_in_trio_thread() -> None:
    trio_thread = threading.current_thread()

    async def check_case(  # type: ignore[explicit-any]
        do_in_trio_thread: Callable[..., threading.Thread],
        fn: Callable[..., T | Awaitable[T]],
        expected: tuple[str, T],
        trio_token: _core.TrioToken | None = None,
    ) -> None:
        record: RecordType = []

        def threadfn() -> None:
            try:
                record.append(("start", threading.current_thread()))
                x = do_in_trio_thread(fn, record, trio_token=trio_token)
                record.append(("got", x))
            except BaseException as exc:
                print(exc)
                record.append(("error", type(exc)))

        child_thread = threading.Thread(target=threadfn, daemon=True)
        child_thread.start()
        while child_thread.is_alive():
            print("yawn")
            await sleep(0.01)
        assert record == [("start", child_thread), ("f", trio_thread), expected]

    token = _core.current_trio_token()

    def f1(record: RecordType) -> int:
        assert not _core.currently_ki_protected()
        record.append(("f", threading.current_thread()))
        return 2

    await check_case(from_thread_run_sync, f1, ("got", 2), trio_token=token)

    def f2(record: RecordType) -> NoReturn:
        assert not _core.currently_ki_protected()
        record.append(("f", threading.current_thread()))
        raise ValueError

    await check_case(from_thread_run_sync, f2, ("error", ValueError), trio_token=token)

    async def f3(record: RecordType) -> int:
        assert not _core.currently_ki_protected()
        await _core.checkpoint()
        record.append(("f", threading.current_thread()))
        return 3

    await check_case(from_thread_run, f3, ("got", 3), trio_token=token)

    async def f4(record: RecordType) -> NoReturn:
        assert not _core.currently_ki_protected()
        await _core.checkpoint()
        record.append(("f", threading.current_thread()))
        raise KeyError

    await check_case(from_thread_run, f4, ("error", KeyError), trio_token=token)


async def test_do_in_trio_thread_from_trio_thread() -> None:
    with pytest.raises(RuntimeError):
        from_thread_run_sync(lambda: None)  # pragma: no branch

    async def foo() -> None:  # pragma: no cover
        pass

    with pytest.raises(RuntimeError):
        from_thread_run(foo)


def test_run_in_trio_thread_ki() -> None:
    # if we get a control-C during a run_in_trio_thread, then it propagates
    # back to the caller (slick!)
    record = set()

    async def check_run_in_trio_thread() -> None:
        token = _core.current_trio_token()

        def trio_thread_fn() -> None:
            print("in Trio thread")
            assert not _core.currently_ki_protected()
            print("ki_self")
            try:
                ki_self()
            finally:
                import sys

                print("finally", sys.exc_info())

        async def trio_thread_afn() -> None:
            trio_thread_fn()

        def external_thread_fn() -> None:
            try:
                print("running")
                from_thread_run_sync(trio_thread_fn, trio_token=token)
            except KeyboardInterrupt:
                print("ok1")
                record.add("ok1")
            try:
                from_thread_run(trio_thread_afn, trio_token=token)
            except KeyboardInterrupt:
                print("ok2")
                record.add("ok2")

        thread = threading.Thread(target=external_thread_fn)
        thread.start()
        print("waiting")
        while thread.is_alive():  # noqa: ASYNC110
            await sleep(0.01)  # Fine to poll in tests.
        print("waited, joining")
        thread.join()
        print("done")

    _core.run(check_run_in_trio_thread)
    assert record == {"ok1", "ok2"}


def test_await_in_trio_thread_while_main_exits() -> None:
    record = []
    ev = Event()

    async def trio_fn() -> None:
        record.append("sleeping")
        ev.set()
        await _core.wait_task_rescheduled(lambda _: _core.Abort.SUCCEEDED)

    def thread_fn(token: _core.TrioToken) -> None:
        try:
            from_thread_run(trio_fn, trio_token=token)
        except _core.Cancelled:
            record.append("cancelled")

    async def main() -> threading.Thread:
        token = _core.current_trio_token()
        thread = threading.Thread(target=thread_fn, args=(token,))
        thread.start()
        await ev.wait()
        assert record == ["sleeping"]
        return thread

    thread = _core.run(main)
    thread.join()
    assert record == ["sleeping", "cancelled"]


async def test_named_thread() -> None:
    ending = " from trio._tests.test_threads.test_named_thread"

    def inner(name: str = "inner" + ending) -> threading.Thread:
        assert threading.current_thread().name == name
        return threading.current_thread()

    def f(name: str) -> Callable[[], threading.Thread]:
        return partial(inner, name)

    # test defaults
    await to_thread_run_sync(inner)
    await to_thread_run_sync(inner, thread_name=None)

    # functools.partial doesn't have __name__, so defaults to None
    await to_thread_run_sync(f("None" + ending))

    # test that you can set a custom name, and that it's reset afterwards
    async def test_thread_name(name: str) -> None:
        thread = await to_thread_run_sync(f(name), thread_name=name)
        assert re.match(r"Trio thread [0-9]*", thread.name)

    await test_thread_name("")
    await test_thread_name("fobiedoo")
    await test_thread_name("name_longer_than_15_characters")

    await test_thread_name("💙")


def _get_thread_name(ident: int | None = None) -> str | None:
    import ctypes
    import ctypes.util

    libpthread_path = ctypes.util.find_library("pthread")
    if not libpthread_path:
        # musl includes pthread functions directly in libc.so
        # (but note that find_library("c") does not work on musl,
        #  see: https://github.com/python/cpython/issues/65821)
        # so try that library instead
        # if it doesn't exist, CDLL() will fail below
        libpthread_path = "libc.so"
    try:
        libpthread = ctypes.CDLL(libpthread_path)
    except Exception:
        print(f"no pthread on {sys.platform}")
        return None

    pthread_getname_np = getattr(libpthread, "pthread_getname_np", None)

    # this should never fail on any platforms afaik
    assert pthread_getname_np

    # thankfully getname signature doesn't differ between platforms
    pthread_getname_np.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    pthread_getname_np.restype = ctypes.c_int

    name_buffer = ctypes.create_string_buffer(b"", size=16)
    if ident is None:
        ident = threading.get_ident()
    assert pthread_getname_np(ident, name_buffer, 16) == 0
    try:
        return name_buffer.value.decode()
    except UnicodeDecodeError as e:  # pragma: no cover
        # used for debugging when testing via CI
        pytest.fail(f"value: {name_buffer.value!r}, exception: {e}")


# test os thread naming
# this depends on pthread being available, which is the case on 99.9% of linux machines
# and most mac machines. So unless the platform is linux it will just skip
# in case it fails to fetch the os thread name.
async def test_named_thread_os() -> None:
    def inner(name: str) -> threading.Thread:
        os_thread_name = _get_thread_name()
        if os_thread_name is None and sys.platform != "linux":
            pytest.skip(f"no pthread OS support on {sys.platform}")
        else:
            assert os_thread_name == name[:15]

        return threading.current_thread()

    def f(name: str) -> Callable[[], threading.Thread]:
        return partial(inner, name)

    # test defaults
    default = "None from trio._tests.test_threads.test_named_thread"
    await to_thread_run_sync(f(default))
    await to_thread_run_sync(f(default), thread_name=None)

    # test that you can set a custom name, and that it's reset afterwards
    async def test_thread_name(name: str, expected: str | None = None) -> None:
        if expected is None:
            expected = name
        thread = await to_thread_run_sync(f(expected), thread_name=name)

        os_thread_name = _get_thread_name(thread.ident)
        assert os_thread_name is not None, "should skip earlier if this is the case"
        assert re.match(r"Trio thread [0-9]*", os_thread_name)

    await test_thread_name("")
    await test_thread_name("fobiedoo")
    await test_thread_name("name_longer_than_15_characters")

    await test_thread_name("💙", expected="?")


def test_has_pthread_setname_np() -> None:
    from trio._core._thread_cache import get_os_thread_name_func

    k = get_os_thread_name_func()
    if k is None:
        assert sys.platform != "linux"
        pytest.skip(f"no pthread_setname_np on {sys.platform}")


async def test_run_in_worker_thread() -> None:
    trio_thread = threading.current_thread()

    def f(x: T) -> tuple[T, threading.Thread]:
        return (x, threading.current_thread())

    x, child_thread = await to_thread_run_sync(f, 1)
    assert x == 1
    assert child_thread != trio_thread

    def g() -> NoReturn:
        raise ValueError(threading.current_thread())

    with pytest.raises(
        ValueError,
        match=r"^<Thread\(Trio thread \d+, started daemon \d+\)>$",
    ) as excinfo:
        await to_thread_run_sync(g)
    print(excinfo.value.args)
    assert excinfo.value.args[0] != trio_thread


async def test_run_in_worker_thread_cancellation() -> None:
    register: list[str | None] = [None]

    def f(q: stdlib_queue.Queue[None]) -> None:
        # Make the thread block for a controlled amount of time
        register[0] = "blocking"
        q.get()
        register[0] = "finished"

    async def child(q: stdlib_queue.Queue[None], abandon_on_cancel: bool) -> None:
        record.append("start")
        try:
            return await to_thread_run_sync(f, q, abandon_on_cancel=abandon_on_cancel)
        finally:
            record.append("exit")

    record: list[str] = []
    q: stdlib_queue.Queue[None] = stdlib_queue.Queue()
    async with _core.open_nursery() as nursery:
        nursery.start_soon(child, q, True)
        # Give it a chance to get started. (This is important because
        # to_thread_run_sync does a checkpoint_if_cancelled before
        # blocking on the thread, and we don't want to trigger this.)
        await wait_all_tasks_blocked()
        assert record == ["start"]
        # Then cancel it.
        nursery.cancel_scope.cancel()
    # The task exited, but the thread didn't:
    assert register[0] != "finished"
    # Put the thread out of its misery:
    q.put(None)
    while register[0] != "finished":
        time.sleep(0.01)  # noqa: ASYNC251  # Need to wait for OS thread

    # This one can't be cancelled
    record = []
    register[0] = None
    async with _core.open_nursery() as nursery:
        nursery.start_soon(child, q, False)
        await wait_all_tasks_blocked()
        nursery.cancel_scope.cancel()
        with _core.CancelScope(shield=True):
            for _ in range(10):
                await _core.checkpoint()
        # It's still running
        assert record == ["start"]
        q.put(None)
        # Now it exits

    # But if we cancel *before* it enters, the entry is itself a cancellation
    # point
    with _core.CancelScope() as scope:
        scope.cancel()
        await child(q, False)
    assert scope.cancelled_caught


# Make sure that if trio.run exits, and then the thread finishes, then that's
# handled gracefully. (Requires that the thread result machinery be prepared
# for call_soon to raise RunFinishedError.)
def test_run_in_worker_thread_abandoned(
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_core._thread_cache, "IDLE_TIMEOUT", 0.01)

    q1: stdlib_queue.Queue[None] = stdlib_queue.Queue()
    q2: stdlib_queue.Queue[threading.Thread] = stdlib_queue.Queue()

    def thread_fn() -> None:
        q1.get()
        q2.put(threading.current_thread())

    async def main() -> None:
        async def child() -> None:
            await to_thread_run_sync(thread_fn, abandon_on_cancel=True)

        async with _core.open_nursery() as nursery:
            nursery.start_soon(child)
            await wait_all_tasks_blocked()
            nursery.cancel_scope.cancel()

    _core.run(main)

    q1.put(None)
    # This makes sure:
    # - the thread actually ran
    # - that thread has finished before we check for its output
    thread = q2.get()
    while thread.is_alive():
        time.sleep(0.01)  # pragma: no cover

    # Make sure we don't have a "Exception in thread ..." dump to the console:
    out, err = capfd.readouterr()
    assert "Exception in thread" not in out
    assert "Exception in thread" not in err


@pytest.mark.parametrize("MAX", [3, 5, 10])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("use_default_limiter", [False, True])
async def test_run_in_worker_thread_limiter(
    MAX: int,
    cancel: bool,
    use_default_limiter: bool,
) -> None:
    # This test is a bit tricky. The goal is to make sure that if we set
    # limiter=CapacityLimiter(MAX), then in fact only MAX threads are ever
    # running at a time, even if there are more concurrent calls to
    # to_thread_run_sync, and even if some of those are cancelled. And
    # also to make sure that the default limiter actually limits.
    COUNT = 2 * MAX
    gate = threading.Event()
    lock = threading.Lock()
    if use_default_limiter:
        c = current_default_thread_limiter()
        orig_total_tokens = c.total_tokens
        c.total_tokens = MAX
        limiter_arg = None
    else:
        c = CapacityLimiter(MAX)
        orig_total_tokens = MAX
        limiter_arg = c
    try:
        # We used to use regular variables and 'nonlocal' here, but it turns
        # out that it's not safe to assign to closed-over variables that are
        # visible in multiple threads, at least as of CPython 3.10 and PyPy
        # 7.3:
        #
        #   https://bugs.python.org/issue30744
        #   https://bitbucket.org/pypy/pypy/issues/2591/
        #
        # Mutating them in-place is OK though (as long as you use proper
        # locking etc.).
        class state:
            ran: int
            high_water: int
            running: int
            parked: int

        state.ran = 0
        state.high_water = 0
        state.running = 0
        state.parked = 0

        token = _core.current_trio_token()

        def thread_fn(cancel_scope: CancelScope) -> None:
            print("thread_fn start")
            from_thread_run_sync(cancel_scope.cancel, trio_token=token)
            with lock:
                state.ran += 1
                state.running += 1
                state.high_water = max(state.high_water, state.running)
                # The Trio thread below watches this value and uses it as a
                # signal that all the stats calculations have finished.
                state.parked += 1
            gate.wait()
            with lock:
                state.parked -= 1
                state.running -= 1
            print("thread_fn exiting")

        async def run_thread(event: Event) -> None:
            with _core.CancelScope() as cancel_scope:
                await to_thread_run_sync(
                    thread_fn,
                    cancel_scope,
                    abandon_on_cancel=cancel,
                    limiter=limiter_arg,
                )
            print("run_thread finished, cancelled:", cancel_scope.cancelled_caught)
            event.set()

        async with _core.open_nursery() as nursery:
            print("spawning")
            events = []
            for _ in range(COUNT):
                events.append(Event())
                nursery.start_soon(run_thread, events[-1])
                await wait_all_tasks_blocked()
            # In the cancel case, we in particular want to make sure that the
            # cancelled tasks don't release the semaphore. So let's wait until
            # at least one of them has exited, and that everything has had a
            # chance to settle down from this, before we check that everyone
            # who's supposed to be waiting is waiting:
            if cancel:
                print("waiting for first cancellation to clear")
                await events[0].wait()
                await wait_all_tasks_blocked()
            # Then wait until the first MAX threads are parked in gate.wait(),
            # and the next MAX threads are parked on the semaphore, to make
            # sure no-one is sneaking past, and to make sure the high_water
            # check below won't fail due to scheduling issues. (It could still
            # fail if too many threads are let through here.)
            while (  # noqa: ASYNC110
                state.parked != MAX or c.statistics().tasks_waiting != MAX
            ):
                await sleep(0.01)  # pragma: no cover
            # Then release the threads
            gate.set()

        assert state.high_water == MAX

        if cancel:
            # Some threads might still be running; need to wait to them to
            # finish before checking that all threads ran. We can do this
            # using the CapacityLimiter.
            while c.borrowed_tokens > 0:  # noqa: ASYNC110
                await sleep(0.01)  # pragma: no cover

        assert state.ran == COUNT
        assert state.running == 0
    finally:
        c.total_tokens = orig_total_tokens


async def test_run_in_worker_thread_custom_limiter() -> None:
    # Basically just checking that we only call acquire_on_behalf_of and
    # release_on_behalf_of, since that's part of our documented API.
    record = []

    class CustomLimiter:
        async def acquire_on_behalf_of(self, borrower: Task) -> None:
            record.append("acquire")
            self._borrower = borrower

        def release_on_behalf_of(self, borrower: Task) -> None:
            record.append("release")
            assert borrower == self._borrower

    # TODO: should CapacityLimiter have an abc or protocol so users can modify it?
    # because currently it's `final` so writing code like this is not allowed.
    await to_thread_run_sync(lambda: None, limiter=CustomLimiter())  # type: ignore[arg-type]
    assert record == ["acquire", "release"]


async def test_run_in_worker_thread_limiter_error() -> None:
    record = []

    class BadCapacityLimiter:
        async def acquire_on_behalf_of(self, borrower: Task) -> None:
            record.append("acquire")

        def release_on_behalf_of(self, borrower: Task) -> NoReturn:
            record.append("release")
            raise ValueError("release on behalf")

    bs = BadCapacityLimiter()

    with pytest.raises(ValueError, match=r"^release on behalf$") as excinfo:
        await to_thread_run_sync(lambda: None, limiter=bs)  # type: ignore[arg-type]
    assert excinfo.value.__context__ is None
    assert record == ["acquire", "release"]
    record = []

    # If the original function raised an error, then the semaphore error
    # chains with it
    d: dict[str, object] = {}
    with pytest.raises(ValueError, match=r"^release on behalf$") as excinfo:
        await to_thread_run_sync(lambda: d["x"], limiter=bs)  # type: ignore[arg-type]
    assert isinstance(excinfo.value.__context__, KeyError)
    assert record == ["acquire", "release"]


async def test_run_in_worker_thread_fail_to_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Test the unlikely but possible case where trying to spawn a thread fails
    def bad_start(self: object, *args: object) -> NoReturn:
        raise RuntimeError("the engines canna take it captain")

    monkeypatch.setattr(_core._thread_cache.ThreadCache, "start_thread_soon", bad_start)

    limiter = current_default_thread_limiter()
    assert limiter.borrowed_tokens == 0

    # We get an appropriate error, and the limiter is cleanly released
    with pytest.raises(RuntimeError) as excinfo:
        await to_thread_run_sync(lambda: None)  # pragma: no cover
    assert "engines" in str(excinfo.value)

    assert limiter.borrowed_tokens == 0


async def test_trio_to_thread_run_sync_token() -> None:
    # Test that to_thread_run_sync automatically injects the current trio token
    # into a spawned thread
    def thread_fn() -> _core.TrioToken:
        callee_token = from_thread_run_sync(_core.current_trio_token)
        return callee_token

    caller_token = _core.current_trio_token()
    callee_token = await to_thread_run_sync(thread_fn)
    assert callee_token == caller_token


async def test_trio_to_thread_run_sync_expected_error() -> None:
    # Test correct error when passed async function
    async def async_fn() -> None:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="expected a sync function"):
        await to_thread_run_sync(async_fn)  # type: ignore[unused-coroutine]


trio_test_contextvar: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trio_test_contextvar",
)


async def test_trio_to_thread_run_sync_contextvars() -> None:
    trio_thread = threading.current_thread()
    trio_test_contextvar.set("main")

    def f() -> tuple[str, threading.Thread]:
        value = trio_test_contextvar.get()
        with pytest.raises(sniffio.AsyncLibraryNotFoundError):
            sniffio.current_async_library()
        return (value, threading.current_thread())

    value, child_thread = await to_thread_run_sync(f)
    assert value == "main"
    assert child_thread != trio_thread

    def g() -> tuple[str, str, threading.Thread]:
        parent_value = trio_test_contextvar.get()
        trio_test_contextvar.set("worker")
        inner_value = trio_test_contextvar.get()
        with pytest.raises(sniffio.AsyncLibraryNotFoundError):
            sniffio.current_async_library()
        return (
            parent_value,
            inner_value,
            threading.current_thread(),
        )

    parent_value, inner_value, child_thread = await to_thread_run_sync(g)
    current_value = trio_test_contextvar.get()
    assert parent_value == "main"
    assert inner_value == "worker"
    assert current_value == "main", (
        "The contextvar value set on the worker would not propagate back to the main"
        " thread"
    )
    assert sniffio.current_async_library() == "trio"


async def test_trio_from_thread_run_sync() -> None:
    # Test that to_thread_run_sync correctly "hands off" the trio token to
    # trio.from_thread.run_sync()
    def thread_fn_1() -> float:
        trio_time = from_thread_run_sync(_core.current_time)
        return trio_time

    trio_time = await to_thread_run_sync(thread_fn_1)
    assert isinstance(trio_time, float)

    # Test correct error when passed async function
    async def async_fn() -> None:  # pragma: no cover
        pass

    def thread_fn_2() -> None:
        from_thread_run_sync(async_fn)  # type: ignore[unused-coroutine]

    with pytest.raises(TypeError, match="expected a synchronous function"):
        await to_thread_run_sync(thread_fn_2)


async def test_trio_from_thread_run() -> None:
    # Test that to_thread_run_sync correctly "hands off" the trio token to
    # trio.from_thread.run()
    record = []

    async def back_in_trio_fn() -> None:
        _core.current_time()  # implicitly checks that we're in trio
        record.append("back in trio")

    def thread_fn() -> None:
        record.append("in thread")
        from_thread_run(back_in_trio_fn)

    await to_thread_run_sync(thread_fn)
    assert record == ["in thread", "back in trio"]

    # Test correct error when passed sync function
    def sync_fn() -> None:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="appears to be synchronous"):
        await to_thread_run_sync(from_thread_run, sync_fn)  # type: ignore[arg-type]


async def test_trio_from_thread_token() -> None:
    # Test that to_thread_run_sync and spawned trio.from_thread.run_sync()
    # share the same Trio token
    def thread_fn() -> _core.TrioToken:
        callee_token = from_thread_run_sync(_core.current_trio_token)
        return callee_token

    caller_token = _core.current_trio_token()
    callee_token = await to_thread_run_sync(thread_fn)
    assert callee_token == caller_token


async def test_trio_from_thread_token_kwarg() -> None:
    # Test that to_thread_run_sync and spawned trio.from_thread.run_sync() can
    # use an explicitly defined token
    def thread_fn(token: _core.TrioToken) -> _core.TrioToken:
        callee_token = from_thread_run_sync(_core.current_trio_token, trio_token=token)
        return callee_token

    caller_token = _core.current_trio_token()
    callee_token = await to_thread_run_sync(thread_fn, caller_token)
    assert callee_token == caller_token


async def test_from_thread_no_token() -> None:
    # Test that a "raw call" to trio.from_thread.run() fails because no token
    # has been provided

    with pytest.raises(RuntimeError):
        from_thread_run_sync(_core.current_time)


async def test_trio_from_thread_run_sync_contextvars() -> None:
    trio_test_contextvar.set("main")

    def thread_fn() -> tuple[str, str, str, str, str]:
        thread_parent_value = trio_test_contextvar.get()
        trio_test_contextvar.set("worker")
        thread_current_value = trio_test_contextvar.get()
        with pytest.raises(sniffio.AsyncLibraryNotFoundError):
            sniffio.current_async_library()

        def back_in_main() -> tuple[str, str]:
            back_parent_value = trio_test_contextvar.get()
            trio_test_contextvar.set("back_in_main")
            back_current_value = trio_test_contextvar.get()
            assert sniffio.current_async_library() == "trio"
            return back_parent_value, back_current_value

        back_parent_value, back_current_value = from_thread_run_sync(back_in_main)
        thread_after_value = trio_test_contextvar.get()
        with pytest.raises(sniffio.AsyncLibraryNotFoundError):
            sniffio.current_async_library()
        return (
            thread_parent_value,
            thread_current_value,
            thread_after_value,
            back_parent_value,
            back_current_value,
        )

    (
        thread_parent_value,
        thread_current_value,
        thread_after_value,
        back_parent_value,
        back_current_value,
    ) = await to_thread_run_sync(thread_fn)
    current_value = trio_test_contextvar.get()
    assert current_value == thread_parent_value == "main"
    assert thread_current_value == back_parent_value == thread_after_value == "worker"
    assert sniffio.current_async_library() == "trio"
    assert back_current_value == "back_in_main"


async def test_trio_from_thread_run_contextvars() -> None:
    trio_test_contextvar.set("main")

    def thread_fn() -> tuple[str, str, str, str, str]:
        thread_parent_value = trio_test_contextvar.get()
        trio_test_contextvar.set("worker")
        thread_current_value = trio_test_contextvar.get()
        with pytest.raises(sniffio.AsyncLibraryNotFoundError):
            sniffio.current_async_library()

        async def async_back_in_main() -> tuple[str, str]:
            back_parent_value = trio_test_contextvar.get()
            trio_test_contextvar.set("back_in_main")
            back_current_value = trio_test_contextvar.get()
            assert sniffio.current_async_library() == "trio"
            return back_parent_value, back_current_value

        back_parent_value, back_current_value = from_thread_run(async_back_in_main)
        thread_after_value = trio_test_contextvar.get()
        with pytest.raises(sniffio.AsyncLibraryNotFoundError):
            sniffio.current_async_library()
        return (
            thread_parent_value,
            thread_current_value,
            thread_after_value,
            back_parent_value,
            back_current_value,
        )

    (
        thread_parent_value,
        thread_current_value,
        thread_after_value,
        back_parent_value,
        back_current_value,
    ) = await to_thread_run_sync(thread_fn)
    current_value = trio_test_contextvar.get()
    assert current_value == thread_parent_value == "main"
    assert thread_current_value == back_parent_value == thread_after_value == "worker"
    assert back_current_value == "back_in_main"
    assert sniffio.current_async_library() == "trio"


def test_run_fn_as_system_task_caught_badly_typed_token() -> None:
    with pytest.raises(RuntimeError):
        from_thread_run_sync(
            _core.current_time,
            trio_token="Not TrioTokentype",  # type: ignore[arg-type]
        )


async def test_from_thread_inside_trio_thread() -> None:
    def not_called() -> None:  # pragma: no cover
        raise AssertionError()

    trio_token = _core.current_trio_token()
    with pytest.raises(RuntimeError):
        from_thread_run_sync(not_called, trio_token=trio_token)


def test_from_thread_run_during_shutdown() -> None:
    save = []
    record = []

    async def agen(token: _core.TrioToken | None) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            with _core.CancelScope(shield=True):
                try:
                    await to_thread_run_sync(
                        partial(from_thread_run, sleep, 0, trio_token=token),
                    )
                except _core.RunFinishedError:
                    record.append("finished")
                else:
                    record.append("clean")

    async def main(use_system_task: bool) -> None:
        save.append(agen(_core.current_trio_token() if use_system_task else None))
        await save[-1].asend(None)

    _core.run(main, True)  # System nursery will be closed and raise RunFinishedError
    _core.run(main, False)  # host task will be rescheduled as normal
    assert record == ["finished", "clean"]


async def test_trio_token_weak_referenceable() -> None:
    token = _core.current_trio_token()
    assert isinstance(token, _core.TrioToken)
    weak_reference = weakref.ref(token)
    assert token is weak_reference()


async def test_unsafe_abandon_on_cancel_kwarg() -> None:
    # This is a stand in for a numpy ndarray or other objects
    # that (maybe surprisingly) lack a notion of truthiness
    class BadBool:
        def __bool__(self) -> bool:
            raise NotImplementedError

    with pytest.raises(NotImplementedError):
        await to_thread_run_sync(int, abandon_on_cancel=BadBool())  # type: ignore[arg-type]


async def test_from_thread_reuses_task() -> None:
    task = _core.current_task()

    async def async_current_task() -> _core.Task:
        return _core.current_task()

    assert task is await to_thread_run_sync(from_thread_run_sync, _core.current_task)
    assert task is await to_thread_run_sync(from_thread_run, async_current_task)


async def test_recursive_to_thread() -> None:
    tid = None

    def get_tid_then_reenter() -> int:
        nonlocal tid
        tid = threading.get_ident()
        return from_thread_run(to_thread_run_sync, threading.get_ident)

    assert tid != await to_thread_run_sync(get_tid_then_reenter)


async def test_from_thread_host_cancelled() -> None:
    queue: stdlib_queue.Queue[bool] = stdlib_queue.Queue()

    def sync_check() -> None:
        from_thread_run_sync(cancel_scope.cancel)
        try:
            from_thread_run_sync(bool)
        except _core.Cancelled:  # pragma: no cover
            queue.put(True)  # sync functions don't raise Cancelled
        else:
            queue.put(False)

    with _core.CancelScope() as cancel_scope:
        await to_thread_run_sync(sync_check)

    assert not cancel_scope.cancelled_caught
    assert not queue.get_nowait()

    with _core.CancelScope() as cancel_scope:
        await to_thread_run_sync(sync_check, abandon_on_cancel=True)

    assert cancel_scope.cancelled_caught
    assert not await to_thread_run_sync(partial(queue.get, timeout=1))

    async def no_checkpoint() -> bool:
        return True

    def async_check() -> None:
        from_thread_run_sync(cancel_scope.cancel)
        try:
            assert from_thread_run(no_checkpoint)
        except _core.Cancelled:  # pragma: no cover
            queue.put(True)  # async functions raise Cancelled at checkpoints
        else:
            queue.put(False)

    with _core.CancelScope() as cancel_scope:
        await to_thread_run_sync(async_check)

    assert not cancel_scope.cancelled_caught
    assert not queue.get_nowait()

    with _core.CancelScope() as cancel_scope:
        await to_thread_run_sync(async_check, abandon_on_cancel=True)

    assert cancel_scope.cancelled_caught
    assert not await to_thread_run_sync(partial(queue.get, timeout=1))

    async def async_time_bomb() -> None:
        cancel_scope.cancel()
        with fail_after(10):
            await sleep_forever()

    with _core.CancelScope() as cancel_scope:
        await to_thread_run_sync(from_thread_run, async_time_bomb)

    assert cancel_scope.cancelled_caught


async def test_from_thread_check_cancelled() -> None:
    q: stdlib_queue.Queue[str] = stdlib_queue.Queue()

    async def child(abandon_on_cancel: bool, scope: CancelScope) -> None:
        with scope:
            record.append("start")
            try:
                return await to_thread_run_sync(f, abandon_on_cancel=abandon_on_cancel)
            except _core.Cancelled:
                record.append("cancel")
                raise
            finally:
                record.append("exit")

    def f() -> None:
        try:
            from_thread_check_cancelled()
        except _core.Cancelled:  # pragma: no cover, test failure path
            q.put("Cancelled")
        else:
            q.put("Not Cancelled")
        ev.wait()
        return from_thread_check_cancelled()

    # Base case: nothing cancelled so we shouldn't see cancels anywhere
    record: list[str] = []
    ev = threading.Event()
    async with _core.open_nursery() as nursery:
        nursery.start_soon(child, False, _core.CancelScope())
        await wait_all_tasks_blocked()
        assert record[0] == "start"
        assert q.get(timeout=1) == "Not Cancelled"
        ev.set()
    # implicit assertion, Cancelled not raised via nursery
    assert record[1] == "exit"

    # abandon_on_cancel=False case: a cancel will pop out but be handled by
    # the appropriate cancel scope
    record = []
    ev = threading.Event()
    scope = _core.CancelScope()  # Nursery cancel scope gives false positives
    async with _core.open_nursery() as nursery:
        nursery.start_soon(child, False, scope)
        await wait_all_tasks_blocked()
        assert record[0] == "start"
        assert q.get(timeout=1) == "Not Cancelled"
        scope.cancel()
        ev.set()
    assert scope.cancelled_caught
    assert "cancel" in record
    assert record[-1] == "exit"

    # abandon_on_cancel=True case: slightly different thread behavior needed
    # check thread is cancelled "soon" after abandonment
    def f() -> None:  # type: ignore[no-redef] # noqa: F811
        ev.wait()
        try:
            from_thread_check_cancelled()
        except _core.Cancelled:
            q.put("Cancelled")
        else:  # pragma: no cover, test failure path
            q.put("Not Cancelled")

    record = []
    ev = threading.Event()
    scope = _core.CancelScope()
    async with _core.open_nursery() as nursery:
        nursery.start_soon(child, True, scope)
        await wait_all_tasks_blocked()
        assert record[0] == "start"
        scope.cancel()
        ev.set()
    assert scope.cancelled_caught
    assert "cancel" in record
    assert record[-1] == "exit"
    assert q.get(timeout=1) == "Cancelled"


def test_from_thread_check_cancelled_raises_in_foreign_threads() -> None:
    with pytest.raises(RuntimeError):
        from_thread_check_cancelled()
    q: stdlib_queue.Queue[Outcome[object]] = stdlib_queue.Queue()
    _core.start_thread_soon(from_thread_check_cancelled, lambda _: q.put(_))
    with pytest.raises(RuntimeError):
        q.get(timeout=1).unwrap()


@slow
async def test_reentry_doesnt_deadlock() -> None:
    # Regression test for issue noticed in GH-2827
    # The failure mode is to hang the whole test suite, unfortunately.
    # XXX consider running this in a subprocess with a timeout, if it comes up again!

    async def child() -> None:
        while True:
            await to_thread_run_sync(from_thread_run, sleep, 0, abandon_on_cancel=False)

    with move_on_after(2):
        async with _core.open_nursery() as nursery:
            for _ in range(4):
                nursery.start_soon(child)


async def test_wait_all_threads_completed() -> None:
    no_threads_left = False
    e1 = Event()
    e2 = Event()

    e1_exited = Event()
    e2_exited = Event()

    async def wait_event(e: Event, e_exit: Event) -> None:
        def thread() -> None:
            from_thread_run(e.wait)

        await to_thread_run_sync(thread)
        e_exit.set()

    async def wait_no_threads_left() -> None:
        nonlocal no_threads_left
        await wait_all_threads_completed()
        no_threads_left = True

    async with _core.open_nursery() as nursery:
        nursery.start_soon(wait_event, e1, e1_exited)
        nursery.start_soon(wait_event, e2, e2_exited)
        await wait_all_tasks_blocked()
        nursery.start_soon(wait_no_threads_left)
        await wait_all_tasks_blocked()
        assert not no_threads_left
        assert active_thread_count() == 2

        e1.set()
        await e1_exited.wait()
        await wait_all_tasks_blocked()
        assert not no_threads_left
        assert active_thread_count() == 1

        e2.set()
        await e2_exited.wait()
        await wait_all_tasks_blocked()
        assert no_threads_left
        assert active_thread_count() == 0


async def test_wait_all_threads_completed_no_threads() -> None:
    await wait_all_threads_completed()
    assert active_thread_count() == 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\connectionpool.py ===
from __future__ import absolute_import

import errno
import logging
import re
import socket
import sys
import warnings
from socket import error as SocketError
from socket import timeout as SocketTimeout

from ._collections import HTTPHeaderDict
from .connection import (
    BaseSSLError,
    BrokenPipeError,
    DummyConnection,
    HTTPConnection,
    HTTPException,
    HTTPSConnection,
    VerifiedHTTPSConnection,
    port_by_scheme,
)
from .exceptions import (
    ClosedPoolError,
    EmptyPoolError,
    HeaderParsingError,
    HostChangedError,
    InsecureRequestWarning,
    LocationValueError,
    MaxRetryError,
    NewConnectionError,
    ProtocolError,
    ProxyError,
    ReadTimeoutError,
    SSLError,
    TimeoutError,
)
from .packages import six
from .packages.six.moves import queue
from .request import RequestMethods
from .response import HTTPResponse
from .util.connection import is_connection_dropped
from .util.proxy import connection_requires_http_tunnel
from .util.queue import LifoQueue
from .util.request import set_file_position
from .util.response import assert_header_parsing
from .util.retry import Retry
from .util.ssl_match_hostname import CertificateError
from .util.timeout import Timeout
from .util.url import Url, _encode_target
from .util.url import _normalize_host as normalize_host
from .util.url import get_host, parse_url

try:  # Platform-specific: Python 3
    import weakref

    weakref_finalize = weakref.finalize
except AttributeError:  # Platform-specific: Python 2
    from .packages.backports.weakref_finalize import weakref_finalize

xrange = six.moves.xrange

log = logging.getLogger(__name__)

_Default = object()


# Pool objects
class ConnectionPool(object):
    """
    Base class for all connection pools, such as
    :class:`.HTTPConnectionPool` and :class:`.HTTPSConnectionPool`.

    .. note::
       ConnectionPool.urlopen() does not normalize or percent-encode target URIs
       which is useful if your target server doesn't support percent-encoded
       target URIs.
    """

    scheme = None
    QueueCls = LifoQueue

    def __init__(self, host, port=None):
        if not host:
            raise LocationValueError("No host specified.")

        self.host = _normalize_host(host, scheme=self.scheme)
        self._proxy_host = host.lower()
        self.port = port

    def __str__(self):
        return "%s(host=%r, port=%r)" % (type(self).__name__, self.host, self.port)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # Return False to re-raise any potential exceptions
        return False

    def close(self):
        """
        Close all pooled connections and disable the pool.
        """
        pass


# This is taken from http://hg.python.org/cpython/file/7aaba721ebc0/Lib/socket.py#l252
_blocking_errnos = {errno.EAGAIN, errno.EWOULDBLOCK}


class HTTPConnectionPool(ConnectionPool, RequestMethods):
    """
    Thread-safe connection pool for one host.

    :param host:
        Host used for this HTTP Connection (e.g. "localhost"), passed into
        :class:`http.client.HTTPConnection`.

    :param port:
        Port used for this HTTP Connection (None is equivalent to 80), passed
        into :class:`http.client.HTTPConnection`.

    :param strict:
        Causes BadStatusLine to be raised if the status line can't be parsed
        as a valid HTTP/1.0 or 1.1 status line, passed into
        :class:`http.client.HTTPConnection`.

        .. note::
           Only works in Python 2. This parameter is ignored in Python 3.

    :param timeout:
        Socket timeout in seconds for each individual connection. This can
        be a float or integer, which sets the timeout for the HTTP request,
        or an instance of :class:`urllib3.util.Timeout` which gives you more
        fine-grained control over request timeouts. After the constructor has
        been parsed, this is always a `urllib3.util.Timeout` object.

    :param maxsize:
        Number of connections to save that can be reused. More than 1 is useful
        in multithreaded situations. If ``block`` is set to False, more
        connections will be created but they will not be saved once they've
        been used.

    :param block:
        If set to True, no more than ``maxsize`` connections will be used at
        a time. When no free connections are available, the call will block
        until a connection has been released. This is a useful side effect for
        particular multithreaded situations where one does not want to use more
        than maxsize connections per host to prevent flooding.

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.

    :param retries:
        Retry configuration to use by default with requests in this pool.

    :param _proxy:
        Parsed proxy URL, should not be used directly, instead, see
        :class:`urllib3.ProxyManager`

    :param _proxy_headers:
        A dictionary with proxy headers, should not be used directly,
        instead, see :class:`urllib3.ProxyManager`

    :param \\**conn_kw:
        Additional parameters are used to create fresh :class:`urllib3.connection.HTTPConnection`,
        :class:`urllib3.connection.HTTPSConnection` instances.
    """

    scheme = "http"
    ConnectionCls = HTTPConnection
    ResponseCls = HTTPResponse

    def __init__(
        self,
        host,
        port=None,
        strict=False,
        timeout=Timeout.DEFAULT_TIMEOUT,
        maxsize=1,
        block=False,
        headers=None,
        retries=None,
        _proxy=None,
        _proxy_headers=None,
        _proxy_config=None,
        **conn_kw
    ):
        ConnectionPool.__init__(self, host, port)
        RequestMethods.__init__(self, headers)

        self.strict = strict

        if not isinstance(timeout, Timeout):
            timeout = Timeout.from_float(timeout)

        if retries is None:
            retries = Retry.DEFAULT

        self.timeout = timeout
        self.retries = retries

        self.pool = self.QueueCls(maxsize)
        self.block = block

        self.proxy = _proxy
        self.proxy_headers = _proxy_headers or {}
        self.proxy_config = _proxy_config

        # Fill the queue up so that doing get() on it will block properly
        for _ in xrange(maxsize):
            self.pool.put(None)

        # These are mostly for testing and debugging purposes.
        self.num_connections = 0
        self.num_requests = 0
        self.conn_kw = conn_kw

        if self.proxy:
            # Enable Nagle's algorithm for proxies, to avoid packet fragmentation.
            # We cannot know if the user has added default socket options, so we cannot replace the
            # list.
            self.conn_kw.setdefault("socket_options", [])

            self.conn_kw["proxy"] = self.proxy
            self.conn_kw["proxy_config"] = self.proxy_config

        # Do not pass 'self' as callback to 'finalize'.
        # Then the 'finalize' would keep an endless living (leak) to self.
        # By just passing a reference to the pool allows the garbage collector
        # to free self if nobody else has a reference to it.
        pool = self.pool

        # Close all the HTTPConnections in the pool before the
        # HTTPConnectionPool object is garbage collected.
        weakref_finalize(self, _close_pool_connections, pool)

    def _new_conn(self):
        """
        Return a fresh :class:`HTTPConnection`.
        """
        self.num_connections += 1
        log.debug(
            "Starting new HTTP connection (%d): %s:%s",
            self.num_connections,
            self.host,
            self.port or "80",
        )

        conn = self.ConnectionCls(
            host=self.host,
            port=self.port,
            timeout=self.timeout.connect_timeout,
            strict=self.strict,
            **self.conn_kw
        )
        return conn

    def _get_conn(self, timeout=None):
        """
        Get a connection. Will return a pooled connection if one is available.

        If no connections are available and :prop:`.block` is ``False``, then a
        fresh connection is returned.

        :param timeout:
            Seconds to wait before giving up and raising
            :class:`urllib3.exceptions.EmptyPoolError` if the pool is empty and
            :prop:`.block` is ``True``.
        """
        conn = None
        try:
            conn = self.pool.get(block=self.block, timeout=timeout)

        except AttributeError:  # self.pool is None
            raise ClosedPoolError(self, "Pool is closed.")

        except queue.Empty:
            if self.block:
                raise EmptyPoolError(
                    self,
                    "Pool reached maximum size and no more connections are allowed.",
                )
            pass  # Oh well, we'll create a new connection then

        # If this is a persistent connection, check if it got disconnected
        if conn and is_connection_dropped(conn):
            log.debug("Resetting dropped connection: %s", self.host)
            conn.close()
            if getattr(conn, "auto_open", 1) == 0:
                # This is a proxied connection that has been mutated by
                # http.client._tunnel() and cannot be reused (since it would
                # attempt to bypass the proxy)
                conn = None

        return conn or self._new_conn()

    def _put_conn(self, conn):
        """
        Put a connection back into the pool.

        :param conn:
            Connection object for the current host and port as returned by
            :meth:`._new_conn` or :meth:`._get_conn`.

        If the pool is already full, the connection is closed and discarded
        because we exceeded maxsize. If connections are discarded frequently,
        then maxsize should be increased.

        If the pool is closed, then the connection will be closed and discarded.
        """
        try:
            self.pool.put(conn, block=False)
            return  # Everything is dandy, done.
        except AttributeError:
            # self.pool is None.
            pass
        except queue.Full:
            # This should never happen if self.block == True
            log.warning(
                "Connection pool is full, discarding connection: %s. Connection pool size: %s",
                self.host,
                self.pool.qsize(),
            )
        # Connection never got put back into the pool, close it.
        if conn:
            conn.close()

    def _validate_conn(self, conn):
        """
        Called right before a request is made, after the socket is created.
        """
        pass

    def _prepare_proxy(self, conn):
        # Nothing to do for HTTP connections.
        pass

    def _get_timeout(self, timeout):
        """Helper that always returns a :class:`urllib3.util.Timeout`"""
        if timeout is _Default:
            return self.timeout.clone()

        if isinstance(timeout, Timeout):
            return timeout.clone()
        else:
            # User passed us an int/float. This is for backwards compatibility,
            # can be removed later
            return Timeout.from_float(timeout)

    def _raise_timeout(self, err, url, timeout_value):
        """Is the error actually a timeout? Will raise a ReadTimeout or pass"""

        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(
                self, url, "Read timed out. (read timeout=%s)" % timeout_value
            )

        # See the above comment about EAGAIN in Python 3. In Python 2 we have
        # to specifically catch it and throw the timeout error
        if hasattr(err, "errno") and err.errno in _blocking_errnos:
            raise ReadTimeoutError(
                self, url, "Read timed out. (read timeout=%s)" % timeout_value
            )

        # Catch possible read timeouts thrown as SSL errors. If not the
        # case, rethrow the original. We need to do this because of:
        # http://bugs.python.org/issue10272
        if "timed out" in str(err) or "did not complete (read)" in str(
            err
        ):  # Python < 2.7.4
            raise ReadTimeoutError(
                self, url, "Read timed out. (read timeout=%s)" % timeout_value
            )

    def _make_request(
        self, conn, method, url, timeout=_Default, chunked=False, **httplib_request_kw
    ):
        """
        Perform a request on a given urllib connection object taken from our
        pool.

        :param conn:
            a connection from one of our connection pools

        :param timeout:
            Socket timeout in seconds for the request. This can be a
            float or integer, which will set the same timeout value for
            the socket connect and the socket read, or an instance of
            :class:`urllib3.util.Timeout`, which gives you more fine-grained
            control over your timeouts.
        """
        self.num_requests += 1

        timeout_obj = self._get_timeout(timeout)
        timeout_obj.start_connect()
        conn.timeout = Timeout.resolve_default_timeout(timeout_obj.connect_timeout)

        # Trigger any extra validation we need to do.
        try:
            self._validate_conn(conn)
        except (SocketTimeout, BaseSSLError) as e:
            # Py2 raises this as a BaseSSLError, Py3 raises it as socket timeout.
            self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)
            raise

        # conn.request() calls http.client.*.request, not the method in
        # urllib3.request. It also calls makefile (recv) on the socket.
        try:
            if chunked:
                conn.request_chunked(method, url, **httplib_request_kw)
            else:
                conn.request(method, url, **httplib_request_kw)

        # We are swallowing BrokenPipeError (errno.EPIPE) since the server is
        # legitimately able to close the connection after sending a valid response.
        # With this behaviour, the received response is still readable.
        except BrokenPipeError:
            # Python 3
            pass
        except IOError as e:
            # Python 2 and macOS/Linux
            # EPIPE and ESHUTDOWN are BrokenPipeError on Python 2, and EPROTOTYPE/ECONNRESET are needed on macOS
            # https://erickt.github.io/blog/2014/11/19/adventures-in-debugging-a-potential-osx-kernel-bug/
            if e.errno not in {
                errno.EPIPE,
                errno.ESHUTDOWN,
                errno.EPROTOTYPE,
                errno.ECONNRESET,
            }:
                raise

        # Reset the timeout for the recv() on the socket
        read_timeout = timeout_obj.read_timeout

        # App Engine doesn't have a sock attr
        if getattr(conn, "sock", None):
            # In Python 3 socket.py will catch EAGAIN and return None when you
            # try and read into the file pointer created by http.client, which
            # instead raises a BadStatusLine exception. Instead of catching
            # the exception and assuming all BadStatusLine exceptions are read
            # timeouts, check for a zero timeout before making the request.
            if read_timeout == 0:
                raise ReadTimeoutError(
                    self, url, "Read timed out. (read timeout=%s)" % read_timeout
                )
            if read_timeout is Timeout.DEFAULT_TIMEOUT:
                conn.sock.settimeout(socket.getdefaulttimeout())
            else:  # None or a value
                conn.sock.settimeout(read_timeout)

        # Receive the response from the server
        try:
            try:
                # Python 2.7, use buffering of HTTP responses
                httplib_response = conn.getresponse(buffering=True)
            except TypeError:
                # Python 3
                try:
                    httplib_response = conn.getresponse()
                except BaseException as e:
                    # Remove the TypeError from the exception chain in
                    # Python 3 (including for exceptions like SystemExit).
                    # Otherwise it looks like a bug in the code.
                    six.raise_from(e, None)
        except (SocketTimeout, BaseSSLError, SocketError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
            raise

        # AppEngine doesn't have a version attr.
        http_version = getattr(conn, "_http_vsn_str", "HTTP/?")
        log.debug(
            '%s://%s:%s "%s %s %s" %s %s',
            self.scheme,
            self.host,
            self.port,
            method,
            url,
            http_version,
            httplib_response.status,
            httplib_response.length,
        )

        try:
            assert_header_parsing(httplib_response.msg)
        except (HeaderParsingError, TypeError) as hpe:  # Platform-specific: Python 3
            log.warning(
                "Failed to parse headers (url=%s): %s",
                self._absolute_url(url),
                hpe,
                exc_info=True,
            )

        return httplib_response

    def _absolute_url(self, path):
        return Url(scheme=self.scheme, host=self.host, port=self.port, path=path).url

    def close(self):
        """
        Close all pooled connections and disable the pool.
        """
        if self.pool is None:
            return
        # Disable access to the pool
        old_pool, self.pool = self.pool, None

        # Close all the HTTPConnections in the pool.
        _close_pool_connections(old_pool)

    def is_same_host(self, url):
        """
        Check if the given ``url`` is a member of the same host as this
        connection pool.
        """
        if url.startswith("/"):
            return True

        # TODO: Add optional support for socket.gethostbyname checking.
        scheme, host, port = get_host(url)
        if host is not None:
            host = _normalize_host(host, scheme=scheme)

        # Use explicit default port for comparison when none is given
        if self.port and not port:
            port = port_by_scheme.get(scheme)
        elif not self.port and port == port_by_scheme.get(scheme):
            port = None

        return (scheme, host, port) == (self.scheme, self.host, self.port)

    def urlopen(
        self,
        method,
        url,
        body=None,
        headers=None,
        retries=None,
        redirect=True,
        assert_same_host=True,
        timeout=_Default,
        pool_timeout=None,
        release_conn=None,
        chunked=False,
        body_pos=None,
        **response_kw
    ):
        """
        Get a connection from the pool and perform an HTTP request. This is the
        lowest level call for making a request, so you'll need to specify all
        the raw details.

        .. note::

           More commonly, it's appropriate to use a convenience method provided
           by :class:`.RequestMethods`, such as :meth:`request`.

        .. note::

           `release_conn` will only behave as expected if
           `preload_content=False` because we want to make
           `preload_content=False` the default behaviour someday soon without
           breaking backwards compatibility.

        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)

        :param url:
            The URL to perform the request on.

        :param body:
            Data to send in the request body, either :class:`str`, :class:`bytes`,
            an iterable of :class:`str`/:class:`bytes`, or a file-like object.

        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.

        :param retries:
            Configure the number of retries to allow before raising a
            :class:`~urllib3.exceptions.MaxRetryError` exception.

            Pass ``None`` to retry until you receive a response. Pass a
            :class:`~urllib3.util.retry.Retry` object for fine-grained control
            over different types of retries.
            Pass an integer number to retry connection errors that many times,
            but no other types of errors. Pass zero to never retry.

            If ``False``, then retries are disabled and any exception is raised
            immediately. Also, instead of raising a MaxRetryError on redirects,
            the redirect response will be returned.

        :type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.

        :param redirect:
            If True, automatically handle redirects (status codes 301, 302,
            303, 307, 308). Each redirect counts as a retry. Disabling retries
            will disable redirect, too.

        :param assert_same_host:
            If ``True``, will make sure that the host of the pool requests is
            consistent else will raise HostChangedError. When ``False``, you can
            use the pool on an HTTP proxy and request foreign hosts.

        :param timeout:
            If specified, overrides the default timeout for this one
            request. It may be a float (in seconds) or an instance of
            :class:`urllib3.util.Timeout`.

        :param pool_timeout:
            If set and the pool is set to block=True, then this method will
            block for ``pool_timeout`` seconds and raise EmptyPoolError if no
            connection is available within the time period.

        :param release_conn:
            If False, then the urlopen call will not release the connection
            back into the pool once a response is received (but will release if
            you read the entire contents of the response such as when
            `preload_content=True`). This is useful if you're not preloading
            the response's content immediately. You will need to call
            ``r.release_conn()`` on the response ``r`` to return the connection
            back into the pool. If None, it takes the value of
            ``response_kw.get('preload_content', True)``.

        :param chunked:
            If True, urllib3 will send the body using chunked transfer
            encoding. Otherwise, urllib3 will send the body using the standard
            content-length form. Defaults to False.

        :param int body_pos:
            Position to seek to in file-like body in the event of a retry or
            redirect. Typically this won't need to be set because urllib3 will
            auto-populate the value when needed.

        :param \\**response_kw:
            Additional parameters are passed to
            :meth:`urllib3.response.HTTPResponse.from_httplib`
        """

        parsed_url = parse_url(url)
        destination_scheme = parsed_url.scheme

        if headers is None:
            headers = self.headers

        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)

        if release_conn is None:
            release_conn = response_kw.get("preload_content", True)

        # Check host
        if assert_same_host and not self.is_same_host(url):
            raise HostChangedError(self, url, retries)

        # Ensure that the URL we're connecting to is properly encoded
        if url.startswith("/"):
            url = six.ensure_str(_encode_target(url))
        else:
            url = six.ensure_str(parsed_url.url)

        conn = None

        # Track whether `conn` needs to be released before
        # returning/raising/recursing. Update this variable if necessary, and
        # leave `release_conn` constant throughout the function. That way, if
        # the function recurses, the original value of `release_conn` will be
        # passed down into the recursive call, and its value will be respected.
        #
        # See issue #651 [1] for details.
        #
        # [1] <https://github.com/urllib3/urllib3/issues/651>
        release_this_conn = release_conn

        http_tunnel_required = connection_requires_http_tunnel(
            self.proxy, self.proxy_config, destination_scheme
        )

        # Merge the proxy headers. Only done when not using HTTP CONNECT. We
        # have to copy the headers dict so we can safely change it without those
        # changes being reflected in anyone else's copy.
        if not http_tunnel_required:
            headers = headers.copy()
            headers.update(self.proxy_headers)

        # Must keep the exception bound to a separate variable or else Python 3
        # complains about UnboundLocalError.
        err = None

        # Keep track of whether we cleanly exited the except block. This
        # ensures we do proper cleanup in finally.
        clean_exit = False

        # Rewind body position, if needed. Record current position
        # for future rewinds in the event of a redirect/retry.
        body_pos = set_file_position(body, body_pos)

        try:
            # Request a connection from the queue.
            timeout_obj = self._get_timeout(timeout)
            conn = self._get_conn(timeout=pool_timeout)

            conn.timeout = timeout_obj.connect_timeout

            is_new_proxy_conn = self.proxy is not None and not getattr(
                conn, "sock", None
            )
            if is_new_proxy_conn and http_tunnel_required:
                self._prepare_proxy(conn)

            # Make the request on the httplib connection object.
            httplib_response = self._make_request(
                conn,
                method,
                url,
                timeout=timeout_obj,
                body=body,
                headers=headers,
                chunked=chunked,
            )

            # If we're going to release the connection in ``finally:``, then
            # the response doesn't need to know about the connection. Otherwise
            # it will also try to release it and we'll have a double-release
            # mess.
            response_conn = conn if not release_conn else None

            # Pass method to Response for length checking
            response_kw["request_method"] = method

            # Import httplib's response into our own wrapper object
            response = self.ResponseCls.from_httplib(
                httplib_response,
                pool=self,
                connection=response_conn,
                retries=retries,
                **response_kw
            )

            # Everything went great!
            clean_exit = True

        except EmptyPoolError:
            # Didn't get a connection from the pool, no need to clean up
            clean_exit = True
            release_this_conn = False
            raise

        except (
            TimeoutError,
            HTTPException,
            SocketError,
            ProtocolError,
            BaseSSLError,
            SSLError,
            CertificateError,
        ) as e:
            # Discard the connection for these exceptions. It will be
            # replaced during the next _get_conn() call.
            clean_exit = False

            def _is_ssl_error_message_from_http_proxy(ssl_error):
                # We're trying to detect the message 'WRONG_VERSION_NUMBER' but
                # SSLErrors are kinda all over the place when it comes to the message,
                # so we try to cover our bases here!
                message = " ".join(re.split("[^a-z]", str(ssl_error).lower()))
                return (
                    "wrong version number" in message
                    or "unknown protocol" in message
                    or "record layer failure" in message
                )

            # Try to detect a common user error with proxies which is to
            # set an HTTP proxy to be HTTPS when it should be 'http://'
            # (ie {'http': 'http://proxy', 'https': 'https://proxy'})
            # Instead we add a nice error message and point to a URL.
            if (
                isinstance(e, BaseSSLError)
                and self.proxy
                and _is_ssl_error_message_from_http_proxy(e)
                and conn.proxy
                and conn.proxy.scheme == "https"
            ):
                e = ProxyError(
                    "Your proxy appears to only use HTTP and not HTTPS, "
                    "try changing your proxy URL to be HTTP. See: "
                    "https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html"
                    "#https-proxy-error-http-proxy",
                    SSLError(e),
                )
            elif isinstance(e, (BaseSSLError, CertificateError)):
                e = SSLError(e)
            elif isinstance(e, (SocketError, NewConnectionError)) and self.proxy:
                e = ProxyError("Cannot connect to proxy.", e)
            elif isinstance(e, (SocketError, HTTPException)):
                e = ProtocolError("Connection aborted.", e)

            retries = retries.increment(
                method, url, error=e, _pool=self, _stacktrace=sys.exc_info()[2]
            )
            retries.sleep()

            # Keep track of the error for the retry warning.
            err = e

        finally:
            if not clean_exit:
                # We hit some kind of exception, handled or otherwise. We need
                # to throw the connection away unless explicitly told not to.
                # Close the connection, set the variable to None, and make sure
                # we put the None back in the pool to avoid leaking it.
                conn = conn and conn.close()
                release_this_conn = True

            if release_this_conn:
                # Put the connection back to be reused. If the connection is
                # expired then it will be None, which will get replaced with a
                # fresh connection during _get_conn.
                self._put_conn(conn)

        if not conn:
            # Try again
            log.warning(
                "Retrying (%r) after connection broken by '%r': %s", retries, err, url
            )
            return self.urlopen(
                method,
                url,
                body,
                headers,
                retries,
                redirect,
                assert_same_host,
                timeout=timeout,
                pool_timeout=pool_timeout,
                release_conn=release_conn,
                chunked=chunked,
                body_pos=body_pos,
                **response_kw
            )

        # Handle redirect?
        redirect_location = redirect and response.get_redirect_location()
        if redirect_location:
            if response.status == 303:
                # Change the method according to RFC 9110, Section 15.4.4.
                method = "GET"
                # And lose the body not to transfer anything sensitive.
                body = None
                headers = HTTPHeaderDict(headers)._prepare_for_method_change()

            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_redirect:
                    response.drain_conn()
                    raise
                return response

            response.drain_conn()
            retries.sleep_for_retry(response)
            log.debug("Redirecting %s -> %s", url, redirect_location)
            return self.urlopen(
                method,
                redirect_location,
                body,
                headers,
                retries=retries,
                redirect=redirect,
                assert_same_host=assert_same_host,
                timeout=timeout,
                pool_timeout=pool_timeout,
                release_conn=release_conn,
                chunked=chunked,
                body_pos=body_pos,
                **response_kw
            )

        # Check if we should retry the HTTP response.
        has_retry_after = bool(response.headers.get("Retry-After"))
        if retries.is_retry(method, response.status, has_retry_after):
            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_status:
                    response.drain_conn()
                    raise
                return response

            response.drain_conn()
            retries.sleep(response)
            log.debug("Retry: %s", url)
            return self.urlopen(
                method,
                url,
                body,
                headers,
                retries=retries,
                redirect=redirect,
                assert_same_host=assert_same_host,
                timeout=timeout,
                pool_timeout=pool_timeout,
                release_conn=release_conn,
                chunked=chunked,
                body_pos=body_pos,
                **response_kw
            )

        return response


class HTTPSConnectionPool(HTTPConnectionPool):
    """
    Same as :class:`.HTTPConnectionPool`, but HTTPS.

    :class:`.HTTPSConnection` uses one of ``assert_fingerprint``,
    ``assert_hostname`` and ``host`` in this order to verify connections.
    If ``assert_hostname`` is False, no verification is done.

    The ``key_file``, ``cert_file``, ``cert_reqs``, ``ca_certs``,
    ``ca_cert_dir``, ``ssl_version``, ``key_password`` are only used if :mod:`ssl`
    is available and are fed into :meth:`urllib3.util.ssl_wrap_socket` to upgrade
    the connection socket into an SSL socket.
    """

    scheme = "https"
    ConnectionCls = HTTPSConnection

    def __init__(
        self,
        host,
        port=None,
        strict=False,
        timeout=Timeout.DEFAULT_TIMEOUT,
        maxsize=1,
        block=False,
        headers=None,
        retries=None,
        _proxy=None,
        _proxy_headers=None,
        key_file=None,
        cert_file=None,
        cert_reqs=None,
        key_password=None,
        ca_certs=None,
        ssl_version=None,
        assert_hostname=None,
        assert_fingerprint=None,
        ca_cert_dir=None,
        **conn_kw
    ):

        HTTPConnectionPool.__init__(
            self,
            host,
            port,
            strict,
            timeout,
            maxsize,
            block,
            headers,
            retries,
            _proxy,
            _proxy_headers,
            **conn_kw
        )

        self.key_file = key_file
        self.cert_file = cert_file
        self.cert_reqs = cert_reqs
        self.key_password = key_password
        self.ca_certs = ca_certs
        self.ca_cert_dir = ca_cert_dir
        self.ssl_version = ssl_version
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint

    def _prepare_conn(self, conn):
        """
        Prepare the ``connection`` for :meth:`urllib3.util.ssl_wrap_socket`
        and establish the tunnel if proxy is used.
        """

        if isinstance(conn, VerifiedHTTPSConnection):
            conn.set_cert(
                key_file=self.key_file,
                key_password=self.key_password,
                cert_file=self.cert_file,
                cert_reqs=self.cert_reqs,
                ca_certs=self.ca_certs,
                ca_cert_dir=self.ca_cert_dir,
                assert_hostname=self.assert_hostname,
                assert_fingerprint=self.assert_fingerprint,
            )
            conn.ssl_version = self.ssl_version
        return conn

    def _prepare_proxy(self, conn):
        """
        Establishes a tunnel connection through HTTP CONNECT.

        Tunnel connection is established early because otherwise httplib would
        improperly set Host: header to proxy's IP:port.
        """

        conn.set_tunnel(self._proxy_host, self.port, self.proxy_headers)

        if self.proxy.scheme == "https":
            conn.tls_in_tls_required = True

        conn.connect()

    def _new_conn(self):
        """
        Return a fresh :class:`http.client.HTTPSConnection`.
        """
        self.num_connections += 1
        log.debug(
            "Starting new HTTPS connection (%d): %s:%s",
            self.num_connections,
            self.host,
            self.port or "443",
        )

        if not self.ConnectionCls or self.ConnectionCls is DummyConnection:
            raise SSLError(
                "Can't connect to HTTPS URL because the SSL module is not available."
            )

        actual_host = self.host
        actual_port = self.port
        if self.proxy is not None:
            actual_host = self.proxy.host
            actual_port = self.proxy.port

        conn = self.ConnectionCls(
            host=actual_host,
            port=actual_port,
            timeout=self.timeout.connect_timeout,
            strict=self.strict,
            cert_file=self.cert_file,
            key_file=self.key_file,
            key_password=self.key_password,
            **self.conn_kw
        )

        return self._prepare_conn(conn)

    def _validate_conn(self, conn):
        """
        Called right before a request is made, after the socket is created.
        """
        super(HTTPSConnectionPool, self)._validate_conn(conn)

        # Force connect early to allow us to validate the connection.
        if not getattr(conn, "sock", None):  # AppEngine might not have  `.sock`
            conn.connect()

        if not conn.is_verified:
            warnings.warn(
                (
                    "Unverified HTTPS request is being made to host '%s'. "
                    "Adding certificate verification is strongly advised. See: "
                    "https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html"
                    "#ssl-warnings" % conn.host
                ),
                InsecureRequestWarning,
            )

        if getattr(conn, "proxy_is_verified", None) is False:
            warnings.warn(
                (
                    "Unverified HTTPS connection done to an HTTPS proxy. "
                    "Adding certificate verification is strongly advised. See: "
                    "https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html"
                    "#ssl-warnings"
                ),
                InsecureRequestWarning,
            )


def connection_from_url(url, **kw):
    """
    Given a url, return an :class:`.ConnectionPool` instance of its host.

    This is a shortcut for not having to parse out the scheme, host, and port
    of the url before creating an :class:`.ConnectionPool` instance.

    :param url:
        Absolute URL string that must include the scheme. Port is optional.

    :param \\**kw:
        Passes additional parameters to the constructor of the appropriate
        :class:`.ConnectionPool`. Useful for specifying things like
        timeout, maxsize, headers, etc.

    Example::

        >>> conn = connection_from_url('http://google.com/')
        >>> r = conn.request('GET', '/')
    """
    scheme, host, port = get_host(url)
    port = port or port_by_scheme.get(scheme, 80)
    if scheme == "https":
        return HTTPSConnectionPool(host, port=port, **kw)
    else:
        return HTTPConnectionPool(host, port=port, **kw)


def _normalize_host(host, scheme):
    """
    Normalize hosts for comparisons and use with sockets.
    """

    host = normalize_host(host, scheme)

    # httplib doesn't like it when we include brackets in IPv6 addresses
    # Specifically, if we include brackets but also pass the port then
    # httplib crazily doubles up the square brackets on the Host header.
    # Instead, we need to make sure we never pass ``None`` as the port.
    # However, for backward compatibility reasons we can't actually
    # *assert* that.  See http://bugs.python.org/issue28539
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host


def _close_pool_connections(pool):
    """Drains a queue of connections and closes each one."""
    try:
        while True:
            conn = pool.get(block=False)
            if conn:
                conn.close()
    except queue.Empty:
        pass  # Done.

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\_matplotlib\converter.py ===
from __future__ import annotations

import contextlib
import datetime as pydt
from datetime import (
    datetime,
    timedelta,
    tzinfo,
)
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

import matplotlib.dates as mdates
from matplotlib.ticker import (
    AutoLocator,
    Formatter,
    Locator,
)
from matplotlib.transforms import nonsingular
import matplotlib.units as munits
import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    Timestamp,
    to_offset,
)
from pandas._libs.tslibs.dtypes import (
    FreqGroup,
    periods_per_day,
)
from pandas._typing import (
    F,
    npt,
)

from pandas.core.dtypes.common import (
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_nested_list_like,
)

from pandas import (
    Index,
    Series,
    get_option,
)
import pandas.core.common as com
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
    Period,
    PeriodIndex,
    period_range,
)
import pandas.core.tools.datetimes as tools

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.axis import Axis

    from pandas._libs.tslibs.offsets import BaseOffset


_mpl_units = {}  # Cache for units overwritten by us


def get_pairs():
    pairs = [
        (Timestamp, DatetimeConverter),
        (Period, PeriodConverter),
        (pydt.datetime, DatetimeConverter),
        (pydt.date, DatetimeConverter),
        (pydt.time, TimeConverter),
        (np.datetime64, DatetimeConverter),
    ]
    return pairs


def register_pandas_matplotlib_converters(func: F) -> F:
    """
    Decorator applying pandas_converters.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with pandas_converters():
            return func(*args, **kwargs)

    return cast(F, wrapper)


@contextlib.contextmanager
def pandas_converters() -> Generator[None, None, None]:
    """
    Context manager registering pandas' converters for a plot.

    See Also
    --------
    register_pandas_matplotlib_converters : Decorator that applies this.
    """
    value = get_option("plotting.matplotlib.register_converters")

    if value:
        # register for True or "auto"
        register()
    try:
        yield
    finally:
        if value == "auto":
            # only deregister for "auto"
            deregister()


def register() -> None:
    pairs = get_pairs()
    for type_, cls in pairs:
        # Cache previous converter if present
        if type_ in munits.registry and not isinstance(munits.registry[type_], cls):
            previous = munits.registry[type_]
            _mpl_units[type_] = previous
        # Replace with pandas converter
        munits.registry[type_] = cls()


def deregister() -> None:
    # Renamed in pandas.plotting.__init__
    for type_, cls in get_pairs():
        # We use type to catch our classes directly, no inheritance
        if type(munits.registry.get(type_)) is cls:
            munits.registry.pop(type_)

    # restore the old keys
    for unit, formatter in _mpl_units.items():
        if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter}:
            # make it idempotent by excluding ours.
            munits.registry[unit] = formatter


def _to_ordinalf(tm: pydt.time) -> float:
    tot_sec = tm.hour * 3600 + tm.minute * 60 + tm.second + tm.microsecond / 10**6
    return tot_sec


def time2num(d):
    if isinstance(d, str):
        parsed = Timestamp(d)
        return _to_ordinalf(parsed.time())
    if isinstance(d, pydt.time):
        return _to_ordinalf(d)
    return d


class TimeConverter(munits.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        valid_types = (str, pydt.time)
        if isinstance(value, valid_types) or is_integer(value) or is_float(value):
            return time2num(value)
        if isinstance(value, Index):
            return value.map(time2num)
        if isinstance(value, (list, tuple, np.ndarray, Index)):
            return [time2num(x) for x in value]
        return value

    @staticmethod
    def axisinfo(unit, axis) -> munits.AxisInfo | None:
        if unit != "time":
            return None

        majloc = AutoLocator()
        majfmt = TimeFormatter(majloc)
        return munits.AxisInfo(majloc=majloc, majfmt=majfmt, label="time")

    @staticmethod
    def default_units(x, axis) -> str:
        return "time"


# time formatter
class TimeFormatter(Formatter):
    def __init__(self, locs) -> None:
        self.locs = locs

    def __call__(self, x, pos: int | None = 0) -> str:
        """
        Return the time of day as a formatted string.

        Parameters
        ----------
        x : float
            The time of day specified as seconds since 00:00 (midnight),
            with up to microsecond precision.
        pos
            Unused

        Returns
        -------
        str
            A string in HH:MM:SS.mmmuuu format. Microseconds,
            milliseconds and seconds are only displayed if non-zero.
        """
        fmt = "%H:%M:%S.%f"
        s = int(x)
        msus = round((x - s) * 10**6)
        ms = msus // 1000
        us = msus % 1000
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        _, h = divmod(h, 24)
        if us != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)
        elif ms != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)[:-3]
        elif s != 0:
            return pydt.time(h, m, s).strftime("%H:%M:%S")

        return pydt.time(h, m).strftime("%H:%M")


# Period Conversion


class PeriodConverter(mdates.DateConverter):
    @staticmethod
    def convert(values, units, axis):
        if is_nested_list_like(values):
            values = [PeriodConverter._convert_1d(v, units, axis) for v in values]
        else:
            values = PeriodConverter._convert_1d(values, units, axis)
        return values

    @staticmethod
    def _convert_1d(values, units, axis):
        if not hasattr(axis, "freq"):
            raise TypeError("Axis must have `freq` set to convert to Periods")
        valid_types = (str, datetime, Period, pydt.date, pydt.time, np.datetime64)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Period with BDay freq is deprecated", category=FutureWarning
            )
            warnings.filterwarnings(
                "ignore", r"PeriodDtype\[B\] is deprecated", category=FutureWarning
            )
            if (
                isinstance(values, valid_types)
                or is_integer(values)
                or is_float(values)
            ):
                return get_datevalue(values, axis.freq)
            elif isinstance(values, PeriodIndex):
                return values.asfreq(axis.freq).asi8
            elif isinstance(values, Index):
                return values.map(lambda x: get_datevalue(x, axis.freq))
            elif lib.infer_dtype(values, skipna=False) == "period":
                # https://github.com/pandas-dev/pandas/issues/24304
                # convert ndarray[period] -> PeriodIndex
                return PeriodIndex(values, freq=axis.freq).asi8
            elif isinstance(values, (list, tuple, np.ndarray, Index)):
                return [get_datevalue(x, axis.freq) for x in values]
        return values


def get_datevalue(date, freq):
    if isinstance(date, Period):
        return date.asfreq(freq).ordinal
    elif isinstance(date, (str, datetime, pydt.date, pydt.time, np.datetime64)):
        return Period(date, freq).ordinal
    elif (
        is_integer(date)
        or is_float(date)
        or (isinstance(date, (np.ndarray, Index)) and (date.size == 1))
    ):
        return date
    elif date is None:
        return None
    raise ValueError(f"Unrecognizable date '{date}'")


# Datetime Conversion
class DatetimeConverter(mdates.DateConverter):
    @staticmethod
    def convert(values, unit, axis):
        # values might be a 1-d array, or a list-like of arrays.
        if is_nested_list_like(values):
            values = [DatetimeConverter._convert_1d(v, unit, axis) for v in values]
        else:
            values = DatetimeConverter._convert_1d(values, unit, axis)
        return values

    @staticmethod
    def _convert_1d(values, unit, axis):
        def try_parse(values):
            try:
                return mdates.date2num(tools.to_datetime(values))
            except Exception:
                return values

        if isinstance(values, (datetime, pydt.date, np.datetime64, pydt.time)):
            return mdates.date2num(values)
        elif is_integer(values) or is_float(values):
            return values
        elif isinstance(values, str):
            return try_parse(values)
        elif isinstance(values, (list, tuple, np.ndarray, Index, Series)):
            if isinstance(values, Series):
                # https://github.com/matplotlib/matplotlib/issues/11391
                # Series was skipped. Convert to DatetimeIndex to get asi8
                values = Index(values)
            if isinstance(values, Index):
                values = values.values
            if not isinstance(values, np.ndarray):
                values = com.asarray_tuplesafe(values)

            if is_integer_dtype(values) or is_float_dtype(values):
                return values

            try:
                values = tools.to_datetime(values)
            except Exception:
                pass

            values = mdates.date2num(values)

        return values

    @staticmethod
    def axisinfo(unit: tzinfo | None, axis) -> munits.AxisInfo:
        """
        Return the :class:`~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a tzinfo instance or None.
        The *axis* argument is required but not used.
        """
        tz = unit

        majloc = PandasAutoDateLocator(tz=tz)
        majfmt = PandasAutoDateFormatter(majloc, tz=tz)
        datemin = pydt.date(2000, 1, 1)
        datemax = pydt.date(2010, 1, 1)

        return munits.AxisInfo(
            majloc=majloc, majfmt=majfmt, label="", default_limits=(datemin, datemax)
        )


class PandasAutoDateFormatter(mdates.AutoDateFormatter):
    def __init__(self, locator, tz=None, defaultfmt: str = "%Y-%m-%d") -> None:
        mdates.AutoDateFormatter.__init__(self, locator, tz, defaultfmt)


class PandasAutoDateLocator(mdates.AutoDateLocator):
    def get_locator(self, dmin, dmax):
        """Pick the best locator based on a distance."""
        tot_sec = (dmax - dmin).total_seconds()

        if abs(tot_sec) < self.minticks:
            self._freq = -1
            locator = MilliSecondLocator(self.tz)
            locator.set_axis(self.axis)

            # error: Item "None" of "Axis | _DummyAxis | _AxisWrapper | None"
            # has no attribute "get_data_interval"
            locator.axis.set_view_interval(  # type: ignore[union-attr]
                *self.axis.get_view_interval()  # type: ignore[union-attr]
            )
            locator.axis.set_data_interval(  # type: ignore[union-attr]
                *self.axis.get_data_interval()  # type: ignore[union-attr]
            )
            return locator

        return mdates.AutoDateLocator.get_locator(self, dmin, dmax)

    def _get_unit(self):
        return MilliSecondLocator.get_unit_generic(self._freq)


class MilliSecondLocator(mdates.DateLocator):
    UNIT = 1.0 / (24 * 3600 * 1000)

    def __init__(self, tz) -> None:
        mdates.DateLocator.__init__(self, tz)
        self._interval = 1.0

    def _get_unit(self):
        return self.get_unit_generic(-1)

    @staticmethod
    def get_unit_generic(freq):
        unit = mdates.RRuleLocator.get_unit_generic(freq)
        if unit < 0:
            return MilliSecondLocator.UNIT
        return unit

    def __call__(self):
        # if no data have been set, this will tank with a ValueError
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []

        # We need to cap at the endpoints of valid datetime
        nmax, nmin = mdates.date2num((dmax, dmin))

        num = (nmax - nmin) * 86400 * 1000
        max_millis_ticks = 6
        for interval in [1, 10, 50, 100, 200, 500]:
            if num <= interval * (max_millis_ticks - 1):
                self._interval = interval
                break
            # We went through the whole loop without breaking, default to 1
            self._interval = 1000.0

        estimate = (nmax - nmin) / (self._get_unit() * self._get_interval())

        if estimate > self.MAXTICKS * 2:
            raise RuntimeError(
                "MillisecondLocator estimated to generate "
                f"{estimate:d} ticks from {dmin} to {dmax}: exceeds Locator.MAXTICKS"
                f"* 2 ({self.MAXTICKS * 2:d}) "
            )

        interval = self._get_interval()
        freq = f"{interval}ms"
        tz = self.tz.tzname(None)
        st = dmin.replace(tzinfo=None)
        ed = dmin.replace(tzinfo=None)
        all_dates = date_range(start=st, end=ed, freq=freq, tz=tz).astype(object)

        try:
            if len(all_dates) > 0:
                locs = self.raise_if_exceeds(mdates.date2num(all_dates))
                return locs
        except Exception:  # pragma: no cover
            pass

        lims = mdates.date2num([dmin, dmax])
        return lims

    def _get_interval(self):
        return self._interval

    def autoscale(self):
        """
        Set the view limits to include the data range.
        """
        # We need to cap at the endpoints of valid datetime
        dmin, dmax = self.datalim_to_dt()

        vmin = mdates.date2num(dmin)
        vmax = mdates.date2num(dmax)

        return self.nonsingular(vmin, vmax)


def _from_ordinal(x, tz: tzinfo | None = None) -> datetime:
    ix = int(x)
    dt = datetime.fromordinal(ix)
    remainder = float(x) - ix
    hour, remainder = divmod(24 * remainder, 1)
    minute, remainder = divmod(60 * remainder, 1)
    second, remainder = divmod(60 * remainder, 1)
    microsecond = int(1_000_000 * remainder)
    if microsecond < 10:
        microsecond = 0  # compensate for rounding errors
    dt = datetime(
        dt.year, dt.month, dt.day, int(hour), int(minute), int(second), microsecond
    )
    if tz is not None:
        dt = dt.astimezone(tz)

    if microsecond > 999990:  # compensate for rounding errors
        dt += timedelta(microseconds=1_000_000 - microsecond)

    return dt


# Fixed frequency dynamic tick locators and formatters

# -------------------------------------------------------------------------
# --- Locators ---
# -------------------------------------------------------------------------


def _get_default_annual_spacing(nyears) -> tuple[int, int]:
    """
    Returns a default spacing between consecutive ticks for annual data.
    """
    if nyears < 11:
        (min_spacing, maj_spacing) = (1, 1)
    elif nyears < 20:
        (min_spacing, maj_spacing) = (1, 2)
    elif nyears < 50:
        (min_spacing, maj_spacing) = (1, 5)
    elif nyears < 100:
        (min_spacing, maj_spacing) = (5, 10)
    elif nyears < 200:
        (min_spacing, maj_spacing) = (5, 25)
    elif nyears < 600:
        (min_spacing, maj_spacing) = (10, 50)
    else:
        factor = nyears // 1000 + 1
        (min_spacing, maj_spacing) = (factor * 20, factor * 100)
    return (min_spacing, maj_spacing)


def _period_break(dates: PeriodIndex, period: str) -> npt.NDArray[np.intp]:
    """
    Returns the indices where the given period changes.

    Parameters
    ----------
    dates : PeriodIndex
        Array of intervals to monitor.
    period : str
        Name of the period to monitor.
    """
    mask = _period_break_mask(dates, period)
    return np.nonzero(mask)[0]


def _period_break_mask(dates: PeriodIndex, period: str) -> npt.NDArray[np.bool_]:
    current = getattr(dates, period)
    previous = getattr(dates - 1 * dates.freq, period)
    return current != previous


def has_level_label(label_flags: npt.NDArray[np.intp], vmin: float) -> bool:
    """
    Returns true if the ``label_flags`` indicate there is at least one label
    for this level.

    if the minimum view limit is not an exact integer, then the first tick
    label won't be shown, so we must adjust for that.
    """
    if label_flags.size == 0 or (
        label_flags.size == 1 and label_flags[0] == 0 and vmin % 1 > 0.0
    ):
        return False
    else:
        return True


def _get_periods_per_ymd(freq: BaseOffset) -> tuple[int, int, int]:
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]
    freq_group = FreqGroup.from_period_dtype_code(dtype_code)

    ppd = -1  # placeholder for above-day freqs

    if dtype_code >= FreqGroup.FR_HR.value:
        # error: "BaseOffset" has no attribute "_creso"
        ppd = periods_per_day(freq._creso)  # type: ignore[attr-defined]
        ppm = 28 * ppd
        ppy = 365 * ppd
    elif freq_group == FreqGroup.FR_BUS:
        ppm = 19
        ppy = 261
    elif freq_group == FreqGroup.FR_DAY:
        ppm = 28
        ppy = 365
    elif freq_group == FreqGroup.FR_WK:
        ppm = 3
        ppy = 52
    elif freq_group == FreqGroup.FR_MTH:
        ppm = 1
        ppy = 12
    elif freq_group == FreqGroup.FR_QTR:
        ppm = -1  # placerholder
        ppy = 4
    elif freq_group == FreqGroup.FR_ANN:
        ppm = -1  # placeholder
        ppy = 1
    else:
        raise NotImplementedError(f"Unsupported frequency: {dtype_code}")

    return ppd, ppm, ppy


@functools.cache
def _daily_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]

    periodsperday, periodspermonth, periodsperyear = _get_periods_per_ymd(freq)

    # save this for later usage
    vmin_orig = vmin
    (vmin, vmax) = (int(vmin), int(vmax))
    span = vmax - vmin + 1

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Period with BDay freq is deprecated", category=FutureWarning
        )
        warnings.filterwarnings(
            "ignore", r"PeriodDtype\[B\] is deprecated", category=FutureWarning
        )
        dates_ = period_range(
            start=Period(ordinal=vmin, freq=freq),
            end=Period(ordinal=vmax, freq=freq),
            freq=freq,
        )

    # Initialize the output
    info = np.zeros(
        span, dtype=[("val", np.int64), ("maj", bool), ("min", bool), ("fmt", "|S20")]
    )
    info["val"][:] = dates_.asi8
    info["fmt"][:] = ""
    info["maj"][[0, -1]] = True
    # .. and set some shortcuts
    info_maj = info["maj"]
    info_min = info["min"]
    info_fmt = info["fmt"]

    def first_label(label_flags):
        if (label_flags[0] == 0) and (label_flags.size > 1) and ((vmin_orig % 1) > 0.0):
            return label_flags[1]
        else:
            return label_flags[0]

    # Case 1. Less than a month
    if span <= periodspermonth:
        day_start = _period_break(dates_, "day")
        month_start = _period_break(dates_, "month")
        year_start = _period_break(dates_, "year")

        def _hour_finder(label_interval: int, force_year_start: bool) -> None:
            target = dates_.hour
            mask = _period_break_mask(dates_, "hour")
            info_maj[day_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M"
            info_fmt[day_start] = "%H:%M\n%d-%b"
            info_fmt[year_start] = "%H:%M\n%d-%b\n%Y"
            if force_year_start and not has_level_label(year_start, vmin_orig):
                info_fmt[first_label(day_start)] = "%H:%M\n%d-%b\n%Y"

        def _minute_finder(label_interval: int) -> None:
            target = dates_.minute
            hour_start = _period_break(dates_, "hour")
            mask = _period_break_mask(dates_, "minute")
            info_maj[hour_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M"
            info_fmt[day_start] = "%H:%M\n%d-%b"
            info_fmt[year_start] = "%H:%M\n%d-%b\n%Y"

        def _second_finder(label_interval: int) -> None:
            target = dates_.second
            minute_start = _period_break(dates_, "minute")
            mask = _period_break_mask(dates_, "second")
            info_maj[minute_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = "%H:%M:%S"
            info_fmt[day_start] = "%H:%M:%S\n%d-%b"
            info_fmt[year_start] = "%H:%M:%S\n%d-%b\n%Y"

        if span < periodsperday / 12000:
            _second_finder(1)
        elif span < periodsperday / 6000:
            _second_finder(2)
        elif span < periodsperday / 2400:
            _second_finder(5)
        elif span < periodsperday / 1200:
            _second_finder(10)
        elif span < periodsperday / 800:
            _second_finder(15)
        elif span < periodsperday / 400:
            _second_finder(30)
        elif span < periodsperday / 150:
            _minute_finder(1)
        elif span < periodsperday / 70:
            _minute_finder(2)
        elif span < periodsperday / 24:
            _minute_finder(5)
        elif span < periodsperday / 12:
            _minute_finder(15)
        elif span < periodsperday / 6:
            _minute_finder(30)
        elif span < periodsperday / 2.5:
            _hour_finder(1, False)
        elif span < periodsperday / 1.5:
            _hour_finder(2, False)
        elif span < periodsperday * 1.25:
            _hour_finder(3, False)
        elif span < periodsperday * 2.5:
            _hour_finder(6, True)
        elif span < periodsperday * 4:
            _hour_finder(12, True)
        else:
            info_maj[month_start] = True
            info_min[day_start] = True
            info_fmt[day_start] = "%d"
            info_fmt[month_start] = "%d\n%b"
            info_fmt[year_start] = "%d\n%b\n%Y"
            if not has_level_label(year_start, vmin_orig):
                if not has_level_label(month_start, vmin_orig):
                    info_fmt[first_label(day_start)] = "%d\n%b\n%Y"
                else:
                    info_fmt[first_label(month_start)] = "%d\n%b\n%Y"

    # Case 2. Less than three months
    elif span <= periodsperyear // 4:
        month_start = _period_break(dates_, "month")
        info_maj[month_start] = True
        if dtype_code < FreqGroup.FR_HR.value:
            info["min"] = True
        else:
            day_start = _period_break(dates_, "day")
            info["min"][day_start] = True
        week_start = _period_break(dates_, "week")
        year_start = _period_break(dates_, "year")
        info_fmt[week_start] = "%d"
        info_fmt[month_start] = "\n\n%b"
        info_fmt[year_start] = "\n\n%b\n%Y"
        if not has_level_label(year_start, vmin_orig):
            if not has_level_label(month_start, vmin_orig):
                info_fmt[first_label(week_start)] = "\n\n%b\n%Y"
            else:
                info_fmt[first_label(month_start)] = "\n\n%b\n%Y"
    # Case 3. Less than 14 months ...............
    elif span <= 1.15 * periodsperyear:
        year_start = _period_break(dates_, "year")
        month_start = _period_break(dates_, "month")
        week_start = _period_break(dates_, "week")
        info_maj[month_start] = True
        info_min[week_start] = True
        info_min[year_start] = False
        info_min[month_start] = False
        info_fmt[month_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"
        if not has_level_label(year_start, vmin_orig):
            info_fmt[first_label(month_start)] = "%b\n%Y"
    # Case 4. Less than 2.5 years ...............
    elif span <= 2.5 * periodsperyear:
        year_start = _period_break(dates_, "year")
        quarter_start = _period_break(dates_, "quarter")
        month_start = _period_break(dates_, "month")
        info_maj[quarter_start] = True
        info_min[month_start] = True
        info_fmt[quarter_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"
    # Case 4. Less than 4 years .................
    elif span <= 4 * periodsperyear:
        year_start = _period_break(dates_, "year")
        month_start = _period_break(dates_, "month")
        info_maj[year_start] = True
        info_min[month_start] = True
        info_min[year_start] = False

        month_break = dates_[month_start].month
        jan_or_jul = month_start[(month_break == 1) | (month_break == 7)]
        info_fmt[jan_or_jul] = "%b"
        info_fmt[year_start] = "%b\n%Y"
    # Case 5. Less than 11 years ................
    elif span <= 11 * periodsperyear:
        year_start = _period_break(dates_, "year")
        quarter_start = _period_break(dates_, "quarter")
        info_maj[year_start] = True
        info_min[quarter_start] = True
        info_min[year_start] = False
        info_fmt[year_start] = "%Y"
    # Case 6. More than 12 years ................
    else:
        year_start = _period_break(dates_, "year")
        year_break = dates_[year_start].year
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        major_idx = year_start[(year_break % maj_anndef == 0)]
        info_maj[major_idx] = True
        minor_idx = year_start[(year_break % min_anndef == 0)]
        info_min[minor_idx] = True
        info_fmt[major_idx] = "%Y"

    return info


@functools.cache
def _monthly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    _, _, periodsperyear = _get_periods_per_ymd(freq)

    vmin_orig = vmin
    (vmin, vmax) = (int(vmin), int(vmax))
    span = vmax - vmin + 1

    # Initialize the output
    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    info["val"] = np.arange(vmin, vmax + 1)
    dates_ = info["val"]
    info["fmt"] = ""
    year_start = (dates_ % 12 == 0).nonzero()[0]
    info_maj = info["maj"]
    info_fmt = info["fmt"]

    if span <= 1.15 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True

        info_fmt[:] = "%b"
        info_fmt[year_start] = "%b\n%Y"

        if not has_level_label(year_start, vmin_orig):
            if dates_.size > 1:
                idx = 1
            else:
                idx = 0
            info_fmt[idx] = "%b\n%Y"

    elif span <= 2.5 * periodsperyear:
        quarter_start = (dates_ % 3 == 0).nonzero()
        info_maj[year_start] = True
        # TODO: Check the following : is it really info['fmt'] ?
        #  2023-09-15 this is reached in test_finder_monthly
        info["fmt"][quarter_start] = True
        info["min"] = True

        info_fmt[quarter_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"

    elif span <= 4 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True

        jan_or_jul = (dates_ % 12 == 0) | (dates_ % 12 == 6)
        info_fmt[jan_or_jul] = "%b"
        info_fmt[year_start] = "%b\n%Y"

    elif span <= 11 * periodsperyear:
        quarter_start = (dates_ % 3 == 0).nonzero()
        info_maj[year_start] = True
        info["min"][quarter_start] = True

        info_fmt[year_start] = "%Y"

    else:
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        years = dates_[year_start] // 12 + 1
        major_idx = year_start[(years % maj_anndef == 0)]
        info_maj[major_idx] = True
        info["min"][year_start[(years % min_anndef == 0)]] = True

        info_fmt[major_idx] = "%Y"

    return info


@functools.cache
def _quarterly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    _, _, periodsperyear = _get_periods_per_ymd(freq)
    vmin_orig = vmin
    (vmin, vmax) = (int(vmin), int(vmax))
    span = vmax - vmin + 1

    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    info["val"] = np.arange(vmin, vmax + 1)
    info["fmt"] = ""
    dates_ = info["val"]
    info_maj = info["maj"]
    info_fmt = info["fmt"]
    year_start = (dates_ % 4 == 0).nonzero()[0]

    if span <= 3.5 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True

        info_fmt[:] = "Q%q"
        info_fmt[year_start] = "Q%q\n%F"
        if not has_level_label(year_start, vmin_orig):
            if dates_.size > 1:
                idx = 1
            else:
                idx = 0
            info_fmt[idx] = "Q%q\n%F"

    elif span <= 11 * periodsperyear:
        info_maj[year_start] = True
        info["min"] = True
        info_fmt[year_start] = "%F"

    else:
        # https://github.com/pandas-dev/pandas/pull/47602
        years = dates_[year_start] // 4 + 1970
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        major_idx = year_start[(years % maj_anndef == 0)]
        info_maj[major_idx] = True
        info["min"][year_start[(years % min_anndef == 0)]] = True
        info_fmt[major_idx] = "%F"

    return info


@functools.cache
def _annual_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    # Note: small difference here vs other finders in adding 1 to vmax
    (vmin, vmax) = (int(vmin), int(vmax + 1))
    span = vmax - vmin + 1

    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    info["val"] = np.arange(vmin, vmax + 1)
    info["fmt"] = ""
    dates_ = info["val"]

    (min_anndef, maj_anndef) = _get_default_annual_spacing(span)
    major_idx = dates_ % maj_anndef == 0
    minor_idx = dates_ % min_anndef == 0
    info["maj"][major_idx] = True
    info["min"][minor_idx] = True
    info["fmt"][major_idx] = "%Y"

    return info


def get_finder(freq: BaseOffset):
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]
    fgroup = FreqGroup.from_period_dtype_code(dtype_code)

    if fgroup == FreqGroup.FR_ANN:
        return _annual_finder
    elif fgroup == FreqGroup.FR_QTR:
        return _quarterly_finder
    elif fgroup == FreqGroup.FR_MTH:
        return _monthly_finder
    elif (dtype_code >= FreqGroup.FR_BUS.value) or fgroup == FreqGroup.FR_WK:
        return _daily_finder
    else:  # pragma: no cover
        raise NotImplementedError(f"Unsupported frequency: {dtype_code}")


class TimeSeries_DateLocator(Locator):
    """
    Locates the ticks along an axis controlled by a :class:`Series`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : {False, True}, optional
        Whether the locator is for minor ticks (True) or not.
    dynamic_mode : {True, False}, optional
        Whether the locator should work in dynamic mode.
    base : {int}, optional
    quarter : {int}, optional
    month : {int}, optional
    day : {int}, optional
    """

    axis: Axis

    def __init__(
        self,
        freq: BaseOffset,
        minor_locator: bool = False,
        dynamic_mode: bool = True,
        base: int = 1,
        quarter: int = 1,
        month: int = 1,
        day: int = 1,
        plot_obj=None,
    ) -> None:
        freq = to_offset(freq, is_period=True)
        self.freq = freq
        self.base = base
        (self.quarter, self.month, self.day) = (quarter, month, day)
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        self.finder = get_finder(freq)

    def _get_default_locs(self, vmin, vmax):
        """Returns the default locations of ticks."""
        locator = self.finder(vmin, vmax, self.freq)

        if self.isminor:
            return np.compress(locator["min"], locator["val"])
        return np.compress(locator["maj"], locator["val"])

    def __call__(self):
        """Return the locations of the ticks."""
        # axis calls Locator.set_axis inside set_m<xxxx>_formatter

        vi = tuple(self.axis.get_view_interval())
        vmin, vmax = vi
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if self.isdynamic:
            locs = self._get_default_locs(vmin, vmax)
        else:  # pragma: no cover
            base = self.base
            (d, m) = divmod(vmin, base)
            vmin = (d + 1) * base
            # error: No overload variant of "range" matches argument types "float",
            # "float", "int"
            locs = list(range(vmin, vmax + 1, base))  # type: ignore[call-overload]
        return locs

    def autoscale(self):
        """
        Sets the view limits to the nearest multiples of base that contain the
        data.
        """
        # requires matplotlib >= 0.98.0
        (vmin, vmax) = self.axis.get_data_interval()

        locs = self._get_default_locs(vmin, vmax)
        (vmin, vmax) = locs[[0, -1]]
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        return nonsingular(vmin, vmax)


# -------------------------------------------------------------------------
# --- Formatter ---
# -------------------------------------------------------------------------


class TimeSeries_DateFormatter(Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`PeriodIndex`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : bool, default False
        Whether the current formatter should apply to minor ticks (True) or
        major ticks (False).
    dynamic_mode : bool, default True
        Whether the formatter works in dynamic mode or not.
    """

    axis: Axis

    def __init__(
        self,
        freq: BaseOffset,
        minor_locator: bool = False,
        dynamic_mode: bool = True,
        plot_obj=None,
    ) -> None:
        freq = to_offset(freq, is_period=True)
        self.format = None
        self.freq = freq
        self.locs: list[Any] = []  # unused, for matplotlib compat
        self.formatdict: dict[Any, Any] | None = None
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        self.finder = get_finder(freq)

    def _set_default_format(self, vmin, vmax):
        """Returns the default ticks spacing."""
        info = self.finder(vmin, vmax, self.freq)

        if self.isminor:
            format = np.compress(info["min"] & np.logical_not(info["maj"]), info)
        else:
            format = np.compress(info["maj"], info)
        self.formatdict = {x: f for (x, _, _, f) in format}
        return self.formatdict

    def set_locs(self, locs) -> None:
        """Sets the locations of the ticks"""
        # don't actually use the locs. This is just needed to work with
        # matplotlib. Force to use vmin, vmax

        self.locs = locs

        (vmin, vmax) = tuple(self.axis.get_view_interval())
        if vmax < vmin:
            (vmin, vmax) = (vmax, vmin)
        self._set_default_format(vmin, vmax)

    def __call__(self, x, pos: int | None = 0) -> str:
        if self.formatdict is None:
            return ""
        else:
            fmt = self.formatdict.pop(x, "")
            if isinstance(fmt, np.bytes_):
                fmt = fmt.decode("utf-8")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Period with BDay freq is deprecated",
                    category=FutureWarning,
                )
                period = Period(ordinal=int(x), freq=self.freq)
            assert isinstance(period, Period)
            return period.strftime(fmt)


class TimeSeries_TimedeltaFormatter(Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`TimedeltaIndex`.
    """

    axis: Axis

    @staticmethod
    def format_timedelta_ticks(x, pos, n_decimals: int) -> str:
        """
        Convert seconds to 'D days HH:MM:SS.F'
        """
        s, ns = divmod(x, 10**9)  # TODO(non-nano): this looks like it assumes ns
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        decimals = int(ns * 10 ** (n_decimals - 9))
        s = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        if n_decimals > 0:
            s += f".{decimals:0{n_decimals}d}"
        if d != 0:
            s = f"{int(d):d} days {s}"
        return s

    def __call__(self, x, pos: int | None = 0) -> str:
        (vmin, vmax) = tuple(self.axis.get_view_interval())
        n_decimals = min(int(np.ceil(np.log10(100 * 10**9 / abs(vmax - vmin)))), 9)
        return self.format_timedelta_ticks(x, pos, n_decimals)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexes\interval.py ===
""" define the IntervalIndex """
from __future__ import annotations

from operator import (
    le,
    lt,
)
import textwrap
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.interval import (
    Interval,
    IntervalMixin,
    IntervalTree,
)
from pandas._libs.tslibs import (
    BaseOffset,
    Period,
    Timedelta,
    Timestamp,
    to_offset,
)
from pandas.errors import InvalidIndexError
from pandas.util._decorators import (
    Appender,
    cache_readonly,
)
from pandas.util._exceptions import rewrite_exception

from pandas.core.dtypes.cast import (
    find_common_type,
    infer_dtype_from_scalar,
    maybe_box_datetimelike,
    maybe_downcast_numeric,
    maybe_upcast_numeric_to_64bit,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_number,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
)
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.algorithms import unique
from pandas.core.arrays.datetimelike import validate_periods
from pandas.core.arrays.interval import (
    IntervalArray,
    _interval_shared_docs,
)
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    _index_shared_docs,
    ensure_index,
    maybe_extract_name,
)
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    date_range,
)
from pandas.core.indexes.extension import (
    ExtensionIndex,
    inherit_names,
)
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import (
    TimedeltaIndex,
    timedelta_range,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        IntervalClosedType,
        Self,
        npt,
    )
_index_doc_kwargs = dict(ibase._index_doc_kwargs)

_index_doc_kwargs.update(
    {
        "klass": "IntervalIndex",
        "qualname": "IntervalIndex",
        "target_klass": "IntervalIndex or list of Intervals",
        "name": textwrap.dedent(
            """\
         name : object, optional
              Name to be stored in the index.
         """
        ),
    }
)


def _get_next_label(label):
    # see test_slice_locs_with_ints_and_floats_succeeds
    dtype = getattr(label, "dtype", type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = "datetime64[ns]"
    dtype = pandas_dtype(dtype)

    if lib.is_np_dtype(dtype, "mM") or isinstance(dtype, DatetimeTZDtype):
        return label + np.timedelta64(1, "ns")
    elif is_integer_dtype(dtype):
        return label + 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, np.inf)
    else:
        raise TypeError(f"cannot determine next label for type {repr(type(label))}")


def _get_prev_label(label):
    # see test_slice_locs_with_ints_and_floats_succeeds
    dtype = getattr(label, "dtype", type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = "datetime64[ns]"
    dtype = pandas_dtype(dtype)

    if lib.is_np_dtype(dtype, "mM") or isinstance(dtype, DatetimeTZDtype):
        return label - np.timedelta64(1, "ns")
    elif is_integer_dtype(dtype):
        return label - 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, -np.inf)
    else:
        raise TypeError(f"cannot determine next label for type {repr(type(label))}")


def _new_IntervalIndex(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't have
    arguments and breaks __new__.
    """
    return cls.from_arrays(**d)


@Appender(
    _interval_shared_docs["class"]
    % {
        "klass": "IntervalIndex",
        "summary": "Immutable index of intervals that are closed on the same side.",
        "name": _index_doc_kwargs["name"],
        "extra_attributes": "is_overlapping\nvalues\n",
        "extra_methods": "",
        "examples": textwrap.dedent(
            """\
    Examples
    --------
    A new ``IntervalIndex`` is typically constructed using
    :func:`interval_range`:

    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')

    It may also be constructed using one of the constructor
    methods: :meth:`IntervalIndex.from_arrays`,
    :meth:`IntervalIndex.from_breaks`, and :meth:`IntervalIndex.from_tuples`.

    See further examples in the doc strings of ``interval_range`` and the
    mentioned constructor methods.
    """
        ),
    }
)
@inherit_names(["set_closed", "to_tuples"], IntervalArray, wrap=True)
@inherit_names(
    [
        "__array__",
        "overlaps",
        "contains",
        "closed_left",
        "closed_right",
        "open_left",
        "open_right",
        "is_empty",
    ],
    IntervalArray,
)
@inherit_names(["is_non_overlapping_monotonic", "closed"], IntervalArray, cache=True)
class IntervalIndex(ExtensionIndex):
    _typ = "intervalindex"

    # annotate properties pinned via inherit_names
    closed: IntervalClosedType
    is_non_overlapping_monotonic: bool
    closed_left: bool
    closed_right: bool
    open_left: bool
    open_right: bool

    _data: IntervalArray
    _values: IntervalArray
    _can_hold_strings = False
    _data_cls = IntervalArray

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data,
        closed: IntervalClosedType | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
        verify_integrity: bool = True,
    ) -> Self:
        name = maybe_extract_name(name, data, cls)

        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray(
                data,
                closed=closed,
                copy=copy,
                dtype=dtype,
                verify_integrity=verify_integrity,
            )

        return cls._simple_new(array, name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_breaks"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_breaks([0, 1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_breaks(
        cls,
        breaks,
        closed: IntervalClosedType | None = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray.from_breaks(
                breaks, closed=closed, copy=copy, dtype=dtype
            )
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_arrays"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_arrays(
        cls,
        left,
        right,
        closed: IntervalClosedType = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray.from_arrays(
                left, right, closed, copy=copy, dtype=dtype
            )
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_tuples"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])
        IntervalIndex([(0, 1], (1, 2]],
                       dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_tuples(
        cls,
        data,
        closed: IntervalClosedType = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        with rewrite_exception("IntervalArray", cls.__name__):
            arr = IntervalArray.from_tuples(data, closed=closed, copy=copy, dtype=dtype)
        return cls._simple_new(arr, name=name)

    # --------------------------------------------------------------------
    # error: Return type "IntervalTree" of "_engine" incompatible with return type
    # "Union[IndexEngine, ExtensionEngine]" in supertype "Index"
    @cache_readonly
    def _engine(self) -> IntervalTree:  # type: ignore[override]
        # IntervalTree does not supports numpy array unless they are 64 bit
        left = self._maybe_convert_i8(self.left)
        left = maybe_upcast_numeric_to_64bit(left)
        right = self._maybe_convert_i8(self.right)
        right = maybe_upcast_numeric_to_64bit(right)
        return IntervalTree(left, right, closed=self.closed)

    def __contains__(self, key: Any) -> bool:
        """
        return a boolean if this key is IN the index
        We *only* accept an Interval

        Parameters
        ----------
        key : Interval

        Returns
        -------
        bool
        """
        hash(key)
        if not isinstance(key, Interval):
            if is_valid_na_for_dtype(key, self.dtype):
                return self.hasnans
            return False

        try:
            self.get_loc(key)
            return True
        except KeyError:
            return False

    def _getitem_slice(self, slobj: slice) -> IntervalIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._data[slobj]
        return type(self)._simple_new(res, name=self._name)

    @cache_readonly
    def _multiindex(self) -> MultiIndex:
        return MultiIndex.from_arrays([self.left, self.right], names=["left", "right"])

    def __reduce__(self):
        d = {
            "left": self.left,
            "right": self.right,
            "closed": self.closed,
            "name": self.name,
        }
        return _new_IntervalIndex, (type(self), d), None

    @property
    def inferred_type(self) -> str:
        """Return a string of the type inferred from the values"""
        return "interval"

    # Cannot determine type of "memory_usage"
    @Appender(Index.memory_usage.__doc__)  # type: ignore[has-type]
    def memory_usage(self, deep: bool = False) -> int:
        # we don't use an explicit engine
        # so return the bytes here
        return self.left.memory_usage(deep=deep) + self.right.memory_usage(deep=deep)

    # IntervalTree doesn't have a is_monotonic_decreasing, so have to override
    #  the Index implementation
    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        """
        Return True if the IntervalIndex is monotonic decreasing (only equal or
        decreasing values), else False
        """
        return self[::-1].is_monotonic_increasing

    @cache_readonly
    def is_unique(self) -> bool:
        """
        Return True if the IntervalIndex contains unique elements, else False.
        """
        left = self.left
        right = self.right

        if self.isna().sum() > 1:
            return False

        if left.is_unique or right.is_unique:
            return True

        seen_pairs = set()
        check_idx = np.where(left.duplicated(keep=False))[0]
        for idx in check_idx:
            pair = (left[idx], right[idx])
            if pair in seen_pairs:
                return False
            seen_pairs.add(pair)

        return True

    @property
    def is_overlapping(self) -> bool:
        """
        Return True if the IntervalIndex has overlapping intervals, else False.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Returns
        -------
        bool
            Boolean indicating if the IntervalIndex has overlapping intervals.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.
        IntervalIndex.overlaps : Check an IntervalIndex elementwise for
            overlaps.

        Examples
        --------
        >>> index = pd.IntervalIndex.from_tuples([(0, 2), (1, 3), (4, 5)])
        >>> index
        IntervalIndex([(0, 2], (1, 3], (4, 5]],
              dtype='interval[int64, right]')
        >>> index.is_overlapping
        True

        Intervals that share closed endpoints overlap:

        >>> index = pd.interval_range(0, 3, closed='both')
        >>> index
        IntervalIndex([[0, 1], [1, 2], [2, 3]],
              dtype='interval[int64, both]')
        >>> index.is_overlapping
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> index = pd.interval_range(0, 3, closed='left')
        >>> index
        IntervalIndex([[0, 1), [1, 2), [2, 3)],
              dtype='interval[int64, left]')
        >>> index.is_overlapping
        False
        """
        # GH 23309
        return self._engine.is_overlapping

    def _needs_i8_conversion(self, key) -> bool:
        """
        Check if a given key needs i8 conversion. Conversion is necessary for
        Timestamp, Timedelta, DatetimeIndex, and TimedeltaIndex keys. An
        Interval-like requires conversion if its endpoints are one of the
        aforementioned types.

        Assumes that any list-like data has already been cast to an Index.

        Parameters
        ----------
        key : scalar or Index-like
            The key that should be checked for i8 conversion

        Returns
        -------
        bool
        """
        key_dtype = getattr(key, "dtype", None)
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            return self._needs_i8_conversion(key.left)

        i8_types = (Timestamp, Timedelta, DatetimeIndex, TimedeltaIndex)
        return isinstance(key, i8_types)

    def _maybe_convert_i8(self, key):
        """
        Maybe convert a given key to its equivalent i8 value(s). Used as a
        preprocessing step prior to IntervalTree queries (self._engine), which
        expects numeric data.

        Parameters
        ----------
        key : scalar or list-like
            The key that should maybe be converted to i8.

        Returns
        -------
        scalar or list-like
            The original key if no conversion occurred, int if converted scalar,
            Index with an int64 dtype if converted list-like.
        """
        if is_list_like(key):
            key = ensure_index(key)
            key = maybe_upcast_numeric_to_64bit(key)

        if not self._needs_i8_conversion(key):
            return key

        scalar = is_scalar(key)
        key_dtype = getattr(key, "dtype", None)
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            # convert left/right and reconstruct
            left = self._maybe_convert_i8(key.left)
            right = self._maybe_convert_i8(key.right)
            constructor = Interval if scalar else IntervalIndex.from_arrays
            # error: "object" not callable
            return constructor(
                left, right, closed=self.closed
            )  # type: ignore[operator]

        if scalar:
            # Timestamp/Timedelta
            key_dtype, key_i8 = infer_dtype_from_scalar(key)
            if isinstance(key, Period):
                key_i8 = key.ordinal
            elif isinstance(key_i8, Timestamp):
                key_i8 = key_i8._value
            elif isinstance(key_i8, (np.datetime64, np.timedelta64)):
                key_i8 = key_i8.view("i8")
        else:
            # DatetimeIndex/TimedeltaIndex
            key_dtype, key_i8 = key.dtype, Index(key.asi8)
            if key.hasnans:
                # convert NaT from its i8 value to np.nan so it's not viewed
                # as a valid value, maybe causing errors (e.g. is_overlapping)
                key_i8 = key_i8.where(~key._isnan)

        # ensure consistency with IntervalIndex subtype
        # error: Item "ExtensionDtype"/"dtype[Any]" of "Union[dtype[Any],
        # ExtensionDtype]" has no attribute "subtype"
        subtype = self.dtype.subtype  # type: ignore[union-attr]

        if subtype != key_dtype:
            raise ValueError(
                f"Cannot index an IntervalIndex of subtype {subtype} with "
                f"values of dtype {key_dtype}"
            )

        return key_i8

    def _searchsorted_monotonic(self, label, side: Literal["left", "right"] = "left"):
        if not self.is_non_overlapping_monotonic:
            raise KeyError(
                "can only get slices from an IntervalIndex if bounds are "
                "non-overlapping and all monotonic increasing or decreasing"
            )

        if isinstance(label, (IntervalMixin, IntervalIndex)):
            raise NotImplementedError("Interval objects are not currently supported")

        # GH 20921: "not is_monotonic_increasing" for the second condition
        # instead of "is_monotonic_decreasing" to account for single element
        # indexes being both increasing and decreasing
        if (side == "left" and self.left.is_monotonic_increasing) or (
            side == "right" and not self.left.is_monotonic_increasing
        ):
            sub_idx = self.right
            if self.open_right:
                label = _get_next_label(label)
        else:
            sub_idx = self.left
            if self.open_left:
                label = _get_prev_label(label)

        return sub_idx._searchsorted_monotonic(label, side)

    # --------------------------------------------------------------------
    # Indexing Methods

    def get_loc(self, key) -> int | slice | np.ndarray:
        """
        Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label

        Returns
        -------
        int if unique index, slice if monotonic index, else mask

        Examples
        --------
        >>> i1, i2 = pd.Interval(0, 1), pd.Interval(1, 2)
        >>> index = pd.IntervalIndex([i1, i2])
        >>> index.get_loc(1)
        0

        You can also supply a point inside an interval.

        >>> index.get_loc(1.5)
        1

        If a label is in several intervals, you get the locations of all the
        relevant intervals.

        >>> i3 = pd.Interval(0, 2)
        >>> overlapping_index = pd.IntervalIndex([i1, i2, i3])
        >>> overlapping_index.get_loc(0.5)
        array([ True, False,  True])

        Only exact matches will be returned if an interval is provided.

        >>> index.get_loc(pd.Interval(0, 1))
        0
        """
        self._check_indexing_error(key)

        if isinstance(key, Interval):
            if self.closed != key.closed:
                raise KeyError(key)
            mask = (self.left == key.left) & (self.right == key.right)
        elif is_valid_na_for_dtype(key, self.dtype):
            mask = self.isna()
        else:
            # assume scalar
            op_left = le if self.closed_left else lt
            op_right = le if self.closed_right else lt
            try:
                mask = op_left(self.left, key) & op_right(key, self.right)
            except TypeError as err:
                # scalar is not comparable to II subtype --> invalid label
                raise KeyError(key) from err

        matches = mask.sum()
        if matches == 0:
            raise KeyError(key)
        if matches == 1:
            return mask.argmax()

        res = lib.maybe_booleans_to_slice(mask.view("u1"))
        if isinstance(res, slice) and res.stop is None:
            # TODO: DO this in maybe_booleans_to_slice?
            res = slice(res.start, len(self), res.step)
        return res

    def _get_indexer(
        self,
        target: Index,
        method: str | None = None,
        limit: int | None = None,
        tolerance: Any | None = None,
    ) -> npt.NDArray[np.intp]:
        if isinstance(target, IntervalIndex):
            # We only get here with not self.is_overlapping
            # -> at most one match per interval in target
            # want exact matches -> need both left/right to match, so defer to
            # left/right get_indexer, compare elementwise, equality -> match
            indexer = self._get_indexer_unique_sides(target)

        elif not (is_object_dtype(target.dtype) or is_string_dtype(target.dtype)):
            # homogeneous scalar index: use IntervalTree
            # we should always have self._should_partial_index(target) here
            target = self._maybe_convert_i8(target)
            indexer = self._engine.get_indexer(target.values)
        else:
            # heterogeneous scalar index: defer elementwise to get_loc
            # we should always have self._should_partial_index(target) here
            return self._get_indexer_pointwise(target)[0]

        return ensure_platform_int(indexer)

    @Appender(_index_shared_docs["get_indexer_non_unique"] % _index_doc_kwargs)
    def get_indexer_non_unique(
        self, target: Index
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        target = ensure_index(target)

        if not self._should_compare(target) and not self._should_partial_index(target):
            # e.g. IntervalIndex with different closed or incompatible subtype
            #  -> no matches
            return self._get_indexer_non_comparable(target, None, unique=False)

        elif isinstance(target, IntervalIndex):
            if self.left.is_unique and self.right.is_unique:
                # fastpath available even if we don't have self._index_as_unique
                indexer = self._get_indexer_unique_sides(target)
                missing = (indexer == -1).nonzero()[0]
            else:
                return self._get_indexer_pointwise(target)

        elif is_object_dtype(target.dtype) or not self._should_partial_index(target):
            # target might contain intervals: defer elementwise to get_loc
            return self._get_indexer_pointwise(target)

        else:
            # Note: this case behaves differently from other Index subclasses
            #  because IntervalIndex does partial-int indexing
            target = self._maybe_convert_i8(target)
            indexer, missing = self._engine.get_indexer_non_unique(target.values)

        return ensure_platform_int(indexer), ensure_platform_int(missing)

    def _get_indexer_unique_sides(self, target: IntervalIndex) -> npt.NDArray[np.intp]:
        """
        _get_indexer specialized to the case where both of our sides are unique.
        """
        # Caller is responsible for checking
        #  `self.left.is_unique and self.right.is_unique`

        left_indexer = self.left.get_indexer(target.left)
        right_indexer = self.right.get_indexer(target.right)
        indexer = np.where(left_indexer == right_indexer, left_indexer, -1)
        return indexer

    def _get_indexer_pointwise(
        self, target: Index
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        pointwise implementation for get_indexer and get_indexer_non_unique.
        """
        indexer, missing = [], []
        for i, key in enumerate(target):
            try:
                locs = self.get_loc(key)
                if isinstance(locs, slice):
                    # Only needed for get_indexer_non_unique
                    locs = np.arange(locs.start, locs.stop, locs.step, dtype="intp")
                elif lib.is_integer(locs):
                    locs = np.array(locs, ndmin=1)
                else:
                    # otherwise we have ndarray[bool]
                    locs = np.where(locs)[0]
            except KeyError:
                missing.append(i)
                locs = np.array([-1])
            except InvalidIndexError:
                # i.e. non-scalar key e.g. a tuple.
                # see test_append_different_columns_types_raises
                missing.append(i)
                locs = np.array([-1])

            indexer.append(locs)

        indexer = np.concatenate(indexer)
        return ensure_platform_int(indexer), ensure_platform_int(missing)

    @cache_readonly
    def _index_as_unique(self) -> bool:
        return not self.is_overlapping and self._engine._na_count < 2

    _requires_unique_msg = (
        "cannot handle overlapping indices; use IntervalIndex.get_indexer_non_unique"
    )

    def _convert_slice_indexer(self, key: slice, kind: Literal["loc", "getitem"]):
        if not (key.step is None or key.step == 1):
            # GH#31658 if label-based, we require step == 1,
            #  if positional, we disallow float start/stop
            msg = "label-based slicing with step!=1 is not supported for IntervalIndex"
            if kind == "loc":
                raise ValueError(msg)
            if kind == "getitem":
                if not is_valid_positional_slice(key):
                    # i.e. this cannot be interpreted as a positional slice
                    raise ValueError(msg)

        return super()._convert_slice_indexer(key, kind)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        # integer lookups in Series.__getitem__ are unambiguously
        #  positional in this case
        # error: Item "ExtensionDtype"/"dtype[Any]" of "Union[dtype[Any],
        # ExtensionDtype]" has no attribute "subtype"
        return self.dtype.subtype.kind in "mM"  # type: ignore[union-attr]

    def _maybe_cast_slice_bound(self, label, side: str):
        return getattr(self, side)._maybe_cast_slice_bound(label, side)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        if not isinstance(dtype, IntervalDtype):
            return False
        common_subtype = find_common_type([self.dtype, dtype])
        return not is_object_dtype(common_subtype)

    # --------------------------------------------------------------------

    @cache_readonly
    def left(self) -> Index:
        return Index(self._data.left, copy=False)

    @cache_readonly
    def right(self) -> Index:
        return Index(self._data.right, copy=False)

    @cache_readonly
    def mid(self) -> Index:
        return Index(self._data.mid, copy=False)

    @property
    def length(self) -> Index:
        return Index(self._data.length, copy=False)

    # --------------------------------------------------------------------
    # Set Operations

    def _intersection(self, other, sort):
        """
        intersection specialized to the case with matching dtypes.
        """
        # For IntervalIndex we also know other.closed == self.closed
        if self.left.is_unique and self.right.is_unique:
            taken = self._intersection_unique(other)
        elif other.left.is_unique and other.right.is_unique and self.isna().sum() <= 1:
            # Swap other/self if other is unique and self does not have
            # multiple NaNs
            taken = other._intersection_unique(self)
        else:
            # duplicates
            taken = self._intersection_non_unique(other)

        if sort is None:
            taken = taken.sort_values()

        return taken

    def _intersection_unique(self, other: IntervalIndex) -> IntervalIndex:
        """
        Used when the IntervalIndex does not have any common endpoint,
        no matter left or right.
        Return the intersection with another IntervalIndex.
        Parameters
        ----------
        other : IntervalIndex
        Returns
        -------
        IntervalIndex
        """
        # Note: this is much more performant than super()._intersection(other)
        lindexer = self.left.get_indexer(other.left)
        rindexer = self.right.get_indexer(other.right)

        match = (lindexer == rindexer) & (lindexer != -1)
        indexer = lindexer.take(match.nonzero()[0])
        indexer = unique(indexer)

        return self.take(indexer)

    def _intersection_non_unique(self, other: IntervalIndex) -> IntervalIndex:
        """
        Used when the IntervalIndex does have some common endpoints,
        on either sides.
        Return the intersection with another IntervalIndex.

        Parameters
        ----------
        other : IntervalIndex

        Returns
        -------
        IntervalIndex
        """
        # Note: this is about 3.25x faster than super()._intersection(other)
        #  in IntervalIndexMethod.time_intersection_both_duplicate(1000)
        mask = np.zeros(len(self), dtype=bool)

        if self.hasnans and other.hasnans:
            first_nan_loc = np.arange(len(self))[self.isna()][0]
            mask[first_nan_loc] = True

        other_tups = set(zip(other.left, other.right))
        for i, tup in enumerate(zip(self.left, self.right)):
            if tup in other_tups:
                mask[i] = True

        return self[mask]

    # --------------------------------------------------------------------

    def _get_engine_target(self) -> np.ndarray:
        # Note: we _could_ use libjoin functions by either casting to object
        #  dtype or constructing tuples (faster than constructing Intervals)
        #  but the libjoin fastpaths are no longer fast in these cases.
        raise NotImplementedError(
            "IntervalIndex does not use libjoin fastpaths or pass values to "
            "IndexEngine objects"
        )

    def _from_join_target(self, result):
        raise NotImplementedError("IntervalIndex does not use libjoin fastpaths")

    # TODO: arithmetic operations


def _is_valid_endpoint(endpoint) -> bool:
    """
    Helper for interval_range to check if start/end are valid types.
    """
    return any(
        [
            is_number(endpoint),
            isinstance(endpoint, Timestamp),
            isinstance(endpoint, Timedelta),
            endpoint is None,
        ]
    )


def _is_type_compatible(a, b) -> bool:
    """
    Helper for interval_range to check type compat of start/end/freq.
    """
    is_ts_compat = lambda x: isinstance(x, (Timestamp, BaseOffset))
    is_td_compat = lambda x: isinstance(x, (Timedelta, BaseOffset))
    return (
        (is_number(a) and is_number(b))
        or (is_ts_compat(a) and is_ts_compat(b))
        or (is_td_compat(a) and is_td_compat(b))
        or com.any_none(a, b)
    )


def interval_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    name: Hashable | None = None,
    closed: IntervalClosedType = "right",
) -> IntervalIndex:
    """
    Return a fixed frequency IntervalIndex.

    Parameters
    ----------
    start : numeric or datetime-like, default None
        Left bound for generating intervals.
    end : numeric or datetime-like, default None
        Right bound for generating intervals.
    periods : int, default None
        Number of periods to generate.
    freq : numeric, str, Timedelta, datetime.timedelta, or DateOffset, default None
        The length of each interval. Must be consistent with the type of start
        and end, e.g. 2 for numeric, or '5H' for datetime-like.  Default is 1
        for numeric and 'D' for datetime-like.
    name : str, default None
        Name of the resulting IntervalIndex.
    closed : {'left', 'right', 'both', 'neither'}, default 'right'
        Whether the intervals are closed on the left-side, right-side, both
        or neither.

    Returns
    -------
    IntervalIndex

    See Also
    --------
    IntervalIndex : An Index of intervals that are all closed on the same side.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``IntervalIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end``, inclusively.

    To learn more about datetime-like frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    Numeric ``start`` and  ``end`` is supported.

    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')

    Additionally, datetime-like input is also supported.

    >>> pd.interval_range(start=pd.Timestamp('2017-01-01'),
    ...                   end=pd.Timestamp('2017-01-04'))
    IntervalIndex([(2017-01-01 00:00:00, 2017-01-02 00:00:00],
                   (2017-01-02 00:00:00, 2017-01-03 00:00:00],
                   (2017-01-03 00:00:00, 2017-01-04 00:00:00]],
                  dtype='interval[datetime64[ns], right]')

    The ``freq`` parameter specifies the frequency between the left and right.
    endpoints of the individual intervals within the ``IntervalIndex``.  For
    numeric ``start`` and ``end``, the frequency must also be numeric.

    >>> pd.interval_range(start=0, periods=4, freq=1.5)
    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],
                  dtype='interval[float64, right]')

    Similarly, for datetime-like ``start`` and ``end``, the frequency must be
    convertible to a DateOffset.

    >>> pd.interval_range(start=pd.Timestamp('2017-01-01'),
    ...                   periods=3, freq='MS')
    IntervalIndex([(2017-01-01 00:00:00, 2017-02-01 00:00:00],
                   (2017-02-01 00:00:00, 2017-03-01 00:00:00],
                   (2017-03-01 00:00:00, 2017-04-01 00:00:00]],
                  dtype='interval[datetime64[ns], right]')

    Specify ``start``, ``end``, and ``periods``; the frequency is generated
    automatically (linearly spaced).

    >>> pd.interval_range(start=0, end=6, periods=4)
    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],
              dtype='interval[float64, right]')

    The ``closed`` parameter specifies which endpoints of the individual
    intervals within the ``IntervalIndex`` are closed.

    >>> pd.interval_range(end=5, periods=4, closed='both')
    IntervalIndex([[1, 2], [2, 3], [3, 4], [4, 5]],
                  dtype='interval[int64, both]')
    """
    start = maybe_box_datetimelike(start)
    end = maybe_box_datetimelike(end)
    endpoint = start if start is not None else end

    if freq is None and com.any_none(periods, start, end):
        freq = 1 if is_number(endpoint) else "D"

    if com.count_not_none(start, end, periods, freq) != 3:
        raise ValueError(
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )

    if not _is_valid_endpoint(start):
        raise ValueError(f"start must be numeric or datetime-like, got {start}")
    if not _is_valid_endpoint(end):
        raise ValueError(f"end must be numeric or datetime-like, got {end}")

    periods = validate_periods(periods)

    if freq is not None and not is_number(freq):
        try:
            freq = to_offset(freq)
        except ValueError as err:
            raise ValueError(
                f"freq must be numeric or convertible to DateOffset, got {freq}"
            ) from err

    # verify type compatibility
    if not all(
        [
            _is_type_compatible(start, end),
            _is_type_compatible(start, freq),
            _is_type_compatible(end, freq),
        ]
    ):
        raise TypeError("start, end, freq need to be type compatible")

    # +1 to convert interval count to breaks count (n breaks = n-1 intervals)
    if periods is not None:
        periods += 1

    breaks: np.ndarray | TimedeltaIndex | DatetimeIndex

    if is_number(endpoint):
        if com.all_not_none(start, end, freq):
            # 0.1 ensures we capture end
            breaks = np.arange(start, end + (freq * 0.1), freq)
        else:
            # compute the period/start/end if unspecified (at most one)
            if periods is None:
                periods = int((end - start) // freq) + 1
            elif start is None:
                start = end - (periods - 1) * freq
            elif end is None:
                end = start + (periods - 1) * freq

            breaks = np.linspace(start, end, periods)
        if all(is_integer(x) for x in com.not_none(start, end, freq)):
            # np.linspace always produces float output

            # error: Argument 1 to "maybe_downcast_numeric" has incompatible type
            # "Union[ndarray[Any, Any], TimedeltaIndex, DatetimeIndex]";
            # expected "ndarray[Any, Any]"  [
            breaks = maybe_downcast_numeric(
                breaks,  # type: ignore[arg-type]
                np.dtype("int64"),
            )
    else:
        # delegate to the appropriate range function
        if isinstance(endpoint, Timestamp):
            breaks = date_range(start=start, end=end, periods=periods, freq=freq)
        else:
            breaks = timedelta_range(start=start, end=end, periods=periods, freq=freq)

    return IntervalIndex.from_breaks(breaks, name=name, closed=closed)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\compat.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013-2017 Vinay Sajip.
# Licensed to the Python Software Foundation under a contributor agreement.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
from __future__ import absolute_import

import os
import re
import shutil
import sys

try:
    import ssl
except ImportError:  # pragma: no cover
    ssl = None

if sys.version_info[0] < 3:  # pragma: no cover
    from StringIO import StringIO
    string_types = basestring,
    text_type = unicode
    from types import FileType as file_type
    import __builtin__ as builtins
    import ConfigParser as configparser
    from urlparse import urlparse, urlunparse, urljoin, urlsplit, urlunsplit
    from urllib import (urlretrieve, quote as _quote, unquote, url2pathname,
                        pathname2url, ContentTooShortError, splittype)

    def quote(s):
        if isinstance(s, unicode):
            s = s.encode('utf-8')
        return _quote(s)

    import urllib2
    from urllib2 import (Request, urlopen, URLError, HTTPError,
                         HTTPBasicAuthHandler, HTTPPasswordMgr, HTTPHandler,
                         HTTPRedirectHandler, build_opener)
    if ssl:
        from urllib2 import HTTPSHandler
    import httplib
    import xmlrpclib
    import Queue as queue
    from HTMLParser import HTMLParser
    import htmlentitydefs
    raw_input = raw_input
    from itertools import ifilter as filter
    from itertools import ifilterfalse as filterfalse

    # Leaving this around for now, in case it needs resurrecting in some way
    # _userprog = None
    # def splituser(host):
    # """splituser('user[:passwd]@host[:port]') --> 'user[:passwd]', 'host[:port]'."""
    # global _userprog
    # if _userprog is None:
    # import re
    # _userprog = re.compile('^(.*)@(.*)$')

    # match = _userprog.match(host)
    # if match: return match.group(1, 2)
    # return None, host

else:  # pragma: no cover
    from io import StringIO
    string_types = str,
    text_type = str
    from io import TextIOWrapper as file_type
    import builtins
    import configparser
    from urllib.parse import (urlparse, urlunparse, urljoin, quote, unquote,
                              urlsplit, urlunsplit, splittype)
    from urllib.request import (urlopen, urlretrieve, Request, url2pathname,
                                pathname2url, HTTPBasicAuthHandler,
                                HTTPPasswordMgr, HTTPHandler,
                                HTTPRedirectHandler, build_opener)
    if ssl:
        from urllib.request import HTTPSHandler
    from urllib.error import HTTPError, URLError, ContentTooShortError
    import http.client as httplib
    import urllib.request as urllib2
    import xmlrpc.client as xmlrpclib
    import queue
    from html.parser import HTMLParser
    import html.entities as htmlentitydefs
    raw_input = input
    from itertools import filterfalse
    filter = filter

try:
    from ssl import match_hostname, CertificateError
except ImportError:  # pragma: no cover

    class CertificateError(ValueError):
        pass

    def _dnsname_match(dn, hostname, max_wildcards=1):
        """Matching according to RFC 6125, section 6.4.3

        http://tools.ietf.org/html/rfc6125#section-6.4.3
        """
        pats = []
        if not dn:
            return False

        parts = dn.split('.')
        leftmost, remainder = parts[0], parts[1:]

        wildcards = leftmost.count('*')
        if wildcards > max_wildcards:
            # Issue #17980: avoid denials of service by refusing more
            # than one wildcard per fragment.  A survey of established
            # policy among SSL implementations showed it to be a
            # reasonable choice.
            raise CertificateError(
                "too many wildcards in certificate DNS name: " + repr(dn))

        # speed up common case w/o wildcards
        if not wildcards:
            return dn.lower() == hostname.lower()

        # RFC 6125, section 6.4.3, subitem 1.
        # The client SHOULD NOT attempt to match a presented identifier in which
        # the wildcard character comprises a label other than the left-most label.
        if leftmost == '*':
            # When '*' is a fragment by itself, it matches a non-empty dotless
            # fragment.
            pats.append('[^.]+')
        elif leftmost.startswith('xn--') or hostname.startswith('xn--'):
            # RFC 6125, section 6.4.3, subitem 3.
            # The client SHOULD NOT attempt to match a presented identifier
            # where the wildcard character is embedded within an A-label or
            # U-label of an internationalized domain name.
            pats.append(re.escape(leftmost))
        else:
            # Otherwise, '*' matches any dotless string, e.g. www*
            pats.append(re.escape(leftmost).replace(r'\*', '[^.]*'))

        # add the remaining fragments, ignore any wildcards
        for frag in remainder:
            pats.append(re.escape(frag))

        pat = re.compile(r'\A' + r'\.'.join(pats) + r'\Z', re.IGNORECASE)
        return pat.match(hostname)

    def match_hostname(cert, hostname):
        """Verify that *cert* (in decoded format as returned by
        SSLSocket.getpeercert()) matches the *hostname*.  RFC 2818 and RFC 6125
        rules are followed, but IP addresses are not accepted for *hostname*.

        CertificateError is raised on failure. On success, the function
        returns nothing.
        """
        if not cert:
            raise ValueError("empty or no certificate, match_hostname needs a "
                             "SSL socket or SSL context with either "
                             "CERT_OPTIONAL or CERT_REQUIRED")
        dnsnames = []
        san = cert.get('subjectAltName', ())
        for key, value in san:
            if key == 'DNS':
                if _dnsname_match(value, hostname):
                    return
                dnsnames.append(value)
        if not dnsnames:
            # The subject is only checked when there is no dNSName entry
            # in subjectAltName
            for sub in cert.get('subject', ()):
                for key, value in sub:
                    # XXX according to RFC 2818, the most specific Common Name
                    # must be used.
                    if key == 'commonName':
                        if _dnsname_match(value, hostname):
                            return
                        dnsnames.append(value)
        if len(dnsnames) > 1:
            raise CertificateError("hostname %r "
                                   "doesn't match either of %s" %
                                   (hostname, ', '.join(map(repr, dnsnames))))
        elif len(dnsnames) == 1:
            raise CertificateError("hostname %r "
                                   "doesn't match %r" %
                                   (hostname, dnsnames[0]))
        else:
            raise CertificateError("no appropriate commonName or "
                                   "subjectAltName fields were found")


try:
    from types import SimpleNamespace as Container
except ImportError:  # pragma: no cover

    class Container(object):
        """
        A generic container for when multiple values need to be returned
        """

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


try:
    from shutil import which
except ImportError:  # pragma: no cover
    # Implementation from Python 3.3
    def which(cmd, mode=os.F_OK | os.X_OK, path=None):
        """Given a command, mode, and a PATH string, return the path which
        conforms to the given mode on the PATH, or None if there is no such
        file.

        `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result
        of os.environ.get("PATH"), or can be overridden with a custom search
        path.

        """

        # Check that a given file can be accessed with the correct mode.
        # Additionally check that `file` is not a directory, as on Windows
        # directories pass the os.access check.
        def _access_check(fn, mode):
            return (os.path.exists(fn) and os.access(fn, mode) and not os.path.isdir(fn))

        # If we're given a path with a directory part, look it up directly rather
        # than referring to PATH directories. This includes checking relative to the
        # current directory, e.g. ./script
        if os.path.dirname(cmd):
            if _access_check(cmd, mode):
                return cmd
            return None

        if path is None:
            path = os.environ.get("PATH", os.defpath)
        if not path:
            return None
        path = path.split(os.pathsep)

        if sys.platform == "win32":
            # The current directory takes precedence on Windows.
            if os.curdir not in path:
                path.insert(0, os.curdir)

            # PATHEXT is necessary to check on Windows.
            pathext = os.environ.get("PATHEXT", "").split(os.pathsep)
            # See if the given file matches any of the expected path extensions.
            # This will allow us to short circuit when given "python.exe".
            # If it does match, only test that one, otherwise we have to try
            # others.
            if any(cmd.lower().endswith(ext.lower()) for ext in pathext):
                files = [cmd]
            else:
                files = [cmd + ext for ext in pathext]
        else:
            # On other platforms you don't have things like PATHEXT to tell you
            # what file suffixes are executable, so just pass on cmd as-is.
            files = [cmd]

        seen = set()
        for dir in path:
            normdir = os.path.normcase(dir)
            if normdir not in seen:
                seen.add(normdir)
                for thefile in files:
                    name = os.path.join(dir, thefile)
                    if _access_check(name, mode):
                        return name
        return None


# ZipFile is a context manager in 2.7, but not in 2.6

from zipfile import ZipFile as BaseZipFile

if hasattr(BaseZipFile, '__enter__'):  # pragma: no cover
    ZipFile = BaseZipFile
else:  # pragma: no cover
    from zipfile import ZipExtFile as BaseZipExtFile

    class ZipExtFile(BaseZipExtFile):

        def __init__(self, base):
            self.__dict__.update(base.__dict__)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self.close()
            # return None, so if an exception occurred, it will propagate

    class ZipFile(BaseZipFile):

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self.close()
            # return None, so if an exception occurred, it will propagate

        def open(self, *args, **kwargs):
            base = BaseZipFile.open(self, *args, **kwargs)
            return ZipExtFile(base)


try:
    from platform import python_implementation
except ImportError:  # pragma: no cover

    def python_implementation():
        """Return a string identifying the Python implementation."""
        if 'PyPy' in sys.version:
            return 'PyPy'
        if os.name == 'java':
            return 'Jython'
        if sys.version.startswith('IronPython'):
            return 'IronPython'
        return 'CPython'


import sysconfig

try:
    callable = callable
except NameError:  # pragma: no cover
    from collections.abc import Callable

    def callable(obj):
        return isinstance(obj, Callable)


try:
    fsencode = os.fsencode
    fsdecode = os.fsdecode
except AttributeError:  # pragma: no cover
    # Issue #99: on some systems (e.g. containerised),
    # sys.getfilesystemencoding() returns None, and we need a real value,
    # so fall back to utf-8. From the CPython 2.7 docs relating to Unix and
    # sys.getfilesystemencoding(): the return value is "the user’s preference
    # according to the result of nl_langinfo(CODESET), or None if the
    # nl_langinfo(CODESET) failed."
    _fsencoding = sys.getfilesystemencoding() or 'utf-8'
    if _fsencoding == 'mbcs':
        _fserrors = 'strict'
    else:
        _fserrors = 'surrogateescape'

    def fsencode(filename):
        if isinstance(filename, bytes):
            return filename
        elif isinstance(filename, text_type):
            return filename.encode(_fsencoding, _fserrors)
        else:
            raise TypeError("expect bytes or str, not %s" %
                            type(filename).__name__)

    def fsdecode(filename):
        if isinstance(filename, text_type):
            return filename
        elif isinstance(filename, bytes):
            return filename.decode(_fsencoding, _fserrors)
        else:
            raise TypeError("expect bytes or str, not %s" %
                            type(filename).__name__)


try:
    from tokenize import detect_encoding
except ImportError:  # pragma: no cover
    from codecs import BOM_UTF8, lookup

    cookie_re = re.compile(r"coding[:=]\s*([-\w.]+)")

    def _get_normal_name(orig_enc):
        """Imitates get_normal_name in tokenizer.c."""
        # Only care about the first 12 characters.
        enc = orig_enc[:12].lower().replace("_", "-")
        if enc == "utf-8" or enc.startswith("utf-8-"):
            return "utf-8"
        if enc in ("latin-1", "iso-8859-1", "iso-latin-1") or \
           enc.startswith(("latin-1-", "iso-8859-1-", "iso-latin-1-")):
            return "iso-8859-1"
        return orig_enc

    def detect_encoding(readline):
        """
        The detect_encoding() function is used to detect the encoding that should
        be used to decode a Python source file.  It requires one argument, readline,
        in the same way as the tokenize() generator.

        It will call readline a maximum of twice, and return the encoding used
        (as a string) and a list of any lines (left as bytes) it has read in.

        It detects the encoding from the presence of a utf-8 bom or an encoding
        cookie as specified in pep-0263.  If both a bom and a cookie are present,
        but disagree, a SyntaxError will be raised.  If the encoding cookie is an
        invalid charset, raise a SyntaxError.  Note that if a utf-8 bom is found,
        'utf-8-sig' is returned.

        If no encoding is specified, then the default of 'utf-8' will be returned.
        """
        try:
            filename = readline.__self__.name
        except AttributeError:
            filename = None
        bom_found = False
        encoding = None
        default = 'utf-8'

        def read_or_stop():
            try:
                return readline()
            except StopIteration:
                return b''

        def find_cookie(line):
            try:
                # Decode as UTF-8. Either the line is an encoding declaration,
                # in which case it should be pure ASCII, or it must be UTF-8
                # per default encoding.
                line_string = line.decode('utf-8')
            except UnicodeDecodeError:
                msg = "invalid or missing encoding declaration"
                if filename is not None:
                    msg = '{} for {!r}'.format(msg, filename)
                raise SyntaxError(msg)

            matches = cookie_re.findall(line_string)
            if not matches:
                return None
            encoding = _get_normal_name(matches[0])
            try:
                codec = lookup(encoding)
            except LookupError:
                # This behaviour mimics the Python interpreter
                if filename is None:
                    msg = "unknown encoding: " + encoding
                else:
                    msg = "unknown encoding for {!r}: {}".format(
                        filename, encoding)
                raise SyntaxError(msg)

            if bom_found:
                if codec.name != 'utf-8':
                    # This behaviour mimics the Python interpreter
                    if filename is None:
                        msg = 'encoding problem: utf-8'
                    else:
                        msg = 'encoding problem for {!r}: utf-8'.format(
                            filename)
                    raise SyntaxError(msg)
                encoding += '-sig'
            return encoding

        first = read_or_stop()
        if first.startswith(BOM_UTF8):
            bom_found = True
            first = first[3:]
            default = 'utf-8-sig'
        if not first:
            return default, []

        encoding = find_cookie(first)
        if encoding:
            return encoding, [first]

        second = read_or_stop()
        if not second:
            return default, [first]

        encoding = find_cookie(second)
        if encoding:
            return encoding, [first, second]

        return default, [first, second]


# For converting & <-> &amp; etc.
try:
    from html import escape
except ImportError:
    from cgi import escape
if sys.version_info[:2] < (3, 4):
    unescape = HTMLParser().unescape
else:
    from html import unescape

try:
    from collections import ChainMap
except ImportError:  # pragma: no cover
    from collections import MutableMapping

    try:
        from reprlib import recursive_repr as _recursive_repr
    except ImportError:

        def _recursive_repr(fillvalue='...'):
            '''
            Decorator to make a repr function return fillvalue for a recursive
            call
            '''

            def decorating_function(user_function):
                repr_running = set()

                def wrapper(self):
                    key = id(self), get_ident()
                    if key in repr_running:
                        return fillvalue
                    repr_running.add(key)
                    try:
                        result = user_function(self)
                    finally:
                        repr_running.discard(key)
                    return result

                # Can't use functools.wraps() here because of bootstrap issues
                wrapper.__module__ = getattr(user_function, '__module__')
                wrapper.__doc__ = getattr(user_function, '__doc__')
                wrapper.__name__ = getattr(user_function, '__name__')
                wrapper.__annotations__ = getattr(user_function,
                                                  '__annotations__', {})
                return wrapper

            return decorating_function

    class ChainMap(MutableMapping):
        '''
        A ChainMap groups multiple dicts (or other mappings) together
        to create a single, updateable view.

        The underlying mappings are stored in a list.  That list is public and can
        accessed or updated using the *maps* attribute.  There is no other state.

        Lookups search the underlying mappings successively until a key is found.
        In contrast, writes, updates, and deletions only operate on the first
        mapping.
        '''

        def __init__(self, *maps):
            '''Initialize a ChainMap by setting *maps* to the given mappings.
            If no mappings are provided, a single empty dictionary is used.

            '''
            self.maps = list(maps) or [{}]  # always at least one map

        def __missing__(self, key):
            raise KeyError(key)

        def __getitem__(self, key):
            for mapping in self.maps:
                try:
                    return mapping[
                        key]  # can't use 'key in mapping' with defaultdict
                except KeyError:
                    pass
            return self.__missing__(
                key)  # support subclasses that define __missing__

        def get(self, key, default=None):
            return self[key] if key in self else default

        def __len__(self):
            return len(set().union(
                *self.maps))  # reuses stored hash values if possible

        def __iter__(self):
            return iter(set().union(*self.maps))

        def __contains__(self, key):
            return any(key in m for m in self.maps)

        def __bool__(self):
            return any(self.maps)

        @_recursive_repr()
        def __repr__(self):
            return '{0.__class__.__name__}({1})'.format(
                self, ', '.join(map(repr, self.maps)))

        @classmethod
        def fromkeys(cls, iterable, *args):
            'Create a ChainMap with a single dict created from the iterable.'
            return cls(dict.fromkeys(iterable, *args))

        def copy(self):
            'New ChainMap or subclass with a new copy of maps[0] and refs to maps[1:]'
            return self.__class__(self.maps[0].copy(), *self.maps[1:])

        __copy__ = copy

        def new_child(self):  # like Django's Context.push()
            'New ChainMap with a new dict followed by all previous maps.'
            return self.__class__({}, *self.maps)

        @property
        def parents(self):  # like Django's Context.pop()
            'New ChainMap from maps[1:].'
            return self.__class__(*self.maps[1:])

        def __setitem__(self, key, value):
            self.maps[0][key] = value

        def __delitem__(self, key):
            try:
                del self.maps[0][key]
            except KeyError:
                raise KeyError(
                    'Key not found in the first mapping: {!r}'.format(key))

        def popitem(self):
            'Remove and return an item pair from maps[0]. Raise KeyError is maps[0] is empty.'
            try:
                return self.maps[0].popitem()
            except KeyError:
                raise KeyError('No keys found in the first mapping.')

        def pop(self, key, *args):
            'Remove *key* from maps[0] and return its value. Raise KeyError if *key* not in maps[0].'
            try:
                return self.maps[0].pop(key, *args)
            except KeyError:
                raise KeyError(
                    'Key not found in the first mapping: {!r}'.format(key))

        def clear(self):
            'Clear maps[0], leaving maps[1:] intact.'
            self.maps[0].clear()


try:
    from importlib.util import cache_from_source  # Python >= 3.4
except ImportError:  # pragma: no cover

    def cache_from_source(path, debug_override=None):
        assert path.endswith('.py')
        if debug_override is None:
            debug_override = __debug__
        if debug_override:
            suffix = 'c'
        else:
            suffix = 'o'
        return path + suffix


try:
    from collections import OrderedDict
except ImportError:  # pragma: no cover
    # {{{ http://code.activestate.com/recipes/576693/ (r9)
    # Backport of OrderedDict() class that runs on Python 2.4, 2.5, 2.6, 2.7 and pypy.
    # Passes Python2.7's test suite and incorporates all the latest updates.
    try:
        from thread import get_ident as _get_ident
    except ImportError:
        from dummy_thread import get_ident as _get_ident

    try:
        from _abcoll import KeysView, ValuesView, ItemsView
    except ImportError:
        pass

    class OrderedDict(dict):
        'Dictionary that remembers insertion order'

        # An inherited dict maps keys to values.
        # The inherited dict provides __getitem__, __len__, __contains__, and get.
        # The remaining methods are order-aware.
        # Big-O running times for all methods are the same as for regular dictionaries.

        # The internal self.__map dictionary maps keys to links in a doubly linked list.
        # The circular doubly linked list starts and ends with a sentinel element.
        # The sentinel element never gets deleted (this simplifies the algorithm).
        # Each link is stored as a list of length three:  [PREV, NEXT, KEY].

        def __init__(self, *args, **kwds):
            '''Initialize an ordered dictionary.  Signature is the same as for
            regular dictionaries, but keyword arguments are not recommended
            because their insertion order is arbitrary.

            '''
            if len(args) > 1:
                raise TypeError('expected at most 1 arguments, got %d' %
                                len(args))
            try:
                self.__root
            except AttributeError:
                self.__root = root = []  # sentinel node
                root[:] = [root, root, None]
                self.__map = {}
            self.__update(*args, **kwds)

        def __setitem__(self, key, value, dict_setitem=dict.__setitem__):
            'od.__setitem__(i, y) <==> od[i]=y'
            # Setting a new item creates a new link which goes at the end of the linked
            # list, and the inherited dictionary is updated with the new key/value pair.
            if key not in self:
                root = self.__root
                last = root[0]
                last[1] = root[0] = self.__map[key] = [last, root, key]
            dict_setitem(self, key, value)

        def __delitem__(self, key, dict_delitem=dict.__delitem__):
            'od.__delitem__(y) <==> del od[y]'
            # Deleting an existing item uses self.__map to find the link which is
            # then removed by updating the links in the predecessor and successor nodes.
            dict_delitem(self, key)
            link_prev, link_next, key = self.__map.pop(key)
            link_prev[1] = link_next
            link_next[0] = link_prev

        def __iter__(self):
            'od.__iter__() <==> iter(od)'
            root = self.__root
            curr = root[1]
            while curr is not root:
                yield curr[2]
                curr = curr[1]

        def __reversed__(self):
            'od.__reversed__() <==> reversed(od)'
            root = self.__root
            curr = root[0]
            while curr is not root:
                yield curr[2]
                curr = curr[0]

        def clear(self):
            'od.clear() -> None.  Remove all items from od.'
            try:
                for node in self.__map.itervalues():
                    del node[:]
                root = self.__root
                root[:] = [root, root, None]
                self.__map.clear()
            except AttributeError:
                pass
            dict.clear(self)

        def popitem(self, last=True):
            '''od.popitem() -> (k, v), return and remove a (key, value) pair.
            Pairs are returned in LIFO order if last is true or FIFO order if false.

            '''
            if not self:
                raise KeyError('dictionary is empty')
            root = self.__root
            if last:
                link = root[0]
                link_prev = link[0]
                link_prev[1] = root
                root[0] = link_prev
            else:
                link = root[1]
                link_next = link[1]
                root[1] = link_next
                link_next[0] = root
            key = link[2]
            del self.__map[key]
            value = dict.pop(self, key)
            return key, value

        # -- the following methods do not depend on the internal structure --

        def keys(self):
            'od.keys() -> list of keys in od'
            return list(self)

        def values(self):
            'od.values() -> list of values in od'
            return [self[key] for key in self]

        def items(self):
            'od.items() -> list of (key, value) pairs in od'
            return [(key, self[key]) for key in self]

        def iterkeys(self):
            'od.iterkeys() -> an iterator over the keys in od'
            return iter(self)

        def itervalues(self):
            'od.itervalues -> an iterator over the values in od'
            for k in self:
                yield self[k]

        def iteritems(self):
            'od.iteritems -> an iterator over the (key, value) items in od'
            for k in self:
                yield (k, self[k])

        def update(*args, **kwds):
            '''od.update(E, **F) -> None.  Update od from dict/iterable E and F.

            If E is a dict instance, does:           for k in E: od[k] = E[k]
            If E has a .keys() method, does:         for k in E.keys(): od[k] = E[k]
            Or if E is an iterable of items, does:   for k, v in E: od[k] = v
            In either case, this is followed by:     for k, v in F.items(): od[k] = v

            '''
            if len(args) > 2:
                raise TypeError('update() takes at most 2 positional '
                                'arguments (%d given)' % (len(args), ))
            elif not args:
                raise TypeError('update() takes at least 1 argument (0 given)')
            self = args[0]
            # Make progressively weaker assumptions about "other"
            other = ()
            if len(args) == 2:
                other = args[1]
            if isinstance(other, dict):
                for key in other:
                    self[key] = other[key]
            elif hasattr(other, 'keys'):
                for key in other.keys():
                    self[key] = other[key]
            else:
                for key, value in other:
                    self[key] = value
            for key, value in kwds.items():
                self[key] = value

        __update = update  # let subclasses override update without breaking __init__

        __marker = object()

        def pop(self, key, default=__marker):
            '''od.pop(k[,d]) -> v, remove specified key and return the corresponding value.
            If key is not found, d is returned if given, otherwise KeyError is raised.

            '''
            if key in self:
                result = self[key]
                del self[key]
                return result
            if default is self.__marker:
                raise KeyError(key)
            return default

        def setdefault(self, key, default=None):
            'od.setdefault(k[,d]) -> od.get(k,d), also set od[k]=d if k not in od'
            if key in self:
                return self[key]
            self[key] = default
            return default

        def __repr__(self, _repr_running=None):
            'od.__repr__() <==> repr(od)'
            if not _repr_running:
                _repr_running = {}
            call_key = id(self), _get_ident()
            if call_key in _repr_running:
                return '...'
            _repr_running[call_key] = 1
            try:
                if not self:
                    return '%s()' % (self.__class__.__name__, )
                return '%s(%r)' % (self.__class__.__name__, self.items())
            finally:
                del _repr_running[call_key]

        def __reduce__(self):
            'Return state information for pickling'
            items = [[k, self[k]] for k in self]
            inst_dict = vars(self).copy()
            for k in vars(OrderedDict()):
                inst_dict.pop(k, None)
            if inst_dict:
                return (self.__class__, (items, ), inst_dict)
            return self.__class__, (items, )

        def copy(self):
            'od.copy() -> a shallow copy of od'
            return self.__class__(self)

        @classmethod
        def fromkeys(cls, iterable, value=None):
            '''OD.fromkeys(S[, v]) -> New ordered dictionary with keys from S
            and values equal to v (which defaults to None).

            '''
            d = cls()
            for key in iterable:
                d[key] = value
            return d

        def __eq__(self, other):
            '''od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive
            while comparison to a regular mapping is order-insensitive.

            '''
            if isinstance(other, OrderedDict):
                return len(self) == len(
                    other) and self.items() == other.items()
            return dict.__eq__(self, other)

        def __ne__(self, other):
            return not self == other

        # -- the following methods are only used in Python 2.7 --

        def viewkeys(self):
            "od.viewkeys() -> a set-like object providing a view on od's keys"
            return KeysView(self)

        def viewvalues(self):
            "od.viewvalues() -> an object providing a view on od's values"
            return ValuesView(self)

        def viewitems(self):
            "od.viewitems() -> a set-like object providing a view on od's items"
            return ItemsView(self)


try:
    from logging.config import BaseConfigurator, valid_ident
except ImportError:  # pragma: no cover
    IDENTIFIER = re.compile('^[a-z_][a-z0-9_]*$', re.I)

    def valid_ident(s):
        m = IDENTIFIER.match(s)
        if not m:
            raise ValueError('Not a valid Python identifier: %r' % s)
        return True

    # The ConvertingXXX classes are wrappers around standard Python containers,
    # and they serve to convert any suitable values in the container. The
    # conversion converts base dicts, lists and tuples to their wrapped
    # equivalents, whereas strings which match a conversion format are converted
    # appropriately.
    #
    # Each wrapper should have a configurator attribute holding the actual
    # configurator to use for conversion.

    class ConvertingDict(dict):
        """A converting dictionary wrapper."""

        def __getitem__(self, key):
            value = dict.__getitem__(self, key)
            result = self.configurator.convert(value)
            # If the converted value is different, save for next time
            if value is not result:
                self[key] = result
                if type(result) in (ConvertingDict, ConvertingList,
                                    ConvertingTuple):
                    result.parent = self
                    result.key = key
            return result

        def get(self, key, default=None):
            value = dict.get(self, key, default)
            result = self.configurator.convert(value)
            # If the converted value is different, save for next time
            if value is not result:
                self[key] = result
                if type(result) in (ConvertingDict, ConvertingList,
                                    ConvertingTuple):
                    result.parent = self
                    result.key = key
            return result

    def pop(self, key, default=None):
        value = dict.pop(self, key, default)
        result = self.configurator.convert(value)
        if value is not result:
            if type(result) in (ConvertingDict, ConvertingList,
                                ConvertingTuple):
                result.parent = self
                result.key = key
        return result

    class ConvertingList(list):
        """A converting list wrapper."""

        def __getitem__(self, key):
            value = list.__getitem__(self, key)
            result = self.configurator.convert(value)
            # If the converted value is different, save for next time
            if value is not result:
                self[key] = result
                if type(result) in (ConvertingDict, ConvertingList,
                                    ConvertingTuple):
                    result.parent = self
                    result.key = key
            return result

        def pop(self, idx=-1):
            value = list.pop(self, idx)
            result = self.configurator.convert(value)
            if value is not result:
                if type(result) in (ConvertingDict, ConvertingList,
                                    ConvertingTuple):
                    result.parent = self
            return result

    class ConvertingTuple(tuple):
        """A converting tuple wrapper."""

        def __getitem__(self, key):
            value = tuple.__getitem__(self, key)
            result = self.configurator.convert(value)
            if value is not result:
                if type(result) in (ConvertingDict, ConvertingList,
                                    ConvertingTuple):
                    result.parent = self
                    result.key = key
            return result

    class BaseConfigurator(object):
        """
        The configurator base class which defines some useful defaults.
        """

        CONVERT_PATTERN = re.compile(r'^(?P<prefix>[a-z]+)://(?P<suffix>.*)$')

        WORD_PATTERN = re.compile(r'^\s*(\w+)\s*')
        DOT_PATTERN = re.compile(r'^\.\s*(\w+)\s*')
        INDEX_PATTERN = re.compile(r'^\[\s*(\w+)\s*\]\s*')
        DIGIT_PATTERN = re.compile(r'^\d+$')

        value_converters = {
            'ext': 'ext_convert',
            'cfg': 'cfg_convert',
        }

        # We might want to use a different one, e.g. importlib
        importer = staticmethod(__import__)

        def __init__(self, config):
            self.config = ConvertingDict(config)
            self.config.configurator = self

        def resolve(self, s):
            """
            Resolve strings to objects using standard import and attribute
            syntax.
            """
            name = s.split('.')
            used = name.pop(0)
            try:
                found = self.importer(used)
                for frag in name:
                    used += '.' + frag
                    try:
                        found = getattr(found, frag)
                    except AttributeError:
                        self.importer(used)
                        found = getattr(found, frag)
                return found
            except ImportError:
                e, tb = sys.exc_info()[1:]
                v = ValueError('Cannot resolve %r: %s' % (s, e))
                v.__cause__, v.__traceback__ = e, tb
                raise v

        def ext_convert(self, value):
            """Default converter for the ext:// protocol."""
            return self.resolve(value)

        def cfg_convert(self, value):
            """Default converter for the cfg:// protocol."""
            rest = value
            m = self.WORD_PATTERN.match(rest)
            if m is None:
                raise ValueError("Unable to convert %r" % value)
            else:
                rest = rest[m.end():]
                d = self.config[m.groups()[0]]
                while rest:
                    m = self.DOT_PATTERN.match(rest)
                    if m:
                        d = d[m.groups()[0]]
                    else:
                        m = self.INDEX_PATTERN.match(rest)
                        if m:
                            idx = m.groups()[0]
                            if not self.DIGIT_PATTERN.match(idx):
                                d = d[idx]
                            else:
                                try:
                                    n = int(
                                        idx
                                    )  # try as number first (most likely)
                                    d = d[n]
                                except TypeError:
                                    d = d[idx]
                    if m:
                        rest = rest[m.end():]
                    else:
                        raise ValueError('Unable to convert '
                                         '%r at %r' % (value, rest))
            # rest should be empty
            return d

        def convert(self, value):
            """
            Convert values to an appropriate type. dicts, lists and tuples are
            replaced by their converting alternatives. Strings are checked to
            see if they have a conversion format and are converted if they do.
            """
            if not isinstance(value, ConvertingDict) and isinstance(
                    value, dict):
                value = ConvertingDict(value)
                value.configurator = self
            elif not isinstance(value, ConvertingList) and isinstance(
                    value, list):
                value = ConvertingList(value)
                value.configurator = self
            elif not isinstance(value, ConvertingTuple) and isinstance(value, tuple):
                value = ConvertingTuple(value)
                value.configurator = self
            elif isinstance(value, string_types):
                m = self.CONVERT_PATTERN.match(value)
                if m:
                    d = m.groupdict()
                    prefix = d['prefix']
                    converter = self.value_converters.get(prefix, None)
                    if converter:
                        suffix = d['suffix']
                        converter = getattr(self, converter)
                        value = converter(suffix)
            return value

        def configure_custom(self, config):
            """Configure an object with a user-supplied factory."""
            c = config.pop('()')
            if not callable(c):
                c = self.resolve(c)
            props = config.pop('.', None)
            # Check for valid identifiers
            kwargs = dict([(k, config[k]) for k in config if valid_ident(k)])
            result = c(**kwargs)
            if props:
                for name, value in props.items():
                    setattr(result, name, value)
            return result

        def as_tuple(self, value):
            """Utility function which converts lists to tuples."""
            if isinstance(value, list):
                value = tuple(value)
            return value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\cli.py ===
from __future__ import annotations

import ast
import collections.abc as cabc
import importlib.metadata
import inspect
import os
import platform
import re
import sys
import traceback
import typing as t
from functools import update_wrapper
from operator import itemgetter
from types import ModuleType

import click
from click.core import ParameterSource
from werkzeug import run_simple
from werkzeug.serving import is_running_from_reloader
from werkzeug.utils import import_string

from .globals import current_app
from .helpers import get_debug_flag
from .helpers import get_load_dotenv

if t.TYPE_CHECKING:
    import ssl

    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment

    from .app import Flask


class NoAppException(click.UsageError):
    """Raised if an application cannot be found or loaded."""


def find_best_app(module: ModuleType) -> Flask:
    """Given a module instance this tries to find the best possible
    application in the module or raises an exception.
    """
    from . import Flask

    # Search for the most common names first.
    for attr_name in ("app", "application"):
        app = getattr(module, attr_name, None)

        if isinstance(app, Flask):
            return app

    # Otherwise find the only object that is a Flask instance.
    matches = [v for v in module.__dict__.values() if isinstance(v, Flask)]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise NoAppException(
            "Detected multiple Flask applications in module"
            f" '{module.__name__}'. Use '{module.__name__}:name'"
            " to specify the correct one."
        )

    # Search for app factory functions.
    for attr_name in ("create_app", "make_app"):
        app_factory = getattr(module, attr_name, None)

        if inspect.isfunction(app_factory):
            try:
                app = app_factory()

                if isinstance(app, Flask):
                    return app
            except TypeError as e:
                if not _called_with_wrong_args(app_factory):
                    raise

                raise NoAppException(
                    f"Detected factory '{attr_name}' in module '{module.__name__}',"
                    " but could not call it without arguments. Use"
                    f" '{module.__name__}:{attr_name}(args)'"
                    " to specify arguments."
                ) from e

    raise NoAppException(
        "Failed to find Flask application or factory in module"
        f" '{module.__name__}'. Use '{module.__name__}:name'"
        " to specify one."
    )


def _called_with_wrong_args(f: t.Callable[..., Flask]) -> bool:
    """Check whether calling a function raised a ``TypeError`` because
    the call failed or because something in the factory raised the
    error.

    :param f: The function that was called.
    :return: ``True`` if the call failed.
    """
    tb = sys.exc_info()[2]

    try:
        while tb is not None:
            if tb.tb_frame.f_code is f.__code__:
                # In the function, it was called successfully.
                return False

            tb = tb.tb_next

        # Didn't reach the function.
        return True
    finally:
        # Delete tb to break a circular reference.
        # https://docs.python.org/2/library/sys.html#sys.exc_info
        del tb


def find_app_by_string(module: ModuleType, app_name: str) -> Flask:
    """Check if the given string is a variable name or a function. Call
    a function to get the app instance, or return the variable directly.
    """
    from . import Flask

    # Parse app_name as a single expression to determine if it's a valid
    # attribute name or function call.
    try:
        expr = ast.parse(app_name.strip(), mode="eval").body
    except SyntaxError:
        raise NoAppException(
            f"Failed to parse {app_name!r} as an attribute name or function call."
        ) from None

    if isinstance(expr, ast.Name):
        name = expr.id
        args = []
        kwargs = {}
    elif isinstance(expr, ast.Call):
        # Ensure the function name is an attribute name only.
        if not isinstance(expr.func, ast.Name):
            raise NoAppException(
                f"Function reference must be a simple name: {app_name!r}."
            )

        name = expr.func.id

        # Parse the positional and keyword arguments as literals.
        try:
            args = [ast.literal_eval(arg) for arg in expr.args]
            kwargs = {
                kw.arg: ast.literal_eval(kw.value)
                for kw in expr.keywords
                if kw.arg is not None
            }
        except ValueError:
            # literal_eval gives cryptic error messages, show a generic
            # message with the full expression instead.
            raise NoAppException(
                f"Failed to parse arguments as literal values: {app_name!r}."
            ) from None
    else:
        raise NoAppException(
            f"Failed to parse {app_name!r} as an attribute name or function call."
        )

    try:
        attr = getattr(module, name)
    except AttributeError as e:
        raise NoAppException(
            f"Failed to find attribute {name!r} in {module.__name__!r}."
        ) from e

    # If the attribute is a function, call it with any args and kwargs
    # to get the real application.
    if inspect.isfunction(attr):
        try:
            app = attr(*args, **kwargs)
        except TypeError as e:
            if not _called_with_wrong_args(attr):
                raise

            raise NoAppException(
                f"The factory {app_name!r} in module"
                f" {module.__name__!r} could not be called with the"
                " specified arguments."
            ) from e
    else:
        app = attr

    if isinstance(app, Flask):
        return app

    raise NoAppException(
        "A valid Flask application was not obtained from"
        f" '{module.__name__}:{app_name}'."
    )


def prepare_import(path: str) -> str:
    """Given a filename this will try to calculate the python path, add it
    to the search path and return the actual module name that is expected.
    """
    path = os.path.realpath(path)

    fname, ext = os.path.splitext(path)
    if ext == ".py":
        path = fname

    if os.path.basename(path) == "__init__":
        path = os.path.dirname(path)

    module_name = []

    # move up until outside package structure (no __init__.py)
    while True:
        path, name = os.path.split(path)
        module_name.append(name)

        if not os.path.exists(os.path.join(path, "__init__.py")):
            break

    if sys.path[0] != path:
        sys.path.insert(0, path)

    return ".".join(module_name[::-1])


@t.overload
def locate_app(
    module_name: str, app_name: str | None, raise_if_not_found: t.Literal[True] = True
) -> Flask: ...


@t.overload
def locate_app(
    module_name: str, app_name: str | None, raise_if_not_found: t.Literal[False] = ...
) -> Flask | None: ...


def locate_app(
    module_name: str, app_name: str | None, raise_if_not_found: bool = True
) -> Flask | None:
    try:
        __import__(module_name)
    except ImportError:
        # Reraise the ImportError if it occurred within the imported module.
        # Determine this by checking whether the trace has a depth > 1.
        if sys.exc_info()[2].tb_next:  # type: ignore[union-attr]
            raise NoAppException(
                f"While importing {module_name!r}, an ImportError was"
                f" raised:\n\n{traceback.format_exc()}"
            ) from None
        elif raise_if_not_found:
            raise NoAppException(f"Could not import {module_name!r}.") from None
        else:
            return None

    module = sys.modules[module_name]

    if app_name is None:
        return find_best_app(module)
    else:
        return find_app_by_string(module, app_name)


def get_version(ctx: click.Context, param: click.Parameter, value: t.Any) -> None:
    if not value or ctx.resilient_parsing:
        return

    flask_version = importlib.metadata.version("flask")
    werkzeug_version = importlib.metadata.version("werkzeug")

    click.echo(
        f"Python {platform.python_version()}\n"
        f"Flask {flask_version}\n"
        f"Werkzeug {werkzeug_version}",
        color=ctx.color,
    )
    ctx.exit()


version_option = click.Option(
    ["--version"],
    help="Show the Flask version.",
    expose_value=False,
    callback=get_version,
    is_flag=True,
    is_eager=True,
)


class ScriptInfo:
    """Helper object to deal with Flask applications.  This is usually not
    necessary to interface with as it's used internally in the dispatching
    to click.  In future versions of Flask this object will most likely play
    a bigger role.  Typically it's created automatically by the
    :class:`FlaskGroup` but you can also manually create it and pass it
    onwards as click object.

    .. versionchanged:: 3.1
        Added the ``load_dotenv_defaults`` parameter and attribute.
    """

    def __init__(
        self,
        app_import_path: str | None = None,
        create_app: t.Callable[..., Flask] | None = None,
        set_debug_flag: bool = True,
        load_dotenv_defaults: bool = True,
    ) -> None:
        #: Optionally the import path for the Flask application.
        self.app_import_path = app_import_path
        #: Optionally a function that is passed the script info to create
        #: the instance of the application.
        self.create_app = create_app
        #: A dictionary with arbitrary data that can be associated with
        #: this script info.
        self.data: dict[t.Any, t.Any] = {}
        self.set_debug_flag = set_debug_flag

        self.load_dotenv_defaults = get_load_dotenv(load_dotenv_defaults)
        """Whether default ``.flaskenv`` and ``.env`` files should be loaded.

        ``ScriptInfo`` doesn't load anything, this is for reference when doing
        the load elsewhere during processing.

        .. versionadded:: 3.1
        """

        self._loaded_app: Flask | None = None

    def load_app(self) -> Flask:
        """Loads the Flask app (if not yet loaded) and returns it.  Calling
        this multiple times will just result in the already loaded app to
        be returned.
        """
        if self._loaded_app is not None:
            return self._loaded_app
        app: Flask | None = None
        if self.create_app is not None:
            app = self.create_app()
        else:
            if self.app_import_path:
                path, name = (
                    re.split(r":(?![\\/])", self.app_import_path, maxsplit=1) + [None]
                )[:2]
                import_name = prepare_import(path)
                app = locate_app(import_name, name)
            else:
                for path in ("wsgi.py", "app.py"):
                    import_name = prepare_import(path)
                    app = locate_app(import_name, None, raise_if_not_found=False)

                    if app is not None:
                        break

        if app is None:
            raise NoAppException(
                "Could not locate a Flask application. Use the"
                " 'flask --app' option, 'FLASK_APP' environment"
                " variable, or a 'wsgi.py' or 'app.py' file in the"
                " current directory."
            )

        if self.set_debug_flag:
            # Update the app's debug flag through the descriptor so that
            # other values repopulate as well.
            app.debug = get_debug_flag()

        self._loaded_app = app
        return app


pass_script_info = click.make_pass_decorator(ScriptInfo, ensure=True)

F = t.TypeVar("F", bound=t.Callable[..., t.Any])


def with_appcontext(f: F) -> F:
    """Wraps a callback so that it's guaranteed to be executed with the
    script's application context.

    Custom commands (and their options) registered under ``app.cli`` or
    ``blueprint.cli`` will always have an app context available, this
    decorator is not required in that case.

    .. versionchanged:: 2.2
        The app context is active for subcommands as well as the
        decorated callback. The app context is always available to
        ``app.cli`` command and parameter callbacks.
    """

    @click.pass_context
    def decorator(ctx: click.Context, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if not current_app:
            app = ctx.ensure_object(ScriptInfo).load_app()
            ctx.with_resource(app.app_context())

        return ctx.invoke(f, *args, **kwargs)

    return update_wrapper(decorator, f)  # type: ignore[return-value]


class AppGroup(click.Group):
    """This works similar to a regular click :class:`~click.Group` but it
    changes the behavior of the :meth:`command` decorator so that it
    automatically wraps the functions in :func:`with_appcontext`.

    Not to be confused with :class:`FlaskGroup`.
    """

    def command(  # type: ignore[override]
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Callable[[t.Callable[..., t.Any]], click.Command]:
        """This works exactly like the method of the same name on a regular
        :class:`click.Group` but it wraps callbacks in :func:`with_appcontext`
        unless it's disabled by passing ``with_appcontext=False``.
        """
        wrap_for_ctx = kwargs.pop("with_appcontext", True)

        def decorator(f: t.Callable[..., t.Any]) -> click.Command:
            if wrap_for_ctx:
                f = with_appcontext(f)
            return super(AppGroup, self).command(*args, **kwargs)(f)  # type: ignore[no-any-return]

        return decorator

    def group(  # type: ignore[override]
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Callable[[t.Callable[..., t.Any]], click.Group]:
        """This works exactly like the method of the same name on a regular
        :class:`click.Group` but it defaults the group class to
        :class:`AppGroup`.
        """
        kwargs.setdefault("cls", AppGroup)
        return super().group(*args, **kwargs)  # type: ignore[no-any-return]


def _set_app(ctx: click.Context, param: click.Option, value: str | None) -> str | None:
    if value is None:
        return None

    info = ctx.ensure_object(ScriptInfo)
    info.app_import_path = value
    return value


# This option is eager so the app will be available if --help is given.
# --help is also eager, so --app must be before it in the param list.
# no_args_is_help bypasses eager processing, so this option must be
# processed manually in that case to ensure FLASK_APP gets picked up.
_app_option = click.Option(
    ["-A", "--app"],
    metavar="IMPORT",
    help=(
        "The Flask application or factory function to load, in the form 'module:name'."
        " Module can be a dotted import or file path. Name is not required if it is"
        " 'app', 'application', 'create_app', or 'make_app', and can be 'name(args)' to"
        " pass arguments."
    ),
    is_eager=True,
    expose_value=False,
    callback=_set_app,
)


def _set_debug(ctx: click.Context, param: click.Option, value: bool) -> bool | None:
    # If the flag isn't provided, it will default to False. Don't use
    # that, let debug be set by env in that case.
    source = ctx.get_parameter_source(param.name)  # type: ignore[arg-type]

    if source is not None and source in (
        ParameterSource.DEFAULT,
        ParameterSource.DEFAULT_MAP,
    ):
        return None

    # Set with env var instead of ScriptInfo.load so that it can be
    # accessed early during a factory function.
    os.environ["FLASK_DEBUG"] = "1" if value else "0"
    return value


_debug_option = click.Option(
    ["--debug/--no-debug"],
    help="Set debug mode.",
    expose_value=False,
    callback=_set_debug,
)


def _env_file_callback(
    ctx: click.Context, param: click.Option, value: str | None
) -> str | None:
    try:
        import dotenv  # noqa: F401
    except ImportError:
        # Only show an error if a value was passed, otherwise we still want to
        # call load_dotenv and show a message without exiting.
        if value is not None:
            raise click.BadParameter(
                "python-dotenv must be installed to load an env file.",
                ctx=ctx,
                param=param,
            ) from None

    # Load if a value was passed, or we want to load default files, or both.
    if value is not None or ctx.obj.load_dotenv_defaults:
        load_dotenv(value, load_defaults=ctx.obj.load_dotenv_defaults)

    return value


# This option is eager so env vars are loaded as early as possible to be
# used by other options.
_env_file_option = click.Option(
    ["-e", "--env-file"],
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Load environment variables from this file, taking precedence over"
        " those set by '.env' and '.flaskenv'. Variables set directly in the"
        " environment take highest precedence. python-dotenv must be installed."
    ),
    is_eager=True,
    expose_value=False,
    callback=_env_file_callback,
)


class FlaskGroup(AppGroup):
    """Special subclass of the :class:`AppGroup` group that supports
    loading more commands from the configured Flask app.  Normally a
    developer does not have to interface with this class but there are
    some very advanced use cases for which it makes sense to create an
    instance of this. see :ref:`custom-scripts`.

    :param add_default_commands: if this is True then the default run and
        shell commands will be added.
    :param add_version_option: adds the ``--version`` option.
    :param create_app: an optional callback that is passed the script info and
        returns the loaded app.
    :param load_dotenv: Load the nearest :file:`.env` and :file:`.flaskenv`
        files to set environment variables. Will also change the working
        directory to the directory containing the first file found.
    :param set_debug_flag: Set the app's debug flag.

    .. versionchanged:: 3.1
        ``-e path`` takes precedence over default ``.env`` and ``.flaskenv`` files.

    .. versionchanged:: 2.2
        Added the ``-A/--app``, ``--debug/--no-debug``, ``-e/--env-file`` options.

    .. versionchanged:: 2.2
        An app context is pushed when running ``app.cli`` commands, so
        ``@with_appcontext`` is no longer required for those commands.

    .. versionchanged:: 1.0
        If installed, python-dotenv will be used to load environment variables
        from :file:`.env` and :file:`.flaskenv` files.
    """

    def __init__(
        self,
        add_default_commands: bool = True,
        create_app: t.Callable[..., Flask] | None = None,
        add_version_option: bool = True,
        load_dotenv: bool = True,
        set_debug_flag: bool = True,
        **extra: t.Any,
    ) -> None:
        params: list[click.Parameter] = list(extra.pop("params", None) or ())
        # Processing is done with option callbacks instead of a group
        # callback. This allows users to make a custom group callback
        # without losing the behavior. --env-file must come first so
        # that it is eagerly evaluated before --app.
        params.extend((_env_file_option, _app_option, _debug_option))

        if add_version_option:
            params.append(version_option)

        if "context_settings" not in extra:
            extra["context_settings"] = {}

        extra["context_settings"].setdefault("auto_envvar_prefix", "FLASK")

        super().__init__(params=params, **extra)

        self.create_app = create_app
        self.load_dotenv = load_dotenv
        self.set_debug_flag = set_debug_flag

        if add_default_commands:
            self.add_command(run_command)
            self.add_command(shell_command)
            self.add_command(routes_command)

        self._loaded_plugin_commands = False

    def _load_plugin_commands(self) -> None:
        if self._loaded_plugin_commands:
            return

        if sys.version_info >= (3, 10):
            from importlib import metadata
        else:
            # Use a backport on Python < 3.10. We technically have
            # importlib.metadata on 3.8+, but the API changed in 3.10,
            # so use the backport for consistency.
            import importlib_metadata as metadata  # pyright: ignore

        for ep in metadata.entry_points(group="flask.commands"):
            self.add_command(ep.load(), ep.name)

        self._loaded_plugin_commands = True

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        self._load_plugin_commands()
        # Look up built-in and plugin commands, which should be
        # available even if the app fails to load.
        rv = super().get_command(ctx, name)

        if rv is not None:
            return rv

        info = ctx.ensure_object(ScriptInfo)

        # Look up commands provided by the app, showing an error and
        # continuing if the app couldn't be loaded.
        try:
            app = info.load_app()
        except NoAppException as e:
            click.secho(f"Error: {e.format_message()}\n", err=True, fg="red")
            return None

        # Push an app context for the loaded app unless it is already
        # active somehow. This makes the context available to parameter
        # and command callbacks without needing @with_appcontext.
        if not current_app or current_app._get_current_object() is not app:  # type: ignore[attr-defined]
            ctx.with_resource(app.app_context())

        return app.cli.get_command(ctx, name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        self._load_plugin_commands()
        # Start with the built-in and plugin commands.
        rv = set(super().list_commands(ctx))
        info = ctx.ensure_object(ScriptInfo)

        # Add commands provided by the app, showing an error and
        # continuing if the app couldn't be loaded.
        try:
            rv.update(info.load_app().cli.list_commands(ctx))
        except NoAppException as e:
            # When an app couldn't be loaded, show the error message
            # without the traceback.
            click.secho(f"Error: {e.format_message()}\n", err=True, fg="red")
        except Exception:
            # When any other errors occurred during loading, show the
            # full traceback.
            click.secho(f"{traceback.format_exc()}\n", err=True, fg="red")

        return sorted(rv)

    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: click.Context | None = None,
        **extra: t.Any,
    ) -> click.Context:
        # Set a flag to tell app.run to become a no-op. If app.run was
        # not in a __name__ == __main__ guard, it would start the server
        # when importing, blocking whatever command is being called.
        os.environ["FLASK_RUN_FROM_CLI"] = "true"

        if "obj" not in extra and "obj" not in self.context_settings:
            extra["obj"] = ScriptInfo(
                create_app=self.create_app,
                set_debug_flag=self.set_debug_flag,
                load_dotenv_defaults=self.load_dotenv,
            )

        return super().make_context(info_name, args, parent=parent, **extra)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if (not args and self.no_args_is_help) or (
            len(args) == 1 and args[0] in self.get_help_option_names(ctx)
        ):
            # Attempt to load --env-file and --app early in case they
            # were given as env vars. Otherwise no_args_is_help will not
            # see commands from app.cli.
            _env_file_option.handle_parse_result(ctx, {}, [])
            _app_option.handle_parse_result(ctx, {}, [])

        return super().parse_args(ctx, args)


def _path_is_ancestor(path: str, other: str) -> bool:
    """Take ``other`` and remove the length of ``path`` from it. Then join it
    to ``path``. If it is the original value, ``path`` is an ancestor of
    ``other``."""
    return os.path.join(path, other[len(path) :].lstrip(os.sep)) == other


def load_dotenv(
    path: str | os.PathLike[str] | None = None, load_defaults: bool = True
) -> bool:
    """Load "dotenv" files to set environment variables. A given path takes
    precedence over ``.env``, which takes precedence over ``.flaskenv``. After
    loading and combining these files, values are only set if the key is not
    already set in ``os.environ``.

    This is a no-op if `python-dotenv`_ is not installed.

    .. _python-dotenv: https://github.com/theskumar/python-dotenv#readme

    :param path: Load the file at this location.
    :param load_defaults: Search for and load the default ``.flaskenv`` and
        ``.env`` files.
    :return: ``True`` if at least one env var was loaded.

    .. versionchanged:: 3.1
        Added the ``load_defaults`` parameter. A given path takes precedence
        over default files.

    .. versionchanged:: 2.0
        The current directory is not changed to the location of the
        loaded file.

    .. versionchanged:: 2.0
        When loading the env files, set the default encoding to UTF-8.

    .. versionchanged:: 1.1.0
        Returns ``False`` when python-dotenv is not installed, or when
        the given path isn't a file.

    .. versionadded:: 1.0
    """
    try:
        import dotenv
    except ImportError:
        if path or os.path.isfile(".env") or os.path.isfile(".flaskenv"):
            click.secho(
                " * Tip: There are .env files present. Install python-dotenv"
                " to use them.",
                fg="yellow",
                err=True,
            )

        return False

    data: dict[str, str | None] = {}

    if load_defaults:
        for default_name in (".flaskenv", ".env"):
            if not (default_path := dotenv.find_dotenv(default_name, usecwd=True)):
                continue

            data |= dotenv.dotenv_values(default_path, encoding="utf-8")

    if path is not None and os.path.isfile(path):
        data |= dotenv.dotenv_values(path, encoding="utf-8")

    for key, value in data.items():
        if key in os.environ or value is None:
            continue

        os.environ[key] = value

    return bool(data)  # True if at least one env var was loaded.


def show_server_banner(debug: bool, app_import_path: str | None) -> None:
    """Show extra startup messages the first time the server is run,
    ignoring the reloader.
    """
    if is_running_from_reloader():
        return

    if app_import_path is not None:
        click.echo(f" * Serving Flask app '{app_import_path}'")

    if debug is not None:
        click.echo(f" * Debug mode: {'on' if debug else 'off'}")


class CertParamType(click.ParamType):
    """Click option type for the ``--cert`` option. Allows either an
    existing file, the string ``'adhoc'``, or an import for a
    :class:`~ssl.SSLContext` object.
    """

    name = "path"

    def __init__(self) -> None:
        self.path_type = click.Path(exists=True, dir_okay=False, resolve_path=True)

    def convert(
        self, value: t.Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> t.Any:
        try:
            import ssl
        except ImportError:
            raise click.BadParameter(
                'Using "--cert" requires Python to be compiled with SSL support.',
                ctx,
                param,
            ) from None

        try:
            return self.path_type(value, param, ctx)
        except click.BadParameter:
            value = click.STRING(value, param, ctx).lower()

            if value == "adhoc":
                try:
                    import cryptography  # noqa: F401
                except ImportError:
                    raise click.BadParameter(
                        "Using ad-hoc certificates requires the cryptography library.",
                        ctx,
                        param,
                    ) from None

                return value

            obj = import_string(value, silent=True)

            if isinstance(obj, ssl.SSLContext):
                return obj

            raise


def _validate_key(ctx: click.Context, param: click.Parameter, value: t.Any) -> t.Any:
    """The ``--key`` option must be specified when ``--cert`` is a file.
    Modifies the ``cert`` param to be a ``(cert, key)`` pair if needed.
    """
    cert = ctx.params.get("cert")
    is_adhoc = cert == "adhoc"

    try:
        import ssl
    except ImportError:
        is_context = False
    else:
        is_context = isinstance(cert, ssl.SSLContext)

    if value is not None:
        if is_adhoc:
            raise click.BadParameter(
                'When "--cert" is "adhoc", "--key" is not used.', ctx, param
            )

        if is_context:
            raise click.BadParameter(
                'When "--cert" is an SSLContext object, "--key" is not used.',
                ctx,
                param,
            )

        if not cert:
            raise click.BadParameter('"--cert" must also be specified.', ctx, param)

        ctx.params["cert"] = cert, value

    else:
        if cert and not (is_adhoc or is_context):
            raise click.BadParameter('Required when using "--cert".', ctx, param)

    return value


class SeparatedPathType(click.Path):
    """Click option type that accepts a list of values separated by the
    OS's path separator (``:``, ``;`` on Windows). Each value is
    validated as a :class:`click.Path` type.
    """

    def convert(
        self, value: t.Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> t.Any:
        items = self.split_envvar_value(value)
        # can't call no-arg super() inside list comprehension until Python 3.12
        super_convert = super().convert
        return [super_convert(item, param, ctx) for item in items]


@click.command("run", short_help="Run a development server.")
@click.option("--host", "-h", default="127.0.0.1", help="The interface to bind to.")
@click.option("--port", "-p", default=5000, help="The port to bind to.")
@click.option(
    "--cert",
    type=CertParamType(),
    help="Specify a certificate file to use HTTPS.",
    is_eager=True,
)
@click.option(
    "--key",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    callback=_validate_key,
    expose_value=False,
    help="The key file to use when specifying a certificate.",
)
@click.option(
    "--reload/--no-reload",
    default=None,
    help="Enable or disable the reloader. By default the reloader "
    "is active if debug is enabled.",
)
@click.option(
    "--debugger/--no-debugger",
    default=None,
    help="Enable or disable the debugger. By default the debugger "
    "is active if debug is enabled.",
)
@click.option(
    "--with-threads/--without-threads",
    default=True,
    help="Enable or disable multithreading.",
)
@click.option(
    "--extra-files",
    default=None,
    type=SeparatedPathType(),
    help=(
        "Extra files that trigger a reload on change. Multiple paths"
        f" are separated by {os.path.pathsep!r}."
    ),
)
@click.option(
    "--exclude-patterns",
    default=None,
    type=SeparatedPathType(),
    help=(
        "Files matching these fnmatch patterns will not trigger a reload"
        " on change. Multiple patterns are separated by"
        f" {os.path.pathsep!r}."
    ),
)
@pass_script_info
def run_command(
    info: ScriptInfo,
    host: str,
    port: int,
    reload: bool,
    debugger: bool,
    with_threads: bool,
    cert: ssl.SSLContext | tuple[str, str | None] | t.Literal["adhoc"] | None,
    extra_files: list[str] | None,
    exclude_patterns: list[str] | None,
) -> None:
    """Run a local development server.

    This server is for development purposes only. It does not provide
    the stability, security, or performance of production WSGI servers.

    The reloader and debugger are enabled by default with the '--debug'
    option.
    """
    try:
        app: WSGIApplication = info.load_app()  # pyright: ignore
    except Exception as e:
        if is_running_from_reloader():
            # When reloading, print out the error immediately, but raise
            # it later so the debugger or server can handle it.
            traceback.print_exc()
            err = e

            def app(
                environ: WSGIEnvironment, start_response: StartResponse
            ) -> cabc.Iterable[bytes]:
                raise err from None

        else:
            # When not reloading, raise the error immediately so the
            # command fails.
            raise e from None

    debug = get_debug_flag()

    if reload is None:
        reload = debug

    if debugger is None:
        debugger = debug

    show_server_banner(debug, info.app_import_path)

    run_simple(
        host,
        port,
        app,
        use_reloader=reload,
        use_debugger=debugger,
        threaded=with_threads,
        ssl_context=cert,
        extra_files=extra_files,
        exclude_patterns=exclude_patterns,
    )


run_command.params.insert(0, _debug_option)


@click.command("shell", short_help="Run a shell in the app context.")
@with_appcontext
def shell_command() -> None:
    """Run an interactive Python shell in the context of a given
    Flask application.  The application will populate the default
    namespace of this shell according to its configuration.

    This is useful for executing small snippets of management code
    without having to manually configure the application.
    """
    import code

    banner = (
        f"Python {sys.version} on {sys.platform}\n"
        f"App: {current_app.import_name}\n"
        f"Instance: {current_app.instance_path}"
    )
    ctx: dict[str, t.Any] = {}

    # Support the regular Python interpreter startup script if someone
    # is using it.
    startup = os.environ.get("PYTHONSTARTUP")
    if startup and os.path.isfile(startup):
        with open(startup) as f:
            eval(compile(f.read(), startup, "exec"), ctx)

    ctx.update(current_app.make_shell_context())

    # Site, customize, or startup script can set a hook to call when
    # entering interactive mode. The default one sets up readline with
    # tab and history completion.
    interactive_hook = getattr(sys, "__interactivehook__", None)

    if interactive_hook is not None:
        try:
            import readline
            from rlcompleter import Completer
        except ImportError:
            pass
        else:
            # rlcompleter uses __main__.__dict__ by default, which is
            # flask.__main__. Use the shell context instead.
            readline.set_completer(Completer(ctx).complete)

        interactive_hook()

    code.interact(banner=banner, local=ctx)


@click.command("routes", short_help="Show the routes for the app.")
@click.option(
    "--sort",
    "-s",
    type=click.Choice(("endpoint", "methods", "domain", "rule", "match")),
    default="endpoint",
    help=(
        "Method to sort routes by. 'match' is the order that Flask will match routes"
        " when dispatching a request."
    ),
)
@click.option("--all-methods", is_flag=True, help="Show HEAD and OPTIONS methods.")
@with_appcontext
def routes_command(sort: str, all_methods: bool) -> None:
    """Show all registered routes with endpoints and methods."""
    rules = list(current_app.url_map.iter_rules())

    if not rules:
        click.echo("No routes were registered.")
        return

    ignored_methods = set() if all_methods else {"HEAD", "OPTIONS"}
    host_matching = current_app.url_map.host_matching
    has_domain = any(rule.host if host_matching else rule.subdomain for rule in rules)
    rows = []

    for rule in rules:
        row = [
            rule.endpoint,
            ", ".join(sorted((rule.methods or set()) - ignored_methods)),
        ]

        if has_domain:
            row.append((rule.host if host_matching else rule.subdomain) or "")

        row.append(rule.rule)
        rows.append(row)

    headers = ["Endpoint", "Methods"]
    sorts = ["endpoint", "methods"]

    if has_domain:
        headers.append("Host" if host_matching else "Subdomain")
        sorts.append("domain")

    headers.append("Rule")
    sorts.append("rule")

    try:
        rows.sort(key=itemgetter(sorts.index(sort)))
    except ValueError:
        pass

    rows.insert(0, headers)
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    rows.insert(1, ["-" * w for w in widths])
    template = "  ".join(f"{{{i}:<{w}}}" for i, w in enumerate(widths))

    for row in rows:
        click.echo(template.format(*row))


cli = FlaskGroup(
    name="flask",
    help="""\
A general utility script for Flask applications.

An application to load must be given with the '--app' option,
'FLASK_APP' environment variable, or with a 'wsgi.py' or 'app.py' file
in the current directory.
""",
)


def main() -> None:
    cli.main()


if __name__ == "__main__":
    main()