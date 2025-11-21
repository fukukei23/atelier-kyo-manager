
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\elliptic.py ===
r"""
Elliptic functions historically comprise the elliptic integrals
and their inverses, and originate from the problem of computing the
arc length of an ellipse. From a more modern point of view,
an elliptic function is defined as a doubly periodic function, i.e.
a function which satisfies

.. math ::

    f(z + 2 \omega_1) = f(z + 2 \omega_2) = f(z)

for some half-periods `\omega_1, \omega_2` with
`\mathrm{Im}[\omega_1 / \omega_2] > 0`. The canonical elliptic
functions are the Jacobi elliptic functions. More broadly, this section
includes  quasi-doubly periodic functions (such as the Jacobi theta
functions) and other functions useful in the study of elliptic functions.

Many different conventions for the arguments of
elliptic functions are in use. It is even standard to use
different parameterizations for different functions in the same
text or software (and mpmath is no exception).
The usual parameters are the elliptic nome `q`, which usually
must satisfy `|q| < 1`; the elliptic parameter `m` (an arbitrary
complex number); the elliptic modulus `k` (an arbitrary complex
number); and the half-period ratio `\tau`, which usually must
satisfy `\mathrm{Im}[\tau] > 0`.
These quantities can be expressed in terms of each other
using the following relations:

.. math ::

    m = k^2

.. math ::

    \tau = i \frac{K(1-m)}{K(m)}

.. math ::

    q = e^{i \pi \tau}

.. math ::

    k = \frac{\vartheta_2^2(q)}{\vartheta_3^2(q)}

In addition, an alternative definition is used for the nome in
number theory, which we here denote by q-bar:

.. math ::

    \bar{q} = q^2 = e^{2 i \pi \tau}

For convenience, mpmath provides functions to convert
between the various parameters (:func:`~mpmath.qfrom`, :func:`~mpmath.mfrom`,
:func:`~mpmath.kfrom`, :func:`~mpmath.taufrom`, :func:`~mpmath.qbarfrom`).

**References**

1. [AbramowitzStegun]_

2. [WhittakerWatson]_

"""

from .functions import defun, defun_wrapped

@defun_wrapped
def eta(ctx, tau):
    r"""
    Returns the Dedekind eta function of tau in the upper half-plane.

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> eta(1j); gamma(0.25) / (2*pi**0.75)
        (0.7682254223260566590025942 + 0.0j)
        0.7682254223260566590025942
        >>> tau = sqrt(2) + sqrt(5)*1j
        >>> eta(-1/tau); sqrt(-1j*tau) * eta(tau)
        (0.9022859908439376463573294 + 0.07985093673948098408048575j)
        (0.9022859908439376463573295 + 0.07985093673948098408048575j)
        >>> eta(tau+1); exp(pi*1j/12) * eta(tau)
        (0.4493066139717553786223114 + 0.3290014793877986663915939j)
        (0.4493066139717553786223114 + 0.3290014793877986663915939j)
        >>> f = lambda z: diff(eta, z) / eta(z)
        >>> chop(36*diff(f,tau)**2 - 24*diff(f,tau,2)*f(tau) + diff(f,tau,3))
        0.0

    """
    if ctx.im(tau) <= 0.0:
        raise ValueError("eta is only defined in the upper half-plane")
    q = ctx.expjpi(tau/12)
    return q * ctx.qp(q**24)

def nome(ctx, m):
    m = ctx.convert(m)
    if not m:
        return m
    if m == ctx.one:
        return m
    if ctx.isnan(m):
        return m
    if ctx.isinf(m):
        if m == ctx.ninf:
            return type(m)(-1)
        else:
            return ctx.mpc(-1)
    a = ctx.ellipk(ctx.one-m)
    b = ctx.ellipk(m)
    v = ctx.exp(-ctx.pi*a/b)
    if not ctx._im(m) and ctx._re(m) < 1:
        if ctx._is_real_type(m):
            return v.real
        else:
            return v.real + 0j
    elif m == 2:
        v = ctx.mpc(0, v.imag)
    return v

@defun_wrapped
def qfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    r"""
    Returns the elliptic nome `q`, given any of `q, m, k, \tau, \bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qfrom(q=0.25)
        0.25
        >>> qfrom(m=mfrom(q=0.25))
        0.25
        >>> qfrom(k=kfrom(q=0.25))
        0.25
        >>> qfrom(tau=taufrom(q=0.25))
        (0.25 + 0.0j)
        >>> qfrom(qbar=qbarfrom(q=0.25))
        0.25

    """
    if q is not None:
        return ctx.convert(q)
    if m is not None:
        return nome(ctx, m)
    if k is not None:
        return nome(ctx, ctx.convert(k)**2)
    if tau is not None:
        return ctx.expjpi(tau)
    if qbar is not None:
        return ctx.sqrt(qbar)

@defun_wrapped
def qbarfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    r"""
    Returns the number-theoretic nome `\bar q`, given any of
    `q, m, k, \tau, \bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qbarfrom(qbar=0.25)
        0.25
        >>> qbarfrom(q=qfrom(qbar=0.25))
        0.25
        >>> qbarfrom(m=extraprec(20)(mfrom)(qbar=0.25))  # ill-conditioned
        0.25
        >>> qbarfrom(k=extraprec(20)(kfrom)(qbar=0.25))  # ill-conditioned
        0.25
        >>> qbarfrom(tau=taufrom(qbar=0.25))
        (0.25 + 0.0j)

    """
    if qbar is not None:
        return ctx.convert(qbar)
    if q is not None:
        return ctx.convert(q) ** 2
    if m is not None:
        return nome(ctx, m) ** 2
    if k is not None:
        return nome(ctx, ctx.convert(k)**2) ** 2
    if tau is not None:
        return ctx.expjpi(2*tau)

@defun_wrapped
def taufrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    r"""
    Returns the elliptic half-period ratio `\tau`, given any of
    `q, m, k, \tau, \bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> taufrom(tau=0.5j)
        (0.0 + 0.5j)
        >>> taufrom(q=qfrom(tau=0.5j))
        (0.0 + 0.5j)
        >>> taufrom(m=mfrom(tau=0.5j))
        (0.0 + 0.5j)
        >>> taufrom(k=kfrom(tau=0.5j))
        (0.0 + 0.5j)
        >>> taufrom(qbar=qbarfrom(tau=0.5j))
        (0.0 + 0.5j)

    """
    if tau is not None:
        return ctx.convert(tau)
    if m is not None:
        m = ctx.convert(m)
        return ctx.j*ctx.ellipk(1-m)/ctx.ellipk(m)
    if k is not None:
        k = ctx.convert(k)
        return ctx.j*ctx.ellipk(1-k**2)/ctx.ellipk(k**2)
    if q is not None:
        return ctx.log(q) / (ctx.pi*ctx.j)
    if qbar is not None:
        qbar = ctx.convert(qbar)
        return ctx.log(qbar) / (2*ctx.pi*ctx.j)

@defun_wrapped
def kfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    r"""
    Returns the elliptic modulus `k`, given any of
    `q, m, k, \tau, \bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> kfrom(k=0.25)
        0.25
        >>> kfrom(m=mfrom(k=0.25))
        0.25
        >>> kfrom(q=qfrom(k=0.25))
        0.25
        >>> kfrom(tau=taufrom(k=0.25))
        (0.25 + 0.0j)
        >>> kfrom(qbar=qbarfrom(k=0.25))
        0.25

    As `q \to 1` and `q \to -1`, `k` rapidly approaches
    `1` and `i \infty` respectively::

        >>> kfrom(q=0.75)
        0.9999999999999899166471767
        >>> kfrom(q=-0.75)
        (0.0 + 7041781.096692038332790615j)
        >>> kfrom(q=1)
        1
        >>> kfrom(q=-1)
        (0.0 + +infj)
    """
    if k is not None:
        return ctx.convert(k)
    if m is not None:
        return ctx.sqrt(m)
    if tau is not None:
        q = ctx.expjpi(tau)
    if qbar is not None:
        q = ctx.sqrt(qbar)
    if q == 1:
        return q
    if q == -1:
        return ctx.mpc(0,'inf')
    return (ctx.jtheta(2,0,q)/ctx.jtheta(3,0,q))**2

@defun_wrapped
def mfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    r"""
    Returns the elliptic parameter `m`, given any of
    `q, m, k, \tau, \bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> mfrom(m=0.25)
        0.25
        >>> mfrom(q=qfrom(m=0.25))
        0.25
        >>> mfrom(k=kfrom(m=0.25))
        0.25
        >>> mfrom(tau=taufrom(m=0.25))
        (0.25 + 0.0j)
        >>> mfrom(qbar=qbarfrom(m=0.25))
        0.25

    As `q \to 1` and `q \to -1`, `m` rapidly approaches
    `1` and `-\infty` respectively::

        >>> mfrom(q=0.75)
        0.9999999999999798332943533
        >>> mfrom(q=-0.75)
        -49586681013729.32611558353
        >>> mfrom(q=1)
        1.0
        >>> mfrom(q=-1)
        -inf

    The inverse nome as a function of `q` has an integer
    Taylor series expansion::

        >>> taylor(lambda q: mfrom(q), 0, 7)
        [0.0, 16.0, -128.0, 704.0, -3072.0, 11488.0, -38400.0, 117632.0]

    """
    if m is not None:
        return m
    if k is not None:
        return k**2
    if tau is not None:
        q = ctx.expjpi(tau)
    if qbar is not None:
        q = ctx.sqrt(qbar)
    if q == 1:
        return ctx.convert(q)
    if q == -1:
        return q*ctx.inf
    v = (ctx.jtheta(2,0,q)/ctx.jtheta(3,0,q))**4
    if ctx._is_real_type(q) and q < 0:
        v = v.real
    return v

jacobi_spec = {
  'sn' : ([3],[2],[1],[4], 'sin', 'tanh'),
  'cn' : ([4],[2],[2],[4], 'cos', 'sech'),
  'dn' : ([4],[3],[3],[4], '1', 'sech'),
  'ns' : ([2],[3],[4],[1], 'csc', 'coth'),
  'nc' : ([2],[4],[4],[2], 'sec', 'cosh'),
  'nd' : ([3],[4],[4],[3], '1', 'cosh'),
  'sc' : ([3],[4],[1],[2], 'tan', 'sinh'),
  'sd' : ([3,3],[2,4],[1],[3], 'sin', 'sinh'),
  'cd' : ([3],[2],[2],[3], 'cos', '1'),
  'cs' : ([4],[3],[2],[1], 'cot', 'csch'),
  'dc' : ([2],[3],[3],[2], 'sec', '1'),
  'ds' : ([2,4],[3,3],[3],[1], 'csc', 'csch'),
  'cc' : None,
  'ss' : None,
  'nn' : None,
  'dd' : None
}

@defun
def ellipfun(ctx, kind, u=None, m=None, q=None, k=None, tau=None):
    try:
        S = jacobi_spec[kind]
    except KeyError:
        raise ValueError("First argument must be a two-character string "
            "containing 's', 'c', 'd' or 'n', e.g.: 'sn'")
    if u is None:
        def f(*args, **kwargs):
            return ctx.ellipfun(kind, *args, **kwargs)
        f.__name__ = kind
        return f
    prec = ctx.prec
    try:
        ctx.prec += 10
        u = ctx.convert(u)
        q = ctx.qfrom(m=m, q=q, k=k, tau=tau)
        if S is None:
            v = ctx.one + 0*q*u
        elif q == ctx.zero:
            if S[4] == '1': v = ctx.one
            else:           v = getattr(ctx, S[4])(u)
            v += 0*q*u
        elif q == ctx.one:
            if S[5] == '1': v = ctx.one
            else:           v = getattr(ctx, S[5])(u)
            v += 0*q*u
        else:
            t = u / ctx.jtheta(3, 0, q)**2
            v = ctx.one
            for a in S[0]: v *= ctx.jtheta(a, 0, q)
            for b in S[1]: v /= ctx.jtheta(b, 0, q)
            for c in S[2]: v *= ctx.jtheta(c, t, q)
            for d in S[3]: v /= ctx.jtheta(d, t, q)
    finally:
        ctx.prec = prec
    return +v

@defun_wrapped
def kleinj(ctx, tau=None, **kwargs):
    r"""
    Evaluates the Klein j-invariant, which is a modular function defined for
    `\tau` in the upper half-plane as

    .. math ::

        J(\tau) = \frac{g_2^3(\tau)}{g_2^3(\tau) - 27 g_3^2(\tau)}

    where `g_2` and `g_3` are the modular invariants of the Weierstrass
    elliptic function,

    .. math ::

        g_2(\tau) = 60 \sum_{(m,n) \in \mathbb{Z}^2 \setminus (0,0)} (m \tau+n)^{-4}

        g_3(\tau) = 140 \sum_{(m,n) \in \mathbb{Z}^2 \setminus (0,0)} (m \tau+n)^{-6}.

    An alternative, common notation is that of the j-function
    `j(\tau) = 1728 J(\tau)`.

    **Plots**

    .. literalinclude :: /plots/kleinj.py
    .. image :: /plots/kleinj.png
    .. literalinclude :: /plots/kleinj2.py
    .. image :: /plots/kleinj2.png

    **Examples**

    Verifying the functional equation `J(\tau) = J(\tau+1) = J(-\tau^{-1})`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> tau = 0.625+0.75*j
        >>> tau = 0.625+0.75*j
        >>> kleinj(tau)
        (-0.1507492166511182267125242 + 0.07595948379084571927228948j)
        >>> kleinj(tau+1)
        (-0.1507492166511182267125242 + 0.07595948379084571927228948j)
        >>> kleinj(-1/tau)
        (-0.1507492166511182267125242 + 0.07595948379084571927228946j)

    The j-function has a famous Laurent series expansion in terms of the nome
    `\bar{q}`, `j(\tau) = \bar{q}^{-1} + 744 + 196884\bar{q} + \ldots`::

        >>> mp.dps = 15
        >>> taylor(lambda q: 1728*q*kleinj(qbar=q), 0, 5, singular=True)
        [1.0, 744.0, 196884.0, 21493760.0, 864299970.0, 20245856256.0]

    The j-function admits exact evaluation at special algebraic points
    related to the Heegner numbers 1, 2, 3, 7, 11, 19, 43, 67, 163::

        >>> @extraprec(10)
        ... def h(n):
        ...     v = (1+sqrt(n)*j)
        ...     if n > 2:
        ...         v *= 0.5
        ...     return v
        ...
        >>> mp.dps = 25
        >>> for n in [1,2,3,7,11,19,43,67,163]:
        ...     n, chop(1728*kleinj(h(n)))
        ...
        (1, 1728.0)
        (2, 8000.0)
        (3, 0.0)
        (7, -3375.0)
        (11, -32768.0)
        (19, -884736.0)
        (43, -884736000.0)
        (67, -147197952000.0)
        (163, -262537412640768000.0)

    Also at other special points, the j-function assumes explicit
    algebraic values, e.g.::

        >>> chop(1728*kleinj(j*sqrt(5)))
        1264538.909475140509320227
        >>> identify(cbrt(_))      # note: not simplified
        '((100+sqrt(13520))/2)'
        >>> (50+26*sqrt(5))**3
        1264538.909475140509320227

    """
    q = ctx.qfrom(tau=tau, **kwargs)
    t2 = ctx.jtheta(2,0,q)
    t3 = ctx.jtheta(3,0,q)
    t4 = ctx.jtheta(4,0,q)
    P = (t2**8 + t3**8 + t4**8)**3
    Q = 54*(t2*t3*t4)**8
    return P/Q


def RF_calc(ctx, x, y, z, r):
    if y == z: return RC_calc(ctx, x, y, r)
    if x == z: return RC_calc(ctx, y, x, r)
    if x == y: return RC_calc(ctx, z, x, r)
    if not (ctx.isnormal(x) and ctx.isnormal(y) and ctx.isnormal(z)):
        if ctx.isnan(x) or ctx.isnan(y) or ctx.isnan(z):
            return x*y*z
        if ctx.isinf(x) or ctx.isinf(y) or ctx.isinf(z):
            return ctx.zero
    xm,ym,zm = x,y,z
    A0 = Am = (x+y+z)/3
    Q = ctx.root(3*r, -6) * max(abs(A0-x),abs(A0-y),abs(A0-z))
    g = ctx.mpf(0.25)
    pow4 = ctx.one
    while 1:
        xs = ctx.sqrt(xm)
        ys = ctx.sqrt(ym)
        zs = ctx.sqrt(zm)
        lm = xs*ys + xs*zs + ys*zs
        Am1 = (Am+lm)*g
        xm, ym, zm = (xm+lm)*g, (ym+lm)*g, (zm+lm)*g
        if pow4 * Q < abs(Am):
            break
        Am = Am1
        pow4 *= g
    t = pow4/Am
    X = (A0-x)*t
    Y = (A0-y)*t
    Z = -X-Y
    E2 = X*Y-Z**2
    E3 = X*Y*Z
    return ctx.power(Am,-0.5) * (9240-924*E2+385*E2**2+660*E3-630*E2*E3)/9240

def RC_calc(ctx, x, y, r, pv=True):
    if not (ctx.isnormal(x) and ctx.isnormal(y)):
        if ctx.isinf(x) or ctx.isinf(y):
            return 1/(x*y)
        if y == 0:
            return ctx.inf
        if x == 0:
            return ctx.pi / ctx.sqrt(y) / 2
        raise ValueError
    # Cauchy principal value
    if pv and ctx._im(y) == 0 and ctx._re(y) < 0:
        return ctx.sqrt(x/(x-y)) * RC_calc(ctx, x-y, -y, r)
    if x == y:
        return 1/ctx.sqrt(x)
    extraprec = 2*max(0,-ctx.mag(x-y)+ctx.mag(x))
    ctx.prec += extraprec
    if ctx._is_real_type(x) and ctx._is_real_type(y):
        x = ctx._re(x)
        y = ctx._re(y)
        a = ctx.sqrt(x/y)
        if x < y:
            b = ctx.sqrt(y-x)
            v = ctx.acos(a)/b
        else:
            b = ctx.sqrt(x-y)
            v = ctx.acosh(a)/b
    else:
        sx = ctx.sqrt(x)
        sy = ctx.sqrt(y)
        v = ctx.acos(sx/sy)/(ctx.sqrt((1-x/y))*sy)
    ctx.prec -= extraprec
    return v

def RJ_calc(ctx, x, y, z, p, r, integration):
    """
    With integration == 0, computes RJ only using Carlson's algorithm
    (may be wrong for some values).
    With integration == 1, uses an initial integration to make sure
    Carlson's algorithm is correct.
    With integration == 2, uses only integration.
    """
    if not (ctx.isnormal(x) and ctx.isnormal(y) and \
        ctx.isnormal(z) and ctx.isnormal(p)):
        if ctx.isnan(x) or ctx.isnan(y) or ctx.isnan(z) or ctx.isnan(p):
            return x*y*z
        if ctx.isinf(x) or ctx.isinf(y) or ctx.isinf(z) or ctx.isinf(p):
            return ctx.zero
    if not p:
        return ctx.inf
    if (not x) + (not y) + (not z) > 1:
        return ctx.inf
    # Check conditions and fall back on integration for argument
    # reduction if needed. The following conditions might be needlessly
    # restrictive.
    initial_integral = ctx.zero
    if integration >= 1:
        ok = (x.real >= 0 and y.real >= 0 and z.real >= 0 and p.real > 0)
        if not ok:
            if x == p or y == p or z == p:
                ok = True
        if not ok:
            if p.imag != 0 or p.real >= 0:
                if (x.imag == 0 and x.real >= 0 and ctx.conj(y) == z):
                    ok = True
                if (y.imag == 0 and y.real >= 0 and ctx.conj(x) == z):
                    ok = True
                if (z.imag == 0 and z.real >= 0 and ctx.conj(x) == y):
                    ok = True
        if not ok or (integration == 2):
            N = ctx.ceil(-min(x.real, y.real, z.real, p.real)) + 1
            # Integrate around any singularities
            if all((t.imag >= 0 or t.real > 0) for t in [x, y, z, p]):
                margin = ctx.j
            elif all((t.imag < 0 or t.real > 0) for t in [x, y, z, p]):
                margin = -ctx.j
            else:
                margin = 1
                # Go through the upper half-plane, but low enough that any
                # parameter starting in the lower plane doesn't cross the
                # branch cut
                for t in [x, y, z, p]:
                    if t.imag >= 0 or t.real > 0:
                        continue
                    margin = min(margin, abs(t.imag) * 0.5)
                margin *= ctx.j
            N += margin
            F = lambda t: 1/(ctx.sqrt(t+x)*ctx.sqrt(t+y)*ctx.sqrt(t+z)*(t+p))
            if integration == 2:
                return 1.5 * ctx.quadsubdiv(F, [0, N, ctx.inf])
            initial_integral = 1.5 * ctx.quadsubdiv(F, [0, N])
            x += N; y += N; z += N; p += N
    xm,ym,zm,pm = x,y,z,p
    A0 = Am = (x + y + z + 2*p)/5
    delta = (p-x)*(p-y)*(p-z)
    Q = ctx.root(0.25*r, -6) * max(abs(A0-x),abs(A0-y),abs(A0-z),abs(A0-p))
    g = ctx.mpf(0.25)
    pow4 = ctx.one
    S = 0
    while 1:
        sx = ctx.sqrt(xm)
        sy = ctx.sqrt(ym)
        sz = ctx.sqrt(zm)
        sp = ctx.sqrt(pm)
        lm = sx*sy + sx*sz + sy*sz
        Am1 = (Am+lm)*g
        xm = (xm+lm)*g; ym = (ym+lm)*g; zm = (zm+lm)*g; pm = (pm+lm)*g
        dm = (sp+sx) * (sp+sy) * (sp+sz)
        em = delta * pow4**3 / dm**2
        if pow4 * Q < abs(Am):
            break
        T = RC_calc(ctx, ctx.one, ctx.one+em, r) * pow4 / dm
        S += T
        pow4 *= g
        Am = Am1
    t = pow4 / Am
    X = (A0-x)*t
    Y = (A0-y)*t
    Z = (A0-z)*t
    P = (-X-Y-Z)/2
    E2 = X*Y + X*Z + Y*Z - 3*P**2
    E3 = X*Y*Z + 2*E2*P + 4*P**3
    E4 = (2*X*Y*Z + E2*P + 3*P**3)*P
    E5 = X*Y*Z*P**2
    P = 24024 - 5148*E2 + 2457*E2**2 + 4004*E3 - 4158*E2*E3 - 3276*E4 + 2772*E5
    Q = 24024
    v1 = pow4 * ctx.power(Am, -1.5) * P/Q
    v2 = 6*S
    return initial_integral + v1 + v2

@defun
def elliprf(ctx, x, y, z):
    r"""
    Evaluates the Carlson symmetric elliptic integral of the first kind

    .. math ::

        R_F(x,y,z) = \frac{1}{2}
            \int_0^{\infty} \frac{dt}{\sqrt{(t+x)(t+y)(t+z)}}

    which is defined for `x,y,z \notin (-\infty,0)`, and with
    at most one of `x,y,z` being zero.

    For real `x,y,z \ge 0`, the principal square root is taken in the integrand.
    For complex `x,y,z`, the principal square root is taken as `t \to \infty`
    and as `t \to 0` non-principal branches are chosen as necessary so as to
    make the integrand continuous.

    **Examples**

    Some basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprf(0,1,1); pi/2
        1.570796326794896619231322
        1.570796326794896619231322
        >>> elliprf(0,1,inf)
        0.0
        >>> elliprf(1,1,1)
        1.0
        >>> elliprf(2,2,2)**2
        0.5
        >>> elliprf(1,0,0); elliprf(0,0,1); elliprf(0,1,0); elliprf(0,0,0)
        +inf
        +inf
        +inf
        +inf

    Representing complete elliptic integrals in terms of `R_F`::

        >>> m = mpf(0.75)
        >>> ellipk(m); elliprf(0,1-m,1)
        2.156515647499643235438675
        2.156515647499643235438675
        >>> ellipe(m); elliprf(0,1-m,1)-m*elliprd(0,1-m,1)/3
        1.211056027568459524803563
        1.211056027568459524803563

    Some symmetries and argument transformations::

        >>> x,y,z = 2,3,4
        >>> elliprf(x,y,z); elliprf(y,x,z); elliprf(z,y,x)
        0.5840828416771517066928492
        0.5840828416771517066928492
        0.5840828416771517066928492
        >>> k = mpf(100000)
        >>> elliprf(k*x,k*y,k*z); k**(-0.5) * elliprf(x,y,z)
        0.001847032121923321253219284
        0.001847032121923321253219284
        >>> l = sqrt(x*y) + sqrt(y*z) + sqrt(z*x)
        >>> elliprf(x,y,z); 2*elliprf(x+l,y+l,z+l)
        0.5840828416771517066928492
        0.5840828416771517066928492
        >>> elliprf((x+l)/4,(y+l)/4,(z+l)/4)
        0.5840828416771517066928492

    Comparing with numerical integration::

        >>> x,y,z = 2,3,4
        >>> elliprf(x,y,z)
        0.5840828416771517066928492
        >>> f = lambda t: 0.5*((t+x)*(t+y)*(t+z))**(-0.5)
        >>> q = extradps(25)(quad)
        >>> q(f, [0,inf])
        0.5840828416771517066928492

    With the following arguments, the square root in the integrand becomes
    discontinuous at `t = 1/2` if the principal branch is used. To obtain
    the right value, `-\sqrt{r}` must be taken instead of `\sqrt{r}`
    on `t \in (0, 1/2)`::

        >>> x,y,z = j-1,j,0
        >>> elliprf(x,y,z)
        (0.7961258658423391329305694 - 1.213856669836495986430094j)
        >>> -q(f, [0,0.5]) + q(f, [0.5,inf])
        (0.7961258658423391329305694 - 1.213856669836495986430094j)

    The so-called *first lemniscate constant*, a transcendental number::

        >>> elliprf(0,1,2)
        1.31102877714605990523242
        >>> extradps(25)(quad)(lambda t: 1/sqrt(1-t**4), [0,1])
        1.31102877714605990523242
        >>> gamma('1/4')**2/(4*sqrt(2*pi))
        1.31102877714605990523242

    **References**

    1. [Carlson]_
    2. [DLMF]_ Chapter 19. Elliptic Integrals

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    z = ctx.convert(z)
    prec = ctx.prec
    try:
        ctx.prec += 20
        tol = ctx.eps * 2**10
        v = RF_calc(ctx, x, y, z, tol)
    finally:
        ctx.prec = prec
    return +v

@defun
def elliprc(ctx, x, y, pv=True):
    r"""
    Evaluates the degenerate Carlson symmetric elliptic integral
    of the first kind

    .. math ::

        R_C(x,y) = R_F(x,y,y) =
            \frac{1}{2} \int_0^{\infty} \frac{dt}{(t+y) \sqrt{(t+x)}}.

    If `y \in (-\infty,0)`, either a value defined by continuity,
    or with *pv=True* the Cauchy principal value, can be computed.

    If `x \ge 0, y > 0`, the value can be expressed in terms of
    elementary functions as

    .. math ::

        R_C(x,y) =
        \begin{cases}
          \dfrac{1}{\sqrt{y-x}}
            \cos^{-1}\left(\sqrt{\dfrac{x}{y}}\right),   & x < y \\
          \dfrac{1}{\sqrt{y}},                          & x = y \\
          \dfrac{1}{\sqrt{x-y}}
            \cosh^{-1}\left(\sqrt{\dfrac{x}{y}}\right),  & x > y \\
        \end{cases}.

    **Examples**

    Some special values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprc(1,2)*4; elliprc(0,1)*2; +pi
        3.141592653589793238462643
        3.141592653589793238462643
        3.141592653589793238462643
        >>> elliprc(1,0)
        +inf
        >>> elliprc(5,5)**2
        0.2
        >>> elliprc(1,inf); elliprc(inf,1); elliprc(inf,inf)
        0.0
        0.0
        0.0

    Comparing with the elementary closed-form solution::

        >>> elliprc('1/3', '1/5'); sqrt(7.5)*acosh(sqrt('5/3'))
        2.041630778983498390751238
        2.041630778983498390751238
        >>> elliprc('1/5', '1/3'); sqrt(7.5)*acos(sqrt('3/5'))
        1.875180765206547065111085
        1.875180765206547065111085

    Comparing with numerical integration::

        >>> q = extradps(25)(quad)
        >>> elliprc(2, -3, pv=True)
        0.3333969101113672670749334
        >>> elliprc(2, -3, pv=False)
        (0.3333969101113672670749334 + 0.7024814731040726393156375j)
        >>> 0.5*q(lambda t: 1/(sqrt(t+2)*(t-3)), [0,3-j,6,inf])
        (0.3333969101113672670749334 + 0.7024814731040726393156375j)

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    prec = ctx.prec
    try:
        ctx.prec += 20
        tol = ctx.eps * 2**10
        v = RC_calc(ctx, x, y, tol, pv)
    finally:
        ctx.prec = prec
    return +v

@defun
def elliprj(ctx, x, y, z, p, integration=1):
    r"""
    Evaluates the Carlson symmetric elliptic integral of the third kind

    .. math ::

        R_J(x,y,z,p) = \frac{3}{2}
            \int_0^{\infty} \frac{dt}{(t+p)\sqrt{(t+x)(t+y)(t+z)}}.

    Like :func:`~mpmath.elliprf`, the branch of the square root in the integrand
    is defined so as to be continuous along the path of integration for
    complex values of the arguments.

    **Examples**

    Some values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprj(1,1,1,1)
        1.0
        >>> elliprj(2,2,2,2); 1/(2*sqrt(2))
        0.3535533905932737622004222
        0.3535533905932737622004222
        >>> elliprj(0,1,2,2)
        1.067937989667395702268688
        >>> 3*(2*gamma('5/4')**2-pi**2/gamma('1/4')**2)/(sqrt(2*pi))
        1.067937989667395702268688
        >>> elliprj(0,1,1,2); 3*pi*(2-sqrt(2))/4
        1.380226776765915172432054
        1.380226776765915172432054
        >>> elliprj(1,3,2,0); elliprj(0,1,1,0); elliprj(0,0,0,0)
        +inf
        +inf
        +inf
        >>> elliprj(1,inf,1,0); elliprj(1,1,1,inf)
        0.0
        0.0
        >>> chop(elliprj(1+j, 1-j, 1, 1))
        0.8505007163686739432927844

    Scale transformation::

        >>> x,y,z,p = 2,3,4,5
        >>> k = mpf(100000)
        >>> elliprj(k*x,k*y,k*z,k*p); k**(-1.5)*elliprj(x,y,z,p)
        4.521291677592745527851168e-9
        4.521291677592745527851168e-9

    Comparing with numerical integration::

        >>> elliprj(1,2,3,4)
        0.2398480997495677621758617
        >>> f = lambda t: 1/((t+4)*sqrt((t+1)*(t+2)*(t+3)))
        >>> 1.5*quad(f, [0,inf])
        0.2398480997495677621758617
        >>> elliprj(1,2+1j,3,4-2j)
        (0.216888906014633498739952 + 0.04081912627366673332369512j)
        >>> f = lambda t: 1/((t+4-2j)*sqrt((t+1)*(t+2+1j)*(t+3)))
        >>> 1.5*quad(f, [0,inf])
        (0.216888906014633498739952 + 0.04081912627366673332369511j)

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    z = ctx.convert(z)
    p = ctx.convert(p)
    prec = ctx.prec
    try:
        ctx.prec += 20
        tol = ctx.eps * 2**10
        v = RJ_calc(ctx, x, y, z, p, tol, integration)
    finally:
        ctx.prec = prec
    return +v

@defun
def elliprd(ctx, x, y, z):
    r"""
    Evaluates the degenerate Carlson symmetric elliptic integral
    of the third kind or Carlson elliptic integral of the
    second kind `R_D(x,y,z) = R_J(x,y,z,z)`.

    See :func:`~mpmath.elliprj` for additional information.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprd(1,2,3)
        0.2904602810289906442326534
        >>> elliprj(1,2,3,3)
        0.2904602810289906442326534

    The so-called *second lemniscate constant*, a transcendental number::

        >>> elliprd(0,2,1)/3
        0.5990701173677961037199612
        >>> extradps(25)(quad)(lambda t: t**2/sqrt(1-t**4), [0,1])
        0.5990701173677961037199612
        >>> gamma('3/4')**2/sqrt(2*pi)
        0.5990701173677961037199612

    """
    return ctx.elliprj(x,y,z,z)

@defun
def elliprg(ctx, x, y, z):
    r"""
    Evaluates the Carlson completely symmetric elliptic integral
    of the second kind

    .. math ::

        R_G(x,y,z) = \frac{1}{4} \int_0^{\infty}
            \frac{t}{\sqrt{(t+x)(t+y)(t+z)}}
            \left( \frac{x}{t+x} + \frac{y}{t+y} + \frac{z}{t+z}\right) dt.

    **Examples**

    Evaluation for real and complex arguments::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprg(0,1,1)*4; +pi
        3.141592653589793238462643
        3.141592653589793238462643
        >>> elliprg(0,0.5,1)
        0.6753219405238377512600874
        >>> chop(elliprg(1+j, 1-j, 2))
        1.172431327676416604532822

    A double integral that can be evaluated in terms of `R_G`::

        >>> x,y,z = 2,3,4
        >>> def f(t,u):
        ...     st = fp.sin(t); ct = fp.cos(t)
        ...     su = fp.sin(u); cu = fp.cos(u)
        ...     return (x*(st*cu)**2 + y*(st*su)**2 + z*ct**2)**0.5 * st
        ...
        >>> nprint(mpf(fp.quad(f, [0,fp.pi], [0,2*fp.pi])/(4*fp.pi)), 13)
        1.725503028069
        >>> nprint(elliprg(x,y,z), 13)
        1.725503028069

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    z = ctx.convert(z)
    zeros = (not x) + (not y) + (not z)
    if zeros == 3:
        return (x+y+z)*0
    if zeros == 2:
        if x: return 0.5*ctx.sqrt(x)
        if y: return 0.5*ctx.sqrt(y)
        return 0.5*ctx.sqrt(z)
    if zeros == 1:
        if not z:
            x, z = z, x
    def terms():
        T1 = 0.5*z*ctx.elliprf(x,y,z)
        T2 = -0.5*(x-z)*(y-z)*ctx.elliprd(x,y,z)/3
        T3 = 0.5*ctx.sqrt(x)*ctx.sqrt(y)/ctx.sqrt(z)
        return T1,T2,T3
    return ctx.sum_accurately(terms)


@defun_wrapped
def ellipf(ctx, phi, m):
    r"""
    Evaluates the Legendre incomplete elliptic integral of the first kind

     .. math ::

        F(\phi,m) = \int_0^{\phi} \frac{dt}{\sqrt{1-m \sin^2 t}}

    or equivalently

    .. math ::

        F(\phi,m) = \int_0^{\sin \phi}
        \frac{dt}{\left(\sqrt{1-t^2}\right)\left(\sqrt{1-mt^2}\right)}.

    The function reduces to a complete elliptic integral of the first kind
    (see :func:`~mpmath.ellipk`) when `\phi = \frac{\pi}{2}`; that is,

    .. math ::

        F\left(\frac{\pi}{2}, m\right) = K(m).

    In the defining integral, it is assumed that the principal branch
    of the square root is taken and that the path of integration avoids
    crossing any branch cuts. Outside `-\pi/2 \le \Re(\phi) \le \pi/2`,
    the function extends quasi-periodically as

    .. math ::

        F(\phi + n \pi, m) = 2 n K(m) + F(\phi,m), n \in \mathbb{Z}.

    **Plots**

    .. literalinclude :: /plots/ellipf.py
    .. image :: /plots/ellipf.png

    **Examples**

    Basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> ellipf(0,1)
        0.0
        >>> ellipf(0,0)
        0.0
        >>> ellipf(1,0); ellipf(2+3j,0)
        1.0
        (2.0 + 3.0j)
        >>> ellipf(1,1); log(sec(1)+tan(1))
        1.226191170883517070813061
        1.226191170883517070813061
        >>> ellipf(pi/2, -0.5); ellipk(-0.5)
        1.415737208425956198892166
        1.415737208425956198892166
        >>> ellipf(pi/2+eps, 1); ellipf(-pi/2-eps, 1)
        +inf
        +inf
        >>> ellipf(1.5, 1)
        3.340677542798311003320813

    Comparing with numerical integration::

        >>> z,m = 0.5, 1.25
        >>> ellipf(z,m)
        0.5287219202206327872978255
        >>> quad(lambda t: (1-m*sin(t)**2)**(-0.5), [0,z])
        0.5287219202206327872978255

    The arguments may be complex numbers::

        >>> ellipf(3j, 0.5)
        (0.0 + 1.713602407841590234804143j)
        >>> ellipf(3+4j, 5-6j)
        (1.269131241950351323305741 - 0.3561052815014558335412538j)
        >>> z,m = 2+3j, 1.25
        >>> k = 1011
        >>> ellipf(z+pi*k,m); ellipf(z,m) + 2*k*ellipk(m)
        (4086.184383622179764082821 - 3003.003538923749396546871j)
        (4086.184383622179764082821 - 3003.003538923749396546871j)

    For `|\Re(z)| < \pi/2`, the function can be expressed as a
    hypergeometric series of two variables
    (see :func:`~mpmath.appellf1`)::

        >>> z,m = 0.5, 0.25
        >>> ellipf(z,m)
        0.5050887275786480788831083
        >>> sin(z)*appellf1(0.5,0.5,0.5,1.5,sin(z)**2,m*sin(z)**2)
        0.5050887275786480788831083

    """
    z = phi
    if not (ctx.isnormal(z) and ctx.isnormal(m)):
        if m == 0:
            return z + m
        if z == 0:
            return z * m
        if m == ctx.inf or m == ctx.ninf: return z/m
        raise ValueError
    x = z.real
    ctx.prec += max(0, ctx.mag(x))
    pi = +ctx.pi
    away = abs(x) > pi/2
    if m == 1:
        if away:
            return ctx.inf
    if away:
        d = ctx.nint(x/pi)
        z = z-pi*d
        P = 2*d*ctx.ellipk(m)
    else:
        P = 0
    c, s = ctx.cos_sin(z)
    return s * ctx.elliprf(c**2, 1-m*s**2, 1) + P

@defun_wrapped
def ellipe(ctx, *args):
    r"""
    Called with a single argument `m`, evaluates the Legendre complete
    elliptic integral of the second kind, `E(m)`, defined by

        .. math :: E(m) = \int_0^{\pi/2} \sqrt{1-m \sin^2 t} \, dt \,=\,
            \frac{\pi}{2}
            \,_2F_1\left(\frac{1}{2}, -\frac{1}{2}, 1, m\right).

    Called with two arguments `\phi, m`, evaluates the incomplete elliptic
    integral of the second kind

     .. math ::

        E(\phi,m) = \int_0^{\phi} \sqrt{1-m \sin^2 t} \, dt =
                    \int_0^{\sin z}
                    \frac{\sqrt{1-mt^2}}{\sqrt{1-t^2}} \, dt.

    The incomplete integral reduces to a complete integral when
    `\phi = \frac{\pi}{2}`; that is,

    .. math ::

        E\left(\frac{\pi}{2}, m\right) = E(m).

    In the defining integral, it is assumed that the principal branch
    of the square root is taken and that the path of integration avoids
    crossing any branch cuts. Outside `-\pi/2 \le \Re(z) \le \pi/2`,
    the function extends quasi-periodically as

    .. math ::

        E(\phi + n \pi, m) = 2 n E(m) + E(\phi,m), n \in \mathbb{Z}.

    **Plots**

    .. literalinclude :: /plots/ellipe.py
    .. image :: /plots/ellipe.png

    **Examples for the complete integral**

    Basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> ellipe(0)
        1.570796326794896619231322
        >>> ellipe(1)
        1.0
        >>> ellipe(-1)
        1.910098894513856008952381
        >>> ellipe(2)
        (0.5990701173677961037199612 + 0.5990701173677961037199612j)
        >>> ellipe(inf)
        (0.0 + +infj)
        >>> ellipe(-inf)
        +inf

    Verifying the defining integral and hypergeometric
    representation::

        >>> ellipe(0.5)
        1.350643881047675502520175
        >>> quad(lambda t: sqrt(1-0.5*sin(t)**2), [0, pi/2])
        1.350643881047675502520175
        >>> pi/2*hyp2f1(0.5,-0.5,1,0.5)
        1.350643881047675502520175

    Evaluation is supported for arbitrary complex `m`::

        >>> ellipe(0.5+0.25j)
        (1.360868682163129682716687 - 0.1238733442561786843557315j)
        >>> ellipe(3+4j)
        (1.499553520933346954333612 - 1.577879007912758274533309j)

    A definite integral::

        >>> quad(ellipe, [0,1])
        1.333333333333333333333333

    **Examples for the incomplete integral**

    Basic values and limits::

        >>> ellipe(0,1)
        0.0
        >>> ellipe(0,0)
        0.0
        >>> ellipe(1,0)
        1.0
        >>> ellipe(2+3j,0)
        (2.0 + 3.0j)
        >>> ellipe(1,1); sin(1)
        0.8414709848078965066525023
        0.8414709848078965066525023
        >>> ellipe(pi/2, -0.5); ellipe(-0.5)
        1.751771275694817862026502
        1.751771275694817862026502
        >>> ellipe(pi/2, 1); ellipe(-pi/2, 1)
        1.0
        -1.0
        >>> ellipe(1.5, 1)
        0.9974949866040544309417234

    Comparing with numerical integration::

        >>> z,m = 0.5, 1.25
        >>> ellipe(z,m)
        0.4740152182652628394264449
        >>> quad(lambda t: sqrt(1-m*sin(t)**2), [0,z])
        0.4740152182652628394264449

    The arguments may be complex numbers::

        >>> ellipe(3j, 0.5)
        (0.0 + 7.551991234890371873502105j)
        >>> ellipe(3+4j, 5-6j)
        (24.15299022574220502424466 + 75.2503670480325997418156j)
        >>> k = 35
        >>> z,m = 2+3j, 1.25
        >>> ellipe(z+pi*k,m); ellipe(z,m) + 2*k*ellipe(m)
        (48.30138799412005235090766 + 17.47255216721987688224357j)
        (48.30138799412005235090766 + 17.47255216721987688224357j)

    For `|\Re(z)| < \pi/2`, the function can be expressed as a
    hypergeometric series of two variables
    (see :func:`~mpmath.appellf1`)::

        >>> z,m = 0.5, 0.25
        >>> ellipe(z,m)
        0.4950017030164151928870375
        >>> sin(z)*appellf1(0.5,0.5,-0.5,1.5,sin(z)**2,m*sin(z)**2)
        0.4950017030164151928870376

    """
    if len(args) == 1:
        return ctx._ellipe(args[0])
    else:
        phi, m = args
    z = phi
    if not (ctx.isnormal(z) and ctx.isnormal(m)):
        if m == 0:
            return z + m
        if z == 0:
            return z * m
        if m == ctx.inf or m == ctx.ninf:
            return ctx.inf
        raise ValueError
    x = z.real
    ctx.prec += max(0, ctx.mag(x))
    pi = +ctx.pi
    away = abs(x) > pi/2
    if away:
        d = ctx.nint(x/pi)
        z = z-pi*d
        P = 2*d*ctx.ellipe(m)
    else:
        P = 0
    def terms():
        c, s = ctx.cos_sin(z)
        x = c**2
        y = 1-m*s**2
        RF = ctx.elliprf(x, y, 1)
        RD = ctx.elliprd(x, y, 1)
        return s*RF, -m*s**3*RD/3
    return ctx.sum_accurately(terms) + P

@defun_wrapped
def ellippi(ctx, *args):
    r"""
    Called with three arguments `n, \phi, m`, evaluates the Legendre
    incomplete elliptic integral of the third kind

    .. math ::

        \Pi(n; \phi, m) = \int_0^{\phi}
            \frac{dt}{(1-n \sin^2 t) \sqrt{1-m \sin^2 t}} =
            \int_0^{\sin \phi}
            \frac{dt}{(1-nt^2) \sqrt{1-t^2} \sqrt{1-mt^2}}.

    Called with two arguments `n, m`, evaluates the complete
    elliptic integral of the third kind
    `\Pi(n,m) = \Pi(n; \frac{\pi}{2},m)`.

    In the defining integral, it is assumed that the principal branch
    of the square root is taken and that the path of integration avoids
    crossing any branch cuts. Outside `-\pi/2 \le \Re(\phi) \le \pi/2`,
    the function extends quasi-periodically as

    .. math ::

        \Pi(n,\phi+k\pi,m) = 2k\Pi(n,m) + \Pi(n,\phi,m), k \in \mathbb{Z}.

    **Plots**

    .. literalinclude :: /plots/ellippi.py
    .. image :: /plots/ellippi.png

    **Examples for the complete integral**

    Some basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> ellippi(0,-5); ellipk(-5)
        0.9555039270640439337379334
        0.9555039270640439337379334
        >>> ellippi(inf,2)
        0.0
        >>> ellippi(2,inf)
        0.0
        >>> abs(ellippi(1,5))
        +inf
        >>> abs(ellippi(0.25,1))
        +inf

    Evaluation in terms of simpler functions::

        >>> ellippi(0.25,0.25); ellipe(0.25)/(1-0.25)
        1.956616279119236207279727
        1.956616279119236207279727
        >>> ellippi(3,0); pi/(2*sqrt(-2))
        (0.0 - 1.11072073453959156175397j)
        (0.0 - 1.11072073453959156175397j)
        >>> ellippi(-3,0); pi/(2*sqrt(4))
        0.7853981633974483096156609
        0.7853981633974483096156609

    **Examples for the incomplete integral**

    Basic values and limits::

        >>> ellippi(0.25,-0.5); ellippi(0.25,pi/2,-0.5)
        1.622944760954741603710555
        1.622944760954741603710555
        >>> ellippi(1,0,1)
        0.0
        >>> ellippi(inf,0,1)
        0.0
        >>> ellippi(0,0.25,0.5); ellipf(0.25,0.5)
        0.2513040086544925794134591
        0.2513040086544925794134591
        >>> ellippi(1,1,1); (log(sec(1)+tan(1))+sec(1)*tan(1))/2
        2.054332933256248668692452
        2.054332933256248668692452
        >>> ellippi(0.25, 53*pi/2, 0.75); 53*ellippi(0.25,0.75)
        135.240868757890840755058
        135.240868757890840755058
        >>> ellippi(0.5,pi/4,0.5); 2*ellipe(pi/4,0.5)-1/sqrt(3)
        0.9190227391656969903987269
        0.9190227391656969903987269

    Complex arguments are supported::

        >>> ellippi(0.5, 5+6j-2*pi, -7-8j)
        (-0.3612856620076747660410167 + 0.5217735339984807829755815j)

    Some degenerate cases::

        >>> ellippi(1,1)
        +inf
        >>> ellippi(1,0)
        +inf
        >>> ellippi(1,2,0)
        +inf
        >>> ellippi(1,2,1)
        +inf
        >>> ellippi(1,0,1)
        0.0

    """
    if len(args) == 2:
        n, m = args
        complete = True
        z = phi = ctx.pi/2
    else:
        n, phi, m = args
        complete = False
        z = phi
    if not (ctx.isnormal(n) and ctx.isnormal(z) and ctx.isnormal(m)):
        if ctx.isnan(n) or ctx.isnan(z) or ctx.isnan(m):
            raise ValueError
        if complete:
            if m == 0:
                if n == 1:
                    return ctx.inf
                return ctx.pi/(2*ctx.sqrt(1-n))
            if n == 0: return ctx.ellipk(m)
            if ctx.isinf(n) or ctx.isinf(m): return ctx.zero
        else:
            if z == 0: return z
            if ctx.isinf(n): return ctx.zero
            if ctx.isinf(m): return ctx.zero
        if ctx.isinf(n) or ctx.isinf(z) or ctx.isinf(m):
            raise ValueError
    if complete:
        if m == 1:
            if n == 1:
                return ctx.inf
            return -ctx.inf/ctx.sign(n-1)
        away = False
    else:
        x = z.real
        ctx.prec += max(0, ctx.mag(x))
        pi = +ctx.pi
        away = abs(x) > pi/2
    if away:
        d = ctx.nint(x/pi)
        z = z-pi*d
        P = 2*d*ctx.ellippi(n,m)
        if ctx.isinf(P):
            return ctx.inf
    else:
        P = 0
    def terms():
        if complete:
            c, s = ctx.zero, ctx.one
        else:
            c, s = ctx.cos_sin(z)
        x = c**2
        y = 1-m*s**2
        RF = ctx.elliprf(x, y, 1)
        RJ = ctx.elliprj(x, y, 1, 1-n*s**2)
        return s*RF, n*s**3*RJ/3
    return ctx.sum_accurately(terms) + P

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\libmp\libelefun.py ===
"""
This module implements computation of elementary transcendental
functions (powers, logarithms, trigonometric and hyperbolic
functions, inverse trigonometric and hyperbolic) for real
floating-point numbers.

For complex and interval implementations of the same functions,
see libmpc and libmpi.

"""

import math
from bisect import bisect

from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND

from .libmpf import (
    round_floor, round_ceiling, round_down, round_up,
    round_nearest, round_fast,
    ComplexResult,
    bitcount, bctable, lshift, rshift, giant_steps, sqrt_fixed,
    from_int, to_int, from_man_exp, to_fixed, to_float, from_float,
    from_rational, normalize,
    fzero, fone, fnone, fhalf, finf, fninf, fnan,
    mpf_cmp, mpf_sign, mpf_abs,
    mpf_pos, mpf_neg, mpf_add, mpf_sub, mpf_mul, mpf_div, mpf_shift,
    mpf_rdiv_int, mpf_pow_int, mpf_sqrt,
    reciprocal_rnd, negative_rnd, mpf_perturb,
    isqrt_fast
)

from .libintmath import ifib


#-------------------------------------------------------------------------------
# Tuning parameters
#-------------------------------------------------------------------------------

# Cutoff for computing exp from cosh+sinh. This reduces the
# number of terms by half, but also requires a square root which
# is expensive with the pure-Python square root code.
if BACKEND == 'python':
    EXP_COSH_CUTOFF = 600
else:
    EXP_COSH_CUTOFF = 400
# Cutoff for using more than 2 series
EXP_SERIES_U_CUTOFF = 1500

# Also basically determined by sqrt
if BACKEND == 'python':
    COS_SIN_CACHE_PREC = 400
else:
    COS_SIN_CACHE_PREC = 200
COS_SIN_CACHE_STEP = 8
cos_sin_cache = {}

# Number of integer logarithms to cache (for zeta sums)
MAX_LOG_INT_CACHE = 2000
log_int_cache = {}

LOG_TAYLOR_PREC = 2500  # Use Taylor series with caching up to this prec
LOG_TAYLOR_SHIFT = 9    # Cache log values in steps of size 2^-N
log_taylor_cache = {}
# prec/size ratio of x for fastest convergence in AGM formula
LOG_AGM_MAG_PREC_RATIO = 20

ATAN_TAYLOR_PREC = 3000  # Same as for log
ATAN_TAYLOR_SHIFT = 7   # steps of size 2^-N
atan_taylor_cache = {}


# ~= next power of two + 20
cache_prec_steps = [22,22]
for k in xrange(1, bitcount(LOG_TAYLOR_PREC)+1):
    cache_prec_steps += [min(2**k,LOG_TAYLOR_PREC)+20] * 2**(k-1)


#----------------------------------------------------------------------------#
#                                                                            #
#                   Elementary mathematical constants                        #
#                                                                            #
#----------------------------------------------------------------------------#

def constant_memo(f):
    """
    Decorator for caching computed values of mathematical
    constants. This decorator should be applied to a
    function taking a single argument prec as input and
    returning a fixed-point value with the given precision.
    """
    f.memo_prec = -1
    f.memo_val = None
    def g(prec, **kwargs):
        memo_prec = f.memo_prec
        if prec <= memo_prec:
            return f.memo_val >> (memo_prec-prec)
        newprec = int(prec*1.05+10)
        f.memo_val = f(newprec, **kwargs)
        f.memo_prec = newprec
        return f.memo_val >> (newprec-prec)
    g.__name__ = f.__name__
    g.__doc__ = f.__doc__
    return g

def def_mpf_constant(fixed):
    """
    Create a function that computes the mpf value for a mathematical
    constant, given a function that computes the fixed-point value.

    Assumptions: the constant is positive and has magnitude ~= 1;
    the fixed-point function rounds to floor.
    """
    def f(prec, rnd=round_fast):
        wp = prec + 20
        v = fixed(wp)
        if rnd in (round_up, round_ceiling):
            v += 1
        return normalize(0, v, -wp, bitcount(v), prec, rnd)
    f.__doc__ = fixed.__doc__
    return f

def bsp_acot(q, a, b, hyperbolic):
    if b - a == 1:
        a1 = MPZ(2*a + 3)
        if hyperbolic or a&1:
            return MPZ_ONE, a1 * q**2, a1
        else:
            return -MPZ_ONE, a1 * q**2, a1
    m = (a+b)//2
    p1, q1, r1 = bsp_acot(q, a, m, hyperbolic)
    p2, q2, r2 = bsp_acot(q, m, b, hyperbolic)
    return q2*p1 + r1*p2, q1*q2, r1*r2

# the acoth(x) series converges like the geometric series for x^2
# N = ceil(p*log(2)/(2*log(x)))
def acot_fixed(a, prec, hyperbolic):
    """
    Compute acot(a) or acoth(a) for an integer a with binary splitting; see
    http://numbers.computation.free.fr/Constants/Algorithms/splitting.html
    """
    N = int(0.35 * prec/math.log(a) + 20)
    p, q, r = bsp_acot(a, 0,N, hyperbolic)
    return ((p+q)<<prec)//(q*a)

def machin(coefs, prec, hyperbolic=False):
    """
    Evaluate a Machin-like formula, i.e., a linear combination of
    acot(n) or acoth(n) for specific integer values of n, using fixed-
    point arithmetic. The input should be a list [(c, n), ...], giving
    c*acot[h](n) + ...
    """
    extraprec = 10
    s = MPZ_ZERO
    for a, b in coefs:
        s += MPZ(a) * acot_fixed(MPZ(b), prec+extraprec, hyperbolic)
    return (s >> extraprec)

# Logarithms of integers are needed for various computations involving
# logarithms, powers, radix conversion, etc

@constant_memo
def ln2_fixed(prec):
    """
    Computes ln(2). This is done with a hyperbolic Machin-type formula,
    with binary splitting at high precision.
    """
    return machin([(18, 26), (-2, 4801), (8, 8749)], prec, True)

@constant_memo
def ln10_fixed(prec):
    """
    Computes ln(10). This is done with a hyperbolic Machin-type formula.
    """
    return machin([(46, 31), (34, 49), (20, 161)], prec, True)


r"""
For computation of pi, we use the Chudnovsky series:

             oo
             ___        k
      1     \       (-1)  (6 k)! (A + B k)
    ----- =  )     -----------------------
    12 pi   /___               3  3k+3/2
                    (3 k)! (k!)  C
            k = 0

where A, B, and C are certain integer constants. This series adds roughly
14 digits per term. Note that C^(3/2) can be extracted so that the
series contains only rational terms. This makes binary splitting very
efficient.

The recurrence formulas for the binary splitting were taken from
ftp://ftp.gmplib.org/pub/src/gmp-chudnovsky.c

Previously, Machin's formula was used at low precision and the AGM iteration
was used at high precision. However, the Chudnovsky series is essentially as
fast as the Machin formula at low precision and in practice about 3x faster
than the AGM at high precision (despite theoretically having a worse
asymptotic complexity), so there is no reason not to use it in all cases.

"""

# Constants in Chudnovsky's series
CHUD_A = MPZ(13591409)
CHUD_B = MPZ(545140134)
CHUD_C = MPZ(640320)
CHUD_D = MPZ(12)

def bs_chudnovsky(a, b, level, verbose):
    """
    Computes the sum from a to b of the series in the Chudnovsky
    formula. Returns g, p, q where p/q is the sum as an exact
    fraction and g is a temporary value used to save work
    for recursive calls.
    """
    if b-a == 1:
        g = MPZ((6*b-5)*(2*b-1)*(6*b-1))
        p = b**3 * CHUD_C**3 // 24
        q = (-1)**b * g * (CHUD_A+CHUD_B*b)
    else:
        if verbose and level < 4:
            print("  binary splitting", a, b)
        mid = (a+b)//2
        g1, p1, q1 = bs_chudnovsky(a, mid, level+1, verbose)
        g2, p2, q2 = bs_chudnovsky(mid, b, level+1, verbose)
        p = p1*p2
        g = g1*g2
        q = q1*p2 + q2*g1
    return g, p, q

@constant_memo
def pi_fixed(prec, verbose=False, verbose_base=None):
    """
    Compute floor(pi * 2**prec) as a big integer.

    This is done using Chudnovsky's series (see comments in
    libelefun.py for details).
    """
    # The Chudnovsky series gives 14.18 digits per term
    N = int(prec/3.3219280948/14.181647462 + 2)
    if verbose:
        print("binary splitting with N =", N)
    g, p, q = bs_chudnovsky(0, N, 0, verbose)
    sqrtC = isqrt_fast(CHUD_C<<(2*prec))
    v = p*CHUD_C*sqrtC//((q+CHUD_A*p)*CHUD_D)
    return v

def degree_fixed(prec):
    return pi_fixed(prec)//180

def bspe(a, b):
    """
    Sum series for exp(1)-1 between a, b, returning the result
    as an exact fraction (p, q).
    """
    if b-a == 1:
        return MPZ_ONE, MPZ(b)
    m = (a+b)//2
    p1, q1 = bspe(a, m)
    p2, q2 = bspe(m, b)
    return p1*q2+p2, q1*q2

@constant_memo
def e_fixed(prec):
    """
    Computes exp(1). This is done using the ordinary Taylor series for
    exp, with binary splitting. For a description of the algorithm,
    see:

        http://numbers.computation.free.fr/Constants/
            Algorithms/splitting.html
    """
    # Slight overestimate of N needed for 1/N! < 2**(-prec)
    # This could be tightened for large N.
    N = int(1.1*prec/math.log(prec) + 20)
    p, q = bspe(0,N)
    return ((p+q)<<prec)//q

@constant_memo
def phi_fixed(prec):
    """
    Computes the golden ratio, (1+sqrt(5))/2
    """
    prec += 10
    a = isqrt_fast(MPZ_FIVE<<(2*prec)) + (MPZ_ONE << prec)
    return a >> 11

mpf_phi    = def_mpf_constant(phi_fixed)
mpf_pi     = def_mpf_constant(pi_fixed)
mpf_e      = def_mpf_constant(e_fixed)
mpf_degree = def_mpf_constant(degree_fixed)
mpf_ln2    = def_mpf_constant(ln2_fixed)
mpf_ln10   = def_mpf_constant(ln10_fixed)


@constant_memo
def ln_sqrt2pi_fixed(prec):
    wp = prec + 10
    # ln(sqrt(2*pi)) = ln(2*pi)/2
    return to_fixed(mpf_log(mpf_shift(mpf_pi(wp), 1), wp), prec-1)

@constant_memo
def sqrtpi_fixed(prec):
    return sqrt_fixed(pi_fixed(prec), prec)

mpf_sqrtpi   = def_mpf_constant(sqrtpi_fixed)
mpf_ln_sqrt2pi   = def_mpf_constant(ln_sqrt2pi_fixed)


#----------------------------------------------------------------------------#
#                                                                            #
#                                    Powers                                  #
#                                                                            #
#----------------------------------------------------------------------------#

def mpf_pow(s, t, prec, rnd=round_fast):
    """
    Compute s**t. Raises ComplexResult if s is negative and t is
    fractional.
    """
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if ssign and texp < 0:
        raise ComplexResult("negative number raised to a fractional power")
    if texp >= 0:
        return mpf_pow_int(s, (-1)**tsign * (tman<<texp), prec, rnd)
    # s**(n/2) = sqrt(s)**n
    if texp == -1:
        if tman == 1:
            if tsign:
                return mpf_div(fone, mpf_sqrt(s, prec+10,
                    reciprocal_rnd[rnd]), prec, rnd)
            return mpf_sqrt(s, prec, rnd)
        else:
            if tsign:
                return mpf_pow_int(mpf_sqrt(s, prec+10,
                    reciprocal_rnd[rnd]), -tman, prec, rnd)
            return mpf_pow_int(mpf_sqrt(s, prec+10, rnd), tman, prec, rnd)
    # General formula: s**t = exp(t*log(s))
    # TODO: handle rnd direction of the logarithm carefully
    c = mpf_log(s, prec+10, rnd)
    return mpf_exp(mpf_mul(t, c), prec, rnd)

def int_pow_fixed(y, n, prec):
    """n-th power of a fixed point number with precision prec

       Returns the power in the form man, exp,
       man * 2**exp ~= y**n
    """
    if n == 2:
        return (y*y), 0
    bc = bitcount(y)
    exp = 0
    workprec = 2 * (prec + 4*bitcount(n) + 4)
    _, pm, pe, pbc = fone
    while 1:
        if n & 1:
            pm = pm*y
            pe = pe+exp
            pbc += bc - 2
            pbc = pbc + bctable[int(pm >> pbc)]
            if pbc > workprec:
                pm = pm >> (pbc-workprec)
                pe += pbc - workprec
                pbc = workprec
            n -= 1
            if not n:
                break
        y = y*y
        exp = exp+exp
        bc = bc + bc - 2
        bc = bc + bctable[int(y >> bc)]
        if bc > workprec:
            y = y >> (bc-workprec)
            exp += bc - workprec
            bc = workprec
        n = n // 2
    return pm, pe

# froot(s, n, prec, rnd) computes the real n-th root of a
# positive mpf tuple s.
# To compute the root we start from a 50-bit estimate for r
# generated with ordinary floating-point arithmetic, and then refine
# the value to full accuracy using the iteration

#            1  /                     y       \
#   r     = --- | (n-1)  * r   +  ----------  |
#    n+1     n  \           n     r_n**(n-1)  /

# which is simply Newton's method applied to the equation r**n = y.
# With giant_steps(start, prec+extra) = [p0,...,pm, prec+extra]
# and y = man * 2**-shift  one has
# (man * 2**exp)**(1/n) =
# y**(1/n) * 2**(start-prec/n) * 2**(p0-start) * ... * 2**(prec+extra-pm) *
# 2**((exp+shift-(n-1)*prec)/n -extra))
# The last factor is accounted for in the last line of froot.

def nthroot_fixed(y, n, prec, exp1):
    start = 50
    try:
        y1 = rshift(y, prec - n*start)
        r = MPZ(int(y1**(1.0/n)))
    except OverflowError:
        y1 = from_int(y1, start)
        fn = from_int(n)
        fn = mpf_rdiv_int(1, fn, start)
        r = mpf_pow(y1, fn, start)
        r = to_int(r)
    extra = 10
    extra1 = n
    prevp = start
    for p in giant_steps(start, prec+extra):
        pm, pe = int_pow_fixed(r, n-1, prevp)
        r2 = rshift(pm, (n-1)*prevp - p - pe - extra1)
        B = lshift(y, 2*p-prec+extra1)//r2
        r = (B + (n-1) * lshift(r, p-prevp))//n
        prevp = p
    return r

def mpf_nthroot(s, n, prec, rnd=round_fast):
    """nth-root of a positive number

    Use the Newton method when faster, otherwise use x**(1/n)
    """
    sign, man, exp, bc = s
    if sign:
        raise ComplexResult("nth root of a negative number")
    if not man:
        if s == fnan:
            return fnan
        if s == fzero:
            if n > 0:
                return fzero
            if n == 0:
                return fone
            return finf
        # Infinity
        if not n:
            return fnan
        if n < 0:
            return fzero
        return finf
    flag_inverse = False
    if n < 2:
        if n == 0:
            return fone
        if n == 1:
            return mpf_pos(s, prec, rnd)
        if n == -1:
            return mpf_div(fone, s, prec, rnd)
        # n < 0
        rnd = reciprocal_rnd[rnd]
        flag_inverse = True
        extra_inverse = 5
        prec += extra_inverse
        n = -n
    if n > 20 and (n >= 20000 or prec < int(233 + 28.3 * n**0.62)):
        prec2 = prec + 10
        fn = from_int(n)
        nth = mpf_rdiv_int(1, fn, prec2)
        r = mpf_pow(s, nth, prec2, rnd)
        s = normalize(r[0], r[1], r[2], r[3], prec, rnd)
        if flag_inverse:
            return mpf_div(fone, s, prec-extra_inverse, rnd)
        else:
            return s
    # Convert to a fixed-point number with prec2 bits.
    prec2 = prec + 2*n - (prec%n)
    # a few tests indicate that
    # for 10 < n < 10**4 a bit more precision is needed
    if n > 10:
        prec2 += prec2//10
        prec2 = prec2 - prec2%n
    # Mantissa may have more bits than we need. Trim it down.
    shift = bc - prec2
    # Adjust exponents to make prec2 and exp+shift multiples of n.
    sign1 = 0
    es = exp+shift
    if es < 0:
        sign1 = 1
        es = -es
    if sign1:
        shift += es%n
    else:
        shift -= es%n
    man = rshift(man, shift)
    extra = 10
    exp1 = ((exp+shift-(n-1)*prec2)//n) - extra
    rnd_shift = 0
    if flag_inverse:
        if rnd == 'u' or rnd == 'c':
            rnd_shift = 1
    else:
        if rnd == 'd' or rnd == 'f':
            rnd_shift = 1
    man = nthroot_fixed(man+rnd_shift, n, prec2, exp1)
    s = from_man_exp(man, exp1, prec, rnd)
    if flag_inverse:
        return mpf_div(fone, s, prec-extra_inverse, rnd)
    else:
        return s

def mpf_cbrt(s, prec, rnd=round_fast):
    """cubic root of a positive number"""
    return mpf_nthroot(s, 3, prec, rnd)

#----------------------------------------------------------------------------#
#                                                                            #
#                                Logarithms                                  #
#                                                                            #
#----------------------------------------------------------------------------#


def log_int_fixed(n, prec, ln2=None):
    """
    Fast computation of log(n), caching the value for small n,
    intended for zeta sums.
    """
    if n in log_int_cache:
        value, vprec = log_int_cache[n]
        if vprec >= prec:
            return value >> (vprec - prec)
    wp = prec + 10
    if wp <= LOG_TAYLOR_SHIFT:
        if ln2 is None:
            ln2 = ln2_fixed(wp)
        r = bitcount(n)
        x = n << (wp-r)
        v = log_taylor_cached(x, wp) + r*ln2
    else:
        v = to_fixed(mpf_log(from_int(n), wp+5), wp)
    if n < MAX_LOG_INT_CACHE:
        log_int_cache[n] = (v, wp)
    return v >> (wp-prec)

def agm_fixed(a, b, prec):
    """
    Fixed-point computation of agm(a,b), assuming
    a, b both close to unit magnitude.
    """
    i = 0
    while 1:
        anew = (a+b)>>1
        if i > 4 and abs(a-anew) < 8:
            return a
        b = isqrt_fast(a*b)
        a = anew
        i += 1
    return a

def log_agm(x, prec):
    """
    Fixed-point computation of -log(x) = log(1/x), suitable
    for large precision. It is required that 0 < x < 1. The
    algorithm used is the Sasaki-Kanada formula

        -log(x) = pi/agm(theta2(x)^2,theta3(x)^2). [1]

    For faster convergence in the theta functions, x should
    be chosen closer to 0.

    Guard bits must be added by the caller.

    HYPOTHESIS: if x = 2^(-n), n bits need to be added to
    account for the truncation to a fixed-point number,
    and this is the only significant cancellation error.

    The number of bits lost to roundoff is small and can be
    considered constant.

    [1] Richard P. Brent, "Fast Algorithms for High-Precision
        Computation of Elementary Functions (extended abstract)",
        http://wwwmaths.anu.edu.au/~brent/pd/RNC7-Brent.pdf

    """
    x2 = (x*x) >> prec
    # Compute jtheta2(x)**2
    s = a = b = x2
    while a:
        b = (b*x2) >> prec
        a = (a*b) >> prec
        s += a
    s += (MPZ_ONE<<prec)
    s = (s*s)>>(prec-2)
    s = (s*isqrt_fast(x<<prec))>>prec
    # Compute jtheta3(x)**2
    t = a = b = x
    while a:
        b = (b*x2) >> prec
        a = (a*b) >> prec
        t += a
    t = (MPZ_ONE<<prec) + (t<<1)
    t = (t*t)>>prec
    # Final formula
    p = agm_fixed(s, t, prec)
    return (pi_fixed(prec) << prec) // p

def log_taylor(x, prec, r=0):
    """
    Fixed-point calculation of log(x). It is assumed that x is close
    enough to 1 for the Taylor series to converge quickly. Convergence
    can be improved by specifying r > 0 to compute
    log(x^(1/2^r))*2^r, at the cost of performing r square roots.

    The caller must provide sufficient guard bits.
    """
    for i in xrange(r):
        x = isqrt_fast(x<<prec)
    one = MPZ_ONE << prec
    v = ((x-one)<<prec)//(x+one)
    sign = v < 0
    if sign:
        v = -v
    v2 = (v*v) >> prec
    v4 = (v2*v2) >> prec
    s0 = v
    s1 = v//3
    v = (v*v4) >> prec
    k = 5
    while v:
        s0 += v // k
        k += 2
        s1 += v // k
        v = (v*v4) >> prec
        k += 2
    s1 = (s1*v2) >> prec
    s = (s0+s1) << (1+r)
    if sign:
        return -s
    return s

def log_taylor_cached(x, prec):
    """
    Fixed-point computation of log(x), assuming x in (0.5, 2)
    and prec <= LOG_TAYLOR_PREC.
    """
    n = x >> (prec-LOG_TAYLOR_SHIFT)
    cached_prec = cache_prec_steps[prec]
    dprec = cached_prec - prec
    if (n, cached_prec) in log_taylor_cache:
        a, log_a = log_taylor_cache[n, cached_prec]
    else:
        a = n << (cached_prec - LOG_TAYLOR_SHIFT)
        log_a = log_taylor(a, cached_prec, 8)
        log_taylor_cache[n, cached_prec] = (a, log_a)
    a >>= dprec
    log_a >>= dprec
    u = ((x - a) << prec) // a
    v = (u << prec) // ((MPZ_TWO << prec) + u)
    v2 = (v*v) >> prec
    v4 = (v2*v2) >> prec
    s0 = v
    s1 = v//3
    v = (v*v4) >> prec
    k = 5
    while v:
        s0 += v//k
        k += 2
        s1 += v//k
        v = (v*v4) >> prec
        k += 2
    s1 = (s1*v2) >> prec
    s = (s0+s1) << 1
    return log_a + s

def mpf_log(x, prec, rnd=round_fast):
    """
    Compute the natural logarithm of the mpf value x. If x is negative,
    ComplexResult is raised.
    """
    sign, man, exp, bc = x
    #------------------------------------------------------------------
    # Handle special values
    if not man:
        if x == fzero: return fninf
        if x == finf: return finf
        if x == fnan: return fnan
    if sign:
        raise ComplexResult("logarithm of a negative number")
    wp = prec + 20
    #------------------------------------------------------------------
    # Handle log(2^n) = log(n)*2.
    # Here we catch the only possible exact value, log(1) = 0
    if man == 1:
        if not exp:
            return fzero
        return from_man_exp(exp*ln2_fixed(wp), -wp, prec, rnd)
    mag = exp+bc
    abs_mag = abs(mag)
    #------------------------------------------------------------------
    # Handle x = 1+eps, where log(x) ~ x. We need to check for
    # cancellation when moving to fixed-point math and compensate
    # by increasing the precision. Note that abs_mag in (0, 1) <=>
    # 0.5 < x < 2 and x != 1
    if abs_mag <= 1:
        # Calculate t = x-1 to measure distance from 1 in bits
        tsign = 1-abs_mag
        if tsign:
            tman = (MPZ_ONE<<bc) - man
        else:
            tman = man - (MPZ_ONE<<(bc-1))
        tbc = bitcount(tman)
        cancellation = bc - tbc
        if cancellation > wp:
            t = normalize(tsign, tman, abs_mag-bc, tbc, tbc, 'n')
            return mpf_perturb(t, tsign, prec, rnd)
        else:
            wp += cancellation
        # TODO: if close enough to 1, we could use Taylor series
        # even in the AGM precision range, since the Taylor series
        # converges rapidly
    #------------------------------------------------------------------
    # Another special case:
    # n*log(2) is a good enough approximation
    if abs_mag > 10000:
        if bitcount(abs_mag) > wp:
            return from_man_exp(exp*ln2_fixed(wp), -wp, prec, rnd)
    #------------------------------------------------------------------
    # General case.
    # Perform argument reduction using log(x) = log(x*2^n) - n*log(2):
    # If we are in the Taylor precision range, choose magnitude 0 or 1.
    # If we are in the AGM precision range, choose magnitude -m for
    # some large m; benchmarking on one machine showed m = prec/20 to be
    # optimal between 1000 and 100,000 digits.
    if wp <= LOG_TAYLOR_PREC:
        m = log_taylor_cached(lshift(man, wp-bc), wp)
        if mag:
            m += mag*ln2_fixed(wp)
    else:
        optimal_mag = -wp//LOG_AGM_MAG_PREC_RATIO
        n = optimal_mag - mag
        x = mpf_shift(x, n)
        wp += (-optimal_mag)
        m = -log_agm(to_fixed(x, wp), wp)
        m -= n*ln2_fixed(wp)
    return from_man_exp(m, -wp, prec, rnd)

def mpf_log_hypot(a, b, prec, rnd):
    """
    Computes log(sqrt(a^2+b^2)) accurately.
    """
    # If either a or b is inf/nan/0, assume it to be a
    if not b[1]:
        a, b = b, a
    # a is inf/nan/0
    if not a[1]:
        # both are inf/nan/0
        if not b[1]:
            if a == b == fzero:
                return fninf
            if fnan in (a, b):
                return fnan
            # at least one term is (+/- inf)^2
            return finf
        # only a is inf/nan/0
        if a == fzero:
            # log(sqrt(0+b^2)) = log(|b|)
            return mpf_log(mpf_abs(b), prec, rnd)
        if a == fnan:
            return fnan
        return finf
    # Exact
    a2 = mpf_mul(a,a)
    b2 = mpf_mul(b,b)
    extra = 20
    # Not exact
    h2 = mpf_add(a2, b2, prec+extra)
    cancelled = mpf_add(h2, fnone, 10)
    mag_cancelled = cancelled[2]+cancelled[3]
    # Just redo the sum exactly if necessary (could be smarter
    # and avoid memory allocation when a or b is precisely 1
    # and the other is tiny...)
    if cancelled == fzero or mag_cancelled < -extra//2:
        h2 = mpf_add(a2, b2, prec+extra-min(a2[2],b2[2]))
    return mpf_shift(mpf_log(h2, prec, rnd), -1)


#----------------------------------------------------------------------
# Inverse tangent
#

def atan_newton(x, prec):
    if prec >= 100:
        r = math.atan(int((x>>(prec-53)))/2.0**53)
    else:
        r = math.atan(int(x)/2.0**prec)
    prevp = 50
    r = MPZ(int(r * 2.0**53) >> (53-prevp))
    extra_p = 50
    for wp in giant_steps(prevp, prec):
        wp += extra_p
        r = r << (wp-prevp)
        cos, sin = cos_sin_fixed(r, wp)
        tan = (sin << wp) // cos
        a = ((tan-rshift(x, prec-wp)) << wp) // ((MPZ_ONE<<wp) + ((tan**2)>>wp))
        r = r - a
        prevp = wp
    return rshift(r, prevp-prec)

def atan_taylor_get_cached(n, prec):
    # Taylor series with caching wins up to huge precisions
    # To avoid unnecessary precomputation at low precision, we
    # do it in steps
    # Round to next power of 2
    prec2 = (1<<(bitcount(prec-1))) + 20
    dprec = prec2 - prec
    if (n, prec2) in atan_taylor_cache:
        a, atan_a = atan_taylor_cache[n, prec2]
    else:
        a = n << (prec2 - ATAN_TAYLOR_SHIFT)
        atan_a = atan_newton(a, prec2)
        atan_taylor_cache[n, prec2] = (a, atan_a)
    return (a >> dprec), (atan_a >> dprec)

def atan_taylor(x, prec):
    n = (x >> (prec-ATAN_TAYLOR_SHIFT))
    a, atan_a = atan_taylor_get_cached(n, prec)
    d = x - a
    s0 = v = (d << prec) // ((a**2 >> prec) + (a*d >> prec) + (MPZ_ONE << prec))
    v2 = (v**2 >> prec)
    v4 = (v2 * v2) >> prec
    s1 = v//3
    v = (v * v4) >> prec
    k = 5
    while v:
        s0 += v // k
        k += 2
        s1 += v // k
        v = (v * v4) >> prec
        k += 2
    s1 = (s1 * v2) >> prec
    s = s0 - s1
    return atan_a + s

def atan_inf(sign, prec, rnd):
    if not sign:
        return mpf_shift(mpf_pi(prec, rnd), -1)
    return mpf_neg(mpf_shift(mpf_pi(prec, negative_rnd[rnd]), -1))

def mpf_atan(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero: return fzero
        if x == finf: return atan_inf(0, prec, rnd)
        if x == fninf: return atan_inf(1, prec, rnd)
        return fnan
    mag = exp + bc
    # Essentially infinity
    if mag > prec+20:
        return atan_inf(sign, prec, rnd)
    # Essentially ~ x
    if -mag > prec+20:
        return mpf_perturb(x, 1-sign, prec, rnd)
    wp = prec + 30 + abs(mag)
    # For large x, use atan(x) = pi/2 - atan(1/x)
    if mag >= 2:
        x = mpf_rdiv_int(1, x, wp)
        reciprocal = True
    else:
        reciprocal = False
    t = to_fixed(x, wp)
    if sign:
        t = -t
    if wp < ATAN_TAYLOR_PREC:
        a = atan_taylor(t, wp)
    else:
        a = atan_newton(t, wp)
    if reciprocal:
        a = ((pi_fixed(wp)>>1)+1) - a
    if sign:
        a = -a
    return from_man_exp(a, -wp, prec, rnd)

# TODO: cleanup the special cases
def mpf_atan2(y, x, prec, rnd=round_fast):
    xsign, xman, xexp, xbc = x
    ysign, yman, yexp, ybc = y
    if not yman:
        if y == fzero and x != fnan:
            if mpf_sign(x) >= 0:
                return fzero
            return mpf_pi(prec, rnd)
        if y in (finf, fninf):
            if x in (finf, fninf):
                return fnan
            # pi/2
            if y == finf:
                return mpf_shift(mpf_pi(prec, rnd), -1)
            # -pi/2
            return mpf_neg(mpf_shift(mpf_pi(prec, negative_rnd[rnd]), -1))
        return fnan
    if ysign:
        return mpf_neg(mpf_atan2(mpf_neg(y), x, prec, negative_rnd[rnd]))
    if not xman:
        if x == fnan:
            return fnan
        if x == finf:
            return fzero
        if x == fninf:
            return mpf_pi(prec, rnd)
        if y == fzero:
            return fzero
        return mpf_shift(mpf_pi(prec, rnd), -1)
    tquo = mpf_atan(mpf_div(y, x, prec+4), prec+4)
    if xsign:
        return mpf_add(mpf_pi(prec+4), tquo, prec, rnd)
    else:
        return mpf_pos(tquo, prec, rnd)

def mpf_asin(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if bc+exp > 0 and x not in (fone, fnone):
        raise ComplexResult("asin(x) is real only for -1 <= x <= 1")
    # asin(x) = 2*atan(x/(1+sqrt(1-x**2)))
    wp = prec + 15
    a = mpf_mul(x, x)
    b = mpf_add(fone, mpf_sqrt(mpf_sub(fone, a, wp), wp), wp)
    c = mpf_div(x, b, wp)
    return mpf_shift(mpf_atan(c, prec, rnd), 1)

def mpf_acos(x, prec, rnd=round_fast):
    # acos(x) = 2*atan(sqrt(1-x**2)/(1+x))
    sign, man, exp, bc = x
    if bc + exp > 0:
        if x not in (fone, fnone):
            raise ComplexResult("acos(x) is real only for -1 <= x <= 1")
        if x == fnone:
            return mpf_pi(prec, rnd)
    wp = prec + 15
    a = mpf_mul(x, x)
    b = mpf_sqrt(mpf_sub(fone, a, wp), wp)
    c = mpf_div(b, mpf_add(fone, x, wp), wp)
    return mpf_shift(mpf_atan(c, prec, rnd), 1)

def mpf_asinh(x, prec, rnd=round_fast):
    wp = prec + 20
    sign, man, exp, bc = x
    mag = exp+bc
    if mag < -8:
        if mag < -wp:
            return mpf_perturb(x, 1-sign, prec, rnd)
        wp += (-mag)
    # asinh(x) = log(x+sqrt(x**2+1))
    # use reflection symmetry to avoid cancellation
    q = mpf_sqrt(mpf_add(mpf_mul(x, x), fone, wp), wp)
    q = mpf_add(mpf_abs(x), q, wp)
    if sign:
        return mpf_neg(mpf_log(q, prec, negative_rnd[rnd]))
    else:
        return mpf_log(q, prec, rnd)

def mpf_acosh(x, prec, rnd=round_fast):
    # acosh(x) = log(x+sqrt(x**2-1))
    wp = prec + 15
    if mpf_cmp(x, fone) == -1:
        raise ComplexResult("acosh(x) is real only for x >= 1")
    q = mpf_sqrt(mpf_add(mpf_mul(x,x), fnone, wp), wp)
    return mpf_log(mpf_add(x, q, wp), prec, rnd)

def mpf_atanh(x, prec, rnd=round_fast):
    # atanh(x) = log((1+x)/(1-x))/2
    sign, man, exp, bc = x
    if (not man) and exp:
        if x in (fzero, fnan):
            return x
        raise ComplexResult("atanh(x) is real only for -1 <= x <= 1")
    mag = bc + exp
    if mag > 0:
        if mag == 1 and man == 1:
            return [finf, fninf][sign]
        raise ComplexResult("atanh(x) is real only for -1 <= x <= 1")
    wp = prec + 15
    if mag < -8:
        if mag < -wp:
            return mpf_perturb(x, sign, prec, rnd)
        wp += (-mag)
    a = mpf_add(x, fone, wp)
    b = mpf_sub(fone, x, wp)
    return mpf_shift(mpf_log(mpf_div(a, b, wp), prec, rnd), -1)

def mpf_fibonacci(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fninf:
            return fnan
        return x
    # F(2^n) ~= 2^(2^n)
    size = abs(exp+bc)
    if exp >= 0:
        # Exact
        if size < 10 or size <= bitcount(prec):
            return from_int(ifib(to_int(x)), prec, rnd)
    # Use the modified Binet formula
    wp = prec + size + 20
    a = mpf_phi(wp)
    b = mpf_add(mpf_shift(a, 1), fnone, wp)
    u = mpf_pow(a, x, wp)
    v = mpf_cos_pi(x, wp)
    v = mpf_div(v, u, wp)
    u = mpf_sub(u, v, wp)
    u = mpf_div(u, b, prec, rnd)
    return u


#-------------------------------------------------------------------------------
# Exponential-type functions
#-------------------------------------------------------------------------------

def exponential_series(x, prec, type=0):
    """
    Taylor series for cosh/sinh or cos/sin.

    type = 0 -- returns exp(x)  (slightly faster than cosh+sinh)
    type = 1 -- returns (cosh(x), sinh(x))
    type = 2 -- returns (cos(x), sin(x))
    """
    if x < 0:
        x = -x
        sign = 1
    else:
        sign = 0
    r = int(0.5*prec**0.5)
    xmag = bitcount(x) - prec
    r = max(0, xmag + r)
    extra = 10 + 2*max(r,-xmag)
    wp = prec + extra
    x <<= (extra - r)
    one = MPZ_ONE << wp
    alt = (type == 2)
    if prec < EXP_SERIES_U_CUTOFF:
        x2 = a = (x*x) >> wp
        x4 = (x2*x2) >> wp
        s0 = s1 = MPZ_ZERO
        k = 2
        while a:
            a //= (k-1)*k; s0 += a; k += 2
            a //= (k-1)*k; s1 += a; k += 2
            a = (a*x4) >> wp
        s1 = (x2*s1) >> wp
        if alt:
            c = s1 - s0 + one
        else:
            c = s1 + s0 + one
    else:
        u = int(0.3*prec**0.35)
        x2 = a = (x*x) >> wp
        xpowers = [one, x2]
        for i in xrange(1, u):
            xpowers.append((xpowers[-1]*x2)>>wp)
        sums = [MPZ_ZERO] * u
        k = 2
        while a:
            for i in xrange(u):
                a //= (k-1)*k
                if alt and k & 2: sums[i] -= a
                else:             sums[i] += a
                k += 2
            a = (a*xpowers[-1]) >> wp
        for i in xrange(1, u):
            sums[i] = (sums[i]*xpowers[i]) >> wp
        c = sum(sums) + one
    if type == 0:
        s = isqrt_fast(c*c - (one<<wp))
        if sign:
            v = c - s
        else:
            v = c + s
        for i in xrange(r):
            v = (v*v) >> wp
        return v >> extra
    else:
        # Repeatedly apply the double-angle formula
        # cosh(2*x) = 2*cosh(x)^2 - 1
        # cos(2*x) = 2*cos(x)^2 - 1
        pshift = wp-1
        for i in xrange(r):
            c = ((c*c) >> pshift) - one
        # With the abs, this is the same for sinh and sin
        s = isqrt_fast(abs((one<<wp) - c*c))
        if sign:
            s = -s
        return (c>>extra), (s>>extra)

def exp_basecase(x, prec):
    """
    Compute exp(x) as a fixed-point number. Works for any x,
    but for speed should have |x| < 1. For an arbitrary number,
    use exp(x) = exp(x-m*log(2)) * 2^m where m = floor(x/log(2)).
    """
    if prec > EXP_COSH_CUTOFF:
        return exponential_series(x, prec, 0)
    r = int(prec**0.5)
    prec += r
    s0 = s1 = (MPZ_ONE << prec)
    k = 2
    a = x2 = (x*x) >> prec
    while a:
        a //= k; s0 += a; k += 1
        a //= k; s1 += a; k += 1
        a = (a*x2) >> prec
    s1 = (s1*x) >> prec
    s = s0 + s1
    u = r
    while r:
        s = (s*s) >> prec
        r -= 1
    return s >> u

def exp_expneg_basecase(x, prec):
    """
    Computation of exp(x), exp(-x)
    """
    if prec > EXP_COSH_CUTOFF:
        cosh, sinh = exponential_series(x, prec, 1)
        return cosh+sinh, cosh-sinh
    a = exp_basecase(x, prec)
    b = (MPZ_ONE << (prec+prec)) // a
    return a, b

def cos_sin_basecase(x, prec):
    """
    Compute cos(x), sin(x) as fixed-point numbers, assuming x
    in [0, pi/2). For an arbitrary number, use x' = x - m*(pi/2)
    where m = floor(x/(pi/2)) along with quarter-period symmetries.
    """
    if prec > COS_SIN_CACHE_PREC:
        return exponential_series(x, prec, 2)
    precs = prec - COS_SIN_CACHE_STEP
    t = x >> precs
    n = int(t)
    if n not in cos_sin_cache:
        w = t<<(10+COS_SIN_CACHE_PREC-COS_SIN_CACHE_STEP)
        cos_t, sin_t = exponential_series(w, 10+COS_SIN_CACHE_PREC, 2)
        cos_sin_cache[n] = (cos_t>>10), (sin_t>>10)
    cos_t, sin_t = cos_sin_cache[n]
    offset = COS_SIN_CACHE_PREC - prec
    cos_t >>= offset
    sin_t >>= offset
    x -= t << precs
    cos = MPZ_ONE << prec
    sin = x
    k = 2
    a = -((x*x) >> prec)
    while a:
        a //= k; cos += a; k += 1; a = (a*x) >> prec
        a //= k; sin += a; k += 1; a = -((a*x) >> prec)
    return ((cos*cos_t-sin*sin_t) >> prec), ((sin*cos_t+cos*sin_t) >> prec)

def mpf_exp(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if man:
        mag = bc + exp
        wp = prec + 14
        if sign:
            man = -man
        # TODO: the best cutoff depends on both x and the precision.
        if prec > 600 and exp >= 0:
            # Need about log2(exp(n)) ~= 1.45*mag extra precision
            e = mpf_e(wp+int(1.45*mag))
            return mpf_pow_int(e, man<<exp, prec, rnd)
        if mag < -wp:
            return mpf_perturb(fone, sign, prec, rnd)
        # |x| >= 2
        if mag > 1:
            # For large arguments: exp(2^mag*(1+eps)) =
            # exp(2^mag)*exp(2^mag*eps) = exp(2^mag)*(1 + 2^mag*eps + ...)
            # so about mag extra bits is required.
            wpmod = wp + mag
            offset = exp + wpmod
            if offset >= 0:
                t = man << offset
            else:
                t = man >> (-offset)
            lg2 = ln2_fixed(wpmod)
            n, t = divmod(t, lg2)
            n = int(n)
            t >>= mag
        else:
            offset = exp + wp
            if offset >= 0:
                t = man << offset
            else:
                t = man >> (-offset)
            n = 0
        man = exp_basecase(t, wp)
        return from_man_exp(man, n-wp, prec, rnd)
    if not exp:
        return fone
    if x == fninf:
        return fzero
    return x


def mpf_cosh_sinh(x, prec, rnd=round_fast, tanh=0):
    """Simultaneously compute (cosh(x), sinh(x)) for real x"""
    sign, man, exp, bc = x
    if (not man) and exp:
        if tanh:
            if x == finf: return fone
            if x == fninf: return fnone
            return fnan
        if x == finf: return (finf, finf)
        if x == fninf: return (finf, fninf)
        return fnan, fnan
    mag = exp+bc
    wp = prec+14
    if mag < -4:
        # Extremely close to 0, sinh(x) ~= x and cosh(x) ~= 1
        if mag < -wp:
            if tanh:
                return mpf_perturb(x, 1-sign, prec, rnd)
            cosh = mpf_perturb(fone, 0, prec, rnd)
            sinh = mpf_perturb(x, sign, prec, rnd)
            return cosh, sinh
        # Fix for cancellation when computing sinh
        wp += (-mag)
    # Does exp(-2*x) vanish?
    if mag > 10:
        if 3*(1<<(mag-1)) > wp:
            # XXX: rounding
            if tanh:
                return mpf_perturb([fone,fnone][sign], 1-sign, prec, rnd)
            c = s = mpf_shift(mpf_exp(mpf_abs(x), prec, rnd), -1)
            if sign:
                s = mpf_neg(s)
            return c, s
    # |x| > 1
    if mag > 1:
        wpmod = wp + mag
        offset = exp + wpmod
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        lg2 = ln2_fixed(wpmod)
        n, t = divmod(t, lg2)
        n = int(n)
        t >>= mag
    else:
        offset = exp + wp
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        n = 0
    a, b = exp_expneg_basecase(t, wp)
    # TODO: optimize division precision
    cosh = a + (b>>(2*n))
    sinh = a - (b>>(2*n))
    if sign:
        sinh = -sinh
    if tanh:
        man = (sinh << wp) // cosh
        return from_man_exp(man, -wp, prec, rnd)
    else:
        cosh = from_man_exp(cosh, n-wp-1, prec, rnd)
        sinh = from_man_exp(sinh, n-wp-1, prec, rnd)
        return cosh, sinh


def mod_pi2(man, exp, mag, wp):
    # Reduce to standard interval
    if mag > 0:
        i = 0
        while 1:
            cancellation_prec = 20 << i
            wpmod = wp + mag + cancellation_prec
            pi2 = pi_fixed(wpmod-1)
            pi4 = pi2 >> 1
            offset = wpmod + exp
            if offset >= 0:
                t = man << offset
            else:
                t = man >> (-offset)
            n, y = divmod(t, pi2)
            if y > pi4:
                small = pi2 - y
            else:
                small = y
            if small >> (wp+mag-10):
                n = int(n)
                t = y >> mag
                wp = wpmod - mag
                break
            i += 1
    else:
        wp += (-mag)
        offset = exp + wp
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        n = 0
    return t, n, wp


def mpf_cos_sin(x, prec, rnd=round_fast, which=0, pi=False):
    """
    which:
    0 -- return cos(x), sin(x)
    1 -- return cos(x)
    2 -- return sin(x)
    3 -- return tan(x)

    if pi=True, compute for pi*x
    """
    sign, man, exp, bc = x
    if not man:
        if exp:
            c, s = fnan, fnan
        else:
            c, s = fone, fzero
        if which == 0: return c, s
        if which == 1: return c
        if which == 2: return s
        if which == 3: return s

    mag = bc + exp
    wp = prec + 10

    # Extremely small?
    if mag < 0:
        if mag < -wp:
            if pi:
                x = mpf_mul(x, mpf_pi(wp))
            c = mpf_perturb(fone, 1, prec, rnd)
            s = mpf_perturb(x, 1-sign, prec, rnd)
            if which == 0: return c, s
            if which == 1: return c
            if which == 2: return s
            if which == 3: return mpf_perturb(x, sign, prec, rnd)
    if pi:
        if exp >= -1:
            if exp == -1:
                c = fzero
                s = (fone, fnone)[bool(man & 2) ^ sign]
            elif exp == 0:
                c, s = (fnone, fzero)
            else:
                c, s = (fone, fzero)
            if which == 0: return c, s
            if which == 1: return c
            if which == 2: return s
            if which == 3: return mpf_div(s, c, prec, rnd)
        # Subtract nearest half-integer (= mod by pi/2)
        n = ((man >> (-exp-2)) + 1) >> 1
        man = man - (n << (-exp-1))
        mag2 = bitcount(man) + exp
        wp = prec + 10 - mag2
        offset = exp + wp
        if offset >= 0:
            t = man << offset
        else:
            t = man >> (-offset)
        t = (t*pi_fixed(wp)) >> wp
    else:
        t, n, wp = mod_pi2(man, exp, mag, wp)
    c, s = cos_sin_basecase(t, wp)
    m = n & 3
    if   m == 1: c, s = -s, c
    elif m == 2: c, s = -c, -s
    elif m == 3: c, s = s, -c
    if sign:
        s = -s
    if which == 0:
        c = from_man_exp(c, -wp, prec, rnd)
        s = from_man_exp(s, -wp, prec, rnd)
        return c, s
    if which == 1:
        return from_man_exp(c, -wp, prec, rnd)
    if which == 2:
        return from_man_exp(s, -wp, prec, rnd)
    if which == 3:
        return from_rational(s, c, prec, rnd)

def mpf_cos(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 1)
def mpf_sin(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 2)
def mpf_tan(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 3)
def mpf_cos_sin_pi(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 0, 1)
def mpf_cos_pi(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 1, 1)
def mpf_sin_pi(x, prec, rnd=round_fast): return mpf_cos_sin(x, prec, rnd, 2, 1)
def mpf_cosh(x, prec, rnd=round_fast): return mpf_cosh_sinh(x, prec, rnd)[0]
def mpf_sinh(x, prec, rnd=round_fast): return mpf_cosh_sinh(x, prec, rnd)[1]
def mpf_tanh(x, prec, rnd=round_fast): return mpf_cosh_sinh(x, prec, rnd, tanh=1)


# Low-overhead fixed-point versions

def cos_sin_fixed(x, prec, pi2=None):
    if pi2 is None:
        pi2 = pi_fixed(prec-1)
    n, t = divmod(x, pi2)
    n = int(n)
    c, s = cos_sin_basecase(t, prec)
    m = n & 3
    if m == 0: return c, s
    if m == 1: return -s, c
    if m == 2: return -c, -s
    if m == 3: return s, -c

def exp_fixed(x, prec, ln2=None):
    if ln2 is None:
        ln2 = ln2_fixed(prec)
    n, t = divmod(x, ln2)
    n = int(n)
    v = exp_basecase(t, prec)
    if n >= 0:
        return v << n
    else:
        return v >> (-n)


if BACKEND == 'sage':
    try:
        import sage.libs.mpmath.ext_libmp as _lbmp
        mpf_sqrt = _lbmp.mpf_sqrt
        mpf_exp = _lbmp.mpf_exp
        mpf_log = _lbmp.mpf_log
        mpf_cos = _lbmp.mpf_cos
        mpf_sin = _lbmp.mpf_sin
        mpf_pow = _lbmp.mpf_pow
        exp_fixed = _lbmp.exp_fixed
        cos_sin_fixed = _lbmp.cos_sin_fixed
        log_int_fixed = _lbmp.log_int_fixed
    except (ImportError, AttributeError):
        print("Warning: Sage imports in libelefun failed")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\defchararray.py ===
"""
This module contains a set of functions for vectorized string
operations and methods.

.. note::
   The `chararray` class exists for backwards compatibility with
   Numarray, it is not recommended for new development. Starting from numpy
   1.4, if one needs arrays of strings, it is recommended to use arrays of
   `dtype` `object_`, `bytes_` or `str_`, and use the free functions
   in the `numpy.char` module for fast vectorized string operations.

Some methods will only be available if the corresponding string method is
available in your version of Python.

The preferred alias for `defchararray` is `numpy.char`.

"""
import functools

import numpy as np
from numpy._core import overrides
from numpy._core.multiarray import compare_chararrays
from numpy._core.strings import (
    _join as join,
)
from numpy._core.strings import (
    _rsplit as rsplit,
)
from numpy._core.strings import (
    _split as split,
)
from numpy._core.strings import (
    _splitlines as splitlines,
)
from numpy._utils import set_module
from numpy.strings import *
from numpy.strings import (
    multiply as strings_multiply,
)
from numpy.strings import (
    partition as strings_partition,
)
from numpy.strings import (
    rpartition as strings_rpartition,
)

from .numeric import array as narray
from .numeric import asarray as asnarray
from .numeric import ndarray
from .numerictypes import bytes_, character, str_

__all__ = [
    'equal', 'not_equal', 'greater_equal', 'less_equal',
    'greater', 'less', 'str_len', 'add', 'multiply', 'mod', 'capitalize',
    'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs',
    'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace',
    'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition',
    'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
    'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase',
    'title', 'translate', 'upper', 'zfill', 'isnumeric', 'isdecimal',
    'array', 'asarray', 'compare_chararrays', 'chararray'
    ]


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.char')


def _binary_op_dispatcher(x1, x2):
    return (x1, x2)


@array_function_dispatch(_binary_op_dispatcher)
def equal(x1, x2):
    """
    Return (x1 == x2) element-wise.

    Unlike `numpy.equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    Examples
    --------
    >>> import numpy as np
    >>> y = "aa "
    >>> x = "aa"
    >>> np.char.equal(x, y)
    array(True)

    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less
    """
    return compare_chararrays(x1, x2, '==', True)


@array_function_dispatch(_binary_op_dispatcher)
def not_equal(x1, x2):
    """
    Return (x1 != x2) element-wise.

    Unlike `numpy.not_equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, greater_equal, less_equal, greater, less

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.not_equal(x1, 'b')
    array([ True, False,  True])

    """
    return compare_chararrays(x1, x2, '!=', True)


@array_function_dispatch(_binary_op_dispatcher)
def greater_equal(x1, x2):
    """
    Return (x1 >= x2) element-wise.

    Unlike `numpy.greater_equal`, this comparison is performed by
    first stripping whitespace characters from the end of the string.
    This behavior is provided for backward-compatibility with
    numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, less_equal, greater, less

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.greater_equal(x1, 'b')
    array([False,  True,  True])

    """
    return compare_chararrays(x1, x2, '>=', True)


@array_function_dispatch(_binary_op_dispatcher)
def less_equal(x1, x2):
    """
    Return (x1 <= x2) element-wise.

    Unlike `numpy.less_equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, greater, less

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.less_equal(x1, 'b')
    array([ True,  True, False])

    """
    return compare_chararrays(x1, x2, '<=', True)


@array_function_dispatch(_binary_op_dispatcher)
def greater(x1, x2):
    """
    Return (x1 > x2) element-wise.

    Unlike `numpy.greater`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, less_equal, less

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.greater(x1, 'b')
    array([False, False,  True])

    """
    return compare_chararrays(x1, x2, '>', True)


@array_function_dispatch(_binary_op_dispatcher)
def less(x1, x2):
    """
    Return (x1 < x2) element-wise.

    Unlike `numpy.greater`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, less_equal, greater

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.less(x1, 'b')
    array([True, False, False])

    """
    return compare_chararrays(x1, x2, '<', True)


@set_module("numpy.char")
def multiply(a, i):
    """
    Return (a * i), that is string multiple concatenation,
    element-wise.

    Values in ``i`` of less than 0 are treated as 0 (which yields an
    empty string).

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype

    i : array_like, with any integer dtype

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input types

    Notes
    -----
    This is a thin wrapper around np.strings.multiply that raises
    `ValueError` when ``i`` is not an integer. It only
    exists for backwards-compatibility.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["a", "b", "c"])
    >>> np.strings.multiply(a, 3)
    array(['aaa', 'bbb', 'ccc'], dtype='<U3')
    >>> i = np.array([1, 2, 3])
    >>> np.strings.multiply(a, i)
    array(['a', 'bb', 'ccc'], dtype='<U3')
    >>> np.strings.multiply(np.array(['a']), i)
    array(['a', 'aa', 'aaa'], dtype='<U3')
    >>> a = np.array(['a', 'b', 'c', 'd', 'e', 'f']).reshape((2, 3))
    >>> np.strings.multiply(a, 3)
    array([['aaa', 'bbb', 'ccc'],
           ['ddd', 'eee', 'fff']], dtype='<U3')
    >>> np.strings.multiply(a, i)
    array([['a', 'bb', 'ccc'],
           ['d', 'ee', 'fff']], dtype='<U3')

    """
    try:
        return strings_multiply(a, i)
    except TypeError:
        raise ValueError("Can only multiply by integers")


@set_module("numpy.char")
def partition(a, sep):
    """
    Partition each element in `a` around `sep`.

    Calls :meth:`str.partition` element-wise.

    For each element in `a`, split the element as the first
    occurrence of `sep`, and return 3 strings containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, return 3 strings
    containing the string itself, followed by two empty strings.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    sep : {str, unicode}
        Separator to split each string element in `a`.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types. The output array will have an extra
        dimension with 3 elements per input element.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array(["Numpy is nice!"])
    >>> np.char.partition(x, " ")
    array([['Numpy', ' ', 'is nice!']], dtype='<U8')

    See Also
    --------
    str.partition

    """
    return np.stack(strings_partition(a, sep), axis=-1)


@set_module("numpy.char")
def rpartition(a, sep):
    """
    Partition (split) each element around the right-most separator.

    Calls :meth:`str.rpartition` element-wise.

    For each element in `a`, split the element as the last
    occurrence of `sep`, and return 3 strings containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, return 3 strings
    containing the string itself, followed by two empty strings.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    sep : str or unicode
        Right-most separator to split each element in array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types. The output array will have an extra
        dimension with 3 elements per input element.

    See Also
    --------
    str.rpartition

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.char.rpartition(a, 'A')
    array([['aAaAa', 'A', ''],
       ['  a', 'A', '  '],
       ['abB', 'A', 'Bba']], dtype='<U5')

    """
    return np.stack(strings_rpartition(a, sep), axis=-1)


@set_module("numpy.char")
class chararray(ndarray):
    """
    chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
              strides=None, order=None)

    Provides a convenient view on arrays of string and unicode values.

    .. note::
       The `chararray` class exists for backwards compatibility with
       Numarray, it is not recommended for new development. Starting from numpy
       1.4, if one needs arrays of strings, it is recommended to use arrays of
       `dtype` `~numpy.object_`, `~numpy.bytes_` or `~numpy.str_`, and use
       the free functions in the `numpy.char` module for fast vectorized
       string operations.

    Versus a NumPy array of dtype `~numpy.bytes_` or `~numpy.str_`, this
    class adds the following functionality:

    1) values automatically have whitespace removed from the end
       when indexed

    2) comparison operators automatically remove whitespace from the
       end when comparing values

    3) vectorized string operations are provided as methods
       (e.g. `.endswith`) and infix operators (e.g. ``"+", "*", "%"``)

    chararrays should be created using `numpy.char.array` or
    `numpy.char.asarray`, rather than this constructor directly.

    This constructor creates the array, using `buffer` (with `offset`
    and `strides`) if it is not ``None``. If `buffer` is ``None``, then
    constructs a new array with `strides` in "C order", unless both
    ``len(shape) >= 2`` and ``order='F'``, in which case `strides`
    is in "Fortran order".

    Methods
    -------
    astype
    argsort
    copy
    count
    decode
    dump
    dumps
    encode
    endswith
    expandtabs
    fill
    find
    flatten
    getfield
    index
    isalnum
    isalpha
    isdecimal
    isdigit
    islower
    isnumeric
    isspace
    istitle
    isupper
    item
    join
    ljust
    lower
    lstrip
    nonzero
    put
    ravel
    repeat
    replace
    reshape
    resize
    rfind
    rindex
    rjust
    rsplit
    rstrip
    searchsorted
    setfield
    setflags
    sort
    split
    splitlines
    squeeze
    startswith
    strip
    swapaxes
    swapcase
    take
    title
    tofile
    tolist
    tostring
    translate
    transpose
    upper
    view
    zfill

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    itemsize : int, optional
        Length of each array element, in number of characters. Default is 1.
    unicode : bool, optional
        Are the array elements of type unicode (True) or string (False).
        Default is False.
    buffer : object exposing the buffer interface or str, optional
        Memory address of the start of the array data.  Default is None,
        in which case a new array is created.
    offset : int, optional
        Fixed stride displacement from the beginning of an axis?
        Default is 0. Needs to be >=0.
    strides : array_like of ints, optional
        Strides for the array (see `~numpy.ndarray.strides` for
        full description). Default is None.
    order : {'C', 'F'}, optional
        The order in which the array data is stored in memory: 'C' ->
        "row major" order (the default), 'F' -> "column major"
        (Fortran) order.

    Examples
    --------
    >>> import numpy as np
    >>> charar = np.char.chararray((3, 3))
    >>> charar[:] = 'a'
    >>> charar
    chararray([[b'a', b'a', b'a'],
               [b'a', b'a', b'a'],
               [b'a', b'a', b'a']], dtype='|S1')

    >>> charar = np.char.chararray(charar.shape, itemsize=5)
    >>> charar[:] = 'abc'
    >>> charar
    chararray([[b'abc', b'abc', b'abc'],
               [b'abc', b'abc', b'abc'],
               [b'abc', b'abc', b'abc']], dtype='|S5')

    """
    def __new__(subtype, shape, itemsize=1, unicode=False, buffer=None,
                offset=0, strides=None, order='C'):
        if unicode:
            dtype = str_
        else:
            dtype = bytes_

        # force itemsize to be a Python int, since using NumPy integer
        # types results in itemsize.itemsize being used as the size of
        # strings in the new array.
        itemsize = int(itemsize)

        if isinstance(buffer, str):
            # unicode objects do not have the buffer interface
            filler = buffer
            buffer = None
        else:
            filler = None

        if buffer is None:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   order=order)
        else:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   buffer=buffer,
                                   offset=offset, strides=strides,
                                   order=order)
        if filler is not None:
            self[...] = filler

        return self

    def __array_wrap__(self, arr, context=None, return_scalar=False):
        # When calling a ufunc (and some other functions), we return a
        # chararray if the ufunc output is a string-like array,
        # or an ndarray otherwise
        if arr.dtype.char in "SUbc":
            return arr.view(type(self))
        return arr

    def __array_finalize__(self, obj):
        # The b is a special case because it is used for reconstructing.
        if self.dtype.char not in 'VSUbc':
            raise ValueError("Can only create a chararray from string data.")

    def __getitem__(self, obj):
        val = ndarray.__getitem__(self, obj)
        if isinstance(val, character):
            return val.rstrip()
        return val

    # IMPLEMENTATION NOTE: Most of the methods of this class are
    # direct delegations to the free functions in this module.
    # However, those that return an array of strings should instead
    # return a chararray, so some extra wrapping is required.

    def __eq__(self, other):
        """
        Return (self == other) element-wise.

        See Also
        --------
        equal
        """
        return equal(self, other)

    def __ne__(self, other):
        """
        Return (self != other) element-wise.

        See Also
        --------
        not_equal
        """
        return not_equal(self, other)

    def __ge__(self, other):
        """
        Return (self >= other) element-wise.

        See Also
        --------
        greater_equal
        """
        return greater_equal(self, other)

    def __le__(self, other):
        """
        Return (self <= other) element-wise.

        See Also
        --------
        less_equal
        """
        return less_equal(self, other)

    def __gt__(self, other):
        """
        Return (self > other) element-wise.

        See Also
        --------
        greater
        """
        return greater(self, other)

    def __lt__(self, other):
        """
        Return (self < other) element-wise.

        See Also
        --------
        less
        """
        return less(self, other)

    def __add__(self, other):
        """
        Return (self + other), that is string concatenation,
        element-wise for a pair of array_likes of str or unicode.

        See Also
        --------
        add
        """
        return add(self, other)

    def __radd__(self, other):
        """
        Return (other + self), that is string concatenation,
        element-wise for a pair of array_likes of `bytes_` or `str_`.

        See Also
        --------
        add
        """
        return add(other, self)

    def __mul__(self, i):
        """
        Return (self * i), that is string multiple concatenation,
        element-wise.

        See Also
        --------
        multiply
        """
        return asarray(multiply(self, i))

    def __rmul__(self, i):
        """
        Return (self * i), that is string multiple concatenation,
        element-wise.

        See Also
        --------
        multiply
        """
        return asarray(multiply(self, i))

    def __mod__(self, i):
        """
        Return (self % i), that is pre-Python 2.6 string formatting
        (interpolation), element-wise for a pair of array_likes of `bytes_`
        or `str_`.

        See Also
        --------
        mod
        """
        return asarray(mod(self, i))

    def __rmod__(self, other):
        return NotImplemented

    def argsort(self, axis=-1, kind=None, order=None):
        """
        Return the indices that sort the array lexicographically.

        For full documentation see `numpy.argsort`, for which this method is
        in fact merely a "thin wrapper."

        Examples
        --------
        >>> c = np.array(['a1b c', '1b ca', 'b ca1', 'Ca1b'], 'S5')
        >>> c = c.view(np.char.chararray); c
        chararray(['a1b c', '1b ca', 'b ca1', 'Ca1b'],
              dtype='|S5')
        >>> c[c.argsort()]
        chararray(['1b ca', 'Ca1b', 'a1b c', 'b ca1'],
              dtype='|S5')

        """
        return self.__array__().argsort(axis, kind, order)
    argsort.__doc__ = ndarray.argsort.__doc__

    def capitalize(self):
        """
        Return a copy of `self` with only the first character of each element
        capitalized.

        See Also
        --------
        char.capitalize

        """
        return asarray(capitalize(self))

    def center(self, width, fillchar=' '):
        """
        Return a copy of `self` with its elements centered in a
        string of length `width`.

        See Also
        --------
        center
        """
        return asarray(center(self, width, fillchar))

    def count(self, sub, start=0, end=None):
        """
        Returns an array with the number of non-overlapping occurrences of
        substring `sub` in the range [`start`, `end`].

        See Also
        --------
        char.count

        """
        return count(self, sub, start, end)

    def decode(self, encoding=None, errors=None):
        """
        Calls ``bytes.decode`` element-wise.

        See Also
        --------
        char.decode

        """
        return decode(self, encoding, errors)

    def encode(self, encoding=None, errors=None):
        """
        Calls :meth:`str.encode` element-wise.

        See Also
        --------
        char.encode

        """
        return encode(self, encoding, errors)

    def endswith(self, suffix, start=0, end=None):
        """
        Returns a boolean array which is `True` where the string element
        in `self` ends with `suffix`, otherwise `False`.

        See Also
        --------
        char.endswith

        """
        return endswith(self, suffix, start, end)

    def expandtabs(self, tabsize=8):
        """
        Return a copy of each string element where all tab characters are
        replaced by one or more spaces.

        See Also
        --------
        char.expandtabs

        """
        return asarray(expandtabs(self, tabsize))

    def find(self, sub, start=0, end=None):
        """
        For each element, return the lowest index in the string where
        substring `sub` is found.

        See Also
        --------
        char.find

        """
        return find(self, sub, start, end)

    def index(self, sub, start=0, end=None):
        """
        Like `find`, but raises :exc:`ValueError` when the substring is not
        found.

        See Also
        --------
        char.index

        """
        return index(self, sub, start, end)

    def isalnum(self):
        """
        Returns true for each element if all characters in the string
        are alphanumeric and there is at least one character, false
        otherwise.

        See Also
        --------
        char.isalnum

        """
        return isalnum(self)

    def isalpha(self):
        """
        Returns true for each element if all characters in the string
        are alphabetic and there is at least one character, false
        otherwise.

        See Also
        --------
        char.isalpha

        """
        return isalpha(self)

    def isdigit(self):
        """
        Returns true for each element if all characters in the string are
        digits and there is at least one character, false otherwise.

        See Also
        --------
        char.isdigit

        """
        return isdigit(self)

    def islower(self):
        """
        Returns true for each element if all cased characters in the
        string are lowercase and there is at least one cased character,
        false otherwise.

        See Also
        --------
        char.islower

        """
        return islower(self)

    def isspace(self):
        """
        Returns true for each element if there are only whitespace
        characters in the string and there is at least one character,
        false otherwise.

        See Also
        --------
        char.isspace

        """
        return isspace(self)

    def istitle(self):
        """
        Returns true for each element if the element is a titlecased
        string and there is at least one character, false otherwise.

        See Also
        --------
        char.istitle

        """
        return istitle(self)

    def isupper(self):
        """
        Returns true for each element if all cased characters in the
        string are uppercase and there is at least one character, false
        otherwise.

        See Also
        --------
        char.isupper

        """
        return isupper(self)

    def join(self, seq):
        """
        Return a string which is the concatenation of the strings in the
        sequence `seq`.

        See Also
        --------
        char.join

        """
        return join(self, seq)

    def ljust(self, width, fillchar=' '):
        """
        Return an array with the elements of `self` left-justified in a
        string of length `width`.

        See Also
        --------
        char.ljust

        """
        return asarray(ljust(self, width, fillchar))

    def lower(self):
        """
        Return an array with the elements of `self` converted to
        lowercase.

        See Also
        --------
        char.lower

        """
        return asarray(lower(self))

    def lstrip(self, chars=None):
        """
        For each element in `self`, return a copy with the leading characters
        removed.

        See Also
        --------
        char.lstrip

        """
        return lstrip(self, chars)

    def partition(self, sep):
        """
        Partition each element in `self` around `sep`.

        See Also
        --------
        partition
        """
        return asarray(partition(self, sep))

    def replace(self, old, new, count=None):
        """
        For each element in `self`, return a copy of the string with all
        occurrences of substring `old` replaced by `new`.

        See Also
        --------
        char.replace

        """
        return replace(self, old, new, count if count is not None else -1)

    def rfind(self, sub, start=0, end=None):
        """
        For each element in `self`, return the highest index in the string
        where substring `sub` is found, such that `sub` is contained
        within [`start`, `end`].

        See Also
        --------
        char.rfind

        """
        return rfind(self, sub, start, end)

    def rindex(self, sub, start=0, end=None):
        """
        Like `rfind`, but raises :exc:`ValueError` when the substring `sub` is
        not found.

        See Also
        --------
        char.rindex

        """
        return rindex(self, sub, start, end)

    def rjust(self, width, fillchar=' '):
        """
        Return an array with the elements of `self`
        right-justified in a string of length `width`.

        See Also
        --------
        char.rjust

        """
        return asarray(rjust(self, width, fillchar))

    def rpartition(self, sep):
        """
        Partition each element in `self` around `sep`.

        See Also
        --------
        rpartition
        """
        return asarray(rpartition(self, sep))

    def rsplit(self, sep=None, maxsplit=None):
        """
        For each element in `self`, return a list of the words in
        the string, using `sep` as the delimiter string.

        See Also
        --------
        char.rsplit

        """
        return rsplit(self, sep, maxsplit)

    def rstrip(self, chars=None):
        """
        For each element in `self`, return a copy with the trailing
        characters removed.

        See Also
        --------
        char.rstrip

        """
        return rstrip(self, chars)

    def split(self, sep=None, maxsplit=None):
        """
        For each element in `self`, return a list of the words in the
        string, using `sep` as the delimiter string.

        See Also
        --------
        char.split

        """
        return split(self, sep, maxsplit)

    def splitlines(self, keepends=None):
        """
        For each element in `self`, return a list of the lines in the
        element, breaking at line boundaries.

        See Also
        --------
        char.splitlines

        """
        return splitlines(self, keepends)

    def startswith(self, prefix, start=0, end=None):
        """
        Returns a boolean array which is `True` where the string element
        in `self` starts with `prefix`, otherwise `False`.

        See Also
        --------
        char.startswith

        """
        return startswith(self, prefix, start, end)

    def strip(self, chars=None):
        """
        For each element in `self`, return a copy with the leading and
        trailing characters removed.

        See Also
        --------
        char.strip

        """
        return strip(self, chars)

    def swapcase(self):
        """
        For each element in `self`, return a copy of the string with
        uppercase characters converted to lowercase and vice versa.

        See Also
        --------
        char.swapcase

        """
        return asarray(swapcase(self))

    def title(self):
        """
        For each element in `self`, return a titlecased version of the
        string: words start with uppercase characters, all remaining cased
        characters are lowercase.

        See Also
        --------
        char.title

        """
        return asarray(title(self))

    def translate(self, table, deletechars=None):
        """
        For each element in `self`, return a copy of the string where
        all characters occurring in the optional argument
        `deletechars` are removed, and the remaining characters have
        been mapped through the given translation table.

        See Also
        --------
        char.translate

        """
        return asarray(translate(self, table, deletechars))

    def upper(self):
        """
        Return an array with the elements of `self` converted to
        uppercase.

        See Also
        --------
        char.upper

        """
        return asarray(upper(self))

    def zfill(self, width):
        """
        Return the numeric string left-filled with zeros in a string of
        length `width`.

        See Also
        --------
        char.zfill

        """
        return asarray(zfill(self, width))

    def isnumeric(self):
        """
        For each element in `self`, return True if there are only
        numeric characters in the element.

        See Also
        --------
        char.isnumeric

        """
        return isnumeric(self)

    def isdecimal(self):
        """
        For each element in `self`, return True if there are only
        decimal characters in the element.

        See Also
        --------
        char.isdecimal

        """
        return isdecimal(self)


@set_module("numpy.char")
def array(obj, itemsize=None, copy=True, unicode=None, order=None):
    """
    Create a `~numpy.char.chararray`.

    .. note::
       This class is provided for numarray backward-compatibility.
       New code (not concerned with numarray compatibility) should use
       arrays of type `bytes_` or `str_` and use the free functions
       in :mod:`numpy.char` for fast vectorized string operations instead.

    Versus a NumPy array of dtype `bytes_` or `str_`, this
    class adds the following functionality:

    1) values automatically have whitespace removed from the end
       when indexed

    2) comparison operators automatically remove whitespace from the
       end when comparing values

    3) vectorized string operations are provided as methods
       (e.g. `chararray.endswith <numpy.char.chararray.endswith>`)
       and infix operators (e.g. ``+, *, %``)

    Parameters
    ----------
    obj : array of str or unicode-like

    itemsize : int, optional
        `itemsize` is the number of characters per scalar in the
        resulting array.  If `itemsize` is None, and `obj` is an
        object array or a Python list, the `itemsize` will be
        automatically determined.  If `itemsize` is provided and `obj`
        is of type str or unicode, then the `obj` string will be
        chunked into `itemsize` pieces.

    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy
        will only be made if ``__array__`` returns a copy, if obj is a
        nested sequence, or if a copy is needed to satisfy any of the other
        requirements (`itemsize`, unicode, `order`, etc.).

    unicode : bool, optional
        When true, the resulting `~numpy.char.chararray` can contain Unicode
        characters, when false only 8-bit characters.  If unicode is
        None and `obj` is one of the following:

        - a `~numpy.char.chararray`,
        - an ndarray of type :class:`str_` or :class:`bytes_`
        - a Python :class:`str` or :class:`bytes` object,

        then the unicode setting of the output array will be
        automatically determined.

    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).  If order is 'A', then the returned array may
        be in any order (either C-, Fortran-contiguous, or even
        discontiguous).

    Examples
    --------

    >>> import numpy as np
    >>> char_array = np.char.array(['hello', 'world', 'numpy','array'])
    >>> char_array
    chararray(['hello', 'world', 'numpy', 'array'], dtype='<U5')

    """
    if isinstance(obj, (bytes, str)):
        if unicode is None:
            if isinstance(obj, str):
                unicode = True
            else:
                unicode = False

        if itemsize is None:
            itemsize = len(obj)
        shape = len(obj) // itemsize

        return chararray(shape, itemsize=itemsize, unicode=unicode,
                         buffer=obj, order=order)

    if isinstance(obj, (list, tuple)):
        obj = asnarray(obj)

    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, character):
        # If we just have a vanilla chararray, create a chararray
        # view around it.
        if not isinstance(obj, chararray):
            obj = obj.view(chararray)

        if itemsize is None:
            itemsize = obj.itemsize
            # itemsize is in 8-bit chars, so for Unicode, we need
            # to divide by the size of a single Unicode character,
            # which for NumPy is always 4
            if issubclass(obj.dtype.type, str_):
                itemsize //= 4

        if unicode is None:
            if issubclass(obj.dtype.type, str_):
                unicode = True
            else:
                unicode = False

        if unicode:
            dtype = str_
        else:
            dtype = bytes_

        if order is not None:
            obj = asnarray(obj, order=order)
        if (copy or
                (itemsize != obj.itemsize) or
                (not unicode and isinstance(obj, str_)) or
                (unicode and isinstance(obj, bytes_))):
            obj = obj.astype((dtype, int(itemsize)))
        return obj

    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, object):
        if itemsize is None:
            # Since no itemsize was specified, convert the input array to
            # a list so the ndarray constructor will automatically
            # determine the itemsize for us.
            obj = obj.tolist()
            # Fall through to the default case

    if unicode:
        dtype = str_
    else:
        dtype = bytes_

    if itemsize is None:
        val = narray(obj, dtype=dtype, order=order, subok=True)
    else:
        val = narray(obj, dtype=(dtype, itemsize), order=order, subok=True)
    return val.view(chararray)


@set_module("numpy.char")
def asarray(obj, itemsize=None, unicode=None, order=None):
    """
    Convert the input to a `~numpy.char.chararray`, copying the data only if
    necessary.

    Versus a NumPy array of dtype `bytes_` or `str_`, this
    class adds the following functionality:

    1) values automatically have whitespace removed from the end
       when indexed

    2) comparison operators automatically remove whitespace from the
       end when comparing values

    3) vectorized string operations are provided as methods
       (e.g. `chararray.endswith <numpy.char.chararray.endswith>`)
       and infix operators (e.g. ``+``, ``*``, ``%``)

    Parameters
    ----------
    obj : array of str or unicode-like

    itemsize : int, optional
        `itemsize` is the number of characters per scalar in the
        resulting array.  If `itemsize` is None, and `obj` is an
        object array or a Python list, the `itemsize` will be
        automatically determined.  If `itemsize` is provided and `obj`
        is of type str or unicode, then the `obj` string will be
        chunked into `itemsize` pieces.

    unicode : bool, optional
        When true, the resulting `~numpy.char.chararray` can contain Unicode
        characters, when false only 8-bit characters.  If unicode is
        None and `obj` is one of the following:

        - a `~numpy.char.chararray`,
        - an ndarray of type `str_` or `unicode_`
        - a Python str or unicode object,

        then the unicode setting of the output array will be
        automatically determined.

    order : {'C', 'F'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).

    Examples
    --------
    >>> import numpy as np
    >>> np.char.asarray(['hello', 'world'])
    chararray(['hello', 'world'], dtype='<U5')

    """
    return array(obj, itemsize, copy=False,
                 unicode=unicode, order=order)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\biomechanics\musculotendon.py ===
"""Implementations of musculotendon models.

Musculotendon models are a critical component of biomechanical models, one that
differentiates them from pure multibody systems. Musculotendon models produce a
force dependent on their level of activation, their length, and their
extension velocity. Length- and extension velocity-dependent force production
are governed by force-length and force-velocity characteristics.
These are normalized functions that are dependent on the musculotendon's state
and are specific to a given musculotendon model.

"""

from abc import abstractmethod
from enum import IntEnum, unique

from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix, diag, eye, zeros
from sympy.physics.biomechanics.activation import ActivationBase
from sympy.physics.biomechanics.curve import (
    CharacteristicCurveCollection,
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceLengthPassiveInverseDeGroote2016,
    FiberForceVelocityDeGroote2016,
    FiberForceVelocityInverseDeGroote2016,
    TendonForceLengthDeGroote2016,
    TendonForceLengthInverseDeGroote2016,
)
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics.actuator import ForceActuator
from sympy.physics.vector.functions import dynamicsymbols


__all__ = [
    'MusculotendonBase',
    'MusculotendonDeGroote2016',
    'MusculotendonFormulation',
]


@unique
class MusculotendonFormulation(IntEnum):
    """Enumeration of types of musculotendon dynamics formulations.

    Explanation
    ===========

    An (integer) enumeration is used as it allows for clearer selection of the
    different formulations of musculotendon dynamics.

    Members
    =======

    RIGID_TENDON : 0
        A rigid tendon model.
    FIBER_LENGTH_EXPLICIT : 1
        An explicit elastic tendon model with the muscle fiber length (l_M) as
        the state variable.
    TENDON_FORCE_EXPLICIT : 2
        An explicit elastic tendon model with the tendon force (F_T) as the
        state variable.
    FIBER_LENGTH_IMPLICIT : 3
        An implicit elastic tendon model with the muscle fiber length (l_M) as
        the state variable and the muscle fiber velocity as an additional input
        variable.
    TENDON_FORCE_IMPLICIT : 4
        An implicit elastic tendon model with the tendon force (F_T) as the
        state variable as the muscle fiber velocity as an additional input
        variable.

    """

    RIGID_TENDON = 0
    FIBER_LENGTH_EXPLICIT = 1
    TENDON_FORCE_EXPLICIT = 2
    FIBER_LENGTH_IMPLICIT = 3
    TENDON_FORCE_IMPLICIT = 4

    def __str__(self):
        """Returns a string representation of the enumeration value.

        Notes
        =====

        This hard coding is required due to an incompatibility between the
        ``IntEnum`` implementations in Python 3.10 and Python 3.11
        (https://github.com/python/cpython/issues/84247). From Python 3.11
        onwards, the ``__str__`` method uses ``int.__str__``, whereas prior it
        used ``Enum.__str__``. Once Python 3.11 becomes the minimum version
        supported by SymPy, this method override can be removed.

        """
        return str(self.value)


_DEFAULT_MUSCULOTENDON_FORMULATION = MusculotendonFormulation.RIGID_TENDON


class MusculotendonBase(ForceActuator, _NamedMixin):
    r"""Abstract base class for all musculotendon classes to inherit from.

    Explanation
    ===========

    A musculotendon generates a contractile force based on its activation,
    length, and shortening velocity. This abstract base class is to be inherited
    by all musculotendon subclasses that implement different characteristic
    musculotendon curves. Characteristic musculotendon curves are required for
    the tendon force-length, passive fiber force-length, active fiber force-
    length, and fiber force-velocity relationships.

    Parameters
    ==========

    name : str
        The name identifier associated with the musculotendon. This name is used
        as a suffix when automatically generated symbols are instantiated. It
        must be a string of nonzero length.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    activation_dynamics : ActivationBase
        The activation dynamics that will be modeled within the musculotendon.
        This must be an instance of a concrete subclass of ``ActivationBase``,
        e.g. ``FirstOrderActivationDeGroote2016``.
    musculotendon_dynamics : MusculotendonFormulation | int
        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.
    tendon_slack_length : Expr | None
        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.
    peak_isometric_force : Expr | None
        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.
    optimal_fiber_length : Expr | None
        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.
    maximal_fiber_velocity : Expr | None
        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.
    optimal_pennation_angle : Expr | None
        The pennation angle when muscle fiber length equals the optimal fiber
        length.
    fiber_damping_coefficient : Expr | None
        The coefficient of damping to be used in the damping element in the
        muscle fiber model.
    with_defaults : bool
        Whether ``with_defaults`` alternate constructors should be used when
        automatically constructing child classes. Default is ``False``.

    """

    def __init__(
        self,
        name,
        pathway,
        activation_dynamics,
        *,
        musculotendon_dynamics=_DEFAULT_MUSCULOTENDON_FORMULATION,
        tendon_slack_length=None,
        peak_isometric_force=None,
        optimal_fiber_length=None,
        maximal_fiber_velocity=None,
        optimal_pennation_angle=None,
        fiber_damping_coefficient=None,
        with_defaults=False,
    ):
        self.name = name

        # Supply a placeholder force to the super initializer, this will be
        # replaced later
        super().__init__(Symbol('F'), pathway)

        # Activation dynamics
        if not isinstance(activation_dynamics, ActivationBase):
            msg = (
                f'Can\'t set attribute `activation_dynamics` to '
                f'{activation_dynamics} as it must be of type '
                f'`ActivationBase`, not {type(activation_dynamics)}.'
            )
            raise TypeError(msg)
        self._activation_dynamics = activation_dynamics
        self._child_objects = (self._activation_dynamics, )

        # Constants
        if tendon_slack_length is not None:
            self._l_T_slack = tendon_slack_length
        else:
            self._l_T_slack = Symbol(f'l_T_slack_{self.name}')
        if peak_isometric_force is not None:
            self._F_M_max = peak_isometric_force
        else:
            self._F_M_max = Symbol(f'F_M_max_{self.name}')
        if optimal_fiber_length is not None:
            self._l_M_opt = optimal_fiber_length
        else:
            self._l_M_opt = Symbol(f'l_M_opt_{self.name}')
        if maximal_fiber_velocity is not None:
            self._v_M_max = maximal_fiber_velocity
        else:
            self._v_M_max = Symbol(f'v_M_max_{self.name}')
        if optimal_pennation_angle is not None:
            self._alpha_opt = optimal_pennation_angle
        else:
            self._alpha_opt = Symbol(f'alpha_opt_{self.name}')
        if fiber_damping_coefficient is not None:
            self._beta = fiber_damping_coefficient
        else:
            self._beta = Symbol(f'beta_{self.name}')

        # Musculotendon dynamics
        self._with_defaults = with_defaults
        if musculotendon_dynamics == MusculotendonFormulation.RIGID_TENDON:
            self._rigid_tendon_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.FIBER_LENGTH_EXPLICIT:
            self._fiber_length_explicit_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.TENDON_FORCE_EXPLICIT:
            self._tendon_force_explicit_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.FIBER_LENGTH_IMPLICIT:
            self._fiber_length_implicit_musculotendon_dynamics()
        elif musculotendon_dynamics == MusculotendonFormulation.TENDON_FORCE_IMPLICIT:
            self._tendon_force_implicit_musculotendon_dynamics()
        else:
            msg = (
                f'Musculotendon dynamics {repr(musculotendon_dynamics)} '
                f'passed to `musculotendon_dynamics` was of type '
                f'{type(musculotendon_dynamics)}, must be '
                f'{MusculotendonFormulation}.'
            )
            raise TypeError(msg)
        self._musculotendon_dynamics = musculotendon_dynamics

        # Must override the placeholder value in `self._force` now that the
        # actual force has been calculated by
        # `self._<MUSCULOTENDON FORMULATION>_musculotendon_dynamics`.
        # Note that `self._force` assumes forces are expansile, musculotendon
        # forces are contractile hence the minus sign preceding `self._F_T`
        # (the tendon force).
        self._force = -self._F_T

    @classmethod
    def with_defaults(
        cls,
        name,
        pathway,
        activation_dynamics,
        *,
        musculotendon_dynamics=_DEFAULT_MUSCULOTENDON_FORMULATION,
        tendon_slack_length=None,
        peak_isometric_force=None,
        optimal_fiber_length=None,
        maximal_fiber_velocity=Float('10.0'),
        optimal_pennation_angle=Float('0.0'),
        fiber_damping_coefficient=Float('0.1'),
    ):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the musculotendon class using recommended
        values for ``v_M_max``, ``alpha_opt``, and ``beta``. The values are:

            :math:`v^M_{max} = 10`
            :math:`\alpha_{opt} = 0`
            :math:`\beta = \frac{1}{10}`

        The musculotendon curves are also instantiated using the constants from
        the original publication.

        Parameters
        ==========

        name : str
            The name identifier associated with the musculotendon. This name is
            used as a suffix when automatically generated symbols are
            instantiated. It must be a string of nonzero length.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of a
            concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
        activation_dynamics : ActivationBase
            The activation dynamics that will be modeled within the
            musculotendon. This must be an instance of a concrete subclass of
            ``ActivationBase``, e.g. ``FirstOrderActivationDeGroote2016``.
        musculotendon_dynamics : MusculotendonFormulation | int
            The formulation of musculotendon dynamics that should be used
            internally, i.e. rigid or elastic tendon model, the choice of
            musculotendon state etc. This must be a member of the integer
            enumeration ``MusculotendonFormulation`` or an integer that can be
            cast to a member. To use a rigid tendon formulation, set this to
            ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value
            ``0``, which will be cast to the enumeration member). There are four
            possible formulations for an elastic tendon model. To use an
            explicit formulation with the fiber length as the state, set this to
            ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer
            value ``1``). To use an explicit formulation with the tendon force
            as the state, set this to
            ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT`` (or the integer
            value ``2``). To use an implicit formulation with the fiber length
            as the state, set this to
            ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer
            value ``3``). To use an implicit formulation with the tendon force
            as the state, set this to
            ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT`` (or the integer
            value ``4``). The default is
            ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a
            rigid tendon formulation.
        tendon_slack_length : Expr | None
            The length of the tendon when the musculotendon is in its unloaded
            state. In a rigid tendon model the tendon length is the tendon slack
            length. In all musculotendon models, tendon slack length is used to
            normalize tendon length to give
            :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.
        peak_isometric_force : Expr | None
            The maximum force that the muscle fiber can produce when it is
            undergoing an isometric contraction (no lengthening velocity). In
            all musculotendon models, peak isometric force is used to normalized
            tendon and muscle fiber force to give
            :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.
        optimal_fiber_length : Expr | None
            The muscle fiber length at which the muscle fibers produce no
            passive force and their maximum active force. In all musculotendon
            models, optimal fiber length is used to normalize muscle fiber
            length to give :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.
        maximal_fiber_velocity : Expr | None
            The fiber velocity at which, during muscle fiber shortening, the
            muscle fibers are unable to produce any active force. In all
            musculotendon models, maximal fiber velocity is used to normalize
            muscle fiber extension velocity to give
            :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.
        optimal_pennation_angle : Expr | None
            The pennation angle when muscle fiber length equals the optimal
            fiber length.
        fiber_damping_coefficient : Expr | None
            The coefficient of damping to be used in the damping element in the
            muscle fiber model.

        """
        return cls(
            name,
            pathway,
            activation_dynamics=activation_dynamics,
            musculotendon_dynamics=musculotendon_dynamics,
            tendon_slack_length=tendon_slack_length,
            peak_isometric_force=peak_isometric_force,
            optimal_fiber_length=optimal_fiber_length,
            maximal_fiber_velocity=maximal_fiber_velocity,
            optimal_pennation_angle=optimal_pennation_angle,
            fiber_damping_coefficient=fiber_damping_coefficient,
            with_defaults=True,
        )

    @abstractmethod
    def curves(cls):
        """Return a ``CharacteristicCurveCollection`` of the curves related to
        the specific model."""
        pass

    @property
    def tendon_slack_length(self):
        r"""Symbol or value corresponding to the tendon slack length constant.

        Explanation
        ===========

        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.

        The alias ``l_T_slack`` can also be used to access the same attribute.

        """
        return self._l_T_slack

    @property
    def l_T_slack(self):
        r"""Symbol or value corresponding to the tendon slack length constant.

        Explanation
        ===========

        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.

        The alias ``tendon_slack_length`` can also be used to access the same
        attribute.

        """
        return self._l_T_slack

    @property
    def peak_isometric_force(self):
        r"""Symbol or value corresponding to the peak isometric force constant.

        Explanation
        ===========

        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.

        The alias ``F_M_max`` can also be used to access the same attribute.

        """
        return self._F_M_max

    @property
    def F_M_max(self):
        r"""Symbol or value corresponding to the peak isometric force constant.

        Explanation
        ===========

        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.

        The alias ``peak_isometric_force`` can also be used to access the same
        attribute.

        """
        return self._F_M_max

    @property
    def optimal_fiber_length(self):
        r"""Symbol or value corresponding to the optimal fiber length constant.

        Explanation
        ===========

        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.

        The alias ``l_M_opt`` can also be used to access the same attribute.

        """
        return self._l_M_opt

    @property
    def l_M_opt(self):
        r"""Symbol or value corresponding to the optimal fiber length constant.

        Explanation
        ===========

        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.

        The alias ``optimal_fiber_length`` can also be used to access the same
        attribute.

        """
        return self._l_M_opt

    @property
    def maximal_fiber_velocity(self):
        r"""Symbol or value corresponding to the maximal fiber velocity constant.

        Explanation
        ===========

        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.

        The alias ``v_M_max`` can also be used to access the same attribute.

        """
        return self._v_M_max

    @property
    def v_M_max(self):
        r"""Symbol or value corresponding to the maximal fiber velocity constant.

        Explanation
        ===========

        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.

        The alias ``maximal_fiber_velocity`` can also be used to access the same
        attribute.

        """
        return self._v_M_max

    @property
    def optimal_pennation_angle(self):
        """Symbol or value corresponding to the optimal pennation angle
        constant.

        Explanation
        ===========

        The pennation angle when muscle fiber length equals the optimal fiber
        length.

        The alias ``alpha_opt`` can also be used to access the same attribute.

        """
        return self._alpha_opt

    @property
    def alpha_opt(self):
        """Symbol or value corresponding to the optimal pennation angle
        constant.

        Explanation
        ===========

        The pennation angle when muscle fiber length equals the optimal fiber
        length.

        The alias ``optimal_pennation_angle`` can also be used to access the
        same attribute.

        """
        return self._alpha_opt

    @property
    def fiber_damping_coefficient(self):
        """Symbol or value corresponding to the fiber damping coefficient
        constant.

        Explanation
        ===========

        The coefficient of damping to be used in the damping element in the
        muscle fiber model.

        The alias ``beta`` can also be used to access the same attribute.

        """
        return self._beta

    @property
    def beta(self):
        """Symbol or value corresponding to the fiber damping coefficient
        constant.

        Explanation
        ===========

        The coefficient of damping to be used in the damping element in the
        muscle fiber model.

        The alias ``fiber_damping_coefficient`` can also be used to access the
        same attribute.

        """
        return self._beta

    @property
    def activation_dynamics(self):
        """Activation dynamics model governing this musculotendon's activation.

        Explanation
        ===========

        Returns the instance of a subclass of ``ActivationBase`` that governs
        the relationship between excitation and activation that is used to
        represent the activation dynamics of this musculotendon.

        """
        return self._activation_dynamics

    @property
    def excitation(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``e`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._e

    @property
    def e(self):
        """Dynamic symbol representing excitation.

        Explanation
        ===========

        The alias ``excitation`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._e

    @property
    def activation(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``a`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._a

    @property
    def a(self):
        """Dynamic symbol representing activation.

        Explanation
        ===========

        The alias ``activation`` can also be used to access the same attribute.

        """
        return self._activation_dynamics._a

    @property
    def musculotendon_dynamics(self):
        """The choice of rigid or type of elastic tendon musculotendon dynamics.

        Explanation
        ===========

        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.

        """
        return self._musculotendon_dynamics

    def _rigid_tendon_musculotendon_dynamics(self):
        """Rigid tendon musculotendon."""
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        self._l_T = self._l_T_slack
        self._l_T_tilde = Integer(1)
        self._l_M = sqrt((self._l_MT - self._l_T)**2 + (self._l_M_opt*sin(self._alpha_opt))**2)
        self._l_M_tilde = self._l_M/self._l_M_opt
        self._v_M = self._v_MT*(self._l_MT - self._l_T_slack)/self._l_M
        self._v_M_tilde = self._v_M/self._v_M_max
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
            self._fv_M = self.curves.fiber_force_velocity.with_defaults(self._v_M_tilde)
        else:
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._fv_M = self.curves.fiber_force_velocity(self._v_M_tilde, *fv_M_constants)
        self._F_M_tilde = self.a*self._fl_M_act*self._fv_M + self._fl_M_pas + self._beta*self._v_M_tilde
        self._F_T_tilde = self._F_M_tilde
        self._F_M = self._F_M_tilde*self._F_M_max
        self._cos_alpha = cos(self._alpha_opt)
        self._F_T = self._F_M*self._cos_alpha

        # Containers
        self._state_vars = zeros(0, 1)
        self._input_vars = zeros(0, 1)
        self._state_eqns = zeros(0, 1)
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)

    def _fiber_length_explicit_musculotendon_dynamics(self):
        """Elastic tendon musculotendon using `l_M_tilde` as a state."""
        self._l_M_tilde = dynamicsymbols(f'l_M_tilde_{self.name}')
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        self._l_M = self._l_M_tilde*self._l_M_opt
        self._l_T = self._l_MT - sqrt(self._l_M**2 - (self._l_M_opt*sin(self._alpha_opt))**2)
        self._l_T_tilde = self._l_T/self._l_T_slack
        self._cos_alpha = (self._l_MT - self._l_T)/self._l_M
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
        else:
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
        self._F_T_tilde = self._fl_T
        self._F_T = self._F_T_tilde*self._F_M_max
        self._F_M = self._F_T/self._cos_alpha
        self._F_M_tilde = self._F_M/self._F_M_max
        self._fv_M = (self._F_M_tilde - self._fl_M_pas)/(self.a*self._fl_M_act)
        if self._with_defaults:
            self._v_M_tilde = self.curves.fiber_force_velocity_inverse.with_defaults(self._fv_M)
        else:
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._v_M_tilde = self.curves.fiber_force_velocity_inverse(self._fv_M, *fv_M_constants)
        self._dl_M_tilde_dt = (self._v_M_max/self._l_M_opt)*self._v_M_tilde

        self._state_vars = Matrix([self._l_M_tilde])
        self._input_vars = zeros(0, 1)
        self._state_eqns = Matrix([self._dl_M_tilde_dt])
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)

    def _tendon_force_explicit_musculotendon_dynamics(self):
        """Elastic tendon musculotendon using `F_T_tilde` as a state."""
        self._F_T_tilde = dynamicsymbols(f'F_T_tilde_{self.name}')
        self._l_MT = self.pathway.length
        self._v_MT = self.pathway.extension_velocity
        self._fl_T = self._F_T_tilde
        if self._with_defaults:
            self._fl_T_inv = self.curves.tendon_force_length_inverse.with_defaults(self._fl_T)
        else:
            fl_T_constants = symbols(f'c_0:4_fl_T_{self.name}')
            self._fl_T_inv = self.curves.tendon_force_length_inverse(self._fl_T, *fl_T_constants)
        self._l_T_tilde = self._fl_T_inv
        self._l_T = self._l_T_tilde*self._l_T_slack
        self._l_M = sqrt((self._l_MT - self._l_T)**2 + (self._l_M_opt*sin(self._alpha_opt))**2)
        self._l_M_tilde = self._l_M/self._l_M_opt
        if self._with_defaults:
            self._fl_M_pas = self.curves.fiber_force_length_passive.with_defaults(self._l_M_tilde)
            self._fl_M_act = self.curves.fiber_force_length_active.with_defaults(self._l_M_tilde)
        else:
            fl_M_pas_constants = symbols(f'c_0:2_fl_M_pas_{self.name}')
            self._fl_M_pas = self.curves.fiber_force_length_passive(self._l_M_tilde, *fl_M_pas_constants)
            fl_M_act_constants = symbols(f'c_0:12_fl_M_act_{self.name}')
            self._fl_M_act = self.curves.fiber_force_length_active(self._l_M_tilde, *fl_M_act_constants)
        self._cos_alpha = (self._l_MT - self._l_T)/self._l_M
        self._F_T = self._F_T_tilde*self._F_M_max
        self._F_M = self._F_T/self._cos_alpha
        self._F_M_tilde = self._F_M/self._F_M_max
        self._fv_M = (self._F_M_tilde - self._fl_M_pas)/(self.a*self._fl_M_act)
        if self._with_defaults:
            self._fv_M_inv = self.curves.fiber_force_velocity_inverse.with_defaults(self._fv_M)
        else:
            fv_M_constants = symbols(f'c_0:4_fv_M_{self.name}')
            self._fv_M_inv = self.curves.fiber_force_velocity_inverse(self._fv_M, *fv_M_constants)
        self._v_M_tilde = self._fv_M_inv
        self._v_M = self._v_M_tilde*self._v_M_max
        self._v_T = self._v_MT - (self._v_M/self._cos_alpha)
        self._v_T_tilde = self._v_T/self._l_T_slack
        if self._with_defaults:
            self._fl_T = self.curves.tendon_force_length.with_defaults(self._l_T_tilde)
        else:
            self._fl_T = self.curves.tendon_force_length(self._l_T_tilde, *fl_T_constants)
        self._dF_T_tilde_dt = self._fl_T.diff(dynamicsymbols._t).subs({self._l_T_tilde.diff(dynamicsymbols._t): self._v_T_tilde})

        self._state_vars = Matrix([self._F_T_tilde])
        self._input_vars = zeros(0, 1)
        self._state_eqns = Matrix([self._dF_T_tilde_dt])
        self._curve_constants = Matrix(
            fl_T_constants
            + fl_M_pas_constants
            + fl_M_act_constants
            + fv_M_constants
        ) if not self._with_defaults else zeros(0, 1)

    def _fiber_length_implicit_musculotendon_dynamics(self):
        raise NotImplementedError

    def _tendon_force_implicit_musculotendon_dynamics(self):
        raise NotImplementedError

    @property
    def state_vars(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``x`` can also be used to access the same attribute.

        """
        state_vars = [self._state_vars]
        for child in self._child_objects:
            state_vars.append(child.state_vars)
        return Matrix.vstack(*state_vars)

    @property
    def x(self):
        """Ordered column matrix of functions of time that represent the state
        variables.

        Explanation
        ===========

        The alias ``state_vars`` can also be used to access the same attribute.

        """
        state_vars = [self._state_vars]
        for child in self._child_objects:
            state_vars.append(child.state_vars)
        return Matrix.vstack(*state_vars)

    @property
    def input_vars(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``r`` can also be used to access the same attribute.

        """
        input_vars = [self._input_vars]
        for child in self._child_objects:
            input_vars.append(child.input_vars)
        return Matrix.vstack(*input_vars)

    @property
    def r(self):
        """Ordered column matrix of functions of time that represent the input
        variables.

        Explanation
        ===========

        The alias ``input_vars`` can also be used to access the same attribute.

        """
        input_vars = [self._input_vars]
        for child in self._child_objects:
            input_vars.append(child.input_vars)
        return Matrix.vstack(*input_vars)

    @property
    def constants(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``p`` can also be used to access the same attribute.

        """
        musculotendon_constants = [
            self._l_T_slack,
            self._F_M_max,
            self._l_M_opt,
            self._v_M_max,
            self._alpha_opt,
            self._beta,
        ]
        musculotendon_constants = [
            c for c in musculotendon_constants if not c.is_number
        ]
        constants = [
            Matrix(musculotendon_constants)
            if musculotendon_constants
            else zeros(0, 1)
        ]
        for child in self._child_objects:
            constants.append(child.constants)
        constants.append(self._curve_constants)
        return Matrix.vstack(*constants)

    @property
    def p(self):
        """Ordered column matrix of non-time varying symbols present in ``M``
        and ``F``.

        Explanation
        ===========

        Only symbolic constants are returned. If a numeric type (e.g. ``Float``)
        has been used instead of ``Symbol`` for a constant then that attribute
        will not be included in the matrix returned by this property. This is
        because the primary use of this property attribute is to provide an
        ordered sequence of the still-free symbols that require numeric values
        during code generation.

        The alias ``constants`` can also be used to access the same attribute.

        """
        musculotendon_constants = [
            self._l_T_slack,
            self._F_M_max,
            self._l_M_opt,
            self._v_M_max,
            self._alpha_opt,
            self._beta,
        ]
        musculotendon_constants = [
            c for c in musculotendon_constants if not c.is_number
        ]
        constants = [
            Matrix(musculotendon_constants)
            if musculotendon_constants
            else zeros(0, 1)
        ]
        for child in self._child_objects:
            constants.append(child.constants)
        constants.append(self._curve_constants)
        return Matrix.vstack(*constants)

    @property
    def M(self):
        """Ordered square matrix of coefficients on the LHS of ``M x' = F``.

        Explanation
        ===========

        The square matrix that forms part of the LHS of the linear system of
        ordinary differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``M`` is an empty square
        ``Matrix`` with shape (0, 0).

        """
        M = [eye(len(self._state_vars))]
        for child in self._child_objects:
            M.append(child.M)
        return diag(*M)

    @property
    def F(self):
        """Ordered column matrix of equations on the RHS of ``M x' = F``.

        Explanation
        ===========

        The column matrix that forms the RHS of the linear system of ordinary
        differential equations governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear system has dimension 0 and therefore ``F`` is an empty column
        ``Matrix`` with shape (0, 1).

        """
        F = [self._state_eqns]
        for child in self._child_objects:
            F.append(child.F)
        return Matrix.vstack(*F)

    def rhs(self):
        """Ordered column matrix of equations for the solution of ``M x' = F``.

        Explanation
        ===========

        The solution to the linear system of ordinary differential equations
        governing the activation dynamics:

        ``M(x, r, t, p) x' = F(x, r, t, p)``.

        As zeroth-order activation dynamics have no state variables, this
        linear has dimension 0 and therefore this method returns an empty
        column ``Matrix`` with shape (0, 1).

        """
        is_explicit = (
            MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
            MusculotendonFormulation.TENDON_FORCE_EXPLICIT,
        )
        if self.musculotendon_dynamics is MusculotendonFormulation.RIGID_TENDON:
            child_rhs = [child.rhs() for child in self._child_objects]
            return Matrix.vstack(*child_rhs)
        elif self.musculotendon_dynamics in is_explicit:
            rhs = self._state_eqns
            child_rhs = [child.rhs() for child in self._child_objects]
            return Matrix.vstack(rhs, *child_rhs)
        return self.M.solve(self.F)

    def __repr__(self):
        """Returns a string representation to reinstantiate the model."""
        return (
            f'{self.__class__.__name__}({self.name!r}, '
            f'pathway={self.pathway!r}, '
            f'activation_dynamics={self.activation_dynamics!r}, '
            f'musculotendon_dynamics={self.musculotendon_dynamics}, '
            f'tendon_slack_length={self._l_T_slack!r}, '
            f'peak_isometric_force={self._F_M_max!r}, '
            f'optimal_fiber_length={self._l_M_opt!r}, '
            f'maximal_fiber_velocity={self._v_M_max!r}, '
            f'optimal_pennation_angle={self._alpha_opt!r}, '
            f'fiber_damping_coefficient={self._beta!r})'
        )

    def __str__(self):
        """Returns a string representation of the expression for musculotendon
        force."""
        return str(self.force)


class MusculotendonDeGroote2016(MusculotendonBase):
    r"""Musculotendon model using the curves of De Groote et al., 2016 [1]_.

    Examples
    ========

    This class models the musculotendon actuator parametrized by the
    characteristic curves described in De Groote et al., 2016 [1]_. Like all
    musculotendon models in SymPy's biomechanics module, it requires a pathway
    to define its line of action. We'll begin by creating a simple
    ``LinearPathway`` between two points that our musculotendon will follow.
    We'll create a point ``O`` to represent the musculotendon's origin and
    another ``I`` to represent its insertion.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearPathway, Point,
    ...     ReferenceFrame, dynamicsymbols)

    >>> N = ReferenceFrame('N')
    >>> O, I = O, P = symbols('O, I', cls=Point)
    >>> q, u = dynamicsymbols('q, u', real=True)
    >>> I.set_pos(O, q*N.x)
    >>> O.set_vel(N, 0)
    >>> I.set_vel(N, u*N.x)
    >>> pathway = LinearPathway(O, I)
    >>> pathway.attachments
    (O, I)
    >>> pathway.length
    Abs(q(t))
    >>> pathway.extension_velocity
    sign(q(t))*Derivative(q(t), t)

    A musculotendon also takes an instance of an activation dynamics model as
    this will be used to provide symbols for the activation in the formulation
    of the musculotendon dynamics. We'll use an instance of
    ``FirstOrderActivationDeGroote2016`` to represent first-order activation
    dynamics. Note that a single name argument needs to be provided as SymPy
    will use this as a suffix.

    >>> from sympy.physics.biomechanics import FirstOrderActivationDeGroote2016

    >>> activation = FirstOrderActivationDeGroote2016('muscle')
    >>> activation.x
    Matrix([[a_muscle(t)]])
    >>> activation.r
    Matrix([[e_muscle(t)]])
    >>> activation.p
    Matrix([
    [tau_a_muscle],
    [tau_d_muscle],
    [    b_muscle]])
    >>> activation.rhs()
    Matrix([[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    The musculotendon class requires symbols or values to be passed to represent
    the constants in the musculotendon dynamics. We'll use SymPy's ``symbols``
    function to create symbols for the maximum isometric force ``F_M_max``,
    optimal fiber length ``l_M_opt``, tendon slack length ``l_T_slack``, maximum
    fiber velocity ``v_M_max``, optimal pennation angle ``alpha_opt, and fiber
    damping coefficient ``beta``.

    >>> F_M_max = symbols('F_M_max', real=True)
    >>> l_M_opt = symbols('l_M_opt', real=True)
    >>> l_T_slack = symbols('l_T_slack', real=True)
    >>> v_M_max = symbols('v_M_max', real=True)
    >>> alpha_opt = symbols('alpha_opt', real=True)
    >>> beta = symbols('beta', real=True)

    We can then import the class ``MusculotendonDeGroote2016`` from the
    biomechanics module and create an instance by passing in the various objects
    we have previously instantiated. By default, a musculotendon model with
    rigid tendon musculotendon dynamics will be created.

    >>> from sympy.physics.biomechanics import MusculotendonDeGroote2016

    >>> rigid_tendon_muscle = MusculotendonDeGroote2016(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ...     maximal_fiber_velocity=v_M_max,
    ...     optimal_pennation_angle=alpha_opt,
    ...     fiber_damping_coefficient=beta,
    ... )

    We can inspect the various properties of the musculotendon, including
    getting the symbolic expression describing the force it produces using its
    ``force`` attribute.

    >>> rigid_tendon_muscle.force
    -F_M_max*(beta*(-l_T_slack + Abs(q(t)))*sign(q(t))*Derivative(q(t), t)...

    When we created the musculotendon object, we passed in an instance of an
    activation dynamics object that governs the activation within the
    musculotendon. SymPy makes a design choice here that the activation dynamics
    instance will be treated as a child object of the musculotendon dynamics.
    Therefore, if we want to inspect the state and input variables associated
    with the musculotendon model, we will also be returned the state and input
    variables associated with the child object, or the activation dynamics in
    this case. As the musculotendon model that we created here uses rigid tendon
    dynamics, no additional states or inputs relating to the musculotendon are
    introduces. Consequently, the model has a single state associated with it,
    the activation, and a single input associated with it, the excitation. The
    states and inputs can be inspected using the ``x`` and ``r`` attributes
    respectively. Note that both ``x`` and ``r`` have the alias attributes of
    ``state_vars`` and ``input_vars``.

    >>> rigid_tendon_muscle.x
    Matrix([[a_muscle(t)]])
    >>> rigid_tendon_muscle.r
    Matrix([[e_muscle(t)]])

    To see which constants are symbolic in the musculotendon model, we can use
    the ``p`` or ``constants`` attribute. This returns a ``Matrix`` populated
    by the constants that are represented by a ``Symbol`` rather than a numeric
    value.

    >>> rigid_tendon_muscle.p
    Matrix([
    [           l_T_slack],
    [             F_M_max],
    [             l_M_opt],
    [             v_M_max],
    [           alpha_opt],
    [                beta],
    [        tau_a_muscle],
    [        tau_d_muscle],
    [            b_muscle],
    [     c_0_fl_T_muscle],
    [     c_1_fl_T_muscle],
    [     c_2_fl_T_muscle],
    [     c_3_fl_T_muscle],
    [ c_0_fl_M_pas_muscle],
    [ c_1_fl_M_pas_muscle],
    [ c_0_fl_M_act_muscle],
    [ c_1_fl_M_act_muscle],
    [ c_2_fl_M_act_muscle],
    [ c_3_fl_M_act_muscle],
    [ c_4_fl_M_act_muscle],
    [ c_5_fl_M_act_muscle],
    [ c_6_fl_M_act_muscle],
    [ c_7_fl_M_act_muscle],
    [ c_8_fl_M_act_muscle],
    [ c_9_fl_M_act_muscle],
    [c_10_fl_M_act_muscle],
    [c_11_fl_M_act_muscle],
    [     c_0_fv_M_muscle],
    [     c_1_fv_M_muscle],
    [     c_2_fv_M_muscle],
    [     c_3_fv_M_muscle]])

    Finally, we can call the ``rhs`` method to return a ``Matrix`` that
    contains as its elements the righthand side of the ordinary differential
    equations corresponding to each of the musculotendon's states. Like the
    method with the same name on the ``Method`` classes in SymPy's mechanics
    module, this returns a column vector where the number of rows corresponds to
    the number of states. For our example here, we have a single state, the
    dynamic symbol ``a_muscle(t)``, so the returned value is a 1-by-1
    ``Matrix``.

    >>> rigid_tendon_muscle.rhs()
    Matrix([[((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    The musculotendon class supports elastic tendon musculotendon models in
    addition to rigid tendon ones. You can choose to either use the fiber length
    or tendon force as an additional state. You can also specify whether an
    explicit or implicit formulation should be used. To select a formulation,
    pass a member of the ``MusculotendonFormulation`` enumeration to the
    ``musculotendon_dynamics`` parameter when calling the constructor. This
    enumeration is an ``IntEnum``, so you can also pass an integer, however it
    is recommended to use the enumeration as it is clearer which formulation you
    are actually selecting. Below, we'll use the ``FIBER_LENGTH_EXPLICIT``
    member to create a musculotendon with an elastic tendon that will use the
    (normalized) muscle fiber length as an additional state and will produce
    the governing ordinary differential equation in explicit form.

    >>> from sympy.physics.biomechanics import MusculotendonFormulation

    >>> elastic_tendon_muscle = MusculotendonDeGroote2016(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ...     maximal_fiber_velocity=v_M_max,
    ...     optimal_pennation_angle=alpha_opt,
    ...     fiber_damping_coefficient=beta,
    ... )

    >>> elastic_tendon_muscle.force
    -F_M_max*TendonForceLengthDeGroote2016((-sqrt(l_M_opt**2*...
    >>> elastic_tendon_muscle.x
    Matrix([
    [l_M_tilde_muscle(t)],
    [        a_muscle(t)]])
    >>> elastic_tendon_muscle.r
    Matrix([[e_muscle(t)]])
    >>> elastic_tendon_muscle.p
    Matrix([
    [           l_T_slack],
    [             F_M_max],
    [             l_M_opt],
    [             v_M_max],
    [           alpha_opt],
    [                beta],
    [        tau_a_muscle],
    [        tau_d_muscle],
    [            b_muscle],
    [     c_0_fl_T_muscle],
    [     c_1_fl_T_muscle],
    [     c_2_fl_T_muscle],
    [     c_3_fl_T_muscle],
    [ c_0_fl_M_pas_muscle],
    [ c_1_fl_M_pas_muscle],
    [ c_0_fl_M_act_muscle],
    [ c_1_fl_M_act_muscle],
    [ c_2_fl_M_act_muscle],
    [ c_3_fl_M_act_muscle],
    [ c_4_fl_M_act_muscle],
    [ c_5_fl_M_act_muscle],
    [ c_6_fl_M_act_muscle],
    [ c_7_fl_M_act_muscle],
    [ c_8_fl_M_act_muscle],
    [ c_9_fl_M_act_muscle],
    [c_10_fl_M_act_muscle],
    [c_11_fl_M_act_muscle],
    [     c_0_fv_M_muscle],
    [     c_1_fv_M_muscle],
    [     c_2_fv_M_muscle],
    [     c_3_fv_M_muscle]])
    >>> elastic_tendon_muscle.rhs()
    Matrix([
    [v_M_max*FiberForceVelocityInverseDeGroote2016((l_M_opt*...],
    [ ((1/2 - tanh(b_muscle*(-a_muscle(t) + e_muscle(t)))/2)*(3*...]])

    It is strongly recommended to use the alternate ``with_defaults``
    constructor when creating an instance because this will ensure that the
    published constants are used in the musculotendon characteristic curves.

    >>> elastic_tendon_muscle = MusculotendonDeGroote2016.with_defaults(
    ...     'muscle',
    ...     pathway,
    ...     activation,
    ...     musculotendon_dynamics=MusculotendonFormulation.FIBER_LENGTH_EXPLICIT,
    ...     tendon_slack_length=l_T_slack,
    ...     peak_isometric_force=F_M_max,
    ...     optimal_fiber_length=l_M_opt,
    ... )

    >>> elastic_tendon_muscle.x
    Matrix([
    [l_M_tilde_muscle(t)],
    [        a_muscle(t)]])
    >>> elastic_tendon_muscle.r
    Matrix([[e_muscle(t)]])
    >>> elastic_tendon_muscle.p
    Matrix([
    [   l_T_slack],
    [     F_M_max],
    [     l_M_opt],
    [tau_a_muscle],
    [tau_d_muscle],
    [    b_muscle]])

    Parameters
    ==========

    name : str
        The name identifier associated with the musculotendon. This name is used
        as a suffix when automatically generated symbols are instantiated. It
        must be a string of nonzero length.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    activation_dynamics : ActivationBase
        The activation dynamics that will be modeled within the musculotendon.
        This must be an instance of a concrete subclass of ``ActivationBase``,
        e.g. ``FirstOrderActivationDeGroote2016``.
    musculotendon_dynamics : MusculotendonFormulation | int
        The formulation of musculotendon dynamics that should be used
        internally, i.e. rigid or elastic tendon model, the choice of
        musculotendon state etc. This must be a member of the integer
        enumeration ``MusculotendonFormulation`` or an integer that can be cast
        to a member. To use a rigid tendon formulation, set this to
        ``MusculotendonFormulation.RIGID_TENDON`` (or the integer value ``0``,
        which will be cast to the enumeration member). There are four possible
        formulations for an elastic tendon model. To use an explicit formulation
        with the fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_EXPLICIT`` (or the integer value
        ``1``). To use an explicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_EXPLICIT``
        (or the integer value ``2``). To use an implicit formulation with the
        fiber length as the state, set this to
        ``MusculotendonFormulation.FIBER_LENGTH_IMPLICIT`` (or the integer value
        ``3``). To use an implicit formulation with the tendon force as the
        state, set this to ``MusculotendonFormulation.TENDON_FORCE_IMPLICIT``
        (or the integer value ``4``). The default is
        ``MusculotendonFormulation.RIGID_TENDON``, which corresponds to a rigid
        tendon formulation.
    tendon_slack_length : Expr | None
        The length of the tendon when the musculotendon is in its unloaded
        state. In a rigid tendon model the tendon length is the tendon slack
        length. In all musculotendon models, tendon slack length is used to
        normalize tendon length to give
        :math:`\tilde{l}^T = \frac{l^T}{l^T_{slack}}`.
    peak_isometric_force : Expr | None
        The maximum force that the muscle fiber can produce when it is
        undergoing an isometric contraction (no lengthening velocity). In all
        musculotendon models, peak isometric force is used to normalized tendon
        and muscle fiber force to give
        :math:`\tilde{F}^T = \frac{F^T}{F^M_{max}}`.
    optimal_fiber_length : Expr | None
        The muscle fiber length at which the muscle fibers produce no passive
        force and their maximum active force. In all musculotendon models,
        optimal fiber length is used to normalize muscle fiber length to give
        :math:`\tilde{l}^M = \frac{l^M}{l^M_{opt}}`.
    maximal_fiber_velocity : Expr | None
        The fiber velocity at which, during muscle fiber shortening, the muscle
        fibers are unable to produce any active force. In all musculotendon
        models, maximal fiber velocity is used to normalize muscle fiber
        extension velocity to give :math:`\tilde{v}^M = \frac{v^M}{v^M_{max}}`.
    optimal_pennation_angle : Expr | None
        The pennation angle when muscle fiber length equals the optimal fiber
        length.
    fiber_damping_coefficient : Expr | None
        The coefficient of damping to be used in the damping element in the
        muscle fiber model.
    with_defaults : bool
        Whether ``with_defaults`` alternate constructors should be used when
        automatically constructing child classes. Default is ``False``.

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    curves = CharacteristicCurveCollection(
        tendon_force_length=TendonForceLengthDeGroote2016,
        tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
        fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
        fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
        fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
        fiber_force_velocity=FiberForceVelocityDeGroote2016,
        fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\libmp\libmpf.py ===
"""
Low-level functions for arbitrary-precision floating-point arithmetic.
"""

__docformat__ = 'plaintext'

import math

from bisect import bisect

import sys

# Importing random is slow
#from random import getrandbits
getrandbits = None

from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
    BACKEND, STRICT, HASH_MODULUS, HASH_BITS, gmpy, sage, sage_utils)

from .libintmath import (giant_steps,
    trailtable, bctable, lshift, rshift, bitcount, trailing,
    sqrt_fixed, numeral, isqrt, isqrt_fast, sqrtrem,
    bin_to_radix)

# We don't pickle tuples directly for the following reasons:
#   1: pickle uses str() for ints, which is inefficient when they are large
#   2: pickle doesn't work for gmpy mpzs
# Both problems are solved by using hex()

if BACKEND == 'sage':
    def to_pickable(x):
        sign, man, exp, bc = x
        return sign, hex(man), exp, bc
else:
    def to_pickable(x):
        sign, man, exp, bc = x
        return sign, hex(man)[2:], exp, bc

def from_pickable(x):
    sign, man, exp, bc = x
    return (sign, MPZ(man, 16), exp, bc)

class ComplexResult(ValueError):
    pass

try:
    intern
except NameError:
    intern = lambda x: x

# All supported rounding modes
round_nearest = intern('n')
round_floor = intern('f')
round_ceiling = intern('c')
round_up = intern('u')
round_down = intern('d')
round_fast = round_down

def prec_to_dps(n):
    """Return number of accurate decimals that can be represented
    with a precision of n bits."""
    return max(1, int(round(int(n)/3.3219280948873626)-1))

def dps_to_prec(n):
    """Return the number of bits required to represent n decimals
    accurately."""
    return max(1, int(round((int(n)+1)*3.3219280948873626)))

def repr_dps(n):
    """Return the number of decimal digits required to represent
    a number with n-bit precision so that it can be uniquely
    reconstructed from the representation."""
    dps = prec_to_dps(n)
    if dps == 15:
        return 17
    return dps + 3

#----------------------------------------------------------------------------#
#                    Some commonly needed float values                       #
#----------------------------------------------------------------------------#

# Regular number format:
# (-1)**sign * mantissa * 2**exponent, plus bitcount of mantissa
fzero = (0, MPZ_ZERO, 0, 0)
fnzero = (1, MPZ_ZERO, 0, 0)
fone = (0, MPZ_ONE, 0, 1)
fnone = (1, MPZ_ONE, 0, 1)
ftwo = (0, MPZ_ONE, 1, 1)
ften = (0, MPZ_FIVE, 1, 3)
fhalf = (0, MPZ_ONE, -1, 1)

# Arbitrary encoding for special numbers: zero mantissa, nonzero exponent
fnan = (0, MPZ_ZERO, -123, -1)
finf = (0, MPZ_ZERO, -456, -2)
fninf = (1, MPZ_ZERO, -789, -3)

# Was 1e1000; this is broken in Python 2.4
math_float_inf = 1e300 * 1e300


#----------------------------------------------------------------------------#
#                                  Rounding                                  #
#----------------------------------------------------------------------------#

# This function can be used to round a mantissa generally. However,
# we will try to do most rounding inline for efficiency.
def round_int(x, n, rnd):
    if rnd == round_nearest:
        if x >= 0:
            t = x >> (n-1)
            if t & 1 and ((t & 2) or (x & h_mask[n<300][n])):
                return (t>>1)+1
            else:
                return t>>1
        else:
            return -round_int(-x, n, rnd)
    if rnd == round_floor:
        return x >> n
    if rnd == round_ceiling:
        return -((-x) >> n)
    if rnd == round_down:
        if x >= 0:
            return x >> n
        return -((-x) >> n)
    if rnd == round_up:
        if x >= 0:
            return -((-x) >> n)
        return x >> n

# These masks are used to pick out segments of numbers to determine
# which direction to round when rounding to nearest.
class h_mask_big:
    def __getitem__(self, n):
        return (MPZ_ONE<<(n-1))-1

h_mask_small = [0]+[((MPZ_ONE<<(_-1))-1) for _ in range(1, 300)]
h_mask = [h_mask_big(), h_mask_small]

# The >> operator rounds to floor. shifts_down[rnd][sign]
# tells whether this is the right direction to use, or if the
# number should be negated before shifting
shifts_down = {round_floor:(1,0), round_ceiling:(0,1),
    round_down:(1,1), round_up:(0,0)}


#----------------------------------------------------------------------------#
#                          Normalization of raw mpfs                         #
#----------------------------------------------------------------------------#

# This function is called almost every time an mpf is created.
# It has been optimized accordingly.

def _normalize(sign, man, exp, bc, prec, rnd):
    """
    Create a raw mpf tuple with value (-1)**sign * man * 2**exp and
    normalized mantissa. The mantissa is rounded in the specified
    direction if its size exceeds the precision. Trailing zero bits
    are also stripped from the mantissa to ensure that the
    representation is canonical.

    Conditions on the input:
    * The input must represent a regular (finite) number
    * The sign bit must be 0 or 1
    * The mantissa must be positive
    * The exponent must be an integer
    * The bitcount must be exact

    If these conditions are not met, use from_man_exp, mpf_pos, or any
    of the conversion functions to create normalized raw mpf tuples.
    """
    if not man:
        return fzero
    # Cut mantissa down to size if larger than target precision
    n = bc - prec
    if n > 0:
        if rnd == round_nearest:
            t = man >> (n-1)
            if t & 1 and ((t & 2) or (man & h_mask[n<300][n])):
                man = (t>>1)+1
            else:
                man = t>>1
        elif shifts_down[rnd][sign]:
            man >>= n
        else:
            man = -((-man)>>n)
        exp += n
        bc = prec
    # Strip trailing bits
    if not man & 1:
        t = trailtable[int(man & 255)]
        if not t:
            while not man & 255:
                man >>= 8
                exp += 8
                bc -= 8
            t = trailtable[int(man & 255)]
        man >>= t
        exp += t
        bc -= t
    # Bit count can be wrong if the input mantissa was 1 less than
    # a power of 2 and got rounded up, thereby adding an extra bit.
    # With trailing bits removed, all powers of two have mantissa 1,
    # so this is easy to check for.
    if man == 1:
        bc = 1
    return sign, man, exp, bc

def _normalize1(sign, man, exp, bc, prec, rnd):
    """same as normalize, but with the added condition that
       man is odd or zero
    """
    if not man:
        return fzero
    if bc <= prec:
        return sign, man, exp, bc
    n = bc - prec
    if rnd == round_nearest:
        t = man >> (n-1)
        if t & 1 and ((t & 2) or (man & h_mask[n<300][n])):
            man = (t>>1)+1
        else:
            man = t>>1
    elif shifts_down[rnd][sign]:
        man >>= n
    else:
        man = -((-man)>>n)
    exp += n
    bc = prec
    # Strip trailing bits
    if not man & 1:
        t = trailtable[int(man & 255)]
        if not t:
            while not man & 255:
                man >>= 8
                exp += 8
                bc -= 8
            t = trailtable[int(man & 255)]
        man >>= t
        exp += t
        bc -= t
    # Bit count can be wrong if the input mantissa was 1 less than
    # a power of 2 and got rounded up, thereby adding an extra bit.
    # With trailing bits removed, all powers of two have mantissa 1,
    # so this is easy to check for.
    if man == 1:
        bc = 1
    return sign, man, exp, bc

try:
    _exp_types = (int, long)
except NameError:
    _exp_types = (int,)

def strict_normalize(sign, man, exp, bc, prec, rnd):
    """Additional checks on the components of an mpf. Enable tests by setting
       the environment variable MPMATH_STRICT to Y."""
    assert type(man) == MPZ_TYPE
    assert type(bc) in _exp_types
    assert type(exp) in _exp_types
    assert bc == bitcount(man)
    return _normalize(sign, man, exp, bc, prec, rnd)

def strict_normalize1(sign, man, exp, bc, prec, rnd):
    """Additional checks on the components of an mpf. Enable tests by setting
       the environment variable MPMATH_STRICT to Y."""
    assert type(man) == MPZ_TYPE
    assert type(bc) in _exp_types
    assert type(exp) in _exp_types
    assert bc == bitcount(man)
    assert (not man) or (man & 1)
    return _normalize1(sign, man, exp, bc, prec, rnd)

if BACKEND == 'gmpy' and '_mpmath_normalize' in dir(gmpy):
    _normalize = gmpy._mpmath_normalize
    _normalize1 = gmpy._mpmath_normalize

if BACKEND == 'sage':
    _normalize = _normalize1 = sage_utils.normalize

if STRICT:
    normalize = strict_normalize
    normalize1 = strict_normalize1
else:
    normalize = _normalize
    normalize1 = _normalize1

#----------------------------------------------------------------------------#
#                            Conversion functions                            #
#----------------------------------------------------------------------------#

def from_man_exp(man, exp, prec=None, rnd=round_fast):
    """Create raw mpf from (man, exp) pair. The mantissa may be signed.
    If no precision is specified, the mantissa is stored exactly."""
    man = MPZ(man)
    sign = 0
    if man < 0:
        sign = 1
        man = -man
    if man < 1024:
        bc = bctable[int(man)]
    else:
        bc = bitcount(man)
    if not prec:
        if not man:
            return fzero
        if not man & 1:
            if man & 2:
                return (sign, man >> 1, exp + 1, bc - 1)
            t = trailtable[int(man & 255)]
            if not t:
                while not man & 255:
                    man >>= 8
                    exp += 8
                    bc -= 8
                t = trailtable[int(man & 255)]
            man >>= t
            exp += t
            bc -= t
        return (sign, man, exp, bc)
    return normalize(sign, man, exp, bc, prec, rnd)

int_cache = dict((n, from_man_exp(n, 0)) for n in range(-10, 257))

if BACKEND == 'gmpy' and '_mpmath_create' in dir(gmpy):
    from_man_exp = gmpy._mpmath_create

if BACKEND == 'sage':
    from_man_exp = sage_utils.from_man_exp

def from_int(n, prec=0, rnd=round_fast):
    """Create a raw mpf from an integer. If no precision is specified,
    the mantissa is stored exactly."""
    if not prec:
        if n in int_cache:
            return int_cache[n]
    return from_man_exp(n, 0, prec, rnd)

def to_man_exp(s):
    """Return (man, exp) of a raw mpf. Raise an error if inf/nan."""
    sign, man, exp, bc = s
    if (not man) and exp:
        raise ValueError("mantissa and exponent are undefined for %s" % man)
    return man, exp

def to_int(s, rnd=None):
    """Convert a raw mpf to the nearest int. Rounding is done down by
    default (same as int(float) in Python), but can be changed. If the
    input is inf/nan, an exception is raised."""
    sign, man, exp, bc = s
    if (not man) and exp:
        raise ValueError("cannot convert inf or nan to int")
    if exp >= 0:
        if sign:
            return (-man) << exp
        return man << exp
    # Make default rounding fast
    if not rnd:
        if sign:
            return -(man >> (-exp))
        else:
            return man >> (-exp)
    if sign:
        return round_int(-man, -exp, rnd)
    else:
        return round_int(man, -exp, rnd)

def mpf_round_int(s, rnd):
    sign, man, exp, bc = s
    if (not man) and exp:
        return s
    if exp >= 0:
        return s
    mag = exp+bc
    if mag < 1:
        if rnd == round_ceiling:
            if sign: return fzero
            else:    return fone
        elif rnd == round_floor:
            if sign: return fnone
            else:    return fzero
        elif rnd == round_nearest:
            if mag < 0 or man == MPZ_ONE: return fzero
            elif sign: return fnone
            else:      return fone
        else:
            raise NotImplementedError
    return mpf_pos(s, min(bc, mag), rnd)

def mpf_floor(s, prec=0, rnd=round_fast):
    v = mpf_round_int(s, round_floor)
    if prec:
        v = mpf_pos(v, prec, rnd)
    return v

def mpf_ceil(s, prec=0, rnd=round_fast):
    v = mpf_round_int(s, round_ceiling)
    if prec:
        v = mpf_pos(v, prec, rnd)
    return v

def mpf_nint(s, prec=0, rnd=round_fast):
    v = mpf_round_int(s, round_nearest)
    if prec:
        v = mpf_pos(v, prec, rnd)
    return v

def mpf_frac(s, prec=0, rnd=round_fast):
    return mpf_sub(s, mpf_floor(s), prec, rnd)

def from_float(x, prec=53, rnd=round_fast):
    """Create a raw mpf from a Python float, rounding if necessary.
    If prec >= 53, the result is guaranteed to represent exactly the
    same number as the input. If prec is not specified, use prec=53."""
    # frexp only raises an exception for nan on some platforms
    if x != x:
        return fnan
    # in Python2.5 math.frexp gives an exception for float infinity
    # in Python2.6 it returns (float infinity, 0)
    try:
        m, e = math.frexp(x)
    except:
        if x == math_float_inf: return finf
        if x == -math_float_inf: return fninf
        return fnan
    if x == math_float_inf: return finf
    if x == -math_float_inf: return fninf
    return from_man_exp(int(m*(1<<53)), e-53, prec, rnd)

def from_npfloat(x, prec=113, rnd=round_fast):
    """Create a raw mpf from a numpy float, rounding if necessary.
    If prec >= 113, the result is guaranteed to represent exactly the
    same number as the input. If prec is not specified, use prec=113."""
    y = float(x)
    if x == y: # ldexp overflows for float16
        return from_float(y, prec, rnd)
    import numpy as np
    if np.isfinite(x):
        m, e = np.frexp(x)
        return from_man_exp(int(np.ldexp(m, 113)), int(e-113), prec, rnd)
    if np.isposinf(x): return finf
    if np.isneginf(x): return fninf
    return fnan

def from_Decimal(x, prec=None, rnd=round_fast):
    """Create a raw mpf from a Decimal, rounding if necessary.
    If prec is not specified, use the equivalent bit precision
    of the number of significant digits in x."""
    if x.is_nan(): return fnan
    if x.is_infinite(): return fninf if x.is_signed() else finf
    if prec is None:
        prec = int(len(x.as_tuple()[1])*3.3219280948873626)
    return from_str(str(x), prec, rnd)

def to_float(s, strict=False, rnd=round_fast):
    """
    Convert a raw mpf to a Python float. The result is exact if the
    bitcount of s is <= 53 and no underflow/overflow occurs.

    If the number is too large or too small to represent as a regular
    float, it will be converted to inf or 0.0. Setting strict=True
    forces an OverflowError to be raised instead.

    Warning: with a directed rounding mode, the correct nearest representable
    floating-point number in the specified direction might not be computed
    in case of overflow or (gradual) underflow.
    """
    sign, man, exp, bc = s
    if not man:
        if s == fzero: return 0.0
        if s == finf: return math_float_inf
        if s == fninf: return -math_float_inf
        return math_float_inf/math_float_inf
    if bc > 53:
        sign, man, exp, bc = normalize1(sign, man, exp, bc, 53, rnd)
    if sign:
        man = -man
    try:
        return math.ldexp(man, exp)
    except OverflowError:
        if strict:
            raise
        # Overflow to infinity
        if exp + bc > 0:
            if sign:
                return -math_float_inf
            else:
                return math_float_inf
        # Underflow to zero
        return 0.0

def from_rational(p, q, prec, rnd=round_fast):
    """Create a raw mpf from a rational number p/q, round if
    necessary."""
    return mpf_div(from_int(p), from_int(q), prec, rnd)

def to_rational(s):
    """Convert a raw mpf to a rational number. Return integers (p, q)
    such that s = p/q exactly."""
    sign, man, exp, bc = s
    if sign:
        man = -man
    if bc == -1:
        raise ValueError("cannot convert %s to a rational number" % man)
    if exp >= 0:
        return man * (1<<exp), 1
    else:
        return man, 1<<(-exp)

def to_fixed(s, prec):
    """Convert a raw mpf to a fixed-point big integer"""
    sign, man, exp, bc = s
    offset = exp + prec
    if sign:
        if offset >= 0: return (-man) << offset
        else:           return (-man) >> (-offset)
    else:
        if offset >= 0: return man << offset
        else:           return man >> (-offset)


##############################################################################
##############################################################################

#----------------------------------------------------------------------------#
#                       Arithmetic operations, etc.                          #
#----------------------------------------------------------------------------#

def mpf_rand(prec):
    """Return a raw mpf chosen randomly from [0, 1), with prec bits
    in the mantissa."""
    global getrandbits
    if not getrandbits:
        import random
        getrandbits = random.getrandbits
    return from_man_exp(getrandbits(prec), -prec, prec, round_floor)

def mpf_eq(s, t):
    """Test equality of two raw mpfs. This is simply tuple comparison
    unless either number is nan, in which case the result is False."""
    if not s[1] or not t[1]:
        if s == fnan or t == fnan:
            return False
    return s == t

def mpf_hash(s):
    # Duplicate the new hash algorithm introduces in Python 3.2.
    if sys.version_info >= (3, 2):
        ssign, sman, sexp, sbc = s

        # Handle special numbers
        if not sman:
            if s == fnan: return sys.hash_info.nan
            if s == finf: return sys.hash_info.inf
            if s == fninf: return -sys.hash_info.inf
        h = sman % HASH_MODULUS
        if sexp >= 0:
            sexp = sexp % HASH_BITS
        else:
            sexp = HASH_BITS - 1 - ((-1 - sexp) % HASH_BITS)
        h = (h << sexp) % HASH_MODULUS
        if ssign: h = -h
        if h == -1: h = -2
        return int(h)
    else:
        try:
            # Try to be compatible with hash values for floats and ints
            return hash(to_float(s, strict=1))
        except OverflowError:
            # We must unfortunately sacrifice compatibility with ints here.
            # We could do hash(man << exp) when the exponent is positive, but
            # this would cause unreasonable inefficiency for large numbers.
            return hash(s)

def mpf_cmp(s, t):
    """Compare the raw mpfs s and t. Return -1 if s < t, 0 if s == t,
    and 1 if s > t. (Same convention as Python's cmp() function.)"""

    # In principle, a comparison amounts to determining the sign of s-t.
    # A full subtraction is relatively slow, however, so we first try to
    # look at the components.
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t

    # Handle zeros and special numbers
    if not sman or not tman:
        if s == fzero: return -mpf_sign(t)
        if t == fzero: return mpf_sign(s)
        if s == t: return 0
        # Follow same convention as Python's cmp for float nan
        if t == fnan: return 1
        if s == finf: return 1
        if t == fninf: return 1
        return -1
    # Different sides of zero
    if ssign != tsign:
        if not ssign: return 1
        return -1
    # This reduces to direct integer comparison
    if sexp == texp:
        if sman == tman:
            return 0
        if sman > tman:
            if ssign: return -1
            else:     return 1
        else:
            if ssign: return 1
            else:     return -1
    # Check position of the highest set bit in each number. If
    # different, there is certainly an inequality.
    a = sbc + sexp
    b = tbc + texp
    if ssign:
        if a < b: return 1
        if a > b: return -1
    else:
        if a < b: return -1
        if a > b: return 1

    # Both numbers have the same highest bit. Subtract to find
    # how the lower bits compare.
    delta = mpf_sub(s, t, 5, round_floor)
    if delta[0]:
        return -1
    return 1

def mpf_lt(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) < 0

def mpf_le(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) <= 0

def mpf_gt(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) > 0

def mpf_ge(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) >= 0

def mpf_min_max(seq):
    min = max = seq[0]
    for x in seq[1:]:
        if mpf_lt(x, min): min = x
        if mpf_gt(x, max): max = x
    return min, max

def mpf_pos(s, prec=0, rnd=round_fast):
    """Calculate 0+s for a raw mpf (i.e., just round s to the specified
    precision)."""
    if prec:
        sign, man, exp, bc = s
        if (not man) and exp:
            return s
        return normalize1(sign, man, exp, bc, prec, rnd)
    return s

def mpf_neg(s, prec=None, rnd=round_fast):
    """Negate a raw mpf (return -s), rounding the result to the
    specified precision. The prec argument can be omitted to do the
    operation exactly."""
    sign, man, exp, bc = s
    if not man:
        if exp:
            if s == finf: return fninf
            if s == fninf: return finf
        return s
    if not prec:
        return (1-sign, man, exp, bc)
    return normalize1(1-sign, man, exp, bc, prec, rnd)

def mpf_abs(s, prec=None, rnd=round_fast):
    """Return abs(s) of the raw mpf s, rounded to the specified
    precision. The prec argument can be omitted to generate an
    exact result."""
    sign, man, exp, bc = s
    if (not man) and exp:
        if s == fninf:
            return finf
        return s
    if not prec:
        if sign:
            return (0, man, exp, bc)
        return s
    return normalize1(0, man, exp, bc, prec, rnd)

def mpf_sign(s):
    """Return -1, 0, or 1 (as a Python int, not a raw mpf) depending on
    whether s is negative, zero, or positive. (Nan is taken to give 0.)"""
    sign, man, exp, bc = s
    if not man:
        if s == finf: return 1
        if s == fninf: return -1
        return 0
    return (-1) ** sign

def mpf_add(s, t, prec=0, rnd=round_fast, _sub=0):
    """
    Add the two raw mpf values s and t.

    With prec=0, no rounding is performed. Note that this can
    produce a very large mantissa (potentially too large to fit
    in memory) if exponents are far apart.
    """
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    tsign ^= _sub
    # Standard case: two nonzero, regular numbers
    if sman and tman:
        offset = sexp - texp
        if offset:
            if offset > 0:
                # Outside precision range; only need to perturb
                if offset > 100 and prec:
                    delta = sbc + sexp - tbc - texp
                    if delta > prec + 4:
                        offset = prec + 4
                        sman <<= offset
                        if tsign == ssign: sman += 1
                        else:              sman -= 1
                        return normalize1(ssign, sman, sexp-offset,
                            bitcount(sman), prec, rnd)
                # Add
                if ssign == tsign:
                    man = tman + (sman << offset)
                # Subtract
                else:
                    if ssign: man = tman - (sman << offset)
                    else:     man = (sman << offset) - tman
                    if man >= 0:
                        ssign = 0
                    else:
                        man = -man
                        ssign = 1
                bc = bitcount(man)
                return normalize1(ssign, man, texp, bc, prec or bc, rnd)
            elif offset < 0:
                # Outside precision range; only need to perturb
                if offset < -100 and prec:
                    delta = tbc + texp - sbc - sexp
                    if delta > prec + 4:
                        offset = prec + 4
                        tman <<= offset
                        if ssign == tsign: tman += 1
                        else:              tman -= 1
                        return normalize1(tsign, tman, texp-offset,
                            bitcount(tman), prec, rnd)
                # Add
                if ssign == tsign:
                    man = sman + (tman << -offset)
                # Subtract
                else:
                    if tsign: man = sman - (tman << -offset)
                    else:     man = (tman << -offset) - sman
                    if man >= 0:
                        ssign = 0
                    else:
                        man = -man
                        ssign = 1
                bc = bitcount(man)
                return normalize1(ssign, man, sexp, bc, prec or bc, rnd)
        # Equal exponents; no shifting necessary
        if ssign == tsign:
            man = tman + sman
        else:
            if ssign: man = tman - sman
            else:     man = sman - tman
            if man >= 0:
                ssign = 0
            else:
                man = -man
                ssign = 1
        bc = bitcount(man)
        return normalize(ssign, man, texp, bc, prec or bc, rnd)
    # Handle zeros and special numbers
    if _sub:
        t = mpf_neg(t)
    if not sman:
        if sexp:
            if s == t or tman or not texp:
                return s
            return fnan
        if tman:
            return normalize1(tsign, tman, texp, tbc, prec or tbc, rnd)
        return t
    if texp:
        return t
    if sman:
        return normalize1(ssign, sman, sexp, sbc, prec or sbc, rnd)
    return s

def mpf_sub(s, t, prec=0, rnd=round_fast):
    """Return the difference of two raw mpfs, s-t. This function is
    simply a wrapper of mpf_add that changes the sign of t."""
    return mpf_add(s, t, prec, rnd, 1)

def mpf_sum(xs, prec=0, rnd=round_fast, absolute=False):
    """
    Sum a list of mpf values efficiently and accurately
    (typically no temporary roundoff occurs). If prec=0,
    the final result will not be rounded either.

    There may be roundoff error or cancellation if extremely
    large exponent differences occur.

    With absolute=True, sums the absolute values.
    """
    man = 0
    exp = 0
    max_extra_prec = prec*2 or 1000000  # XXX
    special = None
    for x in xs:
        xsign, xman, xexp, xbc = x
        if xman:
            if xsign and not absolute:
                xman = -xman
            delta = xexp - exp
            if xexp >= exp:
                # x much larger than existing sum?
                # first: quick test
                if (delta > max_extra_prec) and \
                    ((not man) or delta-bitcount(abs(man)) > max_extra_prec):
                    man = xman
                    exp = xexp
                else:
                    man += (xman << delta)
            else:
                delta = -delta
                # x much smaller than existing sum?
                if delta-xbc > max_extra_prec:
                    if not man:
                        man, exp = xman, xexp
                else:
                    man = (man << delta) + xman
                    exp = xexp
        elif xexp:
            if absolute:
                x = mpf_abs(x)
            special = mpf_add(special or fzero, x, 1)
    # Will be inf or nan
    if special:
        return special
    return from_man_exp(man, exp, prec, rnd)

def gmpy_mpf_mul(s, t, prec=0, rnd=round_fast):
    """Multiply two raw mpfs"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    sign = ssign ^ tsign
    man = sman*tman
    if man:
        bc = bitcount(man)
        if prec:
            return normalize1(sign, man, sexp+texp, bc, prec, rnd)
        else:
            return (sign, man, sexp+texp, bc)
    s_special = (not sman) and sexp
    t_special = (not tman) and texp
    if not s_special and not t_special:
        return fzero
    if fnan in (s, t): return fnan
    if (not tman) and texp: s, t = t, s
    if t == fzero: return fnan
    return {1:finf, -1:fninf}[mpf_sign(s) * mpf_sign(t)]

def gmpy_mpf_mul_int(s, n, prec, rnd=round_fast):
    """Multiply by a Python integer."""
    sign, man, exp, bc = s
    if not man:
        return mpf_mul(s, from_int(n), prec, rnd)
    if not n:
        return fzero
    if n < 0:
        sign ^= 1
        n = -n
    man *= n
    return normalize(sign, man, exp, bitcount(man), prec, rnd)

def python_mpf_mul(s, t, prec=0, rnd=round_fast):
    """Multiply two raw mpfs"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    sign = ssign ^ tsign
    man = sman*tman
    if man:
        bc = sbc + tbc - 1
        bc += int(man>>bc)
        if prec:
            return normalize1(sign, man, sexp+texp, bc, prec, rnd)
        else:
            return (sign, man, sexp+texp, bc)
    s_special = (not sman) and sexp
    t_special = (not tman) and texp
    if not s_special and not t_special:
        return fzero
    if fnan in (s, t): return fnan
    if (not tman) and texp: s, t = t, s
    if t == fzero: return fnan
    return {1:finf, -1:fninf}[mpf_sign(s) * mpf_sign(t)]

def python_mpf_mul_int(s, n, prec, rnd=round_fast):
    """Multiply by a Python integer."""
    sign, man, exp, bc = s
    if not man:
        return mpf_mul(s, from_int(n), prec, rnd)
    if not n:
        return fzero
    if n < 0:
        sign ^= 1
        n = -n
    man *= n
    # Generally n will be small
    if n < 1024:
        bc += bctable[int(n)] - 1
    else:
        bc += bitcount(n) - 1
    bc += int(man>>bc)
    return normalize(sign, man, exp, bc, prec, rnd)


if BACKEND == 'gmpy':
    mpf_mul = gmpy_mpf_mul
    mpf_mul_int = gmpy_mpf_mul_int
else:
    mpf_mul = python_mpf_mul
    mpf_mul_int = python_mpf_mul_int

def mpf_shift(s, n):
    """Quickly multiply the raw mpf s by 2**n without rounding."""
    sign, man, exp, bc = s
    if not man:
        return s
    return sign, man, exp+n, bc

def mpf_frexp(x):
    """Convert x = y*2**n to (y, n) with abs(y) in [0.5, 1) if nonzero"""
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return (fzero, 0)
        else:
            raise ValueError
    return mpf_shift(x, -bc-exp), bc+exp

def mpf_div(s, t, prec, rnd=round_fast):
    """Floating-point division"""
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if not sman or not tman:
        if s == fzero:
            if t == fzero: raise ZeroDivisionError
            if t == fnan: return fnan
            return fzero
        if t == fzero:
            raise ZeroDivisionError
        s_special = (not sman) and sexp
        t_special = (not tman) and texp
        if s_special and t_special:
            return fnan
        if s == fnan or t == fnan:
            return fnan
        if not t_special:
            if t == fzero:
                return fnan
            return {1:finf, -1:fninf}[mpf_sign(s) * mpf_sign(t)]
        return fzero
    sign = ssign ^ tsign
    if tman == 1:
        return normalize1(sign, sman, sexp-texp, sbc, prec, rnd)
    # Same strategy as for addition: if there is a remainder, perturb
    # the result a few bits outside the precision range before rounding
    extra = prec - sbc + tbc + 5
    if extra < 5:
        extra = 5
    quot, rem = divmod(sman<<extra, tman)
    if rem:
        quot = (quot<<1) + 1
        extra += 1
        return normalize1(sign, quot, sexp-texp-extra, bitcount(quot), prec, rnd)
    return normalize(sign, quot, sexp-texp-extra, bitcount(quot), prec, rnd)

def mpf_rdiv_int(n, t, prec, rnd=round_fast):
    """Floating-point division n/t with a Python integer as numerator"""
    sign, man, exp, bc = t
    if not n or not man:
        return mpf_div(from_int(n), t, prec, rnd)
    if n < 0:
        sign ^= 1
        n = -n
    extra = prec + bc + 5
    quot, rem = divmod(n<<extra, man)
    if rem:
        quot = (quot<<1) + 1
        extra += 1
        return normalize1(sign, quot, -exp-extra, bitcount(quot), prec, rnd)
    return normalize(sign, quot, -exp-extra, bitcount(quot), prec, rnd)

def mpf_mod(s, t, prec, rnd=round_fast):
    ssign, sman, sexp, sbc = s
    tsign, tman, texp, tbc = t
    if ((not sman) and sexp) or ((not tman) and texp):
        return fnan
    # Important special case: do nothing if t is larger
    if ssign == tsign and texp > sexp+sbc:
        return s
    # Another important special case: this allows us to do e.g. x % 1.0
    # to find the fractional part of x, and it will work when x is huge.
    if tman == 1 and sexp > texp+tbc:
        return fzero
    base = min(sexp, texp)
    sman = (-1)**ssign * sman
    tman = (-1)**tsign * tman
    man = (sman << (sexp-base)) % (tman << (texp-base))
    if man >= 0:
        sign = 0
    else:
        man = -man
        sign = 1
    return normalize(sign, man, base, bitcount(man), prec, rnd)

reciprocal_rnd = {
  round_down : round_up,
  round_up : round_down,
  round_floor : round_ceiling,
  round_ceiling : round_floor,
  round_nearest : round_nearest
}

negative_rnd = {
  round_down : round_down,
  round_up : round_up,
  round_floor : round_ceiling,
  round_ceiling : round_floor,
  round_nearest : round_nearest
}

def mpf_pow_int(s, n, prec, rnd=round_fast):
    """Compute s**n, where s is a raw mpf and n is a Python integer."""
    sign, man, exp, bc = s

    if (not man) and exp:
        if s == finf:
            if n > 0: return s
            if n == 0: return fnan
            return fzero
        if s == fninf:
            if n > 0: return [finf, fninf][n & 1]
            if n == 0: return fnan
            return fzero
        return fnan

    n = int(n)
    if n == 0: return fone
    if n == 1: return mpf_pos(s, prec, rnd)
    if n == 2:
        _, man, exp, bc = s
        if not man:
            return fzero
        man = man*man
        if man == 1:
            return (0, MPZ_ONE, exp+exp, 1)
        bc = bc + bc - 2
        bc += bctable[int(man>>bc)]
        return normalize1(0, man, exp+exp, bc, prec, rnd)
    if n == -1: return mpf_div(fone, s, prec, rnd)
    if n < 0:
        inverse = mpf_pow_int(s, -n, prec+5, reciprocal_rnd[rnd])
        return mpf_div(fone, inverse, prec, rnd)

    result_sign = sign & n

    # Use exact integer power when the exact mantissa is small
    if man == 1:
        return (result_sign, MPZ_ONE, exp*n, 1)
    if bc*n < 1000:
        man **= n
        return normalize1(result_sign, man, exp*n, bitcount(man), prec, rnd)

    # Use directed rounding all the way through to maintain rigorous
    # bounds for interval arithmetic
    rounds_down = (rnd == round_nearest) or \
        shifts_down[rnd][result_sign]

    # Now we perform binary exponentiation. Need to estimate precision
    # to avoid rounding errors from temporary operations. Roughly log_2(n)
    # operations are performed.
    workprec = prec + 4*bitcount(n) + 4
    _, pm, pe, pbc = fone
    while 1:
        if n & 1:
            pm = pm*man
            pe = pe+exp
            pbc += bc - 2
            pbc = pbc + bctable[int(pm >> pbc)]
            if pbc > workprec:
                if rounds_down:
                    pm = pm >> (pbc-workprec)
                else:
                    pm = -((-pm) >> (pbc-workprec))
                pe += pbc - workprec
                pbc = workprec
            n -= 1
            if not n:
                break
        man = man*man
        exp = exp+exp
        bc = bc + bc - 2
        bc = bc + bctable[int(man >> bc)]
        if bc > workprec:
            if rounds_down:
                man = man >> (bc-workprec)
            else:
                man = -((-man) >> (bc-workprec))
            exp += bc - workprec
            bc = workprec
        n = n // 2

    return normalize(result_sign, pm, pe, pbc, prec, rnd)


def mpf_perturb(x, eps_sign, prec, rnd):
    """
    For nonzero x, calculate x + eps with directed rounding, where
    eps < prec relatively and eps has the given sign (0 for
    positive, 1 for negative).

    With rounding to nearest, this is taken to simply normalize
    x to the given precision.
    """
    if rnd == round_nearest:
        return mpf_pos(x, prec, rnd)
    sign, man, exp, bc = x
    eps = (eps_sign, MPZ_ONE, exp+bc-prec-1, 1)
    if sign:
        away = (rnd in (round_down, round_ceiling)) ^ eps_sign
    else:
        away = (rnd in (round_up, round_ceiling)) ^ eps_sign
    if away:
        return mpf_add(x, eps, prec, rnd)
    else:
        return mpf_pos(x, prec, rnd)


#----------------------------------------------------------------------------#
#                              Radix conversion                              #
#----------------------------------------------------------------------------#

def to_digits_exp(s, dps):
    """Helper function for representing the floating-point number s as
    a decimal with dps digits. Returns (sign, string, exponent) where
    sign is '' or '-', string is the digit string, and exponent is
    the decimal exponent as an int.

    If inexact, the decimal representation is rounded toward zero."""

    # Extract sign first so it doesn't mess up the string digit count
    if s[0]:
        sign = '-'
        s = mpf_neg(s)
    else:
        sign = ''
    _sign, man, exp, bc = s

    if not man:
        return '', '0', 0

    bitprec = int(dps * math.log(10,2)) + 10

    # Cut down to size
    # TODO: account for precision when doing this
    exp_from_1 = exp + bc
    if abs(exp_from_1) > 3500:
        from .libelefun import mpf_ln2, mpf_ln10
        # Set b = int(exp * log(2)/log(10))
        # If exp is huge, we must use high-precision arithmetic to
        # find the nearest power of ten
        expprec = bitcount(abs(exp)) + 5
        tmp = from_int(exp)
        tmp = mpf_mul(tmp, mpf_ln2(expprec))
        tmp = mpf_div(tmp, mpf_ln10(expprec), expprec)
        b = to_int(tmp)
        s = mpf_div(s, mpf_pow_int(ften, b, bitprec), bitprec)
        _sign, man, exp, bc = s
        exponent = b
    else:
        exponent = 0

    # First, calculate mantissa digits by converting to a binary
    # fixed-point number and then converting that number to
    # a decimal fixed-point number.
    fixprec = max(bitprec - exp - bc, 0)
    fixdps = int(fixprec / math.log(10,2) + 0.5)
    sf = to_fixed(s, fixprec)
    sd = bin_to_radix(sf, fixprec, 10, fixdps)
    digits = numeral(sd, base=10, size=dps)

    exponent += len(digits) - fixdps - 1
    return sign, digits, exponent

def to_str(s, dps, strip_zeros=True, min_fixed=None, max_fixed=None,
    show_zero_exponent=False):
    """
    Convert a raw mpf to a decimal floating-point literal with at
    most `dps` decimal digits in the mantissa (not counting extra zeros
    that may be inserted for visual purposes).

    The number will be printed in fixed-point format if the position
    of the leading digit is strictly between min_fixed
    (default = min(-dps/3,-5)) and max_fixed (default = dps).

    To force fixed-point format always, set min_fixed = -inf,
    max_fixed = +inf. To force floating-point format, set
    min_fixed >= max_fixed.

    The literal is formatted so that it can be parsed back to a number
    by to_str, float() or Decimal().
    """

    # Special numbers
    if not s[1]:
        if s == fzero:
            if dps: t = '0.0'
            else:   t = '.0'
            if show_zero_exponent:
                t += 'e+0'
            return t
        if s == finf: return '+inf'
        if s == fninf: return '-inf'
        if s == fnan: return 'nan'
        raise ValueError

    if min_fixed is None: min_fixed = min(-(dps//3), -5)
    if max_fixed is None: max_fixed = dps

    # to_digits_exp rounds to floor.
    # This sometimes kills some instances of "...00001"
    sign, digits, exponent = to_digits_exp(s, dps+3)

    # No digits: show only .0; round exponent to nearest
    if not dps:
        if digits[0] in '56789':
            exponent += 1
        digits = ".0"

    else:
        # Rounding up kills some instances of "...99999"
        if len(digits) > dps and digits[dps] in '56789':
            digits = digits[:dps]
            i = dps - 1
            while i >= 0 and digits[i] == '9':
                i -= 1
            if i >= 0:
                digits = digits[:i] + str(int(digits[i]) + 1) + '0' * (dps - i - 1)
            else:
                digits = '1' + '0' * (dps - 1)
                exponent += 1
        else:
            digits = digits[:dps]

        # Prettify numbers close to unit magnitude
        if min_fixed < exponent < max_fixed:
            if exponent < 0:
                digits = ("0"*int(-exponent)) + digits
                split = 1
            else:
                split = exponent + 1
                if split > dps:
                    digits += "0"*(split-dps)
            exponent = 0
        else:
            split = 1

        digits = (digits[:split] + "." + digits[split:])

        if strip_zeros:
            # Clean up trailing zeros
            digits = digits.rstrip('0')
            if digits[-1] == ".":
                digits += "0"

    if exponent == 0 and dps and not show_zero_exponent: return sign + digits
    if exponent >= 0: return sign + digits + "e+" + str(exponent)
    if exponent < 0: return sign + digits + "e" + str(exponent)

def str_to_man_exp(x, base=10):
    """Helper function for from_str."""
    x = x.lower().rstrip('l')
    # Verify that the input is a valid float literal
    float(x)
    # Split into mantissa, exponent
    parts = x.split('e')
    if len(parts) == 1:
        exp = 0
    else: # == 2
        x = parts[0]
        exp = int(parts[1])
    # Look for radix point in mantissa
    parts = x.split('.')
    if len(parts) == 2:
        a, b = parts[0], parts[1].rstrip('0')
        exp -= len(b)
        x = a + b
    x = MPZ(int(x, base))
    return x, exp

special_str = {'inf':finf, '+inf':finf, '-inf':fninf, 'nan':fnan}

def from_str(x, prec, rnd=round_fast):
    """Create a raw mpf from a decimal literal, rounding in the
    specified direction if the input number cannot be represented
    exactly as a binary floating-point number with the given number of
    bits. The literal syntax accepted is the same as for Python
    floats.

    TODO: the rounding does not work properly for large exponents.
    """
    x = x.lower().strip()
    if x in special_str:
        return special_str[x]

    if '/' in x:
        p, q = x.split('/')
        p, q = p.rstrip('l'), q.rstrip('l')
        return from_rational(int(p), int(q), prec, rnd)

    man, exp = str_to_man_exp(x, base=10)

    # XXX: appropriate cutoffs & track direction
    # note no factors of 5
    if abs(exp) > 400:
        s = from_int(man, prec+10)
        s = mpf_mul(s, mpf_pow_int(ften, exp, prec+10), prec, rnd)
    else:
        if exp >= 0:
            s = from_int(man * 10**exp, prec, rnd)
        else:
            s = from_rational(man, 10**-exp, prec, rnd)
    return s

# Binary string conversion. These are currently mainly used for debugging
# and could use some improvement in the future

def from_bstr(x):
    man, exp = str_to_man_exp(x, base=2)
    man = MPZ(man)
    sign = 0
    if man < 0:
        man = -man
        sign = 1
    bc = bitcount(man)
    return normalize(sign, man, exp, bc, bc, round_floor)

def to_bstr(x):
    sign, man, exp, bc = x
    return ['','-'][sign] + numeral(man, size=bitcount(man), base=2) + ("e%i" % exp)


#----------------------------------------------------------------------------#
#                                Square roots                                #
#----------------------------------------------------------------------------#


def mpf_sqrt(s, prec, rnd=round_fast):
    """
    Compute the square root of a nonnegative mpf value. The
    result is correctly rounded.
    """
    sign, man, exp, bc = s
    if sign:
        raise ComplexResult("square root of a negative number")
    if not man:
        return s
    if exp & 1:
        exp -= 1
        man <<= 1
        bc += 1
    elif man == 1:
        return normalize1(sign, man, exp//2, bc, prec, rnd)
    shift = max(4, 2*prec-bc+4)
    shift += shift & 1
    if rnd in 'fd':
        man = isqrt(man<<shift)
    else:
        man, rem = sqrtrem(man<<shift)
        # Perturb up
        if rem:
            man = (man<<1)+1
            shift += 2
    return from_man_exp(man, (exp-shift)//2, prec, rnd)

def mpf_hypot(x, y, prec, rnd=round_fast):
    """Compute the Euclidean norm sqrt(x**2 + y**2) of two raw mpfs
    x and y."""
    if y == fzero: return mpf_abs(x, prec, rnd)
    if x == fzero: return mpf_abs(y, prec, rnd)
    hypot2 = mpf_add(mpf_mul(x,x), mpf_mul(y,y), prec+4)
    return mpf_sqrt(hypot2, prec, rnd)


if BACKEND == 'sage':
    try:
        import sage.libs.mpmath.ext_libmp as ext_lib
        mpf_add = ext_lib.mpf_add
        mpf_sub = ext_lib.mpf_sub
        mpf_mul = ext_lib.mpf_mul
        mpf_div = ext_lib.mpf_div
        mpf_sqrt = ext_lib.mpf_sqrt
    except ImportError:
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\hypergeometric.py ===
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped

def _check_need_perturb(ctx, terms, prec, discard_known_zeros):
    perturb = recompute = False
    extraprec = 0
    discard = []
    for term_index, term in enumerate(terms):
        w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term
        have_singular_nongamma_weight = False
        # Avoid division by zero in leading factors (TODO:
        # also check for near division by zero?)
        for k, w in enumerate(w_s):
            if not w:
                if ctx.re(c_s[k]) <= 0 and c_s[k]:
                    perturb = recompute = True
                    have_singular_nongamma_weight = True
        pole_count = [0, 0, 0]
        # Check for gamma and series poles and near-poles
        for data_index, data in enumerate([alpha_s, beta_s, b_s]):
            for i, x in enumerate(data):
                n, d = ctx.nint_distance(x)
                # Poles
                if n > 0:
                    continue
                if d == ctx.ninf:
                    # OK if we have a polynomial
                    # ------------------------------
                    ok = False
                    if data_index == 2:
                        for u in a_s:
                            if ctx.isnpint(u) and u >= int(n):
                                ok = True
                                break
                    if ok:
                        continue
                    pole_count[data_index] += 1
                    # ------------------------------
                    #perturb = recompute = True
                    #return perturb, recompute, extraprec
                elif d < -4:
                    extraprec += -d
                    recompute = True
        if discard_known_zeros and pole_count[1] > pole_count[0] + pole_count[2] \
            and not have_singular_nongamma_weight:
            discard.append(term_index)
        elif sum(pole_count):
            perturb = recompute = True
    return perturb, recompute, extraprec, discard

_hypercomb_msg = """
hypercomb() failed to converge to the requested %i bits of accuracy
using a working precision of %i bits. The function value may be zero or
infinite; try passing zeroprec=N or infprec=M to bound finite values between
2^(-N) and 2^M. Otherwise try a higher maxprec or maxterms.
"""

@defun
def hypercomb(ctx, function, params=[], discard_known_zeros=True, **kwargs):
    orig = ctx.prec
    sumvalue = ctx.zero
    dist = ctx.nint_distance
    ninf = ctx.ninf
    orig_params = params[:]
    verbose = kwargs.get('verbose', False)
    maxprec = kwargs.get('maxprec', ctx._default_hyper_maxprec(orig))
    kwargs['maxprec'] = maxprec   # For calls to hypsum
    zeroprec = kwargs.get('zeroprec')
    infprec = kwargs.get('infprec')
    perturbed_reference_value = None
    hextra = 0
    try:
        while 1:
            ctx.prec += 10
            if ctx.prec > maxprec:
                raise ValueError(_hypercomb_msg % (orig, ctx.prec))
            orig2 = ctx.prec
            params = orig_params[:]
            terms = function(*params)
            if verbose:
                print()
                print("ENTERING hypercomb main loop")
                print("prec =", ctx.prec)
                print("hextra", hextra)
            perturb, recompute, extraprec, discard = \
                _check_need_perturb(ctx, terms, orig, discard_known_zeros)
            ctx.prec += extraprec
            if perturb:
                if "hmag" in kwargs:
                    hmag = kwargs["hmag"]
                elif ctx._fixed_precision:
                    hmag = int(ctx.prec*0.3)
                else:
                    hmag = orig + 10 + hextra
                h = ctx.ldexp(ctx.one, -hmag)
                ctx.prec = orig2 + 10 + hmag + 10
                for k in range(len(params)):
                    params[k] += h
                    # Heuristically ensure that the perturbations
                    # are "independent" so that two perturbations
                    # don't accidentally cancel each other out
                    # in a subtraction.
                    h += h/(k+1)
            if recompute:
                terms = function(*params)
            if discard_known_zeros:
                terms = [term for (i, term) in enumerate(terms) if i not in discard]
            if not terms:
                return ctx.zero
            evaluated_terms = []
            for term_index, term_data in enumerate(terms):
                w_s, c_s, alpha_s, beta_s, a_s, b_s, z = term_data
                if verbose:
                    print()
                    print("  Evaluating term %i/%i : %iF%i" % \
                        (term_index+1, len(terms), len(a_s), len(b_s)))
                    print("    powers", ctx.nstr(w_s), ctx.nstr(c_s))
                    print("    gamma", ctx.nstr(alpha_s), ctx.nstr(beta_s))
                    print("    hyper", ctx.nstr(a_s), ctx.nstr(b_s))
                    print("    z", ctx.nstr(z))
                #v = ctx.hyper(a_s, b_s, z, **kwargs)
                #for a in alpha_s: v *= ctx.gamma(a)
                #for b in beta_s: v *= ctx.rgamma(b)
                #for w, c in zip(w_s, c_s): v *= ctx.power(w, c)
                v = ctx.fprod([ctx.hyper(a_s, b_s, z, **kwargs)] + \
                    [ctx.gamma(a) for a in alpha_s] + \
                    [ctx.rgamma(b) for b in beta_s] + \
                    [ctx.power(w,c) for (w,c) in zip(w_s,c_s)])
                if verbose:
                    print("    Value:", v)
                evaluated_terms.append(v)

            if len(terms) == 1 and (not perturb):
                sumvalue = evaluated_terms[0]
                break

            if ctx._fixed_precision:
                sumvalue = ctx.fsum(evaluated_terms)
                break

            sumvalue = ctx.fsum(evaluated_terms)
            term_magnitudes = [ctx.mag(x) for x in evaluated_terms]
            max_magnitude = max(term_magnitudes)
            sum_magnitude = ctx.mag(sumvalue)
            cancellation = max_magnitude - sum_magnitude
            if verbose:
                print()
                print("  Cancellation:", cancellation, "bits")
                print("  Increased precision:", ctx.prec - orig, "bits")

            precision_ok = cancellation < ctx.prec - orig

            if zeroprec is None:
                zero_ok = False
            else:
                zero_ok = max_magnitude - ctx.prec < -zeroprec
            if infprec is None:
                inf_ok = False
            else:
                inf_ok = max_magnitude > infprec

            if precision_ok and (not perturb) or ctx.isnan(cancellation):
                break
            elif precision_ok:
                if perturbed_reference_value is None:
                    hextra += 20
                    perturbed_reference_value = sumvalue
                    continue
                elif ctx.mag(sumvalue - perturbed_reference_value) <= \
                        ctx.mag(sumvalue) - orig:
                    break
                elif zero_ok:
                    sumvalue = ctx.zero
                    break
                elif inf_ok:
                    sumvalue = ctx.inf
                    break
                elif 'hmag' in kwargs:
                    break
                else:
                    hextra *= 2
                    perturbed_reference_value = sumvalue
            # Increase precision
            else:
                increment = min(max(cancellation, orig//2), max(extraprec,orig))
                ctx.prec += increment
                if verbose:
                    print("  Must start over with increased precision")
                continue
    finally:
        ctx.prec = orig
    return +sumvalue

@defun
def hyper(ctx, a_s, b_s, z, **kwargs):
    """
    Hypergeometric function, general case.
    """
    z = ctx.convert(z)
    p = len(a_s)
    q = len(b_s)
    a_s = [ctx._convert_param(a) for a in a_s]
    b_s = [ctx._convert_param(b) for b in b_s]
    # Reduce degree by eliminating common parameters
    if kwargs.get('eliminate', True):
        elim_nonpositive = kwargs.get('eliminate_all', False)
        i = 0
        while i < q and a_s:
            b = b_s[i]
            if b in a_s and (elim_nonpositive or not ctx.isnpint(b[0])):
                a_s.remove(b)
                b_s.remove(b)
                p -= 1
                q -= 1
            else:
                i += 1
    # Handle special cases
    if p == 0:
        if   q == 1: return ctx._hyp0f1(b_s, z, **kwargs)
        elif q == 0: return ctx.exp(z)
    elif p == 1:
        if   q == 1: return ctx._hyp1f1(a_s, b_s, z, **kwargs)
        elif q == 2: return ctx._hyp1f2(a_s, b_s, z, **kwargs)
        elif q == 0: return ctx._hyp1f0(a_s[0][0], z)
    elif p == 2:
        if   q == 1: return ctx._hyp2f1(a_s, b_s, z, **kwargs)
        elif q == 2: return ctx._hyp2f2(a_s, b_s, z, **kwargs)
        elif q == 3: return ctx._hyp2f3(a_s, b_s, z, **kwargs)
        elif q == 0: return ctx._hyp2f0(a_s, b_s, z, **kwargs)
    elif p == q+1:
        return ctx._hypq1fq(p, q, a_s, b_s, z, **kwargs)
    elif p > q+1 and not kwargs.get('force_series'):
        return ctx._hyp_borel(p, q, a_s, b_s, z, **kwargs)
    coeffs, types = zip(*(a_s+b_s))
    return ctx.hypsum(p, q, types, coeffs, z, **kwargs)

@defun
def hyp0f1(ctx,b,z,**kwargs):
    return ctx.hyper([],[b],z,**kwargs)

@defun
def hyp1f1(ctx,a,b,z,**kwargs):
    return ctx.hyper([a],[b],z,**kwargs)

@defun
def hyp1f2(ctx,a1,b1,b2,z,**kwargs):
    return ctx.hyper([a1],[b1,b2],z,**kwargs)

@defun
def hyp2f1(ctx,a,b,c,z,**kwargs):
    return ctx.hyper([a,b],[c],z,**kwargs)

@defun
def hyp2f2(ctx,a1,a2,b1,b2,z,**kwargs):
    return ctx.hyper([a1,a2],[b1,b2],z,**kwargs)

@defun
def hyp2f3(ctx,a1,a2,b1,b2,b3,z,**kwargs):
    return ctx.hyper([a1,a2],[b1,b2,b3],z,**kwargs)

@defun
def hyp2f0(ctx,a,b,z,**kwargs):
    return ctx.hyper([a,b],[],z,**kwargs)

@defun
def hyp3f2(ctx,a1,a2,a3,b1,b2,z,**kwargs):
    return ctx.hyper([a1,a2,a3],[b1,b2],z,**kwargs)

@defun_wrapped
def _hyp1f0(ctx, a, z):
    return (1-z) ** (-a)

@defun
def _hyp0f1(ctx, b_s, z, **kwargs):
    (b, btype), = b_s
    if z:
        magz = ctx.mag(z)
    else:
        magz = 0
    if magz >= 8 and not kwargs.get('force_series'):
        try:
            # http://functions.wolfram.com/HypergeometricFunctions/
            # Hypergeometric0F1/06/02/03/0004/
            # TODO: handle the all-real case more efficiently!
            # TODO: figure out how much precision is needed (exponential growth)
            orig = ctx.prec
            try:
                ctx.prec += 12 + magz//2
                def h():
                    w = ctx.sqrt(-z)
                    jw = ctx.j*w
                    u = 1/(4*jw)
                    c = ctx.mpq_1_2 - b
                    E = ctx.exp(2*jw)
                    T1 = ([-jw,E], [c,-1], [], [], [b-ctx.mpq_1_2, ctx.mpq_3_2-b], [], -u)
                    T2 = ([jw,E], [c,1], [], [], [b-ctx.mpq_1_2, ctx.mpq_3_2-b], [], u)
                    return T1, T2
                v = ctx.hypercomb(h, [], force_series=True)
                v = ctx.gamma(b)/(2*ctx.sqrt(ctx.pi))*v
            finally:
                ctx.prec = orig
            if ctx._is_real_type(b) and ctx._is_real_type(z):
                v = ctx._re(v)
            return +v
        except ctx.NoConvergence:
            pass
    return ctx.hypsum(0, 1, (btype,), [b], z, **kwargs)

@defun
def _hyp1f1(ctx, a_s, b_s, z, **kwargs):
    (a, atype), = a_s
    (b, btype), = b_s
    if not z:
        return ctx.one+z
    magz = ctx.mag(z)
    if magz >= 7 and not (ctx.isint(a) and ctx.re(a) <= 0):
        if ctx.isinf(z):
            if ctx.sign(a) == ctx.sign(b) == ctx.sign(z) == 1:
                return ctx.inf
            return ctx.nan * z
        try:
            try:
                ctx.prec += magz
                sector = ctx._im(z) < 0
                def h(a,b):
                    if sector:
                        E = ctx.expjpi(ctx.fneg(a, exact=True))
                    else:
                        E = ctx.expjpi(a)
                    rz = 1/z
                    T1 = ([E,z], [1,-a], [b], [b-a], [a, 1+a-b], [], -rz)
                    T2 = ([ctx.exp(z),z], [1,a-b], [b], [a], [b-a, 1-a], [], rz)
                    return T1, T2
                v = ctx.hypercomb(h, [a,b], force_series=True)
                if ctx._is_real_type(a) and ctx._is_real_type(b) and ctx._is_real_type(z):
                    v = ctx._re(v)
                return +v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec -= magz
    v = ctx.hypsum(1, 1, (atype, btype), [a, b], z, **kwargs)
    return v

def _hyp2f1_gosper(ctx,a,b,c,z,**kwargs):
    # Use Gosper's recurrence
    # See http://www.math.utexas.edu/pipermail/maxima/2006/000126.html
    _a,_b,_c,_z = a, b, c, z
    orig = ctx.prec
    maxprec = kwargs.get('maxprec', 100*orig)
    extra = 10
    while 1:
        ctx.prec = orig + extra
        #a = ctx.convert(_a)
        #b = ctx.convert(_b)
        #c = ctx.convert(_c)
        z = ctx.convert(_z)
        d = ctx.mpf(0)
        e = ctx.mpf(1)
        f = ctx.mpf(0)
        k = 0
        # Common subexpression elimination, unfortunately making
        # things a bit unreadable. The formula is quite messy to begin
        # with, though...
        abz = a*b*z
        ch = c * ctx.mpq_1_2
        c1h = (c+1) * ctx.mpq_1_2
        nz = 1-z
        g = z/nz
        abg = a*b*g
        cba = c-b-a
        z2 = z-2
        tol = -ctx.prec - 10
        nstr = ctx.nstr
        nprint = ctx.nprint
        mag = ctx.mag
        maxmag = ctx.ninf
        while 1:
            kch = k+ch
            kakbz = (k+a)*(k+b)*z / (4*(k+1)*kch*(k+c1h))
            d1 = kakbz*(e-(k+cba)*d*g)
            e1 = kakbz*(d*abg+(k+c)*e)
            ft = d*(k*(cba*z+k*z2-c)-abz)/(2*kch*nz)
            f1 = f + e - ft
            maxmag = max(maxmag, mag(f1))
            if mag(f1-f) < tol:
                break
            d, e, f = d1, e1, f1
            k += 1
        cancellation = maxmag - mag(f1)
        if cancellation < extra:
            break
        else:
            extra += cancellation
            if extra > maxprec:
                raise ctx.NoConvergence
    return f1

@defun
def _hyp2f1(ctx, a_s, b_s, z, **kwargs):
    (a, atype), (b, btype) = a_s
    (c, ctype), = b_s
    if z == 1:
        # TODO: the following logic can be simplified
        convergent = ctx.re(c-a-b) > 0
        finite = (ctx.isint(a) and a <= 0) or (ctx.isint(b) and b <= 0)
        zerodiv = ctx.isint(c) and c <= 0 and not \
            ((ctx.isint(a) and c <= a <= 0) or (ctx.isint(b) and c <= b <= 0))
        #print "bz", a, b, c, z, convergent, finite, zerodiv
        # Gauss's theorem gives the value if convergent
        if (convergent or finite) and not zerodiv:
            return ctx.gammaprod([c, c-a-b], [c-a, c-b], _infsign=True)
        # Otherwise, there is a pole and we take the
        # sign to be that when approaching from below
        # XXX: this evaluation is not necessarily correct in all cases
        return ctx.hyp2f1(a,b,c,1-ctx.eps*2) * ctx.inf

    # Equal to 1 (first term), unless there is a subsequent
    # division by zero
    if not z:
        # Division by zero but power of z is higher than
        # first order so cancels
        if c or a == 0 or b == 0:
            return 1+z
        # Indeterminate
        return ctx.nan

    # Hit zero denominator unless numerator goes to 0 first
    if ctx.isint(c) and c <= 0:
        if (ctx.isint(a) and c <= a <= 0) or \
           (ctx.isint(b) and c <= b <= 0):
            pass
        else:
            # Pole in series
            return ctx.inf

    absz = abs(z)

    # Fast case: standard series converges rapidly,
    # possibly in finitely many terms
    if absz <= 0.8 or (ctx.isint(a) and a <= 0 and a >= -1000) or \
                      (ctx.isint(b) and b <= 0 and b >= -1000):
        return ctx.hypsum(2, 1, (atype, btype, ctype), [a, b, c], z, **kwargs)

    orig = ctx.prec
    try:
        ctx.prec += 10

        # Use 1/z transformation
        if absz >= 1.3:
            def h(a,b):
                t = ctx.mpq_1-c; ab = a-b; rz = 1/z
                T1 = ([-z],[-a], [c,-ab],[b,c-a], [a,t+a],[ctx.mpq_1+ab],  rz)
                T2 = ([-z],[-b], [c,ab],[a,c-b], [b,t+b],[ctx.mpq_1-ab],  rz)
                return T1, T2
            v = ctx.hypercomb(h, [a,b], **kwargs)

        # Use 1-z transformation
        elif abs(1-z) <= 0.75:
            def h(a,b):
                t = c-a-b; ca = c-a; cb = c-b; rz = 1-z
                T1 = [], [], [c,t], [ca,cb], [a,b], [1-t], rz
                T2 = [rz], [t], [c,a+b-c], [a,b], [ca,cb], [1+t], rz
                return T1, T2
            v = ctx.hypercomb(h, [a,b], **kwargs)

        # Use z/(z-1) transformation
        elif abs(z/(z-1)) <= 0.75:
            v = ctx.hyp2f1(a, c-b, c, z/(z-1)) / (1-z)**a

        # Remaining part of unit circle
        else:
            v = _hyp2f1_gosper(ctx,a,b,c,z,**kwargs)

    finally:
        ctx.prec = orig
    return +v

@defun
def _hypq1fq(ctx, p, q, a_s, b_s, z, **kwargs):
    r"""
    Evaluates 3F2, 4F3, 5F4, ...
    """
    a_s, a_types = zip(*a_s)
    b_s, b_types = zip(*b_s)
    a_s = list(a_s)
    b_s = list(b_s)
    absz = abs(z)
    ispoly = False
    for a in a_s:
        if ctx.isint(a) and a <= 0:
            ispoly = True
            break
    # Direct summation
    if absz < 1 or ispoly:
        try:
            return ctx.hypsum(p, q, a_types+b_types, a_s+b_s, z, **kwargs)
        except ctx.NoConvergence:
            if absz > 1.1 or ispoly:
                raise
    # Use expansion at |z-1| -> 0.
    # Reference: Wolfgang Buhring, "Generalized Hypergeometric Functions at
    #   Unit Argument", Proc. Amer. Math. Soc., Vol. 114, No. 1 (Jan. 1992),
    #   pp.145-153
    # The current implementation has several problems:
    # 1. We only implement it for 3F2. The expansion coefficients are
    #    given by extremely messy nested sums in the higher degree cases
    #    (see reference). Is efficient sequential generation of the coefficients
    #    possible in the > 3F2 case?
    # 2. Although the series converges, it may do so slowly, so we need
    #    convergence acceleration. The acceleration implemented by
    #    nsum does not always help, so results returned are sometimes
    #    inaccurate! Can we do better?
    # 3. We should check conditions for convergence, and possibly
    #    do a better job of cancelling out gamma poles if possible.
    if z == 1:
        # XXX: should also check for division by zero in the
        # denominator of the series (cf. hyp2f1)
        S = ctx.re(sum(b_s)-sum(a_s))
        if S <= 0:
            #return ctx.hyper(a_s, b_s, 1-ctx.eps*2, **kwargs) * ctx.inf
            return ctx.hyper(a_s, b_s, 0.9, **kwargs) * ctx.inf
    if (p,q) == (3,2) and abs(z-1) < 0.05:   # and kwargs.get('sum1')
        #print "Using alternate summation (experimental)"
        a1,a2,a3 = a_s
        b1,b2 = b_s
        u = b1+b2-a3
        initial = ctx.gammaprod([b2-a3,b1-a3,a1,a2],[b2-a3,b1-a3,1,u])
        def term(k, _cache={0:initial}):
            u = b1+b2-a3+k
            if k in _cache:
                t = _cache[k]
            else:
                t = _cache[k-1]
                t *= (b1+k-a3-1)*(b2+k-a3-1)
                t /= k*(u-1)
                _cache[k] = t
            return t * ctx.hyp2f1(a1,a2,u,z)
        try:
            S = ctx.nsum(term, [0,ctx.inf], verbose=kwargs.get('verbose'),
                strict=kwargs.get('strict', True))
            return S * ctx.gammaprod([b1,b2],[a1,a2,a3])
        except ctx.NoConvergence:
            pass
    # Try to use convergence acceleration on and close to the unit circle.
    # Problem: the convergence acceleration degenerates as |z-1| -> 0,
    # except for special cases. Everywhere else, the Shanks transformation
    # is very efficient.
    if absz < 1.1 and ctx._re(z) <= 1:

        def term(kk, _cache={0:ctx.one}):
            k = int(kk)
            if k != kk:
                t = z ** ctx.mpf(kk) / ctx.fac(kk)
                for a in a_s: t *= ctx.rf(a,kk)
                for b in b_s: t /= ctx.rf(b,kk)
                return t
            if k in _cache:
                return _cache[k]
            t = term(k-1)
            m = k-1
            for j in xrange(p): t *= (a_s[j]+m)
            for j in xrange(q): t /= (b_s[j]+m)
            t *= z
            t /= k
            _cache[k] = t
            return t

        sum_method = kwargs.get('sum_method', 'r+s+e')

        try:
            return ctx.nsum(term, [0,ctx.inf], verbose=kwargs.get('verbose'),
                strict=kwargs.get('strict', True),
                method=sum_method.replace('e',''))
        except ctx.NoConvergence:
            if 'e' not in sum_method:
                raise
            pass

        if kwargs.get('verbose'):
            print("Attempting Euler-Maclaurin summation")


        """
        Somewhat slower version (one diffs_exp for each factor).
        However, this would be faster with fast direct derivatives
        of the gamma function.

        def power_diffs(k0):
            r = 0
            l = ctx.log(z)
            while 1:
                yield z**ctx.mpf(k0) * l**r
                r += 1

        def loggamma_diffs(x, reciprocal=False):
            sign = (-1) ** reciprocal
            yield sign * ctx.loggamma(x)
            i = 0
            while 1:
                yield sign * ctx.psi(i,x)
                i += 1

        def hyper_diffs(k0):
            b2 = b_s + [1]
            A = [ctx.diffs_exp(loggamma_diffs(a+k0)) for a in a_s]
            B = [ctx.diffs_exp(loggamma_diffs(b+k0,True)) for b in b2]
            Z = [power_diffs(k0)]
            C = ctx.gammaprod([b for b in b2], [a for a in a_s])
            for d in ctx.diffs_prod(A + B + Z):
                v = C * d
                yield v
        """

        def log_diffs(k0):
            b2 = b_s + [1]
            yield sum(ctx.loggamma(a+k0) for a in a_s) - \
                sum(ctx.loggamma(b+k0) for b in b2) + k0*ctx.log(z)
            i = 0
            while 1:
                v = sum(ctx.psi(i,a+k0) for a in a_s) - \
                    sum(ctx.psi(i,b+k0) for b in b2)
                if i == 0:
                    v += ctx.log(z)
                yield v
                i += 1

        def hyper_diffs(k0):
            C = ctx.gammaprod([b for b in b_s], [a for a in a_s])
            for d in ctx.diffs_exp(log_diffs(k0)):
                v = C * d
                yield v

        tol = ctx.eps / 1024
        prec = ctx.prec
        try:
            trunc = 50 * ctx.dps
            ctx.prec += 20
            for i in xrange(5):
                head = ctx.fsum(term(k) for k in xrange(trunc))
                tail, err = ctx.sumem(term, [trunc, ctx.inf], tol=tol,
                    adiffs=hyper_diffs(trunc),
                    verbose=kwargs.get('verbose'),
                    error=True,
                    _fast_abort=True)
                if err < tol:
                    v = head + tail
                    break
                trunc *= 2
                # Need to increase precision because calculation of
                # derivatives may be inaccurate
                ctx.prec += ctx.prec//2
                if i == 4:
                    raise ctx.NoConvergence(\
                        "Euler-Maclaurin summation did not converge")
        finally:
            ctx.prec = prec
        return +v

    # Use 1/z transformation
    # http://functions.wolfram.com/HypergeometricFunctions/
    #   HypergeometricPFQ/06/01/05/02/0004/
    def h(*args):
        a_s = list(args[:p])
        b_s = list(args[p:])
        Ts = []
        recz = ctx.one/z
        negz = ctx.fneg(z, exact=True)
        for k in range(q+1):
            ak = a_s[k]
            C = [negz]
            Cp = [-ak]
            Gn = b_s + [ak] + [a_s[j]-ak for j in range(q+1) if j != k]
            Gd = a_s + [b_s[j]-ak for j in range(q)]
            Fn = [ak] + [ak-b_s[j]+1 for j in range(q)]
            Fd = [1-a_s[j]+ak for j in range(q+1) if j != k]
            Ts.append((C, Cp, Gn, Gd, Fn, Fd, recz))
        return Ts
    return ctx.hypercomb(h, a_s+b_s, **kwargs)

@defun
def _hyp_borel(ctx, p, q, a_s, b_s, z, **kwargs):
    if a_s:
        a_s, a_types = zip(*a_s)
        a_s = list(a_s)
    else:
        a_s, a_types = [], ()
    if b_s:
        b_s, b_types = zip(*b_s)
        b_s = list(b_s)
    else:
        b_s, b_types = [], ()
    kwargs['maxterms'] = kwargs.get('maxterms', ctx.prec)
    try:
        return ctx.hypsum(p, q, a_types+b_types, a_s+b_s, z, **kwargs)
    except ctx.NoConvergence:
        pass
    prec = ctx.prec
    try:
        tol = kwargs.get('asymp_tol', ctx.eps/4)
        ctx.prec += 10
        # hypsum is has a conservative tolerance. So we try again:
        def term(k, cache={0:ctx.one}):
            if k in cache:
                return cache[k]
            t = term(k-1)
            for a in a_s: t *= (a+(k-1))
            for b in b_s: t /= (b+(k-1))
            t *= z
            t /= k
            cache[k] = t
            return t
        s = ctx.one
        for k in xrange(1, ctx.prec):
            t = term(k)
            s += t
            if abs(t) <= tol:
                return s
    finally:
        ctx.prec = prec
    if p <= q+3:
        contour = kwargs.get('contour')
        if not contour:
            if ctx.arg(z) < 0.25:
                u = z / max(1, abs(z))
                if ctx.arg(z) >= 0:
                    contour = [0, 2j, (2j+2)/u, 2/u, ctx.inf]
                else:
                    contour = [0, -2j, (-2j+2)/u, 2/u, ctx.inf]
                #contour = [0, 2j/z, 2/z, ctx.inf]
                #contour = [0, 2j, 2/z, ctx.inf]
                #contour = [0, 2j, ctx.inf]
            else:
                contour = [0, ctx.inf]
        quad_kwargs = kwargs.get('quad_kwargs', {})
        def g(t):
            return ctx.exp(-t)*ctx.hyper(a_s, b_s+[1], t*z)
        I, err = ctx.quad(g, contour, error=True, **quad_kwargs)
        if err <= abs(I)*ctx.eps*8:
            return I
    raise ctx.NoConvergence


@defun
def _hyp2f2(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), (a2, a2type) = a_s
    (b1, b1type), (b2, b2type) = b_s

    absz = abs(z)
    magz = ctx.mag(z)
    orig = ctx.prec

    # Asymptotic expansion is ~ exp(z)
    asymp_extraprec = magz

    # Asymptotic series is in terms of 3F1
    can_use_asymptotic = (not kwargs.get('force_series')) and \
        (ctx.mag(absz) > 3)

    # TODO: much of the following could be shared with 2F3 instead of
    # copypasted
    if can_use_asymptotic:
        #print "using asymp"
        try:
            try:
                ctx.prec += asymp_extraprec
                # http://functions.wolfram.com/HypergeometricFunctions/
                # Hypergeometric2F2/06/02/02/0002/
                def h(a1,a2,b1,b2):
                    X = a1+a2-b1-b2
                    A2 = a1+a2
                    B2 = b1+b2
                    c = {}
                    c[0] = ctx.one
                    c[1] = (A2-1)*X+b1*b2-a1*a2
                    s1 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = 1-B2+2*a1+a1**2+2*a2+a2**2-A2*B2+a1*a2+b1*b2+(2*B2-3*(A2+1))*k+2*k**2
                            uu2 = (k-A2+b1-1)*(k-A2+b2-1)*(k-X-2)
                            c[k] = ctx.one/k * (uu1*c[k-1]-uu2*c[k-2])
                        t1 = c[k] * z**(-k)
                        if abs(t1) < 0.1*ctx.eps:
                            #print "Convergence :)"
                            break
                        # Quit if the series doesn't converge quickly enough
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            #print "No convergence :("
                            raise ctx.NoConvergence
                        s1 += t1
                        tprev = t1
                        k += 1
                    S = ctx.exp(z)*s1
                    T1 = [z,S], [X,1], [b1,b2],[a1,a2],[],[],0
                    T2 = [-z],[-a1],[b1,b2,a2-a1],[a2,b1-a1,b2-a1],[a1,a1-b1+1,a1-b2+1],[a1-a2+1],-1/z
                    T3 = [-z],[-a2],[b1,b2,a1-a2],[a1,b1-a2,b2-a2],[a2,a2-b1+1,a2-b2+1],[-a1+a2+1],-1/z
                    return T1, T2, T3
                v = ctx.hypercomb(h, [a1,a2,b1,b2], force_series=True, maxterms=4*ctx.prec)
                if sum(ctx._is_real_type(u) for u in [a1,a2,b1,b2,z]) == 5:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig

    return ctx.hypsum(2, 2, (a1type, a2type, b1type, b2type), [a1, a2, b1, b2], z, **kwargs)



@defun
def _hyp1f2(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), = a_s
    (b1, b1type), (b2, b2type) = b_s

    absz = abs(z)
    magz = ctx.mag(z)
    orig = ctx.prec

    # Asymptotic expansion is ~ exp(sqrt(z))
    asymp_extraprec = z and magz//2

    # Asymptotic series is in terms of 3F0
    can_use_asymptotic = (not kwargs.get('force_series')) and \
        (ctx.mag(absz) > 19) and \
        (ctx.sqrt(absz) > 1.5*orig)  # and \
    #   ctx._hyp_check_convergence([a1, a1-b1+1, a1-b2+1], [],
    #                              1/absz, orig+40+asymp_extraprec)

    # TODO: much of the following could be shared with 2F3 instead of
    # copypasted
    if can_use_asymptotic:
        #print "using asymp"
        try:
            try:
                ctx.prec += asymp_extraprec
                # http://functions.wolfram.com/HypergeometricFunctions/
                # Hypergeometric1F2/06/02/03/
                def h(a1,b1,b2):
                    X = ctx.mpq_1_2*(a1-b1-b2+ctx.mpq_1_2)
                    c = {}
                    c[0] = ctx.one
                    c[1] = 2*(ctx.mpq_1_4*(3*a1+b1+b2-2)*(a1-b1-b2)+b1*b2-ctx.mpq_3_16)
                    c[2] = 2*(b1*b2+ctx.mpq_1_4*(a1-b1-b2)*(3*a1+b1+b2-2)-ctx.mpq_3_16)**2+\
                        ctx.mpq_1_16*(-16*(2*a1-3)*b1*b2 + \
                        4*(a1-b1-b2)*(-8*a1**2+11*a1+b1+b2-2)-3)
                    s1 = 0
                    s2 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = (3*k**2+(-6*a1+2*b1+2*b2-4)*k + 3*a1**2 - \
                                (b1-b2)**2 - 2*a1*(b1+b2-2) + ctx.mpq_1_4)
                            uu2 = (k-a1+b1-b2-ctx.mpq_1_2)*(k-a1-b1+b2-ctx.mpq_1_2)*\
                                (k-a1+b1+b2-ctx.mpq_5_2)
                            c[k] = ctx.one/(2*k)*(uu1*c[k-1]-uu2*c[k-2])
                        w = c[k] * (-z)**(-0.5*k)
                        t1 = (-ctx.j)**k * ctx.mpf(2)**(-k) * w
                        t2 = ctx.j**k * ctx.mpf(2)**(-k) * w
                        if abs(t1) < 0.1*ctx.eps:
                            #print "Convergence :)"
                            break
                        # Quit if the series doesn't converge quickly enough
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            #print "No convergence :("
                            raise ctx.NoConvergence
                        s1 += t1
                        s2 += t2
                        tprev = t1
                        k += 1
                    S = ctx.expj(ctx.pi*X+2*ctx.sqrt(-z))*s1 + \
                        ctx.expj(-(ctx.pi*X+2*ctx.sqrt(-z)))*s2
                    T1 = [0.5*S, ctx.pi, -z], [1, -0.5, X], [b1, b2], [a1],\
                        [], [], 0
                    T2 = [-z], [-a1], [b1,b2],[b1-a1,b2-a1], \
                        [a1,a1-b1+1,a1-b2+1], [], 1/z
                    return T1, T2
                v = ctx.hypercomb(h, [a1,b1,b2], force_series=True, maxterms=4*ctx.prec)
                if sum(ctx._is_real_type(u) for u in [a1,b1,b2,z]) == 4:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig

    #print "not using asymp"
    return ctx.hypsum(1, 2, (a1type, b1type, b2type), [a1, b1, b2], z, **kwargs)



@defun
def _hyp2f3(ctx, a_s, b_s, z, **kwargs):
    (a1, a1type), (a2, a2type) = a_s
    (b1, b1type), (b2, b2type), (b3, b3type) = b_s

    absz = abs(z)
    magz = ctx.mag(z)

    # Asymptotic expansion is ~ exp(sqrt(z))
    asymp_extraprec = z and magz//2
    orig = ctx.prec

    # Asymptotic series is in terms of 4F1
    # The square root below empirically provides a plausible criterion
    # for the leading series to converge
    can_use_asymptotic = (not kwargs.get('force_series')) and \
        (ctx.mag(absz) > 19) and (ctx.sqrt(absz) > 1.5*orig)

    if can_use_asymptotic:
        #print "using asymp"
        try:
            try:
                ctx.prec += asymp_extraprec
                # http://functions.wolfram.com/HypergeometricFunctions/
                # Hypergeometric2F3/06/02/03/01/0002/
                def h(a1,a2,b1,b2,b3):
                    X = ctx.mpq_1_2*(a1+a2-b1-b2-b3+ctx.mpq_1_2)
                    A2 = a1+a2
                    B3 = b1+b2+b3
                    A = a1*a2
                    B = b1*b2+b3*b2+b1*b3
                    R = b1*b2*b3
                    c = {}
                    c[0] = ctx.one
                    c[1] = 2*(B - A + ctx.mpq_1_4*(3*A2+B3-2)*(A2-B3) - ctx.mpq_3_16)
                    c[2] = ctx.mpq_1_2*c[1]**2 + ctx.mpq_1_16*(-16*(2*A2-3)*(B-A) + 32*R +\
                        4*(-8*A2**2 + 11*A2 + 8*A + B3 - 2)*(A2-B3)-3)
                    s1 = 0
                    s2 = 0
                    k = 0
                    tprev = 0
                    while 1:
                        if k not in c:
                            uu1 = (k-2*X-3)*(k-2*X-2*b1-1)*(k-2*X-2*b2-1)*\
                                (k-2*X-2*b3-1)
                            uu2 = (4*(k-1)**3 - 6*(4*X+B3)*(k-1)**2 + \
                                2*(24*X**2+12*B3*X+4*B+B3-1)*(k-1) - 32*X**3 - \
                                24*B3*X**2 - 4*B - 8*R - 4*(4*B+B3-1)*X + 2*B3-1)
                            uu3 = (5*(k-1)**2+2*(-10*X+A2-3*B3+3)*(k-1)+2*c[1])
                            c[k] = ctx.one/(2*k)*(uu1*c[k-3]-uu2*c[k-2]+uu3*c[k-1])
                        w = c[k] * ctx.power(-z, -0.5*k)
                        t1 = (-ctx.j)**k * ctx.mpf(2)**(-k) * w
                        t2 = ctx.j**k * ctx.mpf(2)**(-k) * w
                        if abs(t1) < 0.1*ctx.eps:
                            break
                        # Quit if the series doesn't converge quickly enough
                        if k > 5 and abs(tprev) / abs(t1) < 1.5:
                            raise ctx.NoConvergence
                        s1 += t1
                        s2 += t2
                        tprev = t1
                        k += 1
                    S = ctx.expj(ctx.pi*X+2*ctx.sqrt(-z))*s1 + \
                        ctx.expj(-(ctx.pi*X+2*ctx.sqrt(-z)))*s2
                    T1 = [0.5*S, ctx.pi, -z], [1, -0.5, X], [b1, b2, b3], [a1, a2],\
                        [], [], 0
                    T2 = [-z], [-a1], [b1,b2,b3,a2-a1],[a2,b1-a1,b2-a1,b3-a1], \
                        [a1,a1-b1+1,a1-b2+1,a1-b3+1], [a1-a2+1], 1/z
                    T3 = [-z], [-a2], [b1,b2,b3,a1-a2],[a1,b1-a2,b2-a2,b3-a2], \
                        [a2,a2-b1+1,a2-b2+1,a2-b3+1],[-a1+a2+1], 1/z
                    return T1, T2, T3
                v = ctx.hypercomb(h, [a1,a2,b1,b2,b3], force_series=True, maxterms=4*ctx.prec)
                if sum(ctx._is_real_type(u) for u in [a1,a2,b1,b2,b3,z]) == 6:
                    v = ctx.re(v)
                return v
            except ctx.NoConvergence:
                pass
        finally:
            ctx.prec = orig

    return ctx.hypsum(2, 3, (a1type, a2type, b1type, b2type, b3type), [a1, a2, b1, b2, b3], z, **kwargs)

@defun
def _hyp2f0(ctx, a_s, b_s, z, **kwargs):
    (a, atype), (b, btype) = a_s
    # We want to try aggressively to use the asymptotic expansion,
    # and fall back only when absolutely necessary
    try:
        kwargsb = kwargs.copy()
        kwargsb['maxterms'] = kwargsb.get('maxterms', ctx.prec)
        return ctx.hypsum(2, 0, (atype,btype), [a,b], z, **kwargsb)
    except ctx.NoConvergence:
        if kwargs.get('force_series'):
            raise
        pass
    def h(a, b):
        w = ctx.sinpi(b)
        rz = -1/z
        T1 = ([ctx.pi,w,rz],[1,-1,a],[],[a-b+1,b],[a],[b],rz)
        T2 = ([-ctx.pi,w,rz],[1,-1,1+a-b],[],[a,2-b],[a-b+1],[2-b],rz)
        return T1, T2
    return ctx.hypercomb(h, [a, 1+a-b], **kwargs)

@defun
def meijerg(ctx, a_s, b_s, z, r=1, series=None, **kwargs):
    an, ap = a_s
    bm, bq = b_s
    n = len(an)
    p = n + len(ap)
    m = len(bm)
    q = m + len(bq)
    a = an+ap
    b = bm+bq
    a = [ctx.convert(_) for _ in a]
    b = [ctx.convert(_) for _ in b]
    z = ctx.convert(z)
    if series is None:
        if p < q: series = 1
        if p > q: series = 2
        if p == q:
            if m+n == p and abs(z) > 1:
                series = 2
            else:
                series = 1
    if kwargs.get('verbose'):
        print("Meijer G m,n,p,q,series =", m,n,p,q,series)
    if series == 1:
        def h(*args):
            a = args[:p]
            b = args[p:]
            terms = []
            for k in range(m):
                bases = [z]
                expts = [b[k]/r]
                gn = [b[j]-b[k] for j in range(m) if j != k]
                gn += [1-a[j]+b[k] for j in range(n)]
                gd = [a[j]-b[k] for j in range(n,p)]
                gd += [1-b[j]+b[k] for j in range(m,q)]
                hn = [1-a[j]+b[k] for j in range(p)]
                hd = [1-b[j]+b[k] for j in range(q) if j != k]
                hz = (-ctx.one)**(p-m-n) * z**(ctx.one/r)
                terms.append((bases, expts, gn, gd, hn, hd, hz))
            return terms
    else:
        def h(*args):
            a = args[:p]
            b = args[p:]
            terms = []
            for k in range(n):
                bases = [z]
                if r == 1:
                    expts = [a[k]-1]
                else:
                    expts = [(a[k]-1)/ctx.convert(r)]
                gn = [a[k]-a[j] for j in range(n) if j != k]
                gn += [1-a[k]+b[j] for j in range(m)]
                gd = [a[k]-b[j] for j in range(m,q)]
                gd += [1-a[k]+a[j] for j in range(n,p)]
                hn = [1-a[k]+b[j] for j in range(q)]
                hd = [1+a[j]-a[k] for j in range(p) if j != k]
                hz = (-ctx.one)**(q-m-n) / z**(ctx.one/r)
                terms.append((bases, expts, gn, gd, hn, hd, hz))
            return terms
    return ctx.hypercomb(h, a+b, **kwargs)

@defun_wrapped
def appellf1(ctx,a,b1,b2,c,x,y,**kwargs):
    # Assume x smaller
    # We will use x for the outer loop
    if abs(x) > abs(y):
        x, y = y, x
        b1, b2 = b2, b1
    def ok(x):
        return abs(x) < 0.99
    # Finite cases
    if ctx.isnpint(a):
        pass
    elif ctx.isnpint(b1):
        pass
    elif ctx.isnpint(b2):
        x, y, b1, b2 = y, x, b2, b1
    else:
        #print x, y
        # Note: ok if |y| > 1, because
        # 2F1 implements analytic continuation
        if not ok(x):
            u1 = (x-y)/(x-1)
            if not ok(u1):
                raise ValueError("Analytic continuation not implemented")
            #print "Using analytic continuation"
            return (1-x)**(-b1)*(1-y)**(c-a-b2)*\
                ctx.appellf1(c-a,b1,c-b1-b2,c,u1,y,**kwargs)
    return ctx.hyper2d({'m+n':[a],'m':[b1],'n':[b2]}, {'m+n':[c]}, x,y, **kwargs)

@defun
def appellf2(ctx,a,b1,b2,c1,c2,x,y,**kwargs):
    # TODO: continuation
    return ctx.hyper2d({'m+n':[a],'m':[b1],'n':[b2]},
        {'m':[c1],'n':[c2]}, x,y, **kwargs)

@defun
def appellf3(ctx,a1,a2,b1,b2,c,x,y,**kwargs):
    outer_polynomial = ctx.isnpint(a1) or ctx.isnpint(b1)
    inner_polynomial = ctx.isnpint(a2) or ctx.isnpint(b2)
    if not outer_polynomial:
        if inner_polynomial or abs(x) > abs(y):
            x, y = y, x
            a1,a2,b1,b2 = a2,a1,b2,b1
    return ctx.hyper2d({'m':[a1,b1],'n':[a2,b2]}, {'m+n':[c]},x,y,**kwargs)

@defun
def appellf4(ctx,a,b,c1,c2,x,y,**kwargs):
    # TODO: continuation
    return ctx.hyper2d({'m+n':[a,b]}, {'m':[c1],'n':[c2]},x,y,**kwargs)

@defun
def hyper2d(ctx, a, b, x, y, **kwargs):
    r"""
    Sums the generalized 2D hypergeometric series

    .. math ::

        \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
            \frac{P((a),m,n)}{Q((b),m,n)}
            \frac{x^m y^n} {m! n!}

    where `(a) = (a_1,\ldots,a_r)`, `(b) = (b_1,\ldots,b_s)` and where
    `P` and `Q` are products of rising factorials such as `(a_j)_n` or
    `(a_j)_{m+n}`. `P` and `Q` are specified in the form of dicts, with
    the `m` and `n` dependence as keys and parameter lists as values.
    The supported rising factorials are given in the following table
    (note that only a few are supported in `Q`):

    +------------+-------------------+--------+
    | Key        |  Rising factorial | `Q`    |
    +============+===================+========+
    | ``'m'``    |   `(a_j)_m`       | Yes    |
    +------------+-------------------+--------+
    | ``'n'``    |   `(a_j)_n`       | Yes    |
    +------------+-------------------+--------+
    | ``'m+n'``  |   `(a_j)_{m+n}`   | Yes    |
    +------------+-------------------+--------+
    | ``'m-n'``  |   `(a_j)_{m-n}`   | No     |
    +------------+-------------------+--------+
    | ``'n-m'``  |   `(a_j)_{n-m}`   | No     |
    +------------+-------------------+--------+
    | ``'2m+n'`` |   `(a_j)_{2m+n}`  | No     |
    +------------+-------------------+--------+
    | ``'2m-n'`` |   `(a_j)_{2m-n}`  | No     |
    +------------+-------------------+--------+
    | ``'2n-m'`` |   `(a_j)_{2n-m}`  | No     |
    +------------+-------------------+--------+

    For example, the Appell F1 and F4 functions

    .. math ::

        F_1 = \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
              \frac{(a)_{m+n} (b)_m (c)_n}{(d)_{m+n}}
              \frac{x^m y^n}{m! n!}

        F_4 = \sum_{m=0}^{\infty} \sum_{n=0}^{\infty}
              \frac{(a)_{m+n} (b)_{m+n}}{(c)_m (d)_{n}}
              \frac{x^m y^n}{m! n!}

    can be represented respectively as

        ``hyper2d({'m+n':[a], 'm':[b], 'n':[c]}, {'m+n':[d]}, x, y)``

        ``hyper2d({'m+n':[a,b]}, {'m':[c], 'n':[d]}, x, y)``

    More generally, :func:`~mpmath.hyper2d` can evaluate any of the 34 distinct
    convergent second-order (generalized Gaussian) hypergeometric
    series enumerated by Horn, as well as the Kampe de Feriet
    function.

    The series is computed by rewriting it so that the inner
    series (i.e. the series containing `n` and `y`) has the form of an
    ordinary generalized hypergeometric series and thereby can be
    evaluated efficiently using :func:`~mpmath.hyper`. If possible,
    manually swapping `x` and `y` and the corresponding parameters
    can sometimes give better results.

    **Examples**

    Two separable cases: a product of two geometric series, and a
    product of two Gaussian hypergeometric functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> x, y = mpf(0.25), mpf(0.5)
        >>> hyper2d({'m':1,'n':1}, {}, x,y)
        2.666666666666666666666667
        >>> 1/(1-x)/(1-y)
        2.666666666666666666666667
        >>> hyper2d({'m':[1,2],'n':[3,4]}, {'m':[5],'n':[6]}, x,y)
        4.164358531238938319669856
        >>> hyp2f1(1,2,5,x)*hyp2f1(3,4,6,y)
        4.164358531238938319669856

    Some more series that can be done in closed form::

        >>> hyper2d({'m':1,'n':1},{'m+n':1},x,y)
        2.013417124712514809623881
        >>> (exp(x)*x-exp(y)*y)/(x-y)
        2.013417124712514809623881

    Six of the 34 Horn functions, G1-G3 and H1-H3::

        >>> from mpmath import *
        >>> mp.dps = 10; mp.pretty = True
        >>> x, y = 0.0625, 0.125
        >>> a1,a2,b1,b2,c1,c2,d = 1.1,-1.2,-1.3,-1.4,1.5,-1.6,1.7
        >>> hyper2d({'m+n':a1,'n-m':b1,'m-n':b2},{},x,y)  # G1
        1.139090746
        >>> nsum(lambda m,n: rf(a1,m+n)*rf(b1,n-m)*rf(b2,m-n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        1.139090746
        >>> hyper2d({'m':a1,'n':a2,'n-m':b1,'m-n':b2},{},x,y)  # G2
        0.9503682696
        >>> nsum(lambda m,n: rf(a1,m)*rf(a2,n)*rf(b1,n-m)*rf(b2,m-n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        0.9503682696
        >>> hyper2d({'2n-m':a1,'2m-n':a2},{},x,y)  # G3
        1.029372029
        >>> nsum(lambda m,n: rf(a1,2*n-m)*rf(a2,2*m-n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        1.029372029
        >>> hyper2d({'m-n':a1,'m+n':b1,'n':c1},{'m':d},x,y)  # H1
        -1.605331256
        >>> nsum(lambda m,n: rf(a1,m-n)*rf(b1,m+n)*rf(c1,n)/rf(d,m)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        -1.605331256
        >>> hyper2d({'m-n':a1,'m':b1,'n':[c1,c2]},{'m':d},x,y)  # H2
        -2.35405404
        >>> nsum(lambda m,n: rf(a1,m-n)*rf(b1,m)*rf(c1,n)*rf(c2,n)/rf(d,m)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        -2.35405404
        >>> hyper2d({'2m+n':a1,'n':b1},{'m+n':c1},x,y)  # H3
        0.974479074
        >>> nsum(lambda m,n: rf(a1,2*m+n)*rf(b1,n)/rf(c1,m+n)*\
        ...     x**m*y**n/fac(m)/fac(n), [0,inf], [0,inf])
        0.974479074

    **References**

    1. [SrivastavaKarlsson]_
    2. [Weisstein]_ http://mathworld.wolfram.com/HornFunction.html
    3. [Weisstein]_ http://mathworld.wolfram.com/AppellHypergeometricFunction.html

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    def parse(dct, key):
        args = dct.pop(key, [])
        try:
            args = list(args)
        except TypeError:
            args = [args]
        return [ctx.convert(arg) for arg in args]
    a_s = dict(a)
    b_s = dict(b)
    a_m = parse(a, 'm')
    a_n = parse(a, 'n')
    a_m_add_n = parse(a, 'm+n')
    a_m_sub_n = parse(a, 'm-n')
    a_n_sub_m = parse(a, 'n-m')
    a_2m_add_n = parse(a, '2m+n')
    a_2m_sub_n = parse(a, '2m-n')
    a_2n_sub_m = parse(a, '2n-m')
    b_m = parse(b, 'm')
    b_n = parse(b, 'n')
    b_m_add_n = parse(b, 'm+n')
    if a: raise ValueError("unsupported key: %r" % a.keys()[0])
    if b: raise ValueError("unsupported key: %r" % b.keys()[0])
    s = 0
    outer = ctx.one
    m = ctx.mpf(0)
    ok_count = 0
    prec = ctx.prec
    maxterms = kwargs.get('maxterms', 20*prec)
    try:
        ctx.prec += 10
        tol = +ctx.eps
        while 1:
            inner_sign = 1
            outer_sign = 1
            inner_a = list(a_n)
            inner_b = list(b_n)
            outer_a = [a+m for a in a_m]
            outer_b = [b+m for b in b_m]
            # (a)_{m+n} = (a)_m (a+m)_n
            for a in a_m_add_n:
                a = a+m
                inner_a.append(a)
                outer_a.append(a)
            # (b)_{m+n} = (b)_m (b+m)_n
            for b in b_m_add_n:
                b = b+m
                inner_b.append(b)
                outer_b.append(b)
            # (a)_{n-m} = (a-m)_n / (a-m)_m
            for a in a_n_sub_m:
                inner_a.append(a-m)
                outer_b.append(a-m-1)
            # (a)_{m-n} = (-1)^(m+n) (1-a-m)_m / (1-a-m)_n
            for a in a_m_sub_n:
                inner_sign *= (-1)
                outer_sign *= (-1)**(m)
                inner_b.append(1-a-m)
                outer_a.append(-a-m)
            # (a)_{2m+n} = (a)_{2m} (a+2m)_n
            for a in a_2m_add_n:
                inner_a.append(a+2*m)
                outer_a.append((a+2*m)*(1+a+2*m))
            # (a)_{2m-n} = (-1)^(2m+n) (1-a-2m)_{2m} / (1-a-2m)_n
            for a in a_2m_sub_n:
                inner_sign *= (-1)
                inner_b.append(1-a-2*m)
                outer_a.append((a+2*m)*(1+a+2*m))
            # (a)_{2n-m} = 4^n ((a-m)/2)_n ((a-m+1)/2)_n / (a-m)_m
            for a in a_2n_sub_m:
                inner_sign *= 4
                inner_a.append(0.5*(a-m))
                inner_a.append(0.5*(a-m+1))
                outer_b.append(a-m-1)
            inner = ctx.hyper(inner_a, inner_b, inner_sign*y,
                zeroprec=ctx.prec, **kwargs)
            term = outer * inner * outer_sign
            if abs(term) < tol:
                ok_count += 1
            else:
                ok_count = 0
            if ok_count >= 3 or not outer:
                break
            s += term
            for a in outer_a: outer *= a
            for b in outer_b: outer /= b
            m += 1
            outer = outer * x / m
            if m > maxterms:
                raise ctx.NoConvergence("maxterms exceeded in hyper2d")
    finally:
        ctx.prec = prec
    return +s

"""
@defun
def kampe_de_feriet(ctx,a,b,c,d,e,f,x,y,**kwargs):
    return ctx.hyper2d({'m+n':a,'m':b,'n':c},
        {'m+n':d,'m':e,'n':f}, x,y, **kwargs)
"""

@defun
def bihyper(ctx, a_s, b_s, z, **kwargs):
    r"""
    Evaluates the bilateral hypergeometric series

    .. math ::

        \,_AH_B(a_1, \ldots, a_k; b_1, \ldots, b_B; z) =
            \sum_{n=-\infty}^{\infty}
            \frac{(a_1)_n \ldots (a_A)_n}
                 {(b_1)_n \ldots (b_B)_n} \, z^n

    where, for direct convergence, `A = B` and `|z| = 1`, although a
    regularized sum exists more generally by considering the
    bilateral series as a sum of two ordinary hypergeometric
    functions. In order for the series to make sense, none of the
    parameters may be integers.

    **Examples**

    The value of `\,_2H_2` at `z = 1` is given by Dougall's formula::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> a,b,c,d = 0.5, 1.5, 2.25, 3.25
        >>> bihyper([a,b],[c,d],1)
        -14.49118026212345786148847
        >>> gammaprod([c,d,1-a,1-b,c+d-a-b-1],[c-a,d-a,c-b,d-b])
        -14.49118026212345786148847

    The regularized function `\,_1H_0` can be expressed as the
    sum of one `\,_2F_0` function and one `\,_1F_1` function::

        >>> a = mpf(0.25)
        >>> z = mpf(0.75)
        >>> bihyper([a], [], z)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)
        >>> hyper([a,1],[],z) + (hyper([1],[1-a],-1/z)-1)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)
        >>> hyper([a,1],[],z) + hyper([1],[2-a],-1/z)/z/(a-1)
        (0.2454393389657273841385582 + 0.2454393389657273841385582j)

    **References**

    1. [Slater]_ (chapter 6: "Bilateral Series", pp. 180-189)
    2. [Wikipedia]_ http://en.wikipedia.org/wiki/Bilateral_hypergeometric_series

    """
    z = ctx.convert(z)
    c_s = a_s + b_s
    p = len(a_s)
    q = len(b_s)
    if (p, q) == (0,0) or (p, q) == (1,1):
        return ctx.zero * z
    neg = (p-q) % 2
    def h(*c_s):
        a_s = list(c_s[:p])
        b_s = list(c_s[p:])
        aa_s = [2-b for b in b_s]
        bb_s = [2-a for a in a_s]
        rp = [(-1)**neg * z] + [1-b for b in b_s] + [1-a for a in a_s]
        rc = [-1] + [1]*len(b_s) + [-1]*len(a_s)
        T1 = [], [], [], [], a_s + [1], b_s, z
        T2 = rp, rc, [], [], aa_s + [1], bb_s, (-1)**neg / z
        return T1, T2
    return ctx.hypercomb(h, c_s, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\dammit.py ===
# -*- coding: utf-8 -*-
"""Beautiful Soup bonus library: Unicode, Dammit

This library converts a bytestream to Unicode through any means
necessary. It is heavily based on code from Mark Pilgrim's `Universal
Feed Parser <https://pypi.org/project/feedparser/>`_, now maintained
by Kurt McKee. It does not rewrite the body of an XML or HTML document
to reflect a new encoding; that's the job of `TreeBuilder`.

"""

# Use of this source code is governed by the MIT license.
__license__ = "MIT"

from html.entities import codepoint2name
from collections import defaultdict
import codecs
from html.entities import html5
import re
from logging import Logger, getLogger
from types import ModuleType
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)
from typing_extensions import Literal
from bs4._typing import (
    _Encoding,
    _Encodings,
)
import warnings

# Import a library to autodetect character encodings. We'll support
# any of a number of libraries that all support the same API:
#
# * cchardet
# * chardet
# * charset-normalizer
chardet_module: Optional[ModuleType] = None
try:
    #  PyPI package: cchardet
    import cchardet

    chardet_module = cchardet
except ImportError:
    try:
        #  Debian package: python-chardet
        #  PyPI package: chardet
        import chardet

        chardet_module = chardet
    except ImportError:
        try:
            # PyPI package: charset-normalizer
            import charset_normalizer

            chardet_module = charset_normalizer
        except ImportError:
            # No chardet available.
            pass


def _chardet_dammit(s: bytes) -> Optional[str]:
    """Try as hard as possible to detect the encoding of a bytestring."""
    if chardet_module is None or isinstance(s, str):
        return None
    module = chardet_module
    return module.detect(s)["encoding"]


# Build bytestring and Unicode versions of regular expressions for finding
# a declared encoding inside an XML or HTML document.
xml_encoding: str = "^\\s*<\\?.*encoding=['\"](.*?)['\"].*\\?>"  #: :meta private:
html_meta: str = (
    "<\\s*meta[^>]+charset\\s*=\\s*[\"']?([^>]*?)[ /;'\">]"  #: :meta private:
)

# TODO-TYPING: The Pattern type here could use more refinement, but it's tricky.
encoding_res: Dict[Type, Dict[str, Pattern]] = dict()
encoding_res[bytes] = {
    "html": re.compile(html_meta.encode("ascii"), re.I),
    "xml": re.compile(xml_encoding.encode("ascii"), re.I),
}
encoding_res[str] = {
    "html": re.compile(html_meta, re.I),
    "xml": re.compile(xml_encoding, re.I),
}


class EntitySubstitution(object):
    """The ability to substitute XML or HTML entities for certain characters."""

    #: A map of named HTML entities to the corresponding Unicode string.
    #:
    #: :meta hide-value:
    HTML_ENTITY_TO_CHARACTER: Dict[str, str]

    #: A map of Unicode strings to the corresponding named HTML entities;
    #: the inverse of HTML_ENTITY_TO_CHARACTER.
    #:
    #: :meta hide-value:
    CHARACTER_TO_HTML_ENTITY: Dict[str, str]

    #: A regular expression that matches any character (or, in rare
    #: cases, pair of characters) that can be replaced with a named
    #: HTML entity.
    #:
    #: :meta hide-value:
    CHARACTER_TO_HTML_ENTITY_RE: Pattern[str]

    #: A very similar regular expression to
    #: CHARACTER_TO_HTML_ENTITY_RE, but which also matches unescaped
    #: ampersands. This is used by the 'html' formatted to provide
    #: backwards-compatibility, even though the HTML5 spec allows most
    #: ampersands to go unescaped.
    #:
    #: :meta hide-value:
    CHARACTER_TO_HTML_ENTITY_WITH_AMPERSAND_RE: Pattern[str]

    @classmethod
    def _populate_class_variables(cls) -> None:
        """Initialize variables used by this class to manage the plethora of
        HTML5 named entities.

        This function sets the following class variables:

        CHARACTER_TO_HTML_ENTITY - A mapping of Unicode strings like "⦨" to
        entity names like "angmsdaa". When a single Unicode string has
        multiple entity names, we try to choose the most commonly-used
        name.

        HTML_ENTITY_TO_CHARACTER: A mapping of entity names like "angmsdaa" to
        Unicode strings like "⦨".

        CHARACTER_TO_HTML_ENTITY_RE: A regular expression matching (almost) any
        Unicode string that corresponds to an HTML5 named entity.

        CHARACTER_TO_HTML_ENTITY_WITH_AMPERSAND_RE: A very similar
        regular expression to CHARACTER_TO_HTML_ENTITY_RE, but which
        also matches unescaped ampersands. This is used by the 'html'
        formatted to provide backwards-compatibility, even though the HTML5
        spec allows most ampersands to go unescaped.
        """
        unicode_to_name = {}
        name_to_unicode = {}

        short_entities = set()
        long_entities_by_first_character = defaultdict(set)

        for name_with_semicolon, character in sorted(html5.items()):
            # "It is intentional, for legacy compatibility, that many
            # code points have multiple character reference names. For
            # example, some appear both with and without the trailing
            # semicolon, or with different capitalizations."
            # - https://html.spec.whatwg.org/multipage/named-characters.html#named-character-references
            #
            # The parsers are in charge of handling (or not) character
            # references with no trailing semicolon, so we remove the
            # semicolon whenever it appears.
            if name_with_semicolon.endswith(";"):
                name = name_with_semicolon[:-1]
            else:
                name = name_with_semicolon

            # When parsing HTML, we want to recognize any known named
            # entity and convert it to a sequence of Unicode
            # characters.
            if name not in name_to_unicode:
                name_to_unicode[name] = character

            # When _generating_ HTML, we want to recognize special
            # character sequences that _could_ be converted to named
            # entities.
            unicode_to_name[character] = name

            # We also need to build a regular expression that lets us
            # _find_ those characters in output strings so we can
            # replace them.
            #
            # This is tricky, for two reasons.

            if len(character) == 1 and ord(character) < 128 and character not in "<>":
                # First, it would be annoying to turn single ASCII
                # characters like | into named entities like
                # &verbar;. The exceptions are <>, which we _must_
                # turn into named entities to produce valid HTML.
                continue

            if len(character) > 1 and all(ord(x) < 128 for x in character):
                # We also do not want to turn _combinations_ of ASCII
                # characters like 'fj' into named entities like '&fjlig;',
                # though that's more debateable.
                continue

            # Second, some named entities have a Unicode value that's
            # a subset of the Unicode value for some _other_ named
            # entity.  As an example, \u2267' is &GreaterFullEqual;,
            # but '\u2267\u0338' is &NotGreaterFullEqual;. Our regular
            # expression needs to match the first two characters of
            # "\u2267\u0338foo", but only the first character of
            # "\u2267foo".
            #
            # In this step, we build two sets of characters that
            # _eventually_ need to go into the regular expression. But
            # we won't know exactly what the regular expression needs
            # to look like until we've gone through the entire list of
            # named entities.
            if len(character) == 1 and character != "&":
                short_entities.add(character)
            else:
                long_entities_by_first_character[character[0]].add(character)

        # Now that we've been through the entire list of entities, we
        # can create a regular expression that matches any of them.
        particles = set()
        for short in short_entities:
            long_versions = long_entities_by_first_character[short]
            if not long_versions:
                particles.add(short)
            else:
                ignore = "".join([x[1] for x in long_versions])
                # This finds, e.g. \u2267 but only if it is _not_
                # followed by \u0338.
                particles.add("%s(?![%s])" % (short, ignore))

        for long_entities in list(long_entities_by_first_character.values()):
            for long_entity in long_entities:
                particles.add(long_entity)

        re_definition = "(%s)" % "|".join(particles)

        particles.add("&")
        re_definition_with_ampersand = "(%s)" % "|".join(particles)

        # If an entity shows up in both html5 and codepoint2name, it's
        # likely that HTML5 gives it several different names, such as
        # 'rsquo' and 'rsquor'. When converting Unicode characters to
        # named entities, the codepoint2name name should take
        # precedence where possible, since that's the more easily
        # recognizable one.
        for codepoint, name in list(codepoint2name.items()):
            character = chr(codepoint)
            unicode_to_name[character] = name

        cls.CHARACTER_TO_HTML_ENTITY = unicode_to_name
        cls.HTML_ENTITY_TO_CHARACTER = name_to_unicode
        cls.CHARACTER_TO_HTML_ENTITY_RE = re.compile(re_definition)
        cls.CHARACTER_TO_HTML_ENTITY_WITH_AMPERSAND_RE = re.compile(
            re_definition_with_ampersand
        )

    #: A map of Unicode strings to the corresponding named XML entities.
    #:
    #: :meta hide-value:
    CHARACTER_TO_XML_ENTITY: Dict[str, str] = {
        "'": "apos",
        '"': "quot",
        "&": "amp",
        "<": "lt",
        ">": "gt",
    }

    # Matches any named or numeric HTML entity.
    ANY_ENTITY_RE = re.compile("&(#\\d+|#x[0-9a-fA-F]+|\\w+);", re.I)

    #: A regular expression matching an angle bracket or an ampersand that
    #: is not part of an XML or HTML entity.
    #:
    #: :meta hide-value:
    BARE_AMPERSAND_OR_BRACKET: Pattern[str] = re.compile(
        "([<>]|" "&(?!#\\d+;|#x[0-9a-fA-F]+;|\\w+;)" ")"
    )

    #: A regular expression matching an angle bracket or an ampersand.
    #:
    #: :meta hide-value:
    AMPERSAND_OR_BRACKET: Pattern[str] = re.compile("([<>&])")

    @classmethod
    def _substitute_html_entity(cls, matchobj: re.Match) -> str:
        """Used with a regular expression to substitute the
        appropriate HTML entity for a special character string."""
        original_entity = matchobj.group(0)
        entity = cls.CHARACTER_TO_HTML_ENTITY.get(original_entity)
        if entity is None:
            return "&amp;%s;" % original_entity
        return "&%s;" % entity

    @classmethod
    def _substitute_xml_entity(cls, matchobj: re.Match) -> str:
        """Used with a regular expression to substitute the
        appropriate XML entity for a special character string."""
        entity = cls.CHARACTER_TO_XML_ENTITY[matchobj.group(0)]
        return "&%s;" % entity

    @classmethod
    def _escape_entity_name(cls, matchobj: re.Match) -> str:
        return "&amp;%s;" % matchobj.group(1)

    @classmethod
    def _escape_unrecognized_entity_name(cls, matchobj: re.Match) -> str:
        possible_entity = matchobj.group(1)
        if possible_entity in cls.HTML_ENTITY_TO_CHARACTER:
            return "&%s;" % possible_entity
        return "&amp;%s;" % possible_entity

    @classmethod
    def quoted_attribute_value(cls, value: str) -> str:
        """Make a value into a quoted XML attribute, possibly escaping it.

         Most strings will be quoted using double quotes.

          Bob's Bar -> "Bob's Bar"

         If a string contains double quotes, it will be quoted using
         single quotes.

          Welcome to "my bar" -> 'Welcome to "my bar"'

         If a string contains both single and double quotes, the
         double quotes will be escaped, and the string will be quoted
         using double quotes.

          Welcome to "Bob's Bar" -> Welcome to &quot;Bob's bar&quot;

        :param value: The XML attribute value to quote
        :return: The quoted value
        """
        quote_with = '"'
        if '"' in value:
            if "'" in value:
                # The string contains both single and double
                # quotes.  Turn the double quotes into
                # entities. We quote the double quotes rather than
                # the single quotes because the entity name is
                # "&quot;" whether this is HTML or XML.  If we
                # quoted the single quotes, we'd have to decide
                # between &apos; and &squot;.
                replace_with = "&quot;"
                value = value.replace('"', replace_with)
            else:
                # There are double quotes but no single quotes.
                # We can use single quotes to quote the attribute.
                quote_with = "'"
        return quote_with + value + quote_with

    @classmethod
    def substitute_xml(cls, value: str, make_quoted_attribute: bool = False) -> str:
        """Replace special XML characters with named XML entities.

        The less-than sign will become &lt;, the greater-than sign
        will become &gt;, and any ampersands will become &amp;. If you
        want ampersands that seem to be part of an entity definition
        to be left alone, use `substitute_xml_containing_entities`
        instead.

        :param value: A string to be substituted.

        :param make_quoted_attribute: If True, then the string will be
         quoted, as befits an attribute value.

        :return: A version of ``value`` with special characters replaced
         with named entities.
        """
        # Escape angle brackets and ampersands.
        value = cls.AMPERSAND_OR_BRACKET.sub(cls._substitute_xml_entity, value)

        if make_quoted_attribute:
            value = cls.quoted_attribute_value(value)
        return value

    @classmethod
    def substitute_xml_containing_entities(
        cls, value: str, make_quoted_attribute: bool = False
    ) -> str:
        """Substitute XML entities for special XML characters.

        :param value: A string to be substituted. The less-than sign will
          become &lt;, the greater-than sign will become &gt;, and any
          ampersands that are not part of an entity defition will
          become &amp;.

        :param make_quoted_attribute: If True, then the string will be
         quoted, as befits an attribute value.
        """
        # Escape angle brackets, and ampersands that aren't part of
        # entities.
        value = cls.BARE_AMPERSAND_OR_BRACKET.sub(cls._substitute_xml_entity, value)

        if make_quoted_attribute:
            value = cls.quoted_attribute_value(value)
        return value

    @classmethod
    def substitute_html(cls, s: str) -> str:
        """Replace certain Unicode characters with named HTML entities.

        This differs from ``data.encode(encoding, 'xmlcharrefreplace')``
        in that the goal is to make the result more readable (to those
        with ASCII displays) rather than to recover from
        errors. There's absolutely nothing wrong with a UTF-8 string
        containg a LATIN SMALL LETTER E WITH ACUTE, but replacing that
        character with "&eacute;" will make it more readable to some
        people.

        :param s: The string to be modified.
        :return: The string with some Unicode characters replaced with
           HTML entities.
        """
        # Convert any appropriate characters to HTML entities.
        return cls.CHARACTER_TO_HTML_ENTITY_WITH_AMPERSAND_RE.sub(
            cls._substitute_html_entity, s
        )

    @classmethod
    def substitute_html5(cls, s: str) -> str:
        """Replace certain Unicode characters with named HTML entities
        using HTML5 rules.

        Specifically, this method is much less aggressive about
        escaping ampersands than substitute_html. Only ambiguous
        ampersands are escaped, per the HTML5 standard:

        "An ambiguous ampersand is a U+0026 AMPERSAND character (&)
        that is followed by one or more ASCII alphanumerics, followed
        by a U+003B SEMICOLON character (;), where these characters do
        not match any of the names given in the named character
        references section."

        Unlike substitute_html5_raw, this method assumes HTML entities
        were converted to Unicode characters on the way in, as
        Beautiful Soup does. By the time Beautiful Soup does its work,
        the only ambiguous ampersands that need to be escaped are the
        ones that were escaped in the original markup when mentioning
        HTML entities.

        :param s: The string to be modified.
        :return: The string with some Unicode characters replaced with
           HTML entities.
        """
        # First, escape any HTML entities found in the markup.
        s = cls.ANY_ENTITY_RE.sub(cls._escape_entity_name, s)

        # Next, convert any appropriate characters to unescaped HTML entities.
        s = cls.CHARACTER_TO_HTML_ENTITY_RE.sub(cls._substitute_html_entity, s)

        return s

    @classmethod
    def substitute_html5_raw(cls, s: str) -> str:
        """Replace certain Unicode characters with named HTML entities
        using HTML5 rules.

        substitute_html5_raw is similar to substitute_html5 but it is
        designed for standalone use (whereas substitute_html5 is
        designed for use with Beautiful Soup).

        :param s: The string to be modified.
        :return: The string with some Unicode characters replaced with
           HTML entities.
        """
        # First, escape the ampersand for anything that looks like an
        # entity but isn't in the list of recognized entities. All other
        # ampersands can be left alone.
        s = cls.ANY_ENTITY_RE.sub(cls._escape_unrecognized_entity_name, s)

        # Then, convert a range of Unicode characters to unescaped
        # HTML entities.
        s = cls.CHARACTER_TO_HTML_ENTITY_RE.sub(cls._substitute_html_entity, s)

        return s


EntitySubstitution._populate_class_variables()


class EncodingDetector:
    """This class is capable of guessing a number of possible encodings
    for a bytestring.

    Order of precedence:

    1. Encodings you specifically tell EncodingDetector to try first
       (the ``known_definite_encodings`` argument to the constructor).

    2. An encoding determined by sniffing the document's byte-order mark.

    3. Encodings you specifically tell EncodingDetector to try if
       byte-order mark sniffing fails (the ``user_encodings`` argument to the
       constructor).

    4. An encoding declared within the bytestring itself, either in an
       XML declaration (if the bytestring is to be interpreted as an XML
       document), or in a <meta> tag (if the bytestring is to be
       interpreted as an HTML document.)

    5. An encoding detected through textual analysis by chardet,
       cchardet, or a similar external library.

    6. UTF-8.

    7. Windows-1252.

    :param markup: Some markup in an unknown encoding.

    :param known_definite_encodings: When determining the encoding
        of ``markup``, these encodings will be tried first, in
        order. In HTML terms, this corresponds to the "known
        definite encoding" step defined in `section 13.2.3.1 of the HTML standard <https://html.spec.whatwg.org/multipage/parsing.html#parsing-with-a-known-character-encoding>`_.

    :param user_encodings: These encodings will be tried after the
        ``known_definite_encodings`` have been tried and failed, and
        after an attempt to sniff the encoding by looking at a
        byte order mark has failed. In HTML terms, this
        corresponds to the step "user has explicitly instructed
        the user agent to override the document's character
        encoding", defined in `section 13.2.3.2 of the HTML standard <https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding>`_.

    :param override_encodings: A **deprecated** alias for
        ``known_definite_encodings``. Any encodings here will be tried
        immediately after the encodings in
        ``known_definite_encodings``.

    :param is_html: If True, this markup is considered to be
        HTML. Otherwise it's assumed to be XML.

    :param exclude_encodings: These encodings will not be tried,
        even if they otherwise would be.

    """

    def __init__(
        self,
        markup: bytes,
        known_definite_encodings: Optional[_Encodings] = None,
        is_html: Optional[bool] = False,
        exclude_encodings: Optional[_Encodings] = None,
        user_encodings: Optional[_Encodings] = None,
        override_encodings: Optional[_Encodings] = None,
    ):
        self.known_definite_encodings = list(known_definite_encodings or [])
        if override_encodings:
            warnings.warn(
                "The 'override_encodings' argument was deprecated in 4.10.0. Use 'known_definite_encodings' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            self.known_definite_encodings += override_encodings
        self.user_encodings = user_encodings or []
        exclude_encodings = exclude_encodings or []
        self.exclude_encodings = set([x.lower() for x in exclude_encodings])
        self.chardet_encoding = None
        self.is_html = False if is_html is None else is_html
        self.declared_encoding: Optional[str] = None

        # First order of business: strip a byte-order mark.
        self.markup, self.sniffed_encoding = self.strip_byte_order_mark(markup)

    known_definite_encodings: _Encodings
    user_encodings: _Encodings
    exclude_encodings: _Encodings
    chardet_encoding: Optional[_Encoding]
    is_html: bool
    declared_encoding: Optional[_Encoding]
    markup: bytes
    sniffed_encoding: Optional[_Encoding]

    def _usable(self, encoding: Optional[_Encoding], tried: Set[_Encoding]) -> bool:
        """Should we even bother to try this encoding?

        :param encoding: Name of an encoding.
        :param tried: Encodings that have already been tried. This
            will be modified as a side effect.
        """
        if encoding is None:
            return False
        encoding = encoding.lower()
        if encoding in self.exclude_encodings:
            return False
        if encoding not in tried:
            tried.add(encoding)
            return True
        return False

    @property
    def encodings(self) -> Iterator[_Encoding]:
        """Yield a number of encodings that might work for this markup.

        :yield: A sequence of strings. Each is the name of an encoding
           that *might* work to convert a bytestring into Unicode.
        """
        tried: Set[_Encoding] = set()

        # First, try the known definite encodings
        for e in self.known_definite_encodings:
            if self._usable(e, tried):
                yield e

        # Did the document originally start with a byte-order mark
        # that indicated its encoding?
        if self.sniffed_encoding is not None and self._usable(
            self.sniffed_encoding, tried
        ):
            yield self.sniffed_encoding

        # Sniffing the byte-order mark did nothing; try the user
        # encodings.
        for e in self.user_encodings:
            if self._usable(e, tried):
                yield e

        # Look within the document for an XML or HTML encoding
        # declaration.
        if self.declared_encoding is None:
            self.declared_encoding = self.find_declared_encoding(
                self.markup, self.is_html
            )
        if self.declared_encoding is not None and self._usable(
            self.declared_encoding, tried
        ):
            yield self.declared_encoding

        # Use third-party character set detection to guess at the
        # encoding.
        if self.chardet_encoding is None:
            self.chardet_encoding = _chardet_dammit(self.markup)
        if self.chardet_encoding is not None and self._usable(
            self.chardet_encoding, tried
        ):
            yield self.chardet_encoding

        # As a last-ditch effort, try utf-8 and windows-1252.
        for e in ("utf-8", "windows-1252"):
            if self._usable(e, tried):
                yield e

    @classmethod
    def strip_byte_order_mark(cls, data: bytes) -> Tuple[bytes, Optional[_Encoding]]:
        """If a byte-order mark is present, strip it and return the encoding it implies.

        :param data: A bytestring that may or may not begin with a
           byte-order mark.

        :return: A 2-tuple (data stripped of byte-order mark, encoding implied by byte-order mark)
        """
        encoding = None
        if isinstance(data, str):
            # Unicode data cannot have a byte-order mark.
            return data, encoding
        if (
            (len(data) >= 4)
            and (data[:2] == b"\xfe\xff")
            and (data[2:4] != b"\x00\x00")
        ):
            encoding = "utf-16be"
            data = data[2:]
        elif (
            (len(data) >= 4)
            and (data[:2] == b"\xff\xfe")
            and (data[2:4] != b"\x00\x00")
        ):
            encoding = "utf-16le"
            data = data[2:]
        elif data[:3] == b"\xef\xbb\xbf":
            encoding = "utf-8"
            data = data[3:]
        elif data[:4] == b"\x00\x00\xfe\xff":
            encoding = "utf-32be"
            data = data[4:]
        elif data[:4] == b"\xff\xfe\x00\x00":
            encoding = "utf-32le"
            data = data[4:]
        return data, encoding

    @classmethod
    def find_declared_encoding(
        cls,
        markup: Union[bytes, str],
        is_html: bool = False,
        search_entire_document: bool = False,
    ) -> Optional[_Encoding]:
        """Given a document, tries to find an encoding declared within the
        text of the document itself.

        An XML encoding is declared at the beginning of the document.

        An HTML encoding is declared in a <meta> tag, hopefully near the
        beginning of the document.

        :param markup: Some markup.
        :param is_html: If True, this markup is considered to be HTML. Otherwise
            it's assumed to be XML.
        :param search_entire_document: Since an encoding is supposed
            to declared near the beginning of the document, most of
            the time it's only necessary to search a few kilobytes of
            data.  Set this to True to force this method to search the
            entire document.
        :return: The declared encoding, if one is found.
        """
        if search_entire_document:
            xml_endpos = html_endpos = len(markup)
        else:
            xml_endpos = 1024
            html_endpos = max(2048, int(len(markup) * 0.05))

        if isinstance(markup, bytes):
            res = encoding_res[bytes]
        else:
            res = encoding_res[str]

        xml_re = res["xml"]
        html_re = res["html"]
        declared_encoding: Optional[_Encoding] = None
        declared_encoding_match = xml_re.search(markup, endpos=xml_endpos)
        if not declared_encoding_match and is_html:
            declared_encoding_match = html_re.search(markup, endpos=html_endpos)
        if declared_encoding_match is not None:
            declared_encoding = declared_encoding_match.groups()[0]
        if declared_encoding:
            if isinstance(declared_encoding, bytes):
                declared_encoding = declared_encoding.decode("ascii", "replace")
            return declared_encoding.lower()
        return None


class UnicodeDammit:
    """A class for detecting the encoding of a bytestring containing an
    HTML or XML document, and decoding it to Unicode. If the source
    encoding is windows-1252, `UnicodeDammit` can also replace
    Microsoft smart quotes with their HTML or XML equivalents.

    :param markup: HTML or XML markup in an unknown encoding.

    :param known_definite_encodings: When determining the encoding
        of ``markup``, these encodings will be tried first, in
        order. In HTML terms, this corresponds to the "known
        definite encoding" step defined in `section 13.2.3.1 of the HTML standard <https://html.spec.whatwg.org/multipage/parsing.html#parsing-with-a-known-character-encoding>`_.

    :param user_encodings: These encodings will be tried after the
        ``known_definite_encodings`` have been tried and failed, and
        after an attempt to sniff the encoding by looking at a
        byte order mark has failed. In HTML terms, this
        corresponds to the step "user has explicitly instructed
        the user agent to override the document's character
        encoding", defined in `section 13.2.3.2 of the HTML standard <https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding>`_.

    :param override_encodings: A **deprecated** alias for
        ``known_definite_encodings``. Any encodings here will be tried
        immediately after the encodings in
        ``known_definite_encodings``.

    :param smart_quotes_to: By default, Microsoft smart quotes will,
       like all other characters, be converted to Unicode
       characters. Setting this to ``ascii`` will convert them to ASCII
       quotes instead.  Setting it to ``xml`` will convert them to XML
       entity references, and setting it to ``html`` will convert them
       to HTML entity references.

    :param is_html: If True, ``markup`` is treated as an HTML
       document. Otherwise it's treated as an XML document.

    :param exclude_encodings: These encodings will not be considered,
       even if the sniffing code thinks they might make sense.

    """

    def __init__(
        self,
        markup: bytes,
        known_definite_encodings: Optional[_Encodings] = [],
        smart_quotes_to: Optional[Literal["ascii", "xml", "html"]] = None,
        is_html: bool = False,
        exclude_encodings: Optional[_Encodings] = [],
        user_encodings: Optional[_Encodings] = None,
        override_encodings: Optional[_Encodings] = None,
    ):
        self.smart_quotes_to = smart_quotes_to
        self.tried_encodings = []
        self.contains_replacement_characters = False
        self.is_html = is_html
        self.log = getLogger(__name__)
        self.detector = EncodingDetector(
            markup,
            known_definite_encodings,
            is_html,
            exclude_encodings,
            user_encodings,
            override_encodings,
        )

        # Short-circuit if the data is in Unicode to begin with.
        if isinstance(markup, str) or markup == b"":
            self.markup = markup
            self.unicode_markup = str(markup)
            self.original_encoding = None
            return

        # The encoding detector may have stripped a byte-order mark.
        # Use the stripped markup from this point on.
        self.markup = self.detector.markup

        u = None
        for encoding in self.detector.encodings:
            markup = self.detector.markup
            u = self._convert_from(encoding)
            if u is not None:
                break

        if not u:
            # None of the encodings worked. As an absolute last resort,
            # try them again with character replacement.

            for encoding in self.detector.encodings:
                if encoding != "ascii":
                    u = self._convert_from(encoding, "replace")
                if u is not None:
                    self.log.warning(
                        "Some characters could not be decoded, and were "
                        "replaced with REPLACEMENT CHARACTER."
                    )

                    self.contains_replacement_characters = True
                    break

        # If none of that worked, we could at this point force it to
        # ASCII, but that would destroy so much data that I think
        # giving up is better.
        #
        # Note that this is extremely unlikely, probably impossible,
        # because the "replace" strategy is so powerful. Even running
        # the Python binary through Unicode, Dammit gives you Unicode,
        # albeit Unicode riddled with REPLACEMENT CHARACTER.
        if u is None:
            self.original_encoding = None
            self.unicode_markup = None
        else:
            self.unicode_markup = u

    #: The original markup, before it was converted to Unicode.
    #: This is not necessarily the same as what was passed in to the
    #: constructor, since any byte-order mark will be stripped.
    markup: bytes

    #: The Unicode version of the markup, following conversion. This
    #: is set to None if there was simply no way to convert the
    #: bytestring to Unicode (as with binary data).
    unicode_markup: Optional[str]

    #: This is True if `UnicodeDammit.unicode_markup` contains
    #: U+FFFD REPLACEMENT_CHARACTER characters which were not present
    #: in `UnicodeDammit.markup`. These mark character sequences that
    #: could not be represented in Unicode.
    contains_replacement_characters: bool

    #: Unicode, Dammit's best guess as to the original character
    #: encoding of `UnicodeDammit.markup`.
    original_encoding: Optional[_Encoding]

    #: The strategy used to handle Microsoft smart quotes.
    smart_quotes_to: Optional[str]

    #: The (encoding, error handling strategy) 2-tuples that were used to
    #: try and convert the markup to Unicode.
    tried_encodings: List[Tuple[_Encoding, str]]

    log: Logger  #: :meta private:

    def _sub_ms_char(self, match: re.Match) -> bytes:
        """Changes a MS smart quote character to an XML or HTML
        entity, or an ASCII character.

        TODO: Since this is only used to convert smart quotes, it
        could be simplified, and MS_CHARS_TO_ASCII made much less
        parochial.
        """
        orig: bytes = match.group(1)
        sub: bytes
        if self.smart_quotes_to == "ascii":
            if orig in self.MS_CHARS_TO_ASCII:
                sub = self.MS_CHARS_TO_ASCII[orig].encode()
            else:
                # Shouldn't happen; substitute the character
                # with itself.
                sub = orig
        else:
            if orig in self.MS_CHARS:
                substitutions = self.MS_CHARS[orig]
                if type(substitutions) is tuple:
                    if self.smart_quotes_to == "xml":
                        sub = b"&#x" + substitutions[1].encode() + b";"
                    else:
                        sub = b"&" + substitutions[0].encode() + b";"
                else:
                    substitutions = cast(str, substitutions)
                    sub = substitutions.encode()
            else:
                # Shouldn't happen; substitute the character
                # for itself.
                sub = orig
        return sub

    #: This dictionary maps commonly seen values for "charset" in HTML
    #: meta tags to the corresponding Python codec names. It only covers
    #: values that aren't in Python's aliases and can't be determined
    #: by the heuristics in `find_codec`.
    #:
    #: :meta hide-value:
    CHARSET_ALIASES: Dict[str, _Encoding] = {
        "macintosh": "mac-roman",
        "x-sjis": "shift-jis",
    }

    #: A list of encodings that tend to contain Microsoft smart quotes.
    #:
    #: :meta hide-value:
    ENCODINGS_WITH_SMART_QUOTES: _Encodings = [
        "windows-1252",
        "iso-8859-1",
        "iso-8859-2",
    ]

    def _convert_from(
        self, proposed: _Encoding, errors: str = "strict"
    ) -> Optional[str]:
        """Attempt to convert the markup to the proposed encoding.

        :param proposed: The name of a character encoding.
        :param errors: An error handling strategy, used when calling `str`.
        :return: The converted markup, or `None` if the proposed
           encoding/error handling strategy didn't work.
        """
        lookup_result = self.find_codec(proposed)
        if lookup_result is None or (lookup_result, errors) in self.tried_encodings:
            return None
        proposed = lookup_result
        self.tried_encodings.append((proposed, errors))
        markup = self.markup
        # Convert smart quotes to HTML if coming from an encoding
        # that might have them.
        if (
            self.smart_quotes_to is not None
            and proposed in self.ENCODINGS_WITH_SMART_QUOTES
        ):
            smart_quotes_re = b"([\x80-\x9f])"
            smart_quotes_compiled = re.compile(smart_quotes_re)
            markup = smart_quotes_compiled.sub(self._sub_ms_char, markup)

        try:
            # print("Trying to convert document to %s (errors=%s)" % (
            #    proposed, errors))
            u = self._to_unicode(markup, proposed, errors)
            self.unicode_markup = u
            self.original_encoding = proposed
        except Exception:
            # print("That didn't work!")
            # print(e)
            return None
        # print("Correct encoding: %s" % proposed)
        return self.unicode_markup

    def _to_unicode(
        self, data: bytes, encoding: _Encoding, errors: str = "strict"
    ) -> str:
        """Given a bytestring and its encoding, decodes the string into Unicode.

        :param encoding: The name of an encoding.
        :param errors: An error handling strategy, used when calling `str`.
        """
        return str(data, encoding, errors)

    @property
    def declared_html_encoding(self) -> Optional[_Encoding]:
        """If the markup is an HTML document, returns the encoding, if any,
        declared *inside* the document.
        """
        if not self.is_html:
            return None
        return self.detector.declared_encoding

    def find_codec(self, charset: _Encoding) -> Optional[str]:
        """Look up the Python codec corresponding to a given character set.

        :param charset: The name of a character set.
        :return: The name of a Python codec.
        """
        value = (
            self._codec(self.CHARSET_ALIASES.get(charset, charset))
            or (charset and self._codec(charset.replace("-", "")))
            or (charset and self._codec(charset.replace("-", "_")))
            or (charset and charset.lower())
            or charset
        )
        if value:
            return value.lower()
        return None

    def _codec(self, charset: _Encoding) -> Optional[str]:
        if not charset:
            return charset
        codec = None
        try:
            codecs.lookup(charset)
            codec = charset
        except (LookupError, ValueError):
            pass
        return codec

    #: A partial mapping of ISO-Latin-1 to HTML entities/XML numeric entities.
    #:
    #: :meta hide-value:
    MS_CHARS: Dict[bytes, Union[str, Tuple[str, str]]] = {
        b"\x80": ("euro", "20AC"),
        b"\x81": " ",
        b"\x82": ("sbquo", "201A"),
        b"\x83": ("fnof", "192"),
        b"\x84": ("bdquo", "201E"),
        b"\x85": ("hellip", "2026"),
        b"\x86": ("dagger", "2020"),
        b"\x87": ("Dagger", "2021"),
        b"\x88": ("circ", "2C6"),
        b"\x89": ("permil", "2030"),
        b"\x8a": ("Scaron", "160"),
        b"\x8b": ("lsaquo", "2039"),
        b"\x8c": ("OElig", "152"),
        b"\x8d": "?",
        b"\x8e": ("#x17D", "17D"),
        b"\x8f": "?",
        b"\x90": "?",
        b"\x91": ("lsquo", "2018"),
        b"\x92": ("rsquo", "2019"),
        b"\x93": ("ldquo", "201C"),
        b"\x94": ("rdquo", "201D"),
        b"\x95": ("bull", "2022"),
        b"\x96": ("ndash", "2013"),
        b"\x97": ("mdash", "2014"),
        b"\x98": ("tilde", "2DC"),
        b"\x99": ("trade", "2122"),
        b"\x9a": ("scaron", "161"),
        b"\x9b": ("rsaquo", "203A"),
        b"\x9c": ("oelig", "153"),
        b"\x9d": "?",
        b"\x9e": ("#x17E", "17E"),
        b"\x9f": ("Yuml", ""),
    }

    #: A parochial partial mapping of ISO-Latin-1 to ASCII. Contains
    #: horrors like stripping diacritical marks to turn á into a, but also
    #: contains non-horrors like turning “ into ".
    #:
    #: Seriously, don't use this for anything other than removing smart
    #: quotes.
    #:
    #: :meta private:
    MS_CHARS_TO_ASCII: Dict[bytes, str] = {
        b"\x80": "EUR",
        b"\x81": " ",
        b"\x82": ",",
        b"\x83": "f",
        b"\x84": ",,",
        b"\x85": "...",
        b"\x86": "+",
        b"\x87": "++",
        b"\x88": "^",
        b"\x89": "%",
        b"\x8a": "S",
        b"\x8b": "<",
        b"\x8c": "OE",
        b"\x8d": "?",
        b"\x8e": "Z",
        b"\x8f": "?",
        b"\x90": "?",
        b"\x91": "'",
        b"\x92": "'",
        b"\x93": '"',
        b"\x94": '"',
        b"\x95": "*",
        b"\x96": "-",
        b"\x97": "--",
        b"\x98": "~",
        b"\x99": "(TM)",
        b"\x9a": "s",
        b"\x9b": ">",
        b"\x9c": "oe",
        b"\x9d": "?",
        b"\x9e": "z",
        b"\x9f": "Y",
        b"\xa0": " ",
        b"\xa1": "!",
        b"\xa2": "c",
        b"\xa3": "GBP",
        b"\xa4": "$",  # This approximation is especially parochial--this is the
        # generic currency symbol.
        b"\xa5": "YEN",
        b"\xa6": "|",
        b"\xa7": "S",
        b"\xa8": "..",
        b"\xa9": "",
        b"\xaa": "(th)",
        b"\xab": "<<",
        b"\xac": "!",
        b"\xad": " ",
        b"\xae": "(R)",
        b"\xaf": "-",
        b"\xb0": "o",
        b"\xb1": "+-",
        b"\xb2": "2",
        b"\xb3": "3",
        b"\xb4": "'",
        b"\xb5": "u",
        b"\xb6": "P",
        b"\xb7": "*",
        b"\xb8": ",",
        b"\xb9": "1",
        b"\xba": "(th)",
        b"\xbb": ">>",
        b"\xbc": "1/4",
        b"\xbd": "1/2",
        b"\xbe": "3/4",
        b"\xbf": "?",
        b"\xc0": "A",
        b"\xc1": "A",
        b"\xc2": "A",
        b"\xc3": "A",
        b"\xc4": "A",
        b"\xc5": "A",
        b"\xc6": "AE",
        b"\xc7": "C",
        b"\xc8": "E",
        b"\xc9": "E",
        b"\xca": "E",
        b"\xcb": "E",
        b"\xcc": "I",
        b"\xcd": "I",
        b"\xce": "I",
        b"\xcf": "I",
        b"\xd0": "D",
        b"\xd1": "N",
        b"\xd2": "O",
        b"\xd3": "O",
        b"\xd4": "O",
        b"\xd5": "O",
        b"\xd6": "O",
        b"\xd7": "*",
        b"\xd8": "O",
        b"\xd9": "U",
        b"\xda": "U",
        b"\xdb": "U",
        b"\xdc": "U",
        b"\xdd": "Y",
        b"\xde": "b",
        b"\xdf": "B",
        b"\xe0": "a",
        b"\xe1": "a",
        b"\xe2": "a",
        b"\xe3": "a",
        b"\xe4": "a",
        b"\xe5": "a",
        b"\xe6": "ae",
        b"\xe7": "c",
        b"\xe8": "e",
        b"\xe9": "e",
        b"\xea": "e",
        b"\xeb": "e",
        b"\xec": "i",
        b"\xed": "i",
        b"\xee": "i",
        b"\xef": "i",
        b"\xf0": "o",
        b"\xf1": "n",
        b"\xf2": "o",
        b"\xf3": "o",
        b"\xf4": "o",
        b"\xf5": "o",
        b"\xf6": "o",
        b"\xf7": "/",
        b"\xf8": "o",
        b"\xf9": "u",
        b"\xfa": "u",
        b"\xfb": "u",
        b"\xfc": "u",
        b"\xfd": "y",
        b"\xfe": "b",
        b"\xff": "y",
    }

    #: A map used when removing rogue Windows-1252/ISO-8859-1
    #: characters in otherwise UTF-8 documents.
    #:
    #: Note that \\x81, \\x8d, \\x8f, \\x90, and \\x9d are undefined in
    #: Windows-1252.
    #:
    #: :meta hide-value:
    WINDOWS_1252_TO_UTF8: Dict[int, bytes] = {
        0x80: b"\xe2\x82\xac",  # €
        0x82: b"\xe2\x80\x9a",  # ‚
        0x83: b"\xc6\x92",  # ƒ
        0x84: b"\xe2\x80\x9e",  # „
        0x85: b"\xe2\x80\xa6",  # …
        0x86: b"\xe2\x80\xa0",  # †
        0x87: b"\xe2\x80\xa1",  # ‡
        0x88: b"\xcb\x86",  # ˆ
        0x89: b"\xe2\x80\xb0",  # ‰
        0x8A: b"\xc5\xa0",  # Š
        0x8B: b"\xe2\x80\xb9",  # ‹
        0x8C: b"\xc5\x92",  # Œ
        0x8E: b"\xc5\xbd",  # Ž
        0x91: b"\xe2\x80\x98",  # ‘
        0x92: b"\xe2\x80\x99",  # ’
        0x93: b"\xe2\x80\x9c",  # “
        0x94: b"\xe2\x80\x9d",  # ”
        0x95: b"\xe2\x80\xa2",  # •
        0x96: b"\xe2\x80\x93",  # –
        0x97: b"\xe2\x80\x94",  # —
        0x98: b"\xcb\x9c",  # ˜
        0x99: b"\xe2\x84\xa2",  # ™
        0x9A: b"\xc5\xa1",  # š
        0x9B: b"\xe2\x80\xba",  # ›
        0x9C: b"\xc5\x93",  # œ
        0x9E: b"\xc5\xbe",  # ž
        0x9F: b"\xc5\xb8",  # Ÿ
        0xA0: b"\xc2\xa0",  #
        0xA1: b"\xc2\xa1",  # ¡
        0xA2: b"\xc2\xa2",  # ¢
        0xA3: b"\xc2\xa3",  # £
        0xA4: b"\xc2\xa4",  # ¤
        0xA5: b"\xc2\xa5",  # ¥
        0xA6: b"\xc2\xa6",  # ¦
        0xA7: b"\xc2\xa7",  # §
        0xA8: b"\xc2\xa8",  # ¨
        0xA9: b"\xc2\xa9",  # ©
        0xAA: b"\xc2\xaa",  # ª
        0xAB: b"\xc2\xab",  # «
        0xAC: b"\xc2\xac",  # ¬
        0xAD: b"\xc2\xad",  # ­
        0xAE: b"\xc2\xae",  # ®
        0xAF: b"\xc2\xaf",  # ¯
        0xB0: b"\xc2\xb0",  # °
        0xB1: b"\xc2\xb1",  # ±
        0xB2: b"\xc2\xb2",  # ²
        0xB3: b"\xc2\xb3",  # ³
        0xB4: b"\xc2\xb4",  # ´
        0xB5: b"\xc2\xb5",  # µ
        0xB6: b"\xc2\xb6",  # ¶
        0xB7: b"\xc2\xb7",  # ·
        0xB8: b"\xc2\xb8",  # ¸
        0xB9: b"\xc2\xb9",  # ¹
        0xBA: b"\xc2\xba",  # º
        0xBB: b"\xc2\xbb",  # »
        0xBC: b"\xc2\xbc",  # ¼
        0xBD: b"\xc2\xbd",  # ½
        0xBE: b"\xc2\xbe",  # ¾
        0xBF: b"\xc2\xbf",  # ¿
        0xC0: b"\xc3\x80",  # À
        0xC1: b"\xc3\x81",  # Á
        0xC2: b"\xc3\x82",  # Â
        0xC3: b"\xc3\x83",  # Ã
        0xC4: b"\xc3\x84",  # Ä
        0xC5: b"\xc3\x85",  # Å
        0xC6: b"\xc3\x86",  # Æ
        0xC7: b"\xc3\x87",  # Ç
        0xC8: b"\xc3\x88",  # È
        0xC9: b"\xc3\x89",  # É
        0xCA: b"\xc3\x8a",  # Ê
        0xCB: b"\xc3\x8b",  # Ë
        0xCC: b"\xc3\x8c",  # Ì
        0xCD: b"\xc3\x8d",  # Í
        0xCE: b"\xc3\x8e",  # Î
        0xCF: b"\xc3\x8f",  # Ï
        0xD0: b"\xc3\x90",  # Ð
        0xD1: b"\xc3\x91",  # Ñ
        0xD2: b"\xc3\x92",  # Ò
        0xD3: b"\xc3\x93",  # Ó
        0xD4: b"\xc3\x94",  # Ô
        0xD5: b"\xc3\x95",  # Õ
        0xD6: b"\xc3\x96",  # Ö
        0xD7: b"\xc3\x97",  # ×
        0xD8: b"\xc3\x98",  # Ø
        0xD9: b"\xc3\x99",  # Ù
        0xDA: b"\xc3\x9a",  # Ú
        0xDB: b"\xc3\x9b",  # Û
        0xDC: b"\xc3\x9c",  # Ü
        0xDD: b"\xc3\x9d",  # Ý
        0xDE: b"\xc3\x9e",  # Þ
        0xDF: b"\xc3\x9f",  # ß
        0xE0: b"\xc3\xa0",  # à
        0xE1: b"\xa1",  # á
        0xE2: b"\xc3\xa2",  # â
        0xE3: b"\xc3\xa3",  # ã
        0xE4: b"\xc3\xa4",  # ä
        0xE5: b"\xc3\xa5",  # å
        0xE6: b"\xc3\xa6",  # æ
        0xE7: b"\xc3\xa7",  # ç
        0xE8: b"\xc3\xa8",  # è
        0xE9: b"\xc3\xa9",  # é
        0xEA: b"\xc3\xaa",  # ê
        0xEB: b"\xc3\xab",  # ë
        0xEC: b"\xc3\xac",  # ì
        0xED: b"\xc3\xad",  # í
        0xEE: b"\xc3\xae",  # î
        0xEF: b"\xc3\xaf",  # ï
        0xF0: b"\xc3\xb0",  # ð
        0xF1: b"\xc3\xb1",  # ñ
        0xF2: b"\xc3\xb2",  # ò
        0xF3: b"\xc3\xb3",  # ó
        0xF4: b"\xc3\xb4",  # ô
        0xF5: b"\xc3\xb5",  # õ
        0xF6: b"\xc3\xb6",  # ö
        0xF7: b"\xc3\xb7",  # ÷
        0xF8: b"\xc3\xb8",  # ø
        0xF9: b"\xc3\xb9",  # ù
        0xFA: b"\xc3\xba",  # ú
        0xFB: b"\xc3\xbb",  # û
        0xFC: b"\xc3\xbc",  # ü
        0xFD: b"\xc3\xbd",  # ý
        0xFE: b"\xc3\xbe",  # þ
    }

    #: :meta private:
    MULTIBYTE_MARKERS_AND_SIZES: List[Tuple[int, int, int]] = [
        (0xC2, 0xDF, 2),  # 2-byte characters start with a byte C2-DF
        (0xE0, 0xEF, 3),  # 3-byte characters start with E0-EF
        (0xF0, 0xF4, 4),  # 4-byte characters start with F0-F4
    ]

    #: :meta private:
    FIRST_MULTIBYTE_MARKER: int = MULTIBYTE_MARKERS_AND_SIZES[0][0]

    #: :meta private:
    LAST_MULTIBYTE_MARKER: int = MULTIBYTE_MARKERS_AND_SIZES[-1][1]

    @classmethod
    def detwingle(
        cls,
        in_bytes: bytes,
        main_encoding: _Encoding = "utf8",
        embedded_encoding: _Encoding = "windows-1252",
    ) -> bytes:
        """Fix characters from one encoding embedded in some other encoding.

        Currently the only situation supported is Windows-1252 (or its
        subset ISO-8859-1), embedded in UTF-8.

        :param in_bytes: A bytestring that you suspect contains
            characters from multiple encodings. Note that this *must*
            be a bytestring. If you've already converted the document
            to Unicode, you're too late.
        :param main_encoding: The primary encoding of ``in_bytes``.
        :param embedded_encoding: The encoding that was used to embed characters
            in the main document.
        :return: A bytestring similar to ``in_bytes``, in which
          ``embedded_encoding`` characters have been converted to
          their ``main_encoding`` equivalents.
        """
        if embedded_encoding.replace("_", "-").lower() not in (
            "windows-1252",
            "windows_1252",
        ):
            raise NotImplementedError(
                "Windows-1252 and ISO-8859-1 are the only currently supported "
                "embedded encodings."
            )

        if main_encoding.lower() not in ("utf8", "utf-8"):
            raise NotImplementedError(
                "UTF-8 is the only currently supported main encoding."
            )

        byte_chunks = []

        chunk_start = 0
        pos = 0
        while pos < len(in_bytes):
            byte = in_bytes[pos]
            if byte >= cls.FIRST_MULTIBYTE_MARKER and byte <= cls.LAST_MULTIBYTE_MARKER:
                # This is the start of a UTF-8 multibyte character. Skip
                # to the end.
                for start, end, size in cls.MULTIBYTE_MARKERS_AND_SIZES:
                    if byte >= start and byte <= end:
                        pos += size
                        break
            elif byte >= 0x80 and byte in cls.WINDOWS_1252_TO_UTF8:
                # We found a Windows-1252 character!
                # Save the string up to this point as a chunk.
                byte_chunks.append(in_bytes[chunk_start:pos])

                # Now translate the Windows-1252 character into UTF-8
                # and add it as another, one-byte chunk.
                byte_chunks.append(cls.WINDOWS_1252_TO_UTF8[byte])
                pos += 1
                chunk_start = pos
            else:
                # Go on to the next character.
                pos += 1
        if chunk_start == 0:
            # The string is unchanged.
            return in_bytes
        else:
            # Store the final chunk.
            byte_chunks.append(in_bytes[chunk_start:])
        return b"".join(byte_chunks)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\http.py ===
from __future__ import annotations

import email.utils
import re
import typing as t
import warnings
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from enum import Enum
from hashlib import sha1
from time import mktime
from time import struct_time
from urllib.parse import quote
from urllib.parse import unquote
from urllib.request import parse_http_list as _parse_list_header

from ._internal import _dt_as_utc
from ._internal import _plain_int

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIEnvironment

_token_chars = frozenset(
    "!#$%&'*+-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ^_`abcdefghijklmnopqrstuvwxyz|~"
)
_etag_re = re.compile(r'([Ww]/)?(?:"(.*?)"|(.*?))(?:\s*,\s*|$)')
_entity_headers = frozenset(
    [
        "allow",
        "content-encoding",
        "content-language",
        "content-length",
        "content-location",
        "content-md5",
        "content-range",
        "content-type",
        "expires",
        "last-modified",
    ]
)
_hop_by_hop_headers = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ]
)
HTTP_STATUS_CODES = {
    100: "Continue",
    101: "Switching Protocols",
    102: "Processing",
    103: "Early Hints",  # see RFC 8297
    200: "OK",
    201: "Created",
    202: "Accepted",
    203: "Non Authoritative Information",
    204: "No Content",
    205: "Reset Content",
    206: "Partial Content",
    207: "Multi Status",
    208: "Already Reported",  # see RFC 5842
    226: "IM Used",  # see RFC 3229
    300: "Multiple Choices",
    301: "Moved Permanently",
    302: "Found",
    303: "See Other",
    304: "Not Modified",
    305: "Use Proxy",
    306: "Switch Proxy",  # unused
    307: "Temporary Redirect",
    308: "Permanent Redirect",
    400: "Bad Request",
    401: "Unauthorized",
    402: "Payment Required",  # unused
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    406: "Not Acceptable",
    407: "Proxy Authentication Required",
    408: "Request Timeout",
    409: "Conflict",
    410: "Gone",
    411: "Length Required",
    412: "Precondition Failed",
    413: "Request Entity Too Large",
    414: "Request URI Too Long",
    415: "Unsupported Media Type",
    416: "Requested Range Not Satisfiable",
    417: "Expectation Failed",
    418: "I'm a teapot",  # see RFC 2324
    421: "Misdirected Request",  # see RFC 7540
    422: "Unprocessable Entity",
    423: "Locked",
    424: "Failed Dependency",
    425: "Too Early",  # see RFC 8470
    426: "Upgrade Required",
    428: "Precondition Required",  # see RFC 6585
    429: "Too Many Requests",
    431: "Request Header Fields Too Large",
    449: "Retry With",  # proprietary MS extension
    451: "Unavailable For Legal Reasons",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
    505: "HTTP Version Not Supported",
    506: "Variant Also Negotiates",  # see RFC 2295
    507: "Insufficient Storage",
    508: "Loop Detected",  # see RFC 5842
    510: "Not Extended",
    511: "Network Authentication Failed",
}


class COEP(Enum):
    """Cross Origin Embedder Policies"""

    UNSAFE_NONE = "unsafe-none"
    REQUIRE_CORP = "require-corp"


class COOP(Enum):
    """Cross Origin Opener Policies"""

    UNSAFE_NONE = "unsafe-none"
    SAME_ORIGIN_ALLOW_POPUPS = "same-origin-allow-popups"
    SAME_ORIGIN = "same-origin"


def quote_header_value(value: t.Any, allow_token: bool = True) -> str:
    """Add double quotes around a header value. If the header contains only ASCII token
    characters, it will be returned unchanged. If the header contains ``"`` or ``\\``
    characters, they will be escaped with an additional ``\\`` character.

    This is the reverse of :func:`unquote_header_value`.

    :param value: The value to quote. Will be converted to a string.
    :param allow_token: Disable to quote the value even if it only has token characters.

    .. versionchanged:: 3.0
        Passing bytes is not supported.

    .. versionchanged:: 3.0
        The ``extra_chars`` parameter is removed.

    .. versionchanged:: 2.3
        The value is quoted if it is the empty string.

    .. versionadded:: 0.5
    """
    value_str = str(value)

    if not value_str:
        return '""'

    if allow_token:
        token_chars = _token_chars

        if token_chars.issuperset(value_str):
            return value_str

    value_str = value_str.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value_str}"'


def unquote_header_value(value: str) -> str:
    """Remove double quotes and decode slash-escaped ``"`` and ``\\`` characters in a
    header value.

    This is the reverse of :func:`quote_header_value`.

    :param value: The header value to unquote.

    .. versionchanged:: 3.0
        The ``is_filename`` parameter is removed.
    """
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
        return value.replace("\\\\", "\\").replace('\\"', '"')

    return value


def dump_options_header(header: str | None, options: t.Mapping[str, t.Any]) -> str:
    """Produce a header value and ``key=value`` parameters separated by semicolons
    ``;``. For example, the ``Content-Type`` header.

    .. code-block:: python

        dump_options_header("text/html", {"charset": "UTF-8"})
        'text/html; charset=UTF-8'

    This is the reverse of :func:`parse_options_header`.

    If a value contains non-token characters, it will be quoted.

    If a value is ``None``, the parameter is skipped.

    In some keys for some headers, a UTF-8 value can be encoded using a special
    ``key*=UTF-8''value`` form, where ``value`` is percent encoded. This function will
    not produce that format automatically, but if a given key ends with an asterisk
    ``*``, the value is assumed to have that form and will not be quoted further.

    :param header: The primary header value.
    :param options: Parameters to encode as ``key=value`` pairs.

    .. versionchanged:: 2.3
        Keys with ``None`` values are skipped rather than treated as a bare key.

    .. versionchanged:: 2.2.3
        If a key ends with ``*``, its value will not be quoted.
    """
    segments = []

    if header is not None:
        segments.append(header)

    for key, value in options.items():
        if value is None:
            continue

        if key[-1] == "*":
            segments.append(f"{key}={value}")
        else:
            segments.append(f"{key}={quote_header_value(value)}")

    return "; ".join(segments)


def dump_header(iterable: dict[str, t.Any] | t.Iterable[t.Any]) -> str:
    """Produce a header value from a list of items or ``key=value`` pairs, separated by
    commas ``,``.

    This is the reverse of :func:`parse_list_header`, :func:`parse_dict_header`, and
    :func:`parse_set_header`.

    If a value contains non-token characters, it will be quoted.

    If a value is ``None``, the key is output alone.

    In some keys for some headers, a UTF-8 value can be encoded using a special
    ``key*=UTF-8''value`` form, where ``value`` is percent encoded. This function will
    not produce that format automatically, but if a given key ends with an asterisk
    ``*``, the value is assumed to have that form and will not be quoted further.

    .. code-block:: python

        dump_header(["foo", "bar baz"])
        'foo, "bar baz"'

        dump_header({"foo": "bar baz"})
        'foo="bar baz"'

    :param iterable: The items to create a header from.

    .. versionchanged:: 3.0
        The ``allow_token`` parameter is removed.

    .. versionchanged:: 2.2.3
        If a key ends with ``*``, its value will not be quoted.
    """
    if isinstance(iterable, dict):
        items = []

        for key, value in iterable.items():
            if value is None:
                items.append(key)
            elif key[-1] == "*":
                items.append(f"{key}={value}")
            else:
                items.append(f"{key}={quote_header_value(value)}")
    else:
        items = [quote_header_value(x) for x in iterable]

    return ", ".join(items)


def dump_csp_header(header: ds.ContentSecurityPolicy) -> str:
    """Dump a Content Security Policy header.

    These are structured into policies such as "default-src 'self';
    script-src 'self'".

    .. versionadded:: 1.0.0
       Support for Content Security Policy headers was added.

    """
    return "; ".join(f"{key} {value}" for key, value in header.items())


def parse_list_header(value: str) -> list[str]:
    """Parse a header value that consists of a list of comma separated items according
    to `RFC 9110 <https://httpwg.org/specs/rfc9110.html#abnf.extension>`__.

    This extends :func:`urllib.request.parse_http_list` to remove surrounding quotes
    from values.

    .. code-block:: python

        parse_list_header('token, "quoted value"')
        ['token', 'quoted value']

    This is the reverse of :func:`dump_header`.

    :param value: The header value to parse.
    """
    result = []

    for item in _parse_list_header(value):
        if len(item) >= 2 and item[0] == item[-1] == '"':
            item = item[1:-1]

        result.append(item)

    return result


def parse_dict_header(value: str) -> dict[str, str | None]:
    """Parse a list header using :func:`parse_list_header`, then parse each item as a
    ``key=value`` pair.

    .. code-block:: python

        parse_dict_header('a=b, c="d, e", f')
        {"a": "b", "c": "d, e", "f": None}

    This is the reverse of :func:`dump_header`.

    If a key does not have a value, it is ``None``.

    This handles charsets for values as described in
    `RFC 2231 <https://www.rfc-editor.org/rfc/rfc2231#section-3>`__. Only ASCII, UTF-8,
    and ISO-8859-1 charsets are accepted, otherwise the value remains quoted.

    :param value: The header value to parse.

    .. versionchanged:: 3.0
        Passing bytes is not supported.

    .. versionchanged:: 3.0
        The ``cls`` argument is removed.

    .. versionchanged:: 2.3
        Added support for ``key*=charset''value`` encoded items.

    .. versionchanged:: 0.9
       The ``cls`` argument was added.
    """
    result: dict[str, str | None] = {}

    for item in parse_list_header(value):
        key, has_value, value = item.partition("=")
        key = key.strip()

        if not key:
            # =value is not valid
            continue

        if not has_value:
            result[key] = None
            continue

        value = value.strip()
        encoding: str | None = None

        if key[-1] == "*":
            # key*=charset''value becomes key=value, where value is percent encoded
            # adapted from parse_options_header, without the continuation handling
            key = key[:-1]
            match = _charset_value_re.match(value)

            if match:
                # If there is a charset marker in the value, split it off.
                encoding, value = match.groups()
                encoding = encoding.lower()

            # A safe list of encodings. Modern clients should only send ASCII or UTF-8.
            # This list will not be extended further. An invalid encoding will leave the
            # value quoted.
            if encoding in {"ascii", "us-ascii", "utf-8", "iso-8859-1"}:
                # invalid bytes are replaced during unquoting
                value = unquote(value, encoding=encoding)

        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]

        result[key] = value

    return result


# https://httpwg.org/specs/rfc9110.html#parameter
_parameter_key_re = re.compile(r"([\w!#$%&'*+\-.^`|~]+)=", flags=re.ASCII)
_parameter_token_value_re = re.compile(r"[\w!#$%&'*+\-.^`|~]+", flags=re.ASCII)
# https://www.rfc-editor.org/rfc/rfc2231#section-4
_charset_value_re = re.compile(
    r"""
    ([\w!#$%&*+\-.^`|~]*)'  # charset part, could be empty
    [\w!#$%&*+\-.^`|~]*'  # don't care about language part, usually empty
    ([\w!#$%&'*+\-.^`|~]+)  # one or more token chars with percent encoding
    """,
    re.ASCII | re.VERBOSE,
)
# https://www.rfc-editor.org/rfc/rfc2231#section-3
_continuation_re = re.compile(r"\*(\d+)$", re.ASCII)


def parse_options_header(value: str | None) -> tuple[str, dict[str, str]]:
    """Parse a header that consists of a value with ``key=value`` parameters separated
    by semicolons ``;``. For example, the ``Content-Type`` header.

    .. code-block:: python

        parse_options_header("text/html; charset=UTF-8")
        ('text/html', {'charset': 'UTF-8'})

        parse_options_header("")
        ("", {})

    This is the reverse of :func:`dump_options_header`.

    This parses valid parameter parts as described in
    `RFC 9110 <https://httpwg.org/specs/rfc9110.html#parameter>`__. Invalid parts are
    skipped.

    This handles continuations and charsets as described in
    `RFC 2231 <https://www.rfc-editor.org/rfc/rfc2231#section-3>`__, although not as
    strictly as the RFC. Only ASCII, UTF-8, and ISO-8859-1 charsets are accepted,
    otherwise the value remains quoted.

    Clients may not be consistent in how they handle a quote character within a quoted
    value. The `HTML Standard <https://html.spec.whatwg.org/#multipart-form-data>`__
    replaces it with ``%22`` in multipart form data.
    `RFC 9110 <https://httpwg.org/specs/rfc9110.html#quoted.strings>`__ uses backslash
    escapes in HTTP headers. Both are decoded to the ``"`` character.

    Clients may not be consistent in how they handle non-ASCII characters. HTML
    documents must declare ``<meta charset=UTF-8>``, otherwise browsers may replace with
    HTML character references, which can be decoded using :func:`html.unescape`.

    :param value: The header value to parse.
    :return: ``(value, options)``, where ``options`` is a dict

    .. versionchanged:: 2.3
        Invalid parts, such as keys with no value, quoted keys, and incorrectly quoted
        values, are discarded instead of treating as ``None``.

    .. versionchanged:: 2.3
        Only ASCII, UTF-8, and ISO-8859-1 are accepted for charset values.

    .. versionchanged:: 2.3
        Escaped quotes in quoted values, like ``%22`` and ``\\"``, are handled.

    .. versionchanged:: 2.2
        Option names are always converted to lowercase.

    .. versionchanged:: 2.2
        The ``multiple`` parameter was removed.

    .. versionchanged:: 0.15
        :rfc:`2231` parameter continuations are handled.

    .. versionadded:: 0.5
    """
    if value is None:
        return "", {}

    value, _, rest = value.partition(";")
    value = value.strip()
    rest = rest.strip()

    if not value or not rest:
        # empty (invalid) value, or value without options
        return value, {}

    # Collect all valid key=value parts without processing the value.
    parts: list[tuple[str, str]] = []

    while True:
        if (m := _parameter_key_re.match(rest)) is not None:
            pk = m.group(1).lower()
            rest = rest[m.end() :]

            # Value may be a token.
            if (m := _parameter_token_value_re.match(rest)) is not None:
                parts.append((pk, m.group()))

            # Value may be a quoted string, find the closing quote.
            elif rest[:1] == '"':
                pos = 1
                length = len(rest)

                while pos < length:
                    if rest[pos : pos + 2] in {"\\\\", '\\"'}:
                        # Consume escaped slashes and quotes.
                        pos += 2
                    elif rest[pos] == '"':
                        # Stop at an unescaped quote.
                        parts.append((pk, rest[: pos + 1]))
                        rest = rest[pos + 1 :]
                        break
                    else:
                        # Consume any other character.
                        pos += 1

        # Find the next section delimited by `;`, if any.
        if (end := rest.find(";")) == -1:
            break

        rest = rest[end + 1 :].lstrip()

    options: dict[str, str] = {}
    encoding: str | None = None
    continued_encoding: str | None = None

    # For each collected part, process optional charset and continuation,
    # unquote quoted values.
    for pk, pv in parts:
        if pk[-1] == "*":
            # key*=charset''value becomes key=value, where value is percent encoded
            pk = pk[:-1]
            match = _charset_value_re.match(pv)

            if match:
                # If there is a valid charset marker in the value, split it off.
                encoding, pv = match.groups()
                # This might be the empty string, handled next.
                encoding = encoding.lower()

            # No charset marker, or marker with empty charset value.
            if not encoding:
                encoding = continued_encoding

            # A safe list of encodings. Modern clients should only send ASCII or UTF-8.
            # This list will not be extended further. An invalid encoding will leave the
            # value quoted.
            if encoding in {"ascii", "us-ascii", "utf-8", "iso-8859-1"}:
                # Continuation parts don't require their own charset marker. This is
                # looser than the RFC, it will persist across different keys and allows
                # changing the charset during a continuation. But this implementation is
                # much simpler than tracking the full state.
                continued_encoding = encoding
                # invalid bytes are replaced during unquoting
                pv = unquote(pv, encoding=encoding)

        # Remove quotes. At this point the value cannot be empty or a single quote.
        if pv[0] == pv[-1] == '"':
            # HTTP headers use slash, multipart form data uses percent
            pv = pv[1:-1].replace("\\\\", "\\").replace('\\"', '"').replace("%22", '"')

        match = _continuation_re.search(pk)

        if match:
            # key*0=a; key*1=b becomes key=ab
            pk = pk[: match.start()]
            options[pk] = options.get(pk, "") + pv
        else:
            options[pk] = pv

    return value, options


_q_value_re = re.compile(r"-?\d+(\.\d+)?", re.ASCII)
_TAnyAccept = t.TypeVar("_TAnyAccept", bound="ds.Accept")


@t.overload
def parse_accept_header(value: str | None) -> ds.Accept: ...


@t.overload
def parse_accept_header(value: str | None, cls: type[_TAnyAccept]) -> _TAnyAccept: ...


def parse_accept_header(
    value: str | None, cls: type[_TAnyAccept] | None = None
) -> _TAnyAccept:
    """Parse an ``Accept`` header according to
    `RFC 9110 <https://httpwg.org/specs/rfc9110.html#field.accept>`__.

    Returns an :class:`.Accept` instance, which can sort and inspect items based on
    their quality parameter. When parsing ``Accept-Charset``, ``Accept-Encoding``, or
    ``Accept-Language``, pass the appropriate :class:`.Accept` subclass.

    :param value: The header value to parse.
    :param cls: The :class:`.Accept` class to wrap the result in.
    :return: An instance of ``cls``.

    .. versionchanged:: 2.3
        Parse according to RFC 9110. Items with invalid ``q`` values are skipped.
    """
    if cls is None:
        cls = t.cast(type[_TAnyAccept], ds.Accept)

    if not value:
        return cls(None)

    result = []

    for item in parse_list_header(value):
        item, options = parse_options_header(item)

        if "q" in options:
            # pop q, remaining options are reconstructed
            q_str = options.pop("q").strip()

            if _q_value_re.fullmatch(q_str) is None:
                # ignore an invalid q
                continue

            q = float(q_str)

            if q < 0 or q > 1:
                # ignore an invalid q
                continue
        else:
            q = 1

        if options:
            # reconstruct the media type with any options
            item = dump_options_header(item, options)

        result.append((item, q))

    return cls(result)


_TAnyCC = t.TypeVar("_TAnyCC", bound="ds.cache_control._CacheControl")


@t.overload
def parse_cache_control_header(
    value: str | None,
    on_update: t.Callable[[ds.cache_control._CacheControl], None] | None = None,
) -> ds.RequestCacheControl: ...


@t.overload
def parse_cache_control_header(
    value: str | None,
    on_update: t.Callable[[ds.cache_control._CacheControl], None] | None = None,
    cls: type[_TAnyCC] = ...,
) -> _TAnyCC: ...


def parse_cache_control_header(
    value: str | None,
    on_update: t.Callable[[ds.cache_control._CacheControl], None] | None = None,
    cls: type[_TAnyCC] | None = None,
) -> _TAnyCC:
    """Parse a cache control header.  The RFC differs between response and
    request cache control, this method does not.  It's your responsibility
    to not use the wrong control statements.

    .. versionadded:: 0.5
       The `cls` was added.  If not specified an immutable
       :class:`~werkzeug.datastructures.RequestCacheControl` is returned.

    :param value: a cache control header to be parsed.
    :param on_update: an optional callable that is called every time a value
                      on the :class:`~werkzeug.datastructures.CacheControl`
                      object is changed.
    :param cls: the class for the returned object.  By default
                :class:`~werkzeug.datastructures.RequestCacheControl` is used.
    :return: a `cls` object.
    """
    if cls is None:
        cls = t.cast("type[_TAnyCC]", ds.RequestCacheControl)

    if not value:
        return cls((), on_update)

    return cls(parse_dict_header(value), on_update)


_TAnyCSP = t.TypeVar("_TAnyCSP", bound="ds.ContentSecurityPolicy")


@t.overload
def parse_csp_header(
    value: str | None,
    on_update: t.Callable[[ds.ContentSecurityPolicy], None] | None = None,
) -> ds.ContentSecurityPolicy: ...


@t.overload
def parse_csp_header(
    value: str | None,
    on_update: t.Callable[[ds.ContentSecurityPolicy], None] | None = None,
    cls: type[_TAnyCSP] = ...,
) -> _TAnyCSP: ...


def parse_csp_header(
    value: str | None,
    on_update: t.Callable[[ds.ContentSecurityPolicy], None] | None = None,
    cls: type[_TAnyCSP] | None = None,
) -> _TAnyCSP:
    """Parse a Content Security Policy header.

    .. versionadded:: 1.0.0
       Support for Content Security Policy headers was added.

    :param value: a csp header to be parsed.
    :param on_update: an optional callable that is called every time a value
                      on the object is changed.
    :param cls: the class for the returned object.  By default
                :class:`~werkzeug.datastructures.ContentSecurityPolicy` is used.
    :return: a `cls` object.
    """
    if cls is None:
        cls = t.cast("type[_TAnyCSP]", ds.ContentSecurityPolicy)

    if value is None:
        return cls((), on_update)

    items = []

    for policy in value.split(";"):
        policy = policy.strip()

        # Ignore badly formatted policies (no space)
        if " " in policy:
            directive, value = policy.strip().split(" ", 1)
            items.append((directive.strip(), value.strip()))

    return cls(items, on_update)


def parse_set_header(
    value: str | None,
    on_update: t.Callable[[ds.HeaderSet], None] | None = None,
) -> ds.HeaderSet:
    """Parse a set-like header and return a
    :class:`~werkzeug.datastructures.HeaderSet` object:

    >>> hs = parse_set_header('token, "quoted value"')

    The return value is an object that treats the items case-insensitively
    and keeps the order of the items:

    >>> 'TOKEN' in hs
    True
    >>> hs.index('quoted value')
    1
    >>> hs
    HeaderSet(['token', 'quoted value'])

    To create a header from the :class:`HeaderSet` again, use the
    :func:`dump_header` function.

    :param value: a set header to be parsed.
    :param on_update: an optional callable that is called every time a
                      value on the :class:`~werkzeug.datastructures.HeaderSet`
                      object is changed.
    :return: a :class:`~werkzeug.datastructures.HeaderSet`
    """
    if not value:
        return ds.HeaderSet(None, on_update)
    return ds.HeaderSet(parse_list_header(value), on_update)


def parse_if_range_header(value: str | None) -> ds.IfRange:
    """Parses an if-range header which can be an etag or a date.  Returns
    a :class:`~werkzeug.datastructures.IfRange` object.

    .. versionchanged:: 2.0
        If the value represents a datetime, it is timezone-aware.

    .. versionadded:: 0.7
    """
    if not value:
        return ds.IfRange()
    date = parse_date(value)
    if date is not None:
        return ds.IfRange(date=date)
    # drop weakness information
    return ds.IfRange(unquote_etag(value)[0])


def parse_range_header(
    value: str | None, make_inclusive: bool = True
) -> ds.Range | None:
    """Parses a range header into a :class:`~werkzeug.datastructures.Range`
    object.  If the header is missing or malformed `None` is returned.
    `ranges` is a list of ``(start, stop)`` tuples where the ranges are
    non-inclusive.

    .. versionadded:: 0.7
    """
    if not value or "=" not in value:
        return None

    ranges = []
    last_end = 0
    units, rng = value.split("=", 1)
    units = units.strip().lower()

    for item in rng.split(","):
        item = item.strip()
        if "-" not in item:
            return None
        if item.startswith("-"):
            if last_end < 0:
                return None
            try:
                begin = _plain_int(item)
            except ValueError:
                return None
            end = None
            last_end = -1
        elif "-" in item:
            begin_str, end_str = item.split("-", 1)
            begin_str = begin_str.strip()
            end_str = end_str.strip()

            try:
                begin = _plain_int(begin_str)
            except ValueError:
                return None

            if begin < last_end or last_end < 0:
                return None
            if end_str:
                try:
                    end = _plain_int(end_str) + 1
                except ValueError:
                    return None

                if begin >= end:
                    return None
            else:
                end = None
            last_end = end if end is not None else -1
        ranges.append((begin, end))

    return ds.Range(units, ranges)


def parse_content_range_header(
    value: str | None,
    on_update: t.Callable[[ds.ContentRange], None] | None = None,
) -> ds.ContentRange | None:
    """Parses a range header into a
    :class:`~werkzeug.datastructures.ContentRange` object or `None` if
    parsing is not possible.

    .. versionadded:: 0.7

    :param value: a content range header to be parsed.
    :param on_update: an optional callable that is called every time a value
                      on the :class:`~werkzeug.datastructures.ContentRange`
                      object is changed.
    """
    if value is None:
        return None
    try:
        units, rangedef = (value or "").strip().split(None, 1)
    except ValueError:
        return None

    if "/" not in rangedef:
        return None
    rng, length_str = rangedef.split("/", 1)
    if length_str == "*":
        length = None
    else:
        try:
            length = _plain_int(length_str)
        except ValueError:
            return None

    if rng == "*":
        if not is_byte_range_valid(None, None, length):
            return None

        return ds.ContentRange(units, None, None, length, on_update=on_update)
    elif "-" not in rng:
        return None

    start_str, stop_str = rng.split("-", 1)
    try:
        start = _plain_int(start_str)
        stop = _plain_int(stop_str) + 1
    except ValueError:
        return None

    if is_byte_range_valid(start, stop, length):
        return ds.ContentRange(units, start, stop, length, on_update=on_update)

    return None


def quote_etag(etag: str, weak: bool = False) -> str:
    """Quote an etag.

    :param etag: the etag to quote.
    :param weak: set to `True` to tag it "weak".
    """
    if '"' in etag:
        raise ValueError("invalid etag")
    etag = f'"{etag}"'
    if weak:
        etag = f"W/{etag}"
    return etag


@t.overload
def unquote_etag(etag: str) -> tuple[str, bool]: ...
@t.overload
def unquote_etag(etag: None) -> tuple[None, None]: ...
def unquote_etag(
    etag: str | None,
) -> tuple[str, bool] | tuple[None, None]:
    """Unquote a single etag:

    >>> unquote_etag('W/"bar"')
    ('bar', True)
    >>> unquote_etag('"bar"')
    ('bar', False)

    :param etag: the etag identifier to unquote.
    :return: a ``(etag, weak)`` tuple.
    """
    if not etag:
        return None, None
    etag = etag.strip()
    weak = False
    if etag.startswith(("W/", "w/")):
        weak = True
        etag = etag[2:]
    if etag[:1] == etag[-1:] == '"':
        etag = etag[1:-1]
    return etag, weak


def parse_etags(value: str | None) -> ds.ETags:
    """Parse an etag header.

    :param value: the tag header to parse
    :return: an :class:`~werkzeug.datastructures.ETags` object.
    """
    if not value:
        return ds.ETags()
    strong = []
    weak = []
    end = len(value)
    pos = 0
    while pos < end:
        match = _etag_re.match(value, pos)
        if match is None:
            break
        is_weak, quoted, raw = match.groups()
        if raw == "*":
            return ds.ETags(star_tag=True)
        elif quoted:
            raw = quoted
        if is_weak:
            weak.append(raw)
        else:
            strong.append(raw)
        pos = match.end()
    return ds.ETags(strong, weak)


def generate_etag(data: bytes) -> str:
    """Generate an etag for some data.

    .. versionchanged:: 2.0
        Use SHA-1. MD5 may not be available in some environments.
    """
    return sha1(data).hexdigest()


def parse_date(value: str | None) -> datetime | None:
    """Parse an :rfc:`2822` date into a timezone-aware
    :class:`datetime.datetime` object, or ``None`` if parsing fails.

    This is a wrapper for :func:`email.utils.parsedate_to_datetime`. It
    returns ``None`` if parsing fails instead of raising an exception,
    and always returns a timezone-aware datetime object. If the string
    doesn't have timezone information, it is assumed to be UTC.

    :param value: A string with a supported date format.

    .. versionchanged:: 2.0
        Return a timezone-aware datetime object. Use
        ``email.utils.parsedate_to_datetime``.
    """
    if value is None:
        return None

    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt


def http_date(
    timestamp: datetime | date | int | float | struct_time | None = None,
) -> str:
    """Format a datetime object or timestamp into an :rfc:`2822` date
    string.

    This is a wrapper for :func:`email.utils.format_datetime`. It
    assumes naive datetime objects are in UTC instead of raising an
    exception.

    :param timestamp: The datetime or timestamp to format. Defaults to
        the current time.

    .. versionchanged:: 2.0
        Use ``email.utils.format_datetime``. Accept ``date`` objects.
    """
    if isinstance(timestamp, date):
        if not isinstance(timestamp, datetime):
            # Assume plain date is midnight UTC.
            timestamp = datetime.combine(timestamp, time(), tzinfo=timezone.utc)
        else:
            # Ensure datetime is timezone-aware.
            timestamp = _dt_as_utc(timestamp)

        return email.utils.format_datetime(timestamp, usegmt=True)

    if isinstance(timestamp, struct_time):
        timestamp = mktime(timestamp)

    return email.utils.formatdate(timestamp, usegmt=True)


def parse_age(value: str | None = None) -> timedelta | None:
    """Parses a base-10 integer count of seconds into a timedelta.

    If parsing fails, the return value is `None`.

    :param value: a string consisting of an integer represented in base-10
    :return: a :class:`datetime.timedelta` object or `None`.
    """
    if not value:
        return None
    try:
        seconds = int(value)
    except ValueError:
        return None
    if seconds < 0:
        return None
    try:
        return timedelta(seconds=seconds)
    except OverflowError:
        return None


def dump_age(age: timedelta | int | None = None) -> str | None:
    """Formats the duration as a base-10 integer.

    :param age: should be an integer number of seconds,
                a :class:`datetime.timedelta` object, or,
                if the age is unknown, `None` (default).
    """
    if age is None:
        return None
    if isinstance(age, timedelta):
        age = int(age.total_seconds())
    else:
        age = int(age)

    if age < 0:
        raise ValueError("age cannot be negative")

    return str(age)


def is_resource_modified(
    environ: WSGIEnvironment,
    etag: str | None = None,
    data: bytes | None = None,
    last_modified: datetime | str | None = None,
    ignore_if_range: bool = True,
) -> bool:
    """Convenience method for conditional requests.

    :param environ: the WSGI environment of the request to be checked.
    :param etag: the etag for the response for comparison.
    :param data: or alternatively the data of the response to automatically
                 generate an etag using :func:`generate_etag`.
    :param last_modified: an optional date of the last modification.
    :param ignore_if_range: If `False`, `If-Range` header will be taken into
                            account.
    :return: `True` if the resource was modified, otherwise `False`.

    .. versionchanged:: 2.0
        SHA-1 is used to generate an etag value for the data. MD5 may
        not be available in some environments.

    .. versionchanged:: 1.0.0
        The check is run for methods other than ``GET`` and ``HEAD``.
    """
    return _sansio_http.is_resource_modified(
        http_range=environ.get("HTTP_RANGE"),
        http_if_range=environ.get("HTTP_IF_RANGE"),
        http_if_modified_since=environ.get("HTTP_IF_MODIFIED_SINCE"),
        http_if_none_match=environ.get("HTTP_IF_NONE_MATCH"),
        http_if_match=environ.get("HTTP_IF_MATCH"),
        etag=etag,
        data=data,
        last_modified=last_modified,
        ignore_if_range=ignore_if_range,
    )


def remove_entity_headers(
    headers: ds.Headers | list[tuple[str, str]],
    allowed: t.Iterable[str] = ("expires", "content-location"),
) -> None:
    """Remove all entity headers from a list or :class:`Headers` object.  This
    operation works in-place.  `Expires` and `Content-Location` headers are
    by default not removed.  The reason for this is :rfc:`2616` section
    10.3.5 which specifies some entity headers that should be sent.

    .. versionchanged:: 0.5
       added `allowed` parameter.

    :param headers: a list or :class:`Headers` object.
    :param allowed: a list of headers that should still be allowed even though
                    they are entity headers.
    """
    allowed = {x.lower() for x in allowed}
    headers[:] = [
        (key, value)
        for key, value in headers
        if not is_entity_header(key) or key.lower() in allowed
    ]


def remove_hop_by_hop_headers(headers: ds.Headers | list[tuple[str, str]]) -> None:
    """Remove all HTTP/1.1 "Hop-by-Hop" headers from a list or
    :class:`Headers` object.  This operation works in-place.

    .. versionadded:: 0.5

    :param headers: a list or :class:`Headers` object.
    """
    headers[:] = [
        (key, value) for key, value in headers if not is_hop_by_hop_header(key)
    ]


def is_entity_header(header: str) -> bool:
    """Check if a header is an entity header.

    .. versionadded:: 0.5

    :param header: the header to test.
    :return: `True` if it's an entity header, `False` otherwise.
    """
    return header.lower() in _entity_headers


def is_hop_by_hop_header(header: str) -> bool:
    """Check if a header is an HTTP/1.1 "Hop-by-Hop" header.

    .. versionadded:: 0.5

    :param header: the header to test.
    :return: `True` if it's an HTTP/1.1 "Hop-by-Hop" header, `False` otherwise.
    """
    return header.lower() in _hop_by_hop_headers


def parse_cookie(
    header: WSGIEnvironment | str | None,
    cls: type[ds.MultiDict[str, str]] | None = None,
) -> ds.MultiDict[str, str]:
    """Parse a cookie from a string or WSGI environ.

    The same key can be provided multiple times, the values are stored
    in-order. The default :class:`MultiDict` will have the first value
    first, and all values can be retrieved with
    :meth:`MultiDict.getlist`.

    :param header: The cookie header as a string, or a WSGI environ dict
        with a ``HTTP_COOKIE`` key.
    :param cls: A dict-like class to store the parsed cookies in.
        Defaults to :class:`MultiDict`.

    .. versionchanged:: 3.0
        Passing bytes, and the ``charset`` and ``errors`` parameters, were removed.

    .. versionchanged:: 1.0
        Returns a :class:`MultiDict` instead of a ``TypeConversionDict``.

    .. versionchanged:: 0.5
        Returns a :class:`TypeConversionDict` instead of a regular dict. The ``cls``
        parameter was added.
    """
    if isinstance(header, dict):
        cookie = header.get("HTTP_COOKIE")
    else:
        cookie = header

    if cookie:
        cookie = cookie.encode("latin1").decode()

    return _sansio_http.parse_cookie(cookie=cookie, cls=cls)


_cookie_no_quote_re = re.compile(r"[\w!#$%&'()*+\-./:<=>?@\[\]^`{|}~]*", re.A)
_cookie_slash_re = re.compile(rb"[\x00-\x19\",;\\\x7f-\xff]", re.A)
_cookie_slash_map = {b'"': b'\\"', b"\\": b"\\\\"}
_cookie_slash_map.update(
    (v.to_bytes(1, "big"), b"\\%03o" % v)
    for v in [*range(0x20), *b",;", *range(0x7F, 256)]
)


def dump_cookie(
    key: str,
    value: str = "",
    max_age: timedelta | int | None = None,
    expires: str | datetime | int | float | None = None,
    path: str | None = "/",
    domain: str | None = None,
    secure: bool = False,
    httponly: bool = False,
    sync_expires: bool = True,
    max_size: int = 4093,
    samesite: str | None = None,
    partitioned: bool = False,
) -> str:
    """Create a Set-Cookie header without the ``Set-Cookie`` prefix.

    The return value is usually restricted to ascii as the vast majority
    of values are properly escaped, but that is no guarantee. It's
    tunneled through latin1 as required by :pep:`3333`.

    The return value is not ASCII safe if the key contains unicode
    characters.  This is technically against the specification but
    happens in the wild.  It's strongly recommended to not use
    non-ASCII values for the keys.

    :param max_age: should be a number of seconds, or `None` (default) if
                    the cookie should last only as long as the client's
                    browser session.  Additionally `timedelta` objects
                    are accepted, too.
    :param expires: should be a `datetime` object or unix timestamp.
    :param path: limits the cookie to a given path, per default it will
                 span the whole domain.
    :param domain: Use this if you want to set a cross-domain cookie. For
                   example, ``domain="example.com"`` will set a cookie
                   that is readable by the domain ``www.example.com``,
                   ``foo.example.com`` etc. Otherwise, a cookie will only
                   be readable by the domain that set it.
    :param secure: The cookie will only be available via HTTPS
    :param httponly: disallow JavaScript to access the cookie.  This is an
                     extension to the cookie standard and probably not
                     supported by all browsers.
    :param charset: the encoding for string values.
    :param sync_expires: automatically set expires if max_age is defined
                         but expires not.
    :param max_size: Warn if the final header value exceeds this size. The
        default, 4093, should be safely `supported by most browsers
        <cookie_>`_. Set to 0 to disable this check.
    :param samesite: Limits the scope of the cookie such that it will
        only be attached to requests if those requests are same-site.
    :param partitioned: Opts the cookie into partitioned storage. This
        will also set secure to True

    .. _`cookie`: http://browsercookielimits.squawky.net/

    .. versionchanged:: 3.1
        The ``partitioned`` parameter was added.

    .. versionchanged:: 3.0
        Passing bytes, and the ``charset`` parameter, were removed.

    .. versionchanged:: 2.3.3
        The ``path`` parameter is ``/`` by default.

    .. versionchanged:: 2.3.1
        The value allows more characters without quoting.

    .. versionchanged:: 2.3
        ``localhost`` and other names without a dot are allowed for the domain. A
        leading dot is ignored.

    .. versionchanged:: 2.3
        The ``path`` parameter is ``None`` by default.

    .. versionchanged:: 1.0.0
        The string ``'None'`` is accepted for ``samesite``.
    """
    if path is not None:
        # safe = https://url.spec.whatwg.org/#url-path-segment-string
        # as well as percent for things that are already quoted
        # excluding semicolon since it's part of the header syntax
        path = quote(path, safe="%!$&'()*+,/:=@")

    if domain:
        domain = domain.partition(":")[0].lstrip(".").encode("idna").decode("ascii")

    if isinstance(max_age, timedelta):
        max_age = int(max_age.total_seconds())

    if expires is not None:
        if not isinstance(expires, str):
            expires = http_date(expires)
    elif max_age is not None and sync_expires:
        expires = http_date(datetime.now(tz=timezone.utc).timestamp() + max_age)

    if samesite is not None:
        samesite = samesite.title()

        if samesite not in {"Strict", "Lax", "None"}:
            raise ValueError("SameSite must be 'Strict', 'Lax', or 'None'.")

    if partitioned:
        secure = True

    # Quote value if it contains characters not allowed by RFC 6265. Slash-escape with
    # three octal digits, which matches http.cookies, although the RFC suggests base64.
    if not _cookie_no_quote_re.fullmatch(value):
        # Work with bytes here, since a UTF-8 character could be multiple bytes.
        value = _cookie_slash_re.sub(
            lambda m: _cookie_slash_map[m.group()], value.encode()
        ).decode("ascii")
        value = f'"{value}"'

    # Send a non-ASCII key as mojibake. Everything else should already be ASCII.
    # TODO Remove encoding dance, it seems like clients accept UTF-8 keys
    buf = [f"{key.encode().decode('latin1')}={value}"]

    for k, v in (
        ("Domain", domain),
        ("Expires", expires),
        ("Max-Age", max_age),
        ("Secure", secure),
        ("HttpOnly", httponly),
        ("Path", path),
        ("SameSite", samesite),
        ("Partitioned", partitioned),
    ):
        if v is None or v is False:
            continue

        if v is True:
            buf.append(k)
            continue

        buf.append(f"{k}={v}")

    rv = "; ".join(buf)

    # Warn if the final value of the cookie is larger than the limit. If the cookie is
    # too large, then it may be silently ignored by the browser, which can be quite hard
    # to debug.
    cookie_size = len(rv)

    if max_size and cookie_size > max_size:
        value_size = len(value)
        warnings.warn(
            f"The '{key}' cookie is too large: the value was {value_size} bytes but the"
            f" header required {cookie_size - value_size} extra bytes. The final size"
            f" was {cookie_size} bytes but the limit is {max_size} bytes. Browsers may"
            " silently ignore cookies larger than this.",
            stacklevel=2,
        )

    return rv


def is_byte_range_valid(
    start: int | None, stop: int | None, length: int | None
) -> bool:
    """Checks if a given byte content range is valid for the given length.

    .. versionadded:: 0.7
    """
    if (start is None) != (stop is None):
        return False
    elif start is None:
        return length is None or length >= 0
    elif length is None:
        return 0 <= start < stop  # type: ignore
    elif start >= stop:  # type: ignore
        return False
    return 0 <= start < length


# circular dependencies
from . import datastructures as ds
from .sansio import http as _sansio_http