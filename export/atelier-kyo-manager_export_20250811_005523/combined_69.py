
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\legendre.py ===
"""
==================================================
Legendre Series (:mod:`numpy.polynomial.legendre`)
==================================================

This module provides a number of objects (mostly functions) useful for
dealing with Legendre series, including a `Legendre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

    Legendre

Constants
---------

.. autosummary::
   :toctree: generated/

   legdomain
   legzero
   legone
   legx

Arithmetic
----------

.. autosummary::
   :toctree: generated/

   legadd
   legsub
   legmulx
   legmul
   legdiv
   legpow
   legval
   legval2d
   legval3d
   leggrid2d
   leggrid3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   legder
   legint

Misc Functions
--------------

.. autosummary::
   :toctree: generated/

   legfromroots
   legroots
   legvander
   legvander2d
   legvander3d
   leggauss
   legweight
   legcompanion
   legfit
   legtrim
   legline
   leg2poly
   poly2leg

See also
--------
numpy.polynomial

"""
import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

__all__ = [
    'legzero', 'legone', 'legx', 'legdomain', 'legline', 'legadd',
    'legsub', 'legmulx', 'legmul', 'legdiv', 'legpow', 'legval', 'legder',
    'legint', 'leg2poly', 'poly2leg', 'legfromroots', 'legvander',
    'legfit', 'legtrim', 'legroots', 'Legendre', 'legval2d', 'legval3d',
    'leggrid2d', 'leggrid3d', 'legvander2d', 'legvander3d', 'legcompanion',
    'leggauss', 'legweight']

legtrim = pu.trimcoef


def poly2leg(pol):
    """
    Convert a polynomial to a Legendre series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Legendre series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Legendre
        series.

    See Also
    --------
    leg2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy import polynomial as P
    >>> p = P.Polynomial(np.arange(4))
    >>> p
    Polynomial([0.,  1.,  2.,  3.], domain=[-1.,  1.], window=[-1.,  1.], ...
    >>> c = P.Legendre(P.legendre.poly2leg(p.coef))
    >>> c
    Legendre([ 1.  ,  3.25,  1.  ,  0.75], domain=[-1,  1], window=[-1,  1]) # may vary

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = legadd(legmulx(res), pol[i])
    return res


def leg2poly(c):
    """
    Convert a Legendre series to a polynomial.

    Convert an array representing the coefficients of a Legendre series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Legendre series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2leg

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> c = P.Legendre(range(4))
    >>> c
    Legendre([0., 1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.], symbol='x')
    >>> p = c.convert(kind=P.Polynomial)
    >>> p
    Polynomial([-1. , -3.5,  3. ,  7.5], domain=[-1.,  1.], window=[-1., ...
    >>> P.legendre.leg2poly(range(4))
    array([-1. , -3.5,  3. ,  7.5])


    """
    from .polynomial import polyadd, polymulx, polysub

    [c] = pu.as_series([c])
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], (c1 * (i - 1)) / i)
            c1 = polyadd(tmp, (polymulx(c1) * (2 * i - 1)) / i)
        return polyadd(c0, polymulx(c1))


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Legendre
legdomain = np.array([-1., 1.])

# Legendre coefficients representing zero.
legzero = np.array([0])

# Legendre coefficients representing one.
legone = np.array([1])

# Legendre coefficients representing the identity x.
legx = np.array([0, 1])


def legline(off, scl):
    """
    Legendre series whose graph is a straight line.



    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Legendre series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> import numpy.polynomial.legendre as L
    >>> L.legline(3,2)
    array([3, 2])
    >>> L.legval(-3, L.legline(3,2)) # should be -3
    -3.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])


def legfromroots(roots):
    """
    Generate a Legendre series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in Legendre form, where the :math:`r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in Legendre form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of coefficients.  If all roots are real then `out` is a
        real array, if some of the roots are complex, then `out` is complex
        even if all the coefficients in the result are real (see Examples
        below).

    See Also
    --------
    numpy.polynomial.polynomial.polyfromroots
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Examples
    --------
    >>> import numpy.polynomial.legendre as L
    >>> L.legfromroots((-1,0,1)) # x^3 - x relative to the standard basis
    array([ 0. , -0.4,  0. ,  0.4])
    >>> j = complex(0,1)
    >>> L.legfromroots((-j,j)) # x^2 + 1 relative to the standard basis
    array([ 1.33333333+0.j,  0.00000000+0.j,  0.66666667+0.j]) # may vary

    """
    return pu._fromroots(legline, legmul, roots)


def legadd(c1, c2):
    """
    Add one Legendre series to another.

    Returns the sum of two Legendre series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Legendre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Legendre series of their sum.

    See Also
    --------
    legsub, legmulx, legmul, legdiv, legpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Legendre series
    is a Legendre series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> L.legadd(c1,c2)
    array([4.,  4.,  4.])

    """
    return pu._add(c1, c2)


def legsub(c1, c2):
    """
    Subtract one Legendre series from another.

    Returns the difference of two Legendre series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Legendre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Legendre series coefficients representing their difference.

    See Also
    --------
    legadd, legmulx, legmul, legdiv, legpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Legendre
    series is a Legendre series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> L.legsub(c1,c2)
    array([-2.,  0.,  2.])
    >>> L.legsub(c2,c1) # -C.legsub(c1,c2)
    array([ 2.,  0., -2.])

    """
    return pu._sub(c1, c2)


def legmulx(c):
    """Multiply a Legendre series by x.

    Multiply the Legendre series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Legendre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    legadd, legsub, legmul, legdiv, legpow

    Notes
    -----
    The multiplication uses the recursion relationship for Legendre
    polynomials in the form

    .. math::

      xP_i(x) = ((i + 1)*P_{i + 1}(x) + i*P_{i - 1}(x))/(2i + 1)

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> L.legmulx([1,2,3])
    array([ 0.66666667, 2.2, 1.33333333, 1.8]) # may vary

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    for i in range(1, len(c)):
        j = i + 1
        k = i - 1
        s = i + j
        prd[j] = (c[i] * j) / s
        prd[k] += (c[i] * i) / s
    return prd


def legmul(c1, c2):
    """
    Multiply one Legendre series by another.

    Returns the product of two Legendre series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Legendre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Legendre series coefficients representing their product.

    See Also
    --------
    legadd, legsub, legmulx, legdiv, legpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Legendre polynomial basis set.  Thus, to express
    the product as a Legendre series, it is necessary to "reproject" the
    product onto said basis set, which may produce "unintuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> c1 = (1,2,3)
    >>> c2 = (3,2)
    >>> L.legmul(c1,c2) # multiplication requires "reprojection"
    array([  4.33333333,  10.4       ,  11.66666667,   3.6       ]) # may vary

    """
    # s1, s2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0] * xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0] * xs
        c1 = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        c1 = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = legsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = legadd(tmp, (legmulx(c1) * (2 * nd - 1)) / nd)
    return legadd(c0, legmulx(c1))


def legdiv(c1, c2):
    """
    Divide one Legendre series by another.

    Returns the quotient-with-remainder of two Legendre series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Legendre series coefficients ordered from low to
        high.

    Returns
    -------
    quo, rem : ndarrays
        Of Legendre series coefficients representing the quotient and
        remainder.

    See Also
    --------
    legadd, legsub, legmulx, legmul, legpow

    Notes
    -----
    In general, the (polynomial) division of one Legendre series by another
    results in quotient and remainder terms that are not in the Legendre
    polynomial basis set.  Thus, to express these results as a Legendre
    series, it is necessary to "reproject" the results onto the Legendre
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> L.legdiv(c1,c2) # quotient "intuitive," remainder not
    (array([3.]), array([-8., -4.]))
    >>> c2 = (0,1,2,3)
    >>> L.legdiv(c2,c1) # neither "intuitive"
    (array([-0.07407407,  1.66666667]), array([-1.03703704, -2.51851852])) # may vary

    """
    return pu._div(legmul, c1, c2)


def legpow(c, pow, maxpower=16):
    """Raise a Legendre series to a power.

    Returns the Legendre series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Legendre series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Legendre series of power.

    See Also
    --------
    legadd, legsub, legmulx, legmul, legdiv

    """
    return pu._pow(legmul, c, pow, maxpower)


def legder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Legendre series.

    Returns the Legendre series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``
    while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +
    2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Legendre series coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    der : ndarray
        Legendre series of the derivative.

    See Also
    --------
    legint

    Notes
    -----
    In general, the result of differentiating a Legendre series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> c = (1,2,3,4)
    >>> L.legder(c)
    array([  6.,   9.,  20.])
    >>> L.legder(c, 3)
    array([60.])
    >>> L.legder(c, scl=-1)
    array([ -6.,  -9., -20.])
    >>> L.legder(c, 2,-1)
    array([  9.,  60.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    cnt = pu._as_int(m, "the order of derivation")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 2, -1):
                der[j - 1] = (2 * j - 1) * c[j]
                c[j - 2] += c[j]
            if n > 1:
                der[1] = 3 * c[2]
            der[0] = c[1]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def legint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Legendre series.

    Returns the Legendre series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``L_0 + 2*L_1 + 3*L_2`` while [[1,2],[1,2]]
    represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) + 2*L_0(x)*L_1(y) +
    2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Legendre series coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at
        ``lbnd`` is the first value in the list, the value of the second
        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
        default), all constants are set to zero.  If ``m == 1``, a single
        scalar can be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

    Returns
    -------
    S : ndarray
        Legendre series coefficient array of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    legder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "reprojected" onto the C-series basis set.  Thus, typically,
    the result of this function is "unintuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> c = (1,2,3)
    >>> L.legint(c)
    array([ 0.33333333,  0.4       ,  0.66666667,  0.6       ]) # may vary
    >>> L.legint(c, 3)
    array([  1.66666667e-02,  -1.78571429e-02,   4.76190476e-02, # may vary
             -1.73472348e-18,   1.90476190e-02,   9.52380952e-03])
    >>> L.legint(c, k=3)
     array([ 3.33333333,  0.4       ,  0.66666667,  0.6       ]) # may vary
    >>> L.legint(c, lbnd=-2)
    array([ 7.33333333,  0.4       ,  0.66666667,  0.6       ]) # may vary
    >>> L.legint(c, scl=2)
    array([ 0.66666667,  0.8       ,  1.33333333,  1.2       ]) # may vary

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if not np.iterable(k):
        k = [k]
    cnt = pu._as_int(m, "the order of integration")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    k = list(k) + [0] * (cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            if n > 1:
                tmp[2] = c[1] / 3
            for j in range(2, n):
                t = c[j] / (2 * j + 1)
                tmp[j + 1] = t
                tmp[j - 1] -= t
            tmp[0] += k[i] - legval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def legval(x, c, tensor=True):
    """
    Evaluate a Legendre series at points x.

    If `c` is of length ``n + 1``, this function returns the value:

    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.

    See Also
    --------
    legval2d, leggrid2d, legval3d, leggrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    """
    c = np.array(c, ndmin=1, copy=None)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * ((nd - 1) / nd)
            c1 = tmp + c1 * x * ((2 * nd - 1) / nd)
    return c0 + c1 * x


def legval2d(x, y, c):
    """
    Evaluate a 2-D Legendre series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * L_i(x) * L_j(y)

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` is a 1-D array a one is implicitly appended to its shape to make
    it 2-D. The shape of the result will be c.shape[2:] + x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points ``(x, y)``,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and if it isn't an ndarray it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional Legendre series at points formed
        from pairs of corresponding values from `x` and `y`.

    See Also
    --------
    legval, leggrid2d, legval3d, leggrid3d
    """
    return pu._valnd(legval, c, x, y)


def leggrid2d(x, y, c):
    """
    Evaluate a 2-D Legendre series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * L_i(a) * L_j(b)

    where the points ``(a, b)`` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape + y.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j is contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional Chebyshev series at points in the
        Cartesian product of `x` and `y`.

    See Also
    --------
    legval, legval2d, legval3d, leggrid3d
    """
    return pu._gridnd(legval, c, x, y)


def legval3d(x, y, z, c):
    """
    Evaluate a 3-D Legendre series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * L_i(x) * L_j(y) * L_k(z)

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        ``(x, y, z)``, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    legval, legval2d, leggrid2d, leggrid3d
    """
    return pu._valnd(legval, c, x, y, z)


def leggrid3d(x, y, z, c):
    """
    Evaluate a 3-D Legendre series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * L_i(a) * L_j(b) * L_k(c)

    where the points ``(a, b, c)`` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`, `y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    legval, legval2d, leggrid2d, legval3d
    """
    return pu._gridnd(legval, c, x, y, z)


def legvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = L_i(x)

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Legendre polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    array ``V = legvander(x, n)``, then ``np.dot(V, c)`` and
    ``legval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Legendre series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo-Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Legendre polynomial.  The dtype will be the same as
        the converted `x`.

    """
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=None, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    # Use forward recursion to generate the entries. This is not as accurate
    # as reverse recursion in this application but it is more efficient.
    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = (v[i - 1] * x * (2 * i - 1) - v[i - 2] * (i - 1)) / i
    return np.moveaxis(v, 0, -1)


def legvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = L_i(x) * L_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the Legendre polynomials.

    If ``V = legvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``legval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D Legendre
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    legvander, legvander3d, legval2d, legval3d
    """
    return pu._vander_nd_flat((legvander, legvander), (x, y), deg)


def legvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = L_i(x)*L_j(y)*L_k(z),

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the degrees of the Legendre polynomials.

    If ``V = legvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and ``np.dot(V, c.flat)`` and ``legval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Legendre
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    legvander, legvander3d, legval2d, legval3d
    """
    return pu._vander_nd_flat((legvander, legvander, legvander), (x, y, z), deg)


def legfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Legendre series to data.

    Return the coefficients of a Legendre series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Legendre coefficients ordered from low to high. If `y` was
        2-D, the coefficients for the data in column k of `y` are in
        column `k`. If `deg` is specified as a list, coefficients for
        terms not included in the fit are set equal to zero in the
        returned `coef`.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if ``full == False``.  The
        warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    legval : Evaluates a Legendre series.
    legvander : Vandermonde matrix of Legendre series.
    legweight : Legendre weight function (= 1).
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Legendre series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where :math:`w_j` are the weights. This problem is solved by setting up
    as the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `~exceptions.RankWarning` will be issued. This means that
    the coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Legendre series are usually better conditioned than fits
    using power series, but much can depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------

    """
    return pu._fit(legvander, x, y, deg, rcond, full, w)


def legcompanion(c):
    """Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is an Legendre basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of Legendre series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).
    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    scl = 1. / np.sqrt(2 * np.arange(n) + 1)
    top = mat.reshape(-1)[1::n + 1]
    bot = mat.reshape(-1)[n::n + 1]
    top[...] = np.arange(1, n) * scl[:n - 1] * scl[1:n]
    bot[...] = top
    mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2 * n - 1))
    return mat


def legroots(c):
    """
    Compute the roots of a Legendre series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * L_i(x).

    Parameters
    ----------
    c : 1-D array_like
        1-D array of coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the series. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    numpy.polynomial.polynomial.polyroots
    numpy.polynomial.chebyshev.chebroots
    numpy.polynomial.laguerre.lagroots
    numpy.polynomial.hermite.hermroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such values.
    Roots with multiplicity greater than 1 will also show larger errors as
    the value of the series near such points is relatively insensitive to
    errors in the roots. Isolated roots near the origin can be improved by
    a few iterations of Newton's method.

    The Legendre series basis polynomials aren't powers of ``x`` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> import numpy.polynomial.legendre as leg
    >>> leg.legroots((1, 2, 3, 4)) # 4L_3 + 3L_2 + 2L_1 + 1L_0, all real roots
    array([-0.85099543, -0.11407192,  0.51506735]) # may vary

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([-c[0] / c[1]])

    # rotated companion matrix reduces error
    m = legcompanion(c)[::-1, ::-1]
    r = la.eigvals(m)
    r.sort()
    return r


def leggauss(deg):
    """
    Gauss-Legendre quadrature.

    Computes the sample points and weights for Gauss-Legendre quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-1, 1]` with
    the weight function :math:`f(x) = 1`.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.

    Notes
    -----
    The results have only been tested up to degree 100, higher degrees may
    be problematic. The weights are determined by using the fact that

    .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`L_n`, and then scaling the results to get
    the right value when integrating 1.

    """
    ideg = pu._as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    # first approximation of roots. We use the fact that the companion
    # matrix is symmetric in this case in order to obtain better zeros.
    c = np.array([0] * deg + [1])
    m = legcompanion(c)
    x = la.eigvalsh(m)

    # improve roots by one application of Newton
    dy = legval(x, c)
    df = legval(x, legder(c))
    x -= dy / df

    # compute the weights. We scale the factor to avoid possible numerical
    # overflow.
    fm = legval(x, c[1:])
    fm /= np.abs(fm).max()
    df /= np.abs(df).max()
    w = 1 / (fm * df)

    # for Legendre we can also symmetrize
    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    # scale w to get the right value
    w *= 2. / w.sum()

    return x, w


def legweight(x):
    """
    Weight function of the Legendre polynomials.

    The weight function is :math:`1` and the interval of integration is
    :math:`[-1, 1]`. The Legendre polynomials are orthogonal, but not
    normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.
    """
    w = x * 0.0 + 1.0
    return w

#
# Legendre series class
#

class Legendre(ABCPolyBase):
    """A Legendre series class.

    The Legendre class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Legendre coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1., 1.].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1., 1.].
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    _add = staticmethod(legadd)
    _sub = staticmethod(legsub)
    _mul = staticmethod(legmul)
    _div = staticmethod(legdiv)
    _pow = staticmethod(legpow)
    _val = staticmethod(legval)
    _int = staticmethod(legint)
    _der = staticmethod(legder)
    _fit = staticmethod(legfit)
    _line = staticmethod(legline)
    _roots = staticmethod(legroots)
    _fromroots = staticmethod(legfromroots)

    # Virtual properties
    domain = np.array(legdomain)
    window = np.array(legdomain)
    basis_name = 'P'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\cffi\recompiler.py ===
import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *

VERSION_BASE = 0x2601
VERSION_EMBEDDED = 0x2701
VERSION_CHAR16CHAR32 = 0x2801

USE_LIMITED_API = (sys.platform != 'win32' or sys.version_info < (3, 0) or
                   sys.version_info >= (3, 5))


class GlobalExpr:
    def __init__(self, name, address, type_op, size=0, check_value=0):
        self.name = name
        self.address = address
        self.type_op = type_op
        self.size = size
        self.check_value = check_value

    def as_c_expr(self):
        return '  { "%s", (void *)%s, %s, (void *)%s },' % (
            self.name, self.address, self.type_op.as_c_expr(), self.size)

    def as_python_expr(self):
        return "b'%s%s',%d" % (self.type_op.as_python_bytes(), self.name,
                               self.check_value)

class FieldExpr:
    def __init__(self, name, field_offset, field_size, fbitsize, field_type_op):
        self.name = name
        self.field_offset = field_offset
        self.field_size = field_size
        self.fbitsize = fbitsize
        self.field_type_op = field_type_op

    def as_c_expr(self):
        spaces = " " * len(self.name)
        return ('  { "%s", %s,\n' % (self.name, self.field_offset) +
                '     %s   %s,\n' % (spaces, self.field_size) +
                '     %s   %s },' % (spaces, self.field_type_op.as_c_expr()))

    def as_python_expr(self):
        raise NotImplementedError

    def as_field_python_expr(self):
        if self.field_type_op.op == OP_NOOP:
            size_expr = ''
        elif self.field_type_op.op == OP_BITFIELD:
            size_expr = format_four_bytes(self.fbitsize)
        else:
            raise NotImplementedError
        return "b'%s%s%s'" % (self.field_type_op.as_python_bytes(),
                              size_expr,
                              self.name)

class StructUnionExpr:
    def __init__(self, name, type_index, flags, size, alignment, comment,
                 first_field_index, c_fields):
        self.name = name
        self.type_index = type_index
        self.flags = flags
        self.size = size
        self.alignment = alignment
        self.comment = comment
        self.first_field_index = first_field_index
        self.c_fields = c_fields

    def as_c_expr(self):
        return ('  { "%s", %d, %s,' % (self.name, self.type_index, self.flags)
                + '\n    %s, %s, ' % (self.size, self.alignment)
                + '%d, %d ' % (self.first_field_index, len(self.c_fields))
                + ('/* %s */ ' % self.comment if self.comment else '')
                + '},')

    def as_python_expr(self):
        flags = eval(self.flags, G_FLAGS)
        fields_expr = [c_field.as_field_python_expr()
                       for c_field in self.c_fields]
        return "(b'%s%s%s',%s)" % (
            format_four_bytes(self.type_index),
            format_four_bytes(flags),
            self.name,
            ','.join(fields_expr))

class EnumExpr:
    def __init__(self, name, type_index, size, signed, allenums):
        self.name = name
        self.type_index = type_index
        self.size = size
        self.signed = signed
        self.allenums = allenums

    def as_c_expr(self):
        return ('  { "%s", %d, _cffi_prim_int(%s, %s),\n'
                '    "%s" },' % (self.name, self.type_index,
                                 self.size, self.signed, self.allenums))

    def as_python_expr(self):
        prim_index = {
            (1, 0): PRIM_UINT8,  (1, 1):  PRIM_INT8,
            (2, 0): PRIM_UINT16, (2, 1):  PRIM_INT16,
            (4, 0): PRIM_UINT32, (4, 1):  PRIM_INT32,
            (8, 0): PRIM_UINT64, (8, 1):  PRIM_INT64,
            }[self.size, self.signed]
        return "b'%s%s%s\\x00%s'" % (format_four_bytes(self.type_index),
                                     format_four_bytes(prim_index),
                                     self.name, self.allenums)

class TypenameExpr:
    def __init__(self, name, type_index):
        self.name = name
        self.type_index = type_index

    def as_c_expr(self):
        return '  { "%s", %d },' % (self.name, self.type_index)

    def as_python_expr(self):
        return "b'%s%s'" % (format_four_bytes(self.type_index), self.name)


# ____________________________________________________________


class Recompiler:
    _num_externpy = 0

    def __init__(self, ffi, module_name, target_is_python=False):
        self.ffi = ffi
        self.module_name = module_name
        self.target_is_python = target_is_python
        self._version = VERSION_BASE

    def needs_version(self, ver):
        self._version = max(self._version, ver)

    def collect_type_table(self):
        self._typesdict = {}
        self._generate("collecttype")
        #
        all_decls = sorted(self._typesdict, key=str)
        #
        # prepare all FUNCTION bytecode sequences first
        self.cffi_types = []
        for tp in all_decls:
            if tp.is_raw_function:
                assert self._typesdict[tp] is None
                self._typesdict[tp] = len(self.cffi_types)
                self.cffi_types.append(tp)     # placeholder
                for tp1 in tp.args:
                    assert isinstance(tp1, (model.VoidType,
                                            model.BasePrimitiveType,
                                            model.PointerType,
                                            model.StructOrUnionOrEnum,
                                            model.FunctionPtrType))
                    if self._typesdict[tp1] is None:
                        self._typesdict[tp1] = len(self.cffi_types)
                    self.cffi_types.append(tp1)   # placeholder
                self.cffi_types.append('END')     # placeholder
        #
        # prepare all OTHER bytecode sequences
        for tp in all_decls:
            if not tp.is_raw_function and self._typesdict[tp] is None:
                self._typesdict[tp] = len(self.cffi_types)
                self.cffi_types.append(tp)        # placeholder
                if tp.is_array_type and tp.length is not None:
                    self.cffi_types.append('LEN') # placeholder
        assert None not in self._typesdict.values()
        #
        # collect all structs and unions and enums
        self._struct_unions = {}
        self._enums = {}
        for tp in all_decls:
            if isinstance(tp, model.StructOrUnion):
                self._struct_unions[tp] = None
            elif isinstance(tp, model.EnumType):
                self._enums[tp] = None
        for i, tp in enumerate(sorted(self._struct_unions,
                                      key=lambda tp: tp.name)):
            self._struct_unions[tp] = i
        for i, tp in enumerate(sorted(self._enums,
                                      key=lambda tp: tp.name)):
            self._enums[tp] = i
        #
        # emit all bytecode sequences now
        for tp in all_decls:
            method = getattr(self, '_emit_bytecode_' + tp.__class__.__name__)
            method(tp, self._typesdict[tp])
        #
        # consistency check
        for op in self.cffi_types:
            assert isinstance(op, CffiOp)
        self.cffi_types = tuple(self.cffi_types)    # don't change any more

    def _enum_fields(self, tp):
        # When producing C, expand all anonymous struct/union fields.
        # That's necessary to have C code checking the offsets of the
        # individual fields contained in them.  When producing Python,
        # don't do it and instead write it like it is, with the
        # corresponding fields having an empty name.  Empty names are
        # recognized at runtime when we import the generated Python
        # file.
        expand_anonymous_struct_union = not self.target_is_python
        return tp.enumfields(expand_anonymous_struct_union)

    def _do_collect_type(self, tp):
        if not isinstance(tp, model.BaseTypeByIdentity):
            if isinstance(tp, tuple):
                for x in tp:
                    self._do_collect_type(x)
            return
        if tp not in self._typesdict:
            self._typesdict[tp] = None
            if isinstance(tp, model.FunctionPtrType):
                self._do_collect_type(tp.as_raw_function())
            elif isinstance(tp, model.StructOrUnion):
                if tp.fldtypes is not None and (
                        tp not in self.ffi._parser._included_declarations):
                    for name1, tp1, _, _ in self._enum_fields(tp):
                        self._do_collect_type(self._field_type(tp, name1, tp1))
            else:
                for _, x in tp._get_items():
                    self._do_collect_type(x)

    def _generate(self, step_name):
        lst = self.ffi._parser._declarations.items()
        for name, (tp, quals) in sorted(lst):
            kind, realname = name.split(' ', 1)
            try:
                method = getattr(self, '_generate_cpy_%s_%s' % (kind,
                                                                step_name))
            except AttributeError:
                raise VerificationError(
                    "not implemented in recompile(): %r" % name)
            try:
                self._current_quals = quals
                method(tp, realname)
            except Exception as e:
                model.attach_exception_info(e, name)
                raise

    # ----------

    ALL_STEPS = ["global", "field", "struct_union", "enum", "typename"]

    def collect_step_tables(self):
        # collect the declarations for '_cffi_globals', '_cffi_typenames', etc.
        self._lsts = {}
        for step_name in self.ALL_STEPS:
            self._lsts[step_name] = []
        self._seen_struct_unions = set()
        self._generate("ctx")
        self._add_missing_struct_unions()
        #
        for step_name in self.ALL_STEPS:
            lst = self._lsts[step_name]
            if step_name != "field":
                lst.sort(key=lambda entry: entry.name)
            self._lsts[step_name] = tuple(lst)    # don't change any more
        #
        # check for a possible internal inconsistency: _cffi_struct_unions
        # should have been generated with exactly self._struct_unions
        lst = self._lsts["struct_union"]
        for tp, i in self._struct_unions.items():
            assert i < len(lst)
            assert lst[i].name == tp.name
        assert len(lst) == len(self._struct_unions)
        # same with enums
        lst = self._lsts["enum"]
        for tp, i in self._enums.items():
            assert i < len(lst)
            assert lst[i].name == tp.name
        assert len(lst) == len(self._enums)

    # ----------

    def _prnt(self, what=''):
        self._f.write(what + '\n')

    def write_source_to_f(self, f, preamble):
        if self.target_is_python:
            assert preamble is None
            self.write_py_source_to_f(f)
        else:
            assert preamble is not None
            self.write_c_source_to_f(f, preamble)

    def _rel_readlines(self, filename):
        g = open(os.path.join(os.path.dirname(__file__), filename), 'r')
        lines = g.readlines()
        g.close()
        return lines

    def write_c_source_to_f(self, f, preamble):
        self._f = f
        prnt = self._prnt
        if self.ffi._embedding is not None:
            prnt('#define _CFFI_USE_EMBEDDING')
        if not USE_LIMITED_API:
            prnt('#define _CFFI_NO_LIMITED_API')
        #
        # first the '#include' (actually done by inlining the file's content)
        lines = self._rel_readlines('_cffi_include.h')
        i = lines.index('#include "parse_c_type.h"\n')
        lines[i:i+1] = self._rel_readlines('parse_c_type.h')
        prnt(''.join(lines))
        #
        # if we have ffi._embedding != None, we give it here as a macro
        # and include an extra file
        base_module_name = self.module_name.split('.')[-1]
        if self.ffi._embedding is not None:
            prnt('#define _CFFI_MODULE_NAME  "%s"' % (self.module_name,))
            prnt('static const char _CFFI_PYTHON_STARTUP_CODE[] = {')
            self._print_string_literal_in_array(self.ffi._embedding)
            prnt('0 };')
            prnt('#ifdef PYPY_VERSION')
            prnt('# define _CFFI_PYTHON_STARTUP_FUNC  _cffi_pypyinit_%s' % (
                base_module_name,))
            prnt('#elif PY_MAJOR_VERSION >= 3')
            prnt('# define _CFFI_PYTHON_STARTUP_FUNC  PyInit_%s' % (
                base_module_name,))
            prnt('#else')
            prnt('# define _CFFI_PYTHON_STARTUP_FUNC  init%s' % (
                base_module_name,))
            prnt('#endif')
            lines = self._rel_readlines('_embedding.h')
            i = lines.index('#include "_cffi_errors.h"\n')
            lines[i:i+1] = self._rel_readlines('_cffi_errors.h')
            prnt(''.join(lines))
            self.needs_version(VERSION_EMBEDDED)
        #
        # then paste the C source given by the user, verbatim.
        prnt('/************************************************************/')
        prnt()
        prnt(preamble)
        prnt()
        prnt('/************************************************************/')
        prnt()
        #
        # the declaration of '_cffi_types'
        prnt('static void *_cffi_types[] = {')
        typeindex2type = dict([(i, tp) for (tp, i) in self._typesdict.items()])
        for i, op in enumerate(self.cffi_types):
            comment = ''
            if i in typeindex2type:
                comment = ' // ' + typeindex2type[i]._get_c_name()
            prnt('/* %2d */ %s,%s' % (i, op.as_c_expr(), comment))
        if not self.cffi_types:
            prnt('  0')
        prnt('};')
        prnt()
        #
        # call generate_cpy_xxx_decl(), for every xxx found from
        # ffi._parser._declarations.  This generates all the functions.
        self._seen_constants = set()
        self._generate("decl")
        #
        # the declaration of '_cffi_globals' and '_cffi_typenames'
        nums = {}
        for step_name in self.ALL_STEPS:
            lst = self._lsts[step_name]
            nums[step_name] = len(lst)
            if nums[step_name] > 0:
                prnt('static const struct _cffi_%s_s _cffi_%ss[] = {' % (
                    step_name, step_name))
                for entry in lst:
                    prnt(entry.as_c_expr())
                prnt('};')
                prnt()
        #
        # the declaration of '_cffi_includes'
        if self.ffi._included_ffis:
            prnt('static const char * const _cffi_includes[] = {')
            for ffi_to_include in self.ffi._included_ffis:
                try:
                    included_module_name, included_source = (
                        ffi_to_include._assigned_source[:2])
                except AttributeError:
                    raise VerificationError(
                        "ffi object %r includes %r, but the latter has not "
                        "been prepared with set_source()" % (
                            self.ffi, ffi_to_include,))
                if included_source is None:
                    raise VerificationError(
                        "not implemented yet: ffi.include() of a Python-based "
                        "ffi inside a C-based ffi")
                prnt('  "%s",' % (included_module_name,))
            prnt('  NULL')
            prnt('};')
            prnt()
        #
        # the declaration of '_cffi_type_context'
        prnt('static const struct _cffi_type_context_s _cffi_type_context = {')
        prnt('  _cffi_types,')
        for step_name in self.ALL_STEPS:
            if nums[step_name] > 0:
                prnt('  _cffi_%ss,' % step_name)
            else:
                prnt('  NULL,  /* no %ss */' % step_name)
        for step_name in self.ALL_STEPS:
            if step_name != "field":
                prnt('  %d,  /* num_%ss */' % (nums[step_name], step_name))
        if self.ffi._included_ffis:
            prnt('  _cffi_includes,')
        else:
            prnt('  NULL,  /* no includes */')
        prnt('  %d,  /* num_types */' % (len(self.cffi_types),))
        flags = 0
        if self._num_externpy > 0 or self.ffi._embedding is not None:
            flags |= 1     # set to mean that we use extern "Python"
        prnt('  %d,  /* flags */' % flags)
        prnt('};')
        prnt()
        #
        # the init function
        prnt('#ifdef __GNUC__')
        prnt('#  pragma GCC visibility push(default)  /* for -fvisibility= */')
        prnt('#endif')
        prnt()
        prnt('#ifdef PYPY_VERSION')
        prnt('PyMODINIT_FUNC')
        prnt('_cffi_pypyinit_%s(const void *p[])' % (base_module_name,))
        prnt('{')
        if flags & 1:
            prnt('    if (((intptr_t)p[0]) >= 0x0A03) {')
            prnt('        _cffi_call_python_org = '
                 '(void(*)(struct _cffi_externpy_s *, char *))p[1];')
            prnt('    }')
        prnt('    p[0] = (const void *)0x%x;' % self._version)
        prnt('    p[1] = &_cffi_type_context;')
        prnt('#if PY_MAJOR_VERSION >= 3')
        prnt('    return NULL;')
        prnt('#endif')
        prnt('}')
        # on Windows, distutils insists on putting init_cffi_xyz in
        # 'export_symbols', so instead of fighting it, just give up and
        # give it one
        prnt('#  ifdef _MSC_VER')
        prnt('     PyMODINIT_FUNC')
        prnt('#  if PY_MAJOR_VERSION >= 3')
        prnt('     PyInit_%s(void) { return NULL; }' % (base_module_name,))
        prnt('#  else')
        prnt('     init%s(void) { }' % (base_module_name,))
        prnt('#  endif')
        prnt('#  endif')
        prnt('#elif PY_MAJOR_VERSION >= 3')
        prnt('PyMODINIT_FUNC')
        prnt('PyInit_%s(void)' % (base_module_name,))
        prnt('{')
        prnt('  return _cffi_init("%s", 0x%x, &_cffi_type_context);' % (
            self.module_name, self._version))
        prnt('}')
        prnt('#else')
        prnt('PyMODINIT_FUNC')
        prnt('init%s(void)' % (base_module_name,))
        prnt('{')
        prnt('  _cffi_init("%s", 0x%x, &_cffi_type_context);' % (
            self.module_name, self._version))
        prnt('}')
        prnt('#endif')
        prnt()
        prnt('#ifdef __GNUC__')
        prnt('#  pragma GCC visibility pop')
        prnt('#endif')
        self._version = None

    def _to_py(self, x):
        if isinstance(x, str):
            return "b'%s'" % (x,)
        if isinstance(x, (list, tuple)):
            rep = [self._to_py(item) for item in x]
            if len(rep) == 1:
                rep.append('')
            return "(%s)" % (','.join(rep),)
        return x.as_python_expr()  # Py2: unicode unexpected; Py3: bytes unexp.

    def write_py_source_to_f(self, f):
        self._f = f
        prnt = self._prnt
        #
        # header
        prnt("# auto-generated file")
        prnt("import _cffi_backend")
        #
        # the 'import' of the included ffis
        num_includes = len(self.ffi._included_ffis or ())
        for i in range(num_includes):
            ffi_to_include = self.ffi._included_ffis[i]
            try:
                included_module_name, included_source = (
                    ffi_to_include._assigned_source[:2])
            except AttributeError:
                raise VerificationError(
                    "ffi object %r includes %r, but the latter has not "
                    "been prepared with set_source()" % (
                        self.ffi, ffi_to_include,))
            if included_source is not None:
                raise VerificationError(
                    "not implemented yet: ffi.include() of a C-based "
                    "ffi inside a Python-based ffi")
            prnt('from %s import ffi as _ffi%d' % (included_module_name, i))
        prnt()
        prnt("ffi = _cffi_backend.FFI('%s'," % (self.module_name,))
        prnt("    _version = 0x%x," % (self._version,))
        self._version = None
        #
        # the '_types' keyword argument
        self.cffi_types = tuple(self.cffi_types)    # don't change any more
        types_lst = [op.as_python_bytes() for op in self.cffi_types]
        prnt('    _types = %s,' % (self._to_py(''.join(types_lst)),))
        typeindex2type = dict([(i, tp) for (tp, i) in self._typesdict.items()])
        #
        # the keyword arguments from ALL_STEPS
        for step_name in self.ALL_STEPS:
            lst = self._lsts[step_name]
            if len(lst) > 0 and step_name != "field":
                prnt('    _%ss = %s,' % (step_name, self._to_py(lst)))
        #
        # the '_includes' keyword argument
        if num_includes > 0:
            prnt('    _includes = (%s,),' % (
                ', '.join(['_ffi%d' % i for i in range(num_includes)]),))
        #
        # the footer
        prnt(')')

    # ----------

    def _gettypenum(self, type):
        # a KeyError here is a bug.  please report it! :-)
        return self._typesdict[type]

    def _convert_funcarg_to_c(self, tp, fromvar, tovar, errcode):
        extraarg = ''
        if isinstance(tp, model.BasePrimitiveType) and not tp.is_complex_type():
            if tp.is_integer_type() and tp.name != '_Bool':
                converter = '_cffi_to_c_int'
                extraarg = ', %s' % tp.name
            elif isinstance(tp, model.UnknownFloatType):
                # don't check with is_float_type(): it may be a 'long
                # double' here, and _cffi_to_c_double would loose precision
                converter = '(%s)_cffi_to_c_double' % (tp.get_c_name(''),)
            else:
                cname = tp.get_c_name('')
                converter = '(%s)_cffi_to_c_%s' % (cname,
                                                   tp.name.replace(' ', '_'))
                if cname in ('char16_t', 'char32_t'):
                    self.needs_version(VERSION_CHAR16CHAR32)
            errvalue = '-1'
        #
        elif isinstance(tp, model.PointerType):
            self._convert_funcarg_to_c_ptr_or_array(tp, fromvar,
                                                    tovar, errcode)
            return
        #
        elif (isinstance(tp, model.StructOrUnionOrEnum) or
              isinstance(tp, model.BasePrimitiveType)):
            # a struct (not a struct pointer) as a function argument;
            # or, a complex (the same code works)
            self._prnt('  if (_cffi_to_c((char *)&%s, _cffi_type(%d), %s) < 0)'
                      % (tovar, self._gettypenum(tp), fromvar))
            self._prnt('    %s;' % errcode)
            return
        #
        elif isinstance(tp, model.FunctionPtrType):
            converter = '(%s)_cffi_to_c_pointer' % tp.get_c_name('')
            extraarg = ', _cffi_type(%d)' % self._gettypenum(tp)
            errvalue = 'NULL'
        #
        else:
            raise NotImplementedError(tp)
        #
        self._prnt('  %s = %s(%s%s);' % (tovar, converter, fromvar, extraarg))
        self._prnt('  if (%s == (%s)%s && PyErr_Occurred())' % (
            tovar, tp.get_c_name(''), errvalue))
        self._prnt('    %s;' % errcode)

    def _extra_local_variables(self, tp, localvars, freelines):
        if isinstance(tp, model.PointerType):
            localvars.add('Py_ssize_t datasize')
            localvars.add('struct _cffi_freeme_s *large_args_free = NULL')
            freelines.add('if (large_args_free != NULL)'
                          ' _cffi_free_array_arguments(large_args_free);')

    def _convert_funcarg_to_c_ptr_or_array(self, tp, fromvar, tovar, errcode):
        self._prnt('  datasize = _cffi_prepare_pointer_call_argument(')
        self._prnt('      _cffi_type(%d), %s, (char **)&%s);' % (
            self._gettypenum(tp), fromvar, tovar))
        self._prnt('  if (datasize != 0) {')
        self._prnt('    %s = ((size_t)datasize) <= 640 ? '
                   '(%s)alloca((size_t)datasize) : NULL;' % (
            tovar, tp.get_c_name('')))
        self._prnt('    if (_cffi_convert_array_argument(_cffi_type(%d), %s, '
                   '(char **)&%s,' % (self._gettypenum(tp), fromvar, tovar))
        self._prnt('            datasize, &large_args_free) < 0)')
        self._prnt('      %s;' % errcode)
        self._prnt('  }')

    def _convert_expr_from_c(self, tp, var, context):
        if isinstance(tp, model.BasePrimitiveType):
            if tp.is_integer_type() and tp.name != '_Bool':
                return '_cffi_from_c_int(%s, %s)' % (var, tp.name)
            elif isinstance(tp, model.UnknownFloatType):
                return '_cffi_from_c_double(%s)' % (var,)
            elif tp.name != 'long double' and not tp.is_complex_type():
                cname = tp.name.replace(' ', '_')
                if cname in ('char16_t', 'char32_t'):
                    self.needs_version(VERSION_CHAR16CHAR32)
                return '_cffi_from_c_%s(%s)' % (cname, var)
            else:
                return '_cffi_from_c_deref((char *)&%s, _cffi_type(%d))' % (
                    var, self._gettypenum(tp))
        elif isinstance(tp, (model.PointerType, model.FunctionPtrType)):
            return '_cffi_from_c_pointer((char *)%s, _cffi_type(%d))' % (
                var, self._gettypenum(tp))
        elif isinstance(tp, model.ArrayType):
            return '_cffi_from_c_pointer((char *)%s, _cffi_type(%d))' % (
                var, self._gettypenum(model.PointerType(tp.item)))
        elif isinstance(tp, model.StructOrUnion):
            if tp.fldnames is None:
                raise TypeError("'%s' is used as %s, but is opaque" % (
                    tp._get_c_name(), context))
            return '_cffi_from_c_struct((char *)&%s, _cffi_type(%d))' % (
                var, self._gettypenum(tp))
        elif isinstance(tp, model.EnumType):
            return '_cffi_from_c_deref((char *)&%s, _cffi_type(%d))' % (
                var, self._gettypenum(tp))
        else:
            raise NotImplementedError(tp)

    # ----------
    # typedefs

    def _typedef_type(self, tp, name):
        return self._global_type(tp, "(*(%s *)0)" % (name,))

    def _generate_cpy_typedef_collecttype(self, tp, name):
        self._do_collect_type(self._typedef_type(tp, name))

    def _generate_cpy_typedef_decl(self, tp, name):
        pass

    def _typedef_ctx(self, tp, name):
        type_index = self._typesdict[tp]
        self._lsts["typename"].append(TypenameExpr(name, type_index))

    def _generate_cpy_typedef_ctx(self, tp, name):
        tp = self._typedef_type(tp, name)
        self._typedef_ctx(tp, name)
        if getattr(tp, "origin", None) == "unknown_type":
            self._struct_ctx(tp, tp.name, approxname=None)
        elif isinstance(tp, model.NamedPointerType):
            self._struct_ctx(tp.totype, tp.totype.name, approxname=tp.name,
                             named_ptr=tp)

    # ----------
    # function declarations

    def _generate_cpy_function_collecttype(self, tp, name):
        self._do_collect_type(tp.as_raw_function())
        if tp.ellipsis and not self.target_is_python:
            self._do_collect_type(tp)

    def _generate_cpy_function_decl(self, tp, name):
        assert not self.target_is_python
        assert isinstance(tp, model.FunctionPtrType)
        if tp.ellipsis:
            # cannot support vararg functions better than this: check for its
            # exact type (including the fixed arguments), and build it as a
            # constant function pointer (no CPython wrapper)
            self._generate_cpy_constant_decl(tp, name)
            return
        prnt = self._prnt
        numargs = len(tp.args)
        if numargs == 0:
            argname = 'noarg'
        elif numargs == 1:
            argname = 'arg0'
        else:
            argname = 'args'
        #
        # ------------------------------
        # the 'd' version of the function, only for addressof(lib, 'func')
        arguments = []
        call_arguments = []
        context = 'argument of %s' % name
        for i, type in enumerate(tp.args):
            arguments.append(type.get_c_name(' x%d' % i, context))
            call_arguments.append('x%d' % i)
        repr_arguments = ', '.join(arguments)
        repr_arguments = repr_arguments or 'void'
        if tp.abi:
            abi = tp.abi + ' '
        else:
            abi = ''
        name_and_arguments = '%s_cffi_d_%s(%s)' % (abi, name, repr_arguments)
        prnt('static %s' % (tp.result.get_c_name(name_and_arguments),))
        prnt('{')
        call_arguments = ', '.join(call_arguments)
        result_code = 'return '
        if isinstance(tp.result, model.VoidType):
            result_code = ''
        prnt('  %s%s(%s);' % (result_code, name, call_arguments))
        prnt('}')
        #
        prnt('#ifndef PYPY_VERSION')        # ------------------------------
        #
        prnt('static PyObject *')
        prnt('_cffi_f_%s(PyObject *self, PyObject *%s)' % (name, argname))
        prnt('{')
        #
        context = 'argument of %s' % name
        for i, type in enumerate(tp.args):
            arg = type.get_c_name(' x%d' % i, context)
            prnt('  %s;' % arg)
        #
        localvars = set()
        freelines = set()
        for type in tp.args:
            self._extra_local_variables(type, localvars, freelines)
        for decl in sorted(localvars):
            prnt('  %s;' % (decl,))
        #
        if not isinstance(tp.result, model.VoidType):
            result_code = 'result = '
            context = 'result of %s' % name
            result_decl = '  %s;' % tp.result.get_c_name(' result', context)
            prnt(result_decl)
            prnt('  PyObject *pyresult;')
        else:
            result_decl = None
            result_code = ''
        #
        if len(tp.args) > 1:
            rng = range(len(tp.args))
            for i in rng:
                prnt('  PyObject *arg%d;' % i)
            prnt()
            prnt('  if (!PyArg_UnpackTuple(args, "%s", %d, %d, %s))' % (
                name, len(rng), len(rng),
                ', '.join(['&arg%d' % i for i in rng])))
            prnt('    return NULL;')
        prnt()
        #
        for i, type in enumerate(tp.args):
            self._convert_funcarg_to_c(type, 'arg%d' % i, 'x%d' % i,
                                       'return NULL')
            prnt()
        #
        prnt('  Py_BEGIN_ALLOW_THREADS')
        prnt('  _cffi_restore_errno();')
        call_arguments = ['x%d' % i for i in range(len(tp.args))]
        call_arguments = ', '.join(call_arguments)
        prnt('  { %s%s(%s); }' % (result_code, name, call_arguments))
        prnt('  _cffi_save_errno();')
        prnt('  Py_END_ALLOW_THREADS')
        prnt()
        #
        prnt('  (void)self; /* unused */')
        if numargs == 0:
            prnt('  (void)noarg; /* unused */')
        if result_code:
            prnt('  pyresult = %s;' %
                 self._convert_expr_from_c(tp.result, 'result', 'result type'))
            for freeline in freelines:
                prnt('  ' + freeline)
            prnt('  return pyresult;')
        else:
            for freeline in freelines:
                prnt('  ' + freeline)
            prnt('  Py_INCREF(Py_None);')
            prnt('  return Py_None;')
        prnt('}')
        #
        prnt('#else')        # ------------------------------
        #
        # the PyPy version: need to replace struct/union arguments with
        # pointers, and if the result is a struct/union, insert a first
        # arg that is a pointer to the result.  We also do that for
        # complex args and return type.
        def need_indirection(type):
            return (isinstance(type, model.StructOrUnion) or
                    (isinstance(type, model.PrimitiveType) and
                     type.is_complex_type()))
        difference = False
        arguments = []
        call_arguments = []
        context = 'argument of %s' % name
        for i, type in enumerate(tp.args):
            indirection = ''
            if need_indirection(type):
                indirection = '*'
                difference = True
            arg = type.get_c_name(' %sx%d' % (indirection, i), context)
            arguments.append(arg)
            call_arguments.append('%sx%d' % (indirection, i))
        tp_result = tp.result
        if need_indirection(tp_result):
            context = 'result of %s' % name
            arg = tp_result.get_c_name(' *result', context)
            arguments.insert(0, arg)
            tp_result = model.void_type
            result_decl = None
            result_code = '*result = '
            difference = True
        if difference:
            repr_arguments = ', '.join(arguments)
            repr_arguments = repr_arguments or 'void'
            name_and_arguments = '%s_cffi_f_%s(%s)' % (abi, name,
                                                       repr_arguments)
            prnt('static %s' % (tp_result.get_c_name(name_and_arguments),))
            prnt('{')
            if result_decl:
                prnt(result_decl)
            call_arguments = ', '.join(call_arguments)
            prnt('  { %s%s(%s); }' % (result_code, name, call_arguments))
            if result_decl:
                prnt('  return result;')
            prnt('}')
        else:
            prnt('#  define _cffi_f_%s _cffi_d_%s' % (name, name))
        #
        prnt('#endif')        # ------------------------------
        prnt()

    def _generate_cpy_function_ctx(self, tp, name):
        if tp.ellipsis and not self.target_is_python:
            self._generate_cpy_constant_ctx(tp, name)
            return
        type_index = self._typesdict[tp.as_raw_function()]
        numargs = len(tp.args)
        if self.target_is_python:
            meth_kind = OP_DLOPEN_FUNC
        elif numargs == 0:
            meth_kind = OP_CPYTHON_BLTN_N   # 'METH_NOARGS'
        elif numargs == 1:
            meth_kind = OP_CPYTHON_BLTN_O   # 'METH_O'
        else:
            meth_kind = OP_CPYTHON_BLTN_V   # 'METH_VARARGS'
        self._lsts["global"].append(
            GlobalExpr(name, '_cffi_f_%s' % name,
                       CffiOp(meth_kind, type_index),
                       size='_cffi_d_%s' % name))

    # ----------
    # named structs or unions

    def _field_type(self, tp_struct, field_name, tp_field):
        if isinstance(tp_field, model.ArrayType):
            actual_length = tp_field.length
            if actual_length == '...':
                ptr_struct_name = tp_struct.get_c_name('*')
                actual_length = '_cffi_array_len(((%s)0)->%s)' % (
                    ptr_struct_name, field_name)
            tp_item = self._field_type(tp_struct, '%s[0]' % field_name,
                                       tp_field.item)
            tp_field = model.ArrayType(tp_item, actual_length)
        return tp_field

    def _struct_collecttype(self, tp):
        self._do_collect_type(tp)
        if self.target_is_python:
            # also requires nested anon struct/unions in ABI mode, recursively
            for fldtype in tp.anonymous_struct_fields():
                self._struct_collecttype(fldtype)

    def _struct_decl(self, tp, cname, approxname):
        if tp.fldtypes is None:
            return
        prnt = self._prnt
        checkfuncname = '_cffi_checkfld_%s' % (approxname,)
        prnt('_CFFI_UNUSED_FN')
        prnt('static void %s(%s *p)' % (checkfuncname, cname))
        prnt('{')
        prnt('  /* only to generate compile-time warnings or errors */')
        prnt('  (void)p;')
        for fname, ftype, fbitsize, fqual in self._enum_fields(tp):
            try:
                if ftype.is_integer_type() or fbitsize >= 0:
                    # accept all integers, but complain on float or double
                    if fname != '':
                        prnt("  (void)((p->%s) | 0);  /* check that '%s.%s' is "
                             "an integer */" % (fname, cname, fname))
                    continue
                # only accept exactly the type declared, except that '[]'
                # is interpreted as a '*' and so will match any array length.
                # (It would also match '*', but that's harder to detect...)
                while (isinstance(ftype, model.ArrayType)
                       and (ftype.length is None or ftype.length == '...')):
                    ftype = ftype.item
                    fname = fname + '[0]'
                prnt('  { %s = &p->%s; (void)tmp; }' % (
                    ftype.get_c_name('*tmp', 'field %r'%fname, quals=fqual),
                    fname))
            except VerificationError as e:
                prnt('  /* %s */' % str(e))   # cannot verify it, ignore
        prnt('}')
        prnt('struct _cffi_align_%s { char x; %s y; };' % (approxname, cname))
        prnt()

    def _struct_ctx(self, tp, cname, approxname, named_ptr=None):
        type_index = self._typesdict[tp]
        reason_for_not_expanding = None
        flags = []
        if isinstance(tp, model.UnionType):
            flags.append("_CFFI_F_UNION")
        if tp.fldtypes is None:
            flags.append("_CFFI_F_OPAQUE")
            reason_for_not_expanding = "opaque"
        if (tp not in self.ffi._parser._included_declarations and
                (named_ptr is None or
                 named_ptr not in self.ffi._parser._included_declarations)):
            if tp.fldtypes is None:
                pass    # opaque
            elif tp.partial or any(tp.anonymous_struct_fields()):
                pass    # field layout obtained silently from the C compiler
            else:
                flags.append("_CFFI_F_CHECK_FIELDS")
            if tp.packed:
                if tp.packed > 1:
                    raise NotImplementedError(
                        "%r is declared with 'pack=%r'; only 0 or 1 are "
                        "supported in API mode (try to use \"...;\", which "
                        "does not require a 'pack' declaration)" %
                        (tp, tp.packed))
                flags.append("_CFFI_F_PACKED")
        else:
            flags.append("_CFFI_F_EXTERNAL")
            reason_for_not_expanding = "external"
        flags = '|'.join(flags) or '0'
        c_fields = []
        if reason_for_not_expanding is None:
            enumfields = list(self._enum_fields(tp))
            for fldname, fldtype, fbitsize, fqual in enumfields:
                fldtype = self._field_type(tp, fldname, fldtype)
                self._check_not_opaque(fldtype,
                                       "field '%s.%s'" % (tp.name, fldname))
                # cname is None for _add_missing_struct_unions() only
                op = OP_NOOP
                if fbitsize >= 0:
                    op = OP_BITFIELD
                    size = '%d /* bits */' % fbitsize
                elif cname is None or (
                        isinstance(fldtype, model.ArrayType) and
                        fldtype.length is None):
                    size = '(size_t)-1'
                else:
                    size = 'sizeof(((%s)0)->%s)' % (
                        tp.get_c_name('*') if named_ptr is None
                                           else named_ptr.name,
                        fldname)
                if cname is None or fbitsize >= 0:
                    offset = '(size_t)-1'
                elif named_ptr is not None:
                    offset = '((char *)&((%s)4096)->%s) - (char *)4096' % (
                        named_ptr.name, fldname)
                else:
                    offset = 'offsetof(%s, %s)' % (tp.get_c_name(''), fldname)
                c_fields.append(
                    FieldExpr(fldname, offset, size, fbitsize,
                              CffiOp(op, self._typesdict[fldtype])))
            first_field_index = len(self._lsts["field"])
            self._lsts["field"].extend(c_fields)
            #
            if cname is None:  # unknown name, for _add_missing_struct_unions
                size = '(size_t)-2'
                align = -2
                comment = "unnamed"
            else:
                if named_ptr is not None:
                    size = 'sizeof(*(%s)0)' % (named_ptr.name,)
                    align = '-1 /* unknown alignment */'
                else:
                    size = 'sizeof(%s)' % (cname,)
                    align = 'offsetof(struct _cffi_align_%s, y)' % (approxname,)
                comment = None
        else:
            size = '(size_t)-1'
            align = -1
            first_field_index = -1
            comment = reason_for_not_expanding
        self._lsts["struct_union"].append(
            StructUnionExpr(tp.name, type_index, flags, size, align, comment,
                            first_field_index, c_fields))
        self._seen_struct_unions.add(tp)

    def _check_not_opaque(self, tp, location):
        while isinstance(tp, model.ArrayType):
            tp = tp.item
        if isinstance(tp, model.StructOrUnion) and tp.fldtypes is None:
            raise TypeError(
                "%s is of an opaque type (not declared in cdef())" % location)

    def _add_missing_struct_unions(self):
        # not very nice, but some struct declarations might be missing
        # because they don't have any known C name.  Check that they are
        # not partial (we can't complete or verify them!) and emit them
        # anonymously.
        lst = list(self._struct_unions.items())
        lst.sort(key=lambda tp_order: tp_order[1])
        for tp, order in lst:
            if tp not in self._seen_struct_unions:
                if tp.partial:
                    raise NotImplementedError("internal inconsistency: %r is "
                                              "partial but was not seen at "
                                              "this point" % (tp,))
                if tp.name.startswith('$') and tp.name[1:].isdigit():
                    approxname = tp.name[1:]
                elif tp.name == '_IO_FILE' and tp.forcename == 'FILE':
                    approxname = 'FILE'
                    self._typedef_ctx(tp, 'FILE')
                else:
                    raise NotImplementedError("internal inconsistency: %r" %
                                              (tp,))
                self._struct_ctx(tp, None, approxname)

    def _generate_cpy_struct_collecttype(self, tp, name):
        self._struct_collecttype(tp)
    _generate_cpy_union_collecttype = _generate_cpy_struct_collecttype

    def _struct_names(self, tp):
        cname = tp.get_c_name('')
        if ' ' in cname:
            return cname, cname.replace(' ', '_')
        else:
            return cname, '_' + cname

    def _generate_cpy_struct_decl(self, tp, name):
        self._struct_decl(tp, *self._struct_names(tp))
    _generate_cpy_union_decl = _generate_cpy_struct_decl

    def _generate_cpy_struct_ctx(self, tp, name):
        self._struct_ctx(tp, *self._struct_names(tp))
    _generate_cpy_union_ctx = _generate_cpy_struct_ctx

    # ----------
    # 'anonymous' declarations.  These are produced for anonymous structs
    # or unions; the 'name' is obtained by a typedef.

    def _generate_cpy_anonymous_collecttype(self, tp, name):
        if isinstance(tp, model.EnumType):
            self._generate_cpy_enum_collecttype(tp, name)
        else:
            self._struct_collecttype(tp)

    def _generate_cpy_anonymous_decl(self, tp, name):
        if isinstance(tp, model.EnumType):
            self._generate_cpy_enum_decl(tp)
        else:
            self._struct_decl(tp, name, 'typedef_' + name)

    def _generate_cpy_anonymous_ctx(self, tp, name):
        if isinstance(tp, model.EnumType):
            self._enum_ctx(tp, name)
        else:
            self._struct_ctx(tp, name, 'typedef_' + name)

    # ----------
    # constants, declared with "static const ..."

    def _generate_cpy_const(self, is_int, name, tp=None, category='const',
                            check_value=None):
        if (category, name) in self._seen_constants:
            raise VerificationError(
                "duplicate declaration of %s '%s'" % (category, name))
        self._seen_constants.add((category, name))
        #
        prnt = self._prnt
        funcname = '_cffi_%s_%s' % (category, name)
        if is_int:
            prnt('static int %s(unsigned long long *o)' % funcname)
            prnt('{')
            prnt('  int n = (%s) <= 0;' % (name,))
            prnt('  *o = (unsigned long long)((%s) | 0);'
                 '  /* check that %s is an integer */' % (name, name))
            if check_value is not None:
                if check_value > 0:
                    check_value = '%dU' % (check_value,)
                prnt('  if (!_cffi_check_int(*o, n, %s))' % (check_value,))
                prnt('    n |= 2;')
            prnt('  return n;')
            prnt('}')
        else:
            assert check_value is None
            prnt('static void %s(char *o)' % funcname)
            prnt('{')
            prnt('  *(%s)o = %s;' % (tp.get_c_name('*'), name))
            prnt('}')
        prnt()

    def _generate_cpy_constant_collecttype(self, tp, name):
        is_int = tp.is_integer_type()
        if not is_int or self.target_is_python:
            self._do_collect_type(tp)

    def _generate_cpy_constant_decl(self, tp, name):
        is_int = tp.is_integer_type()
        self._generate_cpy_const(is_int, name, tp)

    def _generate_cpy_constant_ctx(self, tp, name):
        if not self.target_is_python and tp.is_integer_type():
            type_op = CffiOp(OP_CONSTANT_INT, -1)
        else:
            if self.target_is_python:
                const_kind = OP_DLOPEN_CONST
            else:
                const_kind = OP_CONSTANT
            type_index = self._typesdict[tp]
            type_op = CffiOp(const_kind, type_index)
        self._lsts["global"].append(
            GlobalExpr(name, '_cffi_const_%s' % name, type_op))

    # ----------
    # enums

    def _generate_cpy_enum_collecttype(self, tp, name):
        self._do_collect_type(tp)

    def _generate_cpy_enum_decl(self, tp, name=None):
        for enumerator in tp.enumerators:
            self._generate_cpy_const(True, enumerator)

    def _enum_ctx(self, tp, cname):
        type_index = self._typesdict[tp]
        type_op = CffiOp(OP_ENUM, -1)
        if self.target_is_python:
            tp.check_not_partial()
        for enumerator, enumvalue in zip(tp.enumerators, tp.enumvalues):
            self._lsts["global"].append(
                GlobalExpr(enumerator, '_cffi_const_%s' % enumerator, type_op,
                           check_value=enumvalue))
        #
        if cname is not None and '$' not in cname and not self.target_is_python:
            size = "sizeof(%s)" % cname
            signed = "((%s)-1) <= 0" % cname
        else:
            basetp = tp.build_baseinttype(self.ffi, [])
            size = self.ffi.sizeof(basetp)
            signed = int(int(self.ffi.cast(basetp, -1)) < 0)
        allenums = ",".join(tp.enumerators)
        self._lsts["enum"].append(
            EnumExpr(tp.name, type_index, size, signed, allenums))

    def _generate_cpy_enum_ctx(self, tp, name):
        self._enum_ctx(tp, tp._get_c_name())

    # ----------
    # macros: for now only for integers

    def _generate_cpy_macro_collecttype(self, tp, name):
        pass

    def _generate_cpy_macro_decl(self, tp, name):
        if tp == '...':
            check_value = None
        else:
            check_value = tp     # an integer
        self._generate_cpy_const(True, name, check_value=check_value)

    def _generate_cpy_macro_ctx(self, tp, name):
        if tp == '...':
            if self.target_is_python:
                raise VerificationError(
                    "cannot use the syntax '...' in '#define %s ...' when "
                    "using the ABI mode" % (name,))
            check_value = None
        else:
            check_value = tp     # an integer
        type_op = CffiOp(OP_CONSTANT_INT, -1)
        self._lsts["global"].append(
            GlobalExpr(name, '_cffi_const_%s' % name, type_op,
                       check_value=check_value))

    # ----------
    # global variables

    def _global_type(self, tp, global_name):
        if isinstance(tp, model.ArrayType):
            actual_length = tp.length
            if actual_length == '...':
                actual_length = '_cffi_array_len(%s)' % (global_name,)
            tp_item = self._global_type(tp.item, '%s[0]' % global_name)
            tp = model.ArrayType(tp_item, actual_length)
        return tp

    def _generate_cpy_variable_collecttype(self, tp, name):
        self._do_collect_type(self._global_type(tp, name))

    def _generate_cpy_variable_decl(self, tp, name):
        prnt = self._prnt
        tp = self._global_type(tp, name)
        if isinstance(tp, model.ArrayType) and tp.length is None:
            tp = tp.item
            ampersand = ''
        else:
            ampersand = '&'
        # This code assumes that casts from "tp *" to "void *" is a
        # no-op, i.e. a function that returns a "tp *" can be called
        # as if it returned a "void *".  This should be generally true
        # on any modern machine.  The only exception to that rule (on
        # uncommon architectures, and as far as I can tell) might be
        # if 'tp' were a function type, but that is not possible here.
        # (If 'tp' is a function _pointer_ type, then casts from "fn_t
        # **" to "void *" are again no-ops, as far as I can tell.)
        decl = '*_cffi_var_%s(void)' % (name,)
        prnt('static ' + tp.get_c_name(decl, quals=self._current_quals))
        prnt('{')
        prnt('  return %s(%s);' % (ampersand, name))
        prnt('}')
        prnt()

    def _generate_cpy_variable_ctx(self, tp, name):
        tp = self._global_type(tp, name)
        type_index = self._typesdict[tp]
        if self.target_is_python:
            op = OP_GLOBAL_VAR
        else:
            op = OP_GLOBAL_VAR_F
        self._lsts["global"].append(
            GlobalExpr(name, '_cffi_var_%s' % name, CffiOp(op, type_index)))

    # ----------
    # extern "Python"

    def _generate_cpy_extern_python_collecttype(self, tp, name):
        assert isinstance(tp, model.FunctionPtrType)
        self._do_collect_type(tp)
    _generate_cpy_dllexport_python_collecttype = \
      _generate_cpy_extern_python_plus_c_collecttype = \
      _generate_cpy_extern_python_collecttype

    def _extern_python_decl(self, tp, name, tag_and_space):
        prnt = self._prnt
        if isinstance(tp.result, model.VoidType):
            size_of_result = '0'
        else:
            context = 'result of %s' % name
            size_of_result = '(int)sizeof(%s)' % (
                tp.result.get_c_name('', context),)
        prnt('static struct _cffi_externpy_s _cffi_externpy__%s =' % name)
        prnt('  { "%s.%s", %s, 0, 0 };' % (
            self.module_name, name, size_of_result))
        prnt()
        #
        arguments = []
        context = 'argument of %s' % name
        for i, type in enumerate(tp.args):
            arg = type.get_c_name(' a%d' % i, context)
            arguments.append(arg)
        #
        repr_arguments = ', '.join(arguments)
        repr_arguments = repr_arguments or 'void'
        name_and_arguments = '%s(%s)' % (name, repr_arguments)
        if tp.abi == "__stdcall":
            name_and_arguments = '_cffi_stdcall ' + name_and_arguments
        #
        def may_need_128_bits(tp):
            return (isinstance(tp, model.PrimitiveType) and
                    tp.name == 'long double')
        #
        size_of_a = max(len(tp.args)*8, 8)
        if may_need_128_bits(tp.result):
            size_of_a = max(size_of_a, 16)
        if isinstance(tp.result, model.StructOrUnion):
            size_of_a = 'sizeof(%s) > %d ? sizeof(%s) : %d' % (
                tp.result.get_c_name(''), size_of_a,
                tp.result.get_c_name(''), size_of_a)
        prnt('%s%s' % (tag_and_space, tp.result.get_c_name(name_and_arguments)))
        prnt('{')
        prnt('  char a[%s];' % size_of_a)
        prnt('  char *p = a;')
        for i, type in enumerate(tp.args):
            arg = 'a%d' % i
            if (isinstance(type, model.StructOrUnion) or
                    may_need_128_bits(type)):
                arg = '&' + arg
                type = model.PointerType(type)
            prnt('  *(%s)(p + %d) = %s;' % (type.get_c_name('*'), i*8, arg))
        prnt('  _cffi_call_python(&_cffi_externpy__%s, p);' % name)
        if not isinstance(tp.result, model.VoidType):
            prnt('  return *(%s)p;' % (tp.result.get_c_name('*'),))
        prnt('}')
        prnt()
        self._num_externpy += 1

    def _generate_cpy_extern_python_decl(self, tp, name):
        self._extern_python_decl(tp, name, 'static ')

    def _generate_cpy_dllexport_python_decl(self, tp, name):
        self._extern_python_decl(tp, name, 'CFFI_DLLEXPORT ')

    def _generate_cpy_extern_python_plus_c_decl(self, tp, name):
        self._extern_python_decl(tp, name, '')

    def _generate_cpy_extern_python_ctx(self, tp, name):
        if self.target_is_python:
            raise VerificationError(
                "cannot use 'extern \"Python\"' in the ABI mode")
        if tp.ellipsis:
            raise NotImplementedError("a vararg function is extern \"Python\"")
        type_index = self._typesdict[tp]
        type_op = CffiOp(OP_EXTERN_PYTHON, type_index)
        self._lsts["global"].append(
            GlobalExpr(name, '&_cffi_externpy__%s' % name, type_op, name))

    _generate_cpy_dllexport_python_ctx = \
      _generate_cpy_extern_python_plus_c_ctx = \
      _generate_cpy_extern_python_ctx

    def _print_string_literal_in_array(self, s):
        prnt = self._prnt
        prnt('// # NB. this is not a string because of a size limit in MSVC')
        if not isinstance(s, bytes):    # unicode
            s = s.encode('utf-8')       # -> bytes
        else:
            s.decode('utf-8')           # got bytes, check for valid utf-8
        try:
            s.decode('ascii')
        except UnicodeDecodeError:
            s = b'# -*- encoding: utf8 -*-\n' + s
        for line in s.splitlines(True):
            comment = line
            if type('//') is bytes:     # python2
                line = map(ord, line)   #     make a list of integers
            else:                       # python3
                # type(line) is bytes, which enumerates like a list of integers
                comment = ascii(comment)[1:-1]
            prnt(('// ' + comment).rstrip())
            printed_line = ''
            for c in line:
                if len(printed_line) >= 76:
                    prnt(printed_line)
                    printed_line = ''
                printed_line += '%d,' % (c,)
            prnt(printed_line)

    # ----------
    # emitting the opcodes for individual types

    def _emit_bytecode_VoidType(self, tp, index):
        self.cffi_types[index] = CffiOp(OP_PRIMITIVE, PRIM_VOID)

    def _emit_bytecode_PrimitiveType(self, tp, index):
        prim_index = PRIMITIVE_TO_INDEX[tp.name]
        self.cffi_types[index] = CffiOp(OP_PRIMITIVE, prim_index)

    def _emit_bytecode_UnknownIntegerType(self, tp, index):
        s = ('_cffi_prim_int(sizeof(%s), (\n'
             '           ((%s)-1) | 0 /* check that %s is an integer type */\n'
             '         ) <= 0)' % (tp.name, tp.name, tp.name))
        self.cffi_types[index] = CffiOp(OP_PRIMITIVE, s)

    def _emit_bytecode_UnknownFloatType(self, tp, index):
        s = ('_cffi_prim_float(sizeof(%s) *\n'
             '           (((%s)1) / 2) * 2 /* integer => 0, float => 1 */\n'
             '         )' % (tp.name, tp.name))
        self.cffi_types[index] = CffiOp(OP_PRIMITIVE, s)

    def _emit_bytecode_RawFunctionType(self, tp, index):
        self.cffi_types[index] = CffiOp(OP_FUNCTION, self._typesdict[tp.result])
        index += 1
        for tp1 in tp.args:
            realindex = self._typesdict[tp1]
            if index != realindex:
                if isinstance(tp1, model.PrimitiveType):
                    self._emit_bytecode_PrimitiveType(tp1, index)
                else:
                    self.cffi_types[index] = CffiOp(OP_NOOP, realindex)
            index += 1
        flags = int(tp.ellipsis)
        if tp.abi is not None:
            if tp.abi == '__stdcall':
                flags |= 2
            else:
                raise NotImplementedError("abi=%r" % (tp.abi,))
        self.cffi_types[index] = CffiOp(OP_FUNCTION_END, flags)

    def _emit_bytecode_PointerType(self, tp, index):
        self.cffi_types[index] = CffiOp(OP_POINTER, self._typesdict[tp.totype])

    _emit_bytecode_ConstPointerType = _emit_bytecode_PointerType
    _emit_bytecode_NamedPointerType = _emit_bytecode_PointerType

    def _emit_bytecode_FunctionPtrType(self, tp, index):
        raw = tp.as_raw_function()
        self.cffi_types[index] = CffiOp(OP_POINTER, self._typesdict[raw])

    def _emit_bytecode_ArrayType(self, tp, index):
        item_index = self._typesdict[tp.item]
        if tp.length is None:
            self.cffi_types[index] = CffiOp(OP_OPEN_ARRAY, item_index)
        elif tp.length == '...':
            raise VerificationError(
                "type %s badly placed: the '...' array length can only be "
                "used on global arrays or on fields of structures" % (
                    str(tp).replace('/*...*/', '...'),))
        else:
            assert self.cffi_types[index + 1] == 'LEN'
            self.cffi_types[index] = CffiOp(OP_ARRAY, item_index)
            self.cffi_types[index + 1] = CffiOp(None, str(tp.length))

    def _emit_bytecode_StructType(self, tp, index):
        struct_index = self._struct_unions[tp]
        self.cffi_types[index] = CffiOp(OP_STRUCT_UNION, struct_index)
    _emit_bytecode_UnionType = _emit_bytecode_StructType

    def _emit_bytecode_EnumType(self, tp, index):
        enum_index = self._enums[tp]
        self.cffi_types[index] = CffiOp(OP_ENUM, enum_index)


if sys.version_info >= (3,):
    NativeIO = io.StringIO
else:
    class NativeIO(io.BytesIO):
        def write(self, s):
            if isinstance(s, unicode):
                s = s.encode('ascii')
            super(NativeIO, self).write(s)

def _is_file_like(maybefile):
    # compare to xml.etree.ElementTree._get_writer
    return hasattr(maybefile, 'write')

def _make_c_or_py_source(ffi, module_name, preamble, target_file, verbose):
    if verbose:
        print("generating %s" % (target_file,))
    recompiler = Recompiler(ffi, module_name,
                            target_is_python=(preamble is None))
    recompiler.collect_type_table()
    recompiler.collect_step_tables()
    if _is_file_like(target_file):
        recompiler.write_source_to_f(target_file, preamble)
        return True
    f = NativeIO()
    recompiler.write_source_to_f(f, preamble)
    output = f.getvalue()
    try:
        with open(target_file, 'r') as f1:
            if f1.read(len(output) + 1) != output:
                raise IOError
        if verbose:
            print("(already up-to-date)")
        return False     # already up-to-date
    except IOError:
        tmp_file = '%s.~%d' % (target_file, os.getpid())
        with open(tmp_file, 'w') as f1:
            f1.write(output)
        try:
            os.rename(tmp_file, target_file)
        except OSError:
            os.unlink(target_file)
            os.rename(tmp_file, target_file)
        return True

def make_c_source(ffi, module_name, preamble, target_c_file, verbose=False):
    assert preamble is not None
    return _make_c_or_py_source(ffi, module_name, preamble, target_c_file,
                                verbose)

def make_py_source(ffi, module_name, target_py_file, verbose=False):
    return _make_c_or_py_source(ffi, module_name, None, target_py_file,
                                verbose)

def _modname_to_file(outputdir, modname, extension):
    parts = modname.split('.')
    try:
        os.makedirs(os.path.join(outputdir, *parts[:-1]))
    except OSError:
        pass
    parts[-1] += extension
    return os.path.join(outputdir, *parts), parts


# Aaargh.  Distutils is not tested at all for the purpose of compiling
# DLLs that are not extension modules.  Here are some hacks to work
# around that, in the _patch_for_*() functions...

def _patch_meth(patchlist, cls, name, new_meth):
    old = getattr(cls, name)
    patchlist.append((cls, name, old))
    setattr(cls, name, new_meth)
    return old

def _unpatch_meths(patchlist):
    for cls, name, old_meth in reversed(patchlist):
        setattr(cls, name, old_meth)

def _patch_for_embedding(patchlist):
    if sys.platform == 'win32':
        # we must not remove the manifest when building for embedding!
        # FUTURE: this module was removed in setuptools 74; this is likely dead code and should be removed,
        #  since the toolchain it supports (VS2005-2008) is also long dead.
        from cffi._shimmed_dist_utils import MSVCCompiler
        if MSVCCompiler is not None:
            _patch_meth(patchlist, MSVCCompiler, '_remove_visual_c_ref',
                        lambda self, manifest_file: manifest_file)

    if sys.platform == 'darwin':
        # we must not make a '-bundle', but a '-dynamiclib' instead
        from cffi._shimmed_dist_utils import CCompiler
        def my_link_shared_object(self, *args, **kwds):
            if '-bundle' in self.linker_so:
                self.linker_so = list(self.linker_so)
                i = self.linker_so.index('-bundle')
                self.linker_so[i] = '-dynamiclib'
            return old_link_shared_object(self, *args, **kwds)
        old_link_shared_object = _patch_meth(patchlist, CCompiler,
                                             'link_shared_object',
                                             my_link_shared_object)

def _patch_for_target(patchlist, target):
    from cffi._shimmed_dist_utils import build_ext
    # if 'target' is different from '*', we need to patch some internal
    # method to just return this 'target' value, instead of having it
    # built from module_name
    if target.endswith('.*'):
        target = target[:-2]
        if sys.platform == 'win32':
            target += '.dll'
        elif sys.platform == 'darwin':
            target += '.dylib'
        else:
            target += '.so'
    _patch_meth(patchlist, build_ext, 'get_ext_filename',
                lambda self, ext_name: target)


def recompile(ffi, module_name, preamble, tmpdir='.', call_c_compiler=True,
              c_file=None, source_extension='.c', extradir=None,
              compiler_verbose=1, target=None, debug=None,
              uses_ffiplatform=True, **kwds):
    if not isinstance(module_name, str):
        module_name = module_name.encode('ascii')
    if ffi._windows_unicode:
        ffi._apply_windows_unicode(kwds)
    if preamble is not None:
        if call_c_compiler and _is_file_like(c_file):
            raise TypeError("Writing to file-like objects is not supported "
                            "with call_c_compiler=True")
        embedding = (ffi._embedding is not None)
        if embedding:
            ffi._apply_embedding_fix(kwds)
        if c_file is None:
            c_file, parts = _modname_to_file(tmpdir, module_name,
                                             source_extension)
            if extradir:
                parts = [extradir] + parts
            ext_c_file = os.path.join(*parts)
        else:
            ext_c_file = c_file
        #
        if target is None:
            if embedding:
                target = '%s.*' % module_name
            else:
                target = '*'
        #
        if uses_ffiplatform:
            ext = ffiplatform.get_extension(ext_c_file, module_name, **kwds)
        else:
            ext = None
        updated = make_c_source(ffi, module_name, preamble, c_file,
                                verbose=compiler_verbose)
        if call_c_compiler:
            patchlist = []
            cwd = os.getcwd()
            try:
                if embedding:
                    _patch_for_embedding(patchlist)
                if target != '*':
                    _patch_for_target(patchlist, target)
                if compiler_verbose:
                    if tmpdir == '.':
                        msg = 'the current directory is'
                    else:
                        msg = 'setting the current directory to'
                    print('%s %r' % (msg, os.path.abspath(tmpdir)))
                os.chdir(tmpdir)
                outputfilename = ffiplatform.compile('.', ext,
                                                     compiler_verbose, debug)
            finally:
                os.chdir(cwd)
                _unpatch_meths(patchlist)
            return outputfilename
        else:
            return ext, updated
    else:
        if c_file is None:
            c_file, _ = _modname_to_file(tmpdir, module_name, '.py')
        updated = make_py_source(ffi, module_name, c_file,
                                 verbose=compiler_verbose)
        if call_c_compiler:
            return c_file
        else:
            return None, updated


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\lambdify.py ===
"""
This module provides convenient functions to transform SymPy expressions to
lambda functions which can be used to calculate numerical values very fast.
"""

from __future__ import annotations
from typing import Any

import builtins
import inspect
import keyword
import textwrap
import linecache
import weakref

# Required despite static analysis claiming it is not used
from sympy.external import import_module # noqa:F401
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import (is_sequence, iterable,
    NotIterable, flatten)
from sympy.utilities.misc import filldedent


__doctest_requires__ = {('lambdify',): ['numpy', 'tensorflow']}


# Default namespaces, letting us define translations that can't be defined
# by simple variable maps, like I => 1j
MATH_DEFAULT: dict[str, Any] = {}
CMATH_DEFAULT: dict[str,Any] = {}
MPMATH_DEFAULT: dict[str, Any] = {}
NUMPY_DEFAULT: dict[str, Any] = {"I": 1j}
SCIPY_DEFAULT: dict[str, Any] = {"I": 1j}
CUPY_DEFAULT: dict[str, Any] = {"I": 1j}
JAX_DEFAULT: dict[str, Any] = {"I": 1j}
TENSORFLOW_DEFAULT: dict[str, Any] = {}
TORCH_DEFAULT: dict[str, Any] = {"I": 1j}
SYMPY_DEFAULT: dict[str, Any] = {}
NUMEXPR_DEFAULT: dict[str, Any] = {}

# These are the namespaces the lambda functions will use.
# These are separate from the names above because they are modified
# throughout this file, whereas the defaults should remain unmodified.

MATH = MATH_DEFAULT.copy()
CMATH = CMATH_DEFAULT.copy()
MPMATH = MPMATH_DEFAULT.copy()
NUMPY = NUMPY_DEFAULT.copy()
SCIPY = SCIPY_DEFAULT.copy()
CUPY = CUPY_DEFAULT.copy()
JAX = JAX_DEFAULT.copy()
TENSORFLOW = TENSORFLOW_DEFAULT.copy()
TORCH = TORCH_DEFAULT.copy()
SYMPY = SYMPY_DEFAULT.copy()
NUMEXPR = NUMEXPR_DEFAULT.copy()


# Mappings between SymPy and other modules function names.
MATH_TRANSLATIONS = {
    "ceiling": "ceil",
    "E": "e",
    "ln": "log",
}

CMATH_TRANSLATIONS: dict[str, str] = {}

# NOTE: This dictionary is reused in Function._eval_evalf to allow subclasses
# of Function to automatically evalf.
MPMATH_TRANSLATIONS = {
    "Abs": "fabs",
    "elliptic_k": "ellipk",
    "elliptic_f": "ellipf",
    "elliptic_e": "ellipe",
    "elliptic_pi": "ellippi",
    "ceiling": "ceil",
    "chebyshevt": "chebyt",
    "chebyshevu": "chebyu",
    "assoc_legendre": "legenp",
    "E": "e",
    "I": "j",
    "ln": "log",
    #"lowergamma":"lower_gamma",
    "oo": "inf",
    #"uppergamma":"upper_gamma",
    "LambertW": "lambertw",
    "MutableDenseMatrix": "matrix",
    "ImmutableDenseMatrix": "matrix",
    "conjugate": "conj",
    "dirichlet_eta": "altzeta",
    "Ei": "ei",
    "Shi": "shi",
    "Chi": "chi",
    "Si": "si",
    "Ci": "ci",
    "RisingFactorial": "rf",
    "FallingFactorial": "ff",
    "betainc_regularized": "betainc",
}

NUMPY_TRANSLATIONS: dict[str, str] = {
    "Heaviside": "heaviside",
}
SCIPY_TRANSLATIONS: dict[str, str] = {
    "jn" : "spherical_jn",
    "yn" : "spherical_yn"
}
CUPY_TRANSLATIONS: dict[str, str] = {}
JAX_TRANSLATIONS: dict[str, str] = {}

TENSORFLOW_TRANSLATIONS: dict[str, str] = {}
TORCH_TRANSLATIONS: dict[str, str] = {}

NUMEXPR_TRANSLATIONS: dict[str, str] = {}

# Available modules:
MODULES = {
    "math": (MATH, MATH_DEFAULT, MATH_TRANSLATIONS, ("from math import *",)),
    "cmath": (CMATH, CMATH_DEFAULT, CMATH_TRANSLATIONS, ("import cmath; from cmath import *",)),
    "mpmath": (MPMATH, MPMATH_DEFAULT, MPMATH_TRANSLATIONS, ("from mpmath import *",)),
    "numpy": (NUMPY, NUMPY_DEFAULT, NUMPY_TRANSLATIONS, ("import numpy; from numpy import *; from numpy.linalg import *",)),
    "scipy": (SCIPY, SCIPY_DEFAULT, SCIPY_TRANSLATIONS, ("import scipy; import numpy; from scipy.special import *",)),
    "cupy": (CUPY, CUPY_DEFAULT, CUPY_TRANSLATIONS, ("import cupy",)),
    "jax": (JAX, JAX_DEFAULT, JAX_TRANSLATIONS, ("import jax",)),
    "tensorflow": (TENSORFLOW, TENSORFLOW_DEFAULT, TENSORFLOW_TRANSLATIONS, ("import tensorflow",)),
    "torch": (TORCH, TORCH_DEFAULT, TORCH_TRANSLATIONS, ("import torch",)),
    "sympy": (SYMPY, SYMPY_DEFAULT, {}, (
        "from sympy.functions import *",
        "from sympy.matrices import *",
        "from sympy import Integral, pi, oo, nan, zoo, E, I",)),
    "numexpr" : (NUMEXPR, NUMEXPR_DEFAULT, NUMEXPR_TRANSLATIONS,
                 ("import_module('numexpr')", )),
}


def _import(module, reload=False):
    """
    Creates a global translation dictionary for module.

    The argument module has to be one of the following strings: "math","cmath"
    "mpmath", "numpy", "sympy", "tensorflow", "jax".
    These dictionaries map names of Python functions to their equivalent in
    other modules.
    """
    try:
        namespace, namespace_default, translations, import_commands = MODULES[
            module]
    except KeyError:
        raise NameError(
            "'%s' module cannot be used for lambdification" % module)

    # Clear namespace or exit
    if namespace != namespace_default:
        # The namespace was already generated, don't do it again if not forced.
        if reload:
            namespace.clear()
            namespace.update(namespace_default)
        else:
            return

    for import_command in import_commands:
        if import_command.startswith('import_module'):
            module = eval(import_command)

            if module is not None:
                namespace.update(module.__dict__)
                continue
        else:
            try:
                exec(import_command, {}, namespace)
                continue
            except ImportError:
                pass

        raise ImportError(
            "Cannot import '%s' with '%s' command" % (module, import_command))

    # Add translated names to namespace
    for sympyname, translation in translations.items():
        namespace[sympyname] = namespace[translation]

    # For computing the modulus of a SymPy expression we use the builtin abs
    # function, instead of the previously used fabs function for all
    # translation modules. This is because the fabs function in the math
    # module does not accept complex valued arguments. (see issue 9474). The
    # only exception, where we don't use the builtin abs function is the
    # mpmath translation module, because mpmath.fabs returns mpf objects in
    # contrast to abs().
    if 'Abs' not in namespace:
        namespace['Abs'] = abs

# Used for dynamically generated filenames that are inserted into the
# linecache.
_lambdify_generated_counter = 1


@doctest_depends_on(modules=('numpy', 'scipy', 'tensorflow',), python_version=(3,))
def lambdify(args, expr, modules=None, printer=None, use_imps=True,
             dummify=False, cse=False, docstring_limit=1000):
    """Convert a SymPy expression into a function that allows for fast
    numeric evaluation.

    .. warning::
       This function uses ``exec``, and thus should not be used on
       unsanitized input.

    .. deprecated:: 1.7
       Passing a set for the *args* parameter is deprecated as sets are
       unordered. Use an ordered iterable such as a list or tuple.

    Explanation
    ===========

    For example, to convert the SymPy expression ``sin(x) + cos(x)`` to an
    equivalent NumPy function that numerically evaluates it:

    >>> from sympy import sin, cos, symbols, lambdify
    >>> import numpy as np
    >>> x = symbols('x')
    >>> expr = sin(x) + cos(x)
    >>> expr
    sin(x) + cos(x)
    >>> f = lambdify(x, expr, 'numpy')
    >>> a = np.array([1, 2])
    >>> f(a)
    [1.38177329 0.49315059]

    The primary purpose of this function is to provide a bridge from SymPy
    expressions to numerical libraries such as NumPy, SciPy, NumExpr, mpmath,
    and tensorflow. In general, SymPy functions do not work with objects from
    other libraries, such as NumPy arrays, and functions from numeric
    libraries like NumPy or mpmath do not work on SymPy expressions.
    ``lambdify`` bridges the two by converting a SymPy expression to an
    equivalent numeric function.

    The basic workflow with ``lambdify`` is to first create a SymPy expression
    representing whatever mathematical function you wish to evaluate. This
    should be done using only SymPy functions and expressions. Then, use
    ``lambdify`` to convert this to an equivalent function for numerical
    evaluation. For instance, above we created ``expr`` using the SymPy symbol
    ``x`` and SymPy functions ``sin`` and ``cos``, then converted it to an
    equivalent NumPy function ``f``, and called it on a NumPy array ``a``.

    Parameters
    ==========

    args : List[Symbol]
        A variable or a list of variables whose nesting represents the
        nesting of the arguments that will be passed to the function.

        Variables can be symbols, undefined functions, or matrix symbols.

        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z

        The list of variables should match the structure of how the
        arguments will be passed to the function. Simply enclose the
        parameters as they will be passed in a list.

        To call a function like ``f(x)`` then ``[x]``
        should be the first argument to ``lambdify``; for this
        case a single ``x`` can also be used:

        >>> f = lambdify(x, x + 1)
        >>> f(1)
        2
        >>> f = lambdify([x], x + 1)
        >>> f(1)
        2

        To call a function like ``f(x, y)`` then ``[x, y]`` will
        be the first argument of the ``lambdify``:

        >>> f = lambdify([x, y], x + y)
        >>> f(1, 1)
        2

        To call a function with a single 3-element tuple like
        ``f((x, y, z))`` then ``[(x, y, z)]`` will be the first
        argument of the ``lambdify``:

        >>> f = lambdify([(x, y, z)], Eq(z**2, x**2 + y**2))
        >>> f((3, 4, 5))
        True

        If two args will be passed and the first is a scalar but
        the second is a tuple with two arguments then the items
        in the list should match that structure:

        >>> f = lambdify([x, (y, z)], x + y + z)
        >>> f(1, (2, 3))
        6

    expr : Expr
        An expression, list of expressions, or matrix to be evaluated.

        Lists may be nested.
        If the expression is a list, the output will also be a list.

        >>> f = lambdify(x, [x, [x + 1, x + 2]])
        >>> f(1)
        [1, [2, 3]]

        If it is a matrix, an array will be returned (for the NumPy module).

        >>> from sympy import Matrix
        >>> f = lambdify(x, Matrix([x, x + 1]))
        >>> f(1)
        [[1]
        [2]]

        Note that the argument order here (variables then expression) is used
        to emulate the Python ``lambda`` keyword. ``lambdify(x, expr)`` works
        (roughly) like ``lambda x: expr``
        (see :ref:`lambdify-how-it-works` below).

    modules : str, optional
        Specifies the numeric library to use.

        If not specified, *modules* defaults to:

        - ``["scipy", "numpy"]`` if SciPy is installed
        - ``["numpy"]`` if only NumPy is installed
        - ``["math","cmath", "mpmath", "sympy"]`` if neither is installed.

        That is, SymPy functions are replaced as far as possible by
        either ``scipy`` or ``numpy`` functions if available, and Python's
        standard library ``math`` and ``cmath``, or ``mpmath`` functions otherwise.

        *modules* can be one of the following types:

        - The strings ``"math"``, ``"cmath"``, ``"mpmath"``, ``"numpy"``, ``"numexpr"``,
          ``"scipy"``, ``"sympy"``, or ``"tensorflow"`` or ``"jax"``. This uses the
          corresponding printer and namespace mapping for that module.
        - A module (e.g., ``math``). This uses the global namespace of the
          module. If the module is one of the above known modules, it will
          also use the corresponding printer and namespace mapping
          (i.e., ``modules=numpy`` is equivalent to ``modules="numpy"``).
        - A dictionary that maps names of SymPy functions to arbitrary
          functions
          (e.g., ``{'sin': custom_sin}``).
        - A list that contains a mix of the arguments above, with higher
          priority given to entries appearing first
          (e.g., to use the NumPy module but override the ``sin`` function
          with a custom version, you can use
          ``[{'sin': custom_sin}, 'numpy']``).

    dummify : bool, optional
        Whether or not the variables in the provided expression that are not
        valid Python identifiers are substituted with dummy symbols.

        This allows for undefined functions like ``Function('f')(t)`` to be
        supplied as arguments. By default, the variables are only dummified
        if they are not valid Python identifiers.

        Set ``dummify=True`` to replace all arguments with dummy symbols
        (if ``args`` is not a string) - for example, to ensure that the
        arguments do not redefine any built-in names.

    cse : bool, or callable, optional
        Large expressions can be computed more efficiently when
        common subexpressions are identified and precomputed before
        being used multiple time. Finding the subexpressions will make
        creation of the 'lambdify' function slower, however.

        When ``True``, ``sympy.simplify.cse`` is used, otherwise (the default)
        the user may pass a function matching the ``cse`` signature.

    docstring_limit : int or None
        When lambdifying large expressions, a significant proportion of the time
        spent inside ``lambdify`` is spent producing a string representation of
        the expression for use in the automatically generated docstring of the
        returned function. For expressions containing hundreds or more nodes the
        resulting docstring often becomes so long and dense that it is difficult
        to read. To reduce the runtime of lambdify, the rendering of the full
        expression inside the docstring can be disabled.

        When ``None``, the full expression is rendered in the docstring. When
        ``0`` or a negative ``int``, an ellipsis is rendering in the docstring
        instead of the expression. When a strictly positive ``int``, if the
        number of nodes in the expression exceeds ``docstring_limit`` an
        ellipsis is rendered in the docstring, otherwise a string representation
        of the expression is rendered as normal. The default is ``1000``.

    Examples
    ========

    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import sqrt, sin, Matrix
    >>> from sympy import Function
    >>> from sympy.abc import w, x, y, z

    >>> f = lambdify(x, x**2)
    >>> f(2)
    4
    >>> f = lambdify((x, y, z), [z, y, x])
    >>> f(1,2,3)
    [3, 2, 1]
    >>> f = lambdify(x, sqrt(x))
    >>> f(4)
    2.0
    >>> f = lambdify((x, y), sin(x*y)**2)
    >>> f(0, 5)
    0.0
    >>> row = lambdify((x, y), Matrix((x, x + y)).T, modules='sympy')
    >>> row(1, 2)
    Matrix([[1, 3]])

    ``lambdify`` can be used to translate SymPy expressions into mpmath
    functions. This may be preferable to using ``evalf`` (which uses mpmath on
    the backend) in some cases.

    >>> f = lambdify(x, sin(x), 'mpmath')
    >>> f(1)
    0.8414709848078965

    Tuple arguments are handled and the lambdified function should
    be called with the same type of arguments as were used to create
    the function:

    >>> f = lambdify((x, (y, z)), x + y)
    >>> f(1, (2, 4))
    3

    The ``flatten`` function can be used to always work with flattened
    arguments:

    >>> from sympy.utilities.iterables import flatten
    >>> args = w, (x, (y, z))
    >>> vals = 1, (2, (3, 4))
    >>> f = lambdify(flatten(args), w + x + y + z)
    >>> f(*flatten(vals))
    10

    Functions present in ``expr`` can also carry their own numerical
    implementations, in a callable attached to the ``_imp_`` attribute. This
    can be used with undefined functions using the ``implemented_function``
    factory:

    >>> f = implemented_function(Function('f'), lambda x: x+1)
    >>> func = lambdify(x, f(x))
    >>> func(4)
    5

    ``lambdify`` always prefers ``_imp_`` implementations to implementations
    in other namespaces, unless the ``use_imps`` input parameter is False.

    Usage with Tensorflow:

    >>> import tensorflow as tf
    >>> from sympy import Max, sin, lambdify
    >>> from sympy.abc import x

    >>> f = Max(x, sin(x))
    >>> func = lambdify(x, f, 'tensorflow')

    After tensorflow v2, eager execution is enabled by default.
    If you want to get the compatible result across tensorflow v1 and v2
    as same as this tutorial, run this line.

    >>> tf.compat.v1.enable_eager_execution()

    If you have eager execution enabled, you can get the result out
    immediately as you can use numpy.

    If you pass tensorflow objects, you may get an ``EagerTensor``
    object instead of value.

    >>> result = func(tf.constant(1.0))
    >>> print(result)
    tf.Tensor(1.0, shape=(), dtype=float32)
    >>> print(result.__class__)
    <class 'tensorflow.python.framework.ops.EagerTensor'>

    You can use ``.numpy()`` to get the numpy value of the tensor.

    >>> result.numpy()
    1.0

    >>> var = tf.Variable(2.0)
    >>> result = func(var) # also works for tf.Variable and tf.Placeholder
    >>> result.numpy()
    2.0

    And it works with any shape array.

    >>> tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> result = func(tensor)
    >>> result.numpy()
    [[1. 2.]
     [3. 4.]]

    Notes
    =====

    - For functions involving large array calculations, numexpr can provide a
      significant speedup over numpy. Please note that the available functions
      for numexpr are more limited than numpy but can be expanded with
      ``implemented_function`` and user defined subclasses of Function. If
      specified, numexpr may be the only option in modules. The official list
      of numexpr functions can be found at:
      https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions

    - In the above examples, the generated functions can accept scalar
      values or numpy arrays as arguments.  However, in some cases
      the generated function relies on the input being a numpy array:

      >>> import numpy
      >>> from sympy import Piecewise
      >>> from sympy.testing.pytest import ignore_warnings
      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "numpy")

      >>> with ignore_warnings(RuntimeWarning):
      ...     f(numpy.array([-1, 0, 1, 2]))
      [-1.   0.   1.   0.5]

      >>> f(0)
      Traceback (most recent call last):
          ...
      ZeroDivisionError: division by zero

      In such cases, the input should be wrapped in a numpy array:

      >>> with ignore_warnings(RuntimeWarning):
      ...     float(f(numpy.array([0])))
      0.0

      Or if numpy functionality is not required another module can be used:

      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "math")
      >>> f(0)
      0

    .. _lambdify-how-it-works:

    How it works
    ============

    When using this function, it helps a great deal to have an idea of what it
    is doing. At its core, lambdify is nothing more than a namespace
    translation, on top of a special printer that makes some corner cases work
    properly.

    To understand lambdify, first we must properly understand how Python
    namespaces work. Say we had two files. One called ``sin_cos_sympy.py``,
    with

    .. code:: python

        # sin_cos_sympy.py

        from sympy.functions.elementary.trigonometric import (cos, sin)

        def sin_cos(x):
            return sin(x) + cos(x)


    and one called ``sin_cos_numpy.py`` with

    .. code:: python

        # sin_cos_numpy.py

        from numpy import sin, cos

        def sin_cos(x):
            return sin(x) + cos(x)

    The two files define an identical function ``sin_cos``. However, in the
    first file, ``sin`` and ``cos`` are defined as the SymPy ``sin`` and
    ``cos``. In the second, they are defined as the NumPy versions.

    If we were to import the first file and use the ``sin_cos`` function, we
    would get something like

    >>> from sin_cos_sympy import sin_cos # doctest: +SKIP
    >>> sin_cos(1) # doctest: +SKIP
    cos(1) + sin(1)

    On the other hand, if we imported ``sin_cos`` from the second file, we
    would get

    >>> from sin_cos_numpy import sin_cos # doctest: +SKIP
    >>> sin_cos(1) # doctest: +SKIP
    1.38177329068

    In the first case we got a symbolic output, because it used the symbolic
    ``sin`` and ``cos`` functions from SymPy. In the second, we got a numeric
    result, because ``sin_cos`` used the numeric ``sin`` and ``cos`` functions
    from NumPy. But notice that the versions of ``sin`` and ``cos`` that were
    used was not inherent to the ``sin_cos`` function definition. Both
    ``sin_cos`` definitions are exactly the same. Rather, it was based on the
    names defined at the module where the ``sin_cos`` function was defined.

    The key point here is that when function in Python references a name that
    is not defined in the function, that name is looked up in the "global"
    namespace of the module where that function is defined.

    Now, in Python, we can emulate this behavior without actually writing a
    file to disk using the ``exec`` function. ``exec`` takes a string
    containing a block of Python code, and a dictionary that should contain
    the global variables of the module. It then executes the code "in" that
    dictionary, as if it were the module globals. The following is equivalent
    to the ``sin_cos`` defined in ``sin_cos_sympy.py``:

    >>> import sympy
    >>> module_dictionary = {'sin': sympy.sin, 'cos': sympy.cos}
    >>> exec('''
    ... def sin_cos(x):
    ...     return sin(x) + cos(x)
    ... ''', module_dictionary)
    >>> sin_cos = module_dictionary['sin_cos']
    >>> sin_cos(1)
    cos(1) + sin(1)

    and similarly with ``sin_cos_numpy``:

    >>> import numpy
    >>> module_dictionary = {'sin': numpy.sin, 'cos': numpy.cos}
    >>> exec('''
    ... def sin_cos(x):
    ...     return sin(x) + cos(x)
    ... ''', module_dictionary)
    >>> sin_cos = module_dictionary['sin_cos']
    >>> sin_cos(1)
    1.38177329068

    So now we can get an idea of how ``lambdify`` works. The name "lambdify"
    comes from the fact that we can think of something like ``lambdify(x,
    sin(x) + cos(x), 'numpy')`` as ``lambda x: sin(x) + cos(x)``, where
    ``sin`` and ``cos`` come from the ``numpy`` namespace. This is also why
    the symbols argument is first in ``lambdify``, as opposed to most SymPy
    functions where it comes after the expression: to better mimic the
    ``lambda`` keyword.

    ``lambdify`` takes the input expression (like ``sin(x) + cos(x)``) and

    1. Converts it to a string
    2. Creates a module globals dictionary based on the modules that are
       passed in (by default, it uses the NumPy module)
    3. Creates the string ``"def func({vars}): return {expr}"``, where ``{vars}`` is the
       list of variables separated by commas, and ``{expr}`` is the string
       created in step 1., then ``exec``s that string with the module globals
       namespace and returns ``func``.

    In fact, functions returned by ``lambdify`` support inspection. So you can
    see exactly how they are defined by using ``inspect.getsource``, or ``??`` if you
    are using IPython or the Jupyter notebook.

    >>> f = lambdify(x, sin(x) + cos(x))
    >>> import inspect
    >>> print(inspect.getsource(f))
    def _lambdifygenerated(x):
        return sin(x) + cos(x)

    This shows us the source code of the function, but not the namespace it
    was defined in. We can inspect that by looking at the ``__globals__``
    attribute of ``f``:

    >>> f.__globals__['sin']
    <ufunc 'sin'>
    >>> f.__globals__['cos']
    <ufunc 'cos'>
    >>> f.__globals__['sin'] is numpy.sin
    True

    This shows us that ``sin`` and ``cos`` in the namespace of ``f`` will be
    ``numpy.sin`` and ``numpy.cos``.

    Note that there are some convenience layers in each of these steps, but at
    the core, this is how ``lambdify`` works. Step 1 is done using the
    ``LambdaPrinter`` printers defined in the printing module (see
    :mod:`sympy.printing.lambdarepr`). This allows different SymPy expressions
    to define how they should be converted to a string for different modules.
    You can change which printer ``lambdify`` uses by passing a custom printer
    in to the ``printer`` argument.

    Step 2 is augmented by certain translations. There are default
    translations for each module, but you can provide your own by passing a
    list to the ``modules`` argument. For instance,

    >>> def mysin(x):
    ...     print('taking the sin of', x)
    ...     return numpy.sin(x)
    ...
    >>> f = lambdify(x, sin(x), [{'sin': mysin}, 'numpy'])
    >>> f(1)
    taking the sin of 1
    0.8414709848078965

    The globals dictionary is generated from the list by merging the
    dictionary ``{'sin': mysin}`` and the module dictionary for NumPy. The
    merging is done so that earlier items take precedence, which is why
    ``mysin`` is used above instead of ``numpy.sin``.

    If you want to modify the way ``lambdify`` works for a given function, it
    is usually easiest to do so by modifying the globals dictionary as such.
    In more complicated cases, it may be necessary to create and pass in a
    custom printer.

    Finally, step 3 is augmented with certain convenience operations, such as
    the addition of a docstring.

    Understanding how ``lambdify`` works can make it easier to avoid certain
    gotchas when using it. For instance, a common mistake is to create a
    lambdified function for one module (say, NumPy), and pass it objects from
    another (say, a SymPy expression).

    For instance, say we create

    >>> from sympy.abc import x
    >>> f = lambdify(x, x + 1, 'numpy')

    Now if we pass in a NumPy array, we get that array plus 1

    >>> import numpy
    >>> a = numpy.array([1, 2])
    >>> f(a)
    [2 3]

    But what happens if you make the mistake of passing in a SymPy expression
    instead of a NumPy array:

    >>> f(x + 1)
    x + 2

    This worked, but it was only by accident. Now take a different lambdified
    function:

    >>> from sympy import sin
    >>> g = lambdify(x, x + sin(x), 'numpy')

    This works as expected on NumPy arrays:

    >>> g(a)
    [1.84147098 2.90929743]

    But if we try to pass in a SymPy expression, it fails

    >>> g(x + 1)
    Traceback (most recent call last):
    ...
    TypeError: loop of ufunc does not support argument 0 of type Add which has
               no callable sin method

    Now, let's look at what happened. The reason this fails is that ``g``
    calls ``numpy.sin`` on the input expression, and ``numpy.sin`` does not
    know how to operate on a SymPy object. **As a general rule, NumPy
    functions do not know how to operate on SymPy expressions, and SymPy
    functions do not know how to operate on NumPy arrays. This is why lambdify
    exists: to provide a bridge between SymPy and NumPy.**

    However, why is it that ``f`` did work? That's because ``f`` does not call
    any functions, it only adds 1. So the resulting function that is created,
    ``def _lambdifygenerated(x): return x + 1`` does not depend on the globals
    namespace it is defined in. Thus it works, but only by accident. A future
    version of ``lambdify`` may remove this behavior.

    Be aware that certain implementation details described here may change in
    future versions of SymPy. The API of passing in custom modules and
    printers will not change, but the details of how a lambda function is
    created may change. However, the basic idea will remain the same, and
    understanding it will be helpful to understanding the behavior of
    lambdify.

    **In general: you should create lambdified functions for one module (say,
    NumPy), and only pass it input types that are compatible with that module
    (say, NumPy arrays).** Remember that by default, if the ``module``
    argument is not provided, ``lambdify`` creates functions using the NumPy
    and SciPy namespaces.
    """
    from sympy.core.symbol import Symbol
    from sympy.core.expr import Expr

    # If the user hasn't specified any modules, use what is available.
    if modules is None:
        try:
            _import("scipy")
        except ImportError:
            try:
                _import("numpy")
            except ImportError:
                # Use either numpy (if available) or python.math where possible.
                # XXX: This leads to different behaviour on different systems and
                #      might be the reason for irreproducible errors.
                modules = ["math", "mpmath", "sympy"]
            else:
                modules = ["numpy"]
        else:
            modules = ["numpy", "scipy"]

    # Get the needed namespaces.
    namespaces = []
    # First find any function implementations
    if use_imps:
        namespaces.append(_imp_namespace(expr))
    # Check for dict before iterating
    if isinstance(modules, (dict, str)) or not hasattr(modules, '__iter__'):
        namespaces.append(modules)
    else:
        # consistency check
        if _module_present('numexpr', modules) and len(modules) > 1:
            raise TypeError("numexpr must be the only item in 'modules'")
        namespaces += list(modules)
    # fill namespace with first having highest priority
    namespace = {}
    for m in namespaces[::-1]:
        buf = _get_namespace(m)
        namespace.update(buf)

    if hasattr(expr, "atoms"):
        #Try if you can extract symbols from the expression.
        #Move on if expr.atoms in not implemented.
        syms = expr.atoms(Symbol)
        for term in syms:
            namespace.update({str(term): term})

    if printer is None:
        if _module_present('mpmath', namespaces):
            from sympy.printing.pycode import MpmathPrinter as Printer # type: ignore
        elif _module_present('scipy', namespaces):
            from sympy.printing.numpy import SciPyPrinter as Printer # type: ignore
        elif _module_present('numpy', namespaces):
            from sympy.printing.numpy import NumPyPrinter as Printer # type: ignore
        elif _module_present('cupy', namespaces):
            from sympy.printing.numpy import CuPyPrinter as Printer # type: ignore
        elif _module_present('jax', namespaces):
            from sympy.printing.numpy import JaxPrinter as Printer # type: ignore
        elif _module_present('numexpr', namespaces):
            from sympy.printing.lambdarepr import NumExprPrinter as Printer # type: ignore
        elif _module_present('tensorflow', namespaces):
            from sympy.printing.tensorflow import TensorflowPrinter as Printer # type: ignore
        elif _module_present('torch', namespaces):
            from sympy.printing.pytorch import TorchPrinter as Printer  # type: ignore
        elif _module_present('sympy', namespaces):
            from sympy.printing.pycode import SymPyPrinter as Printer # type: ignore
        elif _module_present('cmath', namespaces):
            from sympy.printing.pycode import CmathPrinter as Printer # type: ignore
        else:
            from sympy.printing.pycode import PythonCodePrinter as Printer # type: ignore
        user_functions = {}
        for m in namespaces[::-1]:
            if isinstance(m, dict):
                for k in m:
                    user_functions[k] = k
        printer = Printer({'fully_qualified_modules': False, 'inline': True,
                           'allow_unknown_functions': True,
                           'user_functions': user_functions})

    if isinstance(args, set):
        sympy_deprecation_warning(
            """
Passing the function arguments to lambdify() as a set is deprecated. This
leads to unpredictable results since sets are unordered. Instead, use a list
or tuple for the function arguments.
            """,
            deprecated_since_version="1.6.3",
            active_deprecations_target="deprecated-lambdify-arguments-set",
                )

    # Get the names of the args, for creating a docstring
    iterable_args = (args,) if isinstance(args, Expr) else args
    names = []

    # Grab the callers frame, for getting the names by inspection (if needed)
    callers_local_vars = inspect.currentframe().f_back.f_locals.items() # type: ignore
    for n, var in enumerate(iterable_args):
        if hasattr(var, 'name'):
            names.append(var.name)
        else:
            # It's an iterable. Try to get name by inspection of calling frame.
            name_list = [var_name for var_name, var_val in callers_local_vars
                    if var_val is var]
            if len(name_list) == 1:
                names.append(name_list[0])
            else:
                # Cannot infer name with certainty. arg_# will have to do.
                names.append('arg_' + str(n))

    # Create the function definition code and execute it
    funcname = '_lambdifygenerated'
    if _module_present('tensorflow', namespaces):
        funcprinter = _TensorflowEvaluatorPrinter(printer, dummify)
    else:
        funcprinter = _EvaluatorPrinter(printer, dummify)

    if cse == True:
        from sympy.simplify.cse_main import cse as _cse
        cses, _expr = _cse(expr, list=False)
    elif callable(cse):
        cses, _expr = cse(expr)
    else:
        cses, _expr = (), expr
    funcstr = funcprinter.doprint(funcname, iterable_args, _expr, cses=cses)

    # Collect the module imports from the code printers.
    imp_mod_lines = []
    for mod, keys in (getattr(printer, 'module_imports', None) or {}).items():
        for k in keys:
            if k not in namespace:
                ln = "from %s import %s" % (mod, k)
                try:
                    exec(ln, {}, namespace)
                except ImportError:
                    # Tensorflow 2.0 has issues with importing a specific
                    # function from its submodule.
                    # https://github.com/tensorflow/tensorflow/issues/33022
                    ln = "%s = %s.%s" % (k, mod, k)
                    exec(ln, {}, namespace)
                imp_mod_lines.append(ln)

    # Provide lambda expression with builtins, and compatible implementation of range
    namespace.update({'builtins':builtins, 'range':range})

    funclocals = {}
    global _lambdify_generated_counter
    filename = '<lambdifygenerated-%s>' % _lambdify_generated_counter
    _lambdify_generated_counter += 1
    c = compile(funcstr, filename, 'exec')
    exec(c, namespace, funclocals)
    # mtime has to be None or else linecache.checkcache will remove it
    linecache.cache[filename] = (len(funcstr), None, funcstr.splitlines(True), filename) # type: ignore

    # Remove the entry from the linecache when the object is garbage collected
    def cleanup_linecache(filename):
        def _cleanup():
            if filename in linecache.cache:
                del linecache.cache[filename]
        return _cleanup

    func = funclocals[funcname]

    weakref.finalize(func, cleanup_linecache(filename))

    # Apply the docstring
    sig = "func({})".format(", ".join(str(i) for i in names))
    sig = textwrap.fill(sig, subsequent_indent=' '*8)
    if _too_large_for_docstring(expr, docstring_limit):
        expr_str = "EXPRESSION REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        src_str = "SOURCE CODE REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
    else:
        expr_str = str(expr)
        if len(expr_str) > 78:
            expr_str = textwrap.wrap(expr_str, 75)[0] + '...'
        src_str = funcstr
    func.__doc__ = (
        "Created with lambdify. Signature:\n\n"
        "{sig}\n\n"
        "Expression:\n\n"
        "{expr}\n\n"
        "Source code:\n\n"
        "{src}\n\n"
        "Imported modules:\n\n"
        "{imp_mods}"
        ).format(sig=sig, expr=expr_str, src=src_str, imp_mods='\n'.join(imp_mod_lines))
    return func

def _module_present(modname, modlist):
    if modname in modlist:
        return True
    for m in modlist:
        if hasattr(m, '__name__') and m.__name__ == modname:
            return True
    return False

def _get_namespace(m):
    """
    This is used by _lambdify to parse its arguments.
    """
    if isinstance(m, str):
        _import(m)
        return MODULES[m][0]
    elif isinstance(m, dict):
        return m
    elif hasattr(m, "__dict__"):
        return m.__dict__
    else:
        raise TypeError("Argument must be either a string, dict or module but it is: %s" % m)


def _recursive_to_string(doprint, arg):
    """Functions in lambdify accept both SymPy types and non-SymPy types such as python
    lists and tuples. This method ensures that we only call the doprint method of the
    printer with SymPy types (so that the printer safely can use SymPy-methods)."""
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.core.basic import Basic

    if isinstance(arg, (Basic, MatrixBase)):
        return doprint(arg)
    elif iterable(arg):
        if isinstance(arg, list):
            left, right = "[", "]"
        elif isinstance(arg, tuple):
            left, right = "(", ",)"
            if not arg:
                return "()"
        else:
            raise NotImplementedError("unhandled type: %s, %s" % (type(arg), arg))
        return left +', '.join(_recursive_to_string(doprint, e) for e in arg) + right
    elif isinstance(arg, str):
        return arg
    else:
        return doprint(arg)


def lambdastr(args, expr, printer=None, dummify=None):
    """
    Returns a string that can be evaluated to a lambda function.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.lambdify import lambdastr
    >>> lambdastr(x, x**2)
    'lambda x: (x**2)'
    >>> lambdastr((x,y,z), [z,y,x])
    'lambda x,y,z: ([z, y, x])'

    Although tuples may not appear as arguments to lambda in Python 3,
    lambdastr will create a lambda function that will unpack the original
    arguments so that nested arguments can be handled:

    >>> lambdastr((x, (y, z)), x + y)
    'lambda _0,_1: (lambda x,y,z: (x + y))(_0,_1[0],_1[1])'
    """
    # Transforming everything to strings.
    from sympy.matrices import DeferredVector
    from sympy.core.basic import Basic
    from sympy.core.function import (Derivative, Function)
    from sympy.core.symbol import (Dummy, Symbol)
    from sympy.core.sympify import sympify

    if printer is not None:
        if inspect.isfunction(printer):
            lambdarepr = printer
        else:
            if inspect.isclass(printer):
                lambdarepr = lambda expr: printer().doprint(expr)
            else:
                lambdarepr = lambda expr: printer.doprint(expr)
    else:
        #XXX: This has to be done here because of circular imports
        from sympy.printing.lambdarepr import lambdarepr

    def sub_args(args, dummies_dict):
        if isinstance(args, str):
            return args
        elif isinstance(args, DeferredVector):
            return str(args)
        elif iterable(args):
            dummies = flatten([sub_args(a, dummies_dict) for a in args])
            return ",".join(str(a) for a in dummies)
        else:
            # replace these with Dummy symbols
            if isinstance(args, (Function, Symbol, Derivative)):
                dummies = Dummy()
                dummies_dict.update({args : dummies})
                return str(dummies)
            else:
                return str(args)

    def sub_expr(expr, dummies_dict):
        expr = sympify(expr)
        # dict/tuple are sympified to Basic
        if isinstance(expr, Basic):
            expr = expr.xreplace(dummies_dict)
        # list is not sympified to Basic
        elif isinstance(expr, list):
            expr = [sub_expr(a, dummies_dict) for a in expr]
        return expr

    # Transform args
    def isiter(l):
        return iterable(l, exclude=(str, DeferredVector, NotIterable))

    def flat_indexes(iterable):
        n = 0

        for el in iterable:
            if isiter(el):
                for ndeep in flat_indexes(el):
                    yield (n,) + ndeep
            else:
                yield (n,)

            n += 1

    if dummify is None:
        dummify = any(isinstance(a, Basic) and
            a.atoms(Function, Derivative) for a in (
            args if isiter(args) else [args]))

    if isiter(args) and any(isiter(i) for i in args):
        dum_args = [str(Dummy(str(i))) for i in range(len(args))]

        indexed_args = ','.join([
            dum_args[ind[0]] + ''.join(["[%s]" % k for k in ind[1:]])
                    for ind in flat_indexes(args)])

        lstr = lambdastr(flatten(args), expr, printer=printer, dummify=dummify)

        return 'lambda %s: (%s)(%s)' % (','.join(dum_args), lstr, indexed_args)

    dummies_dict = {}
    if dummify:
        args = sub_args(args, dummies_dict)
    else:
        if isinstance(args, str):
            pass
        elif iterable(args, exclude=DeferredVector):
            args = ",".join(str(a) for a in args)

    # Transform expr
    if dummify:
        if isinstance(expr, str):
            pass
        else:
            expr = sub_expr(expr, dummies_dict)
    expr = _recursive_to_string(lambdarepr, expr)
    return "lambda %s: (%s)" % (args, expr)

class _EvaluatorPrinter:
    def __init__(self, printer=None, dummify=False):
        self._dummify = dummify

        #XXX: This has to be done here because of circular imports
        from sympy.printing.lambdarepr import LambdaPrinter

        if printer is None:
            printer = LambdaPrinter()

        if inspect.isfunction(printer):
            self._exprrepr = printer
        else:
            if inspect.isclass(printer):
                printer = printer()

            self._exprrepr = printer.doprint

            #if hasattr(printer, '_print_Symbol'):
            #    symbolrepr = printer._print_Symbol

            #if hasattr(printer, '_print_Dummy'):
            #    dummyrepr = printer._print_Dummy

        # Used to print the generated function arguments in a standard way
        self._argrepr = LambdaPrinter().doprint

    def doprint(self, funcname, args, expr, *, cses=()):
        """
        Returns the function definition code as a string.
        """
        from sympy.core.symbol import Dummy

        funcbody = []

        if not iterable(args):
            args = [args]

        if cses:
            cses = list(cses)
            subvars, subexprs = zip(*cses)
            exprs = [expr] + list(subexprs)
            argstrs, exprs = self._preprocess(args, exprs, cses=cses)
            expr, subexprs = exprs[0], exprs[1:]
            cses = zip(subvars, subexprs)
        else:
            argstrs, expr = self._preprocess(args, expr)

        # Generate argument unpacking and final argument list
        funcargs = []
        unpackings = []

        for argstr in argstrs:
            if iterable(argstr):
                funcargs.append(self._argrepr(Dummy()))
                unpackings.extend(self._print_unpacking(argstr, funcargs[-1]))
            else:
                funcargs.append(argstr)

        funcsig = 'def {}({}):'.format(funcname, ', '.join(funcargs))

        # Wrap input arguments before unpacking
        funcbody.extend(self._print_funcargwrapping(funcargs))

        funcbody.extend(unpackings)

        for s, e in cses:
            if e is None:
                funcbody.append('del {}'.format(self._exprrepr(s)))
            else:
                funcbody.append('{} = {}'.format(self._exprrepr(s), self._exprrepr(e)))

        # Subs may appear in expressions generated by .diff()
        subs_assignments = []
        expr = self._handle_Subs(expr, out=subs_assignments)
        for lhs, rhs in subs_assignments:
            funcbody.append('{} = {}'.format(self._exprrepr(lhs), self._exprrepr(rhs)))

        str_expr = _recursive_to_string(self._exprrepr, expr)

        if '\n' in str_expr:
            str_expr = '({})'.format(str_expr)
        funcbody.append('return {}'.format(str_expr))

        funclines = [funcsig]
        funclines.extend(['    ' + line for line in funcbody])

        return '\n'.join(funclines) + '\n'

    @classmethod
    def _is_safe_ident(cls, ident):
        return isinstance(ident, str) and ident.isidentifier() \
                and not keyword.iskeyword(ident)

    def _preprocess(self, args, expr, cses=(), _dummies_dict=None):
        """Preprocess args, expr to replace arguments that do not map
        to valid Python identifiers.

        Returns string form of args, and updated expr.
        """
        from sympy.core.basic import Basic
        from sympy.core.sorting import ordered
        from sympy.core.function import (Derivative, Function)
        from sympy.core.symbol import Dummy, uniquely_named_symbol
        from sympy.matrices import DeferredVector
        from sympy.core.expr import Expr

        # Args of type Dummy can cause name collisions with args
        # of type Symbol.  Force dummify of everything in this
        # situation.
        dummify = self._dummify or any(
            isinstance(arg, Dummy) for arg in flatten(args))

        argstrs = [None]*len(args)
        if _dummies_dict is None:
            _dummies_dict = {}

        def update_dummies(arg, dummy):
            _dummies_dict[arg] = dummy
            for repl, sub in cses:
                arg = arg.xreplace({sub: repl})
                _dummies_dict[arg] = dummy

        for arg, i in reversed(list(ordered(zip(args, range(len(args)))))):
            if iterable(arg):
                s, expr = self._preprocess(arg, expr, cses=cses, _dummies_dict=_dummies_dict)
            elif isinstance(arg, DeferredVector):
                s = str(arg)
            elif isinstance(arg, Basic) and arg.is_symbol:
                s = str(arg)
                if dummify or not self._is_safe_ident(s):
                    dummy = Dummy()
                    if isinstance(expr, Expr):
                        dummy = uniquely_named_symbol(
                            dummy.name, expr, modify=lambda s: '_' + s)
                    s = self._argrepr(dummy)
                    update_dummies(arg, dummy)
                    expr = self._subexpr(expr, _dummies_dict)
            elif dummify or isinstance(arg, (Function, Derivative)):
                dummy = Dummy()
                s = self._argrepr(dummy)
                update_dummies(arg, dummy)
                expr = self._subexpr(expr, _dummies_dict)
            else:
                s = str(arg)
            argstrs[i] = s
        return argstrs, expr

    def _subexpr(self, expr, dummies_dict):
        from sympy.matrices import DeferredVector
        from sympy.core.sympify import sympify

        expr = sympify(expr)
        xreplace = getattr(expr, 'xreplace', None)
        if xreplace is not None:
            expr = xreplace(dummies_dict)
        else:
            if isinstance(expr, DeferredVector):
                pass
            elif isinstance(expr, dict):
                k = [self._subexpr(sympify(a), dummies_dict) for a in expr.keys()]
                v = [self._subexpr(sympify(a), dummies_dict) for a in expr.values()]
                expr = dict(zip(k, v))
            elif isinstance(expr, tuple):
                expr = tuple(self._subexpr(sympify(a), dummies_dict) for a in expr)
            elif isinstance(expr, list):
                expr = [self._subexpr(sympify(a), dummies_dict) for a in expr]
        return expr

    def _print_funcargwrapping(self, args):
        """Generate argument wrapping code.

        args is the argument list of the generated function (strings).

        Return value is a list of lines of code that will be inserted  at
        the beginning of the function definition.
        """
        return []

    def _print_unpacking(self, unpackto, arg):
        """Generate argument unpacking code.

        arg is the function argument to be unpacked (a string), and
        unpackto is a list or nested lists of the variable names (strings) to
        unpack to.
        """
        def unpack_lhs(lvalues):
            return '[{}]'.format(', '.join(
                unpack_lhs(val) if iterable(val) else val for val in lvalues))

        return ['{} = {}'.format(unpack_lhs(unpackto), arg)]

    def _handle_Subs(self, expr, out):
        """Any instance of Subs is extracted and returned as assignment pairs."""
        from sympy.core.basic import Basic
        from sympy.core.function import Subs
        from sympy.core.symbol import Dummy
        from sympy.matrices.matrixbase import MatrixBase

        def _replace(ex, variables, point):
            safe = {}
            for lhs, rhs in zip(variables, point):
                dummy = Dummy()
                safe[lhs] = dummy
                out.append((dummy, rhs))
            return ex.xreplace(safe)

        if isinstance(expr, (Basic, MatrixBase)):
            expr = expr.replace(Subs, _replace)
        elif iterable(expr):
            expr = type(expr)([self._handle_Subs(e, out) for e in expr])
        return expr

class _TensorflowEvaluatorPrinter(_EvaluatorPrinter):
    def _print_unpacking(self, lvalues, rvalue):
        """Generate argument unpacking code.

        This method is used when the input value is not iterable,
        but can be indexed (see issue #14655).
        """

        def flat_indexes(elems):
            n = 0

            for el in elems:
                if iterable(el):
                    for ndeep in flat_indexes(el):
                        yield (n,) + ndeep
                else:
                    yield (n,)

                n += 1

        indexed = ', '.join('{}[{}]'.format(rvalue, ']['.join(map(str, ind)))
                                for ind in flat_indexes(lvalues))

        return ['[{}] = [{}]'.format(', '.join(flatten(lvalues)), indexed)]

def _imp_namespace(expr, namespace=None):
    """ Return namespace dict with function implementations

    We need to search for functions in anything that can be thrown at
    us - that is - anything that could be passed as ``expr``.  Examples
    include SymPy expressions, as well as tuples, lists and dicts that may
    contain SymPy expressions.

    Parameters
    ----------
    expr : object
       Something passed to lambdify, that will generate valid code from
       ``str(expr)``.
    namespace : None or mapping
       Namespace to fill.  None results in new empty dict

    Returns
    -------
    namespace : dict
       dict with keys of implemented function names within ``expr`` and
       corresponding values being the numerical implementation of
       function

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function, _imp_namespace
    >>> from sympy import Function
    >>> f = implemented_function(Function('f'), lambda x: x+1)
    >>> g = implemented_function(Function('g'), lambda x: x*10)
    >>> namespace = _imp_namespace(f(g(x)))
    >>> sorted(namespace.keys())
    ['f', 'g']
    """
    # Delayed import to avoid circular imports
    from sympy.core.function import FunctionClass
    if namespace is None:
        namespace = {}
    # tuples, lists, dicts are valid expressions
    if is_sequence(expr):
        for arg in expr:
            _imp_namespace(arg, namespace)
        return namespace
    elif isinstance(expr, dict):
        for key, val in expr.items():
            # functions can be in dictionary keys
            _imp_namespace(key, namespace)
            _imp_namespace(val, namespace)
        return namespace
    # SymPy expressions may be Functions themselves
    func = getattr(expr, 'func', None)
    if isinstance(func, FunctionClass):
        imp = getattr(func, '_imp_', None)
        if imp is not None:
            name = expr.func.__name__
            if name in namespace and namespace[name] != imp:
                raise ValueError('We found more than one '
                                 'implementation with name '
                                 '"%s"' % name)
            namespace[name] = imp
    # and / or they may take Functions as arguments
    if hasattr(expr, 'args'):
        for arg in expr.args:
            _imp_namespace(arg, namespace)
    return namespace


def implemented_function(symfunc, implementation):
    """ Add numerical ``implementation`` to function ``symfunc``.

    ``symfunc`` can be an ``UndefinedFunction`` instance, or a name string.
    In the latter case we create an ``UndefinedFunction`` instance with that
    name.

    Be aware that this is a quick workaround, not a general method to create
    special symbolic functions. If you want to create a symbolic function to be
    used by all the machinery of SymPy you should subclass the ``Function``
    class.

    Parameters
    ----------
    symfunc : ``str`` or ``UndefinedFunction`` instance
       If ``str``, then create new ``UndefinedFunction`` with this as
       name.  If ``symfunc`` is an Undefined function, create a new function
       with the same name and the implemented function attached.
    implementation : callable
       numerical implementation to be called by ``evalf()`` or ``lambdify``

    Returns
    -------
    afunc : sympy.FunctionClass instance
       function with attached implementation

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import lambdify
    >>> f = implemented_function('f', lambda x: x+1)
    >>> lam_f = lambdify(x, f(x))
    >>> lam_f(4)
    5
    """
    # Delayed import to avoid circular imports
    from sympy.core.function import UndefinedFunction
    # if name, create function to hold implementation
    kwargs = {}
    if isinstance(symfunc, UndefinedFunction):
        kwargs = symfunc._kwargs
        symfunc = symfunc.__name__
    if isinstance(symfunc, str):
        # Keyword arguments to UndefinedFunction are added as attributes to
        # the created class.
        symfunc = UndefinedFunction(
            symfunc, _imp_=staticmethod(implementation), **kwargs)
    elif not isinstance(symfunc, UndefinedFunction):
        raise ValueError(filldedent('''
            symfunc should be either a string or
            an UndefinedFunction instance.'''))
    return symfunc


def _too_large_for_docstring(expr, limit):
    """Decide whether an ``Expr`` is too large to be fully rendered in a
    ``lambdify`` docstring.

    This is a fast alternative to ``count_ops``, which can become prohibitively
    slow for large expressions, because in this instance we only care whether
    ``limit`` is exceeded rather than counting the exact number of nodes in the
    expression.

    Parameters
    ==========
    expr : ``Expr``, (nested) ``list`` of ``Expr``, or ``Matrix``
        The same objects that can be passed to the ``expr`` argument of
        ``lambdify``.
    limit : ``int`` or ``None``
        The threshold above which an expression contains too many nodes to be
        usefully rendered in the docstring. If ``None`` then there is no limit.

    Returns
    =======
    bool
        ``True`` if the number of nodes in the expression exceeds the limit,
        ``False`` otherwise.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.utilities.lambdify import _too_large_for_docstring
    >>> expr = x
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    False
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    Does this split it?

    >>> expr = [x, y, z]
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> expr = [x, [y], z, [[x+y], [x*y*z, [x+y+z]]]]
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    False
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> expr = ((x + y + z)**5).expand()
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 100)
    True
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    >>> from sympy import Matrix
    >>> expr = Matrix([[(x + y + z), ((x + y + z)**2).expand(),
    ...                 ((x + y + z)**3).expand(), ((x + y + z)**4).expand()]])
    >>> _too_large_for_docstring(expr, None)
    False
    >>> _too_large_for_docstring(expr, 1000)
    False
    >>> _too_large_for_docstring(expr, 100)
    True
    >>> _too_large_for_docstring(expr, 1)
    True
    >>> _too_large_for_docstring(expr, 0)
    True
    >>> _too_large_for_docstring(expr, -1)
    True

    """
    # Must be imported here to avoid a circular import error
    from sympy.core.traversal import postorder_traversal

    if limit is None:
        return False

    i = 0
    for _ in postorder_traversal(expr):
        i += 1
        if i > limit:
            return True
    return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_rotary_attention.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from fusion_attention import FusionAttention
from fusion_base import Fusion
from onnx import FunctionProto, NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class FusionRotaryAttention(FusionAttention):
    """
    Fuse Attention subgraph with rotary positional embeddings into one MultiHeadAttention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__(
            model,
            hidden_size,
            num_heads,
            use_multi_head_attention=True,
            search_op_types=[
                "SimplifiedLayerNormalization",
                "SkipSimplifiedLayerNormalization",
                "LayerNormalization",
                "SkipLayerNormalization",
                "Add",
            ],
        )

    def create_mha_node(
        self,
        input: str,
        output: str,
        q_rotary: NodeProto,
        k_rotary: NodeProto,
        v_matmul: NodeProto,
        attn_mask: str = "",
        add_qk: str = "",
        past_k: str = "",
        past_v: str = "",
        present_k: str = "",
        present_v: str = "",
        scale: float | None = None,
    ) -> NodeProto | None:
        assert self.num_heads > 0

        if self.hidden_size > 0 and (self.hidden_size % self.num_heads) != 0:
            logger.debug(
                f"fuse_rotary_attention: input hidden size {self.hidden_size} is not a multiple of num of heads {self.num_heads}"
            )
            return None

        mha_node_name = self.model.create_node_name("MultiHeadAttention")
        mha_inputs = [
            q_rotary.output[0],
            k_rotary.output[0],
            v_matmul.output[0],
            "",  # bias
            attn_mask,  # key_padding_mask
            add_qk,  # attention_bias
            past_k,
            past_v,
        ]

        mha_outputs = [output]
        if present_k and present_v:
            mha_outputs.extend([present_k, present_v])

        mha_node = helper.make_node(
            "MultiHeadAttention",
            inputs=mha_inputs,
            outputs=mha_outputs,
            name=mha_node_name,
        )

        mha_node.domain = "com.microsoft"
        mha_node.attribute.extend([helper.make_attribute("num_heads", self.num_heads)])
        if scale is not None:
            mha_node.attribute.extend([helper.make_attribute("scale", scale)])
        if self.mask_filter_value is not None:
            mha_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        self.increase_counter("MultiHeadAttention")
        return mha_node

    def check_runtime_shape_paths_for_function(
        self,
        reshape_qkv_2,  # Reshape after Transpose
        reshape_qkv_1,  # Reshape before Transpose
        reshape_q_2,  # Reshape after RotaryEmbedding
        reshape_k_2,  # Reshape after RotaryEmbedding
        reshape_v_2,  # Reshape after Transpose
        reshape_v_1,  # Reshape before Transpose
        add_qk,  # Add before Softmax
        root_input,  # Root input to attention subgraph
    ):
        # Check #1: check paths for qkv nodes
        concat_qkv_2_path = self.model.match_parent_path(reshape_qkv_2, ["Concat"], [1])
        concat_qkv_1_path = self.model.match_parent_path(reshape_qkv_1, ["Concat"], [1])
        if concat_qkv_2_path is None or concat_qkv_1_path is None:
            return False
        concat_qkv_2, concat_qkv_1 = concat_qkv_2_path[0], concat_qkv_1_path[0]

        reshape_qkv_2_path_1 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_qkv_2_path_2 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        reshape_qkv_1_path_1 = self.model.match_parent_path(concat_qkv_1, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_qkv_1_path_2 = self.model.match_parent_path(concat_qkv_1, ["Unsqueeze", "Gather", "Shape"], [2, 0, 0])
        if (
            reshape_qkv_2_path_1 is None
            or reshape_qkv_2_path_2 is None
            or reshape_qkv_1_path_1 is None
            or reshape_qkv_1_path_2 is None
        ):
            return False

        _, gather_1, shape_1 = reshape_qkv_2_path_1
        _, gather_2, shape_2 = reshape_qkv_2_path_2

        # Check root_input --> Shape --> Gather connection
        if shape_1.input[0] != root_input or shape_2.input[0] != root_input:
            return False

        # Check Gather --> Unsqueeze --> Concat --> Reshape connection for reshape_qkv_1_path_1 and reshape_qkv_1_path_2
        if reshape_qkv_1_path_1[1].name != gather_1.name or reshape_qkv_1_path_2[1].name != gather_2.name:
            return False

        # Check #2: check paths for v nodes
        concat_v_2_path = self.model.match_parent_path(reshape_v_2, ["Concat"], [1])
        concat_v_1_path = self.model.match_parent_path(reshape_v_1, ["Concat"], [1])
        if concat_v_2_path is None or concat_v_1_path is None:
            return False
        concat_v_2, concat_v_1 = concat_v_2_path[0], concat_v_1_path[0]

        reshape_v_2_path_1 = self.model.match_parent_path(
            concat_v_2, ["Unsqueeze", "Mul", "Gather", "Shape"], [0, 0, 0, 0]
        )
        reshape_v_2_path_2 = self.model.match_parent_path(
            concat_v_2, ["Unsqueeze", "Add", "Gather", "Shape"], [1, 0, 0, 0]
        )
        reshape_v_1_path_1 = self.model.match_parent_path(concat_v_1, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_v_1_path_2 = self.model.match_parent_path(concat_v_1, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if (
            reshape_v_2_path_1 is None
            or reshape_v_2_path_2 is None
            or reshape_v_1_path_1 is None
            or reshape_v_1_path_2 is None
        ):
            return False

        # Check Gather --> Mul --> Unsqueeze --> Concat --> Reshape connection for reshape_v_2_path_1
        # Check Gather --> Add --> Unsqueeze --> Concat --> Reshape connection for reshape_v_2_path_2
        # Check Gather --> Unsqueeze --> Concat --> Reshape connection for reshape_v_1_path_1 and reshape_v_1_path_2
        if (
            reshape_v_2_path_1[2].name != gather_1.name
            or reshape_v_2_path_2[2].name != gather_2.name
            or reshape_v_1_path_1[1].name != gather_1.name
            or reshape_v_1_path_2[1].name != gather_2.name
        ):
            return False

        # Check #3: check paths for k nodes
        concat_k_2_path = self.model.match_parent_path(reshape_k_2, ["Concat"], [1])
        if concat_k_2_path is None:
            return False
        concat_k_2 = concat_k_2_path[0]

        reshape_k_2_path_1 = self.model.match_parent_path(
            concat_k_2, ["Unsqueeze", "Mul", "Gather", "Shape"], [0, 0, 0, 0]
        )
        reshape_k_2_path_2 = self.model.match_parent_path(
            concat_k_2, ["Unsqueeze", "Add", "Gather", "Shape"], [2, 0, 0, 0]
        )
        if reshape_k_2_path_1 is None or reshape_k_2_path_2 is None:
            return False

        # Check Gather --> Mul --> Unsqueeze --> Concat --> Reshape connection for reshape_k_2_path_1
        # Check Gather --> Add --> Unsqueeze --> Concat --> Reshape connection for reshape_k_2_path_2
        if reshape_k_2_path_1[2].name != gather_1.name or reshape_k_2_path_2[2].name != gather_2.name:
            return False

        # Check #4: check paths for q nodes
        concat_q_2_path = self.model.match_parent_path(reshape_q_2, ["Concat"], [1])
        if concat_q_2_path is None:
            return False
        concat_q_2 = concat_q_2_path[0]

        reshape_q_2_path_1 = self.model.match_parent_path(
            concat_q_2, ["Unsqueeze", "Mul", "Gather", "Shape"], [0, 0, 0, 0]
        )
        reshape_q_2_path_2 = self.model.match_parent_path(concat_q_2, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if reshape_q_2_path_1 is None or reshape_q_2_path_2 is None:
            return False

        # Check Gather --> Mul --> Unsqueeze --> Concat --> Reshape connection for reshape_q_2_path_1
        # Check Gather --> Unsqueeze --> Concat --> Reshape connection for reshape_q_2_path_2
        if reshape_q_2_path_1[2].name != gather_1.name or reshape_q_2_path_2[1].name != gather_2.name:
            return False

        # Check #5: check Mul nodes are the same for q, k, v
        mul_q = reshape_q_2_path_1[1]
        mul_k = reshape_k_2_path_1[1]
        mul_v = reshape_v_2_path_1[1]
        gather_1_out = gather_1.output[0]
        if mul_q.input[0] != gather_1_out or mul_k.input[0] != gather_1_out or mul_v.input[0] != gather_1_out:
            return False

        # Check #6: check paths for attention mask nodes
        attn_mask_path_1 = self.model.match_parent_path(add_qk, ["Concat", "Slice", "Slice"], [1, 0, 0])
        attn_mask_path_2 = self.model.match_parent_path(add_qk, ["Cast", "Concat", "Slice", "Slice"], [1, 0, 0, 0])
        if attn_mask_path_1 is not None:
            _, slice_qk_2, slice_qk_1 = attn_mask_path_1
        elif attn_mask_path_2 is not None:
            _, _, slice_qk_2, slice_qk_1 = attn_mask_path_2
        else:
            return False
        # Check first input to Slice #1 is 3D attention mask of shape (B,S,T)
        if slice_qk_1.input[0] not in {"attn_mask", "attention_mask"}:
            return False

        slice_qk_2_path = self.model.match_parent_path(
            slice_qk_2, ["Unsqueeze", "Add", "Gather", "Shape"], [2, 0, 1, 0]
        )
        slice_qk_1_path_1 = self.model.match_parent_path(
            slice_qk_1, ["Unsqueeze", "Add", "Gather", "Shape"], [2, 0, 1, 0]
        )
        slice_qk_1_path_2 = self.model.match_parent_path(slice_qk_1, ["Unsqueeze"], [1])
        if slice_qk_2_path is None or slice_qk_1_path_1 is None or slice_qk_1_path_2 is None:
            return False

        # Check Gather --> Add --> Unsqueeze #3 --> Slice #2 connection for slice_qk_2_path
        # Check Gather --> Add --> Unsqueeze #2 --> Slice #1 connection for slice_qk_1_path_1
        if slice_qk_2_path[1].name != slice_qk_1_path_1[1].name or slice_qk_2_path[2].name != slice_qk_1_path_1[2].name:
            return False

        # Check Unsqueeze #1 --> Slice #1 connection for slice_qk_1_path_2
        # Check if first input to Add and Unsqueeze #1 is position ids
        if slice_qk_1_path_1[1].input[0] != slice_qk_1_path_2[0].input[0]:
            return False

        return True

    def check_runtime_shape_paths_for_nodes(
        self,
        reshape_qkv,  # Final reshape before o_proj MatMul
        reshape_q,  # Reshape before q_proj MatMul
        reshape_k,  # Reshape before k_proj MatMul
        reshape_v,  # Reshape before v_proj MatMul
        root_input,  # Root input to attention subgraph
    ):
        # Check #1: check paths for qkv nodes
        concat_qkv_path = self.model.match_parent_path(reshape_qkv, ["Concat"], [1])
        if concat_qkv_path is None:
            return False
        concat_qkv = concat_qkv_path[0]

        reshape_qkv_path_1 = self.model.match_parent_path(concat_qkv, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_qkv_path_2 = self.model.match_parent_path(concat_qkv, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if reshape_qkv_path_1 is None or reshape_qkv_path_2 is None:
            return False

        _, gather_1, shape_1 = reshape_qkv_path_1
        _, gather_2, shape_2 = reshape_qkv_path_2

        # Check root_input --> Shape --> Gather connection
        if shape_1.input[0] != root_input or shape_2.input[0] != root_input:
            return False

        # Check #2: check paths for v nodes
        concat_v_path = self.model.match_parent_path(reshape_v, ["Concat"], [1])
        if concat_v_path is None:
            return False
        concat_v = concat_v_path[0]

        reshape_v_path_1 = self.model.match_parent_path(concat_v, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_v_path_2 = self.model.match_parent_path(concat_v, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if reshape_v_path_1 is None or reshape_v_path_2 is None:
            return False

        # Check Gather --> Unsqueeze --> Concat --> Reshape connection
        if reshape_v_path_1[1].name != gather_1.name or reshape_v_path_2[1].name != gather_2.name:
            return False

        # Check #3: check paths for k nodes
        concat_k_path = self.model.match_parent_path(reshape_k, ["Concat"], [1])
        if concat_k_path is None:
            return False
        concat_k = concat_k_path[0]

        reshape_k_path_1 = self.model.match_parent_path(concat_k, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_k_path_2 = self.model.match_parent_path(concat_k, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if reshape_k_path_1 is None or reshape_k_path_2 is None:
            return False

        # Check Gather --> Unsqueeze --> Concat --> Reshape connection
        if reshape_k_path_1[1].name != gather_1.name or reshape_k_path_2[1].name != gather_2.name:
            return False

        # Check #4: check paths for q nodes
        concat_q_path = self.model.match_parent_path(reshape_q, ["Concat"], [1])
        if concat_q_path is None:
            return False
        concat_q = concat_q_path[0]

        reshape_q_path_1 = self.model.match_parent_path(concat_q, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_q_path_2 = self.model.match_parent_path(concat_q, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if reshape_q_path_1 is None or reshape_q_path_2 is None:
            return False

        # Check Gather --> Unsqueeze --> Concat --> Reshape connection
        if reshape_q_path_1[1].name != gather_1.name or reshape_q_path_2[1].name != gather_2.name:
            return False

        return True

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        if normalize_node.op_type not in {"SkipSimplifiedLayerNormalization", "SkipLayerNormalization", "Add"}:
            return

        # qkv_nodes_1 is for LLaMA-2 Microsoft
        # qkv_nodes_2 is for LLaMA-2 Hugging Face
        # qkv_nodes_3 is for LLaMA-2 distribute Hugging Face model
        qkv_nodes = None
        qkv_nodes_1 = self.model.match_parent_path(
            normalize_node,
            ["MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            [1, 0, 0, 0, 0],
        )
        qkv_nodes_2 = self.model.match_parent_path(
            normalize_node,
            ["MatMul", "Reshape", "Transpose", "MatMul"],
            [1, 0, 0, 0],
        )
        qkv_nodes_3 = self.model.match_parent_path(
            normalize_node,
            ["AllReduce", "MatMul", "Reshape", "Transpose", "MatMul"],
            [1, 0, 0, 0, 0],
        )
        if qkv_nodes_1 is not None:
            _, reshape_qkv_2, _, reshape_qkv_1, matmul_qkv = qkv_nodes_1
            qkv_nodes = qkv_nodes_1
        elif qkv_nodes_2 is not None:
            _, reshape_qkv, _, matmul_qkv = qkv_nodes_2
            qkv_nodes = qkv_nodes_2
        elif qkv_nodes_3 is not None:
            _, _, reshape_qkv, _, matmul_qkv = qkv_nodes_3
            qkv_nodes = qkv_nodes_3
        else:
            logger.debug("fuse_rotary_attention: failed to match qkv nodes")
            return

        # v_nodes_1 is for LLaMA-2 Microsoft
        # v_nodes_3 is for LLaMA-2 Hugging Face
        # v_nodes_4 is for LLaMA-2 70B model
        # v_nodes_5 is for Phi-2 DirectML
        past_v, present_v, past_seq_len = "", "", ""
        v_nodes = None
        add_v = None
        v_nodes_1 = self.model.match_parent_path(
            matmul_qkv,
            ["Reshape", "Transpose", "Concat", "Transpose", "Reshape", "MatMul"],
            [1, 0, 0, 1, 0, 0],
        )
        v_nodes_2 = self.model.match_parent_path(
            matmul_qkv,
            ["Concat", "Transpose", "Reshape", "MatMul"],
            [1, 1, 0, 0],
        )
        v_nodes_3 = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Reshape", "MatMul"],
            [1, 0, 0],
        )
        _, v_nodes_4, _ = self.model.match_parent_paths_all(
            matmul_qkv,
            [
                (
                    ["Reshape", "Expand", "Unsqueeze", "Concat", "Transpose", "Reshape", "MatMul"],
                    [1, 0, 0, 0, 1, 0, 0],
                ),
                (
                    [
                        "Reshape",
                        "Expand",
                        "Where",
                        "Equal",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                ),
                (
                    [
                        "Reshape",
                        "Expand",
                        "Where",
                        "Equal",
                        "Mul",
                        "ConstantOfShape",
                        "Shape",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                ),
                (
                    [
                        "Reshape",
                        "Expand",
                        "Where",
                        "ConstantOfShape",
                        "Shape",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 1, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0],
                ),
                (
                    [
                        "Reshape",
                        "Expand",
                        "Where",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 1, 2, 0, 4, 0, 0, 0, 1, 0, 0],
                ),
                (
                    ["Reshape", "Concat", "Unsqueeze", "Gather", "Shape", "Concat", "Transpose", "Reshape", "MatMul"],
                    [1, 1, 0, 0, 0, 0, 1, 0, 0],
                ),
                (
                    [
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Mul",
                        "Gather",
                        "Shape",
                        "Concat",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                ),
                (
                    ["Reshape", "Concat", "Unsqueeze", "Gather", "Shape", "Concat", "Transpose", "Reshape", "MatMul"],
                    [1, 1, 2, 0, 0, 0, 1, 0, 0],
                ),
                (
                    ["Reshape", "Concat", "Unsqueeze", "Gather", "Shape", "Concat", "Transpose", "Reshape", "MatMul"],
                    [1, 1, 3, 0, 0, 0, 1, 0, 0],
                ),
            ],
            output_name_to_node=None,
        )
        v_nodes_5 = self.model.match_parent_path(
            matmul_qkv,
            ["Concat", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 1, 0, 0, 1],
        )
        if v_nodes_1 is not None:
            reshape_v_2, _, concat_v, _, reshape_v_1, matmul_v = v_nodes_1
            v_nodes = v_nodes_1

            concat_v_path = self.model.match_parent_path(
                concat_v,
                ["Slice", "Unsqueeze"],
                [0, 2],
            )
            if concat_v_path is None:
                logger.debug("fuse_rotary_attention: failed to match past/present concat in v path")
                return

            past_v = concat_v_path[0].input[0]
            past_seq_len = concat_v_path[-1].input[0]
            present_v = concat_v.output[0]
        elif v_nodes_2 is not None:
            concat_v, transpose_v, reshape_v, matmul_v = v_nodes_2
            v_nodes = v_nodes_2
            past_v = concat_v.input[0]
            present_v = concat_v.output[0]
        elif v_nodes_3 is not None:
            transpose_v, reshape_v, matmul_v = v_nodes_3
            v_nodes = v_nodes_3
            present_v = transpose_v.output[0]
        elif v_nodes_4 is not None and len(v_nodes_4) == 9:
            concat_v, transpose_v, reshape_v, matmul_v = v_nodes_4[0][-4:]
            v_nodes = v_nodes_4
            past_v = concat_v.input[0]
            present_v = concat_v.output[0]
        elif v_nodes_5 is not None:
            concat_v, transpose_v, reshape_v, add_v, matmul_v = v_nodes_5
            matmul_v = add_v
            v_nodes = v_nodes_5
            past_v = concat_v.input[0]
            present_v = concat_v.output[0]
        else:
            logger.debug("fuse_rotary_attention: failed to match v path")
            return

        qk_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Softmax", "Add", "Div", "MatMul"],
            [0, 0, 0, 0],
        )
        add_qk, matmul_qk = None, None
        if qk_nodes is not None:
            _, add_qk, _, matmul_qk = qk_nodes
        else:
            logger.debug("fuse_rotary_attention: failed to match qk nodes")
            return

        # attn_mask_nodes_1, attn_mask_nodes_2 are for LLaMA-2 Microsoft's 3D attention mask
        # attn_mask_nodes_3, attn_mask_nodes_4 are for LLaMA-2 Hugging Face's 2D attention mask
        # attn_mask_nodes_5, attn_mask_nodes_6 are for LLaMA-2 Microsoft's model for the DML EP
        # attn_mask_nodes_7 is for LLaMA-2 Hugging Face's changes to the attention mask
        attn_mask, add_qk_str = "", ""
        attn_mask_nodes_1 = self.model.match_parent_path(
            add_qk,
            ["Concat", "Slice", "Slice"],
            [1, 0, 0],
        )
        attn_mask_nodes_2 = self.model.match_parent_path(
            add_qk,
            ["Cast", "Concat", "Slice", "Slice"],
            [1, 0, 0, 0],
        )
        attn_mask_nodes_3 = self.model.match_parent_path(
            add_qk,
            ["Add", "Where", "Sub", "Cast", "Expand", "Unsqueeze", "Unsqueeze"],
            [1, 0, 2, 1, 0, 0, 0],
        )
        attn_mask_nodes_4 = self.model.match_parent_path(
            add_qk,
            ["Where", "Sub", "Cast", "Expand", "Unsqueeze", "Unsqueeze"],
            [1, 2, 1, 0, 0, 0],
        )
        attn_mask_nodes_5 = self.model.match_parent_path(
            add_qk,
            ["Expand", "Add", "Where", "Sub", "Cast", "Expand", "Unsqueeze", "Unsqueeze"],
            [1, 0, 0, 2, 1, 0, 0, 0],
        )
        attn_mask_nodes_6 = self.model.match_parent_path(
            add_qk,
            ["Expand", "Where", "Sub", "Cast", "Expand", "Unsqueeze", "Unsqueeze"],
            [1, 0, 2, 1, 0, 0, 0],
        )
        attn_mask_nodes_7 = self.model.match_parent_path(
            add_qk,
            ["Where", "Cast", "Where", "Cast", "Sub", "Cast", "Expand", "Unsqueeze", "Unsqueeze"],
            [1, 0, 0, 0, 0, 1, 0, 0, 0],
        )
        if attn_mask_nodes_1 is not None:
            _, slice_mask_1, slice_mask_2 = attn_mask_nodes_1
            attn_mask = slice_mask_1.output[0]
        elif attn_mask_nodes_2 is not None:
            _, _, slice_mask_1, slice_mask_2 = attn_mask_nodes_2
            attn_mask = slice_mask_1.output[0]
        elif attn_mask_nodes_3 is not None:
            # Reshape from (B,1,S,T) to (B,N,S,T)
            add_qk_str = self.reshape_add_qk(attn_mask_nodes_3[0].output[0])
        elif attn_mask_nodes_4 is not None:
            # Reshape from (B,1,S,T) to (B,N,S,T)
            add_qk_str = self.reshape_add_qk(attn_mask_nodes_4[0].output[0])
        elif attn_mask_nodes_5 is not None:
            # The mask has already been reshaped to (B,N,S,T)
            add_qk_str = attn_mask_nodes_5[0].output[0]
        elif attn_mask_nodes_6 is not None:
            # The mask has already been reshaped to (B,N,S,T)
            add_qk_str = attn_mask_nodes_6[0].output[0]
        elif attn_mask_nodes_7 is not None:
            # Reshape from (B,1,S,T) to (B,N,S,T)
            add_qk_str = self.reshape_add_qk(attn_mask_nodes_7[0].output[0])
        else:
            logger.debug("fuse_rotary_attention: failed to match attention mask nodes")
            return

        # k_nodes_1 is for LLaMA-2 Microsoft
        # k_nodes_2 is for LLaMA-2 Hugging Face
        # k_nodes_4 is for LLaMA-2 70B Hugging Face
        past_k, present_k = "", ""
        k_nodes = None
        slice_k = None
        concat_k_half = None
        k_nodes_1 = self.model.match_parent_path(
            matmul_qk,
            ["Reshape", "Transpose", "Concat", "Transpose", "RotaryEmbedding", "MatMul"],
            [1, 0, 0, 1, 0, 0],
        )
        k_nodes_2 = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "RotaryEmbedding", "Transpose", "Reshape", "MatMul"],
            [1, 0, 0, 0, 0],
        )
        k_nodes_3 = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Concat", "RotaryEmbedding", "Transpose", "Reshape", "MatMul"],
            [1, 0, 1, 0, 0, 0],
        )
        _, k_nodes_4, _ = self.model.match_parent_paths_all(
            matmul_qk,
            [
                (
                    [
                        "Transpose",
                        "Reshape",
                        "Expand",
                        "Unsqueeze",
                        "Concat",
                        "RotaryEmbedding",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0],
                ),
                (
                    [
                        "Transpose",
                        "Reshape",
                        "Expand",
                        "Where",
                        "Equal",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "RotaryEmbedding",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                ),
                (
                    [
                        "Transpose",
                        "Reshape",
                        "Expand",
                        "Where",
                        "Equal",
                        "Mul",
                        "ConstantOfShape",
                        "Shape",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "RotaryEmbedding",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                ),
                (
                    [
                        "Transpose",
                        "Reshape",
                        "Expand",
                        "Where",
                        "ConstantOfShape",
                        "Shape",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "RotaryEmbedding",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 0, 1, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0],
                ),
                (
                    [
                        "Transpose",
                        "Reshape",
                        "Expand",
                        "Where",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "RotaryEmbedding",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 0, 1, 2, 0, 4, 0, 0, 0, 1, 0, 0, 0],
                ),
                (
                    [
                        "Transpose",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "RotaryEmbedding",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                ),
                (
                    [
                        "Transpose",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Mul",
                        "Gather",
                        "Shape",
                        "Concat",
                        "RotaryEmbedding",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                ),
                (
                    [
                        "Transpose",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "RotaryEmbedding",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0],
                ),
                (
                    [
                        "Transpose",
                        "Reshape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                        "Concat",
                        "RotaryEmbedding",
                        "Transpose",
                        "Reshape",
                        "MatMul",
                    ],
                    [1, 0, 1, 3, 0, 0, 0, 1, 0, 0, 0],
                ),
            ],
            output_name_to_node=None,
        )
        k_nodes_5 = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Concat", "Concat", "RotaryEmbedding", "Slice", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 1, 0, 0, 0, 0, 0, 1],
        )
        if k_nodes_1 is not None:
            reshape_k_2, _, concat_k, _, rotary_k, matmul_k = k_nodes_1
            k_nodes = k_nodes_1

            concat_k_path = self.model.match_parent_path(
                concat_k,
                ["Slice", "Unsqueeze"],
                [0, 2],
            )
            if concat_k_path is None:
                logger.debug("fuse_rotary_attention: failed to match past/present concat in k path")
                return

            past_k = concat_k_path[0].input[0]
            shared_past_seq_len = concat_k_path[-1].input[0]
            present_k = concat_k.output[0]

            assert past_seq_len == shared_past_seq_len
        elif k_nodes_2 is not None:
            _, rotary_k, _, reshape_k, matmul_k = k_nodes_2
            k_nodes = k_nodes_2
            present_k = rotary_k.output[0]
        elif k_nodes_3 is not None:
            _, concat_k, rotary_k, _, reshape_k, matmul_k = k_nodes_3
            k_nodes = k_nodes_3
            past_k = concat_k.input[0]
            present_k = concat_k.output[0]
        elif k_nodes_4 is not None and len(k_nodes_4) == 9:
            reshape_k, matmul_k = k_nodes_4[0][-2:]
            concat_k, rotary_k = k_nodes_4[0][-5:-3]
            k_nodes = k_nodes_4
            past_k = concat_k.input[0]
            present_k = concat_k.output[0]
        elif k_nodes_5 is not None:
            _, concat_k, concat_k_half, rotary_k, slice_k, _, reshape_k, _, matmul_k = k_nodes_5
            k_nodes = k_nodes_5
            past_k = concat_k.input[0]
            present_k = concat_k.output[0]
        else:
            logger.debug("fuse_rotary_attention: failed to match k nodes")
            return

        # q_nodes_1 is for LLaMA-2 Microsoft
        # q_nodes_2 is for LLaMA-2 Hugging Face
        # q_nodes_3 is for Phi-2 DirectML
        q_nodes = None
        slice_q = None
        concat_q_half = None
        q_nodes_1 = self.model.match_parent_path(
            matmul_qk,
            ["Reshape", "Transpose", "RotaryEmbedding", "MatMul"],
            [0, 0, 0, 0],
        )
        q_nodes_2 = self.model.match_parent_path(
            matmul_qk,
            ["RotaryEmbedding", "Transpose", "Reshape", "MatMul"],
            [0, 0, 0, 0],
        )
        q_nodes_3 = self.model.match_parent_path(
            matmul_qk,
            ["Concat", "RotaryEmbedding", "Slice", "Transpose", "Reshape", "Add", "MatMul"],
            [0, 0, 0, 0, 0, 0, 1],
        )
        if q_nodes_1 is not None:
            reshape_q_2, _, rotary_q, matmul_q = q_nodes_1
            q_nodes = q_nodes_1
        elif q_nodes_2 is not None:
            rotary_q, _, reshape_q, matmul_q = q_nodes_2
            q_nodes = q_nodes_2
        elif q_nodes_3 is not None:
            concat_q_half, rotary_q, slice_q, _, reshape_q, _, matmul_q = q_nodes_3
            q_nodes = q_nodes_3
        else:
            logger.debug("fuse_rotary_attention: failed to match q nodes")
            return

        if matmul_q.input[0] != matmul_k.input[0] and matmul_k.input[0] != matmul_v.input[0]:
            logger.debug("fuse_rotary_attention: failed to find the same root_input for q, k, v paths")
            return

        root_output = ""
        if qkv_nodes == qkv_nodes_1:
            if not self.check_runtime_shape_paths_for_function(
                reshape_qkv_2,
                reshape_qkv_1,
                reshape_q_2,
                reshape_k_2,
                reshape_v_2,
                reshape_v_1,
                add_qk,
                matmul_q.input[0],
            ):
                logger.debug("fuse_rotary_attention: failed to verify runtime shape paths")
                return
            root_output = reshape_qkv_2.output[0]

        elif qkv_nodes in (qkv_nodes_2, qkv_nodes_3):
            if not self.check_runtime_shape_paths_for_nodes(
                reshape_qkv,
                reshape_q,
                reshape_k,
                reshape_v,
                matmul_q.input[0],
            ):
                logger.debug("fuse_rotary_attention: failed to verify runtime shape paths")
                return
            root_output = reshape_qkv.output[0]

            # Rename inputs of rotary_q/k so it connects with output of matmul_q/k
            # Before: MatMul --> Reshape --> Transpose --> RotaryEmbedding
            # After: MatMul --> RotaryEmbedding
            rotary_q.input[0] = slice_q.output[0] if slice_q else matmul_q.output[0]
            rotary_k.input[0] = slice_k.output[0] if slice_k else matmul_k.output[0]

            # Rename current output of rotary_k (present_key) so it doesn't match output of MHA (present_key)
            if concat_q_half is None:
                rotary_k.output[0] = rotary_k.name + "_output_0"

            if qkv_nodes == qkv_nodes_3:
                qkv_nodes = qkv_nodes[1:]

        def create_hidden_size_concat_node(reshape_q):
            """Detect num_heads and hidden_size for ONNX model from phi-2
            Args:
                reshape_q (NodeProto): reshape node for q
            Returns:
                hidden_size_concat_node(NodeProto): Concat node to be used by reshape
            """
            concat = self.model.match_parent(reshape_q, "Concat", 1)

            if concat is None:
                logger.debug("fuse_rotary_attention: failed to trace the concat node from reshape_q")
                return None

            # The shape is a tensor like [?, ?, num_heads, head_size]
            num_head_constant_node = self.model.get_constant_value(concat.input[2])
            head_size_constant_node = self.model.get_constant_value(concat.input[3])

            if num_head_constant_node is None or head_size_constant_node is None:
                logger.debug("fuse_rotary_attention: failed to get constant nodes of num_heads or head_size")
                return None

            num_head_value = num_head_constant_node[0]
            head_size_value = head_size_constant_node[0]

            hidden_size = num_head_value * head_size_value

            hidden_size_initilizer = self.model.create_node_name("Initializer", name_prefix="hidden_size")
            if self.model.get_initializer(hidden_size_initilizer) is None:
                self.add_initializer(
                    name=hidden_size_initilizer,
                    data_type=TensorProto.INT64,
                    dims=[1],
                    vals=[hidden_size],
                    raw=False,
                )

            hidden_size_reshape_node_name = self.model.create_node_name("Concat", name_prefix="hidden_size_concat")

            hidden_size_concat_node = helper.make_node(
                "Concat",
                inputs=[
                    concat.input[0],
                    concat.input[1],
                    hidden_size_initilizer,
                ],
                outputs=[hidden_size_reshape_node_name + "output_0"],
                name=hidden_size_reshape_node_name,
            )
            hidden_size_concat_node.attribute.extend([helper.make_attribute("axis", 0)])

            return hidden_size_concat_node

        # Add Tranpose and Reshape nodes for patial rotary embedding applied in phi-2 before passing into MHA
        if concat_q_half and concat_k_half:
            # Transpose the key output of rotary Embedding
            k_transpose_node_name = self.model.create_node_name("Transpose")
            k_tranpose_output_name = k_transpose_node_name + "_output_0"
            k_transpose_node = helper.make_node(
                "Transpose",
                inputs=[concat_k_half.output[0]],
                outputs=[k_tranpose_output_name],
                name=k_transpose_node_name,
            )

            k_transpose_node.attribute.extend([helper.make_attribute("perm", [0, 2, 1, 3])])

            # Transpose the query output of rotary Embedding
            q_transpose_node_name = self.model.create_node_name("Transpose")
            q_tranpose_output_name = q_transpose_node_name + "_output_0"
            q_transpose_node = helper.make_node(
                "Transpose",
                inputs=[concat_q_half.output[0]],
                outputs=[q_tranpose_output_name],
                name=q_transpose_node_name,
            )

            q_transpose_node.attribute.extend([helper.make_attribute("perm", [0, 2, 1, 3])])

            hidden_size_concat_node = create_hidden_size_concat_node(reshape_k)
            if hidden_size_concat_node is None:
                logger.debug("fuse_rotary_attention: failed to create hidden_size_concat_node")
                return

            # Reshape the Rotary Embedding output for key for 4D to 3D
            concat_k_reshape_node_name = self.model.create_node_name("Reshape", name_prefix="concat_k_half")
            concat_k_reshape_node = helper.make_node(
                "Reshape",
                inputs=[k_transpose_node.output[0], hidden_size_concat_node.output[0]],
                outputs=[concat_k_reshape_node_name + "_output_0"],
                name=concat_k_reshape_node_name,
            )

            # Reshape the Rotary Embedding output for query from 4D to 3D
            concat_q_reshape_node_name = self.model.create_node_name("Reshape", name_prefix="concat_q_half")
            concat_q_reshape_node = helper.make_node(
                "Reshape",
                inputs=[q_transpose_node.output[0], hidden_size_concat_node.output[0]],
                outputs=[concat_q_reshape_node_name + "_output_0"],
                name=concat_q_reshape_node_name,
            )

            rotary_k = concat_k_reshape_node
            rotary_q = concat_q_reshape_node

            self.nodes_to_add.append(hidden_size_concat_node)
            self.nodes_to_add.append(k_transpose_node)
            self.nodes_to_add.append(q_transpose_node)
            self.nodes_to_add.append(concat_k_reshape_node)
            self.nodes_to_add.append(concat_q_reshape_node)

            self.node_name_to_graph_name[hidden_size_concat_node.name] = self.this_graph_name
            self.node_name_to_graph_name[k_transpose_node.name] = self.this_graph_name
            self.node_name_to_graph_name[q_transpose_node.name] = self.this_graph_name
            self.node_name_to_graph_name[concat_k_reshape_node.name] = self.this_graph_name
            self.node_name_to_graph_name[concat_q_reshape_node.name] = self.this_graph_name

        new_node = self.create_mha_node(
            matmul_q.input[0],
            root_output,
            rotary_q,
            rotary_k,
            matmul_v,
            attn_mask,
            add_qk_str,
            past_k,
            past_v,
            present_k,
            present_v,
        )
        if new_node is None:
            logger.debug("fuse_rotary_attention: failed to create multi-head attention with rotary embeddings")
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend(qkv_nodes[1:])

        if v_nodes != v_nodes_4:
            self.nodes_to_remove.extend(v_nodes[:-1] if add_v is None else v_nodes[:-2])
        else:
            nodes_to_keep = [v_nodes[0][-1]]
            for temp_path in v_nodes:
                self.add_nodes_to_remove_with_nodes_to_keep(temp_path, nodes_to_keep)

        self.nodes_to_remove.extend(qk_nodes)

        if k_nodes == k_nodes_1:
            self.nodes_to_remove.extend(k_nodes[:-2])
        elif k_nodes == k_nodes_2:
            self.nodes_to_remove.append(k_nodes[0])
            self.nodes_to_remove.append(k_nodes[2])
            self.nodes_to_remove.append(k_nodes[3])
        elif k_nodes == k_nodes_3:
            self.nodes_to_remove.append(k_nodes[0])
            self.nodes_to_remove.append(k_nodes[1])
            self.nodes_to_remove.append(k_nodes[3])
            self.nodes_to_remove.append(k_nodes[4])
        elif k_nodes == k_nodes_5:
            self.nodes_to_remove.append(k_nodes[0])
            self.nodes_to_remove.append(k_nodes[1])
        elif k_nodes == k_nodes_4:
            nodes_to_keep = [k_nodes[0][-1], k_nodes[0][-4]]
            for temp_path in k_nodes:
                self.add_nodes_to_remove_with_nodes_to_keep(temp_path, nodes_to_keep)

        if q_nodes == q_nodes_1:
            self.nodes_to_remove.extend(q_nodes[:-2])
        elif q_nodes == q_nodes_2:
            self.nodes_to_remove.append(q_nodes[1])
            self.nodes_to_remove.append(q_nodes[2])
        self.prune_graph = True


class FusionRotaryEmbeddings(Fusion):
    def __init__(self, model: OnnxModel):
        self.base_name = "RotaryEmbedding"
        super().__init__(model, self.base_name, [self.base_name, self.base_name + ".1", "Add"])

    # The RotaryEmbedding function can have multiple extraneous constant outputs even though the function is supposed to produce only one output.
    # This is a byproduct of a potential CSE bug when using `export_modules_as_functions` in the TorchScript exporter.
    # To work around this issue, we set the extraneous constant values from the RotaryEmbedding function as initializers in the locations where they are actually used.
    def reassign_extra_outputs(self, rot_emb_node: NodeProto, function: FunctionProto):
        # Find extra outputs and Constant nodes attached to those outputs
        extra_constants, extra_outputs = [], []
        for fn_node in function.node:
            if fn_node.op_type == "Constant" and fn_node.input == [] and fn_node.output[0] in function.output:
                extra_constants.append(fn_node)
                output_index = list(function.output).index(fn_node.output[0])
                extra_outputs.append(rot_emb_node.output[output_index])

        # Set extra Constant node outputs as initializers
        extra_initializers = []
        for extra_constant in extra_constants:
            constant_tensorproto = extra_constant.attribute[0].t
            constant_tensorproto.name = self.model.create_node_name("Constant")
            self.model.add_initializer(constant_tensorproto)
            extra_initializers.append(constant_tensorproto.name)

        # Update references of Constant node outputs to initializer references
        for extra_output, extra_initializer in zip(extra_outputs, extra_initializers, strict=False):
            nodes_to_update = list(filter(lambda entry: extra_output in entry.input, self.model.model.graph.node))
            for node_to_update in nodes_to_update:
                OnnxModel.replace_node_input(node_to_update, extra_output, extra_initializer)

        return extra_outputs

    def create_rotary_embeddings_from_function(self, node: NodeProto):
        rotary_emb_node_name = self.model.create_node_name(self.base_name)

        matmul_path = self.model.match_parent_path(
            node,
            ["Reshape", "MatMul"],
            [0, 0],
        )
        if matmul_path is not None:
            reshape_node, matmul_node = matmul_path
        else:
            logger.debug("fuse_rotary_embeddings: failed to match MatMul")
            return

        rotary_emb_inputs = [
            matmul_node.output[0],  # x is of shape (B,S,D) instead of (B,S,N,H)
            node.input[1],  # position_ids
        ]

        # Convert cos_cache and sin_cache from node attributes to model initializers
        cos_cache_node = list(filter(lambda constant: constant.output[0] == node.input[2], self.model.model.graph.node))
        sin_cache_node = list(filter(lambda constant: constant.output[0] == node.input[3], self.model.model.graph.node))
        cos_cache_name, sin_cache_name = "cos_cache", "sin_cache"

        if (
            len(cos_cache_node) == 1
            and len(sin_cache_node) == 1
            and self.model.get_initializer(cos_cache_name) is None
            and self.model.get_initializer(sin_cache_name) is None
        ):
            cos_cache = numpy_helper.to_array(cos_cache_node[0].attribute[0].t).squeeze()
            sin_cache = numpy_helper.to_array(sin_cache_node[0].attribute[0].t).squeeze()

            cos_cache_tensor = helper.make_tensor(
                name=cos_cache_name,
                data_type=TensorProto.FLOAT,
                dims=list(cos_cache.shape),
                vals=cos_cache.flatten().tolist(),
            )
            self.model.add_initializer(cos_cache_tensor, self.this_graph_name)
            sin_cache_tensor = helper.make_tensor(
                name=sin_cache_name,
                data_type=TensorProto.FLOAT,
                dims=list(sin_cache.shape),
                vals=sin_cache.flatten().tolist(),
            )
            self.model.add_initializer(sin_cache_tensor, self.this_graph_name)

            self.nodes_to_remove.extend([cos_cache_node[0], sin_cache_node[0]])

        rotary_emb_inputs.extend([cos_cache_name, sin_cache_name])

        rotary_emb_outputs = node.output
        if len(rotary_emb_outputs) > 1:
            # Re-assign extraneous constant outputs in RotaryEmbedding functions as initializers
            func = list(filter(lambda fn: fn.name == node.op_type, self.model.model.functions))
            assert len(func) == 1
            extra_outputs = self.reassign_extra_outputs(node, func[0])
            rotary_emb_outputs = list(filter(lambda output_name: output_name not in extra_outputs, rotary_emb_outputs))
            assert len(rotary_emb_outputs) == 1

        rotary_emb_node = helper.make_node(
            self.base_name,
            inputs=rotary_emb_inputs,
            outputs=rotary_emb_outputs,
            name=rotary_emb_node_name,
            interleaved=1,
        )
        rotary_emb_node.domain = "com.microsoft"

        self.nodes_to_remove.append(reshape_node)

        return rotary_emb_node

    def create_rotary_embeddings_from_nodes(
        self,
        root_input: str,
        position_ids: str,
        cos_slice: str,
        sin_slice: str,
        output: str,
    ):
        rotary_emb_node_name = self.model.create_node_name(self.base_name)

        # Convert cos_cache and sin_cache from node attributes to model initializers
        cos_cache_node = list(filter(lambda constant: constant.output[0] == cos_slice, self.model.model.graph.node))
        sin_cache_node = list(filter(lambda constant: constant.output[0] == sin_slice, self.model.model.graph.node))
        cos_cache_name, sin_cache_name = "cos_cache", "sin_cache"

        if (
            len(cos_cache_node) == 1
            and len(sin_cache_node) == 1
            and self.model.get_initializer(cos_cache_name) is None
            and self.model.get_initializer(sin_cache_name) is None
        ):
            cos_cache = numpy_helper.to_array(cos_cache_node[0].attribute[0].t).squeeze()
            sin_cache = numpy_helper.to_array(sin_cache_node[0].attribute[0].t).squeeze()

            # Reshape cos/sin cache from (M, H) to (M, H/2)
            head_size = cos_cache.shape[1]
            cos_cache = cos_cache[:, : (head_size // 2)]
            sin_cache = sin_cache[:, : (head_size // 2)]

            cos_cache_tensor = helper.make_tensor(
                name=cos_cache_name,
                data_type=TensorProto.FLOAT,
                dims=list(cos_cache.shape),
                vals=cos_cache.flatten().tolist(),
            )
            self.model.add_initializer(cos_cache_tensor, self.this_graph_name)
            sin_cache_tensor = helper.make_tensor(
                name=sin_cache_name,
                data_type=TensorProto.FLOAT,
                dims=list(sin_cache.shape),
                vals=sin_cache.flatten().tolist(),
            )
            self.model.add_initializer(sin_cache_tensor, self.this_graph_name)

            self.nodes_to_remove.extend([cos_cache_node[0], sin_cache_node[0]])

        rotary_emb_node = helper.make_node(
            self.base_name,
            inputs=[root_input, position_ids, cos_cache_name, sin_cache_name],
            outputs=[output],
            name=rotary_emb_node_name,
            interleaved=0,
        )
        rotary_emb_node.domain = "com.microsoft"
        return rotary_emb_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # Node is either RotaryEmbedding function or Add
        if self.base_name not in node.op_type and node.op_type != "Add":
            return

        # Check if node is "RotaryEmbedding nn.Module" exported as a function
        # (e.g. export_modules_as_functions={RotaryEmbedding} in torch.onnx.export)
        rotary_emb_node = None
        if node.op_type != "Add":
            # Verify that function has the correct inputs
            if len(node.input) not in {4, 5} or node.input[1] not in {
                "pos",
                "pos_id",
                "position_id",
                "pos_ids",
                "position_ids",
            }:
                logger.debug("fuse_rotary_embeddings: failed to verify inputs for RotaryEmbedding function")
                return

            rotary_emb_node = self.create_rotary_embeddings_from_function(node)
            if rotary_emb_node is None:
                logger.debug("fuse_rotary_embeddings: failed to create RotaryEmbedding node")
                return

            # Remove RotaryEmbedding function
            self.nodes_to_remove.append(node)

            # Remove RotaryEmbedding function's shape inference stored in value_info
            # The new shape will be calculated during symbolic shape inference
            old_shape_infer = list(
                filter(lambda node: node.name == rotary_emb_node.output[0], self.model.model.graph.value_info)
            )
            assert len(old_shape_infer) == 1
            self.model.model.graph.value_info.remove(old_shape_infer[0])

        else:
            # Rotary embeddings are defined using the below functions:
            #
            # def rotate_half(x):
            #     """Rotates half the hidden dims of the input."""
            #     x1 = x[..., : x.shape[-1] // 2]
            #     x2 = x[..., x.shape[-1] // 2 :]
            #     return torch.cat((-x2, x1), dim=-1)
            #
            # def apply_rope(x, cos, sin, position_ids):
            #     cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            #     sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            #     cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            #     sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            #     x_embed = (x * cos) + (rotate_half(x) * sin)
            #     return x_embed

            # Check paths for rotate_half(x)
            rotate_half_x2_path_1_1 = self.model.match_parent_path(
                node,
                ["Mul", "Concat", "Neg", "Slice", "Transpose"],
                [1, 0, 0, 0, 0],
            )

            rotate_half_x2_path_1_2 = self.model.match_parent_path(
                node,
                ["Mul", "Concat", "Neg", "Slice", "Slice"],
                [1, 0, 0, 0, 0],
            )

            rotate_half_x2_path_1 = rotate_half_x2_path_1_1 or rotate_half_x2_path_1_2

            rotate_half_x2_path_2_1 = self.model.match_parent_path(
                node,
                ["Mul", "Concat", "Neg", "Slice", "Unsqueeze", "Div", "Gather", "Shape", "Transpose"],
                [1, 0, 0, 0, 1, 0, 0, 0, 0],
            )

            rotate_half_x2_path_2_2 = self.model.match_parent_path(
                node,
                ["Mul", "Concat", "Neg", "Slice", "Unsqueeze", "Div", "Gather", "Shape", "Slice"],
                [1, 0, 0, 0, 1, 0, 0, 0, 0],
            )

            rotate_half_x2_path_2 = rotate_half_x2_path_2_1 or rotate_half_x2_path_2_2

            if rotate_half_x2_path_1 is None or rotate_half_x2_path_2 is None:
                logger.debug("fuse_rotary_embeddings: failed to match x2 in rotate_half")
                return

            rotate_half_x1_path_1_1 = self.model.match_parent_path(
                node,
                ["Mul", "Concat", "Slice", "Transpose"],
                [1, 0, 1, 0],
            )

            rotate_half_x1_path_1_2 = self.model.match_parent_path(
                node,
                ["Mul", "Concat", "Slice", "Slice"],
                [1, 0, 1, 0],
            )

            rotate_half_x1_path_1 = rotate_half_x1_path_1_1 or rotate_half_x1_path_1_2

            rotate_half_x1_path_2_1 = self.model.match_parent_path(
                node,
                ["Mul", "Concat", "Slice", "Unsqueeze", "Div", "Gather", "Shape", "Transpose"],
                [1, 0, 1, 2, 0, 0, 0, 0],
            )

            rotate_half_x1_path_2_2 = self.model.match_parent_path(
                node,
                ["Mul", "Concat", "Slice", "Unsqueeze", "Div", "Gather", "Shape", "Slice"],
                [1, 0, 1, 2, 0, 0, 0, 0],
            )

            rotate_half_x1_path_2 = rotate_half_x1_path_2_1 or rotate_half_x1_path_2_2

            if rotate_half_x1_path_1 is None or rotate_half_x1_path_2 is None:
                logger.debug("fuse_rotary_embeddings: failed to match x1 in rotate_half")
                return

            if (
                rotate_half_x1_path_1[-1].name != rotate_half_x1_path_2[-1].name
                or rotate_half_x2_path_1[-1].name != rotate_half_x2_path_2[-1].name
                or rotate_half_x1_path_1[-1].name != rotate_half_x2_path_1[-1].name
                or rotate_half_x1_path_2[-1].name != rotate_half_x2_path_2[-1].name
            ):
                logger.debug("fuse_rotary_embeddings: failed to match common input in rotate_half")
                return

            # Check path for x
            x_path_1 = self.model.match_parent_path(
                node,
                ["Mul", "Transpose"],
                [0, 0],
            )

            x_path_2 = self.model.match_parent_path(
                node,
                ["Mul", "Slice"],
                [0, 0],
            )

            x_path = x_path_1 or x_path_2

            if x_path is None:
                logger.debug("fuse_rotary_embeddings: failed to match x in rotate_half")
                return

            # Check path for sin
            sin_path, sin_cache, position_ids = None, "", ""
            sin_path_1 = self.model.match_parent_path(
                node,
                ["Mul", "Unsqueeze", "Gather", "Squeeze", "Squeeze", "Slice", "Unsqueeze", "Gather", "Shape"],
                [1, 1, 0, 0, 0, 0, 2, 0, 0],
            )
            sin_path_2 = self.model.match_parent_path(
                node,
                ["Mul", "Unsqueeze", "Gather", "Squeeze", "Squeeze", "Slice", "Unsqueeze", "Add"],
                [1, 1, 0, 0, 0, 0, 2, 0],
            )
            sin_path_3 = self.model.match_parent_path(
                node,
                ["Mul", "Unsqueeze", "Gather", "Slice", "Unsqueeze", "Gather", "Shape"],
                [1, 1, 0, 0, 2, 0, 0],
            )
            sin_path_4 = self.model.match_parent_path(
                node,
                ["Mul", "Unsqueeze", "Gather", "Slice", "Unsqueeze", "Add"],
                [1, 1, 0, 0, 2, 0],
            )
            if sin_path_1 is not None:
                sin_path = sin_path_1
                sin_cache = sin_path[-4].input[0]
            elif sin_path_2 is not None:
                sin_path = sin_path_2
                sin_cache = sin_path[-3].input[0]
            elif sin_path_3 is not None:
                sin_path = sin_path_3
                sin_cache = sin_path[-4].input[0]
                position_ids = sin_path[2].input[1]
            elif sin_path_4 is not None:
                sin_path = sin_path_4
                sin_cache = sin_path[-3].input[0]
                position_ids = sin_path[2].input[1]
            else:
                logger.debug("fuse_rotary_embeddings: failed to match sin path in apply_rope")
                return

            # Check path for cos
            cos_path, cos_cache = None, ""
            cos_path_1 = self.model.match_parent_path(
                node,
                ["Mul", "Unsqueeze", "Gather", "Squeeze", "Squeeze", "Slice", "Unsqueeze", "Gather", "Shape"],
                [0, 1, 0, 0, 0, 0, 2, 0, 0],
            )
            cos_path_2 = self.model.match_parent_path(
                node,
                ["Mul", "Unsqueeze", "Gather", "Squeeze", "Squeeze", "Slice", "Unsqueeze", "Add"],
                [0, 1, 0, 0, 0, 0, 2, 0],
            )
            cos_path_3 = self.model.match_parent_path(
                node,
                ["Mul", "Unsqueeze", "Gather", "Slice", "Unsqueeze", "Gather", "Shape"],
                [0, 1, 0, 0, 2, 0, 0],
            )
            cos_path_4 = self.model.match_parent_path(
                node,
                ["Mul", "Unsqueeze", "Gather", "Slice", "Unsqueeze", "Add"],
                [0, 1, 0, 0, 2, 0],
            )
            if cos_path_1 is not None:
                cos_path = cos_path_1
                cos_cache = cos_path[-4].input[0]
            elif cos_path_2 is not None:
                cos_path = cos_path_2
                cos_cache = cos_path[-3].input[0]
            elif cos_path_3 is not None:
                cos_path = cos_path_3
                cos_cache = cos_path[-4].input[0]
                position_ids = cos_path[2].input[1]
            elif cos_path_4 is not None:
                cos_path = cos_path_4
                cos_cache = cos_path[-3].input[0]
                position_ids = cos_path[2].input[1]
            else:
                logger.debug("fuse_rotary_embeddings: failed to match sin path in apply_rope")
                return

            # Check path for position ids
            if position_ids == "":
                position_ids_from_sin_path = self.model.match_parent_path(
                    sin_path[2],
                    ["Reshape"],
                    [1],
                )
                position_ids_from_cos_path = self.model.match_parent_path(
                    cos_path[2],
                    ["Reshape"],
                    [1],
                )
                if (
                    position_ids_from_sin_path is None
                    or position_ids_from_cos_path is None
                    or position_ids_from_sin_path[0].name != position_ids_from_cos_path[0].name
                ):
                    logger.debug("fuse_rotary_embeddings: failed to match position ids path in apply_rope")
                    return
                position_ids = position_ids_from_cos_path[0].input[0]
            else:
                position_ids_from_sin_path = []
                position_ids_from_cos_path = []

            past_seq_len_path, curr_seq_len_path = None, None
            if (sin_path == sin_path_1 and cos_path == cos_path_1) or (
                sin_path == sin_path_3 and cos_path == cos_path_3
            ):
                if sin_path[-2].name != cos_path[-2].name or sin_path[-1].name != cos_path[-1].name:
                    logger.debug(
                        "fuse_rotary_embeddings: failed to match common Gather node and Shape node in sin cache and cos cache"
                    )
                    return
            elif (sin_path == sin_path_2 and cos_path == cos_path_2) or (
                sin_path == sin_path_4 and cos_path == cos_path_4
            ):
                if sin_path[-1].name != cos_path[-1].name:
                    logger.debug("fuse_rotary_embeddings: failed to match common Add node in sin cache and cos cache")
                    return
                # Match past sequence length path: past_key --> Shape --> Gather --> Add
                past_seq_len_path = self.model.match_parent_path(
                    sin_path[-1],
                    ["Gather", "Shape"],
                    [1, 0],
                )
                # Match current sequence length path: transpose_k --> Shape --> Gather --> Add
                curr_seq_len_path = self.model.match_parent_path(
                    sin_path[-1],
                    ["Gather", "Shape", "Transpose"],
                    [0, 0, 0],
                )
                if (
                    past_seq_len_path is None
                    or curr_seq_len_path is None
                    or self.model.find_graph_input(past_seq_len_path[-1].input[0]) is None
                    or curr_seq_len_path[-1].op_type != "Transpose"
                ):
                    logger.debug("fuse_rotary_embeddings: failed to match past_seq_len and curr_seq_len paths")
                    return
            else:
                logger.debug("fuse_rotary_embeddings: failed to match common cache paths")

            rotary_emb_node = self.create_rotary_embeddings_from_nodes(
                rotate_half_x1_path_1[-1].output[0],
                position_ids,
                cos_cache,
                sin_cache,
                node.output[0],
            )
            if rotary_emb_node is None:
                logger.debug("fuse_rotary_embeddings: failed to create RotaryEmbedding node")
                return

            # Remove rotary embedding nodes
            self.add_nodes_to_remove([node])
            self.add_nodes_to_remove(rotate_half_x1_path_1[:-1])
            self.add_nodes_to_remove(rotate_half_x1_path_2[:-1])
            self.add_nodes_to_remove(rotate_half_x2_path_1[:-1])
            self.add_nodes_to_remove(rotate_half_x2_path_2[:-1])
            self.add_nodes_to_remove(x_path[:-1])
            self.add_nodes_to_remove(sin_path)
            self.add_nodes_to_remove(cos_path)
            self.add_nodes_to_remove(position_ids_from_sin_path[:-1])
            self.add_nodes_to_remove(position_ids_from_cos_path[:-1])

            if past_seq_len_path is not None and len(self.model.get_children(past_seq_len_path[0])) == 1:
                # In merged HF model, output of Gather in past_seq_len_path is used twice
                # for past_key_values.0.key and once for other past_key_values
                self.add_nodes_to_remove(past_seq_len_path)
            if curr_seq_len_path is not None:
                self.add_nodes_to_remove(curr_seq_len_path[:-1])

        self.increase_counter(self.base_name)
        self.node_name_to_graph_name[rotary_emb_node.name] = self.this_graph_name
        self.nodes_to_add.append(rotary_emb_node)
        self.prune_graph = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\transforms.py ===
""" Integral Transforms """
from functools import reduce, wraps
from itertools import repeat
from sympy.core import S, pi
from sympy.core.add import Add
from sympy.core.function import (
    AppliedUndef, count_ops, expand, expand_mul, Function)
from sympy.core.mul import Mul
from sympy.core.intfunc import igcd, ilcm
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.core.traversal import postorder_traversal
from sympy.functions.combinatorial.factorials import factorial, rf
from sympy.functions.elementary.complexes import re, arg, Abs
from sympy.functions.elementary.exponential import exp, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import piecewise_fold
from sympy.functions.elementary.trigonometric import cos, cot, sin, tan
from sympy.functions.special.bessel import besselj
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import meijerg
from sympy.integrals import integrate, Integral
from sympy.integrals.meijerint import _dummy
from sympy.logic.boolalg import to_cnf, conjuncts, disjuncts, Or, And
from sympy.polys.polyroots import roots
from sympy.polys.polytools import factor, Poly
from sympy.polys.rootoftools import CRootOf
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug


##########################################################################
# Helpers / Utilities
##########################################################################


class IntegralTransformError(NotImplementedError):
    """
    Exception raised in relation to problems computing transforms.

    Explanation
    ===========

    This class is mostly used internally; if integrals cannot be computed
    objects representing unevaluated transforms are usually returned.

    The hint ``needeval=True`` can be used to disable returning transform
    objects, and instead raise this exception if an integral cannot be
    computed.
    """
    def __init__(self, transform, function, msg):
        super().__init__(
            "%s Transform could not be computed: %s." % (transform, msg))
        self.function = function


class IntegralTransform(Function):
    """
    Base class for integral transforms.

    Explanation
    ===========

    This class represents unevaluated transforms.

    To implement a concrete transform, derive from this class and implement
    the ``_compute_transform(f, x, s, **hints)`` and ``_as_integral(f, x, s)``
    functions. If the transform cannot be computed, raise :obj:`IntegralTransformError`.

    Also set ``cls._name``. For instance,

    >>> from sympy import LaplaceTransform
    >>> LaplaceTransform._name
    'Laplace'

    Implement ``self._collapse_extra`` if your function returns more than just a
    number and possibly a convergence condition.
    """

    @property
    def function(self):
        """ The function to be transformed. """
        return self.args[0]

    @property
    def function_variable(self):
        """ The dependent variable of the function to be transformed. """
        return self.args[1]

    @property
    def transform_variable(self):
        """ The independent transform variable. """
        return self.args[2]

    @property
    def free_symbols(self):
        """
        This method returns the symbols that will exist when the transform
        is evaluated.
        """
        return self.function.free_symbols.union({self.transform_variable}) \
            - {self.function_variable}

    def _compute_transform(self, f, x, s, **hints):
        raise NotImplementedError

    def _as_integral(self, f, x, s):
        raise NotImplementedError

    def _collapse_extra(self, extra):
        cond = And(*extra)
        if cond == False:
            raise IntegralTransformError(self.__class__.name, None, '')
        return cond

    def _try_directly(self, **hints):
        T = None
        try_directly = not any(func.has(self.function_variable)
                               for func in self.function.atoms(AppliedUndef))
        if try_directly:
            try:
                T = self._compute_transform(self.function,
                    self.function_variable, self.transform_variable, **hints)
            except IntegralTransformError:
                debug('[IT _try ] Caught IntegralTransformError, returns None')
                T = None

        fn = self.function
        if not fn.is_Add:
            fn = expand_mul(fn)
        return fn, T

    def doit(self, **hints):
        """
        Try to evaluate the transform in closed form.

        Explanation
        ===========

        This general function handles linearity, but apart from that leaves
        pretty much everything to _compute_transform.

        Standard hints are the following:

        - ``simplify``: whether or not to simplify the result
        - ``noconds``: if True, do not return convergence conditions
        - ``needeval``: if True, raise IntegralTransformError instead of
                        returning IntegralTransform objects

        The default values of these hints depend on the concrete transform,
        usually the default is
        ``(simplify, noconds, needeval) = (True, False, False)``.
        """
        needeval = hints.pop('needeval', False)
        simplify = hints.pop('simplify', True)
        hints['simplify'] = simplify

        fn, T = self._try_directly(**hints)

        if T is not None:
            return T

        if fn.is_Add:
            hints['needeval'] = needeval
            res = [self.__class__(*([x] + list(self.args[1:]))).doit(**hints)
                   for x in fn.args]
            extra = []
            ress = []
            for x in res:
                if not isinstance(x, tuple):
                    x = [x]
                ress.append(x[0])
                if len(x) == 2:
                    # only a condition
                    extra.append(x[1])
                elif len(x) > 2:
                    # some region parameters and a condition (Mellin, Laplace)
                    extra += [x[1:]]
            if simplify==True:
                res = Add(*ress).simplify()
            else:
                res = Add(*ress)
            if not extra:
                return res
            try:
                extra = self._collapse_extra(extra)
                if iterable(extra):
                    return (res,) + tuple(extra)
                else:
                    return (res, extra)
            except IntegralTransformError:
                pass

        if needeval:
            raise IntegralTransformError(
                self.__class__._name, self.function, 'needeval')

        # TODO handle derivatives etc

        # pull out constant coefficients
        coeff, rest = fn.as_coeff_mul(self.function_variable)
        return coeff*self.__class__(*([Mul(*rest)] + list(self.args[1:])))

    @property
    def as_integral(self):
        return self._as_integral(self.function, self.function_variable,
                                 self.transform_variable)

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        return self.as_integral


def _simplify(expr, doit):
    if doit:
        from sympy.simplify import simplify
        from sympy.simplify.powsimp import powdenest
        return simplify(powdenest(piecewise_fold(expr), polar=True))
    return expr


def _noconds_(default):
    """
    This is a decorator generator for dropping convergence conditions.

    Explanation
    ===========

    Suppose you define a function ``transform(*args)`` which returns a tuple of
    the form ``(result, cond1, cond2, ...)``.

    Decorating it ``@_noconds_(default)`` will add a new keyword argument
    ``noconds`` to it. If ``noconds=True``, the return value will be altered to
    be only ``result``, whereas if ``noconds=False`` the return value will not
    be altered.

    The default value of the ``noconds`` keyword will be ``default`` (i.e. the
    argument of this function).
    """
    def make_wrapper(func):
        @wraps(func)
        def wrapper(*args, noconds=default, **kwargs):
            res = func(*args, **kwargs)
            if noconds:
                return res[0]
            return res
        return wrapper
    return make_wrapper
_noconds = _noconds_(False)


##########################################################################
# Mellin Transform
##########################################################################

def _default_integrator(f, x):
    return integrate(f, (x, S.Zero, S.Infinity))


@_noconds
def _mellin_transform(f, x, s_, integrator=_default_integrator, simplify=True):
    """ Backend function to compute Mellin transforms. """
    # We use a fresh dummy, because assumptions on s might drop conditions on
    # convergence of the integral.
    s = _dummy('s', 'mellin-transform', f)
    F = integrator(x**(s - 1) * f, x)

    if not F.has(Integral):
        return _simplify(F.subs(s, s_), simplify), (S.NegativeInfinity, S.Infinity), S.true

    if not F.is_Piecewise:  # XXX can this work if integration gives continuous result now?
        raise IntegralTransformError('Mellin', f, 'could not compute integral')

    F, cond = F.args[0]
    if F.has(Integral):
        raise IntegralTransformError(
            'Mellin', f, 'integral in unexpected form')

    def process_conds(cond):
        """
        Turn ``cond`` into a strip (a, b), and auxiliary conditions.
        """
        from sympy.solvers.inequalities import _solve_inequality
        a = S.NegativeInfinity
        b = S.Infinity
        aux = S.true
        conds = conjuncts(to_cnf(cond))
        t = Dummy('t', real=True)
        for c in conds:
            a_ = S.Infinity
            b_ = S.NegativeInfinity
            aux_ = []
            for d in disjuncts(c):
                d_ = d.replace(
                    re, lambda x: x.as_real_imag()[0]).subs(re(s), t)
                if not d.is_Relational or \
                    d.rel_op in ('==', '!=') \
                        or d_.has(s) or not d_.has(t):
                    aux_ += [d]
                    continue
                soln = _solve_inequality(d_, t)
                if not soln.is_Relational or \
                        soln.rel_op in ('==', '!='):
                    aux_ += [d]
                    continue
                if soln.lts == t:
                    b_ = Max(soln.gts, b_)
                else:
                    a_ = Min(soln.lts, a_)
            if a_ is not S.Infinity and a_ != b:
                a = Max(a_, a)
            elif b_ is not S.NegativeInfinity and b_ != a:
                b = Min(b_, b)
            else:
                aux = And(aux, Or(*aux_))
        return a, b, aux

    conds = [process_conds(c) for c in disjuncts(cond)]
    conds = [x for x in conds if x[2] != False]
    conds.sort(key=lambda x: (x[0] - x[1], count_ops(x[2])))

    if not conds:
        raise IntegralTransformError('Mellin', f, 'no convergence found')

    a, b, aux = conds[0]
    return _simplify(F.subs(s, s_), simplify), (a, b), aux


class MellinTransform(IntegralTransform):
    """
    Class representing unevaluated Mellin transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute Mellin transforms, see the :func:`mellin_transform`
    docstring.
    """

    _name = 'Mellin'

    def _compute_transform(self, f, x, s, **hints):
        return _mellin_transform(f, x, s, **hints)

    def _as_integral(self, f, x, s):
        return Integral(f*x**(s - 1), (x, S.Zero, S.Infinity))

    def _collapse_extra(self, extra):
        a = []
        b = []
        cond = []
        for (sa, sb), c in extra:
            a += [sa]
            b += [sb]
            cond += [c]
        res = (Max(*a), Min(*b)), And(*cond)
        if (res[0][0] >= res[0][1]) == True or res[1] == False:
            raise IntegralTransformError(
                'Mellin', None, 'no combined convergence.')
        return res


def mellin_transform(f, x, s, **hints):
    r"""
    Compute the Mellin transform `F(s)` of `f(x)`,

    .. math :: F(s) = \int_0^\infty x^{s-1} f(x) \mathrm{d}x.

    For all "sensible" functions, this converges absolutely in a strip
      `a < \operatorname{Re}(s) < b`.

    Explanation
    ===========

    The Mellin transform is related via change of variables to the Fourier
    transform, and also to the (bilateral) Laplace transform.

    This function returns ``(F, (a, b), cond)``
    where ``F`` is the Mellin transform of ``f``, ``(a, b)`` is the fundamental strip
    (as above), and ``cond`` are auxiliary convergence conditions.

    If the integral cannot be computed in closed form, this function returns
    an unevaluated :class:`MellinTransform` object.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`. If ``noconds=False``,
    then only `F` will be returned (i.e. not ``cond``, and also not the strip
    ``(a, b)``).

    Examples
    ========

    >>> from sympy import mellin_transform, exp
    >>> from sympy.abc import x, s
    >>> mellin_transform(exp(-x), x, s)
    (gamma(s), (0, oo), True)

    See Also
    ========

    inverse_mellin_transform, laplace_transform, fourier_transform
    hankel_transform, inverse_hankel_transform
    """
    return MellinTransform(f, x, s).doit(**hints)


def _rewrite_sin(m_n, s, a, b):
    """
    Re-write the sine function ``sin(m*s + n)`` as gamma functions, compatible
    with the strip (a, b).

    Return ``(gamma1, gamma2, fac)`` so that ``f == fac/(gamma1 * gamma2)``.

    Examples
    ========

    >>> from sympy.integrals.transforms import _rewrite_sin
    >>> from sympy import pi, S
    >>> from sympy.abc import s
    >>> _rewrite_sin((pi, 0), s, 0, 1)
    (gamma(s), gamma(1 - s), pi)
    >>> _rewrite_sin((pi, 0), s, 1, 0)
    (gamma(s - 1), gamma(2 - s), -pi)
    >>> _rewrite_sin((pi, 0), s, -1, 0)
    (gamma(s + 1), gamma(-s), -pi)
    >>> _rewrite_sin((pi, pi/2), s, S(1)/2, S(3)/2)
    (gamma(s - 1/2), gamma(3/2 - s), -pi)
    >>> _rewrite_sin((pi, pi), s, 0, 1)
    (gamma(s), gamma(1 - s), -pi)
    >>> _rewrite_sin((2*pi, 0), s, 0, S(1)/2)
    (gamma(2*s), gamma(1 - 2*s), pi)
    >>> _rewrite_sin((2*pi, 0), s, S(1)/2, 1)
    (gamma(2*s - 1), gamma(2 - 2*s), -pi)
    """
    # (This is a separate function because it is moderately complicated,
    #  and I want to doctest it.)
    # We want to use pi/sin(pi*x) = gamma(x)*gamma(1-x).
    # But there is one complication: the gamma functions determine the
    # integration contour in the definition of the G-function. Usually
    # it would not matter if this is slightly shifted, unless this way
    # we create an undefined function!
    # So we try to write this in such a way that the gammas are
    # eminently on the right side of the strip.
    m, n = m_n

    m = expand_mul(m/pi)
    n = expand_mul(n/pi)
    r = ceiling(-m*a - n.as_real_imag()[0])  # Don't use re(n), does not expand
    return gamma(m*s + n + r), gamma(1 - n - r - m*s), (-1)**r*pi


class MellinTransformStripError(ValueError):
    """
    Exception raised by _rewrite_gamma. Mainly for internal use.
    """
    pass


def _rewrite_gamma(f, s, a, b):
    """
    Try to rewrite the product f(s) as a product of gamma functions,
    so that the inverse Mellin transform of f can be expressed as a meijer
    G function.

    Explanation
    ===========

    Return (an, ap), (bm, bq), arg, exp, fac such that
    G((an, ap), (bm, bq), arg/z**exp)*fac is the inverse Mellin transform of f(s).

    Raises IntegralTransformError or MellinTransformStripError on failure.

    It is asserted that f has no poles in the fundamental strip designated by
    (a, b). One of a and b is allowed to be None. The fundamental strip is
    important, because it determines the inversion contour.

    This function can handle exponentials, linear factors, trigonometric
    functions.

    This is a helper function for inverse_mellin_transform that will not
    attempt any transformations on f.

    Examples
    ========

    >>> from sympy.integrals.transforms import _rewrite_gamma
    >>> from sympy.abc import s
    >>> from sympy import oo
    >>> _rewrite_gamma(s*(s+3)*(s-1), s, -oo, oo)
    (([], [-3, 0, 1]), ([-2, 1, 2], []), 1, 1, -1)
    >>> _rewrite_gamma((s-1)**2, s, -oo, oo)
    (([], [1, 1]), ([2, 2], []), 1, 1, 1)

    Importance of the fundamental strip:

    >>> _rewrite_gamma(1/s, s, 0, oo)
    (([1], []), ([], [0]), 1, 1, 1)
    >>> _rewrite_gamma(1/s, s, None, oo)
    (([1], []), ([], [0]), 1, 1, 1)
    >>> _rewrite_gamma(1/s, s, 0, None)
    (([1], []), ([], [0]), 1, 1, 1)
    >>> _rewrite_gamma(1/s, s, -oo, 0)
    (([], [1]), ([0], []), 1, 1, -1)
    >>> _rewrite_gamma(1/s, s, None, 0)
    (([], [1]), ([0], []), 1, 1, -1)
    >>> _rewrite_gamma(1/s, s, -oo, None)
    (([], [1]), ([0], []), 1, 1, -1)

    >>> _rewrite_gamma(2**(-s+3), s, -oo, oo)
    (([], []), ([], []), 1/2, 1, 8)
    """
    # Our strategy will be as follows:
    # 1) Guess a constant c such that the inversion integral should be
    #    performed wrt s'=c*s (instead of plain s). Write s for s'.
    # 2) Process all factors, rewrite them independently as gamma functions in
    #    argument s, or exponentials of s.
    # 3) Try to transform all gamma functions s.t. they have argument
    #    a+s or a-s.
    # 4) Check that the resulting G function parameters are valid.
    # 5) Combine all the exponentials.

    a_, b_ = S([a, b])

    def left(c, is_numer):
        """
        Decide whether pole at c lies to the left of the fundamental strip.
        """
        # heuristically, this is the best chance for us to solve the inequalities
        c = expand(re(c))
        if a_ is None and b_ is S.Infinity:
            return True
        if a_ is None:
            return c < b_
        if b_ is None:
            return c <= a_
        if (c >= b_) == True:
            return False
        if (c <= a_) == True:
            return True
        if is_numer:
            return None
        if a_.free_symbols or b_.free_symbols or c.free_symbols:
            return None  # XXX
            #raise IntegralTransformError('Inverse Mellin', f,
            #                     'Could not determine position of singularity %s'
            #                     ' relative to fundamental strip' % c)
        raise MellinTransformStripError('Pole inside critical strip?')

    # 1)
    s_multipliers = []
    for g in f.atoms(gamma):
        if not g.has(s):
            continue
        arg = g.args[0]
        if arg.is_Add:
            arg = arg.as_independent(s)[1]
        coeff, _ = arg.as_coeff_mul(s)
        s_multipliers += [coeff]
    for g in f.atoms(sin, cos, tan, cot):
        if not g.has(s):
            continue
        arg = g.args[0]
        if arg.is_Add:
            arg = arg.as_independent(s)[1]
        coeff, _ = arg.as_coeff_mul(s)
        s_multipliers += [coeff/pi]
    s_multipliers = [Abs(x) if x.is_extended_real else x for x in s_multipliers]
    common_coefficient = S.One
    for x in s_multipliers:
        if not x.is_Rational:
            common_coefficient = x
            break
    s_multipliers = [x/common_coefficient for x in s_multipliers]
    if not (all(x.is_Rational for x in s_multipliers) and
            common_coefficient.is_extended_real):
        raise IntegralTransformError("Gamma", None, "Nonrational multiplier")
    s_multiplier = common_coefficient/reduce(ilcm, [S(x.q)
                                             for x in s_multipliers], S.One)
    if s_multiplier == common_coefficient:
        if len(s_multipliers) == 0:
            s_multiplier = common_coefficient
        else:
            s_multiplier = common_coefficient \
                *reduce(igcd, [S(x.p) for x in s_multipliers])

    f = f.subs(s, s/s_multiplier)
    fac = S.One/s_multiplier
    exponent = S.One/s_multiplier
    if a_ is not None:
        a_ *= s_multiplier
    if b_ is not None:
        b_ *= s_multiplier

    # 2)
    numer, denom = f.as_numer_denom()
    numer = Mul.make_args(numer)
    denom = Mul.make_args(denom)
    args = list(zip(numer, repeat(True))) + list(zip(denom, repeat(False)))

    facs = []
    dfacs = []
    # *_gammas will contain pairs (a, c) representing Gamma(a*s + c)
    numer_gammas = []
    denom_gammas = []
    # exponentials will contain bases for exponentials of s
    exponentials = []

    def exception(fact):
        return IntegralTransformError("Inverse Mellin", f, "Unrecognised form '%s'." % fact)
    while args:
        fact, is_numer = args.pop()
        if is_numer:
            ugammas, lgammas = numer_gammas, denom_gammas
            ufacs = facs
        else:
            ugammas, lgammas = denom_gammas, numer_gammas
            ufacs = dfacs

        def linear_arg(arg):
            """ Test if arg is of form a*s+b, raise exception if not. """
            if not arg.is_polynomial(s):
                raise exception(fact)
            p = Poly(arg, s)
            if p.degree() != 1:
                raise exception(fact)
            return p.all_coeffs()

        # constants
        if not fact.has(s):
            ufacs += [fact]
        # exponentials
        elif fact.is_Pow or isinstance(fact, exp):
            if fact.is_Pow:
                base = fact.base
                exp_ = fact.exp
            else:
                base = exp_polar(1)
                exp_ = fact.exp
            if exp_.is_Integer:
                cond = is_numer
                if exp_ < 0:
                    cond = not cond
                args += [(base, cond)]*Abs(exp_)
                continue
            elif not base.has(s):
                a, b = linear_arg(exp_)
                if not is_numer:
                    base = 1/base
                exponentials += [base**a]
                facs += [base**b]
            else:
                raise exception(fact)
        # linear factors
        elif fact.is_polynomial(s):
            p = Poly(fact, s)
            if p.degree() != 1:
                # We completely factor the poly. For this we need the roots.
                # Now roots() only works in some cases (low degree), and CRootOf
                # only works without parameters. So try both...
                coeff = p.LT()[1]
                rs = roots(p, s)
                if len(rs) != p.degree():
                    rs = CRootOf.all_roots(p)
                ufacs += [coeff]
                args += [(s - c, is_numer) for c in rs]
                continue
            a, c = p.all_coeffs()
            ufacs += [a]
            c /= -a
            # Now need to convert s - c
            if left(c, is_numer):
                ugammas += [(S.One, -c + 1)]
                lgammas += [(S.One, -c)]
            else:
                ufacs += [-1]
                ugammas += [(S.NegativeOne, c + 1)]
                lgammas += [(S.NegativeOne, c)]
        elif isinstance(fact, gamma):
            a, b = linear_arg(fact.args[0])
            if is_numer:
                if (a > 0 and (left(-b/a, is_numer) == False)) or \
                   (a < 0 and (left(-b/a, is_numer) == True)):
                    raise NotImplementedError(
                        'Gammas partially over the strip.')
            ugammas += [(a, b)]
        elif isinstance(fact, sin):
            # We try to re-write all trigs as gammas. This is not in
            # general the best strategy, since sometimes this is impossible,
            # but rewriting as exponentials would work. However trig functions
            # in inverse mellin transforms usually all come from simplifying
            # gamma terms, so this should work.
            a = fact.args[0]
            if is_numer:
                # No problem with the poles.
                gamma1, gamma2, fac_ = gamma(a/pi), gamma(1 - a/pi), pi
            else:
                gamma1, gamma2, fac_ = _rewrite_sin(linear_arg(a), s, a_, b_)
            args += [(gamma1, not is_numer), (gamma2, not is_numer)]
            ufacs += [fac_]
        elif isinstance(fact, tan):
            a = fact.args[0]
            args += [(sin(a, evaluate=False), is_numer),
                     (sin(pi/2 - a, evaluate=False), not is_numer)]
        elif isinstance(fact, cos):
            a = fact.args[0]
            args += [(sin(pi/2 - a, evaluate=False), is_numer)]
        elif isinstance(fact, cot):
            a = fact.args[0]
            args += [(sin(pi/2 - a, evaluate=False), is_numer),
                     (sin(a, evaluate=False), not is_numer)]
        else:
            raise exception(fact)

    fac *= Mul(*facs)/Mul(*dfacs)

    # 3)
    an, ap, bm, bq = [], [], [], []
    for gammas, plus, minus, is_numer in [(numer_gammas, an, bm, True),
                                          (denom_gammas, bq, ap, False)]:
        while gammas:
            a, c = gammas.pop()
            if a != -1 and a != +1:
                # We use the gamma function multiplication theorem.
                p = Abs(S(a))
                newa = a/p
                newc = c/p
                if not a.is_Integer:
                    raise TypeError("a is not an integer")
                for k in range(p):
                    gammas += [(newa, newc + k/p)]
                if is_numer:
                    fac *= (2*pi)**((1 - p)/2) * p**(c - S.Half)
                    exponentials += [p**a]
                else:
                    fac /= (2*pi)**((1 - p)/2) * p**(c - S.Half)
                    exponentials += [p**(-a)]
                continue
            if a == +1:
                plus.append(1 - c)
            else:
                minus.append(c)

    # 4)
    # TODO

    # 5)
    arg = Mul(*exponentials)

    # for testability, sort the arguments
    an.sort(key=default_sort_key)
    ap.sort(key=default_sort_key)
    bm.sort(key=default_sort_key)
    bq.sort(key=default_sort_key)

    return (an, ap), (bm, bq), arg, exponent, fac


@_noconds_(True)
def _inverse_mellin_transform(F, s, x_, strip, as_meijerg=False):
    """ A helper for the real inverse_mellin_transform function, this one here
        assumes x to be real and positive. """
    x = _dummy('t', 'inverse-mellin-transform', F, positive=True)
    # Actually, we won't try integration at all. Instead we use the definition
    # of the Meijer G function as a fairly general inverse mellin transform.
    F = F.rewrite(gamma)
    for g in [factor(F), expand_mul(F), expand(F)]:
        if g.is_Add:
            # do all terms separately
            ress = [_inverse_mellin_transform(G, s, x, strip, as_meijerg,
                                              noconds=False)
                    for G in g.args]
            conds = [p[1] for p in ress]
            ress = [p[0] for p in ress]
            res = Add(*ress)
            if not as_meijerg:
                res = factor(res, gens=res.atoms(Heaviside))
            return res.subs(x, x_), And(*conds)

        try:
            a, b, C, e, fac = _rewrite_gamma(g, s, strip[0], strip[1])
        except IntegralTransformError:
            continue
        try:
            G = meijerg(a, b, C/x**e)
        except ValueError:
            continue
        if as_meijerg:
            h = G
        else:
            try:
                from sympy.simplify import hyperexpand
                h = hyperexpand(G)
            except NotImplementedError:
                raise IntegralTransformError(
                    'Inverse Mellin', F, 'Could not calculate integral')

            if h.is_Piecewise and len(h.args) == 3:
                # XXX we break modularity here!
                h = Heaviside(x - Abs(C))*h.args[0].args[0] \
                    + Heaviside(Abs(C) - x)*h.args[1].args[0]
        # We must ensure that the integral along the line we want converges,
        # and return that value.
        # See [L], 5.2
        cond = [Abs(arg(G.argument)) < G.delta*pi]
        # Note: we allow ">=" here, this corresponds to convergence if we let
        # limits go to oo symmetrically. ">" corresponds to absolute convergence.
        cond += [And(Or(len(G.ap) != len(G.bq), 0 >= re(G.nu) + 1),
                     Abs(arg(G.argument)) == G.delta*pi)]
        cond = Or(*cond)
        if cond == False:
            raise IntegralTransformError(
                'Inverse Mellin', F, 'does not converge')
        return (h*fac).subs(x, x_), cond

    raise IntegralTransformError('Inverse Mellin', F, '')

_allowed = None


class InverseMellinTransform(IntegralTransform):
    """
    Class representing unevaluated inverse Mellin transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse Mellin transforms, see the
    :func:`inverse_mellin_transform` docstring.
    """

    _name = 'Inverse Mellin'
    _none_sentinel = Dummy('None')
    _c = Dummy('c')

    def __new__(cls, F, s, x, a, b, **opts):
        if a is None:
            a = InverseMellinTransform._none_sentinel
        if b is None:
            b = InverseMellinTransform._none_sentinel
        return IntegralTransform.__new__(cls, F, s, x, a, b, **opts)

    @property
    def fundamental_strip(self):
        a, b = self.args[3], self.args[4]
        if a is InverseMellinTransform._none_sentinel:
            a = None
        if b is InverseMellinTransform._none_sentinel:
            b = None
        return a, b

    def _compute_transform(self, F, s, x, **hints):
        # IntegralTransform's doit will cause this hint to exist, but
        # InverseMellinTransform should ignore it
        hints.pop('simplify', True)
        global _allowed
        if _allowed is None:
            _allowed = {
                exp, gamma, sin, cos, tan, cot, cosh, sinh, tanh, coth,
                factorial, rf}
        for f in postorder_traversal(F):
            if f.is_Function and f.has(s) and f.func not in _allowed:
                raise IntegralTransformError('Inverse Mellin', F,
                                     'Component %s not recognised.' % f)
        strip = self.fundamental_strip
        return _inverse_mellin_transform(F, s, x, strip, **hints)

    def _as_integral(self, F, s, x):
        c = self.__class__._c
        return Integral(F*x**(-s), (s, c - S.ImaginaryUnit*S.Infinity, c +
                                    S.ImaginaryUnit*S.Infinity))/(2*S.Pi*S.ImaginaryUnit)


def inverse_mellin_transform(F, s, x, strip, **hints):
    r"""
    Compute the inverse Mellin transform of `F(s)` over the fundamental
    strip given by ``strip=(a, b)``.

    Explanation
    ===========

    This can be defined as

    .. math:: f(x) = \frac{1}{2\pi i} \int_{c - i\infty}^{c + i\infty} x^{-s} F(s) \mathrm{d}s,

    for any `c` in the fundamental strip. Under certain regularity
    conditions on `F` and/or `f`,
    this recovers `f` from its Mellin transform `F`
    (and vice versa), for positive real `x`.

    One of `a` or `b` may be passed as ``None``; a suitable `c` will be
    inferred.

    If the integral cannot be computed in closed form, this function returns
    an unevaluated :class:`InverseMellinTransform` object.

    Note that this function will assume x to be positive and real, regardless
    of the SymPy assumptions!

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.

    Examples
    ========

    >>> from sympy import inverse_mellin_transform, oo, gamma
    >>> from sympy.abc import x, s
    >>> inverse_mellin_transform(gamma(s), s, x, (0, oo))
    exp(-x)

    The fundamental strip matters:

    >>> f = 1/(s**2 - 1)
    >>> inverse_mellin_transform(f, s, x, (-oo, -1))
    x*(1 - 1/x**2)*Heaviside(x - 1)/2
    >>> inverse_mellin_transform(f, s, x, (-1, 1))
    -x*Heaviside(1 - x)/2 - Heaviside(x - 1)/(2*x)
    >>> inverse_mellin_transform(f, s, x, (1, oo))
    (1/2 - x**2/2)*Heaviside(1 - x)/x

    See Also
    ========

    mellin_transform
    hankel_transform, inverse_hankel_transform
    """
    return InverseMellinTransform(F, s, x, strip[0], strip[1]).doit(**hints)


##########################################################################
# Fourier Transform
##########################################################################

@_noconds_(True)
def _fourier_transform(f, x, k, a, b, name, simplify=True):
    r"""
    Compute a general Fourier-type transform

    .. math::

        F(k) = a \int_{-\infty}^{\infty} e^{bixk} f(x)\, dx.

    For suitable choice of *a* and *b*, this reduces to the standard Fourier
    and inverse Fourier transforms.
    """
    F = integrate(a*f*exp(b*S.ImaginaryUnit*x*k), (x, S.NegativeInfinity, S.Infinity))

    if not F.has(Integral):
        return _simplify(F, simplify), S.true

    integral_f = integrate(f, (x, S.NegativeInfinity, S.Infinity))
    if integral_f in (S.NegativeInfinity, S.Infinity, S.NaN) or integral_f.has(Integral):
        raise IntegralTransformError(name, f, 'function not integrable on real axis')

    if not F.is_Piecewise:
        raise IntegralTransformError(name, f, 'could not compute integral')

    F, cond = F.args[0]
    if F.has(Integral):
        raise IntegralTransformError(name, f, 'integral in unexpected form')

    return _simplify(F, simplify), cond


class FourierTypeTransform(IntegralTransform):
    """ Base class for Fourier transforms."""

    def a(self):
        raise NotImplementedError(
            "Class %s must implement a(self) but does not" % self.__class__)

    def b(self):
        raise NotImplementedError(
            "Class %s must implement b(self) but does not" % self.__class__)

    def _compute_transform(self, f, x, k, **hints):
        return _fourier_transform(f, x, k,
                                  self.a(), self.b(),
                                  self.__class__._name, **hints)

    def _as_integral(self, f, x, k):
        a = self.a()
        b = self.b()
        return Integral(a*f*exp(b*S.ImaginaryUnit*x*k), (x, S.NegativeInfinity, S.Infinity))


class FourierTransform(FourierTypeTransform):
    """
    Class representing unevaluated Fourier transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute Fourier transforms, see the :func:`fourier_transform`
    docstring.
    """

    _name = 'Fourier'

    def a(self):
        return 1

    def b(self):
        return -2*S.Pi


def fourier_transform(f, x, k, **hints):
    r"""
    Compute the unitary, ordinary-frequency Fourier transform of ``f``, defined
    as

    .. math:: F(k) = \int_{-\infty}^\infty f(x) e^{-2\pi i x k} \mathrm{d} x.

    Explanation
    ===========

    If the transform cannot be computed in closed form, this
    function returns an unevaluated :class:`FourierTransform` object.

    For other Fourier transform conventions, see the function
    :func:`sympy.integrals.transforms._fourier_transform`.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.
    Note that for this transform, by default ``noconds=True``.

    Examples
    ========

    >>> from sympy import fourier_transform, exp
    >>> from sympy.abc import x, k
    >>> fourier_transform(exp(-x**2), x, k)
    sqrt(pi)*exp(-pi**2*k**2)
    >>> fourier_transform(exp(-x**2), x, k, noconds=False)
    (sqrt(pi)*exp(-pi**2*k**2), True)

    See Also
    ========

    inverse_fourier_transform
    sine_transform, inverse_sine_transform
    cosine_transform, inverse_cosine_transform
    hankel_transform, inverse_hankel_transform
    mellin_transform, laplace_transform
    """
    return FourierTransform(f, x, k).doit(**hints)


class InverseFourierTransform(FourierTypeTransform):
    """
    Class representing unevaluated inverse Fourier transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse Fourier transforms, see the
    :func:`inverse_fourier_transform` docstring.
    """

    _name = 'Inverse Fourier'

    def a(self):
        return 1

    def b(self):
        return 2*S.Pi


def inverse_fourier_transform(F, k, x, **hints):
    r"""
    Compute the unitary, ordinary-frequency inverse Fourier transform of `F`,
    defined as

    .. math:: f(x) = \int_{-\infty}^\infty F(k) e^{2\pi i x k} \mathrm{d} k.

    Explanation
    ===========

    If the transform cannot be computed in closed form, this
    function returns an unevaluated :class:`InverseFourierTransform` object.

    For other Fourier transform conventions, see the function
    :func:`sympy.integrals.transforms._fourier_transform`.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.
    Note that for this transform, by default ``noconds=True``.

    Examples
    ========

    >>> from sympy import inverse_fourier_transform, exp, sqrt, pi
    >>> from sympy.abc import x, k
    >>> inverse_fourier_transform(sqrt(pi)*exp(-(pi*k)**2), k, x)
    exp(-x**2)
    >>> inverse_fourier_transform(sqrt(pi)*exp(-(pi*k)**2), k, x, noconds=False)
    (exp(-x**2), True)

    See Also
    ========

    fourier_transform
    sine_transform, inverse_sine_transform
    cosine_transform, inverse_cosine_transform
    hankel_transform, inverse_hankel_transform
    mellin_transform, laplace_transform
    """
    return InverseFourierTransform(F, k, x).doit(**hints)


##########################################################################
# Fourier Sine and Cosine Transform
##########################################################################

@_noconds_(True)
def _sine_cosine_transform(f, x, k, a, b, K, name, simplify=True):
    """
    Compute a general sine or cosine-type transform
        F(k) = a int_0^oo b*sin(x*k) f(x) dx.
        F(k) = a int_0^oo b*cos(x*k) f(x) dx.

    For suitable choice of a and b, this reduces to the standard sine/cosine
    and inverse sine/cosine transforms.
    """
    F = integrate(a*f*K(b*x*k), (x, S.Zero, S.Infinity))

    if not F.has(Integral):
        return _simplify(F, simplify), S.true

    if not F.is_Piecewise:
        raise IntegralTransformError(name, f, 'could not compute integral')

    F, cond = F.args[0]
    if F.has(Integral):
        raise IntegralTransformError(name, f, 'integral in unexpected form')

    return _simplify(F, simplify), cond


class SineCosineTypeTransform(IntegralTransform):
    """
    Base class for sine and cosine transforms.
    Specify cls._kern.
    """

    def a(self):
        raise NotImplementedError(
            "Class %s must implement a(self) but does not" % self.__class__)

    def b(self):
        raise NotImplementedError(
            "Class %s must implement b(self) but does not" % self.__class__)


    def _compute_transform(self, f, x, k, **hints):
        return _sine_cosine_transform(f, x, k,
                                      self.a(), self.b(),
                                      self.__class__._kern,
                                      self.__class__._name, **hints)

    def _as_integral(self, f, x, k):
        a = self.a()
        b = self.b()
        K = self.__class__._kern
        return Integral(a*f*K(b*x*k), (x, S.Zero, S.Infinity))


class SineTransform(SineCosineTypeTransform):
    """
    Class representing unevaluated sine transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute sine transforms, see the :func:`sine_transform`
    docstring.
    """

    _name = 'Sine'
    _kern = sin

    def a(self):
        return sqrt(2)/sqrt(pi)

    def b(self):
        return S.One


def sine_transform(f, x, k, **hints):
    r"""
    Compute the unitary, ordinary-frequency sine transform of `f`, defined
    as

    .. math:: F(k) = \sqrt{\frac{2}{\pi}} \int_{0}^\infty f(x) \sin(2\pi x k) \mathrm{d} x.

    Explanation
    ===========

    If the transform cannot be computed in closed form, this
    function returns an unevaluated :class:`SineTransform` object.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.
    Note that for this transform, by default ``noconds=True``.

    Examples
    ========

    >>> from sympy import sine_transform, exp
    >>> from sympy.abc import x, k, a
    >>> sine_transform(x*exp(-a*x**2), x, k)
    sqrt(2)*k*exp(-k**2/(4*a))/(4*a**(3/2))
    >>> sine_transform(x**(-a), x, k)
    2**(1/2 - a)*k**(a - 1)*gamma(1 - a/2)/gamma(a/2 + 1/2)

    See Also
    ========

    fourier_transform, inverse_fourier_transform
    inverse_sine_transform
    cosine_transform, inverse_cosine_transform
    hankel_transform, inverse_hankel_transform
    mellin_transform, laplace_transform
    """
    return SineTransform(f, x, k).doit(**hints)


class InverseSineTransform(SineCosineTypeTransform):
    """
    Class representing unevaluated inverse sine transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse sine transforms, see the
    :func:`inverse_sine_transform` docstring.
    """

    _name = 'Inverse Sine'
    _kern = sin

    def a(self):
        return sqrt(2)/sqrt(pi)

    def b(self):
        return S.One


def inverse_sine_transform(F, k, x, **hints):
    r"""
    Compute the unitary, ordinary-frequency inverse sine transform of `F`,
    defined as

    .. math:: f(x) = \sqrt{\frac{2}{\pi}} \int_{0}^\infty F(k) \sin(2\pi x k) \mathrm{d} k.

    Explanation
    ===========

    If the transform cannot be computed in closed form, this
    function returns an unevaluated :class:`InverseSineTransform` object.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.
    Note that for this transform, by default ``noconds=True``.

    Examples
    ========

    >>> from sympy import inverse_sine_transform, exp, sqrt, gamma
    >>> from sympy.abc import x, k, a
    >>> inverse_sine_transform(2**((1-2*a)/2)*k**(a - 1)*
    ...     gamma(-a/2 + 1)/gamma((a+1)/2), k, x)
    x**(-a)
    >>> inverse_sine_transform(sqrt(2)*k*exp(-k**2/(4*a))/(4*sqrt(a)**3), k, x)
    x*exp(-a*x**2)

    See Also
    ========

    fourier_transform, inverse_fourier_transform
    sine_transform
    cosine_transform, inverse_cosine_transform
    hankel_transform, inverse_hankel_transform
    mellin_transform, laplace_transform
    """
    return InverseSineTransform(F, k, x).doit(**hints)


class CosineTransform(SineCosineTypeTransform):
    """
    Class representing unevaluated cosine transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute cosine transforms, see the :func:`cosine_transform`
    docstring.
    """

    _name = 'Cosine'
    _kern = cos

    def a(self):
        return sqrt(2)/sqrt(pi)

    def b(self):
        return S.One


def cosine_transform(f, x, k, **hints):
    r"""
    Compute the unitary, ordinary-frequency cosine transform of `f`, defined
    as

    .. math:: F(k) = \sqrt{\frac{2}{\pi}} \int_{0}^\infty f(x) \cos(2\pi x k) \mathrm{d} x.

    Explanation
    ===========

    If the transform cannot be computed in closed form, this
    function returns an unevaluated :class:`CosineTransform` object.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.
    Note that for this transform, by default ``noconds=True``.

    Examples
    ========

    >>> from sympy import cosine_transform, exp, sqrt, cos
    >>> from sympy.abc import x, k, a
    >>> cosine_transform(exp(-a*x), x, k)
    sqrt(2)*a/(sqrt(pi)*(a**2 + k**2))
    >>> cosine_transform(exp(-a*sqrt(x))*cos(a*sqrt(x)), x, k)
    a*exp(-a**2/(2*k))/(2*k**(3/2))

    See Also
    ========

    fourier_transform, inverse_fourier_transform,
    sine_transform, inverse_sine_transform
    inverse_cosine_transform
    hankel_transform, inverse_hankel_transform
    mellin_transform, laplace_transform
    """
    return CosineTransform(f, x, k).doit(**hints)


class InverseCosineTransform(SineCosineTypeTransform):
    """
    Class representing unevaluated inverse cosine transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse cosine transforms, see the
    :func:`inverse_cosine_transform` docstring.
    """

    _name = 'Inverse Cosine'
    _kern = cos

    def a(self):
        return sqrt(2)/sqrt(pi)

    def b(self):
        return S.One


def inverse_cosine_transform(F, k, x, **hints):
    r"""
    Compute the unitary, ordinary-frequency inverse cosine transform of `F`,
    defined as

    .. math:: f(x) = \sqrt{\frac{2}{\pi}} \int_{0}^\infty F(k) \cos(2\pi x k) \mathrm{d} k.

    Explanation
    ===========

    If the transform cannot be computed in closed form, this
    function returns an unevaluated :class:`InverseCosineTransform` object.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.
    Note that for this transform, by default ``noconds=True``.

    Examples
    ========

    >>> from sympy import inverse_cosine_transform, sqrt, pi
    >>> from sympy.abc import x, k, a
    >>> inverse_cosine_transform(sqrt(2)*a/(sqrt(pi)*(a**2 + k**2)), k, x)
    exp(-a*x)
    >>> inverse_cosine_transform(1/sqrt(k), k, x)
    1/sqrt(x)

    See Also
    ========

    fourier_transform, inverse_fourier_transform,
    sine_transform, inverse_sine_transform
    cosine_transform
    hankel_transform, inverse_hankel_transform
    mellin_transform, laplace_transform
    """
    return InverseCosineTransform(F, k, x).doit(**hints)


##########################################################################
# Hankel Transform
##########################################################################

@_noconds_(True)
def _hankel_transform(f, r, k, nu, name, simplify=True):
    r"""
    Compute a general Hankel transform

    .. math:: F_\nu(k) = \int_{0}^\infty f(r) J_\nu(k r) r \mathrm{d} r.
    """
    F = integrate(f*besselj(nu, k*r)*r, (r, S.Zero, S.Infinity))

    if not F.has(Integral):
        return _simplify(F, simplify), S.true

    if not F.is_Piecewise:
        raise IntegralTransformError(name, f, 'could not compute integral')

    F, cond = F.args[0]
    if F.has(Integral):
        raise IntegralTransformError(name, f, 'integral in unexpected form')

    return _simplify(F, simplify), cond


class HankelTypeTransform(IntegralTransform):
    """
    Base class for Hankel transforms.
    """

    def doit(self, **hints):
        return self._compute_transform(self.function,
                                       self.function_variable,
                                       self.transform_variable,
                                       self.args[3],
                                       **hints)

    def _compute_transform(self, f, r, k, nu, **hints):
        return _hankel_transform(f, r, k, nu, self._name, **hints)

    def _as_integral(self, f, r, k, nu):
        return Integral(f*besselj(nu, k*r)*r, (r, S.Zero, S.Infinity))

    @property
    def as_integral(self):
        return self._as_integral(self.function,
                                 self.function_variable,
                                 self.transform_variable,
                                 self.args[3])


class HankelTransform(HankelTypeTransform):
    """
    Class representing unevaluated Hankel transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute Hankel transforms, see the :func:`hankel_transform`
    docstring.
    """

    _name = 'Hankel'


def hankel_transform(f, r, k, nu, **hints):
    r"""
    Compute the Hankel transform of `f`, defined as

    .. math:: F_\nu(k) = \int_{0}^\infty f(r) J_\nu(k r) r \mathrm{d} r.

    Explanation
    ===========

    If the transform cannot be computed in closed form, this
    function returns an unevaluated :class:`HankelTransform` object.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.
    Note that for this transform, by default ``noconds=True``.

    Examples
    ========

    >>> from sympy import hankel_transform, inverse_hankel_transform
    >>> from sympy import exp
    >>> from sympy.abc import r, k, m, nu, a

    >>> ht = hankel_transform(1/r**m, r, k, nu)
    >>> ht
    2*k**(m - 2)*gamma(-m/2 + nu/2 + 1)/(2**m*gamma(m/2 + nu/2))

    >>> inverse_hankel_transform(ht, k, r, nu)
    r**(-m)

    >>> ht = hankel_transform(exp(-a*r), r, k, 0)
    >>> ht
    a/(k**3*(a**2/k**2 + 1)**(3/2))

    >>> inverse_hankel_transform(ht, k, r, 0)
    exp(-a*r)

    See Also
    ========

    fourier_transform, inverse_fourier_transform
    sine_transform, inverse_sine_transform
    cosine_transform, inverse_cosine_transform
    inverse_hankel_transform
    mellin_transform, laplace_transform
    """
    return HankelTransform(f, r, k, nu).doit(**hints)


class InverseHankelTransform(HankelTypeTransform):
    """
    Class representing unevaluated inverse Hankel transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse Hankel transforms, see the
    :func:`inverse_hankel_transform` docstring.
    """

    _name = 'Inverse Hankel'


def inverse_hankel_transform(F, k, r, nu, **hints):
    r"""
    Compute the inverse Hankel transform of `F` defined as

    .. math:: f(r) = \int_{0}^\infty F_\nu(k) J_\nu(k r) k \mathrm{d} k.

    Explanation
    ===========

    If the transform cannot be computed in closed form, this
    function returns an unevaluated :class:`InverseHankelTransform` object.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.
    Note that for this transform, by default ``noconds=True``.

    Examples
    ========

    >>> from sympy import hankel_transform, inverse_hankel_transform
    >>> from sympy import exp
    >>> from sympy.abc import r, k, m, nu, a

    >>> ht = hankel_transform(1/r**m, r, k, nu)
    >>> ht
    2*k**(m - 2)*gamma(-m/2 + nu/2 + 1)/(2**m*gamma(m/2 + nu/2))

    >>> inverse_hankel_transform(ht, k, r, nu)
    r**(-m)

    >>> ht = hankel_transform(exp(-a*r), r, k, 0)
    >>> ht
    a/(k**3*(a**2/k**2 + 1)**(3/2))

    >>> inverse_hankel_transform(ht, k, r, 0)
    exp(-a*r)

    See Also
    ========

    fourier_transform, inverse_fourier_transform
    sine_transform, inverse_sine_transform
    cosine_transform, inverse_cosine_transform
    hankel_transform
    mellin_transform, laplace_transform
    """
    return InverseHankelTransform(F, k, r, nu).doit(**hints)


##########################################################################
# Laplace Transform
##########################################################################

# Stub classes and functions that used to be here
import sympy.integrals.laplace as _laplace

LaplaceTransform = _laplace.LaplaceTransform
laplace_transform = _laplace.laplace_transform
laplace_correspondence = _laplace.laplace_correspondence
laplace_initial_conds = _laplace.laplace_initial_conds
InverseLaplaceTransform = _laplace.InverseLaplaceTransform
inverse_laplace_transform = _laplace.inverse_laplace_transform

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\runtime.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Runtime
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class ScriptId(str):
    '''
    Unique script identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ScriptId:
        return cls(json)

    def __repr__(self):
        return 'ScriptId({})'.format(super().__repr__())


@dataclass
class SerializationOptions:
    '''
    Represents options for serialization. Overrides ``generatePreview`` and ``returnByValue``.
    '''
    serialization: str

    #: Deep serialization depth. Default is full depth. Respected only in ``deep`` serialization mode.
    max_depth: typing.Optional[int] = None

    #: Embedder-specific parameters. For example if connected to V8 in Chrome these control DOM
    #: serialization via ``maxNodeDepth: integer`` and ``includeShadowTree: "none" `` "open" `` "all"``.
    #: Values can be only of type string or integer.
    additional_parameters: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['serialization'] = self.serialization
        if self.max_depth is not None:
            json['maxDepth'] = self.max_depth
        if self.additional_parameters is not None:
            json['additionalParameters'] = self.additional_parameters
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            serialization=str(json['serialization']),
            max_depth=int(json['maxDepth']) if 'maxDepth' in json else None,
            additional_parameters=dict(json['additionalParameters']) if 'additionalParameters' in json else None,
        )


@dataclass
class DeepSerializedValue:
    '''
    Represents deep serialized value.
    '''
    type_: str

    value: typing.Optional[typing.Any] = None

    object_id: typing.Optional[str] = None

    #: Set if value reference met more then once during serialization. In such
    #: case, value is provided only to one of the serialized values. Unique
    #: per value in the scope of one CDP call.
    weak_local_object_reference: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.value is not None:
            json['value'] = self.value
        if self.object_id is not None:
            json['objectId'] = self.object_id
        if self.weak_local_object_reference is not None:
            json['weakLocalObjectReference'] = self.weak_local_object_reference
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            value=json['value'] if 'value' in json else None,
            object_id=str(json['objectId']) if 'objectId' in json else None,
            weak_local_object_reference=int(json['weakLocalObjectReference']) if 'weakLocalObjectReference' in json else None,
        )


class RemoteObjectId(str):
    '''
    Unique object identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RemoteObjectId:
        return cls(json)

    def __repr__(self):
        return 'RemoteObjectId({})'.format(super().__repr__())


class UnserializableValue(str):
    '''
    Primitive value which cannot be JSON-stringified. Includes values ``-0``, ``NaN``, ``Infinity``,
    ``-Infinity``, and bigint literals.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> UnserializableValue:
        return cls(json)

    def __repr__(self):
        return 'UnserializableValue({})'.format(super().__repr__())


@dataclass
class RemoteObject:
    '''
    Mirror object referencing original JavaScript object.
    '''
    #: Object type.
    type_: str

    #: Object subtype hint. Specified for ``object`` type values only.
    #: NOTE: If you change anything here, make sure to also update
    #: ``subtype`` in ``ObjectPreview`` and ``PropertyPreview`` below.
    subtype: typing.Optional[str] = None

    #: Object class (constructor) name. Specified for ``object`` type values only.
    class_name: typing.Optional[str] = None

    #: Remote object value in case of primitive values or JSON values (if it was requested).
    value: typing.Optional[typing.Any] = None

    #: Primitive value which can not be JSON-stringified does not have ``value``, but gets this
    #: property.
    unserializable_value: typing.Optional[UnserializableValue] = None

    #: String representation of the object.
    description: typing.Optional[str] = None

    #: Deep serialized value.
    deep_serialized_value: typing.Optional[DeepSerializedValue] = None

    #: Unique object identifier (for non-primitive values).
    object_id: typing.Optional[RemoteObjectId] = None

    #: Preview containing abbreviated property values. Specified for ``object`` type values only.
    preview: typing.Optional[ObjectPreview] = None

    custom_preview: typing.Optional[CustomPreview] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.subtype is not None:
            json['subtype'] = self.subtype
        if self.class_name is not None:
            json['className'] = self.class_name
        if self.value is not None:
            json['value'] = self.value
        if self.unserializable_value is not None:
            json['unserializableValue'] = self.unserializable_value.to_json()
        if self.description is not None:
            json['description'] = self.description
        if self.deep_serialized_value is not None:
            json['deepSerializedValue'] = self.deep_serialized_value.to_json()
        if self.object_id is not None:
            json['objectId'] = self.object_id.to_json()
        if self.preview is not None:
            json['preview'] = self.preview.to_json()
        if self.custom_preview is not None:
            json['customPreview'] = self.custom_preview.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            subtype=str(json['subtype']) if 'subtype' in json else None,
            class_name=str(json['className']) if 'className' in json else None,
            value=json['value'] if 'value' in json else None,
            unserializable_value=UnserializableValue.from_json(json['unserializableValue']) if 'unserializableValue' in json else None,
            description=str(json['description']) if 'description' in json else None,
            deep_serialized_value=DeepSerializedValue.from_json(json['deepSerializedValue']) if 'deepSerializedValue' in json else None,
            object_id=RemoteObjectId.from_json(json['objectId']) if 'objectId' in json else None,
            preview=ObjectPreview.from_json(json['preview']) if 'preview' in json else None,
            custom_preview=CustomPreview.from_json(json['customPreview']) if 'customPreview' in json else None,
        )


@dataclass
class CustomPreview:
    #: The JSON-stringified result of formatter.header(object, config) call.
    #: It contains json ML array that represents RemoteObject.
    header: str

    #: If formatter returns true as a result of formatter.hasBody call then bodyGetterId will
    #: contain RemoteObjectId for the function that returns result of formatter.body(object, config) call.
    #: The result value is json ML array.
    body_getter_id: typing.Optional[RemoteObjectId] = None

    def to_json(self):
        json = dict()
        json['header'] = self.header
        if self.body_getter_id is not None:
            json['bodyGetterId'] = self.body_getter_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            header=str(json['header']),
            body_getter_id=RemoteObjectId.from_json(json['bodyGetterId']) if 'bodyGetterId' in json else None,
        )


@dataclass
class ObjectPreview:
    '''
    Object containing abbreviated remote object value.
    '''
    #: Object type.
    type_: str

    #: True iff some of the properties or entries of the original object did not fit.
    overflow: bool

    #: List of the properties.
    properties: typing.List[PropertyPreview]

    #: Object subtype hint. Specified for ``object`` type values only.
    subtype: typing.Optional[str] = None

    #: String representation of the object.
    description: typing.Optional[str] = None

    #: List of the entries. Specified for ``map`` and ``set`` subtype values only.
    entries: typing.Optional[typing.List[EntryPreview]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['overflow'] = self.overflow
        json['properties'] = [i.to_json() for i in self.properties]
        if self.subtype is not None:
            json['subtype'] = self.subtype
        if self.description is not None:
            json['description'] = self.description
        if self.entries is not None:
            json['entries'] = [i.to_json() for i in self.entries]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            overflow=bool(json['overflow']),
            properties=[PropertyPreview.from_json(i) for i in json['properties']],
            subtype=str(json['subtype']) if 'subtype' in json else None,
            description=str(json['description']) if 'description' in json else None,
            entries=[EntryPreview.from_json(i) for i in json['entries']] if 'entries' in json else None,
        )


@dataclass
class PropertyPreview:
    #: Property name.
    name: str

    #: Object type. Accessor means that the property itself is an accessor property.
    type_: str

    #: User-friendly property value string.
    value: typing.Optional[str] = None

    #: Nested value preview.
    value_preview: typing.Optional[ObjectPreview] = None

    #: Object subtype hint. Specified for ``object`` type values only.
    subtype: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['type'] = self.type_
        if self.value is not None:
            json['value'] = self.value
        if self.value_preview is not None:
            json['valuePreview'] = self.value_preview.to_json()
        if self.subtype is not None:
            json['subtype'] = self.subtype
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            type_=str(json['type']),
            value=str(json['value']) if 'value' in json else None,
            value_preview=ObjectPreview.from_json(json['valuePreview']) if 'valuePreview' in json else None,
            subtype=str(json['subtype']) if 'subtype' in json else None,
        )


@dataclass
class EntryPreview:
    #: Preview of the value.
    value: ObjectPreview

    #: Preview of the key. Specified for map-like collection entries.
    key: typing.Optional[ObjectPreview] = None

    def to_json(self):
        json = dict()
        json['value'] = self.value.to_json()
        if self.key is not None:
            json['key'] = self.key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=ObjectPreview.from_json(json['value']),
            key=ObjectPreview.from_json(json['key']) if 'key' in json else None,
        )


@dataclass
class PropertyDescriptor:
    '''
    Object property descriptor.
    '''
    #: Property name or symbol description.
    name: str

    #: True if the type of this property descriptor may be changed and if the property may be
    #: deleted from the corresponding object.
    configurable: bool

    #: True if this property shows up during enumeration of the properties on the corresponding
    #: object.
    enumerable: bool

    #: The value associated with the property.
    value: typing.Optional[RemoteObject] = None

    #: True if the value associated with the property may be changed (data descriptors only).
    writable: typing.Optional[bool] = None

    #: A function which serves as a getter for the property, or ``undefined`` if there is no getter
    #: (accessor descriptors only).
    get: typing.Optional[RemoteObject] = None

    #: A function which serves as a setter for the property, or ``undefined`` if there is no setter
    #: (accessor descriptors only).
    set_: typing.Optional[RemoteObject] = None

    #: True if the result was thrown during the evaluation.
    was_thrown: typing.Optional[bool] = None

    #: True if the property is owned for the object.
    is_own: typing.Optional[bool] = None

    #: Property symbol object, if the property is of the ``symbol`` type.
    symbol: typing.Optional[RemoteObject] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['configurable'] = self.configurable
        json['enumerable'] = self.enumerable
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.writable is not None:
            json['writable'] = self.writable
        if self.get is not None:
            json['get'] = self.get.to_json()
        if self.set_ is not None:
            json['set'] = self.set_.to_json()
        if self.was_thrown is not None:
            json['wasThrown'] = self.was_thrown
        if self.is_own is not None:
            json['isOwn'] = self.is_own
        if self.symbol is not None:
            json['symbol'] = self.symbol.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            configurable=bool(json['configurable']),
            enumerable=bool(json['enumerable']),
            value=RemoteObject.from_json(json['value']) if 'value' in json else None,
            writable=bool(json['writable']) if 'writable' in json else None,
            get=RemoteObject.from_json(json['get']) if 'get' in json else None,
            set_=RemoteObject.from_json(json['set']) if 'set' in json else None,
            was_thrown=bool(json['wasThrown']) if 'wasThrown' in json else None,
            is_own=bool(json['isOwn']) if 'isOwn' in json else None,
            symbol=RemoteObject.from_json(json['symbol']) if 'symbol' in json else None,
        )


@dataclass
class InternalPropertyDescriptor:
    '''
    Object internal property descriptor. This property isn't normally visible in JavaScript code.
    '''
    #: Conventional property name.
    name: str

    #: The value associated with the property.
    value: typing.Optional[RemoteObject] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        if self.value is not None:
            json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=RemoteObject.from_json(json['value']) if 'value' in json else None,
        )


@dataclass
class PrivatePropertyDescriptor:
    '''
    Object private field descriptor.
    '''
    #: Private property name.
    name: str

    #: The value associated with the private property.
    value: typing.Optional[RemoteObject] = None

    #: A function which serves as a getter for the private property,
    #: or ``undefined`` if there is no getter (accessor descriptors only).
    get: typing.Optional[RemoteObject] = None

    #: A function which serves as a setter for the private property,
    #: or ``undefined`` if there is no setter (accessor descriptors only).
    set_: typing.Optional[RemoteObject] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.get is not None:
            json['get'] = self.get.to_json()
        if self.set_ is not None:
            json['set'] = self.set_.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=RemoteObject.from_json(json['value']) if 'value' in json else None,
            get=RemoteObject.from_json(json['get']) if 'get' in json else None,
            set_=RemoteObject.from_json(json['set']) if 'set' in json else None,
        )


@dataclass
class CallArgument:
    '''
    Represents function call argument. Either remote object id ``objectId``, primitive ``value``,
    unserializable primitive value or neither of (for undefined) them should be specified.
    '''
    #: Primitive value or serializable javascript object.
    value: typing.Optional[typing.Any] = None

    #: Primitive value which can not be JSON-stringified.
    unserializable_value: typing.Optional[UnserializableValue] = None

    #: Remote object handle.
    object_id: typing.Optional[RemoteObjectId] = None

    def to_json(self):
        json = dict()
        if self.value is not None:
            json['value'] = self.value
        if self.unserializable_value is not None:
            json['unserializableValue'] = self.unserializable_value.to_json()
        if self.object_id is not None:
            json['objectId'] = self.object_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=json['value'] if 'value' in json else None,
            unserializable_value=UnserializableValue.from_json(json['unserializableValue']) if 'unserializableValue' in json else None,
            object_id=RemoteObjectId.from_json(json['objectId']) if 'objectId' in json else None,
        )


class ExecutionContextId(int):
    '''
    Id of an execution context.
    '''
    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> ExecutionContextId:
        return cls(json)

    def __repr__(self):
        return 'ExecutionContextId({})'.format(super().__repr__())


@dataclass
class ExecutionContextDescription:
    '''
    Description of an isolated world.
    '''
    #: Unique id of the execution context. It can be used to specify in which execution context
    #: script evaluation should be performed.
    id_: ExecutionContextId

    #: Execution context origin.
    origin: str

    #: Human readable name describing given context.
    name: str

    #: A system-unique execution context identifier. Unlike the id, this is unique across
    #: multiple processes, so can be reliably used to identify specific context while backend
    #: performs a cross-process navigation.
    unique_id: str

    #: Embedder-specific auxiliary data likely matching {isDefault: boolean, type: 'default'``'isolated'``'worker', frameId: string}
    aux_data: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['origin'] = self.origin
        json['name'] = self.name
        json['uniqueId'] = self.unique_id
        if self.aux_data is not None:
            json['auxData'] = self.aux_data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=ExecutionContextId.from_json(json['id']),
            origin=str(json['origin']),
            name=str(json['name']),
            unique_id=str(json['uniqueId']),
            aux_data=dict(json['auxData']) if 'auxData' in json else None,
        )


@dataclass
class ExceptionDetails:
    '''
    Detailed information about exception (or error) that was thrown during script compilation or
    execution.
    '''
    #: Exception id.
    exception_id: int

    #: Exception text, which should be used together with exception object when available.
    text: str

    #: Line number of the exception location (0-based).
    line_number: int

    #: Column number of the exception location (0-based).
    column_number: int

    #: Script ID of the exception location.
    script_id: typing.Optional[ScriptId] = None

    #: URL of the exception location, to be used when the script was not reported.
    url: typing.Optional[str] = None

    #: JavaScript stack trace if available.
    stack_trace: typing.Optional[StackTrace] = None

    #: Exception object if available.
    exception: typing.Optional[RemoteObject] = None

    #: Identifier of the context where exception happened.
    execution_context_id: typing.Optional[ExecutionContextId] = None

    #: Dictionary with entries of meta data that the client associated
    #: with this exception, such as information about associated network
    #: requests, etc.
    exception_meta_data: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['exceptionId'] = self.exception_id
        json['text'] = self.text
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        if self.script_id is not None:
            json['scriptId'] = self.script_id.to_json()
        if self.url is not None:
            json['url'] = self.url
        if self.stack_trace is not None:
            json['stackTrace'] = self.stack_trace.to_json()
        if self.exception is not None:
            json['exception'] = self.exception.to_json()
        if self.execution_context_id is not None:
            json['executionContextId'] = self.execution_context_id.to_json()
        if self.exception_meta_data is not None:
            json['exceptionMetaData'] = self.exception_meta_data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            exception_id=int(json['exceptionId']),
            text=str(json['text']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
            script_id=ScriptId.from_json(json['scriptId']) if 'scriptId' in json else None,
            url=str(json['url']) if 'url' in json else None,
            stack_trace=StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None,
            exception=RemoteObject.from_json(json['exception']) if 'exception' in json else None,
            execution_context_id=ExecutionContextId.from_json(json['executionContextId']) if 'executionContextId' in json else None,
            exception_meta_data=dict(json['exceptionMetaData']) if 'exceptionMetaData' in json else None,
        )


class Timestamp(float):
    '''
    Number of milliseconds since epoch.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> Timestamp:
        return cls(json)

    def __repr__(self):
        return 'Timestamp({})'.format(super().__repr__())


class TimeDelta(float):
    '''
    Number of milliseconds.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> TimeDelta:
        return cls(json)

    def __repr__(self):
        return 'TimeDelta({})'.format(super().__repr__())


@dataclass
class CallFrame:
    '''
    Stack entry for runtime errors and assertions.
    '''
    #: JavaScript function name.
    function_name: str

    #: JavaScript script id.
    script_id: ScriptId

    #: JavaScript script name or url.
    url: str

    #: JavaScript script line number (0-based).
    line_number: int

    #: JavaScript script column number (0-based).
    column_number: int

    def to_json(self):
        json = dict()
        json['functionName'] = self.function_name
        json['scriptId'] = self.script_id.to_json()
        json['url'] = self.url
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            function_name=str(json['functionName']),
            script_id=ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
        )


@dataclass
class StackTrace:
    '''
    Call frames for assertions or error messages.
    '''
    #: JavaScript function name.
    call_frames: typing.List[CallFrame]

    #: String label of this stack trace. For async traces this may be a name of the function that
    #: initiated the async call.
    description: typing.Optional[str] = None

    #: Asynchronous JavaScript stack trace that preceded this stack, if available.
    parent: typing.Optional[StackTrace] = None

    #: Asynchronous JavaScript stack trace that preceded this stack, if available.
    parent_id: typing.Optional[StackTraceId] = None

    def to_json(self):
        json = dict()
        json['callFrames'] = [i.to_json() for i in self.call_frames]
        if self.description is not None:
            json['description'] = self.description
        if self.parent is not None:
            json['parent'] = self.parent.to_json()
        if self.parent_id is not None:
            json['parentId'] = self.parent_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            call_frames=[CallFrame.from_json(i) for i in json['callFrames']],
            description=str(json['description']) if 'description' in json else None,
            parent=StackTrace.from_json(json['parent']) if 'parent' in json else None,
            parent_id=StackTraceId.from_json(json['parentId']) if 'parentId' in json else None,
        )


class UniqueDebuggerId(str):
    '''
    Unique identifier of current debugger.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> UniqueDebuggerId:
        return cls(json)

    def __repr__(self):
        return 'UniqueDebuggerId({})'.format(super().__repr__())


@dataclass
class StackTraceId:
    '''
    If ``debuggerId`` is set stack trace comes from another debugger and can be resolved there. This
    allows to track cross-debugger calls. See ``Runtime.StackTrace`` and ``Debugger.paused`` for usages.
    '''
    id_: str

    debugger_id: typing.Optional[UniqueDebuggerId] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        if self.debugger_id is not None:
            json['debuggerId'] = self.debugger_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=str(json['id']),
            debugger_id=UniqueDebuggerId.from_json(json['debuggerId']) if 'debuggerId' in json else None,
        )


def await_promise(
        promise_object_id: RemoteObjectId,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[RemoteObject, typing.Optional[ExceptionDetails]]]:
    '''
    Add handler to promise with given promise object id.

    :param promise_object_id: Identifier of the promise.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object that should be sent by value.
    :param generate_preview: *(Optional)* Whether preview should be generated for the result.
    :returns: A tuple with the following items:

        0. **result** - Promise result. Will contain rejected value if promise was rejected.
        1. **exceptionDetails** - *(Optional)* Exception details if stack strace is available.
    '''
    params: T_JSON_DICT = dict()
    params['promiseObjectId'] = promise_object_id.to_json()
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.awaitPromise',
        'params': params,
    }
    json = yield cmd_dict
    return (
        RemoteObject.from_json(json['result']),
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def call_function_on(
        function_declaration: str,
        object_id: typing.Optional[RemoteObjectId] = None,
        arguments: typing.Optional[typing.List[CallArgument]] = None,
        silent: typing.Optional[bool] = None,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        user_gesture: typing.Optional[bool] = None,
        await_promise: typing.Optional[bool] = None,
        execution_context_id: typing.Optional[ExecutionContextId] = None,
        object_group: typing.Optional[str] = None,
        throw_on_side_effect: typing.Optional[bool] = None,
        unique_context_id: typing.Optional[str] = None,
        serialization_options: typing.Optional[SerializationOptions] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[RemoteObject, typing.Optional[ExceptionDetails]]]:
    '''
    Calls function with given declaration on the given object. Object group of the result is
    inherited from the target object.

    :param function_declaration: Declaration of the function to call.
    :param object_id: *(Optional)* Identifier of the object to call function on. Either objectId or executionContextId should be specified.
    :param arguments: *(Optional)* Call arguments. All call arguments must belong to the same JavaScript world as the target object.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ```setPauseOnException```` state.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object which should be sent by value. Can be overriden by ````serializationOptions````.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the result.
    :param user_gesture: *(Optional)* Whether execution should be treated as initiated by user in the UI.
    :param await_promise: *(Optional)* Whether execution should ````await```` for resulting value and return once awaited promise is resolved.
    :param execution_context_id: *(Optional)* Specifies execution context which global object will be used to call function on. Either executionContextId or objectId should be specified.
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects. If objectGroup is not specified and objectId is, objectGroup will be inherited from object.
    :param throw_on_side_effect: **(EXPERIMENTAL)** *(Optional)* Whether to throw an exception if side effect cannot be ruled out during evaluation.
    :param unique_context_id: **(EXPERIMENTAL)** *(Optional)* An alternative way to specify the execution context to call function on. Compared to contextId that may be reused across processes, this is guaranteed to be system-unique, so it can be used to prevent accidental function call in context different than intended (e.g. as a result of navigation across process boundaries). This is mutually exclusive with ````executionContextId````.
    :param serialization_options: **(EXPERIMENTAL)** *(Optional)* Specifies the result serialization. If provided, overrides ````generatePreview```` and ````returnByValue```.
    :returns: A tuple with the following items:

        0. **result** - Call result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['functionDeclaration'] = function_declaration
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if arguments is not None:
        params['arguments'] = [i.to_json() for i in arguments]
    if silent is not None:
        params['silent'] = silent
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if user_gesture is not None:
        params['userGesture'] = user_gesture
    if await_promise is not None:
        params['awaitPromise'] = await_promise
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    if throw_on_side_effect is not None:
        params['throwOnSideEffect'] = throw_on_side_effect
    if unique_context_id is not None:
        params['uniqueContextId'] = unique_context_id
    if serialization_options is not None:
        params['serializationOptions'] = serialization_options.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.callFunctionOn',
        'params': params,
    }
    json = yield cmd_dict
    return (
        RemoteObject.from_json(json['result']),
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def compile_script(
        expression: str,
        source_url: str,
        persist_script: bool,
        execution_context_id: typing.Optional[ExecutionContextId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[ScriptId], typing.Optional[ExceptionDetails]]]:
    '''
    Compiles expression.

    :param expression: Expression to compile.
    :param source_url: Source url to be set for the script.
    :param persist_script: Specifies whether the compiled script should be persisted.
    :param execution_context_id: *(Optional)* Specifies in which execution context to perform script run. If the parameter is omitted the evaluation will be performed in the context of the inspected page.
    :returns: A tuple with the following items:

        0. **scriptId** - *(Optional)* Id of the script.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['expression'] = expression
    params['sourceURL'] = source_url
    params['persistScript'] = persist_script
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.compileScript',
        'params': params,
    }
    json = yield cmd_dict
    return (
        ScriptId.from_json(json['scriptId']) if 'scriptId' in json else None,
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables reporting of execution contexts creation.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.disable',
    }
    json = yield cmd_dict


def discard_console_entries() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Discards collected exceptions and console API calls.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.discardConsoleEntries',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables reporting of execution contexts creation by means of ``executionContextCreated`` event.
    When the reporting gets enabled the event will be sent immediately for each existing execution
    context.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.enable',
    }
    json = yield cmd_dict


def evaluate(
        expression: str,
        object_group: typing.Optional[str] = None,
        include_command_line_api: typing.Optional[bool] = None,
        silent: typing.Optional[bool] = None,
        context_id: typing.Optional[ExecutionContextId] = None,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        user_gesture: typing.Optional[bool] = None,
        await_promise: typing.Optional[bool] = None,
        throw_on_side_effect: typing.Optional[bool] = None,
        timeout: typing.Optional[TimeDelta] = None,
        disable_breaks: typing.Optional[bool] = None,
        repl_mode: typing.Optional[bool] = None,
        allow_unsafe_eval_blocked_by_csp: typing.Optional[bool] = None,
        unique_context_id: typing.Optional[str] = None,
        serialization_options: typing.Optional[SerializationOptions] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[RemoteObject, typing.Optional[ExceptionDetails]]]:
    '''
    Evaluates expression on global object.

    :param expression: Expression to evaluate.
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects.
    :param include_command_line_api: *(Optional)* Determines whether Command Line API should be available during the evaluation.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ```setPauseOnException```` state.
    :param context_id: *(Optional)* Specifies in which execution context to perform evaluation. If the parameter is omitted the evaluation will be performed in the context of the inspected page. This is mutually exclusive with ````uniqueContextId````, which offers an alternative way to identify the execution context that is more reliable in a multi-process environment.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object that should be sent by value.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the result.
    :param user_gesture: *(Optional)* Whether execution should be treated as initiated by user in the UI.
    :param await_promise: *(Optional)* Whether execution should ````await```` for resulting value and return once awaited promise is resolved.
    :param throw_on_side_effect: **(EXPERIMENTAL)** *(Optional)* Whether to throw an exception if side effect cannot be ruled out during evaluation. This implies ````disableBreaks```` below.
    :param timeout: **(EXPERIMENTAL)** *(Optional)* Terminate execution after timing out (number of milliseconds).
    :param disable_breaks: **(EXPERIMENTAL)** *(Optional)* Disable breakpoints during execution.
    :param repl_mode: **(EXPERIMENTAL)** *(Optional)* Setting this flag to true enables ````let```` re-declaration and top-level ````await````. Note that ````let```` variables can only be re-declared if they originate from ````replMode```` themselves.
    :param allow_unsafe_eval_blocked_by_csp: **(EXPERIMENTAL)** *(Optional)* The Content Security Policy (CSP) for the target might block 'unsafe-eval' which includes eval(), Function(), setTimeout() and setInterval() when called with non-callable arguments. This flag bypasses CSP for this evaluation and allows unsafe-eval. Defaults to true.
    :param unique_context_id: **(EXPERIMENTAL)** *(Optional)* An alternative way to specify the execution context to evaluate in. Compared to contextId that may be reused across processes, this is guaranteed to be system-unique, so it can be used to prevent accidental evaluation of the expression in context different than intended (e.g. as a result of navigation across process boundaries). This is mutually exclusive with ````contextId````.
    :param serialization_options: **(EXPERIMENTAL)** *(Optional)* Specifies the result serialization. If provided, overrides ````generatePreview```` and ````returnByValue```.
    :returns: A tuple with the following items:

        0. **result** - Evaluation result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['expression'] = expression
    if object_group is not None:
        params['objectGroup'] = object_group
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    if silent is not None:
        params['silent'] = silent
    if context_id is not None:
        params['contextId'] = context_id.to_json()
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if user_gesture is not None:
        params['userGesture'] = user_gesture
    if await_promise is not None:
        params['awaitPromise'] = await_promise
    if throw_on_side_effect is not None:
        params['throwOnSideEffect'] = throw_on_side_effect
    if timeout is not None:
        params['timeout'] = timeout.to_json()
    if disable_breaks is not None:
        params['disableBreaks'] = disable_breaks
    if repl_mode is not None:
        params['replMode'] = repl_mode
    if allow_unsafe_eval_blocked_by_csp is not None:
        params['allowUnsafeEvalBlockedByCSP'] = allow_unsafe_eval_blocked_by_csp
    if unique_context_id is not None:
        params['uniqueContextId'] = unique_context_id
    if serialization_options is not None:
        params['serializationOptions'] = serialization_options.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.evaluate',
        'params': params,
    }
    json = yield cmd_dict
    return (
        RemoteObject.from_json(json['result']),
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def get_isolate_id() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Returns the isolate id.

    **EXPERIMENTAL**

    :returns: The isolate id.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.getIsolateId',
    }
    json = yield cmd_dict
    return str(json['id'])


def get_heap_usage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[float, float, float, float]]:
    '''
    Returns the JavaScript heap usage.
    It is the total usage of the corresponding isolate not scoped to a particular Runtime.

    **EXPERIMENTAL**

    :returns: A tuple with the following items:

        0. **usedSize** - Used JavaScript heap size in bytes.
        1. **totalSize** - Allocated JavaScript heap size in bytes.
        2. **embedderHeapUsedSize** - Used size in bytes in the embedder's garbage-collected heap.
        3. **backingStorageSize** - Size in bytes of backing storage for array buffers and external strings.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.getHeapUsage',
    }
    json = yield cmd_dict
    return (
        float(json['usedSize']),
        float(json['totalSize']),
        float(json['embedderHeapUsedSize']),
        float(json['backingStorageSize'])
    )


def get_properties(
        object_id: RemoteObjectId,
        own_properties: typing.Optional[bool] = None,
        accessor_properties_only: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        non_indexed_properties_only: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[PropertyDescriptor], typing.Optional[typing.List[InternalPropertyDescriptor]], typing.Optional[typing.List[PrivatePropertyDescriptor]], typing.Optional[ExceptionDetails]]]:
    '''
    Returns properties of a given object. Object group of the result is inherited from the target
    object.

    :param object_id: Identifier of the object to return properties for.
    :param own_properties: *(Optional)* If true, returns properties belonging only to the element itself, not to its prototype chain.
    :param accessor_properties_only: **(EXPERIMENTAL)** *(Optional)* If true, returns accessor properties (with getter/setter) only; internal properties are not returned either.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the results.
    :param non_indexed_properties_only: **(EXPERIMENTAL)** *(Optional)* If true, returns non-indexed properties only.
    :returns: A tuple with the following items:

        0. **result** - Object properties.
        1. **internalProperties** - *(Optional)* Internal object properties (only of the element itself).
        2. **privateProperties** - *(Optional)* Object private properties.
        3. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if own_properties is not None:
        params['ownProperties'] = own_properties
    if accessor_properties_only is not None:
        params['accessorPropertiesOnly'] = accessor_properties_only
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if non_indexed_properties_only is not None:
        params['nonIndexedPropertiesOnly'] = non_indexed_properties_only
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.getProperties',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [PropertyDescriptor.from_json(i) for i in json['result']],
        [InternalPropertyDescriptor.from_json(i) for i in json['internalProperties']] if 'internalProperties' in json else None,
        [PrivatePropertyDescriptor.from_json(i) for i in json['privateProperties']] if 'privateProperties' in json else None,
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def global_lexical_scope_names(
        execution_context_id: typing.Optional[ExecutionContextId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Returns all let, const and class variables from global scope.

    :param execution_context_id: *(Optional)* Specifies in which execution context to lookup global scope variables.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.globalLexicalScopeNames',
        'params': params,
    }
    json = yield cmd_dict
    return [str(i) for i in json['names']]


def query_objects(
        prototype_object_id: RemoteObjectId,
        object_group: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,RemoteObject]:
    '''
    :param prototype_object_id: Identifier of the prototype to return objects for.
    :param object_group: *(Optional)* Symbolic group name that can be used to release the results.
    :returns: Array with objects.
    '''
    params: T_JSON_DICT = dict()
    params['prototypeObjectId'] = prototype_object_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.queryObjects',
        'params': params,
    }
    json = yield cmd_dict
    return RemoteObject.from_json(json['objects'])


def release_object(
        object_id: RemoteObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Releases remote object with given id.

    :param object_id: Identifier of the object to release.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.releaseObject',
        'params': params,
    }
    json = yield cmd_dict


def release_object_group(
        object_group: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Releases all remote objects that belong to a given group.

    :param object_group: Symbolic object group name.
    '''
    params: T_JSON_DICT = dict()
    params['objectGroup'] = object_group
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.releaseObjectGroup',
        'params': params,
    }
    json = yield cmd_dict


def run_if_waiting_for_debugger() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Tells inspected instance to run if it was waiting for debugger to attach.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.runIfWaitingForDebugger',
    }
    json = yield cmd_dict


def run_script(
        script_id: ScriptId,
        execution_context_id: typing.Optional[ExecutionContextId] = None,
        object_group: typing.Optional[str] = None,
        silent: typing.Optional[bool] = None,
        include_command_line_api: typing.Optional[bool] = None,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        await_promise: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[RemoteObject, typing.Optional[ExceptionDetails]]]:
    '''
    Runs script with given id in a given context.

    :param script_id: Id of the script to run.
    :param execution_context_id: *(Optional)* Specifies in which execution context to perform script run. If the parameter is omitted the evaluation will be performed in the context of the inspected page.
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ```setPauseOnException```` state.
    :param include_command_line_api: *(Optional)* Determines whether Command Line API should be available during the evaluation.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object which should be sent by value.
    :param generate_preview: *(Optional)* Whether preview should be generated for the result.
    :param await_promise: *(Optional)* Whether execution should ````await``` for resulting value and return once awaited promise is resolved.
    :returns: A tuple with the following items:

        0. **result** - Run result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    if silent is not None:
        params['silent'] = silent
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if await_promise is not None:
        params['awaitPromise'] = await_promise
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.runScript',
        'params': params,
    }
    json = yield cmd_dict
    return (
        RemoteObject.from_json(json['result']),
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def set_async_call_stack_depth(
        max_depth: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables or disables async call stacks tracking.

    :param max_depth: Maximum depth of async call stacks. Setting to ```0``` will effectively disable collecting async call stacks (default).
    '''
    params: T_JSON_DICT = dict()
    params['maxDepth'] = max_depth
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.setAsyncCallStackDepth',
        'params': params,
    }
    json = yield cmd_dict


def set_custom_object_formatter_enabled(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param enabled:
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.setCustomObjectFormatterEnabled',
        'params': params,
    }
    json = yield cmd_dict


def set_max_call_stack_size_to_capture(
        size: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param size:
    '''
    params: T_JSON_DICT = dict()
    params['size'] = size
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.setMaxCallStackSizeToCapture',
        'params': params,
    }
    json = yield cmd_dict


def terminate_execution() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Terminate current or next JavaScript execution.
    Will cancel the termination when the outer-most script execution ends.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.terminateExecution',
    }
    json = yield cmd_dict


def add_binding(
        name: str,
        execution_context_id: typing.Optional[ExecutionContextId] = None,
        execution_context_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    If executionContextId is empty, adds binding with the given name on the
    global objects of all inspected contexts, including those created later,
    bindings survive reloads.
    Binding function takes exactly one argument, this argument should be string,
    in case of any other input, function throws an exception.
    Each binding function call produces Runtime.bindingCalled notification.

    :param name:
    :param execution_context_id: **(EXPERIMENTAL)** *(Optional)* If specified, the binding would only be exposed to the specified execution context. If omitted and ```executionContextName```` is not set, the binding is exposed to all execution contexts of the target. This parameter is mutually exclusive with ````executionContextName````. Deprecated in favor of ````executionContextName```` due to an unclear use case and bugs in implementation (crbug.com/1169639). ````executionContextId```` will be removed in the future.
    :param execution_context_name: *(Optional)* If specified, the binding is exposed to the executionContext with matching name, even for contexts created after the binding is added. See also ````ExecutionContext.name```` and ````worldName```` parameter to ````Page.addScriptToEvaluateOnNewDocument````. This parameter is mutually exclusive with ````executionContextId```.
    '''
    params: T_JSON_DICT = dict()
    params['name'] = name
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    if execution_context_name is not None:
        params['executionContextName'] = execution_context_name
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.addBinding',
        'params': params,
    }
    json = yield cmd_dict


def remove_binding(
        name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    This method does not remove binding function from global object but
    unsubscribes current runtime agent from Runtime.bindingCalled notifications.

    :param name:
    '''
    params: T_JSON_DICT = dict()
    params['name'] = name
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.removeBinding',
        'params': params,
    }
    json = yield cmd_dict


def get_exception_details(
        error_object_id: RemoteObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Optional[ExceptionDetails]]:
    '''
    This method tries to lookup and populate exception details for a
    JavaScript Error object.
    Note that the stackTrace portion of the resulting exceptionDetails will
    only be populated if the Runtime domain was enabled at the time when the
    Error was thrown.

    **EXPERIMENTAL**

    :param error_object_id: The error object for which to resolve the exception details.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['errorObjectId'] = error_object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.getExceptionDetails',
        'params': params,
    }
    json = yield cmd_dict
    return ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None


@event_class('Runtime.bindingCalled')
@dataclass
class BindingCalled:
    '''
    **EXPERIMENTAL**

    Notification is issued every time when binding is called.
    '''
    name: str
    payload: str
    #: Identifier of the context where the call was made.
    execution_context_id: ExecutionContextId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BindingCalled:
        return cls(
            name=str(json['name']),
            payload=str(json['payload']),
            execution_context_id=ExecutionContextId.from_json(json['executionContextId'])
        )


@event_class('Runtime.consoleAPICalled')
@dataclass
class ConsoleAPICalled:
    '''
    Issued when console API was called.
    '''
    #: Type of the call.
    type_: str
    #: Call arguments.
    args: typing.List[RemoteObject]
    #: Identifier of the context where the call was made.
    execution_context_id: ExecutionContextId
    #: Call timestamp.
    timestamp: Timestamp
    #: Stack trace captured when the call was made. The async stack chain is automatically reported for
    #: the following call types: ``assert``, ``error``, ``trace``, ``warning``. For other types the async call
    #: chain can be retrieved using ``Debugger.getStackTrace`` and ``stackTrace.parentId`` field.
    stack_trace: typing.Optional[StackTrace]
    #: Console context descriptor for calls on non-default console context (not console.*):
    #: 'anonymous#unique-logger-id' for call on unnamed context, 'name#unique-logger-id' for call
    #: on named context.
    context: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleAPICalled:
        return cls(
            type_=str(json['type']),
            args=[RemoteObject.from_json(i) for i in json['args']],
            execution_context_id=ExecutionContextId.from_json(json['executionContextId']),
            timestamp=Timestamp.from_json(json['timestamp']),
            stack_trace=StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None,
            context=str(json['context']) if 'context' in json else None
        )


@event_class('Runtime.exceptionRevoked')
@dataclass
class ExceptionRevoked:
    '''
    Issued when unhandled exception was revoked.
    '''
    #: Reason describing why exception was revoked.
    reason: str
    #: The id of revoked exception, as reported in ``exceptionThrown``.
    exception_id: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExceptionRevoked:
        return cls(
            reason=str(json['reason']),
            exception_id=int(json['exceptionId'])
        )


@event_class('Runtime.exceptionThrown')
@dataclass
class ExceptionThrown:
    '''
    Issued when exception was thrown and unhandled.
    '''
    #: Timestamp of the exception.
    timestamp: Timestamp
    exception_details: ExceptionDetails

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExceptionThrown:
        return cls(
            timestamp=Timestamp.from_json(json['timestamp']),
            exception_details=ExceptionDetails.from_json(json['exceptionDetails'])
        )


@event_class('Runtime.executionContextCreated')
@dataclass
class ExecutionContextCreated:
    '''
    Issued when new execution context is created.
    '''
    #: A newly created execution context.
    context: ExecutionContextDescription

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExecutionContextCreated:
        return cls(
            context=ExecutionContextDescription.from_json(json['context'])
        )


@event_class('Runtime.executionContextDestroyed')
@dataclass
class ExecutionContextDestroyed:
    '''
    Issued when execution context is destroyed.
    '''
    #: Id of the destroyed context
    execution_context_id: ExecutionContextId
    #: Unique Id of the destroyed context
    execution_context_unique_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExecutionContextDestroyed:
        return cls(
            execution_context_id=ExecutionContextId.from_json(json['executionContextId']),
            execution_context_unique_id=str(json['executionContextUniqueId'])
        )


@event_class('Runtime.executionContextsCleared')
@dataclass
class ExecutionContextsCleared:
    '''
    Issued when all executionContexts were cleared in browser
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExecutionContextsCleared:
        return cls(

        )


@event_class('Runtime.inspectRequested')
@dataclass
class InspectRequested:
    '''
    Issued when object should be inspected (for example, as a result of inspect() command line API
    call).
    '''
    object_: RemoteObject
    hints: dict
    #: Identifier of the context where the call was made.
    execution_context_id: typing.Optional[ExecutionContextId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InspectRequested:
        return cls(
            object_=RemoteObject.from_json(json['object']),
            hints=dict(json['hints']),
            execution_context_id=ExecutionContextId.from_json(json['executionContextId']) if 'executionContextId' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\runtime.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Runtime
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class ScriptId(str):
    '''
    Unique script identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ScriptId:
        return cls(json)

    def __repr__(self):
        return 'ScriptId({})'.format(super().__repr__())


@dataclass
class SerializationOptions:
    '''
    Represents options for serialization. Overrides ``generatePreview`` and ``returnByValue``.
    '''
    serialization: str

    #: Deep serialization depth. Default is full depth. Respected only in ``deep`` serialization mode.
    max_depth: typing.Optional[int] = None

    #: Embedder-specific parameters. For example if connected to V8 in Chrome these control DOM
    #: serialization via ``maxNodeDepth: integer`` and ``includeShadowTree: "none" `` "open" `` "all"``.
    #: Values can be only of type string or integer.
    additional_parameters: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['serialization'] = self.serialization
        if self.max_depth is not None:
            json['maxDepth'] = self.max_depth
        if self.additional_parameters is not None:
            json['additionalParameters'] = self.additional_parameters
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            serialization=str(json['serialization']),
            max_depth=int(json['maxDepth']) if 'maxDepth' in json else None,
            additional_parameters=dict(json['additionalParameters']) if 'additionalParameters' in json else None,
        )


@dataclass
class DeepSerializedValue:
    '''
    Represents deep serialized value.
    '''
    type_: str

    value: typing.Optional[typing.Any] = None

    object_id: typing.Optional[str] = None

    #: Set if value reference met more then once during serialization. In such
    #: case, value is provided only to one of the serialized values. Unique
    #: per value in the scope of one CDP call.
    weak_local_object_reference: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.value is not None:
            json['value'] = self.value
        if self.object_id is not None:
            json['objectId'] = self.object_id
        if self.weak_local_object_reference is not None:
            json['weakLocalObjectReference'] = self.weak_local_object_reference
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            value=json['value'] if 'value' in json else None,
            object_id=str(json['objectId']) if 'objectId' in json else None,
            weak_local_object_reference=int(json['weakLocalObjectReference']) if 'weakLocalObjectReference' in json else None,
        )


class RemoteObjectId(str):
    '''
    Unique object identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RemoteObjectId:
        return cls(json)

    def __repr__(self):
        return 'RemoteObjectId({})'.format(super().__repr__())


class UnserializableValue(str):
    '''
    Primitive value which cannot be JSON-stringified. Includes values ``-0``, ``NaN``, ``Infinity``,
    ``-Infinity``, and bigint literals.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> UnserializableValue:
        return cls(json)

    def __repr__(self):
        return 'UnserializableValue({})'.format(super().__repr__())


@dataclass
class RemoteObject:
    '''
    Mirror object referencing original JavaScript object.
    '''
    #: Object type.
    type_: str

    #: Object subtype hint. Specified for ``object`` type values only.
    #: NOTE: If you change anything here, make sure to also update
    #: ``subtype`` in ``ObjectPreview`` and ``PropertyPreview`` below.
    subtype: typing.Optional[str] = None

    #: Object class (constructor) name. Specified for ``object`` type values only.
    class_name: typing.Optional[str] = None

    #: Remote object value in case of primitive values or JSON values (if it was requested).
    value: typing.Optional[typing.Any] = None

    #: Primitive value which can not be JSON-stringified does not have ``value``, but gets this
    #: property.
    unserializable_value: typing.Optional[UnserializableValue] = None

    #: String representation of the object.
    description: typing.Optional[str] = None

    #: Deep serialized value.
    deep_serialized_value: typing.Optional[DeepSerializedValue] = None

    #: Unique object identifier (for non-primitive values).
    object_id: typing.Optional[RemoteObjectId] = None

    #: Preview containing abbreviated property values. Specified for ``object`` type values only.
    preview: typing.Optional[ObjectPreview] = None

    custom_preview: typing.Optional[CustomPreview] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.subtype is not None:
            json['subtype'] = self.subtype
        if self.class_name is not None:
            json['className'] = self.class_name
        if self.value is not None:
            json['value'] = self.value
        if self.unserializable_value is not None:
            json['unserializableValue'] = self.unserializable_value.to_json()
        if self.description is not None:
            json['description'] = self.description
        if self.deep_serialized_value is not None:
            json['deepSerializedValue'] = self.deep_serialized_value.to_json()
        if self.object_id is not None:
            json['objectId'] = self.object_id.to_json()
        if self.preview is not None:
            json['preview'] = self.preview.to_json()
        if self.custom_preview is not None:
            json['customPreview'] = self.custom_preview.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            subtype=str(json['subtype']) if 'subtype' in json else None,
            class_name=str(json['className']) if 'className' in json else None,
            value=json['value'] if 'value' in json else None,
            unserializable_value=UnserializableValue.from_json(json['unserializableValue']) if 'unserializableValue' in json else None,
            description=str(json['description']) if 'description' in json else None,
            deep_serialized_value=DeepSerializedValue.from_json(json['deepSerializedValue']) if 'deepSerializedValue' in json else None,
            object_id=RemoteObjectId.from_json(json['objectId']) if 'objectId' in json else None,
            preview=ObjectPreview.from_json(json['preview']) if 'preview' in json else None,
            custom_preview=CustomPreview.from_json(json['customPreview']) if 'customPreview' in json else None,
        )


@dataclass
class CustomPreview:
    #: The JSON-stringified result of formatter.header(object, config) call.
    #: It contains json ML array that represents RemoteObject.
    header: str

    #: If formatter returns true as a result of formatter.hasBody call then bodyGetterId will
    #: contain RemoteObjectId for the function that returns result of formatter.body(object, config) call.
    #: The result value is json ML array.
    body_getter_id: typing.Optional[RemoteObjectId] = None

    def to_json(self):
        json = dict()
        json['header'] = self.header
        if self.body_getter_id is not None:
            json['bodyGetterId'] = self.body_getter_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            header=str(json['header']),
            body_getter_id=RemoteObjectId.from_json(json['bodyGetterId']) if 'bodyGetterId' in json else None,
        )


@dataclass
class ObjectPreview:
    '''
    Object containing abbreviated remote object value.
    '''
    #: Object type.
    type_: str

    #: True iff some of the properties or entries of the original object did not fit.
    overflow: bool

    #: List of the properties.
    properties: typing.List[PropertyPreview]

    #: Object subtype hint. Specified for ``object`` type values only.
    subtype: typing.Optional[str] = None

    #: String representation of the object.
    description: typing.Optional[str] = None

    #: List of the entries. Specified for ``map`` and ``set`` subtype values only.
    entries: typing.Optional[typing.List[EntryPreview]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['overflow'] = self.overflow
        json['properties'] = [i.to_json() for i in self.properties]
        if self.subtype is not None:
            json['subtype'] = self.subtype
        if self.description is not None:
            json['description'] = self.description
        if self.entries is not None:
            json['entries'] = [i.to_json() for i in self.entries]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            overflow=bool(json['overflow']),
            properties=[PropertyPreview.from_json(i) for i in json['properties']],
            subtype=str(json['subtype']) if 'subtype' in json else None,
            description=str(json['description']) if 'description' in json else None,
            entries=[EntryPreview.from_json(i) for i in json['entries']] if 'entries' in json else None,
        )


@dataclass
class PropertyPreview:
    #: Property name.
    name: str

    #: Object type. Accessor means that the property itself is an accessor property.
    type_: str

    #: User-friendly property value string.
    value: typing.Optional[str] = None

    #: Nested value preview.
    value_preview: typing.Optional[ObjectPreview] = None

    #: Object subtype hint. Specified for ``object`` type values only.
    subtype: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['type'] = self.type_
        if self.value is not None:
            json['value'] = self.value
        if self.value_preview is not None:
            json['valuePreview'] = self.value_preview.to_json()
        if self.subtype is not None:
            json['subtype'] = self.subtype
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            type_=str(json['type']),
            value=str(json['value']) if 'value' in json else None,
            value_preview=ObjectPreview.from_json(json['valuePreview']) if 'valuePreview' in json else None,
            subtype=str(json['subtype']) if 'subtype' in json else None,
        )


@dataclass
class EntryPreview:
    #: Preview of the value.
    value: ObjectPreview

    #: Preview of the key. Specified for map-like collection entries.
    key: typing.Optional[ObjectPreview] = None

    def to_json(self):
        json = dict()
        json['value'] = self.value.to_json()
        if self.key is not None:
            json['key'] = self.key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=ObjectPreview.from_json(json['value']),
            key=ObjectPreview.from_json(json['key']) if 'key' in json else None,
        )


@dataclass
class PropertyDescriptor:
    '''
    Object property descriptor.
    '''
    #: Property name or symbol description.
    name: str

    #: True if the type of this property descriptor may be changed and if the property may be
    #: deleted from the corresponding object.
    configurable: bool

    #: True if this property shows up during enumeration of the properties on the corresponding
    #: object.
    enumerable: bool

    #: The value associated with the property.
    value: typing.Optional[RemoteObject] = None

    #: True if the value associated with the property may be changed (data descriptors only).
    writable: typing.Optional[bool] = None

    #: A function which serves as a getter for the property, or ``undefined`` if there is no getter
    #: (accessor descriptors only).
    get: typing.Optional[RemoteObject] = None

    #: A function which serves as a setter for the property, or ``undefined`` if there is no setter
    #: (accessor descriptors only).
    set_: typing.Optional[RemoteObject] = None

    #: True if the result was thrown during the evaluation.
    was_thrown: typing.Optional[bool] = None

    #: True if the property is owned for the object.
    is_own: typing.Optional[bool] = None

    #: Property symbol object, if the property is of the ``symbol`` type.
    symbol: typing.Optional[RemoteObject] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['configurable'] = self.configurable
        json['enumerable'] = self.enumerable
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.writable is not None:
            json['writable'] = self.writable
        if self.get is not None:
            json['get'] = self.get.to_json()
        if self.set_ is not None:
            json['set'] = self.set_.to_json()
        if self.was_thrown is not None:
            json['wasThrown'] = self.was_thrown
        if self.is_own is not None:
            json['isOwn'] = self.is_own
        if self.symbol is not None:
            json['symbol'] = self.symbol.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            configurable=bool(json['configurable']),
            enumerable=bool(json['enumerable']),
            value=RemoteObject.from_json(json['value']) if 'value' in json else None,
            writable=bool(json['writable']) if 'writable' in json else None,
            get=RemoteObject.from_json(json['get']) if 'get' in json else None,
            set_=RemoteObject.from_json(json['set']) if 'set' in json else None,
            was_thrown=bool(json['wasThrown']) if 'wasThrown' in json else None,
            is_own=bool(json['isOwn']) if 'isOwn' in json else None,
            symbol=RemoteObject.from_json(json['symbol']) if 'symbol' in json else None,
        )


@dataclass
class InternalPropertyDescriptor:
    '''
    Object internal property descriptor. This property isn't normally visible in JavaScript code.
    '''
    #: Conventional property name.
    name: str

    #: The value associated with the property.
    value: typing.Optional[RemoteObject] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        if self.value is not None:
            json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=RemoteObject.from_json(json['value']) if 'value' in json else None,
        )


@dataclass
class PrivatePropertyDescriptor:
    '''
    Object private field descriptor.
    '''
    #: Private property name.
    name: str

    #: The value associated with the private property.
    value: typing.Optional[RemoteObject] = None

    #: A function which serves as a getter for the private property,
    #: or ``undefined`` if there is no getter (accessor descriptors only).
    get: typing.Optional[RemoteObject] = None

    #: A function which serves as a setter for the private property,
    #: or ``undefined`` if there is no setter (accessor descriptors only).
    set_: typing.Optional[RemoteObject] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        if self.value is not None:
            json['value'] = self.value.to_json()
        if self.get is not None:
            json['get'] = self.get.to_json()
        if self.set_ is not None:
            json['set'] = self.set_.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=RemoteObject.from_json(json['value']) if 'value' in json else None,
            get=RemoteObject.from_json(json['get']) if 'get' in json else None,
            set_=RemoteObject.from_json(json['set']) if 'set' in json else None,
        )


@dataclass
class CallArgument:
    '''
    Represents function call argument. Either remote object id ``objectId``, primitive ``value``,
    unserializable primitive value or neither of (for undefined) them should be specified.
    '''
    #: Primitive value or serializable javascript object.
    value: typing.Optional[typing.Any] = None

    #: Primitive value which can not be JSON-stringified.
    unserializable_value: typing.Optional[UnserializableValue] = None

    #: Remote object handle.
    object_id: typing.Optional[RemoteObjectId] = None

    def to_json(self):
        json = dict()
        if self.value is not None:
            json['value'] = self.value
        if self.unserializable_value is not None:
            json['unserializableValue'] = self.unserializable_value.to_json()
        if self.object_id is not None:
            json['objectId'] = self.object_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=json['value'] if 'value' in json else None,
            unserializable_value=UnserializableValue.from_json(json['unserializableValue']) if 'unserializableValue' in json else None,
            object_id=RemoteObjectId.from_json(json['objectId']) if 'objectId' in json else None,
        )


class ExecutionContextId(int):
    '''
    Id of an execution context.
    '''
    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> ExecutionContextId:
        return cls(json)

    def __repr__(self):
        return 'ExecutionContextId({})'.format(super().__repr__())


@dataclass
class ExecutionContextDescription:
    '''
    Description of an isolated world.
    '''
    #: Unique id of the execution context. It can be used to specify in which execution context
    #: script evaluation should be performed.
    id_: ExecutionContextId

    #: Execution context origin.
    origin: str

    #: Human readable name describing given context.
    name: str

    #: A system-unique execution context identifier. Unlike the id, this is unique across
    #: multiple processes, so can be reliably used to identify specific context while backend
    #: performs a cross-process navigation.
    unique_id: str

    #: Embedder-specific auxiliary data likely matching {isDefault: boolean, type: 'default'``'isolated'``'worker', frameId: string}
    aux_data: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['origin'] = self.origin
        json['name'] = self.name
        json['uniqueId'] = self.unique_id
        if self.aux_data is not None:
            json['auxData'] = self.aux_data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=ExecutionContextId.from_json(json['id']),
            origin=str(json['origin']),
            name=str(json['name']),
            unique_id=str(json['uniqueId']),
            aux_data=dict(json['auxData']) if 'auxData' in json else None,
        )


@dataclass
class ExceptionDetails:
    '''
    Detailed information about exception (or error) that was thrown during script compilation or
    execution.
    '''
    #: Exception id.
    exception_id: int

    #: Exception text, which should be used together with exception object when available.
    text: str

    #: Line number of the exception location (0-based).
    line_number: int

    #: Column number of the exception location (0-based).
    column_number: int

    #: Script ID of the exception location.
    script_id: typing.Optional[ScriptId] = None

    #: URL of the exception location, to be used when the script was not reported.
    url: typing.Optional[str] = None

    #: JavaScript stack trace if available.
    stack_trace: typing.Optional[StackTrace] = None

    #: Exception object if available.
    exception: typing.Optional[RemoteObject] = None

    #: Identifier of the context where exception happened.
    execution_context_id: typing.Optional[ExecutionContextId] = None

    #: Dictionary with entries of meta data that the client associated
    #: with this exception, such as information about associated network
    #: requests, etc.
    exception_meta_data: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['exceptionId'] = self.exception_id
        json['text'] = self.text
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        if self.script_id is not None:
            json['scriptId'] = self.script_id.to_json()
        if self.url is not None:
            json['url'] = self.url
        if self.stack_trace is not None:
            json['stackTrace'] = self.stack_trace.to_json()
        if self.exception is not None:
            json['exception'] = self.exception.to_json()
        if self.execution_context_id is not None:
            json['executionContextId'] = self.execution_context_id.to_json()
        if self.exception_meta_data is not None:
            json['exceptionMetaData'] = self.exception_meta_data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            exception_id=int(json['exceptionId']),
            text=str(json['text']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
            script_id=ScriptId.from_json(json['scriptId']) if 'scriptId' in json else None,
            url=str(json['url']) if 'url' in json else None,
            stack_trace=StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None,
            exception=RemoteObject.from_json(json['exception']) if 'exception' in json else None,
            execution_context_id=ExecutionContextId.from_json(json['executionContextId']) if 'executionContextId' in json else None,
            exception_meta_data=dict(json['exceptionMetaData']) if 'exceptionMetaData' in json else None,
        )


class Timestamp(float):
    '''
    Number of milliseconds since epoch.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> Timestamp:
        return cls(json)

    def __repr__(self):
        return 'Timestamp({})'.format(super().__repr__())


class TimeDelta(float):
    '''
    Number of milliseconds.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> TimeDelta:
        return cls(json)

    def __repr__(self):
        return 'TimeDelta({})'.format(super().__repr__())


@dataclass
class CallFrame:
    '''
    Stack entry for runtime errors and assertions.
    '''
    #: JavaScript function name.
    function_name: str

    #: JavaScript script id.
    script_id: ScriptId

    #: JavaScript script name or url.
    url: str

    #: JavaScript script line number (0-based).
    line_number: int

    #: JavaScript script column number (0-based).
    column_number: int

    def to_json(self):
        json = dict()
        json['functionName'] = self.function_name
        json['scriptId'] = self.script_id.to_json()
        json['url'] = self.url
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            function_name=str(json['functionName']),
            script_id=ScriptId.from_json(json['scriptId']),
            url=str(json['url']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
        )


@dataclass
class StackTrace:
    '''
    Call frames for assertions or error messages.
    '''
    #: JavaScript function name.
    call_frames: typing.List[CallFrame]

    #: String label of this stack trace. For async traces this may be a name of the function that
    #: initiated the async call.
    description: typing.Optional[str] = None

    #: Asynchronous JavaScript stack trace that preceded this stack, if available.
    parent: typing.Optional[StackTrace] = None

    #: Asynchronous JavaScript stack trace that preceded this stack, if available.
    parent_id: typing.Optional[StackTraceId] = None

    def to_json(self):
        json = dict()
        json['callFrames'] = [i.to_json() for i in self.call_frames]
        if self.description is not None:
            json['description'] = self.description
        if self.parent is not None:
            json['parent'] = self.parent.to_json()
        if self.parent_id is not None:
            json['parentId'] = self.parent_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            call_frames=[CallFrame.from_json(i) for i in json['callFrames']],
            description=str(json['description']) if 'description' in json else None,
            parent=StackTrace.from_json(json['parent']) if 'parent' in json else None,
            parent_id=StackTraceId.from_json(json['parentId']) if 'parentId' in json else None,
        )


class UniqueDebuggerId(str):
    '''
    Unique identifier of current debugger.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> UniqueDebuggerId:
        return cls(json)

    def __repr__(self):
        return 'UniqueDebuggerId({})'.format(super().__repr__())


@dataclass
class StackTraceId:
    '''
    If ``debuggerId`` is set stack trace comes from another debugger and can be resolved there. This
    allows to track cross-debugger calls. See ``Runtime.StackTrace`` and ``Debugger.paused`` for usages.
    '''
    id_: str

    debugger_id: typing.Optional[UniqueDebuggerId] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        if self.debugger_id is not None:
            json['debuggerId'] = self.debugger_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=str(json['id']),
            debugger_id=UniqueDebuggerId.from_json(json['debuggerId']) if 'debuggerId' in json else None,
        )


def await_promise(
        promise_object_id: RemoteObjectId,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[RemoteObject, typing.Optional[ExceptionDetails]]]:
    '''
    Add handler to promise with given promise object id.

    :param promise_object_id: Identifier of the promise.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object that should be sent by value.
    :param generate_preview: *(Optional)* Whether preview should be generated for the result.
    :returns: A tuple with the following items:

        0. **result** - Promise result. Will contain rejected value if promise was rejected.
        1. **exceptionDetails** - *(Optional)* Exception details if stack strace is available.
    '''
    params: T_JSON_DICT = dict()
    params['promiseObjectId'] = promise_object_id.to_json()
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.awaitPromise',
        'params': params,
    }
    json = yield cmd_dict
    return (
        RemoteObject.from_json(json['result']),
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def call_function_on(
        function_declaration: str,
        object_id: typing.Optional[RemoteObjectId] = None,
        arguments: typing.Optional[typing.List[CallArgument]] = None,
        silent: typing.Optional[bool] = None,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        user_gesture: typing.Optional[bool] = None,
        await_promise: typing.Optional[bool] = None,
        execution_context_id: typing.Optional[ExecutionContextId] = None,
        object_group: typing.Optional[str] = None,
        throw_on_side_effect: typing.Optional[bool] = None,
        unique_context_id: typing.Optional[str] = None,
        serialization_options: typing.Optional[SerializationOptions] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[RemoteObject, typing.Optional[ExceptionDetails]]]:
    '''
    Calls function with given declaration on the given object. Object group of the result is
    inherited from the target object.

    :param function_declaration: Declaration of the function to call.
    :param object_id: *(Optional)* Identifier of the object to call function on. Either objectId or executionContextId should be specified.
    :param arguments: *(Optional)* Call arguments. All call arguments must belong to the same JavaScript world as the target object.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ```setPauseOnException```` state.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object which should be sent by value. Can be overriden by ````serializationOptions````.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the result.
    :param user_gesture: *(Optional)* Whether execution should be treated as initiated by user in the UI.
    :param await_promise: *(Optional)* Whether execution should ````await```` for resulting value and return once awaited promise is resolved.
    :param execution_context_id: *(Optional)* Specifies execution context which global object will be used to call function on. Either executionContextId or objectId should be specified.
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects. If objectGroup is not specified and objectId is, objectGroup will be inherited from object.
    :param throw_on_side_effect: **(EXPERIMENTAL)** *(Optional)* Whether to throw an exception if side effect cannot be ruled out during evaluation.
    :param unique_context_id: **(EXPERIMENTAL)** *(Optional)* An alternative way to specify the execution context to call function on. Compared to contextId that may be reused across processes, this is guaranteed to be system-unique, so it can be used to prevent accidental function call in context different than intended (e.g. as a result of navigation across process boundaries). This is mutually exclusive with ````executionContextId````.
    :param serialization_options: **(EXPERIMENTAL)** *(Optional)* Specifies the result serialization. If provided, overrides ````generatePreview```` and ````returnByValue```.
    :returns: A tuple with the following items:

        0. **result** - Call result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['functionDeclaration'] = function_declaration
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if arguments is not None:
        params['arguments'] = [i.to_json() for i in arguments]
    if silent is not None:
        params['silent'] = silent
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if user_gesture is not None:
        params['userGesture'] = user_gesture
    if await_promise is not None:
        params['awaitPromise'] = await_promise
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    if throw_on_side_effect is not None:
        params['throwOnSideEffect'] = throw_on_side_effect
    if unique_context_id is not None:
        params['uniqueContextId'] = unique_context_id
    if serialization_options is not None:
        params['serializationOptions'] = serialization_options.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.callFunctionOn',
        'params': params,
    }
    json = yield cmd_dict
    return (
        RemoteObject.from_json(json['result']),
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def compile_script(
        expression: str,
        source_url: str,
        persist_script: bool,
        execution_context_id: typing.Optional[ExecutionContextId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.Optional[ScriptId], typing.Optional[ExceptionDetails]]]:
    '''
    Compiles expression.

    :param expression: Expression to compile.
    :param source_url: Source url to be set for the script.
    :param persist_script: Specifies whether the compiled script should be persisted.
    :param execution_context_id: *(Optional)* Specifies in which execution context to perform script run. If the parameter is omitted the evaluation will be performed in the context of the inspected page.
    :returns: A tuple with the following items:

        0. **scriptId** - *(Optional)* Id of the script.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['expression'] = expression
    params['sourceURL'] = source_url
    params['persistScript'] = persist_script
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.compileScript',
        'params': params,
    }
    json = yield cmd_dict
    return (
        ScriptId.from_json(json['scriptId']) if 'scriptId' in json else None,
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables reporting of execution contexts creation.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.disable',
    }
    json = yield cmd_dict


def discard_console_entries() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Discards collected exceptions and console API calls.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.discardConsoleEntries',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables reporting of execution contexts creation by means of ``executionContextCreated`` event.
    When the reporting gets enabled the event will be sent immediately for each existing execution
    context.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.enable',
    }
    json = yield cmd_dict


def evaluate(
        expression: str,
        object_group: typing.Optional[str] = None,
        include_command_line_api: typing.Optional[bool] = None,
        silent: typing.Optional[bool] = None,
        context_id: typing.Optional[ExecutionContextId] = None,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        user_gesture: typing.Optional[bool] = None,
        await_promise: typing.Optional[bool] = None,
        throw_on_side_effect: typing.Optional[bool] = None,
        timeout: typing.Optional[TimeDelta] = None,
        disable_breaks: typing.Optional[bool] = None,
        repl_mode: typing.Optional[bool] = None,
        allow_unsafe_eval_blocked_by_csp: typing.Optional[bool] = None,
        unique_context_id: typing.Optional[str] = None,
        serialization_options: typing.Optional[SerializationOptions] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[RemoteObject, typing.Optional[ExceptionDetails]]]:
    '''
    Evaluates expression on global object.

    :param expression: Expression to evaluate.
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects.
    :param include_command_line_api: *(Optional)* Determines whether Command Line API should be available during the evaluation.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ```setPauseOnException```` state.
    :param context_id: *(Optional)* Specifies in which execution context to perform evaluation. If the parameter is omitted the evaluation will be performed in the context of the inspected page. This is mutually exclusive with ````uniqueContextId````, which offers an alternative way to identify the execution context that is more reliable in a multi-process environment.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object that should be sent by value.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the result.
    :param user_gesture: *(Optional)* Whether execution should be treated as initiated by user in the UI.
    :param await_promise: *(Optional)* Whether execution should ````await```` for resulting value and return once awaited promise is resolved.
    :param throw_on_side_effect: **(EXPERIMENTAL)** *(Optional)* Whether to throw an exception if side effect cannot be ruled out during evaluation. This implies ````disableBreaks```` below.
    :param timeout: **(EXPERIMENTAL)** *(Optional)* Terminate execution after timing out (number of milliseconds).
    :param disable_breaks: **(EXPERIMENTAL)** *(Optional)* Disable breakpoints during execution.
    :param repl_mode: **(EXPERIMENTAL)** *(Optional)* Setting this flag to true enables ````let```` re-declaration and top-level ````await````. Note that ````let```` variables can only be re-declared if they originate from ````replMode```` themselves.
    :param allow_unsafe_eval_blocked_by_csp: **(EXPERIMENTAL)** *(Optional)* The Content Security Policy (CSP) for the target might block 'unsafe-eval' which includes eval(), Function(), setTimeout() and setInterval() when called with non-callable arguments. This flag bypasses CSP for this evaluation and allows unsafe-eval. Defaults to true.
    :param unique_context_id: **(EXPERIMENTAL)** *(Optional)* An alternative way to specify the execution context to evaluate in. Compared to contextId that may be reused across processes, this is guaranteed to be system-unique, so it can be used to prevent accidental evaluation of the expression in context different than intended (e.g. as a result of navigation across process boundaries). This is mutually exclusive with ````contextId````.
    :param serialization_options: **(EXPERIMENTAL)** *(Optional)* Specifies the result serialization. If provided, overrides ````generatePreview```` and ````returnByValue```.
    :returns: A tuple with the following items:

        0. **result** - Evaluation result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['expression'] = expression
    if object_group is not None:
        params['objectGroup'] = object_group
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    if silent is not None:
        params['silent'] = silent
    if context_id is not None:
        params['contextId'] = context_id.to_json()
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if user_gesture is not None:
        params['userGesture'] = user_gesture
    if await_promise is not None:
        params['awaitPromise'] = await_promise
    if throw_on_side_effect is not None:
        params['throwOnSideEffect'] = throw_on_side_effect
    if timeout is not None:
        params['timeout'] = timeout.to_json()
    if disable_breaks is not None:
        params['disableBreaks'] = disable_breaks
    if repl_mode is not None:
        params['replMode'] = repl_mode
    if allow_unsafe_eval_blocked_by_csp is not None:
        params['allowUnsafeEvalBlockedByCSP'] = allow_unsafe_eval_blocked_by_csp
    if unique_context_id is not None:
        params['uniqueContextId'] = unique_context_id
    if serialization_options is not None:
        params['serializationOptions'] = serialization_options.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.evaluate',
        'params': params,
    }
    json = yield cmd_dict
    return (
        RemoteObject.from_json(json['result']),
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def get_isolate_id() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Returns the isolate id.

    **EXPERIMENTAL**

    :returns: The isolate id.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.getIsolateId',
    }
    json = yield cmd_dict
    return str(json['id'])


def get_heap_usage() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[float, float, float, float]]:
    '''
    Returns the JavaScript heap usage.
    It is the total usage of the corresponding isolate not scoped to a particular Runtime.

    **EXPERIMENTAL**

    :returns: A tuple with the following items:

        0. **usedSize** - Used JavaScript heap size in bytes.
        1. **totalSize** - Allocated JavaScript heap size in bytes.
        2. **embedderHeapUsedSize** - Used size in bytes in the embedder's garbage-collected heap.
        3. **backingStorageSize** - Size in bytes of backing storage for array buffers and external strings.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.getHeapUsage',
    }
    json = yield cmd_dict
    return (
        float(json['usedSize']),
        float(json['totalSize']),
        float(json['embedderHeapUsedSize']),
        float(json['backingStorageSize'])
    )


def get_properties(
        object_id: RemoteObjectId,
        own_properties: typing.Optional[bool] = None,
        accessor_properties_only: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        non_indexed_properties_only: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[PropertyDescriptor], typing.Optional[typing.List[InternalPropertyDescriptor]], typing.Optional[typing.List[PrivatePropertyDescriptor]], typing.Optional[ExceptionDetails]]]:
    '''
    Returns properties of a given object. Object group of the result is inherited from the target
    object.

    :param object_id: Identifier of the object to return properties for.
    :param own_properties: *(Optional)* If true, returns properties belonging only to the element itself, not to its prototype chain.
    :param accessor_properties_only: **(EXPERIMENTAL)** *(Optional)* If true, returns accessor properties (with getter/setter) only; internal properties are not returned either.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the results.
    :param non_indexed_properties_only: **(EXPERIMENTAL)** *(Optional)* If true, returns non-indexed properties only.
    :returns: A tuple with the following items:

        0. **result** - Object properties.
        1. **internalProperties** - *(Optional)* Internal object properties (only of the element itself).
        2. **privateProperties** - *(Optional)* Object private properties.
        3. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    if own_properties is not None:
        params['ownProperties'] = own_properties
    if accessor_properties_only is not None:
        params['accessorPropertiesOnly'] = accessor_properties_only
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if non_indexed_properties_only is not None:
        params['nonIndexedPropertiesOnly'] = non_indexed_properties_only
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.getProperties',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [PropertyDescriptor.from_json(i) for i in json['result']],
        [InternalPropertyDescriptor.from_json(i) for i in json['internalProperties']] if 'internalProperties' in json else None,
        [PrivatePropertyDescriptor.from_json(i) for i in json['privateProperties']] if 'privateProperties' in json else None,
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def global_lexical_scope_names(
        execution_context_id: typing.Optional[ExecutionContextId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Returns all let, const and class variables from global scope.

    :param execution_context_id: *(Optional)* Specifies in which execution context to lookup global scope variables.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.globalLexicalScopeNames',
        'params': params,
    }
    json = yield cmd_dict
    return [str(i) for i in json['names']]


def query_objects(
        prototype_object_id: RemoteObjectId,
        object_group: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,RemoteObject]:
    '''
    :param prototype_object_id: Identifier of the prototype to return objects for.
    :param object_group: *(Optional)* Symbolic group name that can be used to release the results.
    :returns: Array with objects.
    '''
    params: T_JSON_DICT = dict()
    params['prototypeObjectId'] = prototype_object_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.queryObjects',
        'params': params,
    }
    json = yield cmd_dict
    return RemoteObject.from_json(json['objects'])


def release_object(
        object_id: RemoteObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Releases remote object with given id.

    :param object_id: Identifier of the object to release.
    '''
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.releaseObject',
        'params': params,
    }
    json = yield cmd_dict


def release_object_group(
        object_group: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Releases all remote objects that belong to a given group.

    :param object_group: Symbolic object group name.
    '''
    params: T_JSON_DICT = dict()
    params['objectGroup'] = object_group
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.releaseObjectGroup',
        'params': params,
    }
    json = yield cmd_dict


def run_if_waiting_for_debugger() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Tells inspected instance to run if it was waiting for debugger to attach.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.runIfWaitingForDebugger',
    }
    json = yield cmd_dict


def run_script(
        script_id: ScriptId,
        execution_context_id: typing.Optional[ExecutionContextId] = None,
        object_group: typing.Optional[str] = None,
        silent: typing.Optional[bool] = None,
        include_command_line_api: typing.Optional[bool] = None,
        return_by_value: typing.Optional[bool] = None,
        generate_preview: typing.Optional[bool] = None,
        await_promise: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[RemoteObject, typing.Optional[ExceptionDetails]]]:
    '''
    Runs script with given id in a given context.

    :param script_id: Id of the script to run.
    :param execution_context_id: *(Optional)* Specifies in which execution context to perform script run. If the parameter is omitted the evaluation will be performed in the context of the inspected page.
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ```setPauseOnException```` state.
    :param include_command_line_api: *(Optional)* Determines whether Command Line API should be available during the evaluation.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object which should be sent by value.
    :param generate_preview: *(Optional)* Whether preview should be generated for the result.
    :param await_promise: *(Optional)* Whether execution should ````await``` for resulting value and return once awaited promise is resolved.
    :returns: A tuple with the following items:

        0. **result** - Run result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    '''
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    if silent is not None:
        params['silent'] = silent
    if include_command_line_api is not None:
        params['includeCommandLineAPI'] = include_command_line_api
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if await_promise is not None:
        params['awaitPromise'] = await_promise
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.runScript',
        'params': params,
    }
    json = yield cmd_dict
    return (
        RemoteObject.from_json(json['result']),
        ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None
    )


def set_async_call_stack_depth(
        max_depth: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables or disables async call stacks tracking.

    :param max_depth: Maximum depth of async call stacks. Setting to ```0``` will effectively disable collecting async call stacks (default).
    '''
    params: T_JSON_DICT = dict()
    params['maxDepth'] = max_depth
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.setAsyncCallStackDepth',
        'params': params,
    }
    json = yield cmd_dict


def set_custom_object_formatter_enabled(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param enabled:
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.setCustomObjectFormatterEnabled',
        'params': params,
    }
    json = yield cmd_dict


def set_max_call_stack_size_to_capture(
        size: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''


    **EXPERIMENTAL**

    :param size:
    '''
    params: T_JSON_DICT = dict()
    params['size'] = size
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.setMaxCallStackSizeToCapture',
        'params': params,
    }
    json = yield cmd_dict


def terminate_execution() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Terminate current or next JavaScript execution.
    Will cancel the termination when the outer-most script execution ends.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.terminateExecution',
    }
    json = yield cmd_dict


def add_binding(
        name: str,
        execution_context_id: typing.Optional[ExecutionContextId] = None,
        execution_context_name: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    If executionContextId is empty, adds binding with the given name on the
    global objects of all inspected contexts, including those created later,
    bindings survive reloads.
    Binding function takes exactly one argument, this argument should be string,
    in case of any other input, function throws an exception.
    Each binding function call produces Runtime.bindingCalled notification.

    :param name:
    :param execution_context_id: **(EXPERIMENTAL)** *(Optional)* If specified, the binding would only be exposed to the specified execution context. If omitted and ```executionContextName```` is not set, the binding is exposed to all execution contexts of the target. This parameter is mutually exclusive with ````executionContextName````. Deprecated in favor of ````executionContextName```` due to an unclear use case and bugs in implementation (crbug.com/1169639). ````executionContextId```` will be removed in the future.
    :param execution_context_name: *(Optional)* If specified, the binding is exposed to the executionContext with matching name, even for contexts created after the binding is added. See also ````ExecutionContext.name```` and ````worldName```` parameter to ````Page.addScriptToEvaluateOnNewDocument````. This parameter is mutually exclusive with ````executionContextId```.
    '''
    params: T_JSON_DICT = dict()
    params['name'] = name
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    if execution_context_name is not None:
        params['executionContextName'] = execution_context_name
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.addBinding',
        'params': params,
    }
    json = yield cmd_dict


def remove_binding(
        name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    This method does not remove binding function from global object but
    unsubscribes current runtime agent from Runtime.bindingCalled notifications.

    :param name:
    '''
    params: T_JSON_DICT = dict()
    params['name'] = name
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.removeBinding',
        'params': params,
    }
    json = yield cmd_dict


def get_exception_details(
        error_object_id: RemoteObjectId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Optional[ExceptionDetails]]:
    '''
    This method tries to lookup and populate exception details for a
    JavaScript Error object.
    Note that the stackTrace portion of the resulting exceptionDetails will
    only be populated if the Runtime domain was enabled at the time when the
    Error was thrown.

    **EXPERIMENTAL**

    :param error_object_id: The error object for which to resolve the exception details.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['errorObjectId'] = error_object_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Runtime.getExceptionDetails',
        'params': params,
    }
    json = yield cmd_dict
    return ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None


@event_class('Runtime.bindingCalled')
@dataclass
class BindingCalled:
    '''
    **EXPERIMENTAL**

    Notification is issued every time when binding is called.
    '''
    name: str
    payload: str
    #: Identifier of the context where the call was made.
    execution_context_id: ExecutionContextId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BindingCalled:
        return cls(
            name=str(json['name']),
            payload=str(json['payload']),
            execution_context_id=ExecutionContextId.from_json(json['executionContextId'])
        )


@event_class('Runtime.consoleAPICalled')
@dataclass
class ConsoleAPICalled:
    '''
    Issued when console API was called.
    '''
    #: Type of the call.
    type_: str
    #: Call arguments.
    args: typing.List[RemoteObject]
    #: Identifier of the context where the call was made.
    execution_context_id: ExecutionContextId
    #: Call timestamp.
    timestamp: Timestamp
    #: Stack trace captured when the call was made. The async stack chain is automatically reported for
    #: the following call types: ``assert``, ``error``, ``trace``, ``warning``. For other types the async call
    #: chain can be retrieved using ``Debugger.getStackTrace`` and ``stackTrace.parentId`` field.
    stack_trace: typing.Optional[StackTrace]
    #: Console context descriptor for calls on non-default console context (not console.*):
    #: 'anonymous#unique-logger-id' for call on unnamed context, 'name#unique-logger-id' for call
    #: on named context.
    context: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleAPICalled:
        return cls(
            type_=str(json['type']),
            args=[RemoteObject.from_json(i) for i in json['args']],
            execution_context_id=ExecutionContextId.from_json(json['executionContextId']),
            timestamp=Timestamp.from_json(json['timestamp']),
            stack_trace=StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None,
            context=str(json['context']) if 'context' in json else None
        )


@event_class('Runtime.exceptionRevoked')
@dataclass
class ExceptionRevoked:
    '''
    Issued when unhandled exception was revoked.
    '''
    #: Reason describing why exception was revoked.
    reason: str
    #: The id of revoked exception, as reported in ``exceptionThrown``.
    exception_id: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExceptionRevoked:
        return cls(
            reason=str(json['reason']),
            exception_id=int(json['exceptionId'])
        )


@event_class('Runtime.exceptionThrown')
@dataclass
class ExceptionThrown:
    '''
    Issued when exception was thrown and unhandled.
    '''
    #: Timestamp of the exception.
    timestamp: Timestamp
    exception_details: ExceptionDetails

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExceptionThrown:
        return cls(
            timestamp=Timestamp.from_json(json['timestamp']),
            exception_details=ExceptionDetails.from_json(json['exceptionDetails'])
        )


@event_class('Runtime.executionContextCreated')
@dataclass
class ExecutionContextCreated:
    '''
    Issued when new execution context is created.
    '''
    #: A newly created execution context.
    context: ExecutionContextDescription

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExecutionContextCreated:
        return cls(
            context=ExecutionContextDescription.from_json(json['context'])
        )


@event_class('Runtime.executionContextDestroyed')
@dataclass
class ExecutionContextDestroyed:
    '''
    Issued when execution context is destroyed.
    '''
    #: Id of the destroyed context
    execution_context_id: ExecutionContextId
    #: Unique Id of the destroyed context
    execution_context_unique_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExecutionContextDestroyed:
        return cls(
            execution_context_id=ExecutionContextId.from_json(json['executionContextId']),
            execution_context_unique_id=str(json['executionContextUniqueId'])
        )


@event_class('Runtime.executionContextsCleared')
@dataclass
class ExecutionContextsCleared:
    '''
    Issued when all executionContexts were cleared in browser
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExecutionContextsCleared:
        return cls(

        )


@event_class('Runtime.inspectRequested')
@dataclass
class InspectRequested:
    '''
    Issued when object should be inspected (for example, as a result of inspect() command line API
    call).
    '''
    object_: RemoteObject
    hints: dict
    #: Identifier of the context where the call was made.
    execution_context_id: typing.Optional[ExecutionContextId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InspectRequested:
        return cls(
            object_=RemoteObject.from_json(json['object']),
            hints=dict(json['hints']),
            execution_context_id=ExecutionContextId.from_json(json['executionContextId']) if 'executionContextId' in json else None
        )