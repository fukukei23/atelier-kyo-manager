
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\laguerre.py ===
"""
==================================================
Laguerre Series (:mod:`numpy.polynomial.laguerre`)
==================================================

This module provides a number of objects (mostly functions) useful for
dealing with Laguerre series, including a `Laguerre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   Laguerre

Constants
---------
.. autosummary::
   :toctree: generated/

   lagdomain
   lagzero
   lagone
   lagx

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   lagadd
   lagsub
   lagmulx
   lagmul
   lagdiv
   lagpow
   lagval
   lagval2d
   lagval3d
   laggrid2d
   laggrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   lagder
   lagint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   lagfromroots
   lagroots
   lagvander
   lagvander2d
   lagvander3d
   laggauss
   lagweight
   lagcompanion
   lagfit
   lagtrim
   lagline
   lag2poly
   poly2lag

See also
--------
`numpy.polynomial`

"""
import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

__all__ = [
    'lagzero', 'lagone', 'lagx', 'lagdomain', 'lagline', 'lagadd',
    'lagsub', 'lagmulx', 'lagmul', 'lagdiv', 'lagpow', 'lagval', 'lagder',
    'lagint', 'lag2poly', 'poly2lag', 'lagfromroots', 'lagvander',
    'lagfit', 'lagtrim', 'lagroots', 'Laguerre', 'lagval2d', 'lagval3d',
    'laggrid2d', 'laggrid3d', 'lagvander2d', 'lagvander3d', 'lagcompanion',
    'laggauss', 'lagweight']

lagtrim = pu.trimcoef


def poly2lag(pol):
    """
    poly2lag(pol)

    Convert a polynomial to a Laguerre series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Laguerre series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Laguerre
        series.

    See Also
    --------
    lag2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import poly2lag
    >>> poly2lag(np.arange(4))
    array([ 23., -63.,  58., -18.])

    """
    [pol] = pu.as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = lagadd(lagmulx(res), p)
    return res


def lag2poly(c):
    """
    Convert a Laguerre series to a polynomial.

    Convert an array representing the coefficients of a Laguerre series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Laguerre series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2lag

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lag2poly
    >>> lag2poly([ 23., -63.,  58., -18.])
    array([0., 1., 2., 3.])

    """
    from .polynomial import polyadd, polymulx, polysub

    [c] = pu.as_series([c])
    n = len(c)
    if n == 1:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], (c1 * (i - 1)) / i)
            c1 = polyadd(tmp, polysub((2 * i - 1) * c1, polymulx(c1)) / i)
        return polyadd(c0, polysub(c1, polymulx(c1)))


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Laguerre
lagdomain = np.array([0., 1.])

# Laguerre coefficients representing zero.
lagzero = np.array([0])

# Laguerre coefficients representing one.
lagone = np.array([1])

# Laguerre coefficients representing the identity x.
lagx = np.array([1, -1])


def lagline(off, scl):
    """
    Laguerre series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Laguerre series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagline, lagval
    >>> lagval(0,lagline(3, 2))
    3.0
    >>> lagval(1,lagline(3, 2))
    5.0

    """
    if scl != 0:
        return np.array([off + scl, -scl])
    else:
        return np.array([off])


def lagfromroots(roots):
    """
    Generate a Laguerre series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in Laguerre form, where the :math:`r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in Laguerre form.

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
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagfromroots, lagval
    >>> coef = lagfromroots((-1, 0, 1))
    >>> lagval((-1, 0, 1), coef)
    array([0.,  0.,  0.])
    >>> coef = lagfromroots((-1j, 1j))
    >>> lagval((-1j, 1j), coef)
    array([0.+0.j, 0.+0.j])

    """
    return pu._fromroots(lagline, lagmul, roots)


def lagadd(c1, c2):
    """
    Add one Laguerre series to another.

    Returns the sum of two Laguerre series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Laguerre series of their sum.

    See Also
    --------
    lagsub, lagmulx, lagmul, lagdiv, lagpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Laguerre series
    is a Laguerre series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagadd
    >>> lagadd([1, 2, 3], [1, 2, 3, 4])
    array([2.,  4.,  6.,  4.])

    """
    return pu._add(c1, c2)


def lagsub(c1, c2):
    """
    Subtract one Laguerre series from another.

    Returns the difference of two Laguerre series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Laguerre series coefficients representing their difference.

    See Also
    --------
    lagadd, lagmulx, lagmul, lagdiv, lagpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Laguerre
    series is a Laguerre series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagsub
    >>> lagsub([1, 2, 3, 4], [1, 2, 3])
    array([0.,  0.,  0.,  4.])

    """
    return pu._sub(c1, c2)


def lagmulx(c):
    """Multiply a Laguerre series by x.

    Multiply the Laguerre series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    lagadd, lagsub, lagmul, lagdiv, lagpow

    Notes
    -----
    The multiplication uses the recursion relationship for Laguerre
    polynomials in the form

    .. math::

        xP_i(x) = (-(i + 1)*P_{i + 1}(x) + (2i + 1)P_{i}(x) - iP_{i - 1}(x))

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagmulx
    >>> lagmulx([1, 2, 3])
    array([-1.,  -1.,  11.,  -9.])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]
    prd[1] = -c[0]
    for i in range(1, len(c)):
        prd[i + 1] = -c[i] * (i + 1)
        prd[i] += c[i] * (2 * i + 1)
        prd[i - 1] -= c[i] * i
    return prd


def lagmul(c1, c2):
    """
    Multiply one Laguerre series by another.

    Returns the product of two Laguerre series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Laguerre series coefficients representing their product.

    See Also
    --------
    lagadd, lagsub, lagmulx, lagdiv, lagpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Laguerre polynomial basis set.  Thus, to express
    the product as a Laguerre series, it is necessary to "reproject" the
    product onto said basis set, which may produce "unintuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagmul
    >>> lagmul([1, 2, 3], [0, 1, 2])
    array([  8., -13.,  38., -51.,  36.])

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
            c0 = lagsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = lagadd(tmp, lagsub((2 * nd - 1) * c1, lagmulx(c1)) / nd)
    return lagadd(c0, lagsub(c1, lagmulx(c1)))


def lagdiv(c1, c2):
    """
    Divide one Laguerre series by another.

    Returns the quotient-with-remainder of two Laguerre series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Laguerre series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Laguerre series coefficients representing the quotient and
        remainder.

    See Also
    --------
    lagadd, lagsub, lagmulx, lagmul, lagpow

    Notes
    -----
    In general, the (polynomial) division of one Laguerre series by another
    results in quotient and remainder terms that are not in the Laguerre
    polynomial basis set.  Thus, to express these results as a Laguerre
    series, it is necessary to "reproject" the results onto the Laguerre
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagdiv
    >>> lagdiv([  8., -13.,  38., -51.,  36.], [0, 1, 2])
    (array([1., 2., 3.]), array([0.]))
    >>> lagdiv([  9., -12.,  38., -51.,  36.], [0, 1, 2])
    (array([1., 2., 3.]), array([1., 1.]))

    """
    return pu._div(lagmul, c1, c2)


def lagpow(c, pow, maxpower=16):
    """Raise a Laguerre series to a power.

    Returns the Laguerre series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Laguerre series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Laguerre series of power.

    See Also
    --------
    lagadd, lagsub, lagmulx, lagmul, lagdiv

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagpow
    >>> lagpow([1, 2, 3], 2)
    array([ 14., -16.,  56., -72.,  54.])

    """
    return pu._pow(lagmul, c, pow, maxpower)


def lagder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Laguerre series.

    Returns the Laguerre series coefficients `c` differentiated `m` times
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
        Array of Laguerre series coefficients. If `c` is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
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
        Laguerre series of the derivative.

    See Also
    --------
    lagint

    Notes
    -----
    In general, the result of differentiating a Laguerre series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagder
    >>> lagder([ 1.,  1.,  1., -3.])
    array([1.,  2.,  3.])
    >>> lagder([ 1.,  0.,  0., -4.,  3.], m=2)
    array([1.,  2.,  3.])

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
            for j in range(n, 1, -1):
                der[j - 1] = -c[j]
                c[j - 1] += c[j]
            der[0] = -c[1]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def lagint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Laguerre series.

    Returns the Laguerre series coefficients `c` integrated `m` times from
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
        Array of Laguerre series coefficients. If `c` is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
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
        Laguerre series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    lagder

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
    >>> from numpy.polynomial.laguerre import lagint
    >>> lagint([1,2,3])
    array([ 1.,  1.,  1., -3.])
    >>> lagint([1,2,3], m=2)
    array([ 1.,  0.,  0., -4.,  3.])
    >>> lagint([1,2,3], k=1)
    array([ 2.,  1.,  1., -3.])
    >>> lagint([1,2,3], lbnd=-1)
    array([11.5,  1. ,  1. , -3. ])
    >>> lagint([1,2], m=2, k=[1,2], lbnd=-1)
    array([ 11.16666667,  -5.        ,  -3.        ,   2.        ]) # may vary

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
            tmp[0] = c[0]
            tmp[1] = -c[0]
            for j in range(1, n):
                tmp[j] += c[j]
                tmp[j + 1] = -c[j]
            tmp[0] += k[i] - lagval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def lagval(x, c, tensor=True):
    """
    Evaluate a Laguerre series at points x.

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
    lagval2d, laggrid2d, lagval3d, laggrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagval
    >>> coef = [1, 2, 3]
    >>> lagval(1, coef)
    -0.5
    >>> lagval([[1, 2],[3, 4]], coef)
    array([[-0.5, -4. ],
           [-4.5, -2. ]])

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
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * ((2 * nd - 1) - x)) / nd
    return c0 + c1 * (1 - x)


def lagval2d(x, y, c):
    """
    Evaluate a 2-D Laguerre series at points (x, y).

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
        The values of the two dimensional polynomial at points formed with
        pairs of corresponding values from `x` and `y`.

    See Also
    --------
    lagval, laggrid2d, lagval3d, laggrid3d

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagval2d
    >>> c = [[1, 2],[3, 4]]
    >>> lagval2d(1, 1, c)
    1.0
    """
    return pu._valnd(lagval, c, x, y)


def laggrid2d(x, y, c):
    """
    Evaluate a 2-D Laguerre series on the Cartesian product of x and y.

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
    lagval, lagval2d, lagval3d, laggrid3d

    Examples
    --------
    >>> from numpy.polynomial.laguerre import laggrid2d
    >>> c = [[1, 2], [3, 4]]
    >>> laggrid2d([0, 1], [0, 1], c)
    array([[10.,  4.],
           [ 3.,  1.]])

    """
    return pu._gridnd(lagval, c, x, y)


def lagval3d(x, y, z, c):
    """
    Evaluate a 3-D Laguerre series at points (x, y, z).

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
    lagval, lagval2d, laggrid2d, laggrid3d

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagval3d
    >>> c = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    >>> lagval3d(1, 1, 2, c)
    -1.0

    """
    return pu._valnd(lagval, c, x, y, z)


def laggrid3d(x, y, z, c):
    """
    Evaluate a 3-D Laguerre series on the Cartesian product of x, y, and z.

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
    lagval, lagval2d, laggrid2d, lagval3d

    Examples
    --------
    >>> from numpy.polynomial.laguerre import laggrid3d
    >>> c = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    >>> laggrid3d([0, 1], [0, 1], [2, 4], c)
    array([[[ -4., -44.],
            [ -2., -18.]],
           [[ -2., -14.],
            [ -1.,  -5.]]])

    """
    return pu._gridnd(lagval, c, x, y, z)


def lagvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = L_i(x)

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Laguerre polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    array ``V = lagvander(x, n)``, then ``np.dot(V, c)`` and
    ``lagval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Laguerre series of the same degree and sample points.

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
        corresponding Laguerre polynomial.  The dtype will be the same as
        the converted `x`.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import lagvander
    >>> x = np.array([0, 1, 2])
    >>> lagvander(x, 3)
    array([[ 1.        ,  1.        ,  1.        ,  1.        ],
           [ 1.        ,  0.        , -0.5       , -0.66666667],
           [ 1.        , -1.        , -1.        , -0.33333333]])

    """
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=None, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = 1 - x
        for i in range(2, ideg + 1):
            v[i] = (v[i - 1] * (2 * i - 1 - x) - v[i - 2] * (i - 1)) / i
    return np.moveaxis(v, 0, -1)


def lagvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = L_i(x) * L_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the Laguerre polynomials.

    If ``V = lagvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``lagval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D Laguerre
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
    lagvander, lagvander3d, lagval2d, lagval3d

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import lagvander2d
    >>> x = np.array([0])
    >>> y = np.array([2])
    >>> lagvander2d(x, y, [2, 1])
    array([[ 1., -1.,  1., -1.,  1., -1.]])

    """
    return pu._vander_nd_flat((lagvander, lagvander), (x, y), deg)


def lagvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = L_i(x)*L_j(y)*L_k(z),

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the degrees of the Laguerre polynomials.

    If ``V = lagvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``lagval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Laguerre
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
    lagvander, lagvander3d, lagval2d, lagval3d

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import lagvander3d
    >>> x = np.array([0])
    >>> y = np.array([2])
    >>> z = np.array([0])
    >>> lagvander3d(x, y, z, [2, 1, 3])
    array([[ 1.,  1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.,  1.,  1., -1.,
            -1., -1., -1.,  1.,  1.,  1.,  1., -1., -1., -1., -1.]])

    """
    return pu._vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), deg)


def lagfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Laguerre series to data.

    Return the coefficients of a Laguerre series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),

    where ``n`` is `deg`.

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
        Laguerre coefficients ordered from low to high. If `y` was 2-D,
        the coefficients for the data in column *k*  of `y` are in column
        *k*.

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
    numpy.polynomial.legendre.legfit
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    lagval : Evaluates a Laguerre series.
    lagvander : pseudo Vandermonde matrix of Laguerre series.
    lagweight : Laguerre weight function.
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Laguerre series ``p`` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up as the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where ``V`` is the weighted pseudo Vandermonde matrix of `x`, ``c`` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of ``V``.

    If some of the singular values of `V` are so small that they are
    neglected, then a `~exceptions.RankWarning` will be issued. This means that
    the coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Laguerre series are probably most useful when the data can
    be approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the Laguerre
    weight. In that case the weight ``sqrt(w(x[i]))`` should be used
    together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is
    available as `lagweight`.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import lagfit, lagval
    >>> x = np.linspace(0, 10)
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(scale=1./10, size=len(x))
    >>> y = lagval(x, [1, 2, 3]) + err
    >>> lagfit(x, y, 2)
    array([1.00578369, 1.99417356, 2.99827656]) # may vary

    """
    return pu._fit(lagvander, x, y, deg, rcond, full, w)


def lagcompanion(c):
    """
    Return the companion matrix of c.

    The usual companion matrix of the Laguerre polynomials is already
    symmetric when `c` is a basis Laguerre polynomial, so no scaling is
    applied.

    Parameters
    ----------
    c : array_like
        1-D array of Laguerre series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Companion matrix of dimensions (deg, deg).

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagcompanion
    >>> lagcompanion([1, 2, 3])
    array([[ 1.        , -0.33333333],
           [-1.        ,  4.33333333]])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[1 + c[0] / c[1]]])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    top = mat.reshape(-1)[1::n + 1]
    mid = mat.reshape(-1)[0::n + 1]
    bot = mat.reshape(-1)[n::n + 1]
    top[...] = -np.arange(1, n)
    mid[...] = 2. * np.arange(n) + 1.
    bot[...] = top
    mat[:, -1] += (c[:-1] / c[-1]) * n
    return mat


def lagroots(c):
    """
    Compute the roots of a Laguerre series.

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
    numpy.polynomial.legendre.legroots
    numpy.polynomial.chebyshev.chebroots
    numpy.polynomial.hermite.hermroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    The Laguerre series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagroots, lagfromroots
    >>> coef = lagfromroots([0, 1, 2])
    >>> coef
    array([  2.,  -8.,  12.,  -6.])
    >>> lagroots(coef)
    array([-4.4408921e-16,  1.0000000e+00,  2.0000000e+00])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) <= 1:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([1 + c[0] / c[1]])

    # rotated companion matrix reduces error
    m = lagcompanion(c)[::-1, ::-1]
    r = la.eigvals(m)
    r.sort()
    return r


def laggauss(deg):
    """
    Gauss-Laguerre quadrature.

    Computes the sample points and weights for Gauss-Laguerre quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[0, \\inf]`
    with the weight function :math:`f(x) = \\exp(-x)`.

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
    The results have only been tested up to degree 100 higher degrees may
    be problematic. The weights are determined by using the fact that

    .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`L_n`, and then scaling the results to get
    the right value when integrating 1.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import laggauss
    >>> laggauss(2)
    (array([0.58578644, 3.41421356]), array([0.85355339, 0.14644661]))

    """
    ideg = pu._as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    # first approximation of roots. We use the fact that the companion
    # matrix is symmetric in this case in order to obtain better zeros.
    c = np.array([0] * deg + [1])
    m = lagcompanion(c)
    x = la.eigvalsh(m)

    # improve roots by one application of Newton
    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy / df

    # compute the weights. We scale the factor to avoid possible numerical
    # overflow.
    fm = lagval(x, c[1:])
    fm /= np.abs(fm).max()
    df /= np.abs(df).max()
    w = 1 / (fm * df)

    # scale w to get the right value, 1 in this case
    w /= w.sum()

    return x, w


def lagweight(x):
    """Weight function of the Laguerre polynomials.

    The weight function is :math:`exp(-x)` and the interval of integration
    is :math:`[0, \\inf]`. The Laguerre polynomials are orthogonal, but not
    normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagweight
    >>> x = np.array([0, 1, 2])
    >>> lagweight(x)
    array([1.        , 0.36787944, 0.13533528])

    """
    w = np.exp(-x)
    return w

#
# Laguerre series class
#

class Laguerre(ABCPolyBase):
    """A Laguerre series class.

    The Laguerre class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Laguerre coefficients in order of increasing degree, i.e,
        ``(1, 2, 3)`` gives ``1*L_0(x) + 2*L_1(X) + 3*L_2(x)``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [0., 1.].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [0., 1.].
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    _add = staticmethod(lagadd)
    _sub = staticmethod(lagsub)
    _mul = staticmethod(lagmul)
    _div = staticmethod(lagdiv)
    _pow = staticmethod(lagpow)
    _val = staticmethod(lagval)
    _int = staticmethod(lagint)
    _der = staticmethod(lagder)
    _fit = staticmethod(lagfit)
    _line = staticmethod(lagline)
    _roots = staticmethod(lagroots)
    _fromroots = staticmethod(lagfromroots)

    # Virtual properties
    domain = np.array(lagdomain)
    window = np.array(lagdomain)
    basis_name = 'L'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\environment.py ===
"""Classes for managing templates and their runtime and compile time
options.
"""

import os
import typing
import typing as t
import weakref
from collections import ChainMap
from functools import lru_cache
from functools import partial
from functools import reduce
from types import CodeType

from markupsafe import Markup

from . import nodes
from .compiler import CodeGenerator
from .compiler import generate
from .defaults import BLOCK_END_STRING
from .defaults import BLOCK_START_STRING
from .defaults import COMMENT_END_STRING
from .defaults import COMMENT_START_STRING
from .defaults import DEFAULT_FILTERS  # type: ignore[attr-defined]
from .defaults import DEFAULT_NAMESPACE
from .defaults import DEFAULT_POLICIES
from .defaults import DEFAULT_TESTS  # type: ignore[attr-defined]
from .defaults import KEEP_TRAILING_NEWLINE
from .defaults import LINE_COMMENT_PREFIX
from .defaults import LINE_STATEMENT_PREFIX
from .defaults import LSTRIP_BLOCKS
from .defaults import NEWLINE_SEQUENCE
from .defaults import TRIM_BLOCKS
from .defaults import VARIABLE_END_STRING
from .defaults import VARIABLE_START_STRING
from .exceptions import TemplateNotFound
from .exceptions import TemplateRuntimeError
from .exceptions import TemplatesNotFound
from .exceptions import TemplateSyntaxError
from .exceptions import UndefinedError
from .lexer import get_lexer
from .lexer import Lexer
from .lexer import TokenStream
from .nodes import EvalContext
from .parser import Parser
from .runtime import Context
from .runtime import new_context
from .runtime import Undefined
from .utils import _PassArg
from .utils import concat
from .utils import consume
from .utils import import_string
from .utils import internalcode
from .utils import LRUCache
from .utils import missing

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .bccache import BytecodeCache
    from .ext import Extension
    from .loaders import BaseLoader

_env_bound = t.TypeVar("_env_bound", bound="Environment")


# for direct template usage we have up to ten living environments
@lru_cache(maxsize=10)
def get_spontaneous_environment(cls: t.Type[_env_bound], *args: t.Any) -> _env_bound:
    """Return a new spontaneous environment. A spontaneous environment
    is used for templates created directly rather than through an
    existing environment.

    :param cls: Environment class to create.
    :param args: Positional arguments passed to environment.
    """
    env = cls(*args)
    env.shared = True
    return env


def create_cache(
    size: int,
) -> t.Optional[t.MutableMapping[t.Tuple["weakref.ref[t.Any]", str], "Template"]]:
    """Return the cache class for the given size."""
    if size == 0:
        return None

    if size < 0:
        return {}

    return LRUCache(size)  # type: ignore


def copy_cache(
    cache: t.Optional[t.MutableMapping[t.Any, t.Any]],
) -> t.Optional[t.MutableMapping[t.Tuple["weakref.ref[t.Any]", str], "Template"]]:
    """Create an empty copy of the given cache."""
    if cache is None:
        return None

    if type(cache) is dict:  # noqa E721
        return {}

    return LRUCache(cache.capacity)  # type: ignore


def load_extensions(
    environment: "Environment",
    extensions: t.Sequence[t.Union[str, t.Type["Extension"]]],
) -> t.Dict[str, "Extension"]:
    """Load the extensions from the list and bind it to the environment.
    Returns a dict of instantiated extensions.
    """
    result = {}

    for extension in extensions:
        if isinstance(extension, str):
            extension = t.cast(t.Type["Extension"], import_string(extension))

        result[extension.identifier] = extension(environment)

    return result


def _environment_config_check(environment: _env_bound) -> _env_bound:
    """Perform a sanity check on the environment."""
    assert issubclass(
        environment.undefined, Undefined
    ), "'undefined' must be a subclass of 'jinja2.Undefined'."
    assert (
        environment.block_start_string
        != environment.variable_start_string
        != environment.comment_start_string
    ), "block, variable and comment start strings must be different."
    assert environment.newline_sequence in {
        "\r",
        "\r\n",
        "\n",
    }, "'newline_sequence' must be one of '\\n', '\\r\\n', or '\\r'."
    return environment


class Environment:
    r"""The core component of Jinja is the `Environment`.  It contains
    important shared variables like configuration, filters, tests,
    globals and others.  Instances of this class may be modified if
    they are not shared and if no template was loaded so far.
    Modifications on environments after the first template was loaded
    will lead to surprising effects and undefined behavior.

    Here are the possible initialization parameters:

        `block_start_string`
            The string marking the beginning of a block.  Defaults to ``'{%'``.

        `block_end_string`
            The string marking the end of a block.  Defaults to ``'%}'``.

        `variable_start_string`
            The string marking the beginning of a print statement.
            Defaults to ``'{{'``.

        `variable_end_string`
            The string marking the end of a print statement.  Defaults to
            ``'}}'``.

        `comment_start_string`
            The string marking the beginning of a comment.  Defaults to ``'{#'``.

        `comment_end_string`
            The string marking the end of a comment.  Defaults to ``'#}'``.

        `line_statement_prefix`
            If given and a string, this will be used as prefix for line based
            statements.  See also :ref:`line-statements`.

        `line_comment_prefix`
            If given and a string, this will be used as prefix for line based
            comments.  See also :ref:`line-statements`.

            .. versionadded:: 2.2

        `trim_blocks`
            If this is set to ``True`` the first newline after a block is
            removed (block, not variable tag!).  Defaults to `False`.

        `lstrip_blocks`
            If this is set to ``True`` leading spaces and tabs are stripped
            from the start of a line to a block.  Defaults to `False`.

        `newline_sequence`
            The sequence that starts a newline.  Must be one of ``'\r'``,
            ``'\n'`` or ``'\r\n'``.  The default is ``'\n'`` which is a
            useful default for Linux and OS X systems as well as web
            applications.

        `keep_trailing_newline`
            Preserve the trailing newline when rendering templates.
            The default is ``False``, which causes a single newline,
            if present, to be stripped from the end of the template.

            .. versionadded:: 2.7

        `extensions`
            List of Jinja extensions to use.  This can either be import paths
            as strings or extension classes.  For more information have a
            look at :ref:`the extensions documentation <jinja-extensions>`.

        `optimized`
            should the optimizer be enabled?  Default is ``True``.

        `undefined`
            :class:`Undefined` or a subclass of it that is used to represent
            undefined values in the template.

        `finalize`
            A callable that can be used to process the result of a variable
            expression before it is output.  For example one can convert
            ``None`` implicitly into an empty string here.

        `autoescape`
            If set to ``True`` the XML/HTML autoescaping feature is enabled by
            default.  For more details about autoescaping see
            :class:`~markupsafe.Markup`.  As of Jinja 2.4 this can also
            be a callable that is passed the template name and has to
            return ``True`` or ``False`` depending on autoescape should be
            enabled by default.

            .. versionchanged:: 2.4
               `autoescape` can now be a function

        `loader`
            The template loader for this environment.

        `cache_size`
            The size of the cache.  Per default this is ``400`` which means
            that if more than 400 templates are loaded the loader will clean
            out the least recently used template.  If the cache size is set to
            ``0`` templates are recompiled all the time, if the cache size is
            ``-1`` the cache will not be cleaned.

            .. versionchanged:: 2.8
               The cache size was increased to 400 from a low 50.

        `auto_reload`
            Some loaders load templates from locations where the template
            sources may change (ie: file system or database).  If
            ``auto_reload`` is set to ``True`` (default) every time a template is
            requested the loader checks if the source changed and if yes, it
            will reload the template.  For higher performance it's possible to
            disable that.

        `bytecode_cache`
            If set to a bytecode cache object, this object will provide a
            cache for the internal Jinja bytecode so that templates don't
            have to be parsed if they were not changed.

            See :ref:`bytecode-cache` for more information.

        `enable_async`
            If set to true this enables async template execution which
            allows using async functions and generators.
    """

    #: if this environment is sandboxed.  Modifying this variable won't make
    #: the environment sandboxed though.  For a real sandboxed environment
    #: have a look at jinja2.sandbox.  This flag alone controls the code
    #: generation by the compiler.
    sandboxed = False

    #: True if the environment is just an overlay
    overlayed = False

    #: the environment this environment is linked to if it is an overlay
    linked_to: t.Optional["Environment"] = None

    #: shared environments have this set to `True`.  A shared environment
    #: must not be modified
    shared = False

    #: the class that is used for code generation.  See
    #: :class:`~jinja2.compiler.CodeGenerator` for more information.
    code_generator_class: t.Type["CodeGenerator"] = CodeGenerator

    concat = "".join

    #: the context class that is used for templates.  See
    #: :class:`~jinja2.runtime.Context` for more information.
    context_class: t.Type[Context] = Context

    template_class: t.Type["Template"]

    def __init__(
        self,
        block_start_string: str = BLOCK_START_STRING,
        block_end_string: str = BLOCK_END_STRING,
        variable_start_string: str = VARIABLE_START_STRING,
        variable_end_string: str = VARIABLE_END_STRING,
        comment_start_string: str = COMMENT_START_STRING,
        comment_end_string: str = COMMENT_END_STRING,
        line_statement_prefix: t.Optional[str] = LINE_STATEMENT_PREFIX,
        line_comment_prefix: t.Optional[str] = LINE_COMMENT_PREFIX,
        trim_blocks: bool = TRIM_BLOCKS,
        lstrip_blocks: bool = LSTRIP_BLOCKS,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = NEWLINE_SEQUENCE,
        keep_trailing_newline: bool = KEEP_TRAILING_NEWLINE,
        extensions: t.Sequence[t.Union[str, t.Type["Extension"]]] = (),
        optimized: bool = True,
        undefined: t.Type[Undefined] = Undefined,
        finalize: t.Optional[t.Callable[..., t.Any]] = None,
        autoescape: t.Union[bool, t.Callable[[t.Optional[str]], bool]] = False,
        loader: t.Optional["BaseLoader"] = None,
        cache_size: int = 400,
        auto_reload: bool = True,
        bytecode_cache: t.Optional["BytecodeCache"] = None,
        enable_async: bool = False,
    ):
        # !!Important notice!!
        #   The constructor accepts quite a few arguments that should be
        #   passed by keyword rather than position.  However it's important to
        #   not change the order of arguments because it's used at least
        #   internally in those cases:
        #       -   spontaneous environments (i18n extension and Template)
        #       -   unittests
        #   If parameter changes are required only add parameters at the end
        #   and don't change the arguments (or the defaults!) of the arguments
        #   existing already.

        # lexer / parser information
        self.block_start_string = block_start_string
        self.block_end_string = block_end_string
        self.variable_start_string = variable_start_string
        self.variable_end_string = variable_end_string
        self.comment_start_string = comment_start_string
        self.comment_end_string = comment_end_string
        self.line_statement_prefix = line_statement_prefix
        self.line_comment_prefix = line_comment_prefix
        self.trim_blocks = trim_blocks
        self.lstrip_blocks = lstrip_blocks
        self.newline_sequence = newline_sequence
        self.keep_trailing_newline = keep_trailing_newline

        # runtime information
        self.undefined: t.Type[Undefined] = undefined
        self.optimized = optimized
        self.finalize = finalize
        self.autoescape = autoescape

        # defaults
        self.filters = DEFAULT_FILTERS.copy()
        self.tests = DEFAULT_TESTS.copy()
        self.globals = DEFAULT_NAMESPACE.copy()

        # set the loader provided
        self.loader = loader
        self.cache = create_cache(cache_size)
        self.bytecode_cache = bytecode_cache
        self.auto_reload = auto_reload

        # configurable policies
        self.policies = DEFAULT_POLICIES.copy()

        # load extensions
        self.extensions = load_extensions(self, extensions)

        self.is_async = enable_async
        _environment_config_check(self)

    def add_extension(self, extension: t.Union[str, t.Type["Extension"]]) -> None:
        """Adds an extension after the environment was created.

        .. versionadded:: 2.5
        """
        self.extensions.update(load_extensions(self, [extension]))

    def extend(self, **attributes: t.Any) -> None:
        """Add the items to the instance of the environment if they do not exist
        yet.  This is used by :ref:`extensions <writing-extensions>` to register
        callbacks and configuration values without breaking inheritance.
        """
        for key, value in attributes.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def overlay(
        self,
        block_start_string: str = missing,
        block_end_string: str = missing,
        variable_start_string: str = missing,
        variable_end_string: str = missing,
        comment_start_string: str = missing,
        comment_end_string: str = missing,
        line_statement_prefix: t.Optional[str] = missing,
        line_comment_prefix: t.Optional[str] = missing,
        trim_blocks: bool = missing,
        lstrip_blocks: bool = missing,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = missing,
        keep_trailing_newline: bool = missing,
        extensions: t.Sequence[t.Union[str, t.Type["Extension"]]] = missing,
        optimized: bool = missing,
        undefined: t.Type[Undefined] = missing,
        finalize: t.Optional[t.Callable[..., t.Any]] = missing,
        autoescape: t.Union[bool, t.Callable[[t.Optional[str]], bool]] = missing,
        loader: t.Optional["BaseLoader"] = missing,
        cache_size: int = missing,
        auto_reload: bool = missing,
        bytecode_cache: t.Optional["BytecodeCache"] = missing,
        enable_async: bool = missing,
    ) -> "te.Self":
        """Create a new overlay environment that shares all the data with the
        current environment except for cache and the overridden attributes.
        Extensions cannot be removed for an overlayed environment.  An overlayed
        environment automatically gets all the extensions of the environment it
        is linked to plus optional extra extensions.

        Creating overlays should happen after the initial environment was set
        up completely.  Not all attributes are truly linked, some are just
        copied over so modifications on the original environment may not shine
        through.

        .. versionchanged:: 3.1.5
            ``enable_async`` is applied correctly.

        .. versionchanged:: 3.1.2
            Added the ``newline_sequence``, ``keep_trailing_newline``,
            and ``enable_async`` parameters to match ``__init__``.
        """
        args = dict(locals())
        del args["self"], args["cache_size"], args["extensions"], args["enable_async"]

        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.overlayed = True
        rv.linked_to = self

        for key, value in args.items():
            if value is not missing:
                setattr(rv, key, value)

        if cache_size is not missing:
            rv.cache = create_cache(cache_size)
        else:
            rv.cache = copy_cache(self.cache)

        rv.extensions = {}
        for key, value in self.extensions.items():
            rv.extensions[key] = value.bind(rv)
        if extensions is not missing:
            rv.extensions.update(load_extensions(rv, extensions))

        if enable_async is not missing:
            rv.is_async = enable_async

        return _environment_config_check(rv)

    @property
    def lexer(self) -> Lexer:
        """The lexer for this environment."""
        return get_lexer(self)

    def iter_extensions(self) -> t.Iterator["Extension"]:
        """Iterates over the extensions by priority."""
        return iter(sorted(self.extensions.values(), key=lambda x: x.priority))

    def getitem(
        self, obj: t.Any, argument: t.Union[str, t.Any]
    ) -> t.Union[t.Any, Undefined]:
        """Get an item or attribute of an object but prefer the item."""
        try:
            return obj[argument]
        except (AttributeError, TypeError, LookupError):
            if isinstance(argument, str):
                try:
                    attr = str(argument)
                except Exception:
                    pass
                else:
                    try:
                        return getattr(obj, attr)
                    except AttributeError:
                        pass
            return self.undefined(obj=obj, name=argument)

    def getattr(self, obj: t.Any, attribute: str) -> t.Any:
        """Get an item or attribute of an object but prefer the attribute.
        Unlike :meth:`getitem` the attribute *must* be a string.
        """
        try:
            return getattr(obj, attribute)
        except AttributeError:
            pass
        try:
            return obj[attribute]
        except (TypeError, LookupError, AttributeError):
            return self.undefined(obj=obj, name=attribute)

    def _filter_test_common(
        self,
        name: t.Union[str, Undefined],
        value: t.Any,
        args: t.Optional[t.Sequence[t.Any]],
        kwargs: t.Optional[t.Mapping[str, t.Any]],
        context: t.Optional[Context],
        eval_ctx: t.Optional[EvalContext],
        is_filter: bool,
    ) -> t.Any:
        if is_filter:
            env_map = self.filters
            type_name = "filter"
        else:
            env_map = self.tests
            type_name = "test"

        func = env_map.get(name)  # type: ignore

        if func is None:
            msg = f"No {type_name} named {name!r}."

            if isinstance(name, Undefined):
                try:
                    name._fail_with_undefined_error()
                except Exception as e:
                    msg = f"{msg} ({e}; did you forget to quote the callable name?)"

            raise TemplateRuntimeError(msg)

        args = [value, *(args if args is not None else ())]
        kwargs = kwargs if kwargs is not None else {}
        pass_arg = _PassArg.from_obj(func)

        if pass_arg is _PassArg.context:
            if context is None:
                raise TemplateRuntimeError(
                    f"Attempted to invoke a context {type_name} without context."
                )

            args.insert(0, context)
        elif pass_arg is _PassArg.eval_context:
            if eval_ctx is None:
                if context is not None:
                    eval_ctx = context.eval_ctx
                else:
                    eval_ctx = EvalContext(self)

            args.insert(0, eval_ctx)
        elif pass_arg is _PassArg.environment:
            args.insert(0, self)

        return func(*args, **kwargs)

    def call_filter(
        self,
        name: str,
        value: t.Any,
        args: t.Optional[t.Sequence[t.Any]] = None,
        kwargs: t.Optional[t.Mapping[str, t.Any]] = None,
        context: t.Optional[Context] = None,
        eval_ctx: t.Optional[EvalContext] = None,
    ) -> t.Any:
        """Invoke a filter on a value the same way the compiler does.

        This might return a coroutine if the filter is running from an
        environment in async mode and the filter supports async
        execution. It's your responsibility to await this if needed.

        .. versionadded:: 2.7
        """
        return self._filter_test_common(
            name, value, args, kwargs, context, eval_ctx, True
        )

    def call_test(
        self,
        name: str,
        value: t.Any,
        args: t.Optional[t.Sequence[t.Any]] = None,
        kwargs: t.Optional[t.Mapping[str, t.Any]] = None,
        context: t.Optional[Context] = None,
        eval_ctx: t.Optional[EvalContext] = None,
    ) -> t.Any:
        """Invoke a test on a value the same way the compiler does.

        This might return a coroutine if the test is running from an
        environment in async mode and the test supports async execution.
        It's your responsibility to await this if needed.

        .. versionchanged:: 3.0
            Tests support ``@pass_context``, etc. decorators. Added
            the ``context`` and ``eval_ctx`` parameters.

        .. versionadded:: 2.7
        """
        return self._filter_test_common(
            name, value, args, kwargs, context, eval_ctx, False
        )

    @internalcode
    def parse(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> nodes.Template:
        """Parse the sourcecode and return the abstract syntax tree.  This
        tree of nodes is used by the compiler to convert the template into
        executable source- or bytecode.  This is useful for debugging or to
        extract information from templates.

        If you are :ref:`developing Jinja extensions <writing-extensions>`
        this gives you a good overview of the node tree generated.
        """
        try:
            return self._parse(source, name, filename)
        except TemplateSyntaxError:
            self.handle_exception(source=source)

    def _parse(
        self, source: str, name: t.Optional[str], filename: t.Optional[str]
    ) -> nodes.Template:
        """Internal parsing function used by `parse` and `compile`."""
        return Parser(self, source, name, filename).parse()

    def lex(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> t.Iterator[t.Tuple[int, str, str]]:
        """Lex the given sourcecode and return a generator that yields
        tokens as tuples in the form ``(lineno, token_type, value)``.
        This can be useful for :ref:`extension development <writing-extensions>`
        and debugging templates.

        This does not perform preprocessing.  If you want the preprocessing
        of the extensions to be applied you have to filter source through
        the :meth:`preprocess` method.
        """
        source = str(source)
        try:
            return self.lexer.tokeniter(source, name, filename)
        except TemplateSyntaxError:
            self.handle_exception(source=source)

    def preprocess(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> str:
        """Preprocesses the source with all extensions.  This is automatically
        called for all parsing and compiling methods but *not* for :meth:`lex`
        because there you usually only want the actual source tokenized.
        """
        return reduce(
            lambda s, e: e.preprocess(s, name, filename),
            self.iter_extensions(),
            str(source),
        )

    def _tokenize(
        self,
        source: str,
        name: t.Optional[str],
        filename: t.Optional[str] = None,
        state: t.Optional[str] = None,
    ) -> TokenStream:
        """Called by the parser to do the preprocessing and filtering
        for all the extensions.  Returns a :class:`~jinja2.lexer.TokenStream`.
        """
        source = self.preprocess(source, name, filename)
        stream = self.lexer.tokenize(source, name, filename, state)

        for ext in self.iter_extensions():
            stream = ext.filter_stream(stream)  # type: ignore

            if not isinstance(stream, TokenStream):
                stream = TokenStream(stream, name, filename)

        return stream

    def _generate(
        self,
        source: nodes.Template,
        name: t.Optional[str],
        filename: t.Optional[str],
        defer_init: bool = False,
    ) -> str:
        """Internal hook that can be overridden to hook a different generate
        method in.

        .. versionadded:: 2.5
        """
        return generate(  # type: ignore
            source,
            self,
            name,
            filename,
            defer_init=defer_init,
            optimized=self.optimized,
        )

    def _compile(self, source: str, filename: str) -> CodeType:
        """Internal hook that can be overridden to hook a different compile
        method in.

        .. versionadded:: 2.5
        """
        return compile(source, filename, "exec")

    @typing.overload
    def compile(
        self,
        source: t.Union[str, nodes.Template],
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        raw: "te.Literal[False]" = False,
        defer_init: bool = False,
    ) -> CodeType: ...

    @typing.overload
    def compile(
        self,
        source: t.Union[str, nodes.Template],
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        raw: "te.Literal[True]" = ...,
        defer_init: bool = False,
    ) -> str: ...

    @internalcode
    def compile(
        self,
        source: t.Union[str, nodes.Template],
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        raw: bool = False,
        defer_init: bool = False,
    ) -> t.Union[str, CodeType]:
        """Compile a node or template source code.  The `name` parameter is
        the load name of the template after it was joined using
        :meth:`join_path` if necessary, not the filename on the file system.
        the `filename` parameter is the estimated filename of the template on
        the file system.  If the template came from a database or memory this
        can be omitted.

        The return value of this method is a python code object.  If the `raw`
        parameter is `True` the return value will be a string with python
        code equivalent to the bytecode returned otherwise.  This method is
        mainly used internally.

        `defer_init` is use internally to aid the module code generator.  This
        causes the generated code to be able to import without the global
        environment variable to be set.

        .. versionadded:: 2.4
           `defer_init` parameter added.
        """
        source_hint = None
        try:
            if isinstance(source, str):
                source_hint = source
                source = self._parse(source, name, filename)
            source = self._generate(source, name, filename, defer_init=defer_init)
            if raw:
                return source
            if filename is None:
                filename = "<template>"
            return self._compile(source, filename)
        except TemplateSyntaxError:
            self.handle_exception(source=source_hint)

    def compile_expression(
        self, source: str, undefined_to_none: bool = True
    ) -> "TemplateExpression":
        """A handy helper method that returns a callable that accepts keyword
        arguments that appear as variables in the expression.  If called it
        returns the result of the expression.

        This is useful if applications want to use the same rules as Jinja
        in template "configuration files" or similar situations.

        Example usage:

        >>> env = Environment()
        >>> expr = env.compile_expression('foo == 42')
        >>> expr(foo=23)
        False
        >>> expr(foo=42)
        True

        Per default the return value is converted to `None` if the
        expression returns an undefined value.  This can be changed
        by setting `undefined_to_none` to `False`.

        >>> env.compile_expression('var')() is None
        True
        >>> env.compile_expression('var', undefined_to_none=False)()
        Undefined

        .. versionadded:: 2.1
        """
        parser = Parser(self, source, state="variable")
        try:
            expr = parser.parse_expression()
            if not parser.stream.eos:
                raise TemplateSyntaxError(
                    "chunk after expression", parser.stream.current.lineno, None, None
                )
            expr.set_environment(self)
        except TemplateSyntaxError:
            self.handle_exception(source=source)

        body = [nodes.Assign(nodes.Name("result", "store"), expr, lineno=1)]
        template = self.from_string(nodes.Template(body, lineno=1))
        return TemplateExpression(template, undefined_to_none)

    def compile_templates(
        self,
        target: t.Union[str, "os.PathLike[str]"],
        extensions: t.Optional[t.Collection[str]] = None,
        filter_func: t.Optional[t.Callable[[str], bool]] = None,
        zip: t.Optional[str] = "deflated",
        log_function: t.Optional[t.Callable[[str], None]] = None,
        ignore_errors: bool = True,
    ) -> None:
        """Finds all the templates the loader can find, compiles them
        and stores them in `target`.  If `zip` is `None`, instead of in a
        zipfile, the templates will be stored in a directory.
        By default a deflate zip algorithm is used. To switch to
        the stored algorithm, `zip` can be set to ``'stored'``.

        `extensions` and `filter_func` are passed to :meth:`list_templates`.
        Each template returned will be compiled to the target folder or
        zipfile.

        By default template compilation errors are ignored.  In case a
        log function is provided, errors are logged.  If you want template
        syntax errors to abort the compilation you can set `ignore_errors`
        to `False` and you will get an exception on syntax errors.

        .. versionadded:: 2.4
        """
        from .loaders import ModuleLoader

        if log_function is None:

            def log_function(x: str) -> None:
                pass

        assert log_function is not None
        assert self.loader is not None, "No loader configured."

        def write_file(filename: str, data: str) -> None:
            if zip:
                info = ZipInfo(filename)
                info.external_attr = 0o755 << 16
                zip_file.writestr(info, data)
            else:
                with open(os.path.join(target, filename), "wb") as f:
                    f.write(data.encode("utf8"))

        if zip is not None:
            from zipfile import ZIP_DEFLATED
            from zipfile import ZIP_STORED
            from zipfile import ZipFile
            from zipfile import ZipInfo

            zip_file = ZipFile(
                target, "w", dict(deflated=ZIP_DEFLATED, stored=ZIP_STORED)[zip]
            )
            log_function(f"Compiling into Zip archive {target!r}")
        else:
            if not os.path.isdir(target):
                os.makedirs(target)
            log_function(f"Compiling into folder {target!r}")

        try:
            for name in self.list_templates(extensions, filter_func):
                source, filename, _ = self.loader.get_source(self, name)
                try:
                    code = self.compile(source, name, filename, True, True)
                except TemplateSyntaxError as e:
                    if not ignore_errors:
                        raise
                    log_function(f'Could not compile "{name}": {e}')
                    continue

                filename = ModuleLoader.get_module_filename(name)

                write_file(filename, code)
                log_function(f'Compiled "{name}" as {filename}')
        finally:
            if zip:
                zip_file.close()

        log_function("Finished compiling templates")

    def list_templates(
        self,
        extensions: t.Optional[t.Collection[str]] = None,
        filter_func: t.Optional[t.Callable[[str], bool]] = None,
    ) -> t.List[str]:
        """Returns a list of templates for this environment.  This requires
        that the loader supports the loader's
        :meth:`~BaseLoader.list_templates` method.

        If there are other files in the template folder besides the
        actual templates, the returned list can be filtered.  There are two
        ways: either `extensions` is set to a list of file extensions for
        templates, or a `filter_func` can be provided which is a callable that
        is passed a template name and should return `True` if it should end up
        in the result list.

        If the loader does not support that, a :exc:`TypeError` is raised.

        .. versionadded:: 2.4
        """
        assert self.loader is not None, "No loader configured."
        names = self.loader.list_templates()

        if extensions is not None:
            if filter_func is not None:
                raise TypeError(
                    "either extensions or filter_func can be passed, but not both"
                )

            def filter_func(x: str) -> bool:
                return "." in x and x.rsplit(".", 1)[1] in extensions

        if filter_func is not None:
            names = [name for name in names if filter_func(name)]

        return names

    def handle_exception(self, source: t.Optional[str] = None) -> "te.NoReturn":
        """Exception handling helper.  This is used internally to either raise
        rewritten exceptions or return a rendered traceback for the template.
        """
        from .debug import rewrite_traceback_stack

        raise rewrite_traceback_stack(source=source)

    def join_path(self, template: str, parent: str) -> str:
        """Join a template with the parent.  By default all the lookups are
        relative to the loader root so this method returns the `template`
        parameter unchanged, but if the paths should be relative to the
        parent template, this function can be used to calculate the real
        template name.

        Subclasses may override this method and implement template path
        joining here.
        """
        return template

    @internalcode
    def _load_template(
        self, name: str, globals: t.Optional[t.MutableMapping[str, t.Any]]
    ) -> "Template":
        if self.loader is None:
            raise TypeError("no loader for this environment specified")
        cache_key = (weakref.ref(self.loader), name)
        if self.cache is not None:
            template = self.cache.get(cache_key)
            if template is not None and (
                not self.auto_reload or template.is_up_to_date
            ):
                # template.globals is a ChainMap, modifying it will only
                # affect the template, not the environment globals.
                if globals:
                    template.globals.update(globals)

                return template

        template = self.loader.load(self, name, self.make_globals(globals))

        if self.cache is not None:
            self.cache[cache_key] = template
        return template

    @internalcode
    def get_template(
        self,
        name: t.Union[str, "Template"],
        parent: t.Optional[str] = None,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Load a template by name with :attr:`loader` and return a
        :class:`Template`. If the template does not exist a
        :exc:`TemplateNotFound` exception is raised.

        :param name: Name of the template to load. When loading
            templates from the filesystem, "/" is used as the path
            separator, even on Windows.
        :param parent: The name of the parent template importing this
            template. :meth:`join_path` can be used to implement name
            transformations with this.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.

        .. versionchanged:: 3.0
            If a template is loaded from cache, ``globals`` will update
            the template's globals instead of ignoring the new values.

        .. versionchanged:: 2.4
            If ``name`` is a :class:`Template` object it is returned
            unchanged.
        """
        if isinstance(name, Template):
            return name
        if parent is not None:
            name = self.join_path(name, parent)

        return self._load_template(name, globals)

    @internalcode
    def select_template(
        self,
        names: t.Iterable[t.Union[str, "Template"]],
        parent: t.Optional[str] = None,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Like :meth:`get_template`, but tries loading multiple names.
        If none of the names can be loaded a :exc:`TemplatesNotFound`
        exception is raised.

        :param names: List of template names to try loading in order.
        :param parent: The name of the parent template importing this
            template. :meth:`join_path` can be used to implement name
            transformations with this.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.

        .. versionchanged:: 3.0
            If a template is loaded from cache, ``globals`` will update
            the template's globals instead of ignoring the new values.

        .. versionchanged:: 2.11
            If ``names`` is :class:`Undefined`, an :exc:`UndefinedError`
            is raised instead. If no templates were found and ``names``
            contains :class:`Undefined`, the message is more helpful.

        .. versionchanged:: 2.4
            If ``names`` contains a :class:`Template` object it is
            returned unchanged.

        .. versionadded:: 2.3
        """
        if isinstance(names, Undefined):
            names._fail_with_undefined_error()

        if not names:
            raise TemplatesNotFound(
                message="Tried to select from an empty list of templates."
            )

        for name in names:
            if isinstance(name, Template):
                return name
            if parent is not None:
                name = self.join_path(name, parent)
            try:
                return self._load_template(name, globals)
            except (TemplateNotFound, UndefinedError):
                pass
        raise TemplatesNotFound(names)  # type: ignore

    @internalcode
    def get_or_select_template(
        self,
        template_name_or_list: t.Union[
            str, "Template", t.List[t.Union[str, "Template"]]
        ],
        parent: t.Optional[str] = None,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Use :meth:`select_template` if an iterable of template names
        is given, or :meth:`get_template` if one name is given.

        .. versionadded:: 2.3
        """
        if isinstance(template_name_or_list, (str, Undefined)):
            return self.get_template(template_name_or_list, parent, globals)
        elif isinstance(template_name_or_list, Template):
            return template_name_or_list
        return self.select_template(template_name_or_list, parent, globals)

    def from_string(
        self,
        source: t.Union[str, nodes.Template],
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
        template_class: t.Optional[t.Type["Template"]] = None,
    ) -> "Template":
        """Load a template from a source string without using
        :attr:`loader`.

        :param source: Jinja source to compile into a template.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.
        :param template_class: Return an instance of this
            :class:`Template` class.
        """
        gs = self.make_globals(globals)
        cls = template_class or self.template_class
        return cls.from_code(self, self.compile(source), gs, None)

    def make_globals(
        self, d: t.Optional[t.MutableMapping[str, t.Any]]
    ) -> t.MutableMapping[str, t.Any]:
        """Make the globals map for a template. Any given template
        globals overlay the environment :attr:`globals`.

        Returns a :class:`collections.ChainMap`. This allows any changes
        to a template's globals to only affect that template, while
        changes to the environment's globals are still reflected.
        However, avoid modifying any globals after a template is loaded.

        :param d: Dict of template-specific globals.

        .. versionchanged:: 3.0
            Use :class:`collections.ChainMap` to always prevent mutating
            environment globals.
        """
        if d is None:
            d = {}

        return ChainMap(d, self.globals)


class Template:
    """A compiled template that can be rendered.

    Use the methods on :class:`Environment` to create or load templates.
    The environment is used to configure how templates are compiled and
    behave.

    It is also possible to create a template object directly. This is
    not usually recommended. The constructor takes most of the same
    arguments as :class:`Environment`. All templates created with the
    same environment arguments share the same ephemeral ``Environment``
    instance behind the scenes.

    A template object should be considered immutable. Modifications on
    the object are not supported.
    """

    #: Type of environment to create when creating a template directly
    #: rather than through an existing environment.
    environment_class: t.Type[Environment] = Environment

    environment: Environment
    globals: t.MutableMapping[str, t.Any]
    name: t.Optional[str]
    filename: t.Optional[str]
    blocks: t.Dict[str, t.Callable[[Context], t.Iterator[str]]]
    root_render_func: t.Callable[[Context], t.Iterator[str]]
    _module: t.Optional["TemplateModule"]
    _debug_info: str
    _uptodate: t.Optional[t.Callable[[], bool]]

    def __new__(
        cls,
        source: t.Union[str, nodes.Template],
        block_start_string: str = BLOCK_START_STRING,
        block_end_string: str = BLOCK_END_STRING,
        variable_start_string: str = VARIABLE_START_STRING,
        variable_end_string: str = VARIABLE_END_STRING,
        comment_start_string: str = COMMENT_START_STRING,
        comment_end_string: str = COMMENT_END_STRING,
        line_statement_prefix: t.Optional[str] = LINE_STATEMENT_PREFIX,
        line_comment_prefix: t.Optional[str] = LINE_COMMENT_PREFIX,
        trim_blocks: bool = TRIM_BLOCKS,
        lstrip_blocks: bool = LSTRIP_BLOCKS,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = NEWLINE_SEQUENCE,
        keep_trailing_newline: bool = KEEP_TRAILING_NEWLINE,
        extensions: t.Sequence[t.Union[str, t.Type["Extension"]]] = (),
        optimized: bool = True,
        undefined: t.Type[Undefined] = Undefined,
        finalize: t.Optional[t.Callable[..., t.Any]] = None,
        autoescape: t.Union[bool, t.Callable[[t.Optional[str]], bool]] = False,
        enable_async: bool = False,
    ) -> t.Any:  # it returns a `Template`, but this breaks the sphinx build...
        env = get_spontaneous_environment(
            cls.environment_class,  # type: ignore
            block_start_string,
            block_end_string,
            variable_start_string,
            variable_end_string,
            comment_start_string,
            comment_end_string,
            line_statement_prefix,
            line_comment_prefix,
            trim_blocks,
            lstrip_blocks,
            newline_sequence,
            keep_trailing_newline,
            frozenset(extensions),
            optimized,
            undefined,  # type: ignore
            finalize,
            autoescape,
            None,
            0,
            False,
            None,
            enable_async,
        )
        return env.from_string(source, template_class=cls)

    @classmethod
    def from_code(
        cls,
        environment: Environment,
        code: CodeType,
        globals: t.MutableMapping[str, t.Any],
        uptodate: t.Optional[t.Callable[[], bool]] = None,
    ) -> "Template":
        """Creates a template object from compiled code and the globals.  This
        is used by the loaders and environment to create a template object.
        """
        namespace = {"environment": environment, "__file__": code.co_filename}
        exec(code, namespace)
        rv = cls._from_namespace(environment, namespace, globals)
        rv._uptodate = uptodate
        return rv

    @classmethod
    def from_module_dict(
        cls,
        environment: Environment,
        module_dict: t.MutableMapping[str, t.Any],
        globals: t.MutableMapping[str, t.Any],
    ) -> "Template":
        """Creates a template object from a module.  This is used by the
        module loader to create a template object.

        .. versionadded:: 2.4
        """
        return cls._from_namespace(environment, module_dict, globals)

    @classmethod
    def _from_namespace(
        cls,
        environment: Environment,
        namespace: t.MutableMapping[str, t.Any],
        globals: t.MutableMapping[str, t.Any],
    ) -> "Template":
        t: Template = object.__new__(cls)
        t.environment = environment
        t.globals = globals
        t.name = namespace["name"]
        t.filename = namespace["__file__"]
        t.blocks = namespace["blocks"]

        # render function and module
        t.root_render_func = namespace["root"]
        t._module = None

        # debug and loader helpers
        t._debug_info = namespace["debug_info"]
        t._uptodate = None

        # store the reference
        namespace["environment"] = environment
        namespace["__jinja_template__"] = t

        return t

    def render(self, *args: t.Any, **kwargs: t.Any) -> str:
        """This method accepts the same arguments as the `dict` constructor:
        A dict, a dict subclass or some keyword arguments.  If no arguments
        are given the context will be empty.  These two calls do the same::

            template.render(knights='that say nih')
            template.render({'knights': 'that say nih'})

        This will return the rendered template as a string.
        """
        if self.environment.is_async:
            import asyncio

            return asyncio.run(self.render_async(*args, **kwargs))

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            return self.environment.concat(self.root_render_func(ctx))  # type: ignore
        except Exception:
            self.environment.handle_exception()

    async def render_async(self, *args: t.Any, **kwargs: t.Any) -> str:
        """This works similar to :meth:`render` but returns a coroutine
        that when awaited returns the entire rendered template string.  This
        requires the async feature to be enabled.

        Example usage::

            await template.render_async(knights='that say nih; asynchronously')
        """
        if not self.environment.is_async:
            raise RuntimeError(
                "The environment was not created with async mode enabled."
            )

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            return self.environment.concat(  # type: ignore
                [n async for n in self.root_render_func(ctx)]  # type: ignore
            )
        except Exception:
            return self.environment.handle_exception()

    def stream(self, *args: t.Any, **kwargs: t.Any) -> "TemplateStream":
        """Works exactly like :meth:`generate` but returns a
        :class:`TemplateStream`.
        """
        return TemplateStream(self.generate(*args, **kwargs))

    def generate(self, *args: t.Any, **kwargs: t.Any) -> t.Iterator[str]:
        """For very large templates it can be useful to not render the whole
        template at once but evaluate each statement after another and yield
        piece for piece.  This method basically does exactly that and returns
        a generator that yields one item after another as strings.

        It accepts the same arguments as :meth:`render`.
        """
        if self.environment.is_async:
            import asyncio

            async def to_list() -> t.List[str]:
                return [x async for x in self.generate_async(*args, **kwargs)]

            yield from asyncio.run(to_list())
            return

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            yield from self.root_render_func(ctx)
        except Exception:
            yield self.environment.handle_exception()

    async def generate_async(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.AsyncGenerator[str, object]:
        """An async version of :meth:`generate`.  Works very similarly but
        returns an async iterator instead.
        """
        if not self.environment.is_async:
            raise RuntimeError(
                "The environment was not created with async mode enabled."
            )

        ctx = self.new_context(dict(*args, **kwargs))

        try:
            agen = self.root_render_func(ctx)
            try:
                async for event in agen:  # type: ignore
                    yield event
            finally:
                # we can't use async with aclosing(...) because that's only
                # in 3.10+
                await agen.aclose()  # type: ignore
        except Exception:
            yield self.environment.handle_exception()

    def new_context(
        self,
        vars: t.Optional[t.Dict[str, t.Any]] = None,
        shared: bool = False,
        locals: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> Context:
        """Create a new :class:`Context` for this template.  The vars
        provided will be passed to the template.  Per default the globals
        are added to the context.  If shared is set to `True` the data
        is passed as is to the context without adding the globals.

        `locals` can be a dict of local variables for internal usage.
        """
        return new_context(
            self.environment, self.name, self.blocks, vars, shared, self.globals, locals
        )

    def make_module(
        self,
        vars: t.Optional[t.Dict[str, t.Any]] = None,
        shared: bool = False,
        locals: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> "TemplateModule":
        """This method works like the :attr:`module` attribute when called
        without arguments but it will evaluate the template on every call
        rather than caching it.  It's also possible to provide
        a dict which is then used as context.  The arguments are the same
        as for the :meth:`new_context` method.
        """
        ctx = self.new_context(vars, shared, locals)
        return TemplateModule(self, ctx)

    async def make_module_async(
        self,
        vars: t.Optional[t.Dict[str, t.Any]] = None,
        shared: bool = False,
        locals: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> "TemplateModule":
        """As template module creation can invoke template code for
        asynchronous executions this method must be used instead of the
        normal :meth:`make_module` one.  Likewise the module attribute
        becomes unavailable in async mode.
        """
        ctx = self.new_context(vars, shared, locals)
        return TemplateModule(
            self,
            ctx,
            [x async for x in self.root_render_func(ctx)],  # type: ignore
        )

    @internalcode
    def _get_default_module(self, ctx: t.Optional[Context] = None) -> "TemplateModule":
        """If a context is passed in, this means that the template was
        imported. Imported templates have access to the current
        template's globals by default, but they can only be accessed via
        the context during runtime.

        If there are new globals, we need to create a new module because
        the cached module is already rendered and will not have access
        to globals from the current context. This new module is not
        cached because the template can be imported elsewhere, and it
        should have access to only the current template's globals.
        """
        if self.environment.is_async:
            raise RuntimeError("Module is not available in async mode.")

        if ctx is not None:
            keys = ctx.globals_keys - self.globals.keys()

            if keys:
                return self.make_module({k: ctx.parent[k] for k in keys})

        if self._module is None:
            self._module = self.make_module()

        return self._module

    async def _get_default_module_async(
        self, ctx: t.Optional[Context] = None
    ) -> "TemplateModule":
        if ctx is not None:
            keys = ctx.globals_keys - self.globals.keys()

            if keys:
                return await self.make_module_async({k: ctx.parent[k] for k in keys})

        if self._module is None:
            self._module = await self.make_module_async()

        return self._module

    @property
    def module(self) -> "TemplateModule":
        """The template as module.  This is used for imports in the
        template runtime but is also useful if one wants to access
        exported template variables from the Python layer:

        >>> t = Template('{% macro foo() %}42{% endmacro %}23')
        >>> str(t.module)
        '23'
        >>> t.module.foo() == u'42'
        True

        This attribute is not available if async mode is enabled.
        """
        return self._get_default_module()

    def get_corresponding_lineno(self, lineno: int) -> int:
        """Return the source line number of a line number in the
        generated bytecode as they are not in sync.
        """
        for template_line, code_line in reversed(self.debug_info):
            if code_line <= lineno:
                return template_line
        return 1

    @property
    def is_up_to_date(self) -> bool:
        """If this variable is `False` there is a newer version available."""
        if self._uptodate is None:
            return True
        return self._uptodate()

    @property
    def debug_info(self) -> t.List[t.Tuple[int, int]]:
        """The debug info mapping."""
        if self._debug_info:
            return [
                tuple(map(int, x.split("=")))  # type: ignore
                for x in self._debug_info.split("&")
            ]

        return []

    def __repr__(self) -> str:
        if self.name is None:
            name = f"memory:{id(self):x}"
        else:
            name = repr(self.name)
        return f"<{type(self).__name__} {name}>"


class TemplateModule:
    """Represents an imported template.  All the exported names of the
    template are available as attributes on this object.  Additionally
    converting it into a string renders the contents.
    """

    def __init__(
        self,
        template: Template,
        context: Context,
        body_stream: t.Optional[t.Iterable[str]] = None,
    ) -> None:
        if body_stream is None:
            if context.environment.is_async:
                raise RuntimeError(
                    "Async mode requires a body stream to be passed to"
                    " a template module. Use the async methods of the"
                    " API you are using."
                )

            body_stream = list(template.root_render_func(context))

        self._body_stream = body_stream
        self.__dict__.update(context.get_exported())
        self.__name__ = template.name

    def __html__(self) -> Markup:
        return Markup(concat(self._body_stream))

    def __str__(self) -> str:
        return concat(self._body_stream)

    def __repr__(self) -> str:
        if self.__name__ is None:
            name = f"memory:{id(self):x}"
        else:
            name = repr(self.__name__)
        return f"<{type(self).__name__} {name}>"


class TemplateExpression:
    """The :meth:`jinja2.Environment.compile_expression` method returns an
    instance of this object.  It encapsulates the expression-like access
    to the template with an expression it wraps.
    """

    def __init__(self, template: Template, undefined_to_none: bool) -> None:
        self._template = template
        self._undefined_to_none = undefined_to_none

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Optional[t.Any]:
        context = self._template.new_context(dict(*args, **kwargs))
        consume(self._template.root_render_func(context))
        rv = context.vars["result"]
        if self._undefined_to_none and isinstance(rv, Undefined):
            rv = None
        return rv


class TemplateStream:
    """A template stream works pretty much like an ordinary python generator
    but it can buffer multiple items to reduce the number of total iterations.
    Per default the output is unbuffered which means that for every unbuffered
    instruction in the template one string is yielded.

    If buffering is enabled with a buffer size of 5, five items are combined
    into a new string.  This is mainly useful if you are streaming
    big templates to a client via WSGI which flushes after each iteration.
    """

    def __init__(self, gen: t.Iterator[str]) -> None:
        self._gen = gen
        self.disable_buffering()

    def dump(
        self,
        fp: t.Union[str, t.IO[bytes]],
        encoding: t.Optional[str] = None,
        errors: t.Optional[str] = "strict",
    ) -> None:
        """Dump the complete stream into a file or file-like object.
        Per default strings are written, if you want to encode
        before writing specify an `encoding`.

        Example usage::

            Template('Hello {{ name }}!').stream(name='foo').dump('hello.html')
        """
        close = False

        if isinstance(fp, str):
            if encoding is None:
                encoding = "utf-8"

            real_fp: t.IO[bytes] = open(fp, "wb")
            close = True
        else:
            real_fp = fp

        try:
            if encoding is not None:
                iterable = (x.encode(encoding, errors) for x in self)  # type: ignore
            else:
                iterable = self  # type: ignore

            if hasattr(real_fp, "writelines"):
                real_fp.writelines(iterable)
            else:
                for item in iterable:
                    real_fp.write(item)
        finally:
            if close:
                real_fp.close()

    def disable_buffering(self) -> None:
        """Disable the output buffering."""
        self._next = partial(next, self._gen)
        self.buffered = False

    def _buffered_generator(self, size: int) -> t.Iterator[str]:
        buf: t.List[str] = []
        c_size = 0
        push = buf.append

        while True:
            try:
                while c_size < size:
                    c = next(self._gen)
                    push(c)
                    if c:
                        c_size += 1
            except StopIteration:
                if not c_size:
                    return
            yield concat(buf)
            del buf[:]
            c_size = 0

    def enable_buffering(self, size: int = 5) -> None:
        """Enable buffering.  Buffer `size` items before yielding them."""
        if size <= 1:
            raise ValueError("buffer size too small")

        self.buffered = True
        self._next = partial(next, self._buffered_generator(size))

    def __iter__(self) -> "TemplateStream":
        return self

    def __next__(self) -> str:
        return self._next()  # type: ignore


# hook in default template class.  if anyone reads this comment: ignore that
# it's possible to use custom templates ;-)
Environment.template_class = Template

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\masked.py ===
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas._libs.tslibs import is_supported_dtype
from pandas._typing import (
    ArrayLike,
    AstypeArg,
    AxisInt,
    DtypeObj,
    FillnaOptions,
    InterpolateOptions,
    NpDtype,
    PositionalIndexer,
    Scalar,
    ScalarIndexer,
    Self,
    SequenceIndexer,
    Shape,
    npt,
)
from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_fillna_kwargs

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    is_bool,
    is_integer_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import BaseMaskedDtype
from pandas.core.dtypes.missing import (
    array_equivalent,
    is_valid_na_for_dtype,
    isna,
    notna,
)

from pandas.core import (
    algorithms as algos,
    arraylike,
    missing,
    nanops,
    ops,
)
from pandas.core.algorithms import (
    factorize_array,
    isin,
    map_array,
    mode,
    take,
)
from pandas.core.array_algos import (
    masked_accumulations,
    masked_reductions,
)
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import ExtensionArray
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import invalid_comparison
from pandas.core.util.hashing import hash_array

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )
    from pandas import Series
    from pandas.core.arrays import BooleanArray
    from pandas._typing import (
        NumpySorter,
        NumpyValueArrayLike,
    )
    from pandas.core.arrays import FloatingArray

from pandas.compat.numpy import function as nv


class BaseMaskedArray(OpsMixin, ExtensionArray):
    """
    Base class for masked arrays (which use _data and _mask to store the data).

    numpy based
    """

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value: Scalar
    # our underlying data and mask are each ndarrays
    _data: np.ndarray
    _mask: npt.NDArray[np.bool_]

    # Fill values used for any/all
    _truthy_value = Scalar  # bool(_truthy_value) = True
    _falsey_value = Scalar  # bool(_falsey_value) = False

    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self:
        result = BaseMaskedArray.__new__(cls)
        result._data = values
        result._mask = mask
        return result

    def __init__(
        self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = False
    ) -> None:
        # values is supposed to already be validated in the subclass
        if not (isinstance(mask, np.ndarray) and mask.dtype == np.bool_):
            raise TypeError(
                "mask should be boolean numpy array. Use "
                "the 'pd.array' function instead"
            )
        if values.shape != mask.shape:
            raise ValueError("values.shape must match mask.shape")

        if copy:
            values = values.copy()
            mask = mask.copy()

        self._data = values
        self._mask = mask

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False) -> Self:
        values, mask = cls._coerce_to_array(scalars, dtype=dtype, copy=copy)
        return cls(values, mask)

    @classmethod
    @doc(ExtensionArray._empty)
    def _empty(cls, shape: Shape, dtype: ExtensionDtype):
        values = np.empty(shape, dtype=dtype.type)
        values.fill(cls._internal_fill_value)
        mask = np.ones(shape, dtype=bool)
        result = cls(values, mask)
        if not isinstance(result, cls) or dtype != result.dtype:
            raise NotImplementedError(
                f"Default 'empty' implementation is invalid for dtype='{dtype}'"
            )
        return result

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        # NEP 51: https://github.com/numpy/numpy/pull/22449
        return str

    @property
    def dtype(self) -> BaseMaskedDtype:
        raise AbstractMethodError(self)

    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any:
        ...

    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self:
        ...

    def __getitem__(self, item: PositionalIndexer) -> Self | Any:
        item = check_array_indexer(self, item)

        newmask = self._mask[item]
        if is_bool(newmask):
            # This is a scalar indexing
            if newmask:
                return self.dtype.na_value
            return self._data[item]

        return self._simple_new(self._data[item], newmask)

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        mask = self._mask

        if mask.any():
            func = missing.get_fill_func(method, ndim=self.ndim)

            npvalues = self._data.T
            new_mask = mask.T
            if copy:
                npvalues = npvalues.copy()
                new_mask = new_mask.copy()
            elif limit_area is not None:
                mask = mask.copy()
            func(npvalues, limit=limit, mask=new_mask)

            if limit_area is not None and not mask.all():
                mask = mask.T
                neg_mask = ~mask
                first = neg_mask.argmax()
                last = len(neg_mask) - neg_mask[::-1].argmax() - 1
                if limit_area == "inside":
                    new_mask[:first] |= mask[:first]
                    new_mask[last + 1 :] |= mask[last + 1 :]
                elif limit_area == "outside":
                    new_mask[first + 1 : last] |= mask[first + 1 : last]

            if copy:
                return self._simple_new(npvalues.T, new_mask.T)
            else:
                return self
        else:
            if copy:
                new_values = self.copy()
            else:
                new_values = self
        return new_values

    @doc(ExtensionArray.fillna)
    def fillna(
        self, value=None, method=None, limit: int | None = None, copy: bool = True
    ) -> Self:
        value, method = validate_fillna_kwargs(value, method)

        mask = self._mask

        value = missing.check_value_size(value, mask, len(self))

        if mask.any():
            if method is not None:
                func = missing.get_fill_func(method, ndim=self.ndim)
                npvalues = self._data.T
                new_mask = mask.T
                if copy:
                    npvalues = npvalues.copy()
                    new_mask = new_mask.copy()
                func(npvalues, limit=limit, mask=new_mask)
                return self._simple_new(npvalues.T, new_mask.T)
            else:
                # fill with value
                if copy:
                    new_values = self.copy()
                else:
                    new_values = self[:]
                new_values[mask] = value
        else:
            if copy:
                new_values = self.copy()
            else:
                new_values = self[:]
        return new_values

    @classmethod
    def _coerce_to_array(
        cls, values, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        raise AbstractMethodError(cls)

    def _validate_setitem_value(self, value):
        """
        Check if we have a scalar that we can cast losslessly.

        Raises
        ------
        TypeError
        """
        kind = self.dtype.kind
        # TODO: get this all from np_can_hold_element?
        if kind == "b":
            if lib.is_bool(value):
                return value

        elif kind == "f":
            if lib.is_integer(value) or lib.is_float(value):
                return value

        else:
            if lib.is_integer(value) or (lib.is_float(value) and value.is_integer()):
                return value
            # TODO: unsigned checks

        # Note: without the "str" here, the f-string rendering raises in
        #  py38 builds.
        raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")

    def __setitem__(self, key, value) -> None:
        key = check_array_indexer(self, key)

        if is_scalar(value):
            if is_valid_na_for_dtype(value, self.dtype):
                self._mask[key] = True
            else:
                value = self._validate_setitem_value(value)
                self._data[key] = value
                self._mask[key] = False
            return

        value, mask = self._coerce_to_array(value, dtype=self.dtype)

        self._data[key] = value
        self._mask[key] = mask

    def __contains__(self, key) -> bool:
        if isna(key) and key is not self.dtype.na_value:
            # GH#52840
            if self._data.dtype.kind == "f" and lib.is_float(key):
                return bool((np.isnan(self._data) & ~self._mask).any())

        return bool(super().__contains__(key))

    def __iter__(self) -> Iterator:
        if self.ndim == 1:
            if not self._hasna:
                for val in self._data:
                    yield val
            else:
                na_value = self.dtype.na_value
                for isna_, val in zip(self._mask, self._data):
                    if isna_:
                        yield na_value
                    else:
                        yield val
        else:
            for i in range(len(self)):
                yield self[i]

    def __len__(self) -> int:
        return len(self._data)

    @property
    def shape(self) -> Shape:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    def swapaxes(self, axis1, axis2) -> Self:
        data = self._data.swapaxes(axis1, axis2)
        mask = self._mask.swapaxes(axis1, axis2)
        return self._simple_new(data, mask)

    def delete(self, loc, axis: AxisInt = 0) -> Self:
        data = np.delete(self._data, loc, axis=axis)
        mask = np.delete(self._mask, loc, axis=axis)
        return self._simple_new(data, mask)

    def reshape(self, *args, **kwargs) -> Self:
        data = self._data.reshape(*args, **kwargs)
        mask = self._mask.reshape(*args, **kwargs)
        return self._simple_new(data, mask)

    def ravel(self, *args, **kwargs) -> Self:
        # TODO: need to make sure we have the same order for data/mask
        data = self._data.ravel(*args, **kwargs)
        mask = self._mask.ravel(*args, **kwargs)
        return type(self)(data, mask)

    @property
    def T(self) -> Self:
        return self._simple_new(self._data.T, self._mask.T)

    def round(self, decimals: int = 0, *args, **kwargs):
        """
        Round each value in the array a to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        NumericArray
            Rounded values of the NumericArray.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
        if self.dtype.kind == "b":
            return self
        nv.validate_round(args, kwargs)
        values = np.round(self._data, decimals=decimals, **kwargs)

        # Usually we'll get same type as self, but ndarray[bool] casts to float
        return self._maybe_mask_result(values, self._mask.copy())

    # ------------------------------------------------------------------
    # Unary Methods

    def __invert__(self) -> Self:
        return self._simple_new(~self._data, self._mask.copy())

    def __neg__(self) -> Self:
        return self._simple_new(-self._data, self._mask.copy())

    def __pos__(self) -> Self:
        return self.copy()

    def __abs__(self) -> Self:
        return self._simple_new(abs(self._data), self._mask.copy())

    # ------------------------------------------------------------------

    def _values_for_json(self) -> np.ndarray:
        return np.asarray(self, dtype=object)

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert to a NumPy Array.

        By default converts to an object-dtype NumPy array. Specify the `dtype` and
        `na_value` keywords to customize the conversion.

        Parameters
        ----------
        dtype : dtype, default object
            The numpy dtype to convert to.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            the array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary. This is typically
            only possible when no missing values are present and `dtype`
            is the equivalent numpy dtype.
        na_value : scalar, optional
             Scalar missing value indicator to use in numpy array. Defaults
             to the native missing value indicator of this array (pd.NA).

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        An object-dtype is the default result

        >>> a = pd.array([True, False, pd.NA], dtype="boolean")
        >>> a.to_numpy()
        array([True, False, <NA>], dtype=object)

        When no missing values are present, an equivalent dtype can be used.

        >>> pd.array([True, False], dtype="boolean").to_numpy(dtype="bool")
        array([ True, False])
        >>> pd.array([1, 2], dtype="Int64").to_numpy("int64")
        array([1, 2])

        However, requesting such dtype will raise a ValueError if
        missing values are present and the default missing value :attr:`NA`
        is used.

        >>> a = pd.array([True, False, pd.NA], dtype="boolean")
        >>> a
        <BooleanArray>
        [True, False, <NA>]
        Length: 3, dtype: boolean

        >>> a.to_numpy(dtype="bool")
        Traceback (most recent call last):
        ...
        ValueError: cannot convert to bool numpy array in presence of missing values

        Specify a valid `na_value` instead

        >>> a.to_numpy(dtype="bool", na_value=False)
        array([ True, False, False])
        """
        hasna = self._hasna
        dtype, na_value = to_numpy_dtype_inference(self, dtype, na_value, hasna)
        if dtype is None:
            dtype = object

        if hasna:
            if (
                dtype != object
                and not is_string_dtype(dtype)
                and na_value is libmissing.NA
            ):
                raise ValueError(
                    f"cannot convert to '{dtype}'-dtype NumPy array "
                    "with missing values. Specify an appropriate 'na_value' "
                    "for this dtype."
                )
            # don't pass copy to astype -> always need a copy since we are mutating
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                data = self._data.astype(dtype)
            data[self._mask] = na_value
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                data = self._data.astype(dtype, copy=copy)
        return data

    @doc(ExtensionArray.tolist)
    def tolist(self):
        if self.ndim > 1:
            return [x.tolist() for x in self]
        dtype = None if self._hasna else self._data.dtype
        return self.to_numpy(dtype=dtype, na_value=libmissing.NA).tolist()

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray:
        ...

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray:
        ...

    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike:
        ...

    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        # if we are astyping to another nullable masked dtype, we can fastpath
        if isinstance(dtype, BaseMaskedDtype):
            # TODO deal with NaNs for FloatingArray case
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # TODO: Is rounding what we want long term?
                data = self._data.astype(dtype.numpy_dtype, copy=copy)
            # mask is copied depending on whether the data was copied, and
            # not directly depending on the `copy` keyword
            mask = self._mask if data is self._data else self._mask.copy()
            cls = dtype.construct_array_type()
            return cls(data, mask, copy=False)

        if isinstance(dtype, ExtensionDtype):
            eacls = dtype.construct_array_type()
            return eacls._from_sequence(self, dtype=dtype, copy=copy)

        na_value: float | np.datetime64 | lib.NoDefault

        # coerce
        if dtype.kind == "f":
            # In astype, we consider dtype=float to also mean na_value=np.nan
            na_value = np.nan
        elif dtype.kind == "M":
            na_value = np.datetime64("NaT")
        else:
            na_value = lib.no_default

        # to_numpy will also raise, but we get somewhat nicer exception messages here
        if dtype.kind in "iu" and self._hasna:
            raise ValueError("cannot convert NA to integer")
        if dtype.kind == "b" and self._hasna:
            # careful: astype_nansafe converts np.nan to True
            raise ValueError("cannot convert float NaN to bool")

        data = self.to_numpy(dtype=dtype, na_value=na_value, copy=copy)
        return data

    __array_priority__ = 1000  # higher than ndarray so ops dispatch to us

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        the array interface, return my values
        We return an object array here to preserve our scalar values
        """
        if copy is False:
            if not self._hasna:
                # special case, here we can simply return the underlying data
                return np.array(self._data, dtype=dtype, copy=copy)

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

        if copy is None:
            copy = False  # The NumPy copy=False meaning is different here.
        return self.to_numpy(dtype=dtype, copy=copy)

    _HANDLED_TYPES: tuple[type, ...]

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # For MaskedArray inputs, we apply the ufunc to ._data
        # and mask the result.

        out = kwargs.get("out", ())

        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (BaseMaskedArray,)):
                return NotImplemented

        # for binary ops, use our custom dunder methods
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. test_ufunc_with_out
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        mask = np.zeros(len(self), dtype=bool)
        inputs2 = []
        for x in inputs:
            if isinstance(x, BaseMaskedArray):
                mask |= x._mask
                inputs2.append(x._data)
            else:
                inputs2.append(x)

        def reconstruct(x: np.ndarray):
            # we don't worry about scalar `x` here, since we
            # raise for reduce up above.
            from pandas.core.arrays import (
                BooleanArray,
                FloatingArray,
                IntegerArray,
            )

            if x.dtype.kind == "b":
                m = mask.copy()
                return BooleanArray(x, m)
            elif x.dtype.kind in "iu":
                m = mask.copy()
                return IntegerArray(x, m)
            elif x.dtype.kind == "f":
                m = mask.copy()
                if x.dtype == np.float16:
                    # reached in e.g. np.sqrt on BooleanArray
                    # we don't support float16
                    x = x.astype(np.float32)
                return FloatingArray(x, m)
            else:
                x[mask] = np.nan
            return x

        result = getattr(ufunc, method)(*inputs2, **kwargs)
        if ufunc.nout > 1:
            # e.g. np.divmod
            return tuple(reconstruct(x) for x in result)
        elif method == "reduce":
            # e.g. np.add.reduce; test_ufunc_reduce_raises
            if self._mask.any():
                return self._na_value
            return result
        else:
            return reconstruct(result)

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow as pa

        return pa.array(self._data, mask=self._mask, type=type)

    @property
    def _hasna(self) -> bool:
        # Note: this is expensive right now! The hope is that we can
        # make this faster by having an optional mask, but not have to change
        # source code using it..

        # error: Incompatible return value type (got "bool_", expected "bool")
        return self._mask.any()  # type: ignore[return-value]

    def _propagate_mask(
        self, mask: npt.NDArray[np.bool_] | None, other
    ) -> npt.NDArray[np.bool_]:
        if mask is None:
            mask = self._mask.copy()  # TODO: need test for BooleanArray needing a copy
            if other is libmissing.NA:
                # GH#45421 don't alter inplace
                mask = mask | True
            elif is_list_like(other) and len(other) == len(mask):
                mask = mask | isna(other)
        else:
            mask = self._mask | mask
        # Incompatible return value type (got "Optional[ndarray[Any, dtype[bool_]]]",
        # expected "ndarray[Any, dtype[bool_]]")
        return mask  # type: ignore[return-value]

    def _arith_method(self, other, op):
        op_name = op.__name__
        omask = None

        if (
            not hasattr(other, "dtype")
            and is_list_like(other)
            and len(other) == len(self)
        ):
            # Try inferring masked dtype instead of casting to object
            other = pd_array(other)
            other = extract_array(other, extract_numpy=True)

        if isinstance(other, BaseMaskedArray):
            other, omask = other._data, other._mask

        elif is_list_like(other):
            if not isinstance(other, ExtensionArray):
                other = np.asarray(other)
            if other.ndim > 1:
                raise NotImplementedError("can only perform ops with 1-d structures")

        # We wrap the non-masked arithmetic logic used for numpy dtypes
        #  in Series/Index arithmetic ops.
        other = ops.maybe_prepare_scalar_for_op(other, (len(self),))
        pd_op = ops.get_array_op(op)
        other = ensure_wrapped_if_datetimelike(other)

        if op_name in {"pow", "rpow"} and isinstance(other, np.bool_):
            # Avoid DeprecationWarning: In future, it will be an error
            #  for 'np.bool_' scalars to be interpreted as an index
            #  e.g. test_array_scalar_like_equivalence
            other = bool(other)

        mask = self._propagate_mask(omask, other)

        if other is libmissing.NA:
            result = np.ones_like(self._data)
            if self.dtype.kind == "b":
                if op_name in {
                    "floordiv",
                    "rfloordiv",
                    "pow",
                    "rpow",
                    "truediv",
                    "rtruediv",
                }:
                    # GH#41165 Try to match non-masked Series behavior
                    #  This is still imperfect GH#46043
                    raise NotImplementedError(
                        f"operator '{op_name}' not implemented for bool dtypes"
                    )
                if op_name in {"mod", "rmod"}:
                    dtype = "int8"
                else:
                    dtype = "bool"
                result = result.astype(dtype)
            elif "truediv" in op_name and self.dtype.kind != "f":
                # The actual data here doesn't matter since the mask
                #  will be all-True, but since this is division, we want
                #  to end up with floating dtype.
                result = result.astype(np.float64)
        else:
            # Make sure we do this before the "pow" mask checks
            #  to get an expected exception message on shape mismatch.
            if self.dtype.kind in "iu" and op_name in ["floordiv", "mod"]:
                # TODO(GH#30188) ATM we don't match the behavior of non-masked
                #  types with respect to floordiv-by-zero
                pd_op = op

            with np.errstate(all="ignore"):
                result = pd_op(self._data, other)

        if op_name == "pow":
            # 1 ** x is 1.
            mask = np.where((self._data == 1) & ~self._mask, False, mask)
            # x ** 0 is 1.
            if omask is not None:
                mask = np.where((other == 0) & ~omask, False, mask)
            elif other is not libmissing.NA:
                mask = np.where(other == 0, False, mask)

        elif op_name == "rpow":
            # 1 ** x is 1.
            if omask is not None:
                mask = np.where((other == 1) & ~omask, False, mask)
            elif other is not libmissing.NA:
                mask = np.where(other == 1, False, mask)
            # x ** 0 is 1.
            mask = np.where((self._data == 0) & ~self._mask, False, mask)

        return self._maybe_mask_result(result, mask)

    _logical_method = _arith_method

    def _cmp_method(self, other, op) -> BooleanArray:
        from pandas.core.arrays import BooleanArray

        mask = None

        if isinstance(other, BaseMaskedArray):
            other, mask = other._data, other._mask

        elif is_list_like(other):
            other = np.asarray(other)
            if other.ndim > 1:
                raise NotImplementedError("can only perform ops with 1-d structures")
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")

        if other is libmissing.NA:
            # numpy does not handle pd.NA well as "other" scalar (it returns
            # a scalar False instead of an array)
            # This may be fixed by NA.__array_ufunc__. Revisit this check
            # once that's implemented.
            result = np.zeros(self._data.shape, dtype="bool")
            mask = np.ones(self._data.shape, dtype="bool")
        else:
            with warnings.catch_warnings():
                # numpy may show a FutureWarning or DeprecationWarning:
                #     elementwise comparison failed; returning scalar instead,
                #     but in the future will perform elementwise comparison
                # before returning NotImplemented. We fall back to the correct
                # behavior today, so that should be fine to ignore.
                warnings.filterwarnings("ignore", "elementwise", FutureWarning)
                warnings.filterwarnings("ignore", "elementwise", DeprecationWarning)
                method = getattr(self._data, f"__{op.__name__}__")
                result = method(other)

                if result is NotImplemented:
                    result = invalid_comparison(self._data, other, op)

        mask = self._propagate_mask(mask, other)
        return BooleanArray(result, mask, copy=False)

    def _maybe_mask_result(
        self, result: np.ndarray | tuple[np.ndarray, np.ndarray], mask: np.ndarray
    ):
        """
        Parameters
        ----------
        result : array-like or tuple[array-like]
        mask : array-like bool
        """
        if isinstance(result, tuple):
            # i.e. divmod
            div, mod = result
            return (
                self._maybe_mask_result(div, mask),
                self._maybe_mask_result(mod, mask),
            )

        if result.dtype.kind == "f":
            from pandas.core.arrays import FloatingArray

            return FloatingArray(result, mask, copy=False)

        elif result.dtype.kind == "b":
            from pandas.core.arrays import BooleanArray

            return BooleanArray(result, mask, copy=False)

        elif lib.is_np_dtype(result.dtype, "m") and is_supported_dtype(result.dtype):
            # e.g. test_numeric_arr_mul_tdscalar_numexpr_path
            from pandas.core.arrays import TimedeltaArray

            result[mask] = result.dtype.type("NaT")

            if not isinstance(result, TimedeltaArray):
                return TimedeltaArray._simple_new(result, dtype=result.dtype)

            return result

        elif result.dtype.kind in "iu":
            from pandas.core.arrays import IntegerArray

            return IntegerArray(result, mask, copy=False)

        else:
            result[mask] = np.nan
            return result

    def isna(self) -> np.ndarray:
        return self._mask.copy()

    @property
    def _na_value(self):
        return self.dtype.na_value

    @property
    def nbytes(self) -> int:
        return self._data.nbytes + self._mask.nbytes

    @classmethod
    def _concat_same_type(
        cls,
        to_concat: Sequence[Self],
        axis: AxisInt = 0,
    ) -> Self:
        data = np.concatenate([x._data for x in to_concat], axis=axis)
        mask = np.concatenate([x._mask for x in to_concat], axis=axis)
        return cls(data, mask)

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        hashed_array = hash_array(
            self._data, encoding=encoding, hash_key=hash_key, categorize=categorize
        )
        hashed_array[self.isna()] = hash(self.dtype.na_value)
        return hashed_array

    def take(
        self,
        indexer,
        *,
        allow_fill: bool = False,
        fill_value: Scalar | None = None,
        axis: AxisInt = 0,
    ) -> Self:
        # we always fill with 1 internally
        # to avoid upcasting
        data_fill_value = self._internal_fill_value if isna(fill_value) else fill_value
        result = take(
            self._data,
            indexer,
            fill_value=data_fill_value,
            allow_fill=allow_fill,
            axis=axis,
        )

        mask = take(
            self._mask, indexer, fill_value=True, allow_fill=allow_fill, axis=axis
        )

        # if we are filling
        # we only fill where the indexer is null
        # not existing missing values
        # TODO(jreback) what if we have a non-na float as a fill value?
        if allow_fill and notna(fill_value):
            fill_mask = np.asarray(indexer) == -1
            result[fill_mask] = fill_value
            mask = mask ^ fill_mask

        return self._simple_new(result, mask)

    # error: Return type "BooleanArray" of "isin" incompatible with return type
    # "ndarray" in supertype "ExtensionArray"
    def isin(self, values: ArrayLike) -> BooleanArray:  # type: ignore[override]
        from pandas.core.arrays import BooleanArray

        # algorithms.isin will eventually convert values to an ndarray, so no extra
        # cost to doing it here first
        values_arr = np.asarray(values)
        result = isin(self._data, values_arr)

        if self._hasna:
            values_have_NA = values_arr.dtype == object and any(
                val is self.dtype.na_value for val in values_arr
            )

            # For now, NA does not propagate so set result according to presence of NA,
            # see https://github.com/pandas-dev/pandas/pull/38379 for some discussion
            result[self._mask] = values_have_NA

        mask = np.zeros(self._data.shape, dtype=bool)
        return BooleanArray(result, mask, copy=False)

    def copy(self) -> Self:
        data = self._data.copy()
        mask = self._mask.copy()
        return self._simple_new(data, mask)

    @doc(ExtensionArray.duplicated)
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        values = self._data
        mask = self._mask
        return algos.duplicated(values, keep=keep, mask=mask)

    def unique(self) -> Self:
        """
        Compute the BaseMaskedArray of unique values.

        Returns
        -------
        uniques : BaseMaskedArray
        """
        uniques, mask = algos.unique_with_mask(self._data, self._mask)
        return self._simple_new(uniques, mask)

    @doc(ExtensionArray.searchsorted)
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        if self._hasna:
            raise ValueError(
                "searchsorted requires array to be sorted, which is impossible "
                "with NAs present."
            )
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        # Base class searchsorted would cast to object, which is *much* slower.
        return self._data.searchsorted(value, side=side, sorter=sorter)

    @doc(ExtensionArray.factorize)
    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, ExtensionArray]:
        arr = self._data
        mask = self._mask

        # Use a sentinel for na; recode and add NA to uniques if necessary below
        codes, uniques = factorize_array(arr, use_na_sentinel=True, mask=mask)

        # check that factorize_array correctly preserves dtype.
        assert uniques.dtype == self.dtype.numpy_dtype, (uniques.dtype, self.dtype)

        has_na = mask.any()
        if use_na_sentinel or not has_na:
            size = len(uniques)
        else:
            # Make room for an NA value
            size = len(uniques) + 1
        uniques_mask = np.zeros(size, dtype=bool)
        if not use_na_sentinel and has_na:
            na_index = mask.argmax()
            # Insert na with the proper code
            if na_index == 0:
                na_code = np.intp(0)
            else:
                na_code = codes[:na_index].max() + 1
            codes[codes >= na_code] += 1
            codes[codes == -1] = na_code
            # dummy value for uniques; not used since uniques_mask will be True
            uniques = np.insert(uniques, na_code, 0)
            uniques_mask[na_code] = True
        uniques_ea = self._simple_new(uniques, uniques_mask)

        return codes, uniques_ea

    @doc(ExtensionArray._values_for_argsort)
    def _values_for_argsort(self) -> np.ndarray:
        return self._data

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Returns a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        from pandas import (
            Index,
            Series,
        )
        from pandas.arrays import IntegerArray

        keys, value_counts, na_counter = algos.value_counts_arraylike(
            self._data, dropna=dropna, mask=self._mask
        )
        mask_index = np.zeros((len(value_counts),), dtype=np.bool_)
        mask = mask_index.copy()

        if na_counter > 0:
            mask_index[-1] = True

        arr = IntegerArray(value_counts, mask)
        index = Index(
            self.dtype.construct_array_type()(
                keys, mask_index  # type: ignore[arg-type]
            )
        )
        return Series(arr, index=index, name="count", copy=False)

    def _mode(self, dropna: bool = True) -> Self:
        if dropna:
            result = mode(self._data, dropna=dropna, mask=self._mask)
            res_mask = np.zeros(result.shape, dtype=np.bool_)
        else:
            result, res_mask = mode(self._data, dropna=dropna, mask=self._mask)
        result = type(self)(result, res_mask)  # type: ignore[arg-type]
        return result[result.argsort()]

    @doc(ExtensionArray.equals)
    def equals(self, other) -> bool:
        if type(self) != type(other):
            return False
        if other.dtype != self.dtype:
            return False

        # GH#44382 if e.g. self[1] is np.nan and other[1] is pd.NA, we are NOT
        #  equal.
        if not np.array_equal(self._mask, other._mask):
            return False

        left = self._data[~self._mask]
        right = other._data[~other._mask]
        return array_equivalent(left, right, strict_nan=True, dtype_equal=True)

    def _quantile(
        self, qs: npt.NDArray[np.float64], interpolation: str
    ) -> BaseMaskedArray:
        """
        Dispatch to quantile_with_mask, needed because we do not have
        _from_factorized.

        Notes
        -----
        We assume that all impacted cases are 1D-only.
        """
        res = quantile_with_mask(
            self._data,
            mask=self._mask,
            # TODO(GH#40932): na_value_for_dtype(self.dtype.numpy_dtype)
            #  instead of np.nan
            fill_value=np.nan,
            qs=qs,
            interpolation=interpolation,
        )

        if self._hasna:
            # Our result mask is all-False unless we are all-NA, in which
            #  case it is all-True.
            if self.ndim == 2:
                # I think this should be out_mask=self.isna().all(axis=1)
                #  but am holding off until we have tests
                raise NotImplementedError
            if self.isna().all():
                out_mask = np.ones(res.shape, dtype=bool)

                if is_integer_dtype(self.dtype):
                    # We try to maintain int dtype if possible for not all-na case
                    # as well
                    res = np.zeros(res.shape, dtype=self.dtype.numpy_dtype)
            else:
                out_mask = np.zeros(res.shape, dtype=bool)
        else:
            out_mask = np.zeros(res.shape, dtype=bool)
        return self._maybe_mask_result(res, mask=out_mask)

    # ------------------------------------------------------------------
    # Reductions

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        if name in {"any", "all", "min", "max", "sum", "prod", "mean", "var", "std"}:
            result = getattr(self, name)(skipna=skipna, **kwargs)
        else:
            # median, skew, kurt, sem
            data = self._data
            mask = self._mask
            op = getattr(nanops, f"nan{name}")
            axis = kwargs.pop("axis", None)
            result = op(data, axis=axis, skipna=skipna, mask=mask, **kwargs)

        if keepdims:
            if isna(result):
                return self._wrap_na_result(name=name, axis=0, mask_size=(1,))
            else:
                result = result.reshape(1)
                mask = np.zeros(1, dtype=bool)
                return self._maybe_mask_result(result, mask)

        if isna(result):
            return libmissing.NA
        else:
            return result

    def _wrap_reduction_result(self, name: str, result, *, skipna, axis):
        if isinstance(result, np.ndarray):
            if skipna:
                # we only retain mask for all-NA rows/columns
                mask = self._mask.all(axis=axis)
            else:
                mask = self._mask.any(axis=axis)

            return self._maybe_mask_result(result, mask)
        return result

    def _wrap_na_result(self, *, name, axis, mask_size):
        mask = np.ones(mask_size, dtype=bool)

        float_dtyp = "float32" if self.dtype == "Float32" else "float64"
        if name in ["mean", "median", "var", "std", "skew", "kurt"]:
            np_dtype = float_dtyp
        elif name in ["min", "max"] or self.dtype.itemsize == 8:
            np_dtype = self.dtype.numpy_dtype.name
        else:
            is_windows_or_32bit = is_platform_windows() or not IS64
            int_dtyp = "int32" if is_windows_or_32bit else "int64"
            uint_dtyp = "uint32" if is_windows_or_32bit else "uint64"
            np_dtype = {"b": int_dtyp, "i": int_dtyp, "u": uint_dtyp, "f": float_dtyp}[
                self.dtype.kind
            ]

        value = np.array([1], dtype=np_dtype)
        return self._maybe_mask_result(value, mask=mask)

    def _wrap_min_count_reduction_result(
        self, name: str, result, *, skipna, min_count, axis
    ):
        if min_count == 0 and isinstance(result, np.ndarray):
            return self._maybe_mask_result(result, np.zeros(result.shape, dtype=bool))
        return self._wrap_reduction_result(name, result, skipna=skipna, axis=axis)

    def sum(
        self,
        *,
        skipna: bool = True,
        min_count: int = 0,
        axis: AxisInt | None = 0,
        **kwargs,
    ):
        nv.validate_sum((), kwargs)

        result = masked_reductions.sum(
            self._data,
            self._mask,
            skipna=skipna,
            min_count=min_count,
            axis=axis,
        )
        return self._wrap_min_count_reduction_result(
            "sum", result, skipna=skipna, min_count=min_count, axis=axis
        )

    def prod(
        self,
        *,
        skipna: bool = True,
        min_count: int = 0,
        axis: AxisInt | None = 0,
        **kwargs,
    ):
        nv.validate_prod((), kwargs)

        result = masked_reductions.prod(
            self._data,
            self._mask,
            skipna=skipna,
            min_count=min_count,
            axis=axis,
        )
        return self._wrap_min_count_reduction_result(
            "prod", result, skipna=skipna, min_count=min_count, axis=axis
        )

    def mean(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
        nv.validate_mean((), kwargs)
        result = masked_reductions.mean(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
        )
        return self._wrap_reduction_result("mean", result, skipna=skipna, axis=axis)

    def var(
        self, *, skipna: bool = True, axis: AxisInt | None = 0, ddof: int = 1, **kwargs
    ):
        nv.validate_stat_ddof_func((), kwargs, fname="var")
        result = masked_reductions.var(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
            ddof=ddof,
        )
        return self._wrap_reduction_result("var", result, skipna=skipna, axis=axis)

    def std(
        self, *, skipna: bool = True, axis: AxisInt | None = 0, ddof: int = 1, **kwargs
    ):
        nv.validate_stat_ddof_func((), kwargs, fname="std")
        result = masked_reductions.std(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
            ddof=ddof,
        )
        return self._wrap_reduction_result("std", result, skipna=skipna, axis=axis)

    def min(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
        nv.validate_min((), kwargs)
        result = masked_reductions.min(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
        )
        return self._wrap_reduction_result("min", result, skipna=skipna, axis=axis)

    def max(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
        nv.validate_max((), kwargs)
        result = masked_reductions.max(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
        )
        return self._wrap_reduction_result("max", result, skipna=skipna, axis=axis)

    def map(self, mapper, na_action=None):
        return map_array(self.to_numpy(), mapper, na_action=na_action)

    def any(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
        """
        Return whether any element is truthy.

        Returns False unless there is at least one element that is truthy.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        .. versionchanged:: 1.4.0

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be False, as for an empty array.
            If `skipna` is False, the result will still be True if there is
            at least one element that is truthy, otherwise NA will be returned
            if there are NA's present.
        axis : int, optional, default 0
        **kwargs : any, default None
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        numpy.any : Numpy version of this method.
        BaseMaskedArray.all : Return whether all elements are truthy.

        Examples
        --------
        The result indicates whether any element is truthy (and by default
        skips NAs):

        >>> pd.array([True, False, True]).any()
        True
        >>> pd.array([True, False, pd.NA]).any()
        True
        >>> pd.array([False, False, pd.NA]).any()
        False
        >>> pd.array([], dtype="boolean").any()
        False
        >>> pd.array([pd.NA], dtype="boolean").any()
        False
        >>> pd.array([pd.NA], dtype="Float64").any()
        False

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, False, pd.NA]).any(skipna=False)
        True
        >>> pd.array([1, 0, pd.NA]).any(skipna=False)
        True
        >>> pd.array([False, False, pd.NA]).any(skipna=False)
        <NA>
        >>> pd.array([0, 0, pd.NA]).any(skipna=False)
        <NA>
        """
        nv.validate_any((), kwargs)

        values = self._data.copy()
        # error: Argument 3 to "putmask" has incompatible type "object";
        # expected "Union[_SupportsArray[dtype[Any]],
        # _NestedSequence[_SupportsArray[dtype[Any]]],
        # bool, int, float, complex, str, bytes,
        # _NestedSequence[Union[bool, int, float, complex, str, bytes]]]"
        np.putmask(values, self._mask, self._falsey_value)  # type: ignore[arg-type]
        result = values.any()
        if skipna:
            return result
        else:
            if result or len(self) == 0 or not self._mask.any():
                return result
            else:
                return self.dtype.na_value

    def all(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
        """
        Return whether all elements are truthy.

        Returns True unless there is at least one element that is falsey.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        .. versionchanged:: 1.4.0

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be True, as for an empty array.
            If `skipna` is False, the result will still be False if there is
            at least one element that is falsey, otherwise NA will be returned
            if there are NA's present.
        axis : int, optional, default 0
        **kwargs : any, default None
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        numpy.all : Numpy version of this method.
        BooleanArray.any : Return whether any element is truthy.

        Examples
        --------
        The result indicates whether all elements are truthy (and by default
        skips NAs):

        >>> pd.array([True, True, pd.NA]).all()
        True
        >>> pd.array([1, 1, pd.NA]).all()
        True
        >>> pd.array([True, False, pd.NA]).all()
        False
        >>> pd.array([], dtype="boolean").all()
        True
        >>> pd.array([pd.NA], dtype="boolean").all()
        True
        >>> pd.array([pd.NA], dtype="Float64").all()
        True

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, True, pd.NA]).all(skipna=False)
        <NA>
        >>> pd.array([1, 1, pd.NA]).all(skipna=False)
        <NA>
        >>> pd.array([True, False, pd.NA]).all(skipna=False)
        False
        >>> pd.array([1, 0, pd.NA]).all(skipna=False)
        False
        """
        nv.validate_all((), kwargs)

        values = self._data.copy()
        # error: Argument 3 to "putmask" has incompatible type "object";
        # expected "Union[_SupportsArray[dtype[Any]],
        # _NestedSequence[_SupportsArray[dtype[Any]]],
        # bool, int, float, complex, str, bytes,
        # _NestedSequence[Union[bool, int, float, complex, str, bytes]]]"
        np.putmask(values, self._mask, self._truthy_value)  # type: ignore[arg-type]
        result = values.all(axis=axis)

        if skipna:
            return result
        else:
            if not result or len(self) == 0 or not self._mask.any():
                return result
            else:
                return self.dtype.na_value

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> FloatingArray:
        """
        See NDFrame.interpolate.__doc__.
        """
        # NB: we return type(self) even if copy=False
        if self.dtype.kind == "f":
            if copy:
                data = self._data.copy()
                mask = self._mask.copy()
            else:
                data = self._data
                mask = self._mask
        elif self.dtype.kind in "iu":
            copy = True
            data = self._data.astype("f8")
            mask = self._mask.copy()
        else:
            raise NotImplementedError(
                f"interpolate is not implemented for dtype={self.dtype}"
            )

        missing.interpolate_2d_inplace(
            data,
            method=method,
            axis=0,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            mask=mask,
            **kwargs,
        )
        if not copy:
            return self  # type: ignore[return-value]
        if self.dtype.kind == "f":
            return type(self)._simple_new(data, mask)  # type: ignore[return-value]
        else:
            from pandas.core.arrays import FloatingArray

            return FloatingArray._simple_new(data, mask)

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> BaseMaskedArray:
        data = self._data
        mask = self._mask

        op = getattr(masked_accumulations, name)
        data, mask = op(data, mask, skipna=skipna, **kwargs)

        return self._simple_new(data, mask)

    # ------------------------------------------------------------------
    # GroupBy Methods

    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
    ):
        from pandas.core.groupby.ops import WrappedCythonOp

        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        # libgroupby functions are responsible for NOT altering mask
        mask = self._mask
        if op.kind != "aggregate":
            result_mask = mask.copy()
        else:
            result_mask = np.zeros(ngroups, dtype=bool)

        if how == "rank" and kwargs.get("na_option") in ["top", "bottom"]:
            result_mask[:] = False

        res_values = op._cython_op_ndim_compat(
            self._data,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=mask,
            result_mask=result_mask,
            **kwargs,
        )

        if op.how == "ohlc":
            arity = op._cython_arity.get(op.how, 1)
            result_mask = np.tile(result_mask, (arity, 1)).T

        if op.how in ["idxmin", "idxmax"]:
            # Result values are indexes to take, keep as ndarray
            return res_values
        else:
            # res_values should already have the correct dtype, we just need to
            #  wrap in a MaskedArray
            return self._maybe_mask_result(res_values, result_mask)


def transpose_homogeneous_masked_arrays(
    masked_arrays: Sequence[BaseMaskedArray],
) -> list[BaseMaskedArray]:
    """Transpose masked arrays in a list, but faster.

    Input should be a list of 1-dim masked arrays of equal length and all have the
    same dtype. The caller is responsible for ensuring validity of input data.
    """
    masked_arrays = list(masked_arrays)
    dtype = masked_arrays[0].dtype

    values = [arr._data.reshape(1, -1) for arr in masked_arrays]
    transposed_values = np.concatenate(
        values,
        axis=0,
        out=np.empty(
            (len(masked_arrays), len(masked_arrays[0])),
            order="F",
            dtype=dtype.numpy_dtype,
        ),
    )

    masks = [arr._mask.reshape(1, -1) for arr in masked_arrays]
    transposed_masks = np.concatenate(
        masks, axis=0, out=np.empty_like(transposed_values, dtype=bool)
    )

    arr_type = dtype.construct_array_type()
    transposed_arrays: list[BaseMaskedArray] = []
    for i in range(transposed_values.shape[1]):
        transposed_arr = arr_type(transposed_values[:, i], mask=transposed_masks[:, i])
        transposed_arrays.append(transposed_arr)

    return transposed_arrays

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\algebras\quaternion.py ===
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log as ln)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.trigsimp import trigsimp
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.core.sympify import sympify, _sympify
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.utilities.misc import as_int

from mpmath.libmp.libmpf import prec_to_dps


def _check_norm(elements, norm):
    """validate if input norm is consistent"""
    if norm is not None and norm.is_number:
        if norm.is_positive is False:
            raise ValueError("Input norm must be positive.")

        numerical = all(i.is_number and i.is_real is True for i in elements)
        if numerical and is_eq(norm**2, sum(i**2 for i in elements)) is False:
            raise ValueError("Incompatible value for norm.")


def _is_extrinsic(seq):
    """validate seq and return True if seq is lowercase and False if uppercase"""
    if type(seq) != str:
        raise ValueError('Expected seq to be a string.')
    if len(seq) != 3:
        raise ValueError("Expected 3 axes, got `{}`.".format(seq))

    intrinsic = seq.isupper()
    extrinsic = seq.islower()
    if not (intrinsic or extrinsic):
        raise ValueError("seq must either be fully uppercase (for extrinsic "
                         "rotations), or fully lowercase, for intrinsic "
                         "rotations).")

    i, j, k = seq.lower()
    if (i == j) or (j == k):
        raise ValueError("Consecutive axes must be different")

    bad = set(seq) - set('xyzXYZ')
    if bad:
        raise ValueError("Expected axes from `seq` to be from "
                         "['x', 'y', 'z'] or ['X', 'Y', 'Z'], "
                         "got {}".format(''.join(bad)))

    return extrinsic


class Quaternion(Expr):
    """Provides basic quaternion operations.
    Quaternion objects can be instantiated as ``Quaternion(a, b, c, d)``
    as in $q = a + bi + cj + dk$.

    Parameters
    ==========

    norm : None or number
        Pre-defined quaternion norm. If a value is given, Quaternion.norm
        returns this pre-defined value instead of calculating the norm

    Examples
    ========

    >>> from sympy import Quaternion
    >>> q = Quaternion(1, 2, 3, 4)
    >>> q
    1 + 2*i + 3*j + 4*k

    Quaternions over complex fields can be defined as:

    >>> from sympy import Quaternion
    >>> from sympy import symbols, I
    >>> x = symbols('x')
    >>> q1 = Quaternion(x, x**3, x, x**2, real_field = False)
    >>> q2 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
    >>> q1
    x + x**3*i + x*j + x**2*k
    >>> q2
    (3 + 4*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

    Defining symbolic unit quaternions:

    >>> from sympy import Quaternion
    >>> from sympy.abc import w, x, y, z
    >>> q = Quaternion(w, x, y, z, norm=1)
    >>> q
    w + x*i + y*j + z*k
    >>> q.norm()
    1

    References
    ==========

    .. [1] https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/
    .. [2] https://en.wikipedia.org/wiki/Quaternion

    """
    _op_priority = 11.0

    is_commutative = False

    def __new__(cls, a=0, b=0, c=0, d=0, real_field=True, norm=None):
        a, b, c, d = map(sympify, (a, b, c, d))

        if any(i.is_commutative is False for i in [a, b, c, d]):
            raise ValueError("arguments have to be commutative")
        obj = super().__new__(cls, a, b, c, d)
        obj._real_field = real_field
        obj.set_norm(norm)
        return obj

    def set_norm(self, norm):
        """Sets norm of an already instantiated quaternion.

        Parameters
        ==========

        norm : None or number
            Pre-defined quaternion norm. If a value is given, Quaternion.norm
            returns this pre-defined value instead of calculating the norm

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion(a, b, c, d)
        >>> q.norm()
        sqrt(a**2 + b**2 + c**2 + d**2)

        Setting the norm:

        >>> q.set_norm(1)
        >>> q.norm()
        1

        Removing set norm:

        >>> q.set_norm(None)
        >>> q.norm()
        sqrt(a**2 + b**2 + c**2 + d**2)

        """
        norm = sympify(norm)
        _check_norm(self.args, norm)
        self._norm = norm

    @property
    def a(self):
        return self.args[0]

    @property
    def b(self):
        return self.args[1]

    @property
    def c(self):
        return self.args[2]

    @property
    def d(self):
        return self.args[3]

    @property
    def real_field(self):
        return self._real_field

    @property
    def product_matrix_left(self):
        r"""Returns 4 x 4 Matrix equivalent to a Hamilton product from the
        left. This can be useful when treating quaternion elements as column
        vectors. Given a quaternion $q = a + bi + cj + dk$ where a, b, c and d
        are real numbers, the product matrix from the left is:

        .. math::

            M  =  \begin{bmatrix} a  &-b  &-c  &-d \\
                                  b  & a  &-d  & c \\
                                  c  & d  & a  &-b \\
                                  d  &-c  & b  & a \end{bmatrix}

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q1 = Quaternion(1, 0, 0, 1)
        >>> q2 = Quaternion(a, b, c, d)
        >>> q1.product_matrix_left
        Matrix([
        [1, 0,  0, -1],
        [0, 1, -1,  0],
        [0, 1,  1,  0],
        [1, 0,  0,  1]])

        >>> q1.product_matrix_left * q2.to_Matrix()
        Matrix([
        [a - d],
        [b - c],
        [b + c],
        [a + d]])

        This is equivalent to:

        >>> (q1 * q2).to_Matrix()
        Matrix([
        [a - d],
        [b - c],
        [b + c],
        [a + d]])
        """
        return Matrix([
                [self.a, -self.b, -self.c, -self.d],
                [self.b, self.a, -self.d, self.c],
                [self.c, self.d, self.a, -self.b],
                [self.d, -self.c, self.b, self.a]])

    @property
    def product_matrix_right(self):
        r"""Returns 4 x 4 Matrix equivalent to a Hamilton product from the
        right. This can be useful when treating quaternion elements as column
        vectors. Given a quaternion $q = a + bi + cj + dk$ where a, b, c and d
        are real numbers, the product matrix from the left is:

        .. math::

            M  =  \begin{bmatrix} a  &-b  &-c  &-d \\
                                  b  & a  & d  &-c \\
                                  c  &-d  & a  & b \\
                                  d  & c  &-b  & a \end{bmatrix}


        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q1 = Quaternion(a, b, c, d)
        >>> q2 = Quaternion(1, 0, 0, 1)
        >>> q2.product_matrix_right
        Matrix([
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, -1, 1, 0],
        [1, 0, 0, 1]])

        Note the switched arguments: the matrix represents the quaternion on
        the right, but is still considered as a matrix multiplication from the
        left.

        >>> q2.product_matrix_right * q1.to_Matrix()
        Matrix([
        [ a - d],
        [ b + c],
        [-b + c],
        [ a + d]])

        This is equivalent to:

        >>> (q1 * q2).to_Matrix()
        Matrix([
        [ a - d],
        [ b + c],
        [-b + c],
        [ a + d]])
        """
        return Matrix([
                [self.a, -self.b, -self.c, -self.d],
                [self.b, self.a, self.d, -self.c],
                [self.c, -self.d, self.a, self.b],
                [self.d, self.c, -self.b, self.a]])

    def to_Matrix(self, vector_only=False):
        """Returns elements of quaternion as a column vector.
        By default, a ``Matrix`` of length 4 is returned, with the real part as the
        first element.
        If ``vector_only`` is ``True``, returns only imaginary part as a Matrix of
        length 3.

        Parameters
        ==========

        vector_only : bool
            If True, only imaginary part is returned.
            Default value: False

        Returns
        =======

        Matrix
            A column vector constructed by the elements of the quaternion.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion(a, b, c, d)
        >>> q
        a + b*i + c*j + d*k

        >>> q.to_Matrix()
        Matrix([
        [a],
        [b],
        [c],
        [d]])


        >>> q.to_Matrix(vector_only=True)
        Matrix([
        [b],
        [c],
        [d]])

        """
        if vector_only:
            return Matrix(self.args[1:])
        else:
            return Matrix(self.args)

    @classmethod
    def from_Matrix(cls, elements):
        """Returns quaternion from elements of a column vector`.
        If vector_only is True, returns only imaginary part as a Matrix of
        length 3.

        Parameters
        ==========

        elements : Matrix, list or tuple of length 3 or 4. If length is 3,
            assume real part is zero.
            Default value: False

        Returns
        =======

        Quaternion
            A quaternion created from the input elements.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q = Quaternion.from_Matrix([a, b, c, d])
        >>> q
        a + b*i + c*j + d*k

        >>> q = Quaternion.from_Matrix([b, c, d])
        >>> q
        0 + b*i + c*j + d*k

        """
        length = len(elements)
        if length != 3 and length != 4:
            raise ValueError("Input elements must have length 3 or 4, got {} "
                             "elements".format(length))

        if length == 3:
            return Quaternion(0, *elements)
        else:
            return Quaternion(*elements)

    @classmethod
    def from_euler(cls, angles, seq):
        """Returns quaternion equivalent to rotation represented by the Euler
        angles, in the sequence defined by ``seq``.

        Parameters
        ==========

        angles : list, tuple or Matrix of 3 numbers
            The Euler angles (in radians).
        seq : string of length 3
            Represents the sequence of rotations.
            For extrinsic rotations, seq must be all lowercase and its elements
            must be from the set ``{'x', 'y', 'z'}``
            For intrinsic rotations, seq must be all uppercase and its elements
            must be from the set ``{'X', 'Y', 'Z'}``

        Returns
        =======

        Quaternion
            The normalized rotation quaternion calculated from the Euler angles
            in the given sequence.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import pi
        >>> q = Quaternion.from_euler([pi/2, 0, 0], 'xyz')
        >>> q
        sqrt(2)/2 + sqrt(2)/2*i + 0*j + 0*k

        >>> q = Quaternion.from_euler([0, pi/2, pi] , 'zyz')
        >>> q
        0 + (-sqrt(2)/2)*i + 0*j + sqrt(2)/2*k

        >>> q = Quaternion.from_euler([0, pi/2, pi] , 'ZYZ')
        >>> q
        0 + sqrt(2)/2*i + 0*j + sqrt(2)/2*k

        """

        if len(angles) != 3:
            raise ValueError("3 angles must be given.")

        extrinsic = _is_extrinsic(seq)
        i, j, k = seq.lower()

        # get elementary basis vectors
        ei = [1 if n == i else 0 for n in 'xyz']
        ej = [1 if n == j else 0 for n in 'xyz']
        ek = [1 if n == k else 0 for n in 'xyz']

        # calculate distinct quaternions
        qi = cls.from_axis_angle(ei, angles[0])
        qj = cls.from_axis_angle(ej, angles[1])
        qk = cls.from_axis_angle(ek, angles[2])

        if extrinsic:
            return trigsimp(qk * qj * qi)
        else:
            return trigsimp(qi * qj * qk)

    def to_euler(self, seq, angle_addition=True, avoid_square_root=False):
        r"""Returns Euler angles representing same rotation as the quaternion,
        in the sequence given by ``seq``. This implements the method described
        in [1]_.

        For degenerate cases (gymbal lock cases), the third angle is
        set to zero.

        Parameters
        ==========

        seq : string of length 3
            Represents the sequence of rotations.
            For extrinsic rotations, seq must be all lowercase and its elements
            must be from the set ``{'x', 'y', 'z'}``
            For intrinsic rotations, seq must be all uppercase and its elements
            must be from the set ``{'X', 'Y', 'Z'}``

        angle_addition : bool
            When True, first and third angles are given as an addition and
            subtraction of two simpler ``atan2`` expressions. When False, the
            first and third angles are each given by a single more complicated
            ``atan2`` expression. This equivalent expression is given by:

            .. math::

                \operatorname{atan_2} (b,a) \pm \operatorname{atan_2} (d,c) =
                \operatorname{atan_2} (bc\pm ad, ac\mp bd)

            Default value: True

        avoid_square_root : bool
            When True, the second angle is calculated with an expression based
            on ``acos``, which is slightly more complicated but avoids a square
            root. When False, second angle is calculated with ``atan2``, which
            is simpler and can be better for numerical reasons (some
            numerical implementations of ``acos`` have problems near zero).
            Default value: False


        Returns
        =======

        Tuple
            The Euler angles calculated from the quaternion

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> euler = Quaternion(a, b, c, d).to_euler('zyz')
        >>> euler
        (-atan2(-b, c) + atan2(d, a),
         2*atan2(sqrt(b**2 + c**2), sqrt(a**2 + d**2)),
         atan2(-b, c) + atan2(d, a))


        References
        ==========

        .. [1] https://doi.org/10.1371/journal.pone.0276302

        """
        if self.is_zero_quaternion():
            raise ValueError('Cannot convert a quaternion with norm 0.')

        angles = [0, 0, 0]

        extrinsic = _is_extrinsic(seq)
        i, j, k = seq.lower()

        # get index corresponding to elementary basis vectors
        i = 'xyz'.index(i) + 1
        j = 'xyz'.index(j) + 1
        k = 'xyz'.index(k) + 1

        if not extrinsic:
            i, k = k, i

        # check if sequence is symmetric
        symmetric = i == k
        if symmetric:
            k = 6 - i - j

        # parity of the permutation
        sign = (i - j) * (j - k) * (k - i) // 2

        # permutate elements
        elements = [self.a, self.b, self.c, self.d]
        a = elements[0]
        b = elements[i]
        c = elements[j]
        d = elements[k] * sign

        if not symmetric:
            a, b, c, d = a - c, b + d, c + a, d - b

        if avoid_square_root:
            if symmetric:
                n2 = self.norm()**2
                angles[1] = acos((a * a + b * b - c * c - d * d) / n2)
            else:
                n2 = 2 * self.norm()**2
                angles[1] = asin((c * c + d * d - a * a - b * b) / n2)
        else:
            angles[1] = 2 * atan2(sqrt(c * c + d * d), sqrt(a * a + b * b))
            if not symmetric:
                angles[1] -= S.Pi / 2

        # Check for singularities in numerical cases
        case = 0
        if is_eq(c, S.Zero) and is_eq(d, S.Zero):
            case = 1
        if is_eq(a, S.Zero) and is_eq(b, S.Zero):
            case = 2

        if case == 0:
            if angle_addition:
                angles[0] = atan2(b, a) + atan2(d, c)
                angles[2] = atan2(b, a) - atan2(d, c)
            else:
                angles[0] = atan2(b*c + a*d, a*c - b*d)
                angles[2] = atan2(b*c - a*d, a*c + b*d)

        else:  # any degenerate case
            angles[2 * (not extrinsic)] = S.Zero
            if case == 1:
                angles[2 * extrinsic] = 2 * atan2(b, a)
            else:
                angles[2 * extrinsic] = 2 * atan2(d, c)
                angles[2 * extrinsic] *= (-1 if extrinsic else 1)

        # for Tait-Bryan angles
        if not symmetric:
            angles[0] *= sign

        if extrinsic:
            return tuple(angles[::-1])
        else:
            return tuple(angles)

    @classmethod
    def from_axis_angle(cls, vector, angle):
        """Returns a rotation quaternion given the axis and the angle of rotation.

        Parameters
        ==========

        vector : tuple of three numbers
            The vector representation of the given axis.
        angle : number
            The angle by which axis is rotated (in radians).

        Returns
        =======

        Quaternion
            The normalized rotation quaternion calculated from the given axis and the angle of rotation.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import pi, sqrt
        >>> q = Quaternion.from_axis_angle((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3), 2*pi/3)
        >>> q
        1/2 + 1/2*i + 1/2*j + 1/2*k

        """
        (x, y, z) = vector
        norm = sqrt(x**2 + y**2 + z**2)
        (x, y, z) = (x / norm, y / norm, z / norm)
        s = sin(angle * S.Half)
        a = cos(angle * S.Half)
        b = x * s
        c = y * s
        d = z * s

        # note that this quaternion is already normalized by construction:
        # c^2 + (s*x)^2 + (s*y)^2 + (s*z)^2 = c^2 + s^2*(x^2 + y^2 + z^2) = c^2 + s^2 * 1 = c^2 + s^2 = 1
        # so, what we return is a normalized quaternion

        return cls(a, b, c, d)

    @classmethod
    def from_rotation_matrix(cls, M):
        """Returns the equivalent quaternion of a matrix. The quaternion will be normalized
        only if the matrix is special orthogonal (orthogonal and det(M) = 1).

        Parameters
        ==========

        M : Matrix
            Input matrix to be converted to equivalent quaternion. M must be special
            orthogonal (orthogonal and det(M) = 1) for the quaternion to be normalized.

        Returns
        =======

        Quaternion
            The quaternion equivalent to given matrix.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import Matrix, symbols, cos, sin, trigsimp
        >>> x = symbols('x')
        >>> M = Matrix([[cos(x), -sin(x), 0], [sin(x), cos(x), 0], [0, 0, 1]])
        >>> q = trigsimp(Quaternion.from_rotation_matrix(M))
        >>> q
        sqrt(2)*sqrt(cos(x) + 1)/2 + 0*i + 0*j + sqrt(2 - 2*cos(x))*sign(sin(x))/2*k

        """

        absQ = M.det()**Rational(1, 3)

        a = sqrt(absQ + M[0, 0] + M[1, 1] + M[2, 2]) / 2
        b = sqrt(absQ + M[0, 0] - M[1, 1] - M[2, 2]) / 2
        c = sqrt(absQ - M[0, 0] + M[1, 1] - M[2, 2]) / 2
        d = sqrt(absQ - M[0, 0] - M[1, 1] + M[2, 2]) / 2

        b = b * sign(M[2, 1] - M[1, 2])
        c = c * sign(M[0, 2] - M[2, 0])
        d = d * sign(M[1, 0] - M[0, 1])

        return Quaternion(a, b, c, d)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.add(other*-1)

    def __mul__(self, other):
        return self._generic_mul(self, _sympify(other))

    def __rmul__(self, other):
        return self._generic_mul(_sympify(other), self)

    def __pow__(self, p):
        return self.pow(p)

    def __neg__(self):
        return Quaternion(-self.a, -self.b, -self.c, -self.d)

    def __truediv__(self, other):
        return self * sympify(other)**-1

    def __rtruediv__(self, other):
        return sympify(other) * self**-1

    def _eval_Integral(self, *args):
        return self.integrate(*args)

    def diff(self, *symbols, **kwargs):
        kwargs.setdefault('evaluate', True)
        return self.func(*[a.diff(*symbols, **kwargs) for a  in self.args])

    def add(self, other):
        """Adds quaternions.

        Parameters
        ==========

        other : Quaternion
            The quaternion to add to current (self) quaternion.

        Returns
        =======

        Quaternion
            The resultant quaternion after adding self to other

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.add(q2)
        6 + 8*i + 10*j + 12*k
        >>> q1 + 5
        6 + 2*i + 3*j + 4*k
        >>> x = symbols('x', real = True)
        >>> q1.add(x)
        (x + 1) + 2*i + 3*j + 4*k

        Quaternions over complex fields :

        >>> from sympy import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.add(2 + 3*I)
        (5 + 7*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

        """
        q1 = self
        q2 = sympify(other)

        # If q2 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return Quaternion(re(q2) + q1.a, im(q2) + q1.b, q1.c, q1.d)
            elif q2.is_commutative:
                return Quaternion(q1.a + q2, q1.b, q1.c, q1.d)
            else:
                raise ValueError("Only commutative expressions can be added with a Quaternion.")

        return Quaternion(q1.a + q2.a, q1.b + q2.b, q1.c + q2.c, q1.d
                          + q2.d)

    def mul(self, other):
        """Multiplies quaternions.

        Parameters
        ==========

        other : Quaternion or symbol
            The quaternion to multiply to current (self) quaternion.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying self with other

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.mul(q2)
        (-60) + 12*i + 30*j + 24*k
        >>> q1.mul(2)
        2 + 4*i + 6*j + 8*k
        >>> x = symbols('x', real = True)
        >>> q1.mul(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.mul(2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        return self._generic_mul(self, _sympify(other))

    @staticmethod
    def _generic_mul(q1, q2):
        """Generic multiplication.

        Parameters
        ==========

        q1 : Quaternion or symbol
        q2 : Quaternion or symbol

        It is important to note that if neither q1 nor q2 is a Quaternion,
        this function simply returns q1 * q2.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying q1 and q2

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import Symbol, S
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> Quaternion._generic_mul(q1, q2)
        (-60) + 12*i + 30*j + 24*k
        >>> Quaternion._generic_mul(q1, S(2))
        2 + 4*i + 6*j + 8*k
        >>> x = Symbol('x', real = True)
        >>> Quaternion._generic_mul(q1, x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> Quaternion._generic_mul(q3, 2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        # None is a Quaternion:
        if not isinstance(q1, Quaternion) and not isinstance(q2, Quaternion):
            return q1 * q2

        # If q1 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q1, Quaternion):
            if q2.real_field and q1.is_complex:
                return Quaternion(re(q1), im(q1), 0, 0) * q2
            elif q1.is_commutative:
                return Quaternion(q1 * q2.a, q1 * q2.b, q1 * q2.c, q1 * q2.d)
            else:
                raise ValueError("Only commutative expressions can be multiplied with a Quaternion.")

        # If q2 is a number or a SymPy expression instead of a quaternion
        if not isinstance(q2, Quaternion):
            if q1.real_field and q2.is_complex:
                return q1 * Quaternion(re(q2), im(q2), 0, 0)
            elif q2.is_commutative:
                return Quaternion(q2 * q1.a, q2 * q1.b, q2 * q1.c, q2 * q1.d)
            else:
                raise ValueError("Only commutative expressions can be multiplied with a Quaternion.")

        # If any of the quaternions has a fixed norm, pre-compute norm
        if q1._norm is None and q2._norm is None:
            norm = None
        else:
            norm = q1.norm() * q2.norm()

        return Quaternion(-q1.b*q2.b - q1.c*q2.c - q1.d*q2.d + q1.a*q2.a,
                          q1.b*q2.a + q1.c*q2.d - q1.d*q2.c + q1.a*q2.b,
                          -q1.b*q2.d + q1.c*q2.a + q1.d*q2.b + q1.a*q2.c,
                          q1.b*q2.c - q1.c*q2.b + q1.d*q2.a + q1.a * q2.d,
                          norm=norm)

    def _eval_conjugate(self):
        """Returns the conjugate of the quaternion."""
        q = self
        return Quaternion(q.a, -q.b, -q.c, -q.d, norm=q._norm)

    def norm(self):
        """Returns the norm of the quaternion."""
        if self._norm is None:  # check if norm is pre-defined
            q = self
            # trigsimp is used to simplify sin(x)^2 + cos(x)^2 (these terms
            # arise when from_axis_angle is used).
            return sqrt(trigsimp(q.a**2 + q.b**2 + q.c**2 + q.d**2))

        return self._norm

    def normalize(self):
        """Returns the normalized form of the quaternion."""
        q = self
        return q * (1/q.norm())

    def inverse(self):
        """Returns the inverse of the quaternion."""
        q = self
        if not q.norm():
            raise ValueError("Cannot compute inverse for a quaternion with zero norm")
        return conjugate(q) * (1/q.norm()**2)

    def pow(self, p):
        """Finds the pth power of the quaternion.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            Returns the p-th power of the current quaternion.
            Returns the inverse if p = -1.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow(4)
        668 + (-224)*i + (-336)*j + (-448)*k

        """
        try:
            q, p = self, as_int(p)
        except ValueError:
            return NotImplemented

        if p < 0:
            q, p = q.inverse(), -p

        if p == 1:
            return q

        res = Quaternion(1, 0, 0, 0)
        while p > 0:
            if p & 1:
                res *= q
            q *= q
            p >>= 1

        return res

    def exp(self):
        """Returns the exponential of $q$, given by $e^q$.

        Returns
        =======

        Quaternion
            The exponential of the quaternion.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.exp()
        E*cos(sqrt(29))
        + 2*sqrt(29)*E*sin(sqrt(29))/29*i
        + 3*sqrt(29)*E*sin(sqrt(29))/29*j
        + 4*sqrt(29)*E*sin(sqrt(29))/29*k

        """
        # exp(q) = e^a(cos||v|| + v/||v||*sin||v||)
        q = self
        vector_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
        a = exp(q.a) * cos(vector_norm)
        b = exp(q.a) * sin(vector_norm) * q.b / vector_norm
        c = exp(q.a) * sin(vector_norm) * q.c / vector_norm
        d = exp(q.a) * sin(vector_norm) * q.d / vector_norm

        return Quaternion(a, b, c, d)

    def log(self):
        r"""Returns the logarithm of the quaternion, given by $\log q$.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.log()
        log(sqrt(30))
        + 2*sqrt(29)*acos(sqrt(30)/30)/29*i
        + 3*sqrt(29)*acos(sqrt(30)/30)/29*j
        + 4*sqrt(29)*acos(sqrt(30)/30)/29*k

        """
        # log(q) = log||q|| + v/||v||*arccos(a/||q||)
        q = self
        vector_norm = sqrt(q.b**2 + q.c**2 + q.d**2)
        q_norm = q.norm()
        a = ln(q_norm)
        b = q.b * acos(q.a / q_norm) / vector_norm
        c = q.c * acos(q.a / q_norm) / vector_norm
        d = q.d * acos(q.a / q_norm) / vector_norm

        return Quaternion(a, b, c, d)

    def _eval_subs(self, *args):
        elements = [i.subs(*args) for i in self.args]
        norm = self._norm
        if norm is not None:
            norm = norm.subs(*args)
        _check_norm(elements, norm)
        return Quaternion(*elements, norm=norm)

    def _eval_evalf(self, prec):
        """Returns the floating point approximations (decimal numbers) of the quaternion.

        Returns
        =======

        Quaternion
            Floating point approximations of quaternion(self)

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import sqrt
        >>> q = Quaternion(1/sqrt(1), 1/sqrt(2), 1/sqrt(3), 1/sqrt(4))
        >>> q.evalf()
        1.00000000000000
        + 0.707106781186547*i
        + 0.577350269189626*j
        + 0.500000000000000*k

        """
        nprec = prec_to_dps(prec)
        return Quaternion(*[arg.evalf(n=nprec) for arg in self.args])

    def pow_cos_sin(self, p):
        """Computes the pth power in the cos-sin form.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            The p-th power in the cos-sin form.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow_cos_sin(4)
        900*cos(4*acos(sqrt(30)/30))
        + 1800*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*i
        + 2700*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*j
        + 3600*sqrt(29)*sin(4*acos(sqrt(30)/30))/29*k

        """
        # q = ||q||*(cos(a) + u*sin(a))
        # q^p = ||q||^p * (cos(p*a) + u*sin(p*a))

        q = self
        (v, angle) = q.to_axis_angle()
        q2 = Quaternion.from_axis_angle(v, p * angle)
        return q2 * (q.norm()**p)

    def integrate(self, *args):
        """Computes integration of quaternion.

        Returns
        =======

        Quaternion
            Integration of the quaternion(self) with the given variable.

        Examples
        ========

        Indefinite Integral of quaternion :

        >>> from sympy import Quaternion
        >>> from sympy.abc import x
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.integrate(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Definite integral of quaternion :

        >>> from sympy import Quaternion
        >>> from sympy.abc import x
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.integrate((x, 1, 5))
        4 + 8*i + 12*j + 16*k

        """
        return Quaternion(integrate(self.a, *args), integrate(self.b, *args),
                          integrate(self.c, *args), integrate(self.d, *args))

    @staticmethod
    def rotate_point(pin, r):
        """Returns the coordinates of the point pin (a 3 tuple) after rotation.

        Parameters
        ==========

        pin : tuple
            A 3-element tuple of coordinates of a point which needs to be
            rotated.
        r : Quaternion or tuple
            Axis and angle of rotation.

            It's important to note that when r is a tuple, it must be of the form
            (axis, angle)

        Returns
        =======

        tuple
            The coordinates of the point after rotation.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), q))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> trigsimp(Quaternion.rotate_point((1, 1, 1), (axis, angle)))
        (sqrt(2)*cos(x + pi/4), sqrt(2)*sin(x + pi/4), 1)

        """
        if isinstance(r, tuple):
            # if r is of the form (vector, angle)
            q = Quaternion.from_axis_angle(r[0], r[1])
        else:
            # if r is a quaternion
            q = r.normalize()
        pout = q * Quaternion(0, pin[0], pin[1], pin[2]) * conjugate(q)
        return (pout.b, pout.c, pout.d)

    def to_axis_angle(self):
        """Returns the axis and angle of rotation of a quaternion.

        Returns
        =======

        tuple
            Tuple of (axis, angle)

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> (axis, angle) = q.to_axis_angle()
        >>> axis
        (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)
        >>> angle
        2*pi/3

        """
        q = self
        if q.a.is_negative:
            q = q * -1

        q = q.normalize()
        angle = trigsimp(2 * acos(q.a))

        # Since quaternion is normalised, q.a is less than 1.
        s = sqrt(1 - q.a*q.a)

        x = trigsimp(q.b / s)
        y = trigsimp(q.c / s)
        z = trigsimp(q.d / s)

        v = (x, y, z)
        t = (v, angle)

        return t

    def to_rotation_matrix(self, v=None, homogeneous=True):
        """Returns the equivalent rotation transformation matrix of the quaternion
        which represents rotation about the origin if ``v`` is not passed.

        Parameters
        ==========

        v : tuple or None
            Default value: None
        homogeneous : bool
            When True, gives an expression that may be more efficient for
            symbolic calculations but less so for direct evaluation. Both
            formulas are mathematically equivalent.
            Default value: True

        Returns
        =======

        tuple
            Returns the equivalent rotation transformation matrix of the quaternion
            which represents rotation about the origin if v is not passed.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(q.to_rotation_matrix())
        Matrix([
        [cos(x), -sin(x), 0],
        [sin(x),  cos(x), 0],
        [     0,       0, 1]])

        Generates a 4x4 transformation matrix (used for rotation about a point
        other than the origin) if the point(v) is passed as an argument.
        """

        q = self
        s = q.norm()**-2

        # diagonal elements are different according to parameter normal
        if homogeneous:
            m00 = s*(q.a**2 + q.b**2 - q.c**2 - q.d**2)
            m11 = s*(q.a**2 - q.b**2 + q.c**2 - q.d**2)
            m22 = s*(q.a**2 - q.b**2 - q.c**2 + q.d**2)
        else:
            m00 = 1 - 2*s*(q.c**2 + q.d**2)
            m11 = 1 - 2*s*(q.b**2 + q.d**2)
            m22 = 1 - 2*s*(q.b**2 + q.c**2)

        m01 = 2*s*(q.b*q.c - q.d*q.a)
        m02 = 2*s*(q.b*q.d + q.c*q.a)

        m10 = 2*s*(q.b*q.c + q.d*q.a)
        m12 = 2*s*(q.c*q.d - q.b*q.a)

        m20 = 2*s*(q.b*q.d - q.c*q.a)
        m21 = 2*s*(q.c*q.d + q.b*q.a)

        if not v:
            return Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])

        else:
            (x, y, z) = v

            m03 = x - x*m00 - y*m01 - z*m02
            m13 = y - x*m10 - y*m11 - z*m12
            m23 = z - x*m20 - y*m21 - z*m22
            m30 = m31 = m32 = 0
            m33 = 1

            return Matrix([[m00, m01, m02, m03], [m10, m11, m12, m13],
                          [m20, m21, m22, m23], [m30, m31, m32, m33]])

    def scalar_part(self):
        r"""Returns scalar part($\mathbf{S}(q)$) of the quaternion q.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{S}(q) = a$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(4, 8, 13, 12)
        >>> q.scalar_part()
        4

        """

        return self.a

    def vector_part(self):
        r"""
        Returns $\mathbf{V}(q)$, the vector part of the quaternion $q$.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{V}(q) = bi + cj + dk$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> q.vector_part()
        0 + 1*i + 1*j + 1*k

        >>> q = Quaternion(4, 8, 13, 12)
        >>> q.vector_part()
        0 + 8*i + 13*j + 12*k

        """

        return Quaternion(0, self.b, self.c, self.d)

    def axis(self):
        r"""
        Returns $\mathbf{Ax}(q)$, the axis of the quaternion $q$.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$, returns $\mathbf{Ax}(q)$  i.e., the versor of the vector part of that quaternion
        equal to $\mathbf{U}[\mathbf{V}(q)]$.
        The axis is always an imaginary unit with square equal to $-1 + 0i + 0j + 0k$.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 1, 1, 1)
        >>> q.axis()
        0 + sqrt(3)/3*i + sqrt(3)/3*j + sqrt(3)/3*k

        See Also
        ========

        vector_part

        """
        axis = self.vector_part().normalize()

        return Quaternion(0, axis.b, axis.c, axis.d)

    def is_pure(self):
        """
        Returns true if the quaternion is pure, false if the quaternion is not pure
        or returns none if it is unknown.

        Explanation
        ===========

        A pure quaternion (also a vector quaternion) is a quaternion with scalar
        part equal to 0.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 8, 13, 12)
        >>> q.is_pure()
        True

        See Also
        ========
        scalar_part

        """

        return self.a.is_zero

    def is_zero_quaternion(self):
        """
        Returns true if the quaternion is a zero quaternion or false if it is not a zero quaternion
        and None if the value is unknown.

        Explanation
        ===========

        A zero quaternion is a quaternion with both scalar part and
        vector part equal to 0.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 0, 0, 0)
        >>> q.is_zero_quaternion()
        False

        >>> q = Quaternion(0, 0, 0, 0)
        >>> q.is_zero_quaternion()
        True

        See Also
        ========
        scalar_part
        vector_part

        """

        return self.norm().is_zero

    def angle(self):
        r"""
        Returns the angle of the quaternion measured in the real-axis plane.

        Explanation
        ===========

        Given a quaternion $q = a + bi + cj + dk$ where $a$, $b$, $c$ and $d$
        are real numbers, returns the angle of the quaternion given by

        .. math::
            \theta := 2 \operatorname{atan_2}\left(\sqrt{b^2 + c^2 + d^2}, {a}\right)

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 4, 4, 4)
        >>> q.angle()
        2*atan(4*sqrt(3))

        """

        return 2 * atan2(self.vector_part().norm(), self.scalar_part())


    def arc_coplanar(self, other):
        """
        Returns True if the transformation arcs represented by the input quaternions happen in the same plane.

        Explanation
        ===========

        Two quaternions are said to be coplanar (in this arc sense) when their axes are parallel.
        The plane of a quaternion is the one normal to its axis.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the planes of the two quaternions are the same, apart from its orientation/sign.
        False : if the planes of the two quaternions are not the same, apart from its orientation/sign.
        None : if plane of either of the quaternion is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q1 = Quaternion(1, 4, 4, 4)
        >>> q2 = Quaternion(3, 8, 8, 8)
        >>> Quaternion.arc_coplanar(q1, q2)
        True

        >>> q1 = Quaternion(2, 8, 13, 12)
        >>> Quaternion.arc_coplanar(q1, q2)
        False

        See Also
        ========

        vector_coplanar
        is_pure

        """
        if (self.is_zero_quaternion()) or (other.is_zero_quaternion()):
            raise ValueError('Neither of the given quaternions can be 0')

        return fuzzy_or([(self.axis() - other.axis()).is_zero_quaternion(), (self.axis() + other.axis()).is_zero_quaternion()])

    @classmethod
    def vector_coplanar(cls, q1, q2, q3):
        r"""
        Returns True if the axis of the pure quaternions seen as 3D vectors
        ``q1``, ``q2``, and ``q3`` are coplanar.

        Explanation
        ===========

        Three pure quaternions are vector coplanar if the quaternions seen as 3D vectors are coplanar.

        Parameters
        ==========

        q1
            A pure Quaternion.
        q2
            A pure Quaternion.
        q3
            A pure Quaternion.

        Returns
        =======

        True : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar.
        False : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are not coplanar.
        None : if the axis of the pure quaternions seen as 3D vectors
        q1, q2, and q3 are coplanar is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q1 = Quaternion(0, 4, 4, 4)
        >>> q2 = Quaternion(0, 8, 8, 8)
        >>> q3 = Quaternion(0, 24, 24, 24)
        >>> Quaternion.vector_coplanar(q1, q2, q3)
        True

        >>> q1 = Quaternion(0, 8, 16, 8)
        >>> q2 = Quaternion(0, 8, 3, 12)
        >>> Quaternion.vector_coplanar(q1, q2, q3)
        False

        See Also
        ========

        axis
        is_pure

        """

        if fuzzy_not(q1.is_pure()) or fuzzy_not(q2.is_pure()) or fuzzy_not(q3.is_pure()):
            raise ValueError('The given quaternions must be pure')

        M = Matrix([[q1.b, q1.c, q1.d], [q2.b, q2.c, q2.d], [q3.b, q3.c, q3.d]]).det()
        return M.is_zero

    def parallel(self, other):
        """
        Returns True if the two pure quaternions seen as 3D vectors are parallel.

        Explanation
        ===========

        Two pure quaternions are called parallel when their vector product is commutative which
        implies that the quaternions seen as 3D vectors have same direction.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the two pure quaternions seen as 3D vectors are parallel.
        False : if the two pure quaternions seen as 3D vectors are not parallel.
        None : if the two pure quaternions seen as 3D vectors are parallel is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 4, 4, 4)
        >>> q1 = Quaternion(0, 8, 8, 8)
        >>> q.parallel(q1)
        True

        >>> q1 = Quaternion(0, 8, 13, 12)
        >>> q.parallel(q1)
        False

        """

        if fuzzy_not(self.is_pure()) or fuzzy_not(other.is_pure()):
            raise ValueError('The provided quaternions must be pure')

        return (self*other - other*self).is_zero_quaternion()

    def orthogonal(self, other):
        """
        Returns the orthogonality of two quaternions.

        Explanation
        ===========

        Two pure quaternions are called orthogonal when their product is anti-commutative.

        Parameters
        ==========

        other : a Quaternion

        Returns
        =======

        True : if the two pure quaternions seen as 3D vectors are orthogonal.
        False : if the two pure quaternions seen as 3D vectors are not orthogonal.
        None : if the two pure quaternions seen as 3D vectors are orthogonal is unknown.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(0, 4, 4, 4)
        >>> q1 = Quaternion(0, 8, 8, 8)
        >>> q.orthogonal(q1)
        False

        >>> q1 = Quaternion(0, 2, 2, 0)
        >>> q = Quaternion(0, 2, -2, 0)
        >>> q.orthogonal(q1)
        True

        """

        if fuzzy_not(self.is_pure()) or fuzzy_not(other.is_pure()):
            raise ValueError('The given quaternions must be pure')

        return (self*other + other*self).is_zero_quaternion()

    def index_vector(self):
        r"""
        Returns the index vector of the quaternion.

        Explanation
        ===========

        The index vector is given by $\mathbf{T}(q)$, the norm (or magnitude) of
        the quaternion $q$, multiplied by $\mathbf{Ax}(q)$, the axis of $q$.

        Returns
        =======

        Quaternion: representing index vector of the provided quaternion.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(2, 4, 2, 4)
        >>> q.index_vector()
        0 + 4*sqrt(10)/3*i + 2*sqrt(10)/3*j + 4*sqrt(10)/3*k

        See Also
        ========

        axis
        norm

        """

        return self.norm() * self.axis()

    def mensor(self):
        """
        Returns the natural logarithm of the norm(magnitude) of the quaternion.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(2, 4, 2, 4)
        >>> q.mensor()
        log(2*sqrt(10))
        >>> q.norm()
        2*sqrt(10)

        See Also
        ========

        norm

        """

        return ln(self.norm())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\excel\_base.py ===
from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
import datetime
from functools import partial
from io import BytesIO
import os
from textwrap import fill
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)
import warnings
import zipfile

from pandas._config import config

from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.compat._optional import (
    get_version,
    import_optional_dependency,
)
from pandas.errors import EmptyDataError
from pandas.util._decorators import (
    Appender,
    doc,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    is_bool,
    is_float,
    is_integer,
    is_list_like,
)

from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util.version import Version

from pandas.io.common import (
    IOHandles,
    get_handle,
    stringify_path,
    validate_header_arg,
)
from pandas.io.excel._util import (
    fill_mi_header,
    get_default_engine,
    get_writer,
    maybe_convert_usecols,
    pop_header_name,
)
from pandas.io.parsers import TextParser
from pandas.io.parsers.readers import validate_integer

if TYPE_CHECKING:
    from types import TracebackType

    from pandas._typing import (
        DtypeArg,
        DtypeBackend,
        ExcelWriterIfSheetExists,
        FilePath,
        IntStrT,
        ReadBuffer,
        Self,
        SequenceNotStr,
        StorageOptions,
        WriteExcelBuffer,
    )
_read_excel_doc = (
    """
Read an Excel file into a ``pandas`` ``DataFrame``.

Supports `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt` file extensions
read from a local filesystem or URL. Supports an option to read
a single sheet or a list of sheets.

Parameters
----------
io : str, bytes, ExcelFile, xlrd.Book, path object, or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, and file. For file URLs, a host is
    expected. A local file could be: ``file://localhost/path/to/table.xlsx``.

    If you want to pass in a path object, pandas accepts any ``os.PathLike``.

    By file-like object, we refer to objects with a ``read()`` method,
    such as a file handle (e.g. via builtin ``open`` function)
    or ``StringIO``.

    .. deprecated:: 2.1.0
        Passing byte strings is deprecated. To read from a
        byte string, wrap it in a ``BytesIO`` object.
sheet_name : str, int, list, or None, default 0
    Strings are used for sheet names. Integers are used in zero-indexed
    sheet positions (chart sheets do not count as a sheet position).
    Lists of strings/integers are used to request multiple sheets.
    Specify ``None`` to get all worksheets.

    Available cases:

    * Defaults to ``0``: 1st sheet as a `DataFrame`
    * ``1``: 2nd sheet as a `DataFrame`
    * ``"Sheet1"``: Load sheet with name "Sheet1"
    * ``[0, 1, "Sheet5"]``: Load first, second and sheet named "Sheet5"
      as a dict of `DataFrame`
    * ``None``: All worksheets.

header : int, list of int, default 0
    Row (0-indexed) to use for the column labels of the parsed
    DataFrame. If a list of integers is passed those row positions will
    be combined into a ``MultiIndex``. Use None if there is no header.
names : array-like, default None
    List of column names to use. If file contains no header row,
    then you should explicitly pass header=None.
index_col : int, str, list of int, default None
    Column (0-indexed) to use as the row labels of the DataFrame.
    Pass None if there is no such column.  If a list is passed,
    those columns will be combined into a ``MultiIndex``.  If a
    subset of data is selected with ``usecols``, index_col
    is based on the subset.

    Missing values will be forward filled to allow roundtripping with
    ``to_excel`` for ``merged_cells=True``. To avoid forward filling the
    missing values use ``set_index`` after reading the data instead of
    ``index_col``.
usecols : str, list-like, or callable, default None
    * If None, then parse all columns.
    * If str, then indicates comma separated list of Excel column letters
      and column ranges (e.g. "A:E" or "A,C,E:F"). Ranges are inclusive of
      both sides.
    * If list of int, then indicates list of column numbers to be parsed
      (0-indexed).
    * If list of string, then indicates list of column names to be parsed.
    * If callable, then evaluate each column name against it and parse the
      column if the callable returns ``True``.

    Returns a subset of the columns according to behavior above.
dtype : Type name or dict of column -> type, default None
    Data type for data or columns. E.g. {{'a': np.float64, 'b': np.int32}}
    Use ``object`` to preserve data as stored in Excel and not interpret dtype,
    which will necessarily result in ``object`` dtype.
    If converters are specified, they will be applied INSTEAD
    of dtype conversion.
    If you use ``None``, it will infer the dtype of each column based on the data.
engine : {{'openpyxl', 'calamine', 'odf', 'pyxlsb', 'xlrd'}}, default None
    If io is not a buffer or path, this must be set to identify io.
    Engine compatibility :

    - ``openpyxl`` supports newer Excel file formats.
    - ``calamine`` supports Excel (.xls, .xlsx, .xlsm, .xlsb)
      and OpenDocument (.ods) file formats.
    - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).
    - ``pyxlsb`` supports Binary Excel files.
    - ``xlrd`` supports old-style Excel files (.xls).

    When ``engine=None``, the following logic will be used to determine the engine:

    - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),
      then `odf <https://pypi.org/project/odfpy/>`_ will be used.
    - Otherwise if ``path_or_buffer`` is an xls format, ``xlrd`` will be used.
    - Otherwise if ``path_or_buffer`` is in xlsb format, ``pyxlsb`` will be used.
    - Otherwise ``openpyxl`` will be used.
converters : dict, default None
    Dict of functions for converting values in certain columns. Keys can
    either be integers or column labels, values are functions that take one
    input argument, the Excel cell content, and return the transformed
    content.
true_values : list, default None
    Values to consider as True.
false_values : list, default None
    Values to consider as False.
skiprows : list-like, int, or callable, optional
    Line numbers to skip (0-indexed) or number of lines to skip (int) at the
    start of the file. If callable, the callable function will be evaluated
    against the row indices, returning True if the row should be skipped and
    False otherwise. An example of a valid callable argument would be ``lambda
    x: x in [0, 2]``.
nrows : int, default None
    Number of rows to parse.
na_values : scalar, str, list-like, or dict, default None
    Additional strings to recognize as NA/NaN. If dict passed, specific
    per-column NA values. By default the following values are interpreted
    as NaN: '"""
    + fill("', '".join(sorted(STR_NA_VALUES)), 70, subsequent_indent="    ")
    + """'.
keep_default_na : bool, default True
    Whether or not to include the default NaN values when parsing the data.
    Depending on whether ``na_values`` is passed in, the behavior is as follows:

    * If ``keep_default_na`` is True, and ``na_values`` are specified,
      ``na_values`` is appended to the default NaN values used for parsing.
    * If ``keep_default_na`` is True, and ``na_values`` are not specified, only
      the default NaN values are used for parsing.
    * If ``keep_default_na`` is False, and ``na_values`` are specified, only
      the NaN values specified ``na_values`` are used for parsing.
    * If ``keep_default_na`` is False, and ``na_values`` are not specified, no
      strings will be parsed as NaN.

    Note that if `na_filter` is passed in as False, the ``keep_default_na`` and
    ``na_values`` parameters will be ignored.
na_filter : bool, default True
    Detect missing value markers (empty strings and the value of na_values). In
    data without any NAs, passing ``na_filter=False`` can improve the
    performance of reading a large file.
verbose : bool, default False
    Indicate number of NA values placed in non-numeric columns.
parse_dates : bool, list-like, or dict, default False
    The behavior is as follows:

    * ``bool``. If True -> try parsing the index.
    * ``list`` of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
      each as a separate date column.
    * ``list`` of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
      a single date column.
    * ``dict``, e.g. {{'foo' : [1, 3]}} -> parse columns 1, 3 as date and call
      result 'foo'

    If a column or index contains an unparsable date, the entire column or
    index will be returned unaltered as an object data type. If you don`t want to
    parse some cells as date just change their type in Excel to "Text".
    For non-standard datetime parsing, use ``pd.to_datetime`` after ``pd.read_excel``.

    Note: A fast-path exists for iso8601-formatted dates.
date_parser : function, optional
    Function to use for converting a sequence of string columns to an array of
    datetime instances. The default uses ``dateutil.parser.parser`` to do the
    conversion. Pandas will try to call `date_parser` in three different ways,
    advancing to the next if an exception occurs: 1) Pass one or more arrays
    (as defined by `parse_dates`) as arguments; 2) concatenate (row-wise) the
    string values from the columns defined by `parse_dates` into a single array
    and pass that; and 3) call `date_parser` once for each row using one or
    more strings (corresponding to the columns defined by `parse_dates`) as
    arguments.

    .. deprecated:: 2.0.0
       Use ``date_format`` instead, or read in as ``object`` and then apply
       :func:`to_datetime` as-needed.
date_format : str or dict of column -> format, default ``None``
   If used in conjunction with ``parse_dates``, will parse dates according to this
   format. For anything more complex,
   please read in as ``object`` and then apply :func:`to_datetime` as-needed.

   .. versionadded:: 2.0.0
thousands : str, default None
    Thousands separator for parsing string columns to numeric.  Note that
    this parameter is only necessary for columns stored as TEXT in Excel,
    any numeric columns will automatically be parsed, regardless of display
    format.
decimal : str, default '.'
    Character to recognize as decimal point for parsing string columns to numeric.
    Note that this parameter is only necessary for columns stored as TEXT in Excel,
    any numeric columns will automatically be parsed, regardless of display
    format.(e.g. use ',' for European data).

    .. versionadded:: 1.4.0

comment : str, default None
    Comments out remainder of line. Pass a character or characters to this
    argument to indicate comments in the input file. Any data between the
    comment string and the end of the current line is ignored.
skipfooter : int, default 0
    Rows at the end to skip (0-indexed).
{storage_options}

dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
    Back-end data type applied to the resultant :class:`DataFrame`
    (still experimental). Behaviour is as follows:

    * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
      (default).
    * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
      DataFrame.

    .. versionadded:: 2.0

engine_kwargs : dict, optional
    Arbitrary keyword arguments passed to excel engine.

Returns
-------
DataFrame or dict of DataFrames
    DataFrame from the passed in Excel file. See notes in sheet_name
    argument for more information on when a dict of DataFrames is returned.

See Also
--------
DataFrame.to_excel : Write DataFrame to an Excel file.
DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
read_csv : Read a comma-separated values (csv) file into DataFrame.
read_fwf : Read a table of fixed-width formatted lines into DataFrame.

Notes
-----
For specific information on the methods used for each Excel engine, refer to the pandas
:ref:`user guide <io.excel_reader>`

Examples
--------
The file can be read using the file name as string or an open file object:

>>> pd.read_excel('tmp.xlsx', index_col=0)  # doctest: +SKIP
       Name  Value
0   string1      1
1   string2      2
2  #Comment      3

>>> pd.read_excel(open('tmp.xlsx', 'rb'),
...               sheet_name='Sheet3')  # doctest: +SKIP
   Unnamed: 0      Name  Value
0           0   string1      1
1           1   string2      2
2           2  #Comment      3

Index and header can be specified via the `index_col` and `header` arguments

>>> pd.read_excel('tmp.xlsx', index_col=None, header=None)  # doctest: +SKIP
     0         1      2
0  NaN      Name  Value
1  0.0   string1      1
2  1.0   string2      2
3  2.0  #Comment      3

Column types are inferred but can be explicitly specified

>>> pd.read_excel('tmp.xlsx', index_col=0,
...               dtype={{'Name': str, 'Value': float}})  # doctest: +SKIP
       Name  Value
0   string1    1.0
1   string2    2.0
2  #Comment    3.0

True, False, and NA values, and thousands separators have defaults,
but can be explicitly specified, too. Supply the values you would like
as strings or lists of strings!

>>> pd.read_excel('tmp.xlsx', index_col=0,
...               na_values=['string1', 'string2'])  # doctest: +SKIP
       Name  Value
0       NaN      1
1       NaN      2
2  #Comment      3

Comment lines in the excel input file can be skipped using the
``comment`` kwarg.

>>> pd.read_excel('tmp.xlsx', index_col=0, comment='#')  # doctest: +SKIP
      Name  Value
0  string1    1.0
1  string2    2.0
2     None    NaN
"""
)


@overload
def read_excel(
    io,
    # sheet name is str or int -> DataFrame
    sheet_name: str | int = ...,
    *,
    header: int | Sequence[int] | None = ...,
    names: SequenceNotStr[Hashable] | range | None = ...,
    index_col: int | str | Sequence[int] | None = ...,
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[str], bool]
    | None = ...,
    dtype: DtypeArg | None = ...,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = ...,
    converters: dict[str, Callable] | dict[int, Callable] | None = ...,
    true_values: Iterable[Hashable] | None = ...,
    false_values: Iterable[Hashable] | None = ...,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: list | dict | bool = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...


@overload
def read_excel(
    io,
    # sheet name is list or None -> dict[IntStrT, DataFrame]
    sheet_name: list[IntStrT] | None,
    *,
    header: int | Sequence[int] | None = ...,
    names: SequenceNotStr[Hashable] | range | None = ...,
    index_col: int | str | Sequence[int] | None = ...,
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[str], bool]
    | None = ...,
    dtype: DtypeArg | None = ...,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = ...,
    converters: dict[str, Callable] | dict[int, Callable] | None = ...,
    true_values: Iterable[Hashable] | None = ...,
    false_values: Iterable[Hashable] | None = ...,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: list | dict | bool = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> dict[IntStrT, DataFrame]:
    ...


@doc(storage_options=_shared_docs["storage_options"])
@Appender(_read_excel_doc)
def read_excel(
    io,
    sheet_name: str | int | list[IntStrT] | None = 0,
    *,
    header: int | Sequence[int] | None = 0,
    names: SequenceNotStr[Hashable] | range | None = None,
    index_col: int | str | Sequence[int] | None = None,
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[str], bool]
    | None = None,
    dtype: DtypeArg | None = None,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = None,
    converters: dict[str, Callable] | dict[int, Callable] | None = None,
    true_values: Iterable[Hashable] | None = None,
    false_values: Iterable[Hashable] | None = None,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
    nrows: int | None = None,
    na_values=None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: list | dict | bool = False,
    date_parser: Callable | lib.NoDefault = lib.no_default,
    date_format: dict[Hashable, str] | str | None = None,
    thousands: str | None = None,
    decimal: str = ".",
    comment: str | None = None,
    skipfooter: int = 0,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    engine_kwargs: dict | None = None,
) -> DataFrame | dict[IntStrT, DataFrame]:
    check_dtype_backend(dtype_backend)
    should_close = False
    if engine_kwargs is None:
        engine_kwargs = {}

    if not isinstance(io, ExcelFile):
        should_close = True
        io = ExcelFile(
            io,
            storage_options=storage_options,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )
    elif engine and engine != io.engine:
        raise ValueError(
            "Engine should not be specified when passing "
            "an ExcelFile - ExcelFile already has the engine set"
        )

    try:
        data = io.parse(
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            na_filter=na_filter,
            verbose=verbose,
            parse_dates=parse_dates,
            date_parser=date_parser,
            date_format=date_format,
            thousands=thousands,
            decimal=decimal,
            comment=comment,
            skipfooter=skipfooter,
            dtype_backend=dtype_backend,
        )
    finally:
        # make sure to close opened file handles
        if should_close:
            io.close()
    return data


_WorkbookT = TypeVar("_WorkbookT")


class BaseExcelReader(Generic[_WorkbookT]):
    book: _WorkbookT

    def __init__(
        self,
        filepath_or_buffer,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}

        # First argument can also be bytes, so create a buffer
        if isinstance(filepath_or_buffer, bytes):
            filepath_or_buffer = BytesIO(filepath_or_buffer)

        self.handles = IOHandles(
            handle=filepath_or_buffer, compression={"method": None}
        )
        if not isinstance(filepath_or_buffer, (ExcelFile, self._workbook_class)):
            self.handles = get_handle(
                filepath_or_buffer, "rb", storage_options=storage_options, is_text=False
            )

        if isinstance(self.handles.handle, self._workbook_class):
            self.book = self.handles.handle
        elif hasattr(self.handles.handle, "read"):
            # N.B. xlrd.Book has a read attribute too
            self.handles.handle.seek(0)
            try:
                self.book = self.load_workbook(self.handles.handle, engine_kwargs)
            except Exception:
                self.close()
                raise
        else:
            raise ValueError(
                "Must explicitly set engine if not passing in buffer or path for io."
            )

    @property
    def _workbook_class(self) -> type[_WorkbookT]:
        raise NotImplementedError

    def load_workbook(self, filepath_or_buffer, engine_kwargs) -> _WorkbookT:
        raise NotImplementedError

    def close(self) -> None:
        if hasattr(self, "book"):
            if hasattr(self.book, "close"):
                # pyxlsb: opens a TemporaryFile
                # openpyxl: https://stackoverflow.com/questions/31416842/
                #     openpyxl-does-not-close-excel-workbook-in-read-only-mode
                self.book.close()
            elif hasattr(self.book, "release_resources"):
                # xlrd
                # https://github.com/python-excel/xlrd/blob/2.0.1/xlrd/book.py#L548
                self.book.release_resources()
        self.handles.close()

    @property
    def sheet_names(self) -> list[str]:
        raise NotImplementedError

    def get_sheet_by_name(self, name: str):
        raise NotImplementedError

    def get_sheet_by_index(self, index: int):
        raise NotImplementedError

    def get_sheet_data(self, sheet, rows: int | None = None):
        raise NotImplementedError

    def raise_if_bad_sheet_by_index(self, index: int) -> None:
        n_sheets = len(self.sheet_names)
        if index >= n_sheets:
            raise ValueError(
                f"Worksheet index {index} is invalid, {n_sheets} worksheets found"
            )

    def raise_if_bad_sheet_by_name(self, name: str) -> None:
        if name not in self.sheet_names:
            raise ValueError(f"Worksheet named '{name}' not found")

    def _check_skiprows_func(
        self,
        skiprows: Callable,
        rows_to_use: int,
    ) -> int:
        """
        Determine how many file rows are required to obtain `nrows` data
        rows when `skiprows` is a function.

        Parameters
        ----------
        skiprows : function
            The function passed to read_excel by the user.
        rows_to_use : int
            The number of rows that will be needed for the header and
            the data.

        Returns
        -------
        int
        """
        i = 0
        rows_used_so_far = 0
        while rows_used_so_far < rows_to_use:
            if not skiprows(i):
                rows_used_so_far += 1
            i += 1
        return i

    def _calc_rows(
        self,
        header: int | Sequence[int] | None,
        index_col: int | Sequence[int] | None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None,
        nrows: int | None,
    ) -> int | None:
        """
        If nrows specified, find the number of rows needed from the
        file, otherwise return None.


        Parameters
        ----------
        header : int, list of int, or None
            See read_excel docstring.
        index_col : int, str, list of int, or None
            See read_excel docstring.
        skiprows : list-like, int, callable, or None
            See read_excel docstring.
        nrows : int or None
            See read_excel docstring.

        Returns
        -------
        int or None
        """
        if nrows is None:
            return None
        if header is None:
            header_rows = 1
        elif is_integer(header):
            header = cast(int, header)
            header_rows = 1 + header
        else:
            header = cast(Sequence, header)
            header_rows = 1 + header[-1]
        # If there is a MultiIndex header and an index then there is also
        # a row containing just the index name(s)
        if is_list_like(header) and index_col is not None:
            header = cast(Sequence, header)
            if len(header) > 1:
                header_rows += 1
        if skiprows is None:
            return header_rows + nrows
        if is_integer(skiprows):
            skiprows = cast(int, skiprows)
            return header_rows + nrows + skiprows
        if is_list_like(skiprows):

            def f(skiprows: Sequence, x: int) -> bool:
                return x in skiprows

            skiprows = cast(Sequence, skiprows)
            return self._check_skiprows_func(partial(f, skiprows), header_rows + nrows)
        if callable(skiprows):
            return self._check_skiprows_func(
                skiprows,
                header_rows + nrows,
            )
        # else unexpected skiprows type: read_excel will not optimize
        # the number of rows read from file
        return None

    def parse(
        self,
        sheet_name: str | int | list[int] | list[str] | None = 0,
        header: int | Sequence[int] | None = 0,
        names: SequenceNotStr[Hashable] | range | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols=None,
        dtype: DtypeArg | None = None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        nrows: int | None = None,
        na_values=None,
        verbose: bool = False,
        parse_dates: list | dict | bool = False,
        date_parser: Callable | lib.NoDefault = lib.no_default,
        date_format: dict[Hashable, str] | str | None = None,
        thousands: str | None = None,
        decimal: str = ".",
        comment: str | None = None,
        skipfooter: int = 0,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwds,
    ):
        validate_header_arg(header)
        validate_integer("nrows", nrows)

        ret_dict = False

        # Keep sheetname to maintain backwards compatibility.
        sheets: list[int] | list[str]
        if isinstance(sheet_name, list):
            sheets = sheet_name
            ret_dict = True
        elif sheet_name is None:
            sheets = self.sheet_names
            ret_dict = True
        elif isinstance(sheet_name, str):
            sheets = [sheet_name]
        else:
            sheets = [sheet_name]

        # handle same-type duplicates.
        sheets = cast(Union[list[int], list[str]], list(dict.fromkeys(sheets).keys()))

        output = {}

        last_sheetname = None
        for asheetname in sheets:
            last_sheetname = asheetname
            if verbose:
                print(f"Reading sheet {asheetname}")

            if isinstance(asheetname, str):
                sheet = self.get_sheet_by_name(asheetname)
            else:  # assume an integer if not a string
                sheet = self.get_sheet_by_index(asheetname)

            file_rows_needed = self._calc_rows(header, index_col, skiprows, nrows)
            data = self.get_sheet_data(sheet, file_rows_needed)
            if hasattr(sheet, "close"):
                # pyxlsb opens two TemporaryFiles
                sheet.close()
            usecols = maybe_convert_usecols(usecols)

            if not data:
                output[asheetname] = DataFrame()
                continue

            is_list_header = False
            is_len_one_list_header = False
            if is_list_like(header):
                assert isinstance(header, Sequence)
                is_list_header = True
                if len(header) == 1:
                    is_len_one_list_header = True

            if is_len_one_list_header:
                header = cast(Sequence[int], header)[0]

            # forward fill and pull out names for MultiIndex column
            header_names = None
            if header is not None and is_list_like(header):
                assert isinstance(header, Sequence)

                header_names = []
                control_row = [True] * len(data[0])

                for row in header:
                    if is_integer(skiprows):
                        assert isinstance(skiprows, int)
                        row += skiprows

                    if row > len(data) - 1:
                        raise ValueError(
                            f"header index {row} exceeds maximum index "
                            f"{len(data) - 1} of data.",
                        )

                    data[row], control_row = fill_mi_header(data[row], control_row)

                    if index_col is not None:
                        header_name, _ = pop_header_name(data[row], index_col)
                        header_names.append(header_name)

            # If there is a MultiIndex header and an index then there is also
            # a row containing just the index name(s)
            has_index_names = False
            if is_list_header and not is_len_one_list_header and index_col is not None:
                index_col_list: Sequence[int]
                if isinstance(index_col, int):
                    index_col_list = [index_col]
                else:
                    assert isinstance(index_col, Sequence)
                    index_col_list = index_col

                # We have to handle mi without names. If any of the entries in the data
                # columns are not empty, this is a regular row
                assert isinstance(header, Sequence)
                if len(header) < len(data):
                    potential_index_names = data[len(header)]
                    potential_data = [
                        x
                        for i, x in enumerate(potential_index_names)
                        if not control_row[i] and i not in index_col_list
                    ]
                    has_index_names = all(x == "" or x is None for x in potential_data)

            if is_list_like(index_col):
                # Forward fill values for MultiIndex index.
                if header is None:
                    offset = 0
                elif isinstance(header, int):
                    offset = 1 + header
                else:
                    offset = 1 + max(header)

                # GH34673: if MultiIndex names present and not defined in the header,
                # offset needs to be incremented so that forward filling starts
                # from the first MI value instead of the name
                if has_index_names:
                    offset += 1

                # Check if we have an empty dataset
                # before trying to collect data.
                if offset < len(data):
                    assert isinstance(index_col, Sequence)

                    for col in index_col:
                        last = data[offset][col]

                        for row in range(offset + 1, len(data)):
                            if data[row][col] == "" or data[row][col] is None:
                                data[row][col] = last
                            else:
                                last = data[row][col]

            # GH 12292 : error when read one empty column from excel file
            try:
                parser = TextParser(
                    data,
                    names=names,
                    header=header,
                    index_col=index_col,
                    has_index_names=has_index_names,
                    dtype=dtype,
                    true_values=true_values,
                    false_values=false_values,
                    skiprows=skiprows,
                    nrows=nrows,
                    na_values=na_values,
                    skip_blank_lines=False,  # GH 39808
                    parse_dates=parse_dates,
                    date_parser=date_parser,
                    date_format=date_format,
                    thousands=thousands,
                    decimal=decimal,
                    comment=comment,
                    skipfooter=skipfooter,
                    usecols=usecols,
                    dtype_backend=dtype_backend,
                    **kwds,
                )

                output[asheetname] = parser.read(nrows=nrows)

                if header_names:
                    output[asheetname].columns = output[asheetname].columns.set_names(
                        header_names
                    )

            except EmptyDataError:
                # No Data, return an empty DataFrame
                output[asheetname] = DataFrame()

            except Exception as err:
                err.args = (f"{err.args[0]} (sheet: {asheetname})", *err.args[1:])
                raise err

        if last_sheetname is None:
            raise ValueError("Sheet name is an empty list")

        if ret_dict:
            return output
        else:
            return output[last_sheetname]


@doc(storage_options=_shared_docs["storage_options"])
class ExcelWriter(Generic[_WorkbookT]):
    """
    Class for writing DataFrame objects into excel sheets.

    Default is to use:

    * `xlsxwriter <https://pypi.org/project/XlsxWriter/>`__ for xlsx files if xlsxwriter
      is installed otherwise `openpyxl <https://pypi.org/project/openpyxl/>`__
    * `odswriter <https://pypi.org/project/odswriter/>`__ for ods files

    See ``DataFrame.to_excel`` for typical usage.

    The writer should be used as a context manager. Otherwise, call `close()` to save
    and close any opened file handles.

    Parameters
    ----------
    path : str or typing.BinaryIO
        Path to xls or xlsx or ods file.
    engine : str (optional)
        Engine to use for writing. If None, defaults to
        ``io.excel.<extension>.writer``.  NOTE: can only be passed as a keyword
        argument.
    date_format : str, default None
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
    datetime_format : str, default None
        Format string for datetime objects written into Excel files.
        (e.g. 'YYYY-MM-DD HH:MM:SS').
    mode : {{'w', 'a'}}, default 'w'
        File mode to use (write or append). Append does not work with fsspec URLs.
    {storage_options}

    if_sheet_exists : {{'error', 'new', 'replace', 'overlay'}}, default 'error'
        How to behave when trying to write to a sheet that already
        exists (append mode only).

        * error: raise a ValueError.
        * new: Create a new sheet, with a name determined by the engine.
        * replace: Delete the contents of the sheet before writing to it.
        * overlay: Write contents to the existing sheet without first removing,
          but possibly over top of, the existing contents.

        .. versionadded:: 1.3.0

        .. versionchanged:: 1.4.0

           Added ``overlay`` option

    engine_kwargs : dict, optional
        Keyword arguments to be passed into the engine. These will be passed to
        the following functions of the respective engines:

        * xlsxwriter: ``xlsxwriter.Workbook(file, **engine_kwargs)``
        * openpyxl (write mode): ``openpyxl.Workbook(**engine_kwargs)``
        * openpyxl (append mode): ``openpyxl.load_workbook(file, **engine_kwargs)``
        * odswriter: ``odf.opendocument.OpenDocumentSpreadsheet(**engine_kwargs)``

        .. versionadded:: 1.3.0

    Notes
    -----
    For compatibility with CSV writers, ExcelWriter serializes lists
    and dicts to strings before writing.

    Examples
    --------
    Default usage:

    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with pd.ExcelWriter("path_to_file.xlsx") as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    To write to separate sheets in a single file:

    >>> df1 = pd.DataFrame([["AAA", "BBB"]], columns=["Spam", "Egg"])  # doctest: +SKIP
    >>> df2 = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with pd.ExcelWriter("path_to_file.xlsx") as writer:
    ...     df1.to_excel(writer, sheet_name="Sheet1")  # doctest: +SKIP
    ...     df2.to_excel(writer, sheet_name="Sheet2")  # doctest: +SKIP

    You can set the date format or datetime format:

    >>> from datetime import date, datetime  # doctest: +SKIP
    >>> df = pd.DataFrame(
    ...     [
    ...         [date(2014, 1, 31), date(1999, 9, 24)],
    ...         [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)],
    ...     ],
    ...     index=["Date", "Datetime"],
    ...     columns=["X", "Y"],
    ... )  # doctest: +SKIP
    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     date_format="YYYY-MM-DD",
    ...     datetime_format="YYYY-MM-DD HH:MM:SS"
    ... ) as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    You can also append to an existing Excel file:

    >>> with pd.ExcelWriter("path_to_file.xlsx", mode="a", engine="openpyxl") as writer:
    ...     df.to_excel(writer, sheet_name="Sheet3")  # doctest: +SKIP

    Here, the `if_sheet_exists` parameter can be set to replace a sheet if it
    already exists:

    >>> with ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     mode="a",
    ...     engine="openpyxl",
    ...     if_sheet_exists="replace",
    ... ) as writer:
    ...     df.to_excel(writer, sheet_name="Sheet1")  # doctest: +SKIP

    You can also write multiple DataFrames to a single sheet. Note that the
    ``if_sheet_exists`` parameter needs to be set to ``overlay``:

    >>> with ExcelWriter("path_to_file.xlsx",
    ...     mode="a",
    ...     engine="openpyxl",
    ...     if_sheet_exists="overlay",
    ... ) as writer:
    ...     df1.to_excel(writer, sheet_name="Sheet1")
    ...     df2.to_excel(writer, sheet_name="Sheet1", startcol=3)  # doctest: +SKIP

    You can store Excel file in RAM:

    >>> import io
    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])
    >>> buffer = io.BytesIO()
    >>> with pd.ExcelWriter(buffer) as writer:
    ...     df.to_excel(writer)

    You can pack Excel file into zip archive:

    >>> import zipfile  # doctest: +SKIP
    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with zipfile.ZipFile("path_to_file.zip", "w") as zf:
    ...     with zf.open("filename.xlsx", "w") as buffer:
    ...         with pd.ExcelWriter(buffer) as writer:
    ...             df.to_excel(writer)  # doctest: +SKIP

    You can specify additional arguments to the underlying engine:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     engine="xlsxwriter",
    ...     engine_kwargs={{"options": {{"nan_inf_to_errors": True}}}}
    ... ) as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    In append mode, ``engine_kwargs`` are passed through to
    openpyxl's ``load_workbook``:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     engine="openpyxl",
    ...     mode="a",
    ...     engine_kwargs={{"keep_vba": True}}
    ... ) as writer:
    ...     df.to_excel(writer, sheet_name="Sheet2")  # doctest: +SKIP
    """

    # Defining an ExcelWriter implementation (see abstract methods for more...)

    # - Mandatory
    #   - ``write_cells(self, cells, sheet_name=None, startrow=0, startcol=0)``
    #     --> called to write additional DataFrames to disk
    #   - ``_supported_extensions`` (tuple of supported extensions), used to
    #      check that engine supports the given extension.
    #   - ``_engine`` - string that gives the engine name. Necessary to
    #     instantiate class directly and bypass ``ExcelWriterMeta`` engine
    #     lookup.
    #   - ``save(self)`` --> called to save file to disk
    # - Mostly mandatory (i.e. should at least exist)
    #   - book, cur_sheet, path

    # - Optional:
    #   - ``__init__(self, path, engine=None, **kwargs)`` --> always called
    #     with path as first argument.

    # You also need to register the class with ``register_writer()``.
    # Technically, ExcelWriter implementations don't need to subclass
    # ExcelWriter.

    _engine: str
    _supported_extensions: tuple[str, ...]

    def __new__(
        cls,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict | None = None,
    ) -> Self:
        # only switch class if generic(ExcelWriter)
        if cls is ExcelWriter:
            if engine is None or (isinstance(engine, str) and engine == "auto"):
                if isinstance(path, str):
                    ext = os.path.splitext(path)[-1][1:]
                else:
                    ext = "xlsx"

                try:
                    engine = config.get_option(f"io.excel.{ext}.writer", silent=True)
                    if engine == "auto":
                        engine = get_default_engine(ext, mode="writer")
                except KeyError as err:
                    raise ValueError(f"No engine for filetype: '{ext}'") from err

            # for mypy
            assert engine is not None
            #  error: Incompatible types in assignment (expression has type
            #  "type[ExcelWriter[Any]]", variable has type "type[Self]")
            cls = get_writer(engine)  # type: ignore[assignment]

        return object.__new__(cls)

    # declare external properties you can count on
    _path = None

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Extensions that writer engine supports."""
        return self._supported_extensions

    @property
    def engine(self) -> str:
        """Name of engine."""
        return self._engine

    @property
    def sheets(self) -> dict[str, Any]:
        """Mapping of sheet names to sheet objects."""
        raise NotImplementedError

    @property
    def book(self) -> _WorkbookT:
        """
        Book instance. Class type will depend on the engine used.

        This attribute can be used to access engine-specific features.
        """
        raise NotImplementedError

    def _write_cells(
        self,
        cells,
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
    ) -> None:
        """
        Write given formatted cells into Excel an excel sheet

        Parameters
        ----------
        cells : generator
            cell of formatted data to save to Excel sheet
        sheet_name : str, default None
            Name of Excel sheet, if None, then use self.cur_sheet
        startrow : upper left cell row to dump data frame
        startcol : upper left cell column to dump data frame
        freeze_panes: int tuple of length 2
            contains the bottom-most row and right-most column to freeze
        """
        raise NotImplementedError

    def _save(self) -> None:
        """
        Save workbook to disk.
        """
        raise NotImplementedError

    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # validate that this engine can handle the extension
        if isinstance(path, str):
            ext = os.path.splitext(path)[-1]
            self.check_extension(ext)

        # use mode to open the file
        if "b" not in mode:
            mode += "b"
        # use "a" for the user to append data to excel but internally use "r+" to let
        # the excel backend first read the existing file and then write any data to it
        mode = mode.replace("a", "r+")

        if if_sheet_exists not in (None, "error", "new", "replace", "overlay"):
            raise ValueError(
                f"'{if_sheet_exists}' is not valid for if_sheet_exists. "
                "Valid options are 'error', 'new', 'replace' and 'overlay'."
            )
        if if_sheet_exists and "r+" not in mode:
            raise ValueError("if_sheet_exists is only valid in append mode (mode='a')")
        if if_sheet_exists is None:
            if_sheet_exists = "error"
        self._if_sheet_exists = if_sheet_exists

        # cast ExcelWriter to avoid adding 'if self._handles is not None'
        self._handles = IOHandles(
            cast(IO[bytes], path), compression={"compression": None}
        )
        if not isinstance(path, ExcelWriter):
            self._handles = get_handle(
                path, mode, storage_options=storage_options, is_text=False
            )
        self._cur_sheet = None

        if date_format is None:
            self._date_format = "YYYY-MM-DD"
        else:
            self._date_format = date_format
        if datetime_format is None:
            self._datetime_format = "YYYY-MM-DD HH:MM:SS"
        else:
            self._datetime_format = datetime_format

        self._mode = mode

    @property
    def date_format(self) -> str:
        """
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
        """
        return self._date_format

    @property
    def datetime_format(self) -> str:
        """
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
        """
        return self._datetime_format

    @property
    def if_sheet_exists(self) -> str:
        """
        How to behave when writing to a sheet that already exists in append mode.
        """
        return self._if_sheet_exists

    def __fspath__(self) -> str:
        return getattr(self._handles.handle, "name", "")

    def _get_sheet_name(self, sheet_name: str | None) -> str:
        if sheet_name is None:
            sheet_name = self._cur_sheet
        if sheet_name is None:  # pragma: no cover
            raise ValueError("Must pass explicit sheet_name or set _cur_sheet property")
        return sheet_name

    def _value_with_fmt(
        self, val
    ) -> tuple[
        int | float | bool | str | datetime.datetime | datetime.date, str | None
    ]:
        """
        Convert numpy types to Python types for the Excel writers.

        Parameters
        ----------
        val : object
            Value to be written into cells

        Returns
        -------
        Tuple with the first element being the converted value and the second
            being an optional format
        """
        fmt = None

        if is_integer(val):
            val = int(val)
        elif is_float(val):
            val = float(val)
        elif is_bool(val):
            val = bool(val)
        elif isinstance(val, datetime.datetime):
            fmt = self._datetime_format
        elif isinstance(val, datetime.date):
            fmt = self._date_format
        elif isinstance(val, datetime.timedelta):
            val = val.total_seconds() / 86400
            fmt = "0"
        else:
            val = str(val)

        return val, fmt

    @classmethod
    def check_extension(cls, ext: str) -> Literal[True]:
        """
        checks that path's extension against the Writer's supported
        extensions.  If it isn't supported, raises UnsupportedFiletypeError.
        """
        if ext.startswith("."):
            ext = ext[1:]
        if not any(ext in extension for extension in cls._supported_extensions):
            raise ValueError(f"Invalid extension for engine '{cls.engine}': '{ext}'")
        return True

    # Allow use as a contextmanager
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """synonym for save, to make it more file-like"""
        self._save()
        self._handles.close()


XLS_SIGNATURES = (
    b"\x09\x00\x04\x00\x07\x00\x10\x00",  # BIFF2
    b"\x09\x02\x06\x00\x00\x00\x10\x00",  # BIFF3
    b"\x09\x04\x06\x00\x00\x00\x10\x00",  # BIFF4
    b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1",  # Compound File Binary
)
ZIP_SIGNATURE = b"PK\x03\x04"
PEEK_SIZE = max(map(len, XLS_SIGNATURES + (ZIP_SIGNATURE,)))


@doc(storage_options=_shared_docs["storage_options"])
def inspect_excel_format(
    content_or_path: FilePath | ReadBuffer[bytes],
    storage_options: StorageOptions | None = None,
) -> str | None:
    """
    Inspect the path or content of an excel file and get its format.

    Adopted from xlrd: https://github.com/python-excel/xlrd.

    Parameters
    ----------
    content_or_path : str or file-like object
        Path to file or content of file to inspect. May be a URL.
    {storage_options}

    Returns
    -------
    str or None
        Format of file if it can be determined.

    Raises
    ------
    ValueError
        If resulting stream is empty.
    BadZipFile
        If resulting stream does not have an XLS signature and is not a valid zipfile.
    """
    if isinstance(content_or_path, bytes):
        content_or_path = BytesIO(content_or_path)

    with get_handle(
        content_or_path, "rb", storage_options=storage_options, is_text=False
    ) as handle:
        stream = handle.handle
        stream.seek(0)
        buf = stream.read(PEEK_SIZE)
        if buf is None:
            raise ValueError("stream is empty")
        assert isinstance(buf, bytes)
        peek = buf
        stream.seek(0)

        if any(peek.startswith(sig) for sig in XLS_SIGNATURES):
            return "xls"
        elif not peek.startswith(ZIP_SIGNATURE):
            return None

        with zipfile.ZipFile(stream) as zf:
            # Workaround for some third party files that use forward slashes and
            # lower case names.
            component_names = [
                name.replace("\\", "/").lower() for name in zf.namelist()
            ]

        if "xl/workbook.xml" in component_names:
            return "xlsx"
        if "xl/workbook.bin" in component_names:
            return "xlsb"
        if "content.xml" in component_names:
            return "ods"
        return "zip"


class ExcelFile:
    """
    Class for parsing tabular Excel sheets into DataFrame objects.

    See read_excel for more documentation.

    Parameters
    ----------
    path_or_buffer : str, bytes, path object (pathlib.Path or py._path.local.LocalPath),
        A file-like object, xlrd workbook or openpyxl workbook.
        If a string or path object, expected to be a path to a
        .xls, .xlsx, .xlsb, .xlsm, .odf, .ods, or .odt file.
    engine : str, default None
        If io is not a buffer or path, this must be set to identify io.
        Supported engines: ``xlrd``, ``openpyxl``, ``odf``, ``pyxlsb``, ``calamine``
        Engine compatibility :

        - ``xlrd`` supports old-style Excel files (.xls).
        - ``openpyxl`` supports newer Excel file formats.
        - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).
        - ``pyxlsb`` supports Binary Excel files.
        - ``calamine`` supports Excel (.xls, .xlsx, .xlsm, .xlsb)
          and OpenDocument (.ods) file formats.

        .. versionchanged:: 1.2.0

           The engine `xlrd <https://xlrd.readthedocs.io/en/latest/>`_
           now only supports old-style ``.xls`` files.
           When ``engine=None``, the following logic will be
           used to determine the engine:

           - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),
             then `odf <https://pypi.org/project/odfpy/>`_ will be used.
           - Otherwise if ``path_or_buffer`` is an xls format,
             ``xlrd`` will be used.
           - Otherwise if ``path_or_buffer`` is in xlsb format,
             `pyxlsb <https://pypi.org/project/pyxlsb/>`_ will be used.

           .. versionadded:: 1.3.0

           - Otherwise if `openpyxl <https://pypi.org/project/openpyxl/>`_ is installed,
             then ``openpyxl`` will be used.
           - Otherwise if ``xlrd >= 2.0`` is installed, a ``ValueError`` will be raised.

           .. warning::

            Please do not report issues when using ``xlrd`` to read ``.xlsx`` files.
            This is not supported, switch to using ``openpyxl`` instead.
    engine_kwargs : dict, optional
        Arbitrary keyword arguments passed to excel engine.

    Examples
    --------
    >>> file = pd.ExcelFile('myfile.xlsx')  # doctest: +SKIP
    >>> with pd.ExcelFile("myfile.xls") as xls:  # doctest: +SKIP
    ...     df1 = pd.read_excel(xls, "Sheet1")  # doctest: +SKIP
    """

    from pandas.io.excel._calamine import CalamineReader
    from pandas.io.excel._odfreader import ODFReader
    from pandas.io.excel._openpyxl import OpenpyxlReader
    from pandas.io.excel._pyxlsb import PyxlsbReader
    from pandas.io.excel._xlrd import XlrdReader

    _engines: Mapping[str, Any] = {
        "xlrd": XlrdReader,
        "openpyxl": OpenpyxlReader,
        "odf": ODFReader,
        "pyxlsb": PyxlsbReader,
        "calamine": CalamineReader,
    }

    def __init__(
        self,
        path_or_buffer,
        engine: str | None = None,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}

        if engine is not None and engine not in self._engines:
            raise ValueError(f"Unknown engine: {engine}")

        # First argument can also be bytes, so create a buffer
        if isinstance(path_or_buffer, bytes):
            path_or_buffer = BytesIO(path_or_buffer)
            warnings.warn(
                "Passing bytes to 'read_excel' is deprecated and "
                "will be removed in a future version. To read from a "
                "byte string, wrap it in a `BytesIO` object.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        # Could be a str, ExcelFile, Book, etc.
        self.io = path_or_buffer
        # Always a string
        self._io = stringify_path(path_or_buffer)

        # Determine xlrd version if installed
        if import_optional_dependency("xlrd", errors="ignore") is None:
            xlrd_version = None
        else:
            import xlrd

            xlrd_version = Version(get_version(xlrd))

        if engine is None:
            # Only determine ext if it is needed
            ext: str | None
            if xlrd_version is not None and isinstance(path_or_buffer, xlrd.Book):
                ext = "xls"
            else:
                ext = inspect_excel_format(
                    content_or_path=path_or_buffer, storage_options=storage_options
                )
                if ext is None:
                    raise ValueError(
                        "Excel file format cannot be determined, you must specify "
                        "an engine manually."
                    )

            engine = config.get_option(f"io.excel.{ext}.reader", silent=True)
            if engine == "auto":
                engine = get_default_engine(ext, mode="reader")

        assert engine is not None
        self.engine = engine
        self.storage_options = storage_options

        self._reader = self._engines[engine](
            self._io,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    def __fspath__(self):
        return self._io

    def parse(
        self,
        sheet_name: str | int | list[int] | list[str] | None = 0,
        header: int | Sequence[int] | None = 0,
        names: SequenceNotStr[Hashable] | range | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols=None,
        converters=None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        nrows: int | None = None,
        na_values=None,
        parse_dates: list | dict | bool = False,
        date_parser: Callable | lib.NoDefault = lib.no_default,
        date_format: str | dict[Hashable, str] | None = None,
        thousands: str | None = None,
        comment: str | None = None,
        skipfooter: int = 0,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwds,
    ) -> DataFrame | dict[str, DataFrame] | dict[int, DataFrame]:
        """
        Parse specified sheet(s) into a DataFrame.

        Equivalent to read_excel(ExcelFile, ...)  See the read_excel
        docstring for more info on accepted parameters.

        Returns
        -------
        DataFrame or dict of DataFrames
            DataFrame from the passed in Excel file.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
        >>> df.to_excel('myfile.xlsx')  # doctest: +SKIP
        >>> file = pd.ExcelFile('myfile.xlsx')  # doctest: +SKIP
        >>> file.parse()  # doctest: +SKIP
        """
        return self._reader.parse(
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            parse_dates=parse_dates,
            date_parser=date_parser,
            date_format=date_format,
            thousands=thousands,
            comment=comment,
            skipfooter=skipfooter,
            dtype_backend=dtype_backend,
            **kwds,
        )

    @property
    def book(self):
        return self._reader.book

    @property
    def sheet_names(self):
        return self._reader.sheet_names

    def close(self) -> None:
        """close io if necessary"""
        self._reader.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\concrete\summations.py ===
from __future__ import annotations

from sympy.calculus.singularities import is_decreasing
from sympy.calculus.accumulationbounds import AccumulationBounds
from .expr_with_intlimits import ExprWithIntLimits
from .expr_with_limits import AddWithLimits
from .gosper import gosper_sum
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Derivative, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Float, _illegal
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Wild, Symbol, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cot, csc
from sympy.functions.special.hyper import hyper
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Not
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import PolynomialError, PolificationFailed
from sympy.polys.polytools import parallel_poly_from_expr, Poly, factor
from sympy.polys.rationaltools import together
from sympy.series.limitseq import limit_seq
from sympy.series.order import O
from sympy.series.residues import residue
from sympy.sets.contains import Contains
from sympy.sets.sets import FiniteSet, Interval
from sympy.utilities.iterables import sift
import itertools


class Sum(AddWithLimits, ExprWithIntLimits):
    r"""
    Represents unevaluated summation.

    Explanation
    ===========

    ``Sum`` represents a finite or infinite series, with the first argument
    being the general form of terms in the series, and the second argument
    being ``(dummy_variable, start, end)``, with ``dummy_variable`` taking
    all integer values from ``start`` through ``end``. In accordance with
    long-standing mathematical convention, the end term is included in the
    summation.

    Finite sums
    ===========

    For finite sums (and sums with symbolic limits assumed to be finite) we
    follow the summation convention described by Karr [1], especially
    definition 3 of section 1.4. The sum:

    .. math::

        \sum_{m \leq i < n} f(i)

    has *the obvious meaning* for `m < n`, namely:

    .. math::

        \sum_{m \leq i < n} f(i) = f(m) + f(m+1) + \ldots + f(n-2) + f(n-1)

    with the upper limit value `f(n)` excluded. The sum over an empty set is
    zero if and only if `m = n`:

    .. math::

        \sum_{m \leq i < n} f(i) = 0  \quad \mathrm{for} \quad  m = n

    Finally, for all other sums over empty sets we assume the following
    definition:

    .. math::

        \sum_{m \leq i < n} f(i) = - \sum_{n \leq i < m} f(i)  \quad \mathrm{for} \quad  m > n

    It is important to note that Karr defines all sums with the upper
    limit being exclusive. This is in contrast to the usual mathematical notation,
    but does not affect the summation convention. Indeed we have:

    .. math::

        \sum_{m \leq i < n} f(i) = \sum_{i = m}^{n - 1} f(i)

    where the difference in notation is intentional to emphasize the meaning,
    with limits typeset on the top being inclusive.

    Examples
    ========

    >>> from sympy.abc import i, k, m, n, x
    >>> from sympy import Sum, factorial, oo, IndexedBase, Function
    >>> Sum(k, (k, 1, m))
    Sum(k, (k, 1, m))
    >>> Sum(k, (k, 1, m)).doit()
    m**2/2 + m/2
    >>> Sum(k**2, (k, 1, m))
    Sum(k**2, (k, 1, m))
    >>> Sum(k**2, (k, 1, m)).doit()
    m**3/3 + m**2/2 + m/6
    >>> Sum(x**k, (k, 0, oo))
    Sum(x**k, (k, 0, oo))
    >>> Sum(x**k, (k, 0, oo)).doit()
    Piecewise((1/(1 - x), Abs(x) < 1), (Sum(x**k, (k, 0, oo)), True))
    >>> Sum(x**k/factorial(k), (k, 0, oo)).doit()
    exp(x)

    Here are examples to do summation with symbolic indices.  You
    can use either Function of IndexedBase classes:

    >>> f = Function('f')
    >>> Sum(f(n), (n, 0, 3)).doit()
    f(0) + f(1) + f(2) + f(3)
    >>> Sum(f(n), (n, 0, oo)).doit()
    Sum(f(n), (n, 0, oo))
    >>> f = IndexedBase('f')
    >>> Sum(f[n]**2, (n, 0, 3)).doit()
    f[0]**2 + f[1]**2 + f[2]**2 + f[3]**2

    An example showing that the symbolic result of a summation is still
    valid for seemingly nonsensical values of the limits. Then the Karr
    convention allows us to give a perfectly valid interpretation to
    those sums by interchanging the limits according to the above rules:

    >>> S = Sum(i, (i, 1, n)).doit()
    >>> S
    n**2/2 + n/2
    >>> S.subs(n, -4)
    6
    >>> Sum(i, (i, 1, -4)).doit()
    6
    >>> Sum(-i, (i, -3, 0)).doit()
    6

    An explicit example of the Karr summation convention:

    >>> S1 = Sum(i**2, (i, m, m+n-1)).doit()
    >>> S1
    m**2*n + m*n**2 - m*n + n**3/3 - n**2/2 + n/6
    >>> S2 = Sum(i**2, (i, m+n, m-1)).doit()
    >>> S2
    -m**2*n - m*n**2 + m*n - n**3/3 + n**2/2 - n/6
    >>> S1 + S2
    0
    >>> S3 = Sum(i, (i, m, m-1)).doit()
    >>> S3
    0

    See Also
    ========

    summation
    Product, sympy.concrete.products.product

    References
    ==========

    .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
           Volume 28 Issue 2, April 1981, Pages 305-350
           https://dl.acm.org/doi/10.1145/322248.322255
    .. [2] https://en.wikipedia.org/wiki/Summation#Capital-sigma_notation
    .. [3] https://en.wikipedia.org/wiki/Empty_sum
    """

    __slots__ = ()

    limits: tuple[tuple[Symbol, Expr, Expr]]

    def __new__(cls, function, *symbols, **assumptions):
        obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
        if not hasattr(obj, 'limits'):
            return obj
        if any(len(l) != 3 or None in l for l in obj.limits):
            raise ValueError('Sum requires values for lower and upper bounds.')

        return obj

    def _eval_is_zero(self):
        # a Sum is only zero if its function is zero or if all terms
        # cancel out. This only answers whether the summand is zero; if
        # not then None is returned since we don't analyze whether all
        # terms cancel out.
        if self.function.is_zero or self.has_empty_sequence:
            return True

    def _eval_is_extended_real(self):
        if self.has_empty_sequence:
            return True
        return self.function.is_extended_real

    def _eval_is_positive(self):
        if self.has_finite_limits and self.has_reversed_limits is False:
            return self.function.is_positive

    def _eval_is_negative(self):
        if self.has_finite_limits and self.has_reversed_limits is False:
            return self.function.is_negative

    def _eval_is_finite(self):
        if self.has_finite_limits and self.function.is_finite:
            return True

    def doit(self, **hints):
        if hints.get('deep', True):
            f = self.function.doit(**hints)
        else:
            f = self.function

        # first make sure any definite limits have summation
        # variables with matching assumptions
        reps = {}
        for xab in self.limits:
            d = _dummy_with_inherited_properties_concrete(xab)
            if d:
                reps[xab[0]] = d
        if reps:
            undo = {v: k for k, v in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):  # when separate=True
                did = tuple([i.xreplace(undo) for i in did])
            elif did is not None:
                did = did.xreplace(undo)
            else:
                did = self
            return did


        if self.function.is_Matrix:
            expanded = self.expand()
            if self != expanded:
                return expanded.doit()
            return _eval_matrix_sum(self)

        for n, limit in enumerate(self.limits):
            i, a, b = limit
            dif = b - a
            if dif == -1:
                # Any summation over an empty set is zero
                return S.Zero
            if dif.is_integer and dif.is_negative:
                a, b = b + 1, a - 1
                f = -f

            newf = eval_sum(f, (i, a, b))
            if newf is None:
                if f == self.function:
                    zeta_function = self.eval_zeta_function(f, (i, a, b))
                    if zeta_function is not None:
                        return zeta_function
                    return self
                else:
                    return self.func(f, *self.limits[n:])
            f = newf

        if hints.get('deep', True):
            # eval_sum could return partially unevaluated
            # result with Piecewise.  In this case we won't
            # doit() recursively.
            if not isinstance(f, Piecewise):
                return f.doit(**hints)

        return f

    def eval_zeta_function(self, f, limits):
        """
        Check whether the function matches with the zeta function.

        If it matches, then return a `Piecewise` expression because
        zeta function does not converge unless `s > 1` and `q > 0`
        """
        i, a, b = limits
        if a.is_comparable and b.is_comparable and a > b:
            return self.eval_zeta_function(f, (i, b + S.One, a - S.One))
        if b is not S.Infinity:
            return
        w, y, z = Wild('w', exclude=[i]), Wild('y', exclude=[i]), Wild('z', exclude=[i])
        if result := f.match((w * i + y) ** (-z)):
            coeff = 1 / result[w] ** result[z]
            s = result[z]
            q = result[y] / result[w] + a
            return Piecewise((coeff * zeta(s, q),
                              And(Not(Contains(-q, S.Naturals0)), re(s) > S.One)),
                             (self, True))

    def _eval_derivative(self, x):
        """
        Differentiate wrt x as long as x is not in the free symbols of any of
        the upper or lower limits.

        Explanation
        ===========

        Sum(a*b*x, (x, 1, a)) can be differentiated wrt x or b but not `a`
        since the value of the sum is discontinuous in `a`. In a case
        involving a limit variable, the unevaluated derivative is returned.
        """

        # diff already confirmed that x is in the free symbols of self, but we
        # don't want to differentiate wrt any free symbol in the upper or lower
        # limits
        # XXX remove this test for free_symbols when the default _eval_derivative is in
        if isinstance(x, Symbol) and x not in self.free_symbols:
            return S.Zero

        # get limits and the function
        f, limits = self.function, list(self.limits)

        limit = limits.pop(-1)

        if limits:  # f is the argument to a Sum
            f = self.func(f, *limits)

        _, a, b = limit
        if x in a.free_symbols or x in b.free_symbols:
            return None
        df = Derivative(f, x, evaluate=True)
        rv = self.func(df, limit)
        return rv

    def _eval_difference_delta(self, n, step):
        k, _, upper = self.args[-1]
        new_upper = upper.subs(n, n + step)

        if len(self.args) == 2:
            f = self.args[0]
        else:
            f = self.func(*self.args[:-1])

        return Sum(f, (k, upper + 1, new_upper)).doit()

    def _eval_simplify(self, **kwargs):

        function = self.function

        if kwargs.get('deep', True):
            function = function.simplify(**kwargs)

        # split the function into adds
        terms = Add.make_args(expand(function))
        s_t = [] # Sum Terms
        o_t = [] # Other Terms

        for term in terms:
            if term.has(Sum):
                # if there is an embedded sum here
                # it is of the form x * (Sum(whatever))
                # hence we make a Mul out of it, and simplify all interior sum terms
                subterms = Mul.make_args(expand(term))
                out_terms = []
                for subterm in subterms:
                    # go through each term
                    if isinstance(subterm, Sum):
                        # if it's a sum, simplify it
                        out_terms.append(subterm._eval_simplify(**kwargs))
                    else:
                        # otherwise, add it as is
                        out_terms.append(subterm)

                # turn it back into a Mul
                s_t.append(Mul(*out_terms))
            else:
                o_t.append(term)

        # next try to combine any interior sums for further simplification
        from sympy.simplify.simplify import factor_sum, sum_combine
        result = Add(sum_combine(s_t), *o_t)

        return factor_sum(result, limits=self.limits)

    def is_convergent(self):
        r"""
        Checks for the convergence of a Sum.

        Explanation
        ===========

        We divide the study of convergence of infinite sums and products in
        two parts.

        First Part:
        One part is the question whether all the terms are well defined, i.e.,
        they are finite in a sum and also non-zero in a product. Zero
        is the analogy of (minus) infinity in products as
        :math:`e^{-\infty} = 0`.

        Second Part:
        The second part is the question of convergence after infinities,
        and zeros in products, have been omitted assuming that their number
        is finite. This means that we only consider the tail of the sum or
        product, starting from some point after which all terms are well
        defined.

        For example, in a sum of the form:

        .. math::

            \sum_{1 \leq i < \infty} \frac{1}{n^2 + an + b}

        where a and b are numbers. The routine will return true, even if there
        are infinities in the term sequence (at most two). An analogous
        product would be:

        .. math::

            \prod_{1 \leq i < \infty} e^{\frac{1}{n^2 + an + b}}

        This is how convergence is interpreted. It is concerned with what
        happens at the limit. Finding the bad terms is another independent
        matter.

        Note: It is responsibility of user to see that the sum or product
        is well defined.

        There are various tests employed to check the convergence like
        divergence test, root test, integral test, alternating series test,
        comparison tests, Dirichlet tests. It returns true if Sum is convergent
        and false if divergent and NotImplementedError if it cannot be checked.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Convergence_tests

        Examples
        ========

        >>> from sympy import factorial, S, Sum, Symbol, oo
        >>> n = Symbol('n', integer=True)
        >>> Sum(n/(n - 1), (n, 4, 7)).is_convergent()
        True
        >>> Sum(n/(2*n + 1), (n, 1, oo)).is_convergent()
        False
        >>> Sum(factorial(n)/5**n, (n, 1, oo)).is_convergent()
        False
        >>> Sum(1/n**(S(6)/5), (n, 1, oo)).is_convergent()
        True

        See Also
        ========

        Sum.is_absolutely_convergent
        sympy.concrete.products.Product.is_convergent
        """
        p, q, r = symbols('p q r', cls=Wild)

        sym = self.limits[0][0]
        lower_limit = self.limits[0][1]
        upper_limit = self.limits[0][2]
        sequence_term = self.function.simplify()

        if len(sequence_term.free_symbols) > 1:
            raise NotImplementedError("convergence checking for more than one symbol "
                                      "containing series is not handled")

        if lower_limit.is_finite and upper_limit.is_finite:
            return S.true

        # transform sym -> -sym and swap the upper_limit = S.Infinity
        # and lower_limit = - upper_limit
        if lower_limit is S.NegativeInfinity:
            if upper_limit is S.Infinity:
                return Sum(sequence_term, (sym, 0, S.Infinity)).is_convergent() and \
                        Sum(sequence_term, (sym, S.NegativeInfinity, 0)).is_convergent()
            from sympy.simplify.simplify import simplify
            sequence_term = simplify(sequence_term.xreplace({sym: -sym}))
            lower_limit = -upper_limit
            upper_limit = S.Infinity

        sym_ = Dummy(sym.name, integer=True, positive=True)
        sequence_term = sequence_term.xreplace({sym: sym_})
        sym = sym_

        interval = Interval(lower_limit, upper_limit)

        # Piecewise function handle
        if sequence_term.is_Piecewise:
            for func, cond in sequence_term.args:
                # see if it represents something going to oo
                if cond == True or cond.as_set().sup is S.Infinity:
                    s = Sum(func, (sym, lower_limit, upper_limit))
                    return s.is_convergent()
            return S.true

        ###  -------- Divergence test ----------- ###
        try:
            lim_val = limit_seq(sequence_term, sym)
            if lim_val is not None and lim_val.is_zero is False:
                return S.false
        except NotImplementedError:
            pass

        try:
            lim_val_abs = limit_seq(abs(sequence_term), sym)
            if lim_val_abs is not None and lim_val_abs.is_zero is False:
                return S.false
        except NotImplementedError:
            pass

        order = O(sequence_term, (sym, S.Infinity))

        ### --------- p-series test (1/n**p) ---------- ###
        p_series_test = order.expr.match(sym**p)
        if p_series_test is not None:
            if p_series_test[p] < -1:
                return S.true
            if p_series_test[p] >= -1:
                return S.false

        ### ------------- comparison test ------------- ###
        # 1/(n**p*log(n)**q*log(log(n))**r) comparison
        n_log_test = (order.expr.match(1/(sym**p*log(1/sym)**q*log(-log(1/sym))**r)) or
                      order.expr.match(1/(sym**p*(-log(1/sym))**q*log(-log(1/sym))**r)))
        if n_log_test is not None:
            if (n_log_test[p] > 1 or
                (n_log_test[p] == 1 and n_log_test[q] > 1) or
                (n_log_test[p] == n_log_test[q] == 1 and n_log_test[r] > 1)):
                    return S.true
            return S.false

        ### ------------- Limit comparison test -----------###
        # (1/n) comparison
        try:
            lim_comp = limit_seq(sym*sequence_term, sym)
            if lim_comp is not None and lim_comp.is_number and lim_comp > 0:
                return S.false
        except NotImplementedError:
            pass

        ### ----------- ratio test ---------------- ###
        next_sequence_term = sequence_term.xreplace({sym: sym + 1})
        from sympy.simplify.combsimp import combsimp
        from sympy.simplify.powsimp import powsimp
        ratio = combsimp(powsimp(next_sequence_term/sequence_term))
        try:
            lim_ratio = limit_seq(ratio, sym)
            if lim_ratio is not None and lim_ratio.is_number and lim_ratio is not S.NaN:
                if abs(lim_ratio) > 1:
                    return S.false
                if abs(lim_ratio) < 1:
                    return S.true
        except NotImplementedError:
            lim_ratio = None

        ### ---------- Raabe's test -------------- ###
        if lim_ratio == 1:  # ratio test inconclusive
            test_val = sym*(sequence_term/
                         sequence_term.subs(sym, sym + 1) - 1)
            test_val = test_val.gammasimp()
            try:
                lim_val = limit_seq(test_val, sym)
                if lim_val is not None and lim_val.is_number:
                    if lim_val > 1:
                        return S.true
                    if lim_val < 1:
                        return S.false
            except NotImplementedError:
                pass

        ### ----------- root test ---------------- ###
        # lim = Limit(abs(sequence_term)**(1/sym), sym, S.Infinity)
        try:
            lim_evaluated = limit_seq(abs(sequence_term)**(1/sym), sym)
            if lim_evaluated is not None and lim_evaluated.is_number:
                if lim_evaluated < 1:
                    return S.true
                if lim_evaluated > 1:
                    return S.false
        except NotImplementedError:
            pass

        ### ------------- alternating series test ----------- ###
        dict_val = sequence_term.match(S.NegativeOne**(sym + p)*q)
        if not dict_val[p].has(sym) and is_decreasing(dict_val[q], interval):
            return S.true

        ### ------------- integral test -------------- ###
        check_interval = None
        from sympy.solvers.solveset import solveset
        maxima = solveset(sequence_term.diff(sym), sym, interval)
        if not maxima:
            check_interval = interval
        elif isinstance(maxima, FiniteSet) and maxima.sup.is_number:
            check_interval = Interval(maxima.sup, interval.sup)
        if (check_interval is not None and
            (is_decreasing(sequence_term, check_interval) or
            is_decreasing(-sequence_term, check_interval))):
                integral_val = Integral(
                    sequence_term, (sym, lower_limit, upper_limit))
                try:
                    integral_val_evaluated = integral_val.doit()
                    if integral_val_evaluated.is_number:
                        return S(integral_val_evaluated.is_finite)
                except NotImplementedError:
                    pass

        ### ----- Dirichlet and bounded times convergent tests ----- ###
        # TODO
        #
        # Dirichlet_test
        # https://en.wikipedia.org/wiki/Dirichlet%27s_test
        #
        # Bounded times convergent test
        # It is based on comparison theorems for series.
        # In particular, if the general term of a series can
        # be written as a product of two terms a_n and b_n
        # and if a_n is bounded and if Sum(b_n) is absolutely
        # convergent, then the original series Sum(a_n * b_n)
        # is absolutely convergent and so convergent.
        #
        # The following code can grows like 2**n where n is the
        # number of args in order.expr
        # Possibly combined with the potentially slow checks
        # inside the loop, could make this test extremely slow
        # for larger summation expressions.

        if order.expr.is_Mul:
            args = order.expr.args
            argset = set(args)

            ### -------------- Dirichlet tests -------------- ###
            m = Dummy('m', integer=True)
            def _dirichlet_test(g_n):
                try:
                    ing_val = limit_seq(Sum(g_n, (sym, interval.inf, m)).doit(), m)
                    if ing_val is not None and ing_val.is_finite:
                        return S.true
                except NotImplementedError:
                    pass

            ### -------- bounded times convergent test ---------###
            def _bounded_convergent_test(g1_n, g2_n):
                try:
                    lim_val = limit_seq(g1_n, sym)
                    if lim_val is not None and (lim_val.is_finite or (
                        isinstance(lim_val, AccumulationBounds)
                        and (lim_val.max - lim_val.min).is_finite)):
                            if Sum(g2_n, (sym, lower_limit, upper_limit)).is_absolutely_convergent():
                                return S.true
                except NotImplementedError:
                    pass

            for n in range(1, len(argset)):
                for a_tuple in itertools.combinations(args, n):
                    b_set = argset - set(a_tuple)
                    a_n = Mul(*a_tuple)
                    b_n = Mul(*b_set)

                    if is_decreasing(a_n, interval):
                        dirich = _dirichlet_test(b_n)
                        if dirich is not None:
                            return dirich

                    bc_test = _bounded_convergent_test(a_n, b_n)
                    if bc_test is not None:
                        return bc_test

        _sym = self.limits[0][0]
        sequence_term = sequence_term.xreplace({sym: _sym})
        raise NotImplementedError("The algorithm to find the Sum convergence of %s "
                                  "is not yet implemented" % (sequence_term))

    def is_absolutely_convergent(self):
        """
        Checks for the absolute convergence of an infinite series.

        Same as checking convergence of absolute value of sequence_term of
        an infinite series.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Absolute_convergence

        Examples
        ========

        >>> from sympy import Sum, Symbol, oo
        >>> n = Symbol('n', integer=True)
        >>> Sum((-1)**n, (n, 1, oo)).is_absolutely_convergent()
        False
        >>> Sum((-1)**n/n**2, (n, 1, oo)).is_absolutely_convergent()
        True

        See Also
        ========

        Sum.is_convergent
        """
        return Sum(abs(self.function), self.limits).is_convergent()

    def euler_maclaurin(self, m=0, n=0, eps=0, eval_integral=True):
        """
        Return an Euler-Maclaurin approximation of self, where m is the
        number of leading terms to sum directly and n is the number of
        terms in the tail.

        With m = n = 0, this is simply the corresponding integral
        plus a first-order endpoint correction.

        Returns (s, e) where s is the Euler-Maclaurin approximation
        and e is the estimated error (taken to be the magnitude of
        the first omitted term in the tail):

            >>> from sympy.abc import k, a, b
            >>> from sympy import Sum
            >>> Sum(1/k, (k, 2, 5)).doit().evalf()
            1.28333333333333
            >>> s, e = Sum(1/k, (k, 2, 5)).euler_maclaurin()
            >>> s
            -log(2) + 7/20 + log(5)
            >>> from sympy import sstr
            >>> print(sstr((s.evalf(), e.evalf()), full_prec=True))
            (1.26629073187415, 0.0175000000000000)

        The endpoints may be symbolic:

            >>> s, e = Sum(1/k, (k, a, b)).euler_maclaurin()
            >>> s
            -log(a) + log(b) + 1/(2*b) + 1/(2*a)
            >>> e
            Abs(1/(12*b**2) - 1/(12*a**2))

        If the function is a polynomial of degree at most 2n+1, the
        Euler-Maclaurin formula becomes exact (and e = 0 is returned):

            >>> Sum(k, (k, 2, b)).euler_maclaurin()
            (b**2/2 + b/2 - 1, 0)
            >>> Sum(k, (k, 2, b)).doit()
            b**2/2 + b/2 - 1

        With a nonzero eps specified, the summation is ended
        as soon as the remainder term is less than the epsilon.
        """
        m = int(m)
        n = int(n)
        f = self.function
        if len(self.limits) != 1:
            raise ValueError("More than 1 limit")
        i, a, b = self.limits[0]
        if (a > b) == True:
            if a - b == 1:
                return S.Zero, S.Zero
            a, b = b + 1, a - 1
            f = -f
        s = S.Zero
        if m:
            if b.is_Integer and a.is_Integer:
                m = min(m, b - a + 1)
            if not eps or f.is_polynomial(i):
                s = Add(*[f.subs(i, a + k) for k in range(m)])
            else:
                term = f.subs(i, a)
                if term:
                    test = abs(term.evalf(3)) < eps
                    if test == True:
                        return s, abs(term)
                    elif not (test == False):
                        # a symbolic Relational class, can't go further
                        return term, S.Zero
                s = term
                for k in range(1, m):
                    term = f.subs(i, a + k)
                    if abs(term.evalf(3)) < eps and term != 0:
                        return s, abs(term)
                    s += term
            if b - a + 1 == m:
                return s, S.Zero
            a += m
        x = Dummy('x')
        I = Integral(f.subs(i, x), (x, a, b))
        if eval_integral:
            I = I.doit()
        s += I

        def fpoint(expr):
            if b is S.Infinity:
                return expr.subs(i, a), 0
            return expr.subs(i, a), expr.subs(i, b)
        fa, fb = fpoint(f)
        iterm = (fa + fb)/2
        g = f.diff(i)
        for k in range(1, n + 2):
            ga, gb = fpoint(g)
            term = bernoulli(2*k)/factorial(2*k)*(gb - ga)
            if k > n:
                break
            if eps and term:
                term_evalf = term.evalf(3)
                if term_evalf is S.NaN:
                    return S.NaN, S.NaN
                if abs(term_evalf) < eps:
                    break
            s += term
            g = g.diff(i, 2, simplify=False)
        return s + iterm, abs(term)


    def reverse_order(self, *indices):
        """
        Reverse the order of a limit in a Sum.

        Explanation
        ===========

        ``reverse_order(self, *indices)`` reverses some limits in the expression
        ``self`` which can be either a ``Sum`` or a ``Product``. The selectors in
        the argument ``indices`` specify some indices whose limits get reversed.
        These selectors are either variable names or numerical indices counted
        starting from the inner-most limit tuple.

        Examples
        ========

        >>> from sympy import Sum
        >>> from sympy.abc import x, y, a, b, c, d

        >>> Sum(x, (x, 0, 3)).reverse_order(x)
        Sum(-x, (x, 4, -1))
        >>> Sum(x*y, (x, 1, 5), (y, 0, 6)).reverse_order(x, y)
        Sum(x*y, (x, 6, 0), (y, 7, -1))
        >>> Sum(x, (x, a, b)).reverse_order(x)
        Sum(-x, (x, b + 1, a - 1))
        >>> Sum(x, (x, a, b)).reverse_order(0)
        Sum(-x, (x, b + 1, a - 1))

        While one should prefer variable names when specifying which limits
        to reverse, the index counting notation comes in handy in case there
        are several symbols with the same name.

        >>> S = Sum(x**2, (x, a, b), (x, c, d))
        >>> S
        Sum(x**2, (x, a, b), (x, c, d))
        >>> S0 = S.reverse_order(0)
        >>> S0
        Sum(-x**2, (x, b + 1, a - 1), (x, c, d))
        >>> S1 = S0.reverse_order(1)
        >>> S1
        Sum(x**2, (x, b + 1, a - 1), (x, d + 1, c - 1))

        Of course we can mix both notations:

        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))
        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index, reorder_limit,
        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder

        References
        ==========

        .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
               Volume 28 Issue 2, April 1981, Pages 305-350
               https://dl.acm.org/doi/10.1145/322248.322255
        """
        l_indices = list(indices)

        for i, indx in enumerate(l_indices):
            if not isinstance(indx, int):
                l_indices[i] = self.index(indx)

        e = 1
        limits = []
        for i, limit in enumerate(self.limits):
            l = limit
            if i in l_indices:
                e = -e
                l = (limit[0], limit[2] + 1, limit[1] - 1)
            limits.append(l)

        return Sum(e * self.function, *limits)

    def _eval_rewrite_as_Product(self, *args, **kwargs):
        from sympy.concrete.products import Product
        if self.function.is_extended_real:
            return log(Product(exp(self.function), *self.limits))


def summation(f, *symbols, **kwargs):
    r"""
    Compute the summation of f with respect to symbols.

    Explanation
    ===========

    The notation for symbols is similar to the notation used in Integral.
    summation(f, (i, a, b)) computes the sum of f with respect to i from a to b,
    i.e.,

    ::

                                    b
                                  ____
                                  \   `
        summation(f, (i, a, b)) =  )    f
                                  /___,
                                  i = a

    If it cannot compute the sum, it returns an unevaluated Sum object.
    Repeated sums can be computed by introducing additional symbols tuples::

    Examples
    ========

    >>> from sympy import summation, oo, symbols, log
    >>> i, n, m = symbols('i n m', integer=True)

    >>> summation(2*i - 1, (i, 1, n))
    n**2
    >>> summation(1/2**i, (i, 0, oo))
    2
    >>> summation(1/log(n)**n, (n, 2, oo))
    Sum(log(n)**(-n), (n, 2, oo))
    >>> summation(i, (i, 0, n), (n, 0, m))
    m**3/6 + m**2/2 + m/3

    >>> from sympy.abc import x
    >>> from sympy import factorial
    >>> summation(x**n/factorial(n), (n, 0, oo))
    exp(x)

    See Also
    ========

    Sum
    Product, sympy.concrete.products.product

    """
    return Sum(f, *symbols, **kwargs).doit(deep=False)


def telescopic_direct(L, R, n, limits):
    """
    Returns the direct summation of the terms of a telescopic sum

    Explanation
    ===========

    L is the term with lower index
    R is the term with higher index
    n difference between the indexes of L and R

    Examples
    ========

    >>> from sympy.concrete.summations import telescopic_direct
    >>> from sympy.abc import k, a, b
    >>> telescopic_direct(1/k, -1/(k+2), 2, (k, a, b))
    -1/(b + 2) - 1/(b + 1) + 1/(a + 1) + 1/a

    """
    (i, a, b) = limits
    return Add(*[L.subs(i, a + m) + R.subs(i, b - m) for m in range(n)])


def telescopic(L, R, limits):
    '''
    Tries to perform the summation using the telescopic property.

    Return None if not possible.
    '''
    (i, a, b) = limits
    if L.is_Add or R.is_Add:
        return None

    # We want to solve(L.subs(i, i + m) + R, m)
    # First we try a simple match since this does things that
    # solve doesn't do, e.g. solve(cos(k+m)-cos(k), m) gives
    # a more complicated solution than m == 0.

    k = Wild("k")
    sol = (-R).match(L.subs(i, i + k))
    s = None
    if sol and k in sol:
        s = sol[k]
        if not (s.is_Integer and L.subs(i, i + s) + R == 0):
            # invalid match or match didn't work
            s = None

    # But there are things that match doesn't do that solve
    # can do, e.g. determine that 1/(x + m) = 1/(1 - x) when m = 1

    if s is None:
        m = Dummy('m')
        try:
            from sympy.solvers.solvers import solve
            sol = solve(L.subs(i, i + m) + R, m) or []
        except NotImplementedError:
            return None
        sol = [si for si in sol if si.is_Integer and
               (L.subs(i, i + si) + R).expand().is_zero]
        if len(sol) != 1:
            return None
        s = sol[0]

    if s < 0:
        return telescopic_direct(R, L, abs(s), (i, a, b))
    elif s > 0:
        return telescopic_direct(L, R, s, (i, a, b))


def eval_sum(f, limits):
    (i, a, b) = limits
    if f.is_zero:
        return S.Zero
    if i not in f.free_symbols:
        return f*(b - a + 1)
    if a == b:
        return f.subs(i, a)
    if a.is_comparable and b.is_comparable and a > b:
        return eval_sum(f, (i, b + S.One, a - S.One))
    if isinstance(f, Piecewise):
        if not any(i in arg.args[1].free_symbols for arg in f.args):
            # Piecewise conditions do not depend on the dummy summation variable,
            # therefore we can fold:     Sum(Piecewise((e, c), ...), limits)
            #                        --> Piecewise((Sum(e, limits), c), ...)
            newargs = []
            for arg in f.args:
                newexpr = eval_sum(arg.expr, limits)
                if newexpr is None:
                    return None
                newargs.append((newexpr, arg.cond))
            return f.func(*newargs)

    if f.has(KroneckerDelta):
        from .delta import deltasummation, _has_simple_delta
        f = f.replace(
            lambda x: isinstance(x, Sum),
            lambda x: x.factor()
        )
        if _has_simple_delta(f, limits[0]):
            return deltasummation(f, limits)

    dif = b - a
    definite = dif.is_Integer
    # Doing it directly may be faster if there are very few terms.
    if definite and (dif < 100):
        return eval_sum_direct(f, (i, a, b))
    if isinstance(f, Piecewise):
        return None
    # Try to do it symbolically. Even when the number of terms is
    # known, this can save time when b-a is big.
    value = eval_sum_symbolic(f.expand(), (i, a, b))
    if value is not None:
        return value
    # Do it directly
    if definite:
        return eval_sum_direct(f, (i, a, b))


def eval_sum_direct(expr, limits):
    """
    Evaluate expression directly, but perform some simple checks first
    to possibly result in a smaller expression and faster execution.
    """
    (i, a, b) = limits

    dif = b - a
    # Linearity
    if expr.is_Mul:
        # Try factor out everything not including i
        without_i, with_i = expr.as_independent(i)
        if without_i != 1:
            s = eval_sum_direct(with_i, (i, a, b))
            if s:
                r = without_i*s
                if r is not S.NaN:
                    return r
        else:
            # Try term by term
            L, R = expr.as_two_terms()

            if not L.has(i):
                sR = eval_sum_direct(R, (i, a, b))
                if sR:
                    return L*sR

            if not R.has(i):
                sL = eval_sum_direct(L, (i, a, b))
                if sL:
                    return sL*R

    # do this whether its an Add or Mul
    # e.g. apart(1/(25*i**2 + 45*i + 14)) and
    # apart(1/((5*i + 2)*(5*i + 7))) ->
    # -1/(5*(5*i + 7)) + 1/(5*(5*i + 2))
    try:
        expr = apart(expr, i)  # see if it becomes an Add
    except PolynomialError:
        pass

    if expr.is_Add:
        # Try factor out everything not including i
        without_i, with_i = expr.as_independent(i)
        if without_i != 0:
            s = eval_sum_direct(with_i, (i, a, b))
            if s:
                r = without_i*(dif + 1) + s
                if r is not S.NaN:
                    return r
        else:
            # Try term by term
            L, R = expr.as_two_terms()
            lsum = eval_sum_direct(L, (i, a, b))
            rsum = eval_sum_direct(R, (i, a, b))

            if None not in (lsum, rsum):
                r = lsum + rsum
                if r is not S.NaN:
                    return r

    return Add(*[expr.subs(i, a + j) for j in range(dif + 1)])


def eval_sum_symbolic(f, limits):
    f_orig = f
    (i, a, b) = limits
    if not f.has(i):
        return f*(b - a + 1)

    # Linearity
    if f.is_Mul:
        # Try factor out everything not including i
        without_i, with_i = f.as_independent(i)
        if without_i != 1:
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i*s
                if r is not S.NaN:
                    return r
        else:
            # Try term by term
            L, R = f.as_two_terms()

            if not L.has(i):
                sR = eval_sum_symbolic(R, (i, a, b))
                if sR:
                    return L*sR

            if not R.has(i):
                sL = eval_sum_symbolic(L, (i, a, b))
                if sL:
                    return sL*R

    # do this whether its an Add or Mul
    # e.g. apart(1/(25*i**2 + 45*i + 14)) and
    # apart(1/((5*i + 2)*(5*i + 7))) ->
    # -1/(5*(5*i + 7)) + 1/(5*(5*i + 2))
    try:
        f = apart(f, i)
    except PolynomialError:
        pass

    if f.is_Add:
        L, R = f.as_two_terms()
        lrsum = telescopic(L, R, (i, a, b))

        if lrsum:
            return lrsum

        # Try factor out everything not including i
        without_i, with_i = f.as_independent(i)
        if without_i != 0:
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i*(b - a + 1) + s
                if r is not S.NaN:
                    return r
        else:
            # Try term by term
            lsum = eval_sum_symbolic(L, (i, a, b))
            rsum = eval_sum_symbolic(R, (i, a, b))

            if None not in (lsum, rsum):
                r = lsum + rsum
                if r is not S.NaN:
                    return r


    # Polynomial terms with Faulhaber's formula
    n = Wild('n')
    result = f.match(i**n)

    if result is not None:
        n = result[n]

        if n.is_Integer:
            if n >= 0:
                if (b is S.Infinity and a is not S.NegativeInfinity) or \
                   (a is S.NegativeInfinity and b is not S.Infinity):
                    return S.Infinity
                return ((bernoulli(n + 1, b + 1) - bernoulli(n + 1, a))/(n + 1)).expand()
            elif a.is_Integer and a >= 1:
                if n == -1:
                    return harmonic(b) - harmonic(a - 1)
                else:
                    return harmonic(b, abs(n)) - harmonic(a - 1, abs(n))

    if not (a.has(S.Infinity, S.NegativeInfinity) or
            b.has(S.Infinity, S.NegativeInfinity)):
        # Geometric terms
        c1 = Wild('c1', exclude=[i])
        c2 = Wild('c2', exclude=[i])
        c3 = Wild('c3', exclude=[i])
        wexp = Wild('wexp')

        # Here we first attempt powsimp on f for easier matching with the
        # exponential pattern, and attempt expansion on the exponent for easier
        # matching with the linear pattern.
        e = f.powsimp().match(c1 ** wexp)
        if e is not None:
            e_exp = e.pop(wexp).expand().match(c2*i + c3)
            if e_exp is not None:
                e.update(e_exp)

                p = (c1**c3).subs(e)
                q = (c1**c2).subs(e)
                r = p*(q**a - q**(b + 1))/(1 - q)
                l = p*(b - a + 1)
                return Piecewise((l, Eq(q, S.One)), (r, True))

        r = gosper_sum(f, (i, a, b))

        if isinstance(r, (Mul,Add)):
            from sympy.simplify.radsimp import denom
            from sympy.solvers.solvers import solve
            non_limit = r.free_symbols - Tuple(*limits[1:]).free_symbols
            den = denom(together(r))
            den_sym = non_limit & den.free_symbols
            args = []
            for v in ordered(den_sym):
                try:
                    s = solve(den, v)
                    m = Eq(v, s[0]) if s else S.false
                    if m != False:
                        args.append((Sum(f_orig.subs(*m.args), limits).doit(), m))
                    break
                except NotImplementedError:
                    continue

            args.append((r, True))
            return Piecewise(*args)

        if r not in (None, S.NaN):
            return r

    h = eval_sum_hyper(f_orig, (i, a, b))
    if h is not None:
        return h

    r = eval_sum_residue(f_orig, (i, a, b))
    if r is not None:
        return r

    factored = f_orig.factor()
    if factored != f_orig:
        return eval_sum_symbolic(factored, (i, a, b))


def _eval_sum_hyper(f, i, a):
    """ Returns (res, cond). Sums from a to oo. """
    if a != 0:
        return _eval_sum_hyper(f.subs(i, i + a), i, 0)

    if f.subs(i, 0) == 0:
        from sympy.simplify.simplify import simplify
        if simplify(f.subs(i, Dummy('i', integer=True, positive=True))) == 0:
            return S.Zero, True
        return _eval_sum_hyper(f.subs(i, i + 1), i, 0)

    from sympy.simplify.simplify import hypersimp
    hs = hypersimp(f, i)
    if hs is None:
        return None

    if isinstance(hs, Float):
        from sympy.simplify.simplify import nsimplify
        hs = nsimplify(hs)

    from sympy.simplify.combsimp import combsimp
    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.simplify.radsimp import fraction
    numer, denom = fraction(factor(hs))
    top, topl = numer.as_coeff_mul(i)
    bot, botl = denom.as_coeff_mul(i)
    ab = [top, bot]
    factors = [topl, botl]
    params = [[], []]
    for k in range(2):
        for fac in factors[k]:
            mul = 1
            if fac.is_Pow:
                mul = fac.exp
                fac = fac.base
                if not mul.is_Integer:
                    return None
            p = Poly(fac, i)
            if p.degree() != 1:
                return None
            m, n = p.all_coeffs()
            ab[k] *= m**mul
            params[k] += [n/m]*mul

    # Add "1" to numerator parameters, to account for implicit n! in
    # hypergeometric series.
    ap = params[0] + [1]
    bq = params[1]
    x = ab[0]/ab[1]
    h = hyper(ap, bq, x)
    f = combsimp(f)
    return f.subs(i, 0)*hyperexpand(h), h.convergence_statement


def eval_sum_hyper(f, i_a_b):
    i, a, b = i_a_b

    if f.is_hypergeometric(i) is False:
        return

    if (b - a).is_Integer:
        # We are never going to do better than doing the sum in the obvious way
        return None

    old_sum = Sum(f, (i, a, b))

    if b != S.Infinity:
        if a is S.NegativeInfinity:
            res = _eval_sum_hyper(f.subs(i, -i), i, -b)
            if res is not None:
                return Piecewise(res, (old_sum, True))
        else:
            n_illegal = lambda x: sum(x.count(_) for _ in _illegal)
            had = n_illegal(f)
            # check that no extra illegals are introduced
            res1 = _eval_sum_hyper(f, i, a)
            if res1 is None or n_illegal(res1) > had:
                return
            res2 = _eval_sum_hyper(f, i, b + 1)
            if res2 is None or n_illegal(res2) > had:
                return
            (res1, cond1), (res2, cond2) = res1, res2
            cond = And(cond1, cond2)
            if cond == False:
                return None
            return Piecewise((res1 - res2, cond), (old_sum, True))

    if a is S.NegativeInfinity:
        res1 = _eval_sum_hyper(f.subs(i, -i), i, 1)
        res2 = _eval_sum_hyper(f, i, 0)
        if res1 is None or res2 is None:
            return None
        res1, cond1 = res1
        res2, cond2 = res2
        cond = And(cond1, cond2)
        if cond == False or cond.as_set() == S.EmptySet:
            return None
        return Piecewise((res1 + res2, cond), (old_sum, True))

    # Now b == oo, a != -oo
    res = _eval_sum_hyper(f, i, a)
    if res is not None:
        r, c = res
        if c == False:
            if r.is_number:
                f = f.subs(i, Dummy('i', integer=True, positive=True) + a)
                if f.is_positive or f.is_zero:
                    return S.Infinity
                elif f.is_negative:
                    return S.NegativeInfinity
            return None
        return Piecewise(res, (old_sum, True))


def eval_sum_residue(f, i_a_b):
    r"""Compute the infinite summation with residues

    Notes
    =====

    If $f(n), g(n)$ are polynomials with $\deg(g(n)) - \deg(f(n)) \ge 2$,
    some infinite summations can be computed by the following residue
    evaluations.

    .. math::
        \sum_{n=-\infty, g(n) \ne 0}^{\infty} \frac{f(n)}{g(n)} =
        -\pi \sum_{\alpha|g(\alpha)=0}
        \text{Res}(\cot(\pi x) \frac{f(x)}{g(x)}, \alpha)

    .. math::
        \sum_{n=-\infty, g(n) \ne 0}^{\infty} (-1)^n \frac{f(n)}{g(n)} =
        -\pi \sum_{\alpha|g(\alpha)=0}
        \text{Res}(\csc(\pi x) \frac{f(x)}{g(x)}, \alpha)

    Examples
    ========

    >>> from sympy import Sum, oo, Symbol
    >>> x = Symbol('x')

    Doubly infinite series of rational functions.

    >>> Sum(1 / (x**2 + 1), (x, -oo, oo)).doit()
    pi/tanh(pi)

    Doubly infinite alternating series of rational functions.

    >>> Sum((-1)**x / (x**2 + 1), (x, -oo, oo)).doit()
    pi/sinh(pi)

    Infinite series of even rational functions.

    >>> Sum(1 / (x**2 + 1), (x, 0, oo)).doit()
    1/2 + pi/(2*tanh(pi))

    Infinite series of alternating even rational functions.

    >>> Sum((-1)**x / (x**2 + 1), (x, 0, oo)).doit()
    pi/(2*sinh(pi)) + 1/2

    This also have heuristics to transform arbitrarily shifted summand or
    arbitrarily shifted summation range to the canonical problem the
    formula can handle.

    >>> Sum(1 / (x**2 + 2*x + 2), (x, -1, oo)).doit()
    1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 4*x + 5), (x, -2, oo)).doit()
    1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 1), (x, 1, oo)).doit()
    -1/2 + pi/(2*tanh(pi))
    >>> Sum(1 / (x**2 + 1), (x, 2, oo)).doit()
    -1 + pi/(2*tanh(pi))

    References
    ==========

    .. [#] http://www.supermath.info/InfiniteSeriesandtheResidueTheorem.pdf

    .. [#] Asmar N.H., Grafakos L. (2018) Residue Theory.
           In: Complex Analysis with Applications.
           Undergraduate Texts in Mathematics. Springer, Cham.
           https://doi.org/10.1007/978-3-319-94063-2_5
    """
    i, a, b = i_a_b

    # If lower limit > upper limit: Karr Summation Convention
    if a.is_comparable and b.is_comparable and a > b:
        return eval_sum_residue(f, (i, b + S.One, a - S.One))

    def is_even_function(numer, denom):
        """Test if the rational function is an even function"""
        numer_even = all(i % 2 == 0 for (i,) in numer.monoms())
        denom_even = all(i % 2 == 0 for (i,) in denom.monoms())
        numer_odd = all(i % 2 == 1 for (i,) in numer.monoms())
        denom_odd = all(i % 2 == 1 for (i,) in denom.monoms())
        return (numer_even and denom_even) or (numer_odd and denom_odd)

    def match_rational(f, i):
        numer, denom = f.as_numer_denom()
        try:
            (numer, denom), opt = parallel_poly_from_expr((numer, denom), i)
        except (PolificationFailed, PolynomialError):
            return None
        return numer, denom

    def get_poles(denom):
        roots = denom.sqf_part().all_roots()
        roots = sift(roots, lambda x: x.is_integer)
        if None in roots:
            return None
        int_roots, nonint_roots = roots[True], roots[False]
        return int_roots, nonint_roots

    def get_shift(denom):
        n = denom.degree(i)
        a = denom.coeff_monomial(i**n)
        b = denom.coeff_monomial(i**(n-1))
        shift = - b / a / n
        return shift

    #Need a dummy symbol with no assumptions set for get_residue_factor
    z = Dummy('z')

    def get_residue_factor(numer, denom, alternating):
        residue_factor = (numer.as_expr() / denom.as_expr()).subs(i, z)
        if not alternating:
            residue_factor *= cot(S.Pi * z)
        else:
            residue_factor *= csc(S.Pi * z)
        return residue_factor

    # We don't know how to deal with symbolic constants in summand
    if f.free_symbols - {i}:
        return None

    if not (a.is_Integer or a in (S.Infinity, S.NegativeInfinity)):
        return None
    if not (b.is_Integer or b in (S.Infinity, S.NegativeInfinity)):
        return None

    # Quick exit heuristic for the sums which doesn't have infinite range
    if a != S.NegativeInfinity and b != S.Infinity:
        return None

    match = match_rational(f, i)
    if match:
        alternating = False
        numer, denom = match
    else:
        match = match_rational(f / S.NegativeOne**i, i)
        if match:
            alternating = True
            numer, denom = match
        else:
            return None

    if denom.degree(i) - numer.degree(i) < 2:
        return None

    if (a, b) == (S.NegativeInfinity, S.Infinity):
        poles = get_poles(denom)
        if poles is None:
            return None
        int_roots, nonint_roots = poles

        if int_roots:
            return None

        residue_factor = get_residue_factor(numer, denom, alternating)
        residues = [residue(residue_factor, z, root) for root in nonint_roots]
        return -S.Pi * sum(residues)

    if not (a.is_finite and b is S.Infinity):
        return None

    if not is_even_function(numer, denom):
        # Try shifting summation and check if the summand can be made
        # and even function from the origin.
        # Sum(f(n), (n, a, b)) => Sum(f(n + s), (n, a - s, b - s))
        shift = get_shift(denom)

        if not shift.is_Integer:
            return None
        if shift == 0:
            return None

        numer = numer.shift(shift)
        denom = denom.shift(shift)

        if not is_even_function(numer, denom):
            return None

        if alternating:
            f = S.NegativeOne**i * (S.NegativeOne**shift * numer.as_expr() / denom.as_expr())
        else:
            f = numer.as_expr() / denom.as_expr()
        return eval_sum_residue(f, (i, a-shift, b-shift))

    poles = get_poles(denom)
    if poles is None:
        return None
    int_roots, nonint_roots = poles

    if int_roots:
        int_roots = [int(root) for root in int_roots]
        int_roots_max = max(int_roots)
        int_roots_min = min(int_roots)
        # Integer valued poles must be next to each other
        # and also symmetric from origin (Because the function is even)
        if not len(int_roots) == int_roots_max - int_roots_min + 1:
            return None

        # Check whether the summation indices contain poles
        if a <= max(int_roots):
            return None

    residue_factor = get_residue_factor(numer, denom, alternating)
    residues = [residue(residue_factor, z, root) for root in int_roots + nonint_roots]
    full_sum = -S.Pi * sum(residues)

    if not int_roots:
        # Compute Sum(f, (i, 0, oo)) by adding a extraneous evaluation
        # at the origin.
        half_sum = (full_sum + f.xreplace({i: 0})) / 2

        # Add and subtract extraneous evaluations
        extraneous_neg = [f.xreplace({i: i0}) for i0 in range(int(a), 0)]
        extraneous_pos = [f.xreplace({i: i0}) for i0 in range(0, int(a))]
        result = half_sum + sum(extraneous_neg) - sum(extraneous_pos)

        return result

    # Compute Sum(f, (i, min(poles) + 1, oo))
    half_sum = full_sum / 2

    # Subtract extraneous evaluations
    extraneous = [f.xreplace({i: i0}) for i0 in range(max(int_roots) + 1, int(a))]
    result = half_sum - sum(extraneous)

    return result


def _eval_matrix_sum(expression):
    f = expression.function
    for limit in expression.limits:
        i, a, b = limit
        dif = b - a
        if dif.is_Integer:
            if (dif < 0) == True:
                a, b = b + 1, a - 1
                f = -f

            newf = eval_sum_direct(f, (i, a, b))
            if newf is not None:
                return newf.doit()


def _dummy_with_inherited_properties_concrete(limits):
    """
    Return a Dummy symbol that inherits as many assumptions as possible
    from the provided symbol and limits.

    If the symbol already has all True assumption shared by the limits
    then return None.
    """
    x, a, b = limits
    l = [a, b]

    assumptions_to_consider = ['extended_nonnegative', 'nonnegative',
                               'extended_nonpositive', 'nonpositive',
                               'extended_positive', 'positive',
                               'extended_negative', 'negative',
                               'integer', 'rational', 'finite',
                               'zero', 'real', 'extended_real']

    assumptions_to_keep = {}
    assumptions_to_add = {}
    for assum in assumptions_to_consider:
        assum_true = x._assumptions.get(assum, None)
        if assum_true:
            assumptions_to_keep[assum] = True
        elif all(getattr(i, 'is_' + assum) for i in l):
            assumptions_to_add[assum] = True
    if assumptions_to_add:
        assumptions_to_keep.update(assumptions_to_add)
        return Dummy('d', **assumptions_to_keep)