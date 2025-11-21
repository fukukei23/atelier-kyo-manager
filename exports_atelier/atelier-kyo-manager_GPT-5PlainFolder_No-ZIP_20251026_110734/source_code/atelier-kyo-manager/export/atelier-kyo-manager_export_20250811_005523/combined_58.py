
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\chebyshev.py ===
"""
====================================================
Chebyshev Series (:mod:`numpy.polynomial.chebyshev`)
====================================================

This module provides a number of objects (mostly functions) useful for
dealing with Chebyshev series, including a `Chebyshev` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------

.. autosummary::
   :toctree: generated/

   Chebyshev


Constants
---------

.. autosummary::
   :toctree: generated/

   chebdomain
   chebzero
   chebone
   chebx

Arithmetic
----------

.. autosummary::
   :toctree: generated/

   chebadd
   chebsub
   chebmulx
   chebmul
   chebdiv
   chebpow
   chebval
   chebval2d
   chebval3d
   chebgrid2d
   chebgrid3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   chebder
   chebint

Misc Functions
--------------

.. autosummary::
   :toctree: generated/

   chebfromroots
   chebroots
   chebvander
   chebvander2d
   chebvander3d
   chebgauss
   chebweight
   chebcompanion
   chebfit
   chebpts1
   chebpts2
   chebtrim
   chebline
   cheb2poly
   poly2cheb
   chebinterpolate

See also
--------
`numpy.polynomial`

Notes
-----
The implementations of multiplication, division, integration, and
differentiation use the algebraic identities [1]_:

.. math::
    T_n(x) = \\frac{z^n + z^{-n}}{2} \\\\
    z\\frac{dx}{dz} = \\frac{z - z^{-1}}{2}.

where

.. math:: x = \\frac{z + z^{-1}}{2}.

These identities allow a Chebyshev series to be expressed as a finite,
symmetric Laurent series.  In this module, this sort of Laurent series
is referred to as a "z-series."

References
----------
.. [1] A. T. Benjamin, et al., "Combinatorial Trigonometry with Chebyshev
  Polynomials," *Journal of Statistical Planning and Inference 14*, 2008
  (https://web.archive.org/web/20080221202153/https://www.math.hmc.edu/~benjamin/papers/CombTrig.pdf, pg. 4)

"""  # noqa: E501
import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

__all__ = [
    'chebzero', 'chebone', 'chebx', 'chebdomain', 'chebline', 'chebadd',
    'chebsub', 'chebmulx', 'chebmul', 'chebdiv', 'chebpow', 'chebval',
    'chebder', 'chebint', 'cheb2poly', 'poly2cheb', 'chebfromroots',
    'chebvander', 'chebfit', 'chebtrim', 'chebroots', 'chebpts1',
    'chebpts2', 'Chebyshev', 'chebval2d', 'chebval3d', 'chebgrid2d',
    'chebgrid3d', 'chebvander2d', 'chebvander3d', 'chebcompanion',
    'chebgauss', 'chebweight', 'chebinterpolate']

chebtrim = pu.trimcoef

#
# A collection of functions for manipulating z-series. These are private
# functions and do minimal error checking.
#

def _cseries_to_zseries(c):
    """Convert Chebyshev series to z-series.

    Convert a Chebyshev series to the equivalent z-series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from low to high

    Returns
    -------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    """
    n = c.size
    zs = np.zeros(2 * n - 1, dtype=c.dtype)
    zs[n - 1:] = c / 2
    return zs + zs[::-1]


def _zseries_to_cseries(zs):
    """Convert z-series to a Chebyshev series.

    Convert a z series to the equivalent Chebyshev series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    Returns
    -------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from  low to high.

    """
    n = (zs.size + 1) // 2
    c = zs[n - 1:].copy()
    c[1:n] *= 2
    return c


def _zseries_mul(z1, z2):
    """Multiply two z-series.

    Multiply two z-series to produce a z-series.

    Parameters
    ----------
    z1, z2 : 1-D ndarray
        The arrays must be 1-D but this is not checked.

    Returns
    -------
    product : 1-D ndarray
        The product z-series.

    Notes
    -----
    This is simply convolution. If symmetric/anti-symmetric z-series are
    denoted by S/A then the following rules apply:

    S*S, A*A -> S
    S*A, A*S -> A

    """
    return np.convolve(z1, z2)


def _zseries_div(z1, z2):
    """Divide the first z-series by the second.

    Divide `z1` by `z2` and return the quotient and remainder as z-series.
    Warning: this implementation only applies when both z1 and z2 have the
    same symmetry, which is sufficient for present purposes.

    Parameters
    ----------
    z1, z2 : 1-D ndarray
        The arrays must be 1-D and have the same symmetry, but this is not
        checked.

    Returns
    -------

    (quotient, remainder) : 1-D ndarrays
        Quotient and remainder as z-series.

    Notes
    -----
    This is not the same as polynomial division on account of the desired form
    of the remainder. If symmetric/anti-symmetric z-series are denoted by S/A
    then the following rules apply:

    S/S -> S,S
    A/A -> S,A

    The restriction to types of the same symmetry could be fixed but seems like
    unneeded generality. There is no natural form for the remainder in the case
    where there is no symmetry.

    """
    z1 = z1.copy()
    z2 = z2.copy()
    lc1 = len(z1)
    lc2 = len(z2)
    if lc2 == 1:
        z1 /= z2
        return z1, z1[:1] * 0
    elif lc1 < lc2:
        return z1[:1] * 0, z1
    else:
        dlen = lc1 - lc2
        scl = z2[0]
        z2 /= scl
        quo = np.empty(dlen + 1, dtype=z1.dtype)
        i = 0
        j = dlen
        while i < j:
            r = z1[i]
            quo[i] = z1[i]
            quo[dlen - i] = r
            tmp = r * z2
            z1[i:i + lc2] -= tmp
            z1[j:j + lc2] -= tmp
            i += 1
            j -= 1
        r = z1[i]
        quo[i] = r
        tmp = r * z2
        z1[i:i + lc2] -= tmp
        quo /= scl
        rem = z1[i + 1:i - 1 + lc2].copy()
        return quo, rem


def _zseries_der(zs):
    """Differentiate a z-series.

    The derivative is with respect to x, not z. This is achieved using the
    chain rule and the value of dx/dz given in the module notes.

    Parameters
    ----------
    zs : z-series
        The z-series to differentiate.

    Returns
    -------
    derivative : z-series
        The derivative

    Notes
    -----
    The zseries for x (ns) has been multiplied by two in order to avoid
    using floats that are incompatible with Decimal and likely other
    specialized scalar types. This scaling has been compensated by
    multiplying the value of zs by two also so that the two cancels in the
    division.

    """
    n = len(zs) // 2
    ns = np.array([-1, 0, 1], dtype=zs.dtype)
    zs *= np.arange(-n, n + 1) * 2
    d, r = _zseries_div(zs, ns)
    return d


def _zseries_int(zs):
    """Integrate a z-series.

    The integral is with respect to x, not z. This is achieved by a change
    of variable using dx/dz given in the module notes.

    Parameters
    ----------
    zs : z-series
        The z-series to integrate

    Returns
    -------
    integral : z-series
        The indefinite integral

    Notes
    -----
    The zseries for x (ns) has been multiplied by two in order to avoid
    using floats that are incompatible with Decimal and likely other
    specialized scalar types. This scaling has been compensated by
    dividing the resulting zs by two.

    """
    n = 1 + len(zs) // 2
    ns = np.array([-1, 0, 1], dtype=zs.dtype)
    zs = _zseries_mul(zs, ns)
    div = np.arange(-n, n + 1) * 2
    zs[:n] /= div[:n]
    zs[n + 1:] /= div[n + 1:]
    zs[n] = 0
    return zs

#
# Chebyshev series functions
#


def poly2cheb(pol):
    """
    Convert a polynomial to a Chebyshev series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Chebyshev series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Chebyshev
        series.

    See Also
    --------
    cheb2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> p = P.Polynomial(range(4))
    >>> p
    Polynomial([0., 1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.], symbol='x')
    >>> c = p.convert(kind=P.Chebyshev)
    >>> c
    Chebyshev([1.  , 3.25, 1.  , 0.75], domain=[-1.,  1.], window=[-1., ...
    >>> P.chebyshev.poly2cheb(range(4))
    array([1.  , 3.25, 1.  , 0.75])

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = chebadd(chebmulx(res), pol[i])
    return res


def cheb2poly(c):
    """
    Convert a Chebyshev series to a polynomial.

    Convert an array representing the coefficients of a Chebyshev series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Chebyshev series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2cheb

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> c = P.Chebyshev(range(4))
    >>> c
    Chebyshev([0., 1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.], symbol='x')
    >>> p = c.convert(kind=P.Polynomial)
    >>> p
    Polynomial([-2., -8.,  4., 12.], domain=[-1.,  1.], window=[-1.,  1.], ...
    >>> P.chebyshev.cheb2poly(range(4))
    array([-2.,  -8.,   4.,  12.])

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
            c0 = polysub(c[i - 2], c1)
            c1 = polyadd(tmp, polymulx(c1) * 2)
        return polyadd(c0, polymulx(c1))


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Chebyshev default domain.
chebdomain = np.array([-1., 1.])

# Chebyshev coefficients representing zero.
chebzero = np.array([0])

# Chebyshev coefficients representing one.
chebone = np.array([1])

# Chebyshev coefficients representing the identity x.
chebx = np.array([0, 1])


def chebline(off, scl):
    """
    Chebyshev series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Chebyshev series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebline(3,2)
    array([3, 2])
    >>> C.chebval(-3, C.chebline(3,2)) # should be -3
    -3.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])


def chebfromroots(roots):
    """
    Generate a Chebyshev series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in Chebyshev form, where the :math:`r_n` are the roots specified in
    `roots`.  If a zero has multiplicity n, then it must appear in `roots`
    n times.  For instance, if 2 is a root of multiplicity three and 3 is a
    root of multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3].
    The roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * T_1(x) + ... +  c_n * T_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in Chebyshev form.

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
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebfromroots((-1,0,1)) # x^3 - x relative to the standard basis
    array([ 0.  , -0.25,  0.  ,  0.25])
    >>> j = complex(0,1)
    >>> C.chebfromroots((-j,j)) # x^2 + 1 relative to the standard basis
    array([1.5+0.j, 0. +0.j, 0.5+0.j])

    """
    return pu._fromroots(chebline, chebmul, roots)


def chebadd(c1, c2):
    """
    Add one Chebyshev series to another.

    Returns the sum of two Chebyshev series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Chebyshev series of their sum.

    See Also
    --------
    chebsub, chebmulx, chebmul, chebdiv, chebpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Chebyshev series
    is a Chebyshev series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebadd(c1,c2)
    array([4., 4., 4.])

    """
    return pu._add(c1, c2)


def chebsub(c1, c2):
    """
    Subtract one Chebyshev series from another.

    Returns the difference of two Chebyshev series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their difference.

    See Also
    --------
    chebadd, chebmulx, chebmul, chebdiv, chebpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Chebyshev
    series is a Chebyshev series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebsub(c1,c2)
    array([-2.,  0.,  2.])
    >>> C.chebsub(c2,c1) # -C.chebsub(c1,c2)
    array([ 2.,  0., -2.])

    """
    return pu._sub(c1, c2)


def chebmulx(c):
    """Multiply a Chebyshev series by x.

    Multiply the polynomial `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    chebadd, chebsub, chebmul, chebdiv, chebpow

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> C.chebmulx([1,2,3])
    array([1. , 2.5, 1. , 1.5])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    if len(c) > 1:
        tmp = c[1:] / 2
        prd[2:] = tmp
        prd[0:-2] += tmp
    return prd


def chebmul(c1, c2):
    """
    Multiply one Chebyshev series by another.

    Returns the product of two Chebyshev series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their product.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebdiv, chebpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Chebyshev polynomial basis set.  Thus, to express
    the product as a C-series, it is typically necessary to "reproject"
    the product onto said basis set, which typically produces
    "unintuitive live" (but correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebmul(c1,c2) # multiplication requires "reprojection"
    array([  6.5,  12. ,  12. ,   4. ,   1.5])

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = _zseries_mul(z1, z2)
    ret = _zseries_to_cseries(prd)
    return pu.trimseq(ret)


def chebdiv(c1, c2):
    """
    Divide one Chebyshev series by another.

    Returns the quotient-with-remainder of two Chebyshev series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Chebyshev series coefficients representing the quotient and
        remainder.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebmul, chebpow

    Notes
    -----
    In general, the (polynomial) division of one C-series by another
    results in quotient and remainder terms that are not in the Chebyshev
    polynomial basis set.  Thus, to express these results as C-series, it
    is typically necessary to "reproject" the results onto said basis
    set, which typically produces "unintuitive" (but correct) results;
    see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebdiv(c1,c2) # quotient "intuitive," remainder not
    (array([3.]), array([-8., -4.]))
    >>> c2 = (0,1,2,3)
    >>> C.chebdiv(c2,c1) # neither "intuitive"
    (array([0., 2.]), array([-2., -4.]))

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError  # FIXME: add message with details to exception

    # note: this is more efficient than `pu._div(chebmul, c1, c2)`
    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        z1 = _cseries_to_zseries(c1)
        z2 = _cseries_to_zseries(c2)
        quo, rem = _zseries_div(z1, z2)
        quo = pu.trimseq(_zseries_to_cseries(quo))
        rem = pu.trimseq(_zseries_to_cseries(rem))
        return quo, rem


def chebpow(c, pow, maxpower=16):
    """Raise a Chebyshev series to a power.

    Returns the Chebyshev series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``T_0 + 2*T_1 + 3*T_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Chebyshev series of power.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebmul, chebdiv

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> C.chebpow([1, 2, 3, 4], 2)
    array([15.5, 22. , 16. , ..., 12.5, 12. ,  8. ])

    """
    # note: this is more efficient than `pu._pow(chebmul, c1, c2)`, as it
    # avoids converting between z and c series repeatedly

    # c is a trimmed copy
    [c] = pu.as_series([c])
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        return np.array([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        # This can be made more efficient by using powers of two
        # in the usual way.
        zs = _cseries_to_zseries(c)
        prd = zs
        for i in range(2, power + 1):
            prd = np.convolve(prd, zs)
        return _zseries_to_cseries(prd)


def chebder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Chebyshev series.

    Returns the Chebyshev series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*T_0 + 2*T_1 + 3*T_2``
    while [[1,2],[1,2]] represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) +
    2*T_0(x)*T_1(y) + 2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Chebyshev series coefficients. If c is multidimensional
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
        Chebyshev series of the derivative.

    See Also
    --------
    chebint

    Notes
    -----
    In general, the result of differentiating a C-series needs to be
    "reprojected" onto the C-series basis set. Thus, typically, the
    result of this function is "unintuitive," albeit correct; see Examples
    section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c = (1,2,3,4)
    >>> C.chebder(c)
    array([14., 12., 24.])
    >>> C.chebder(c,3)
    array([96.])
    >>> C.chebder(c,scl=-1)
    array([-14., -12., -24.])
    >>> C.chebder(c,2,-1)
    array([12.,  96.])

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
                der[j - 1] = (2 * j) * c[j]
                c[j - 2] += (j * c[j]) / (j - 2)
            if n > 1:
                der[1] = 4 * c[2]
            der[0] = c[1]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def chebint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Chebyshev series.

    Returns the Chebyshev series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``T_0 + 2*T_1 + 3*T_2`` while [[1,2],[1,2]]
    represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) + 2*T_0(x)*T_1(y) +
    2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Chebyshev series coefficients. If c is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at zero
        is the first value in the list, the value of the second integral
        at zero is the second value, etc.  If ``k == []`` (the default),
        all constants are set to zero.  If ``m == 1``, a single scalar can
        be given instead of a list.
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
        C-series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 1``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    chebder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a`- perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "reprojected" onto the C-series basis set.  Thus, typically,
    the result of this function is "unintuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c = (1,2,3)
    >>> C.chebint(c)
    array([ 0.5, -0.5,  0.5,  0.5])
    >>> C.chebint(c,3)
    array([ 0.03125   , -0.1875    ,  0.04166667, -0.05208333,  0.01041667, # may vary
        0.00625   ])
    >>> C.chebint(c, k=3)
    array([ 3.5, -0.5,  0.5,  0.5])
    >>> C.chebint(c,lbnd=-2)
    array([ 8.5, -0.5,  0.5,  0.5])
    >>> C.chebint(c,scl=-2)
    array([-1.,  1., -1., -1.])

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
                tmp[2] = c[1] / 4
            for j in range(2, n):
                tmp[j + 1] = c[j] / (2 * (j + 1))
                tmp[j - 1] -= c[j] / (2 * (j - 1))
            tmp[0] += k[i] - chebval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def chebval(x, c, tensor=True):
    """
    Evaluate a Chebyshev series at points x.

    If `c` is of length `n + 1`, this function returns the value:

    .. math:: p(x) = c_0 * T_0(x) + c_1 * T_1(x) + ... + c_n * T_n(x)

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
    chebval2d, chebgrid2d, chebval3d, chebgrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    """
    c = np.array(c, ndmin=1, copy=True)
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
        x2 = 2 * x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    return c0 + c1 * x


def chebval2d(x, y, c):
    """
    Evaluate a 2-D Chebyshev series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * T_i(x) * T_j(y)

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
        dimension greater than 2 the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional Chebyshev series at points formed
        from pairs of corresponding values from `x` and `y`.

    See Also
    --------
    chebval, chebgrid2d, chebval3d, chebgrid3d
    """
    return pu._valnd(chebval, c, x, y)


def chebgrid2d(x, y, c):
    """
    Evaluate a 2-D Chebyshev series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * T_i(a) * T_j(b),

    where the points `(a, b)` consist of all pairs formed by taking
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
    chebval, chebval2d, chebval3d, chebgrid3d
    """
    return pu._gridnd(chebval, c, x, y)


def chebval3d(x, y, z, c):
    """
    Evaluate a 3-D Chebyshev series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * T_i(x) * T_j(y) * T_k(z)

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
    chebval, chebval2d, chebgrid2d, chebgrid3d
    """
    return pu._valnd(chebval, c, x, y, z)


def chebgrid3d(x, y, z, c):
    """
    Evaluate a 3-D Chebyshev series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * T_i(a) * T_j(b) * T_k(c)

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
    chebval, chebval2d, chebgrid2d, chebval3d
    """
    return pu._gridnd(chebval, c, x, y, z)


def chebvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = T_i(x),

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Chebyshev polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    matrix ``V = chebvander(x, n)``, then ``np.dot(V, c)`` and
    ``chebval(x, c)`` are the same up to roundoff.  This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Chebyshev series of the same degree and sample points.

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
        The pseudo Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Chebyshev polynomial.  The dtype will be the same as
        the converted `x`.

    """
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=None, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    # Use forward recursion to generate the entries.
    v[0] = x * 0 + 1
    if ideg > 0:
        x2 = 2 * x
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x2 - v[i - 2]
    return np.moveaxis(v, 0, -1)


def chebvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = T_i(x) * T_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the Chebyshev polynomials.

    If ``V = chebvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``chebval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D Chebyshev
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
    chebvander, chebvander3d, chebval2d, chebval3d
    """
    return pu._vander_nd_flat((chebvander, chebvander), (x, y), deg)


def chebvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = T_i(x)*T_j(y)*T_k(z),

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the degrees of the Chebyshev polynomials.

    If ``V = chebvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and ``np.dot(V, c.flat)`` and ``chebval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Chebyshev
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
    chebvander, chebvander3d, chebval2d, chebval3d
    """
    return pu._vander_nd_flat((chebvander, chebvander, chebvander), (x, y, z), deg)


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Chebyshev series to data.

    Return the coefficients of a Chebyshev series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * T_1(x) + ... + c_n * T_n(x),

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
        Degree(s) of the fitting polynomials. If `deg` is a single integer,
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is ``len(x)*eps``, where eps is the relative precision of
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
        Chebyshev coefficients ordered from low to high. If `y` was 2-D,
        the coefficients for the data in column k  of `y` are in column
        `k`.

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
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    chebval : Evaluates a Chebyshev series.
    chebvander : Vandermonde matrix of Chebyshev series.
    chebweight : Chebyshev weight function.
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Chebyshev series `p` that
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

    Fits using Chebyshev series are usually better conditioned than fits
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
    return pu._fit(chebvander, x, y, deg, rcond, full, w)


def chebcompanion(c):
    """Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is a Chebyshev basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to high
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
    scl = np.array([1.] + [np.sqrt(.5)] * (n - 1))
    top = mat.reshape(-1)[1::n + 1]
    bot = mat.reshape(-1)[n::n + 1]
    top[0] = np.sqrt(.5)
    top[1:] = 1 / 2
    bot[...] = top
    mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * .5
    return mat


def chebroots(c):
    """
    Compute the roots of a Chebyshev series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * T_i(x).

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
    numpy.polynomial.laguerre.lagroots
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

    The Chebyshev series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as cheb
    >>> cheb.chebroots((-1, 1,-1, 1)) # T3 - T2 + T1 - T0 has real roots
    array([ -5.00000000e-01,   2.60860684e-17,   1.00000000e+00]) # may vary

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([-c[0] / c[1]])

    # rotated companion matrix reduces error
    m = chebcompanion(c)[::-1, ::-1]
    r = la.eigvals(m)
    r.sort()
    return r


def chebinterpolate(func, deg, args=()):
    """Interpolate a function at the Chebyshev points of the first kind.

    Returns the Chebyshev series that interpolates `func` at the Chebyshev
    points of the first kind in the interval [-1, 1]. The interpolating
    series tends to a minmax approximation to `func` with increasing `deg`
    if the function is continuous in the interval.

    Parameters
    ----------
    func : function
        The function to be approximated. It must be a function of a single
        variable of the form ``f(x, a, b, c...)``, where ``a, b, c...`` are
        extra arguments passed in the `args` parameter.
    deg : int
        Degree of the interpolating polynomial
    args : tuple, optional
        Extra arguments to be used in the function call. Default is no extra
        arguments.

    Returns
    -------
    coef : ndarray, shape (deg + 1,)
        Chebyshev coefficients of the interpolating series ordered from low to
        high.

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebinterpolate(lambda x: np.tanh(x) + 0.5, 8)
    array([  5.00000000e-01,   8.11675684e-01,  -9.86864911e-17,
            -5.42457905e-02,  -2.71387850e-16,   4.51658839e-03,
             2.46716228e-17,  -3.79694221e-04,  -3.26899002e-16])

    Notes
    -----
    The Chebyshev polynomials used in the interpolation are orthogonal when
    sampled at the Chebyshev points of the first kind. If it is desired to
    constrain some of the coefficients they can simply be set to the desired
    value after the interpolation, no new interpolation or fit is needed. This
    is especially useful if it is known apriori that some of coefficients are
    zero. For instance, if the function is even then the coefficients of the
    terms of odd degree in the result can be set to zero.

    """
    deg = np.asarray(deg)

    # check arguments.
    if deg.ndim > 0 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError("deg must be an int")
    if deg < 0:
        raise ValueError("expected deg >= 0")

    order = deg + 1
    xcheb = chebpts1(order)
    yfunc = func(xcheb, *args)
    m = chebvander(xcheb, deg)
    c = np.dot(m.T, yfunc)
    c[0] /= order
    c[1:] /= 0.5 * order

    return c


def chebgauss(deg):
    """
    Gauss-Chebyshev quadrature.

    Computes the sample points and weights for Gauss-Chebyshev quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-1, 1]` with
    the weight function :math:`f(x) = 1/\\sqrt{1 - x^2}`.

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
    be problematic. For Gauss-Chebyshev there are closed form solutions for
    the sample points and weights. If n = `deg`, then

    .. math:: x_i = \\cos(\\pi (2 i - 1) / (2 n))

    .. math:: w_i = \\pi / n

    """
    ideg = pu._as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    x = np.cos(np.pi * np.arange(1, 2 * ideg, 2) / (2.0 * ideg))
    w = np.ones(ideg) * (np.pi / ideg)

    return x, w


def chebweight(x):
    """
    The weight function of the Chebyshev polynomials.

    The weight function is :math:`1/\\sqrt{1 - x^2}` and the interval of
    integration is :math:`[-1, 1]`. The Chebyshev polynomials are
    orthogonal, but not normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.
    """
    w = 1. / (np.sqrt(1. + x) * np.sqrt(1. - x))
    return w


def chebpts1(npts):
    """
    Chebyshev points of the first kind.

    The Chebyshev points of the first kind are the points ``cos(x)``,
    where ``x = [pi*(k + .5)/npts for k in range(npts)]``.

    Parameters
    ----------
    npts : int
        Number of sample points desired.

    Returns
    -------
    pts : ndarray
        The Chebyshev points of the first kind.

    See Also
    --------
    chebpts2
    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 1:
        raise ValueError("npts must be >= 1")

    x = 0.5 * np.pi / _npts * np.arange(-_npts + 1, _npts + 1, 2)
    return np.sin(x)


def chebpts2(npts):
    """
    Chebyshev points of the second kind.

    The Chebyshev points of the second kind are the points ``cos(x)``,
    where ``x = [pi*k/(npts - 1) for k in range(npts)]`` sorted in ascending
    order.

    Parameters
    ----------
    npts : int
        Number of sample points desired.

    Returns
    -------
    pts : ndarray
        The Chebyshev points of the second kind.
    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 2:
        raise ValueError("npts must be >= 2")

    x = np.linspace(-np.pi, 0, _npts)
    return np.cos(x)


#
# Chebyshev series class
#

class Chebyshev(ABCPolyBase):
    """A Chebyshev series class.

    The Chebyshev class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Chebyshev coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*T_0(x) + 2*T_1(x) + 3*T_2(x)``.
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
    _add = staticmethod(chebadd)
    _sub = staticmethod(chebsub)
    _mul = staticmethod(chebmul)
    _div = staticmethod(chebdiv)
    _pow = staticmethod(chebpow)
    _val = staticmethod(chebval)
    _int = staticmethod(chebint)
    _der = staticmethod(chebder)
    _fit = staticmethod(chebfit)
    _line = staticmethod(chebline)
    _roots = staticmethod(chebroots)
    _fromroots = staticmethod(chebfromroots)

    @classmethod
    def interpolate(cls, func, deg, domain=None, args=()):
        """Interpolate a function at the Chebyshev points of the first kind.

        Returns the series that interpolates `func` at the Chebyshev points of
        the first kind scaled and shifted to the `domain`. The resulting series
        tends to a minmax approximation of `func` when the function is
        continuous in the domain.

        Parameters
        ----------
        func : function
            The function to be interpolated. It must be a function of a single
            variable of the form ``f(x, a, b, c...)``, where ``a, b, c...`` are
            extra arguments passed in the `args` parameter.
        deg : int
            Degree of the interpolating polynomial.
        domain : {None, [beg, end]}, optional
            Domain over which `func` is interpolated. The default is None, in
            which case the domain is [-1, 1].
        args : tuple, optional
            Extra arguments to be used in the function call. Default is no
            extra arguments.

        Returns
        -------
        polynomial : Chebyshev instance
            Interpolating Chebyshev instance.

        Notes
        -----
        See `numpy.polynomial.chebinterpolate` for more details.

        """
        if domain is None:
            domain = cls.domain
        xfunc = lambda x: func(pu.mapdomain(x, cls.window, domain), *args)
        coef = chebinterpolate(xfunc, deg)
        return cls(coef, domain=domain)

    # Virtual properties
    domain = np.array(chebdomain)
    window = np.array(chebdomain)
    basis_name = 'T'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\compiler.py ===
"""Compiles nodes from the parser into Python code."""

import typing as t
from contextlib import contextmanager
from functools import update_wrapper
from io import StringIO
from itertools import chain
from keyword import iskeyword as is_python_keyword

from markupsafe import escape
from markupsafe import Markup

from . import nodes
from .exceptions import TemplateAssertionError
from .idtracking import Symbols
from .idtracking import VAR_LOAD_ALIAS
from .idtracking import VAR_LOAD_PARAMETER
from .idtracking import VAR_LOAD_RESOLVE
from .idtracking import VAR_LOAD_UNDEFINED
from .nodes import EvalContext
from .optimizer import Optimizer
from .utils import _PassArg
from .utils import concat
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

F = t.TypeVar("F", bound=t.Callable[..., t.Any])

operators = {
    "eq": "==",
    "ne": "!=",
    "gt": ">",
    "gteq": ">=",
    "lt": "<",
    "lteq": "<=",
    "in": "in",
    "notin": "not in",
}


def optimizeconst(f: F) -> F:
    def new_func(
        self: "CodeGenerator", node: nodes.Expr, frame: "Frame", **kwargs: t.Any
    ) -> t.Any:
        # Only optimize if the frame is not volatile
        if self.optimizer is not None and not frame.eval_ctx.volatile:
            new_node = self.optimizer.visit(node, frame.eval_ctx)

            if new_node != node:
                return self.visit(new_node, frame)

        return f(self, node, frame, **kwargs)

    return update_wrapper(new_func, f)  # type: ignore[return-value]


def _make_binop(op: str) -> t.Callable[["CodeGenerator", nodes.BinExpr, "Frame"], None]:
    @optimizeconst
    def visitor(self: "CodeGenerator", node: nodes.BinExpr, frame: Frame) -> None:
        if (
            self.environment.sandboxed and op in self.environment.intercepted_binops  # type: ignore
        ):
            self.write(f"environment.call_binop(context, {op!r}, ")
            self.visit(node.left, frame)
            self.write(", ")
            self.visit(node.right, frame)
        else:
            self.write("(")
            self.visit(node.left, frame)
            self.write(f" {op} ")
            self.visit(node.right, frame)

        self.write(")")

    return visitor


def _make_unop(
    op: str,
) -> t.Callable[["CodeGenerator", nodes.UnaryExpr, "Frame"], None]:
    @optimizeconst
    def visitor(self: "CodeGenerator", node: nodes.UnaryExpr, frame: Frame) -> None:
        if (
            self.environment.sandboxed and op in self.environment.intercepted_unops  # type: ignore
        ):
            self.write(f"environment.call_unop(context, {op!r}, ")
            self.visit(node.node, frame)
        else:
            self.write("(" + op)
            self.visit(node.node, frame)

        self.write(")")

    return visitor


def generate(
    node: nodes.Template,
    environment: "Environment",
    name: t.Optional[str],
    filename: t.Optional[str],
    stream: t.Optional[t.TextIO] = None,
    defer_init: bool = False,
    optimized: bool = True,
) -> t.Optional[str]:
    """Generate the python source for a node tree."""
    if not isinstance(node, nodes.Template):
        raise TypeError("Can't compile non template nodes")

    generator = environment.code_generator_class(
        environment, name, filename, stream, defer_init, optimized
    )
    generator.visit(node)

    if stream is None:
        return generator.stream.getvalue()  # type: ignore

    return None


def has_safe_repr(value: t.Any) -> bool:
    """Does the node have a safe representation?"""
    if value is None or value is NotImplemented or value is Ellipsis:
        return True

    if type(value) in {bool, int, float, complex, range, str, Markup}:
        return True

    if type(value) in {tuple, list, set, frozenset}:
        return all(has_safe_repr(v) for v in value)

    if type(value) is dict:  # noqa E721
        return all(has_safe_repr(k) and has_safe_repr(v) for k, v in value.items())

    return False


def find_undeclared(
    nodes: t.Iterable[nodes.Node], names: t.Iterable[str]
) -> t.Set[str]:
    """Check if the names passed are accessed undeclared.  The return value
    is a set of all the undeclared names from the sequence of names found.
    """
    visitor = UndeclaredNameVisitor(names)
    try:
        for node in nodes:
            visitor.visit(node)
    except VisitorExit:
        pass
    return visitor.undeclared


class MacroRef:
    def __init__(self, node: t.Union[nodes.Macro, nodes.CallBlock]) -> None:
        self.node = node
        self.accesses_caller = False
        self.accesses_kwargs = False
        self.accesses_varargs = False


class Frame:
    """Holds compile time information for us."""

    def __init__(
        self,
        eval_ctx: EvalContext,
        parent: t.Optional["Frame"] = None,
        level: t.Optional[int] = None,
    ) -> None:
        self.eval_ctx = eval_ctx

        # the parent of this frame
        self.parent = parent

        if parent is None:
            self.symbols = Symbols(level=level)

            # in some dynamic inheritance situations the compiler needs to add
            # write tests around output statements.
            self.require_output_check = False

            # inside some tags we are using a buffer rather than yield statements.
            # this for example affects {% filter %} or {% macro %}.  If a frame
            # is buffered this variable points to the name of the list used as
            # buffer.
            self.buffer: t.Optional[str] = None

            # the name of the block we're in, otherwise None.
            self.block: t.Optional[str] = None

        else:
            self.symbols = Symbols(parent.symbols, level=level)
            self.require_output_check = parent.require_output_check
            self.buffer = parent.buffer
            self.block = parent.block

        # a toplevel frame is the root + soft frames such as if conditions.
        self.toplevel = False

        # the root frame is basically just the outermost frame, so no if
        # conditions.  This information is used to optimize inheritance
        # situations.
        self.rootlevel = False

        # variables set inside of loops and blocks should not affect outer frames,
        # but they still needs to be kept track of as part of the active context.
        self.loop_frame = False
        self.block_frame = False

        # track whether the frame is being used in an if-statement or conditional
        # expression as it determines which errors should be raised during runtime
        # or compile time.
        self.soft_frame = False

    def copy(self) -> "te.Self":
        """Create a copy of the current one."""
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.symbols = self.symbols.copy()
        return rv

    def inner(self, isolated: bool = False) -> "Frame":
        """Return an inner frame."""
        if isolated:
            return Frame(self.eval_ctx, level=self.symbols.level + 1)
        return Frame(self.eval_ctx, self)

    def soft(self) -> "te.Self":
        """Return a soft frame.  A soft frame may not be modified as
        standalone thing as it shares the resources with the frame it
        was created of, but it's not a rootlevel frame any longer.

        This is only used to implement if-statements and conditional
        expressions.
        """
        rv = self.copy()
        rv.rootlevel = False
        rv.soft_frame = True
        return rv

    __copy__ = copy


class VisitorExit(RuntimeError):
    """Exception used by the `UndeclaredNameVisitor` to signal a stop."""


class DependencyFinderVisitor(NodeVisitor):
    """A visitor that collects filter and test calls."""

    def __init__(self) -> None:
        self.filters: t.Set[str] = set()
        self.tests: t.Set[str] = set()

    def visit_Filter(self, node: nodes.Filter) -> None:
        self.generic_visit(node)
        self.filters.add(node.name)

    def visit_Test(self, node: nodes.Test) -> None:
        self.generic_visit(node)
        self.tests.add(node.name)

    def visit_Block(self, node: nodes.Block) -> None:
        """Stop visiting at blocks."""


class UndeclaredNameVisitor(NodeVisitor):
    """A visitor that checks if a name is accessed without being
    declared.  This is different from the frame visitor as it will
    not stop at closure frames.
    """

    def __init__(self, names: t.Iterable[str]) -> None:
        self.names = set(names)
        self.undeclared: t.Set[str] = set()

    def visit_Name(self, node: nodes.Name) -> None:
        if node.ctx == "load" and node.name in self.names:
            self.undeclared.add(node.name)
            if self.undeclared == self.names:
                raise VisitorExit()
        else:
            self.names.discard(node.name)

    def visit_Block(self, node: nodes.Block) -> None:
        """Stop visiting a blocks."""


class CompilerExit(Exception):
    """Raised if the compiler encountered a situation where it just
    doesn't make sense to further process the code.  Any block that
    raises such an exception is not further processed.
    """


class CodeGenerator(NodeVisitor):
    def __init__(
        self,
        environment: "Environment",
        name: t.Optional[str],
        filename: t.Optional[str],
        stream: t.Optional[t.TextIO] = None,
        defer_init: bool = False,
        optimized: bool = True,
    ) -> None:
        if stream is None:
            stream = StringIO()
        self.environment = environment
        self.name = name
        self.filename = filename
        self.stream = stream
        self.created_block_context = False
        self.defer_init = defer_init
        self.optimizer: t.Optional[Optimizer] = None

        if optimized:
            self.optimizer = Optimizer(environment)

        # aliases for imports
        self.import_aliases: t.Dict[str, str] = {}

        # a registry for all blocks.  Because blocks are moved out
        # into the global python scope they are registered here
        self.blocks: t.Dict[str, nodes.Block] = {}

        # the number of extends statements so far
        self.extends_so_far = 0

        # some templates have a rootlevel extends.  In this case we
        # can safely assume that we're a child template and do some
        # more optimizations.
        self.has_known_extends = False

        # the current line number
        self.code_lineno = 1

        # registry of all filters and tests (global, not block local)
        self.tests: t.Dict[str, str] = {}
        self.filters: t.Dict[str, str] = {}

        # the debug information
        self.debug_info: t.List[t.Tuple[int, int]] = []
        self._write_debug_info: t.Optional[int] = None

        # the number of new lines before the next write()
        self._new_lines = 0

        # the line number of the last written statement
        self._last_line = 0

        # true if nothing was written so far.
        self._first_write = True

        # used by the `temporary_identifier` method to get new
        # unique, temporary identifier
        self._last_identifier = 0

        # the current indentation
        self._indentation = 0

        # Tracks toplevel assignments
        self._assign_stack: t.List[t.Set[str]] = []

        # Tracks parameter definition blocks
        self._param_def_block: t.List[t.Set[str]] = []

        # Tracks the current context.
        self._context_reference_stack = ["context"]

    @property
    def optimized(self) -> bool:
        return self.optimizer is not None

    # -- Various compilation helpers

    def fail(self, msg: str, lineno: int) -> "te.NoReturn":
        """Fail with a :exc:`TemplateAssertionError`."""
        raise TemplateAssertionError(msg, lineno, self.name, self.filename)

    def temporary_identifier(self) -> str:
        """Get a new unique identifier."""
        self._last_identifier += 1
        return f"t_{self._last_identifier}"

    def buffer(self, frame: Frame) -> None:
        """Enable buffering for the frame from that point onwards."""
        frame.buffer = self.temporary_identifier()
        self.writeline(f"{frame.buffer} = []")

    def return_buffer_contents(
        self, frame: Frame, force_unescaped: bool = False
    ) -> None:
        """Return the buffer contents of the frame."""
        if not force_unescaped:
            if frame.eval_ctx.volatile:
                self.writeline("if context.eval_ctx.autoescape:")
                self.indent()
                self.writeline(f"return Markup(concat({frame.buffer}))")
                self.outdent()
                self.writeline("else:")
                self.indent()
                self.writeline(f"return concat({frame.buffer})")
                self.outdent()
                return
            elif frame.eval_ctx.autoescape:
                self.writeline(f"return Markup(concat({frame.buffer}))")
                return
        self.writeline(f"return concat({frame.buffer})")

    def indent(self) -> None:
        """Indent by one."""
        self._indentation += 1

    def outdent(self, step: int = 1) -> None:
        """Outdent by step."""
        self._indentation -= step

    def start_write(self, frame: Frame, node: t.Optional[nodes.Node] = None) -> None:
        """Yield or write into the frame buffer."""
        if frame.buffer is None:
            self.writeline("yield ", node)
        else:
            self.writeline(f"{frame.buffer}.append(", node)

    def end_write(self, frame: Frame) -> None:
        """End the writing process started by `start_write`."""
        if frame.buffer is not None:
            self.write(")")

    def simple_write(
        self, s: str, frame: Frame, node: t.Optional[nodes.Node] = None
    ) -> None:
        """Simple shortcut for start_write + write + end_write."""
        self.start_write(frame, node)
        self.write(s)
        self.end_write(frame)

    def blockvisit(self, nodes: t.Iterable[nodes.Node], frame: Frame) -> None:
        """Visit a list of nodes as block in a frame.  If the current frame
        is no buffer a dummy ``if 0: yield None`` is written automatically.
        """
        try:
            self.writeline("pass")
            for node in nodes:
                self.visit(node, frame)
        except CompilerExit:
            pass

    def write(self, x: str) -> None:
        """Write a string into the output stream."""
        if self._new_lines:
            if not self._first_write:
                self.stream.write("\n" * self._new_lines)
                self.code_lineno += self._new_lines
                if self._write_debug_info is not None:
                    self.debug_info.append((self._write_debug_info, self.code_lineno))
                    self._write_debug_info = None
            self._first_write = False
            self.stream.write("    " * self._indentation)
            self._new_lines = 0
        self.stream.write(x)

    def writeline(
        self, x: str, node: t.Optional[nodes.Node] = None, extra: int = 0
    ) -> None:
        """Combination of newline and write."""
        self.newline(node, extra)
        self.write(x)

    def newline(self, node: t.Optional[nodes.Node] = None, extra: int = 0) -> None:
        """Add one or more newlines before the next write."""
        self._new_lines = max(self._new_lines, 1 + extra)
        if node is not None and node.lineno != self._last_line:
            self._write_debug_info = node.lineno
            self._last_line = node.lineno

    def signature(
        self,
        node: t.Union[nodes.Call, nodes.Filter, nodes.Test],
        frame: Frame,
        extra_kwargs: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> None:
        """Writes a function call to the stream for the current node.
        A leading comma is added automatically.  The extra keyword
        arguments may not include python keywords otherwise a syntax
        error could occur.  The extra keyword arguments should be given
        as python dict.
        """
        # if any of the given keyword arguments is a python keyword
        # we have to make sure that no invalid call is created.
        kwarg_workaround = any(
            is_python_keyword(t.cast(str, k))
            for k in chain((x.key for x in node.kwargs), extra_kwargs or ())
        )

        for arg in node.args:
            self.write(", ")
            self.visit(arg, frame)

        if not kwarg_workaround:
            for kwarg in node.kwargs:
                self.write(", ")
                self.visit(kwarg, frame)
            if extra_kwargs is not None:
                for key, value in extra_kwargs.items():
                    self.write(f", {key}={value}")
        if node.dyn_args:
            self.write(", *")
            self.visit(node.dyn_args, frame)

        if kwarg_workaround:
            if node.dyn_kwargs is not None:
                self.write(", **dict({")
            else:
                self.write(", **{")
            for kwarg in node.kwargs:
                self.write(f"{kwarg.key!r}: ")
                self.visit(kwarg.value, frame)
                self.write(", ")
            if extra_kwargs is not None:
                for key, value in extra_kwargs.items():
                    self.write(f"{key!r}: {value}, ")
            if node.dyn_kwargs is not None:
                self.write("}, **")
                self.visit(node.dyn_kwargs, frame)
                self.write(")")
            else:
                self.write("}")

        elif node.dyn_kwargs is not None:
            self.write(", **")
            self.visit(node.dyn_kwargs, frame)

    def pull_dependencies(self, nodes: t.Iterable[nodes.Node]) -> None:
        """Find all filter and test names used in the template and
        assign them to variables in the compiled namespace. Checking
        that the names are registered with the environment is done when
        compiling the Filter and Test nodes. If the node is in an If or
        CondExpr node, the check is done at runtime instead.

        .. versionchanged:: 3.0
            Filters and tests in If and CondExpr nodes are checked at
            runtime instead of compile time.
        """
        visitor = DependencyFinderVisitor()

        for node in nodes:
            visitor.visit(node)

        for id_map, names, dependency in (
            (self.filters, visitor.filters, "filters"),
            (
                self.tests,
                visitor.tests,
                "tests",
            ),
        ):
            for name in sorted(names):
                if name not in id_map:
                    id_map[name] = self.temporary_identifier()

                # add check during runtime that dependencies used inside of executed
                # blocks are defined, as this step may be skipped during compile time
                self.writeline("try:")
                self.indent()
                self.writeline(f"{id_map[name]} = environment.{dependency}[{name!r}]")
                self.outdent()
                self.writeline("except KeyError:")
                self.indent()
                self.writeline("@internalcode")
                self.writeline(f"def {id_map[name]}(*unused):")
                self.indent()
                self.writeline(
                    f'raise TemplateRuntimeError("No {dependency[:-1]}'
                    f' named {name!r} found.")'
                )
                self.outdent()
                self.outdent()

    def enter_frame(self, frame: Frame) -> None:
        undefs = []
        for target, (action, param) in frame.symbols.loads.items():
            if action == VAR_LOAD_PARAMETER:
                pass
            elif action == VAR_LOAD_RESOLVE:
                self.writeline(f"{target} = {self.get_resolve_func()}({param!r})")
            elif action == VAR_LOAD_ALIAS:
                self.writeline(f"{target} = {param}")
            elif action == VAR_LOAD_UNDEFINED:
                undefs.append(target)
            else:
                raise NotImplementedError("unknown load instruction")
        if undefs:
            self.writeline(f"{' = '.join(undefs)} = missing")

    def leave_frame(self, frame: Frame, with_python_scope: bool = False) -> None:
        if not with_python_scope:
            undefs = []
            for target in frame.symbols.loads:
                undefs.append(target)
            if undefs:
                self.writeline(f"{' = '.join(undefs)} = missing")

    def choose_async(self, async_value: str = "async ", sync_value: str = "") -> str:
        return async_value if self.environment.is_async else sync_value

    def func(self, name: str) -> str:
        return f"{self.choose_async()}def {name}"

    def macro_body(
        self, node: t.Union[nodes.Macro, nodes.CallBlock], frame: Frame
    ) -> t.Tuple[Frame, MacroRef]:
        """Dump the function def of a macro or call block."""
        frame = frame.inner()
        frame.symbols.analyze_node(node)
        macro_ref = MacroRef(node)

        explicit_caller = None
        skip_special_params = set()
        args = []

        for idx, arg in enumerate(node.args):
            if arg.name == "caller":
                explicit_caller = idx
            if arg.name in ("kwargs", "varargs"):
                skip_special_params.add(arg.name)
            args.append(frame.symbols.ref(arg.name))

        undeclared = find_undeclared(node.body, ("caller", "kwargs", "varargs"))

        if "caller" in undeclared:
            # In older Jinja versions there was a bug that allowed caller
            # to retain the special behavior even if it was mentioned in
            # the argument list.  However thankfully this was only really
            # working if it was the last argument.  So we are explicitly
            # checking this now and error out if it is anywhere else in
            # the argument list.
            if explicit_caller is not None:
                try:
                    node.defaults[explicit_caller - len(node.args)]
                except IndexError:
                    self.fail(
                        "When defining macros or call blocks the "
                        'special "caller" argument must be omitted '
                        "or be given a default.",
                        node.lineno,
                    )
            else:
                args.append(frame.symbols.declare_parameter("caller"))
            macro_ref.accesses_caller = True
        if "kwargs" in undeclared and "kwargs" not in skip_special_params:
            args.append(frame.symbols.declare_parameter("kwargs"))
            macro_ref.accesses_kwargs = True
        if "varargs" in undeclared and "varargs" not in skip_special_params:
            args.append(frame.symbols.declare_parameter("varargs"))
            macro_ref.accesses_varargs = True

        # macros are delayed, they never require output checks
        frame.require_output_check = False
        frame.symbols.analyze_node(node)
        self.writeline(f"{self.func('macro')}({', '.join(args)}):", node)
        self.indent()

        self.buffer(frame)
        self.enter_frame(frame)

        self.push_parameter_definitions(frame)
        for idx, arg in enumerate(node.args):
            ref = frame.symbols.ref(arg.name)
            self.writeline(f"if {ref} is missing:")
            self.indent()
            try:
                default = node.defaults[idx - len(node.args)]
            except IndexError:
                self.writeline(
                    f'{ref} = undefined("parameter {arg.name!r} was not provided",'
                    f" name={arg.name!r})"
                )
            else:
                self.writeline(f"{ref} = ")
                self.visit(default, frame)
            self.mark_parameter_stored(ref)
            self.outdent()
        self.pop_parameter_definitions()

        self.blockvisit(node.body, frame)
        self.return_buffer_contents(frame, force_unescaped=True)
        self.leave_frame(frame, with_python_scope=True)
        self.outdent()

        return frame, macro_ref

    def macro_def(self, macro_ref: MacroRef, frame: Frame) -> None:
        """Dump the macro definition for the def created by macro_body."""
        arg_tuple = ", ".join(repr(x.name) for x in macro_ref.node.args)
        name = getattr(macro_ref.node, "name", None)
        if len(macro_ref.node.args) == 1:
            arg_tuple += ","
        self.write(
            f"Macro(environment, macro, {name!r}, ({arg_tuple}),"
            f" {macro_ref.accesses_kwargs!r}, {macro_ref.accesses_varargs!r},"
            f" {macro_ref.accesses_caller!r}, context.eval_ctx.autoescape)"
        )

    def position(self, node: nodes.Node) -> str:
        """Return a human readable position for the node."""
        rv = f"line {node.lineno}"
        if self.name is not None:
            rv = f"{rv} in {self.name!r}"
        return rv

    def dump_local_context(self, frame: Frame) -> str:
        items_kv = ", ".join(
            f"{name!r}: {target}"
            for name, target in frame.symbols.dump_stores().items()
        )
        return f"{{{items_kv}}}"

    def write_commons(self) -> None:
        """Writes a common preamble that is used by root and block functions.
        Primarily this sets up common local helpers and enforces a generator
        through a dead branch.
        """
        self.writeline("resolve = context.resolve_or_missing")
        self.writeline("undefined = environment.undefined")
        self.writeline("concat = environment.concat")
        # always use the standard Undefined class for the implicit else of
        # conditional expressions
        self.writeline("cond_expr_undefined = Undefined")
        self.writeline("if 0: yield None")

    def push_parameter_definitions(self, frame: Frame) -> None:
        """Pushes all parameter targets from the given frame into a local
        stack that permits tracking of yet to be assigned parameters.  In
        particular this enables the optimization from `visit_Name` to skip
        undefined expressions for parameters in macros as macros can reference
        otherwise unbound parameters.
        """
        self._param_def_block.append(frame.symbols.dump_param_targets())

    def pop_parameter_definitions(self) -> None:
        """Pops the current parameter definitions set."""
        self._param_def_block.pop()

    def mark_parameter_stored(self, target: str) -> None:
        """Marks a parameter in the current parameter definitions as stored.
        This will skip the enforced undefined checks.
        """
        if self._param_def_block:
            self._param_def_block[-1].discard(target)

    def push_context_reference(self, target: str) -> None:
        self._context_reference_stack.append(target)

    def pop_context_reference(self) -> None:
        self._context_reference_stack.pop()

    def get_context_ref(self) -> str:
        return self._context_reference_stack[-1]

    def get_resolve_func(self) -> str:
        target = self._context_reference_stack[-1]
        if target == "context":
            return "resolve"
        return f"{target}.resolve"

    def derive_context(self, frame: Frame) -> str:
        return f"{self.get_context_ref()}.derived({self.dump_local_context(frame)})"

    def parameter_is_undeclared(self, target: str) -> bool:
        """Checks if a given target is an undeclared parameter."""
        if not self._param_def_block:
            return False
        return target in self._param_def_block[-1]

    def push_assign_tracking(self) -> None:
        """Pushes a new layer for assignment tracking."""
        self._assign_stack.append(set())

    def pop_assign_tracking(self, frame: Frame) -> None:
        """Pops the topmost level for assignment tracking and updates the
        context variables if necessary.
        """
        vars = self._assign_stack.pop()
        if (
            not frame.block_frame
            and not frame.loop_frame
            and not frame.toplevel
            or not vars
        ):
            return
        public_names = [x for x in vars if x[:1] != "_"]
        if len(vars) == 1:
            name = next(iter(vars))
            ref = frame.symbols.ref(name)
            if frame.loop_frame:
                self.writeline(f"_loop_vars[{name!r}] = {ref}")
                return
            if frame.block_frame:
                self.writeline(f"_block_vars[{name!r}] = {ref}")
                return
            self.writeline(f"context.vars[{name!r}] = {ref}")
        else:
            if frame.loop_frame:
                self.writeline("_loop_vars.update({")
            elif frame.block_frame:
                self.writeline("_block_vars.update({")
            else:
                self.writeline("context.vars.update({")
            for idx, name in enumerate(sorted(vars)):
                if idx:
                    self.write(", ")
                ref = frame.symbols.ref(name)
                self.write(f"{name!r}: {ref}")
            self.write("})")
        if not frame.block_frame and not frame.loop_frame and public_names:
            if len(public_names) == 1:
                self.writeline(f"context.exported_vars.add({public_names[0]!r})")
            else:
                names_str = ", ".join(map(repr, sorted(public_names)))
                self.writeline(f"context.exported_vars.update(({names_str}))")

    # -- Statement Visitors

    def visit_Template(
        self, node: nodes.Template, frame: t.Optional[Frame] = None
    ) -> None:
        assert frame is None, "no root frame allowed"
        eval_ctx = EvalContext(self.environment, self.name)

        from .runtime import async_exported
        from .runtime import exported

        if self.environment.is_async:
            exported_names = sorted(exported + async_exported)
        else:
            exported_names = sorted(exported)

        self.writeline("from jinja2.runtime import " + ", ".join(exported_names))

        # if we want a deferred initialization we cannot move the
        # environment into a local name
        envenv = "" if self.defer_init else ", environment=environment"

        # do we have an extends tag at all?  If not, we can save some
        # overhead by just not processing any inheritance code.
        have_extends = node.find(nodes.Extends) is not None

        # find all blocks
        for block in node.find_all(nodes.Block):
            if block.name in self.blocks:
                self.fail(f"block {block.name!r} defined twice", block.lineno)
            self.blocks[block.name] = block

        # find all imports and import them
        for import_ in node.find_all(nodes.ImportedName):
            if import_.importname not in self.import_aliases:
                imp = import_.importname
                self.import_aliases[imp] = alias = self.temporary_identifier()
                if "." in imp:
                    module, obj = imp.rsplit(".", 1)
                    self.writeline(f"from {module} import {obj} as {alias}")
                else:
                    self.writeline(f"import {imp} as {alias}")

        # add the load name
        self.writeline(f"name = {self.name!r}")

        # generate the root render function.
        self.writeline(
            f"{self.func('root')}(context, missing=missing{envenv}):", extra=1
        )
        self.indent()
        self.write_commons()

        # process the root
        frame = Frame(eval_ctx)
        if "self" in find_undeclared(node.body, ("self",)):
            ref = frame.symbols.declare_parameter("self")
            self.writeline(f"{ref} = TemplateReference(context)")
        frame.symbols.analyze_node(node)
        frame.toplevel = frame.rootlevel = True
        frame.require_output_check = have_extends and not self.has_known_extends
        if have_extends:
            self.writeline("parent_template = None")
        self.enter_frame(frame)
        self.pull_dependencies(node.body)
        self.blockvisit(node.body, frame)
        self.leave_frame(frame, with_python_scope=True)
        self.outdent()

        # make sure that the parent root is called.
        if have_extends:
            if not self.has_known_extends:
                self.indent()
                self.writeline("if parent_template is not None:")
            self.indent()
            if not self.environment.is_async:
                self.writeline("yield from parent_template.root_render_func(context)")
            else:
                self.writeline("agen = parent_template.root_render_func(context)")
                self.writeline("try:")
                self.indent()
                self.writeline("async for event in agen:")
                self.indent()
                self.writeline("yield event")
                self.outdent()
                self.outdent()
                self.writeline("finally: await agen.aclose()")
            self.outdent(1 + (not self.has_known_extends))

        # at this point we now have the blocks collected and can visit them too.
        for name, block in self.blocks.items():
            self.writeline(
                f"{self.func('block_' + name)}(context, missing=missing{envenv}):",
                block,
                1,
            )
            self.indent()
            self.write_commons()
            # It's important that we do not make this frame a child of the
            # toplevel template.  This would cause a variety of
            # interesting issues with identifier tracking.
            block_frame = Frame(eval_ctx)
            block_frame.block_frame = True
            undeclared = find_undeclared(block.body, ("self", "super"))
            if "self" in undeclared:
                ref = block_frame.symbols.declare_parameter("self")
                self.writeline(f"{ref} = TemplateReference(context)")
            if "super" in undeclared:
                ref = block_frame.symbols.declare_parameter("super")
                self.writeline(f"{ref} = context.super({name!r}, block_{name})")
            block_frame.symbols.analyze_node(block)
            block_frame.block = name
            self.writeline("_block_vars = {}")
            self.enter_frame(block_frame)
            self.pull_dependencies(block.body)
            self.blockvisit(block.body, block_frame)
            self.leave_frame(block_frame, with_python_scope=True)
            self.outdent()

        blocks_kv_str = ", ".join(f"{x!r}: block_{x}" for x in self.blocks)
        self.writeline(f"blocks = {{{blocks_kv_str}}}", extra=1)
        debug_kv_str = "&".join(f"{k}={v}" for k, v in self.debug_info)
        self.writeline(f"debug_info = {debug_kv_str!r}")

    def visit_Block(self, node: nodes.Block, frame: Frame) -> None:
        """Call a block and register it for the template."""
        level = 0
        if frame.toplevel:
            # if we know that we are a child template, there is no need to
            # check if we are one
            if self.has_known_extends:
                return
            if self.extends_so_far > 0:
                self.writeline("if parent_template is None:")
                self.indent()
                level += 1

        if node.scoped:
            context = self.derive_context(frame)
        else:
            context = self.get_context_ref()

        if node.required:
            self.writeline(f"if len(context.blocks[{node.name!r}]) <= 1:", node)
            self.indent()
            self.writeline(
                f'raise TemplateRuntimeError("Required block {node.name!r} not found")',
                node,
            )
            self.outdent()

        if not self.environment.is_async and frame.buffer is None:
            self.writeline(
                f"yield from context.blocks[{node.name!r}][0]({context})", node
            )
        else:
            self.writeline(f"gen = context.blocks[{node.name!r}][0]({context})")
            self.writeline("try:")
            self.indent()
            self.writeline(
                f"{self.choose_async()}for event in gen:",
                node,
            )
            self.indent()
            self.simple_write("event", frame)
            self.outdent()
            self.outdent()
            self.writeline(
                f"finally: {self.choose_async('await gen.aclose()', 'gen.close()')}"
            )

        self.outdent(level)

    def visit_Extends(self, node: nodes.Extends, frame: Frame) -> None:
        """Calls the extender."""
        if not frame.toplevel:
            self.fail("cannot use extend from a non top-level scope", node.lineno)

        # if the number of extends statements in general is zero so
        # far, we don't have to add a check if something extended
        # the template before this one.
        if self.extends_so_far > 0:
            # if we have a known extends we just add a template runtime
            # error into the generated code.  We could catch that at compile
            # time too, but i welcome it not to confuse users by throwing the
            # same error at different times just "because we can".
            if not self.has_known_extends:
                self.writeline("if parent_template is not None:")
                self.indent()
            self.writeline('raise TemplateRuntimeError("extended multiple times")')

            # if we have a known extends already we don't need that code here
            # as we know that the template execution will end here.
            if self.has_known_extends:
                raise CompilerExit()
            else:
                self.outdent()

        self.writeline("parent_template = environment.get_template(", node)
        self.visit(node.template, frame)
        self.write(f", {self.name!r})")
        self.writeline("for name, parent_block in parent_template.blocks.items():")
        self.indent()
        self.writeline("context.blocks.setdefault(name, []).append(parent_block)")
        self.outdent()

        # if this extends statement was in the root level we can take
        # advantage of that information and simplify the generated code
        # in the top level from this point onwards
        if frame.rootlevel:
            self.has_known_extends = True

        # and now we have one more
        self.extends_so_far += 1

    def visit_Include(self, node: nodes.Include, frame: Frame) -> None:
        """Handles includes."""
        if node.ignore_missing:
            self.writeline("try:")
            self.indent()

        func_name = "get_or_select_template"
        if isinstance(node.template, nodes.Const):
            if isinstance(node.template.value, str):
                func_name = "get_template"
            elif isinstance(node.template.value, (tuple, list)):
                func_name = "select_template"
        elif isinstance(node.template, (nodes.Tuple, nodes.List)):
            func_name = "select_template"

        self.writeline(f"template = environment.{func_name}(", node)
        self.visit(node.template, frame)
        self.write(f", {self.name!r})")
        if node.ignore_missing:
            self.outdent()
            self.writeline("except TemplateNotFound:")
            self.indent()
            self.writeline("pass")
            self.outdent()
            self.writeline("else:")
            self.indent()

        def loop_body() -> None:
            self.indent()
            self.simple_write("event", frame)
            self.outdent()

        if node.with_context:
            self.writeline(
                f"gen = template.root_render_func("
                "template.new_context(context.get_all(), True,"
                f" {self.dump_local_context(frame)}))"
            )
            self.writeline("try:")
            self.indent()
            self.writeline(f"{self.choose_async()}for event in gen:")
            loop_body()
            self.outdent()
            self.writeline(
                f"finally: {self.choose_async('await gen.aclose()', 'gen.close()')}"
            )
        elif self.environment.is_async:
            self.writeline(
                "for event in (await template._get_default_module_async())"
                "._body_stream:"
            )
            loop_body()
        else:
            self.writeline("yield from template._get_default_module()._body_stream")

        if node.ignore_missing:
            self.outdent()

    def _import_common(
        self, node: t.Union[nodes.Import, nodes.FromImport], frame: Frame
    ) -> None:
        self.write(f"{self.choose_async('await ')}environment.get_template(")
        self.visit(node.template, frame)
        self.write(f", {self.name!r}).")

        if node.with_context:
            f_name = f"make_module{self.choose_async('_async')}"
            self.write(
                f"{f_name}(context.get_all(), True, {self.dump_local_context(frame)})"
            )
        else:
            self.write(f"_get_default_module{self.choose_async('_async')}(context)")

    def visit_Import(self, node: nodes.Import, frame: Frame) -> None:
        """Visit regular imports."""
        self.writeline(f"{frame.symbols.ref(node.target)} = ", node)
        if frame.toplevel:
            self.write(f"context.vars[{node.target!r}] = ")

        self._import_common(node, frame)

        if frame.toplevel and not node.target.startswith("_"):
            self.writeline(f"context.exported_vars.discard({node.target!r})")

    def visit_FromImport(self, node: nodes.FromImport, frame: Frame) -> None:
        """Visit named imports."""
        self.newline(node)
        self.write("included_template = ")
        self._import_common(node, frame)
        var_names = []
        discarded_names = []
        for name in node.names:
            if isinstance(name, tuple):
                name, alias = name
            else:
                alias = name
            self.writeline(
                f"{frame.symbols.ref(alias)} ="
                f" getattr(included_template, {name!r}, missing)"
            )
            self.writeline(f"if {frame.symbols.ref(alias)} is missing:")
            self.indent()
            # The position will contain the template name, and will be formatted
            # into a string that will be compiled into an f-string. Curly braces
            # in the name must be replaced with escapes so that they will not be
            # executed as part of the f-string.
            position = self.position(node).replace("{", "{{").replace("}", "}}")
            message = (
                "the template {included_template.__name__!r}"
                f" (imported on {position})"
                f" does not export the requested name {name!r}"
            )
            self.writeline(
                f"{frame.symbols.ref(alias)} = undefined(f{message!r}, name={name!r})"
            )
            self.outdent()
            if frame.toplevel:
                var_names.append(alias)
                if not alias.startswith("_"):
                    discarded_names.append(alias)

        if var_names:
            if len(var_names) == 1:
                name = var_names[0]
                self.writeline(f"context.vars[{name!r}] = {frame.symbols.ref(name)}")
            else:
                names_kv = ", ".join(
                    f"{name!r}: {frame.symbols.ref(name)}" for name in var_names
                )
                self.writeline(f"context.vars.update({{{names_kv}}})")
        if discarded_names:
            if len(discarded_names) == 1:
                self.writeline(f"context.exported_vars.discard({discarded_names[0]!r})")
            else:
                names_str = ", ".join(map(repr, discarded_names))
                self.writeline(
                    f"context.exported_vars.difference_update(({names_str}))"
                )

    def visit_For(self, node: nodes.For, frame: Frame) -> None:
        loop_frame = frame.inner()
        loop_frame.loop_frame = True
        test_frame = frame.inner()
        else_frame = frame.inner()

        # try to figure out if we have an extended loop.  An extended loop
        # is necessary if the loop is in recursive mode if the special loop
        # variable is accessed in the body if the body is a scoped block.
        extended_loop = (
            node.recursive
            or "loop"
            in find_undeclared(node.iter_child_nodes(only=("body",)), ("loop",))
            or any(block.scoped for block in node.find_all(nodes.Block))
        )

        loop_ref = None
        if extended_loop:
            loop_ref = loop_frame.symbols.declare_parameter("loop")

        loop_frame.symbols.analyze_node(node, for_branch="body")
        if node.else_:
            else_frame.symbols.analyze_node(node, for_branch="else")

        if node.test:
            loop_filter_func = self.temporary_identifier()
            test_frame.symbols.analyze_node(node, for_branch="test")
            self.writeline(f"{self.func(loop_filter_func)}(fiter):", node.test)
            self.indent()
            self.enter_frame(test_frame)
            self.writeline(self.choose_async("async for ", "for "))
            self.visit(node.target, loop_frame)
            self.write(" in ")
            self.write(self.choose_async("auto_aiter(fiter)", "fiter"))
            self.write(":")
            self.indent()
            self.writeline("if ", node.test)
            self.visit(node.test, test_frame)
            self.write(":")
            self.indent()
            self.writeline("yield ")
            self.visit(node.target, loop_frame)
            self.outdent(3)
            self.leave_frame(test_frame, with_python_scope=True)

        # if we don't have an recursive loop we have to find the shadowed
        # variables at that point.  Because loops can be nested but the loop
        # variable is a special one we have to enforce aliasing for it.
        if node.recursive:
            self.writeline(
                f"{self.func('loop')}(reciter, loop_render_func, depth=0):", node
            )
            self.indent()
            self.buffer(loop_frame)

            # Use the same buffer for the else frame
            else_frame.buffer = loop_frame.buffer

        # make sure the loop variable is a special one and raise a template
        # assertion error if a loop tries to write to loop
        if extended_loop:
            self.writeline(f"{loop_ref} = missing")

        for name in node.find_all(nodes.Name):
            if name.ctx == "store" and name.name == "loop":
                self.fail(
                    "Can't assign to special loop variable in for-loop target",
                    name.lineno,
                )

        if node.else_:
            iteration_indicator = self.temporary_identifier()
            self.writeline(f"{iteration_indicator} = 1")

        self.writeline(self.choose_async("async for ", "for "), node)
        self.visit(node.target, loop_frame)
        if extended_loop:
            self.write(f", {loop_ref} in {self.choose_async('Async')}LoopContext(")
        else:
            self.write(" in ")

        if node.test:
            self.write(f"{loop_filter_func}(")
        if node.recursive:
            self.write("reciter")
        else:
            if self.environment.is_async and not extended_loop:
                self.write("auto_aiter(")
            self.visit(node.iter, frame)
            if self.environment.is_async and not extended_loop:
                self.write(")")
        if node.test:
            self.write(")")

        if node.recursive:
            self.write(", undefined, loop_render_func, depth):")
        else:
            self.write(", undefined):" if extended_loop else ":")

        self.indent()
        self.enter_frame(loop_frame)

        self.writeline("_loop_vars = {}")
        self.blockvisit(node.body, loop_frame)
        if node.else_:
            self.writeline(f"{iteration_indicator} = 0")
        self.outdent()
        self.leave_frame(
            loop_frame, with_python_scope=node.recursive and not node.else_
        )

        if node.else_:
            self.writeline(f"if {iteration_indicator}:")
            self.indent()
            self.enter_frame(else_frame)
            self.blockvisit(node.else_, else_frame)
            self.leave_frame(else_frame)
            self.outdent()

        # if the node was recursive we have to return the buffer contents
        # and start the iteration code
        if node.recursive:
            self.return_buffer_contents(loop_frame)
            self.outdent()
            self.start_write(frame, node)
            self.write(f"{self.choose_async('await ')}loop(")
            if self.environment.is_async:
                self.write("auto_aiter(")
            self.visit(node.iter, frame)
            if self.environment.is_async:
                self.write(")")
            self.write(", loop)")
            self.end_write(frame)

        # at the end of the iteration, clear any assignments made in the
        # loop from the top level
        if self._assign_stack:
            self._assign_stack[-1].difference_update(loop_frame.symbols.stores)

    def visit_If(self, node: nodes.If, frame: Frame) -> None:
        if_frame = frame.soft()
        self.writeline("if ", node)
        self.visit(node.test, if_frame)
        self.write(":")
        self.indent()
        self.blockvisit(node.body, if_frame)
        self.outdent()
        for elif_ in node.elif_:
            self.writeline("elif ", elif_)
            self.visit(elif_.test, if_frame)
            self.write(":")
            self.indent()
            self.blockvisit(elif_.body, if_frame)
            self.outdent()
        if node.else_:
            self.writeline("else:")
            self.indent()
            self.blockvisit(node.else_, if_frame)
            self.outdent()

    def visit_Macro(self, node: nodes.Macro, frame: Frame) -> None:
        macro_frame, macro_ref = self.macro_body(node, frame)
        self.newline()
        if frame.toplevel:
            if not node.name.startswith("_"):
                self.write(f"context.exported_vars.add({node.name!r})")
            self.writeline(f"context.vars[{node.name!r}] = ")
        self.write(f"{frame.symbols.ref(node.name)} = ")
        self.macro_def(macro_ref, macro_frame)

    def visit_CallBlock(self, node: nodes.CallBlock, frame: Frame) -> None:
        call_frame, macro_ref = self.macro_body(node, frame)
        self.writeline("caller = ")
        self.macro_def(macro_ref, call_frame)
        self.start_write(frame, node)
        self.visit_Call(node.call, frame, forward_caller=True)
        self.end_write(frame)

    def visit_FilterBlock(self, node: nodes.FilterBlock, frame: Frame) -> None:
        filter_frame = frame.inner()
        filter_frame.symbols.analyze_node(node)
        self.enter_frame(filter_frame)
        self.buffer(filter_frame)
        self.blockvisit(node.body, filter_frame)
        self.start_write(frame, node)
        self.visit_Filter(node.filter, filter_frame)
        self.end_write(frame)
        self.leave_frame(filter_frame)

    def visit_With(self, node: nodes.With, frame: Frame) -> None:
        with_frame = frame.inner()
        with_frame.symbols.analyze_node(node)
        self.enter_frame(with_frame)
        for target, expr in zip(node.targets, node.values):
            self.newline()
            self.visit(target, with_frame)
            self.write(" = ")
            self.visit(expr, frame)
        self.blockvisit(node.body, with_frame)
        self.leave_frame(with_frame)

    def visit_ExprStmt(self, node: nodes.ExprStmt, frame: Frame) -> None:
        self.newline(node)
        self.visit(node.node, frame)

    class _FinalizeInfo(t.NamedTuple):
        const: t.Optional[t.Callable[..., str]]
        src: t.Optional[str]

    @staticmethod
    def _default_finalize(value: t.Any) -> t.Any:
        """The default finalize function if the environment isn't
        configured with one. Or, if the environment has one, this is
        called on that function's output for constants.
        """
        return str(value)

    _finalize: t.Optional[_FinalizeInfo] = None

    def _make_finalize(self) -> _FinalizeInfo:
        """Build the finalize function to be used on constants and at
        runtime. Cached so it's only created once for all output nodes.

        Returns a ``namedtuple`` with the following attributes:

        ``const``
            A function to finalize constant data at compile time.

        ``src``
            Source code to output around nodes to be evaluated at
            runtime.
        """
        if self._finalize is not None:
            return self._finalize

        finalize: t.Optional[t.Callable[..., t.Any]]
        finalize = default = self._default_finalize
        src = None

        if self.environment.finalize:
            src = "environment.finalize("
            env_finalize = self.environment.finalize
            pass_arg = {
                _PassArg.context: "context",
                _PassArg.eval_context: "context.eval_ctx",
                _PassArg.environment: "environment",
            }.get(
                _PassArg.from_obj(env_finalize)  # type: ignore
            )
            finalize = None

            if pass_arg is None:

                def finalize(value: t.Any) -> t.Any:  # noqa: F811
                    return default(env_finalize(value))

            else:
                src = f"{src}{pass_arg}, "

                if pass_arg == "environment":

                    def finalize(value: t.Any) -> t.Any:  # noqa: F811
                        return default(env_finalize(self.environment, value))

        self._finalize = self._FinalizeInfo(finalize, src)
        return self._finalize

    def _output_const_repr(self, group: t.Iterable[t.Any]) -> str:
        """Given a group of constant values converted from ``Output``
        child nodes, produce a string to write to the template module
        source.
        """
        return repr(concat(group))

    def _output_child_to_const(
        self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo
    ) -> str:
        """Try to optimize a child of an ``Output`` node by trying to
        convert it to constant, finalized data at compile time.

        If :exc:`Impossible` is raised, the node is not constant and
        will be evaluated at runtime. Any other exception will also be
        evaluated at runtime for easier debugging.
        """
        const = node.as_const(frame.eval_ctx)

        if frame.eval_ctx.autoescape:
            const = escape(const)

        # Template data doesn't go through finalize.
        if isinstance(node, nodes.TemplateData):
            return str(const)

        return finalize.const(const)  # type: ignore

    def _output_child_pre(
        self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo
    ) -> None:
        """Output extra source code before visiting a child of an
        ``Output`` node.
        """
        if frame.eval_ctx.volatile:
            self.write("(escape if context.eval_ctx.autoescape else str)(")
        elif frame.eval_ctx.autoescape:
            self.write("escape(")
        else:
            self.write("str(")

        if finalize.src is not None:
            self.write(finalize.src)

    def _output_child_post(
        self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo
    ) -> None:
        """Output extra source code after visiting a child of an
        ``Output`` node.
        """
        self.write(")")

        if finalize.src is not None:
            self.write(")")

    def visit_Output(self, node: nodes.Output, frame: Frame) -> None:
        # If an extends is active, don't render outside a block.
        if frame.require_output_check:
            # A top-level extends is known to exist at compile time.
            if self.has_known_extends:
                return

            self.writeline("if parent_template is None:")
            self.indent()

        finalize = self._make_finalize()
        body: t.List[t.Union[t.List[t.Any], nodes.Expr]] = []

        # Evaluate constants at compile time if possible. Each item in
        # body will be either a list of static data or a node to be
        # evaluated at runtime.
        for child in node.nodes:
            try:
                if not (
                    # If the finalize function requires runtime context,
                    # constants can't be evaluated at compile time.
                    finalize.const
                    # Unless it's basic template data that won't be
                    # finalized anyway.
                    or isinstance(child, nodes.TemplateData)
                ):
                    raise nodes.Impossible()

                const = self._output_child_to_const(child, frame, finalize)
            except (nodes.Impossible, Exception):
                # The node was not constant and needs to be evaluated at
                # runtime. Or another error was raised, which is easier
                # to debug at runtime.
                body.append(child)
                continue

            if body and isinstance(body[-1], list):
                body[-1].append(const)
            else:
                body.append([const])

        if frame.buffer is not None:
            if len(body) == 1:
                self.writeline(f"{frame.buffer}.append(")
            else:
                self.writeline(f"{frame.buffer}.extend((")

            self.indent()

        for item in body:
            if isinstance(item, list):
                # A group of constant data to join and output.
                val = self._output_const_repr(item)

                if frame.buffer is None:
                    self.writeline("yield " + val)
                else:
                    self.writeline(val + ",")
            else:
                if frame.buffer is None:
                    self.writeline("yield ", item)
                else:
                    self.newline(item)

                # A node to be evaluated at runtime.
                self._output_child_pre(item, frame, finalize)
                self.visit(item, frame)
                self._output_child_post(item, frame, finalize)

                if frame.buffer is not None:
                    self.write(",")

        if frame.buffer is not None:
            self.outdent()
            self.writeline(")" if len(body) == 1 else "))")

        if frame.require_output_check:
            self.outdent()

    def visit_Assign(self, node: nodes.Assign, frame: Frame) -> None:
        self.push_assign_tracking()

        # ``a.b`` is allowed for assignment, and is parsed as an NSRef. However,
        # it is only valid if it references a Namespace object. Emit a check for
        # that for each ref here, before assignment code is emitted. This can't
        # be done in visit_NSRef as the ref could be in the middle of a tuple.
        seen_refs: t.Set[str] = set()

        for nsref in node.find_all(nodes.NSRef):
            if nsref.name in seen_refs:
                # Only emit the check for each reference once, in case the same
                # ref is used multiple times in a tuple, `ns.a, ns.b = c, d`.
                continue

            seen_refs.add(nsref.name)
            ref = frame.symbols.ref(nsref.name)
            self.writeline(f"if not isinstance({ref}, Namespace):")
            self.indent()
            self.writeline(
                "raise TemplateRuntimeError"
                '("cannot assign attribute on non-namespace object")'
            )
            self.outdent()

        self.newline(node)
        self.visit(node.target, frame)
        self.write(" = ")
        self.visit(node.node, frame)
        self.pop_assign_tracking(frame)

    def visit_AssignBlock(self, node: nodes.AssignBlock, frame: Frame) -> None:
        self.push_assign_tracking()
        block_frame = frame.inner()
        # This is a special case.  Since a set block always captures we
        # will disable output checks.  This way one can use set blocks
        # toplevel even in extended templates.
        block_frame.require_output_check = False
        block_frame.symbols.analyze_node(node)
        self.enter_frame(block_frame)
        self.buffer(block_frame)
        self.blockvisit(node.body, block_frame)
        self.newline(node)
        self.visit(node.target, frame)
        self.write(" = (Markup if context.eval_ctx.autoescape else identity)(")
        if node.filter is not None:
            self.visit_Filter(node.filter, block_frame)
        else:
            self.write(f"concat({block_frame.buffer})")
        self.write(")")
        self.pop_assign_tracking(frame)
        self.leave_frame(block_frame)

    # -- Expression Visitors

    def visit_Name(self, node: nodes.Name, frame: Frame) -> None:
        if node.ctx == "store" and (
            frame.toplevel or frame.loop_frame or frame.block_frame
        ):
            if self._assign_stack:
                self._assign_stack[-1].add(node.name)
        ref = frame.symbols.ref(node.name)

        # If we are looking up a variable we might have to deal with the
        # case where it's undefined.  We can skip that case if the load
        # instruction indicates a parameter which are always defined.
        if node.ctx == "load":
            load = frame.symbols.find_load(ref)
            if not (
                load is not None
                and load[0] == VAR_LOAD_PARAMETER
                and not self.parameter_is_undeclared(ref)
            ):
                self.write(
                    f"(undefined(name={node.name!r}) if {ref} is missing else {ref})"
                )
                return

        self.write(ref)

    def visit_NSRef(self, node: nodes.NSRef, frame: Frame) -> None:
        # NSRef is a dotted assignment target a.b=c, but uses a[b]=c internally.
        # visit_Assign emits code to validate that each ref is to a Namespace
        # object only. That can't be emitted here as the ref could be in the
        # middle of a tuple assignment.
        ref = frame.symbols.ref(node.name)
        self.writeline(f"{ref}[{node.attr!r}]")

    def visit_Const(self, node: nodes.Const, frame: Frame) -> None:
        val = node.as_const(frame.eval_ctx)
        if isinstance(val, float):
            self.write(str(val))
        else:
            self.write(repr(val))

    def visit_TemplateData(self, node: nodes.TemplateData, frame: Frame) -> None:
        try:
            self.write(repr(node.as_const(frame.eval_ctx)))
        except nodes.Impossible:
            self.write(
                f"(Markup if context.eval_ctx.autoescape else identity)({node.data!r})"
            )

    def visit_Tuple(self, node: nodes.Tuple, frame: Frame) -> None:
        self.write("(")
        idx = -1
        for idx, item in enumerate(node.items):
            if idx:
                self.write(", ")
            self.visit(item, frame)
        self.write(",)" if idx == 0 else ")")

    def visit_List(self, node: nodes.List, frame: Frame) -> None:
        self.write("[")
        for idx, item in enumerate(node.items):
            if idx:
                self.write(", ")
            self.visit(item, frame)
        self.write("]")

    def visit_Dict(self, node: nodes.Dict, frame: Frame) -> None:
        self.write("{")
        for idx, item in enumerate(node.items):
            if idx:
                self.write(", ")
            self.visit(item.key, frame)
            self.write(": ")
            self.visit(item.value, frame)
        self.write("}")

    visit_Add = _make_binop("+")
    visit_Sub = _make_binop("-")
    visit_Mul = _make_binop("*")
    visit_Div = _make_binop("/")
    visit_FloorDiv = _make_binop("//")
    visit_Pow = _make_binop("**")
    visit_Mod = _make_binop("%")
    visit_And = _make_binop("and")
    visit_Or = _make_binop("or")
    visit_Pos = _make_unop("+")
    visit_Neg = _make_unop("-")
    visit_Not = _make_unop("not ")

    @optimizeconst
    def visit_Concat(self, node: nodes.Concat, frame: Frame) -> None:
        if frame.eval_ctx.volatile:
            func_name = "(markup_join if context.eval_ctx.volatile else str_join)"
        elif frame.eval_ctx.autoescape:
            func_name = "markup_join"
        else:
            func_name = "str_join"
        self.write(f"{func_name}((")
        for arg in node.nodes:
            self.visit(arg, frame)
            self.write(", ")
        self.write("))")

    @optimizeconst
    def visit_Compare(self, node: nodes.Compare, frame: Frame) -> None:
        self.write("(")
        self.visit(node.expr, frame)
        for op in node.ops:
            self.visit(op, frame)
        self.write(")")

    def visit_Operand(self, node: nodes.Operand, frame: Frame) -> None:
        self.write(f" {operators[node.op]} ")
        self.visit(node.expr, frame)

    @optimizeconst
    def visit_Getattr(self, node: nodes.Getattr, frame: Frame) -> None:
        if self.environment.is_async:
            self.write("(await auto_await(")

        self.write("environment.getattr(")
        self.visit(node.node, frame)
        self.write(f", {node.attr!r})")

        if self.environment.is_async:
            self.write("))")

    @optimizeconst
    def visit_Getitem(self, node: nodes.Getitem, frame: Frame) -> None:
        # slices bypass the environment getitem method.
        if isinstance(node.arg, nodes.Slice):
            self.visit(node.node, frame)
            self.write("[")
            self.visit(node.arg, frame)
            self.write("]")
        else:
            if self.environment.is_async:
                self.write("(await auto_await(")

            self.write("environment.getitem(")
            self.visit(node.node, frame)
            self.write(", ")
            self.visit(node.arg, frame)
            self.write(")")

            if self.environment.is_async:
                self.write("))")

    def visit_Slice(self, node: nodes.Slice, frame: Frame) -> None:
        if node.start is not None:
            self.visit(node.start, frame)
        self.write(":")
        if node.stop is not None:
            self.visit(node.stop, frame)
        if node.step is not None:
            self.write(":")
            self.visit(node.step, frame)

    @contextmanager
    def _filter_test_common(
        self, node: t.Union[nodes.Filter, nodes.Test], frame: Frame, is_filter: bool
    ) -> t.Iterator[None]:
        if self.environment.is_async:
            self.write("(await auto_await(")

        if is_filter:
            self.write(f"{self.filters[node.name]}(")
            func = self.environment.filters.get(node.name)
        else:
            self.write(f"{self.tests[node.name]}(")
            func = self.environment.tests.get(node.name)

        # When inside an If or CondExpr frame, allow the filter to be
        # undefined at compile time and only raise an error if it's
        # actually called at runtime. See pull_dependencies.
        if func is None and not frame.soft_frame:
            type_name = "filter" if is_filter else "test"
            self.fail(f"No {type_name} named {node.name!r}.", node.lineno)

        pass_arg = {
            _PassArg.context: "context",
            _PassArg.eval_context: "context.eval_ctx",
            _PassArg.environment: "environment",
        }.get(
            _PassArg.from_obj(func)  # type: ignore
        )

        if pass_arg is not None:
            self.write(f"{pass_arg}, ")

        # Back to the visitor function to handle visiting the target of
        # the filter or test.
        yield

        self.signature(node, frame)
        self.write(")")

        if self.environment.is_async:
            self.write("))")

    @optimizeconst
    def visit_Filter(self, node: nodes.Filter, frame: Frame) -> None:
        with self._filter_test_common(node, frame, True):
            # if the filter node is None we are inside a filter block
            # and want to write to the current buffer
            if node.node is not None:
                self.visit(node.node, frame)
            elif frame.eval_ctx.volatile:
                self.write(
                    f"(Markup(concat({frame.buffer}))"
                    f" if context.eval_ctx.autoescape else concat({frame.buffer}))"
                )
            elif frame.eval_ctx.autoescape:
                self.write(f"Markup(concat({frame.buffer}))")
            else:
                self.write(f"concat({frame.buffer})")

    @optimizeconst
    def visit_Test(self, node: nodes.Test, frame: Frame) -> None:
        with self._filter_test_common(node, frame, False):
            self.visit(node.node, frame)

    @optimizeconst
    def visit_CondExpr(self, node: nodes.CondExpr, frame: Frame) -> None:
        frame = frame.soft()

        def write_expr2() -> None:
            if node.expr2 is not None:
                self.visit(node.expr2, frame)
                return

            self.write(
                f'cond_expr_undefined("the inline if-expression on'
                f" {self.position(node)} evaluated to false and no else"
                f' section was defined.")'
            )

        self.write("(")
        self.visit(node.expr1, frame)
        self.write(" if ")
        self.visit(node.test, frame)
        self.write(" else ")
        write_expr2()
        self.write(")")

    @optimizeconst
    def visit_Call(
        self, node: nodes.Call, frame: Frame, forward_caller: bool = False
    ) -> None:
        if self.environment.is_async:
            self.write("(await auto_await(")
        if self.environment.sandboxed:
            self.write("environment.call(context, ")
        else:
            self.write("context.call(")
        self.visit(node.node, frame)
        extra_kwargs = {"caller": "caller"} if forward_caller else None
        loop_kwargs = {"_loop_vars": "_loop_vars"} if frame.loop_frame else {}
        block_kwargs = {"_block_vars": "_block_vars"} if frame.block_frame else {}
        if extra_kwargs:
            extra_kwargs.update(loop_kwargs, **block_kwargs)
        elif loop_kwargs or block_kwargs:
            extra_kwargs = dict(loop_kwargs, **block_kwargs)
        self.signature(node, frame, extra_kwargs)
        self.write(")")
        if self.environment.is_async:
            self.write("))")

    def visit_Keyword(self, node: nodes.Keyword, frame: Frame) -> None:
        self.write(node.key + "=")
        self.visit(node.value, frame)

    # -- Unused nodes for extensions

    def visit_MarkSafe(self, node: nodes.MarkSafe, frame: Frame) -> None:
        self.write("Markup(")
        self.visit(node.expr, frame)
        self.write(")")

    def visit_MarkSafeIfAutoescape(
        self, node: nodes.MarkSafeIfAutoescape, frame: Frame
    ) -> None:
        self.write("(Markup if context.eval_ctx.autoescape else identity)(")
        self.visit(node.expr, frame)
        self.write(")")

    def visit_EnvironmentAttribute(
        self, node: nodes.EnvironmentAttribute, frame: Frame
    ) -> None:
        self.write("environment." + node.name)

    def visit_ExtensionAttribute(
        self, node: nodes.ExtensionAttribute, frame: Frame
    ) -> None:
        self.write(f"environment.extensions[{node.identifier!r}].{node.name}")

    def visit_ImportedName(self, node: nodes.ImportedName, frame: Frame) -> None:
        self.write(self.import_aliases[node.importname])

    def visit_InternalName(self, node: nodes.InternalName, frame: Frame) -> None:
        self.write(node.name)

    def visit_ContextReference(
        self, node: nodes.ContextReference, frame: Frame
    ) -> None:
        self.write("context")

    def visit_DerivedContextReference(
        self, node: nodes.DerivedContextReference, frame: Frame
    ) -> None:
        self.write(self.derive_context(frame))

    def visit_Continue(self, node: nodes.Continue, frame: Frame) -> None:
        self.writeline("continue", node)

    def visit_Break(self, node: nodes.Break, frame: Frame) -> None:
        self.writeline("break", node)

    def visit_Scope(self, node: nodes.Scope, frame: Frame) -> None:
        scope_frame = frame.inner()
        scope_frame.symbols.analyze_node(node)
        self.enter_frame(scope_frame)
        self.blockvisit(node.body, scope_frame)
        self.leave_frame(scope_frame)

    def visit_OverlayScope(self, node: nodes.OverlayScope, frame: Frame) -> None:
        ctx = self.temporary_identifier()
        self.writeline(f"{ctx} = {self.derive_context(frame)}")
        self.writeline(f"{ctx}.vars = ")
        self.visit(node.context, frame)
        self.push_context_reference(ctx)

        scope_frame = frame.inner(isolated=True)
        scope_frame.symbols.analyze_node(node)
        self.enter_frame(scope_frame)
        self.blockvisit(node.body, scope_frame)
        self.leave_frame(scope_frame)
        self.pop_context_reference()

    def visit_EvalContextModifier(
        self, node: nodes.EvalContextModifier, frame: Frame
    ) -> None:
        for keyword in node.options:
            self.writeline(f"context.eval_ctx.{keyword.key} = ")
            self.visit(keyword.value, frame)
            try:
                val = keyword.value.as_const(frame.eval_ctx)
            except nodes.Impossible:
                frame.eval_ctx.volatile = True
            else:
                setattr(frame.eval_ctx, keyword.key, val)

    def visit_ScopedEvalContextModifier(
        self, node: nodes.ScopedEvalContextModifier, frame: Frame
    ) -> None:
        old_ctx_name = self.temporary_identifier()
        saved_ctx = frame.eval_ctx.save()
        self.writeline(f"{old_ctx_name} = context.eval_ctx.save()")
        self.visit_EvalContextModifier(node, frame)
        for child in node.body:
            self.visit(child, frame)
        frame.eval_ctx.revert(saved_ctx)
        self.writeline(f"context.eval_ctx.revert({old_ctx_name})")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\dtypes\cast.py ===
"""
Routines for casting.
"""

from __future__ import annotations

import datetime as dt
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import (
    Interval,
    Period,
    lib,
)
from pandas._libs.missing import (
    NA,
    NAType,
    checknull,
)
from pandas._libs.tslibs import (
    NaT,
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
    Timedelta,
    Timestamp,
    is_supported_dtype,
)
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import (
    IntCastingNaNError,
    LossySetitemError,
)

from pandas.core.dtypes.common import (
    ensure_int8,
    ensure_int16,
    ensure_int32,
    ensure_int64,
    ensure_object,
    ensure_str,
    is_bool,
    is_complex,
    is_float,
    is_integer,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype as pandas_dtype_func,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    BaseMaskedDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PandasExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCExtensionArray,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
    notna,
)

from pandas.io._util import _arrow_dtype_mapping

if TYPE_CHECKING:
    from collections.abc import (
        Collection,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        Dtype,
        DtypeObj,
        NumpyIndexT,
        Scalar,
        npt,
    )

    from pandas import Index
    from pandas.core.arrays import (
        Categorical,
        DatetimeArray,
        ExtensionArray,
        IntervalArray,
        PeriodArray,
        TimedeltaArray,
    )


_int8_max = np.iinfo(np.int8).max
_int16_max = np.iinfo(np.int16).max
_int32_max = np.iinfo(np.int32).max

_dtype_obj = np.dtype(object)

NumpyArrayT = TypeVar("NumpyArrayT", bound=np.ndarray)


def maybe_convert_platform(
    values: list | tuple | range | np.ndarray | ExtensionArray,
) -> ArrayLike:
    """try to do platform conversion, allow ndarray or list here"""
    arr: ArrayLike

    if isinstance(values, (list, tuple, range)):
        arr = construct_1d_object_array_from_listlike(values)
    else:
        # The caller is responsible for ensuring that we have np.ndarray
        #  or ExtensionArray here.
        arr = values

    if arr.dtype == _dtype_obj:
        arr = cast(np.ndarray, arr)
        arr = lib.maybe_convert_objects(arr)

    return arr


def is_nested_object(obj) -> bool:
    """
    return a boolean if we have a nested object, e.g. a Series with 1 or
    more Series elements

    This may not be necessarily be performant.

    """
    return bool(
        isinstance(obj, ABCSeries)
        and is_object_dtype(obj.dtype)
        and any(isinstance(v, ABCSeries) for v in obj._values)
    )


def maybe_box_datetimelike(value: Scalar, dtype: Dtype | None = None) -> Scalar:
    """
    Cast scalar to Timestamp or Timedelta if scalar is datetime-like
    and dtype is not object.

    Parameters
    ----------
    value : scalar
    dtype : Dtype, optional

    Returns
    -------
    scalar
    """
    if dtype == _dtype_obj:
        pass
    elif isinstance(value, (np.datetime64, dt.datetime)):
        value = Timestamp(value)
    elif isinstance(value, (np.timedelta64, dt.timedelta)):
        value = Timedelta(value)

    return value


def maybe_box_native(value: Scalar | None | NAType) -> Scalar | None | NAType:
    """
    If passed a scalar cast the scalar to a python native type.

    Parameters
    ----------
    value : scalar or Series

    Returns
    -------
    scalar or Series
    """
    if is_float(value):
        value = float(value)
    elif is_integer(value):
        value = int(value)
    elif is_bool(value):
        value = bool(value)
    elif isinstance(value, (np.datetime64, np.timedelta64)):
        value = maybe_box_datetimelike(value)
    elif value is NA:
        value = None
    return value


def _maybe_unbox_datetimelike(value: Scalar, dtype: DtypeObj) -> Scalar:
    """
    Convert a Timedelta or Timestamp to timedelta64 or datetime64 for setting
    into a numpy array.  Failing to unbox would risk dropping nanoseconds.

    Notes
    -----
    Caller is responsible for checking dtype.kind in "mM"
    """
    if is_valid_na_for_dtype(value, dtype):
        # GH#36541: can't fill array directly with pd.NaT
        # > np.empty(10, dtype="datetime64[ns]").fill(pd.NaT)
        # ValueError: cannot convert float NaN to integer
        value = dtype.type("NaT", "ns")
    elif isinstance(value, Timestamp):
        if value.tz is None:
            value = value.to_datetime64()
        elif not isinstance(dtype, DatetimeTZDtype):
            raise TypeError("Cannot unbox tzaware Timestamp to tznaive dtype")
    elif isinstance(value, Timedelta):
        value = value.to_timedelta64()

    _disallow_mismatched_datetimelike(value, dtype)
    return value


def _disallow_mismatched_datetimelike(value, dtype: DtypeObj):
    """
    numpy allows np.array(dt64values, dtype="timedelta64[ns]") and
    vice-versa, but we do not want to allow this, so we need to
    check explicitly
    """
    vdtype = getattr(value, "dtype", None)
    if vdtype is None:
        return
    elif (vdtype.kind == "m" and dtype.kind == "M") or (
        vdtype.kind == "M" and dtype.kind == "m"
    ):
        raise TypeError(f"Cannot cast {repr(value)} to {dtype}")


@overload
def maybe_downcast_to_dtype(result: np.ndarray, dtype: str | np.dtype) -> np.ndarray:
    ...


@overload
def maybe_downcast_to_dtype(result: ExtensionArray, dtype: str | np.dtype) -> ArrayLike:
    ...


def maybe_downcast_to_dtype(result: ArrayLike, dtype: str | np.dtype) -> ArrayLike:
    """
    try to cast to the specified dtype (e.g. convert back to bool/int
    or could be an astype of float64->float32
    """
    if isinstance(result, ABCSeries):
        result = result._values
    do_round = False

    if isinstance(dtype, str):
        if dtype == "infer":
            inferred_type = lib.infer_dtype(result, skipna=False)
            if inferred_type == "boolean":
                dtype = "bool"
            elif inferred_type == "integer":
                dtype = "int64"
            elif inferred_type == "datetime64":
                dtype = "datetime64[ns]"
            elif inferred_type in ["timedelta", "timedelta64"]:
                dtype = "timedelta64[ns]"

            # try to upcast here
            elif inferred_type == "floating":
                dtype = "int64"
                if issubclass(result.dtype.type, np.number):
                    do_round = True

            else:
                # TODO: complex?  what if result is already non-object?
                dtype = "object"

        dtype = np.dtype(dtype)

    if not isinstance(dtype, np.dtype):
        # enforce our signature annotation
        raise TypeError(dtype)  # pragma: no cover

    converted = maybe_downcast_numeric(result, dtype, do_round)
    if converted is not result:
        return converted

    # a datetimelike
    # GH12821, iNaT is cast to float
    if dtype.kind in "mM" and result.dtype.kind in "if":
        result = result.astype(dtype)

    elif dtype.kind == "m" and result.dtype == _dtype_obj:
        # test_where_downcast_to_td64
        result = cast(np.ndarray, result)
        result = array_to_timedelta64(result)

    elif dtype == np.dtype("M8[ns]") and result.dtype == _dtype_obj:
        result = cast(np.ndarray, result)
        return np.asarray(maybe_cast_to_datetime(result, dtype=dtype))

    return result


@overload
def maybe_downcast_numeric(
    result: np.ndarray, dtype: np.dtype, do_round: bool = False
) -> np.ndarray:
    ...


@overload
def maybe_downcast_numeric(
    result: ExtensionArray, dtype: DtypeObj, do_round: bool = False
) -> ArrayLike:
    ...


def maybe_downcast_numeric(
    result: ArrayLike, dtype: DtypeObj, do_round: bool = False
) -> ArrayLike:
    """
    Subset of maybe_downcast_to_dtype restricted to numeric dtypes.

    Parameters
    ----------
    result : ndarray or ExtensionArray
    dtype : np.dtype or ExtensionDtype
    do_round : bool

    Returns
    -------
    ndarray or ExtensionArray
    """
    if not isinstance(dtype, np.dtype) or not isinstance(result.dtype, np.dtype):
        # e.g. SparseDtype has no itemsize attr
        return result

    def trans(x):
        if do_round:
            return x.round()
        return x

    if dtype.kind == result.dtype.kind:
        # don't allow upcasts here (except if empty)
        if result.dtype.itemsize <= dtype.itemsize and result.size:
            return result

    if dtype.kind in "biu":
        if not result.size:
            # if we don't have any elements, just astype it
            return trans(result).astype(dtype)

        if isinstance(result, np.ndarray):
            element = result.item(0)
        else:
            element = result.iloc[0]
        if not isinstance(element, (np.integer, np.floating, int, float, bool)):
            # a comparable, e.g. a Decimal may slip in here
            return result

        if (
            issubclass(result.dtype.type, (np.object_, np.number))
            and notna(result).all()
        ):
            new_result = trans(result).astype(dtype)
            if new_result.dtype.kind == "O" or result.dtype.kind == "O":
                # np.allclose may raise TypeError on object-dtype
                if (new_result == result).all():
                    return new_result
            else:
                if np.allclose(new_result, result, rtol=0):
                    return new_result

    elif (
        issubclass(dtype.type, np.floating)
        and result.dtype.kind != "b"
        and not is_string_dtype(result.dtype)
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "overflow encountered in cast", RuntimeWarning
            )
            new_result = result.astype(dtype)

        # Adjust tolerances based on floating point size
        size_tols = {4: 5e-4, 8: 5e-8, 16: 5e-16}

        atol = size_tols.get(new_result.dtype.itemsize, 0.0)

        # Check downcast float values are still equal within 7 digits when
        # converting from float64 to float32
        if np.allclose(new_result, result, equal_nan=True, rtol=0.0, atol=atol):
            return new_result

    elif dtype.kind == result.dtype.kind == "c":
        new_result = result.astype(dtype)

        if np.array_equal(new_result, result, equal_nan=True):
            # TODO: use tolerance like we do for float?
            return new_result

    return result


def maybe_upcast_numeric_to_64bit(arr: NumpyIndexT) -> NumpyIndexT:
    """
    If array is a int/uint/float bit size lower than 64 bit, upcast it to 64 bit.

    Parameters
    ----------
    arr : ndarray or ExtensionArray

    Returns
    -------
    ndarray or ExtensionArray
    """
    dtype = arr.dtype
    if dtype.kind == "i" and dtype != np.int64:
        return arr.astype(np.int64)
    elif dtype.kind == "u" and dtype != np.uint64:
        return arr.astype(np.uint64)
    elif dtype.kind == "f" and dtype != np.float64:
        return arr.astype(np.float64)
    else:
        return arr


def maybe_cast_pointwise_result(
    result: ArrayLike,
    dtype: DtypeObj,
    numeric_only: bool = False,
    same_dtype: bool = True,
) -> ArrayLike:
    """
    Try casting result of a pointwise operation back to the original dtype if
    appropriate.

    Parameters
    ----------
    result : array-like
        Result to cast.
    dtype : np.dtype or ExtensionDtype
        Input Series from which result was calculated.
    numeric_only : bool, default False
        Whether to cast only numerics or datetimes as well.
    same_dtype : bool, default True
        Specify dtype when calling _from_sequence

    Returns
    -------
    result : array-like
        result maybe casted to the dtype.
    """

    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        if same_dtype:
            result = _maybe_cast_to_extension_array(cls, result, dtype=dtype)
        else:
            result = _maybe_cast_to_extension_array(cls, result)

    elif (numeric_only and dtype.kind in "iufcb") or not numeric_only:
        result = maybe_downcast_to_dtype(result, dtype)

    return result


def _maybe_cast_to_extension_array(
    cls: type[ExtensionArray], obj: ArrayLike, dtype: ExtensionDtype | None = None
) -> ArrayLike:
    """
    Call to `_from_sequence` that returns the object unchanged on Exception.

    Parameters
    ----------
    cls : class, subclass of ExtensionArray
    obj : arraylike
        Values to pass to cls._from_sequence
    dtype : ExtensionDtype, optional

    Returns
    -------
    ExtensionArray or obj
    """
    result: ArrayLike

    if dtype is not None:
        try:
            result = cls._from_scalars(obj, dtype=dtype)
        except (TypeError, ValueError):
            return obj
        return result

    try:
        result = cls._from_sequence(obj, dtype=dtype)
    except Exception:
        # We can't predict what downstream EA constructors may raise
        result = obj
    return result


@overload
def ensure_dtype_can_hold_na(dtype: np.dtype) -> np.dtype:
    ...


@overload
def ensure_dtype_can_hold_na(dtype: ExtensionDtype) -> ExtensionDtype:
    ...


def ensure_dtype_can_hold_na(dtype: DtypeObj) -> DtypeObj:
    """
    If we have a dtype that cannot hold NA values, find the best match that can.
    """
    if isinstance(dtype, ExtensionDtype):
        if dtype._can_hold_na:
            return dtype
        elif isinstance(dtype, IntervalDtype):
            # TODO(GH#45349): don't special-case IntervalDtype, allow
            #  overriding instead of returning object below.
            return IntervalDtype(np.float64, closed=dtype.closed)
        return _dtype_obj
    elif dtype.kind == "b":
        return _dtype_obj
    elif dtype.kind in "iu":
        return np.dtype(np.float64)
    return dtype


_canonical_nans = {
    np.datetime64: np.datetime64("NaT", "ns"),
    np.timedelta64: np.timedelta64("NaT", "ns"),
    type(np.nan): np.nan,
}


def maybe_promote(dtype: np.dtype, fill_value=np.nan):
    """
    Find the minimal dtype that can hold both the given dtype and fill_value.

    Parameters
    ----------
    dtype : np.dtype
    fill_value : scalar, default np.nan

    Returns
    -------
    dtype
        Upcasted from dtype argument if necessary.
    fill_value
        Upcasted from fill_value argument if necessary.

    Raises
    ------
    ValueError
        If fill_value is a non-scalar and dtype is not object.
    """
    orig = fill_value
    orig_is_nat = False
    if checknull(fill_value):
        # https://github.com/pandas-dev/pandas/pull/39692#issuecomment-1441051740
        #  avoid cache misses with NaN/NaT values that are not singletons
        if fill_value is not NA:
            try:
                orig_is_nat = np.isnat(fill_value)
            except TypeError:
                pass

        fill_value = _canonical_nans.get(type(fill_value), fill_value)

    # for performance, we are using a cached version of the actual implementation
    # of the function in _maybe_promote. However, this doesn't always work (in case
    # of non-hashable arguments), so we fallback to the actual implementation if needed
    try:
        # error: Argument 3 to "__call__" of "_lru_cache_wrapper" has incompatible type
        # "Type[Any]"; expected "Hashable"  [arg-type]
        dtype, fill_value = _maybe_promote_cached(
            dtype, fill_value, type(fill_value)  # type: ignore[arg-type]
        )
    except TypeError:
        # if fill_value is not hashable (required for caching)
        dtype, fill_value = _maybe_promote(dtype, fill_value)

    if (dtype == _dtype_obj and orig is not None) or (
        orig_is_nat and np.datetime_data(orig)[0] != "ns"
    ):
        # GH#51592,53497 restore our potentially non-canonical fill_value
        fill_value = orig
    return dtype, fill_value


@functools.lru_cache
def _maybe_promote_cached(dtype, fill_value, fill_value_type):
    # The cached version of _maybe_promote below
    # This also use fill_value_type as (unused) argument to use this in the
    # cache lookup -> to differentiate 1 and True
    return _maybe_promote(dtype, fill_value)


def _maybe_promote(dtype: np.dtype, fill_value=np.nan):
    # The actual implementation of the function, use `maybe_promote` above for
    # a cached version.
    if not is_scalar(fill_value):
        # with object dtype there is nothing to promote, and the user can
        #  pass pretty much any weird fill_value they like
        if dtype != object:
            # with object dtype there is nothing to promote, and the user can
            #  pass pretty much any weird fill_value they like
            raise ValueError("fill_value must be a scalar")
        dtype = _dtype_obj
        return dtype, fill_value

    if is_valid_na_for_dtype(fill_value, dtype) and dtype.kind in "iufcmM":
        dtype = ensure_dtype_can_hold_na(dtype)
        fv = na_value_for_dtype(dtype)
        return dtype, fv

    elif isinstance(dtype, CategoricalDtype):
        if fill_value in dtype.categories or isna(fill_value):
            return dtype, fill_value
        else:
            return object, ensure_object(fill_value)

    elif isna(fill_value):
        dtype = _dtype_obj
        if fill_value is None:
            # but we retain e.g. pd.NA
            fill_value = np.nan
        return dtype, fill_value

    # returns tuple of (dtype, fill_value)
    if issubclass(dtype.type, np.datetime64):
        inferred, fv = infer_dtype_from_scalar(fill_value)
        if inferred == dtype:
            return dtype, fv

        from pandas.core.arrays import DatetimeArray

        dta = DatetimeArray._from_sequence([], dtype="M8[ns]")
        try:
            fv = dta._validate_setitem_value(fill_value)
            return dta.dtype, fv
        except (ValueError, TypeError):
            return _dtype_obj, fill_value

    elif issubclass(dtype.type, np.timedelta64):
        inferred, fv = infer_dtype_from_scalar(fill_value)
        if inferred == dtype:
            return dtype, fv

        elif inferred.kind == "m":
            # different unit, e.g. passed np.timedelta64(24, "h") with dtype=m8[ns]
            # see if we can losslessly cast it to our dtype
            unit = np.datetime_data(dtype)[0]
            try:
                td = Timedelta(fill_value).as_unit(unit, round_ok=False)
            except OutOfBoundsTimedelta:
                return _dtype_obj, fill_value
            else:
                return dtype, td.asm8

        return _dtype_obj, fill_value

    elif is_float(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)

        elif issubclass(dtype.type, np.integer):
            dtype = np.dtype(np.float64)

        elif dtype.kind == "f":
            mst = np.min_scalar_type(fill_value)
            if mst > dtype:
                # e.g. mst is np.float64 and dtype is np.float32
                dtype = mst

        elif dtype.kind == "c":
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)

    elif is_bool(fill_value):
        if not issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)

    elif is_integer(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)

        elif issubclass(dtype.type, np.integer):
            if not np_can_cast_scalar(fill_value, dtype):  # type: ignore[arg-type]
                # upcast to prevent overflow
                mst = np.min_scalar_type(fill_value)
                dtype = np.promote_types(dtype, mst)
                if dtype.kind == "f":
                    # Case where we disagree with numpy
                    dtype = np.dtype(np.object_)

    elif is_complex(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)

        elif issubclass(dtype.type, (np.integer, np.floating)):
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)

        elif dtype.kind == "c":
            mst = np.min_scalar_type(fill_value)
            if mst > dtype:
                # e.g. mst is np.complex128 and dtype is np.complex64
                dtype = mst

    else:
        dtype = np.dtype(np.object_)

    # in case we have a string that looked like a number
    if issubclass(dtype.type, (bytes, str)):
        dtype = np.dtype(np.object_)

    fill_value = _ensure_dtype_type(fill_value, dtype)
    return dtype, fill_value


def _ensure_dtype_type(value, dtype: np.dtype):
    """
    Ensure that the given value is an instance of the given dtype.

    e.g. if out dtype is np.complex64_, we should have an instance of that
    as opposed to a python complex object.

    Parameters
    ----------
    value : object
    dtype : np.dtype

    Returns
    -------
    object
    """
    # Start with exceptions in which we do _not_ cast to numpy types

    if dtype == _dtype_obj:
        return value

    # Note: before we get here we have already excluded isna(value)
    return dtype.type(value)


def infer_dtype_from(val) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar or array.

    Parameters
    ----------
    val : object
    """
    if not is_list_like(val):
        return infer_dtype_from_scalar(val)
    return infer_dtype_from_array(val)


def infer_dtype_from_scalar(val) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar.

    Parameters
    ----------
    val : object
    """
    dtype: DtypeObj = _dtype_obj

    # a 1-element ndarray
    if isinstance(val, np.ndarray):
        if val.ndim != 0:
            msg = "invalid ndarray passed to infer_dtype_from_scalar"
            raise ValueError(msg)

        dtype = val.dtype
        val = lib.item_from_zerodim(val)

    elif isinstance(val, str):
        # If we create an empty array using a string to infer
        # the dtype, NumPy will only allocate one character per entry
        # so this is kind of bad. Alternately we could use np.repeat
        # instead of np.empty (but then you still don't want things
        # coming out as np.str_!

        dtype = _dtype_obj
        if using_string_dtype():
            from pandas.core.arrays.string_ import StringDtype

            dtype = StringDtype(na_value=np.nan)

    elif isinstance(val, (np.datetime64, dt.datetime)):
        try:
            val = Timestamp(val)
        except OutOfBoundsDatetime:
            return _dtype_obj, val

        if val is NaT or val.tz is None:
            val = val.to_datetime64()
            dtype = val.dtype
            # TODO: test with datetime(2920, 10, 1) based on test_replace_dtypes
        else:
            dtype = DatetimeTZDtype(unit=val.unit, tz=val.tz)

    elif isinstance(val, (np.timedelta64, dt.timedelta)):
        try:
            val = Timedelta(val)
        except (OutOfBoundsTimedelta, OverflowError):
            dtype = _dtype_obj
        else:
            if val is NaT:
                val = np.timedelta64("NaT", "ns")
            else:
                val = val.asm8
            dtype = val.dtype

    elif is_bool(val):
        dtype = np.dtype(np.bool_)

    elif is_integer(val):
        if isinstance(val, np.integer):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.int64)

        try:
            np.array(val, dtype=dtype)
        except OverflowError:
            dtype = np.array(val).dtype

    elif is_float(val):
        if isinstance(val, np.floating):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.float64)

    elif is_complex(val):
        dtype = np.dtype(np.complex128)

    if isinstance(val, Period):
        dtype = PeriodDtype(freq=val.freq)
    elif isinstance(val, Interval):
        subtype = infer_dtype_from_scalar(val.left)[0]
        dtype = IntervalDtype(subtype=subtype, closed=val.closed)

    return dtype, val


def dict_compat(d: dict[Scalar, Scalar]) -> dict[Scalar, Scalar]:
    """
    Convert datetimelike-keyed dicts to a Timestamp-keyed dict.

    Parameters
    ----------
    d: dict-like object

    Returns
    -------
    dict
    """
    return {maybe_box_datetimelike(key): value for key, value in d.items()}


def infer_dtype_from_array(arr) -> tuple[DtypeObj, ArrayLike]:
    """
    Infer the dtype from an array.

    Parameters
    ----------
    arr : array

    Returns
    -------
    tuple (pandas-compat dtype, array)


    Examples
    --------
    >>> np.asarray([1, '1'])
    array(['1', '1'], dtype='<U21')

    >>> infer_dtype_from_array([1, '1'])
    (dtype('O'), [1, '1'])
    """
    if isinstance(arr, np.ndarray):
        return arr.dtype, arr

    if not is_list_like(arr):
        raise TypeError("'arr' must be list-like")

    arr_dtype = getattr(arr, "dtype", None)
    if isinstance(arr_dtype, ExtensionDtype):
        return arr.dtype, arr

    elif isinstance(arr, ABCSeries):
        return arr.dtype, np.asarray(arr)

    # don't force numpy coerce with nan's
    inferred = lib.infer_dtype(arr, skipna=False)
    if inferred in ["string", "bytes", "mixed", "mixed-integer"]:
        return (np.dtype(np.object_), arr)

    arr = np.asarray(arr)
    return arr.dtype, arr


def _maybe_infer_dtype_type(element):
    """
    Try to infer an object's dtype, for use in arithmetic ops.

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> _maybe_infer_dtype_type(Foo(np.dtype("i8")))
    dtype('int64')
    """
    tipo = None
    if hasattr(element, "dtype"):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo


def invalidate_string_dtypes(dtype_set: set[DtypeObj]) -> None:
    """
    Change string like dtypes to object for
    ``DataFrame.select_dtypes()``.
    """
    # error: Argument 1 to <set> has incompatible type "Type[generic]"; expected
    # "Union[dtype[Any], ExtensionDtype, None]"
    # error: Argument 2 to <set> has incompatible type "Type[generic]"; expected
    # "Union[dtype[Any], ExtensionDtype, None]"
    non_string_dtypes = dtype_set - {
        np.dtype("S").type,  # type: ignore[arg-type]
        np.dtype("<U").type,  # type: ignore[arg-type]
    }
    if non_string_dtypes != dtype_set:
        raise TypeError("string dtypes are not allowed, use 'object' instead")


def coerce_indexer_dtype(indexer, categories) -> np.ndarray:
    """coerce the indexer input array to the smallest dtype possible"""
    length = len(categories)
    if length < _int8_max:
        return ensure_int8(indexer)
    elif length < _int16_max:
        return ensure_int16(indexer)
    elif length < _int32_max:
        return ensure_int32(indexer)
    return ensure_int64(indexer)


def convert_dtypes(
    input_array: ArrayLike,
    convert_string: bool = True,
    convert_integer: bool = True,
    convert_boolean: bool = True,
    convert_floating: bool = True,
    infer_objects: bool = False,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> DtypeObj:
    """
    Convert objects to best possible type, and optionally,
    to types supporting ``pd.NA``.

    Parameters
    ----------
    input_array : ExtensionArray or np.ndarray
    convert_string : bool, default True
        Whether object dtypes should be converted to ``StringDtype()``.
    convert_integer : bool, default True
        Whether, if possible, conversion can be done to integer extension types.
    convert_boolean : bool, defaults True
        Whether object dtypes should be converted to ``BooleanDtypes()``.
    convert_floating : bool, defaults True
        Whether, if possible, conversion can be done to floating extension types.
        If `convert_integer` is also True, preference will be give to integer
        dtypes if the floats can be faithfully casted to integers.
    infer_objects : bool, defaults False
        Whether to also infer objects to float/int if possible. Is only hit if the
        object array contains pd.NA.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    np.dtype, or ExtensionDtype
    """
    from pandas.core.arrays.string_ import StringDtype

    inferred_dtype: str | DtypeObj

    if (
        convert_string or convert_integer or convert_boolean or convert_floating
    ) and isinstance(input_array, np.ndarray):
        if input_array.dtype == object:
            inferred_dtype = lib.infer_dtype(input_array)
        else:
            inferred_dtype = input_array.dtype

        if is_string_dtype(inferred_dtype):
            if not convert_string or inferred_dtype == "bytes":
                inferred_dtype = input_array.dtype
            else:
                inferred_dtype = pandas_dtype_func("string")

        if convert_integer:
            target_int_dtype = pandas_dtype_func("Int64")

            if input_array.dtype.kind in "iu":
                from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE

                inferred_dtype = NUMPY_INT_TO_DTYPE.get(
                    input_array.dtype, target_int_dtype
                )
            elif input_array.dtype.kind in "fcb":
                # TODO: de-dup with maybe_cast_to_integer_array?
                arr = input_array[notna(input_array)]
                if (arr.astype(int) == arr).all():
                    inferred_dtype = target_int_dtype
                else:
                    inferred_dtype = input_array.dtype
            elif (
                infer_objects
                and input_array.dtype == object
                and (isinstance(inferred_dtype, str) and inferred_dtype == "integer")
            ):
                inferred_dtype = target_int_dtype

        if convert_floating:
            if input_array.dtype.kind in "fcb":
                # i.e. numeric but not integer
                from pandas.core.arrays.floating import NUMPY_FLOAT_TO_DTYPE

                inferred_float_dtype: DtypeObj = NUMPY_FLOAT_TO_DTYPE.get(
                    input_array.dtype, pandas_dtype_func("Float64")
                )
                # if we could also convert to integer, check if all floats
                # are actually integers
                if convert_integer:
                    # TODO: de-dup with maybe_cast_to_integer_array?
                    arr = input_array[notna(input_array)]
                    if (arr.astype(int) == arr).all():
                        inferred_dtype = pandas_dtype_func("Int64")
                    else:
                        inferred_dtype = inferred_float_dtype
                else:
                    inferred_dtype = inferred_float_dtype
            elif (
                infer_objects
                and input_array.dtype == object
                and (
                    isinstance(inferred_dtype, str)
                    and inferred_dtype == "mixed-integer-float"
                )
            ):
                inferred_dtype = pandas_dtype_func("Float64")

        if convert_boolean:
            if input_array.dtype.kind == "b":
                inferred_dtype = pandas_dtype_func("boolean")
            elif isinstance(inferred_dtype, str) and inferred_dtype == "boolean":
                inferred_dtype = pandas_dtype_func("boolean")

        if isinstance(inferred_dtype, str):
            # If we couldn't do anything else, then we retain the dtype
            inferred_dtype = input_array.dtype

    elif (
        convert_string
        and isinstance(input_array.dtype, StringDtype)
        and input_array.dtype.na_value is np.nan
    ):
        inferred_dtype = pandas_dtype_func("string")

    else:
        inferred_dtype = input_array.dtype

    if dtype_backend == "pyarrow":
        from pandas.core.arrays.arrow.array import to_pyarrow_type

        assert not isinstance(inferred_dtype, str)

        if (
            (convert_integer and inferred_dtype.kind in "iu")
            or (convert_floating and inferred_dtype.kind in "fc")
            or (convert_boolean and inferred_dtype.kind == "b")
            or (convert_string and isinstance(inferred_dtype, StringDtype))
            or (
                inferred_dtype.kind not in "iufcb"
                and not isinstance(inferred_dtype, StringDtype)
            )
        ):
            if isinstance(inferred_dtype, PandasExtensionDtype) and not isinstance(
                inferred_dtype, DatetimeTZDtype
            ):
                base_dtype = inferred_dtype.base
            elif isinstance(inferred_dtype, (BaseMaskedDtype, ArrowDtype)):
                base_dtype = inferred_dtype.numpy_dtype
            elif isinstance(inferred_dtype, StringDtype):
                base_dtype = np.dtype(str)
            else:
                base_dtype = inferred_dtype
            if (
                base_dtype.kind == "O"  # type: ignore[union-attr]
                and input_array.size > 0
                and isna(input_array).all()
            ):
                import pyarrow as pa

                pa_type = pa.null()
            else:
                pa_type = to_pyarrow_type(base_dtype)
            if pa_type is not None:
                inferred_dtype = ArrowDtype(pa_type)
    elif dtype_backend == "numpy_nullable" and isinstance(inferred_dtype, ArrowDtype):
        # GH 53648
        inferred_dtype = _arrow_dtype_mapping()[inferred_dtype.pyarrow_dtype]

    # error: Incompatible return value type (got "Union[str, Union[dtype[Any],
    # ExtensionDtype]]", expected "Union[dtype[Any], ExtensionDtype]")
    return inferred_dtype  # type: ignore[return-value]


def maybe_infer_to_datetimelike(
    value: npt.NDArray[np.object_],
    convert_to_nullable_dtype: bool = False,
) -> np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray | IntervalArray:
    """
    we might have a array (or single object) that is datetime like,
    and no dtype is passed don't change the value unless we find a
    datetime/timedelta set

    this is pretty strict in that a datetime/timedelta is REQUIRED
    in addition to possible nulls/string likes

    Parameters
    ----------
    value : np.ndarray[object]

    Returns
    -------
    np.ndarray, DatetimeArray, TimedeltaArray, PeriodArray, or IntervalArray

    """
    if not isinstance(value, np.ndarray) or value.dtype != object:
        # Caller is responsible for passing only ndarray[object]
        raise TypeError(type(value))  # pragma: no cover
    if value.ndim != 1:
        # Caller is responsible
        raise ValueError(value.ndim)  # pragma: no cover

    if not len(value):
        return value

    # error: Incompatible return value type (got "Union[ExtensionArray,
    # ndarray[Any, Any]]", expected "Union[ndarray[Any, Any], DatetimeArray,
    # TimedeltaArray, PeriodArray, IntervalArray]")
    return lib.maybe_convert_objects(  # type: ignore[return-value]
        value,
        # Here we do not convert numeric dtypes, as if we wanted that,
        #  numpy would have done it for us.
        convert_numeric=False,
        convert_non_numeric=True,
        convert_to_nullable_dtype=convert_to_nullable_dtype,
        dtype_if_all_nat=np.dtype("M8[ns]"),
    )


def maybe_cast_to_datetime(
    value: np.ndarray | list, dtype: np.dtype
) -> ExtensionArray | np.ndarray:
    """
    try to cast the array/value to a datetimelike dtype, converting float
    nan to iNaT

    Caller is responsible for handling ExtensionDtype cases and non dt64/td64
    cases.
    """
    from pandas.core.arrays.datetimes import DatetimeArray
    from pandas.core.arrays.timedeltas import TimedeltaArray

    assert dtype.kind in "mM"
    if not is_list_like(value):
        raise TypeError("value must be listlike")

    # TODO: _from_sequence would raise ValueError in cases where
    #  _ensure_nanosecond_dtype raises TypeError
    _ensure_nanosecond_dtype(dtype)

    if lib.is_np_dtype(dtype, "m"):
        res = TimedeltaArray._from_sequence(value, dtype=dtype)
        return res
    else:
        try:
            dta = DatetimeArray._from_sequence(value, dtype=dtype)
        except ValueError as err:
            # We can give a Series-specific exception message.
            if "cannot supply both a tz and a timezone-naive dtype" in str(err):
                raise ValueError(
                    "Cannot convert timezone-aware data to "
                    "timezone-naive dtype. Use "
                    "pd.Series(values).dt.tz_localize(None) instead."
                ) from err
            raise

        return dta


def _ensure_nanosecond_dtype(dtype: DtypeObj) -> None:
    """
    Convert dtypes with granularity less than nanosecond to nanosecond

    >>> _ensure_nanosecond_dtype(np.dtype("M8[us]"))

    >>> _ensure_nanosecond_dtype(np.dtype("M8[D]"))
    Traceback (most recent call last):
        ...
    TypeError: dtype=datetime64[D] is not supported. Supported resolutions are 's', 'ms', 'us', and 'ns'

    >>> _ensure_nanosecond_dtype(np.dtype("m8[ps]"))
    Traceback (most recent call last):
        ...
    TypeError: dtype=timedelta64[ps] is not supported. Supported resolutions are 's', 'ms', 'us', and 'ns'
    """  # noqa: E501
    msg = (
        f"The '{dtype.name}' dtype has no unit. "
        f"Please pass in '{dtype.name}[ns]' instead."
    )

    # unpack e.g. SparseDtype
    dtype = getattr(dtype, "subtype", dtype)

    if not isinstance(dtype, np.dtype):
        # i.e. datetime64tz
        pass

    elif dtype.kind in "mM":
        if not is_supported_dtype(dtype):
            # pre-2.0 we would silently swap in nanos for lower-resolutions,
            #  raise for above-nano resolutions
            if dtype.name in ["datetime64", "timedelta64"]:
                raise ValueError(msg)
            # TODO: ValueError or TypeError? existing test
            #  test_constructor_generic_timestamp_bad_frequency expects TypeError
            raise TypeError(
                f"dtype={dtype} is not supported. Supported resolutions are 's', "
                "'ms', 'us', and 'ns'"
            )


# TODO: other value-dependent functions to standardize here include
#  Index._find_common_type_compat
def find_result_type(left_dtype: DtypeObj, right: Any) -> DtypeObj:
    """
    Find the type/dtype for the result of an operation between objects.

    This is similar to find_common_type, but looks at the right object instead
    of just its dtype. This can be useful in particular when the right
    object does not have a `dtype`.

    Parameters
    ----------
    left_dtype : np.dtype or ExtensionDtype
    right : Any

    Returns
    -------
    np.dtype or ExtensionDtype

    See also
    --------
    find_common_type
    numpy.result_type
    """
    new_dtype: DtypeObj

    if (
        isinstance(left_dtype, np.dtype)
        and left_dtype.kind in "iuc"
        and (lib.is_integer(right) or lib.is_float(right))
    ):
        # e.g. with int8 dtype and right=512, we want to end up with
        # np.int16, whereas infer_dtype_from(512) gives np.int64,
        #  which will make us upcast too far.
        if lib.is_float(right) and right.is_integer() and left_dtype.kind != "f":
            right = int(right)
        # After NEP 50, numpy won't inspect Python scalars
        # TODO: do we need to recreate numpy's inspection logic for floats too
        # (this breaks some tests)
        if isinstance(right, int) and not isinstance(right, np.integer):
            # This gives an unsigned type by default
            # (if our number is positive)

            # If our left dtype is signed, we might not want this since
            # this might give us 1 dtype too big
            # We should check if the corresponding int dtype (e.g. int64 for uint64)
            # can hold the number
            right_dtype = np.min_scalar_type(right)
            if right == 0:
                # Special case 0
                right = left_dtype
            elif (
                not np.issubdtype(left_dtype, np.unsignedinteger)
                and 0 < right <= np.iinfo(right_dtype).max
            ):
                # If left dtype isn't unsigned, check if it fits in the signed dtype
                right = np.dtype(f"i{right_dtype.itemsize}")
            else:
                right = right_dtype

        new_dtype = np.result_type(left_dtype, right)

    elif is_valid_na_for_dtype(right, left_dtype):
        # e.g. IntervalDtype[int] and None/np.nan
        new_dtype = ensure_dtype_can_hold_na(left_dtype)

    else:
        dtype, _ = infer_dtype_from(right)
        new_dtype = find_common_type([left_dtype, dtype])

    return new_dtype


def common_dtype_categorical_compat(
    objs: Sequence[Index | ArrayLike], dtype: DtypeObj
) -> DtypeObj:
    """
    Update the result of find_common_type to account for NAs in a Categorical.

    Parameters
    ----------
    objs : list[np.ndarray | ExtensionArray | Index]
    dtype : np.dtype or ExtensionDtype

    Returns
    -------
    np.dtype or ExtensionDtype
    """
    # GH#38240

    # TODO: more generally, could do `not can_hold_na(dtype)`
    if lib.is_np_dtype(dtype, "iu"):
        for obj in objs:
            # We don't want to accientally allow e.g. "categorical" str here
            obj_dtype = getattr(obj, "dtype", None)
            if isinstance(obj_dtype, CategoricalDtype):
                if isinstance(obj, ABCIndex):
                    # This check may already be cached
                    hasnas = obj.hasnans
                else:
                    # Categorical
                    hasnas = cast("Categorical", obj)._hasna

                if hasnas:
                    # see test_union_int_categorical_with_nan
                    dtype = np.dtype(np.float64)
                    break
    return dtype


def np_find_common_type(*dtypes: np.dtype) -> np.dtype:
    """
    np.find_common_type implementation pre-1.25 deprecation using np.result_type
    https://github.com/pandas-dev/pandas/pull/49569#issuecomment-1308300065

    Parameters
    ----------
    dtypes : np.dtypes

    Returns
    -------
    np.dtype
    """
    try:
        common_dtype = np.result_type(*dtypes)
        if common_dtype.kind in "mMSU":
            # NumPy promotion currently (1.25) misbehaves for for times and strings,
            # so fall back to object (find_common_dtype did unless there
            # was only one dtype)
            common_dtype = np.dtype("O")

    except TypeError:
        common_dtype = np.dtype("O")
    return common_dtype


@overload
def find_common_type(types: list[np.dtype]) -> np.dtype:
    ...


@overload
def find_common_type(types: list[ExtensionDtype]) -> DtypeObj:
    ...


@overload
def find_common_type(types: list[DtypeObj]) -> DtypeObj:
    ...


def find_common_type(types):
    """
    Find a common data type among the given dtypes.

    Parameters
    ----------
    types : list of dtypes

    Returns
    -------
    pandas extension or numpy dtype

    See Also
    --------
    numpy.find_common_type

    """
    if not types:
        raise ValueError("no types given")

    first = types[0]

    # workaround for find_common_type([np.dtype('datetime64[ns]')] * 2)
    # => object
    if lib.dtypes_all_equal(list(types)):
        return first

    # get unique types (dict.fromkeys is used as order-preserving set())
    types = list(dict.fromkeys(types).keys())

    if any(isinstance(t, ExtensionDtype) for t in types):
        for t in types:
            if isinstance(t, ExtensionDtype):
                res = t._get_common_dtype(types)
                if res is not None:
                    return res
        return np.dtype("object")

    # take lowest unit
    if all(lib.is_np_dtype(t, "M") for t in types):
        return np.dtype(max(types))
    if all(lib.is_np_dtype(t, "m") for t in types):
        return np.dtype(max(types))

    # don't mix bool / int or float or complex
    # this is different from numpy, which casts bool with float/int as int
    has_bools = any(t.kind == "b" for t in types)
    if has_bools:
        for t in types:
            if t.kind in "iufc":
                return np.dtype("object")

    return np_find_common_type(*types)


def construct_2d_arraylike_from_scalar(
    value: Scalar, length: int, width: int, dtype: np.dtype, copy: bool
) -> np.ndarray:
    shape = (length, width)

    if dtype.kind in "mM":
        value = _maybe_box_and_unbox_datetimelike(value, dtype)
    elif dtype == _dtype_obj:
        if isinstance(value, (np.timedelta64, np.datetime64)):
            # calling np.array below would cast to pytimedelta/pydatetime
            out = np.empty(shape, dtype=object)
            out.fill(value)
            return out

    # Attempt to coerce to a numpy array
    try:
        if not copy:
            arr = np.asarray(value, dtype=dtype)
        else:
            arr = np.array(value, dtype=dtype, copy=copy)
    except (ValueError, TypeError) as err:
        raise TypeError(
            f"DataFrame constructor called with incompatible data and dtype: {err}"
        ) from err

    if arr.ndim != 0:
        raise ValueError("DataFrame constructor not properly called!")

    return np.full(shape, arr)


def construct_1d_arraylike_from_scalar(
    value: Scalar, length: int, dtype: DtypeObj | None
) -> ArrayLike:
    """
    create a np.ndarray / pandas type of specified shape and dtype
    filled with values

    Parameters
    ----------
    value : scalar value
    length : int
    dtype : pandas_dtype or np.dtype

    Returns
    -------
    np.ndarray / pandas type of length, filled with value

    """

    if dtype is None:
        try:
            dtype, value = infer_dtype_from_scalar(value)
        except OutOfBoundsDatetime:
            dtype = _dtype_obj

    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        seq = [] if length == 0 else [value]
        subarr = cls._from_sequence(seq, dtype=dtype).repeat(length)

    else:
        if length and dtype.kind in "iu" and isna(value):
            # coerce if we have nan for an integer dtype
            dtype = np.dtype("float64")
        elif lib.is_np_dtype(dtype, "US"):
            # we need to coerce to object dtype to avoid
            # to allow numpy to take our string as a scalar value
            dtype = np.dtype("object")
            if not isna(value):
                value = ensure_str(value)
        elif dtype.kind in "mM":
            value = _maybe_box_and_unbox_datetimelike(value, dtype)

        subarr = np.empty(length, dtype=dtype)
        if length:
            # GH 47391: numpy > 1.24 will raise filling np.nan into int dtypes
            subarr.fill(value)

    return subarr


def _maybe_box_and_unbox_datetimelike(value: Scalar, dtype: DtypeObj):
    # Caller is responsible for checking dtype.kind in "mM"

    if isinstance(value, dt.datetime):
        # we dont want to box dt64, in particular datetime64("NaT")
        value = maybe_box_datetimelike(value, dtype)

    return _maybe_unbox_datetimelike(value, dtype)


def construct_1d_object_array_from_listlike(values: Collection) -> np.ndarray:
    """
    Transform any list-like object in a 1-dimensional numpy array of object
    dtype.

    Parameters
    ----------
    values : any iterable which has a len()

    Raises
    ------
    TypeError
        * If `values` does not have a len()

    Returns
    -------
    1-dimensional numpy array of dtype object
    """
    # numpy will try to interpret nested lists as further dimensions in np.array(),
    # hence explicitly making a 1D array using np.fromiter
    result = np.empty(len(values), dtype="object")
    for i, obj in enumerate(values):
        result[i] = obj
    return result


def maybe_cast_to_integer_array(arr: list | np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Takes any dtype and returns the casted version, raising for when data is
    incompatible with integer/unsigned integer dtypes.

    Parameters
    ----------
    arr : np.ndarray or list
        The array to cast.
    dtype : np.dtype
        The integer dtype to cast the array to.

    Returns
    -------
    ndarray
        Array of integer or unsigned integer dtype.

    Raises
    ------
    OverflowError : the dtype is incompatible with the data
    ValueError : loss of precision has occurred during casting

    Examples
    --------
    If you try to coerce negative values to unsigned integers, it raises:

    >>> pd.Series([-1], dtype="uint64")
    Traceback (most recent call last):
        ...
    OverflowError: Trying to coerce negative values to unsigned integers

    Also, if you try to coerce float values to integers, it raises:

    >>> maybe_cast_to_integer_array([1, 2, 3.5], dtype=np.dtype("int64"))
    Traceback (most recent call last):
        ...
    ValueError: Trying to coerce float values to integers
    """
    assert dtype.kind in "iu"

    try:
        if not isinstance(arr, np.ndarray):
            with warnings.catch_warnings():
                # We already disallow dtype=uint w/ negative numbers
                # (test_constructor_coercion_signed_to_unsigned) so safe to ignore.
                warnings.filterwarnings(
                    "ignore",
                    "NumPy will stop allowing conversion of out-of-bound Python int",
                    DeprecationWarning,
                )
                casted = np.asarray(arr, dtype=dtype)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                casted = arr.astype(dtype, copy=False)
    except OverflowError as err:
        raise OverflowError(
            "The elements provided in the data cannot all be "
            f"casted to the dtype {dtype}"
        ) from err

    if isinstance(arr, np.ndarray) and arr.dtype == dtype:
        # avoid expensive array_equal check
        return casted

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore", "elementwise comparison failed", FutureWarning
        )
        if np.array_equal(arr, casted):
            return casted

    # We do this casting to allow for proper
    # data and dtype checking.
    #
    # We didn't do this earlier because NumPy
    # doesn't handle `uint64` correctly.
    arr = np.asarray(arr)

    if np.issubdtype(arr.dtype, str):
        # TODO(numpy-2.0 min): This case will raise an OverflowError above
        if (casted.astype(str) == arr).all():
            return casted
        raise ValueError(f"string values cannot be losslessly cast to {dtype}")

    if dtype.kind == "u" and (arr < 0).any():
        # TODO: can this be hit anymore after numpy 2.0?
        raise OverflowError("Trying to coerce negative values to unsigned integers")

    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all():
            raise IntCastingNaNError(
                "Cannot convert non-finite values (NA or inf) to integer"
            )
        raise ValueError("Trying to coerce float values to integers")
    if arr.dtype == object:
        raise ValueError("Trying to coerce float values to integers")

    if casted.dtype < arr.dtype:
        # TODO: Can this path be hit anymore with numpy > 2
        # GH#41734 e.g. [1, 200, 923442] and dtype="int8" -> overflows
        raise ValueError(
            f"Values are too large to be losslessly converted to {dtype}. "
            f"To cast anyway, use pd.Series(values).astype({dtype})"
        )

    if arr.dtype.kind in "mM":
        # test_constructor_maskedarray_nonfloat
        raise TypeError(
            f"Constructing a Series or DataFrame from {arr.dtype} values and "
            f"dtype={dtype} is not supported. Use values.view({dtype}) instead."
        )

    # No known cases that get here, but raising explicitly to cover our bases.
    raise ValueError(f"values cannot be losslessly cast to {dtype}")


def can_hold_element(arr: ArrayLike, element: Any) -> bool:
    """
    Can we do an inplace setitem with this element in an array with this dtype?

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
    element : Any

    Returns
    -------
    bool
    """
    dtype = arr.dtype
    if not isinstance(dtype, np.dtype) or dtype.kind in "mM":
        if isinstance(dtype, (PeriodDtype, IntervalDtype, DatetimeTZDtype, np.dtype)):
            # np.dtype here catches datetime64ns and timedelta64ns; we assume
            #  in this case that we have DatetimeArray/TimedeltaArray
            arr = cast(
                "PeriodArray | DatetimeArray | TimedeltaArray | IntervalArray", arr
            )
            try:
                arr._validate_setitem_value(element)
                return True
            except (ValueError, TypeError):
                return False

        if dtype == "string":
            try:
                arr._maybe_convert_setitem_value(element)  # type: ignore[union-attr]
                return True
            except (ValueError, TypeError):
                return False

        # This is technically incorrect, but maintains the behavior of
        # ExtensionBlock._can_hold_element
        return True

    try:
        np_can_hold_element(dtype, element)
        return True
    except (TypeError, LossySetitemError):
        return False


def np_can_hold_element(dtype: np.dtype, element: Any) -> Any:
    """
    Raise if we cannot losslessly set this element into an ndarray with this dtype.

    Specifically about places where we disagree with numpy.  i.e. there are
    cases where numpy will raise in doing the setitem that we do not check
    for here, e.g. setting str "X" into a numeric ndarray.

    Returns
    -------
    Any
        The element, potentially cast to the dtype.

    Raises
    ------
    ValueError : If we cannot losslessly store this element with this dtype.
    """
    if dtype == _dtype_obj:
        return element

    tipo = _maybe_infer_dtype_type(element)

    if dtype.kind in "iu":
        if isinstance(element, range):
            if _dtype_can_hold_range(element, dtype):
                return element
            raise LossySetitemError

        if is_integer(element) or (is_float(element) and element.is_integer()):
            # e.g. test_setitem_series_int8 if we have a python int 1
            #  tipo may be np.int32, despite the fact that it will fit
            #  in smaller int dtypes.
            info = np.iinfo(dtype)
            if info.min <= element <= info.max:
                return dtype.type(element)
            raise LossySetitemError

        if tipo is not None:
            if tipo.kind not in "iu":
                if isinstance(element, np.ndarray) and element.dtype.kind == "f":
                    # If all can be losslessly cast to integers, then we can hold them
                    with np.errstate(invalid="ignore"):
                        # We check afterwards if cast was losslessly, so no need to show
                        # the warning
                        casted = element.astype(dtype)
                    comp = casted == element
                    if comp.all():
                        # Return the casted values bc they can be passed to
                        #  np.putmask, whereas the raw values cannot.
                        #  see TestSetitemFloatNDarrayIntoIntegerSeries
                        return casted
                    raise LossySetitemError

                elif isinstance(element, ABCExtensionArray) and isinstance(
                    element.dtype, CategoricalDtype
                ):
                    # GH#52927 setting Categorical value into non-EA frame
                    # TODO: general-case for EAs?
                    try:
                        casted = element.astype(dtype)
                    except (ValueError, TypeError):
                        raise LossySetitemError
                    # Check for cases of either
                    #  a) lossy overflow/rounding or
                    #  b) semantic changes like dt64->int64
                    comp = casted == element
                    if not comp.all():
                        raise LossySetitemError
                    return casted

                # Anything other than integer we cannot hold
                raise LossySetitemError
            if (
                dtype.kind == "u"
                and isinstance(element, np.ndarray)
                and element.dtype.kind == "i"
            ):
                # see test_where_uint64
                casted = element.astype(dtype)
                if (casted == element).all():
                    # TODO: faster to check (element >=0).all()?  potential
                    #  itemsize issues there?
                    return casted
                raise LossySetitemError
            if dtype.itemsize < tipo.itemsize:
                raise LossySetitemError
            if not isinstance(tipo, np.dtype):
                # i.e. nullable IntegerDtype; we can put this into an ndarray
                #  losslessly iff it has no NAs
                arr = element._values if isinstance(element, ABCSeries) else element
                if arr._hasna:
                    raise LossySetitemError
                return element

            return element

        raise LossySetitemError

    if dtype.kind == "f":
        if lib.is_integer(element) or lib.is_float(element):
            casted = dtype.type(element)
            if np.isnan(casted) or casted == element:
                return casted
            # otherwise e.g. overflow see TestCoercionFloat32
            raise LossySetitemError

        if tipo is not None:
            # TODO: itemsize check?
            if tipo.kind not in "iuf":
                # Anything other than float/integer we cannot hold
                raise LossySetitemError
            if not isinstance(tipo, np.dtype):
                # i.e. nullable IntegerDtype or FloatingDtype;
                #  we can put this into an ndarray losslessly iff it has no NAs
                if element._hasna:
                    raise LossySetitemError
                return element
            elif tipo.itemsize > dtype.itemsize or tipo.kind != dtype.kind:
                if isinstance(element, np.ndarray):
                    # e.g. TestDataFrameIndexingWhere::test_where_alignment
                    casted = element.astype(dtype)
                    if np.array_equal(casted, element, equal_nan=True):
                        return casted
                    raise LossySetitemError

            return element

        raise LossySetitemError

    if dtype.kind == "c":
        if lib.is_integer(element) or lib.is_complex(element) or lib.is_float(element):
            if np.isnan(element):
                # see test_where_complex GH#6345
                return dtype.type(element)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                casted = dtype.type(element)
            if casted == element:
                return casted
            # otherwise e.g. overflow see test_32878_complex_itemsize
            raise LossySetitemError

        if tipo is not None:
            if tipo.kind in "iufc":
                return element
            raise LossySetitemError
        raise LossySetitemError

    if dtype.kind == "b":
        if tipo is not None:
            if tipo.kind == "b":
                if not isinstance(tipo, np.dtype):
                    # i.e. we have a BooleanArray
                    if element._hasna:
                        # i.e. there are pd.NA elements
                        raise LossySetitemError
                return element
            raise LossySetitemError
        if lib.is_bool(element):
            return element
        raise LossySetitemError

    if dtype.kind == "S":
        # TODO: test tests.frame.methods.test_replace tests get here,
        #  need more targeted tests.  xref phofl has a PR about this
        if tipo is not None:
            if tipo.kind == "S" and tipo.itemsize <= dtype.itemsize:
                return element
            raise LossySetitemError
        if isinstance(element, bytes) and len(element) <= dtype.itemsize:
            return element
        raise LossySetitemError

    if dtype.kind == "V":
        # i.e. np.void, which cannot hold _anything_
        raise LossySetitemError

    raise NotImplementedError(dtype)


def _dtype_can_hold_range(rng: range, dtype: np.dtype) -> bool:
    """
    _maybe_infer_dtype_type infers to int64 (and float64 for very large endpoints),
    but in many cases a range can be held by a smaller integer dtype.
    Check if this is one of those cases.
    """
    if not len(rng):
        return True
    return np_can_cast_scalar(rng.start, dtype) and np_can_cast_scalar(rng.stop, dtype)


def np_can_cast_scalar(element: Scalar, dtype: np.dtype) -> bool:
    """
    np.can_cast pandas-equivalent for pre 2-0 behavior that allowed scalar
    inference

    Parameters
    ----------
    element : Scalar
    dtype : np.dtype

    Returns
    -------
    bool
    """
    try:
        np_can_hold_element(dtype, element)
        return True
    except (LossySetitemError, NotImplementedError):
        return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\util.py ===
#
# Copyright (C) 2012-2023 The Python Software Foundation.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
import codecs
from collections import deque
import contextlib
import csv
from glob import iglob as std_iglob
import io
import json
import logging
import os
import py_compile
import re
import socket
try:
    import ssl
except ImportError:  # pragma: no cover
    ssl = None
import subprocess
import sys
import tarfile
import tempfile
import textwrap

try:
    import threading
except ImportError:  # pragma: no cover
    import dummy_threading as threading
import time

from . import DistlibException
from .compat import (string_types, text_type, shutil, raw_input, StringIO, cache_from_source, urlopen, urljoin, httplib,
                     xmlrpclib, HTTPHandler, BaseConfigurator, valid_ident, Container, configparser, URLError, ZipFile,
                     fsdecode, unquote, urlparse)

logger = logging.getLogger(__name__)

#
# Requirement parsing code as per PEP 508
#

IDENTIFIER = re.compile(r'^([\w\.-]+)\s*')
VERSION_IDENTIFIER = re.compile(r'^([\w\.*+-]+)\s*')
COMPARE_OP = re.compile(r'^(<=?|>=?|={2,3}|[~!]=)\s*')
MARKER_OP = re.compile(r'^((<=?)|(>=?)|={2,3}|[~!]=|in|not\s+in)\s*')
OR = re.compile(r'^or\b\s*')
AND = re.compile(r'^and\b\s*')
NON_SPACE = re.compile(r'(\S+)\s*')
STRING_CHUNK = re.compile(r'([\s\w\.{}()*+#:;,/?!~`@$%^&=|<>\[\]-]+)')


def parse_marker(marker_string):
    """
    Parse a marker string and return a dictionary containing a marker expression.

    The dictionary will contain keys "op", "lhs" and "rhs" for non-terminals in
    the expression grammar, or strings. A string contained in quotes is to be
    interpreted as a literal string, and a string not contained in quotes is a
    variable (such as os_name).
    """

    def marker_var(remaining):
        # either identifier, or literal string
        m = IDENTIFIER.match(remaining)
        if m:
            result = m.groups()[0]
            remaining = remaining[m.end():]
        elif not remaining:
            raise SyntaxError('unexpected end of input')
        else:
            q = remaining[0]
            if q not in '\'"':
                raise SyntaxError('invalid expression: %s' % remaining)
            oq = '\'"'.replace(q, '')
            remaining = remaining[1:]
            parts = [q]
            while remaining:
                # either a string chunk, or oq, or q to terminate
                if remaining[0] == q:
                    break
                elif remaining[0] == oq:
                    parts.append(oq)
                    remaining = remaining[1:]
                else:
                    m = STRING_CHUNK.match(remaining)
                    if not m:
                        raise SyntaxError('error in string literal: %s' % remaining)
                    parts.append(m.groups()[0])
                    remaining = remaining[m.end():]
            else:
                s = ''.join(parts)
                raise SyntaxError('unterminated string: %s' % s)
            parts.append(q)
            result = ''.join(parts)
            remaining = remaining[1:].lstrip()  # skip past closing quote
        return result, remaining

    def marker_expr(remaining):
        if remaining and remaining[0] == '(':
            result, remaining = marker(remaining[1:].lstrip())
            if remaining[0] != ')':
                raise SyntaxError('unterminated parenthesis: %s' % remaining)
            remaining = remaining[1:].lstrip()
        else:
            lhs, remaining = marker_var(remaining)
            while remaining:
                m = MARKER_OP.match(remaining)
                if not m:
                    break
                op = m.groups()[0]
                remaining = remaining[m.end():]
                rhs, remaining = marker_var(remaining)
                lhs = {'op': op, 'lhs': lhs, 'rhs': rhs}
            result = lhs
        return result, remaining

    def marker_and(remaining):
        lhs, remaining = marker_expr(remaining)
        while remaining:
            m = AND.match(remaining)
            if not m:
                break
            remaining = remaining[m.end():]
            rhs, remaining = marker_expr(remaining)
            lhs = {'op': 'and', 'lhs': lhs, 'rhs': rhs}
        return lhs, remaining

    def marker(remaining):
        lhs, remaining = marker_and(remaining)
        while remaining:
            m = OR.match(remaining)
            if not m:
                break
            remaining = remaining[m.end():]
            rhs, remaining = marker_and(remaining)
            lhs = {'op': 'or', 'lhs': lhs, 'rhs': rhs}
        return lhs, remaining

    return marker(marker_string)


def parse_requirement(req):
    """
    Parse a requirement passed in as a string. Return a Container
    whose attributes contain the various parts of the requirement.
    """
    remaining = req.strip()
    if not remaining or remaining.startswith('#'):
        return None
    m = IDENTIFIER.match(remaining)
    if not m:
        raise SyntaxError('name expected: %s' % remaining)
    distname = m.groups()[0]
    remaining = remaining[m.end():]
    extras = mark_expr = versions = uri = None
    if remaining and remaining[0] == '[':
        i = remaining.find(']', 1)
        if i < 0:
            raise SyntaxError('unterminated extra: %s' % remaining)
        s = remaining[1:i]
        remaining = remaining[i + 1:].lstrip()
        extras = []
        while s:
            m = IDENTIFIER.match(s)
            if not m:
                raise SyntaxError('malformed extra: %s' % s)
            extras.append(m.groups()[0])
            s = s[m.end():]
            if not s:
                break
            if s[0] != ',':
                raise SyntaxError('comma expected in extras: %s' % s)
            s = s[1:].lstrip()
        if not extras:
            extras = None
    if remaining:
        if remaining[0] == '@':
            # it's a URI
            remaining = remaining[1:].lstrip()
            m = NON_SPACE.match(remaining)
            if not m:
                raise SyntaxError('invalid URI: %s' % remaining)
            uri = m.groups()[0]
            t = urlparse(uri)
            # there are issues with Python and URL parsing, so this test
            # is a bit crude. See bpo-20271, bpo-23505. Python doesn't
            # always parse invalid URLs correctly - it should raise
            # exceptions for malformed URLs
            if not (t.scheme and t.netloc):
                raise SyntaxError('Invalid URL: %s' % uri)
            remaining = remaining[m.end():].lstrip()
        else:

            def get_versions(ver_remaining):
                """
                Return a list of operator, version tuples if any are
                specified, else None.
                """
                m = COMPARE_OP.match(ver_remaining)
                versions = None
                if m:
                    versions = []
                    while True:
                        op = m.groups()[0]
                        ver_remaining = ver_remaining[m.end():]
                        m = VERSION_IDENTIFIER.match(ver_remaining)
                        if not m:
                            raise SyntaxError('invalid version: %s' % ver_remaining)
                        v = m.groups()[0]
                        versions.append((op, v))
                        ver_remaining = ver_remaining[m.end():]
                        if not ver_remaining or ver_remaining[0] != ',':
                            break
                        ver_remaining = ver_remaining[1:].lstrip()
                        # Some packages have a trailing comma which would break things
                        # See issue #148
                        if not ver_remaining:
                            break
                        m = COMPARE_OP.match(ver_remaining)
                        if not m:
                            raise SyntaxError('invalid constraint: %s' % ver_remaining)
                    if not versions:
                        versions = None
                return versions, ver_remaining

            if remaining[0] != '(':
                versions, remaining = get_versions(remaining)
            else:
                i = remaining.find(')', 1)
                if i < 0:
                    raise SyntaxError('unterminated parenthesis: %s' % remaining)
                s = remaining[1:i]
                remaining = remaining[i + 1:].lstrip()
                # As a special diversion from PEP 508, allow a version number
                # a.b.c in parentheses as a synonym for ~= a.b.c (because this
                # is allowed in earlier PEPs)
                if COMPARE_OP.match(s):
                    versions, _ = get_versions(s)
                else:
                    m = VERSION_IDENTIFIER.match(s)
                    if not m:
                        raise SyntaxError('invalid constraint: %s' % s)
                    v = m.groups()[0]
                    s = s[m.end():].lstrip()
                    if s:
                        raise SyntaxError('invalid constraint: %s' % s)
                    versions = [('~=', v)]

    if remaining:
        if remaining[0] != ';':
            raise SyntaxError('invalid requirement: %s' % remaining)
        remaining = remaining[1:].lstrip()

        mark_expr, remaining = parse_marker(remaining)

    if remaining and remaining[0] != '#':
        raise SyntaxError('unexpected trailing data: %s' % remaining)

    if not versions:
        rs = distname
    else:
        rs = '%s %s' % (distname, ', '.join(['%s %s' % con for con in versions]))
    return Container(name=distname, extras=extras, constraints=versions, marker=mark_expr, url=uri, requirement=rs)


def get_resources_dests(resources_root, rules):
    """Find destinations for resources files"""

    def get_rel_path(root, path):
        # normalizes and returns a lstripped-/-separated path
        root = root.replace(os.path.sep, '/')
        path = path.replace(os.path.sep, '/')
        assert path.startswith(root)
        return path[len(root):].lstrip('/')

    destinations = {}
    for base, suffix, dest in rules:
        prefix = os.path.join(resources_root, base)
        for abs_base in iglob(prefix):
            abs_glob = os.path.join(abs_base, suffix)
            for abs_path in iglob(abs_glob):
                resource_file = get_rel_path(resources_root, abs_path)
                if dest is None:  # remove the entry if it was here
                    destinations.pop(resource_file, None)
                else:
                    rel_path = get_rel_path(abs_base, abs_path)
                    rel_dest = dest.replace(os.path.sep, '/').rstrip('/')
                    destinations[resource_file] = rel_dest + '/' + rel_path
    return destinations


def in_venv():
    if hasattr(sys, 'real_prefix'):
        # virtualenv venvs
        result = True
    else:
        # PEP 405 venvs
        result = sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
    return result


def get_executable():
    # The __PYVENV_LAUNCHER__ dance is apparently no longer needed, as
    # changes to the stub launcher mean that sys.executable always points
    # to the stub on OS X
    #    if sys.platform == 'darwin' and ('__PYVENV_LAUNCHER__'
    #                                     in os.environ):
    #        result =  os.environ['__PYVENV_LAUNCHER__']
    #    else:
    #        result = sys.executable
    #    return result
    # Avoid normcasing: see issue #143
    # result = os.path.normcase(sys.executable)
    result = sys.executable
    if not isinstance(result, text_type):
        result = fsdecode(result)
    return result


def proceed(prompt, allowed_chars, error_prompt=None, default=None):
    p = prompt
    while True:
        s = raw_input(p)
        p = prompt
        if not s and default:
            s = default
        if s:
            c = s[0].lower()
            if c in allowed_chars:
                break
            if error_prompt:
                p = '%c: %s\n%s' % (c, error_prompt, prompt)
    return c


def extract_by_key(d, keys):
    if isinstance(keys, string_types):
        keys = keys.split()
    result = {}
    for key in keys:
        if key in d:
            result[key] = d[key]
    return result


def read_exports(stream):
    if sys.version_info[0] >= 3:
        # needs to be a text stream
        stream = codecs.getreader('utf-8')(stream)
    # Try to load as JSON, falling back on legacy format
    data = stream.read()
    stream = StringIO(data)
    try:
        jdata = json.load(stream)
        result = jdata['extensions']['python.exports']['exports']
        for group, entries in result.items():
            for k, v in entries.items():
                s = '%s = %s' % (k, v)
                entry = get_export_entry(s)
                assert entry is not None
                entries[k] = entry
        return result
    except Exception:
        stream.seek(0, 0)

    def read_stream(cp, stream):
        if hasattr(cp, 'read_file'):
            cp.read_file(stream)
        else:
            cp.readfp(stream)

    cp = configparser.ConfigParser()
    try:
        read_stream(cp, stream)
    except configparser.MissingSectionHeaderError:
        stream.close()
        data = textwrap.dedent(data)
        stream = StringIO(data)
        read_stream(cp, stream)

    result = {}
    for key in cp.sections():
        result[key] = entries = {}
        for name, value in cp.items(key):
            s = '%s = %s' % (name, value)
            entry = get_export_entry(s)
            assert entry is not None
            # entry.dist = self
            entries[name] = entry
    return result


def write_exports(exports, stream):
    if sys.version_info[0] >= 3:
        # needs to be a text stream
        stream = codecs.getwriter('utf-8')(stream)
    cp = configparser.ConfigParser()
    for k, v in exports.items():
        # TODO check k, v for valid values
        cp.add_section(k)
        for entry in v.values():
            if entry.suffix is None:
                s = entry.prefix
            else:
                s = '%s:%s' % (entry.prefix, entry.suffix)
            if entry.flags:
                s = '%s [%s]' % (s, ', '.join(entry.flags))
            cp.set(k, entry.name, s)
    cp.write(stream)


@contextlib.contextmanager
def tempdir():
    td = tempfile.mkdtemp()
    try:
        yield td
    finally:
        shutil.rmtree(td)


@contextlib.contextmanager
def chdir(d):
    cwd = os.getcwd()
    try:
        os.chdir(d)
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def socket_timeout(seconds=15):
    cto = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(seconds)
        yield
    finally:
        socket.setdefaulttimeout(cto)


class cached_property(object):

    def __init__(self, func):
        self.func = func
        # for attr in ('__name__', '__module__', '__doc__'):
        #     setattr(self, attr, getattr(func, attr, None))

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        value = self.func(obj)
        object.__setattr__(obj, self.func.__name__, value)
        # obj.__dict__[self.func.__name__] = value = self.func(obj)
        return value


def convert_path(pathname):
    """Return 'pathname' as a name that will work on the native filesystem.

    The path is split on '/' and put back together again using the current
    directory separator.  Needed because filenames in the setup script are
    always supplied in Unix style, and have to be converted to the local
    convention before we can actually use them in the filesystem.  Raises
    ValueError on non-Unix-ish systems if 'pathname' either starts or
    ends with a slash.
    """
    if os.sep == '/':
        return pathname
    if not pathname:
        return pathname
    if pathname[0] == '/':
        raise ValueError("path '%s' cannot be absolute" % pathname)
    if pathname[-1] == '/':
        raise ValueError("path '%s' cannot end with '/'" % pathname)

    paths = pathname.split('/')
    while os.curdir in paths:
        paths.remove(os.curdir)
    if not paths:
        return os.curdir
    return os.path.join(*paths)


class FileOperator(object):

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.ensured = set()
        self._init_record()

    def _init_record(self):
        self.record = False
        self.files_written = set()
        self.dirs_created = set()

    def record_as_written(self, path):
        if self.record:
            self.files_written.add(path)

    def newer(self, source, target):
        """Tell if the target is newer than the source.

        Returns true if 'source' exists and is more recently modified than
        'target', or if 'source' exists and 'target' doesn't.

        Returns false if both exist and 'target' is the same age or younger
        than 'source'. Raise PackagingFileError if 'source' does not exist.

        Note that this test is not very accurate: files created in the same
        second will have the same "age".
        """
        if not os.path.exists(source):
            raise DistlibException("file '%r' does not exist" % os.path.abspath(source))
        if not os.path.exists(target):
            return True

        return os.stat(source).st_mtime > os.stat(target).st_mtime

    def copy_file(self, infile, outfile, check=True):
        """Copy a file respecting dry-run and force flags.
        """
        self.ensure_dir(os.path.dirname(outfile))
        logger.info('Copying %s to %s', infile, outfile)
        if not self.dry_run:
            msg = None
            if check:
                if os.path.islink(outfile):
                    msg = '%s is a symlink' % outfile
                elif os.path.exists(outfile) and not os.path.isfile(outfile):
                    msg = '%s is a non-regular file' % outfile
            if msg:
                raise ValueError(msg + ' which would be overwritten')
            shutil.copyfile(infile, outfile)
        self.record_as_written(outfile)

    def copy_stream(self, instream, outfile, encoding=None):
        assert not os.path.isdir(outfile)
        self.ensure_dir(os.path.dirname(outfile))
        logger.info('Copying stream %s to %s', instream, outfile)
        if not self.dry_run:
            if encoding is None:
                outstream = open(outfile, 'wb')
            else:
                outstream = codecs.open(outfile, 'w', encoding=encoding)
            try:
                shutil.copyfileobj(instream, outstream)
            finally:
                outstream.close()
        self.record_as_written(outfile)

    def write_binary_file(self, path, data):
        self.ensure_dir(os.path.dirname(path))
        if not self.dry_run:
            if os.path.exists(path):
                os.remove(path)
            with open(path, 'wb') as f:
                f.write(data)
        self.record_as_written(path)

    def write_text_file(self, path, data, encoding):
        self.write_binary_file(path, data.encode(encoding))

    def set_mode(self, bits, mask, files):
        if os.name == 'posix' or (os.name == 'java' and os._name == 'posix'):
            # Set the executable bits (owner, group, and world) on
            # all the files specified.
            for f in files:
                if self.dry_run:
                    logger.info("changing mode of %s", f)
                else:
                    mode = (os.stat(f).st_mode | bits) & mask
                    logger.info("changing mode of %s to %o", f, mode)
                    os.chmod(f, mode)

    set_executable_mode = lambda s, f: s.set_mode(0o555, 0o7777, f)

    def ensure_dir(self, path):
        path = os.path.abspath(path)
        if path not in self.ensured and not os.path.exists(path):
            self.ensured.add(path)
            d, f = os.path.split(path)
            self.ensure_dir(d)
            logger.info('Creating %s' % path)
            if not self.dry_run:
                os.mkdir(path)
            if self.record:
                self.dirs_created.add(path)

    def byte_compile(self, path, optimize=False, force=False, prefix=None, hashed_invalidation=False):
        dpath = cache_from_source(path, not optimize)
        logger.info('Byte-compiling %s to %s', path, dpath)
        if not self.dry_run:
            if force or self.newer(path, dpath):
                if not prefix:
                    diagpath = None
                else:
                    assert path.startswith(prefix)
                    diagpath = path[len(prefix):]
            compile_kwargs = {}
            if hashed_invalidation and hasattr(py_compile, 'PycInvalidationMode'):
                if not isinstance(hashed_invalidation, py_compile.PycInvalidationMode):
                    hashed_invalidation = py_compile.PycInvalidationMode.CHECKED_HASH
                compile_kwargs['invalidation_mode'] = hashed_invalidation
            py_compile.compile(path, dpath, diagpath, True, **compile_kwargs)  # raise error
        self.record_as_written(dpath)
        return dpath

    def ensure_removed(self, path):
        if os.path.exists(path):
            if os.path.isdir(path) and not os.path.islink(path):
                logger.debug('Removing directory tree at %s', path)
                if not self.dry_run:
                    shutil.rmtree(path)
                if self.record:
                    if path in self.dirs_created:
                        self.dirs_created.remove(path)
            else:
                if os.path.islink(path):
                    s = 'link'
                else:
                    s = 'file'
                logger.debug('Removing %s %s', s, path)
                if not self.dry_run:
                    os.remove(path)
                if self.record:
                    if path in self.files_written:
                        self.files_written.remove(path)

    def is_writable(self, path):
        result = False
        while not result:
            if os.path.exists(path):
                result = os.access(path, os.W_OK)
                break
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        return result

    def commit(self):
        """
        Commit recorded changes, turn off recording, return
        changes.
        """
        assert self.record
        result = self.files_written, self.dirs_created
        self._init_record()
        return result

    def rollback(self):
        if not self.dry_run:
            for f in list(self.files_written):
                if os.path.exists(f):
                    os.remove(f)
            # dirs should all be empty now, except perhaps for
            # __pycache__ subdirs
            # reverse so that subdirs appear before their parents
            dirs = sorted(self.dirs_created, reverse=True)
            for d in dirs:
                flist = os.listdir(d)
                if flist:
                    assert flist == ['__pycache__']
                    sd = os.path.join(d, flist[0])
                    os.rmdir(sd)
                os.rmdir(d)  # should fail if non-empty
        self._init_record()


def resolve(module_name, dotted_path):
    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        mod = __import__(module_name)
    if dotted_path is None:
        result = mod
    else:
        parts = dotted_path.split('.')
        result = getattr(mod, parts.pop(0))
        for p in parts:
            result = getattr(result, p)
    return result


class ExportEntry(object):

    def __init__(self, name, prefix, suffix, flags):
        self.name = name
        self.prefix = prefix
        self.suffix = suffix
        self.flags = flags

    @cached_property
    def value(self):
        return resolve(self.prefix, self.suffix)

    def __repr__(self):  # pragma: no cover
        return '<ExportEntry %s = %s:%s %s>' % (self.name, self.prefix, self.suffix, self.flags)

    def __eq__(self, other):
        if not isinstance(other, ExportEntry):
            result = False
        else:
            result = (self.name == other.name and self.prefix == other.prefix and self.suffix == other.suffix and
                      self.flags == other.flags)
        return result

    __hash__ = object.__hash__


ENTRY_RE = re.compile(
    r'''(?P<name>([^\[]\S*))
                      \s*=\s*(?P<callable>(\w+)([:\.]\w+)*)
                      \s*(\[\s*(?P<flags>[\w-]+(=\w+)?(,\s*\w+(=\w+)?)*)\s*\])?
                      ''', re.VERBOSE)


def get_export_entry(specification):
    m = ENTRY_RE.search(specification)
    if not m:
        result = None
        if '[' in specification or ']' in specification:
            raise DistlibException("Invalid specification "
                                   "'%s'" % specification)
    else:
        d = m.groupdict()
        name = d['name']
        path = d['callable']
        colons = path.count(':')
        if colons == 0:
            prefix, suffix = path, None
        else:
            if colons != 1:
                raise DistlibException("Invalid specification "
                                       "'%s'" % specification)
            prefix, suffix = path.split(':')
        flags = d['flags']
        if flags is None:
            if '[' in specification or ']' in specification:
                raise DistlibException("Invalid specification "
                                       "'%s'" % specification)
            flags = []
        else:
            flags = [f.strip() for f in flags.split(',')]
        result = ExportEntry(name, prefix, suffix, flags)
    return result


def get_cache_base(suffix=None):
    """
    Return the default base location for distlib caches. If the directory does
    not exist, it is created. Use the suffix provided for the base directory,
    and default to '.distlib' if it isn't provided.

    On Windows, if LOCALAPPDATA is defined in the environment, then it is
    assumed to be a directory, and will be the parent directory of the result.
    On POSIX, and on Windows if LOCALAPPDATA is not defined, the user's home
    directory - using os.expanduser('~') - will be the parent directory of
    the result.

    The result is just the directory '.distlib' in the parent directory as
    determined above, or with the name specified with ``suffix``.
    """
    if suffix is None:
        suffix = '.distlib'
    if os.name == 'nt' and 'LOCALAPPDATA' in os.environ:
        result = os.path.expandvars('$localappdata')
    else:
        # Assume posix, or old Windows
        result = os.path.expanduser('~')
    # we use 'isdir' instead of 'exists', because we want to
    # fail if there's a file with that name
    if os.path.isdir(result):
        usable = os.access(result, os.W_OK)
        if not usable:
            logger.warning('Directory exists but is not writable: %s', result)
    else:
        try:
            os.makedirs(result)
            usable = True
        except OSError:
            logger.warning('Unable to create %s', result, exc_info=True)
            usable = False
    if not usable:
        result = tempfile.mkdtemp()
        logger.warning('Default location unusable, using %s', result)
    return os.path.join(result, suffix)


def path_to_cache_dir(path, use_abspath=True):
    """
    Convert an absolute path to a directory name for use in a cache.

    The algorithm used is:

    #. On Windows, any ``':'`` in the drive is replaced with ``'---'``.
    #. Any occurrence of ``os.sep`` is replaced with ``'--'``.
    #. ``'.cache'`` is appended.
    """
    d, p = os.path.splitdrive(os.path.abspath(path) if use_abspath else path)
    if d:
        d = d.replace(':', '---')
    p = p.replace(os.sep, '--')
    return d + p + '.cache'


def ensure_slash(s):
    if not s.endswith('/'):
        return s + '/'
    return s


def parse_credentials(netloc):
    username = password = None
    if '@' in netloc:
        prefix, netloc = netloc.rsplit('@', 1)
        if ':' not in prefix:
            username = prefix
        else:
            username, password = prefix.split(':', 1)
    if username:
        username = unquote(username)
    if password:
        password = unquote(password)
    return username, password, netloc


def get_process_umask():
    result = os.umask(0o22)
    os.umask(result)
    return result


def is_string_sequence(seq):
    result = True
    i = None
    for i, s in enumerate(seq):
        if not isinstance(s, string_types):
            result = False
            break
    assert i is not None
    return result


PROJECT_NAME_AND_VERSION = re.compile('([a-z0-9_]+([.-][a-z_][a-z0-9_]*)*)-'
                                      '([a-z0-9_.+-]+)', re.I)
PYTHON_VERSION = re.compile(r'-py(\d\.?\d?)')


def split_filename(filename, project_name=None):
    """
    Extract name, version, python version from a filename (no extension)

    Return name, version, pyver or None
    """
    result = None
    pyver = None
    filename = unquote(filename).replace(' ', '-')
    m = PYTHON_VERSION.search(filename)
    if m:
        pyver = m.group(1)
        filename = filename[:m.start()]
    if project_name and len(filename) > len(project_name) + 1:
        m = re.match(re.escape(project_name) + r'\b', filename)
        if m:
            n = m.end()
            result = filename[:n], filename[n + 1:], pyver
    if result is None:
        m = PROJECT_NAME_AND_VERSION.match(filename)
        if m:
            result = m.group(1), m.group(3), pyver
    return result


# Allow spaces in name because of legacy dists like "Twisted Core"
NAME_VERSION_RE = re.compile(r'(?P<name>[\w .-]+)\s*'
                             r'\(\s*(?P<ver>[^\s)]+)\)$')


def parse_name_and_version(p):
    """
    A utility method used to get name and version from a string.

    From e.g. a Provides-Dist value.

    :param p: A value in a form 'foo (1.0)'
    :return: The name and version as a tuple.
    """
    m = NAME_VERSION_RE.match(p)
    if not m:
        raise DistlibException('Ill-formed name/version string: \'%s\'' % p)
    d = m.groupdict()
    return d['name'].strip().lower(), d['ver']


def get_extras(requested, available):
    result = set()
    requested = set(requested or [])
    available = set(available or [])
    if '*' in requested:
        requested.remove('*')
        result |= available
    for r in requested:
        if r == '-':
            result.add(r)
        elif r.startswith('-'):
            unwanted = r[1:]
            if unwanted not in available:
                logger.warning('undeclared extra: %s' % unwanted)
            if unwanted in result:
                result.remove(unwanted)
        else:
            if r not in available:
                logger.warning('undeclared extra: %s' % r)
            result.add(r)
    return result


#
# Extended metadata functionality
#


def _get_external_data(url):
    result = {}
    try:
        # urlopen might fail if it runs into redirections,
        # because of Python issue #13696. Fixed in locators
        # using a custom redirect handler.
        resp = urlopen(url)
        headers = resp.info()
        ct = headers.get('Content-Type')
        if not ct.startswith('application/json'):
            logger.debug('Unexpected response for JSON request: %s', ct)
        else:
            reader = codecs.getreader('utf-8')(resp)
            # data = reader.read().decode('utf-8')
            # result = json.loads(data)
            result = json.load(reader)
    except Exception as e:
        logger.exception('Failed to get external data for %s: %s', url, e)
    return result


_external_data_base_url = 'https://www.red-dove.com/pypi/projects/'


def get_project_data(name):
    url = '%s/%s/project.json' % (name[0].upper(), name)
    url = urljoin(_external_data_base_url, url)
    result = _get_external_data(url)
    return result


def get_package_data(name, version):
    url = '%s/%s/package-%s.json' % (name[0].upper(), name, version)
    url = urljoin(_external_data_base_url, url)
    return _get_external_data(url)


class Cache(object):
    """
    A class implementing a cache for resources that need to live in the file system
    e.g. shared libraries. This class was moved from resources to here because it
    could be used by other modules, e.g. the wheel module.
    """

    def __init__(self, base):
        """
        Initialise an instance.

        :param base: The base directory where the cache should be located.
        """
        # we use 'isdir' instead of 'exists', because we want to
        # fail if there's a file with that name
        if not os.path.isdir(base):  # pragma: no cover
            os.makedirs(base)
        if (os.stat(base).st_mode & 0o77) != 0:
            logger.warning('Directory \'%s\' is not private', base)
        self.base = os.path.abspath(os.path.normpath(base))

    def prefix_to_dir(self, prefix, use_abspath=True):
        """
        Converts a resource prefix to a directory name in the cache.
        """
        return path_to_cache_dir(prefix, use_abspath=use_abspath)

    def clear(self):
        """
        Clear the cache.
        """
        not_removed = []
        for fn in os.listdir(self.base):
            fn = os.path.join(self.base, fn)
            try:
                if os.path.islink(fn) or os.path.isfile(fn):
                    os.remove(fn)
                elif os.path.isdir(fn):
                    shutil.rmtree(fn)
            except Exception:
                not_removed.append(fn)
        return not_removed


class EventMixin(object):
    """
    A very simple publish/subscribe system.
    """

    def __init__(self):
        self._subscribers = {}

    def add(self, event, subscriber, append=True):
        """
        Add a subscriber for an event.

        :param event: The name of an event.
        :param subscriber: The subscriber to be added (and called when the
                           event is published).
        :param append: Whether to append or prepend the subscriber to an
                       existing subscriber list for the event.
        """
        subs = self._subscribers
        if event not in subs:
            subs[event] = deque([subscriber])
        else:
            sq = subs[event]
            if append:
                sq.append(subscriber)
            else:
                sq.appendleft(subscriber)

    def remove(self, event, subscriber):
        """
        Remove a subscriber for an event.

        :param event: The name of an event.
        :param subscriber: The subscriber to be removed.
        """
        subs = self._subscribers
        if event not in subs:
            raise ValueError('No subscribers: %r' % event)
        subs[event].remove(subscriber)

    def get_subscribers(self, event):
        """
        Return an iterator for the subscribers for an event.
        :param event: The event to return subscribers for.
        """
        return iter(self._subscribers.get(event, ()))

    def publish(self, event, *args, **kwargs):
        """
        Publish a event and return a list of values returned by its
        subscribers.

        :param event: The event to publish.
        :param args: The positional arguments to pass to the event's
                     subscribers.
        :param kwargs: The keyword arguments to pass to the event's
                       subscribers.
        """
        result = []
        for subscriber in self.get_subscribers(event):
            try:
                value = subscriber(event, *args, **kwargs)
            except Exception:
                logger.exception('Exception during event publication')
                value = None
            result.append(value)
        logger.debug('publish %s: args = %s, kwargs = %s, result = %s', event, args, kwargs, result)
        return result


#
# Simple sequencing
#
class Sequencer(object):

    def __init__(self):
        self._preds = {}
        self._succs = {}
        self._nodes = set()  # nodes with no preds/succs

    def add_node(self, node):
        self._nodes.add(node)

    def remove_node(self, node, edges=False):
        if node in self._nodes:
            self._nodes.remove(node)
        if edges:
            for p in set(self._preds.get(node, ())):
                self.remove(p, node)
            for s in set(self._succs.get(node, ())):
                self.remove(node, s)
            # Remove empties
            for k, v in list(self._preds.items()):
                if not v:
                    del self._preds[k]
            for k, v in list(self._succs.items()):
                if not v:
                    del self._succs[k]

    def add(self, pred, succ):
        assert pred != succ
        self._preds.setdefault(succ, set()).add(pred)
        self._succs.setdefault(pred, set()).add(succ)

    def remove(self, pred, succ):
        assert pred != succ
        try:
            preds = self._preds[succ]
            succs = self._succs[pred]
        except KeyError:  # pragma: no cover
            raise ValueError('%r not a successor of anything' % succ)
        try:
            preds.remove(pred)
            succs.remove(succ)
        except KeyError:  # pragma: no cover
            raise ValueError('%r not a successor of %r' % (succ, pred))

    def is_step(self, step):
        return (step in self._preds or step in self._succs or step in self._nodes)

    def get_steps(self, final):
        if not self.is_step(final):
            raise ValueError('Unknown: %r' % final)
        result = []
        todo = []
        seen = set()
        todo.append(final)
        while todo:
            step = todo.pop(0)
            if step in seen:
                # if a step was already seen,
                # move it to the end (so it will appear earlier
                # when reversed on return) ... but not for the
                # final step, as that would be confusing for
                # users
                if step != final:
                    result.remove(step)
                    result.append(step)
            else:
                seen.add(step)
                result.append(step)
                preds = self._preds.get(step, ())
                todo.extend(preds)
        return reversed(result)

    @property
    def strong_connections(self):
        # http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        result = []

        graph = self._succs

        def strongconnect(node):
            # set the depth index for this node to the smallest unused index
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)

            # Consider successors
            try:
                successors = graph[node]
            except Exception:
                successors = []
            for successor in successors:
                if successor not in lowlinks:
                    # Successor has not yet been visited
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif successor in stack:
                    # the successor is in the stack and hence in the current
                    # strongly connected component (SCC)
                    lowlinks[node] = min(lowlinks[node], index[successor])

            # If `node` is a root node, pop the stack and generate an SCC
            if lowlinks[node] == index[node]:
                connected_component = []

                while True:
                    successor = stack.pop()
                    connected_component.append(successor)
                    if successor == node:
                        break
                component = tuple(connected_component)
                # storing the result
                result.append(component)

        for node in graph:
            if node not in lowlinks:
                strongconnect(node)

        return result

    @property
    def dot(self):
        result = ['digraph G {']
        for succ in self._preds:
            preds = self._preds[succ]
            for pred in preds:
                result.append('  %s -> %s;' % (pred, succ))
        for node in self._nodes:
            result.append('  %s;' % node)
        result.append('}')
        return '\n'.join(result)


#
# Unarchiving functionality for zip, tar, tgz, tbz, whl
#

ARCHIVE_EXTENSIONS = ('.tar.gz', '.tar.bz2', '.tar', '.zip', '.tgz', '.tbz', '.whl')


def unarchive(archive_filename, dest_dir, format=None, check=True):

    def check_path(path):
        if not isinstance(path, text_type):
            path = path.decode('utf-8')
        p = os.path.abspath(os.path.join(dest_dir, path))
        if not p.startswith(dest_dir) or p[plen] != os.sep:
            raise ValueError('path outside destination: %r' % p)

    dest_dir = os.path.abspath(dest_dir)
    plen = len(dest_dir)
    archive = None
    if format is None:
        if archive_filename.endswith(('.zip', '.whl')):
            format = 'zip'
        elif archive_filename.endswith(('.tar.gz', '.tgz')):
            format = 'tgz'
            mode = 'r:gz'
        elif archive_filename.endswith(('.tar.bz2', '.tbz')):
            format = 'tbz'
            mode = 'r:bz2'
        elif archive_filename.endswith('.tar'):
            format = 'tar'
            mode = 'r'
        else:  # pragma: no cover
            raise ValueError('Unknown format for %r' % archive_filename)
    try:
        if format == 'zip':
            archive = ZipFile(archive_filename, 'r')
            if check:
                names = archive.namelist()
                for name in names:
                    check_path(name)
        else:
            archive = tarfile.open(archive_filename, mode)
            if check:
                names = archive.getnames()
                for name in names:
                    check_path(name)
        if format != 'zip' and sys.version_info[0] < 3:
            # See Python issue 17153. If the dest path contains Unicode,
            # tarfile extraction fails on Python 2.x if a member path name
            # contains non-ASCII characters - it leads to an implicit
            # bytes -> unicode conversion using ASCII to decode.
            for tarinfo in archive.getmembers():
                if not isinstance(tarinfo.name, text_type):
                    tarinfo.name = tarinfo.name.decode('utf-8')

        # Limit extraction of dangerous items, if this Python
        # allows it easily. If not, just trust the input.
        # See: https://docs.python.org/3/library/tarfile.html#extraction-filters
        def extraction_filter(member, path):
            """Run tarfile.tar_filter, but raise the expected ValueError"""
            # This is only called if the current Python has tarfile filters
            try:
                return tarfile.tar_filter(member, path)
            except tarfile.FilterError as exc:
                raise ValueError(str(exc))

        archive.extraction_filter = extraction_filter

        archive.extractall(dest_dir)

    finally:
        if archive:
            archive.close()


def zip_dir(directory):
    """zip a directory tree into a BytesIO object"""
    result = io.BytesIO()
    dlen = len(directory)
    with ZipFile(result, "w") as zf:
        for root, dirs, files in os.walk(directory):
            for name in files:
                full = os.path.join(root, name)
                rel = root[dlen:]
                dest = os.path.join(rel, name)
                zf.write(full, dest)
    return result


#
# Simple progress bar
#

UNITS = ('', 'K', 'M', 'G', 'T', 'P')


class Progress(object):
    unknown = 'UNKNOWN'

    def __init__(self, minval=0, maxval=100):
        assert maxval is None or maxval >= minval
        self.min = self.cur = minval
        self.max = maxval
        self.started = None
        self.elapsed = 0
        self.done = False

    def update(self, curval):
        assert self.min <= curval
        assert self.max is None or curval <= self.max
        self.cur = curval
        now = time.time()
        if self.started is None:
            self.started = now
        else:
            self.elapsed = now - self.started

    def increment(self, incr):
        assert incr >= 0
        self.update(self.cur + incr)

    def start(self):
        self.update(self.min)
        return self

    def stop(self):
        if self.max is not None:
            self.update(self.max)
        self.done = True

    @property
    def maximum(self):
        return self.unknown if self.max is None else self.max

    @property
    def percentage(self):
        if self.done:
            result = '100 %'
        elif self.max is None:
            result = ' ?? %'
        else:
            v = 100.0 * (self.cur - self.min) / (self.max - self.min)
            result = '%3d %%' % v
        return result

    def format_duration(self, duration):
        if (duration <= 0) and self.max is None or self.cur == self.min:
            result = '??:??:??'
        # elif duration < 1:
        #     result = '--:--:--'
        else:
            result = time.strftime('%H:%M:%S', time.gmtime(duration))
        return result

    @property
    def ETA(self):
        if self.done:
            prefix = 'Done'
            t = self.elapsed
            # import pdb; pdb.set_trace()
        else:
            prefix = 'ETA '
            if self.max is None:
                t = -1
            elif self.elapsed == 0 or (self.cur == self.min):
                t = 0
            else:
                # import pdb; pdb.set_trace()
                t = float(self.max - self.min)
                t /= self.cur - self.min
                t = (t - 1) * self.elapsed
        return '%s: %s' % (prefix, self.format_duration(t))

    @property
    def speed(self):
        if self.elapsed == 0:
            result = 0.0
        else:
            result = (self.cur - self.min) / self.elapsed
        for unit in UNITS:
            if result < 1000:
                break
            result /= 1000.0
        return '%d %sB/s' % (result, unit)


#
# Glob functionality
#

RICH_GLOB = re.compile(r'\{([^}]*)\}')
_CHECK_RECURSIVE_GLOB = re.compile(r'[^/\\,{]\*\*|\*\*[^/\\,}]')
_CHECK_MISMATCH_SET = re.compile(r'^[^{]*\}|\{[^}]*$')


def iglob(path_glob):
    """Extended globbing function that supports ** and {opt1,opt2,opt3}."""
    if _CHECK_RECURSIVE_GLOB.search(path_glob):
        msg = """invalid glob %r: recursive glob "**" must be used alone"""
        raise ValueError(msg % path_glob)
    if _CHECK_MISMATCH_SET.search(path_glob):
        msg = """invalid glob %r: mismatching set marker '{' or '}'"""
        raise ValueError(msg % path_glob)
    return _iglob(path_glob)


def _iglob(path_glob):
    rich_path_glob = RICH_GLOB.split(path_glob, 1)
    if len(rich_path_glob) > 1:
        assert len(rich_path_glob) == 3, rich_path_glob
        prefix, set, suffix = rich_path_glob
        for item in set.split(','):
            for path in _iglob(''.join((prefix, item, suffix))):
                yield path
    else:
        if '**' not in path_glob:
            for item in std_iglob(path_glob):
                yield item
        else:
            prefix, radical = path_glob.split('**', 1)
            if prefix == '':
                prefix = '.'
            if radical == '':
                radical = '*'
            else:
                # we support both
                radical = radical.lstrip('/')
                radical = radical.lstrip('\\')
            for path, dir, files in os.walk(prefix):
                path = os.path.normpath(path)
                for fn in _iglob(os.path.join(path, radical)):
                    yield fn


if ssl:
    from .compat import (HTTPSHandler as BaseHTTPSHandler, match_hostname, CertificateError)

    #
    # HTTPSConnection which verifies certificates/matches domains
    #

    class HTTPSConnection(httplib.HTTPSConnection):
        ca_certs = None  # set this to the path to the certs file (.pem)
        check_domain = True  # only used if ca_certs is not None

        # noinspection PyPropertyAccess
        def connect(self):
            sock = socket.create_connection((self.host, self.port), self.timeout)
            if getattr(self, '_tunnel_host', False):
                self.sock = sock
                self._tunnel()

            context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
            if hasattr(ssl, 'OP_NO_SSLv2'):
                context.options |= ssl.OP_NO_SSLv2
            if getattr(self, 'cert_file', None):
                context.load_cert_chain(self.cert_file, self.key_file)
            kwargs = {}
            if self.ca_certs:
                context.verify_mode = ssl.CERT_REQUIRED
                context.load_verify_locations(cafile=self.ca_certs)
                if getattr(ssl, 'HAS_SNI', False):
                    kwargs['server_hostname'] = self.host

            self.sock = context.wrap_socket(sock, **kwargs)
            if self.ca_certs and self.check_domain:
                try:
                    match_hostname(self.sock.getpeercert(), self.host)
                    logger.debug('Host verified: %s', self.host)
                except CertificateError:  # pragma: no cover
                    self.sock.shutdown(socket.SHUT_RDWR)
                    self.sock.close()
                    raise

    class HTTPSHandler(BaseHTTPSHandler):

        def __init__(self, ca_certs, check_domain=True):
            BaseHTTPSHandler.__init__(self)
            self.ca_certs = ca_certs
            self.check_domain = check_domain

        def _conn_maker(self, *args, **kwargs):
            """
            This is called to create a connection instance. Normally you'd
            pass a connection class to do_open, but it doesn't actually check for
            a class, and just expects a callable. As long as we behave just as a
            constructor would have, we should be OK. If it ever changes so that
            we *must* pass a class, we'll create an UnsafeHTTPSConnection class
            which just sets check_domain to False in the class definition, and
            choose which one to pass to do_open.
            """
            result = HTTPSConnection(*args, **kwargs)
            if self.ca_certs:
                result.ca_certs = self.ca_certs
                result.check_domain = self.check_domain
            return result

        def https_open(self, req):
            try:
                return self.do_open(self._conn_maker, req)
            except URLError as e:
                if 'certificate verify failed' in str(e.reason):
                    raise CertificateError('Unable to verify server certificate '
                                           'for %s' % req.host)
                else:
                    raise

    #
    # To prevent against mixing HTTP traffic with HTTPS (examples: A Man-In-The-
    # Middle proxy using HTTP listens on port 443, or an index mistakenly serves
    # HTML containing a http://xyz link when it should be https://xyz),
    # you can use the following handler class, which does not allow HTTP traffic.
    #
    # It works by inheriting from HTTPHandler - so build_opener won't add a
    # handler for HTTP itself.
    #
    class HTTPSOnlyHandler(HTTPSHandler, HTTPHandler):

        def http_open(self, req):
            raise URLError('Unexpected HTTP request on what should be a secure '
                           'connection: %s' % req)


#
# XML-RPC with timeouts
#
class Transport(xmlrpclib.Transport):

    def __init__(self, timeout, use_datetime=0):
        self.timeout = timeout
        xmlrpclib.Transport.__init__(self, use_datetime)

    def make_connection(self, host):
        h, eh, x509 = self.get_host_info(host)
        if not self._connection or host != self._connection[0]:
            self._extra_headers = eh
            self._connection = host, httplib.HTTPConnection(h)
        return self._connection[1]


if ssl:

    class SafeTransport(xmlrpclib.SafeTransport):

        def __init__(self, timeout, use_datetime=0):
            self.timeout = timeout
            xmlrpclib.SafeTransport.__init__(self, use_datetime)

        def make_connection(self, host):
            h, eh, kwargs = self.get_host_info(host)
            if not kwargs:
                kwargs = {}
            kwargs['timeout'] = self.timeout
            if not self._connection or host != self._connection[0]:
                self._extra_headers = eh
                self._connection = host, httplib.HTTPSConnection(h, None, **kwargs)
            return self._connection[1]


class ServerProxy(xmlrpclib.ServerProxy):

    def __init__(self, uri, **kwargs):
        self.timeout = timeout = kwargs.pop('timeout', None)
        # The above classes only come into play if a timeout
        # is specified
        if timeout is not None:
            # scheme = splittype(uri)  # deprecated as of Python 3.8
            scheme = urlparse(uri)[0]
            use_datetime = kwargs.get('use_datetime', 0)
            if scheme == 'https':
                tcls = SafeTransport
            else:
                tcls = Transport
            kwargs['transport'] = t = tcls(timeout, use_datetime=use_datetime)
            self.transport = t
        xmlrpclib.ServerProxy.__init__(self, uri, **kwargs)


#
# CSV functionality. This is provided because on 2.x, the csv module can't
# handle Unicode. However, we need to deal with Unicode in e.g. RECORD files.
#


def _csv_open(fn, mode, **kwargs):
    if sys.version_info[0] < 3:
        mode += 'b'
    else:
        kwargs['newline'] = ''
        # Python 3 determines encoding from locale. Force 'utf-8'
        # file encoding to match other forced utf-8 encoding
        kwargs['encoding'] = 'utf-8'
    return open(fn, mode, **kwargs)


class CSVBase(object):
    defaults = {
        'delimiter': str(','),  # The strs are used because we need native
        'quotechar': str('"'),  # str in the csv API (2.x won't take
        'lineterminator': str('\n')  # Unicode)
    }

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.stream.close()


class CSVReader(CSVBase):

    def __init__(self, **kwargs):
        if 'stream' in kwargs:
            stream = kwargs['stream']
            if sys.version_info[0] >= 3:
                # needs to be a text stream
                stream = codecs.getreader('utf-8')(stream)
            self.stream = stream
        else:
            self.stream = _csv_open(kwargs['path'], 'r')
        self.reader = csv.reader(self.stream, **self.defaults)

    def __iter__(self):
        return self

    def next(self):
        result = next(self.reader)
        if sys.version_info[0] < 3:
            for i, item in enumerate(result):
                if not isinstance(item, text_type):
                    result[i] = item.decode('utf-8')
        return result

    __next__ = next


class CSVWriter(CSVBase):

    def __init__(self, fn, **kwargs):
        self.stream = _csv_open(fn, 'w')
        self.writer = csv.writer(self.stream, **self.defaults)

    def writerow(self, row):
        if sys.version_info[0] < 3:
            r = []
            for item in row:
                if isinstance(item, text_type):
                    item = item.encode('utf-8')
                r.append(item)
            row = r
        self.writer.writerow(row)


#
#   Configurator functionality
#


class Configurator(BaseConfigurator):

    value_converters = dict(BaseConfigurator.value_converters)
    value_converters['inc'] = 'inc_convert'

    def __init__(self, config, base=None):
        super(Configurator, self).__init__(config)
        self.base = base or os.getcwd()

    def configure_custom(self, config):

        def convert(o):
            if isinstance(o, (list, tuple)):
                result = type(o)([convert(i) for i in o])
            elif isinstance(o, dict):
                if '()' in o:
                    result = self.configure_custom(o)
                else:
                    result = {}
                    for k in o:
                        result[k] = convert(o[k])
            else:
                result = self.convert(o)
            return result

        c = config.pop('()')
        if not callable(c):
            c = self.resolve(c)
        props = config.pop('.', None)
        # Check for valid identifiers
        args = config.pop('[]', ())
        if args:
            args = tuple([convert(o) for o in args])
        items = [(k, convert(config[k])) for k in config if valid_ident(k)]
        kwargs = dict(items)
        result = c(*args, **kwargs)
        if props:
            for n, v in props.items():
                setattr(result, n, convert(v))
        return result

    def __getitem__(self, key):
        result = self.config[key]
        if isinstance(result, dict) and '()' in result:
            self.config[key] = result = self.configure_custom(result)
        return result

    def inc_convert(self, value):
        """Default converter for the inc:// protocol."""
        if not os.path.isabs(value):
            value = os.path.join(self.base, value)
        with codecs.open(value, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return result


class SubprocessMixin(object):
    """
    Mixin for running subprocesses and capturing their output
    """

    def __init__(self, verbose=False, progress=None):
        self.verbose = verbose
        self.progress = progress

    def reader(self, stream, context):
        """
        Read lines from a subprocess' output stream and either pass to a progress
        callable (if specified) or write progress information to sys.stderr.
        """
        progress = self.progress
        verbose = self.verbose
        while True:
            s = stream.readline()
            if not s:
                break
            if progress is not None:
                progress(s, context)
            else:
                if not verbose:
                    sys.stderr.write('.')
                else:
                    sys.stderr.write(s.decode('utf-8'))
                sys.stderr.flush()
        stream.close()

    def run_command(self, cmd, **kwargs):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        t1 = threading.Thread(target=self.reader, args=(p.stdout, 'stdout'))
        t1.start()
        t2 = threading.Thread(target=self.reader, args=(p.stderr, 'stderr'))
        t2.start()
        p.wait()
        t1.join()
        t2.join()
        if self.progress is not None:
            self.progress('done.', 'main')
        elif self.verbose:
            sys.stderr.write('done.\n')
        return p


def normalize_name(name):
    """Normalize a python package name a la PEP 503"""
    # https://www.python.org/dev/peps/pep-0503/#normalized-names
    return re.sub('[-_.]+', '-', name).lower()


# def _get_pypirc_command():
# """
# Get the distutils command for interacting with PyPI configurations.
# :return: the command.
# """
# from distutils.core import Distribution
# from distutils.config import PyPIRCCommand
# d = Distribution()
# return PyPIRCCommand(d)


class PyPIRCFile(object):

    DEFAULT_REPOSITORY = 'https://upload.pypi.org/legacy/'
    DEFAULT_REALM = 'pypi'

    def __init__(self, fn=None, url=None):
        if fn is None:
            fn = os.path.join(os.path.expanduser('~'), '.pypirc')
        self.filename = fn
        self.url = url

    def read(self):
        result = {}

        if os.path.exists(self.filename):
            repository = self.url or self.DEFAULT_REPOSITORY

            config = configparser.RawConfigParser()
            config.read(self.filename)
            sections = config.sections()
            if 'distutils' in sections:
                # let's get the list of servers
                index_servers = config.get('distutils', 'index-servers')
                _servers = [server.strip() for server in index_servers.split('\n') if server.strip() != '']
                if _servers == []:
                    # nothing set, let's try to get the default pypi
                    if 'pypi' in sections:
                        _servers = ['pypi']
                else:
                    for server in _servers:
                        result = {'server': server}
                        result['username'] = config.get(server, 'username')

                        # optional params
                        for key, default in (('repository', self.DEFAULT_REPOSITORY), ('realm', self.DEFAULT_REALM),
                                             ('password', None)):
                            if config.has_option(server, key):
                                result[key] = config.get(server, key)
                            else:
                                result[key] = default

                        # work around people having "repository" for the "pypi"
                        # section of their config set to the HTTP (rather than
                        # HTTPS) URL
                        if (server == 'pypi' and repository in (self.DEFAULT_REPOSITORY, 'pypi')):
                            result['repository'] = self.DEFAULT_REPOSITORY
                        elif (result['server'] != repository and result['repository'] != repository):
                            result = {}
            elif 'server-login' in sections:
                # old format
                server = 'server-login'
                if config.has_option(server, 'repository'):
                    repository = config.get(server, 'repository')
                else:
                    repository = self.DEFAULT_REPOSITORY
                result = {
                    'username': config.get(server, 'username'),
                    'password': config.get(server, 'password'),
                    'repository': repository,
                    'server': server,
                    'realm': self.DEFAULT_REALM
                }
        return result

    def update(self, username, password):
        # import pdb; pdb.set_trace()
        config = configparser.RawConfigParser()
        fn = self.filename
        config.read(fn)
        if not config.has_section('pypi'):
            config.add_section('pypi')
        config.set('pypi', 'username', username)
        config.set('pypi', 'password', password)
        with open(fn, 'w') as f:
            config.write(f)


def _load_pypirc(index):
    """
    Read the PyPI access configuration as supported by distutils.
    """
    return PyPIRCFile(url=index.url).read()


def _store_pypirc(index):
    PyPIRCFile().update(index.username, index.password)


#
# get_platform()/get_host_platform() copied from Python 3.10.a0 source, with some minor
# tweaks
#


def get_host_platform():
    """Return a string that identifies the current platform.  This is used mainly to
    distinguish platform-specific build directories and platform-specific built
    distributions.  Typically includes the OS name and version and the
    architecture (as supplied by 'os.uname()'), although the exact information
    included depends on the OS; eg. on Linux, the kernel version isn't
    particularly important.

    Examples of returned values:
       linux-i586
       linux-alpha (?)
       solaris-2.6-sun4u

    Windows will return one of:
       win-amd64 (64bit Windows on AMD64 (aka x86_64, Intel64, EM64T, etc)
       win32 (all others - specifically, sys.platform is returned)

    For other non-POSIX platforms, currently just returns 'sys.platform'.

    """
    if os.name == 'nt':
        if 'amd64' in sys.version.lower():
            return 'win-amd64'
        if '(arm)' in sys.version.lower():
            return 'win-arm32'
        if '(arm64)' in sys.version.lower():
            return 'win-arm64'
        return sys.platform

    # Set for cross builds explicitly
    if "_PYTHON_HOST_PLATFORM" in os.environ:
        return os.environ["_PYTHON_HOST_PLATFORM"]

    if os.name != 'posix' or not hasattr(os, 'uname'):
        # XXX what about the architecture? NT is Intel or Alpha,
        # Mac OS is M68k or PPC, etc.
        return sys.platform

    # Try to distinguish various flavours of Unix

    (osname, host, release, version, machine) = os.uname()

    # Convert the OS name to lowercase, remove '/' characters, and translate
    # spaces (for "Power Macintosh")
    osname = osname.lower().replace('/', '')
    machine = machine.replace(' ', '_').replace('/', '-')

    if osname[:5] == 'linux':
        # At least on Linux/Intel, 'machine' is the processor --
        # i386, etc.
        # XXX what about Alpha, SPARC, etc?
        return "%s-%s" % (osname, machine)

    elif osname[:5] == 'sunos':
        if release[0] >= '5':  # SunOS 5 == Solaris 2
            osname = 'solaris'
            release = '%d.%s' % (int(release[0]) - 3, release[2:])
            # We can't use 'platform.architecture()[0]' because a
            # bootstrap problem. We use a dict to get an error
            # if some suspicious happens.
            bitness = {2147483647: '32bit', 9223372036854775807: '64bit'}
            machine += '.%s' % bitness[sys.maxsize]
        # fall through to standard osname-release-machine representation
    elif osname[:3] == 'aix':
        from _aix_support import aix_platform
        return aix_platform()
    elif osname[:6] == 'cygwin':
        osname = 'cygwin'
        rel_re = re.compile(r'[\d.]+', re.ASCII)
        m = rel_re.match(release)
        if m:
            release = m.group()
    elif osname[:6] == 'darwin':
        import _osx_support
        try:
            from distutils import sysconfig
        except ImportError:
            import sysconfig
        osname, release, machine = _osx_support.get_platform_osx(sysconfig.get_config_vars(), osname, release, machine)

    return '%s-%s-%s' % (osname, release, machine)


_TARGET_TO_PLAT = {
    'x86': 'win32',
    'x64': 'win-amd64',
    'arm': 'win-arm32',
}


def get_platform():
    if os.name != 'nt':
        return get_host_platform()
    cross_compilation_target = os.environ.get('VSCMD_ARG_TGT_ARCH')
    if cross_compilation_target not in _TARGET_TO_PLAT:
        return get_host_platform()
    return _TARGET_TO_PLAT[cross_compilation_target]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\array_expressions.py ===
from __future__ import annotations
import collections.abc
import operator
from collections import defaultdict, Counter
from functools import reduce
import itertools
from itertools import accumulate

import typing

from sympy.core.numbers import Integer
from sympy.core.relational import Equality
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import (Dummy, Symbol)
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.expressions.diagonal import diagonalize_vector
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensordiagonal, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.utils import _apply_recursively_over_nested_lists, _sort_contraction_indices, \
    _get_mapping_from_subranks, _build_push_indices_up_func_transformation, _get_contraction_links, \
    _build_push_indices_down_func_transformation
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.core.sympify import _sympify


class _ArrayExpr(Expr):
    shape: tuple[Expr, ...]

    def __getitem__(self, item):
        if not isinstance(item, collections.abc.Iterable):
            item = (item,)
        ArrayElement._check_shape(self, item)
        return self._get(item)

    def _get(self, item):
        return _get_array_element_or_slice(self, item)


class ArraySymbol(_ArrayExpr):
    """
    Symbol representing an array expression
    """

    _iterable = False

    def __new__(cls, symbol, shape: typing.Iterable) -> "ArraySymbol":
        if isinstance(symbol, str):
            symbol = Symbol(symbol)
        # symbol = _sympify(symbol)
        shape = Tuple(*map(_sympify, shape))
        obj = Expr.__new__(cls, symbol, shape)
        return obj

    @property
    def name(self):
        return self._args[0]

    @property
    def shape(self):
        return self._args[1]

    def as_explicit(self):
        if not all(i.is_Integer for i in self.shape):
            raise ValueError("cannot express explicit array with symbolic shape")
        data = [self[i] for i in itertools.product(*[range(j) for j in self.shape])]
        return ImmutableDenseNDimArray(data).reshape(*self.shape)


class ArrayElement(Expr):
    """
    An element of an array.
    """

    _diff_wrt = True
    is_symbol = True
    is_commutative = True

    def __new__(cls, name, indices):
        if isinstance(name, str):
            name = Symbol(name)
        name = _sympify(name)
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)
        indices = _sympify(tuple(indices))
        cls._check_shape(name, indices)
        obj = Expr.__new__(cls, name, indices)
        return obj

    @classmethod
    def _check_shape(cls, name, indices):
        indices = tuple(indices)
        if hasattr(name, "shape"):
            index_error = IndexError("number of indices does not match shape of the array")
            if len(indices) != len(name.shape):
                raise index_error
            if any((i >= s) == True for i, s in zip(indices, name.shape)):
                raise ValueError("shape is out of bounds")
        if any((i < 0) == True for i in indices):
            raise ValueError("shape contains negative values")

    @property
    def name(self):
        return self._args[0]

    @property
    def indices(self):
        return self._args[1]

    def _eval_derivative(self, s):
        if not isinstance(s, ArrayElement):
            return S.Zero

        if s == self:
            return S.One

        if s.name != self.name:
            return S.Zero

        return Mul.fromiter(KroneckerDelta(i, j) for i, j in zip(self.indices, s.indices))


class ZeroArray(_ArrayExpr):
    """
    Symbolic array of zeros. Equivalent to ``ZeroMatrix`` for matrices.
    """

    def __new__(cls, *shape):
        if len(shape) == 0:
            return S.Zero
        shape = map(_sympify, shape)
        obj = Expr.__new__(cls, *shape)
        return obj

    @property
    def shape(self):
        return self._args

    def as_explicit(self):
        if not all(i.is_Integer for i in self.shape):
            raise ValueError("Cannot return explicit form for symbolic shape.")
        return ImmutableDenseNDimArray.zeros(*self.shape)

    def _get(self, item):
        return S.Zero


class OneArray(_ArrayExpr):
    """
    Symbolic array of ones.
    """

    def __new__(cls, *shape):
        if len(shape) == 0:
            return S.One
        shape = map(_sympify, shape)
        obj = Expr.__new__(cls, *shape)
        return obj

    @property
    def shape(self):
        return self._args

    def as_explicit(self):
        if not all(i.is_Integer for i in self.shape):
            raise ValueError("Cannot return explicit form for symbolic shape.")
        return ImmutableDenseNDimArray([S.One for i in range(reduce(operator.mul, self.shape))]).reshape(*self.shape)

    def _get(self, item):
        return S.One


class _CodegenArrayAbstract(Basic):

    @property
    def subranks(self):
        """
        Returns the ranks of the objects in the uppermost tensor product inside
        the current object.  In case no tensor products are contained, return
        the atomic ranks.

        Examples
        ========

        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> from sympy import MatrixSymbol
        >>> M = MatrixSymbol("M", 3, 3)
        >>> N = MatrixSymbol("N", 3, 3)
        >>> P = MatrixSymbol("P", 3, 3)

        Important: do not confuse the rank of the matrix with the rank of an array.

        >>> tp = tensorproduct(M, N, P)
        >>> tp.subranks
        [2, 2, 2]

        >>> co = tensorcontraction(tp, (1, 2), (3, 4))
        >>> co.subranks
        [2, 2, 2]
        """
        return self._subranks[:]

    def subrank(self):
        """
        The sum of ``subranks``.
        """
        return sum(self.subranks)

    @property
    def shape(self):
        return self._shape

    def doit(self, **hints):
        deep = hints.get("deep", True)
        if deep:
            return self.func(*[arg.doit(**hints) for arg in self.args])._canonicalize()
        else:
            return self._canonicalize()

class ArrayTensorProduct(_CodegenArrayAbstract):
    r"""
    Class to represent the tensor product of array-like objects.
    """

    def __new__(cls, *args, **kwargs):
        args = [_sympify(arg) for arg in args]

        canonicalize = kwargs.pop("canonicalize", False)

        ranks = [get_rank(arg) for arg in args]

        obj = Basic.__new__(cls, *args)
        obj._subranks = ranks
        shapes = [get_shape(i) for i in args]

        if any(i is None for i in shapes):
            obj._shape = None
        else:
            obj._shape = tuple(j for i in shapes for j in i)
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        args = self.args
        args = self._flatten(args)

        ranks = [get_rank(arg) for arg in args]

        # Check if there are nested permutation and lift them up:
        permutation_cycles = []
        for i, arg in enumerate(args):
            if not isinstance(arg, PermuteDims):
                continue
            permutation_cycles.extend([[k + sum(ranks[:i]) for k in j] for j in arg.permutation.cyclic_form])
            args[i] = arg.expr
        if permutation_cycles:
            return _permute_dims(_array_tensor_product(*args), Permutation(sum(ranks)-1)*Permutation(permutation_cycles))

        if len(args) == 1:
            return args[0]

        # If any object is a ZeroArray, return a ZeroArray:
        if any(isinstance(arg, (ZeroArray, ZeroMatrix)) for arg in args):
            shapes = reduce(operator.add, [get_shape(i) for i in args], ())
            return ZeroArray(*shapes)

        # If there are contraction objects inside, transform the whole
        # expression into `ArrayContraction`:
        contractions = {i: arg for i, arg in enumerate(args) if isinstance(arg, ArrayContraction)}
        if contractions:
            ranks = [_get_subrank(arg) if isinstance(arg, ArrayContraction) else get_rank(arg) for arg in args]
            cumulative_ranks = list(accumulate([0] + ranks))[:-1]
            tp = _array_tensor_product(*[arg.expr if isinstance(arg, ArrayContraction) else arg for arg in args])
            contraction_indices = [tuple(cumulative_ranks[i] + k for k in j) for i, arg in contractions.items() for j in arg.contraction_indices]
            return _array_contraction(tp, *contraction_indices)

        diagonals = {i: arg for i, arg in enumerate(args) if isinstance(arg, ArrayDiagonal)}
        if diagonals:
            inverse_permutation = []
            last_perm = []
            ranks = [get_rank(arg) for arg in args]
            cumulative_ranks = list(accumulate([0] + ranks))[:-1]
            for i, arg in enumerate(args):
                if isinstance(arg, ArrayDiagonal):
                    i1 = get_rank(arg) - len(arg.diagonal_indices)
                    i2 = len(arg.diagonal_indices)
                    inverse_permutation.extend([cumulative_ranks[i] + j for j in range(i1)])
                    last_perm.extend([cumulative_ranks[i] + j for j in range(i1, i1 + i2)])
                else:
                    inverse_permutation.extend([cumulative_ranks[i] + j for j in range(get_rank(arg))])
            inverse_permutation.extend(last_perm)
            tp = _array_tensor_product(*[arg.expr if isinstance(arg, ArrayDiagonal) else arg for arg in args])
            ranks2 = [_get_subrank(arg) if isinstance(arg, ArrayDiagonal) else get_rank(arg) for arg in args]
            cumulative_ranks2 = list(accumulate([0] + ranks2))[:-1]
            diagonal_indices = [tuple(cumulative_ranks2[i] + k for k in j) for i, arg in diagonals.items() for j in arg.diagonal_indices]
            return _permute_dims(_array_diagonal(tp, *diagonal_indices), _af_invert(inverse_permutation))

        return self.func(*args, canonicalize=False)

    @classmethod
    def _flatten(cls, args):
        args = [i for arg in args for i in (arg.args if isinstance(arg, cls) else [arg])]
        return args

    def as_explicit(self):
        return tensorproduct(*[arg.as_explicit() if hasattr(arg, "as_explicit") else arg for arg in self.args])


class ArrayAdd(_CodegenArrayAbstract):
    r"""
    Class for elementwise array additions.
    """

    def __new__(cls, *args, **kwargs):
        args = [_sympify(arg) for arg in args]
        ranks = [get_rank(arg) for arg in args]
        ranks = list(set(ranks))
        if len(ranks) != 1:
            raise ValueError("summing arrays of different ranks")
        shapes = [arg.shape for arg in args]
        if len({i for i in shapes if i is not None}) > 1:
            raise ValueError("mismatching shapes in addition")

        canonicalize = kwargs.pop("canonicalize", False)

        obj = Basic.__new__(cls, *args)
        obj._subranks = ranks
        if any(i is None for i in shapes):
            obj._shape = None
        else:
            obj._shape = shapes[0]
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        args = self.args

        # Flatten:
        args = self._flatten_args(args)

        shapes = [get_shape(arg) for arg in args]
        args = [arg for arg in args if not isinstance(arg, (ZeroArray, ZeroMatrix))]
        if len(args) == 0:
            if any(i for i in shapes if i is None):
                raise NotImplementedError("cannot handle addition of ZeroMatrix/ZeroArray and undefined shape object")
            return ZeroArray(*shapes[0])
        elif len(args) == 1:
            return args[0]
        return self.func(*args, canonicalize=False)

    @classmethod
    def _flatten_args(cls, args):
        new_args = []
        for arg in args:
            if isinstance(arg, ArrayAdd):
                new_args.extend(arg.args)
            else:
                new_args.append(arg)
        return new_args

    def as_explicit(self):
        return reduce(
            operator.add,
            [arg.as_explicit() if hasattr(arg, "as_explicit") else arg for arg in self.args])


class PermuteDims(_CodegenArrayAbstract):
    r"""
    Class to represent permutation of axes of arrays.

    Examples
    ========

    >>> from sympy.tensor.array import permutedims
    >>> from sympy import MatrixSymbol
    >>> M = MatrixSymbol("M", 3, 3)
    >>> cg = permutedims(M, [1, 0])

    The object ``cg`` represents the transposition of ``M``, as the permutation
    ``[1, 0]`` will act on its indices by switching them:

    `M_{ij} \Rightarrow M_{ji}`

    This is evident when transforming back to matrix form:

    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> convert_array_to_matrix(cg)
    M.T

    >>> N = MatrixSymbol("N", 3, 2)
    >>> cg = permutedims(N, [1, 0])
    >>> cg.shape
    (2, 3)

    There are optional parameters that can be used as alternative to the permutation:

    >>> from sympy.tensor.array.expressions import ArraySymbol, PermuteDims
    >>> M = ArraySymbol("M", (1, 2, 3, 4, 5))
    >>> expr = PermuteDims(M, index_order_old="ijklm", index_order_new="kijml")
    >>> expr
    PermuteDims(M, (0 2 1)(3 4))
    >>> expr.shape
    (3, 1, 2, 5, 4)

    Permutations of tensor products are simplified in order to achieve a
    standard form:

    >>> from sympy.tensor.array import tensorproduct
    >>> M = MatrixSymbol("M", 4, 5)
    >>> tp = tensorproduct(M, N)
    >>> tp.shape
    (4, 5, 3, 2)
    >>> perm1 = permutedims(tp, [2, 3, 1, 0])

    The args ``(M, N)`` have been sorted and the permutation has been
    simplified, the expression is equivalent:

    >>> perm1.expr.args
    (N, M)
    >>> perm1.shape
    (3, 2, 5, 4)
    >>> perm1.permutation
    (2 3)

    The permutation in its array form has been simplified from
    ``[2, 3, 1, 0]`` to ``[0, 1, 3, 2]``, as the arguments of the tensor
    product `M` and `N` have been switched:

    >>> perm1.permutation.array_form
    [0, 1, 3, 2]

    We can nest a second permutation:

    >>> perm2 = permutedims(perm1, [1, 0, 2, 3])
    >>> perm2.shape
    (2, 3, 5, 4)
    >>> perm2.permutation.array_form
    [1, 0, 3, 2]
    """

    def __new__(cls, expr, permutation=None, index_order_old=None, index_order_new=None, **kwargs):
        from sympy.combinatorics import Permutation
        expr = _sympify(expr)
        expr_rank = get_rank(expr)
        permutation = cls._get_permutation_from_arguments(permutation, index_order_old, index_order_new, expr_rank)
        permutation = Permutation(permutation)
        permutation_size = permutation.size
        if permutation_size != expr_rank:
            raise ValueError("Permutation size must be the length of the shape of expr")

        canonicalize = kwargs.pop("canonicalize", False)

        obj = Basic.__new__(cls, expr, permutation)
        obj._subranks = [get_rank(expr)]
        shape = get_shape(expr)
        if shape is None:
            obj._shape = None
        else:
            obj._shape = tuple(shape[permutation(i)] for i in range(len(shape)))
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        expr = self.expr
        permutation = self.permutation
        if isinstance(expr, PermuteDims):
            subexpr = expr.expr
            subperm = expr.permutation
            permutation = permutation * subperm
            expr = subexpr
        if isinstance(expr, ArrayContraction):
            expr, permutation = self._PermuteDims_denestarg_ArrayContraction(expr, permutation)
        if isinstance(expr, ArrayTensorProduct):
            expr, permutation = self._PermuteDims_denestarg_ArrayTensorProduct(expr, permutation)
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            return ZeroArray(*[expr.shape[i] for i in permutation.array_form])
        plist = permutation.array_form
        if plist == sorted(plist):
            return expr
        return self.func(expr, permutation, canonicalize=False)

    @property
    def expr(self):
        return self.args[0]

    @property
    def permutation(self):
        return self.args[1]

    @classmethod
    def _PermuteDims_denestarg_ArrayTensorProduct(cls, expr, permutation):
        # Get the permutation in its image-form:
        perm_image_form = _af_invert(permutation.array_form)
        args = list(expr.args)
        # Starting index global position for every arg:
        cumul = list(accumulate([0] + expr.subranks))
        # Split `perm_image_form` into a list of list corresponding to the indices
        # of every argument:
        perm_image_form_in_components = [perm_image_form[cumul[i]:cumul[i+1]] for i in range(len(args))]
        # Create an index, target-position-key array:
        ps = [(i, sorted(comp)) for i, comp in enumerate(perm_image_form_in_components)]
        # Sort the array according to the target-position-key:
        # In this way, we define a canonical way to sort the arguments according
        # to the permutation.
        ps.sort(key=lambda x: x[1])
        # Read the inverse-permutation (i.e. image-form) of the args:
        perm_args_image_form = [i[0] for i in ps]
        # Apply the args-permutation to the `args`:
        args_sorted = [args[i] for i in perm_args_image_form]
        # Apply the args-permutation to the array-form of the permutation of the axes (of `expr`):
        perm_image_form_sorted_args = [perm_image_form_in_components[i] for i in perm_args_image_form]
        new_permutation = Permutation(_af_invert([j for i in perm_image_form_sorted_args for j in i]))
        return _array_tensor_product(*args_sorted), new_permutation

    @classmethod
    def _PermuteDims_denestarg_ArrayContraction(cls, expr, permutation):
        if not isinstance(expr, ArrayContraction):
            return expr, permutation
        if not isinstance(expr.expr, ArrayTensorProduct):
            return expr, permutation
        args = expr.expr.args
        subranks = [get_rank(arg) for arg in expr.expr.args]

        contraction_indices = expr.contraction_indices
        contraction_indices_flat = [j for i in contraction_indices for j in i]
        cumul = list(accumulate([0] + subranks))

        # Spread the permutation in its array form across the args in the corresponding
        # tensor-product arguments with free indices:
        permutation_array_blocks_up = []
        image_form = _af_invert(permutation.array_form)
        counter = 0
        for i in range(len(subranks)):
            current = []
            for j in range(cumul[i], cumul[i+1]):
                if j in contraction_indices_flat:
                    continue
                current.append(image_form[counter])
                counter += 1
            permutation_array_blocks_up.append(current)

        # Get the map of axis repositioning for every argument of tensor-product:
        index_blocks = [list(range(cumul[i], cumul[i+1])) for i, e in enumerate(expr.subranks)]
        index_blocks_up = expr._push_indices_up(expr.contraction_indices, index_blocks)
        inverse_permutation = permutation**(-1)
        index_blocks_up_permuted = [[inverse_permutation(j) for j in i if j is not None] for i in index_blocks_up]

        # Sorting key is a list of tuple, first element is the index of `args`, second element of
        # the tuple is the sorting key to sort `args` of the tensor product:
        sorting_keys = list(enumerate(index_blocks_up_permuted))
        sorting_keys.sort(key=lambda x: x[1])

        # Now we can get the permutation acting on the args in its image-form:
        new_perm_image_form = [i[0] for i in sorting_keys]
        # Apply the args-level permutation to various elements:
        new_index_blocks = [index_blocks[i] for i in new_perm_image_form]
        new_index_perm_array_form = _af_invert([j for i in new_index_blocks for j in i])
        new_args = [args[i] for i in new_perm_image_form]
        new_contraction_indices = [tuple(new_index_perm_array_form[j] for j in i) for i in contraction_indices]
        new_expr = _array_contraction(_array_tensor_product(*new_args), *new_contraction_indices)
        new_permutation = Permutation(_af_invert([j for i in [permutation_array_blocks_up[k] for k in new_perm_image_form] for j in i]))
        return new_expr, new_permutation

    @classmethod
    def _check_permutation_mapping(cls, expr, permutation):
        subranks = expr.subranks
        index2arg = [i for i, arg in enumerate(expr.args) for j in range(expr.subranks[i])]
        permuted_indices = [permutation(i) for i in range(expr.subrank())]
        new_args = list(expr.args)
        arg_candidate_index = index2arg[permuted_indices[0]]
        current_indices = []
        new_permutation = []
        inserted_arg_cand_indices = set()
        for i, idx in enumerate(permuted_indices):
            if index2arg[idx] != arg_candidate_index:
                new_permutation.extend(current_indices)
                current_indices = []
                arg_candidate_index = index2arg[idx]
            current_indices.append(idx)
            arg_candidate_rank = subranks[arg_candidate_index]
            if len(current_indices) == arg_candidate_rank:
                new_permutation.extend(sorted(current_indices))
                local_current_indices = [j - min(current_indices) for j in current_indices]
                i1 = index2arg[i]
                new_args[i1] = _permute_dims(new_args[i1], Permutation(local_current_indices))
                inserted_arg_cand_indices.add(arg_candidate_index)
                current_indices = []
        new_permutation.extend(current_indices)

        # TODO: swap args positions in order to simplify the expression:
        # TODO: this should be in a function
        args_positions = list(range(len(new_args)))
        # Get possible shifts:
        maps = {}
        cumulative_subranks = [0] + list(accumulate(subranks))
        for i in range(len(subranks)):
            s = {index2arg[new_permutation[j]] for j in range(cumulative_subranks[i], cumulative_subranks[i+1])}
            if len(s) != 1:
                continue
            elem = next(iter(s))
            if i != elem:
                maps[i] = elem

        # Find cycles in the map:
        lines = []
        current_line = []
        while maps:
            if len(current_line) == 0:
                k, v = maps.popitem()
                current_line.append(k)
            else:
                k = current_line[-1]
                if k not in maps:
                    current_line = []
                    continue
                v = maps.pop(k)
            if v in current_line:
                lines.append(current_line)
                current_line = []
                continue
            current_line.append(v)
        for line in lines:
            for i, e in enumerate(line):
                args_positions[line[(i + 1) % len(line)]] = e

        # TODO: function in order to permute the args:
        permutation_blocks = [[new_permutation[cumulative_subranks[i] + j] for j in range(e)] for i, e in enumerate(subranks)]
        new_args = [new_args[i] for i in args_positions]
        new_permutation_blocks = [permutation_blocks[i] for i in args_positions]
        new_permutation2 = [j for i in new_permutation_blocks for j in i]
        return _array_tensor_product(*new_args), Permutation(new_permutation2)  # **(-1)

    @classmethod
    def _check_if_there_are_closed_cycles(cls, expr, permutation):
        args = list(expr.args)
        subranks = expr.subranks
        cyclic_form = permutation.cyclic_form
        cumulative_subranks = [0] + list(accumulate(subranks))
        cyclic_min = [min(i) for i in cyclic_form]
        cyclic_max = [max(i) for i in cyclic_form]
        cyclic_keep = []
        for i, cycle in enumerate(cyclic_form):
            flag = True
            for j in range(len(cumulative_subranks) - 1):
                if cyclic_min[i] >= cumulative_subranks[j] and cyclic_max[i] < cumulative_subranks[j+1]:
                    # Found a sinkable cycle.
                    args[j] = _permute_dims(args[j], Permutation([[k - cumulative_subranks[j] for k in cycle]]))
                    flag = False
                    break
            if flag:
                cyclic_keep.append(cycle)
        return _array_tensor_product(*args), Permutation(cyclic_keep, size=permutation.size)

    def nest_permutation(self):
        r"""
        DEPRECATED.
        """
        ret = self._nest_permutation(self.expr, self.permutation)
        if ret is None:
            return self
        return ret

    @classmethod
    def _nest_permutation(cls, expr, permutation):
        if isinstance(expr, ArrayTensorProduct):
            return _permute_dims(*cls._check_if_there_are_closed_cycles(expr, permutation))
        elif isinstance(expr, ArrayContraction):
            # Invert tree hierarchy: put the contraction above.
            cycles = permutation.cyclic_form
            newcycles = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *cycles)
            newpermutation = Permutation(newcycles)
            new_contr_indices = [tuple(newpermutation(j) for j in i) for i in expr.contraction_indices]
            return _array_contraction(PermuteDims(expr.expr, newpermutation), *new_contr_indices)
        elif isinstance(expr, ArrayAdd):
            return _array_add(*[PermuteDims(arg, permutation) for arg in expr.args])
        return None

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        return permutedims(expr, self.permutation)

    @classmethod
    def _get_permutation_from_arguments(cls, permutation, index_order_old, index_order_new, dim):
        if permutation is None:
            if index_order_new is None or index_order_old is None:
                raise ValueError("Permutation not defined")
            return PermuteDims._get_permutation_from_index_orders(index_order_old, index_order_new, dim)
        else:
            if index_order_new is not None:
                raise ValueError("index_order_new cannot be defined with permutation")
            if index_order_old is not None:
                raise ValueError("index_order_old cannot be defined with permutation")
            return permutation

    @classmethod
    def _get_permutation_from_index_orders(cls, index_order_old, index_order_new, dim):
        if len(set(index_order_new)) != dim:
            raise ValueError("wrong number of indices in index_order_new")
        if len(set(index_order_old)) != dim:
            raise ValueError("wrong number of indices in index_order_old")
        if len(set.symmetric_difference(set(index_order_new), set(index_order_old))) > 0:
            raise ValueError("index_order_new and index_order_old must have the same indices")
        permutation = [index_order_old.index(i) for i in index_order_new]
        return permutation


class ArrayDiagonal(_CodegenArrayAbstract):
    r"""
    Class to represent the diagonal operator.

    Explanation
    ===========

    In a 2-dimensional array it returns the diagonal, this looks like the
    operation:

    `A_{ij} \rightarrow A_{ii}`

    The diagonal over axes 1 and 2 (the second and third) of the tensor product
    of two 2-dimensional arrays `A \otimes B` is

    `\Big[ A_{ab} B_{cd} \Big]_{abcd} \rightarrow \Big[ A_{ai} B_{id} \Big]_{adi}`

    In this last example the array expression has been reduced from
    4-dimensional to 3-dimensional. Notice that no contraction has occurred,
    rather there is a new index `i` for the diagonal, contraction would have
    reduced the array to 2 dimensions.

    Notice that the diagonalized out dimensions are added as new dimensions at
    the end of the indices.
    """

    def __new__(cls, expr, *diagonal_indices, **kwargs):
        expr = _sympify(expr)
        diagonal_indices = [Tuple(*sorted(i)) for i in diagonal_indices]
        canonicalize = kwargs.get("canonicalize", False)

        shape = get_shape(expr)
        if shape is not None:
            cls._validate(expr, *diagonal_indices, **kwargs)
            # Get new shape:
            positions, shape = cls._get_positions_shape(shape, diagonal_indices)
        else:
            positions = None
        if len(diagonal_indices) == 0:
            return expr
        obj = Basic.__new__(cls, expr, *diagonal_indices)
        obj._positions = positions
        obj._subranks = _get_subranks(expr)
        obj._shape = shape
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        expr = self.expr
        diagonal_indices = self.diagonal_indices
        trivial_diags = [i for i in diagonal_indices if len(i) == 1]
        if len(trivial_diags) > 0:
            trivial_pos = {e[0]: i for i, e in enumerate(diagonal_indices) if len(e) == 1}
            diag_pos = {e: i for i, e in enumerate(diagonal_indices) if len(e) > 1}
            diagonal_indices_short = [i for i in diagonal_indices if len(i) > 1]
            rank1 = get_rank(self)
            rank2 = len(diagonal_indices)
            rank3 = rank1 - rank2
            inv_permutation = []
            counter1 = 0
            indices_down = ArrayDiagonal._push_indices_down(diagonal_indices_short, list(range(rank1)), get_rank(expr))
            for i in indices_down:
                if i in trivial_pos:
                    inv_permutation.append(rank3 + trivial_pos[i])
                elif isinstance(i, (Integer, int)):
                    inv_permutation.append(counter1)
                    counter1 += 1
                else:
                    inv_permutation.append(rank3 + diag_pos[i])
            permutation = _af_invert(inv_permutation)
            if len(diagonal_indices_short) > 0:
                return _permute_dims(_array_diagonal(expr, *diagonal_indices_short), permutation)
            else:
                return _permute_dims(expr, permutation)
        if isinstance(expr, ArrayAdd):
            return self._ArrayDiagonal_denest_ArrayAdd(expr, *diagonal_indices)
        if isinstance(expr, ArrayDiagonal):
            return self._ArrayDiagonal_denest_ArrayDiagonal(expr, *diagonal_indices)
        if isinstance(expr, PermuteDims):
            return self._ArrayDiagonal_denest_PermuteDims(expr, *diagonal_indices)
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            positions, shape = self._get_positions_shape(expr.shape, diagonal_indices)
            return ZeroArray(*shape)
        return self.func(expr, *diagonal_indices, canonicalize=False)

    @staticmethod
    def _validate(expr, *diagonal_indices, **kwargs):
        # Check that no diagonalization happens on indices with mismatched
        # dimensions:
        shape = get_shape(expr)
        for i in diagonal_indices:
            if any(j >= len(shape) for j in i):
                raise ValueError("index is larger than expression shape")
            if len({shape[j] for j in i}) != 1:
                raise ValueError("diagonalizing indices of different dimensions")
            if not kwargs.get("allow_trivial_diags", False) and len(i) <= 1:
                raise ValueError("need at least two axes to diagonalize")
            if len(set(i)) != len(i):
                raise ValueError("axis index cannot be repeated")

    @staticmethod
    def _remove_trivial_dimensions(shape, *diagonal_indices):
        return [tuple(j for j in i) for i in diagonal_indices if shape[i[0]] != 1]

    @property
    def expr(self):
        return self.args[0]

    @property
    def diagonal_indices(self):
        return self.args[1:]

    @staticmethod
    def _flatten(expr, *outer_diagonal_indices):
        inner_diagonal_indices = expr.diagonal_indices
        all_inner = [j for i in inner_diagonal_indices for j in i]
        all_inner.sort()
        # TODO: add API for total rank and cumulative rank:
        total_rank = _get_subrank(expr)
        inner_rank = len(all_inner)
        outer_rank = total_rank - inner_rank
        shifts = [0 for i in range(outer_rank)]
        counter = 0
        pointer = 0
        for i in range(outer_rank):
            while pointer < inner_rank and counter >= all_inner[pointer]:
                counter += 1
                pointer += 1
            shifts[i] += pointer
            counter += 1
        outer_diagonal_indices = tuple(tuple(shifts[j] + j for j in i) for i in outer_diagonal_indices)
        diagonal_indices = inner_diagonal_indices + outer_diagonal_indices
        return _array_diagonal(expr.expr, *diagonal_indices)

    @classmethod
    def _ArrayDiagonal_denest_ArrayAdd(cls, expr, *diagonal_indices):
        return _array_add(*[_array_diagonal(arg, *diagonal_indices) for arg in expr.args])

    @classmethod
    def _ArrayDiagonal_denest_ArrayDiagonal(cls, expr, *diagonal_indices):
        return cls._flatten(expr, *diagonal_indices)

    @classmethod
    def _ArrayDiagonal_denest_PermuteDims(cls, expr: PermuteDims, *diagonal_indices):
        back_diagonal_indices = [[expr.permutation(j) for j in i] for i in diagonal_indices]
        nondiag = [i for i in range(get_rank(expr)) if not any(i in j for j in diagonal_indices)]
        back_nondiag = [expr.permutation(i) for i in nondiag]
        remap = {e: i for i, e in enumerate(sorted(back_nondiag))}
        new_permutation1 = [remap[i] for i in back_nondiag]
        shift = len(new_permutation1)
        diag_block_perm = [i + shift for i in range(len(back_diagonal_indices))]
        new_permutation = new_permutation1 + diag_block_perm
        return _permute_dims(
            _array_diagonal(
                expr.expr,
                *back_diagonal_indices
            ),
            new_permutation
        )

    def _push_indices_down_nonstatic(self, indices):
        transform = lambda x: self._positions[x] if x < len(self._positions) else None
        return _apply_recursively_over_nested_lists(transform, indices)

    def _push_indices_up_nonstatic(self, indices):

        def transform(x):
            for i, e in enumerate(self._positions):
                if (isinstance(e, int) and x == e) or (isinstance(e, tuple) and x in e):
                    return i

        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_down(cls, diagonal_indices, indices, rank):
        positions, shape = cls._get_positions_shape(range(rank), diagonal_indices)
        transform = lambda x: positions[x] if x < len(positions) else None
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_up(cls, diagonal_indices, indices, rank):
        positions, shape = cls._get_positions_shape(range(rank), diagonal_indices)

        def transform(x):
            for i, e in enumerate(positions):
                if (isinstance(e, int) and x == e) or (isinstance(e, (tuple, Tuple)) and (x in e)):
                    return i

        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _get_positions_shape(cls, shape, diagonal_indices):
        data1 = tuple((i, shp) for i, shp in enumerate(shape) if not any(i in j for j in diagonal_indices))
        pos1, shp1 = zip(*data1) if data1 else ((), ())
        data2 = tuple((i, shape[i[0]]) for i in diagonal_indices)
        pos2, shp2 = zip(*data2) if data2 else ((), ())
        positions = pos1 + pos2
        shape = shp1 + shp2
        return positions, shape

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        return tensordiagonal(expr, *self.diagonal_indices)


class ArrayElementwiseApplyFunc(_CodegenArrayAbstract):

    def __new__(cls, function, element):

        if not isinstance(function, Lambda):
            d = Dummy('d')
            function = Lambda(d, function(d))

        obj = _CodegenArrayAbstract.__new__(cls, function, element)
        obj._subranks = _get_subranks(element)
        return obj

    @property
    def function(self):
        return self.args[0]

    @property
    def expr(self):
        return self.args[1]

    @property
    def shape(self):
        return self.expr.shape

    def _get_function_fdiff(self):
        d = Dummy("d")
        function = self.function(d)
        fdiff = function.diff(d)
        if isinstance(fdiff, Function):
            fdiff = type(fdiff)
        else:
            fdiff = Lambda(d, fdiff)
        return fdiff

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        return expr.applyfunc(self.function)


class ArrayContraction(_CodegenArrayAbstract):
    r"""
    This class is meant to represent contractions of arrays in a form easily
    processable by the code printers.
    """

    def __new__(cls, expr, *contraction_indices, **kwargs):
        contraction_indices = _sort_contraction_indices(contraction_indices)
        expr = _sympify(expr)

        canonicalize = kwargs.get("canonicalize", False)

        obj = Basic.__new__(cls, expr, *contraction_indices)
        obj._subranks = _get_subranks(expr)
        obj._mapping = _get_mapping_from_subranks(obj._subranks)

        free_indices_to_position = {i: i for i in range(sum(obj._subranks)) if all(i not in cind for cind in contraction_indices)}
        obj._free_indices_to_position = free_indices_to_position

        shape = get_shape(expr)
        cls._validate(expr, *contraction_indices)
        if shape:
            shape = tuple(shp for i, shp in enumerate(shape) if not any(i in j for j in contraction_indices))
        obj._shape = shape
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        expr = self.expr
        contraction_indices = self.contraction_indices

        if len(contraction_indices) == 0:
            return expr

        if isinstance(expr, ArrayContraction):
            return self._ArrayContraction_denest_ArrayContraction(expr, *contraction_indices)

        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            return self._ArrayContraction_denest_ZeroArray(expr, *contraction_indices)

        if isinstance(expr, PermuteDims):
            return self._ArrayContraction_denest_PermuteDims(expr, *contraction_indices)

        if isinstance(expr, ArrayTensorProduct):
            expr, contraction_indices = self._sort_fully_contracted_args(expr, contraction_indices)
            expr, contraction_indices = self._lower_contraction_to_addends(expr, contraction_indices)
            if len(contraction_indices) == 0:
                return expr

        if isinstance(expr, ArrayDiagonal):
            return self._ArrayContraction_denest_ArrayDiagonal(expr, *contraction_indices)

        if isinstance(expr, ArrayAdd):
            return self._ArrayContraction_denest_ArrayAdd(expr, *contraction_indices)

        # Check single index contractions on 1-dimensional axes:
        contraction_indices = [i for i in contraction_indices if len(i) > 1 or get_shape(expr)[i[0]] != 1]
        if len(contraction_indices) == 0:
            return expr

        return self.func(expr, *contraction_indices, canonicalize=False)

    def __mul__(self, other):
        if other == 1:
            return self
        else:
            raise NotImplementedError("Product of N-dim arrays is not uniquely defined. Use another method.")

    def __rmul__(self, other):
        if other == 1:
            return self
        else:
            raise NotImplementedError("Product of N-dim arrays is not uniquely defined. Use another method.")

    @staticmethod
    def _validate(expr, *contraction_indices):
        shape = get_shape(expr)
        if shape is None:
            return

        # Check that no contraction happens when the shape is mismatched:
        for i in contraction_indices:
            if len({shape[j] for j in i if shape[j] != -1}) != 1:
                raise ValueError("contracting indices of different dimensions")

    @classmethod
    def _push_indices_down(cls, contraction_indices, indices):
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        flattened_contraction_indices.sort()
        transform = _build_push_indices_down_func_transformation(flattened_contraction_indices)
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_up(cls, contraction_indices, indices):
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        flattened_contraction_indices.sort()
        transform = _build_push_indices_up_func_transformation(flattened_contraction_indices)
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _lower_contraction_to_addends(cls, expr, contraction_indices):
        if isinstance(expr, ArrayAdd):
            raise NotImplementedError()
        if not isinstance(expr, ArrayTensorProduct):
            return expr, contraction_indices
        subranks = expr.subranks
        cumranks = list(accumulate([0] + subranks))
        contraction_indices_remaining = []
        contraction_indices_args = [[] for i in expr.args]
        backshift = set()
        for contraction_group in contraction_indices:
            for j in range(len(expr.args)):
                if not isinstance(expr.args[j], ArrayAdd):
                    continue
                if all(cumranks[j] <= k < cumranks[j+1] for k in contraction_group):
                    contraction_indices_args[j].append([k - cumranks[j] for k in contraction_group])
                    backshift.update(contraction_group)
                    break
            else:
                contraction_indices_remaining.append(contraction_group)
        if len(contraction_indices_remaining) == len(contraction_indices):
            return expr, contraction_indices
        total_rank = get_rank(expr)
        shifts = list(accumulate([1 if i in backshift else 0 for i in range(total_rank)]))
        contraction_indices_remaining = [Tuple.fromiter(j - shifts[j] for j in i) for i in contraction_indices_remaining]
        ret = _array_tensor_product(*[
            _array_contraction(arg, *contr) for arg, contr in zip(expr.args, contraction_indices_args)
        ])
        return ret, contraction_indices_remaining

    def split_multiple_contractions(self):
        """
        Recognize multiple contractions and attempt at rewriting them as paired-contractions.

        This allows some contractions involving more than two indices to be
        rewritten as multiple contractions involving two indices, thus allowing
        the expression to be rewritten as a matrix multiplication line.

        Examples:

        * `A_ij b_j0 C_jk` ===> `A*DiagMatrix(b)*C`

        Care for:
        - matrix being diagonalized (i.e. `A_ii`)
        - vectors being diagonalized (i.e. `a_i0`)

        Multiple contractions can be split into matrix multiplications if
        not more than two arguments are non-diagonals or non-vectors.
        Vectors get diagonalized while diagonal matrices remain diagonal.
        The non-diagonal matrices can be at the beginning or at the end
        of the final matrix multiplication line.
        """

        editor = _EditArrayContraction(self)

        contraction_indices = self.contraction_indices

        onearray_insert = []

        for indl, links in enumerate(contraction_indices):
            if len(links) <= 2:
                continue

            # Check multiple contractions:
            #
            # Examples:
            #
            # * `A_ij b_j0 C_jk` ===> `A*DiagMatrix(b)*C \otimes OneArray(1)` with permutation (1 2)
            #
            # Care for:
            # - matrix being diagonalized (i.e. `A_ii`)
            # - vectors being diagonalized (i.e. `a_i0`)

            # Multiple contractions can be split into matrix multiplications if
            # not more than three arguments are non-diagonals or non-vectors.
            #
            # Vectors get diagonalized while diagonal matrices remain diagonal.
            # The non-diagonal matrices can be at the beginning or at the end
            # of the final matrix multiplication line.

            positions = editor.get_mapping_for_index(indl)

            # Also consider the case of diagonal matrices being contracted:
            current_dimension = self.expr.shape[links[0]]

            not_vectors = []
            vectors = []
            for arg_ind, rel_ind in positions:
                arg = editor.args_with_ind[arg_ind]
                mat = arg.element
                abs_arg_start, abs_arg_end = editor.get_absolute_range(arg)
                other_arg_pos = 1-rel_ind
                other_arg_abs = abs_arg_start + other_arg_pos
                if ((1 not in mat.shape) or
                    ((current_dimension == 1) is True and mat.shape != (1, 1)) or
                    any(other_arg_abs in l for li, l in enumerate(contraction_indices) if li != indl)
                ):
                    not_vectors.append((arg, rel_ind))
                else:
                    vectors.append((arg, rel_ind))
            if len(not_vectors) > 2:
                # If more than two arguments in the multiple contraction are
                # non-vectors and non-diagonal matrices, we cannot find a way
                # to split this contraction into a matrix multiplication line:
                continue
            # Three cases to handle:
            # - zero non-vectors
            # - one non-vector
            # - two non-vectors
            for v, rel_ind in vectors:
                v.element = diagonalize_vector(v.element)
            vectors_to_loop = not_vectors[:1] + vectors + not_vectors[1:]
            first_not_vector, rel_ind = vectors_to_loop[0]
            new_index = first_not_vector.indices[rel_ind]

            for v, rel_ind in vectors_to_loop[1:-1]:
                v.indices[rel_ind] = new_index
                new_index = editor.get_new_contraction_index()
                assert v.indices.index(None) == 1 - rel_ind
                v.indices[v.indices.index(None)] = new_index
                onearray_insert.append(v)

            last_vec, rel_ind = vectors_to_loop[-1]
            last_vec.indices[rel_ind] = new_index

        for v in onearray_insert:
            editor.insert_after(v, _ArgE(OneArray(1), [None]))

        return editor.to_array_contraction()

    def flatten_contraction_of_diagonal(self):
        if not isinstance(self.expr, ArrayDiagonal):
            return self
        contraction_down = self.expr._push_indices_down(self.expr.diagonal_indices, self.contraction_indices)
        new_contraction_indices = []
        diagonal_indices = self.expr.diagonal_indices[:]
        for i in contraction_down:
            contraction_group = list(i)
            for j in i:
                diagonal_with = [k for k in diagonal_indices if j in k]
                contraction_group.extend([l for k in diagonal_with for l in k])
                diagonal_indices = [k for k in diagonal_indices if k not in diagonal_with]
            new_contraction_indices.append(sorted(set(contraction_group)))

        new_contraction_indices = ArrayDiagonal._push_indices_up(diagonal_indices, new_contraction_indices)
        return _array_contraction(
            _array_diagonal(
                self.expr.expr,
                *diagonal_indices
            ),
            *new_contraction_indices
        )

    @staticmethod
    def _get_free_indices_to_position_map(free_indices, contraction_indices):
        free_indices_to_position = {}
        flattened_contraction_indices = [j for i in contraction_indices for j in i]
        counter = 0
        for ind in free_indices:
            while counter in flattened_contraction_indices:
                counter += 1
            free_indices_to_position[ind] = counter
            counter += 1
        return free_indices_to_position

    @staticmethod
    def _get_index_shifts(expr):
        """
        Get the mapping of indices at the positions before the contraction
        occurs.

        Examples
        ========

        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> from sympy import MatrixSymbol
        >>> M = MatrixSymbol("M", 3, 3)
        >>> N = MatrixSymbol("N", 3, 3)
        >>> cg = tensorcontraction(tensorproduct(M, N), [1, 2])
        >>> cg._get_index_shifts(cg)
        [0, 2]

        Indeed, ``cg`` after the contraction has two dimensions, 0 and 1. They
        need to be shifted by 0 and 2 to get the corresponding positions before
        the contraction (that is, 0 and 3).
        """
        inner_contraction_indices = expr.contraction_indices
        all_inner = [j for i in inner_contraction_indices for j in i]
        all_inner.sort()
        # TODO: add API for total rank and cumulative rank:
        total_rank = _get_subrank(expr)
        inner_rank = len(all_inner)
        outer_rank = total_rank - inner_rank
        shifts = [0 for i in range(outer_rank)]
        counter = 0
        pointer = 0
        for i in range(outer_rank):
            while pointer < inner_rank and counter >= all_inner[pointer]:
                counter += 1
                pointer += 1
            shifts[i] += pointer
            counter += 1
        return shifts

    @staticmethod
    def _convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices):
        shifts = ArrayContraction._get_index_shifts(expr)
        outer_contraction_indices = tuple(tuple(shifts[j] + j for j in i) for i in outer_contraction_indices)
        return outer_contraction_indices

    @staticmethod
    def _flatten(expr, *outer_contraction_indices):
        inner_contraction_indices = expr.contraction_indices
        outer_contraction_indices = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices)
        contraction_indices = inner_contraction_indices + outer_contraction_indices
        return _array_contraction(expr.expr, *contraction_indices)

    @classmethod
    def _ArrayContraction_denest_ArrayContraction(cls, expr, *contraction_indices):
        return cls._flatten(expr, *contraction_indices)

    @classmethod
    def _ArrayContraction_denest_ZeroArray(cls, expr, *contraction_indices):
        contraction_indices_flat = [j for i in contraction_indices for j in i]
        shape = [e for i, e in enumerate(expr.shape) if i not in contraction_indices_flat]
        return ZeroArray(*shape)

    @classmethod
    def _ArrayContraction_denest_ArrayAdd(cls, expr, *contraction_indices):
        return _array_add(*[_array_contraction(i, *contraction_indices) for i in expr.args])

    @classmethod
    def _ArrayContraction_denest_PermuteDims(cls, expr, *contraction_indices):
        permutation = expr.permutation
        plist = permutation.array_form
        new_contraction_indices = [tuple(permutation(j) for j in i) for i in contraction_indices]
        new_plist = [i for i in plist if not any(i in j for j in new_contraction_indices)]
        new_plist = cls._push_indices_up(new_contraction_indices, new_plist)
        return _permute_dims(
            _array_contraction(expr.expr, *new_contraction_indices),
            Permutation(new_plist)
        )

    @classmethod
    def _ArrayContraction_denest_ArrayDiagonal(cls, expr: 'ArrayDiagonal', *contraction_indices):
        diagonal_indices = list(expr.diagonal_indices)
        down_contraction_indices = expr._push_indices_down(expr.diagonal_indices, contraction_indices, get_rank(expr.expr))
        # Flatten diagonally contracted indices:
        down_contraction_indices = [[k for j in i for k in (j if isinstance(j, (tuple, Tuple)) else [j])] for i in down_contraction_indices]
        new_contraction_indices = []
        for contr_indgrp in down_contraction_indices:
            ind = contr_indgrp[:]
            for j, diag_indgrp in enumerate(diagonal_indices):
                if diag_indgrp is None:
                    continue
                if any(i in diag_indgrp for i in contr_indgrp):
                    ind.extend(diag_indgrp)
                    diagonal_indices[j] = None
            new_contraction_indices.append(sorted(set(ind)))

        new_diagonal_indices_down = [i for i in diagonal_indices if i is not None]
        new_diagonal_indices = ArrayContraction._push_indices_up(new_contraction_indices, new_diagonal_indices_down)
        return _array_diagonal(
            _array_contraction(expr.expr, *new_contraction_indices),
            *new_diagonal_indices
        )

    @classmethod
    def _sort_fully_contracted_args(cls, expr, contraction_indices):
        if expr.shape is None:
            return expr, contraction_indices
        cumul = list(accumulate([0] + expr.subranks))
        index_blocks = [list(range(cumul[i], cumul[i+1])) for i in range(len(expr.args))]
        contraction_indices_flat = {j for i in contraction_indices for j in i}
        fully_contracted = [all(j in contraction_indices_flat for j in range(cumul[i], cumul[i+1])) for i, arg in enumerate(expr.args)]
        new_pos = sorted(range(len(expr.args)), key=lambda x: (0, default_sort_key(expr.args[x])) if fully_contracted[x] else (1,))
        new_args = [expr.args[i] for i in new_pos]
        new_index_blocks_flat = [j for i in new_pos for j in index_blocks[i]]
        index_permutation_array_form = _af_invert(new_index_blocks_flat)
        new_contraction_indices = [tuple(index_permutation_array_form[j] for j in i) for i in contraction_indices]
        new_contraction_indices = _sort_contraction_indices(new_contraction_indices)
        return _array_tensor_product(*new_args), new_contraction_indices

    def _get_contraction_tuples(self):
        r"""
        Return tuples containing the argument index and position within the
        argument of the index position.

        Examples
        ========

        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)

        >>> cg = tensorcontraction(tensorproduct(A, B), (1, 2))
        >>> cg._get_contraction_tuples()
        [[(0, 1), (1, 0)]]

        Notes
        =====

        Here the contraction pair `(1, 2)` meaning that the 2nd and 3rd indices
        of the tensor product `A\otimes B` are contracted, has been transformed
        into `(0, 1)` and `(1, 0)`, identifying the same indices in a different
        notation. `(0, 1)` is the second index (1) of the first argument (i.e.
                0 or `A`). `(1, 0)` is the first index (i.e. 0) of the second
        argument (i.e. 1 or `B`).
        """
        mapping = self._mapping
        return [[mapping[j] for j in i] for i in self.contraction_indices]

    @staticmethod
    def _contraction_tuples_to_contraction_indices(expr, contraction_tuples):
        # TODO: check that `expr` has `.subranks`:
        ranks = expr.subranks
        cumulative_ranks = [0] + list(accumulate(ranks))
        return [tuple(cumulative_ranks[j]+k for j, k in i) for i in contraction_tuples]

    @property
    def free_indices(self):
        return self._free_indices[:]

    @property
    def free_indices_to_position(self):
        return dict(self._free_indices_to_position)

    @property
    def expr(self):
        return self.args[0]

    @property
    def contraction_indices(self):
        return self.args[1:]

    def _contraction_indices_to_components(self):
        expr = self.expr
        if not isinstance(expr, ArrayTensorProduct):
            raise NotImplementedError("only for contractions of tensor products")
        ranks = expr.subranks
        mapping = {}
        counter = 0
        for i, rank in enumerate(ranks):
            for j in range(rank):
                mapping[counter] = (i, j)
                counter += 1
        return mapping

    def sort_args_by_name(self):
        """
        Sort arguments in the tensor product so that their order is lexicographical.

        Examples
        ========

        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> C = MatrixSymbol("C", N, N)
        >>> D = MatrixSymbol("D", N, N)

        >>> cg = convert_matrix_to_array(C*D*A*B)
        >>> cg
        ArrayContraction(ArrayTensorProduct(A, D, C, B), (0, 3), (1, 6), (2, 5))
        >>> cg.sort_args_by_name()
        ArrayContraction(ArrayTensorProduct(A, D, B, C), (0, 3), (1, 4), (2, 7))
        """
        expr = self.expr
        if not isinstance(expr, ArrayTensorProduct):
            return self
        args = expr.args
        sorted_data = sorted(enumerate(args), key=lambda x: default_sort_key(x[1]))
        pos_sorted, args_sorted = zip(*sorted_data)
        reordering_map = {i: pos_sorted.index(i) for i, arg in enumerate(args)}
        contraction_tuples = self._get_contraction_tuples()
        contraction_tuples = [[(reordering_map[j], k) for j, k in i] for i in contraction_tuples]
        c_tp = _array_tensor_product(*args_sorted)
        new_contr_indices = self._contraction_tuples_to_contraction_indices(
                c_tp,
                contraction_tuples
        )
        return _array_contraction(c_tp, *new_contr_indices)

    def _get_contraction_links(self):
        r"""
        Returns a dictionary of links between arguments in the tensor product
        being contracted.

        See the example for an explanation of the values.

        Examples
        ========

        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> C = MatrixSymbol("C", N, N)
        >>> D = MatrixSymbol("D", N, N)

        Matrix multiplications are pairwise contractions between neighboring
        matrices:

        `A_{ij} B_{jk} C_{kl} D_{lm}`

        >>> cg = convert_matrix_to_array(A*B*C*D)
        >>> cg
        ArrayContraction(ArrayTensorProduct(B, C, A, D), (0, 5), (1, 2), (3, 6))

        >>> cg._get_contraction_links()
        {0: {0: (2, 1), 1: (1, 0)}, 1: {0: (0, 1), 1: (3, 0)}, 2: {1: (0, 0)}, 3: {0: (1, 1)}}

        This dictionary is interpreted as follows: argument in position 0 (i.e.
        matrix `A`) has its second index (i.e. 1) contracted to `(1, 0)`, that
        is argument in position 1 (matrix `B`) on the first index slot of `B`,
        this is the contraction provided by the index `j` from `A`.

        The argument in position 1 (that is, matrix `B`) has two contractions,
        the ones provided by the indices `j` and `k`, respectively the first
        and second indices (0 and 1 in the sub-dict).  The link `(0, 1)` and
        `(2, 0)` respectively. `(0, 1)` is the index slot 1 (the 2nd) of
        argument in position 0 (that is, `A_{\ldot j}`), and so on.
        """
        args, dlinks = _get_contraction_links([self], self.subranks, *self.contraction_indices)
        return dlinks

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, "as_explicit"):
            expr = expr.as_explicit()
        return tensorcontraction(expr, *self.contraction_indices)


class Reshape(_CodegenArrayAbstract):
    """
    Reshape the dimensions of an array expression.

    Examples
    ========

    >>> from sympy.tensor.array.expressions import ArraySymbol, Reshape
    >>> A = ArraySymbol("A", (6,))
    >>> A.shape
    (6,)
    >>> Reshape(A, (3, 2)).shape
    (3, 2)

    Check the component-explicit forms:

    >>> A.as_explicit()
    [A[0], A[1], A[2], A[3], A[4], A[5]]
    >>> Reshape(A, (3, 2)).as_explicit()
    [[A[0], A[1]], [A[2], A[3]], [A[4], A[5]]]

    """

    def __new__(cls, expr, shape):
        expr = _sympify(expr)
        if not isinstance(shape, Tuple):
            shape = Tuple(*shape)
        if Equality(Mul.fromiter(expr.shape), Mul.fromiter(shape)) == False:
            raise ValueError("shape mismatch")
        obj = Expr.__new__(cls, expr, shape)
        obj._shape = tuple(shape)
        obj._expr = expr
        return obj

    @property
    def shape(self):
        return self._shape

    @property
    def expr(self):
        return self._expr

    def doit(self, *args, **kwargs):
        if kwargs.get("deep", True):
            expr = self.expr.doit(*args, **kwargs)
        else:
            expr = self.expr
        if isinstance(expr, (MatrixBase, NDimArray)):
            return expr.reshape(*self.shape)
        return Reshape(expr, self.shape)

    def as_explicit(self):
        ee = self.expr
        if hasattr(ee, "as_explicit"):
            ee = ee.as_explicit()
        if isinstance(ee, MatrixBase):
            from sympy import Array
            ee = Array(ee)
        elif isinstance(ee, MatrixExpr):
            return self
        return ee.reshape(*self.shape)


class _ArgE:
    """
    The ``_ArgE`` object contains references to the array expression
    (``.element``) and a list containing the information about index
    contractions (``.indices``).

    Index contractions are numbered and contracted indices show the number of
    the contraction. Uncontracted indices have ``None`` value.

    For example:
    ``_ArgE(M, [None, 3])``
    This object means that expression ``M`` is part of an array contraction
    and has two indices, the first is not contracted (value ``None``),
    the second index is contracted to the 4th (i.e. number ``3``) group of the
    array contraction object.
    """
    indices: list[int | None]

    def __init__(self, element, indices: list[int | None] | None = None):
        self.element = element
        if indices is None:
            self.indices = [None for i in range(get_rank(element))]
        else:
            self.indices = indices

    def __str__(self):
        return "_ArgE(%s, %s)" % (self.element, self.indices)

    __repr__ = __str__


class _IndPos:
    """
    Index position, requiring two integers in the constructor:

    - arg: the position of the argument in the tensor product,
    - rel: the relative position of the index inside the argument.
    """
    def __init__(self, arg: int, rel: int):
        self.arg = arg
        self.rel = rel

    def __str__(self):
        return "_IndPos(%i, %i)" % (self.arg, self.rel)

    __repr__ = __str__

    def __iter__(self):
        yield from [self.arg, self.rel]


class _EditArrayContraction:
    """
    Utility class to help manipulate array contraction objects.

    This class takes as input an ``ArrayContraction`` object and turns it into
    an editable object.

    The field ``args_with_ind`` of this class is a list of ``_ArgE`` objects
    which can be used to easily edit the contraction structure of the
    expression.

    Once editing is finished, the ``ArrayContraction`` object may be recreated
    by calling the ``.to_array_contraction()`` method.
    """

    def __init__(self, base_array: typing.Union[ArrayContraction, ArrayDiagonal, ArrayTensorProduct]):

        expr: Basic
        diagonalized: tuple[tuple[int, ...], ...]
        contraction_indices: list[tuple[int]]
        if isinstance(base_array, ArrayContraction):
            mapping = _get_mapping_from_subranks(base_array.subranks)
            expr = base_array.expr
            contraction_indices = base_array.contraction_indices
            diagonalized = ()
        elif isinstance(base_array, ArrayDiagonal):

            if isinstance(base_array.expr, ArrayContraction):
                mapping = _get_mapping_from_subranks(base_array.expr.subranks)
                expr = base_array.expr.expr
                diagonalized = ArrayContraction._push_indices_down(base_array.expr.contraction_indices, base_array.diagonal_indices)
                contraction_indices = base_array.expr.contraction_indices
            elif isinstance(base_array.expr, ArrayTensorProduct):
                mapping = {}
                expr = base_array.expr
                diagonalized = base_array.diagonal_indices
                contraction_indices = []
            else:
                mapping = {}
                expr = base_array.expr
                diagonalized = base_array.diagonal_indices
                contraction_indices = []

        elif isinstance(base_array, ArrayTensorProduct):
            expr = base_array
            contraction_indices = []
            diagonalized = ()
        else:
            raise NotImplementedError()

        if isinstance(expr, ArrayTensorProduct):
            args = list(expr.args)
        else:
            args = [expr]

        args_with_ind: list[_ArgE] = [_ArgE(arg) for arg in args]
        for i, contraction_tuple in enumerate(contraction_indices):
            for j in contraction_tuple:
                arg_pos, rel_pos = mapping[j]
                args_with_ind[arg_pos].indices[rel_pos] = i
        self.args_with_ind: list[_ArgE] = args_with_ind
        self.number_of_contraction_indices: int = len(contraction_indices)
        self._track_permutation: list[list[int]] | None = None

        mapping = _get_mapping_from_subranks(base_array.subranks)

        # Trick: add diagonalized indices as negative indices into the editor object:
        for i, e in enumerate(diagonalized):
            for j in e:
                arg_pos, rel_pos = mapping[j]
                self.args_with_ind[arg_pos].indices[rel_pos] = -1 - i

    def insert_after(self, arg: _ArgE, new_arg: _ArgE):
        pos = self.args_with_ind.index(arg)
        self.args_with_ind.insert(pos + 1, new_arg)

    def get_new_contraction_index(self):
        self.number_of_contraction_indices += 1
        return self.number_of_contraction_indices - 1

    def refresh_indices(self):
        updates = {}
        for arg_with_ind in self.args_with_ind:
            updates.update({i: -1 for i in arg_with_ind.indices if i is not None})
        for i, e in enumerate(sorted(updates)):
            updates[e] = i
        self.number_of_contraction_indices = len(updates)
        for arg_with_ind in self.args_with_ind:
            arg_with_ind.indices = [updates.get(i, None) for i in arg_with_ind.indices]

    def merge_scalars(self):
        scalars = []
        for arg_with_ind in self.args_with_ind:
            if len(arg_with_ind.indices) == 0:
                scalars.append(arg_with_ind)
        for i in scalars:
            self.args_with_ind.remove(i)
        scalar = Mul.fromiter([i.element for i in scalars])
        if len(self.args_with_ind) == 0:
            self.args_with_ind.append(_ArgE(scalar))
        else:
            from sympy.tensor.array.expressions.from_array_to_matrix import _a2m_tensor_product
            self.args_with_ind[0].element = _a2m_tensor_product(scalar, self.args_with_ind[0].element)

    def to_array_contraction(self):

        # Count the ranks of the arguments:
        counter = 0
        # Create a collector for the new diagonal indices:
        diag_indices = defaultdict(list)

        count_index_freq = Counter()
        for arg_with_ind in self.args_with_ind:
            count_index_freq.update(Counter(arg_with_ind.indices))

        free_index_count = count_index_freq[None]

        # Construct the inverse permutation:
        inv_perm1 = []
        inv_perm2 = []
        # Keep track of which diagonal indices have already been processed:
        done = set()

        # Counter for the diagonal indices:
        counter4 = 0

        for arg_with_ind in self.args_with_ind:
            # If some diagonalization axes have been removed, they should be
            # permuted in order to keep the permutation.
            # Add permutation here
            counter2 = 0  # counter for the indices
            for i in arg_with_ind.indices:
                if i is None:
                    inv_perm1.append(counter4)
                    counter2 += 1
                    counter4 += 1
                    continue
                if i >= 0:
                    continue
                # Reconstruct the diagonal indices:
                diag_indices[-1 - i].append(counter + counter2)
                if count_index_freq[i] == 1 and i not in done:
                    inv_perm1.append(free_index_count - 1 - i)
                    done.add(i)
                elif i not in done:
                    inv_perm2.append(free_index_count - 1 - i)
                    done.add(i)
                counter2 += 1
            # Remove negative indices to restore a proper editor object:
            arg_with_ind.indices = [i if i is not None and i >= 0 else None for i in arg_with_ind.indices]
            counter += len([i for i in arg_with_ind.indices if i is None or i < 0])

        inverse_permutation = inv_perm1 + inv_perm2
        permutation = _af_invert(inverse_permutation)

        # Get the diagonal indices after the detection of HadamardProduct in the expression:
        diag_indices_filtered = [tuple(v) for v in diag_indices.values() if len(v) > 1]

        self.merge_scalars()
        self.refresh_indices()
        args = [arg.element for arg in self.args_with_ind]
        contraction_indices = self.get_contraction_indices()
        expr = _array_contraction(_array_tensor_product(*args), *contraction_indices)
        expr2 = _array_diagonal(expr, *diag_indices_filtered)
        if self._track_permutation is not None:
            permutation2 = _af_invert([j for i in self._track_permutation for j in i])
            expr2 = _permute_dims(expr2, permutation2)

        expr3 = _permute_dims(expr2, permutation)
        return expr3

    def get_contraction_indices(self) -> list[list[int]]:
        contraction_indices: list[list[int]] = [[] for i in range(self.number_of_contraction_indices)]
        current_position: int = 0
        for arg_with_ind in self.args_with_ind:
            for j in arg_with_ind.indices:
                if j is not None:
                    contraction_indices[j].append(current_position)
                current_position += 1
        return contraction_indices

    def get_mapping_for_index(self, ind) -> list[_IndPos]:
        if ind >= self.number_of_contraction_indices:
            raise ValueError("index value exceeding the index range")
        positions: list[_IndPos] = []
        for i, arg_with_ind in enumerate(self.args_with_ind):
            for j, arg_ind in enumerate(arg_with_ind.indices):
                if ind == arg_ind:
                    positions.append(_IndPos(i, j))
        return positions

    def get_contraction_indices_to_ind_rel_pos(self) -> list[list[_IndPos]]:
        contraction_indices: list[list[_IndPos]] = [[] for i in range(self.number_of_contraction_indices)]
        for i, arg_with_ind in enumerate(self.args_with_ind):
            for j, ind in enumerate(arg_with_ind.indices):
                if ind is not None:
                    contraction_indices[ind].append(_IndPos(i, j))
        return contraction_indices

    def count_args_with_index(self, index: int) -> int:
        """
        Count the number of arguments that have the given index.
        """
        counter: int = 0
        for arg_with_ind in self.args_with_ind:
            if index in arg_with_ind.indices:
                counter += 1
        return counter

    def get_args_with_index(self, index: int) -> list[_ArgE]:
        """
        Get a list of arguments having the given index.
        """
        ret: list[_ArgE] = [i for i in self.args_with_ind if index in i.indices]
        return ret

    @property
    def number_of_diagonal_indices(self):
        data = set()
        for arg in self.args_with_ind:
            data.update({i for i in arg.indices if i is not None and i < 0})
        return len(data)

    def track_permutation_start(self):
        permutation = []
        perm_diag = []
        counter = 0
        counter2 = -1
        for arg_with_ind in self.args_with_ind:
            perm = []
            for i in arg_with_ind.indices:
                if i is not None:
                    if i < 0:
                        perm_diag.append(counter2)
                        counter2 -= 1
                    continue
                perm.append(counter)
                counter += 1
            permutation.append(perm)
        max_ind = max(max(i) if i else -1 for i in permutation) if permutation else -1
        perm_diag = [max_ind - i for i in perm_diag]
        self._track_permutation = permutation + [perm_diag]

    def track_permutation_merge(self, destination: _ArgE, from_element: _ArgE):
        index_destination = self.args_with_ind.index(destination)
        index_element = self.args_with_ind.index(from_element)
        self._track_permutation[index_destination].extend(self._track_permutation[index_element]) # type: ignore
        self._track_permutation.pop(index_element) # type: ignore

    def get_absolute_free_range(self, arg: _ArgE) -> typing.Tuple[int, int]:
        """
        Return the range of the free indices of the arg as absolute positions
        among all free indices.
        """
        counter = 0
        for arg_with_ind in self.args_with_ind:
            number_free_indices = len([i for i in arg_with_ind.indices if i is None])
            if arg_with_ind == arg:
                return counter, counter + number_free_indices
            counter += number_free_indices
        raise IndexError("argument not found")

    def get_absolute_range(self, arg: _ArgE) -> typing.Tuple[int, int]:
        """
        Return the absolute range of indices for arg, disregarding dummy
        indices.
        """
        counter = 0
        for arg_with_ind in self.args_with_ind:
            number_indices = len(arg_with_ind.indices)
            if arg_with_ind == arg:
                return counter, counter + number_indices
            counter += number_indices
        raise IndexError("argument not found")


def get_rank(expr):
    if isinstance(expr, (MatrixExpr, MatrixElement)):
        return 2
    if isinstance(expr, _CodegenArrayAbstract):
        return len(expr.shape)
    if isinstance(expr, NDimArray):
        return expr.rank()
    if isinstance(expr, Indexed):
        return expr.rank
    if isinstance(expr, IndexedBase):
        shape = expr.shape
        if shape is None:
            return -1
        else:
            return len(shape)
    if hasattr(expr, "shape"):
        return len(expr.shape)
    return 0


def _get_subrank(expr):
    if isinstance(expr, _CodegenArrayAbstract):
        return expr.subrank()
    return get_rank(expr)


def _get_subranks(expr):
    if isinstance(expr, _CodegenArrayAbstract):
        return expr.subranks
    else:
        return [get_rank(expr)]


def get_shape(expr):
    if hasattr(expr, "shape"):
        return expr.shape
    return ()


def nest_permutation(expr):
    if isinstance(expr, PermuteDims):
        return expr.nest_permutation()
    else:
        return expr


def _array_tensor_product(*args, **kwargs):
    return ArrayTensorProduct(*args, canonicalize=True, **kwargs)


def _array_contraction(expr, *contraction_indices, **kwargs):
    return ArrayContraction(expr, *contraction_indices, canonicalize=True, **kwargs)


def _array_diagonal(expr, *diagonal_indices, **kwargs):
    return ArrayDiagonal(expr, *diagonal_indices, canonicalize=True, **kwargs)


def _permute_dims(expr, permutation, **kwargs):
    return PermuteDims(expr, permutation, canonicalize=True, **kwargs)


def _array_add(*args, **kwargs):
    return ArrayAdd(*args, canonicalize=True, **kwargs)


def _get_array_element_or_slice(expr, indices):
    return ArrayElement(expr, indices)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\residue_ntheory.py ===
from __future__ import annotations

from sympy.external.gmpy import (gcd, lcm, invert, sqrt, jacobi,
                                 bit_scan1, remove)
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence, gf_csolve
from .primetest import isprime
from .generate import primerange
from .factor_ import factorint, _perfect_power
from .modular import crt
from sympy.utilities.decorator import deprecated
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int
from sympy.utilities.iterables import iproduct
from sympy.core.random import _randint, randint

from itertools import product


def n_order(a, n):
    r""" Returns the order of ``a`` modulo ``n``.

    Explanation
    ===========

    The order of ``a`` modulo ``n`` is the smallest integer
    ``k`` such that `a^k` leaves a remainder of 1 with ``n``.

    Parameters
    ==========

    a : integer
    n : integer, n > 1. a and n should be relatively prime

    Returns
    =======

    int : the order of ``a`` modulo ``n``

    Raises
    ======

    ValueError
        If `n \le 1` or `\gcd(a, n) \neq 1`.
        If ``a`` or ``n`` is not an integer.

    Examples
    ========

    >>> from sympy.ntheory import n_order
    >>> n_order(3, 7)
    6
    >>> n_order(4, 7)
    3

    See Also
    ========

    is_primitive_root
        We say that ``a`` is a primitive root of ``n``
        when the order of ``a`` modulo ``n`` equals ``totient(n)``

    """
    a, n = as_int(a), as_int(n)
    if n <= 1:
        raise ValueError("n should be an integer greater than 1")
    a = a % n
    # Trivial
    if a == 1:
        return 1
    if gcd(a, n) != 1:
        raise ValueError("The two numbers should be relatively prime")
    a_order = 1
    for p, e in factorint(n).items():
        pe = p**e
        pe_order = (p - 1) * p**(e - 1)
        factors = factorint(p - 1)
        if e > 1:
            factors[p] = e - 1
        order = 1
        for px, ex in factors.items():
            x = pow(a, pe_order // px**ex, pe)
            while x != 1:
                x = pow(x, px, pe)
                order *= px
        a_order = lcm(a_order, order)
    return int(a_order)


def _primitive_root_prime_iter(p):
    r""" Generates the primitive roots for a prime ``p``.

    Explanation
    ===========

    The primitive roots generated are not necessarily sorted.
    However, the first one is the smallest primitive root.

    Find the element whose order is ``p-1`` from the smaller one.
    If we can find the first primitive root ``g``, we can use the following theorem.

    .. math ::
        \operatorname{ord}(g^k) = \frac{\operatorname{ord}(g)}{\gcd(\operatorname{ord}(g), k)}

    From the assumption that `\operatorname{ord}(g)=p-1`,
    it is a necessary and sufficient condition for
    `\operatorname{ord}(g^k)=p-1` that `\gcd(p-1, k)=1`.

    Parameters
    ==========

    p : odd prime

    Yields
    ======

    int
        the primitive roots of ``p``

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_iter
    >>> sorted(_primitive_root_prime_iter(19))
    [2, 3, 10, 13, 14, 15]

    References
    ==========

    .. [1] W. Stein "Elementary Number Theory" (2011), page 44

    """
    if p == 3:
        yield 2
        return
    # Let p = +-1 (mod 4a). Legendre symbol (a/p) = 1, so `a` is not the primitive root.
    # Corollary : If p = +-1 (mod 8), then 2 is not the primitive root of p.
    g_min = 3 if p % 8 in [1, 7] else 2
    if p < 41:
        # small case
        g = 5 if p == 23 else g_min
    else:
        v = [(p - 1) // i for i in factorint(p - 1).keys()]
        for g in range(g_min, p):
            if all(pow(g, pw, p) != 1 for pw in v):
                break
    yield g
    # g**k is the primitive root of p iff gcd(p - 1, k) = 1
    for k in range(3, p, 2):
        if gcd(p - 1, k) == 1:
            yield pow(g, k, p)


def _primitive_root_prime_power_iter(p, e):
    r""" Generates the primitive roots of `p^e`.

    Explanation
    ===========

    Let ``g`` be the primitive root of ``p``.
    If `g^{p-1} \not\equiv 1 \pmod{p^2}`, then ``g`` is primitive root of `p^e`.
    Thus, if we find a primitive root ``g`` of ``p``,
    then `g, g+p, g+2p, \ldots, g+(p-1)p` are primitive roots of `p^2` except one.
    That one satisfies `\hat{g}^{p-1} \equiv 1 \pmod{p^2}`.
    If ``h`` is the primitive root of `p^2`,
    then `h, h+p^2, h+2p^2, \ldots, h+(p^{e-2}-1)p^e` are primitive roots of `p^e`.

    Parameters
    ==========

    p : odd prime
    e : positive integer

    Yields
    ======

    int
        the primitive roots of `p^e`

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_power_iter
    >>> sorted(_primitive_root_prime_power_iter(5, 2))
    [2, 3, 8, 12, 13, 17, 22, 23]

    """
    if e == 1:
        yield from _primitive_root_prime_iter(p)
    else:
        p2 = p**2
        for g in _primitive_root_prime_iter(p):
            t = (g - pow(g, 2 - p, p2)) % p2
            for k in range(0, p2, p):
                if k != t:
                    yield from (g + k + m for m in range(0, p**e, p2))


def _primitive_root_prime_power2_iter(p, e):
    r""" Generates the primitive roots of `2p^e`.

    Explanation
    ===========

    If ``g`` is the primitive root of ``p**e``,
    then the odd one of ``g`` and ``g+p**e`` is the primitive root of ``2*p**e``.

    Parameters
    ==========

    p : odd prime
    e : positive integer

    Yields
    ======

    int
        the primitive roots of `2p^e`

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_power2_iter
    >>> sorted(_primitive_root_prime_power2_iter(5, 2))
    [3, 13, 17, 23, 27, 33, 37, 47]

    """
    for g in _primitive_root_prime_power_iter(p, e):
        if g % 2 == 1:
            yield g
        else:
            yield g + p**e


def primitive_root(p, smallest=True):
    r""" Returns a primitive root of ``p`` or None.

    Explanation
    ===========

    For the definition of primitive root,
    see the explanation of ``is_primitive_root``.

    The primitive root of ``p`` exist only for
    `p = 2, 4, q^e, 2q^e` (``q`` is an odd prime).
    Now, if we know the primitive root of ``q``,
    we can calculate the primitive root of `q^e`,
    and if we know the primitive root of `q^e`,
    we can calculate the primitive root of `2q^e`.
    When there is no need to find the smallest primitive root,
    this property can be used to obtain a fast primitive root.
    On the other hand, when we want the smallest primitive root,
    we naively determine whether it is a primitive root or not.

    Parameters
    ==========

    p : integer, p > 1
    smallest : if True the smallest primitive root is returned or None

    Returns
    =======

    int | None :
        If the primitive root exists, return the primitive root of ``p``.
        If not, return None.

    Raises
    ======

    ValueError
        If `p \le 1` or ``p`` is not an integer.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import primitive_root
    >>> primitive_root(19)
    2
    >>> primitive_root(21) is None
    True
    >>> primitive_root(50, smallest=False)
    27

    See Also
    ========

    is_primitive_root

    References
    ==========

    .. [1] W. Stein "Elementary Number Theory" (2011), page 44
    .. [2] P. Hackman "Elementary Number Theory" (2009), Chapter C

    """
    p = as_int(p)
    if p <= 1:
        raise ValueError("p should be an integer greater than 1")
    if p <= 4:
        return p - 1
    p_even = p % 2 == 0
    if not p_even:
        q = p  # p is odd
    elif p % 4:
        q = p//2  # p had 1 factor of 2
    else:
        return None  # p had more than one factor of 2
    if isprime(q):
        e = 1
    else:
        m = _perfect_power(q, 3)
        if not m:
            return None
        q, e = m
        if not isprime(q):
            return None
    if not smallest:
        if p_even:
            return next(_primitive_root_prime_power2_iter(q, e))
        return next(_primitive_root_prime_power_iter(q, e))
    if p_even:
        for i in range(3, p, 2):
            if i % q and is_primitive_root(i, p):
                return i
    g = next(_primitive_root_prime_iter(q))
    if e == 1 or pow(g, q - 1, q**2) != 1:
        return g
    for i in range(g + 1, p):
        if i % q and is_primitive_root(i, p):
            return i


def is_primitive_root(a, p):
    r""" Returns True if ``a`` is a primitive root of ``p``.

    Explanation
    ===========

    ``a`` is said to be the primitive root of ``p`` if `\gcd(a, p) = 1` and
    `\phi(p)` is the smallest positive number s.t.

        `a^{\phi(p)} \equiv 1 \pmod{p}`.

    where `\phi(p)` is Euler's totient function.

    The primitive root of ``p`` exist only for
    `p = 2, 4, q^e, 2q^e` (``q`` is an odd prime).
    Hence, if it is not such a ``p``, it returns False.
    To determine the primitive root, we need to know
    the prime factorization of ``q-1``.
    The hardness of the determination depends on this complexity.

    Parameters
    ==========

    a : integer
    p : integer, ``p`` > 1. ``a`` and ``p`` should be relatively prime

    Returns
    =======

    bool : If True, ``a`` is the primitive root of ``p``.

    Raises
    ======

    ValueError
        If `p \le 1` or `\gcd(a, p) \neq 1`.
        If ``a`` or ``p`` is not an integer.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import totient
    >>> from sympy.ntheory import is_primitive_root, n_order
    >>> is_primitive_root(3, 10)
    True
    >>> is_primitive_root(9, 10)
    False
    >>> n_order(3, 10) == totient(10)
    True
    >>> n_order(9, 10) == totient(10)
    False

    See Also
    ========

    primitive_root

    """
    a, p = as_int(a), as_int(p)
    if p <= 1:
        raise ValueError("p should be an integer greater than 1")
    a = a % p
    if gcd(a, p) != 1:
        raise ValueError("The two numbers should be relatively prime")
    # Primitive root of p exist only for
    # p = 2, 4, q**e, 2*q**e (q is odd prime)
    if p <= 4:
        # The primitive root is only p-1.
        return a == p - 1
    if p % 2:
        q = p  # p is odd
    elif p % 4:
        q = p//2  # p had 1 factor of 2
    else:
        return False  # p had more than one factor of 2
    if isprime(q):
        group_order = q - 1
        factors = factorint(q - 1).keys()
    else:
        m = _perfect_power(q, 3)
        if not m:
            return False
        q, e = m
        if not isprime(q):
            return False
        group_order = q**(e - 1)*(q - 1)
        factors = set(factorint(q - 1).keys())
        factors.add(q)
    return all(pow(a, group_order // prime, p) != 1 for prime in factors)


def _sqrt_mod_tonelli_shanks(a, p):
    """
    Returns the square root in the case of ``p`` prime with ``p == 1 (mod 8)``

    Assume that the root exists.

    Parameters
    ==========

    a : int
    p : int
        prime number. should be ``p % 8 == 1``

    Returns
    =======

    int : Generally, there are two roots, but only one is returned.
          Which one is returned is random.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _sqrt_mod_tonelli_shanks
    >>> _sqrt_mod_tonelli_shanks(2, 17) in [6, 11]
    True

    References
    ==========

    .. [1] Carl Pomerance, Richard Crandall, Prime Numbers: A Computational Perspective,
           2nd Edition (2005), page 101, ISBN:978-0387252827

    """
    s = bit_scan1(p - 1)
    t = p >> s
    # find a non-quadratic residue
    if p % 12 == 5:
        # Legendre symbol (3/p) == -1 if p % 12 in [5, 7]
        d = 3
    elif p % 5 in [2, 3]:
        # Legendre symbol (5/p) == -1 if p % 5 in [2, 3]
        d = 5
    else:
        while 1:
            d = randint(6, p - 1)
            if jacobi(d, p) == -1:
                break
    #assert legendre_symbol(d, p) == -1
    A = pow(a, t, p)
    D = pow(d, t, p)
    m = 0
    for i in range(s):
        adm = A*pow(D, m, p) % p
        adm = pow(adm, 2**(s - 1 - i), p)
        if adm % p == p - 1:
            m += 2**i
    #assert A*pow(D, m, p) % p == 1
    x = pow(a, (t + 1)//2, p)*pow(D, m//2, p) % p
    return x


def sqrt_mod(a, p, all_roots=False):
    """
    Find a root of ``x**2 = a mod p``.

    Parameters
    ==========

    a : integer
    p : positive integer
    all_roots : if True the list of roots is returned or None

    Notes
    =====

    If there is no root it is returned None; else the returned root
    is less or equal to ``p // 2``; in general is not the smallest one.
    It is returned ``p // 2`` only if it is the only root.

    Use ``all_roots`` only when it is expected that all the roots fit
    in memory; otherwise use ``sqrt_mod_iter``.

    Examples
    ========

    >>> from sympy.ntheory import sqrt_mod
    >>> sqrt_mod(11, 43)
    21
    >>> sqrt_mod(17, 32, True)
    [7, 9, 23, 25]
    """
    if all_roots:
        return sorted(sqrt_mod_iter(a, p))
    p = abs(as_int(p))
    halfp = p // 2
    x = None
    for r in sqrt_mod_iter(a, p):
        if r < halfp:
            return r
        elif r > halfp:
            return p - r
        else:
            x = r
    return x


def sqrt_mod_iter(a, p, domain=int):
    """
    Iterate over solutions to ``x**2 = a mod p``.

    Parameters
    ==========

    a : integer
    p : positive integer
    domain : integer domain, ``int``, ``ZZ`` or ``Integer``

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import sqrt_mod_iter
    >>> list(sqrt_mod_iter(11, 43))
    [21, 22]

    See Also
    ========

    sqrt_mod : Same functionality, but you want a sorted list or only one solution.

    """
    a, p = as_int(a), abs(as_int(p))
    v = []
    pv = []
    _product = product
    for px, ex in factorint(p).items():
        if a % px:
            # `len(rx)` is at most 4
            rx = _sqrt_mod_prime_power(a, px, ex)
        else:
            # `len(list(rx))` can be assumed to be large.
            # The `itertools.product` is disadvantageous in terms of memory usage.
            # It is also inferior to iproduct in speed if not all Cartesian products are needed.
            rx = _sqrt_mod1(a, px, ex)
            _product = iproduct
        if not rx:
            return
        v.append(rx)
        pv.append(px**ex)
    if len(v) == 1:
        yield from map(domain, v[0])
    else:
        mm, e, s = gf_crt1(pv, ZZ)
        for vx in _product(*v):
            yield domain(gf_crt2(vx, pv, mm, e, s, ZZ))


def _sqrt_mod_prime_power(a, p, k):
    """
    Find the solutions to ``x**2 = a mod p**k`` when ``a % p != 0``.
    If no solution exists, return ``None``.
    Solutions are returned in an ascending list.

    Parameters
    ==========

    a : integer
    p : prime number
    k : positive integer

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
    >>> _sqrt_mod_prime_power(11, 43, 1)
    [21, 22]

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 160
    .. [2] http://www.numbertheory.org/php/squareroot.html
    .. [3] [Gathen99]_
    """
    pk = p**k
    a = a % pk

    if p == 2:
        # see Ref.[2]
        if a % 8 != 1:
            return None
        # Trivial
        if k <= 3:
            return list(range(1, pk, 2))
        r = 1
        # r is one of the solutions to x**2 - a = 0 (mod 2**3).
        # Hensel lift them to solutions of x**2 - a = 0 (mod 2**k)
        # if r**2 - a = 0 mod 2**nx but not mod 2**(nx+1)
        # then r + 2**(nx - 1) is a root mod 2**(nx+1)
        for nx in range(3, k):
            if ((r**2 - a) >> nx) % 2:
                r += 1 << (nx - 1)
        # r is a solution of x**2 - a = 0 (mod 2**k), and
        # there exist other solutions -r, r+h, -(r+h), and these are all solutions.
        h = 1 << (k - 1)
        return sorted([r, pk - r, (r + h) % pk, -(r + h) % pk])

    # If the Legendre symbol (a/p) is not 1, no solution exists.
    if jacobi(a, p) != 1:
        return None
    if p % 4 == 3:
        res = pow(a, (p + 1) // 4, p)
    elif p % 8 == 5:
        res = pow(a, (p + 3) // 8, p)
        if pow(res, 2, p) != a % p:
            res = res * pow(2, (p - 1) // 4, p) % p
    else:
        res = _sqrt_mod_tonelli_shanks(a, p)
    if k > 1:
        # Hensel lifting with Newton iteration, see Ref.[3] chapter 9
        # with f(x) = x**2 - a; one has f'(a) != 0 (mod p) for p != 2
        px = p
        for _ in range(k.bit_length() - 1):
            px = px**2
            frinv = invert(2*res, px)
            res = (res - (res**2 - a)*frinv) % px
        if k & (k - 1): # If k is not a power of 2
            frinv = invert(2*res, pk)
            res = (res - (res**2 - a)*frinv) % pk
    return sorted([res, pk - res])


def _sqrt_mod1(a, p, n):
    """
    Find solution to ``x**2 == a mod p**n`` when ``a % p == 0``.
    If no solution exists, return ``None``.

    Parameters
    ==========

    a : integer
    p : prime number, p must divide a
    n : positive integer

    References
    ==========

    .. [1] http://www.numbertheory.org/php/squareroot.html
    """
    pn = p**n
    a = a % pn
    if a == 0:
        # case gcd(a, p**k) = p**n
        return range(0, pn, p**((n + 1) // 2))
    # case gcd(a, p**k) = p**r, r < n
    a, r = remove(a, p)
    if r % 2 == 1:
        return None
    res = _sqrt_mod_prime_power(a, p, n - r)
    if res is None:
        return None
    m = r // 2
    return (x for rx in res for x in range(rx*p**m, pn, p**(n - m)))


def is_quad_residue(a, p):
    """
    Returns True if ``a`` (mod ``p``) is in the set of squares mod ``p``,
    i.e a % p in set([i**2 % p for i in range(p)]).

    Parameters
    ==========

    a : integer
    p : positive integer

    Returns
    =======

    bool : If True, ``x**2 == a (mod p)`` has solution.

    Raises
    ======

    ValueError
        If ``a``, ``p`` is not integer.
        If ``p`` is not positive.

    Examples
    ========

    >>> from sympy.ntheory import is_quad_residue
    >>> is_quad_residue(21, 100)
    True

    Indeed, ``pow(39, 2, 100)`` would be 21.

    >>> is_quad_residue(21, 120)
    False

    That is, for any integer ``x``, ``pow(x, 2, 120)`` is not 21.

    If ``p`` is an odd
    prime, an iterative method is used to make the determination:

    >>> from sympy.ntheory import is_quad_residue
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]
    >>> [j for j in range(7) if is_quad_residue(j, 7)]
    [0, 1, 2, 4]

    See Also
    ========

    legendre_symbol, jacobi_symbol, sqrt_mod
    """
    a, p = as_int(a), as_int(p)
    if p < 1:
        raise ValueError('p must be > 0')
    a %= p
    if a < 2 or p < 3:
        return True
    # Since we want to compute the Jacobi symbol,
    # we separate p into the odd part and the rest.
    t = bit_scan1(p)
    if t:
        # The existence of a solution to a power of 2 is determined
        # using the logic of `p==2` in `_sqrt_mod_prime_power` and `_sqrt_mod1`.
        a_ = a % (1 << t)
        if a_:
            r = bit_scan1(a_)
            if r % 2 or (a_ >> r) & 6:
                return False
        p >>= t
        a %= p
        if a < 2 or p < 3:
            return True
    # If Jacobi symbol is -1 or p is prime, can be determined by Jacobi symbol only
    j = jacobi(a, p)
    if j == -1 or isprime(p):
        return j == 1
    # Checks if `x**2 = a (mod p)` has a solution
    for px, ex in factorint(p).items():
        if a % px:
            if jacobi(a, px) != 1:
                return False
        else:
            a_ = a % px**ex
            if a_ == 0:
                continue
            a_, r = remove(a_, px)
            if r % 2 or jacobi(a_, px) != 1:
                return False
    return True


def is_nthpow_residue(a, n, m):
    """
    Returns True if ``x**n == a (mod m)`` has solutions.

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 76

    """
    a = a % m
    a, n, m = as_int(a), as_int(n), as_int(m)
    if m <= 0:
        raise ValueError('m must be > 0')
    if n < 0:
        raise ValueError('n must be >= 0')
    if n == 0:
        if m == 1:
            return False
        return a == 1
    if a == 0:
        return True
    if n == 1:
        return True
    if n == 2:
        return is_quad_residue(a, m)
    return all(_is_nthpow_residue_bign_prime_power(a, n, p, e)
               for p, e in factorint(m).items())


def _is_nthpow_residue_bign_prime_power(a, n, p, k):
    r"""
    Returns True if `x^n = a \pmod{p^k}` has solutions for `n > 2`.

    Parameters
    ==========

    a : positive integer
    n : integer, n > 2
    p : prime number
    k : positive integer

    """
    while a % p == 0:
        a %= pow(p, k)
        if not a:
            return True
        a, mu = remove(a, p)
        if mu % n:
            return False
        k -= mu
    if p != 2:
        f = p**(k - 1)*(p - 1) # f = totient(p**k)
        return pow(a, f // gcd(f, n), pow(p, k)) == 1
    if n & 1:
        return True
    c = min(bit_scan1(n) + 2, k)
    return a % pow(2, c) == 1


def _nthroot_mod1(s, q, p, all_roots):
    """
    Root of ``x**q = s mod p``, ``p`` prime and ``q`` divides ``p - 1``.
    Assume that the root exists.

    Parameters
    ==========

    s : integer
    q : integer, n > 2. ``q`` divides ``p - 1``.
    p : prime number
    all_roots : if False returns the smallest root, else the list of roots

    Returns
    =======

    list[int] | int :
        Root of ``x**q = s mod p``. If ``all_roots == True``,
        returned ascending list. otherwise, returned an int.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _nthroot_mod1
    >>> _nthroot_mod1(5, 3, 13, False)
    7
    >>> _nthroot_mod1(13, 4, 17, True)
    [3, 5, 12, 14]

    References
    ==========

    .. [1] A. M. Johnston, A Generalized qth Root Algorithm,
           ACM-SIAM Symposium on Discrete Algorithms (1999), pp. 929-930

    """
    g = next(_primitive_root_prime_iter(p))
    r = s
    for qx, ex in factorint(q).items():
        f = (p - 1) // qx**ex
        while f % qx == 0:
            f //= qx
        z = f*invert(-f, qx)
        x = (1 + z) // qx
        t = discrete_log(p, pow(r, f, p), pow(g, f*qx, p))
        for _ in range(ex):
            # assert t == discrete_log(p, pow(r, f, p), pow(g, f*qx, p))
            r = pow(r, x, p)*pow(g, -z*t % (p - 1), p) % p
            t //= qx
    res = [r]
    h = pow(g, (p - 1) // q, p)
    #assert pow(h, q, p) == 1
    hx = r
    for _ in range(q - 1):
        hx = (hx*h) % p
        res.append(hx)
    if all_roots:
        res.sort()
        return res
    return min(res)


def _nthroot_mod_prime_power(a, n, p, k):
    """ Root of ``x**n = a mod p**k``.

    Parameters
    ==========

    a : integer
    n : integer, n > 2
    p : prime number
    k : positive integer

    Returns
    =======

    list[int] :
        Ascending list of roots of ``x**n = a mod p**k``.
        If no solution exists, return ``[]``.

    """
    if not _is_nthpow_residue_bign_prime_power(a, n, p, k):
        return []
    a_mod_p = a % p
    if a_mod_p == 0:
        base_roots = [0]
    elif (p - 1) % n == 0:
        base_roots = _nthroot_mod1(a_mod_p, n, p, all_roots=True)
    else:
        # The roots of ``x**n - a = 0 (mod p)`` are roots of
        # ``gcd(x**n - a, x**(p - 1) - 1) = 0 (mod p)``
        pa = n
        pb = p - 1
        b = 1
        if pa < pb:
            a_mod_p, pa, b, pb = b, pb, a_mod_p, pa
        # gcd(x**pa - a, x**pb - b) = gcd(x**pb - b, x**pc - c)
        # where pc = pa % pb; c = b**-q * a mod p
        while pb:
            q, pc = divmod(pa, pb)
            c = pow(b, -q, p) * a_mod_p % p
            pa, pb = pb, pc
            a_mod_p, b = b, c
        if pa == 1:
            base_roots = [a_mod_p]
        elif pa == 2:
            base_roots = sqrt_mod(a_mod_p, p, all_roots=True)
        else:
            base_roots = _nthroot_mod1(a_mod_p, pa, p, all_roots=True)
    if k == 1:
        return base_roots
    a %= p**k
    tot_roots = set()
    for root in base_roots:
        diff = pow(root, n - 1, p)*n % p
        new_base = p
        if diff != 0:
            m_inv = invert(diff, p)
            for _ in range(k - 1):
                new_base *= p
                tmp = pow(root, n, new_base) - a
                tmp *= m_inv
                root = (root - tmp) % new_base
            tot_roots.add(root)
        else:
            roots_in_base = {root}
            for _ in range(k - 1):
                new_base *= p
                new_roots = set()
                for k_ in roots_in_base:
                    if pow(k_, n, new_base) != a % new_base:
                        continue
                    while k_ not in new_roots:
                        new_roots.add(k_)
                        k_ = (k_ + (new_base // p)) % new_base
                roots_in_base = new_roots
            tot_roots = tot_roots | roots_in_base
    return sorted(tot_roots)


def nthroot_mod(a, n, p, all_roots=False):
    """
    Find the solutions to ``x**n = a mod p``.

    Parameters
    ==========

    a : integer
    n : positive integer
    p : positive integer
    all_roots : if False returns the smallest root, else the list of roots

    Returns
    =======

        list[int] | int | None :
            solutions to ``x**n = a mod p``.
            The table of the output type is:

            ========== ========== ==========
            all_roots  has roots  Returns
            ========== ========== ==========
            True       Yes        list[int]
            True       No         []
            False      Yes        int
            False      No         None
            ========== ========== ==========

    Raises
    ======

        ValueError
            If ``a``, ``n`` or ``p`` is not integer.
            If ``n`` or ``p`` is not positive.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import nthroot_mod
    >>> nthroot_mod(11, 4, 19)
    8
    >>> nthroot_mod(11, 4, 19, True)
    [8, 11]
    >>> nthroot_mod(68, 3, 109)
    23

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 76

    """
    a = a % p
    a, n, p = as_int(a), as_int(n), as_int(p)

    if n < 1:
        raise ValueError("n should be positive")
    if p < 1:
        raise ValueError("p should be positive")
    if n == 1:
        return [a] if all_roots else a
    if n == 2:
        return sqrt_mod(a, p, all_roots)
    base = []
    prime_power = []
    for q, e in factorint(p).items():
        tot_roots = _nthroot_mod_prime_power(a, n, q, e)
        if not tot_roots:
            return [] if all_roots else None
        prime_power.append(q**e)
        base.append(sorted(tot_roots))
    P, E, S = gf_crt1(prime_power, ZZ)
    ret = sorted(map(int, {gf_crt2(c, prime_power, P, E, S, ZZ)
                           for c in product(*base)}))
    if all_roots:
        return ret
    if ret:
        return ret[0]


def quadratic_residues(p) -> list[int]:
    """
    Returns the list of quadratic residues.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import quadratic_residues
    >>> quadratic_residues(7)
    [0, 1, 2, 4]
    """
    p = as_int(p)
    r = {pow(i, 2, p) for i in range(p // 2 + 1)}
    return sorted(r)


@deprecated("""\
The `sympy.ntheory.residue_ntheory.legendre_symbol` has been moved to `sympy.functions.combinatorial.numbers.legendre_symbol`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def legendre_symbol(a, p):
    r"""
    Returns the Legendre symbol `(a / p)`.

    .. deprecated:: 1.13

        The ``legendre_symbol`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.legendre_symbol`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    For an integer ``a`` and an odd prime ``p``, the Legendre symbol is
    defined as

    .. math ::
        \genfrac(){}{}{a}{p} = \begin{cases}
             0 & \text{if } p \text{ divides } a\\
             1 & \text{if } a \text{ is a quadratic residue modulo } p\\
            -1 & \text{if } a \text{ is a quadratic nonresidue modulo } p
        \end{cases}

    Parameters
    ==========

    a : integer
    p : odd prime

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import legendre_symbol
    >>> [legendre_symbol(i, 7) for i in range(7)]
    [0, 1, 1, -1, 1, -1, -1]
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]

    See Also
    ========

    is_quad_residue, jacobi_symbol

    """
    from sympy.functions.combinatorial.numbers import legendre_symbol as _legendre_symbol
    return _legendre_symbol(a, p)


@deprecated("""\
The `sympy.ntheory.residue_ntheory.jacobi_symbol` has been moved to `sympy.functions.combinatorial.numbers.jacobi_symbol`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def jacobi_symbol(m, n):
    r"""
    Returns the Jacobi symbol `(m / n)`.

    .. deprecated:: 1.13

        The ``jacobi_symbol`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.jacobi_symbol`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    For any integer ``m`` and any positive odd integer ``n`` the Jacobi symbol
    is defined as the product of the Legendre symbols corresponding to the
    prime factors of ``n``:

    .. math ::
        \genfrac(){}{}{m}{n} =
            \genfrac(){}{}{m}{p^{1}}^{\alpha_1}
            \genfrac(){}{}{m}{p^{2}}^{\alpha_2}
            ...
            \genfrac(){}{}{m}{p^{k}}^{\alpha_k}
            \text{ where } n =
                p_1^{\alpha_1}
                p_2^{\alpha_2}
                ...
                p_k^{\alpha_k}

    Like the Legendre symbol, if the Jacobi symbol `\genfrac(){}{}{m}{n} = -1`
    then ``m`` is a quadratic nonresidue modulo ``n``.

    But, unlike the Legendre symbol, if the Jacobi symbol
    `\genfrac(){}{}{m}{n} = 1` then ``m`` may or may not be a quadratic residue
    modulo ``n``.

    Parameters
    ==========

    m : integer
    n : odd positive integer

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import jacobi_symbol, legendre_symbol
    >>> from sympy import S
    >>> jacobi_symbol(45, 77)
    -1
    >>> jacobi_symbol(60, 121)
    1

    The relationship between the ``jacobi_symbol`` and ``legendre_symbol`` can
    be demonstrated as follows:

    >>> L = legendre_symbol
    >>> S(45).factors()
    {3: 2, 5: 1}
    >>> jacobi_symbol(7, 45) == L(7, 3)**2 * L(7, 5)**1
    True

    See Also
    ========

    is_quad_residue, legendre_symbol
    """
    from sympy.functions.combinatorial.numbers import jacobi_symbol as _jacobi_symbol
    return _jacobi_symbol(m, n)


@deprecated("""\
The `sympy.ntheory.residue_ntheory.mobius` has been moved to `sympy.functions.combinatorial.numbers.mobius`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def mobius(n):
    """
    Mobius function maps natural number to {-1, 0, 1}

    .. deprecated:: 1.13

        The ``mobius`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.mobius`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    It is defined as follows:
        1) `1` if `n = 1`.
        2) `0` if `n` has a squared prime factor.
        3) `(-1)^k` if `n` is a square-free positive integer with `k`
           number of prime factors.

    It is an important multiplicative function in number theory
    and combinatorics.  It has applications in mathematical series,
    algebraic number theory and also physics (Fermion operator has very
    concrete realization with Mobius Function model).

    Parameters
    ==========

    n : positive integer

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import mobius
    >>> mobius(13*7)
    1
    >>> mobius(1)
    1
    >>> mobius(13*7*5)
    -1
    >>> mobius(13**2)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_function
    .. [2] Thomas Koshy "Elementary Number Theory with Applications"

    """
    from sympy.functions.combinatorial.numbers import mobius as _mobius
    return _mobius(n)


def _discrete_log_trial_mul(n, a, b, order=None):
    """
    Trial multiplication algorithm for computing the discrete logarithm of
    ``a`` to the base ``b`` modulo ``n``.

    The algorithm finds the discrete logarithm using exhaustive search. This
    naive method is used as fallback algorithm of ``discrete_log`` when the
    group order is very small. The value ``n`` must be greater than 1.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_trial_mul
    >>> _discrete_log_trial_mul(41, 15, 7)
    3

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    a %= n
    b %= n
    if order is None:
        order = n
    x = 1
    for i in range(order):
        if x == a:
            return i
        x = x * b % n
    raise ValueError("Log does not exist")


def _discrete_log_shanks_steps(n, a, b, order=None):
    """
    Baby-step giant-step algorithm for computing the discrete logarithm of
    ``a`` to the base ``b`` modulo ``n``.

    The algorithm is a time-memory trade-off of the method of exhaustive
    search. It uses `O(sqrt(m))` memory, where `m` is the group order.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_shanks_steps
    >>> _discrete_log_shanks_steps(41, 15, 7)
    3

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    a %= n
    b %= n
    if order is None:
        order = n_order(b, n)
    m = sqrt(order) + 1
    T = {}
    x = 1
    for i in range(m):
        T[x] = i
        x = x * b % n
    z = pow(b, -m, n)
    x = a
    for i in range(m):
        if x in T:
            return i * m + T[x]
        x = x * z % n
    raise ValueError("Log does not exist")


def _discrete_log_pollard_rho(n, a, b, order=None, retries=10, rseed=None):
    """
    Pollard's Rho algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    It is a randomized algorithm with the same expected running time as
    ``_discrete_log_shanks_steps``, but requires a negligible amount of memory.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_pollard_rho
    >>> _discrete_log_pollard_rho(227, 3**7, 3)
    7

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    a %= n
    b %= n

    if order is None:
        order = n_order(b, n)
    randint = _randint(rseed)

    for i in range(retries):
        aa = randint(1, order - 1)
        ba = randint(1, order - 1)
        xa = pow(b, aa, n) * pow(a, ba, n) % n

        c = xa % 3
        if c == 0:
            xb = a * xa % n
            ab = aa
            bb = (ba + 1) % order
        elif c == 1:
            xb = xa * xa % n
            ab = (aa + aa) % order
            bb = (ba + ba) % order
        else:
            xb = b * xa % n
            ab = (aa + 1) % order
            bb = ba

        for j in range(order):
            c = xa % 3
            if c == 0:
                xa = a * xa % n
                ba = (ba + 1) % order
            elif c == 1:
                xa = xa * xa % n
                aa = (aa + aa) % order
                ba = (ba + ba) % order
            else:
                xa = b * xa % n
                aa = (aa + 1) % order

            c = xb % 3
            if c == 0:
                xb = a * xb % n
                bb = (bb + 1) % order
            elif c == 1:
                xb = xb * xb % n
                ab = (ab + ab) % order
                bb = (bb + bb) % order
            else:
                xb = b * xb % n
                ab = (ab + 1) % order

            c = xb % 3
            if c == 0:
                xb = a * xb % n
                bb = (bb + 1) % order
            elif c == 1:
                xb = xb * xb % n
                ab = (ab + ab) % order
                bb = (bb + bb) % order
            else:
                xb = b * xb % n
                ab = (ab + 1) % order

            if xa == xb:
                r = (ba - bb) % order
                try:
                    e = invert(r, order) * (ab - aa) % order
                    if (pow(b, e, n) - a) % n == 0:
                        return e
                except ZeroDivisionError:
                    pass
                break
    raise ValueError("Pollard's Rho failed to find logarithm")


def _discrete_log_is_smooth(n: int, factorbase: list):
    """Try to factor n with respect to a given factorbase.
    Upon success a list of exponents with respect to the factorbase is returned.
    Otherwise None."""
    factors = [0]*len(factorbase)
    for i, p in enumerate(factorbase):
        while n % p == 0: # divide by p as many times as possible
            factors[i] += 1
            n = n // p
    if n != 1:
        return None # the number factors if at the end nothing is left
    return factors


def _discrete_log_index_calculus(n, a, b, order, rseed=None):
    """
    Index Calculus algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    The group order must be given and prime. It is not suitable for small orders
    and the algorithm might fail to find a solution in such situations.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_index_calculus
    >>> _discrete_log_index_calculus(24570203447, 23859756228, 2, 12285101723)
    4519867240

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    randint = _randint(rseed)
    from math import sqrt, exp, log
    a %= n
    b %= n
    # assert isprime(order), "The order of the base must be prime."
    # First choose a heuristic the bound B for the factorbase.
    # We have added an extra term to the asymptotic value which
    # is closer to the theoretical optimum for n up to 2^70.
    B = int(exp(0.5 * sqrt( log(n) * log(log(n)) )*( 1 + 1/log(log(n)) )))
    max = 5 * B * B  # expected number of tries to find a relation
    factorbase = list(primerange(B)) # compute the factorbase
    lf = len(factorbase) # length of the factorbase
    ordermo = order-1
    abx = a
    for x in range(order):
        if abx == 1:
            return (order - x) % order
        relationa = _discrete_log_is_smooth(abx, factorbase)
        if relationa:
            relationa = [r % order for r in relationa] + [x]
            break
        abx = abx * b % n # abx = a*pow(b, x, n) % n

    else:
        raise ValueError("Index Calculus failed")

    relations = [None] * lf
    k = 1  # number of relations found
    kk = 0
    while k < 3 * lf and kk < max:  # find relations for all primes in our factor base
        x = randint(1,ordermo)
        relation = _discrete_log_is_smooth(pow(b,x,n), factorbase)
        if relation is None:
            kk += 1
            continue
        k += 1
        kk = 0
        relation += [ x ]
        index = lf  # determine the index of the first nonzero entry
        for i in range(lf):
            ri = relation[i] % order
            if ri> 0 and relations[i] is not None:  # make this entry zero if we can
                for j in range(lf+1):
                    relation[j] = (relation[j] - ri*relations[i][j]) % order
            else:
                relation[i] = ri
            if relation[i] > 0 and index == lf:  # is this the index of the first nonzero entry?
                index = i
        if index == lf or relations[index] is not None:  # the relation contains no new information
            continue
        # the relation contains new information
        rinv = pow(relation[index],-1,order)  # normalize the first nonzero entry
        for j in range(index,lf+1):
            relation[j] = rinv * relation[j] % order
        relations[index] = relation
        for i in range(lf):  # subtract the new relation from the one for a
            if relationa[i] > 0 and relations[i] is not None:
                rbi = relationa[i]
                for j in range(lf+1):
                    relationa[j] = (relationa[j] - rbi*relations[i][j]) % order
            if relationa[i] > 0:  # the index of the first nonzero entry
                break  # we do not need to reduce further at this point
        else:  # all unknowns are gone
            #print(f"Success after {k} relations out of {lf}")
            x = (order -relationa[lf]) % order
            if pow(b,x,n) == a:
                return x
            raise ValueError("Index Calculus failed")
    raise ValueError("Index Calculus failed")


def _discrete_log_pohlig_hellman(n, a, b, order=None, order_factors=None):
    """
    Pohlig-Hellman algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    In order to compute the discrete logarithm, the algorithm takes advantage
    of the factorization of the group order. It is more efficient when the
    group order factors into many small primes.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_pohlig_hellman
    >>> _discrete_log_pohlig_hellman(251, 210, 71)
    197

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    from .modular import crt
    a %= n
    b %= n

    if order is None:
        order = n_order(b, n)
    if order_factors is None:
        order_factors = factorint(order)
    l = [0] * len(order_factors)

    for i, (pi, ri) in enumerate(order_factors.items()):
        for j in range(ri):
            aj = pow(a * pow(b, -l[i], n), order // pi**(j + 1), n)
            bj = pow(b, order // pi, n)
            cj = discrete_log(n, aj, bj, pi, True)
            l[i] += cj * pi**j

    d, _ = crt([pi**ri for pi, ri in order_factors.items()], l)
    return d


def discrete_log(n, a, b, order=None, prime_order=None):
    """
    Compute the discrete logarithm of ``a`` to the base ``b`` modulo ``n``.

    This is a recursive function to reduce the discrete logarithm problem in
    cyclic groups of composite order to the problem in cyclic groups of prime
    order.

    It employs different algorithms depending on the problem (subgroup order
    size, prime order or not):

        * Trial multiplication
        * Baby-step giant-step
        * Pollard's Rho
        * Index Calculus
        * Pohlig-Hellman

    Examples
    ========

    >>> from sympy.ntheory import discrete_log
    >>> discrete_log(41, 15, 7)
    3

    References
    ==========

    .. [1] https://mathworld.wolfram.com/DiscreteLogarithm.html
    .. [2] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).

    """
    from math import sqrt, log
    n, a, b = as_int(n), as_int(a), as_int(b)

    if n < 1:
        raise ValueError("n should be positive")
    if n == 1:
        return 0

    if order is None:
        # Compute the order and its factoring in one pass
        # order = totient(n), factors = factorint(order)
        factors = {}
        for px, kx in factorint(n).items():
            if kx > 1:
                if px in factors:
                    factors[px] += kx - 1
                else:
                    factors[px] = kx - 1
            for py, ky in factorint(px - 1).items():
                if py in factors:
                    factors[py] += ky
                else:
                    factors[py] = ky
        order = 1
        for px, kx in factors.items():
            order *= px**kx
        # Now the `order` is the order of the group and factors = factorint(order)
        # The order of `b` divides the order of the group.
        order_factors = {}
        for p, e in factors.items():
            i = 0
            for _ in range(e):
                if pow(b, order // p, n) == 1:
                    order //= p
                    i += 1
                else:
                    break
            if i < e:
                order_factors[p] = e - i

    if prime_order is None:
        prime_order = isprime(order)

    if order < 1000:
        return _discrete_log_trial_mul(n, a, b, order)
    elif prime_order:
        # Shanks and Pollard rho are O(sqrt(order)) while index calculus is O(exp(2*sqrt(log(n)log(log(n)))))
        # we compare the expected running times to determine the algorithm which is expected to be faster
        if 4*sqrt(log(n)*log(log(n))) < log(order) - 10:  # the number 10 was determined experimental
            return _discrete_log_index_calculus(n, a, b, order)
        elif order < 1000000000000:
            # Shanks seems typically faster, but uses O(sqrt(order)) memory
            return _discrete_log_shanks_steps(n, a, b, order)
        return _discrete_log_pollard_rho(n, a, b, order)

    return _discrete_log_pohlig_hellman(n, a, b, order, order_factors)



def quadratic_congruence(a, b, c, n):
    r"""
    Find the solutions to `a x^2 + b x + c \equiv 0 \pmod{n}`.

    Parameters
    ==========

    a : int
    b : int
    c : int
    n : int
        A positive integer.

    Returns
    =======

    list[int] :
        A sorted list of solutions. If no solution exists, ``[]``.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import quadratic_congruence
    >>> quadratic_congruence(2, 5, 3, 7) # 2x^2 + 5x + 3 = 0 (mod 7)
    [2, 6]
    >>> quadratic_congruence(8, 6, 4, 15) # No solution
    []

    See Also
    ========

    polynomial_congruence : Solve the polynomial congruence

    """
    a = as_int(a)
    b = as_int(b)
    c = as_int(c)
    n = as_int(n)
    if n <= 1:
        raise ValueError("n should be an integer greater than 1")
    a %= n
    b %= n
    c %= n

    if a == 0:
        return linear_congruence(b, -c, n)
    if n == 2:
        # assert a == 1
        roots = []
        if c == 0:
            roots.append(0)
        if (b + c) % 2:
            roots.append(1)
        return roots
    if gcd(2*a, n) == 1:
        inv_a = invert(a, n)
        b *= inv_a
        c *= inv_a
        if b % 2:
            b += n
        b >>= 1
        return sorted((i - b) % n for i in sqrt_mod_iter(b**2 - c, n))
    res = set()
    for i in sqrt_mod_iter(b**2 - 4*a*c, 4*a*n):
        q, rem = divmod(i - b, 2*a)
        if rem == 0:
            res.add(q % n)

    return sorted(res)


def _valid_expr(expr):
    """
    return coefficients of expr if it is a univariate polynomial
    with integer coefficients else raise a ValueError.
    """

    if not expr.is_polynomial():
        raise ValueError("The expression should be a polynomial")
    polynomial = Poly(expr)
    if not polynomial.is_univariate:
        raise ValueError("The expression should be univariate")
    if not polynomial.domain == ZZ:
        raise ValueError("The expression should should have integer coefficients")
    return polynomial.all_coeffs()


def polynomial_congruence(expr, m):
    """
    Find the solutions to a polynomial congruence equation modulo m.

    Parameters
    ==========

    expr : integer coefficient polynomial
    m : positive integer

    Examples
    ========

    >>> from sympy.ntheory import polynomial_congruence
    >>> from sympy.abc import x
    >>> expr = x**6 - 2*x**5 -35
    >>> polynomial_congruence(expr, 6125)
    [3257]

    See Also
    ========

    sympy.polys.galoistools.gf_csolve : low level solving routine used by this routine

    """
    coefficients = _valid_expr(expr)
    coefficients = [num % m for num in coefficients]
    rank = len(coefficients)
    if rank == 3:
        return quadratic_congruence(*coefficients, m)
    if rank == 2:
        return quadratic_congruence(0, *coefficients, m)
    if coefficients[0] == 1 and 1 + coefficients[-1] == sum(coefficients):
        return nthroot_mod(-coefficients[-1], rank - 1, m, True)
    return gf_csolve(coefficients, m)


def binomial_mod(n, m, k):
    """Compute ``binomial(n, m) % k``.

    Explanation
    ===========

    Returns ``binomial(n, m) % k`` using a generalization of Lucas'
    Theorem for prime powers given by Granville [1]_, in conjunction with
    the Chinese Remainder Theorem.  The residue for each prime power
    is calculated in time O(log^2(n) + q^4*log(n)log(p) + q^4*p*log^3(p)).

    Parameters
    ==========

    n : an integer
    m : an integer
    k : a positive integer

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import binomial_mod
    >>> binomial_mod(10, 2, 6)  # binomial(10, 2) = 45
    3
    >>> binomial_mod(17, 9, 10)  # binomial(17, 9) = 24310
    0

    References
    ==========

    .. [1] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """
    if k < 1: raise ValueError('k is required to be positive')
    # We decompose q into a product of prime powers and apply
    # the generalization of Lucas' Theorem given by Granville
    # to obtain binomial(n, k) mod p^e, and then use the Chinese
    # Remainder Theorem to obtain the result mod q
    if n < 0 or m < 0 or m > n: return 0
    factorisation = factorint(k)
    residues = [_binomial_mod_prime_power(n, m, p, e) for p, e in factorisation.items()]
    return crt([p**pw for p, pw in factorisation.items()], residues, check=False)[0]


def _binomial_mod_prime_power(n, m, p, q):
    """Compute ``binomial(n, m) % p**q`` for a prime ``p``.

    Parameters
    ==========

    n : positive integer
    m : a nonnegative integer
    p : a prime
    q : a positive integer (the prime exponent)

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _binomial_mod_prime_power
    >>> _binomial_mod_prime_power(10, 2, 3, 2)  # binomial(10, 2) = 45
    0
    >>> _binomial_mod_prime_power(17, 9, 2, 4)  # binomial(17, 9) = 24310
    6

    References
    ==========

    .. [1] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """
    # Function/variable naming within this function follows Ref.[1]
    # n!_p will be used to denote the product of integers <= n not divisible by
    # p, with binomial(n, m)_p the same as binomial(n, m), but defined using
    # n!_p in place of n!
    modulo = pow(p, q)

    def up_factorial(u):
        """Compute (u*p)!_p modulo p^q."""
        r = q // 2
        fac = prod = 1
        if r == 1 and p == 2 or 2*r + 1 in (p, p*p):
            if q % 2 == 1: r += 1
            modulo, div = pow(p, 2*r), pow(p, 2*r - q)
        else:
            modulo, div = pow(p, 2*r + 1), pow(p, (2*r + 1) - q)
        for j in range(1, r + 1):
            for mul in range((j - 1)*p + 1, j*p):  # ignore jp itself
                fac *= mul
                fac %= modulo
            bj_ = bj(u, j, r)
            prod *= pow(fac, bj_, modulo)
            prod %= modulo
        if p == 2:
            sm = u // 2
            for j in range(1, r + 1): sm += j//2 * bj(u, j, r)
            if sm % 2 == 1: prod *= -1
        prod %= modulo//div
        return prod % modulo

    def bj(u, j, r):
        """Compute the exponent of (j*p)!_p in the calculation of (u*p)!_p."""
        prod = u
        for i in range(1, r + 1):
            if i != j: prod *= u*u - i*i
        for i in range(1, r + 1):
            if i != j: prod //= j*j - i*i
        return prod // j

    def up_plus_v_binom(u, v):
        """Compute binomial(u*p + v, v)_p modulo p^q."""
        prod = 1
        div = invert(factorial(v), modulo)
        for j in range(1, q):
            b = div
            for v_ in range(j*p + 1, j*p + v + 1):
                b *= v_
                b %= modulo
            aj = u
            for i in range(1, q):
                if i != j: aj *= u - i
            for i in range(1, q):
                if i != j: aj //= j - i
            aj //= j
            prod *= pow(b, aj, modulo)
            prod %= modulo
        return prod

    @recurrence_memo([1])
    def factorial(v, prev):
        """Compute v! modulo p^q."""
        return v*prev[-1] % modulo

    def factorial_p(n):
        """Compute n!_p modulo p^q."""
        u, v = divmod(n, p)
        return (factorial(v) * up_factorial(u) * up_plus_v_binom(u, v)) % modulo

    prod = 1
    Nj, Mj, Rj = n, m, n - m
    # e0 will be the p-adic valuation of binomial(n, m) at p
    e0 = carry = eq_1 = j = 0
    while Nj:
        numerator = factorial_p(Nj % modulo)
        denominator = factorial_p(Mj % modulo) * factorial_p(Rj % modulo) % modulo
        Nj, (Mj, mj), (Rj, rj) = Nj//p, divmod(Mj, p), divmod(Rj, p)
        carry = (mj + rj + carry) // p
        e0 += carry
        if j >= q - 1: eq_1 += carry
        prod *= numerator * invert(denominator, modulo)
        prod %= modulo
        j += 1

    mul = pow(1 if p == 2 and q >= 3 else -1, eq_1, modulo)
    return (pow(p, e0, modulo) * mul * prod) % modulo