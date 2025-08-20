
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\wigner.py ===
# -*- coding: utf-8 -*-
r"""
Wigner, Clebsch-Gordan, Racah, and Gaunt coefficients

Collection of functions for calculating Wigner 3j, 6j, 9j,
Clebsch-Gordan, Racah as well as Gaunt coefficients exactly, all
evaluating to a rational number times the square root of a rational
number [Rasch03]_.

Please see the description of the individual functions for further
details and examples.

References
==========

.. [Regge58] 'Symmetry Properties of Clebsch-Gordan Coefficients',
  T. Regge, Nuovo Cimento, Volume 10, pp. 544 (1958)
.. [Regge59] 'Symmetry Properties of Racah Coefficients',
  T. Regge, Nuovo Cimento, Volume 11, pp. 116 (1959)
.. [Edmonds74] A. R. Edmonds. Angular momentum in quantum mechanics.
  Investigations in physics, 4.; Investigations in physics, no. 4.
  Princeton, N.J., Princeton University Press, 1957.
.. [Rasch03] J. Rasch and A. C. H. Yu, 'Efficient Storage Scheme for
  Pre-calculated Wigner 3j, 6j and Gaunt Coefficients', SIAM
  J. Sci. Comput. Volume 25, Issue 4, pp. 1416-1428 (2003)
.. [Liberatodebrito82] 'FORTRAN program for the integral of three
  spherical harmonics', A. Liberato de Brito,
  Comput. Phys. Commun., Volume 25, pp. 81-85 (1982)
.. [Homeier96] 'Some Properties of the Coupling Coefficients of Real
  Spherical Harmonics and Their Relation to Gaunt Coefficients',
  H. H. H. Homeier and E. O. Steinborn J. Mol. Struct., Volume 368,
  pp. 31-37 (1996)

Credits and Copyright
=====================

This code was taken from Sage with the permission of all authors:

https://groups.google.com/forum/#!topic/sage-devel/M4NZdu-7O38

Authors
=======

- Jens Rasch (2009-03-24): initial version for Sage

- Jens Rasch (2009-05-31): updated to sage-4.0

- Oscar Gerardo Lazo Arjona (2017-06-18): added Wigner D matrices

- Phil Adam LeMaitre (2022-09-19): added real Gaunt coefficient

Copyright (C) 2008 Jens Rasch <jyr2000@gmail.com>

"""
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.numbers import int_valued
from sympy.core.function import Function
from sympy.core.numbers import (Float, I, Integer, pi, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.utilities.misc import as_int

# This list of precomputed factorials is needed to massively
# accelerate future calculations of the various coefficients
_Factlist = [1]


def _calc_factlist(nn):
    r"""
    Function calculates a list of precomputed factorials in order to
    massively accelerate future calculations of the various
    coefficients.

    Parameters
    ==========

    nn : integer
        Highest factorial to be computed.

    Returns
    =======

    list of integers :
        The list of precomputed factorials.

    Examples
    ========

    Calculate list of factorials::

        sage: from sage.functions.wigner import _calc_factlist
        sage: _calc_factlist(10)
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    """
    if nn >= len(_Factlist):
        for ii in range(len(_Factlist), int(nn + 1)):
            _Factlist.append(_Factlist[ii - 1] * ii)
    return _Factlist[:int(nn) + 1]


def _int_or_halfint(value):
    """return Python int unless value is half-int (then return float)"""
    if isinstance(value, int):
        return value
    elif type(value) is float:
        if value.is_integer():
            return int(value)  # an int
        if (2*value).is_integer():
            return value  # a float
    elif isinstance(value, Rational):
        if value.q == 2:
            return value.p/value.q  # a float
        elif value.q == 1:
            return value.p  # an int
    elif isinstance(value, Float):
        return _int_or_halfint(float(value))
    raise ValueError("expecting integer or half-integer, got %s" % value)


def wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3):
    r"""
    Calculate the Wigner 3j symbol `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.

    Parameters
    ==========

    j_1, j_2, j_3, m_1, m_2, m_3 :
        Integer or half integer.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_3j
    >>> wigner_3j(2, 6, 4, 0, 0, 0)
    sqrt(715)/143
    >>> wigner_3j(2, 6, 4, 0, 0, 1)
    0

    It is an error to have arguments that are not integer or half
    integer values::

        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer
        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer or half integer

    Notes
    =====

    The Wigner 3j symbol obeys the following symmetry rules:

    - invariant under any permutation of the columns (with the
      exception of a sign change where `J:=j_1+j_2+j_3`):

      .. math::

         \begin{aligned}
         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
          &=\operatorname{Wigner3j}(j_3,j_1,j_2,m_3,m_1,m_2) \\
          &=\operatorname{Wigner3j}(j_2,j_3,j_1,m_2,m_3,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_3,j_2,j_1,m_3,m_2,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_1,j_3,j_2,m_1,m_3,m_2) \\
          &=(-1)^J \operatorname{Wigner3j}(j_2,j_1,j_3,m_2,m_1,m_3)
         \end{aligned}

    - invariant under space inflection, i.e.

      .. math::

         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
         =(-1)^J \operatorname{Wigner3j}(j_1,j_2,j_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 additional symmetries based on
      the work by [Regge58]_

    - zero for `j_1`, `j_2`, `j_3` not fulfilling triangle relation

    - zero for `m_1 + m_2 + m_3 \neq 0`

    - zero for violating any one of the conditions
         `m_1  \in \{-|j_1|, \ldots, |j_1|\}`,
         `m_2  \in \{-|j_2|, \ldots, |j_2|\}`,
         `m_3  \in \{-|j_3|, \ldots, |j_3|\}`

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """

    j_1, j_2, j_3, m_1, m_2, m_3 = \
        map(_int_or_halfint, map(sympify,
            [j_1, j_2, j_3, m_1, m_2, m_3]))

    if m_1 + m_2 + m_3 != 0:
        return S.Zero
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return S.Zero
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return S.Zero
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return S.Zero
    if (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3):
        return S.Zero
    if not (int_valued(j_1 - m_1) and \
            int_valued(j_2 - m_2) and \
            int_valued(j_3 - m_3)):
        return S.Zero

    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2),
                  j_3 + abs(m_3))
    _calc_factlist(int(maxfact))

    argsqrt = Integer(_Factlist[int(j_1 + j_2 - j_3)] *
                     _Factlist[int(j_1 - j_2 + j_3)] *
                     _Factlist[int(-j_1 + j_2 + j_3)] *
                     _Factlist[int(j_1 - m_1)] *
                     _Factlist[int(j_1 + m_1)] *
                     _Factlist[int(j_2 - m_2)] *
                     _Factlist[int(j_2 + m_2)] *
                     _Factlist[int(j_3 - m_3)] *
                     _Factlist[int(j_3 + m_3)]) / \
        _Factlist[int(j_1 + j_2 + j_3 + 1)]

    ressqrt = sqrt(argsqrt)
    if ressqrt.is_complex or ressqrt.is_infinite:
        ressqrt = ressqrt.as_real_imag()[0]

    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * \
            _Factlist[int(ii + j_3 - j_1 - m_2)] * \
            _Factlist[int(j_2 + m_2 - ii)] * \
            _Factlist[int(j_1 - ii - m_1)] * \
            _Factlist[int(ii + j_3 - j_2 + m_1)] * \
            _Factlist[int(j_1 + j_2 - j_3 - ii)]
        sumres = sumres + Integer((-1) ** ii) / den

    prefid = Integer((-1) ** int(j_1 - j_2 - m_3))
    res = ressqrt * sumres * prefid
    return res


def clebsch_gordan(j_1, j_2, j_3, m_1, m_2, m_3):
    r"""
    Calculates the Clebsch-Gordan coefficient.
    `\left\langle j_1 m_1 \; j_2 m_2 | j_3 m_3 \right\rangle`.

    The reference for this function is [Edmonds74]_.

    Parameters
    ==========

    j_1, j_2, j_3, m_1, m_2, m_3 :
        Integer or half integer.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.physics.wigner import clebsch_gordan
    >>> clebsch_gordan(S(3)/2, S(1)/2, 2, S(3)/2, S(1)/2, 2)
    1
    >>> clebsch_gordan(S(3)/2, S(1)/2, 1, S(3)/2, -S(1)/2, 1)
    sqrt(3)/2
    >>> clebsch_gordan(S(3)/2, S(1)/2, 1, -S(1)/2, S(1)/2, 0)
    -sqrt(2)/2

    Notes
    =====

    The Clebsch-Gordan coefficient will be evaluated via its relation
    to Wigner 3j symbols:

    .. math::

        \left\langle j_1 m_1 \; j_2 m_2 | j_3 m_3 \right\rangle
        =(-1)^{j_1-j_2+m_3} \sqrt{2j_3+1}
        \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,-m_3)

    See also the documentation on Wigner 3j symbols which exhibit much
    higher symmetry relations than the Clebsch-Gordan coefficient.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """
    j_1 = sympify(j_1)
    j_2 = sympify(j_2)
    j_3 = sympify(j_3)
    m_1 = sympify(m_1)
    m_2 = sympify(m_2)
    m_3 = sympify(m_3)

    w = wigner_3j(j_1, j_2, j_3, m_1, m_2, -m_3)

    return (-1) ** (j_1 - j_2 + m_3) * sqrt(2 * j_3 + 1) * w


def _big_delta_coeff(aa, bb, cc, prec=None):
    r"""
    Calculates the Delta coefficient of the 3 angular momenta for
    Racah symbols. Also checks that the differences are of integer
    value.

    Parameters
    ==========

    aa :
        First angular momentum, integer or half integer.
    bb :
        Second angular momentum, integer or half integer.
    cc :
        Third angular momentum, integer or half integer.
    prec :
        Precision of the ``sqrt()`` calculation.

    Returns
    =======

    double : Value of the Delta coefficient.

    Examples
    ========

        sage: from sage.functions.wigner import _big_delta_coeff
        sage: _big_delta_coeff(1,1,1)
        1/2*sqrt(1/6)
    """

    # the triangle test will only pass if a) all 3 values are ints or
    # b) 1 is an int and the other two are half-ints
    if not int_valued(aa + bb - cc):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if not int_valued(aa + cc - bb):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if not int_valued(bb + cc - aa):
        raise ValueError("j values must be integer or half integer and fulfill the triangle relation")
    if (aa + bb - cc) < 0:
        return S.Zero
    if (aa + cc - bb) < 0:
        return S.Zero
    if (bb + cc - aa) < 0:
        return S.Zero

    maxfact = max(aa + bb - cc, aa + cc - bb, bb + cc - aa, aa + bb + cc + 1)
    _calc_factlist(maxfact)

    argsqrt = Integer(_Factlist[int(aa + bb - cc)] *
                     _Factlist[int(aa + cc - bb)] *
                     _Factlist[int(bb + cc - aa)]) / \
        Integer(_Factlist[int(aa + bb + cc + 1)])

    ressqrt = sqrt(argsqrt)
    if prec:
        ressqrt = ressqrt.evalf(prec).as_real_imag()[0]
    return ressqrt


def racah(aa, bb, cc, dd, ee, ff, prec=None):
    r"""
    Calculate the Racah symbol `W(a,b,c,d;e,f)`.

    Parameters
    ==========

    a, ..., f :
        Integer or half integer.
    prec :
        Precision, default: ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import racah
    >>> racah(3,3,3,3,3,3)
    -1/14

    Notes
    =====

    The Racah symbol is related to the Wigner 6j symbol:

    .. math::

       \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
       =(-1)^{j_1+j_2+j_4+j_5} W(j_1,j_2,j_5,j_4,j_3,j_6)

    Please see the 6j symbol for its much richer symmetries and for
    additional properties.

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 6j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    Authors
    =======

    - Jens Rasch (2009-03-24): initial version
    """
    prefac = _big_delta_coeff(aa, bb, ee, prec) * \
        _big_delta_coeff(cc, dd, ee, prec) * \
        _big_delta_coeff(aa, cc, ff, prec) * \
        _big_delta_coeff(bb, dd, ff, prec)
    if prefac == 0:
        return S.Zero
    imin = max(aa + bb + ee, cc + dd + ee, aa + cc + ff, bb + dd + ff)
    imax = min(aa + bb + cc + dd, aa + dd + ee + ff, bb + cc + ee + ff)

    maxfact = max(imax + 1, aa + bb + cc + dd, aa + dd + ee + ff,
                 bb + cc + ee + ff)
    _calc_factlist(maxfact)

    sumres = 0
    for kk in range(int(imin), int(imax) + 1):
        den = _Factlist[int(kk - aa - bb - ee)] * \
            _Factlist[int(kk - cc - dd - ee)] * \
            _Factlist[int(kk - aa - cc - ff)] * \
            _Factlist[int(kk - bb - dd - ff)] * \
            _Factlist[int(aa + bb + cc + dd - kk)] * \
            _Factlist[int(aa + dd + ee + ff - kk)] * \
            _Factlist[int(bb + cc + ee + ff - kk)]
        sumres = sumres + Integer((-1) ** kk * _Factlist[kk + 1]) / den

    res = prefac * sumres * (-1) ** int(aa + bb + cc + dd)
    return res


def wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6, prec=None):
    r"""
    Calculate the Wigner 6j symbol `\operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)`.

    Parameters
    ==========

    j_1, ..., j_6 :
        Integer or half integer.
    prec :
        Precision, default: ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_6j
    >>> wigner_6j(3,3,3,3,3,3)
    -1/14
    >>> wigner_6j(5,5,5,5,5,5)
    1/52

    It is an error to have arguments that are not integer or half
    integer values or do not fulfill the triangle relation::

        sage: wigner_6j(2.5,2.5,2.5,2.5,2.5,2.5)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation
        sage: wigner_6j(0.5,0.5,1.1,0.5,0.5,1.1)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation

    Notes
    =====

    The Wigner 6j symbol is related to the Racah symbol but exhibits
    more symmetries as detailed below.

    .. math::

       \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
        =(-1)^{j_1+j_2+j_4+j_5} W(j_1,j_2,j_5,j_4,j_3,j_6)

    The Wigner 6j symbol obeys the following symmetry rules:

    - Wigner 6j symbols are left invariant under any permutation of
      the columns:

      .. math::

         \begin{aligned}
         \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
          &=\operatorname{Wigner6j}(j_3,j_1,j_2,j_6,j_4,j_5) \\
          &=\operatorname{Wigner6j}(j_2,j_3,j_1,j_5,j_6,j_4) \\
          &=\operatorname{Wigner6j}(j_3,j_2,j_1,j_6,j_5,j_4) \\
          &=\operatorname{Wigner6j}(j_1,j_3,j_2,j_4,j_6,j_5) \\
          &=\operatorname{Wigner6j}(j_2,j_1,j_3,j_5,j_4,j_6)
         \end{aligned}

    - They are invariant under the exchange of the upper and lower
      arguments in each of any two columns, i.e.

      .. math::

         \begin{aligned}
         \operatorname{Wigner6j}(j_1,j_2,j_3,j_4,j_5,j_6)
          &=\operatorname{Wigner6j}(j_1,j_5,j_6,j_4,j_2,j_3)\\
          &=\operatorname{Wigner6j}(j_4,j_2,j_6,j_1,j_5,j_3)\\
          &=\operatorname{Wigner6j}(j_4,j_5,j_3,j_1,j_2,j_6)
         \end{aligned}

    - additional 6 symmetries [Regge59]_ giving rise to 144 symmetries
      in total

    - only non-zero if any triple of `j`'s fulfill a triangle relation

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 6j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    """
    j_1, j_2, j_3, j_4, j_5, j_6 = map(sympify, \
                [j_1, j_2, j_3, j_4, j_5, j_6])
    res = (-1) ** int(j_1 + j_2 + j_4 + j_5) * \
        racah(j_1, j_2, j_5, j_4, j_3, j_6, prec)
    return res


def wigner_9j(j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9, prec=None):
    r"""
    Calculate the Wigner 9j symbol
    `\operatorname{Wigner9j}(j_1,j_2,j_3,j_4,j_5,j_6,j_7,j_8,j_9)`.

    Parameters
    ==========

    j_1, ..., j_9 :
        Integer or half integer.
    prec : precision, default
        ``None``. Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_9j
    >>> wigner_9j(1,1,1, 1,1,1, 1,1,0, prec=64)
    0.05555555555555555555555555555555555555555555555555555555555555555

    >>> wigner_9j(1/2,1/2,0, 1/2,3/2,1, 0,1,1, prec=64)
    0.1666666666666666666666666666666666666666666666666666666666666667

    It is an error to have arguments that are not integer or half
    integer values or do not fulfill the triangle relation::

        sage: wigner_9j(0.5,0.5,0.5, 0.5,0.5,0.5, 0.5,0.5,0.5,prec=64)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation
        sage: wigner_9j(1,1,1, 0.5,1,1.5, 0.5,1,2.5,prec=64)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer and fulfill the triangle relation

    Algorithm
    =========

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.
    """
    j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9 = map(sympify, \
                [j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9])
    imax = int(min(j_1 + j_9, j_2 + j_6, j_4 + j_8) * 2)
    imin = imax % 2
    sumres = 0
    for kk in range(imin, int(imax) + 1, 2):
        sumres = sumres + (kk + 1) * \
            racah(j_1, j_2, j_9, j_6, j_3, kk / 2, prec) * \
            racah(j_4, j_6, j_8, j_2, j_5, kk / 2, prec) * \
            racah(j_1, j_4, j_9, j_8, j_7, kk / 2, prec)
    return sumres


def gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    r"""
    Calculate the Gaunt coefficient.

    Explanation
    ===========

    The Gaunt coefficient is defined as the integral over three
    spherical harmonics:

    .. math::

        \begin{aligned}
        \operatorname{Gaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\int Y_{l_1,m_1}(\Omega)
         Y_{l_2,m_2}(\Omega) Y_{l_3,m_3}(\Omega) \,d\Omega \\
        &=\sqrt{\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\pi}}
         \operatorname{Wigner3j}(l_1,l_2,l_3,0,0,0)
         \operatorname{Wigner3j}(l_1,l_2,l_3,m_1,m_2,m_3)
        \end{aligned}

    Parameters
    ==========

    l_1, l_2, l_3, m_1, m_2, m_3 :
        Integer.
    prec - precision, default: ``None``.
        Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import gaunt
    >>> gaunt(1,0,1,1,0,-1)
    -1/(2*sqrt(pi))
    >>> gaunt(1000,1000,1200,9,3,-12).n(64)
    0.006895004219221134484332976156744208248842039317638217822322799675

    It is an error to use non-integer values for `l` and `m`::

        sage: gaunt(1.2,0,1.2,0,0,0)
        Traceback (most recent call last):
        ...
        ValueError: l values must be integer
        sage: gaunt(1,0,1,1.1,0,-1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer

    Notes
    =====

    The Gaunt coefficient obeys the following symmetry rules:

    - invariant under any permutation of the columns

      .. math::
        \begin{aligned}
          Y(l_1,l_2,l_3,m_1,m_2,m_3)
          &=Y(l_3,l_1,l_2,m_3,m_1,m_2) \\
          &=Y(l_2,l_3,l_1,m_2,m_3,m_1) \\
          &=Y(l_3,l_2,l_1,m_3,m_2,m_1) \\
          &=Y(l_1,l_3,l_2,m_1,m_3,m_2) \\
          &=Y(l_2,l_1,l_3,m_2,m_1,m_3)
        \end{aligned}

    - invariant under space inflection, i.e.

      .. math::
          Y(l_1,l_2,l_3,m_1,m_2,m_3)
          =Y(l_1,l_2,l_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 Regge symmetries as inherited
      for the `3j` symbols [Regge58]_

    - zero for `l_1`, `l_2`, `l_3` not fulfilling triangle relation

    - zero for violating any one of the conditions: `l_1 \ge |m_1|`,
      `l_2 \ge |m_2|`, `l_3 \ge |m_3|`

    - non-zero only for an even sum of the `l_i`, i.e.
      `L = l_1 + l_2 + l_3 = 2n` for `n` in `\mathbb{N}`

    Algorithms
    ==========

    This function uses the algorithm of [Liberatodebrito82]_ to
    calculate the value of the Gaunt coefficient exactly. Note that
    the formula contains alternating sums over large factorials and is
    therefore unsuitable for finite precision arithmetic and only
    useful for a computer algebra system [Rasch03]_.

    Authors
    =======

    Jens Rasch (2009-03-24): initial version for Sage.
    """
    l_1, l_2, l_3, m_1, m_2, m_3 = [
        as_int(i) for i in (l_1, l_2, l_3, m_1, m_2, m_3)]

    if l_1 + l_2 - l_3 < 0:
        return S.Zero
    if l_1 - l_2 + l_3 < 0:
        return S.Zero
    if -l_1 + l_2 + l_3 < 0:
        return S.Zero
    if (m_1 + m_2 + m_3) != 0:
        return S.Zero
    if (abs(m_1) > l_1) or (abs(m_2) > l_2) or (abs(m_3) > l_3):
        return S.Zero
    bigL, remL = divmod(l_1 + l_2 + l_3, 2)
    if remL % 2:
        return S.Zero

    imin = max(-l_3 + l_1 + m_2, -l_3 + l_2 - m_1, 0)
    imax = min(l_2 + m_2, l_1 - m_1, l_1 + l_2 - l_3)

    _calc_factlist(max(l_1 + l_2 + l_3 + 1, imax + 1))

    ressqrt = sqrt((2 * l_1 + 1) * (2 * l_2 + 1) * (2 * l_3 + 1) * \
        _Factlist[l_1 - m_1] * _Factlist[l_1 + m_1] * _Factlist[l_2 - m_2] * \
        _Factlist[l_2 + m_2] * _Factlist[l_3 - m_3] * _Factlist[l_3 + m_3] / \
        (4*pi))

    prefac = Integer(_Factlist[bigL] * _Factlist[l_2 - l_1 + l_3] *
                     _Factlist[l_1 - l_2 + l_3] * _Factlist[l_1 + l_2 - l_3])/ \
        _Factlist[2 * bigL + 1]/ \
        (_Factlist[bigL - l_1] *
         _Factlist[bigL - l_2] * _Factlist[bigL - l_3])

    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * _Factlist[ii + l_3 - l_1 - m_2] * \
            _Factlist[l_2 + m_2 - ii] * _Factlist[l_1 - ii - m_1] * \
            _Factlist[ii + l_3 - l_2 + m_1] * _Factlist[l_1 + l_2 - l_3 - ii]
        sumres = sumres + Integer((-1) ** ii) / den

    res = ressqrt * prefac * sumres * Integer((-1) ** (bigL + l_3 + m_1 - m_2))
    if prec is not None:
        res = res.n(prec)
    return res


def real_gaunt(l_1, l_2, l_3, mu_1, mu_2, mu_3, prec=None):
    r"""
    Calculate the real Gaunt coefficient.

    Explanation
    ===========

    The real Gaunt coefficient is defined as the integral over three
    real spherical harmonics:

    .. math::
        \begin{aligned}
        \operatorname{RealGaunt}(l_1,l_2,l_3,\mu_1,\mu_2,\mu_3)
        &=\int Z^{\mu_1}_{l_1}(\Omega)
         Z^{\mu_2}_{l_2}(\Omega) Z^{\mu_3}_{l_3}(\Omega) \,d\Omega \\
        \end{aligned}

    Alternatively, it can be defined in terms of the standard Gaunt
    coefficient by relating the real spherical harmonics to the standard
    spherical harmonics via a unitary transformation `U`, i.e.
    `Z^{\mu}_{l}(\Omega)=\sum_{m'}U^{\mu}_{m'}Y^{m'}_{l}(\Omega)` [Homeier96]_.
    The real Gaunt coefficient is then defined as

    .. math::
        \begin{aligned}
        \operatorname{RealGaunt}(l_1,l_2,l_3,\mu_1,\mu_2,\mu_3)
        &=\int Z^{\mu_1}_{l_1}(\Omega)
         Z^{\mu_2}_{l_2}(\Omega) Z^{\mu_3}_{l_3}(\Omega) \,d\Omega \\
        &=\sum_{m'_1 m'_2 m'_3} U^{\mu_1}_{m'_1}U^{\mu_2}_{m'_2}U^{\mu_3}_{m'_3}
         \operatorname{Gaunt}(l_1,l_2,l_3,m'_1,m'_2,m'_3)
        \end{aligned}

    The unitary matrix `U` has components

    .. math::
        \begin{aligned}
        U^\mu_{m} = \delta_{|\mu||m|}*(\delta_{m0}\delta_{\mu 0} + \frac{1}{\sqrt{2}}\big[\Theta(\mu)\big(\delta_{m\mu}+(-1)^{m}\delta_{m-\mu}\big)
        +i \Theta(-\mu)\big((-1)^{m}\delta_{m\mu}-\delta_{m-\mu}\big)\big])
        \end{aligned}


    where `\delta_{ij}` is the Kronecker delta symbol and `\Theta` is a step
    function defined as

    .. math::
        \begin{aligned}
        \Theta(x) = \begin{cases} 1 \,\text{for}\, x > 0 \\ 0 \,\text{for}\, x \leq 0 \end{cases}
        \end{aligned}

    Parameters
    ==========

    l_1, l_2, l_3, mu_1, mu_2, mu_3 :
        Integer degree and order

    prec - precision, default: ``None``.
        Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number.

    Examples
    ========
    >>> from sympy.physics.wigner import real_gaunt
    >>> real_gaunt(1,1,2,-1,1,-2)
    sqrt(15)/(10*sqrt(pi))
    >>> real_gaunt(10,10,20,-9,-9,0,prec=64)
    -0.00002480019791932209313156167176797577821140084216297395518482071448

    It is an error to use non-integer values for `l` and `\mu`::
        real_gaunt(2.8,0.5,1.3,0,0,0)
        Traceback (most recent call last):
        ...
        ValueError: l values must be integer

        real_gaunt(2,2,4,0.7,1,-3.4)
        Traceback (most recent call last):
        ...
        ValueError: mu values must be integer

    Notes
    =====

    The real Gaunt coefficient inherits from the standard Gaunt coefficient,
    the invariance under any permutation of the pairs `(l_i, \mu_i)` and the
    requirement that the sum of the `l_i` be even to yield a non-zero value.
    It also obeys the following symmetry rules:

    - zero for `l_1`, `l_2`, `l_3` not fulfilling the condition
      `l_1 \in \{l_{\text{max}}, l_{\text{max}}-2, \ldots, l_{\text{min}}\}`,
      where `l_{\text{max}} = l_2+l_3`,

      .. math::
          \begin{aligned}
          l_{\text{min}} = \begin{cases} \kappa(l_2, l_3, \mu_2, \mu_3) & \text{if}\,
          \kappa(l_2, l_3, \mu_2, \mu_3) + l_{\text{max}}\, \text{is even} \\
          \kappa(l_2, l_3, \mu_2, \mu_3)+1 & \text{if}\, \kappa(l_2, l_3, \mu_2, \mu_3) +
          l_{\text{max}}\, \text{is odd}\end{cases}
          \end{aligned}

      and `\kappa(l_2, l_3, \mu_2, \mu_3) = \max{\big(|l_2-l_3|, \min{\big(|\mu_2+\mu_3|,
      |\mu_2-\mu_3|\big)}\big)}`

    - zero for an odd number of negative `\mu_i`

    Algorithms
    ==========

    This function uses the algorithms of [Homeier96]_ and [Rasch03]_ to
    calculate the value of the real Gaunt coefficient exactly. Note that
    the formula used in [Rasch03]_ contains alternating sums over large
    factorials and is therefore unsuitable for finite precision arithmetic
    and only useful for a computer algebra system [Rasch03]_. However, this
    function can in principle use any algorithm that computes the Gaunt
    coefficient, so it is suitable for finite precision arithmetic in so far
    as the algorithm which computes the Gaunt coefficient is.
    """
    l_1, l_2, l_3, mu_1, mu_2, mu_3 = [
        as_int(i) for i in (l_1, l_2, l_3, mu_1, mu_2, mu_3)]

    # check for quick exits
    if sum(1 for i in (mu_1, mu_2, mu_3) if i < 0) % 2:
        return S.Zero  # odd number of negative m
    if (l_1 + l_2 + l_3) % 2:
        return S.Zero  # sum of l is odd
    lmax = l_2 + l_3
    lmin = max(abs(l_2 - l_3), min(abs(mu_2 + mu_3), abs(mu_2 - mu_3)))
    if (lmin + lmax) % 2:
        lmin += 1
    if lmin not in range(lmax, lmin - 2, -2):
        return S.Zero

    kron_del = lambda i, j: 1 if i == j else 0
    s = lambda e: -1 if e % 2 else 1  #  (-1)**e to give +/-1, avoiding float when e<0

    t = lambda x: 1 if x > 0 else 0
    A = lambda mu, m: t(-mu) * (s(m) * kron_del(m, mu) - kron_del(m, -mu))
    B = lambda mu, m: t(mu) * (kron_del(m, mu) + s(m) * kron_del(m, -mu))
    U = lambda mu, m: kron_del(abs(mu), abs(m)) * (kron_del(mu, 0) * kron_del(m, 0) + (B(mu, m) + I * A(mu, m))/sqrt(2))

    ugnt = 0
    for m1 in range(-l_1, l_1+1):
        U1 = U(mu_1, m1)
        for m2 in range(-l_2, l_2+1):
            U2 = U(mu_2, m2)
            U3 = U(mu_3,-m1-m2)
            ugnt = ugnt + re(U1*U2*U3)*gaunt(l_1, l_2, l_3, m1, m2, -m1 - m2, prec=prec)

    return ugnt


class Wigner3j(Function):

    def doit(self, **hints):
        if all(obj.is_number for obj in self.args):
            return wigner_3j(*self.args)
        else:
            return self

def dot_rot_grad_Ynm(j, p, l, m, theta, phi):
    r"""
    Returns dot product of rotational gradients of spherical harmonics.

    Explanation
    ===========

    This function returns the right hand side of the following expression:

    .. math ::
        \vec{R}Y{_j^{p}} \cdot \vec{R}Y{_l^{m}} = (-1)^{m+p}
        \sum\limits_{k=|l-j|}^{l+j}Y{_k^{m+p}}  * \alpha_{l,m,j,p,k} *
        \frac{1}{2} (k^2-j^2-l^2+k-j-l)


    Arguments
    =========

    j, p, l, m .... indices in spherical harmonics (expressions or integers)
    theta, phi .... angle arguments in spherical harmonics

    Example
    =======

    >>> from sympy import symbols
    >>> from sympy.physics.wigner import dot_rot_grad_Ynm
    >>> theta, phi = symbols("theta phi")
    >>> dot_rot_grad_Ynm(3, 2, 2, 0, theta, phi).doit()
    3*sqrt(55)*Ynm(5, 2, theta, phi)/(11*sqrt(pi))

    """
    j = sympify(j)
    p = sympify(p)
    l = sympify(l)
    m = sympify(m)
    theta = sympify(theta)
    phi = sympify(phi)
    k = Dummy("k")

    def alpha(l,m,j,p,k):
        return sqrt((2*l+1)*(2*j+1)*(2*k+1)/(4*pi)) * \
                Wigner3j(j, l, k, S.Zero, S.Zero, S.Zero) * \
                Wigner3j(j, l, k, p, m, -m-p)

    return (S.NegativeOne)**(m+p) * Sum(Ynm(k, m+p, theta, phi) * alpha(l,m,j,p,k) / 2 \
        *(k**2-j**2-l**2+k-j-l), (k, abs(l-j), l+j))


def wigner_d_small(J, beta):
    """Return the small Wigner d matrix for angular momentum J.

    Explanation
    ===========

    J : An integer, half-integer, or SymPy symbol for the total angular
        momentum of the angular momentum space being rotated.
    beta : A real number representing the Euler angle of rotation about
        the so-called line of nodes. See [Edmonds74]_.

    Returns
    =======

    A matrix representing the corresponding Euler angle rotation( in the basis
    of eigenvectors of `J_z`).

    .. math ::
        \\mathcal{d}_{\\beta} = \\exp\\big( \\frac{i\\beta}{\\hbar} J_y\\big)

    such that

    .. math ::
        d^{(J)}_{m',m}(\\beta) = \\mathtt{wigner\\_d\\_small(J,beta)[J-mprime,J-m]}

    The components are calculated using the general form [Edmonds74]_,
    equation 4.1.15.

    Examples
    ========

    >>> from sympy import Integer, symbols, pi, pprint
    >>> from sympy.physics.wigner import wigner_d_small
    >>> half = 1/Integer(2)
    >>> beta = symbols("beta", real=True)
    >>> pprint(wigner_d_small(half, beta), use_unicode=True)
    ⎡   ⎛β⎞      ⎛β⎞⎤
    ⎢cos⎜─⎟   sin⎜─⎟⎥
    ⎢   ⎝2⎠      ⎝2⎠⎥
    ⎢               ⎥
    ⎢    ⎛β⎞     ⎛β⎞⎥
    ⎢-sin⎜─⎟  cos⎜─⎟⎥
    ⎣    ⎝2⎠     ⎝2⎠⎦

    >>> pprint(wigner_d_small(2*half, beta), use_unicode=True)
    ⎡        2⎛β⎞              ⎛β⎞    ⎛β⎞           2⎛β⎞     ⎤
    ⎢     cos ⎜─⎟        √2⋅sin⎜─⎟⋅cos⎜─⎟        sin ⎜─⎟     ⎥
    ⎢         ⎝2⎠              ⎝2⎠    ⎝2⎠            ⎝2⎠     ⎥
    ⎢                                                        ⎥
    ⎢       ⎛β⎞    ⎛β⎞       2⎛β⎞      2⎛β⎞        ⎛β⎞    ⎛β⎞⎥
    ⎢-√2⋅sin⎜─⎟⋅cos⎜─⎟  - sin ⎜─⎟ + cos ⎜─⎟  √2⋅sin⎜─⎟⋅cos⎜─⎟⎥
    ⎢       ⎝2⎠    ⎝2⎠        ⎝2⎠       ⎝2⎠        ⎝2⎠    ⎝2⎠⎥
    ⎢                                                        ⎥
    ⎢        2⎛β⎞               ⎛β⎞    ⎛β⎞          2⎛β⎞     ⎥
    ⎢     sin ⎜─⎟        -√2⋅sin⎜─⎟⋅cos⎜─⎟       cos ⎜─⎟     ⎥
    ⎣         ⎝2⎠               ⎝2⎠    ⎝2⎠           ⎝2⎠     ⎦

    From table 4 in [Edmonds74]_

    >>> pprint(wigner_d_small(half, beta).subs({beta:pi/2}), use_unicode=True)
    ⎡ √2   √2⎤
    ⎢ ──   ──⎥
    ⎢ 2    2 ⎥
    ⎢        ⎥
    ⎢-√2   √2⎥
    ⎢────  ──⎥
    ⎣ 2    2 ⎦

    >>> pprint(wigner_d_small(2*half, beta).subs({beta:pi/2}),
    ... use_unicode=True)
    ⎡       √2      ⎤
    ⎢1/2    ──   1/2⎥
    ⎢       2       ⎥
    ⎢               ⎥
    ⎢-√2         √2 ⎥
    ⎢────   0    ── ⎥
    ⎢ 2          2  ⎥
    ⎢               ⎥
    ⎢      -√2      ⎥
    ⎢1/2   ────  1/2⎥
    ⎣       2       ⎦

    >>> pprint(wigner_d_small(3*half, beta).subs({beta:pi/2}),
    ... use_unicode=True)
    ⎡ √2    √6    √6   √2⎤
    ⎢ ──    ──    ──   ──⎥
    ⎢ 4     4     4    4 ⎥
    ⎢                    ⎥
    ⎢-√6   -√2    √2   √6⎥
    ⎢────  ────   ──   ──⎥
    ⎢ 4     4     4    4 ⎥
    ⎢                    ⎥
    ⎢ √6   -√2   -√2   √6⎥
    ⎢ ──   ────  ────  ──⎥
    ⎢ 4     4     4    4 ⎥
    ⎢                    ⎥
    ⎢-√2    √6   -√6   √2⎥
    ⎢────   ──   ────  ──⎥
    ⎣ 4     4     4    4 ⎦

    >>> pprint(wigner_d_small(4*half, beta).subs({beta:pi/2}),
    ... use_unicode=True)
    ⎡             √6            ⎤
    ⎢1/4   1/2    ──   1/2   1/4⎥
    ⎢             4             ⎥
    ⎢                           ⎥
    ⎢-1/2  -1/2   0    1/2   1/2⎥
    ⎢                           ⎥
    ⎢ √6                     √6 ⎥
    ⎢ ──    0    -1/2   0    ── ⎥
    ⎢ 4                      4  ⎥
    ⎢                           ⎥
    ⎢-1/2  1/2    0    -1/2  1/2⎥
    ⎢                           ⎥
    ⎢             √6            ⎥
    ⎢1/4   -1/2   ──   -1/2  1/4⎥
    ⎣             4             ⎦

    """
    M = [J-i for i in range(2*J+1)]
    d = zeros(2*J+1)

    # Mi corresponds to Edmonds' $m'$, and Mj to $m$.
    for i, Mi in enumerate(M):
        for j, Mj in enumerate(M):

            # We get the maximum and minimum value of sigma.
            sigmamax = min([J-Mi, J-Mj])
            sigmamin = max([0, -Mi-Mj])

            dij = sqrt(factorial(J+Mi)*factorial(J-Mi) /
                       factorial(J+Mj)/factorial(J-Mj))
            terms = [(-1)**(J-Mi-s) *
                     binomial(J+Mj, J-Mi-s) *
                     binomial(J-Mj, s) *
                     cos(beta/2)**(2*s+Mi+Mj) *
                     sin(beta/2)**(2*J-2*s-Mj-Mi)
                     for s in range(sigmamin, sigmamax+1)]

            d[i, j] = dij*Add(*terms)

    return ImmutableMatrix(d)


def wigner_d(J, alpha, beta, gamma):
    """Return the Wigner D matrix for angular momentum J.

    Explanation
    ===========

    J :
        An integer, half-integer, or SymPy symbol for the total angular
        momentum of the angular momentum space being rotated.
    alpha, beta, gamma - Real numbers representing the Euler.
        Angles of rotation about the so-called figure axis, line of nodes,
        and vertical. See [Edmonds74]_, however note that the symbols alpha
        and gamma are swapped in this implementation.

    Returns
    =======

    A matrix representing the corresponding Euler angle rotation (in the basis
    of eigenvectors of `J_z`).

    .. math ::
        \\mathcal{D}_{\\alpha \\beta \\gamma} =
        \\exp\\big( \\frac{i\\alpha}{\\hbar} J_z\\big)
        \\exp\\big( \\frac{i\\beta}{\\hbar} J_y\\big)
        \\exp\\big( \\frac{i\\gamma}{\\hbar} J_z\\big)

    such that

    .. math ::
        \\mathcal{D}^{(J)}_{m',m}(\\alpha, \\beta, \\gamma) =
        \\mathtt{wigner_d(J, alpha, beta, gamma)[J-mprime,J-m]}

    The components are calculated using the general form [Edmonds74]_,
    equation 4.1.12, however note that the angles alpha and gamma are swapped
    in this implementation.

    Examples
    ========

    The simplest possible example:

    >>> from sympy.physics.wigner import wigner_d
    >>> from sympy import Integer, symbols, pprint
    >>> half = 1/Integer(2)
    >>> alpha, beta, gamma = symbols("alpha, beta, gamma", real=True)
    >>> pprint(wigner_d(half, alpha, beta, gamma), use_unicode=True)
    ⎡  ⅈ⋅α  ⅈ⋅γ             ⅈ⋅α  -ⅈ⋅γ         ⎤
    ⎢  ───  ───             ───  ─────        ⎥
    ⎢   2    2     ⎛β⎞       2     2      ⎛β⎞ ⎥
    ⎢ ℯ   ⋅ℯ   ⋅cos⎜─⎟     ℯ   ⋅ℯ     ⋅sin⎜─⎟ ⎥
    ⎢              ⎝2⎠                    ⎝2⎠ ⎥
    ⎢                                         ⎥
    ⎢  -ⅈ⋅α   ⅈ⋅γ          -ⅈ⋅α   -ⅈ⋅γ        ⎥
    ⎢  ─────  ───          ─────  ─────       ⎥
    ⎢    2     2     ⎛β⎞     2      2      ⎛β⎞⎥
    ⎢-ℯ     ⋅ℯ   ⋅sin⎜─⎟  ℯ     ⋅ℯ     ⋅cos⎜─⎟⎥
    ⎣                ⎝2⎠                   ⎝2⎠⎦

    """
    d = wigner_d_small(J, beta)
    M = [J-i for i in range(2*J+1)]
    # Mi corresponds to Edmonds' $m'$, and Mj to $m$.
    D = [[exp(I*Mi*alpha)*d[i, j]*exp(I*Mj*gamma)
          for j, Mj in enumerate(M)] for i, Mi in enumerate(M)]
    return ImmutableMatrix(D)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\groupby\ops.py ===
"""
Provide classes to perform the groupby aggregate operations.

These are not exposed to the user and provide implementations of the grouping
operations, primarily in cython. These classes (BaseGrouper and BinGrouper)
are contained *in* the SeriesGroupBy and DataFrameGroupBy objects.
"""
from __future__ import annotations

import collections
import functools
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    final,
)

import numpy as np

from pandas._libs import (
    NaT,
    lib,
)
import pandas._libs.groupby as libgroupby
from pandas._typing import (
    ArrayLike,
    AxisInt,
    NDFrameT,
    Shape,
    npt,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.cast import (
    maybe_cast_pointwise_result,
    maybe_downcast_to_dtype,
)
from pandas.core.dtypes.common import (
    ensure_float64,
    ensure_int64,
    ensure_platform_int,
    ensure_uint64,
    is_1d_only_ea_dtype,
)
from pandas.core.dtypes.missing import (
    isna,
    maybe_fill,
)

from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
    CategoricalIndex,
    Index,
    MultiIndex,
    ensure_index,
)
from pandas.core.series import Series
from pandas.core.sorting import (
    compress_group_index,
    decons_obs_group_ids,
    get_flattened_list,
    get_group_index,
    get_group_index_sorter,
    get_indexer_dict,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
        Sequence,
    )

    from pandas.core.generic import NDFrame


def check_result_array(obj, dtype) -> None:
    # Our operation is supposed to be an aggregation/reduction. If
    #  it returns an ndarray, this likely means an invalid operation has
    #  been passed. See test_apply_without_aggregation, test_agg_must_agg
    if isinstance(obj, np.ndarray):
        if dtype != object:
            # If it is object dtype, the function can be a reduction/aggregation
            #  and still return an ndarray e.g. test_agg_over_numpy_arrays
            raise ValueError("Must produce aggregated value")


def extract_result(res):
    """
    Extract the result object, it might be a 0-dim ndarray
    or a len-1 0-dim, or a scalar
    """
    if hasattr(res, "_values"):
        # Preserve EA
        res = res._values
        if res.ndim == 1 and len(res) == 1:
            # see test_agg_lambda_with_timezone, test_resampler_grouper.py::test_apply
            res = res[0]
    return res


class WrappedCythonOp:
    """
    Dispatch logic for functions defined in _libs.groupby

    Parameters
    ----------
    kind: str
        Whether the operation is an aggregate or transform.
    how: str
        Operation name, e.g. "mean".
    has_dropped_na: bool
        True precisely when dropna=True and the grouper contains a null value.
    """

    # Functions for which we do _not_ attempt to cast the cython result
    #  back to the original dtype.
    cast_blocklist = frozenset(
        ["any", "all", "rank", "count", "size", "idxmin", "idxmax"]
    )

    def __init__(self, kind: str, how: str, has_dropped_na: bool) -> None:
        self.kind = kind
        self.how = how
        self.has_dropped_na = has_dropped_na

    _CYTHON_FUNCTIONS: dict[str, dict] = {
        "aggregate": {
            "any": functools.partial(libgroupby.group_any_all, val_test="any"),
            "all": functools.partial(libgroupby.group_any_all, val_test="all"),
            "sum": "group_sum",
            "prod": "group_prod",
            "idxmin": functools.partial(libgroupby.group_idxmin_idxmax, name="idxmin"),
            "idxmax": functools.partial(libgroupby.group_idxmin_idxmax, name="idxmax"),
            "min": "group_min",
            "max": "group_max",
            "mean": "group_mean",
            "median": "group_median_float64",
            "var": "group_var",
            "std": functools.partial(libgroupby.group_var, name="std"),
            "sem": functools.partial(libgroupby.group_var, name="sem"),
            "skew": "group_skew",
            "first": "group_nth",
            "last": "group_last",
            "ohlc": "group_ohlc",
        },
        "transform": {
            "cumprod": "group_cumprod",
            "cumsum": "group_cumsum",
            "cummin": "group_cummin",
            "cummax": "group_cummax",
            "rank": "group_rank",
        },
    }

    _cython_arity = {"ohlc": 4}  # OHLC

    @classmethod
    def get_kind_from_how(cls, how: str) -> str:
        if how in cls._CYTHON_FUNCTIONS["aggregate"]:
            return "aggregate"
        return "transform"

    # Note: we make this a classmethod and pass kind+how so that caching
    #  works at the class level and not the instance level
    @classmethod
    @functools.cache
    def _get_cython_function(
        cls, kind: str, how: str, dtype: np.dtype, is_numeric: bool
    ):
        dtype_str = dtype.name
        ftype = cls._CYTHON_FUNCTIONS[kind][how]

        # see if there is a fused-type version of function
        # only valid for numeric
        if callable(ftype):
            f = ftype
        else:
            f = getattr(libgroupby, ftype)
        if is_numeric:
            return f
        elif dtype == np.dtype(object):
            if how in ["median", "cumprod"]:
                # no fused types -> no __signatures__
                raise NotImplementedError(
                    f"function is not implemented for this dtype: "
                    f"[how->{how},dtype->{dtype_str}]"
                )
            elif how in ["std", "sem", "idxmin", "idxmax"]:
                # We have a partial object that does not have __signatures__
                return f
            elif how == "skew":
                # _get_cython_vals will convert to float64
                pass
            elif "object" not in f.__signatures__:
                # raise NotImplementedError here rather than TypeError later
                raise NotImplementedError(
                    f"function is not implemented for this dtype: "
                    f"[how->{how},dtype->{dtype_str}]"
                )
            return f
        else:
            raise NotImplementedError(
                "This should not be reached. Please report a bug at "
                "github.com/pandas-dev/pandas/",
                dtype,
            )

    def _get_cython_vals(self, values: np.ndarray) -> np.ndarray:
        """
        Cast numeric dtypes to float64 for functions that only support that.

        Parameters
        ----------
        values : np.ndarray

        Returns
        -------
        values : np.ndarray
        """
        how = self.how

        if how in ["median", "std", "sem", "skew"]:
            # median only has a float64 implementation
            # We should only get here with is_numeric, as non-numeric cases
            #  should raise in _get_cython_function
            values = ensure_float64(values)

        elif values.dtype.kind in "iu":
            if how in ["var", "mean"] or (
                self.kind == "transform" and self.has_dropped_na
            ):
                # has_dropped_na check need for test_null_group_str_transformer
                # result may still include NaN, so we have to cast
                values = ensure_float64(values)

            elif how in ["sum", "ohlc", "prod", "cumsum", "cumprod"]:
                # Avoid overflow during group op
                if values.dtype.kind == "i":
                    values = ensure_int64(values)
                else:
                    values = ensure_uint64(values)

        return values

    def _get_output_shape(self, ngroups: int, values: np.ndarray) -> Shape:
        how = self.how
        kind = self.kind

        arity = self._cython_arity.get(how, 1)

        out_shape: Shape
        if how == "ohlc":
            out_shape = (ngroups, arity)
        elif arity > 1:
            raise NotImplementedError(
                "arity of more than 1 is not supported for the 'how' argument"
            )
        elif kind == "transform":
            out_shape = values.shape
        else:
            out_shape = (ngroups,) + values.shape[1:]
        return out_shape

    def _get_out_dtype(self, dtype: np.dtype) -> np.dtype:
        how = self.how

        if how == "rank":
            out_dtype = "float64"
        elif how in ["idxmin", "idxmax"]:
            # The Cython implementation only produces the row number; we'll take
            # from the index using this in post processing
            out_dtype = "intp"
        else:
            if dtype.kind in "iufcb":
                out_dtype = f"{dtype.kind}{dtype.itemsize}"
            else:
                out_dtype = "object"
        return np.dtype(out_dtype)

    def _get_result_dtype(self, dtype: np.dtype) -> np.dtype:
        """
        Get the desired dtype of a result based on the
        input dtype and how it was computed.

        Parameters
        ----------
        dtype : np.dtype

        Returns
        -------
        np.dtype
            The desired dtype of the result.
        """
        how = self.how

        if how in ["sum", "cumsum", "sum", "prod", "cumprod"]:
            if dtype == np.dtype(bool):
                return np.dtype(np.int64)
        elif how in ["mean", "median", "var", "std", "sem"]:
            if dtype.kind in "fc":
                return dtype
            elif dtype.kind in "iub":
                return np.dtype(np.float64)
        return dtype

    @final
    def _cython_op_ndim_compat(
        self,
        values: np.ndarray,
        *,
        min_count: int,
        ngroups: int,
        comp_ids: np.ndarray,
        mask: npt.NDArray[np.bool_] | None = None,
        result_mask: npt.NDArray[np.bool_] | None = None,
        **kwargs,
    ) -> np.ndarray:
        if values.ndim == 1:
            # expand to 2d, dispatch, then squeeze if appropriate
            values2d = values[None, :]
            if mask is not None:
                mask = mask[None, :]
            if result_mask is not None:
                result_mask = result_mask[None, :]
            res = self._call_cython_op(
                values2d,
                min_count=min_count,
                ngroups=ngroups,
                comp_ids=comp_ids,
                mask=mask,
                result_mask=result_mask,
                **kwargs,
            )
            if res.shape[0] == 1:
                return res[0]

            # otherwise we have OHLC
            return res.T

        return self._call_cython_op(
            values,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=comp_ids,
            mask=mask,
            result_mask=result_mask,
            **kwargs,
        )

    @final
    def _call_cython_op(
        self,
        values: np.ndarray,  # np.ndarray[ndim=2]
        *,
        min_count: int,
        ngroups: int,
        comp_ids: np.ndarray,
        mask: npt.NDArray[np.bool_] | None,
        result_mask: npt.NDArray[np.bool_] | None,
        **kwargs,
    ) -> np.ndarray:  # np.ndarray[ndim=2]
        orig_values = values

        dtype = values.dtype
        is_numeric = dtype.kind in "iufcb"

        is_datetimelike = dtype.kind in "mM"

        if is_datetimelike:
            values = values.view("int64")
            is_numeric = True
        elif dtype.kind == "b":
            values = values.view("uint8")
        if values.dtype == "float16":
            values = values.astype(np.float32)

        if self.how in ["any", "all"]:
            if mask is None:
                mask = isna(values)
            if dtype == object:
                if kwargs["skipna"]:
                    # GH#37501: don't raise on pd.NA when skipna=True
                    if mask.any():
                        # mask on original values computed separately
                        values = values.copy()
                        values[mask] = True
            values = values.astype(bool, copy=False).view(np.int8)
            is_numeric = True

        values = values.T
        if mask is not None:
            mask = mask.T
            if result_mask is not None:
                result_mask = result_mask.T

        out_shape = self._get_output_shape(ngroups, values)
        func = self._get_cython_function(self.kind, self.how, values.dtype, is_numeric)
        values = self._get_cython_vals(values)
        out_dtype = self._get_out_dtype(values.dtype)

        result = maybe_fill(np.empty(out_shape, dtype=out_dtype))
        if self.kind == "aggregate":
            counts = np.zeros(ngroups, dtype=np.int64)
            if self.how in [
                "idxmin",
                "idxmax",
                "min",
                "max",
                "mean",
                "last",
                "first",
                "sum",
            ]:
                func(
                    out=result,
                    counts=counts,
                    values=values,
                    labels=comp_ids,
                    min_count=min_count,
                    mask=mask,
                    result_mask=result_mask,
                    is_datetimelike=is_datetimelike,
                    **kwargs,
                )
            elif self.how in ["sem", "std", "var", "ohlc", "prod", "median"]:
                if self.how in ["std", "sem"]:
                    kwargs["is_datetimelike"] = is_datetimelike
                func(
                    result,
                    counts,
                    values,
                    comp_ids,
                    min_count=min_count,
                    mask=mask,
                    result_mask=result_mask,
                    **kwargs,
                )
            elif self.how in ["any", "all"]:
                func(
                    out=result,
                    values=values,
                    labels=comp_ids,
                    mask=mask,
                    result_mask=result_mask,
                    **kwargs,
                )
                result = result.astype(bool, copy=False)
            elif self.how in ["skew"]:
                func(
                    out=result,
                    counts=counts,
                    values=values,
                    labels=comp_ids,
                    mask=mask,
                    result_mask=result_mask,
                    **kwargs,
                )
                if dtype == object:
                    result = result.astype(object)

            else:
                raise NotImplementedError(f"{self.how} is not implemented")
        else:
            # TODO: min_count
            if self.how != "rank":
                # TODO: should rank take result_mask?
                kwargs["result_mask"] = result_mask
            func(
                out=result,
                values=values,
                labels=comp_ids,
                ngroups=ngroups,
                is_datetimelike=is_datetimelike,
                mask=mask,
                **kwargs,
            )

        if self.kind == "aggregate" and self.how not in ["idxmin", "idxmax"]:
            # i.e. counts is defined.  Locations where count<min_count
            # need to have the result set to np.nan, which may require casting,
            # see GH#40767. For idxmin/idxmax is handled specially via post-processing
            if result.dtype.kind in "iu" and not is_datetimelike:
                # if the op keeps the int dtypes, we have to use 0
                cutoff = max(0 if self.how in ["sum", "prod"] else 1, min_count)
                empty_groups = counts < cutoff
                if empty_groups.any():
                    if result_mask is not None:
                        assert result_mask[empty_groups].all()
                    else:
                        # Note: this conversion could be lossy, see GH#40767
                        result = result.astype("float64")
                        result[empty_groups] = np.nan

        result = result.T

        if self.how not in self.cast_blocklist:
            # e.g. if we are int64 and need to restore to datetime64/timedelta64
            # "rank" is the only member of cast_blocklist we get here
            # Casting only needed for float16, bool, datetimelike,
            #  and self.how in ["sum", "prod", "ohlc", "cumprod"]
            res_dtype = self._get_result_dtype(orig_values.dtype)
            op_result = maybe_downcast_to_dtype(result, res_dtype)
        else:
            op_result = result

        return op_result

    @final
    def _validate_axis(self, axis: AxisInt, values: ArrayLike) -> None:
        if values.ndim > 2:
            raise NotImplementedError("number of dimensions is currently limited to 2")
        if values.ndim == 2:
            assert axis == 1, axis
        elif not is_1d_only_ea_dtype(values.dtype):
            # Note: it is *not* the case that axis is always 0 for 1-dim values,
            #  as we can have 1D ExtensionArrays that we need to treat as 2D
            assert axis == 0

    @final
    def cython_operation(
        self,
        *,
        values: ArrayLike,
        axis: AxisInt,
        min_count: int = -1,
        comp_ids: np.ndarray,
        ngroups: int,
        **kwargs,
    ) -> ArrayLike:
        """
        Call our cython function, with appropriate pre- and post- processing.
        """
        self._validate_axis(axis, values)

        if not isinstance(values, np.ndarray):
            # i.e. ExtensionArray
            return values._groupby_op(
                how=self.how,
                has_dropped_na=self.has_dropped_na,
                min_count=min_count,
                ngroups=ngroups,
                ids=comp_ids,
                **kwargs,
            )

        return self._cython_op_ndim_compat(
            values,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=comp_ids,
            mask=None,
            **kwargs,
        )


class BaseGrouper:
    """
    This is an internal Grouper class, which actually holds
    the generated groups

    Parameters
    ----------
    axis : Index
    groupings : Sequence[Grouping]
        all the grouping instances to handle in this grouper
        for example for grouper list to groupby, need to pass the list
    sort : bool, default True
        whether this grouper will give sorted result or not

    """

    axis: Index

    def __init__(
        self,
        axis: Index,
        groupings: Sequence[grouper.Grouping],
        sort: bool = True,
        dropna: bool = True,
    ) -> None:
        assert isinstance(axis, Index), axis

        self.axis = axis
        self._groupings: list[grouper.Grouping] = list(groupings)
        self._sort = sort
        self.dropna = dropna

    @property
    def groupings(self) -> list[grouper.Grouping]:
        return self._groupings

    @property
    def shape(self) -> Shape:
        return tuple(ping.ngroups for ping in self.groupings)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.indices)

    @property
    def nkeys(self) -> int:
        return len(self.groupings)

    def get_iterator(
        self, data: NDFrameT, axis: AxisInt = 0
    ) -> Iterator[tuple[Hashable, NDFrameT]]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        splitter = self._get_splitter(data, axis=axis)
        keys = self.group_keys_seq
        yield from zip(keys, splitter)

    @final
    def _get_splitter(self, data: NDFrame, axis: AxisInt = 0) -> DataSplitter:
        """
        Returns
        -------
        Generator yielding subsetted objects
        """
        ids, _, ngroups = self.group_info
        return _get_splitter(
            data,
            ids,
            ngroups,
            sorted_ids=self._sorted_ids,
            sort_idx=self._sort_idx,
            axis=axis,
        )

    @final
    @cache_readonly
    def group_keys_seq(self):
        if len(self.groupings) == 1:
            return self.levels[0]
        else:
            ids, _, ngroups = self.group_info

            # provide "flattened" iterator for multi-group setting
            return get_flattened_list(ids, ngroups, self.levels, self.codes)

    @cache_readonly
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """dict {group name -> group indices}"""
        if len(self.groupings) == 1 and isinstance(self.result_index, CategoricalIndex):
            # This shows unused categories in indices GH#38642
            return self.groupings[0].indices
        codes_list = [ping.codes for ping in self.groupings]
        keys = [ping._group_index for ping in self.groupings]
        return get_indexer_dict(codes_list, keys)

    @final
    def result_ilocs(self) -> npt.NDArray[np.intp]:
        """
        Get the original integer locations of result_index in the input.
        """
        # Original indices are where group_index would go via sorting.
        # But when dropna is true, we need to remove null values while accounting for
        # any gaps that then occur because of them.
        group_index = get_group_index(
            self.codes, self.shape, sort=self._sort, xnull=True
        )
        group_index, _ = compress_group_index(group_index, sort=self._sort)

        if self.has_dropped_na:
            mask = np.where(group_index >= 0)
            # Count how many gaps are caused by previous null values for each position
            null_gaps = np.cumsum(group_index == -1)[mask]
            group_index = group_index[mask]

        result = get_group_index_sorter(group_index, self.ngroups)

        if self.has_dropped_na:
            # Shift by the number of prior null gaps
            result += np.take(null_gaps, result)

        return result

    @final
    @property
    def codes(self) -> list[npt.NDArray[np.signedinteger]]:
        return [ping.codes for ping in self.groupings]

    @property
    def levels(self) -> list[Index]:
        return [ping._group_index for ping in self.groupings]

    @property
    def names(self) -> list[Hashable]:
        return [ping.name for ping in self.groupings]

    @final
    def size(self) -> Series:
        """
        Compute group sizes.
        """
        ids, _, ngroups = self.group_info
        out: np.ndarray | list
        if ngroups:
            out = np.bincount(ids[ids != -1], minlength=ngroups)
        else:
            out = []
        return Series(out, index=self.result_index, dtype="int64", copy=False)

    @cache_readonly
    def groups(self) -> dict[Hashable, np.ndarray]:
        """dict {group name -> group labels}"""
        if len(self.groupings) == 1:
            return self.groupings[0].groups
        else:
            to_groupby = []
            for ping in self.groupings:
                gv = ping.grouping_vector
                if not isinstance(gv, BaseGrouper):
                    to_groupby.append(gv)
                else:
                    to_groupby.append(gv.groupings[0].grouping_vector)
            index = MultiIndex.from_arrays(to_groupby)
            return self.axis.groupby(index)

    @final
    @cache_readonly
    def is_monotonic(self) -> bool:
        # return if my group orderings are monotonic
        return Index(self.group_info[0]).is_monotonic_increasing

    @final
    @cache_readonly
    def has_dropped_na(self) -> bool:
        """
        Whether grouper has null value(s) that are dropped.
        """
        return bool((self.group_info[0] < 0).any())

    @cache_readonly
    def group_info(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
        comp_ids, obs_group_ids = self._get_compressed_codes()

        ngroups = len(obs_group_ids)
        comp_ids = ensure_platform_int(comp_ids)

        return comp_ids, obs_group_ids, ngroups

    @cache_readonly
    def codes_info(self) -> npt.NDArray[np.intp]:
        # return the codes of items in original grouped axis
        ids, _, _ = self.group_info
        return ids

    @final
    def _get_compressed_codes(
        self,
    ) -> tuple[npt.NDArray[np.signedinteger], npt.NDArray[np.intp]]:
        # The first returned ndarray may have any signed integer dtype
        if len(self.groupings) > 1:
            group_index = get_group_index(self.codes, self.shape, sort=True, xnull=True)
            return compress_group_index(group_index, sort=self._sort)
            # FIXME: compress_group_index's second return value is int64, not intp

        ping = self.groupings[0]
        return ping.codes, np.arange(len(ping._group_index), dtype=np.intp)

    @final
    @cache_readonly
    def ngroups(self) -> int:
        return len(self.result_index)

    @property
    def reconstructed_codes(self) -> list[npt.NDArray[np.intp]]:
        codes = self.codes
        ids, obs_ids, _ = self.group_info
        return decons_obs_group_ids(ids, obs_ids, self.shape, codes, xnull=True)

    @cache_readonly
    def result_index(self) -> Index:
        if len(self.groupings) == 1:
            return self.groupings[0]._result_index.rename(self.names[0])

        codes = self.reconstructed_codes
        levels = [ping._result_index for ping in self.groupings]
        return MultiIndex(
            levels=levels, codes=codes, verify_integrity=False, names=self.names
        )

    @final
    def get_group_levels(self) -> list[ArrayLike]:
        # Note: only called from _insert_inaxis_grouper, which
        #  is only called for BaseGrouper, never for BinGrouper
        if len(self.groupings) == 1:
            return [self.groupings[0]._group_arraylike]

        name_list = []
        for ping, codes in zip(self.groupings, self.reconstructed_codes):
            codes = ensure_platform_int(codes)
            levels = ping._group_arraylike.take(codes)

            name_list.append(levels)

        return name_list

    # ------------------------------------------------------------
    # Aggregation functions

    @final
    def _cython_operation(
        self,
        kind: str,
        values,
        how: str,
        axis: AxisInt,
        min_count: int = -1,
        **kwargs,
    ) -> ArrayLike:
        """
        Returns the values of a cython operation.
        """
        assert kind in ["transform", "aggregate"]

        cy_op = WrappedCythonOp(kind=kind, how=how, has_dropped_na=self.has_dropped_na)

        ids, _, _ = self.group_info
        ngroups = self.ngroups
        return cy_op.cython_operation(
            values=values,
            axis=axis,
            min_count=min_count,
            comp_ids=ids,
            ngroups=ngroups,
            **kwargs,
        )

    @final
    def agg_series(
        self, obj: Series, func: Callable, preserve_dtype: bool = False
    ) -> ArrayLike:
        """
        Parameters
        ----------
        obj : Series
        func : function taking a Series and returning a scalar-like
        preserve_dtype : bool
            Whether the aggregation is known to be dtype-preserving.

        Returns
        -------
        np.ndarray or ExtensionArray
        """

        if not isinstance(obj._values, np.ndarray):
            # we can preserve a little bit more aggressively with EA dtype
            #  because maybe_cast_pointwise_result will do a try/except
            #  with _from_sequence.  NB we are assuming here that _from_sequence
            #  is sufficiently strict that it casts appropriately.
            preserve_dtype = True

        result = self._aggregate_series_pure_python(obj, func)

        npvalues = lib.maybe_convert_objects(result, try_float=False)
        if preserve_dtype:
            out = maybe_cast_pointwise_result(npvalues, obj.dtype, numeric_only=True)
        else:
            out = npvalues
        return out

    @final
    def _aggregate_series_pure_python(
        self, obj: Series, func: Callable
    ) -> npt.NDArray[np.object_]:
        _, _, ngroups = self.group_info

        result = np.empty(ngroups, dtype="O")
        initialized = False

        splitter = self._get_splitter(obj, axis=0)

        for i, group in enumerate(splitter):
            res = func(group)
            res = extract_result(res)

            if not initialized:
                # We only do this validation on the first iteration
                check_result_array(res, group.dtype)
                initialized = True

            result[i] = res

        return result

    @final
    def apply_groupwise(
        self, f: Callable, data: DataFrame | Series, axis: AxisInt = 0
    ) -> tuple[list, bool]:
        mutated = False
        splitter = self._get_splitter(data, axis=axis)
        group_keys = self.group_keys_seq
        result_values = []

        # This calls DataSplitter.__iter__
        zipped = zip(group_keys, splitter)

        for key, group in zipped:
            # Pinning name is needed for
            #  test_group_apply_once_per_group,
            #  test_inconsistent_return_type, test_set_group_name,
            #  test_group_name_available_in_inference_pass,
            #  test_groupby_multi_timezone
            object.__setattr__(group, "name", key)

            # group might be modified
            group_axes = group.axes
            res = f(group)
            if not mutated and not _is_indexed_like(res, group_axes, axis):
                mutated = True
            result_values.append(res)
        # getattr pattern for __name__ is needed for functools.partial objects
        if len(group_keys) == 0 and getattr(f, "__name__", None) in [
            "skew",
            "sum",
            "prod",
        ]:
            #  If group_keys is empty, then no function calls have been made,
            #  so we will not have raised even if this is an invalid dtype.
            #  So do one dummy call here to raise appropriate TypeError.
            f(data.iloc[:0])

        return result_values, mutated

    # ------------------------------------------------------------
    # Methods for sorting subsets of our GroupBy's object

    @final
    @cache_readonly
    def _sort_idx(self) -> npt.NDArray[np.intp]:
        # Counting sort indexer
        ids, _, ngroups = self.group_info
        return get_group_index_sorter(ids, ngroups)

    @final
    @cache_readonly
    def _sorted_ids(self) -> npt.NDArray[np.intp]:
        ids, _, _ = self.group_info
        return ids.take(self._sort_idx)


class BinGrouper(BaseGrouper):
    """
    This is an internal Grouper class

    Parameters
    ----------
    bins : the split index of binlabels to group the item of axis
    binlabels : the label list
    indexer : np.ndarray[np.intp], optional
        the indexer created by Grouper
        some groupers (TimeGrouper) will sort its axis and its
        group_info is also sorted, so need the indexer to reorder

    Examples
    --------
    bins: [2, 4, 6, 8, 10]
    binlabels: DatetimeIndex(['2005-01-01', '2005-01-03',
        '2005-01-05', '2005-01-07', '2005-01-09'],
        dtype='datetime64[ns]', freq='2D')

    the group_info, which contains the label of each item in grouped
    axis, the index of label in label list, group number, is

    (array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), array([0, 1, 2, 3, 4]), 5)

    means that, the grouped axis has 10 items, can be grouped into 5
    labels, the first and second items belong to the first label, the
    third and forth items belong to the second label, and so on

    """

    bins: npt.NDArray[np.int64]
    binlabels: Index

    def __init__(
        self,
        bins,
        binlabels,
        indexer=None,
    ) -> None:
        self.bins = ensure_int64(bins)
        self.binlabels = ensure_index(binlabels)
        self.indexer = indexer

        # These lengths must match, otherwise we could call agg_series
        #  with empty self.bins, which would raise later.
        assert len(self.binlabels) == len(self.bins)

    @cache_readonly
    def groups(self):
        """dict {group name -> group labels}"""
        # this is mainly for compat
        # GH 3881
        result = {
            key: value
            for key, value in zip(self.binlabels, self.bins)
            if key is not NaT
        }
        return result

    @property
    def nkeys(self) -> int:
        # still matches len(self.groupings), but we can hard-code
        return 1

    @cache_readonly
    def codes_info(self) -> npt.NDArray[np.intp]:
        # return the codes of items in original grouped axis
        ids, _, _ = self.group_info
        if self.indexer is not None:
            sorter = np.lexsort((ids, self.indexer))
            ids = ids[sorter]
        return ids

    def get_iterator(self, data: NDFrame, axis: AxisInt = 0):
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        if axis == 0:
            slicer = lambda start, edge: data.iloc[start:edge]
        else:
            slicer = lambda start, edge: data.iloc[:, start:edge]

        length = len(data.axes[axis])

        start = 0
        for edge, label in zip(self.bins, self.binlabels):
            if label is not NaT:
                yield label, slicer(start, edge)
            start = edge

        if start < length:
            yield self.binlabels[-1], slicer(start, None)

    @cache_readonly
    def indices(self):
        indices = collections.defaultdict(list)

        i = 0
        for label, bin in zip(self.binlabels, self.bins):
            if i < bin:
                if label is not NaT:
                    indices[label] = list(range(i, bin))
                i = bin
        return indices

    @cache_readonly
    def group_info(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
        ngroups = self.ngroups
        obs_group_ids = np.arange(ngroups, dtype=np.intp)
        rep = np.diff(np.r_[0, self.bins])

        rep = ensure_platform_int(rep)
        if ngroups == len(self.bins):
            comp_ids = np.repeat(np.arange(ngroups), rep)
        else:
            comp_ids = np.repeat(np.r_[-1, np.arange(ngroups)], rep)

        return (
            ensure_platform_int(comp_ids),
            obs_group_ids,
            ngroups,
        )

    @cache_readonly
    def reconstructed_codes(self) -> list[np.ndarray]:
        # get unique result indices, and prepend 0 as groupby starts from the first
        return [np.r_[0, np.flatnonzero(self.bins[1:] != self.bins[:-1]) + 1]]

    @cache_readonly
    def result_index(self) -> Index:
        if len(self.binlabels) != 0 and isna(self.binlabels[0]):
            return self.binlabels[1:]

        return self.binlabels

    @property
    def levels(self) -> list[Index]:
        return [self.binlabels]

    @property
    def names(self) -> list[Hashable]:
        return [self.binlabels.name]

    @property
    def groupings(self) -> list[grouper.Grouping]:
        lev = self.binlabels
        codes = self.group_info[0]
        labels = lev.take(codes)
        ping = grouper.Grouping(
            labels, labels, in_axis=False, level=None, uniques=lev._values
        )
        return [ping]


def _is_indexed_like(obj, axes, axis: AxisInt) -> bool:
    if isinstance(obj, Series):
        if len(axes) > 1:
            return False
        return obj.axes[axis].equals(axes[axis])
    elif isinstance(obj, DataFrame):
        return obj.axes[axis].equals(axes[axis])

    return False


# ----------------------------------------------------------------------
# Splitting / application


class DataSplitter(Generic[NDFrameT]):
    def __init__(
        self,
        data: NDFrameT,
        labels: npt.NDArray[np.intp],
        ngroups: int,
        *,
        sort_idx: npt.NDArray[np.intp],
        sorted_ids: npt.NDArray[np.intp],
        axis: AxisInt = 0,
    ) -> None:
        self.data = data
        self.labels = ensure_platform_int(labels)  # _should_ already be np.intp
        self.ngroups = ngroups

        self._slabels = sorted_ids
        self._sort_idx = sort_idx

        self.axis = axis
        assert isinstance(axis, int), axis

    def __iter__(self) -> Iterator:
        sdata = self._sorted_data

        if self.ngroups == 0:
            # we are inside a generator, rather than raise StopIteration
            # we merely return signal the end
            return

        starts, ends = lib.generate_slices(self._slabels, self.ngroups)

        for start, end in zip(starts, ends):
            yield self._chop(sdata, slice(start, end))

    @cache_readonly
    def _sorted_data(self) -> NDFrameT:
        return self.data.take(self._sort_idx, axis=self.axis)

    def _chop(self, sdata, slice_obj: slice) -> NDFrame:
        raise AbstractMethodError(self)


class SeriesSplitter(DataSplitter):
    def _chop(self, sdata: Series, slice_obj: slice) -> Series:
        # fastpath equivalent to `sdata.iloc[slice_obj]`
        mgr = sdata._mgr.get_slice(slice_obj)
        ser = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
        ser._name = sdata.name
        return ser.__finalize__(sdata, method="groupby")


class FrameSplitter(DataSplitter):
    def _chop(self, sdata: DataFrame, slice_obj: slice) -> DataFrame:
        # Fastpath equivalent to:
        # if self.axis == 0:
        #     return sdata.iloc[slice_obj]
        # else:
        #     return sdata.iloc[:, slice_obj]
        mgr = sdata._mgr.get_slice(slice_obj, axis=1 - self.axis)
        df = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(sdata, method="groupby")


def _get_splitter(
    data: NDFrame,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    *,
    sort_idx: npt.NDArray[np.intp],
    sorted_ids: npt.NDArray[np.intp],
    axis: AxisInt = 0,
) -> DataSplitter:
    if isinstance(data, Series):
        klass: type[DataSplitter] = SeriesSplitter
    else:
        # i.e. DataFrame
        klass = FrameSplitter

    return klass(
        data, labels, ngroups, sort_idx=sort_idx, sorted_ids=sorted_ids, axis=axis
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\nodes.py ===
"""AST nodes generated by the parser for the compiler. Also provides
some node tree helper functions used by the parser and compiler in order
to normalize nodes.
"""

import inspect
import operator
import typing as t
from collections import deque

from markupsafe import Markup

from .utils import _PassArg

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

_NodeBound = t.TypeVar("_NodeBound", bound="Node")

_binop_to_func: t.Dict[str, t.Callable[[t.Any, t.Any], t.Any]] = {
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "**": operator.pow,
    "%": operator.mod,
    "+": operator.add,
    "-": operator.sub,
}

_uaop_to_func: t.Dict[str, t.Callable[[t.Any], t.Any]] = {
    "not": operator.not_,
    "+": operator.pos,
    "-": operator.neg,
}

_cmpop_to_func: t.Dict[str, t.Callable[[t.Any, t.Any], t.Any]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gteq": operator.ge,
    "lt": operator.lt,
    "lteq": operator.le,
    "in": lambda a, b: a in b,
    "notin": lambda a, b: a not in b,
}


class Impossible(Exception):
    """Raised if the node could not perform a requested action."""


class NodeType(type):
    """A metaclass for nodes that handles the field and attribute
    inheritance.  fields and attributes from the parent class are
    automatically forwarded to the child."""

    def __new__(mcs, name, bases, d):  # type: ignore
        for attr in "fields", "attributes":
            storage: t.List[t.Tuple[str, ...]] = []
            storage.extend(getattr(bases[0] if bases else object, attr, ()))
            storage.extend(d.get(attr, ()))
            assert len(bases) <= 1, "multiple inheritance not allowed"
            assert len(storage) == len(set(storage)), "layout conflict"
            d[attr] = tuple(storage)
        d.setdefault("abstract", False)
        return type.__new__(mcs, name, bases, d)


class EvalContext:
    """Holds evaluation time information.  Custom attributes can be attached
    to it in extensions.
    """

    def __init__(
        self, environment: "Environment", template_name: t.Optional[str] = None
    ) -> None:
        self.environment = environment
        if callable(environment.autoescape):
            self.autoescape = environment.autoescape(template_name)
        else:
            self.autoescape = environment.autoescape
        self.volatile = False

    def save(self) -> t.Mapping[str, t.Any]:
        return self.__dict__.copy()

    def revert(self, old: t.Mapping[str, t.Any]) -> None:
        self.__dict__.clear()
        self.__dict__.update(old)


def get_eval_context(node: "Node", ctx: t.Optional[EvalContext]) -> EvalContext:
    if ctx is None:
        if node.environment is None:
            raise RuntimeError(
                "if no eval context is passed, the node must have an"
                " attached environment."
            )
        return EvalContext(node.environment)
    return ctx


class Node(metaclass=NodeType):
    """Baseclass for all Jinja nodes.  There are a number of nodes available
    of different types.  There are four major types:

    -   :class:`Stmt`: statements
    -   :class:`Expr`: expressions
    -   :class:`Helper`: helper nodes
    -   :class:`Template`: the outermost wrapper node

    All nodes have fields and attributes.  Fields may be other nodes, lists,
    or arbitrary values.  Fields are passed to the constructor as regular
    positional arguments, attributes as keyword arguments.  Each node has
    two attributes: `lineno` (the line number of the node) and `environment`.
    The `environment` attribute is set at the end of the parsing process for
    all nodes automatically.
    """

    fields: t.Tuple[str, ...] = ()
    attributes: t.Tuple[str, ...] = ("lineno", "environment")
    abstract = True

    lineno: int
    environment: t.Optional["Environment"]

    def __init__(self, *fields: t.Any, **attributes: t.Any) -> None:
        if self.abstract:
            raise TypeError("abstract nodes are not instantiable")
        if fields:
            if len(fields) != len(self.fields):
                if not self.fields:
                    raise TypeError(f"{type(self).__name__!r} takes 0 arguments")
                raise TypeError(
                    f"{type(self).__name__!r} takes 0 or {len(self.fields)}"
                    f" argument{'s' if len(self.fields) != 1 else ''}"
                )
            for name, arg in zip(self.fields, fields):
                setattr(self, name, arg)
        for attr in self.attributes:
            setattr(self, attr, attributes.pop(attr, None))
        if attributes:
            raise TypeError(f"unknown attribute {next(iter(attributes))!r}")

    def iter_fields(
        self,
        exclude: t.Optional[t.Container[str]] = None,
        only: t.Optional[t.Container[str]] = None,
    ) -> t.Iterator[t.Tuple[str, t.Any]]:
        """This method iterates over all fields that are defined and yields
        ``(key, value)`` tuples.  Per default all fields are returned, but
        it's possible to limit that to some fields by providing the `only`
        parameter or to exclude some using the `exclude` parameter.  Both
        should be sets or tuples of field names.
        """
        for name in self.fields:
            if (
                (exclude is None and only is None)
                or (exclude is not None and name not in exclude)
                or (only is not None and name in only)
            ):
                try:
                    yield name, getattr(self, name)
                except AttributeError:
                    pass

    def iter_child_nodes(
        self,
        exclude: t.Optional[t.Container[str]] = None,
        only: t.Optional[t.Container[str]] = None,
    ) -> t.Iterator["Node"]:
        """Iterates over all direct child nodes of the node.  This iterates
        over all fields and yields the values of they are nodes.  If the value
        of a field is a list all the nodes in that list are returned.
        """
        for _, item in self.iter_fields(exclude, only):
            if isinstance(item, list):
                for n in item:
                    if isinstance(n, Node):
                        yield n
            elif isinstance(item, Node):
                yield item

    def find(self, node_type: t.Type[_NodeBound]) -> t.Optional[_NodeBound]:
        """Find the first node of a given type.  If no such node exists the
        return value is `None`.
        """
        for result in self.find_all(node_type):
            return result

        return None

    def find_all(
        self, node_type: t.Union[t.Type[_NodeBound], t.Tuple[t.Type[_NodeBound], ...]]
    ) -> t.Iterator[_NodeBound]:
        """Find all the nodes of a given type.  If the type is a tuple,
        the check is performed for any of the tuple items.
        """
        for child in self.iter_child_nodes():
            if isinstance(child, node_type):
                yield child  # type: ignore
            yield from child.find_all(node_type)

    def set_ctx(self, ctx: str) -> "Node":
        """Reset the context of a node and all child nodes.  Per default the
        parser will all generate nodes that have a 'load' context as it's the
        most common one.  This method is used in the parser to set assignment
        targets and other nodes to a store context.
        """
        todo = deque([self])
        while todo:
            node = todo.popleft()
            if "ctx" in node.fields:
                node.ctx = ctx  # type: ignore
            todo.extend(node.iter_child_nodes())
        return self

    def set_lineno(self, lineno: int, override: bool = False) -> "Node":
        """Set the line numbers of the node and children."""
        todo = deque([self])
        while todo:
            node = todo.popleft()
            if "lineno" in node.attributes:
                if node.lineno is None or override:
                    node.lineno = lineno
            todo.extend(node.iter_child_nodes())
        return self

    def set_environment(self, environment: "Environment") -> "Node":
        """Set the environment for all nodes."""
        todo = deque([self])
        while todo:
            node = todo.popleft()
            node.environment = environment
            todo.extend(node.iter_child_nodes())
        return self

    def __eq__(self, other: t.Any) -> bool:
        if type(self) is not type(other):
            return NotImplemented

        return tuple(self.iter_fields()) == tuple(other.iter_fields())

    __hash__ = object.__hash__

    def __repr__(self) -> str:
        args_str = ", ".join(f"{a}={getattr(self, a, None)!r}" for a in self.fields)
        return f"{type(self).__name__}({args_str})"

    def dump(self) -> str:
        def _dump(node: t.Union[Node, t.Any]) -> None:
            if not isinstance(node, Node):
                buf.append(repr(node))
                return

            buf.append(f"nodes.{type(node).__name__}(")
            if not node.fields:
                buf.append(")")
                return
            for idx, field in enumerate(node.fields):
                if idx:
                    buf.append(", ")
                value = getattr(node, field)
                if isinstance(value, list):
                    buf.append("[")
                    for idx, item in enumerate(value):
                        if idx:
                            buf.append(", ")
                        _dump(item)
                    buf.append("]")
                else:
                    _dump(value)
            buf.append(")")

        buf: t.List[str] = []
        _dump(self)
        return "".join(buf)


class Stmt(Node):
    """Base node for all statements."""

    abstract = True


class Helper(Node):
    """Nodes that exist in a specific context only."""

    abstract = True


class Template(Node):
    """Node that represents a template.  This must be the outermost node that
    is passed to the compiler.
    """

    fields = ("body",)
    body: t.List[Node]


class Output(Stmt):
    """A node that holds multiple expressions which are then printed out.
    This is used both for the `print` statement and the regular template data.
    """

    fields = ("nodes",)
    nodes: t.List["Expr"]


class Extends(Stmt):
    """Represents an extends statement."""

    fields = ("template",)
    template: "Expr"


class For(Stmt):
    """The for loop.  `target` is the target for the iteration (usually a
    :class:`Name` or :class:`Tuple`), `iter` the iterable.  `body` is a list
    of nodes that are used as loop-body, and `else_` a list of nodes for the
    `else` block.  If no else node exists it has to be an empty list.

    For filtered nodes an expression can be stored as `test`, otherwise `None`.
    """

    fields = ("target", "iter", "body", "else_", "test", "recursive")
    target: Node
    iter: Node
    body: t.List[Node]
    else_: t.List[Node]
    test: t.Optional[Node]
    recursive: bool


class If(Stmt):
    """If `test` is true, `body` is rendered, else `else_`."""

    fields = ("test", "body", "elif_", "else_")
    test: Node
    body: t.List[Node]
    elif_: t.List["If"]
    else_: t.List[Node]


class Macro(Stmt):
    """A macro definition.  `name` is the name of the macro, `args` a list of
    arguments and `defaults` a list of defaults if there are any.  `body` is
    a list of nodes for the macro body.
    """

    fields = ("name", "args", "defaults", "body")
    name: str
    args: t.List["Name"]
    defaults: t.List["Expr"]
    body: t.List[Node]


class CallBlock(Stmt):
    """Like a macro without a name but a call instead.  `call` is called with
    the unnamed macro as `caller` argument this node holds.
    """

    fields = ("call", "args", "defaults", "body")
    call: "Call"
    args: t.List["Name"]
    defaults: t.List["Expr"]
    body: t.List[Node]


class FilterBlock(Stmt):
    """Node for filter sections."""

    fields = ("body", "filter")
    body: t.List[Node]
    filter: "Filter"


class With(Stmt):
    """Specific node for with statements.  In older versions of Jinja the
    with statement was implemented on the base of the `Scope` node instead.

    .. versionadded:: 2.9.3
    """

    fields = ("targets", "values", "body")
    targets: t.List["Expr"]
    values: t.List["Expr"]
    body: t.List[Node]


class Block(Stmt):
    """A node that represents a block.

    .. versionchanged:: 3.0.0
        the `required` field was added.
    """

    fields = ("name", "body", "scoped", "required")
    name: str
    body: t.List[Node]
    scoped: bool
    required: bool


class Include(Stmt):
    """A node that represents the include tag."""

    fields = ("template", "with_context", "ignore_missing")
    template: "Expr"
    with_context: bool
    ignore_missing: bool


class Import(Stmt):
    """A node that represents the import tag."""

    fields = ("template", "target", "with_context")
    template: "Expr"
    target: str
    with_context: bool


class FromImport(Stmt):
    """A node that represents the from import tag.  It's important to not
    pass unsafe names to the name attribute.  The compiler translates the
    attribute lookups directly into getattr calls and does *not* use the
    subscript callback of the interface.  As exported variables may not
    start with double underscores (which the parser asserts) this is not a
    problem for regular Jinja code, but if this node is used in an extension
    extra care must be taken.

    The list of names may contain tuples if aliases are wanted.
    """

    fields = ("template", "names", "with_context")
    template: "Expr"
    names: t.List[t.Union[str, t.Tuple[str, str]]]
    with_context: bool


class ExprStmt(Stmt):
    """A statement that evaluates an expression and discards the result."""

    fields = ("node",)
    node: Node


class Assign(Stmt):
    """Assigns an expression to a target."""

    fields = ("target", "node")
    target: "Expr"
    node: Node


class AssignBlock(Stmt):
    """Assigns a block to a target."""

    fields = ("target", "filter", "body")
    target: "Expr"
    filter: t.Optional["Filter"]
    body: t.List[Node]


class Expr(Node):
    """Baseclass for all expressions."""

    abstract = True

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        """Return the value of the expression as constant or raise
        :exc:`Impossible` if this was not possible.

        An :class:`EvalContext` can be provided, if none is given
        a default context is created which requires the nodes to have
        an attached environment.

        .. versionchanged:: 2.4
           the `eval_ctx` parameter was added.
        """
        raise Impossible()

    def can_assign(self) -> bool:
        """Check if it's possible to assign something to this node."""
        return False


class BinExpr(Expr):
    """Baseclass for all binary expressions."""

    fields = ("left", "right")
    left: Expr
    right: Expr
    operator: str
    abstract = True

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)

        # intercepted operators cannot be folded at compile time
        if (
            eval_ctx.environment.sandboxed
            and self.operator in eval_ctx.environment.intercepted_binops  # type: ignore
        ):
            raise Impossible()
        f = _binop_to_func[self.operator]
        try:
            return f(self.left.as_const(eval_ctx), self.right.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e


class UnaryExpr(Expr):
    """Baseclass for all unary expressions."""

    fields = ("node",)
    node: Expr
    operator: str
    abstract = True

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)

        # intercepted operators cannot be folded at compile time
        if (
            eval_ctx.environment.sandboxed
            and self.operator in eval_ctx.environment.intercepted_unops  # type: ignore
        ):
            raise Impossible()
        f = _uaop_to_func[self.operator]
        try:
            return f(self.node.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e


class Name(Expr):
    """Looks up a name or stores a value in a name.
    The `ctx` of the node can be one of the following values:

    -   `store`: store a value in the name
    -   `load`: load that name
    -   `param`: like `store` but if the name was defined as function parameter.
    """

    fields = ("name", "ctx")
    name: str
    ctx: str

    def can_assign(self) -> bool:
        return self.name not in {"true", "false", "none", "True", "False", "None"}


class NSRef(Expr):
    """Reference to a namespace value assignment"""

    fields = ("name", "attr")
    name: str
    attr: str

    def can_assign(self) -> bool:
        # We don't need any special checks here; NSRef assignments have a
        # runtime check to ensure the target is a namespace object which will
        # have been checked already as it is created using a normal assignment
        # which goes through a `Name` node.
        return True


class Literal(Expr):
    """Baseclass for literals."""

    abstract = True


class Const(Literal):
    """All constant values.  The parser will return this node for simple
    constants such as ``42`` or ``"foo"`` but it can be used to store more
    complex values such as lists too.  Only constants with a safe
    representation (objects where ``eval(repr(x)) == x`` is true).
    """

    fields = ("value",)
    value: t.Any

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        return self.value

    @classmethod
    def from_untrusted(
        cls,
        value: t.Any,
        lineno: t.Optional[int] = None,
        environment: "t.Optional[Environment]" = None,
    ) -> "Const":
        """Return a const object if the value is representable as
        constant value in the generated code, otherwise it will raise
        an `Impossible` exception.
        """
        from .compiler import has_safe_repr

        if not has_safe_repr(value):
            raise Impossible()
        return cls(value, lineno=lineno, environment=environment)


class TemplateData(Literal):
    """A constant template string."""

    fields = ("data",)
    data: str

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> str:
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.volatile:
            raise Impossible()
        if eval_ctx.autoescape:
            return Markup(self.data)
        return self.data


class Tuple(Literal):
    """For loop unpacking and some other things like multiple arguments
    for subscripts.  Like for :class:`Name` `ctx` specifies if the tuple
    is used for loading the names or storing.
    """

    fields = ("items", "ctx")
    items: t.List[Expr]
    ctx: str

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Tuple[t.Any, ...]:
        eval_ctx = get_eval_context(self, eval_ctx)
        return tuple(x.as_const(eval_ctx) for x in self.items)

    def can_assign(self) -> bool:
        for item in self.items:
            if not item.can_assign():
                return False
        return True


class List(Literal):
    """Any list literal such as ``[1, 2, 3]``"""

    fields = ("items",)
    items: t.List[Expr]

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.List[t.Any]:
        eval_ctx = get_eval_context(self, eval_ctx)
        return [x.as_const(eval_ctx) for x in self.items]


class Dict(Literal):
    """Any dict literal such as ``{1: 2, 3: 4}``.  The items must be a list of
    :class:`Pair` nodes.
    """

    fields = ("items",)
    items: t.List["Pair"]

    def as_const(
        self, eval_ctx: t.Optional[EvalContext] = None
    ) -> t.Dict[t.Any, t.Any]:
        eval_ctx = get_eval_context(self, eval_ctx)
        return dict(x.as_const(eval_ctx) for x in self.items)


class Pair(Helper):
    """A key, value pair for dicts."""

    fields = ("key", "value")
    key: Expr
    value: Expr

    def as_const(
        self, eval_ctx: t.Optional[EvalContext] = None
    ) -> t.Tuple[t.Any, t.Any]:
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.key.as_const(eval_ctx), self.value.as_const(eval_ctx)


class Keyword(Helper):
    """A key, value pair for keyword arguments where key is a string."""

    fields = ("key", "value")
    key: str
    value: Expr

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Tuple[str, t.Any]:
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.key, self.value.as_const(eval_ctx)


class CondExpr(Expr):
    """A conditional expression (inline if expression).  (``{{
    foo if bar else baz }}``)
    """

    fields = ("test", "expr1", "expr2")
    test: Expr
    expr1: Expr
    expr2: t.Optional[Expr]

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        if self.test.as_const(eval_ctx):
            return self.expr1.as_const(eval_ctx)

        # if we evaluate to an undefined object, we better do that at runtime
        if self.expr2 is None:
            raise Impossible()

        return self.expr2.as_const(eval_ctx)


def args_as_const(
    node: t.Union["_FilterTestCommon", "Call"], eval_ctx: t.Optional[EvalContext]
) -> t.Tuple[t.List[t.Any], t.Dict[t.Any, t.Any]]:
    args = [x.as_const(eval_ctx) for x in node.args]
    kwargs = dict(x.as_const(eval_ctx) for x in node.kwargs)

    if node.dyn_args is not None:
        try:
            args.extend(node.dyn_args.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e

    if node.dyn_kwargs is not None:
        try:
            kwargs.update(node.dyn_kwargs.as_const(eval_ctx))
        except Exception as e:
            raise Impossible() from e

    return args, kwargs


class _FilterTestCommon(Expr):
    fields = ("node", "name", "args", "kwargs", "dyn_args", "dyn_kwargs")
    node: Expr
    name: str
    args: t.List[Expr]
    kwargs: t.List[Pair]
    dyn_args: t.Optional[Expr]
    dyn_kwargs: t.Optional[Expr]
    abstract = True
    _is_filter = True

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)

        if eval_ctx.volatile:
            raise Impossible()

        if self._is_filter:
            env_map = eval_ctx.environment.filters
        else:
            env_map = eval_ctx.environment.tests

        func = env_map.get(self.name)
        pass_arg = _PassArg.from_obj(func)  # type: ignore

        if func is None or pass_arg is _PassArg.context:
            raise Impossible()

        if eval_ctx.environment.is_async and (
            getattr(func, "jinja_async_variant", False) is True
            or inspect.iscoroutinefunction(func)
        ):
            raise Impossible()

        args, kwargs = args_as_const(self, eval_ctx)
        args.insert(0, self.node.as_const(eval_ctx))

        if pass_arg is _PassArg.eval_context:
            args.insert(0, eval_ctx)
        elif pass_arg is _PassArg.environment:
            args.insert(0, eval_ctx.environment)

        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise Impossible() from e


class Filter(_FilterTestCommon):
    """Apply a filter to an expression. ``name`` is the name of the
    filter, the other fields are the same as :class:`Call`.

    If ``node`` is ``None``, the filter is being used in a filter block
    and is applied to the content of the block.
    """

    node: t.Optional[Expr]  # type: ignore

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        if self.node is None:
            raise Impossible()

        return super().as_const(eval_ctx=eval_ctx)


class Test(_FilterTestCommon):
    """Apply a test to an expression. ``name`` is the name of the test,
    the other field are the same as :class:`Call`.

    .. versionchanged:: 3.0
        ``as_const`` shares the same logic for filters and tests. Tests
        check for volatile, async, and ``@pass_context`` etc.
        decorators.
    """

    _is_filter = False


class Call(Expr):
    """Calls an expression.  `args` is a list of arguments, `kwargs` a list
    of keyword arguments (list of :class:`Keyword` nodes), and `dyn_args`
    and `dyn_kwargs` has to be either `None` or a node that is used as
    node for dynamic positional (``*args``) or keyword (``**kwargs``)
    arguments.
    """

    fields = ("node", "args", "kwargs", "dyn_args", "dyn_kwargs")
    node: Expr
    args: t.List[Expr]
    kwargs: t.List[Keyword]
    dyn_args: t.Optional[Expr]
    dyn_kwargs: t.Optional[Expr]


class Getitem(Expr):
    """Get an attribute or item from an expression and prefer the item."""

    fields = ("node", "arg", "ctx")
    node: Expr
    arg: Expr
    ctx: str

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        if self.ctx != "load":
            raise Impossible()

        eval_ctx = get_eval_context(self, eval_ctx)

        try:
            return eval_ctx.environment.getitem(
                self.node.as_const(eval_ctx), self.arg.as_const(eval_ctx)
            )
        except Exception as e:
            raise Impossible() from e


class Getattr(Expr):
    """Get an attribute or item from an expression that is a ascii-only
    bytestring and prefer the attribute.
    """

    fields = ("node", "attr", "ctx")
    node: Expr
    attr: str
    ctx: str

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        if self.ctx != "load":
            raise Impossible()

        eval_ctx = get_eval_context(self, eval_ctx)

        try:
            return eval_ctx.environment.getattr(self.node.as_const(eval_ctx), self.attr)
        except Exception as e:
            raise Impossible() from e


class Slice(Expr):
    """Represents a slice object.  This must only be used as argument for
    :class:`Subscript`.
    """

    fields = ("start", "stop", "step")
    start: t.Optional[Expr]
    stop: t.Optional[Expr]
    step: t.Optional[Expr]

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> slice:
        eval_ctx = get_eval_context(self, eval_ctx)

        def const(obj: t.Optional[Expr]) -> t.Optional[t.Any]:
            if obj is None:
                return None
            return obj.as_const(eval_ctx)

        return slice(const(self.start), const(self.stop), const(self.step))


class Concat(Expr):
    """Concatenates the list of expressions provided after converting
    them to strings.
    """

    fields = ("nodes",)
    nodes: t.List[Expr]

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> str:
        eval_ctx = get_eval_context(self, eval_ctx)
        return "".join(str(x.as_const(eval_ctx)) for x in self.nodes)


class Compare(Expr):
    """Compares an expression with some other expressions.  `ops` must be a
    list of :class:`Operand`\\s.
    """

    fields = ("expr", "ops")
    expr: Expr
    ops: t.List["Operand"]

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        result = value = self.expr.as_const(eval_ctx)

        try:
            for op in self.ops:
                new_value = op.expr.as_const(eval_ctx)
                result = _cmpop_to_func[op.op](value, new_value)

                if not result:
                    return False

                value = new_value
        except Exception as e:
            raise Impossible() from e

        return result


class Operand(Helper):
    """Holds an operator and an expression."""

    fields = ("op", "expr")
    op: str
    expr: Expr


class Mul(BinExpr):
    """Multiplies the left with the right node."""

    operator = "*"


class Div(BinExpr):
    """Divides the left by the right node."""

    operator = "/"


class FloorDiv(BinExpr):
    """Divides the left by the right node and converts the
    result into an integer by truncating.
    """

    operator = "//"


class Add(BinExpr):
    """Add the left to the right node."""

    operator = "+"


class Sub(BinExpr):
    """Subtract the right from the left node."""

    operator = "-"


class Mod(BinExpr):
    """Left modulo right."""

    operator = "%"


class Pow(BinExpr):
    """Left to the power of right."""

    operator = "**"


class And(BinExpr):
    """Short circuited AND."""

    operator = "and"

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.left.as_const(eval_ctx) and self.right.as_const(eval_ctx)


class Or(BinExpr):
    """Short circuited OR."""

    operator = "or"

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> t.Any:
        eval_ctx = get_eval_context(self, eval_ctx)
        return self.left.as_const(eval_ctx) or self.right.as_const(eval_ctx)


class Not(UnaryExpr):
    """Negate the expression."""

    operator = "not"


class Neg(UnaryExpr):
    """Make the expression negative."""

    operator = "-"


class Pos(UnaryExpr):
    """Make the expression positive (noop for most expressions)"""

    operator = "+"


# Helpers for extensions


class EnvironmentAttribute(Expr):
    """Loads an attribute from the environment object.  This is useful for
    extensions that want to call a callback stored on the environment.
    """

    fields = ("name",)
    name: str


class ExtensionAttribute(Expr):
    """Returns the attribute of an extension bound to the environment.
    The identifier is the identifier of the :class:`Extension`.

    This node is usually constructed by calling the
    :meth:`~jinja2.ext.Extension.attr` method on an extension.
    """

    fields = ("identifier", "name")
    identifier: str
    name: str


class ImportedName(Expr):
    """If created with an import name the import name is returned on node
    access.  For example ``ImportedName('cgi.escape')`` returns the `escape`
    function from the cgi module on evaluation.  Imports are optimized by the
    compiler so there is no need to assign them to local variables.
    """

    fields = ("importname",)
    importname: str


class InternalName(Expr):
    """An internal name in the compiler.  You cannot create these nodes
    yourself but the parser provides a
    :meth:`~jinja2.parser.Parser.free_identifier` method that creates
    a new identifier for you.  This identifier is not available from the
    template and is not treated specially by the compiler.
    """

    fields = ("name",)
    name: str

    def __init__(self) -> None:
        raise TypeError(
            "Can't create internal names.  Use the "
            "`free_identifier` method on a parser."
        )


class MarkSafe(Expr):
    """Mark the wrapped expression as safe (wrap it as `Markup`)."""

    fields = ("expr",)
    expr: Expr

    def as_const(self, eval_ctx: t.Optional[EvalContext] = None) -> Markup:
        eval_ctx = get_eval_context(self, eval_ctx)
        return Markup(self.expr.as_const(eval_ctx))


class MarkSafeIfAutoescape(Expr):
    """Mark the wrapped expression as safe (wrap it as `Markup`) but
    only if autoescaping is active.

    .. versionadded:: 2.5
    """

    fields = ("expr",)
    expr: Expr

    def as_const(
        self, eval_ctx: t.Optional[EvalContext] = None
    ) -> t.Union[Markup, t.Any]:
        eval_ctx = get_eval_context(self, eval_ctx)
        if eval_ctx.volatile:
            raise Impossible()
        expr = self.expr.as_const(eval_ctx)
        if eval_ctx.autoescape:
            return Markup(expr)
        return expr


class ContextReference(Expr):
    """Returns the current template context.  It can be used like a
    :class:`Name` node, with a ``'load'`` ctx and will return the
    current :class:`~jinja2.runtime.Context` object.

    Here an example that assigns the current template name to a
    variable named `foo`::

        Assign(Name('foo', ctx='store'),
               Getattr(ContextReference(), 'name'))

    This is basically equivalent to using the
    :func:`~jinja2.pass_context` decorator when using the high-level
    API, which causes a reference to the context to be passed as the
    first argument to a function.
    """


class DerivedContextReference(Expr):
    """Return the current template context including locals. Behaves
    exactly like :class:`ContextReference`, but includes local
    variables, such as from a ``for`` loop.

    .. versionadded:: 2.11
    """


class Continue(Stmt):
    """Continue a loop."""


class Break(Stmt):
    """Break a loop."""


class Scope(Stmt):
    """An artificial scope."""

    fields = ("body",)
    body: t.List[Node]


class OverlayScope(Stmt):
    """An overlay scope for extensions.  This is a largely unoptimized scope
    that however can be used to introduce completely arbitrary variables into
    a sub scope from a dictionary or dictionary like object.  The `context`
    field has to evaluate to a dictionary object.

    Example usage::

        OverlayScope(context=self.call_method('get_context'),
                     body=[...])

    .. versionadded:: 2.10
    """

    fields = ("context", "body")
    context: Expr
    body: t.List[Node]


class EvalContextModifier(Stmt):
    """Modifies the eval context.  For each option that should be modified,
    a :class:`Keyword` has to be added to the :attr:`options` list.

    Example to change the `autoescape` setting::

        EvalContextModifier(options=[Keyword('autoescape', Const(True))])
    """

    fields = ("options",)
    options: t.List[Keyword]


class ScopedEvalContextModifier(EvalContextModifier):
    """Modifies the eval context and reverts it later.  Works exactly like
    :class:`EvalContextModifier` but will only modify the
    :class:`~jinja2.nodes.EvalContext` for nodes in the :attr:`body`.
    """

    fields = ("body",)
    body: t.List[Node]


# make sure nobody creates custom nodes
def _failing_new(*args: t.Any, **kwargs: t.Any) -> "te.NoReturn":
    raise TypeError("can't create custom node types")


NodeType.__new__ = staticmethod(_failing_new)  # type: ignore
del _failing_new

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\lexers\python.py ===
"""
    pygments.lexers.python
    ~~~~~~~~~~~~~~~~~~~~~~

    Lexers for Python and related languages.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import keyword

from pip._vendor.pygments.lexer import DelegatingLexer, RegexLexer, include, \
    bygroups, using, default, words, combined, this
from pip._vendor.pygments.util import get_bool_opt, shebang_matches
from pip._vendor.pygments.token import Text, Comment, Operator, Keyword, Name, String, \
    Number, Punctuation, Generic, Other, Error, Whitespace
from pip._vendor.pygments import unistring as uni

__all__ = ['PythonLexer', 'PythonConsoleLexer', 'PythonTracebackLexer',
           'Python2Lexer', 'Python2TracebackLexer',
           'CythonLexer', 'DgLexer', 'NumPyLexer']


class PythonLexer(RegexLexer):
    """
    For Python source code (version 3.x).

    .. versionchanged:: 2.5
       This is now the default ``PythonLexer``.  It is still available as the
       alias ``Python3Lexer``.
    """

    name = 'Python'
    url = 'https://www.python.org'
    aliases = ['python', 'py', 'sage', 'python3', 'py3', 'bazel', 'starlark', 'pyi']
    filenames = [
        '*.py',
        '*.pyw',
        # Type stubs
        '*.pyi',
        # Jython
        '*.jy',
        # Sage
        '*.sage',
        # SCons
        '*.sc',
        'SConstruct',
        'SConscript',
        # Skylark/Starlark (used by Bazel, Buck, and Pants)
        '*.bzl',
        'BUCK',
        'BUILD',
        'BUILD.bazel',
        'WORKSPACE',
        # Twisted Application infrastructure
        '*.tac',
    ]
    mimetypes = ['text/x-python', 'application/x-python',
                 'text/x-python3', 'application/x-python3']
    version_added = '0.10'

    uni_name = f"[{uni.xid_start}][{uni.xid_continue}]*"

    def innerstring_rules(ttype):
        return [
            # the old style '%s' % (...) string formatting (still valid in Py3)
            (r'%(\(\w+\))?[-#0 +]*([0-9]+|[*])?(\.([0-9]+|[*]))?'
             '[hlL]?[E-GXc-giorsaux%]', String.Interpol),
            # the new style '{}'.format(...) string formatting
            (r'\{'
             r'((\w+)((\.\w+)|(\[[^\]]+\]))*)?'  # field name
             r'(\![sra])?'                       # conversion
             r'(\:(.?[<>=\^])?[-+ ]?#?0?(\d+)?,?(\.\d+)?[E-GXb-gnosx%]?)?'
             r'\}', String.Interpol),

            # backslashes, quotes and formatting signs must be parsed one at a time
            (r'[^\\\'"%{\n]+', ttype),
            (r'[\'"\\]', ttype),
            # unhandled string formatting sign
            (r'%|(\{{1,2})', ttype)
            # newlines are an error (use "nl" state)
        ]

    def fstring_rules(ttype):
        return [
            # Assuming that a '}' is the closing brace after format specifier.
            # Sadly, this means that we won't detect syntax error. But it's
            # more important to parse correct syntax correctly, than to
            # highlight invalid syntax.
            (r'\}', String.Interpol),
            (r'\{', String.Interpol, 'expr-inside-fstring'),
            # backslashes, quotes and formatting signs must be parsed one at a time
            (r'[^\\\'"{}\n]+', ttype),
            (r'[\'"\\]', ttype),
            # newlines are an error (use "nl" state)
        ]

    tokens = {
        'root': [
            (r'\n', Whitespace),
            (r'^(\s*)([rRuUbB]{,2})("""(?:.|\n)*?""")',
             bygroups(Whitespace, String.Affix, String.Doc)),
            (r"^(\s*)([rRuUbB]{,2})('''(?:.|\n)*?''')",
             bygroups(Whitespace, String.Affix, String.Doc)),
            (r'\A#!.+$', Comment.Hashbang),
            (r'#.*$', Comment.Single),
            (r'\\\n', Text),
            (r'\\', Text),
            include('keywords'),
            include('soft-keywords'),
            (r'(def)((?:\s|\\\s)+)', bygroups(Keyword, Whitespace), 'funcname'),
            (r'(class)((?:\s|\\\s)+)', bygroups(Keyword, Whitespace), 'classname'),
            (r'(from)((?:\s|\\\s)+)', bygroups(Keyword.Namespace, Whitespace),
             'fromimport'),
            (r'(import)((?:\s|\\\s)+)', bygroups(Keyword.Namespace, Whitespace),
             'import'),
            include('expr'),
        ],
        'expr': [
            # raw f-strings
            ('(?i)(rf|fr)(""")',
             bygroups(String.Affix, String.Double),
             combined('rfstringescape', 'tdqf')),
            ("(?i)(rf|fr)(''')",
             bygroups(String.Affix, String.Single),
             combined('rfstringescape', 'tsqf')),
            ('(?i)(rf|fr)(")',
             bygroups(String.Affix, String.Double),
             combined('rfstringescape', 'dqf')),
            ("(?i)(rf|fr)(')",
             bygroups(String.Affix, String.Single),
             combined('rfstringescape', 'sqf')),
            # non-raw f-strings
            ('([fF])(""")', bygroups(String.Affix, String.Double),
             combined('fstringescape', 'tdqf')),
            ("([fF])(''')", bygroups(String.Affix, String.Single),
             combined('fstringescape', 'tsqf')),
            ('([fF])(")', bygroups(String.Affix, String.Double),
             combined('fstringescape', 'dqf')),
            ("([fF])(')", bygroups(String.Affix, String.Single),
             combined('fstringescape', 'sqf')),
            # raw bytes and strings
            ('(?i)(rb|br|r)(""")',
             bygroups(String.Affix, String.Double), 'tdqs'),
            ("(?i)(rb|br|r)(''')",
             bygroups(String.Affix, String.Single), 'tsqs'),
            ('(?i)(rb|br|r)(")',
             bygroups(String.Affix, String.Double), 'dqs'),
            ("(?i)(rb|br|r)(')",
             bygroups(String.Affix, String.Single), 'sqs'),
            # non-raw strings
            ('([uU]?)(""")', bygroups(String.Affix, String.Double),
             combined('stringescape', 'tdqs')),
            ("([uU]?)(''')", bygroups(String.Affix, String.Single),
             combined('stringescape', 'tsqs')),
            ('([uU]?)(")', bygroups(String.Affix, String.Double),
             combined('stringescape', 'dqs')),
            ("([uU]?)(')", bygroups(String.Affix, String.Single),
             combined('stringescape', 'sqs')),
            # non-raw bytes
            ('([bB])(""")', bygroups(String.Affix, String.Double),
             combined('bytesescape', 'tdqs')),
            ("([bB])(''')", bygroups(String.Affix, String.Single),
             combined('bytesescape', 'tsqs')),
            ('([bB])(")', bygroups(String.Affix, String.Double),
             combined('bytesescape', 'dqs')),
            ("([bB])(')", bygroups(String.Affix, String.Single),
             combined('bytesescape', 'sqs')),

            (r'[^\S\n]+', Text),
            include('numbers'),
            (r'!=|==|<<|>>|:=|[-~+/*%=<>&^|.]', Operator),
            (r'[]{}:(),;[]', Punctuation),
            (r'(in|is|and|or|not)\b', Operator.Word),
            include('expr-keywords'),
            include('builtins'),
            include('magicfuncs'),
            include('magicvars'),
            include('name'),
        ],
        'expr-inside-fstring': [
            (r'[{([]', Punctuation, 'expr-inside-fstring-inner'),
            # without format specifier
            (r'(=\s*)?'         # debug (https://bugs.python.org/issue36817)
             r'(\![sraf])?'     # conversion
             r'\}', String.Interpol, '#pop'),
            # with format specifier
            # we'll catch the remaining '}' in the outer scope
            (r'(=\s*)?'         # debug (https://bugs.python.org/issue36817)
             r'(\![sraf])?'     # conversion
             r':', String.Interpol, '#pop'),
            (r'\s+', Whitespace),  # allow new lines
            include('expr'),
        ],
        'expr-inside-fstring-inner': [
            (r'[{([]', Punctuation, 'expr-inside-fstring-inner'),
            (r'[])}]', Punctuation, '#pop'),
            (r'\s+', Whitespace),  # allow new lines
            include('expr'),
        ],
        'expr-keywords': [
            # Based on https://docs.python.org/3/reference/expressions.html
            (words((
                'async for', 'await', 'else', 'for', 'if', 'lambda',
                'yield', 'yield from'), suffix=r'\b'),
             Keyword),
            (words(('True', 'False', 'None'), suffix=r'\b'), Keyword.Constant),
        ],
        'keywords': [
            (words((
                'assert', 'async', 'await', 'break', 'continue', 'del', 'elif',
                'else', 'except', 'finally', 'for', 'global', 'if', 'lambda',
                'pass', 'raise', 'nonlocal', 'return', 'try', 'while', 'yield',
                'yield from', 'as', 'with'), suffix=r'\b'),
             Keyword),
            (words(('True', 'False', 'None'), suffix=r'\b'), Keyword.Constant),
        ],
        'soft-keywords': [
            # `match`, `case` and `_` soft keywords
            (r'(^[ \t]*)'              # at beginning of line + possible indentation
             r'(match|case)\b'         # a possible keyword
             r'(?![ \t]*(?:'           # not followed by...
             r'[:,;=^&|@~)\]}]|(?:' +  # characters and keywords that mean this isn't
                                       # pattern matching (but None/True/False is ok)
             r'|'.join(k for k in keyword.kwlist if k[0].islower()) + r')\b))',
             bygroups(Text, Keyword), 'soft-keywords-inner'),
        ],
        'soft-keywords-inner': [
            # optional `_` keyword
            (r'(\s+)([^\n_]*)(_\b)', bygroups(Whitespace, using(this), Keyword)),
            default('#pop')
        ],
        'builtins': [
            (words((
                '__import__', 'abs', 'aiter', 'all', 'any', 'bin', 'bool', 'bytearray',
                'breakpoint', 'bytes', 'callable', 'chr', 'classmethod', 'compile',
                'complex', 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval',
                'filter', 'float', 'format', 'frozenset', 'getattr', 'globals',
                'hasattr', 'hash', 'hex', 'id', 'input', 'int', 'isinstance',
                'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
                'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow',
                'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
                'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super',
                'tuple', 'type', 'vars', 'zip'), prefix=r'(?<!\.)', suffix=r'\b'),
             Name.Builtin),
            (r'(?<!\.)(self|Ellipsis|NotImplemented|cls)\b', Name.Builtin.Pseudo),
            (words((
                'ArithmeticError', 'AssertionError', 'AttributeError',
                'BaseException', 'BufferError', 'BytesWarning', 'DeprecationWarning',
                'EOFError', 'EnvironmentError', 'Exception', 'FloatingPointError',
                'FutureWarning', 'GeneratorExit', 'IOError', 'ImportError',
                'ImportWarning', 'IndentationError', 'IndexError', 'KeyError',
                'KeyboardInterrupt', 'LookupError', 'MemoryError', 'NameError',
                'NotImplementedError', 'OSError', 'OverflowError',
                'PendingDeprecationWarning', 'ReferenceError', 'ResourceWarning',
                'RuntimeError', 'RuntimeWarning', 'StopIteration',
                'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit',
                'TabError', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError',
                'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError',
                'UnicodeWarning', 'UserWarning', 'ValueError', 'VMSError',
                'Warning', 'WindowsError', 'ZeroDivisionError',
                # new builtin exceptions from PEP 3151
                'BlockingIOError', 'ChildProcessError', 'ConnectionError',
                'BrokenPipeError', 'ConnectionAbortedError', 'ConnectionRefusedError',
                'ConnectionResetError', 'FileExistsError', 'FileNotFoundError',
                'InterruptedError', 'IsADirectoryError', 'NotADirectoryError',
                'PermissionError', 'ProcessLookupError', 'TimeoutError',
                # others new in Python 3
                'StopAsyncIteration', 'ModuleNotFoundError', 'RecursionError',
                'EncodingWarning'),
                prefix=r'(?<!\.)', suffix=r'\b'),
             Name.Exception),
        ],
        'magicfuncs': [
            (words((
                '__abs__', '__add__', '__aenter__', '__aexit__', '__aiter__',
                '__and__', '__anext__', '__await__', '__bool__', '__bytes__',
                '__call__', '__complex__', '__contains__', '__del__', '__delattr__',
                '__delete__', '__delitem__', '__dir__', '__divmod__', '__enter__',
                '__eq__', '__exit__', '__float__', '__floordiv__', '__format__',
                '__ge__', '__get__', '__getattr__', '__getattribute__',
                '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__',
                '__ifloordiv__', '__ilshift__', '__imatmul__', '__imod__',
                '__imul__', '__index__', '__init__', '__instancecheck__',
                '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__',
                '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__',
                '__len__', '__length_hint__', '__lshift__', '__lt__', '__matmul__',
                '__missing__', '__mod__', '__mul__', '__ne__', '__neg__',
                '__new__', '__next__', '__or__', '__pos__', '__pow__',
                '__prepare__', '__radd__', '__rand__', '__rdivmod__', '__repr__',
                '__reversed__', '__rfloordiv__', '__rlshift__', '__rmatmul__',
                '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__',
                '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__',
                '__rxor__', '__set__', '__setattr__', '__setitem__', '__str__',
                '__sub__', '__subclasscheck__', '__truediv__',
                '__xor__'), suffix=r'\b'),
             Name.Function.Magic),
        ],
        'magicvars': [
            (words((
                '__annotations__', '__bases__', '__class__', '__closure__',
                '__code__', '__defaults__', '__dict__', '__doc__', '__file__',
                '__func__', '__globals__', '__kwdefaults__', '__module__',
                '__mro__', '__name__', '__objclass__', '__qualname__',
                '__self__', '__slots__', '__weakref__'), suffix=r'\b'),
             Name.Variable.Magic),
        ],
        'numbers': [
            (r'(\d(?:_?\d)*\.(?:\d(?:_?\d)*)?|(?:\d(?:_?\d)*)?\.\d(?:_?\d)*)'
             r'([eE][+-]?\d(?:_?\d)*)?', Number.Float),
            (r'\d(?:_?\d)*[eE][+-]?\d(?:_?\d)*j?', Number.Float),
            (r'0[oO](?:_?[0-7])+', Number.Oct),
            (r'0[bB](?:_?[01])+', Number.Bin),
            (r'0[xX](?:_?[a-fA-F0-9])+', Number.Hex),
            (r'\d(?:_?\d)*', Number.Integer),
        ],
        'name': [
            (r'@' + uni_name, Name.Decorator),
            (r'@', Operator),  # new matrix multiplication operator
            (uni_name, Name),
        ],
        'funcname': [
            include('magicfuncs'),
            (uni_name, Name.Function, '#pop'),
            default('#pop'),
        ],
        'classname': [
            (uni_name, Name.Class, '#pop'),
        ],
        'import': [
            (r'(\s+)(as)(\s+)', bygroups(Whitespace, Keyword, Whitespace)),
            (r'\.', Name.Namespace),
            (uni_name, Name.Namespace),
            (r'(\s*)(,)(\s*)', bygroups(Whitespace, Operator, Whitespace)),
            default('#pop')  # all else: go back
        ],
        'fromimport': [
            (r'(\s+)(import)\b', bygroups(Whitespace, Keyword.Namespace), '#pop'),
            (r'\.', Name.Namespace),
            # if None occurs here, it's "raise x from None", since None can
            # never be a module name
            (r'None\b', Keyword.Constant, '#pop'),
            (uni_name, Name.Namespace),
            default('#pop'),
        ],
        'rfstringescape': [
            (r'\{\{', String.Escape),
            (r'\}\}', String.Escape),
        ],
        'fstringescape': [
            include('rfstringescape'),
            include('stringescape'),
        ],
        'bytesescape': [
            (r'\\([\\abfnrtv"\']|\n|x[a-fA-F0-9]{2}|[0-7]{1,3})', String.Escape)
        ],
        'stringescape': [
            (r'\\(N\{.*?\}|u[a-fA-F0-9]{4}|U[a-fA-F0-9]{8})', String.Escape),
            include('bytesescape')
        ],
        'fstrings-single': fstring_rules(String.Single),
        'fstrings-double': fstring_rules(String.Double),
        'strings-single': innerstring_rules(String.Single),
        'strings-double': innerstring_rules(String.Double),
        'dqf': [
            (r'"', String.Double, '#pop'),
            (r'\\\\|\\"|\\\n', String.Escape),  # included here for raw strings
            include('fstrings-double')
        ],
        'sqf': [
            (r"'", String.Single, '#pop'),
            (r"\\\\|\\'|\\\n", String.Escape),  # included here for raw strings
            include('fstrings-single')
        ],
        'dqs': [
            (r'"', String.Double, '#pop'),
            (r'\\\\|\\"|\\\n', String.Escape),  # included here for raw strings
            include('strings-double')
        ],
        'sqs': [
            (r"'", String.Single, '#pop'),
            (r"\\\\|\\'|\\\n", String.Escape),  # included here for raw strings
            include('strings-single')
        ],
        'tdqf': [
            (r'"""', String.Double, '#pop'),
            include('fstrings-double'),
            (r'\n', String.Double)
        ],
        'tsqf': [
            (r"'''", String.Single, '#pop'),
            include('fstrings-single'),
            (r'\n', String.Single)
        ],
        'tdqs': [
            (r'"""', String.Double, '#pop'),
            include('strings-double'),
            (r'\n', String.Double)
        ],
        'tsqs': [
            (r"'''", String.Single, '#pop'),
            include('strings-single'),
            (r'\n', String.Single)
        ],
    }

    def analyse_text(text):
        return shebang_matches(text, r'pythonw?(3(\.\d)?)?') or \
            'import ' in text[:1000]


Python3Lexer = PythonLexer


class Python2Lexer(RegexLexer):
    """
    For Python 2.x source code.

    .. versionchanged:: 2.5
       This class has been renamed from ``PythonLexer``.  ``PythonLexer`` now
       refers to the Python 3 variant.  File name patterns like ``*.py`` have
       been moved to Python 3 as well.
    """

    name = 'Python 2.x'
    url = 'https://www.python.org'
    aliases = ['python2', 'py2']
    filenames = []  # now taken over by PythonLexer (3.x)
    mimetypes = ['text/x-python2', 'application/x-python2']
    version_added = ''

    def innerstring_rules(ttype):
        return [
            # the old style '%s' % (...) string formatting
            (r'%(\(\w+\))?[-#0 +]*([0-9]+|[*])?(\.([0-9]+|[*]))?'
             '[hlL]?[E-GXc-giorsux%]', String.Interpol),
            # backslashes, quotes and formatting signs must be parsed one at a time
            (r'[^\\\'"%\n]+', ttype),
            (r'[\'"\\]', ttype),
            # unhandled string formatting sign
            (r'%', ttype),
            # newlines are an error (use "nl" state)
        ]

    tokens = {
        'root': [
            (r'\n', Whitespace),
            (r'^(\s*)([rRuUbB]{,2})("""(?:.|\n)*?""")',
             bygroups(Whitespace, String.Affix, String.Doc)),
            (r"^(\s*)([rRuUbB]{,2})('''(?:.|\n)*?''')",
             bygroups(Whitespace, String.Affix, String.Doc)),
            (r'[^\S\n]+', Text),
            (r'\A#!.+$', Comment.Hashbang),
            (r'#.*$', Comment.Single),
            (r'[]{}:(),;[]', Punctuation),
            (r'\\\n', Text),
            (r'\\', Text),
            (r'(in|is|and|or|not)\b', Operator.Word),
            (r'!=|==|<<|>>|[-~+/*%=<>&^|.]', Operator),
            include('keywords'),
            (r'(def)((?:\s|\\\s)+)', bygroups(Keyword, Whitespace), 'funcname'),
            (r'(class)((?:\s|\\\s)+)', bygroups(Keyword, Whitespace), 'classname'),
            (r'(from)((?:\s|\\\s)+)', bygroups(Keyword.Namespace, Whitespace),
             'fromimport'),
            (r'(import)((?:\s|\\\s)+)', bygroups(Keyword.Namespace, Whitespace),
             'import'),
            include('builtins'),
            include('magicfuncs'),
            include('magicvars'),
            include('backtick'),
            ('([rR]|[uUbB][rR]|[rR][uUbB])(""")',
             bygroups(String.Affix, String.Double), 'tdqs'),
            ("([rR]|[uUbB][rR]|[rR][uUbB])(''')",
             bygroups(String.Affix, String.Single), 'tsqs'),
            ('([rR]|[uUbB][rR]|[rR][uUbB])(")',
             bygroups(String.Affix, String.Double), 'dqs'),
            ("([rR]|[uUbB][rR]|[rR][uUbB])(')",
             bygroups(String.Affix, String.Single), 'sqs'),
            ('([uUbB]?)(""")', bygroups(String.Affix, String.Double),
             combined('stringescape', 'tdqs')),
            ("([uUbB]?)(''')", bygroups(String.Affix, String.Single),
             combined('stringescape', 'tsqs')),
            ('([uUbB]?)(")', bygroups(String.Affix, String.Double),
             combined('stringescape', 'dqs')),
            ("([uUbB]?)(')", bygroups(String.Affix, String.Single),
             combined('stringescape', 'sqs')),
            include('name'),
            include('numbers'),
        ],
        'keywords': [
            (words((
                'assert', 'break', 'continue', 'del', 'elif', 'else', 'except',
                'exec', 'finally', 'for', 'global', 'if', 'lambda', 'pass',
                'print', 'raise', 'return', 'try', 'while', 'yield',
                'yield from', 'as', 'with'), suffix=r'\b'),
             Keyword),
        ],
        'builtins': [
            (words((
                '__import__', 'abs', 'all', 'any', 'apply', 'basestring', 'bin',
                'bool', 'buffer', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod',
                'cmp', 'coerce', 'compile', 'complex', 'delattr', 'dict', 'dir', 'divmod',
                'enumerate', 'eval', 'execfile', 'exit', 'file', 'filter', 'float',
                'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'hex', 'id',
                'input', 'int', 'intern', 'isinstance', 'issubclass', 'iter', 'len',
                'list', 'locals', 'long', 'map', 'max', 'min', 'next', 'object',
                'oct', 'open', 'ord', 'pow', 'property', 'range', 'raw_input', 'reduce',
                'reload', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice',
                'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type',
                'unichr', 'unicode', 'vars', 'xrange', 'zip'),
                prefix=r'(?<!\.)', suffix=r'\b'),
             Name.Builtin),
            (r'(?<!\.)(self|None|Ellipsis|NotImplemented|False|True|cls'
             r')\b', Name.Builtin.Pseudo),
            (words((
                'ArithmeticError', 'AssertionError', 'AttributeError',
                'BaseException', 'DeprecationWarning', 'EOFError', 'EnvironmentError',
                'Exception', 'FloatingPointError', 'FutureWarning', 'GeneratorExit',
                'IOError', 'ImportError', 'ImportWarning', 'IndentationError',
                'IndexError', 'KeyError', 'KeyboardInterrupt', 'LookupError',
                'MemoryError', 'NameError',
                'NotImplementedError', 'OSError', 'OverflowError', 'OverflowWarning',
                'PendingDeprecationWarning', 'ReferenceError',
                'RuntimeError', 'RuntimeWarning', 'StandardError', 'StopIteration',
                'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit',
                'TabError', 'TypeError', 'UnboundLocalError', 'UnicodeDecodeError',
                'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError',
                'UnicodeWarning', 'UserWarning', 'ValueError', 'VMSError', 'Warning',
                'WindowsError', 'ZeroDivisionError'), prefix=r'(?<!\.)', suffix=r'\b'),
             Name.Exception),
        ],
        'magicfuncs': [
            (words((
                '__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__',
                '__complex__', '__contains__', '__del__', '__delattr__', '__delete__',
                '__delitem__', '__delslice__', '__div__', '__divmod__', '__enter__',
                '__eq__', '__exit__', '__float__', '__floordiv__', '__ge__', '__get__',
                '__getattr__', '__getattribute__', '__getitem__', '__getslice__', '__gt__',
                '__hash__', '__hex__', '__iadd__', '__iand__', '__idiv__', '__ifloordiv__',
                '__ilshift__', '__imod__', '__imul__', '__index__', '__init__',
                '__instancecheck__', '__int__', '__invert__', '__iop__', '__ior__',
                '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__',
                '__ixor__', '__le__', '__len__', '__long__', '__lshift__', '__lt__',
                '__missing__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__',
                '__nonzero__', '__oct__', '__op__', '__or__', '__pos__', '__pow__',
                '__radd__', '__rand__', '__rcmp__', '__rdiv__', '__rdivmod__', '__repr__',
                '__reversed__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__',
                '__rop__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__',
                '__rtruediv__', '__rxor__', '__set__', '__setattr__', '__setitem__',
                '__setslice__', '__str__', '__sub__', '__subclasscheck__', '__truediv__',
                '__unicode__', '__xor__'), suffix=r'\b'),
             Name.Function.Magic),
        ],
        'magicvars': [
            (words((
                '__bases__', '__class__', '__closure__', '__code__', '__defaults__',
                '__dict__', '__doc__', '__file__', '__func__', '__globals__',
                '__metaclass__', '__module__', '__mro__', '__name__', '__self__',
                '__slots__', '__weakref__'),
                suffix=r'\b'),
             Name.Variable.Magic),
        ],
        'numbers': [
            (r'(\d+\.\d*|\d*\.\d+)([eE][+-]?[0-9]+)?j?', Number.Float),
            (r'\d+[eE][+-]?[0-9]+j?', Number.Float),
            (r'0[0-7]+j?', Number.Oct),
            (r'0[bB][01]+', Number.Bin),
            (r'0[xX][a-fA-F0-9]+', Number.Hex),
            (r'\d+L', Number.Integer.Long),
            (r'\d+j?', Number.Integer)
        ],
        'backtick': [
            ('`.*?`', String.Backtick),
        ],
        'name': [
            (r'@[\w.]+', Name.Decorator),
            (r'[a-zA-Z_]\w*', Name),
        ],
        'funcname': [
            include('magicfuncs'),
            (r'[a-zA-Z_]\w*', Name.Function, '#pop'),
            default('#pop'),
        ],
        'classname': [
            (r'[a-zA-Z_]\w*', Name.Class, '#pop')
        ],
        'import': [
            (r'(?:[ \t]|\\\n)+', Text),
            (r'as\b', Keyword.Namespace),
            (r',', Operator),
            (r'[a-zA-Z_][\w.]*', Name.Namespace),
            default('#pop')  # all else: go back
        ],
        'fromimport': [
            (r'(?:[ \t]|\\\n)+', Text),
            (r'import\b', Keyword.Namespace, '#pop'),
            # if None occurs here, it's "raise x from None", since None can
            # never be a module name
            (r'None\b', Name.Builtin.Pseudo, '#pop'),
            # sadly, in "raise x from y" y will be highlighted as namespace too
            (r'[a-zA-Z_.][\w.]*', Name.Namespace),
            # anything else here also means "raise x from y" and is therefore
            # not an error
            default('#pop'),
        ],
        'stringescape': [
            (r'\\([\\abfnrtv"\']|\n|N\{.*?\}|u[a-fA-F0-9]{4}|'
             r'U[a-fA-F0-9]{8}|x[a-fA-F0-9]{2}|[0-7]{1,3})', String.Escape)
        ],
        'strings-single': innerstring_rules(String.Single),
        'strings-double': innerstring_rules(String.Double),
        'dqs': [
            (r'"', String.Double, '#pop'),
            (r'\\\\|\\"|\\\n', String.Escape),  # included here for raw strings
            include('strings-double')
        ],
        'sqs': [
            (r"'", String.Single, '#pop'),
            (r"\\\\|\\'|\\\n", String.Escape),  # included here for raw strings
            include('strings-single')
        ],
        'tdqs': [
            (r'"""', String.Double, '#pop'),
            include('strings-double'),
            (r'\n', String.Double)
        ],
        'tsqs': [
            (r"'''", String.Single, '#pop'),
            include('strings-single'),
            (r'\n', String.Single)
        ],
    }

    def analyse_text(text):
        return shebang_matches(text, r'pythonw?2(\.\d)?')


class _PythonConsoleLexerBase(RegexLexer):
    name = 'Python console session'
    aliases = ['pycon', 'python-console']
    mimetypes = ['text/x-python-doctest']

    """Auxiliary lexer for `PythonConsoleLexer`.

    Code tokens are output as ``Token.Other.Code``, traceback tokens as
    ``Token.Other.Traceback``.
    """
    tokens = {
        'root': [
            (r'(>>> )(.*\n)', bygroups(Generic.Prompt, Other.Code), 'continuations'),
            # This happens, e.g., when tracebacks are embedded in documentation;
            # trailing whitespaces are often stripped in such contexts.
            (r'(>>>)(\n)', bygroups(Generic.Prompt, Whitespace)),
            (r'(\^C)?Traceback \(most recent call last\):\n', Other.Traceback, 'traceback'),
            # SyntaxError starts with this
            (r'  File "[^"]+", line \d+', Other.Traceback, 'traceback'),
            (r'.*\n', Generic.Output),
        ],
        'continuations': [
            (r'(\.\.\. )(.*\n)', bygroups(Generic.Prompt, Other.Code)),
            # See above.
            (r'(\.\.\.)(\n)', bygroups(Generic.Prompt, Whitespace)),
            default('#pop'),
        ],
        'traceback': [
            # As soon as we see a traceback, consume everything until the next
            # >>> prompt.
            (r'(?=>>>( |$))', Text, '#pop'),
            (r'(KeyboardInterrupt)(\n)', bygroups(Name.Class, Whitespace)),
            (r'.*\n', Other.Traceback),
        ],
    }


class PythonConsoleLexer(DelegatingLexer):
    """
    For Python console output or doctests, such as:

    .. sourcecode:: pycon

        >>> a = 'foo'
        >>> print(a)
        foo
        >>> 1 / 0
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        ZeroDivisionError: integer division or modulo by zero

    Additional options:

    `python3`
        Use Python 3 lexer for code.  Default is ``True``.

        .. versionadded:: 1.0
        .. versionchanged:: 2.5
           Now defaults to ``True``.
    """

    name = 'Python console session'
    aliases = ['pycon', 'python-console']
    mimetypes = ['text/x-python-doctest']
    url = 'https://python.org'
    version_added = ''

    def __init__(self, **options):
        python3 = get_bool_opt(options, 'python3', True)
        if python3:
            pylexer = PythonLexer
            tblexer = PythonTracebackLexer
        else:
            pylexer = Python2Lexer
            tblexer = Python2TracebackLexer
        # We have two auxiliary lexers. Use DelegatingLexer twice with
        # different tokens.  TODO: DelegatingLexer should support this
        # directly, by accepting a tuplet of auxiliary lexers and a tuple of
        # distinguishing tokens. Then we wouldn't need this intermediary
        # class.
        class _ReplaceInnerCode(DelegatingLexer):
            def __init__(self, **options):
                super().__init__(pylexer, _PythonConsoleLexerBase, Other.Code, **options)
        super().__init__(tblexer, _ReplaceInnerCode, Other.Traceback, **options)


class PythonTracebackLexer(RegexLexer):
    """
    For Python 3.x tracebacks, with support for chained exceptions.

    .. versionchanged:: 2.5
       This is now the default ``PythonTracebackLexer``.  It is still available
       as the alias ``Python3TracebackLexer``.
    """

    name = 'Python Traceback'
    aliases = ['pytb', 'py3tb']
    filenames = ['*.pytb', '*.py3tb']
    mimetypes = ['text/x-python-traceback', 'text/x-python3-traceback']
    url = 'https://python.org'
    version_added = '1.0'

    tokens = {
        'root': [
            (r'\n', Whitespace),
            (r'^(\^C)?Traceback \(most recent call last\):\n', Generic.Traceback, 'intb'),
            (r'^During handling of the above exception, another '
             r'exception occurred:\n\n', Generic.Traceback),
            (r'^The above exception was the direct cause of the '
             r'following exception:\n\n', Generic.Traceback),
            (r'^(?=  File "[^"]+", line \d+)', Generic.Traceback, 'intb'),
            (r'^.*\n', Other),
        ],
        'intb': [
            (r'^(  File )("[^"]+")(, line )(\d+)(, in )(.+)(\n)',
             bygroups(Text, Name.Builtin, Text, Number, Text, Name, Whitespace)),
            (r'^(  File )("[^"]+")(, line )(\d+)(\n)',
             bygroups(Text, Name.Builtin, Text, Number, Whitespace)),
            (r'^(    )(.+)(\n)',
             bygroups(Whitespace, using(PythonLexer), Whitespace), 'markers'),
            (r'^([ \t]*)(\.\.\.)(\n)',
             bygroups(Whitespace, Comment, Whitespace)),  # for doctests...
            (r'^([^:]+)(: )(.+)(\n)',
             bygroups(Generic.Error, Text, Name, Whitespace), '#pop'),
            (r'^([a-zA-Z_][\w.]*)(:?\n)',
             bygroups(Generic.Error, Whitespace), '#pop'),
            default('#pop'),
        ],
        'markers': [
            # Either `PEP 657 <https://www.python.org/dev/peps/pep-0657/>`
            # error locations in Python 3.11+, or single-caret markers
            # for syntax errors before that.
            (r'^( {4,})([~^]+)(\n)',
             bygroups(Whitespace, Punctuation.Marker, Whitespace),
             '#pop'),
            default('#pop'),
        ],
    }


Python3TracebackLexer = PythonTracebackLexer


class Python2TracebackLexer(RegexLexer):
    """
    For Python tracebacks.

    .. versionchanged:: 2.5
       This class has been renamed from ``PythonTracebackLexer``.
       ``PythonTracebackLexer`` now refers to the Python 3 variant.
    """

    name = 'Python 2.x Traceback'
    aliases = ['py2tb']
    filenames = ['*.py2tb']
    mimetypes = ['text/x-python2-traceback']
    url = 'https://python.org'
    version_added = '0.7'

    tokens = {
        'root': [
            # Cover both (most recent call last) and (innermost last)
            # The optional ^C allows us to catch keyboard interrupt signals.
            (r'^(\^C)?(Traceback.*\n)',
             bygroups(Text, Generic.Traceback), 'intb'),
            # SyntaxError starts with this.
            (r'^(?=  File "[^"]+", line \d+)', Generic.Traceback, 'intb'),
            (r'^.*\n', Other),
        ],
        'intb': [
            (r'^(  File )("[^"]+")(, line )(\d+)(, in )(.+)(\n)',
             bygroups(Text, Name.Builtin, Text, Number, Text, Name, Whitespace)),
            (r'^(  File )("[^"]+")(, line )(\d+)(\n)',
             bygroups(Text, Name.Builtin, Text, Number, Whitespace)),
            (r'^(    )(.+)(\n)',
             bygroups(Text, using(Python2Lexer), Whitespace), 'marker'),
            (r'^([ \t]*)(\.\.\.)(\n)',
             bygroups(Text, Comment, Whitespace)),  # for doctests...
            (r'^([^:]+)(: )(.+)(\n)',
             bygroups(Generic.Error, Text, Name, Whitespace), '#pop'),
            (r'^([a-zA-Z_]\w*)(:?\n)',
             bygroups(Generic.Error, Whitespace), '#pop')
        ],
        'marker': [
            # For syntax errors.
            (r'( {4,})(\^)', bygroups(Text, Punctuation.Marker), '#pop'),
            default('#pop'),
        ],
    }


class CythonLexer(RegexLexer):
    """
    For Pyrex and Cython source code.
    """

    name = 'Cython'
    url = 'https://cython.org'
    aliases = ['cython', 'pyx', 'pyrex']
    filenames = ['*.pyx', '*.pxd', '*.pxi']
    mimetypes = ['text/x-cython', 'application/x-cython']
    version_added = '1.1'

    tokens = {
        'root': [
            (r'\n', Whitespace),
            (r'^(\s*)("""(?:.|\n)*?""")', bygroups(Whitespace, String.Doc)),
            (r"^(\s*)('''(?:.|\n)*?''')", bygroups(Whitespace, String.Doc)),
            (r'[^\S\n]+', Text),
            (r'#.*$', Comment),
            (r'[]{}:(),;[]', Punctuation),
            (r'\\\n', Whitespace),
            (r'\\', Text),
            (r'(in|is|and|or|not)\b', Operator.Word),
            (r'(<)([a-zA-Z0-9.?]+)(>)',
             bygroups(Punctuation, Keyword.Type, Punctuation)),
            (r'!=|==|<<|>>|[-~+/*%=<>&^|.?]', Operator),
            (r'(from)(\d+)(<=)(\s+)(<)(\d+)(:)',
             bygroups(Keyword, Number.Integer, Operator, Whitespace, Operator,
                      Name, Punctuation)),
            include('keywords'),
            (r'(def|property)(\s+)', bygroups(Keyword, Whitespace), 'funcname'),
            (r'(cp?def)(\s+)', bygroups(Keyword, Whitespace), 'cdef'),
            # (should actually start a block with only cdefs)
            (r'(cdef)(:)', bygroups(Keyword, Punctuation)),
            (r'(class|struct)(\s+)', bygroups(Keyword, Whitespace), 'classname'),
            (r'(from)(\s+)', bygroups(Keyword, Whitespace), 'fromimport'),
            (r'(c?import)(\s+)', bygroups(Keyword, Whitespace), 'import'),
            include('builtins'),
            include('backtick'),
            ('(?:[rR]|[uU][rR]|[rR][uU])"""', String, 'tdqs'),
            ("(?:[rR]|[uU][rR]|[rR][uU])'''", String, 'tsqs'),
            ('(?:[rR]|[uU][rR]|[rR][uU])"', String, 'dqs'),
            ("(?:[rR]|[uU][rR]|[rR][uU])'", String, 'sqs'),
            ('[uU]?"""', String, combined('stringescape', 'tdqs')),
            ("[uU]?'''", String, combined('stringescape', 'tsqs')),
            ('[uU]?"', String, combined('stringescape', 'dqs')),
            ("[uU]?'", String, combined('stringescape', 'sqs')),
            include('name'),
            include('numbers'),
        ],
        'keywords': [
            (words((
                'assert', 'async', 'await', 'break', 'by', 'continue', 'ctypedef', 'del', 'elif',
                'else', 'except', 'except?', 'exec', 'finally', 'for', 'fused', 'gil',
                'global', 'if', 'include', 'lambda', 'nogil', 'pass', 'print',
                'raise', 'return', 'try', 'while', 'yield', 'as', 'with'), suffix=r'\b'),
             Keyword),
            (r'(DEF|IF|ELIF|ELSE)\b', Comment.Preproc),
        ],
        'builtins': [
            (words((
                '__import__', 'abs', 'all', 'any', 'apply', 'basestring', 'bin', 'bint',
                'bool', 'buffer', 'bytearray', 'bytes', 'callable', 'chr',
                'classmethod', 'cmp', 'coerce', 'compile', 'complex', 'delattr',
                'dict', 'dir', 'divmod', 'enumerate', 'eval', 'execfile', 'exit',
                'file', 'filter', 'float', 'frozenset', 'getattr', 'globals',
                'hasattr', 'hash', 'hex', 'id', 'input', 'int', 'intern', 'isinstance',
                'issubclass', 'iter', 'len', 'list', 'locals', 'long', 'map', 'max',
                'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'property', 'Py_ssize_t',
                'range', 'raw_input', 'reduce', 'reload', 'repr', 'reversed',
                'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod',
                'str', 'sum', 'super', 'tuple', 'type', 'unichr', 'unicode', 'unsigned',
                'vars', 'xrange', 'zip'), prefix=r'(?<!\.)', suffix=r'\b'),
             Name.Builtin),
            (r'(?<!\.)(self|None|Ellipsis|NotImplemented|False|True|NULL'
             r')\b', Name.Builtin.Pseudo),
            (words((
                'ArithmeticError', 'AssertionError', 'AttributeError',
                'BaseException', 'DeprecationWarning', 'EOFError', 'EnvironmentError',
                'Exception', 'FloatingPointError', 'FutureWarning', 'GeneratorExit',
                'IOError', 'ImportError', 'ImportWarning', 'IndentationError',
                'IndexError', 'KeyError', 'KeyboardInterrupt', 'LookupError',
                'MemoryError', 'NameError', 'NotImplemented', 'NotImplementedError',
                'OSError', 'OverflowError', 'OverflowWarning',
                'PendingDeprecationWarning', 'ReferenceError', 'RuntimeError',
                'RuntimeWarning', 'StandardError', 'StopIteration', 'SyntaxError',
                'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError',
                'TypeError', 'UnboundLocalError', 'UnicodeDecodeError',
                'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError',
                'UnicodeWarning', 'UserWarning', 'ValueError', 'Warning',
                'ZeroDivisionError'), prefix=r'(?<!\.)', suffix=r'\b'),
             Name.Exception),
        ],
        'numbers': [
            (r'(\d+\.?\d*|\d*\.\d+)([eE][+-]?[0-9]+)?', Number.Float),
            (r'0\d+', Number.Oct),
            (r'0[xX][a-fA-F0-9]+', Number.Hex),
            (r'\d+L', Number.Integer.Long),
            (r'\d+', Number.Integer)
        ],
        'backtick': [
            ('`.*?`', String.Backtick),
        ],
        'name': [
            (r'@\w+', Name.Decorator),
            (r'[a-zA-Z_]\w*', Name),
        ],
        'funcname': [
            (r'[a-zA-Z_]\w*', Name.Function, '#pop')
        ],
        'cdef': [
            (r'(public|readonly|extern|api|inline)\b', Keyword.Reserved),
            (r'(struct|enum|union|class)\b', Keyword),
            (r'([a-zA-Z_]\w*)(\s*)(?=[(:#=]|$)',
             bygroups(Name.Function, Whitespace), '#pop'),
            (r'([a-zA-Z_]\w*)(\s*)(,)',
             bygroups(Name.Function, Whitespace, Punctuation)),
            (r'from\b', Keyword, '#pop'),
            (r'as\b', Keyword),
            (r':', Punctuation, '#pop'),
            (r'(?=["\'])', Text, '#pop'),
            (r'[a-zA-Z_]\w*', Keyword.Type),
            (r'.', Text),
        ],
        'classname': [
            (r'[a-zA-Z_]\w*', Name.Class, '#pop')
        ],
        'import': [
            (r'(\s+)(as)(\s+)', bygroups(Whitespace, Keyword, Whitespace)),
            (r'[a-zA-Z_][\w.]*', Name.Namespace),
            (r'(\s*)(,)(\s*)', bygroups(Whitespace, Operator, Whitespace)),
            default('#pop')  # all else: go back
        ],
        'fromimport': [
            (r'(\s+)(c?import)\b', bygroups(Whitespace, Keyword), '#pop'),
            (r'[a-zA-Z_.][\w.]*', Name.Namespace),
            # ``cdef foo from "header"``, or ``for foo from 0 < i < 10``
            default('#pop'),
        ],
        'stringescape': [
            (r'\\([\\abfnrtv"\']|\n|N\{.*?\}|u[a-fA-F0-9]{4}|'
             r'U[a-fA-F0-9]{8}|x[a-fA-F0-9]{2}|[0-7]{1,3})', String.Escape)
        ],
        'strings': [
            (r'%(\([a-zA-Z0-9]+\))?[-#0 +]*([0-9]+|[*])?(\.([0-9]+|[*]))?'
             '[hlL]?[E-GXc-giorsux%]', String.Interpol),
            (r'[^\\\'"%\n]+', String),
            # quotes, percents and backslashes must be parsed one at a time
            (r'[\'"\\]', String),
            # unhandled string formatting sign
            (r'%', String)
            # newlines are an error (use "nl" state)
        ],
        'nl': [
            (r'\n', String)
        ],
        'dqs': [
            (r'"', String, '#pop'),
            (r'\\\\|\\"|\\\n', String.Escape),  # included here again for raw strings
            include('strings')
        ],
        'sqs': [
            (r"'", String, '#pop'),
            (r"\\\\|\\'|\\\n", String.Escape),  # included here again for raw strings
            include('strings')
        ],
        'tdqs': [
            (r'"""', String, '#pop'),
            include('strings'),
            include('nl')
        ],
        'tsqs': [
            (r"'''", String, '#pop'),
            include('strings'),
            include('nl')
        ],
    }


class DgLexer(RegexLexer):
    """
    Lexer for dg,
    a functional and object-oriented programming language
    running on the CPython 3 VM.
    """
    name = 'dg'
    aliases = ['dg']
    filenames = ['*.dg']
    mimetypes = ['text/x-dg']
    url = 'http://pyos.github.io/dg'
    version_added = '1.6'

    tokens = {
        'root': [
            (r'\s+', Text),
            (r'#.*?$', Comment.Single),

            (r'(?i)0b[01]+', Number.Bin),
            (r'(?i)0o[0-7]+', Number.Oct),
            (r'(?i)0x[0-9a-f]+', Number.Hex),
            (r'(?i)[+-]?[0-9]+\.[0-9]+(e[+-]?[0-9]+)?j?', Number.Float),
            (r'(?i)[+-]?[0-9]+e[+-]?\d+j?', Number.Float),
            (r'(?i)[+-]?[0-9]+j?', Number.Integer),

            (r"(?i)(br|r?b?)'''", String, combined('stringescape', 'tsqs', 'string')),
            (r'(?i)(br|r?b?)"""', String, combined('stringescape', 'tdqs', 'string')),
            (r"(?i)(br|r?b?)'", String, combined('stringescape', 'sqs', 'string')),
            (r'(?i)(br|r?b?)"', String, combined('stringescape', 'dqs', 'string')),

            (r"`\w+'*`", Operator),
            (r'\b(and|in|is|or|where)\b', Operator.Word),
            (r'[!$%&*+\-./:<-@\\^|~;,]+', Operator),

            (words((
                'bool', 'bytearray', 'bytes', 'classmethod', 'complex', 'dict', 'dict\'',
                'float', 'frozenset', 'int', 'list', 'list\'', 'memoryview', 'object',
                'property', 'range', 'set', 'set\'', 'slice', 'staticmethod', 'str',
                'super', 'tuple', 'tuple\'', 'type'),
                   prefix=r'(?<!\.)', suffix=r'(?![\'\w])'),
             Name.Builtin),
            (words((
                '__import__', 'abs', 'all', 'any', 'bin', 'bind', 'chr', 'cmp', 'compile',
                'complex', 'delattr', 'dir', 'divmod', 'drop', 'dropwhile', 'enumerate',
                'eval', 'exhaust', 'filter', 'flip', 'foldl1?', 'format', 'fst',
                'getattr', 'globals', 'hasattr', 'hash', 'head', 'hex', 'id', 'init',
                'input', 'isinstance', 'issubclass', 'iter', 'iterate', 'last', 'len',
                'locals', 'map', 'max', 'min', 'next', 'oct', 'open', 'ord', 'pow',
                'print', 'repr', 'reversed', 'round', 'setattr', 'scanl1?', 'snd',
                'sorted', 'sum', 'tail', 'take', 'takewhile', 'vars', 'zip'),
                   prefix=r'(?<!\.)', suffix=r'(?![\'\w])'),
             Name.Builtin),
            (r"(?<!\.)(self|Ellipsis|NotImplemented|None|True|False)(?!['\w])",
             Name.Builtin.Pseudo),

            (r"(?<!\.)[A-Z]\w*(Error|Exception|Warning)'*(?!['\w])",
             Name.Exception),
            (r"(?<!\.)(Exception|GeneratorExit|KeyboardInterrupt|StopIteration|"
             r"SystemExit)(?!['\w])", Name.Exception),

            (r"(?<![\w.])(except|finally|for|if|import|not|otherwise|raise|"
             r"subclass|while|with|yield)(?!['\w])", Keyword.Reserved),

            (r"[A-Z_]+'*(?!['\w])", Name),
            (r"[A-Z]\w+'*(?!['\w])", Keyword.Type),
            (r"\w+'*", Name),

            (r'[()]', Punctuation),
            (r'.', Error),
        ],
        'stringescape': [
            (r'\\([\\abfnrtv"\']|\n|N\{.*?\}|u[a-fA-F0-9]{4}|'
             r'U[a-fA-F0-9]{8}|x[a-fA-F0-9]{2}|[0-7]{1,3})', String.Escape)
        ],
        'string': [
            (r'%(\(\w+\))?[-#0 +]*([0-9]+|[*])?(\.([0-9]+|[*]))?'
             '[hlL]?[E-GXc-giorsux%]', String.Interpol),
            (r'[^\\\'"%\n]+', String),
            # quotes, percents and backslashes must be parsed one at a time
            (r'[\'"\\]', String),
            # unhandled string formatting sign
            (r'%', String),
            (r'\n', String)
        ],
        'dqs': [
            (r'"', String, '#pop')
        ],
        'sqs': [
            (r"'", String, '#pop')
        ],
        'tdqs': [
            (r'"""', String, '#pop')
        ],
        'tsqs': [
            (r"'''", String, '#pop')
        ],
    }


class NumPyLexer(PythonLexer):
    """
    A Python lexer recognizing Numerical Python builtins.
    """

    name = 'NumPy'
    url = 'https://numpy.org/'
    aliases = ['numpy']
    version_added = '0.10'

    # override the mimetypes to not inherit them from python
    mimetypes = []
    filenames = []

    EXTRA_KEYWORDS = {
        'abs', 'absolute', 'accumulate', 'add', 'alen', 'all', 'allclose',
        'alltrue', 'alterdot', 'amax', 'amin', 'angle', 'any', 'append',
        'apply_along_axis', 'apply_over_axes', 'arange', 'arccos', 'arccosh',
        'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin',
        'argsort', 'argwhere', 'around', 'array', 'array2string', 'array_equal',
        'array_equiv', 'array_repr', 'array_split', 'array_str', 'arrayrange',
        'asanyarray', 'asarray', 'asarray_chkfinite', 'ascontiguousarray',
        'asfarray', 'asfortranarray', 'asmatrix', 'asscalar', 'astype',
        'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'bartlett',
        'base_repr', 'beta', 'binary_repr', 'bincount', 'binomial',
        'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'blackman',
        'bmat', 'broadcast', 'byte_bounds', 'bytes', 'byteswap', 'c_',
        'can_cast', 'ceil', 'choose', 'clip', 'column_stack', 'common_type',
        'compare_chararrays', 'compress', 'concatenate', 'conj', 'conjugate',
        'convolve', 'copy', 'corrcoef', 'correlate', 'cos', 'cosh', 'cov',
        'cross', 'cumprod', 'cumproduct', 'cumsum', 'delete', 'deprecate',
        'diag', 'diagflat', 'diagonal', 'diff', 'digitize', 'disp', 'divide',
        'dot', 'dsplit', 'dstack', 'dtype', 'dump', 'dumps', 'ediff1d', 'empty',
        'empty_like', 'equal', 'exp', 'expand_dims', 'expm1', 'extract', 'eye',
        'fabs', 'fastCopyAndTranspose', 'fft', 'fftfreq', 'fftshift', 'fill',
        'finfo', 'fix', 'flat', 'flatnonzero', 'flatten', 'fliplr', 'flipud',
        'floor', 'floor_divide', 'fmod', 'frexp', 'fromarrays', 'frombuffer',
        'fromfile', 'fromfunction', 'fromiter', 'frompyfunc', 'fromstring',
        'generic', 'get_array_wrap', 'get_include', 'get_numarray_include',
        'get_numpy_include', 'get_printoptions', 'getbuffer', 'getbufsize',
        'geterr', 'geterrcall', 'geterrobj', 'getfield', 'gradient', 'greater',
        'greater_equal', 'gumbel', 'hamming', 'hanning', 'histogram',
        'histogram2d', 'histogramdd', 'hsplit', 'hstack', 'hypot', 'i0',
        'identity', 'ifft', 'imag', 'index_exp', 'indices', 'inf', 'info',
        'inner', 'insert', 'int_asbuffer', 'interp', 'intersect1d',
        'intersect1d_nu', 'inv', 'invert', 'iscomplex', 'iscomplexobj',
        'isfinite', 'isfortran', 'isinf', 'isnan', 'isneginf', 'isposinf',
        'isreal', 'isrealobj', 'isscalar', 'issctype', 'issubclass_',
        'issubdtype', 'issubsctype', 'item', 'itemset', 'iterable', 'ix_',
        'kaiser', 'kron', 'ldexp', 'left_shift', 'less', 'less_equal', 'lexsort',
        'linspace', 'load', 'loads', 'loadtxt', 'log', 'log10', 'log1p', 'log2',
        'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logspace',
        'lstsq', 'mat', 'matrix', 'max', 'maximum', 'maximum_sctype',
        'may_share_memory', 'mean', 'median', 'meshgrid', 'mgrid', 'min',
        'minimum', 'mintypecode', 'mod', 'modf', 'msort', 'multiply', 'nan',
        'nan_to_num', 'nanargmax', 'nanargmin', 'nanmax', 'nanmin', 'nansum',
        'ndenumerate', 'ndim', 'ndindex', 'negative', 'newaxis', 'newbuffer',
        'newbyteorder', 'nonzero', 'not_equal', 'obj2sctype', 'ogrid', 'ones',
        'ones_like', 'outer', 'permutation', 'piecewise', 'pinv', 'pkgload',
        'place', 'poisson', 'poly', 'poly1d', 'polyadd', 'polyder', 'polydiv',
        'polyfit', 'polyint', 'polymul', 'polysub', 'polyval', 'power', 'prod',
        'product', 'ptp', 'put', 'putmask', 'r_', 'randint', 'random_integers',
        'random_sample', 'ranf', 'rank', 'ravel', 'real', 'real_if_close',
        'recarray', 'reciprocal', 'reduce', 'remainder', 'repeat', 'require',
        'reshape', 'resize', 'restoredot', 'right_shift', 'rint', 'roll',
        'rollaxis', 'roots', 'rot90', 'round', 'round_', 'row_stack', 's_',
        'sample', 'savetxt', 'sctype2char', 'searchsorted', 'seed', 'select',
        'set_numeric_ops', 'set_printoptions', 'set_string_function',
        'setbufsize', 'setdiff1d', 'seterr', 'seterrcall', 'seterrobj',
        'setfield', 'setflags', 'setmember1d', 'setxor1d', 'shape',
        'show_config', 'shuffle', 'sign', 'signbit', 'sin', 'sinc', 'sinh',
        'size', 'slice', 'solve', 'sometrue', 'sort', 'sort_complex', 'source',
        'split', 'sqrt', 'square', 'squeeze', 'standard_normal', 'std',
        'subtract', 'sum', 'svd', 'swapaxes', 'take', 'tan', 'tanh', 'tensordot',
        'test', 'tile', 'tofile', 'tolist', 'tostring', 'trace', 'transpose',
        'trapz', 'tri', 'tril', 'trim_zeros', 'triu', 'true_divide', 'typeDict',
        'typename', 'uniform', 'union1d', 'unique', 'unique1d', 'unravel_index',
        'unwrap', 'vander', 'var', 'vdot', 'vectorize', 'view', 'vonmises',
        'vsplit', 'vstack', 'weibull', 'where', 'who', 'zeros', 'zeros_like'
    }

    def get_tokens_unprocessed(self, text):
        for index, token, value in \
                PythonLexer.get_tokens_unprocessed(self, text):
            if token is Name and value in self.EXTRA_KEYWORDS:
                yield index, Keyword.Pseudo, value
            else:
                yield index, token, value

    def analyse_text(text):
        ltext = text[:1000]
        return (shebang_matches(text, r'pythonw?(3(\.\d)?)?') or
                'import ' in ltext) \
            and ('import numpy' in ltext or 'from numpy import' in ltext)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\emulation.py ===
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
    :param display_feature: **(EXPERIMENTAL)** *(Optional)* If set, the display feature of a multi-segment screen. If not set, multi-segment support is turned-off.
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\_polybase.py ===
"""
Abstract base class for the various polynomial Classes.

The ABCPolyBase class provides the methods needed to implement the common API
for the various polynomial classes. It operates as a mixin, but uses the
abc module from the stdlib, hence it is only available for Python >= 2.6.

"""
import abc
import numbers
import os
from collections.abc import Callable

import numpy as np

from . import polyutils as pu

__all__ = ['ABCPolyBase']

class ABCPolyBase(abc.ABC):
    """An abstract base class for immutable series classes.

    ABCPolyBase provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' along with the
    methods listed below.

    Parameters
    ----------
    coef : array_like
        Series coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``, where
        ``P_i`` is the basis polynomials of degree ``i``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is the derived class domain.
    window : (2,) array_like, optional
        Window, see domain for its use. The default value is the
        derived class window.
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    Attributes
    ----------
    coef : (N,) ndarray
        Series coefficients in order of increasing degree.
    domain : (2,) ndarray
        Domain that is mapped to window.
    window : (2,) ndarray
        Window that domain is mapped to.
    symbol : str
        Symbol representing the independent variable.

    Class Attributes
    ----------------
    maxpower : int
        Maximum power allowed, i.e., the largest number ``n`` such that
        ``p(x)**n`` is allowed. This is to limit runaway polynomial size.
    domain : (2,) ndarray
        Default domain of the class.
    window : (2,) ndarray
        Default window of the class.

    """

    # Not hashable
    __hash__ = None

    # Opt out of numpy ufuncs and Python ops with ndarray subclasses.
    __array_ufunc__ = None

    # Limit runaway size. T_n^m has degree n*m
    maxpower = 100

    # Unicode character mappings for improved __str__
    _superscript_mapping = str.maketrans({
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹"
    })
    _subscript_mapping = str.maketrans({
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉"
    })
    # Some fonts don't support full unicode character ranges necessary for
    # the full set of superscripts and subscripts, including common/default
    # fonts in Windows shells/terminals. Therefore, default to ascii-only
    # printing on windows.
    _use_unicode = not os.name == 'nt'

    @property
    def symbol(self):
        return self._symbol

    @property
    @abc.abstractmethod
    def domain(self):
        pass

    @property
    @abc.abstractmethod
    def window(self):
        pass

    @property
    @abc.abstractmethod
    def basis_name(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def _add(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _sub(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _mul(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _div(c1, c2):
        pass

    @staticmethod
    @abc.abstractmethod
    def _pow(c, pow, maxpower=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def _val(x, c):
        pass

    @staticmethod
    @abc.abstractmethod
    def _int(c, m, k, lbnd, scl):
        pass

    @staticmethod
    @abc.abstractmethod
    def _der(c, m, scl):
        pass

    @staticmethod
    @abc.abstractmethod
    def _fit(x, y, deg, rcond, full):
        pass

    @staticmethod
    @abc.abstractmethod
    def _line(off, scl):
        pass

    @staticmethod
    @abc.abstractmethod
    def _roots(c):
        pass

    @staticmethod
    @abc.abstractmethod
    def _fromroots(r):
        pass

    def has_samecoef(self, other):
        """Check if coefficients match.

        Parameters
        ----------
        other : class instance
            The other class must have the ``coef`` attribute.

        Returns
        -------
        bool : boolean
            True if the coefficients are the same, False otherwise.

        """
        return (
            len(self.coef) == len(other.coef)
            and np.all(self.coef == other.coef)
        )

    def has_samedomain(self, other):
        """Check if domains match.

        Parameters
        ----------
        other : class instance
            The other class must have the ``domain`` attribute.

        Returns
        -------
        bool : boolean
            True if the domains are the same, False otherwise.

        """
        return np.all(self.domain == other.domain)

    def has_samewindow(self, other):
        """Check if windows match.

        Parameters
        ----------
        other : class instance
            The other class must have the ``window`` attribute.

        Returns
        -------
        bool : boolean
            True if the windows are the same, False otherwise.

        """
        return np.all(self.window == other.window)

    def has_sametype(self, other):
        """Check if types match.

        Parameters
        ----------
        other : object
            Class instance.

        Returns
        -------
        bool : boolean
            True if other is same class as self

        """
        return isinstance(other, self.__class__)

    def _get_coefficients(self, other):
        """Interpret other as polynomial coefficients.

        The `other` argument is checked to see if it is of the same
        class as self with identical domain and window. If so,
        return its coefficients, otherwise return `other`.

        Parameters
        ----------
        other : anything
            Object to be checked.

        Returns
        -------
        coef
            The coefficients of`other` if it is a compatible instance,
            of ABCPolyBase, otherwise `other`.

        Raises
        ------
        TypeError
            When `other` is an incompatible instance of ABCPolyBase.

        """
        if isinstance(other, ABCPolyBase):
            if not isinstance(other, self.__class__):
                raise TypeError("Polynomial types differ")
            elif not np.all(self.domain == other.domain):
                raise TypeError("Domains differ")
            elif not np.all(self.window == other.window):
                raise TypeError("Windows differ")
            elif self.symbol != other.symbol:
                raise ValueError("Polynomial symbols differ")
            return other.coef
        return other

    def __init__(self, coef, domain=None, window=None, symbol='x'):
        [coef] = pu.as_series([coef], trim=False)
        self.coef = coef

        if domain is not None:
            [domain] = pu.as_series([domain], trim=False)
            if len(domain) != 2:
                raise ValueError("Domain has wrong number of elements.")
            self.domain = domain

        if window is not None:
            [window] = pu.as_series([window], trim=False)
            if len(window) != 2:
                raise ValueError("Window has wrong number of elements.")
            self.window = window

        # Validation for symbol
        try:
            if not symbol.isidentifier():
                raise ValueError(
                    "Symbol string must be a valid Python identifier"
                )
        # If a user passes in something other than a string, the above
        # results in an AttributeError. Catch this and raise a more
        # informative exception
        except AttributeError:
            raise TypeError("Symbol must be a non-empty string")

        self._symbol = symbol

    def __repr__(self):
        coef = repr(self.coef)[6:-1]
        domain = repr(self.domain)[6:-1]
        window = repr(self.window)[6:-1]
        name = self.__class__.__name__
        return (f"{name}({coef}, domain={domain}, window={window}, "
                f"symbol='{self.symbol}')")

    def __format__(self, fmt_str):
        if fmt_str == '':
            return self.__str__()
        if fmt_str not in ('ascii', 'unicode'):
            raise ValueError(
                f"Unsupported format string '{fmt_str}' passed to "
                f"{self.__class__}.__format__. Valid options are "
                f"'ascii' and 'unicode'"
            )
        if fmt_str == 'ascii':
            return self._generate_string(self._str_term_ascii)
        return self._generate_string(self._str_term_unicode)

    def __str__(self):
        if self._use_unicode:
            return self._generate_string(self._str_term_unicode)
        return self._generate_string(self._str_term_ascii)

    def _generate_string(self, term_method):
        """
        Generate the full string representation of the polynomial, using
        ``term_method`` to generate each polynomial term.
        """
        # Get configuration for line breaks
        linewidth = np.get_printoptions().get('linewidth', 75)
        if linewidth < 1:
            linewidth = 1
        out = pu.format_float(self.coef[0])

        off, scale = self.mapparms()

        scaled_symbol, needs_parens = self._format_term(pu.format_float,
                                                        off, scale)
        if needs_parens:
            scaled_symbol = '(' + scaled_symbol + ')'

        for i, coef in enumerate(self.coef[1:]):
            out += " "
            power = str(i + 1)
            # Polynomial coefficient
            # The coefficient array can be an object array with elements that
            # will raise a TypeError with >= 0 (e.g. strings or Python
            # complex). In this case, represent the coefficient as-is.
            try:
                if coef >= 0:
                    next_term = "+ " + pu.format_float(coef, parens=True)
                else:
                    next_term = "- " + pu.format_float(-coef, parens=True)
            except TypeError:
                next_term = f"+ {coef}"
            # Polynomial term
            next_term += term_method(power, scaled_symbol)
            # Length of the current line with next term added
            line_len = len(out.split('\n')[-1]) + len(next_term)
            # If not the last term in the polynomial, it will be two
            # characters longer due to the +/- with the next term
            if i < len(self.coef[1:]) - 1:
                line_len += 2
            # Handle linebreaking
            if line_len >= linewidth:
                next_term = next_term.replace(" ", "\n", 1)
            out += next_term
        return out

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        """
        String representation of single polynomial term using unicode
        characters for superscripts and subscripts.
        """
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis_name, or override "
                "_str_term_unicode(cls, i, arg_str)"
            )
        return (f"·{cls.basis_name}{i.translate(cls._subscript_mapping)}"
                f"({arg_str})")

    @classmethod
    def _str_term_ascii(cls, i, arg_str):
        """
        String representation of a single polynomial term using ** and _ to
        represent superscripts and subscripts, respectively.
        """
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis_name, or override "
                "_str_term_ascii(cls, i, arg_str)"
            )
        return f" {cls.basis_name}_{i}({arg_str})"

    @classmethod
    def _repr_latex_term(cls, i, arg_str, needs_parens):
        if cls.basis_name is None:
            raise NotImplementedError(
                "Subclasses must define either a basis name, or override "
                "_repr_latex_term(i, arg_str, needs_parens)")
        # since we always add parens, we don't care if the expression needs them
        return f"{{{cls.basis_name}}}_{{{i}}}({arg_str})"

    @staticmethod
    def _repr_latex_scalar(x, parens=False):
        # TODO: we're stuck with disabling math formatting until we handle
        # exponents in this function
        return fr'\text{{{pu.format_float(x, parens=parens)}}}'

    def _format_term(self, scalar_format: Callable, off: float, scale: float):
        """ Format a single term in the expansion """
        if off == 0 and scale == 1:
            term = self.symbol
            needs_parens = False
        elif scale == 1:
            term = f"{scalar_format(off)} + {self.symbol}"
            needs_parens = True
        elif off == 0:
            term = f"{scalar_format(scale)}{self.symbol}"
            needs_parens = True
        else:
            term = (
                f"{scalar_format(off)} + "
                f"{scalar_format(scale)}{self.symbol}"
            )
            needs_parens = True
        return term, needs_parens

    def _repr_latex_(self):
        # get the scaled argument string to the basis functions
        off, scale = self.mapparms()
        term, needs_parens = self._format_term(self._repr_latex_scalar,
                                               off, scale)

        mute = r"\color{{LightGray}}{{{}}}".format

        parts = []
        for i, c in enumerate(self.coef):
            # prevent duplication of + and - signs
            if i == 0:
                coef_str = f"{self._repr_latex_scalar(c)}"
            elif not isinstance(c, numbers.Real):
                coef_str = f" + ({self._repr_latex_scalar(c)})"
            elif c >= 0:
                coef_str = f" + {self._repr_latex_scalar(c, parens=True)}"
            else:
                coef_str = f" - {self._repr_latex_scalar(-c, parens=True)}"

            # produce the string for the term
            term_str = self._repr_latex_term(i, term, needs_parens)
            if term_str == '1':
                part = coef_str
            else:
                part = rf"{coef_str}\,{term_str}"

            if c == 0:
                part = mute(part)

            parts.append(part)

        if parts:
            body = ''.join(parts)
        else:
            # in case somehow there are no coefficients at all
            body = '0'

        return rf"${self.symbol} \mapsto {body}$"

    # Pickle and copy

    def __getstate__(self):
        ret = self.__dict__.copy()
        ret['coef'] = self.coef.copy()
        ret['domain'] = self.domain.copy()
        ret['window'] = self.window.copy()
        ret['symbol'] = self.symbol
        return ret

    def __setstate__(self, dict):
        self.__dict__ = dict

    # Call

    def __call__(self, arg):
        arg = pu.mapdomain(arg, self.domain, self.window)
        return self._val(arg, self.coef)

    def __iter__(self):
        return iter(self.coef)

    def __len__(self):
        return len(self.coef)

    # Numeric properties.

    def __neg__(self):
        return self.__class__(
            -self.coef, self.domain, self.window, self.symbol
        )

    def __pos__(self):
        return self

    def __add__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            coef = self._add(self.coef, othercoef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __sub__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            coef = self._sub(self.coef, othercoef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __mul__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            coef = self._mul(self.coef, othercoef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __truediv__(self, other):
        # there is no true divide if the rhs is not a Number, although it
        # could return the first n elements of an infinite series.
        # It is hard to see where n would come from, though.
        if not isinstance(other, numbers.Number) or isinstance(other, bool):
            raise TypeError(
                f"unsupported types for true division: "
                f"'{type(self)}', '{type(other)}'"
            )
        return self.__floordiv__(other)

    def __floordiv__(self, other):
        res = self.__divmod__(other)
        if res is NotImplemented:
            return res
        return res[0]

    def __mod__(self, other):
        res = self.__divmod__(other)
        if res is NotImplemented:
            return res
        return res[1]

    def __divmod__(self, other):
        othercoef = self._get_coefficients(other)
        try:
            quo, rem = self._div(self.coef, othercoef)
        except ZeroDivisionError:
            raise
        except Exception:
            return NotImplemented
        quo = self.__class__(quo, self.domain, self.window, self.symbol)
        rem = self.__class__(rem, self.domain, self.window, self.symbol)
        return quo, rem

    def __pow__(self, other):
        coef = self._pow(self.coef, other, maxpower=self.maxpower)
        res = self.__class__(coef, self.domain, self.window, self.symbol)
        return res

    def __radd__(self, other):
        try:
            coef = self._add(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rsub__(self, other):
        try:
            coef = self._sub(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rmul__(self, other):
        try:
            coef = self._mul(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rtruediv__(self, other):
        # An instance of ABCPolyBase is not considered a
        # Number.
        return NotImplemented

    def __rfloordiv__(self, other):
        res = self.__rdivmod__(other)
        if res is NotImplemented:
            return res
        return res[0]

    def __rmod__(self, other):
        res = self.__rdivmod__(other)
        if res is NotImplemented:
            return res
        return res[1]

    def __rdivmod__(self, other):
        try:
            quo, rem = self._div(other, self.coef)
        except ZeroDivisionError:
            raise
        except Exception:
            return NotImplemented
        quo = self.__class__(quo, self.domain, self.window, self.symbol)
        rem = self.__class__(rem, self.domain, self.window, self.symbol)
        return quo, rem

    def __eq__(self, other):
        res = (isinstance(other, self.__class__) and
               np.all(self.domain == other.domain) and
               np.all(self.window == other.window) and
               (self.coef.shape == other.coef.shape) and
               np.all(self.coef == other.coef) and
               (self.symbol == other.symbol))
        return res

    def __ne__(self, other):
        return not self.__eq__(other)

    #
    # Extra methods.
    #

    def copy(self):
        """Return a copy.

        Returns
        -------
        new_series : series
            Copy of self.

        """
        return self.__class__(self.coef, self.domain, self.window, self.symbol)

    def degree(self):
        """The degree of the series.

        Returns
        -------
        degree : int
            Degree of the series, one less than the number of coefficients.

        Examples
        --------

        Create a polynomial object for ``1 + 7*x + 4*x**2``:

        >>> np.polynomial.set_default_printstyle("unicode")
        >>> poly = np.polynomial.Polynomial([1, 7, 4])
        >>> print(poly)
        1.0 + 7.0·x + 4.0·x²
        >>> poly.degree()
        2

        Note that this method does not check for non-zero coefficients.
        You must trim the polynomial to remove any trailing zeroes:

        >>> poly = np.polynomial.Polynomial([1, 7, 0])
        >>> print(poly)
        1.0 + 7.0·x + 0.0·x²
        >>> poly.degree()
        2
        >>> poly.trim().degree()
        1

        """
        return len(self) - 1

    def cutdeg(self, deg):
        """Truncate series to the given degree.

        Reduce the degree of the series to `deg` by discarding the
        high order terms. If `deg` is greater than the current degree a
        copy of the current series is returned. This can be useful in least
        squares where the coefficients of the high degree terms may be very
        small.

        Parameters
        ----------
        deg : non-negative int
            The series is reduced to degree `deg` by discarding the high
            order terms. The value of `deg` must be a non-negative integer.

        Returns
        -------
        new_series : series
            New instance of series with reduced degree.

        """
        return self.truncate(deg + 1)

    def trim(self, tol=0):
        """Remove trailing coefficients

        Remove trailing coefficients until a coefficient is reached whose
        absolute value greater than `tol` or the beginning of the series is
        reached. If all the coefficients would be removed the series is set
        to ``[0]``. A new series instance is returned with the new
        coefficients.  The current instance remains unchanged.

        Parameters
        ----------
        tol : non-negative number.
            All trailing coefficients less than `tol` will be removed.

        Returns
        -------
        new_series : series
            New instance of series with trimmed coefficients.

        """
        coef = pu.trimcoef(self.coef, tol)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def truncate(self, size):
        """Truncate series to length `size`.

        Reduce the series to length `size` by discarding the high
        degree terms. The value of `size` must be a positive integer. This
        can be useful in least squares where the coefficients of the
        high degree terms may be very small.

        Parameters
        ----------
        size : positive int
            The series is reduced to length `size` by discarding the high
            degree terms. The value of `size` must be a positive integer.

        Returns
        -------
        new_series : series
            New instance of series with truncated coefficients.

        """
        isize = int(size)
        if isize != size or isize < 1:
            raise ValueError("size must be a positive integer")
        if isize >= len(self.coef):
            coef = self.coef
        else:
            coef = self.coef[:isize]
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def convert(self, domain=None, kind=None, window=None):
        """Convert series to a different kind and/or domain and/or window.

        Parameters
        ----------
        domain : array_like, optional
            The domain of the converted series. If the value is None,
            the default domain of `kind` is used.
        kind : class, optional
            The polynomial series type class to which the current instance
            should be converted. If kind is None, then the class of the
            current instance is used.
        window : array_like, optional
            The window of the converted series. If the value is None,
            the default window of `kind` is used.

        Returns
        -------
        new_series : series
            The returned class can be of different type than the current
            instance and/or have a different domain and/or different
            window.

        Notes
        -----
        Conversion between domains and class types can result in
        numerically ill defined series.

        """
        if kind is None:
            kind = self.__class__
        if domain is None:
            domain = kind.domain
        if window is None:
            window = kind.window
        return self(kind.identity(domain, window=window, symbol=self.symbol))

    def mapparms(self):
        """Return the mapping parameters.

        The returned values define a linear map ``off + scl*x`` that is
        applied to the input arguments before the series is evaluated. The
        map depends on the ``domain`` and ``window``; if the current
        ``domain`` is equal to the ``window`` the resulting map is the
        identity.  If the coefficients of the series instance are to be
        used by themselves outside this class, then the linear function
        must be substituted for the ``x`` in the standard representation of
        the base polynomials.

        Returns
        -------
        off, scl : float or complex
            The mapping function is defined by ``off + scl*x``.

        Notes
        -----
        If the current domain is the interval ``[l1, r1]`` and the window
        is ``[l2, r2]``, then the linear mapping function ``L`` is
        defined by the equations::

            L(l1) = l2
            L(r1) = r2

        """
        return pu.mapparms(self.domain, self.window)

    def integ(self, m=1, k=[], lbnd=None):
        """Integrate.

        Return a series instance that is the definite integral of the
        current series.

        Parameters
        ----------
        m : non-negative int
            The number of integrations to perform.
        k : array_like
            Integration constants. The first constant is applied to the
            first integration, the second to the second, and so on. The
            list of values must less than or equal to `m` in length and any
            missing values are set to zero.
        lbnd : Scalar
            The lower bound of the definite integral.

        Returns
        -------
        new_series : series
            A new series representing the integral. The domain is the same
            as the domain of the integrated series.

        """
        off, scl = self.mapparms()
        if lbnd is None:
            lbnd = 0
        else:
            lbnd = off + scl * lbnd
        coef = self._int(self.coef, m, k, lbnd, 1. / scl)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def deriv(self, m=1):
        """Differentiate.

        Return a series instance of that is the derivative of the current
        series.

        Parameters
        ----------
        m : non-negative int
            Find the derivative of order `m`.

        Returns
        -------
        new_series : series
            A new series representing the derivative. The domain is the same
            as the domain of the differentiated series.

        """
        off, scl = self.mapparms()
        coef = self._der(self.coef, m, scl)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def roots(self):
        """Return the roots of the series polynomial.

        Compute the roots for the series. Note that the accuracy of the
        roots decreases the further outside the `domain` they lie.

        Returns
        -------
        roots : ndarray
            Array containing the roots of the series.

        """
        roots = self._roots(self.coef)
        return pu.mapdomain(roots, self.window, self.domain)

    def linspace(self, n=100, domain=None):
        """Return x, y values at equally spaced points in domain.

        Returns the x, y values at `n` linearly spaced points across the
        domain.  Here y is the value of the polynomial at the points x. By
        default the domain is the same as that of the series instance.
        This method is intended mostly as a plotting aid.

        Parameters
        ----------
        n : int, optional
            Number of point pairs to return. The default value is 100.
        domain : {None, array_like}, optional
            If not None, the specified domain is used instead of that of
            the calling instance. It should be of the form ``[beg,end]``.
            The default is None which case the class domain is used.

        Returns
        -------
        x, y : ndarray
            x is equal to linspace(self.domain[0], self.domain[1], n) and
            y is the series evaluated at element of x.

        """
        if domain is None:
            domain = self.domain
        x = np.linspace(domain[0], domain[1], n)
        y = self(x)
        return x, y

    @classmethod
    def fit(cls, x, y, deg, domain=None, rcond=None, full=False, w=None,
        window=None, symbol='x'):
        """Least squares fit to data.

        Return a series instance that is the least squares fit to the data
        `y` sampled at `x`. The domain of the returned instance can be
        specified and this will often result in a superior fit with less
        chance of ill conditioning.

        Parameters
        ----------
        x : array_like, shape (M,)
            x-coordinates of the M sample points ``(x[i], y[i])``.
        y : array_like, shape (M,)
            y-coordinates of the M sample points ``(x[i], y[i])``.
        deg : int or 1-D array_like
            Degree(s) of the fitting polynomials. If `deg` is a single integer
            all terms up to and including the `deg`'th term are included in the
            fit. For NumPy versions >= 1.11.0 a list of integers specifying the
            degrees of the terms to include may be used instead.
        domain : {None, [beg, end], []}, optional
            Domain to use for the returned series. If ``None``,
            then a minimal domain that covers the points `x` is chosen.  If
            ``[]`` the class domain is used. The default value was the
            class domain in NumPy 1.4 and ``None`` in later versions.
            The ``[]`` option was added in numpy 1.5.0.
        rcond : float, optional
            Relative condition number of the fit. Singular values smaller
            than this relative to the largest singular value will be
            ignored. The default value is ``len(x)*eps``, where eps is the
            relative precision of the float type, about 2e-16 in most
            cases.
        full : bool, optional
            Switch determining nature of return value. When it is False
            (the default) just the coefficients are returned, when True
            diagnostic information from the singular value decomposition is
            also returned.
        w : array_like, shape (M,), optional
            Weights. If not None, the weight ``w[i]`` applies to the unsquared
            residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
            chosen so that the errors of the products ``w[i]*y[i]`` all have
            the same variance.  When using inverse-variance weighting, use
            ``w[i] = 1/sigma(y[i])``.  The default value is None.
        window : {[beg, end]}, optional
            Window to use for the returned series. The default
            value is the default class domain
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
            A series that represents the least squares fit to the data and
            has the domain and window specified in the call. If the
            coefficients for the unscaled and unshifted basis polynomials are
            of interest, do ``new_series.convert().coef``.

        [resid, rank, sv, rcond] : list
            These values are only returned if ``full == True``

            - resid -- sum of squared residuals of the least squares fit
            - rank -- the numerical rank of the scaled Vandermonde matrix
            - sv -- singular values of the scaled Vandermonde matrix
            - rcond -- value of `rcond`.

            For more details, see `linalg.lstsq`.

        """
        if domain is None:
            domain = pu.getdomain(x)
            if domain[0] == domain[1]:
                domain[0] -= 1
                domain[1] += 1
        elif isinstance(domain, list) and len(domain) == 0:
            domain = cls.domain

        if window is None:
            window = cls.window

        xnew = pu.mapdomain(x, domain, window)
        res = cls._fit(xnew, y, deg, w=w, rcond=rcond, full=full)
        if full:
            [coef, status] = res
            return (
                cls(coef, domain=domain, window=window, symbol=symbol), status
            )
        else:
            coef = res
            return cls(coef, domain=domain, window=window, symbol=symbol)

    @classmethod
    def fromroots(cls, roots, domain=[], window=None, symbol='x'):
        """Return series instance that has the specified roots.

        Returns a series representing the product
        ``(x - r[0])*(x - r[1])*...*(x - r[n-1])``, where ``r`` is a
        list of roots.

        Parameters
        ----------
        roots : array_like
            List of roots.
        domain : {[], None, array_like}, optional
            Domain for the resulting series. If None the domain is the
            interval from the smallest root to the largest. If [] the
            domain is the class domain. The default is [].
        window : {None, array_like}, optional
            Window for the returned series. If None the class window is
            used. The default is None.
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
            Series with the specified roots.

        """
        [roots] = pu.as_series([roots], trim=False)
        if domain is None:
            domain = pu.getdomain(roots)
        elif isinstance(domain, list) and len(domain) == 0:
            domain = cls.domain

        if window is None:
            window = cls.window

        deg = len(roots)
        off, scl = pu.mapparms(domain, window)
        rnew = off + scl * roots
        coef = cls._fromroots(rnew) / scl**deg
        return cls(coef, domain=domain, window=window, symbol=symbol)

    @classmethod
    def identity(cls, domain=None, window=None, symbol='x'):
        """Identity function.

        If ``p`` is the returned series, then ``p(x) == x`` for all
        values of x.

        Parameters
        ----------
        domain : {None, array_like}, optional
            If given, the array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain. If None is
            given then the class domain is used. The default is None.
        window : {None, array_like}, optional
            If given, the resulting array must be if the form
            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
            the window. If None is given then the class window is used. The
            default is None.
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
             Series of representing the identity.

        """
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        off, scl = pu.mapparms(window, domain)
        coef = cls._line(off, scl)
        return cls(coef, domain, window, symbol)

    @classmethod
    def basis(cls, deg, domain=None, window=None, symbol='x'):
        """Series basis polynomial of degree `deg`.

        Returns the series representing the basis polynomial of degree `deg`.

        Parameters
        ----------
        deg : int
            Degree of the basis polynomial for the series. Must be >= 0.
        domain : {None, array_like}, optional
            If given, the array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain. If None is
            given then the class domain is used. The default is None.
        window : {None, array_like}, optional
            If given, the resulting array must be if the form
            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
            the window. If None is given then the class window is used. The
            default is None.
        symbol : str, optional
            Symbol representing the independent variable. Default is 'x'.

        Returns
        -------
        new_series : series
            A series with the coefficient of the `deg` term set to one and
            all others zero.

        """
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        ideg = int(deg)

        if ideg != deg or ideg < 0:
            raise ValueError("deg must be non-negative integer")
        return cls([0] * ideg + [1], domain, window, symbol)

    @classmethod
    def cast(cls, series, domain=None, window=None):
        """Convert series to series of this class.

        The `series` is expected to be an instance of some polynomial
        series of one of the types supported by by the numpy.polynomial
        module, but could be some other class that supports the convert
        method.

        Parameters
        ----------
        series : series
            The series instance to be converted.
        domain : {None, array_like}, optional
            If given, the array must be of the form ``[beg, end]``, where
            ``beg`` and ``end`` are the endpoints of the domain. If None is
            given then the class domain is used. The default is None.
        window : {None, array_like}, optional
            If given, the resulting array must be if the form
            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
            the window. If None is given then the class window is used. The
            default is None.

        Returns
        -------
        new_series : series
            A series of the same kind as the calling class and equal to
            `series` when evaluated.

        See Also
        --------
        convert : similar instance method

        """
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        return series.convert(domain, cls, window)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_attention.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

import numpy as np
from fusion_base import Fusion
from fusion_options import AttentionMaskFormat
from fusion_utils import FusionUtils, NumpyHelper
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class AttentionMask:
    """
    Fuse Attention subgraph into one Attention node.
    """

    def __init__(self, model: OnnxModel):
        self.model = model
        # A lookup table with mask input as key, and mask index output as value
        self.mask_indice = {}
        # A lookup table with mask input as key, and cast (to int32) output as value
        self.mask_casted = {}
        self.utils = FusionUtils(model)
        self.mask_format = AttentionMaskFormat.MaskIndexEnd
        self.opset_version = model.get_opset_version()

    def set_mask_format(self, mask_format: AttentionMaskFormat):
        self.mask_format = mask_format

    def set_mask_indice(self, mask, mask_index):
        if mask in self.mask_indice:
            assert mask_index == self.mask_indice[mask]
        self.mask_indice[mask] = mask_index

    def get_first_mask(self):
        assert len(self.mask_indice) > 0
        return next(iter(self.mask_indice))

    def process_mask(self, mask_2d: str) -> str | None:
        if self.mask_format == AttentionMaskFormat.NoMask:
            return None

        if mask_2d in self.mask_indice:
            return self.mask_indice[mask_2d]

        # Add cast to convert int64 to int32
        if self.model.find_graph_input(mask_2d):
            casted, input_name = self.utils.cast_graph_input_to_int32(mask_2d)
        else:
            input_name, _cast_node = self.utils.cast_input_to_int32(mask_2d)
            casted = True

        if casted:
            self.mask_casted[mask_2d] = input_name

        # Attention supports int32 attention mask (2D) since 1.4.0
        if self.mask_format == AttentionMaskFormat.AttentionMask:
            self.mask_indice[mask_2d] = input_name
            return input_name

        # Add a mask processing node to convert attention mask to mask index (1D)
        output_name = self.model.create_node_name("mask_index")
        if self.opset_version < 13:
            mask_index_node = helper.make_node(
                "ReduceSum",
                inputs=[input_name],
                outputs=[output_name],
                name=self.model.create_node_name("ReduceSum", "MaskReduceSum"),
            )
            mask_index_node.attribute.extend([helper.make_attribute("axes", [1]), helper.make_attribute("keepdims", 0)])
        else:
            # ReduceSum-13: axes is moved from attribute to input
            axes_name = "ort_const_1_reduce_sum_axes"
            if self.model.get_initializer(axes_name) is None:
                self.model.add_initializer(
                    helper.make_tensor(
                        name=axes_name,
                        data_type=TensorProto.INT64,
                        dims=[1],
                        vals=[1],
                        raw=False,
                    )
                )
            mask_index_node = helper.make_node(
                "ReduceSum",
                inputs=[input_name, axes_name],
                outputs=[output_name],
                name=self.model.create_node_name("ReduceSum", "MaskReduceSum"),
            )
            mask_index_node.attribute.extend([helper.make_attribute("keepdims", 0)])

        self.model.add_node(mask_index_node)

        self.mask_indice[mask_2d] = output_name
        return output_name


class FusionAttention(Fusion):
    """
    Fuse Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask | None = None,
        use_multi_head_attention: bool = False,
        disable_multi_head_attention_bias: bool = False,
        search_op_types: list[str] = ["SkipLayerNormalization", "LayerNormalization"],  # noqa: B006
    ):
        attention_op_name = "MultiHeadAttention" if use_multi_head_attention else "Attention"
        super().__init__(model, attention_op_name, search_op_types)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_mask = attention_mask if attention_mask else AttentionMask(model)
        self.use_multi_head_attention = use_multi_head_attention
        self.disable_multi_head_attention_bias = disable_multi_head_attention_bias
        self.mask_filter_value = None

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

        self.shape_infer = None
        self.shape_infer_done = True

    def get_num_heads_and_hidden_size_from_concat(self, concat: NodeProto) -> tuple[int, int]:
        """
        Detect num_heads and hidden_size from Concat node in the following subgraph:

        SkipLayerNormalization or EmbedLayerNormalization
                        /        |
                     MatMul    Shape
                        |        |
                       Add     Gather(indices=0)
                        |        |
                        |      Unsqueeze
                        |        |
                        |     Concat (*, -1, 12, 64)
                        |     /
                       Reshape
                          |
                       Transpose
        """
        if len(concat.input) == 4:
            num_heads = self.model.get_constant_value(concat.input[2])
            head_size = self.model.get_constant_value(concat.input[3])
            if (
                isinstance(num_heads, np.ndarray)
                and num_heads.size == 1
                and isinstance(head_size, np.ndarray)
                and head_size.size == 1
            ):
                return num_heads[0], num_heads[0] * head_size[0]

        return self.num_heads, self.hidden_size

    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto) -> tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """
        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        q_shape_value = self.model.get_constant_value(reshape_q.input[1])
        if q_shape_value is None:
            concat = self.model.get_parent(reshape_q, 1)
            if concat is not None and concat.op_type == "Concat":
                return self.get_num_heads_and_hidden_size_from_concat(concat)
            logger.debug("%s is not initializer.", reshape_q.input[1])
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        if (
            (not isinstance(q_shape_value, np.ndarray))
            or len(q_shape_value) != 4
            or (q_shape_value[2] <= 0 or q_shape_value[3] <= 0)
        ):
            logger.debug("q_shape_value=%s. Expected value are like [0, 0, num_heads, head_size].", q_shape_value)
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        num_heads = q_shape_value[2]
        head_size = q_shape_value[3]
        hidden_size = num_heads * head_size

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(
                    "--num_heads is %d. Detected value is %d. Using detected value.", self.num_heads, num_heads
                )
                self.num_heads_warning = False  # Do not show the warning more than once

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    "--hidden_size is %d. Detected value is %d. Using detected value.", self.hidden_size, hidden_size
                )
                self.hidden_size_warning = False  # Do not show the warning more than once

        return num_heads, hidden_size

    def get_add_qk_str(self, add_qk: NodeProto):
        if not self.shape_infer_done:
            self.shape_infer = self.model.infer_runtime_shape(update=True)
            self.shape_infer_done = True

        if self.shape_infer is None:
            return None

        input_0_shape = self.shape_infer.get_edge_shape(add_qk.input[0])
        input_1_shape = self.shape_infer.get_edge_shape(add_qk.input[1])

        if input_0_shape is None or input_1_shape is None:
            logger.debug("one of the inputs of %s is None", add_qk)
            return None

        if input_0_shape != input_1_shape:
            logger.debug("the shape of two inputs of %s is not same", add_qk)
            return None

        return add_qk.input[1]

    def reshape_add_qk(self, add_qk: str):
        # Convert 4D mask from (B,1,S,T) to (B,N,S,T)
        # B = batch size, N = num heads, S = source sequence length, T = target sequence length
        mask_output_name = add_qk + "_mask"

        # Check if concat node for (B,1,S,T) --> (B,N,S,T) already exists
        concat_node = list(filter(lambda node: node.output[0] == mask_output_name, self.nodes_to_add))
        if len(concat_node) == 1:
            return mask_output_name

        assert len(concat_node) == 0
        concat_node_name = self.model.create_node_name("Concat")
        concat_add_qk_fp32 = helper.make_node(
            "Concat",
            inputs=[add_qk for _ in range(self.num_heads)],
            outputs=[mask_output_name],
            name=concat_node_name,
            axis=1,
        )
        # Add new node to graph
        self.nodes_to_add.append(concat_add_qk_fp32)
        self.node_name_to_graph_name[concat_node_name] = self.this_graph_name

        return mask_output_name

    def concat_kv(self, past_k: str, past_v: str) -> str:
        """Concatenate past_k and past_v inputs to create past_kv input.

        Args:
            past_k (str): name of past K value
            past_v (str): name of past V value

        Returns:
            kv_output_name (str): name of past KV value
        """
        # Unsqueeze K and V nodes from (B,N,P,H) to (1,B,N,P,H)
        # B = batch size, N = num heads, P = past sequence length, H = head size
        unsqueeze_k_name = self.model.create_node_name("Unsqueeze")
        unsqueeze_v_name = self.model.create_node_name("Unsqueeze")
        k_5d_name = (past_k + "_5d").replace(".", "_")
        v_5d_name = (past_v + "_5d").replace(".", "_")

        k_5d = helper.make_node(
            "Unsqueeze",
            inputs=[past_k],
            outputs=[k_5d_name],
            name=unsqueeze_k_name,
            axes=[0],
        )
        v_5d = helper.make_node(
            "Unsqueeze",
            inputs=[past_v],
            outputs=[v_5d_name],
            name=unsqueeze_v_name,
            axes=[0],
        )

        # Add unsqueeze nodes to graph
        self.nodes_to_add.append(k_5d)
        self.nodes_to_add.append(v_5d)
        self.node_name_to_graph_name[unsqueeze_k_name] = self.this_graph_name
        self.node_name_to_graph_name[unsqueeze_v_name] = self.this_graph_name

        # Concat K and V to get one node of size (2,B,N,P,H)
        concat_node_name = self.model.create_node_name("Concat")
        kv_output_name = past_v.replace(".value", ".kv").replace(".", "_").replace("_value", "_kv")
        concat_kv = helper.make_node(
            "Concat",
            inputs=[k_5d_name, v_5d_name],
            outputs=[kv_output_name],
            name=concat_node_name,
            axis=0,
        )

        # Add concat node to graph
        self.nodes_to_add.append(concat_kv)
        self.node_name_to_graph_name[concat_node_name] = self.this_graph_name

        return kv_output_name

    def split_kv(self, present_k_name: str, present_v_name: str, kv_node: str):
        """Split kv_node containing present KV values into separate present K and present V values.

        Args:
            present_k_name (str): name of output to store present K value in
            present_v_name (str): name of output to store present V value in
            kv_node (str): name of present KV values
        """
        # Split kv_node into present_k and present_v nodes

        # Create initializers for indexing kv_node, whose shape is (2,B,N,P,H)
        k_index, v_index = "index_0", "index_1"
        k_dim = self.model.get_initializer(k_index)
        v_dim = self.model.get_initializer(v_index)
        if k_dim is None:
            k_dim = numpy_helper.from_array(np.array(0, dtype="int64"), name=k_index)
            self.model.add_initializer(k_dim, self.this_graph_name)
        if v_dim is None:
            v_dim = numpy_helper.from_array(np.array(1, dtype="int64"), name=v_index)
            self.model.add_initializer(v_dim, self.this_graph_name)

        # Create nodes to index kv_node
        gather_k_name = self.model.create_node_name("Gather")
        gather_v_name = self.model.create_node_name("Gather")
        present_k = helper.make_node(
            "Gather",
            inputs=[kv_node, k_index],
            outputs=[present_k_name],
            name=gather_k_name,
            axis=0,
        )
        present_v = helper.make_node(
            "Gather",
            inputs=[kv_node, v_index],
            outputs=[present_v_name],
            name=gather_v_name,
            axis=0,
        )

        # Add gather nodes to graph
        self.nodes_to_add.append(present_k)
        self.nodes_to_add.append(present_v)
        self.node_name_to_graph_name[gather_k_name] = self.this_graph_name
        self.node_name_to_graph_name[gather_v_name] = self.this_graph_name

    def create_combined_qkv_bias(
        self,
        q_add: NodeProto,
        k_add: NodeProto | None,
        v_add: NodeProto | None,
        name_prefix: str,
    ) -> NodeProto | None:
        q_bias = self.model.get_initializer(q_add.input[1]) or self.model.get_initializer(q_add.input[0])
        qb = NumpyHelper.to_array(q_bias)
        kb = np.zeros_like(qb)
        vb = np.zeros_like(qb)
        if k_add is not None:
            k_bias = self.model.get_initializer(k_add.input[1]) or self.model.get_initializer(k_add.input[0])
            kb = NumpyHelper.to_array(k_bias)
        if v_add is not None:
            v_bias = self.model.get_initializer(v_add.input[1]) or self.model.get_initializer(v_add.input[0])
            vb = NumpyHelper.to_array(v_bias)

        qkv_bias = np.stack((qb, kb, vb), axis=0)
        qkv_bias_dim = 3 * np.prod(qb.shape)

        bias_name = name_prefix + "_qkv_bias"
        self.add_initializer(
            name=bias_name,
            data_type=q_bias.data_type,
            dims=[qkv_bias_dim],
            vals=qkv_bias,
        )
        return bias_name

    def create_packed_qkv_matmul_node(
        self,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        q_add: NodeProto,
        k_add: NodeProto | None,
        v_add: NodeProto | None,
    ) -> tuple[NodeProto, NodeProto, NodeProto]:
        """Create packed QKV MatMul node before MultiHeadAttention node.
           This is for the scenario where an Attention node should be created but cannot be created
           because past_key and past_value are separate inputs and not one concatenated input.

        Args:
            q_matmul (NodeProto): name of MatMul from Q path - (batch_size, sequence_length, hidden_size)
            k_matmul (NodeProto): name of MatMul from K path - (batch_size, sequence_length, hidden_size)
            v_matmul (NodeProto): name of MatMul from V path - (batch_size, sequence_length, hidden_size)
            q_add (NodeProto): name of Add from Q path
            k_add (NodeProto): name of Add from K path
            v_add (NodeProto): name of Add from V path

        Returns:
             q_output (NodeProto): Slice node for Q
             k_output (NodeProto): Slice node for K
             v_output (NodeProto): Slice node for V
        """
        matmul_node_name = self.model.create_node_name("MatMul")

        # Check that input for Q, K, V is the same
        assert q_matmul.input[0] == k_matmul.input[0] and k_matmul.input[0] == v_matmul.input[0]

        # Created packed QKV weight
        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        assert qw.shape == kw.shape and kw.shape == vw.shape
        d = qw.shape[0]

        qkv_weight = np.stack((qw, kw, vw), axis=1).reshape((d, 3 * d))
        qkv_weight_name = matmul_node_name + "_qkv_weight"

        self.add_initializer(
            name=qkv_weight_name,
            data_type=q_weight.data_type,
            dims=[qkv_weight.shape[0], qkv_weight.shape[1]],
            vals=qkv_weight,
        )

        # Created packed QKV MatMul with output (B, S, 3*D)
        # Output is of the form:
        #
        # [[[Q Q ... Q Q K K ... K K V V ... V V]]]
        #   [Q Q ... Q Q K K ... K K V V ... V V]
        #                     .
        #                     .
        #                     .
        #  [[Q Q ... Q Q K K ... K K V V ... V V]
        #   [Q Q ... Q Q K K ... K K V V ... V V]]]
        qkv_matmul_output = matmul_node_name + "_qkv_out"
        qkv_matmul = helper.make_node(
            "MatMul",
            inputs=[q_matmul.input[0], qkv_weight_name],
            outputs=[qkv_matmul_output],
            name=matmul_node_name,
        )
        self.node_name_to_graph_name[matmul_node_name] = self.this_graph_name

        qkv_nodes = [qkv_matmul]

        # Create Slice nodes to access Q, K, V
        q_slice_name = matmul_node_name + "_q_start_index"
        self.add_initializer(name=q_slice_name, data_type=TensorProto.INT64, dims=[1], vals=[0], raw=False)
        k_slice_name = matmul_node_name + "_k_start_index"
        self.add_initializer(name=k_slice_name, data_type=TensorProto.INT64, dims=[1], vals=[d], raw=False)
        v_slice_name = matmul_node_name + "_v_start_index"
        self.add_initializer(name=v_slice_name, data_type=TensorProto.INT64, dims=[1], vals=[2 * d], raw=False)
        end_of_qkv_name = matmul_node_name + "_end_of_qkv_index"
        self.add_initializer(name=end_of_qkv_name, data_type=TensorProto.INT64, dims=[1], vals=[3 * d], raw=False)
        qkv_last_axis_name = matmul_node_name + "_qkv_last_axis"
        self.add_initializer(name=qkv_last_axis_name, data_type=TensorProto.INT64, dims=[1], vals=[-1], raw=False)

        q_slice_output = matmul_node_name + "_q_out"
        q_slice = helper.make_node(
            "Slice",
            inputs=[qkv_matmul_output, q_slice_name, k_slice_name, qkv_last_axis_name],
            outputs=[q_slice_output],
            name=self.model.create_node_name("Slice"),
        )
        self.node_name_to_graph_name[q_slice.name] = self.this_graph_name
        k_slice_output = matmul_node_name + "_k_out"
        k_slice = helper.make_node(
            "Slice",
            inputs=[qkv_matmul_output, k_slice_name, v_slice_name, qkv_last_axis_name],
            outputs=[k_slice_output],
            name=self.model.create_node_name("Slice"),
        )
        self.node_name_to_graph_name[k_slice.name] = self.this_graph_name
        v_slice_output = matmul_node_name + "_v_out"
        v_slice = helper.make_node(
            "Slice",
            inputs=[qkv_matmul_output, v_slice_name, end_of_qkv_name, qkv_last_axis_name],
            outputs=[v_slice_output],
            name=self.model.create_node_name("Slice"),
        )
        self.node_name_to_graph_name[v_slice.name] = self.this_graph_name

        q_output = q_slice
        k_output = k_slice
        v_output = v_slice
        qkv_nodes.extend([q_slice, k_slice, v_slice])

        if self.disable_multi_head_attention_bias:
            if q_add is not None:
                initializer_input = 1 if self.model.get_initializer(q_add.input[1]) else 0
                if np.any(NumpyHelper.to_array(self.model.get_initializer(q_add.input[initializer_input]))):
                    q_add.input[1 - initializer_input] = q_slice_output
                    q_output = q_add
                    qkv_nodes.append(q_add)
                    self.node_name_to_graph_name[q_add.name] = self.this_graph_name
            if k_add is not None:
                initializer_input = 1 if self.model.get_initializer(k_add.input[1]) else 0
                if np.any(NumpyHelper.to_array(self.model.get_initializer(k_add.input[initializer_input]))):
                    k_add.input[1 - initializer_input] = k_slice_output
                    k_output = k_add
                    qkv_nodes.append(k_add)
                    self.node_name_to_graph_name[k_add.name] = self.this_graph_name
            if v_add is not None:
                initializer_input = 1 if self.model.get_initializer(v_add.input[1]) else 0
                if np.any(NumpyHelper.to_array(self.model.get_initializer(v_add.input[initializer_input]))):
                    v_add.input[1 - initializer_input] = v_slice_output
                    v_output = v_add
                    qkv_nodes.append(v_add)
                    self.node_name_to_graph_name[v_add.name] = self.this_graph_name

        # Add nodes to graph
        self.nodes_to_add.extend(qkv_nodes)
        return q_output, k_output, v_output

    # This function is used in child classes for bart or conformer model.
    def create_multihead_attention_node(
        self,
        q_matmul: NodeProto,
        k_matmul: NodeProto | str | None,
        v_matmul: NodeProto | str | None,
        q_add: NodeProto,
        k_add: NodeProto | None,
        v_add: NodeProto | None,
        num_heads: int,
        hidden_size: int,
        output: str,
        key_padding_mask: str = "",
        add_qk: str = "",
        unidirectional: bool = False,
        past_k: str = "",
        past_v: str = "",
        present_k: str = "",
        present_v: str = "",
        packed_qkv: bool = False,
    ) -> NodeProto | None:
        """Create a MultiHeadAttention node.

        Args:
            q_matmul (NodeProto): name of MatMul from Q path - (batch_size, sequence_length, hidden_size)
            k_matmul (NodeProto): name of MatMul from K path - (batch_size, sequence_length, hidden_size) or (batch_size, num_heads, past_sequence_length, head_size)
            v_matmul (NodeProto): name of MatMul from V path - (batch_size, sequence_length, hidden_size) or (batch_size, num_heads, past_sequence_length, head_size)
            q_add (NodeProto): name of Add from Q path
            k_add (NodeProto): name of Add from K path
            v_add (NodeProto): name of Add from V path
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            output (str): output name of MHA
            key_padding_mask (str): name of key padding mask
            add_qk (str): name of add after Q x K'
            unidirectional (bool): whether to apply causal attention mask automatically or not
            past_k (str): name of past K value - (batch_size, num_heads, past_sequence_length, head_size)
            past_v (str): name of past V value - (batch_size, num_heads, past_sequence_length, head_size)
            present_k (str): name of present K value - (batch_size, num_heads, sequence_length, head_size)
            present_v (str): name of present V value - (batch_size, num_heads, sequence_length, head_size)
            packed_qkv (bool): whether to combine MatMuls from Q, K, V paths
                               Note: This is for the scenario where an Attention node should be created but cannot be created
                               because past_key and past_value are separate inputs and not one concatenated input.

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        # B = batch size, N = num heads, P = past seq len, H = head size
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug("input hidden size %d is not a multiple of num of heads %d", hidden_size, num_heads)
            return None

        graph_input_names = {node.name for node in self.model.graph().input}
        mha_node_name = self.model.create_node_name("Attention")

        # Add initial Q/K/V inputs for MHA
        mha_inputs = []
        if packed_qkv:
            q_slice, k_slice, v_slice = self.create_packed_qkv_matmul_node(
                q_matmul,
                k_matmul,
                v_matmul,
                q_add,
                k_add,
                v_add,
            )
            mha_inputs.extend([q_slice.output[0], k_slice.output[0], v_slice.output[0]])
        elif isinstance(k_matmul, NodeProto) and isinstance(v_matmul, NodeProto):
            if self.disable_multi_head_attention_bias:
                mha_inputs.extend([q_add.output[0], k_matmul.output[0], v_add.output[0]])
            else:
                mha_inputs.extend([q_matmul.output[0], k_matmul.output[0], v_matmul.output[0]])
        elif (
            isinstance(k_matmul, str)
            and isinstance(v_matmul, str)
            and k_matmul in graph_input_names
            and v_matmul in graph_input_names
        ):
            if self.disable_multi_head_attention_bias:
                mha_inputs.extend([q_add.output[0], k_matmul, v_matmul])
            else:
                mha_inputs.extend([q_matmul.output[0], k_matmul, v_matmul])
        else:
            return None

        # Add bias to inputs for MHA
        # Bias for cross attention is not fully supported in DMMHA and cpu MHA kernels since they assume
        # bias has been added to key and value when they are in BNSH format, so only bias for query is used.
        # Need add checks if we found such assumption is not true.
        if not self.disable_multi_head_attention_bias:
            bias_name = self.create_combined_qkv_bias(q_add, k_add, v_add, mha_node_name)
            mha_inputs.append(bias_name)
        else:
            mha_inputs.append("")

        # Add optional inputs for MHA
        if past_k and past_v:
            mha_inputs.extend([key_padding_mask, add_qk, past_k, past_v])
        elif key_padding_mask or add_qk:
            mha_inputs.extend([key_padding_mask, add_qk])

        # Add outputs for MHA
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
        mha_node.attribute.append(helper.make_attribute("num_heads", num_heads))
        if unidirectional:
            mha_node.attribute.append(helper.make_attribute("unidirectional", int(unidirectional)))

        self.increase_counter("MultiHeadAttention")
        return mha_node

    def create_attention_node(
        self,
        mask_index: str | None,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        q_add: NodeProto,
        k_add: NodeProto,
        v_add: NodeProto,
        num_heads: int,
        hidden_size: int,
        first_input: str,
        output: str,
        add_qk_str: str = "",
        past_k: str = "",
        past_v: str = "",
        present_k: str = "",
        present_v: str = "",
        scale: float | None = None,
        causal: bool = False,
    ) -> NodeProto | None:
        """Create an Attention node.

        Args:
            mask_index (str | None): mask input
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            q_add (NodeProto): Add bias node in fully connection for Q
            k_add (NodeProto): Add bias node in fully connection for K
            v_add (NodeProto): Add bias node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            first_input (str): first input name
            output (str): output name
            add_qk_str (str): name of Add node after Q x K'
            past_k (str): name of input for past K value
            past_v (str): name of input for past V value
            present_k (str): name of output to store present K value
            present_v (str): name of output to store present V value
            scale: scale before softmax
            causal: whether it is uni-directional mask.

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug("input hidden size %d is not a multiple of num of heads %d", hidden_size, num_heads)
            return None

        has_bias = True
        if q_add is None and k_add is None and v_add is None:
            has_bias = False

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])

        q_bias, k_bias, v_bias = None, None, None
        if has_bias:
            q_bias = self.model.get_initializer(q_add.input[1]) or self.model.get_initializer(q_add.input[0])
            k_bias = self.model.get_initializer(k_add.input[1]) or self.model.get_initializer(k_add.input[0])
            v_bias = self.model.get_initializer(v_add.input[1]) or self.model.get_initializer(v_add.input[0])

            if not (k_weight and v_weight and q_bias and k_bias):
                return None

        if q_weight is None:
            print(
                f"{q_matmul.input[1]} is not an initializer. "
                "Please set do_constant_folding=True in torch.onnx.export to unblock attention fusion"
            )
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        # assert q and k have same shape as expected
        assert qw.shape == kw.shape

        qw_in_size = qw.shape[0]
        kw_in_size = kw.shape[0]
        vw_in_size = vw.shape[0]

        assert qw_in_size == kw_in_size == vw_in_size

        if hidden_size > 0 and hidden_size != qw_in_size:
            logger.warning(
                "Input hidden size (%d) is not same as weight matrix dimension of q,k,v (%d). "
                "Please provide a correct input hidden size or pass in 0",
                hidden_size,
                qw_in_size,
            )

        is_qkv_diff_dims = False
        if qw.shape != vw.shape:
            is_qkv_diff_dims = True

        # All the matrices can have the same shape or q, k matrices can have the same shape with v being different
        # For 2d weights, the shapes would be [in_size, out_size].
        # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
        qw_out_size = np.prod(qw.shape[1:])
        kw_out_size = np.prod(kw.shape[1:])
        vw_out_size = np.prod(vw.shape[1:])

        qkv_weight_dim = 0
        if is_qkv_diff_dims:
            qkv_weight = np.concatenate((qw, kw, vw), axis=1)
            qkv_weight_dim = qw_out_size + kw_out_size + vw_out_size
        else:
            qkv_weight = np.stack((qw, kw, vw), axis=1)
            qkv_weight_dim = 3 * qw_out_size

        qkv_bias_dim = 0
        qkv_bias: np.ndarray | None = None
        if has_bias:
            qb = NumpyHelper.to_array(q_bias)
            kb = NumpyHelper.to_array(k_bias)
            vb = NumpyHelper.to_array(v_bias)

            q_bias_shape = np.prod(qb.shape)
            k_bias_shape = np.prod(kb.shape)
            v_bias_shape = np.prod(vb.shape)

            assert q_bias_shape == k_bias_shape == qw_out_size
            assert v_bias_shape == vw_out_size

            if is_qkv_diff_dims:
                qkv_bias = np.concatenate((qb, kb, vb), axis=0)
                qkv_bias_dim = q_bias_shape + k_bias_shape + v_bias_shape
            else:
                qkv_bias = np.stack((qb, kb, vb), axis=0)
                qkv_bias_dim = 3 * q_bias_shape

        attention_node_name = self.model.create_node_name("Attention")

        if not self.use_multi_head_attention:
            self.add_initializer(
                name=attention_node_name + "_qkv_weight",
                data_type=q_weight.data_type,
                dims=[qw_in_size, int(qkv_weight_dim)],
                vals=qkv_weight,
            )

        if has_bias:
            self.add_initializer(
                name=attention_node_name + "_qkv_bias",
                data_type=q_bias.data_type,
                dims=[int(qkv_bias_dim)],
                vals=qkv_bias,
            )

        # For MultiHeadAttention operator, use separated inputs for query, key and value, and no weights.
        if self.use_multi_head_attention:
            if add_qk_str:
                logger.debug("MultiHeadAttention does not support relative_position_bias: cannot fuse the attention.")
                return None

            attention_inputs = [
                q_matmul.output[0],
                k_matmul.output[0],
                v_matmul.output[0],
                attention_node_name + "_qkv_bias",
            ]

            if mask_index is not None:
                attention_inputs.append(mask_index)

            attention_node = helper.make_node(
                "MultiHeadAttention",
                inputs=attention_inputs,
                outputs=[output],
                name=attention_node_name,
            )
            self.increase_counter("MultiHeadAttention")

        else:
            attention_inputs = [
                first_input,
                attention_node_name + "_qkv_weight",
                attention_node_name + "_qkv_bias" if has_bias else "",
            ]
            if mask_index is not None:
                attention_inputs.append(mask_index)
            else:
                attention_inputs.append("")

            past_exists = past_k and past_v
            if past_exists:
                past_kv = self.concat_kv(past_k, past_v)
                attention_inputs.append(past_kv)

            if add_qk_str:
                # Add additional add to attention node (input name = attention_bias)
                if not past_exists:
                    attention_inputs.append("")
                attention_inputs.append(add_qk_str)

            attention_outputs = [output]
            if present_k and present_v:
                present_kv = present_k.replace(".key", "").replace("_key", "").replace(".", "_")
                attention_outputs.append(present_kv)
                self.split_kv(present_k, present_v, present_kv)

            attention_node = helper.make_node(
                "Attention",
                inputs=attention_inputs,
                outputs=attention_outputs,
                name=attention_node_name,
            )
            self.increase_counter("Attention")

        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        if causal:
            attention_node.attribute.extend([helper.make_attribute("unidirectional", 1)])

        if scale is not None:
            attention_node.attribute.extend([helper.make_attribute("scale", scale)])

        if is_qkv_diff_dims:
            attention_node.attribute.extend(
                [helper.make_attribute("qkv_hidden_sizes", [qw_out_size, kw_out_size, vw_out_size])]
            )

        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        return attention_node

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        normalize_node = node
        start_node = normalize_node
        if normalize_node.op_type == "LayerNormalization":
            add_before_layernorm = self.model.match_parent(normalize_node, "Add", 0)
            if add_before_layernorm is not None:
                start_node = add_before_layernorm
            else:
                return

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            start_node,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [None, None, 0, 0, 0],
        )
        einsum_node = None
        if qkv_nodes is not None:
            (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
        else:
            # Match Albert
            qkv_nodes = self.model.match_parent_path(
                start_node, ["Add", "Einsum", "Transpose", "MatMul"], [1, None, 0, 0]
            )
            if qkv_nodes is not None:
                (_, einsum_node, transpose_qkv, matmul_qkv) = qkv_nodes
            else:
                return

        other_inputs = []
        for _i, node_input in enumerate(start_node.input):
            if node_input not in output_name_to_node:
                continue

            if node_input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(node_input)
        if len(other_inputs) != 1:
            return

        root_input = other_inputs[0]

        # Match flaubert                     Mask
        #                                     |
        # Mul --> LayerNormalization -->  Attention --> MatMul --> Add
        #  |                                                        |
        #  |                                                        |
        #  +---------------------------------------------------------
        mul_before_layernorm = self.model.match_parent(start_node, "Mul", 0)
        if mul_before_layernorm is not None:
            mul_children = input_name_to_nodes[mul_before_layernorm.output[0]]
            if mul_children is not None and len(mul_children) == 2:
                layernorm_node = mul_children[1]
                if layernorm_node.op_type == "LayerNormalization":
                    root_input = layernorm_node.output[0]
                else:
                    return
            elif mul_children is not None and len(mul_children) == 5:
                root_input = mul_before_layernorm.output[0]
            else:
                return
        elif normalize_node.op_type == "LayerNormalization":
            children = input_name_to_nodes[root_input]
            for child in children:
                if child.op_type == "LayerNormalization":
                    root_input = child.output[0]

        # When Add before the LayerNormalization produces an output
        # that is consumed by some other nodes other than the LayerNormalization itself,
        # fused SkipLayerNormalization will have several outputs.
        # In this case we need to pick the one used in Attention
        # For example, this is the case for ViT
        # SkipLayerNormalization --> Attention --> MatMul --> Add --> SkipLayerNormalization
        #  |                                                                     |
        #  |                                                                     |
        #  +---------------------------------------------------------------------+
        parent_node = output_name_to_node[root_input]
        if parent_node.op_type == "SkipLayerNormalization" and len(parent_node.output) == 4:
            root_input = parent_node.output[0]

        children = input_name_to_nodes[root_input]
        children_types = [child.op_type for child in children]
        if children_types.count("MatMul") != 3:
            return

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (_, _, add_v, matmul_v) = v_nodes

        is_distill = False
        is_distill_add = False
        is_no_mask_attention = False
        is_sdpa = False
        qk_paths = {
            "path1": (["Softmax", "Add", "Div", "MatMul"], [0, 0, None, 0]),
            "path2": (["Softmax", "Add", "Mul", "MatMul"], [0, 0, None, 0]),
            "path3": (["Softmax", "Where", "MatMul", "Div"], [0, 0, 2, 0]),
            "path4": (["Softmax", "Add", "Where", "MatMul"], [0, 0, 0, 2]),
            "path5": (["Softmax", "Div", "MatMul"], [0, 0, 0]),
            "sdpa": (["Softmax", "Add", "MatMul", "Mul", "Sqrt"], [0, 0, None, 0, 1]),
        }

        qk_nodes = None
        for k, v in qk_paths.items():
            qk_nodes = self.model.match_parent_path(matmul_qkv, v[0], v[1])
            if qk_nodes is None:
                continue
            if k == "path3":
                is_distill = True
            elif k == "path4":
                is_distill_add = True
            elif k == "path5":
                is_no_mask_attention = True
            elif k == "sdpa":
                is_sdpa = True
            break

        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return

        add_qk = None
        matmul_qk = None
        where_qk = None
        after_q = None
        if is_distill:
            (_, where_qk, matmul_qk, _) = qk_nodes
        elif is_distill_add:
            (_, add_qk, where_qk, matmul_qk) = qk_nodes
        elif is_no_mask_attention:
            (_, _, matmul_qk) = qk_nodes
        elif is_sdpa:
            (_, add_qk, matmul_qk, after_q, _) = qk_nodes
        else:
            (_, add_qk, _, matmul_qk) = qk_nodes

        after_q = after_q or matmul_qk
        q_nodes = self.model.match_parent_path(after_q, ["Transpose", "Reshape", "Add", "MatMul"], [0, 0, 0, None])
        if q_nodes is None:
            q_nodes = self.model.match_parent_path(
                after_q,
                ["Div", "Transpose", "Reshape", "Add", "MatMul"],
                [0, 0, 0, 0, None],
            )
            if q_nodes is None:
                logger.debug("fuse_attention: failed to match q path")
                return
        reshape_q = q_nodes[-3]
        add_q = q_nodes[-2]
        matmul_q = q_nodes[-1]

        after_k = matmul_qk
        if is_sdpa:
            mul_k_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Sqrt"], [1, None])
            if mul_k_nodes is None:
                logger.debug("fuse_attention: failed to match mul sqrt q path")
                return
            (after_k, _) = mul_k_nodes

        k_nodes = self.model.match_parent_path(
            after_k, ["Transpose", "Reshape", "Add", "MatMul"], [0 if is_sdpa else 1, 0, 0, None]
        )
        if k_nodes is None:
            k_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Transpose", "Transpose", "Reshape", "Add", "MatMul"],
                [1, 0, 0, 0, None],
            )
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                return
        add_k = k_nodes[-2]
        matmul_k = k_nodes[-1]

        # Note that Cast might be removed by OnnxRuntime so we match two patterns here.
        mask_nodes = None
        add_qk_str = ""
        if is_distill:
            _, mask_nodes, _ = self.model.match_parent_paths(
                where_qk,
                [
                    (["Expand", "Reshape", "Equal"], [0, 0, 0]),
                    (["Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0]),
                    (["Cast", "Expand", "Reshape", "Equal"], [0, 0, 0, 0]),
                ],
                output_name_to_node,
            )
        elif is_distill_add:
            _, mask_nodes, _ = self.model.match_parent_paths(
                where_qk,
                [
                    (["Cast", "Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0, 0]),
                    (["Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0]),
                ],
                output_name_to_node,
            )
            if add_qk is not None:
                add_qk_str = self.get_add_qk_str(add_qk)
                if add_qk_str is None:
                    logger.debug("fuse_attention: failed to verify shape inference of %s", add_qk)
                    return
        elif is_no_mask_attention:
            pass
        else:
            _, mask_nodes, _ = self.model.match_parent_paths(
                add_qk,
                [
                    (["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"], [None, 0, 1, 0, 0]),
                    (["Mul", "Sub", "Unsqueeze", "Unsqueeze"], [None, 0, 1, 0]),
                    # The following two patterns are for SDPA.
                    (["Where", "Cast", "Sub", "Expand", "Unsqueeze", "Unsqueeze"], [None, 0, 0, 1, 0, 0]),
                    (["Where", "Cast", "Sub", "Cast", "Expand", "Unsqueeze", "Unsqueeze"], [None, 0, 0, 1, 0, 0, 0]),
                ],
                output_name_to_node,
            )
        if not is_no_mask_attention and mask_nodes is None:
            logger.debug("fuse_attention: failed to match mask path")
            return

        if not is_no_mask_attention and len(mask_nodes) > 1:
            _, mul_val = self.model.get_constant_input(mask_nodes[0])
            # The mask value shall be a float scalar (usually is the lowest float value).
            if (
                (mul_val is None)
                or not (isinstance(mul_val, np.ndarray) and mul_val.size == 1)
                or (float(mul_val) >= 0)
            ):
                return
            if float(mul_val) != -10000:
                self.mask_filter_value = float(mul_val)

        if matmul_v.input[0] == root_input and matmul_q.input[0] == root_input and matmul_k.input[0] == root_input:
            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0]) if not is_no_mask_attention else None

            attention_last_node = reshape_qkv if einsum_node is None else transpose_qkv

            q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)
            if q_num_heads <= 0 or q_hidden_size <= 0:
                logger.warning(
                    "Failed to detect num_heads and hidden_size for Attention fusion. "
                    "Please specify those parameters in argument."
                )
                return

            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            # the input_hidden_size represents the input hidden size, this is used as needed but hidden sizes for Q, K are extracted appropriately
            new_node = self.create_attention_node(
                mask_index=mask_index,
                q_matmul=matmul_q,
                k_matmul=matmul_k,
                v_matmul=matmul_v,
                q_add=add_q,
                k_add=add_k,
                v_add=add_v,
                num_heads=q_num_heads,
                hidden_size=q_hidden_size,
                first_input=root_input,
                output=attention_last_node.output[0],
                add_qk_str=add_qk_str,
            )

            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            if einsum_node is not None:
                unique_index = einsum_node.input[0]
                new_edge = "edge_modified_" + unique_index

                shape_tensor = self.add_initializer(
                    name="shape_modified_tensor" + unique_index,
                    data_type=TensorProto.INT64,
                    dims=[4],
                    vals=[0, 0, q_num_heads, int(q_hidden_size / q_num_heads)],
                    raw=False,
                )

                self.model.add_node(
                    helper.make_node(
                        "Reshape",
                        [attention_last_node.output[0], shape_tensor.name],
                        [new_edge],
                        "reshape_modified_" + unique_index,
                    ),
                    self.this_graph_name,
                )
                einsum_node.input[0] = new_edge

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)

            # For MultiHeadAttention operator, MatMul nodes for Q/K/V projection shall not be fused.
            self.nodes_to_remove.extend(q_nodes if not self.use_multi_head_attention else q_nodes[:-1])
            self.nodes_to_remove.extend(k_nodes if not self.use_multi_head_attention else k_nodes[:-1])
            self.nodes_to_remove.extend(v_nodes if not self.use_multi_head_attention else v_nodes[:-1])

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            self.prune_graph = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\tensor_can.py ===
from sympy.combinatorics.permutations import Permutation, _af_rmul, \
    _af_invert, _af_new
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
    _orbit_transversal
from sympy.combinatorics.util import _distribute_gens_by_base, \
    _orbits_transversals_from_bsgs

"""
    References for tensor canonicalization:

    [1] R. Portugal "Algorithmic simplification of tensor expressions",
        J. Phys. A 32 (1999) 7779-7789

    [2] R. Portugal, B.F. Svaiter "Group-theoretic Approach for Symbolic
        Tensor Manipulation: I. Free Indices"
        arXiv:math-ph/0107031v1

    [3] L.R.U. Manssur, R. Portugal "Group-theoretic Approach for Symbolic
        Tensor Manipulation: II. Dummy Indices"
        arXiv:math-ph/0107032v1

    [4] xperm.c part of XPerm written by J. M. Martin-Garcia
        http://www.xact.es/index.html
"""


def dummy_sgs(dummies, sym, n):
    """
    Return the strong generators for dummy indices.

    Parameters
    ==========

    dummies : List of dummy indices.
        `dummies[2k], dummies[2k+1]` are paired indices.
        In base form, the dummy indices are always in
        consecutive positions.
    sym : symmetry under interchange of contracted dummies::
        * None  no symmetry
        * 0     commuting
        * 1     anticommuting

    n : number of indices

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import dummy_sgs
    >>> dummy_sgs(list(range(2, 8)), 0, 8)
    [[0, 1, 3, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 5, 4, 6, 7, 8, 9],
     [0, 1, 2, 3, 4, 5, 7, 6, 8, 9], [0, 1, 4, 5, 2, 3, 6, 7, 8, 9],
     [0, 1, 2, 3, 6, 7, 4, 5, 8, 9]]
    """
    if len(dummies) > n:
        raise ValueError("List too large")
    res = []
    # exchange of contravariant and covariant indices
    if sym is not None:
        for j in dummies[::2]:
            a = list(range(n + 2))
            if sym == 1:
                a[n] = n + 1
                a[n + 1] = n
            a[j], a[j + 1] = a[j + 1], a[j]
            res.append(a)
    # rename dummy indices
    for j in dummies[:-3:2]:
        a = list(range(n + 2))
        a[j:j + 4] = a[j + 2], a[j + 3], a[j], a[j + 1]
        res.append(a)
    return res


def _min_dummies(dummies, sym, indices):
    """
    Return list of minima of the orbits of indices in group of dummies.
    See ``double_coset_can_rep`` for the description of ``dummies`` and ``sym``.
    ``indices`` is the initial list of dummy indices.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import _min_dummies
    >>> _min_dummies([list(range(2, 8))], [0], list(range(10)))
    [0, 1, 2, 2, 2, 2, 2, 2, 8, 9]
    """
    num_types = len(sym)
    m = [min(dx) if dx else None for dx in dummies]
    res = indices[:]
    for i in range(num_types):
        for c, i in enumerate(indices):
            for j in range(num_types):
                if i in dummies[j]:
                    res[c] = m[j]
                    break
    return res


def _trace_S(s, j, b, S_cosets):
    """
    Return the representative h satisfying s[h[b]] == j

    If there is not such a representative return None
    """
    for h in S_cosets[b]:
        if s[h[b]] == j:
            return h
    return None


def _trace_D(gj, p_i, Dxtrav):
    """
    Return the representative h satisfying h[gj] == p_i

    If there is not such a representative return None
    """
    for h in Dxtrav:
        if h[gj] == p_i:
            return h
    return None


def _dumx_remove(dumx, dumx_flat, p0):
    """
    remove p0 from dumx
    """
    res = []
    for dx in dumx:
        if p0 not in dx:
            res.append(dx)
            continue
        k = dx.index(p0)
        if k % 2 == 0:
            p0_paired = dx[k + 1]
        else:
            p0_paired = dx[k - 1]
        dx.remove(p0)
        dx.remove(p0_paired)
        dumx_flat.remove(p0)
        dumx_flat.remove(p0_paired)
        res.append(dx)


def transversal2coset(size, base, transversal):
    a = []
    j = 0
    for i in range(size):
        if i in base:
            a.append(sorted(transversal[j].values()))
            j += 1
        else:
            a.append([list(range(size))])
    j = len(a) - 1
    while a[j] == [list(range(size))]:
        j -= 1
    return a[:j + 1]


def double_coset_can_rep(dummies, sym, b_S, sgens, S_transversals, g):
    r"""
    Butler-Portugal algorithm for tensor canonicalization with dummy indices.

    Parameters
    ==========

      dummies
        list of lists of dummy indices,
        one list for each type of index;
        the dummy indices are put in order contravariant, covariant
        [d0, -d0, d1, -d1, ...].

      sym
        list of the symmetries of the index metric for each type.

      possible symmetries of the metrics
              * 0     symmetric
              * 1     antisymmetric
              * None  no symmetry

      b_S
        base of a minimal slot symmetry BSGS.

      sgens
        generators of the slot symmetry BSGS.

      S_transversals
        transversals for the slot BSGS.

      g
        permutation representing the tensor.

    Returns
    =======

    Return 0 if the tensor is zero, else return the array form of
    the permutation representing the canonical form of the tensor.

    Notes
    =====

    A tensor with dummy indices can be represented in a number
    of equivalent ways which typically grows exponentially with
    the number of indices. To be able to establish if two tensors
    with many indices are equal becomes computationally very slow
    in absence of an efficient algorithm.

    The Butler-Portugal algorithm [3] is an efficient algorithm to
    put tensors in canonical form, solving the above problem.

    Portugal observed that a tensor can be represented by a permutation,
    and that the class of tensors equivalent to it under slot and dummy
    symmetries is equivalent to the double coset `D*g*S`
    (Note: in this documentation we use the conventions for multiplication
    of permutations p, q with (p*q)(i) = p[q[i]] which is opposite
    to the one used in the Permutation class)

    Using the algorithm by Butler to find a representative of the
    double coset one can find a canonical form for the tensor.

    To see this correspondence,
    let `g` be a permutation in array form; a tensor with indices `ind`
    (the indices including both the contravariant and the covariant ones)
    can be written as

    `t = T(ind[g[0]], \dots, ind[g[n-1]])`,

    where `n = len(ind)`;
    `g` has size `n + 2`, the last two indices for the sign of the tensor
    (trick introduced in [4]).

    A slot symmetry transformation `s` is a permutation acting on the slots
    `t \rightarrow T(ind[(g*s)[0]], \dots, ind[(g*s)[n-1]])`

    A dummy symmetry transformation acts on `ind`
    `t \rightarrow T(ind[(d*g)[0]], \dots, ind[(d*g)[n-1]])`

    Being interested only in the transformations of the tensor under
    these symmetries, one can represent the tensor by `g`, which transforms
    as

    `g -> d*g*s`, so it belongs to the coset `D*g*S`, or in other words
    to the set of all permutations allowed by the slot and dummy symmetries.

    Let us explain the conventions by an example.

    Given a tensor `T^{d3 d2 d1}{}_{d1 d2 d3}` with the slot symmetries
          `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`

          `T^{a0 a1 a2 a3 a4 a5} = -T^{a4 a1 a2 a3 a0 a5}`

    and symmetric metric, find the tensor equivalent to it which
    is the lowest under the ordering of indices:
    lexicographic ordering `d1, d2, d3` and then contravariant
    before covariant index; that is the canonical form of the tensor.

    The canonical form is `-T^{d1 d2 d3}{}_{d1 d2 d3}`
    obtained using `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`.

    To convert this problem in the input for this function,
    use the following ordering of the index names
    (- for covariant for short) `d1, -d1, d2, -d2, d3, -d3`

    `T^{d3 d2 d1}{}_{d1 d2 d3}` corresponds to `g = [4, 2, 0, 1, 3, 5, 6, 7]`
    where the last two indices are for the sign

    `sgens = [Permutation(0, 2)(6, 7), Permutation(0, 4)(6, 7)]`

    sgens[0] is the slot symmetry `-(0, 2)`
    `T^{a0 a1 a2 a3 a4 a5} = -T^{a2 a1 a0 a3 a4 a5}`

    sgens[1] is the slot symmetry `-(0, 4)`
    `T^{a0 a1 a2 a3 a4 a5} = -T^{a4 a1 a2 a3 a0 a5}`

    The dummy symmetry group D is generated by the strong base generators
    `[(0, 1), (2, 3), (4, 5), (0, 2)(1, 3), (0, 4)(1, 5)]`
    where the first three interchange covariant and contravariant
    positions of the same index (d1 <-> -d1) and the last two interchange
    the dummy indices themselves (d1 <-> d2).

    The dummy symmetry acts from the left
    `d = [1, 0, 2, 3, 4, 5, 6, 7]`  exchange `d1 \leftrightarrow -d1`
    `T^{d3 d2 d1}{}_{d1 d2 d3} == T^{d3 d2}{}_{d1}{}^{d1}{}_{d2 d3}`

    `g=[4, 2, 0, 1, 3, 5, 6, 7]  -> [4, 2, 1, 0, 3, 5, 6, 7] = _af_rmul(d, g)`
    which differs from `_af_rmul(g, d)`.

    The slot symmetry acts from the right
    `s = [2, 1, 0, 3, 4, 5, 7, 6]`  exchanges slots 0 and 2 and changes sign
    `T^{d3 d2 d1}{}_{d1 d2 d3} == -T^{d1 d2 d3}{}_{d1 d2 d3}`

    `g=[4,2,0,1,3,5,6,7]  -> [0, 2, 4, 1, 3, 5, 7, 6] = _af_rmul(g, s)`

    Example in which the tensor is zero, same slot symmetries as above:
    `T^{d2}{}_{d1 d3}{}^{d1 d3}{}_{d2}`

    `= -T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}`   under slot symmetry `-(0,4)`;

    `= T_{d3 d1}{}^{d3}{}^{d1 d2}{}_{d2}`    under slot symmetry `-(0,2)`;

    `= T^{d3}{}_{d1 d3}{}^{d1 d2}{}_{d2}`    symmetric metric;

    `= 0`  since two of these lines have tensors differ only for the sign.

    The double coset D*g*S consists of permutations `h = d*g*s` corresponding
    to equivalent tensors; if there are two `h` which are the same apart
    from the sign, return zero; otherwise
    choose as representative the tensor with indices
    ordered lexicographically according to `[d1, -d1, d2, -d2, d3, -d3]`
    that is ``rep = min(D*g*S) = min([d*g*s for d in D for s in S])``

    The indices are fixed one by one; first choose the lowest index
    for slot 0, then the lowest remaining index for slot 1, etc.
    Doing this one obtains a chain of stabilizers

    `S \rightarrow S_{b0} \rightarrow S_{b0,b1} \rightarrow \dots` and
    `D \rightarrow D_{p0} \rightarrow D_{p0,p1} \rightarrow \dots`

    where ``[b0, b1, ...] = range(b)`` is a base of the symmetric group;
    the strong base `b_S` of S is an ordered sublist of it;
    therefore it is sufficient to compute once the
    strong base generators of S using the Schreier-Sims algorithm;
    the stabilizers of the strong base generators are the
    strong base generators of the stabilizer subgroup.

    ``dbase = [p0, p1, ...]`` is not in general in lexicographic order,
    so that one must recompute the strong base generators each time;
    however this is trivial, there is no need to use the Schreier-Sims
    algorithm for D.

    The algorithm keeps a TAB of elements `(s_i, d_i, h_i)`
    where `h_i = d_i \times g \times s_i` satisfying `h_i[j] = p_j` for `0 \le j < i`
    starting from `s_0 = id, d_0 = id, h_0 = g`.

    The equations `h_0[0] = p_0, h_1[1] = p_1, \dots` are solved in this order,
    choosing each time the lowest possible value of p_i

    For `j < i`
    `d_i*g*s_i*S_{b_0, \dots, b_{i-1}}*b_j = D_{p_0, \dots, p_{i-1}}*p_j`
    so that for dx in `D_{p_0,\dots,p_{i-1}}` and sx in
    `S_{base[0], \dots, base[i-1]}` one has `dx*d_i*g*s_i*sx*b_j = p_j`

    Search for dx, sx such that this equation holds for `j = i`;
    it can be written as `s_i*sx*b_j = J, dx*d_i*g*J = p_j`
    `sx*b_j = s_i**-1*J; sx = trace(s_i**-1, S_{b_0,...,b_{i-1}})`
    `dx**-1*p_j = d_i*g*J; dx = trace(d_i*g*J, D_{p_0,...,p_{i-1}})`

    `s_{i+1} = s_i*trace(s_i**-1*J, S_{b_0,...,b_{i-1}})`
    `d_{i+1} = trace(d_i*g*J, D_{p_0,...,p_{i-1}})**-1*d_i`
    `h_{i+1}*b_i = d_{i+1}*g*s_{i+1}*b_i = p_i`

    `h_n*b_j = p_j` for all j, so that `h_n` is the solution.

    Add the found `(s, d, h)` to TAB1.

    At the end of the iteration sort TAB1 with respect to the `h`;
    if there are two consecutive `h` in TAB1 which differ only for the
    sign, the tensor is zero, so return 0;
    if there are two consecutive `h` which are equal, keep only one.

    Then stabilize the slot generators under `i` and the dummy generators
    under `p_i`.

    Assign `TAB = TAB1` at the end of the iteration step.

    At the end `TAB` contains a unique `(s, d, h)`, since all the slots
    of the tensor `h` have been fixed to have the minimum value according
    to the symmetries. The algorithm returns `h`.

    It is important that the slot BSGS has lexicographic minimal base,
    otherwise there is an `i` which does not belong to the slot base
    for which `p_i` is fixed by the dummy symmetry only, while `i`
    is not invariant from the slot stabilizer, so `p_i` is not in
    general the minimal value.

    This algorithm differs slightly from the original algorithm [3]:
      the canonical form is minimal lexicographically, and
      the BSGS has minimal base under lexicographic order.
      Equal tensors `h` are eliminated from TAB.


    Examples
    ========

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.combinatorics.tensor_can import double_coset_can_rep, get_transversals
    >>> gens = [Permutation(x) for x in [[2, 1, 0, 3, 4, 5, 7, 6], [4, 1, 2, 3, 0, 5, 7, 6]]]
    >>> base = [0, 2]
    >>> g = Permutation([4, 2, 0, 1, 3, 5, 6, 7])
    >>> transversals = get_transversals(base, gens)
    >>> double_coset_can_rep([list(range(6))], [0], base, gens, transversals, g)
    [0, 1, 2, 3, 4, 5, 7, 6]

    >>> g = Permutation([4, 1, 3, 0, 5, 2, 6, 7])
    >>> double_coset_can_rep([list(range(6))], [0], base, gens, transversals, g)
    0
    """
    size = g.size
    g = g.array_form
    num_dummies = size - 2
    indices = list(range(num_dummies))
    all_metrics_with_sym = not any(_ is None for _ in sym)
    num_types = len(sym)
    dumx = dummies[:]
    dumx_flat = []
    for dx in dumx:
        dumx_flat.extend(dx)
    b_S = b_S[:]
    sgensx = [h._array_form for h in sgens]
    if b_S:
        S_transversals = transversal2coset(size, b_S, S_transversals)
    # strong generating set for D
    dsgsx = []
    for i in range(num_types):
        dsgsx.extend(dummy_sgs(dumx[i], sym[i], num_dummies))
    idn = list(range(size))
    # TAB = list of entries (s, d, h) where h = _af_rmuln(d,g,s)
    # for short, in the following d*g*s means _af_rmuln(d,g,s)
    TAB = [(idn, idn, g)]
    for i in range(size - 2):
        b = i
        testb = b in b_S and sgensx
        if testb:
            sgensx1 = [_af_new(_) for _ in sgensx]
            deltab = _orbit(size, sgensx1, b)
        else:
            deltab = {b}
        # p1 = min(IMAGES) = min(Union D_p*h*deltab for h in TAB)
        if all_metrics_with_sym:
            md = _min_dummies(dumx, sym, indices)
        else:
            md = [min(_orbit(size, [_af_new(
                ddx) for ddx in dsgsx], ii)) for ii in range(size - 2)]

        p_i = min(min(md[h[x]] for x in deltab) for s, d, h in TAB)
        dsgsx1 = [_af_new(_) for _ in dsgsx]
        Dxtrav = _orbit_transversal(size, dsgsx1, p_i, False, af=True) \
            if dsgsx else None
        if Dxtrav:
            Dxtrav = [_af_invert(x) for x in Dxtrav]
        # compute the orbit of p_i
        for ii in range(num_types):
            if p_i in dumx[ii]:
                # the orbit is made by all the indices in dum[ii]
                if sym[ii] is not None:
                    deltap = dumx[ii]
                else:
                    # the orbit is made by all the even indices if p_i
                    # is even, by all the odd indices if p_i is odd
                    p_i_index = dumx[ii].index(p_i) % 2
                    deltap = dumx[ii][p_i_index::2]
                break
        else:
            deltap = [p_i]
        TAB1 = []
        while TAB:
            s, d, h = TAB.pop()
            if min(md[h[x]] for x in deltab) != p_i:
                continue
            deltab1 = [x for x in deltab if md[h[x]] == p_i]
            # NEXT = s*deltab1 intersection (d*g)**-1*deltap
            dg = _af_rmul(d, g)
            dginv = _af_invert(dg)
            sdeltab = [s[x] for x in deltab1]
            gdeltap = [dginv[x] for x in deltap]
            NEXT = [x for x in sdeltab if x in gdeltap]
            # d, s satisfy
            # d*g*s*base[i-1] = p_{i-1}; using the stabilizers
            # d*g*s*S_{base[0],...,base[i-1]}*base[i-1] =
            # D_{p_0,...,p_{i-1}}*p_{i-1}
            # so that to find d1, s1 satisfying d1*g*s1*b = p_i
            # one can look for dx in D_{p_0,...,p_{i-1}} and
            # sx in S_{base[0],...,base[i-1]}
            # d1 = dx*d; s1 = s*sx
            # d1*g*s1*b = dx*d*g*s*sx*b = p_i
            for j in NEXT:
                if testb:
                    # solve s1*b = j with s1 = s*sx for some element sx
                    # of the stabilizer of ..., base[i-1]
                    # sx*b = s**-1*j; sx = _trace_S(s, j,...)
                    # s1 = s*trace_S(s**-1*j,...)
                    s1 = _trace_S(s, j, b, S_transversals)
                    if not s1:
                        continue
                    else:
                        s1 = [s[ix] for ix in s1]
                else:
                    s1 = s
                # assert s1[b] == j  # invariant
                # solve d1*g*j = p_i with d1 = dx*d for some element dg
                # of the stabilizer of ..., p_{i-1}
                # dx**-1*p_i = d*g*j; dx**-1 = trace_D(d*g*j,...)
                # d1 = trace_D(d*g*j,...)**-1*d
                # to save an inversion in the inner loop; notice we did
                # Dxtrav = [perm_af_invert(x) for x in Dxtrav] out of the loop
                if Dxtrav:
                    d1 = _trace_D(dg[j], p_i, Dxtrav)
                    if not d1:
                        continue
                else:
                    if p_i != dg[j]:
                        continue
                    d1 = idn
                assert d1[dg[j]] == p_i  # invariant
                d1 = [d1[ix] for ix in d]
                h1 = [d1[g[ix]] for ix in s1]
                # assert h1[b] == p_i  # invariant
                TAB1.append((s1, d1, h1))

        # if TAB contains equal permutations, keep only one of them;
        # if TAB contains equal permutations up to the sign, return 0
        TAB1.sort(key=lambda x: x[-1])
        prev = [0] * size
        while TAB1:
            s, d, h = TAB1.pop()
            if h[:-2] == prev[:-2]:
                if h[-1] != prev[-1]:
                    return 0
            else:
                TAB.append((s, d, h))
            prev = h

        # stabilize the SGS
        sgensx = [h for h in sgensx if h[b] == b]
        if b in b_S:
            b_S.remove(b)
        _dumx_remove(dumx, dumx_flat, p_i)
        dsgsx = []
        for i in range(num_types):
            dsgsx.extend(dummy_sgs(dumx[i], sym[i], num_dummies))
    return TAB[0][-1]


def canonical_free(base, gens, g, num_free):
    """
    Canonicalization of a tensor with respect to free indices
    choosing the minimum with respect to lexicographical ordering
    in the free indices.

    Explanation
    ===========

    ``base``, ``gens``  BSGS for slot permutation group
    ``g``               permutation representing the tensor
    ``num_free``        number of free indices
    The indices must be ordered with first the free indices

    See explanation in double_coset_can_rep
    The algorithm is a variation of the one given in [2].

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import canonical_free
    >>> gens = [[1, 0, 2, 3, 5, 4], [2, 3, 0, 1, 4, 5],[0, 1, 3, 2, 5, 4]]
    >>> gens = [Permutation(h) for h in gens]
    >>> base = [0, 2]
    >>> g = Permutation([2, 1, 0, 3, 4, 5])
    >>> canonical_free(base, gens, g, 4)
    [0, 3, 1, 2, 5, 4]

    Consider the product of Riemann tensors
    ``T = R^{a}_{d0}^{d1,d2}*R_{d2,d1}^{d0,b}``
    The order of the indices is ``[a, b, d0, -d0, d1, -d1, d2, -d2]``
    The permutation corresponding to the tensor is
    ``g = [0, 3, 4, 6, 7, 5, 2, 1, 8, 9]``

    In particular ``a`` is position ``0``, ``b`` is in position ``9``.
    Use the slot symmetries to get `T` is a form which is the minimal
    in lexicographic order in the free indices ``a`` and ``b``, e.g.
    ``-R^{a}_{d0}^{d1,d2}*R^{b,d0}_{d2,d1}`` corresponding to
    ``[0, 3, 4, 6, 1, 2, 7, 5, 9, 8]``

    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, tensor_gens
    >>> base, gens = riemann_bsgs
    >>> size, sbase, sgens = tensor_gens(base, gens, [[], []], 0)
    >>> g = Permutation([0, 3, 4, 6, 7, 5, 2, 1, 8, 9])
    >>> canonical_free(sbase, [Permutation(h) for h in sgens], g, 2)
    [0, 3, 4, 6, 1, 2, 7, 5, 9, 8]
    """
    g = g.array_form
    size = len(g)
    if not base:
        return g[:]

    transversals = get_transversals(base, gens)
    for x in sorted(g[:-2]):
        if x not in base:
            base.append(x)
    h = g
    for transv in transversals:
        h_i = [size]*num_free
        # find the element s in transversals[i] such that
        # _af_rmul(h, s) has its free elements with the lowest position in h
        s = None
        for sk in transv.values():
            h1 = _af_rmul(h, sk)
            hi = [h1.index(ix) for ix in range(num_free)]
            if hi < h_i:
                h_i = hi
                s = sk
        if s:
            h = _af_rmul(h, s)
    return h


def _get_map_slots(size, fixed_slots):
    res = list(range(size))
    pos = 0
    for i in range(size):
        if i in fixed_slots:
            continue
        res[i] = pos
        pos += 1
    return res


def _lift_sgens(size, fixed_slots, free, s):
    a = []
    j = k = 0
    fd = [y for _, y in sorted(zip(fixed_slots, free))]
    num_free = len(free)
    for i in range(size):
        if i in fixed_slots:
            a.append(fd[k])
            k += 1
        else:
            a.append(s[j] + num_free)
            j += 1
    return a


def canonicalize(g, dummies, msym, *v):
    """
    canonicalize tensor formed by tensors

    Parameters
    ==========

    g : permutation representing the tensor

    dummies : list representing the dummy indices
      it can be a list of dummy indices of the same type
      or a list of lists of dummy indices, one list for each
      type of index;
      the dummy indices must come after the free indices,
      and put in order contravariant, covariant
      [d0, -d0, d1,-d1,...]

    msym :  symmetry of the metric(s)
        it can be an integer or a list;
        in the first case it is the symmetry of the dummy index metric;
        in the second case it is the list of the symmetries of the
        index metric for each type

    v : list, (base_i, gens_i, n_i, sym_i) for tensors of type `i`

    base_i, gens_i : BSGS for tensors of this type.
        The BSGS should have minimal base under lexicographic ordering;
        if not, an attempt is made do get the minimal BSGS;
        in case of failure,
        canonicalize_naive is used, which is much slower.

    n_i :    number of tensors of type `i`.

    sym_i :  symmetry under exchange of component tensors of type `i`.

        Both for msym and sym_i the cases are
            * None  no symmetry
            * 0     commuting
            * 1     anticommuting

    Returns
    =======

    0 if the tensor is zero, else return the array form of
    the permutation representing the canonical form of the tensor.

    Algorithm
    =========

    First one uses canonical_free to get the minimum tensor under
    lexicographic order, using only the slot symmetries.
    If the component tensors have not minimal BSGS, it is attempted
    to find it; if the attempt fails canonicalize_naive
    is used instead.

    Compute the residual slot symmetry keeping fixed the free indices
    using tensor_gens(base, gens, list_free_indices, sym).

    Reduce the problem eliminating the free indices.

    Then use double_coset_can_rep and lift back the result reintroducing
    the free indices.

    Examples
    ========

    one type of index with commuting metric;

    `A_{a b}` and `B_{a b}` antisymmetric and commuting

    `T = A_{d0 d1} * B^{d0}{}_{d2} * B^{d2 d1}`

    `ord = [d0,-d0,d1,-d1,d2,-d2]` order of the indices

    g = [1, 3, 0, 5, 4, 2, 6, 7]

    `T_c = 0`

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, canonicalize, bsgs_direct_product
    >>> from sympy.combinatorics import Permutation
    >>> base2a, gens2a = get_symmetric_group_sgs(2, 1)
    >>> t0 = (base2a, gens2a, 1, 0)
    >>> t1 = (base2a, gens2a, 2, 0)
    >>> g = Permutation([1, 3, 0, 5, 4, 2, 6, 7])
    >>> canonicalize(g, range(6), 0, t0, t1)
    0

    same as above, but with `B_{a b}` anticommuting

    `T_c = -A^{d0 d1} * B_{d0}{}^{d2} * B_{d1 d2}`

    can = [0,2,1,4,3,5,7,6]

    >>> t1 = (base2a, gens2a, 2, 1)
    >>> canonicalize(g, range(6), 0, t0, t1)
    [0, 2, 1, 4, 3, 5, 7, 6]

    two types of indices `[a,b,c,d,e,f]` and `[m,n]`, in this order,
    both with commuting metric

    `f^{a b c}` antisymmetric, commuting

    `A_{m a}` no symmetry, commuting

    `T = f^c{}_{d a} * f^f{}_{e b} * A_m{}^d * A^{m b} * A_n{}^a * A^{n e}`

    ord = [c,f,a,-a,b,-b,d,-d,e,-e,m,-m,n,-n]

    g = [0,7,3, 1,9,5, 11,6, 10,4, 13,2, 12,8, 14,15]

    The canonical tensor is
    `T_c = -f^{c a b} * f^{f d e} * A^m{}_a * A_{m d} * A^n{}_b * A_{n e}`

    can = [0,2,4, 1,6,8, 10,3, 11,7, 12,5, 13,9, 15,14]

    >>> base_f, gens_f = get_symmetric_group_sgs(3, 1)
    >>> base1, gens1 = get_symmetric_group_sgs(1)
    >>> base_A, gens_A = bsgs_direct_product(base1, gens1, base1, gens1)
    >>> t0 = (base_f, gens_f, 2, 0)
    >>> t1 = (base_A, gens_A, 4, 0)
    >>> dummies = [range(2, 10), range(10, 14)]
    >>> g = Permutation([0, 7, 3, 1, 9, 5, 11, 6, 10, 4, 13, 2, 12, 8, 14, 15])
    >>> canonicalize(g, dummies, [0, 0], t0, t1)
    [0, 2, 4, 1, 6, 8, 10, 3, 11, 7, 12, 5, 13, 9, 15, 14]
    """
    from sympy.combinatorics.testutil import canonicalize_naive
    if not isinstance(msym, list):
        if msym not in (0, 1, None):
            raise ValueError('msym must be 0, 1 or None')
        num_types = 1
    else:
        num_types = len(msym)
        if not all(msymx in (0, 1, None) for msymx in msym):
            raise ValueError('msym entries must be 0, 1 or None')
        if len(dummies) != num_types:
            raise ValueError(
                'dummies and msym must have the same number of elements')
    size = g.size
    num_tensors = 0
    v1 = []
    for base_i, gens_i, n_i, sym_i in v:
        # check that the BSGS is minimal;
        # this property is used in double_coset_can_rep;
        # if it is not minimal use canonicalize_naive
        if not _is_minimal_bsgs(base_i, gens_i):
            mbsgs = get_minimal_bsgs(base_i, gens_i)
            if not mbsgs:
                can = canonicalize_naive(g, dummies, msym, *v)
                return can
            base_i, gens_i = mbsgs
        v1.append((base_i, gens_i, [[]] * n_i, sym_i))
        num_tensors += n_i

    if num_types == 1 and not isinstance(msym, list):
        dummies = [dummies]
        msym = [msym]
    flat_dummies = []
    for dumx in dummies:
        flat_dummies.extend(dumx)

    if flat_dummies and flat_dummies != list(range(flat_dummies[0], flat_dummies[-1] + 1)):
        raise ValueError('dummies is not valid')

    # slot symmetry of the tensor
    size1, sbase, sgens = gens_products(*v1)
    if size != size1:
        raise ValueError(
            'g has size %d, generators have size %d' % (size, size1))
    free = [i for i in range(size - 2) if i not in flat_dummies]
    num_free = len(free)

    # g1 minimal tensor under slot symmetry
    g1 = canonical_free(sbase, sgens, g, num_free)
    if not flat_dummies:
        return g1
    # save the sign of g1
    sign = 0 if g1[-1] == size - 1 else 1

    # the free indices are kept fixed.
    # Determine free_i, the list of slots of tensors which are fixed
    # since they are occupied by free indices, which are fixed.
    start = 0
    for i, (base_i, gens_i, n_i, sym_i) in enumerate(v):
        free_i = []
        len_tens = gens_i[0].size - 2
        # for each component tensor get a list od fixed islots
        for j in range(n_i):
            # get the elements corresponding to the component tensor
            h = g1[start:(start + len_tens)]
            fr = []
            # get the positions of the fixed elements in h
            for k in free:
                if k in h:
                    fr.append(h.index(k))
            free_i.append(fr)
            start += len_tens
        v1[i] = (base_i, gens_i, free_i, sym_i)
    # BSGS of the tensor with fixed free indices
    # if tensor_gens fails in gens_product, use canonicalize_naive
    size, sbase, sgens = gens_products(*v1)

    # reduce the permutations getting rid of the free indices
    pos_free = [g1.index(x) for x in range(num_free)]
    size_red = size - num_free
    g1_red = [x - num_free for x in g1 if x in flat_dummies]
    if sign:
        g1_red.extend([size_red - 1, size_red - 2])
    else:
        g1_red.extend([size_red - 2, size_red - 1])
    map_slots = _get_map_slots(size, pos_free)
    sbase_red = [map_slots[i] for i in sbase if i not in pos_free]
    sgens_red = [_af_new([map_slots[i] for i in y._array_form if i not in pos_free]) for y in sgens]
    dummies_red = [[x - num_free for x in y] for y in dummies]
    transv_red = get_transversals(sbase_red, sgens_red)
    g1_red = _af_new(g1_red)
    g2 = double_coset_can_rep(
        dummies_red, msym, sbase_red, sgens_red, transv_red, g1_red)
    if g2 == 0:
        return 0
    # lift to the case with the free indices
    g3 = _lift_sgens(size, pos_free, free, g2)
    return g3


def perm_af_direct_product(gens1, gens2, signed=True):
    """
    Direct products of the generators gens1 and gens2.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import perm_af_direct_product
    >>> gens1 = [[1, 0, 2, 3], [0, 1, 3, 2]]
    >>> gens2 = [[1, 0]]
    >>> perm_af_direct_product(gens1, gens2, False)
    [[1, 0, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5], [0, 1, 2, 3, 5, 4]]
    >>> gens1 = [[1, 0, 2, 3, 5, 4], [0, 1, 3, 2, 4, 5]]
    >>> gens2 = [[1, 0, 2, 3]]
    >>> perm_af_direct_product(gens1, gens2, True)
    [[1, 0, 2, 3, 4, 5, 7, 6], [0, 1, 3, 2, 4, 5, 6, 7], [0, 1, 2, 3, 5, 4, 6, 7]]
    """
    gens1 = [list(x) for x in gens1]
    gens2 = [list(x) for x in gens2]
    s = 2 if signed else 0
    n1 = len(gens1[0]) - s
    n2 = len(gens2[0]) - s
    start = list(range(n1))
    end = list(range(n1, n1 + n2))
    if signed:
        gens1 = [gen[:-2] + end + [gen[-2] + n2, gen[-1] + n2]
                 for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]
    else:
        gens1 = [gen + end for gen in gens1]
        gens2 = [start + [x + n1 for x in gen] for gen in gens2]

    res = gens1 + gens2

    return res


def bsgs_direct_product(base1, gens1, base2, gens2, signed=True):
    """
    Direct product of two BSGS.

    Parameters
    ==========

    base1 : base of the first BSGS.

    gens1 : strong generating sequence of the first BSGS.

    base2, gens2 : similarly for the second BSGS.

    signed : flag for signed permutations.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import (get_symmetric_group_sgs, bsgs_direct_product)
    >>> base1, gens1 = get_symmetric_group_sgs(1)
    >>> base2, gens2 = get_symmetric_group_sgs(2)
    >>> bsgs_direct_product(base1, gens1, base2, gens2)
    ([1], [(4)(1 2)])
    """
    s = 2 if signed else 0
    n1 = gens1[0].size - s
    base = list(base1)
    base += [x + n1 for x in base2]
    gens1 = [h._array_form for h in gens1]
    gens2 = [h._array_form for h in gens2]
    gens = perm_af_direct_product(gens1, gens2, signed)
    size = len(gens[0])
    id_af = list(range(size))
    gens = [h for h in gens if h != id_af]
    if not gens:
        gens = [id_af]
    return base, [_af_new(h) for h in gens]


def get_symmetric_group_sgs(n, antisym=False):
    """
    Return base, gens of the minimal BSGS for (anti)symmetric tensor

    Parameters
    ==========

    n : rank of the tensor
    antisym : bool
        ``antisym = False`` symmetric tensor
        ``antisym = True``  antisymmetric tensor

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs
    >>> get_symmetric_group_sgs(3)
    ([0, 1], [(4)(0 1), (4)(1 2)])
    """
    if n == 1:
        return [], [_af_new(list(range(3)))]
    gens = [Permutation(n - 1)(i, i + 1)._array_form for i in range(n - 1)]
    if antisym == 0:
        gens = [x + [n, n + 1] for x in gens]
    else:
        gens = [x + [n + 1, n] for x in gens]
    base = list(range(n - 1))
    return base, [_af_new(h) for h in gens]

riemann_bsgs = [0, 2], [Permutation(0, 1)(4, 5), Permutation(2, 3)(4, 5),
                        Permutation(5)(0, 2)(1, 3)]


def get_transversals(base, gens):
    """
    Return transversals for the group with BSGS base, gens
    """
    if not base:
        return []
    stabs = _distribute_gens_by_base(base, gens)
    orbits, transversals = _orbits_transversals_from_bsgs(base, stabs)
    transversals = [{x: h._array_form for x, h in y.items()} for y in
                    transversals]
    return transversals


def _is_minimal_bsgs(base, gens):
    """
    Check if the BSGS has minimal base under lexigographic order.

    base, gens BSGS

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import riemann_bsgs, _is_minimal_bsgs
    >>> _is_minimal_bsgs(*riemann_bsgs)
    True
    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
    >>> _is_minimal_bsgs(*riemann_bsgs1)
    False
    """
    base1 = []
    sgs1 = gens[:]
    size = gens[0].size
    for i in range(size):
        if not all(h._array_form[i] == i for h in sgs1):
            base1.append(i)
            sgs1 = [h for h in sgs1 if h._array_form[i] == i]
    return base1 == base


def get_minimal_bsgs(base, gens):
    """
    Compute a minimal GSGS

    base, gens BSGS

    If base, gens is a minimal BSGS return it; else return a minimal BSGS
    if it fails in finding one, it returns None

    TODO: use baseswap in the case in which if it fails in finding a
    minimal BSGS

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.tensor_can import get_minimal_bsgs
    >>> riemann_bsgs1 = ([2, 0], ([Permutation(5)(0, 1)(4, 5), Permutation(5)(0, 2)(1, 3)]))
    >>> get_minimal_bsgs(*riemann_bsgs1)
    ([0, 2], [(0 1)(4 5), (5)(0 2)(1 3), (2 3)(4 5)])
    """
    G = PermutationGroup(gens)
    base, gens = G.schreier_sims_incremental()
    if not _is_minimal_bsgs(base, gens):
        return None
    return base, gens


def tensor_gens(base, gens, list_free_indices, sym=0):
    """
    Returns size, res_base, res_gens BSGS for n tensors of the
    same type.

    Explanation
    ===========

    base, gens BSGS for tensors of this type
    list_free_indices  list of the slots occupied by fixed indices
                       for each of the tensors

    sym symmetry under commutation of two tensors
    sym   None  no symmetry
    sym   0     commuting
    sym   1     anticommuting

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import tensor_gens, get_symmetric_group_sgs

    two symmetric tensors with 3 indices without free indices

    >>> base, gens = get_symmetric_group_sgs(3)
    >>> tensor_gens(base, gens, [[], []])
    (8, [0, 1, 3, 4], [(7)(0 1), (7)(1 2), (7)(3 4), (7)(4 5), (7)(0 3)(1 4)(2 5)])

    two symmetric tensors with 3 indices with free indices in slot 1 and 0

    >>> tensor_gens(base, gens, [[1], [0]])
    (8, [0, 4], [(7)(0 2), (7)(4 5)])

    four symmetric tensors with 3 indices, two of which with free indices

    """
    def _get_bsgs(G, base, gens, free_indices):
        """
        return the BSGS for G.pointwise_stabilizer(free_indices)
        """
        if not free_indices:
            return base[:], gens[:]
        else:
            H = G.pointwise_stabilizer(free_indices)
            base, sgs = H.schreier_sims_incremental()
            return base, sgs

    # if not base there is no slot symmetry for the component tensors
    # if list_free_indices.count([]) < 2 there is no commutation symmetry
    # so there is no resulting slot symmetry
    if not base and list_free_indices.count([]) < 2:
        n = len(list_free_indices)
        size = gens[0].size
        size = n * (size - 2) + 2
        return size, [], [_af_new(list(range(size)))]

    # if any(list_free_indices) one needs to compute the pointwise
    # stabilizer, so G is needed
    if any(list_free_indices):
        G = PermutationGroup(gens)
    else:
        G = None

    # no_free list of lists of indices for component tensors without fixed
    # indices
    no_free = []
    size = gens[0].size
    id_af = list(range(size))
    num_indices = size - 2
    if not list_free_indices[0]:
        no_free.append(list(range(num_indices)))
    res_base, res_gens = _get_bsgs(G, base, gens, list_free_indices[0])
    for i in range(1, len(list_free_indices)):
        base1, gens1 = _get_bsgs(G, base, gens, list_free_indices[i])
        res_base, res_gens = bsgs_direct_product(res_base, res_gens,
                                                 base1, gens1, 1)
        if not list_free_indices[i]:
            no_free.append(list(range(size - 2, size - 2 + num_indices)))
        size += num_indices
    nr = size - 2
    res_gens = [h for h in res_gens if h._array_form != id_af]
    # if sym there are no commuting tensors stop here
    if sym is None or not no_free:
        if not res_gens:
            res_gens = [_af_new(id_af)]
        return size, res_base, res_gens

    # if the component tensors have moinimal BSGS, so is their direct
    # product P; the slot symmetry group is S = P*C, where C is the group
    # to (anti)commute the component tensors with no free indices
    # a stabilizer has the property S_i = P_i*C_i;
    # the BSGS of P*C has SGS_P + SGS_C and the base is
    # the ordered union of the bases of P and C.
    # If P has minimal BSGS, so has S with this base.
    base_comm = []
    for i in range(len(no_free) - 1):
        ind1 = no_free[i]
        ind2 = no_free[i + 1]
        a = list(range(ind1[0]))
        a.extend(ind2)
        a.extend(ind1)
        base_comm.append(ind1[0])
        a.extend(list(range(ind2[-1] + 1, nr)))
        if sym == 0:
            a.extend([nr, nr + 1])
        else:
            a.extend([nr + 1, nr])
        res_gens.append(_af_new(a))
    res_base = list(res_base)
    # each base is ordered; order the union of the two bases
    for i in base_comm:
        if i not in res_base:
            res_base.append(i)
    res_base.sort()
    if not res_gens:
        res_gens = [_af_new(id_af)]

    return size, res_base, res_gens


def gens_products(*v):
    """
    Returns size, res_base, res_gens BSGS for n tensors of different types.

    Explanation
    ===========

    v is a sequence of (base_i, gens_i, free_i, sym_i)
    where
    base_i, gens_i  BSGS of tensor of type `i`
    free_i          list of the fixed slots for each of the tensors
                    of type `i`; if there are `n_i` tensors of type `i`
                    and none of them have fixed slots, `free = [[]]*n_i`
    sym   0 (1) if the tensors of type `i` (anti)commute among themselves

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, gens_products
    >>> base, gens = get_symmetric_group_sgs(2)
    >>> gens_products((base, gens, [[], []], 0))
    (6, [0, 2], [(5)(0 1), (5)(2 3), (5)(0 2)(1 3)])
    >>> gens_products((base, gens, [[1], []], 0))
    (6, [2], [(5)(2 3)])
    """
    res_size, res_base, res_gens = tensor_gens(*v[0])
    for i in range(1, len(v)):
        size, base, gens = tensor_gens(*v[i])
        res_base, res_gens = bsgs_direct_product(res_base, res_gens, base,
                                                 gens, 1)
    res_size = res_gens[0].size
    id_af = list(range(res_size))
    res_gens = [h for h in res_gens if h != id_af]
    if not res_gens:
        res_gens = [id_af]
    return res_size, res_base, res_gens

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexes\range.py ===
from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterator,
)
from datetime import timedelta
import operator
from sys import getsizeof
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
    overload,
)

import numpy as np

from pandas._libs import (
    index as libindex,
    lib,
)
from pandas._libs.algos import unique_deltas
from pandas._libs.lib import no_default
from pandas.compat.numpy import function as nv
from pandas.util._decorators import (
    cache_readonly,
    deprecate_nonkeyword_arguments,
    doc,
)

from pandas.core.dtypes.common import (
    ensure_platform_int,
    ensure_python_int,
    is_float,
    is_integer,
    is_scalar,
    is_signed_integer_dtype,
)
from pandas.core.dtypes.generic import ABCTimedeltaIndex

from pandas.core import ops
import pandas.core.common as com
from pandas.core.construction import extract_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        Dtype,
        NaPosition,
        Self,
        npt,
    )
_empty_range = range(0)
_dtype_int64 = np.dtype(np.int64)


class RangeIndex(Index):
    """
    Immutable Index implementing a monotonic integer range.

    RangeIndex is a memory-saving special case of an Index limited to representing
    monotonic ranges with a 64-bit dtype. Using RangeIndex may in some instances
    improve computing speed.

    This is the default index type used
    by DataFrame and Series when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), range, or other RangeIndex instance
        If int and "stop" is not given, interpreted as "stop" instead.
    stop : int (default: 0)
    step : int (default: 1)
    dtype : np.int64
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    from_range

    See Also
    --------
    Index : The base pandas Index type.

    Examples
    --------
    >>> list(pd.RangeIndex(5))
    [0, 1, 2, 3, 4]

    >>> list(pd.RangeIndex(-2, 4))
    [-2, -1, 0, 1, 2, 3]

    >>> list(pd.RangeIndex(0, 10, 2))
    [0, 2, 4, 6, 8]

    >>> list(pd.RangeIndex(2, -10, -3))
    [2, -1, -4, -7]

    >>> list(pd.RangeIndex(0))
    []

    >>> list(pd.RangeIndex(1, 0))
    []
    """

    _typ = "rangeindex"
    _dtype_validation_metadata = (is_signed_integer_dtype, "signed integer")
    _range: range
    _values: np.ndarray

    @property
    def _engine_type(self) -> type[libindex.Int64Engine]:
        return libindex.Int64Engine

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        start=None,
        stop=None,
        step=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        cls._validate_dtype(dtype)
        name = maybe_extract_name(name, start, cls)

        # RangeIndex
        if isinstance(start, cls):
            return start.copy(name=name)
        elif isinstance(start, range):
            return cls._simple_new(start, name=name)

        # validate the arguments
        if com.all_none(start, stop, step):
            raise TypeError("RangeIndex(...) must be called with integers")

        start = ensure_python_int(start) if start is not None else 0

        if stop is None:
            start, stop = 0, start
        else:
            stop = ensure_python_int(stop)

        step = ensure_python_int(step) if step is not None else 1
        if step == 0:
            raise ValueError("Step must not be zero")

        rng = range(start, stop, step)
        return cls._simple_new(rng, name=name)

    @classmethod
    def from_range(cls, data: range, name=None, dtype: Dtype | None = None) -> Self:
        """
        Create :class:`pandas.RangeIndex` from a ``range`` object.

        Returns
        -------
        RangeIndex

        Examples
        --------
        >>> pd.RangeIndex.from_range(range(5))
        RangeIndex(start=0, stop=5, step=1)

        >>> pd.RangeIndex.from_range(range(2, -10, -3))
        RangeIndex(start=2, stop=-10, step=-3)
        """
        if not isinstance(data, range):
            raise TypeError(
                f"{cls.__name__}(...) must be called with object coercible to a "
                f"range, {repr(data)} was passed"
            )
        cls._validate_dtype(dtype)
        return cls._simple_new(data, name=name)

    #  error: Argument 1 of "_simple_new" is incompatible with supertype "Index";
    #  supertype defines the argument type as
    #  "Union[ExtensionArray, ndarray[Any, Any]]"  [override]
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls, values: range, name: Hashable | None = None
    ) -> Self:
        result = object.__new__(cls)

        assert isinstance(values, range)

        result._range = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        result._references = None
        return result

    @classmethod
    def _validate_dtype(cls, dtype: Dtype | None) -> None:
        if dtype is None:
            return

        validation_func, expected = cls._dtype_validation_metadata
        if not validation_func(dtype):
            raise ValueError(
                f"Incorrect `dtype` passed: expected {expected}, received {dtype}"
            )

    # --------------------------------------------------------------------

    # error: Return type "Type[Index]" of "_constructor" incompatible with return
    # type "Type[RangeIndex]" in supertype "Index"
    @cache_readonly
    def _constructor(self) -> type[Index]:  # type: ignore[override]
        """return the class to use for construction"""
        return Index

    # error: Signature of "_data" incompatible with supertype "Index"
    @cache_readonly
    def _data(self) -> np.ndarray:  # type: ignore[override]
        """
        An int array that for performance reasons is created only when needed.

        The constructed array is saved in ``_cache``.
        """
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)

    def _get_data_as_items(self) -> list[tuple[str, int]]:
        """return a list of tuples of start, stop, step"""
        rng = self._range
        return [("start", rng.start), ("stop", rng.stop), ("step", rng.step)]

    def __reduce__(self):
        d = {"name": self._name}
        d.update(dict(self._get_data_as_items()))
        return ibase._new_Index, (type(self), d), None

    # --------------------------------------------------------------------
    # Rendering Methods

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr, formatted_value)
        """
        attrs = cast("list[tuple[str, str | int]]", self._get_data_as_items())
        if self._name is not None:
            attrs.append(("name", ibase.default_pprint(self._name)))
        return attrs

    def _format_with_header(self, *, header: list[str], na_rep: str) -> list[str]:
        # Equivalent to Index implementation, but faster
        if not len(self._range):
            return header
        first_val_str = str(self._range[0])
        last_val_str = str(self._range[-1])
        max_length = max(len(first_val_str), len(last_val_str))

        return header + [f"{x:<{max_length}}" for x in self._range]

    # --------------------------------------------------------------------

    @property
    def start(self) -> int:
        """
        The value of the `start` parameter (``0`` if this was not supplied).

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.start
        0

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.start
        2
        """
        # GH 25710
        return self._range.start

    @property
    def stop(self) -> int:
        """
        The value of the `stop` parameter.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.stop
        5

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.stop
        -10
        """
        return self._range.stop

    @property
    def step(self) -> int:
        """
        The value of the `step` parameter (``1`` if this was not supplied).

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.step
        1

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.step
        -3

        Even if :class:`pandas.RangeIndex` is empty, ``step`` is still ``1`` if
        not supplied.

        >>> idx = pd.RangeIndex(1, 0)
        >>> idx.step
        1
        """
        # GH 25710
        return self._range.step

    @cache_readonly
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.
        """
        rng = self._range
        return getsizeof(rng) + sum(
            getsizeof(getattr(rng, attr_name))
            for attr_name in ["start", "stop", "step"]
        )

    def memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        return self.nbytes

    @property
    def dtype(self) -> np.dtype:
        return _dtype_int64

    @property
    def is_unique(self) -> bool:
        """return if the index has unique values"""
        return True

    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        return self._range.step > 0 or len(self) <= 1

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        return self._range.step < 0 or len(self) <= 1

    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            key = ensure_python_int(key)
        except TypeError:
            return False
        return key in self._range

    @property
    def inferred_type(self) -> str:
        return "integer"

    # --------------------------------------------------------------------
    # Indexing Methods

    @doc(Index.get_loc)
    def get_loc(self, key) -> int:
        if is_integer(key) or (is_float(key) and key.is_integer()):
            new_key = int(key)
            try:
                return self._range.index(new_key)
            except ValueError as err:
                raise KeyError(key) from err
        if isinstance(key, Hashable):
            raise KeyError(key)
        self._check_indexing_error(key)
        raise KeyError(key)

    def _get_indexer(
        self,
        target: Index,
        method: str | None = None,
        limit: int | None = None,
        tolerance=None,
    ) -> npt.NDArray[np.intp]:
        if com.any_not_none(method, tolerance, limit):
            return super()._get_indexer(
                target, method=method, tolerance=tolerance, limit=limit
            )

        if self.step > 0:
            start, stop, step = self.start, self.stop, self.step
        else:
            # GH 28678: work on reversed range for simplicity
            reverse = self._range[::-1]
            start, stop, step = reverse.start, reverse.stop, reverse.step

        target_array = np.asarray(target)
        locs = target_array - start
        valid = (locs % step == 0) & (locs >= 0) & (target_array < stop)
        locs[~valid] = -1
        locs[valid] = locs[valid] / step

        if step != self.step:
            # We reversed this range: transform to original locs
            locs[valid] = len(self) - 1 - locs[valid]
        return ensure_platform_int(locs)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        """
        Should an integer key be treated as positional?
        """
        return False

    # --------------------------------------------------------------------

    def tolist(self) -> list[int]:
        return list(self._range)

    @doc(Index.__iter__)
    def __iter__(self) -> Iterator[int]:
        yield from self._range

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values, name: Hashable = no_default):
        name = self._name if name is no_default else name

        if values.dtype.kind == "f":
            return Index(values, name=name, dtype=np.float64)
        # GH 46675 & 43885: If values is equally spaced, return a
        # more memory-compact RangeIndex instead of Index with 64-bit dtype
        unique_diffs = unique_deltas(values)
        if len(unique_diffs) == 1 and unique_diffs[0] != 0:
            diff = unique_diffs[0]
            new_range = range(values[0], values[-1] + diff, diff)
            return type(self)._simple_new(new_range, name=name)
        else:
            return self._constructor._simple_new(values, name=name)

    def _view(self) -> Self:
        result = type(self)._simple_new(self._range, name=self._name)
        result._cache = self._cache
        return result

    @doc(Index.copy)
    def copy(self, name: Hashable | None = None, deep: bool = False) -> Self:
        name = self._validate_names(name=name, deep=deep)[0]
        new_index = self._rename(name=name)
        return new_index

    def _minmax(self, meth: str):
        no_steps = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif (meth == "min" and self.step > 0) or (meth == "max" and self.step < 0):
            return self.start

        return self.start + self.step * no_steps

    def min(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        """The minimum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_min(args, kwargs)
        return self._minmax("min")

    def max(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        """The maximum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_max(args, kwargs)
        return self._minmax("max")

    def argsort(self, *args, **kwargs) -> npt.NDArray[np.intp]:
        """
        Returns the indices that would sort the index and its
        underlying data.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort
        """
        ascending = kwargs.pop("ascending", True)  # EA compat
        kwargs.pop("kind", None)  # e.g. "mergesort" is irrelevant
        nv.validate_argsort(args, kwargs)

        if self._range.step > 0:
            result = np.arange(len(self), dtype=np.intp)
        else:
            result = np.arange(len(self) - 1, -1, -1, dtype=np.intp)

        if not ascending:
            result = result[::-1]
        return result

    def factorize(
        self,
        sort: bool = False,
        use_na_sentinel: bool = True,
    ) -> tuple[npt.NDArray[np.intp], RangeIndex]:
        codes = np.arange(len(self), dtype=np.intp)
        uniques = self
        if sort and self.step < 0:
            codes = codes[::-1]
            uniques = uniques[::-1]
        return codes, uniques

    def equals(self, other: object) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        if isinstance(other, RangeIndex):
            return self._range == other._range
        return super().equals(other)

    # error: Signature of "sort_values" incompatible with supertype "Index"
    @overload  # type: ignore[override]
    def sort_values(
        self,
        *,
        return_indexer: Literal[False] = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> Self:
        ...

    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[True],
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> tuple[Self, np.ndarray | RangeIndex]:
        ...

    @overload
    def sort_values(
        self,
        *,
        return_indexer: bool = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> Self | tuple[Self, np.ndarray | RangeIndex]:
        ...

    @deprecate_nonkeyword_arguments(
        version="3.0", allowed_args=["self"], name="sort_values"
    )
    def sort_values(
        self,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: NaPosition = "last",
        key: Callable | None = None,
    ) -> Self | tuple[Self, np.ndarray | RangeIndex]:
        if key is not None:
            return super().sort_values(
                return_indexer=return_indexer,
                ascending=ascending,
                na_position=na_position,
                key=key,
            )
        else:
            sorted_index = self
            inverse_indexer = False
            if ascending:
                if self.step < 0:
                    sorted_index = self[::-1]
                    inverse_indexer = True
            else:
                if self.step > 0:
                    sorted_index = self[::-1]
                    inverse_indexer = True

        if return_indexer:
            if inverse_indexer:
                rng = range(len(self) - 1, -1, -1)
            else:
                rng = range(len(self))
            return sorted_index, RangeIndex(rng)
        else:
            return sorted_index

    # --------------------------------------------------------------------
    # Set Operations

    def _intersection(self, other: Index, sort: bool = False):
        # caller is responsible for checking self and other are both non-empty

        if not isinstance(other, RangeIndex):
            return super()._intersection(other, sort=sort)

        first = self._range[::-1] if self.step < 0 else self._range
        second = other._range[::-1] if other.step < 0 else other._range

        # check whether intervals intersect
        # deals with in- and decreasing ranges
        int_low = max(first.start, second.start)
        int_high = min(first.stop, second.stop)
        if int_high <= int_low:
            return self._simple_new(_empty_range)

        # Method hint: linear Diophantine equation
        # solve intersection problem
        # performance hint: for identical step sizes, could use
        # cheaper alternative
        gcd, s, _ = self._extended_gcd(first.step, second.step)

        # check whether element sets intersect
        if (first.start - second.start) % gcd:
            return self._simple_new(_empty_range)

        # calculate parameters for the RangeIndex describing the
        # intersection disregarding the lower bounds
        tmp_start = first.start + (second.start - first.start) * first.step // gcd * s
        new_step = first.step * second.step // gcd
        new_range = range(tmp_start, int_high, new_step)
        new_index = self._simple_new(new_range)

        # adjust index to limiting interval
        new_start = new_index._min_fitting_element(int_low)
        new_range = range(new_start, new_index.stop, new_index.step)
        new_index = self._simple_new(new_range)

        if (self.step < 0 and other.step < 0) is not (new_index.step < 0):
            new_index = new_index[::-1]

        if sort is None:
            new_index = new_index.sort_values()

        return new_index

    def _min_fitting_element(self, lower_limit: int) -> int:
        """Returns the smallest element greater than or equal to the limit"""
        no_steps = -(-(lower_limit - self.start) // abs(self.step))
        return self.start + abs(self.step) * no_steps

    def _extended_gcd(self, a: int, b: int) -> tuple[int, int, int]:
        """
        Extended Euclidean algorithms to solve Bezout's identity:
           a*x + b*y = gcd(x, y)
        Finds one particular solution for x, y: s, t
        Returns: gcd, s, t
        """
        s, old_s = 0, 1
        t, old_t = 1, 0
        r, old_r = b, a
        while r:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t
        return old_r, old_s, old_t

    def _range_in_self(self, other: range) -> bool:
        """Check if other range is contained in self"""
        # https://stackoverflow.com/a/32481015
        if not other:
            return True
        if not self._range:
            return False
        if len(other) > 1 and other.step % self._range.step:
            return False
        return other.start in self._range and other[-1] in self._range

    def _union(self, other: Index, sort: bool | None):
        """
        Form the union of two Index objects and sorts if possible

        Parameters
        ----------
        other : Index or array-like

        sort : bool or None, default None
            Whether to sort (monotonically increasing) the resulting index.
            ``sort=None|True`` returns a ``RangeIndex`` if possible or a sorted
            ``Index`` with a int64 dtype if not.
            ``sort=False`` can return a ``RangeIndex`` if self is monotonically
            increasing and other is fully contained in self. Otherwise, returns
            an unsorted ``Index`` with an int64 dtype.

        Returns
        -------
        union : Index
        """
        if isinstance(other, RangeIndex):
            if sort in (None, True) or (
                sort is False and self.step > 0 and self._range_in_self(other._range)
            ):
                # GH 47557: Can still return a RangeIndex
                # if other range in self and sort=False
                start_s, step_s = self.start, self.step
                end_s = self.start + self.step * (len(self) - 1)
                start_o, step_o = other.start, other.step
                end_o = other.start + other.step * (len(other) - 1)
                if self.step < 0:
                    start_s, step_s, end_s = end_s, -step_s, start_s
                if other.step < 0:
                    start_o, step_o, end_o = end_o, -step_o, start_o
                if len(self) == 1 and len(other) == 1:
                    step_s = step_o = abs(self.start - other.start)
                elif len(self) == 1:
                    step_s = step_o
                elif len(other) == 1:
                    step_o = step_s
                start_r = min(start_s, start_o)
                end_r = max(end_s, end_o)
                if step_o == step_s:
                    if (
                        (start_s - start_o) % step_s == 0
                        and (start_s - end_o) <= step_s
                        and (start_o - end_s) <= step_s
                    ):
                        return type(self)(start_r, end_r + step_s, step_s)
                    if (
                        (step_s % 2 == 0)
                        and (abs(start_s - start_o) == step_s / 2)
                        and (abs(end_s - end_o) == step_s / 2)
                    ):
                        # e.g. range(0, 10, 2) and range(1, 11, 2)
                        #  but not range(0, 20, 4) and range(1, 21, 4) GH#44019
                        return type(self)(start_r, end_r + step_s / 2, step_s / 2)

                elif step_o % step_s == 0:
                    if (
                        (start_o - start_s) % step_s == 0
                        and (start_o + step_s >= start_s)
                        and (end_o - step_s <= end_s)
                    ):
                        return type(self)(start_r, end_r + step_s, step_s)
                elif step_s % step_o == 0:
                    if (
                        (start_s - start_o) % step_o == 0
                        and (start_s + step_o >= start_o)
                        and (end_s - step_o <= end_o)
                    ):
                        return type(self)(start_r, end_r + step_o, step_o)

        return super()._union(other, sort=sort)

    def _difference(self, other, sort=None):
        # optimized set operation if we have another RangeIndex
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)

        if not isinstance(other, RangeIndex):
            return super()._difference(other, sort=sort)

        if sort is not False and self.step < 0:
            return self[::-1]._difference(other)

        res_name = ops.get_op_result_name(self, other)

        first = self._range[::-1] if self.step < 0 else self._range
        overlap = self.intersection(other)
        if overlap.step < 0:
            overlap = overlap[::-1]

        if len(overlap) == 0:
            return self.rename(name=res_name)
        if len(overlap) == len(self):
            return self[:0].rename(res_name)

        # overlap.step will always be a multiple of self.step (see _intersection)

        if len(overlap) == 1:
            if overlap[0] == self[0]:
                return self[1:]

            elif overlap[0] == self[-1]:
                return self[:-1]

            elif len(self) == 3 and overlap[0] == self[1]:
                return self[::2]

            else:
                return super()._difference(other, sort=sort)

        elif len(overlap) == 2 and overlap[0] == first[0] and overlap[-1] == first[-1]:
            # e.g. range(-8, 20, 7) and range(13, -9, -3)
            return self[1:-1]

        if overlap.step == first.step:
            if overlap[0] == first.start:
                # The difference is everything after the intersection
                new_rng = range(overlap[-1] + first.step, first.stop, first.step)
            elif overlap[-1] == first[-1]:
                # The difference is everything before the intersection
                new_rng = range(first.start, overlap[0], first.step)
            elif overlap._range == first[1:-1]:
                # e.g. range(4) and range(1, 3)
                step = len(first) - 1
                new_rng = first[::step]
            else:
                # The difference is not range-like
                # e.g. range(1, 10, 1) and range(3, 7, 1)
                return super()._difference(other, sort=sort)

        else:
            # We must have len(self) > 1, bc we ruled out above
            #  len(overlap) == 0 and len(overlap) == len(self)
            assert len(self) > 1

            if overlap.step == first.step * 2:
                if overlap[0] == first[0] and overlap[-1] in (first[-1], first[-2]):
                    # e.g. range(1, 10, 1) and range(1, 10, 2)
                    new_rng = first[1::2]

                elif overlap[0] == first[1] and overlap[-1] in (first[-1], first[-2]):
                    # e.g. range(1, 10, 1) and range(2, 10, 2)
                    new_rng = first[::2]

                else:
                    # We can get here with  e.g. range(20) and range(0, 10, 2)
                    return super()._difference(other, sort=sort)

            else:
                # e.g. range(10) and range(0, 10, 3)
                return super()._difference(other, sort=sort)

        new_index = type(self)._simple_new(new_rng, name=res_name)
        if first is not self._range:
            new_index = new_index[::-1]

        return new_index

    def symmetric_difference(
        self, other, result_name: Hashable | None = None, sort=None
    ):
        if not isinstance(other, RangeIndex) or sort is not None:
            return super().symmetric_difference(other, result_name, sort)

        left = self.difference(other)
        right = other.difference(self)
        result = left.union(right)

        if result_name is not None:
            result = result.rename(result_name)
        return result

    # --------------------------------------------------------------------

    # error: Return type "Index" of "delete" incompatible with return type
    #  "RangeIndex" in supertype "Index"
    def delete(self, loc) -> Index:  # type: ignore[override]
        # In some cases we can retain RangeIndex, see also
        #  DatetimeTimedeltaMixin._get_delete_Freq
        if is_integer(loc):
            if loc in (0, -len(self)):
                return self[1:]
            if loc in (-1, len(self) - 1):
                return self[:-1]
            if len(self) == 3 and loc in (1, -2):
                return self[::2]

        elif lib.is_list_like(loc):
            slc = lib.maybe_indices_to_slice(np.asarray(loc, dtype=np.intp), len(self))

            if isinstance(slc, slice):
                # defer to RangeIndex._difference, which is optimized to return
                #  a RangeIndex whenever possible
                other = self[slc]
                return self.difference(other, sort=False)

        return super().delete(loc)

    def insert(self, loc: int, item) -> Index:
        if len(self) and (is_integer(item) or is_float(item)):
            # We can retain RangeIndex is inserting at the beginning or end,
            #  or right in the middle.
            rng = self._range
            if loc == 0 and item == self[0] - self.step:
                new_rng = range(rng.start - rng.step, rng.stop, rng.step)
                return type(self)._simple_new(new_rng, name=self._name)

            elif loc == len(self) and item == self[-1] + self.step:
                new_rng = range(rng.start, rng.stop + rng.step, rng.step)
                return type(self)._simple_new(new_rng, name=self._name)

            elif len(self) == 2 and item == self[0] + self.step / 2:
                # e.g. inserting 1 into [0, 2]
                step = int(self.step / 2)
                new_rng = range(self.start, self.stop, step)
                return type(self)._simple_new(new_rng, name=self._name)

        return super().insert(loc, item)

    def _concat(self, indexes: list[Index], name: Hashable) -> Index:
        """
        Overriding parent method for the case of all RangeIndex instances.

        When all members of "indexes" are of type RangeIndex: result will be
        RangeIndex if possible, Index with a int64 dtype otherwise. E.g.:
        indexes = [RangeIndex(3), RangeIndex(3, 6)] -> RangeIndex(6)
        indexes = [RangeIndex(3), RangeIndex(4, 6)] -> Index([0,1,2,4,5], dtype='int64')
        """
        if not all(isinstance(x, RangeIndex) for x in indexes):
            return super()._concat(indexes, name)

        elif len(indexes) == 1:
            return indexes[0]

        rng_indexes = cast(list[RangeIndex], indexes)

        start = step = next_ = None

        # Filter the empty indexes
        non_empty_indexes = [obj for obj in rng_indexes if len(obj)]

        for obj in non_empty_indexes:
            rng = obj._range

            if start is None:
                # This is set by the first non-empty index
                start = rng.start
                if step is None and len(rng) > 1:
                    step = rng.step
            elif step is None:
                # First non-empty index had only one element
                if rng.start == start:
                    values = np.concatenate([x._values for x in rng_indexes])
                    result = self._constructor(values)
                    return result.rename(name)

                step = rng.start - start

            non_consecutive = (step != rng.step and len(rng) > 1) or (
                next_ is not None and rng.start != next_
            )
            if non_consecutive:
                result = self._constructor(
                    np.concatenate([x._values for x in rng_indexes])
                )
                return result.rename(name)

            if step is not None:
                next_ = rng[-1] + step

        if non_empty_indexes:
            # Get the stop value from "next" or alternatively
            # from the last non-empty index
            stop = non_empty_indexes[-1].stop if next_ is None else next_
            return RangeIndex(start, stop, step).rename(name)

        # Here all "indexes" had 0 length, i.e. were empty.
        # In this case return an empty range index.
        return RangeIndex(0, 0).rename(name)

    def __len__(self) -> int:
        """
        return the length of the RangeIndex
        """
        return len(self._range)

    @property
    def size(self) -> int:
        return len(self)

    def __getitem__(self, key):
        """
        Conserve RangeIndex type for scalar and slice keys.
        """
        if isinstance(key, slice):
            return self._getitem_slice(key)
        elif is_integer(key):
            new_key = int(key)
            try:
                return self._range[new_key]
            except IndexError as err:
                raise IndexError(
                    f"index {key} is out of bounds for axis 0 with size {len(self)}"
                ) from err
        elif is_scalar(key):
            raise IndexError(
                "only integers, slices (`:`), "
                "ellipsis (`...`), numpy.newaxis (`None`) "
                "and integer or boolean "
                "arrays are valid indices"
            )
        return super().__getitem__(key)

    def _getitem_slice(self, slobj: slice) -> Self:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._range[slobj]
        return type(self)._simple_new(res, name=self._name)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        if is_integer(other) and other != 0:
            if len(self) == 0 or self.start % other == 0 and self.step % other == 0:
                start = self.start // other
                step = self.step // other
                stop = start + len(self) * step
                new_range = range(start, stop, step or 1)
                return self._simple_new(new_range, name=self._name)
            if len(self) == 1:
                start = self.start // other
                new_range = range(start, start + 1, 1)
                return self._simple_new(new_range, name=self._name)

        return super().__floordiv__(other)

    # --------------------------------------------------------------------
    # Reductions

    def all(self, *args, **kwargs) -> bool:
        return 0 not in self._range

    def any(self, *args, **kwargs) -> bool:
        return any(self._range)

    # --------------------------------------------------------------------

    def _cmp_method(self, other, op):
        if isinstance(other, RangeIndex) and self._range == other._range:
            # Both are immutable so if ._range attr. are equal, shortcut is possible
            return super()._cmp_method(self, op)
        return super()._cmp_method(other, op)

    def _arith_method(self, other, op):
        """
        Parameters
        ----------
        other : Any
        op : callable that accepts 2 params
            perform the binary op
        """

        if isinstance(other, ABCTimedeltaIndex):
            # Defer to TimedeltaIndex implementation
            return NotImplemented
        elif isinstance(other, (timedelta, np.timedelta64)):
            # GH#19333 is_integer evaluated True on timedelta64,
            # so we need to catch these explicitly
            return super()._arith_method(other, op)
        elif lib.is_np_dtype(getattr(other, "dtype", None), "m"):
            # Must be an np.ndarray; GH#22390
            return super()._arith_method(other, op)

        if op in [
            operator.pow,
            ops.rpow,
            operator.mod,
            ops.rmod,
            operator.floordiv,
            ops.rfloordiv,
            divmod,
            ops.rdivmod,
        ]:
            return super()._arith_method(other, op)

        step: Callable | None = None
        if op in [operator.mul, ops.rmul, operator.truediv, ops.rtruediv]:
            step = op

        # TODO: if other is a RangeIndex we may have more efficient options
        right = extract_array(other, extract_numpy=True, extract_range=True)
        left = self

        try:
            # apply if we have an override
            if step:
                with np.errstate(all="ignore"):
                    rstep = step(left.step, right)

                # we don't have a representable op
                # so return a base index
                if not is_integer(rstep) or not rstep:
                    raise ValueError

            # GH#53255
            else:
                rstep = -left.step if op == ops.rsub else left.step

            with np.errstate(all="ignore"):
                rstart = op(left.start, right)
                rstop = op(left.stop, right)

            res_name = ops.get_op_result_name(self, other)
            result = type(self)(rstart, rstop, rstep, name=res_name)

            # for compat with numpy / Index with int64 dtype
            # even if we can represent as a RangeIndex, return
            # as a float64 Index if we have float-like descriptors
            if not all(is_integer(x) for x in [rstart, rstop, rstep]):
                result = result.astype("float64")

            return result

        except (ValueError, TypeError, ZeroDivisionError):
            # test_arithmetic_explicit_conversions
            return super()._arith_method(other, op)

    # error: Return type "Index" of "take" incompatible with return type
    # "RangeIndex" in supertype "Index"
    def take(  # type: ignore[override]
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Index:
        if kwargs:
            nv.validate_take((), kwargs)
        if is_scalar(indices):
            raise TypeError("Expected indices to be array-like")
        indices = ensure_platform_int(indices)

        # raise an exception if allow_fill is True and fill_value is not None
        self._maybe_disallow_fill(allow_fill, fill_value, indices)

        if len(indices) == 0:
            taken = np.array([], dtype=self.dtype)
        else:
            ind_max = indices.max()
            if ind_max >= len(self):
                raise IndexError(
                    f"index {ind_max} is out of bounds for axis 0 with size {len(self)}"
                )
            ind_min = indices.min()
            if ind_min < -len(self):
                raise IndexError(
                    f"index {ind_min} is out of bounds for axis 0 with size {len(self)}"
                )
            taken = indices.astype(self.dtype, casting="safe")
            if ind_min < 0:
                taken %= len(self)
            if self.step != 1:
                taken *= self.step
            if self.start != 0:
                taken += self.start

        # _constructor so RangeIndex-> Index with an int64 dtype
        return self._constructor._simple_new(taken, name=self.name)