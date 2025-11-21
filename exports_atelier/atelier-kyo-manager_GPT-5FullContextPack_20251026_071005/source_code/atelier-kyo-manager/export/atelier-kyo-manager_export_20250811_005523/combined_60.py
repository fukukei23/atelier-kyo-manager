
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\euclidtools.py ===
"""Euclidean algorithms, GCDs, LCMs and polynomial remainder sequences. """


from sympy.polys.densearith import (
    dup_sub_mul,
    dup_neg, dmp_neg,
    dmp_add,
    dmp_sub,
    dup_mul, dmp_mul,
    dmp_pow,
    dup_div, dmp_div,
    dup_rem,
    dup_quo, dmp_quo,
    dup_prem, dmp_prem,
    dup_mul_ground, dmp_mul_ground,
    dmp_mul_term,
    dup_quo_ground, dmp_quo_ground,
    dup_max_norm, dmp_max_norm)
from sympy.polys.densebasic import (
    dup_strip, dmp_raise,
    dmp_zero, dmp_one, dmp_ground,
    dmp_one_p, dmp_zero_p,
    dmp_zeros,
    dup_degree, dmp_degree, dmp_degree_in,
    dup_LC, dmp_LC, dmp_ground_LC,
    dmp_multi_deflate, dmp_inflate,
    dup_convert, dmp_convert,
    dmp_apply_pairs)
from sympy.polys.densetools import (
    dup_clear_denoms, dmp_clear_denoms,
    dup_diff, dmp_diff,
    dup_eval, dmp_eval, dmp_eval_in,
    dup_trunc, dmp_ground_trunc,
    dup_monic, dmp_ground_monic,
    dup_primitive, dmp_ground_primitive,
    dup_extract, dmp_ground_extract)
from sympy.polys.galoistools import (
    gf_int, gf_crt)
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    HeuristicGCDFailed,
    HomomorphismFailed,
    NotInvertible,
    DomainError)




def dup_half_gcdex(f, g, K):
    """
    Half extended Euclidean algorithm in `F[x]`.

    Returns ``(s, h)`` such that ``h = gcd(f, g)`` and ``s*f = h (mod g)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
    >>> g = x**3 + x**2 - 4*x - 4

    >>> R.dup_half_gcdex(f, g)
    (-1/5*x + 3/5, x + 1)

    """
    if not K.is_Field:
        raise DomainError("Cannot compute half extended GCD over %s" % K)

    a, b = [K.one], []

    while g:
        q, r = dup_div(f, g, K)
        f, g = g, r
        a, b = b, dup_sub_mul(a, q, b, K)

    a = dup_quo_ground(a, dup_LC(f, K), K)
    f = dup_monic(f, K)

    return a, f


def dmp_half_gcdex(f, g, u, K):
    """
    Half extended Euclidean algorithm in `F[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_half_gcdex(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)


def dup_gcdex(f, g, K):
    """
    Extended Euclidean algorithm in `F[x]`.

    Returns ``(s, t, h)`` such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
    >>> g = x**3 + x**2 - 4*x - 4

    >>> R.dup_gcdex(f, g)
    (-1/5*x + 3/5, 1/5*x**2 - 6/5*x + 2, x + 1)

    """
    s, h = dup_half_gcdex(f, g, K)

    F = dup_sub_mul(h, s, f, K)
    t = dup_quo(F, g, K)

    return s, t, h


def dmp_gcdex(f, g, u, K):
    """
    Extended Euclidean algorithm in `F[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_gcdex(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)


def dup_invert(f, g, K):
    """
    Compute multiplicative inverse of `f` modulo `g` in `F[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = x**2 - 1
    >>> g = 2*x - 1
    >>> h = x - 1

    >>> R.dup_invert(f, g)
    -4/3

    >>> R.dup_invert(f, h)
    Traceback (most recent call last):
    ...
    NotInvertible: zero divisor

    """
    s, h = dup_half_gcdex(f, g, K)

    if h == [K.one]:
        return dup_rem(s, g, K)
    else:
        raise NotInvertible("zero divisor")


def dmp_invert(f, g, u, K):
    """
    Compute multiplicative inverse of `f` modulo `g` in `F[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    """
    if not u:
        return dup_invert(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)


def dup_euclidean_prs(f, g, K):
    """
    Euclidean polynomial remainder sequence (PRS) in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    >>> g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    >>> prs = R.dup_euclidean_prs(f, g)

    >>> prs[0]
    x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    >>> prs[1]
    3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    >>> prs[2]
    -5/9*x**4 + 1/9*x**2 - 1/3
    >>> prs[3]
    -117/25*x**2 - 9*x + 441/25
    >>> prs[4]
    233150/19773*x - 102500/6591
    >>> prs[5]
    -1288744821/543589225

    """
    prs = [f, g]
    h = dup_rem(f, g, K)

    while h:
        prs.append(h)
        f, g = g, h
        h = dup_rem(f, g, K)

    return prs


def dmp_euclidean_prs(f, g, u, K):
    """
    Euclidean polynomial remainder sequence (PRS) in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_euclidean_prs(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)


def dup_primitive_prs(f, g, K):
    """
    Primitive polynomial remainder sequence (PRS) in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    >>> g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    >>> prs = R.dup_primitive_prs(f, g)

    >>> prs[0]
    x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    >>> prs[1]
    3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    >>> prs[2]
    -5*x**4 + x**2 - 3
    >>> prs[3]
    13*x**2 + 25*x - 49
    >>> prs[4]
    4663*x - 6150
    >>> prs[5]
    1

    """
    prs = [f, g]
    _, h = dup_primitive(dup_prem(f, g, K), K)

    while h:
        prs.append(h)
        f, g = g, h
        _, h = dup_primitive(dup_prem(f, g, K), K)

    return prs


def dmp_primitive_prs(f, g, u, K):
    """
    Primitive polynomial remainder sequence (PRS) in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_primitive_prs(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)


def dup_inner_subresultants(f, g, K):
    """
    Subresultant PRS algorithm in `K[x]`.

    Computes the subresultant polynomial remainder sequence (PRS)
    and the non-zero scalar subresultants of `f` and `g`.
    By [1] Thm. 3, these are the constants '-c' (- to optimize
    computation of sign).
    The first subdeterminant is set to 1 by convention to match
    the polynomial and the scalar subdeterminants.
    If 'deg(f) < deg(g)', the subresultants of '(g,f)' are computed.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_inner_subresultants(x**2 + 1, x**2 - 1)
    ([x**2 + 1, x**2 - 1, -2], [1, 1, 4])

    References
    ==========

    .. [1] W.S. Brown, The Subresultant PRS Algorithm.
           ACM Transaction of Mathematical Software 4 (1978) 237-249

    """
    n = dup_degree(f)
    m = dup_degree(g)

    if n < m:
        f, g = g, f
        n, m = m, n

    if not f:
        return [], []

    if not g:
        return [f], [K.one]

    R = [f, g]
    d = n - m

    b = (-K.one)**(d + 1)

    h = dup_prem(f, g, K)
    h = dup_mul_ground(h, b, K)

    lc = dup_LC(g, K)
    c = lc**d

    # Conventional first scalar subdeterminant is 1
    S = [K.one, c]
    c = -c

    while h:
        k = dup_degree(h)
        R.append(h)

        f, g, m, d = g, h, k, m - k

        b = -lc * c**d

        h = dup_prem(f, g, K)
        h = dup_quo_ground(h, b, K)

        lc = dup_LC(g, K)

        if d > 1:        # abnormal case
            q = c**(d - 1)
            c = K.quo((-lc)**d, q)
        else:
            c = -lc

        S.append(-c)

    return R, S


def dup_subresultants(f, g, K):
    """
    Computes subresultant PRS of two polynomials in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_subresultants(x**2 + 1, x**2 - 1)
    [x**2 + 1, x**2 - 1, -2]

    """
    return dup_inner_subresultants(f, g, K)[0]


def dup_prs_resultant(f, g, K):
    """
    Resultant algorithm in `K[x]` using subresultant PRS.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_prs_resultant(x**2 + 1, x**2 - 1)
    (4, [x**2 + 1, x**2 - 1, -2])

    """
    if not f or not g:
        return (K.zero, [])

    R, S = dup_inner_subresultants(f, g, K)

    if dup_degree(R[-1]) > 0:
        return (K.zero, R)

    return S[-1], R


def dup_resultant(f, g, K, includePRS=False):
    """
    Computes resultant of two polynomials in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_resultant(x**2 + 1, x**2 - 1)
    4

    """
    if includePRS:
        return dup_prs_resultant(f, g, K)
    return dup_prs_resultant(f, g, K)[0]


def dmp_inner_subresultants(f, g, u, K):
    """
    Subresultant PRS algorithm in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y - y**3 - 4
    >>> g = x**2 + x*y**3 - 9

    >>> a = 3*x*y**4 + y**3 - 27*y + 4
    >>> b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    >>> prs = [f, g, a, b]
    >>> sres = [[1], [1], [3, 0, 0, 0, 0], [-3, 0, 0, -12, 1, 0, -54, 8, 729, -216, 16]]

    >>> R.dmp_inner_subresultants(f, g) == (prs, sres)
    True

    """
    if not u:
        return dup_inner_subresultants(f, g, K)

    n = dmp_degree(f, u)
    m = dmp_degree(g, u)

    if n < m:
        f, g = g, f
        n, m = m, n

    if dmp_zero_p(f, u):
        return [], []

    v = u - 1
    if dmp_zero_p(g, u):
        return [f], [dmp_ground(K.one, v)]

    R = [f, g]
    d = n - m

    b = dmp_pow(dmp_ground(-K.one, v), d + 1, v, K)

    h = dmp_prem(f, g, u, K)
    h = dmp_mul_term(h, b, 0, u, K)

    lc = dmp_LC(g, K)
    c = dmp_pow(lc, d, v, K)

    S = [dmp_ground(K.one, v), c]
    c = dmp_neg(c, v, K)

    while not dmp_zero_p(h, u):
        k = dmp_degree(h, u)
        R.append(h)

        f, g, m, d = g, h, k, m - k

        b = dmp_mul(dmp_neg(lc, v, K),
                    dmp_pow(c, d, v, K), v, K)

        h = dmp_prem(f, g, u, K)
        h = [ dmp_quo(ch, b, v, K) for ch in h ]

        lc = dmp_LC(g, K)

        if d > 1:
            p = dmp_pow(dmp_neg(lc, v, K), d, v, K)
            q = dmp_pow(c, d - 1, v, K)
            c = dmp_quo(p, q, v, K)
        else:
            c = dmp_neg(lc, v, K)

        S.append(dmp_neg(c, v, K))

    return R, S


def dmp_subresultants(f, g, u, K):
    """
    Computes subresultant PRS of two polynomials in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y - y**3 - 4
    >>> g = x**2 + x*y**3 - 9

    >>> a = 3*x*y**4 + y**3 - 27*y + 4
    >>> b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    >>> R.dmp_subresultants(f, g) == [f, g, a, b]
    True

    """
    return dmp_inner_subresultants(f, g, u, K)[0]


def dmp_prs_resultant(f, g, u, K):
    """
    Resultant algorithm in `K[X]` using subresultant PRS.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y - y**3 - 4
    >>> g = x**2 + x*y**3 - 9

    >>> a = 3*x*y**4 + y**3 - 27*y + 4
    >>> b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    >>> res, prs = R.dmp_prs_resultant(f, g)

    >>> res == b             # resultant has n-1 variables
    False
    >>> res == b.drop(x)
    True
    >>> prs == [f, g, a, b]
    True

    """
    if not u:
        return dup_prs_resultant(f, g, K)

    if dmp_zero_p(f, u) or dmp_zero_p(g, u):
        return (dmp_zero(u - 1), [])

    R, S = dmp_inner_subresultants(f, g, u, K)

    if dmp_degree(R[-1], u) > 0:
        return (dmp_zero(u - 1), R)

    return S[-1], R


def dmp_zz_modular_resultant(f, g, p, u, K):
    """
    Compute resultant of `f` and `g` modulo a prime `p`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x + y + 2
    >>> g = 2*x*y + x + 3

    >>> R.dmp_zz_modular_resultant(f, g, 5)
    -2*y**2 + 1

    """
    if not u:
        return gf_int(dup_prs_resultant(f, g, K)[0] % p, p)

    v = u - 1

    n = dmp_degree(f, u)
    m = dmp_degree(g, u)

    N = dmp_degree_in(f, 1, u)
    M = dmp_degree_in(g, 1, u)

    B = n*M + m*N

    D, a = [K.one], -K.one
    r = dmp_zero(v)

    while dup_degree(D) <= B:
        while True:
            a += K.one

            if a == p:
                raise HomomorphismFailed('no luck')

            F = dmp_eval_in(f, gf_int(a, p), 1, u, K)

            if dmp_degree(F, v) == n:
                G = dmp_eval_in(g, gf_int(a, p), 1, u, K)

                if dmp_degree(G, v) == m:
                    break

        R = dmp_zz_modular_resultant(F, G, p, v, K)
        e = dmp_eval(r, a, v, K)

        if not v:
            R = dup_strip([R])
            e = dup_strip([e])
        else:
            R = [R]
            e = [e]

        d = K.invert(dup_eval(D, a, K), p)
        d = dup_mul_ground(D, d, K)
        d = dmp_raise(d, v, 0, K)

        c = dmp_mul(d, dmp_sub(R, e, v, K), v, K)
        r = dmp_add(r, c, v, K)

        r = dmp_ground_trunc(r, p, v, K)

        D = dup_mul(D, [K.one, -a], K)
        D = dup_trunc(D, p, K)

    return r


def _collins_crt(r, R, P, p, K):
    """Wrapper of CRT for Collins's resultant algorithm. """
    return gf_int(gf_crt([r, R], [P, p], K), P*p)


def dmp_zz_collins_resultant(f, g, u, K):
    """
    Collins's modular resultant algorithm in `Z[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x + y + 2
    >>> g = 2*x*y + x + 3

    >>> R.dmp_zz_collins_resultant(f, g)
    -2*y**2 - 5*y + 1

    """

    n = dmp_degree(f, u)
    m = dmp_degree(g, u)

    if n < 0 or m < 0:
        return dmp_zero(u - 1)

    A = dmp_max_norm(f, u, K)
    B = dmp_max_norm(g, u, K)

    a = dmp_ground_LC(f, u, K)
    b = dmp_ground_LC(g, u, K)

    v = u - 1

    B = K(2)*K.factorial(K(n + m))*A**m*B**n
    r, p, P = dmp_zero(v), K.one, K.one

    from sympy.ntheory import nextprime

    while P <= B:
        p = K(nextprime(p))

        while not (a % p) or not (b % p):
            p = K(nextprime(p))

        F = dmp_ground_trunc(f, p, u, K)
        G = dmp_ground_trunc(g, p, u, K)

        try:
            R = dmp_zz_modular_resultant(F, G, p, u, K)
        except HomomorphismFailed:
            continue

        if K.is_one(P):
            r = R
        else:
            r = dmp_apply_pairs(r, R, _collins_crt, (P, p, K), v, K)

        P *= p

    return r


def dmp_qq_collins_resultant(f, g, u, K0):
    """
    Collins's modular resultant algorithm in `Q[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> f = QQ(1,2)*x + y + QQ(2,3)
    >>> g = 2*x*y + x + 3

    >>> R.dmp_qq_collins_resultant(f, g)
    -2*y**2 - 7/3*y + 5/6

    """
    n = dmp_degree(f, u)
    m = dmp_degree(g, u)

    if n < 0 or m < 0:
        return dmp_zero(u - 1)

    K1 = K0.get_ring()

    cf, f = dmp_clear_denoms(f, u, K0, K1)
    cg, g = dmp_clear_denoms(g, u, K0, K1)

    f = dmp_convert(f, u, K0, K1)
    g = dmp_convert(g, u, K0, K1)

    r = dmp_zz_collins_resultant(f, g, u, K1)
    r = dmp_convert(r, u - 1, K1, K0)

    c = K0.convert(cf**m * cg**n, K1)

    return dmp_quo_ground(r, c, u - 1, K0)


def dmp_resultant(f, g, u, K, includePRS=False):
    """
    Computes resultant of two polynomials in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y - y**3 - 4
    >>> g = x**2 + x*y**3 - 9

    >>> R.dmp_resultant(f, g)
    -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    """
    if not u:
        return dup_resultant(f, g, K, includePRS=includePRS)

    if includePRS:
        return dmp_prs_resultant(f, g, u, K)

    if K.is_Field:
        if K.is_QQ and query('USE_COLLINS_RESULTANT'):
            return dmp_qq_collins_resultant(f, g, u, K)
    else:
        if K.is_ZZ and query('USE_COLLINS_RESULTANT'):
            return dmp_zz_collins_resultant(f, g, u, K)

    return dmp_prs_resultant(f, g, u, K)[0]


def dup_discriminant(f, K):
    """
    Computes discriminant of a polynomial in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_discriminant(x**2 + 2*x + 3)
    -8

    """
    d = dup_degree(f)

    if d <= 0:
        return K.zero
    else:
        s = (-1)**((d*(d - 1)) // 2)
        c = dup_LC(f, K)

        r = dup_resultant(f, dup_diff(f, 1, K), K)

        return K.quo(r, c*K(s))


def dmp_discriminant(f, u, K):
    """
    Computes discriminant of a polynomial in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y,z,t = ring("x,y,z,t", ZZ)

    >>> R.dmp_discriminant(x**2*y + x*z + t)
    -4*y*t + z**2

    """
    if not u:
        return dup_discriminant(f, K)

    d, v = dmp_degree(f, u), u - 1

    if d <= 0:
        return dmp_zero(v)
    else:
        s = (-1)**((d*(d - 1)) // 2)
        c = dmp_LC(f, K)

        r = dmp_resultant(f, dmp_diff(f, 1, u, K), u, K)
        c = dmp_mul_ground(c, K(s), v, K)

        return dmp_quo(r, c, v, K)


def _dup_rr_trivial_gcd(f, g, K):
    """Handle trivial cases in GCD algorithm over a ring. """
    if not (f or g):
        return [], [], []
    elif not f:
        if K.is_nonnegative(dup_LC(g, K)):
            return g, [], [K.one]
        else:
            return dup_neg(g, K), [], [-K.one]
    elif not g:
        if K.is_nonnegative(dup_LC(f, K)):
            return f, [K.one], []
        else:
            return dup_neg(f, K), [-K.one], []

    return None


def _dup_ff_trivial_gcd(f, g, K):
    """Handle trivial cases in GCD algorithm over a field. """
    if not (f or g):
        return [], [], []
    elif not f:
        return dup_monic(g, K), [], [dup_LC(g, K)]
    elif not g:
        return dup_monic(f, K), [dup_LC(f, K)], []
    else:
        return None


def _dmp_rr_trivial_gcd(f, g, u, K):
    """Handle trivial cases in GCD algorithm over a ring. """
    zero_f = dmp_zero_p(f, u)
    zero_g = dmp_zero_p(g, u)
    if_contain_one = dmp_one_p(f, u, K) or dmp_one_p(g, u, K)

    if zero_f and zero_g:
        return tuple(dmp_zeros(3, u, K))
    elif zero_f:
        if K.is_nonnegative(dmp_ground_LC(g, u, K)):
            return g, dmp_zero(u), dmp_one(u, K)
        else:
            return dmp_neg(g, u, K), dmp_zero(u), dmp_ground(-K.one, u)
    elif zero_g:
        if K.is_nonnegative(dmp_ground_LC(f, u, K)):
            return f, dmp_one(u, K), dmp_zero(u)
        else:
            return dmp_neg(f, u, K), dmp_ground(-K.one, u), dmp_zero(u)
    elif if_contain_one:
        return dmp_one(u, K), f, g
    elif query('USE_SIMPLIFY_GCD'):
        return _dmp_simplify_gcd(f, g, u, K)
    else:
        return None


def _dmp_ff_trivial_gcd(f, g, u, K):
    """Handle trivial cases in GCD algorithm over a field. """
    zero_f = dmp_zero_p(f, u)
    zero_g = dmp_zero_p(g, u)

    if zero_f and zero_g:
        return tuple(dmp_zeros(3, u, K))
    elif zero_f:
        return (dmp_ground_monic(g, u, K),
                dmp_zero(u),
                dmp_ground(dmp_ground_LC(g, u, K), u))
    elif zero_g:
        return (dmp_ground_monic(f, u, K),
                dmp_ground(dmp_ground_LC(f, u, K), u),
                dmp_zero(u))
    elif query('USE_SIMPLIFY_GCD'):
        return _dmp_simplify_gcd(f, g, u, K)
    else:
        return None


def _dmp_simplify_gcd(f, g, u, K):
    """Try to eliminate `x_0` from GCD computation in `K[X]`. """
    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if df > 0 and dg > 0:
        return None

    if not (df or dg):
        F = dmp_LC(f, K)
        G = dmp_LC(g, K)
    else:
        if not df:
            F = dmp_LC(f, K)
            G = dmp_content(g, u, K)
        else:
            F = dmp_content(f, u, K)
            G = dmp_LC(g, K)

    v = u - 1
    h = dmp_gcd(F, G, v, K)

    cff = [ dmp_quo(cf, h, v, K) for cf in f ]
    cfg = [ dmp_quo(cg, h, v, K) for cg in g ]

    return [h], cff, cfg


def dup_rr_prs_gcd(f, g, K):
    """
    Computes polynomial GCD using subresultants over a ring.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``, ``cff = quo(f, h)``,
    and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_rr_prs_gcd(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    """
    result = _dup_rr_trivial_gcd(f, g, K)

    if result is not None:
        return result

    fc, F = dup_primitive(f, K)
    gc, G = dup_primitive(g, K)

    c = K.gcd(fc, gc)

    h = dup_subresultants(F, G, K)[-1]
    _, h = dup_primitive(h, K)

    c *= K.canonical_unit(dup_LC(h, K))

    h = dup_mul_ground(h, c, K)

    cff = dup_quo(f, h, K)
    cfg = dup_quo(g, h, K)

    return h, cff, cfg


def dup_ff_prs_gcd(f, g, K):
    """
    Computes polynomial GCD using subresultants over a field.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``, ``cff = quo(f, h)``,
    and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_ff_prs_gcd(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    """
    result = _dup_ff_trivial_gcd(f, g, K)

    if result is not None:
        return result

    h = dup_subresultants(f, g, K)[-1]
    h = dup_monic(h, K)

    cff = dup_quo(f, h, K)
    cfg = dup_quo(g, h, K)

    return h, cff, cfg


def dmp_rr_prs_gcd(f, g, u, K):
    """
    Computes polynomial GCD using subresultants over a ring.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``, ``cff = quo(f, h)``,
    and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_rr_prs_gcd(f, g)
    (x + y, x + y, x)

    """
    if not u:
        return dup_rr_prs_gcd(f, g, K)

    result = _dmp_rr_trivial_gcd(f, g, u, K)

    if result is not None:
        return result

    fc, F = dmp_primitive(f, u, K)
    gc, G = dmp_primitive(g, u, K)

    h = dmp_subresultants(F, G, u, K)[-1]
    c, _, _ = dmp_rr_prs_gcd(fc, gc, u - 1, K)

    _, h = dmp_primitive(h, u, K)
    h = dmp_mul_term(h, c, 0, u, K)

    unit = K.canonical_unit(dmp_ground_LC(h, u, K))

    if unit != K.one:
        h = dmp_mul_ground(h, unit, u, K)

    cff = dmp_quo(f, h, u, K)
    cfg = dmp_quo(g, h, u, K)

    return h, cff, cfg


def dmp_ff_prs_gcd(f, g, u, K):
    """
    Computes polynomial GCD using subresultants over a field.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``, ``cff = quo(f, h)``,
    and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y, = ring("x,y", QQ)

    >>> f = QQ(1,2)*x**2 + x*y + QQ(1,2)*y**2
    >>> g = x**2 + x*y

    >>> R.dmp_ff_prs_gcd(f, g)
    (x + y, 1/2*x + 1/2*y, x)

    """
    if not u:
        return dup_ff_prs_gcd(f, g, K)

    result = _dmp_ff_trivial_gcd(f, g, u, K)

    if result is not None:
        return result

    fc, F = dmp_primitive(f, u, K)
    gc, G = dmp_primitive(g, u, K)

    h = dmp_subresultants(F, G, u, K)[-1]
    c, _, _ = dmp_ff_prs_gcd(fc, gc, u - 1, K)

    _, h = dmp_primitive(h, u, K)
    h = dmp_mul_term(h, c, 0, u, K)
    h = dmp_ground_monic(h, u, K)

    cff = dmp_quo(f, h, u, K)
    cfg = dmp_quo(g, h, u, K)

    return h, cff, cfg

HEU_GCD_MAX = 6


def _dup_zz_gcd_interpolate(h, x, K):
    """Interpolate polynomial GCD from integer GCD. """
    f = []

    while h:
        g = h % x

        if g > x // 2:
            g -= x

        f.insert(0, g)
        h = (h - g) // x

    return f


def dup_zz_heu_gcd(f, g, K):
    """
    Heuristic polynomial GCD in `Z[x]`.

    Given univariate polynomials `f` and `g` in `Z[x]`, returns
    their GCD and cofactors, i.e. polynomials ``h``, ``cff`` and ``cfg``
    such that::

          h = gcd(f, g), cff = quo(f, h) and cfg = quo(g, h)

    The algorithm is purely heuristic which means it may fail to compute
    the GCD. This will be signaled by raising an exception. In this case
    you will need to switch to another GCD method.

    The algorithm computes the polynomial GCD by evaluating polynomials
    f and g at certain points and computing (fast) integer GCD of those
    evaluations. The polynomial GCD is recovered from the integer image
    by interpolation.  The final step is to verify if the result is the
    correct GCD. This gives cofactors as a side effect.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_zz_heu_gcd(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    References
    ==========

    .. [1] [Liao95]_

    """
    result = _dup_rr_trivial_gcd(f, g, K)

    if result is not None:
        return result

    df = dup_degree(f)
    dg = dup_degree(g)

    gcd, f, g = dup_extract(f, g, K)

    if df == 0 or dg == 0:
        return [gcd], f, g

    f_norm = dup_max_norm(f, K)
    g_norm = dup_max_norm(g, K)

    B = K(2*min(f_norm, g_norm) + 29)

    x = max(min(B, 99*K.sqrt(B)),
            2*min(f_norm // abs(dup_LC(f, K)),
                  g_norm // abs(dup_LC(g, K))) + 4)

    for i in range(0, HEU_GCD_MAX):
        ff = dup_eval(f, x, K)
        gg = dup_eval(g, x, K)

        if ff and gg:
            h = K.gcd(ff, gg)

            cff = ff // h
            cfg = gg // h

            h = _dup_zz_gcd_interpolate(h, x, K)
            h = dup_primitive(h, K)[1]

            cff_, r = dup_div(f, h, K)

            if not r:
                cfg_, r = dup_div(g, h, K)

                if not r:
                    h = dup_mul_ground(h, gcd, K)
                    return h, cff_, cfg_

            cff = _dup_zz_gcd_interpolate(cff, x, K)

            h, r = dup_div(f, cff, K)

            if not r:
                cfg_, r = dup_div(g, h, K)

                if not r:
                    h = dup_mul_ground(h, gcd, K)
                    return h, cff, cfg_

            cfg = _dup_zz_gcd_interpolate(cfg, x, K)

            h, r = dup_div(g, cfg, K)

            if not r:
                cff_, r = dup_div(f, h, K)

                if not r:
                    h = dup_mul_ground(h, gcd, K)
                    return h, cff_, cfg

        x = 73794*x * K.sqrt(K.sqrt(x)) // 27011

    raise HeuristicGCDFailed('no luck')


def _dmp_zz_gcd_interpolate(h, x, v, K):
    """Interpolate polynomial GCD from integer GCD. """
    f = []

    while not dmp_zero_p(h, v):
        g = dmp_ground_trunc(h, x, v, K)
        f.insert(0, g)

        h = dmp_sub(h, g, v, K)
        h = dmp_quo_ground(h, x, v, K)

    if K.is_negative(dmp_ground_LC(f, v + 1, K)):
        return dmp_neg(f, v + 1, K)
    else:
        return f


def dmp_zz_heu_gcd(f, g, u, K):
    """
    Heuristic polynomial GCD in `Z[X]`.

    Given univariate polynomials `f` and `g` in `Z[X]`, returns
    their GCD and cofactors, i.e. polynomials ``h``, ``cff`` and ``cfg``
    such that::

          h = gcd(f, g), cff = quo(f, h) and cfg = quo(g, h)

    The algorithm is purely heuristic which means it may fail to compute
    the GCD. This will be signaled by raising an exception. In this case
    you will need to switch to another GCD method.

    The algorithm computes the polynomial GCD by evaluating polynomials
    f and g at certain points and computing (fast) integer GCD of those
    evaluations. The polynomial GCD is recovered from the integer image
    by interpolation. The evaluation process reduces f and g variable by
    variable into a large integer.  The final step is to verify if the
    interpolated polynomial is the correct GCD. This gives cofactors of
    the input polynomials as a side effect.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_zz_heu_gcd(f, g)
    (x + y, x + y, x)

    References
    ==========

    .. [1] [Liao95]_

    """
    if not u:
        return dup_zz_heu_gcd(f, g, K)

    result = _dmp_rr_trivial_gcd(f, g, u, K)

    if result is not None:
        return result

    gcd, f, g = dmp_ground_extract(f, g, u, K)

    f_norm = dmp_max_norm(f, u, K)
    g_norm = dmp_max_norm(g, u, K)

    B = K(2*min(f_norm, g_norm) + 29)

    x = max(min(B, 99*K.sqrt(B)),
            2*min(f_norm // abs(dmp_ground_LC(f, u, K)),
                  g_norm // abs(dmp_ground_LC(g, u, K))) + 4)

    for i in range(0, HEU_GCD_MAX):
        ff = dmp_eval(f, x, u, K)
        gg = dmp_eval(g, x, u, K)

        v = u - 1

        if not (dmp_zero_p(ff, v) or dmp_zero_p(gg, v)):
            h, cff, cfg = dmp_zz_heu_gcd(ff, gg, v, K)

            h = _dmp_zz_gcd_interpolate(h, x, v, K)
            h = dmp_ground_primitive(h, u, K)[1]

            cff_, r = dmp_div(f, h, u, K)

            if dmp_zero_p(r, u):
                cfg_, r = dmp_div(g, h, u, K)

                if dmp_zero_p(r, u):
                    h = dmp_mul_ground(h, gcd, u, K)
                    return h, cff_, cfg_

            cff = _dmp_zz_gcd_interpolate(cff, x, v, K)

            h, r = dmp_div(f, cff, u, K)

            if dmp_zero_p(r, u):
                cfg_, r = dmp_div(g, h, u, K)

                if dmp_zero_p(r, u):
                    h = dmp_mul_ground(h, gcd, u, K)
                    return h, cff, cfg_

            cfg = _dmp_zz_gcd_interpolate(cfg, x, v, K)

            h, r = dmp_div(g, cfg, u, K)

            if dmp_zero_p(r, u):
                cff_, r = dmp_div(f, h, u, K)

                if dmp_zero_p(r, u):
                    h = dmp_mul_ground(h, gcd, u, K)
                    return h, cff_, cfg

        x = 73794*x * K.sqrt(K.sqrt(x)) // 27011

    raise HeuristicGCDFailed('no luck')


def dup_qq_heu_gcd(f, g, K0):
    """
    Heuristic polynomial GCD in `Q[x]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = QQ(1,2)*x**2 + QQ(7,4)*x + QQ(3,2)
    >>> g = QQ(1,2)*x**2 + x

    >>> R.dup_qq_heu_gcd(f, g)
    (x + 2, 1/2*x + 3/4, 1/2*x)

    """
    result = _dup_ff_trivial_gcd(f, g, K0)

    if result is not None:
        return result

    K1 = K0.get_ring()

    cf, f = dup_clear_denoms(f, K0, K1)
    cg, g = dup_clear_denoms(g, K0, K1)

    f = dup_convert(f, K0, K1)
    g = dup_convert(g, K0, K1)

    h, cff, cfg = dup_zz_heu_gcd(f, g, K1)

    h = dup_convert(h, K1, K0)

    c = dup_LC(h, K0)
    h = dup_monic(h, K0)

    cff = dup_convert(cff, K1, K0)
    cfg = dup_convert(cfg, K1, K0)

    cff = dup_mul_ground(cff, K0.quo(c, cf), K0)
    cfg = dup_mul_ground(cfg, K0.quo(c, cg), K0)

    return h, cff, cfg


def dmp_qq_heu_gcd(f, g, u, K0):
    """
    Heuristic polynomial GCD in `Q[X]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y, = ring("x,y", QQ)

    >>> f = QQ(1,4)*x**2 + x*y + y**2
    >>> g = QQ(1,2)*x**2 + x*y

    >>> R.dmp_qq_heu_gcd(f, g)
    (x + 2*y, 1/4*x + 1/2*y, 1/2*x)

    """
    result = _dmp_ff_trivial_gcd(f, g, u, K0)

    if result is not None:
        return result

    K1 = K0.get_ring()

    cf, f = dmp_clear_denoms(f, u, K0, K1)
    cg, g = dmp_clear_denoms(g, u, K0, K1)

    f = dmp_convert(f, u, K0, K1)
    g = dmp_convert(g, u, K0, K1)

    h, cff, cfg = dmp_zz_heu_gcd(f, g, u, K1)

    h = dmp_convert(h, u, K1, K0)

    c = dmp_ground_LC(h, u, K0)
    h = dmp_ground_monic(h, u, K0)

    cff = dmp_convert(cff, u, K1, K0)
    cfg = dmp_convert(cfg, u, K1, K0)

    cff = dmp_mul_ground(cff, K0.quo(c, cf), u, K0)
    cfg = dmp_mul_ground(cfg, K0.quo(c, cg), u, K0)

    return h, cff, cfg


def dup_inner_gcd(f, g, K):
    """
    Computes polynomial GCD and cofactors of `f` and `g` in `K[x]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_inner_gcd(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    """
    # XXX: This used to check for K.is_Exact but leads to awkward results when
    # the domain is something like RR[z] e.g.:
    #
    # >>> g, p, q = Poly(1, x).cancel(Poly(51.05*x*y - 1.0, x))
    # >>> g
    # 1.0
    # >>> p
    # Poly(17592186044421.0, x, domain='RR[y]')
    # >>> q
    # Poly(898081097567692.0*y*x - 17592186044421.0, x, domain='RR[y]'))
    #
    # Maybe it would be better to flatten into multivariate polynomials first.
    if K.is_RR or K.is_CC:
        try:
            exact = K.get_exact()
        except DomainError:
            return [K.one], f, g

        f = dup_convert(f, K, exact)
        g = dup_convert(g, K, exact)

        h, cff, cfg = dup_inner_gcd(f, g, exact)

        h = dup_convert(h, exact, K)
        cff = dup_convert(cff, exact, K)
        cfg = dup_convert(cfg, exact, K)

        return h, cff, cfg
    elif K.is_Field:
        if K.is_QQ and query('USE_HEU_GCD'):
            try:
                return dup_qq_heu_gcd(f, g, K)
            except HeuristicGCDFailed:
                pass

        return dup_ff_prs_gcd(f, g, K)
    else:
        if K.is_ZZ and query('USE_HEU_GCD'):
            try:
                return dup_zz_heu_gcd(f, g, K)
            except HeuristicGCDFailed:
                pass

        return dup_rr_prs_gcd(f, g, K)


def _dmp_inner_gcd(f, g, u, K):
    """Helper function for `dmp_inner_gcd()`. """
    if not K.is_Exact:
        try:
            exact = K.get_exact()
        except DomainError:
            return dmp_one(u, K), f, g

        f = dmp_convert(f, u, K, exact)
        g = dmp_convert(g, u, K, exact)

        h, cff, cfg = _dmp_inner_gcd(f, g, u, exact)

        h = dmp_convert(h, u, exact, K)
        cff = dmp_convert(cff, u, exact, K)
        cfg = dmp_convert(cfg, u, exact, K)

        return h, cff, cfg
    elif K.is_Field:
        if K.is_QQ and query('USE_HEU_GCD'):
            try:
                return dmp_qq_heu_gcd(f, g, u, K)
            except HeuristicGCDFailed:
                pass

        return dmp_ff_prs_gcd(f, g, u, K)
    else:
        if K.is_ZZ and query('USE_HEU_GCD'):
            try:
                return dmp_zz_heu_gcd(f, g, u, K)
            except HeuristicGCDFailed:
                pass

        return dmp_rr_prs_gcd(f, g, u, K)


def dmp_inner_gcd(f, g, u, K):
    """
    Computes polynomial GCD and cofactors of `f` and `g` in `K[X]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_inner_gcd(f, g)
    (x + y, x + y, x)

    """
    if not u:
        return dup_inner_gcd(f, g, K)

    J, (f, g) = dmp_multi_deflate((f, g), u, K)
    h, cff, cfg = _dmp_inner_gcd(f, g, u, K)

    return (dmp_inflate(h, J, u, K),
            dmp_inflate(cff, J, u, K),
            dmp_inflate(cfg, J, u, K))


def dup_gcd(f, g, K):
    """
    Computes polynomial GCD of `f` and `g` in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_gcd(x**2 - 1, x**2 - 3*x + 2)
    x - 1

    """
    return dup_inner_gcd(f, g, K)[0]


def dmp_gcd(f, g, u, K):
    """
    Computes polynomial GCD of `f` and `g` in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_gcd(f, g)
    x + y

    """
    return dmp_inner_gcd(f, g, u, K)[0]


def dup_rr_lcm(f, g, K):
    """
    Computes polynomial LCM over a ring in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_rr_lcm(x**2 - 1, x**2 - 3*x + 2)
    x**3 - 2*x**2 - x + 2

    """
    if not f or not g:
        return dmp_zero(0)

    fc, f = dup_primitive(f, K)
    gc, g = dup_primitive(g, K)

    c = K.lcm(fc, gc)

    h = dup_quo(dup_mul(f, g, K),
                dup_gcd(f, g, K), K)

    u = K.canonical_unit(dup_LC(h, K))

    return dup_mul_ground(h, c*u, K)


def dup_ff_lcm(f, g, K):
    """
    Computes polynomial LCM over a field in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = QQ(1,2)*x**2 + QQ(7,4)*x + QQ(3,2)
    >>> g = QQ(1,2)*x**2 + x

    >>> R.dup_ff_lcm(f, g)
    x**3 + 7/2*x**2 + 3*x

    """
    h = dup_quo(dup_mul(f, g, K),
                dup_gcd(f, g, K), K)

    return dup_monic(h, K)


def dup_lcm(f, g, K):
    """
    Computes polynomial LCM of `f` and `g` in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_lcm(x**2 - 1, x**2 - 3*x + 2)
    x**3 - 2*x**2 - x + 2

    """
    if K.is_Field:
        return dup_ff_lcm(f, g, K)
    else:
        return dup_rr_lcm(f, g, K)


def dmp_rr_lcm(f, g, u, K):
    """
    Computes polynomial LCM over a ring in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_rr_lcm(f, g)
    x**3 + 2*x**2*y + x*y**2

    """
    fc, f = dmp_ground_primitive(f, u, K)
    gc, g = dmp_ground_primitive(g, u, K)

    c = K.lcm(fc, gc)

    h = dmp_quo(dmp_mul(f, g, u, K),
                dmp_gcd(f, g, u, K), u, K)

    return dmp_mul_ground(h, c, u, K)


def dmp_ff_lcm(f, g, u, K):
    """
    Computes polynomial LCM over a field in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y, = ring("x,y", QQ)

    >>> f = QQ(1,4)*x**2 + x*y + y**2
    >>> g = QQ(1,2)*x**2 + x*y

    >>> R.dmp_ff_lcm(f, g)
    x**3 + 4*x**2*y + 4*x*y**2

    """
    h = dmp_quo(dmp_mul(f, g, u, K),
                dmp_gcd(f, g, u, K), u, K)

    return dmp_ground_monic(h, u, K)


def dmp_lcm(f, g, u, K):
    """
    Computes polynomial LCM of `f` and `g` in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_lcm(f, g)
    x**3 + 2*x**2*y + x*y**2

    """
    if not u:
        return dup_lcm(f, g, K)

    if K.is_Field:
        return dmp_ff_lcm(f, g, u, K)
    else:
        return dmp_rr_lcm(f, g, u, K)


def dmp_content(f, u, K):
    """
    Returns GCD of multivariate coefficients.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> R.dmp_content(2*x*y + 6*x + 4*y + 12)
    2*y + 6

    """
    cont, v = dmp_LC(f, K), u - 1

    if dmp_zero_p(f, u):
        return cont

    for c in f[1:]:
        cont = dmp_gcd(cont, c, v, K)

        if dmp_one_p(cont, v, K):
            break

    if K.is_negative(dmp_ground_LC(cont, v, K)):
        return dmp_neg(cont, v, K)
    else:
        return cont


def dmp_primitive(f, u, K):
    """
    Returns multivariate content and a primitive polynomial.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> R.dmp_primitive(2*x*y + 6*x + 4*y + 12)
    (2*y + 6, x + 2)

    """
    cont, v = dmp_content(f, u, K), u - 1

    if dmp_zero_p(f, u) or dmp_one_p(cont, v, K):
        return cont, f
    else:
        return cont, [ dmp_quo(c, cont, v, K) for c in f ]


def dup_cancel(f, g, K, include=True):
    """
    Cancel common factors in a rational function `f/g`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_cancel(2*x**2 - 2, x**2 - 2*x + 1)
    (2*x + 2, x - 1)

    """
    return dmp_cancel(f, g, 0, K, include=include)


def dmp_cancel(f, g, u, K, include=True):
    """
    Cancel common factors in a rational function `f/g`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_cancel(2*x**2 - 2, x**2 - 2*x + 1)
    (2*x + 2, x - 1)

    """
    K0 = None

    if K.is_Field and K.has_assoc_Ring:
        K0, K = K, K.get_ring()

        cq, f = dmp_clear_denoms(f, u, K0, K, convert=True)
        cp, g = dmp_clear_denoms(g, u, K0, K, convert=True)
    else:
        cp, cq = K.one, K.one

    _, p, q = dmp_inner_gcd(f, g, u, K)

    if K0 is not None:
        _, cp, cq = K.cofactors(cp, cq)

        p = dmp_convert(p, u, K, K0)
        q = dmp_convert(q, u, K, K0)

        K = K0

    p_neg = K.is_negative(dmp_ground_LC(p, u, K))
    q_neg = K.is_negative(dmp_ground_LC(q, u, K))

    if p_neg and q_neg:
        p, q = dmp_neg(p, u, K), dmp_neg(q, u, K)
    elif p_neg:
        cp, p = -cp, dmp_neg(p, u, K)
    elif q_neg:
        cp, q = -cp, dmp_neg(q, u, K)

    if not include:
        return cp, cq, p, q

    p = dmp_mul_ground(p, cp, u, K)
    q = dmp_mul_ground(q, cq, u, K)

    return p, q

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\ast.py ===
"""
Types used to represent a full function/module as an Abstract Syntax Tree.

Most types are small, and are merely used as tokens in the AST. A tree diagram
has been included below to illustrate the relationships between the AST types.


AST Type Tree
-------------
::

  *Basic*
       |
       |
   CodegenAST
       |
       |--->AssignmentBase
       |             |--->Assignment
       |             |--->AugmentedAssignment
       |                                    |--->AddAugmentedAssignment
       |                                    |--->SubAugmentedAssignment
       |                                    |--->MulAugmentedAssignment
       |                                    |--->DivAugmentedAssignment
       |                                    |--->ModAugmentedAssignment
       |
       |--->CodeBlock
       |
       |
       |--->Token
                |--->Attribute
                |--->For
                |--->String
                |       |--->QuotedString
                |       |--->Comment
                |--->Type
                |       |--->IntBaseType
                |       |              |--->_SizedIntType
                |       |                               |--->SignedIntType
                |       |                               |--->UnsignedIntType
                |       |--->FloatBaseType
                |                        |--->FloatType
                |                        |--->ComplexBaseType
                |                                           |--->ComplexType
                |--->Node
                |       |--->Variable
                |       |           |---> Pointer
                |       |--->FunctionPrototype
                |                            |--->FunctionDefinition
                |--->Element
                |--->Declaration
                |--->While
                |--->Scope
                |--->Stream
                |--->Print
                |--->FunctionCall
                |--->BreakToken
                |--->ContinueToken
                |--->NoneToken
                |--->Return


Predefined types
----------------

A number of ``Type`` instances are provided in the ``sympy.codegen.ast`` module
for convenience. Perhaps the two most common ones for code-generation (of numeric
codes) are ``float32`` and ``float64`` (known as single and double precision respectively).
There are also precision generic versions of Types (for which the codeprinters selects the
underlying data type at time of printing): ``real``, ``integer``, ``complex_``, ``bool_``.

The other ``Type`` instances defined are:

- ``intc``: Integer type used by C's "int".
- ``intp``: Integer type used by C's "unsigned".
- ``int8``, ``int16``, ``int32``, ``int64``: n-bit integers.
- ``uint8``, ``uint16``, ``uint32``, ``uint64``: n-bit unsigned integers.
- ``float80``: known as "extended precision" on modern x86/amd64 hardware.
- ``complex64``: Complex number represented by two ``float32`` numbers
- ``complex128``: Complex number represented by two ``float64`` numbers

Using the nodes
---------------

It is possible to construct simple algorithms using the AST nodes. Let's construct a loop applying
Newton's method::

    >>> from sympy import symbols, cos
    >>> from sympy.codegen.ast import While, Assignment, aug_assign, Print, QuotedString
    >>> t, dx, x = symbols('tol delta val')
    >>> expr = cos(x) - x**3
    >>> whl = While(abs(dx) > t, [
    ...     Assignment(dx, -expr/expr.diff(x)),
    ...     aug_assign(x, '+', dx),
    ...     Print([x])
    ... ])
    >>> from sympy import pycode
    >>> py_str = pycode(whl)
    >>> print(py_str)
    while (abs(delta) > tol):
        delta = (val**3 - math.cos(val))/(-3*val**2 - math.sin(val))
        val += delta
        print(val)
    >>> import math
    >>> tol, val, delta = 1e-5, 0.5, float('inf')
    >>> exec(py_str)
    1.1121416371
    0.909672693737
    0.867263818209
    0.865477135298
    0.865474033111
    >>> print('%3.1g' % (math.cos(val) - val**3))
    -3e-11

If we want to generate Fortran code for the same while loop we simple call ``fcode``::

    >>> from sympy import fcode
    >>> print(fcode(whl, standard=2003, source_format='free'))
    do while (abs(delta) > tol)
       delta = (val**3 - cos(val))/(-3*val**2 - sin(val))
       val = val + delta
       print *, val
    end do

There is a function constructing a loop (or a complete function) like this in
:mod:`sympy.codegen.algorithms`.

"""

from __future__ import annotations
from typing import Any

from collections import defaultdict

from sympy.core.relational import (Ge, Gt, Le, Lt)
from sympy.core import Symbol, Tuple, Dummy
from sympy.core.basic import Basic
from sympy.core.expr import Expr, Atom
from sympy.core.numbers import Float, Integer, oo
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy.utilities.iterables import (iterable, topological_sort,
                                       numbered_symbols, filter_symbols)


def _mk_Tuple(args):
    """
    Create a SymPy Tuple object from an iterable, converting Python strings to
    AST strings.

    Parameters
    ==========

    args: iterable
        Arguments to :class:`sympy.Tuple`.

    Returns
    =======

    sympy.Tuple
    """
    args = [String(arg) if isinstance(arg, str) else arg for arg in args]
    return Tuple(*args)


class CodegenAST(Basic):
    __slots__ = ()


class Token(CodegenAST):
    """ Base class for the AST types.

    Explanation
    ===========

    Defining fields are set in ``_fields``. Attributes (defined in _fields)
    are only allowed to contain instances of Basic (unless atomic, see
    ``String``). The arguments to ``__new__()`` correspond to the attributes in
    the order defined in ``_fields`. The ``defaults`` class attribute is a
    dictionary mapping attribute names to their default values.

    Subclasses should not need to override the ``__new__()`` method. They may
    define a class or static method named ``_construct_<attr>`` for each
    attribute to process the value passed to ``__new__()``. Attributes listed
    in the class attribute ``not_in_args`` are not passed to :class:`~.Basic`.
    """

    __slots__: tuple[str, ...] = ()
    _fields = __slots__
    defaults: dict[str, Any] = {}
    not_in_args: list[str] = []
    indented_args = ['body']

    @property
    def is_Atom(self):
        return len(self._fields) == 0

    @classmethod
    def _get_constructor(cls, attr):
        """ Get the constructor function for an attribute by name. """
        return getattr(cls, '_construct_%s' % attr, lambda x: x)

    @classmethod
    def _construct(cls, attr, arg):
        """ Construct an attribute value from argument passed to ``__new__()``. """
        # arg may be ``NoneToken()``, so comparison is done using == instead of ``is`` operator
        if arg == None:
            return cls.defaults.get(attr, none)
        else:
            if isinstance(arg, Dummy):  # SymPy's replace uses Dummy instances
                return arg
            else:
                return cls._get_constructor(attr)(arg)

    def __new__(cls, *args, **kwargs):
        # Pass through existing instances when given as sole argument
        if len(args) == 1 and not kwargs and isinstance(args[0], cls):
            return args[0]

        if len(args) > len(cls._fields):
            raise ValueError("Too many arguments (%d), expected at most %d" % (len(args), len(cls._fields)))

        attrvals = []

        # Process positional arguments
        for attrname, argval in zip(cls._fields, args):
            if attrname in kwargs:
                raise TypeError('Got multiple values for attribute %r' % attrname)

            attrvals.append(cls._construct(attrname, argval))

        # Process keyword arguments
        for attrname in cls._fields[len(args):]:
            if attrname in kwargs:
                argval = kwargs.pop(attrname)

            elif attrname in cls.defaults:
                argval = cls.defaults[attrname]

            else:
                raise TypeError('No value for %r given and attribute has no default' % attrname)

            attrvals.append(cls._construct(attrname, argval))

        if kwargs:
            raise ValueError("Unknown keyword arguments: %s" % ' '.join(kwargs))

        # Parent constructor
        basic_args = [
            val for attr, val in zip(cls._fields, attrvals)
            if attr not in cls.not_in_args
        ]
        obj = CodegenAST.__new__(cls, *basic_args)

        # Set attributes
        for attr, arg in zip(cls._fields, attrvals):
            setattr(obj, attr, arg)

        return obj

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for attr in self._fields:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def _hashable_content(self):
        return tuple([getattr(self, attr) for attr in self._fields])

    def __hash__(self):
        return super().__hash__()

    def _joiner(self, k, indent_level):
        return (',\n' + ' '*indent_level) if k in self.indented_args else ', '

    def _indented(self, printer, k, v, *args, **kwargs):
        il = printer._context['indent_level']
        def _print(arg):
            if isinstance(arg, Token):
                return printer._print(arg, *args, joiner=self._joiner(k, il), **kwargs)
            else:
                return printer._print(arg, *args, **kwargs)

        if isinstance(v, Tuple):
            joined = self._joiner(k, il).join([_print(arg) for arg in v.args])
            if k in self.indented_args:
                return '(\n' + ' '*il + joined + ',\n' + ' '*(il - 4) + ')'
            else:
                return ('({0},)' if len(v.args) == 1 else '({0})').format(joined)
        else:
            return _print(v)

    def _sympyrepr(self, printer, *args, joiner=', ', **kwargs):
        from sympy.printing.printer import printer_context
        exclude = kwargs.get('exclude', ())
        values = [getattr(self, k) for k in self._fields]
        indent_level = printer._context.get('indent_level', 0)

        arg_reprs = []

        for i, (attr, value) in enumerate(zip(self._fields, values)):
            if attr in exclude:
                continue

            # Skip attributes which have the default value
            if attr in self.defaults and value == self.defaults[attr]:
                continue

            ilvl = indent_level + 4 if attr in self.indented_args else 0
            with printer_context(printer, indent_level=ilvl):
                indented = self._indented(printer, attr, value, *args, **kwargs)
            arg_reprs.append(('{1}' if i == 0 else '{0}={1}').format(attr, indented.lstrip()))

        return "{}({})".format(self.__class__.__name__, joiner.join(arg_reprs))

    _sympystr = _sympyrepr

    def __repr__(self):  # sympy.core.Basic.__repr__ uses sstr
        from sympy.printing import srepr
        return srepr(self)

    def kwargs(self, exclude=(), apply=None):
        """ Get instance's attributes as dict of keyword arguments.

        Parameters
        ==========

        exclude : collection of str
            Collection of keywords to exclude.

        apply : callable, optional
            Function to apply to all values.
        """
        kwargs = {k: getattr(self, k) for k in self._fields if k not in exclude}
        if apply is not None:
            return {k: apply(v) for k, v in kwargs.items()}
        else:
            return kwargs

class BreakToken(Token):
    """ Represents 'break' in C/Python ('exit' in Fortran).

    Use the premade instance ``break_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import break_
    >>> ccode(break_)
    'break'
    >>> fcode(break_, source_format='free')
    'exit'
    """

break_ = BreakToken()


class ContinueToken(Token):
    """ Represents 'continue' in C/Python ('cycle' in Fortran)

    Use the premade instance ``continue_`` or instantiate manually.

    Examples
    ========

    >>> from sympy import ccode, fcode
    >>> from sympy.codegen.ast import continue_
    >>> ccode(continue_)
    'continue'
    >>> fcode(continue_, source_format='free')
    'cycle'
    """

continue_ = ContinueToken()

class NoneToken(Token):
    """ The AST equivalence of Python's NoneType

    The corresponding instance of Python's ``None`` is ``none``.

    Examples
    ========

    >>> from sympy.codegen.ast import none, Variable
    >>> from sympy import pycode
    >>> print(pycode(Variable('x').as_Declaration(value=none)))
    x = None

    """
    def __eq__(self, other):
        return other is None or isinstance(other, NoneToken)

    def _hashable_content(self):
        return ()

    def __hash__(self):
        return super().__hash__()


none = NoneToken()


class AssignmentBase(CodegenAST):
    """ Abstract base class for Assignment and AugmentedAssignment.

    Attributes:
    ===========

    op : str
        Symbol for assignment operator, e.g. "=", "+=", etc.
    """

    def __new__(cls, lhs, rhs):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)

        cls._check_args(lhs, rhs)

        return super().__new__(cls, lhs, rhs)

    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]

    @classmethod
    def _check_args(cls, lhs, rhs):
        """ Check arguments to __new__ and raise exception if any problems found.

        Derived classes may wish to override this.
        """
        from sympy.matrices.expressions.matexpr import (
            MatrixElement, MatrixSymbol)
        from sympy.tensor.indexed import Indexed
        from sympy.tensor.array.expressions import ArrayElement

        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Element, Variable,
                ArrayElement)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))

        # Indexed types implement shape, but don't define it until later. This
        # causes issues in assignment validation. For now, matrices are defined
        # as anything with a shape that is not an Indexed
        lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs, Indexed)
        rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs, Indexed)

        # If lhs and rhs have same structure, then this assignment is ok
        if lhs_is_mat:
            if not rhs_is_mat:
                raise ValueError("Cannot assign a scalar to a matrix.")
            elif lhs.shape != rhs.shape:
                raise ValueError("Dimensions of lhs and rhs do not align.")
        elif rhs_is_mat and not lhs_is_mat:
            raise ValueError("Cannot assign a matrix to a scalar.")


class Assignment(AssignmentBase):
    """
    Represents variable assignment for code generation.

    Parameters
    ==========

    lhs : Expr
        SymPy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Expr
        SymPy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    Examples
    ========

    >>> from sympy import symbols, MatrixSymbol, Matrix
    >>> from sympy.codegen.ast import Assignment
    >>> x, y, z = symbols('x, y, z')
    >>> Assignment(x, y)
    Assignment(x, y)
    >>> Assignment(x, 0)
    Assignment(x, 0)
    >>> A = MatrixSymbol('A', 1, 3)
    >>> mat = Matrix([x, y, z]).T
    >>> Assignment(A, mat)
    Assignment(A, Matrix([[x, y, z]]))
    >>> Assignment(A[0, 1], x)
    Assignment(A[0, 1], x)
    """

    op = ':='


class AugmentedAssignment(AssignmentBase):
    """
    Base class for augmented assignments.

    Attributes:
    ===========

    binop : str
       Symbol for binary operation being applied in the assignment, such as "+",
       "*", etc.
    """
    binop: str | None

    @property
    def op(self):
        return self.binop + '='


class AddAugmentedAssignment(AugmentedAssignment):
    binop = '+'


class SubAugmentedAssignment(AugmentedAssignment):
    binop = '-'


class MulAugmentedAssignment(AugmentedAssignment):
    binop = '*'


class DivAugmentedAssignment(AugmentedAssignment):
    binop = '/'


class ModAugmentedAssignment(AugmentedAssignment):
    binop = '%'


# Mapping from binary op strings to AugmentedAssignment subclasses
augassign_classes = {
    cls.binop: cls for cls in [
        AddAugmentedAssignment, SubAugmentedAssignment, MulAugmentedAssignment,
        DivAugmentedAssignment, ModAugmentedAssignment
    ]
}


def aug_assign(lhs, op, rhs):
    """
    Create 'lhs op= rhs'.

    Explanation
    ===========

    Represents augmented variable assignment for code generation. This is a
    convenience function. You can also use the AugmentedAssignment classes
    directly, like AddAugmentedAssignment(x, y).

    Parameters
    ==========

    lhs : Expr
        SymPy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    op : str
        Operator (+, -, /, \\*, %).

    rhs : Expr
        SymPy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.codegen.ast import aug_assign
    >>> x, y = symbols('x, y')
    >>> aug_assign(x, '+', y)
    AddAugmentedAssignment(x, y)
    """
    if op not in augassign_classes:
        raise ValueError("Unrecognized operator %s" % op)
    return augassign_classes[op](lhs, rhs)


class CodeBlock(CodegenAST):
    """
    Represents a block of code.

    Explanation
    ===========

    For now only assignments are supported. This restriction will be lifted in
    the future.

    Useful attributes on this object are:

    ``left_hand_sides``:
        Tuple of left-hand sides of assignments, in order.
    ``left_hand_sides``:
        Tuple of right-hand sides of assignments, in order.
    ``free_symbols``: Free symbols of the expressions in the right-hand sides
        which do not appear in the left-hand side of an assignment.

    Useful methods on this object are:

    ``topological_sort``:
        Class method. Return a CodeBlock with assignments
        sorted so that variables are assigned before they
        are used.
    ``cse``:
        Return a new CodeBlock with common subexpressions eliminated and
        pulled out as assignments.

    Examples
    ========

    >>> from sympy import symbols, ccode
    >>> from sympy.codegen.ast import CodeBlock, Assignment
    >>> x, y = symbols('x y')
    >>> c = CodeBlock(Assignment(x, 1), Assignment(y, x + 1))
    >>> print(ccode(c))
    x = 1;
    y = x + 1;

    """
    def __new__(cls, *args):
        left_hand_sides = []
        right_hand_sides = []
        for i in args:
            if isinstance(i, Assignment):
                lhs, rhs = i.args
                left_hand_sides.append(lhs)
                right_hand_sides.append(rhs)

        obj = CodegenAST.__new__(cls, *args)

        obj.left_hand_sides = Tuple(*left_hand_sides)
        obj.right_hand_sides = Tuple(*right_hand_sides)
        return obj

    def __iter__(self):
        return iter(self.args)

    def _sympyrepr(self, printer, *args, **kwargs):
        il = printer._context.get('indent_level', 0)
        joiner = ',\n' + ' '*il
        joined = joiner.join(map(printer._print, self.args))
        return ('{}(\n'.format(' '*(il-4) + self.__class__.__name__,) +
                ' '*il + joined + '\n' + ' '*(il - 4) + ')')

    _sympystr = _sympyrepr

    @property
    def free_symbols(self):
        return super().free_symbols - set(self.left_hand_sides)

    @classmethod
    def topological_sort(cls, assignments):
        """
        Return a CodeBlock with topologically sorted assignments so that
        variables are assigned before they are used.

        Examples
        ========

        The existing order of assignments is preserved as much as possible.

        This function assumes that variables are assigned to only once.

        This is a class constructor so that the default constructor for
        CodeBlock can error when variables are used before they are assigned.

        >>> from sympy import symbols
        >>> from sympy.codegen.ast import CodeBlock, Assignment
        >>> x, y, z = symbols('x y z')

        >>> assignments = [
        ...     Assignment(x, y + z),
        ...     Assignment(y, z + 1),
        ...     Assignment(z, 2),
        ... ]
        >>> CodeBlock.topological_sort(assignments)
        CodeBlock(
            Assignment(z, 2),
            Assignment(y, z + 1),
            Assignment(x, y + z)
        )

        """

        if not all(isinstance(i, Assignment) for i in assignments):
            # Will support more things later
            raise NotImplementedError("CodeBlock.topological_sort only supports Assignments")

        if any(isinstance(i, AugmentedAssignment) for i in assignments):
            raise NotImplementedError("CodeBlock.topological_sort does not yet work with AugmentedAssignments")

        # Create a graph where the nodes are assignments and there is a directed edge
        # between nodes that use a variable and nodes that assign that
        # variable, like

        # [(x := 1, y := x + 1), (x := 1, z := y + z), (y := x + 1, z := y + z)]

        # If we then topologically sort these nodes, they will be in
        # assignment order, like

        # x := 1
        # y := x + 1
        # z := y + z

        # A = The nodes
        #
        # enumerate keeps nodes in the same order they are already in if
        # possible. It will also allow us to handle duplicate assignments to
        # the same variable when those are implemented.
        A = list(enumerate(assignments))

        # var_map = {variable: [nodes for which this variable is assigned to]}
        # like {x: [(1, x := y + z), (4, x := 2 * w)], ...}
        var_map = defaultdict(list)
        for node in A:
            i, a = node
            var_map[a.lhs].append(node)

        # E = Edges in the graph
        E = []
        for dst_node in A:
            i, a = dst_node
            for s in a.rhs.free_symbols:
                for src_node in var_map[s]:
                    E.append((src_node, dst_node))

        ordered_assignments = topological_sort([A, E])

        # De-enumerate the result
        return cls(*[a for i, a in ordered_assignments])

    def cse(self, symbols=None, optimizations=None, postprocess=None,
        order='canonical'):
        """
        Return a new code block with common subexpressions eliminated.

        Explanation
        ===========

        See the docstring of :func:`sympy.simplify.cse_main.cse` for more
        information.

        Examples
        ========

        >>> from sympy import symbols, sin
        >>> from sympy.codegen.ast import CodeBlock, Assignment
        >>> x, y, z = symbols('x y z')

        >>> c = CodeBlock(
        ...     Assignment(x, 1),
        ...     Assignment(y, sin(x) + 1),
        ...     Assignment(z, sin(x) - 1),
        ... )
        ...
        >>> c.cse()
        CodeBlock(
            Assignment(x, 1),
            Assignment(x0, sin(x)),
            Assignment(y, x0 + 1),
            Assignment(z, x0 - 1)
        )

        """
        from sympy.simplify.cse_main import cse

        # Check that the CodeBlock only contains assignments to unique variables
        if not all(isinstance(i, Assignment) for i in self.args):
            # Will support more things later
            raise NotImplementedError("CodeBlock.cse only supports Assignments")

        if any(isinstance(i, AugmentedAssignment) for i in self.args):
            raise NotImplementedError("CodeBlock.cse does not yet work with AugmentedAssignments")

        for i, lhs in enumerate(self.left_hand_sides):
            if lhs in self.left_hand_sides[:i]:
                raise NotImplementedError("Duplicate assignments to the same "
                    "variable are not yet supported (%s)" % lhs)

        # Ensure new symbols for subexpressions do not conflict with existing
        existing_symbols = self.atoms(Symbol)
        if symbols is None:
            symbols = numbered_symbols()
        symbols = filter_symbols(symbols, existing_symbols)

        replacements, reduced_exprs = cse(list(self.right_hand_sides),
            symbols=symbols, optimizations=optimizations, postprocess=postprocess,
            order=order)

        new_block = [Assignment(var, expr) for var, expr in
            zip(self.left_hand_sides, reduced_exprs)]
        new_assignments = [Assignment(var, expr) for var, expr in replacements]
        return self.topological_sort(new_assignments + new_block)


class For(Token):
    """Represents a 'for-loop' in the code.

    Expressions are of the form:
        "for target in iter:
            body..."

    Parameters
    ==========

    target : symbol
    iter : iterable
    body : CodeBlock or iterable
!        When passed an iterable it is used to instantiate a CodeBlock.

    Examples
    ========

    >>> from sympy import symbols, Range
    >>> from sympy.codegen.ast import aug_assign, For
    >>> x, i, j, k = symbols('x i j k')
    >>> for_i = For(i, Range(10), [aug_assign(x, '+', i*j*k)])
    >>> for_i  # doctest: -NORMALIZE_WHITESPACE
    For(i, iterable=Range(0, 10, 1), body=CodeBlock(
        AddAugmentedAssignment(x, i*j*k)
    ))
    >>> for_ji = For(j, Range(7), [for_i])
    >>> for_ji  # doctest: -NORMALIZE_WHITESPACE
    For(j, iterable=Range(0, 7, 1), body=CodeBlock(
        For(i, iterable=Range(0, 10, 1), body=CodeBlock(
            AddAugmentedAssignment(x, i*j*k)
        ))
    ))
    >>> for_kji =For(k, Range(5), [for_ji])
    >>> for_kji  # doctest: -NORMALIZE_WHITESPACE
    For(k, iterable=Range(0, 5, 1), body=CodeBlock(
        For(j, iterable=Range(0, 7, 1), body=CodeBlock(
            For(i, iterable=Range(0, 10, 1), body=CodeBlock(
                AddAugmentedAssignment(x, i*j*k)
            ))
        ))
    ))
    """
    __slots__ = _fields = ('target', 'iterable', 'body')
    _construct_target = staticmethod(_sympify)

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

    @classmethod
    def _construct_iterable(cls, itr):
        if not iterable(itr):
            raise TypeError("iterable must be an iterable")
        if isinstance(itr, list):  # _sympify errors on lists because they are mutable
            itr = tuple(itr)
        return _sympify(itr)


class String(Atom, Token):
    """ SymPy object representing a string.

    Atomic object which is not an expression (as opposed to Symbol).

    Parameters
    ==========

    text : str

    Examples
    ========

    >>> from sympy.codegen.ast import String
    >>> f = String('foo')
    >>> f
    foo
    >>> str(f)
    'foo'
    >>> f.text
    'foo'
    >>> print(repr(f))
    String('foo')

    """
    __slots__ = _fields = ('text',)
    not_in_args = ['text']
    is_Atom = True

    @classmethod
    def _construct_text(cls, text):
        if not isinstance(text, str):
            raise TypeError("Argument text is not a string type.")
        return text

    def _sympystr(self, printer, *args, **kwargs):
        return self.text

    def kwargs(self, exclude = (), apply = None):
        return {}

    #to be removed when Atom is given a suitable func
    @property
    def func(self):
        return lambda: self

    def _latex(self, printer):
        from sympy.printing.latex import latex_escape
        return r'\texttt{{"{}"}}'.format(latex_escape(self.text))

class QuotedString(String):
    """ Represents a string which should be printed with quotes. """

class Comment(String):
    """ Represents a comment. """

class Node(Token):
    """ Subclass of Token, carrying the attribute 'attrs' (Tuple)

    Examples
    ========

    >>> from sympy.codegen.ast import Node, value_const, pointer_const
    >>> n1 = Node([value_const])
    >>> n1.attr_params('value_const')  # get the parameters of attribute (by name)
    ()
    >>> from sympy.codegen.fnodes import dimension
    >>> n2 = Node([value_const, dimension(5, 3)])
    >>> n2.attr_params(value_const)  # get the parameters of attribute (by Attribute instance)
    ()
    >>> n2.attr_params('dimension')  # get the parameters of attribute (by name)
    (5, 3)
    >>> n2.attr_params(pointer_const) is None
    True

    """

    __slots__: tuple[str, ...] = ('attrs',)
    _fields = __slots__

    defaults: dict[str, Any] = {'attrs': Tuple()}

    _construct_attrs = staticmethod(_mk_Tuple)

    def attr_params(self, looking_for):
        """ Returns the parameters of the Attribute with name ``looking_for`` in self.attrs """
        for attr in self.attrs:
            if str(attr.name) == str(looking_for):
                return attr.parameters


class Type(Token):
    """ Represents a type.

    Explanation
    ===========

    The naming is a super-set of NumPy naming. Type has a classmethod
    ``from_expr`` which offer type deduction. It also has a method
    ``cast_check`` which casts the argument to its type, possibly raising an
    exception if rounding error is not within tolerances, or if the value is not
    representable by the underlying data type (e.g. unsigned integers).

    Parameters
    ==========

    name : str
        Name of the type, e.g. ``object``, ``int16``, ``float16`` (where the latter two
        would use the ``Type`` sub-classes ``IntType`` and ``FloatType`` respectively).
        If a ``Type`` instance is given, the said instance is returned.

    Examples
    ========

    >>> from sympy.codegen.ast import Type
    >>> t = Type.from_expr(42)
    >>> t
    integer
    >>> print(repr(t))
    IntBaseType(String('integer'))
    >>> from sympy.codegen.ast import uint8
    >>> uint8.cast_check(-1)   # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Minimum value for data type bigger than new value.
    >>> from sympy.codegen.ast import float32
    >>> v6 = 0.123456
    >>> float32.cast_check(v6)
    0.123456
    >>> v10 = 12345.67894
    >>> float32.cast_check(v10)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Casting gives a significantly different value.
    >>> boost_mp50 = Type('boost::multiprecision::cpp_dec_float_50')
    >>> from sympy import cxxcode
    >>> from sympy.codegen.ast import Declaration, Variable
    >>> cxxcode(Declaration(Variable('x', type=boost_mp50)))
    'boost::multiprecision::cpp_dec_float_50 x'

    References
    ==========

    .. [1] https://numpy.org/doc/stable/user/basics.types.html

    """
    __slots__: tuple[str, ...] = ('name',)
    _fields = __slots__

    _construct_name = String

    def _sympystr(self, printer, *args, **kwargs):
        return str(self.name)

    @classmethod
    def from_expr(cls, expr):
        """ Deduces type from an expression or a ``Symbol``.

        Parameters
        ==========

        expr : number or SymPy object
            The type will be deduced from type or properties.

        Examples
        ========

        >>> from sympy.codegen.ast import Type, integer, complex_
        >>> Type.from_expr(2) == integer
        True
        >>> from sympy import Symbol
        >>> Type.from_expr(Symbol('z', complex=True)) == complex_
        True
        >>> Type.from_expr(sum)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Could not deduce type from expr.

        Raises
        ======

        ValueError when type deduction fails.

        """
        if isinstance(expr, (float, Float)):
            return real
        if isinstance(expr, (int, Integer)) or getattr(expr, 'is_integer', False):
            return integer
        if getattr(expr, 'is_real', False):
            return real
        if isinstance(expr, complex) or getattr(expr, 'is_complex', False):
            return complex_
        if isinstance(expr, bool) or getattr(expr, 'is_Relational', False):
            return bool_
        else:
            raise ValueError("Could not deduce type from expr.")

    def _check(self, value):
        pass

    def cast_check(self, value, rtol=None, atol=0, precision_targets=None):
        """ Casts a value to the data type of the instance.

        Parameters
        ==========

        value : number
        rtol : floating point number
            Relative tolerance. (will be deduced if not given).
        atol : floating point number
            Absolute tolerance (in addition to ``rtol``).
        type_aliases : dict
            Maps substitutions for Type, e.g. {integer: int64, real: float32}

        Examples
        ========

        >>> from sympy.codegen.ast import integer, float32, int8
        >>> integer.cast_check(3.0) == 3
        True
        >>> float32.cast_check(1e-40)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Minimum value for data type bigger than new value.
        >>> int8.cast_check(256)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Maximum value for data type smaller than new value.
        >>> v10 = 12345.67894
        >>> float32.cast_check(v10)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: Casting gives a significantly different value.
        >>> from sympy.codegen.ast import float64
        >>> float64.cast_check(v10)
        12345.67894
        >>> from sympy import Float
        >>> v18 = Float('0.123456789012345646')
        >>> float64.cast_check(v18)
        Traceback (most recent call last):
          ...
        ValueError: Casting gives a significantly different value.
        >>> from sympy.codegen.ast import float80
        >>> float80.cast_check(v18)
        0.123456789012345649

        """
        val = sympify(value)

        ten = Integer(10)
        exp10 = getattr(self, 'decimal_dig', None)

        if rtol is None:
            rtol = 1e-15 if exp10 is None else 2.0*ten**(-exp10)

        def tol(num):
            return atol + rtol*abs(num)

        new_val = self.cast_nocheck(value)
        self._check(new_val)

        delta = new_val - val
        if abs(delta) > tol(val):  # rounding, e.g. int(3.5) != 3.5
            raise ValueError("Casting gives a significantly different value.")

        return new_val

    def _latex(self, printer):
        from sympy.printing.latex import latex_escape
        type_name = latex_escape(self.__class__.__name__)
        name = latex_escape(self.name.text)
        return r"\text{{{}}}\left(\texttt{{{}}}\right)".format(type_name, name)


class IntBaseType(Type):
    """ Integer base type, contains no size information. """
    __slots__ = ()
    cast_nocheck = lambda self, i: Integer(int(i))


class _SizedIntType(IntBaseType):
    __slots__ = ('nbits',)
    _fields = Type._fields + __slots__

    _construct_nbits = Integer

    def _check(self, value):
        if value < self.min:
            raise ValueError("Value is too small: %d < %d" % (value, self.min))
        if value > self.max:
            raise ValueError("Value is too big: %d > %d" % (value, self.max))


class SignedIntType(_SizedIntType):
    """ Represents a signed integer type. """
    __slots__ = ()
    @property
    def min(self):
        return -2**(self.nbits-1)

    @property
    def max(self):
        return 2**(self.nbits-1) - 1


class UnsignedIntType(_SizedIntType):
    """ Represents an unsigned integer type. """
    __slots__ = ()
    @property
    def min(self):
        return 0

    @property
    def max(self):
        return 2**self.nbits - 1

two = Integer(2)

class FloatBaseType(Type):
    """ Represents a floating point number type. """
    __slots__ = ()
    cast_nocheck = Float

class FloatType(FloatBaseType):
    """ Represents a floating point type with fixed bit width.

    Base 2 & one sign bit is assumed.

    Parameters
    ==========

    name : str
        Name of the type.
    nbits : integer
        Number of bits used (storage).
    nmant : integer
        Number of bits used to represent the mantissa.
    nexp : integer
        Number of bits used to represent the mantissa.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.codegen.ast import FloatType
    >>> half_precision = FloatType('f16', nbits=16, nmant=10, nexp=5)
    >>> half_precision.max
    65504
    >>> half_precision.tiny == S(2)**-14
    True
    >>> half_precision.eps == S(2)**-10
    True
    >>> half_precision.dig == 3
    True
    >>> half_precision.decimal_dig == 5
    True
    >>> half_precision.cast_check(1.0)
    1.0
    >>> half_precision.cast_check(1e5)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Maximum value for data type smaller than new value.
    """

    __slots__ = ('nbits', 'nmant', 'nexp',)
    _fields = Type._fields + __slots__

    _construct_nbits = _construct_nmant = _construct_nexp = Integer


    @property
    def max_exponent(self):
        """ The largest positive number n, such that 2**(n - 1) is a representable finite value. """
        # cf. C++'s ``std::numeric_limits::max_exponent``
        return two**(self.nexp - 1)

    @property
    def min_exponent(self):
        """ The lowest negative number n, such that 2**(n - 1) is a valid normalized number. """
        # cf. C++'s ``std::numeric_limits::min_exponent``
        return 3 - self.max_exponent

    @property
    def max(self):
        """ Maximum value representable. """
        return (1 - two**-(self.nmant+1))*two**self.max_exponent

    @property
    def tiny(self):
        """ The minimum positive normalized value. """
        # See C macros: FLT_MIN, DBL_MIN, LDBL_MIN
        # or C++'s ``std::numeric_limits::min``
        # or numpy.finfo(dtype).tiny
        return two**(self.min_exponent - 1)


    @property
    def eps(self):
        """ Difference between 1.0 and the next representable value. """
        return two**(-self.nmant)

    @property
    def dig(self):
        """ Number of decimal digits that are guaranteed to be preserved in text.

        When converting text -> float -> text, you are guaranteed that at least ``dig``
        number of digits are preserved with respect to rounding or overflow.
        """
        from sympy.functions import floor, log
        return floor(self.nmant * log(2)/log(10))

    @property
    def decimal_dig(self):
        """ Number of digits needed to store & load without loss.

        Explanation
        ===========

        Number of decimal digits needed to guarantee that two consecutive conversions
        (float -> text -> float) to be idempotent. This is useful when one do not want
        to loose precision due to rounding errors when storing a floating point value
        as text.
        """
        from sympy.functions import ceiling, log
        return ceiling((self.nmant + 1) * log(2)/log(10) + 1)

    def cast_nocheck(self, value):
        """ Casts without checking if out of bounds or subnormal. """
        if value == oo:  # float(oo) or oo
            return float(oo)
        elif value == -oo:  # float(-oo) or -oo
            return float(-oo)
        return Float(str(sympify(value).evalf(self.decimal_dig)), self.decimal_dig)

    def _check(self, value):
        if value < -self.max:
            raise ValueError("Value is too small: %d < %d" % (value, -self.max))
        if value > self.max:
            raise ValueError("Value is too big: %d > %d" % (value, self.max))
        if abs(value) < self.tiny:
            raise ValueError("Smallest (absolute) value for data type bigger than new value.")

class ComplexBaseType(FloatBaseType):

    __slots__ = ()

    def cast_nocheck(self, value):
        """ Casts without checking if out of bounds or subnormal. """
        from sympy.functions import re, im
        return (
            super().cast_nocheck(re(value)) +
            super().cast_nocheck(im(value))*1j
        )

    def _check(self, value):
        from sympy.functions import re, im
        super()._check(re(value))
        super()._check(im(value))


class ComplexType(ComplexBaseType, FloatType):
    """ Represents a complex floating point number. """
    __slots__ = ()


# NumPy types:
intc = IntBaseType('intc')
intp = IntBaseType('intp')
int8 = SignedIntType('int8', 8)
int16 = SignedIntType('int16', 16)
int32 = SignedIntType('int32', 32)
int64 = SignedIntType('int64', 64)
uint8 = UnsignedIntType('uint8', 8)
uint16 = UnsignedIntType('uint16', 16)
uint32 = UnsignedIntType('uint32', 32)
uint64 = UnsignedIntType('uint64', 64)
float16 = FloatType('float16', 16, nexp=5, nmant=10)  # IEEE 754 binary16, Half precision
float32 = FloatType('float32', 32, nexp=8, nmant=23)  # IEEE 754 binary32, Single precision
float64 = FloatType('float64', 64, nexp=11, nmant=52)  # IEEE 754 binary64, Double precision
float80 = FloatType('float80', 80, nexp=15, nmant=63)  # x86 extended precision (1 integer part bit), "long double"
float128 = FloatType('float128', 128, nexp=15, nmant=112)  # IEEE 754 binary128, Quadruple precision
float256 = FloatType('float256', 256, nexp=19, nmant=236)  # IEEE 754 binary256, Octuple precision

complex64 = ComplexType('complex64', nbits=64, **float32.kwargs(exclude=('name', 'nbits')))
complex128 = ComplexType('complex128', nbits=128, **float64.kwargs(exclude=('name', 'nbits')))

# Generic types (precision may be chosen by code printers):
untyped = Type('untyped')
real = FloatBaseType('real')
integer = IntBaseType('integer')
complex_ = ComplexBaseType('complex')
bool_ = Type('bool')


class Attribute(Token):
    """ Attribute (possibly parametrized)

    For use with :class:`sympy.codegen.ast.Node` (which takes instances of
    ``Attribute`` as ``attrs``).

    Parameters
    ==========

    name : str
    parameters : Tuple

    Examples
    ========

    >>> from sympy.codegen.ast import Attribute
    >>> volatile = Attribute('volatile')
    >>> volatile
    volatile
    >>> print(repr(volatile))
    Attribute(String('volatile'))
    >>> a = Attribute('foo', [1, 2, 3])
    >>> a
    foo(1, 2, 3)
    >>> a.parameters == (1, 2, 3)
    True
    """
    __slots__ = _fields = ('name', 'parameters')
    defaults = {'parameters': Tuple()}

    _construct_name = String
    _construct_parameters = staticmethod(_mk_Tuple)

    def _sympystr(self, printer, *args, **kwargs):
        result = str(self.name)
        if self.parameters:
            result += '(%s)' % ', '.join((printer._print(
                arg, *args, **kwargs) for arg in self.parameters))
        return result

value_const = Attribute('value_const')
pointer_const = Attribute('pointer_const')


class Variable(Node):
    """ Represents a variable.

    Parameters
    ==========

    symbol : Symbol
    type : Type (optional)
        Type of the variable.
    attrs : iterable of Attribute instances
        Will be stored as a Tuple.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Variable, float32, integer
    >>> x = Symbol('x')
    >>> v = Variable(x, type=float32)
    >>> v.attrs
    ()
    >>> v == Variable('x')
    False
    >>> v == Variable('x', type=float32)
    True
    >>> v
    Variable(x, type=float32)

    One may also construct a ``Variable`` instance with the type deduced from
    assumptions about the symbol using the ``deduced`` classmethod:

    >>> i = Symbol('i', integer=True)
    >>> v = Variable.deduced(i)
    >>> v.type == integer
    True
    >>> v == Variable('i')
    False
    >>> from sympy.codegen.ast import value_const
    >>> value_const in v.attrs
    False
    >>> w = Variable('w', attrs=[value_const])
    >>> w
    Variable(w, attrs=(value_const,))
    >>> value_const in w.attrs
    True
    >>> w.as_Declaration(value=42)
    Declaration(Variable(w, value=42, attrs=(value_const,)))

    """

    __slots__ = ('symbol', 'type', 'value')
    _fields = __slots__ + Node._fields

    defaults = Node.defaults.copy()
    defaults.update({'type': untyped, 'value': none})

    _construct_symbol = staticmethod(sympify)
    _construct_value = staticmethod(sympify)

    @classmethod
    def deduced(cls, symbol, value=None, attrs=Tuple(), cast_check=True):
        """ Alt. constructor with type deduction from ``Type.from_expr``.

        Deduces type primarily from ``symbol``, secondarily from ``value``.

        Parameters
        ==========

        symbol : Symbol
        value : expr
            (optional) value of the variable.
        attrs : iterable of Attribute instances
        cast_check : bool
            Whether to apply ``Type.cast_check`` on ``value``.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.codegen.ast import Variable, complex_
        >>> n = Symbol('n', integer=True)
        >>> str(Variable.deduced(n).type)
        'integer'
        >>> x = Symbol('x', real=True)
        >>> v = Variable.deduced(x)
        >>> v.type
        real
        >>> z = Symbol('z', complex=True)
        >>> Variable.deduced(z).type == complex_
        True

        """
        if isinstance(symbol, Variable):
            return symbol

        try:
            type_ = Type.from_expr(symbol)
        except ValueError:
            type_ = Type.from_expr(value)

        if value is not None and cast_check:
            value = type_.cast_check(value)
        return cls(symbol, type=type_, value=value, attrs=attrs)

    def as_Declaration(self, **kwargs):
        """ Convenience method for creating a Declaration instance.

        Explanation
        ===========

        If the variable of the Declaration need to wrap a modified
        variable keyword arguments may be passed (overriding e.g.
        the ``value`` of the Variable instance).

        Examples
        ========

        >>> from sympy.codegen.ast import Variable, NoneToken
        >>> x = Variable('x')
        >>> decl1 = x.as_Declaration()
        >>> # value is special NoneToken() which must be tested with == operator
        >>> decl1.variable.value is None  # won't work
        False
        >>> decl1.variable.value == None  # not PEP-8 compliant
        True
        >>> decl1.variable.value == NoneToken()  # OK
        True
        >>> decl2 = x.as_Declaration(value=42.0)
        >>> decl2.variable.value == 42.0
        True

        """
        kw = self.kwargs()
        kw.update(kwargs)
        return Declaration(self.func(**kw))

    def _relation(self, rhs, op):
        try:
            rhs = _sympify(rhs)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, rhs))
        return op(self, rhs, evaluate=False)

    __lt__ = lambda self, other: self._relation(other, Lt)
    __le__ = lambda self, other: self._relation(other, Le)
    __ge__ = lambda self, other: self._relation(other, Ge)
    __gt__ = lambda self, other: self._relation(other, Gt)

class Pointer(Variable):
    """ Represents a pointer. See ``Variable``.

    Examples
    ========

    Can create instances of ``Element``:

    >>> from sympy import Symbol
    >>> from sympy.codegen.ast import Pointer
    >>> i = Symbol('i', integer=True)
    >>> p = Pointer('x')
    >>> p[i+1]
    Element(x, indices=(i + 1,))

    """
    __slots__ = ()

    def __getitem__(self, key):
        try:
            return Element(self.symbol, key)
        except TypeError:
            return Element(self.symbol, (key,))


class Element(Token):
    """ Element in (a possibly N-dimensional) array.

    Examples
    ========

    >>> from sympy.codegen.ast import Element
    >>> elem = Element('x', 'ijk')
    >>> elem.symbol.name == 'x'
    True
    >>> elem.indices
    (i, j, k)
    >>> from sympy import ccode
    >>> ccode(elem)
    'x[i][j][k]'
    >>> ccode(Element('x', 'ijk', strides='lmn', offset='o'))
    'x[i*l + j*m + k*n + o]'

    """
    __slots__ = _fields = ('symbol', 'indices', 'strides', 'offset')
    defaults = {'strides': none, 'offset': none}
    _construct_symbol = staticmethod(sympify)
    _construct_indices = staticmethod(lambda arg: Tuple(*arg))
    _construct_strides = staticmethod(lambda arg: Tuple(*arg))
    _construct_offset = staticmethod(sympify)


class Declaration(Token):
    """ Represents a variable declaration

    Parameters
    ==========

    variable : Variable

    Examples
    ========

    >>> from sympy.codegen.ast import Declaration, NoneToken, untyped
    >>> z = Declaration('z')
    >>> z.variable.type == untyped
    True
    >>> # value is special NoneToken() which must be tested with == operator
    >>> z.variable.value is None  # won't work
    False
    >>> z.variable.value == None  # not PEP-8 compliant
    True
    >>> z.variable.value == NoneToken()  # OK
    True
    """
    __slots__ = _fields = ('variable',)
    _construct_variable = Variable


class While(Token):
    """ Represents a 'for-loop' in the code.

    Expressions are of the form:
        "while condition:
             body..."

    Parameters
    ==========

    condition : expression convertible to Boolean
    body : CodeBlock or iterable
        When passed an iterable it is used to instantiate a CodeBlock.

    Examples
    ========

    >>> from sympy import symbols, Gt, Abs
    >>> from sympy.codegen import aug_assign, Assignment, While
    >>> x, dx = symbols('x dx')
    >>> expr = 1 - x**2
    >>> whl = While(Gt(Abs(dx), 1e-9), [
    ...     Assignment(dx, -expr/expr.diff(x)),
    ...     aug_assign(x, '+', dx)
    ... ])

    """
    __slots__ = _fields = ('condition', 'body')
    _construct_condition = staticmethod(lambda cond: _sympify(cond))

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)


class Scope(Token):
    """ Represents a scope in the code.

    Parameters
    ==========

    body : CodeBlock or iterable
        When passed an iterable it is used to instantiate a CodeBlock.

    """
    __slots__ = _fields = ('body',)

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)


class Stream(Token):
    """ Represents a stream.

    There are two predefined Stream instances ``stdout`` & ``stderr``.

    Parameters
    ==========

    name : str

    Examples
    ========

    >>> from sympy import pycode, Symbol
    >>> from sympy.codegen.ast import Print, stderr, QuotedString
    >>> print(pycode(Print(['x'], file=stderr)))
    print(x, file=sys.stderr)
    >>> x = Symbol('x')
    >>> print(pycode(Print([QuotedString('x')], file=stderr)))  # print literally "x"
    print("x", file=sys.stderr)

    """
    __slots__ = _fields = ('name',)
    _construct_name = String

stdout = Stream('stdout')
stderr = Stream('stderr')


class Print(Token):
    r""" Represents print command in the code.

    Parameters
    ==========

    formatstring : str
    *args : Basic instances (or convertible to such through sympify)

    Examples
    ========

    >>> from sympy.codegen.ast import Print
    >>> from sympy import pycode
    >>> print(pycode(Print('x y'.split(), "coordinate: %12.5g %12.5g\\n")))
    print("coordinate: %12.5g %12.5g\n" % (x, y), end="")

    """

    __slots__ = _fields = ('print_args', 'format_string', 'file')
    defaults = {'format_string': none, 'file': none}

    _construct_print_args = staticmethod(_mk_Tuple)
    _construct_format_string = QuotedString
    _construct_file = Stream


class FunctionPrototype(Node):
    """ Represents a function prototype

    Allows the user to generate forward declaration in e.g. C/C++.

    Parameters
    ==========

    return_type : Type
    name : str
    parameters: iterable of Variable instances
    attrs : iterable of Attribute instances

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'

    """

    __slots__ = ('return_type', 'name', 'parameters')
    _fields: tuple[str, ...] = __slots__ + Node._fields

    _construct_return_type = Type
    _construct_name = String

    @staticmethod
    def _construct_parameters(args):
        def _var(arg):
            if isinstance(arg, Declaration):
                return arg.variable
            elif isinstance(arg, Variable):
                return arg
            else:
                return Variable.deduced(arg)
        return Tuple(*map(_var, args))

    @classmethod
    def from_FunctionDefinition(cls, func_def):
        if not isinstance(func_def, FunctionDefinition):
            raise TypeError("func_def is not an instance of FunctionDefinition")
        return cls(**func_def.kwargs(exclude=('body',)))


class FunctionDefinition(FunctionPrototype):
    """ Represents a function definition in the code.

    Parameters
    ==========

    return_type : Type
    name : str
    parameters: iterable of Variable instances
    body : CodeBlock or iterable
    attrs : iterable of Attribute instances

    Examples
    ========

    >>> from sympy import ccode, symbols
    >>> from sympy.codegen.ast import real, FunctionPrototype
    >>> x, y = symbols('x y', real=True)
    >>> fp = FunctionPrototype(real, 'foo', [x, y])
    >>> ccode(fp)
    'double foo(double x, double y)'
    >>> from sympy.codegen.ast import FunctionDefinition, Return
    >>> body = [Return(x*y)]
    >>> fd = FunctionDefinition.from_FunctionPrototype(fp, body)
    >>> print(ccode(fd))
    double foo(double x, double y){
        return x*y;
    }
    """

    __slots__ = ('body', )
    _fields = FunctionPrototype._fields[:-1] + __slots__ + Node._fields

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

    @classmethod
    def from_FunctionPrototype(cls, func_proto, body):
        if not isinstance(func_proto, FunctionPrototype):
            raise TypeError("func_proto is not an instance of FunctionPrototype")
        return cls(body=body, **func_proto.kwargs())


class Return(Token):
    """ Represents a return command in the code.

    Parameters
    ==========

    return : Basic

    Examples
    ========

    >>> from sympy.codegen.ast import Return
    >>> from sympy.printing.pycode import pycode
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> print(pycode(Return(x)))
    return x

    """
    __slots__ = _fields = ('return',)
    _construct_return=staticmethod(_sympify)


class FunctionCall(Token, Expr):
    """ Represents a call to a function in the code.

    Parameters
    ==========

    name : str
    function_args : Tuple

    Examples
    ========

    >>> from sympy.codegen.ast import FunctionCall
    >>> from sympy import pycode
    >>> fcall = FunctionCall('foo', 'bar baz'.split())
    >>> print(pycode(fcall))
    foo(bar, baz)

    """
    __slots__ = _fields = ('name', 'function_args')

    _construct_name = String
    _construct_function_args = staticmethod(lambda args: Tuple(*args))


class Raise(Token):
    """ Prints as 'raise ...' in Python, 'throw ...' in C++"""
    __slots__ = _fields = ('exception',)


class RuntimeError_(Token):
    """ Represents 'std::runtime_error' in C++ and 'RuntimeError' in Python.

    Note that the latter is uncommon, and you might want to use e.g. ValueError.
    """
    __slots__ = _fields = ('message',)
    _construct_message = String

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\requirements.py ===
# testing/requirements.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""Global database feature support policy.

Provides decorators to mark tests requiring specific feature support from the
target database.

External dialect test suites should subclass SuiteRequirements
to provide specific inclusion/exclusions.

"""

from __future__ import annotations

import os
import platform

from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool


class Requirements:
    pass


class SuiteRequirements(Requirements):
    @property
    def create_table(self):
        """target platform can emit basic CreateTable DDL."""

        return exclusions.open()

    @property
    def drop_table(self):
        """target platform can emit basic DropTable DDL."""

        return exclusions.open()

    @property
    def table_ddl_if_exists(self):
        """target platform supports IF NOT EXISTS / IF EXISTS for tables."""

        return exclusions.closed()

    @property
    def index_ddl_if_exists(self):
        """target platform supports IF NOT EXISTS / IF EXISTS for indexes."""

        return exclusions.closed()

    @property
    def uuid_data_type(self):
        """Return databases that support the UUID datatype."""

        return exclusions.closed()

    @property
    def foreign_keys(self):
        """Target database must support foreign keys."""

        return exclusions.open()

    @property
    def foreign_keys_reflect_as_index(self):
        """Target database creates an index that's reflected for
        foreign keys."""

        return exclusions.closed()

    @property
    def unique_index_reflect_as_unique_constraints(self):
        """Target database reflects unique indexes as unique constrains."""

        return exclusions.closed()

    @property
    def unique_constraints_reflect_as_index(self):
        """Target database reflects unique constraints as indexes."""

        return exclusions.closed()

    @property
    def table_value_constructor(self):
        """Database / dialect supports a query like:

        .. sourcecode:: sql

             SELECT * FROM VALUES ( (c1, c2), (c1, c2), ...)
             AS some_table(col1, col2)

        SQLAlchemy generates this with the :func:`_sql.values` function.

        """
        return exclusions.closed()

    @property
    def standard_cursor_sql(self):
        """Target database passes SQL-92 style statements to cursor.execute()
        when a statement like select() or insert() is run.

        A very small portion of dialect-level tests will ensure that certain
        conditions are present in SQL strings, and these tests use very basic
        SQL that will work on any SQL-like platform in order to assert results.

        It's normally a given for any pep-249 DBAPI that a statement like
        "SELECT id, name FROM table WHERE some_table.id=5" will work.
        However, there are dialects that don't actually produce SQL Strings
        and instead may work with symbolic objects instead, or dialects that
        aren't working with SQL, so for those this requirement can be marked
        as excluded.

        """

        return exclusions.open()

    @property
    def on_update_cascade(self):
        """target database must support ON UPDATE..CASCADE behavior in
        foreign keys."""

        return exclusions.open()

    @property
    def non_updating_cascade(self):
        """target database must *not* support ON UPDATE..CASCADE behavior in
        foreign keys."""
        return exclusions.closed()

    @property
    def deferrable_fks(self):
        return exclusions.closed()

    @property
    def on_update_or_deferrable_fks(self):
        # TODO: exclusions should be composable,
        # somehow only_if([x, y]) isn't working here, negation/conjunctions
        # getting confused.
        return exclusions.only_if(
            lambda: self.on_update_cascade.enabled
            or self.deferrable_fks.enabled
        )

    @property
    def queue_pool(self):
        """target database is using QueuePool"""

        def go(config):
            return isinstance(config.db.pool, QueuePool)

        return exclusions.only_if(go)

    @property
    def self_referential_foreign_keys(self):
        """Target database must support self-referential foreign keys."""

        return exclusions.open()

    @property
    def foreign_key_ddl(self):
        """Target database must support the DDL phrases for FOREIGN KEY."""

        return exclusions.open()

    @property
    def named_constraints(self):
        """target database must support names for constraints."""

        return exclusions.open()

    @property
    def implicitly_named_constraints(self):
        """target database must apply names to unnamed constraints."""

        return exclusions.open()

    @property
    def unusual_column_name_characters(self):
        """target database allows column names that have unusual characters
        in them, such as dots, spaces, slashes, or percent signs.

        The column names are as always in such a case quoted, however the
        DB still needs to support those characters in the name somehow.

        """
        return exclusions.open()

    @property
    def subqueries(self):
        """Target database must support subqueries."""

        return exclusions.open()

    @property
    def offset(self):
        """target database can render OFFSET, or an equivalent, in a
        SELECT.
        """

        return exclusions.open()

    @property
    def bound_limit_offset(self):
        """target database can render LIMIT and/or OFFSET using a bound
        parameter
        """

        return exclusions.open()

    @property
    def sql_expression_limit_offset(self):
        """target database can render LIMIT and/or OFFSET with a complete
        SQL expression, such as one that uses the addition operator.
        parameter
        """

        return exclusions.open()

    @property
    def parens_in_union_contained_select_w_limit_offset(self):
        """Target database must support parenthesized SELECT in UNION
        when LIMIT/OFFSET is specifically present.

        E.g. (SELECT ...) UNION (SELECT ..)

        This is known to fail on SQLite.

        """
        return exclusions.open()

    @property
    def parens_in_union_contained_select_wo_limit_offset(self):
        """Target database must support parenthesized SELECT in UNION
        when OFFSET/LIMIT is specifically not present.

        E.g. (SELECT ... LIMIT ..) UNION (SELECT .. OFFSET ..)

        This is known to fail on SQLite.  It also fails on Oracle
        because without LIMIT/OFFSET, there is currently no step that
        creates an additional subquery.

        """
        return exclusions.open()

    @property
    def boolean_col_expressions(self):
        """Target database must support boolean expressions as columns"""

        return exclusions.closed()

    @property
    def nullable_booleans(self):
        """Target database allows boolean columns to store NULL."""

        return exclusions.open()

    @property
    def nullsordering(self):
        """Target backends that support nulls ordering."""

        return exclusions.closed()

    @property
    def standalone_binds(self):
        """target database/driver supports bound parameters as column
        expressions without being in the context of a typed column.
        """
        return exclusions.open()

    @property
    def standalone_null_binds_whereclause(self):
        """target database/driver supports bound parameters with NULL in the
        WHERE clause, in situations where it has to be typed.

        """
        return exclusions.open()

    @property
    def intersect(self):
        """Target database must support INTERSECT or equivalent."""
        return exclusions.closed()

    @property
    def except_(self):
        """Target database must support EXCEPT or equivalent (i.e. MINUS)."""
        return exclusions.closed()

    @property
    def window_functions(self):
        """Target database must support window functions."""
        return exclusions.closed()

    @property
    def ctes(self):
        """Target database supports CTEs"""

        return exclusions.closed()

    @property
    def ctes_with_update_delete(self):
        """target database supports CTES that ride on top of a normal UPDATE
        or DELETE statement which refers to the CTE in a correlated subquery.

        """

        return exclusions.closed()

    @property
    def ctes_on_dml(self):
        """target database supports CTES which consist of INSERT, UPDATE
        or DELETE *within* the CTE, e.g. WITH x AS (UPDATE....)"""

        return exclusions.closed()

    @property
    def autoincrement_insert(self):
        """target platform generates new surrogate integer primary key values
        when insert() is executed, excluding the pk column."""

        return exclusions.open()

    @property
    def fetch_rows_post_commit(self):
        """target platform will allow cursor.fetchone() to proceed after a
        COMMIT.

        Typically this refers to an INSERT statement with RETURNING which
        is invoked within "autocommit".   If the row can be returned
        after the autocommit, then this rule can be open.

        """

        return exclusions.open()

    @property
    def group_by_complex_expression(self):
        """target platform supports SQL expressions in GROUP BY

        e.g.

        SELECT x + y AS somelabel FROM table GROUP BY x + y

        """

        return exclusions.open()

    @property
    def sane_rowcount(self):
        return exclusions.skip_if(
            lambda config: not config.db.dialect.supports_sane_rowcount,
            "driver doesn't support 'sane' rowcount",
        )

    @property
    def sane_multi_rowcount(self):
        return exclusions.fails_if(
            lambda config: not config.db.dialect.supports_sane_multi_rowcount,
            "driver %(driver)s %(doesnt_support)s 'sane' multi row count",
        )

    @property
    def sane_rowcount_w_returning(self):
        return exclusions.fails_if(
            lambda config: not (
                config.db.dialect.supports_sane_rowcount_returning
            ),
            "driver doesn't support 'sane' rowcount when returning is on",
        )

    @property
    def empty_inserts(self):
        """target platform supports INSERT with no values, i.e.
        INSERT DEFAULT VALUES or equivalent."""

        return exclusions.only_if(
            lambda config: config.db.dialect.supports_empty_insert
            or config.db.dialect.supports_default_values
            or config.db.dialect.supports_default_metavalue,
            "empty inserts not supported",
        )

    @property
    def empty_inserts_executemany(self):
        """target platform supports INSERT with no values, i.e.
        INSERT DEFAULT VALUES or equivalent, within executemany()"""

        return self.empty_inserts

    @property
    def insert_from_select(self):
        """target platform supports INSERT from a SELECT."""

        return exclusions.open()

    @property
    def delete_returning(self):
        """target platform supports DELETE ... RETURNING."""

        return exclusions.only_if(
            lambda config: config.db.dialect.delete_returning,
            "%(database)s %(does_support)s 'DELETE ... RETURNING'",
        )

    @property
    def insert_returning(self):
        """target platform supports INSERT ... RETURNING."""

        return exclusions.only_if(
            lambda config: config.db.dialect.insert_returning,
            "%(database)s %(does_support)s 'INSERT ... RETURNING'",
        )

    @property
    def update_returning(self):
        """target platform supports UPDATE ... RETURNING."""

        return exclusions.only_if(
            lambda config: config.db.dialect.update_returning,
            "%(database)s %(does_support)s 'UPDATE ... RETURNING'",
        )

    @property
    def insert_executemany_returning(self):
        """target platform supports RETURNING when INSERT is used with
        executemany(), e.g. multiple parameter sets, indicating
        as many rows come back as do parameter sets were passed.

        """

        return exclusions.only_if(
            lambda config: config.db.dialect.insert_executemany_returning,
            "%(database)s %(does_support)s 'RETURNING of "
            "multiple rows with INSERT executemany'",
        )

    @property
    def insertmanyvalues(self):
        return exclusions.only_if(
            lambda config: config.db.dialect.supports_multivalues_insert
            and config.db.dialect.insert_returning
            and config.db.dialect.use_insertmanyvalues,
            "%(database)s %(does_support)s 'insertmanyvalues functionality",
        )

    @property
    def tuple_in(self):
        """Target platform supports the syntax
        "(x, y) IN ((x1, y1), (x2, y2), ...)"
        """

        return exclusions.closed()

    @property
    def tuple_in_w_empty(self):
        """Target platform tuple IN w/ empty set"""
        return self.tuple_in

    @property
    def duplicate_names_in_cursor_description(self):
        """target platform supports a SELECT statement that has
        the same name repeated more than once in the columns list."""

        return exclusions.open()

    @property
    def denormalized_names(self):
        """Target database must have 'denormalized', i.e.
        UPPERCASE as case insensitive names."""

        return exclusions.skip_if(
            lambda config: not config.db.dialect.requires_name_normalize,
            "Backend does not require denormalized names.",
        )

    @property
    def multivalues_inserts(self):
        """target database must support multiple VALUES clauses in an
        INSERT statement."""

        return exclusions.skip_if(
            lambda config: not config.db.dialect.supports_multivalues_insert,
            "Backend does not support multirow inserts.",
        )

    @property
    def implements_get_lastrowid(self):
        """target dialect implements the executioncontext.get_lastrowid()
        method without reliance on RETURNING.

        """
        return exclusions.open()

    @property
    def arraysize(self):
        """dialect includes the required pep-249 attribute
        ``cursor.arraysize``"""

        return exclusions.open()

    @property
    def emulated_lastrowid(self):
        """target dialect retrieves cursor.lastrowid, or fetches
        from a database-side function after an insert() construct executes,
        within the get_lastrowid() method.

        Only dialects that "pre-execute", or need RETURNING to get last
        inserted id, would return closed/fail/skip for this.

        """
        return exclusions.closed()

    @property
    def emulated_lastrowid_even_with_sequences(self):
        """target dialect retrieves cursor.lastrowid or an equivalent
        after an insert() construct executes, even if the table has a
        Sequence on it.

        """
        return exclusions.closed()

    @property
    def dbapi_lastrowid(self):
        """target platform includes a 'lastrowid' accessor on the DBAPI
        cursor object.

        """
        return exclusions.closed()

    @property
    def views(self):
        """Target database must support VIEWs."""

        return exclusions.closed()

    @property
    def schemas(self):
        """Target database must support external schemas, and have one
        named 'test_schema'."""

        return only_on(lambda config: config.db.dialect.supports_schemas)

    @property
    def cross_schema_fk_reflection(self):
        """target system must support reflection of inter-schema
        foreign keys"""
        return exclusions.closed()

    @property
    def foreign_key_constraint_name_reflection(self):
        """Target supports reflection of FOREIGN KEY constraints and
        will return the name of the constraint that was used in the
        "CONSTRAINT <name> FOREIGN KEY" DDL.

        MySQL prior to version 8 and MariaDB prior to version 10.5
        don't support this.

        """
        return exclusions.closed()

    @property
    def implicit_default_schema(self):
        """target system has a strong concept of 'default' schema that can
        be referred to implicitly.

        basically, PostgreSQL.

        """
        return exclusions.closed()

    @property
    def default_schema_name_switch(self):
        """target dialect implements provisioning module including
        set_default_schema_on_connection"""

        return exclusions.closed()

    @property
    def server_side_cursors(self):
        """Target dialect must support server side cursors."""

        return exclusions.only_if(
            [lambda config: config.db.dialect.supports_server_side_cursors],
            "no server side cursors support",
        )

    @property
    def sequences(self):
        """Target database must support SEQUENCEs."""

        return exclusions.only_if(
            [lambda config: config.db.dialect.supports_sequences],
            "no sequence support",
        )

    @property
    def no_sequences(self):
        """the opposite of "sequences", DB does not support sequences at
        all."""

        return exclusions.NotPredicate(self.sequences)

    @property
    def sequences_optional(self):
        """Target database supports sequences, but also optionally
        as a means of generating new PK values."""

        return exclusions.only_if(
            [
                lambda config: config.db.dialect.supports_sequences
                and config.db.dialect.sequences_optional
            ],
            "no sequence support, or sequences not optional",
        )

    @property
    def supports_lastrowid(self):
        """target database / driver supports cursor.lastrowid as a means
        of retrieving the last inserted primary key value.

        note that if the target DB supports sequences also, this is still
        assumed to work.  This is a new use case brought on by MariaDB 10.3.

        """
        return exclusions.only_if(
            [lambda config: config.db.dialect.postfetch_lastrowid]
        )

    @property
    def no_lastrowid_support(self):
        """the opposite of supports_lastrowid"""
        return exclusions.only_if(
            [lambda config: not config.db.dialect.postfetch_lastrowid]
        )

    @property
    def reflects_pk_names(self):
        return exclusions.closed()

    @property
    def table_reflection(self):
        """target database has general support for table reflection"""
        return exclusions.open()

    @property
    def reflect_tables_no_columns(self):
        """target database supports creation and reflection of tables with no
        columns, or at least tables that seem to have no columns."""

        return exclusions.closed()

    @property
    def comment_reflection(self):
        """Indicates if the database support table comment reflection"""
        return exclusions.closed()

    @property
    def comment_reflection_full_unicode(self):
        """Indicates if the database support table comment reflection in the
        full unicode range, including emoji etc.
        """
        return exclusions.closed()

    @property
    def constraint_comment_reflection(self):
        """indicates if the database support comments on constraints
        and their reflection"""
        return exclusions.closed()

    @property
    def view_column_reflection(self):
        """target database must support retrieval of the columns in a view,
        similarly to how a table is inspected.

        This does not include the full CREATE VIEW definition.

        """
        return self.views

    @property
    def view_reflection(self):
        """target database must support inspection of the full CREATE VIEW
        definition."""
        return self.views

    @property
    def schema_reflection(self):
        return self.schemas

    @property
    def schema_create_delete(self):
        """target database supports schema create and dropped with
        'CREATE SCHEMA' and 'DROP SCHEMA'"""
        return exclusions.closed()

    @property
    def primary_key_constraint_reflection(self):
        return exclusions.open()

    @property
    def foreign_key_constraint_reflection(self):
        return exclusions.open()

    @property
    def foreign_key_constraint_option_reflection_ondelete(self):
        return exclusions.closed()

    @property
    def fk_constraint_option_reflection_ondelete_restrict(self):
        return exclusions.closed()

    @property
    def fk_constraint_option_reflection_ondelete_noaction(self):
        return exclusions.closed()

    @property
    def foreign_key_constraint_option_reflection_onupdate(self):
        return exclusions.closed()

    @property
    def fk_constraint_option_reflection_onupdate_restrict(self):
        return exclusions.closed()

    @property
    def temp_table_reflection(self):
        return exclusions.open()

    @property
    def temp_table_reflect_indexes(self):
        return self.temp_table_reflection

    @property
    def temp_table_names(self):
        """target dialect supports listing of temporary table names"""
        return exclusions.closed()

    @property
    def has_temp_table(self):
        """target dialect supports checking a single temp table name"""
        return exclusions.closed()

    @property
    def temporary_tables(self):
        """target database supports temporary tables"""
        return exclusions.open()

    @property
    def temporary_views(self):
        """target database supports temporary views"""
        return exclusions.closed()

    @property
    def index_reflection(self):
        return exclusions.open()

    @property
    def index_reflects_included_columns(self):
        return exclusions.closed()

    @property
    def indexes_with_ascdesc(self):
        """target database supports CREATE INDEX with per-column ASC/DESC."""
        return exclusions.open()

    @property
    def reflect_indexes_with_ascdesc(self):
        """target database supports reflecting INDEX with per-column
        ASC/DESC."""
        return exclusions.open()

    @property
    def reflect_indexes_with_ascdesc_as_expression(self):
        """target database supports reflecting INDEX with per-column
        ASC/DESC but reflects them as expressions (like oracle)."""
        return exclusions.closed()

    @property
    def indexes_with_expressions(self):
        """target database supports CREATE INDEX against SQL expressions."""
        return exclusions.closed()

    @property
    def reflect_indexes_with_expressions(self):
        """target database supports reflection of indexes with
        SQL expressions."""
        return exclusions.closed()

    @property
    def unique_constraint_reflection(self):
        """target dialect supports reflection of unique constraints"""
        return exclusions.open()

    @property
    def inline_check_constraint_reflection(self):
        """target dialect supports reflection of inline check constraints"""
        return exclusions.closed()

    @property
    def check_constraint_reflection(self):
        """target dialect supports reflection of check constraints"""
        return exclusions.closed()

    @property
    def duplicate_key_raises_integrity_error(self):
        """target dialect raises IntegrityError when reporting an INSERT
        with a primary key violation.  (hint: it should)

        """
        return exclusions.open()

    @property
    def unbounded_varchar(self):
        """Target database must support VARCHAR with no length"""

        return exclusions.open()

    @property
    def unicode_data_no_special_types(self):
        """Target database/dialect can receive / deliver / compare data with
        non-ASCII characters in plain VARCHAR, TEXT columns, without the need
        for special "national" datatypes like NVARCHAR or similar.

        """
        return exclusions.open()

    @property
    def unicode_data(self):
        """Target database/dialect must support Python unicode objects with
        non-ASCII characters represented, delivered as bound parameters
        as well as in result rows.

        """
        return exclusions.open()

    @property
    def unicode_ddl(self):
        """Target driver must support some degree of non-ascii symbol
        names.
        """
        return exclusions.closed()

    @property
    def symbol_names_w_double_quote(self):
        """Target driver can create tables with a name like 'some " table'"""
        return exclusions.open()

    @property
    def datetime_interval(self):
        """target dialect supports rendering of a datetime.timedelta as a
        literal string, e.g. via the TypeEngine.literal_processor() method.

        """
        return exclusions.closed()

    @property
    def datetime_literals(self):
        """target dialect supports rendering of a date, time, or datetime as a
        literal string, e.g. via the TypeEngine.literal_processor() method.

        """

        return exclusions.closed()

    @property
    def datetime(self):
        """target dialect supports representation of Python
        datetime.datetime() objects."""

        return exclusions.open()

    @property
    def datetime_timezone(self):
        """target dialect supports representation of Python
        datetime.datetime() with tzinfo with DateTime(timezone=True)."""

        return exclusions.closed()

    @property
    def time_timezone(self):
        """target dialect supports representation of Python
        datetime.time() with tzinfo with Time(timezone=True)."""

        return exclusions.closed()

    @property
    def date_implicit_bound(self):
        """target dialect when given a date object will bind it such
        that the database server knows the object is a date, and not
        a plain string.

        """
        return exclusions.open()

    @property
    def time_implicit_bound(self):
        """target dialect when given a time object will bind it such
        that the database server knows the object is a time, and not
        a plain string.

        """
        return exclusions.open()

    @property
    def datetime_implicit_bound(self):
        """target dialect when given a datetime object will bind it such
        that the database server knows the object is a datetime, and not
        a plain string.

        """
        return exclusions.open()

    @property
    def datetime_microseconds(self):
        """target dialect supports representation of Python
        datetime.datetime() with microsecond objects."""

        return exclusions.open()

    @property
    def timestamp_microseconds(self):
        """target dialect supports representation of Python
        datetime.datetime() with microsecond objects but only
        if TIMESTAMP is used."""
        return exclusions.closed()

    @property
    def timestamp_microseconds_implicit_bound(self):
        """target dialect when given a datetime object which also includes
        a microseconds portion when using the TIMESTAMP data type
        will bind it such that the database server knows
        the object is a datetime with microseconds, and not a plain string.

        """
        return self.timestamp_microseconds

    @property
    def datetime_historic(self):
        """target dialect supports representation of Python
        datetime.datetime() objects with historic (pre 1970) values."""

        return exclusions.closed()

    @property
    def date(self):
        """target dialect supports representation of Python
        datetime.date() objects."""

        return exclusions.open()

    @property
    def date_coerces_from_datetime(self):
        """target dialect accepts a datetime object as the target
        of a date column."""

        return exclusions.open()

    @property
    def date_historic(self):
        """target dialect supports representation of Python
        datetime.datetime() objects with historic (pre 1970) values."""

        return exclusions.closed()

    @property
    def time(self):
        """target dialect supports representation of Python
        datetime.time() objects."""

        return exclusions.open()

    @property
    def time_microseconds(self):
        """target dialect supports representation of Python
        datetime.time() with microsecond objects."""

        return exclusions.open()

    @property
    def binary_comparisons(self):
        """target database/driver can allow BLOB/BINARY fields to be compared
        against a bound parameter value.
        """

        return exclusions.open()

    @property
    def binary_literals(self):
        """target backend supports simple binary literals, e.g. an
        expression like:

        .. sourcecode:: sql

            SELECT CAST('foo' AS BINARY)

        Where ``BINARY`` is the type emitted from :class:`.LargeBinary`,
        e.g. it could be ``BLOB`` or similar.

        Basically fails on Oracle.

        """

        return exclusions.open()

    @property
    def autocommit(self):
        """target dialect supports 'AUTOCOMMIT' as an isolation_level"""
        return exclusions.closed()

    @property
    def isolation_level(self):
        """target dialect supports general isolation level settings.

        Note that this requirement, when enabled, also requires that
        the get_isolation_levels() method be implemented.

        """
        return exclusions.closed()

    def get_isolation_levels(self, config):
        """Return a structure of supported isolation levels for the current
        testing dialect.

        The structure indicates to the testing suite what the expected
        "default" isolation should be, as well as the other values that
        are accepted.  The dictionary has two keys, "default" and "supported".
        The "supported" key refers to a list of all supported levels and
        it should include AUTOCOMMIT if the dialect supports it.

        If the :meth:`.DefaultRequirements.isolation_level` requirement is
        not open, then this method has no return value.

        E.g.::

            >>> testing.requirements.get_isolation_levels()
            {
                "default": "READ_COMMITTED",
                "supported": [
                    "SERIALIZABLE", "READ UNCOMMITTED",
                    "READ COMMITTED", "REPEATABLE READ",
                    "AUTOCOMMIT"
                ]
            }
        """
        with config.db.connect() as conn:
            try:
                supported = conn.dialect.get_isolation_level_values(
                    conn.connection.dbapi_connection
                )
            except NotImplementedError:
                return None
            else:
                return {
                    "default": conn.dialect.default_isolation_level,
                    "supported": supported,
                }

    @property
    def get_isolation_level_values(self):
        """target dialect supports the
        :meth:`_engine.Dialect.get_isolation_level_values`
        method added in SQLAlchemy 2.0.

        """

        def go(config):
            with config.db.connect() as conn:
                try:
                    conn.dialect.get_isolation_level_values(
                        conn.connection.dbapi_connection
                    )
                except NotImplementedError:
                    return False
                else:
                    return True

        return exclusions.only_if(go)

    @property
    def dialect_level_isolation_level_param(self):
        """test that the dialect allows the 'isolation_level' argument
        to be handled by DefaultDialect"""

        def go(config):
            try:
                e = create_engine(
                    config.db.url, isolation_level="READ COMMITTED"
                )
            except:
                return False
            else:
                return (
                    e.dialect._on_connect_isolation_level == "READ COMMITTED"
                )

        return exclusions.only_if(go)

    @property
    def array_type(self):
        """Target platform implements a native ARRAY type"""
        return exclusions.closed()

    @property
    def json_type(self):
        """target platform implements a native JSON type."""

        return exclusions.closed()

    @property
    def json_array_indexes(self):
        """target platform supports numeric array indexes
        within a JSON structure"""

        return self.json_type

    @property
    def json_index_supplementary_unicode_element(self):
        return exclusions.open()

    @property
    def legacy_unconditional_json_extract(self):
        """Backend has a JSON_EXTRACT or similar function that returns a
        valid JSON string in all cases.

        Used to test a legacy feature and is not needed.

        """
        return exclusions.closed()

    @property
    def precision_numerics_general(self):
        """target backend has general support for moderately high-precision
        numerics."""
        return exclusions.open()

    @property
    def precision_numerics_enotation_small(self):
        """target backend supports Decimal() objects using E notation
        to represent very small values."""
        return exclusions.closed()

    @property
    def precision_numerics_enotation_large(self):
        """target backend supports Decimal() objects using E notation
        to represent very large values."""
        return exclusions.open()

    @property
    def precision_numerics_many_significant_digits(self):
        """target backend supports values with many digits on both sides,
        such as 319438950232418390.273596, 87673.594069654243

        """
        return exclusions.closed()

    @property
    def cast_precision_numerics_many_significant_digits(self):
        """same as precision_numerics_many_significant_digits but within the
        context of a CAST statement (hello MySQL)

        """
        return self.precision_numerics_many_significant_digits

    @property
    def server_defaults(self):
        """Target backend supports server side defaults for columns"""

        return exclusions.closed()

    @property
    def expression_server_defaults(self):
        """Target backend supports server side defaults with SQL expressions
        for columns"""

        return exclusions.closed()

    @property
    def implicit_decimal_binds(self):
        """target backend will return a selected Decimal as a Decimal, not
        a string.

        e.g.::

            expr = decimal.Decimal("15.7563")

            value = e.scalar(select(literal(expr)))

            assert value == expr

        See :ticket:`4036`

        """

        return exclusions.open()

    @property
    def numeric_received_as_decimal_untyped(self):
        """target backend will return result columns that are explicitly
        against NUMERIC or similar precision-numeric datatypes (not including
        FLOAT or INT types) as Python Decimal objects, and not as floats
        or ints, including when no SQLAlchemy-side typing information is
        associated with the statement (e.g. such as a raw SQL string).

        This should be enabled if either the DBAPI itself returns Decimal
        objects, or if the dialect has set up DBAPI-specific return type
        handlers such that Decimal objects come back automatically.

        """
        return exclusions.open()

    @property
    def nested_aggregates(self):
        """target database can select an aggregate from a subquery that's
        also using an aggregate

        """
        return exclusions.open()

    @property
    def recursive_fk_cascade(self):
        """target database must support ON DELETE CASCADE on a self-referential
        foreign key

        """
        return exclusions.open()

    @property
    def precision_numerics_retains_significant_digits(self):
        """A precision numeric type will return empty significant digits,
        i.e. a value such as 10.000 will come back in Decimal form with
        the .000 maintained."""

        return exclusions.closed()

    @property
    def infinity_floats(self):
        """The Float type can persist and load float('inf'), float('-inf')."""

        return exclusions.closed()

    @property
    def float_or_double_precision_behaves_generically(self):
        return exclusions.closed()

    @property
    def precision_generic_float_type(self):
        """target backend will return native floating point numbers with at
        least seven decimal places when using the generic Float type.

        """
        return exclusions.open()

    @property
    def literal_float_coercion(self):
        """target backend will return the exact float value 15.7563
        with only four significant digits from this statement:

        SELECT :param

        where :param is the Python float 15.7563

        i.e. it does not return 15.75629997253418

        """
        return exclusions.open()

    @property
    def floats_to_four_decimals(self):
        """target backend can return a floating-point number with four
        significant digits (such as 15.7563) accurately
        (i.e. without FP inaccuracies, such as 15.75629997253418).

        """
        return exclusions.open()

    @property
    def fetch_null_from_numeric(self):
        """target backend doesn't crash when you try to select a NUMERIC
        value that has a value of NULL.

        Added to support Pyodbc bug #351.
        """

        return exclusions.open()

    @property
    def float_is_numeric(self):
        """target backend uses Numeric for Float/Dual"""

        return exclusions.open()

    @property
    def text_type(self):
        """Target database must support an unbounded Text() "
        "type such as TEXT or CLOB"""

        return exclusions.open()

    @property
    def empty_strings_varchar(self):
        """target database can persist/return an empty string with a
        varchar.

        """
        return exclusions.open()

    @property
    def empty_strings_text(self):
        """target database can persist/return an empty string with an
        unbounded text."""

        return exclusions.open()

    @property
    def expressions_against_unbounded_text(self):
        """target database supports use of an unbounded textual field in a
        WHERE clause."""

        return exclusions.open()

    @property
    def selectone(self):
        """target driver must support the literal statement 'select 1'"""
        return exclusions.open()

    @property
    def savepoints(self):
        """Target database must support savepoints."""

        return exclusions.closed()

    @property
    def two_phase_transactions(self):
        """Target database must support two-phase transactions."""

        return exclusions.closed()

    @property
    def update_from(self):
        """Target must support UPDATE..FROM syntax"""
        return exclusions.closed()

    @property
    def delete_from(self):
        """Target must support DELETE FROM..FROM or DELETE..USING syntax"""
        return exclusions.closed()

    @property
    def update_where_target_in_subquery(self):
        """Target must support UPDATE (or DELETE) where the same table is
        present in a subquery in the WHERE clause.

        This is an ANSI-standard syntax that apparently MySQL can't handle,
        such as:

        .. sourcecode:: sql

            UPDATE documents SET flag=1 WHERE documents.title IN
                (SELECT max(documents.title) AS title
                    FROM documents GROUP BY documents.user_id
                )

        """
        return exclusions.open()

    @property
    def mod_operator_as_percent_sign(self):
        """target database must use a plain percent '%' as the 'modulus'
        operator."""
        return exclusions.closed()

    @property
    def percent_schema_names(self):
        """target backend supports weird identifiers with percent signs
        in them, e.g. 'some % column'.

        this is a very weird use case but often has problems because of
        DBAPIs that use python formatting.  It's not a critical use
        case either.

        """
        return exclusions.closed()

    @property
    def order_by_col_from_union(self):
        """target database supports ordering by a column from a SELECT
        inside of a UNION

        E.g.:

        .. sourcecode:: sql

            (SELECT id, ...) UNION (SELECT id, ...) ORDER BY id

        """
        return exclusions.open()

    @property
    def order_by_label_with_expression(self):
        """target backend supports ORDER BY a column label within an
        expression.

        Basically this:

        .. sourcecode:: sql

            select data as foo from test order by foo || 'bar'

        Lots of databases including PostgreSQL don't support this,
        so this is off by default.

        """
        return exclusions.closed()

    @property
    def order_by_collation(self):
        def check(config):
            try:
                self.get_order_by_collation(config)
                return False
            except NotImplementedError:
                return True

        return exclusions.skip_if(check)

    def get_order_by_collation(self, config):
        raise NotImplementedError()

    @property
    def unicode_connections(self):
        """Target driver must support non-ASCII characters being passed at
        all.
        """
        return exclusions.open()

    @property
    def graceful_disconnects(self):
        """Target driver must raise a DBAPI-level exception, such as
        InterfaceError, when the underlying connection has been closed
        and the execute() method is called.
        """
        return exclusions.open()

    @property
    def independent_connections(self):
        """
        Target must support simultaneous, independent database connections.
        """
        return exclusions.open()

    @property
    def independent_readonly_connections(self):
        """
        Target must support simultaneous, independent database connections
        that will be used in a readonly fashion.

        """
        return exclusions.open()

    @property
    def skip_mysql_on_windows(self):
        """Catchall for a large variety of MySQL on Windows failures"""
        return exclusions.open()

    @property
    def ad_hoc_engines(self):
        """Test environment must allow ad-hoc engine/connection creation.

        DBs that scale poorly for many connections, even when closed, i.e.
        Oracle, may use the "--low-connections" option which flags this
        requirement as not present.

        """
        return exclusions.skip_if(
            lambda config: config.options.low_connections
        )

    @property
    def no_windows(self):
        return exclusions.skip_if(self._running_on_windows())

    def _running_on_windows(self):
        return exclusions.LambdaPredicate(
            lambda: platform.system() == "Windows",
            description="running on Windows",
        )

    @property
    def timing_intensive(self):
        from . import config

        return config.add_to_marker.timing_intensive

    @property
    def posix(self):
        return exclusions.skip_if(lambda: os.name != "posix")

    @property
    def memory_intensive(self):
        from . import config

        return config.add_to_marker.memory_intensive

    @property
    def threading_with_mock(self):
        """Mark tests that use threading and mock at the same time - stability
        issues have been observed with coverage

        """
        return exclusions.skip_if(
            lambda config: config.options.has_coverage,
            "Stability issues with coverage",
        )

    @property
    def sqlalchemy2_stubs(self):
        def check(config):
            try:
                __import__("sqlalchemy-stubs.ext.mypy")
            except ImportError:
                return False
            else:
                return True

        return exclusions.only_if(check)

    @property
    def no_sqlalchemy2_stubs(self):
        def check(config):
            try:
                __import__("sqlalchemy-stubs.ext.mypy")
            except ImportError:
                return False
            else:
                return True

        return exclusions.skip_if(check)

    @property
    def up_to_date_typealias_type(self):
        # this checks a particular quirk found in typing_extensions <=4.12.0
        # using older python versions like 3.10 or 3.9, we use TypeAliasType
        # from typing_extensions which does not provide for sufficient
        # introspection prior to 4.13.0
        def check(config):
            import typing
            import typing_extensions

            TypeAliasType = getattr(
                typing, "TypeAliasType", typing_extensions.TypeAliasType
            )
            TV = typing.TypeVar("TV")
            TA_generic = TypeAliasType(  # type: ignore
                "TA_generic", typing.List[TV], type_params=(TV,)
            )
            return hasattr(TA_generic[int], "__value__")

        return exclusions.only_if(check)

    @property
    def python38(self):
        return exclusions.only_if(
            lambda: util.py38, "Python 3.8 or above required"
        )

    @property
    def python39(self):
        return exclusions.only_if(
            lambda: util.py39, "Python 3.9 or above required"
        )

    @property
    def python310(self):
        return exclusions.only_if(
            lambda: util.py310, "Python 3.10 or above required"
        )

    @property
    def python311(self):
        return exclusions.only_if(
            lambda: util.py311, "Python 3.11 or above required"
        )

    @property
    def python312(self):
        return exclusions.only_if(
            lambda: util.py312, "Python 3.12 or above required"
        )

    @property
    def fail_python314b1(self):
        return exclusions.fails_if(
            lambda: util.compat.py314b1, "Fails as of python 3.14.0b1"
        )

    @property
    def not_python314(self):
        """This requirement is interim to assist with backporting of
        issue #12405.

        SQLAlchemy 2.0 still includes the ``await_fallback()`` method that
        makes use of ``asyncio.get_event_loop_policy()``.  This is removed
        in SQLAlchemy 2.1.

        """
        return exclusions.skip_if(
            lambda: util.py314, "Python 3.14 or above not supported"
        )

    @property
    def cpython(self):
        return exclusions.only_if(
            lambda: util.cpython, "cPython interpreter needed"
        )

    @property
    def is64bit(self):
        return exclusions.only_if(lambda: util.is64bit, "64bit required")

    @property
    def patch_library(self):
        def check_lib():
            try:
                __import__("patch")
            except ImportError:
                return False
            else:
                return True

        return exclusions.only_if(check_lib, "patch library needed")

    @property
    def predictable_gc(self):
        """target platform must remove all cycles unconditionally when
        gc.collect() is called, as well as clean out unreferenced subclasses.

        """
        return self.cpython

    @property
    def no_coverage(self):
        """Test should be skipped if coverage is enabled.

        This is to block tests that exercise libraries that seem to be
        sensitive to coverage, such as PostgreSQL notice logging.

        """
        return exclusions.skip_if(
            lambda config: config.options.has_coverage,
            "Issues observed when coverage is enabled",
        )

    def _has_mysql_on_windows(self, config):
        return False

    def _has_mysql_fully_case_sensitive(self, config):
        return False

    @property
    def sqlite(self):
        return exclusions.skip_if(lambda: not self._has_sqlite())

    @property
    def cextensions(self):
        return exclusions.skip_if(
            lambda: not util.has_compiled_ext(),
            "Cython extensions not installed",
        )

    def _has_sqlite(self):
        from sqlalchemy import create_engine

        try:
            create_engine("sqlite://")
            return True
        except ImportError:
            return False

    @property
    def async_dialect(self):
        """dialect makes use of await_() to invoke operations on the DBAPI."""

        return exclusions.closed()

    @property
    def asyncio(self):
        return self.greenlet

    @property
    def no_greenlet(self):
        def go(config):
            try:
                import greenlet  # noqa: F401
            except ImportError:
                return True
            else:
                return False

        return exclusions.only_if(go)

    @property
    def greenlet(self):
        def go(config):
            if not _test_asyncio.ENABLE_ASYNCIO:
                return False

            try:
                import greenlet  # noqa: F401
            except ImportError:
                return False
            else:
                return True

        return exclusions.only_if(go)

    @property
    def computed_columns(self):
        "Supports computed columns"
        return exclusions.closed()

    @property
    def computed_columns_stored(self):
        "Supports computed columns with `persisted=True`"
        return exclusions.closed()

    @property
    def computed_columns_virtual(self):
        "Supports computed columns with `persisted=False`"
        return exclusions.closed()

    @property
    def computed_columns_default_persisted(self):
        """If the default persistence is virtual or stored when `persisted`
        is omitted"""
        return exclusions.closed()

    @property
    def computed_columns_reflect_persisted(self):
        """If persistence information is returned by the reflection of
        computed columns"""
        return exclusions.closed()

    @property
    def supports_distinct_on(self):
        """If a backend supports the DISTINCT ON in a select"""
        return exclusions.closed()

    @property
    def supports_is_distinct_from(self):
        """Supports some form of "x IS [NOT] DISTINCT FROM y" construct.
        Different dialects will implement their own flavour, e.g.,
        sqlite will emit "x IS NOT y" instead of "x IS DISTINCT FROM y".

        .. seealso::

            :meth:`.ColumnOperators.is_distinct_from`

        """
        return exclusions.skip_if(
            lambda config: not config.db.dialect.supports_is_distinct_from,
            "driver doesn't support an IS DISTINCT FROM construct",
        )

    @property
    def identity_columns(self):
        """If a backend supports GENERATED { ALWAYS | BY DEFAULT }
        AS IDENTITY"""
        return exclusions.closed()

    @property
    def identity_columns_standard(self):
        """If a backend supports GENERATED { ALWAYS | BY DEFAULT }
        AS IDENTITY with a standard syntax.
        This is mainly to exclude MSSql.
        """
        return exclusions.closed()

    @property
    def regexp_match(self):
        """backend supports the regexp_match operator."""
        return exclusions.closed()

    @property
    def regexp_replace(self):
        """backend supports the regexp_replace operator."""
        return exclusions.closed()

    @property
    def fetch_first(self):
        """backend supports the fetch first clause."""
        return exclusions.closed()

    @property
    def fetch_percent(self):
        """backend supports the fetch first clause with percent."""
        return exclusions.closed()

    @property
    def fetch_ties(self):
        """backend supports the fetch first clause with ties."""
        return exclusions.closed()

    @property
    def fetch_no_order_by(self):
        """backend supports the fetch first without order by"""
        return exclusions.closed()

    @property
    def fetch_offset_with_options(self):
        """backend supports the offset when using fetch first with percent
        or ties. basically this is "not mssql"
        """
        return exclusions.closed()

    @property
    def fetch_expression(self):
        """backend supports fetch / offset with expression in them, like

        SELECT * FROM some_table
        OFFSET 1 + 1 ROWS FETCH FIRST 1 + 1 ROWS ONLY
        """
        return exclusions.closed()

    @property
    def autoincrement_without_sequence(self):
        """If autoincrement=True on a column does not require an explicit
        sequence. This should be false only for oracle.
        """
        return exclusions.open()

    @property
    def generic_classes(self):
        "If X[Y] can be implemented with ``__class_getitem__``. py3.7+"
        return exclusions.open()

    @property
    def json_deserializer_binary(self):
        "indicates if the json_deserializer function is called with bytes"
        return exclusions.closed()

    @property
    def reflect_table_options(self):
        """Target database must support reflecting table_options."""
        return exclusions.closed()

    @property
    def materialized_views(self):
        """Target database must support MATERIALIZED VIEWs."""
        return exclusions.closed()

    @property
    def materialized_views_reflect_pk(self):
        """Target database reflect MATERIALIZED VIEWs pks."""
        return exclusions.closed()

    @property
    def supports_bitwise_or(self):
        """Target database supports bitwise or"""
        return exclusions.closed()

    @property
    def supports_bitwise_and(self):
        """Target database supports bitwise and"""
        return exclusions.closed()

    @property
    def supports_bitwise_not(self):
        """Target database supports bitwise not"""
        return exclusions.closed()

    @property
    def supports_bitwise_xor(self):
        """Target database supports bitwise xor"""
        return exclusions.closed()

    @property
    def supports_bitwise_shift(self):
        """Target database supports bitwise left or right shift"""
        return exclusions.closed()

    @property
    def like_escapes(self):
        """Target backend supports custom ESCAPE characters
        with LIKE comparisons"""
        return exclusions.open()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\densebasic.py ===
"""Basic tools for dense recursive polynomials in ``K[x]`` or ``K[X]``. """


from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key

import random


ninf = float('-inf')


def poly_LC(f, K):
    """
    Return leading coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import poly_LC

    >>> poly_LC([], ZZ)
    0
    >>> poly_LC([ZZ(1), ZZ(2), ZZ(3)], ZZ)
    1

    """
    if not f:
        return K.zero
    else:
        return f[0]


def poly_TC(f, K):
    """
    Return trailing coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import poly_TC

    >>> poly_TC([], ZZ)
    0
    >>> poly_TC([ZZ(1), ZZ(2), ZZ(3)], ZZ)
    3

    """
    if not f:
        return K.zero
    else:
        return f[-1]

dup_LC = dmp_LC = poly_LC
dup_TC = dmp_TC = poly_TC


def dmp_ground_LC(f, u, K):
    """
    Return the ground leading coefficient.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_LC

    >>> f = ZZ.map([[[1], [2, 3]]])

    >>> dmp_ground_LC(f, 2, ZZ)
    1

    """
    while u:
        f = dmp_LC(f, K)
        u -= 1

    return dup_LC(f, K)


def dmp_ground_TC(f, u, K):
    """
    Return the ground trailing coefficient.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_TC

    >>> f = ZZ.map([[[1], [2, 3]]])

    >>> dmp_ground_TC(f, 2, ZZ)
    3

    """
    while u:
        f = dmp_TC(f, K)
        u -= 1

    return dup_TC(f, K)


def dmp_true_LT(f, u, K):
    """
    Return the leading term ``c * x_1**n_1 ... x_k**n_k``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_true_LT

    >>> f = ZZ.map([[4], [2, 0], [3, 0, 0]])

    >>> dmp_true_LT(f, 1, ZZ)
    ((2, 0), 4)

    """
    monom = []

    while u:
        monom.append(len(f) - 1)
        f, u = f[0], u - 1

    if not f:
        monom.append(0)
    else:
        monom.append(len(f) - 1)

    return tuple(monom), dup_LC(f, K)


def dup_degree(f):
    """
    Return the leading degree of ``f`` in ``K[x]``.

    Note that the degree of 0 is negative infinity (``float('-inf')``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_degree

    >>> f = ZZ.map([1, 2, 0, 3])

    >>> dup_degree(f)
    3

    """
    if not f:
        return ninf
    return len(f) - 1


def dmp_degree(f, u):
    """
    Return the leading degree of ``f`` in ``x_0`` in ``K[X]``.

    Note that the degree of 0 is negative infinity (``float('-inf')``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree

    >>> dmp_degree([[[]]], 2)
    -inf

    >>> f = ZZ.map([[2], [1, 2, 3]])

    >>> dmp_degree(f, 1)
    1

    """
    if dmp_zero_p(f, u):
        return ninf
    else:
        return len(f) - 1


def _rec_degree_in(g, v, i, j):
    """Recursive helper function for :func:`dmp_degree_in`."""
    if i == j:
        return dmp_degree(g, v)

    v, i = v - 1, i + 1

    return max(_rec_degree_in(c, v, i, j) for c in g)


def dmp_degree_in(f, j, u):
    """
    Return the leading degree of ``f`` in ``x_j`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree_in

    >>> f = ZZ.map([[2], [1, 2, 3]])

    >>> dmp_degree_in(f, 0, 1)
    1
    >>> dmp_degree_in(f, 1, 1)
    2

    """
    if not j:
        return dmp_degree(f, u)
    if j < 0 or j > u:
        raise IndexError("0 <= j <= %s expected, got %s" % (u, j))

    return _rec_degree_in(f, u, 0, j)


def _rec_degree_list(g, v, i, degs):
    """Recursive helper for :func:`dmp_degree_list`."""
    degs[i] = max(degs[i], dmp_degree(g, v))

    if v > 0:
        v, i = v - 1, i + 1

        for c in g:
            _rec_degree_list(c, v, i, degs)


def dmp_degree_list(f, u):
    """
    Return a list of degrees of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree_list

    >>> f = ZZ.map([[1], [1, 2, 3]])

    >>> dmp_degree_list(f, 1)
    (1, 2)

    """
    degs = [ninf]*(u + 1)
    _rec_degree_list(f, u, 0, degs)
    return tuple(degs)


def dup_strip(f):
    """
    Remove leading zeros from ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dup_strip

    >>> dup_strip([0, 0, 1, 2, 3, 0])
    [1, 2, 3, 0]

    """
    if not f or f[0]:
        return f

    i = 0

    for cf in f:
        if cf:
            break
        else:
            i += 1

    return f[i:]


def dmp_strip(f, u):
    """
    Remove leading zeros from ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_strip

    >>> dmp_strip([[], [0, 1, 2], [1]], 1)
    [[0, 1, 2], [1]]

    """
    if not u:
        return dup_strip(f)

    if dmp_zero_p(f, u):
        return f

    i, v = 0, u - 1

    for c in f:
        if not dmp_zero_p(c, v):
            break
        else:
            i += 1

    if i == len(f):
        return dmp_zero(u)
    else:
        return f[i:]


def _rec_validate(f, g, i, K):
    """Recursive helper for :func:`dmp_validate`."""
    if not isinstance(g, list):
        if K is not None and not K.of_type(g):
            raise TypeError("%s in %s in not of type %s" % (g, f, K.dtype))

        return {i - 1}
    elif not g:
        return {i}
    else:
        levels = set()

        for c in g:
            levels |= _rec_validate(f, c, i + 1, K)

        return levels


def _rec_strip(g, v):
    """Recursive helper for :func:`_rec_strip`."""
    if not v:
        return dup_strip(g)

    w = v - 1

    return dmp_strip([ _rec_strip(c, w) for c in g ], v)


def dmp_validate(f, K=None):
    """
    Return the number of levels in ``f`` and recursively strip it.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_validate

    >>> dmp_validate([[], [0, 1, 2], [1]])
    ([[1, 2], [1]], 1)

    >>> dmp_validate([[1], 1])
    Traceback (most recent call last):
    ...
    ValueError: invalid data structure for a multivariate polynomial

    """
    levels = _rec_validate(f, f, 0, K)

    u = levels.pop()

    if not levels:
        return _rec_strip(f, u), u
    else:
        raise ValueError(
            "invalid data structure for a multivariate polynomial")


def dup_reverse(f):
    """
    Compute ``x**n * f(1/x)``, i.e.: reverse ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_reverse

    >>> f = ZZ.map([1, 2, 3, 0])

    >>> dup_reverse(f)
    [3, 2, 1]

    """
    return dup_strip(list(reversed(f)))


def dup_copy(f):
    """
    Create a new copy of a polynomial ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_copy

    >>> f = ZZ.map([1, 2, 3, 0])

    >>> dup_copy([1, 2, 3, 0])
    [1, 2, 3, 0]

    """
    return list(f)


def dmp_copy(f, u):
    """
    Create a new copy of a polynomial ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_copy

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_copy(f, 1)
    [[1], [1, 2]]

    """
    if not u:
        return list(f)

    v = u - 1

    return [ dmp_copy(c, v) for c in f ]


def dup_to_tuple(f):
    """
    Convert `f` into a tuple.

    This is needed for hashing. This is similar to dup_copy().

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_copy

    >>> f = ZZ.map([1, 2, 3, 0])

    >>> dup_copy([1, 2, 3, 0])
    [1, 2, 3, 0]

    """
    return tuple(f)


def dmp_to_tuple(f, u):
    """
    Convert `f` into a nested tuple of tuples.

    This is needed for hashing.  This is similar to dmp_copy().

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_to_tuple

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_to_tuple(f, 1)
    ((1,), (1, 2))

    """
    if not u:
        return tuple(f)
    v = u - 1

    return tuple(dmp_to_tuple(c, v) for c in f)


def dup_normal(f, K):
    """
    Normalize univariate polynomial in the given domain.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_normal

    >>> dup_normal([0, 1, 2, 3], ZZ)
    [1, 2, 3]

    """
    return dup_strip([ K.normal(c) for c in f ])


def dmp_normal(f, u, K):
    """
    Normalize a multivariate polynomial in the given domain.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_normal

    >>> dmp_normal([[], [0, 1, 2]], 1, ZZ)
    [[1, 2]]

    """
    if not u:
        return dup_normal(f, K)

    v = u - 1

    return dmp_strip([ dmp_normal(c, v, K) for c in f ], u)


def dup_convert(f, K0, K1):
    """
    Convert the ground domain of ``f`` from ``K0`` to ``K1``.

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_convert

    >>> R, x = ring("x", ZZ)

    >>> dup_convert([R(1), R(2)], R.to_domain(), ZZ)
    [1, 2]
    >>> dup_convert([ZZ(1), ZZ(2)], ZZ, R.to_domain())
    [1, 2]

    """
    if K0 is not None and K0 == K1:
        return f
    else:
        return dup_strip([ K1.convert(c, K0) for c in f ])


def dmp_convert(f, u, K0, K1):
    """
    Convert the ground domain of ``f`` from ``K0`` to ``K1``.

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_convert

    >>> R, x = ring("x", ZZ)

    >>> dmp_convert([[R(1)], [R(2)]], 1, R.to_domain(), ZZ)
    [[1], [2]]
    >>> dmp_convert([[ZZ(1)], [ZZ(2)]], 1, ZZ, R.to_domain())
    [[1], [2]]

    """
    if not u:
        return dup_convert(f, K0, K1)
    if K0 is not None and K0 == K1:
        return f

    v = u - 1

    return dmp_strip([ dmp_convert(c, v, K0, K1) for c in f ], u)


def dup_from_sympy(f, K):
    """
    Convert the ground domain of ``f`` from SymPy to ``K``.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_from_sympy

    >>> dup_from_sympy([S(1), S(2)], ZZ) == [ZZ(1), ZZ(2)]
    True

    """
    return dup_strip([ K.from_sympy(c) for c in f ])


def dmp_from_sympy(f, u, K):
    """
    Convert the ground domain of ``f`` from SymPy to ``K``.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_from_sympy

    >>> dmp_from_sympy([[S(1)], [S(2)]], 1, ZZ) == [[ZZ(1)], [ZZ(2)]]
    True

    """
    if not u:
        return dup_from_sympy(f, K)

    v = u - 1

    return dmp_strip([ dmp_from_sympy(c, v, K) for c in f ], u)


def dup_nth(f, n, K):
    """
    Return the ``n``-th coefficient of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_nth

    >>> f = ZZ.map([1, 2, 3])

    >>> dup_nth(f, 0, ZZ)
    3
    >>> dup_nth(f, 4, ZZ)
    0

    """
    if n < 0:
        raise IndexError("'n' must be non-negative, got %i" % n)
    elif n >= len(f):
        return K.zero
    else:
        return f[dup_degree(f) - n]


def dmp_nth(f, n, u, K):
    """
    Return the ``n``-th coefficient of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_nth

    >>> f = ZZ.map([[1], [2], [3]])

    >>> dmp_nth(f, 0, 1, ZZ)
    [3]
    >>> dmp_nth(f, 4, 1, ZZ)
    []

    """
    if n < 0:
        raise IndexError("'n' must be non-negative, got %i" % n)
    elif n >= len(f):
        return dmp_zero(u - 1)
    else:
        return f[dmp_degree(f, u) - n]


def dmp_ground_nth(f, N, u, K):
    """
    Return the ground ``n``-th coefficient of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_nth

    >>> f = ZZ.map([[1], [2, 3]])

    >>> dmp_ground_nth(f, (0, 1), 1, ZZ)
    2

    """
    v = u

    for n in N:
        if n < 0:
            raise IndexError("`n` must be non-negative, got %i" % n)
        elif n >= len(f):
            return K.zero
        else:
            d = dmp_degree(f, v)
            if d == ninf:
                d = -1
            f, v = f[d - n], v - 1

    return f


def dmp_zero_p(f, u):
    """
    Return ``True`` if ``f`` is zero in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_zero_p

    >>> dmp_zero_p([[[[[]]]]], 4)
    True
    >>> dmp_zero_p([[[[[1]]]]], 4)
    False

    """
    while u:
        if len(f) != 1:
            return False

        f = f[0]
        u -= 1

    return not f


def dmp_zero(u):
    """
    Return a multivariate zero.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_zero

    >>> dmp_zero(4)
    [[[[[]]]]]

    """
    r = []

    for i in range(u):
        r = [r]

    return r


def dmp_one_p(f, u, K):
    """
    Return ``True`` if ``f`` is one in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_one_p

    >>> dmp_one_p([[[ZZ(1)]]], 2, ZZ)
    True

    """
    return dmp_ground_p(f, K.one, u)


def dmp_one(u, K):
    """
    Return a multivariate one over ``K``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_one

    >>> dmp_one(2, ZZ)
    [[[1]]]

    """
    return dmp_ground(K.one, u)


def dmp_ground_p(f, c, u):
    """
    Return True if ``f`` is constant in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_ground_p

    >>> dmp_ground_p([[[3]]], 3, 2)
    True
    >>> dmp_ground_p([[[4]]], None, 2)
    True

    """
    if c is not None and not c:
        return dmp_zero_p(f, u)

    while u:
        if len(f) != 1:
            return False
        f = f[0]
        u -= 1

    if c is None:
        return len(f) <= 1
    else:
        return f == [c]


def dmp_ground(c, u):
    """
    Return a multivariate constant.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_ground

    >>> dmp_ground(3, 5)
    [[[[[[3]]]]]]
    >>> dmp_ground(1, -1)
    1

    """
    if not c:
        return dmp_zero(u)

    for i in range(u + 1):
        c = [c]

    return c


def dmp_zeros(n, u, K):
    """
    Return a list of multivariate zeros.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_zeros

    >>> dmp_zeros(3, 2, ZZ)
    [[[[]]], [[[]]], [[[]]]]
    >>> dmp_zeros(3, -1, ZZ)
    [0, 0, 0]

    """
    if not n:
        return []

    if u < 0:
        return [K.zero]*n
    else:
        return [ dmp_zero(u) for i in range(n) ]


def dmp_grounds(c, n, u):
    """
    Return a list of multivariate constants.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_grounds

    >>> dmp_grounds(ZZ(4), 3, 2)
    [[[[4]]], [[[4]]], [[[4]]]]
    >>> dmp_grounds(ZZ(4), 3, -1)
    [4, 4, 4]

    """
    if not n:
        return []

    if u < 0:
        return [c]*n
    else:
        return [ dmp_ground(c, u) for i in range(n) ]


def dmp_negative_p(f, u, K):
    """
    Return ``True`` if ``LC(f)`` is negative.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_negative_p

    >>> dmp_negative_p([[ZZ(1)], [-ZZ(1)]], 1, ZZ)
    False
    >>> dmp_negative_p([[-ZZ(1)], [ZZ(1)]], 1, ZZ)
    True

    """
    return K.is_negative(dmp_ground_LC(f, u, K))


def dmp_positive_p(f, u, K):
    """
    Return ``True`` if ``LC(f)`` is positive.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_positive_p

    >>> dmp_positive_p([[ZZ(1)], [-ZZ(1)]], 1, ZZ)
    True
    >>> dmp_positive_p([[-ZZ(1)], [ZZ(1)]], 1, ZZ)
    False

    """
    return K.is_positive(dmp_ground_LC(f, u, K))


def dup_from_dict(f, K):
    """
    Create a ``K[x]`` polynomial from a ``dict``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_from_dict

    >>> dup_from_dict({(0,): ZZ(7), (2,): ZZ(5), (4,): ZZ(1)}, ZZ)
    [1, 0, 5, 0, 7]
    >>> dup_from_dict({}, ZZ)
    []

    """
    if not f:
        return []

    n, h = max(f.keys()), []

    if isinstance(n, int):
        for k in range(n, -1, -1):
            h.append(f.get(k, K.zero))
    else:
        (n,) = n

        for k in range(n, -1, -1):
            h.append(f.get((k,), K.zero))

    return dup_strip(h)


def dup_from_raw_dict(f, K):
    """
    Create a ``K[x]`` polynomial from a raw ``dict``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_from_raw_dict

    >>> dup_from_raw_dict({0: ZZ(7), 2: ZZ(5), 4: ZZ(1)}, ZZ)
    [1, 0, 5, 0, 7]

    """
    if not f:
        return []

    n, h = max(f.keys()), []

    for k in range(n, -1, -1):
        h.append(f.get(k, K.zero))

    return dup_strip(h)


def dmp_from_dict(f, u, K):
    """
    Create a ``K[X]`` polynomial from a ``dict``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_from_dict

    >>> dmp_from_dict({(0, 0): ZZ(3), (0, 1): ZZ(2), (2, 1): ZZ(1)}, 1, ZZ)
    [[1, 0], [], [2, 3]]
    >>> dmp_from_dict({}, 0, ZZ)
    []

    """
    if not u:
        return dup_from_dict(f, K)
    if not f:
        return dmp_zero(u)

    coeffs = {}

    for monom, coeff in f.items():
        head, tail = monom[0], monom[1:]

        if head in coeffs:
            coeffs[head][tail] = coeff
        else:
            coeffs[head] = { tail: coeff }

    n, v, h = max(coeffs.keys()), u - 1, []

    for k in range(n, -1, -1):
        coeff = coeffs.get(k)

        if coeff is not None:
            h.append(dmp_from_dict(coeff, v, K))
        else:
            h.append(dmp_zero(v))

    return dmp_strip(h, u)


def dup_to_dict(f, K=None, zero=False):
    """
    Convert ``K[x]`` polynomial to a ``dict``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dup_to_dict

    >>> dup_to_dict([1, 0, 5, 0, 7])
    {(0,): 7, (2,): 5, (4,): 1}
    >>> dup_to_dict([])
    {}

    """
    if not f and zero:
        return {(0,): K.zero}

    n, result = len(f) - 1, {}

    for k in range(0, n + 1):
        if f[n - k]:
            result[(k,)] = f[n - k]

    return result


def dup_to_raw_dict(f, K=None, zero=False):
    """
    Convert a ``K[x]`` polynomial to a raw ``dict``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dup_to_raw_dict

    >>> dup_to_raw_dict([1, 0, 5, 0, 7])
    {0: 7, 2: 5, 4: 1}

    """
    if not f and zero:
        return {0: K.zero}

    n, result = len(f) - 1, {}

    for k in range(0, n + 1):
        if f[n - k]:
            result[k] = f[n - k]

    return result


def dmp_to_dict(f, u, K=None, zero=False):
    """
    Convert a ``K[X]`` polynomial to a ``dict````.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_to_dict

    >>> dmp_to_dict([[1, 0], [], [2, 3]], 1)
    {(0, 0): 3, (0, 1): 2, (2, 1): 1}
    >>> dmp_to_dict([], 0)
    {}

    """
    if not u:
        return dup_to_dict(f, K, zero=zero)

    if dmp_zero_p(f, u) and zero:
        return {(0,)*(u + 1): K.zero}

    n, v, result = dmp_degree(f, u), u - 1, {}

    if n == ninf:
        n = -1

    for k in range(0, n + 1):
        h = dmp_to_dict(f[n - k], v)

        for exp, coeff in h.items():
            result[(k,) + exp] = coeff

    return result


def dmp_swap(f, i, j, u, K):
    """
    Transform ``K[..x_i..x_j..]`` to ``K[..x_j..x_i..]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_swap

    >>> f = ZZ.map([[[2], [1, 0]], []])

    >>> dmp_swap(f, 0, 1, 2, ZZ)
    [[[2], []], [[1, 0], []]]
    >>> dmp_swap(f, 1, 2, 2, ZZ)
    [[[1], [2, 0]], [[]]]
    >>> dmp_swap(f, 0, 2, 2, ZZ)
    [[[1, 0]], [[2, 0], []]]

    """
    if i < 0 or j < 0 or i > u or j > u:
        raise IndexError("0 <= i < j <= %s expected" % u)
    elif i == j:
        return f

    F, H = dmp_to_dict(f, u), {}

    for exp, coeff in F.items():
        H[exp[:i] + (exp[j],) +
          exp[i + 1:j] +
          (exp[i],) + exp[j + 1:]] = coeff

    return dmp_from_dict(H, u, K)


def dmp_permute(f, P, u, K):
    """
    Return a polynomial in ``K[x_{P(1)},..,x_{P(n)}]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_permute

    >>> f = ZZ.map([[[2], [1, 0]], []])

    >>> dmp_permute(f, [1, 0, 2], 2, ZZ)
    [[[2], []], [[1, 0], []]]
    >>> dmp_permute(f, [1, 2, 0], 2, ZZ)
    [[[1], []], [[2, 0], []]]

    """
    F, H = dmp_to_dict(f, u), {}

    for exp, coeff in F.items():
        new_exp = [0]*len(exp)

        for e, p in zip(exp, P):
            new_exp[p] = e

        H[tuple(new_exp)] = coeff

    return dmp_from_dict(H, u, K)


def dmp_nest(f, l, K):
    """
    Return a multivariate value nested ``l``-levels.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_nest

    >>> dmp_nest([[ZZ(1)]], 2, ZZ)
    [[[[1]]]]

    """
    if not isinstance(f, list):
        return dmp_ground(f, l)

    for i in range(l):
        f = [f]

    return f


def dmp_raise(f, l, u, K):
    """
    Return a multivariate polynomial raised ``l``-levels.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_raise

    >>> f = ZZ.map([[], [1, 2]])

    >>> dmp_raise(f, 2, 1, ZZ)
    [[[[]]], [[[1]], [[2]]]]

    """
    if not l:
        return f

    if not u:
        if not f:
            return dmp_zero(l)

        k = l - 1

        return [ dmp_ground(c, k) for c in f ]

    v = u - 1

    return [ dmp_raise(c, l, v, K) for c in f ]


def dup_deflate(f, K):
    """
    Map ``x**m`` to ``y`` in a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_deflate

    >>> f = ZZ.map([1, 0, 0, 1, 0, 0, 1])

    >>> dup_deflate(f, ZZ)
    (3, [1, 1, 1])

    """
    if dup_degree(f) <= 0:
        return 1, f

    g = 0

    for i in range(len(f)):
        if not f[-i - 1]:
            continue

        g = igcd(g, i)

        if g == 1:
            return 1, f

    return g, f[::g]


def dmp_deflate(f, u, K):
    """
    Map ``x_i**m_i`` to ``y_i`` in a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_deflate

    >>> f = ZZ.map([[1, 0, 0, 2], [], [3, 0, 0, 4]])

    >>> dmp_deflate(f, 1, ZZ)
    ((2, 3), [[1, 2], [3, 4]])

    """
    if dmp_zero_p(f, u):
        return (1,)*(u + 1), f

    F = dmp_to_dict(f, u)
    B = [0]*(u + 1)

    for M in F.keys():
        for i, m in enumerate(M):
            B[i] = igcd(B[i], m)

    for i, b in enumerate(B):
        if not b:
            B[i] = 1

    B = tuple(B)

    if all(b == 1 for b in B):
        return B, f

    H = {}

    for A, coeff in F.items():
        N = [ a // b for a, b in zip(A, B) ]
        H[tuple(N)] = coeff

    return B, dmp_from_dict(H, u, K)


def dup_multi_deflate(polys, K):
    """
    Map ``x**m`` to ``y`` in a set of polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_multi_deflate

    >>> f = ZZ.map([1, 0, 2, 0, 3])
    >>> g = ZZ.map([4, 0, 0])

    >>> dup_multi_deflate((f, g), ZZ)
    (2, ([1, 2, 3], [4, 0]))

    """
    G = 0

    for p in polys:
        if dup_degree(p) <= 0:
            return 1, polys

        g = 0

        for i in range(len(p)):
            if not p[-i - 1]:
                continue

            g = igcd(g, i)

            if g == 1:
                return 1, polys

        G = igcd(G, g)

    return G, tuple([ p[::G] for p in polys ])


def dmp_multi_deflate(polys, u, K):
    """
    Map ``x_i**m_i`` to ``y_i`` in a set of polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_multi_deflate

    >>> f = ZZ.map([[1, 0, 0, 2], [], [3, 0, 0, 4]])
    >>> g = ZZ.map([[1, 0, 2], [], [3, 0, 4]])

    >>> dmp_multi_deflate((f, g), 1, ZZ)
    ((2, 1), ([[1, 0, 0, 2], [3, 0, 0, 4]], [[1, 0, 2], [3, 0, 4]]))

    """
    if not u:
        M, H = dup_multi_deflate(polys, K)
        return (M,), H

    F, B = [], [0]*(u + 1)

    for p in polys:
        f = dmp_to_dict(p, u)

        if not dmp_zero_p(p, u):
            for M in f.keys():
                for i, m in enumerate(M):
                    B[i] = igcd(B[i], m)

        F.append(f)

    for i, b in enumerate(B):
        if not b:
            B[i] = 1

    B = tuple(B)

    if all(b == 1 for b in B):
        return B, polys

    H = []

    for f in F:
        h = {}

        for A, coeff in f.items():
            N = [ a // b for a, b in zip(A, B) ]
            h[tuple(N)] = coeff

        H.append(dmp_from_dict(h, u, K))

    return B, tuple(H)


def dup_inflate(f, m, K):
    """
    Map ``y`` to ``x**m`` in a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_inflate

    >>> f = ZZ.map([1, 1, 1])

    >>> dup_inflate(f, 3, ZZ)
    [1, 0, 0, 1, 0, 0, 1]

    """
    if m <= 0:
        raise IndexError("'m' must be positive, got %s" % m)
    if m == 1 or not f:
        return f

    result = [f[0]]

    for coeff in f[1:]:
        result.extend([K.zero]*(m - 1))
        result.append(coeff)

    return result


def _rec_inflate(g, M, v, i, K):
    """Recursive helper for :func:`dmp_inflate`."""
    if not v:
        return dup_inflate(g, M[i], K)
    if M[i] <= 0:
        raise IndexError("all M[i] must be positive, got %s" % M[i])

    w, j = v - 1, i + 1

    g = [ _rec_inflate(c, M, w, j, K) for c in g ]

    result = [g[0]]

    for coeff in g[1:]:
        for _ in range(1, M[i]):
            result.append(dmp_zero(w))

        result.append(coeff)

    return result


def dmp_inflate(f, M, u, K):
    """
    Map ``y_i`` to ``x_i**k_i`` in a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_inflate

    >>> f = ZZ.map([[1, 2], [3, 4]])

    >>> dmp_inflate(f, (2, 3), 1, ZZ)
    [[1, 0, 0, 2], [], [3, 0, 0, 4]]

    """
    if not u:
        return dup_inflate(f, M[0], K)

    if all(m == 1 for m in M):
        return f
    else:
        return _rec_inflate(f, M, u, 0, K)


def dmp_exclude(f, u, K):
    """
    Exclude useless levels from ``f``.

    Return the levels excluded, the new excluded ``f``, and the new ``u``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_exclude

    >>> f = ZZ.map([[[1]], [[1], [2]]])

    >>> dmp_exclude(f, 2, ZZ)
    ([2], [[1], [1, 2]], 1)

    """
    if not u or dmp_ground_p(f, None, u):
        return [], f, u

    J, F = [], dmp_to_dict(f, u)

    for j in range(0, u + 1):
        for monom in F.keys():
            if monom[j]:
                break
        else:
            J.append(j)

    if not J:
        return [], f, u

    f = {}

    for monom, coeff in F.items():
        monom = list(monom)

        for j in reversed(J):
            del monom[j]

        f[tuple(monom)] = coeff

    u -= len(J)

    return J, dmp_from_dict(f, u, K), u


def dmp_include(f, J, u, K):
    """
    Include useless levels in ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_include

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_include(f, [2], 1, ZZ)
    [[[1]], [[1], [2]]]

    """
    if not J:
        return f

    F, f = dmp_to_dict(f, u), {}

    for monom, coeff in F.items():
        monom = list(monom)

        for j in J:
            monom.insert(j, 0)

        f[tuple(monom)] = coeff

    u += len(J)

    return dmp_from_dict(f, u, K)


def dmp_inject(f, u, K, front=False):
    """
    Convert ``f`` from ``K[X][Y]`` to ``K[X,Y]``.

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_inject

    >>> R, x,y = ring("x,y", ZZ)

    >>> dmp_inject([R(1), x + 2], 0, R.to_domain())
    ([[[1]], [[1], [2]]], 2)
    >>> dmp_inject([R(1), x + 2], 0, R.to_domain(), front=True)
    ([[[1]], [[1, 2]]], 2)

    """
    f, h = dmp_to_dict(f, u), {}

    v = K.ngens - 1

    for f_monom, g in f.items():
        g = g.to_dict()

        for g_monom, c in g.items():
            if front:
                h[g_monom + f_monom] = c
            else:
                h[f_monom + g_monom] = c

    w = u + v + 1

    return dmp_from_dict(h, w, K.dom), w


def dmp_eject(f, u, K, front=False):
    """
    Convert ``f`` from ``K[X,Y]`` to ``K[X][Y]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_eject

    >>> dmp_eject([[[1]], [[1], [2]]], 2, ZZ['x', 'y'])
    [1, x + 2]

    """
    f, h = dmp_to_dict(f, u), {}

    n = K.ngens
    v = u - K.ngens + 1

    for monom, c in f.items():
        if front:
            g_monom, f_monom = monom[:n], monom[n:]
        else:
            g_monom, f_monom = monom[-n:], monom[:-n]

        if f_monom in h:
            h[f_monom][g_monom] = c
        else:
            h[f_monom] = {g_monom: c}

    for monom, c in h.items():
        h[monom] = K(c)

    return dmp_from_dict(h, v - 1, K)


def dup_terms_gcd(f, K):
    """
    Remove GCD of terms from ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_terms_gcd

    >>> f = ZZ.map([1, 0, 1, 0, 0])

    >>> dup_terms_gcd(f, ZZ)
    (2, [1, 0, 1])

    """
    if dup_TC(f, K) or not f:
        return 0, f

    i = 0

    for c in reversed(f):
        if not c:
            i += 1
        else:
            break

    return i, f[:-i]


def dmp_terms_gcd(f, u, K):
    """
    Remove GCD of terms from ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_terms_gcd

    >>> f = ZZ.map([[1, 0], [1, 0, 0], [], []])

    >>> dmp_terms_gcd(f, 1, ZZ)
    ((2, 1), [[1], [1, 0]])

    """
    if dmp_ground_TC(f, u, K) or dmp_zero_p(f, u):
        return (0,)*(u + 1), f

    F = dmp_to_dict(f, u)
    G = monomial_min(*list(F.keys()))

    if all(g == 0 for g in G):
        return G, f

    f = {}

    for monom, coeff in F.items():
        f[monomial_div(monom, G)] = coeff

    return G, dmp_from_dict(f, u, K)


def _rec_list_terms(g, v, monom):
    """Recursive helper for :func:`dmp_list_terms`."""
    d, terms = dmp_degree(g, v), []

    if not v:
        for i, c in enumerate(g):
            if not c:
                continue

            terms.append((monom + (d - i,), c))
    else:
        w = v - 1

        for i, c in enumerate(g):
            terms.extend(_rec_list_terms(c, w, monom + (d - i,)))

    return terms


def dmp_list_terms(f, u, K, order=None):
    """
    List all non-zero terms from ``f`` in the given order ``order``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_list_terms

    >>> f = ZZ.map([[1, 1], [2, 3]])

    >>> dmp_list_terms(f, 1, ZZ)
    [((1, 1), 1), ((1, 0), 1), ((0, 1), 2), ((0, 0), 3)]
    >>> dmp_list_terms(f, 1, ZZ, order='grevlex')
    [((1, 1), 1), ((1, 0), 1), ((0, 1), 2), ((0, 0), 3)]

    """
    def sort(terms, O):
        return sorted(terms, key=lambda term: O(term[0]), reverse=True)

    terms = _rec_list_terms(f, u, ())

    if not terms:
        return [((0,)*(u + 1), K.zero)]

    if order is None:
        return terms
    else:
        return sort(terms, monomial_key(order))


def dup_apply_pairs(f, g, h, args, K):
    """
    Apply ``h`` to pairs of coefficients of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_apply_pairs

    >>> h = lambda x, y, z: 2*x + y - z

    >>> dup_apply_pairs([1, 2, 3], [3, 2, 1], h, (1,), ZZ)
    [4, 5, 6]

    """
    n, m = len(f), len(g)

    if n != m:
        if n > m:
            g = [K.zero]*(n - m) + g
        else:
            f = [K.zero]*(m - n) + f

    result = []

    for a, b in zip(f, g):
        result.append(h(a, b, *args))

    return dup_strip(result)


def dmp_apply_pairs(f, g, h, args, u, K):
    """
    Apply ``h`` to pairs of coefficients of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_apply_pairs

    >>> h = lambda x, y, z: 2*x + y - z

    >>> dmp_apply_pairs([[1], [2, 3]], [[3], [2, 1]], h, (1,), 1, ZZ)
    [[4], [5, 6]]

    """
    if not u:
        return dup_apply_pairs(f, g, h, args, K)

    n, m, v = len(f), len(g), u - 1

    if n != m:
        if n > m:
            g = dmp_zeros(n - m, v, K) + g
        else:
            f = dmp_zeros(m - n, v, K) + f

    result = []

    for a, b in zip(f, g):
        result.append(dmp_apply_pairs(a, b, h, args, v, K))

    return dmp_strip(result, u)


def dup_slice(f, m, n, K):
    """Take a continuous subsequence of terms of ``f`` in ``K[x]``. """
    k = len(f)

    if k >= m:
        M = k - m
    else:
        M = 0
    if k >= n:
        N = k - n
    else:
        N = 0

    f = f[N:M]

    while f and f[0] == K.zero:
        f.pop(0)

    if not f:
        return []
    else:
        return f + [K.zero]*m


def dmp_slice(f, m, n, u, K):
    """Take a continuous subsequence of terms of ``f`` in ``K[X]``. """
    return dmp_slice_in(f, m, n, 0, u, K)


def dmp_slice_in(f, m, n, j, u, K):
    """Take a continuous subsequence of terms of ``f`` in ``x_j`` in ``K[X]``. """
    if j < 0 or j > u:
        raise IndexError("-%s <= j < %s expected, got %s" % (u, u, j))

    if not u:
        return dup_slice(f, m, n, K)

    f, g = dmp_to_dict(f, u), {}

    for monom, coeff in f.items():
        k = monom[j]

        if k < m or k >= n:
            monom = monom[:j] + (0,) + monom[j + 1:]

        if monom in g:
            g[monom] += coeff
        else:
            g[monom] = coeff

    return dmp_from_dict(g, u, K)


def dup_random(n, a, b, K):
    """
    Return a polynomial of degree ``n`` with coefficients in ``[a, b]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_random

    >>> dup_random(3, -10, 10, ZZ) #doctest: +SKIP
    [-2, -8, 9, -4]

    """
    f = [ K.convert(random.randint(a, b)) for _ in range(0, n + 1) ]

    while not f[0]:
        f[0] = K.convert(random.randint(a, b))

    return f

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\test_ddl.py ===
# testing/suite/test_ddl.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

import random

from . import testing
from .. import config
from .. import fixtures
from .. import util
from ..assertions import eq_
from ..assertions import is_false
from ..assertions import is_true
from ..config import requirements
from ..schema import Table
from ... import CheckConstraint
from ... import Column
from ... import ForeignKeyConstraint
from ... import Index
from ... import inspect
from ... import Integer
from ... import schema
from ... import String
from ... import UniqueConstraint


class TableDDLTest(fixtures.TestBase):
    __backend__ = True

    def _simple_fixture(self, schema=None):
        return Table(
            "test_table",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("data", String(50)),
            schema=schema,
        )

    def _underscore_fixture(self):
        return Table(
            "_test_table",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("_data", String(50)),
        )

    def _table_index_fixture(self, schema=None):
        table = self._simple_fixture(schema=schema)
        idx = Index("test_index", table.c.data)
        return table, idx

    def _simple_roundtrip(self, table):
        with config.db.begin() as conn:
            conn.execute(table.insert().values((1, "some data")))
            result = conn.execute(table.select())
            eq_(result.first(), (1, "some data"))

    @requirements.create_table
    @util.provide_metadata
    def test_create_table(self):
        table = self._simple_fixture()
        table.create(config.db, checkfirst=False)
        self._simple_roundtrip(table)

    @requirements.create_table
    @requirements.schemas
    @util.provide_metadata
    def test_create_table_schema(self):
        table = self._simple_fixture(schema=config.test_schema)
        table.create(config.db, checkfirst=False)
        self._simple_roundtrip(table)

    @requirements.drop_table
    @util.provide_metadata
    def test_drop_table(self):
        table = self._simple_fixture()
        table.create(config.db, checkfirst=False)
        table.drop(config.db, checkfirst=False)

    @requirements.create_table
    @util.provide_metadata
    def test_underscore_names(self):
        table = self._underscore_fixture()
        table.create(config.db, checkfirst=False)
        self._simple_roundtrip(table)

    @requirements.comment_reflection
    @util.provide_metadata
    def test_add_table_comment(self, connection):
        table = self._simple_fixture()
        table.create(connection, checkfirst=False)
        table.comment = "a comment"
        connection.execute(schema.SetTableComment(table))
        eq_(
            inspect(connection).get_table_comment("test_table"),
            {"text": "a comment"},
        )

    @requirements.comment_reflection
    @util.provide_metadata
    def test_drop_table_comment(self, connection):
        table = self._simple_fixture()
        table.create(connection, checkfirst=False)
        table.comment = "a comment"
        connection.execute(schema.SetTableComment(table))
        connection.execute(schema.DropTableComment(table))
        eq_(
            inspect(connection).get_table_comment("test_table"), {"text": None}
        )

    @requirements.table_ddl_if_exists
    @util.provide_metadata
    def test_create_table_if_not_exists(self, connection):
        table = self._simple_fixture()

        connection.execute(schema.CreateTable(table, if_not_exists=True))

        is_true(inspect(connection).has_table("test_table"))
        connection.execute(schema.CreateTable(table, if_not_exists=True))

    @requirements.index_ddl_if_exists
    @util.provide_metadata
    def test_create_index_if_not_exists(self, connection):
        table, idx = self._table_index_fixture()

        connection.execute(schema.CreateTable(table, if_not_exists=True))
        is_true(inspect(connection).has_table("test_table"))
        is_false(
            "test_index"
            in [
                ix["name"]
                for ix in inspect(connection).get_indexes("test_table")
            ]
        )

        connection.execute(schema.CreateIndex(idx, if_not_exists=True))

        is_true(
            "test_index"
            in [
                ix["name"]
                for ix in inspect(connection).get_indexes("test_table")
            ]
        )

        connection.execute(schema.CreateIndex(idx, if_not_exists=True))

    @requirements.table_ddl_if_exists
    @util.provide_metadata
    def test_drop_table_if_exists(self, connection):
        table = self._simple_fixture()

        table.create(connection)

        is_true(inspect(connection).has_table("test_table"))

        connection.execute(schema.DropTable(table, if_exists=True))

        is_false(inspect(connection).has_table("test_table"))

        connection.execute(schema.DropTable(table, if_exists=True))

    @requirements.index_ddl_if_exists
    @util.provide_metadata
    def test_drop_index_if_exists(self, connection):
        table, idx = self._table_index_fixture()

        table.create(connection)

        is_true(
            "test_index"
            in [
                ix["name"]
                for ix in inspect(connection).get_indexes("test_table")
            ]
        )

        connection.execute(schema.DropIndex(idx, if_exists=True))

        is_false(
            "test_index"
            in [
                ix["name"]
                for ix in inspect(connection).get_indexes("test_table")
            ]
        )

        connection.execute(schema.DropIndex(idx, if_exists=True))


class FutureTableDDLTest(fixtures.FutureEngineMixin, TableDDLTest):
    pass


class LongNameBlowoutTest(fixtures.TestBase):
    """test the creation of a variety of DDL structures and ensure
    label length limits pass on backends

    """

    __backend__ = True

    def fk(self, metadata, connection):
        convention = {
            "fk": "foreign_key_%(table_name)s_"
            "%(column_0_N_name)s_"
            "%(referred_table_name)s_"
            + (
                "_".join(
                    "".join(random.choice("abcdef") for j in range(20))
                    for i in range(10)
                )
            ),
        }
        metadata.naming_convention = convention

        Table(
            "a_things_with_stuff",
            metadata,
            Column("id_long_column_name", Integer, primary_key=True),
            test_needs_fk=True,
        )

        cons = ForeignKeyConstraint(
            ["aid"], ["a_things_with_stuff.id_long_column_name"]
        )
        Table(
            "b_related_things_of_value",
            metadata,
            Column(
                "aid",
            ),
            cons,
            test_needs_fk=True,
        )
        actual_name = cons.name

        metadata.create_all(connection)

        if testing.requires.foreign_key_constraint_name_reflection.enabled:
            insp = inspect(connection)
            fks = insp.get_foreign_keys("b_related_things_of_value")
            reflected_name = fks[0]["name"]

            return actual_name, reflected_name
        else:
            return actual_name, None

    def pk(self, metadata, connection):
        convention = {
            "pk": "primary_key_%(table_name)s_"
            "%(column_0_N_name)s"
            + (
                "_".join(
                    "".join(random.choice("abcdef") for j in range(30))
                    for i in range(10)
                )
            ),
        }
        metadata.naming_convention = convention

        a = Table(
            "a_things_with_stuff",
            metadata,
            Column("id_long_column_name", Integer, primary_key=True),
            Column("id_another_long_name", Integer, primary_key=True),
        )
        cons = a.primary_key
        actual_name = cons.name

        metadata.create_all(connection)
        insp = inspect(connection)
        pk = insp.get_pk_constraint("a_things_with_stuff")
        reflected_name = pk["name"]
        return actual_name, reflected_name

    def ix(self, metadata, connection):
        convention = {
            "ix": "index_%(table_name)s_"
            "%(column_0_N_name)s"
            + (
                "_".join(
                    "".join(random.choice("abcdef") for j in range(30))
                    for i in range(10)
                )
            ),
        }
        metadata.naming_convention = convention

        a = Table(
            "a_things_with_stuff",
            metadata,
            Column("id_long_column_name", Integer, primary_key=True),
            Column("id_another_long_name", Integer),
        )
        cons = Index(None, a.c.id_long_column_name, a.c.id_another_long_name)
        actual_name = cons.name

        metadata.create_all(connection)
        insp = inspect(connection)
        ix = insp.get_indexes("a_things_with_stuff")
        reflected_name = ix[0]["name"]
        return actual_name, reflected_name

    def uq(self, metadata, connection):
        convention = {
            "uq": "unique_constraint_%(table_name)s_"
            "%(column_0_N_name)s"
            + (
                "_".join(
                    "".join(random.choice("abcdef") for j in range(30))
                    for i in range(10)
                )
            ),
        }
        metadata.naming_convention = convention

        cons = UniqueConstraint("id_long_column_name", "id_another_long_name")
        Table(
            "a_things_with_stuff",
            metadata,
            Column("id_long_column_name", Integer, primary_key=True),
            Column("id_another_long_name", Integer),
            cons,
        )
        actual_name = cons.name

        metadata.create_all(connection)
        insp = inspect(connection)
        uq = insp.get_unique_constraints("a_things_with_stuff")
        reflected_name = uq[0]["name"]
        return actual_name, reflected_name

    def ck(self, metadata, connection):
        convention = {
            "ck": "check_constraint_%(table_name)s"
            + (
                "_".join(
                    "".join(random.choice("abcdef") for j in range(30))
                    for i in range(10)
                )
            ),
        }
        metadata.naming_convention = convention

        cons = CheckConstraint("some_long_column_name > 5")
        Table(
            "a_things_with_stuff",
            metadata,
            Column("id_long_column_name", Integer, primary_key=True),
            Column("some_long_column_name", Integer),
            cons,
        )
        actual_name = cons.name

        metadata.create_all(connection)
        insp = inspect(connection)
        ck = insp.get_check_constraints("a_things_with_stuff")
        reflected_name = ck[0]["name"]
        return actual_name, reflected_name

    @testing.combinations(
        ("fk",),
        ("pk",),
        ("ix",),
        ("ck", testing.requires.check_constraint_reflection.as_skips()),
        ("uq", testing.requires.unique_constraint_reflection.as_skips()),
        argnames="type_",
    )
    def test_long_convention_name(self, type_, metadata, connection):
        actual_name, reflected_name = getattr(self, type_)(
            metadata, connection
        )

        assert len(actual_name) > 255

        if reflected_name is not None:
            overlap = actual_name[0 : len(reflected_name)]
            if len(overlap) < len(actual_name):
                eq_(overlap[0:-5], reflected_name[0 : len(overlap) - 5])
            else:
                eq_(overlap, reflected_name)


__all__ = ("TableDDLTest", "FutureTableDDLTest", "LongNameBlowoutTest")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\densearith.py ===
"""Arithmetics for dense recursive polynomials in ``K[x]`` or ``K[X]``. """


from sympy.polys.densebasic import (
    dup_slice,
    dup_LC, dmp_LC,
    dup_degree, dmp_degree,
    dup_strip, dmp_strip,
    dmp_zero_p, dmp_zero,
    dmp_one_p, dmp_one,
    dmp_ground, dmp_zeros)
from sympy.polys.polyerrors import (ExactQuotientFailed, PolynomialDivisionFailed)

def dup_add_term(f, c, i, K):
    """
    Add ``c*x**i`` to ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_add_term(x**2 - 1, ZZ(2), 4)
    2*x**4 + x**2 - 1

    """
    if not c:
        return f

    n = len(f)
    m = n - i - 1

    if i == n - 1:
        return dup_strip([f[0] + c] + f[1:])
    else:
        if i >= n:
            return [c] + [K.zero]*(i - n) + f
        else:
            return f[:m] + [f[m] + c] + f[m + 1:]


def dmp_add_term(f, c, i, u, K):
    """
    Add ``c(x_2..x_u)*x_0**i`` to ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_add_term(x*y + 1, 2, 2)
    2*x**2 + x*y + 1

    """
    if not u:
        return dup_add_term(f, c, i, K)

    v = u - 1

    if dmp_zero_p(c, v):
        return f

    n = len(f)
    m = n - i - 1

    if i == n - 1:
        return dmp_strip([dmp_add(f[0], c, v, K)] + f[1:], u)
    else:
        if i >= n:
            return [c] + dmp_zeros(i - n, v, K) + f
        else:
            return f[:m] + [dmp_add(f[m], c, v, K)] + f[m + 1:]


def dup_sub_term(f, c, i, K):
    """
    Subtract ``c*x**i`` from ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sub_term(2*x**4 + x**2 - 1, ZZ(2), 4)
    x**2 - 1

    """
    if not c:
        return f

    n = len(f)
    m = n - i - 1

    if i == n - 1:
        return dup_strip([f[0] - c] + f[1:])
    else:
        if i >= n:
            return [-c] + [K.zero]*(i - n) + f
        else:
            return f[:m] + [f[m] - c] + f[m + 1:]


def dmp_sub_term(f, c, i, u, K):
    """
    Subtract ``c(x_2..x_u)*x_0**i`` from ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sub_term(2*x**2 + x*y + 1, 2, 2)
    x*y + 1

    """
    if not u:
        return dup_add_term(f, -c, i, K)

    v = u - 1

    if dmp_zero_p(c, v):
        return f

    n = len(f)
    m = n - i - 1

    if i == n - 1:
        return dmp_strip([dmp_sub(f[0], c, v, K)] + f[1:], u)
    else:
        if i >= n:
            return [dmp_neg(c, v, K)] + dmp_zeros(i - n, v, K) + f
        else:
            return f[:m] + [dmp_sub(f[m], c, v, K)] + f[m + 1:]


def dup_mul_term(f, c, i, K):
    """
    Multiply ``f`` by ``c*x**i`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_mul_term(x**2 - 1, ZZ(3), 2)
    3*x**4 - 3*x**2

    """
    if not c or not f:
        return []
    else:
        return [ cf * c for cf in f ] + [K.zero]*i


def dmp_mul_term(f, c, i, u, K):
    """
    Multiply ``f`` by ``c(x_2..x_u)*x_0**i`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_mul_term(x**2*y + x, 3*y, 2)
    3*x**4*y**2 + 3*x**3*y

    """
    if not u:
        return dup_mul_term(f, c, i, K)

    v = u - 1

    if dmp_zero_p(f, u):
        return f
    if dmp_zero_p(c, v):
        return dmp_zero(u)
    else:
        return [ dmp_mul(cf, c, v, K) for cf in f ] + dmp_zeros(i, v, K)


def dup_add_ground(f, c, K):
    """
    Add an element of the ground domain to ``f``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_add_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))
    x**3 + 2*x**2 + 3*x + 8

    """
    return dup_add_term(f, c, 0, K)


def dmp_add_ground(f, c, u, K):
    """
    Add an element of the ground domain to ``f``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_add_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))
    x**3 + 2*x**2 + 3*x + 8

    """
    return dmp_add_term(f, dmp_ground(c, u - 1), 0, u, K)


def dup_sub_ground(f, c, K):
    """
    Subtract an element of the ground domain from ``f``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sub_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))
    x**3 + 2*x**2 + 3*x

    """
    return dup_sub_term(f, c, 0, K)


def dmp_sub_ground(f, c, u, K):
    """
    Subtract an element of the ground domain from ``f``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sub_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))
    x**3 + 2*x**2 + 3*x

    """
    return dmp_sub_term(f, dmp_ground(c, u - 1), 0, u, K)


def dup_mul_ground(f, c, K):
    """
    Multiply ``f`` by a constant value in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_mul_ground(x**2 + 2*x - 1, ZZ(3))
    3*x**2 + 6*x - 3

    """
    if not c or not f:
        return []
    else:
        return [ cf * c for cf in f ]


def dmp_mul_ground(f, c, u, K):
    """
    Multiply ``f`` by a constant value in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_mul_ground(2*x + 2*y, ZZ(3))
    6*x + 6*y

    """
    if not u:
        return dup_mul_ground(f, c, K)

    v = u - 1

    return [ dmp_mul_ground(cf, c, v, K) for cf in f ]


def dup_quo_ground(f, c, K):
    """
    Quotient by a constant in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_quo_ground(3*x**2 + 2, ZZ(2))
    x**2 + 1

    >>> R, x = ring("x", QQ)
    >>> R.dup_quo_ground(3*x**2 + 2, QQ(2))
    3/2*x**2 + 1

    """
    if not c:
        raise ZeroDivisionError('polynomial division')
    if not f:
        return f

    if K.is_Field:
        return [ K.quo(cf, c) for cf in f ]
    else:
        return [ cf // c for cf in f ]


def dmp_quo_ground(f, c, u, K):
    """
    Quotient by a constant in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> R.dmp_quo_ground(2*x**2*y + 3*x, ZZ(2))
    x**2*y + x

    >>> R, x,y = ring("x,y", QQ)
    >>> R.dmp_quo_ground(2*x**2*y + 3*x, QQ(2))
    x**2*y + 3/2*x

    """
    if not u:
        return dup_quo_ground(f, c, K)

    v = u - 1

    return [ dmp_quo_ground(cf, c, v, K) for cf in f ]


def dup_exquo_ground(f, c, K):
    """
    Exact quotient by a constant in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_exquo_ground(x**2 + 2, QQ(2))
    1/2*x**2 + 1

    """
    if not c:
        raise ZeroDivisionError('polynomial division')
    if not f:
        return f

    return [ K.exquo(cf, c) for cf in f ]


def dmp_exquo_ground(f, c, u, K):
    """
    Exact quotient by a constant in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_exquo_ground(x**2*y + 2*x, QQ(2))
    1/2*x**2*y + x

    """
    if not u:
        return dup_exquo_ground(f, c, K)

    v = u - 1

    return [ dmp_exquo_ground(cf, c, v, K) for cf in f ]


def dup_lshift(f, n, K):
    """
    Efficiently multiply ``f`` by ``x**n`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_lshift(x**2 + 1, 2)
    x**4 + x**2

    """
    if not f:
        return f
    else:
        return f + [K.zero]*n


def dup_rshift(f, n, K):
    """
    Efficiently divide ``f`` by ``x**n`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_rshift(x**4 + x**2, 2)
    x**2 + 1
    >>> R.dup_rshift(x**4 + x**2 + 2, 2)
    x**2 + 1

    """
    return f[:-n]


def dup_abs(f, K):
    """
    Make all coefficients positive in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_abs(x**2 - 1)
    x**2 + 1

    """
    return [ K.abs(coeff) for coeff in f ]


def dmp_abs(f, u, K):
    """
    Make all coefficients positive in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_abs(x**2*y - x)
    x**2*y + x

    """
    if not u:
        return dup_abs(f, K)

    v = u - 1

    return [ dmp_abs(cf, v, K) for cf in f ]


def dup_neg(f, K):
    """
    Negate a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_neg(x**2 - 1)
    -x**2 + 1

    """
    return [ -coeff for coeff in f ]


def dmp_neg(f, u, K):
    """
    Negate a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_neg(x**2*y - x)
    -x**2*y + x

    """
    if not u:
        return dup_neg(f, K)

    v = u - 1

    return [ dmp_neg(cf, v, K) for cf in f ]


def dup_add(f, g, K):
    """
    Add dense polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_add(x**2 - 1, x - 2)
    x**2 + x - 3

    """
    if not f:
        return g
    if not g:
        return f

    df = dup_degree(f)
    dg = dup_degree(g)

    if df == dg:
        return dup_strip([ a + b for a, b in zip(f, g) ])
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = g[:k], g[k:]

        return h + [ a + b for a, b in zip(f, g) ]


def dmp_add(f, g, u, K):
    """
    Add dense polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_add(x**2 + y, x**2*y + x)
    x**2*y + x**2 + x + y

    """
    if not u:
        return dup_add(f, g, K)

    df = dmp_degree(f, u)

    if df < 0:
        return g

    dg = dmp_degree(g, u)

    if dg < 0:
        return f

    v = u - 1

    if df == dg:
        return dmp_strip([ dmp_add(a, b, v, K) for a, b in zip(f, g) ], u)
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = g[:k], g[k:]

        return h + [ dmp_add(a, b, v, K) for a, b in zip(f, g) ]


def dup_sub(f, g, K):
    """
    Subtract dense polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sub(x**2 - 1, x - 2)
    x**2 - x + 1

    """
    if not f:
        return dup_neg(g, K)
    if not g:
        return f

    df = dup_degree(f)
    dg = dup_degree(g)

    if df == dg:
        return dup_strip([ a - b for a, b in zip(f, g) ])
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = dup_neg(g[:k], K), g[k:]

        return h + [ a - b for a, b in zip(f, g) ]


def dmp_sub(f, g, u, K):
    """
    Subtract dense polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sub(x**2 + y, x**2*y + x)
    -x**2*y + x**2 - x + y

    """
    if not u:
        return dup_sub(f, g, K)

    df = dmp_degree(f, u)

    if df < 0:
        return dmp_neg(g, u, K)

    dg = dmp_degree(g, u)

    if dg < 0:
        return f

    v = u - 1

    if df == dg:
        return dmp_strip([ dmp_sub(a, b, v, K) for a, b in zip(f, g) ], u)
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = dmp_neg(g[:k], u, K), g[k:]

        return h + [ dmp_sub(a, b, v, K) for a, b in zip(f, g) ]


def dup_add_mul(f, g, h, K):
    """
    Returns ``f + g*h`` where ``f, g, h`` are in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_add_mul(x**2 - 1, x - 2, x + 2)
    2*x**2 - 5

    """
    return dup_add(f, dup_mul(g, h, K), K)


def dmp_add_mul(f, g, h, u, K):
    """
    Returns ``f + g*h`` where ``f, g, h`` are in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_add_mul(x**2 + y, x, x + 2)
    2*x**2 + 2*x + y

    """
    return dmp_add(f, dmp_mul(g, h, u, K), u, K)


def dup_sub_mul(f, g, h, K):
    """
    Returns ``f - g*h`` where ``f, g, h`` are in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sub_mul(x**2 - 1, x - 2, x + 2)
    3

    """
    return dup_sub(f, dup_mul(g, h, K), K)


def dmp_sub_mul(f, g, h, u, K):
    """
    Returns ``f - g*h`` where ``f, g, h`` are in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sub_mul(x**2 + y, x, x + 2)
    -2*x + y

    """
    return dmp_sub(f, dmp_mul(g, h, u, K), u, K)


def dup_mul(f, g, K):
    """
    Multiply dense polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_mul(x - 2, x + 2)
    x**2 - 4

    """
    if f == g:
        return dup_sqr(f, K)

    if not (f and g):
        return []

    df = dup_degree(f)
    dg = dup_degree(g)

    n = max(df, dg) + 1

    if n < 100 or not K.is_Exact:
        h = []

        for i in range(0, df + dg + 1):
            coeff = K.zero

            for j in range(max(0, i - dg), min(df, i) + 1):
                coeff += f[j]*g[i - j]

            h.append(coeff)

        return dup_strip(h)
    else:
        # Use Karatsuba's algorithm (divide and conquer), see e.g.:
        # Joris van der Hoeven, Relax But Don't Be Too Lazy,
        # J. Symbolic Computation, 11 (2002), section 3.1.1.
        n2 = n//2

        fl, gl = dup_slice(f, 0, n2, K), dup_slice(g, 0, n2, K)

        fh = dup_rshift(dup_slice(f, n2, n, K), n2, K)
        gh = dup_rshift(dup_slice(g, n2, n, K), n2, K)

        lo, hi = dup_mul(fl, gl, K), dup_mul(fh, gh, K)

        mid = dup_mul(dup_add(fl, fh, K), dup_add(gl, gh, K), K)
        mid = dup_sub(mid, dup_add(lo, hi, K), K)

        return dup_add(dup_add(lo, dup_lshift(mid, n2, K), K),
                       dup_lshift(hi, 2*n2, K), K)


def dmp_mul(f, g, u, K):
    """
    Multiply dense polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_mul(x*y + 1, x)
    x**2*y + x

    """
    if not u:
        return dup_mul(f, g, K)

    if f == g:
        return dmp_sqr(f, u, K)

    df = dmp_degree(f, u)

    if df < 0:
        return f

    dg = dmp_degree(g, u)

    if dg < 0:
        return g

    h, v = [], u - 1

    for i in range(0, df + dg + 1):
        coeff = dmp_zero(v)

        for j in range(max(0, i - dg), min(df, i) + 1):
            coeff = dmp_add(coeff, dmp_mul(f[j], g[i - j], v, K), v, K)

        h.append(coeff)

    return dmp_strip(h, u)


def dup_sqr(f, K):
    """
    Square dense polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqr(x**2 + 1)
    x**4 + 2*x**2 + 1

    """
    df, h = len(f) - 1, []

    for i in range(0, 2*df + 1):
        c = K.zero

        jmin = max(0, i - df)
        jmax = min(i, df)

        n = jmax - jmin + 1

        jmax = jmin + n // 2 - 1

        for j in range(jmin, jmax + 1):
            c += f[j]*f[i - j]

        c += c

        if n & 1:
            elem = f[jmax + 1]
            c += elem**2

        h.append(c)

    return dup_strip(h)


def dmp_sqr(f, u, K):
    """
    Square dense polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqr(x**2 + x*y + y**2)
    x**4 + 2*x**3*y + 3*x**2*y**2 + 2*x*y**3 + y**4

    """
    if not u:
        return dup_sqr(f, K)

    df = dmp_degree(f, u)

    if df < 0:
        return f

    h, v = [], u - 1

    for i in range(0, 2*df + 1):
        c = dmp_zero(v)

        jmin = max(0, i - df)
        jmax = min(i, df)

        n = jmax - jmin + 1

        jmax = jmin + n // 2 - 1

        for j in range(jmin, jmax + 1):
            c = dmp_add(c, dmp_mul(f[j], f[i - j], v, K), v, K)

        c = dmp_mul_ground(c, K(2), v, K)

        if n & 1:
            elem = dmp_sqr(f[jmax + 1], v, K)
            c = dmp_add(c, elem, v, K)

        h.append(c)

    return dmp_strip(h, u)


def dup_pow(f, n, K):
    """
    Raise ``f`` to the ``n``-th power in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_pow(x - 2, 3)
    x**3 - 6*x**2 + 12*x - 8

    """
    if not n:
        return [K.one]
    if n < 0:
        raise ValueError("Cannot raise polynomial to a negative power")
    if n == 1 or not f or f == [K.one]:
        return f

    g = [K.one]

    while True:
        n, m = n//2, n

        if m % 2:
            g = dup_mul(g, f, K)

            if not n:
                break

        f = dup_sqr(f, K)

    return g


def dmp_pow(f, n, u, K):
    """
    Raise ``f`` to the ``n``-th power in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_pow(x*y + 1, 3)
    x**3*y**3 + 3*x**2*y**2 + 3*x*y + 1

    """
    if not u:
        return dup_pow(f, n, K)

    if not n:
        return dmp_one(u, K)
    if n < 0:
        raise ValueError("Cannot raise polynomial to a negative power")
    if n == 1 or dmp_zero_p(f, u) or dmp_one_p(f, u, K):
        return f

    g = dmp_one(u, K)

    while True:
        n, m = n//2, n

        if m & 1:
            g = dmp_mul(g, f, u, K)

            if not n:
                break

        f = dmp_sqr(f, u, K)

    return g


def dup_pdiv(f, g, K):
    """
    Polynomial pseudo-division in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_pdiv(x**2 + 1, 2*x - 4)
    (2*x + 4, 20)

    """
    df = dup_degree(f)
    dg = dup_degree(g)

    q, r, dr = [], f, df

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return q, r

    N = df - dg + 1
    lc_g = dup_LC(g, K)

    while True:
        lc_r = dup_LC(r, K)
        j, N = dr - dg, N - 1

        Q = dup_mul_ground(q, lc_g, K)
        q = dup_add_term(Q, lc_r, j, K)

        R = dup_mul_ground(r, lc_g, K)
        G = dup_mul_term(g, lc_r, j, K)
        r = dup_sub(R, G, K)

        _dr, dr = dr, dup_degree(r)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    c = lc_g**N

    q = dup_mul_ground(q, c, K)
    r = dup_mul_ground(r, c, K)

    return q, r


def dup_prem(f, g, K):
    """
    Polynomial pseudo-remainder in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_prem(x**2 + 1, 2*x - 4)
    20

    """
    df = dup_degree(f)
    dg = dup_degree(g)

    r, dr = f, df

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return r

    N = df - dg + 1
    lc_g = dup_LC(g, K)

    while True:
        lc_r = dup_LC(r, K)
        j, N = dr - dg, N - 1

        R = dup_mul_ground(r, lc_g, K)
        G = dup_mul_term(g, lc_r, j, K)
        r = dup_sub(R, G, K)

        _dr, dr = dr, dup_degree(r)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    return dup_mul_ground(r, lc_g**N, K)


def dup_pquo(f, g, K):
    """
    Polynomial exact pseudo-quotient in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_pquo(x**2 - 1, 2*x - 2)
    2*x + 2

    >>> R.dup_pquo(x**2 + 1, 2*x - 4)
    2*x + 4

    """
    return dup_pdiv(f, g, K)[0]


def dup_pexquo(f, g, K):
    """
    Polynomial pseudo-quotient in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_pexquo(x**2 - 1, 2*x - 2)
    2*x + 2

    >>> R.dup_pexquo(x**2 + 1, 2*x - 4)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [2, -4] does not divide [1, 0, 1]

    """
    q, r = dup_pdiv(f, g, K)

    if not r:
        return q
    else:
        raise ExactQuotientFailed(f, g)


def dmp_pdiv(f, g, u, K):
    """
    Polynomial pseudo-division in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_pdiv(x**2 + x*y, 2*x + 2)
    (2*x + 2*y - 2, -4*y + 4)

    """
    if not u:
        return dup_pdiv(f, g, K)

    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if dg < 0:
        raise ZeroDivisionError("polynomial division")

    q, r, dr = dmp_zero(u), f, df

    if df < dg:
        return q, r

    N = df - dg + 1
    lc_g = dmp_LC(g, K)

    while True:
        lc_r = dmp_LC(r, K)
        j, N = dr - dg, N - 1

        Q = dmp_mul_term(q, lc_g, 0, u, K)
        q = dmp_add_term(Q, lc_r, j, u, K)

        R = dmp_mul_term(r, lc_g, 0, u, K)
        G = dmp_mul_term(g, lc_r, j, u, K)
        r = dmp_sub(R, G, u, K)

        _dr, dr = dr, dmp_degree(r, u)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    c = dmp_pow(lc_g, N, u - 1, K)

    q = dmp_mul_term(q, c, 0, u, K)
    r = dmp_mul_term(r, c, 0, u, K)

    return q, r


def dmp_prem(f, g, u, K):
    """
    Polynomial pseudo-remainder in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_prem(x**2 + x*y, 2*x + 2)
    -4*y + 4

    """
    if not u:
        return dup_prem(f, g, K)

    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if dg < 0:
        raise ZeroDivisionError("polynomial division")

    r, dr = f, df

    if df < dg:
        return r

    N = df - dg + 1
    lc_g = dmp_LC(g, K)

    while True:
        lc_r = dmp_LC(r, K)
        j, N = dr - dg, N - 1

        R = dmp_mul_term(r, lc_g, 0, u, K)
        G = dmp_mul_term(g, lc_r, j, u, K)
        r = dmp_sub(R, G, u, K)

        _dr, dr = dr, dmp_degree(r, u)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    c = dmp_pow(lc_g, N, u - 1, K)

    return dmp_mul_term(r, c, 0, u, K)


def dmp_pquo(f, g, u, K):
    """
    Polynomial exact pseudo-quotient in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**2 + x*y
    >>> g = 2*x + 2*y
    >>> h = 2*x + 2

    >>> R.dmp_pquo(f, g)
    2*x

    >>> R.dmp_pquo(f, h)
    2*x + 2*y - 2

    """
    return dmp_pdiv(f, g, u, K)[0]


def dmp_pexquo(f, g, u, K):
    """
    Polynomial pseudo-quotient in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**2 + x*y
    >>> g = 2*x + 2*y
    >>> h = 2*x + 2

    >>> R.dmp_pexquo(f, g)
    2*x

    >>> R.dmp_pexquo(f, h)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [[2], [2]] does not divide [[1], [1, 0], []]

    """
    q, r = dmp_pdiv(f, g, u, K)

    if dmp_zero_p(r, u):
        return q
    else:
        raise ExactQuotientFailed(f, g)


def dup_rr_div(f, g, K):
    """
    Univariate division with remainder over a ring.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_rr_div(x**2 + 1, 2*x - 4)
    (0, x**2 + 1)

    """
    df = dup_degree(f)
    dg = dup_degree(g)

    q, r, dr = [], f, df

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return q, r

    lc_g = dup_LC(g, K)

    while True:
        lc_r = dup_LC(r, K)

        if lc_r % lc_g:
            break

        c = K.exquo(lc_r, lc_g)
        j = dr - dg

        q = dup_add_term(q, c, j, K)
        h = dup_mul_term(g, c, j, K)
        r = dup_sub(r, h, K)

        _dr, dr = dr, dup_degree(r)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    return q, r


def dmp_rr_div(f, g, u, K):
    """
    Multivariate division with remainder over a ring.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_rr_div(x**2 + x*y, 2*x + 2)
    (0, x**2 + x*y)

    """
    if not u:
        return dup_rr_div(f, g, K)

    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if dg < 0:
        raise ZeroDivisionError("polynomial division")

    q, r, dr = dmp_zero(u), f, df

    if df < dg:
        return q, r

    lc_g, v = dmp_LC(g, K), u - 1

    while True:
        lc_r = dmp_LC(r, K)
        c, R = dmp_rr_div(lc_r, lc_g, v, K)

        if not dmp_zero_p(R, v):
            break

        j = dr - dg

        q = dmp_add_term(q, c, j, u, K)
        h = dmp_mul_term(g, c, j, u, K)
        r = dmp_sub(r, h, u, K)

        _dr, dr = dr, dmp_degree(r, u)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    return q, r


def dup_ff_div(f, g, K):
    """
    Polynomial division with remainder over a field.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_ff_div(x**2 + 1, 2*x - 4)
    (1/2*x + 1, 5)

    """
    df = dup_degree(f)
    dg = dup_degree(g)

    q, r, dr = [], f, df

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return q, r

    lc_g = dup_LC(g, K)

    while True:
        lc_r = dup_LC(r, K)

        c = K.exquo(lc_r, lc_g)
        j = dr - dg

        q = dup_add_term(q, c, j, K)
        h = dup_mul_term(g, c, j, K)
        r = dup_sub(r, h, K)

        _dr, dr = dr, dup_degree(r)

        if dr < dg:
            break
        elif dr == _dr and not K.is_Exact:
            # remove leading term created by rounding error
            r = dup_strip(r[1:])
            dr = dup_degree(r)
            if dr < dg:
                break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    return q, r


def dmp_ff_div(f, g, u, K):
    """
    Polynomial division with remainder over a field.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_ff_div(x**2 + x*y, 2*x + 2)
    (1/2*x + 1/2*y - 1/2, -y + 1)

    """
    if not u:
        return dup_ff_div(f, g, K)

    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if dg < 0:
        raise ZeroDivisionError("polynomial division")

    q, r, dr = dmp_zero(u), f, df

    if df < dg:
        return q, r

    lc_g, v = dmp_LC(g, K), u - 1

    while True:
        lc_r = dmp_LC(r, K)
        c, R = dmp_ff_div(lc_r, lc_g, v, K)

        if not dmp_zero_p(R, v):
            break

        j = dr - dg

        q = dmp_add_term(q, c, j, u, K)
        h = dmp_mul_term(g, c, j, u, K)
        r = dmp_sub(r, h, u, K)

        _dr, dr = dr, dmp_degree(r, u)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    return q, r


def dup_div(f, g, K):
    """
    Polynomial division with remainder in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_div(x**2 + 1, 2*x - 4)
    (0, x**2 + 1)

    >>> R, x = ring("x", QQ)
    >>> R.dup_div(x**2 + 1, 2*x - 4)
    (1/2*x + 1, 5)

    """
    if K.is_Field:
        return dup_ff_div(f, g, K)
    else:
        return dup_rr_div(f, g, K)


def dup_rem(f, g, K):
    """
    Returns polynomial remainder in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_rem(x**2 + 1, 2*x - 4)
    x**2 + 1

    >>> R, x = ring("x", QQ)
    >>> R.dup_rem(x**2 + 1, 2*x - 4)
    5

    """
    return dup_div(f, g, K)[1]


def dup_quo(f, g, K):
    """
    Returns exact polynomial quotient in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_quo(x**2 + 1, 2*x - 4)
    0

    >>> R, x = ring("x", QQ)
    >>> R.dup_quo(x**2 + 1, 2*x - 4)
    1/2*x + 1

    """
    return dup_div(f, g, K)[0]


def dup_exquo(f, g, K):
    """
    Returns polynomial quotient in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_exquo(x**2 - 1, x - 1)
    x + 1

    >>> R.dup_exquo(x**2 + 1, 2*x - 4)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [2, -4] does not divide [1, 0, 1]

    """
    q, r = dup_div(f, g, K)

    if not r:
        return q
    else:
        raise ExactQuotientFailed(f, g)


def dmp_div(f, g, u, K):
    """
    Polynomial division with remainder in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> R.dmp_div(x**2 + x*y, 2*x + 2)
    (0, x**2 + x*y)

    >>> R, x,y = ring("x,y", QQ)
    >>> R.dmp_div(x**2 + x*y, 2*x + 2)
    (1/2*x + 1/2*y - 1/2, -y + 1)

    """
    if K.is_Field:
        return dmp_ff_div(f, g, u, K)
    else:
        return dmp_rr_div(f, g, u, K)


def dmp_rem(f, g, u, K):
    """
    Returns polynomial remainder in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> R.dmp_rem(x**2 + x*y, 2*x + 2)
    x**2 + x*y

    >>> R, x,y = ring("x,y", QQ)
    >>> R.dmp_rem(x**2 + x*y, 2*x + 2)
    -y + 1

    """
    return dmp_div(f, g, u, K)[1]


def dmp_quo(f, g, u, K):
    """
    Returns exact polynomial quotient in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> R.dmp_quo(x**2 + x*y, 2*x + 2)
    0

    >>> R, x,y = ring("x,y", QQ)
    >>> R.dmp_quo(x**2 + x*y, 2*x + 2)
    1/2*x + 1/2*y - 1/2

    """
    return dmp_div(f, g, u, K)[0]


def dmp_exquo(f, g, u, K):
    """
    Returns polynomial quotient in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**2 + x*y
    >>> g = x + y
    >>> h = 2*x + 2

    >>> R.dmp_exquo(f, g)
    x

    >>> R.dmp_exquo(f, h)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [[2], [2]] does not divide [[1], [1, 0], []]

    """
    q, r = dmp_div(f, g, u, K)

    if dmp_zero_p(r, u):
        return q
    else:
        raise ExactQuotientFailed(f, g)


def dup_max_norm(f, K):
    """
    Returns maximum norm of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_max_norm(-x**2 + 2*x - 3)
    3

    """
    if not f:
        return K.zero
    else:
        return max(dup_abs(f, K))


def dmp_max_norm(f, u, K):
    """
    Returns maximum norm of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_max_norm(2*x*y - x - 3)
    3

    """
    if not u:
        return dup_max_norm(f, K)

    v = u - 1

    return max(dmp_max_norm(c, v, K) for c in f)


def dup_l1_norm(f, K):
    """
    Returns l1 norm of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_l1_norm(2*x**3 - 3*x**2 + 1)
    6

    """
    if not f:
        return K.zero
    else:
        return sum(dup_abs(f, K))


def dmp_l1_norm(f, u, K):
    """
    Returns l1 norm of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_l1_norm(2*x*y - x - 3)
    6

    """
    if not u:
        return dup_l1_norm(f, K)

    v = u - 1

    return sum(dmp_l1_norm(c, v, K) for c in f)


def dup_l2_norm_squared(f, K):
    """
    Returns squared l2 norm of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_l2_norm_squared(2*x**3 - 3*x**2 + 1)
    14

    """
    return sum([coeff**2 for coeff in f], K.zero)


def dmp_l2_norm_squared(f, u, K):
    """
    Returns squared l2 norm of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_l2_norm_squared(2*x*y - x - 3)
    14

    """
    if not u:
        return dup_l2_norm_squared(f, K)

    v = u - 1

    return sum(dmp_l2_norm_squared(c, v, K) for c in f)


def dup_expand(polys, K):
    """
    Multiply together several polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_expand([x**2 - 1, x, 2])
    2*x**3 - 2*x

    """
    if not polys:
        return [K.one]

    f = polys[0]

    for g in polys[1:]:
        f = dup_mul(f, g, K)

    return f


def dmp_expand(polys, u, K):
    """
    Multiply together several polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_expand([x**2 + y**2, x + 1])
    x**3 + x**2 + x*y**2 + y**2

    """
    if not polys:
        return dmp_one(u, K)

    f = polys[0]

    for g in polys[1:]:
        f = dmp_mul(f, g, u, K)

    return f

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\text_format.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Contains routines for printing protocol messages in text format.

Simple usage example::

  # Create a proto object and serialize it to a text proto string.
  message = my_proto_pb2.MyMessage(foo='bar')
  text_proto = text_format.MessageToString(message)

  # Parse a text proto string.
  message = text_format.Parse(text_proto, my_proto_pb2.MyMessage())
"""

__author__ = 'kenton@google.com (Kenton Varda)'

# TODO Import thread contention leads to test failures.
import encodings.raw_unicode_escape  # pylint: disable=unused-import
import encodings.unicode_escape  # pylint: disable=unused-import
import io
import math
import re

from google.protobuf.internal import decoder
from google.protobuf.internal import type_checkers
from google.protobuf import descriptor
from google.protobuf import text_encoding
from google.protobuf import unknown_fields

# pylint: disable=g-import-not-at-top
__all__ = ['MessageToString', 'Parse', 'PrintMessage', 'PrintField',
           'PrintFieldValue', 'Merge', 'MessageToBytes']

_INTEGER_CHECKERS = (type_checkers.Uint32ValueChecker(),
                     type_checkers.Int32ValueChecker(),
                     type_checkers.Uint64ValueChecker(),
                     type_checkers.Int64ValueChecker())
_FLOAT_INFINITY = re.compile('-?inf(?:inity)?f?$', re.IGNORECASE)
_FLOAT_NAN = re.compile('nanf?$', re.IGNORECASE)
_FLOAT_OCTAL_PREFIX = re.compile('-?0[0-9]+')
_QUOTES = frozenset(("'", '"'))
_ANY_FULL_TYPE_NAME = 'google.protobuf.Any'
_DEBUG_STRING_SILENT_MARKER = '\t '

_as_utf8_default = True


class Error(Exception):
  """Top-level module error for text_format."""


class ParseError(Error):
  """Thrown in case of text parsing or tokenizing error."""

  def __init__(self, message=None, line=None, column=None):
    if message is not None and line is not None:
      loc = str(line)
      if column is not None:
        loc += ':{0}'.format(column)
      message = '{0} : {1}'.format(loc, message)
    if message is not None:
      super(ParseError, self).__init__(message)
    else:
      super(ParseError, self).__init__()
    self._line = line
    self._column = column

  def GetLine(self):
    return self._line

  def GetColumn(self):
    return self._column


class TextWriter(object):

  def __init__(self, as_utf8):
    self._writer = io.StringIO()

  def write(self, val):
    return self._writer.write(val)

  def close(self):
    return self._writer.close()

  def getvalue(self):
    return self._writer.getvalue()


def MessageToString(
    message,
    as_utf8=_as_utf8_default,
    as_one_line=False,
    use_short_repeated_primitives=False,
    pointy_brackets=False,
    use_index_order=False,
    float_format=None,
    double_format=None,
    use_field_number=False,
    descriptor_pool=None,
    indent=0,
    message_formatter=None,
    print_unknown_fields=False,
    force_colon=False) -> str:
  """Convert protobuf message to text format.

  Double values can be formatted compactly with 15 digits of
  precision (which is the most that IEEE 754 "double" can guarantee)
  using double_format='.15g'. To ensure that converting to text and back to a
  proto will result in an identical value, double_format='.17g' should be used.

  Args:
    message: The protocol buffers message.
    as_utf8: Return unescaped Unicode for non-ASCII characters.
    as_one_line: Don't introduce newlines between fields.
    use_short_repeated_primitives: Use short repeated format for primitives.
    pointy_brackets: If True, use angle brackets instead of curly braces for
      nesting.
    use_index_order: If True, fields of a proto message will be printed using
      the order defined in source code instead of the field number, extensions
      will be printed at the end of the message and their relative order is
      determined by the extension number. By default, use the field number
      order.
    float_format (str): If set, use this to specify float field formatting
      (per the "Format Specification Mini-Language"); otherwise, shortest float
      that has same value in wire will be printed. Also affect double field
      if double_format is not set but float_format is set.
    double_format (str): If set, use this to specify double field formatting
      (per the "Format Specification Mini-Language"); if it is not set but
      float_format is set, use float_format. Otherwise, use ``str()``
    use_field_number: If True, print field numbers instead of names.
    descriptor_pool (DescriptorPool): Descriptor pool used to resolve Any types.
    indent (int): The initial indent level, in terms of spaces, for pretty
      print.
    message_formatter (function(message, indent, as_one_line) -> unicode|None):
      Custom formatter for selected sub-messages (usually based on message
      type). Use to pretty print parts of the protobuf for easier diffing.
    print_unknown_fields: If True, unknown fields will be printed.
    force_colon: If set, a colon will be added after the field name even if the
      field is a proto message.

  Returns:
    str: A string of the text formatted protocol buffer message.
  """
  out = TextWriter(as_utf8)
  printer = _Printer(
      out,
      indent,
      as_utf8,
      as_one_line,
      use_short_repeated_primitives,
      pointy_brackets,
      use_index_order,
      float_format,
      double_format,
      use_field_number,
      descriptor_pool,
      message_formatter,
      print_unknown_fields=print_unknown_fields,
      force_colon=force_colon)
  printer.PrintMessage(message)
  result = out.getvalue()
  out.close()
  if as_one_line:
    return result.rstrip()
  return result


def MessageToBytes(message, **kwargs) -> bytes:
  """Convert protobuf message to encoded text format.  See MessageToString."""
  text = MessageToString(message, **kwargs)
  if isinstance(text, bytes):
    return text
  codec = 'utf-8' if kwargs.get('as_utf8') else 'ascii'
  return text.encode(codec)


def _IsMapEntry(field):
  return (field.type == descriptor.FieldDescriptor.TYPE_MESSAGE and
          field.message_type.has_options and
          field.message_type.GetOptions().map_entry)


def _IsGroupLike(field):
  """Determines if a field is consistent with a proto2 group.

  Args:
    field: The field descriptor.

  Returns:
    True if this field is group-like, false otherwise.
  """
  # Groups are always tag-delimited.
  if field.type != descriptor.FieldDescriptor.TYPE_GROUP:
    return False

  # Group fields always are always the lowercase type name.
  if field.name != field.message_type.name.lower():
    return False

  if field.message_type.file != field.file:
    return False

  # Group messages are always defined in the same scope as the field.  File
  # level extensions will compare NULL == NULL here, which is why the file
  # comparison above is necessary to ensure both come from the same file.
  return (
      field.message_type.containing_type == field.extension_scope
      if field.is_extension
      else field.message_type.containing_type == field.containing_type
  )


def PrintMessage(message,
                 out,
                 indent=0,
                 as_utf8=_as_utf8_default,
                 as_one_line=False,
                 use_short_repeated_primitives=False,
                 pointy_brackets=False,
                 use_index_order=False,
                 float_format=None,
                 double_format=None,
                 use_field_number=False,
                 descriptor_pool=None,
                 message_formatter=None,
                 print_unknown_fields=False,
                 force_colon=False):
  """Convert the message to text format and write it to the out stream.

  Args:
    message: The Message object to convert to text format.
    out: A file handle to write the message to.
    indent: The initial indent level for pretty print.
    as_utf8: Return unescaped Unicode for non-ASCII characters.
    as_one_line: Don't introduce newlines between fields.
    use_short_repeated_primitives: Use short repeated format for primitives.
    pointy_brackets: If True, use angle brackets instead of curly braces for
      nesting.
    use_index_order: If True, print fields of a proto message using the order
      defined in source code instead of the field number. By default, use the
      field number order.
    float_format: If set, use this to specify float field formatting
      (per the "Format Specification Mini-Language"); otherwise, shortest
      float that has same value in wire will be printed. Also affect double
      field if double_format is not set but float_format is set.
    double_format: If set, use this to specify double field formatting
      (per the "Format Specification Mini-Language"); if it is not set but
      float_format is set, use float_format. Otherwise, str() is used.
    use_field_number: If True, print field numbers instead of names.
    descriptor_pool: A DescriptorPool used to resolve Any types.
    message_formatter: A function(message, indent, as_one_line): unicode|None
      to custom format selected sub-messages (usually based on message type).
      Use to pretty print parts of the protobuf for easier diffing.
    print_unknown_fields: If True, unknown fields will be printed.
    force_colon: If set, a colon will be added after the field name even if
      the field is a proto message.
  """
  printer = _Printer(
      out=out, indent=indent, as_utf8=as_utf8,
      as_one_line=as_one_line,
      use_short_repeated_primitives=use_short_repeated_primitives,
      pointy_brackets=pointy_brackets,
      use_index_order=use_index_order,
      float_format=float_format,
      double_format=double_format,
      use_field_number=use_field_number,
      descriptor_pool=descriptor_pool,
      message_formatter=message_formatter,
      print_unknown_fields=print_unknown_fields,
      force_colon=force_colon)
  printer.PrintMessage(message)


def PrintField(field,
               value,
               out,
               indent=0,
               as_utf8=_as_utf8_default,
               as_one_line=False,
               use_short_repeated_primitives=False,
               pointy_brackets=False,
               use_index_order=False,
               float_format=None,
               double_format=None,
               message_formatter=None,
               print_unknown_fields=False,
               force_colon=False):
  """Print a single field name/value pair."""
  printer = _Printer(out, indent, as_utf8, as_one_line,
                     use_short_repeated_primitives, pointy_brackets,
                     use_index_order, float_format, double_format,
                     message_formatter=message_formatter,
                     print_unknown_fields=print_unknown_fields,
                     force_colon=force_colon)
  printer.PrintField(field, value)


def PrintFieldValue(field,
                    value,
                    out,
                    indent=0,
                    as_utf8=_as_utf8_default,
                    as_one_line=False,
                    use_short_repeated_primitives=False,
                    pointy_brackets=False,
                    use_index_order=False,
                    float_format=None,
                    double_format=None,
                    message_formatter=None,
                    print_unknown_fields=False,
                    force_colon=False):
  """Print a single field value (not including name)."""
  printer = _Printer(out, indent, as_utf8, as_one_line,
                     use_short_repeated_primitives, pointy_brackets,
                     use_index_order, float_format, double_format,
                     message_formatter=message_formatter,
                     print_unknown_fields=print_unknown_fields,
                     force_colon=force_colon)
  printer.PrintFieldValue(field, value)


def _BuildMessageFromTypeName(type_name, descriptor_pool):
  """Returns a protobuf message instance.

  Args:
    type_name: Fully-qualified protobuf  message type name string.
    descriptor_pool: DescriptorPool instance.

  Returns:
    A Message instance of type matching type_name, or None if the a Descriptor
    wasn't found matching type_name.
  """
  # pylint: disable=g-import-not-at-top
  if descriptor_pool is None:
    from google.protobuf import descriptor_pool as pool_mod
    descriptor_pool = pool_mod.Default()
  from google.protobuf import message_factory
  try:
    message_descriptor = descriptor_pool.FindMessageTypeByName(type_name)
  except KeyError:
    return None
  message_type = message_factory.GetMessageClass(message_descriptor)
  return message_type()


# These values must match WireType enum in //google/protobuf/wire_format.h.
WIRETYPE_LENGTH_DELIMITED = 2
WIRETYPE_START_GROUP = 3


class _Printer(object):
  """Text format printer for protocol message."""

  def __init__(
      self,
      out,
      indent=0,
      as_utf8=_as_utf8_default,
      as_one_line=False,
      use_short_repeated_primitives=False,
      pointy_brackets=False,
      use_index_order=False,
      float_format=None,
      double_format=None,
      use_field_number=False,
      descriptor_pool=None,
      message_formatter=None,
      print_unknown_fields=False,
      force_colon=False):
    """Initialize the Printer.

    Double values can be formatted compactly with 15 digits of precision
    (which is the most that IEEE 754 "double" can guarantee) using
    double_format='.15g'. To ensure that converting to text and back to a proto
    will result in an identical value, double_format='.17g' should be used.

    Args:
      out: To record the text format result.
      indent: The initial indent level for pretty print.
      as_utf8: Return unescaped Unicode for non-ASCII characters.
      as_one_line: Don't introduce newlines between fields.
      use_short_repeated_primitives: Use short repeated format for primitives.
      pointy_brackets: If True, use angle brackets instead of curly braces for
        nesting.
      use_index_order: If True, print fields of a proto message using the order
        defined in source code instead of the field number. By default, use the
        field number order.
      float_format: If set, use this to specify float field formatting
        (per the "Format Specification Mini-Language"); otherwise, shortest
        float that has same value in wire will be printed. Also affect double
        field if double_format is not set but float_format is set.
      double_format: If set, use this to specify double field formatting
        (per the "Format Specification Mini-Language"); if it is not set but
        float_format is set, use float_format. Otherwise, str() is used.
      use_field_number: If True, print field numbers instead of names.
      descriptor_pool: A DescriptorPool used to resolve Any types.
      message_formatter: A function(message, indent, as_one_line): unicode|None
        to custom format selected sub-messages (usually based on message type).
        Use to pretty print parts of the protobuf for easier diffing.
      print_unknown_fields: If True, unknown fields will be printed.
      force_colon: If set, a colon will be added after the field name even if
        the field is a proto message.
    """
    self.out = out
    self.indent = indent
    self.as_utf8 = as_utf8
    self.as_one_line = as_one_line
    self.use_short_repeated_primitives = use_short_repeated_primitives
    self.pointy_brackets = pointy_brackets
    self.use_index_order = use_index_order
    self.float_format = float_format
    if double_format is not None:
      self.double_format = double_format
    else:
      self.double_format = float_format
    self.use_field_number = use_field_number
    self.descriptor_pool = descriptor_pool
    self.message_formatter = message_formatter
    self.print_unknown_fields = print_unknown_fields
    self.force_colon = force_colon

  def _TryPrintAsAnyMessage(self, message):
    """Serializes if message is a google.protobuf.Any field."""
    if '/' not in message.type_url:
      return False
    packed_message = _BuildMessageFromTypeName(message.TypeName(),
                                               self.descriptor_pool)
    if packed_message is not None:
      packed_message.MergeFromString(message.value)
      colon = ':' if self.force_colon else ''
      self.out.write('%s[%s]%s ' % (self.indent * ' ', message.type_url, colon))
      self._PrintMessageFieldValue(packed_message)
      self.out.write(' ' if self.as_one_line else '\n')
      return True
    else:
      return False

  def _TryCustomFormatMessage(self, message):
    formatted = self.message_formatter(message, self.indent, self.as_one_line)
    if formatted is None:
      return False

    out = self.out
    out.write(' ' * self.indent)
    out.write(formatted)
    out.write(' ' if self.as_one_line else '\n')
    return True

  def PrintMessage(self, message):
    """Convert protobuf message to text format.

    Args:
      message: The protocol buffers message.
    """
    if self.message_formatter and self._TryCustomFormatMessage(message):
      return
    if (message.DESCRIPTOR.full_name == _ANY_FULL_TYPE_NAME and
        self._TryPrintAsAnyMessage(message)):
      return
    fields = message.ListFields()
    if self.use_index_order:
      fields.sort(
          key=lambda x: x[0].number if x[0].is_extension else x[0].index)
    for field, value in fields:
      if _IsMapEntry(field):
        for key in sorted(value):
          # This is slow for maps with submessage entries because it copies the
          # entire tree.  Unfortunately this would take significant refactoring
          # of this file to work around.
          #
          # TODO: refactor and optimize if this becomes an issue.
          entry_submsg = value.GetEntryClass()(key=key, value=value[key])
          self.PrintField(field, entry_submsg)
      elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
        if (self.use_short_repeated_primitives
            and field.cpp_type != descriptor.FieldDescriptor.CPPTYPE_MESSAGE
            and field.cpp_type != descriptor.FieldDescriptor.CPPTYPE_STRING):
          self._PrintShortRepeatedPrimitivesValue(field, value)
        else:
          for element in value:
            self.PrintField(field, element)
      else:
        self.PrintField(field, value)

    if self.print_unknown_fields:
      self._PrintUnknownFields(unknown_fields.UnknownFieldSet(message))

  def _PrintUnknownFields(self, unknown_field_set):
    """Print unknown fields."""
    out = self.out
    for field in unknown_field_set:
      out.write(' ' * self.indent)
      out.write(str(field.field_number))
      if field.wire_type == WIRETYPE_START_GROUP:
        if self.as_one_line:
          out.write(' { ')
        else:
          out.write(' {\n')
          self.indent += 2

        self._PrintUnknownFields(field.data)

        if self.as_one_line:
          out.write('} ')
        else:
          self.indent -= 2
          out.write(' ' * self.indent + '}\n')
      elif field.wire_type == WIRETYPE_LENGTH_DELIMITED:
        try:
          # If this field is parseable as a Message, it is probably
          # an embedded message.
          # pylint: disable=protected-access
          (embedded_unknown_message, pos) = decoder._DecodeUnknownFieldSet(
              memoryview(field.data), 0, len(field.data))
        except Exception:    # pylint: disable=broad-except
          pos = 0

        if pos == len(field.data):
          if self.as_one_line:
            out.write(' { ')
          else:
            out.write(' {\n')
            self.indent += 2

          self._PrintUnknownFields(embedded_unknown_message)

          if self.as_one_line:
            out.write('} ')
          else:
            self.indent -= 2
            out.write(' ' * self.indent + '}\n')
        else:
          # A string or bytes field. self.as_utf8 may not work.
          out.write(': \"')
          out.write(text_encoding.CEscape(field.data, False))
          out.write('\" ' if self.as_one_line else '\"\n')
      else:
        # varint, fixed32, fixed64
        out.write(': ')
        out.write(str(field.data))
        out.write(' ' if self.as_one_line else '\n')

  def _PrintFieldName(self, field):
    """Print field name."""
    out = self.out
    out.write(' ' * self.indent)
    if self.use_field_number:
      out.write(str(field.number))
    else:
      if field.is_extension:
        out.write('[')
        if (field.containing_type.GetOptions().message_set_wire_format and
            field.type == descriptor.FieldDescriptor.TYPE_MESSAGE and
            field.label == descriptor.FieldDescriptor.LABEL_OPTIONAL):
          out.write(field.message_type.full_name)
        else:
          out.write(field.full_name)
        out.write(']')
      elif _IsGroupLike(field):
        # For groups, use the capitalized name.
        out.write(field.message_type.name)
      else:
        out.write(field.name)

    if (self.force_colon or
        field.cpp_type != descriptor.FieldDescriptor.CPPTYPE_MESSAGE):
      # The colon is optional in this case, but our cross-language golden files
      # don't include it. Here, the colon is only included if force_colon is
      # set to True
      out.write(':')

  def PrintField(self, field, value):
    """Print a single field name/value pair."""
    self._PrintFieldName(field)
    self.out.write(' ')
    self.PrintFieldValue(field, value)
    self.out.write(' ' if self.as_one_line else '\n')

  def _PrintShortRepeatedPrimitivesValue(self, field, value):
    """"Prints short repeated primitives value."""
    # Note: this is called only when value has at least one element.
    self._PrintFieldName(field)
    self.out.write(' [')
    for i in range(len(value) - 1):
      self.PrintFieldValue(field, value[i])
      self.out.write(', ')
    self.PrintFieldValue(field, value[-1])
    self.out.write(']')
    self.out.write(' ' if self.as_one_line else '\n')

  def _PrintMessageFieldValue(self, value):
    if self.pointy_brackets:
      openb = '<'
      closeb = '>'
    else:
      openb = '{'
      closeb = '}'

    if self.as_one_line:
      self.out.write('%s ' % openb)
      self.PrintMessage(value)
      self.out.write(closeb)
    else:
      self.out.write('%s\n' % openb)
      self.indent += 2
      self.PrintMessage(value)
      self.indent -= 2
      self.out.write(' ' * self.indent + closeb)

  def PrintFieldValue(self, field, value):
    """Print a single field value (not including name).

    For repeated fields, the value should be a single element.

    Args:
      field: The descriptor of the field to be printed.
      value: The value of the field.
    """
    out = self.out
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
      self._PrintMessageFieldValue(value)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
      enum_value = field.enum_type.values_by_number.get(value, None)
      if enum_value is not None:
        out.write(enum_value.name)
      else:
        out.write(str(value))
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
      out.write('\"')
      if isinstance(value, str) and not self.as_utf8:
        out_value = value.encode('utf-8')
      else:
        out_value = value
      if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
        # We always need to escape all binary data in TYPE_BYTES fields.
        out_as_utf8 = False
      else:
        out_as_utf8 = self.as_utf8
      out.write(text_encoding.CEscape(out_value, out_as_utf8))
      out.write('\"')
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
      if value:
        out.write('true')
      else:
        out.write('false')
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_FLOAT:
      if self.float_format is not None:
        out.write('{1:{0}}'.format(self.float_format, value))
      else:
        if math.isnan(value):
          out.write(str(value))
        else:
          out.write(str(type_checkers.ToShortestFloat(value)))
    elif (field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_DOUBLE and
          self.double_format is not None):
      out.write('{1:{0}}'.format(self.double_format, value))
    else:
      out.write(str(value))


def Parse(text,
          message,
          allow_unknown_extension=False,
          allow_field_number=False,
          descriptor_pool=None,
          allow_unknown_field=False):
  """Parses a text representation of a protocol message into a message.

  NOTE: for historical reasons this function does not clear the input
  message. This is different from what the binary msg.ParseFrom(...) does.
  If text contains a field already set in message, the value is appended if the
  field is repeated. Otherwise, an error is raised.

  Example::

    a = MyProto()
    a.repeated_field.append('test')
    b = MyProto()

    # Repeated fields are combined
    text_format.Parse(repr(a), b)
    text_format.Parse(repr(a), b) # repeated_field contains ["test", "test"]

    # Non-repeated fields cannot be overwritten
    a.singular_field = 1
    b.singular_field = 2
    text_format.Parse(repr(a), b) # ParseError

    # Binary version:
    b.ParseFromString(a.SerializeToString()) # repeated_field is now "test"

  Caller is responsible for clearing the message as needed.

  Args:
    text (str): Message text representation.
    message (Message): A protocol buffer message to merge into.
    allow_unknown_extension: if True, skip over missing extensions and keep
      parsing
    allow_field_number: if True, both field number and field name are allowed.
    descriptor_pool (DescriptorPool): Descriptor pool used to resolve Any types.
    allow_unknown_field: if True, skip over unknown field and keep
      parsing. Avoid to use this option if possible. It may hide some
      errors (e.g. spelling error on field name)

  Returns:
    Message: The same message passed as argument.

  Raises:
    ParseError: On text parsing problems.
  """
  return ParseLines(text.split(b'\n' if isinstance(text, bytes) else u'\n'),
                    message,
                    allow_unknown_extension,
                    allow_field_number,
                    descriptor_pool=descriptor_pool,
                    allow_unknown_field=allow_unknown_field)


def Merge(text,
          message,
          allow_unknown_extension=False,
          allow_field_number=False,
          descriptor_pool=None,
          allow_unknown_field=False):
  """Parses a text representation of a protocol message into a message.

  Like Parse(), but allows repeated values for a non-repeated field, and uses
  the last one. This means any non-repeated, top-level fields specified in text
  replace those in the message.

  Args:
    text (str): Message text representation.
    message (Message): A protocol buffer message to merge into.
    allow_unknown_extension: if True, skip over missing extensions and keep
      parsing
    allow_field_number: if True, both field number and field name are allowed.
    descriptor_pool (DescriptorPool): Descriptor pool used to resolve Any types.
    allow_unknown_field: if True, skip over unknown field and keep
      parsing. Avoid to use this option if possible. It may hide some
      errors (e.g. spelling error on field name)

  Returns:
    Message: The same message passed as argument.

  Raises:
    ParseError: On text parsing problems.
  """
  return MergeLines(
      text.split(b'\n' if isinstance(text, bytes) else u'\n'),
      message,
      allow_unknown_extension,
      allow_field_number,
      descriptor_pool=descriptor_pool,
      allow_unknown_field=allow_unknown_field)


def ParseLines(lines,
               message,
               allow_unknown_extension=False,
               allow_field_number=False,
               descriptor_pool=None,
               allow_unknown_field=False):
  """Parses a text representation of a protocol message into a message.

  See Parse() for caveats.

  Args:
    lines: An iterable of lines of a message's text representation.
    message: A protocol buffer message to merge into.
    allow_unknown_extension: if True, skip over missing extensions and keep
      parsing
    allow_field_number: if True, both field number and field name are allowed.
    descriptor_pool: A DescriptorPool used to resolve Any types.
    allow_unknown_field: if True, skip over unknown field and keep
      parsing. Avoid to use this option if possible. It may hide some
      errors (e.g. spelling error on field name)

  Returns:
    The same message passed as argument.

  Raises:
    ParseError: On text parsing problems.
  """
  parser = _Parser(allow_unknown_extension,
                   allow_field_number,
                   descriptor_pool=descriptor_pool,
                   allow_unknown_field=allow_unknown_field)
  return parser.ParseLines(lines, message)


def MergeLines(lines,
               message,
               allow_unknown_extension=False,
               allow_field_number=False,
               descriptor_pool=None,
               allow_unknown_field=False):
  """Parses a text representation of a protocol message into a message.

  See Merge() for more details.

  Args:
    lines: An iterable of lines of a message's text representation.
    message: A protocol buffer message to merge into.
    allow_unknown_extension: if True, skip over missing extensions and keep
      parsing
    allow_field_number: if True, both field number and field name are allowed.
    descriptor_pool: A DescriptorPool used to resolve Any types.
    allow_unknown_field: if True, skip over unknown field and keep
      parsing. Avoid to use this option if possible. It may hide some
      errors (e.g. spelling error on field name)

  Returns:
    The same message passed as argument.

  Raises:
    ParseError: On text parsing problems.
  """
  parser = _Parser(allow_unknown_extension,
                   allow_field_number,
                   descriptor_pool=descriptor_pool,
                   allow_unknown_field=allow_unknown_field)
  return parser.MergeLines(lines, message)


class _Parser(object):
  """Text format parser for protocol message."""

  def __init__(self,
               allow_unknown_extension=False,
               allow_field_number=False,
               descriptor_pool=None,
               allow_unknown_field=False):
    self.allow_unknown_extension = allow_unknown_extension
    self.allow_field_number = allow_field_number
    self.descriptor_pool = descriptor_pool
    self.allow_unknown_field = allow_unknown_field

  def ParseLines(self, lines, message):
    """Parses a text representation of a protocol message into a message."""
    self._allow_multiple_scalars = False
    self._ParseOrMerge(lines, message)
    return message

  def MergeLines(self, lines, message):
    """Merges a text representation of a protocol message into a message."""
    self._allow_multiple_scalars = True
    self._ParseOrMerge(lines, message)
    return message

  def _ParseOrMerge(self, lines, message):
    """Converts a text representation of a protocol message into a message.

    Args:
      lines: Lines of a message's text representation.
      message: A protocol buffer message to merge into.

    Raises:
      ParseError: On text parsing problems.
    """
    # Tokenize expects native str lines.
    try:
      str_lines = (
          line if isinstance(line, str) else line.decode('utf-8')
          for line in lines)
      tokenizer = Tokenizer(str_lines)
    except UnicodeDecodeError as e:
      raise ParseError from e
    if message:
      self.root_type = message.DESCRIPTOR.full_name
    while not tokenizer.AtEnd():
      self._MergeField(tokenizer, message)

  def _MergeField(self, tokenizer, message):
    """Merges a single protocol message field into a message.

    Args:
      tokenizer: A tokenizer to parse the field name and values.
      message: A protocol message to record the data.

    Raises:
      ParseError: In case of text parsing problems.
    """
    message_descriptor = message.DESCRIPTOR
    if (message_descriptor.full_name == _ANY_FULL_TYPE_NAME and
        tokenizer.TryConsume('[')):
      type_url_prefix, packed_type_name = self._ConsumeAnyTypeUrl(tokenizer)
      tokenizer.Consume(']')
      tokenizer.TryConsume(':')
      self._DetectSilentMarker(tokenizer, message_descriptor.full_name,
                               type_url_prefix + '/' + packed_type_name)
      if tokenizer.TryConsume('<'):
        expanded_any_end_token = '>'
      else:
        tokenizer.Consume('{')
        expanded_any_end_token = '}'
      expanded_any_sub_message = _BuildMessageFromTypeName(packed_type_name,
                                                           self.descriptor_pool)
      # Direct comparison with None is used instead of implicit bool conversion
      # to avoid false positives with falsy initial values, e.g. for
      # google.protobuf.ListValue.
      if expanded_any_sub_message is None:
        raise ParseError('Type %s not found in descriptor pool' %
                         packed_type_name)
      while not tokenizer.TryConsume(expanded_any_end_token):
        if tokenizer.AtEnd():
          raise tokenizer.ParseErrorPreviousToken('Expected "%s".' %
                                                  (expanded_any_end_token,))
        self._MergeField(tokenizer, expanded_any_sub_message)
      deterministic = False

      message.Pack(expanded_any_sub_message,
                   type_url_prefix=type_url_prefix,
                   deterministic=deterministic)
      return

    if tokenizer.TryConsume('['):
      name = [tokenizer.ConsumeIdentifier()]
      while tokenizer.TryConsume('.'):
        name.append(tokenizer.ConsumeIdentifier())
      name = '.'.join(name)

      if not message_descriptor.is_extendable:
        raise tokenizer.ParseErrorPreviousToken(
            'Message type "%s" does not have extensions.' %
            message_descriptor.full_name)
      # pylint: disable=protected-access
      field = message.Extensions._FindExtensionByName(name)
      # pylint: enable=protected-access
      if not field:
        if self.allow_unknown_extension:
          field = None
        else:
          raise tokenizer.ParseErrorPreviousToken(
              'Extension "%s" not registered. '
              'Did you import the _pb2 module which defines it? '
              'If you are trying to place the extension in the MessageSet '
              'field of another message that is in an Any or MessageSet field, '
              'that message\'s _pb2 module must be imported as well' % name)
      elif message_descriptor != field.containing_type:
        raise tokenizer.ParseErrorPreviousToken(
            'Extension "%s" does not extend message type "%s".' %
            (name, message_descriptor.full_name))

      tokenizer.Consume(']')

    else:
      name = tokenizer.ConsumeIdentifierOrNumber()
      if self.allow_field_number and name.isdigit():
        number = ParseInteger(name, True, True)
        field = message_descriptor.fields_by_number.get(number, None)
        if not field and message_descriptor.is_extendable:
          field = message.Extensions._FindExtensionByNumber(number)
      else:
        field = message_descriptor.fields_by_name.get(name, None)

        # Group names are expected to be capitalized as they appear in the
        # .proto file, which actually matches their type names, not their field
        # names.
        if not field:
          field = message_descriptor.fields_by_name.get(name.lower(), None)
          if field and not _IsGroupLike(field):
            field = None
          if field and field.message_type.name != name:
            field = None

      if not field and not self.allow_unknown_field:
        raise tokenizer.ParseErrorPreviousToken(
            'Message type "%s" has no field named "%s".' %
            (message_descriptor.full_name, name))

    if field:
      if not self._allow_multiple_scalars and field.containing_oneof:
        # Check if there's a different field set in this oneof.
        # Note that we ignore the case if the same field was set before, and we
        # apply _allow_multiple_scalars to non-scalar fields as well.
        which_oneof = message.WhichOneof(field.containing_oneof.name)
        if which_oneof is not None and which_oneof != field.name:
          raise tokenizer.ParseErrorPreviousToken(
              'Field "%s" is specified along with field "%s", another member '
              'of oneof "%s" for message type "%s".' %
              (field.name, which_oneof, field.containing_oneof.name,
               message_descriptor.full_name))

      if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
        tokenizer.TryConsume(':')
        self._DetectSilentMarker(tokenizer, message_descriptor.full_name,
                                 field.full_name)
        merger = self._MergeMessageField
      else:
        tokenizer.Consume(':')
        self._DetectSilentMarker(tokenizer, message_descriptor.full_name,
                                 field.full_name)
        merger = self._MergeScalarField

      if (field.label == descriptor.FieldDescriptor.LABEL_REPEATED and
          tokenizer.TryConsume('[')):
        # Short repeated format, e.g. "foo: [1, 2, 3]"
        if not tokenizer.TryConsume(']'):
          while True:
            merger(tokenizer, message, field)
            if tokenizer.TryConsume(']'):
              break
            tokenizer.Consume(',')

      else:
        merger(tokenizer, message, field)

    else:  # Proto field is unknown.
      assert (self.allow_unknown_extension or self.allow_unknown_field)
      self._SkipFieldContents(tokenizer, name, message_descriptor.full_name)

    # For historical reasons, fields may optionally be separated by commas or
    # semicolons.
    if not tokenizer.TryConsume(','):
      tokenizer.TryConsume(';')

  def _LogSilentMarker(self, immediate_message_type, field_name):
    pass

  def _DetectSilentMarker(self, tokenizer, immediate_message_type, field_name):
    if tokenizer.contains_silent_marker_before_current_token:
      self._LogSilentMarker(immediate_message_type, field_name)

  def _ConsumeAnyTypeUrl(self, tokenizer):
    """Consumes a google.protobuf.Any type URL and returns the type name."""
    # Consume "type.googleapis.com/".
    prefix = [tokenizer.ConsumeIdentifier()]
    tokenizer.Consume('.')
    prefix.append(tokenizer.ConsumeIdentifier())
    tokenizer.Consume('.')
    prefix.append(tokenizer.ConsumeIdentifier())
    tokenizer.Consume('/')
    # Consume the fully-qualified type name.
    name = [tokenizer.ConsumeIdentifier()]
    while tokenizer.TryConsume('.'):
      name.append(tokenizer.ConsumeIdentifier())
    return '.'.join(prefix), '.'.join(name)

  def _MergeMessageField(self, tokenizer, message, field):
    """Merges a single scalar field into a message.

    Args:
      tokenizer: A tokenizer to parse the field value.
      message: The message of which field is a member.
      field: The descriptor of the field to be merged.

    Raises:
      ParseError: In case of text parsing problems.
    """
    is_map_entry = _IsMapEntry(field)

    if tokenizer.TryConsume('<'):
      end_token = '>'
    else:
      tokenizer.Consume('{')
      end_token = '}'

    if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
      if field.is_extension:
        sub_message = message.Extensions[field].add()
      elif is_map_entry:
        sub_message = getattr(message, field.name).GetEntryClass()()
      else:
        sub_message = getattr(message, field.name).add()
    else:
      if field.is_extension:
        if (not self._allow_multiple_scalars and
            message.HasExtension(field)):
          raise tokenizer.ParseErrorPreviousToken(
              'Message type "%s" should not have multiple "%s" extensions.' %
              (message.DESCRIPTOR.full_name, field.full_name))
        sub_message = message.Extensions[field]
      else:
        # Also apply _allow_multiple_scalars to message field.
        # TODO: Change to _allow_singular_overwrites.
        if (not self._allow_multiple_scalars and
            message.HasField(field.name)):
          raise tokenizer.ParseErrorPreviousToken(
              'Message type "%s" should not have multiple "%s" fields.' %
              (message.DESCRIPTOR.full_name, field.name))
        sub_message = getattr(message, field.name)
      sub_message.SetInParent()

    while not tokenizer.TryConsume(end_token):
      if tokenizer.AtEnd():
        raise tokenizer.ParseErrorPreviousToken('Expected "%s".' % (end_token,))
      self._MergeField(tokenizer, sub_message)

    if is_map_entry:
      value_cpptype = field.message_type.fields_by_name['value'].cpp_type
      if value_cpptype == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
        value = getattr(message, field.name)[sub_message.key]
        value.CopyFrom(sub_message.value)
      else:
        getattr(message, field.name)[sub_message.key] = sub_message.value

  def _MergeScalarField(self, tokenizer, message, field):
    """Merges a single scalar field into a message.

    Args:
      tokenizer: A tokenizer to parse the field value.
      message: A protocol message to record the data.
      field: The descriptor of the field to be merged.

    Raises:
      ParseError: In case of text parsing problems.
      RuntimeError: On runtime errors.
    """
    _ = self.allow_unknown_extension
    value = None

    if field.type in (descriptor.FieldDescriptor.TYPE_INT32,
                      descriptor.FieldDescriptor.TYPE_SINT32,
                      descriptor.FieldDescriptor.TYPE_SFIXED32):
      value = _ConsumeInt32(tokenizer)
    elif field.type in (descriptor.FieldDescriptor.TYPE_INT64,
                        descriptor.FieldDescriptor.TYPE_SINT64,
                        descriptor.FieldDescriptor.TYPE_SFIXED64):
      value = _ConsumeInt64(tokenizer)
    elif field.type in (descriptor.FieldDescriptor.TYPE_UINT32,
                        descriptor.FieldDescriptor.TYPE_FIXED32):
      value = _ConsumeUint32(tokenizer)
    elif field.type in (descriptor.FieldDescriptor.TYPE_UINT64,
                        descriptor.FieldDescriptor.TYPE_FIXED64):
      value = _ConsumeUint64(tokenizer)
    elif field.type in (descriptor.FieldDescriptor.TYPE_FLOAT,
                        descriptor.FieldDescriptor.TYPE_DOUBLE):
      value = tokenizer.ConsumeFloat()
    elif field.type == descriptor.FieldDescriptor.TYPE_BOOL:
      value = tokenizer.ConsumeBool()
    elif field.type == descriptor.FieldDescriptor.TYPE_STRING:
      value = tokenizer.ConsumeString()
    elif field.type == descriptor.FieldDescriptor.TYPE_BYTES:
      value = tokenizer.ConsumeByteString()
    elif field.type == descriptor.FieldDescriptor.TYPE_ENUM:
      value = tokenizer.ConsumeEnum(field)
    else:
      raise RuntimeError('Unknown field type %d' % field.type)

    if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
      if field.is_extension:
        message.Extensions[field].append(value)
      else:
        getattr(message, field.name).append(value)
    else:
      if field.is_extension:
        if (not self._allow_multiple_scalars and
            field.has_presence and
            message.HasExtension(field)):
          raise tokenizer.ParseErrorPreviousToken(
              'Message type "%s" should not have multiple "%s" extensions.' %
              (message.DESCRIPTOR.full_name, field.full_name))
        else:
          message.Extensions[field] = value
      else:
        duplicate_error = False
        if not self._allow_multiple_scalars:
          if field.has_presence:
            duplicate_error = message.HasField(field.name)
          else:
            # For field that doesn't represent presence, try best effort to
            # check multiple scalars by compare to default values.
            duplicate_error = not decoder.IsDefaultScalarValue(
                getattr(message, field.name)
            )

        if duplicate_error:
          raise tokenizer.ParseErrorPreviousToken(
              'Message type "%s" should not have multiple "%s" fields.' %
              (message.DESCRIPTOR.full_name, field.name))
        else:
          setattr(message, field.name, value)

  def _SkipFieldContents(self, tokenizer, field_name, immediate_message_type):
    """Skips over contents (value or message) of a field.

    Args:
      tokenizer: A tokenizer to parse the field name and values.
      field_name: The field name currently being parsed.
      immediate_message_type: The type of the message immediately containing
        the silent marker.
    """
    # Try to guess the type of this field.
    # If this field is not a message, there should be a ":" between the
    # field name and the field value and also the field value should not
    # start with "{" or "<" which indicates the beginning of a message body.
    # If there is no ":" or there is a "{" or "<" after ":", this field has
    # to be a message or the input is ill-formed.
    if tokenizer.TryConsume(
        ':') and not tokenizer.LookingAt('{') and not tokenizer.LookingAt('<'):
      self._DetectSilentMarker(tokenizer, immediate_message_type, field_name)
      if tokenizer.LookingAt('['):
        self._SkipRepeatedFieldValue(tokenizer, immediate_message_type)
      else:
        self._SkipFieldValue(tokenizer)
    else:
      self._DetectSilentMarker(tokenizer, immediate_message_type, field_name)
      self._SkipFieldMessage(tokenizer, immediate_message_type)

  def _SkipField(self, tokenizer, immediate_message_type):
    """Skips over a complete field (name and value/message).

    Args:
      tokenizer: A tokenizer to parse the field name and values.
      immediate_message_type: The type of the message immediately containing
        the silent marker.
    """
    field_name = ''
    if tokenizer.TryConsume('['):
      # Consume extension or google.protobuf.Any type URL
      field_name += '[' + tokenizer.ConsumeIdentifier()
      num_identifiers = 1
      while tokenizer.TryConsume('.'):
        field_name += '.' + tokenizer.ConsumeIdentifier()
        num_identifiers += 1
      # This is possibly a type URL for an Any message.
      if num_identifiers == 3 and tokenizer.TryConsume('/'):
        field_name += '/' + tokenizer.ConsumeIdentifier()
        while tokenizer.TryConsume('.'):
          field_name += '.' + tokenizer.ConsumeIdentifier()
      tokenizer.Consume(']')
      field_name += ']'
    else:
      field_name += tokenizer.ConsumeIdentifierOrNumber()

    self._SkipFieldContents(tokenizer, field_name, immediate_message_type)

    # For historical reasons, fields may optionally be separated by commas or
    # semicolons.
    if not tokenizer.TryConsume(','):
      tokenizer.TryConsume(';')

  def _SkipFieldMessage(self, tokenizer, immediate_message_type):
    """Skips over a field message.

    Args:
      tokenizer: A tokenizer to parse the field name and values.
      immediate_message_type: The type of the message immediately containing
        the silent marker
    """
    if tokenizer.TryConsume('<'):
      delimiter = '>'
    else:
      tokenizer.Consume('{')
      delimiter = '}'

    while not tokenizer.LookingAt('>') and not tokenizer.LookingAt('}'):
      self._SkipField(tokenizer, immediate_message_type)

    tokenizer.Consume(delimiter)

  def _SkipFieldValue(self, tokenizer):
    """Skips over a field value.

    Args:
      tokenizer: A tokenizer to parse the field name and values.

    Raises:
      ParseError: In case an invalid field value is found.
    """
    if (not tokenizer.TryConsumeByteString()and
        not tokenizer.TryConsumeIdentifier() and
        not _TryConsumeInt64(tokenizer) and
        not _TryConsumeUint64(tokenizer) and
        not tokenizer.TryConsumeFloat()):
      raise ParseError('Invalid field value: ' + tokenizer.token)

  def _SkipRepeatedFieldValue(self, tokenizer, immediate_message_type):
    """Skips over a repeated field value.

    Args:
      tokenizer: A tokenizer to parse the field value.
    """
    tokenizer.Consume('[')
    if not tokenizer.TryConsume(']'):
      while True:
        if tokenizer.LookingAt('<') or tokenizer.LookingAt('{'):
          self._SkipFieldMessage(tokenizer, immediate_message_type)
        else:
          self._SkipFieldValue(tokenizer)
        if tokenizer.TryConsume(']'):
          break
        tokenizer.Consume(',')


class Tokenizer(object):
  """Protocol buffer text representation tokenizer.

  This class handles the lower level string parsing by splitting it into
  meaningful tokens.

  It was directly ported from the Java protocol buffer API.
  """

  _WHITESPACE = re.compile(r'\s+')
  _COMMENT = re.compile(r'(\s*#.*$)', re.MULTILINE)
  _WHITESPACE_OR_COMMENT = re.compile(r'(\s|(#.*$))+', re.MULTILINE)
  _TOKEN = re.compile('|'.join([
      r'[a-zA-Z_][0-9a-zA-Z_+-]*',  # an identifier
      r'([0-9+-]|(\.[0-9]))[0-9a-zA-Z_.+-]*',  # a number
  ] + [  # quoted str for each quote mark
      # Avoid backtracking! https://stackoverflow.com/a/844267
      r'{qt}[^{qt}\n\\]*((\\.)+[^{qt}\n\\]*)*({qt}|\\?$)'.format(qt=mark)
      for mark in _QUOTES
  ]))

  _IDENTIFIER = re.compile(r'[^\d\W]\w*')
  _IDENTIFIER_OR_NUMBER = re.compile(r'\w+')

  def __init__(self, lines, skip_comments=True):
    self._position = 0
    self._line = -1
    self._column = 0
    self._token_start = None
    self.token = ''
    self._lines = iter(lines)
    self._current_line = ''
    self._previous_line = 0
    self._previous_column = 0
    self._more_lines = True
    self._skip_comments = skip_comments
    self._whitespace_pattern = (skip_comments and self._WHITESPACE_OR_COMMENT
                                or self._WHITESPACE)
    self.contains_silent_marker_before_current_token = False

    self._SkipWhitespace()
    self.NextToken()

  def LookingAt(self, token):
    return self.token == token

  def AtEnd(self):
    """Checks the end of the text was reached.

    Returns:
      True iff the end was reached.
    """
    return not self.token

  def _PopLine(self):
    while len(self._current_line) <= self._column:
      try:
        self._current_line = next(self._lines)
      except StopIteration:
        self._current_line = ''
        self._more_lines = False
        return
      else:
        self._line += 1
        self._column = 0

  def _SkipWhitespace(self):
    while True:
      self._PopLine()
      match = self._whitespace_pattern.match(self._current_line, self._column)
      if not match:
        break
      self.contains_silent_marker_before_current_token = match.group(0) == (
          ' ' + _DEBUG_STRING_SILENT_MARKER)
      length = len(match.group(0))
      self._column += length

  def TryConsume(self, token):
    """Tries to consume a given piece of text.

    Args:
      token: Text to consume.

    Returns:
      True iff the text was consumed.
    """
    if self.token == token:
      self.NextToken()
      return True
    return False

  def Consume(self, token):
    """Consumes a piece of text.

    Args:
      token: Text to consume.

    Raises:
      ParseError: If the text couldn't be consumed.
    """
    if not self.TryConsume(token):
      raise self.ParseError('Expected "%s".' % token)

  def ConsumeComment(self):
    result = self.token
    if not self._COMMENT.match(result):
      raise self.ParseError('Expected comment.')
    self.NextToken()
    return result

  def ConsumeCommentOrTrailingComment(self):
    """Consumes a comment, returns a 2-tuple (trailing bool, comment str)."""

    # Tokenizer initializes _previous_line and _previous_column to 0. As the
    # tokenizer starts, it looks like there is a previous token on the line.
    just_started = self._line == 0 and self._column == 0

    before_parsing = self._previous_line
    comment = self.ConsumeComment()

    # A trailing comment is a comment on the same line than the previous token.
    trailing = (self._previous_line == before_parsing
                and not just_started)

    return trailing, comment

  def TryConsumeIdentifier(self):
    try:
      self.ConsumeIdentifier()
      return True
    except ParseError:
      return False

  def ConsumeIdentifier(self):
    """Consumes protocol message field identifier.

    Returns:
      Identifier string.

    Raises:
      ParseError: If an identifier couldn't be consumed.
    """
    result = self.token
    if not self._IDENTIFIER.match(result):
      raise self.ParseError('Expected identifier.')
    self.NextToken()
    return result

  def TryConsumeIdentifierOrNumber(self):
    try:
      self.ConsumeIdentifierOrNumber()
      return True
    except ParseError:
      return False

  def ConsumeIdentifierOrNumber(self):
    """Consumes protocol message field identifier.

    Returns:
      Identifier string.

    Raises:
      ParseError: If an identifier couldn't be consumed.
    """
    result = self.token
    if not self._IDENTIFIER_OR_NUMBER.match(result):
      raise self.ParseError('Expected identifier or number, got %s.' % result)
    self.NextToken()
    return result

  def TryConsumeInteger(self):
    try:
      self.ConsumeInteger()
      return True
    except ParseError:
      return False

  def ConsumeInteger(self):
    """Consumes an integer number.

    Returns:
      The integer parsed.

    Raises:
      ParseError: If an integer couldn't be consumed.
    """
    try:
      result = _ParseAbstractInteger(self.token)
    except ValueError as e:
      raise self.ParseError(str(e))
    self.NextToken()
    return result

  def TryConsumeFloat(self):
    try:
      self.ConsumeFloat()
      return True
    except ParseError:
      return False

  def ConsumeFloat(self):
    """Consumes an floating point number.

    Returns:
      The number parsed.

    Raises:
      ParseError: If a floating point number couldn't be consumed.
    """
    try:
      result = ParseFloat(self.token)
    except ValueError as e:
      raise self.ParseError(str(e))
    self.NextToken()
    return result

  def ConsumeBool(self):
    """Consumes a boolean value.

    Returns:
      The bool parsed.

    Raises:
      ParseError: If a boolean value couldn't be consumed.
    """
    try:
      result = ParseBool(self.token)
    except ValueError as e:
      raise self.ParseError(str(e))
    self.NextToken()
    return result

  def TryConsumeByteString(self):
    try:
      self.ConsumeByteString()
      return True
    except ParseError:
      return False

  def ConsumeString(self):
    """Consumes a string value.

    Returns:
      The string parsed.

    Raises:
      ParseError: If a string value couldn't be consumed.
    """
    the_bytes = self.ConsumeByteString()
    try:
      return str(the_bytes, 'utf-8')
    except UnicodeDecodeError as e:
      raise self._StringParseError(e)

  def ConsumeByteString(self):
    """Consumes a byte array value.

    Returns:
      The array parsed (as a string).

    Raises:
      ParseError: If a byte array value couldn't be consumed.
    """
    the_list = [self._ConsumeSingleByteString()]
    while self.token and self.token[0] in _QUOTES:
      the_list.append(self._ConsumeSingleByteString())
    return b''.join(the_list)

  def _ConsumeSingleByteString(self):
    """Consume one token of a string literal.

    String literals (whether bytes or text) can come in multiple adjacent
    tokens which are automatically concatenated, like in C or Python.  This
    method only consumes one token.

    Returns:
      The token parsed.
    Raises:
      ParseError: When the wrong format data is found.
    """
    text = self.token
    if len(text) < 1 or text[0] not in _QUOTES:
      raise self.ParseError('Expected string but found: %r' % (text,))

    if len(text) < 2 or text[-1] != text[0]:
      raise self.ParseError('String missing ending quote: %r' % (text,))

    try:
      result = text_encoding.CUnescape(text[1:-1])
    except ValueError as e:
      raise self.ParseError(str(e))
    self.NextToken()
    return result

  def ConsumeEnum(self, field):
    try:
      result = ParseEnum(field, self.token)
    except ValueError as e:
      raise self.ParseError(str(e))
    self.NextToken()
    return result

  def ParseErrorPreviousToken(self, message):
    """Creates and *returns* a ParseError for the previously read token.

    Args:
      message: A message to set for the exception.

    Returns:
      A ParseError instance.
    """
    return ParseError(message, self._previous_line + 1,
                      self._previous_column + 1)

  def ParseError(self, message):
    """Creates and *returns* a ParseError for the current token."""
    return ParseError('\'' + self._current_line + '\': ' + message,
                      self._line + 1, self._column + 1)

  def _StringParseError(self, e):
    return self.ParseError('Couldn\'t parse string: ' + str(e))

  def NextToken(self):
    """Reads the next meaningful token."""
    self._previous_line = self._line
    self._previous_column = self._column
    self.contains_silent_marker_before_current_token = False

    self._column += len(self.token)
    self._SkipWhitespace()

    if not self._more_lines:
      self.token = ''
      return

    match = self._TOKEN.match(self._current_line, self._column)
    if not match and not self._skip_comments:
      match = self._COMMENT.match(self._current_line, self._column)
    if match:
      token = match.group(0)
      self.token = token
    else:
      self.token = self._current_line[self._column]

# Aliased so it can still be accessed by current visibility violators.
# TODO: Migrate violators to textformat_tokenizer.
_Tokenizer = Tokenizer  # pylint: disable=invalid-name


def _ConsumeInt32(tokenizer):
  """Consumes a signed 32bit integer number from tokenizer.

  Args:
    tokenizer: A tokenizer used to parse the number.

  Returns:
    The integer parsed.

  Raises:
    ParseError: If a signed 32bit integer couldn't be consumed.
  """
  return _ConsumeInteger(tokenizer, is_signed=True, is_long=False)


def _ConsumeUint32(tokenizer):
  """Consumes an unsigned 32bit integer number from tokenizer.

  Args:
    tokenizer: A tokenizer used to parse the number.

  Returns:
    The integer parsed.

  Raises:
    ParseError: If an unsigned 32bit integer couldn't be consumed.
  """
  return _ConsumeInteger(tokenizer, is_signed=False, is_long=False)


def _TryConsumeInt64(tokenizer):
  try:
    _ConsumeInt64(tokenizer)
    return True
  except ParseError:
    return False


def _ConsumeInt64(tokenizer):
  """Consumes a signed 32bit integer number from tokenizer.

  Args:
    tokenizer: A tokenizer used to parse the number.

  Returns:
    The integer parsed.

  Raises:
    ParseError: If a signed 32bit integer couldn't be consumed.
  """
  return _ConsumeInteger(tokenizer, is_signed=True, is_long=True)


def _TryConsumeUint64(tokenizer):
  try:
    _ConsumeUint64(tokenizer)
    return True
  except ParseError:
    return False


def _ConsumeUint64(tokenizer):
  """Consumes an unsigned 64bit integer number from tokenizer.

  Args:
    tokenizer: A tokenizer used to parse the number.

  Returns:
    The integer parsed.

  Raises:
    ParseError: If an unsigned 64bit integer couldn't be consumed.
  """
  return _ConsumeInteger(tokenizer, is_signed=False, is_long=True)


def _ConsumeInteger(tokenizer, is_signed=False, is_long=False):
  """Consumes an integer number from tokenizer.

  Args:
    tokenizer: A tokenizer used to parse the number.
    is_signed: True if a signed integer must be parsed.
    is_long: True if a long integer must be parsed.

  Returns:
    The integer parsed.

  Raises:
    ParseError: If an integer with given characteristics couldn't be consumed.
  """
  try:
    result = ParseInteger(tokenizer.token, is_signed=is_signed, is_long=is_long)
  except ValueError as e:
    raise tokenizer.ParseError(str(e))
  tokenizer.NextToken()
  return result


def ParseInteger(text, is_signed=False, is_long=False):
  """Parses an integer.

  Args:
    text: The text to parse.
    is_signed: True if a signed integer must be parsed.
    is_long: True if a long integer must be parsed.

  Returns:
    The integer value.

  Raises:
    ValueError: Thrown Iff the text is not a valid integer.
  """
  # Do the actual parsing. Exception handling is propagated to caller.
  result = _ParseAbstractInteger(text)

  # Check if the integer is sane. Exceptions handled by callers.
  checker = _INTEGER_CHECKERS[2 * int(is_long) + int(is_signed)]
  checker.CheckValue(result)
  return result


def _ParseAbstractInteger(text):
  """Parses an integer without checking size/signedness.

  Args:
    text: The text to parse.

  Returns:
    The integer value.

  Raises:
    ValueError: Thrown Iff the text is not a valid integer.
  """
  # Do the actual parsing. Exception handling is propagated to caller.
  orig_text = text
  c_octal_match = re.match(r'(-?)0(\d+)$', text)
  if c_octal_match:
    # Python 3 no longer supports 0755 octal syntax without the 'o', so
    # we always use the '0o' prefix for multi-digit numbers starting with 0.
    text = c_octal_match.group(1) + '0o' + c_octal_match.group(2)
  try:
    return int(text, 0)
  except ValueError:
    raise ValueError('Couldn\'t parse integer: %s' % orig_text)


def ParseFloat(text):
  """Parse a floating point number.

  Args:
    text: Text to parse.

  Returns:
    The number parsed.

  Raises:
    ValueError: If a floating point number couldn't be parsed.
  """
  if _FLOAT_OCTAL_PREFIX.match(text):
    raise ValueError('Invalid octal float: %s' % text)
  try:
    # Assume Python compatible syntax.
    return float(text)
  except ValueError:
    # Check alternative spellings.
    if _FLOAT_INFINITY.match(text):
      if text[0] == '-':
        return float('-inf')
      else:
        return float('inf')
    elif _FLOAT_NAN.match(text):
      return float('nan')
    else:
      # assume '1.0f' format
      try:
        return float(text.rstrip('fF'))
      except ValueError:
        raise ValueError("Couldn't parse float: %s" % text)


def ParseBool(text):
  """Parse a boolean value.

  Args:
    text: Text to parse.

  Returns:
    Boolean values parsed

  Raises:
    ValueError: If text is not a valid boolean.
  """
  if text in ('true', 't', '1', 'True'):
    return True
  elif text in ('false', 'f', '0', 'False'):
    return False
  else:
    raise ValueError('Expected "true" or "false".')


def ParseEnum(field, value):
  """Parse an enum value.

  The value can be specified by a number (the enum value), or by
  a string literal (the enum name).

  Args:
    field: Enum field descriptor.
    value: String value.

  Returns:
    Enum value number.

  Raises:
    ValueError: If the enum value could not be parsed.
  """
  enum_descriptor = field.enum_type
  try:
    number = int(value, 0)
  except ValueError:
    # Identifier.
    enum_value = enum_descriptor.values_by_name.get(value, None)
    if enum_value is None:
      raise ValueError('Enum type "%s" has no value named %s.' %
                       (enum_descriptor.full_name, value))
  else:
    if not field.enum_type.is_closed:
      return number
    enum_value = enum_descriptor.values_by_number.get(number, None)
    if enum_value is None:
      raise ValueError('Enum type "%s" has no value with number %d.' %
                       (enum_descriptor.full_name, number))
  return enum_value.number