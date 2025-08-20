
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\factortools.py ===
"""Polynomial factorization routines in characteristic zero. """

from sympy.external.gmpy import GROUND_TYPES

from sympy.core.random import _randint

from sympy.polys.galoistools import (
    gf_from_int_poly, gf_to_int_poly,
    gf_lshift, gf_add_mul, gf_mul,
    gf_div, gf_rem,
    gf_gcdex,
    gf_sqf_p,
    gf_factor_sqf, gf_factor)

from sympy.polys.densebasic import (
    dup_LC, dmp_LC, dmp_ground_LC,
    dup_TC,
    dup_convert, dmp_convert,
    dup_degree, dmp_degree,
    dmp_degree_in, dmp_degree_list,
    dmp_from_dict,
    dmp_zero_p,
    dmp_one,
    dmp_nest, dmp_raise,
    dup_strip,
    dmp_ground,
    dup_inflate,
    dmp_exclude, dmp_include,
    dmp_inject, dmp_eject,
    dup_terms_gcd, dmp_terms_gcd)

from sympy.polys.densearith import (
    dup_neg, dmp_neg,
    dup_add, dmp_add,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_sqr,
    dmp_pow,
    dup_div, dmp_div,
    dup_quo, dmp_quo,
    dmp_expand,
    dmp_add_mul,
    dup_sub_mul, dmp_sub_mul,
    dup_lshift,
    dup_max_norm, dmp_max_norm,
    dup_l1_norm,
    dup_mul_ground, dmp_mul_ground,
    dup_quo_ground, dmp_quo_ground)

from sympy.polys.densetools import (
    dup_clear_denoms, dmp_clear_denoms,
    dup_trunc, dmp_ground_trunc,
    dup_content,
    dup_monic, dmp_ground_monic,
    dup_primitive, dmp_ground_primitive,
    dmp_eval_tail,
    dmp_eval_in, dmp_diff_eval_in,
    dup_shift, dmp_shift, dup_mirror)

from sympy.polys.euclidtools import (
    dmp_primitive,
    dup_inner_gcd, dmp_inner_gcd)

from sympy.polys.sqfreetools import (
    dup_sqf_p,
    dup_sqf_norm, dmp_sqf_norm,
    dup_sqf_part, dmp_sqf_part,
    _dup_check_degrees, _dmp_check_degrees,
    )

from sympy.polys.polyutils import _sort_factors
from sympy.polys.polyconfig import query

from sympy.polys.polyerrors import (
    ExtraneousFactors, DomainError, CoercionFailed, EvaluationFailed)

from sympy.utilities import subsets

from math import ceil as _ceil, log as _log, log2 as _log2


if GROUND_TYPES == 'flint':
    from flint import fmpz_poly
else:
    fmpz_poly = None


def dup_trial_division(f, factors, K):
    """
    Determine multiplicities of factors for a univariate polynomial
    using trial division.

    An error will be raised if any factor does not divide ``f``.
    """
    result = []

    for factor in factors:
        k = 0

        while True:
            q, r = dup_div(f, factor, K)

            if not r:
                f, k = q, k + 1
            else:
                break

        if k == 0:
            raise RuntimeError("trial division failed")

        result.append((factor, k))

    return _sort_factors(result)


def dmp_trial_division(f, factors, u, K):
    """
    Determine multiplicities of factors for a multivariate polynomial
    using trial division.

    An error will be raised if any factor does not divide ``f``.
    """
    result = []

    for factor in factors:
        k = 0

        while True:
            q, r = dmp_div(f, factor, u, K)

            if dmp_zero_p(r, u):
                f, k = q, k + 1
            else:
                break

        if k == 0:
            raise RuntimeError("trial division failed")

        result.append((factor, k))

    return _sort_factors(result)


def dup_zz_mignotte_bound(f, K):
    """
    The Knuth-Cohen variant of Mignotte bound for
    univariate polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = x**3 + 14*x**2 + 56*x + 64
    >>> R.dup_zz_mignotte_bound(f)
    152

    By checking ``factor(f)`` we can see that max coeff is 8

    Also consider a case that ``f`` is irreducible for example
    ``f = 2*x**2 + 3*x + 4``. To avoid a bug for these cases, we return the
    bound plus the max coefficient of ``f``

    >>> f = 2*x**2 + 3*x + 4
    >>> R.dup_zz_mignotte_bound(f)
    6

    Lastly, to see the difference between the new and the old Mignotte bound
    consider the irreducible polynomial:

    >>> f = 87*x**7 + 4*x**6 + 80*x**5 + 17*x**4 + 9*x**3 + 12*x**2 + 49*x + 26
    >>> R.dup_zz_mignotte_bound(f)
    744

    The new Mignotte bound is 744 whereas the old one (SymPy 1.5.1) is 1937664.


    References
    ==========

    ..[1] [Abbott13]_

    """
    from sympy.functions.combinatorial.factorials import binomial
    d = dup_degree(f)
    delta = _ceil(d / 2)
    delta2 = _ceil(delta / 2)

    # euclidean-norm
    eucl_norm = K.sqrt( sum( cf**2 for cf in f ) )

    # biggest values of binomial coefficients (p. 538 of reference)
    t1 = binomial(delta - 1, delta2)
    t2 = binomial(delta - 1, delta2 - 1)

    lc = K.abs(dup_LC(f, K))   # leading coefficient
    bound = t1 * eucl_norm + t2 * lc   # (p. 538 of reference)
    bound += dup_max_norm(f, K) # add max coeff for irreducible polys
    bound = _ceil(bound / 2) * 2   # round up to even integer

    return bound

def dmp_zz_mignotte_bound(f, u, K):
    """Mignotte bound for multivariate polynomials in `K[X]`. """
    a = dmp_max_norm(f, u, K)
    b = abs(dmp_ground_LC(f, u, K))
    n = sum(dmp_degree_list(f, u))

    return K.sqrt(K(n + 1))*2**n*a*b


def dup_zz_hensel_step(m, f, g, h, s, t, K):
    """
    One step in Hensel lifting in `Z[x]`.

    Given positive integer `m` and `Z[x]` polynomials `f`, `g`, `h`, `s`
    and `t` such that::

        f = g*h (mod m)
        s*g + t*h = 1 (mod m)

        lc(f) is not a zero divisor (mod m)
        lc(h) = 1

        deg(f) = deg(g) + deg(h)
        deg(s) < deg(h)
        deg(t) < deg(g)

    returns polynomials `G`, `H`, `S` and `T`, such that::

        f = G*H (mod m**2)
        S*G + T*H = 1 (mod m**2)

    References
    ==========

    .. [1] [Gathen99]_

    """
    M = m**2

    e = dup_sub_mul(f, g, h, K)
    e = dup_trunc(e, M, K)

    q, r = dup_div(dup_mul(s, e, K), h, K)

    q = dup_trunc(q, M, K)
    r = dup_trunc(r, M, K)

    u = dup_add(dup_mul(t, e, K), dup_mul(q, g, K), K)
    G = dup_trunc(dup_add(g, u, K), M, K)
    H = dup_trunc(dup_add(h, r, K), M, K)

    u = dup_add(dup_mul(s, G, K), dup_mul(t, H, K), K)
    b = dup_trunc(dup_sub(u, [K.one], K), M, K)

    c, d = dup_div(dup_mul(s, b, K), H, K)

    c = dup_trunc(c, M, K)
    d = dup_trunc(d, M, K)

    u = dup_add(dup_mul(t, b, K), dup_mul(c, G, K), K)
    S = dup_trunc(dup_sub(s, d, K), M, K)
    T = dup_trunc(dup_sub(t, u, K), M, K)

    return G, H, S, T


def dup_zz_hensel_lift(p, f, f_list, l, K):
    r"""
    Multifactor Hensel lifting in `Z[x]`.

    Given a prime `p`, polynomial `f` over `Z[x]` such that `lc(f)`
    is a unit modulo `p`, monic pair-wise coprime polynomials `f_i`
    over `Z[x]` satisfying::

        f = lc(f) f_1 ... f_r (mod p)

    and a positive integer `l`, returns a list of monic polynomials
    `F_1,\ F_2,\ \dots,\ F_r` satisfying::

       f = lc(f) F_1 ... F_r (mod p**l)

       F_i = f_i (mod p), i = 1..r

    References
    ==========

    .. [1] [Gathen99]_

    """
    r = len(f_list)
    lc = dup_LC(f, K)

    if r == 1:
        F = dup_mul_ground(f, K.gcdex(lc, p**l)[0], K)
        return [ dup_trunc(F, p**l, K) ]

    m = p
    k = r // 2
    d = int(_ceil(_log2(l)))

    g = gf_from_int_poly([lc], p)

    for f_i in f_list[:k]:
        g = gf_mul(g, gf_from_int_poly(f_i, p), p, K)

    h = gf_from_int_poly(f_list[k], p)

    for f_i in f_list[k + 1:]:
        h = gf_mul(h, gf_from_int_poly(f_i, p), p, K)

    s, t, _ = gf_gcdex(g, h, p, K)

    g = gf_to_int_poly(g, p)
    h = gf_to_int_poly(h, p)
    s = gf_to_int_poly(s, p)
    t = gf_to_int_poly(t, p)

    for _ in range(1, d + 1):
        (g, h, s, t), m = dup_zz_hensel_step(m, f, g, h, s, t, K), m**2

    return dup_zz_hensel_lift(p, g, f_list[:k], l, K) \
        + dup_zz_hensel_lift(p, h, f_list[k:], l, K)

def _test_pl(fc, q, pl):
    if q > pl // 2:
        q = q - pl
    if not q:
        return True
    return fc % q == 0

def dup_zz_zassenhaus(f, K):
    """Factor primitive square-free polynomials in `Z[x]`. """
    n = dup_degree(f)

    if n == 1:
        return [f]

    from sympy.ntheory import isprime

    fc = f[-1]
    A = dup_max_norm(f, K)
    b = dup_LC(f, K)
    B = int(abs(K.sqrt(K(n + 1))*2**n*A*b))
    C = int((n + 1)**(2*n)*A**(2*n - 1))
    gamma = int(_ceil(2*_log2(C)))
    bound = int(2*gamma*_log(gamma))
    a = []
    # choose a prime number `p` such that `f` be square free in Z_p
    # if there are many factors in Z_p, choose among a few different `p`
    # the one with fewer factors
    for px in range(3, bound + 1):
        if not isprime(px) or b % px == 0:
            continue

        px = K.convert(px)

        F = gf_from_int_poly(f, px)

        if not gf_sqf_p(F, px, K):
            continue
        fsqfx = gf_factor_sqf(F, px, K)[1]
        a.append((px, fsqfx))
        if len(fsqfx) < 15 or len(a) > 4:
            break
    p, fsqf = min(a, key=lambda x: len(x[1]))

    l = int(_ceil(_log(2*B + 1, p)))

    modular = [gf_to_int_poly(ff, p) for ff in fsqf]

    g = dup_zz_hensel_lift(p, f, modular, l, K)

    sorted_T = range(len(g))
    T = set(sorted_T)
    factors, s = [], 1
    pl = p**l

    while 2*s <= len(T):
        for S in subsets(sorted_T, s):
            # lift the constant coefficient of the product `G` of the factors
            # in the subset `S`; if it is does not divide `fc`, `G` does
            # not divide the input polynomial

            if b == 1:
                q = 1
                for i in S:
                    q = q*g[i][-1]
                q = q % pl
                if not _test_pl(fc, q, pl):
                    continue
            else:
                G = [b]
                for i in S:
                    G = dup_mul(G, g[i], K)
                G = dup_trunc(G, pl, K)
                G = dup_primitive(G, K)[1]
                q = G[-1]
                if q and fc % q != 0:
                    continue

            H = [b]
            S = set(S)
            T_S = T - S

            if b == 1:
                G = [b]
                for i in S:
                    G = dup_mul(G, g[i], K)
                G = dup_trunc(G, pl, K)

            for i in T_S:
                H = dup_mul(H, g[i], K)

            H = dup_trunc(H, pl, K)

            G_norm = dup_l1_norm(G, K)
            H_norm = dup_l1_norm(H, K)

            if G_norm*H_norm <= B:
                T = T_S
                sorted_T = [i for i in sorted_T if i not in S]

                G = dup_primitive(G, K)[1]
                f = dup_primitive(H, K)[1]

                factors.append(G)
                b = dup_LC(f, K)

                break
        else:
            s += 1

    return factors + [f]


def dup_zz_irreducible_p(f, K):
    """Test irreducibility using Eisenstein's criterion. """
    lc = dup_LC(f, K)
    tc = dup_TC(f, K)

    e_fc = dup_content(f[1:], K)

    if e_fc:
        from sympy.ntheory import factorint
        e_ff = factorint(int(e_fc))

        for p in e_ff.keys():
            if (lc % p) and (tc % p**2):
                return True


def dup_cyclotomic_p(f, K, irreducible=False):
    """
    Efficiently test if ``f`` is a cyclotomic polynomial.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1
    >>> R.dup_cyclotomic_p(f)
    False

    >>> g = x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1
    >>> R.dup_cyclotomic_p(g)
    True

    References
    ==========

    Bradford, Russell J., and James H. Davenport. "Effective tests for
    cyclotomic polynomials." In International Symposium on Symbolic and
    Algebraic Computation, pp. 244-251. Springer, Berlin, Heidelberg, 1988.

    """
    if K.is_QQ:
        try:
            K0, K = K, K.get_ring()
            f = dup_convert(f, K0, K)
        except CoercionFailed:
            return False
    elif not K.is_ZZ:
        return False

    lc = dup_LC(f, K)
    tc = dup_TC(f, K)

    if lc != 1 or (tc != -1 and tc != 1):
        return False

    if not irreducible:
        coeff, factors = dup_factor_list(f, K)

        if coeff != K.one or factors != [(f, 1)]:
            return False

    n = dup_degree(f)
    g, h = [], []

    for i in range(n, -1, -2):
        g.insert(0, f[i])

    for i in range(n - 1, -1, -2):
        h.insert(0, f[i])

    g = dup_sqr(dup_strip(g), K)
    h = dup_sqr(dup_strip(h), K)

    F = dup_sub(g, dup_lshift(h, 1, K), K)

    if K.is_negative(dup_LC(F, K)):
        F = dup_neg(F, K)

    if F == f:
        return True

    g = dup_mirror(f, K)

    if K.is_negative(dup_LC(g, K)):
        g = dup_neg(g, K)

    if F == g and dup_cyclotomic_p(g, K):
        return True

    G = dup_sqf_part(F, K)

    if dup_sqr(G, K) == F and dup_cyclotomic_p(G, K):
        return True

    return False


def dup_zz_cyclotomic_poly(n, K):
    """Efficiently generate n-th cyclotomic polynomial. """
    from sympy.ntheory import factorint
    h = [K.one, -K.one]

    for p, k in factorint(n).items():
        h = dup_quo(dup_inflate(h, p, K), h, K)
        h = dup_inflate(h, p**(k - 1), K)

    return h


def _dup_cyclotomic_decompose(n, K):
    from sympy.ntheory import factorint

    H = [[K.one, -K.one]]

    for p, k in factorint(n).items():
        Q = [ dup_quo(dup_inflate(h, p, K), h, K) for h in H ]
        H.extend(Q)

        for i in range(1, k):
            Q = [ dup_inflate(q, p, K) for q in Q ]
            H.extend(Q)

    return H


def dup_zz_cyclotomic_factor(f, K):
    """
    Efficiently factor polynomials `x**n - 1` and `x**n + 1` in `Z[x]`.

    Given a univariate polynomial `f` in `Z[x]` returns a list of factors
    of `f`, provided that `f` is in the form `x**n - 1` or `x**n + 1` for
    `n >= 1`. Otherwise returns None.

    Factorization is performed using cyclotomic decomposition of `f`,
    which makes this method much faster that any other direct factorization
    approach (e.g. Zassenhaus's).

    References
    ==========

    .. [1] [Weisstein09]_

    """
    lc_f, tc_f = dup_LC(f, K), dup_TC(f, K)

    if dup_degree(f) <= 0:
        return None

    if lc_f != 1 or tc_f not in [-1, 1]:
        return None

    if any(bool(cf) for cf in f[1:-1]):
        return None

    n = dup_degree(f)
    F = _dup_cyclotomic_decompose(n, K)

    if not K.is_one(tc_f):
        return F
    else:
        H = []

        for h in _dup_cyclotomic_decompose(2*n, K):
            if h not in F:
                H.append(h)

        return H


def dup_zz_factor_sqf(f, K):
    """Factor square-free (non-primitive) polynomials in `Z[x]`. """
    cont, g = dup_primitive(f, K)

    n = dup_degree(g)

    if dup_LC(g, K) < 0:
        cont, g = -cont, dup_neg(g, K)

    if n <= 0:
        return cont, []
    elif n == 1:
        return cont, [g]

    if query('USE_IRREDUCIBLE_IN_FACTOR'):
        if dup_zz_irreducible_p(g, K):
            return cont, [g]

    factors = None

    if query('USE_CYCLOTOMIC_FACTOR'):
        factors = dup_zz_cyclotomic_factor(g, K)

    if factors is None:
        factors = dup_zz_zassenhaus(g, K)

    return cont, _sort_factors(factors, multiple=False)


def dup_zz_factor(f, K):
    """
    Factor (non square-free) polynomials in `Z[x]`.

    Given a univariate polynomial `f` in `Z[x]` computes its complete
    factorization `f_1, ..., f_n` into irreducibles over integers::

                f = content(f) f_1**k_1 ... f_n**k_n

    The factorization is computed by reducing the input polynomial
    into a primitive square-free polynomial and factoring it using
    Zassenhaus algorithm. Trial division is used to recover the
    multiplicities of factors.

    The result is returned as a tuple consisting of::

              (content(f), [(f_1, k_1), ..., (f_n, k_n))

    Examples
    ========

    Consider the polynomial `f = 2*x**4 - 2`::

        >>> from sympy.polys import ring, ZZ
        >>> R, x = ring("x", ZZ)

        >>> R.dup_zz_factor(2*x**4 - 2)
        (2, [(x - 1, 1), (x + 1, 1), (x**2 + 1, 1)])

    In result we got the following factorization::

                 f = 2 (x - 1) (x + 1) (x**2 + 1)

    Note that this is a complete factorization over integers,
    however over Gaussian integers we can factor the last term.

    By default, polynomials `x**n - 1` and `x**n + 1` are factored
    using cyclotomic decomposition to speedup computations. To
    disable this behaviour set cyclotomic=False.

    References
    ==========

    .. [1] [Gathen99]_

    """
    if GROUND_TYPES == 'flint':
        f_flint = fmpz_poly(f[::-1])
        cont, factors = f_flint.factor()
        factors = [(fac.coeffs()[::-1], exp) for fac, exp in factors]
        return cont, _sort_factors(factors)

    cont, g = dup_primitive(f, K)

    n = dup_degree(g)

    if dup_LC(g, K) < 0:
        cont, g = -cont, dup_neg(g, K)

    if n <= 0:
        return cont, []
    elif n == 1:
        return cont, [(g, 1)]

    if query('USE_IRREDUCIBLE_IN_FACTOR'):
        if dup_zz_irreducible_p(g, K):
            return cont, [(g, 1)]

    g = dup_sqf_part(g, K)
    H = None

    if query('USE_CYCLOTOMIC_FACTOR'):
        H = dup_zz_cyclotomic_factor(g, K)

    if H is None:
        H = dup_zz_zassenhaus(g, K)

    factors = dup_trial_division(f, H, K)

    _dup_check_degrees(f, factors)

    return cont, factors


def dmp_zz_wang_non_divisors(E, cs, ct, K):
    """Wang/EEZ: Compute a set of valid divisors.  """
    result = [ cs*ct ]

    for q in E:
        q = abs(q)

        for r in reversed(result):
            while r != 1:
                r = K.gcd(r, q)
                q = q // r

            if K.is_one(q):
                return None

        result.append(q)

    return result[1:]


def dmp_zz_wang_test_points(f, T, ct, A, u, K):
    """Wang/EEZ: Test evaluation points for suitability. """
    if not dmp_eval_tail(dmp_LC(f, K), A, u - 1, K):
        raise EvaluationFailed('no luck')

    g = dmp_eval_tail(f, A, u, K)

    if not dup_sqf_p(g, K):
        raise EvaluationFailed('no luck')

    c, h = dup_primitive(g, K)

    if K.is_negative(dup_LC(h, K)):
        c, h = -c, dup_neg(h, K)

    v = u - 1

    E = [ dmp_eval_tail(t, A, v, K) for t, _ in T ]
    D = dmp_zz_wang_non_divisors(E, c, ct, K)

    if D is not None:
        return c, h, E
    else:
        raise EvaluationFailed('no luck')


def dmp_zz_wang_lead_coeffs(f, T, cs, E, H, A, u, K):
    """Wang/EEZ: Compute correct leading coefficients. """
    C, J, v = [], [0]*len(E), u - 1

    for h in H:
        c = dmp_one(v, K)
        d = dup_LC(h, K)*cs

        for i in reversed(range(len(E))):
            k, e, (t, _) = 0, E[i], T[i]

            while not (d % e):
                d, k = d//e, k + 1

            if k != 0:
                c, J[i] = dmp_mul(c, dmp_pow(t, k, v, K), v, K), 1

        C.append(c)

    if not all(J):
        raise ExtraneousFactors  # pragma: no cover

    CC, HH = [], []

    for c, h in zip(C, H):
        d = dmp_eval_tail(c, A, v, K)
        lc = dup_LC(h, K)

        if K.is_one(cs):
            cc = lc//d
        else:
            g = K.gcd(lc, d)
            d, cc = d//g, lc//g
            h, cs = dup_mul_ground(h, d, K), cs//d

        c = dmp_mul_ground(c, cc, v, K)

        CC.append(c)
        HH.append(h)

    if K.is_one(cs):
        return f, HH, CC

    CCC, HHH = [], []

    for c, h in zip(CC, HH):
        CCC.append(dmp_mul_ground(c, cs, v, K))
        HHH.append(dmp_mul_ground(h, cs, 0, K))

    f = dmp_mul_ground(f, cs**(len(H) - 1), u, K)

    return f, HHH, CCC


def dup_zz_diophantine(F, m, p, K):
    """Wang/EEZ: Solve univariate Diophantine equations. """
    if len(F) == 2:
        a, b = F

        f = gf_from_int_poly(a, p)
        g = gf_from_int_poly(b, p)

        s, t, G = gf_gcdex(g, f, p, K)

        s = gf_lshift(s, m, K)
        t = gf_lshift(t, m, K)

        q, s = gf_div(s, f, p, K)

        t = gf_add_mul(t, q, g, p, K)

        s = gf_to_int_poly(s, p)
        t = gf_to_int_poly(t, p)

        result = [s, t]
    else:
        G = [F[-1]]

        for f in reversed(F[1:-1]):
            G.insert(0, dup_mul(f, G[0], K))

        S, T = [], [[1]]

        for f, g in zip(F, G):
            t, s = dmp_zz_diophantine([g, f], T[-1], [], 0, p, 1, K)
            T.append(t)
            S.append(s)

        result, S = [], S + [T[-1]]

        for s, f in zip(S, F):
            s = gf_from_int_poly(s, p)
            f = gf_from_int_poly(f, p)

            r = gf_rem(gf_lshift(s, m, K), f, p, K)
            s = gf_to_int_poly(r, p)

            result.append(s)

    return result


def dmp_zz_diophantine(F, c, A, d, p, u, K):
    """Wang/EEZ: Solve multivariate Diophantine equations. """
    if not A:
        S = [ [] for _ in F ]
        n = dup_degree(c)

        for i, coeff in enumerate(c):
            if not coeff:
                continue

            T = dup_zz_diophantine(F, n - i, p, K)

            for j, (s, t) in enumerate(zip(S, T)):
                t = dup_mul_ground(t, coeff, K)
                S[j] = dup_trunc(dup_add(s, t, K), p, K)
    else:
        n = len(A)
        e = dmp_expand(F, u, K)

        a, A = A[-1], A[:-1]
        B, G = [], []

        for f in F:
            B.append(dmp_quo(e, f, u, K))
            G.append(dmp_eval_in(f, a, n, u, K))

        C = dmp_eval_in(c, a, n, u, K)

        v = u - 1

        S = dmp_zz_diophantine(G, C, A, d, p, v, K)
        S = [ dmp_raise(s, 1, v, K) for s in S ]

        for s, b in zip(S, B):
            c = dmp_sub_mul(c, s, b, u, K)

        c = dmp_ground_trunc(c, p, u, K)

        m = dmp_nest([K.one, -a], n, K)
        M = dmp_one(n, K)

        for k in range(0, d):
            if dmp_zero_p(c, u):
                break

            M = dmp_mul(M, m, u, K)
            C = dmp_diff_eval_in(c, k + 1, a, n, u, K)

            if not dmp_zero_p(C, v):
                C = dmp_quo_ground(C, K.factorial(K(k) + 1), v, K)
                T = dmp_zz_diophantine(G, C, A, d, p, v, K)

                for i, t in enumerate(T):
                    T[i] = dmp_mul(dmp_raise(t, 1, v, K), M, u, K)

                for i, (s, t) in enumerate(zip(S, T)):
                    S[i] = dmp_add(s, t, u, K)

                for t, b in zip(T, B):
                    c = dmp_sub_mul(c, t, b, u, K)

                c = dmp_ground_trunc(c, p, u, K)

        S = [ dmp_ground_trunc(s, p, u, K) for s in S ]

    return S


def dmp_zz_wang_hensel_lifting(f, H, LC, A, p, u, K):
    """Wang/EEZ: Parallel Hensel lifting algorithm. """
    S, n, v = [f], len(A), u - 1

    H = list(H)

    for i, a in enumerate(reversed(A[1:])):
        s = dmp_eval_in(S[0], a, n - i, u - i, K)
        S.insert(0, dmp_ground_trunc(s, p, v - i, K))

    d = max(dmp_degree_list(f, u)[1:])

    for j, s, a in zip(range(2, n + 2), S, A):
        G, w = list(H), j - 1

        I, J = A[:j - 2], A[j - 1:]

        for i, (h, lc) in enumerate(zip(H, LC)):
            lc = dmp_ground_trunc(dmp_eval_tail(lc, J, v, K), p, w - 1, K)
            H[i] = [lc] + dmp_raise(h[1:], 1, w - 1, K)

        m = dmp_nest([K.one, -a], w, K)
        M = dmp_one(w, K)

        c = dmp_sub(s, dmp_expand(H, w, K), w, K)

        dj = dmp_degree_in(s, w, w)

        for k in range(0, dj):
            if dmp_zero_p(c, w):
                break

            M = dmp_mul(M, m, w, K)
            C = dmp_diff_eval_in(c, k + 1, a, w, w, K)

            if not dmp_zero_p(C, w - 1):
                C = dmp_quo_ground(C, K.factorial(K(k) + 1), w - 1, K)
                T = dmp_zz_diophantine(G, C, I, d, p, w - 1, K)

                for i, (h, t) in enumerate(zip(H, T)):
                    h = dmp_add_mul(h, dmp_raise(t, 1, w - 1, K), M, w, K)
                    H[i] = dmp_ground_trunc(h, p, w, K)

                h = dmp_sub(s, dmp_expand(H, w, K), w, K)
                c = dmp_ground_trunc(h, p, w, K)

    if dmp_expand(H, u, K) != f:
        raise ExtraneousFactors  # pragma: no cover
    else:
        return H


def dmp_zz_wang(f, u, K, mod=None, seed=None):
    r"""
    Factor primitive square-free polynomials in `Z[X]`.

    Given a multivariate polynomial `f` in `Z[x_1,...,x_n]`, which is
    primitive and square-free in `x_1`, computes factorization of `f` into
    irreducibles over integers.

    The procedure is based on Wang's Enhanced Extended Zassenhaus
    algorithm. The algorithm works by viewing `f` as a univariate polynomial
    in `Z[x_2,...,x_n][x_1]`, for which an evaluation mapping is computed::

                      x_2 -> a_2, ..., x_n -> a_n

    where `a_i`, for `i = 2, \dots, n`, are carefully chosen integers.  The
    mapping is used to transform `f` into a univariate polynomial in `Z[x_1]`,
    which can be factored efficiently using Zassenhaus algorithm. The last
    step is to lift univariate factors to obtain true multivariate
    factors. For this purpose a parallel Hensel lifting procedure is used.

    The parameter ``seed`` is passed to _randint and can be used to seed randint
    (when an integer) or (for testing purposes) can be a sequence of numbers.

    References
    ==========

    .. [1] [Wang78]_
    .. [2] [Geddes92]_

    """
    from sympy.ntheory import nextprime

    randint = _randint(seed)

    ct, T = dmp_zz_factor(dmp_LC(f, K), u - 1, K)

    b = dmp_zz_mignotte_bound(f, u, K)
    p = K(nextprime(b))

    if mod is None:
        if u == 1:
            mod = 2
        else:
            mod = 1

    history, configs, A, r = set(), [], [K.zero]*u, None

    try:
        cs, s, E = dmp_zz_wang_test_points(f, T, ct, A, u, K)

        _, H = dup_zz_factor_sqf(s, K)

        r = len(H)

        if r == 1:
            return [f]

        configs = [(s, cs, E, H, A)]
    except EvaluationFailed:
        pass

    eez_num_configs = query('EEZ_NUMBER_OF_CONFIGS')
    eez_num_tries = query('EEZ_NUMBER_OF_TRIES')
    eez_mod_step = query('EEZ_MODULUS_STEP')

    while len(configs) < eez_num_configs:
        for _ in range(eez_num_tries):
            A = [ K(randint(-mod, mod)) for _ in range(u) ]

            if tuple(A) not in history:
                history.add(tuple(A))
            else:
                continue

            try:
                cs, s, E = dmp_zz_wang_test_points(f, T, ct, A, u, K)
            except EvaluationFailed:
                continue

            _, H = dup_zz_factor_sqf(s, K)

            rr = len(H)

            if r is not None:
                if rr != r:  # pragma: no cover
                    if rr < r:
                        configs, r = [], rr
                    else:
                        continue
            else:
                r = rr

            if r == 1:
                return [f]

            configs.append((s, cs, E, H, A))

            if len(configs) == eez_num_configs:
                break
        else:
            mod += eez_mod_step

    s_norm, s_arg, i = None, 0, 0

    for s, _, _, _, _ in configs:
        _s_norm = dup_max_norm(s, K)

        if s_norm is not None:
            if _s_norm < s_norm:
                s_norm = _s_norm
                s_arg = i
        else:
            s_norm = _s_norm

        i += 1

    _, cs, E, H, A = configs[s_arg]
    orig_f = f

    try:
        f, H, LC = dmp_zz_wang_lead_coeffs(f, T, cs, E, H, A, u, K)
        factors = dmp_zz_wang_hensel_lifting(f, H, LC, A, p, u, K)
    except ExtraneousFactors:  # pragma: no cover
        if query('EEZ_RESTART_IF_NEEDED'):
            return dmp_zz_wang(orig_f, u, K, mod + 1)
        else:
            raise ExtraneousFactors(
                "we need to restart algorithm with better parameters")

    result = []

    for f in factors:
        _, f = dmp_ground_primitive(f, u, K)

        if K.is_negative(dmp_ground_LC(f, u, K)):
            f = dmp_neg(f, u, K)

        result.append(f)

    return result


def dmp_zz_factor(f, u, K):
    r"""
    Factor (non square-free) polynomials in `Z[X]`.

    Given a multivariate polynomial `f` in `Z[x]` computes its complete
    factorization `f_1, \dots, f_n` into irreducibles over integers::

                 f = content(f) f_1**k_1 ... f_n**k_n

    The factorization is computed by reducing the input polynomial
    into a primitive square-free polynomial and factoring it using
    Enhanced Extended Zassenhaus (EEZ) algorithm. Trial division
    is used to recover the multiplicities of factors.

    The result is returned as a tuple consisting of::

             (content(f), [(f_1, k_1), ..., (f_n, k_n))

    Consider polynomial `f = 2*(x**2 - y**2)`::

        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> R.dmp_zz_factor(2*x**2 - 2*y**2)
        (2, [(x - y, 1), (x + y, 1)])

    In result we got the following factorization::

                    f = 2 (x - y) (x + y)

    References
    ==========

    .. [1] [Gathen99]_

    """
    if not u:
        return dup_zz_factor(f, K)

    if dmp_zero_p(f, u):
        return K.zero, []

    cont, g = dmp_ground_primitive(f, u, K)

    if dmp_ground_LC(g, u, K) < 0:
        cont, g = -cont, dmp_neg(g, u, K)

    if all(d <= 0 for d in dmp_degree_list(g, u)):
        return cont, []

    G, g = dmp_primitive(g, u, K)

    factors = []

    if dmp_degree(g, u) > 0:
        g = dmp_sqf_part(g, u, K)
        H = dmp_zz_wang(g, u, K)
        factors = dmp_trial_division(f, H, u, K)

    for g, k in dmp_zz_factor(G, u - 1, K)[1]:
        factors.insert(0, ([g], k))

    _dmp_check_degrees(f, u, factors)

    return cont, _sort_factors(factors)


def dup_qq_i_factor(f, K0):
    """Factor univariate polynomials into irreducibles in `QQ_I[x]`. """
    # Factor in QQ<I>
    K1 = K0.as_AlgebraicField()
    f = dup_convert(f, K0, K1)
    coeff, factors = dup_factor_list(f, K1)
    factors = [(dup_convert(fac, K1, K0), i) for fac, i in factors]
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dup_zz_i_factor(f, K0):
    """Factor univariate polynomials into irreducibles in `ZZ_I[x]`. """
    # First factor in QQ_I
    K1 = K0.get_field()
    f = dup_convert(f, K0, K1)
    coeff, factors = dup_qq_i_factor(f, K1)

    new_factors = []
    for fac, i in factors:
        # Extract content
        fac_denom, fac_num = dup_clear_denoms(fac, K1)
        fac_num_ZZ_I = dup_convert(fac_num, K1, K0)
        content, fac_prim = dmp_ground_primitive(fac_num_ZZ_I, 0, K0)

        coeff = (coeff * content ** i) // fac_denom ** i
        new_factors.append((fac_prim, i))

    factors = new_factors
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dmp_qq_i_factor(f, u, K0):
    """Factor multivariate polynomials into irreducibles in `QQ_I[X]`. """
    # Factor in QQ<I>
    K1 = K0.as_AlgebraicField()
    f = dmp_convert(f, u, K0, K1)
    coeff, factors = dmp_factor_list(f, u, K1)
    factors = [(dmp_convert(fac, u, K1, K0), i) for fac, i in factors]
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dmp_zz_i_factor(f, u, K0):
    """Factor multivariate polynomials into irreducibles in `ZZ_I[X]`. """
    # First factor in QQ_I
    K1 = K0.get_field()
    f = dmp_convert(f, u, K0, K1)
    coeff, factors = dmp_qq_i_factor(f, u, K1)

    new_factors = []
    for fac, i in factors:
        # Extract content
        fac_denom, fac_num = dmp_clear_denoms(fac, u, K1)
        fac_num_ZZ_I = dmp_convert(fac_num, u, K1, K0)
        content, fac_prim = dmp_ground_primitive(fac_num_ZZ_I, u, K0)

        coeff = (coeff * content ** i) // fac_denom ** i
        new_factors.append((fac_prim, i))

    factors = new_factors
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dup_ext_factor(f, K):
    r"""Factor univariate polynomials over algebraic number fields.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Examples
    ========

    First define the algebraic number field `K = \mathbb{Q}(\sqrt{2})`:

    >>> from sympy import QQ, sqrt
    >>> from sympy.polys.factortools import dup_ext_factor
    >>> K = QQ.algebraic_field(sqrt(2))

    We can now factorise the polynomial `x^2 - 2` over `K`:

    >>> p = [K(1), K(0), K(-2)] # x^2 - 2
    >>> p1 = [K(1), -K.unit]    # x - sqrt(2)
    >>> p2 = [K(1), +K.unit]    # x + sqrt(2)
    >>> dup_ext_factor(p, K) == (K.one, [(p1, 1), (p2, 1)])
    True

    Usually this would be done at a higher level:

    >>> from sympy import factor
    >>> from sympy.abc import x
    >>> factor(x**2 - 2, extension=sqrt(2))
    (x - sqrt(2))*(x + sqrt(2))

    Explanation
    ===========

    Uses Trager's algorithm. In particular this function is algorithm
    ``alg_factor`` from [Trager76]_.

    If `f` is a polynomial in `k(a)[x]` then its norm `g(x)` is a polynomial in
    `k[x]`. If `g(x)` is square-free and has irreducible factors `g_1(x)`,
    `g_2(x)`, `\cdots` then the irreducible factors of `f` in `k(a)[x]` are
    given by `f_i(x) = \gcd(f(x), g_i(x))` where the GCD is computed in
    `k(a)[x]`.

    The first step in Trager's algorithm is to find an integer shift `s` so
    that `f(x-sa)` has square-free norm. Then the norm is factorized in `k[x]`
    and the GCD of (shifted) `f` with each factor gives the shifted factors of
    `f`. At the end the shift is undone to recover the unshifted factors of `f`
    in `k(a)[x]`.

    The algorithm reduces the problem of factorization in `k(a)[x]` to
    factorization in `k[x]` with the main additional steps being to compute the
    norm (a resultant calculation in `k[x,y]`) and some polynomial GCDs in
    `k(a)[x]`.

    In practice in SymPy the base field `k` will be the rationals :ref:`QQ` and
    this function factorizes a polynomial with coefficients in an algebraic
    number field  like `\mathbb{Q}(\sqrt{2})`.

    See Also
    ========

    dmp_ext_factor:
        Analogous function for multivariate polynomials over ``k(a)``.
    dup_sqf_norm:
        Subroutine ``sqfr_norm`` also from [Trager76]_.
    sympy.polys.polytools.factor:
        The high-level function that ultimately uses this function as needed.
    """
    n, lc = dup_degree(f), dup_LC(f, K)

    f = dup_monic(f, K)

    if n <= 0:
        return lc, []
    if n == 1:
        return lc, [(f, 1)]

    f, F = dup_sqf_part(f, K), f
    s, g, r = dup_sqf_norm(f, K)

    factors = dup_factor_list_include(r, K.dom)

    if len(factors) == 1:
        return lc, [(f, n//dup_degree(f))]

    H = s*K.unit

    for i, (factor, _) in enumerate(factors):
        h = dup_convert(factor, K.dom, K)
        h, _, g = dup_inner_gcd(h, g, K)
        h = dup_shift(h, H, K)
        factors[i] = h

    factors = dup_trial_division(F, factors, K)

    _dup_check_degrees(F, factors)

    return lc, factors


def dmp_ext_factor(f, u, K):
    r"""Factor multivariate polynomials over algebraic number fields.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Examples
    ========

    First define the algebraic number field `K = \mathbb{Q}(\sqrt{2})`:

    >>> from sympy import QQ, sqrt
    >>> from sympy.polys.factortools import dmp_ext_factor
    >>> K = QQ.algebraic_field(sqrt(2))

    We can now factorise the polynomial `x^2 y^2 - 2` over `K`:

    >>> p = [[K(1),K(0),K(0)], [], [K(-2)]] # x**2*y**2 - 2
    >>> p1 = [[K(1),K(0)], [-K.unit]]       # x*y - sqrt(2)
    >>> p2 = [[K(1),K(0)], [+K.unit]]       # x*y + sqrt(2)
    >>> dmp_ext_factor(p, 1, K) == (K.one, [(p1, 1), (p2, 1)])
    True

    Usually this would be done at a higher level:

    >>> from sympy import factor
    >>> from sympy.abc import x, y
    >>> factor(x**2*y**2 - 2, extension=sqrt(2))
    (x*y - sqrt(2))*(x*y + sqrt(2))

    Explanation
    ===========

    This is Trager's algorithm for multivariate polynomials. In particular this
    function is algorithm ``alg_factor`` from [Trager76]_.

    See :func:`dup_ext_factor` for explanation.

    See Also
    ========

    dup_ext_factor:
        Analogous function for univariate polynomials over ``k(a)``.
    dmp_sqf_norm:
        Multivariate version of subroutine ``sqfr_norm`` also from [Trager76]_.
    sympy.polys.polytools.factor:
        The high-level function that ultimately uses this function as needed.
    """
    if not u:
        return dup_ext_factor(f, K)

    lc = dmp_ground_LC(f, u, K)
    f = dmp_ground_monic(f, u, K)

    if all(d <= 0 for d in dmp_degree_list(f, u)):
        return lc, []

    f, F = dmp_sqf_part(f, u, K), f
    s, g, r = dmp_sqf_norm(f, u, K)

    factors = dmp_factor_list_include(r, u, K.dom)

    if len(factors) == 1:
        factors = [f]
    else:
        for i, (factor, _) in enumerate(factors):
            h = dmp_convert(factor, u, K.dom, K)
            h, _, g = dmp_inner_gcd(h, g, u, K)
            a = [si*K.unit for si in s]
            h = dmp_shift(h, a, u, K)
            factors[i] = h

    result = dmp_trial_division(F, factors, u, K)

    _dmp_check_degrees(F, u, result)

    return lc, result


def dup_gf_factor(f, K):
    """Factor univariate polynomials over finite fields. """
    f = dup_convert(f, K, K.dom)

    coeff, factors = gf_factor(f, K.mod, K.dom)

    for i, (f, k) in enumerate(factors):
        factors[i] = (dup_convert(f, K.dom, K), k)

    return K.convert(coeff, K.dom), factors


def dmp_gf_factor(f, u, K):
    """Factor multivariate polynomials over finite fields. """
    raise NotImplementedError('multivariate polynomials over finite fields')


def dup_factor_list(f, K0):
    """Factor univariate polynomials into irreducibles in `K[x]`. """
    j, f = dup_terms_gcd(f, K0)
    cont, f = dup_primitive(f, K0)

    if K0.is_FiniteField:
        coeff, factors = dup_gf_factor(f, K0)
    elif K0.is_Algebraic:
        coeff, factors = dup_ext_factor(f, K0)
    elif K0.is_GaussianRing:
        coeff, factors = dup_zz_i_factor(f, K0)
    elif K0.is_GaussianField:
        coeff, factors = dup_qq_i_factor(f, K0)
    else:
        if not K0.is_Exact:
            K0_inexact, K0 = K0, K0.get_exact()
            f = dup_convert(f, K0_inexact, K0)
        else:
            K0_inexact = None

        if K0.is_Field:
            K = K0.get_ring()

            denom, f = dup_clear_denoms(f, K0, K)
            f = dup_convert(f, K0, K)
        else:
            K = K0

        if K.is_ZZ:
            coeff, factors = dup_zz_factor(f, K)
        elif K.is_Poly:
            f, u = dmp_inject(f, 0, K)

            coeff, factors = dmp_factor_list(f, u, K.dom)

            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_eject(f, u, K), k)

            coeff = K.convert(coeff, K.dom)
        else:  # pragma: no cover
            raise DomainError('factorization not supported over %s' % K0)

        if K0.is_Field:
            for i, (f, k) in enumerate(factors):
                factors[i] = (dup_convert(f, K, K0), k)

            coeff = K0.convert(coeff, K)
            coeff = K0.quo(coeff, denom)

            if K0_inexact:
                for i, (f, k) in enumerate(factors):
                    max_norm = dup_max_norm(f, K0)
                    f = dup_quo_ground(f, max_norm, K0)
                    f = dup_convert(f, K0, K0_inexact)
                    factors[i] = (f, k)
                    coeff = K0.mul(coeff, K0.pow(max_norm, k))

                coeff = K0_inexact.convert(coeff, K0)
                K0 = K0_inexact

    if j:
        factors.insert(0, ([K0.one, K0.zero], j))

    return coeff*cont, _sort_factors(factors)


def dup_factor_list_include(f, K):
    """Factor univariate polynomials into irreducibles in `K[x]`. """
    coeff, factors = dup_factor_list(f, K)

    if not factors:
        return [(dup_strip([coeff]), 1)]
    else:
        g = dup_mul_ground(factors[0][0], coeff, K)
        return [(g, factors[0][1])] + factors[1:]


def dmp_factor_list(f, u, K0):
    """Factor multivariate polynomials into irreducibles in `K[X]`. """
    if not u:
        return dup_factor_list(f, K0)

    J, f = dmp_terms_gcd(f, u, K0)
    cont, f = dmp_ground_primitive(f, u, K0)

    if K0.is_FiniteField:  # pragma: no cover
        coeff, factors = dmp_gf_factor(f, u, K0)
    elif K0.is_Algebraic:
        coeff, factors = dmp_ext_factor(f, u, K0)
    elif K0.is_GaussianRing:
        coeff, factors = dmp_zz_i_factor(f, u, K0)
    elif K0.is_GaussianField:
        coeff, factors = dmp_qq_i_factor(f, u, K0)
    else:
        if not K0.is_Exact:
            K0_inexact, K0 = K0, K0.get_exact()
            f = dmp_convert(f, u, K0_inexact, K0)
        else:
            K0_inexact = None

        if K0.is_Field:
            K = K0.get_ring()

            denom, f = dmp_clear_denoms(f, u, K0, K)
            f = dmp_convert(f, u, K0, K)
        else:
            K = K0

        if K.is_ZZ:
            levels, f, v = dmp_exclude(f, u, K)
            coeff, factors = dmp_zz_factor(f, v, K)

            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_include(f, levels, v, K), k)
        elif K.is_Poly:
            f, v = dmp_inject(f, u, K)

            coeff, factors = dmp_factor_list(f, v, K.dom)

            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_eject(f, v, K), k)

            coeff = K.convert(coeff, K.dom)
        else:  # pragma: no cover
            raise DomainError('factorization not supported over %s' % K0)

        if K0.is_Field:
            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_convert(f, u, K, K0), k)

            coeff = K0.convert(coeff, K)
            coeff = K0.quo(coeff, denom)

            if K0_inexact:
                for i, (f, k) in enumerate(factors):
                    max_norm = dmp_max_norm(f, u, K0)
                    f = dmp_quo_ground(f, max_norm, u, K0)
                    f = dmp_convert(f, u, K0, K0_inexact)
                    factors[i] = (f, k)
                    coeff = K0.mul(coeff, K0.pow(max_norm, k))

                coeff = K0_inexact.convert(coeff, K0)
                K0 = K0_inexact

    for i, j in enumerate(reversed(J)):
        if not j:
            continue

        term = {(0,)*(u - i) + (1,) + (0,)*i: K0.one}
        factors.insert(0, (dmp_from_dict(term, u, K0), j))

    return coeff*cont, _sort_factors(factors)


def dmp_factor_list_include(f, u, K):
    """Factor multivariate polynomials into irreducibles in `K[X]`. """
    if not u:
        return dup_factor_list_include(f, K)

    coeff, factors = dmp_factor_list(f, u, K)

    if not factors:
        return [(dmp_ground(coeff, u), 1)]
    else:
        g = dmp_mul_ground(factors[0][0], coeff, u, K)
        return [(g, factors[0][1])] + factors[1:]


def dup_irreducible_p(f, K):
    """
    Returns ``True`` if a univariate polynomial ``f`` has no factors
    over its domain.
    """
    return dmp_irreducible_p(f, 0, K)


def dmp_irreducible_p(f, u, K):
    """
    Returns ``True`` if a multivariate polynomial ``f`` has no factors
    over its domain.
    """
    _, factors = dmp_factor_list(f, u, K)

    if not factors:
        return True
    elif len(factors) > 1:
        return False
    else:
        _, k = factors[0]
        return k == 1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\soupsieve\css_match.py ===
"""CSS matcher."""
from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4
from typing import Iterator, Iterable, Any, Callable, Sequence, Any, cast  # noqa: F401, F811

# Empty tag pattern (whitespace okay)
RE_NOT_EMPTY = re.compile('[^ \t\r\n\f]')

RE_NOT_WS = re.compile('[^ \t\r\n\f]+')

# Relationships
REL_PARENT = ' '
REL_CLOSE_PARENT = '>'
REL_SIBLING = '~'
REL_CLOSE_SIBLING = '+'

# Relationships for :has() (forward looking)
REL_HAS_PARENT = ': '
REL_HAS_CLOSE_PARENT = ':>'
REL_HAS_SIBLING = ':~'
REL_HAS_CLOSE_SIBLING = ':+'

NS_XHTML = 'http://www.w3.org/1999/xhtml'
NS_XML = 'http://www.w3.org/XML/1998/namespace'

DIR_FLAGS = ct.SEL_DIR_LTR | ct.SEL_DIR_RTL
RANGES = ct.SEL_IN_RANGE | ct.SEL_OUT_OF_RANGE

DIR_MAP = {
    'ltr': ct.SEL_DIR_LTR,
    'rtl': ct.SEL_DIR_RTL,
    'auto': 0
}

RE_NUM = re.compile(r"^(?P<value>-?(?:[0-9]{1,}(\.[0-9]+)?|\.[0-9]+))$")
RE_TIME = re.compile(r'^(?P<hour>[0-9]{2}):(?P<minutes>[0-9]{2})$')
RE_MONTH = re.compile(r'^(?P<year>[0-9]{4,})-(?P<month>[0-9]{2})$')
RE_WEEK = re.compile(r'^(?P<year>[0-9]{4,})-W(?P<week>[0-9]{2})$')
RE_DATE = re.compile(r'^(?P<year>[0-9]{4,})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})$')
RE_DATETIME = re.compile(
    r'^(?P<year>[0-9]{4,})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})T(?P<hour>[0-9]{2}):(?P<minutes>[0-9]{2})$'
)
RE_WILD_STRIP = re.compile(r'(?:(?:-\*-)(?:\*(?:-|$))*|-\*$)')

MONTHS_30 = (4, 6, 9, 11)  # April, June, September, and November
FEB = 2
SHORT_MONTH = 30
LONG_MONTH = 31
FEB_MONTH = 28
FEB_LEAP_MONTH = 29
DAYS_IN_WEEK = 7


class _FakeParent:
    """
    Fake parent class.

    When we have a fragment with no `BeautifulSoup` document object,
    we can't evaluate `nth` selectors properly.  Create a temporary
    fake parent so we can traverse the root element as a child.
    """

    def __init__(self, element: bs4.Tag) -> None:
        """Initialize."""

        self.contents = [element]

    def __len__(self) -> int:
        """Length."""

        return len(self.contents)


class _DocumentNav:
    """Navigate a Beautiful Soup document."""

    @classmethod
    def assert_valid_input(cls, tag: Any) -> None:
        """Check if valid input tag or document."""

        # Fail on unexpected types.
        if not cls.is_tag(tag):
            raise TypeError(f"Expected a BeautifulSoup 'Tag', but instead received type {type(tag)}")

    @staticmethod
    def is_doc(obj: bs4.element.PageElement | None) -> bool:
        """Is `BeautifulSoup` object."""
        return isinstance(obj, bs4.BeautifulSoup)

    @staticmethod
    def is_tag(obj: bs4.element.PageElement | None) -> bool:
        """Is tag."""
        return isinstance(obj, bs4.Tag)

    @staticmethod
    def is_declaration(obj: bs4.element.PageElement | None) -> bool:  # pragma: no cover
        """Is declaration."""
        return isinstance(obj, bs4.Declaration)

    @staticmethod
    def is_cdata(obj: bs4.element.PageElement | None) -> bool:
        """Is CDATA."""
        return isinstance(obj, bs4.CData)

    @staticmethod
    def is_processing_instruction(obj: bs4.element.PageElement | None) -> bool:  # pragma: no cover
        """Is processing instruction."""
        return isinstance(obj, bs4.ProcessingInstruction)

    @staticmethod
    def is_navigable_string(obj: bs4.element.PageElement | None) -> bool:
        """Is navigable string."""
        return isinstance(obj, bs4.element.NavigableString)

    @staticmethod
    def is_special_string(obj: bs4.element.PageElement | None) -> bool:
        """Is special string."""
        return isinstance(obj, (bs4.Comment, bs4.Declaration, bs4.CData, bs4.ProcessingInstruction, bs4.Doctype))

    @classmethod
    def is_content_string(cls, obj: bs4.element.PageElement | None) -> bool:
        """Check if node is content string."""

        return cls.is_navigable_string(obj) and not cls.is_special_string(obj)

    @staticmethod
    def create_fake_parent(el: bs4.Tag) -> _FakeParent:
        """Create fake parent for a given element."""

        return _FakeParent(el)

    @staticmethod
    def is_xml_tree(el: bs4.Tag | None) -> bool:
        """Check if element (or document) is from a XML tree."""

        return el is not None and bool(el._is_xml)

    def is_iframe(self, el: bs4.Tag | None) -> bool:
        """Check if element is an `iframe`."""

        if el is None:  # pragma: no cover
            return False

        return bool(
            ((el.name if self.is_xml_tree(el) else util.lower(el.name)) == 'iframe') and
            self.is_html_tag(el)  # type: ignore[attr-defined]
        )

    def is_root(self, el: bs4.Tag) -> bool:
        """
        Return whether element is a root element.

        We check that the element is the root of the tree (which we have already pre-calculated),
        and we check if it is the root element under an `iframe`.
        """

        root = self.root and self.root is el  # type: ignore[attr-defined]
        if not root:
            parent = self.get_parent(el)
            root = parent is not None and self.is_html and self.is_iframe(parent)  # type: ignore[attr-defined]
        return root

    def get_contents(self, el: bs4.Tag | None, no_iframe: bool = False) -> Iterator[bs4.element.PageElement]:
        """Get contents or contents in reverse."""

        if el is not None:
            if not no_iframe or not self.is_iframe(el):
                yield from el.contents

    def get_tag_children(
        self,
        el: bs4.Tag | None,
        start: int | None = None,
        reverse: bool = False,
        no_iframe: bool = False
    ) -> Iterator[bs4.Tag]:
        """Get tag children."""

        return self.get_children(el, start, reverse, True, no_iframe)  # type: ignore[return-value]

    def get_children(
        self,
        el: bs4.Tag | None,
        start: int | None = None,
        reverse: bool = False,
        tags: bool = False,
        no_iframe: bool = False
    ) -> Iterator[bs4.element.PageElement]:
        """Get children."""

        if el is not None and (not no_iframe or not self.is_iframe(el)):
            last = len(el.contents) - 1
            if start is None:
                index = last if reverse else 0
            else:
                index = start
            end = -1 if reverse else last + 1
            incr = -1 if reverse else 1

            if 0 <= index <= last:
                while index != end:
                    node = el.contents[index]
                    index += incr
                    if not tags or self.is_tag(node):
                        yield node

    def get_tag_descendants(
        self,
        el: bs4.Tag | None,
        no_iframe: bool = False
    ) -> Iterator[bs4.Tag]:
        """Specifically get tag descendants."""

        yield from self.get_descendants(el, tags=True, no_iframe=no_iframe)  # type: ignore[misc]

    def get_descendants(
        self,
        el: bs4.Tag | None,
        tags: bool = False,
        no_iframe: bool = False
    ) -> Iterator[bs4.element.PageElement]:
        """Get descendants."""

        if el is not None and (not no_iframe or not self.is_iframe(el)):
            next_good = None
            for child in el.descendants:

                if next_good is not None:
                    if child is not next_good:
                        continue
                    next_good = None

                if isinstance(child, bs4.Tag):
                    if no_iframe and self.is_iframe(child):
                        if child.next_sibling is not None:
                            next_good = child.next_sibling
                        else:
                            last_child = child  # type: bs4.element.PageElement
                            while isinstance(last_child, bs4.Tag) and last_child.contents:
                                last_child = last_child.contents[-1]
                            next_good = last_child.next_element
                        yield child
                        if next_good is None:
                            break
                        # Coverage isn't seeing this even though it's executed
                        continue  # pragma: no cover
                    yield child

                elif not tags:
                    yield child

    def get_parent(self, el: bs4.Tag | None, no_iframe: bool = False) -> bs4.Tag | None:
        """Get parent."""

        parent = el.parent if el is not None else None
        if no_iframe and parent is not None and self.is_iframe(parent):
            parent = None
        return parent

    @staticmethod
    def get_tag_name(el: bs4.Tag | None) -> str | None:
        """Get tag."""

        return el.name if el is not None else None

    @staticmethod
    def get_prefix_name(el: bs4.Tag) -> str | None:
        """Get prefix."""

        return el.prefix

    @staticmethod
    def get_uri(el: bs4.Tag | None) -> str | None:
        """Get namespace `URI`."""

        return el.namespace if el is not None else None

    @classmethod
    def get_next_tag(cls, el: bs4.Tag) -> bs4.Tag | None:
        """Get next sibling tag."""

        return cls.get_next(el, tags=True)  # type: ignore[return-value]

    @classmethod
    def get_next(cls, el: bs4.Tag, tags: bool = False) -> bs4.element.PageElement | None:
        """Get next sibling tag."""

        sibling = el.next_sibling
        while tags and not isinstance(sibling, bs4.Tag) and sibling is not None:
            sibling = sibling.next_sibling

        if tags and not isinstance(sibling, bs4.Tag):
            sibling = None

        return sibling

    @classmethod
    def get_previous_tag(cls, el: bs4.Tag, tags: bool = True) -> bs4.Tag | None:
        """Get previous sibling tag."""

        return cls.get_previous(el, True)  # type: ignore[return-value]

    @classmethod
    def get_previous(cls, el: bs4.Tag, tags: bool = False) -> bs4.element.PageElement | None:
        """Get previous sibling tag."""

        sibling = el.previous_sibling
        while tags and not isinstance(sibling, bs4.Tag) and sibling is not None:
            sibling = sibling.previous_sibling

        if tags and not isinstance(sibling, bs4.Tag):
            sibling = None

        return sibling

    @staticmethod
    def has_html_ns(el: bs4.Tag | None) -> bool:
        """
        Check if element has an HTML namespace.

        This is a bit different than whether a element is treated as having an HTML namespace,
        like we do in the case of `is_html_tag`.
        """

        ns = getattr(el, 'namespace') if el is not None else None  # noqa: B009
        return bool(ns and ns == NS_XHTML)

    @staticmethod
    def split_namespace(el: bs4.Tag | None, attr_name: str) -> tuple[str | None, str | None]:
        """Return namespace and attribute name without the prefix."""

        if el is None:  # pragma: no cover
            return None, None

        return getattr(attr_name, 'namespace', None), getattr(attr_name, 'name', None)

    @classmethod
    def normalize_value(cls, value: Any) -> str | Sequence[str]:
        """Normalize the value to be a string or list of strings."""

        # Treat `None` as empty string.
        if value is None:
            return ''

        # Pass through strings
        if (isinstance(value, str)):
            return value

        # If it's a byte string, convert it to Unicode, treating it as UTF-8.
        if isinstance(value, bytes):
            return value.decode("utf8")

        # BeautifulSoup supports sequences of attribute values, so make sure the children are strings.
        if isinstance(value, Sequence):
            new_value = []
            for v in value:
                if not isinstance(v, (str, bytes)) and isinstance(v, Sequence):
                    # This is most certainly a user error and will crash and burn later.
                    # To keep things working, we'll do what we do with all objects,
                    # And convert them to strings.
                    new_value.append(str(v))
                else:
                    # Convert the child to a string
                    new_value.append(cast(str, cls.normalize_value(v)))
            return new_value

        # Try and make anything else a string
        return str(value)

    @classmethod
    def get_attribute_by_name(
        cls,
        el: bs4.Tag,
        name: str,
        default: str | Sequence[str] | None = None
    ) -> str | Sequence[str] | None:
        """Get attribute by name."""

        value = default
        if el._is_xml:
            try:
                value = cls.normalize_value(el.attrs[name])
            except KeyError:
                pass
        else:
            for k, v in el.attrs.items():
                if util.lower(k) == name:
                    value = cls.normalize_value(v)
                    break
        return value

    @classmethod
    def iter_attributes(cls, el: bs4.Tag | None) -> Iterator[tuple[str, str | Sequence[str] | None]]:
        """Iterate attributes."""

        if el is not None:
            for k, v in el.attrs.items():
                yield k, cls.normalize_value(v)

    @classmethod
    def get_classes(cls, el: bs4.Tag) -> Sequence[str]:
        """Get classes."""

        classes = cls.get_attribute_by_name(el, 'class', [])
        if isinstance(classes, str):
            classes = RE_NOT_WS.findall(classes)
        return cast(Sequence[str], classes)

    def get_text(self, el: bs4.Tag, no_iframe: bool = False) -> str:
        """Get text."""

        return ''.join(
            [
                node for node in self.get_descendants(el, no_iframe=no_iframe)  # type: ignore[misc]
                if self.is_content_string(node)
            ]
        )

    def get_own_text(self, el: bs4.Tag, no_iframe: bool = False) -> list[str]:
        """Get Own Text."""

        return [
            node for node in self.get_contents(el, no_iframe=no_iframe) if self.is_content_string(node)  # type: ignore[misc]
        ]


class Inputs:
    """Class for parsing and validating input items."""

    @staticmethod
    def validate_day(year: int, month: int, day: int) -> bool:
        """Validate day."""

        max_days = LONG_MONTH
        if month == FEB:
            max_days = FEB_LEAP_MONTH if ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0) else FEB_MONTH
        elif month in MONTHS_30:
            max_days = SHORT_MONTH
        return 1 <= day <= max_days

    @staticmethod
    def validate_week(year: int, week: int) -> bool:
        """Validate week."""

        max_week = datetime.strptime(f"{12}-{31}-{year}", "%m-%d-%Y").isocalendar()[1]
        if max_week == 1:
            max_week = 53
        return 1 <= week <= max_week

    @staticmethod
    def validate_month(month: int) -> bool:
        """Validate month."""

        return 1 <= month <= 12

    @staticmethod
    def validate_year(year: int) -> bool:
        """Validate year."""

        return 1 <= year

    @staticmethod
    def validate_hour(hour: int) -> bool:
        """Validate hour."""

        return 0 <= hour <= 23

    @staticmethod
    def validate_minutes(minutes: int) -> bool:
        """Validate minutes."""

        return 0 <= minutes <= 59

    @classmethod
    def parse_value(cls, itype: str, value: str | None) -> tuple[float, ...] | None:
        """Parse the input value."""

        parsed = None  # type: tuple[float, ...] | None
        if value is None:
            return value
        if itype == "date":
            m = RE_DATE.match(value)
            if m:
                year = int(m.group('year'), 10)
                month = int(m.group('month'), 10)
                day = int(m.group('day'), 10)
                if cls.validate_year(year) and cls.validate_month(month) and cls.validate_day(year, month, day):
                    parsed = (year, month, day)
        elif itype == "month":
            m = RE_MONTH.match(value)
            if m:
                year = int(m.group('year'), 10)
                month = int(m.group('month'), 10)
                if cls.validate_year(year) and cls.validate_month(month):
                    parsed = (year, month)
        elif itype == "week":
            m = RE_WEEK.match(value)
            if m:
                year = int(m.group('year'), 10)
                week = int(m.group('week'), 10)
                if cls.validate_year(year) and cls.validate_week(year, week):
                    parsed = (year, week)
        elif itype == "time":
            m = RE_TIME.match(value)
            if m:
                hour = int(m.group('hour'), 10)
                minutes = int(m.group('minutes'), 10)
                if cls.validate_hour(hour) and cls.validate_minutes(minutes):
                    parsed = (hour, minutes)
        elif itype == "datetime-local":
            m = RE_DATETIME.match(value)
            if m:
                year = int(m.group('year'), 10)
                month = int(m.group('month'), 10)
                day = int(m.group('day'), 10)
                hour = int(m.group('hour'), 10)
                minutes = int(m.group('minutes'), 10)
                if (
                    cls.validate_year(year) and cls.validate_month(month) and cls.validate_day(year, month, day) and
                    cls.validate_hour(hour) and cls.validate_minutes(minutes)
                ):
                    parsed = (year, month, day, hour, minutes)
        elif itype in ("number", "range"):
            m = RE_NUM.match(value)
            if m:
                parsed = (float(m.group('value')),)
        return parsed


class CSSMatch(_DocumentNav):
    """Perform CSS matching."""

    def __init__(
        self,
        selectors: ct.SelectorList,
        scope: bs4.Tag | None,
        namespaces: ct.Namespaces | None,
        flags: int
    ) -> None:
        """Initialize."""

        self.assert_valid_input(scope)
        self.tag = scope
        self.cached_meta_lang = []  # type: list[tuple[str, str]]
        self.cached_default_forms = []  # type: list[tuple[bs4.Tag, bs4.Tag]]
        self.cached_indeterminate_forms = []  # type: list[tuple[bs4.Tag, str, bool]]
        self.selectors = selectors
        self.namespaces = {} if namespaces is None else namespaces  # type: ct.Namespaces | dict[str, str]
        self.flags = flags
        self.iframe_restrict = False

        # Find the root element for the whole tree
        doc = scope
        parent = self.get_parent(doc)
        while parent:
            doc = parent
            parent = self.get_parent(doc)
        root = None  # type: bs4.Tag | None
        if not self.is_doc(doc):
            root = doc
        else:
            for child in self.get_tag_children(doc):
                root = child
                break

        self.root = root
        self.scope = scope if scope is not doc else root
        self.has_html_namespace = self.has_html_ns(root)

        # A document can be both XML and HTML (XHTML)
        self.is_xml = self.is_xml_tree(doc)
        self.is_html = not self.is_xml or self.has_html_namespace

    def supports_namespaces(self) -> bool:
        """Check if namespaces are supported in the HTML type."""

        return self.is_xml or self.has_html_namespace

    def get_tag_ns(self, el: bs4.Tag | None) -> str:
        """Get tag namespace."""

        namespace = ''
        if el is None:  # pragma: no cover
            return namespace

        if self.supports_namespaces():
            ns = self.get_uri(el)
            if ns:
                namespace = ns
        else:
            namespace = NS_XHTML
        return namespace

    def is_html_tag(self, el: bs4.Tag | None) -> bool:
        """Check if tag is in HTML namespace."""

        return self.get_tag_ns(el) == NS_XHTML

    def get_tag(self, el: bs4.Tag | None) -> str | None:
        """Get tag."""

        name = self.get_tag_name(el)
        return util.lower(name) if name is not None and not self.is_xml else name

    def get_prefix(self, el: bs4.Tag) -> str | None:
        """Get prefix."""

        prefix = self.get_prefix_name(el)
        return util.lower(prefix) if prefix is not None and not self.is_xml else prefix

    def find_bidi(self, el: bs4.Tag) -> int | None:
        """Get directionality from element text."""

        for node in self.get_children(el):

            # Analyze child text nodes
            if self.is_tag(node):

                # Avoid analyzing certain elements specified in the specification.
                direction = DIR_MAP.get(util.lower(self.get_attribute_by_name(node, 'dir', '')), None)  # type: ignore[arg-type]
                name = self.get_tag(node)  # type: ignore[arg-type]
                if (
                    (name and name in ('bdi', 'script', 'style', 'textarea', 'iframe')) or
                    not self.is_html_tag(node) or  # type: ignore[arg-type]
                    direction is not None
                ):
                    continue  # pragma: no cover

                # Check directionality of this node's text
                value = self.find_bidi(node)  # type: ignore[arg-type]
                if value is not None:
                    return value

                # Direction could not be determined
                continue  # pragma: no cover

            # Skip `doctype` comments, etc.
            if self.is_special_string(node):
                continue

            # Analyze text nodes for directionality.
            for c in node:  # type: ignore[attr-defined]
                bidi = unicodedata.bidirectional(c)
                if bidi in ('AL', 'R', 'L'):
                    return ct.SEL_DIR_LTR if bidi == 'L' else ct.SEL_DIR_RTL
        return None

    def extended_language_filter(self, lang_range: str, lang_tag: str) -> bool:
        """Filter the language tags."""

        match = True
        lang_range = RE_WILD_STRIP.sub('-', lang_range).lower()
        ranges = lang_range.split('-')
        subtags = lang_tag.lower().split('-')
        length = len(ranges)
        slength = len(subtags)
        rindex = 0
        sindex = 0
        r = ranges[rindex]
        s = subtags[sindex]

        # Empty specified language should match unspecified language attributes
        if length == 1 and slength == 1 and not r and r == s:
            return True

        # Primary tag needs to match
        if (r != '*' and r != s) or (r == '*' and slength == 1 and not s):
            match = False

        rindex += 1
        sindex += 1

        # Match until we run out of ranges
        while match and rindex < length:
            r = ranges[rindex]
            try:
                s = subtags[sindex]
            except IndexError:
                # Ran out of subtags,
                # but we still have ranges
                match = False
                continue

            # Empty range
            if not r:
                match = False
                continue

            # Matched range
            elif s == r:
                rindex += 1

            # Implicit wildcard cannot match
            # singletons
            elif len(s) == 1:
                match = False
                continue

            # Implicitly matched, so grab next subtag
            sindex += 1

        return match

    def match_attribute_name(
        self,
        el: bs4.Tag,
        attr: str,
        prefix: str | None
    ) -> str | Sequence[str] | None:
        """Match attribute name and return value if it exists."""

        value = None
        if self.supports_namespaces():
            value = None
            # If we have not defined namespaces, we can't very well find them, so don't bother trying.
            if prefix:
                ns = self.namespaces.get(prefix)
                if ns is None and prefix != '*':
                    return None
            else:
                ns = None

            for k, v in self.iter_attributes(el):

                # Get attribute parts
                namespace, name = self.split_namespace(el, k)

                # Can't match a prefix attribute as we haven't specified one to match
                # Try to match it normally as a whole `p:a` as selector may be trying `p\:a`.
                if ns is None:
                    if (self.is_xml and attr == k) or (not self.is_xml and util.lower(attr) == util.lower(k)):
                        value = v
                        break
                    # Coverage is not finding this even though it is executed.
                    # Adding a print statement before this (and erasing coverage) causes coverage to find the line.
                    # Ignore the false positive message.
                    continue  # pragma: no cover

                # We can't match our desired prefix attribute as the attribute doesn't have a prefix
                if namespace is None or (ns != namespace and prefix != '*'):
                    continue

                # The attribute doesn't match.
                if (util.lower(attr) != util.lower(name)) if not self.is_xml else (attr != name):
                    continue

                value = v
                break
        else:
            for k, v in self.iter_attributes(el):
                if util.lower(attr) != util.lower(k):
                    continue
                value = v
                break
        return value

    def match_namespace(self, el: bs4.Tag, tag: ct.SelectorTag) -> bool:
        """Match the namespace of the element."""

        match = True
        namespace = self.get_tag_ns(el)
        default_namespace = self.namespaces.get('')
        tag_ns = '' if tag.prefix is None else self.namespaces.get(tag.prefix)
        # We must match the default namespace if one is not provided
        if tag.prefix is None and (default_namespace is not None and namespace != default_namespace):
            match = False
        # If we specified `|tag`, we must not have a namespace.
        elif (tag.prefix is not None and tag.prefix == '' and namespace):
            match = False
        # Verify prefix matches
        elif (
            tag.prefix and
            tag.prefix != '*' and (tag_ns is None or namespace != tag_ns)
        ):
            match = False
        return match

    def match_attributes(self, el: bs4.Tag, attributes: tuple[ct.SelectorAttribute, ...]) -> bool:
        """Match attributes."""

        match = True
        if attributes:
            for a in attributes:
                temp = self.match_attribute_name(el, a.attribute, a.prefix)
                pattern = a.xml_type_pattern if self.is_xml and a.xml_type_pattern else a.pattern
                if temp is None:
                    match = False
                    break
                value = temp if isinstance(temp, str) else ' '.join(temp)
                if pattern is None:
                    continue
                elif pattern.match(value) is None:
                    match = False
                    break
        return match

    def match_tagname(self, el: bs4.Tag, tag: ct.SelectorTag) -> bool:
        """Match tag name."""

        name = (util.lower(tag.name) if not self.is_xml and tag.name is not None else tag.name)
        return not (
            name is not None and
            name not in (self.get_tag(el), '*')
        )

    def match_tag(self, el: bs4.Tag, tag: ct.SelectorTag | None) -> bool:
        """Match the tag."""

        match = True
        if tag is not None:
            # Verify namespace
            if not self.match_namespace(el, tag):
                match = False
            if not self.match_tagname(el, tag):
                match = False
        return match

    def match_past_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
        """Match past relationship."""

        found = False
        # I don't think this can ever happen, but it makes `mypy` happy
        if isinstance(relation[0], ct.SelectorNull):  # pragma: no cover
            return found

        if relation[0].rel_type == REL_PARENT:
            parent = self.get_parent(el, no_iframe=self.iframe_restrict)
            while not found and parent:
                found = self.match_selectors(parent, relation)
                parent = self.get_parent(parent, no_iframe=self.iframe_restrict)
        elif relation[0].rel_type == REL_CLOSE_PARENT:
            parent = self.get_parent(el, no_iframe=self.iframe_restrict)
            if parent:
                found = self.match_selectors(parent, relation)
        elif relation[0].rel_type == REL_SIBLING:
            sibling = self.get_previous_tag(el)
            while not found and sibling:
                found = self.match_selectors(sibling, relation)
                sibling = self.get_previous_tag(sibling)
        elif relation[0].rel_type == REL_CLOSE_SIBLING:
            sibling = self.get_previous_tag(el)
            if sibling and self.is_tag(sibling):
                found = self.match_selectors(sibling, relation)
        return found

    def match_future_child(self, parent: bs4.Tag, relation: ct.SelectorList, recursive: bool = False) -> bool:
        """Match future child."""

        match = False
        if recursive:
            children = self.get_tag_descendants  # type: Callable[..., Iterator[bs4.Tag]]
        else:
            children = self.get_tag_children
        for child in children(parent, no_iframe=self.iframe_restrict):
            match = self.match_selectors(child, relation)
            if match:
                break
        return match

    def match_future_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
        """Match future relationship."""

        found = False
        # I don't think this can ever happen, but it makes `mypy` happy
        if isinstance(relation[0], ct.SelectorNull):  # pragma: no cover
            return found

        if relation[0].rel_type == REL_HAS_PARENT:
            found = self.match_future_child(el, relation, True)
        elif relation[0].rel_type == REL_HAS_CLOSE_PARENT:
            found = self.match_future_child(el, relation)
        elif relation[0].rel_type == REL_HAS_SIBLING:
            sibling = self.get_next_tag(el)
            while not found and sibling:
                found = self.match_selectors(sibling, relation)
                sibling = self.get_next_tag(sibling)
        elif relation[0].rel_type == REL_HAS_CLOSE_SIBLING:
            sibling = self.get_next_tag(el)
            if sibling and self.is_tag(sibling):
                found = self.match_selectors(sibling, relation)
        return found

    def match_relations(self, el: bs4.Tag, relation: ct.SelectorList) -> bool:
        """Match relationship to other elements."""

        found = False

        if isinstance(relation[0], ct.SelectorNull) or relation[0].rel_type is None:
            return found

        if relation[0].rel_type.startswith(':'):
            found = self.match_future_relations(el, relation)
        else:
            found = self.match_past_relations(el, relation)

        return found

    def match_id(self, el: bs4.Tag, ids: tuple[str, ...]) -> bool:
        """Match element's ID."""

        found = True
        for i in ids:
            if i != self.get_attribute_by_name(el, 'id', ''):
                found = False
                break
        return found

    def match_classes(self, el: bs4.Tag, classes: tuple[str, ...]) -> bool:
        """Match element's classes."""

        current_classes = self.get_classes(el)
        found = True
        for c in classes:
            if c not in current_classes:
                found = False
                break
        return found

    def match_root(self, el: bs4.Tag) -> bool:
        """Match element as root."""

        is_root = self.is_root(el)
        if is_root:
            sibling = self.get_previous(el)  # type: Any
            while is_root and sibling is not None:
                if (
                    self.is_tag(sibling) or (self.is_content_string(sibling) and sibling.strip()) or
                    self.is_cdata(sibling)
                ):
                    is_root = False
                else:
                    sibling = self.get_previous(sibling)
        if is_root:
            sibling = self.get_next(el)
            while is_root and sibling is not None:
                if (
                    self.is_tag(sibling) or (self.is_content_string(sibling) and sibling.strip()) or
                    self.is_cdata(sibling)
                ):
                    is_root = False
                else:
                    sibling = self.get_next(sibling)
        return is_root

    def match_scope(self, el: bs4.Tag) -> bool:
        """Match element as scope."""

        return self.scope is el

    def match_nth_tag_type(self, el: bs4.Tag, child: bs4.Tag) -> bool:
        """Match tag type for `nth` matches."""

        return (
            (self.get_tag(child) == self.get_tag(el)) and
            (self.get_tag_ns(child) == self.get_tag_ns(el))
        )

    def match_nth(self, el: bs4.Tag, nth: tuple[ct.SelectorNth, ...]) -> bool:
        """Match `nth` elements."""

        matched = True

        for n in nth:
            matched = False
            if n.selectors and not self.match_selectors(el, n.selectors):
                break
            parent = self.get_parent(el)  # type: bs4.Tag | None
            if parent is None:
                parent = cast('bs4.Tag', self.create_fake_parent(el))
            last = n.last
            last_index = len(parent) - 1
            index = last_index if last else 0
            relative_index = 0
            a = n.a
            b = n.b
            var = n.n
            count = 0
            count_incr = 1
            factor = -1 if last else 1
            idx = last_idx = a * count + b if var else a

            # We can only adjust bounds within a variable index
            if var:
                # Abort if our nth index is out of bounds and only getting further out of bounds as we increment.
                # Otherwise, increment to try to get in bounds.
                adjust = None
                while idx < 1 or idx > last_index:
                    if idx < 0:
                        diff_low = 0 - idx
                        if adjust is not None and adjust == 1:
                            break
                        adjust = -1
                        count += count_incr
                        idx = last_idx = a * count + b if var else a
                        diff = 0 - idx
                        if diff >= diff_low:
                            break
                    else:
                        diff_high = idx - last_index
                        if adjust is not None and adjust == -1:
                            break
                        adjust = 1
                        count += count_incr
                        idx = last_idx = a * count + b if var else a
                        diff = idx - last_index
                        if diff >= diff_high:
                            break
                        diff_high = diff

                # If a < 0, our count is working backwards, so floor the index by increasing the count.
                # Find the count that yields the lowest, in bound value and use that.
                # Lastly reverse count increment so that we'll increase our index.
                lowest = count
                if a < 0:
                    while idx >= 1:
                        lowest = count
                        count += count_incr
                        idx = last_idx = a * count + b if var else a
                    count_incr = -1
                count = lowest
                idx = last_idx = a * count + b if var else a

            # Evaluate elements while our calculated nth index is still in range
            while 1 <= idx <= last_index + 1:
                child = None  # type: bs4.element.PageElement | None
                # Evaluate while our child index is still in range.
                for child in self.get_children(parent, start=index, reverse=factor < 0):
                    index += factor
                    if not isinstance(child, bs4.Tag):
                        continue
                    # Handle `of S` in `nth-child`
                    if n.selectors and not self.match_selectors(child, n.selectors):
                        continue
                    # Handle `of-type`
                    if n.of_type and not self.match_nth_tag_type(el, child):
                        continue
                    relative_index += 1
                    if relative_index == idx:
                        if child is el:
                            matched = True
                        else:
                            break
                    if child is el:
                        break
                if child is el:
                    break
                last_idx = idx
                count += count_incr
                if count < 0:
                    # Count is counting down and has now ventured into invalid territory.
                    break
                idx = a * count + b if var else a
                if last_idx == idx:
                    break
            if not matched:
                break
        return matched

    def match_empty(self, el: bs4.Tag) -> bool:
        """Check if element is empty (if requested)."""

        is_empty = True
        for child in self.get_children(el):
            if self.is_tag(child):
                is_empty = False
                break
            elif self.is_content_string(child) and RE_NOT_EMPTY.search(child):  # type: ignore[call-overload]
                is_empty = False
                break
        return is_empty

    def match_subselectors(self, el: bs4.Tag, selectors: tuple[ct.SelectorList, ...]) -> bool:
        """Match selectors."""

        match = True
        for sel in selectors:
            if not self.match_selectors(el, sel):
                match = False
        return match

    def match_contains(self, el: bs4.Tag, contains: tuple[ct.SelectorContains, ...]) -> bool:
        """Match element if it contains text."""

        match = True
        content = None  # type: str | Sequence[str] | None
        for contain_list in contains:
            if content is None:
                if contain_list.own:
                    content = self.get_own_text(el, no_iframe=self.is_html)
                else:
                    content = self.get_text(el, no_iframe=self.is_html)
            found = False
            for text in contain_list.text:
                if contain_list.own:
                    for c in content:
                        if text in c:
                            found = True
                            break
                    if found:
                        break
                else:
                    if text in content:
                        found = True
                        break
            if not found:
                match = False
        return match

    def match_default(self, el: bs4.Tag) -> bool:
        """Match default."""

        match = False

        # Find this input's form
        form = None  # type: bs4.Tag | None
        parent = self.get_parent(el, no_iframe=True)
        while parent and form is None:
            if self.get_tag(parent) == 'form' and self.is_html_tag(parent):
                form = parent
            else:
                parent = self.get_parent(parent, no_iframe=True)

        if form is not None:
            # Look in form cache to see if we've already located its default button
            found_form = False
            for f, t in self.cached_default_forms:
                if f is form:
                    found_form = True
                    if t is el:
                        match = True
                    break

            # We didn't have the form cached, so look for its default button
            if not found_form:
                for child in self.get_tag_descendants(form, no_iframe=True):
                    name = self.get_tag(child)
                    # Can't do nested forms (haven't figured out why we never hit this)
                    if name == 'form':  # pragma: no cover
                        break
                    if name in ('input', 'button'):
                        v = self.get_attribute_by_name(child, 'type', '')
                        if v and util.lower(v) == 'submit':
                            self.cached_default_forms.append((form, child))
                            if el is child:
                                match = True
                            break
        return match

    def match_indeterminate(self, el: bs4.Tag) -> bool:
        """Match default."""

        match = False
        name = cast(str, self.get_attribute_by_name(el, 'name'))

        def get_parent_form(el: bs4.Tag) -> bs4.Tag | None:
            """Find this input's form."""
            form = None
            parent = self.get_parent(el, no_iframe=True)
            while form is None:
                if self.get_tag(parent) == 'form' and self.is_html_tag(parent):
                    form = parent
                    break
                last_parent = parent
                parent = self.get_parent(parent, no_iframe=True)
                if parent is None:
                    form = last_parent
                    break
            return form

        form = get_parent_form(el)

        # Look in form cache to see if we've already evaluated that its fellow radio buttons are indeterminate
        if form is not None:
            found_form = False
            for f, n, i in self.cached_indeterminate_forms:
                if f is form and n == name:
                    found_form = True
                    if i is True:
                        match = True
                    break

            # We didn't have the form cached, so validate that the radio button is indeterminate
            if not found_form:
                checked = False
                for child in self.get_tag_descendants(form, no_iframe=True):
                    if child is el:
                        continue
                    tag_name = self.get_tag(child)
                    if tag_name == 'input':
                        is_radio = False
                        check = False
                        has_name = False
                        for k, v in self.iter_attributes(child):
                            if util.lower(k) == 'type' and util.lower(v) == 'radio':
                                is_radio = True
                            elif util.lower(k) == 'name' and v == name:
                                has_name = True
                            elif util.lower(k) == 'checked':
                                check = True
                            if is_radio and check and has_name and get_parent_form(child) is form:
                                checked = True
                                break
                    if checked:
                        break
                if not checked:
                    match = True
                self.cached_indeterminate_forms.append((form, name, match))

        return match

    def match_lang(self, el: bs4.Tag, langs: tuple[ct.SelectorLang, ...]) -> bool:
        """Match languages."""

        match = False
        has_ns = self.supports_namespaces()
        root = self.root
        has_html_namespace = self.has_html_namespace

        # Walk parents looking for `lang` (HTML) or `xml:lang` XML property.
        parent = el  # type: bs4.Tag | None
        found_lang = None
        last = None
        while not found_lang:
            has_html_ns = self.has_html_ns(parent)
            for k, v in self.iter_attributes(parent):
                attr_ns, attr = self.split_namespace(parent, k)
                if (
                    ((not has_ns or has_html_ns) and (util.lower(k) if not self.is_xml else k) == 'lang') or
                    (
                        has_ns and not has_html_ns and attr_ns == NS_XML and
                        (util.lower(attr) if not self.is_xml and attr is not None else attr) == 'lang'
                    )
                ):
                    found_lang = v
                    break
            last = parent
            parent = self.get_parent(parent, no_iframe=self.is_html)

            if parent is None:
                root = last
                has_html_namespace = self.has_html_ns(root)
                parent = last
                break

        # Use cached meta language.
        if found_lang is None and self.cached_meta_lang:
            for cache in self.cached_meta_lang:
                if root is cache[0]:
                    found_lang = cache[1]

        # If we couldn't find a language, and the document is HTML, look to meta to determine language.
        if found_lang is None and (not self.is_xml or (has_html_namespace and root and root.name == 'html')):
            # Find head
            found = False
            for tag in ('html', 'head'):
                found = False
                for child in self.get_tag_children(parent, no_iframe=self.is_html):
                    if self.get_tag(child) == tag and self.is_html_tag(child):
                        found = True
                        parent = child
                        break
                if not found:  # pragma: no cover
                    break

            # Search meta tags
            if found and parent is not None:
                for child2 in parent:
                    if isinstance(child2, bs4.Tag) and self.get_tag(child2) == 'meta' and self.is_html_tag(parent):
                        c_lang = False
                        content = None
                        for k, v in self.iter_attributes(child2):
                            if util.lower(k) == 'http-equiv' and util.lower(v) == 'content-language':
                                c_lang = True
                            if util.lower(k) == 'content':
                                content = v
                            if c_lang and content:
                                found_lang = content
                                self.cached_meta_lang.append((cast(str, root), cast(str, found_lang)))
                                break
                    if found_lang is not None:
                        break
                if found_lang is None:
                    self.cached_meta_lang.append((cast(str, root), ''))

        # If we determined a language, compare.
        if found_lang is not None:
            for patterns in langs:
                match = False
                for pattern in patterns:
                    if self.extended_language_filter(pattern, cast(str, found_lang)):
                        match = True
                if not match:
                    break

        return match

    def match_dir(self, el: bs4.Tag | None, directionality: int) -> bool:
        """Check directionality."""

        # If we have to match both left and right, we can't match either.
        if directionality & ct.SEL_DIR_LTR and directionality & ct.SEL_DIR_RTL:
            return False

        if el is None or not self.is_html_tag(el):
            return False

        # Element has defined direction of left to right or right to left
        direction = DIR_MAP.get(util.lower(self.get_attribute_by_name(el, 'dir', '')), None)
        if direction not in (None, 0):
            return direction == directionality

        # Element is the document element (the root) and no direction assigned, assume left to right.
        is_root = self.is_root(el)
        if is_root and direction is None:
            return ct.SEL_DIR_LTR == directionality

        # If `input[type=telephone]` and no direction is assigned, assume left to right.
        name = self.get_tag(el)
        is_input = name == 'input'
        is_textarea = name == 'textarea'
        is_bdi = name == 'bdi'
        itype = util.lower(self.get_attribute_by_name(el, 'type', '')) if is_input else ''
        if is_input and itype == 'tel' and direction is None:
            return ct.SEL_DIR_LTR == directionality

        # Auto handling for text inputs
        if ((is_input and itype in ('text', 'search', 'tel', 'url', 'email')) or is_textarea) and direction == 0:
            if is_textarea:
                value = ''.join(node for node in self.get_contents(el, no_iframe=True) if self.is_content_string(node))  # type: ignore[misc]
            else:
                value = cast(str, self.get_attribute_by_name(el, 'value', ''))
            if value:
                for c in value:
                    bidi = unicodedata.bidirectional(c)
                    if bidi in ('AL', 'R', 'L'):
                        direction = ct.SEL_DIR_LTR if bidi == 'L' else ct.SEL_DIR_RTL
                        return direction == directionality
                # Assume left to right
                return ct.SEL_DIR_LTR == directionality
            elif is_root:
                return ct.SEL_DIR_LTR == directionality
            return self.match_dir(self.get_parent(el, no_iframe=True), directionality)

        # Auto handling for `bdi` and other non text inputs.
        if (is_bdi and direction is None) or direction == 0:
            direction = self.find_bidi(el)
            if direction is not None:
                return direction == directionality
            elif is_root:
                return ct.SEL_DIR_LTR == directionality
            return self.match_dir(self.get_parent(el, no_iframe=True), directionality)

        # Match parents direction
        return self.match_dir(self.get_parent(el, no_iframe=True), directionality)

    def match_range(self, el: bs4.Tag, condition: int) -> bool:
        """
        Match range.

        Behavior is modeled after what we see in browsers. Browsers seem to evaluate
        if the value is out of range, and if not, it is in range. So a missing value
        will not evaluate out of range; therefore, value is in range. Personally, I
        feel like this should evaluate as neither in or out of range.
        """

        out_of_range = False

        itype = util.lower(self.get_attribute_by_name(el, 'type'))
        mn = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'min', None)))
        mx = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'max', None)))

        # There is no valid min or max, so we cannot evaluate a range
        if mn is None and mx is None:
            return False

        value = Inputs.parse_value(itype, cast(str, self.get_attribute_by_name(el, 'value', None)))
        if value is not None:
            if itype in ("date", "datetime-local", "month", "week", "number", "range"):
                if mn is not None and value < mn:
                    out_of_range = True
                if not out_of_range and mx is not None and value > mx:
                    out_of_range = True
            elif itype == "time":
                if mn is not None and mx is not None and mn > mx:
                    # Time is periodic, so this is a reversed/discontinuous range
                    if value < mn and value > mx:
                        out_of_range = True
                else:
                    if mn is not None and value < mn:
                        out_of_range = True
                    if not out_of_range and mx is not None and value > mx:
                        out_of_range = True

        return not out_of_range if condition & ct.SEL_IN_RANGE else out_of_range

    def match_defined(self, el: bs4.Tag) -> bool:
        """
        Match defined.

        `:defined` is related to custom elements in a browser.

        - If the document is XML (not XHTML), all tags will match.
        - Tags that are not custom (don't have a hyphen) are marked defined.
        - If the tag has a prefix (without or without a namespace), it will not match.

        This is of course requires the parser to provide us with the proper prefix and namespace info,
        if it doesn't, there is nothing we can do.
        """

        name = self.get_tag(el)
        return (
            name is not None and (
                name.find('-') == -1 or
                name.find(':') != -1 or
                self.get_prefix(el) is not None
            )
        )

    def match_placeholder_shown(self, el: bs4.Tag) -> bool:
        """
        Match placeholder shown according to HTML spec.

        - text area should be checked if they have content. A single newline does not count as content.

        """

        match = False
        content = self.get_text(el)
        if content in ('', '\n'):
            match = True

        return match

    def match_selectors(self, el: bs4.Tag, selectors: ct.SelectorList) -> bool:
        """Check if element matches one of the selectors."""

        match = False
        is_not = selectors.is_not
        is_html = selectors.is_html

        # Internal selector lists that use the HTML flag, will automatically get the `html` namespace.
        if is_html:
            namespaces = self.namespaces
            iframe_restrict = self.iframe_restrict
            self.namespaces = {'html': NS_XHTML}
            self.iframe_restrict = True

        if not is_html or self.is_html:
            for selector in selectors:
                match = is_not
                # We have a un-matchable situation (like `:focus` as you can focus an element in this environment)
                if isinstance(selector, ct.SelectorNull):
                    continue
                # Verify tag matches
                if not self.match_tag(el, selector.tag):
                    continue
                # Verify tag is defined
                if selector.flags & ct.SEL_DEFINED and not self.match_defined(el):
                    continue
                # Verify element is root
                if selector.flags & ct.SEL_ROOT and not self.match_root(el):
                    continue
                # Verify element is scope
                if selector.flags & ct.SEL_SCOPE and not self.match_scope(el):
                    continue
                # Verify element has placeholder shown
                if selector.flags & ct.SEL_PLACEHOLDER_SHOWN and not self.match_placeholder_shown(el):
                    continue
                # Verify `nth` matches
                if not self.match_nth(el, selector.nth):
                    continue
                if selector.flags & ct.SEL_EMPTY and not self.match_empty(el):
                    continue
                # Verify id matches
                if selector.ids and not self.match_id(el, selector.ids):
                    continue
                # Verify classes match
                if selector.classes and not self.match_classes(el, selector.classes):
                    continue
                # Verify attribute(s) match
                if not self.match_attributes(el, selector.attributes):
                    continue
                # Verify ranges
                if selector.flags & RANGES and not self.match_range(el, selector.flags & RANGES):
                    continue
                # Verify language patterns
                if selector.lang and not self.match_lang(el, selector.lang):
                    continue
                # Verify pseudo selector patterns
                if selector.selectors and not self.match_subselectors(el, selector.selectors):
                    continue
                # Verify relationship selectors
                if selector.relation and not self.match_relations(el, selector.relation):
                    continue
                # Validate that the current default selector match corresponds to the first submit button in the form
                if selector.flags & ct.SEL_DEFAULT and not self.match_default(el):
                    continue
                # Validate that the unset radio button is among radio buttons with the same name in a form that are
                # also not set.
                if selector.flags & ct.SEL_INDETERMINATE and not self.match_indeterminate(el):
                    continue
                # Validate element directionality
                if selector.flags & DIR_FLAGS and not self.match_dir(el, selector.flags & DIR_FLAGS):
                    continue
                # Validate that the tag contains the specified text.
                if selector.contains and not self.match_contains(el, selector.contains):
                    continue
                match = not is_not
                break

        # Restore actual namespaces being used for external selector lists
        if is_html:
            self.namespaces = namespaces
            self.iframe_restrict = iframe_restrict

        return match

    def select(self, limit: int = 0) -> Iterator[bs4.Tag]:
        """Match all tags under the targeted tag."""

        lim = None if limit < 1 else limit

        for child in self.get_tag_descendants(self.tag):
            if self.match(child):
                yield child
                if lim is not None:
                    lim -= 1
                    if lim < 1:
                        break

    def closest(self) -> bs4.Tag | None:
        """Match closest ancestor."""

        current = self.tag  # type: bs4.Tag | None
        closest = None
        while closest is None and current is not None:
            if self.match(current):
                closest = current
            else:
                current = self.get_parent(current)
        return closest

    def filter(self) -> list[bs4.Tag]:  # noqa A001
        """Filter tag's children."""

        return [
            tag for tag in self.get_contents(self.tag)
            if isinstance(tag, bs4.Tag) and self.match(tag)
        ]

    def match(self, el: bs4.Tag) -> bool:
        """Match."""

        return not self.is_doc(el) and self.is_tag(el) and self.match_selectors(el, self.selectors)


class SoupSieve(ct.Immutable):
    """Compiled Soup Sieve selector matching object."""

    pattern: str
    selectors: ct.SelectorList
    namespaces: ct.Namespaces | None
    custom: dict[str, str]
    flags: int

    __slots__ = ("pattern", "selectors", "namespaces", "custom", "flags", "_hash")

    def __init__(
        self,
        pattern: str,
        selectors: ct.SelectorList,
        namespaces: ct.Namespaces | None,
        custom: ct.CustomSelectors | None,
        flags: int
    ):
        """Initialize."""

        super().__init__(
            pattern=pattern,
            selectors=selectors,
            namespaces=namespaces,
            custom=custom,
            flags=flags
        )

    def match(self, tag: bs4.Tag) -> bool:
        """Match."""

        return CSSMatch(self.selectors, tag, self.namespaces, self.flags).match(tag)

    def closest(self, tag: bs4.Tag) -> bs4.Tag | None:
        """Match closest ancestor."""

        return CSSMatch(self.selectors, tag, self.namespaces, self.flags).closest()

    def filter(self, iterable: Iterable[bs4.Tag]) -> list[bs4.Tag]:  # noqa A001
        """
        Filter.

        `CSSMatch` can cache certain searches for tags of the same document,
        so if we are given a tag, all tags are from the same document,
        and we can take advantage of the optimization.

        Any other kind of iterable could have tags from different documents or detached tags,
        so for those, we use a new `CSSMatch` for each item in the iterable.
        """

        if isinstance(iterable, bs4.Tag):
            return CSSMatch(self.selectors, iterable, self.namespaces, self.flags).filter()
        else:
            return [node for node in iterable if not CSSMatch.is_navigable_string(node) and self.match(node)]

    def select_one(self, tag: bs4.Tag) -> bs4.Tag | None:
        """Select a single tag."""

        tags = self.select(tag, limit=1)
        return tags[0] if tags else None

    def select(self, tag: bs4.Tag, limit: int = 0) -> list[bs4.Tag]:
        """Select the specified tags."""

        return list(self.iselect(tag, limit))

    def iselect(self, tag: bs4.Tag, limit: int = 0) -> Iterator[bs4.Tag]:
        """Iterate the specified tags."""

        yield from CSSMatch(self.selectors, tag, self.namespaces, self.flags).select(limit)

    def __repr__(self) -> str:  # pragma: no cover
        """Representation."""

        return (
            f"SoupSieve(pattern={self.pattern!r}, namespaces={self.namespaces!r}, "
            f"custom={self.custom!r}, flags={self.flags!r})"
        )

    __str__ = __repr__


ct.pickle_register(SoupSieve)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\hermite_e.py ===
"""
===================================================================
HermiteE Series, "Probabilists" (:mod:`numpy.polynomial.hermite_e`)
===================================================================

This module provides a number of objects (mostly functions) useful for
dealing with Hermite_e series, including a `HermiteE` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   HermiteE

Constants
---------
.. autosummary::
   :toctree: generated/

   hermedomain
   hermezero
   hermeone
   hermex

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   hermeadd
   hermesub
   hermemulx
   hermemul
   hermediv
   hermepow
   hermeval
   hermeval2d
   hermeval3d
   hermegrid2d
   hermegrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   hermeder
   hermeint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   hermefromroots
   hermeroots
   hermevander
   hermevander2d
   hermevander3d
   hermegauss
   hermeweight
   hermecompanion
   hermefit
   hermetrim
   hermeline
   herme2poly
   poly2herme

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
    'hermezero', 'hermeone', 'hermex', 'hermedomain', 'hermeline',
    'hermeadd', 'hermesub', 'hermemulx', 'hermemul', 'hermediv',
    'hermepow', 'hermeval', 'hermeder', 'hermeint', 'herme2poly',
    'poly2herme', 'hermefromroots', 'hermevander', 'hermefit', 'hermetrim',
    'hermeroots', 'HermiteE', 'hermeval2d', 'hermeval3d', 'hermegrid2d',
    'hermegrid3d', 'hermevander2d', 'hermevander3d', 'hermecompanion',
    'hermegauss', 'hermeweight']

hermetrim = pu.trimcoef


def poly2herme(pol):
    """
    poly2herme(pol)

    Convert a polynomial to a Hermite series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Hermite series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Hermite
        series.

    See Also
    --------
    herme2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.hermite_e import poly2herme
    >>> poly2herme(np.arange(4))
    array([  2.,  10.,   2.,   3.])

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermeadd(hermemulx(res), pol[i])
    return res


def herme2poly(c):
    """
    Convert a Hermite series to a polynomial.

    Convert an array representing the coefficients of a Hermite series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Hermite series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2herme

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import herme2poly
    >>> herme2poly([  2.,  10.,   2.,   3.])
    array([0.,  1.,  2.,  3.])

    """
    from .polynomial import polyadd, polymulx, polysub

    [c] = pu.as_series([c])
    n = len(c)
    if n == 1:
        return c
    if n == 2:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (i - 1))
            c1 = polyadd(tmp, polymulx(c1))
        return polyadd(c0, polymulx(c1))


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Hermite
hermedomain = np.array([-1., 1.])

# Hermite coefficients representing zero.
hermezero = np.array([0])

# Hermite coefficients representing one.
hermeone = np.array([1])

# Hermite coefficients representing the identity x.
hermex = np.array([0, 1])


def hermeline(off, scl):
    """
    Hermite series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Hermite series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeline
    >>> from numpy.polynomial.hermite_e import hermeline, hermeval
    >>> hermeval(0,hermeline(3, 2))
    3.0
    >>> hermeval(1,hermeline(3, 2))
    5.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])


def hermefromroots(roots):
    """
    Generate a HermiteE series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in HermiteE form, where the :math:`r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * He_1(x) + ... +  c_n * He_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in HermiteE form.

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
    numpy.polynomial.chebyshev.chebfromroots

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermefromroots, hermeval
    >>> coef = hermefromroots((-1, 0, 1))
    >>> hermeval((-1, 0, 1), coef)
    array([0., 0., 0.])
    >>> coef = hermefromroots((-1j, 1j))
    >>> hermeval((-1j, 1j), coef)
    array([0.+0.j, 0.+0.j])

    """
    return pu._fromroots(hermeline, hermemul, roots)


def hermeadd(c1, c2):
    """
    Add one Hermite series to another.

    Returns the sum of two Hermite series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Hermite series of their sum.

    See Also
    --------
    hermesub, hermemulx, hermemul, hermediv, hermepow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Hermite series
    is a Hermite series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeadd
    >>> hermeadd([1, 2, 3], [1, 2, 3, 4])
    array([2.,  4.,  6.,  4.])

    """
    return pu._add(c1, c2)


def hermesub(c1, c2):
    """
    Subtract one Hermite series from another.

    Returns the difference of two Hermite series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their difference.

    See Also
    --------
    hermeadd, hermemulx, hermemul, hermediv, hermepow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Hermite
    series is a Hermite series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermesub
    >>> hermesub([1, 2, 3, 4], [1, 2, 3])
    array([0., 0., 0., 4.])

    """
    return pu._sub(c1, c2)


def hermemulx(c):
    """Multiply a Hermite series by x.

    Multiply the Hermite series `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    hermeadd, hermesub, hermemul, hermediv, hermepow

    Notes
    -----
    The multiplication uses the recursion relationship for Hermite
    polynomials in the form

    .. math::

        xP_i(x) = (P_{i + 1}(x) + iP_{i - 1}(x)))

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermemulx
    >>> hermemulx([1, 2, 3])
    array([2.,  7.,  2.,  3.])

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
        prd[i + 1] = c[i]
        prd[i - 1] += c[i] * i
    return prd


def hermemul(c1, c2):
    """
    Multiply one Hermite series by another.

    Returns the product of two Hermite series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Hermite series coefficients representing their product.

    See Also
    --------
    hermeadd, hermesub, hermemulx, hermediv, hermepow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Hermite polynomial basis set.  Thus, to express
    the product as a Hermite series, it is necessary to "reproject" the
    product onto said basis set, which may produce "unintuitive" (but
    correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermemul
    >>> hermemul([1, 2, 3], [0, 1, 2])
    array([14.,  15.,  28.,   7.,   6.])

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
            c0 = hermesub(c[-i] * xs, c1 * (nd - 1))
            c1 = hermeadd(tmp, hermemulx(c1))
    return hermeadd(c0, hermemulx(c1))


def hermediv(c1, c2):
    """
    Divide one Hermite series by another.

    Returns the quotient-with-remainder of two Hermite series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Hermite series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Hermite series coefficients representing the quotient and
        remainder.

    See Also
    --------
    hermeadd, hermesub, hermemulx, hermemul, hermepow

    Notes
    -----
    In general, the (polynomial) division of one Hermite series by another
    results in quotient and remainder terms that are not in the Hermite
    polynomial basis set.  Thus, to express these results as a Hermite
    series, it is necessary to "reproject" the results onto the Hermite
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermediv
    >>> hermediv([ 14.,  15.,  28.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([0.]))
    >>> hermediv([ 15.,  17.,  28.,   7.,   6.], [0, 1, 2])
    (array([1., 2., 3.]), array([1., 2.]))

    """
    return pu._div(hermemul, c1, c2)


def hermepow(c, pow, maxpower=16):
    """Raise a Hermite series to a power.

    Returns the Hermite series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Hermite series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Hermite series of power.

    See Also
    --------
    hermeadd, hermesub, hermemulx, hermemul, hermediv

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermepow
    >>> hermepow([1, 2, 3], 2)
    array([23.,  28.,  46.,  12.,   9.])

    """
    return pu._pow(hermemul, c, pow, maxpower)


def hermeder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Hermite_e series.

    Returns the series coefficients `c` differentiated `m` times along
    `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*He_0 + 2*He_1 + 3*He_2``
    while [[1,2],[1,2]] represents ``1*He_0(x)*He_0(y) + 1*He_1(x)*He_0(y)
    + 2*He_0(x)*He_1(y) + 2*He_1(x)*He_1(y)`` if axis=0 is ``x`` and axis=1
    is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite_e series coefficients. If `c` is multidimensional
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
        Hermite series of the derivative.

    See Also
    --------
    hermeint

    Notes
    -----
    In general, the result of differentiating a Hermite series does not
    resemble the same operation on a power series. Thus the result of this
    function may be "unintuitive," albeit correct; see Examples section
    below.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeder
    >>> hermeder([ 1.,  1.,  1.,  1.])
    array([1.,  2.,  3.])
    >>> hermeder([-0.25,  1.,  1./2.,  1./3.,  1./4 ], m=2)
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
        return c[:1] * 0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = j * c[j]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def hermeint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Hermite_e series.

    Returns the Hermite_e series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``H_0 + 2*H_1 + 3*H_2`` while [[1,2],[1,2]]
    represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) + 2*H_0(x)*H_1(y) +
    2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Hermite_e series coefficients. If c is multidimensional
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
        Hermite_e series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    hermeder

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
    >>> from numpy.polynomial.hermite_e import hermeint
    >>> hermeint([1, 2, 3]) # integrate once, value 0 at 0.
    array([1., 1., 1., 1.])
    >>> hermeint([1, 2, 3], m=2) # integrate twice, value & deriv 0 at 0
    array([-0.25      ,  1.        ,  0.5       ,  0.33333333,  0.25      ]) # may vary
    >>> hermeint([1, 2, 3], k=1) # integrate once, value 1 at 0.
    array([2., 1., 1., 1.])
    >>> hermeint([1, 2, 3], lbnd=-1) # integrate once, value 0 at -1
    array([-1.,  1.,  1.,  1.])
    >>> hermeint([1, 2, 3], m=2, k=[1, 2], lbnd=-1)
    array([ 1.83333333,  0.        ,  0.5       ,  0.33333333,  0.25      ]) # may vary

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
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            tmp[0] += k[i] - hermeval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def hermeval(x, c, tensor=True):
    """
    Evaluate an HermiteE series at points x.

    If `c` is of length ``n + 1``, this function returns the value:

    .. math:: p(x) = c_0 * He_0(x) + c_1 * He_1(x) + ... + c_n * He_n(x)

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
        with themselves and with the elements of `c`.
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
    hermeval2d, hermegrid2d, hermeval3d, hermegrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeval
    >>> coef = [1,2,3]
    >>> hermeval(1, coef)
    3.0
    >>> hermeval([[1,2],[3,4]], coef)
    array([[ 3., 14.],
           [31., 54.]])

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
            c0 = c[-i] - c1 * (nd - 1)
            c1 = tmp + c1 * x
    return c0 + c1 * x


def hermeval2d(x, y, c):
    """
    Evaluate a 2-D HermiteE series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * He_i(x) * He_j(y)

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
    hermeval, hermegrid2d, hermeval3d, hermegrid3d
    """
    return pu._valnd(hermeval, c, x, y)


def hermegrid2d(x, y, c):
    """
    Evaluate a 2-D HermiteE series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * H_i(a) * H_j(b)

    where the points ``(a, b)`` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
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
    hermeval, hermeval2d, hermeval3d, hermegrid3d
    """
    return pu._gridnd(hermeval, c, x, y)


def hermeval3d(x, y, z, c):
    """
    Evaluate a 3-D Hermite_e series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * He_i(x) * He_j(y) * He_k(z)

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
        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
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
    hermeval, hermeval2d, hermegrid2d, hermegrid3d
    """
    return pu._valnd(hermeval, c, x, y, z)


def hermegrid3d(x, y, z, c):
    """
    Evaluate a 3-D HermiteE series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * He_i(a) * He_j(b) * He_k(c)

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
    hermeval, hermeval2d, hermegrid2d, hermeval3d
    """
    return pu._gridnd(hermeval, c, x, y, z)


def hermevander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = He_i(x),

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the HermiteE polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    array ``V = hermevander(x, n)``, then ``np.dot(V, c)`` and
    ``hermeval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of HermiteE series of the same degree and sample points.

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
        corresponding HermiteE polynomial.  The dtype will be the same as
        the converted `x`.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.hermite_e import hermevander
    >>> x = np.array([-1, 0, 1])
    >>> hermevander(x, 3)
    array([[ 1., -1.,  0.,  2.],
           [ 1.,  0., -1., -0.],
           [ 1.,  1.,  0., -2.]])

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
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = (v[i - 1] * x - v[i - 2] * (i - 1))
    return np.moveaxis(v, 0, -1)


def hermevander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = He_i(x) * He_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the HermiteE polynomials.

    If ``V = hermevander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``hermeval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D HermiteE
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
    hermevander, hermevander3d, hermeval2d, hermeval3d
    """
    return pu._vander_nd_flat((hermevander, hermevander), (x, y), deg)


def hermevander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then Hehe pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = He_i(x)*He_j(y)*He_k(z),

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the degrees of the HermiteE polynomials.

    If ``V = hermevander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``hermeval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D HermiteE
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
    hermevander, hermevander3d, hermeval2d, hermeval3d
    """
    return pu._vander_nd_flat((hermevander, hermevander, hermevander), (x, y, z), deg)


def hermefit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Hermite series to data.

    Return the coefficients of a HermiteE series of degree `deg` that is
    the least squares fit to the data values `y` given at points `x`. If
    `y` is 1-D the returned coefficients will also be 1-D. If `y` is 2-D
    multiple fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * He_1(x) + ... + c_n * He_n(x),

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
        Hermite coefficients ordered from low to high. If `y` was 2-D,
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
        deficient. The warning is only raised if ``full = False``.  The
        warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.laguerre.lagfit
    hermeval : Evaluates a Hermite series.
    hermevander : pseudo Vandermonde matrix of Hermite series.
    hermeweight : HermiteE weight function.
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the HermiteE series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the pseudo Vandermonde matrix of `x`, the elements of `c`
    are the coefficients to be solved for, and the elements of `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `~exceptions.RankWarning` will be issued. This means that
    the coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using HermiteE series are probably most useful when the data can
    be approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the HermiteE
    weight. In that case the weight ``sqrt(w(x[i]))`` should be used
    together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is
    available as `hermeweight`.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.hermite_e import hermefit, hermeval
    >>> x = np.linspace(-10, 10)
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(scale=1./10, size=len(x))
    >>> y = hermeval(x, [1, 2, 3]) + err
    >>> hermefit(x, y, 2)
    array([1.02284196, 2.00032805, 2.99978457]) # may vary

    """
    return pu._fit(hermevander, x, y, deg, rcond, full, w)


def hermecompanion(c):
    """
    Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is an HermiteE basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of HermiteE series coefficients ordered from low to high
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
    scl = np.hstack((1., 1. / np.sqrt(np.arange(n - 1, 0, -1))))
    scl = np.multiply.accumulate(scl)[::-1]
    top = mat.reshape(-1)[1::n + 1]
    bot = mat.reshape(-1)[n::n + 1]
    top[...] = np.sqrt(np.arange(1, n))
    bot[...] = top
    mat[:, -1] -= scl * c[:-1] / c[-1]
    return mat


def hermeroots(c):
    """
    Compute the roots of a HermiteE series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * He_i(x).

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
    numpy.polynomial.chebyshev.chebroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    The HermiteE series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> from numpy.polynomial.hermite_e import hermeroots, hermefromroots
    >>> coef = hermefromroots([-1, 0, 1])
    >>> coef
    array([0., 2., 0., 1.])
    >>> hermeroots(coef)
    array([-1.,  0.,  1.]) # may vary

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) <= 1:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([-c[0] / c[1]])

    # rotated companion matrix reduces error
    m = hermecompanion(c)[::-1, ::-1]
    r = la.eigvals(m)
    r.sort()
    return r


def _normed_hermite_e_n(x, n):
    """
    Evaluate a normalized HermiteE polynomial.

    Compute the value of the normalized HermiteE polynomial of degree ``n``
    at the points ``x``.


    Parameters
    ----------
    x : ndarray of double.
        Points at which to evaluate the function
    n : int
        Degree of the normalized HermiteE function to be evaluated.

    Returns
    -------
    values : ndarray
        The shape of the return value is described above.

    Notes
    -----
    This function is needed for finding the Gauss points and integration
    weights for high degrees. The values of the standard HermiteE functions
    overflow when n >= 207.

    """
    if n == 0:
        return np.full(x.shape, 1 / np.sqrt(np.sqrt(2 * np.pi)))

    c0 = 0.
    c1 = 1. / np.sqrt(np.sqrt(2 * np.pi))
    nd = float(n)
    for i in range(n - 1):
        tmp = c0
        c0 = -c1 * np.sqrt((nd - 1.) / nd)
        c1 = tmp + c1 * x * np.sqrt(1. / nd)
        nd = nd - 1.0
    return c0 + c1 * x


def hermegauss(deg):
    """
    Gauss-HermiteE quadrature.

    Computes the sample points and weights for Gauss-HermiteE quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-\\inf, \\inf]`
    with the weight function :math:`f(x) = \\exp(-x^2/2)`.

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

    .. math:: w_k = c / (He'_n(x_k) * He_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`He_n`, and then scaling the results to get
    the right value when integrating 1.

    """
    ideg = pu._as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    # first approximation of roots. We use the fact that the companion
    # matrix is symmetric in this case in order to obtain better zeros.
    c = np.array([0] * deg + [1])
    m = hermecompanion(c)
    x = la.eigvalsh(m)

    # improve roots by one application of Newton
    dy = _normed_hermite_e_n(x, ideg)
    df = _normed_hermite_e_n(x, ideg - 1) * np.sqrt(ideg)
    x -= dy / df

    # compute the weights. We scale the factor to avoid possible numerical
    # overflow.
    fm = _normed_hermite_e_n(x, ideg - 1)
    fm /= np.abs(fm).max()
    w = 1 / (fm * fm)

    # for Hermite_e we can also symmetrize
    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    # scale w to get the right value
    w *= np.sqrt(2 * np.pi) / w.sum()

    return x, w


def hermeweight(x):
    """Weight function of the Hermite_e polynomials.

    The weight function is :math:`\\exp(-x^2/2)` and the interval of
    integration is :math:`[-\\inf, \\inf]`. the HermiteE polynomials are
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
    w = np.exp(-.5 * x**2)
    return w


#
# HermiteE series class
#

class HermiteE(ABCPolyBase):
    """An HermiteE series class.

    The HermiteE class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        HermiteE coefficients in order of increasing degree, i.e,
        ``(1, 2, 3)`` gives ``1*He_0(x) + 2*He_1(X) + 3*He_2(x)``.
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
    _add = staticmethod(hermeadd)
    _sub = staticmethod(hermesub)
    _mul = staticmethod(hermemul)
    _div = staticmethod(hermediv)
    _pow = staticmethod(hermepow)
    _val = staticmethod(hermeval)
    _int = staticmethod(hermeint)
    _der = staticmethod(hermeder)
    _fit = staticmethod(hermefit)
    _line = staticmethod(hermeline)
    _roots = staticmethod(hermeroots)
    _fromroots = staticmethod(hermefromroots)

    # Virtual properties
    domain = np.array(hermedomain)
    window = np.array(hermedomain)
    basis_name = 'He'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\integrals.py ===
from __future__ import annotations

from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import diff
from sympy.core.logic import fuzzy_bool
from sympy.core.mul import Mul
from sympy.core.numbers import oo, pi
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.core.sympify import sympify
from sympy.functions import Piecewise, sqrt, piecewise_fold, tan, cot, atan
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.special.singularity_functions import Heaviside
from .rationaltools import ratint
from sympy.matrices import MatrixBase
from sympy.polys import Poly, PolynomialError
from sympy.series.formal import FormalPowerSeries
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.tensor.functions import shape
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent


class Integral(AddWithLimits):
    """Represents unevaluated integral."""

    __slots__ = ()

    args: tuple[Expr, Tuple] # type: ignore

    def __new__(cls, function, *symbols, **assumptions) -> Integral:
        """Create an unevaluated integral.

        Explanation
        ===========

        Arguments are an integrand followed by one or more limits.

        If no limits are given and there is only one free symbol in the
        expression, that symbol will be used, otherwise an error will be
        raised.

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> Integral(x)
        Integral(x, x)
        >>> Integral(y)
        Integral(y, y)

        When limits are provided, they are interpreted as follows (using
        ``x`` as though it were the variable of integration):

            (x,) or x - indefinite integral
            (x, a) - "evaluate at" integral is an abstract antiderivative
            (x, a, b) - definite integral

        The ``as_dummy`` method can be used to see which symbols cannot be
        targeted by subs: those with a prepended underscore cannot be
        changed with ``subs``. (Also, the integration variables themselves --
        the first element of a limit -- can never be changed by subs.)

        >>> i = Integral(x, x)
        >>> at = Integral(x, (x, x))
        >>> i.as_dummy()
        Integral(x, x)
        >>> at.as_dummy()
        Integral(_0, (_0, x))

        """

        #This will help other classes define their own definitions
        #of behaviour with Integral.
        if hasattr(function, '_eval_Integral'):
            return function._eval_Integral(*symbols, **assumptions)

        if isinstance(function, Poly):
            sympy_deprecation_warning(
                """
                integrate(Poly) and Integral(Poly) are deprecated. Instead,
                use the Poly.integrate() method, or convert the Poly to an
                Expr first with the Poly.as_expr() method.
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-integrate-poly")

        obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
        return obj

    def __getnewargs__(self):
        return (self.function,) + tuple([tuple(xab) for xab in self.limits])

    @property
    def free_symbols(self):
        """
        This method returns the symbols that will exist when the
        integral is evaluated. This is useful if one is trying to
        determine whether an integral depends on a certain
        symbol or not.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> Integral(x, (x, y, 1)).free_symbols
        {y}

        See Also
        ========

        sympy.concrete.expr_with_limits.ExprWithLimits.function
        sympy.concrete.expr_with_limits.ExprWithLimits.limits
        sympy.concrete.expr_with_limits.ExprWithLimits.variables
        """
        return super().free_symbols

    def _eval_is_zero(self):
        # This is a very naive and quick test, not intended to do the integral to
        # answer whether it is zero or not, e.g. Integral(sin(x), (x, 0, 2*pi))
        # is zero but this routine should return None for that case. But, like
        # Mul, there are trivial situations for which the integral will be
        # zero so we check for those.
        if self.function.is_zero:
            return True
        got_none = False
        for l in self.limits:
            if len(l) == 3:
                z = (l[1] == l[2]) or (l[1] - l[2]).is_zero
                if z:
                    return True
                elif z is None:
                    got_none = True
        free = self.function.free_symbols
        for xab in self.limits:
            if len(xab) == 1:
                free.add(xab[0])
                continue
            if len(xab) == 2 and xab[0] not in free:
                if xab[1].is_zero:
                    return True
                elif xab[1].is_zero is None:
                    got_none = True
            # take integration symbol out of free since it will be replaced
            # with the free symbols in the limits
            free.discard(xab[0])
            # add in the new symbols
            for i in xab[1:]:
                free.update(i.free_symbols)
        if self.function.is_zero is False and got_none is False:
            return False

    def transform(self, x, u):
        r"""
        Performs a change of variables from `x` to `u` using the relationship
        given by `x` and `u` which will define the transformations `f` and `F`
        (which are inverses of each other) as follows:

        1) If `x` is a Symbol (which is a variable of integration) then `u`
           will be interpreted as some function, f(u), with inverse F(u).
           This, in effect, just makes the substitution of x with f(x).

        2) If `u` is a Symbol then `x` will be interpreted as some function,
           F(x), with inverse f(u). This is commonly referred to as
           u-substitution.

        Once f and F have been identified, the transformation is made as
        follows:

        .. math:: \int_a^b x \mathrm{d}x \rightarrow \int_{F(a)}^{F(b)} f(x)
                  \frac{\mathrm{d}}{\mathrm{d}x}

        where `F(x)` is the inverse of `f(x)` and the limits and integrand have
        been corrected so as to retain the same value after integration.

        Notes
        =====

        The mappings, F(x) or f(u), must lead to a unique integral. Linear
        or rational linear expression, ``2*x``, ``1/x`` and ``sqrt(x)``, will
        always work; quadratic expressions like ``x**2 - 1`` are acceptable
        as long as the resulting integrand does not depend on the sign of
        the solutions (see examples).

        The integral will be returned unchanged if ``x`` is not a variable of
        integration.

        ``x`` must be (or contain) only one of of the integration variables. If
        ``u`` has more than one free symbol then it should be sent as a tuple
        (``u``, ``uvar``) where ``uvar`` identifies which variable is replacing
        the integration variable.
        XXX can it contain another integration variable?

        Examples
        ========

        >>> from sympy.abc import a, x, u
        >>> from sympy import Integral, cos, sqrt

        >>> i = Integral(x*cos(x**2 - 1), (x, 0, 1))

        transform can change the variable of integration

        >>> i.transform(x, u)
        Integral(u*cos(u**2 - 1), (u, 0, 1))

        transform can perform u-substitution as long as a unique
        integrand is obtained:

        >>> ui = i.transform(x**2 - 1, u)
        >>> ui
        Integral(cos(u)/2, (u, -1, 0))

        This attempt fails because x = +/-sqrt(u + 1) and the
        sign does not cancel out of the integrand:

        >>> Integral(cos(x**2 - 1), (x, 0, 1)).transform(x**2 - 1, u)
        Traceback (most recent call last):
        ...
        ValueError:
        The mapping between F(x) and f(u) did not give a unique integrand.

        transform can do a substitution. Here, the previous
        result is transformed back into the original expression
        using "u-substitution":

        >>> ui.transform(sqrt(u + 1), x) == i
        True

        We can accomplish the same with a regular substitution:

        >>> ui.transform(u, x**2 - 1) == i
        True

        If the `x` does not contain a symbol of integration then
        the integral will be returned unchanged. Integral `i` does
        not have an integration variable `a` so no change is made:

        >>> i.transform(a, x) == i
        True

        When `u` has more than one free symbol the symbol that is
        replacing `x` must be identified by passing `u` as a tuple:

        >>> Integral(x, (x, 0, 1)).transform(x, (u + a, u))
        Integral(a + u, (u, -a, 1 - a))
        >>> Integral(x, (x, 0, 1)).transform(x, (u + a, a))
        Integral(a + u, (a, -u, 1 - u))

        See Also
        ========

        sympy.concrete.expr_with_limits.ExprWithLimits.variables : Lists the integration variables
        as_dummy : Replace integration variables with dummy ones
        """
        d = Dummy('d')

        xfree = x.free_symbols.intersection(self.variables)
        if len(xfree) > 1:
            raise ValueError(
                'F(x) can only contain one of: %s' % self.variables)
        xvar = xfree.pop() if xfree else d

        if xvar not in self.variables:
            return self

        u = sympify(u)
        if isinstance(u, Expr):
            ufree = u.free_symbols
            if len(ufree) == 0:
                raise ValueError(filldedent('''
                f(u) cannot be a constant'''))
            if len(ufree) > 1:
                raise ValueError(filldedent('''
                When f(u) has more than one free symbol, the one replacing x
                must be identified: pass f(u) as (f(u), u)'''))
            uvar = ufree.pop()
        else:
            u, uvar = u
            if uvar not in u.free_symbols:
                raise ValueError(filldedent('''
                Expecting a tuple (expr, symbol) where symbol identified
                a free symbol in expr, but symbol is not in expr's free
                symbols.'''))
            if not isinstance(uvar, Symbol):
                # This probably never evaluates to True
                raise ValueError(filldedent('''
                Expecting a tuple (expr, symbol) but didn't get
                a symbol; got %s''' % uvar))

        if x.is_Symbol and u.is_Symbol:
            return self.xreplace({x: u})

        if not x.is_Symbol and not u.is_Symbol:
            raise ValueError('either x or u must be a symbol')

        if uvar == xvar:
            return self.transform(x, (u.subs(uvar, d), d)).xreplace({d: uvar})

        if uvar in self.limits:
            raise ValueError(filldedent('''
            u must contain the same variable as in x
            or a variable that is not already an integration variable'''))

        from sympy.solvers.solvers import solve
        if not x.is_Symbol:
            F = [x.subs(xvar, d)]
            soln = solve(u - x, xvar, check=False)
            if not soln:
                raise ValueError('no solution for solve(F(x) - f(u), x)')
            f = [fi.subs(uvar, d) for fi in soln]
        else:
            f = [u.subs(uvar, d)]
            from sympy.simplify.simplify import posify
            pdiff, reps = posify(u - x)
            puvar = uvar.subs([(v, k) for k, v in reps.items()])
            soln = [s.subs(reps) for s in solve(pdiff, puvar)]
            if not soln:
                raise ValueError('no solution for solve(F(x) - f(u), u)')
            F = [fi.subs(xvar, d) for fi in soln]

        newfuncs = {(self.function.subs(xvar, fi)*fi.diff(d)
                        ).subs(d, uvar) for fi in f}
        if len(newfuncs) > 1:
            raise ValueError(filldedent('''
            The mapping between F(x) and f(u) did not give
            a unique integrand.'''))
        newfunc = newfuncs.pop()

        def _calc_limit_1(F, a, b):
            """
            replace d with a, using subs if possible, otherwise limit
            where sign of b is considered
            """
            wok = F.subs(d, a)
            if wok is S.NaN or wok.is_finite is False and a.is_finite:
                return limit(sign(b)*F, d, a)
            return wok

        def _calc_limit(a, b):
            """
            replace d with a, using subs if possible, otherwise limit
            where sign of b is considered
            """
            avals = list({_calc_limit_1(Fi, a, b) for Fi in F})
            if len(avals) > 1:
                raise ValueError(filldedent('''
                The mapping between F(x) and f(u) did not
                give a unique limit.'''))
            return avals[0]

        newlimits = []
        for xab in self.limits:
            sym = xab[0]
            if sym == xvar:
                if len(xab) == 3:
                    a, b = xab[1:]
                    a, b = _calc_limit(a, b), _calc_limit(b, a)
                    if fuzzy_bool(a - b > 0):
                        a, b = b, a
                        newfunc = -newfunc
                    newlimits.append((uvar, a, b))
                elif len(xab) == 2:
                    a = _calc_limit(xab[1], 1)
                    newlimits.append((uvar, a))
                else:
                    newlimits.append(uvar)
            else:
                newlimits.append(xab)

        return self.func(newfunc, *newlimits)

    def doit(self, **hints):
        """
        Perform the integration using any hints given.

        Examples
        ========

        >>> from sympy import Piecewise, S
        >>> from sympy.abc import x, t
        >>> p = x**2 + Piecewise((0, x/t < 0), (1, True))
        >>> p.integrate((t, S(4)/5, 1), (x, -1, 1))
        1/3

        See Also
        ========

        sympy.integrals.trigonometry.trigintegrate
        sympy.integrals.heurisch.heurisch
        sympy.integrals.rationaltools.ratint
        as_sum : Approximate the integral using a sum
        """
        if not hints.get('integrals', True):
            return self

        deep = hints.get('deep', True)
        meijerg = hints.get('meijerg', None)
        conds = hints.get('conds', 'piecewise')
        risch = hints.get('risch', None)
        heurisch = hints.get('heurisch', None)
        manual = hints.get('manual', None)
        if len(list(filter(None, (manual, meijerg, risch, heurisch)))) > 1:
            raise ValueError("At most one of manual, meijerg, risch, heurisch can be True")
        elif manual:
            meijerg = risch = heurisch = False
        elif meijerg:
            manual = risch = heurisch = False
        elif risch:
            manual = meijerg = heurisch = False
        elif heurisch:
            manual = meijerg = risch = False
        eval_kwargs = {"meijerg": meijerg, "risch": risch, "manual": manual, "heurisch": heurisch,
            "conds": conds}

        if conds not in ('separate', 'piecewise', 'none'):
            raise ValueError('conds must be one of "separate", "piecewise", '
                             '"none", got: %s' % conds)

        if risch and any(len(xab) > 1 for xab in self.limits):
            raise ValueError('risch=True is only allowed for indefinite integrals.')

        # check for the trivial zero
        if self.is_zero:
            return S.Zero

        # hacks to handle integrals of
        # nested summations
        from sympy.concrete.summations import Sum
        if isinstance(self.function, Sum):
            if any(v in self.function.limits[0] for v in self.variables):
                raise ValueError('Limit of the sum cannot be an integration variable.')
            if any(l.is_infinite for l in self.function.limits[0][1:]):
                return self
            _i = self
            _sum = self.function
            return _sum.func(_i.func(_sum.function, *_i.limits).doit(), *_sum.limits).doit()

        # now compute and check the function
        function = self.function

        # hack to use a consistent Heaviside(x, 1/2)
        function = function.replace(
            lambda x: isinstance(x, Heaviside) and x.args[1]*2 != 1,
            lambda x: Heaviside(x.args[0]))

        if deep:
            function = function.doit(**hints)
        if function.is_zero:
            return S.Zero

        # hacks to handle special cases
        if isinstance(function, MatrixBase):
            return function.applyfunc(
                lambda f: self.func(f, *self.limits).doit(**hints))

        if isinstance(function, FormalPowerSeries):
            if len(self.limits) > 1:
                raise NotImplementedError
            xab = self.limits[0]
            if len(xab) > 1:
                return function.integrate(xab, **eval_kwargs)
            else:
                return function.integrate(xab[0], **eval_kwargs)

        # There is no trivial answer and special handling
        # is done so continue

        # first make sure any definite limits have integration
        # variables with matching assumptions
        reps = {}
        for xab in self.limits:
            if len(xab) != 3:
                # it makes sense to just make
                # all x real but in practice with the
                # current state of integration...this
                # doesn't work out well
                # x = xab[0]
                # if x not in reps and not x.is_real:
                #     reps[x] = Dummy(real=True)
                continue
            x, a, b = xab
            l = (a, b)
            if all(i.is_nonnegative for i in l) and not x.is_nonnegative:
                d = Dummy(positive=True)
            elif all(i.is_nonpositive for i in l) and not x.is_nonpositive:
                d = Dummy(negative=True)
            elif all(i.is_real for i in l) and not x.is_real:
                d = Dummy(real=True)
            else:
                d = None
            if d:
                reps[x] = d
        if reps:
            undo = {v: k for k, v in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):  # when separate=True
                did = tuple([i.xreplace(undo) for i in did])
            else:
                did = did.xreplace(undo)
            return did

        # continue with existing assumptions
        undone_limits = []
        # ulj = free symbols of any undone limits' upper and lower limits
        ulj = set()
        for xab in self.limits:
            # compute uli, the free symbols in the
            # Upper and Lower limits of limit I
            if len(xab) == 1:
                uli = set(xab[:1])
            elif len(xab) == 2:
                uli = xab[1].free_symbols
            elif len(xab) == 3:
                uli = xab[1].free_symbols.union(xab[2].free_symbols)
            # this integral can be done as long as there is no blocking
            # limit that has been undone. An undone limit is blocking if
            # it contains an integration variable that is in this limit's
            # upper or lower free symbols or vice versa
            if xab[0] in ulj or any(v[0] in uli for v in undone_limits):
                undone_limits.append(xab)
                ulj.update(uli)
                function = self.func(*([function] + [xab]))
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue

            if function.has(Abs, sign) and (
                (len(xab) < 3 and all(x.is_extended_real for x in xab)) or
                (len(xab) == 3 and all(x.is_extended_real and not x.is_infinite for
                 x in xab[1:]))):
                    # some improper integrals are better off with Abs
                    xr = Dummy("xr", real=True)
                    function = (function.xreplace({xab[0]: xr})
                        .rewrite(Piecewise).xreplace({xr: xab[0]}))
            elif function.has(Min, Max):
                function = function.rewrite(Piecewise)
            if (function.has(Piecewise) and
                not isinstance(function, Piecewise)):
                    function = piecewise_fold(function)
            if isinstance(function, Piecewise):
                if len(xab) == 1:
                    antideriv = function._eval_integral(xab[0],
                        **eval_kwargs)
                else:
                    antideriv = self._eval_integral(
                        function, xab[0], **eval_kwargs)
            else:
                # There are a number of tradeoffs in using the
                # Meijer G method. It can sometimes be a lot faster
                # than other methods, and sometimes slower. And
                # there are certain types of integrals for which it
                # is more likely to work than others. These
                # heuristics are incorporated in deciding what
                # integration methods to try, in what order. See the
                # integrate() docstring for details.
                def try_meijerg(function, xab):
                    ret = None
                    if len(xab) == 3 and meijerg is not False:
                        x, a, b = xab
                        try:
                            res = meijerint_definite(function, x, a, b)
                        except NotImplementedError:
                            _debug('NotImplementedError '
                                'from meijerint_definite')
                            res = None
                        if res is not None:
                            f, cond = res
                            if conds == 'piecewise':
                                u = self.func(function, (x, a, b))
                                # if Piecewise modifies cond too
                                # much it may not be recognized by
                                # _condsimp pattern matching so just
                                # turn off all evaluation
                                return Piecewise((f, cond), (u, True),
                                    evaluate=False)
                            elif conds == 'separate':
                                if len(self.limits) != 1:
                                    raise ValueError(filldedent('''
                                        conds=separate not supported in
                                        multiple integrals'''))
                                ret = f, cond
                            else:
                                ret = f
                    return ret

                meijerg1 = meijerg
                if (meijerg is not False and
                        len(xab) == 3 and xab[1].is_extended_real and xab[2].is_extended_real
                        and not function.is_Poly and
                        (xab[1].has(oo, -oo) or xab[2].has(oo, -oo))):
                    ret = try_meijerg(function, xab)
                    if ret is not None:
                        function = ret
                        continue
                    meijerg1 = False
                # If the special meijerg code did not succeed in
                # finding a definite integral, then the code using
                # meijerint_indefinite will not either (it might
                # find an antiderivative, but the answer is likely
                # to be nonsensical). Thus if we are requested to
                # only use Meijer G-function methods, we give up at
                # this stage. Otherwise we just disable G-function
                # methods.
                if meijerg1 is False and meijerg is True:
                    antideriv = None
                else:
                    antideriv = self._eval_integral(
                        function, xab[0], **eval_kwargs)
                    if antideriv is None and meijerg is True:
                        ret = try_meijerg(function, xab)
                        if ret is not None:
                            function = ret
                            continue

            final = hints.get('final', True)
            # dotit may be iterated but floor terms making atan and acot
            # continuous should only be added in the final round
            if (final and not isinstance(antideriv, Integral) and
                antideriv is not None):
                for atan_term in antideriv.atoms(atan):
                    atan_arg = atan_term.args[0]
                    # Checking `atan_arg` to be linear combination of `tan` or `cot`
                    for tan_part in atan_arg.atoms(tan):
                        x1 = Dummy('x1')
                        tan_exp1 = atan_arg.subs(tan_part, x1)
                        # The coefficient of `tan` should be constant
                        coeff = tan_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = tan_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term,
                                sign(coeff)*pi*floor((a-pi/2)/pi)))
                    for cot_part in atan_arg.atoms(cot):
                        x1 = Dummy('x1')
                        cot_exp1 = atan_arg.subs(cot_part, x1)
                        # The coefficient of `cot` should be constant
                        coeff = cot_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = cot_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term,
                                sign(coeff)*pi*floor((a)/pi)))

            if antideriv is None:
                undone_limits.append(xab)
                function = self.func(*([function] + [xab])).factor()
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue
            else:
                if len(xab) == 1:
                    function = antideriv
                else:
                    if len(xab) == 3:
                        x, a, b = xab
                    elif len(xab) == 2:
                        x, b = xab
                        a = None
                    else:
                        raise NotImplementedError

                    if deep:
                        if isinstance(a, Basic):
                            a = a.doit(**hints)
                        if isinstance(b, Basic):
                            b = b.doit(**hints)

                    if antideriv.is_Poly:
                        gens = list(antideriv.gens)
                        gens.remove(x)

                        antideriv = antideriv.as_expr()

                        function = antideriv._eval_interval(x, a, b)
                        function = Poly(function, *gens)
                    else:
                        def is_indef_int(g, x):
                            return (isinstance(g, Integral) and
                                    any(i == (x,) for i in g.limits))

                        def eval_factored(f, x, a, b):
                            # _eval_interval for integrals with
                            # (constant) factors
                            # a single indefinite integral is assumed
                            args = []
                            for g in Mul.make_args(f):
                                if is_indef_int(g, x):
                                    args.append(g._eval_interval(x, a, b))
                                else:
                                    args.append(g)
                            return Mul(*args)

                        integrals, others, piecewises = [], [], []
                        for f in Add.make_args(antideriv):
                            if any(is_indef_int(g, x)
                                   for g in Mul.make_args(f)):
                                integrals.append(f)
                            elif any(isinstance(g, Piecewise)
                                     for g in Mul.make_args(f)):
                                piecewises.append(piecewise_fold(f))
                            else:
                                others.append(f)
                        uneval = Add(*[eval_factored(f, x, a, b)
                                       for f in integrals])
                        try:
                            evalued = Add(*others)._eval_interval(x, a, b)
                            evalued_pw = piecewise_fold(Add(*piecewises))._eval_interval(x, a, b)
                            function = uneval + evalued + evalued_pw
                        except NotImplementedError:
                            # This can happen if _eval_interval depends in a
                            # complicated way on limits that cannot be computed
                            undone_limits.append(xab)
                            function = self.func(*([function] + [xab]))
                            factored_function = function.factor()
                            if not isinstance(factored_function, Integral):
                                function = factored_function
        return function

    def _eval_derivative(self, sym):
        """Evaluate the derivative of the current Integral object by
        differentiating under the integral sign [1], using the Fundamental
        Theorem of Calculus [2] when possible.

        Explanation
        ===========

        Whenever an Integral is encountered that is equivalent to zero or
        has an integrand that is independent of the variable of integration
        those integrals are performed. All others are returned as Integral
        instances which can be resolved with doit() (provided they are integrable).

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Differentiation_under_the_integral_sign
        .. [2] https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> i = Integral(x + y, y, (y, 1, x))
        >>> i.diff(x)
        Integral(x + y, (y, x)) + Integral(1, y, (y, 1, x))
        >>> i.doit().diff(x) == i.diff(x).doit()
        True
        >>> i.diff(y)
        0

        The previous must be true since there is no y in the evaluated integral:

        >>> i.free_symbols
        {x}
        >>> i.doit()
        2*x**3/3 - x/2 - 1/6

        """

        # differentiate under the integral sign; we do not
        # check for regularity conditions (TODO), see issue 4215

        # get limits and the function
        f, limits = self.function, list(self.limits)

        # the order matters if variables of integration appear in the limits
        # so work our way in from the outside to the inside.
        limit = limits.pop(-1)
        if len(limit) == 3:
            x, a, b = limit
        elif len(limit) == 2:
            x, b = limit
            a = None
        else:
            a = b = None
            x = limit[0]

        if limits:  # f is the argument to an integral
            f = self.func(f, *tuple(limits))

        # assemble the pieces
        def _do(f, ab):
            dab_dsym = diff(ab, sym)
            if not dab_dsym:
                return S.Zero
            if isinstance(f, Integral):
                limits = [(x, x) if (len(l) == 1 and l[0] == x) else l
                          for l in f.limits]
                f = self.func(f.function, *limits)
            return f.subs(x, ab)*dab_dsym

        rv = S.Zero
        if b is not None:
            rv += _do(f, b)
        if a is not None:
            rv -= _do(f, a)
        if len(limit) == 1 and sym == x:
            # the dummy variable *is* also the real-world variable
            arg = f
            rv += arg
        else:
            # the dummy variable might match sym but it's
            # only a dummy and the actual variable is determined
            # by the limits, so mask off the variable of integration
            # while differentiating
            u = Dummy('u')
            arg = f.subs(x, u).diff(sym).subs(u, x)
            if arg:
                rv += self.func(arg, (x, a, b))
        return rv

    def _eval_integral(self, f, x, meijerg=None, risch=None, manual=None,
                       heurisch=None, conds='piecewise',final=None):
        """
        Calculate the anti-derivative to the function f(x).

        Explanation
        ===========

        The following algorithms are applied (roughly in this order):

        1. Simple heuristics (based on pattern matching and integral table):

           - most frequently used functions (e.g. polynomials, products of
             trig functions)

        2. Integration of rational functions:

           - A complete algorithm for integrating rational functions is
             implemented (the Lazard-Rioboo-Trager algorithm).  The algorithm
             also uses the partial fraction decomposition algorithm
             implemented in apart() as a preprocessor to make this process
             faster.  Note that the integral of a rational function is always
             elementary, but in general, it may include a RootSum.

        3. Full Risch algorithm:

           - The Risch algorithm is a complete decision
             procedure for integrating elementary functions, which means that
             given any elementary function, it will either compute an
             elementary antiderivative, or else prove that none exists.
             Currently, part of transcendental case is implemented, meaning
             elementary integrals containing exponentials, logarithms, and
             (soon!) trigonometric functions can be computed.  The algebraic
             case, e.g., functions containing roots, is much more difficult
             and is not implemented yet.

           - If the routine fails (because the integrand is not elementary, or
             because a case is not implemented yet), it continues on to the
             next algorithms below.  If the routine proves that the integrals
             is nonelementary, it still moves on to the algorithms below,
             because we might be able to find a closed-form solution in terms
             of special functions.  If risch=True, however, it will stop here.

        4. The Meijer G-Function algorithm:

           - This algorithm works by first rewriting the integrand in terms of
             very general Meijer G-Function (meijerg in SymPy), integrating
             it, and then rewriting the result back, if possible.  This
             algorithm is particularly powerful for definite integrals (which
             is actually part of a different method of Integral), since it can
             compute closed-form solutions of definite integrals even when no
             closed-form indefinite integral exists.  But it also is capable
             of computing many indefinite integrals as well.

           - Another advantage of this method is that it can use some results
             about the Meijer G-Function to give a result in terms of a
             Piecewise expression, which allows to express conditionally
             convergent integrals.

           - Setting meijerg=True will cause integrate() to use only this
             method.

        5. The "manual integration" algorithm:

           - This algorithm tries to mimic how a person would find an
             antiderivative by hand, for example by looking for a
             substitution or applying integration by parts. This algorithm
             does not handle as many integrands but can return results in a
             more familiar form.

           - Sometimes this algorithm can evaluate parts of an integral; in
             this case integrate() will try to evaluate the rest of the
             integrand using the other methods here.

           - Setting manual=True will cause integrate() to use only this
             method.

        6. The Heuristic Risch algorithm:

           - This is a heuristic version of the Risch algorithm, meaning that
             it is not deterministic.  This is tried as a last resort because
             it can be very slow.  It is still used because not enough of the
             full Risch algorithm is implemented, so that there are still some
             integrals that can only be computed using this method.  The goal
             is to implement enough of the Risch and Meijer G-function methods
             so that this can be deleted.

             Setting heurisch=True will cause integrate() to use only this
             method. Set heurisch=False to not use it.

        """

        from sympy.integrals.risch import risch_integrate, NonElementaryIntegral
        from sympy.integrals.manualintegrate import manualintegrate

        if risch:
            try:
                return risch_integrate(f, x, conds=conds)
            except NotImplementedError:
                return None

        if manual:
            try:
                result = manualintegrate(f, x)
                if result is not None and result.func != Integral:
                    return result
            except (ValueError, PolynomialError):
                pass

        eval_kwargs = {"meijerg": meijerg, "risch": risch, "manual": manual,
            "heurisch": heurisch, "conds": conds}

        # if it is a poly(x) then let the polynomial integrate itself (fast)
        #
        # It is important to make this check first, otherwise the other code
        # will return a SymPy expression instead of a Polynomial.
        #
        # see Polynomial for details.
        if isinstance(f, Poly) and not (manual or meijerg or risch):
            # Note: this is deprecated, but the deprecation warning is already
            # issued in the Integral constructor.
            return f.integrate(x)

        # Piecewise antiderivatives need to call special integrate.
        if isinstance(f, Piecewise):
            return f.piecewise_integrate(x, **eval_kwargs)

        # let's cut it short if `f` does not depend on `x`; if
        # x is only a dummy, that will be handled below
        if not f.has(x):
            return f*x

        # try to convert to poly(x) and then integrate if successful (fast)
        poly = f.as_poly(x)
        if poly is not None and not (manual or meijerg or risch):
            return poly.integrate().as_expr()

        if risch is not False:
            try:
                result, i = risch_integrate(f, x, separate_integral=True,
                    conds=conds)
            except NotImplementedError:
                pass
            else:
                if i:
                    # There was a nonelementary integral. Try integrating it.

                    # if no part of the NonElementaryIntegral is integrated by
                    # the Risch algorithm, then use the original function to
                    # integrate, instead of re-written one
                    if result == 0:
                        return NonElementaryIntegral(f, x).doit(risch=False)
                    else:
                        return result + i.doit(risch=False)
                else:
                    return result

        # since Integral(f=g1+g2+...) == Integral(g1) + Integral(g2) + ...
        # we are going to handle Add terms separately,
        # if `f` is not Add -- we only have one term

        # Note that in general, this is a bad idea, because Integral(g1) +
        # Integral(g2) might not be computable, even if Integral(g1 + g2) is.
        # For example, Integral(x**x + x**x*log(x)).  But many heuristics only
        # work term-wise.  So we compute this step last, after trying
        # risch_integrate.  We also try risch_integrate again in this loop,
        # because maybe the integral is a sum of an elementary part and a
        # nonelementary part (like erf(x) + exp(x)).  risch_integrate() is
        # quite fast, so this is acceptable.
        from sympy.simplify.fu import sincos_to_sum
        parts = []
        args = Add.make_args(f)
        for g in args:
            coeff, g = g.as_independent(x)

            # g(x) = const
            if g is S.One and not meijerg:
                parts.append(coeff*x)
                continue

            # g(x) = expr + O(x**n)
            order_term = g.getO()

            if order_term is not None:
                h = self._eval_integral(g.removeO(), x, **eval_kwargs)

                if h is not None:
                    h_order_expr = self._eval_integral(order_term.expr, x, **eval_kwargs)

                    if h_order_expr is not None:
                        h_order_term = order_term.func(
                            h_order_expr, *order_term.variables)
                        parts.append(coeff*(h + h_order_term))
                        continue

                # NOTE: if there is O(x**n) and we fail to integrate then
                # there is no point in trying other methods because they
                # will fail, too.
                return None

            #               c
            # g(x) = (a*x+b)
            if g.is_Pow and not g.exp.has(x) and not meijerg:
                a = Wild('a', exclude=[x])
                b = Wild('b', exclude=[x])

                M = g.base.match(a*x + b)

                if M is not None:
                    if g.exp == -1:
                        h = log(g.base)
                    elif conds != 'piecewise':
                        h = g.base**(g.exp + 1) / (g.exp + 1)
                    else:
                        h1 = log(g.base)
                        h2 = g.base**(g.exp + 1) / (g.exp + 1)
                        h = Piecewise((h2, Ne(g.exp, -1)), (h1, True))

                    parts.append(coeff * h / M[a])
                    continue

            #        poly(x)
            # g(x) = -------
            #        poly(x)
            if g.is_rational_function(x) and not (manual or meijerg or risch):
                parts.append(coeff * ratint(g, x))
                continue

            if not (manual or meijerg or risch):
                # g(x) = Mul(trig)
                h = trigintegrate(g, x, conds=conds)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # g(x) has at least a DiracDelta term
                h = deltaintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                from .singularityfunctions import singularityintegrate
                # g(x) has at least a Singularity Function term
                h = singularityintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue

                # Try risch again.
                if risch is not False:
                    try:
                        h, i = risch_integrate(g, x,
                            separate_integral=True, conds=conds)
                    except NotImplementedError:
                        h = None
                    else:
                        if i:
                            h = h + i.doit(risch=False)

                        parts.append(coeff*h)
                        continue

                # fall back to heurisch
                if heurisch is not False:
                    from sympy.integrals.heurisch import (heurisch as heurisch_,
                                                          heurisch_wrapper)
                    try:
                        if conds == 'piecewise':
                            h = heurisch_wrapper(g, x, hints=[])
                        else:
                            h = heurisch_(g, x, hints=[])
                    except PolynomialError:
                        # XXX: this exception means there is a bug in the
                        # implementation of heuristic Risch integration
                        # algorithm.
                        h = None
            else:
                h = None

            if meijerg is not False and h is None:
                # rewrite using G functions
                try:
                    h = meijerint_indefinite(g, x)
                except NotImplementedError:
                    _debug('NotImplementedError from meijerint_definite')
                if h is not None:
                    parts.append(coeff * h)
                    continue

            if h is None and manual is not False:
                try:
                    result = manualintegrate(g, x)
                    if result is not None and not isinstance(result, Integral):
                        if result.has(Integral) and not manual:
                            # Try to have other algorithms do the integrals
                            # manualintegrate can't handle,
                            # unless we were asked to use manual only.
                            # Keep the rest of eval_kwargs in case another
                            # method was set to False already
                            new_eval_kwargs = eval_kwargs
                            new_eval_kwargs["manual"] = False
                            new_eval_kwargs["final"] = False
                            result = result.func(*[
                                arg.doit(**new_eval_kwargs) if
                                arg.has(Integral) else arg
                                for arg in result.args
                            ]).expand(multinomial=False,
                                      log=False,
                                      power_exp=False,
                                      power_base=False)
                        if not result.has(Integral):
                            parts.append(coeff * result)
                            continue
                except (ValueError, PolynomialError):
                    # can't handle some SymPy expressions
                    pass

            # if we failed maybe it was because we had
            # a product that could have been expanded,
            # so let's try an expansion of the whole
            # thing before giving up; we don't try this
            # at the outset because there are things
            # that cannot be solved unless they are
            # NOT expanded e.g., x**x*(1+log(x)). There
            # should probably be a checker somewhere in this
            # routine to look for such cases and try to do
            # collection on the expressions if they are already
            # in an expanded form
            if not h and len(args) == 1:
                f = sincos_to_sum(f).expand(mul=True, deep=False)
                if f.is_Add:
                    # Note: risch will be identical on the expanded
                    # expression, but maybe it will be able to pick out parts,
                    # like x*(exp(x) + erf(x)).
                    return self._eval_integral(f, x, **eval_kwargs)

            if h is not None:
                parts.append(coeff * h)
            else:
                return None

        return Add(*parts)

    def _eval_lseries(self, x, logx=None, cdir=0):
        expr = self.as_dummy()
        symb = x
        for l in expr.limits:
            if x in l[1:]:
                symb = l[0]
                break
        for term in expr.function.lseries(symb, logx):
            yield integrate(term, *expr.limits)

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        symb = x
        for l in self.limits:
            if x in l[1:]:
                symb = l[0]
                break
        terms, order = self.function.nseries(
            x=symb, n=n, logx=logx).as_coeff_add(Order)
        order = [o.subs(symb, x) for o in order]
        return integrate(terms, *self.limits) + Add(*order)*x

    def _eval_as_leading_term(self, x, logx, cdir):
        series_gen = self.args[0].lseries(x)
        for leading_term in series_gen:
            if leading_term != 0:
                break
        return integrate(leading_term, *self.args[1:])

    def _eval_simplify(self, **kwargs):
        expr = factor_terms(self)
        if isinstance(expr, Integral):
            from sympy.simplify.simplify import simplify
            return expr.func(*[simplify(i, **kwargs) for i in expr.args])
        return expr.simplify(**kwargs)

    def as_sum(self, n=None, method="midpoint", evaluate=True):
        """
        Approximates a definite integral by a sum.

        Parameters
        ==========

        n :
            The number of subintervals to use, optional.
        method :
            One of: 'left', 'right', 'midpoint', 'trapezoid'.
        evaluate : bool
            If False, returns an unevaluated Sum expression. The default
            is True, evaluate the sum.

        Notes
        =====

        These methods of approximate integration are described in [1].

        Examples
        ========

        >>> from sympy import Integral, sin, sqrt
        >>> from sympy.abc import x, n
        >>> e = Integral(sin(x), (x, 3, 7))
        >>> e
        Integral(sin(x), (x, 3, 7))

        For demonstration purposes, this interval will only be split into 2
        regions, bounded by [3, 5] and [5, 7].

        The left-hand rule uses function evaluations at the left of each
        interval:

        >>> e.as_sum(2, 'left')
        2*sin(5) + 2*sin(3)

        The midpoint rule uses evaluations at the center of each interval:

        >>> e.as_sum(2, 'midpoint')
        2*sin(4) + 2*sin(6)

        The right-hand rule uses function evaluations at the right of each
        interval:

        >>> e.as_sum(2, 'right')
        2*sin(5) + 2*sin(7)

        The trapezoid rule uses function evaluations on both sides of the
        intervals. This is equivalent to taking the average of the left and
        right hand rule results:

        >>> s = e.as_sum(2, 'trapezoid')
        >>> s
        2*sin(5) + sin(3) + sin(7)
        >>> (e.as_sum(2, 'left') + e.as_sum(2, 'right'))/2 == s
        True

        Here, the discontinuity at x = 0 can be avoided by using the
        midpoint or right-hand method:

        >>> e = Integral(1/sqrt(x), (x, 0, 1))
        >>> e.as_sum(5).n(4)
        1.730
        >>> e.as_sum(10).n(4)
        1.809
        >>> e.doit().n(4)  # the actual value is 2
        2.000

        The left- or trapezoid method will encounter the discontinuity and
        return infinity:

        >>> e.as_sum(5, 'left')
        zoo

        The number of intervals can be symbolic. If omitted, a dummy symbol
        will be used for it.

        >>> e = Integral(x**2, (x, 0, 2))
        >>> e.as_sum(n, 'right').expand()
        8/3 + 4/n + 4/(3*n**2)

        This shows that the midpoint rule is more accurate, as its error
        term decays as the square of n:

        >>> e.as_sum(method='midpoint').expand()
        8/3 - 2/(3*_n**2)

        A symbolic sum is returned with evaluate=False:

        >>> e.as_sum(n, 'midpoint', evaluate=False)
        2*Sum((2*_k/n - 1/n)**2, (_k, 1, n))/n

        See Also
        ========

        Integral.doit : Perform the integration using any hints

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Riemann_sum#Riemann_summation_methods
        """

        from sympy.concrete.summations import Sum
        limits = self.limits
        if len(limits) > 1:
            raise NotImplementedError(
                "Multidimensional midpoint rule not implemented yet")
        else:
            limit = limits[0]
            if (len(limit) != 3 or limit[1].is_finite is False or
                limit[2].is_finite is False):
                raise ValueError("Expecting a definite integral over "
                                  "a finite interval.")
        if n is None:
            n = Dummy('n', integer=True, positive=True)
        else:
            n = sympify(n)
        if (n.is_positive is False or n.is_integer is False or
            n.is_finite is False):
            raise ValueError("n must be a positive integer, got %s" % n)
        x, a, b = limit
        dx = (b - a)/n
        k = Dummy('k', integer=True, positive=True)
        f = self.function

        if method == "left":
            result = dx*Sum(f.subs(x, a + (k-1)*dx), (k, 1, n))
        elif method == "right":
            result = dx*Sum(f.subs(x, a + k*dx), (k, 1, n))
        elif method == "midpoint":
            result = dx*Sum(f.subs(x, a + k*dx - dx/2), (k, 1, n))
        elif method == "trapezoid":
            result = dx*((f.subs(x, a) + f.subs(x, b))/2 +
                Sum(f.subs(x, a + k*dx), (k, 1, n - 1)))
        else:
            raise ValueError("Unknown method %s" % method)
        return result.doit() if evaluate else result

    def principal_value(self, **kwargs):
        """
        Compute the Cauchy Principal Value of the definite integral of a real function in the given interval
        on the real axis.

        Explanation
        ===========

        In mathematics, the Cauchy principal value, is a method for assigning values to certain improper
        integrals which would otherwise be undefined.

        Examples
        ========

        >>> from sympy import Integral, oo
        >>> from sympy.abc import x
        >>> Integral(x+1, (x, -oo, oo)).principal_value()
        oo
        >>> f = 1 / (x**3)
        >>> Integral(f, (x, -oo, oo)).principal_value()
        0
        >>> Integral(f, (x, -10, 10)).principal_value()
        0
        >>> Integral(f, (x, -10, oo)).principal_value() + Integral(f, (x, -oo, 10)).principal_value()
        0

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Cauchy_principal_value
        .. [2] https://mathworld.wolfram.com/CauchyPrincipalValue.html
        """
        if len(self.limits) != 1 or len(list(self.limits[0])) != 3:
            raise ValueError("You need to insert a variable, lower_limit, and upper_limit correctly to calculate "
                             "cauchy's principal value")
        x, a, b = self.limits[0]
        if not (a.is_comparable and b.is_comparable and a <= b):
            raise ValueError("The lower_limit must be smaller than or equal to the upper_limit to calculate "
                             "cauchy's principal value. Also, a and b need to be comparable.")
        if a == b:
            return S.Zero

        from sympy.calculus.singularities import singularities

        r = Dummy('r')
        f = self.function
        singularities_list = [s for s in singularities(f, x) if s.is_comparable and a <= s <= b]
        for i in singularities_list:
            if i in (a, b):
                raise ValueError(
                    'The principal value is not defined in the given interval due to singularity at %d.' % (i))
        F = integrate(f, x, **kwargs)
        if F.has(Integral):
            return self
        if a is -oo and b is oo:
            I = limit(F - F.subs(x, -x), x, oo)
        else:
            I = limit(F, x, b, '-') - limit(F, x, a, '+')
        for s in singularities_list:
            I += limit(((F.subs(x, s - r)) - F.subs(x, s + r)), r, 0, '+')
        return I



def integrate(*args, meijerg=None, conds='piecewise', risch=None, heurisch=None, manual=None, **kwargs):
    """integrate(f, var, ...)

    .. deprecated:: 1.6

       Using ``integrate()`` with :class:`~.Poly` is deprecated. Use
       :meth:`.Poly.integrate` instead. See :ref:`deprecated-integrate-poly`.

    Explanation
    ===========

    Compute definite or indefinite integral of one or more variables
    using Risch-Norman algorithm and table lookup. This procedure is
    able to handle elementary algebraic and transcendental functions
    and also a huge class of special functions, including Airy,
    Bessel, Whittaker and Lambert.

    var can be:

    - a symbol                   -- indefinite integration
    - a tuple (symbol, a)        -- indefinite integration with result
                                    given with ``a`` replacing ``symbol``
    - a tuple (symbol, a, b)     -- definite integration

    Several variables can be specified, in which case the result is
    multiple integration. (If var is omitted and the integrand is
    univariate, the indefinite integral in that variable will be performed.)

    Indefinite integrals are returned without terms that are independent
    of the integration variables. (see examples)

    Definite improper integrals often entail delicate convergence
    conditions. Pass conds='piecewise', 'separate' or 'none' to have
    these returned, respectively, as a Piecewise function, as a separate
    result (i.e. result will be a tuple), or not at all (default is
    'piecewise').

    **Strategy**

    SymPy uses various approaches to definite integration. One method is to
    find an antiderivative for the integrand, and then use the fundamental
    theorem of calculus. Various functions are implemented to integrate
    polynomial, rational and trigonometric functions, and integrands
    containing DiracDelta terms.

    SymPy also implements the part of the Risch algorithm, which is a decision
    procedure for integrating elementary functions, i.e., the algorithm can
    either find an elementary antiderivative, or prove that one does not
    exist.  There is also a (very successful, albeit somewhat slow) general
    implementation of the heuristic Risch algorithm.  This algorithm will
    eventually be phased out as more of the full Risch algorithm is
    implemented. See the docstring of Integral._eval_integral() for more
    details on computing the antiderivative using algebraic methods.

    The option risch=True can be used to use only the (full) Risch algorithm.
    This is useful if you want to know if an elementary function has an
    elementary antiderivative.  If the indefinite Integral returned by this
    function is an instance of NonElementaryIntegral, that means that the
    Risch algorithm has proven that integral to be non-elementary.  Note that
    by default, additional methods (such as the Meijer G method outlined
    below) are tried on these integrals, as they may be expressible in terms
    of special functions, so if you only care about elementary answers, use
    risch=True.  Also note that an unevaluated Integral returned by this
    function is not necessarily a NonElementaryIntegral, even with risch=True,
    as it may just be an indication that the particular part of the Risch
    algorithm needed to integrate that function is not yet implemented.

    Another family of strategies comes from re-writing the integrand in
    terms of so-called Meijer G-functions. Indefinite integrals of a
    single G-function can always be computed, and the definite integral
    of a product of two G-functions can be computed from zero to
    infinity. Various strategies are implemented to rewrite integrands
    as G-functions, and use this information to compute integrals (see
    the ``meijerint`` module).

    The option manual=True can be used to use only an algorithm that tries
    to mimic integration by hand. This algorithm does not handle as many
    integrands as the other algorithms implemented but may return results in
    a more familiar form. The ``manualintegrate`` module has functions that
    return the steps used (see the module docstring for more information).

    In general, the algebraic methods work best for computing
    antiderivatives of (possibly complicated) combinations of elementary
    functions. The G-function methods work best for computing definite
    integrals from zero to infinity of moderately complicated
    combinations of special functions, or indefinite integrals of very
    simple combinations of special functions.

    The strategy employed by the integration code is as follows:

    - If computing a definite integral, and both limits are real,
      and at least one limit is +- oo, try the G-function method of
      definite integration first.

    - Try to find an antiderivative, using all available methods, ordered
      by performance (that is try fastest method first, slowest last; in
      particular polynomial integration is tried first, Meijer
      G-functions second to last, and heuristic Risch last).

    - If still not successful, try G-functions irrespective of the
      limits.

    The option meijerg=True, False, None can be used to, respectively:
    always use G-function methods and no others, never use G-function
    methods, or use all available methods (in order as described above).
    It defaults to None.

    Examples
    ========

    >>> from sympy import integrate, log, exp, oo
    >>> from sympy.abc import a, x, y

    >>> integrate(x*y, x)
    x**2*y/2

    >>> integrate(log(x), x)
    x*log(x) - x

    >>> integrate(log(x), (x, 1, a))
    a*log(a) - a + 1

    >>> integrate(x)
    x**2/2

    Terms that are independent of x are dropped by indefinite integration:

    >>> from sympy import sqrt
    >>> integrate(sqrt(1 + x), (x, 0, x))
    2*(x + 1)**(3/2)/3 - 2/3
    >>> integrate(sqrt(1 + x), x)
    2*(x + 1)**(3/2)/3

    >>> integrate(x*y)
    Traceback (most recent call last):
    ...
    ValueError: specify integration variables to integrate x*y

    Note that ``integrate(x)`` syntax is meant only for convenience
    in interactive sessions and should be avoided in library code.

    >>> integrate(x**a*exp(-x), (x, 0, oo)) # same as conds='piecewise'
    Piecewise((gamma(a + 1), re(a) > -1),
        (Integral(x**a*exp(-x), (x, 0, oo)), True))

    >>> integrate(x**a*exp(-x), (x, 0, oo), conds='none')
    gamma(a + 1)

    >>> integrate(x**a*exp(-x), (x, 0, oo), conds='separate')
    (gamma(a + 1), re(a) > -1)

    See Also
    ========

    Integral, Integral.doit

    """
    doit_flags = {
        'deep': False,
        'meijerg': meijerg,
        'conds': conds,
        'risch': risch,
        'heurisch': heurisch,
        'manual': manual
        }

    integral = Integral(*args, **kwargs)

    if isinstance(integral, Integral):
        return integral.doit(**doit_flags)
    else:
        new_args = [a.doit(**doit_flags) if isinstance(a, Integral) else a
            for a in integral.args]
        return integral.func(*new_args)

def line_integrate(field, curve, vars):
    """line_integrate(field, Curve, variables)

    Compute the line integral.

    Examples
    ========

    >>> from sympy import Curve, line_integrate, E, ln
    >>> from sympy.abc import x, y, t
    >>> C = Curve([E**t + 1, E**t - 1], (t, 0, ln(2)))
    >>> line_integrate(x + y, C, [x, y])
    3*sqrt(2)

    See Also
    ========

    sympy.integrals.integrals.integrate, Integral
    """
    from sympy.geometry import Curve
    F = sympify(field)
    if not F:
        raise ValueError(
            "Expecting function specifying field as first argument.")
    if not isinstance(curve, Curve):
        raise ValueError("Expecting Curve entity as second argument.")
    if not is_sequence(vars):
        raise ValueError("Expecting ordered iterable for variables.")
    if len(curve.functions) != len(vars):
        raise ValueError("Field variable size does not match curve dimension.")

    if curve.parameter in vars:
        raise ValueError("Curve parameter clashes with field parameters.")

    # Calculate derivatives for line parameter functions
    # F(r) -> F(r(t)) and finally F(r(t)*r'(t))
    Ft = F
    dldt = 0
    for i, var in enumerate(vars):
        _f = curve.functions[i]
        _dn = diff(_f, curve.parameter)
        # ...arc length
        dldt = dldt + (_dn * _dn)
        Ft = Ft.subs(var, _f)
    Ft = Ft * sqrt(dldt)

    integral = Integral(Ft, curve.limits).doit(deep=False)
    return integral


### Property function dispatching ###

@shape.register(Integral)
def _(expr):
    return shape(expr.function)

# Delayed imports
from .deltafunctions import deltaintegrate
from .meijerint import meijerint_definite, meijerint_indefinite, _debug
from .trigonometry import trigintegrate

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\webdriver.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""The WebDriver implementation."""

import base64
import contextlib
import copy
import os
import pkgutil
import tempfile
import types
import warnings
import zipfile
from abc import ABCMeta
from base64 import b64decode, urlsafe_b64encode
from contextlib import asynccontextmanager, contextmanager
from importlib import import_module
from typing import Any, Dict, List, Optional, Type, Union

from selenium.common.exceptions import (
    InvalidArgumentException,
    JavascriptException,
    NoSuchCookieException,
    NoSuchElementException,
    WebDriverException,
)
from selenium.webdriver.common.bidi.browser import Browser
from selenium.webdriver.common.bidi.browsing_context import BrowsingContext
from selenium.webdriver.common.bidi.network import Network
from selenium.webdriver.common.bidi.script import Script
from selenium.webdriver.common.bidi.session import Session
from selenium.webdriver.common.bidi.storage import Storage
from selenium.webdriver.common.bidi.webextension import WebExtension
from selenium.webdriver.common.by import By
from selenium.webdriver.common.options import ArgOptions, BaseOptions
from selenium.webdriver.common.print_page_options import PrintOptions
from selenium.webdriver.common.timeouts import Timeouts
from selenium.webdriver.common.virtual_authenticator import (
    Credential,
    VirtualAuthenticatorOptions,
    required_virtual_authenticator,
)
from selenium.webdriver.support.relative_locator import RelativeBy

from ..common.fedcm.dialog import Dialog
from .bidi_connection import BidiConnection
from .client_config import ClientConfig
from .command import Command
from .errorhandler import ErrorHandler
from .fedcm import FedCM
from .file_detector import FileDetector, LocalFileDetector
from .locator_converter import LocatorConverter
from .mobile import Mobile
from .remote_connection import RemoteConnection
from .script_key import ScriptKey
from .shadowroot import ShadowRoot
from .switch_to import SwitchTo
from .webelement import WebElement
from .websocket_connection import WebSocketConnection

cdp = None
devtools = None


def import_cdp():
    global cdp
    if not cdp:
        cdp = import_module("selenium.webdriver.common.bidi.cdp")


def _create_caps(caps):
    """Makes a W3C alwaysMatch capabilities object.

    Filters out capability names that are not in the W3C spec. Spec-compliant
    drivers will reject requests containing unknown capability names.

    Moves the Firefox profile, if present, from the old location to the new Firefox
    options object.

    Parameters:
    -----------
    caps : dict
        - A dictionary of capabilities requested by the caller.
    """
    caps = copy.deepcopy(caps)
    always_match = {}
    for k, v in caps.items():
        always_match[k] = v
    return {"capabilities": {"firstMatch": [{}], "alwaysMatch": always_match}}


def get_remote_connection(
    capabilities: dict,
    command_executor: Union[str, RemoteConnection],
    keep_alive: bool,
    ignore_local_proxy: bool,
    client_config: Optional[ClientConfig] = None,
) -> RemoteConnection:
    if isinstance(command_executor, str):
        client_config = client_config or ClientConfig(remote_server_addr=command_executor)
        client_config.remote_server_addr = command_executor
        command_executor = RemoteConnection(client_config=client_config)
    from selenium.webdriver.chrome.remote_connection import ChromeRemoteConnection
    from selenium.webdriver.edge.remote_connection import EdgeRemoteConnection
    from selenium.webdriver.firefox.remote_connection import FirefoxRemoteConnection
    from selenium.webdriver.safari.remote_connection import SafariRemoteConnection

    candidates = [ChromeRemoteConnection, EdgeRemoteConnection, SafariRemoteConnection, FirefoxRemoteConnection]
    handler = next((c for c in candidates if c.browser_name == capabilities.get("browserName")), RemoteConnection)

    return handler(
        remote_server_addr=command_executor,
        keep_alive=keep_alive,
        ignore_proxy=ignore_local_proxy,
        client_config=client_config,
    )


def create_matches(options: List[BaseOptions]) -> Dict:
    capabilities = {"capabilities": {}}
    opts = []
    for opt in options:
        opts.append(opt.to_capabilities())
    opts_size = len(opts)
    samesies = {}

    # Can not use bitwise operations on the dicts or lists due to
    # https://bugs.python.org/issue38210
    for i in range(opts_size):
        min_index = i
        if i + 1 < opts_size:
            first_keys = opts[min_index].keys()

            for kys in first_keys:
                if kys in opts[i + 1].keys():
                    if opts[min_index][kys] == opts[i + 1][kys]:
                        samesies.update({kys: opts[min_index][kys]})

    always = {}
    for k, v in samesies.items():
        always[k] = v

    for i in opts:
        for k in always:
            del i[k]

    capabilities["capabilities"]["alwaysMatch"] = always
    capabilities["capabilities"]["firstMatch"] = opts

    return capabilities


class BaseWebDriver(metaclass=ABCMeta):
    """Abstract Base Class for all Webdriver subtypes.

    ABC's allow custom implementations of Webdriver to be registered so
    that isinstance type checks will succeed.
    """


class WebDriver(BaseWebDriver):
    """Controls a browser by sending commands to a remote server. This server
    is expected to be running the WebDriver wire protocol as defined at
    https://www.selenium.dev/documentation/legacy/json_wire_protocol/.

    Attributes:
    -----------
    session_id - String ID of the browser session started and controlled by this WebDriver.
    capabilities - Dictionary of effective capabilities of this browser session as returned
        by the remote server. See https://www.selenium.dev/documentation/legacy/desired_capabilities/
    command_executor : str or remote_connection.RemoteConnection object used to execute commands.
    error_handler - errorhandler.ErrorHandler object used to handle errors.
    """

    _web_element_cls = WebElement
    _shadowroot_cls = ShadowRoot

    def __init__(
        self,
        command_executor: Union[str, RemoteConnection] = "http://127.0.0.1:4444",
        keep_alive: bool = True,
        file_detector: Optional[FileDetector] = None,
        options: Optional[Union[BaseOptions, List[BaseOptions]]] = None,
        locator_converter: Optional[LocatorConverter] = None,
        web_element_cls: Optional[type] = None,
        client_config: Optional[ClientConfig] = None,
    ) -> None:
        """Create a new driver that will issue commands using the wire
        protocol.

        Parameters:
        -----------
        command_executor : str or remote_connection.RemoteConnection
            - Either a string representing the URL of the remote server or a custom
            remote_connection.RemoteConnection object. Defaults to 'http://127.0.0.1:4444/wd/hub'.
        keep_alive : bool (Deprecated)
            - Whether to configure remote_connection.RemoteConnection to use HTTP keep-alive. Defaults to True.
        file_detector : object or None
            - Pass a custom file detector object during instantiation. If None, the default
                LocalFileDetector() will be used.
        options : options.Options
            - Instance of a driver options.Options class.
        locator_converter : object or None
            - Custom locator converter to use. Defaults to None.
        web_element_cls : class
            - Custom class to use for web elements. Defaults to WebElement.
        client_config : object or None
            - Custom client configuration to use. Defaults to None.
        """

        if options is None:
            raise TypeError(
                "missing 1 required keyword-only argument: 'options' (instance of driver `options.Options` class)"
            )
        elif isinstance(options, list):
            capabilities = create_matches(options)
            _ignore_local_proxy = False
        else:
            capabilities = options.to_capabilities()
            _ignore_local_proxy = options._ignore_local_proxy
        self.command_executor = command_executor
        if isinstance(self.command_executor, (str, bytes)):
            self.command_executor = get_remote_connection(
                capabilities,
                command_executor=command_executor,
                keep_alive=keep_alive,
                ignore_local_proxy=_ignore_local_proxy,
                client_config=client_config,
            )
        self._is_remote = True
        self.session_id = None
        self.caps = {}
        self.pinned_scripts = {}
        self.error_handler = ErrorHandler()
        self._switch_to = SwitchTo(self)
        self._mobile = Mobile(self)
        self.file_detector = file_detector or LocalFileDetector()
        self.locator_converter = locator_converter or LocatorConverter()
        self._web_element_cls = web_element_cls or self._web_element_cls
        self._authenticator_id = None
        self.start_client()
        self.start_session(capabilities)
        self._fedcm = FedCM(self)

        self._websocket_connection = None
        self._script = None
        self._network = None
        self._browser = None
        self._bidi_session = None
        self._browsing_context = None
        self._storage = None
        self._webextension = None

    def __repr__(self):
        return f'<{type(self).__module__}.{type(self).__name__} (session="{self.session_id}")>'

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ):
        self.quit()

    @contextmanager
    def file_detector_context(self, file_detector_class, *args, **kwargs):
        """Overrides the current file detector (if necessary) in limited
        context. Ensures the original file detector is set afterwards.

        Parameters:
        -----------
        file_detector_class : object
            - Class of the desired file detector. If the class is different
            from the current file_detector, then the class is instantiated with args and kwargs
            and used as a file detector during the duration of the context manager.
        args : tuple
            - Optional arguments that get passed to the file detector class during instantiation.
        kwargs : dict
            - Keyword arguments, passed the same way as args.

        Example:
        --------
        >>> with webdriver.file_detector_context(UselessFileDetector):
        >>>    someinput.send_keys('/etc/hosts')
        """
        last_detector = None
        if not isinstance(self.file_detector, file_detector_class):
            last_detector = self.file_detector
            self.file_detector = file_detector_class(*args, **kwargs)
        try:
            yield
        finally:
            if last_detector:
                self.file_detector = last_detector

    @property
    def mobile(self) -> Mobile:
        return self._mobile

    @property
    def name(self) -> str:
        """Returns the name of the underlying browser for this instance.

        Example:
        --------
        >>> name = driver.name
        """
        if "browserName" in self.caps:
            return self.caps["browserName"]
        raise KeyError("browserName not specified in session capabilities")

    def start_client(self):
        """Called before starting a new session.

        This method may be overridden to define custom startup behavior.
        """
        pass

    def stop_client(self):
        """Called after executing a quit command.

        This method may be overridden to define custom shutdown
        behavior.
        """
        pass

    def start_session(self, capabilities: dict) -> None:
        """Creates a new session with the desired capabilities.

        Parameters:
        -----------
        capabilities : dict
            - A capabilities dict to start the session with.
        """

        caps = _create_caps(capabilities)
        try:
            response = self.execute(Command.NEW_SESSION, caps)["value"]
            self.session_id = response.get("sessionId")
            self.caps = response.get("capabilities")
        except Exception:
            if hasattr(self, "service") and self.service is not None:
                self.service.stop()
            raise

    def _wrap_value(self, value):
        if isinstance(value, dict):
            converted = {}
            for key, val in value.items():
                converted[key] = self._wrap_value(val)
            return converted
        if isinstance(value, self._web_element_cls):
            return {"element-6066-11e4-a52e-4f735466cecf": value.id}
        if isinstance(value, self._shadowroot_cls):
            return {"shadow-6066-11e4-a52e-4f735466cecf": value.id}
        if isinstance(value, list):
            return list(self._wrap_value(item) for item in value)
        return value

    def create_web_element(self, element_id: str) -> WebElement:
        """Creates a web element with the specified `element_id`."""
        return self._web_element_cls(self, element_id)

    def _unwrap_value(self, value):
        if isinstance(value, dict):
            if "element-6066-11e4-a52e-4f735466cecf" in value:
                return self.create_web_element(value["element-6066-11e4-a52e-4f735466cecf"])
            if "shadow-6066-11e4-a52e-4f735466cecf" in value:
                return self._shadowroot_cls(self, value["shadow-6066-11e4-a52e-4f735466cecf"])
            for key, val in value.items():
                value[key] = self._unwrap_value(val)
            return value
        if isinstance(value, list):
            return list(self._unwrap_value(item) for item in value)
        return value

    def execute_cdp_cmd(self, cmd: str, cmd_args: dict):
        """Execute Chrome Devtools Protocol command and get returned result The
        command and command args should follow chrome devtools protocol
        domains/commands, refer to link
        https://chromedevtools.github.io/devtools-protocol/

        Parameters:
        -----------
        cmd : str,
            - Command name

        cmd_args : dict
            - Command args
            - Empty dict {} if there is no command args

        Returns:
        --------
            A dict, empty dict {} if there is no result to return.
                - To getResponseBody: {'base64Encoded': False, 'body': 'response body string'}

        Example:
        --------
        >>> driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": requestId})

        """
        return self.execute("executeCdpCommand", {"cmd": cmd, "params": cmd_args})["value"]

    def execute(self, driver_command: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Sends a command to be executed by a command.CommandExecutor.

        Parameters:
        -----------
        driver_command : str
            - The name of the command to execute as a string.

        params : dict
            - A dictionary of named Parameters to send with the command.

        Returns:
        --------
          dict - The command's JSON response loaded into a dictionary object.
        """
        params = self._wrap_value(params)

        if self.session_id:
            if not params:
                params = {"sessionId": self.session_id}
            elif "sessionId" not in params:
                params["sessionId"] = self.session_id

        response = self.command_executor.execute(driver_command, params)
        if response:
            self.error_handler.check_response(response)
            response["value"] = self._unwrap_value(response.get("value", None))
            return response
        # If the server doesn't send a response, assume the command was
        # a success
        return {"success": 0, "value": None, "sessionId": self.session_id}

    def get(self, url: str) -> None:
        """Navigate the browser to the specified URL in the current window or
        tab.

        The method does not return until the page is fully loaded (i.e. the
        onload event has fired).

        Parameters:
        -----------
        url : str
            - The URL to be opened by the browser.
            - Must include the protocol (e.g., http://, https://).

        Example:
        --------
        >>> driver = webdriver.Chrome()
        >>> driver.get("https://example.com")
        """
        self.execute(Command.GET, {"url": url})

    @property
    def title(self) -> str:
        """Returns the title of the current page.

        Example:
        --------
        >>> element = driver.find_element(By.ID, "foo")
        >>> print(element.title())
        """
        return self.execute(Command.GET_TITLE).get("value", "")

    def pin_script(self, script: str, script_key=None) -> ScriptKey:
        """Store common javascript scripts to be executed later by a unique
        hashable ID.

        Example:
        --------
        >>> script = "return document.getElementById('foo').value"
        """
        script_key_instance = ScriptKey(script_key)
        self.pinned_scripts[script_key_instance.id] = script
        return script_key_instance

    def unpin(self, script_key: ScriptKey) -> None:
        """Remove a pinned script from storage.

        Example:
        --------
        >>> driver.unpin(script_key)
        """
        try:
            self.pinned_scripts.pop(script_key.id)
        except KeyError:
            raise KeyError(f"No script with key: {script_key} existed in {self.pinned_scripts}") from None

    def get_pinned_scripts(self) -> List[str]:
        """Return a list of all pinned scripts.

        Example:
        --------
        >>> pinned_scripts = driver.get_pinned_scripts()
        """
        return list(self.pinned_scripts)

    def execute_script(self, script: str, *args):
        """Synchronously Executes JavaScript in the current window/frame.

        Parameters:
        -----------
        script : str
            - The javascript to execute.

        *args : tuple
            - Any applicable arguments for your JavaScript.

        Example:
        --------
        >>> input_id = "username"
        >>> input_value = "test_user"
        >>> driver.execute_script("document.getElementById(arguments[0]).value = arguments[1];", input_id, input_value)
        """
        if isinstance(script, ScriptKey):
            try:
                script = self.pinned_scripts[script.id]
            except KeyError:
                raise JavascriptException("Pinned script could not be found")

        converted_args = list(args)
        command = Command.W3C_EXECUTE_SCRIPT

        return self.execute(command, {"script": script, "args": converted_args})["value"]

    def execute_async_script(self, script: str, *args):
        """Asynchronously Executes JavaScript in the current window/frame.

        Parameters:
        -----------
        script : str
            - The javascript to execute.

        *args : tuple
            - Any applicable arguments for your JavaScript.

        Example:
        --------
        >>> script = "var callback = arguments[arguments.length - 1]; "
        ...     "window.setTimeout(function(){ callback('timeout') }, 3000);"
        >>> driver.execute_async_script(script)
        """
        converted_args = list(args)
        command = Command.W3C_EXECUTE_SCRIPT_ASYNC

        return self.execute(command, {"script": script, "args": converted_args})["value"]

    @property
    def current_url(self) -> str:
        """Gets the URL of the current page.

        Example:
        --------
        >>> print(driver.current_url)
        """
        return self.execute(Command.GET_CURRENT_URL)["value"]

    @property
    def page_source(self) -> str:
        """Gets the source of the current page.

        Example:
        --------
        >>> print(driver.page_source)
        """
        return self.execute(Command.GET_PAGE_SOURCE)["value"]

    def close(self) -> None:
        """Closes the current window.

        Example:
        --------
        >>> driver.close()
        """
        self.execute(Command.CLOSE)

    def quit(self) -> None:
        """Quits the driver and closes every associated window.

        Example:
        --------
        >>> driver.quit()
        """
        try:
            self.execute(Command.QUIT)
        finally:
            self.stop_client()
            self.command_executor.close()

    @property
    def current_window_handle(self) -> str:
        """Returns the handle of the current window.

        Example:
        --------
        >>> print(driver.current_window_handle)
        """
        return self.execute(Command.W3C_GET_CURRENT_WINDOW_HANDLE)["value"]

    @property
    def window_handles(self) -> List[str]:
        """Returns the handles of all windows within the current session.

        Example:
        --------
        >>> print(driver.window_handles)
        """
        return self.execute(Command.W3C_GET_WINDOW_HANDLES)["value"]

    def maximize_window(self) -> None:
        """Maximizes the current window that webdriver is using.

        Example:
        --------
        >>> driver.maximize_window()
        """
        command = Command.W3C_MAXIMIZE_WINDOW
        self.execute(command, None)

    def fullscreen_window(self) -> None:
        """Invokes the window manager-specific 'full screen' operation.

        Example:
        --------
        >>> driver.fullscreen_window()
        """
        self.execute(Command.FULLSCREEN_WINDOW)

    def minimize_window(self) -> None:
        """Invokes the window manager-specific 'minimize' operation."""
        self.execute(Command.MINIMIZE_WINDOW)

    def print_page(self, print_options: Optional[PrintOptions] = None) -> str:
        """Takes PDF of the current page.

        The driver makes a best effort to return a PDF based on the
        provided Parameters.

        Example:
        --------
        >>> driver.print_page()
        """
        options = {}
        if print_options:
            options = print_options.to_dict()

        return self.execute(Command.PRINT_PAGE, options)["value"]

    @property
    def switch_to(self) -> SwitchTo:
        """Return an object containing all options to switch focus into.

        Returns:
        --------
        SwitchTo: an object containing all options to switch focus into.

        Examples:
        --------
        >>> element = driver.switch_to.active_element
        >>> alert = driver.switch_to.alert
        >>> driver.switch_to.default_content()
        >>> driver.switch_to.frame("frame_name")
        >>> driver.switch_to.frame(1)
        >>> driver.switch_to.frame(driver.find_elements(By.TAG_NAME, "iframe")[0])
        >>> driver.switch_to.parent_frame()
        >>> driver.switch_to.window("main")
        """
        return self._switch_to

    # Navigation
    def back(self) -> None:
        """Goes one step backward in the browser history.

        Example:
        --------
        >>> driver.back()
        """
        self.execute(Command.GO_BACK)

    def forward(self) -> None:
        """Goes one step forward in the browser history.

        Example:
        --------
        >>> driver.forward()
        """
        self.execute(Command.GO_FORWARD)

    def refresh(self) -> None:
        """Refreshes the current page.

        Example:
        --------
        >>> driver.refresh()
        """
        self.execute(Command.REFRESH)

    # Options
    def get_cookies(self) -> List[dict]:
        """Returns a set of dictionaries, corresponding to cookies visible in
        the current session.

        Returns:
        --------
        cookies:List[dict] : A list of dictionaries, corresponding to cookies visible in the current

        Example:
        --------
        >>> cookies = driver.get_cookies()
        """
        return self.execute(Command.GET_ALL_COOKIES)["value"]

    def get_cookie(self, name) -> Optional[Dict]:
        """Get a single cookie by name. Raises ValueError if the name is empty
        or whitespace. Returns the cookie if found, None if not.

        Example:
        --------
        >>> cookie = driver.get_cookie("my_cookie")
        """
        if not name or name.isspace():
            raise ValueError("Cookie name cannot be empty")

        with contextlib.suppress(NoSuchCookieException):
            return self.execute(Command.GET_COOKIE, {"name": name})["value"]

        return None

    def delete_cookie(self, name) -> None:
        """Deletes a single cookie with the given name. Raises ValueError if
        the name is empty or whitespace.

        Example:
        --------
        >>> driver.delete_cookie("my_cookie")
        """

        # firefox deletes all cookies when "" is passed as name
        if not name or name.isspace():
            raise ValueError("Cookie name cannot be empty")

        self.execute(Command.DELETE_COOKIE, {"name": name})

    def delete_all_cookies(self) -> None:
        """Delete all cookies in the scope of the session.

        Example:
        --------
        >>> driver.delete_all_cookies()
        """
        self.execute(Command.DELETE_ALL_COOKIES)

    def add_cookie(self, cookie_dict) -> None:
        """Adds a cookie to your current session.

        Parameters:
        -----------
        cookie_dict : dict
            - A dictionary object, with required keys - "name" and "value";
            - Optional keys - "path", "domain", "secure", "httpOnly", "expiry", "sameSite"

        Examples:
        --------
        >>> driver.add_cookie({"name": "foo", "value": "bar"})
        >>> driver.add_cookie({"name": "foo", "value": "bar", "path": "/"})
        >>> driver.add_cookie({"name": "foo", "value": "bar", "path": "/", "secure": True})
        >>> driver.add_cookie({"name": "foo", "value": "bar", "sameSite": "Strict"})
        """
        if "sameSite" in cookie_dict:
            assert cookie_dict["sameSite"] in ["Strict", "Lax", "None"]
            self.execute(Command.ADD_COOKIE, {"cookie": cookie_dict})
        else:
            self.execute(Command.ADD_COOKIE, {"cookie": cookie_dict})

    # Timeouts
    def implicitly_wait(self, time_to_wait: float) -> None:
        """Sets a sticky timeout to implicitly wait for an element to be found,
        or a command to complete. This method only needs to be called one time
        per session. To set the timeout for calls to execute_async_script, see
        set_script_timeout.

        Parameters:
        -----------
        time_to_wait : float
            - Amount of time to wait (in seconds)

        Example:
        --------
        >>> driver.implicitly_wait(30)
        """
        self.execute(Command.SET_TIMEOUTS, {"implicit": int(float(time_to_wait) * 1000)})

    def set_script_timeout(self, time_to_wait: float) -> None:
        """Set the amount of time that the script should wait during an
        execute_async_script call before throwing an error.

        Parameters:
        -----------
        time_to_wait : float
            - The amount of time to wait (in seconds)

        Example:
        --------
        >>> driver.set_script_timeout(30)
        """
        self.execute(Command.SET_TIMEOUTS, {"script": int(float(time_to_wait) * 1000)})

    def set_page_load_timeout(self, time_to_wait: float) -> None:
        """Set the amount of time to wait for a page load to complete before
        throwing an error.

        Parameters:
        -----------
        time_to_wait : float
             - The amount of time to wait (in seconds)

        Example:
        --------
        >>> driver.set_page_load_timeout(30)
        """
        try:
            self.execute(Command.SET_TIMEOUTS, {"pageLoad": int(float(time_to_wait) * 1000)})
        except WebDriverException:
            self.execute(Command.SET_TIMEOUTS, {"ms": float(time_to_wait) * 1000, "type": "page load"})

    @property
    def timeouts(self) -> Timeouts:
        """Get all the timeouts that have been set on the current session.

        Returns:
        --------
        Timeouts: A named tuple with the following fields:
            - implicit_wait: The time to wait for elements to be found.
            - page_load: The time to wait for a page to load.
            - script: The time to wait for scripts to execute.

        Example:
        --------
        >>> driver.timeouts
        """
        timeouts = self.execute(Command.GET_TIMEOUTS)["value"]
        timeouts["implicit_wait"] = timeouts.pop("implicit") / 1000
        timeouts["page_load"] = timeouts.pop("pageLoad") / 1000
        timeouts["script"] = timeouts.pop("script") / 1000
        return Timeouts(**timeouts)

    @timeouts.setter
    def timeouts(self, timeouts) -> None:
        """Set all timeouts for the session. This will override any previously
        set timeouts.

        Example:
        --------
        >>> my_timeouts = Timeouts()
        >>> my_timeouts.implicit_wait = 10
        >>> driver.timeouts = my_timeouts
        """
        _ = self.execute(Command.SET_TIMEOUTS, timeouts._to_json())["value"]

    def find_element(self, by=By.ID, value: Optional[str] = None) -> WebElement:
        """Find an element given a By strategy and locator.

        Parameters:
        -----------
        by : selenium.webdriver.common.by.By
            The locating strategy to use. Default is `By.ID`. Supported values include:
            - By.ID: Locate by element ID.
            - By.NAME: Locate by the `name` attribute.
            - By.XPATH: Locate by an XPath expression.
            - By.CSS_SELECTOR: Locate by a CSS selector.
            - By.CLASS_NAME: Locate by the `class` attribute.
            - By.TAG_NAME: Locate by the tag name (e.g., "input", "button").
            - By.LINK_TEXT: Locate a link element by its exact text.
            - By.PARTIAL_LINK_TEXT: Locate a link element by partial text match.
            - RelativeBy: Locate elements relative to a specified root element.

        Example:
        --------
        element = driver.find_element(By.ID, 'foo')

        Returns:
        -------
        WebElement
            The first matching `WebElement` found on the page.
        """
        by, value = self.locator_converter.convert(by, value)

        if isinstance(by, RelativeBy):
            elements = self.find_elements(by=by, value=value)
            if not elements:
                raise NoSuchElementException(f"Cannot locate relative element with: {by.root}")
            return elements[0]

        return self.execute(Command.FIND_ELEMENT, {"using": by, "value": value})["value"]

    def find_elements(self, by=By.ID, value: Optional[str] = None) -> List[WebElement]:
        """Find elements given a By strategy and locator.

        Parameters:
        -----------
        by : selenium.webdriver.common.by.By
            The locating strategy to use. Default is `By.ID`. Supported values include:
            - By.ID: Locate by element ID.
            - By.NAME: Locate by the `name` attribute.
            - By.XPATH: Locate by an XPath expression.
            - By.CSS_SELECTOR: Locate by a CSS selector.
            - By.CLASS_NAME: Locate by the `class` attribute.
            - By.TAG_NAME: Locate by the tag name (e.g., "input", "button").
            - By.LINK_TEXT: Locate a link element by its exact text.
            - By.PARTIAL_LINK_TEXT: Locate a link element by partial text match.
            - RelativeBy: Locate elements relative to a specified root element.

        Example:
        --------
        element = driver.find_elements(By.ID, 'foo')

        Returns:
        -------
        List[WebElement]
            list of `WebElements` matching locator strategy found on the page.
        """
        by, value = self.locator_converter.convert(by, value)

        if isinstance(by, RelativeBy):
            _pkg = ".".join(__name__.split(".")[:-1])
            raw_function = pkgutil.get_data(_pkg, "findElements.js").decode("utf8")
            find_element_js = f"/* findElements */return ({raw_function}).apply(null, arguments);"
            return self.execute_script(find_element_js, by.to_dict())

        # Return empty list if driver returns null
        # See https://github.com/SeleniumHQ/selenium/issues/4555
        return self.execute(Command.FIND_ELEMENTS, {"using": by, "value": value})["value"] or []

    @property
    def capabilities(self) -> dict:
        """Returns the drivers current capabilities being used.

        Example:
        --------
        >>> print(driver.capabilities)
        """
        return self.caps

    def get_screenshot_as_file(self, filename) -> bool:
        """Saves a screenshot of the current window to a PNG image file.
        Returns False if there is any IOError, else returns True. Use full
        paths in your filename.

        Parameters:
        -----------
        filename : str
            - The full path you wish to save your screenshot to. This
            - should end with a `.png` extension.

        Example:
        --------
        >>> driver.get_screenshot_as_file("/Screenshots/foo.png")
        """
        if not str(filename).lower().endswith(".png"):
            warnings.warn(
                "name used for saved screenshot does not match file type. It should end with a `.png` extension",
                UserWarning,
                stacklevel=2,
            )
        png = self.get_screenshot_as_png()
        try:
            with open(filename, "wb") as f:
                f.write(png)
        except OSError:
            return False
        finally:
            del png
        return True

    def save_screenshot(self, filename) -> bool:
        """Saves a screenshot of the current window to a PNG image file.
        Returns False if there is any IOError, else returns True. Use full
        paths in your filename.

        Parameters:
        -----------
        filename : str
            - The full path you wish to save your screenshot to. This
            - should end with a `.png` extension.

        Example:
        --------
        >>> driver.save_screenshot("/Screenshots/foo.png")
        """
        return self.get_screenshot_as_file(filename)

    def get_screenshot_as_png(self) -> bytes:
        """Gets the screenshot of the current window as a binary data.

        Example:
        --------
        >>> driver.get_screenshot_as_png()
        """
        return b64decode(self.get_screenshot_as_base64().encode("ascii"))

    def get_screenshot_as_base64(self) -> str:
        """Gets the screenshot of the current window as a base64 encoded string
        which is useful in embedded images in HTML.

        Example:
        --------
        >>> driver.get_screenshot_as_base64()
        """
        return self.execute(Command.SCREENSHOT)["value"]

    def set_window_size(self, width, height, windowHandle: str = "current") -> None:
        """Sets the width and height of the current window. (window.resizeTo)

        Parameters:
        -----------
        width : int
            - the width in pixels to set the window to

        height : int
            - the height in pixels to set the window to

        Example:
        --------
        >>> driver.set_window_size(800, 600)
        """
        self._check_if_window_handle_is_current(windowHandle)
        self.set_window_rect(width=int(width), height=int(height))

    def get_window_size(self, windowHandle: str = "current") -> dict:
        """Gets the width and height of the current window.

        Example:
        --------
        >>> driver.get_window_size()
        """

        self._check_if_window_handle_is_current(windowHandle)
        size = self.get_window_rect()

        if size.get("value", None):
            size = size["value"]

        return {k: size[k] for k in ("width", "height")}

    def set_window_position(self, x: float, y: float, windowHandle: str = "current") -> dict:
        """Sets the x,y position of the current window. (window.moveTo)

        Parameters:
        ---------
        x : float
            - The x-coordinate in pixels to set the window position

        y : float
            - The y-coordinate in pixels to set the window position

        Example:
        --------
        >>> driver.set_window_position(0, 0)
        """
        self._check_if_window_handle_is_current(windowHandle)
        return self.set_window_rect(x=int(x), y=int(y))

    def get_window_position(self, windowHandle="current") -> dict:
        """Gets the x,y position of the current window.

        Example:
        --------
        >>> driver.get_window_position()
        """

        self._check_if_window_handle_is_current(windowHandle)
        position = self.get_window_rect()

        return {k: position[k] for k in ("x", "y")}

    def _check_if_window_handle_is_current(self, windowHandle: str) -> None:
        """Warns if the window handle is not equal to `current`."""
        if windowHandle != "current":
            warnings.warn("Only 'current' window is supported for W3C compatible browsers.", stacklevel=2)

    def get_window_rect(self) -> dict:
        """Gets the x, y coordinates of the window as well as height and width
        of the current window.

        Example:
        --------
        >>> driver.get_window_rect()
        """
        return self.execute(Command.GET_WINDOW_RECT)["value"]

    def set_window_rect(self, x=None, y=None, width=None, height=None) -> dict:
        """Sets the x, y coordinates of the window as well as height and width
        of the current window. This method is only supported for W3C compatible
        browsers; other browsers should use `set_window_position` and
        `set_window_size`.

        Example:
        --------
        >>> driver.set_window_rect(x=10, y=10)
        >>> driver.set_window_rect(width=100, height=200)
        >>> driver.set_window_rect(x=10, y=10, width=100, height=200)
        """

        if (x is None and y is None) and (not height and not width):
            raise InvalidArgumentException("x and y or height and width need values")

        return self.execute(Command.SET_WINDOW_RECT, {"x": x, "y": y, "width": width, "height": height})["value"]

    @property
    def file_detector(self) -> FileDetector:
        return self._file_detector

    @file_detector.setter
    def file_detector(self, detector) -> None:
        """Set the file detector to be used when sending keyboard input. By
        default, this is set to a file detector that does nothing.

        - see FileDetector
        - see LocalFileDetector
        - see UselessFileDetector

        Parameters:
        -----------
        detector : Any
            - The detector to use. Must not be None.
        """
        if not detector:
            raise WebDriverException("You may not set a file detector that is null")
        if not isinstance(detector, FileDetector):
            raise WebDriverException("Detector has to be instance of FileDetector")
        self._file_detector = detector

    @property
    def orientation(self):
        """Gets the current orientation of the device.

        Example:
        --------
        >>> orientation = driver.orientation
        """
        return self.execute(Command.GET_SCREEN_ORIENTATION)["value"]

    @orientation.setter
    def orientation(self, value) -> None:
        """Sets the current orientation of the device.

        Parameters:
        -----------
        value : str
            - orientation to set it to.

        Example:
        --------
        >>> driver.orientation = "landscape"
        """
        allowed_values = ["LANDSCAPE", "PORTRAIT"]
        if value.upper() in allowed_values:
            self.execute(Command.SET_SCREEN_ORIENTATION, {"orientation": value})
        else:
            raise WebDriverException("You can only set the orientation to 'LANDSCAPE' and 'PORTRAIT'")

    def start_devtools(self):
        global devtools
        if self._websocket_connection:
            return devtools, self._websocket_connection
        else:
            global cdp
            import_cdp()

            if not devtools:
                if self.caps.get("se:cdp"):
                    ws_url = self.caps.get("se:cdp")
                    version = self.caps.get("se:cdpVersion").split(".")[0]
                else:
                    version, ws_url = self._get_cdp_details()

                if not ws_url:
                    raise WebDriverException("Unable to find url to connect to from capabilities")

                devtools = cdp.import_devtools(version)
                if self.caps["browserName"].lower() == "firefox":
                    raise RuntimeError("CDP support for Firefox has been removed. Please switch to WebDriver BiDi.")
            self._websocket_connection = WebSocketConnection(ws_url)
            targets = self._websocket_connection.execute(devtools.target.get_targets())
            target_id = targets[0].target_id
            session = self._websocket_connection.execute(devtools.target.attach_to_target(target_id, True))
            self._websocket_connection.session_id = session
            return devtools, self._websocket_connection

    @asynccontextmanager
    async def bidi_connection(self):
        global cdp
        import_cdp()
        if self.caps.get("se:cdp"):
            ws_url = self.caps.get("se:cdp")
            version = self.caps.get("se:cdpVersion").split(".")[0]
        else:
            version, ws_url = self._get_cdp_details()

        if not ws_url:
            raise WebDriverException("Unable to find url to connect to from capabilities")

        devtools = cdp.import_devtools(version)
        async with cdp.open_cdp(ws_url) as conn:
            targets = await conn.execute(devtools.target.get_targets())
            target_id = targets[0].target_id
            async with conn.open_session(target_id) as session:
                yield BidiConnection(session, cdp, devtools)

    @property
    def script(self):
        if not self._websocket_connection:
            self._start_bidi()

        if not self._script:
            self._script = Script(self._websocket_connection)

        return self._script

    def _start_bidi(self):
        if self.caps.get("webSocketUrl"):
            ws_url = self.caps.get("webSocketUrl")
        else:
            raise WebDriverException("Unable to find url to connect to from capabilities")

        self._websocket_connection = WebSocketConnection(ws_url)

    @property
    def network(self):
        if not self._websocket_connection:
            self._start_bidi()

        if not hasattr(self, "_network") or self._network is None:
            self._network = Network(self._websocket_connection)

        return self._network

    @property
    def browser(self):
        """Returns a browser module object for BiDi browser commands.

        Returns:
        --------
        Browser: an object containing access to BiDi browser commands.

        Examples:
        ---------
        >>> user_context = driver.browser.create_user_context()
        >>> user_contexts = driver.browser.get_user_contexts()
        >>> client_windows = driver.browser.get_client_windows()
        >>> driver.browser.remove_user_context(user_context)
        """
        if not self._websocket_connection:
            self._start_bidi()

        if self._browser is None:
            self._browser = Browser(self._websocket_connection)

        return self._browser

    @property
    def _session(self):
        """
        Returns the BiDi session object for the current WebDriver session.
        """
        if not self._websocket_connection:
            self._start_bidi()

        if self._bidi_session is None:
            self._bidi_session = Session(self._websocket_connection)

        return self._bidi_session

    @property
    def browsing_context(self):
        """Returns a browsing context module object for BiDi browsing context commands.

        Returns:
        --------
        BrowsingContext: an object containing access to BiDi browsing context commands.

        Examples:
        ---------
        >>> context_id = driver.browsing_context.create(type="tab")
        >>> driver.browsing_context.navigate(context=context_id, url="https://www.selenium.dev")
        >>> driver.browsing_context.capture_screenshot(context=context_id)
        >>> driver.browsing_context.close(context_id)
        """
        if not self._websocket_connection:
            self._start_bidi()

        if self._browsing_context is None:
            self._browsing_context = BrowsingContext(self._websocket_connection)

        return self._browsing_context

    @property
    def storage(self):
        """Returns a storage module object for BiDi storage commands.

        Returns:
        --------
        Storage: an object containing access to BiDi storage commands.

        Examples:
        ---------
        >>> cookie_filter = CookieFilter(name="example")
        >>> result = driver.storage.get_cookies(filter=cookie_filter)
        >>> driver.storage.set_cookie(cookie=PartialCookie(
                "name", BytesValue(BytesValue.TYPE_STRING, "value"), "domain")
            )
        >>> driver.storage.delete_cookies(filter=CookieFilter(name="example"))
        """
        if not self._websocket_connection:
            self._start_bidi()

        if self._storage is None:
            self._storage = Storage(self._websocket_connection)

        return self._storage

    @property
    def webextension(self):
        """Returns a webextension module object for BiDi webextension commands.

        Returns:
        --------
        WebExtension: an object containing access to BiDi webextension commands.

        Examples:
        ---------
        >>> extension_path = "/path/to/extension"
        >>> extension_result = driver.webextension.install(path=extension_path)
        >>> driver.webextension.uninstall(extension_result)
        """
        if not self._websocket_connection:
            self._start_bidi()

        if self._webextension is None:
            self._webextension = WebExtension(self._websocket_connection)

        return self._webextension

    def _get_cdp_details(self):
        import json

        import urllib3

        http = urllib3.PoolManager()
        if self.caps.get("browserName") == "chrome":
            debugger_address = self.caps.get("goog:chromeOptions").get("debuggerAddress")
        elif self.caps.get("browserName") == "MicrosoftEdge":
            debugger_address = self.caps.get("ms:edgeOptions").get("debuggerAddress")

        res = http.request("GET", f"http://{debugger_address}/json/version")
        data = json.loads(res.data)

        browser_version = data.get("Browser")
        websocket_url = data.get("webSocketDebuggerUrl")

        import re

        version = re.search(r".*/(\d+)\.", browser_version).group(1)

        return version, websocket_url

    # Virtual Authenticator Methods
    def add_virtual_authenticator(self, options: VirtualAuthenticatorOptions) -> None:
        """Adds a virtual authenticator with the given options.

        Example:
        --------
        >>> from selenium.webdriver.common.virtual_authenticator import VirtualAuthenticatorOptions
        >>> options = VirtualAuthenticatorOptions(protocol="u2f", transport="usb", device_id="myDevice123")
        >>> driver.add_virtual_authenticator(options)
        """
        self._authenticator_id = self.execute(Command.ADD_VIRTUAL_AUTHENTICATOR, options.to_dict())["value"]

    @property
    def virtual_authenticator_id(self) -> str:
        """Returns the id of the virtual authenticator.

        Example:
        --------
        >>> print(driver.virtual_authenticator_id)
        """
        return self._authenticator_id

    @required_virtual_authenticator
    def remove_virtual_authenticator(self) -> None:
        """Removes a previously added virtual authenticator.

        The authenticator is no longer valid after removal, so no
        methods may be called.

        Example:
        --------
        >>> driver.remove_virtual_authenticator()
        """
        self.execute(Command.REMOVE_VIRTUAL_AUTHENTICATOR, {"authenticatorId": self._authenticator_id})
        self._authenticator_id = None

    @required_virtual_authenticator
    def add_credential(self, credential: Credential) -> None:
        """Injects a credential into the authenticator.

        Example:
        --------
        >>> from selenium.webdriver.common.credential import Credential
        >>> credential = Credential(id="user@example.com", password="aPassword")
        >>> driver.add_credential(credential)
        """
        self.execute(Command.ADD_CREDENTIAL, {**credential.to_dict(), "authenticatorId": self._authenticator_id})

    @required_virtual_authenticator
    def get_credentials(self) -> List[Credential]:
        """Returns the list of credentials owned by the authenticator.

        Example:
        --------
        >>> credentials = driver.get_credentials()
        """
        credential_data = self.execute(Command.GET_CREDENTIALS, {"authenticatorId": self._authenticator_id})
        return [Credential.from_dict(credential) for credential in credential_data["value"]]

    @required_virtual_authenticator
    def remove_credential(self, credential_id: Union[str, bytearray]) -> None:
        """Removes a credential from the authenticator.

        Example:
        --------
        >>> credential_id = "user@example.com"
        >>> driver.remove_credential(credential_id)
        """
        # Check if the credential is bytearray converted to b64 string
        if isinstance(credential_id, bytearray):
            credential_id = urlsafe_b64encode(credential_id).decode()

        self.execute(
            Command.REMOVE_CREDENTIAL, {"credentialId": credential_id, "authenticatorId": self._authenticator_id}
        )

    @required_virtual_authenticator
    def remove_all_credentials(self) -> None:
        """Removes all credentials from the authenticator.

        Example:
        --------
        >>> driver.remove_all_credentials()
        """
        self.execute(Command.REMOVE_ALL_CREDENTIALS, {"authenticatorId": self._authenticator_id})

    @required_virtual_authenticator
    def set_user_verified(self, verified: bool) -> None:
        """Sets whether the authenticator will simulate success or fail on user
        verification.

        Parameters:
        -----------
        verified: True if the authenticator will pass user verification, False otherwise.

        Example:
        --------
        >>> driver.set_user_verified(True)
        """
        self.execute(Command.SET_USER_VERIFIED, {"authenticatorId": self._authenticator_id, "isUserVerified": verified})

    def get_downloadable_files(self) -> list:
        """Retrieves the downloadable files as a list of file names.

        Example:
        --------
        >>> files = driver.get_downloadable_files()
        """
        if "se:downloadsEnabled" not in self.capabilities:
            raise WebDriverException("You must enable downloads in order to work with downloadable files.")

        return self.execute(Command.GET_DOWNLOADABLE_FILES)["value"]["names"]

    def download_file(self, file_name: str, target_directory: str) -> None:
        """Downloads a file with the specified file name to the target
        directory.

        Parameters:
        -----------
        file_name : str
            - The name of the file to download.

        target_directory : str
            - The path to the directory to save the downloaded file.

        Example:
        --------
        >>> driver.download_file("example.zip", "/path/to/directory")
        """
        if "se:downloadsEnabled" not in self.capabilities:
            raise WebDriverException("You must enable downloads in order to work with downloadable files.")

        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        contents = self.execute(Command.DOWNLOAD_FILE, {"name": file_name})["value"]["contents"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_file = os.path.join(tmp_dir, file_name + ".zip")
            with open(zip_file, "wb") as file:
                file.write(base64.b64decode(contents))

            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(target_directory)

    def delete_downloadable_files(self) -> None:
        """Deletes all downloadable files.

        Example:
        --------
        >>> driver.delete_downloadable_files()
        """
        if "se:downloadsEnabled" not in self.capabilities:
            raise WebDriverException("You must enable downloads in order to work with downloadable files.")

        self.execute(Command.DELETE_DOWNLOADABLE_FILES)

    @property
    def fedcm(self) -> FedCM:
        """Returns the Federated Credential Management (FedCM) dialog object
        for interaction.

        Returns:
        -------
        FedCM: an object providing access to all Federated Credential Management (FedCM) dialog commands.

        Examples:
        --------
        >>> title = driver.fedcm.title
        >>> subtitle = driver.fedcm.subtitle
        >>> dialog_type = driver.fedcm.dialog_type
        >>> accounts = driver.fedcm.account_list
        >>> driver.fedcm.select_account(0)
        >>> driver.fedcm.accept()
        >>> driver.fedcm.dismiss()
        >>> driver.fedcm.enable_delay()
        >>> driver.fedcm.disable_delay()
        >>> driver.fedcm.reset_cooldown()
        """
        return self._fedcm

    @property
    def supports_fedcm(self) -> bool:
        """Returns whether the browser supports FedCM capabilities.

        Example:
        --------
        >>> print(driver.supports_fedcm)
        """
        return self.capabilities.get(ArgOptions.FEDCM_CAPABILITY, False)

    def _require_fedcm_support(self):
        """Raises an exception if FedCM is not supported."""
        if not self.supports_fedcm:
            raise WebDriverException(
                "This browser does not support Federated Credential Management. "
                "Please ensure you're using a supported browser."
            )

    @property
    def dialog(self):
        """Returns the FedCM dialog object for interaction.

        Example:
        --------
        >>> dialog = driver.dialog
        """
        self._require_fedcm_support()
        return Dialog(self)

    def fedcm_dialog(self, timeout=5, poll_frequency=0.5, ignored_exceptions=None):
        """Waits for and returns the FedCM dialog.

        Parameters:
        -----------
        timeout : int
            - How long to wait for the dialog

        poll_frequency : floatHow
            - Frequently to poll

        ignored_exceptions : Any
            - Exceptions to ignore while waiting

        Returns:
        -------
            The FedCM dialog object if found

        Raises:
        -------
            TimeoutException if dialog doesn't appear
            WebDriverException if FedCM not supported
        """
        from selenium.common.exceptions import NoAlertPresentException
        from selenium.webdriver.support.wait import WebDriverWait

        self._require_fedcm_support()

        if ignored_exceptions is None:
            ignored_exceptions = (NoAlertPresentException,)

        def _check_fedcm():
            try:
                dialog = Dialog(self)
                return dialog if dialog.type else None
            except NoAlertPresentException:
                return None

        wait = WebDriverWait(self, timeout, poll_frequency=poll_frequency, ignored_exceptions=ignored_exceptions)
        return wait.until(lambda _: _check_fedcm())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\rules.py ===
"""

Rules for building C/API module with f2py2e.

Here is a skeleton of a new wrapper function (13Dec2001):

wrapper_function(args)
  declarations
  get_python_arguments, say, `a' and `b'

  get_a_from_python
  if (successful) {

    get_b_from_python
    if (successful) {

      callfortran
      if (successful) {

        put_a_to_python
        if (successful) {

          put_b_to_python
          if (successful) {

            buildvalue = ...

          }

        }

      }

    }
    cleanup_b

  }
  cleanup_a

  return buildvalue

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import copy
import os
import sys
import time
from pathlib import Path

# __version__.version is now the same as the NumPy version
from . import (
    __version__,
    capi_maps,
    cfuncs,
    common_rules,
    f90mod_rules,
    func2subr,
    use_rules,
)
from .auxfuncs import (
    applyrules,
    debugcapi,
    dictappend,
    errmess,
    gentitle,
    getargs2,
    hascallstatement,
    hasexternals,
    hasinitvalue,
    hasnote,
    hasresultnote,
    isarray,
    isarrayofstrings,
    isattr_value,
    ischaracter,
    ischaracter_or_characterarray,
    ischaracterarray,
    iscomplex,
    iscomplexarray,
    iscomplexfunction,
    iscomplexfunction_warn,
    isdummyroutine,
    isexternal,
    isfunction,
    isfunction_wrap,
    isint1,
    isint1array,
    isintent_aux,
    isintent_c,
    isintent_callback,
    isintent_copy,
    isintent_hide,
    isintent_inout,
    isintent_nothide,
    isintent_out,
    isintent_overwrite,
    islogical,
    islong_complex,
    islong_double,
    islong_doublefunction,
    islong_long,
    islong_longfunction,
    ismoduleroutine,
    isoptional,
    isrequired,
    isscalar,
    issigned_long_longarray,
    isstring,
    isstringarray,
    isstringfunction,
    issubroutine,
    issubroutine_wrap,
    isthreadsafe,
    isunsigned,
    isunsigned_char,
    isunsigned_chararray,
    isunsigned_long_long,
    isunsigned_long_longarray,
    isunsigned_short,
    isunsigned_shortarray,
    l_and,
    l_not,
    l_or,
    outmess,
    replace,
    requiresf90wrapper,
    stripcomma,
)

f2py_version = __version__.version
numpy_version = __version__.version

options = {}
sepdict = {}
# for k in ['need_cfuncs']: sepdict[k]=','
for k in ['decl',
          'frompyobj',
          'cleanupfrompyobj',
          'topyarr', 'method',
          'pyobjfrom', 'closepyobjfrom',
          'freemem',
          'userincludes',
          'includes0', 'includes', 'typedefs', 'typedefs_generated',
          'cppmacros', 'cfuncs', 'callbacks',
          'latexdoc',
          'restdoc',
          'routine_defs', 'externroutines',
          'initf2pywraphooks',
          'commonhooks', 'initcommonhooks',
          'f90modhooks', 'initf90modhooks']:
    sepdict[k] = '\n'

#################### Rules for C/API module #################

generationtime = int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))
module_rules = {
    'modulebody': """\
/* File: #modulename#module.c
 * This file is auto-generated with f2py (version:#f2py_version#).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * Generation date: """ + time.asctime(time.gmtime(generationtime)) + """
 * Do not edit this file directly unless you know what you are doing!!!
 */

#ifdef __cplusplus
extern \"C\" {
#endif

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif /* PY_SSIZE_T_CLEAN */

/* Unconditionally included */
#include <Python.h>
#include <numpy/npy_os.h>

""" + gentitle("See f2py2e/cfuncs.py: includes") + """
#includes#
#includes0#

""" + gentitle("See f2py2e/rules.py: mod_rules['modulebody']") + """
static PyObject *#modulename#_error;
static PyObject *#modulename#_module;

""" + gentitle("See f2py2e/cfuncs.py: typedefs") + """
#typedefs#

""" + gentitle("See f2py2e/cfuncs.py: typedefs_generated") + """
#typedefs_generated#

""" + gentitle("See f2py2e/cfuncs.py: cppmacros") + """
#cppmacros#

""" + gentitle("See f2py2e/cfuncs.py: cfuncs") + """
#cfuncs#

""" + gentitle("See f2py2e/cfuncs.py: userincludes") + """
#userincludes#

""" + gentitle("See f2py2e/capi_rules.py: usercode") + """
#usercode#

/* See f2py2e/rules.py */
#externroutines#

""" + gentitle("See f2py2e/capi_rules.py: usercode1") + """
#usercode1#

""" + gentitle("See f2py2e/cb_rules.py: buildcallback") + """
#callbacks#

""" + gentitle("See f2py2e/rules.py: buildapi") + """
#body#

""" + gentitle("See f2py2e/f90mod_rules.py: buildhooks") + """
#f90modhooks#

""" + gentitle("See f2py2e/rules.py: module_rules['modulebody']") + """

""" + gentitle("See f2py2e/common_rules.py: buildhooks") + """
#commonhooks#

""" + gentitle("See f2py2e/rules.py") + """

static FortranDataDef f2py_routine_defs[] = {
#routine_defs#
    {NULL}
};

static PyMethodDef f2py_module_methods[] = {
#pymethoddef#
    {NULL,NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "#modulename#",
    NULL,
    -1,
    f2py_module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_#modulename#(void) {
    int i;
    PyObject *m,*d, *s, *tmp;
    m = #modulename#_module = PyModule_Create(&moduledef);
    Py_SET_TYPE(&PyFortran_Type, &PyType_Type);
    import_array();
    if (PyErr_Occurred())
        {PyErr_SetString(PyExc_ImportError, \"can't initialize module #modulename# (failed to import numpy)\"); return m;}
    d = PyModule_GetDict(m);
    s = PyUnicode_FromString(\"#f2py_version#\");
    PyDict_SetItemString(d, \"__version__\", s);
    Py_DECREF(s);
    s = PyUnicode_FromString(
        \"This module '#modulename#' is auto-generated with f2py (version:#f2py_version#).\\nFunctions:\\n\"\n#docs#\".\");
    PyDict_SetItemString(d, \"__doc__\", s);
    Py_DECREF(s);
    s = PyUnicode_FromString(\"""" + numpy_version + """\");
    PyDict_SetItemString(d, \"__f2py_numpy_version__\", s);
    Py_DECREF(s);
    #modulename#_error = PyErr_NewException (\"#modulename#.error\", NULL, NULL);
    /*
     * Store the error object inside the dict, so that it could get deallocated.
     * (in practice, this is a module, so it likely will not and cannot.)
     */
    PyDict_SetItemString(d, \"_#modulename#_error\", #modulename#_error);
    Py_DECREF(#modulename#_error);
    for(i=0;f2py_routine_defs[i].name!=NULL;i++) {
        tmp = PyFortranObject_NewAsAttr(&f2py_routine_defs[i]);
        PyDict_SetItemString(d, f2py_routine_defs[i].name, tmp);
        Py_DECREF(tmp);
    }
#initf2pywraphooks#
#initf90modhooks#
#initcommonhooks#
#interface_usercode#

#if Py_GIL_DISABLED
    // signal whether this module supports running with the GIL disabled
    PyUnstable_Module_SetGIL(m , #gil_used#);
#endif

#ifdef F2PY_REPORT_ATEXIT
    if (! PyErr_Occurred())
        on_exit(f2py_report_on_exit,(void*)\"#modulename#\");
#endif

    if (PyType_Ready(&PyFortran_Type) < 0) {
        return NULL;
    }

    return m;
}
#ifdef __cplusplus
}
#endif
""",
    'separatorsfor': {'latexdoc': '\n\n',
                      'restdoc': '\n\n'},
    'latexdoc': ['\\section{Module \\texttt{#texmodulename#}}\n',
                 '#modnote#\n',
                 '#latexdoc#'],
    'restdoc': ['Module #modulename#\n' + '=' * 80,
                '\n#restdoc#']
}

defmod_rules = [
    {'body': '/*eof body*/',
     'method': '/*eof method*/',
     'externroutines': '/*eof externroutines*/',
     'routine_defs': '/*eof routine_defs*/',
     'initf90modhooks': '/*eof initf90modhooks*/',
     'initf2pywraphooks': '/*eof initf2pywraphooks*/',
     'initcommonhooks': '/*eof initcommonhooks*/',
     'latexdoc': '',
     'restdoc': '',
     'modnote': {hasnote: '#note#', l_not(hasnote): ''},
     }
]

routine_rules = {
    'separatorsfor': sepdict,
    'body': """
#begintitle#
static char doc_#apiname#[] = \"\\\n#docreturn##name#(#docsignatureshort#)\\n\\nWrapper for ``#name#``.\\\n\\n#docstrsigns#\";
/* #declfortranroutine# */
static PyObject *#apiname#(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           #functype# (*f2py_func)(#callprotoargument#)) {
    PyObject * volatile capi_buildvalue = NULL;
    volatile int f2py_success = 1;
#decl#
    static char *capi_kwlist[] = {#kwlist##kwlistopt##kwlistxa#NULL};
#usercode#
#routdebugenter#
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
    if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\\
        \"#argformat#|#keyformat##xaformat#:#pyname#\",\\
        capi_kwlist#args_capi##keys_capi##keys_xa#))\n        return NULL;
#frompyobj#
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
#callfortranroutine#
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
        if (f2py_success) {
#pyobjfrom#
/*end of pyobjfrom*/
        CFUNCSMESS(\"Building return value.\\n\");
        capi_buildvalue = Py_BuildValue(\"#returnformat#\"#return#);
/*closepyobjfrom*/
#closepyobjfrom#
        } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
#cleanupfrompyobj#
    if (capi_buildvalue == NULL) {
#routdebugfailure#
    } else {
#routdebugleave#
    }
    CFUNCSMESS(\"Freeing memory.\\n\");
#freemem#
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
    return capi_buildvalue;
}
#endtitle#
""",
    'routine_defs': '#routine_def#',
    'initf2pywraphooks': '#initf2pywraphook#',
    'externroutines': '#declfortranroutine#',
    'doc': '#docreturn##name#(#docsignature#)',
    'docshort': '#docreturn##name#(#docsignatureshort#)',
    'docs': '"    #docreturn##name#(#docsignature#)\\n"\n',
    'need': ['arrayobject.h', 'CFUNCSMESS', 'MINMAX'],
    'cppmacros': {debugcapi: '#define DEBUGCFUNCS'},
    'latexdoc': ['\\subsection{Wrapper function \\texttt{#texname#}}\n',
                 """
\\noindent{{}\\verb@#docreturn##name#@{}}\\texttt{(#latexdocsignatureshort#)}
#routnote#

#latexdocstrsigns#
"""],
    'restdoc': ['Wrapped function ``#name#``\n' + '-' * 80,

                ]
}

################## Rules for C/API function ##############

rout_rules = [
    {  # Init
        'separatorsfor': {'callfortranroutine': '\n', 'routdebugenter': '\n', 'decl': '\n',
                          'routdebugleave': '\n', 'routdebugfailure': '\n',
                          'setjmpbuf': ' || ',
                          'docstrreq': '\n', 'docstropt': '\n', 'docstrout': '\n',
                          'docstrcbs': '\n', 'docstrsigns': '\\n"\n"',
                          'latexdocstrsigns': '\n',
                          'latexdocstrreq': '\n', 'latexdocstropt': '\n',
                          'latexdocstrout': '\n', 'latexdocstrcbs': '\n',
                          },
        'kwlist': '', 'kwlistopt': '', 'callfortran': '', 'callfortranappend': '',
        'docsign': '', 'docsignopt': '', 'decl': '/*decl*/',
        'freemem': '/*freemem*/',
        'docsignshort': '', 'docsignoptshort': '',
        'docstrsigns': '', 'latexdocstrsigns': '',
        'docstrreq': '\\nParameters\\n----------',
        'docstropt': '\\nOther Parameters\\n----------------',
        'docstrout': '\\nReturns\\n-------',
        'docstrcbs': '\\nNotes\\n-----\\nCall-back functions::\\n',
        'latexdocstrreq': '\\noindent Required arguments:',
        'latexdocstropt': '\\noindent Optional arguments:',
        'latexdocstrout': '\\noindent Return objects:',
        'latexdocstrcbs': '\\noindent Call-back functions:',
        'args_capi': '', 'keys_capi': '', 'functype': '',
        'frompyobj': '/*frompyobj*/',
        # this list will be reversed
        'cleanupfrompyobj': ['/*end of cleanupfrompyobj*/'],
        'pyobjfrom': '/*pyobjfrom*/',
        # this list will be reversed
        'closepyobjfrom': ['/*end of closepyobjfrom*/'],
        'topyarr': '/*topyarr*/', 'routdebugleave': '/*routdebugleave*/',
        'routdebugenter': '/*routdebugenter*/',
        'routdebugfailure': '/*routdebugfailure*/',
        'callfortranroutine': '/*callfortranroutine*/',
        'argformat': '', 'keyformat': '', 'need_cfuncs': '',
        'docreturn': '', 'return': '', 'returnformat': '', 'rformat': '',
        'kwlistxa': '', 'keys_xa': '', 'xaformat': '', 'docsignxa': '', 'docsignxashort': '',
        'initf2pywraphook': '',
        'routnote': {hasnote: '--- #note#', l_not(hasnote): ''},
    }, {
        'apiname': 'f2py_rout_#modulename#_#name#',
        'pyname': '#modulename#.#name#',
        'decl': '',
        '_check': l_not(ismoduleroutine)
    }, {
        'apiname': 'f2py_rout_#modulename#_#f90modulename#_#name#',
        'pyname': '#modulename#.#f90modulename#.#name#',
        'decl': '',
        '_check': ismoduleroutine
    }, {  # Subroutine
        'functype': 'void',
        'declfortranroutine': {l_and(l_not(l_or(ismoduleroutine, isintent_c)), l_not(isdummyroutine)): 'extern void #F_FUNC#(#fortranname#,#FORTRANNAME#)(#callprotoargument#);',
                               l_and(l_not(ismoduleroutine), isintent_c, l_not(isdummyroutine)): 'extern void #fortranname#(#callprotoargument#);',
                               ismoduleroutine: '',
                               isdummyroutine: ''
                               },
        'routine_def': {
            l_not(l_or(ismoduleroutine, isintent_c, isdummyroutine)):
            '    {\"#name#\",-1,{{-1}},0,0,(char *)'
            '  #F_FUNC#(#fortranname#,#FORTRANNAME#),'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
            l_and(l_not(ismoduleroutine), isintent_c, l_not(isdummyroutine)):
            '    {\"#name#\",-1,{{-1}},0,0,(char *)#fortranname#,'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
            l_and(l_not(ismoduleroutine), isdummyroutine):
            '    {\"#name#\",-1,{{-1}},0,0,NULL,'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
        },
        'need': {l_and(l_not(l_or(ismoduleroutine, isintent_c)), l_not(isdummyroutine)): 'F_FUNC'},
        'callfortranroutine': [
            {debugcapi: [
                """    fprintf(stderr,\"debug-capi:Fortran subroutine `#fortranname#(#callfortran#)\'\\n\");"""]},
            {hasexternals: """\
        if (#setjmpbuf#) {
            f2py_success = 0;
        } else {"""},
            {isthreadsafe: '            Py_BEGIN_ALLOW_THREADS'},
            {hascallstatement: '''                #callstatement#;
                /*(*f2py_func)(#callfortran#);*/'''},
            {l_not(l_or(hascallstatement, isdummyroutine))
                   : '                (*f2py_func)(#callfortran#);'},
            {isthreadsafe: '            Py_END_ALLOW_THREADS'},
            {hasexternals: """        }"""}
        ],
        '_check': l_and(issubroutine, l_not(issubroutine_wrap)),
    }, {  # Wrapped function
        'functype': 'void',
        'declfortranroutine': {l_not(l_or(ismoduleroutine, isdummyroutine)): 'extern void #F_WRAPPEDFUNC#(#name_lower#,#NAME#)(#callprotoargument#);',
                               isdummyroutine: '',
                               },

        'routine_def': {
            l_not(l_or(ismoduleroutine, isdummyroutine)):
            '    {\"#name#\",-1,{{-1}},0,0,(char *)'
            '  #F_WRAPPEDFUNC#(#name_lower#,#NAME#),'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
            isdummyroutine:
            '    {\"#name#\",-1,{{-1}},0,0,NULL,'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
        },
        'initf2pywraphook': {l_not(l_or(ismoduleroutine, isdummyroutine)): '''
    {
      extern #ctype# #F_FUNC#(#name_lower#,#NAME#)(void);
      PyObject* o = PyDict_GetItemString(d,"#name#");
      tmp = F2PyCapsule_FromVoidPtr((void*)#F_WRAPPEDFUNC#(#name_lower#,#NAME#),NULL);
      PyObject_SetAttrString(o,"_cpointer", tmp);
      Py_DECREF(tmp);
      s = PyUnicode_FromString("#name#");
      PyObject_SetAttrString(o,"__name__", s);
      Py_DECREF(s);
    }
    '''},
        'need': {l_not(l_or(ismoduleroutine, isdummyroutine)): ['F_WRAPPEDFUNC', 'F_FUNC']},
        'callfortranroutine': [
            {debugcapi: [
                """    fprintf(stderr,\"debug-capi:Fortran subroutine `f2pywrap#name_lower#(#callfortran#)\'\\n\");"""]},
            {hasexternals: """\
    if (#setjmpbuf#) {
        f2py_success = 0;
    } else {"""},
            {isthreadsafe: '    Py_BEGIN_ALLOW_THREADS'},
            {l_not(l_or(hascallstatement, isdummyroutine))
                   : '    (*f2py_func)(#callfortran#);'},
            {hascallstatement:
                '    #callstatement#;\n    /*(*f2py_func)(#callfortran#);*/'},
            {isthreadsafe: '    Py_END_ALLOW_THREADS'},
            {hasexternals: '    }'}
        ],
        '_check': isfunction_wrap,
    }, {  # Wrapped subroutine
        'functype': 'void',
        'declfortranroutine': {l_not(l_or(ismoduleroutine, isdummyroutine)): 'extern void #F_WRAPPEDFUNC#(#name_lower#,#NAME#)(#callprotoargument#);',
                               isdummyroutine: '',
                               },

        'routine_def': {
            l_not(l_or(ismoduleroutine, isdummyroutine)):
            '    {\"#name#\",-1,{{-1}},0,0,(char *)'
            '  #F_WRAPPEDFUNC#(#name_lower#,#NAME#),'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
            isdummyroutine:
            '    {\"#name#\",-1,{{-1}},0,0,NULL,'
            '  (f2py_init_func)#apiname#,doc_#apiname#},',
        },
        'initf2pywraphook': {l_not(l_or(ismoduleroutine, isdummyroutine)): '''
    {
      extern void #F_FUNC#(#name_lower#,#NAME#)(void);
      PyObject* o = PyDict_GetItemString(d,"#name#");
      tmp = F2PyCapsule_FromVoidPtr((void*)#F_FUNC#(#name_lower#,#NAME#),NULL);
      PyObject_SetAttrString(o,"_cpointer", tmp);
      Py_DECREF(tmp);
      s = PyUnicode_FromString("#name#");
      PyObject_SetAttrString(o,"__name__", s);
      Py_DECREF(s);
    }
    '''},
        'need': {l_not(l_or(ismoduleroutine, isdummyroutine)): ['F_WRAPPEDFUNC', 'F_FUNC']},
        'callfortranroutine': [
            {debugcapi: [
                """    fprintf(stderr,\"debug-capi:Fortran subroutine `f2pywrap#name_lower#(#callfortran#)\'\\n\");"""]},
            {hasexternals: """\
    if (#setjmpbuf#) {
        f2py_success = 0;
    } else {"""},
            {isthreadsafe: '    Py_BEGIN_ALLOW_THREADS'},
            {l_not(l_or(hascallstatement, isdummyroutine))
                   : '    (*f2py_func)(#callfortran#);'},
            {hascallstatement:
                '    #callstatement#;\n    /*(*f2py_func)(#callfortran#);*/'},
            {isthreadsafe: '    Py_END_ALLOW_THREADS'},
            {hasexternals: '    }'}
        ],
        '_check': issubroutine_wrap,
    }, {  # Function
        'functype': '#ctype#',
        'docreturn': {l_not(isintent_hide): '#rname#,'},
        'docstrout': '#pydocsignout#',
        'latexdocstrout': ['\\item[]{{}\\verb@#pydocsignout#@{}}',
                           {hasresultnote: '--- #resultnote#'}],
        'callfortranroutine': [{l_and(debugcapi, isstringfunction): """\
#ifdef USESCOMPAQFORTRAN
    fprintf(stderr,\"debug-capi:Fortran function #ctype# #fortranname#(#callcompaqfortran#)\\n\");
#else
    fprintf(stderr,\"debug-capi:Fortran function #ctype# #fortranname#(#callfortran#)\\n\");
#endif
"""},
                               {l_and(debugcapi, l_not(isstringfunction)): """\
    fprintf(stderr,\"debug-capi:Fortran function #ctype# #fortranname#(#callfortran#)\\n\");
"""}
                               ],
        '_check': l_and(isfunction, l_not(isfunction_wrap))
    }, {  # Scalar function
        'declfortranroutine': {l_and(l_not(l_or(ismoduleroutine, isintent_c)), l_not(isdummyroutine)): 'extern #ctype# #F_FUNC#(#fortranname#,#FORTRANNAME#)(#callprotoargument#);',
                               l_and(l_not(ismoduleroutine), isintent_c, l_not(isdummyroutine)): 'extern #ctype# #fortranname#(#callprotoargument#);',
                               isdummyroutine: ''
                               },
        'routine_def': {
            l_and(l_not(l_or(ismoduleroutine, isintent_c)),
                  l_not(isdummyroutine)):
            ('    {\"#name#\",-1,{{-1}},0,0,(char *)'
             '  #F_FUNC#(#fortranname#,#FORTRANNAME#),'
             '  (f2py_init_func)#apiname#,doc_#apiname#},'),
            l_and(l_not(ismoduleroutine), isintent_c, l_not(isdummyroutine)):
            ('    {\"#name#\",-1,{{-1}},0,0,(char *)#fortranname#,'
             '  (f2py_init_func)#apiname#,doc_#apiname#},'),
            isdummyroutine:
            '    {\"#name#\",-1,{{-1}},0,0,NULL,'
            '(f2py_init_func)#apiname#,doc_#apiname#},',
        },
        'decl': [{iscomplexfunction_warn: '    #ctype# #name#_return_value={0,0};',
                  l_not(iscomplexfunction): '    #ctype# #name#_return_value=0;'},
                 {iscomplexfunction:
                  '    PyObject *#name#_return_value_capi = Py_None;'}
                 ],
        'callfortranroutine': [
            {hasexternals: """\
    if (#setjmpbuf#) {
        f2py_success = 0;
    } else {"""},
            {isthreadsafe: '    Py_BEGIN_ALLOW_THREADS'},
            {hascallstatement: '''    #callstatement#;
/*    #name#_return_value = (*f2py_func)(#callfortran#);*/
'''},
            {l_not(l_or(hascallstatement, isdummyroutine))
                   : '    #name#_return_value = (*f2py_func)(#callfortran#);'},
            {isthreadsafe: '    Py_END_ALLOW_THREADS'},
            {hasexternals: '    }'},
            {l_and(debugcapi, iscomplexfunction)
                   : '    fprintf(stderr,"#routdebugshowvalue#\\n",#name#_return_value.r,#name#_return_value.i);'},
            {l_and(debugcapi, l_not(iscomplexfunction)): '    fprintf(stderr,"#routdebugshowvalue#\\n",#name#_return_value);'}],
        'pyobjfrom': {iscomplexfunction: '    #name#_return_value_capi = pyobj_from_#ctype#1(#name#_return_value);'},
        'need': [{l_not(isdummyroutine): 'F_FUNC'},
                 {iscomplexfunction: 'pyobj_from_#ctype#1'},
                 {islong_longfunction: 'long_long'},
                 {islong_doublefunction: 'long_double'}],
        'returnformat': {l_not(isintent_hide): '#rformat#'},
        'return': {iscomplexfunction: ',#name#_return_value_capi',
                   l_not(l_or(iscomplexfunction, isintent_hide)): ',#name#_return_value'},
        '_check': l_and(isfunction, l_not(isstringfunction), l_not(isfunction_wrap))
    }, {  # String function # in use for --no-wrap
        'declfortranroutine': 'extern void #F_FUNC#(#fortranname#,#FORTRANNAME#)(#callprotoargument#);',
        'routine_def': {l_not(l_or(ismoduleroutine, isintent_c)):
                        '    {\"#name#\",-1,{{-1}},0,0,(char *)#F_FUNC#(#fortranname#,#FORTRANNAME#),(f2py_init_func)#apiname#,doc_#apiname#},',
                        l_and(l_not(ismoduleroutine), isintent_c):
                        '    {\"#name#\",-1,{{-1}},0,0,(char *)#fortranname#,(f2py_init_func)#apiname#,doc_#apiname#},'
                        },
        'decl': ['    #ctype# #name#_return_value = NULL;',
                 '    int #name#_return_value_len = 0;'],
        'callfortran': '#name#_return_value,#name#_return_value_len,',
        'callfortranroutine': ['    #name#_return_value_len = #rlength#;',
                               '    if ((#name#_return_value = (string)malloc(#name#_return_value_len+1) == NULL) {',
                               '        PyErr_SetString(PyExc_MemoryError, \"out of memory\");',
                               '        f2py_success = 0;',
                               '    } else {',
                               "        (#name#_return_value)[#name#_return_value_len] = '\\0';",
                               '    }',
                               '    if (f2py_success) {',
                               {hasexternals: """\
        if (#setjmpbuf#) {
            f2py_success = 0;
        } else {"""},
                               {isthreadsafe: '        Py_BEGIN_ALLOW_THREADS'},
                              """\
#ifdef USESCOMPAQFORTRAN
        (*f2py_func)(#callcompaqfortran#);
#else
        (*f2py_func)(#callfortran#);
#endif
""",
                               {isthreadsafe: '        Py_END_ALLOW_THREADS'},
                               {hasexternals: '        }'},
                               {debugcapi:
                                  '        fprintf(stderr,"#routdebugshowvalue#\\n",#name#_return_value_len,#name#_return_value);'},
                               '    } /* if (f2py_success) after (string)malloc */',
                              ],
        'returnformat': '#rformat#',
        'return': ',#name#_return_value',
        'freemem': '    STRINGFREE(#name#_return_value);',
        'need': ['F_FUNC', '#ctype#', 'STRINGFREE'],
        '_check': l_and(isstringfunction, l_not(isfunction_wrap))  # ???obsolete
    },
    {  # Debugging
        'routdebugenter': '    fprintf(stderr,"debug-capi:Python C/API function #modulename#.#name#(#docsignature#)\\n");',
        'routdebugleave': '    fprintf(stderr,"debug-capi:Python C/API function #modulename#.#name#: successful.\\n");',
        'routdebugfailure': '    fprintf(stderr,"debug-capi:Python C/API function #modulename#.#name#: failure.\\n");',
        '_check': debugcapi
    }
]

################ Rules for arguments ##################

typedef_need_dict = {islong_long: 'long_long',
                     islong_double: 'long_double',
                     islong_complex: 'complex_long_double',
                     isunsigned_char: 'unsigned_char',
                     isunsigned_short: 'unsigned_short',
                     isunsigned: 'unsigned',
                     isunsigned_long_long: 'unsigned_long_long',
                     isunsigned_chararray: 'unsigned_char',
                     isunsigned_shortarray: 'unsigned_short',
                     isunsigned_long_longarray: 'unsigned_long_long',
                     issigned_long_longarray: 'long_long',
                     isint1: 'signed_char',
                     ischaracter_or_characterarray: 'character',
                     }

aux_rules = [
    {
        'separatorsfor': sepdict
    },
    {  # Common
        'frompyobj': ['    /* Processing auxiliary variable #varname# */',
                      {debugcapi: '    fprintf(stderr,"#vardebuginfo#\\n");'}, ],
        'cleanupfrompyobj': '    /* End of cleaning variable #varname# */',
        'need': typedef_need_dict,
    },
    # Scalars (not complex)
    {  # Common
        'decl': '    #ctype# #varname# = 0;',
        'need': {hasinitvalue: 'math.h'},
        'frompyobj': {hasinitvalue: '    #varname# = #init#;'},
        '_check': l_and(isscalar, l_not(iscomplex)),
    },
    {
        'return': ',#varname#',
        'docstrout': '#pydocsignout#',
        'docreturn': '#outvarname#,',
        'returnformat': '#varrformat#',
        '_check': l_and(isscalar, l_not(iscomplex), isintent_out),
    },
    # Complex scalars
    {  # Common
        'decl': '    #ctype# #varname#;',
        'frompyobj': {hasinitvalue: '    #varname#.r = #init.r#, #varname#.i = #init.i#;'},
        '_check': iscomplex
    },
    # String
    {  # Common
        'decl': ['    #ctype# #varname# = NULL;',
                 '    int slen(#varname#);',
                 ],
        'need': ['len..'],
        '_check': isstring
    },
    # Array
    {  # Common
        'decl': ['    #ctype# *#varname# = NULL;',
                 '    npy_intp #varname#_Dims[#rank#] = {#rank*[-1]#};',
                 '    const int #varname#_Rank = #rank#;',
                 ],
        'need': ['len..', {hasinitvalue: 'forcomb'}, {hasinitvalue: 'CFUNCSMESS'}],
        '_check': isarray
    },
    # Scalararray
    {  # Common
        '_check': l_and(isarray, l_not(iscomplexarray))
    }, {  # Not hidden
        '_check': l_and(isarray, l_not(iscomplexarray), isintent_nothide)
    },
    # Integer*1 array
    {'need': '#ctype#',
     '_check': isint1array,
     '_depend': ''
     },
    # Integer*-1 array
    {'need': '#ctype#',
     '_check': l_or(isunsigned_chararray, isunsigned_char),
     '_depend': ''
     },
    # Integer*-2 array
    {'need': '#ctype#',
     '_check': isunsigned_shortarray,
     '_depend': ''
     },
    # Integer*-8 array
    {'need': '#ctype#',
     '_check': isunsigned_long_longarray,
     '_depend': ''
     },
    # Complexarray
    {'need': '#ctype#',
     '_check': iscomplexarray,
     '_depend': ''
     },
    # Stringarray
    {
        'callfortranappend': {isarrayofstrings: 'flen(#varname#),'},
        'need': 'string',
        '_check': isstringarray
    }
]

arg_rules = [
    {
        'separatorsfor': sepdict
    },
    {  # Common
        'frompyobj': ['    /* Processing variable #varname# */',
                      {debugcapi: '    fprintf(stderr,"#vardebuginfo#\\n");'}, ],
        'cleanupfrompyobj': '    /* End of cleaning variable #varname# */',
        '_depend': '',
        'need': typedef_need_dict,
    },
    # Doc signatures
    {
        'docstropt': {l_and(isoptional, isintent_nothide): '#pydocsign#'},
        'docstrreq': {l_and(isrequired, isintent_nothide): '#pydocsign#'},
        'docstrout': {isintent_out: '#pydocsignout#'},
        'latexdocstropt': {l_and(isoptional, isintent_nothide): ['\\item[]{{}\\verb@#pydocsign#@{}}',
                                                                 {hasnote: '--- #note#'}]},
        'latexdocstrreq': {l_and(isrequired, isintent_nothide): ['\\item[]{{}\\verb@#pydocsign#@{}}',
                                                                 {hasnote: '--- #note#'}]},
        'latexdocstrout': {isintent_out: ['\\item[]{{}\\verb@#pydocsignout#@{}}',
                                          {l_and(hasnote, isintent_hide): '--- #note#',
                                           l_and(hasnote, isintent_nothide): '--- See above.'}]},
        'depend': ''
    },
    # Required/Optional arguments
    {
        'kwlist': '"#varname#",',
        'docsign': '#varname#,',
        '_check': l_and(isintent_nothide, l_not(isoptional))
    },
    {
        'kwlistopt': '"#varname#",',
        'docsignopt': '#varname#=#showinit#,',
        'docsignoptshort': '#varname#,',
        '_check': l_and(isintent_nothide, isoptional)
    },
    # Docstring/BuildValue
    {
        'docreturn': '#outvarname#,',
        'returnformat': '#varrformat#',
        '_check': isintent_out
    },
    # Externals (call-back functions)
    {  # Common
        'docsignxa': {isintent_nothide: '#varname#_extra_args=(),'},
        'docsignxashort': {isintent_nothide: '#varname#_extra_args,'},
        'docstropt': {isintent_nothide: '#varname#_extra_args : input tuple, optional\\n    Default: ()'},
        'docstrcbs': '#cbdocstr#',
        'latexdocstrcbs': '\\item[] #cblatexdocstr#',
        'latexdocstropt': {isintent_nothide: '\\item[]{{}\\verb@#varname#_extra_args := () input tuple@{}} --- Extra arguments for call-back function {{}\\verb@#varname#@{}}.'},
        'decl': ['    #cbname#_t #varname#_cb = { Py_None, NULL, 0 };',
                 '    #cbname#_t *#varname#_cb_ptr = &#varname#_cb;',
                 '    PyTupleObject *#varname#_xa_capi = NULL;',
                 {l_not(isintent_callback):
                  '    #cbname#_typedef #varname#_cptr;'}
                 ],
        'kwlistxa': {isintent_nothide: '"#varname#_extra_args",'},
        'argformat': {isrequired: 'O'},
        'keyformat': {isoptional: 'O'},
        'xaformat': {isintent_nothide: 'O!'},
        'args_capi': {isrequired: ',&#varname#_cb.capi'},
        'keys_capi': {isoptional: ',&#varname#_cb.capi'},
        'keys_xa': ',&PyTuple_Type,&#varname#_xa_capi',
        'setjmpbuf': '(setjmp(#varname#_cb.jmpbuf))',
        'callfortran': {l_not(isintent_callback): '#varname#_cptr,'},
        'need': ['#cbname#', 'setjmp.h'],
        '_check': isexternal
    },
    {
        'frompyobj': [{l_not(isintent_callback): """\
if(F2PyCapsule_Check(#varname#_cb.capi)) {
  #varname#_cptr = F2PyCapsule_AsVoidPtr(#varname#_cb.capi);
} else {
  #varname#_cptr = #cbname#;
}
"""}, {isintent_callback: """\
if (#varname#_cb.capi==Py_None) {
  #varname#_cb.capi = PyObject_GetAttrString(#modulename#_module,\"#varname#\");
  if (#varname#_cb.capi) {
    if (#varname#_xa_capi==NULL) {
      if (PyObject_HasAttrString(#modulename#_module,\"#varname#_extra_args\")) {
        PyObject* capi_tmp = PyObject_GetAttrString(#modulename#_module,\"#varname#_extra_args\");
        if (capi_tmp) {
          #varname#_xa_capi = (PyTupleObject *)PySequence_Tuple(capi_tmp);
          Py_DECREF(capi_tmp);
        }
        else {
          #varname#_xa_capi = (PyTupleObject *)Py_BuildValue(\"()\");
        }
        if (#varname#_xa_capi==NULL) {
          PyErr_SetString(#modulename#_error,\"Failed to convert #modulename#.#varname#_extra_args to tuple.\\n\");
          return NULL;
        }
      }
    }
  }
  if (#varname#_cb.capi==NULL) {
    PyErr_SetString(#modulename#_error,\"Callback #varname# not defined (as an argument or module #modulename# attribute).\\n\");
    return NULL;
  }
}
"""},
            """\
    if (create_cb_arglist(#varname#_cb.capi,#varname#_xa_capi,#maxnofargs#,#nofoptargs#,&#varname#_cb.nofargs,&#varname#_cb.args_capi,\"failed in processing argument list for call-back #varname#.\")) {
""",
            {debugcapi: ["""\
        fprintf(stderr,\"debug-capi:Assuming %d arguments; at most #maxnofargs#(-#nofoptargs#) is expected.\\n\",#varname#_cb.nofargs);
        CFUNCSMESSPY(\"for #varname#=\",#varname#_cb.capi);""",
                         {l_not(isintent_callback): """        fprintf(stderr,\"#vardebugshowvalue# (call-back in C).\\n\",#cbname#);"""}]},
            """\
        CFUNCSMESS(\"Saving callback variables for `#varname#`.\\n\");
        #varname#_cb_ptr = swap_active_#cbname#(#varname#_cb_ptr);""",
        ],
        'cleanupfrompyobj':
        """\
        CFUNCSMESS(\"Restoring callback variables for `#varname#`.\\n\");
        #varname#_cb_ptr = swap_active_#cbname#(#varname#_cb_ptr);
        Py_DECREF(#varname#_cb.args_capi);
    }""",
        'need': ['SWAP', 'create_cb_arglist'],
        '_check': isexternal,
        '_depend': ''
    },
    # Scalars (not complex)
    {  # Common
        'decl': '    #ctype# #varname# = 0;',
        'pyobjfrom': {debugcapi: '    fprintf(stderr,"#vardebugshowvalue#\\n",#varname#);'},
        'callfortran': {l_or(isintent_c, isattr_value): '#varname#,', l_not(l_or(isintent_c, isattr_value)): '&#varname#,'},
        'return': {isintent_out: ',#varname#'},
        '_check': l_and(isscalar, l_not(iscomplex))
    }, {
        'need': {hasinitvalue: 'math.h'},
        '_check': l_and(isscalar, l_not(iscomplex)),
    }, {  # Not hidden
        'decl': '    PyObject *#varname#_capi = Py_None;',
        'argformat': {isrequired: 'O'},
        'keyformat': {isoptional: 'O'},
        'args_capi': {isrequired: ',&#varname#_capi'},
        'keys_capi': {isoptional: ',&#varname#_capi'},
        'pyobjfrom': {isintent_inout: """\
    f2py_success = try_pyarr_from_#ctype#(#varname#_capi,&#varname#);
    if (f2py_success) {"""},
        'closepyobjfrom': {isintent_inout: "    } /*if (f2py_success) of #varname# pyobjfrom*/"},
        'need': {isintent_inout: 'try_pyarr_from_#ctype#'},
        '_check': l_and(isscalar, l_not(iscomplex), l_not(isstring),
                        isintent_nothide)
    }, {
        'frompyobj': [
            # hasinitvalue...
            #   if pyobj is None:
            #     varname = init
            #   else
            #     from_pyobj(varname)
            #
            # isoptional and noinitvalue...
            #   if pyobj is not None:
            #     from_pyobj(varname)
            #   else:
            #     varname is uninitialized
            #
            # ...
            #   from_pyobj(varname)
            #
            {hasinitvalue: '    if (#varname#_capi == Py_None) #varname# = #init#; else',
             '_depend': ''},
            {l_and(isoptional, l_not(hasinitvalue)): '    if (#varname#_capi != Py_None)',
             '_depend': ''},
            {l_not(islogical): '''\
        f2py_success = #ctype#_from_pyobj(&#varname#,#varname#_capi,"#pyname#() #nth# (#varname#) can\'t be converted to #ctype#");
    if (f2py_success) {'''},
            {islogical: '''\
        #varname# = (#ctype#)PyObject_IsTrue(#varname#_capi);
        f2py_success = 1;
    if (f2py_success) {'''},
        ],
        'cleanupfrompyobj': '    } /*if (f2py_success) of #varname#*/',
        'need': {l_not(islogical): '#ctype#_from_pyobj'},
        '_check': l_and(isscalar, l_not(iscomplex), isintent_nothide),
        '_depend': ''
    }, {  # Hidden
        'frompyobj': {hasinitvalue: '    #varname# = #init#;'},
        'need': typedef_need_dict,
        '_check': l_and(isscalar, l_not(iscomplex), isintent_hide),
        '_depend': ''
    }, {  # Common
        'frompyobj': {debugcapi: '    fprintf(stderr,"#vardebugshowvalue#\\n",#varname#);'},
        '_check': l_and(isscalar, l_not(iscomplex)),
        '_depend': ''
    },
    # Complex scalars
    {  # Common
        'decl': '    #ctype# #varname#;',
        'callfortran': {isintent_c: '#varname#,', l_not(isintent_c): '&#varname#,'},
        'pyobjfrom': {debugcapi: '    fprintf(stderr,"#vardebugshowvalue#\\n",#varname#.r,#varname#.i);'},
        'return': {isintent_out: ',#varname#_capi'},
        '_check': iscomplex
    }, {  # Not hidden
        'decl': '    PyObject *#varname#_capi = Py_None;',
        'argformat': {isrequired: 'O'},
        'keyformat': {isoptional: 'O'},
        'args_capi': {isrequired: ',&#varname#_capi'},
        'keys_capi': {isoptional: ',&#varname#_capi'},
        'need': {isintent_inout: 'try_pyarr_from_#ctype#'},
        'pyobjfrom': {isintent_inout: """\
        f2py_success = try_pyarr_from_#ctype#(#varname#_capi,&#varname#);
        if (f2py_success) {"""},
        'closepyobjfrom': {isintent_inout: "        } /*if (f2py_success) of #varname# pyobjfrom*/"},
        '_check': l_and(iscomplex, isintent_nothide)
    }, {
        'frompyobj': [{hasinitvalue: '    if (#varname#_capi==Py_None) {#varname#.r = #init.r#, #varname#.i = #init.i#;} else'},
                      {l_and(isoptional, l_not(hasinitvalue))
                             : '    if (#varname#_capi != Py_None)'},
                      '        f2py_success = #ctype#_from_pyobj(&#varname#,#varname#_capi,"#pyname#() #nth# (#varname#) can\'t be converted to #ctype#");'
                      '\n    if (f2py_success) {'],
        'cleanupfrompyobj': '    }  /*if (f2py_success) of #varname# frompyobj*/',
        'need': ['#ctype#_from_pyobj'],
        '_check': l_and(iscomplex, isintent_nothide),
        '_depend': ''
    }, {  # Hidden
        'decl': {isintent_out: '    PyObject *#varname#_capi = Py_None;'},
        '_check': l_and(iscomplex, isintent_hide)
    }, {
        'frompyobj': {hasinitvalue: '    #varname#.r = #init.r#, #varname#.i = #init.i#;'},
        '_check': l_and(iscomplex, isintent_hide),
        '_depend': ''
    }, {  # Common
        'pyobjfrom': {isintent_out: '    #varname#_capi = pyobj_from_#ctype#1(#varname#);'},
        'need': ['pyobj_from_#ctype#1'],
        '_check': iscomplex
    }, {
        'frompyobj': {debugcapi: '    fprintf(stderr,"#vardebugshowvalue#\\n",#varname#.r,#varname#.i);'},
        '_check': iscomplex,
        '_depend': ''
    },
    # String
    {  # Common
        'decl': ['    #ctype# #varname# = NULL;',
                 '    int slen(#varname#);',
                 '    PyObject *#varname#_capi = Py_None;'],
        'callfortran': '#varname#,',
        'callfortranappend': 'slen(#varname#),',
        'pyobjfrom': [
            {debugcapi:
             '    fprintf(stderr,'
             '"#vardebugshowvalue#\\n",slen(#varname#),#varname#);'},
            # The trailing null value for Fortran is blank.
            {l_and(isintent_out, l_not(isintent_c)):
             "        STRINGPADN(#varname#, slen(#varname#), ' ', '\\0');"},
        ],
        'return': {isintent_out: ',#varname#'},
        'need': ['len..',
                 {l_and(isintent_out, l_not(isintent_c)): 'STRINGPADN'}],
        '_check': isstring
    }, {  # Common
        'frompyobj': [
            """\
    slen(#varname#) = #elsize#;
    f2py_success = #ctype#_from_pyobj(&#varname#,&slen(#varname#),#init#,"""
"""#varname#_capi,\"#ctype#_from_pyobj failed in converting #nth#"""
"""`#varname#\' of #pyname# to C #ctype#\");
    if (f2py_success) {""",
            # The trailing null value for Fortran is blank.
            {l_not(isintent_c):
             "        STRINGPADN(#varname#, slen(#varname#), '\\0', ' ');"},
        ],
        'cleanupfrompyobj': """\
        STRINGFREE(#varname#);
    }  /*if (f2py_success) of #varname#*/""",
        'need': ['#ctype#_from_pyobj', 'len..', 'STRINGFREE',
                 {l_not(isintent_c): 'STRINGPADN'}],
        '_check': isstring,
        '_depend': ''
    }, {  # Not hidden
        'argformat': {isrequired: 'O'},
        'keyformat': {isoptional: 'O'},
        'args_capi': {isrequired: ',&#varname#_capi'},
        'keys_capi': {isoptional: ',&#varname#_capi'},
        'pyobjfrom': [
            {l_and(isintent_inout, l_not(isintent_c)):
             "        STRINGPADN(#varname#, slen(#varname#), ' ', '\\0');"},
            {isintent_inout: '''\
    f2py_success = try_pyarr_from_#ctype#(#varname#_capi, #varname#,
                                          slen(#varname#));
    if (f2py_success) {'''}],
        'closepyobjfrom': {isintent_inout: '    } /*if (f2py_success) of #varname# pyobjfrom*/'},
        'need': {isintent_inout: 'try_pyarr_from_#ctype#',
                 l_and(isintent_inout, l_not(isintent_c)): 'STRINGPADN'},
        '_check': l_and(isstring, isintent_nothide)
    }, {  # Hidden
        '_check': l_and(isstring, isintent_hide)
    }, {
        'frompyobj': {debugcapi: '    fprintf(stderr,"#vardebugshowvalue#\\n",slen(#varname#),#varname#);'},
        '_check': isstring,
        '_depend': ''
    },
    # Array
    {  # Common
        'decl': ['    #ctype# *#varname# = NULL;',
                 '    npy_intp #varname#_Dims[#rank#] = {#rank*[-1]#};',
                 '    const int #varname#_Rank = #rank#;',
                 '    PyArrayObject *capi_#varname#_as_array = NULL;',
                 '    int capi_#varname#_intent = 0;',
                 {isstringarray: '    int slen(#varname#) = 0;'},
                 ],
        'callfortran': '#varname#,',
        'callfortranappend': {isstringarray: 'slen(#varname#),'},
        'return': {isintent_out: ',capi_#varname#_as_array'},
        'need': 'len..',
        '_check': isarray
    }, {  # intent(overwrite) array
        'decl': '    int capi_overwrite_#varname# = 1;',
        'kwlistxa': '"overwrite_#varname#",',
        'xaformat': 'i',
        'keys_xa': ',&capi_overwrite_#varname#',
        'docsignxa': 'overwrite_#varname#=1,',
        'docsignxashort': 'overwrite_#varname#,',
        'docstropt': 'overwrite_#varname# : input int, optional\\n    Default: 1',
        '_check': l_and(isarray, isintent_overwrite),
    }, {
        'frompyobj': '    capi_#varname#_intent |= (capi_overwrite_#varname#?0:F2PY_INTENT_COPY);',
        '_check': l_and(isarray, isintent_overwrite),
        '_depend': '',
    },
    {  # intent(copy) array
        'decl': '    int capi_overwrite_#varname# = 0;',
        'kwlistxa': '"overwrite_#varname#",',
        'xaformat': 'i',
        'keys_xa': ',&capi_overwrite_#varname#',
        'docsignxa': 'overwrite_#varname#=0,',
        'docsignxashort': 'overwrite_#varname#,',
        'docstropt': 'overwrite_#varname# : input int, optional\\n    Default: 0',
        '_check': l_and(isarray, isintent_copy),
    }, {
        'frompyobj': '    capi_#varname#_intent |= (capi_overwrite_#varname#?0:F2PY_INTENT_COPY);',
        '_check': l_and(isarray, isintent_copy),
        '_depend': '',
    }, {
        'need': [{hasinitvalue: 'forcomb'}, {hasinitvalue: 'CFUNCSMESS'}],
        '_check': isarray,
        '_depend': ''
    }, {  # Not hidden
        'decl': '    PyObject *#varname#_capi = Py_None;',
        'argformat': {isrequired: 'O'},
        'keyformat': {isoptional: 'O'},
        'args_capi': {isrequired: ',&#varname#_capi'},
        'keys_capi': {isoptional: ',&#varname#_capi'},
        '_check': l_and(isarray, isintent_nothide)
    }, {
        'frompyobj': [
            '    #setdims#;',
            '    capi_#varname#_intent |= #intent#;',
            ('    const char capi_errmess[] = "#modulename#.#pyname#:'
             ' failed to create array from the #nth# `#varname#`";'),
            {isintent_hide:
             '    capi_#varname#_as_array = ndarray_from_pyobj('
             '  #atype#,#elsize#,#varname#_Dims,#varname#_Rank,'
             '  capi_#varname#_intent,Py_None,capi_errmess);'},
            {isintent_nothide:
             '    capi_#varname#_as_array = ndarray_from_pyobj('
             '  #atype#,#elsize#,#varname#_Dims,#varname#_Rank,'
             '  capi_#varname#_intent,#varname#_capi,capi_errmess);'},
            """\
    if (capi_#varname#_as_array == NULL) {
        PyObject* capi_err = PyErr_Occurred();
        if (capi_err == NULL) {
            capi_err = #modulename#_error;
            PyErr_SetString(capi_err, capi_errmess);
        }
    } else {
        #varname# = (#ctype# *)(PyArray_DATA(capi_#varname#_as_array));
""",
            {isstringarray:
             '    slen(#varname#) = f2py_itemsize(#varname#);'},
            {hasinitvalue: [
                {isintent_nothide:
                 '    if (#varname#_capi == Py_None) {'},
                {isintent_hide: '    {'},
                {iscomplexarray: '        #ctype# capi_c;'},
                """\
        int *_i,capi_i=0;
        CFUNCSMESS(\"#name#: Initializing #varname#=#init#\\n\");
        struct ForcombCache cache;
        if (initforcomb(&cache, PyArray_DIMS(capi_#varname#_as_array),
                        PyArray_NDIM(capi_#varname#_as_array),1)) {
            while ((_i = nextforcomb(&cache)))
                #varname#[capi_i++] = #init#; /* fortran way */
        } else {
            PyObject *exc, *val, *tb;
            PyErr_Fetch(&exc, &val, &tb);
            PyErr_SetString(exc ? exc : #modulename#_error,
                \"Initialization of #nth# #varname# failed (initforcomb).\");
            npy_PyErr_ChainExceptionsCause(exc, val, tb);
            f2py_success = 0;
        }
    }
    if (f2py_success) {"""]},
                      ],
        'cleanupfrompyobj': [  # note that this list will be reversed
            '    }  '
            '/* if (capi_#varname#_as_array == NULL) ... else of #varname# */',
            {l_not(l_or(isintent_out, isintent_hide)): """\
    if((PyObject *)capi_#varname#_as_array!=#varname#_capi) {
        Py_XDECREF(capi_#varname#_as_array); }"""},
            {l_and(isintent_hide, l_not(isintent_out))
                   : """        Py_XDECREF(capi_#varname#_as_array);"""},
            {hasinitvalue: '    }  /*if (f2py_success) of #varname# init*/'},
        ],
        '_check': isarray,
        '_depend': ''
    },
    # Scalararray
    {  # Common
        '_check': l_and(isarray, l_not(iscomplexarray))
    }, {  # Not hidden
        '_check': l_and(isarray, l_not(iscomplexarray), isintent_nothide)
    },
    # Integer*1 array
    {'need': '#ctype#',
     '_check': isint1array,
     '_depend': ''
     },
    # Integer*-1 array
    {'need': '#ctype#',
     '_check': isunsigned_chararray,
     '_depend': ''
     },
    # Integer*-2 array
    {'need': '#ctype#',
     '_check': isunsigned_shortarray,
     '_depend': ''
     },
    # Integer*-8 array
    {'need': '#ctype#',
     '_check': isunsigned_long_longarray,
     '_depend': ''
     },
    # Complexarray
    {'need': '#ctype#',
     '_check': iscomplexarray,
     '_depend': ''
     },
    # Character
    {
        'need': 'string',
        '_check': ischaracter,
    },
    # Character array
    {
        'need': 'string',
        '_check': ischaracterarray,
    },
    # Stringarray
    {
        'callfortranappend': {isarrayofstrings: 'flen(#varname#),'},
        'need': 'string',
        '_check': isstringarray
    }
]

################# Rules for checking ###############

check_rules = [
    {
        'frompyobj': {debugcapi: '    fprintf(stderr,\"debug-capi:Checking `#check#\'\\n\");'},
        'need': 'len..'
    }, {
        'frompyobj': '    CHECKSCALAR(#check#,\"#check#\",\"#nth# #varname#\",\"#varshowvalue#\",#varname#) {',
        'cleanupfrompyobj': '    } /*CHECKSCALAR(#check#)*/',
        'need': 'CHECKSCALAR',
        '_check': l_and(isscalar, l_not(iscomplex)),
        '_break': ''
    }, {
        'frompyobj': '    CHECKSTRING(#check#,\"#check#\",\"#nth# #varname#\",\"#varshowvalue#\",#varname#) {',
        'cleanupfrompyobj': '    } /*CHECKSTRING(#check#)*/',
        'need': 'CHECKSTRING',
        '_check': isstring,
        '_break': ''
    }, {
        'need': 'CHECKARRAY',
        'frompyobj': '    CHECKARRAY(#check#,\"#check#\",\"#nth# #varname#\") {',
        'cleanupfrompyobj': '    } /*CHECKARRAY(#check#)*/',
        '_check': isarray,
        '_break': ''
    }, {
        'need': 'CHECKGENERIC',
        'frompyobj': '    CHECKGENERIC(#check#,\"#check#\",\"#nth# #varname#\") {',
        'cleanupfrompyobj': '    } /*CHECKGENERIC(#check#)*/',
    }
]

########## Applying the rules. No need to modify what follows #############

#################### Build C/API module #######################


def buildmodule(m, um):
    """
    Return
    """
    outmess(f"    Building module \"{m['name']}\"...\n")
    ret = {}
    mod_rules = defmod_rules[:]
    vrd = capi_maps.modsign2map(m)
    rd = dictappend({'f2py_version': f2py_version}, vrd)
    funcwrappers = []
    funcwrappers2 = []  # F90 codes
    for n in m['interfaced']:
        nb = None
        for bi in m['body']:
            if bi['block'] not in ['interface', 'abstract interface']:
                errmess('buildmodule: Expected interface block. Skipping.\n')
                continue
            for b in bi['body']:
                if b['name'] == n:
                    nb = b
                    break

        if not nb:
            print(
                f'buildmodule: Could not find the body of interfaced routine "{n}". Skipping.\n', file=sys.stderr)
            continue
        nb_list = [nb]
        if 'entry' in nb:
            for k, a in nb['entry'].items():
                nb1 = copy.deepcopy(nb)
                del nb1['entry']
                nb1['name'] = k
                nb1['args'] = a
                nb_list.append(nb1)
        for nb in nb_list:
            # requiresf90wrapper must be called before buildapi as it
            # rewrites assumed shape arrays as automatic arrays.
            isf90 = requiresf90wrapper(nb)
            # options is in scope here
            if options['emptygen']:
                b_path = options['buildpath']
                m_name = vrd['modulename']
                outmess('    Generating possibly empty wrappers"\n')
                Path(f"{b_path}/{vrd['coutput']}").touch()
                if isf90:
                    # f77 + f90 wrappers
                    outmess(f'    Maybe empty "{m_name}-f2pywrappers2.f90"\n')
                    Path(f'{b_path}/{m_name}-f2pywrappers2.f90').touch()
                    outmess(f'    Maybe empty "{m_name}-f2pywrappers.f"\n')
                    Path(f'{b_path}/{m_name}-f2pywrappers.f').touch()
                else:
                    # only f77 wrappers
                    outmess(f'    Maybe empty "{m_name}-f2pywrappers.f"\n')
                    Path(f'{b_path}/{m_name}-f2pywrappers.f').touch()
            api, wrap = buildapi(nb)
            if wrap:
                if isf90:
                    funcwrappers2.append(wrap)
                else:
                    funcwrappers.append(wrap)
            ar = applyrules(api, vrd)
            rd = dictappend(rd, ar)

    # Construct COMMON block support
    cr, wrap = common_rules.buildhooks(m)
    if wrap:
        funcwrappers.append(wrap)
    ar = applyrules(cr, vrd)
    rd = dictappend(rd, ar)

    # Construct F90 module support
    mr, wrap = f90mod_rules.buildhooks(m)
    if wrap:
        funcwrappers2.append(wrap)
    ar = applyrules(mr, vrd)
    rd = dictappend(rd, ar)

    for u in um:
        ar = use_rules.buildusevars(u, m['use'][u['name']])
        rd = dictappend(rd, ar)

    needs = cfuncs.get_needs()
    # Add mapped definitions
    needs['typedefs'] += [cvar for cvar in capi_maps.f2cmap_mapped  #
                          if cvar in typedef_need_dict.values()]
    code = {}
    for n in needs.keys():
        code[n] = []
        for k in needs[n]:
            c = ''
            if k in cfuncs.includes0:
                c = cfuncs.includes0[k]
            elif k in cfuncs.includes:
                c = cfuncs.includes[k]
            elif k in cfuncs.userincludes:
                c = cfuncs.userincludes[k]
            elif k in cfuncs.typedefs:
                c = cfuncs.typedefs[k]
            elif k in cfuncs.typedefs_generated:
                c = cfuncs.typedefs_generated[k]
            elif k in cfuncs.cppmacros:
                c = cfuncs.cppmacros[k]
            elif k in cfuncs.cfuncs:
                c = cfuncs.cfuncs[k]
            elif k in cfuncs.callbacks:
                c = cfuncs.callbacks[k]
            elif k in cfuncs.f90modhooks:
                c = cfuncs.f90modhooks[k]
            elif k in cfuncs.commonhooks:
                c = cfuncs.commonhooks[k]
            else:
                errmess(f'buildmodule: unknown need {repr(k)}.\n')
                continue
            code[n].append(c)
    mod_rules.append(code)
    for r in mod_rules:
        if ('_check' in r and r['_check'](m)) or ('_check' not in r):
            ar = applyrules(r, vrd, m)
            rd = dictappend(rd, ar)
    ar = applyrules(module_rules, rd)

    fn = os.path.join(options['buildpath'], vrd['coutput'])
    ret['csrc'] = fn
    with open(fn, 'w') as f:
        f.write(ar['modulebody'].replace('\t', 2 * ' '))
    outmess(f"    Wrote C/API module \"{m['name']}\" to file \"{fn}\"\n")

    if options['dorestdoc']:
        fn = os.path.join(
            options['buildpath'], vrd['modulename'] + 'module.rest')
        with open(fn, 'w') as f:
            f.write('.. -*- rest -*-\n')
            f.write('\n'.join(ar['restdoc']))
        outmess('    ReST Documentation is saved to file "%s/%smodule.rest"\n' %
                (options['buildpath'], vrd['modulename']))
    if options['dolatexdoc']:
        fn = os.path.join(
            options['buildpath'], vrd['modulename'] + 'module.tex')
        ret['ltx'] = fn
        with open(fn, 'w') as f:
            f.write(
                f'% This file is auto-generated with f2py (version:{f2py_version})\n')
            if 'shortlatex' not in options:
                f.write(
                    '\\documentclass{article}\n\\usepackage{a4wide}\n\\begin{document}\n\\tableofcontents\n\n')
                f.write('\n'.join(ar['latexdoc']))
            if 'shortlatex' not in options:
                f.write('\\end{document}')
        outmess('    Documentation is saved to file "%s/%smodule.tex"\n' %
                (options['buildpath'], vrd['modulename']))
    if funcwrappers:
        wn = os.path.join(options['buildpath'], vrd['f2py_wrapper_output'])
        ret['fsrc'] = wn
        with open(wn, 'w') as f:
            f.write('C     -*- fortran -*-\n')
            f.write(
                f'C     This file is autogenerated with f2py (version:{f2py_version})\n')
            f.write(
                'C     It contains Fortran 77 wrappers to fortran functions.\n')
            lines = []
            for l in ('\n\n'.join(funcwrappers) + '\n').split('\n'):
                if 0 <= l.find('!') < 66:
                    # don't split comment lines
                    lines.append(l + '\n')
                elif l and l[0] == ' ':
                    while len(l) >= 66:
                        lines.append(l[:66] + '\n     &')
                        l = l[66:]
                    lines.append(l + '\n')
                else:
                    lines.append(l + '\n')
            lines = ''.join(lines).replace('\n     &\n', '\n')
            f.write(lines)
        outmess(f'    Fortran 77 wrappers are saved to "{wn}\"\n')
    if funcwrappers2:
        wn = os.path.join(
            options['buildpath'], f"{vrd['modulename']}-f2pywrappers2.f90")
        ret['fsrc'] = wn
        with open(wn, 'w') as f:
            f.write('!     -*- f90 -*-\n')
            f.write(
                f'!     This file is autogenerated with f2py (version:{f2py_version})\n')
            f.write(
                '!     It contains Fortran 90 wrappers to fortran functions.\n')
            lines = []
            for l in ('\n\n'.join(funcwrappers2) + '\n').split('\n'):
                if 0 <= l.find('!') < 72:
                    # don't split comment lines
                    lines.append(l + '\n')
                elif len(l) > 72 and l[0] == ' ':
                    lines.append(l[:72] + '&\n     &')
                    l = l[72:]
                    while len(l) > 66:
                        lines.append(l[:66] + '&\n     &')
                        l = l[66:]
                    lines.append(l + '\n')
                else:
                    lines.append(l + '\n')
            lines = ''.join(lines).replace('\n     &\n', '\n')
            f.write(lines)
        outmess(f'    Fortran 90 wrappers are saved to "{wn}\"\n')
    return ret

################## Build C/API function #############


stnd = {1: 'st', 2: 'nd', 3: 'rd', 4: 'th', 5: 'th',
        6: 'th', 7: 'th', 8: 'th', 9: 'th', 0: 'th'}


def buildapi(rout):
    rout, wrap = func2subr.assubr(rout)
    args, depargs = getargs2(rout)
    capi_maps.depargs = depargs
    var = rout['vars']

    if ismoduleroutine(rout):
        outmess('            Constructing wrapper function "%s.%s"...\n' %
                (rout['modulename'], rout['name']))
    else:
        outmess(f"        Constructing wrapper function \"{rout['name']}\"...\n")
    # Routine
    vrd = capi_maps.routsign2map(rout)
    rd = dictappend({}, vrd)
    for r in rout_rules:
        if ('_check' in r and r['_check'](rout)) or ('_check' not in r):
            ar = applyrules(r, vrd, rout)
            rd = dictappend(rd, ar)

    # Args
    nth, nthk = 0, 0
    savevrd = {}
    for a in args:
        vrd = capi_maps.sign2map(a, var[a])
        if isintent_aux(var[a]):
            _rules = aux_rules
        else:
            _rules = arg_rules
            if not isintent_hide(var[a]):
                if not isoptional(var[a]):
                    nth = nth + 1
                    vrd['nth'] = repr(nth) + stnd[nth % 10] + ' argument'
                else:
                    nthk = nthk + 1
                    vrd['nth'] = repr(nthk) + stnd[nthk % 10] + ' keyword'
            else:
                vrd['nth'] = 'hidden'
        savevrd[a] = vrd
        for r in _rules:
            if '_depend' in r:
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
    for a in depargs:
        if isintent_aux(var[a]):
            _rules = aux_rules
        else:
            _rules = arg_rules
        vrd = savevrd[a]
        for r in _rules:
            if '_depend' not in r:
                continue
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                if '_break' in r:
                    break
        if 'check' in var[a]:
            for c in var[a]['check']:
                vrd['check'] = c
                ar = applyrules(check_rules, vrd, var[a])
                rd = dictappend(rd, ar)
    if isinstance(rd['cleanupfrompyobj'], list):
        rd['cleanupfrompyobj'].reverse()
    if isinstance(rd['closepyobjfrom'], list):
        rd['closepyobjfrom'].reverse()
    rd['docsignature'] = stripcomma(replace('#docsign##docsignopt##docsignxa#',
                                            {'docsign': rd['docsign'],
                                             'docsignopt': rd['docsignopt'],
                                             'docsignxa': rd['docsignxa']}))
    optargs = stripcomma(replace('#docsignopt##docsignxa#',
                                 {'docsignxa': rd['docsignxashort'],
                                  'docsignopt': rd['docsignoptshort']}
                                 ))
    if optargs == '':
        rd['docsignatureshort'] = stripcomma(
            replace('#docsign#', {'docsign': rd['docsign']}))
    else:
        rd['docsignatureshort'] = replace('#docsign#[#docsignopt#]',
                                          {'docsign': rd['docsign'],
                                           'docsignopt': optargs,
                                           })
    rd['latexdocsignatureshort'] = rd['docsignatureshort'].replace('_', '\\_')
    rd['latexdocsignatureshort'] = rd[
        'latexdocsignatureshort'].replace(',', ', ')
    cfs = stripcomma(replace('#callfortran##callfortranappend#', {
                     'callfortran': rd['callfortran'], 'callfortranappend': rd['callfortranappend']}))
    if len(rd['callfortranappend']) > 1:
        rd['callcompaqfortran'] = stripcomma(replace('#callfortran# 0,#callfortranappend#', {
                                             'callfortran': rd['callfortran'], 'callfortranappend': rd['callfortranappend']}))
    else:
        rd['callcompaqfortran'] = cfs
    rd['callfortran'] = cfs
    if isinstance(rd['docreturn'], list):
        rd['docreturn'] = stripcomma(
            replace('#docreturn#', {'docreturn': rd['docreturn']})) + ' = '
    rd['docstrsigns'] = []
    rd['latexdocstrsigns'] = []
    for k in ['docstrreq', 'docstropt', 'docstrout', 'docstrcbs']:
        if k in rd and isinstance(rd[k], list):
            rd['docstrsigns'] = rd['docstrsigns'] + rd[k]
        k = 'latex' + k
        if k in rd and isinstance(rd[k], list):
            rd['latexdocstrsigns'] = rd['latexdocstrsigns'] + rd[k][0:1] +\
                ['\\begin{description}'] + rd[k][1:] +\
                ['\\end{description}']

    ar = applyrules(routine_rules, rd)
    if ismoduleroutine(rout):
        outmess(f"              {ar['docshort']}\n")
    else:
        outmess(f"          {ar['docshort']}\n")
    return ar, wrap


#################### EOF rules.py #######################

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\collections.py ===
# orm/collections.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Support for collections of mapped entities.

The collections package supplies the machinery used to inform the ORM of
collection membership changes.  An instrumentation via decoration approach is
used, allowing arbitrary types (including built-ins) to be used as entity
collections without requiring inheritance from a base class.

Instrumentation decoration relays membership change events to the
:class:`.CollectionAttributeImpl` that is currently managing the collection.
The decorators observe function call arguments and return values, tracking
entities entering or leaving the collection.  Two decorator approaches are
provided.  One is a bundle of generic decorators that map function arguments
and return values to events::

  from sqlalchemy.orm.collections import collection


  class MyClass:
      # ...

      @collection.adds(1)
      def store(self, item):
          self.data.append(item)

      @collection.removes_return()
      def pop(self):
          return self.data.pop()

The second approach is a bundle of targeted decorators that wrap appropriate
append and remove notifiers around the mutation methods present in the
standard Python ``list``, ``set`` and ``dict`` interfaces.  These could be
specified in terms of generic decorator recipes, but are instead hand-tooled
for increased efficiency.  The targeted decorators occasionally implement
adapter-like behavior, such as mapping bulk-set methods (``extend``,
``update``, ``__setslice__``, etc.) into the series of atomic mutation events
that the ORM requires.

The targeted decorators are used internally for automatic instrumentation of
entity collection classes.  Every collection class goes through a
transformation process roughly like so:

1. If the class is a built-in, substitute a trivial sub-class
2. Is this class already instrumented?
3. Add in generic decorators
4. Sniff out the collection interface through duck-typing
5. Add targeted decoration to any undecorated interface method

This process modifies the class at runtime, decorating methods and adding some
bookkeeping properties.  This isn't possible (or desirable) for built-in
classes like ``list``, so trivial sub-classes are substituted to hold
decoration::

  class InstrumentedList(list):
      pass

Collection classes can be specified in ``relationship(collection_class=)`` as
types or a function that returns an instance.  Collection classes are
inspected and instrumented during the mapper compilation phase.  The
collection_class callable will be executed once to produce a specimen
instance, and the type of that specimen will be instrumented.  Functions that
return built-in types like ``lists`` will be adapted to produce instrumented
instances.

When extending a known type like ``list``, additional decorations are not
generally not needed.  Odds are, the extension method will delegate to a
method that's already instrumented.  For example::

  class QueueIsh(list):
      def push(self, item):
          self.append(item)

      def shift(self):
          return self.pop(0)

There's no need to decorate these methods.  ``append`` and ``pop`` are already
instrumented as part of the ``list`` interface.  Decorating them would fire
duplicate events, which should be avoided.

The targeted decoration tries not to rely on other methods in the underlying
collection class, but some are unavoidable.  Many depend on 'read' methods
being present to properly instrument a 'write', for example, ``__setitem__``
needs ``__getitem__``.  "Bulk" methods like ``update`` and ``extend`` may also
reimplemented in terms of atomic appends and removes, so the ``extend``
decoration will actually perform many ``append`` operations and not call the
underlying method at all.

Tight control over bulk operation and the firing of events is also possible by
implementing the instrumentation internally in your methods.  The basic
instrumentation package works under the general assumption that collection
mutation will not raise unusual exceptions.  If you want to closely
orchestrate append and remove events with exception management, internal
instrumentation may be the answer.  Within your method,
``collection_adapter(self)`` will retrieve an object that you can use for
explicit control over triggering append and remove events.

The owning object and :class:`.CollectionAttributeImpl` are also reachable
through the adapter, allowing for some very sophisticated behavior.

"""
from __future__ import annotations

import operator
import threading
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref

from .base import NO_KEY
from .. import exc as sa_exc
from .. import util
from ..sql.base import NO_ARG
from ..util.compat import inspect_getfullargspec
from ..util.typing import Protocol

if typing.TYPE_CHECKING:
    from .attributes import AttributeEventToken
    from .attributes import CollectionAttributeImpl
    from .mapped_collection import attribute_keyed_dict
    from .mapped_collection import column_keyed_dict
    from .mapped_collection import keyfunc_mapping
    from .mapped_collection import KeyFuncDict  # noqa: F401
    from .state import InstanceState


__all__ = [
    "collection",
    "collection_adapter",
    "keyfunc_mapping",
    "column_keyed_dict",
    "attribute_keyed_dict",
    "KeyFuncDict",
    # old names in < 2.0
    "mapped_collection",
    "column_mapped_collection",
    "attribute_mapped_collection",
    "MappedCollection",
]

__instrumentation_mutex = threading.Lock()


_CollectionFactoryType = Callable[[], "_AdaptedCollectionProtocol"]

_T = TypeVar("_T", bound=Any)
_KT = TypeVar("_KT", bound=Any)
_VT = TypeVar("_VT", bound=Any)
_COL = TypeVar("_COL", bound="Collection[Any]")
_FN = TypeVar("_FN", bound="Callable[..., Any]")


class _CollectionConverterProtocol(Protocol):
    def __call__(self, collection: _COL) -> _COL: ...


class _AdaptedCollectionProtocol(Protocol):
    _sa_adapter: CollectionAdapter
    _sa_appender: Callable[..., Any]
    _sa_remover: Callable[..., Any]
    _sa_iterator: Callable[..., Iterable[Any]]
    _sa_converter: _CollectionConverterProtocol


class collection:
    """Decorators for entity collection classes.

    The decorators fall into two groups: annotations and interception recipes.

    The annotating decorators (appender, remover, iterator, converter,
    internally_instrumented) indicate the method's purpose and take no
    arguments.  They are not written with parens::

        @collection.appender
        def append(self, append): ...

    The recipe decorators all require parens, even those that take no
    arguments::

        @collection.adds("entity")
        def insert(self, position, entity): ...


        @collection.removes_return()
        def popitem(self): ...

    """

    # Bundled as a class solely for ease of use: packaging, doc strings,
    # importability.

    @staticmethod
    def appender(fn):
        """Tag the method as the collection appender.

        The appender method is called with one positional argument: the value
        to append. The method will be automatically decorated with 'adds(1)'
        if not already decorated::

            @collection.appender
            def add(self, append): ...


            # or, equivalently
            @collection.appender
            @collection.adds(1)
            def add(self, append): ...


            # for mapping type, an 'append' may kick out a previous value
            # that occupies that slot.  consider d['a'] = 'foo'- any previous
            # value in d['a'] is discarded.
            @collection.appender
            @collection.replaces(1)
            def add(self, entity):
                key = some_key_func(entity)
                previous = None
                if key in self:
                    previous = self[key]
                self[key] = entity
                return previous

        If the value to append is not allowed in the collection, you may
        raise an exception.  Something to remember is that the appender
        will be called for each object mapped by a database query.  If the
        database contains rows that violate your collection semantics, you
        will need to get creative to fix the problem, as access via the
        collection will not work.

        If the appender method is internally instrumented, you must also
        receive the keyword argument '_sa_initiator' and ensure its
        promulgation to collection events.

        """
        fn._sa_instrument_role = "appender"
        return fn

    @staticmethod
    def remover(fn):
        """Tag the method as the collection remover.

        The remover method is called with one positional argument: the value
        to remove. The method will be automatically decorated with
        :meth:`removes_return` if not already decorated::

            @collection.remover
            def zap(self, entity): ...


            # or, equivalently
            @collection.remover
            @collection.removes_return()
            def zap(self): ...

        If the value to remove is not present in the collection, you may
        raise an exception or return None to ignore the error.

        If the remove method is internally instrumented, you must also
        receive the keyword argument '_sa_initiator' and ensure its
        promulgation to collection events.

        """
        fn._sa_instrument_role = "remover"
        return fn

    @staticmethod
    def iterator(fn):
        """Tag the method as the collection remover.

        The iterator method is called with no arguments.  It is expected to
        return an iterator over all collection members::

            @collection.iterator
            def __iter__(self): ...

        """
        fn._sa_instrument_role = "iterator"
        return fn

    @staticmethod
    def internally_instrumented(fn):
        """Tag the method as instrumented.

        This tag will prevent any decoration from being applied to the
        method. Use this if you are orchestrating your own calls to
        :func:`.collection_adapter` in one of the basic SQLAlchemy
        interface methods, or to prevent an automatic ABC method
        decoration from wrapping your implementation::

            # normally an 'extend' method on a list-like class would be
            # automatically intercepted and re-implemented in terms of
            # SQLAlchemy events and append().  your implementation will
            # never be called, unless:
            @collection.internally_instrumented
            def extend(self, items): ...

        """
        fn._sa_instrumented = True
        return fn

    @staticmethod
    @util.deprecated(
        "1.3",
        "The :meth:`.collection.converter` handler is deprecated and will "
        "be removed in a future release.  Please refer to the "
        ":class:`.AttributeEvents.bulk_replace` listener interface in "
        "conjunction with the :func:`.event.listen` function.",
    )
    def converter(fn):
        """Tag the method as the collection converter.

        This optional method will be called when a collection is being
        replaced entirely, as in::

            myobj.acollection = [newvalue1, newvalue2]

        The converter method will receive the object being assigned and should
        return an iterable of values suitable for use by the ``appender``
        method.  A converter must not assign values or mutate the collection,
        its sole job is to adapt the value the user provides into an iterable
        of values for the ORM's use.

        The default converter implementation will use duck-typing to do the
        conversion.  A dict-like collection will be convert into an iterable
        of dictionary values, and other types will simply be iterated::

            @collection.converter
            def convert(self, other): ...

        If the duck-typing of the object does not match the type of this
        collection, a TypeError is raised.

        Supply an implementation of this method if you want to expand the
        range of possible types that can be assigned in bulk or perform
        validation on the values about to be assigned.

        """
        fn._sa_instrument_role = "converter"
        return fn

    @staticmethod
    def adds(arg):
        """Mark the method as adding an entity to the collection.

        Adds "add to collection" handling to the method.  The decorator
        argument indicates which method argument holds the SQLAlchemy-relevant
        value.  Arguments can be specified positionally (i.e. integer) or by
        name::

            @collection.adds(1)
            def push(self, item): ...


            @collection.adds("entity")
            def do_stuff(self, thing, entity=None): ...

        """

        def decorator(fn):
            fn._sa_instrument_before = ("fire_append_event", arg)
            return fn

        return decorator

    @staticmethod
    def replaces(arg):
        """Mark the method as replacing an entity in the collection.

        Adds "add to collection" and "remove from collection" handling to
        the method.  The decorator argument indicates which method argument
        holds the SQLAlchemy-relevant value to be added, and return value, if
        any will be considered the value to remove.

        Arguments can be specified positionally (i.e. integer) or by name::

            @collection.replaces(2)
            def __setitem__(self, index, item): ...

        """

        def decorator(fn):
            fn._sa_instrument_before = ("fire_append_event", arg)
            fn._sa_instrument_after = "fire_remove_event"
            return fn

        return decorator

    @staticmethod
    def removes(arg):
        """Mark the method as removing an entity in the collection.

        Adds "remove from collection" handling to the method.  The decorator
        argument indicates which method argument holds the SQLAlchemy-relevant
        value to be removed. Arguments can be specified positionally (i.e.
        integer) or by name::

            @collection.removes(1)
            def zap(self, item): ...

        For methods where the value to remove is not known at call-time, use
        collection.removes_return.

        """

        def decorator(fn):
            fn._sa_instrument_before = ("fire_remove_event", arg)
            return fn

        return decorator

    @staticmethod
    def removes_return():
        """Mark the method as removing an entity in the collection.

        Adds "remove from collection" handling to the method.  The return
        value of the method, if any, is considered the value to remove.  The
        method arguments are not inspected::

            @collection.removes_return()
            def pop(self): ...

        For methods where the value to remove is known at call-time, use
        collection.remove.

        """

        def decorator(fn):
            fn._sa_instrument_after = "fire_remove_event"
            return fn

        return decorator


if TYPE_CHECKING:

    def collection_adapter(collection: Collection[Any]) -> CollectionAdapter:
        """Fetch the :class:`.CollectionAdapter` for a collection."""

else:
    collection_adapter = operator.attrgetter("_sa_adapter")


class CollectionAdapter:
    """Bridges between the ORM and arbitrary Python collections.

    Proxies base-level collection operations (append, remove, iterate)
    to the underlying Python collection, and emits add/remove events for
    entities entering or leaving the collection.

    The ORM uses :class:`.CollectionAdapter` exclusively for interaction with
    entity collections.


    """

    __slots__ = (
        "attr",
        "_key",
        "_data",
        "owner_state",
        "_converter",
        "invalidated",
        "empty",
    )

    attr: CollectionAttributeImpl
    _key: str

    # this is actually a weakref; see note in constructor
    _data: Callable[..., _AdaptedCollectionProtocol]

    owner_state: InstanceState[Any]
    _converter: _CollectionConverterProtocol
    invalidated: bool
    empty: bool

    def __init__(
        self,
        attr: CollectionAttributeImpl,
        owner_state: InstanceState[Any],
        data: _AdaptedCollectionProtocol,
    ):
        self.attr = attr
        self._key = attr.key

        # this weakref stays referenced throughout the lifespan of
        # CollectionAdapter.  so while the weakref can return None, this
        # is realistically only during garbage collection of this object, so
        # we type this as a callable that returns _AdaptedCollectionProtocol
        # in all cases.
        self._data = weakref.ref(data)  # type: ignore

        self.owner_state = owner_state
        data._sa_adapter = self
        self._converter = data._sa_converter
        self.invalidated = False
        self.empty = False

    def _warn_invalidated(self) -> None:
        util.warn("This collection has been invalidated.")

    @property
    def data(self) -> _AdaptedCollectionProtocol:
        "The entity collection being adapted."
        return self._data()

    @property
    def _referenced_by_owner(self) -> bool:
        """return True if the owner state still refers to this collection.

        This will return False within a bulk replace operation,
        where this collection is the one being replaced.

        """
        return self.owner_state.dict[self._key] is self._data()

    def bulk_appender(self):
        return self._data()._sa_appender

    def append_with_event(
        self, item: Any, initiator: Optional[AttributeEventToken] = None
    ) -> None:
        """Add an entity to the collection, firing mutation events."""

        self._data()._sa_appender(item, _sa_initiator=initiator)

    def _set_empty(self, user_data):
        assert (
            not self.empty
        ), "This collection adapter is already in the 'empty' state"
        self.empty = True
        self.owner_state._empty_collections[self._key] = user_data

    def _reset_empty(self) -> None:
        assert (
            self.empty
        ), "This collection adapter is not in the 'empty' state"
        self.empty = False
        self.owner_state.dict[self._key] = (
            self.owner_state._empty_collections.pop(self._key)
        )

    def _refuse_empty(self) -> NoReturn:
        raise sa_exc.InvalidRequestError(
            "This is a special 'empty' collection which cannot accommodate "
            "internal mutation operations"
        )

    def append_without_event(self, item: Any) -> None:
        """Add or restore an entity to the collection, firing no events."""

        if self.empty:
            self._refuse_empty()
        self._data()._sa_appender(item, _sa_initiator=False)

    def append_multiple_without_event(self, items: Iterable[Any]) -> None:
        """Add or restore an entity to the collection, firing no events."""
        if self.empty:
            self._refuse_empty()
        appender = self._data()._sa_appender
        for item in items:
            appender(item, _sa_initiator=False)

    def bulk_remover(self):
        return self._data()._sa_remover

    def remove_with_event(
        self, item: Any, initiator: Optional[AttributeEventToken] = None
    ) -> None:
        """Remove an entity from the collection, firing mutation events."""
        self._data()._sa_remover(item, _sa_initiator=initiator)

    def remove_without_event(self, item: Any) -> None:
        """Remove an entity from the collection, firing no events."""
        if self.empty:
            self._refuse_empty()
        self._data()._sa_remover(item, _sa_initiator=False)

    def clear_with_event(
        self, initiator: Optional[AttributeEventToken] = None
    ) -> None:
        """Empty the collection, firing a mutation event for each entity."""

        if self.empty:
            self._refuse_empty()
        remover = self._data()._sa_remover
        for item in list(self):
            remover(item, _sa_initiator=initiator)

    def clear_without_event(self) -> None:
        """Empty the collection, firing no events."""

        if self.empty:
            self._refuse_empty()
        remover = self._data()._sa_remover
        for item in list(self):
            remover(item, _sa_initiator=False)

    def __iter__(self):
        """Iterate over entities in the collection."""

        return iter(self._data()._sa_iterator())

    def __len__(self):
        """Count entities in the collection."""
        return len(list(self._data()._sa_iterator()))

    def __bool__(self):
        return True

    def _fire_append_wo_mutation_event_bulk(
        self, items, initiator=None, key=NO_KEY
    ):
        if not items:
            return

        if initiator is not False:
            if self.invalidated:
                self._warn_invalidated()

            if self.empty:
                self._reset_empty()

            for item in items:
                self.attr.fire_append_wo_mutation_event(
                    self.owner_state,
                    self.owner_state.dict,
                    item,
                    initiator,
                    key,
                )

    def fire_append_wo_mutation_event(self, item, initiator=None, key=NO_KEY):
        """Notify that a entity is entering the collection but is already
        present.


        Initiator is a token owned by the InstrumentedAttribute that
        initiated the membership mutation, and should be left as None
        unless you are passing along an initiator value from a chained
        operation.

        .. versionadded:: 1.4.15

        """
        if initiator is not False:
            if self.invalidated:
                self._warn_invalidated()

            if self.empty:
                self._reset_empty()

            return self.attr.fire_append_wo_mutation_event(
                self.owner_state, self.owner_state.dict, item, initiator, key
            )
        else:
            return item

    def fire_append_event(self, item, initiator=None, key=NO_KEY):
        """Notify that a entity has entered the collection.

        Initiator is a token owned by the InstrumentedAttribute that
        initiated the membership mutation, and should be left as None
        unless you are passing along an initiator value from a chained
        operation.

        """
        if initiator is not False:
            if self.invalidated:
                self._warn_invalidated()

            if self.empty:
                self._reset_empty()

            return self.attr.fire_append_event(
                self.owner_state, self.owner_state.dict, item, initiator, key
            )
        else:
            return item

    def _fire_remove_event_bulk(self, items, initiator=None, key=NO_KEY):
        if not items:
            return

        if initiator is not False:
            if self.invalidated:
                self._warn_invalidated()

            if self.empty:
                self._reset_empty()

            for item in items:
                self.attr.fire_remove_event(
                    self.owner_state,
                    self.owner_state.dict,
                    item,
                    initiator,
                    key,
                )

    def fire_remove_event(self, item, initiator=None, key=NO_KEY):
        """Notify that a entity has been removed from the collection.

        Initiator is the InstrumentedAttribute that initiated the membership
        mutation, and should be left as None unless you are passing along
        an initiator value from a chained operation.

        """
        if initiator is not False:
            if self.invalidated:
                self._warn_invalidated()

            if self.empty:
                self._reset_empty()

            self.attr.fire_remove_event(
                self.owner_state, self.owner_state.dict, item, initiator, key
            )

    def fire_pre_remove_event(self, initiator=None, key=NO_KEY):
        """Notify that an entity is about to be removed from the collection.

        Only called if the entity cannot be removed after calling
        fire_remove_event().

        """
        if self.invalidated:
            self._warn_invalidated()
        self.attr.fire_pre_remove_event(
            self.owner_state,
            self.owner_state.dict,
            initiator=initiator,
            key=key,
        )

    def __getstate__(self):
        return {
            "key": self._key,
            "owner_state": self.owner_state,
            "owner_cls": self.owner_state.class_,
            "data": self.data,
            "invalidated": self.invalidated,
            "empty": self.empty,
        }

    def __setstate__(self, d):
        self._key = d["key"]
        self.owner_state = d["owner_state"]

        # see note in constructor regarding this type: ignore
        self._data = weakref.ref(d["data"])  # type: ignore

        self._converter = d["data"]._sa_converter
        d["data"]._sa_adapter = self
        self.invalidated = d["invalidated"]
        self.attr = getattr(d["owner_cls"], self._key).impl
        self.empty = d.get("empty", False)


def bulk_replace(values, existing_adapter, new_adapter, initiator=None):
    """Load a new collection, firing events based on prior like membership.

    Appends instances in ``values`` onto the ``new_adapter``. Events will be
    fired for any instance not present in the ``existing_adapter``.  Any
    instances in ``existing_adapter`` not present in ``values`` will have
    remove events fired upon them.

    :param values: An iterable of collection member instances

    :param existing_adapter: A :class:`.CollectionAdapter` of
     instances to be replaced

    :param new_adapter: An empty :class:`.CollectionAdapter`
     to load with ``values``


    """

    assert isinstance(values, list)

    idset = util.IdentitySet
    existing_idset = idset(existing_adapter or ())
    constants = existing_idset.intersection(values or ())
    additions = idset(values or ()).difference(constants)
    removals = existing_idset.difference(constants)

    appender = new_adapter.bulk_appender()

    for member in values or ():
        if member in additions:
            appender(member, _sa_initiator=initiator)
        elif member in constants:
            appender(member, _sa_initiator=False)

    if existing_adapter:
        existing_adapter._fire_append_wo_mutation_event_bulk(
            constants, initiator=initiator
        )
        existing_adapter._fire_remove_event_bulk(removals, initiator=initiator)


def prepare_instrumentation(
    factory: Union[Type[Collection[Any]], _CollectionFactoryType],
) -> _CollectionFactoryType:
    """Prepare a callable for future use as a collection class factory.

    Given a collection class factory (either a type or no-arg callable),
    return another factory that will produce compatible instances when
    called.

    This function is responsible for converting collection_class=list
    into the run-time behavior of collection_class=InstrumentedList.

    """

    impl_factory: _CollectionFactoryType

    # Convert a builtin to 'Instrumented*'
    if factory in __canned_instrumentation:
        impl_factory = __canned_instrumentation[factory]
    else:
        impl_factory = cast(_CollectionFactoryType, factory)

    cls: Union[_CollectionFactoryType, Type[Collection[Any]]]

    # Create a specimen
    cls = type(impl_factory())

    # Did factory callable return a builtin?
    if cls in __canned_instrumentation:
        # if so, just convert.
        # in previous major releases, this codepath wasn't working and was
        # not covered by tests.   prior to that it supplied a "wrapper"
        # function that would return the class, though the rationale for this
        # case is not known
        impl_factory = __canned_instrumentation[cls]
        cls = type(impl_factory())

    # Instrument the class if needed.
    if __instrumentation_mutex.acquire():
        try:
            if getattr(cls, "_sa_instrumented", None) != id(cls):
                _instrument_class(cls)
        finally:
            __instrumentation_mutex.release()

    return impl_factory


def _instrument_class(cls):
    """Modify methods in a class and install instrumentation."""

    # In the normal call flow, a request for any of the 3 basic collection
    # types is transformed into one of our trivial subclasses
    # (e.g. InstrumentedList).  Catch anything else that sneaks in here...
    if cls.__module__ == "__builtin__":
        raise sa_exc.ArgumentError(
            "Can not instrument a built-in type. Use a "
            "subclass, even a trivial one."
        )

    roles, methods = _locate_roles_and_methods(cls)

    _setup_canned_roles(cls, roles, methods)

    _assert_required_roles(cls, roles, methods)

    _set_collection_attributes(cls, roles, methods)


def _locate_roles_and_methods(cls):
    """search for _sa_instrument_role-decorated methods in
    method resolution order, assign to roles.

    """

    roles: Dict[str, str] = {}
    methods: Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]] = {}

    for supercls in cls.__mro__:
        for name, method in vars(supercls).items():
            if not callable(method):
                continue

            # note role declarations
            if hasattr(method, "_sa_instrument_role"):
                role = method._sa_instrument_role
                assert role in (
                    "appender",
                    "remover",
                    "iterator",
                    "converter",
                )
                roles.setdefault(role, name)

            # transfer instrumentation requests from decorated function
            # to the combined queue
            before: Optional[Tuple[str, int]] = None
            after: Optional[str] = None

            if hasattr(method, "_sa_instrument_before"):
                op, argument = method._sa_instrument_before
                assert op in ("fire_append_event", "fire_remove_event")
                before = op, argument
            if hasattr(method, "_sa_instrument_after"):
                op = method._sa_instrument_after
                assert op in ("fire_append_event", "fire_remove_event")
                after = op
            if before:
                methods[name] = before + (after,)
            elif after:
                methods[name] = None, None, after
    return roles, methods


def _setup_canned_roles(cls, roles, methods):
    """see if this class has "canned" roles based on a known
    collection type (dict, set, list).  Apply those roles
    as needed to the "roles" dictionary, and also
    prepare "decorator" methods

    """
    collection_type = util.duck_type_collection(cls)
    if collection_type in __interfaces:
        assert collection_type is not None
        canned_roles, decorators = __interfaces[collection_type]
        for role, name in canned_roles.items():
            roles.setdefault(role, name)

        # apply ABC auto-decoration to methods that need it
        for method, decorator in decorators.items():
            fn = getattr(cls, method, None)
            if (
                fn
                and method not in methods
                and not hasattr(fn, "_sa_instrumented")
            ):
                setattr(cls, method, decorator(fn))


def _assert_required_roles(cls, roles, methods):
    """ensure all roles are present, and apply implicit instrumentation if
    needed

    """
    if "appender" not in roles or not hasattr(cls, roles["appender"]):
        raise sa_exc.ArgumentError(
            "Type %s must elect an appender method to be "
            "a collection class" % cls.__name__
        )
    elif roles["appender"] not in methods and not hasattr(
        getattr(cls, roles["appender"]), "_sa_instrumented"
    ):
        methods[roles["appender"]] = ("fire_append_event", 1, None)

    if "remover" not in roles or not hasattr(cls, roles["remover"]):
        raise sa_exc.ArgumentError(
            "Type %s must elect a remover method to be "
            "a collection class" % cls.__name__
        )
    elif roles["remover"] not in methods and not hasattr(
        getattr(cls, roles["remover"]), "_sa_instrumented"
    ):
        methods[roles["remover"]] = ("fire_remove_event", 1, None)

    if "iterator" not in roles or not hasattr(cls, roles["iterator"]):
        raise sa_exc.ArgumentError(
            "Type %s must elect an iterator method to be "
            "a collection class" % cls.__name__
        )


def _set_collection_attributes(cls, roles, methods):
    """apply ad-hoc instrumentation from decorators, class-level defaults
    and implicit role declarations

    """
    for method_name, (before, argument, after) in methods.items():
        setattr(
            cls,
            method_name,
            _instrument_membership_mutator(
                getattr(cls, method_name), before, argument, after
            ),
        )
    # intern the role map
    for role, method_name in roles.items():
        setattr(cls, "_sa_%s" % role, getattr(cls, method_name))

    cls._sa_adapter = None

    if not hasattr(cls, "_sa_converter"):
        cls._sa_converter = None
    cls._sa_instrumented = id(cls)


def _instrument_membership_mutator(method, before, argument, after):
    """Route method args and/or return value through the collection
    adapter."""
    # This isn't smart enough to handle @adds(1) for 'def fn(self, (a, b))'
    if before:
        fn_args = list(
            util.flatten_iterator(inspect_getfullargspec(method)[0])
        )
        if isinstance(argument, int):
            pos_arg = argument
            named_arg = len(fn_args) > argument and fn_args[argument] or None
        else:
            if argument in fn_args:
                pos_arg = fn_args.index(argument)
            else:
                pos_arg = None
            named_arg = argument
        del fn_args

    def wrapper(*args, **kw):
        if before:
            if pos_arg is None:
                if named_arg not in kw:
                    raise sa_exc.ArgumentError(
                        "Missing argument %s" % argument
                    )
                value = kw[named_arg]
            else:
                if len(args) > pos_arg:
                    value = args[pos_arg]
                elif named_arg in kw:
                    value = kw[named_arg]
                else:
                    raise sa_exc.ArgumentError(
                        "Missing argument %s" % argument
                    )

        initiator = kw.pop("_sa_initiator", None)
        if initiator is False:
            executor = None
        else:
            executor = args[0]._sa_adapter

        if before and executor:
            getattr(executor, before)(value, initiator)

        if not after or not executor:
            return method(*args, **kw)
        else:
            res = method(*args, **kw)
            if res is not None:
                getattr(executor, after)(res, initiator)
            return res

    wrapper._sa_instrumented = True  # type: ignore[attr-defined]
    if hasattr(method, "_sa_instrument_role"):
        wrapper._sa_instrument_role = method._sa_instrument_role  # type: ignore[attr-defined]  # noqa: E501
    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


def __set_wo_mutation(collection, item, _sa_initiator=None):
    """Run set wo mutation events.

    The collection is not mutated.

    """
    if _sa_initiator is not False:
        executor = collection._sa_adapter
        if executor:
            executor.fire_append_wo_mutation_event(
                item, _sa_initiator, key=None
            )


def __set(collection, item, _sa_initiator, key):
    """Run set events.

    This event always occurs before the collection is actually mutated.

    """

    if _sa_initiator is not False:
        executor = collection._sa_adapter
        if executor:
            item = executor.fire_append_event(item, _sa_initiator, key=key)
    return item


def __del(collection, item, _sa_initiator, key):
    """Run del events.

    This event occurs before the collection is actually mutated, *except*
    in the case of a pop operation, in which case it occurs afterwards.
    For pop operations, the __before_pop hook is called before the
    operation occurs.

    """
    if _sa_initiator is not False:
        executor = collection._sa_adapter
        if executor:
            executor.fire_remove_event(item, _sa_initiator, key=key)


def __before_pop(collection, _sa_initiator=None):
    """An event which occurs on a before a pop() operation occurs."""
    executor = collection._sa_adapter
    if executor:
        executor.fire_pre_remove_event(_sa_initiator)


def _list_decorators() -> Dict[str, Callable[[_FN], _FN]]:
    """Tailored instrumentation wrappers for any list-like class."""

    def _tidy(fn):
        fn._sa_instrumented = True
        fn.__doc__ = getattr(list, fn.__name__).__doc__

    def append(fn):
        def append(self, item, _sa_initiator=None):
            item = __set(self, item, _sa_initiator, NO_KEY)
            fn(self, item)

        _tidy(append)
        return append

    def remove(fn):
        def remove(self, value, _sa_initiator=None):
            __del(self, value, _sa_initiator, NO_KEY)
            # testlib.pragma exempt:__eq__
            fn(self, value)

        _tidy(remove)
        return remove

    def insert(fn):
        def insert(self, index, value):
            value = __set(self, value, None, index)
            fn(self, index, value)

        _tidy(insert)
        return insert

    def __setitem__(fn):
        def __setitem__(self, index, value):
            if not isinstance(index, slice):
                existing = self[index]
                if existing is not None:
                    __del(self, existing, None, index)
                value = __set(self, value, None, index)
                fn(self, index, value)
            else:
                # slice assignment requires __delitem__, insert, __len__
                step = index.step or 1
                start = index.start or 0
                if start < 0:
                    start += len(self)
                if index.stop is not None:
                    stop = index.stop
                else:
                    stop = len(self)
                if stop < 0:
                    stop += len(self)

                if step == 1:
                    if value is self:
                        return
                    for i in range(start, stop, step):
                        if len(self) > start:
                            del self[start]

                    for i, item in enumerate(value):
                        self.insert(i + start, item)
                else:
                    rng = list(range(start, stop, step))
                    if len(value) != len(rng):
                        raise ValueError(
                            "attempt to assign sequence of size %s to "
                            "extended slice of size %s"
                            % (len(value), len(rng))
                        )
                    for i, item in zip(rng, value):
                        self.__setitem__(i, item)

        _tidy(__setitem__)
        return __setitem__

    def __delitem__(fn):
        def __delitem__(self, index):
            if not isinstance(index, slice):
                item = self[index]
                __del(self, item, None, index)
                fn(self, index)
            else:
                # slice deletion requires __getslice__ and a slice-groking
                # __getitem__ for stepped deletion
                # note: not breaking this into atomic dels
                for item in self[index]:
                    __del(self, item, None, index)
                fn(self, index)

        _tidy(__delitem__)
        return __delitem__

    def extend(fn):
        def extend(self, iterable):
            for value in list(iterable):
                self.append(value)

        _tidy(extend)
        return extend

    def __iadd__(fn):
        def __iadd__(self, iterable):
            # list.__iadd__ takes any iterable and seems to let TypeError
            # raise as-is instead of returning NotImplemented
            for value in list(iterable):
                self.append(value)
            return self

        _tidy(__iadd__)
        return __iadd__

    def pop(fn):
        def pop(self, index=-1):
            __before_pop(self)
            item = fn(self, index)
            __del(self, item, None, index)
            return item

        _tidy(pop)
        return pop

    def clear(fn):
        def clear(self, index=-1):
            for item in self:
                __del(self, item, None, index)
            fn(self)

        _tidy(clear)
        return clear

    # __imul__ : not wrapping this.  all members of the collection are already
    # present, so no need to fire appends... wrapping it with an explicit
    # decorator is still possible, so events on *= can be had if they're
    # desired.  hard to imagine a use case for __imul__, though.

    l = locals().copy()
    l.pop("_tidy")
    return l


def _dict_decorators() -> Dict[str, Callable[[_FN], _FN]]:
    """Tailored instrumentation wrappers for any dict-like mapping class."""

    def _tidy(fn):
        fn._sa_instrumented = True
        fn.__doc__ = getattr(dict, fn.__name__).__doc__

    def __setitem__(fn):
        def __setitem__(self, key, value, _sa_initiator=None):
            if key in self:
                __del(self, self[key], _sa_initiator, key)
            value = __set(self, value, _sa_initiator, key)
            fn(self, key, value)

        _tidy(__setitem__)
        return __setitem__

    def __delitem__(fn):
        def __delitem__(self, key, _sa_initiator=None):
            if key in self:
                __del(self, self[key], _sa_initiator, key)
            fn(self, key)

        _tidy(__delitem__)
        return __delitem__

    def clear(fn):
        def clear(self):
            for key in self:
                __del(self, self[key], None, key)
            fn(self)

        _tidy(clear)
        return clear

    def pop(fn):
        def pop(self, key, default=NO_ARG):
            __before_pop(self)
            _to_del = key in self
            if default is NO_ARG:
                item = fn(self, key)
            else:
                item = fn(self, key, default)
            if _to_del:
                __del(self, item, None, key)
            return item

        _tidy(pop)
        return pop

    def popitem(fn):
        def popitem(self):
            __before_pop(self)
            item = fn(self)
            __del(self, item[1], None, 1)
            return item

        _tidy(popitem)
        return popitem

    def setdefault(fn):
        def setdefault(self, key, default=None):
            if key not in self:
                self.__setitem__(key, default)
                return default
            else:
                value = self.__getitem__(key)
                if value is default:
                    __set_wo_mutation(self, value, None)

                return value

        _tidy(setdefault)
        return setdefault

    def update(fn):
        def update(self, __other=NO_ARG, **kw):
            if __other is not NO_ARG:
                if hasattr(__other, "keys"):
                    for key in list(__other):
                        if key not in self or self[key] is not __other[key]:
                            self[key] = __other[key]
                        else:
                            __set_wo_mutation(self, __other[key], None)
                else:
                    for key, value in __other:
                        if key not in self or self[key] is not value:
                            self[key] = value
                        else:
                            __set_wo_mutation(self, value, None)
            for key in kw:
                if key not in self or self[key] is not kw[key]:
                    self[key] = kw[key]
                else:
                    __set_wo_mutation(self, kw[key], None)

        _tidy(update)
        return update

    l = locals().copy()
    l.pop("_tidy")
    return l


_set_binop_bases = (set, frozenset)


def _set_binops_check_strict(self: Any, obj: Any) -> bool:
    """Allow only set, frozenset and self.__class__-derived
    objects in binops."""
    return isinstance(obj, _set_binop_bases + (self.__class__,))


def _set_binops_check_loose(self: Any, obj: Any) -> bool:
    """Allow anything set-like to participate in set binops."""
    return (
        isinstance(obj, _set_binop_bases + (self.__class__,))
        or util.duck_type_collection(obj) == set
    )


def _set_decorators() -> Dict[str, Callable[[_FN], _FN]]:
    """Tailored instrumentation wrappers for any set-like class."""

    def _tidy(fn):
        fn._sa_instrumented = True
        fn.__doc__ = getattr(set, fn.__name__).__doc__

    def add(fn):
        def add(self, value, _sa_initiator=None):
            if value not in self:
                value = __set(self, value, _sa_initiator, NO_KEY)
            else:
                __set_wo_mutation(self, value, _sa_initiator)
            # testlib.pragma exempt:__hash__
            fn(self, value)

        _tidy(add)
        return add

    def discard(fn):
        def discard(self, value, _sa_initiator=None):
            # testlib.pragma exempt:__hash__
            if value in self:
                __del(self, value, _sa_initiator, NO_KEY)
                # testlib.pragma exempt:__hash__
            fn(self, value)

        _tidy(discard)
        return discard

    def remove(fn):
        def remove(self, value, _sa_initiator=None):
            # testlib.pragma exempt:__hash__
            if value in self:
                __del(self, value, _sa_initiator, NO_KEY)
            # testlib.pragma exempt:__hash__
            fn(self, value)

        _tidy(remove)
        return remove

    def pop(fn):
        def pop(self):
            __before_pop(self)
            item = fn(self)
            # for set in particular, we have no way to access the item
            # that will be popped before pop is called.
            __del(self, item, None, NO_KEY)
            return item

        _tidy(pop)
        return pop

    def clear(fn):
        def clear(self):
            for item in list(self):
                self.remove(item)

        _tidy(clear)
        return clear

    def update(fn):
        def update(self, value):
            for item in value:
                self.add(item)

        _tidy(update)
        return update

    def __ior__(fn):
        def __ior__(self, value):
            if not _set_binops_check_strict(self, value):
                return NotImplemented
            for item in value:
                self.add(item)
            return self

        _tidy(__ior__)
        return __ior__

    def difference_update(fn):
        def difference_update(self, value):
            for item in value:
                self.discard(item)

        _tidy(difference_update)
        return difference_update

    def __isub__(fn):
        def __isub__(self, value):
            if not _set_binops_check_strict(self, value):
                return NotImplemented
            for item in value:
                self.discard(item)
            return self

        _tidy(__isub__)
        return __isub__

    def intersection_update(fn):
        def intersection_update(self, other):
            want, have = self.intersection(other), set(self)
            remove, add = have - want, want - have

            for item in remove:
                self.remove(item)
            for item in add:
                self.add(item)

        _tidy(intersection_update)
        return intersection_update

    def __iand__(fn):
        def __iand__(self, other):
            if not _set_binops_check_strict(self, other):
                return NotImplemented
            want, have = self.intersection(other), set(self)
            remove, add = have - want, want - have

            for item in remove:
                self.remove(item)
            for item in add:
                self.add(item)
            return self

        _tidy(__iand__)
        return __iand__

    def symmetric_difference_update(fn):
        def symmetric_difference_update(self, other):
            want, have = self.symmetric_difference(other), set(self)
            remove, add = have - want, want - have

            for item in remove:
                self.remove(item)
            for item in add:
                self.add(item)

        _tidy(symmetric_difference_update)
        return symmetric_difference_update

    def __ixor__(fn):
        def __ixor__(self, other):
            if not _set_binops_check_strict(self, other):
                return NotImplemented
            want, have = self.symmetric_difference(other), set(self)
            remove, add = have - want, want - have

            for item in remove:
                self.remove(item)
            for item in add:
                self.add(item)
            return self

        _tidy(__ixor__)
        return __ixor__

    l = locals().copy()
    l.pop("_tidy")
    return l


class InstrumentedList(List[_T]):
    """An instrumented version of the built-in list."""


class InstrumentedSet(Set[_T]):
    """An instrumented version of the built-in set."""


class InstrumentedDict(Dict[_KT, _VT]):
    """An instrumented version of the built-in dict."""


__canned_instrumentation: util.immutabledict[Any, _CollectionFactoryType] = (
    util.immutabledict(
        {
            list: InstrumentedList,
            set: InstrumentedSet,
            dict: InstrumentedDict,
        }
    )
)

__interfaces: util.immutabledict[
    Any,
    Tuple[
        Dict[str, str],
        Dict[str, Callable[..., Any]],
    ],
] = util.immutabledict(
    {
        list: (
            {
                "appender": "append",
                "remover": "remove",
                "iterator": "__iter__",
            },
            _list_decorators(),
        ),
        set: (
            {"appender": "add", "remover": "remove", "iterator": "__iter__"},
            _set_decorators(),
        ),
        # decorators are required for dicts and object collections.
        dict: ({"iterator": "values"}, _dict_decorators()),
    }
)


def __go(lcls):
    global keyfunc_mapping, mapped_collection
    global column_keyed_dict, column_mapped_collection
    global MappedCollection, KeyFuncDict
    global attribute_keyed_dict, attribute_mapped_collection

    from .mapped_collection import keyfunc_mapping
    from .mapped_collection import column_keyed_dict
    from .mapped_collection import attribute_keyed_dict
    from .mapped_collection import KeyFuncDict

    from .mapped_collection import mapped_collection
    from .mapped_collection import column_mapped_collection
    from .mapped_collection import attribute_mapped_collection
    from .mapped_collection import MappedCollection

    # ensure instrumentation is associated with
    # these built-in classes; if a user-defined class
    # subclasses these and uses @internally_instrumented,
    # the superclass is otherwise not instrumented.
    # see [ticket:2406].
    _instrument_class(InstrumentedList)
    _instrument_class(InstrumentedSet)
    _instrument_class(KeyFuncDict)


__go(locals())