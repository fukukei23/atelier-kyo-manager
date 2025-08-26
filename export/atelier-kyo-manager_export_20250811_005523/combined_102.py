
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\recurr.py ===
r"""
This module is intended for solving recurrences or, in other words,
difference equations. Currently supported are linear, inhomogeneous
equations with polynomial or rational coefficients.

The solutions are obtained among polynomials, rational functions,
hypergeometric terms, or combinations of hypergeometric term which
are pairwise dissimilar.

``rsolve_X`` functions were meant as a low level interface
for ``rsolve`` which would use Mathematica's syntax.

Given a recurrence relation:

    .. math:: a_{k}(n) y(n+k) + a_{k-1}(n) y(n+k-1) +
              ... + a_{0}(n) y(n) = f(n)

where `k > 0` and `a_{i}(n)` are polynomials in `n`. To use
``rsolve_X`` we need to put all coefficients in to a list ``L`` of
`k+1` elements the following way:

    ``L = [a_{0}(n), ..., a_{k-1}(n), a_{k}(n)]``

where ``L[i]``, for `i=0, \ldots, k`, maps to
`a_{i}(n) y(n+i)` (`y(n+i)` is implicit).

For example if we would like to compute `m`-th Bernoulli polynomial
up to a constant (example was taken from rsolve_poly docstring),
then we would use `b(n+1) - b(n) = m n^{m-1}` recurrence, which
has solution `b(n) = B_m + C`.

Then ``L = [-1, 1]`` and `f(n) = m n^(m-1)` and finally for `m=4`:

>>> from sympy import Symbol, bernoulli, rsolve_poly
>>> n = Symbol('n', integer=True)

>>> rsolve_poly([-1, 1], 4*n**3, n)
C0 + n**4 - 2*n**3 + n**2

>>> bernoulli(4, n)
n**4 - 2*n**3 + n**2 - 1/30

For the sake of completeness, `f(n)` can be:

    [1] a polynomial               -> rsolve_poly
    [2] a rational function        -> rsolve_ratio
    [3] a hypergeometric function  -> rsolve_hyper
"""
from collections import defaultdict

from sympy.concrete import product
from sympy.core.singleton import S
from sympy.core.numbers import Rational, I
from sympy.core.symbol import Symbol, Wild, Dummy
from sympy.core.relational import Equality
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify

from sympy.simplify import simplify, hypersimp, hypersimilar  # type: ignore
from sympy.solvers import solve, solve_undetermined_coeffs
from sympy.polys import Poly, quo, gcd, lcm, roots, resultant
from sympy.functions import binomial, factorial, FallingFactorial, RisingFactorial
from sympy.matrices import Matrix, casoratian
from sympy.utilities.iterables import numbered_symbols


def rsolve_poly(coeffs, f, n, shift=0, **hints):
    r"""
    Given linear recurrence operator `\operatorname{L}` of order
    `k` with polynomial coefficients and inhomogeneous equation
    `\operatorname{L} y = f`, where `f` is a polynomial, we seek for
    all polynomial solutions over field `K` of characteristic zero.

    The algorithm performs two basic steps:

        (1) Compute degree `N` of the general polynomial solution.
        (2) Find all polynomials of degree `N` or less
            of `\operatorname{L} y = f`.

    There are two methods for computing the polynomial solutions.
    If the degree bound is relatively small, i.e. it's smaller than
    or equal to the order of the recurrence, then naive method of
    undetermined coefficients is being used. This gives a system
    of algebraic equations with `N+1` unknowns.

    In the other case, the algorithm performs transformation of the
    initial equation to an equivalent one for which the system of
    algebraic equations has only `r` indeterminates. This method is
    quite sophisticated (in comparison with the naive one) and was
    invented together by Abramov, Bronstein and Petkovsek.

    It is possible to generalize the algorithm implemented here to
    the case of linear q-difference and differential equations.

    Lets say that we would like to compute `m`-th Bernoulli polynomial
    up to a constant. For this we can use `b(n+1) - b(n) = m n^{m-1}`
    recurrence, which has solution `b(n) = B_m + C`. For example:

    >>> from sympy import Symbol, rsolve_poly
    >>> n = Symbol('n', integer=True)

    >>> rsolve_poly([-1, 1], 4*n**3, n)
    C0 + n**4 - 2*n**3 + n**2

    References
    ==========

    .. [1] S. A. Abramov, M. Bronstein and M. Petkovsek, On polynomial
           solutions of linear operator equations, in: T. Levelt, ed.,
           Proc. ISSAC '95, ACM Press, New York, 1995, 290-296.

    .. [2] M. Petkovsek, Hypergeometric solutions of linear recurrences
           with polynomial coefficients, J. Symbolic Computation,
           14 (1992), 243-264.

    .. [3] M. Petkovsek, H. S. Wilf, D. Zeilberger, A = B, 1996.

    """
    f = sympify(f)

    if not f.is_polynomial(n):
        return None

    homogeneous = f.is_zero

    r = len(coeffs) - 1

    coeffs = [Poly(coeff, n) for coeff in coeffs]

    polys = [Poly(0, n)]*(r + 1)
    terms = [(S.Zero, S.NegativeInfinity)]*(r + 1)

    for i in range(r + 1):
        for j in range(i, r + 1):
            polys[i] += coeffs[j]*(binomial(j, i).as_poly(n))

        if not polys[i].is_zero:
            (exp,), coeff = polys[i].LT()
            terms[i] = (coeff, exp)

    d = b = terms[0][1]

    for i in range(1, r + 1):
        if terms[i][1] > d:
            d = terms[i][1]

        if terms[i][1] - i > b:
            b = terms[i][1] - i

    d, b = int(d), int(b)

    x = Dummy('x')

    degree_poly = S.Zero

    for i in range(r + 1):
        if terms[i][1] - i == b:
            degree_poly += terms[i][0]*FallingFactorial(x, i)

    nni_roots = list(roots(degree_poly, x, filter='Z',
        predicate=lambda r: r >= 0).keys())

    if nni_roots:
        N = [max(nni_roots)]
    else:
        N = []

    if homogeneous:
        N += [-b - 1]
    else:
        N += [f.as_poly(n).degree() - b, -b - 1]

    N = int(max(N))

    if N < 0:
        if homogeneous:
            if hints.get('symbols', False):
                return (S.Zero, [])
            else:
                return S.Zero
        else:
            return None

    if N <= r:
        C = []
        y = E = S.Zero

        for i in range(N + 1):
            C.append(Symbol('C' + str(i + shift)))
            y += C[i] * n**i

        for i in range(r + 1):
            E += coeffs[i].as_expr()*y.subs(n, n + i)

        solutions = solve_undetermined_coeffs(E - f, C, n)

        if solutions is not None:
            _C = C
            C = [c for c in C if (c not in solutions)]
            result = y.subs(solutions)
        else:
            return None  # TBD
    else:
        A = r
        U = N + A + b + 1

        nni_roots = list(roots(polys[r], filter='Z',
            predicate=lambda r: r >= 0).keys())

        if nni_roots != []:
            a = max(nni_roots) + 1
        else:
            a = S.Zero

        def _zero_vector(k):
            return [S.Zero] * k

        def _one_vector(k):
            return [S.One] * k

        def _delta(p, k):
            B = S.One
            D = p.subs(n, a + k)

            for i in range(1, k + 1):
                B *= Rational(i - k - 1, i)
                D += B * p.subs(n, a + k - i)

            return D

        alpha = {}

        for i in range(-A, d + 1):
            I = _one_vector(d + 1)

            for k in range(1, d + 1):
                I[k] = I[k - 1] * (x + i - k + 1)/k

            alpha[i] = S.Zero

            for j in range(A + 1):
                for k in range(d + 1):
                    B = binomial(k, i + j)
                    D = _delta(polys[j].as_expr(), k)

                    alpha[i] += I[k]*B*D

        V = Matrix(U, A, lambda i, j: int(i == j))

        if homogeneous:
            for i in range(A, U):
                v = _zero_vector(A)

                for k in range(1, A + b + 1):
                    if i - k < 0:
                        break

                    B = alpha[k - A].subs(x, i - k)

                    for j in range(A):
                        v[j] += B * V[i - k, j]

                denom = alpha[-A].subs(x, i)

                for j in range(A):
                    V[i, j] = -v[j] / denom
        else:
            G = _zero_vector(U)

            for i in range(A, U):
                v = _zero_vector(A)
                g = S.Zero

                for k in range(1, A + b + 1):
                    if i - k < 0:
                        break

                    B = alpha[k - A].subs(x, i - k)

                    for j in range(A):
                        v[j] += B * V[i - k, j]

                    g += B * G[i - k]

                denom = alpha[-A].subs(x, i)

                for j in range(A):
                    V[i, j] = -v[j] / denom

                G[i] = (_delta(f, i - A) - g) / denom

        P, Q = _one_vector(U), _zero_vector(A)

        for i in range(1, U):
            P[i] = (P[i - 1] * (n - a - i + 1)/i).expand()

        for i in range(A):
            Q[i] = Add(*[(v*p).expand() for v, p in zip(V[:, i], P)])

        if not homogeneous:
            h = Add(*[(g*p).expand() for g, p in zip(G, P)])

        C = [Symbol('C' + str(i + shift)) for i in range(A)]

        g = lambda i: Add(*[c*_delta(q, i) for c, q in zip(C, Q)])

        if homogeneous:
            E = [g(i) for i in range(N + 1, U)]
        else:
            E = [g(i) + _delta(h, i) for i in range(N + 1, U)]

        if E != []:
            solutions = solve(E, *C)

            if not solutions:
                if homogeneous:
                    if hints.get('symbols', False):
                        return (S.Zero, [])
                    else:
                        return S.Zero
                else:
                    return None
        else:
            solutions = {}

        if homogeneous:
            result = S.Zero
        else:
            result = h

        _C = C[:]
        for c, q in list(zip(C, Q)):
            if c in solutions:
                s = solutions[c]*q
                C.remove(c)
            else:
                s = c*q

            result += s.expand()

    if C != _C:
        # renumber so they are contiguous
        result = result.xreplace(dict(zip(C, _C)))
        C = _C[:len(C)]

    if hints.get('symbols', False):
        return (result, C)
    else:
        return result


def rsolve_ratio(coeffs, f, n, **hints):
    r"""
    Given linear recurrence operator `\operatorname{L}` of order `k`
    with polynomial coefficients and inhomogeneous equation
    `\operatorname{L} y = f`, where `f` is a polynomial, we seek
    for all rational solutions over field `K` of characteristic zero.

    This procedure accepts only polynomials, however if you are
    interested in solving recurrence with rational coefficients
    then use ``rsolve`` which will pre-process the given equation
    and run this procedure with polynomial arguments.

    The algorithm performs two basic steps:

        (1) Compute polynomial `v(n)` which can be used as universal
            denominator of any rational solution of equation
            `\operatorname{L} y = f`.

        (2) Construct new linear difference equation by substitution
            `y(n) = u(n)/v(n)` and solve it for `u(n)` finding all its
            polynomial solutions. Return ``None`` if none were found.

    The algorithm implemented here is a revised version of the original
    Abramov's algorithm, developed in 1989. The new approach is much
    simpler to implement and has better overall efficiency. This
    method can be easily adapted to the q-difference equations case.

    Besides finding rational solutions alone, this functions is
    an important part of Hyper algorithm where it is used to find
    a particular solution for the inhomogeneous part of a recurrence.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.solvers.recurr import rsolve_ratio
    >>> rsolve_ratio([-2*x**3 + x**2 + 2*x - 1, 2*x**3 + x**2 - 6*x,
    ... - 2*x**3 - 11*x**2 - 18*x - 9, 2*x**3 + 13*x**2 + 22*x + 8], 0, x)
    C0*(2*x - 3)/(2*(x**2 - 1))

    References
    ==========

    .. [1] S. A. Abramov, Rational solutions of linear difference
           and q-difference equations with polynomial coefficients,
           in: T. Levelt, ed., Proc. ISSAC '95, ACM Press, New York,
           1995, 285-289

    See Also
    ========

    rsolve_hyper
    """
    f = sympify(f)

    if not f.is_polynomial(n):
        return None

    coeffs = list(map(sympify, coeffs))

    r = len(coeffs) - 1

    A, B = coeffs[r], coeffs[0]
    A = A.subs(n, n - r).expand()

    h = Dummy('h')

    res = resultant(A, B.subs(n, n + h), n)

    if not res.is_polynomial(h):
        p, q = res.as_numer_denom()
        res = quo(p, q, h)

    nni_roots = list(roots(res, h, filter='Z',
        predicate=lambda r: r >= 0).keys())

    if not nni_roots:
        return rsolve_poly(coeffs, f, n, **hints)
    else:
        C, numers = S.One, [S.Zero]*(r + 1)

        for i in range(int(max(nni_roots)), -1, -1):
            d = gcd(A, B.subs(n, n + i), n)

            A = quo(A, d, n)
            B = quo(B, d.subs(n, n - i), n)

            C *= Mul(*[d.subs(n, n - j) for j in range(i + 1)])

        denoms = [C.subs(n, n + i) for i in range(r + 1)]

        for i in range(r + 1):
            g = gcd(coeffs[i], denoms[i], n)

            numers[i] = quo(coeffs[i], g, n)
            denoms[i] = quo(denoms[i], g, n)

        for i in range(r + 1):
            numers[i] *= Mul(*(denoms[:i] + denoms[i + 1:]))

        result = rsolve_poly(numers, f * Mul(*denoms), n, **hints)

        if result is not None:
            if hints.get('symbols', False):
                return (simplify(result[0] / C), result[1])
            else:
                return simplify(result / C)
        else:
            return None


def rsolve_hyper(coeffs, f, n, **hints):
    r"""
    Given linear recurrence operator `\operatorname{L}` of order `k`
    with polynomial coefficients and inhomogeneous equation
    `\operatorname{L} y = f` we seek for all hypergeometric solutions
    over field `K` of characteristic zero.

    The inhomogeneous part can be either hypergeometric or a sum
    of a fixed number of pairwise dissimilar hypergeometric terms.

    The algorithm performs three basic steps:

        (1) Group together similar hypergeometric terms in the
            inhomogeneous part of `\operatorname{L} y = f`, and find
            particular solution using Abramov's algorithm.

        (2) Compute generating set of `\operatorname{L}` and find basis
            in it, so that all solutions are linearly independent.

        (3) Form final solution with the number of arbitrary
            constants equal to dimension of basis of `\operatorname{L}`.

    Term `a(n)` is hypergeometric if it is annihilated by first order
    linear difference equations with polynomial coefficients or, in
    simpler words, if consecutive term ratio is a rational function.

    The output of this procedure is a linear combination of fixed
    number of hypergeometric terms. However the underlying method
    can generate larger class of solutions - D'Alembertian terms.

    Note also that this method not only computes the kernel of the
    inhomogeneous equation, but also reduces in to a basis so that
    solutions generated by this procedure are linearly independent

    Examples
    ========

    >>> from sympy.solvers import rsolve_hyper
    >>> from sympy.abc import x

    >>> rsolve_hyper([-1, -1, 1], 0, x)
    C0*(1/2 - sqrt(5)/2)**x + C1*(1/2 + sqrt(5)/2)**x

    >>> rsolve_hyper([-1, 1], 1 + x, x)
    C0 + x*(x + 1)/2

    References
    ==========

    .. [1] M. Petkovsek, Hypergeometric solutions of linear recurrences
           with polynomial coefficients, J. Symbolic Computation,
           14 (1992), 243-264.

    .. [2] M. Petkovsek, H. S. Wilf, D. Zeilberger, A = B, 1996.
    """
    coeffs = list(map(sympify, coeffs))

    f = sympify(f)

    r, kernel, symbols = len(coeffs) - 1, [], set()

    if not f.is_zero:
        if f.is_Add:
            similar = {}

            for g in f.expand().args:
                if not g.is_hypergeometric(n):
                    return None

                for h in similar.keys():
                    if hypersimilar(g, h, n):
                        similar[h] += g
                        break
                else:
                    similar[g] = S.Zero

            inhomogeneous = [g + h for g, h in similar.items()]
        elif f.is_hypergeometric(n):
            inhomogeneous = [f]
        else:
            return None

        for i, g in enumerate(inhomogeneous):
            coeff, polys = S.One, coeffs[:]
            denoms = [S.One]*(r + 1)

            s = hypersimp(g, n)

            for j in range(1, r + 1):
                coeff *= s.subs(n, n + j - 1)

                p, q = coeff.as_numer_denom()

                polys[j] *= p
                denoms[j] = q

            for j in range(r + 1):
                polys[j] *= Mul(*(denoms[:j] + denoms[j + 1:]))

            # FIXME: The call to rsolve_ratio below should suffice (rsolve_poly
            # call can be removed) but the XFAIL test_rsolve_ratio_missed must
            # be fixed first.
            R = rsolve_ratio(polys, Mul(*denoms), n, symbols=True)
            if R is not None:
                R, syms = R
                if syms:
                    R = R.subs(zip(syms, [0]*len(syms)))
            else:
                R = rsolve_poly(polys, Mul(*denoms), n)

            if R:
                inhomogeneous[i] *= R
            else:
                return None

            result = Add(*inhomogeneous)
            result = simplify(result)
    else:
        result = S.Zero

    Z = Dummy('Z')

    p, q = coeffs[0], coeffs[r].subs(n, n - r + 1)

    p_factors = list(roots(p, n).keys())
    q_factors = list(roots(q, n).keys())

    factors = [(S.One, S.One)]

    for p in p_factors:
        for q in q_factors:
            if p.is_integer and q.is_integer and p <= q:
                continue
            else:
                factors += [(n - p, n - q)]

    p = [(n - p, S.One) for p in p_factors]
    q = [(S.One, n - q) for q in q_factors]

    factors = p + factors + q

    for A, B in factors:
        polys, degrees = [], []
        D = A*B.subs(n, n + r - 1)

        for i in range(r + 1):
            a = Mul(*[A.subs(n, n + j) for j in range(i)])
            b = Mul(*[B.subs(n, n + j) for j in range(i, r)])

            poly = quo(coeffs[i]*a*b, D, n)
            polys.append(poly.as_poly(n))

            if not poly.is_zero:
                degrees.append(polys[i].degree())

        if degrees:
            d, poly = max(degrees), S.Zero
        else:
            return None

        for i in range(r + 1):
            coeff = polys[i].nth(d)

            if coeff is not S.Zero:
                poly += coeff * Z**i

        for z in roots(poly, Z).keys():
            if z.is_zero:
                continue

            recurr_coeffs = [polys[i].as_expr()*z**i for i in range(r + 1)]
            if d == 0 and 0 != Add(*[recurr_coeffs[j]*j for j in range(1, r + 1)]):
                # faster inline check (than calling rsolve_poly) for a
                # constant solution to a constant coefficient recurrence.
                sol = [Symbol("C" + str(len(symbols)))]
            else:
                sol, syms = rsolve_poly(recurr_coeffs, 0, n, len(symbols), symbols=True)
                sol = sol.collect(syms)
                sol = [sol.coeff(s) for s in syms]

            for C in sol:
                ratio = z * A * C.subs(n, n + 1) / B / C
                ratio = simplify(ratio)
                # If there is a nonnegative root in the denominator of the ratio,
                # this indicates that the term y(n_root) is zero, and one should
                # start the product with the term y(n_root + 1).
                n0 = 0
                for n_root in roots(ratio.as_numer_denom()[1], n).keys():
                    if n_root.has(I):
                        return None
                    elif (n0 < (n_root + 1)) == True:
                        n0 = n_root + 1
                K = product(ratio, (n, n0, n - 1))
                if K.has(factorial, FallingFactorial, RisingFactorial):
                    K = simplify(K)

                if casoratian(kernel + [K], n, zero=False) != 0:
                    kernel.append(K)

    kernel.sort(key=default_sort_key)
    sk = list(zip(numbered_symbols('C'), kernel))

    for C, ker in sk:
        result += C * ker

    if hints.get('symbols', False):
        # XXX: This returns the symbols in a non-deterministic order
        symbols |= {s for s, k in sk}
        return (result, list(symbols))
    else:
        return result


def rsolve(f, y, init=None):
    r"""
    Solve univariate recurrence with rational coefficients.

    Given `k`-th order linear recurrence `\operatorname{L} y = f`,
    or equivalently:

    .. math:: a_{k}(n) y(n+k) + a_{k-1}(n) y(n+k-1) +
              \cdots + a_{0}(n) y(n) = f(n)

    where `a_{i}(n)`, for `i=0, \ldots, k`, are polynomials or rational
    functions in `n`, and `f` is a hypergeometric function or a sum
    of a fixed number of pairwise dissimilar hypergeometric terms in
    `n`, finds all solutions or returns ``None``, if none were found.

    Initial conditions can be given as a dictionary in two forms:

        (1) ``{  n_0  : v_0,   n_1  : v_1, ...,   n_m  : v_m}``
        (2) ``{y(n_0) : v_0, y(n_1) : v_1, ..., y(n_m) : v_m}``

    or as a list ``L`` of values:

        ``L = [v_0, v_1, ..., v_m]``

    where ``L[i] = v_i``, for `i=0, \ldots, m`, maps to `y(n_i)`.

    Examples
    ========

    Lets consider the following recurrence:

    .. math:: (n - 1) y(n + 2) - (n^2 + 3 n - 2) y(n + 1) +
              2 n (n + 1) y(n) = 0

    >>> from sympy import Function, rsolve
    >>> from sympy.abc import n
    >>> y = Function('y')

    >>> f = (n - 1)*y(n + 2) - (n**2 + 3*n - 2)*y(n + 1) + 2*n*(n + 1)*y(n)

    >>> rsolve(f, y(n))
    2**n*C0 + C1*factorial(n)

    >>> rsolve(f, y(n), {y(0):0, y(1):3})
    3*2**n - 3*factorial(n)

    See Also
    ========

    rsolve_poly, rsolve_ratio, rsolve_hyper

    """
    if isinstance(f, Equality):
        f = f.lhs - f.rhs

    n = y.args[0]
    k = Wild('k', exclude=(n,))

    # Preprocess user input to allow things like
    # y(n) + a*(y(n + 1) + y(n - 1))/2
    f = f.expand().collect(y.func(Wild('m', integer=True)))

    h_part = defaultdict(list)
    i_part = []
    for g in Add.make_args(f):
        coeff, dep = g.as_coeff_mul(y.func)
        if not dep:
            i_part.append(coeff)
            continue
        for h in dep:
            if h.is_Function and h.func == y.func:
                result = h.args[0].match(n + k)
                if result is not None:
                    h_part[int(result[k])].append(coeff)
                    continue
            raise ValueError(
                "'%s(%s + k)' expected, got '%s'" % (y.func, n, h))
    for k in h_part:
        h_part[k] = Add(*h_part[k])
    h_part.default_factory = lambda: 0
    i_part = Add(*i_part)

    for k, coeff in h_part.items():
        h_part[k] = simplify(coeff)

    common = S.One

    if not i_part.is_zero and not i_part.is_hypergeometric(n) and \
       not (i_part.is_Add and all((x.is_hypergeometric(n) for x in i_part.expand().args))):
        raise ValueError("The independent term should be a sum of hypergeometric functions, got '%s'" % i_part)

    for coeff in h_part.values():
        if coeff.is_rational_function(n):
            if not coeff.is_polynomial(n):
                common = lcm(common, coeff.as_numer_denom()[1], n)
        else:
            raise ValueError(
                "Polynomial or rational function expected, got '%s'" % coeff)

    i_numer, i_denom = i_part.as_numer_denom()

    if i_denom.is_polynomial(n):
        common = lcm(common, i_denom, n)

    if common is not S.One:
        for k, coeff in h_part.items():
            numer, denom = coeff.as_numer_denom()
            h_part[k] = numer*quo(common, denom, n)

        i_part = i_numer*quo(common, i_denom, n)

    K_min = min(h_part.keys())

    if K_min < 0:
        K = abs(K_min)

        H_part = defaultdict(lambda: S.Zero)
        i_part = i_part.subs(n, n + K).expand()
        common = common.subs(n, n + K).expand()

        for k, coeff in h_part.items():
            H_part[k + K] = coeff.subs(n, n + K).expand()
    else:
        H_part = h_part

    K_max = max(H_part.keys())
    coeffs = [H_part[i] for i in range(K_max + 1)]

    result = rsolve_hyper(coeffs, -i_part, n, symbols=True)

    if result is None:
        return None

    solution, symbols = result

    if init in ({}, []):
        init = None

    if symbols and init is not None:
        if isinstance(init, list):
            init = {i: init[i] for i in range(len(init))}

        equations = []

        for k, v in init.items():
            try:
                i = int(k)
            except TypeError:
                if k.is_Function and k.func == y.func:
                    i = int(k.args[0])
                else:
                    raise ValueError("Integer or term expected, got '%s'" % k)

            eq = solution.subs(n, i) - v
            if eq.has(S.NaN):
                eq = solution.limit(n, i) - v
            equations.append(eq)

        result = solve(equations, *symbols)

        if not result:
            return None
        else:
            solution = solution.subs(result)

    return solution

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\computation\expr.py ===
"""
:func:`~pandas.eval` parsers.
"""
from __future__ import annotations

import ast
from functools import (
    partial,
    reduce,
)
from keyword import iskeyword
import tokenize
from typing import (
    Callable,
    ClassVar,
    TypeVar,
)

import numpy as np

from pandas.errors import UndefinedVariableError

from pandas.core.dtypes.common import is_string_dtype

import pandas.core.common as com
from pandas.core.computation.ops import (
    ARITH_OPS_SYMS,
    BOOL_OPS_SYMS,
    CMP_OPS_SYMS,
    LOCAL_TAG,
    MATHOPS,
    REDUCTIONS,
    UNARY_OPS_SYMS,
    BinOp,
    Constant,
    FuncNode,
    Op,
    Term,
    UnaryOp,
    is_term,
)
from pandas.core.computation.parsing import (
    clean_backtick_quoted_toks,
    tokenize_string,
)
from pandas.core.computation.scope import Scope

from pandas.io.formats import printing


def _rewrite_assign(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Rewrite the assignment operator for PyTables expressions that use ``=``
    as a substitute for ``==``.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    return toknum, "==" if tokval == "=" else tokval


def _replace_booleans(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Replace ``&`` with ``and`` and ``|`` with ``or`` so that bitwise
    precedence is changed to boolean precedence.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    if toknum == tokenize.OP:
        if tokval == "&":
            return tokenize.NAME, "and"
        elif tokval == "|":
            return tokenize.NAME, "or"
        return toknum, tokval
    return toknum, tokval


def _replace_locals(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Replace local variables with a syntactically valid name.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values

    Notes
    -----
    This is somewhat of a hack in that we rewrite a string such as ``'@a'`` as
    ``'__pd_eval_local_a'`` by telling the tokenizer that ``__pd_eval_local_``
    is a ``tokenize.OP`` and to replace the ``'@'`` symbol with it.
    """
    toknum, tokval = tok
    if toknum == tokenize.OP and tokval == "@":
        return tokenize.OP, LOCAL_TAG
    return toknum, tokval


def _compose2(f, g):
    """
    Compose 2 callables.
    """
    return lambda *args, **kwargs: f(g(*args, **kwargs))


def _compose(*funcs):
    """
    Compose 2 or more callables.
    """
    assert len(funcs) > 1, "At least 2 callables must be passed to compose"
    return reduce(_compose2, funcs)


def _preparse(
    source: str,
    f=_compose(
        _replace_locals, _replace_booleans, _rewrite_assign, clean_backtick_quoted_toks
    ),
) -> str:
    """
    Compose a collection of tokenization functions.

    Parameters
    ----------
    source : str
        A Python source code string
    f : callable
        This takes a tuple of (toknum, tokval) as its argument and returns a
        tuple with the same structure but possibly different elements. Defaults
        to the composition of ``_rewrite_assign``, ``_replace_booleans``, and
        ``_replace_locals``.

    Returns
    -------
    str
        Valid Python source code

    Notes
    -----
    The `f` parameter can be any callable that takes *and* returns input of the
    form ``(toknum, tokval)``, where ``toknum`` is one of the constants from
    the ``tokenize`` module and ``tokval`` is a string.
    """
    assert callable(f), "f must be callable"
    return tokenize.untokenize(f(x) for x in tokenize_string(source))


def _is_type(t):
    """
    Factory for a type checking function of type ``t`` or tuple of types.
    """
    return lambda x: isinstance(x.value, t)


_is_list = _is_type(list)
_is_str = _is_type(str)


# partition all AST nodes
_all_nodes = frozenset(
    node
    for node in (getattr(ast, name) for name in dir(ast))
    if isinstance(node, type) and issubclass(node, ast.AST)
)


def _filter_nodes(superclass, all_nodes=_all_nodes):
    """
    Filter out AST nodes that are subclasses of ``superclass``.
    """
    node_names = (node.__name__ for node in all_nodes if issubclass(node, superclass))
    return frozenset(node_names)


_all_node_names = frozenset(x.__name__ for x in _all_nodes)
_mod_nodes = _filter_nodes(ast.mod)
_stmt_nodes = _filter_nodes(ast.stmt)
_expr_nodes = _filter_nodes(ast.expr)
_expr_context_nodes = _filter_nodes(ast.expr_context)
_boolop_nodes = _filter_nodes(ast.boolop)
_operator_nodes = _filter_nodes(ast.operator)
_unary_op_nodes = _filter_nodes(ast.unaryop)
_cmp_op_nodes = _filter_nodes(ast.cmpop)
_comprehension_nodes = _filter_nodes(ast.comprehension)
_handler_nodes = _filter_nodes(ast.excepthandler)
_arguments_nodes = _filter_nodes(ast.arguments)
_keyword_nodes = _filter_nodes(ast.keyword)
_alias_nodes = _filter_nodes(ast.alias)


# nodes that we don't support directly but are needed for parsing
_hacked_nodes = frozenset(["Assign", "Module", "Expr"])


_unsupported_expr_nodes = frozenset(
    [
        "Yield",
        "GeneratorExp",
        "IfExp",
        "DictComp",
        "SetComp",
        "Repr",
        "Lambda",
        "Set",
        "AST",
        "Is",
        "IsNot",
    ]
)

# these nodes are low priority or won't ever be supported (e.g., AST)
_unsupported_nodes = (
    _stmt_nodes
    | _mod_nodes
    | _handler_nodes
    | _arguments_nodes
    | _keyword_nodes
    | _alias_nodes
    | _expr_context_nodes
    | _unsupported_expr_nodes
) - _hacked_nodes

# we're adding a different assignment in some cases to be equality comparison
# and we don't want `stmt` and friends in their so get only the class whose
# names are capitalized
_base_supported_nodes = (_all_node_names - _unsupported_nodes) | _hacked_nodes
intersection = _unsupported_nodes & _base_supported_nodes
_msg = f"cannot both support and not support {intersection}"
assert not intersection, _msg


def _node_not_implemented(node_name: str) -> Callable[..., None]:
    """
    Return a function that raises a NotImplementedError with a passed node name.
    """

    def f(self, *args, **kwargs):
        raise NotImplementedError(f"'{node_name}' nodes are not implemented")

    return f


# should be bound by BaseExprVisitor but that creates a circular dependency:
# _T is used in disallow, but disallow is used to define BaseExprVisitor
# https://github.com/microsoft/pyright/issues/2315
_T = TypeVar("_T")


def disallow(nodes: set[str]) -> Callable[[type[_T]], type[_T]]:
    """
    Decorator to disallow certain nodes from parsing. Raises a
    NotImplementedError instead.

    Returns
    -------
    callable
    """

    def disallowed(cls: type[_T]) -> type[_T]:
        # error: "Type[_T]" has no attribute "unsupported_nodes"
        cls.unsupported_nodes = ()  # type: ignore[attr-defined]
        for node in nodes:
            new_method = _node_not_implemented(node)
            name = f"visit_{node}"
            # error: "Type[_T]" has no attribute "unsupported_nodes"
            cls.unsupported_nodes += (name,)  # type: ignore[attr-defined]
            setattr(cls, name, new_method)
        return cls

    return disallowed


def _op_maker(op_class, op_symbol):
    """
    Return a function to create an op class with its symbol already passed.

    Returns
    -------
    callable
    """

    def f(self, node, *args, **kwargs):
        """
        Return a partial function with an Op subclass with an operator already passed.

        Returns
        -------
        callable
        """
        return partial(op_class, op_symbol, *args, **kwargs)

    return f


_op_classes = {"binary": BinOp, "unary": UnaryOp}


def add_ops(op_classes):
    """
    Decorator to add default implementation of ops.
    """

    def f(cls):
        for op_attr_name, op_class in op_classes.items():
            ops = getattr(cls, f"{op_attr_name}_ops")
            ops_map = getattr(cls, f"{op_attr_name}_op_nodes_map")
            for op in ops:
                op_node = ops_map[op]
                if op_node is not None:
                    made_op = _op_maker(op_class, op)
                    setattr(cls, f"visit_{op_node}", made_op)
        return cls

    return f


@disallow(_unsupported_nodes)
@add_ops(_op_classes)
class BaseExprVisitor(ast.NodeVisitor):
    """
    Custom ast walker. Parsers of other engines should subclass this class
    if necessary.

    Parameters
    ----------
    env : Scope
    engine : str
    parser : str
    preparser : callable
    """

    const_type: ClassVar[type[Term]] = Constant
    term_type: ClassVar[type[Term]] = Term

    binary_ops = CMP_OPS_SYMS + BOOL_OPS_SYMS + ARITH_OPS_SYMS
    binary_op_nodes = (
        "Gt",
        "Lt",
        "GtE",
        "LtE",
        "Eq",
        "NotEq",
        "In",
        "NotIn",
        "BitAnd",
        "BitOr",
        "And",
        "Or",
        "Add",
        "Sub",
        "Mult",
        "Div",
        "Pow",
        "FloorDiv",
        "Mod",
    )
    binary_op_nodes_map = dict(zip(binary_ops, binary_op_nodes))

    unary_ops = UNARY_OPS_SYMS
    unary_op_nodes = "UAdd", "USub", "Invert", "Not"
    unary_op_nodes_map = dict(zip(unary_ops, unary_op_nodes))

    rewrite_map = {
        ast.Eq: ast.In,
        ast.NotEq: ast.NotIn,
        ast.In: ast.In,
        ast.NotIn: ast.NotIn,
    }

    unsupported_nodes: tuple[str, ...]

    def __init__(self, env, engine, parser, preparser=_preparse) -> None:
        self.env = env
        self.engine = engine
        self.parser = parser
        self.preparser = preparser
        self.assigner = None

    def visit(self, node, **kwargs):
        if isinstance(node, str):
            clean = self.preparser(node)
            try:
                node = ast.fix_missing_locations(ast.parse(clean))
            except SyntaxError as e:
                if any(iskeyword(x) for x in clean.split()):
                    e.msg = "Python keyword not valid identifier in numexpr query"
                raise e

        method = f"visit_{type(node).__name__}"
        visitor = getattr(self, method)
        return visitor(node, **kwargs)

    def visit_Module(self, node, **kwargs):
        if len(node.body) != 1:
            raise SyntaxError("only a single expression is allowed")
        expr = node.body[0]
        return self.visit(expr, **kwargs)

    def visit_Expr(self, node, **kwargs):
        return self.visit(node.value, **kwargs)

    def _rewrite_membership_op(self, node, left, right):
        # the kind of the operator (is actually an instance)
        op_instance = node.op
        op_type = type(op_instance)

        # must be two terms and the comparison operator must be ==/!=/in/not in
        if is_term(left) and is_term(right) and op_type in self.rewrite_map:
            left_list, right_list = map(_is_list, (left, right))
            left_str, right_str = map(_is_str, (left, right))

            # if there are any strings or lists in the expression
            if left_list or right_list or left_str or right_str:
                op_instance = self.rewrite_map[op_type]()

            # pop the string variable out of locals and replace it with a list
            # of one string, kind of a hack
            if right_str:
                name = self.env.add_tmp([right.value])
                right = self.term_type(name, self.env)

            if left_str:
                name = self.env.add_tmp([left.value])
                left = self.term_type(name, self.env)

        op = self.visit(op_instance)
        return op, op_instance, left, right

    def _maybe_transform_eq_ne(self, node, left=None, right=None):
        if left is None:
            left = self.visit(node.left, side="left")
        if right is None:
            right = self.visit(node.right, side="right")
        op, op_class, left, right = self._rewrite_membership_op(node, left, right)
        return op, op_class, left, right

    def _maybe_downcast_constants(self, left, right):
        f32 = np.dtype(np.float32)
        if (
            left.is_scalar
            and hasattr(left, "value")
            and not right.is_scalar
            and right.return_type == f32
        ):
            # right is a float32 array, left is a scalar
            name = self.env.add_tmp(np.float32(left.value))
            left = self.term_type(name, self.env)
        if (
            right.is_scalar
            and hasattr(right, "value")
            and not left.is_scalar
            and left.return_type == f32
        ):
            # left is a float32 array, right is a scalar
            name = self.env.add_tmp(np.float32(right.value))
            right = self.term_type(name, self.env)

        return left, right

    def _maybe_eval(self, binop, eval_in_python):
        # eval `in` and `not in` (for now) in "partial" python space
        # things that can be evaluated in "eval" space will be turned into
        # temporary variables. for example,
        # [1,2] in a + 2 * b
        # in that case a + 2 * b will be evaluated using numexpr, and the "in"
        # call will be evaluated using isin (in python space)
        return binop.evaluate(
            self.env, self.engine, self.parser, self.term_type, eval_in_python
        )

    def _maybe_evaluate_binop(
        self,
        op,
        op_class,
        lhs,
        rhs,
        eval_in_python=("in", "not in"),
        maybe_eval_in_python=("==", "!=", "<", ">", "<=", ">="),
    ):
        res = op(lhs, rhs)

        if res.has_invalid_return_type:
            raise TypeError(
                f"unsupported operand type(s) for {res.op}: "
                f"'{lhs.type}' and '{rhs.type}'"
            )

        if self.engine != "pytables" and (
            res.op in CMP_OPS_SYMS
            and getattr(lhs, "is_datetime", False)
            or getattr(rhs, "is_datetime", False)
        ):
            # all date ops must be done in python bc numexpr doesn't work
            # well with NaT
            return self._maybe_eval(res, self.binary_ops)

        if res.op in eval_in_python:
            # "in"/"not in" ops are always evaluated in python
            return self._maybe_eval(res, eval_in_python)
        elif self.engine != "pytables":
            if (
                getattr(lhs, "return_type", None) == object
                or is_string_dtype(getattr(lhs, "return_type", None))
                or getattr(rhs, "return_type", None) == object
                or is_string_dtype(getattr(rhs, "return_type", None))
            ):
                # evaluate "==" and "!=" in python if either of our operands
                # has an object or string return type
                return self._maybe_eval(res, eval_in_python + maybe_eval_in_python)
        return res

    def visit_BinOp(self, node, **kwargs):
        op, op_class, left, right = self._maybe_transform_eq_ne(node)
        left, right = self._maybe_downcast_constants(left, right)
        return self._maybe_evaluate_binop(op, op_class, left, right)

    def visit_UnaryOp(self, node, **kwargs):
        op = self.visit(node.op)
        operand = self.visit(node.operand)
        return op(operand)

    def visit_Name(self, node, **kwargs) -> Term:
        return self.term_type(node.id, self.env, **kwargs)

    # TODO(py314): deprecated since Python 3.8. Remove after Python 3.14 is min
    def visit_NameConstant(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    # TODO(py314): deprecated since Python 3.8. Remove after Python 3.14 is min
    def visit_Num(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    def visit_Constant(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    # TODO(py314): deprecated since Python 3.8. Remove after Python 3.14 is min
    def visit_Str(self, node, **kwargs) -> Term:
        name = self.env.add_tmp(node.s)
        return self.term_type(name, self.env)

    def visit_List(self, node, **kwargs) -> Term:
        name = self.env.add_tmp([self.visit(e)(self.env) for e in node.elts])
        return self.term_type(name, self.env)

    visit_Tuple = visit_List

    def visit_Index(self, node, **kwargs):
        """df.index[4]"""
        return self.visit(node.value)

    def visit_Subscript(self, node, **kwargs) -> Term:
        from pandas import eval as pd_eval

        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        result = pd_eval(
            slobj, local_dict=self.env, engine=self.engine, parser=self.parser
        )
        try:
            # a Term instance
            v = value.value[result]
        except AttributeError:
            # an Op instance
            lhs = pd_eval(
                value, local_dict=self.env, engine=self.engine, parser=self.parser
            )
            v = lhs[result]
        name = self.env.add_tmp(v)
        return self.term_type(name, env=self.env)

    def visit_Slice(self, node, **kwargs) -> slice:
        """df.index[slice(4,6)]"""
        lower = node.lower
        if lower is not None:
            lower = self.visit(lower).value
        upper = node.upper
        if upper is not None:
            upper = self.visit(upper).value
        step = node.step
        if step is not None:
            step = self.visit(step).value

        return slice(lower, upper, step)

    def visit_Assign(self, node, **kwargs):
        """
        support a single assignment node, like

        c = a + b

        set the assigner at the top level, must be a Name node which
        might or might not exist in the resolvers

        """
        if len(node.targets) != 1:
            raise SyntaxError("can only assign a single expression")
        if not isinstance(node.targets[0], ast.Name):
            raise SyntaxError("left hand side of an assignment must be a single name")
        if self.env.target is None:
            raise ValueError("cannot assign without a target object")

        try:
            assigner = self.visit(node.targets[0], **kwargs)
        except UndefinedVariableError:
            assigner = node.targets[0].id

        self.assigner = getattr(assigner, "name", assigner)
        if self.assigner is None:
            raise SyntaxError(
                "left hand side of an assignment must be a single resolvable name"
            )

        return self.visit(node.value, **kwargs)

    def visit_Attribute(self, node, **kwargs):
        attr = node.attr
        value = node.value

        ctx = node.ctx
        if isinstance(ctx, ast.Load):
            # resolve the value
            resolved = self.visit(value).value
            try:
                v = getattr(resolved, attr)
                name = self.env.add_tmp(v)
                return self.term_type(name, self.env)
            except AttributeError:
                # something like datetime.datetime where scope is overridden
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved
                raise

        raise ValueError(f"Invalid Attribute context {type(ctx).__name__}")

    def visit_Call(self, node, side=None, **kwargs):
        if isinstance(node.func, ast.Attribute) and node.func.attr != "__call__":
            res = self.visit_Attribute(node.func)
        elif not isinstance(node.func, ast.Name):
            raise TypeError("Only named functions are supported")
        else:
            try:
                res = self.visit(node.func)
            except UndefinedVariableError:
                # Check if this is a supported function name
                try:
                    res = FuncNode(node.func.id)
                except ValueError:
                    # Raise original error
                    raise

        if res is None:
            # error: "expr" has no attribute "id"
            raise ValueError(
                f"Invalid function call {node.func.id}"  # type: ignore[attr-defined]
            )
        if hasattr(res, "value"):
            res = res.value

        if isinstance(res, FuncNode):
            new_args = [self.visit(arg) for arg in node.args]

            if node.keywords:
                raise TypeError(
                    f'Function "{res.name}" does not support keyword arguments'
                )

            return res(*new_args)

        else:
            new_args = [self.visit(arg)(self.env) for arg in node.args]

            for key in node.keywords:
                if not isinstance(key, ast.keyword):
                    # error: "expr" has no attribute "id"
                    raise ValueError(
                        "keyword error in function call "
                        f"'{node.func.id}'"  # type: ignore[attr-defined]
                    )

                if key.arg:
                    kwargs[key.arg] = self.visit(key.value)(self.env)

            name = self.env.add_tmp(res(*new_args, **kwargs))
            return self.term_type(name=name, env=self.env)

    def translate_In(self, op):
        return op

    def visit_Compare(self, node, **kwargs):
        ops = node.ops
        comps = node.comparators

        # base case: we have something like a CMP b
        if len(comps) == 1:
            op = self.translate_In(ops[0])
            binop = ast.BinOp(op=op, left=node.left, right=comps[0])
            return self.visit(binop)

        # recursive case: we have a chained comparison, a CMP b CMP c, etc.
        left = node.left
        values = []
        for op, comp in zip(ops, comps):
            new_node = self.visit(
                ast.Compare(comparators=[comp], left=left, ops=[self.translate_In(op)])
            )
            left = comp
            values.append(new_node)
        return self.visit(ast.BoolOp(op=ast.And(), values=values))

    def _try_visit_binop(self, bop):
        if isinstance(bop, (Op, Term)):
            return bop
        return self.visit(bop)

    def visit_BoolOp(self, node, **kwargs):
        def visitor(x, y):
            lhs = self._try_visit_binop(x)
            rhs = self._try_visit_binop(y)

            op, op_class, lhs, rhs = self._maybe_transform_eq_ne(node, lhs, rhs)
            return self._maybe_evaluate_binop(op, node.op, lhs, rhs)

        operands = node.values
        return reduce(visitor, operands)


_python_not_supported = frozenset(["Dict", "BoolOp", "In", "NotIn"])
_numexpr_supported_calls = frozenset(REDUCTIONS + MATHOPS)


@disallow(
    (_unsupported_nodes | _python_not_supported)
    - (_boolop_nodes | frozenset(["BoolOp", "Attribute", "In", "NotIn", "Tuple"]))
)
class PandasExprVisitor(BaseExprVisitor):
    def __init__(
        self,
        env,
        engine,
        parser,
        preparser=partial(
            _preparse,
            f=_compose(_replace_locals, _replace_booleans, clean_backtick_quoted_toks),
        ),
    ) -> None:
        super().__init__(env, engine, parser, preparser)


@disallow(_unsupported_nodes | _python_not_supported | frozenset(["Not"]))
class PythonExprVisitor(BaseExprVisitor):
    def __init__(
        self, env, engine, parser, preparser=lambda source, f=None: source
    ) -> None:
        super().__init__(env, engine, parser, preparser=preparser)


class Expr:
    """
    Object encapsulating an expression.

    Parameters
    ----------
    expr : str
    engine : str, optional, default 'numexpr'
    parser : str, optional, default 'pandas'
    env : Scope, optional, default None
    level : int, optional, default 2
    """

    env: Scope
    engine: str
    parser: str

    def __init__(
        self,
        expr,
        engine: str = "numexpr",
        parser: str = "pandas",
        env: Scope | None = None,
        level: int = 0,
    ) -> None:
        self.expr = expr
        self.env = env or Scope(level=level + 1)
        self.engine = engine
        self.parser = parser
        self._visitor = PARSERS[parser](self.env, self.engine, self.parser)
        self.terms = self.parse()

    @property
    def assigner(self):
        return getattr(self._visitor, "assigner", None)

    def __call__(self):
        return self.terms(self.env)

    def __repr__(self) -> str:
        return printing.pprint_thing(self.terms)

    def __len__(self) -> int:
        return len(self.expr)

    def parse(self):
        """
        Parse an expression.
        """
        return self._visitor.visit(self.expr)

    @property
    def names(self):
        """
        Get the names in an expression.
        """
        if is_term(self.terms):
            return frozenset([self.terms.name])
        return frozenset(term.name for term in com.flatten(self.terms))


PARSERS = {"python": PythonExprVisitor, "pandas": PandasExprVisitor}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\_termui_impl.py ===
"""
This module contains implementations for the termui module. To keep the
import time of Click down, some infrequently used functionality is
placed in this module and only imported as needed.
"""

from __future__ import annotations

import collections.abc as cabc
import contextlib
import math
import os
import shlex
import sys
import time
import typing as t
from gettext import gettext as _
from io import StringIO
from pathlib import Path
from shutil import which
from types import TracebackType

from ._compat import _default_text_stdout
from ._compat import CYGWIN
from ._compat import get_best_encoding
from ._compat import isatty
from ._compat import open_stream
from ._compat import strip_ansi
from ._compat import term_len
from ._compat import WIN
from .exceptions import ClickException
from .utils import echo

V = t.TypeVar("V")

if os.name == "nt":
    BEFORE_BAR = "\r"
    AFTER_BAR = "\n"
else:
    BEFORE_BAR = "\r\033[?25l"
    AFTER_BAR = "\033[?25h\n"


class ProgressBar(t.Generic[V]):
    def __init__(
        self,
        iterable: cabc.Iterable[V] | None,
        length: int | None = None,
        fill_char: str = "#",
        empty_char: str = " ",
        bar_template: str = "%(bar)s",
        info_sep: str = "  ",
        hidden: bool = False,
        show_eta: bool = True,
        show_percent: bool | None = None,
        show_pos: bool = False,
        item_show_func: t.Callable[[V | None], str | None] | None = None,
        label: str | None = None,
        file: t.TextIO | None = None,
        color: bool | None = None,
        update_min_steps: int = 1,
        width: int = 30,
    ) -> None:
        self.fill_char = fill_char
        self.empty_char = empty_char
        self.bar_template = bar_template
        self.info_sep = info_sep
        self.hidden = hidden
        self.show_eta = show_eta
        self.show_percent = show_percent
        self.show_pos = show_pos
        self.item_show_func = item_show_func
        self.label: str = label or ""

        if file is None:
            file = _default_text_stdout()

            # There are no standard streams attached to write to. For example,
            # pythonw on Windows.
            if file is None:
                file = StringIO()

        self.file = file
        self.color = color
        self.update_min_steps = update_min_steps
        self._completed_intervals = 0
        self.width: int = width
        self.autowidth: bool = width == 0

        if length is None:
            from operator import length_hint

            length = length_hint(iterable, -1)

            if length == -1:
                length = None
        if iterable is None:
            if length is None:
                raise TypeError("iterable or length is required")
            iterable = t.cast("cabc.Iterable[V]", range(length))
        self.iter: cabc.Iterable[V] = iter(iterable)
        self.length = length
        self.pos: int = 0
        self.avg: list[float] = []
        self.last_eta: float
        self.start: float
        self.start = self.last_eta = time.time()
        self.eta_known: bool = False
        self.finished: bool = False
        self.max_width: int | None = None
        self.entered: bool = False
        self.current_item: V | None = None
        self._is_atty = isatty(self.file)
        self._last_line: str | None = None

    def __enter__(self) -> ProgressBar[V]:
        self.entered = True
        self.render_progress()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.render_finish()

    def __iter__(self) -> cabc.Iterator[V]:
        if not self.entered:
            raise RuntimeError("You need to use progress bars in a with block.")
        self.render_progress()
        return self.generator()

    def __next__(self) -> V:
        # Iteration is defined in terms of a generator function,
        # returned by iter(self); use that to define next(). This works
        # because `self.iter` is an iterable consumed by that generator,
        # so it is re-entry safe. Calling `next(self.generator())`
        # twice works and does "what you want".
        return next(iter(self))

    def render_finish(self) -> None:
        if self.hidden or not self._is_atty:
            return
        self.file.write(AFTER_BAR)
        self.file.flush()

    @property
    def pct(self) -> float:
        if self.finished:
            return 1.0
        return min(self.pos / (float(self.length or 1) or 1), 1.0)

    @property
    def time_per_iteration(self) -> float:
        if not self.avg:
            return 0.0
        return sum(self.avg) / float(len(self.avg))

    @property
    def eta(self) -> float:
        if self.length is not None and not self.finished:
            return self.time_per_iteration * (self.length - self.pos)
        return 0.0

    def format_eta(self) -> str:
        if self.eta_known:
            t = int(self.eta)
            seconds = t % 60
            t //= 60
            minutes = t % 60
            t //= 60
            hours = t % 24
            t //= 24
            if t > 0:
                return f"{t}d {hours:02}:{minutes:02}:{seconds:02}"
            else:
                return f"{hours:02}:{minutes:02}:{seconds:02}"
        return ""

    def format_pos(self) -> str:
        pos = str(self.pos)
        if self.length is not None:
            pos += f"/{self.length}"
        return pos

    def format_pct(self) -> str:
        return f"{int(self.pct * 100): 4}%"[1:]

    def format_bar(self) -> str:
        if self.length is not None:
            bar_length = int(self.pct * self.width)
            bar = self.fill_char * bar_length
            bar += self.empty_char * (self.width - bar_length)
        elif self.finished:
            bar = self.fill_char * self.width
        else:
            chars = list(self.empty_char * (self.width or 1))
            if self.time_per_iteration != 0:
                chars[
                    int(
                        (math.cos(self.pos * self.time_per_iteration) / 2.0 + 0.5)
                        * self.width
                    )
                ] = self.fill_char
            bar = "".join(chars)
        return bar

    def format_progress_line(self) -> str:
        show_percent = self.show_percent

        info_bits = []
        if self.length is not None and show_percent is None:
            show_percent = not self.show_pos

        if self.show_pos:
            info_bits.append(self.format_pos())
        if show_percent:
            info_bits.append(self.format_pct())
        if self.show_eta and self.eta_known and not self.finished:
            info_bits.append(self.format_eta())
        if self.item_show_func is not None:
            item_info = self.item_show_func(self.current_item)
            if item_info is not None:
                info_bits.append(item_info)

        return (
            self.bar_template
            % {
                "label": self.label,
                "bar": self.format_bar(),
                "info": self.info_sep.join(info_bits),
            }
        ).rstrip()

    def render_progress(self) -> None:
        import shutil

        if self.hidden:
            return

        if not self._is_atty:
            # Only output the label once if the output is not a TTY.
            if self._last_line != self.label:
                self._last_line = self.label
                echo(self.label, file=self.file, color=self.color)
            return

        buf = []
        # Update width in case the terminal has been resized
        if self.autowidth:
            old_width = self.width
            self.width = 0
            clutter_length = term_len(self.format_progress_line())
            new_width = max(0, shutil.get_terminal_size().columns - clutter_length)
            if new_width < old_width and self.max_width is not None:
                buf.append(BEFORE_BAR)
                buf.append(" " * self.max_width)
                self.max_width = new_width
            self.width = new_width

        clear_width = self.width
        if self.max_width is not None:
            clear_width = self.max_width

        buf.append(BEFORE_BAR)
        line = self.format_progress_line()
        line_len = term_len(line)
        if self.max_width is None or self.max_width < line_len:
            self.max_width = line_len

        buf.append(line)
        buf.append(" " * (clear_width - line_len))
        line = "".join(buf)
        # Render the line only if it changed.

        if line != self._last_line:
            self._last_line = line
            echo(line, file=self.file, color=self.color, nl=False)
            self.file.flush()

    def make_step(self, n_steps: int) -> None:
        self.pos += n_steps
        if self.length is not None and self.pos >= self.length:
            self.finished = True

        if (time.time() - self.last_eta) < 1.0:
            return

        self.last_eta = time.time()

        # self.avg is a rolling list of length <= 7 of steps where steps are
        # defined as time elapsed divided by the total progress through
        # self.length.
        if self.pos:
            step = (time.time() - self.start) / self.pos
        else:
            step = time.time() - self.start

        self.avg = self.avg[-6:] + [step]

        self.eta_known = self.length is not None

    def update(self, n_steps: int, current_item: V | None = None) -> None:
        """Update the progress bar by advancing a specified number of
        steps, and optionally set the ``current_item`` for this new
        position.

        :param n_steps: Number of steps to advance.
        :param current_item: Optional item to set as ``current_item``
            for the updated position.

        .. versionchanged:: 8.0
            Added the ``current_item`` optional parameter.

        .. versionchanged:: 8.0
            Only render when the number of steps meets the
            ``update_min_steps`` threshold.
        """
        if current_item is not None:
            self.current_item = current_item

        self._completed_intervals += n_steps

        if self._completed_intervals >= self.update_min_steps:
            self.make_step(self._completed_intervals)
            self.render_progress()
            self._completed_intervals = 0

    def finish(self) -> None:
        self.eta_known = False
        self.current_item = None
        self.finished = True

    def generator(self) -> cabc.Iterator[V]:
        """Return a generator which yields the items added to the bar
        during construction, and updates the progress bar *after* the
        yielded block returns.
        """
        # WARNING: the iterator interface for `ProgressBar` relies on
        # this and only works because this is a simple generator which
        # doesn't create or manage additional state. If this function
        # changes, the impact should be evaluated both against
        # `iter(bar)` and `next(bar)`. `next()` in particular may call
        # `self.generator()` repeatedly, and this must remain safe in
        # order for that interface to work.
        if not self.entered:
            raise RuntimeError("You need to use progress bars in a with block.")

        if not self._is_atty:
            yield from self.iter
        else:
            for rv in self.iter:
                self.current_item = rv

                # This allows show_item_func to be updated before the
                # item is processed. Only trigger at the beginning of
                # the update interval.
                if self._completed_intervals == 0:
                    self.render_progress()

                yield rv
                self.update(1)

            self.finish()
            self.render_progress()


def pager(generator: cabc.Iterable[str], color: bool | None = None) -> None:
    """Decide what method to use for paging through text."""
    stdout = _default_text_stdout()

    # There are no standard streams attached to write to. For example,
    # pythonw on Windows.
    if stdout is None:
        stdout = StringIO()

    if not isatty(sys.stdin) or not isatty(stdout):
        return _nullpager(stdout, generator, color)

    # Split and normalize the pager command into parts.
    pager_cmd_parts = shlex.split(os.environ.get("PAGER", ""), posix=False)
    if pager_cmd_parts:
        if WIN:
            if _tempfilepager(generator, pager_cmd_parts, color):
                return
        elif _pipepager(generator, pager_cmd_parts, color):
            return

    if os.environ.get("TERM") in ("dumb", "emacs"):
        return _nullpager(stdout, generator, color)
    if (WIN or sys.platform.startswith("os2")) and _tempfilepager(
        generator, ["more"], color
    ):
        return
    if _pipepager(generator, ["less"], color):
        return

    import tempfile

    fd, filename = tempfile.mkstemp()
    os.close(fd)
    try:
        if _pipepager(generator, ["more"], color):
            return
        return _nullpager(stdout, generator, color)
    finally:
        os.unlink(filename)


def _pipepager(
    generator: cabc.Iterable[str], cmd_parts: list[str], color: bool | None
) -> bool:
    """Page through text by feeding it to another program. Invoking a
    pager through this might support colors.

    Returns `True` if the command was found, `False` otherwise and thus another
    pager should be attempted.
    """
    # Split the command into the invoked CLI and its parameters.
    if not cmd_parts:
        return False
    cmd = cmd_parts[0]
    cmd_params = cmd_parts[1:]

    cmd_filepath = which(cmd)
    if not cmd_filepath:
        return False
    # Resolves symlinks and produces a normalized absolute path string.
    cmd_path = Path(cmd_filepath).resolve()
    cmd_name = cmd_path.name

    import subprocess

    # Make a local copy of the environment to not affect the global one.
    env = dict(os.environ)

    # If we're piping to less and the user hasn't decided on colors, we enable
    # them by default we find the -R flag in the command line arguments.
    if color is None and cmd_name == "less":
        less_flags = f"{os.environ.get('LESS', '')}{' '.join(cmd_params)}"
        if not less_flags:
            env["LESS"] = "-R"
            color = True
        elif "r" in less_flags or "R" in less_flags:
            color = True

    c = subprocess.Popen(
        [str(cmd_path)] + cmd_params,
        shell=True,
        stdin=subprocess.PIPE,
        env=env,
        errors="replace",
        text=True,
    )
    assert c.stdin is not None
    try:
        for text in generator:
            if not color:
                text = strip_ansi(text)

            c.stdin.write(text)
    except BrokenPipeError:
        # In case the pager exited unexpectedly, ignore the broken pipe error.
        pass
    except Exception as e:
        # In case there is an exception we want to close the pager immediately
        # and let the caller handle it.
        # Otherwise the pager will keep running, and the user may not notice
        # the error message, or worse yet it may leave the terminal in a broken state.
        c.terminate()
        raise e
    finally:
        # We must close stdin and wait for the pager to exit before we continue
        try:
            c.stdin.close()
        # Close implies flush, so it might throw a BrokenPipeError if the pager
        # process exited already.
        except BrokenPipeError:
            pass

        # Less doesn't respect ^C, but catches it for its own UI purposes (aborting
        # search or other commands inside less).
        #
        # That means when the user hits ^C, the parent process (click) terminates,
        # but less is still alive, paging the output and messing up the terminal.
        #
        # If the user wants to make the pager exit on ^C, they should set
        # `LESS='-K'`. It's not our decision to make.
        while True:
            try:
                c.wait()
            except KeyboardInterrupt:
                pass
            else:
                break

    return True


def _tempfilepager(
    generator: cabc.Iterable[str], cmd_parts: list[str], color: bool | None
) -> bool:
    """Page through text by invoking a program on a temporary file.

    Returns `True` if the command was found, `False` otherwise and thus another
    pager should be attempted.
    """
    # Split the command into the invoked CLI and its parameters.
    if not cmd_parts:
        return False
    cmd = cmd_parts[0]

    cmd_filepath = which(cmd)
    if not cmd_filepath:
        return False
    # Resolves symlinks and produces a normalized absolute path string.
    cmd_path = Path(cmd_filepath).resolve()

    import subprocess
    import tempfile

    fd, filename = tempfile.mkstemp()
    # TODO: This never terminates if the passed generator never terminates.
    text = "".join(generator)
    if not color:
        text = strip_ansi(text)
    encoding = get_best_encoding(sys.stdout)
    with open_stream(filename, "wb")[0] as f:
        f.write(text.encode(encoding))
    try:
        subprocess.call([str(cmd_path), filename])
    except OSError:
        # Command not found
        pass
    finally:
        os.close(fd)
        os.unlink(filename)

    return True


def _nullpager(
    stream: t.TextIO, generator: cabc.Iterable[str], color: bool | None
) -> None:
    """Simply print unformatted text.  This is the ultimate fallback."""
    for text in generator:
        if not color:
            text = strip_ansi(text)
        stream.write(text)


class Editor:
    def __init__(
        self,
        editor: str | None = None,
        env: cabc.Mapping[str, str] | None = None,
        require_save: bool = True,
        extension: str = ".txt",
    ) -> None:
        self.editor = editor
        self.env = env
        self.require_save = require_save
        self.extension = extension

    def get_editor(self) -> str:
        if self.editor is not None:
            return self.editor
        for key in "VISUAL", "EDITOR":
            rv = os.environ.get(key)
            if rv:
                return rv
        if WIN:
            return "notepad"
        for editor in "sensible-editor", "vim", "nano":
            if which(editor) is not None:
                return editor
        return "vi"

    def edit_files(self, filenames: cabc.Iterable[str]) -> None:
        import subprocess

        editor = self.get_editor()
        environ: dict[str, str] | None = None

        if self.env:
            environ = os.environ.copy()
            environ.update(self.env)

        exc_filename = " ".join(f'"{filename}"' for filename in filenames)

        try:
            c = subprocess.Popen(
                args=f"{editor} {exc_filename}", env=environ, shell=True
            )
            exit_code = c.wait()
            if exit_code != 0:
                raise ClickException(
                    _("{editor}: Editing failed").format(editor=editor)
                )
        except OSError as e:
            raise ClickException(
                _("{editor}: Editing failed: {e}").format(editor=editor, e=e)
            ) from e

    @t.overload
    def edit(self, text: bytes | bytearray) -> bytes | None: ...

    # We cannot know whether or not the type expected is str or bytes when None
    # is passed, so str is returned as that was what was done before.
    @t.overload
    def edit(self, text: str | None) -> str | None: ...

    def edit(self, text: str | bytes | bytearray | None) -> str | bytes | None:
        import tempfile

        if text is None:
            data = b""
        elif isinstance(text, (bytes, bytearray)):
            data = text
        else:
            if text and not text.endswith("\n"):
                text += "\n"

            if WIN:
                data = text.replace("\n", "\r\n").encode("utf-8-sig")
            else:
                data = text.encode("utf-8")

        fd, name = tempfile.mkstemp(prefix="editor-", suffix=self.extension)
        f: t.BinaryIO

        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)

            # If the filesystem resolution is 1 second, like Mac OS
            # 10.12 Extended, or 2 seconds, like FAT32, and the editor
            # closes very fast, require_save can fail. Set the modified
            # time to be 2 seconds in the past to work around this.
            os.utime(name, (os.path.getatime(name), os.path.getmtime(name) - 2))
            # Depending on the resolution, the exact value might not be
            # recorded, so get the new recorded value.
            timestamp = os.path.getmtime(name)

            self.edit_files((name,))

            if self.require_save and os.path.getmtime(name) == timestamp:
                return None

            with open(name, "rb") as f:
                rv = f.read()

            if isinstance(text, (bytes, bytearray)):
                return rv

            return rv.decode("utf-8-sig").replace("\r\n", "\n")
        finally:
            os.unlink(name)


def open_url(url: str, wait: bool = False, locate: bool = False) -> int:
    import subprocess

    def _unquote_file(url: str) -> str:
        from urllib.parse import unquote

        if url.startswith("file://"):
            url = unquote(url[7:])

        return url

    if sys.platform == "darwin":
        args = ["open"]
        if wait:
            args.append("-W")
        if locate:
            args.append("-R")
        args.append(_unquote_file(url))
        null = open("/dev/null", "w")
        try:
            return subprocess.Popen(args, stderr=null).wait()
        finally:
            null.close()
    elif WIN:
        if locate:
            url = _unquote_file(url)
            args = ["explorer", f"/select,{url}"]
        else:
            args = ["start"]
            if wait:
                args.append("/WAIT")
            args.append("")
            args.append(url)
        try:
            return subprocess.call(args)
        except OSError:
            # Command not found
            return 127
    elif CYGWIN:
        if locate:
            url = _unquote_file(url)
            args = ["cygstart", os.path.dirname(url)]
        else:
            args = ["cygstart"]
            if wait:
                args.append("-w")
            args.append(url)
        try:
            return subprocess.call(args)
        except OSError:
            # Command not found
            return 127

    try:
        if locate:
            url = os.path.dirname(_unquote_file(url)) or "."
        else:
            url = _unquote_file(url)
        c = subprocess.Popen(["xdg-open", url])
        if wait:
            return c.wait()
        return 0
    except OSError:
        if url.startswith(("http://", "https://")) and not locate and not wait:
            import webbrowser

            webbrowser.open(url)
            return 0
        return 1


def _translate_ch_to_exc(ch: str) -> None:
    if ch == "\x03":
        raise KeyboardInterrupt()

    if ch == "\x04" and not WIN:  # Unix-like, Ctrl+D
        raise EOFError()

    if ch == "\x1a" and WIN:  # Windows, Ctrl+Z
        raise EOFError()

    return None


if sys.platform == "win32":
    import msvcrt

    @contextlib.contextmanager
    def raw_terminal() -> cabc.Iterator[int]:
        yield -1

    def getchar(echo: bool) -> str:
        # The function `getch` will return a bytes object corresponding to
        # the pressed character. Since Windows 10 build 1803, it will also
        # return \x00 when called a second time after pressing a regular key.
        #
        # `getwch` does not share this probably-bugged behavior. Moreover, it
        # returns a Unicode object by default, which is what we want.
        #
        # Either of these functions will return \x00 or \xe0 to indicate
        # a special key, and you need to call the same function again to get
        # the "rest" of the code. The fun part is that \u00e0 is
        # "latin small letter a with grave", so if you type that on a French
        # keyboard, you _also_ get a \xe0.
        # E.g., consider the Up arrow. This returns \xe0 and then \x48. The
        # resulting Unicode string reads as "a with grave" + "capital H".
        # This is indistinguishable from when the user actually types
        # "a with grave" and then "capital H".
        #
        # When \xe0 is returned, we assume it's part of a special-key sequence
        # and call `getwch` again, but that means that when the user types
        # the \u00e0 character, `getchar` doesn't return until a second
        # character is typed.
        # The alternative is returning immediately, but that would mess up
        # cross-platform handling of arrow keys and others that start with
        # \xe0. Another option is using `getch`, but then we can't reliably
        # read non-ASCII characters, because return values of `getch` are
        # limited to the current 8-bit codepage.
        #
        # Anyway, Click doesn't claim to do this Right(tm), and using `getwch`
        # is doing the right thing in more situations than with `getch`.

        if echo:
            func = t.cast(t.Callable[[], str], msvcrt.getwche)
        else:
            func = t.cast(t.Callable[[], str], msvcrt.getwch)

        rv = func()

        if rv in ("\x00", "\xe0"):
            # \x00 and \xe0 are control characters that indicate special key,
            # see above.
            rv += func()

        _translate_ch_to_exc(rv)
        return rv

else:
    import termios
    import tty

    @contextlib.contextmanager
    def raw_terminal() -> cabc.Iterator[int]:
        f: t.TextIO | None
        fd: int

        if not isatty(sys.stdin):
            f = open("/dev/tty")
            fd = f.fileno()
        else:
            fd = sys.stdin.fileno()
            f = None

        try:
            old_settings = termios.tcgetattr(fd)

            try:
                tty.setraw(fd)
                yield fd
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                sys.stdout.flush()

                if f is not None:
                    f.close()
        except termios.error:
            pass

    def getchar(echo: bool) -> str:
        with raw_terminal() as fd:
            ch = os.read(fd, 32).decode(get_best_encoding(sys.stdin), "replace")

            if echo and isatty(sys.stdout):
                sys.stdout.write(ch)

            _translate_ch_to_exc(ch)
            return ch

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\__init__.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: September 17, 2021
# URL: https://humanfriendly.readthedocs.io

"""The main module of the `humanfriendly` package."""

# Standard library modules.
import collections
import datetime
import decimal
import numbers
import os
import os.path
import re
import time

# Modules included in our package.
from humanfriendly.compat import is_string, monotonic
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format, pluralize, tokenize

# Public identifiers that require documentation.
__all__ = (
    'CombinedUnit',
    'InvalidDate',
    'InvalidLength',
    'InvalidSize',
    'InvalidTimespan',
    'SizeUnit',
    'Timer',
    '__version__',
    'coerce_boolean',
    'coerce_pattern',
    'coerce_seconds',
    'disk_size_units',
    'format_length',
    'format_number',
    'format_path',
    'format_size',
    'format_timespan',
    'length_size_units',
    'parse_date',
    'parse_length',
    'parse_path',
    'parse_size',
    'parse_timespan',
    'round_number',
    'time_units',
)

# Semi-standard module versioning.
__version__ = '10.0'

# Named tuples to define units of size.
SizeUnit = collections.namedtuple('SizeUnit', 'divider, symbol, name')
CombinedUnit = collections.namedtuple('CombinedUnit', 'decimal, binary')

# Common disk size units in binary (base-2) and decimal (base-10) multiples.
disk_size_units = (
    CombinedUnit(SizeUnit(1000**1, 'KB', 'kilobyte'), SizeUnit(1024**1, 'KiB', 'kibibyte')),
    CombinedUnit(SizeUnit(1000**2, 'MB', 'megabyte'), SizeUnit(1024**2, 'MiB', 'mebibyte')),
    CombinedUnit(SizeUnit(1000**3, 'GB', 'gigabyte'), SizeUnit(1024**3, 'GiB', 'gibibyte')),
    CombinedUnit(SizeUnit(1000**4, 'TB', 'terabyte'), SizeUnit(1024**4, 'TiB', 'tebibyte')),
    CombinedUnit(SizeUnit(1000**5, 'PB', 'petabyte'), SizeUnit(1024**5, 'PiB', 'pebibyte')),
    CombinedUnit(SizeUnit(1000**6, 'EB', 'exabyte'), SizeUnit(1024**6, 'EiB', 'exbibyte')),
    CombinedUnit(SizeUnit(1000**7, 'ZB', 'zettabyte'), SizeUnit(1024**7, 'ZiB', 'zebibyte')),
    CombinedUnit(SizeUnit(1000**8, 'YB', 'yottabyte'), SizeUnit(1024**8, 'YiB', 'yobibyte')),
)

# Common length size units, used for formatting and parsing.
length_size_units = (dict(prefix='nm', divider=1e-09, singular='nm', plural='nm'),
                     dict(prefix='mm', divider=1e-03, singular='mm', plural='mm'),
                     dict(prefix='cm', divider=1e-02, singular='cm', plural='cm'),
                     dict(prefix='m', divider=1, singular='metre', plural='metres'),
                     dict(prefix='km', divider=1000, singular='km', plural='km'))

# Common time units, used for formatting of time spans.
time_units = (dict(divider=1e-9, singular='nanosecond', plural='nanoseconds', abbreviations=['ns']),
              dict(divider=1e-6, singular='microsecond', plural='microseconds', abbreviations=['us']),
              dict(divider=1e-3, singular='millisecond', plural='milliseconds', abbreviations=['ms']),
              dict(divider=1, singular='second', plural='seconds', abbreviations=['s', 'sec', 'secs']),
              dict(divider=60, singular='minute', plural='minutes', abbreviations=['m', 'min', 'mins']),
              dict(divider=60 * 60, singular='hour', plural='hours', abbreviations=['h']),
              dict(divider=60 * 60 * 24, singular='day', plural='days', abbreviations=['d']),
              dict(divider=60 * 60 * 24 * 7, singular='week', plural='weeks', abbreviations=['w']),
              dict(divider=60 * 60 * 24 * 7 * 52, singular='year', plural='years', abbreviations=['y']))


def coerce_boolean(value):
    """
    Coerce any value to a boolean.

    :param value: Any Python value. If the value is a string:

                  - The strings '1', 'yes', 'true' and 'on' are coerced to :data:`True`.
                  - The strings '0', 'no', 'false' and 'off' are coerced to :data:`False`.
                  - Other strings raise an exception.

                  Other Python values are coerced using :class:`bool`.
    :returns: A proper boolean value.
    :raises: :exc:`exceptions.ValueError` when the value is a string but
             cannot be coerced with certainty.
    """
    if is_string(value):
        normalized = value.strip().lower()
        if normalized in ('1', 'yes', 'true', 'on'):
            return True
        elif normalized in ('0', 'no', 'false', 'off', ''):
            return False
        else:
            msg = "Failed to coerce string to boolean! (%r)"
            raise ValueError(format(msg, value))
    else:
        return bool(value)


def coerce_pattern(value, flags=0):
    """
    Coerce strings to compiled regular expressions.

    :param value: A string containing a regular expression pattern
                  or a compiled regular expression.
    :param flags: The flags used to compile the pattern (an integer).
    :returns: A compiled regular expression.
    :raises: :exc:`~exceptions.ValueError` when `value` isn't a string
             and also isn't a compiled regular expression.
    """
    if is_string(value):
        value = re.compile(value, flags)
    else:
        empty_pattern = re.compile('')
        pattern_type = type(empty_pattern)
        if not isinstance(value, pattern_type):
            msg = "Failed to coerce value to compiled regular expression! (%r)"
            raise ValueError(format(msg, value))
    return value


def coerce_seconds(value):
    """
    Coerce a value to the number of seconds.

    :param value: An :class:`int`, :class:`float` or
                  :class:`datetime.timedelta` object.
    :returns: An :class:`int` or :class:`float` value.

    When `value` is a :class:`datetime.timedelta` object the
    :meth:`~datetime.timedelta.total_seconds()` method is called.
    """
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if not isinstance(value, numbers.Number):
        msg = "Failed to coerce value to number of seconds! (%r)"
        raise ValueError(format(msg, value))
    return value


def format_size(num_bytes, keep_width=False, binary=False):
    """
    Format a byte count as a human readable file size.

    :param num_bytes: The size to format in bytes (an integer).
    :param keep_width: :data:`True` if trailing zeros should not be stripped,
                       :data:`False` if they can be stripped.
    :param binary: :data:`True` to use binary multiples of bytes (base-2),
                   :data:`False` to use decimal multiples of bytes (base-10).
    :returns: The corresponding human readable file size (a string).

    This function knows how to format sizes in bytes, kilobytes, megabytes,
    gigabytes, terabytes and petabytes. Some examples:

    >>> from humanfriendly import format_size
    >>> format_size(0)
    '0 bytes'
    >>> format_size(1)
    '1 byte'
    >>> format_size(5)
    '5 bytes'
    > format_size(1000)
    '1 KB'
    > format_size(1024, binary=True)
    '1 KiB'
    >>> format_size(1000 ** 3 * 4)
    '4 GB'
    """
    for unit in reversed(disk_size_units):
        if num_bytes >= unit.binary.divider and binary:
            number = round_number(float(num_bytes) / unit.binary.divider, keep_width=keep_width)
            return pluralize(number, unit.binary.symbol, unit.binary.symbol)
        elif num_bytes >= unit.decimal.divider and not binary:
            number = round_number(float(num_bytes) / unit.decimal.divider, keep_width=keep_width)
            return pluralize(number, unit.decimal.symbol, unit.decimal.symbol)
    return pluralize(num_bytes, 'byte')


def parse_size(size, binary=False):
    """
    Parse a human readable data size and return the number of bytes.

    :param size: The human readable file size to parse (a string).
    :param binary: :data:`True` to use binary multiples of bytes (base-2) for
                   ambiguous unit symbols and names, :data:`False` to use
                   decimal multiples of bytes (base-10).
    :returns: The corresponding size in bytes (an integer).
    :raises: :exc:`InvalidSize` when the input can't be parsed.

    This function knows how to parse sizes in bytes, kilobytes, megabytes,
    gigabytes, terabytes and petabytes. Some examples:

    >>> from humanfriendly import parse_size
    >>> parse_size('42')
    42
    >>> parse_size('13b')
    13
    >>> parse_size('5 bytes')
    5
    >>> parse_size('1 KB')
    1000
    >>> parse_size('1 kilobyte')
    1000
    >>> parse_size('1 KiB')
    1024
    >>> parse_size('1 KB', binary=True)
    1024
    >>> parse_size('1.5 GB')
    1500000000
    >>> parse_size('1.5 GB', binary=True)
    1610612736
    """
    tokens = tokenize(size)
    if tokens and isinstance(tokens[0], numbers.Number):
        # Get the normalized unit (if any) from the tokenized input.
        normalized_unit = tokens[1].lower() if len(tokens) == 2 and is_string(tokens[1]) else ''
        # If the input contains only a number, it's assumed to be the number of
        # bytes. The second token can also explicitly reference the unit bytes.
        if len(tokens) == 1 or normalized_unit.startswith('b'):
            return int(tokens[0])
        # Otherwise we expect two tokens: A number and a unit.
        if normalized_unit:
            # Convert plural units to singular units, for details:
            # https://github.com/xolox/python-humanfriendly/issues/26
            normalized_unit = normalized_unit.rstrip('s')
            for unit in disk_size_units:
                # First we check for unambiguous symbols (KiB, MiB, GiB, etc)
                # and names (kibibyte, mebibyte, gibibyte, etc) because their
                # handling is always the same.
                if normalized_unit in (unit.binary.symbol.lower(), unit.binary.name.lower()):
                    return int(tokens[0] * unit.binary.divider)
                # Now we will deal with ambiguous prefixes (K, M, G, etc),
                # symbols (KB, MB, GB, etc) and names (kilobyte, megabyte,
                # gigabyte, etc) according to the caller's preference.
                if (normalized_unit in (unit.decimal.symbol.lower(), unit.decimal.name.lower()) or
                        normalized_unit.startswith(unit.decimal.symbol[0].lower())):
                    return int(tokens[0] * (unit.binary.divider if binary else unit.decimal.divider))
    # We failed to parse the size specification.
    msg = "Failed to parse size! (input %r was tokenized as %r)"
    raise InvalidSize(format(msg, size, tokens))


def format_length(num_metres, keep_width=False):
    """
    Format a metre count as a human readable length.

    :param num_metres: The length to format in metres (float / integer).
    :param keep_width: :data:`True` if trailing zeros should not be stripped,
                       :data:`False` if they can be stripped.
    :returns: The corresponding human readable length (a string).

    This function supports ranges from nanometres to kilometres.

    Some examples:

    >>> from humanfriendly import format_length
    >>> format_length(0)
    '0 metres'
    >>> format_length(1)
    '1 metre'
    >>> format_length(5)
    '5 metres'
    >>> format_length(1000)
    '1 km'
    >>> format_length(0.004)
    '4 mm'
    """
    for unit in reversed(length_size_units):
        if num_metres >= unit['divider']:
            number = round_number(float(num_metres) / unit['divider'], keep_width=keep_width)
            return pluralize(number, unit['singular'], unit['plural'])
    return pluralize(num_metres, 'metre')


def parse_length(length):
    """
    Parse a human readable length and return the number of metres.

    :param length: The human readable length to parse (a string).
    :returns: The corresponding length in metres (a float).
    :raises: :exc:`InvalidLength` when the input can't be parsed.

    Some examples:

    >>> from humanfriendly import parse_length
    >>> parse_length('42')
    42
    >>> parse_length('1 km')
    1000
    >>> parse_length('5mm')
    0.005
    >>> parse_length('15.3cm')
    0.153
    """
    tokens = tokenize(length)
    if tokens and isinstance(tokens[0], numbers.Number):
        # If the input contains only a number, it's assumed to be the number of metres.
        if len(tokens) == 1:
            return tokens[0]
        # Otherwise we expect to find two tokens: A number and a unit.
        if len(tokens) == 2 and is_string(tokens[1]):
            normalized_unit = tokens[1].lower()
            # Try to match the first letter of the unit.
            for unit in length_size_units:
                if normalized_unit.startswith(unit['prefix']):
                    return tokens[0] * unit['divider']
    # We failed to parse the length specification.
    msg = "Failed to parse length! (input %r was tokenized as %r)"
    raise InvalidLength(format(msg, length, tokens))


def format_number(number, num_decimals=2):
    """
    Format a number as a string including thousands separators.

    :param number: The number to format (a number like an :class:`int`,
                   :class:`long` or :class:`float`).
    :param num_decimals: The number of decimals to render (2 by default). If no
                         decimal places are required to represent the number
                         they will be omitted regardless of this argument.
    :returns: The formatted number (a string).

    This function is intended to make it easier to recognize the order of size
    of the number being formatted.

    Here's an example:

    >>> from humanfriendly import format_number
    >>> print(format_number(6000000))
    6,000,000
    > print(format_number(6000000000.42))
    6,000,000,000.42
    > print(format_number(6000000000.42, num_decimals=0))
    6,000,000,000
    """
    integer_part, _, decimal_part = str(float(number)).partition('.')
    negative_sign = integer_part.startswith('-')
    reversed_digits = ''.join(reversed(integer_part.lstrip('-')))
    parts = []
    while reversed_digits:
        parts.append(reversed_digits[:3])
        reversed_digits = reversed_digits[3:]
    formatted_number = ''.join(reversed(','.join(parts)))
    decimals_to_add = decimal_part[:num_decimals].rstrip('0')
    if decimals_to_add:
        formatted_number += '.' + decimals_to_add
    if negative_sign:
        formatted_number = '-' + formatted_number
    return formatted_number


def round_number(count, keep_width=False):
    """
    Round a floating point number to two decimal places in a human friendly format.

    :param count: The number to format.
    :param keep_width: :data:`True` if trailing zeros should not be stripped,
                       :data:`False` if they can be stripped.
    :returns: The formatted number as a string. If no decimal places are
              required to represent the number, they will be omitted.

    The main purpose of this function is to be used by functions like
    :func:`format_length()`, :func:`format_size()` and
    :func:`format_timespan()`.

    Here are some examples:

    >>> from humanfriendly import round_number
    >>> round_number(1)
    '1'
    >>> round_number(math.pi)
    '3.14'
    >>> round_number(5.001)
    '5'
    """
    text = '%.2f' % float(count)
    if not keep_width:
        text = re.sub('0+$', '', text)
        text = re.sub(r'\.$', '', text)
    return text


def format_timespan(num_seconds, detailed=False, max_units=3):
    """
    Format a timespan in seconds as a human readable string.

    :param num_seconds: Any value accepted by :func:`coerce_seconds()`.
    :param detailed: If :data:`True` milliseconds are represented separately
                     instead of being represented as fractional seconds
                     (defaults to :data:`False`).
    :param max_units: The maximum number of units to show in the formatted time
                      span (an integer, defaults to three).
    :returns: The formatted timespan as a string.
    :raise: See :func:`coerce_seconds()`.

    Some examples:

    >>> from humanfriendly import format_timespan
    >>> format_timespan(0)
    '0 seconds'
    >>> format_timespan(1)
    '1 second'
    >>> import math
    >>> format_timespan(math.pi)
    '3.14 seconds'
    >>> hour = 60 * 60
    >>> day = hour * 24
    >>> week = day * 7
    >>> format_timespan(week * 52 + day * 2 + hour * 3)
    '1 year, 2 days and 3 hours'
    """
    num_seconds = coerce_seconds(num_seconds)
    if num_seconds < 60 and not detailed:
        # Fast path.
        return pluralize(round_number(num_seconds), 'second')
    else:
        # Slow path.
        result = []
        num_seconds = decimal.Decimal(str(num_seconds))
        relevant_units = list(reversed(time_units[0 if detailed else 3:]))
        for unit in relevant_units:
            # Extract the unit count from the remaining time.
            divider = decimal.Decimal(str(unit['divider']))
            count = num_seconds / divider
            num_seconds %= divider
            # Round the unit count appropriately.
            if unit != relevant_units[-1]:
                # Integer rounding for all but the smallest unit.
                count = int(count)
            else:
                # Floating point rounding for the smallest unit.
                count = round_number(count)
            # Only include relevant units in the result.
            if count not in (0, '0'):
                result.append(pluralize(count, unit['singular'], unit['plural']))
        if len(result) == 1:
            # A single count/unit combination.
            return result[0]
        else:
            if not detailed:
                # Remove `insignificant' data from the formatted timespan.
                result = result[:max_units]
            # Format the timespan in a readable way.
            return concatenate(result)


def parse_timespan(timespan):
    """
    Parse a "human friendly" timespan into the number of seconds.

    :param value: A string like ``5h`` (5 hours), ``10m`` (10 minutes) or
                  ``42s`` (42 seconds).
    :returns: The number of seconds as a floating point number.
    :raises: :exc:`InvalidTimespan` when the input can't be parsed.

    Note that the :func:`parse_timespan()` function is not meant to be the
    "mirror image" of the :func:`format_timespan()` function. Instead it's
    meant to allow humans to easily and succinctly specify a timespan with a
    minimal amount of typing. It's very useful to accept easy to write time
    spans as e.g. command line arguments to programs.

    The time units (and abbreviations) supported by this function are:

    - ms, millisecond, milliseconds
    - s, sec, secs, second, seconds
    - m, min, mins, minute, minutes
    - h, hour, hours
    - d, day, days
    - w, week, weeks
    - y, year, years

    Some examples:

    >>> from humanfriendly import parse_timespan
    >>> parse_timespan('42')
    42.0
    >>> parse_timespan('42s')
    42.0
    >>> parse_timespan('1m')
    60.0
    >>> parse_timespan('1h')
    3600.0
    >>> parse_timespan('1d')
    86400.0
    """
    tokens = tokenize(timespan)
    if tokens and isinstance(tokens[0], numbers.Number):
        # If the input contains only a number, it's assumed to be the number of seconds.
        if len(tokens) == 1:
            return float(tokens[0])
        # Otherwise we expect to find two tokens: A number and a unit.
        if len(tokens) == 2 and is_string(tokens[1]):
            normalized_unit = tokens[1].lower()
            for unit in time_units:
                if (normalized_unit == unit['singular'] or
                        normalized_unit == unit['plural'] or
                        normalized_unit in unit['abbreviations']):
                    return float(tokens[0]) * unit['divider']
    # We failed to parse the timespan specification.
    msg = "Failed to parse timespan! (input %r was tokenized as %r)"
    raise InvalidTimespan(format(msg, timespan, tokens))


def parse_date(datestring):
    """
    Parse a date/time string into a tuple of integers.

    :param datestring: The date/time string to parse.
    :returns: A tuple with the numbers ``(year, month, day, hour, minute,
              second)`` (all numbers are integers).
    :raises: :exc:`InvalidDate` when the date cannot be parsed.

    Supported date/time formats:

    - ``YYYY-MM-DD``
    - ``YYYY-MM-DD HH:MM:SS``

    .. note:: If you want to parse date/time strings with a fixed, known
              format and :func:`parse_date()` isn't useful to you, consider
              :func:`time.strptime()` or :meth:`datetime.datetime.strptime()`,
              both of which are included in the Python standard library.
              Alternatively for more complex tasks consider using the date/time
              parsing module in the dateutil_ package.

    Examples:

    >>> from humanfriendly import parse_date
    >>> parse_date('2013-06-17')
    (2013, 6, 17, 0, 0, 0)
    >>> parse_date('2013-06-17 02:47:42')
    (2013, 6, 17, 2, 47, 42)

    Here's how you convert the result to a number (`Unix time`_):

    >>> from humanfriendly import parse_date
    >>> from time import mktime
    >>> mktime(parse_date('2013-06-17 02:47:42') + (-1, -1, -1))
    1371430062.0

    And here's how you convert it to a :class:`datetime.datetime` object:

    >>> from humanfriendly import parse_date
    >>> from datetime import datetime
    >>> datetime(*parse_date('2013-06-17 02:47:42'))
    datetime.datetime(2013, 6, 17, 2, 47, 42)

    Here's an example that combines :func:`format_timespan()` and
    :func:`parse_date()` to calculate a human friendly timespan since a
    given date:

    >>> from humanfriendly import format_timespan, parse_date
    >>> from time import mktime, time
    >>> unix_time = mktime(parse_date('2013-06-17 02:47:42') + (-1, -1, -1))
    >>> seconds_since_then = time() - unix_time
    >>> print(format_timespan(seconds_since_then))
    1 year, 43 weeks and 1 day

    .. _dateutil: https://dateutil.readthedocs.io/en/latest/parser.html
    .. _Unix time: http://en.wikipedia.org/wiki/Unix_time
    """
    try:
        tokens = [t.strip() for t in datestring.split()]
        if len(tokens) >= 2:
            date_parts = list(map(int, tokens[0].split('-'))) + [1, 1]
            time_parts = list(map(int, tokens[1].split(':'))) + [0, 0, 0]
            return tuple(date_parts[0:3] + time_parts[0:3])
        else:
            year, month, day = (list(map(int, datestring.split('-'))) + [1, 1])[0:3]
            return (year, month, day, 0, 0, 0)
    except Exception:
        msg = "Invalid date! (expected 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' but got: %r)"
        raise InvalidDate(format(msg, datestring))


def format_path(pathname):
    """
    Shorten a pathname to make it more human friendly.

    :param pathname: An absolute pathname (a string).
    :returns: The pathname with the user's home directory abbreviated.

    Given an absolute pathname, this function abbreviates the user's home
    directory to ``~/`` in order to shorten the pathname without losing
    information. It is not an error if the pathname is not relative to the
    current user's home directory.

    Here's an example of its usage:

    >>> from os import environ
    >>> from os.path import join
    >>> vimrc = join(environ['HOME'], '.vimrc')
    >>> vimrc
    '/home/peter/.vimrc'
    >>> from humanfriendly import format_path
    >>> format_path(vimrc)
    '~/.vimrc'
    """
    pathname = os.path.abspath(pathname)
    home = os.environ.get('HOME')
    if home:
        home = os.path.abspath(home)
        if pathname.startswith(home):
            pathname = os.path.join('~', os.path.relpath(pathname, home))
    return pathname


def parse_path(pathname):
    """
    Convert a human friendly pathname to an absolute pathname.

    Expands leading tildes using :func:`os.path.expanduser()` and
    environment variables using :func:`os.path.expandvars()` and makes the
    resulting pathname absolute using :func:`os.path.abspath()`.

    :param pathname: A human friendly pathname (a string).
    :returns: An absolute pathname (a string).
    """
    return os.path.abspath(os.path.expanduser(os.path.expandvars(pathname)))


class Timer(object):

    """
    Easy to use timer to keep track of long during operations.
    """

    def __init__(self, start_time=None, resumable=False):
        """
        Remember the time when the :class:`Timer` was created.

        :param start_time: The start time (a float, defaults to the current time).
        :param resumable: Create a resumable timer (defaults to :data:`False`).

        When `start_time` is given :class:`Timer` uses :func:`time.time()` as a
        clock source, otherwise it uses :func:`humanfriendly.compat.monotonic()`.
        """
        if resumable:
            self.monotonic = True
            self.resumable = True
            self.start_time = 0.0
            self.total_time = 0.0
        elif start_time:
            self.monotonic = False
            self.resumable = False
            self.start_time = start_time
        else:
            self.monotonic = True
            self.resumable = False
            self.start_time = monotonic()

    def __enter__(self):
        """
        Start or resume counting elapsed time.

        :returns: The :class:`Timer` object.
        :raises: :exc:`~exceptions.ValueError` when the timer isn't resumable.
        """
        if not self.resumable:
            raise ValueError("Timer is not resumable!")
        self.start_time = monotonic()
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """
        Stop counting elapsed time.

        :raises: :exc:`~exceptions.ValueError` when the timer isn't resumable.
        """
        if not self.resumable:
            raise ValueError("Timer is not resumable!")
        if self.start_time:
            self.total_time += monotonic() - self.start_time
            self.start_time = 0.0

    def sleep(self, seconds):
        """
        Easy to use rate limiting of repeating actions.

        :param seconds: The number of seconds to sleep (an
                        integer or floating point number).

        This method sleeps for the given number of seconds minus the
        :attr:`elapsed_time`. If the resulting duration is negative
        :func:`time.sleep()` will still be called, but the argument
        given to it will be the number 0 (negative numbers cause
        :func:`time.sleep()` to raise an exception).

        The use case for this is to initialize a :class:`Timer` inside
        the body of a :keyword:`for` or :keyword:`while` loop and call
        :func:`Timer.sleep()` at the end of the loop body to rate limit
        whatever it is that is being done inside the loop body.

        For posterity: Although the implementation of :func:`sleep()` only
        requires a single line of code I've added it to :mod:`humanfriendly`
        anyway because now that I've thought about how to tackle this once I
        never want to have to think about it again :-P (unless I find ways to
        improve this).
        """
        time.sleep(max(0, seconds - self.elapsed_time))

    @property
    def elapsed_time(self):
        """
        Get the number of seconds counted so far.
        """
        elapsed_time = 0
        if self.resumable:
            elapsed_time += self.total_time
        if self.start_time:
            current_time = monotonic() if self.monotonic else time.time()
            elapsed_time += current_time - self.start_time
        return elapsed_time

    @property
    def rounded(self):
        """Human readable timespan rounded to seconds (a string)."""
        return format_timespan(round(self.elapsed_time))

    def __str__(self):
        """Show the elapsed time since the :class:`Timer` was created."""
        return format_timespan(self.elapsed_time)


class InvalidDate(Exception):

    """
    Raised when a string cannot be parsed into a date.

    For example:

    >>> from humanfriendly import parse_date
    >>> parse_date('2013-06-XY')
    Traceback (most recent call last):
      File "humanfriendly.py", line 206, in parse_date
        raise InvalidDate(format(msg, datestring))
    humanfriendly.InvalidDate: Invalid date! (expected 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' but got: '2013-06-XY')
    """


class InvalidSize(Exception):

    """
    Raised when a string cannot be parsed into a file size.

    For example:

    >>> from humanfriendly import parse_size
    >>> parse_size('5 Z')
    Traceback (most recent call last):
      File "humanfriendly/__init__.py", line 267, in parse_size
        raise InvalidSize(format(msg, size, tokens))
    humanfriendly.InvalidSize: Failed to parse size! (input '5 Z' was tokenized as [5, 'Z'])
    """


class InvalidLength(Exception):

    """
    Raised when a string cannot be parsed into a length.

    For example:

    >>> from humanfriendly import parse_length
    >>> parse_length('5 Z')
    Traceback (most recent call last):
      File "humanfriendly/__init__.py", line 267, in parse_length
        raise InvalidLength(format(msg, length, tokens))
    humanfriendly.InvalidLength: Failed to parse length! (input '5 Z' was tokenized as [5, 'Z'])
    """


class InvalidTimespan(Exception):

    """
    Raised when a string cannot be parsed into a timespan.

    For example:

    >>> from humanfriendly import parse_timespan
    >>> parse_timespan('1 age')
    Traceback (most recent call last):
      File "humanfriendly/__init__.py", line 419, in parse_timespan
        raise InvalidTimespan(format(msg, timespan, tokens))
    humanfriendly.InvalidTimespan: Failed to parse timespan! (input '1 age' was tokenized as [1, 'age'])
    """


# Define aliases for backwards compatibility.
define_aliases(
    module_name=__name__,
    # In humanfriendly 1.23 the format_table() function was added to render a
    # table using characters like dashes and vertical bars to emulate borders.
    # Since then support for other tables has been added and the name of
    # format_table() has changed.
    format_table='humanfriendly.tables.format_pretty_table',
    # In humanfriendly 1.30 the following text manipulation functions were
    # moved out into a separate module to enable their usage in other modules
    # of the humanfriendly package (without causing circular imports).
    compact='humanfriendly.text.compact',
    concatenate='humanfriendly.text.concatenate',
    dedent='humanfriendly.text.dedent',
    format='humanfriendly.text.format',
    is_empty_line='humanfriendly.text.is_empty_line',
    pluralize='humanfriendly.text.pluralize',
    tokenize='humanfriendly.text.tokenize',
    trim_empty_lines='humanfriendly.text.trim_empty_lines',
    # In humanfriendly 1.38 the prompt_for_choice() function was moved out into a
    # separate module because several variants of interactive prompts were added.
    prompt_for_choice='humanfriendly.prompts.prompt_for_choice',
    # In humanfriendly 8.0 the Spinner class and minimum_spinner_interval
    # variable were extracted to a new module and the erase_line_code,
    # hide_cursor_code and show_cursor_code variables were moved.
    AutomaticSpinner='humanfriendly.terminal.spinners.AutomaticSpinner',
    Spinner='humanfriendly.terminal.spinners.Spinner',
    erase_line_code='humanfriendly.terminal.ANSI_ERASE_LINE',
    hide_cursor_code='humanfriendly.terminal.ANSI_SHOW_CURSOR',
    minimum_spinner_interval='humanfriendly.terminal.spinners.MINIMUM_INTERVAL',
    show_cursor_code='humanfriendly.terminal.ANSI_HIDE_CURSOR',
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\libmp\libmpc.py ===
"""
Low-level functions for complex arithmetic.
"""

import sys

from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND

from .libmpf import (\
    round_floor, round_ceiling, round_down, round_up,
    round_nearest, round_fast, bitcount,
    bctable, normalize, normalize1, reciprocal_rnd, rshift, lshift, giant_steps,
    negative_rnd,
    to_str, to_fixed, from_man_exp, from_float, to_float, from_int, to_int,
    fzero, fone, ftwo, fhalf, finf, fninf, fnan, fnone,
    mpf_abs, mpf_pos, mpf_neg, mpf_add, mpf_sub, mpf_mul,
    mpf_div, mpf_mul_int, mpf_shift, mpf_sqrt, mpf_hypot,
    mpf_rdiv_int, mpf_floor, mpf_ceil, mpf_nint, mpf_frac,
    mpf_sign, mpf_hash,
    ComplexResult
)

from .libelefun import (\
    mpf_pi, mpf_exp, mpf_log, mpf_cos_sin, mpf_cosh_sinh, mpf_tan, mpf_pow_int,
    mpf_log_hypot,
    mpf_cos_sin_pi, mpf_phi,
    mpf_cos, mpf_sin, mpf_cos_pi, mpf_sin_pi,
    mpf_atan, mpf_atan2, mpf_cosh, mpf_sinh, mpf_tanh,
    mpf_asin, mpf_acos, mpf_acosh, mpf_nthroot, mpf_fibonacci
)

# An mpc value is a (real, imag) tuple
mpc_one = fone, fzero
mpc_zero = fzero, fzero
mpc_two = ftwo, fzero
mpc_half = (fhalf, fzero)

_infs = (finf, fninf)
_infs_nan = (finf, fninf, fnan)

def mpc_is_inf(z):
    """Check if either real or imaginary part is infinite"""
    re, im = z
    if re in _infs: return True
    if im in _infs: return True
    return False

def mpc_is_infnan(z):
    """Check if either real or imaginary part is infinite or nan"""
    re, im = z
    if re in _infs_nan: return True
    if im in _infs_nan: return True
    return False

def mpc_to_str(z, dps, **kwargs):
    re, im = z
    rs = to_str(re, dps)
    if im[0]:
        return rs + " - " + to_str(mpf_neg(im), dps, **kwargs) + "j"
    else:
        return rs + " + " + to_str(im, dps, **kwargs) + "j"

def mpc_to_complex(z, strict=False, rnd=round_fast):
    re, im = z
    return complex(to_float(re, strict, rnd), to_float(im, strict, rnd))

def mpc_hash(z):
    if sys.version_info >= (3, 2):
        re, im = z
        h = mpf_hash(re) + sys.hash_info.imag * mpf_hash(im)
        # Need to reduce either module 2^32 or 2^64
        h = h % (2**sys.hash_info.width)
        return int(h)
    else:
        try:
            return hash(mpc_to_complex(z, strict=True))
        except OverflowError:
            return hash(z)

def mpc_conjugate(z, prec, rnd=round_fast):
    re, im = z
    return re, mpf_neg(im, prec, rnd)

def mpc_is_nonzero(z):
    return z != mpc_zero

def mpc_add(z, w, prec, rnd=round_fast):
    a, b = z
    c, d = w
    return mpf_add(a, c, prec, rnd), mpf_add(b, d, prec, rnd)

def mpc_add_mpf(z, x, prec, rnd=round_fast):
    a, b = z
    return mpf_add(a, x, prec, rnd), b

def mpc_sub(z, w, prec=0, rnd=round_fast):
    a, b = z
    c, d = w
    return mpf_sub(a, c, prec, rnd), mpf_sub(b, d, prec, rnd)

def mpc_sub_mpf(z, p, prec=0, rnd=round_fast):
    a, b = z
    return mpf_sub(a, p, prec, rnd), b

def mpc_pos(z, prec, rnd=round_fast):
    a, b = z
    return mpf_pos(a, prec, rnd), mpf_pos(b, prec, rnd)

def mpc_neg(z, prec=None, rnd=round_fast):
    a, b = z
    return mpf_neg(a, prec, rnd), mpf_neg(b, prec, rnd)

def mpc_shift(z, n):
    a, b = z
    return mpf_shift(a, n), mpf_shift(b, n)

def mpc_abs(z, prec, rnd=round_fast):
    """Absolute value of a complex number, |a+bi|.
    Returns an mpf value."""
    a, b = z
    return mpf_hypot(a, b, prec, rnd)

def mpc_arg(z, prec, rnd=round_fast):
    """Argument of a complex number. Returns an mpf value."""
    a, b = z
    return mpf_atan2(b, a, prec, rnd)

def mpc_floor(z, prec, rnd=round_fast):
    a, b = z
    return mpf_floor(a, prec, rnd), mpf_floor(b, prec, rnd)

def mpc_ceil(z, prec, rnd=round_fast):
    a, b = z
    return mpf_ceil(a, prec, rnd), mpf_ceil(b, prec, rnd)

def mpc_nint(z, prec, rnd=round_fast):
    a, b = z
    return mpf_nint(a, prec, rnd), mpf_nint(b, prec, rnd)

def mpc_frac(z, prec, rnd=round_fast):
    a, b = z
    return mpf_frac(a, prec, rnd), mpf_frac(b, prec, rnd)


def mpc_mul(z, w, prec, rnd=round_fast):
    """
    Complex multiplication.

    Returns the real and imaginary part of (a+bi)*(c+di), rounded to
    the specified precision. The rounding mode applies to the real and
    imaginary parts separately.
    """
    a, b = z
    c, d = w
    p = mpf_mul(a, c)
    q = mpf_mul(b, d)
    r = mpf_mul(a, d)
    s = mpf_mul(b, c)
    re = mpf_sub(p, q, prec, rnd)
    im = mpf_add(r, s, prec, rnd)
    return re, im

def mpc_square(z, prec, rnd=round_fast):
    # (a+b*I)**2 == a**2 - b**2 + 2*I*a*b
    a, b = z
    p = mpf_mul(a,a)
    q = mpf_mul(b,b)
    r = mpf_mul(a,b, prec, rnd)
    re = mpf_sub(p, q, prec, rnd)
    im = mpf_shift(r, 1)
    return re, im

def mpc_mul_mpf(z, p, prec, rnd=round_fast):
    a, b = z
    re = mpf_mul(a, p, prec, rnd)
    im = mpf_mul(b, p, prec, rnd)
    return re, im

def mpc_mul_imag_mpf(z, x, prec, rnd=round_fast):
    """
    Multiply the mpc value z by I*x where x is an mpf value.
    """
    a, b = z
    re = mpf_neg(mpf_mul(b, x, prec, rnd))
    im = mpf_mul(a, x, prec, rnd)
    return re, im

def mpc_mul_int(z, n, prec, rnd=round_fast):
    a, b = z
    re = mpf_mul_int(a, n, prec, rnd)
    im = mpf_mul_int(b, n, prec, rnd)
    return re, im

def mpc_div(z, w, prec, rnd=round_fast):
    a, b = z
    c, d = w
    wp = prec + 10
    # mag = c*c + d*d
    mag = mpf_add(mpf_mul(c, c), mpf_mul(d, d), wp)
    # (a*c+b*d)/mag, (b*c-a*d)/mag
    t = mpf_add(mpf_mul(a,c), mpf_mul(b,d), wp)
    u = mpf_sub(mpf_mul(b,c), mpf_mul(a,d), wp)
    return mpf_div(t,mag,prec,rnd), mpf_div(u,mag,prec,rnd)

def mpc_div_mpf(z, p, prec, rnd=round_fast):
    """Calculate z/p where p is real"""
    a, b = z
    re = mpf_div(a, p, prec, rnd)
    im = mpf_div(b, p, prec, rnd)
    return re, im

def mpc_reciprocal(z, prec, rnd=round_fast):
    """Calculate 1/z efficiently"""
    a, b = z
    m = mpf_add(mpf_mul(a,a),mpf_mul(b,b),prec+10)
    re = mpf_div(a, m, prec, rnd)
    im = mpf_neg(mpf_div(b, m, prec, rnd))
    return re, im

def mpc_mpf_div(p, z, prec, rnd=round_fast):
    """Calculate p/z where p is real efficiently"""
    a, b = z
    m = mpf_add(mpf_mul(a,a),mpf_mul(b,b), prec+10)
    re = mpf_div(mpf_mul(a,p), m, prec, rnd)
    im = mpf_div(mpf_neg(mpf_mul(b,p)), m, prec, rnd)
    return re, im

def complex_int_pow(a, b, n):
    """Complex integer power: computes (a+b*I)**n exactly for
    nonnegative n (a and b must be Python ints)."""
    wre = 1
    wim = 0
    while n:
        if n & 1:
            wre, wim = wre*a - wim*b, wim*a + wre*b
            n -= 1
        a, b = a*a - b*b, 2*a*b
        n //= 2
    return wre, wim

def mpc_pow(z, w, prec, rnd=round_fast):
    if w[1] == fzero:
        return mpc_pow_mpf(z, w[0], prec, rnd)
    return mpc_exp(mpc_mul(mpc_log(z, prec+10), w, prec+10), prec, rnd)

def mpc_pow_mpf(z, p, prec, rnd=round_fast):
    psign, pman, pexp, pbc = p
    if pexp >= 0:
        return mpc_pow_int(z, (-1)**psign * (pman<<pexp), prec, rnd)
    if pexp == -1:
        sqrtz = mpc_sqrt(z, prec+10)
        return mpc_pow_int(sqrtz, (-1)**psign * pman, prec, rnd)
    return mpc_exp(mpc_mul_mpf(mpc_log(z, prec+10), p, prec+10), prec, rnd)

def mpc_pow_int(z, n, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return mpf_pow_int(a, n, prec, rnd), fzero
    if a == fzero:
        v = mpf_pow_int(b, n, prec, rnd)
        n %= 4
        if n == 0:
            return v, fzero
        elif n == 1:
            return fzero, v
        elif n == 2:
            return mpf_neg(v), fzero
        elif n == 3:
            return fzero, mpf_neg(v)
    if n == 0: return mpc_one
    if n == 1: return mpc_pos(z, prec, rnd)
    if n == 2: return mpc_square(z, prec, rnd)
    if n == -1: return mpc_reciprocal(z, prec, rnd)
    if n < 0: return mpc_reciprocal(mpc_pow_int(z, -n, prec+4), prec, rnd)
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if asign: aman = -aman
    if bsign: bman = -bman
    de = aexp - bexp
    abs_de = abs(de)
    exact_size = n*(abs_de + max(abc, bbc))
    if exact_size < 10000:
        if de > 0:
            aman <<= de
            aexp = bexp
        else:
            bman <<= (-de)
            bexp = aexp
        re, im = complex_int_pow(aman, bman, n)
        re = from_man_exp(re, int(n*aexp), prec, rnd)
        im = from_man_exp(im, int(n*bexp), prec, rnd)
        return re, im
    return mpc_exp(mpc_mul_int(mpc_log(z, prec+10), n, prec+10), prec, rnd)

def mpc_sqrt(z, prec, rnd=round_fast):
    """Complex square root (principal branch).

    We have sqrt(a+bi) = sqrt((r+a)/2) + b/sqrt(2*(r+a))*i where
    r = abs(a+bi), when a+bi is not a negative real number."""
    a, b = z
    if b == fzero:
        if a == fzero:
            return (a, b)
        # When a+bi is a negative real number, we get a real sqrt times i
        if a[0]:
            im = mpf_sqrt(mpf_neg(a), prec, rnd)
            return (fzero, im)
        else:
            re = mpf_sqrt(a, prec, rnd)
            return (re, fzero)
    wp = prec+20
    if not a[0]:                               # case a positive
        t  = mpf_add(mpc_abs((a, b), wp), a, wp)  # t = abs(a+bi) + a
        u = mpf_shift(t, -1)                      # u = t/2
        re = mpf_sqrt(u, prec, rnd)               # re = sqrt(u)
        v = mpf_shift(t, 1)                       # v = 2*t
        w  = mpf_sqrt(v, wp)                      # w = sqrt(v)
        im = mpf_div(b, w, prec, rnd)             # im = b / w
    else:                                      # case a negative
        t = mpf_sub(mpc_abs((a, b), wp), a, wp)   # t = abs(a+bi) - a
        u = mpf_shift(t, -1)                      # u = t/2
        im = mpf_sqrt(u, prec, rnd)               # im = sqrt(u)
        v = mpf_shift(t, 1)                       # v = 2*t
        w  = mpf_sqrt(v, wp)                      # w = sqrt(v)
        re = mpf_div(b, w, prec, rnd)             # re = b/w
        if b[0]:
            re = mpf_neg(re)
            im = mpf_neg(im)
    return re, im

def mpc_nthroot_fixed(a, b, n, prec):
    # a, b signed integers at fixed precision prec
    start = 50
    a1 = int(rshift(a, prec - n*start))
    b1 = int(rshift(b, prec - n*start))
    try:
        r = (a1 + 1j * b1)**(1.0/n)
        re = r.real
        im = r.imag
        re = MPZ(int(re))
        im = MPZ(int(im))
    except OverflowError:
        a1 = from_int(a1, start)
        b1 = from_int(b1, start)
        fn = from_int(n)
        nth = mpf_rdiv_int(1, fn, start)
        re, im = mpc_pow((a1, b1), (nth, fzero), start)
        re = to_int(re)
        im = to_int(im)
    extra = 10
    prevp = start
    extra1 = n
    for p in giant_steps(start, prec+extra):
        # this is slow for large n, unlike int_pow_fixed
        re2, im2 = complex_int_pow(re, im, n-1)
        re2 = rshift(re2, (n-1)*prevp - p - extra1)
        im2 = rshift(im2, (n-1)*prevp - p - extra1)
        r4 = (re2*re2 + im2*im2) >> (p + extra1)
        ap = rshift(a, prec - p)
        bp = rshift(b, prec - p)
        rec = (ap * re2 + bp * im2) >> p
        imc = (-ap * im2 + bp * re2) >> p
        reb = (rec << p) // r4
        imb = (imc << p) // r4
        re = (reb + (n-1)*lshift(re, p-prevp))//n
        im = (imb + (n-1)*lshift(im, p-prevp))//n
        prevp = p
    return re, im

def mpc_nthroot(z, n, prec, rnd=round_fast):
    """
    Complex n-th root.

    Use Newton method as in the real case when it is faster,
    otherwise use z**(1/n)
    """
    a, b = z
    if a[0] == 0 and b == fzero:
        re = mpf_nthroot(a, n, prec, rnd)
        return (re, fzero)
    if n < 2:
        if n == 0:
            return mpc_one
        if n == 1:
            return mpc_pos((a, b), prec, rnd)
        if n == -1:
            return mpc_div(mpc_one, (a, b), prec, rnd)
        inverse = mpc_nthroot((a, b), -n, prec+5, reciprocal_rnd[rnd])
        return mpc_div(mpc_one, inverse, prec, rnd)
    if n <= 20:
        prec2 = int(1.2 * (prec + 10))
        asign, aman, aexp, abc = a
        bsign, bman, bexp, bbc = b
        pf = mpc_abs((a,b), prec)
        if pf[-2] + pf[-1] > -10  and pf[-2] + pf[-1] < prec:
            af = to_fixed(a, prec2)
            bf = to_fixed(b, prec2)
            re, im = mpc_nthroot_fixed(af, bf, n, prec2)
            extra = 10
            re = from_man_exp(re, -prec2-extra, prec2, rnd)
            im = from_man_exp(im, -prec2-extra, prec2, rnd)
            return re, im
    fn = from_int(n)
    prec2 = prec+10 + 10
    nth = mpf_rdiv_int(1, fn, prec2)
    re, im = mpc_pow((a, b), (nth, fzero), prec2, rnd)
    re = normalize(re[0], re[1], re[2], re[3], prec, rnd)
    im = normalize(im[0], im[1], im[2], im[3], prec, rnd)
    return re, im

def mpc_cbrt(z, prec, rnd=round_fast):
    """
    Complex cubic root.
    """
    return mpc_nthroot(z, 3, prec, rnd)

def mpc_exp(z, prec, rnd=round_fast):
    """
    Complex exponential function.

    We use the direct formula exp(a+bi) = exp(a) * (cos(b) + sin(b)*i)
    for the computation. This formula is very nice because it is
    pefectly stable; since we just do real multiplications, the only
    numerical errors that can creep in are single-ulp rounding errors.

    The formula is efficient since mpmath's real exp is quite fast and
    since we can compute cos and sin simultaneously.

    It is no problem if a and b are large; if the implementations of
    exp/cos/sin are accurate and efficient for all real numbers, then
    so is this function for all complex numbers.
    """
    a, b = z
    if a == fzero:
        return mpf_cos_sin(b, prec, rnd)
    if b == fzero:
        return mpf_exp(a, prec, rnd), fzero
    mag = mpf_exp(a, prec+4, rnd)
    c, s = mpf_cos_sin(b, prec+4, rnd)
    re = mpf_mul(mag, c, prec, rnd)
    im = mpf_mul(mag, s, prec, rnd)
    return re, im

def mpc_log(z, prec, rnd=round_fast):
    re = mpf_log_hypot(z[0], z[1], prec, rnd)
    im = mpc_arg(z, prec, rnd)
    return re, im

def mpc_cos(z, prec, rnd=round_fast):
    """Complex cosine. The formula used is cos(a+bi) = cos(a)*cosh(b) -
    sin(a)*sinh(b)*i.

    The same comments apply as for the complex exp: only real
    multiplications are pewrormed, so no cancellation errors are
    possible. The formula is also efficient since we can compute both
    pairs (cos, sin) and (cosh, sinh) in single stwps."""
    a, b = z
    if b == fzero:
        return mpf_cos(a, prec, rnd), fzero
    if a == fzero:
        return mpf_cosh(b, prec, rnd), fzero
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(c, ch, prec, rnd)
    im = mpf_mul(s, sh, prec, rnd)
    return re, mpf_neg(im)

def mpc_sin(z, prec, rnd=round_fast):
    """Complex sine. We have sin(a+bi) = sin(a)*cosh(b) +
    cos(a)*sinh(b)*i. See the docstring for mpc_cos for additional
    comments."""
    a, b = z
    if b == fzero:
        return mpf_sin(a, prec, rnd), fzero
    if a == fzero:
        return fzero, mpf_sinh(b, prec, rnd)
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(s, ch, prec, rnd)
    im = mpf_mul(c, sh, prec, rnd)
    return re, im

def mpc_tan(z, prec, rnd=round_fast):
    """Complex tangent. Computed as tan(a+bi) = sin(2a)/M + sinh(2b)/M*i
    where M = cos(2a) + cosh(2b)."""
    a, b = z
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if b == fzero: return mpf_tan(a, prec, rnd), fzero
    if a == fzero: return fzero, mpf_tanh(b, prec, rnd)
    wp = prec + 15
    a = mpf_shift(a, 1)
    b = mpf_shift(b, 1)
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    # TODO: handle cancellation when c ~=  -1 and ch ~= 1
    mag = mpf_add(c, ch, wp)
    re = mpf_div(s, mag, prec, rnd)
    im = mpf_div(sh, mag, prec, rnd)
    return re, im

def mpc_cos_pi(z, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return mpf_cos_pi(a, prec, rnd), fzero
    b = mpf_mul(b, mpf_pi(prec+5), prec+5)
    if a == fzero:
        return mpf_cosh(b, prec, rnd), fzero
    wp = prec + 6
    c, s = mpf_cos_sin_pi(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(c, ch, prec, rnd)
    im = mpf_mul(s, sh, prec, rnd)
    return re, mpf_neg(im)

def mpc_sin_pi(z, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return mpf_sin_pi(a, prec, rnd), fzero
    b = mpf_mul(b, mpf_pi(prec+5), prec+5)
    if a == fzero:
        return fzero, mpf_sinh(b, prec, rnd)
    wp = prec + 6
    c, s = mpf_cos_sin_pi(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(s, ch, prec, rnd)
    im = mpf_mul(c, sh, prec, rnd)
    return re, im

def mpc_cos_sin(z, prec, rnd=round_fast):
    a, b = z
    if a == fzero:
        ch, sh = mpf_cosh_sinh(b, prec, rnd)
        return (ch, fzero), (fzero, sh)
    if b == fzero:
        c, s = mpf_cos_sin(a, prec, rnd)
        return (c, fzero), (s, fzero)
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    cre = mpf_mul(c, ch, prec, rnd)
    cim = mpf_mul(s, sh, prec, rnd)
    sre = mpf_mul(s, ch, prec, rnd)
    sim = mpf_mul(c, sh, prec, rnd)
    return (cre, mpf_neg(cim)), (sre, sim)

def mpc_cos_sin_pi(z, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        c, s = mpf_cos_sin_pi(a, prec, rnd)
        return (c, fzero), (s, fzero)
    b = mpf_mul(b, mpf_pi(prec+5), prec+5)
    if a == fzero:
        ch, sh = mpf_cosh_sinh(b, prec, rnd)
        return (ch, fzero), (fzero, sh)
    wp = prec + 6
    c, s = mpf_cos_sin_pi(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    cre = mpf_mul(c, ch, prec, rnd)
    cim = mpf_mul(s, sh, prec, rnd)
    sre = mpf_mul(s, ch, prec, rnd)
    sim = mpf_mul(c, sh, prec, rnd)
    return (cre, mpf_neg(cim)), (sre, sim)

def mpc_cosh(z, prec, rnd=round_fast):
    """Complex hyperbolic cosine. Computed as cosh(z) = cos(z*i)."""
    a, b = z
    return mpc_cos((b, mpf_neg(a)), prec, rnd)

def mpc_sinh(z, prec, rnd=round_fast):
    """Complex hyperbolic sine. Computed as sinh(z) = -i*sin(z*i)."""
    a, b = z
    b, a = mpc_sin((b, a), prec, rnd)
    return a, b

def mpc_tanh(z, prec, rnd=round_fast):
    """Complex hyperbolic tangent. Computed as tanh(z) = -i*tan(z*i)."""
    a, b = z
    b, a = mpc_tan((b, a), prec, rnd)
    return a, b

# TODO: avoid loss of accuracy
def mpc_atan(z, prec, rnd=round_fast):
    a, b = z
    # atan(z) = (I/2)*(log(1-I*z) - log(1+I*z))
    # x = 1-I*z = 1 + b - I*a
    # y = 1+I*z = 1 - b + I*a
    wp = prec + 15
    x = mpf_add(fone, b, wp), mpf_neg(a)
    y = mpf_sub(fone, b, wp), a
    l1 = mpc_log(x, wp)
    l2 = mpc_log(y, wp)
    a, b = mpc_sub(l1, l2, prec, rnd)
    # (I/2) * (a+b*I) = (-b/2 + a/2*I)
    v = mpf_neg(mpf_shift(b,-1)), mpf_shift(a,-1)
    # Subtraction at infinity gives correct real part but
    # wrong imaginary part (should be zero)
    if v[1] == fnan and mpc_is_inf(z):
        v = (v[0], fzero)
    return v

beta_crossover = from_float(0.6417)
alpha_crossover = from_float(1.5)

def acos_asin(z, prec, rnd, n):
    """ complex acos for n = 0, asin for n = 1
    The algorithm is described in
    T.E. Hull, T.F. Fairgrieve and P.T.P. Tang
    'Implementing the Complex Arcsine and Arcosine Functions
    using Exception Handling',
    ACM Trans. on Math. Software Vol. 23 (1997), p299
    The complex acos and asin can be defined as
    acos(z) = acos(beta) - I*sign(a)* log(alpha + sqrt(alpha**2 -1))
    asin(z) = asin(beta) + I*sign(a)* log(alpha + sqrt(alpha**2 -1))
    where z = a + I*b
    alpha = (1/2)*(r + s); beta = (1/2)*(r - s) = a/alpha
    r = sqrt((a+1)**2 + y**2); s = sqrt((a-1)**2 + y**2)
    These expressions are rewritten in different ways in different
    regions, delimited by two crossovers alpha_crossover and beta_crossover,
    and by abs(a) <= 1, in order to improve the numerical accuracy.
    """
    a, b = z
    wp = prec + 10
    # special cases with real argument
    if b == fzero:
        am = mpf_sub(fone, mpf_abs(a), wp)
        # case abs(a) <= 1
        if not am[0]:
            if n == 0:
                return mpf_acos(a, prec, rnd), fzero
            else:
                return mpf_asin(a, prec, rnd), fzero
        # cases abs(a) > 1
        else:
            # case a < -1
            if a[0]:
                pi = mpf_pi(prec, rnd)
                c = mpf_acosh(mpf_neg(a), prec, rnd)
                if n == 0:
                    return pi, mpf_neg(c)
                else:
                    return mpf_neg(mpf_shift(pi, -1)), c
            # case a > 1
            else:
                c = mpf_acosh(a, prec, rnd)
                if n == 0:
                    return fzero, c
                else:
                    pi = mpf_pi(prec, rnd)
                    return mpf_shift(pi, -1), mpf_neg(c)
    asign = bsign = 0
    if a[0]:
        a = mpf_neg(a)
        asign = 1
    if b[0]:
        b = mpf_neg(b)
        bsign = 1
    am = mpf_sub(fone, a, wp)
    ap = mpf_add(fone, a, wp)
    r = mpf_hypot(ap, b, wp)
    s = mpf_hypot(am, b, wp)
    alpha = mpf_shift(mpf_add(r, s, wp), -1)
    beta = mpf_div(a, alpha, wp)
    b2 = mpf_mul(b,b, wp)
    # case beta <= beta_crossover
    if not mpf_sub(beta_crossover, beta, wp)[0]:
        if n == 0:
            re = mpf_acos(beta, wp)
        else:
            re = mpf_asin(beta, wp)
    else:
        # to compute the real part in this region use the identity
        # asin(beta) = atan(beta/sqrt(1-beta**2))
        # beta/sqrt(1-beta**2) = (alpha + a) * (alpha - a)
        # alpha + a is numerically accurate; alpha - a can have
        # cancellations leading to numerical inaccuracies, so rewrite
        # it in differente ways according to the region
        Ax = mpf_add(alpha, a, wp)
        # case a <= 1
        if not am[0]:
            # c = b*b/(r + (a+1)); d = (s + (1-a))
            # alpha - a = (1/2)*(c + d)
            # case n=0: re = atan(sqrt((1/2) * Ax * (c + d))/a)
            # case n=1: re = atan(a/sqrt((1/2) * Ax * (c + d)))
            c = mpf_div(b2, mpf_add(r, ap, wp), wp)
            d = mpf_add(s, am, wp)
            re = mpf_shift(mpf_mul(Ax, mpf_add(c, d, wp), wp), -1)
            if n == 0:
                re = mpf_atan(mpf_div(mpf_sqrt(re, wp), a, wp), wp)
            else:
                re = mpf_atan(mpf_div(a, mpf_sqrt(re, wp), wp), wp)
        else:
            # c = Ax/(r + (a+1)); d = Ax/(s - (1-a))
            # alpha - a = (1/2)*(c + d)
            # case n = 0: re = atan(b*sqrt(c + d)/2/a)
            # case n = 1: re = atan(a/(b*sqrt(c + d)/2)
            c = mpf_div(Ax, mpf_add(r, ap, wp), wp)
            d = mpf_div(Ax, mpf_sub(s, am, wp), wp)
            re = mpf_shift(mpf_add(c, d, wp), -1)
            re = mpf_mul(b, mpf_sqrt(re, wp), wp)
            if n == 0:
                re = mpf_atan(mpf_div(re, a, wp), wp)
            else:
                re = mpf_atan(mpf_div(a, re, wp), wp)
    # to compute alpha + sqrt(alpha**2 - 1), if alpha <= alpha_crossover
    # replace it with 1 + Am1 + sqrt(Am1*(alpha+1)))
    # where Am1 = alpha -1
    # if alpha <= alpha_crossover:
    if not mpf_sub(alpha_crossover, alpha, wp)[0]:
        c1 = mpf_div(b2, mpf_add(r, ap, wp), wp)
        # case a < 1
        if mpf_neg(am)[0]:
            # Am1 = (1/2) * (b*b/(r + (a+1)) + b*b/(s + (1-a))
            c2 = mpf_add(s, am, wp)
            c2 = mpf_div(b2, c2, wp)
            Am1 = mpf_shift(mpf_add(c1, c2, wp), -1)
        else:
            # Am1 = (1/2) * (b*b/(r + (a+1)) + (s - (1-a)))
            c2 = mpf_sub(s, am, wp)
            Am1 = mpf_shift(mpf_add(c1, c2, wp), -1)
        # im = log(1 + Am1 + sqrt(Am1*(alpha+1)))
        im = mpf_mul(Am1, mpf_add(alpha, fone, wp), wp)
        im = mpf_log(mpf_add(fone, mpf_add(Am1, mpf_sqrt(im, wp), wp), wp), wp)
    else:
        # im = log(alpha + sqrt(alpha*alpha - 1))
        im = mpf_sqrt(mpf_sub(mpf_mul(alpha, alpha, wp), fone, wp), wp)
        im = mpf_log(mpf_add(alpha, im, wp), wp)
    if asign:
        if n == 0:
            re = mpf_sub(mpf_pi(wp), re, wp)
        else:
            re = mpf_neg(re)
    if not bsign and n == 0:
        im = mpf_neg(im)
    if bsign and n == 1:
        im = mpf_neg(im)
    re = normalize(re[0], re[1], re[2], re[3], prec, rnd)
    im = normalize(im[0], im[1], im[2], im[3], prec, rnd)
    return re, im

def mpc_acos(z, prec, rnd=round_fast):
    return acos_asin(z, prec, rnd, 0)

def mpc_asin(z, prec, rnd=round_fast):
    return acos_asin(z, prec, rnd, 1)

def mpc_asinh(z, prec, rnd=round_fast):
    # asinh(z) = I * asin(-I z)
    a, b = z
    a, b =  mpc_asin((b, mpf_neg(a)), prec, rnd)
    return mpf_neg(b), a

def mpc_acosh(z, prec, rnd=round_fast):
    # acosh(z) = -I * acos(z)   for Im(acos(z)) <= 0
    #            +I * acos(z)   otherwise
    a, b = mpc_acos(z, prec, rnd)
    if b[0] or b == fzero:
        return mpf_neg(b), a
    else:
        return b, mpf_neg(a)

def mpc_atanh(z, prec, rnd=round_fast):
    # atanh(z) = (log(1+z)-log(1-z))/2
    wp = prec + 15
    a = mpc_add(z, mpc_one, wp)
    b = mpc_sub(mpc_one, z, wp)
    a = mpc_log(a, wp)
    b = mpc_log(b, wp)
    v = mpc_shift(mpc_sub(a, b, wp), -1)
    # Subtraction at infinity gives correct imaginary part but
    # wrong real part (should be zero)
    if v[0] == fnan and mpc_is_inf(z):
        v = (fzero, v[1])
    return v

def mpc_fibonacci(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        return (mpf_fibonacci(re, prec, rnd), fzero)
    size = max(abs(re[2]+re[3]), abs(re[2]+re[3]))
    wp = prec + size + 20
    a = mpf_phi(wp)
    b = mpf_add(mpf_shift(a, 1), fnone, wp)
    u = mpc_pow((a, fzero), z, wp)
    v = mpc_cos_pi(z, wp)
    v = mpc_div(v, u, wp)
    u = mpc_sub(u, v, wp)
    u = mpc_div_mpf(u, b, prec, rnd)
    return u

def mpf_expj(x, prec, rnd='f'):
    raise ComplexResult

def mpc_expj(z, prec, rnd='f'):
    re, im = z
    if im == fzero:
        return mpf_cos_sin(re, prec, rnd)
    if re == fzero:
        return mpf_exp(mpf_neg(im), prec, rnd), fzero
    ey = mpf_exp(mpf_neg(im), prec+10)
    c, s = mpf_cos_sin(re, prec+10)
    re = mpf_mul(ey, c, prec, rnd)
    im = mpf_mul(ey, s, prec, rnd)
    return re, im

def mpf_expjpi(x, prec, rnd='f'):
    raise ComplexResult

def mpc_expjpi(z, prec, rnd='f'):
    re, im = z
    if im == fzero:
        return mpf_cos_sin_pi(re, prec, rnd)
    sign, man, exp, bc = im
    wp = prec+10
    if man:
        wp += max(0, exp+bc)
    im = mpf_neg(mpf_mul(mpf_pi(wp), im, wp))
    if re == fzero:
        return mpf_exp(im, prec, rnd), fzero
    ey = mpf_exp(im, prec+10)
    c, s = mpf_cos_sin_pi(re, prec+10)
    re = mpf_mul(ey, c, prec, rnd)
    im = mpf_mul(ey, s, prec, rnd)
    return re, im


if BACKEND == 'sage':
    try:
        import sage.libs.mpmath.ext_libmp as _lbmp
        mpc_exp = _lbmp.mpc_exp
        mpc_sqrt = _lbmp.mpc_sqrt
    except (ImportError, AttributeError):
        print("Warning: Sage imports in libmpc failed")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\exc.py ===
# exc.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Exceptions used with SQLAlchemy.

The base exception class is :exc:`.SQLAlchemyError`.  Exceptions which are
raised as a result of DBAPI exceptions are all subclasses of
:exc:`.DBAPIError`.

"""
from __future__ import annotations

import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union

from .util import compat
from .util import preloaded as _preloaded

if typing.TYPE_CHECKING:
    from .engine.interfaces import _AnyExecuteParams
    from .engine.interfaces import Dialect
    from .sql.compiler import Compiled
    from .sql.compiler import TypeCompiler
    from .sql.elements import ClauseElement

if typing.TYPE_CHECKING:
    _version_token: str
else:
    # set by __init__.py
    _version_token = None


class HasDescriptionCode:
    """helper which adds 'code' as an attribute and '_code_str' as a method"""

    code: Optional[str] = None

    def __init__(self, *arg: Any, **kw: Any):
        code = kw.pop("code", None)
        if code is not None:
            self.code = code
        super().__init__(*arg, **kw)

    _what_are_we = "error"

    def _code_str(self) -> str:
        if not self.code:
            return ""
        else:
            return (
                f"(Background on this {self._what_are_we} at: "
                f"https://sqlalche.me/e/{_version_token}/{self.code})"
            )

    def __str__(self) -> str:
        message = super().__str__()
        if self.code:
            message = "%s %s" % (message, self._code_str())
        return message


class SQLAlchemyError(HasDescriptionCode, Exception):
    """Generic error class."""

    def _message(self) -> str:
        # rules:
        #
        # 1. single arg string will usually be a unicode
        # object, but since __str__() must return unicode, check for
        # bytestring just in case
        #
        # 2. for multiple self.args, this is not a case in current
        # SQLAlchemy though this is happening in at least one known external
        # library, call str() which does a repr().
        #
        text: str

        if len(self.args) == 1:
            arg_text = self.args[0]

            if isinstance(arg_text, bytes):
                text = compat.decode_backslashreplace(arg_text, "utf-8")
            # This is for when the argument is not a string of any sort.
            # Otherwise, converting this exception to string would fail for
            # non-string arguments.
            else:
                text = str(arg_text)

            return text
        else:
            # this is not a normal case within SQLAlchemy but is here for
            # compatibility with Exception.args - the str() comes out as
            # a repr() of the tuple
            return str(self.args)

    def _sql_message(self) -> str:
        message = self._message()

        if self.code:
            message = "%s %s" % (message, self._code_str())

        return message

    def __str__(self) -> str:
        return self._sql_message()


class ArgumentError(SQLAlchemyError):
    """Raised when an invalid or conflicting function argument is supplied.

    This error generally corresponds to construction time state errors.

    """


class DuplicateColumnError(ArgumentError):
    """a Column is being added to a Table that would replace another
    Column, without appropriate parameters to allow this in place.

    .. versionadded:: 2.0.0b4

    """


class ObjectNotExecutableError(ArgumentError):
    """Raised when an object is passed to .execute() that can't be
    executed as SQL.

    """

    def __init__(self, target: Any):
        super().__init__("Not an executable object: %r" % target)
        self.target = target

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return self.__class__, (self.target,)


class NoSuchModuleError(ArgumentError):
    """Raised when a dynamically-loaded module (usually a database dialect)
    of a particular name cannot be located."""


class NoForeignKeysError(ArgumentError):
    """Raised when no foreign keys can be located between two selectables
    during a join."""


class AmbiguousForeignKeysError(ArgumentError):
    """Raised when more than one foreign key matching can be located
    between two selectables during a join."""


class ConstraintColumnNotFoundError(ArgumentError):
    """raised when a constraint refers to a string column name that
    is not present in the table being constrained.

    .. versionadded:: 2.0

    """


class CircularDependencyError(SQLAlchemyError):
    """Raised by topological sorts when a circular dependency is detected.

    There are two scenarios where this error occurs:

    * In a Session flush operation, if two objects are mutually dependent
      on each other, they can not be inserted or deleted via INSERT or
      DELETE statements alone; an UPDATE will be needed to post-associate
      or pre-deassociate one of the foreign key constrained values.
      The ``post_update`` flag described at :ref:`post_update` can resolve
      this cycle.
    * In a :attr:`_schema.MetaData.sorted_tables` operation, two
      :class:`_schema.ForeignKey`
      or :class:`_schema.ForeignKeyConstraint` objects mutually refer to each
      other.  Apply the ``use_alter=True`` flag to one or both,
      see :ref:`use_alter`.

    """

    def __init__(
        self,
        message: str,
        cycles: Any,
        edges: Any,
        msg: Optional[str] = None,
        code: Optional[str] = None,
    ):
        if msg is None:
            message += " (%s)" % ", ".join(repr(s) for s in cycles)
        else:
            message = msg
        SQLAlchemyError.__init__(self, message, code=code)
        self.cycles = cycles
        self.edges = edges

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return (
            self.__class__,
            (None, self.cycles, self.edges, self.args[0]),
            {"code": self.code} if self.code is not None else {},
        )


class CompileError(SQLAlchemyError):
    """Raised when an error occurs during SQL compilation"""


class UnsupportedCompilationError(CompileError):
    """Raised when an operation is not supported by the given compiler.

    .. seealso::

        :ref:`faq_sql_expression_string`

        :ref:`error_l7de`
    """

    code = "l7de"

    def __init__(
        self,
        compiler: Union[Compiled, TypeCompiler],
        element_type: Type[ClauseElement],
        message: Optional[str] = None,
    ):
        super().__init__(
            "Compiler %r can't render element of type %s%s"
            % (compiler, element_type, ": %s" % message if message else "")
        )
        self.compiler = compiler
        self.element_type = element_type
        self.message = message

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return self.__class__, (self.compiler, self.element_type, self.message)


class IdentifierError(SQLAlchemyError):
    """Raised when a schema name is beyond the max character limit"""


class DisconnectionError(SQLAlchemyError):
    """A disconnect is detected on a raw DB-API connection.

    This error is raised and consumed internally by a connection pool.  It can
    be raised by the :meth:`_events.PoolEvents.checkout`
    event so that the host pool
    forces a retry; the exception will be caught three times in a row before
    the pool gives up and raises :class:`~sqlalchemy.exc.InvalidRequestError`
    regarding the connection attempt.

    """

    invalidate_pool: bool = False


class InvalidatePoolError(DisconnectionError):
    """Raised when the connection pool should invalidate all stale connections.

    A subclass of :class:`_exc.DisconnectionError` that indicates that the
    disconnect situation encountered on the connection probably means the
    entire pool should be invalidated, as the database has been restarted.

    This exception will be handled otherwise the same way as
    :class:`_exc.DisconnectionError`, allowing three attempts to reconnect
    before giving up.

    .. versionadded:: 1.2

    """

    invalidate_pool: bool = True


class TimeoutError(SQLAlchemyError):  # noqa
    """Raised when a connection pool times out on getting a connection."""


class InvalidRequestError(SQLAlchemyError):
    """SQLAlchemy was asked to do something it can't do.

    This error generally corresponds to runtime state errors.

    """


class IllegalStateChangeError(InvalidRequestError):
    """An object that tracks state encountered an illegal state change
    of some kind.

    .. versionadded:: 2.0

    """


class NoInspectionAvailable(InvalidRequestError):
    """A subject passed to :func:`sqlalchemy.inspection.inspect` produced
    no context for inspection."""


class PendingRollbackError(InvalidRequestError):
    """A transaction has failed and needs to be rolled back before
    continuing.

    .. versionadded:: 1.4

    """


class ResourceClosedError(InvalidRequestError):
    """An operation was requested from a connection, cursor, or other
    object that's in a closed state."""


class NoSuchColumnError(InvalidRequestError, KeyError):
    """A nonexistent column is requested from a ``Row``."""


class NoResultFound(InvalidRequestError):
    """A database result was required but none was found.


    .. versionchanged:: 1.4  This exception is now part of the
       ``sqlalchemy.exc`` module in Core, moved from the ORM.  The symbol
       remains importable from ``sqlalchemy.orm.exc``.


    """


class MultipleResultsFound(InvalidRequestError):
    """A single database result was required but more than one were found.

    .. versionchanged:: 1.4  This exception is now part of the
       ``sqlalchemy.exc`` module in Core, moved from the ORM.  The symbol
       remains importable from ``sqlalchemy.orm.exc``.


    """


class NoReferenceError(InvalidRequestError):
    """Raised by ``ForeignKey`` to indicate a reference cannot be resolved."""

    table_name: str


class AwaitRequired(InvalidRequestError):
    """Error raised by the async greenlet spawn if no async operation
    was awaited when it required one.

    """

    code = "xd1r"


class MissingGreenlet(InvalidRequestError):
    r"""Error raised by the async greenlet await\_ if called while not inside
    the greenlet spawn context.

    """

    code = "xd2s"


class NoReferencedTableError(NoReferenceError):
    """Raised by ``ForeignKey`` when the referred ``Table`` cannot be
    located.

    """

    def __init__(self, message: str, tname: str):
        NoReferenceError.__init__(self, message)
        self.table_name = tname

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return self.__class__, (self.args[0], self.table_name)


class NoReferencedColumnError(NoReferenceError):
    """Raised by ``ForeignKey`` when the referred ``Column`` cannot be
    located.

    """

    def __init__(self, message: str, tname: str, cname: str):
        NoReferenceError.__init__(self, message)
        self.table_name = tname
        self.column_name = cname

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return (
            self.__class__,
            (self.args[0], self.table_name, self.column_name),
        )


class NoSuchTableError(InvalidRequestError):
    """Table does not exist or is not visible to a connection."""


class UnreflectableTableError(InvalidRequestError):
    """Table exists but can't be reflected for some reason.

    .. versionadded:: 1.2

    """


class UnboundExecutionError(InvalidRequestError):
    """SQL was attempted without a database connection to execute it on."""


class DontWrapMixin:
    """A mixin class which, when applied to a user-defined Exception class,
    will not be wrapped inside of :exc:`.StatementError` if the error is
    emitted within the process of executing a statement.

    E.g.::

        from sqlalchemy.exc import DontWrapMixin


        class MyCustomException(Exception, DontWrapMixin):
            pass


        class MySpecialType(TypeDecorator):
            impl = String

            def process_bind_param(self, value, dialect):
                if value == "invalid":
                    raise MyCustomException("invalid!")

    """


class StatementError(SQLAlchemyError):
    """An error occurred during execution of a SQL statement.

    :class:`StatementError` wraps the exception raised
    during execution, and features :attr:`.statement`
    and :attr:`.params` attributes which supply context regarding
    the specifics of the statement which had an issue.

    The wrapped exception object is available in
    the :attr:`.orig` attribute.

    """

    statement: Optional[str] = None
    """The string SQL statement being invoked when this exception occurred."""

    params: Optional[_AnyExecuteParams] = None
    """The parameter list being used when this exception occurred."""

    orig: Optional[BaseException] = None
    """The original exception that was thrown.

    """

    ismulti: Optional[bool] = None
    """multi parameter passed to repr_params().  None is meaningful."""

    connection_invalidated: bool = False

    def __init__(
        self,
        message: str,
        statement: Optional[str],
        params: Optional[_AnyExecuteParams],
        orig: Optional[BaseException],
        hide_parameters: bool = False,
        code: Optional[str] = None,
        ismulti: Optional[bool] = None,
    ):
        SQLAlchemyError.__init__(self, message, code=code)
        self.statement = statement
        self.params = params
        self.orig = orig
        self.ismulti = ismulti
        self.hide_parameters = hide_parameters
        self.detail: List[str] = []

    def add_detail(self, msg: str) -> None:
        self.detail.append(msg)

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return (
            self.__class__,
            (
                self.args[0],
                self.statement,
                self.params,
                self.orig,
                self.hide_parameters,
                self.__dict__.get("code"),
                self.ismulti,
            ),
            {"detail": self.detail},
        )

    @_preloaded.preload_module("sqlalchemy.sql.util")
    def _sql_message(self) -> str:
        util = _preloaded.sql_util

        details = [self._message()]
        if self.statement:
            stmt_detail = "[SQL: %s]" % self.statement
            details.append(stmt_detail)
            if self.params:
                if self.hide_parameters:
                    details.append(
                        "[SQL parameters hidden due to hide_parameters=True]"
                    )
                else:
                    params_repr = util._repr_params(
                        self.params, 10, ismulti=self.ismulti
                    )
                    details.append("[parameters: %r]" % params_repr)
        code_str = self._code_str()
        if code_str:
            details.append(code_str)
        return "\n".join(["(%s)" % det for det in self.detail] + details)


class DBAPIError(StatementError):
    """Raised when the execution of a database operation fails.

    Wraps exceptions raised by the DB-API underlying the
    database operation.  Driver-specific implementations of the standard
    DB-API exception types are wrapped by matching sub-types of SQLAlchemy's
    :class:`DBAPIError` when possible.  DB-API's ``Error`` type maps to
    :class:`DBAPIError` in SQLAlchemy, otherwise the names are identical.  Note
    that there is no guarantee that different DB-API implementations will
    raise the same exception type for any given error condition.

    :class:`DBAPIError` features :attr:`~.StatementError.statement`
    and :attr:`~.StatementError.params` attributes which supply context
    regarding the specifics of the statement which had an issue, for the
    typical case when the error was raised within the context of
    emitting a SQL statement.

    The wrapped exception object is available in the
    :attr:`~.StatementError.orig` attribute. Its type and properties are
    DB-API implementation specific.

    """

    code = "dbapi"

    @overload
    @classmethod
    def instance(
        cls,
        statement: Optional[str],
        params: Optional[_AnyExecuteParams],
        orig: Exception,
        dbapi_base_err: Type[Exception],
        hide_parameters: bool = False,
        connection_invalidated: bool = False,
        dialect: Optional[Dialect] = None,
        ismulti: Optional[bool] = None,
    ) -> StatementError: ...

    @overload
    @classmethod
    def instance(
        cls,
        statement: Optional[str],
        params: Optional[_AnyExecuteParams],
        orig: DontWrapMixin,
        dbapi_base_err: Type[Exception],
        hide_parameters: bool = False,
        connection_invalidated: bool = False,
        dialect: Optional[Dialect] = None,
        ismulti: Optional[bool] = None,
    ) -> DontWrapMixin: ...

    @overload
    @classmethod
    def instance(
        cls,
        statement: Optional[str],
        params: Optional[_AnyExecuteParams],
        orig: BaseException,
        dbapi_base_err: Type[Exception],
        hide_parameters: bool = False,
        connection_invalidated: bool = False,
        dialect: Optional[Dialect] = None,
        ismulti: Optional[bool] = None,
    ) -> BaseException: ...

    @classmethod
    def instance(
        cls,
        statement: Optional[str],
        params: Optional[_AnyExecuteParams],
        orig: Union[BaseException, DontWrapMixin],
        dbapi_base_err: Type[Exception],
        hide_parameters: bool = False,
        connection_invalidated: bool = False,
        dialect: Optional[Dialect] = None,
        ismulti: Optional[bool] = None,
    ) -> Union[BaseException, DontWrapMixin]:
        # Don't ever wrap these, just return them directly as if
        # DBAPIError didn't exist.
        if (
            isinstance(orig, BaseException) and not isinstance(orig, Exception)
        ) or isinstance(orig, DontWrapMixin):
            return orig

        if orig is not None:
            # not a DBAPI error, statement is present.
            # raise a StatementError
            if isinstance(orig, SQLAlchemyError) and statement:
                return StatementError(
                    "(%s.%s) %s"
                    % (
                        orig.__class__.__module__,
                        orig.__class__.__name__,
                        orig.args[0],
                    ),
                    statement,
                    params,
                    orig,
                    hide_parameters=hide_parameters,
                    code=orig.code,
                    ismulti=ismulti,
                )
            elif not isinstance(orig, dbapi_base_err) and statement:
                return StatementError(
                    "(%s.%s) %s"
                    % (
                        orig.__class__.__module__,
                        orig.__class__.__name__,
                        orig,
                    ),
                    statement,
                    params,
                    orig,
                    hide_parameters=hide_parameters,
                    ismulti=ismulti,
                )

            glob = globals()
            for super_ in orig.__class__.__mro__:
                name = super_.__name__
                if dialect:
                    name = dialect.dbapi_exception_translation_map.get(
                        name, name
                    )
                if name in glob and issubclass(glob[name], DBAPIError):
                    cls = glob[name]
                    break

        return cls(
            statement,
            params,
            orig,
            connection_invalidated=connection_invalidated,
            hide_parameters=hide_parameters,
            code=cls.code,
            ismulti=ismulti,
        )

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return (
            self.__class__,
            (
                self.statement,
                self.params,
                self.orig,
                self.hide_parameters,
                self.connection_invalidated,
                self.__dict__.get("code"),
                self.ismulti,
            ),
            {"detail": self.detail},
        )

    def __init__(
        self,
        statement: Optional[str],
        params: Optional[_AnyExecuteParams],
        orig: BaseException,
        hide_parameters: bool = False,
        connection_invalidated: bool = False,
        code: Optional[str] = None,
        ismulti: Optional[bool] = None,
    ):
        try:
            text = str(orig)
        except Exception as e:
            text = "Error in str() of DB-API-generated exception: " + str(e)
        StatementError.__init__(
            self,
            "(%s.%s) %s"
            % (orig.__class__.__module__, orig.__class__.__name__, text),
            statement,
            params,
            orig,
            hide_parameters,
            code=code,
            ismulti=ismulti,
        )
        self.connection_invalidated = connection_invalidated


class InterfaceError(DBAPIError):
    """Wraps a DB-API InterfaceError."""

    code = "rvf5"


class DatabaseError(DBAPIError):
    """Wraps a DB-API DatabaseError."""

    code = "4xp6"


class DataError(DatabaseError):
    """Wraps a DB-API DataError."""

    code = "9h9h"


class OperationalError(DatabaseError):
    """Wraps a DB-API OperationalError."""

    code = "e3q8"


class IntegrityError(DatabaseError):
    """Wraps a DB-API IntegrityError."""

    code = "gkpj"


class InternalError(DatabaseError):
    """Wraps a DB-API InternalError."""

    code = "2j85"


class ProgrammingError(DatabaseError):
    """Wraps a DB-API ProgrammingError."""

    code = "f405"


class NotSupportedError(DatabaseError):
    """Wraps a DB-API NotSupportedError."""

    code = "tw8g"


# Warnings


class SATestSuiteWarning(Warning):
    """warning for a condition detected during tests that is non-fatal

    Currently outside of SAWarning so that we can work around tools like
    Alembic doing the wrong thing with warnings.

    """


class SADeprecationWarning(HasDescriptionCode, DeprecationWarning):
    """Issued for usage of deprecated APIs."""

    deprecated_since: Optional[str] = None
    "Indicates the version that started raising this deprecation warning"


class Base20DeprecationWarning(SADeprecationWarning):
    """Issued for usage of APIs specifically deprecated or legacy in
    SQLAlchemy 2.0.

    .. seealso::

        :ref:`error_b8d9`.

        :ref:`deprecation_20_mode`

    """

    deprecated_since: Optional[str] = "1.4"
    "Indicates the version that started raising this deprecation warning"

    def __str__(self) -> str:
        return (
            super().__str__()
            + " (Background on SQLAlchemy 2.0 at: https://sqlalche.me/e/b8d9)"
        )


class LegacyAPIWarning(Base20DeprecationWarning):
    """indicates an API that is in 'legacy' status, a long term deprecation."""


class MovedIn20Warning(Base20DeprecationWarning):
    """Subtype of RemovedIn20Warning to indicate an API that moved only."""


class SAPendingDeprecationWarning(PendingDeprecationWarning):
    """A similar warning as :class:`_exc.SADeprecationWarning`, this warning
    is not used in modern versions of SQLAlchemy.

    """

    deprecated_since: Optional[str] = None
    "Indicates the version that started raising this deprecation warning"


class SAWarning(HasDescriptionCode, RuntimeWarning):
    """Issued at runtime."""

    _what_are_we = "warning"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\pipeline_stable_diffusion.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modified from TensorRT demo diffusion, which has the following license:
#
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
# --------------------------------------------------------------------------

import os
import pathlib
import random
import time
from typing import Any

import numpy as np
import nvtx
import torch
from cuda import cudart
from diffusion_models import PipelineInfo, get_tokenizer
from diffusion_schedulers import DDIMScheduler, EulerAncestralDiscreteScheduler, LCMScheduler, UniPCMultistepScheduler
from engine_builder import EngineType
from engine_builder_ort_cuda import OrtCudaEngineBuilder
from engine_builder_ort_trt import OrtTensorrtEngineBuilder
from engine_builder_tensorrt import TensorrtEngineBuilder
from engine_builder_torch import TorchEngineBuilder
from PIL import Image


class StableDiffusionPipeline:
    """
    Stable Diffusion pipeline using TensorRT.
    """

    def __init__(
        self,
        pipeline_info: PipelineInfo,
        max_batch_size=16,
        scheduler="DDIM",
        device="cuda",
        output_dir=".",
        verbose=False,
        nvtx_profile=False,
        use_cuda_graph=False,
        framework_model_dir="pytorch_model",
        engine_type: EngineType = EngineType.ORT_CUDA,
    ):
        """
        Initializes the Diffusion pipeline.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of pipeline.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of [DDIM, EulerA, UniPC, LCM].
            device (str):
                PyTorch device to run inference. Default: 'cuda'
            output_dir (str):
                Output directory for log files and image artifacts
            verbose (bool):
                Enable verbose logging.
            nvtx_profile (bool):
                Insert NVTX profiling markers.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
            framework_model_dir (str):
                cache directory for framework checkpoints
            engine_type (EngineType)
                backend engine type like ORT_TRT or TRT
        """

        self.pipeline_info = pipeline_info
        self.version = pipeline_info.version

        self.vae_scaling_factor = pipeline_info.vae_scaling_factor()

        self.max_batch_size = max_batch_size

        self.framework_model_dir = framework_model_dir
        self.output_dir = output_dir
        for directory in [self.framework_model_dir, self.output_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        self.device = device
        self.torch_device = torch.device(device, torch.cuda.current_device())
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile

        self.use_cuda_graph = use_cuda_graph

        self.tokenizer = None
        self.tokenizer2 = None

        self.generator = torch.Generator(device="cuda")
        self.actual_steps = None

        self.current_scheduler = None
        self.set_scheduler(scheduler)

        # backend engine
        self.engine_type = engine_type
        if engine_type == EngineType.TRT:
            self.backend = TensorrtEngineBuilder(pipeline_info, max_batch_size, device, use_cuda_graph)
        elif engine_type == EngineType.ORT_TRT:
            self.backend = OrtTensorrtEngineBuilder(pipeline_info, max_batch_size, device, use_cuda_graph)
        elif engine_type == EngineType.ORT_CUDA:
            self.backend = OrtCudaEngineBuilder(pipeline_info, max_batch_size, device, use_cuda_graph)
        elif engine_type == EngineType.TORCH:
            self.backend = TorchEngineBuilder(pipeline_info, max_batch_size, device, use_cuda_graph)
        else:
            raise RuntimeError(f"Backend engine type {engine_type.name} is not supported")

        # Load text tokenizer
        if not self.pipeline_info.is_xl_refiner():
            self.tokenizer = get_tokenizer(self.pipeline_info, self.framework_model_dir, subfolder="tokenizer")

        if self.pipeline_info.is_xl():
            self.tokenizer2 = get_tokenizer(self.pipeline_info, self.framework_model_dir, subfolder="tokenizer_2")

        self.control_image_processor = None
        if self.pipeline_info.is_xl() and self.pipeline_info.controlnet:
            from diffusers.image_processor import VaeImageProcessor

            self.control_image_processor = VaeImageProcessor(
                vae_scale_factor=8, do_convert_rgb=True, do_normalize=False
            )

        # Create CUDA events
        self.events = {}
        for stage in ["clip", "denoise", "vae", "vae_encoder", "pil"]:
            for marker in ["start", "stop"]:
                self.events[stage + "-" + marker] = cudart.cudaEventCreate()[1]
        self.markers = {}

    def is_backend_tensorrt(self):
        return self.engine_type == EngineType.TRT

    def set_scheduler(self, scheduler: str):
        if scheduler == self.current_scheduler:
            return

        # Scheduler options
        sched_opts = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012}
        if self.version in ("2.0", "2.1"):
            sched_opts["prediction_type"] = "v_prediction"
        else:
            sched_opts["prediction_type"] = "epsilon"

        if scheduler == "DDIM":
            self.scheduler = DDIMScheduler(device=self.device, **sched_opts)
        elif scheduler == "EulerA":
            self.scheduler = EulerAncestralDiscreteScheduler(device=self.device, **sched_opts)
        elif scheduler == "UniPC":
            self.scheduler = UniPCMultistepScheduler(device=self.device, **sched_opts)
        elif scheduler == "LCM":
            self.scheduler = LCMScheduler(device=self.device, **sched_opts)
        else:
            raise ValueError("Scheduler should be either DDIM, EulerA, UniPC or LCM")

        self.current_scheduler = scheduler
        self.denoising_steps = None

    def set_denoising_steps(self, denoising_steps: int):
        if not (self.denoising_steps == denoising_steps and isinstance(self.scheduler, DDIMScheduler)):
            self.scheduler.set_timesteps(denoising_steps)
            self.scheduler.configure()
            self.denoising_steps = denoising_steps

    def load_resources(self, image_height, image_width, batch_size):
        # If engine is built with static input shape, call this only once after engine build.
        # Otherwise, it need be called before every inference run.
        self.backend.load_resources(image_height, image_width, batch_size)

    def set_random_seed(self, seed):
        if isinstance(seed, int):
            self.generator.manual_seed(seed)
        else:
            self.generator.seed()

    def get_current_seed(self):
        return self.generator.initial_seed()

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e)

        if self.backend:
            self.backend.teardown()

    def run_engine(self, model_name, feed_dict):
        return self.backend.run_engine(model_name, feed_dict)

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width):
        latents_dtype = torch.float16
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=self.generator)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def initialize_timesteps(self, timesteps, strength):
        """Initialize timesteps for refiner."""
        self.scheduler.set_timesteps(timesteps)
        offset = self.scheduler.steps_offset if hasattr(self.scheduler, "steps_offset") else 0
        init_timestep = int(timesteps * strength) + offset
        init_timestep = min(init_timestep, timesteps)
        t_start = max(timesteps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)
        return timesteps, t_start

    def initialize_refiner(self, batch_size, image, strength):
        """Add noise to a reference image."""
        # Initialize timesteps
        timesteps, t_start = self.initialize_timesteps(self.denoising_steps, strength)

        latent_timestep = timesteps[:1].repeat(batch_size)

        # Pre-process input image
        image = self.preprocess_images(batch_size, (image,))[0]

        # VAE encode init image
        if image.shape[1] == 4:
            init_latents = image
        else:
            init_latents = self.encode_image(image)

        # Add noise to latents using timesteps
        noise = torch.randn(init_latents.shape, device=self.device, dtype=torch.float16, generator=self.generator)

        latents = self.scheduler.add_noise(init_latents, noise, t_start, latent_timestep)

        return timesteps, t_start, latents

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        dtype,
        requires_aesthetics_score,
    ):
        if requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    def start_profile(self, name, color="blue"):
        if self.nvtx_profile:
            self.markers[name] = nvtx.start_range(message=name, color=color)
        event_name = name + "-start"
        if event_name in self.events:
            cudart.cudaEventRecord(self.events[event_name], 0)

    def stop_profile(self, name):
        event_name = name + "-stop"
        if event_name in self.events:
            cudart.cudaEventRecord(self.events[event_name], 0)
        if self.nvtx_profile:
            nvtx.end_range(self.markers[name])

    def preprocess_images(self, batch_size, images=()):
        self.start_profile("preprocess", color="pink")
        init_images = []
        for i in images:
            image = i.to(self.device)
            if image.shape[0] != batch_size:
                image = image.repeat(batch_size, 1, 1, 1)
            init_images.append(image)
        self.stop_profile("preprocess")
        return tuple(init_images)

    def preprocess_controlnet_images(
        self, batch_size, images=None, do_classifier_free_guidance=True, height=1024, width=1024
    ):
        """
        Process a list of PIL.Image.Image as control images, and return a torch tensor.
        """
        if images is None:
            return None
        self.start_profile("preprocess", color="pink")

        if not self.pipeline_info.is_xl():
            images = [
                torch.from_numpy(
                    (np.array(image.convert("RGB")).astype(np.float32) / 255.0)[..., None].transpose(3, 2, 0, 1)
                )
                .to(device=self.device, dtype=torch.float16)
                .repeat_interleave(batch_size, dim=0)
                for image in images
            ]
        else:
            images = [
                self.control_image_processor.preprocess(image, height=height, width=width)
                .to(device=self.device, dtype=torch.float16)
                .repeat_interleave(batch_size, dim=0)
                for image in images
            ]

        if do_classifier_free_guidance:
            images = [torch.cat([i] * 2) for i in images]
        images = torch.cat([image[None, ...] for image in images], dim=0)

        self.stop_profile("preprocess")
        return images

    def encode_prompt(
        self,
        prompt,
        negative_prompt,
        encoder="clip",
        tokenizer=None,
        pooled_outputs=False,
        output_hidden_states=False,
        force_zeros_for_empty_prompt=False,
        do_classifier_free_guidance=True,
        dtype=torch.float16,
    ):
        if tokenizer is None:
            tokenizer = self.tokenizer

        self.start_profile("clip", color="green")

        def tokenize(prompt, output_hidden_states):
            text_input_ids = (
                tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                .input_ids.type(torch.int32)
                .to(self.device)
            )

            hidden_states = None
            if self.engine_type == EngineType.TORCH:
                outputs = self.backend.engines[encoder](text_input_ids)
                text_embeddings = outputs[0]
                if output_hidden_states:
                    hidden_states = outputs["last_hidden_state"]
            else:
                outputs = self.run_engine(encoder, {"input_ids": text_input_ids})
                text_embeddings = outputs["text_embeddings"]
                if output_hidden_states:
                    hidden_states = outputs["hidden_states"]
            return text_embeddings, hidden_states

        # Tokenize prompt
        text_embeddings, hidden_states = tokenize(prompt, output_hidden_states)

        # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
        text_embeddings = text_embeddings.clone()
        if hidden_states is not None:
            hidden_states = hidden_states.clone()

        # Note: negative prompt embedding is not needed for SD XL when guidance <= 1
        if do_classifier_free_guidance:
            # For SD XL base, handle force_zeros_for_empty_prompt
            is_empty_negative_prompt = all(not i for i in negative_prompt)
            if force_zeros_for_empty_prompt and is_empty_negative_prompt:
                uncond_embeddings = torch.zeros_like(text_embeddings)
                if output_hidden_states:
                    uncond_hidden_states = torch.zeros_like(hidden_states)
            else:
                # Tokenize negative prompt
                uncond_embeddings, uncond_hidden_states = tokenize(negative_prompt, output_hidden_states)

            # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            if output_hidden_states:
                hidden_states = torch.cat([uncond_hidden_states, hidden_states])

        self.stop_profile("clip")

        if pooled_outputs:
            # For text encoder in sdxl base
            return hidden_states.to(dtype=dtype), text_embeddings.to(dtype=dtype)

        if output_hidden_states:
            # For text encoder 2 in sdxl base or refiner
            return hidden_states.to(dtype=dtype)

        # For text encoder in sd 1.5
        return text_embeddings.to(dtype=dtype)

    def denoise_latent(
        self,
        latents,
        text_embeddings,
        denoiser="unet",
        timesteps=None,
        step_offset=0,
        guidance=7.5,
        add_kwargs=None,
    ):
        do_classifier_free_guidance = guidance > 1.0

        self.start_profile("denoise", color="blue")

        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps

        for step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, step_offset + step_index, timestep
            )

            # Predict the noise residual
            if self.nvtx_profile:
                nvtx_unet = nvtx.start_range(message="unet", color="blue")

            params = {
                "sample": latent_model_input,
                "timestep": timestep.to(latents.dtype),
                "encoder_hidden_states": text_embeddings,
            }

            if add_kwargs:
                params.update(add_kwargs)

            noise_pred = self.run_engine(denoiser, params)["latent"]

            if self.nvtx_profile:
                nvtx.end_range(nvtx_unet)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

            if type(self.scheduler) is UniPCMultistepScheduler:
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            elif type(self.scheduler) is LCMScheduler:
                latents = self.scheduler.step(noise_pred, timestep, latents, generator=self.generator)[0]
            else:
                latents = self.scheduler.step(noise_pred, latents, step_offset + step_index, timestep)

        # The actual number of steps. It might be different from denoising_steps.
        self.actual_steps = len(timesteps)

        self.stop_profile("denoise")
        return latents

    def encode_image(self, image):
        self.start_profile("vae_encoder", color="red")
        init_latents = self.run_engine("vae_encoder", {"images": image})["latent"]
        init_latents = self.vae_scaling_factor * init_latents
        self.stop_profile("vae_encoder")
        return init_latents

    def decode_latent(self, latents):
        self.start_profile("vae", color="red")
        images = self.backend.vae_decode(latents)
        self.stop_profile("vae")
        return images

    def print_summary(self, tic, toc, batch_size, vae_enc=False, pil=False) -> dict[str, Any]:
        throughput = batch_size / (toc - tic)
        latency_clip = cudart.cudaEventElapsedTime(self.events["clip-start"], self.events["clip-stop"])[1]
        latency_unet = cudart.cudaEventElapsedTime(self.events["denoise-start"], self.events["denoise-stop"])[1]
        latency_vae = cudart.cudaEventElapsedTime(self.events["vae-start"], self.events["vae-stop"])[1]
        latency_vae_encoder = (
            cudart.cudaEventElapsedTime(self.events["vae_encoder-start"], self.events["vae_encoder-stop"])[1]
            if vae_enc
            else None
        )
        latency_pil = cudart.cudaEventElapsedTime(self.events["pil-start"], self.events["pil-stop"])[1] if pil else None

        latency = (toc - tic) * 1000.0

        print("|----------------|--------------|")
        print("| {:^14} | {:^12} |".format("Module", "Latency"))
        print("|----------------|--------------|")
        if vae_enc:
            print("| {:^14} | {:>9.2f} ms |".format("VAE-Enc", latency_vae_encoder))
        print("| {:^14} | {:>9.2f} ms |".format("CLIP", latency_clip))
        print(
            "| {:^14} | {:>9.2f} ms |".format(
                "UNet" + ("+CNet" if self.pipeline_info.controlnet else "") + " x " + str(self.actual_steps),
                latency_unet,
            )
        )
        print("| {:^14} | {:>9.2f} ms |".format("VAE-Dec", latency_vae))
        pipeline = "Refiner" if self.pipeline_info.is_xl_refiner() else "Pipeline"
        if pil:
            print("| {:^14} | {:>9.2f} ms |".format("PIL", latency_pil))
        print("|----------------|--------------|")
        print(f"| {pipeline:^14} | {latency:>9.2f} ms |")
        print("|----------------|--------------|")
        print(f"Throughput: {throughput:.2f} image/s")

        perf_data = {
            "latency_clip": latency_clip,
            "latency_unet": latency_unet,
            "latency_vae": latency_vae,
            "latency_pil": latency_pil,
            "latency": latency,
            "throughput": throughput,
        }
        if vae_enc:
            perf_data["latency_vae_encoder"] = latency_vae_encoder
        return perf_data

    @staticmethod
    def pt_to_pil(images):
        images = (
            ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
        )
        return [Image.fromarray(images[i]) for i in range(images.shape[0])]

    @staticmethod
    def pt_to_numpy(images: torch.FloatTensor):
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        return ((images + 1) / 2).clamp(0, 1).detach().permute(0, 2, 3, 1).float().cpu().numpy()

    def metadata(self) -> dict[str, Any]:
        data = {
            "actual_steps": self.actual_steps,
            "seed": self.get_current_seed(),
            "name": self.pipeline_info.name(),
            "custom_vae": self.pipeline_info.custom_fp16_vae(),
            "custom_unet": self.pipeline_info.custom_unet(),
        }

        if self.engine_type == EngineType.ORT_CUDA:
            for engine_name, engine in self.backend.engines.items():
                data.update(engine.metadata(engine_name))

        return data

    def save_images(self, images: list, prompt: list[str], negative_prompt: list[str], metadata: dict[str, Any]):
        session_id = str(random.randint(1000, 9999))
        for i, image in enumerate(images):
            seed = str(self.get_current_seed())
            prefix = "".join(x for x in prompt[i] if x.isalnum() or x in ", -").replace(" ", "_")[:20]
            parts = [prefix, session_id, str(i + 1), str(seed), self.current_scheduler, str(self.actual_steps)]
            image_path = os.path.join(self.output_dir, "-".join(parts) + ".png")
            print(f"Saving image {i + 1} / {len(images)} to: {image_path}")

            from PIL import PngImagePlugin

            info = PngImagePlugin.PngInfo()
            for k, v in metadata.items():
                info.add_text(k, str(v))
            info.add_text("prompt", prompt[i])
            info.add_text("negative_prompt", negative_prompt[i])

            image.save(image_path, "PNG", pnginfo=info)

    def _infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        denoising_steps=30,
        guidance=5.0,
        seed=None,
        image=None,
        strength=0.3,
        controlnet_images=None,
        controlnet_scales=None,
        show_latency=False,
        output_type="pil",
    ):
        if show_latency:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

        assert len(prompt) == len(negative_prompt)
        batch_size = len(prompt)

        self.set_denoising_steps(denoising_steps)
        self.set_random_seed(seed)

        timesteps = None
        step_offset = 0
        with torch.inference_mode(), torch.autocast("cuda"):
            if image is not None:
                timesteps, step_offset, latents = self.initialize_refiner(
                    batch_size=batch_size,
                    image=image,
                    strength=strength,
                )
            else:
                # Pre-initialize latents
                latents = self.initialize_latents(
                    batch_size=batch_size,
                    unet_channels=4,
                    latent_height=(image_height // 8),
                    latent_width=(image_width // 8),
                )

            do_classifier_free_guidance = guidance > 1.0
            if not self.pipeline_info.is_xl():
                denoiser = "unet"
                text_embeddings = self.encode_prompt(
                    prompt,
                    negative_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    dtype=latents.dtype,
                )
                add_kwargs = {}
            else:
                denoiser = "unetxl"

                # Time embeddings
                original_size = (image_height, image_width)
                crops_coords_top_left = (0, 0)
                target_size = (image_height, image_width)
                aesthetic_score = 6.0
                negative_aesthetic_score = 2.5
                add_time_ids, add_negative_time_ids = self._get_add_time_ids(
                    original_size,
                    crops_coords_top_left,
                    target_size,
                    aesthetic_score,
                    negative_aesthetic_score,
                    dtype=latents.dtype,
                    requires_aesthetics_score=self.pipeline_info.is_xl_refiner(),
                )
                if do_classifier_free_guidance:
                    add_time_ids = torch.cat([add_negative_time_ids, add_time_ids], dim=0)
                add_time_ids = add_time_ids.to(device=self.device).repeat(batch_size, 1)

                if self.pipeline_info.is_xl_refiner():
                    # CLIP text encoder 2
                    text_embeddings, pooled_embeddings2 = self.encode_prompt(
                        prompt,
                        negative_prompt,
                        encoder="clip2",
                        tokenizer=self.tokenizer2,
                        pooled_outputs=True,
                        output_hidden_states=True,
                        dtype=latents.dtype,
                    )
                    add_kwargs = {"text_embeds": pooled_embeddings2, "time_ids": add_time_ids}
                else:  # XL Base
                    # CLIP text encoder
                    text_embeddings = self.encode_prompt(
                        prompt,
                        negative_prompt,
                        encoder="clip",
                        tokenizer=self.tokenizer,
                        output_hidden_states=True,
                        force_zeros_for_empty_prompt=True,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        dtype=latents.dtype,
                    )
                    # CLIP text encoder 2
                    text_embeddings2, pooled_embeddings2 = self.encode_prompt(
                        prompt,
                        negative_prompt,
                        encoder="clip2",
                        tokenizer=self.tokenizer2,
                        pooled_outputs=True,
                        output_hidden_states=True,
                        force_zeros_for_empty_prompt=True,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        dtype=latents.dtype,
                    )

                    # Merged text embeddings
                    text_embeddings = torch.cat([text_embeddings, text_embeddings2], dim=-1)

                    add_kwargs = {"text_embeds": pooled_embeddings2, "time_ids": add_time_ids}

            if self.pipeline_info.controlnet:
                controlnet_images = self.preprocess_controlnet_images(
                    latents.shape[0],
                    controlnet_images,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    height=image_height,
                    width=image_width,
                )
                add_kwargs.update(
                    {
                        "controlnet_images": controlnet_images,
                        "controlnet_scales": controlnet_scales.to(controlnet_images.dtype).to(controlnet_images.device),
                    }
                )

            # UNet denoiser
            latents = self.denoise_latent(
                latents,
                text_embeddings,
                timesteps=timesteps,
                step_offset=step_offset,
                denoiser=denoiser,
                guidance=guidance,
                add_kwargs=add_kwargs,
            )

        with torch.inference_mode():
            # VAE decode latent
            if output_type == "latent":
                images = latents
            else:
                images = self.decode_latent(latents / self.vae_scaling_factor)
                if output_type == "pil":
                    self.start_profile("pil", color="green")
                    images = self.pt_to_pil(images)
                    self.stop_profile("pil")

        perf_data = None
        if show_latency:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            perf_data = self.print_summary(
                start_time, end_time, batch_size, vae_enc=self.pipeline_info.is_xl_refiner(), pil=(output_type == "pil")
            )

        return images, perf_data

    def run(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        image_height: int,
        image_width: int,
        denoising_steps: int = 30,
        guidance: float = 5.0,
        seed: int | None = None,
        image: torch.Tensor | None = None,
        strength: float = 0.3,
        controlnet_images: torch.Tensor | None = None,
        controlnet_scales: torch.Tensor | None = None,
        show_latency: bool = False,
        output_type: str = "pil",
        deterministic: bool = False,
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (List[str]):
                The text prompt to guide image generation.
            negative_prompt (List[str]):
                The prompt not to guide the image generation.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            denoising_steps (int):
                Number of denoising steps. More steps usually lead to higher quality image at the expense of slower inference.
            guidance (float):
                Higher guidance scale encourages to generate images that are closely linked to the text prompt.
            seed (int):
                Seed for the random generator
            image (tuple[torch.Tensor]):
                Reference image.
            strength (float):
                Indicates extent to transform the reference image, which is used as a starting point,
                and more noise is added the higher the strength.
            show_latency (bool):
                Whether return latency data.
            output_type (str):
                It can be "latent", "pt" or "pil".
        """
        if deterministic:
            torch.use_deterministic_algorithms(True)

        if self.is_backend_tensorrt():
            import tensorrt as trt
            from trt_utilities import TRT_LOGGER

            with trt.Runtime(TRT_LOGGER):
                return self._infer(
                    prompt,
                    negative_prompt,
                    image_height,
                    image_width,
                    denoising_steps=denoising_steps,
                    guidance=guidance,
                    seed=seed,
                    image=image,
                    strength=strength,
                    controlnet_images=controlnet_images,
                    controlnet_scales=controlnet_scales,
                    show_latency=show_latency,
                    output_type=output_type,
                )
        else:
            return self._infer(
                prompt,
                negative_prompt,
                image_height,
                image_width,
                denoising_steps=denoising_steps,
                guidance=guidance,
                seed=seed,
                image=image,
                strength=strength,
                controlnet_images=controlnet_images,
                controlnet_scales=controlnet_scales,
                show_latency=show_latency,
                output_type=output_type,
            )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\sessions.py ===
"""
requests.sessions
~~~~~~~~~~~~~~~~~

This module provides a Session object to manage and persist settings across
requests (cookies, auth, proxies).
"""
import os
import sys
import time
from collections import OrderedDict
from datetime import timedelta

from ._internal_utils import to_native_string
from .adapters import HTTPAdapter
from .auth import _basic_auth_str
from .compat import Mapping, cookielib, urljoin, urlparse
from .cookies import (
    RequestsCookieJar,
    cookiejar_from_dict,
    extract_cookies_to_jar,
    merge_cookies,
)
from .exceptions import (
    ChunkedEncodingError,
    ContentDecodingError,
    InvalidSchema,
    TooManyRedirects,
)
from .hooks import default_hooks, dispatch_hook

# formerly defined here, reexposed here for backward compatibility
from .models import (  # noqa: F401
    DEFAULT_REDIRECT_LIMIT,
    REDIRECT_STATI,
    PreparedRequest,
    Request,
)
from .status_codes import codes
from .structures import CaseInsensitiveDict
from .utils import (  # noqa: F401
    DEFAULT_PORTS,
    default_headers,
    get_auth_from_url,
    get_environ_proxies,
    get_netrc_auth,
    requote_uri,
    resolve_proxies,
    rewind_body,
    should_bypass_proxies,
    to_key_val_list,
)

# Preferred clock, based on which one is more accurate on a given system.
if sys.platform == "win32":
    preferred_clock = time.perf_counter
else:
    preferred_clock = time.time


def merge_setting(request_setting, session_setting, dict_class=OrderedDict):
    """Determines appropriate setting for a given request, taking into account
    the explicit setting on that request, and the setting in the session. If a
    setting is a dictionary, they will be merged together using `dict_class`
    """

    if session_setting is None:
        return request_setting

    if request_setting is None:
        return session_setting

    # Bypass if not a dictionary (e.g. verify)
    if not (
        isinstance(session_setting, Mapping) and isinstance(request_setting, Mapping)
    ):
        return request_setting

    merged_setting = dict_class(to_key_val_list(session_setting))
    merged_setting.update(to_key_val_list(request_setting))

    # Remove keys that are set to None. Extract keys first to avoid altering
    # the dictionary during iteration.
    none_keys = [k for (k, v) in merged_setting.items() if v is None]
    for key in none_keys:
        del merged_setting[key]

    return merged_setting


def merge_hooks(request_hooks, session_hooks, dict_class=OrderedDict):
    """Properly merges both requests and session hooks.

    This is necessary because when request_hooks == {'response': []}, the
    merge breaks Session hooks entirely.
    """
    if session_hooks is None or session_hooks.get("response") == []:
        return request_hooks

    if request_hooks is None or request_hooks.get("response") == []:
        return session_hooks

    return merge_setting(request_hooks, session_hooks, dict_class)


class SessionRedirectMixin:
    def get_redirect_target(self, resp):
        """Receives a Response. Returns a redirect URI or ``None``"""
        # Due to the nature of how requests processes redirects this method will
        # be called at least once upon the original response and at least twice
        # on each subsequent redirect response (if any).
        # If a custom mixin is used to handle this logic, it may be advantageous
        # to cache the redirect location onto the response object as a private
        # attribute.
        if resp.is_redirect:
            location = resp.headers["location"]
            # Currently the underlying http module on py3 decode headers
            # in latin1, but empirical evidence suggests that latin1 is very
            # rarely used with non-ASCII characters in HTTP headers.
            # It is more likely to get UTF8 header rather than latin1.
            # This causes incorrect handling of UTF8 encoded location headers.
            # To solve this, we re-encode the location in latin1.
            location = location.encode("latin1")
            return to_native_string(location, "utf8")
        return None

    def should_strip_auth(self, old_url, new_url):
        """Decide whether Authorization header should be removed when redirecting"""
        old_parsed = urlparse(old_url)
        new_parsed = urlparse(new_url)
        if old_parsed.hostname != new_parsed.hostname:
            return True
        # Special case: allow http -> https redirect when using the standard
        # ports. This isn't specified by RFC 7235, but is kept to avoid
        # breaking backwards compatibility with older versions of requests
        # that allowed any redirects on the same host.
        if (
            old_parsed.scheme == "http"
            and old_parsed.port in (80, None)
            and new_parsed.scheme == "https"
            and new_parsed.port in (443, None)
        ):
            return False

        # Handle default port usage corresponding to scheme.
        changed_port = old_parsed.port != new_parsed.port
        changed_scheme = old_parsed.scheme != new_parsed.scheme
        default_port = (DEFAULT_PORTS.get(old_parsed.scheme, None), None)
        if (
            not changed_scheme
            and old_parsed.port in default_port
            and new_parsed.port in default_port
        ):
            return False

        # Standard case: root URI must match
        return changed_port or changed_scheme

    def resolve_redirects(
        self,
        resp,
        req,
        stream=False,
        timeout=None,
        verify=True,
        cert=None,
        proxies=None,
        yield_requests=False,
        **adapter_kwargs,
    ):
        """Receives a Response. Returns a generator of Responses or Requests."""

        hist = []  # keep track of history

        url = self.get_redirect_target(resp)
        previous_fragment = urlparse(req.url).fragment
        while url:
            prepared_request = req.copy()

            # Update history and keep track of redirects.
            # resp.history must ignore the original request in this loop
            hist.append(resp)
            resp.history = hist[1:]

            try:
                resp.content  # Consume socket so it can be released
            except (ChunkedEncodingError, ContentDecodingError, RuntimeError):
                resp.raw.read(decode_content=False)

            if len(resp.history) >= self.max_redirects:
                raise TooManyRedirects(
                    f"Exceeded {self.max_redirects} redirects.", response=resp
                )

            # Release the connection back into the pool.
            resp.close()

            # Handle redirection without scheme (see: RFC 1808 Section 4)
            if url.startswith("//"):
                parsed_rurl = urlparse(resp.url)
                url = ":".join([to_native_string(parsed_rurl.scheme), url])

            # Normalize url case and attach previous fragment if needed (RFC 7231 7.1.2)
            parsed = urlparse(url)
            if parsed.fragment == "" and previous_fragment:
                parsed = parsed._replace(fragment=previous_fragment)
            elif parsed.fragment:
                previous_fragment = parsed.fragment
            url = parsed.geturl()

            # Facilitate relative 'location' headers, as allowed by RFC 7231.
            # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
            # Compliant with RFC3986, we percent encode the url.
            if not parsed.netloc:
                url = urljoin(resp.url, requote_uri(url))
            else:
                url = requote_uri(url)

            prepared_request.url = to_native_string(url)

            self.rebuild_method(prepared_request, resp)

            # https://github.com/psf/requests/issues/1084
            if resp.status_code not in (
                codes.temporary_redirect,
                codes.permanent_redirect,
            ):
                # https://github.com/psf/requests/issues/3490
                purged_headers = ("Content-Length", "Content-Type", "Transfer-Encoding")
                for header in purged_headers:
                    prepared_request.headers.pop(header, None)
                prepared_request.body = None

            headers = prepared_request.headers
            headers.pop("Cookie", None)

            # Extract any cookies sent on the response to the cookiejar
            # in the new request. Because we've mutated our copied prepared
            # request, use the old one that we haven't yet touched.
            extract_cookies_to_jar(prepared_request._cookies, req, resp.raw)
            merge_cookies(prepared_request._cookies, self.cookies)
            prepared_request.prepare_cookies(prepared_request._cookies)

            # Rebuild auth and proxy information.
            proxies = self.rebuild_proxies(prepared_request, proxies)
            self.rebuild_auth(prepared_request, resp)

            # A failed tell() sets `_body_position` to `object()`. This non-None
            # value ensures `rewindable` will be True, allowing us to raise an
            # UnrewindableBodyError, instead of hanging the connection.
            rewindable = prepared_request._body_position is not None and (
                "Content-Length" in headers or "Transfer-Encoding" in headers
            )

            # Attempt to rewind consumed file-like object.
            if rewindable:
                rewind_body(prepared_request)

            # Override the original request.
            req = prepared_request

            if yield_requests:
                yield req
            else:
                resp = self.send(
                    req,
                    stream=stream,
                    timeout=timeout,
                    verify=verify,
                    cert=cert,
                    proxies=proxies,
                    allow_redirects=False,
                    **adapter_kwargs,
                )

                extract_cookies_to_jar(self.cookies, prepared_request, resp.raw)

                # extract redirect url, if any, for the next loop
                url = self.get_redirect_target(resp)
                yield resp

    def rebuild_auth(self, prepared_request, response):
        """When being redirected we may want to strip authentication from the
        request to avoid leaking credentials. This method intelligently removes
        and reapplies authentication where possible to avoid credential loss.
        """
        headers = prepared_request.headers
        url = prepared_request.url

        if "Authorization" in headers and self.should_strip_auth(
            response.request.url, url
        ):
            # If we get redirected to a new host, we should strip out any
            # authentication headers.
            del headers["Authorization"]

        # .netrc might have more auth for us on our new host.
        new_auth = get_netrc_auth(url) if self.trust_env else None
        if new_auth is not None:
            prepared_request.prepare_auth(new_auth)

    def rebuild_proxies(self, prepared_request, proxies):
        """This method re-evaluates the proxy configuration by considering the
        environment variables. If we are redirected to a URL covered by
        NO_PROXY, we strip the proxy configuration. Otherwise, we set missing
        proxy keys for this URL (in case they were stripped by a previous
        redirect).

        This method also replaces the Proxy-Authorization header where
        necessary.

        :rtype: dict
        """
        headers = prepared_request.headers
        scheme = urlparse(prepared_request.url).scheme
        new_proxies = resolve_proxies(prepared_request, proxies, self.trust_env)

        if "Proxy-Authorization" in headers:
            del headers["Proxy-Authorization"]

        try:
            username, password = get_auth_from_url(new_proxies[scheme])
        except KeyError:
            username, password = None, None

        # urllib3 handles proxy authorization for us in the standard adapter.
        # Avoid appending this to TLS tunneled requests where it may be leaked.
        if not scheme.startswith("https") and username and password:
            headers["Proxy-Authorization"] = _basic_auth_str(username, password)

        return new_proxies

    def rebuild_method(self, prepared_request, response):
        """When being redirected we may want to change the method of the request
        based on certain specs or browser behavior.
        """
        method = prepared_request.method

        # https://tools.ietf.org/html/rfc7231#section-6.4.4
        if response.status_code == codes.see_other and method != "HEAD":
            method = "GET"

        # Do what the browsers do, despite standards...
        # First, turn 302s into GETs.
        if response.status_code == codes.found and method != "HEAD":
            method = "GET"

        # Second, if a POST is responded to with a 301, turn it into a GET.
        # This bizarre behaviour is explained in Issue 1704.
        if response.status_code == codes.moved and method == "POST":
            method = "GET"

        prepared_request.method = method


class Session(SessionRedirectMixin):
    """A Requests session.

    Provides cookie persistence, connection-pooling, and configuration.

    Basic Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> s.get('https://httpbin.org/get')
      <Response [200]>

    Or as a context manager::

      >>> with requests.Session() as s:
      ...     s.get('https://httpbin.org/get')
      <Response [200]>
    """

    __attrs__ = [
        "headers",
        "cookies",
        "auth",
        "proxies",
        "hooks",
        "params",
        "verify",
        "cert",
        "adapters",
        "stream",
        "trust_env",
        "max_redirects",
    ]

    def __init__(self):
        #: A case-insensitive dictionary of headers to be sent on each
        #: :class:`Request <Request>` sent from this
        #: :class:`Session <Session>`.
        self.headers = default_headers()

        #: Default Authentication tuple or object to attach to
        #: :class:`Request <Request>`.
        self.auth = None

        #: Dictionary mapping protocol or protocol and host to the URL of the proxy
        #: (e.g. {'http': 'foo.bar:3128', 'http://host.name': 'foo.bar:4012'}) to
        #: be used on each :class:`Request <Request>`.
        self.proxies = {}

        #: Event-handling hooks.
        self.hooks = default_hooks()

        #: Dictionary of querystring data to attach to each
        #: :class:`Request <Request>`. The dictionary values may be lists for
        #: representing multivalued query parameters.
        self.params = {}

        #: Stream response content default.
        self.stream = False

        #: SSL Verification default.
        #: Defaults to `True`, requiring requests to verify the TLS certificate at the
        #: remote end.
        #: If verify is set to `False`, requests will accept any TLS certificate
        #: presented by the server, and will ignore hostname mismatches and/or
        #: expired certificates, which will make your application vulnerable to
        #: man-in-the-middle (MitM) attacks.
        #: Only set this to `False` for testing.
        self.verify = True

        #: SSL client certificate default, if String, path to ssl client
        #: cert file (.pem). If Tuple, ('cert', 'key') pair.
        self.cert = None

        #: Maximum number of redirects allowed. If the request exceeds this
        #: limit, a :class:`TooManyRedirects` exception is raised.
        #: This defaults to requests.models.DEFAULT_REDIRECT_LIMIT, which is
        #: 30.
        self.max_redirects = DEFAULT_REDIRECT_LIMIT

        #: Trust environment settings for proxy configuration, default
        #: authentication and similar.
        self.trust_env = True

        #: A CookieJar containing all currently outstanding cookies set on this
        #: session. By default it is a
        #: :class:`RequestsCookieJar <requests.cookies.RequestsCookieJar>`, but
        #: may be any other ``cookielib.CookieJar`` compatible object.
        self.cookies = cookiejar_from_dict({})

        # Default connection adapters.
        self.adapters = OrderedDict()
        self.mount("https://", HTTPAdapter())
        self.mount("http://", HTTPAdapter())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def prepare_request(self, request):
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for
        transmission and returns it. The :class:`PreparedRequest` has settings
        merged from the :class:`Request <Request>` instance and those of the
        :class:`Session`.

        :param request: :class:`Request` instance to prepare with this
            session's settings.
        :rtype: requests.PreparedRequest
        """
        cookies = request.cookies or {}

        # Bootstrap CookieJar.
        if not isinstance(cookies, cookielib.CookieJar):
            cookies = cookiejar_from_dict(cookies)

        # Merge with session cookies
        merged_cookies = merge_cookies(
            merge_cookies(RequestsCookieJar(), self.cookies), cookies
        )

        # Set environment's basic authentication if not explicitly set.
        auth = request.auth
        if self.trust_env and not auth and not self.auth:
            auth = get_netrc_auth(request.url)

        p = PreparedRequest()
        p.prepare(
            method=request.method.upper(),
            url=request.url,
            files=request.files,
            data=request.data,
            json=request.json,
            headers=merge_setting(
                request.headers, self.headers, dict_class=CaseInsensitiveDict
            ),
            params=merge_setting(request.params, self.params),
            auth=merge_setting(auth, self.auth),
            cookies=merged_cookies,
            hooks=merge_hooks(request.hooks, self.hooks),
        )
        return p

    def request(
        self,
        method,
        url,
        params=None,
        data=None,
        headers=None,
        cookies=None,
        files=None,
        auth=None,
        timeout=None,
        allow_redirects=True,
        proxies=None,
        hooks=None,
        stream=None,
        verify=None,
        cert=None,
        json=None,
    ):
        """Constructs a :class:`Request <Request>`, prepares it and sends it.
        Returns :class:`Response <Response>` object.

        :param method: method for the new :class:`Request` object.
        :param url: URL for the new :class:`Request` object.
        :param params: (optional) Dictionary or bytes to be sent in the query
            string for the :class:`Request`.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the
            :class:`Request`.
        :param headers: (optional) Dictionary of HTTP Headers to send with the
            :class:`Request`.
        :param cookies: (optional) Dict or CookieJar object to send with the
            :class:`Request`.
        :param files: (optional) Dictionary of ``'filename': file-like-objects``
            for multipart encoding upload.
        :param auth: (optional) Auth tuple or callable to enable
            Basic/Digest/Custom HTTP Auth.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param allow_redirects: (optional) Set to True by default.
        :type allow_redirects: bool
        :param proxies: (optional) Dictionary mapping protocol or protocol and
            hostname to the URL of the proxy.
        :param hooks: (optional) Dictionary mapping hook name to one event or
            list of events, event must be callable.
        :param stream: (optional) whether to immediately download the response
            content. Defaults to ``False``.
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use. Defaults to ``True``. When set to
            ``False``, requests will accept any TLS certificate presented by
            the server, and will ignore hostname mismatches and/or expired
            certificates, which will make your application vulnerable to
            man-in-the-middle (MitM) attacks. Setting verify to ``False``
            may be useful during local development or testing.
        :param cert: (optional) if String, path to ssl client cert file (.pem).
            If Tuple, ('cert', 'key') pair.
        :rtype: requests.Response
        """
        # Create the Request.
        req = Request(
            method=method.upper(),
            url=url,
            headers=headers,
            files=files,
            data=data or {},
            json=json,
            params=params or {},
            auth=auth,
            cookies=cookies,
            hooks=hooks,
        )
        prep = self.prepare_request(req)

        proxies = proxies or {}

        settings = self.merge_environment_settings(
            prep.url, proxies, stream, verify, cert
        )

        # Send the request.
        send_kwargs = {
            "timeout": timeout,
            "allow_redirects": allow_redirects,
        }
        send_kwargs.update(settings)
        resp = self.send(prep, **send_kwargs)

        return resp

    def get(self, url, **kwargs):
        r"""Sends a GET request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", True)
        return self.request("GET", url, **kwargs)

    def options(self, url, **kwargs):
        r"""Sends a OPTIONS request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", True)
        return self.request("OPTIONS", url, **kwargs)

    def head(self, url, **kwargs):
        r"""Sends a HEAD request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", False)
        return self.request("HEAD", url, **kwargs)

    def post(self, url, data=None, json=None, **kwargs):
        r"""Sends a POST request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("POST", url, data=data, json=json, **kwargs)

    def put(self, url, data=None, **kwargs):
        r"""Sends a PUT request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("PUT", url, data=data, **kwargs)

    def patch(self, url, data=None, **kwargs):
        r"""Sends a PATCH request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("PATCH", url, data=data, **kwargs)

    def delete(self, url, **kwargs):
        r"""Sends a DELETE request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("DELETE", url, **kwargs)

    def send(self, request, **kwargs):
        """Send a given PreparedRequest.

        :rtype: requests.Response
        """
        # Set defaults that the hooks can utilize to ensure they always have
        # the correct parameters to reproduce the previous request.
        kwargs.setdefault("stream", self.stream)
        kwargs.setdefault("verify", self.verify)
        kwargs.setdefault("cert", self.cert)
        if "proxies" not in kwargs:
            kwargs["proxies"] = resolve_proxies(request, self.proxies, self.trust_env)

        # It's possible that users might accidentally send a Request object.
        # Guard against that specific failure case.
        if isinstance(request, Request):
            raise ValueError("You can only send PreparedRequests.")

        # Set up variables needed for resolve_redirects and dispatching of hooks
        allow_redirects = kwargs.pop("allow_redirects", True)
        stream = kwargs.get("stream")
        hooks = request.hooks

        # Get the appropriate adapter to use
        adapter = self.get_adapter(url=request.url)

        # Start time (approximately) of the request
        start = preferred_clock()

        # Send the request
        r = adapter.send(request, **kwargs)

        # Total elapsed time of the request (approximately)
        elapsed = preferred_clock() - start
        r.elapsed = timedelta(seconds=elapsed)

        # Response manipulation hooks
        r = dispatch_hook("response", hooks, r, **kwargs)

        # Persist cookies
        if r.history:
            # If the hooks create history then we want those cookies too
            for resp in r.history:
                extract_cookies_to_jar(self.cookies, resp.request, resp.raw)

        extract_cookies_to_jar(self.cookies, request, r.raw)

        # Resolve redirects if allowed.
        if allow_redirects:
            # Redirect resolving generator.
            gen = self.resolve_redirects(r, request, **kwargs)
            history = [resp for resp in gen]
        else:
            history = []

        # Shuffle things around if there's history.
        if history:
            # Insert the first (original) request at the start
            history.insert(0, r)
            # Get the last request made
            r = history.pop()
            r.history = history

        # If redirects aren't being followed, store the response on the Request for Response.next().
        if not allow_redirects:
            try:
                r._next = next(
                    self.resolve_redirects(r, request, yield_requests=True, **kwargs)
                )
            except StopIteration:
                pass

        if not stream:
            r.content

        return r

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        """
        Check the environment and merge it with some settings.

        :rtype: dict
        """
        # Gather clues from the surrounding environment.
        if self.trust_env:
            # Set environment's proxies.
            no_proxy = proxies.get("no_proxy") if proxies is not None else None
            env_proxies = get_environ_proxies(url, no_proxy=no_proxy)
            for k, v in env_proxies.items():
                proxies.setdefault(k, v)

            # Look for requests environment configuration
            # and be compatible with cURL.
            if verify is True or verify is None:
                verify = (
                    os.environ.get("REQUESTS_CA_BUNDLE")
                    or os.environ.get("CURL_CA_BUNDLE")
                    or verify
                )

        # Merge all the kwargs.
        proxies = merge_setting(proxies, self.proxies)
        stream = merge_setting(stream, self.stream)
        verify = merge_setting(verify, self.verify)
        cert = merge_setting(cert, self.cert)

        return {"proxies": proxies, "stream": stream, "verify": verify, "cert": cert}

    def get_adapter(self, url):
        """
        Returns the appropriate connection adapter for the given URL.

        :rtype: requests.adapters.BaseAdapter
        """
        for prefix, adapter in self.adapters.items():
            if url.lower().startswith(prefix.lower()):
                return adapter

        # Nothing matches :-/
        raise InvalidSchema(f"No connection adapters were found for {url!r}")

    def close(self):
        """Closes all adapters and as such the session"""
        for v in self.adapters.values():
            v.close()

    def mount(self, prefix, adapter):
        """Registers a connection adapter to a prefix.

        Adapters are sorted in descending order by prefix length.
        """
        self.adapters[prefix] = adapter
        keys_to_move = [k for k in self.adapters if len(k) < len(prefix)]

        for key in keys_to_move:
            self.adapters[key] = self.adapters.pop(key)

    def __getstate__(self):
        state = {attr: getattr(self, attr, None) for attr in self.__attrs__}
        return state

    def __setstate__(self, state):
        for attr, value in state.items():
            setattr(self, attr, value)


def session():
    """
    Returns a :class:`Session` for context-management.

    .. deprecated:: 1.0.0

        This method has been deprecated since version 1.0.0 and is only kept for
        backwards compatibility. New code should use :class:`~requests.sessions.Session`
        to create a session. This may be removed at a future date.

    :rtype: Session
    """
    return Session()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\sessions.py ===
"""
requests.sessions
~~~~~~~~~~~~~~~~~

This module provides a Session object to manage and persist settings across
requests (cookies, auth, proxies).
"""
import os
import sys
import time
from collections import OrderedDict
from datetime import timedelta

from ._internal_utils import to_native_string
from .adapters import HTTPAdapter
from .auth import _basic_auth_str
from .compat import Mapping, cookielib, urljoin, urlparse
from .cookies import (
    RequestsCookieJar,
    cookiejar_from_dict,
    extract_cookies_to_jar,
    merge_cookies,
)
from .exceptions import (
    ChunkedEncodingError,
    ContentDecodingError,
    InvalidSchema,
    TooManyRedirects,
)
from .hooks import default_hooks, dispatch_hook

# formerly defined here, reexposed here for backward compatibility
from .models import (  # noqa: F401
    DEFAULT_REDIRECT_LIMIT,
    REDIRECT_STATI,
    PreparedRequest,
    Request,
)
from .status_codes import codes
from .structures import CaseInsensitiveDict
from .utils import (  # noqa: F401
    DEFAULT_PORTS,
    default_headers,
    get_auth_from_url,
    get_environ_proxies,
    get_netrc_auth,
    requote_uri,
    resolve_proxies,
    rewind_body,
    should_bypass_proxies,
    to_key_val_list,
)

# Preferred clock, based on which one is more accurate on a given system.
if sys.platform == "win32":
    preferred_clock = time.perf_counter
else:
    preferred_clock = time.time


def merge_setting(request_setting, session_setting, dict_class=OrderedDict):
    """Determines appropriate setting for a given request, taking into account
    the explicit setting on that request, and the setting in the session. If a
    setting is a dictionary, they will be merged together using `dict_class`
    """

    if session_setting is None:
        return request_setting

    if request_setting is None:
        return session_setting

    # Bypass if not a dictionary (e.g. verify)
    if not (
        isinstance(session_setting, Mapping) and isinstance(request_setting, Mapping)
    ):
        return request_setting

    merged_setting = dict_class(to_key_val_list(session_setting))
    merged_setting.update(to_key_val_list(request_setting))

    # Remove keys that are set to None. Extract keys first to avoid altering
    # the dictionary during iteration.
    none_keys = [k for (k, v) in merged_setting.items() if v is None]
    for key in none_keys:
        del merged_setting[key]

    return merged_setting


def merge_hooks(request_hooks, session_hooks, dict_class=OrderedDict):
    """Properly merges both requests and session hooks.

    This is necessary because when request_hooks == {'response': []}, the
    merge breaks Session hooks entirely.
    """
    if session_hooks is None or session_hooks.get("response") == []:
        return request_hooks

    if request_hooks is None or request_hooks.get("response") == []:
        return session_hooks

    return merge_setting(request_hooks, session_hooks, dict_class)


class SessionRedirectMixin:
    def get_redirect_target(self, resp):
        """Receives a Response. Returns a redirect URI or ``None``"""
        # Due to the nature of how requests processes redirects this method will
        # be called at least once upon the original response and at least twice
        # on each subsequent redirect response (if any).
        # If a custom mixin is used to handle this logic, it may be advantageous
        # to cache the redirect location onto the response object as a private
        # attribute.
        if resp.is_redirect:
            location = resp.headers["location"]
            # Currently the underlying http module on py3 decode headers
            # in latin1, but empirical evidence suggests that latin1 is very
            # rarely used with non-ASCII characters in HTTP headers.
            # It is more likely to get UTF8 header rather than latin1.
            # This causes incorrect handling of UTF8 encoded location headers.
            # To solve this, we re-encode the location in latin1.
            location = location.encode("latin1")
            return to_native_string(location, "utf8")
        return None

    def should_strip_auth(self, old_url, new_url):
        """Decide whether Authorization header should be removed when redirecting"""
        old_parsed = urlparse(old_url)
        new_parsed = urlparse(new_url)
        if old_parsed.hostname != new_parsed.hostname:
            return True
        # Special case: allow http -> https redirect when using the standard
        # ports. This isn't specified by RFC 7235, but is kept to avoid
        # breaking backwards compatibility with older versions of requests
        # that allowed any redirects on the same host.
        if (
            old_parsed.scheme == "http"
            and old_parsed.port in (80, None)
            and new_parsed.scheme == "https"
            and new_parsed.port in (443, None)
        ):
            return False

        # Handle default port usage corresponding to scheme.
        changed_port = old_parsed.port != new_parsed.port
        changed_scheme = old_parsed.scheme != new_parsed.scheme
        default_port = (DEFAULT_PORTS.get(old_parsed.scheme, None), None)
        if (
            not changed_scheme
            and old_parsed.port in default_port
            and new_parsed.port in default_port
        ):
            return False

        # Standard case: root URI must match
        return changed_port or changed_scheme

    def resolve_redirects(
        self,
        resp,
        req,
        stream=False,
        timeout=None,
        verify=True,
        cert=None,
        proxies=None,
        yield_requests=False,
        **adapter_kwargs,
    ):
        """Receives a Response. Returns a generator of Responses or Requests."""

        hist = []  # keep track of history

        url = self.get_redirect_target(resp)
        previous_fragment = urlparse(req.url).fragment
        while url:
            prepared_request = req.copy()

            # Update history and keep track of redirects.
            # resp.history must ignore the original request in this loop
            hist.append(resp)
            resp.history = hist[1:]

            try:
                resp.content  # Consume socket so it can be released
            except (ChunkedEncodingError, ContentDecodingError, RuntimeError):
                resp.raw.read(decode_content=False)

            if len(resp.history) >= self.max_redirects:
                raise TooManyRedirects(
                    f"Exceeded {self.max_redirects} redirects.", response=resp
                )

            # Release the connection back into the pool.
            resp.close()

            # Handle redirection without scheme (see: RFC 1808 Section 4)
            if url.startswith("//"):
                parsed_rurl = urlparse(resp.url)
                url = ":".join([to_native_string(parsed_rurl.scheme), url])

            # Normalize url case and attach previous fragment if needed (RFC 7231 7.1.2)
            parsed = urlparse(url)
            if parsed.fragment == "" and previous_fragment:
                parsed = parsed._replace(fragment=previous_fragment)
            elif parsed.fragment:
                previous_fragment = parsed.fragment
            url = parsed.geturl()

            # Facilitate relative 'location' headers, as allowed by RFC 7231.
            # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
            # Compliant with RFC3986, we percent encode the url.
            if not parsed.netloc:
                url = urljoin(resp.url, requote_uri(url))
            else:
                url = requote_uri(url)

            prepared_request.url = to_native_string(url)

            self.rebuild_method(prepared_request, resp)

            # https://github.com/psf/requests/issues/1084
            if resp.status_code not in (
                codes.temporary_redirect,
                codes.permanent_redirect,
            ):
                # https://github.com/psf/requests/issues/3490
                purged_headers = ("Content-Length", "Content-Type", "Transfer-Encoding")
                for header in purged_headers:
                    prepared_request.headers.pop(header, None)
                prepared_request.body = None

            headers = prepared_request.headers
            headers.pop("Cookie", None)

            # Extract any cookies sent on the response to the cookiejar
            # in the new request. Because we've mutated our copied prepared
            # request, use the old one that we haven't yet touched.
            extract_cookies_to_jar(prepared_request._cookies, req, resp.raw)
            merge_cookies(prepared_request._cookies, self.cookies)
            prepared_request.prepare_cookies(prepared_request._cookies)

            # Rebuild auth and proxy information.
            proxies = self.rebuild_proxies(prepared_request, proxies)
            self.rebuild_auth(prepared_request, resp)

            # A failed tell() sets `_body_position` to `object()`. This non-None
            # value ensures `rewindable` will be True, allowing us to raise an
            # UnrewindableBodyError, instead of hanging the connection.
            rewindable = prepared_request._body_position is not None and (
                "Content-Length" in headers or "Transfer-Encoding" in headers
            )

            # Attempt to rewind consumed file-like object.
            if rewindable:
                rewind_body(prepared_request)

            # Override the original request.
            req = prepared_request

            if yield_requests:
                yield req
            else:
                resp = self.send(
                    req,
                    stream=stream,
                    timeout=timeout,
                    verify=verify,
                    cert=cert,
                    proxies=proxies,
                    allow_redirects=False,
                    **adapter_kwargs,
                )

                extract_cookies_to_jar(self.cookies, prepared_request, resp.raw)

                # extract redirect url, if any, for the next loop
                url = self.get_redirect_target(resp)
                yield resp

    def rebuild_auth(self, prepared_request, response):
        """When being redirected we may want to strip authentication from the
        request to avoid leaking credentials. This method intelligently removes
        and reapplies authentication where possible to avoid credential loss.
        """
        headers = prepared_request.headers
        url = prepared_request.url

        if "Authorization" in headers and self.should_strip_auth(
            response.request.url, url
        ):
            # If we get redirected to a new host, we should strip out any
            # authentication headers.
            del headers["Authorization"]

        # .netrc might have more auth for us on our new host.
        new_auth = get_netrc_auth(url) if self.trust_env else None
        if new_auth is not None:
            prepared_request.prepare_auth(new_auth)

    def rebuild_proxies(self, prepared_request, proxies):
        """This method re-evaluates the proxy configuration by considering the
        environment variables. If we are redirected to a URL covered by
        NO_PROXY, we strip the proxy configuration. Otherwise, we set missing
        proxy keys for this URL (in case they were stripped by a previous
        redirect).

        This method also replaces the Proxy-Authorization header where
        necessary.

        :rtype: dict
        """
        headers = prepared_request.headers
        scheme = urlparse(prepared_request.url).scheme
        new_proxies = resolve_proxies(prepared_request, proxies, self.trust_env)

        if "Proxy-Authorization" in headers:
            del headers["Proxy-Authorization"]

        try:
            username, password = get_auth_from_url(new_proxies[scheme])
        except KeyError:
            username, password = None, None

        # urllib3 handles proxy authorization for us in the standard adapter.
        # Avoid appending this to TLS tunneled requests where it may be leaked.
        if not scheme.startswith("https") and username and password:
            headers["Proxy-Authorization"] = _basic_auth_str(username, password)

        return new_proxies

    def rebuild_method(self, prepared_request, response):
        """When being redirected we may want to change the method of the request
        based on certain specs or browser behavior.
        """
        method = prepared_request.method

        # https://tools.ietf.org/html/rfc7231#section-6.4.4
        if response.status_code == codes.see_other and method != "HEAD":
            method = "GET"

        # Do what the browsers do, despite standards...
        # First, turn 302s into GETs.
        if response.status_code == codes.found and method != "HEAD":
            method = "GET"

        # Second, if a POST is responded to with a 301, turn it into a GET.
        # This bizarre behaviour is explained in Issue 1704.
        if response.status_code == codes.moved and method == "POST":
            method = "GET"

        prepared_request.method = method


class Session(SessionRedirectMixin):
    """A Requests session.

    Provides cookie persistence, connection-pooling, and configuration.

    Basic Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> s.get('https://httpbin.org/get')
      <Response [200]>

    Or as a context manager::

      >>> with requests.Session() as s:
      ...     s.get('https://httpbin.org/get')
      <Response [200]>
    """

    __attrs__ = [
        "headers",
        "cookies",
        "auth",
        "proxies",
        "hooks",
        "params",
        "verify",
        "cert",
        "adapters",
        "stream",
        "trust_env",
        "max_redirects",
    ]

    def __init__(self):
        #: A case-insensitive dictionary of headers to be sent on each
        #: :class:`Request <Request>` sent from this
        #: :class:`Session <Session>`.
        self.headers = default_headers()

        #: Default Authentication tuple or object to attach to
        #: :class:`Request <Request>`.
        self.auth = None

        #: Dictionary mapping protocol or protocol and host to the URL of the proxy
        #: (e.g. {'http': 'foo.bar:3128', 'http://host.name': 'foo.bar:4012'}) to
        #: be used on each :class:`Request <Request>`.
        self.proxies = {}

        #: Event-handling hooks.
        self.hooks = default_hooks()

        #: Dictionary of querystring data to attach to each
        #: :class:`Request <Request>`. The dictionary values may be lists for
        #: representing multivalued query parameters.
        self.params = {}

        #: Stream response content default.
        self.stream = False

        #: SSL Verification default.
        #: Defaults to `True`, requiring requests to verify the TLS certificate at the
        #: remote end.
        #: If verify is set to `False`, requests will accept any TLS certificate
        #: presented by the server, and will ignore hostname mismatches and/or
        #: expired certificates, which will make your application vulnerable to
        #: man-in-the-middle (MitM) attacks.
        #: Only set this to `False` for testing.
        self.verify = True

        #: SSL client certificate default, if String, path to ssl client
        #: cert file (.pem). If Tuple, ('cert', 'key') pair.
        self.cert = None

        #: Maximum number of redirects allowed. If the request exceeds this
        #: limit, a :class:`TooManyRedirects` exception is raised.
        #: This defaults to requests.models.DEFAULT_REDIRECT_LIMIT, which is
        #: 30.
        self.max_redirects = DEFAULT_REDIRECT_LIMIT

        #: Trust environment settings for proxy configuration, default
        #: authentication and similar.
        self.trust_env = True

        #: A CookieJar containing all currently outstanding cookies set on this
        #: session. By default it is a
        #: :class:`RequestsCookieJar <requests.cookies.RequestsCookieJar>`, but
        #: may be any other ``cookielib.CookieJar`` compatible object.
        self.cookies = cookiejar_from_dict({})

        # Default connection adapters.
        self.adapters = OrderedDict()
        self.mount("https://", HTTPAdapter())
        self.mount("http://", HTTPAdapter())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def prepare_request(self, request):
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for
        transmission and returns it. The :class:`PreparedRequest` has settings
        merged from the :class:`Request <Request>` instance and those of the
        :class:`Session`.

        :param request: :class:`Request` instance to prepare with this
            session's settings.
        :rtype: requests.PreparedRequest
        """
        cookies = request.cookies or {}

        # Bootstrap CookieJar.
        if not isinstance(cookies, cookielib.CookieJar):
            cookies = cookiejar_from_dict(cookies)

        # Merge with session cookies
        merged_cookies = merge_cookies(
            merge_cookies(RequestsCookieJar(), self.cookies), cookies
        )

        # Set environment's basic authentication if not explicitly set.
        auth = request.auth
        if self.trust_env and not auth and not self.auth:
            auth = get_netrc_auth(request.url)

        p = PreparedRequest()
        p.prepare(
            method=request.method.upper(),
            url=request.url,
            files=request.files,
            data=request.data,
            json=request.json,
            headers=merge_setting(
                request.headers, self.headers, dict_class=CaseInsensitiveDict
            ),
            params=merge_setting(request.params, self.params),
            auth=merge_setting(auth, self.auth),
            cookies=merged_cookies,
            hooks=merge_hooks(request.hooks, self.hooks),
        )
        return p

    def request(
        self,
        method,
        url,
        params=None,
        data=None,
        headers=None,
        cookies=None,
        files=None,
        auth=None,
        timeout=None,
        allow_redirects=True,
        proxies=None,
        hooks=None,
        stream=None,
        verify=None,
        cert=None,
        json=None,
    ):
        """Constructs a :class:`Request <Request>`, prepares it and sends it.
        Returns :class:`Response <Response>` object.

        :param method: method for the new :class:`Request` object.
        :param url: URL for the new :class:`Request` object.
        :param params: (optional) Dictionary or bytes to be sent in the query
            string for the :class:`Request`.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the
            :class:`Request`.
        :param headers: (optional) Dictionary of HTTP Headers to send with the
            :class:`Request`.
        :param cookies: (optional) Dict or CookieJar object to send with the
            :class:`Request`.
        :param files: (optional) Dictionary of ``'filename': file-like-objects``
            for multipart encoding upload.
        :param auth: (optional) Auth tuple or callable to enable
            Basic/Digest/Custom HTTP Auth.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param allow_redirects: (optional) Set to True by default.
        :type allow_redirects: bool
        :param proxies: (optional) Dictionary mapping protocol or protocol and
            hostname to the URL of the proxy.
        :param hooks: (optional) Dictionary mapping hook name to one event or
            list of events, event must be callable.
        :param stream: (optional) whether to immediately download the response
            content. Defaults to ``False``.
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use. Defaults to ``True``. When set to
            ``False``, requests will accept any TLS certificate presented by
            the server, and will ignore hostname mismatches and/or expired
            certificates, which will make your application vulnerable to
            man-in-the-middle (MitM) attacks. Setting verify to ``False``
            may be useful during local development or testing.
        :param cert: (optional) if String, path to ssl client cert file (.pem).
            If Tuple, ('cert', 'key') pair.
        :rtype: requests.Response
        """
        # Create the Request.
        req = Request(
            method=method.upper(),
            url=url,
            headers=headers,
            files=files,
            data=data or {},
            json=json,
            params=params or {},
            auth=auth,
            cookies=cookies,
            hooks=hooks,
        )
        prep = self.prepare_request(req)

        proxies = proxies or {}

        settings = self.merge_environment_settings(
            prep.url, proxies, stream, verify, cert
        )

        # Send the request.
        send_kwargs = {
            "timeout": timeout,
            "allow_redirects": allow_redirects,
        }
        send_kwargs.update(settings)
        resp = self.send(prep, **send_kwargs)

        return resp

    def get(self, url, **kwargs):
        r"""Sends a GET request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", True)
        return self.request("GET", url, **kwargs)

    def options(self, url, **kwargs):
        r"""Sends a OPTIONS request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", True)
        return self.request("OPTIONS", url, **kwargs)

    def head(self, url, **kwargs):
        r"""Sends a HEAD request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        kwargs.setdefault("allow_redirects", False)
        return self.request("HEAD", url, **kwargs)

    def post(self, url, data=None, json=None, **kwargs):
        r"""Sends a POST request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("POST", url, data=data, json=json, **kwargs)

    def put(self, url, data=None, **kwargs):
        r"""Sends a PUT request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("PUT", url, data=data, **kwargs)

    def patch(self, url, data=None, **kwargs):
        r"""Sends a PATCH request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("PATCH", url, data=data, **kwargs)

    def delete(self, url, **kwargs):
        r"""Sends a DELETE request. Returns :class:`Response` object.

        :param url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :rtype: requests.Response
        """

        return self.request("DELETE", url, **kwargs)

    def send(self, request, **kwargs):
        """Send a given PreparedRequest.

        :rtype: requests.Response
        """
        # Set defaults that the hooks can utilize to ensure they always have
        # the correct parameters to reproduce the previous request.
        kwargs.setdefault("stream", self.stream)
        kwargs.setdefault("verify", self.verify)
        kwargs.setdefault("cert", self.cert)
        if "proxies" not in kwargs:
            kwargs["proxies"] = resolve_proxies(request, self.proxies, self.trust_env)

        # It's possible that users might accidentally send a Request object.
        # Guard against that specific failure case.
        if isinstance(request, Request):
            raise ValueError("You can only send PreparedRequests.")

        # Set up variables needed for resolve_redirects and dispatching of hooks
        allow_redirects = kwargs.pop("allow_redirects", True)
        stream = kwargs.get("stream")
        hooks = request.hooks

        # Get the appropriate adapter to use
        adapter = self.get_adapter(url=request.url)

        # Start time (approximately) of the request
        start = preferred_clock()

        # Send the request
        r = adapter.send(request, **kwargs)

        # Total elapsed time of the request (approximately)
        elapsed = preferred_clock() - start
        r.elapsed = timedelta(seconds=elapsed)

        # Response manipulation hooks
        r = dispatch_hook("response", hooks, r, **kwargs)

        # Persist cookies
        if r.history:
            # If the hooks create history then we want those cookies too
            for resp in r.history:
                extract_cookies_to_jar(self.cookies, resp.request, resp.raw)

        extract_cookies_to_jar(self.cookies, request, r.raw)

        # Resolve redirects if allowed.
        if allow_redirects:
            # Redirect resolving generator.
            gen = self.resolve_redirects(r, request, **kwargs)
            history = [resp for resp in gen]
        else:
            history = []

        # Shuffle things around if there's history.
        if history:
            # Insert the first (original) request at the start
            history.insert(0, r)
            # Get the last request made
            r = history.pop()
            r.history = history

        # If redirects aren't being followed, store the response on the Request for Response.next().
        if not allow_redirects:
            try:
                r._next = next(
                    self.resolve_redirects(r, request, yield_requests=True, **kwargs)
                )
            except StopIteration:
                pass

        if not stream:
            r.content

        return r

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        """
        Check the environment and merge it with some settings.

        :rtype: dict
        """
        # Gather clues from the surrounding environment.
        if self.trust_env:
            # Set environment's proxies.
            no_proxy = proxies.get("no_proxy") if proxies is not None else None
            env_proxies = get_environ_proxies(url, no_proxy=no_proxy)
            for k, v in env_proxies.items():
                proxies.setdefault(k, v)

            # Look for requests environment configuration
            # and be compatible with cURL.
            if verify is True or verify is None:
                verify = (
                    os.environ.get("REQUESTS_CA_BUNDLE")
                    or os.environ.get("CURL_CA_BUNDLE")
                    or verify
                )

        # Merge all the kwargs.
        proxies = merge_setting(proxies, self.proxies)
        stream = merge_setting(stream, self.stream)
        verify = merge_setting(verify, self.verify)
        cert = merge_setting(cert, self.cert)

        return {"proxies": proxies, "stream": stream, "verify": verify, "cert": cert}

    def get_adapter(self, url):
        """
        Returns the appropriate connection adapter for the given URL.

        :rtype: requests.adapters.BaseAdapter
        """
        for prefix, adapter in self.adapters.items():
            if url.lower().startswith(prefix.lower()):
                return adapter

        # Nothing matches :-/
        raise InvalidSchema(f"No connection adapters were found for {url!r}")

    def close(self):
        """Closes all adapters and as such the session"""
        for v in self.adapters.values():
            v.close()

    def mount(self, prefix, adapter):
        """Registers a connection adapter to a prefix.

        Adapters are sorted in descending order by prefix length.
        """
        self.adapters[prefix] = adapter
        keys_to_move = [k for k in self.adapters if len(k) < len(prefix)]

        for key in keys_to_move:
            self.adapters[key] = self.adapters.pop(key)

    def __getstate__(self):
        state = {attr: getattr(self, attr, None) for attr in self.__attrs__}
        return state

    def __setstate__(self, state):
        for attr, value in state.items():
            setattr(self, attr, value)


def session():
    """
    Returns a :class:`Session` for context-management.

    .. deprecated:: 1.0.0

        This method has been deprecated since version 1.0.0 and is only kept for
        backwards compatibility. New code should use :class:`~requests.sessions.Session`
        to create a session. This may be removed at a future date.

    :rtype: Session
    """
    return Session()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\wrappers\response.py ===
from __future__ import annotations

import json
import typing as t
from http import HTTPStatus
from urllib.parse import urljoin

from .._internal import _get_environ
from ..datastructures import Headers
from ..http import generate_etag
from ..http import http_date
from ..http import is_resource_modified
from ..http import parse_etags
from ..http import parse_range_header
from ..http import remove_entity_headers
from ..sansio.response import Response as _SansIOResponse
from ..urls import iri_to_uri
from ..utils import cached_property
from ..wsgi import _RangeWrapper
from ..wsgi import ClosingIterator
from ..wsgi import get_current_url

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment

    from .request import Request


def _iter_encoded(iterable: t.Iterable[str | bytes]) -> t.Iterator[bytes]:
    for item in iterable:
        if isinstance(item, str):
            yield item.encode()
        else:
            yield item


class Response(_SansIOResponse):
    """Represents an outgoing WSGI HTTP response with body, status, and
    headers. Has properties and methods for using the functionality
    defined by various HTTP specs.

    The response body is flexible to support different use cases. The
    simple form is passing bytes, or a string which will be encoded as
    UTF-8. Passing an iterable of bytes or strings makes this a
    streaming response. A generator is particularly useful for building
    a CSV file in memory or using SSE (Server Sent Events). A file-like
    object is also iterable, although the
    :func:`~werkzeug.utils.send_file` helper should be used in that
    case.

    The response object is itself a WSGI application callable. When
    called (:meth:`__call__`) with ``environ`` and ``start_response``,
    it will pass its status and headers to ``start_response`` then
    return its body as an iterable.

    .. code-block:: python

        from werkzeug.wrappers.response import Response

        def index():
            return Response("Hello, World!")

        def application(environ, start_response):
            path = environ.get("PATH_INFO") or "/"

            if path == "/":
                response = index()
            else:
                response = Response("Not Found", status=404)

            return response(environ, start_response)

    :param response: The data for the body of the response. A string or
        bytes, or tuple or list of strings or bytes, for a fixed-length
        response, or any other iterable of strings or bytes for a
        streaming response. Defaults to an empty body.
    :param status: The status code for the response. Either an int, in
        which case the default status message is added, or a string in
        the form ``{code} {message}``, like ``404 Not Found``. Defaults
        to 200.
    :param headers: A :class:`~werkzeug.datastructures.Headers` object,
        or a list of ``(key, value)`` tuples that will be converted to a
        ``Headers`` object.
    :param mimetype: The mime type (content type without charset or
        other parameters) of the response. If the value starts with
        ``text/`` (or matches some other special cases), the charset
        will be added to create the ``content_type``.
    :param content_type: The full content type of the response.
        Overrides building the value from ``mimetype``.
    :param direct_passthrough: Pass the response body directly through
        as the WSGI iterable. This can be used when the body is a binary
        file or other iterator of bytes, to skip some unnecessary
        checks. Use :func:`~werkzeug.utils.send_file` instead of setting
        this manually.

    .. versionchanged:: 2.1
        Old ``BaseResponse`` and mixin classes were removed.

    .. versionchanged:: 2.0
        Combine ``BaseResponse`` and mixins into a single ``Response``
        class.

    .. versionchanged:: 0.5
        The ``direct_passthrough`` parameter was added.
    """

    #: if set to `False` accessing properties on the response object will
    #: not try to consume the response iterator and convert it into a list.
    #:
    #: .. versionadded:: 0.6.2
    #:
    #:    That attribute was previously called `implicit_seqence_conversion`.
    #:    (Notice the typo).  If you did use this feature, you have to adapt
    #:    your code to the name change.
    implicit_sequence_conversion = True

    #: If a redirect ``Location`` header is a relative URL, make it an
    #: absolute URL, including scheme and domain.
    #:
    #: .. versionchanged:: 2.1
    #:     This is disabled by default, so responses will send relative
    #:     redirects.
    #:
    #: .. versionadded:: 0.8
    autocorrect_location_header = False

    #: Should this response object automatically set the content-length
    #: header if possible?  This is true by default.
    #:
    #: .. versionadded:: 0.8
    automatically_set_content_length = True

    #: The response body to send as the WSGI iterable. A list of strings
    #: or bytes represents a fixed-length response, any other iterable
    #: is a streaming response. Strings are encoded to bytes as UTF-8.
    #:
    #: Do not set to a plain string or bytes, that will cause sending
    #: the response to be very inefficient as it will iterate one byte
    #: at a time.
    response: t.Iterable[str] | t.Iterable[bytes]

    def __init__(
        self,
        response: t.Iterable[bytes] | bytes | t.Iterable[str] | str | None = None,
        status: int | str | HTTPStatus | None = None,
        headers: t.Mapping[str, str | t.Iterable[str]]
        | t.Iterable[tuple[str, str]]
        | None = None,
        mimetype: str | None = None,
        content_type: str | None = None,
        direct_passthrough: bool = False,
    ) -> None:
        super().__init__(
            status=status,
            headers=headers,
            mimetype=mimetype,
            content_type=content_type,
        )

        #: Pass the response body directly through as the WSGI iterable.
        #: This can be used when the body is a binary file or other
        #: iterator of bytes, to skip some unnecessary checks. Use
        #: :func:`~werkzeug.utils.send_file` instead of setting this
        #: manually.
        self.direct_passthrough = direct_passthrough
        self._on_close: list[t.Callable[[], t.Any]] = []

        # we set the response after the headers so that if a class changes
        # the charset attribute, the data is set in the correct charset.
        if response is None:
            self.response = []
        elif isinstance(response, (str, bytes, bytearray)):
            self.set_data(response)
        else:
            self.response = response

    def call_on_close(self, func: t.Callable[[], t.Any]) -> t.Callable[[], t.Any]:
        """Adds a function to the internal list of functions that should
        be called as part of closing down the response.  Since 0.7 this
        function also returns the function that was passed so that this
        can be used as a decorator.

        .. versionadded:: 0.6
        """
        self._on_close.append(func)
        return func

    def __repr__(self) -> str:
        if self.is_sequence:
            body_info = f"{sum(map(len, self.iter_encoded()))} bytes"
        else:
            body_info = "streamed" if self.is_streamed else "likely-streamed"
        return f"<{type(self).__name__} {body_info} [{self.status}]>"

    @classmethod
    def force_type(
        cls, response: Response, environ: WSGIEnvironment | None = None
    ) -> Response:
        """Enforce that the WSGI response is a response object of the current
        type.  Werkzeug will use the :class:`Response` internally in many
        situations like the exceptions.  If you call :meth:`get_response` on an
        exception you will get back a regular :class:`Response` object, even
        if you are using a custom subclass.

        This method can enforce a given response type, and it will also
        convert arbitrary WSGI callables into response objects if an environ
        is provided::

            # convert a Werkzeug response object into an instance of the
            # MyResponseClass subclass.
            response = MyResponseClass.force_type(response)

            # convert any WSGI application into a response object
            response = MyResponseClass.force_type(response, environ)

        This is especially useful if you want to post-process responses in
        the main dispatcher and use functionality provided by your subclass.

        Keep in mind that this will modify response objects in place if
        possible!

        :param response: a response object or wsgi application.
        :param environ: a WSGI environment object.
        :return: a response object.
        """
        if not isinstance(response, Response):
            if environ is None:
                raise TypeError(
                    "cannot convert WSGI application into response"
                    " objects without an environ"
                )

            from ..test import run_wsgi_app

            response = Response(*run_wsgi_app(response, environ))

        response.__class__ = cls
        return response

    @classmethod
    def from_app(
        cls, app: WSGIApplication, environ: WSGIEnvironment, buffered: bool = False
    ) -> Response:
        """Create a new response object from an application output.  This
        works best if you pass it an application that returns a generator all
        the time.  Sometimes applications may use the `write()` callable
        returned by the `start_response` function.  This tries to resolve such
        edge cases automatically.  But if you don't get the expected output
        you should set `buffered` to `True` which enforces buffering.

        :param app: the WSGI application to execute.
        :param environ: the WSGI environment to execute against.
        :param buffered: set to `True` to enforce buffering.
        :return: a response object.
        """
        from ..test import run_wsgi_app

        return cls(*run_wsgi_app(app, environ, buffered))

    @t.overload
    def get_data(self, as_text: t.Literal[False] = False) -> bytes: ...

    @t.overload
    def get_data(self, as_text: t.Literal[True]) -> str: ...

    def get_data(self, as_text: bool = False) -> bytes | str:
        """The string representation of the response body.  Whenever you call
        this property the response iterable is encoded and flattened.  This
        can lead to unwanted behavior if you stream big data.

        This behavior can be disabled by setting
        :attr:`implicit_sequence_conversion` to `False`.

        If `as_text` is set to `True` the return value will be a decoded
        string.

        .. versionadded:: 0.9
        """
        self._ensure_sequence()
        rv = b"".join(self.iter_encoded())

        if as_text:
            return rv.decode()

        return rv

    def set_data(self, value: bytes | str) -> None:
        """Sets a new string as response.  The value must be a string or
        bytes. If a string is set it's encoded to the charset of the
        response (utf-8 by default).

        .. versionadded:: 0.9
        """
        if isinstance(value, str):
            value = value.encode()
        self.response = [value]
        if self.automatically_set_content_length:
            self.headers["Content-Length"] = str(len(value))

    data = property(
        get_data,
        set_data,
        doc="A descriptor that calls :meth:`get_data` and :meth:`set_data`.",
    )

    def calculate_content_length(self) -> int | None:
        """Returns the content length if available or `None` otherwise."""
        try:
            self._ensure_sequence()
        except RuntimeError:
            return None
        return sum(len(x) for x in self.iter_encoded())

    def _ensure_sequence(self, mutable: bool = False) -> None:
        """This method can be called by methods that need a sequence.  If
        `mutable` is true, it will also ensure that the response sequence
        is a standard Python list.

        .. versionadded:: 0.6
        """
        if self.is_sequence:
            # if we need a mutable object, we ensure it's a list.
            if mutable and not isinstance(self.response, list):
                self.response = list(self.response)  # type: ignore
            return
        if self.direct_passthrough:
            raise RuntimeError(
                "Attempted implicit sequence conversion but the"
                " response object is in direct passthrough mode."
            )
        if not self.implicit_sequence_conversion:
            raise RuntimeError(
                "The response object required the iterable to be a"
                " sequence, but the implicit conversion was disabled."
                " Call make_sequence() yourself."
            )
        self.make_sequence()

    def make_sequence(self) -> None:
        """Converts the response iterator in a list.  By default this happens
        automatically if required.  If `implicit_sequence_conversion` is
        disabled, this method is not automatically called and some properties
        might raise exceptions.  This also encodes all the items.

        .. versionadded:: 0.6
        """
        if not self.is_sequence:
            # if we consume an iterable we have to ensure that the close
            # method of the iterable is called if available when we tear
            # down the response
            close = getattr(self.response, "close", None)
            self.response = list(self.iter_encoded())
            if close is not None:
                self.call_on_close(close)

    def iter_encoded(self) -> t.Iterator[bytes]:
        """Iter the response encoded with the encoding of the response.
        If the response object is invoked as WSGI application the return
        value of this method is used as application iterator unless
        :attr:`direct_passthrough` was activated.
        """
        # Encode in a separate function so that self.response is fetched
        # early.  This allows us to wrap the response with the return
        # value from get_app_iter or iter_encoded.
        return _iter_encoded(self.response)

    @property
    def is_streamed(self) -> bool:
        """If the response is streamed (the response is not an iterable with
        a length information) this property is `True`.  In this case streamed
        means that there is no information about the number of iterations.
        This is usually `True` if a generator is passed to the response object.

        This is useful for checking before applying some sort of post
        filtering that should not take place for streamed responses.
        """
        try:
            len(self.response)  # type: ignore
        except (TypeError, AttributeError):
            return True
        return False

    @property
    def is_sequence(self) -> bool:
        """If the iterator is buffered, this property will be `True`.  A
        response object will consider an iterator to be buffered if the
        response attribute is a list or tuple.

        .. versionadded:: 0.6
        """
        return isinstance(self.response, (tuple, list))

    def close(self) -> None:
        """Close the wrapped response if possible.  You can also use the object
        in a with statement which will automatically close it.

        .. versionadded:: 0.9
           Can now be used in a with statement.
        """
        if hasattr(self.response, "close"):
            self.response.close()
        for func in self._on_close:
            func()

    def __enter__(self) -> Response:
        return self

    def __exit__(self, exc_type, exc_value, tb):  # type: ignore
        self.close()

    def freeze(self) -> None:
        """Make the response object ready to be pickled. Does the
        following:

        *   Buffer the response into a list, ignoring
            :attr:`implicity_sequence_conversion` and
            :attr:`direct_passthrough`.
        *   Set the ``Content-Length`` header.
        *   Generate an ``ETag`` header if one is not already set.

        .. versionchanged:: 2.1
            Removed the ``no_etag`` parameter.

        .. versionchanged:: 2.0
            An ``ETag`` header is always added.

        .. versionchanged:: 0.6
            The ``Content-Length`` header is set.
        """
        # Always freeze the encoded response body, ignore
        # implicit_sequence_conversion and direct_passthrough.
        self.response = list(self.iter_encoded())
        self.headers["Content-Length"] = str(sum(map(len, self.response)))
        self.add_etag()

    def get_wsgi_headers(self, environ: WSGIEnvironment) -> Headers:
        """This is automatically called right before the response is started
        and returns headers modified for the given environment.  It returns a
        copy of the headers from the response with some modifications applied
        if necessary.

        For example the location header (if present) is joined with the root
        URL of the environment.  Also the content length is automatically set
        to zero here for certain status codes.

        .. versionchanged:: 0.6
           Previously that function was called `fix_headers` and modified
           the response object in place.  Also since 0.6, IRIs in location
           and content-location headers are handled properly.

           Also starting with 0.6, Werkzeug will attempt to set the content
           length if it is able to figure it out on its own.  This is the
           case if all the strings in the response iterable are already
           encoded and the iterable is buffered.

        :param environ: the WSGI environment of the request.
        :return: returns a new :class:`~werkzeug.datastructures.Headers`
                 object.
        """
        headers = Headers(self.headers)
        location: str | None = None
        content_location: str | None = None
        content_length: str | int | None = None
        status = self.status_code

        # iterate over the headers to find all values in one go.  Because
        # get_wsgi_headers is used each response that gives us a tiny
        # speedup.
        for key, value in headers:
            ikey = key.lower()
            if ikey == "location":
                location = value
            elif ikey == "content-location":
                content_location = value
            elif ikey == "content-length":
                content_length = value

        if location is not None:
            location = iri_to_uri(location)

            if self.autocorrect_location_header:
                # Make the location header an absolute URL.
                current_url = get_current_url(environ, strip_querystring=True)
                current_url = iri_to_uri(current_url)
                location = urljoin(current_url, location)

            headers["Location"] = location

        # make sure the content location is a URL
        if content_location is not None:
            headers["Content-Location"] = iri_to_uri(content_location)

        if 100 <= status < 200 or status == 204:
            # Per section 3.3.2 of RFC 7230, "a server MUST NOT send a
            # Content-Length header field in any response with a status
            # code of 1xx (Informational) or 204 (No Content)."
            headers.remove("Content-Length")
        elif status == 304:
            remove_entity_headers(headers)

        # if we can determine the content length automatically, we
        # should try to do that.  But only if this does not involve
        # flattening the iterator or encoding of strings in the
        # response. We however should not do that if we have a 304
        # response.
        if (
            self.automatically_set_content_length
            and self.is_sequence
            and content_length is None
            and status not in (204, 304)
            and not (100 <= status < 200)
        ):
            content_length = sum(len(x) for x in self.iter_encoded())
            headers["Content-Length"] = str(content_length)

        return headers

    def get_app_iter(self, environ: WSGIEnvironment) -> t.Iterable[bytes]:
        """Returns the application iterator for the given environ.  Depending
        on the request method and the current status code the return value
        might be an empty response rather than the one from the response.

        If the request method is `HEAD` or the status code is in a range
        where the HTTP specification requires an empty response, an empty
        iterable is returned.

        .. versionadded:: 0.6

        :param environ: the WSGI environment of the request.
        :return: a response iterable.
        """
        status = self.status_code
        if (
            environ["REQUEST_METHOD"] == "HEAD"
            or 100 <= status < 200
            or status in (204, 304)
        ):
            iterable: t.Iterable[bytes] = ()
        elif self.direct_passthrough:
            return self.response  # type: ignore
        else:
            iterable = self.iter_encoded()
        return ClosingIterator(iterable, self.close)

    def get_wsgi_response(
        self, environ: WSGIEnvironment
    ) -> tuple[t.Iterable[bytes], str, list[tuple[str, str]]]:
        """Returns the final WSGI response as tuple.  The first item in
        the tuple is the application iterator, the second the status and
        the third the list of headers.  The response returned is created
        specially for the given environment.  For example if the request
        method in the WSGI environment is ``'HEAD'`` the response will
        be empty and only the headers and status code will be present.

        .. versionadded:: 0.6

        :param environ: the WSGI environment of the request.
        :return: an ``(app_iter, status, headers)`` tuple.
        """
        headers = self.get_wsgi_headers(environ)
        app_iter = self.get_app_iter(environ)
        return app_iter, self.status, headers.to_wsgi_list()

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        """Process this response as WSGI application.

        :param environ: the WSGI environment.
        :param start_response: the response callable provided by the WSGI
                               server.
        :return: an application iterator
        """
        app_iter, status, headers = self.get_wsgi_response(environ)
        start_response(status, headers)
        return app_iter

    # JSON

    #: A module or other object that has ``dumps`` and ``loads``
    #: functions that match the API of the built-in :mod:`json` module.
    json_module = json

    @property
    def json(self) -> t.Any | None:
        """The parsed JSON data if :attr:`mimetype` indicates JSON
        (:mimetype:`application/json`, see :attr:`is_json`).

        Calls :meth:`get_json` with default arguments.
        """
        return self.get_json()

    @t.overload
    def get_json(self, force: bool = ..., silent: t.Literal[False] = ...) -> t.Any: ...

    @t.overload
    def get_json(self, force: bool = ..., silent: bool = ...) -> t.Any | None: ...

    def get_json(self, force: bool = False, silent: bool = False) -> t.Any | None:
        """Parse :attr:`data` as JSON. Useful during testing.

        If the mimetype does not indicate JSON
        (:mimetype:`application/json`, see :attr:`is_json`), this
        returns ``None``.

        Unlike :meth:`Request.get_json`, the result is not cached.

        :param force: Ignore the mimetype and always try to parse JSON.
        :param silent: Silence parsing errors and return ``None``
            instead.
        """
        if not (force or self.is_json):
            return None

        data = self.get_data()

        try:
            return self.json_module.loads(data)
        except ValueError:
            if not silent:
                raise

            return None

    # Stream

    @cached_property
    def stream(self) -> ResponseStream:
        """The response iterable as write-only stream."""
        return ResponseStream(self)

    def _wrap_range_response(self, start: int, length: int) -> None:
        """Wrap existing Response in case of Range Request context."""
        if self.status_code == 206:
            self.response = _RangeWrapper(self.response, start, length)  # type: ignore

    def _is_range_request_processable(self, environ: WSGIEnvironment) -> bool:
        """Return ``True`` if `Range` header is present and if underlying
        resource is considered unchanged when compared with `If-Range` header.
        """
        return (
            "HTTP_IF_RANGE" not in environ
            or not is_resource_modified(
                environ,
                self.headers.get("etag"),
                None,
                self.headers.get("last-modified"),
                ignore_if_range=False,
            )
        ) and "HTTP_RANGE" in environ

    def _process_range_request(
        self,
        environ: WSGIEnvironment,
        complete_length: int | None,
        accept_ranges: bool | str,
    ) -> bool:
        """Handle Range Request related headers (RFC7233).  If `Accept-Ranges`
        header is valid, and Range Request is processable, we set the headers
        as described by the RFC, and wrap the underlying response in a
        RangeWrapper.

        Returns ``True`` if Range Request can be fulfilled, ``False`` otherwise.

        :raises: :class:`~werkzeug.exceptions.RequestedRangeNotSatisfiable`
                 if `Range` header could not be parsed or satisfied.

        .. versionchanged:: 2.0
            Returns ``False`` if the length is 0.
        """
        from ..exceptions import RequestedRangeNotSatisfiable

        if (
            not accept_ranges
            or complete_length is None
            or complete_length == 0
            or not self._is_range_request_processable(environ)
        ):
            return False

        if accept_ranges is True:
            accept_ranges = "bytes"

        parsed_range = parse_range_header(environ.get("HTTP_RANGE"))

        if parsed_range is None:
            raise RequestedRangeNotSatisfiable(complete_length)

        range_tuple = parsed_range.range_for_length(complete_length)
        content_range_header = parsed_range.to_content_range_header(complete_length)

        if range_tuple is None or content_range_header is None:
            raise RequestedRangeNotSatisfiable(complete_length)

        content_length = range_tuple[1] - range_tuple[0]
        self.headers["Content-Length"] = str(content_length)
        self.headers["Accept-Ranges"] = accept_ranges
        self.content_range = content_range_header  # type: ignore
        self.status_code = 206
        self._wrap_range_response(range_tuple[0], content_length)
        return True

    def make_conditional(
        self,
        request_or_environ: WSGIEnvironment | Request,
        accept_ranges: bool | str = False,
        complete_length: int | None = None,
    ) -> Response:
        """Make the response conditional to the request.  This method works
        best if an etag was defined for the response already.  The `add_etag`
        method can be used to do that.  If called without etag just the date
        header is set.

        This does nothing if the request method in the request or environ is
        anything but GET or HEAD.

        For optimal performance when handling range requests, it's recommended
        that your response data object implements `seekable`, `seek` and `tell`
        methods as described by :py:class:`io.IOBase`.  Objects returned by
        :meth:`~werkzeug.wsgi.wrap_file` automatically implement those methods.

        It does not remove the body of the response because that's something
        the :meth:`__call__` function does for us automatically.

        Returns self so that you can do ``return resp.make_conditional(req)``
        but modifies the object in-place.

        :param request_or_environ: a request object or WSGI environment to be
                                   used to make the response conditional
                                   against.
        :param accept_ranges: This parameter dictates the value of
                              `Accept-Ranges` header. If ``False`` (default),
                              the header is not set. If ``True``, it will be set
                              to ``"bytes"``. If it's a string, it will use this
                              value.
        :param complete_length: Will be used only in valid Range Requests.
                                It will set `Content-Range` complete length
                                value and compute `Content-Length` real value.
                                This parameter is mandatory for successful
                                Range Requests completion.
        :raises: :class:`~werkzeug.exceptions.RequestedRangeNotSatisfiable`
                 if `Range` header could not be parsed or satisfied.

        .. versionchanged:: 2.0
            Range processing is skipped if length is 0 instead of
            raising a 416 Range Not Satisfiable error.
        """
        environ = _get_environ(request_or_environ)
        if environ["REQUEST_METHOD"] in ("GET", "HEAD"):
            # if the date is not in the headers, add it now.  We however
            # will not override an already existing header.  Unfortunately
            # this header will be overridden by many WSGI servers including
            # wsgiref.
            if "date" not in self.headers:
                self.headers["Date"] = http_date()
            is206 = self._process_range_request(environ, complete_length, accept_ranges)
            if not is206 and not is_resource_modified(
                environ,
                self.headers.get("etag"),
                None,
                self.headers.get("last-modified"),
            ):
                if parse_etags(environ.get("HTTP_IF_MATCH")):
                    self.status_code = 412
                else:
                    self.status_code = 304
            if (
                self.automatically_set_content_length
                and "content-length" not in self.headers
            ):
                length = self.calculate_content_length()
                if length is not None:
                    self.headers["Content-Length"] = str(length)
        return self

    def add_etag(self, overwrite: bool = False, weak: bool = False) -> None:
        """Add an etag for the current response if there is none yet.

        .. versionchanged:: 2.0
            SHA-1 is used to generate the value. MD5 may not be
            available in some environments.
        """
        if overwrite or "etag" not in self.headers:
            self.set_etag(generate_etag(self.get_data()), weak)


class ResponseStream:
    """A file descriptor like object used by :meth:`Response.stream` to
    represent the body of the stream. It directly pushes into the
    response iterable of the response object.
    """

    mode = "wb+"

    def __init__(self, response: Response):
        self.response = response
        self.closed = False

    def write(self, value: bytes) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        self.response._ensure_sequence(mutable=True)
        self.response.response.append(value)  # type: ignore
        self.response.headers.pop("Content-Length", None)
        return len(value)

    def writelines(self, seq: t.Iterable[bytes]) -> None:
        for item in seq:
            self.write(item)

    def close(self) -> None:
        self.closed = True

    def flush(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def isatty(self) -> bool:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return False

    def tell(self) -> int:
        self.response._ensure_sequence()
        return sum(map(len, self.response.response))

    @property
    def encoding(self) -> str:
        return "utf-8"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\primetest.py ===
"""
Primality testing

"""

from itertools import count

from sympy.core.sympify import sympify
from sympy.external.gmpy import (gmpy as _gmpy, gcd, jacobi,
                                 is_square as gmpy_is_square,
                                 bit_scan1, is_fermat_prp, is_euler_prp,
                                 is_selfridge_prp, is_strong_selfridge_prp,
                                 is_strong_bpsw_prp)
from sympy.external.ntheory import _lucas_sequence
from sympy.utilities.misc import as_int, filldedent

# Note: This list should be updated whenever new Mersenne primes are found.
# Refer: https://www.mersenne.org/
MERSENNE_PRIME_EXPONENTS = (2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203,
 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049,
 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583,
 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933,
 136279841)


def is_fermat_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{n-1} \equiv 1 \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Fermat pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_fermat_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_fermat_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    341
    561
    645

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fermat_pseudoprime
    """
    n, a = as_int(n), as_int(a)
    if a == 1:
        return n == 2 or bool(n % 2)
    return is_fermat_prp(n, a)


def is_euler_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{(n-1)/2} \equiv \pm 1 \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Euler pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_euler_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_euler_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    341
    561

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler_pseudoprime
    """
    n, a = as_int(n), as_int(a)
    if a < 1:
        raise ValueError("a should be an integer greater than 0")
    if n < 1:
        raise ValueError("n should be an integer greater than 0")
    if n == 1:
        return False
    if a == 1:
        return n == 2 or bool(n % 2)  # (prime or odd composite)
    if n % 2 == 0:
        return n == 2
    if gcd(n, a) != 1:
        raise ValueError("The two numbers should be relatively prime")
    return pow(a, (n - 1) // 2, n) in [1, n - 1]


def is_euler_jacobi_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{(n-1)/2} \equiv \left(\frac{a}{n}\right) \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Euler-Jacobi pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_euler_jacobi_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_euler_jacobi_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    561

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Jacobi_pseudoprime
    """
    n, a = as_int(n), as_int(a)
    if a == 1:
        return n == 2 or bool(n % 2)
    return is_euler_prp(n, a)


def is_square(n, prep=True):
    """Return True if n == a * a for some integer a, else False.
    If n is suspected of *not* being a square then this is a
    quick method of confirming that it is not.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_square
    >>> is_square(25)
    True
    >>> is_square(2)
    False

    References
    ==========

    .. [1]  https://mersenneforum.org/showpost.php?p=110896

    See Also
    ========
    sympy.core.intfunc.isqrt
    """
    if prep:
        n = as_int(n)
        if n < 0:
            return False
        if n in (0, 1):
            return True
    return gmpy_is_square(n)


def _test(n, base, s, t):
    """Miller-Rabin strong pseudoprime test for one base.
    Return False if n is definitely composite, True if n is
    probably prime, with a probability greater than 3/4.

    """
    # do the Fermat test
    b = pow(base, t, n)
    if b == 1 or b == n - 1:
        return True
    for _ in range(s - 1):
        b = pow(b, 2, n)
        if b == n - 1:
            return True
        # see I. Niven et al. "An Introduction to Theory of Numbers", page 78
        if b == 1:
            return False
    return False


def mr(n, bases):
    """Perform a Miller-Rabin strong pseudoprime test on n using a
    given list of bases/witnesses.

    References
    ==========

    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:
           A Computational Perspective", Springer, 2nd edition, 135-138

    A list of thresholds and the bases they require are here:
    https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Deterministic_variants

    Examples
    ========

    >>> from sympy.ntheory.primetest import mr
    >>> mr(1373651, [2, 3])
    False
    >>> mr(479001599, [31, 73])
    True

    """
    from sympy.polys.domains import ZZ

    n = as_int(n)
    if n < 2 or (n > 2 and n % 2 == 0):
        return False
    # remove powers of 2 from n-1 (= t * 2**s)
    s = bit_scan1(n - 1)
    t = n >> s
    for base in bases:
        # Bases >= n are wrapped, bases < 2 are invalid
        if base >= n:
            base %= n
        if base >= 2:
            base = ZZ(base)
            if not _test(n, base, s, t):
                return False
    return True


def _lucas_extrastrong_params(n):
    """Calculates the "extra strong" parameters (D, P, Q) for n.

    Parameters
    ==========

    n : int
        positive odd integer

    Returns
    =======

    D, P, Q: "extra strong" parameters.
             ``(0, 0, 0)`` if we find a nontrivial divisor of ``n``.

    Examples
    ========

    >>> from sympy.ntheory.primetest import _lucas_extrastrong_params
    >>> _lucas_extrastrong_params(101)
    (12, 4, 1)
    >>> _lucas_extrastrong_params(15)
    (0, 0, 0)

    References
    ==========
    .. [1] OEIS A217719: Extra Strong Lucas Pseudoprimes
           https://oeis.org/A217719
    .. [2] https://en.wikipedia.org/wiki/Lucas_pseudoprime

    """
    for P in count(3):
        D = P**2 - 4
        j = jacobi(D, n)
        if j == -1:
            return (D, P, 1)
        elif j == 0 and D % n:
            return (0, 0, 0)


def is_lucas_prp(n):
    """Standard Lucas compositeness test with Selfridge parameters.  Returns
    False if n is definitely composite, and True if n is a Lucas probable
    prime.

    This is typically used in combination with the Miller-Rabin test.

    References
    ==========
    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf
    .. [2] OEIS A217120: Lucas Pseudoprimes
           https://oeis.org/A217120
    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_lucas_prp
    >>> for i in range(10000):
    ...     if is_lucas_prp(i) and not isprime(i):
    ...         print(i)
    323
    377
    1159
    1829
    3827
    5459
    5777
    9071
    9179
    """
    n = as_int(n)
    if n < 2:
        return False
    return is_selfridge_prp(n)


def is_strong_lucas_prp(n):
    """Strong Lucas compositeness test with Selfridge parameters.  Returns
    False if n is definitely composite, and True if n is a strong Lucas
    probable prime.

    This is often used in combination with the Miller-Rabin test, and
    in particular, when combined with M-R base 2 creates the strong BPSW test.

    References
    ==========
    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf
    .. [2] OEIS A217255: Strong Lucas Pseudoprimes
           https://oeis.org/A217255
    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime
    .. [4] https://en.wikipedia.org/wiki/Baillie-PSW_primality_test

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_strong_lucas_prp
    >>> for i in range(20000):
    ...     if is_strong_lucas_prp(i) and not isprime(i):
    ...        print(i)
    5459
    5777
    10877
    16109
    18971
    """
    n = as_int(n)
    if n < 2:
        return False
    return is_strong_selfridge_prp(n)


def is_extra_strong_lucas_prp(n):
    """Extra Strong Lucas compositeness test.  Returns False if n is
    definitely composite, and True if n is an "extra strong" Lucas probable
    prime.

    The parameters are selected using P = 3, Q = 1, then incrementing P until
    (D|n) == -1.  The test itself is as defined in [1]_, from the
    Mo and Jones preprint.  The parameter selection and test are the same as
    used in OEIS A217719, Perl's Math::Prime::Util, and the Lucas pseudoprime
    page on Wikipedia.

    It is 20-50% faster than the strong test.

    Because of the different parameters selected, there is no relationship
    between the strong Lucas pseudoprimes and extra strong Lucas pseudoprimes.
    In particular, one is not a subset of the other.

    References
    ==========
    .. [1] Jon Grantham, Frobenius Pseudoprimes,
           Math. Comp. Vol 70, Number 234 (2001), pp. 873-891,
           https://doi.org/10.1090%2FS0025-5718-00-01197-2
    .. [2] OEIS A217719: Extra Strong Lucas Pseudoprimes
           https://oeis.org/A217719
    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_extra_strong_lucas_prp
    >>> for i in range(20000):
    ...     if is_extra_strong_lucas_prp(i) and not isprime(i):
    ...        print(i)
    989
    3239
    5777
    10877
    """
    # Implementation notes:
    #   1) the parameters differ from Thomas R. Nicely's.  His parameter
    #      selection leads to pseudoprimes that overlap M-R tests, and
    #      contradict Baillie and Wagstaff's suggestion of (D|n) = -1.
    #   2) The MathWorld page as of June 2013 specifies Q=-1.  The Lucas
    #      sequence must have Q=1.  See Grantham theorem 2.3, any of the
    #      references on the MathWorld page, or run it and see Q=-1 is wrong.
    n = as_int(n)
    if n == 2:
        return True
    if n < 2 or (n % 2) == 0:
        return False
    if gmpy_is_square(n):
        return False

    D, P, Q = _lucas_extrastrong_params(n)
    if D == 0:
        return False

    # remove powers of 2 from n+1 (= k * 2**s)
    s = bit_scan1(n + 1)
    k = (n + 1) >> s

    U, V, _ = _lucas_sequence(n, P, Q, k)

    if U == 0 and (V == 2 or V == n - 2):
        return True
    for _ in range(1, s):
        if V == 0:
            return True
        V = (V*V - 2) % n
    return False


def proth_test(n):
    r""" Test if the Proth number `n = k2^m + 1` is prime. where k is a positive odd number and `2^m > k`.

    Parameters
    ==========

    n : Integer
        ``n`` is Proth number

    Returns
    =======

    bool : If ``True``, then ``n`` is the Proth prime

    Raises
    ======

    ValueError
        If ``n`` is not Proth number.

    Examples
    ========

    >>> from sympy.ntheory.primetest import proth_test
    >>> proth_test(41)
    True
    >>> proth_test(57)
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Proth_prime

    """
    n = as_int(n)
    if n < 3:
        raise ValueError("n is not Proth number")
    m = bit_scan1(n - 1)
    k = n >> m
    if m < k.bit_length():
        raise ValueError("n is not Proth number")
    if n % 3 == 0:
        return n == 3
    if k % 3: # n % 12 == 5
        return pow(3, n >> 1, n) == n - 1
    # If `n` is a square number, then `jacobi(a, n) = 1` for any `a`
    if gmpy_is_square(n):
        return False
    # `a` may be chosen at random.
    # In any case, we want to find `a` such that `jacobi(a, n) = -1`.
    for a in range(5, n):
        j = jacobi(a, n)
        if j == -1:
            return pow(a, n >> 1, n) == n - 1
        if j == 0:
            return False


def _lucas_lehmer_primality_test(p):
    r""" Test if the Mersenne number `M_p = 2^p-1` is prime.

    Parameters
    ==========

    p : int
        ``p`` is an odd prime number

    Returns
    =======

    bool : If ``True``, then `M_p` is the Mersenne prime

    Examples
    ========

    >>> from sympy.ntheory.primetest import _lucas_lehmer_primality_test
    >>> _lucas_lehmer_primality_test(5) # 2**5 - 1 = 31 is prime
    True
    >>> _lucas_lehmer_primality_test(11) # 2**11 - 1 = 2047 is not prime
    False

    See Also
    ========

    is_mersenne_prime

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas%E2%80%93Lehmer_primality_test

    """
    v = 4
    m = 2**p - 1
    for _ in range(p - 2):
        v = pow(v, 2, m) - 2
    return v == 0


def is_mersenne_prime(n):
    """Returns True if  ``n`` is a Mersenne prime, else False.

    A Mersenne prime is a prime number having the form `2^i - 1`.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import is_mersenne_prime
    >>> is_mersenne_prime(6)
    False
    >>> is_mersenne_prime(127)
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/MersennePrime.html

    """
    n = as_int(n)
    if n < 1:
        return False
    if n & (n + 1):
        # n is not Mersenne number
        return False
    p = n.bit_length()
    if p in MERSENNE_PRIME_EXPONENTS:
        return True
    if p < 65_000_000 or not isprime(p):
        # According to GIMPS, verification was completed on September 19, 2023 for p less than 65 million.
        # https://www.mersenne.org/report_milestones/
        # If p is composite number, then n=2**p-1 is composite number.
        return False
    result = _lucas_lehmer_primality_test(p)
    if result:
        raise ValueError(filldedent('''
            This Mersenne Prime, 2^%s - 1, should
            be added to SymPy's known values.''' % p))
    return result


_MR_BASES_32 = [15591, 2018, 166, 7429, 8064, 16045, 10503, 4399, 1949, 1295,
                2776, 3620, 560, 3128, 5212, 2657, 2300, 2021, 4652, 1471,
                9336, 4018, 2398, 20462, 10277, 8028, 2213, 6219, 620, 3763,
                4852, 5012, 3185, 1333, 6227,5298, 1074, 2391, 5113, 7061,
                803, 1269, 3875, 422, 751, 580, 4729, 10239, 746, 2951, 556,
                2206, 3778, 481, 1522, 3476, 481, 2487, 3266, 5633, 488, 3373,
                6441, 3344, 17, 15105, 1490, 4154, 2036, 1882, 1813, 467,
                3307, 14042, 6371, 658, 1005, 903, 737, 1887, 7447, 1888,
                2848, 1784, 7559, 3400, 951, 13969, 4304, 177, 41, 19875,
                3110, 13221, 8726, 571, 7043, 6943, 1199, 352, 6435, 165,
                1169, 3315, 978, 233, 3003, 2562, 2994, 10587, 10030, 2377,
                1902, 5354, 4447, 1555, 263, 27027, 2283, 305, 669, 1912, 601,
                6186, 429, 1930, 14873, 1784, 1661, 524, 3577, 236, 2360,
                6146, 2850, 55637, 1753, 4178, 8466, 222, 2579, 2743, 2031,
                2226, 2276, 374, 2132, 813, 23788, 1610, 4422, 5159, 1725,
                3597, 3366, 14336, 579, 165, 1375, 10018, 12616, 9816, 1371,
                536, 1867, 10864, 857, 2206, 5788, 434, 8085, 17618, 727,
                3639, 1595, 4944, 2129, 2029, 8195, 8344, 6232, 9183, 8126,
                1870, 3296, 7455, 8947, 25017, 541, 19115, 368, 566, 5674,
                411, 522, 1027, 8215, 2050, 6544, 10049, 614, 774, 2333, 3007,
                35201, 4706, 1152, 1785, 1028, 1540, 3743, 493, 4474, 2521,
                26845, 8354, 864, 18915, 5465, 2447, 42, 4511, 1660, 166,
                1249, 6259, 2553, 304, 272, 7286, 73, 6554, 899, 2816, 5197,
                13330, 7054, 2818, 3199, 811, 922, 350, 7514, 4452, 3449,
                2663, 4708, 418, 1621, 1171, 3471, 88, 11345, 412, 1559, 194]


def isprime(n):
    """
    Test if n is a prime number (True) or not (False). For n < 2^64 the
    answer is definitive; larger n values have a small probability of actually
    being pseudoprimes.

    Negative numbers (e.g. -2) are not considered prime.

    The first step is looking for trivial factors, which if found enables
    a quick return.  Next, if the sieve is large enough, use bisection search
    on the sieve.  For small numbers, a set of deterministic Miller-Rabin
    tests are performed with bases that are known to have no counterexamples
    in their range.  Finally if the number is larger than 2^64, a strong
    BPSW test is performed.  While this is a probable prime test and we
    believe counterexamples exist, there are no known counterexamples.

    Examples
    ========

    >>> from sympy.ntheory import isprime
    >>> isprime(13)
    True
    >>> isprime(15)
    False

    Notes
    =====

    This routine is intended only for integer input, not numerical
    expressions which may represent numbers. Floats are also
    rejected as input because they represent numbers of limited
    precision. While it is tempting to permit 7.0 to represent an
    integer there are errors that may "pass silently" if this is
    allowed:

    >>> from sympy import Float, S
    >>> int(1e3) == 1e3 == 10**3
    True
    >>> int(1e23) == 1e23
    True
    >>> int(1e23) == 10**23
    False

    >>> near_int = 1 + S(1)/10**19
    >>> near_int == int(near_int)
    False
    >>> n = Float(near_int, 10)  # truncated by precision
    >>> n % 1 == 0
    True
    >>> n = Float(near_int, 20)
    >>> n % 1 == 0
    False

    See Also
    ========

    sympy.ntheory.generate.primerange : Generates all primes in a given range
    sympy.functions.combinatorial.numbers.primepi : Return the number of primes less than or equal to n
    sympy.ntheory.generate.prime : Return the nth prime

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Strong_pseudoprime
    .. [2] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf
    .. [3] https://en.wikipedia.org/wiki/Baillie-PSW_primality_test
    """
    n = as_int(n)

    # Step 1, do quick composite testing via trial division.  The individual
    # modulo tests benchmark faster than one or two primorial igcds for me.
    # The point here is just to speedily handle small numbers and many
    # composites.  Step 2 only requires that n <= 2 get handled here.
    if n in [2, 3, 5]:
        return True
    if n < 2 or (n % 2) == 0 or (n % 3) == 0 or (n % 5) == 0:
        return False
    if n < 49:
        return True
    if (n %  7) == 0 or (n % 11) == 0 or (n % 13) == 0 or (n % 17) == 0 or \
       (n % 19) == 0 or (n % 23) == 0 or (n % 29) == 0 or (n % 31) == 0 or \
       (n % 37) == 0 or (n % 41) == 0 or (n % 43) == 0 or (n % 47) == 0:
        return False
    if n < 2809:
        return True
    if n < 65077:
        # There are only five Euler pseudoprimes with a least prime factor greater than 47
        return pow(2, n >> 1, n) in [1, n - 1] and n not in [8321, 31621, 42799, 49141, 49981]

    # bisection search on the sieve if the sieve is large enough
    from sympy.ntheory.generate import sieve as s
    if n <= s._list[-1]:
        l, u = s.search(n)
        return l == u
    from sympy.ntheory.factor_ import factor_cache
    if (ret := factor_cache.get(n)) is not None:
        return ret == n

    # If we have GMPY2, skip straight to step 3 and do a strong BPSW test.
    # This should be a bit faster than our step 2, and for large values will
    # be a lot faster than our step 3 (C+GMP vs. Python).
    if _gmpy is not None:
        return is_strong_bpsw_prp(n)


    # Step 2: deterministic Miller-Rabin testing for numbers < 2^64.  See:
    #    https://miller-rabin.appspot.com/
    # for lists.  We have made sure the M-R routine will successfully handle
    # bases larger than n, so we can use the minimal set.
    # In September 2015 deterministic numbers were extended to over 2^81.
    #    https://arxiv.org/pdf/1509.00864.pdf
    #    https://oeis.org/A014233
    if n < 341531:
        return mr(n, [9345883071009581737])
    if n < 4296595241:
        # Michal Forisek and Jakub Jancina,
        # Fast Primality Testing for Integers That Fit into a Machine Word
        # https://ceur-ws.org/Vol-1326/020-Forisek.pdf
        h = ((n >> 16) ^ n) * 0x45d9f3b
        h = ((h >> 16) ^ h) * 0x45d9f3b
        h = ((h >> 16) ^ h) & 255
        return mr(n, [_MR_BASES_32[h]])
    if n < 350269456337:
        return mr(n, [4230279247111683200, 14694767155120705706, 16641139526367750375])
    if n < 55245642489451:
        return mr(n, [2, 141889084524735, 1199124725622454117, 11096072698276303650])
    if n < 7999252175582851:
        return mr(n, [2, 4130806001517, 149795463772692060, 186635894390467037, 3967304179347715805])
    if n < 585226005592931977:
        return mr(n, [2, 123635709730000, 9233062284813009, 43835965440333360, 761179012939631437, 1263739024124850375])
    if n < 18446744073709551616:
        return mr(n, [2, 325, 9375, 28178, 450775, 9780504, 1795265022])
    if n < 318665857834031151167461:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    if n < 3317044064679887385961981:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])

    # We could do this instead at any point:
    #if n < 18446744073709551616:
    #   return mr(n, [2]) and is_extra_strong_lucas_prp(n)

    # Here are tests that are safe for MR routines that don't understand
    # large bases.
    #if n < 9080191:
    #    return mr(n, [31, 73])
    #if n < 19471033:
    #    return mr(n, [2, 299417])
    #if n < 38010307:
    #    return mr(n, [2, 9332593])
    #if n < 316349281:
    #    return mr(n, [11000544, 31481107])
    #if n < 4759123141:
    #    return mr(n, [2, 7, 61])
    #if n < 105936894253:
    #    return mr(n, [2, 1005905886, 1340600841])
    #if n < 31858317218647:
    #    return mr(n, [2, 642735, 553174392, 3046413974])
    #if n < 3071837692357849:
    #    return mr(n, [2, 75088, 642735, 203659041, 3613982119])
    #if n < 18446744073709551616:
    #    return mr(n, [2, 325, 9375, 28178, 450775, 9780504, 1795265022])

    # Step 3: BPSW.
    #
    #  Time for isprime(10**2000 + 4561), no gmpy or gmpy2 installed
    #     44.0s   old isprime using 46 bases
    #      5.3s   strong BPSW + one random base
    #      4.3s   extra strong BPSW + one random base
    #      4.1s   strong BPSW
    #      3.2s   extra strong BPSW

    # Classic BPSW from page 1401 of the paper.  See alternate ideas below.
    return is_strong_bpsw_prp(n)

    # Using extra strong test, which is somewhat faster
    #return mr(n, [2]) and is_extra_strong_lucas_prp(n)

    # Add a random M-R base
    #import random
    #return mr(n, [2, random.randint(3, n-1)]) and is_strong_lucas_prp(n)


def is_gaussian_prime(num):
    r"""Test if num is a Gaussian prime number.

    References
    ==========

    .. [1] https://oeis.org/wiki/Gaussian_primes
    """

    num = sympify(num)
    a, b = num.as_real_imag()
    a = as_int(a, strict=False)
    b = as_int(b, strict=False)
    if a == 0:
        b = abs(b)
        return isprime(b) and b % 4 == 3
    elif b == 0:
        a = abs(a)
        return isprime(a) and a % 4 == 3
    return isprime(a**2 + b**2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flatbuffers\builder.py ===
# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)

from . import encode
from . import packer

from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature

import warnings

np = import_numpy()
## @file
## @addtogroup flatbuffers_python_api
## @{

## @cond FLATBUFFERS_INTERNAL
class OffsetArithmeticError(RuntimeError):
    """
    Error caused by an Offset arithmetic error. Probably caused by bad
    writing of fields. This is considered an unreachable situation in
    normal circumstances.
    """
    pass


class IsNotNestedError(RuntimeError):
    """
    Error caused by using a Builder to write Object data when not inside
    an Object.
    """
    pass


class IsNestedError(RuntimeError):
    """
    Error caused by using a Builder to begin an Object when an Object is
    already being built.
    """
    pass


class StructIsNotInlineError(RuntimeError):
    """
    Error caused by using a Builder to write a Struct at a location that
    is not the current Offset.
    """
    pass


class BuilderSizeError(RuntimeError):
    """
    Error caused by causing a Builder to exceed the hardcoded limit of 2
    gigabytes.
    """
    pass

class BuilderNotFinishedError(RuntimeError):
    """
    Error caused by not calling `Finish` before calling `Output`.
    """
    pass

class EndVectorLengthMismatched(RuntimeError):
    """
    The number of elements passed to EndVector does not match the number
    specified in StartVector.
    """
    pass


# VtableMetadataFields is the count of metadata fields in each vtable.
VtableMetadataFields = 2
## @endcond

class Builder(object):
    """ A Builder is used to construct one or more FlatBuffers.

    Typically, Builder objects will be used from code generated by the `flatc`
    compiler.

    A Builder constructs byte buffers in a last-first manner for simplicity and
    performance during reading.

    Internally, a Builder is a state machine for creating FlatBuffer objects.

    It holds the following internal state:
        - Bytes: an array of bytes.
        - current_vtable: a list of integers.
        - vtables: a hash of vtable entries.

    Attributes:
      Bytes: The internal `bytearray` for the Builder.
      finished: A boolean determining if the Builder has been finalized.
    """

    ## @cond FLATBUFFERS_INTENRAL
    __slots__ = ("Bytes", "current_vtable", "head", "minalign", "objectEnd",
                 "vtables", "nested", "forceDefaults", "finished", "vectorNumElems",
                 "sharedStrings")

    """Maximum buffer size constant, in bytes.

    Builder will never allow it's buffer grow over this size.
    Currently equals 2Gb.
    """
    MAX_BUFFER_SIZE = 2**31
    ## @endcond

    def __init__(self, initialSize=1024):
        """Initializes a Builder of size `initial_size`.

        The internal buffer is grown as needed.
        """

        if not (0 <= initialSize <= Builder.MAX_BUFFER_SIZE):
            msg = "flatbuffers: Cannot create Builder larger than 2 gigabytes."
            raise BuilderSizeError(msg)

        self.Bytes = bytearray(initialSize)
        ## @cond FLATBUFFERS_INTERNAL
        self.current_vtable = None
        self.head = UOffsetTFlags.py_type(initialSize)
        self.minalign = 1
        self.objectEnd = None
        self.vtables = {}
        self.nested = False
        self.forceDefaults = False
        self.sharedStrings = {}
        ## @endcond
        self.finished = False

    def Clear(self) -> None:
        ## @cond FLATBUFFERS_INTERNAL
        self.current_vtable = None
        self.head = UOffsetTFlags.py_type(len(self.Bytes))
        self.minalign = 1
        self.objectEnd = None
        self.vtables = {}
        self.nested = False
        self.forceDefaults = False
        self.sharedStrings = {}
        self.vectorNumElems = None
        ## @endcond
        self.finished = False

    def Output(self):
        """Return the portion of the buffer that has been used for writing data.

        This is the typical way to access the FlatBuffer data inside the
        builder. If you try to access `Builder.Bytes` directly, you would need
        to manually index it with `Head()`, since the buffer is constructed
        backwards.

        It raises BuilderNotFinishedError if the buffer has not been finished
        with `Finish`.
        """

        if not self.finished:
            raise BuilderNotFinishedError()

        return self.Bytes[self.Head():]

    ## @cond FLATBUFFERS_INTERNAL
    def StartObject(self, numfields):
        """StartObject initializes bookkeeping for writing a new object."""

        self.assertNotNested()

        # use 32-bit offsets so that arithmetic doesn't overflow.
        self.current_vtable = [0 for _ in range_func(numfields)]
        self.objectEnd = self.Offset()
        self.nested = True

    def WriteVtable(self):
        """
        WriteVtable serializes the vtable for the current object, if needed.

        Before writing out the vtable, this checks pre-existing vtables for
        equality to this one. If an equal vtable is found, point the object to
        the existing vtable and return.

        Because vtable values are sensitive to alignment of object data, not
        all logically-equal vtables will be deduplicated.

        A vtable has the following format:
          <VOffsetT: size of the vtable in bytes, including this value>
          <VOffsetT: size of the object in bytes, including the vtable offset>
          <VOffsetT: offset for a field> * N, where N is the number of fields
                     in the schema for this type. Includes deprecated fields.
        Thus, a vtable is made of 2 + N elements, each VOffsetT bytes wide.

        An object has the following format:
          <SOffsetT: offset to this object's vtable (may be negative)>
          <byte: data>+
        """

        # Prepend a zero scalar to the object. Later in this function we'll
        # write an offset here that points to the object's vtable:
        self.PrependSOffsetTRelative(0)

        objectOffset = self.Offset()

        vtKey = []
        trim = True
        for elem in reversed(self.current_vtable):
            if elem == 0:
                if trim:
                    continue
            else:
                elem = objectOffset - elem
                trim = False

            vtKey.append(elem)

        vtKey = tuple(vtKey)
        vt2Offset = self.vtables.get(vtKey)
        if vt2Offset is None:
            # Did not find a vtable, so write this one to the buffer.

            # Write out the current vtable in reverse , because
            # serialization occurs in last-first order:
            i = len(self.current_vtable) - 1
            trailing = 0
            trim = True
            while i >= 0:
                off = 0
                elem = self.current_vtable[i]
                i -= 1

                if elem == 0:
                    if trim:
                        trailing += 1
                        continue
                else:
                    # Forward reference to field;
                    # use 32bit number to ensure no overflow:
                    off = objectOffset - elem
                    trim = False

                self.PrependVOffsetT(off)

            # The two metadata fields are written last.

            # First, store the object bytesize:
            objectSize = UOffsetTFlags.py_type(objectOffset - self.objectEnd)
            self.PrependVOffsetT(VOffsetTFlags.py_type(objectSize))

            # Second, store the vtable bytesize:
            vBytes = len(self.current_vtable) - trailing + VtableMetadataFields
            vBytes *= N.VOffsetTFlags.bytewidth
            self.PrependVOffsetT(VOffsetTFlags.py_type(vBytes))

            # Next, write the offset to the new vtable in the
            # already-allocated SOffsetT at the beginning of this object:
            objectStart = SOffsetTFlags.py_type(len(self.Bytes) - objectOffset)
            encode.Write(packer.soffset, self.Bytes, objectStart,
                         SOffsetTFlags.py_type(self.Offset() - objectOffset))

            # Finally, store this vtable in memory for future
            # deduplication:
            self.vtables[vtKey] = self.Offset()
        else:
            # Found a duplicate vtable.
            objectStart = SOffsetTFlags.py_type(len(self.Bytes) - objectOffset)
            self.head = UOffsetTFlags.py_type(objectStart)

            # Write the offset to the found vtable in the
            # already-allocated SOffsetT at the beginning of this object:
            encode.Write(packer.soffset, self.Bytes, self.Head(),
                         SOffsetTFlags.py_type(vt2Offset - objectOffset))

        self.current_vtable = None
        return objectOffset

    def EndObject(self):
        """EndObject writes data necessary to finish object construction."""
        self.assertNested()
        self.nested = False
        return self.WriteVtable()

    def growByteBuffer(self):
        """Doubles the size of the byteslice, and copies the old data towards
           the end of the new buffer (since we build the buffer backwards)."""
        if len(self.Bytes) == Builder.MAX_BUFFER_SIZE:
            msg = "flatbuffers: cannot grow buffer beyond 2 gigabytes"
            raise BuilderSizeError(msg)

        newSize = min(len(self.Bytes) * 2, Builder.MAX_BUFFER_SIZE)
        if newSize == 0:
            newSize = 1
        bytes2 = bytearray(newSize)
        bytes2[newSize-len(self.Bytes):] = self.Bytes
        self.Bytes = bytes2
    ## @endcond

    def Head(self):
        """Get the start of useful data in the underlying byte buffer.

        Note: unlike other functions, this value is interpreted as from the
        left.
        """
        ## @cond FLATBUFFERS_INTERNAL
        return self.head
        ## @endcond

    ## @cond FLATBUFFERS_INTERNAL
    def Offset(self):
        """Offset relative to the end of the buffer."""
        return UOffsetTFlags.py_type(len(self.Bytes) - self.Head())

    def Pad(self, n):
        """Pad places zeros at the current offset."""
        for i in range_func(n):
            self.Place(0, N.Uint8Flags)

    def Prep(self, size, additionalBytes):
        """
        Prep prepares to write an element of `size` after `additional_bytes`
        have been written, e.g. if you write a string, you need to align
        such the int length field is aligned to SizeInt32, and the string
        data follows it directly.
        If all you need to do is align, `additionalBytes` will be 0.
        """

        # Track the biggest thing we've ever aligned to.
        if size > self.minalign:
            self.minalign = size

        # Find the amount of alignment needed such that `size` is properly
        # aligned after `additionalBytes`:
        alignSize = (~(len(self.Bytes) - self.Head() + additionalBytes)) + 1
        alignSize &= (size - 1)

        # Reallocate the buffer if needed:
        while self.Head() < alignSize+size+additionalBytes:
            oldBufSize = len(self.Bytes)
            self.growByteBuffer()
            updated_head = self.head + len(self.Bytes) - oldBufSize
            self.head = UOffsetTFlags.py_type(updated_head)
        self.Pad(alignSize)

    def PrependSOffsetTRelative(self, off):
        """
        PrependSOffsetTRelative prepends an SOffsetT, relative to where it
        will be written.
        """

        # Ensure alignment is already done:
        self.Prep(N.SOffsetTFlags.bytewidth, 0)
        if not (off <= self.Offset()):
            msg = "flatbuffers: Offset arithmetic error."
            raise OffsetArithmeticError(msg)
        off2 = self.Offset() - off + N.SOffsetTFlags.bytewidth
        self.PlaceSOffsetT(off2)
    ## @endcond

    def PrependUOffsetTRelative(self, off):
        """Prepends an unsigned offset into vector data, relative to where it
        will be written.
        """

        # Ensure alignment is already done:
        self.Prep(N.UOffsetTFlags.bytewidth, 0)
        if not (off <= self.Offset()):
            msg = "flatbuffers: Offset arithmetic error."
            raise OffsetArithmeticError(msg)
        off2 = self.Offset() - off + N.UOffsetTFlags.bytewidth
        self.PlaceUOffsetT(off2)

    ## @cond FLATBUFFERS_INTERNAL
    def StartVector(self, elemSize, numElems, alignment):
        """
        StartVector initializes bookkeeping for writing a new vector.

        A vector has the following format:
          - <UOffsetT: number of elements in this vector>
          - <T: data>+, where T is the type of elements of this vector.
        """

        self.assertNotNested()
        self.nested = True
        self.vectorNumElems = numElems
        self.Prep(N.Uint32Flags.bytewidth, elemSize*numElems)
        self.Prep(alignment, elemSize*numElems)  # In case alignment > int.
        return self.Offset()
    ## @endcond

    def EndVector(self, numElems = None):
        """EndVector writes data necessary to finish vector construction."""

        self.assertNested()
        ## @cond FLATBUFFERS_INTERNAL
        self.nested = False
        ## @endcond

        if numElems:
            warnings.warn("numElems is deprecated.",
                          DeprecationWarning, stacklevel=2)
            if numElems != self.vectorNumElems:
                raise EndVectorLengthMismatched();

        # we already made space for this, so write without PrependUint32
        self.PlaceUOffsetT(self.vectorNumElems)
        self.vectorNumElems = None
        return self.Offset()

    def CreateSharedString(self, s, encoding='utf-8', errors='strict'):
        """
        CreateSharedString checks if the string is already written to the buffer
        before calling CreateString.
        """

        if s in self.sharedStrings:
            return self.sharedStrings[s]

        off = self.CreateString(s, encoding, errors)
        self.sharedStrings[s] = off

        return off

    def CreateString(self, s, encoding='utf-8', errors='strict'):
        """CreateString writes a null-terminated byte string as a vector."""

        self.assertNotNested()
        ## @cond FLATBUFFERS_INTERNAL
        self.nested = True
        ## @endcond

        if isinstance(s, compat.string_types):
            x = s.encode(encoding, errors)
        elif isinstance(s, compat.binary_types):
            x = s
        else:
            raise TypeError("non-string passed to CreateString")

        self.Prep(N.UOffsetTFlags.bytewidth, (len(x)+1)*N.Uint8Flags.bytewidth)
        self.Place(0, N.Uint8Flags)

        l = UOffsetTFlags.py_type(len(s))
        ## @cond FLATBUFFERS_INTERNAL
        self.head = UOffsetTFlags.py_type(self.Head() - l)
        ## @endcond
        self.Bytes[self.Head():self.Head()+l] = x

        self.vectorNumElems = len(x)
        return self.EndVector()

    def CreateByteVector(self, x):
        """CreateString writes a byte vector."""

        self.assertNotNested()
        ## @cond FLATBUFFERS_INTERNAL
        self.nested = True
        ## @endcond

        if not isinstance(x, compat.binary_types):
            raise TypeError("non-byte vector passed to CreateByteVector")

        self.Prep(N.UOffsetTFlags.bytewidth, len(x)*N.Uint8Flags.bytewidth)

        l = UOffsetTFlags.py_type(len(x))
        ## @cond FLATBUFFERS_INTERNAL
        self.head = UOffsetTFlags.py_type(self.Head() - l)
        ## @endcond
        self.Bytes[self.Head():self.Head()+l] = x

        self.vectorNumElems = len(x)
        return self.EndVector()

    def CreateNumpyVector(self, x):
        """CreateNumpyVector writes a numpy array into the buffer."""

        if np is None:
            # Numpy is required for this feature
            raise NumpyRequiredForThisFeature("Numpy was not found.")

        if not isinstance(x, np.ndarray):
            raise TypeError("non-numpy-ndarray passed to CreateNumpyVector")

        if x.dtype.kind not in ['b', 'i', 'u', 'f']:
            raise TypeError("numpy-ndarray holds elements of unsupported datatype")

        if x.ndim > 1:
            raise TypeError("multidimensional-ndarray passed to CreateNumpyVector")

        self.StartVector(x.itemsize, x.size, x.dtype.alignment)

        # Ensure little endian byte ordering
        if x.dtype.str[0] == "<":
            x_lend = x
        else:
            x_lend = x.byteswap(inplace=False)

        # Calculate total length
        l = UOffsetTFlags.py_type(x_lend.itemsize * x_lend.size)
        ## @cond FLATBUFFERS_INTERNAL
        self.head = UOffsetTFlags.py_type(self.Head() - l)
        ## @endcond

        # tobytes ensures c_contiguous ordering
        self.Bytes[self.Head():self.Head()+l] = x_lend.tobytes(order='C')

        self.vectorNumElems = x.size
        return self.EndVector()

    ## @cond FLATBUFFERS_INTERNAL
    def assertNested(self):
        """
        Check that we are in the process of building an object.
        """

        if not self.nested:
            raise IsNotNestedError()

    def assertNotNested(self):
        """
        Check that no other objects are being built while making this
        object. If not, raise an exception.
        """

        if self.nested:
            raise IsNestedError()

    def assertStructIsInline(self, obj):
        """
        Structs are always stored inline, so need to be created right
        where they are used. You'll get this error if you created it
        elsewhere.
        """

        N.enforce_number(obj, N.UOffsetTFlags)
        if obj != self.Offset():
            msg = ("flatbuffers: Tried to write a Struct at an Offset that "
                   "is different from the current Offset of the Builder.")
            raise StructIsNotInlineError(msg)

    def Slot(self, slotnum):
        """
        Slot sets the vtable key `voffset` to the current location in the
        buffer.

        """
        self.assertNested()
        self.current_vtable[slotnum] = self.Offset()
    ## @endcond

    def __Finish(self, rootTable, sizePrefix, file_identifier=None):
        """Finish finalizes a buffer, pointing to the given `rootTable`."""
        N.enforce_number(rootTable, N.UOffsetTFlags)

        prepSize = N.UOffsetTFlags.bytewidth
        if file_identifier is not None:
            prepSize += N.Int32Flags.bytewidth
        if sizePrefix:
            prepSize += N.Int32Flags.bytewidth
        self.Prep(self.minalign, prepSize)

        if file_identifier is not None:
            self.Prep(N.UOffsetTFlags.bytewidth, encode.FILE_IDENTIFIER_LENGTH)

            # Convert bytes object file_identifier to an array of 4 8-bit integers,
            # and use big-endian to enforce size compliance.
            # https://docs.python.org/2/library/struct.html#format-characters
            file_identifier = N.struct.unpack(">BBBB", file_identifier)
            for i in range(encode.FILE_IDENTIFIER_LENGTH-1, -1, -1):
                # Place the bytes of the file_identifer in reverse order:
                self.Place(file_identifier[i], N.Uint8Flags)

        self.PrependUOffsetTRelative(rootTable)
        if sizePrefix:
            size = len(self.Bytes) - self.Head()
            N.enforce_number(size, N.Int32Flags)
            self.PrependInt32(size)
        self.finished = True
        return self.Head()

    def Finish(self, rootTable, file_identifier=None):
        """Finish finalizes a buffer, pointing to the given `rootTable`."""
        return self.__Finish(rootTable, False, file_identifier=file_identifier)

    def FinishSizePrefixed(self, rootTable, file_identifier=None):
        """
        Finish finalizes a buffer, pointing to the given `rootTable`,
        with the size prefixed.
        """
        return self.__Finish(rootTable, True, file_identifier=file_identifier)

    ## @cond FLATBUFFERS_INTERNAL
    def Prepend(self, flags, off):
        self.Prep(flags.bytewidth, 0)
        self.Place(off, flags)

    def PrependSlot(self, flags, o, x, d):
        if x is not None:
            N.enforce_number(x, flags)
        if d is not None:
            N.enforce_number(d, flags)
        if x != d or (self.forceDefaults and d is not None):
            self.Prepend(flags, x)
            self.Slot(o)

    def PrependBoolSlot(self, *args): self.PrependSlot(N.BoolFlags, *args)

    def PrependByteSlot(self, *args): self.PrependSlot(N.Uint8Flags, *args)

    def PrependUint8Slot(self, *args): self.PrependSlot(N.Uint8Flags, *args)

    def PrependUint16Slot(self, *args): self.PrependSlot(N.Uint16Flags, *args)

    def PrependUint32Slot(self, *args): self.PrependSlot(N.Uint32Flags, *args)

    def PrependUint64Slot(self, *args): self.PrependSlot(N.Uint64Flags, *args)

    def PrependInt8Slot(self, *args): self.PrependSlot(N.Int8Flags, *args)

    def PrependInt16Slot(self, *args): self.PrependSlot(N.Int16Flags, *args)

    def PrependInt32Slot(self, *args): self.PrependSlot(N.Int32Flags, *args)

    def PrependInt64Slot(self, *args): self.PrependSlot(N.Int64Flags, *args)

    def PrependFloat32Slot(self, *args): self.PrependSlot(N.Float32Flags,
                                                          *args)

    def PrependFloat64Slot(self, *args): self.PrependSlot(N.Float64Flags,
                                                          *args)

    def PrependUOffsetTRelativeSlot(self, o, x, d):
        """
        PrependUOffsetTRelativeSlot prepends an UOffsetT onto the object at
        vtable slot `o`. If value `x` equals default `d`, then the slot will
        be set to zero and no other data will be written.
        """

        if x != d or self.forceDefaults:
            self.PrependUOffsetTRelative(x)
            self.Slot(o)

    def PrependStructSlot(self, v, x, d):
        """
        PrependStructSlot prepends a struct onto the object at vtable slot `o`.
        Structs are stored inline, so nothing additional is being added.
        In generated code, `d` is always 0.
        """

        N.enforce_number(d, N.UOffsetTFlags)
        if x != d:
            self.assertStructIsInline(x)
            self.Slot(v)

    ## @endcond

    def PrependBool(self, x):
        """Prepend a `bool` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.BoolFlags, x)

    def PrependByte(self, x):
        """Prepend a `byte` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Uint8Flags, x)

    def PrependUint8(self, x):
        """Prepend an `uint8` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Uint8Flags, x)

    def PrependUint16(self, x):
        """Prepend an `uint16` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Uint16Flags, x)

    def PrependUint32(self, x):
        """Prepend an `uint32` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Uint32Flags, x)

    def PrependUint64(self, x):
        """Prepend an `uint64` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Uint64Flags, x)

    def PrependInt8(self, x):
        """Prepend an `int8` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Int8Flags, x)

    def PrependInt16(self, x):
        """Prepend an `int16` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Int16Flags, x)

    def PrependInt32(self, x):
        """Prepend an `int32` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Int32Flags, x)

    def PrependInt64(self, x):
        """Prepend an `int64` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Int64Flags, x)

    def PrependFloat32(self, x):
        """Prepend a `float32` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Float32Flags, x)

    def PrependFloat64(self, x):
        """Prepend a `float64` to the Builder buffer.

        Note: aligns and checks for space.
        """
        self.Prepend(N.Float64Flags, x)

    def ForceDefaults(self, forceDefaults):
        """
        In order to save space, fields that are set to their default value
        don't get serialized into the buffer. Forcing defaults provides a
        way to manually disable this optimization. When set to `True`, will
        always serialize default values.
        """
        self.forceDefaults = forceDefaults

##############################################################

    ## @cond FLATBUFFERS_INTERNAL
    def PrependVOffsetT(self, x): self.Prepend(N.VOffsetTFlags, x)

    def Place(self, x, flags):
        """
        Place prepends a value specified by `flags` to the Builder,
        without checking for available space.
        """

        N.enforce_number(x, flags)
        self.head = self.head - flags.bytewidth
        encode.Write(flags.packer_type, self.Bytes, self.Head(), x)

    def PlaceVOffsetT(self, x):
        """PlaceVOffsetT prepends a VOffsetT to the Builder, without checking
        for space.
        """
        N.enforce_number(x, N.VOffsetTFlags)
        self.head = self.head - N.VOffsetTFlags.bytewidth
        encode.Write(packer.voffset, self.Bytes, self.Head(), x)

    def PlaceSOffsetT(self, x):
        """PlaceSOffsetT prepends a SOffsetT to the Builder, without checking
        for space.
        """
        N.enforce_number(x, N.SOffsetTFlags)
        self.head = self.head - N.SOffsetTFlags.bytewidth
        encode.Write(packer.soffset, self.Bytes, self.Head(), x)

    def PlaceUOffsetT(self, x):
        """PlaceUOffsetT prepends a UOffsetT to the Builder, without checking
        for space.
        """
        N.enforce_number(x, N.UOffsetTFlags)
        self.head = self.head - N.UOffsetTFlags.bytewidth
        encode.Write(packer.uoffset, self.Bytes, self.Head(), x)
    ## @endcond

## @cond FLATBUFFERS_INTERNAL
def vtableEqual(a, objectStart, b):
    """vtableEqual compares an unwritten vtable to a written vtable."""

    N.enforce_number(objectStart, N.UOffsetTFlags)

    if len(a) * N.VOffsetTFlags.bytewidth != len(b):
        return False

    for i, elem in enumerate(a):
        x = encode.Get(packer.voffset, b, i * N.VOffsetTFlags.bytewidth)

        # Skip vtable entries that indicate a default value.
        if x == 0 and elem == 0:
            pass
        else:
            y = objectStart - elem
            if x != y:
                return False
    return True
## @endcond
## @}