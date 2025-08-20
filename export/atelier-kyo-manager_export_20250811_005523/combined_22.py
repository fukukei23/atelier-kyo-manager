
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\solveset.py ===
"""
This module contains functions to:

    - solve a single equation for a single variable, in any domain either real or complex.

    - solve a single transcendental equation for a single variable in any domain either real or complex.
      (currently supports solving in real domain only)

    - solve a system of linear equations with N variables and M equations.

    - solve a system of Non Linear Equations with N variables and M equations
"""
from sympy.core.sympify import sympify
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul,
                        Add, Basic)
from sympy.core.containers import Tuple
from sympy.core.function import (Lambda, expand_complex, AppliedUndef,
                                expand_log, _mexpand, expand_trig, nfloat)
from sympy.core.mod import Mod
from sympy.core.numbers import I, Number, Rational, oo
from sympy.core.intfunc import integer_log
from sympy.core.relational import Eq, Ne, Relational
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, _uniquely_named_symbol
from sympy.core.sympify import _sympify
from sympy.core.traversal import preorder_traversal
from sympy.external.gmpy import gcd as number_gcd, lcm as number_lcm
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.polys.polyroots import UnsolvableFactorError
from sympy.simplify.simplify import simplify, fraction, trigsimp, nsimplify
from sympy.simplify import powdenest, logcombine
from sympy.functions import (log, tan, cot, sin, cos, sec, csc, exp,
                             acos, asin, atan, acot, acsc, asec,
                             piecewise_fold, Piecewise)
from sympy.functions.combinatorial.numbers import totient
from sympy.functions.elementary.complexes import Abs, arg, re, im
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
                            sinh, cosh, tanh, coth, sech, csch,
                            asinh, acosh, atanh, acoth, asech, acsch)
from sympy.functions.elementary.miscellaneous import real_root
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import And, BooleanTrue
from sympy.sets import (FiniteSet, imageset, Interval, Intersection,
                        Union, ConditionSet, ImageSet, Complement, Contains)
from sympy.sets.sets import Set, ProductSet
from sympy.matrices import zeros, Matrix, MatrixBase
from sympy.ntheory.factor_ import divisors
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod
from sympy.polys import (roots, Poly, degree, together, PolynomialError,
                         RootOf, factor, lcm, gcd)
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import invert, groebner, poly
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,
    PolyNonlinearError)
from sympy.polys.matrices.linsolve import _linsolve
from sympy.solvers.solvers import (checksol, denoms, unrad,
    _simple_dens, recast_to_symbols)
from sympy.solvers.polysys import solve_poly_system
from sympy.utilities import filldedent
from sympy.utilities.iterables import (numbered_symbols, has_dups,
                                       is_sequence, iterable)
from sympy.calculus.util import periodicity, continuous_domain, function_range


from types import GeneratorType


class NonlinearError(ValueError):
    """Raised when unexpectedly encountering nonlinear equations"""
    pass


def _masked(f, *atoms):
    """Return ``f``, with all objects given by ``atoms`` replaced with
    Dummy symbols, ``d``, and the list of replacements, ``(d, e)``,
    where ``e`` is an object of type given by ``atoms`` in which
    any other instances of atoms have been recursively replaced with
    Dummy symbols, too. The tuples are ordered so that if they are
    applied in sequence, the origin ``f`` will be restored.

    Examples
    ========

    >>> from sympy import cos
    >>> from sympy.abc import x
    >>> from sympy.solvers.solveset import _masked

    >>> f = cos(cos(x) + 1)
    >>> f, reps = _masked(cos(1 + cos(x)), cos)
    >>> f
    _a1
    >>> reps
    [(_a1, cos(_a0 + 1)), (_a0, cos(x))]
    >>> for d, e in reps:
    ...     f = f.xreplace({d: e})
    >>> f
    cos(cos(x) + 1)
    """
    sym = numbered_symbols('a', cls=Dummy, real=True)
    mask = []
    for a in ordered(f.atoms(*atoms)):
        for i in mask:
            a = a.replace(*i)
        mask.append((a, next(sym)))
    for i, (o, n) in enumerate(mask):
        f = f.replace(o, n)
        mask[i] = (n, o)
    mask = list(reversed(mask))
    return f, mask


def _invert(f_x, y, x, domain=S.Complexes):
    r"""
    Reduce the complex valued equation $f(x) = y$ to a set of equations

    $$\left\{g(x) = h_1(y),\  g(x) = h_2(y),\ \dots,\  g(x) = h_n(y) \right\}$$

    where $g(x)$ is a simpler function than $f(x)$.  The return value is a tuple
    $(g(x), \mathrm{set}_h)$, where $g(x)$ is a function of $x$ and $\mathrm{set}_h$ is
    the set of function $\left\{h_1(y), h_2(y), \dots, h_n(y)\right\}$.
    Here, $y$ is not necessarily a symbol.

    $\mathrm{set}_h$ contains the functions, along with the information
    about the domain in which they are valid, through set
    operations. For instance, if :math:`y = |x| - n` is inverted
    in the real domain, then $\mathrm{set}_h$ is not simply
    $\{-n, n\}$ as the nature of `n` is unknown; rather, it is:

    $$ \left(\left[0, \infty\right) \cap \left\{n\right\}\right) \cup
                       \left(\left(-\infty, 0\right] \cap \left\{- n\right\}\right)$$

    By default, the complex domain is used which means that inverting even
    seemingly simple functions like $\exp(x)$ will give very different
    results from those obtained in the real domain.
    (In the case of $\exp(x)$, the inversion via $\log$ is multi-valued
    in the complex domain, having infinitely many branches.)

    If you are working with real values only (or you are not sure which
    function to use) you should probably set the domain to
    ``S.Reals`` (or use ``invert_real`` which does that automatically).


    Examples
    ========

    >>> from sympy.solvers.solveset import invert_complex, invert_real
    >>> from sympy.abc import x, y
    >>> from sympy import exp

    When does exp(x) == y?

    >>> invert_complex(exp(x), y, x)
    (x, ImageSet(Lambda(_n, I*(2*_n*pi + arg(y)) + log(Abs(y))), Integers))
    >>> invert_real(exp(x), y, x)
    (x, Intersection({log(y)}, Reals))

    When does exp(x) == 1?

    >>> invert_complex(exp(x), 1, x)
    (x, ImageSet(Lambda(_n, 2*_n*I*pi), Integers))
    >>> invert_real(exp(x), 1, x)
    (x, {0})

    See Also
    ========
    invert_real, invert_complex
    """
    x = sympify(x)
    if not x.is_Symbol:
        raise ValueError("x must be a symbol")
    f_x = sympify(f_x)
    if x not in f_x.free_symbols:
        raise ValueError("Inverse of constant function doesn't exist")
    y = sympify(y)
    if x in y.free_symbols:
        raise ValueError("y should be independent of x ")

    if domain.is_subset(S.Reals):
        x1, s = _invert_real(f_x, FiniteSet(y), x)
    else:
        x1, s = _invert_complex(f_x, FiniteSet(y), x)

    # f couldn't be inverted completely; return unmodified.
    if  x1 != x:
        return x1, s

    # Avoid adding gratuitous intersections with S.Complexes. Actual
    # conditions should be handled by the respective inverters.
    if domain is S.Complexes:
        return x1, s

    if isinstance(s, FiniteSet):
        return x1, s.intersect(domain)

    # "Fancier" solution sets like those obtained by inversion of trigonometric
    # functions already include general validity conditions (i.e. conditions on
    # the domain of the respective inverse functions), so we should avoid adding
    # blanket intersections with S.Reals. But subsets of R (or C) must still be
    # accounted for.
    if domain is S.Reals:
        return x1, s
    else:
        return x1, s.intersect(domain)


invert_complex = _invert


def invert_real(f_x, y, x):
    """
    Inverts a real-valued function. Same as :func:`invert_complex`, but sets
    the domain to ``S.Reals`` before inverting.
    """
    return _invert(f_x, y, x, S.Reals)


def _invert_real(f, g_ys, symbol):
    """Helper function for _invert."""

    if f == symbol or g_ys is S.EmptySet:
        return (symbol, g_ys)

    n = Dummy('n', real=True)

    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        return _invert_real(f.exp,
                            imageset(Lambda(n, log(n)), g_ys),
                            symbol)

    if hasattr(f, 'inverse') and f.inverse() is not None and not isinstance(f, (
            TrigonometricFunction,
            HyperbolicFunction,
            )):
        if len(f.args) > 1:
            raise ValueError("Only functions with one argument are supported.")
        return _invert_real(f.args[0],
                            imageset(Lambda(n, f.inverse()(n)), g_ys),
                            symbol)

    if isinstance(f, Abs):
        return _invert_abs(f.args[0], g_ys, symbol)

    if f.is_Add:
        # f = g + h
        g, h = f.as_independent(symbol)
        if g is not S.Zero:
            return _invert_real(h, imageset(Lambda(n, n - g), g_ys), symbol)

    if f.is_Mul:
        # f = g*h
        g, h = f.as_independent(symbol)

        if g is not S.One:
            return _invert_real(h, imageset(Lambda(n, n/g), g_ys), symbol)

    if f.is_Pow:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)

        if not expo_has_sym:

            if expo.is_rational:
                num, den = expo.as_numer_denom()

                if den % 2 == 0 and num % 2 == 1 and den.is_zero is False:
                    # Here we have f(x)**(num/den) = y
                    # where den is nonzero and even and y is an element
                    # of the set g_ys.
                    # den is even, so we are only interested in the cases
                    # where both f(x) and y are positive.
                    # Restricting y to be positive (using the set g_ys_pos)
                    # means that y**(den/num) is always positive.
                    # Therefore it isn't necessary to also constrain f(x)
                    # to be positive because we are only going to
                    # find solutions of f(x) = y**(d/n)
                    # where the rhs is already required to be positive.
                    root = Lambda(n, real_root(n, expo))
                    g_ys_pos = g_ys & Interval(0, oo)
                    res = imageset(root, g_ys_pos)
                    _inv, _set = _invert_real(base, res, symbol)
                    return (_inv, _set)

                if den % 2 == 1:
                    root = Lambda(n, real_root(n, expo))
                    res = imageset(root, g_ys)
                    if num % 2 == 0:
                        neg_res = imageset(Lambda(n, -n), res)
                        return _invert_real(base, res + neg_res, symbol)
                    if num % 2 == 1:
                        return _invert_real(base, res, symbol)

            elif expo.is_irrational:
                root = Lambda(n, real_root(n, expo))
                g_ys_pos = g_ys & Interval(0, oo)
                res = imageset(root, g_ys_pos)
                return _invert_real(base, res, symbol)

            else:
                # indeterminate exponent, e.g. Float or parity of
                # num, den of rational could not be determined
                pass  # use default return

        if not base_has_sym:
            rhs = g_ys.args[0]
            if base.is_positive:
                return _invert_real(expo,
                    imageset(Lambda(n, log(n, base, evaluate=False)), g_ys), symbol)
            elif base.is_negative:
                s, b = integer_log(rhs, base)
                if b:
                    return _invert_real(expo, FiniteSet(s), symbol)
                else:
                    return (expo, S.EmptySet)
            elif base.is_zero:
                one = Eq(rhs, 1)
                if one == S.true:
                    # special case: 0**x - 1
                    return _invert_real(expo, FiniteSet(0), symbol)
                elif one == S.false:
                    return (expo, S.EmptySet)

    if isinstance(f, (TrigonometricFunction, HyperbolicFunction)):
        return _invert_trig_hyp_real(f, g_ys, symbol)

    return (f, g_ys)


# Dictionaries of inverses will be cached after first use.
_trig_inverses = None
_hyp_inverses = None

def _invert_trig_hyp_real(f, g_ys, symbol):
    """Helper function for inverting trigonometric and hyperbolic functions.

    This helper only handles inversion over the reals.

    For trigonometric functions only finite `g_ys` sets are implemented.

    For hyperbolic functions the set `g_ys` is checked against the domain of the
    respective inverse functions. Infinite `g_ys` sets are also supported.
    """

    if isinstance(f, HyperbolicFunction):
        n = Dummy('n', real=True)

        if isinstance(f, sinh):
            # asinh is defined over R.
            return _invert_real(f.args[0], imageset(n, asinh(n), g_ys), symbol)

        if isinstance(f, cosh):
            g_ys_dom = g_ys.intersect(Interval(1, oo))
            if isinstance(g_ys_dom, Intersection):
                # could not properly resolve domain check
                if isinstance(g_ys, FiniteSet):
                    # If g_ys is a `FiniteSet`` it should be sufficient to just
                    # let the calling `_invert_real()` add an intersection with
                    # `S.Reals` (or a subset `domain`) to ensure that only valid
                    # (real) solutions are returned.
                    # This avoids adding "too many" Intersections or
                    # ConditionSets in the returned set.
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            return _invert_real(f.args[0], Union(
                imageset(n, acosh(n), g_ys_dom),
                imageset(n, -acosh(n), g_ys_dom)), symbol)

        if isinstance(f, sech):
            g_ys_dom = g_ys.intersect(Interval.Lopen(0, 1))
            if isinstance(g_ys_dom, Intersection):
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            return _invert_real(f.args[0], Union(
                imageset(n, asech(n), g_ys_dom),
                imageset(n, -asech(n), g_ys_dom)), symbol)

        if isinstance(f, tanh):
            g_ys_dom = g_ys.intersect(Interval.open(-1, 1))
            if isinstance(g_ys_dom, Intersection):
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            return _invert_real(f.args[0],
                imageset(n, atanh(n), g_ys_dom), symbol)

        if isinstance(f, coth):
            g_ys_dom = g_ys - Interval(-1, 1)
            if isinstance(g_ys_dom, Complement):
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            return _invert_real(f.args[0],
                imageset(n, acoth(n), g_ys_dom), symbol)

        if isinstance(f, csch):
            g_ys_dom = g_ys - FiniteSet(0)
            if isinstance(g_ys_dom, Complement):
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            return _invert_real(f.args[0],
                imageset(n, acsch(n), g_ys_dom), symbol)

    elif isinstance(f, TrigonometricFunction) and isinstance(g_ys, FiniteSet):
        def _get_trig_inverses(func):
            global _trig_inverses
            if _trig_inverses is None:
                _trig_inverses = {
                    sin : ((asin, lambda y: pi-asin(y)), 2*pi, Interval(-1, 1)),
                    cos : ((acos, lambda y: -acos(y)), 2*pi, Interval(-1, 1)),
                    tan : ((atan,), pi, S.Reals),
                    cot : ((acot,), pi, S.Reals),
                    sec : ((asec, lambda y: -asec(y)), 2*pi,
                        Union(Interval(-oo, -1), Interval(1, oo))),
                    csc : ((acsc, lambda y: pi-acsc(y)), 2*pi,
                        Union(Interval(-oo, -1), Interval(1, oo)))}
            return _trig_inverses[func]

        invs, period, rng = _get_trig_inverses(f.func)
        n = Dummy('n', integer=True)
        def create_return_set(g):
            # returns ConditionSet that will be part of the final (x, set) tuple
            invsimg = Union(*[
                imageset(n, period*n + inv(g), S.Integers) for inv in invs])
            inv_f, inv_g_ys = _invert_real(f.args[0], invsimg, symbol)
            if inv_f == symbol:     # inversion successful
                conds = rng.contains(g)
                return ConditionSet(symbol, conds, inv_g_ys)
            else:
                return ConditionSet(symbol, Eq(f, g), S.Reals)

        retset = Union(*[create_return_set(g) for g in g_ys])
        return (symbol, retset)

    else:
        return (f, g_ys)


def _invert_trig_hyp_complex(f, g_ys, symbol):
    """Helper function for inverting trigonometric and hyperbolic functions.

    This helper only handles inversion over the complex numbers.
    Only finite `g_ys` sets are implemented.

    Handling of singularities is only implemented for hyperbolic equations.
    In case of a symbolic element g in g_ys a ConditionSet may be returned.
    """

    if isinstance(f, TrigonometricFunction) and isinstance(g_ys, FiniteSet):
        def inv(trig):
            if isinstance(trig, (sin, csc)):
                F = asin if isinstance(trig, sin) else acsc
                return (
                    lambda a: 2*n*pi + F(a),
                    lambda a: 2*n*pi + pi - F(a))
            if isinstance(trig, (cos, sec)):
                F = acos if isinstance(trig, cos) else asec
                return (
                    lambda a: 2*n*pi + F(a),
                    lambda a: 2*n*pi - F(a))
            if isinstance(trig, (tan, cot)):
                return (lambda a: n*pi + trig.inverse()(a),)

        n = Dummy('n', integer=True)
        invs = S.EmptySet
        for L in inv(f):
            invs += Union(*[imageset(Lambda(n, L(g)), S.Integers) for g in g_ys])
        return _invert_complex(f.args[0], invs, symbol)

    elif isinstance(f, HyperbolicFunction) and isinstance(g_ys, FiniteSet):
        # There are two main options regarding singularities / domain checking
        # for symbolic elements in g_ys:
        # 1. Add a "catch-all" intersection with S.Complexes.
        # 2. ConditionSets.
        # At present ConditionSets seem to work better and have the additional
        # benefit of representing the precise conditions that must be satisfied.
        # The conditions are also rather straightforward. (At most two isolated
        # points.)
        def _get_hyp_inverses(func):
            global _hyp_inverses
            if _hyp_inverses is None:
                _hyp_inverses = {
                    sinh : ((asinh, lambda y: I*pi-asinh(y)), 2*I*pi, ()),
                    cosh : ((acosh, lambda y: -acosh(y)), 2*I*pi, ()),
                    tanh : ((atanh,), I*pi, (-1, 1)),
                    coth : ((acoth,), I*pi, (-1, 1)),
                    sech : ((asech, lambda y: -asech(y)), 2*I*pi, (0, )),
                    csch : ((acsch, lambda y: I*pi-acsch(y)), 2*I*pi, (0, ))}
            return _hyp_inverses[func]

        # invs: iterable of main inverses, e.g. (acosh, -acosh).
        # excl: iterable of singularities to be checked for.
        invs, period, excl = _get_hyp_inverses(f.func)
        n = Dummy('n', integer=True)
        def create_return_set(g):
            # returns ConditionSet that will be part of the final (x, set) tuple
            invsimg = Union(*[
                imageset(n, period*n + inv(g), S.Integers) for inv in invs])
            inv_f, inv_g_ys = _invert_complex(f.args[0], invsimg, symbol)
            if inv_f == symbol:     # inversion successful
                conds = And(*[Ne(g, e) for e in excl])
                return ConditionSet(symbol, conds, inv_g_ys)
            else:
                return ConditionSet(symbol, Eq(f, g), S.Complexes)

        retset = Union(*[create_return_set(g) for g in g_ys])
        return (symbol, retset)

    else:
        return (f, g_ys)


def _invert_complex(f, g_ys, symbol):
    """Helper function for _invert."""

    if f == symbol or g_ys is S.EmptySet:
        return (symbol, g_ys)

    n = Dummy('n')

    if f.is_Add:
        # f = g + h
        g, h = f.as_independent(symbol)
        if g is not S.Zero:
            return _invert_complex(h, imageset(Lambda(n, n - g), g_ys), symbol)

    if f.is_Mul:
        # f = g*h
        g, h = f.as_independent(symbol)

        if g is not S.One:
            if g in {S.NegativeInfinity, S.ComplexInfinity, S.Infinity}:
                return (h, S.EmptySet)
            return _invert_complex(h, imageset(Lambda(n, n/g), g_ys), symbol)

    if f.is_Pow:
        base, expo = f.args
        # special case: g**r = 0
        # Could be improved like `_invert_real` to handle more general cases.
        if expo.is_Rational and g_ys == FiniteSet(0):
            if expo.is_positive:
                return _invert_complex(base, g_ys, symbol)

    if hasattr(f, 'inverse') and f.inverse() is not None and \
       not isinstance(f, TrigonometricFunction) and \
       not isinstance(f, HyperbolicFunction) and \
       not isinstance(f, exp):
        if len(f.args) > 1:
            raise ValueError("Only functions with one argument are supported.")
        return _invert_complex(f.args[0],
                               imageset(Lambda(n, f.inverse()(n)), g_ys), symbol)

    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        if isinstance(g_ys, ImageSet):
            # can solve up to `(d*exp(exp(...(exp(a*x + b))...) + c)` format.
            # Further can be improved to `(d*exp(exp(...(exp(a*x**n + b*x**(n-1) + ... + f))...) + c)`.
            g_ys_expr = g_ys.lamda.expr
            g_ys_vars = g_ys.lamda.variables
            k = Dummy('k{}'.format(len(g_ys_vars)))
            g_ys_vars_1 = (k,) + g_ys_vars
            exp_invs = Union(*[imageset(Lambda((g_ys_vars_1,), (I*(2*k*pi + arg(g_ys_expr))
                                         + log(Abs(g_ys_expr)))), S.Integers**(len(g_ys_vars_1)))])
            return _invert_complex(f.exp, exp_invs, symbol)

        elif isinstance(g_ys, FiniteSet):
            exp_invs = Union(*[imageset(Lambda(n, I*(2*n*pi + arg(g_y)) +
                                               log(Abs(g_y))), S.Integers)
                               for g_y in g_ys if g_y != 0])
            return _invert_complex(f.exp, exp_invs, symbol)

    if isinstance(f, (TrigonometricFunction, HyperbolicFunction)):
        return _invert_trig_hyp_complex(f, g_ys, symbol)

    return (f, g_ys)


def _invert_abs(f, g_ys, symbol):
    """Helper function for inverting absolute value functions.

    Returns the complete result of inverting an absolute value
    function along with the conditions which must also be satisfied.

    If it is certain that all these conditions are met, a :class:`~.FiniteSet`
    of all possible solutions is returned. If any condition cannot be
    satisfied, an :class:`~.EmptySet` is returned. Otherwise, a
    :class:`~.ConditionSet` of the solutions, with all the required conditions
    specified, is returned.

    """
    if not g_ys.is_FiniteSet:
        # this could be used for FiniteSet, but the
        # results are more compact if they aren't, e.g.
        # ConditionSet(x, Contains(n, Interval(0, oo)), {-n, n}) vs
        # Union(Intersection(Interval(0, oo), {n}), Intersection(Interval(-oo, 0), {-n}))
        # for the solution of abs(x) - n
        pos = Intersection(g_ys, Interval(0, S.Infinity))
        parg = _invert_real(f, pos, symbol)
        narg = _invert_real(-f, pos, symbol)
        if parg[0] != narg[0]:
            raise NotImplementedError
        return parg[0], Union(narg[1], parg[1])

    # check conditions: all these must be true. If any are unknown
    # then return them as conditions which must be satisfied
    unknown = []
    for a in g_ys.args:
        ok = a.is_nonnegative if a.is_Number else a.is_positive
        if ok is None:
            unknown.append(a)
        elif not ok:
            return symbol, S.EmptySet
    if unknown:
        conditions = And(*[Contains(i, Interval(0, oo))
            for i in unknown])
    else:
        conditions = True
    n = Dummy('n', real=True)
    # this is slightly different than above: instead of solving
    # +/-f on positive values, here we solve for f on +/- g_ys
    g_x, values = _invert_real(f, Union(
        imageset(Lambda(n, n), g_ys),
        imageset(Lambda(n, -n), g_ys)), symbol)
    return g_x, ConditionSet(g_x, conditions, values)


def domain_check(f, symbol, p):
    """Returns False if point p is infinite or any subexpression of f
    is infinite or becomes so after replacing symbol with p. If none of
    these conditions is met then True will be returned.

    Examples
    ========

    >>> from sympy import Mul, oo
    >>> from sympy.abc import x
    >>> from sympy.solvers.solveset import domain_check
    >>> g = 1/(1 + (1/(x + 1))**2)
    >>> domain_check(g, x, -1)
    False
    >>> domain_check(x**2, x, 0)
    True
    >>> domain_check(1/x, x, oo)
    False

    * The function relies on the assumption that the original form
      of the equation has not been changed by automatic simplification.

    >>> domain_check(x/x, x, 0) # x/x is automatically simplified to 1
    True

    * To deal with automatic evaluations use evaluate=False:

    >>> domain_check(Mul(x, 1/x, evaluate=False), x, 0)
    False
    """
    f, p = sympify(f), sympify(p)
    if p.is_infinite:
        return False
    return _domain_check(f, symbol, p)


def _domain_check(f, symbol, p):
    # helper for domain check
    if f.is_Atom and f.is_finite:
        return True
    elif f.subs(symbol, p).is_infinite:
        return False
    elif isinstance(f, Piecewise):
        # Check the cases of the Piecewise in turn. There might be invalid
        # expressions in later cases that don't apply e.g.
        #    solveset(Piecewise((0, Eq(x, 0)), (1/x, True)), x)
        for expr, cond in f.args:
            condsubs = cond.subs(symbol, p)
            if condsubs is S.false:
                continue
            elif condsubs is S.true:
                return _domain_check(expr, symbol, p)
            else:
                # We don't know which case of the Piecewise holds. On this
                # basis we cannot decide whether any solution is in or out of
                # the domain. Ideally this function would allow returning a
                # symbolic condition for the validity of the solution that
                # could be handled in the calling code. In the mean time we'll
                # give this particular solution the benefit of the doubt and
                # let it pass.
                return True
    else:
        # TODO : We should not blindly recurse through all args of arbitrary expressions like this
        return all(_domain_check(g, symbol, p)
                   for g in f.args)


def _is_finite_with_finite_vars(f, domain=S.Complexes):
    """
    Return True if the given expression is finite. For symbols that
    do not assign a value for `complex` and/or `real`, the domain will
    be used to assign a value; symbols that do not assign a value
    for `finite` will be made finite. All other assumptions are
    left unmodified.
    """
    def assumptions(s):
        A = s.assumptions0
        A.setdefault('finite', A.get('finite', True))
        if domain.is_subset(S.Reals):
            # if this gets set it will make complex=True, too
            A.setdefault('real', True)
        else:
            # don't change 'real' because being complex implies
            # nothing about being real
            A.setdefault('complex', True)
        return A

    reps = {s: Dummy(**assumptions(s)) for s in f.free_symbols}
    return f.xreplace(reps).is_finite


def _is_function_class_equation(func_class, f, symbol):
    """ Tests whether the equation is an equation of the given function class.

    The given equation belongs to the given function class if it is
    comprised of functions of the function class which are multiplied by
    or added to expressions independent of the symbol. In addition, the
    arguments of all such functions must be linear in the symbol as well.

    Examples
    ========

    >>> from sympy.solvers.solveset import _is_function_class_equation
    >>> from sympy import tan, sin, tanh, sinh, exp
    >>> from sympy.abc import x
    >>> from sympy.functions.elementary.trigonometric import TrigonometricFunction
    >>> from sympy.functions.elementary.hyperbolic import HyperbolicFunction
    >>> _is_function_class_equation(TrigonometricFunction, exp(x) + tan(x), x)
    False
    >>> _is_function_class_equation(TrigonometricFunction, tan(x) + sin(x), x)
    True
    >>> _is_function_class_equation(TrigonometricFunction, tan(x**2), x)
    False
    >>> _is_function_class_equation(TrigonometricFunction, tan(x + 2), x)
    True
    >>> _is_function_class_equation(HyperbolicFunction, tanh(x) + sinh(x), x)
    True
    """
    if f.is_Mul or f.is_Add:
        return all(_is_function_class_equation(func_class, arg, symbol)
                   for arg in f.args)

    if f.is_Pow:
        if not f.exp.has(symbol):
            return _is_function_class_equation(func_class, f.base, symbol)
        else:
            return False

    if not f.has(symbol):
        return True

    if isinstance(f, func_class):
        try:
            g = Poly(f.args[0], symbol)
            return g.degree() <= 1
        except PolynomialError:
            return False
    else:
        return False


def _solve_as_rational(f, symbol, domain):
    """ solve rational functions"""
    f = together(_mexpand(f, recursive=True), deep=True)
    g, h = fraction(f)
    if not h.has(symbol):
        try:
            return _solve_as_poly(g, symbol, domain)
        except NotImplementedError:
            # The polynomial formed from g could end up having
            # coefficients in a ring over which finding roots
            # isn't implemented yet, e.g. ZZ[a] for some symbol a
            return ConditionSet(symbol, Eq(f, 0), domain)
        except CoercionFailed:
            # contained oo, zoo or nan
            return S.EmptySet
    else:
        valid_solns = _solveset(g, symbol, domain)
        invalid_solns = _solveset(h, symbol, domain)
        return valid_solns - invalid_solns


class _SolveTrig1Error(Exception):
    """Raised when _solve_trig1 heuristics do not apply"""

def _solve_trig(f, symbol, domain):
    """Function to call other helpers to solve trigonometric equations """
    # If f is composed of a single trig function (potentially appearing multiple
    # times) we should solve by either inverting directly or inverting after a
    # suitable change of variable.
    #
    # _solve_trig is currently only called by _solveset for trig/hyperbolic
    # functions of an argument linear in x. Inverting a symbolic argument should
    # include a guard against division by zero in order to have a result that is
    # consistent with similar processing done by _solve_trig1.
    # (Ideally _invert should add these conditions by itself.)
    trig_expr, count = None, 0
    for expr in preorder_traversal(f):
        if isinstance(expr, (TrigonometricFunction,
                            HyperbolicFunction)) and expr.has(symbol):
            if not trig_expr:
                trig_expr, count = expr, 1
            elif expr == trig_expr:
                count += 1
            else:
                trig_expr, count = False, 0
                break
    if count == 1:
        # direct inversion
        x, sol = _invert(f, 0, symbol, domain)
        if x == symbol:
            cond = True
            if trig_expr.free_symbols - {symbol}:
                a, h = trig_expr.args[0].as_independent(symbol, as_Add=True)
                m, h = h.as_independent(symbol, as_Add=False)
                num, den = m.as_numer_denom()
                cond = Ne(num, 0) & Ne(den, 0)
            return ConditionSet(symbol, cond, sol)
        else:
            return ConditionSet(symbol, Eq(f, 0), domain)
    elif count:
        # solve by change of variable
        y = Dummy('y')
        f_cov = f.subs(trig_expr, y)
        sol_cov = solveset(f_cov, y, domain)
        if isinstance(sol_cov, FiniteSet):
            return Union(
                *[_solve_trig(trig_expr-s, symbol, domain) for s in sol_cov])

    sol = None
    try:
        # multiple trig/hyp functions; solve by rewriting to exp
        sol = _solve_trig1(f, symbol, domain)
    except _SolveTrig1Error:
        try:
            # multiple trig/hyp functions; solve by rewriting to tan(x/2)
            sol = _solve_trig2(f, symbol, domain)
        except ValueError:
            raise NotImplementedError(filldedent('''
                Solution to this kind of trigonometric equations
                is yet to be implemented'''))
    return sol


def _solve_trig1(f, symbol, domain):
    """Primary solver for trigonometric and hyperbolic equations

    Returns either the solution set as a ConditionSet (auto-evaluated to a
    union of ImageSets if no variables besides 'symbol' are involved) or
    raises _SolveTrig1Error if f == 0 cannot be solved.

    Notes
    =====
    Algorithm:
    1. Do a change of variable x -> mu*x in arguments to trigonometric and
    hyperbolic functions, in order to reduce them to small integers. (This
    step is crucial to keep the degrees of the polynomials of step 4 low.)
    2. Rewrite trigonometric/hyperbolic functions as exponentials.
    3. Proceed to a 2nd change of variable, replacing exp(I*x) or exp(x) by y.
    4. Solve the resulting rational equation.
    5. Use invert_complex or invert_real to return to the original variable.
    6. If the coefficients of 'symbol' were symbolic in nature, add the
    necessary consistency conditions in a ConditionSet.

    """
    # Prepare change of variable
    x = Dummy('x')
    if _is_function_class_equation(HyperbolicFunction, f, symbol):
        cov = exp(x)
        inverter = invert_real if domain.is_subset(S.Reals) else invert_complex
    else:
        cov = exp(I*x)
        inverter = invert_complex

    f = trigsimp(f)
    f_original = f
    trig_functions = f.atoms(TrigonometricFunction, HyperbolicFunction)
    trig_arguments = [e.args[0] for e in trig_functions]
    # trigsimp may have reduced the equation to an expression
    # that is independent of 'symbol' (e.g. cos**2+sin**2)
    if not any(a.has(symbol) for a in trig_arguments):
        return solveset(f_original, symbol, domain)

    denominators = []
    numerators = []
    for ar in trig_arguments:
        try:
            poly_ar = Poly(ar, symbol)
        except PolynomialError:
            raise _SolveTrig1Error("trig argument is not a polynomial")
        if poly_ar.degree() > 1:  # degree >1 still bad
            raise _SolveTrig1Error("degree of variable must not exceed one")
        if poly_ar.degree() == 0:  # degree 0, don't care
            continue
        c = poly_ar.all_coeffs()[0]   # got the coefficient of 'symbol'
        numerators.append(fraction(c)[0])
        denominators.append(fraction(c)[1])

    mu = lcm(denominators)/gcd(numerators)
    f = f.subs(symbol, mu*x)
    f = f.rewrite(exp)
    f = together(f)
    g, h = fraction(f)
    y = Dummy('y')
    g, h = g.expand(), h.expand()
    g, h = g.subs(cov, y), h.subs(cov, y)
    if g.has(x) or h.has(x):
        raise _SolveTrig1Error("change of variable not possible")

    solns = solveset_complex(g, y) - solveset_complex(h, y)
    if isinstance(solns, ConditionSet):
        raise _SolveTrig1Error("polynomial has ConditionSet solution")

    if isinstance(solns, FiniteSet):
        if any(isinstance(s, RootOf) for s in solns):
            raise _SolveTrig1Error("polynomial results in RootOf object")
        # revert the change of variable
        cov = cov.subs(x, symbol/mu)
        result = Union(*[inverter(cov, s, symbol)[1] for s in solns])
        # In case of symbolic coefficients, the solution set is only valid
        # if numerator and denominator of mu are non-zero.
        if mu.has(Symbol):
            syms = (mu).atoms(Symbol)
            munum, muden = fraction(mu)
            condnum = munum.as_independent(*syms, as_Add=False)[1]
            condden = muden.as_independent(*syms, as_Add=False)[1]
            cond = And(Ne(condnum, 0), Ne(condden, 0))
        else:
            cond = True
        # Actual conditions are returned as part of the ConditionSet. Adding an
        # intersection with C would only complicate some solution sets due to
        # current limitations of intersection code. (e.g. #19154)
        if domain is S.Complexes:
            # This is a slight abuse of ConditionSet. Ideally this should
            # be some kind of "PiecewiseSet". (See #19507 discussion)
            return ConditionSet(symbol, cond, result)
        else:
            return ConditionSet(symbol, cond, Intersection(result, domain))
    elif solns is S.EmptySet:
        return S.EmptySet
    else:
        raise _SolveTrig1Error("polynomial solutions must form FiniteSet")


def _solve_trig2(f, symbol, domain):
    """Secondary helper to solve trigonometric equations,
    called when first helper fails """
    f = trigsimp(f)
    f_original = f
    trig_functions = f.atoms(sin, cos, tan, sec, cot, csc)
    trig_arguments = [e.args[0] for e in trig_functions]
    denominators = []
    numerators = []

    # todo: This solver can be extended to hyperbolics if the
    # analogous change of variable to tanh (instead of tan)
    # is used.
    if not trig_functions:
        return ConditionSet(symbol, Eq(f_original, 0), domain)

    # todo: The pre-processing below (extraction of numerators, denominators,
    # gcd, lcm, mu, etc.) should be updated to the enhanced version in
    # _solve_trig1. (See #19507)
    for ar in trig_arguments:
        try:
            poly_ar = Poly(ar, symbol)
        except PolynomialError:
            raise ValueError("give up, we cannot solve if this is not a polynomial in x")
        if poly_ar.degree() > 1:  # degree >1 still bad
            raise ValueError("degree of variable inside polynomial should not exceed one")
        if poly_ar.degree() == 0:  # degree 0, don't care
            continue
        c = poly_ar.all_coeffs()[0]   # got the coefficient of 'symbol'
        try:
            numerators.append(Rational(c).p)
            denominators.append(Rational(c).q)
        except TypeError:
            return ConditionSet(symbol, Eq(f_original, 0), domain)

    x = Dummy('x')

    mu = Rational(2)*number_lcm(*denominators)/number_gcd(*numerators)
    f = f.subs(symbol, mu*x)
    f = f.rewrite(tan)
    f = expand_trig(f)
    f = together(f)

    g, h = fraction(f)
    y = Dummy('y')
    g, h = g.expand(), h.expand()
    g, h = g.subs(tan(x), y), h.subs(tan(x), y)

    if g.has(x) or h.has(x):
        return ConditionSet(symbol, Eq(f_original, 0), domain)
    solns = solveset(g, y, S.Reals) - solveset(h, y, S.Reals)

    if isinstance(solns, FiniteSet):
        result = Union(*[invert_real(tan(symbol/mu), s, symbol)[1]
                       for s in solns])
        dsol = invert_real(tan(symbol/mu), oo, symbol)[1]
        if degree(h) > degree(g):                   # If degree(denom)>degree(num) then there
            result = Union(result, dsol)            # would be another sol at Lim(denom-->oo)
        return Intersection(result, domain)
    elif solns is S.EmptySet:
        return S.EmptySet
    else:
        return ConditionSet(symbol, Eq(f_original, 0), S.Reals)


def _solve_as_poly(f, symbol, domain=S.Complexes):
    """
    Solve the equation using polynomial techniques if it already is a
    polynomial equation or, with a change of variables, can be made so.
    """
    result = None
    if f.is_polynomial(symbol):
        solns = roots(f, symbol, cubics=True, quartics=True,
                      quintics=True, domain='EX')
        num_roots = sum(solns.values())
        if degree(f, symbol) <= num_roots:
            result = FiniteSet(*solns.keys())
        else:
            poly = Poly(f, symbol)
            solns = poly.all_roots()
            if poly.degree() <= len(solns):
                result = FiniteSet(*solns)
            else:
                result = ConditionSet(symbol, Eq(f, 0), domain)
    else:
        poly = Poly(f)
        if poly is None:
            result = ConditionSet(symbol, Eq(f, 0), domain)
        gens = [g for g in poly.gens if g.has(symbol)]

        if len(gens) == 1:
            poly = Poly(poly, gens[0])
            gen = poly.gen
            deg = poly.degree()
            poly = Poly(poly.as_expr(), poly.gen, composite=True)
            poly_solns = FiniteSet(*roots(poly, cubics=True, quartics=True,
                                          quintics=True).keys())

            if len(poly_solns) < deg:
                result = ConditionSet(symbol, Eq(f, 0), domain)

            if gen != symbol:
                y = Dummy('y')
                inverter = invert_real if domain.is_subset(S.Reals) else invert_complex
                lhs, rhs_s = inverter(gen, y, symbol)
                if lhs == symbol:
                    result = Union(*[rhs_s.subs(y, s) for s in poly_solns])
                    if isinstance(result, FiniteSet) and isinstance(gen, Pow
                            ) and gen.base.is_Rational:
                        result = FiniteSet(*[expand_log(i) for i in result])
                else:
                    result = ConditionSet(symbol, Eq(f, 0), domain)
        else:
            result = ConditionSet(symbol, Eq(f, 0), domain)

    if result is not None:
        if isinstance(result, FiniteSet):
            # this is to simplify solutions like -sqrt(-I) to sqrt(2)/2
            # - sqrt(2)*I/2. We are not expanding for solution with symbols
            # or undefined functions because that makes the solution more complicated.
            # For example, expand_complex(a) returns re(a) + I*im(a)
            if all(s.atoms(Symbol, AppliedUndef) == set() and not isinstance(s, RootOf)
                   for s in result):
                s = Dummy('s')
                result = imageset(Lambda(s, expand_complex(s)), result)
        if isinstance(result, FiniteSet) and domain != S.Complexes:
            # Avoid adding gratuitous intersections with S.Complexes. Actual
            # conditions should be handled elsewhere.
            result = result.intersection(domain)
        return result
    else:
        return ConditionSet(symbol, Eq(f, 0), domain)


def _solve_radical(f, unradf, symbol, solveset_solver):
    """ Helper function to solve equations with radicals """
    res = unradf
    eq, cov = res if res else (f, [])
    if not cov:
        result = solveset_solver(eq, symbol) - \
            Union(*[solveset_solver(g, symbol) for g in denoms(f, symbol)])
    else:
        y, yeq = cov
        if not solveset_solver(y - I, y):
            yreal = Dummy('yreal', real=True)
            yeq = yeq.xreplace({y: yreal})
            eq = eq.xreplace({y: yreal})
            y = yreal
        g_y_s = solveset_solver(yeq, symbol)
        f_y_sols = solveset_solver(eq, y)
        result = Union(*[imageset(Lambda(y, g_y), f_y_sols)
                         for g_y in g_y_s])

    def check_finiteset(solutions):
        f_set = []  # solutions for FiniteSet
        c_set = []  # solutions for ConditionSet
        for s in solutions:
            if checksol(f, symbol, s):
                f_set.append(s)
            else:
                c_set.append(s)
        return FiniteSet(*f_set) + ConditionSet(symbol, Eq(f, 0), FiniteSet(*c_set))

    def check_set(solutions):
        if solutions is S.EmptySet:
            return solutions
        elif isinstance(solutions, ConditionSet):
            # XXX: Maybe the base set should be checked?
            return solutions
        elif isinstance(solutions, FiniteSet):
            return check_finiteset(solutions)
        elif isinstance(solutions, Complement):
            A, B = solutions.args
            return Complement(check_set(A), B)
        elif isinstance(solutions, Union):
            return Union(*[check_set(s) for s in solutions.args])
        else:
            # XXX: There should be more cases checked here. The cases above
            # are all those that come up in the test suite for now.
            return solutions

    solution_set = check_set(result)

    return solution_set


def _solve_abs(f, symbol, domain):
    """ Helper function to solve equation involving absolute value function """
    if not domain.is_subset(S.Reals):
        raise ValueError(filldedent('''
            Absolute values cannot be inverted in the
            complex domain.'''))
    p, q, r = Wild('p'), Wild('q'), Wild('r')
    pattern_match = f.match(p*Abs(q) + r) or {}
    f_p, f_q, f_r = [pattern_match.get(i, S.Zero) for i in (p, q, r)]

    if not (f_p.is_zero or f_q.is_zero):
        domain = continuous_domain(f_q, symbol, domain)
        from .inequalities import solve_univariate_inequality
        q_pos_cond = solve_univariate_inequality(f_q >= 0, symbol,
                                                 relational=False, domain=domain, continuous=True)
        q_neg_cond = q_pos_cond.complement(domain)

        sols_q_pos = solveset_real(f_p*f_q + f_r,
                                           symbol).intersect(q_pos_cond)
        sols_q_neg = solveset_real(f_p*(-f_q) + f_r,
                                           symbol).intersect(q_neg_cond)
        return Union(sols_q_pos, sols_q_neg)
    else:
        return ConditionSet(symbol, Eq(f, 0), domain)


def solve_decomposition(f, symbol, domain):
    """
    Function to solve equations via the principle of "Decomposition
    and Rewriting".

    Examples
    ========
    >>> from sympy import exp, sin, Symbol, pprint, S
    >>> from sympy.solvers.solveset import solve_decomposition as sd
    >>> x = Symbol('x')
    >>> f1 = exp(2*x) - 3*exp(x) + 2
    >>> sd(f1, x, S.Reals)
    {0, log(2)}
    >>> f2 = sin(x)**2 + 2*sin(x) + 1
    >>> pprint(sd(f2, x, S.Reals), use_unicode=False)
              3*pi
    {2*n*pi + ---- | n in Integers}
               2
    >>> f3 = sin(x + 2)
    >>> pprint(sd(f3, x, S.Reals), use_unicode=False)
    {2*n*pi - 2 | n in Integers} U {2*n*pi - 2 + pi | n in Integers}

    """
    from sympy.solvers.decompogen import decompogen
    # decompose the given function
    g_s = decompogen(f, symbol)
    # `y_s` represents the set of values for which the function `g` is to be
    # solved.
    # `solutions` represent the solutions of the equations `g = y_s` or
    # `g = 0` depending on the type of `y_s`.
    # As we are interested in solving the equation: f = 0
    y_s = FiniteSet(0)
    for g in g_s:
        frange = function_range(g, symbol, domain)
        y_s = Intersection(frange, y_s)
        result = S.EmptySet
        if isinstance(y_s, FiniteSet):
            for y in y_s:
                solutions = solveset(Eq(g, y), symbol, domain)
                if not isinstance(solutions, ConditionSet):
                    result += solutions

        else:
            if isinstance(y_s, ImageSet):
                iter_iset = (y_s,)

            elif isinstance(y_s, Union):
                iter_iset = y_s.args

            elif y_s is S.EmptySet:
                # y_s is not in the range of g in g_s, so no solution exists
                #in the given domain
                return S.EmptySet

            for iset in iter_iset:
                new_solutions = solveset(Eq(iset.lamda.expr, g), symbol, domain)
                dummy_var = tuple(iset.lamda.expr.free_symbols)[0]
                (base_set,) = iset.base_sets
                if isinstance(new_solutions, FiniteSet):
                    new_exprs = new_solutions

                elif isinstance(new_solutions, Intersection):
                    if isinstance(new_solutions.args[1], FiniteSet):
                        new_exprs = new_solutions.args[1]

                for new_expr in new_exprs:
                    result += ImageSet(Lambda(dummy_var, new_expr), base_set)

        if result is S.EmptySet:
            return ConditionSet(symbol, Eq(f, 0), domain)

        y_s = result

    return y_s


def _solveset(f, symbol, domain, _check=False):
    """Helper for solveset to return a result from an expression
    that has already been sympify'ed and is known to contain the
    given symbol."""
    # _check controls whether the answer is checked or not
    from sympy.simplify.simplify import signsimp

    if isinstance(f, BooleanTrue):
        return domain

    orig_f = f
    if f.is_Mul:
        coeff, f = f.as_independent(symbol, as_Add=False)
        if coeff in {S.ComplexInfinity, S.NegativeInfinity, S.Infinity}:
            f = together(orig_f)
    elif f.is_Add:
        a, h = f.as_independent(symbol)
        m, h = h.as_independent(symbol, as_Add=False)
        if m not in {S.ComplexInfinity, S.Zero, S.Infinity,
                              S.NegativeInfinity}:
            f = a/m + h  # XXX condition `m != 0` should be added to soln

    # assign the solvers to use
    solver = lambda f, x, domain=domain: _solveset(f, x, domain)
    inverter = lambda f, rhs, symbol: _invert(f, rhs, symbol, domain)

    result = S.EmptySet

    if f.expand().is_zero:
        return domain
    elif not f.has(symbol):
        return S.EmptySet
    elif f.is_Mul and all(_is_finite_with_finite_vars(m, domain)
            for m in f.args):
        # if f(x) and g(x) are both finite we can say that the solution of
        # f(x)*g(x) == 0 is same as Union(f(x) == 0, g(x) == 0) is not true in
        # general. g(x) can grow to infinitely large for the values where
        # f(x) == 0. To be sure that we are not silently allowing any
        # wrong solutions we are using this technique only if both f and g are
        # finite for a finite input.
        result = Union(*[solver(m, symbol) for m in f.args])
    elif (_is_function_class_equation(TrigonometricFunction, f, symbol) or \
            _is_function_class_equation(HyperbolicFunction, f, symbol)):
        result = _solve_trig(f, symbol, domain)
    elif isinstance(f, arg):
        a = f.args[0]
        result = Intersection(_solveset(re(a) > 0, symbol, domain),
                              _solveset(im(a), symbol, domain))
    elif f.is_Piecewise:
        expr_set_pairs = f.as_expr_set_pairs(domain)
        for (expr, in_set) in expr_set_pairs:
            if in_set.is_Relational:
                in_set = in_set.as_set()
            solns = solver(expr, symbol, in_set)
            result += solns
    elif isinstance(f, Eq):
        result = solver(Add(f.lhs, -f.rhs, evaluate=False), symbol, domain)

    elif f.is_Relational:
        from .inequalities import solve_univariate_inequality
        try:
            result = solve_univariate_inequality(
            f, symbol, domain=domain, relational=False)
        except NotImplementedError:
            result = ConditionSet(symbol, f, domain)
        return result
    elif _is_modular(f, symbol):
        result = _solve_modular(f, symbol, domain)
    else:
        lhs, rhs_s = inverter(f, 0, symbol)
        if lhs == symbol:
            # do some very minimal simplification since
            # repeated inversion may have left the result
            # in a state that other solvers (e.g. poly)
            # would have simplified; this is done here
            # rather than in the inverter since here it
            # is only done once whereas there it would
            # be repeated for each step of the inversion
            if isinstance(rhs_s, FiniteSet):
                rhs_s = FiniteSet(*[Mul(*
                    signsimp(i).as_content_primitive())
                    for i in rhs_s])
            result = rhs_s

        elif isinstance(rhs_s, FiniteSet):
            for equation in [lhs - rhs for rhs in rhs_s]:
                if equation == f:
                    u = unrad(f, symbol)
                    if u:
                        result += _solve_radical(equation, u,
                                                 symbol,
                                                 solver)
                    elif equation.has(Abs):
                        result += _solve_abs(f, symbol, domain)
                    else:
                        result_rational = _solve_as_rational(equation, symbol, domain)
                        if not isinstance(result_rational, ConditionSet):
                            result += result_rational
                        else:
                            # may be a transcendental type equation
                            t_result = _transolve(equation, symbol, domain)
                            if isinstance(t_result, ConditionSet):
                                # might need factoring; this is expensive so we
                                # have delayed until now. To avoid recursion
                                # errors look for a non-trivial factoring into
                                # a product of symbol dependent terms; I think
                                # that something that factors as a Pow would
                                # have already been recognized by now.
                                factored = equation.factor()
                                if factored.is_Mul and equation != factored:
                                    _, dep = factored.as_independent(symbol)
                                    if not dep.is_Add:
                                        # non-trivial factoring of equation
                                        # but use form with constants
                                        # in case they need special handling
                                        t_results = []
                                        for fac in Mul.make_args(factored):
                                            if fac.has(symbol):
                                                t_results.append(solver(fac, symbol))
                                        t_result = Union(*t_results)
                            result += t_result
                else:
                    result += solver(equation, symbol)

        elif rhs_s is not S.EmptySet:
            result = ConditionSet(symbol, Eq(f, 0), domain)

    if isinstance(result, ConditionSet):
        if isinstance(f, Expr):
            num, den = f.as_numer_denom()
            if den.has(symbol):
                _result = _solveset(num, symbol, domain)
                if not isinstance(_result, ConditionSet):
                    singularities = _solveset(den, symbol, domain)
                    result = _result - singularities

    if _check:
        if isinstance(result, ConditionSet):
            # it wasn't solved or has enumerated all conditions
            # -- leave it alone
            return result

        # whittle away all but the symbol-containing core
        # to use this for testing
        if isinstance(orig_f, Expr):
            fx = orig_f.as_independent(symbol, as_Add=True)[1]
            fx = fx.as_independent(symbol, as_Add=False)[1]
        else:
            fx = orig_f

        if isinstance(result, FiniteSet):
            # check the result for invalid solutions
            result = FiniteSet(*[s for s in result
                      if isinstance(s, RootOf)
                      or domain_check(fx, symbol, s)])

    return result


def _is_modular(f, symbol):
    """
    Helper function to check below mentioned types of modular equations.
    ``A - Mod(B, C) = 0``

    A -> This can or cannot be a function of symbol.
    B -> This is surely a function of symbol.
    C -> It is an integer.

    Parameters
    ==========

    f : Expr
        The equation to be checked.

    symbol : Symbol
        The concerned variable for which the equation is to be checked.

    Examples
    ========

    >>> from sympy import symbols, exp, Mod
    >>> from sympy.solvers.solveset import _is_modular as check
    >>> x, y = symbols('x y')
    >>> check(Mod(x, 3) - 1, x)
    True
    >>> check(Mod(x, 3) - 1, y)
    False
    >>> check(Mod(x, 3)**2 - 5, x)
    False
    >>> check(Mod(x, 3)**2 - y, x)
    False
    >>> check(exp(Mod(x, 3)) - 1, x)
    False
    >>> check(Mod(3, y) - 1, y)
    False
    """

    if not f.has(Mod):
        return False

    # extract modterms from f.
    modterms = list(f.atoms(Mod))

    return (len(modterms) == 1 and  # only one Mod should be present
            modterms[0].args[0].has(symbol) and  # B-> function of symbol
            modterms[0].args[1].is_integer and  # C-> to be an integer.
            any(isinstance(term, Mod)
            for term in list(_term_factors(f)))  # free from other funcs
            )


def _invert_modular(modterm, rhs, n, symbol):
    """
    Helper function to invert modular equation.
    ``Mod(a, m) - rhs = 0``

    Generally it is inverted as (a, ImageSet(Lambda(n, m*n + rhs), S.Integers)).
    More simplified form will be returned if possible.

    If it is not invertible then (modterm, rhs) is returned.

    The following cases arise while inverting equation ``Mod(a, m) - rhs = 0``:

    1. If a is symbol then  m*n + rhs is the required solution.

    2. If a is an instance of ``Add`` then we try to find two symbol independent
       parts of a and the symbol independent part gets transferred to the other
       side and again the ``_invert_modular`` is called on the symbol
       dependent part.

    3. If a is an instance of ``Mul`` then same as we done in ``Add`` we separate
       out the symbol dependent and symbol independent parts and transfer the
       symbol independent part to the rhs with the help of invert and again the
       ``_invert_modular`` is called on the symbol dependent part.

    4. If a is an instance of ``Pow`` then two cases arise as following:

        - If a is of type (symbol_indep)**(symbol_dep) then the remainder is
          evaluated with the help of discrete_log function and then the least
          period is being found out with the help of totient function.
          period*n + remainder is the required solution in this case.
          For reference: (https://en.wikipedia.org/wiki/Euler's_theorem)

        - If a is of type (symbol_dep)**(symbol_indep) then we try to find all
          primitive solutions list with the help of nthroot_mod function.
          m*n + rem is the general solution where rem belongs to solutions list
          from nthroot_mod function.

    Parameters
    ==========

    modterm, rhs : Expr
        The modular equation to be inverted, ``modterm - rhs = 0``

    symbol : Symbol
        The variable in the equation to be inverted.

    n : Dummy
        Dummy variable for output g_n.

    Returns
    =======

    A tuple (f_x, g_n) is being returned where f_x is modular independent function
    of symbol and g_n being set of values f_x can have.

    Examples
    ========

    >>> from sympy import symbols, exp, Mod, Dummy, S
    >>> from sympy.solvers.solveset import _invert_modular as invert_modular
    >>> x, y = symbols('x y')
    >>> n = Dummy('n')
    >>> invert_modular(Mod(exp(x), 7), S(5), n, x)
    (Mod(exp(x), 7), 5)
    >>> invert_modular(Mod(x, 7), S(5), n, x)
    (x, ImageSet(Lambda(_n, 7*_n + 5), Integers))
    >>> invert_modular(Mod(3*x + 8, 7), S(5), n, x)
    (x, ImageSet(Lambda(_n, 7*_n + 6), Integers))
    >>> invert_modular(Mod(x**4, 7), S(5), n, x)
    (x, EmptySet)
    >>> invert_modular(Mod(2**(x**2 + x + 1), 7), S(2), n, x)
    (x**2 + x + 1, ImageSet(Lambda(_n, 3*_n + 1), Naturals0))

    """
    a, m = modterm.args

    if rhs.is_integer is False:
        return symbol, S.EmptySet

    if rhs.is_real is False or any(term.is_real is False
            for term in list(_term_factors(a))):
        # Check for complex arguments
        return modterm, rhs

    if abs(rhs) >= abs(m):
        # if rhs has value greater than value of m.
        return symbol, S.EmptySet

    if a == symbol:
        return symbol, ImageSet(Lambda(n, m*n + rhs), S.Integers)

    if a.is_Add:
        # g + h = a
        g, h = a.as_independent(symbol)
        if g is not S.Zero:
            x_indep_term = rhs - Mod(g, m)
            return _invert_modular(Mod(h, m), Mod(x_indep_term, m), n, symbol)

    if a.is_Mul:
        # g*h = a
        g, h = a.as_independent(symbol)
        if g is not S.One:
            x_indep_term = rhs*invert(g, m)
            return _invert_modular(Mod(h, m), Mod(x_indep_term, m), n, symbol)

    if a.is_Pow:
        # base**expo = a
        base, expo = a.args
        if expo.has(symbol) and not base.has(symbol):
            # remainder -> solution independent of n of equation.
            # m, rhs are made coprime by dividing number_gcd(m, rhs)
            if not m.is_Integer and rhs.is_Integer and a.base.is_Integer:
                return modterm, rhs

            mdiv = m.p // number_gcd(m.p, rhs.p)
            try:
                remainder = discrete_log(mdiv, rhs.p, a.base.p)
            except ValueError:  # log does not exist
                return modterm, rhs
            # period -> coefficient of n in the solution and also referred as
            # the least period of expo in which it is repeats itself.
            # (a**(totient(m)) - 1) divides m. Here is link of theorem:
            # (https://en.wikipedia.org/wiki/Euler's_theorem)
            period = totient(m)
            for p in divisors(period):
                # there might a lesser period exist than totient(m).
                if pow(a.base, p, m / number_gcd(m.p, a.base.p)) == 1:
                    period = p
                    break
            # recursion is not applied here since _invert_modular is currently
            # not smart enough to handle infinite rhs as here expo has infinite
            # rhs = ImageSet(Lambda(n, period*n + remainder), S.Naturals0).
            return expo, ImageSet(Lambda(n, period*n + remainder), S.Naturals0)
        elif base.has(symbol) and not expo.has(symbol):
            try:
                remainder_list = nthroot_mod(rhs, expo, m, all_roots=True)
                if remainder_list == []:
                    return symbol, S.EmptySet
            except (ValueError, NotImplementedError):
                return modterm, rhs
            g_n = S.EmptySet
            for rem in remainder_list:
                g_n += ImageSet(Lambda(n, m*n + rem), S.Integers)
            return base, g_n

    return modterm, rhs


def _solve_modular(f, symbol, domain):
    r"""
    Helper function for solving modular equations of type ``A - Mod(B, C) = 0``,
    where A can or cannot be a function of symbol, B is surely a function of
    symbol and C is an integer.

    Currently ``_solve_modular`` is only able to solve cases
    where A is not a function of symbol.

    Parameters
    ==========

    f : Expr
        The modular equation to be solved, ``f = 0``

    symbol : Symbol
        The variable in the equation to be solved.

    domain : Set
        A set over which the equation is solved. It has to be a subset of
        Integers.

    Returns
    =======

    A set of integer solutions satisfying the given modular equation.
    A ``ConditionSet`` if the equation is unsolvable.

    Examples
    ========

    >>> from sympy.solvers.solveset import _solve_modular as solve_modulo
    >>> from sympy import S, Symbol, sin, Intersection, Interval, Mod
    >>> x = Symbol('x')
    >>> solve_modulo(Mod(5*x - 8, 7) - 3, x, S.Integers)
    ImageSet(Lambda(_n, 7*_n + 5), Integers)
    >>> solve_modulo(Mod(5*x - 8, 7) - 3, x, S.Reals)  # domain should be subset of integers.
    ConditionSet(x, Eq(Mod(5*x + 6, 7) - 3, 0), Reals)
    >>> solve_modulo(-7 + Mod(x, 5), x, S.Integers)
    EmptySet
    >>> solve_modulo(Mod(12**x, 21) - 18, x, S.Integers)
    ImageSet(Lambda(_n, 6*_n + 2), Naturals0)
    >>> solve_modulo(Mod(sin(x), 7) - 3, x, S.Integers) # not solvable
    ConditionSet(x, Eq(Mod(sin(x), 7) - 3, 0), Integers)
    >>> solve_modulo(3 - Mod(x, 5), x, Intersection(S.Integers, Interval(0, 100)))
    Intersection(ImageSet(Lambda(_n, 5*_n + 3), Integers), Range(0, 101, 1))
    """
    # extract modterm and g_y from f
    unsolved_result = ConditionSet(symbol, Eq(f, 0), domain)
    modterm = list(f.atoms(Mod))[0]
    rhs = -S.One*(f.subs(modterm, S.Zero))
    if f.as_coefficients_dict()[modterm].is_negative:
        # checks if coefficient of modterm is negative in main equation.
        rhs *= -S.One

    if not domain.is_subset(S.Integers):
        return unsolved_result

    if rhs.has(symbol):
        # TODO Case: A-> function of symbol, can be extended here
        # in future.
        return unsolved_result

    n = Dummy('n', integer=True)
    f_x, g_n = _invert_modular(modterm, rhs, n, symbol)

    if f_x == modterm and g_n == rhs:
        return unsolved_result

    if f_x == symbol:
        if domain is not S.Integers:
            return domain.intersect(g_n)
        return g_n

    if isinstance(g_n, ImageSet):
        lamda_expr = g_n.lamda.expr
        lamda_vars = g_n.lamda.variables
        base_sets = g_n.base_sets
        sol_set = _solveset(f_x - lamda_expr, symbol, S.Integers)
        if isinstance(sol_set, FiniteSet):
            tmp_sol = S.EmptySet
            for sol in sol_set:
                tmp_sol += ImageSet(Lambda(lamda_vars, sol), *base_sets)
            sol_set = tmp_sol
        else:
            sol_set =  ImageSet(Lambda(lamda_vars, sol_set), *base_sets)
        return domain.intersect(sol_set)

    return unsolved_result


def _term_factors(f):
    """
    Iterator to get the factors of all terms present
    in the given equation.

    Parameters
    ==========
    f : Expr
        Equation that needs to be addressed

    Returns
    =======
    Factors of all terms present in the equation.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.solvers.solveset import _term_factors
    >>> x = symbols('x')
    >>> list(_term_factors(-2 - x**2 + x*(x + 1)))
    [-2, -1, x**2, x, x + 1]
    """
    for add_arg in Add.make_args(f):
        yield from Mul.make_args(add_arg)


def _solve_exponential(lhs, rhs, symbol, domain):
    r"""
    Helper function for solving (supported) exponential equations.

    Exponential equations are the sum of (currently) at most
    two terms with one or both of them having a power with a
    symbol-dependent exponent.

    For example

    .. math:: 5^{2x + 3} - 5^{3x - 1}

    .. math:: 4^{5 - 9x} - e^{2 - x}

    Parameters
    ==========

    lhs, rhs : Expr
        The exponential equation to be solved, `lhs = rhs`

    symbol : Symbol
        The variable in which the equation is solved

    domain : Set
        A set over which the equation is solved.

    Returns
    =======

    A set of solutions satisfying the given equation.
    A ``ConditionSet`` if the equation is unsolvable or
    if the assumptions are not properly defined, in that case
    a different style of ``ConditionSet`` is returned having the
    solution(s) of the equation with the desired assumptions.

    Examples
    ========

    >>> from sympy.solvers.solveset import _solve_exponential as solve_expo
    >>> from sympy import symbols, S
    >>> x = symbols('x', real=True)
    >>> a, b = symbols('a b')
    >>> solve_expo(2**x + 3**x - 5**x, 0, x, S.Reals)  # not solvable
    ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), Reals)
    >>> solve_expo(a**x - b**x, 0, x, S.Reals)  # solvable but incorrect assumptions
    ConditionSet(x, (a > 0) & (b > 0), {0})
    >>> solve_expo(3**(2*x) - 2**(x + 3), 0, x, S.Reals)
    {-3*log(2)/(-2*log(3) + log(2))}
    >>> solve_expo(2**x - 4**x, 0, x, S.Reals)
    {0}

    * Proof of correctness of the method

    The logarithm function is the inverse of the exponential function.
    The defining relation between exponentiation and logarithm is:

    .. math:: {\log_b x} = y \enspace if \enspace b^y = x

    Therefore if we are given an equation with exponent terms, we can
    convert every term to its corresponding logarithmic form. This is
    achieved by taking logarithms and expanding the equation using
    logarithmic identities so that it can easily be handled by ``solveset``.

    For example:

    .. math:: 3^{2x} = 2^{x + 3}

    Taking log both sides will reduce the equation to

    .. math:: (2x)\log(3) = (x + 3)\log(2)

    This form can be easily handed by ``solveset``.
    """
    unsolved_result = ConditionSet(symbol, Eq(lhs - rhs, 0), domain)
    newlhs = powdenest(lhs)
    if lhs != newlhs:
        # it may also be advantageous to factor the new expr
        neweq = factor(newlhs - rhs)
        if neweq != (lhs - rhs):
            return _solveset(neweq, symbol, domain)  # try again with _solveset

    if not (isinstance(lhs, Add) and len(lhs.args) == 2):
        # solving for the sum of more than two powers is possible
        # but not yet implemented
        return unsolved_result

    if rhs != 0:
        return unsolved_result

    a, b = list(ordered(lhs.args))
    a_term = a.as_independent(symbol)[1]
    b_term = b.as_independent(symbol)[1]

    a_base, a_exp = a_term.as_base_exp()
    b_base, b_exp = b_term.as_base_exp()

    if domain.is_subset(S.Reals):
        conditions = And(
            a_base > 0,
            b_base > 0,
            Eq(im(a_exp), 0),
            Eq(im(b_exp), 0))
    else:
        conditions = And(
            Ne(a_base, 0),
            Ne(b_base, 0))

    L, R = (expand_log(log(i), force=True) for i in (a, -b))
    solutions = _solveset(L - R, symbol, domain)

    return ConditionSet(symbol, conditions, solutions)


def _is_exponential(f, symbol):
    r"""
    Return ``True`` if one or more terms contain ``symbol`` only in
    exponents, else ``False``.

    Parameters
    ==========

    f : Expr
        The equation to be checked

    symbol : Symbol
        The variable in which the equation is checked

    Examples
    ========

    >>> from sympy import symbols, cos, exp
    >>> from sympy.solvers.solveset import _is_exponential as check
    >>> x, y = symbols('x y')
    >>> check(y, y)
    False
    >>> check(x**y - 1, y)
    True
    >>> check(x**y*2**y - 1, y)
    True
    >>> check(exp(x + 3) + 3**x, x)
    True
    >>> check(cos(2**x), x)
    False

    * Philosophy behind the helper

    The function extracts each term of the equation and checks if it is
    of exponential form w.r.t ``symbol``.
    """
    rv = False
    for expr_arg in _term_factors(f):
        if symbol not in expr_arg.free_symbols:
            continue
        if (isinstance(expr_arg, Pow) and
           symbol not in expr_arg.base.free_symbols or
           isinstance(expr_arg, exp)):
            rv = True  # symbol in exponent
        else:
            return False  # dependent on symbol in non-exponential way
    return rv


def _solve_logarithm(lhs, rhs, symbol, domain):
    r"""
    Helper to solve logarithmic equations which are reducible
    to a single instance of `\log`.

    Logarithmic equations are (currently) the equations that contains
    `\log` terms which can be reduced to a single `\log` term or
    a constant using various logarithmic identities.

    For example:

    .. math:: \log(x) + \log(x - 4)

    can be reduced to:

    .. math:: \log(x(x - 4))

    Parameters
    ==========

    lhs, rhs : Expr
        The logarithmic equation to be solved, `lhs = rhs`

    symbol : Symbol
        The variable in which the equation is solved

    domain : Set
        A set over which the equation is solved.

    Returns
    =======

    A set of solutions satisfying the given equation.
    A ``ConditionSet`` if the equation is unsolvable.

    Examples
    ========

    >>> from sympy import symbols, log, S
    >>> from sympy.solvers.solveset import _solve_logarithm as solve_log
    >>> x = symbols('x')
    >>> f = log(x - 3) + log(x + 3)
    >>> solve_log(f, 0, x, S.Reals)
    {-sqrt(10), sqrt(10)}

    * Proof of correctness

    A logarithm is another way to write exponent and is defined by

    .. math:: {\log_b x} = y \enspace if \enspace b^y = x

    When one side of the equation contains a single logarithm, the
    equation can be solved by rewriting the equation as an equivalent
    exponential equation as defined above. But if one side contains
    more than one logarithm, we need to use the properties of logarithm
    to condense it into a single logarithm.

    Take for example

    .. math:: \log(2x) - 15 = 0

    contains single logarithm, therefore we can directly rewrite it to
    exponential form as

    .. math:: x = \frac{e^{15}}{2}

    But if the equation has more than one logarithm as

    .. math:: \log(x - 3) + \log(x + 3) = 0

    we use logarithmic identities to convert it into a reduced form

    Using,

    .. math:: \log(a) + \log(b) = \log(ab)

    the equation becomes,

    .. math:: \log((x - 3)(x + 3))

    This equation contains one logarithm and can be solved by rewriting
    to exponents.
    """
    new_lhs = logcombine(lhs, force=True)
    new_f = new_lhs - rhs

    return _solveset(new_f, symbol, domain)


def _is_logarithmic(f, symbol):
    r"""
    Return ``True`` if the equation is in the form
    `a\log(f(x)) + b\log(g(x)) + ... + c` else ``False``.

    Parameters
    ==========

    f : Expr
        The equation to be checked

    symbol : Symbol
        The variable in which the equation is checked

    Returns
    =======

    ``True`` if the equation is logarithmic otherwise ``False``.

    Examples
    ========

    >>> from sympy import symbols, tan, log
    >>> from sympy.solvers.solveset import _is_logarithmic as check
    >>> x, y = symbols('x y')
    >>> check(log(x + 2) - log(x + 3), x)
    True
    >>> check(tan(log(2*x)), x)
    False
    >>> check(x*log(x), x)
    False
    >>> check(x + log(x), x)
    False
    >>> check(y + log(x), x)
    True

    * Philosophy behind the helper

    The function extracts each term and checks whether it is
    logarithmic w.r.t ``symbol``.
    """
    rv = False
    for term in Add.make_args(f):
        saw_log = False
        for term_arg in Mul.make_args(term):
            if symbol not in term_arg.free_symbols:
                continue
            if isinstance(term_arg, log):
                if saw_log:
                    return False  # more than one log in term
                saw_log = True
            else:
                return False  # dependent on symbol in non-log way
        if saw_log:
            rv = True
    return rv


def _is_lambert(f, symbol):
    r"""
    If this returns ``False`` then the Lambert solver (``_solve_lambert``) will not be called.

    Explanation
    ===========

    Quick check for cases that the Lambert solver might be able to handle.

    1. Equations containing more than two operands and `symbol`s involving any of
       `Pow`, `exp`, `HyperbolicFunction`,`TrigonometricFunction`, `log` terms.

    2. In `Pow`, `exp` the exponent should have `symbol` whereas for
       `HyperbolicFunction`,`TrigonometricFunction`, `log` should contain `symbol`.

    3. For `HyperbolicFunction`,`TrigonometricFunction` the number of trigonometric functions in
       equation should be less than number of symbols. (since `A*cos(x) + B*sin(x) - c`
       is not the Lambert type).

    Some forms of lambert equations are:
        1. X**X = C
        2. X*(B*log(X) + D)**A = C
        3. A*log(B*X + A) + d*X = C
        4. (B*X + A)*exp(d*X + g) = C
        5. g*exp(B*X + h) - B*X = C
        6. A*D**(E*X + g) - B*X = C
        7. A*cos(X) + B*sin(X) - D*X = C
        8. A*cosh(X) + B*sinh(X) - D*X = C

    Where X is any variable,
          A, B, C, D, E are any constants,
          g, h are linear functions or log terms.

    Parameters
    ==========

    f : Expr
        The equation to be checked

    symbol : Symbol
        The variable in which the equation is checked

    Returns
    =======

    If this returns ``False`` then the Lambert solver (``_solve_lambert``) will not be called.

    Examples
    ========

    >>> from sympy.solvers.solveset import _is_lambert
    >>> from sympy import symbols, cosh, sinh, log
    >>> x = symbols('x')

    >>> _is_lambert(3*log(x) - x*log(3), x)
    True
    >>> _is_lambert(log(log(x - 3)) + log(x-3), x)
    True
    >>> _is_lambert(cosh(x) - sinh(x), x)
    False
    >>> _is_lambert((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1), x)
    True

    See Also
    ========

    _solve_lambert

    """
    term_factors = list(_term_factors(f.expand()))

    # total number of symbols in equation
    no_of_symbols = len([arg for arg in term_factors if arg.has(symbol)])
    # total number of trigonometric terms in equation
    no_of_trig = len([arg for arg in term_factors \
        if arg.has(HyperbolicFunction, TrigonometricFunction)])

    if f.is_Add and no_of_symbols >= 2:
        # `log`, `HyperbolicFunction`, `TrigonometricFunction` should have symbols
        # and no_of_trig < no_of_symbols
        lambert_funcs = (log, HyperbolicFunction, TrigonometricFunction)
        if any(isinstance(arg, lambert_funcs)\
            for arg in term_factors if arg.has(symbol)):
                if no_of_trig < no_of_symbols:
                    return True
        # here, `Pow`, `exp` exponent should have symbols
        elif any(isinstance(arg, (Pow, exp)) \
            for arg in term_factors if (arg.as_base_exp()[1]).has(symbol)):
            return True
    return False


def _transolve(f, symbol, domain):
    r"""
    Function to solve transcendental equations. It is a helper to
    ``solveset`` and should be used internally. ``_transolve``
    currently supports the following class of equations:

        - Exponential equations
        - Logarithmic equations

    Parameters
    ==========

    f : Any transcendental equation that needs to be solved.
        This needs to be an expression, which is assumed
        to be equal to ``0``.

    symbol : The variable for which the equation is solved.
        This needs to be of class ``Symbol``.

    domain : A set over which the equation is solved.
        This needs to be of class ``Set``.

    Returns
    =======

    Set
        A set of values for ``symbol`` for which ``f`` is equal to
        zero. An ``EmptySet`` is returned if ``f`` does not have solutions
        in respective domain. A ``ConditionSet`` is returned as unsolved
        object if algorithms to evaluate complete solution are not
        yet implemented.

    How to use ``_transolve``
    =========================

    ``_transolve`` should not be used as an independent function, because
    it assumes that the equation (``f``) and the ``symbol`` comes from
    ``solveset`` and might have undergone a few modification(s).
    To use ``_transolve`` as an independent function the equation (``f``)
    and the ``symbol`` should be passed as they would have been by
    ``solveset``.

    Examples
    ========

    >>> from sympy.solvers.solveset import _transolve as transolve
    >>> from sympy.solvers.solvers import _tsolve as tsolve
    >>> from sympy import symbols, S, pprint
    >>> x = symbols('x', real=True) # assumption added
    >>> transolve(5**(x - 3) - 3**(2*x + 1), x, S.Reals)
    {-(log(3) + 3*log(5))/(-log(5) + 2*log(3))}

    How ``_transolve`` works
    ========================

    ``_transolve`` uses two types of helper functions to solve equations
    of a particular class:

    Identifying helpers: To determine whether a given equation
    belongs to a certain class of equation or not. Returns either
    ``True`` or ``False``.

    Solving helpers: Once an equation is identified, a corresponding
    helper either solves the equation or returns a form of the equation
    that ``solveset`` might better be able to handle.

    * Philosophy behind the module

    The purpose of ``_transolve`` is to take equations which are not
    already polynomial in their generator(s) and to either recast them
    as such through a valid transformation or to solve them outright.
    A pair of helper functions for each class of supported
    transcendental functions are employed for this purpose. One
    identifies the transcendental form of an equation and the other
    either solves it or recasts it into a tractable form that can be
    solved by  ``solveset``.
    For example, an equation in the form `ab^{f(x)} - cd^{g(x)} = 0`
    can be transformed to
    `\log(a) + f(x)\log(b) - \log(c) - g(x)\log(d) = 0`
    (under certain assumptions) and this can be solved with ``solveset``
    if `f(x)` and `g(x)` are in polynomial form.

    How ``_transolve`` is better than ``_tsolve``
    =============================================

    1) Better output

    ``_transolve`` provides expressions in a more simplified form.

    Consider a simple exponential equation

    >>> f = 3**(2*x) - 2**(x + 3)
    >>> pprint(transolve(f, x, S.Reals), use_unicode=False)
        -3*log(2)
    {------------------}
     -2*log(3) + log(2)
    >>> pprint(tsolve(f, x), use_unicode=False)
         /   3     \
         | --------|
         | log(2/9)|
    [-log\2         /]

    2) Extensible

    The API of ``_transolve`` is designed such that it is easily
    extensible, i.e. the code that solves a given class of
    equations is encapsulated in a helper and not mixed in with
    the code of ``_transolve`` itself.

    3) Modular

    ``_transolve`` is designed to be modular i.e, for every class of
    equation a separate helper for identification and solving is
    implemented. This makes it easy to change or modify any of the
    method implemented directly in the helpers without interfering
    with the actual structure of the API.

    4) Faster Computation

    Solving equation via ``_transolve`` is much faster as compared to
    ``_tsolve``. In ``solve``, attempts are made computing every possibility
    to get the solutions. This series of attempts makes solving a bit
    slow. In ``_transolve``, computation begins only after a particular
    type of equation is identified.

    How to add new class of equations
    =================================

    Adding a new class of equation solver is a three-step procedure:

    - Identify the type of the equations

      Determine the type of the class of equations to which they belong:
      it could be of ``Add``, ``Pow``, etc. types. Separate internal functions
      are used for each type. Write identification and solving helpers
      and use them from within the routine for the given type of equation
      (after adding it, if necessary). Something like:

      .. code-block:: python

        def add_type(lhs, rhs, x):
            ....
            if _is_exponential(lhs, x):
                new_eq = _solve_exponential(lhs, rhs, x)
        ....
        rhs, lhs = eq.as_independent(x)
        if lhs.is_Add:
            result = add_type(lhs, rhs, x)

    - Define the identification helper.

    - Define the solving helper.

    Apart from this, a few other things needs to be taken care while
    adding an equation solver:

    - Naming conventions:
      Name of the identification helper should be as
      ``_is_class`` where class will be the name or abbreviation
      of the class of equation. The solving helper will be named as
      ``_solve_class``.
      For example: for exponential equations it becomes
      ``_is_exponential`` and ``_solve_expo``.
    - The identifying helpers should take two input parameters,
      the equation to be checked and the variable for which a solution
      is being sought, while solving helpers would require an additional
      domain parameter.
    - Be sure to consider corner cases.
    - Add tests for each helper.
    - Add a docstring to your helper that describes the method
      implemented.
      The documentation of the helpers should identify:

      - the purpose of the helper,
      - the method used to identify and solve the equation,
      - a proof of correctness
      - the return values of the helpers
    """

    def add_type(lhs, rhs, symbol, domain):
        """
        Helper for ``_transolve`` to handle equations of
        ``Add`` type, i.e. equations taking the form as
        ``a*f(x) + b*g(x) + .... = c``.
        For example: 4**x + 8**x = 0
        """
        result = ConditionSet(symbol, Eq(lhs - rhs, 0), domain)

        # check if it is exponential type equation
        if _is_exponential(lhs, symbol):
            result = _solve_exponential(lhs, rhs, symbol, domain)
        # check if it is logarithmic type equation
        elif _is_logarithmic(lhs, symbol):
            result = _solve_logarithm(lhs, rhs, symbol, domain)

        return result

    result = ConditionSet(symbol, Eq(f, 0), domain)

    # invert_complex handles the call to the desired inverter based
    # on the domain specified.
    lhs, rhs_s = invert_complex(f, 0, symbol, domain)

    if isinstance(rhs_s, FiniteSet):
        assert (len(rhs_s.args)) == 1
        rhs = rhs_s.args[0]

        if lhs.is_Add:
            result = add_type(lhs, rhs, symbol, domain)
    else:
        result = rhs_s

    return result


def solveset(f, symbol=None, domain=S.Complexes):
    r"""Solves a given inequality or equation with set as output

    Parameters
    ==========

    f : Expr or a relational.
        The target equation or inequality
    symbol : Symbol
        The variable for which the equation is solved
    domain : Set
        The domain over which the equation is solved

    Returns
    =======

    Set
        A set of values for `symbol` for which `f` is True or is equal to
        zero. An :class:`~.EmptySet` is returned if `f` is False or nonzero.
        A :class:`~.ConditionSet` is returned as unsolved object if algorithms
        to evaluate complete solution are not yet implemented.

    ``solveset`` claims to be complete in the solution set that it returns.

    Raises
    ======

    NotImplementedError
        The algorithms to solve inequalities in complex domain  are
        not yet implemented.
    ValueError
        The input is not valid.
    RuntimeError
        It is a bug, please report to the github issue tracker.


    Notes
    =====

    Python interprets 0 and 1 as False and True, respectively, but
    in this function they refer to solutions of an expression. So 0 and 1
    return the domain and EmptySet, respectively, while True and False
    return the opposite (as they are assumed to be solutions of relational
    expressions).


    See Also
    ========

    solveset_real: solver for real domain
    solveset_complex: solver for complex domain

    Examples
    ========

    >>> from sympy import exp, sin, Symbol, pprint, S, Eq
    >>> from sympy.solvers.solveset import solveset, solveset_real

    * The default domain is complex. Not specifying a domain will lead
      to the solving of the equation in the complex domain (and this
      is not affected by the assumptions on the symbol):

    >>> x = Symbol('x')
    >>> pprint(solveset(exp(x) - 1, x), use_unicode=False)
    {2*n*I*pi | n in Integers}

    >>> x = Symbol('x', real=True)
    >>> pprint(solveset(exp(x) - 1, x), use_unicode=False)
    {2*n*I*pi | n in Integers}

    * If you want to use ``solveset`` to solve the equation in the
      real domain, provide a real domain. (Using ``solveset_real``
      does this automatically.)

    >>> R = S.Reals
    >>> x = Symbol('x')
    >>> solveset(exp(x) - 1, x, R)
    {0}
    >>> solveset_real(exp(x) - 1, x)
    {0}

    The solution is unaffected by assumptions on the symbol:

    >>> p = Symbol('p', positive=True)
    >>> pprint(solveset(p**2 - 4))
    {-2, 2}

    When a :class:`~.ConditionSet` is returned, symbols with assumptions that
    would alter the set are replaced with more generic symbols:

    >>> i = Symbol('i', imaginary=True)
    >>> solveset(Eq(i**2 + i*sin(i), 1), i, domain=S.Reals)
    ConditionSet(_R, Eq(_R**2 + _R*sin(_R) - 1, 0), Reals)

    * Inequalities can be solved over the real domain only. Use of a complex
      domain leads to a NotImplementedError.

    >>> solveset(exp(x) > 1, x, R)
    Interval.open(0, oo)

    """
    f = sympify(f)
    symbol = sympify(symbol)

    if f is S.true:
        return domain

    if f is S.false:
        return S.EmptySet

    if not isinstance(f, (Expr, Relational, Number)):
        raise ValueError("%s is not a valid SymPy expression" % f)

    if not isinstance(symbol, (Expr, Relational)) and  symbol is not None:
        raise ValueError("%s is not a valid SymPy symbol" % (symbol,))

    if not isinstance(domain, Set):
        raise ValueError("%s is not a valid domain" %(domain))

    free_symbols = f.free_symbols

    if f.has(Piecewise):
        f = piecewise_fold(f)

    if symbol is None and not free_symbols:
        b = Eq(f, 0)
        if b is S.true:
            return domain
        elif b is S.false:
            return S.EmptySet
        else:
            raise NotImplementedError(filldedent('''
                relationship between value and 0 is unknown: %s''' % b))

    if symbol is None:
        if len(free_symbols) == 1:
            symbol = free_symbols.pop()
        elif free_symbols:
            raise ValueError(filldedent('''
                The independent variable must be specified for a
                multivariate equation.'''))
    elif not isinstance(symbol, Symbol):
        f, s, swap = recast_to_symbols([f], [symbol])
        # the xreplace will be needed if a ConditionSet is returned
        return solveset(f[0], s[0], domain).xreplace(swap)

    # solveset should ignore assumptions on symbols
    newsym = None
    if domain.is_subset(S.Reals):
        if symbol._assumptions_orig != {'real': True}:
            newsym = Dummy('R', real=True)
    elif domain.is_subset(S.Complexes):
        if symbol._assumptions_orig != {'complex': True}:
            newsym = Dummy('C', complex=True)

    if newsym is not None:
        rv = solveset(f.xreplace({symbol: newsym}), newsym, domain)
        # try to use the original symbol if possible
        try:
            _rv = rv.xreplace({newsym: symbol})
        except TypeError:
            _rv = rv
        if rv.dummy_eq(_rv):
            rv = _rv
        return rv

    # Abs has its own handling method which avoids the
    # rewriting property that the first piece of abs(x)
    # is for x >= 0 and the 2nd piece for x < 0 -- solutions
    # can look better if the 2nd condition is x <= 0. Since
    # the solution is a set, duplication of results is not
    # an issue, e.g. {y, -y} when y is 0 will be {0}
    f, mask = _masked(f, Abs)
    f = f.rewrite(Piecewise) # everything that's not an Abs
    for d, e in mask:
        # everything *in* an Abs
        e = e.func(e.args[0].rewrite(Piecewise))
        f = f.xreplace({d: e})
    f = piecewise_fold(f)

    return _solveset(f, symbol, domain, _check=True)


def solveset_real(f, symbol):
    return solveset(f, symbol, S.Reals)


def solveset_complex(f, symbol):
    return solveset(f, symbol, S.Complexes)


def _solveset_multi(eqs, syms, domains):
    '''Basic implementation of a multivariate solveset.

    For internal use (not ready for public consumption)'''

    rep = {}
    for sym, dom in zip(syms, domains):
        if dom is S.Reals:
            rep[sym] = Symbol(sym.name, real=True)
    eqs = [eq.subs(rep) for eq in eqs]
    syms = [sym.subs(rep) for sym in syms]

    syms = tuple(syms)

    if len(eqs) == 0:
        return ProductSet(*domains)

    if len(syms) == 1:
        sym = syms[0]
        domain = domains[0]
        solsets = [solveset(eq, sym, domain) for eq in eqs]
        solset = Intersection(*solsets)
        return ImageSet(Lambda((sym,), (sym,)), solset).doit()

    eqs = sorted(eqs, key=lambda eq: len(eq.free_symbols & set(syms)))

    for n, eq in enumerate(eqs):
        sols = []
        all_handled = True
        for sym in syms:
            if sym not in eq.free_symbols:
                continue
            sol = solveset(eq, sym, domains[syms.index(sym)])

            if isinstance(sol, FiniteSet):
                i = syms.index(sym)
                symsp = syms[:i] + syms[i+1:]
                domainsp = domains[:i] + domains[i+1:]
                eqsp = eqs[:n] + eqs[n+1:]
                for s in sol:
                    eqsp_sub = [eq.subs(sym, s) for eq in eqsp]
                    sol_others = _solveset_multi(eqsp_sub, symsp, domainsp)
                    fun = Lambda((symsp,), symsp[:i] + (s,) + symsp[i:])
                    sols.append(ImageSet(fun, sol_others).doit())
            else:
                all_handled = False
        if all_handled:
            return Union(*sols)


def solvify(f, symbol, domain):
    """Solves an equation using solveset and returns the solution in accordance
    with the `solve` output API.

    Returns
    =======

    We classify the output based on the type of solution returned by `solveset`.

    Solution    |    Output
    ----------------------------------------
    FiniteSet   | list

    ImageSet,   | list (if `f` is periodic)
    Union       |

    Union       | list (with FiniteSet)

    EmptySet    | empty list

    Others      | None


    Raises
    ======

    NotImplementedError
        A ConditionSet is the input.

    Examples
    ========

    >>> from sympy.solvers.solveset import solvify
    >>> from sympy.abc import x
    >>> from sympy import S, tan, sin, exp
    >>> solvify(x**2 - 9, x, S.Reals)
    [-3, 3]
    >>> solvify(sin(x) - 1, x, S.Reals)
    [pi/2]
    >>> solvify(tan(x), x, S.Reals)
    [0]
    >>> solvify(exp(x) - 1, x, S.Complexes)

    >>> solvify(exp(x) - 1, x, S.Reals)
    [0]

    """
    solution_set = solveset(f, symbol, domain)
    result = None
    if solution_set is S.EmptySet:
        result = []

    elif isinstance(solution_set, ConditionSet):
        raise NotImplementedError('solveset is unable to solve this equation.')

    elif isinstance(solution_set, FiniteSet):
        result = list(solution_set)

    else:
        period = periodicity(f, symbol)
        if period is not None:
            solutions = S.EmptySet
            iter_solutions = ()
            if isinstance(solution_set, ImageSet):
                iter_solutions = (solution_set,)
            elif isinstance(solution_set, Union):
                if all(isinstance(i, ImageSet) for i in solution_set.args):
                    iter_solutions = solution_set.args

            for solution in iter_solutions:
                solutions += solution.intersect(Interval(0, period, False, True))

            if isinstance(solutions, FiniteSet):
                result = list(solutions)

        else:
            solution = solution_set.intersect(domain)
            if isinstance(solution, Union):
                # concerned about only FiniteSet with Union but not about ImageSet
                # if required could be extend
                if any(isinstance(i, FiniteSet) for i in solution.args):
                    result = [sol for soln in solution.args \
                     for sol in soln.args if isinstance(soln,FiniteSet)]
                else:
                    return None

            elif isinstance(solution, FiniteSet):
                result += solution

    return result


###############################################################################
################################ LINSOLVE #####################################
###############################################################################


def linear_coeffs(eq, *syms, dict=False):
    """Return a list whose elements are the coefficients of the
    corresponding symbols in the sum of terms in  ``eq``.
    The additive constant is returned as the last element of the
    list.

    Raises
    ======

    NonlinearError
        The equation contains a nonlinear term
    ValueError
        duplicate or unordered symbols are passed

    Parameters
    ==========

    dict - (default False) when True, return coefficients as a
        dictionary with coefficients keyed to syms that were present;
        key 1 gives the constant term

    Examples
    ========

    >>> from sympy.solvers.solveset import linear_coeffs
    >>> from sympy.abc import x, y, z
    >>> linear_coeffs(3*x + 2*y - 1, x, y)
    [3, 2, -1]

    It is not necessary to expand the expression:

        >>> linear_coeffs(x + y*(z*(x*3 + 2) + 3), x)
        [3*y*z + 1, y*(2*z + 3)]

    When nonlinear is detected, an error will be raised:

        * even if they would cancel after expansion (so the
        situation does not pass silently past the caller's
        attention)

        >>> eq = 1/x*(x - 1) + 1/x
        >>> linear_coeffs(eq.expand(), x)
        [0, 1]
        >>> linear_coeffs(eq, x)
        Traceback (most recent call last):
        ...
        NonlinearError:
        nonlinear in given generators

        * when there are cross terms

        >>> linear_coeffs(x*(y + 1), x, y)
        Traceback (most recent call last):
        ...
        NonlinearError:
        symbol-dependent cross-terms encountered

        * when there are terms that contain an expression
        dependent on the symbols that is not linear

        >>> linear_coeffs(x**2, x)
        Traceback (most recent call last):
        ...
        NonlinearError:
        nonlinear in given generators
    """
    eq = _sympify(eq)
    if len(syms) == 1 and iterable(syms[0]) and not isinstance(syms[0], Basic):
        raise ValueError('expecting unpacked symbols, *syms')
    symset = set(syms)
    if len(symset) != len(syms):
        raise ValueError('duplicate symbols given')
    try:
        d, c = _linear_eq_to_dict([eq], symset)
        d = d[0]
        c = c[0]
    except PolyNonlinearError as err:
        raise NonlinearError(str(err))
    if dict:
        if c:
            d[S.One] = c
        return d
    rv = [S.Zero]*(len(syms) + 1)
    rv[-1] = c
    for i, k in enumerate(syms):
        if k not in d:
            continue
        rv[i] = d[k]
    return rv


def linear_eq_to_matrix(equations, *symbols):
    r"""
    Converts a given System of Equations into Matrix form. Here ``equations``
    must be a linear system of equations in ``symbols``. Element ``M[i, j]``
    corresponds to the coefficient of the jth symbol in the ith equation.

    The Matrix form corresponds to the augmented matrix form. For example:

    .. math::

       4x + 2y + 3z & = 1 \\
       3x +  y +  z & = -6 \\
       2x + 4y + 9z & = 2

    This system will return :math:`A` and :math:`b` as:

    .. math::

       A = \left[\begin{array}{ccc}
       4 & 2 & 3 \\
       3 & 1 & 1 \\
       2 & 4 & 9
       \end{array}\right] \\

    .. math::

       b = \left[\begin{array}{c}
       1 \\ -6 \\ 2
       \end{array}\right]

    The only simplification performed is to convert
    ``Eq(a, b)`` :math:`\Rightarrow a - b`.

    Raises
    ======

    NonlinearError
        The equations contain a nonlinear term.
    ValueError
        The symbols are not given or are not unique.

    Examples
    ========

    >>> from sympy import linear_eq_to_matrix, symbols
    >>> c, x, y, z = symbols('c, x, y, z')

    The coefficients (numerical or symbolic) of the symbols will
    be returned as matrices:

    >>> eqns = [c*x + z - 1 - c, y + z, x - y]
    >>> A, b = linear_eq_to_matrix(eqns, [x, y, z])
    >>> A
    Matrix([
    [c,  0, 1],
    [0,  1, 1],
    [1, -1, 0]])
    >>> b
    Matrix([
    [c + 1],
    [    0],
    [    0]])

    This routine does not simplify expressions and will raise an error
    if nonlinearity is encountered:

    >>> eqns = [
    ...     (x**2 - 3*x)/(x - 3) - 3,
    ...     y**2 - 3*y - y*(y - 4) + x - 4]
    >>> linear_eq_to_matrix(eqns, [x, y])
    Traceback (most recent call last):
    ...
    NonlinearError:
    symbol-dependent term can be ignored using `strict=False`

    Simplifying these equations will discard the removable singularity in the
    first and reveal the linear structure of the second:

    >>> [e.simplify() for e in eqns]
    [x - 3, x + y - 4]

    Any such simplification needed to eliminate nonlinear terms must be done
    *before* calling this routine.

    """
    if not symbols:
        raise ValueError(filldedent('''
            Symbols must be given, for which coefficients
            are to be found.
            '''))

    # Check if 'symbols' is a set and raise an error if it is
    if isinstance(symbols[0], set):
        raise TypeError(
            "Unordered 'set' type is not supported as input for symbols.")

    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]

    if has_dups(symbols):
        raise ValueError('Symbols must be unique')

    equations = sympify(equations)
    if isinstance(equations, MatrixBase):
        equations = list(equations)
    elif isinstance(equations, (Expr, Eq)):
        equations = [equations]
    elif not is_sequence(equations):
        raise ValueError(filldedent('''
            Equation(s) must be given as a sequence, Expr,
            Eq or Matrix.
            '''))

    # construct the dictionaries
    try:
        eq, c = _linear_eq_to_dict(equations, symbols)
    except PolyNonlinearError as err:
        raise NonlinearError(str(err))
    # prepare output matrices
    n, m = shape = len(eq), len(symbols)
    ix = dict(zip(symbols, range(m)))
    A = zeros(*shape)
    for row, d in enumerate(eq):
        for k in d:
            col = ix[k]
            A[row, col] = d[k]
    b = Matrix(n, 1, [-i for i in c])
    return A, b


def linsolve(system, *symbols):
    r"""
    Solve system of $N$ linear equations with $M$ variables; both
    underdetermined and overdetermined systems are supported.
    The possible number of solutions is zero, one or infinite.
    Zero solutions throws a ValueError, whereas infinite
    solutions are represented parametrically in terms of the given
    symbols. For unique solution a :class:`~.FiniteSet` of ordered tuples
    is returned.

    All standard input formats are supported:
    For the given set of equations, the respective input types
    are given below:

    .. math:: 3x + 2y -   z = 1
    .. math:: 2x - 2y + 4z = -2
    .. math:: 2x -   y + 2z = 0

    * Augmented matrix form, ``system`` given below:

    $$ \text{system} = \left[{array}{cccc}
        3 &  2 & -1 &  1\\
        2 & -2 &  4 & -2\\
        2 & -1 &  2 &  0
        \end{array}\right] $$

    ::

        system = Matrix([[3, 2, -1, 1], [2, -2, 4, -2], [2, -1, 2, 0]])

    * List of equations form

    ::

        system  =  [3x + 2y - z - 1, 2x - 2y + 4z + 2, 2x - y + 2z]

    * Input $A$ and $b$ in matrix form (from $Ax = b$) are given as:

    $$ A = \left[\begin{array}{ccc}
        3 &  2 & -1 \\
        2 & -2 &  4 \\
        2 & -1 &  2
        \end{array}\right] \ \  b = \left[\begin{array}{c}
        1 \\ -2 \\ 0
        \end{array}\right] $$

    ::

        A = Matrix([[3, 2, -1], [2, -2, 4], [2, -1, 2]])
        b = Matrix([[1], [-2], [0]])
        system = (A, b)

    Symbols can always be passed but are actually only needed
    when 1) a system of equations is being passed and 2) the
    system is passed as an underdetermined matrix and one wants
    to control the name of the free variables in the result.
    An error is raised if no symbols are used for case 1, but if
    no symbols are provided for case 2, internally generated symbols
    will be provided. When providing symbols for case 2, there should
    be at least as many symbols are there are columns in matrix A.

    The algorithm used here is Gauss-Jordan elimination, which
    results, after elimination, in a row echelon form matrix.

    Returns
    =======

    A FiniteSet containing an ordered tuple of values for the
    unknowns for which the `system` has a solution. (Wrapping
    the tuple in FiniteSet is used to maintain a consistent
    output format throughout solveset.)

    Returns EmptySet, if the linear system is inconsistent.

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.

    Examples
    ========

    >>> from sympy import Matrix, linsolve, symbols
    >>> x, y, z = symbols("x, y, z")
    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    >>> b = Matrix([3, 6, 9])
    >>> A
    Matrix([
    [1, 2,  3],
    [4, 5,  6],
    [7, 8, 10]])
    >>> b
    Matrix([
    [3],
    [6],
    [9]])
    >>> linsolve((A, b), [x, y, z])
    {(-1, 2, 0)}

    * Parametric Solution: In case the system is underdetermined, the
      function will return a parametric solution in terms of the given
      symbols. Those that are free will be returned unchanged. e.g. in
      the system below, `z` is returned as the solution for variable z;
      it can take on any value.

    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> b = Matrix([3, 6, 9])
    >>> linsolve((A, b), x, y, z)
    {(z - 1, 2 - 2*z, z)}

    If no symbols are given, internally generated symbols will be used.
    The ``tau0`` in the third position indicates (as before) that the third
    variable -- whatever it is named -- can take on any value:

    >>> linsolve((A, b))
    {(tau0 - 1, 2 - 2*tau0, tau0)}

    * List of equations as input

    >>> Eqns = [3*x + 2*y - z - 1, 2*x - 2*y + 4*z + 2, - x + y/2 - z]
    >>> linsolve(Eqns, x, y, z)
    {(1, -2, -2)}

    * Augmented matrix as input

    >>> aug = Matrix([[2, 1, 3, 1], [2, 6, 8, 3], [6, 8, 18, 5]])
    >>> aug
    Matrix([
    [2, 1,  3, 1],
    [2, 6,  8, 3],
    [6, 8, 18, 5]])
    >>> linsolve(aug, x, y, z)
    {(3/10, 2/5, 0)}

    * Solve for symbolic coefficients

    >>> a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    >>> eqns = [a*x + b*y - c, d*x + e*y - f]
    >>> linsolve(eqns, x, y)
    {((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d))}

    * A degenerate system returns solution as set of given
      symbols.

    >>> system = Matrix(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
    >>> linsolve(system, x, y)
    {(x, y)}

    * For an empty system linsolve returns empty set

    >>> linsolve([], x)
    EmptySet

    * An error is raised if any nonlinearity is detected, even
      if it could be removed with expansion

    >>> linsolve([x*(1/x - 1)], x)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear term: 1/x

    >>> linsolve([x*(y + 1)], x, y)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear cross-term: x*(y + 1)

    >>> linsolve([x**2 - 1], x)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear term: x**2
    """
    if not system:
        return S.EmptySet

    # If second argument is an iterable
    if symbols and hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]
    sym_gen = isinstance(symbols, GeneratorType)
    dup_msg = 'duplicate symbols given'


    b = None  # if we don't get b the input was bad
    # unpack system

    if hasattr(system, '__iter__'):

        # 1). (A, b)
        if len(system) == 2 and isinstance(system[0], MatrixBase):
            A, b = system

        # 2). (eq1, eq2, ...)
        if not isinstance(system[0], MatrixBase):
            if sym_gen or not symbols:
                raise ValueError(filldedent('''
                    When passing a system of equations, the explicit
                    symbols for which a solution is being sought must
                    be given as a sequence, too.
                '''))
            if len(set(symbols)) != len(symbols):
                raise ValueError(dup_msg)

            #
            # Pass to the sparse solver implemented in polys. It is important
            # that we do not attempt to convert the equations to a matrix
            # because that would be very inefficient for large sparse systems
            # of equations.
            #
            eqs = system
            eqs = [sympify(eq) for eq in eqs]
            try:
                sol = _linsolve(eqs, symbols)
            except PolyNonlinearError as exc:
                # e.g. cos(x) contains an element of the set of generators
                raise NonlinearError(str(exc))

            if sol is None:
                return S.EmptySet

            sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))
            return sol

    elif isinstance(system, MatrixBase) and not (
            symbols and not isinstance(symbols, GeneratorType) and
            isinstance(symbols[0], MatrixBase)):
        # 3). A augmented with b
        A, b = system[:, :-1], system[:, -1:]

    if b is None:
        raise ValueError("Invalid arguments")
    if sym_gen:
        symbols = [next(symbols) for i in range(A.cols)]
        symset = set(symbols)
        if any(symset & (A.free_symbols | b.free_symbols)):
            raise ValueError(filldedent('''
                At least one of the symbols provided
                already appears in the system to be solved.
                One way to avoid this is to use Dummy symbols in
                the generator, e.g. numbered_symbols('%s', cls=Dummy)
            ''' % symbols[0].name.rstrip('1234567890')))
        elif len(symset) != len(symbols):
            raise ValueError(dup_msg)

    if not symbols:
        symbols = [Dummy() for _ in range(A.cols)]
        name = _uniquely_named_symbol('tau', (A, b),
            compare=lambda i: str(i).rstrip('1234567890')).name
        gen  = numbered_symbols(name)
    else:
        gen = None

    # This is just a wrapper for solve_lin_sys
    eqs = []
    rows = A.tolist()
    for rowi, bi in zip(rows, b):
        terms = [elem * sym for elem, sym in zip(rowi, symbols) if elem]
        terms.append(-bi)
        eqs.append(Add(*terms))

    eqs, ring = sympy_eqs_to_ring(eqs, symbols)
    sol = solve_lin_sys(eqs, ring, _raw=False)
    if sol is None:
        return S.EmptySet
    #sol = {sym:val for sym, val in sol.items() if sym != val}
    sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))

    if gen is not None:
        solsym = sol.free_symbols
        rep = {sym: next(gen) for sym in symbols if sym in solsym}
        sol = sol.subs(rep)

    return sol


##############################################################################
# ------------------------------nonlinsolve ---------------------------------#
##############################################################################


def _return_conditionset(eqs, symbols):
    # return conditionset
    eqs = (Eq(lhs, 0) for lhs in eqs)
    condition_set = ConditionSet(
        Tuple(*symbols), And(*eqs), S.Complexes**len(symbols))
    return condition_set


def substitution(system, symbols, result=[{}], known_symbols=[],
                 exclude=[], all_symbols=None):
    r"""
    Solves the `system` using substitution method. It is used in
    :func:`~.nonlinsolve`. This will be called from :func:`~.nonlinsolve` when any
    equation(s) is non polynomial equation.

    Parameters
    ==========

    system : list of equations
        The target system of equations
    symbols : list of symbols to be solved.
        The variable(s) for which the system is solved
    known_symbols : list of solved symbols
        Values are known for these variable(s)
    result : An empty list or list of dict
        If No symbol values is known then empty list otherwise
        symbol as keys and corresponding value in dict.
    exclude : Set of expression.
        Mostly denominator expression(s) of the equations of the system.
        Final solution should not satisfy these expressions.
    all_symbols : known_symbols + symbols(unsolved).

    Returns
    =======

    A FiniteSet of ordered tuple of values of `all_symbols` for which the
    `system` has solution. Order of values in the tuple is same as symbols
    present in the parameter `all_symbols`. If parameter `all_symbols` is None
    then same as symbols present in the parameter `symbols`.

    Please note that general FiniteSet is unordered, the solution returned
    here is not simply a FiniteSet of solutions, rather it is a FiniteSet of
    ordered tuple, i.e. the first & only argument to FiniteSet is a tuple of
    solutions, which is ordered, & hence the returned solution is ordered.

    Also note that solution could also have been returned as an ordered tuple,
    FiniteSet is just a wrapper `{}` around the tuple. It has no other
    significance except for the fact it is just used to maintain a consistent
    output format throughout the solveset.

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.
    AttributeError
        The input symbols are not :class:`~.Symbol` type.

    Examples
    ========

    >>> from sympy import symbols, substitution
    >>> x, y = symbols('x, y', real=True)
    >>> substitution([x + y], [x], [{y: 1}], [y], set([]), [x, y])
    {(-1, 1)}

    * When you want a soln not satisfying $x + 1 = 0$

    >>> substitution([x + y], [x], [{y: 1}], [y], set([x + 1]), [y, x])
    EmptySet
    >>> substitution([x + y], [x], [{y: 1}], [y], set([x - 1]), [y, x])
    {(1, -1)}
    >>> substitution([x + y - 1, y - x**2 + 5], [x, y])
    {(-3, 4), (2, -1)}

    * Returns both real and complex solution

    >>> x, y, z = symbols('x, y, z')
    >>> from sympy import exp, sin
    >>> substitution([exp(x) - sin(y), y**2 - 4], [x, y])
    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}

    >>> eqs = [z**2 + exp(2*x) - sin(y), -3 + exp(-y)]
    >>> substitution(eqs, [y, z])
    {(-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
     (-log(3), sqrt(-exp(2*x) - sin(log(3)))),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
      ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
      ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers))}

    """

    if not system:
        return S.EmptySet

    for i, e in enumerate(system):
        if isinstance(e, Eq):
            system[i] = e.lhs - e.rhs

    if not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise ValueError(filldedent(msg))

    if not is_sequence(symbols):
        msg = ('symbols should be given as a sequence, e.g. a list.'
               'Not type %s: %s')
        raise TypeError(filldedent(msg % (type(symbols), symbols)))

    if not getattr(symbols[0], 'is_Symbol', False):
        msg = ('Iterable of symbols must be given as '
               'second argument, not type %s: %s')
        raise ValueError(filldedent(msg % (type(symbols[0]), symbols[0])))

    # By default `all_symbols` will be same as `symbols`
    if all_symbols is None:
        all_symbols = symbols

    old_result = result
    # storing complements and intersection for particular symbol
    complements = {}
    intersections = {}

    # when total_solveset_call equals total_conditionset
    # it means that solveset failed to solve all eqs.
    total_conditionset = -1
    total_solveset_call = -1

    def _unsolved_syms(eq, sort=False):
        """Returns the unsolved symbol present
        in the equation `eq`.
        """
        free = eq.free_symbols
        unsolved = (free - set(known_symbols)) & set(all_symbols)
        if sort:
            unsolved = list(unsolved)
            unsolved.sort(key=default_sort_key)
        return unsolved

    # sort such that equation with the fewest potential symbols is first.
    # means eq with less number of variable first in the list.
    eqs_in_better_order = list(
        ordered(system, lambda _: len(_unsolved_syms(_))))

    def add_intersection_complement(result, intersection_dict, complement_dict):
        # If solveset has returned some intersection/complement
        # for any symbol, it will be added in the final solution.
        final_result = []
        for res in result:
            res_copy = res
            for key_res, value_res in res.items():
                intersect_set, complement_set = None, None
                for key_sym, value_sym in intersection_dict.items():
                    if key_sym == key_res:
                        intersect_set = value_sym
                for key_sym, value_sym in complement_dict.items():
                    if key_sym == key_res:
                        complement_set = value_sym
                if intersect_set or complement_set:
                    new_value = FiniteSet(value_res)
                    if intersect_set and intersect_set != S.Complexes:
                        new_value = Intersection(new_value, intersect_set)
                    if complement_set:
                        new_value = Complement(new_value, complement_set)
                    if new_value is S.EmptySet:
                        res_copy = None
                        break
                    elif new_value.is_FiniteSet and len(new_value) == 1:
                        res_copy[key_res] = set(new_value).pop()
                    else:
                        res_copy[key_res] = new_value

            if res_copy is not None:
                final_result.append(res_copy)
        return final_result

    def _extract_main_soln(sym, sol, soln_imageset):
        """Separate the Complements, Intersections, ImageSet lambda expr and
        its base_set. This function returns the unmasked sol from different classes
        of sets and also returns the appended ImageSet elements in a
        soln_imageset dict: `{unmasked element: ImageSet}`.
        """
        # if there is union, then need to check
        # Complement, Intersection, Imageset.
        # Order should not be changed.
        if isinstance(sol, ConditionSet):
            # extracts any solution in ConditionSet
            sol = sol.base_set

        if isinstance(sol, Complement):
            # extract solution and complement
            complements[sym] = sol.args[1]
            sol = sol.args[0]
            # complement will be added at the end
            # using `add_intersection_complement` method

        # if there is union of Imageset or other in soln.
        # no testcase is written for this if block
        if isinstance(sol, Union):
            sol_args = sol.args
            sol = S.EmptySet
            # We need in sequence so append finteset elements
            # and then imageset or other.
            for sol_arg2 in sol_args:
                if isinstance(sol_arg2, FiniteSet):
                    sol += sol_arg2
                else:
                    # ImageSet, Intersection, complement then
                    # append them directly
                    sol += FiniteSet(sol_arg2)

        if isinstance(sol, Intersection):
            # Interval/Set will be at 0th index always
            if sol.args[0] not in (S.Reals, S.Complexes):
                # Sometimes solveset returns soln with intersection
                # S.Reals or S.Complexes. We don't consider that
                # intersection.
                intersections[sym] = sol.args[0]
            sol = sol.args[1]
        # after intersection and complement Imageset should
        # be checked.
        if isinstance(sol, ImageSet):
            soln_imagest = sol
            expr2 = sol.lamda.expr
            sol = FiniteSet(expr2)
            soln_imageset[expr2] = soln_imagest

        if not isinstance(sol, FiniteSet):
            sol = FiniteSet(sol)
        return sol, soln_imageset

    def _check_exclude(rnew, imgset_yes):
        rnew_ = rnew
        if imgset_yes:
            # replace all dummy variables (Imageset lambda variables)
            # with zero before `checksol`. Considering fundamental soln
            # for `checksol`.
            rnew_copy = rnew.copy()
            dummy_n = imgset_yes[0]
            for key_res, value_res in rnew_copy.items():
                rnew_copy[key_res] = value_res.subs(dummy_n, 0)
            rnew_ = rnew_copy
        # satisfy_exclude == true if it satisfies the expr of `exclude` list.
        try:
            # something like : `Mod(-log(3), 2*I*pi)` can't be
            # simplified right now, so `checksol` returns `TypeError`.
            # when this issue is fixed this try block should be
            # removed. Mod(-log(3), 2*I*pi) == -log(3)
            satisfy_exclude = any(
                checksol(d, rnew_) for d in exclude)
        except TypeError:
            satisfy_exclude = None
        return satisfy_exclude

    def _restore_imgset(rnew, original_imageset, newresult):
        restore_sym = set(rnew.keys()) & \
            set(original_imageset.keys())
        for key_sym in restore_sym:
            img = original_imageset[key_sym]
            rnew[key_sym] = img
        if rnew not in newresult:
            newresult.append(rnew)

    def _append_eq(eq, result, res, delete_soln, n=None):
        u = Dummy('u')
        if n:
            eq = eq.subs(n, 0)
        satisfy = eq if eq in (True, False) else checksol(u, u, eq, minimal=True)
        if satisfy is False:
            delete_soln = True
            res = {}
        else:
            result.append(res)
        return result, res, delete_soln

    def _append_new_soln(rnew, sym, sol, imgset_yes, soln_imageset,
                         original_imageset, newresult, eq=None):
        """If `rnew` (A dict <symbol: soln>) contains valid soln
        append it to `newresult` list.
        `imgset_yes` is (base, dummy_var) if there was imageset in previously
         calculated result(otherwise empty tuple). `original_imageset` is dict
         of imageset expr and imageset from this result.
        `soln_imageset` dict of imageset expr and imageset of new soln.
        """
        satisfy_exclude = _check_exclude(rnew, imgset_yes)
        delete_soln = False
        # soln should not satisfy expr present in `exclude` list.
        if not satisfy_exclude:
            local_n = None
            # if it is imageset
            if imgset_yes:
                local_n = imgset_yes[0]
                base = imgset_yes[1]
                if sym and sol:
                    # when `sym` and `sol` is `None` means no new
                    # soln. In that case we will append rnew directly after
                    # substituting original imagesets in rnew values if present
                    # (second last line of this function using _restore_imgset)
                    dummy_list = list(sol.atoms(Dummy))
                    # use one dummy `n` which is in
                    # previous imageset
                    local_n_list = [
                        local_n for i in range(
                            0, len(dummy_list))]

                    dummy_zip = zip(dummy_list, local_n_list)
                    lam = Lambda(local_n, sol.subs(dummy_zip))
                    rnew[sym] = ImageSet(lam, base)
                if eq is not None:
                    newresult, rnew, delete_soln = _append_eq(
                        eq, newresult, rnew, delete_soln, local_n)
            elif eq is not None:
                newresult, rnew, delete_soln = _append_eq(
                    eq, newresult, rnew, delete_soln)
            elif sol in soln_imageset.keys():
                rnew[sym] = soln_imageset[sol]
                # restore original imageset
                _restore_imgset(rnew, original_imageset, newresult)
            else:
                newresult.append(rnew)
        elif satisfy_exclude:
            delete_soln = True
            rnew = {}
        _restore_imgset(rnew, original_imageset, newresult)
        return newresult, delete_soln

    def _new_order_result(result, eq):
        # separate first, second priority. `res` that makes `eq` value equals
        # to zero, should be used first then other result(second priority).
        # If it is not done then we may miss some soln.
        first_priority = []
        second_priority = []
        for res in result:
            if not any(isinstance(val, ImageSet) for val in res.values()):
                if eq.subs(res) == 0:
                    first_priority.append(res)
                else:
                    second_priority.append(res)
        if first_priority or second_priority:
            return first_priority + second_priority
        return result

    def _solve_using_known_values(result, solver):
        """Solves the system using already known solution
        (result contains the dict <symbol: value>).
        solver is :func:`~.solveset_complex` or :func:`~.solveset_real`.
        """
        # stores imageset <expr: imageset(Lambda(n, expr), base)>.
        soln_imageset = {}
        total_solvest_call = 0
        total_conditionst = 0

        # sort equations so the one with the fewest potential
        # symbols appears first
        for index, eq in enumerate(eqs_in_better_order):
            newresult = []
            # if imageset, expr is used to solve for other symbol
            imgset_yes = False
            for res in result:
                original_imageset = {}
                got_symbol = set()  # symbols solved in one iteration
                # find the imageset and use its expr.
                for k, v in res.items():
                    if isinstance(v, ImageSet):
                        res[k] = v.lamda.expr
                        original_imageset[k] = v
                        dummy_n = v.lamda.expr.atoms(Dummy).pop()
                        (base,) = v.base_sets
                        imgset_yes = (dummy_n, base)
                    assert not isinstance(v, FiniteSet)  # if so, internal error
                # update eq with everything that is known so far
                eq2 = eq.subs(res).expand()
                if imgset_yes and not eq2.has(imgset_yes[0]):
                    # The substituted equation simplified in such a way that
                    # it's no longer necessary to encapsulate a potential new
                    # solution in an ImageSet. (E.g. at the previous step some
                    # {n*2*pi} was found as partial solution for one of the
                    # unknowns, but its main solution expression n*2*pi has now
                    # been substituted in a trigonometric function.)
                    imgset_yes = False

                unsolved_syms = _unsolved_syms(eq2, sort=True)
                if not unsolved_syms:
                    if res:
                        newresult, delete_res = _append_new_soln(
                            res, None, None, imgset_yes, soln_imageset,
                            original_imageset, newresult, eq2)
                        if delete_res:
                            # `delete_res` is true, means substituting `res` in
                            # eq2 doesn't return `zero` or deleting the `res`
                            # (a soln) since it satisfies expr of `exclude`
                            # list.
                            result.remove(res)
                    continue  # skip as it's independent of desired symbols
                depen1, depen2 = eq2.as_independent(*unsolved_syms)
                if (depen1.has(Abs) or depen2.has(Abs)) and solver == solveset_complex:
                    # Absolute values cannot be inverted in the
                    # complex domain
                    continue
                soln_imageset = {}
                for sym in unsolved_syms:
                    not_solvable = False
                    try:
                        soln = solver(eq2, sym)
                        total_solvest_call += 1
                        soln_new = S.EmptySet
                        if isinstance(soln, Complement):
                            # separate solution and complement
                            complements[sym] = soln.args[1]
                            soln = soln.args[0]
                            # complement will be added at the end
                        if isinstance(soln, Intersection):
                            # Interval will be at 0th index always
                            if soln.args[0] != Interval(-oo, oo):
                                # sometimes solveset returns soln
                                # with intersection S.Reals, to confirm that
                                # soln is in domain=S.Reals
                                intersections[sym] = soln.args[0]
                            soln_new += soln.args[1]
                        soln = soln_new if soln_new else soln
                        if index > 0 and solver == solveset_real:
                            # one symbol's real soln, another symbol may have
                            # corresponding complex soln.
                            if not isinstance(soln, (ImageSet, ConditionSet)):
                                soln += solveset_complex(eq2, sym)  # might give ValueError with Abs
                    except (NotImplementedError, ValueError):
                        # If solveset is not able to solve equation `eq2`. Next
                        # time we may get soln using next equation `eq2`
                        continue
                    if isinstance(soln, ConditionSet):
                        if soln.base_set in (S.Reals, S.Complexes):
                            soln = S.EmptySet
                            # don't do `continue` we may get soln
                            # in terms of other symbol(s)
                            not_solvable = True
                            total_conditionst += 1
                        else:
                            soln = soln.base_set

                    if soln is not S.EmptySet:
                        soln, soln_imageset = _extract_main_soln(
                            sym, soln, soln_imageset)

                    for sol in soln:
                        # sol is not a `Union` since we checked it
                        # before this loop
                        sol, soln_imageset = _extract_main_soln(
                            sym, sol, soln_imageset)
                        sol = set(sol).pop()  # XXX what if there are more solutions?
                        free = sol.free_symbols
                        if got_symbol and any(
                            ss in free for ss in got_symbol
                        ):
                            # sol depends on previously solved symbols
                            # then continue
                            continue
                        rnew = res.copy()
                        # put each solution in res and append the new  result
                        # in the new result list (solution for symbol `s`)
                        # along with old results.
                        for k, v in res.items():
                            if isinstance(v, Expr) and isinstance(sol, Expr):
                                # if any unsolved symbol is present
                                # Then subs known value
                                rnew[k] = v.subs(sym, sol)
                        # and add this new solution
                        if sol in soln_imageset.keys():
                            # replace all lambda variables with 0.
                            imgst = soln_imageset[sol]
                            rnew[sym] = imgst.lamda(
                                *[0 for i in range(0, len(
                                    imgst.lamda.variables))])
                        else:
                            rnew[sym] = sol
                        newresult, delete_res = _append_new_soln(
                            rnew, sym, sol, imgset_yes, soln_imageset,
                            original_imageset, newresult)
                        if delete_res:
                            # deleting the `res` (a soln) since it satisfies
                            # eq of `exclude` list
                            result.remove(res)
                    # solution got for sym
                    if not not_solvable:
                        got_symbol.add(sym)
            # next time use this new soln
            if newresult:
                result = newresult
        return result, total_solvest_call, total_conditionst

    new_result_real, solve_call1, cnd_call1 = _solve_using_known_values(
        old_result, solveset_real)
    new_result_complex, solve_call2, cnd_call2 = _solve_using_known_values(
        old_result, solveset_complex)

    # If total_solveset_call is equal to total_conditionset
    # then solveset failed to solve all of the equations.
    # In this case we return a ConditionSet here.
    total_conditionset += (cnd_call1 + cnd_call2)
    total_solveset_call += (solve_call1 + solve_call2)

    if total_conditionset == total_solveset_call and total_solveset_call != -1:
        return _return_conditionset(eqs_in_better_order, all_symbols)

    # don't keep duplicate solutions
    filtered_complex = []
    for i in list(new_result_complex):
        for j in list(new_result_real):
            if i.keys() != j.keys():
                continue
            if all(a.dummy_eq(b) for a, b in zip(i.values(), j.values()) \
                if not (isinstance(a, int) and isinstance(b, int))):
                break
        else:
            filtered_complex.append(i)
    # overall result
    result = new_result_real + filtered_complex

    result_all_variables = []
    result_infinite = []
    for res in result:
        if not res:
            # means {None : None}
            continue
        # If length < len(all_symbols) means infinite soln.
        # Some or all the soln is dependent on 1 symbol.
        # eg. {x: y+2} then final soln {x: y+2, y: y}
        if len(res) < len(all_symbols):
            solved_symbols = res.keys()
            unsolved = list(filter(
                lambda x: x not in solved_symbols, all_symbols))
            for unsolved_sym in unsolved:
                res[unsolved_sym] = unsolved_sym
            result_infinite.append(res)
        if res not in result_all_variables:
            result_all_variables.append(res)

    if result_infinite:
        # we have general soln
        # eg : [{x: -1, y : 1}, {x : -y, y: y}] then
        # return [{x : -y, y : y}]
        result_all_variables = result_infinite
    if intersections or complements:
        result_all_variables = add_intersection_complement(
            result_all_variables, intersections, complements)

    # convert to ordered tuple
    result = S.EmptySet
    for r in result_all_variables:
        temp = [r[symb] for symb in all_symbols]
        result += FiniteSet(tuple(temp))
    return result


def _solveset_work(system, symbols):
    soln = solveset(system[0], symbols[0])
    if isinstance(soln, FiniteSet):
        _soln = FiniteSet(*[(s,) for s in soln])
        return _soln
    else:
        return FiniteSet(tuple(FiniteSet(soln)))


def _handle_positive_dimensional(polys, symbols, denominators):
    from sympy.polys.polytools import groebner
    # substitution method where new system is groebner basis of the system
    _symbols = list(symbols)
    _symbols.sort(key=default_sort_key)
    basis = groebner(polys, _symbols, polys=True)
    new_system = []
    for poly_eq in basis:
        new_system.append(poly_eq.as_expr())
    result = [{}]
    result = substitution(
        new_system, symbols, result, [],
        denominators)
    return result


def _handle_zero_dimensional(polys, symbols, system):
    # solve 0 dimensional poly system using `solve_poly_system`
    result = solve_poly_system(polys, *symbols)
    # May be some extra soln is added because
    # we used `unrad` in `_separate_poly_nonpoly`, so
    # need to check and remove if it is not a soln.
    result_update = S.EmptySet
    for res in result:
        dict_sym_value = dict(list(zip(symbols, res)))
        if all(checksol(eq, dict_sym_value) for eq in system):
            result_update += FiniteSet(res)
    return result_update


def _separate_poly_nonpoly(system, symbols):
    polys = []
    polys_expr = []
    nonpolys = []
    # unrad_changed stores a list of expressions containing
    # radicals that were processed using unrad
    # this is useful if solutions need to be checked later.
    unrad_changed = []
    denominators = set()
    poly = None
    for eq in system:
        # Store denom expressions that contain symbols
        denominators.update(_simple_dens(eq, symbols))
        # Convert equality to expression
        if isinstance(eq, Eq):
            eq = eq.lhs - eq.rhs
        # try to remove sqrt and rational power
        without_radicals = unrad(simplify(eq), *symbols)
        if without_radicals:
            unrad_changed.append(eq)
            eq_unrad, cov = without_radicals
            if not cov:
                eq = eq_unrad
        if isinstance(eq, Expr):
            eq = eq.as_numer_denom()[0]
            poly = eq.as_poly(*symbols, extension=True)
        elif simplify(eq).is_number:
            continue
        if poly is not None:
            polys.append(poly)
            polys_expr.append(poly.as_expr())
        else:
            nonpolys.append(eq)
    return polys, polys_expr, nonpolys, denominators, unrad_changed


def _handle_poly(polys, symbols):
    # _handle_poly(polys, symbols) -> (poly_sol, poly_eqs)
    #
    # We will return possible solution information to nonlinsolve as well as a
    # new system of polynomial equations to be solved if we cannot solve
    # everything directly here. The new system of polynomial equations will be
    # a lex-order Groebner basis for the original system. The lex basis
    # hopefully separate some of the variables and equations and give something
    # easier for substitution to work with.

    # The format for representing solution sets in nonlinsolve and substitution
    # is a list of dicts. These are the special cases:
    no_information = [{}]   # No equations solved yet
    no_solutions = []       # The system is inconsistent and has no solutions.

    # If there is no need to attempt further solution of these equations then
    # we return no equations:
    no_equations = []

    inexact = any(not p.domain.is_Exact for p in polys)
    if inexact:
        # The use of Groebner over RR is likely to result incorrectly in an
        # inconsistent Groebner basis. So, convert any float coefficients to
        # Rational before computing the Groebner basis.
        polys = [poly(nsimplify(p, rational=True)) for p in polys]

    # Compute a Groebner basis in grevlex order wrt the ordering given. We will
    # try to convert this to lex order later. Usually it seems to be more
    # efficient to compute a lex order basis by computing a grevlex basis and
    # converting to lex with fglm.
    basis = groebner(polys, symbols, order='grevlex', polys=False)

    #
    # No solutions (inconsistent equations)?
    #
    if 1 in basis:

        # No solutions:
        poly_sol = no_solutions
        poly_eqs = no_equations

    #
    # Finite number of solutions (zero-dimensional case)
    #
    elif basis.is_zero_dimensional:

        # Convert Groebner basis to lex ordering
        basis = basis.fglm('lex')

        # Convert polynomial coefficients back to float before calling
        # solve_poly_system
        if inexact:
            basis = [nfloat(p) for p in basis]

        # Solve the zero-dimensional case using solve_poly_system if possible.
        # If some polynomials have factors that cannot be solved in radicals
        # then this will fail. Using solve_poly_system(..., strict=True)
        # ensures that we either get a complete solution set in radicals or
        # UnsolvableFactorError will be raised.
        try:
            result = solve_poly_system(basis, *symbols, strict=True)
        except UnsolvableFactorError:
            # Failure... not fully solvable in radicals. Return the lex-order
            # basis for substitution to handle.
            poly_sol = no_information
            poly_eqs = list(basis)
        else:
            # Success! We have a finite solution set and solve_poly_system has
            # succeeded in finding all solutions. Return the solutions and also
            # an empty list of remaining equations to be solved.
            poly_sol = [dict(zip(symbols, res)) for res in result]
            poly_eqs = no_equations

    #
    # Infinite families of solutions (positive-dimensional case)
    #
    else:
        # In this case the grevlex basis cannot be converted to lex using the
        # fglm method and also solve_poly_system cannot solve the equations. We
        # would like to return a lex basis but since we can't use fglm we
        # compute the lex basis directly here. The time required to recompute
        # the basis is generally significantly less than the time required by
        # substitution to solve the new system.
        poly_sol = no_information
        poly_eqs = list(groebner(polys, symbols, order='lex', polys=False))

        if inexact:
            poly_eqs = [nfloat(p) for p in poly_eqs]

    return poly_sol, poly_eqs


def nonlinsolve(system, *symbols):
    r"""
    Solve system of $N$ nonlinear equations with $M$ variables, which means both
    under and overdetermined systems are supported. Positive dimensional
    system is also supported (A system with infinitely many solutions is said
    to be positive-dimensional). In a positive dimensional system the solution will
    be dependent on at least one symbol. Returns both real solution
    and complex solution (if they exist).

    Parameters
    ==========

    system : list of equations
        The target system of equations
    symbols : list of Symbols
        symbols should be given as a sequence eg. list

    Returns
    =======

    A :class:`~.FiniteSet` of ordered tuple of values of `symbols` for which the `system`
    has solution. Order of values in the tuple is same as symbols present in
    the parameter `symbols`.

    Please note that general :class:`~.FiniteSet` is unordered, the solution
    returned here is not simply a :class:`~.FiniteSet` of solutions, rather it
    is a :class:`~.FiniteSet` of ordered tuple, i.e. the first and only
    argument to :class:`~.FiniteSet` is a tuple of solutions, which is
    ordered, and, hence ,the returned solution is ordered.

    Also note that solution could also have been returned as an ordered tuple,
    FiniteSet is just a wrapper ``{}`` around the tuple. It has no other
    significance except for the fact it is just used to maintain a consistent
    output format throughout the solveset.

    For the given set of equations, the respective input types
    are given below:

    .. math:: xy - 1 = 0
    .. math:: 4x^2 + y^2 - 5 = 0

    ::

       system  = [x*y - 1, 4*x**2 + y**2 - 5]
       symbols = [x, y]

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.
    AttributeError
        The input symbols are not `Symbol` type.

    Examples
    ========

    >>> from sympy import symbols, nonlinsolve
    >>> x, y, z = symbols('x, y, z', real=True)
    >>> nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y])
    {(-1, -1), (-1/2, -2), (1/2, 2), (1, 1)}

    1. Positive dimensional system and complements:

    >>> from sympy import pprint
    >>> from sympy.polys.polytools import is_zero_dimensional
    >>> a, b, c, d = symbols('a, b, c, d', extended_real=True)
    >>> eq1 =  a + b + c + d
    >>> eq2 = a*b + b*c + c*d + d*a
    >>> eq3 = a*b*c + b*c*d + c*d*a + d*a*b
    >>> eq4 = a*b*c*d - 1
    >>> system = [eq1, eq2, eq3, eq4]
    >>> is_zero_dimensional(system)
    False
    >>> pprint(nonlinsolve(system, [a, b, c, d]), use_unicode=False)
      -1       1               1      -1
    {(---, -d, -, {d} \ {0}), (-, -d, ---, {d} \ {0})}
       d       d               d       d
    >>> nonlinsolve([(x+y)**2 - 4, x + y - 2], [x, y])
    {(2 - y, y)}

    2. If some of the equations are non-polynomial then `nonlinsolve`
    will call the ``substitution`` function and return real and complex solutions,
    if present.

    >>> from sympy import exp, sin
    >>> nonlinsolve([exp(x) - sin(y), y**2 - 4], [x, y])
    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}

    3. If system is non-linear polynomial and zero-dimensional then it
    returns both solution (real and complex solutions, if present) using
    :func:`~.solve_poly_system`:

    >>> from sympy import sqrt
    >>> nonlinsolve([x**2 - 2*y**2 -2, x*y - 2], [x, y])
    {(-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I)}

    4. ``nonlinsolve`` can solve some linear (zero or positive dimensional)
    system (because it uses the :func:`sympy.polys.polytools.groebner` function to get the
    groebner basis and then uses the ``substitution`` function basis as the
    new `system`). But it is not recommended to solve linear system using
    ``nonlinsolve``, because :func:`~.linsolve` is better for general linear systems.

    >>> nonlinsolve([x + 2*y -z - 3, x - y - 4*z + 9, y + z - 4], [x, y, z])
    {(3*z - 5, 4 - z, z)}

    5. System having polynomial equations and only real solution is
    solved using :func:`~.solve_poly_system`:

    >>> e1 = sqrt(x**2 + y**2) - 10
    >>> e2 = sqrt(y**2 + (-x + 10)**2) - 3
    >>> nonlinsolve((e1, e2), (x, y))
    {(191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20)}
    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [x, y])
    {(1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5))}
    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [y, x])
    {(2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5))}

    6. It is better to use symbols instead of trigonometric functions or
    :class:`~.Function`. For example, replace $\sin(x)$ with a symbol, replace
    $f(x)$ with a symbol and so on. Get a solution from ``nonlinsolve`` and then
    use :func:`~.solveset` to get the value of $x$.

    How nonlinsolve is better than old solver ``_solve_system`` :
    =============================================================

    1. A positive dimensional system solver: nonlinsolve can return
    solution for positive dimensional system. It finds the
    Groebner Basis of the positive dimensional system(calling it as
    basis) then we can start solving equation(having least number of
    variable first in the basis) using solveset and substituting that
    solved solutions into other equation(of basis) to get solution in
    terms of minimum variables. Here the important thing is how we
    are substituting the known values and in which equations.

    2. Real and complex solutions: nonlinsolve returns both real
    and complex solution. If all the equations in the system are polynomial
    then using :func:`~.solve_poly_system` both real and complex solution is returned.
    If all the equations in the system are not polynomial equation then goes to
    ``substitution`` method with this polynomial and non polynomial equation(s),
    to solve for unsolved variables. Here to solve for particular variable
    solveset_real and solveset_complex is used. For both real and complex
    solution ``_solve_using_known_values`` is used inside ``substitution``
    (``substitution`` will be called when any non-polynomial equation is present).
    If a solution is valid its general solution is added to the final result.

    3. :class:`~.Complement` and :class:`~.Intersection` will be added:
    nonlinsolve maintains dict for complements and intersections. If solveset
    find complements or/and intersections with any interval or set during the
    execution of ``substitution`` function, then complement or/and
    intersection for that variable is added before returning final solution.

    """
    if not system:
        return S.EmptySet

    if not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise ValueError(filldedent(msg))

    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]

    if not is_sequence(symbols) or not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise IndexError(filldedent(msg))

    symbols = list(map(_sympify, symbols))
    system, symbols, swap = recast_to_symbols(system, symbols)
    if swap:
        soln = nonlinsolve(system, symbols)
        return FiniteSet(*[tuple(i.xreplace(swap) for i in s) for s in soln])

    if len(system) == 1 and len(symbols) == 1:
        return _solveset_work(system, symbols)

    # main code of def nonlinsolve() starts from here

    polys, polys_expr, nonpolys, denominators, unrad_changed = \
        _separate_poly_nonpoly(system, symbols)

    poly_eqs = []
    poly_sol = [{}]

    if polys:
        poly_sol, poly_eqs = _handle_poly(polys, symbols)
        if poly_sol and poly_sol[0]:
            poly_syms = set().union(*(eq.free_symbols for eq in polys))
            unrad_syms = set().union(*(eq.free_symbols for eq in unrad_changed))
            if unrad_syms == poly_syms and unrad_changed:
                # if all the symbols have been solved by _handle_poly
                # and unrad has been used then check solutions
                poly_sol = [sol for sol in poly_sol if checksol(unrad_changed, sol)]

    # Collect together the unsolved polynomials with the non-polynomial
    # equations.
    remaining = poly_eqs + nonpolys

    # to_tuple converts a solution dictionary to a tuple containing the
    # value for each symbol
    to_tuple = lambda sol: tuple(sol[s] for s in symbols)

    if not remaining:
        # If there is nothing left to solve then return the solution from
        # solve_poly_system directly.
        return FiniteSet(*map(to_tuple, poly_sol))
    else:
        # Here we handle:
        #
        #  1. The Groebner basis if solve_poly_system failed.
        #  2. The Groebner basis in the positive-dimensional case.
        #  3. Any non-polynomial equations
        #
        # If solve_poly_system did succeed then we pass those solutions in as
        # preliminary results.
        subs_res = substitution(remaining, symbols, result=poly_sol, exclude=denominators)

        if not isinstance(subs_res, FiniteSet):
            return subs_res

        # check solutions produced by substitution. Currently, checking is done for
        # only those solutions which have non-Set variable values.
        if unrad_changed:
            result = [dict(zip(symbols, sol)) for sol in subs_res.args]
            correct_sols = [sol for sol in result if any(isinstance(v, Set) for v in sol)
                            or checksol(unrad_changed, sol) != False]
            return FiniteSet(*map(to_tuple, correct_sols))
        else:
            return subs_res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\network.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Network
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security


class ResourceType(enum.Enum):
    '''
    Resource type as it was perceived by the rendering engine.
    '''
    DOCUMENT = "Document"
    STYLESHEET = "Stylesheet"
    IMAGE = "Image"
    MEDIA = "Media"
    FONT = "Font"
    SCRIPT = "Script"
    TEXT_TRACK = "TextTrack"
    XHR = "XHR"
    FETCH = "Fetch"
    PREFETCH = "Prefetch"
    EVENT_SOURCE = "EventSource"
    WEB_SOCKET = "WebSocket"
    MANIFEST = "Manifest"
    SIGNED_EXCHANGE = "SignedExchange"
    PING = "Ping"
    CSP_VIOLATION_REPORT = "CSPViolationReport"
    PREFLIGHT = "Preflight"
    OTHER = "Other"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class LoaderId(str):
    '''
    Unique loader identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> LoaderId:
        return cls(json)

    def __repr__(self):
        return 'LoaderId({})'.format(super().__repr__())


class RequestId(str):
    '''
    Unique network request identifier.
    Note that this does not identify individual HTTP requests that are part of
    a network request.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RequestId:
        return cls(json)

    def __repr__(self):
        return 'RequestId({})'.format(super().__repr__())


class InterceptionId(str):
    '''
    Unique intercepted request identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> InterceptionId:
        return cls(json)

    def __repr__(self):
        return 'InterceptionId({})'.format(super().__repr__())


class ErrorReason(enum.Enum):
    '''
    Network level fetch failure reason.
    '''
    FAILED = "Failed"
    ABORTED = "Aborted"
    TIMED_OUT = "TimedOut"
    ACCESS_DENIED = "AccessDenied"
    CONNECTION_CLOSED = "ConnectionClosed"
    CONNECTION_RESET = "ConnectionReset"
    CONNECTION_REFUSED = "ConnectionRefused"
    CONNECTION_ABORTED = "ConnectionAborted"
    CONNECTION_FAILED = "ConnectionFailed"
    NAME_NOT_RESOLVED = "NameNotResolved"
    INTERNET_DISCONNECTED = "InternetDisconnected"
    ADDRESS_UNREACHABLE = "AddressUnreachable"
    BLOCKED_BY_CLIENT = "BlockedByClient"
    BLOCKED_BY_RESPONSE = "BlockedByResponse"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class TimeSinceEpoch(float):
    '''
    UTC time in seconds, counted from January 1, 1970.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> TimeSinceEpoch:
        return cls(json)

    def __repr__(self):
        return 'TimeSinceEpoch({})'.format(super().__repr__())


class MonotonicTime(float):
    '''
    Monotonically increasing time in seconds since an arbitrary point in the past.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> MonotonicTime:
        return cls(json)

    def __repr__(self):
        return 'MonotonicTime({})'.format(super().__repr__())


class Headers(dict):
    '''
    Request / response headers as keys / values of JSON object.
    '''
    def to_json(self) -> dict:
        return self

    @classmethod
    def from_json(cls, json: dict) -> Headers:
        return cls(json)

    def __repr__(self):
        return 'Headers({})'.format(super().__repr__())


class ConnectionType(enum.Enum):
    '''
    The underlying connection technology that the browser is supposedly using.
    '''
    NONE = "none"
    CELLULAR2G = "cellular2g"
    CELLULAR3G = "cellular3g"
    CELLULAR4G = "cellular4g"
    BLUETOOTH = "bluetooth"
    ETHERNET = "ethernet"
    WIFI = "wifi"
    WIMAX = "wimax"
    OTHER = "other"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieSameSite(enum.Enum):
    '''
    Represents the cookie's 'SameSite' status:
    https://tools.ietf.org/html/draft-west-first-party-cookies
    '''
    STRICT = "Strict"
    LAX = "Lax"
    NONE = "None"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookiePriority(enum.Enum):
    '''
    Represents the cookie's 'Priority' status:
    https://tools.ietf.org/html/draft-west-cookie-priority-00
    '''
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieSourceScheme(enum.Enum):
    '''
    Represents the source scheme of the origin that originally set the cookie.
    A value of "Unset" allows protocol clients to emulate legacy cookie scope for the scheme.
    This is a temporary ability and it will be removed in the future.
    '''
    UNSET = "Unset"
    NON_SECURE = "NonSecure"
    SECURE = "Secure"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ResourceTiming:
    '''
    Timing information for the request.
    '''
    #: Timing's requestTime is a baseline in seconds, while the other numbers are ticks in
    #: milliseconds relatively to this requestTime.
    request_time: float

    #: Started resolving proxy.
    proxy_start: float

    #: Finished resolving proxy.
    proxy_end: float

    #: Started DNS address resolve.
    dns_start: float

    #: Finished DNS address resolve.
    dns_end: float

    #: Started connecting to the remote host.
    connect_start: float

    #: Connected to the remote host.
    connect_end: float

    #: Started SSL handshake.
    ssl_start: float

    #: Finished SSL handshake.
    ssl_end: float

    #: Started running ServiceWorker.
    worker_start: float

    #: Finished Starting ServiceWorker.
    worker_ready: float

    #: Started fetch event.
    worker_fetch_start: float

    #: Settled fetch event respondWith promise.
    worker_respond_with_settled: float

    #: Started sending request.
    send_start: float

    #: Finished sending request.
    send_end: float

    #: Time the server started pushing request.
    push_start: float

    #: Time the server finished pushing request.
    push_end: float

    #: Started receiving response headers.
    receive_headers_start: float

    #: Finished receiving response headers.
    receive_headers_end: float

    #: Started ServiceWorker static routing source evaluation.
    worker_router_evaluation_start: typing.Optional[float] = None

    #: Started cache lookup when the source was evaluated to ``cache``.
    worker_cache_lookup_start: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['requestTime'] = self.request_time
        json['proxyStart'] = self.proxy_start
        json['proxyEnd'] = self.proxy_end
        json['dnsStart'] = self.dns_start
        json['dnsEnd'] = self.dns_end
        json['connectStart'] = self.connect_start
        json['connectEnd'] = self.connect_end
        json['sslStart'] = self.ssl_start
        json['sslEnd'] = self.ssl_end
        json['workerStart'] = self.worker_start
        json['workerReady'] = self.worker_ready
        json['workerFetchStart'] = self.worker_fetch_start
        json['workerRespondWithSettled'] = self.worker_respond_with_settled
        json['sendStart'] = self.send_start
        json['sendEnd'] = self.send_end
        json['pushStart'] = self.push_start
        json['pushEnd'] = self.push_end
        json['receiveHeadersStart'] = self.receive_headers_start
        json['receiveHeadersEnd'] = self.receive_headers_end
        if self.worker_router_evaluation_start is not None:
            json['workerRouterEvaluationStart'] = self.worker_router_evaluation_start
        if self.worker_cache_lookup_start is not None:
            json['workerCacheLookupStart'] = self.worker_cache_lookup_start
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request_time=float(json['requestTime']),
            proxy_start=float(json['proxyStart']),
            proxy_end=float(json['proxyEnd']),
            dns_start=float(json['dnsStart']),
            dns_end=float(json['dnsEnd']),
            connect_start=float(json['connectStart']),
            connect_end=float(json['connectEnd']),
            ssl_start=float(json['sslStart']),
            ssl_end=float(json['sslEnd']),
            worker_start=float(json['workerStart']),
            worker_ready=float(json['workerReady']),
            worker_fetch_start=float(json['workerFetchStart']),
            worker_respond_with_settled=float(json['workerRespondWithSettled']),
            send_start=float(json['sendStart']),
            send_end=float(json['sendEnd']),
            push_start=float(json['pushStart']),
            push_end=float(json['pushEnd']),
            receive_headers_start=float(json['receiveHeadersStart']),
            receive_headers_end=float(json['receiveHeadersEnd']),
            worker_router_evaluation_start=float(json['workerRouterEvaluationStart']) if 'workerRouterEvaluationStart' in json else None,
            worker_cache_lookup_start=float(json['workerCacheLookupStart']) if 'workerCacheLookupStart' in json else None,
        )


class ResourcePriority(enum.Enum):
    '''
    Loading priority of a resource request.
    '''
    VERY_LOW = "VeryLow"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "VeryHigh"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PostDataEntry:
    '''
    Post data entry for HTTP request
    '''
    bytes_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.bytes_ is not None:
            json['bytes'] = self.bytes_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            bytes_=str(json['bytes']) if 'bytes' in json else None,
        )


@dataclass
class Request:
    '''
    HTTP request data.
    '''
    #: Request URL (without fragment).
    url: str

    #: HTTP request method.
    method: str

    #: HTTP request headers.
    headers: Headers

    #: Priority of the resource request at the time request is sent.
    initial_priority: ResourcePriority

    #: The referrer policy of the request, as defined in https://www.w3.org/TR/referrer-policy/
    referrer_policy: str

    #: Fragment of the requested URL starting with hash, if present.
    url_fragment: typing.Optional[str] = None

    #: HTTP POST request data.
    #: Use postDataEntries instead.
    post_data: typing.Optional[str] = None

    #: True when the request has POST data. Note that postData might still be omitted when this flag is true when the data is too long.
    has_post_data: typing.Optional[bool] = None

    #: Request body elements (post data broken into individual entries).
    post_data_entries: typing.Optional[typing.List[PostDataEntry]] = None

    #: The mixed content type of the request.
    mixed_content_type: typing.Optional[security.MixedContentType] = None

    #: Whether is loaded via link preload.
    is_link_preload: typing.Optional[bool] = None

    #: Set for requests when the TrustToken API is used. Contains the parameters
    #: passed by the developer (e.g. via "fetch") as understood by the backend.
    trust_token_params: typing.Optional[TrustTokenParams] = None

    #: True if this resource request is considered to be the 'same site' as the
    #: request corresponding to the main frame.
    is_same_site: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['method'] = self.method
        json['headers'] = self.headers.to_json()
        json['initialPriority'] = self.initial_priority.to_json()
        json['referrerPolicy'] = self.referrer_policy
        if self.url_fragment is not None:
            json['urlFragment'] = self.url_fragment
        if self.post_data is not None:
            json['postData'] = self.post_data
        if self.has_post_data is not None:
            json['hasPostData'] = self.has_post_data
        if self.post_data_entries is not None:
            json['postDataEntries'] = [i.to_json() for i in self.post_data_entries]
        if self.mixed_content_type is not None:
            json['mixedContentType'] = self.mixed_content_type.to_json()
        if self.is_link_preload is not None:
            json['isLinkPreload'] = self.is_link_preload
        if self.trust_token_params is not None:
            json['trustTokenParams'] = self.trust_token_params.to_json()
        if self.is_same_site is not None:
            json['isSameSite'] = self.is_same_site
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            method=str(json['method']),
            headers=Headers.from_json(json['headers']),
            initial_priority=ResourcePriority.from_json(json['initialPriority']),
            referrer_policy=str(json['referrerPolicy']),
            url_fragment=str(json['urlFragment']) if 'urlFragment' in json else None,
            post_data=str(json['postData']) if 'postData' in json else None,
            has_post_data=bool(json['hasPostData']) if 'hasPostData' in json else None,
            post_data_entries=[PostDataEntry.from_json(i) for i in json['postDataEntries']] if 'postDataEntries' in json else None,
            mixed_content_type=security.MixedContentType.from_json(json['mixedContentType']) if 'mixedContentType' in json else None,
            is_link_preload=bool(json['isLinkPreload']) if 'isLinkPreload' in json else None,
            trust_token_params=TrustTokenParams.from_json(json['trustTokenParams']) if 'trustTokenParams' in json else None,
            is_same_site=bool(json['isSameSite']) if 'isSameSite' in json else None,
        )


@dataclass
class SignedCertificateTimestamp:
    '''
    Details of a signed certificate timestamp (SCT).
    '''
    #: Validation status.
    status: str

    #: Origin.
    origin: str

    #: Log name / description.
    log_description: str

    #: Log ID.
    log_id: str

    #: Issuance date. Unlike TimeSinceEpoch, this contains the number of
    #: milliseconds since January 1, 1970, UTC, not the number of seconds.
    timestamp: float

    #: Hash algorithm.
    hash_algorithm: str

    #: Signature algorithm.
    signature_algorithm: str

    #: Signature data.
    signature_data: str

    def to_json(self):
        json = dict()
        json['status'] = self.status
        json['origin'] = self.origin
        json['logDescription'] = self.log_description
        json['logId'] = self.log_id
        json['timestamp'] = self.timestamp
        json['hashAlgorithm'] = self.hash_algorithm
        json['signatureAlgorithm'] = self.signature_algorithm
        json['signatureData'] = self.signature_data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            status=str(json['status']),
            origin=str(json['origin']),
            log_description=str(json['logDescription']),
            log_id=str(json['logId']),
            timestamp=float(json['timestamp']),
            hash_algorithm=str(json['hashAlgorithm']),
            signature_algorithm=str(json['signatureAlgorithm']),
            signature_data=str(json['signatureData']),
        )


@dataclass
class SecurityDetails:
    '''
    Security details about a request.
    '''
    #: Protocol name (e.g. "TLS 1.2" or "QUIC").
    protocol: str

    #: Key Exchange used by the connection, or the empty string if not applicable.
    key_exchange: str

    #: Cipher name.
    cipher: str

    #: Certificate ID value.
    certificate_id: security.CertificateId

    #: Certificate subject name.
    subject_name: str

    #: Subject Alternative Name (SAN) DNS names and IP addresses.
    san_list: typing.List[str]

    #: Name of the issuing CA.
    issuer: str

    #: Certificate valid from date.
    valid_from: TimeSinceEpoch

    #: Certificate valid to (expiration) date
    valid_to: TimeSinceEpoch

    #: List of signed certificate timestamps (SCTs).
    signed_certificate_timestamp_list: typing.List[SignedCertificateTimestamp]

    #: Whether the request complied with Certificate Transparency policy
    certificate_transparency_compliance: CertificateTransparencyCompliance

    #: Whether the connection used Encrypted ClientHello
    encrypted_client_hello: bool

    #: (EC)DH group used by the connection, if applicable.
    key_exchange_group: typing.Optional[str] = None

    #: TLS MAC. Note that AEAD ciphers do not have separate MACs.
    mac: typing.Optional[str] = None

    #: The signature algorithm used by the server in the TLS server signature,
    #: represented as a TLS SignatureScheme code point. Omitted if not
    #: applicable or not known.
    server_signature_algorithm: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol
        json['keyExchange'] = self.key_exchange
        json['cipher'] = self.cipher
        json['certificateId'] = self.certificate_id.to_json()
        json['subjectName'] = self.subject_name
        json['sanList'] = [i for i in self.san_list]
        json['issuer'] = self.issuer
        json['validFrom'] = self.valid_from.to_json()
        json['validTo'] = self.valid_to.to_json()
        json['signedCertificateTimestampList'] = [i.to_json() for i in self.signed_certificate_timestamp_list]
        json['certificateTransparencyCompliance'] = self.certificate_transparency_compliance.to_json()
        json['encryptedClientHello'] = self.encrypted_client_hello
        if self.key_exchange_group is not None:
            json['keyExchangeGroup'] = self.key_exchange_group
        if self.mac is not None:
            json['mac'] = self.mac
        if self.server_signature_algorithm is not None:
            json['serverSignatureAlgorithm'] = self.server_signature_algorithm
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            protocol=str(json['protocol']),
            key_exchange=str(json['keyExchange']),
            cipher=str(json['cipher']),
            certificate_id=security.CertificateId.from_json(json['certificateId']),
            subject_name=str(json['subjectName']),
            san_list=[str(i) for i in json['sanList']],
            issuer=str(json['issuer']),
            valid_from=TimeSinceEpoch.from_json(json['validFrom']),
            valid_to=TimeSinceEpoch.from_json(json['validTo']),
            signed_certificate_timestamp_list=[SignedCertificateTimestamp.from_json(i) for i in json['signedCertificateTimestampList']],
            certificate_transparency_compliance=CertificateTransparencyCompliance.from_json(json['certificateTransparencyCompliance']),
            encrypted_client_hello=bool(json['encryptedClientHello']),
            key_exchange_group=str(json['keyExchangeGroup']) if 'keyExchangeGroup' in json else None,
            mac=str(json['mac']) if 'mac' in json else None,
            server_signature_algorithm=int(json['serverSignatureAlgorithm']) if 'serverSignatureAlgorithm' in json else None,
        )


class CertificateTransparencyCompliance(enum.Enum):
    '''
    Whether the request complied with Certificate Transparency policy.
    '''
    UNKNOWN = "unknown"
    NOT_COMPLIANT = "not-compliant"
    COMPLIANT = "compliant"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class BlockedReason(enum.Enum):
    '''
    The reason why request was blocked.
    '''
    OTHER = "other"
    CSP = "csp"
    MIXED_CONTENT = "mixed-content"
    ORIGIN = "origin"
    INSPECTOR = "inspector"
    SUBRESOURCE_FILTER = "subresource-filter"
    CONTENT_TYPE = "content-type"
    COEP_FRAME_RESOURCE_NEEDS_COEP_HEADER = "coep-frame-resource-needs-coep-header"
    COOP_SANDBOXED_IFRAME_CANNOT_NAVIGATE_TO_COOP_PAGE = "coop-sandboxed-iframe-cannot-navigate-to-coop-page"
    CORP_NOT_SAME_ORIGIN = "corp-not-same-origin"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_COEP = "corp-not-same-origin-after-defaulted-to-same-origin-by-coep"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_DIP = "corp-not-same-origin-after-defaulted-to-same-origin-by-dip"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_COEP_AND_DIP = "corp-not-same-origin-after-defaulted-to-same-origin-by-coep-and-dip"
    CORP_NOT_SAME_SITE = "corp-not-same-site"
    SRI_MESSAGE_SIGNATURE_MISMATCH = "sri-message-signature-mismatch"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CorsError(enum.Enum):
    '''
    The reason why request was blocked.
    '''
    DISALLOWED_BY_MODE = "DisallowedByMode"
    INVALID_RESPONSE = "InvalidResponse"
    WILDCARD_ORIGIN_NOT_ALLOWED = "WildcardOriginNotAllowed"
    MISSING_ALLOW_ORIGIN_HEADER = "MissingAllowOriginHeader"
    MULTIPLE_ALLOW_ORIGIN_VALUES = "MultipleAllowOriginValues"
    INVALID_ALLOW_ORIGIN_VALUE = "InvalidAllowOriginValue"
    ALLOW_ORIGIN_MISMATCH = "AllowOriginMismatch"
    INVALID_ALLOW_CREDENTIALS = "InvalidAllowCredentials"
    CORS_DISABLED_SCHEME = "CorsDisabledScheme"
    PREFLIGHT_INVALID_STATUS = "PreflightInvalidStatus"
    PREFLIGHT_DISALLOWED_REDIRECT = "PreflightDisallowedRedirect"
    PREFLIGHT_WILDCARD_ORIGIN_NOT_ALLOWED = "PreflightWildcardOriginNotAllowed"
    PREFLIGHT_MISSING_ALLOW_ORIGIN_HEADER = "PreflightMissingAllowOriginHeader"
    PREFLIGHT_MULTIPLE_ALLOW_ORIGIN_VALUES = "PreflightMultipleAllowOriginValues"
    PREFLIGHT_INVALID_ALLOW_ORIGIN_VALUE = "PreflightInvalidAllowOriginValue"
    PREFLIGHT_ALLOW_ORIGIN_MISMATCH = "PreflightAllowOriginMismatch"
    PREFLIGHT_INVALID_ALLOW_CREDENTIALS = "PreflightInvalidAllowCredentials"
    PREFLIGHT_MISSING_ALLOW_EXTERNAL = "PreflightMissingAllowExternal"
    PREFLIGHT_INVALID_ALLOW_EXTERNAL = "PreflightInvalidAllowExternal"
    PREFLIGHT_MISSING_ALLOW_PRIVATE_NETWORK = "PreflightMissingAllowPrivateNetwork"
    PREFLIGHT_INVALID_ALLOW_PRIVATE_NETWORK = "PreflightInvalidAllowPrivateNetwork"
    INVALID_ALLOW_METHODS_PREFLIGHT_RESPONSE = "InvalidAllowMethodsPreflightResponse"
    INVALID_ALLOW_HEADERS_PREFLIGHT_RESPONSE = "InvalidAllowHeadersPreflightResponse"
    METHOD_DISALLOWED_BY_PREFLIGHT_RESPONSE = "MethodDisallowedByPreflightResponse"
    HEADER_DISALLOWED_BY_PREFLIGHT_RESPONSE = "HeaderDisallowedByPreflightResponse"
    REDIRECT_CONTAINS_CREDENTIALS = "RedirectContainsCredentials"
    INSECURE_PRIVATE_NETWORK = "InsecurePrivateNetwork"
    INVALID_PRIVATE_NETWORK_ACCESS = "InvalidPrivateNetworkAccess"
    UNEXPECTED_PRIVATE_NETWORK_ACCESS = "UnexpectedPrivateNetworkAccess"
    NO_CORS_REDIRECT_MODE_NOT_FOLLOW = "NoCorsRedirectModeNotFollow"
    PREFLIGHT_MISSING_PRIVATE_NETWORK_ACCESS_ID = "PreflightMissingPrivateNetworkAccessId"
    PREFLIGHT_MISSING_PRIVATE_NETWORK_ACCESS_NAME = "PreflightMissingPrivateNetworkAccessName"
    PRIVATE_NETWORK_ACCESS_PERMISSION_UNAVAILABLE = "PrivateNetworkAccessPermissionUnavailable"
    PRIVATE_NETWORK_ACCESS_PERMISSION_DENIED = "PrivateNetworkAccessPermissionDenied"
    LOCAL_NETWORK_ACCESS_PERMISSION_DENIED = "LocalNetworkAccessPermissionDenied"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CorsErrorStatus:
    cors_error: CorsError

    failed_parameter: str

    def to_json(self):
        json = dict()
        json['corsError'] = self.cors_error.to_json()
        json['failedParameter'] = self.failed_parameter
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cors_error=CorsError.from_json(json['corsError']),
            failed_parameter=str(json['failedParameter']),
        )


class ServiceWorkerResponseSource(enum.Enum):
    '''
    Source of serviceworker response.
    '''
    CACHE_STORAGE = "cache-storage"
    HTTP_CACHE = "http-cache"
    FALLBACK_CODE = "fallback-code"
    NETWORK = "network"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class TrustTokenParams:
    '''
    Determines what type of Trust Token operation is executed and
    depending on the type, some additional parameters. The values
    are specified in third_party/blink/renderer/core/fetch/trust_token.idl.
    '''
    operation: TrustTokenOperationType

    #: Only set for "token-redemption" operation and determine whether
    #: to request a fresh SRR or use a still valid cached SRR.
    refresh_policy: str

    #: Origins of issuers from whom to request tokens or redemption
    #: records.
    issuers: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['operation'] = self.operation.to_json()
        json['refreshPolicy'] = self.refresh_policy
        if self.issuers is not None:
            json['issuers'] = [i for i in self.issuers]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            operation=TrustTokenOperationType.from_json(json['operation']),
            refresh_policy=str(json['refreshPolicy']),
            issuers=[str(i) for i in json['issuers']] if 'issuers' in json else None,
        )


class TrustTokenOperationType(enum.Enum):
    ISSUANCE = "Issuance"
    REDEMPTION = "Redemption"
    SIGNING = "Signing"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AlternateProtocolUsage(enum.Enum):
    '''
    The reason why Chrome uses a specific transport protocol for HTTP semantics.
    '''
    ALTERNATIVE_JOB_WON_WITHOUT_RACE = "alternativeJobWonWithoutRace"
    ALTERNATIVE_JOB_WON_RACE = "alternativeJobWonRace"
    MAIN_JOB_WON_RACE = "mainJobWonRace"
    MAPPING_MISSING = "mappingMissing"
    BROKEN = "broken"
    DNS_ALPN_H3_JOB_WON_WITHOUT_RACE = "dnsAlpnH3JobWonWithoutRace"
    DNS_ALPN_H3_JOB_WON_RACE = "dnsAlpnH3JobWonRace"
    UNSPECIFIED_REASON = "unspecifiedReason"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ServiceWorkerRouterSource(enum.Enum):
    '''
    Source of service worker router.
    '''
    NETWORK = "network"
    CACHE = "cache"
    FETCH_EVENT = "fetch-event"
    RACE_NETWORK_AND_FETCH_HANDLER = "race-network-and-fetch-handler"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ServiceWorkerRouterInfo:
    #: ID of the rule matched. If there is a matched rule, this field will
    #: be set, otherwiser no value will be set.
    rule_id_matched: typing.Optional[int] = None

    #: The router source of the matched rule. If there is a matched rule, this
    #: field will be set, otherwise no value will be set.
    matched_source_type: typing.Optional[ServiceWorkerRouterSource] = None

    #: The actual router source used.
    actual_source_type: typing.Optional[ServiceWorkerRouterSource] = None

    def to_json(self):
        json = dict()
        if self.rule_id_matched is not None:
            json['ruleIdMatched'] = self.rule_id_matched
        if self.matched_source_type is not None:
            json['matchedSourceType'] = self.matched_source_type.to_json()
        if self.actual_source_type is not None:
            json['actualSourceType'] = self.actual_source_type.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            rule_id_matched=int(json['ruleIdMatched']) if 'ruleIdMatched' in json else None,
            matched_source_type=ServiceWorkerRouterSource.from_json(json['matchedSourceType']) if 'matchedSourceType' in json else None,
            actual_source_type=ServiceWorkerRouterSource.from_json(json['actualSourceType']) if 'actualSourceType' in json else None,
        )


@dataclass
class Response:
    '''
    HTTP response data.
    '''
    #: Response URL. This URL can be different from CachedResource.url in case of redirect.
    url: str

    #: HTTP response status code.
    status: int

    #: HTTP response status text.
    status_text: str

    #: HTTP response headers.
    headers: Headers

    #: Resource mimeType as determined by the browser.
    mime_type: str

    #: Resource charset as determined by the browser (if applicable).
    charset: str

    #: Specifies whether physical connection was actually reused for this request.
    connection_reused: bool

    #: Physical connection id that was actually used for this request.
    connection_id: float

    #: Total number of bytes received for this request so far.
    encoded_data_length: float

    #: Security state of the request resource.
    security_state: security.SecurityState

    #: HTTP response headers text. This has been replaced by the headers in Network.responseReceivedExtraInfo.
    headers_text: typing.Optional[str] = None

    #: Refined HTTP request headers that were actually transmitted over the network.
    request_headers: typing.Optional[Headers] = None

    #: HTTP request headers text. This has been replaced by the headers in Network.requestWillBeSentExtraInfo.
    request_headers_text: typing.Optional[str] = None

    #: Remote IP address.
    remote_ip_address: typing.Optional[str] = None

    #: Remote port.
    remote_port: typing.Optional[int] = None

    #: Specifies that the request was served from the disk cache.
    from_disk_cache: typing.Optional[bool] = None

    #: Specifies that the request was served from the ServiceWorker.
    from_service_worker: typing.Optional[bool] = None

    #: Specifies that the request was served from the prefetch cache.
    from_prefetch_cache: typing.Optional[bool] = None

    #: Specifies that the request was served from the prefetch cache.
    from_early_hints: typing.Optional[bool] = None

    #: Information about how ServiceWorker Static Router API was used. If this
    #: field is set with ``matchedSourceType`` field, a matching rule is found.
    #: If this field is set without ``matchedSource``, no matching rule is found.
    #: Otherwise, the API is not used.
    service_worker_router_info: typing.Optional[ServiceWorkerRouterInfo] = None

    #: Timing information for the given request.
    timing: typing.Optional[ResourceTiming] = None

    #: Response source of response from ServiceWorker.
    service_worker_response_source: typing.Optional[ServiceWorkerResponseSource] = None

    #: The time at which the returned response was generated.
    response_time: typing.Optional[TimeSinceEpoch] = None

    #: Cache Storage Cache Name.
    cache_storage_cache_name: typing.Optional[str] = None

    #: Protocol used to fetch this request.
    protocol: typing.Optional[str] = None

    #: The reason why Chrome uses a specific transport protocol for HTTP semantics.
    alternate_protocol_usage: typing.Optional[AlternateProtocolUsage] = None

    #: Security details for the request.
    security_details: typing.Optional[SecurityDetails] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['status'] = self.status
        json['statusText'] = self.status_text
        json['headers'] = self.headers.to_json()
        json['mimeType'] = self.mime_type
        json['charset'] = self.charset
        json['connectionReused'] = self.connection_reused
        json['connectionId'] = self.connection_id
        json['encodedDataLength'] = self.encoded_data_length
        json['securityState'] = self.security_state.to_json()
        if self.headers_text is not None:
            json['headersText'] = self.headers_text
        if self.request_headers is not None:
            json['requestHeaders'] = self.request_headers.to_json()
        if self.request_headers_text is not None:
            json['requestHeadersText'] = self.request_headers_text
        if self.remote_ip_address is not None:
            json['remoteIPAddress'] = self.remote_ip_address
        if self.remote_port is not None:
            json['remotePort'] = self.remote_port
        if self.from_disk_cache is not None:
            json['fromDiskCache'] = self.from_disk_cache
        if self.from_service_worker is not None:
            json['fromServiceWorker'] = self.from_service_worker
        if self.from_prefetch_cache is not None:
            json['fromPrefetchCache'] = self.from_prefetch_cache
        if self.from_early_hints is not None:
            json['fromEarlyHints'] = self.from_early_hints
        if self.service_worker_router_info is not None:
            json['serviceWorkerRouterInfo'] = self.service_worker_router_info.to_json()
        if self.timing is not None:
            json['timing'] = self.timing.to_json()
        if self.service_worker_response_source is not None:
            json['serviceWorkerResponseSource'] = self.service_worker_response_source.to_json()
        if self.response_time is not None:
            json['responseTime'] = self.response_time.to_json()
        if self.cache_storage_cache_name is not None:
            json['cacheStorageCacheName'] = self.cache_storage_cache_name
        if self.protocol is not None:
            json['protocol'] = self.protocol
        if self.alternate_protocol_usage is not None:
            json['alternateProtocolUsage'] = self.alternate_protocol_usage.to_json()
        if self.security_details is not None:
            json['securityDetails'] = self.security_details.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            status=int(json['status']),
            status_text=str(json['statusText']),
            headers=Headers.from_json(json['headers']),
            mime_type=str(json['mimeType']),
            charset=str(json['charset']),
            connection_reused=bool(json['connectionReused']),
            connection_id=float(json['connectionId']),
            encoded_data_length=float(json['encodedDataLength']),
            security_state=security.SecurityState.from_json(json['securityState']),
            headers_text=str(json['headersText']) if 'headersText' in json else None,
            request_headers=Headers.from_json(json['requestHeaders']) if 'requestHeaders' in json else None,
            request_headers_text=str(json['requestHeadersText']) if 'requestHeadersText' in json else None,
            remote_ip_address=str(json['remoteIPAddress']) if 'remoteIPAddress' in json else None,
            remote_port=int(json['remotePort']) if 'remotePort' in json else None,
            from_disk_cache=bool(json['fromDiskCache']) if 'fromDiskCache' in json else None,
            from_service_worker=bool(json['fromServiceWorker']) if 'fromServiceWorker' in json else None,
            from_prefetch_cache=bool(json['fromPrefetchCache']) if 'fromPrefetchCache' in json else None,
            from_early_hints=bool(json['fromEarlyHints']) if 'fromEarlyHints' in json else None,
            service_worker_router_info=ServiceWorkerRouterInfo.from_json(json['serviceWorkerRouterInfo']) if 'serviceWorkerRouterInfo' in json else None,
            timing=ResourceTiming.from_json(json['timing']) if 'timing' in json else None,
            service_worker_response_source=ServiceWorkerResponseSource.from_json(json['serviceWorkerResponseSource']) if 'serviceWorkerResponseSource' in json else None,
            response_time=TimeSinceEpoch.from_json(json['responseTime']) if 'responseTime' in json else None,
            cache_storage_cache_name=str(json['cacheStorageCacheName']) if 'cacheStorageCacheName' in json else None,
            protocol=str(json['protocol']) if 'protocol' in json else None,
            alternate_protocol_usage=AlternateProtocolUsage.from_json(json['alternateProtocolUsage']) if 'alternateProtocolUsage' in json else None,
            security_details=SecurityDetails.from_json(json['securityDetails']) if 'securityDetails' in json else None,
        )


@dataclass
class WebSocketRequest:
    '''
    WebSocket request data.
    '''
    #: HTTP request headers.
    headers: Headers

    def to_json(self):
        json = dict()
        json['headers'] = self.headers.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            headers=Headers.from_json(json['headers']),
        )


@dataclass
class WebSocketResponse:
    '''
    WebSocket response data.
    '''
    #: HTTP response status code.
    status: int

    #: HTTP response status text.
    status_text: str

    #: HTTP response headers.
    headers: Headers

    #: HTTP response headers text.
    headers_text: typing.Optional[str] = None

    #: HTTP request headers.
    request_headers: typing.Optional[Headers] = None

    #: HTTP request headers text.
    request_headers_text: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['status'] = self.status
        json['statusText'] = self.status_text
        json['headers'] = self.headers.to_json()
        if self.headers_text is not None:
            json['headersText'] = self.headers_text
        if self.request_headers is not None:
            json['requestHeaders'] = self.request_headers.to_json()
        if self.request_headers_text is not None:
            json['requestHeadersText'] = self.request_headers_text
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            status=int(json['status']),
            status_text=str(json['statusText']),
            headers=Headers.from_json(json['headers']),
            headers_text=str(json['headersText']) if 'headersText' in json else None,
            request_headers=Headers.from_json(json['requestHeaders']) if 'requestHeaders' in json else None,
            request_headers_text=str(json['requestHeadersText']) if 'requestHeadersText' in json else None,
        )


@dataclass
class WebSocketFrame:
    '''
    WebSocket message data. This represents an entire WebSocket message, not just a fragmented frame as the name suggests.
    '''
    #: WebSocket message opcode.
    opcode: float

    #: WebSocket message mask.
    mask: bool

    #: WebSocket message payload data.
    #: If the opcode is 1, this is a text message and payloadData is a UTF-8 string.
    #: If the opcode isn't 1, then payloadData is a base64 encoded string representing binary data.
    payload_data: str

    def to_json(self):
        json = dict()
        json['opcode'] = self.opcode
        json['mask'] = self.mask
        json['payloadData'] = self.payload_data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            opcode=float(json['opcode']),
            mask=bool(json['mask']),
            payload_data=str(json['payloadData']),
        )


@dataclass
class CachedResource:
    '''
    Information about the cached resource.
    '''
    #: Resource URL. This is the url of the original network request.
    url: str

    #: Type of this resource.
    type_: ResourceType

    #: Cached response body size.
    body_size: float

    #: Cached response data.
    response: typing.Optional[Response] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['type'] = self.type_.to_json()
        json['bodySize'] = self.body_size
        if self.response is not None:
            json['response'] = self.response.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            type_=ResourceType.from_json(json['type']),
            body_size=float(json['bodySize']),
            response=Response.from_json(json['response']) if 'response' in json else None,
        )


@dataclass
class Initiator:
    '''
    Information about the request initiator.
    '''
    #: Type of this initiator.
    type_: str

    #: Initiator JavaScript stack trace, set for Script only.
    #: Requires the Debugger domain to be enabled.
    stack: typing.Optional[runtime.StackTrace] = None

    #: Initiator URL, set for Parser type or for Script type (when script is importing module) or for SignedExchange type.
    url: typing.Optional[str] = None

    #: Initiator line number, set for Parser type or for Script type (when script is importing
    #: module) (0-based).
    line_number: typing.Optional[float] = None

    #: Initiator column number, set for Parser type or for Script type (when script is importing
    #: module) (0-based).
    column_number: typing.Optional[float] = None

    #: Set if another request triggered this request (e.g. preflight).
    request_id: typing.Optional[RequestId] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.stack is not None:
            json['stack'] = self.stack.to_json()
        if self.url is not None:
            json['url'] = self.url
        if self.line_number is not None:
            json['lineNumber'] = self.line_number
        if self.column_number is not None:
            json['columnNumber'] = self.column_number
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            stack=runtime.StackTrace.from_json(json['stack']) if 'stack' in json else None,
            url=str(json['url']) if 'url' in json else None,
            line_number=float(json['lineNumber']) if 'lineNumber' in json else None,
            column_number=float(json['columnNumber']) if 'columnNumber' in json else None,
            request_id=RequestId.from_json(json['requestId']) if 'requestId' in json else None,
        )


@dataclass
class CookiePartitionKey:
    '''
    cookiePartitionKey object
    The representation of the components of the key that are created by the cookiePartitionKey class contained in net/cookies/cookie_partition_key.h.
    '''
    #: The site of the top-level URL the browser was visiting at the start
    #: of the request to the endpoint that set the cookie.
    top_level_site: str

    #: Indicates if the cookie has any ancestors that are cross-site to the topLevelSite.
    has_cross_site_ancestor: bool

    def to_json(self):
        json = dict()
        json['topLevelSite'] = self.top_level_site
        json['hasCrossSiteAncestor'] = self.has_cross_site_ancestor
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            top_level_site=str(json['topLevelSite']),
            has_cross_site_ancestor=bool(json['hasCrossSiteAncestor']),
        )


@dataclass
class Cookie:
    '''
    Cookie object
    '''
    #: Cookie name.
    name: str

    #: Cookie value.
    value: str

    #: Cookie domain.
    domain: str

    #: Cookie path.
    path: str

    #: Cookie expiration date as the number of seconds since the UNIX epoch.
    expires: float

    #: Cookie size.
    size: int

    #: True if cookie is http-only.
    http_only: bool

    #: True if cookie is secure.
    secure: bool

    #: True in case of session cookie.
    session: bool

    #: Cookie Priority
    priority: CookiePriority

    #: True if cookie is SameParty.
    same_party: bool

    #: Cookie source scheme type.
    source_scheme: CookieSourceScheme

    #: Cookie source port. Valid values are {-1, [1, 65535]}, -1 indicates an unspecified port.
    #: An unspecified port value allows protocol clients to emulate legacy cookie scope for the port.
    #: This is a temporary ability and it will be removed in the future.
    source_port: int

    #: Cookie SameSite type.
    same_site: typing.Optional[CookieSameSite] = None

    #: Cookie partition key.
    partition_key: typing.Optional[CookiePartitionKey] = None

    #: True if cookie partition key is opaque.
    partition_key_opaque: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        json['domain'] = self.domain
        json['path'] = self.path
        json['expires'] = self.expires
        json['size'] = self.size
        json['httpOnly'] = self.http_only
        json['secure'] = self.secure
        json['session'] = self.session
        json['priority'] = self.priority.to_json()
        json['sameParty'] = self.same_party
        json['sourceScheme'] = self.source_scheme.to_json()
        json['sourcePort'] = self.source_port
        if self.same_site is not None:
            json['sameSite'] = self.same_site.to_json()
        if self.partition_key is not None:
            json['partitionKey'] = self.partition_key.to_json()
        if self.partition_key_opaque is not None:
            json['partitionKeyOpaque'] = self.partition_key_opaque
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
            domain=str(json['domain']),
            path=str(json['path']),
            expires=float(json['expires']),
            size=int(json['size']),
            http_only=bool(json['httpOnly']),
            secure=bool(json['secure']),
            session=bool(json['session']),
            priority=CookiePriority.from_json(json['priority']),
            same_party=bool(json['sameParty']),
            source_scheme=CookieSourceScheme.from_json(json['sourceScheme']),
            source_port=int(json['sourcePort']),
            same_site=CookieSameSite.from_json(json['sameSite']) if 'sameSite' in json else None,
            partition_key=CookiePartitionKey.from_json(json['partitionKey']) if 'partitionKey' in json else None,
            partition_key_opaque=bool(json['partitionKeyOpaque']) if 'partitionKeyOpaque' in json else None,
        )


class SetCookieBlockedReason(enum.Enum):
    '''
    Types of reasons why a cookie may not be stored from a response.
    '''
    SECURE_ONLY = "SecureOnly"
    SAME_SITE_STRICT = "SameSiteStrict"
    SAME_SITE_LAX = "SameSiteLax"
    SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = "SameSiteUnspecifiedTreatedAsLax"
    SAME_SITE_NONE_INSECURE = "SameSiteNoneInsecure"
    USER_PREFERENCES = "UserPreferences"
    THIRD_PARTY_PHASEOUT = "ThirdPartyPhaseout"
    THIRD_PARTY_BLOCKED_IN_FIRST_PARTY_SET = "ThirdPartyBlockedInFirstPartySet"
    SYNTAX_ERROR = "SyntaxError"
    SCHEME_NOT_SUPPORTED = "SchemeNotSupported"
    OVERWRITE_SECURE = "OverwriteSecure"
    INVALID_DOMAIN = "InvalidDomain"
    INVALID_PREFIX = "InvalidPrefix"
    UNKNOWN_ERROR = "UnknownError"
    SCHEMEFUL_SAME_SITE_STRICT = "SchemefulSameSiteStrict"
    SCHEMEFUL_SAME_SITE_LAX = "SchemefulSameSiteLax"
    SCHEMEFUL_SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = "SchemefulSameSiteUnspecifiedTreatedAsLax"
    SAME_PARTY_FROM_CROSS_PARTY_CONTEXT = "SamePartyFromCrossPartyContext"
    SAME_PARTY_CONFLICTS_WITH_OTHER_ATTRIBUTES = "SamePartyConflictsWithOtherAttributes"
    NAME_VALUE_PAIR_EXCEEDS_MAX_SIZE = "NameValuePairExceedsMaxSize"
    DISALLOWED_CHARACTER = "DisallowedCharacter"
    NO_COOKIE_CONTENT = "NoCookieContent"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieBlockedReason(enum.Enum):
    '''
    Types of reasons why a cookie may not be sent with a request.
    '''
    SECURE_ONLY = "SecureOnly"
    NOT_ON_PATH = "NotOnPath"
    DOMAIN_MISMATCH = "DomainMismatch"
    SAME_SITE_STRICT = "SameSiteStrict"
    SAME_SITE_LAX = "SameSiteLax"
    SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = "SameSiteUnspecifiedTreatedAsLax"
    SAME_SITE_NONE_INSECURE = "SameSiteNoneInsecure"
    USER_PREFERENCES = "UserPreferences"
    THIRD_PARTY_PHASEOUT = "ThirdPartyPhaseout"
    THIRD_PARTY_BLOCKED_IN_FIRST_PARTY_SET = "ThirdPartyBlockedInFirstPartySet"
    UNKNOWN_ERROR = "UnknownError"
    SCHEMEFUL_SAME_SITE_STRICT = "SchemefulSameSiteStrict"
    SCHEMEFUL_SAME_SITE_LAX = "SchemefulSameSiteLax"
    SCHEMEFUL_SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = "SchemefulSameSiteUnspecifiedTreatedAsLax"
    SAME_PARTY_FROM_CROSS_PARTY_CONTEXT = "SamePartyFromCrossPartyContext"
    NAME_VALUE_PAIR_EXCEEDS_MAX_SIZE = "NameValuePairExceedsMaxSize"
    PORT_MISMATCH = "PortMismatch"
    SCHEME_MISMATCH = "SchemeMismatch"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieExemptionReason(enum.Enum):
    '''
    Types of reasons why a cookie should have been blocked by 3PCD but is exempted for the request.
    '''
    NONE = "None"
    USER_SETTING = "UserSetting"
    TPCD_METADATA = "TPCDMetadata"
    TPCD_DEPRECATION_TRIAL = "TPCDDeprecationTrial"
    TOP_LEVEL_TPCD_DEPRECATION_TRIAL = "TopLevelTPCDDeprecationTrial"
    TPCD_HEURISTICS = "TPCDHeuristics"
    ENTERPRISE_POLICY = "EnterprisePolicy"
    STORAGE_ACCESS = "StorageAccess"
    TOP_LEVEL_STORAGE_ACCESS = "TopLevelStorageAccess"
    SCHEME = "Scheme"
    SAME_SITE_NONE_COOKIES_IN_SANDBOX = "SameSiteNoneCookiesInSandbox"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class BlockedSetCookieWithReason:
    '''
    A cookie which was not stored from a response with the corresponding reason.
    '''
    #: The reason(s) this cookie was blocked.
    blocked_reasons: typing.List[SetCookieBlockedReason]

    #: The string representing this individual cookie as it would appear in the header.
    #: This is not the entire "cookie" or "set-cookie" header which could have multiple cookies.
    cookie_line: str

    #: The cookie object which represents the cookie which was not stored. It is optional because
    #: sometimes complete cookie information is not available, such as in the case of parsing
    #: errors.
    cookie: typing.Optional[Cookie] = None

    def to_json(self):
        json = dict()
        json['blockedReasons'] = [i.to_json() for i in self.blocked_reasons]
        json['cookieLine'] = self.cookie_line
        if self.cookie is not None:
            json['cookie'] = self.cookie.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            blocked_reasons=[SetCookieBlockedReason.from_json(i) for i in json['blockedReasons']],
            cookie_line=str(json['cookieLine']),
            cookie=Cookie.from_json(json['cookie']) if 'cookie' in json else None,
        )


@dataclass
class ExemptedSetCookieWithReason:
    '''
    A cookie should have been blocked by 3PCD but is exempted and stored from a response with the
    corresponding reason. A cookie could only have at most one exemption reason.
    '''
    #: The reason the cookie was exempted.
    exemption_reason: CookieExemptionReason

    #: The string representing this individual cookie as it would appear in the header.
    cookie_line: str

    #: The cookie object representing the cookie.
    cookie: Cookie

    def to_json(self):
        json = dict()
        json['exemptionReason'] = self.exemption_reason.to_json()
        json['cookieLine'] = self.cookie_line
        json['cookie'] = self.cookie.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            exemption_reason=CookieExemptionReason.from_json(json['exemptionReason']),
            cookie_line=str(json['cookieLine']),
            cookie=Cookie.from_json(json['cookie']),
        )


@dataclass
class AssociatedCookie:
    '''
    A cookie associated with the request which may or may not be sent with it.
    Includes the cookies itself and reasons for blocking or exemption.
    '''
    #: The cookie object representing the cookie which was not sent.
    cookie: Cookie

    #: The reason(s) the cookie was blocked. If empty means the cookie is included.
    blocked_reasons: typing.List[CookieBlockedReason]

    #: The reason the cookie should have been blocked by 3PCD but is exempted. A cookie could
    #: only have at most one exemption reason.
    exemption_reason: typing.Optional[CookieExemptionReason] = None

    def to_json(self):
        json = dict()
        json['cookie'] = self.cookie.to_json()
        json['blockedReasons'] = [i.to_json() for i in self.blocked_reasons]
        if self.exemption_reason is not None:
            json['exemptionReason'] = self.exemption_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cookie=Cookie.from_json(json['cookie']),
            blocked_reasons=[CookieBlockedReason.from_json(i) for i in json['blockedReasons']],
            exemption_reason=CookieExemptionReason.from_json(json['exemptionReason']) if 'exemptionReason' in json else None,
        )


@dataclass
class CookieParam:
    '''
    Cookie parameter object
    '''
    #: Cookie name.
    name: str

    #: Cookie value.
    value: str

    #: The request-URI to associate with the setting of the cookie. This value can affect the
    #: default domain, path, source port, and source scheme values of the created cookie.
    url: typing.Optional[str] = None

    #: Cookie domain.
    domain: typing.Optional[str] = None

    #: Cookie path.
    path: typing.Optional[str] = None

    #: True if cookie is secure.
    secure: typing.Optional[bool] = None

    #: True if cookie is http-only.
    http_only: typing.Optional[bool] = None

    #: Cookie SameSite type.
    same_site: typing.Optional[CookieSameSite] = None

    #: Cookie expiration date, session cookie if not set
    expires: typing.Optional[TimeSinceEpoch] = None

    #: Cookie Priority.
    priority: typing.Optional[CookiePriority] = None

    #: True if cookie is SameParty.
    same_party: typing.Optional[bool] = None

    #: Cookie source scheme type.
    source_scheme: typing.Optional[CookieSourceScheme] = None

    #: Cookie source port. Valid values are {-1, [1, 65535]}, -1 indicates an unspecified port.
    #: An unspecified port value allows protocol clients to emulate legacy cookie scope for the port.
    #: This is a temporary ability and it will be removed in the future.
    source_port: typing.Optional[int] = None

    #: Cookie partition key. If not set, the cookie will be set as not partitioned.
    partition_key: typing.Optional[CookiePartitionKey] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        if self.url is not None:
            json['url'] = self.url
        if self.domain is not None:
            json['domain'] = self.domain
        if self.path is not None:
            json['path'] = self.path
        if self.secure is not None:
            json['secure'] = self.secure
        if self.http_only is not None:
            json['httpOnly'] = self.http_only
        if self.same_site is not None:
            json['sameSite'] = self.same_site.to_json()
        if self.expires is not None:
            json['expires'] = self.expires.to_json()
        if self.priority is not None:
            json['priority'] = self.priority.to_json()
        if self.same_party is not None:
            json['sameParty'] = self.same_party
        if self.source_scheme is not None:
            json['sourceScheme'] = self.source_scheme.to_json()
        if self.source_port is not None:
            json['sourcePort'] = self.source_port
        if self.partition_key is not None:
            json['partitionKey'] = self.partition_key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
            url=str(json['url']) if 'url' in json else None,
            domain=str(json['domain']) if 'domain' in json else None,
            path=str(json['path']) if 'path' in json else None,
            secure=bool(json['secure']) if 'secure' in json else None,
            http_only=bool(json['httpOnly']) if 'httpOnly' in json else None,
            same_site=CookieSameSite.from_json(json['sameSite']) if 'sameSite' in json else None,
            expires=TimeSinceEpoch.from_json(json['expires']) if 'expires' in json else None,
            priority=CookiePriority.from_json(json['priority']) if 'priority' in json else None,
            same_party=bool(json['sameParty']) if 'sameParty' in json else None,
            source_scheme=CookieSourceScheme.from_json(json['sourceScheme']) if 'sourceScheme' in json else None,
            source_port=int(json['sourcePort']) if 'sourcePort' in json else None,
            partition_key=CookiePartitionKey.from_json(json['partitionKey']) if 'partitionKey' in json else None,
        )


@dataclass
class AuthChallenge:
    '''
    Authorization challenge for HTTP status code 401 or 407.
    '''
    #: Origin of the challenger.
    origin: str

    #: The authentication scheme used, such as basic or digest
    scheme: str

    #: The realm of the challenge. May be empty.
    realm: str

    #: Source of the authentication challenge.
    source: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['origin'] = self.origin
        json['scheme'] = self.scheme
        json['realm'] = self.realm
        if self.source is not None:
            json['source'] = self.source
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            origin=str(json['origin']),
            scheme=str(json['scheme']),
            realm=str(json['realm']),
            source=str(json['source']) if 'source' in json else None,
        )


@dataclass
class AuthChallengeResponse:
    '''
    Response to an AuthChallenge.
    '''
    #: The decision on what to do in response to the authorization challenge.  Default means
    #: deferring to the default behavior of the net stack, which will likely either the Cancel
    #: authentication or display a popup dialog box.
    response: str

    #: The username to provide, possibly empty. Should only be set if response is
    #: ProvideCredentials.
    username: typing.Optional[str] = None

    #: The password to provide, possibly empty. Should only be set if response is
    #: ProvideCredentials.
    password: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['response'] = self.response
        if self.username is not None:
            json['username'] = self.username
        if self.password is not None:
            json['password'] = self.password
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            response=str(json['response']),
            username=str(json['username']) if 'username' in json else None,
            password=str(json['password']) if 'password' in json else None,
        )


class InterceptionStage(enum.Enum):
    '''
    Stages of the interception to begin intercepting. Request will intercept before the request is
    sent. Response will intercept after the response is received.
    '''
    REQUEST = "Request"
    HEADERS_RECEIVED = "HeadersReceived"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class RequestPattern:
    '''
    Request pattern for interception.
    '''
    #: Wildcards (``'*'`` -> zero or more, ``'?'`` -> exactly one) are allowed. Escape character is
    #: backslash. Omitting is equivalent to ``"*"``.
    url_pattern: typing.Optional[str] = None

    #: If set, only requests for matching resource types will be intercepted.
    resource_type: typing.Optional[ResourceType] = None

    #: Stage at which to begin intercepting requests. Default is Request.
    interception_stage: typing.Optional[InterceptionStage] = None

    def to_json(self):
        json = dict()
        if self.url_pattern is not None:
            json['urlPattern'] = self.url_pattern
        if self.resource_type is not None:
            json['resourceType'] = self.resource_type.to_json()
        if self.interception_stage is not None:
            json['interceptionStage'] = self.interception_stage.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url_pattern=str(json['urlPattern']) if 'urlPattern' in json else None,
            resource_type=ResourceType.from_json(json['resourceType']) if 'resourceType' in json else None,
            interception_stage=InterceptionStage.from_json(json['interceptionStage']) if 'interceptionStage' in json else None,
        )


@dataclass
class SignedExchangeSignature:
    '''
    Information about a signed exchange signature.
    https://wicg.github.io/webpackage/draft-yasskin-httpbis-origin-signed-exchanges-impl.html#rfc.section.3.1
    '''
    #: Signed exchange signature label.
    label: str

    #: The hex string of signed exchange signature.
    signature: str

    #: Signed exchange signature integrity.
    integrity: str

    #: Signed exchange signature validity Url.
    validity_url: str

    #: Signed exchange signature date.
    date: int

    #: Signed exchange signature expires.
    expires: int

    #: Signed exchange signature cert Url.
    cert_url: typing.Optional[str] = None

    #: The hex string of signed exchange signature cert sha256.
    cert_sha256: typing.Optional[str] = None

    #: The encoded certificates.
    certificates: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['label'] = self.label
        json['signature'] = self.signature
        json['integrity'] = self.integrity
        json['validityUrl'] = self.validity_url
        json['date'] = self.date
        json['expires'] = self.expires
        if self.cert_url is not None:
            json['certUrl'] = self.cert_url
        if self.cert_sha256 is not None:
            json['certSha256'] = self.cert_sha256
        if self.certificates is not None:
            json['certificates'] = [i for i in self.certificates]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            label=str(json['label']),
            signature=str(json['signature']),
            integrity=str(json['integrity']),
            validity_url=str(json['validityUrl']),
            date=int(json['date']),
            expires=int(json['expires']),
            cert_url=str(json['certUrl']) if 'certUrl' in json else None,
            cert_sha256=str(json['certSha256']) if 'certSha256' in json else None,
            certificates=[str(i) for i in json['certificates']] if 'certificates' in json else None,
        )


@dataclass
class SignedExchangeHeader:
    '''
    Information about a signed exchange header.
    https://wicg.github.io/webpackage/draft-yasskin-httpbis-origin-signed-exchanges-impl.html#cbor-representation
    '''
    #: Signed exchange request URL.
    request_url: str

    #: Signed exchange response code.
    response_code: int

    #: Signed exchange response headers.
    response_headers: Headers

    #: Signed exchange response signature.
    signatures: typing.List[SignedExchangeSignature]

    #: Signed exchange header integrity hash in the form of ``sha256-<base64-hash-value>``.
    header_integrity: str

    def to_json(self):
        json = dict()
        json['requestUrl'] = self.request_url
        json['responseCode'] = self.response_code
        json['responseHeaders'] = self.response_headers.to_json()
        json['signatures'] = [i.to_json() for i in self.signatures]
        json['headerIntegrity'] = self.header_integrity
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request_url=str(json['requestUrl']),
            response_code=int(json['responseCode']),
            response_headers=Headers.from_json(json['responseHeaders']),
            signatures=[SignedExchangeSignature.from_json(i) for i in json['signatures']],
            header_integrity=str(json['headerIntegrity']),
        )


class SignedExchangeErrorField(enum.Enum):
    '''
    Field type for a signed exchange related error.
    '''
    SIGNATURE_SIG = "signatureSig"
    SIGNATURE_INTEGRITY = "signatureIntegrity"
    SIGNATURE_CERT_URL = "signatureCertUrl"
    SIGNATURE_CERT_SHA256 = "signatureCertSha256"
    SIGNATURE_VALIDITY_URL = "signatureValidityUrl"
    SIGNATURE_TIMESTAMPS = "signatureTimestamps"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SignedExchangeError:
    '''
    Information about a signed exchange response.
    '''
    #: Error message.
    message: str

    #: The index of the signature which caused the error.
    signature_index: typing.Optional[int] = None

    #: The field which caused the error.
    error_field: typing.Optional[SignedExchangeErrorField] = None

    def to_json(self):
        json = dict()
        json['message'] = self.message
        if self.signature_index is not None:
            json['signatureIndex'] = self.signature_index
        if self.error_field is not None:
            json['errorField'] = self.error_field.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            message=str(json['message']),
            signature_index=int(json['signatureIndex']) if 'signatureIndex' in json else None,
            error_field=SignedExchangeErrorField.from_json(json['errorField']) if 'errorField' in json else None,
        )


@dataclass
class SignedExchangeInfo:
    '''
    Information about a signed exchange response.
    '''
    #: The outer response of signed HTTP exchange which was received from network.
    outer_response: Response

    #: Information about the signed exchange header.
    header: typing.Optional[SignedExchangeHeader] = None

    #: Security details for the signed exchange header.
    security_details: typing.Optional[SecurityDetails] = None

    #: Errors occurred while handling the signed exchange.
    errors: typing.Optional[typing.List[SignedExchangeError]] = None

    def to_json(self):
        json = dict()
        json['outerResponse'] = self.outer_response.to_json()
        if self.header is not None:
            json['header'] = self.header.to_json()
        if self.security_details is not None:
            json['securityDetails'] = self.security_details.to_json()
        if self.errors is not None:
            json['errors'] = [i.to_json() for i in self.errors]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            outer_response=Response.from_json(json['outerResponse']),
            header=SignedExchangeHeader.from_json(json['header']) if 'header' in json else None,
            security_details=SecurityDetails.from_json(json['securityDetails']) if 'securityDetails' in json else None,
            errors=[SignedExchangeError.from_json(i) for i in json['errors']] if 'errors' in json else None,
        )


class ContentEncoding(enum.Enum):
    '''
    List of content encodings supported by the backend.
    '''
    DEFLATE = "deflate"
    GZIP = "gzip"
    BR = "br"
    ZSTD = "zstd"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PrivateNetworkRequestPolicy(enum.Enum):
    ALLOW = "Allow"
    BLOCK_FROM_INSECURE_TO_MORE_PRIVATE = "BlockFromInsecureToMorePrivate"
    WARN_FROM_INSECURE_TO_MORE_PRIVATE = "WarnFromInsecureToMorePrivate"
    PREFLIGHT_BLOCK = "PreflightBlock"
    PREFLIGHT_WARN = "PreflightWarn"
    PERMISSION_BLOCK = "PermissionBlock"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class IPAddressSpace(enum.Enum):
    LOCAL = "Local"
    PRIVATE = "Private"
    PUBLIC = "Public"
    UNKNOWN = "Unknown"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ConnectTiming:
    #: Timing's requestTime is a baseline in seconds, while the other numbers are ticks in
    #: milliseconds relatively to this requestTime. Matches ResourceTiming's requestTime for
    #: the same request (but not for redirected requests).
    request_time: float

    def to_json(self):
        json = dict()
        json['requestTime'] = self.request_time
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request_time=float(json['requestTime']),
        )


@dataclass
class ClientSecurityState:
    initiator_is_secure_context: bool

    initiator_ip_address_space: IPAddressSpace

    private_network_request_policy: PrivateNetworkRequestPolicy

    def to_json(self):
        json = dict()
        json['initiatorIsSecureContext'] = self.initiator_is_secure_context
        json['initiatorIPAddressSpace'] = self.initiator_ip_address_space.to_json()
        json['privateNetworkRequestPolicy'] = self.private_network_request_policy.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            initiator_is_secure_context=bool(json['initiatorIsSecureContext']),
            initiator_ip_address_space=IPAddressSpace.from_json(json['initiatorIPAddressSpace']),
            private_network_request_policy=PrivateNetworkRequestPolicy.from_json(json['privateNetworkRequestPolicy']),
        )


class CrossOriginOpenerPolicyValue(enum.Enum):
    SAME_ORIGIN = "SameOrigin"
    SAME_ORIGIN_ALLOW_POPUPS = "SameOriginAllowPopups"
    RESTRICT_PROPERTIES = "RestrictProperties"
    UNSAFE_NONE = "UnsafeNone"
    SAME_ORIGIN_PLUS_COEP = "SameOriginPlusCoep"
    RESTRICT_PROPERTIES_PLUS_COEP = "RestrictPropertiesPlusCoep"
    NOOPENER_ALLOW_POPUPS = "NoopenerAllowPopups"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CrossOriginOpenerPolicyStatus:
    value: CrossOriginOpenerPolicyValue

    report_only_value: CrossOriginOpenerPolicyValue

    reporting_endpoint: typing.Optional[str] = None

    report_only_reporting_endpoint: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['value'] = self.value.to_json()
        json['reportOnlyValue'] = self.report_only_value.to_json()
        if self.reporting_endpoint is not None:
            json['reportingEndpoint'] = self.reporting_endpoint
        if self.report_only_reporting_endpoint is not None:
            json['reportOnlyReportingEndpoint'] = self.report_only_reporting_endpoint
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=CrossOriginOpenerPolicyValue.from_json(json['value']),
            report_only_value=CrossOriginOpenerPolicyValue.from_json(json['reportOnlyValue']),
            reporting_endpoint=str(json['reportingEndpoint']) if 'reportingEndpoint' in json else None,
            report_only_reporting_endpoint=str(json['reportOnlyReportingEndpoint']) if 'reportOnlyReportingEndpoint' in json else None,
        )


class CrossOriginEmbedderPolicyValue(enum.Enum):
    NONE = "None"
    CREDENTIALLESS = "Credentialless"
    REQUIRE_CORP = "RequireCorp"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CrossOriginEmbedderPolicyStatus:
    value: CrossOriginEmbedderPolicyValue

    report_only_value: CrossOriginEmbedderPolicyValue

    reporting_endpoint: typing.Optional[str] = None

    report_only_reporting_endpoint: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['value'] = self.value.to_json()
        json['reportOnlyValue'] = self.report_only_value.to_json()
        if self.reporting_endpoint is not None:
            json['reportingEndpoint'] = self.reporting_endpoint
        if self.report_only_reporting_endpoint is not None:
            json['reportOnlyReportingEndpoint'] = self.report_only_reporting_endpoint
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=CrossOriginEmbedderPolicyValue.from_json(json['value']),
            report_only_value=CrossOriginEmbedderPolicyValue.from_json(json['reportOnlyValue']),
            reporting_endpoint=str(json['reportingEndpoint']) if 'reportingEndpoint' in json else None,
            report_only_reporting_endpoint=str(json['reportOnlyReportingEndpoint']) if 'reportOnlyReportingEndpoint' in json else None,
        )


class ContentSecurityPolicySource(enum.Enum):
    HTTP = "HTTP"
    META = "Meta"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ContentSecurityPolicyStatus:
    effective_directives: str

    is_enforced: bool

    source: ContentSecurityPolicySource

    def to_json(self):
        json = dict()
        json['effectiveDirectives'] = self.effective_directives
        json['isEnforced'] = self.is_enforced
        json['source'] = self.source.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            effective_directives=str(json['effectiveDirectives']),
            is_enforced=bool(json['isEnforced']),
            source=ContentSecurityPolicySource.from_json(json['source']),
        )


@dataclass
class SecurityIsolationStatus:
    coop: typing.Optional[CrossOriginOpenerPolicyStatus] = None

    coep: typing.Optional[CrossOriginEmbedderPolicyStatus] = None

    csp: typing.Optional[typing.List[ContentSecurityPolicyStatus]] = None

    def to_json(self):
        json = dict()
        if self.coop is not None:
            json['coop'] = self.coop.to_json()
        if self.coep is not None:
            json['coep'] = self.coep.to_json()
        if self.csp is not None:
            json['csp'] = [i.to_json() for i in self.csp]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            coop=CrossOriginOpenerPolicyStatus.from_json(json['coop']) if 'coop' in json else None,
            coep=CrossOriginEmbedderPolicyStatus.from_json(json['coep']) if 'coep' in json else None,
            csp=[ContentSecurityPolicyStatus.from_json(i) for i in json['csp']] if 'csp' in json else None,
        )


class ReportStatus(enum.Enum):
    '''
    The status of a Reporting API report.
    '''
    QUEUED = "Queued"
    PENDING = "Pending"
    MARKED_FOR_REMOVAL = "MarkedForRemoval"
    SUCCESS = "Success"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ReportId(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ReportId:
        return cls(json)

    def __repr__(self):
        return 'ReportId({})'.format(super().__repr__())


@dataclass
class ReportingApiReport:
    '''
    An object representing a report generated by the Reporting API.
    '''
    id_: ReportId

    #: The URL of the document that triggered the report.
    initiator_url: str

    #: The name of the endpoint group that should be used to deliver the report.
    destination: str

    #: The type of the report (specifies the set of data that is contained in the report body).
    type_: str

    #: When the report was generated.
    timestamp: network.TimeSinceEpoch

    #: How many uploads deep the related request was.
    depth: int

    #: The number of delivery attempts made so far, not including an active attempt.
    completed_attempts: int

    body: dict

    status: ReportStatus

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['initiatorUrl'] = self.initiator_url
        json['destination'] = self.destination
        json['type'] = self.type_
        json['timestamp'] = self.timestamp.to_json()
        json['depth'] = self.depth
        json['completedAttempts'] = self.completed_attempts
        json['body'] = self.body
        json['status'] = self.status.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=ReportId.from_json(json['id']),
            initiator_url=str(json['initiatorUrl']),
            destination=str(json['destination']),
            type_=str(json['type']),
            timestamp=network.TimeSinceEpoch.from_json(json['timestamp']),
            depth=int(json['depth']),
            completed_attempts=int(json['completedAttempts']),
            body=dict(json['body']),
            status=ReportStatus.from_json(json['status']),
        )


@dataclass
class ReportingApiEndpoint:
    #: The URL of the endpoint to which reports may be delivered.
    url: str

    #: Name of the endpoint group.
    group_name: str

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['groupName'] = self.group_name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            group_name=str(json['groupName']),
        )


@dataclass
class LoadNetworkResourcePageResult:
    '''
    An object providing the result of a network resource load.
    '''
    success: bool

    #: Optional values used for error reporting.
    net_error: typing.Optional[float] = None

    net_error_name: typing.Optional[str] = None

    http_status_code: typing.Optional[float] = None

    #: If successful, one of the following two fields holds the result.
    stream: typing.Optional[io.StreamHandle] = None

    #: Response headers.
    headers: typing.Optional[network.Headers] = None

    def to_json(self):
        json = dict()
        json['success'] = self.success
        if self.net_error is not None:
            json['netError'] = self.net_error
        if self.net_error_name is not None:
            json['netErrorName'] = self.net_error_name
        if self.http_status_code is not None:
            json['httpStatusCode'] = self.http_status_code
        if self.stream is not None:
            json['stream'] = self.stream.to_json()
        if self.headers is not None:
            json['headers'] = self.headers.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            success=bool(json['success']),
            net_error=float(json['netError']) if 'netError' in json else None,
            net_error_name=str(json['netErrorName']) if 'netErrorName' in json else None,
            http_status_code=float(json['httpStatusCode']) if 'httpStatusCode' in json else None,
            stream=io.StreamHandle.from_json(json['stream']) if 'stream' in json else None,
            headers=network.Headers.from_json(json['headers']) if 'headers' in json else None,
        )


@dataclass
class LoadNetworkResourceOptions:
    '''
    An options object that may be extended later to better support CORS,
    CORB and streaming.
    '''
    disable_cache: bool

    include_credentials: bool

    def to_json(self):
        json = dict()
        json['disableCache'] = self.disable_cache
        json['includeCredentials'] = self.include_credentials
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            disable_cache=bool(json['disableCache']),
            include_credentials=bool(json['includeCredentials']),
        )


def set_accepted_encodings(
        encodings: typing.List[ContentEncoding]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets a list of content encodings that will be accepted. Empty list means no encoding is accepted.

    **EXPERIMENTAL**

    :param encodings: List of accepted content encodings.
    '''
    params: T_JSON_DICT = dict()
    params['encodings'] = [i.to_json() for i in encodings]
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setAcceptedEncodings',
        'params': params,
    }
    json = yield cmd_dict


def clear_accepted_encodings_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears accepted encodings set by setAcceptedEncodings

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.clearAcceptedEncodingsOverride',
    }
    json = yield cmd_dict


def can_clear_browser_cache() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Tells whether clearing browser cache is supported.

    :returns: True if browser cache can be cleared.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.canClearBrowserCache',
    }
    json = yield cmd_dict
    return bool(json['result'])


def can_clear_browser_cookies() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Tells whether clearing browser cookies is supported.

    :returns: True if browser cookies can be cleared.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.canClearBrowserCookies',
    }
    json = yield cmd_dict
    return bool(json['result'])


def can_emulate_network_conditions() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Tells whether emulation of network conditions is supported.

    :returns: True if emulation of network conditions is supported.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.canEmulateNetworkConditions',
    }
    json = yield cmd_dict
    return bool(json['result'])


def clear_browser_cache() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears browser cache.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.clearBrowserCache',
    }
    json = yield cmd_dict


def clear_browser_cookies() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears browser cookies.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.clearBrowserCookies',
    }
    json = yield cmd_dict


def continue_intercepted_request(
        interception_id: InterceptionId,
        error_reason: typing.Optional[ErrorReason] = None,
        raw_response: typing.Optional[str] = None,
        url: typing.Optional[str] = None,
        method: typing.Optional[str] = None,
        post_data: typing.Optional[str] = None,
        headers: typing.Optional[Headers] = None,
        auth_challenge_response: typing.Optional[AuthChallengeResponse] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Response to Network.requestIntercepted which either modifies the request to continue with any
    modifications, or blocks it, or completes it with the provided response bytes. If a network
    fetch occurs as a result which encounters a redirect an additional Network.requestIntercepted
    event will be sent with the same InterceptionId.
    Deprecated, use Fetch.continueRequest, Fetch.fulfillRequest and Fetch.failRequest instead.

    **EXPERIMENTAL**

    :param interception_id:
    :param error_reason: *(Optional)* If set this causes the request to fail with the given reason. Passing ```Aborted```` for requests marked with ````isNavigationRequest``` also cancels the navigation. Must not be set in response to an authChallenge.
    :param raw_response: *(Optional)* If set the requests completes using with the provided base64 encoded raw response, including HTTP status line and headers etc... Must not be set in response to an authChallenge.
    :param url: *(Optional)* If set the request url will be modified in a way that's not observable by page. Must not be set in response to an authChallenge.
    :param method: *(Optional)* If set this allows the request method to be overridden. Must not be set in response to an authChallenge.
    :param post_data: *(Optional)* If set this allows postData to be set. Must not be set in response to an authChallenge.
    :param headers: *(Optional)* If set this allows the request headers to be changed. Must not be set in response to an authChallenge.
    :param auth_challenge_response: *(Optional)* Response to a requestIntercepted with an authChallenge. Must not be set otherwise.
    '''
    params: T_JSON_DICT = dict()
    params['interceptionId'] = interception_id.to_json()
    if error_reason is not None:
        params['errorReason'] = error_reason.to_json()
    if raw_response is not None:
        params['rawResponse'] = raw_response
    if url is not None:
        params['url'] = url
    if method is not None:
        params['method'] = method
    if post_data is not None:
        params['postData'] = post_data
    if headers is not None:
        params['headers'] = headers.to_json()
    if auth_challenge_response is not None:
        params['authChallengeResponse'] = auth_challenge_response.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.continueInterceptedRequest',
        'params': params,
    }
    json = yield cmd_dict


def delete_cookies(
        name: str,
        url: typing.Optional[str] = None,
        domain: typing.Optional[str] = None,
        path: typing.Optional[str] = None,
        partition_key: typing.Optional[CookiePartitionKey] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes browser cookies with matching name and url or domain/path/partitionKey pair.

    :param name: Name of the cookies to remove.
    :param url: *(Optional)* If specified, deletes all the cookies with the given name where domain and path match provided URL.
    :param domain: *(Optional)* If specified, deletes only cookies with the exact domain.
    :param path: *(Optional)* If specified, deletes only cookies with the exact path.
    :param partition_key: **(EXPERIMENTAL)** *(Optional)* If specified, deletes only cookies with the the given name and partitionKey where all partition key attributes match the cookie partition key attribute.
    '''
    params: T_JSON_DICT = dict()
    params['name'] = name
    if url is not None:
        params['url'] = url
    if domain is not None:
        params['domain'] = domain
    if path is not None:
        params['path'] = path
    if partition_key is not None:
        params['partitionKey'] = partition_key.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.deleteCookies',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables network tracking, prevents network events from being sent to the client.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.disable',
    }
    json = yield cmd_dict


def emulate_network_conditions(
        offline: bool,
        latency: float,
        download_throughput: float,
        upload_throughput: float,
        connection_type: typing.Optional[ConnectionType] = None,
        packet_loss: typing.Optional[float] = None,
        packet_queue_length: typing.Optional[int] = None,
        packet_reordering: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Activates emulation of network conditions.

    :param offline: True to emulate internet disconnection.
    :param latency: Minimum latency from request sent to response headers received (ms).
    :param download_throughput: Maximal aggregated download throughput (bytes/sec). -1 disables download throttling.
    :param upload_throughput: Maximal aggregated upload throughput (bytes/sec).  -1 disables upload throttling.
    :param connection_type: *(Optional)* Connection type if known.
    :param packet_loss: **(EXPERIMENTAL)** *(Optional)* WebRTC packet loss (percent, 0-100). 0 disables packet loss emulation, 100 drops all the packets.
    :param packet_queue_length: **(EXPERIMENTAL)** *(Optional)* WebRTC packet queue length (packet). 0 removes any queue length limitations.
    :param packet_reordering: **(EXPERIMENTAL)** *(Optional)* WebRTC packetReordering feature.
    '''
    params: T_JSON_DICT = dict()
    params['offline'] = offline
    params['latency'] = latency
    params['downloadThroughput'] = download_throughput
    params['uploadThroughput'] = upload_throughput
    if connection_type is not None:
        params['connectionType'] = connection_type.to_json()
    if packet_loss is not None:
        params['packetLoss'] = packet_loss
    if packet_queue_length is not None:
        params['packetQueueLength'] = packet_queue_length
    if packet_reordering is not None:
        params['packetReordering'] = packet_reordering
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.emulateNetworkConditions',
        'params': params,
    }
    json = yield cmd_dict


def enable(
        max_total_buffer_size: typing.Optional[int] = None,
        max_resource_buffer_size: typing.Optional[int] = None,
        max_post_data_size: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables network tracking, network events will now be delivered to the client.

    :param max_total_buffer_size: **(EXPERIMENTAL)** *(Optional)* Buffer size in bytes to use when preserving network payloads (XHRs, etc).
    :param max_resource_buffer_size: **(EXPERIMENTAL)** *(Optional)* Per-resource buffer size in bytes to use when preserving network payloads (XHRs, etc).
    :param max_post_data_size: *(Optional)* Longest post body size (in bytes) that would be included in requestWillBeSent notification
    '''
    params: T_JSON_DICT = dict()
    if max_total_buffer_size is not None:
        params['maxTotalBufferSize'] = max_total_buffer_size
    if max_resource_buffer_size is not None:
        params['maxResourceBufferSize'] = max_resource_buffer_size
    if max_post_data_size is not None:
        params['maxPostDataSize'] = max_post_data_size
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.enable',
        'params': params,
    }
    json = yield cmd_dict


def get_all_cookies() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Cookie]]:
    '''
    Returns all browser cookies. Depending on the backend support, will return detailed cookie
    information in the ``cookies`` field.
    Deprecated. Use Storage.getCookies instead.

    :returns: Array of cookie objects.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getAllCookies',
    }
    json = yield cmd_dict
    return [Cookie.from_json(i) for i in json['cookies']]


def get_certificate(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Returns the DER-encoded certificate.

    **EXPERIMENTAL**

    :param origin: Origin to get certificate for.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getCertificate',
        'params': params,
    }
    json = yield cmd_dict
    return [str(i) for i in json['tableNames']]


def get_cookies(
        urls: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Cookie]]:
    '''
    Returns all browser cookies for the current URL. Depending on the backend support, will return
    detailed cookie information in the ``cookies`` field.

    :param urls: *(Optional)* The list of URLs for which applicable cookies will be fetched. If not specified, it's assumed to be set to the list containing the URLs of the page and all of its subframes.
    :returns: Array of cookie objects.
    '''
    params: T_JSON_DICT = dict()
    if urls is not None:
        params['urls'] = [i for i in urls]
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getCookies',
        'params': params,
    }
    json = yield cmd_dict
    return [Cookie.from_json(i) for i in json['cookies']]


def get_response_body(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Returns content served for the given request.

    :param request_id: Identifier of the network request to get content for.
    :returns: A tuple with the following items:

        0. **body** - Response body.
        1. **base64Encoded** - True, if content was sent as base64.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getResponseBody',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['body']),
        bool(json['base64Encoded'])
    )


def get_request_post_data(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Returns post data sent with the request. Returns an error when no data was sent with the request.

    :param request_id: Identifier of the network request to get content for.
    :returns: Request body string, omitting files from multipart requests
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getRequestPostData',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['postData'])


def get_response_body_for_interception(
        interception_id: InterceptionId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Returns content served for the given currently intercepted request.

    **EXPERIMENTAL**

    :param interception_id: Identifier for the intercepted request to get body for.
    :returns: A tuple with the following items:

        0. **body** - Response body.
        1. **base64Encoded** - True, if content was sent as base64.
    '''
    params: T_JSON_DICT = dict()
    params['interceptionId'] = interception_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getResponseBodyForInterception',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['body']),
        bool(json['base64Encoded'])
    )


def take_response_body_for_interception_as_stream(
        interception_id: InterceptionId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,io.StreamHandle]:
    '''
    Returns a handle to the stream representing the response body. Note that after this command,
    the intercepted request can't be continued as is -- you either need to cancel it or to provide
    the response body. The stream only supports sequential read, IO.read will fail if the position
    is specified.

    **EXPERIMENTAL**

    :param interception_id:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['interceptionId'] = interception_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.takeResponseBodyForInterceptionAsStream',
        'params': params,
    }
    json = yield cmd_dict
    return io.StreamHandle.from_json(json['stream'])


def replay_xhr(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    This method sends a new XMLHttpRequest which is identical to the original one. The following
    parameters should be identical: method, url, async, request body, extra headers, withCredentials
    attribute, user, password.

    **EXPERIMENTAL**

    :param request_id: Identifier of XHR to replay.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.replayXHR',
        'params': params,
    }
    json = yield cmd_dict


def search_in_response_body(
        request_id: RequestId,
        query: str,
        case_sensitive: typing.Optional[bool] = None,
        is_regex: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[debugger.SearchMatch]]:
    '''
    Searches for given string in response content.

    **EXPERIMENTAL**

    :param request_id: Identifier of the network response to search.
    :param query: String to search for.
    :param case_sensitive: *(Optional)* If true, search is case sensitive.
    :param is_regex: *(Optional)* If true, treats string parameter as regex.
    :returns: List of search matches.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['query'] = query
    if case_sensitive is not None:
        params['caseSensitive'] = case_sensitive
    if is_regex is not None:
        params['isRegex'] = is_regex
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.searchInResponseBody',
        'params': params,
    }
    json = yield cmd_dict
    return [debugger.SearchMatch.from_json(i) for i in json['result']]


def set_blocked_ur_ls(
        urls: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Blocks URLs from loading.

    **EXPERIMENTAL**

    :param urls: URL patterns to block. Wildcards ('*') are allowed.
    '''
    params: T_JSON_DICT = dict()
    params['urls'] = [i for i in urls]
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setBlockedURLs',
        'params': params,
    }
    json = yield cmd_dict


def set_bypass_service_worker(
        bypass: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Toggles ignoring of service worker for each request.

    :param bypass: Bypass service worker and load from network.
    '''
    params: T_JSON_DICT = dict()
    params['bypass'] = bypass
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setBypassServiceWorker',
        'params': params,
    }
    json = yield cmd_dict


def set_cache_disabled(
        cache_disabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Toggles ignoring cache for each request. If ``true``, cache will not be used.

    :param cache_disabled: Cache disabled state.
    '''
    params: T_JSON_DICT = dict()
    params['cacheDisabled'] = cache_disabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setCacheDisabled',
        'params': params,
    }
    json = yield cmd_dict


def set_cookie(
        name: str,
        value: str,
        url: typing.Optional[str] = None,
        domain: typing.Optional[str] = None,
        path: typing.Optional[str] = None,
        secure: typing.Optional[bool] = None,
        http_only: typing.Optional[bool] = None,
        same_site: typing.Optional[CookieSameSite] = None,
        expires: typing.Optional[TimeSinceEpoch] = None,
        priority: typing.Optional[CookiePriority] = None,
        same_party: typing.Optional[bool] = None,
        source_scheme: typing.Optional[CookieSourceScheme] = None,
        source_port: typing.Optional[int] = None,
        partition_key: typing.Optional[CookiePartitionKey] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Sets a cookie with the given cookie data; may overwrite equivalent cookies if they exist.

    :param name: Cookie name.
    :param value: Cookie value.
    :param url: *(Optional)* The request-URI to associate with the setting of the cookie. This value can affect the default domain, path, source port, and source scheme values of the created cookie.
    :param domain: *(Optional)* Cookie domain.
    :param path: *(Optional)* Cookie path.
    :param secure: *(Optional)* True if cookie is secure.
    :param http_only: *(Optional)* True if cookie is http-only.
    :param same_site: *(Optional)* Cookie SameSite type.
    :param expires: *(Optional)* Cookie expiration date, session cookie if not set
    :param priority: **(EXPERIMENTAL)** *(Optional)* Cookie Priority type.
    :param same_party: **(EXPERIMENTAL)** *(Optional)* True if cookie is SameParty.
    :param source_scheme: **(EXPERIMENTAL)** *(Optional)* Cookie source scheme type.
    :param source_port: **(EXPERIMENTAL)** *(Optional)* Cookie source port. Valid values are {-1, [1, 65535]}, -1 indicates an unspecified port. An unspecified port value allows protocol clients to emulate legacy cookie scope for the port. This is a temporary ability and it will be removed in the future.
    :param partition_key: **(EXPERIMENTAL)** *(Optional)* Cookie partition key. If not set, the cookie will be set as not partitioned.
    :returns: Always set to true. If an error occurs, the response indicates protocol error.
    '''
    params: T_JSON_DICT = dict()
    params['name'] = name
    params['value'] = value
    if url is not None:
        params['url'] = url
    if domain is not None:
        params['domain'] = domain
    if path is not None:
        params['path'] = path
    if secure is not None:
        params['secure'] = secure
    if http_only is not None:
        params['httpOnly'] = http_only
    if same_site is not None:
        params['sameSite'] = same_site.to_json()
    if expires is not None:
        params['expires'] = expires.to_json()
    if priority is not None:
        params['priority'] = priority.to_json()
    if same_party is not None:
        params['sameParty'] = same_party
    if source_scheme is not None:
        params['sourceScheme'] = source_scheme.to_json()
    if source_port is not None:
        params['sourcePort'] = source_port
    if partition_key is not None:
        params['partitionKey'] = partition_key.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setCookie',
        'params': params,
    }
    json = yield cmd_dict
    return bool(json['success'])


def set_cookies(
        cookies: typing.List[CookieParam]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets given cookies.

    :param cookies: Cookies to be set.
    '''
    params: T_JSON_DICT = dict()
    params['cookies'] = [i.to_json() for i in cookies]
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setCookies',
        'params': params,
    }
    json = yield cmd_dict


def set_extra_http_headers(
        headers: Headers
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Specifies whether to always send extra HTTP headers with the requests from this page.

    :param headers: Map with extra HTTP headers.
    '''
    params: T_JSON_DICT = dict()
    params['headers'] = headers.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setExtraHTTPHeaders',
        'params': params,
    }
    json = yield cmd_dict


def set_attach_debug_stack(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Specifies whether to attach a page script stack id in requests

    **EXPERIMENTAL**

    :param enabled: Whether to attach a page script stack for debugging purpose.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setAttachDebugStack',
        'params': params,
    }
    json = yield cmd_dict


def set_request_interception(
        patterns: typing.List[RequestPattern]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the requests to intercept that match the provided patterns and optionally resource types.
    Deprecated, please use Fetch.enable instead.

    **EXPERIMENTAL**

    :param patterns: Requests matching any of these patterns will be forwarded and wait for the corresponding continueInterceptedRequest call.
    '''
    params: T_JSON_DICT = dict()
    params['patterns'] = [i.to_json() for i in patterns]
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setRequestInterception',
        'params': params,
    }
    json = yield cmd_dict


def set_user_agent_override(
        user_agent: str,
        accept_language: typing.Optional[str] = None,
        platform: typing.Optional[str] = None,
        user_agent_metadata: typing.Optional[emulation.UserAgentMetadata] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Allows overriding user agent with the given string.

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
        'method': 'Network.setUserAgentOverride',
        'params': params,
    }
    json = yield cmd_dict


def stream_resource_content(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Enables streaming of the response for the given requestId.
    If enabled, the dataReceived event contains the data that was received during streaming.

    **EXPERIMENTAL**

    :param request_id: Identifier of the request to stream.
    :returns: Data that has been buffered until streaming is enabled.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.streamResourceContent',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['bufferedData'])


def get_security_isolation_status(
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SecurityIsolationStatus]:
    '''
    Returns information about the COEP/COOP isolation status.

    **EXPERIMENTAL**

    :param frame_id: *(Optional)* If no frameId is provided, the status of the target is provided.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getSecurityIsolationStatus',
        'params': params,
    }
    json = yield cmd_dict
    return SecurityIsolationStatus.from_json(json['status'])


def enable_reporting_api(
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables tracking for the Reporting API, events generated by the Reporting API will now be delivered to the client.
    Enabling triggers 'reportingApiReportAdded' for all existing reports.

    **EXPERIMENTAL**

    :param enable: Whether to enable or disable events for the Reporting API
    '''
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.enableReportingApi',
        'params': params,
    }
    json = yield cmd_dict


def load_network_resource(
        frame_id: typing.Optional[page.FrameId] = None,
        url: str = None,
        options: LoadNetworkResourceOptions = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,LoadNetworkResourcePageResult]:
    '''
    Fetches the resource and returns the content.

    **EXPERIMENTAL**

    :param frame_id: *(Optional)* Frame id to get the resource for. Mandatory for frame targets, and should be omitted for worker targets.
    :param url: URL of the resource to get content for.
    :param options: Options for the request.
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    params['url'] = url
    params['options'] = options.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.loadNetworkResource',
        'params': params,
    }
    json = yield cmd_dict
    return LoadNetworkResourcePageResult.from_json(json['resource'])


def set_cookie_controls(
        enable_third_party_cookie_restriction: bool,
        disable_third_party_cookie_metadata: bool,
        disable_third_party_cookie_heuristics: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets Controls for third-party cookie access
    Page reload is required before the new cookie bahavior will be observed

    **EXPERIMENTAL**

    :param enable_third_party_cookie_restriction: Whether 3pc restriction is enabled.
    :param disable_third_party_cookie_metadata: Whether 3pc grace period exception should be enabled; false by default.
    :param disable_third_party_cookie_heuristics: Whether 3pc heuristics exceptions should be enabled; false by default.
    '''
    params: T_JSON_DICT = dict()
    params['enableThirdPartyCookieRestriction'] = enable_third_party_cookie_restriction
    params['disableThirdPartyCookieMetadata'] = disable_third_party_cookie_metadata
    params['disableThirdPartyCookieHeuristics'] = disable_third_party_cookie_heuristics
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setCookieControls',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Network.dataReceived')
@dataclass
class DataReceived:
    '''
    Fired when data chunk was received over the network.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: Data chunk length.
    data_length: int
    #: Actual bytes received (might be less than dataLength for compressed encodings).
    encoded_data_length: int
    #: Data that was received.
    data: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DataReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            data_length=int(json['dataLength']),
            encoded_data_length=int(json['encodedDataLength']),
            data=str(json['data']) if 'data' in json else None
        )


@event_class('Network.eventSourceMessageReceived')
@dataclass
class EventSourceMessageReceived:
    '''
    Fired when EventSource message is received.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: Message type.
    event_name: str
    #: Message identifier.
    event_id: str
    #: Message content.
    data: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> EventSourceMessageReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            event_name=str(json['eventName']),
            event_id=str(json['eventId']),
            data=str(json['data'])
        )


@event_class('Network.loadingFailed')
@dataclass
class LoadingFailed:
    '''
    Fired when HTTP request has failed to load.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: Resource type.
    type_: ResourceType
    #: Error message. List of network errors: https://cs.chromium.org/chromium/src/net/base/net_error_list.h
    error_text: str
    #: True if loading was canceled.
    canceled: typing.Optional[bool]
    #: The reason why loading was blocked, if any.
    blocked_reason: typing.Optional[BlockedReason]
    #: The reason why loading was blocked by CORS, if any.
    cors_error_status: typing.Optional[CorsErrorStatus]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadingFailed:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            type_=ResourceType.from_json(json['type']),
            error_text=str(json['errorText']),
            canceled=bool(json['canceled']) if 'canceled' in json else None,
            blocked_reason=BlockedReason.from_json(json['blockedReason']) if 'blockedReason' in json else None,
            cors_error_status=CorsErrorStatus.from_json(json['corsErrorStatus']) if 'corsErrorStatus' in json else None
        )


@event_class('Network.loadingFinished')
@dataclass
class LoadingFinished:
    '''
    Fired when HTTP request has finished loading.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: Total number of bytes received for this request.
    encoded_data_length: float

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadingFinished:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            encoded_data_length=float(json['encodedDataLength'])
        )


@event_class('Network.requestIntercepted')
@dataclass
class RequestIntercepted:
    '''
    **EXPERIMENTAL**

    Details of an intercepted HTTP request, which must be either allowed, blocked, modified or
    mocked.
    Deprecated, use Fetch.requestPaused instead.
    '''
    #: Each request the page makes will have a unique id, however if any redirects are encountered
    #: while processing that fetch, they will be reported with the same id as the original fetch.
    #: Likewise if HTTP authentication is needed then the same fetch id will be used.
    interception_id: InterceptionId
    request: Request
    #: The id of the frame that initiated the request.
    frame_id: page.FrameId
    #: How the requested resource will be used.
    resource_type: ResourceType
    #: Whether this is a navigation request, which can abort the navigation completely.
    is_navigation_request: bool
    #: Set if the request is a navigation that will result in a download.
    #: Only present after response is received from the server (i.e. HeadersReceived stage).
    is_download: typing.Optional[bool]
    #: Redirect location, only sent if a redirect was intercepted.
    redirect_url: typing.Optional[str]
    #: Details of the Authorization Challenge encountered. If this is set then
    #: continueInterceptedRequest must contain an authChallengeResponse.
    auth_challenge: typing.Optional[AuthChallenge]
    #: Response error if intercepted at response stage or if redirect occurred while intercepting
    #: request.
    response_error_reason: typing.Optional[ErrorReason]
    #: Response code if intercepted at response stage or if redirect occurred while intercepting
    #: request or auth retry occurred.
    response_status_code: typing.Optional[int]
    #: Response headers if intercepted at the response stage or if redirect occurred while
    #: intercepting request or auth retry occurred.
    response_headers: typing.Optional[Headers]
    #: If the intercepted request had a corresponding requestWillBeSent event fired for it, then
    #: this requestId will be the same as the requestId present in the requestWillBeSent event.
    request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestIntercepted:
        return cls(
            interception_id=InterceptionId.from_json(json['interceptionId']),
            request=Request.from_json(json['request']),
            frame_id=page.FrameId.from_json(json['frameId']),
            resource_type=ResourceType.from_json(json['resourceType']),
            is_navigation_request=bool(json['isNavigationRequest']),
            is_download=bool(json['isDownload']) if 'isDownload' in json else None,
            redirect_url=str(json['redirectUrl']) if 'redirectUrl' in json else None,
            auth_challenge=AuthChallenge.from_json(json['authChallenge']) if 'authChallenge' in json else None,
            response_error_reason=ErrorReason.from_json(json['responseErrorReason']) if 'responseErrorReason' in json else None,
            response_status_code=int(json['responseStatusCode']) if 'responseStatusCode' in json else None,
            response_headers=Headers.from_json(json['responseHeaders']) if 'responseHeaders' in json else None,
            request_id=RequestId.from_json(json['requestId']) if 'requestId' in json else None
        )


@event_class('Network.requestServedFromCache')
@dataclass
class RequestServedFromCache:
    '''
    Fired if request ended up loading from cache.
    '''
    #: Request identifier.
    request_id: RequestId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestServedFromCache:
        return cls(
            request_id=RequestId.from_json(json['requestId'])
        )


@event_class('Network.requestWillBeSent')
@dataclass
class RequestWillBeSent:
    '''
    Fired when page is about to send HTTP request.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Loader identifier. Empty string if the request is fetched from worker.
    loader_id: LoaderId
    #: URL of the document this request is loaded for.
    document_url: str
    #: Request data.
    request: Request
    #: Timestamp.
    timestamp: MonotonicTime
    #: Timestamp.
    wall_time: TimeSinceEpoch
    #: Request initiator.
    initiator: Initiator
    #: In the case that redirectResponse is populated, this flag indicates whether
    #: requestWillBeSentExtraInfo and responseReceivedExtraInfo events will be or were emitted
    #: for the request which was just redirected.
    redirect_has_extra_info: bool
    #: Redirect response data.
    redirect_response: typing.Optional[Response]
    #: Type of this resource.
    type_: typing.Optional[ResourceType]
    #: Frame identifier.
    frame_id: typing.Optional[page.FrameId]
    #: Whether the request is initiated by a user gesture. Defaults to false.
    has_user_gesture: typing.Optional[bool]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestWillBeSent:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            loader_id=LoaderId.from_json(json['loaderId']),
            document_url=str(json['documentURL']),
            request=Request.from_json(json['request']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            wall_time=TimeSinceEpoch.from_json(json['wallTime']),
            initiator=Initiator.from_json(json['initiator']),
            redirect_has_extra_info=bool(json['redirectHasExtraInfo']),
            redirect_response=Response.from_json(json['redirectResponse']) if 'redirectResponse' in json else None,
            type_=ResourceType.from_json(json['type']) if 'type' in json else None,
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None,
            has_user_gesture=bool(json['hasUserGesture']) if 'hasUserGesture' in json else None
        )


@event_class('Network.resourceChangedPriority')
@dataclass
class ResourceChangedPriority:
    '''
    **EXPERIMENTAL**

    Fired when resource loading priority is changed
    '''
    #: Request identifier.
    request_id: RequestId
    #: New priority
    new_priority: ResourcePriority
    #: Timestamp.
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResourceChangedPriority:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            new_priority=ResourcePriority.from_json(json['newPriority']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.signedExchangeReceived')
@dataclass
class SignedExchangeReceived:
    '''
    **EXPERIMENTAL**

    Fired when a signed exchange was received over the network
    '''
    #: Request identifier.
    request_id: RequestId
    #: Information about the signed exchange response.
    info: SignedExchangeInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SignedExchangeReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            info=SignedExchangeInfo.from_json(json['info'])
        )


@event_class('Network.responseReceived')
@dataclass
class ResponseReceived:
    '''
    Fired when HTTP response is available.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Loader identifier. Empty string if the request is fetched from worker.
    loader_id: LoaderId
    #: Timestamp.
    timestamp: MonotonicTime
    #: Resource type.
    type_: ResourceType
    #: Response data.
    response: Response
    #: Indicates whether requestWillBeSentExtraInfo and responseReceivedExtraInfo events will be
    #: or were emitted for this request.
    has_extra_info: bool
    #: Frame identifier.
    frame_id: typing.Optional[page.FrameId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResponseReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            loader_id=LoaderId.from_json(json['loaderId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            type_=ResourceType.from_json(json['type']),
            response=Response.from_json(json['response']),
            has_extra_info=bool(json['hasExtraInfo']),
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None
        )


@event_class('Network.webSocketClosed')
@dataclass
class WebSocketClosed:
    '''
    Fired when WebSocket is closed.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketClosed:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.webSocketCreated')
@dataclass
class WebSocketCreated:
    '''
    Fired upon WebSocket creation.
    '''
    #: Request identifier.
    request_id: RequestId
    #: WebSocket request URL.
    url: str
    #: Request initiator.
    initiator: typing.Optional[Initiator]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketCreated:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            url=str(json['url']),
            initiator=Initiator.from_json(json['initiator']) if 'initiator' in json else None
        )


@event_class('Network.webSocketFrameError')
@dataclass
class WebSocketFrameError:
    '''
    Fired when WebSocket message error occurs.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: WebSocket error message.
    error_message: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketFrameError:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            error_message=str(json['errorMessage'])
        )


@event_class('Network.webSocketFrameReceived')
@dataclass
class WebSocketFrameReceived:
    '''
    Fired when WebSocket message is received.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: WebSocket response data.
    response: WebSocketFrame

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketFrameReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            response=WebSocketFrame.from_json(json['response'])
        )


@event_class('Network.webSocketFrameSent')
@dataclass
class WebSocketFrameSent:
    '''
    Fired when WebSocket message is sent.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: WebSocket response data.
    response: WebSocketFrame

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketFrameSent:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            response=WebSocketFrame.from_json(json['response'])
        )


@event_class('Network.webSocketHandshakeResponseReceived')
@dataclass
class WebSocketHandshakeResponseReceived:
    '''
    Fired when WebSocket handshake response becomes available.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: WebSocket response data.
    response: WebSocketResponse

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketHandshakeResponseReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            response=WebSocketResponse.from_json(json['response'])
        )


@event_class('Network.webSocketWillSendHandshakeRequest')
@dataclass
class WebSocketWillSendHandshakeRequest:
    '''
    Fired when WebSocket is about to initiate handshake.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: UTC Timestamp.
    wall_time: TimeSinceEpoch
    #: WebSocket request data.
    request: WebSocketRequest

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketWillSendHandshakeRequest:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            wall_time=TimeSinceEpoch.from_json(json['wallTime']),
            request=WebSocketRequest.from_json(json['request'])
        )


@event_class('Network.webTransportCreated')
@dataclass
class WebTransportCreated:
    '''
    Fired upon WebTransport creation.
    '''
    #: WebTransport identifier.
    transport_id: RequestId
    #: WebTransport request URL.
    url: str
    #: Timestamp.
    timestamp: MonotonicTime
    #: Request initiator.
    initiator: typing.Optional[Initiator]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebTransportCreated:
        return cls(
            transport_id=RequestId.from_json(json['transportId']),
            url=str(json['url']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            initiator=Initiator.from_json(json['initiator']) if 'initiator' in json else None
        )


@event_class('Network.webTransportConnectionEstablished')
@dataclass
class WebTransportConnectionEstablished:
    '''
    Fired when WebTransport handshake is finished.
    '''
    #: WebTransport identifier.
    transport_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebTransportConnectionEstablished:
        return cls(
            transport_id=RequestId.from_json(json['transportId']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.webTransportClosed')
@dataclass
class WebTransportClosed:
    '''
    Fired when WebTransport is disposed.
    '''
    #: WebTransport identifier.
    transport_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebTransportClosed:
        return cls(
            transport_id=RequestId.from_json(json['transportId']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.requestWillBeSentExtraInfo')
@dataclass
class RequestWillBeSentExtraInfo:
    '''
    **EXPERIMENTAL**

    Fired when additional information about a requestWillBeSent event is available from the
    network stack. Not every requestWillBeSent event will have an additional
    requestWillBeSentExtraInfo fired for it, and there is no guarantee whether requestWillBeSent
    or requestWillBeSentExtraInfo will be fired first for the same request.
    '''
    #: Request identifier. Used to match this information to an existing requestWillBeSent event.
    request_id: RequestId
    #: A list of cookies potentially associated to the requested URL. This includes both cookies sent with
    #: the request and the ones not sent; the latter are distinguished by having blockedReasons field set.
    associated_cookies: typing.List[AssociatedCookie]
    #: Raw request headers as they will be sent over the wire.
    headers: Headers
    #: Connection timing information for the request.
    connect_timing: ConnectTiming
    #: The client security state set for the request.
    client_security_state: typing.Optional[ClientSecurityState]
    #: Whether the site has partitioned cookies stored in a partition different than the current one.
    site_has_cookie_in_other_partition: typing.Optional[bool]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestWillBeSentExtraInfo:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            associated_cookies=[AssociatedCookie.from_json(i) for i in json['associatedCookies']],
            headers=Headers.from_json(json['headers']),
            connect_timing=ConnectTiming.from_json(json['connectTiming']),
            client_security_state=ClientSecurityState.from_json(json['clientSecurityState']) if 'clientSecurityState' in json else None,
            site_has_cookie_in_other_partition=bool(json['siteHasCookieInOtherPartition']) if 'siteHasCookieInOtherPartition' in json else None
        )


@event_class('Network.responseReceivedExtraInfo')
@dataclass
class ResponseReceivedExtraInfo:
    '''
    **EXPERIMENTAL**

    Fired when additional information about a responseReceived event is available from the network
    stack. Not every responseReceived event will have an additional responseReceivedExtraInfo for
    it, and responseReceivedExtraInfo may be fired before or after responseReceived.
    '''
    #: Request identifier. Used to match this information to another responseReceived event.
    request_id: RequestId
    #: A list of cookies which were not stored from the response along with the corresponding
    #: reasons for blocking. The cookies here may not be valid due to syntax errors, which
    #: are represented by the invalid cookie line string instead of a proper cookie.
    blocked_cookies: typing.List[BlockedSetCookieWithReason]
    #: Raw response headers as they were received over the wire.
    #: Duplicate headers in the response are represented as a single key with their values
    #: concatentated using ``\n`` as the separator.
    #: See also ``headersText`` that contains verbatim text for HTTP/1.*.
    headers: Headers
    #: The IP address space of the resource. The address space can only be determined once the transport
    #: established the connection, so we can't send it in ``requestWillBeSentExtraInfo``.
    resource_ip_address_space: IPAddressSpace
    #: The status code of the response. This is useful in cases the request failed and no responseReceived
    #: event is triggered, which is the case for, e.g., CORS errors. This is also the correct status code
    #: for cached requests, where the status in responseReceived is a 200 and this will be 304.
    status_code: int
    #: Raw response header text as it was received over the wire. The raw text may not always be
    #: available, such as in the case of HTTP/2 or QUIC.
    headers_text: typing.Optional[str]
    #: The cookie partition key that will be used to store partitioned cookies set in this response.
    #: Only sent when partitioned cookies are enabled.
    cookie_partition_key: typing.Optional[CookiePartitionKey]
    #: True if partitioned cookies are enabled, but the partition key is not serializable to string.
    cookie_partition_key_opaque: typing.Optional[bool]
    #: A list of cookies which should have been blocked by 3PCD but are exempted and stored from
    #: the response with the corresponding reason.
    exempted_cookies: typing.Optional[typing.List[ExemptedSetCookieWithReason]]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResponseReceivedExtraInfo:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            blocked_cookies=[BlockedSetCookieWithReason.from_json(i) for i in json['blockedCookies']],
            headers=Headers.from_json(json['headers']),
            resource_ip_address_space=IPAddressSpace.from_json(json['resourceIPAddressSpace']),
            status_code=int(json['statusCode']),
            headers_text=str(json['headersText']) if 'headersText' in json else None,
            cookie_partition_key=CookiePartitionKey.from_json(json['cookiePartitionKey']) if 'cookiePartitionKey' in json else None,
            cookie_partition_key_opaque=bool(json['cookiePartitionKeyOpaque']) if 'cookiePartitionKeyOpaque' in json else None,
            exempted_cookies=[ExemptedSetCookieWithReason.from_json(i) for i in json['exemptedCookies']] if 'exemptedCookies' in json else None
        )


@event_class('Network.responseReceivedEarlyHints')
@dataclass
class ResponseReceivedEarlyHints:
    '''
    **EXPERIMENTAL**

    Fired when 103 Early Hints headers is received in addition to the common response.
    Not every responseReceived event will have an responseReceivedEarlyHints fired.
    Only one responseReceivedEarlyHints may be fired for eached responseReceived event.
    '''
    #: Request identifier. Used to match this information to another responseReceived event.
    request_id: RequestId
    #: Raw response headers as they were received over the wire.
    #: Duplicate headers in the response are represented as a single key with their values
    #: concatentated using ``\n`` as the separator.
    #: See also ``headersText`` that contains verbatim text for HTTP/1.*.
    headers: Headers

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResponseReceivedEarlyHints:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            headers=Headers.from_json(json['headers'])
        )


@event_class('Network.trustTokenOperationDone')
@dataclass
class TrustTokenOperationDone:
    '''
    **EXPERIMENTAL**

    Fired exactly once for each Trust Token operation. Depending on
    the type of the operation and whether the operation succeeded or
    failed, the event is fired before the corresponding request was sent
    or after the response was received.
    '''
    #: Detailed success or error status of the operation.
    #: 'AlreadyExists' also signifies a successful operation, as the result
    #: of the operation already exists und thus, the operation was abort
    #: preemptively (e.g. a cache hit).
    status: str
    type_: TrustTokenOperationType
    request_id: RequestId
    #: Top level origin. The context in which the operation was attempted.
    top_level_origin: typing.Optional[str]
    #: Origin of the issuer in case of a "Issuance" or "Redemption" operation.
    issuer_origin: typing.Optional[str]
    #: The number of obtained Trust Tokens on a successful "Issuance" operation.
    issued_token_count: typing.Optional[int]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TrustTokenOperationDone:
        return cls(
            status=str(json['status']),
            type_=TrustTokenOperationType.from_json(json['type']),
            request_id=RequestId.from_json(json['requestId']),
            top_level_origin=str(json['topLevelOrigin']) if 'topLevelOrigin' in json else None,
            issuer_origin=str(json['issuerOrigin']) if 'issuerOrigin' in json else None,
            issued_token_count=int(json['issuedTokenCount']) if 'issuedTokenCount' in json else None
        )


@event_class('Network.policyUpdated')
@dataclass
class PolicyUpdated:
    '''
    **EXPERIMENTAL**

    Fired once security policy has been updated.
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PolicyUpdated:
        return cls(

        )


@event_class('Network.subresourceWebBundleMetadataReceived')
@dataclass
class SubresourceWebBundleMetadataReceived:
    '''
    **EXPERIMENTAL**

    Fired once when parsing the .wbn file has succeeded.
    The event contains the information about the web bundle contents.
    '''
    #: Request identifier. Used to match this information to another event.
    request_id: RequestId
    #: A list of URLs of resources in the subresource Web Bundle.
    urls: typing.List[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleMetadataReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            urls=[str(i) for i in json['urls']]
        )


@event_class('Network.subresourceWebBundleMetadataError')
@dataclass
class SubresourceWebBundleMetadataError:
    '''
    **EXPERIMENTAL**

    Fired once when parsing the .wbn file has failed.
    '''
    #: Request identifier. Used to match this information to another event.
    request_id: RequestId
    #: Error message
    error_message: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleMetadataError:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            error_message=str(json['errorMessage'])
        )


@event_class('Network.subresourceWebBundleInnerResponseParsed')
@dataclass
class SubresourceWebBundleInnerResponseParsed:
    '''
    **EXPERIMENTAL**

    Fired when handling requests for resources within a .wbn file.
    Note: this will only be fired for resources that are requested by the webpage.
    '''
    #: Request identifier of the subresource request
    inner_request_id: RequestId
    #: URL of the subresource resource.
    inner_request_url: str
    #: Bundle request identifier. Used to match this information to another event.
    #: This made be absent in case when the instrumentation was enabled only
    #: after webbundle was parsed.
    bundle_request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleInnerResponseParsed:
        return cls(
            inner_request_id=RequestId.from_json(json['innerRequestId']),
            inner_request_url=str(json['innerRequestURL']),
            bundle_request_id=RequestId.from_json(json['bundleRequestId']) if 'bundleRequestId' in json else None
        )


@event_class('Network.subresourceWebBundleInnerResponseError')
@dataclass
class SubresourceWebBundleInnerResponseError:
    '''
    **EXPERIMENTAL**

    Fired when request for resources within a .wbn file failed.
    '''
    #: Request identifier of the subresource request
    inner_request_id: RequestId
    #: URL of the subresource resource.
    inner_request_url: str
    #: Error message
    error_message: str
    #: Bundle request identifier. Used to match this information to another event.
    #: This made be absent in case when the instrumentation was enabled only
    #: after webbundle was parsed.
    bundle_request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleInnerResponseError:
        return cls(
            inner_request_id=RequestId.from_json(json['innerRequestId']),
            inner_request_url=str(json['innerRequestURL']),
            error_message=str(json['errorMessage']),
            bundle_request_id=RequestId.from_json(json['bundleRequestId']) if 'bundleRequestId' in json else None
        )


@event_class('Network.reportingApiReportAdded')
@dataclass
class ReportingApiReportAdded:
    '''
    **EXPERIMENTAL**

    Is sent whenever a new report is added.
    And after 'enableReportingApi' for all existing reports.
    '''
    report: ReportingApiReport

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportingApiReportAdded:
        return cls(
            report=ReportingApiReport.from_json(json['report'])
        )


@event_class('Network.reportingApiReportUpdated')
@dataclass
class ReportingApiReportUpdated:
    '''
    **EXPERIMENTAL**


    '''
    report: ReportingApiReport

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportingApiReportUpdated:
        return cls(
            report=ReportingApiReport.from_json(json['report'])
        )


@event_class('Network.reportingApiEndpointsChangedForOrigin')
@dataclass
class ReportingApiEndpointsChangedForOrigin:
    '''
    **EXPERIMENTAL**


    '''
    #: Origin of the document(s) which configured the endpoints.
    origin: str
    endpoints: typing.List[ReportingApiEndpoint]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportingApiEndpointsChangedForOrigin:
        return cls(
            origin=str(json['origin']),
            endpoints=[ReportingApiEndpoint.from_json(i) for i in json['endpoints']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mssql\base.py ===
# dialects/mssql/base.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

"""
.. dialect:: mssql
    :name: Microsoft SQL Server
    :normal_support: 2012+
    :best_effort: 2005+

.. _mssql_external_dialects:

External Dialects
-----------------

In addition to the above DBAPI layers with native SQLAlchemy support, there
are third-party dialects for other DBAPI layers that are compatible
with SQL Server. See the "External Dialects" list on the
:ref:`dialect_toplevel` page.

.. _mssql_identity:

Auto Increment Behavior / IDENTITY Columns
------------------------------------------

SQL Server provides so-called "auto incrementing" behavior using the
``IDENTITY`` construct, which can be placed on any single integer column in a
table. SQLAlchemy considers ``IDENTITY`` within its default "autoincrement"
behavior for an integer primary key column, described at
:paramref:`_schema.Column.autoincrement`.  This means that by default,
the first integer primary key column in a :class:`_schema.Table` will be
considered to be the identity column - unless it is associated with a
:class:`.Sequence` - and will generate DDL as such::

    from sqlalchemy import Table, MetaData, Column, Integer

    m = MetaData()
    t = Table(
        "t",
        m,
        Column("id", Integer, primary_key=True),
        Column("x", Integer),
    )
    m.create_all(engine)

The above example will generate DDL as:

.. sourcecode:: sql

    CREATE TABLE t (
        id INTEGER NOT NULL IDENTITY,
        x INTEGER NULL,
        PRIMARY KEY (id)
    )

For the case where this default generation of ``IDENTITY`` is not desired,
specify ``False`` for the :paramref:`_schema.Column.autoincrement` flag,
on the first integer primary key column::

    m = MetaData()
    t = Table(
        "t",
        m,
        Column("id", Integer, primary_key=True, autoincrement=False),
        Column("x", Integer),
    )
    m.create_all(engine)

To add the ``IDENTITY`` keyword to a non-primary key column, specify
``True`` for the :paramref:`_schema.Column.autoincrement` flag on the desired
:class:`_schema.Column` object, and ensure that
:paramref:`_schema.Column.autoincrement`
is set to ``False`` on any integer primary key column::

    m = MetaData()
    t = Table(
        "t",
        m,
        Column("id", Integer, primary_key=True, autoincrement=False),
        Column("x", Integer, autoincrement=True),
    )
    m.create_all(engine)

.. versionchanged::  1.4   Added :class:`_schema.Identity` construct
   in a :class:`_schema.Column` to specify the start and increment
   parameters of an IDENTITY. These replace
   the use of the :class:`.Sequence` object in order to specify these values.

.. deprecated:: 1.4

   The ``mssql_identity_start`` and ``mssql_identity_increment`` parameters
   to :class:`_schema.Column` are deprecated and should we replaced by
   an :class:`_schema.Identity` object. Specifying both ways of configuring
   an IDENTITY will result in a compile error.
   These options are also no longer returned as part of the
   ``dialect_options`` key in :meth:`_reflection.Inspector.get_columns`.
   Use the information in the ``identity`` key instead.

.. deprecated:: 1.3

   The use of :class:`.Sequence` to specify IDENTITY characteristics is
   deprecated and will be removed in a future release.   Please use
   the :class:`_schema.Identity` object parameters
   :paramref:`_schema.Identity.start` and
   :paramref:`_schema.Identity.increment`.

.. versionchanged::  1.4   Removed the ability to use a :class:`.Sequence`
   object to modify IDENTITY characteristics. :class:`.Sequence` objects
   now only manipulate true T-SQL SEQUENCE types.

.. note::

    There can only be one IDENTITY column on the table.  When using
    ``autoincrement=True`` to enable the IDENTITY keyword, SQLAlchemy does not
    guard against multiple columns specifying the option simultaneously.  The
    SQL Server database will instead reject the ``CREATE TABLE`` statement.

.. note::

    An INSERT statement which attempts to provide a value for a column that is
    marked with IDENTITY will be rejected by SQL Server.   In order for the
    value to be accepted, a session-level option "SET IDENTITY_INSERT" must be
    enabled.   The SQLAlchemy SQL Server dialect will perform this operation
    automatically when using a core :class:`_expression.Insert`
    construct; if the
    execution specifies a value for the IDENTITY column, the "IDENTITY_INSERT"
    option will be enabled for the span of that statement's invocation.However,
    this scenario is not high performing and should not be relied upon for
    normal use.   If a table doesn't actually require IDENTITY behavior in its
    integer primary key column, the keyword should be disabled when creating
    the table by ensuring that ``autoincrement=False`` is set.

Controlling "Start" and "Increment"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specific control over the "start" and "increment" values for
the ``IDENTITY`` generator are provided using the
:paramref:`_schema.Identity.start` and :paramref:`_schema.Identity.increment`
parameters passed to the :class:`_schema.Identity` object::

    from sqlalchemy import Table, Integer, Column, Identity

    test = Table(
        "test",
        metadata,
        Column(
            "id", Integer, primary_key=True, Identity(start=100, increment=10)
        ),
        Column("name", String(20)),
    )

The CREATE TABLE for the above :class:`_schema.Table` object would be:

.. sourcecode:: sql

   CREATE TABLE test (
     id INTEGER NOT NULL IDENTITY(100,10) PRIMARY KEY,
     name VARCHAR(20) NULL,
   )

.. note::

   The :class:`_schema.Identity` object supports many other parameter in
   addition to ``start`` and ``increment``. These are not supported by
   SQL Server and will be ignored when generating the CREATE TABLE ddl.

.. versionchanged:: 1.3.19  The :class:`_schema.Identity` object is
   now used to affect the
   ``IDENTITY`` generator for a :class:`_schema.Column` under  SQL Server.
   Previously, the :class:`.Sequence` object was used.  As SQL Server now
   supports real sequences as a separate construct, :class:`.Sequence` will be
   functional in the normal way starting from SQLAlchemy version 1.4.


Using IDENTITY with Non-Integer numeric types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SQL Server also allows ``IDENTITY`` to be used with ``NUMERIC`` columns.  To
implement this pattern smoothly in SQLAlchemy, the primary datatype of the
column should remain as ``Integer``, however the underlying implementation
type deployed to the SQL Server database can be specified as ``Numeric`` using
:meth:`.TypeEngine.with_variant`::

    from sqlalchemy import Column
    from sqlalchemy import Integer
    from sqlalchemy import Numeric
    from sqlalchemy import String
    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()


    class TestTable(Base):
        __tablename__ = "test"
        id = Column(
            Integer().with_variant(Numeric(10, 0), "mssql"),
            primary_key=True,
            autoincrement=True,
        )
        name = Column(String)

In the above example, ``Integer().with_variant()`` provides clear usage
information that accurately describes the intent of the code. The general
restriction that ``autoincrement`` only applies to ``Integer`` is established
at the metadata level and not at the per-dialect level.

When using the above pattern, the primary key identifier that comes back from
the insertion of a row, which is also the value that would be assigned to an
ORM object such as ``TestTable`` above, will be an instance of ``Decimal()``
and not ``int`` when using SQL Server. The numeric return type of the
:class:`_types.Numeric` type can be changed to return floats by passing False
to :paramref:`_types.Numeric.asdecimal`. To normalize the return type of the
above ``Numeric(10, 0)`` to return Python ints (which also support "long"
integer values in Python 3), use :class:`_types.TypeDecorator` as follows::

    from sqlalchemy import TypeDecorator


    class NumericAsInteger(TypeDecorator):
        "normalize floating point return values into ints"

        impl = Numeric(10, 0, asdecimal=False)
        cache_ok = True

        def process_result_value(self, value, dialect):
            if value is not None:
                value = int(value)
            return value


    class TestTable(Base):
        __tablename__ = "test"
        id = Column(
            Integer().with_variant(NumericAsInteger, "mssql"),
            primary_key=True,
            autoincrement=True,
        )
        name = Column(String)

.. _mssql_insert_behavior:

INSERT behavior
^^^^^^^^^^^^^^^^

Handling of the ``IDENTITY`` column at INSERT time involves two key
techniques. The most common is being able to fetch the "last inserted value"
for a given ``IDENTITY`` column, a process which SQLAlchemy performs
implicitly in many cases, most importantly within the ORM.

The process for fetching this value has several variants:

* In the vast majority of cases, RETURNING is used in conjunction with INSERT
  statements on SQL Server in order to get newly generated primary key values:

  .. sourcecode:: sql

    INSERT INTO t (x) OUTPUT inserted.id VALUES (?)

  As of SQLAlchemy 2.0, the :ref:`engine_insertmanyvalues` feature is also
  used by default to optimize many-row INSERT statements; for SQL Server
  the feature takes place for both RETURNING and-non RETURNING
  INSERT statements.

  .. versionchanged:: 2.0.10 The :ref:`engine_insertmanyvalues` feature for
     SQL Server was temporarily disabled for SQLAlchemy version 2.0.9 due to
     issues with row ordering. As of 2.0.10 the feature is re-enabled, with
     special case handling for the unit of work's requirement for RETURNING to
     be ordered.

* When RETURNING is not available or has been disabled via
  ``implicit_returning=False``, either the ``scope_identity()`` function or
  the ``@@identity`` variable is used; behavior varies by backend:

  * when using PyODBC, the phrase ``; select scope_identity()`` will be
    appended to the end of the INSERT statement; a second result set will be
    fetched in order to receive the value.  Given a table as::

        t = Table(
            "t",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("x", Integer),
            implicit_returning=False,
        )

    an INSERT will look like:

    .. sourcecode:: sql

        INSERT INTO t (x) VALUES (?); select scope_identity()

  * Other dialects such as pymssql will call upon
    ``SELECT scope_identity() AS lastrowid`` subsequent to an INSERT
    statement. If the flag ``use_scope_identity=False`` is passed to
    :func:`_sa.create_engine`,
    the statement ``SELECT @@identity AS lastrowid``
    is used instead.

A table that contains an ``IDENTITY`` column will prohibit an INSERT statement
that refers to the identity column explicitly.  The SQLAlchemy dialect will
detect when an INSERT construct, created using a core
:func:`_expression.insert`
construct (not a plain string SQL), refers to the identity column, and
in this case will emit ``SET IDENTITY_INSERT ON`` prior to the insert
statement proceeding, and ``SET IDENTITY_INSERT OFF`` subsequent to the
execution.  Given this example::

    m = MetaData()
    t = Table(
        "t", m, Column("id", Integer, primary_key=True), Column("x", Integer)
    )
    m.create_all(engine)

    with engine.begin() as conn:
        conn.execute(t.insert(), {"id": 1, "x": 1}, {"id": 2, "x": 2})

The above column will be created with IDENTITY, however the INSERT statement
we emit is specifying explicit values.  In the echo output we can see
how SQLAlchemy handles this:

.. sourcecode:: sql

    CREATE TABLE t (
        id INTEGER NOT NULL IDENTITY(1,1),
        x INTEGER NULL,
        PRIMARY KEY (id)
    )

    COMMIT
    SET IDENTITY_INSERT t ON
    INSERT INTO t (id, x) VALUES (?, ?)
    ((1, 1), (2, 2))
    SET IDENTITY_INSERT t OFF
    COMMIT



This is an auxiliary use case suitable for testing and bulk insert scenarios.

SEQUENCE support
----------------

The :class:`.Sequence` object creates "real" sequences, i.e.,
``CREATE SEQUENCE``:

.. sourcecode:: pycon+sql

    >>> from sqlalchemy import Sequence
    >>> from sqlalchemy.schema import CreateSequence
    >>> from sqlalchemy.dialects import mssql
    >>> print(
    ...     CreateSequence(Sequence("my_seq", start=1)).compile(
    ...         dialect=mssql.dialect()
    ...     )
    ... )
    {printsql}CREATE SEQUENCE my_seq START WITH 1

For integer primary key generation, SQL Server's ``IDENTITY`` construct should
generally be preferred vs. sequence.

.. tip::

    The default start value for T-SQL is ``-2**63`` instead of 1 as
    in most other SQL databases. Users should explicitly set the
    :paramref:`.Sequence.start` to 1 if that's the expected default::

        seq = Sequence("my_sequence", start=1)

.. versionadded:: 1.4 added SQL Server support for :class:`.Sequence`

.. versionchanged:: 2.0 The SQL Server dialect will no longer implicitly
   render "START WITH 1" for ``CREATE SEQUENCE``, which was the behavior
   first implemented in version 1.4.

MAX on VARCHAR / NVARCHAR
-------------------------

SQL Server supports the special string "MAX" within the
:class:`_types.VARCHAR` and :class:`_types.NVARCHAR` datatypes,
to indicate "maximum length possible".   The dialect currently handles this as
a length of "None" in the base type, rather than supplying a
dialect-specific version of these types, so that a base type
specified such as ``VARCHAR(None)`` can assume "unlengthed" behavior on
more than one backend without using dialect-specific types.

To build a SQL Server VARCHAR or NVARCHAR with MAX length, use None::

    my_table = Table(
        "my_table",
        metadata,
        Column("my_data", VARCHAR(None)),
        Column("my_n_data", NVARCHAR(None)),
    )

Collation Support
-----------------

Character collations are supported by the base string types,
specified by the string argument "collation"::

    from sqlalchemy import VARCHAR

    Column("login", VARCHAR(32, collation="Latin1_General_CI_AS"))

When such a column is associated with a :class:`_schema.Table`, the
CREATE TABLE statement for this column will yield:

.. sourcecode:: sql

    login VARCHAR(32) COLLATE Latin1_General_CI_AS NULL

LIMIT/OFFSET Support
--------------------

MSSQL has added support for LIMIT / OFFSET as of SQL Server 2012, via the
"OFFSET n ROWS" and "FETCH NEXT n ROWS" clauses.  SQLAlchemy supports these
syntaxes automatically if SQL Server 2012 or greater is detected.

.. versionchanged:: 1.4 support added for SQL Server "OFFSET n ROWS" and
   "FETCH NEXT n ROWS" syntax.

For statements that specify only LIMIT and no OFFSET, all versions of SQL
Server support the TOP keyword.   This syntax is used for all SQL Server
versions when no OFFSET clause is present.  A statement such as::

    select(some_table).limit(5)

will render similarly to:

.. sourcecode:: sql

    SELECT TOP 5 col1, col2.. FROM table

For versions of SQL Server prior to SQL Server 2012, a statement that uses
LIMIT and OFFSET, or just OFFSET alone, will be rendered using the
``ROW_NUMBER()`` window function.   A statement such as::

    select(some_table).order_by(some_table.c.col3).limit(5).offset(10)

will render similarly to:

.. sourcecode:: sql

    SELECT anon_1.col1, anon_1.col2 FROM (SELECT col1, col2,
    ROW_NUMBER() OVER (ORDER BY col3) AS
    mssql_rn FROM table WHERE t.x = :x_1) AS
    anon_1 WHERE mssql_rn > :param_1 AND mssql_rn <= :param_2 + :param_1

Note that when using LIMIT and/or OFFSET, whether using the older
or newer SQL Server syntaxes, the statement must have an ORDER BY as well,
else a :class:`.CompileError` is raised.

.. _mssql_comment_support:

DDL Comment Support
--------------------

Comment support, which includes DDL rendering for attributes such as
:paramref:`_schema.Table.comment` and :paramref:`_schema.Column.comment`, as
well as the ability to reflect these comments, is supported assuming a
supported version of SQL Server is in use. If a non-supported version such as
Azure Synapse is detected at first-connect time (based on the presence
of the ``fn_listextendedproperty`` SQL function), comment support including
rendering and table-comment reflection is disabled, as both features rely upon
SQL Server stored procedures and functions that are not available on all
backend types.

To force comment support to be on or off, bypassing autodetection, set the
parameter ``supports_comments`` within :func:`_sa.create_engine`::

    e = create_engine("mssql+pyodbc://u:p@dsn", supports_comments=False)

.. versionadded:: 2.0 Added support for table and column comments for
   the SQL Server dialect, including DDL generation and reflection.

.. _mssql_isolation_level:

Transaction Isolation Level
---------------------------

All SQL Server dialects support setting of transaction isolation level
both via a dialect-specific parameter
:paramref:`_sa.create_engine.isolation_level`
accepted by :func:`_sa.create_engine`,
as well as the :paramref:`.Connection.execution_options.isolation_level`
argument as passed to
:meth:`_engine.Connection.execution_options`.
This feature works by issuing the
command ``SET TRANSACTION ISOLATION LEVEL <level>`` for
each new connection.

To set isolation level using :func:`_sa.create_engine`::

    engine = create_engine(
        "mssql+pyodbc://scott:tiger@ms_2008", isolation_level="REPEATABLE READ"
    )

To set using per-connection execution options::

    connection = engine.connect()
    connection = connection.execution_options(isolation_level="READ COMMITTED")

Valid values for ``isolation_level`` include:

* ``AUTOCOMMIT`` - pyodbc / pymssql-specific
* ``READ COMMITTED``
* ``READ UNCOMMITTED``
* ``REPEATABLE READ``
* ``SERIALIZABLE``
* ``SNAPSHOT`` - specific to SQL Server

There are also more options for isolation level configurations, such as
"sub-engine" objects linked to a main :class:`_engine.Engine` which each apply
different isolation level settings.  See the discussion at
:ref:`dbapi_autocommit` for background.

.. seealso::

    :ref:`dbapi_autocommit`

.. _mssql_reset_on_return:

Temporary Table / Resource Reset for Connection Pooling
-------------------------------------------------------

The :class:`.QueuePool` connection pool implementation used
by the SQLAlchemy :class:`.Engine` object includes
:ref:`reset on return <pool_reset_on_return>` behavior that will invoke
the DBAPI ``.rollback()`` method when connections are returned to the pool.
While this rollback will clear out the immediate state used by the previous
transaction, it does not cover a wider range of session-level state, including
temporary tables as well as other server state such as prepared statement
handles and statement caches.   An undocumented SQL Server procedure known
as ``sp_reset_connection`` is known to be a workaround for this issue which
will reset most of the session state that builds up on a connection, including
temporary tables.

To install ``sp_reset_connection`` as the means of performing reset-on-return,
the :meth:`.PoolEvents.reset` event hook may be used, as demonstrated in the
example below. The :paramref:`_sa.create_engine.pool_reset_on_return` parameter
is set to ``None`` so that the custom scheme can replace the default behavior
completely.   The custom hook implementation calls ``.rollback()`` in any case,
as it's usually important that the DBAPI's own tracking of commit/rollback
will remain consistent with the state of the transaction::

    from sqlalchemy import create_engine
    from sqlalchemy import event

    mssql_engine = create_engine(
        "mssql+pyodbc://scott:tiger^5HHH@mssql2017:1433/test?driver=ODBC+Driver+17+for+SQL+Server",
        # disable default reset-on-return scheme
        pool_reset_on_return=None,
    )


    @event.listens_for(mssql_engine, "reset")
    def _reset_mssql(dbapi_connection, connection_record, reset_state):
        if not reset_state.terminate_only:
            dbapi_connection.execute("{call sys.sp_reset_connection}")

        # so that the DBAPI itself knows that the connection has been
        # reset
        dbapi_connection.rollback()

.. versionchanged:: 2.0.0b3  Added additional state arguments to
   the :meth:`.PoolEvents.reset` event and additionally ensured the event
   is invoked for all "reset" occurrences, so that it's appropriate
   as a place for custom "reset" handlers.   Previous schemes which
   use the :meth:`.PoolEvents.checkin` handler remain usable as well.

.. seealso::

    :ref:`pool_reset_on_return` - in the :ref:`pooling_toplevel` documentation

Nullability
-----------
MSSQL has support for three levels of column nullability. The default
nullability allows nulls and is explicit in the CREATE TABLE
construct:

.. sourcecode:: sql

    name VARCHAR(20) NULL

If ``nullable=None`` is specified then no specification is made. In
other words the database's configured default is used. This will
render:

.. sourcecode:: sql

    name VARCHAR(20)

If ``nullable`` is ``True`` or ``False`` then the column will be
``NULL`` or ``NOT NULL`` respectively.

Date / Time Handling
--------------------
DATE and TIME are supported.   Bind parameters are converted
to datetime.datetime() objects as required by most MSSQL drivers,
and results are processed from strings if needed.
The DATE and TIME types are not available for MSSQL 2005 and
previous - if a server version below 2008 is detected, DDL
for these types will be issued as DATETIME.

.. _mssql_large_type_deprecation:

Large Text/Binary Type Deprecation
----------------------------------

Per
`SQL Server 2012/2014 Documentation <https://technet.microsoft.com/en-us/library/ms187993.aspx>`_,
the ``NTEXT``, ``TEXT`` and ``IMAGE`` datatypes are to be removed from SQL
Server in a future release.   SQLAlchemy normally relates these types to the
:class:`.UnicodeText`, :class:`_expression.TextClause` and
:class:`.LargeBinary` datatypes.

In order to accommodate this change, a new flag ``deprecate_large_types``
is added to the dialect, which will be automatically set based on detection
of the server version in use, if not otherwise set by the user.  The
behavior of this flag is as follows:

* When this flag is ``True``, the :class:`.UnicodeText`,
  :class:`_expression.TextClause` and
  :class:`.LargeBinary` datatypes, when used to render DDL, will render the
  types ``NVARCHAR(max)``, ``VARCHAR(max)``, and ``VARBINARY(max)``,
  respectively.  This is a new behavior as of the addition of this flag.

* When this flag is ``False``, the :class:`.UnicodeText`,
  :class:`_expression.TextClause` and
  :class:`.LargeBinary` datatypes, when used to render DDL, will render the
  types ``NTEXT``, ``TEXT``, and ``IMAGE``,
  respectively.  This is the long-standing behavior of these types.

* The flag begins with the value ``None``, before a database connection is
  established.   If the dialect is used to render DDL without the flag being
  set, it is interpreted the same as ``False``.

* On first connection, the dialect detects if SQL Server version 2012 or
  greater is in use; if the flag is still at ``None``, it sets it to ``True``
  or ``False`` based on whether 2012 or greater is detected.

* The flag can be set to either ``True`` or ``False`` when the dialect
  is created, typically via :func:`_sa.create_engine`::

        eng = create_engine(
            "mssql+pymssql://user:pass@host/db", deprecate_large_types=True
        )

* Complete control over whether the "old" or "new" types are rendered is
  available in all SQLAlchemy versions by using the UPPERCASE type objects
  instead: :class:`_types.NVARCHAR`, :class:`_types.VARCHAR`,
  :class:`_types.VARBINARY`, :class:`_types.TEXT`, :class:`_mssql.NTEXT`,
  :class:`_mssql.IMAGE`
  will always remain fixed and always output exactly that
  type.

.. _multipart_schema_names:

Multipart Schema Names
----------------------

SQL Server schemas sometimes require multiple parts to their "schema"
qualifier, that is, including the database name and owner name as separate
tokens, such as ``mydatabase.dbo.some_table``. These multipart names can be set
at once using the :paramref:`_schema.Table.schema` argument of
:class:`_schema.Table`::

    Table(
        "some_table",
        metadata,
        Column("q", String(50)),
        schema="mydatabase.dbo",
    )

When performing operations such as table or component reflection, a schema
argument that contains a dot will be split into separate
"database" and "owner"  components in order to correctly query the SQL
Server information schema tables, as these two values are stored separately.
Additionally, when rendering the schema name for DDL or SQL, the two
components will be quoted separately for case sensitive names and other
special characters.   Given an argument as below::

    Table(
        "some_table",
        metadata,
        Column("q", String(50)),
        schema="MyDataBase.dbo",
    )

The above schema would be rendered as ``[MyDataBase].dbo``, and also in
reflection, would be reflected using "dbo" as the owner and "MyDataBase"
as the database name.

To control how the schema name is broken into database / owner,
specify brackets (which in SQL Server are quoting characters) in the name.
Below, the "owner" will be considered as ``MyDataBase.dbo`` and the
"database" will be None::

    Table(
        "some_table",
        metadata,
        Column("q", String(50)),
        schema="[MyDataBase.dbo]",
    )

To individually specify both database and owner name with special characters
or embedded dots, use two sets of brackets::

    Table(
        "some_table",
        metadata,
        Column("q", String(50)),
        schema="[MyDataBase.Period].[MyOwner.Dot]",
    )

.. versionchanged:: 1.2 the SQL Server dialect now treats brackets as
   identifier delimiters splitting the schema into separate database
   and owner tokens, to allow dots within either name itself.

.. _legacy_schema_rendering:

Legacy Schema Mode
------------------

Very old versions of the MSSQL dialect introduced the behavior such that a
schema-qualified table would be auto-aliased when used in a
SELECT statement; given a table::

    account_table = Table(
        "account",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("info", String(100)),
        schema="customer_schema",
    )

this legacy mode of rendering would assume that "customer_schema.account"
would not be accepted by all parts of the SQL statement, as illustrated
below:

.. sourcecode:: pycon+sql

    >>> eng = create_engine("mssql+pymssql://mydsn", legacy_schema_aliasing=True)
    >>> print(account_table.select().compile(eng))
    {printsql}SELECT account_1.id, account_1.info
    FROM customer_schema.account AS account_1

This mode of behavior is now off by default, as it appears to have served
no purpose; however in the case that legacy applications rely upon it,
it is available using the ``legacy_schema_aliasing`` argument to
:func:`_sa.create_engine` as illustrated above.

.. deprecated:: 1.4

   The ``legacy_schema_aliasing`` flag is now
   deprecated and will be removed in a future release.

.. _mssql_indexes:

Clustered Index Support
-----------------------

The MSSQL dialect supports clustered indexes (and primary keys) via the
``mssql_clustered`` option.  This option is available to :class:`.Index`,
:class:`.UniqueConstraint`. and :class:`.PrimaryKeyConstraint`.
For indexes this option can be combined with the ``mssql_columnstore`` one
to create a clustered columnstore index.

To generate a clustered index::

    Index("my_index", table.c.x, mssql_clustered=True)

which renders the index as ``CREATE CLUSTERED INDEX my_index ON table (x)``.

To generate a clustered primary key use::

    Table(
        "my_table",
        metadata,
        Column("x", ...),
        Column("y", ...),
        PrimaryKeyConstraint("x", "y", mssql_clustered=True),
    )

which will render the table, for example, as:

.. sourcecode:: sql

  CREATE TABLE my_table (
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    PRIMARY KEY CLUSTERED (x, y)
  )

Similarly, we can generate a clustered unique constraint using::

    Table(
        "my_table",
        metadata,
        Column("x", ...),
        Column("y", ...),
        PrimaryKeyConstraint("x"),
        UniqueConstraint("y", mssql_clustered=True),
    )

To explicitly request a non-clustered primary key (for example, when
a separate clustered index is desired), use::

    Table(
        "my_table",
        metadata,
        Column("x", ...),
        Column("y", ...),
        PrimaryKeyConstraint("x", "y", mssql_clustered=False),
    )

which will render the table, for example, as:

.. sourcecode:: sql

  CREATE TABLE my_table (
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    PRIMARY KEY NONCLUSTERED (x, y)
  )

Columnstore Index Support
-------------------------

The MSSQL dialect supports columnstore indexes via the ``mssql_columnstore``
option.  This option is available to :class:`.Index`. It be combined with
the ``mssql_clustered`` option to create a clustered columnstore index.

To generate a columnstore index::

    Index("my_index", table.c.x, mssql_columnstore=True)

which renders the index as ``CREATE COLUMNSTORE INDEX my_index ON table (x)``.

To generate a clustered columnstore index provide no columns::

    idx = Index("my_index", mssql_clustered=True, mssql_columnstore=True)
    # required to associate the index with the table
    table.append_constraint(idx)

the above renders the index as
``CREATE CLUSTERED COLUMNSTORE INDEX my_index ON table``.

.. versionadded:: 2.0.18

MSSQL-Specific Index Options
-----------------------------

In addition to clustering, the MSSQL dialect supports other special options
for :class:`.Index`.

INCLUDE
^^^^^^^

The ``mssql_include`` option renders INCLUDE(colname) for the given string
names::

    Index("my_index", table.c.x, mssql_include=["y"])

would render the index as ``CREATE INDEX my_index ON table (x) INCLUDE (y)``

.. _mssql_index_where:

Filtered Indexes
^^^^^^^^^^^^^^^^

The ``mssql_where`` option renders WHERE(condition) for the given string
names::

    Index("my_index", table.c.x, mssql_where=table.c.x > 10)

would render the index as ``CREATE INDEX my_index ON table (x) WHERE x > 10``.

.. versionadded:: 1.3.4

Index ordering
^^^^^^^^^^^^^^

Index ordering is available via functional expressions, such as::

    Index("my_index", table.c.x.desc())

would render the index as ``CREATE INDEX my_index ON table (x DESC)``

.. seealso::

    :ref:`schema_indexes_functional`

Compatibility Levels
--------------------
MSSQL supports the notion of setting compatibility levels at the
database level. This allows, for instance, to run a database that
is compatible with SQL2000 while running on a SQL2005 database
server. ``server_version_info`` will always return the database
server version information (in this case SQL2005) and not the
compatibility level information. Because of this, if running under
a backwards compatibility mode SQLAlchemy may attempt to use T-SQL
statements that are unable to be parsed by the database server.

.. _mssql_triggers:

Triggers
--------

SQLAlchemy by default uses OUTPUT INSERTED to get at newly
generated primary key values via IDENTITY columns or other
server side defaults.   MS-SQL does not
allow the usage of OUTPUT INSERTED on tables that have triggers.
To disable the usage of OUTPUT INSERTED on a per-table basis,
specify ``implicit_returning=False`` for each :class:`_schema.Table`
which has triggers::

    Table(
        "mytable",
        metadata,
        Column("id", Integer, primary_key=True),
        # ...,
        implicit_returning=False,
    )

Declarative form::

    class MyClass(Base):
        # ...
        __table_args__ = {"implicit_returning": False}

.. _mssql_rowcount_versioning:

Rowcount Support / ORM Versioning
---------------------------------

The SQL Server drivers may have limited ability to return the number
of rows updated from an UPDATE or DELETE statement.

As of this writing, the PyODBC driver is not able to return a rowcount when
OUTPUT INSERTED is used.    Previous versions of SQLAlchemy therefore had
limitations for features such as the "ORM Versioning" feature that relies upon
accurate rowcounts in order to match version numbers with matched rows.

SQLAlchemy 2.0 now retrieves the "rowcount" manually for these particular use
cases based on counting the rows that arrived back within RETURNING; so while
the driver still has this limitation, the ORM Versioning feature is no longer
impacted by it. As of SQLAlchemy 2.0.5, ORM versioning has been fully
re-enabled for the pyodbc driver.

.. versionchanged:: 2.0.5  ORM versioning support is restored for the pyodbc
   driver.  Previously, a warning would be emitted during ORM flush that
   versioning was not supported.


Enabling Snapshot Isolation
---------------------------

SQL Server has a default transaction
isolation mode that locks entire tables, and causes even mildly concurrent
applications to have long held locks and frequent deadlocks.
Enabling snapshot isolation for the database as a whole is recommended
for modern levels of concurrency support.  This is accomplished via the
following ALTER DATABASE commands executed at the SQL prompt:

.. sourcecode:: sql

    ALTER DATABASE MyDatabase SET ALLOW_SNAPSHOT_ISOLATION ON

    ALTER DATABASE MyDatabase SET READ_COMMITTED_SNAPSHOT ON

Background on SQL Server snapshot isolation is available at
https://msdn.microsoft.com/en-us/library/ms175095.aspx.

"""  # noqa

from __future__ import annotations

import codecs
import datetime
import operator
import re
from typing import overload
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID

from . import information_schema as ischema
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import Identity
from ... import schema as sa_schema
from ... import Sequence
from ... import sql
from ... import text
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import func
from ...sql import quoted_name
from ...sql import roles
from ...sql import sqltypes
from ...sql import try_cast as try_cast  # noqa: F401
from ...sql import util as sql_util
from ...sql._typing import is_sql_compiler
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.elements import TryCast as TryCast  # noqa: F401
from ...types import BIGINT
from ...types import BINARY
from ...types import CHAR
from ...types import DATE
from ...types import DATETIME
from ...types import DECIMAL
from ...types import FLOAT
from ...types import INTEGER
from ...types import NCHAR
from ...types import NUMERIC
from ...types import NVARCHAR
from ...types import SMALLINT
from ...types import TEXT
from ...types import VARCHAR
from ...util import update_wrapper
from ...util.typing import Literal

if TYPE_CHECKING:
    from ...sql.dml import DMLState
    from ...sql.selectable import TableClause

# https://sqlserverbuilds.blogspot.com/
MS_2017_VERSION = (14,)
MS_2016_VERSION = (13,)
MS_2014_VERSION = (12,)
MS_2012_VERSION = (11,)
MS_2008_VERSION = (10,)
MS_2005_VERSION = (9,)
MS_2000_VERSION = (8,)

RESERVED_WORDS = {
    "add",
    "all",
    "alter",
    "and",
    "any",
    "as",
    "asc",
    "authorization",
    "backup",
    "begin",
    "between",
    "break",
    "browse",
    "bulk",
    "by",
    "cascade",
    "case",
    "check",
    "checkpoint",
    "close",
    "clustered",
    "coalesce",
    "collate",
    "column",
    "commit",
    "compute",
    "constraint",
    "contains",
    "containstable",
    "continue",
    "convert",
    "create",
    "cross",
    "current",
    "current_date",
    "current_time",
    "current_timestamp",
    "current_user",
    "cursor",
    "database",
    "dbcc",
    "deallocate",
    "declare",
    "default",
    "delete",
    "deny",
    "desc",
    "disk",
    "distinct",
    "distributed",
    "double",
    "drop",
    "dump",
    "else",
    "end",
    "errlvl",
    "escape",
    "except",
    "exec",
    "execute",
    "exists",
    "exit",
    "external",
    "fetch",
    "file",
    "fillfactor",
    "for",
    "foreign",
    "freetext",
    "freetexttable",
    "from",
    "full",
    "function",
    "goto",
    "grant",
    "group",
    "having",
    "holdlock",
    "identity",
    "identity_insert",
    "identitycol",
    "if",
    "in",
    "index",
    "inner",
    "insert",
    "intersect",
    "into",
    "is",
    "join",
    "key",
    "kill",
    "left",
    "like",
    "lineno",
    "load",
    "merge",
    "national",
    "nocheck",
    "nonclustered",
    "not",
    "null",
    "nullif",
    "of",
    "off",
    "offsets",
    "on",
    "open",
    "opendatasource",
    "openquery",
    "openrowset",
    "openxml",
    "option",
    "or",
    "order",
    "outer",
    "over",
    "percent",
    "pivot",
    "plan",
    "precision",
    "primary",
    "print",
    "proc",
    "procedure",
    "public",
    "raiserror",
    "read",
    "readtext",
    "reconfigure",
    "references",
    "replication",
    "restore",
    "restrict",
    "return",
    "revert",
    "revoke",
    "right",
    "rollback",
    "rowcount",
    "rowguidcol",
    "rule",
    "save",
    "schema",
    "securityaudit",
    "select",
    "session_user",
    "set",
    "setuser",
    "shutdown",
    "some",
    "statistics",
    "system_user",
    "table",
    "tablesample",
    "textsize",
    "then",
    "to",
    "top",
    "tran",
    "transaction",
    "trigger",
    "truncate",
    "tsequal",
    "union",
    "unique",
    "unpivot",
    "update",
    "updatetext",
    "use",
    "user",
    "values",
    "varying",
    "view",
    "waitfor",
    "when",
    "where",
    "while",
    "with",
    "writetext",
}


class REAL(sqltypes.REAL):
    """the SQL Server REAL datatype."""

    def __init__(self, **kw):
        # REAL is a synonym for FLOAT(24) on SQL server.
        # it is only accepted as the word "REAL" in DDL, the numeric
        # precision value is not allowed to be present
        kw.setdefault("precision", 24)
        super().__init__(**kw)


class DOUBLE_PRECISION(sqltypes.DOUBLE_PRECISION):
    """the SQL Server DOUBLE PRECISION datatype.

    .. versionadded:: 2.0.11

    """

    def __init__(self, **kw):
        # DOUBLE PRECISION is a synonym for FLOAT(53) on SQL server.
        # it is only accepted as the word "DOUBLE PRECISION" in DDL,
        # the numeric precision value is not allowed to be present
        kw.setdefault("precision", 53)
        super().__init__(**kw)


class TINYINT(sqltypes.Integer):
    __visit_name__ = "TINYINT"


# MSSQL DATE/TIME types have varied behavior, sometimes returning
# strings.  MSDate/TIME check for everything, and always
# filter bind parameters into datetime objects (required by pyodbc,
# not sure about other dialects).


class _MSDate(sqltypes.Date):
    def bind_processor(self, dialect):
        def process(value):
            if type(value) == datetime.date:
                return datetime.datetime(value.year, value.month, value.day)
            else:
                return value

        return process

    _reg = re.compile(r"(\d+)-(\d+)-(\d+)")

    def result_processor(self, dialect, coltype):
        def process(value):
            if isinstance(value, datetime.datetime):
                return value.date()
            elif isinstance(value, str):
                m = self._reg.match(value)
                if not m:
                    raise ValueError(
                        "could not parse %r as a date value" % (value,)
                    )
                return datetime.date(*[int(x or 0) for x in m.groups()])
            else:
                return value

        return process


class TIME(sqltypes.TIME):
    def __init__(self, precision=None, **kwargs):
        self.precision = precision
        super().__init__()

    __zero_date = datetime.date(1900, 1, 1)

    def bind_processor(self, dialect):
        def process(value):
            if isinstance(value, datetime.datetime):
                value = datetime.datetime.combine(
                    self.__zero_date, value.time()
                )
            elif isinstance(value, datetime.time):
                """issue #5339
                per: https://github.com/mkleehammer/pyodbc/wiki/Tips-and-Tricks-by-Database-Platform#time-columns
                pass TIME value as string
                """  # noqa
                value = str(value)
            return value

        return process

    _reg = re.compile(r"(\d+):(\d+):(\d+)(?:\.(\d{0,6}))?")

    def result_processor(self, dialect, coltype):
        def process(value):
            if isinstance(value, datetime.datetime):
                return value.time()
            elif isinstance(value, str):
                m = self._reg.match(value)
                if not m:
                    raise ValueError(
                        "could not parse %r as a time value" % (value,)
                    )
                return datetime.time(*[int(x or 0) for x in m.groups()])
            else:
                return value

        return process


_MSTime = TIME


class _BASETIMEIMPL(TIME):
    __visit_name__ = "_BASETIMEIMPL"


class _DateTimeBase:
    def bind_processor(self, dialect):
        def process(value):
            if type(value) == datetime.date:
                return datetime.datetime(value.year, value.month, value.day)
            else:
                return value

        return process


class _MSDateTime(_DateTimeBase, sqltypes.DateTime):
    pass


class SMALLDATETIME(_DateTimeBase, sqltypes.DateTime):
    __visit_name__ = "SMALLDATETIME"


class DATETIME2(_DateTimeBase, sqltypes.DateTime):
    __visit_name__ = "DATETIME2"

    def __init__(self, precision=None, **kw):
        super().__init__(**kw)
        self.precision = precision


class DATETIMEOFFSET(_DateTimeBase, sqltypes.DateTime):
    __visit_name__ = "DATETIMEOFFSET"

    def __init__(self, precision=None, **kw):
        super().__init__(**kw)
        self.precision = precision


class _UnicodeLiteral:
    def literal_processor(self, dialect):
        def process(value):
            value = value.replace("'", "''")

            if dialect.identifier_preparer._double_percents:
                value = value.replace("%", "%%")

            return "N'%s'" % value

        return process


class _MSUnicode(_UnicodeLiteral, sqltypes.Unicode):
    pass


class _MSUnicodeText(_UnicodeLiteral, sqltypes.UnicodeText):
    pass


class TIMESTAMP(sqltypes._Binary):
    """Implement the SQL Server TIMESTAMP type.

    Note this is **completely different** than the SQL Standard
    TIMESTAMP type, which is not supported by SQL Server.  It
    is a read-only datatype that does not support INSERT of values.

    .. versionadded:: 1.2

    .. seealso::

        :class:`_mssql.ROWVERSION`

    """

    __visit_name__ = "TIMESTAMP"

    # expected by _Binary to be present
    length = None

    def __init__(self, convert_int=False):
        """Construct a TIMESTAMP or ROWVERSION type.

        :param convert_int: if True, binary integer values will
         be converted to integers on read.

        .. versionadded:: 1.2

        """
        self.convert_int = convert_int

    def result_processor(self, dialect, coltype):
        super_ = super().result_processor(dialect, coltype)
        if self.convert_int:

            def process(value):
                if super_:
                    value = super_(value)
                if value is not None:
                    # https://stackoverflow.com/a/30403242/34549
                    value = int(codecs.encode(value, "hex"), 16)
                return value

            return process
        else:
            return super_


class ROWVERSION(TIMESTAMP):
    """Implement the SQL Server ROWVERSION type.

    The ROWVERSION datatype is a SQL Server synonym for the TIMESTAMP
    datatype, however current SQL Server documentation suggests using
    ROWVERSION for new datatypes going forward.

    The ROWVERSION datatype does **not** reflect (e.g. introspect) from the
    database as itself; the returned datatype will be
    :class:`_mssql.TIMESTAMP`.

    This is a read-only datatype that does not support INSERT of values.

    .. versionadded:: 1.2

    .. seealso::

        :class:`_mssql.TIMESTAMP`

    """

    __visit_name__ = "ROWVERSION"


class NTEXT(sqltypes.UnicodeText):
    """MSSQL NTEXT type, for variable-length unicode text up to 2^30
    characters."""

    __visit_name__ = "NTEXT"


class VARBINARY(sqltypes.VARBINARY, sqltypes.LargeBinary):
    """The MSSQL VARBINARY type.

    This type adds additional features to the core :class:`_types.VARBINARY`
    type, including "deprecate_large_types" mode where
    either ``VARBINARY(max)`` or IMAGE is rendered, as well as the SQL
    Server ``FILESTREAM`` option.

    .. seealso::

        :ref:`mssql_large_type_deprecation`

    """

    __visit_name__ = "VARBINARY"

    def __init__(self, length=None, filestream=False):
        """
        Construct a VARBINARY type.

        :param length: optional, a length for the column for use in
          DDL statements, for those binary types that accept a length,
          such as the MySQL BLOB type.

        :param filestream=False: if True, renders the ``FILESTREAM`` keyword
          in the table definition. In this case ``length`` must be ``None``
          or ``'max'``.

          .. versionadded:: 1.4.31

        """

        self.filestream = filestream
        if self.filestream and length not in (None, "max"):
            raise ValueError(
                "length must be None or 'max' when setting filestream"
            )
        super().__init__(length=length)


class IMAGE(sqltypes.LargeBinary):
    __visit_name__ = "IMAGE"


class XML(sqltypes.Text):
    """MSSQL XML type.

    This is a placeholder type for reflection purposes that does not include
    any Python-side datatype support.   It also does not currently support
    additional arguments, such as "CONTENT", "DOCUMENT",
    "xml_schema_collection".

    """

    __visit_name__ = "XML"


class BIT(sqltypes.Boolean):
    """MSSQL BIT type.

    Both pyodbc and pymssql return values from BIT columns as
    Python <class 'bool'> so just subclass Boolean.

    """

    __visit_name__ = "BIT"


class MONEY(sqltypes.TypeEngine):
    __visit_name__ = "MONEY"


class SMALLMONEY(sqltypes.TypeEngine):
    __visit_name__ = "SMALLMONEY"


class MSUUid(sqltypes.Uuid):
    def bind_processor(self, dialect):
        if self.native_uuid:
            # this is currently assuming pyodbc; might not work for
            # some other mssql driver
            return None
        else:
            if self.as_uuid:

                def process(value):
                    if value is not None:
                        value = value.hex
                    return value

                return process
            else:

                def process(value):
                    if value is not None:
                        value = value.replace("-", "").replace("''", "'")
                    return value

                return process

    def literal_processor(self, dialect):
        if self.native_uuid:

            def process(value):
                return f"""'{str(value).replace("''", "'")}'"""

            return process
        else:
            if self.as_uuid:

                def process(value):
                    return f"""'{value.hex}'"""

                return process
            else:

                def process(value):
                    return f"""'{
                        value.replace("-", "").replace("'", "''")
                    }'"""

                return process


class UNIQUEIDENTIFIER(sqltypes.Uuid[sqltypes._UUID_RETURN]):
    __visit_name__ = "UNIQUEIDENTIFIER"

    @overload
    def __init__(
        self: UNIQUEIDENTIFIER[_python_UUID], as_uuid: Literal[True] = ...
    ): ...

    @overload
    def __init__(
        self: UNIQUEIDENTIFIER[str], as_uuid: Literal[False] = ...
    ): ...

    def __init__(self, as_uuid: bool = True):
        """Construct a :class:`_mssql.UNIQUEIDENTIFIER` type.


        :param as_uuid=True: if True, values will be interpreted
         as Python uuid objects, converting to/from string via the
         DBAPI.

         .. versionchanged: 2.0 Added direct "uuid" support to the
            :class:`_mssql.UNIQUEIDENTIFIER` datatype; uuid interpretation
            defaults to ``True``.

        """
        self.as_uuid = as_uuid
        self.native_uuid = True


class SQL_VARIANT(sqltypes.TypeEngine):
    __visit_name__ = "SQL_VARIANT"


# old names.
MSDateTime = _MSDateTime
MSDate = _MSDate
MSReal = REAL
MSTinyInteger = TINYINT
MSTime = TIME
MSSmallDateTime = SMALLDATETIME
MSDateTime2 = DATETIME2
MSDateTimeOffset = DATETIMEOFFSET
MSText = TEXT
MSNText = NTEXT
MSString = VARCHAR
MSNVarchar = NVARCHAR
MSChar = CHAR
MSNChar = NCHAR
MSBinary = BINARY
MSVarBinary = VARBINARY
MSImage = IMAGE
MSBit = BIT
MSMoney = MONEY
MSSmallMoney = SMALLMONEY
MSUniqueIdentifier = UNIQUEIDENTIFIER
MSVariant = SQL_VARIANT

ischema_names = {
    "int": INTEGER,
    "bigint": BIGINT,
    "smallint": SMALLINT,
    "tinyint": TINYINT,
    "varchar": VARCHAR,
    "nvarchar": NVARCHAR,
    "char": CHAR,
    "nchar": NCHAR,
    "text": TEXT,
    "ntext": NTEXT,
    "decimal": DECIMAL,
    "numeric": NUMERIC,
    "float": FLOAT,
    "datetime": DATETIME,
    "datetime2": DATETIME2,
    "datetimeoffset": DATETIMEOFFSET,
    "date": DATE,
    "time": TIME,
    "smalldatetime": SMALLDATETIME,
    "binary": BINARY,
    "varbinary": VARBINARY,
    "bit": BIT,
    "real": REAL,
    "double precision": DOUBLE_PRECISION,
    "image": IMAGE,
    "xml": XML,
    "timestamp": TIMESTAMP,
    "money": MONEY,
    "smallmoney": SMALLMONEY,
    "uniqueidentifier": UNIQUEIDENTIFIER,
    "sql_variant": SQL_VARIANT,
}


class MSTypeCompiler(compiler.GenericTypeCompiler):
    def _extend(self, spec, type_, length=None):
        """Extend a string-type declaration with standard SQL
        COLLATE annotations.

        """

        if getattr(type_, "collation", None):
            collation = "COLLATE %s" % type_.collation
        else:
            collation = None

        if not length:
            length = type_.length

        if length:
            spec = spec + "(%s)" % length

        return " ".join([c for c in (spec, collation) if c is not None])

    def visit_double(self, type_, **kw):
        return self.visit_DOUBLE_PRECISION(type_, **kw)

    def visit_FLOAT(self, type_, **kw):
        precision = getattr(type_, "precision", None)
        if precision is None:
            return "FLOAT"
        else:
            return "FLOAT(%(precision)s)" % {"precision": precision}

    def visit_TINYINT(self, type_, **kw):
        return "TINYINT"

    def visit_TIME(self, type_, **kw):
        precision = getattr(type_, "precision", None)
        if precision is not None:
            return "TIME(%s)" % precision
        else:
            return "TIME"

    def visit_TIMESTAMP(self, type_, **kw):
        return "TIMESTAMP"

    def visit_ROWVERSION(self, type_, **kw):
        return "ROWVERSION"

    def visit_datetime(self, type_, **kw):
        if type_.timezone:
            return self.visit_DATETIMEOFFSET(type_, **kw)
        else:
            return self.visit_DATETIME(type_, **kw)

    def visit_DATETIMEOFFSET(self, type_, **kw):
        precision = getattr(type_, "precision", None)
        if precision is not None:
            return "DATETIMEOFFSET(%s)" % type_.precision
        else:
            return "DATETIMEOFFSET"

    def visit_DATETIME2(self, type_, **kw):
        precision = getattr(type_, "precision", None)
        if precision is not None:
            return "DATETIME2(%s)" % precision
        else:
            return "DATETIME2"

    def visit_SMALLDATETIME(self, type_, **kw):
        return "SMALLDATETIME"

    def visit_unicode(self, type_, **kw):
        return self.visit_NVARCHAR(type_, **kw)

    def visit_text(self, type_, **kw):
        if self.dialect.deprecate_large_types:
            return self.visit_VARCHAR(type_, **kw)
        else:
            return self.visit_TEXT(type_, **kw)

    def visit_unicode_text(self, type_, **kw):
        if self.dialect.deprecate_large_types:
            return self.visit_NVARCHAR(type_, **kw)
        else:
            return self.visit_NTEXT(type_, **kw)

    def visit_NTEXT(self, type_, **kw):
        return self._extend("NTEXT", type_)

    def visit_TEXT(self, type_, **kw):
        return self._extend("TEXT", type_)

    def visit_VARCHAR(self, type_, **kw):
        return self._extend("VARCHAR", type_, length=type_.length or "max")

    def visit_CHAR(self, type_, **kw):
        return self._extend("CHAR", type_)

    def visit_NCHAR(self, type_, **kw):
        return self._extend("NCHAR", type_)

    def visit_NVARCHAR(self, type_, **kw):
        return self._extend("NVARCHAR", type_, length=type_.length or "max")

    def visit_date(self, type_, **kw):
        if self.dialect.server_version_info < MS_2008_VERSION:
            return self.visit_DATETIME(type_, **kw)
        else:
            return self.visit_DATE(type_, **kw)

    def visit__BASETIMEIMPL(self, type_, **kw):
        return self.visit_time(type_, **kw)

    def visit_time(self, type_, **kw):
        if self.dialect.server_version_info < MS_2008_VERSION:
            return self.visit_DATETIME(type_, **kw)
        else:
            return self.visit_TIME(type_, **kw)

    def visit_large_binary(self, type_, **kw):
        if self.dialect.deprecate_large_types:
            return self.visit_VARBINARY(type_, **kw)
        else:
            return self.visit_IMAGE(type_, **kw)

    def visit_IMAGE(self, type_, **kw):
        return "IMAGE"

    def visit_XML(self, type_, **kw):
        return "XML"

    def visit_VARBINARY(self, type_, **kw):
        text = self._extend("VARBINARY", type_, length=type_.length or "max")
        if getattr(type_, "filestream", False):
            text += " FILESTREAM"
        return text

    def visit_boolean(self, type_, **kw):
        return self.visit_BIT(type_)

    def visit_BIT(self, type_, **kw):
        return "BIT"

    def visit_JSON(self, type_, **kw):
        # this is a bit of a break with SQLAlchemy's convention of
        # "UPPERCASE name goes to UPPERCASE type name with no modification"
        return self._extend("NVARCHAR", type_, length="max")

    def visit_MONEY(self, type_, **kw):
        return "MONEY"

    def visit_SMALLMONEY(self, type_, **kw):
        return "SMALLMONEY"

    def visit_uuid(self, type_, **kw):
        if type_.native_uuid:
            return self.visit_UNIQUEIDENTIFIER(type_, **kw)
        else:
            return super().visit_uuid(type_, **kw)

    def visit_UNIQUEIDENTIFIER(self, type_, **kw):
        return "UNIQUEIDENTIFIER"

    def visit_SQL_VARIANT(self, type_, **kw):
        return "SQL_VARIANT"


class MSExecutionContext(default.DefaultExecutionContext):
    _enable_identity_insert = False
    _select_lastrowid = False
    _lastrowid = None

    dialect: MSDialect

    def _opt_encode(self, statement):
        if self.compiled and self.compiled.schema_translate_map:
            rst = self.compiled.preparer._render_schema_translates
            statement = rst(statement, self.compiled.schema_translate_map)

        return statement

    def pre_exec(self):
        """Activate IDENTITY_INSERT if needed."""

        if self.isinsert:
            if TYPE_CHECKING:
                assert is_sql_compiler(self.compiled)
                assert isinstance(self.compiled.compile_state, DMLState)
                assert isinstance(
                    self.compiled.compile_state.dml_table, TableClause
                )

            tbl = self.compiled.compile_state.dml_table
            id_column = tbl._autoincrement_column

            if id_column is not None and (
                not isinstance(id_column.default, Sequence)
            ):
                insert_has_identity = True
                compile_state = self.compiled.dml_compile_state
                self._enable_identity_insert = (
                    id_column.key in self.compiled_parameters[0]
                ) or (
                    compile_state._dict_parameters
                    and (id_column.key in compile_state._insert_col_keys)
                )

            else:
                insert_has_identity = False
                self._enable_identity_insert = False

            self._select_lastrowid = (
                not self.compiled.inline
                and insert_has_identity
                and not self.compiled.effective_returning
                and not self._enable_identity_insert
                and not self.executemany
            )

            if self._enable_identity_insert:
                self.root_connection._cursor_execute(
                    self.cursor,
                    self._opt_encode(
                        "SET IDENTITY_INSERT %s ON"
                        % self.identifier_preparer.format_table(tbl)
                    ),
                    (),
                    self,
                )

    def post_exec(self):
        """Disable IDENTITY_INSERT if enabled."""

        conn = self.root_connection

        if self.isinsert or self.isupdate or self.isdelete:
            self._rowcount = self.cursor.rowcount

        if self._select_lastrowid:
            if self.dialect.use_scope_identity:
                conn._cursor_execute(
                    self.cursor,
                    "SELECT scope_identity() AS lastrowid",
                    (),
                    self,
                )
            else:
                conn._cursor_execute(
                    self.cursor, "SELECT @@identity AS lastrowid", (), self
                )
            # fetchall() ensures the cursor is consumed without closing it
            row = self.cursor.fetchall()[0]
            self._lastrowid = int(row[0])

            self.cursor_fetch_strategy = _cursor._NO_CURSOR_DML
        elif (
            self.compiled is not None
            and is_sql_compiler(self.compiled)
            and self.compiled.effective_returning
        ):
            self.cursor_fetch_strategy = (
                _cursor.FullyBufferedCursorFetchStrategy(
                    self.cursor,
                    self.cursor.description,
                    self.cursor.fetchall(),
                )
            )

        if self._enable_identity_insert:
            if TYPE_CHECKING:
                assert is_sql_compiler(self.compiled)
                assert isinstance(self.compiled.compile_state, DMLState)
                assert isinstance(
                    self.compiled.compile_state.dml_table, TableClause
                )
            conn._cursor_execute(
                self.cursor,
                self._opt_encode(
                    "SET IDENTITY_INSERT %s OFF"
                    % self.identifier_preparer.format_table(
                        self.compiled.compile_state.dml_table
                    )
                ),
                (),
                self,
            )

    def get_lastrowid(self):
        return self._lastrowid

    def handle_dbapi_exception(self, e):
        if self._enable_identity_insert:
            try:
                self.cursor.execute(
                    self._opt_encode(
                        "SET IDENTITY_INSERT %s OFF"
                        % self.identifier_preparer.format_table(
                            self.compiled.compile_state.dml_table
                        )
                    )
                )
            except Exception:
                pass

    def fire_sequence(self, seq, type_):
        return self._execute_scalar(
            (
                "SELECT NEXT VALUE FOR %s"
                % self.identifier_preparer.format_sequence(seq)
            ),
            type_,
        )

    def get_insert_default(self, column):
        if (
            isinstance(column, sa_schema.Column)
            and column is column.table._autoincrement_column
            and isinstance(column.default, sa_schema.Sequence)
            and column.default.optional
        ):
            return None
        return super().get_insert_default(column)


class MSSQLCompiler(compiler.SQLCompiler):
    returning_precedes_values = True

    extract_map = util.update_copy(
        compiler.SQLCompiler.extract_map,
        {
            "doy": "dayofyear",
            "dow": "weekday",
            "milliseconds": "millisecond",
            "microseconds": "microsecond",
        },
    )

    def __init__(self, *args, **kwargs):
        self.tablealiases = {}
        super().__init__(*args, **kwargs)

    def _format_frame_clause(self, range_, **kw):
        kw["literal_execute"] = True
        return super()._format_frame_clause(range_, **kw)

    def _with_legacy_schema_aliasing(fn):
        def decorate(self, *arg, **kw):
            if self.dialect.legacy_schema_aliasing:
                return fn(self, *arg, **kw)
            else:
                super_ = getattr(super(MSSQLCompiler, self), fn.__name__)
                return super_(*arg, **kw)

        return decorate

    def visit_now_func(self, fn, **kw):
        return "CURRENT_TIMESTAMP"

    def visit_current_date_func(self, fn, **kw):
        return "GETDATE()"

    def visit_length_func(self, fn, **kw):
        return "LEN%s" % self.function_argspec(fn, **kw)

    def visit_char_length_func(self, fn, **kw):
        return "LEN%s" % self.function_argspec(fn, **kw)

    def visit_aggregate_strings_func(self, fn, **kw):
        expr = fn.clauses.clauses[0]._compiler_dispatch(self, **kw)
        kw["literal_execute"] = True
        delimeter = fn.clauses.clauses[1]._compiler_dispatch(self, **kw)
        return f"string_agg({expr}, {delimeter})"

    def visit_concat_op_expression_clauselist(
        self, clauselist, operator, **kw
    ):
        return " + ".join(self.process(elem, **kw) for elem in clauselist)

    def visit_concat_op_binary(self, binary, operator, **kw):
        return "%s + %s" % (
            self.process(binary.left, **kw),
            self.process(binary.right, **kw),
        )

    def visit_true(self, expr, **kw):
        return "1"

    def visit_false(self, expr, **kw):
        return "0"

    def visit_match_op_binary(self, binary, operator, **kw):
        return "CONTAINS (%s, %s)" % (
            self.process(binary.left, **kw),
            self.process(binary.right, **kw),
        )

    def get_select_precolumns(self, select, **kw):
        """MS-SQL puts TOP, it's version of LIMIT here"""

        s = super().get_select_precolumns(select, **kw)

        if select._has_row_limiting_clause and self._use_top(select):
            # ODBC drivers and possibly others
            # don't support bind params in the SELECT clause on SQL Server.
            # so have to use literal here.
            kw["literal_execute"] = True
            s += "TOP %s " % self.process(
                self._get_limit_or_fetch(select), **kw
            )
            if select._fetch_clause is not None:
                if select._fetch_clause_options["percent"]:
                    s += "PERCENT "
                if select._fetch_clause_options["with_ties"]:
                    s += "WITH TIES "

        return s

    def get_from_hint_text(self, table, text):
        return text

    def get_crud_hint_text(self, table, text):
        return text

    def _get_limit_or_fetch(self, select):
        if select._fetch_clause is None:
            return select._limit_clause
        else:
            return select._fetch_clause

    def _use_top(self, select):
        return (select._offset_clause is None) and (
            select._simple_int_clause(select._limit_clause)
            or (
                # limit can use TOP with is by itself. fetch only uses TOP
                # when it needs to because of PERCENT and/or WITH TIES
                # TODO: Why?  shouldn't we use TOP always ?
                select._simple_int_clause(select._fetch_clause)
                and (
                    select._fetch_clause_options["percent"]
                    or select._fetch_clause_options["with_ties"]
                )
            )
        )

    def limit_clause(self, cs, **kwargs):
        return ""

    def _check_can_use_fetch_limit(self, select):
        # to use ROW_NUMBER(), an ORDER BY is required.
        # OFFSET are FETCH are options of the ORDER BY clause
        if not select._order_by_clause.clauses:
            raise exc.CompileError(
                "MSSQL requires an order_by when "
                "using an OFFSET or a non-simple "
                "LIMIT clause"
            )

        if select._fetch_clause_options is not None and (
            select._fetch_clause_options["percent"]
            or select._fetch_clause_options["with_ties"]
        ):
            raise exc.CompileError(
                "MSSQL needs TOP to use PERCENT and/or WITH TIES. "
                "Only simple fetch without offset can be used."
            )

    def _row_limit_clause(self, select, **kw):
        """MSSQL 2012 supports OFFSET/FETCH operators
        Use it instead subquery with row_number

        """

        if self.dialect._supports_offset_fetch and not self._use_top(select):
            self._check_can_use_fetch_limit(select)

            return self.fetch_clause(
                select,
                fetch_clause=self._get_limit_or_fetch(select),
                require_offset=True,
                **kw,
            )

        else:
            return ""

    def visit_try_cast(self, element, **kw):
        return "TRY_CAST (%s AS %s)" % (
            self.process(element.clause, **kw),
            self.process(element.typeclause, **kw),
        )

    def translate_select_structure(self, select_stmt, **kwargs):
        """Look for ``LIMIT`` and OFFSET in a select statement, and if
        so tries to wrap it in a subquery with ``row_number()`` criterion.
        MSSQL 2012 and above are excluded

        """
        select = select_stmt

        if (
            select._has_row_limiting_clause
            and not self.dialect._supports_offset_fetch
            and not self._use_top(select)
            and not getattr(select, "_mssql_visit", None)
        ):
            self._check_can_use_fetch_limit(select)

            _order_by_clauses = [
                sql_util.unwrap_label_reference(elem)
                for elem in select._order_by_clause.clauses
            ]

            limit_clause = self._get_limit_or_fetch(select)
            offset_clause = select._offset_clause

            select = select._generate()
            select._mssql_visit = True
            select = (
                select.add_columns(
                    sql.func.ROW_NUMBER()
                    .over(order_by=_order_by_clauses)
                    .label("mssql_rn")
                )
                .order_by(None)
                .alias()
            )

            mssql_rn = sql.column("mssql_rn")
            limitselect = sql.select(
                *[c for c in select.c if c.key != "mssql_rn"]
            )
            if offset_clause is not None:
                limitselect = limitselect.where(mssql_rn > offset_clause)
                if limit_clause is not None:
                    limitselect = limitselect.where(
                        mssql_rn <= (limit_clause + offset_clause)
                    )
            else:
                limitselect = limitselect.where(mssql_rn <= (limit_clause))
            return limitselect
        else:
            return select

    @_with_legacy_schema_aliasing
    def visit_table(self, table, mssql_aliased=False, iscrud=False, **kwargs):
        if mssql_aliased is table or iscrud:
            return super().visit_table(table, **kwargs)

        # alias schema-qualified tables
        alias = self._schema_aliased_table(table)
        if alias is not None:
            return self.process(alias, mssql_aliased=table, **kwargs)
        else:
            return super().visit_table(table, **kwargs)

    @_with_legacy_schema_aliasing
    def visit_alias(self, alias, **kw):
        # translate for schema-qualified table aliases
        kw["mssql_aliased"] = alias.element
        return super().visit_alias(alias, **kw)

    @_with_legacy_schema_aliasing
    def visit_column(self, column, add_to_result_map=None, **kw):
        if (
            column.table is not None
            and (not self.isupdate and not self.isdelete)
            or self.is_subquery()
        ):
            # translate for schema-qualified table aliases
            t = self._schema_aliased_table(column.table)
            if t is not None:
                converted = elements._corresponding_column_or_error(t, column)
                if add_to_result_map is not None:
                    add_to_result_map(
                        column.name,
                        column.name,
                        (column, column.name, column.key),
                        column.type,
                    )

                return super().visit_column(converted, **kw)

        return super().visit_column(
            column, add_to_result_map=add_to_result_map, **kw
        )

    def _schema_aliased_table(self, table):
        if getattr(table, "schema", None) is not None:
            if table not in self.tablealiases:
                self.tablealiases[table] = table.alias()
            return self.tablealiases[table]
        else:
            return None

    def visit_extract(self, extract, **kw):
        field = self.extract_map.get(extract.field, extract.field)
        return "DATEPART(%s, %s)" % (field, self.process(extract.expr, **kw))

    def visit_savepoint(self, savepoint_stmt, **kw):
        return "SAVE TRANSACTION %s" % self.preparer.format_savepoint(
            savepoint_stmt
        )

    def visit_rollback_to_savepoint(self, savepoint_stmt, **kw):
        return "ROLLBACK TRANSACTION %s" % self.preparer.format_savepoint(
            savepoint_stmt
        )

    def visit_binary(self, binary, **kwargs):
        """Move bind parameters to the right-hand side of an operator, where
        possible.

        """
        if (
            isinstance(binary.left, expression.BindParameter)
            and binary.operator == operator.eq
            and not isinstance(binary.right, expression.BindParameter)
        ):
            return self.process(
                expression.BinaryExpression(
                    binary.right, binary.left, binary.operator
                ),
                **kwargs,
            )
        return super().visit_binary(binary, **kwargs)

    def returning_clause(
        self, stmt, returning_cols, *, populate_result_map, **kw
    ):
        # SQL server returning clause requires that the columns refer to
        # the virtual table names "inserted" or "deleted".   Here, we make
        # a simple alias of our table with that name, and then adapt the
        # columns we have from the list of RETURNING columns to that new name
        # so that they render as "inserted.<colname>" / "deleted.<colname>".

        if stmt.is_insert or stmt.is_update:
            target = stmt.table.alias("inserted")
        elif stmt.is_delete:
            target = stmt.table.alias("deleted")
        else:
            assert False, "expected Insert, Update or Delete statement"

        adapter = sql_util.ClauseAdapter(target)

        # adapter.traverse() takes a column from our target table and returns
        # the one that is linked to the "inserted" / "deleted" tables.  So  in
        # order to retrieve these values back from the result  (e.g. like
        # row[column]), tell the compiler to also add the original unadapted
        # column to the result map.   Before #4877, these were  (unknowingly)
        # falling back using string name matching in the result set which
        # necessarily used an expensive KeyError in order to match.

        columns = [
            self._label_returning_column(
                stmt,
                adapter.traverse(column),
                populate_result_map,
                {"result_map_targets": (column,)},
                fallback_label_name=fallback_label_name,
                column_is_repeated=repeated,
                name=name,
                proxy_name=proxy_name,
                **kw,
            )
            for (
                name,
                proxy_name,
                fallback_label_name,
                column,
                repeated,
            ) in stmt._generate_columns_plus_names(
                True, cols=expression._select_iterables(returning_cols)
            )
        ]

        return "OUTPUT " + ", ".join(columns)

    def get_cte_preamble(self, recursive):
        # SQL Server finds it too inconvenient to accept
        # an entirely optional, SQL standard specified,
        # "RECURSIVE" word with their "WITH",
        # so here we go
        return "WITH"

    def label_select_column(self, select, column, asfrom):
        if isinstance(column, expression.Function):
            return column.label(None)
        else:
            return super().label_select_column(select, column, asfrom)

    def for_update_clause(self, select, **kw):
        # "FOR UPDATE" is only allowed on "DECLARE CURSOR" which
        # SQLAlchemy doesn't use
        return ""

    def order_by_clause(self, select, **kw):
        # MSSQL only allows ORDER BY in subqueries if there is a LIMIT:
        # "The ORDER BY clause is invalid in views, inline functions,
        # derived tables, subqueries, and common table expressions,
        # unless TOP, OFFSET or FOR XML is also specified."
        if (
            self.is_subquery()
            and not self._use_top(select)
            and (
                select._offset is None
                or not self.dialect._supports_offset_fetch
            )
        ):
            # avoid processing the order by clause if we won't end up
            # using it, because we don't want all the bind params tacked
            # onto the positional list if that is what the dbapi requires
            return ""

        order_by = self.process(select._order_by_clause, **kw)

        if order_by:
            return " ORDER BY " + order_by
        else:
            return ""

    def update_from_clause(
        self, update_stmt, from_table, extra_froms, from_hints, **kw
    ):
        """Render the UPDATE..FROM clause specific to MSSQL.

        In MSSQL, if the UPDATE statement involves an alias of the table to
        be updated, then the table itself must be added to the FROM list as
        well. Otherwise, it is optional. Here, we add it regardless.

        """
        return "FROM " + ", ".join(
            t._compiler_dispatch(self, asfrom=True, fromhints=from_hints, **kw)
            for t in [from_table] + extra_froms
        )

    def delete_table_clause(self, delete_stmt, from_table, extra_froms, **kw):
        """If we have extra froms make sure we render any alias as hint."""
        ashint = False
        if extra_froms:
            ashint = True
        return from_table._compiler_dispatch(
            self, asfrom=True, iscrud=True, ashint=ashint, **kw
        )

    def delete_extra_from_clause(
        self, delete_stmt, from_table, extra_froms, from_hints, **kw
    ):
        """Render the DELETE .. FROM clause specific to MSSQL.

        Yes, it has the FROM keyword twice.

        """
        return "FROM " + ", ".join(
            t._compiler_dispatch(self, asfrom=True, fromhints=from_hints, **kw)
            for t in [from_table] + extra_froms
        )

    def visit_empty_set_expr(self, type_, **kw):
        return "SELECT 1 WHERE 1!=1"

    def visit_is_distinct_from_binary(self, binary, operator, **kw):
        return "NOT EXISTS (SELECT %s INTERSECT SELECT %s)" % (
            self.process(binary.left),
            self.process(binary.right),
        )

    def visit_is_not_distinct_from_binary(self, binary, operator, **kw):
        return "EXISTS (SELECT %s INTERSECT SELECT %s)" % (
            self.process(binary.left),
            self.process(binary.right),
        )

    def _render_json_extract_from_binary(self, binary, operator, **kw):
        # note we are intentionally calling upon the process() calls in the
        # order in which they appear in the SQL String as this is used
        # by positional parameter rendering

        if binary.type._type_affinity is sqltypes.JSON:
            return "JSON_QUERY(%s, %s)" % (
                self.process(binary.left, **kw),
                self.process(binary.right, **kw),
            )

        # as with other dialects, start with an explicit test for NULL
        case_expression = "CASE JSON_VALUE(%s, %s) WHEN NULL THEN NULL" % (
            self.process(binary.left, **kw),
            self.process(binary.right, **kw),
        )

        if binary.type._type_affinity is sqltypes.Integer:
            type_expression = "ELSE CAST(JSON_VALUE(%s, %s) AS INTEGER)" % (
                self.process(binary.left, **kw),
                self.process(binary.right, **kw),
            )
        elif binary.type._type_affinity is sqltypes.Numeric:
            type_expression = "ELSE CAST(JSON_VALUE(%s, %s) AS %s)" % (
                self.process(binary.left, **kw),
                self.process(binary.right, **kw),
                (
                    "FLOAT"
                    if isinstance(binary.type, sqltypes.Float)
                    else "NUMERIC(%s, %s)"
                    % (binary.type.precision, binary.type.scale)
                ),
            )
        elif binary.type._type_affinity is sqltypes.Boolean:
            # the NULL handling is particularly weird with boolean, so
            # explicitly return numeric (BIT) constants
            type_expression = (
                "WHEN 'true' THEN 1 WHEN 'false' THEN 0 ELSE NULL"
            )
        elif binary.type._type_affinity is sqltypes.String:
            # TODO: does this comment (from mysql) apply to here, too?
            #       this fails with a JSON value that's a four byte unicode
            #       string.  SQLite has the same problem at the moment
            type_expression = "ELSE JSON_VALUE(%s, %s)" % (
                self.process(binary.left, **kw),
                self.process(binary.right, **kw),
            )
        else:
            # other affinity....this is not expected right now
            type_expression = "ELSE JSON_QUERY(%s, %s)" % (
                self.process(binary.left, **kw),
                self.process(binary.right, **kw),
            )

        return case_expression + " " + type_expression + " END"

    def visit_json_getitem_op_binary(self, binary, operator, **kw):
        return self._render_json_extract_from_binary(binary, operator, **kw)

    def visit_json_path_getitem_op_binary(self, binary, operator, **kw):
        return self._render_json_extract_from_binary(binary, operator, **kw)

    def visit_sequence(self, seq, **kw):
        return "NEXT VALUE FOR %s" % self.preparer.format_sequence(seq)


class MSSQLStrictCompiler(MSSQLCompiler):
    """A subclass of MSSQLCompiler which disables the usage of bind
    parameters where not allowed natively by MS-SQL.

    A dialect may use this compiler on a platform where native
    binds are used.

    """

    ansi_bind_rules = True

    def visit_in_op_binary(self, binary, operator, **kw):
        kw["literal_execute"] = True
        return "%s IN %s" % (
            self.process(binary.left, **kw),
            self.process(binary.right, **kw),
        )

    def visit_not_in_op_binary(self, binary, operator, **kw):
        kw["literal_execute"] = True
        return "%s NOT IN %s" % (
            self.process(binary.left, **kw),
            self.process(binary.right, **kw),
        )

    def render_literal_value(self, value, type_):
        """
        For date and datetime values, convert to a string
        format acceptable to MSSQL. That seems to be the
        so-called ODBC canonical date format which looks
        like this:

            yyyy-mm-dd hh:mi:ss.mmm(24h)

        For other data types, call the base class implementation.
        """
        # datetime and date are both subclasses of datetime.date
        if issubclass(type(value), datetime.date):
            # SQL Server wants single quotes around the date string.
            return "'" + str(value) + "'"
        else:
            return super().render_literal_value(value, type_)


class MSDDLCompiler(compiler.DDLCompiler):
    def get_column_specification(self, column, **kwargs):
        colspec = self.preparer.format_column(column)

        # type is not accepted in a computed column
        if column.computed is not None:
            colspec += " " + self.process(column.computed)
        else:
            colspec += " " + self.dialect.type_compiler_instance.process(
                column.type, type_expression=column
            )

        if column.nullable is not None:
            if (
                not column.nullable
                or column.primary_key
                or isinstance(column.default, sa_schema.Sequence)
                or column.autoincrement is True
                or column.identity
            ):
                colspec += " NOT NULL"
            elif column.computed is None:
                # don't specify "NULL" for computed columns
                colspec += " NULL"

        if column.table is None:
            raise exc.CompileError(
                "mssql requires Table-bound columns "
                "in order to generate DDL"
            )

        d_opt = column.dialect_options["mssql"]
        start = d_opt["identity_start"]
        increment = d_opt["identity_increment"]
        if start is not None or increment is not None:
            if column.identity:
                raise exc.CompileError(
                    "Cannot specify options 'mssql_identity_start' and/or "
                    "'mssql_identity_increment' while also using the "
                    "'Identity' construct."
                )
            util.warn_deprecated(
                "The dialect options 'mssql_identity_start' and "
                "'mssql_identity_increment' are deprecated. "
                "Use the 'Identity' object instead.",
                "1.4",
            )

        if column.identity:
            colspec += self.process(column.identity, **kwargs)
        elif (
            column is column.table._autoincrement_column
            or column.autoincrement is True
        ) and (
            not isinstance(column.default, Sequence) or column.default.optional
        ):
            colspec += self.process(Identity(start=start, increment=increment))
        else:
            default = self.get_column_default_string(column)
            if default is not None:
                colspec += " DEFAULT " + default

        return colspec

    def visit_create_index(self, create, include_schema=False, **kw):
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        text = "CREATE "
        if index.unique:
            text += "UNIQUE "

        # handle clustering option
        clustered = index.dialect_options["mssql"]["clustered"]
        if clustered is not None:
            if clustered:
                text += "CLUSTERED "
            else:
                text += "NONCLUSTERED "

        # handle columnstore option (has no negative value)
        columnstore = index.dialect_options["mssql"]["columnstore"]
        if columnstore:
            text += "COLUMNSTORE "

        text += "INDEX %s ON %s" % (
            self._prepared_index_name(index, include_schema=include_schema),
            preparer.format_table(index.table),
        )

        # in some case mssql allows indexes with no columns defined
        if len(index.expressions) > 0:
            text += " (%s)" % ", ".join(
                self.sql_compiler.process(
                    expr, include_table=False, literal_binds=True
                )
                for expr in index.expressions
            )

        # handle other included columns
        if index.dialect_options["mssql"]["include"]:
            inclusions = [
                index.table.c[col] if isinstance(col, str) else col
                for col in index.dialect_options["mssql"]["include"]
            ]

            text += " INCLUDE (%s)" % ", ".join(
                [preparer.quote(c.name) for c in inclusions]
            )

        whereclause = index.dialect_options["mssql"]["where"]

        if whereclause is not None:
            whereclause = coercions.expect(
                roles.DDLExpressionRole, whereclause
            )

            where_compiled = self.sql_compiler.process(
                whereclause, include_table=False, literal_binds=True
            )
            text += " WHERE " + where_compiled

        return text

    def visit_drop_index(self, drop, **kw):
        return "\nDROP INDEX %s ON %s" % (
            self._prepared_index_name(drop.element, include_schema=False),
            self.preparer.format_table(drop.element.table),
        )

    def visit_primary_key_constraint(self, constraint, **kw):
        if len(constraint) == 0:
            return ""
        text = ""
        if constraint.name is not None:
            text += "CONSTRAINT %s " % self.preparer.format_constraint(
                constraint
            )
        text += "PRIMARY KEY "

        clustered = constraint.dialect_options["mssql"]["clustered"]
        if clustered is not None:
            if clustered:
                text += "CLUSTERED "
            else:
                text += "NONCLUSTERED "

        text += "(%s)" % ", ".join(
            self.preparer.quote(c.name) for c in constraint
        )
        text += self.define_constraint_deferrability(constraint)
        return text

    def visit_unique_constraint(self, constraint, **kw):
        if len(constraint) == 0:
            return ""
        text = ""
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += "CONSTRAINT %s " % formatted_name
        text += "UNIQUE %s" % self.define_unique_constraint_distinct(
            constraint, **kw
        )
        clustered = constraint.dialect_options["mssql"]["clustered"]
        if clustered is not None:
            if clustered:
                text += "CLUSTERED "
            else:
                text += "NONCLUSTERED "

        text += "(%s)" % ", ".join(
            self.preparer.quote(c.name) for c in constraint
        )
        text += self.define_constraint_deferrability(constraint)
        return text

    def visit_computed_column(self, generated, **kw):
        text = "AS (%s)" % self.sql_compiler.process(
            generated.sqltext, include_table=False, literal_binds=True
        )
        # explicitly check for True|False since None means server default
        if generated.persisted is True:
            text += " PERSISTED"
        return text

    def visit_set_table_comment(self, create, **kw):
        schema = self.preparer.schema_for_object(create.element)
        schema_name = schema if schema else self.dialect.default_schema_name
        return (
            "execute sp_addextendedproperty 'MS_Description', "
            "{}, 'schema', {}, 'table', {}".format(
                self.sql_compiler.render_literal_value(
                    create.element.comment, sqltypes.NVARCHAR()
                ),
                self.preparer.quote_schema(schema_name),
                self.preparer.format_table(create.element, use_schema=False),
            )
        )

    def visit_drop_table_comment(self, drop, **kw):
        schema = self.preparer.schema_for_object(drop.element)
        schema_name = schema if schema else self.dialect.default_schema_name
        return (
            "execute sp_dropextendedproperty 'MS_Description', 'schema', "
            "{}, 'table', {}".format(
                self.preparer.quote_schema(schema_name),
                self.preparer.format_table(drop.element, use_schema=False),
            )
        )

    def visit_set_column_comment(self, create, **kw):
        schema = self.preparer.schema_for_object(create.element.table)
        schema_name = schema if schema else self.dialect.default_schema_name
        return (
            "execute sp_addextendedproperty 'MS_Description', "
            "{}, 'schema', {}, 'table', {}, 'column', {}".format(
                self.sql_compiler.render_literal_value(
                    create.element.comment, sqltypes.NVARCHAR()
                ),
                self.preparer.quote_schema(schema_name),
                self.preparer.format_table(
                    create.element.table, use_schema=False
                ),
                self.preparer.format_column(create.element),
            )
        )

    def visit_drop_column_comment(self, drop, **kw):
        schema = self.preparer.schema_for_object(drop.element.table)
        schema_name = schema if schema else self.dialect.default_schema_name
        return (
            "execute sp_dropextendedproperty 'MS_Description', 'schema', "
            "{}, 'table', {}, 'column', {}".format(
                self.preparer.quote_schema(schema_name),
                self.preparer.format_table(
                    drop.element.table, use_schema=False
                ),
                self.preparer.format_column(drop.element),
            )
        )

    def visit_create_sequence(self, create, **kw):
        prefix = None
        if create.element.data_type is not None:
            data_type = create.element.data_type
            prefix = " AS %s" % self.type_compiler.process(data_type)
        return super().visit_create_sequence(create, prefix=prefix, **kw)

    def visit_identity_column(self, identity, **kw):
        text = " IDENTITY"
        if identity.start is not None or identity.increment is not None:
            start = 1 if identity.start is None else identity.start
            increment = 1 if identity.increment is None else identity.increment
            text += "(%s,%s)" % (start, increment)
        return text


class MSIdentifierPreparer(compiler.IdentifierPreparer):
    reserved_words = RESERVED_WORDS

    def __init__(self, dialect):
        super().__init__(
            dialect,
            initial_quote="[",
            final_quote="]",
            quote_case_sensitive_collations=False,
        )

    def _escape_identifier(self, value):
        return value.replace("]", "]]")

    def _unescape_identifier(self, value):
        return value.replace("]]", "]")

    def quote_schema(self, schema, force=None):
        """Prepare a quoted table and schema name."""

        # need to re-implement the deprecation warning entirely
        if force is not None:
            # not using the util.deprecated_params() decorator in this
            # case because of the additional function call overhead on this
            # very performance-critical spot.
            util.warn_deprecated(
                "The IdentifierPreparer.quote_schema.force parameter is "
                "deprecated and will be removed in a future release.  This "
                "flag has no effect on the behavior of the "
                "IdentifierPreparer.quote method; please refer to "
                "quoted_name().",
                version="1.3",
            )

        dbname, owner = _schema_elements(schema)
        if dbname:
            result = "%s.%s" % (self.quote(dbname), self.quote(owner))
        elif owner:
            result = self.quote(owner)
        else:
            result = ""
        return result


def _db_plus_owner_listing(fn):
    def wrap(dialect, connection, schema=None, **kw):
        dbname, owner = _owner_plus_db(dialect, schema)
        return _switch_db(
            dbname,
            connection,
            fn,
            dialect,
            connection,
            dbname,
            owner,
            schema,
            **kw,
        )

    return update_wrapper(wrap, fn)


def _db_plus_owner(fn):
    def wrap(dialect, connection, tablename, schema=None, **kw):
        dbname, owner = _owner_plus_db(dialect, schema)
        return _switch_db(
            dbname,
            connection,
            fn,
            dialect,
            connection,
            tablename,
            dbname,
            owner,
            schema,
            **kw,
        )

    return update_wrapper(wrap, fn)


def _switch_db(dbname, connection, fn, *arg, **kw):
    if dbname:
        current_db = connection.exec_driver_sql("select db_name()").scalar()
        if current_db != dbname:
            connection.exec_driver_sql(
                "use %s" % connection.dialect.identifier_preparer.quote(dbname)
            )
    try:
        return fn(*arg, **kw)
    finally:
        if dbname and current_db != dbname:
            connection.exec_driver_sql(
                "use %s"
                % connection.dialect.identifier_preparer.quote(current_db)
            )


def _owner_plus_db(dialect, schema):
    if not schema:
        return None, dialect.default_schema_name
    else:
        return _schema_elements(schema)


_memoized_schema = util.LRUCache()


def _schema_elements(schema):
    if isinstance(schema, quoted_name) and schema.quote:
        return None, schema

    if schema in _memoized_schema:
        return _memoized_schema[schema]

    # tests for this function are in:
    # test/dialect/mssql/test_reflection.py ->
    #           OwnerPlusDBTest.test_owner_database_pairs
    # test/dialect/mssql/test_compiler.py -> test_force_schema_*
    # test/dialect/mssql/test_compiler.py -> test_schema_many_tokens_*
    #

    if schema.startswith("__[SCHEMA_"):
        return None, schema

    push = []
    symbol = ""
    bracket = False
    has_brackets = False
    for token in re.split(r"(\[|\]|\.)", schema):
        if not token:
            continue
        if token == "[":
            bracket = True
            has_brackets = True
        elif token == "]":
            bracket = False
        elif not bracket and token == ".":
            if has_brackets:
                push.append("[%s]" % symbol)
            else:
                push.append(symbol)
            symbol = ""
            has_brackets = False
        else:
            symbol += token
    if symbol:
        push.append(symbol)
    if len(push) > 1:
        dbname, owner = ".".join(push[0:-1]), push[-1]

        # test for internal brackets
        if re.match(r".*\].*\[.*", dbname[1:-1]):
            dbname = quoted_name(dbname, quote=False)
        else:
            dbname = dbname.lstrip("[").rstrip("]")

    elif len(push):
        dbname, owner = None, push[0]
    else:
        dbname, owner = None, None

    _memoized_schema[schema] = dbname, owner
    return dbname, owner


class MSDialect(default.DefaultDialect):
    # will assume it's at least mssql2005
    name = "mssql"
    supports_statement_cache = True
    supports_default_values = True
    supports_empty_insert = False
    favor_returning_over_lastrowid = True

    returns_native_bytes = True

    supports_comments = True
    supports_default_metavalue = False
    """dialect supports INSERT... VALUES (DEFAULT) syntax -
    SQL Server **does** support this, but **not** for the IDENTITY column,
    so we can't turn this on.

    """

    # supports_native_uuid is partial here, so we implement our
    # own impl type

    execution_ctx_cls = MSExecutionContext
    use_scope_identity = True
    max_identifier_length = 128
    schema_name = "dbo"

    insert_returning = True
    update_returning = True
    delete_returning = True
    update_returning_multifrom = True
    delete_returning_multifrom = True

    colspecs = {
        sqltypes.DateTime: _MSDateTime,
        sqltypes.Date: _MSDate,
        sqltypes.JSON: JSON,
        sqltypes.JSON.JSONIndexType: JSONIndexType,
        sqltypes.JSON.JSONPathType: JSONPathType,
        sqltypes.Time: _BASETIMEIMPL,
        sqltypes.Unicode: _MSUnicode,
        sqltypes.UnicodeText: _MSUnicodeText,
        DATETIMEOFFSET: DATETIMEOFFSET,
        DATETIME2: DATETIME2,
        SMALLDATETIME: SMALLDATETIME,
        DATETIME: DATETIME,
        sqltypes.Uuid: MSUUid,
    }

    engine_config_types = default.DefaultDialect.engine_config_types.union(
        {"legacy_schema_aliasing": util.asbool}
    )

    ischema_names = ischema_names

    supports_sequences = True
    sequences_optional = True
    # This is actually used for autoincrement, where itentity is used that
    # starts with 1.
    # for sequences T-SQL's actual default is -9223372036854775808
    default_sequence_base = 1

    supports_native_boolean = False
    non_native_boolean_check_constraint = False
    supports_unicode_binds = True
    postfetch_lastrowid = True

    # may be changed at server inspection time for older SQL server versions
    supports_multivalues_insert = True

    use_insertmanyvalues = True

    # note pyodbc will set this to False if fast_executemany is set,
    # as of SQLAlchemy 2.0.9
    use_insertmanyvalues_wo_returning = True

    insertmanyvalues_implicit_sentinel = (
        InsertmanyvaluesSentinelOpts.AUTOINCREMENT
        | InsertmanyvaluesSentinelOpts.IDENTITY
        | InsertmanyvaluesSentinelOpts.USE_INSERT_FROM_SELECT
    )

    # "The incoming request has too many parameters. The server supports a "
    # "maximum of 2100 parameters."
    # in fact you can have 2099 parameters.
    insertmanyvalues_max_parameters = 2099

    _supports_offset_fetch = False
    _supports_nvarchar_max = False

    legacy_schema_aliasing = False

    server_version_info = ()

    statement_compiler = MSSQLCompiler
    ddl_compiler = MSDDLCompiler
    type_compiler_cls = MSTypeCompiler
    preparer = MSIdentifierPreparer

    construct_arguments = [
        (sa_schema.PrimaryKeyConstraint, {"clustered": None}),
        (sa_schema.UniqueConstraint, {"clustered": None}),
        (
            sa_schema.Index,
            {
                "clustered": None,
                "include": None,
                "where": None,
                "columnstore": None,
            },
        ),
        (
            sa_schema.Column,
            {"identity_start": None, "identity_increment": None},
        ),
    ]

    def __init__(
        self,
        query_timeout=None,
        use_scope_identity=True,
        schema_name="dbo",
        deprecate_large_types=None,
        supports_comments=None,
        json_serializer=None,
        json_deserializer=None,
        legacy_schema_aliasing=None,
        ignore_no_transaction_on_rollback=False,
        **opts,
    ):
        self.query_timeout = int(query_timeout or 0)
        self.schema_name = schema_name

        self.use_scope_identity = use_scope_identity
        self.deprecate_large_types = deprecate_large_types
        self.ignore_no_transaction_on_rollback = (
            ignore_no_transaction_on_rollback
        )
        self._user_defined_supports_comments = uds = supports_comments
        if uds is not None:
            self.supports_comments = uds

        if legacy_schema_aliasing is not None:
            util.warn_deprecated(
                "The legacy_schema_aliasing parameter is "
                "deprecated and will be removed in a future release.",
                "1.4",
            )
            self.legacy_schema_aliasing = legacy_schema_aliasing

        super().__init__(**opts)

        self._json_serializer = json_serializer
        self._json_deserializer = json_deserializer

    def do_savepoint(self, connection, name):
        # give the DBAPI a push
        connection.exec_driver_sql("IF @@TRANCOUNT = 0 BEGIN TRANSACTION")
        super().do_savepoint(connection, name)

    def do_release_savepoint(self, connection, name):
        # SQL Server does not support RELEASE SAVEPOINT
        pass

    def do_rollback(self, dbapi_connection):
        try:
            super().do_rollback(dbapi_connection)
        except self.dbapi.ProgrammingError as e:
            if self.ignore_no_transaction_on_rollback and re.match(
                r".*\b111214\b", str(e)
            ):
                util.warn(
                    "ProgrammingError 111214 "
                    "'No corresponding transaction found.' "
                    "has been suppressed via "
                    "ignore_no_transaction_on_rollback=True"
                )
            else:
                raise

    _isolation_lookup = {
        "SERIALIZABLE",
        "READ UNCOMMITTED",
        "READ COMMITTED",
        "REPEATABLE READ",
        "SNAPSHOT",
    }

    def get_isolation_level_values(self, dbapi_connection):
        return list(self._isolation_lookup)

    def set_isolation_level(self, dbapi_connection, level):
        cursor = dbapi_connection.cursor()
        cursor.execute(f"SET TRANSACTION ISOLATION LEVEL {level}")
        cursor.close()
        if level == "SNAPSHOT":
            dbapi_connection.commit()

    def get_isolation_level(self, dbapi_connection):
        cursor = dbapi_connection.cursor()
        view_name = "sys.system_views"
        try:
            cursor.execute(
                (
                    "SELECT name FROM {} WHERE name IN "
                    "('dm_exec_sessions', 'dm_pdw_nodes_exec_sessions')"
                ).format(view_name)
            )
            row = cursor.fetchone()
            if not row:
                raise NotImplementedError(
                    "Can't fetch isolation level on this particular "
                    "SQL Server version."
                )

            view_name = f"sys.{row[0]}"

            cursor.execute(
                """
                    SELECT CASE transaction_isolation_level
                    WHEN 0 THEN NULL
                    WHEN 1 THEN 'READ UNCOMMITTED'
                    WHEN 2 THEN 'READ COMMITTED'
                    WHEN 3 THEN 'REPEATABLE READ'
                    WHEN 4 THEN 'SERIALIZABLE'
                    WHEN 5 THEN 'SNAPSHOT' END
                    AS TRANSACTION_ISOLATION_LEVEL
                    FROM {}
                    where session_id = @@SPID
                """.format(
                    view_name
                )
            )
        except self.dbapi.Error as err:
            raise NotImplementedError(
                "Can't fetch isolation level;  encountered error {} when "
                'attempting to query the "{}" view.'.format(err, view_name)
            ) from err
        else:
            row = cursor.fetchone()
            return row[0].upper()
        finally:
            cursor.close()

    def initialize(self, connection):
        super().initialize(connection)
        self._setup_version_attributes()
        self._setup_supports_nvarchar_max(connection)
        self._setup_supports_comments(connection)

    def _setup_version_attributes(self):
        if self.server_version_info[0] not in list(range(8, 17)):
            util.warn(
                "Unrecognized server version info '%s'.  Some SQL Server "
                "features may not function properly."
                % ".".join(str(x) for x in self.server_version_info)
            )

        if self.server_version_info >= MS_2008_VERSION:
            self.supports_multivalues_insert = True
        else:
            self.supports_multivalues_insert = False

        if self.deprecate_large_types is None:
            self.deprecate_large_types = (
                self.server_version_info >= MS_2012_VERSION
            )

        self._supports_offset_fetch = (
            self.server_version_info and self.server_version_info[0] >= 11
        )

    def _setup_supports_nvarchar_max(self, connection):
        try:
            connection.scalar(
                sql.text("SELECT CAST('test max support' AS NVARCHAR(max))")
            )
        except exc.DBAPIError:
            self._supports_nvarchar_max = False
        else:
            self._supports_nvarchar_max = True

    def _setup_supports_comments(self, connection):
        if self._user_defined_supports_comments is not None:
            return

        try:
            connection.scalar(
                sql.text(
                    "SELECT 1 FROM fn_listextendedproperty"
                    "(default, default, default, default, "
                    "default, default, default)"
                )
            )
        except exc.DBAPIError:
            self.supports_comments = False
        else:
            self.supports_comments = True

    def _get_default_schema_name(self, connection):
        query = sql.text("SELECT schema_name()")
        default_schema_name = connection.scalar(query)
        if default_schema_name is not None:
            # guard against the case where the default_schema_name is being
            # fed back into a table reflection function.
            return quoted_name(default_schema_name, quote=True)
        else:
            return self.schema_name

    @_db_plus_owner
    def has_table(self, connection, tablename, dbname, owner, schema, **kw):
        self._ensure_has_table_connection(connection)

        return self._internal_has_table(connection, tablename, owner, **kw)

    @reflection.cache
    @_db_plus_owner
    def has_sequence(
        self, connection, sequencename, dbname, owner, schema, **kw
    ):
        sequences = ischema.sequences

        s = sql.select(sequences.c.sequence_name).where(
            sequences.c.sequence_name == sequencename
        )

        if owner:
            s = s.where(sequences.c.sequence_schema == owner)

        c = connection.execute(s)

        return c.first() is not None

    @reflection.cache
    @_db_plus_owner_listing
    def get_sequence_names(self, connection, dbname, owner, schema, **kw):
        sequences = ischema.sequences

        s = sql.select(sequences.c.sequence_name)
        if owner:
            s = s.where(sequences.c.sequence_schema == owner)

        c = connection.execute(s)

        return [row[0] for row in c]

    @reflection.cache
    def get_schema_names(self, connection, **kw):
        s = sql.select(ischema.schemata.c.schema_name).order_by(
            ischema.schemata.c.schema_name
        )
        schema_names = [r[0] for r in connection.execute(s)]
        return schema_names

    @reflection.cache
    @_db_plus_owner_listing
    def get_table_names(self, connection, dbname, owner, schema, **kw):
        tables = ischema.tables
        s = (
            sql.select(tables.c.table_name)
            .where(
                sql.and_(
                    tables.c.table_schema == owner,
                    tables.c.table_type == "BASE TABLE",
                )
            )
            .order_by(tables.c.table_name)
        )
        table_names = [r[0] for r in connection.execute(s)]
        return table_names

    @reflection.cache
    @_db_plus_owner_listing
    def get_view_names(self, connection, dbname, owner, schema, **kw):
        tables = ischema.tables
        s = (
            sql.select(tables.c.table_name)
            .where(
                sql.and_(
                    tables.c.table_schema == owner,
                    tables.c.table_type == "VIEW",
                )
            )
            .order_by(tables.c.table_name)
        )
        view_names = [r[0] for r in connection.execute(s)]
        return view_names

    @reflection.cache
    def _internal_has_table(self, connection, tablename, owner, **kw):
        if tablename.startswith("#"):  # temporary table
            # mssql does not support temporary views
            # SQL Error [4103] [S0001]: "#v": Temporary views are not allowed
            return bool(
                connection.scalar(
                    # U filters on user tables only.
                    text("SELECT object_id(:table_name, 'U')"),
                    {"table_name": f"tempdb.dbo.[{tablename}]"},
                )
            )
        else:
            tables = ischema.tables

            s = sql.select(tables.c.table_name).where(
                sql.and_(
                    sql.or_(
                        tables.c.table_type == "BASE TABLE",
                        tables.c.table_type == "VIEW",
                    ),
                    tables.c.table_name == tablename,
                )
            )

            if owner:
                s = s.where(tables.c.table_schema == owner)

            c = connection.execute(s)

            return c.first() is not None

    def _default_or_error(self, connection, tablename, owner, method, **kw):
        # TODO: try to avoid having to run a separate query here
        if self._internal_has_table(connection, tablename, owner, **kw):
            return method()
        else:
            raise exc.NoSuchTableError(f"{owner}.{tablename}")

    @reflection.cache
    @_db_plus_owner
    def get_indexes(self, connection, tablename, dbname, owner, schema, **kw):
        filter_definition = (
            "ind.filter_definition"
            if self.server_version_info >= MS_2008_VERSION
            else "NULL as filter_definition"
        )
        rp = connection.execution_options(future_result=True).execute(
            sql.text(
                f"""
select
    ind.index_id,
    ind.is_unique,
    ind.name,
    ind.type,
    {filter_definition}
from
    sys.indexes as ind
join sys.tables as tab on
    ind.object_id = tab.object_id
join sys.schemas as sch on
    sch.schema_id = tab.schema_id
where
    tab.name = :tabname
    and sch.name = :schname
    and ind.is_primary_key = 0
    and ind.type != 0
order by
    ind.name
                """
            )
            .bindparams(
                sql.bindparam("tabname", tablename, ischema.CoerceUnicode()),
                sql.bindparam("schname", owner, ischema.CoerceUnicode()),
            )
            .columns(name=sqltypes.Unicode())
        )
        indexes = {}
        for row in rp.mappings():
            indexes[row["index_id"]] = current = {
                "name": row["name"],
                "unique": row["is_unique"] == 1,
                "column_names": [],
                "include_columns": [],
                "dialect_options": {},
            }

            do = current["dialect_options"]
            index_type = row["type"]
            if index_type in {1, 2}:
                do["mssql_clustered"] = index_type == 1
            if index_type in {5, 6}:
                do["mssql_clustered"] = index_type == 5
                do["mssql_columnstore"] = True
            if row["filter_definition"] is not None:
                do["mssql_where"] = row["filter_definition"]

        rp = connection.execution_options(future_result=True).execute(
            sql.text(
                """
select
    ind_col.index_id,
    col.name,
    ind_col.is_included_column
from
    sys.columns as col
join sys.tables as tab on
    tab.object_id = col.object_id
join sys.index_columns as ind_col on
    ind_col.column_id = col.column_id
    and ind_col.object_id = tab.object_id
join sys.schemas as sch on
    sch.schema_id = tab.schema_id
where
    tab.name = :tabname
    and sch.name = :schname
            """
            )
            .bindparams(
                sql.bindparam("tabname", tablename, ischema.CoerceUnicode()),
                sql.bindparam("schname", owner, ischema.CoerceUnicode()),
            )
            .columns(name=sqltypes.Unicode())
        )
        for row in rp.mappings():
            if row["index_id"] not in indexes:
                continue
            index_def = indexes[row["index_id"]]
            is_colstore = index_def["dialect_options"].get("mssql_columnstore")
            is_clustered = index_def["dialect_options"].get("mssql_clustered")
            if not (is_colstore and is_clustered):
                # a clustered columnstore index includes all columns but does
                # not want them in the index definition
                if row["is_included_column"] and not is_colstore:
                    # a noncludsted columnstore index reports that includes
                    # columns but requires that are listed as normal columns
                    index_def["include_columns"].append(row["name"])
                else:
                    index_def["column_names"].append(row["name"])
        for index_info in indexes.values():
            # NOTE: "root level" include_columns is legacy, now part of
            #       dialect_options (issue #7382)
            index_info["dialect_options"]["mssql_include"] = index_info[
                "include_columns"
            ]

        if indexes:
            return list(indexes.values())
        else:
            return self._default_or_error(
                connection, tablename, owner, ReflectionDefaults.indexes, **kw
            )

    @reflection.cache
    @_db_plus_owner
    def get_view_definition(
        self, connection, viewname, dbname, owner, schema, **kw
    ):
        view_def = connection.execute(
            sql.text(
                "select mod.definition "
                "from sys.sql_modules as mod "
                "join sys.views as views on mod.object_id = views.object_id "
                "join sys.schemas as sch on views.schema_id = sch.schema_id "
                "where views.name=:viewname and sch.name=:schname"
            ).bindparams(
                sql.bindparam("viewname", viewname, ischema.CoerceUnicode()),
                sql.bindparam("schname", owner, ischema.CoerceUnicode()),
            )
        ).scalar()
        if view_def:
            return view_def
        else:
            raise exc.NoSuchTableError(f"{owner}.{viewname}")

    @reflection.cache
    def get_table_comment(self, connection, table_name, schema=None, **kw):
        if not self.supports_comments:
            raise NotImplementedError(
                "Can't get table comments on current SQL Server version in use"
            )

        schema_name = schema if schema else self.default_schema_name
        COMMENT_SQL = """
            SELECT cast(com.value as nvarchar(max))
            FROM fn_listextendedproperty('MS_Description',
                'schema', :schema, 'table', :table, NULL, NULL
            ) as com;
        """

        comment = connection.execute(
            sql.text(COMMENT_SQL).bindparams(
                sql.bindparam("schema", schema_name, ischema.CoerceUnicode()),
                sql.bindparam("table", table_name, ischema.CoerceUnicode()),
            )
        ).scalar()
        if comment:
            return {"text": comment}
        else:
            return self._default_or_error(
                connection,
                table_name,
                None,
                ReflectionDefaults.table_comment,
                **kw,
            )

    def _temp_table_name_like_pattern(self, tablename):
        # LIKE uses '%' to match zero or more characters and '_' to match any
        # single character. We want to match literal underscores, so T-SQL
        # requires that we enclose them in square brackets.
        return tablename + (
            ("[_][_][_]%") if not tablename.startswith("##") else ""
        )

    def _get_internal_temp_table_name(self, connection, tablename):
        # it's likely that schema is always "dbo", but since we can
        # get it here, let's get it.
        # see https://stackoverflow.com/questions/8311959/
        # specifying-schema-for-temporary-tables

        try:
            return connection.execute(
                sql.text(
                    "select table_schema, table_name "
                    "from tempdb.information_schema.tables "
                    "where table_name like :p1"
                ),
                {"p1": self._temp_table_name_like_pattern(tablename)},
            ).one()
        except exc.MultipleResultsFound as me:
            raise exc.UnreflectableTableError(
                "Found more than one temporary table named '%s' in tempdb "
                "at this time. Cannot reliably resolve that name to its "
                "internal table name." % tablename
            ) from me
        except exc.NoResultFound as ne:
            raise exc.NoSuchTableError(
                "Unable to find a temporary table named '%s' in tempdb."
                % tablename
            ) from ne

    @reflection.cache
    @_db_plus_owner
    def get_columns(self, connection, tablename, dbname, owner, schema, **kw):
        is_temp_table = tablename.startswith("#")
        if is_temp_table:
            owner, tablename = self._get_internal_temp_table_name(
                connection, tablename
            )

            columns = ischema.mssql_temp_table_columns
        else:
            columns = ischema.columns

        computed_cols = ischema.computed_columns
        identity_cols = ischema.identity_columns
        if owner:
            whereclause = sql.and_(
                columns.c.table_name == tablename,
                columns.c.table_schema == owner,
            )
            full_name = columns.c.table_schema + "." + columns.c.table_name
        else:
            whereclause = columns.c.table_name == tablename
            full_name = columns.c.table_name

        if self._supports_nvarchar_max:
            computed_definition = computed_cols.c.definition
        else:
            # tds_version 4.2 does not support NVARCHAR(MAX)
            computed_definition = sql.cast(
                computed_cols.c.definition, NVARCHAR(4000)
            )

        object_id = func.object_id(full_name)

        s = (
            sql.select(
                columns.c.column_name,
                columns.c.data_type,
                columns.c.is_nullable,
                columns.c.character_maximum_length,
                columns.c.numeric_precision,
                columns.c.numeric_scale,
                columns.c.column_default,
                columns.c.collation_name,
                computed_definition,
                computed_cols.c.is_persisted,
                identity_cols.c.is_identity,
                identity_cols.c.seed_value,
                identity_cols.c.increment_value,
                ischema.extended_properties.c.value.label("comment"),
            )
            .select_from(columns)
            .outerjoin(
                computed_cols,
                onclause=sql.and_(
                    computed_cols.c.object_id == object_id,
                    computed_cols.c.name
                    == columns.c.column_name.collate("DATABASE_DEFAULT"),
                ),
            )
            .outerjoin(
                identity_cols,
                onclause=sql.and_(
                    identity_cols.c.object_id == object_id,
                    identity_cols.c.name
                    == columns.c.column_name.collate("DATABASE_DEFAULT"),
                ),
            )
            .outerjoin(
                ischema.extended_properties,
                onclause=sql.and_(
                    ischema.extended_properties.c["class"] == 1,
                    ischema.extended_properties.c.major_id == object_id,
                    ischema.extended_properties.c.minor_id
                    == columns.c.ordinal_position,
                    ischema.extended_properties.c.name == "MS_Description",
                ),
            )
            .where(whereclause)
            .order_by(columns.c.ordinal_position)
        )

        c = connection.execution_options(future_result=True).execute(s)

        cols = []
        for row in c.mappings():
            name = row[columns.c.column_name]
            type_ = row[columns.c.data_type]
            nullable = row[columns.c.is_nullable] == "YES"
            charlen = row[columns.c.character_maximum_length]
            numericprec = row[columns.c.numeric_precision]
            numericscale = row[columns.c.numeric_scale]
            default = row[columns.c.column_default]
            collation = row[columns.c.collation_name]
            definition = row[computed_definition]
            is_persisted = row[computed_cols.c.is_persisted]
            is_identity = row[identity_cols.c.is_identity]
            identity_start = row[identity_cols.c.seed_value]
            identity_increment = row[identity_cols.c.increment_value]
            comment = row[ischema.extended_properties.c.value]

            coltype = self.ischema_names.get(type_, None)

            kwargs = {}
            if coltype in (
                MSString,
                MSChar,
                MSNVarchar,
                MSNChar,
                MSText,
                MSNText,
                MSBinary,
                MSVarBinary,
                sqltypes.LargeBinary,
            ):
                if charlen == -1:
                    charlen = None
                kwargs["length"] = charlen
                if collation:
                    kwargs["collation"] = collation

            if coltype is None:
                util.warn(
                    "Did not recognize type '%s' of column '%s'"
                    % (type_, name)
                )
                coltype = sqltypes.NULLTYPE
            else:
                if issubclass(coltype, sqltypes.Numeric):
                    kwargs["precision"] = numericprec

                    if not issubclass(coltype, sqltypes.Float):
                        kwargs["scale"] = numericscale

                coltype = coltype(**kwargs)
            cdict = {
                "name": name,
                "type": coltype,
                "nullable": nullable,
                "default": default,
                "autoincrement": is_identity is not None,
                "comment": comment,
            }

            if definition is not None and is_persisted is not None:
                cdict["computed"] = {
                    "sqltext": definition,
                    "persisted": is_persisted,
                }

            if is_identity is not None:
                # identity_start and identity_increment are Decimal or None
                if identity_start is None or identity_increment is None:
                    cdict["identity"] = {}
                else:
                    if isinstance(coltype, sqltypes.BigInteger):
                        start = int(identity_start)
                        increment = int(identity_increment)
                    elif isinstance(coltype, sqltypes.Integer):
                        start = int(identity_start)
                        increment = int(identity_increment)
                    else:
                        start = identity_start
                        increment = identity_increment

                    cdict["identity"] = {
                        "start": start,
                        "increment": increment,
                    }

            cols.append(cdict)

        if cols:
            return cols
        else:
            return self._default_or_error(
                connection, tablename, owner, ReflectionDefaults.columns, **kw
            )

    @reflection.cache
    @_db_plus_owner
    def get_pk_constraint(
        self, connection, tablename, dbname, owner, schema, **kw
    ):
        pkeys = []
        TC = ischema.constraints
        C = ischema.key_constraints.alias("C")

        # Primary key constraints
        s = (
            sql.select(
                C.c.column_name,
                TC.c.constraint_type,
                C.c.constraint_name,
                func.objectproperty(
                    func.object_id(
                        C.c.table_schema + "." + C.c.constraint_name
                    ),
                    "CnstIsClustKey",
                ).label("is_clustered"),
            )
            .where(
                sql.and_(
                    TC.c.constraint_name == C.c.constraint_name,
                    TC.c.table_schema == C.c.table_schema,
                    C.c.table_name == tablename,
                    C.c.table_schema == owner,
                ),
            )
            .order_by(TC.c.constraint_name, C.c.ordinal_position)
        )
        c = connection.execution_options(future_result=True).execute(s)
        constraint_name = None
        is_clustered = None
        for row in c.mappings():
            if "PRIMARY" in row[TC.c.constraint_type.name]:
                pkeys.append(row["COLUMN_NAME"])
                if constraint_name is None:
                    constraint_name = row[C.c.constraint_name.name]
                if is_clustered is None:
                    is_clustered = row["is_clustered"]
        if pkeys:
            return {
                "constrained_columns": pkeys,
                "name": constraint_name,
                "dialect_options": {"mssql_clustered": is_clustered},
            }
        else:
            return self._default_or_error(
                connection,
                tablename,
                owner,
                ReflectionDefaults.pk_constraint,
                **kw,
            )

    @reflection.cache
    @_db_plus_owner
    def get_foreign_keys(
        self, connection, tablename, dbname, owner, schema, **kw
    ):
        # Foreign key constraints
        s = (
            text(
                """\
WITH fk_info AS (
    SELECT
        ischema_ref_con.constraint_schema,
        ischema_ref_con.constraint_name,
        ischema_key_col.ordinal_position,
        ischema_key_col.table_schema,
        ischema_key_col.table_name,
        ischema_ref_con.unique_constraint_schema,
        ischema_ref_con.unique_constraint_name,
        ischema_ref_con.match_option,
        ischema_ref_con.update_rule,
        ischema_ref_con.delete_rule,
        ischema_key_col.column_name AS constrained_column
    FROM
        INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS ischema_ref_con
        INNER JOIN
        INFORMATION_SCHEMA.KEY_COLUMN_USAGE ischema_key_col ON
            ischema_key_col.table_schema = ischema_ref_con.constraint_schema
            AND ischema_key_col.constraint_name =
            ischema_ref_con.constraint_name
    WHERE ischema_key_col.table_name = :tablename
        AND ischema_key_col.table_schema = :owner
),
constraint_info AS (
    SELECT
        ischema_key_col.constraint_schema,
        ischema_key_col.constraint_name,
        ischema_key_col.ordinal_position,
        ischema_key_col.table_schema,
        ischema_key_col.table_name,
        ischema_key_col.column_name
    FROM
        INFORMATION_SCHEMA.KEY_COLUMN_USAGE ischema_key_col
),
index_info AS (
    SELECT
        sys.schemas.name AS index_schema,
        sys.indexes.name AS index_name,
        sys.index_columns.key_ordinal AS ordinal_position,
        sys.schemas.name AS table_schema,
        sys.objects.name AS table_name,
        sys.columns.name AS column_name
    FROM
        sys.indexes
        INNER JOIN
        sys.objects ON
            sys.objects.object_id = sys.indexes.object_id
        INNER JOIN
        sys.schemas ON
            sys.schemas.schema_id = sys.objects.schema_id
        INNER JOIN
        sys.index_columns ON
            sys.index_columns.object_id = sys.objects.object_id
            AND sys.index_columns.index_id = sys.indexes.index_id
        INNER JOIN
        sys.columns ON
            sys.columns.object_id = sys.indexes.object_id
            AND sys.columns.column_id = sys.index_columns.column_id
)
    SELECT
        fk_info.constraint_schema,
        fk_info.constraint_name,
        fk_info.ordinal_position,
        fk_info.constrained_column,
        constraint_info.table_schema AS referred_table_schema,
        constraint_info.table_name AS referred_table_name,
        constraint_info.column_name AS referred_column,
        fk_info.match_option,
        fk_info.update_rule,
        fk_info.delete_rule
    FROM
        fk_info INNER JOIN constraint_info ON
            constraint_info.constraint_schema =
                fk_info.unique_constraint_schema
            AND constraint_info.constraint_name =
                fk_info.unique_constraint_name
            AND constraint_info.ordinal_position = fk_info.ordinal_position
    UNION
    SELECT
        fk_info.constraint_schema,
        fk_info.constraint_name,
        fk_info.ordinal_position,
        fk_info.constrained_column,
        index_info.table_schema AS referred_table_schema,
        index_info.table_name AS referred_table_name,
        index_info.column_name AS referred_column,
        fk_info.match_option,
        fk_info.update_rule,
        fk_info.delete_rule
    FROM
        fk_info INNER JOIN index_info ON
            index_info.index_schema = fk_info.unique_constraint_schema
            AND index_info.index_name = fk_info.unique_constraint_name
            AND index_info.ordinal_position = fk_info.ordinal_position

    ORDER BY fk_info.constraint_schema, fk_info.constraint_name,
        fk_info.ordinal_position
"""
            )
            .bindparams(
                sql.bindparam("tablename", tablename, ischema.CoerceUnicode()),
                sql.bindparam("owner", owner, ischema.CoerceUnicode()),
            )
            .columns(
                constraint_schema=sqltypes.Unicode(),
                constraint_name=sqltypes.Unicode(),
                table_schema=sqltypes.Unicode(),
                table_name=sqltypes.Unicode(),
                constrained_column=sqltypes.Unicode(),
                referred_table_schema=sqltypes.Unicode(),
                referred_table_name=sqltypes.Unicode(),
                referred_column=sqltypes.Unicode(),
            )
        )

        # group rows by constraint ID, to handle multi-column FKs
        fkeys = util.defaultdict(
            lambda: {
                "name": None,
                "constrained_columns": [],
                "referred_schema": None,
                "referred_table": None,
                "referred_columns": [],
                "options": {},
            }
        )

        for r in connection.execute(s).all():
            (
                _,  # constraint schema
                rfknm,
                _,  # ordinal position
                scol,
                rschema,
                rtbl,
                rcol,
                # TODO: we support match=<keyword> for foreign keys so
                # we can support this also, PG has match=FULL for example
                # but this seems to not be a valid value for SQL Server
                _,  # match rule
                fkuprule,
                fkdelrule,
            ) = r

            rec = fkeys[rfknm]
            rec["name"] = rfknm

            if fkuprule != "NO ACTION":
                rec["options"]["onupdate"] = fkuprule

            if fkdelrule != "NO ACTION":
                rec["options"]["ondelete"] = fkdelrule

            if not rec["referred_table"]:
                rec["referred_table"] = rtbl
                if schema is not None or owner != rschema:
                    if dbname:
                        rschema = dbname + "." + rschema
                    rec["referred_schema"] = rschema

            local_cols, remote_cols = (
                rec["constrained_columns"],
                rec["referred_columns"],
            )

            local_cols.append(scol)
            remote_cols.append(rcol)

        if fkeys:
            return list(fkeys.values())
        else:
            return self._default_or_error(
                connection,
                tablename,
                owner,
                ReflectionDefaults.foreign_keys,
                **kw,
            )